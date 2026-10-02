"""Bind ArrayExpress platform features to the frozen therapeutic objective.

This stage is deliberately separate from the historical structure materialization
builder.  It consumes the already materialized E-GEOD source rows, records an
authoritative feature mapping, and emits a new semantic evidence-domain version.
No mapping choice is made from expression values, outcomes, or model output.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_therapeutic_evidence_expansion import (  # noqa: E402
    BUILDER_PATH as STRUCTURE_BUILDER_PATH,
    CANDIDATE_PATH as STRUCTURE_CANDIDATE_PATH,
    OBSERVATION_PATH as STRUCTURE_OBSERVATION_PATH,
    PRIORITY_AUDIT_PATH,
    PRIORITY_TABLE_PATH,
    SOURCE_ROOT,
    _artifact,
    _uid,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402


TARGETED_OBJECTIVE_CONFIG = REPO_ROOT / "Libs/configs/data/targeted_therapeutic_evidence.yaml"
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
PLATFORM_ROOT = REPO_ROOT / "Input/raw/arrayexpress/platform_annotations"
PLATFORM_MANIFEST_PATH = REPO_ROOT / "Input/manifests/therapeutic_outcomes/platform_annotations_v2.tsv"
OBJECTIVE_BINDING_PATH = REPO_ROOT / "Input/manifests/therapeutic_outcomes/objective_namespace_binding_v2.tsv"
PLATFORM_AUDIT_PATH = REPO_ROOT / "Output/audits/therapeutic_platform_annotation_binding.json"
OBJECTIVE_AUDIT_PATH = REPO_ROOT / "Output/audits/therapeutic_objective_namespace_binding.json"
OBSERVATION_V3_PATH = REPO_ROOT / "Input/manifests/therapeutic_outcomes/selected_source_observations_v3.tsv"
CANDIDATE_V3_PATH = REPO_ROOT / "Input/manifests/therapeutic_outcomes/selected_source_candidate_sets_v3.tsv"
QUALIFICATION_V3_PATH = REPO_ROOT / "Output/audits/therapeutic_candidate_qualification_v3.tsv"
MATERIALIZATION_V3_PATH = REPO_ROOT / "Output/audits/therapeutic_evidence_materialization_v3.json"
SPLIT_V3_PATH = REPO_ROOT / "Output/audits/therapeutic_benchmark_split_feasibility_v3.json"
DOMAIN_V3_PATH = REPO_ROOT / "Output/audits/therapeutic_decision_domain_v3.json"
GAP_MAP_PATH = REPO_ROOT / "Output/audits/therapeutic_evidence_gap_map_v3.json"
ACQUISITION_QUEUE_PATH = REPO_ROOT / "Output/audits/therapeutic_targeted_acquisition_queue_v3.json"
PLATFORM_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/therapeutic_platform_annotation_binding.schema.json"
OBJECTIVE_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/therapeutic_objective_namespace_binding.schema.json"
DOMAIN_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/therapeutic_semantic_domain.schema.json"

OBJECTIVE_UID = "ad_transcriptomic_therapeutic_benefit_source_context_v1"
OBJECTIVE_NAMESPACE = "species_scoped_ncbi_gene_or_symbol"
AGGREGATION_RULE = "median_over_unique_objective_member_features_prespecified_before_outcomes"
BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tsv_artifact(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> dict[str, object]:
    """Return the deterministic artifact identity produced by ``write_tsv``."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    encoded = buffer.getvalue().encode("utf-8")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _artifact_with_source(path: Path, source_url: str) -> dict[str, object]:
    artifact = _artifact(path)
    artifact["source_url"] = source_url
    artifact["content_sha256"] = artifact["sha256"]
    return artifact


def _objective() -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["targeted_therapeutic_expansion"]
    objective = policy["therapeutic_decision_objective"]
    if objective.get("objective_uid") != OBJECTIVE_UID:
        raise ValueError("The semantic binding objective UID drifted")
    if objective.get("freeze_status") != "frozen_before_outcome_materialization":
        raise ValueError("The therapeutic objective is not frozen before outcome materialization")
    components = [dict(objective["primary_goal"]), *[dict(item) for item in objective["secondary_goals"]]]
    members = {
        str(symbol).strip()
        for component in components
        for symbol in component["gene_symbols"]
    }
    if len(members) != sum(len(component["gene_symbols"]) for component in components):
        raise ValueError("The frozen objective contains duplicate members")
    return dict(objective), components, members


def _platform_for_source(source_accession: str) -> tuple[str, Path, str, str]:
    platforms = {
        "E-GEOD-55369": (
            "A-GEOD-14797",
            PLATFORM_ROOT / "A-GEOD-14797" / "A-GEOD-14797.adf.txt",
            "https://www.ebi.ac.uk/biostudies/files/A-GEOD-14797/A-GEOD-14797.adf.txt",
            "Rattus norvegicus",
        ),
        "E-GEOD-41345": (
            "A-MEXP-1174",
            PLATFORM_ROOT / "A-MEXP-1174" / "A-MEXP-1174.adf.txt",
            "https://www.ebi.ac.uk/biostudies/files/A-MEXP-1174/A-MEXP-1174.adf.txt",
            "Mus musculus",
        ),
    }
    try:
        return platforms[source_accession]
    except KeyError as exc:
        raise ValueError(f"No frozen platform binding exists for {source_accession}") from exc


def _split_annotation(value: str) -> list[str]:
    value = value.strip()
    if not value or value in {"---", "NA", "N/A", "null", "None"}:
        return []
    # ADF files use a number of delimiters for one-to-many annotations.  A
    # delimiter is never resolved by choosing the first token.
    tokens = [token.strip() for token in re.split(r"\s*(?:///|\|;|;|\|)\s*", value)]
    return [token for token in tokens if token and token not in {"---", "NA", "N/A"}]


def _header_and_rows(path: Path) -> tuple[dict[str, str], Iterable[dict[str, str]]]:
    """Return metadata and data rows from either current ADF layout."""

    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header_index = next((i for i, row in enumerate(rows) if "Reporter Name" in row), None)
    if header_index is None:
        raise ValueError(f"ADF lacks Reporter Name header: {path}")
    metadata: dict[str, str] = {}
    for row in rows[:header_index]:
        if len(row) >= 2 and row[0].strip() and row[1].strip():
            metadata.setdefault(row[0].strip(), row[1].strip())
    header = [value.strip() for value in rows[header_index]]
    if len(set(header)) != len(header):
        raise ValueError(f"ADF header contains duplicate fields: {path}")
    output: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not row or not any(str(value).strip() for value in row):
            continue
        if len(row) < len(header):
            row = [*row, *([""] * (len(header) - len(row)))]
        output.append(dict(zip(header, row, strict=False)))
    metadata["_header"] = "\t".join(header)
    metadata["_row_count"] = str(len(output))
    return metadata, output


def _field_names(header: str, *, rat: bool) -> tuple[str, str]:
    values = set(header.split("\t"))
    if rat:
        symbol, entity = "Comment[GENE_SYMBOL]", "Reporter Database Entry [locus]"
    else:
        symbol, entity = "Reporter Database Entry[hugo]", "Reporter Database Entry[entrez]"
    if "Reporter Name" not in values or symbol not in values or entity not in values:
        raise ValueError(f"Platform annotation lacks required identity fields: {symbol}, {entity}")
    return symbol, entity


def _canonical_entity(taxon: str, symbol: str, entity_tokens: list[str]) -> tuple[str, str, str]:
    numeric = [token for token in entity_tokens if token.isdigit()]
    non_numeric = [token for token in entity_tokens if not token.isdigit()]
    if len(set(numeric)) > 1 or non_numeric:
        return "", "", "contradictory_annotations"
    if numeric:
        return f"NCBIGene:{taxon}:{numeric[0]}", "NCBIGene", "bound_numeric"
    # The platform's curated species gene symbol is retained as a canonical
    # fallback only when the annotation omits an Entrez identifier.
    return f"NCBIGeneSymbol:{taxon}:{symbol.casefold()}", "NCBIGeneSymbol", "bound_symbol_fallback"


def bind_platform(source_accession: str, objective: Mapping[str, object], components: list[dict[str, object]], members: set[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    platform_accession, path, source_url, species = _platform_for_source(source_accession)
    if not path.is_file():
        raise FileNotFoundError(path)
    annotation_sha256 = _sha256(path)
    metadata, rows = _header_and_rows(path)
    rat = species == "Rattus norvegicus"
    symbol_field, entity_field = _field_names(metadata["_header"], rat=rat)
    member_lookup = {member.casefold(): member for member in members}
    feature_rows: list[dict[str, object]] = []
    member_features: dict[str, list[dict[str, object]]] = defaultdict(list)
    feature_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    for row in rows:
        feature_id = str(row.get("Reporter Name", "")).strip()
        if not feature_id:
            status_counts["unmapped_feature"] += 1
            continue
        if feature_id in feature_ids:
            raise ValueError(f"Duplicate reporter ID in {platform_accession}: {feature_id}")
        feature_ids.add(feature_id)
        symbol_tokens = _split_annotation(str(row.get(symbol_field, "")))
        entity_tokens = _split_annotation(str(row.get(entity_field, "")))
        status = "non_objective_feature"
        objective_member = ""
        canonical_entity_id = ""
        canonical_namespace = ""
        if len(symbol_tokens) > 1:
            status = "one_feature_to_multiple_entities"
        elif symbol_tokens and symbol_tokens[0].casefold() in member_lookup:
            objective_member = member_lookup[symbol_tokens[0].casefold()]
            canonical_entity_id, canonical_namespace, status = _canonical_entity(
                "10116" if rat else "10090", objective_member, entity_tokens
            )
            member_features[objective_member].append(
                {
                    "feature_id": feature_id,
                    "canonical_entity_id": canonical_entity_id,
                    "canonical_namespace": canonical_namespace,
                    "mapping_status": status,
                }
            )
        elif not symbol_tokens:
            status = "unmapped_feature"
        status_counts[status] += 1
        feature_rows.append(
            {
                "platform_accession": platform_accession,
                "source_accession": source_accession,
                "species": species,
                "feature_id": feature_id,
                "raw_symbol": str(row.get(symbol_field, "")).strip(),
                "raw_entity_id": str(row.get(entity_field, "")).strip(),
                "canonical_entity_id": canonical_entity_id,
                "canonical_namespace": canonical_namespace,
                "objective_member": objective_member,
                "mapping_status": status,
                "annotation_file_sha256": annotation_sha256,
                **BOUNDARY,
            }
        )

    binding_rows: list[dict[str, object]] = []
    missing_members: list[str] = []
    contradictory_members: list[str] = []
    duplicate_member_counts: dict[str, int] = {}
    component_by_member = {
        str(symbol): str(component["component_uid"])
        for component in components
        for symbol in component["gene_symbols"]
    }
    for member in sorted(members):
        features = member_features.get(member, [])
        ids = sorted({str(row["canonical_entity_id"]) for row in features if row["canonical_entity_id"]})
        statuses = {str(row["mapping_status"]) for row in features}
        contradictory = "contradictory_annotations" in statuses or len(ids) > 1
        if not features:
            mapping_status = "missing_on_platform"
            missing_members.append(member)
            canonical_entity_id = ""
            namespace = ""
        elif contradictory:
            mapping_status = "contradictory_annotations"
            contradictory_members.append(member)
            canonical_entity_id = ""
            namespace = ""
        else:
            mapping_status = "bound"
            canonical_entity_id = ids[0] if ids else ""
            namespace = str(features[0]["canonical_namespace"])
            duplicate_member_counts[member] = len(features)
        binding_rows.append(
            {
                "platform_accession": platform_accession,
                "source_accession": source_accession,
                "species": species,
                "objective_uid": OBJECTIVE_UID,
                "component_uid": component_by_member[member],
                "objective_member": member,
                "canonical_entity_id": canonical_entity_id,
                "canonical_namespace": namespace,
                "feature_count": len(features),
                "feature_ids": json.dumps(sorted(str(row["feature_id"]) for row in features), separators=(",", ":")),
                "mapping_status": mapping_status,
                "aggregation_rule": AGGREGATION_RULE,
                "objective_selection_outcome_blind": True,
                **BOUNDARY,
            }
        )
    coverage = len(members) - len(missing_members) - len(contradictory_members)
    source_status = "bound" if coverage == len(members) and not contradictory_members else "rejected_incomplete_objective_coverage"
    platform_row = {
        "platform_accession": platform_accession,
        "source_accession": source_accession,
        "species": species,
        "annotation_path": path.relative_to(REPO_ROOT).as_posix(),
        "annotation_source_url": source_url,
        "annotation_content_sha256": annotation_sha256,
        "annotation_identity_bound": True,
        "annotation_header_sha256": hashlib.sha256(metadata["_header"].encode("utf-8")).hexdigest(),
        "feature_count": len(feature_rows),
        "mapping_status_counts": dict(sorted(status_counts.items())),
        "objective_member_count": len(members),
        "objective_member_bound_count": coverage,
        "objective_coverage_fraction": coverage / len(members),
        "missing_objective_members": missing_members,
        "contradictory_objective_members": contradictory_members,
        "duplicate_probe_objective_members": sorted(duplicate_member_counts),
        "duplicate_probe_count": sum(max(count - 1, 0) for count in duplicate_member_counts.values()),
        "ambiguous_mapping_policy_frozen": True,
        "aggregation_rule": AGGREGATION_RULE,
        "outcome_driven_mapping_choices": False,
        "status": source_status,
        **BOUNDARY,
    }
    return feature_rows, binding_rows, platform_row


def _write_binding_artifacts(platform_rows: list[dict[str, object]], binding_rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    feature_fields = list(platform_rows[0])
    binding_fields = list(binding_rows[0])
    write_tsv(PLATFORM_MANIFEST_PATH, feature_fields, platform_rows)
    write_tsv(OBJECTIVE_BINDING_PATH, binding_fields, binding_rows)
    by_source: dict[str, dict[str, object]] = {}
    for row in platform_rows:
        by_source.setdefault(str(row["source_accession"]), {})
    # Platform summaries are already authoritative and are written by build().
    return _artifact(PLATFORM_MANIFEST_PATH), _artifact(OBJECTIVE_BINDING_PATH)


def _clone_semantic_rows(platform_status: Mapping[str, str]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    observations = read_tsv(STRUCTURE_OBSERVATION_PATH)
    candidates = read_tsv(STRUCTURE_CANDIDATE_PATH)
    output_observations: list[dict[str, object]] = []
    for row in observations:
        accession = str(row["source_accession"])
        status = platform_status[accession]
        enriched = dict(row)
        enriched["measurement_namespace"] = "platform_gene_objective" if status == "bound" else "platform_objective_binding_rejected"
        enriched["objective_namespace_status"] = "bound" if status == "bound" else status
        enriched["outcome"] = "numeric_objective_measurement_pending" if status == "bound" else "explicit_unavailable_incomplete_objective_coverage"
        enriched["unit"] = "source_processed_VALUE" if status != "bound" else "platform_objective_aggregate"
        enriched["uncertainty"] = "source_replicate_group_pending" if status == "bound" else "explicit_unavailable_incomplete_objective_coverage"
        enriched["candidate_set_status"] = "pending_numeric_qualification" if status == "bound" else "not_qualified_objective_namespace_incomplete"
        enriched["admission_status"] = "quarantined_not_admitted"
        enriched["admission_rejection_reason"] = "" if status == "bound" else "frozen_objective_not_fully_supported_on_platform"
        output_observations.append(enriched)
    output_candidates: list[dict[str, object]] = []
    for row in candidates:
        accession = str(row["source_accession"])
        status = platform_status[accession]
        enriched = dict(row)
        enriched["classification"] = "pending_numeric_qualification" if status == "bound" else "rejected_incomplete_objective_coverage"
        enriched["qualification_status"] = enriched["classification"]
        output_candidates.append(enriched)
    action_uids = {str(row["world_action_uid"]) for row in output_observations}
    therapeutic_action_uids = {
        str(row["world_action_uid"])
        for row in output_observations
        if row.get("action_is_therapeutic") == "True"
    }
    return output_observations, output_candidates, {
        "observation_count": len(output_observations),
        "action_count": len(action_uids),
        "therapeutic_action_count": len(therapeutic_action_uids),
    }


def build(*, write_output: bool = True) -> dict[str, object]:
    objective, components, members = _objective()
    selected_sources = [str(row["source_accession"]) for row in read_tsv(PRIORITY_TABLE_PATH) if str(row.get("selection_class")) == "selected_for_materialization"]
    if selected_sources != ["E-GEOD-55369", "E-GEOD-41345"]:
        raise ValueError(f"Current selected source order drifted: {selected_sources}")
    all_feature_rows: list[dict[str, object]] = []
    all_binding_rows: list[dict[str, object]] = []
    platform_summaries: list[dict[str, object]] = []
    platform_status: dict[str, str] = {}
    for accession in selected_sources:
        feature_rows, binding_rows, summary = bind_platform(accession, objective, components, members)
        all_feature_rows.extend(feature_rows)
        all_binding_rows.extend(binding_rows)
        platform_summaries.append(summary)
        platform_status[accession] = str(summary["status"])

    if write_output:
        _write_binding_artifacts(all_feature_rows, all_binding_rows)
    platform_manifest_artifact = _tsv_artifact(
        PLATFORM_MANIFEST_PATH,
        list(all_feature_rows[0]),
        all_feature_rows,
    )
    objective_binding_artifact = _tsv_artifact(
        OBJECTIVE_BINDING_PATH,
        list(all_binding_rows[0]),
        all_binding_rows,
    )
    source_binding = {
        str(row["source_accession"]): {
            "platform_accession": row["platform_accession"],
            "status": row["status"],
            "objective_coverage_fraction": row["objective_coverage_fraction"],
            "missing_objective_members": row["missing_objective_members"],
            "contradictory_objective_members": row["contradictory_objective_members"],
        }
        for row in platform_summaries
    }
    platform_audit = {
        "schema_version": "neuronsifter-therapeutic-platform-annotation-binding-v2",
        "created_at": _now(),
        "status": "bound" if all(row["status"] == "bound" for row in platform_summaries) else "partial_binding_with_rejected_sources",
        "selected_sources": selected_sources,
        "platform_summaries": platform_summaries,
        "annotation_identity_bound": all(bool(row["annotation_identity_bound"]) for row in platform_summaries),
        "ambiguous_mapping_policy_frozen": True,
        "aggregation_rule": AGGREGATION_RULE,
        "outcome_driven_mapping_choices": False,
        "source_bindings": source_binding,
        "protected_holdout_access": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "input_bindings": {
            "source_priority_audit": _artifact(PRIORITY_AUDIT_PATH),
            "source_priority_table": _artifact(PRIORITY_TABLE_PATH),
            "objective_config": _artifact(TARGETED_OBJECTIVE_CONFIG),
        },
        "output_bindings": {
            "platform_manifest": platform_manifest_artifact,
            "objective_binding": objective_binding_artifact,
        },
        **BOUNDARY,
    }
    objective_audit = {
        "schema_version": "neuronsifter-therapeutic-objective-namespace-binding-v2",
        "created_at": _now(),
        "objective_uid": OBJECTIVE_UID,
        "objective_namespace": OBJECTIVE_NAMESPACE,
        "objective_freeze_status": objective["freeze_status"],
        "objective_member_count": len(members),
        "components": components,
        "source_bindings": source_binding,
        "objective_namespace_bound": all(row["status"] == "bound" for row in platform_summaries),
        "outcome_driven_mapping_choices": False,
        "ambiguous_mapping_policy_frozen": True,
        "aggregation_rule": AGGREGATION_RULE,
        "protected_holdout_access": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "input_bindings": {"objective_config": _artifact(TARGETED_OBJECTIVE_CONFIG)},
        "output_bindings": {
            "objective_binding": objective_binding_artifact,
        },
        **BOUNDARY,
    }
    if write_output:
        write_json(PLATFORM_AUDIT_PATH, platform_audit)
        write_json(OBJECTIVE_AUDIT_PATH, objective_audit)

    semantic_observations, semantic_candidates, counts = _clone_semantic_rows(platform_status)
    if write_output:
        write_tsv(OBSERVATION_V3_PATH, list(semantic_observations[0]), semantic_observations)
        if semantic_candidates:
            write_tsv(CANDIDATE_V3_PATH, list(semantic_candidates[0]), semantic_candidates)
    rejected_sources = [row for row in platform_summaries if row["status"] != "bound"]
    qualified_count = 0
    candidate_count = 0 if rejected_sources else len(semantic_candidates)
    source_rows = []
    for summary in platform_summaries:
        source_rows.append(
            {
                "source_accession": summary["source_accession"],
                "platform_accession": summary["platform_accession"],
                "platform_status": summary["status"],
                "objective_coverage_fraction": summary["objective_coverage_fraction"],
                "candidate_set_count": 0,
                "qualified_decision_sensitive_count": 0,
                "classification": "rejected_incomplete_objective_coverage" if summary["status"] != "bound" else "pending_numeric_qualification",
                "objective_uid": OBJECTIVE_UID,
                "outcome_blind_objective": True,
                **BOUNDARY,
            }
        )
    if write_output:
        write_tsv(QUALIFICATION_V3_PATH, list(source_rows[0]), source_rows)
    split_audit = {
        "schema_version": "neuronsifter-therapeutic-benchmark-split-feasibility-v3",
        "status": "not_feasible_rejected_incomplete_objective_coverage" if rejected_sources else "pending_numeric_qualification",
        "candidate_set_count": candidate_count,
        "qualified_candidate_set_count": qualified_count,
        "independent_publication_groups": 0,
        "independent_context_groups": 0,
        "group_overlap_count": 0,
        "split_files_created": False,
        "protected_holdout_access": 0,
        **BOUNDARY,
    }
    priority_rows = read_tsv(PRIORITY_TABLE_PATH)
    queue_rows: list[dict[str, object]] = []
    for row in sorted(priority_rows, key=lambda item: int(item["priority_rank"])):
        selected = str(row["selection_class"]) == "selected_for_materialization"
        arm_count = int(row["expected_therapeutic_arm_count"])
        if selected:
            eligibility = "materialized_semantic_binding_rejected"
            action = "retain_rejection_and_do_not_redefine_frozen_objective"
        elif arm_count >= 2:
            eligibility = "eligible_for_targeted_materialization"
            action = "materialize_only_after_authoritative_platform_and_objective_support"
        else:
            eligibility = "not_decision_sensitive_multi_action"
            action = "do_not_materialize_as_ranking_benchmark_source"
        queue_rows.append(
            {
                "priority_rank": int(row["priority_rank"]),
                "source_accession": row["source_accession"],
                "source_repository": row["source_repository"],
                "selection_class": row["selection_class"],
                "expected_therapeutic_arm_count": arm_count,
                "expected_candidate_pair_count": int(row["expected_candidate_pair_count"]),
                "objective_compatibility": row["objective_compatibility"],
                "metadata_only_reason": row["rationale"],
                "eligibility": eligibility,
                "next_action": action,
                "outcome_values_inspected": False,
                "model_predictions_inspected": False,
                **BOUNDARY,
            }
        )
    acquisition_queue = {
        "schema_version": "neuronsifter-therapeutic-targeted-acquisition-queue-v3",
        "created_at": _now(),
        "status": "metadata_queue_exhausted_for_ordinary_transcriptomic_multi_action_sources",
        "selection_basis": "metadata_only",
        "rows": queue_rows,
        "eligible_remaining_multi_action_source_count": sum(row["eligibility"] == "eligible_for_targeted_materialization" for row in queue_rows),
        "non_decision_sensitive_remaining_source_count": sum(row["eligibility"] == "not_decision_sensitive_multi_action" for row in queue_rows),
        "selected_source_binding_status": {str(row["source_accession"]): platform_status[str(row["source_accession"])] for row in platform_summaries},
        "ordinary_transcriptomic_expansion_authorized": False,
        "protected_holdout_access": 0,
        **BOUNDARY,
    }
    if write_output:
        write_json(ACQUISITION_QUEUE_PATH, acquisition_queue)
    gap_map = {
        "schema_version": "neuronsifter-therapeutic-evidence-gap-map-v3",
        "created_at": _now(),
        "terminal_gate": "BLOCKED_INSUFFICIENT_DECISION_SENSITIVE_THERAPEUTIC_OUTCOMES",
        "reason": "frozen_objective_not_fully_supported_on_selected_platforms",
        "targeted_acquisition_queue": ACQUISITION_QUEUE_PATH.relative_to(REPO_ROOT).as_posix(),
        "eligible_remaining_multi_action_source_count": acquisition_queue["eligible_remaining_multi_action_source_count"],
        "non_decision_sensitive_remaining_source_count": acquisition_queue["non_decision_sensitive_remaining_source_count"],
        "source_bindings": source_binding,
        "qualified_candidate_set_count": qualified_count,
        "candidate_set_count": candidate_count,
        "ordinary_transcriptomic_expansion_authorized": False,
        "recommended_next_evidence": [
            "multi_arm_functional_therapeutic_outcomes",
            "dose_or_schedule_comparisons",
            "direct_electrophysiology_or_phenotype_rescue_endpoints",
        ],
        "protected_holdout_access": 0,
        **BOUNDARY,
    }
    domain = {
        "schema_version": "neuronsifter-therapeutic-decision-domain-v3",
        "created_at": _now(),
        "status": "blocked_insufficient_decision_sensitive_therapeutic_outcomes",
        "therapeutic_domain": {
            "status": "blocked_insufficient_decision_sensitive_therapeutic_outcomes",
            "qualified_candidate_set_count": qualified_count,
        },
        "semantic_binding_status": platform_audit["status"],
        "source_count": len(selected_sources),
        "observation_count": counts["observation_count"],
        "action_count": counts["action_count"],
        "therapeutic_action_count": counts["therapeutic_action_count"],
        "matched_control_context_count": len({str(row["context_uid"]) for row in semantic_observations}),
        "structural_multi_action_context_count": len(semantic_candidates),
        "candidate_set_count": candidate_count,
        "qualified_decision_sensitive_sets": qualified_count,
        "objective_namespace_status": "rejected_incomplete_objective_coverage" if rejected_sources else "bound_pending_numeric_qualification",
        "benchmark_created": False,
        "world_action_authorized": False,
        "gpu_campaign_authorized": False,
        "static_reference_authorized": False,
        "tri_authorized": False,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "world_action": {
            "status": "not_authorized",
            "reason": "therapeutic_decision_domain_blocked",
        },
        "tri": {
            "status": "not_authorized",
            "reason": "therapeutic_decision_domain_blocked",
        },
        "agent_in_twin": {
            "status": "not_authorized",
            "reason": "therapeutic_decision_domain_blocked",
        },
        "coevolution": {
            "status": "not_authorized",
            "reason": "therapeutic_decision_domain_blocked",
        },
        "publication_comparison_open": False,
        "protected_holdout_access": 0,
        "historical_round21_authority_recreated": False,
        "old_round22_tier_c_replay_used": False,
        "evidence_gap_map": GAP_MAP_PATH.relative_to(REPO_ROOT).as_posix(),
        **BOUNDARY,
    }
    materialization = {
        "schema_version": "neuronsifter-therapeutic-evidence-materialization-v3",
        "created_at": _now(),
        "status": "semantic_binding_completed_with_rejected_sources" if rejected_sources else "semantic_binding_completed_pending_numeric_qualification",
        "selected_sources": selected_sources,
        "source_count": len(selected_sources),
        "observation_count": counts["observation_count"],
        "action_count": counts["action_count"],
        "therapeutic_action_count": counts["therapeutic_action_count"],
        "structural_candidate_set_count": len(semantic_candidates),
        "candidate_set_count": candidate_count,
        "qualified_candidate_set_count": qualified_count,
        "objective_namespace_status": domain["objective_namespace_status"],
        "outcome_values_inspected_for_source_selection": False,
        "model_predictions_inspected": False,
        "source_summaries": platform_summaries,
        "input_bindings": {
            "structure_observations": _artifact(STRUCTURE_OBSERVATION_PATH),
            "structure_candidates": _artifact(STRUCTURE_CANDIDATE_PATH),
            "source_priority_audit": _artifact(PRIORITY_AUDIT_PATH),
            "source_priority_table": _artifact(PRIORITY_TABLE_PATH),
            "platform_audit": _artifact(PLATFORM_AUDIT_PATH) if write_output else {"path": PLATFORM_AUDIT_PATH.relative_to(REPO_ROOT).as_posix()},
            "objective_audit": _artifact(OBJECTIVE_AUDIT_PATH) if write_output else {"path": OBJECTIVE_AUDIT_PATH.relative_to(REPO_ROOT).as_posix()},
        },
        "output_bindings": {
            "observations": _artifact(OBSERVATION_V3_PATH) if write_output else {"path": OBSERVATION_V3_PATH.relative_to(REPO_ROOT).as_posix()},
            "candidate_sets": _artifact(CANDIDATE_V3_PATH) if write_output and CANDIDATE_V3_PATH.is_file() else {"path": CANDIDATE_V3_PATH.relative_to(REPO_ROOT).as_posix()},
            "qualification": _artifact(QUALIFICATION_V3_PATH) if write_output else {"path": QUALIFICATION_V3_PATH.relative_to(REPO_ROOT).as_posix()},
            "split_feasibility": SPLIT_V3_PATH.relative_to(REPO_ROOT).as_posix(),
            "decision_domain": DOMAIN_V3_PATH.relative_to(REPO_ROOT).as_posix(),
            "targeted_acquisition_queue": ACQUISITION_QUEUE_PATH.relative_to(REPO_ROOT).as_posix(),
        },
        "implementation_binding": _artifact(REPO_ROOT / "Libs/scripts/builders/build_therapeutic_semantic_domain.py"),
        "schema_bindings": {
            "platform": _artifact(PLATFORM_SCHEMA_PATH),
            "objective": _artifact(OBJECTIVE_SCHEMA_PATH),
            "domain": _artifact(DOMAIN_SCHEMA_PATH),
        },
        **BOUNDARY,
    }
    if write_output:
        write_json(SPLIT_V3_PATH, split_audit)
        write_json(GAP_MAP_PATH, gap_map)
        write_json(DOMAIN_V3_PATH, domain)
        write_json(MATERIALIZATION_V3_PATH, materialization)
    return {
        "platform_audit": platform_audit,
        "objective_audit": objective_audit,
        "materialization": materialization,
        "domain": domain,
        "split": split_audit,
        "gap_map": gap_map,
        "acquisition_queue": acquisition_queue,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Build the semantic payload without replacing artifacts")
    args = parser.parse_args()
    result = build(write_output=not args.no_write)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
