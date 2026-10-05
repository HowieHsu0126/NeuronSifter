"""Preregister and freeze one bounded, metadata-only public CNS ranking intake."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable

from Bio import Entrez
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
PREREGISTRATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/retrospective_public_cns_ranking_intake_preregistration.schema.json"
)
INTAKE_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/retrospective_public_cns_ranking_intake.schema.json"
)
FROZEN_QUEUE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_ranking_source_queue.json"
)
REFRESH_QUEUE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_ranking_source_refresh_queue.json"
)
REFRESH_RECOVERY_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_ranking_source_recovery.json"
)
MATERIALIZATION_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_evidence_materialization.json"
)
REPLAY_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
PARSER_FAILURE_SNAPSHOT_PATH = (
    REPO_ROOT
    / "Input/raw/public/retrospective/cns_ranking_intake/pubmed_metadata_snapshot_parser_failure.json"
)
PARSER_FAILURE_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_queue_parser_failure.tsv"
)
PARSER_FAILURE_AUDIT_PAYLOAD_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake_parser_failure_payload.json"
)
PARSER_FAILURE_INVALIDATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake_parser_failure.json"
)
SELECTION_FAILURE_SNAPSHOT_PATH = (
    REPO_ROOT
    / "Input/raw/public/retrospective/cns_ranking_intake/pubmed_metadata_snapshot_selection_failure.json"
)
SELECTION_FAILURE_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_queue_selection_failure.tsv"
)
SELECTION_FAILURE_AUDIT_PAYLOAD_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake_selection_failure_payload.json"
)
SELECTION_FAILURE_INVALIDATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake_selection_failure.json"
)
ENDPOINT_FILTER_FAILURE_SNAPSHOT_PATH = (
    REPO_ROOT
    / "Input/raw/public/retrospective/cns_ranking_intake/pubmed_metadata_snapshot_endpoint_filter_failure.json"
)
ENDPOINT_FILTER_FAILURE_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_queue_endpoint_filter_failure.tsv"
)
ENDPOINT_FILTER_FAILURE_AUDIT_PAYLOAD_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake_endpoint_filter_failure_payload.json"
)
ENDPOINT_FILTER_FAILURE_INVALIDATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake_endpoint_filter_failure.json"
)

EXCLUSION_TABLES = (
    (
        REPO_ROOT / "Input/manifests/retrospective_public/evidence_registry.tsv",
        "publication_uid",
    ),
    (
        REPO_ROOT / "Output/audits/retrospective_public_evidence_rejection_catalog.tsv",
        "publication_uid",
    ),
    (
        REPO_ROOT / "Input/manifests/retrospective_public/ranking_informative_source_queue.tsv",
        "publication_uid",
    ),
    (
        REPO_ROOT / "Input/manifests/retrospective_public/ranking_source_refresh_queue.tsv",
        "publication_uid",
    ),
)

QUEUE_FIELDS = (
    "intake_uid",
    "intake_sequence",
    "preregistration_digest",
    "parent_frozen_queue_digest",
    "parent_refresh_queue_digest",
    "publication_uid",
    "source_identifier",
    "publication_date",
    "journal",
    "title",
    "source_accession",
    "doi",
    "context_hint",
    "context_family",
    "objective_uid",
    "study_design_class",
    "therapeutic_arm_structure",
    "shared_control_expected",
    "same_endpoint_time_expected",
    "replicate_structure_expected",
    "uncertainty_source_expected",
    "independence_key",
    "ranking_informativeness_prior",
    "priority_order",
    "queue_status",
    "metadata_basis",
    "metadata_snapshot_date",
    "metadata_snapshot_path",
    "metadata_snapshot_sha256",
    "preexposed_sources_excluded",
    "abstracts_retrieved_before_priority",
    "full_text_retrieved_before_priority",
    "figures_retrieved_before_priority",
    "outcome_values_used_for_priority",
    "observed_effect_used_for_priority",
    "p_values_used_for_priority",
    "protected_holdout_access",
)

DESIGN_PATTERNS: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    (
        "direct_head_to_head",
        (" versus ", "comparison", "comparative", "head-to-head"),
        "multiple therapeutic conditions under a shared comparison design",
        "high",
    ),
    (
        "combination_vs_monotherapy",
        ("combination", "monotherapy"),
        "combination and component treatment conditions",
        "high",
    ),
    (
        "dose_response",
        ("dose-response", "dose response", "concentration-response", "concentration response"),
        "multiple dose or concentration conditions",
        "medium_high",
    ),
    (
        "multi_arm_screen",
        ("screening", "screen", "high-throughput", "high throughput"),
        "multiple screened intervention conditions",
        "medium",
    ),
)

OBJECTIVE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("electrophysiology", ("electrophysiology", "excitability", "seizure", "firing", "electrical")),
    ("calcium_activity", ("calcium",)),
    ("neurite_morphology", ("neurite", "axon", "dendrite", "morphology")),
    ("synaptic_phenotype", ("synaptic", "synapse", "network activity")),
    ("cell_viability_survival", ("viability", "survival", "cell death", "cytotoxicity")),
    (
        "quantitative_pathology_phenotype",
        ("amyloid", "tau", "aggregate", "pathology", "inflammation", "microglia", "glial"),
    ),
)

CONTEXT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("alzheimer_disease", ("alzheimer", "amyloid", "tau")),
    ("parkinson_disease", ("parkinson", "dopaminergic", "midbrain")),
    ("amyotrophic_lateral_sclerosis", ("amyotrophic", "als", "motor neuron")),
    ("frontotemporal_degeneration", ("frontotemporal", "tdp-43", "tdp43")),
    ("huntington_disease", ("huntington",)),
    ("rett_syndrome", ("rett",)),
    ("epilepsy_hyperexcitability", ("epilep", "seizure", "hyperexcit")),
    ("peripheral_neuropathy", ("neuropathy",)),
    ("cns_injury", ("spinal cord", "brain injury", "ischemia", "stroke")),
    ("neurodegeneration_other", ("neurodegener", "neuronal", "neuron", "organoid")),
)

EXCLUDED_CONTEXT_TERMS = (
    "glioma",
    "glioblastoma",
    "neuroblastoma",
    "medulloblastoma",
    "brain tumor",
    "brain tumour",
)

REQUIRED_CNS_TERMS = (
    "neuron",
    "neuronal",
    "brain",
    "central nervous system",
    "cns",
    "organoid",
    "microglia",
    "astrocyte",
    "synaptic",
    "synapse",
    "neurodegener",
    "alzheimer",
    "parkinson",
    "amyotrophic",
    "epilep",
    "seizure",
    "spinal cord",
)

REQUIRED_THERAPEUTIC_TERMS = (
    "drug",
    "compound",
    "medication",
    "treatment",
    "therapy",
    "therapeutic",
    "inhibitor",
    "agonist",
    "antagonist",
    "pharmacolog",
)

NONTHERAPEUTIC_COMPARISON_TERMS = (
    "classification",
    "diagnostic",
    "biomarker",
    "association",
    "cohort study",
    "unitary synaptic currents",
)

THERAPEUTIC_QUERY_UIDS = {
    "neural_model_compound_screen",
    "neural_dose_response",
    "neurodegeneration_treatment_comparison",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _canonical_digest(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("retrospective_public_cns_ranking_intake")
    if not isinstance(policy, dict):
        raise ValueError("Missing retrospective_public_cns_ranking_intake configuration")
    return policy


def _output_paths(policy: dict[str, Any]) -> dict[str, Path]:
    outputs = policy["outputs"]
    paths = {key: (REPO_ROOT / str(value)).resolve() for key, value in outputs.items()}
    for key, path in paths.items():
        if path != REPO_ROOT and REPO_ROOT not in path.parents:
            raise ValueError(f"Configured intake output escapes repository: {key}={path}")
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_publication_uids(path: Path, field: str) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if field not in (reader.fieldnames or ()):
            raise ValueError(f"Missing {field} in exclusion table: {path}")
        return {str(row.get(field) or "").strip() for row in reader if str(row.get(field) or "").strip()}


def _collect_exclusions(policy: dict[str, Any]) -> dict[str, list[str]]:
    publication_uids: set[str] = set()
    for path, field in EXCLUSION_TABLES:
        publication_uids.update(_read_publication_uids(path, field))
    recovery = _read_json(REFRESH_RECOVERY_AUDIT_PATH)
    publication_uid = str(recovery.get("publication_uid") or "")
    if publication_uid:
        publication_uids.add(publication_uid)
    pmids = sorted(
        value.removeprefix("pubmed:")
        for value in publication_uids
        if value.startswith("pubmed:")
    )
    dois = sorted(
        {
            value.removeprefix("doi:").lower()
            for value in publication_uids
            if value.startswith("doi:")
        }
        | {
            str(value).lower()
            for value in policy["selection_policy"]["explicit_preexposed_source_dois"]
        }
    )
    return {
        "preexposed_publication_uids": sorted(publication_uids),
        "preexposed_pmids": pmids,
        "preexposed_dois": dois,
    }


def _assert_parent_state(policy: dict[str, Any]) -> dict[str, dict[str, object]]:
    expected = policy["parent_state"]
    frozen = _read_json(FROZEN_QUEUE_AUDIT_PATH)
    refresh = _read_json(REFRESH_QUEUE_AUDIT_PATH)
    recovery = _read_json(REFRESH_RECOVERY_AUDIT_PATH)
    materialization = _read_json(MATERIALIZATION_AUDIT_PATH)
    replay = _read_json(REPLAY_AUDIT_PATH)
    checks = {
        "frozen_queue_digest": frozen.get("queue_digest"),
        "refresh_queue_digest": refresh.get("queue_digest"),
        "parent_queue_unchanged": refresh.get("parent_queue_unchanged"),
        "refresh_terminal_action": recovery.get("next_action"),
        "status": materialization.get("status"),
        "accepted_records": materialization.get("accepted_record_count"),
        "rejected_records": materialization.get("rejected_record_count"),
        "independent_decision_sensitive_groups": replay.get("independent_decision_sensitive_group_count"),
        "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
        "split_files_created": replay.get("split_files_created"),
        "protected_holdout_access": replay.get("protected_holdout_access"),
    }
    required = {
        "frozen_queue_digest": expected["frozen_queue_digest"],
        "refresh_queue_digest": expected["refresh_queue_digest"],
        "parent_queue_unchanged": True,
        "refresh_terminal_action": "stop_bounded_refresh_preserve_two_of_three_decision_sensitive_groups",
        "status": expected["status"],
        "accepted_records": expected["accepted_records"],
        "rejected_records": expected["rejected_records"],
        "independent_decision_sensitive_groups": expected["independent_decision_sensitive_groups"],
        "required_groups": expected["required_groups"],
        "split_files_created": False,
        "protected_holdout_access": 0,
    }
    if checks != required:
        raise ValueError(f"Parent retrospective state drifted: observed={checks} expected={required}")
    return {
        "frozen_queue_audit": _binding(FROZEN_QUEUE_AUDIT_PATH),
        "refresh_queue_audit": _binding(REFRESH_QUEUE_AUDIT_PATH),
        "refresh_recovery_audit": _binding(REFRESH_RECOVERY_AUDIT_PATH),
        "materialization_audit": _binding(MATERIALIZATION_AUDIT_PATH),
        "replay_audit": _binding(REPLAY_AUDIT_PATH),
    }


def _schema_errors(payload: dict[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _validate_preregistration(payload: dict[str, Any], policy: dict[str, Any]) -> None:
    errors = _schema_errors(payload, PREREGISTRATION_SCHEMA_PATH)
    if errors:
        raise ValueError("Preregistration schema errors: " + "; ".join(errors[:8]))
    digest_payload = dict(payload)
    observed_digest = str(digest_payload.pop("preregistration_digest"))
    if _canonical_digest(digest_payload) != observed_digest:
        raise ValueError("Preregistration digest drifted")
    if payload["intake_uid"] != policy["intake_uid"]:
        raise ValueError("Preregistration intake UID drifted")
    if payload["authorization"] != policy["authorization"]:
        raise ValueError("Preregistration authorization drifted")
    if payload["search_plan"] != policy["search_plan"]:
        raise ValueError("Preregistration search plan drifted")
    if payload["selection_policy"] != policy["selection_policy"]:
        raise ValueError("Preregistration selection policy drifted")
    if payload["boundaries"] != policy["boundaries"]:
        raise ValueError("Preregistration boundaries drifted")


def preregister(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    if paths["preregistration_manifest"].exists():
        payload = _read_json(paths["preregistration_manifest"])
        _validate_preregistration(payload, policy)
        return payload
    parent_bindings = _assert_parent_state(policy)
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-preregistration-v1",
        "status": "preregistered_before_metadata_retrieval",
        "intake_uid": policy["intake_uid"],
        "preregistered_at": _now(),
        "authorization": policy["authorization"],
        "parent_state": policy["parent_state"],
        "parent_bindings": parent_bindings,
        "search_plan": policy["search_plan"],
        "selection_policy": policy["selection_policy"],
        "exclusions": _collect_exclusions(policy),
        "boundaries": policy["boundaries"],
        "outputs": policy["outputs"],
    }
    payload["preregistration_digest"] = _canonical_digest(payload)
    _validate_preregistration(payload, policy)
    if write_output:
        write_json(paths["preregistration_manifest"], payload)
    return payload


def _entrez_email() -> str:
    configured = os.environ.get("NCBI_EMAIL", "").strip()
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "config", "--get", "user.email"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    configured = completed.stdout.strip()
    if not configured:
        raise ValueError("NCBI metadata retrieval requires NCBI_EMAIL or git user.email")
    return configured


def _search_pubmed(query: str, *, retmax: int, sort: str) -> tuple[int, list[str]]:
    with Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort=sort) as handle:
        result = Entrez.read(handle)
    return int(result["Count"]), [str(value) for value in result["IdList"]]


def _article_ids(record: Any) -> dict[str, str]:
    raw = record.get("ArticleIds", {})
    if isinstance(raw, dict):
        values: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, (list, tuple)):
                normalized = str(value[0]).strip() if value else ""
            else:
                normalized = str(value).strip()
            if normalized:
                values[str(key).lower()] = normalized
        return values
    values: dict[str, str] = {}
    for item in raw:
        attributes = getattr(item, "attributes", {})
        key = str(attributes.get("IdType") or "").lower()
        if key:
            values[key] = str(item).strip()
    return values


def archive_parser_failure(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    required_paths = (
        paths["metadata_snapshot"],
        paths["source_queue"],
        paths["intake_audit"],
    )
    if not all(path.is_file() for path in required_paths):
        raise ValueError("Parser-failure archival requires the complete failed discovery output set")
    audit = _read_json(paths["intake_audit"])
    snapshot = _read_json(paths["metadata_snapshot"])
    rejected = snapshot.get("exclusions", {}).get("rejected_metadata_records", [])
    if audit.get("status") != "metadata_only_queue_exhausted" or audit.get("record_count") != 0:
        raise ValueError("Current intake output is not the expected zero-record parser failure")
    if not rejected or any(item.get("reason") != "no_public_pmc_link" for item in rejected):
        raise ValueError("Parser-failure signature is not the all-PMC-link false-negative pattern")
    invalidation = {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-parser-failure-v1",
        "status": "invalidated_engineering_parser_failure",
        "reason": "biopython_esummary_article_ids_mapping_misparsed_as_iterable_keys",
        "scientific_evidence_admitted": False,
        "formal_queue_status": "not_admitted",
        "query_plan_changed": False,
        "query_result_set_expanded": False,
        "abstracts_retrieved": False,
        "full_text_retrieved": False,
        "figures_retrieved": False,
        "outcome_values_retrieved": False,
        "protected_holdout_access": 0,
        "failed_snapshot_binding": _binding(paths["metadata_snapshot"]),
        "failed_queue_binding": _binding(paths["source_queue"]),
        "failed_audit_binding": _binding(paths["intake_audit"]),
        "recovery_rule": "reuse_exact_preserved_esearch_pmids_and_refetch_esummary_metadata_only",
        "invalidated_at": _now(),
    }
    if write_output:
        targets = (
            (paths["metadata_snapshot"], PARSER_FAILURE_SNAPSHOT_PATH),
            (paths["source_queue"], PARSER_FAILURE_QUEUE_PATH),
            (paths["intake_audit"], PARSER_FAILURE_AUDIT_PAYLOAD_PATH),
        )
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError(f"Existing parser-failure archive drifted: {target}")
            else:
                shutil.copy2(source, target)
        write_json(PARSER_FAILURE_INVALIDATION_PATH, invalidation)
    return invalidation


def archive_selection_failure(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    required_paths = (
        paths["metadata_snapshot"],
        paths["source_queue"],
        paths["intake_audit"],
    )
    if not all(path.is_file() for path in required_paths):
        raise ValueError("Selection-failure archival requires the complete discovery output set")
    audit = _read_json(paths["intake_audit"])
    rows: list[dict[str, str]] = []
    with paths["source_queue"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
    if audit.get("status") != "metadata_only_queue_frozen" or audit.get("record_count") != len(rows):
        raise ValueError("Current intake output is not a complete frozen queue")
    if not rows:
        raise ValueError("Selection-failure archival requires at least one selected row")
    invalid_rows = []
    for row in rows:
        lowered = str(row.get("title") or "").lower()
        has_cns = _contains_any(lowered, REQUIRED_CNS_TERMS)
        has_therapeutic = _contains_any(lowered, REQUIRED_THERAPEUTIC_TERMS)
        is_nontherapeutic = _contains_any(lowered, NONTHERAPEUTIC_COMPARISON_TERMS)
        if has_cns and has_therapeutic and not is_nontherapeutic:
            raise ValueError("Current queue contains a row that passes the repaired semantic eligibility rule")
        invalid_rows.append(
            {
                "publication_uid": row.get("publication_uid"),
                "has_cns_signal": has_cns,
                "has_therapeutic_action_signal": has_therapeutic,
                "nontherapeutic_comparison_signal": is_nontherapeutic,
            }
        )
    invalidation = {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-selection-failure-v1",
        "status": "invalidated_engineering_selection_failure",
        "reason": "metadata_classifier_allowed_nontherapeutic_or_non_cns_comparisons",
        "scientific_evidence_admitted": False,
        "formal_queue_status": "not_admitted",
        "query_plan_changed": False,
        "query_result_set_expanded": False,
        "ranking_order_changed": False,
        "abstracts_retrieved": False,
        "full_text_retrieved": False,
        "figures_retrieved": False,
        "outcome_values_retrieved": False,
        "protected_holdout_access": 0,
        "invalid_rows": invalid_rows,
        "failed_snapshot_binding": _binding(paths["metadata_snapshot"]),
        "failed_queue_binding": _binding(paths["source_queue"]),
        "failed_audit_binding": _binding(paths["intake_audit"]),
        "recovery_rule": "reuse_exact_preserved_esearch_pmids_with_repaired_cns_therapeutic_eligibility",
        "invalidated_at": _now(),
    }
    if write_output:
        targets = (
            (paths["metadata_snapshot"], SELECTION_FAILURE_SNAPSHOT_PATH),
            (paths["source_queue"], SELECTION_FAILURE_QUEUE_PATH),
            (paths["intake_audit"], SELECTION_FAILURE_AUDIT_PAYLOAD_PATH),
        )
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError(f"Existing selection-failure archive drifted: {target}")
            else:
                shutil.copy2(source, target)
        write_json(SELECTION_FAILURE_INVALIDATION_PATH, invalidation)
    return invalidation


def archive_endpoint_filter_failure(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    required_paths = (
        paths["metadata_snapshot"],
        paths["source_queue"],
        paths["intake_audit"],
    )
    if not all(path.is_file() for path in required_paths):
        raise ValueError("Endpoint-filter archival requires the complete discovery output set")
    audit = _read_json(paths["intake_audit"])
    snapshot = _read_json(paths["metadata_snapshot"])
    if audit.get("status") != "metadata_only_queue_exhausted" or audit.get("record_count") != 0:
        raise ValueError("Current intake output is not the expected zero-record filter failure")
    rejected = snapshot.get("exclusions", {}).get("rejected_metadata_records", [])
    reasons = {str(item.get("reason") or "") for item in rejected}
    if "no_therapeutic_action_signal" not in reasons:
        raise ValueError("Endpoint-filter failure signature is missing")
    invalidation = {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-endpoint-filter-failure-v1",
        "status": "invalidated_engineering_endpoint_filter_failure",
        "reason": "functional_calcium_imaging_overexcluded_and_drug_query_signal_ignored",
        "scientific_evidence_admitted": False,
        "formal_queue_status": "not_admitted",
        "query_plan_changed": False,
        "query_result_set_expanded": False,
        "ranking_order_changed": False,
        "abstracts_retrieved": False,
        "full_text_retrieved": False,
        "figures_retrieved": False,
        "outcome_values_retrieved": False,
        "protected_holdout_access": 0,
        "failed_snapshot_binding": _binding(paths["metadata_snapshot"]),
        "failed_queue_binding": _binding(paths["source_queue"]),
        "failed_audit_binding": _binding(paths["intake_audit"]),
        "recovery_rule": "reuse_exact_preserved_esearch_pmids_with_corrected_functional_endpoint_filter",
        "invalidated_at": _now(),
    }
    if write_output:
        targets = (
            (paths["metadata_snapshot"], ENDPOINT_FILTER_FAILURE_SNAPSHOT_PATH),
            (paths["source_queue"], ENDPOINT_FILTER_FAILURE_QUEUE_PATH),
            (paths["intake_audit"], ENDPOINT_FILTER_FAILURE_AUDIT_PAYLOAD_PATH),
        )
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError(f"Existing endpoint-filter archive drifted: {target}")
            else:
                shutil.copy2(source, target)
        write_json(ENDPOINT_FILTER_FAILURE_INVALIDATION_PATH, invalidation)
    return invalidation


def _summaries(pmids: Iterable[str]) -> list[dict[str, Any]]:
    identifiers = list(pmids)
    records: list[dict[str, Any]] = []
    for start in range(0, len(identifiers), 50):
        batch = identifiers[start : start + 50]
        if not batch:
            continue
        with Entrez.esummary(db="pubmed", id=",".join(batch)) as handle:
            payload = Entrez.read(handle)
        for item in payload:
            article_ids = _article_ids(item)
            records.append(
                {
                    "pmid": str(item.get("Id") or article_ids.get("pubmed") or "").strip(),
                    "title": " ".join(str(item.get("Title") or "").split()),
                    "publication_date": str(item.get("PubDate") or "").strip(),
                    "journal": str(item.get("FullJournalName") or item.get("Source") or "").strip(),
                    "publication_types": sorted(str(value) for value in item.get("PubTypeList", [])),
                    "pmcid": article_ids.get("pmc", ""),
                    "doi": article_ids.get("doi", "").lower(),
                }
            )
    return records


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _design_class(title: str) -> tuple[str, str, str] | None:
    padded = f" {title.lower()} "
    for design_class, terms, arm_structure, prior in DESIGN_PATTERNS:
        if _contains_any(padded, terms):
            return design_class, arm_structure, prior
    return None


def _objective_family(title: str) -> str | None:
    lowered = title.lower()
    for family, terms in OBJECTIVE_PATTERNS:
        if _contains_any(lowered, terms):
            return family
    return None


def _context_family(title: str) -> str:
    lowered = title.lower()
    for family, terms in CONTEXT_PATTERNS:
        if _contains_any(lowered, terms):
            return family
    return "other_public_cns_context"


def _publication_year(value: str) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else 0


def _screen_metadata_record(
    record: dict[str, Any],
    *,
    query_uids: list[str],
    policy: dict[str, Any],
    exclusions: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, str | None]:
    pmid = str(record.get("pmid") or "")
    doi = str(record.get("doi") or "").lower()
    title = str(record.get("title") or "")
    lowered = title.lower()
    if not pmid or not title:
        return None, "missing_identifier_or_title"
    if pmid in exclusions["preexposed_pmids"] or doi in exclusions["preexposed_dois"]:
        return None, "preexposed_source"
    if not record.get("pmcid"):
        return None, "no_public_pmc_link"
    publication_types = {str(value) for value in record.get("publication_types", [])}
    required_type = str(policy["selection_policy"]["required_publication_type"])
    excluded_types = set(policy["selection_policy"]["excluded_publication_types"])
    if required_type not in publication_types:
        return None, "not_journal_article"
    if publication_types & excluded_types:
        return None, "excluded_publication_type"
    if _contains_any(lowered, policy["selection_policy"]["outcome_bearing_title_terms"]):
        return None, "outcome_bearing_title"
    if _contains_any(lowered, EXCLUDED_CONTEXT_TERMS):
        return None, "out_of_scope_oncology_context"
    if not _contains_any(lowered, REQUIRED_CNS_TERMS):
        return None, "not_cns_context"
    if _contains_any(lowered, NONTHERAPEUTIC_COMPARISON_TERMS):
        return None, "diagnostic_or_observational_comparison"
    therapeutic_query_signal = any(query_uid in THERAPEUTIC_QUERY_UIDS for query_uid in query_uids)
    if not _contains_any(lowered, REQUIRED_THERAPEUTIC_TERMS) and not therapeutic_query_signal:
        return None, "no_therapeutic_action_signal"
    design = _design_class(title)
    if design is None:
        return None, "no_prespecified_design_signal"
    objective_family = _objective_family(title)
    if objective_family is None:
        return None, "no_prespecified_functional_endpoint_signal"
    design_class, arm_structure, prior = design
    objective_uid = policy["selection_policy"]["objective_uids"][objective_family]
    context_family = _context_family(title)
    return {
        **record,
        "query_uids": sorted(query_uids),
        "query_support_count": len(query_uids),
        "study_design_class": design_class,
        "therapeutic_arm_structure": arm_structure,
        "ranking_informativeness_prior": prior,
        "objective_family": objective_family,
        "objective_uid": objective_uid,
        "context_family": context_family,
        "publication_year": _publication_year(str(record.get("publication_date") or "")),
    }, None


def _select_records(
    summaries: list[dict[str, Any]],
    *,
    query_membership: dict[str, list[str]],
    policy: dict[str, Any],
    exclusions: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for record in summaries:
        pmid = str(record.get("pmid") or "")
        screened, reason = _screen_metadata_record(
            record,
            query_uids=query_membership.get(pmid, []),
            policy=policy,
            exclusions=exclusions,
        )
        if screened is not None:
            eligible.append(screened)
            continue
        rejected.append(
            {
                "pmid": pmid,
                "reason": str(reason),
                "title_sha256": hashlib.sha256(
                    str(record.get("title") or "").encode("utf-8")
                ).hexdigest(),
            }
        )
    design_rank = {
        "direct_head_to_head": 0,
        "combination_vs_monotherapy": 1,
        "dose_response": 2,
        "multi_arm_screen": 3,
    }
    selected: list[dict[str, Any]] = []
    remaining = list(eligible)
    cap = int(policy["authorization"]["source_count_cap"])
    while remaining and len(selected) < cap:
        selected_contexts = {item["context_family"] for item in selected}
        remaining.sort(
            key=lambda item: (
                design_rank[item["study_design_class"]],
                item["context_family"] in selected_contexts,
                -int(item["query_support_count"]),
                -int(item["publication_year"]),
                str(item["pmid"]),
            )
        )
        selected.append(remaining.pop(0))
    return selected, sorted(rejected, key=lambda item: (item["reason"], item["pmid"]))


def _queue_rows(
    selected: list[dict[str, Any]],
    *,
    preregistration: dict[str, Any],
    snapshot_path: Path,
    snapshot_sha256: str,
) -> list[dict[str, str]]:
    parent = preregistration["parent_state"]
    rows: list[dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        pmid = str(record["pmid"])
        rows.append(
            {
                "intake_uid": str(preregistration["intake_uid"]),
                "intake_sequence": str(index),
                "preregistration_digest": str(preregistration["preregistration_digest"]),
                "parent_frozen_queue_digest": str(parent["frozen_queue_digest"]),
                "parent_refresh_queue_digest": str(parent["refresh_queue_digest"]),
                "publication_uid": f"pubmed:{pmid}",
                "source_identifier": f"pmid:{pmid}",
                "publication_date": str(record["publication_date"]),
                "journal": str(record["journal"]),
                "title": str(record["title"]),
                "source_accession": str(record["pmcid"]),
                "doi": str(record["doi"]),
                "context_hint": f"{record['context_family']} public CNS functional comparison",
                "context_family": str(record["context_family"]),
                "objective_uid": str(record["objective_uid"]),
                "study_design_class": str(record["study_design_class"]),
                "therapeutic_arm_structure": str(record["therapeutic_arm_structure"]),
                "shared_control_expected": "true",
                "same_endpoint_time_expected": "true",
                "replicate_structure_expected": "biological replicate structure requires artifact inspection",
                "uncertainty_source_expected": "source table figure or supplementary uncertainty required",
                "independence_key": f"pubmed:{pmid}",
                "ranking_informativeness_prior": str(record["ranking_informativeness_prior"]),
                "priority_order": str(index),
                "queue_status": "preregistered_for_public_artifact_recovery",
                "metadata_basis": (
                    "pubmed esearch and esummary only; neutral title design signal; "
                    f"pmc link; query support {record['query_support_count']}"
                ),
                "metadata_snapshot_date": str(preregistration["preregistered_at"])[:10],
                "metadata_snapshot_path": snapshot_path.relative_to(REPO_ROOT).as_posix(),
                "metadata_snapshot_sha256": snapshot_sha256,
                "preexposed_sources_excluded": "true",
                "abstracts_retrieved_before_priority": "false",
                "full_text_retrieved_before_priority": "false",
                "figures_retrieved_before_priority": "false",
                "outcome_values_used_for_priority": "false",
                "observed_effect_used_for_priority": "false",
                "p_values_used_for_priority": "false",
                "protected_holdout_access": "0",
            }
        )
    return rows


def discover(
    *,
    write_output: bool = True,
    repair_parser_failure: bool = False,
    repair_selection_failure: bool = False,
    repair_endpoint_filter_failure: bool = False,
) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=write_output)
    repair_requested = (
        repair_parser_failure
        or repair_selection_failure
        or repair_endpoint_filter_failure
    )
    if paths["intake_audit"].exists() and not repair_requested:
        return _read_json(paths["intake_audit"])
    if repair_parser_failure and not PARSER_FAILURE_INVALIDATION_PATH.is_file():
        raise ValueError("Parser repair requires a persisted invalidation audit")
    if repair_selection_failure and not SELECTION_FAILURE_INVALIDATION_PATH.is_file():
        raise ValueError("Selection repair requires a persisted invalidation audit")
    if repair_endpoint_filter_failure and not ENDPOINT_FILTER_FAILURE_INVALIDATION_PATH.is_file():
        raise ValueError("Endpoint-filter repair requires a persisted invalidation audit")
    if sum((repair_parser_failure, repair_selection_failure, repair_endpoint_filter_failure)) > 1:
        raise ValueError("Only one repair mode may be selected")
    if not repair_requested and (
        paths["metadata_snapshot"].exists() or paths["source_queue"].exists()
    ):
        raise ValueError("Partial intake outputs exist without a canonical intake audit")
    _assert_parent_state(policy)
    Entrez.email = _entrez_email()
    Entrez.tool = "NeuronSifterPublicCNSRankingIntake"
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        Entrez.api_key = api_key
    search_plan = policy["search_plan"]
    if len(search_plan["queries"]) != int(search_plan["max_queries"]):
        raise ValueError("Configured query count does not match the bounded search plan")
    query_results: list[dict[str, Any]] = []
    query_membership: dict[str, list[str]] = {}
    query_result_reuse_binding: dict[str, object] | None = None
    if repair_requested:
        preserved_path = (
            PARSER_FAILURE_SNAPSHOT_PATH
            if repair_parser_failure
            else (
                SELECTION_FAILURE_SNAPSHOT_PATH
                if repair_selection_failure
                else ENDPOINT_FILTER_FAILURE_SNAPSHOT_PATH
            )
        )
        preserved_snapshot = _read_json(preserved_path)
        query_results = list(preserved_snapshot["search_plan"]["queries"])
        query_result_reuse_binding = _binding(preserved_path)
        if [item["query_uid"] for item in query_results] != [
            item["query_uid"] for item in search_plan["queries"]
        ]:
            raise ValueError("Preserved query results do not match the preregistered query plan")
        for query in query_results:
            for pmid in query["returned_pmids"]:
                query_membership.setdefault(str(pmid), []).append(str(query["query_uid"]))
    else:
        for query_spec in search_plan["queries"]:
            count, pmids = _search_pubmed(
                str(query_spec["query"]),
                retmax=int(search_plan["max_ids_per_query"]),
                sort=str(search_plan["sort"]),
            )
            query_uid = str(query_spec["query_uid"])
            query_results.append(
                {
                    "query_uid": query_uid,
                    "query": query_spec["query"],
                    "count": count,
                    "retmax": int(search_plan["max_ids_per_query"]),
                    "returned_pmids": pmids,
                }
            )
            for pmid in pmids:
                query_membership.setdefault(pmid, []).append(query_uid)
    all_pmids = sorted(query_membership, key=int)
    metadata_records = _summaries(all_pmids)
    selected, rejected = _select_records(
        metadata_records,
        query_membership=query_membership,
        policy=policy,
        exclusions=preregistration["exclusions"],
    )
    snapshot = {
        "schema_version": "neuronsifter-pubmed-metadata-snapshot-v2",
        "source": "NCBI PubMed ESearch and ESummary",
        "retrieved_at": _now(),
        "retrieval_mode": "metadata_only",
        "preregistration_binding": {
            "path": paths["preregistration_manifest"].relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(paths["preregistration_manifest"]),
            "preregistration_digest": preregistration["preregistration_digest"],
        },
        "search_plan": {
            "bounded_query_count": len(query_results),
            "maximum_ids_per_query": int(search_plan["max_ids_per_query"]),
            "queries": query_results,
        },
        "source_boundaries": {
            "abstracts_retrieved": False,
            "full_text_retrieved": False,
            "figures_retrieved": False,
            "outcome_values_retrieved": False,
            "p_values_retrieved": False,
            "protected_holdout_access": 0,
        },
        "exclusions": {
            **preregistration["exclusions"],
            "outcome_bearing_titles_excluded_before_priority": True,
            "rejected_metadata_records": rejected,
        },
        "selection": {
            "source_count_cap": int(policy["authorization"]["source_count_cap"]),
            "unique_metadata_record_count": len(metadata_records),
            "eligible_metadata_record_count": len(selected),
            "selected_pmids": [record["pmid"] for record in selected],
            "ranking_order": policy["selection_policy"]["ranking_order"],
            "outcome_values_used_for_priority": False,
            "observed_effect_used_for_priority": False,
            "p_values_used_for_priority": False,
        },
        "selected_records": selected,
    }
    if query_result_reuse_binding is not None:
        snapshot["query_result_reuse_binding"] = query_result_reuse_binding
        snapshot["query_result_source"] = (
            "preserved_preregistered_esearch_results_after_parser_repair"
            if repair_parser_failure
            else (
                "preserved_preregistered_esearch_results_after_selection_repair"
                if repair_selection_failure
                else "preserved_preregistered_esearch_results_after_endpoint_filter_repair"
            )
        )
    if write_output:
        write_json(paths["metadata_snapshot"], snapshot)
        snapshot_binding = _binding(paths["metadata_snapshot"])
    else:
        serialized = json.dumps(snapshot, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
        snapshot_binding = {
            "path": paths["metadata_snapshot"].relative_to(REPO_ROOT).as_posix(),
            "bytes": len(serialized),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        }
    rows = _queue_rows(
        selected,
        preregistration=preregistration,
        snapshot_path=paths["metadata_snapshot"],
        snapshot_sha256=str(snapshot_binding["sha256"]),
    )
    queue_digest = _canonical_digest(rows)
    if write_output:
        write_tsv(paths["source_queue"], list(QUEUE_FIELDS), rows)
    status = "metadata_only_queue_frozen" if rows else "metadata_only_queue_exhausted"
    audit: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-v1",
        "status": status,
        "intake_uid": preregistration["intake_uid"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "source_count_cap": int(policy["authorization"]["source_count_cap"]),
        "record_count": len(rows),
        "queue_digest": queue_digest,
        "queue_path": paths["source_queue"].relative_to(REPO_ROOT).as_posix(),
        "metadata_snapshot_binding": snapshot_binding,
        "priority_order": [row["publication_uid"] for row in rows],
        "selection_summary": {
            "query_count": len(query_results),
            "returned_identifier_count": len(all_pmids),
            "metadata_record_count": len(metadata_records),
            "selected_record_count": len(rows),
            "rejected_metadata_record_count": len(rejected),
        },
        "parent_state": preregistration["parent_state"],
        "parent_queue_unchanged": True,
        "refresh_queue_unchanged": True,
        "metadata_network_access_performed": True,
        "retrieval_mode": "metadata_only",
        "abstracts_retrieved_before_priority": False,
        "full_text_retrieved_before_priority": False,
        "figures_retrieved_before_priority": False,
        "outcome_values_used_for_priority": False,
        "observed_effect_used_for_priority": False,
        "p_values_used_for_priority": False,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": 0,
        "next_action": (
            "recover_preregistered_sources_in_priority_order"
            if rows
            else "stop_queue_exhausted_preserve_two_of_three_groups"
        ),
        "records": rows,
        "created_at": _now(),
    }
    errors = _schema_errors(audit, INTAKE_SCHEMA_PATH)
    if errors:
        raise ValueError("Intake audit schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(paths["intake_audit"], audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preregister",
            "discover",
            "archive-parser-failure",
            "archive-selection-failure",
            "archive-endpoint-filter-failure",
        ),
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--repair-parser-failure", action="store_true")
    parser.add_argument("--repair-selection-failure", action="store_true")
    parser.add_argument("--repair-endpoint-filter-failure", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        result = preregister(write_output=not args.no_write)
    elif args.mode == "archive-parser-failure":
        result = archive_parser_failure(write_output=not args.no_write)
    elif args.mode == "archive-selection-failure":
        result = archive_selection_failure(write_output=not args.no_write)
    elif args.mode == "archive-endpoint-filter-failure":
        result = archive_endpoint_filter_failure(write_output=not args.no_write)
    else:
        result = discover(
            write_output=not args.no_write,
            repair_parser_failure=args.repair_parser_failure,
            repair_selection_failure=args.repair_selection_failure,
            repair_endpoint_filter_failure=args.repair_endpoint_filter_failure,
        )
    summary_keys = (
        "status",
        "intake_uid",
        "preregistration_digest",
        "source_count_cap",
        "record_count",
        "queue_digest",
        "next_action",
    )
    print(json.dumps({key: result[key] for key in summary_keys if key in result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
