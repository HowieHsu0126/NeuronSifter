"""Validate platform/objective semantic closure and the v3 evidence-domain gate."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_therapeutic_semantic_domain import (  # noqa: E402
    AGGREGATION_RULE,
    ACQUISITION_QUEUE_PATH,
    CANDIDATE_V3_PATH,
    DOMAIN_SCHEMA_PATH,
    DOMAIN_V3_PATH,
    GAP_MAP_PATH,
    MATERIALIZATION_V3_PATH,
    OBJECTIVE_AUDIT_PATH,
    OBJECTIVE_BINDING_PATH,
    OBJECTIVE_NAMESPACE,
    OBJECTIVE_UID,
    OBJECTIVE_SCHEMA_PATH,
    OBSERVATION_V3_PATH,
    PLATFORM_AUDIT_PATH,
    PLATFORM_MANIFEST_PATH,
    PLATFORM_SCHEMA_PATH,
    QUALIFICATION_V3_PATH,
    SPLIT_V3_PATH,
    _objective,
    _platform_for_source,
    build,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_schema(payload: Mapping[str, object], schema_path: Path, blockers: list[str], label: str) -> None:
    schema = _json(schema_path)
    Draft202012Validator.check_schema(schema)
    for error in Draft202012Validator(schema).iter_errors(payload):
        location = ".".join(str(value) for value in error.absolute_path) or "root"
        blockers.append(f"{label}_schema:{location}:{error.message}")


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def _validate_annotation_files(platform_audit: Mapping[str, object], blockers: list[str]) -> None:
    """Verify the downloaded ADF bytes, not only hashes copied into manifests."""

    for summary in platform_audit.get("platform_summaries", []):
        source = str(summary.get("source_accession", ""))
        annotation_value = summary.get("annotation_path")
        if not isinstance(annotation_value, str) or not annotation_value:
            blockers.append(f"annotation_path:{source}")
            continue
        raw_path = REPO_ROOT / annotation_value
        if raw_path.is_symlink():
            blockers.append(f"annotation_symlink:{source}")
        path = raw_path.resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            blockers.append(f"annotation_path_escape:{source}")
            continue
        if not path.is_file():
            blockers.append(f"annotation_missing:{source}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != summary.get("annotation_content_sha256"):
            blockers.append(f"annotation_hash_actual:{source}")


def _validate_semantic_rows(
    feature_rows: list[dict[str, str]],
    binding_rows: list[dict[str, str]],
    platform_audit: Mapping[str, object],
    objective_audit: Mapping[str, object],
) -> list[str]:
    """Validate TSV contracts that cannot be expressed by the JSON schemas."""

    blockers: list[str] = []
    summaries = {
        str(summary["source_accession"]): summary
        for summary in platform_audit.get("platform_summaries", [])
    }
    component_by_member: dict[str, str] = {}
    for component in objective_audit.get("components", []):
        component_uid = str(component.get("component_uid", ""))
        for member in component.get("gene_symbols", []):
            member_text = str(member)
            if member_text in component_by_member or not component_uid:
                blockers.append(f"objective_component_mapping:{member_text}")
            component_by_member[member_text] = component_uid
    expected_members = set(component_by_member)
    if len(expected_members) != int(objective_audit.get("objective_member_count", -1)):
        blockers.append("objective_member_count_consistency")

    allowed_feature_statuses = {
        "non_objective_feature",
        "unmapped_feature",
        "one_feature_to_multiple_entities",
        "bound_numeric",
        "bound_symbol_fallback",
        "contradictory_annotations",
    }
    features_by_source: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in feature_rows:
        source = _text(row, "source_accession")
        feature_id = _text(row, "feature_id")
        if source not in summaries:
            blockers.append(f"feature_unknown_source:{source}")
            continue
        if not feature_id:
            blockers.append(f"feature_missing_id:{source}")
            continue
        if feature_id in features_by_source[source]:
            blockers.append(f"feature_duplicate_id:{source}:{feature_id}")
        features_by_source[source][feature_id] = row
        summary = summaries[source]
        if _text(row, "platform_accession") != str(summary["platform_accession"]):
            blockers.append(f"feature_platform:{source}:{feature_id}")
        if _text(row, "species") != str(summary["species"]):
            blockers.append(f"feature_species:{source}:{feature_id}")
        if _text(row, "annotation_file_sha256") != str(summary["annotation_content_sha256"]):
            blockers.append(f"feature_annotation_hash:{source}:{feature_id}")
        member = _text(row, "objective_member")
        status = _text(row, "mapping_status")
        canonical_id = _text(row, "canonical_entity_id")
        namespace = _text(row, "canonical_namespace")
        if status not in allowed_feature_statuses:
            blockers.append(f"feature_mapping_status:{source}:{feature_id}")
        if member and member not in expected_members:
            blockers.append(f"feature_unknown_objective_member:{source}:{feature_id}")
        if not member and (canonical_id or namespace):
            blockers.append(f"feature_nonobjective_canonical_id:{source}:{feature_id}")
        if member and status in {"bound_numeric", "bound_symbol_fallback"} and not (canonical_id and namespace):
            blockers.append(f"feature_bound_identity_missing:{source}:{feature_id}")
        if status == "bound_numeric" and namespace != "NCBIGene":
            blockers.append(f"feature_numeric_namespace:{source}:{feature_id}")
        if status == "bound_symbol_fallback" and namespace != "NCBIGeneSymbol":
            blockers.append(f"feature_symbol_namespace:{source}:{feature_id}")
        if status in {"non_objective_feature", "unmapped_feature", "one_feature_to_multiple_entities"} and member:
            blockers.append(f"feature_nonobjective_member:{source}:{feature_id}")
        if status in {"non_objective_feature", "unmapped_feature", "one_feature_to_multiple_entities"} and (canonical_id or namespace):
            blockers.append(f"feature_nonobjective_identity:{source}:{feature_id}")
        if status == "contradictory_annotations" and (canonical_id or namespace):
            blockers.append(f"feature_contradictory_identity:{source}:{feature_id}")

    seen_binding_features: set[tuple[str, str]] = set()
    binding_by_source: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in binding_rows:
        source = _text(row, "source_accession")
        member = _text(row, "objective_member")
        if source not in summaries:
            blockers.append(f"binding_unknown_source:{source}")
            continue
        if member not in expected_members:
            blockers.append(f"binding_unknown_objective_member:{source}:{member}")
        if _text(row, "platform_accession") != str(summaries[source]["platform_accession"]):
            blockers.append(f"binding_platform:{source}:{member}")
        if _text(row, "species") != str(summaries[source]["species"]):
            blockers.append(f"binding_species:{source}:{member}")
        if _text(row, "objective_uid") != OBJECTIVE_UID:
            blockers.append(f"binding_objective_uid:{source}:{member}")
        if _text(row, "component_uid") != component_by_member.get(member, ""):
            blockers.append(f"binding_component_uid:{source}:{member}")
        if _text(row, "aggregation_rule") != AGGREGATION_RULE:
            blockers.append(f"binding_aggregation_rule:{source}:{member}")
        if _text(row, "objective_selection_outcome_blind") != "True":
            blockers.append(f"binding_outcome_blind:{source}:{member}")
        try:
            feature_count = int(_text(row, "feature_count"))
        except ValueError:
            feature_count = -1
            blockers.append(f"binding_feature_count:{source}:{member}")
        try:
            feature_ids = json.loads(_text(row, "feature_ids"))
        except json.JSONDecodeError:
            feature_ids = None
        if not isinstance(feature_ids, list) or any(not isinstance(value, str) or not value for value in feature_ids):
            blockers.append(f"binding_feature_ids:{source}:{member}")
            feature_ids = []
        if len(set(feature_ids)) != len(feature_ids) or feature_count != len(feature_ids):
            blockers.append(f"binding_feature_count_ids:{source}:{member}")
        status = _text(row, "mapping_status")
        if status not in {"bound", "missing_on_platform", "contradictory_annotations"}:
            blockers.append(f"binding_mapping_status:{source}:{member}")
        canonical_id = _text(row, "canonical_entity_id")
        namespace = _text(row, "canonical_namespace")
        if status == "missing_on_platform" and (feature_count != 0 or feature_ids or canonical_id or namespace):
            blockers.append(f"binding_missing_semantics:{source}:{member}")
        if status == "bound" and (feature_count <= 0 or not feature_ids or not canonical_id or not namespace):
            blockers.append(f"binding_bound_semantics:{source}:{member}")
        if status == "contradictory_annotations" and (feature_count <= 0 or not feature_ids or canonical_id or namespace):
            blockers.append(f"binding_contradictory_semantics:{source}:{member}")
        for feature_id in feature_ids:
            key = (source, feature_id)
            if key in seen_binding_features:
                blockers.append(f"binding_feature_reused:{source}:{feature_id}")
            seen_binding_features.add(key)
            feature = features_by_source[source].get(feature_id)
            if feature is None or _text(feature, "objective_member") != member:
                blockers.append(f"binding_feature_reference:{source}:{member}:{feature_id}")
        binding_by_source[source][member] = row
        referenced = [features_by_source[source].get(feature_id) for feature_id in feature_ids]
        if all(feature is not None for feature in referenced):
            referenced_rows = [feature for feature in referenced if feature is not None]
            referenced_statuses = {_text(feature, "mapping_status") for feature in referenced_rows}
            referenced_identities = {
                (_text(feature, "canonical_entity_id"), _text(feature, "canonical_namespace"))
                for feature in referenced_rows
            }
            if status == "bound":
                if not referenced_statuses.issubset({"bound_numeric", "bound_symbol_fallback"}) or len(referenced_identities) != 1:
                    blockers.append(f"binding_referenced_status:{source}:{member}")
                elif next(iter(referenced_identities)) != (canonical_id, namespace):
                    blockers.append(f"binding_referenced_identity:{source}:{member}")
            elif status == "contradictory_annotations":
                all_bound = referenced_statuses.issubset({"bound_numeric", "bound_symbol_fallback"})
                all_explicitly_contradictory = referenced_statuses == {"contradictory_annotations"}
                if not all_explicitly_contradictory and (not all_bound or len(referenced_identities) <= 1):
                    blockers.append(f"binding_referenced_contradiction:{source}:{member}")

    for source, summary in summaries.items():
        source_bindings = binding_by_source[source]
        if set(source_bindings) != expected_members:
            blockers.append(f"binding_member_set:{source}")
        bound_count = sum(_text(row, "mapping_status") == "bound" for row in source_bindings.values())
        if bound_count != int(summary["objective_member_bound_count"]):
            blockers.append(f"binding_bound_count:{source}")
        missing = sorted(member for member, row in source_bindings.items() if _text(row, "mapping_status") == "missing_on_platform")
        contradictory = sorted(member for member, row in source_bindings.items() if _text(row, "mapping_status") == "contradictory_annotations")
        if missing != sorted(str(value) for value in summary["missing_objective_members"]):
            blockers.append(f"binding_missing_members:{source}")
        if contradictory != sorted(str(value) for value in summary["contradictory_objective_members"]):
            blockers.append(f"binding_contradictory_members:{source}")
    return blockers


def _compare_recomputed(
    current: Mapping[str, object],
    rebuilt: Mapping[str, object],
    keys: tuple[str, ...],
    label: str,
    blockers: list[str],
) -> None:
    for key in keys:
        if current.get(key) != rebuilt.get(key):
            blockers.append(f"recompute:{label}:{key}")


def _validate_artifact_bindings(payload: Mapping[str, object], blockers: list[str], label: str) -> None:
    bindings = payload.get("output_bindings", {})
    if not isinstance(bindings, Mapping):
        return
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or "sha256" not in binding:
            continue
        path_value = binding.get("path")
        if not isinstance(path_value, str):
            blockers.append(f"artifact_path:{label}:{name}")
            continue
        path = (REPO_ROOT / path_value).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            blockers.append(f"artifact_path_escape:{label}:{name}")
            continue
        if not path.is_file():
            blockers.append(f"artifact_missing:{label}:{name}")
            continue
        if _sha256(path) != binding.get("sha256"):
            blockers.append(f"artifact_hash:{label}:{name}")
        if path.stat().st_size != binding.get("bytes"):
            blockers.append(f"artifact_bytes:{label}:{name}")


def validate(*, recompute: bool = False) -> dict[str, object]:
    blockers: list[str] = []
    paths = [
        PLATFORM_AUDIT_PATH,
        OBJECTIVE_AUDIT_PATH,
        MATERIALIZATION_V3_PATH,
        DOMAIN_V3_PATH,
        SPLIT_V3_PATH,
        GAP_MAP_PATH,
        ACQUISITION_QUEUE_PATH,
        PLATFORM_MANIFEST_PATH,
        OBJECTIVE_BINDING_PATH,
        OBSERVATION_V3_PATH,
        CANDIDATE_V3_PATH,
        QUALIFICATION_V3_PATH,
    ]
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in paths if not path.is_file()]
    if missing:
        return {"status": "blocked", "passed": False, "blockers": [f"missing:{path}" for path in missing]}

    platform_audit = _json(PLATFORM_AUDIT_PATH)
    objective_audit = _json(OBJECTIVE_AUDIT_PATH)
    materialization = _json(MATERIALIZATION_V3_PATH)
    domain = _json(DOMAIN_V3_PATH)
    split = _json(SPLIT_V3_PATH)
    gap_map = _json(GAP_MAP_PATH)
    acquisition_queue = _json(ACQUISITION_QUEUE_PATH)
    _validate_schema(platform_audit, PLATFORM_SCHEMA_PATH, blockers, "platform")
    _validate_schema(objective_audit, OBJECTIVE_SCHEMA_PATH, blockers, "objective")
    _validate_schema(domain, DOMAIN_SCHEMA_PATH, blockers, "domain")
    queue_schema_path = REPO_ROOT / "Libs/configs/schema/therapeutic_targeted_acquisition_queue.schema.json"
    _validate_schema(acquisition_queue, queue_schema_path, blockers, "acquisition_queue")

    feature_rows = _tsv(PLATFORM_MANIFEST_PATH)
    binding_rows = _tsv(OBJECTIVE_BINDING_PATH)
    observations = _tsv(OBSERVATION_V3_PATH)
    candidates = _tsv(CANDIDATE_V3_PATH)
    qualification = _tsv(QUALIFICATION_V3_PATH)
    _validate_annotation_files(platform_audit, blockers)
    blockers.extend(_validate_semantic_rows(feature_rows, binding_rows, platform_audit, objective_audit))
    _validate_artifact_bindings(platform_audit, blockers, "platform_audit")
    _validate_artifact_bindings(objective_audit, blockers, "objective_audit")
    _validate_artifact_bindings(materialization, blockers, "materialization")
    if len(feature_rows) != sum(int(row["feature_count"]) for row in platform_audit["platform_summaries"]):
        blockers.append("feature_count")
    if len({(row.get("source_accession"), row.get("feature_id")) for row in feature_rows}) != len(feature_rows):
        blockers.append("duplicate_feature_identity")
    if len(binding_rows) != len(platform_audit["selected_sources"]) * int(objective_audit["objective_member_count"]):
        blockers.append("objective_binding_row_count")
    for row in feature_rows + binding_rows + observations + candidates + qualification:
        if row.get("evidence_role") != "development_only":
            blockers.append("evidence_role")
        if row.get("publication_claim_eligible") != "False":
            blockers.append("publication_claim_boundary")
        if row.get("protected_holdout_access") != "0":
            blockers.append("protected_holdout_access")
    for summary in platform_audit["platform_summaries"]:
        source = str(summary["source_accession"])
        source_rows = [row for row in feature_rows if row.get("source_accession") == source]
        if not source_rows:
            blockers.append(f"source_feature_rows:{source}")
        if any(row.get("annotation_file_sha256") != summary["annotation_content_sha256"] for row in source_rows):
            blockers.append(f"annotation_hash:{source}")
        binding_source_rows = [row for row in binding_rows if row.get("source_accession") == source]
        if len(binding_source_rows) != int(summary["objective_member_count"]):
            blockers.append(f"source_objective_rows:{source}")
        status = str(summary["status"])
        statuses = {row.get("mapping_status") for row in binding_source_rows}
        if status == "rejected_incomplete_objective_coverage" and "missing_on_platform" not in statuses and "contradictory_annotations" not in statuses:
            blockers.append(f"rejection_without_reason:{source}")
    if platform_audit.get("annotation_identity_bound") is not True:
        blockers.append("annotation_identity_bound")
    if platform_audit.get("ambiguous_mapping_policy_frozen") is not True:
        blockers.append("ambiguous_mapping_policy")
    if platform_audit.get("outcome_driven_mapping_choices") is not False:
        blockers.append("outcome_driven_mapping")
    if objective_audit.get("objective_namespace_bound") is not False:
        # Current materialized sources are expected to be rejected because the
        # frozen objective is incomplete on both platforms.
        if any(str(row.get("status")) != "bound" for row in platform_audit["platform_summaries"]):
            blockers.append("objective_bound_status_drift")
    if domain.get("candidate_set_count") != 0 or domain.get("qualified_decision_sensitive_sets") != 0:
        blockers.append("candidate_qualification_gate")
    if domain.get("therapeutic_domain", {}).get("status") != domain.get("status"):
        blockers.append("therapeutic_domain_status")
    if domain.get("therapeutic_domain", {}).get("qualified_candidate_set_count") != 0:
        blockers.append("therapeutic_domain_qualified_count")
    if domain.get("benchmark_created") is not False:
        blockers.append("benchmark_created")
    for key in ("world_action_authorized", "gpu_campaign_authorized", "static_reference_authorized", "tri_authorized", "agent_in_twin_authorized", "coevolution_authorized"):
        if domain.get(key) is not False:
            blockers.append(f"authorization:{key}")
    for key in ("world_action", "tri", "agent_in_twin", "coevolution"):
        status_payload = domain.get(key)
        if not isinstance(status_payload, Mapping) or status_payload.get("status") != "not_authorized" or status_payload.get("reason") != "therapeutic_decision_domain_blocked":
            blockers.append(f"status_object:{key}")
    if split.get("split_files_created") is not False:
        blockers.append("split_files_created")
    if gap_map.get("terminal_gate") != "BLOCKED_INSUFFICIENT_DECISION_SENSITIVE_THERAPEUTIC_OUTCOMES":
        blockers.append("terminal_gate")
    if acquisition_queue.get("selection_basis") != "metadata_only":
        blockers.append("acquisition_queue_selection_basis")
    if acquisition_queue.get("eligible_remaining_multi_action_source_count") != 0:
        blockers.append("acquisition_queue_remaining_multi_action_sources")
    if acquisition_queue.get("ordinary_transcriptomic_expansion_authorized") is not False:
        blockers.append("acquisition_queue_authorization")
    if len(observations) != int(materialization["observation_count"]):
        blockers.append("observation_count")
    action_uids = {row.get("world_action_uid") for row in observations}
    therapeutic_action_uids = {
        row.get("world_action_uid")
        for row in observations
        if row.get("action_is_therapeutic") == "True"
    }
    if int(materialization.get("action_count", -1)) != len(action_uids):
        blockers.append("action_count")
    if int(domain.get("action_count", -1)) != len(action_uids):
        blockers.append("domain_action_count")
    if int(materialization.get("therapeutic_action_count", -1)) != len(therapeutic_action_uids):
        blockers.append("therapeutic_action_count")
    if int(domain.get("therapeutic_action_count", -1)) != len(therapeutic_action_uids):
        blockers.append("domain_therapeutic_action_count")
    if len(candidates) != int(materialization["structural_candidate_set_count"]):
        blockers.append("structural_candidate_count")
    if len(qualification) != len(platform_audit["selected_sources"]):
        blockers.append("qualification_rows")
    # The correction for E-GEOD-41345 remains a hard semantic invariant.
    for row in observations:
        if row.get("source_accession") != "E-GEOD-41345":
            continue
        if row.get("context_label") == "mTBI" and row.get("control_label") != "mTBI vehicle":
            blockers.append("E-GEOD-41345:mTBI_control_semantics")
        if row.get("context_label") == "sham" and row.get("control_label") != "sham vehicle":
            blockers.append("E-GEOD-41345:sham_control_semantics")
        if row.get("action_label") == "no_intervention" and row.get("action_is_therapeutic") == "True":
            blockers.append("E-GEOD-41345:baseline_marked_therapeutic")
    if recompute and not blockers:
        rebuilt = build(write_output=False)
        _compare_recomputed(
            platform_audit,
            rebuilt["platform_audit"],
            (
                "selected_sources",
                "platform_summaries",
                "annotation_identity_bound",
                "ambiguous_mapping_policy_frozen",
                "aggregation_rule",
                "outcome_driven_mapping_choices",
                "source_bindings",
                "output_bindings",
            ),
            "platform_audit",
            blockers,
        )
        _compare_recomputed(
            objective_audit,
            rebuilt["objective_audit"],
            (
                "objective_uid",
                "objective_namespace",
                "objective_freeze_status",
                "objective_member_count",
                "components",
                "source_bindings",
                "objective_namespace_bound",
                "outcome_driven_mapping_choices",
                "ambiguous_mapping_policy_frozen",
                "aggregation_rule",
                "output_bindings",
            ),
            "objective_audit",
            blockers,
        )
        _compare_recomputed(
            materialization,
            rebuilt["materialization"],
            (
                "status",
                "selected_sources",
                "source_count",
                "observation_count",
                "action_count",
                "therapeutic_action_count",
                "structural_candidate_set_count",
                "candidate_set_count",
                "qualified_candidate_set_count",
                "objective_namespace_status",
                "source_summaries",
            ),
            "materialization",
            blockers,
        )
        _compare_recomputed(
            domain,
            rebuilt["domain"],
            (
                "status",
                "therapeutic_domain",
                "semantic_binding_status",
                "source_count",
                "observation_count",
                "action_count",
                "therapeutic_action_count",
                "matched_control_context_count",
                "structural_multi_action_context_count",
                "candidate_set_count",
                "qualified_decision_sensitive_sets",
                "objective_namespace_status",
                "benchmark_created",
                "world_action_authorized",
                "gpu_campaign_authorized",
                "static_reference_authorized",
                "tri_authorized",
                "agent_in_twin_authorized",
                "coevolution_authorized",
                "world_action",
                "tri",
                "agent_in_twin",
                "coevolution",
            ),
            "domain",
            blockers,
        )
        _compare_recomputed(split, rebuilt["split"], tuple(split.keys()), "split", blockers)
        _compare_recomputed(gap_map, rebuilt["gap_map"], ("terminal_gate", "reason", "source_bindings", "qualified_candidate_set_count", "candidate_set_count", "eligible_remaining_multi_action_source_count", "non_decision_sensitive_remaining_source_count", "ordinary_transcriptomic_expansion_authorized", "recommended_next_evidence"), "gap_map", blockers)
        _compare_recomputed(acquisition_queue, rebuilt["acquisition_queue"], ("status", "selection_basis", "rows", "eligible_remaining_multi_action_source_count", "non_decision_sensitive_remaining_source_count", "selected_source_binding_status", "ordinary_transcriptomic_expansion_authorized"), "acquisition_queue", blockers)
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-therapeutic-semantic-domain-validation-v3",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "source_count": domain.get("source_count"),
        "observation_count": domain.get("observation_count"),
        "candidate_set_count": domain.get("candidate_set_count"),
        "qualified_decision_sensitive_sets": domain.get("qualified_decision_sensitive_sets"),
        "protected_holdout_access": domain.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
