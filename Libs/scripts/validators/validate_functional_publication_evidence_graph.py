"""Validate the bounded publication-centered functional evidence graph."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
GRAPH_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_publication_evidence_graph.schema.json"
VALIDATION_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_publication_evidence_graph_validation.schema.json"

_ARTIFACT_STATUSES = {
    "FUNCTIONAL_DATA_RECOVERED",
    "FUNCTIONAL_DATA_PRESENT_BUT_NON_MACHINE_READABLE",
    "FUNCTIONAL_DATA_PRESENT_BUT_INSUFFICIENT_REPLICATES",
    "FUNCTIONAL_DATA_PRESENT_BUT_UNCERTAINTY_UNAVAILABLE",
}
_ACCESS_STATUSES = {"access_blocked", "download_limit", "http_error"}


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return [dict(row) for row in reader]


def _core_graph(audit: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Project the audit wrapper onto the strict graph contract."""
    return {key: audit[key] for key in schema.get("properties", {}) if key in audit}


def validate(*, write_output: bool = True) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        config = load_config(CONFIG_PATH)
        recovery = config["functional_therapeutic_evidence"]["source_discovery"]["publication_recovery"]
        audit_path = _repo_path(str(recovery["recovery_audit"]))
        graph_path = _repo_path(str(recovery["evidence_graph"]))
        metadata_path = _repo_path(str(recovery["seed_registry"]))
        validation_path = _repo_path(str(recovery["validation_audit"]))
        audit = _json(audit_path)
        schema = _json(GRAPH_SCHEMA_PATH)
        validation_schema = _json(VALIDATION_SCHEMA_PATH)
        metadata_rows = _tsv(metadata_path)
        graph_rows = _tsv(graph_path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        report = {"status": "blocked", "passed": False, "blocker_count": 1, "blockers": [f"artifact_read:{exc}"]}
        return report

    for error in Draft202012Validator(schema).iter_errors(_core_graph(audit, schema)):
        blockers.append(f"graph_schema:{list(error.absolute_path)}:{error.message}")

    expected_sources = {row.get("source_uid") for row in metadata_rows}
    records = audit.get("records")
    if len(metadata_rows) != int(recovery["max_seed_sources"]):
        blockers.append("seed_registry_count")
    if not isinstance(records, list) or len(records) != len(expected_sources):
        blockers.append("record_count")
        records = records if isinstance(records, list) else []
    record_sources = [record.get("source_uid") for record in records if isinstance(record, dict)]
    if len(record_sources) != len(set(record_sources)):
        blockers.append("duplicate_record_source_uid")
    if set(record_sources) != expected_sources:
        blockers.append("record_source_coverage")

    expected_nodes: dict[str, dict[str, Any]] = {}
    access_blocked_nodes = False
    for record in records:
        source_uid = record.get("source_uid")
        nodes = record.get("nodes", [])
        seen_nodes: set[str] = set()
        accession_nodes = 0
        for node in nodes:
            node_uid = node.get("node_uid")
            if node_uid in seen_nodes or node_uid in expected_nodes:
                blockers.append(f"duplicate_node_uid:{node_uid}")
            seen_nodes.add(node_uid)
            expected_nodes[node_uid] = {**node, "source_uid": source_uid, "study_accession": record.get("study_accession")}
            if node.get("node_type") == "source_accession":
                accession_nodes += 1
            if node.get("access_status") in _ACCESS_STATUSES:
                access_blocked_nodes = True
        if accession_nodes != 1:
            blockers.append(f"accession_node_count:{source_uid}")

    graph_sources = {row.get("source_uid") for row in graph_rows}
    if graph_sources != expected_sources:
        blockers.append("tsv_source_coverage")
    if len(graph_rows) != len(expected_nodes):
        blockers.append("tsv_node_row_count")
    for row in graph_rows:
        node_uid = row.get("node_uid")
        node = expected_nodes.get(node_uid)
        if node is None:
            blockers.append(f"tsv_unknown_node:{node_uid}")
            continue
        for field in ("source_uid", "study_accession", "node_type", "relationship", "identifier", "url", "access_status"):
            expected = "" if node.get(field) is None else str(node.get(field))
            if row.get(field, "") != expected:
                blockers.append(f"tsv_node_binding:{node_uid}:{field}")

    valid_statuses = set(recovery["statuses"])
    for record in records:
        source_uid = record.get("source_uid")
        status = record.get("status")
        if status not in valid_statuses:
            blockers.append(f"status:{source_uid}")
        record_access_blocked = any(
            node.get("access_status") in _ACCESS_STATUSES for node in record.get("nodes", [])
        ) or record.get("publication_resolution") == "access_blocked" or any(
            record.get(field) == "access_blocked"
            for field in ("machine_readable_status", "replicate_identity_status", "uncertainty_status", "shared_control_status", "therapeutic_arm_status")
        )
        if status == "ACCESS_BLOCKED" and not record_access_blocked:
            blockers.append(f"access_status_without_evidence:{source_uid}")
        if status != "ACCESS_BLOCKED" and record.get("publication_resolution") == "access_blocked":
            blockers.append(f"publication_access_status_mismatch:{source_uid}")

    if audit.get("status") not in {"completed_bounded_recovery", "completed_with_access_blocks", "blocked_network_access"}:
        blockers.append("campaign_status")
    if access_blocked_nodes and audit.get("status") == "completed_bounded_recovery":
        blockers.append("campaign_access_status")
    for key in ("outcome_values_used_for_priority", "model_predictions_inspected", "figure_digitization_admitted", "publication_comparison_open"):
        if audit.get(key) is not False:
            blockers.append(key)
    if audit.get("protected_holdout_access") != 0:
        blockers.append("protected_holdout_access")
    if audit.get("admission_eligible_source_count") != 0:
        blockers.append("formal_admission_not_fail_closed")
    if audit.get("legal_candidate_set_count") != 0:
        blockers.append("legal_candidate_sets_not_fail_closed")
    if audit.get("qualified_candidate_set_count") != 0:
        blockers.append("qualified_candidate_sets_not_fail_closed")
    if audit.get("terminal_state") != "BLOCKED_REQUIRES_EXTERNAL_MULTI_ARM_FUNCTIONAL_OUTCOMES":
        blockers.append("terminal_state")
    if audit.get("next_route") != "prepare_external_multi_arm_functional_outcome_intake":
        blockers.append("next_route")
    expected_artifacts = sum(record.get("status") in _ARTIFACT_STATUSES for record in records)
    if audit.get("functional_artifact_present_count") != expected_artifacts:
        blockers.append("functional_artifact_present_count")
    expected_machine = sum(record.get("machine_readable_status") == "confirmed" for record in records)
    if audit.get("machine_readable_candidate_count") != expected_machine:
        blockers.append("machine_readable_candidate_count")

    unique = sorted(set(blockers))
    report = {
        "schema_version": "neuronsifter-functional-publication-evidence-graph-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "source_count": len(expected_sources),
        "publication_resolved_count": audit.get("publication_resolved_count"),
        "functional_artifact_present_count": audit.get("functional_artifact_present_count"),
        "machine_readable_candidate_count": audit.get("machine_readable_candidate_count"),
        "admission_eligible_source_count": audit.get("admission_eligible_source_count"),
        "legal_candidate_set_count": audit.get("legal_candidate_set_count"),
        "qualified_candidate_set_count": audit.get("qualified_candidate_set_count"),
        "protected_holdout_access": audit.get("protected_holdout_access"),
    }
    report_schema_errors = list(Draft202012Validator(validation_schema).iter_errors(report))
    if report_schema_errors:
        unique_report_errors = [f"validation_report_schema:{list(error.absolute_path)}:{error.message}" for error in report_schema_errors]
        report["blockers"] = sorted(set([*report["blockers"], *unique_report_errors]))
        report["blocker_count"] = len(report["blockers"])
        report["status"] = "blocked"
        report["passed"] = False
    if write_output:
        write_json(validation_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Do not write the validation audit")
    args = parser.parse_args()
    result = validate(write_output=not args.no_write)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
