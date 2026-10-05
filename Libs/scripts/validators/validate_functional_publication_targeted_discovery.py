"""Validate the single bounded metadata-only functional publication wave."""

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
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_publication_targeted_discovery.schema.json"
VALIDATION_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_publication_targeted_discovery_validation.schema.json"


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


def validate(*, write_output: bool = True) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        config = load_config(CONFIG_PATH)
        discovery = config["functional_therapeutic_evidence"]["source_discovery"]["targeted_discovery"]
        audit_path = _repo_path(str(discovery["outputs"]["audit"]))
        manifest_path = _repo_path(str(discovery["outputs"]["manifest"]))
        validation_path = _repo_path(str(discovery["outputs"]["validation_audit"]))
        audit = _json(audit_path)
        schema = _json(SCHEMA_PATH)
        validation_schema = _json(VALIDATION_SCHEMA_PATH)
        manifest_rows = _tsv(manifest_path)
        seed_rows = _tsv(_repo_path(str(config["functional_therapeutic_evidence"]["source_discovery"]["metadata_registry"])))
        publication_audit = _json(REPO_ROOT / "Output/audits/functional_publication_evidence_recovery.json")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blocker_count": 1, "blockers": [f"artifact_read:{exc}"]}

    core = {key: audit[key] for key in schema.get("properties", {}) if key in audit}
    for error in Draft202012Validator(schema).iter_errors(core):
        blockers.append(f"audit_schema:{list(error.absolute_path)}:{error.message}")
    if audit.get("query_count") != len(audit.get("queries", [])):
        blockers.append("query_count")
    if audit.get("result_count") != sum(int(query.get("result_count", -1)) for query in audit.get("queries", [])):
        blockers.append("result_count")
    records = audit.get("records", [])
    lead_uids = [record.get("lead_uid") for record in records]
    if len(lead_uids) != len(set(lead_uids)):
        blockers.append("duplicate_lead_uid")
    if audit.get("novel_lead_count") != sum(record.get("metadata_design_status") == "candidate_multi_arm_metadata" for record in records):
        blockers.append("novel_lead_count")
    if len(manifest_rows) != len(records):
        blockers.append("manifest_record_count")
    manifest_by_uid = {row.get("lead_uid"): row for row in manifest_rows}
    for record in records:
        row = manifest_by_uid.get(record.get("lead_uid"))
        if row is None:
            blockers.append(f"manifest_missing:{record.get('lead_uid')}")
            continue
        if row.get("query_uids", "").split(";") != record.get("query_uids"):
            blockers.append(f"manifest_query_binding:{record.get('lead_uid')}")
        for field in ("identifier", "identifier_type", "title", "metadata_design_status", "expected_action_mode", "artifact_status", "functional_values_admitted"):
            if row.get(field, "") != str(record.get(field, "")):
                blockers.append(f"manifest_binding:{record.get('lead_uid')}:{field}")

    excluded: set[str] = set()
    for row in seed_rows:
        for value in str(row.get("publication_uid", "")).split(";"):
            if value.startswith("pubmed:"):
                excluded.add(f"pmid:{value.split(':', 1)[1]}")
            elif value.startswith("doi:"):
                excluded.add(f"doi:{value.split(':', 1)[1].lower()}")
    for record in publication_audit.get("records", []):
        publication_uid = str(record.get("publication_uid") or "")
        doi = str(record.get("doi") or "")
        if publication_uid.startswith("pubmed:"):
            excluded.add(f"pmid:{publication_uid.split(':', 1)[1]}")
        if doi:
            excluded.add(f"doi:{doi.lower()}")
    if any(uid in excluded for uid in lead_uids):
        blockers.append("seed_leakage")
    for key in ("outcome_values_used_for_priority", "abstract_values_inspected", "model_predictions_inspected", "publication_comparison_open"):
        if audit.get(key) is not False:
            blockers.append(key)
    if audit.get("protected_holdout_access") != 0:
        blockers.append("protected_holdout_access")
    if audit.get("admitted_source_count") != 0 or audit.get("legal_candidate_set_count") != 0 or audit.get("qualified_candidate_set_count") != 0:
        blockers.append("downstream_admission")
    if audit.get("novel_lead_count") == 0:
        query_access_blocked = any(query.get("access_status") == "access_blocked" for query in audit.get("queries", []))
        expected_terminal = "TARGETED_DISCOVERY_ACCESS_BLOCKED" if query_access_blocked else "BLOCKED_REQUIRES_EXTERNAL_MULTI_ARM_FUNCTIONAL_OUTCOMES"
        expected_route = "repeat_bounded_metadata_query_after_access_recovers" if query_access_blocked else "prepare_external_multi_arm_functional_outcome_intake"
        if audit.get("terminal_state") != expected_terminal:
            blockers.append("terminal_state")
        if audit.get("next_route") != expected_route:
            blockers.append("next_route")

    unique = sorted(set(blockers))
    report = {
        "schema_version": "neuronsifter-functional-publication-targeted-discovery-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "query_count": audit.get("query_count"),
        "result_count": audit.get("result_count"),
        "novel_lead_count": audit.get("novel_lead_count"),
        "admitted_source_count": audit.get("admitted_source_count"),
        "legal_candidate_set_count": audit.get("legal_candidate_set_count"),
        "qualified_candidate_set_count": audit.get("qualified_candidate_set_count"),
        "protected_holdout_access": audit.get("protected_holdout_access"),
    }
    for error in Draft202012Validator(validation_schema).iter_errors(report):
        report["blockers"].append(f"validation_report_schema:{list(error.absolute_path)}:{error.message}")
    report["blockers"] = sorted(set(report["blockers"]))
    report["blocker_count"] = len(report["blockers"])
    report["status"] = "passed" if not report["blockers"] else "blocked"
    report["passed"] = not report["blockers"]
    if write_output:
        write_json(validation_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = validate(write_output=not args.no_write)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
