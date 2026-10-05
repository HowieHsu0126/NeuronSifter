from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_first_qualification import (  # noqa: E402
    DEFAULT_AUDIT_ROOT,
    DEFAULT_REGISTRY,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SCHEMA,
    build_all,
    repo_path,
)


AUDIT_JSON = {
    "freeze": "baseline_registry_freeze.json",
    "fairness": "baseline_fairness_contract.json",
    "task_compatibility": "baseline_task_compatibility.json",
    "upstream": "baseline_upstream_reproduction_summary.json",
    "task_sanity": "baseline_task_sanity.json",
    "thresholds": "neuronsifter_challenge_thresholds.json",
    "authorization": "neuronsifter_headline_training_authorization.json",
}
AUDIT_TSV = {
    "information_access": "baseline_information_access_matrix.tsv",
    "compute_budget": "baseline_compute_budget_matrix.tsv",
    "source_shortcut": "baseline_source_shortcut_diagnostics.tsv",
}
REPORT_TSV = {
    "floor_matrix": "baseline_floor_matrix.tsv",
    "result_matrix": "baseline_result_matrix.tsv",
    "resource_matrix": "baseline_resource_matrix.tsv",
    "failure_catalog": "baseline_failure_catalog.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the fail-closed NeuronSifter Baseline-First qualification artifacts.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--audit-root", default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _serialize(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]


def validate(
    registry: str | Path = DEFAULT_REGISTRY,
    schema: str | Path = DEFAULT_SCHEMA,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    try:
        expected = build_all(registry, schema)
    except Exception as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "registry_or_schema_invalid", "message": str(exc)}]}

    audits = repo_path(audit_root)
    reports = repo_path(report_root)
    for key, filename in AUDIT_JSON.items():
        path = audits / filename
        if not path.exists():
            blockers.append({"code": "missing_required_audit", "message": filename})
            continue
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blockers.append({"code": "invalid_json_audit", "message": f"{filename}: {exc}"})
            continue
        if actual != expected[key]:
            blockers.append({"code": "stale_or_tampered_audit", "message": filename})
    for key, filename in AUDIT_TSV.items():
        path = audits / filename
        if not path.exists():
            blockers.append({"code": "missing_required_audit", "message": filename})
            continue
        if _read_tsv(path) != _serialize(expected[key]):
            blockers.append({"code": "stale_or_tampered_audit", "message": filename})
    for key, filename in REPORT_TSV.items():
        path = reports / filename
        if not path.exists():
            blockers.append({"code": "missing_required_report", "message": filename})
            continue
        if _read_tsv(path) != _serialize(expected[key]):
            blockers.append({"code": "stale_or_tampered_report", "message": filename})

    for uid, artifact in expected["source_artifacts"].items():
        base = audits / "baselines" / uid
        for artifact_name in ("source_manifest", "license_audit", "environment_lock", "patch_manifest"):
            path = base / f"{artifact_name}.json"
            if not path.exists():
                blockers.append({"code": "missing_source_artifact", "message": f"{uid}/{artifact_name}.json"})
                continue
            if json.loads(path.read_text(encoding="utf-8")) != artifact[artifact_name]:
                blockers.append({"code": "stale_or_tampered_source_artifact", "message": f"{uid}/{artifact_name}.json"})

    freeze = expected["freeze"]
    authorization = expected["authorization"]
    if freeze["status"] != "frozen":
        blockers.append({"code": "registry_freeze_not_valid", "message": "Mandatory baseline coverage or class separation failed."})
    if authorization["headline_training_authorized"]:
        if not authorization["baseline_challenge_suite_resolved"]:
            blockers.append({"code": "authorization_invariant", "message": "Headline authorization requires a resolved challenge suite."})
    else:
        if authorization["baseline_challenge_suite_resolved"]:
            blockers.append({"code": "authorization_invariant", "message": "Resolved challenge suite must authorize headline training."})
        if authorization["status"] == "baseline_challenge_suite_resolved":
            blockers.append({"code": "authorization_status_invariant", "message": "Resolved status cannot remain unauthorized."})
    if expected["fairness"]["status"] != "frozen_contract_only":
        blockers.append({"code": "fairness_contract_not_frozen", "message": "Fairness contract must remain explicit."})
    if expected["thresholds"]["status"] != "not_frozen_no_strong_baseline_results" and not authorization["baseline_challenge_suite_resolved"]:
        blockers.append({"code": "threshold_premature_promotion", "message": "Challenge thresholds cannot be promoted before strong results."})
    if authorization["baseline_challenge_suite_resolved"]:
        bundle_sha256 = authorization["qualification_evidence_bundle_sha256"]
        if expected["thresholds"]["status"] != "frozen" or expected["task_sanity"]["status"] != "passed" or expected["upstream"]["status"] != "complete":
            blockers.append({"code": "resolved_report_materialization_missing", "message": "Resolved authorization requires materialized threshold, sanity, and upstream reports."})
        if any(row["result_status"] != "available" for row in expected["floor_matrix"] + expected["result_matrix"] + expected["resource_matrix"]):
            blockers.append({"code": "resolved_result_matrix_incomplete", "message": "Resolved authorization requires all result and resource rows to be available."})
        report_bundle_hashes = {
            expected["thresholds"].get("qualification_evidence_bundle_sha256"),
            expected["task_sanity"].get("qualification_evidence_bundle_sha256"),
            expected["upstream"].get("qualification_evidence_bundle_sha256"),
            *(row.get("qualification_evidence_bundle_sha256") for row in expected["floor_matrix"] + expected["result_matrix"] + expected["resource_matrix"]),
        }
        if report_bundle_hashes != {bundle_sha256}:
            blockers.append({"code": "authorization_report_bundle_mismatch", "message": "Authorization and generated reports must share one evidence bundle."})
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "authorization_status": authorization["status"],
        "baseline_challenge_suite_resolved": authorization["baseline_challenge_suite_resolved"],
        "headline_training_authorized": authorization["headline_training_authorized"],
        "baseline_count": freeze["baseline_count"],
    }


def main() -> None:
    args = parse_args()
    report = validate(args.registry, args.schema, args.audit_root, args.report_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
