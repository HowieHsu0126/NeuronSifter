"""Validate BASELINE_FIRST's generated, fail-closed pre-execution audits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_first_preflight import (  # noqa: E402
    DEFAULT_CONFIG,
    build,
)


JSON_OUTPUTS = {
    "full_suite": "baseline_first_full_suite_execution.json",
    "runtime_dependencies": "baseline_first_runtime_dependency_manifest.json",
    "inventory": "baseline_first_test_inventory.json",
    "inventory_diff": "baseline_first_test_inventory_diff.json",
    "validation": "baseline_first_validation.json",
    "gate": "baseline_first_gate.json",
    "handoff": "baseline_first_mainline_handoff.json",
    "baseline_first_scoped_authorization": "baseline_first_claim_scoped_authorization.json",
}


def _completed_execution_is_valid(payload: Mapping[str, Any]) -> bool:
    """Accept a real BASELINE_FIRST execution record without treating it as a preflight snapshot."""
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        return False
    required = {
        "schema_version": "neuronsifter-baseline_first-full-suite-execution-v2",
        "terminal_execution": "completed",
        "test_inventory_bound": True,
        "unknown_failures": 0,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        return False
    counts = ("full_suite_failures", "full_suite_errors", "mandatory_skips")
    if any(not isinstance(payload.get(key), int) or payload[key] < 0 for key in counts):
        return False
    expected_green = bool(payload.get("exit_code") == 0 and all(payload[key] == 0 for key in counts))
    if payload.get("full_suite_green") is not expected_green:
        return False
    return bool(
        candidate.get("git_backed") is True
        and candidate.get("live_worktree_clean") is True
        and candidate.get("clean_checkout_ready") is True
        and candidate.get("node_local_staging_ready") is True
        and candidate.get("candidate_commit")
        and candidate.get("candidate_tree")
        and isinstance(payload.get("run_uid"), str)
        and isinstance(payload.get("artifacts"), Mapping)
    )


def _serialize(rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]


def validate(
    config: str | Path = DEFAULT_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
    audit_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    audits = (root / "Output/audits") if audit_root is None else audit_root.resolve()
    blockers: list[dict[str, str]] = []
    try:
        expected = build(config, repo_root=root)
    except Exception as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "baseline_first_contract_invalid", "message": str(exc)}]}
    for key, filename in JSON_OUTPUTS.items():
        path = audits / filename
        if not path.is_file():
            blockers.append({"code": "missing_generated_artifact", "message": filename})
            continue
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append({"code": "invalid_generated_artifact", "message": filename})
            continue
        if key == "full_suite" and _completed_execution_is_valid(actual):
            continue
        if key in {"runtime_dependencies", "inventory", "inventory_diff"} and (audits / "baseline_first_full_suite_execution.json").is_file():
            execution = json.loads((audits / "baseline_first_full_suite_execution.json").read_text(encoding="utf-8"))
            if _completed_execution_is_valid(execution) and isinstance(actual, dict) and actual.get("run_uid") == execution.get("run_uid"):
                continue
        if actual != expected[key]:
            blockers.append({"code": "stale_or_tampered_generated_artifact", "message": filename})
    tsv_path = audits / "baseline_first_full_suite_failure_decomposition.tsv"
    if not tsv_path.is_file():
        blockers.append({"code": "missing_generated_artifact", "message": tsv_path.name})
    else:
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            actual_rows = list(csv.DictReader(handle, delimiter="\t"))
        if actual_rows != _serialize(expected["decomposition"]):
            blockers.append({"code": "stale_or_tampered_generated_artifact", "message": tsv_path.name})
    gate = expected["gate"]
    if gate["baseline_first_c1_c2_challenge_resolved"]:
        blockers.append({"code": "unsafe_gate_promotion", "message": "Pre-execution BASELINE_FIRST builder cannot resolve the empirical gate."})
    if expected["validation"]["full_suite_green"]:
        blockers.append({"code": "unsafe_full_suite_promotion", "message": "A pre-execution artifact cannot report a green full suite."})
    if expected["handoff"]["headline_training_authorized"]:
        blockers.append({"code": "unsafe_headline_promotion", "message": "BASELINE_FIRST preflight must not authorize headline training."})
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "gate_status": gate["status"],
        "gate_resolved": gate["baseline_first_c1_c2_challenge_resolved"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BASELINE_FIRST pre-execution audit artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-root", default="Output/audits")
    args = parser.parse_args()
    report = validate(args.config, audit_root=REPO_ROOT / args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
