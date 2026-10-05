"""Record source-bound validation evidence for the decision-utility diagnosis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.record_static_decision_loop_validation import (  # noqa: E402
    identity,
    parse_junit,
    write_json,
)


AUDIT_PATH = REPO_ROOT / "Output/audits/round26_full_suite_execution.json"
ACCOUNTING_PATH = REPO_ROOT / "Output/audits/round26_full_suite_slurm_accounting.json"
TEST_INVENTORY_PATH = REPO_ROOT / "Output/audits/round26_test_inventory.json"

SOURCE_FILES = (
    "Libs/configs/neuronsifter/static_decision_loop.yaml",
    "Libs/configs/neuronsifter/round24b_posterior_bound_evaluation.yaml",
    "Libs/models/modules/round24b_posterior_adapter.py",
    "Libs/models/modules/round24b_typed_actions.py",
    "Libs/models/modules/static_information_policy.py",
    "Libs/scripts/experiments/run_round24b_posterior_bound_campaign.py",
    "Libs/scripts/experiments/run_static_decision_loop_campaign.py",
    "Libs/scripts/experiments/diagnose_decision_utility_mechanism.py",
    "Libs/scripts/validators/validate_decision_utility_mechanism.py",
    "Libs/scripts/validators/record_decision_utility_validation.py",
    "Libs/tests/test_decision_utility_mechanism.py",
)

EVALUATION_FILES = (
    "Output/audits/round26_diagnostic_unit_manifest.tsv",
    "Output/audits/round26_round25_negative_result_freeze.json",
    "Output/paper_reports/round26_s_u_policy_equivalence.tsv",
    "Output/audits/round26_utility_decomposition.json",
    "Output/paper_reports/round26_decision_sensitivity.tsv",
    "Output/paper_reports/round26_tri_flattening.tsv",
    "Output/audits/round26_decision_transmission_analysis.json",
    "Output/paper_reports/round26_information_action_value.tsv",
    "Output/audits/round26_decision_sensitive_tool_coverage.json",
    "Output/audits/round26_mechanism_root_cause.json",
    "Output/audits/round26_information_action_gate.json",
    "Output/audits/round26_canonical_architecture.json",
    "Output/audits/round26_next_scientific_route.json",
)

BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_bindings() -> list[dict[str, Any]]:
    path = REPO_ROOT / "Output/audits/round25_u0_canonical_static_twin.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for role, key in (
        ("scientific_belief_encoder", "scientific_belief_encoder"),
        ("world_action_capability", "world_action_capability"),
    ):
        binding = record[key]
        checkpoint = REPO_ROOT / str(binding["path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows.append(
            {
                "role": role,
                "path": str(binding["path"]),
                "sha256": str(binding["sha256"]),
                "bytes": int(checkpoint.stat().st_size),
                "identity_source": str(path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def record(
    *,
    junit: Path,
    stdout: Path,
    stderr: Path,
    pytest_exit_code: int,
    job_id: str,
    job_name: str,
    node: str,
    slurm_state: str,
    slurm_elapsed: str,
    slurm_exit_code: str,
) -> dict[str, Any]:
    if AUDIT_PATH.exists() or ACCOUNTING_PATH.exists():
        raise FileExistsError("Decision-utility validation evidence already exists")
    junit = junit.resolve()
    stdout = stdout.resolve()
    stderr = stderr.resolve()
    counts, test_inventory = parse_junit(junit)
    source_files = [identity(REPO_ROOT / path) for path in SOURCE_FILES]
    evaluation_files = [identity(REPO_ROOT / path) for path in EVALUATION_FILES]
    accounting = {
        "schema_version": "neuronsifter-decision-utility-slurm-accounting-v1",
        "job_id": str(job_id),
        "job_name": str(job_name),
        "state": str(slurm_state),
        "elapsed": str(slurm_elapsed),
        "exit_code": str(slurm_exit_code),
        "node": str(node),
        **BOUNDARY,
    }
    full_suite_green = bool(
        pytest_exit_code == 0
        and slurm_state == "COMPLETED"
        and slurm_exit_code == "0:0"
        and int(counts["tests_failed"]) == 0
        and int(counts["tests_errored"]) == 0
        and int(counts["tests_skipped"]) == 0
        and int(counts["tests_collected"]) == int(counts["tests_passed"])
    )
    write_json(TEST_INVENTORY_PATH, test_inventory)
    payload = {
        "schema_version": "neuronsifter-decision-utility-full-suite-execution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if full_suite_green else "failed",
        "full_suite_green": full_suite_green,
        "command": "python -m pytest -W ignore::UserWarning Libs/tests -q --junitxml=<node-local-path>",
        "pytest_exit_code": int(pytest_exit_code),
        **counts,
        "junit": identity(junit),
        "stdout": identity(stdout),
        "stderr": identity(stderr),
        "test_inventory": identity(TEST_INVENTORY_PATH),
        "source_files": source_files,
        "source_file_count": len(source_files),
        "candidate_source_manifest_sha256": _manifest_digest(source_files),
        "evaluation_files": evaluation_files,
        "evaluation_file_count": len(evaluation_files),
        "evaluation_manifest_sha256": _manifest_digest(evaluation_files),
        "checkpoints": _checkpoint_bindings(),
        "candidate_identity": {
            "kind": "source_manifest_sha256",
            "value": _manifest_digest(source_files),
            "candidate_commit_available": False,
        },
        "slurm": accounting,
        **BOUNDARY,
    }
    write_json(ACCOUNTING_PATH, accounting)
    write_json(AUDIT_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--pytest-exit-code", required=True, type=int)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--slurm-state", required=True)
    parser.add_argument("--slurm-elapsed", required=True)
    parser.add_argument("--slurm-exit-code", required=True)
    args = parser.parse_args()
    payload = record(
        junit=args.junit,
        stdout=args.stdout,
        stderr=args.stderr,
        pytest_exit_code=args.pytest_exit_code,
        job_id=args.job_id,
        job_name=args.job_name,
        node=args.node,
        slurm_state=args.slurm_state,
        slurm_elapsed=args.slurm_elapsed,
        slurm_exit_code=args.slurm_exit_code,
    )
    print(json.dumps({
        "status": payload["status"],
        "tests_collected": payload["tests_collected"],
        "tests_passed": payload["tests_passed"],
        "job_id": payload["slurm"]["job_id"],
    }, sort_keys=True))
    return 0 if payload["full_suite_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
