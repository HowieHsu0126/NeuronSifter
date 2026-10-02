"""Bind terminal full-suite evidence to the static decision-loop handoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.record_static_decision_loop_validation import (  # noqa: E402
    CHECKPOINT_FILES,
    EVALUATION_FILES,
    SOURCE_FILES,
    identity,
    receipt_files,
)
from Libs.scripts.validators.validate_static_decision_loop import validate  # noqa: E402


AUDIT_ROOT = REPO_ROOT / "Output/audits"
TERMINAL_STATES = {
    "neuronsifter_static_decision_loop_development_resolved",
    "partial_static_decision_loop_evidence",
    "failed_static_decision_loop",
    "blocked_static_decision_loop",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_identities(
    rows: list[Mapping[str, Any]],
    label: str,
    expected_paths: tuple[str, ...],
) -> None:
    if not rows:
        raise ValueError(f"Static-loop full-suite audit lacks {label}")
    actual_paths = tuple(str(row.get("path", "")) for row in rows)
    expected = tuple(Path(path).as_posix() for path in expected_paths)
    if len(actual_paths) != len(expected) or set(actual_paths) != set(expected):
        raise ValueError(
            f"Static-loop {label} inventory does not match the declared contract"
        )
    for row in rows:
        path = REPO_ROOT / str(row.get("path", ""))
        if not path.is_file() or identity(path) != dict(row):
            raise ValueError(f"Static-loop {label} identity drift: {row}")


def _manifest_digest(rows: list[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finalize(job_id: str) -> dict[str, Any]:
    full_path = AUDIT_ROOT / "round25_full_suite_execution.json"
    accounting_path = AUDIT_ROOT / "round25_full_suite_slurm_accounting.json"
    gate_path = AUDIT_ROOT / "round25_static_architecture_gate.json"
    handoff_path = AUDIT_ROOT / "round25_development_handoff.json"
    full = _load_json(full_path)
    accounting = _load_json(accounting_path)
    gate = _load_json(gate_path)
    handoff = _load_json(handoff_path)
    if (
        full.get("schema_version")
        != "neuronsifter-static-loop-full-suite-execution-v1"
        or full.get("status") != "passed"
        or full.get("full_suite_green") is not True
        or str(full.get("slurm", {}).get("job_id", "")) != job_id
        or int(full.get("tests_failed", -1)) != 0
        or int(full.get("tests_errored", -1)) != 0
        or int(full.get("tests_skipped", -1)) != 0
    ):
        raise ValueError("Static-loop full-suite execution is not green")
    if (
        accounting.get("schema_version")
        != "neuronsifter-static-loop-slurm-terminal-accounting-v1"
        or accounting.get("status") != "terminal_completed"
        or str(accounting.get("job", {}).get("job_id", "")) != job_id
        or accounting.get("job", {}).get("state") != "COMPLETED"
        or accounting.get("job", {}).get("exit_code") != "0:0"
        or full.get("slurm", {}).get("terminal_accounting")
        != identity(accounting_path)
    ):
        raise ValueError(
            "Static-loop terminal accounting is not bound to the full suite"
        )
    expected_evaluation_files = (*EVALUATION_FILES, *receipt_files())
    _verify_identities(full["source_files"], "source files", SOURCE_FILES)
    _verify_identities(full["checkpoints"], "checkpoint files", CHECKPOINT_FILES)
    _verify_identities(
        full["evaluation_manifests"],
        "evaluation manifests",
        expected_evaluation_files,
    )
    if int(full.get("source_file_count", -1)) != len(SOURCE_FILES):
        raise ValueError("Static-loop source-file count does not match the contract")
    if int(full.get("evaluation_manifest_count", -1)) != len(expected_evaluation_files):
        raise ValueError(
            "Static-loop evaluation-manifest count does not match the contract"
        )
    if full.get("candidate_source_manifest_sha256") != _manifest_digest(
        full["source_files"]
    ):
        raise ValueError("Static-loop source-manifest digest does not match rows")
    validation = validate(require_full_suite=False)
    if validation.get("status") != "passed":
        raise ValueError("Static-loop scientific validator did not pass")
    if (
        gate.get("status") not in TERMINAL_STATES
        or handoff.get("status") != gate.get("status")
    ):
        raise ValueError("Static-loop Gate and handoff terminal states differ")
    finalized_at = datetime.now(timezone.utc).isoformat()
    gate["full_suite_pending"] = False
    gate["validation"] = {
        "static_loop_validator_green": True,
        "full_suite_green": True,
        "tests_collected": int(full["tests_collected"]),
        "tests_passed": int(full["tests_passed"]),
        "tests_failed": 0,
        "tests_errored": 0,
        "tests_skipped": 0,
        "full_suite_execution": identity(full_path),
        "slurm_terminal_accounting": identity(accounting_path),
    }
    gate["finalized_at"] = finalized_at
    handoff["full_suite_pending"] = False
    handoff["full_suite_green"] = True
    handoff["full_suite_execution"] = identity(full_path)
    handoff["slurm_terminal_accounting"] = identity(accounting_path)
    handoff["finalized_at"] = finalized_at
    _write_json(gate_path, gate)
    _write_json(handoff_path, handoff)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-suite-job-id", required=True)
    args = parser.parse_args()
    gate = finalize(args.full_suite_job_id)
    print(
        json.dumps(
            {"status": gate["status"], "full_suite_green": True},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
