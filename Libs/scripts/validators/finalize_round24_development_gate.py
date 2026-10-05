"""Bind terminal full-suite accounting and finalize the blocked Round-24 gate."""

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
AUDIT_ROOT = REPO_ROOT / "Output/audits"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_round24_adaptation import validate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_terminal_accounting(
    job_id: str, full: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    accounting_path = AUDIT_ROOT / "round24_full_suite_slurm_accounting.json"
    accounting = _load(accounting_path)
    job = accounting.get("job", {})
    if (
        accounting.get("schema_version") != "neuronsifter-round24-slurm-terminal-accounting-v1"
        or accounting.get("status") != "terminal_completed"
        or str(job.get("job_id", "")) != job_id
        or job.get("state") != "COMPLETED"
        or job.get("exit_code") != "0:0"
        or job.get("job_name") != full.get("slurm", {}).get("job_name")
        or job.get("node_list") != full.get("slurm", {}).get("node")
        or accounting.get("evidence_role") != "development_only"
        or accounting.get("publication_claim_eligible") is not False
        or int(accounting.get("protected_test_access", -1)) != 0
    ):
        raise ValueError("Round-24 terminal accounting artifact is not a matching green readback")
    expected_identity = full.get("slurm", {}).get("terminal_accounting")
    actual_identity = _identity(accounting_path)
    if (
        full.get("slurm", {}).get("terminal_accounting_status") != "terminal_completed"
        or not isinstance(expected_identity, Mapping)
        or dict(expected_identity) != actual_identity
    ):
        raise ValueError("Round-24 full-suite audit does not bind terminal accounting")
    return accounting_path, accounting


def finalize(job_id: str) -> dict[str, Any]:
    full_path = AUDIT_ROOT / "round24_full_suite_execution.json"
    gate_path = AUDIT_ROOT / "round24_vnext_development_gate.json"
    full = _load(full_path)
    if full.get("slurm", {}).get("job_id") != job_id or full.get("full_suite_green") is not True:
        raise ValueError("Round-24 full-suite audit does not match the terminal job")

    accounting_path, _ = _validate_terminal_accounting(job_id, full)

    pre_final = validate(require_final=False)
    if pre_final.get("status") != "passed" or pre_final.get("full_suite_green") is not True:
        raise ValueError("Round-24 pre-final validator did not bind the full suite")
    pre_final_path = AUDIT_ROOT / "round24_pre_final_validation.json"
    _write(pre_final_path, pre_final)

    gate = _load(gate_path)
    if gate.get("status") != "blocked_adaptive_development":
        raise ValueError("Round-24 finalizer only supports the evidence-bound blocked state")
    gate["validation"] = {
        "round24_validator_green": True,
        "full_suite_green": True,
        "pending_full_suite_execution": False,
        "tests_collected": int(full["tests_collected"]),
        "tests_passed": int(full["tests_passed"]),
        "tests_failed": int(full["tests_failed"]),
        "tests_errored": int(full["tests_errored"]),
        "tests_skipped": int(full["tests_skipped"]),
    }
    gate.setdefault("artifacts", {}).update(
        {
            "full_suite_execution": _identity(full_path),
            "pre_final_validation": _identity(pre_final_path),
            "slurm_terminal_accounting": _identity(accounting_path),
        }
    )
    gate["finalized_at"] = datetime.now(timezone.utc).isoformat()
    _write(gate_path, gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-suite-job-id", required=True)
    arguments = parser.parse_args()
    result = finalize(arguments.full_suite_job_id)
    print(json.dumps({"status": result["status"], "full_suite_green": result["validation"]["full_suite_green"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
