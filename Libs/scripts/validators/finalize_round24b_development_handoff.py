"""Bind terminal full-suite evidence to the Round-24B Gate and handoff."""

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

from Libs.scripts.validators.validate_round24b_posterior_bound import validate  # noqa: E402


AUDIT_ROOT = REPO_ROOT / "Output/audits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_identities(rows: list[Mapping[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"Round-24B full-suite audit lacks {label}")
    for row in rows:
        path = REPO_ROOT / str(row.get("path", ""))
        if not path.is_file() or identity(path) != dict(row):
            raise ValueError(f"Round-24B {label} identity drift: {row}")


def finalize(job_id: str) -> dict[str, Any]:
    full_path = AUDIT_ROOT / "round24b_full_suite_execution.json"
    accounting_path = AUDIT_ROOT / "round24b_full_suite_slurm_accounting.json"
    gate_path = AUDIT_ROOT / "round24b_twin_adaptation_gate.json"
    handoff_path = AUDIT_ROOT / "round24b_development_handoff.json"
    full = load_json(full_path)
    accounting = load_json(accounting_path)
    gate = load_json(gate_path)
    handoff = load_json(handoff_path)
    if (
        full.get("schema_version") != "neuronsifter-round24b-full-suite-execution-v1"
        or full.get("status") != "passed"
        or full.get("full_suite_green") is not True
        or str(full.get("slurm", {}).get("job_id", "")) != job_id
        or int(full.get("tests_failed", -1)) != 0
        or int(full.get("tests_errored", -1)) != 0
        or int(full.get("tests_skipped", -1)) != 0
    ):
        raise ValueError("Round-24B full-suite execution is not green")
    if (
        accounting.get("schema_version") != "neuronsifter-round24b-slurm-terminal-accounting-v1"
        or accounting.get("status") != "terminal_completed"
        or str(accounting.get("job", {}).get("job_id", "")) != job_id
        or accounting.get("job", {}).get("state") != "COMPLETED"
        or accounting.get("job", {}).get("exit_code") != "0:0"
        or full.get("slurm", {}).get("terminal_accounting") != identity(accounting_path)
    ):
        raise ValueError("Round-24B terminal accounting is not bound to the full suite")
    verify_identities(full["source_files"], "source files")
    verify_identities(full["checkpoints"], "checkpoint files")
    verify_identities(full["evaluation_manifests"], "evaluation manifests")
    validation = validate(require_campaign=True)
    if validation.get("status") != "passed":
        raise ValueError("Round-24B scientific validator did not pass before finalization")
    terminal_states = {
        "neuronsifter_prospective_twin_adaptation_resolved",
        "failed_prospective_twin_adaptation",
        "blocked_prospective_twin_adaptation",
    }
    if gate.get("status") not in terminal_states or handoff.get("status") != gate.get("status"):
        raise ValueError("Round-24B Gate and handoff terminal states differ")
    finalized_at = datetime.now(timezone.utc).isoformat()
    gate["validation"] = {
        "round24b_validator_green": True,
        "full_suite_green": True,
        "pending_full_suite_execution": False,
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
    write_json(gate_path, gate)
    write_json(handoff_path, handoff)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-suite-job-id", required=True)
    args = parser.parse_args()
    gate = finalize(args.full_suite_job_id)
    print(json.dumps({"status": gate["status"], "full_suite_green": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
