"""Validate the inert successor handoff and collect actual source-check receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from Libs.utils.evaluation.operational_twin_candidate_execution import write_json_exclusive
from Libs.utils.evaluation.operational_twin_successor_execution_handoff import (
    BLOCKED_TERMINAL,
    CONFIG_KEY,
    OperationalTwinProtocolError,
    _contract,
    _git,
    _path,
    authenticate_frozen_successor_package,
    build_pre_change_identity_manifest,
    build_successor_execution_handoff,
    canonical_hash,
    executing_source_identity,
    require_approved_successor_execution_handoff,
    validate_materialized_handoff,
)
from Libs.utils.exp_utils import load_config_stack

REPO_ROOT = Path(__file__).resolve().parents[3]


def collect_source_gate(config: dict[str, object], root: Path) -> dict[str, object]:
    contract = _contract(config)
    source_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if _git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise OperationalTwinProtocolError("handoff_source_gate_requires_clean_source")
    source = executing_source_identity(config, root, source_commit)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise OperationalTwinProtocolError("handoff_source_gate_requires_slurm_job")
    destination = _path(root, contract["evidence_inputs"]["source_gate"], absent=True)
    checks = []
    for check in contract["source_checks"]:
        log_path = destination.parent / (check["name"] + ".source_" + source_commit + ".job_" + job_id + ".log")
        _path(root, log_path.relative_to(root).as_posix(), absent=True)
        started = time.monotonic()
        result = subprocess.run(check["argv"], cwd=root, capture_output=True, check=False)
        content = result.stdout + result.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("xb") as handle:
            handle.write(content)
        checks.append({"name": check["name"], "argv": check["argv"],
                       "returncode": result.returncode, "source_commit": source_commit,
                       "slurm_job_id": job_id, "elapsed_seconds": time.monotonic() - started,
                       "log_path": log_path.relative_to(root).as_posix(),
                       "log_sha256": hashlib.sha256(content).hexdigest()})
    receipt = {"schema_version": "neuronsifter-operational-twin-handoff-source-gate-v1",
               "status": "PASS" if all(row["returncode"] == 0 for row in checks) else "BLOCKED",
               "source_identity": source, "checks": checks}
    receipt["receipt_hash"] = canonical_hash(receipt)
    write_json_exclusive(destination, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--mode", choices=("frozen", "source", "adjudicate", "materialized", "approved"), required=True)
    parser.add_argument("--write-pre-change-identity-manifest", type=Path)
    parser.add_argument("--write-source-gate-evidence", action="store_true")
    args = parser.parse_args()
    config = load_config_stack(args.config.resolve())
    result: dict[str, object] = {"mode": args.mode, "status": "PASS", "candidate_dispatch_authorized": False,
                                 "scientific_twin_result": "NOT_RUN", "access_counters": config[CONFIG_KEY]["access_counters"]}
    try:
        contract = _contract(config)
        if args.write_pre_change_identity_manifest:
            expected = _path(REPO_ROOT, contract["evidence_inputs"]["pre_change_identity_manifest"], absent=True)
            if args.write_pre_change_identity_manifest.resolve() != expected:
                raise OperationalTwinProtocolError("handoff_preservation_output_path_mismatch")
            preservation = build_pre_change_identity_manifest(config, REPO_ROOT)
            write_json_exclusive(expected, preservation)
            result["preservation_manifest_hash"] = preservation["manifest_hash"]
        if args.mode == "frozen":
            frozen = authenticate_frozen_successor_package(config, REPO_ROOT)
            result["frozen_package_binding_uid"] = frozen["frozen_package_binding_uid"]
            result["input_identities"] = frozen["input_identities"]
            result["input_read_events"] = frozen["input_read_events"]
        elif args.mode in ("source", "adjudicate"):
            source = _git(REPO_ROOT, "rev-parse", "HEAD").decode().strip()
            checked = build_successor_execution_handoff(config, REPO_ROOT, source_commit=source, require_source_gate=False)
            result["executing_source_identity_hash"] = checked["source_identity"]["source_identity_hash"]
        elif args.mode == "materialized":
            result["handoff_uid"] = validate_materialized_handoff(config, REPO_ROOT)["handoff_uid"]
        else:
            approved = require_approved_successor_execution_handoff(config, REPO_ROOT)
            result["handoff_uid"] = approved["handoff_uid"]
            result["handoff_publication_binding_uid"] = approved["handoff_publication_binding_uid"]
        if args.write_source_gate_evidence:
            if args.mode != "source":
                raise OperationalTwinProtocolError("handoff_source_gate_requires_source_mode")
            receipt = collect_source_gate(config, REPO_ROOT)
            if receipt["status"] != "PASS":
                raise OperationalTwinProtocolError("handoff_source_gate_commands_failed")
            result["source_gate_receipt_hash"] = receipt["receipt_hash"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="BLOCKED", terminal_decision=BLOCKED_TERMINAL,
                      blockers=[{"code": type(error).__name__, "message": str(error)}])
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
