"""Fail-closed Round-11E full-suite submission records.

Round-11E may produce these records only when the targeted gate prevents a
full-suite job.  They document a non-execution and bind the immutable
Round-11D closure decision instead of manufacturing a job ID or JUnit report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from Libs.utils.data.raw_data_registry import write_json


AUDIT_FILENAMES = {
    "execution": "dt_round11e_full_suite_execution.json",
    "failure_triage": "dt_round11e_full_suite_failure_triage.json",
}
SOURCE_FILENAMES = {
    "targeted_gate": "dt_round11e_targeted_gate.json",
    "repair_wave_record": "dt_round11e_repair_wave_record.json",
    "input_snapshot": "dt_round11e_input_evidence_snapshot.json",
    "closure_decision": "dt_round11_closure_decision.json",
}
EXPECTED_SOURCE_JOB_ID = "490754"
PLANNED_COMMAND = "python -m pytest Libs/tests -q --junitxml=Output/audits/dt_round11e_full_suite_results.xml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for {label}")
    return dict(value)


def _artifact(audit_root: Path, filename: str) -> dict[str, object]:
    path = audit_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Round-11E required audit is unavailable: {path}")
    return {
        "path": f"Output/audits/{filename}",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _wave(waves: object, name: str) -> dict[str, Any]:
    if not isinstance(waves, list):
        raise ValueError("Round-11E wave list is invalid")
    for candidate in waves:
        row = _mapping(candidate, "wave")
        if row.get("wave") == name:
            return row
    raise ValueError(f"Round-11E required wave is unavailable: {name}")


def build_full_suite_deferral(audit_root: Path) -> dict[str, dict[str, object]]:
    """Build records that prove why no Round-11E full-suite job was submitted."""

    targeted_gate_path = audit_root / SOURCE_FILENAMES["targeted_gate"]
    repair_path = audit_root / SOURCE_FILENAMES["repair_wave_record"]
    snapshot_path = audit_root / SOURCE_FILENAMES["input_snapshot"]
    closure_path = audit_root / SOURCE_FILENAMES["closure_decision"]
    targeted_gate = _load_object(targeted_gate_path)
    repair = _load_object(repair_path)
    snapshot = _load_object(snapshot_path)
    closure = _load_object(closure_path)

    if targeted_gate.get("source_job_id") != EXPECTED_SOURCE_JOB_ID:
        raise ValueError("Round-11E targeted gate references the wrong frozen job")
    if targeted_gate.get("status") != "blocked_pending_owner_decision":
        raise ValueError("Round-11E full-suite deferral requires the owner-decision targeted gate")
    if targeted_gate.get("full_suite_submission_permitted") is not False:
        raise ValueError("Round-11E full-suite deferral cannot be emitted with an open submission gate")
    wave_d = _wave(targeted_gate.get("waves"), "Wave D")
    if wave_d.get("status") != "blocked_pending_owner_decision":
        raise ValueError("Round-11E Wave D must remain blocked before deferring the full suite")
    repair_wave_d = _wave(repair.get("waves"), "Wave D")
    if repair_wave_d.get("status") != "blocked_pending_owner_decision":
        raise ValueError("Round-11E repair record does not preserve the Wave D owner blocker")
    if snapshot.get("source_job_id") != EXPECTED_SOURCE_JOB_ID:
        raise ValueError("Round-11E input snapshot references the wrong frozen job")
    frozen_result = {
        "tests": snapshot.get("junit_testcase_count"),
        "failures": snapshot.get("failure_count"),
        "errors": snapshot.get("error_count"),
        "skipped": snapshot.get("skipped_count"),
    }
    if frozen_result != {"tests": 366, "failures": 17, "errors": 16, "skipped": 0}:
        raise ValueError(f"Round-11D frozen result is inconsistent: {frozen_result}")
    round_11 = _mapping(closure.get("round_11"), "closure.round_11")
    if round_11.get("closure") != "blocked_full_suite":
        raise ValueError("Round-11 closure must remain blocked while the Round-11E full suite is deferred")

    source_artifacts = {
        key: _artifact(audit_root, filename) for key, filename in SOURCE_FILENAMES.items()
    }
    execution = {
        "schema_version": "dt-round11e-full-suite-execution-v1",
        "audit_kind": "round11e_full_suite_execution",
        "status": "not_submitted_targeted_gate_blocked",
        "source_job_id": EXPECTED_SOURCE_JOB_ID,
        "job_name_prefix": "NrnSift_DT_R11E_",
        "planned_command": PLANNED_COMMAND,
        "submission_attempted": False,
        "job_id": None,
        "terminal_state": "NOT_SUBMITTED",
        "exit_code": None,
        "streams": {
            "stdout": "not_created_submission_prevented",
            "stderr": "not_created_submission_prevented",
            "junit": "not_created_submission_prevented",
            "scontrol": "not_created_submission_prevented",
        },
        "execution_identity": {
            "dirty_status": "not_captured_no_full_suite_execution",
            "git_commit": "not_captured_no_full_suite_execution",
            "config_hash": "not_captured_no_full_suite_execution",
            "resolved_schema_registry_hash": "not_captured_no_full_suite_execution",
            "environment_hash": "not_captured_no_full_suite_execution",
            "clean_worktree_required_before_execution": True,
        },
        "submission_blocker": {
            "targeted_gate_status": targeted_gate.get("status"),
            "wave": wave_d.get("wave"),
            "root_cause_uids": wave_d.get("observed_root_cause_uids"),
            "reason": wave_d.get("blocker"),
        },
        "source_artifacts": source_artifacts,
        "closure_decision_update": {
            "path": source_artifacts["closure_decision"]["path"],
            "sha256": source_artifacts["closure_decision"]["sha256"],
            "read_only_input_preserved": True,
            "overwrite_attempted": False,
            "effective_round11_closure": "blocked_full_suite",
        },
        "prohibited_actions_not_taken": [
            "full-suite submission",
            "test deletion",
            "mandatory test skip",
            "broad xfail",
            "controller state mutation",
            "active contract-union mutation",
            "roadmap mutation",
        ],
    }
    failure_triage = {
        "schema_version": "dt-round11e-full-suite-failure-triage-v1",
        "audit_kind": "round11e_full_suite_failure_triage",
        "status": "not_run_targeted_gate_blocked",
        "source_job_id": EXPECTED_SOURCE_JOB_ID,
        "fresh_full_suite_result": None,
        "fresh_failure_count": None,
        "fresh_error_count": None,
        "fresh_junit_path": None,
        "classification": "not_applicable_no_full_suite_submission",
        "submission_blocker": execution["submission_blocker"],
        "prior_frozen_full_suite_result": frozen_result,
        "prior_frozen_job_terminal_evidence": snapshot.get("job_terminal_evidence"),
        "source_artifacts": source_artifacts,
        "closure": {
            "effective_round11_closure": "blocked_full_suite",
            "closure_decision_read_only_preserved": True,
            "full_suite_rerun_required_before_acceptance": True,
        },
    }
    return {"execution": execution, "failure_triage": failure_triage}


def write_full_suite_deferral(records: Mapping[str, Mapping[str, object]], audit_root: Path) -> None:
    """Write the two non-execution records; never write the frozen closure input."""

    for key, filename in AUDIT_FILENAMES.items():
        payload = records.get(key)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Round-11E full-suite record is missing: {key}")
        write_json(audit_root / filename, dict(payload))


def validate_full_suite_deferral(audit_root: Path) -> dict[str, object]:
    """Validate exact records and the invariant that no full suite was submitted."""

    blockers: list[dict[str, str]] = []
    try:
        expected = build_full_suite_deferral(audit_root)
    except Exception as exc:
        return {
            "status": "failed",
            "blocker_count": 1,
            "blockers": [{"code": "invalid_full_suite_deferral_inputs", "message": str(exc)}],
        }
    for key, filename in AUDIT_FILENAMES.items():
        path = audit_root / filename
        if not path.is_file():
            blockers.append({"code": "missing_required_audit", "message": filename})
            continue
        if _load_object(path) != expected[key]:
            blockers.append({"code": "stale_or_tampered_audit", "message": filename})
    execution = expected["execution"]
    failure_triage = expected["failure_triage"]
    if execution["submission_attempted"] is not False or execution["job_id"] is not None:
        blockers.append({"code": "unexpected_full_suite_submission", "message": "No R11E full-suite job is authorized."})
    if failure_triage["fresh_full_suite_result"] is not None:
        blockers.append({"code": "unexpected_fresh_full_suite_result", "message": "A deferred suite cannot claim fresh results."})
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "source_job_id": EXPECTED_SOURCE_JOB_ID,
        "submission_attempted": False,
        "effective_round11_closure": "blocked_full_suite",
    }
