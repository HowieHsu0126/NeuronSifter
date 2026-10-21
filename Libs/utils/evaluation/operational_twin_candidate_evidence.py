"""Aggregate retained evidence for one frozen Operational Twin candidate lineage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

import torch

from Libs.exp.training.run_operational_twin_candidate_attempt import (
    PHASE_RECEIPT_NAME,
    PHASE_RECEIPT_SCHEMA_VERSION,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (
    resolve_candidate_execution_protocol,
    require_candidate_attempt_ownership,
    require_candidate_runtime_ownership,
    write_json_exclusive,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    OperationalTwinProtocolError,
    ResolvedCandidateExecutionProtocol,
    _memory_mebibytes,
    _strict_tres_fields,
    candidate_execution_artifact_paths,
    candidate_fresh_initialization_receipt,
    candidate_runtime_slurm_identity,
    load_candidate_protocol_artifacts,
    repo_path,
    require_candidate_execution_authority,
    sha256_file,
    sha256_text,
    validate_assignment_runtime_evidence,
    validate_candidate_recovery_predecessor,
)


SCHEDULER_RECEIPT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-scheduler-receipt-v3"
)
SCHEDULER_SACCT_FIELDS = (
    "JobID",
    "JobIDRaw",
    "JobName",
    "Partition",
    "State",
    "ExitCode",
    "NodeList",
    "AllocTRES",
    "Elapsed",
)
SCHEDULER_SACCT_FORMAT = ",".join(SCHEDULER_SACCT_FIELDS)
RECOVERY_ELIGIBILITY_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-recovery-eligibility-v2"
)
EXECUTION_CATALOG_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-execution-catalog-v2"
)
EXECUTION_AUDIT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-execution-audit-v2"
)
DISPATCH_MANIFEST_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-dispatch-manifest-v2"
)
TERMINAL_SCHEDULER_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
RECOVERY_CLASSIFICATIONS = {
    "NOT_NEEDED",
    "ELIGIBLE",
    "NOT_ELIGIBLE_NON_INFRASTRUCTURE_FAILURE",
    "BLOCKED_INVALID_PREDECESSOR",
}
SLOT_STATUSES = {"COMPLETED", "FAILED", "NOT_MATERIALIZED_NOT_ELIGIBLE"}
ARTIFACT_IDENTITY_KEYS = {"path", "sha256", "size_bytes"}
INVALID_EXECUTION_BLOCKER_CODES = {
    "candidate_attempt_scheduler_resource_mismatch",
    "candidate_attempt_scheduler_outcome_mismatch",
    "candidate_attempt_access_boundary_invalid",
    "candidate_attempt_evaluation_identity_mismatch",
    "candidate_attempt_execution_gate_invalid",
    "candidate_attempt_failed_with_success_evaluation",
    "candidate_attempt_identity_mismatch",
    "candidate_attempt_initialization_mismatch",
    "candidate_attempt_phase_receipt_invalid",
    "candidate_attempt_runtime_binding_mismatch",
    "candidate_attempt_scheduler_identity_mismatch",
    "candidate_attempt_state_reuse_detected",
    "candidate_attempt_terminal_status_invalid",
    "candidate_best_checkpoint_selection_mismatch",
    "candidate_calibration_role_invalid",
    "candidate_checkpoint_identity_mismatch",
    "candidate_checkpoint_payload_invalid",
    "candidate_checkpoint_unloadable",
    "candidate_extra_attempt_root",
    "candidate_last_checkpoint_progress_mismatch",
    "candidate_no_fit_checkpoint_present",
    "candidate_no_fit_contract_mismatch",
}


def _block(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _attempt_slot_identity(
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    command: Mapping[str, object],
) -> dict[str, object]:
    return {
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "attempt_kind": attempt["attempt_kind"],
        "attempt_ordinal": attempt["attempt_ordinal"],
        "command_index": command["command_index"],
    }


def _empty_attempt_slot(
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    command: Mapping[str, object],
) -> dict[str, object]:
    return {
        **_attempt_slot_identity(assignment, attempt, command),
        "materialized": False,
        "attempt_ownership": None,
        "status": "FAILED",
        "failure_class": None,
        "terminal_reason": None,
        "scheduler_terminal": None,
        "scheduler_receipt": None,
        "execution_authority": None,
        "resource_environment_receipt": None,
        "terminal_status": None,
        "phase_receipt": None,
        "runtime_provenance": None,
        "evaluation_summary": None,
        "checkpoint": None,
        "validation_blockers": [],
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OperationalTwinProtocolError(
            f"candidate_execution_json_invalid: {path} does not contain an object."
        )
    return payload


def artifact_identity(root: Path, path: Path) -> dict[str, object]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    relative = path.relative_to(root).as_posix()
    safe = _path(root, relative)
    return {"path": relative, "sha256": sha256_file(safe), "size_bytes": safe.stat().st_size}


def _capture(argv: list[str], root: Path) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"candidate_scheduler_capture_failed: {' '.join(argv)}"
        ) from error
    if not completed.stdout.strip():
        raise OperationalTwinProtocolError(
            f"candidate_scheduler_capture_empty: {' '.join(argv)}"
        )
    return completed.stdout


def parse_sacct_rows(stdout: str) -> list[dict[str, str]]:
    fields = (
        "job_id",
        "job_id_raw",
        "job_name",
        "partition",
        "state",
        "exit_code",
        "node_list",
        "alloc_tres",
        "elapsed",
    )
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != len(fields):
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_receipt_parse_error: malformed sacct row {line!r}."
            )
        rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _array_terminal_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    job_id: str,
    command_indices: Sequence[int],
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    selected_internal_ids: set[str] = set()
    for index in command_indices:
        expected = f"{job_id}_{int(index)}"
        matches = [dict(row) for row in rows if row.get("job_id") == expected]
        if len(matches) != 1:
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_terminal_missing: {expected} is not present exactly once."
            )
        selected = matches[0]
        internal_id = str(selected.get("job_id_raw") or "").strip()
        if not internal_id:
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_internal_identity_missing: {expected} lacks JobIDRaw."
            )
        if not internal_id.isascii() or not internal_id.isdecimal():
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_internal_identity_invalid: {expected} lacks a numeric internal JobIDRaw."
            )
        if internal_id in selected_internal_ids:
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_internal_identity_reused: JobIDRaw {internal_id} maps to multiple elements."
            )
        selected_internal_ids.add(internal_id)
        state = str(selected["state"]).split("+", 1)[0]
        if state not in TERMINAL_SCHEDULER_STATES:
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_not_terminal: {expected} remains {state}."
            )
        selected["state"] = state
        result[int(index)] = selected
    return result


def validate_scheduler_accounting_fixture(
    payload: Mapping[str, object],
) -> list[dict[str, str]]:
    """Validate the retained real array accounting capture used by the repair."""

    blockers: list[dict[str, str]] = []
    required_keys = {
        "schema_version",
        "source_job_id",
        "captured_at",
        "argv",
        "command_indices",
        "stdout",
        "stdout_sha256",
        "expected_terminal_rows",
    }
    if set(payload) != required_keys:
        blockers.append(
            _block(
                "candidate_scheduler_fixture_keyset_mismatch",
                "Scheduler accounting fixture keyset is not exact.",
            )
        )
        return blockers
    job_id = str(payload.get("source_job_id") or "")
    command_indices = payload.get("command_indices")
    expected_argv = [
        "sacct",
        "-j",
        job_id,
        f"--format={SCHEDULER_SACCT_FORMAT}",
        "-n",
        "-P",
    ]
    stdout = payload.get("stdout")
    if (
        payload.get("schema_version")
        != "neuronsifter-operational-twin-scheduler-fixture-v1"
        or not job_id.isdigit()
        or not isinstance(command_indices, list)
        or not command_indices
        or len(set(command_indices)) != len(command_indices)
        or not all(isinstance(value, int) and value >= 0 for value in command_indices)
        or payload.get("argv") != expected_argv
        or not isinstance(stdout, str)
        or not stdout.strip()
        or payload.get("stdout_sha256") != sha256_text(str(stdout or ""))
    ):
        blockers.append(
            _block(
                "candidate_scheduler_fixture_invalid",
                "Scheduler accounting fixture metadata, command, or raw-output hash drifted.",
            )
        )
        return blockers
    try:
        rows = parse_sacct_rows(stdout)
        selected = _array_terminal_rows(
            rows,
            job_id=job_id,
            command_indices=command_indices,
        )
    except OperationalTwinProtocolError as error:
        blockers.append(_block("candidate_scheduler_fixture_invalid", str(error)))
        return blockers
    expected_rows = [selected[index] for index in command_indices]
    if payload.get("expected_terminal_rows") != expected_rows:
        blockers.append(
            _block(
                "candidate_scheduler_fixture_terminal_mismatch",
                "Selected terminal rows differ from the retained raw accounting output.",
            )
        )
    if not any("." in str(row.get("job_id", "")) for row in rows):
        blockers.append(
            _block(
                "candidate_scheduler_fixture_context_missing",
                "Retained raw accounting output must include unselected step context.",
            )
        )
    return blockers


def _scheduler_dispatch(
    root: Path, execution: Mapping[str, object], arrays: Sequence[Mapping[str, object]],
    phase: str, source_commit: str, resolved: ResolvedCandidateExecutionProtocol | None,
) -> tuple[dict[str, object], dict[str, object]]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import require_candidate_dispatch_manifest
    from Libs.utils.exp_utils import load_config_stack
    if resolved is None or dict(execution) != dict(resolved.execution):
        raise OperationalTwinProtocolError("candidate_scheduler_successor_context_required")
    if phase not in {"initial", "recovery"}:
        raise OperationalTwinProtocolError("candidate_scheduler_phase_invalid")
    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    path = _path(root, execution["output_paths"]["dispatch_manifest" if phase == "initial" else "recovery_dispatch_manifest"])
    dispatch = require_candidate_dispatch_manifest(config, root, _read_json(path), mode=phase, resolved_context=resolved)
    if dispatch["source_commit"] != source_commit or list(arrays) != dispatch[phase + "_arrays"]:
        raise OperationalTwinProtocolError("candidate_scheduler_dispatch_mismatch")
    return dispatch, artifact_identity(root, path)


def merge_scheduler_evidence(
    *collections: Mapping[int, Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    """Join retained arrays without losing duplicate command or Slurm identities."""
    merged: dict[int, dict[str, object]] = {}
    external_ids: set[str] = set()
    internal_ids: set[str] = set()
    for collection in collections:
        for index, evidence in collection.items():
            if type(index) is not int or index in merged:
                raise OperationalTwinProtocolError("candidate_scheduler_index_duplicate: " + str(index))
            terminal = evidence["terminal"]
            external = str(terminal["job_id"])
            internal = str(terminal["job_id_raw"])
            if not internal.isascii() or not internal.isdecimal():
                raise OperationalTwinProtocolError("candidate_scheduler_internal_identity_invalid: " + external)
            if internal in internal_ids:
                raise OperationalTwinProtocolError("candidate_scheduler_internal_identity_reused: " + internal)
            if not external or external in external_ids:
                raise OperationalTwinProtocolError("candidate_scheduler_external_identity_reused: " + external)
            merged[index] = dict(evidence)
            external_ids.add(external)
            internal_ids.add(internal)
    return merged


def collect_scheduler_receipts(
    root: Path,
    execution: Mapping[str, object],
    arrays: Sequence[Mapping[str, object]],
    *,
    phase: str,
    source_commit: str,
    command_capture: Callable[[list[str], Path], str] = _capture,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[int, dict[str, object]]:
    """Capture bounded sacct evidence for only dispatch-manifest job IDs."""

    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    _, dispatch_identity = _scheduler_dispatch(
        root, execution, arrays, phase, source_commit, resolved_context
    )
    scheduler_root = _path(
        root,
        str(dict(execution["output_paths"])["scheduler_receipts_root"]),
    )
    pending: list[tuple[Path, dict[str, object], dict[int, dict[str, str]]]] = []
    checked: dict[int, dict[str, object]] = {}
    if phase == "recovery":
        initial = _read_json(_path(root, execution["output_paths"]["dispatch_manifest"]))
        checked = load_scheduler_receipts(
            root, execution, initial["initial_arrays"], phase="initial",
            source_commit=initial["source_commit"], resolved_context=resolved_context,
        )
    for array in arrays:
        job_id = str(array["job_id"])
        indices = [int(value) for value in array["command_indices"]]
        argv = [
            "sacct",
            "-j",
            job_id,
            f"--format={SCHEDULER_SACCT_FORMAT}",
            "-n",
            "-P",
        ]
        stdout = command_capture(argv, root)
        rows = parse_sacct_rows(stdout)
        terminals = _array_terminal_rows(rows, job_id=job_id, command_indices=indices)
        receipt = {
            "schema_version": SCHEDULER_RECEIPT_SCHEMA_VERSION,
            "successor_execution_binding": dict(resolved_context.binding),
            "dispatch_manifest": dispatch_identity,
            "array_submission_receipt": array["submission_receipt"],
            "phase": phase,
            "job_id": job_id,
            "job_name": array["job_name"],
            "command_indices": indices,
            "source_commit": source_commit,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "argv": argv,
            "stdout": stdout,
            "stdout_sha256": sha256_text(stdout),
            "terminal_rows": [terminals[index] for index in indices],
        }
        receipt_path = _path(root, (scheduler_root / f"{phase}_{job_id}.json").relative_to(root).as_posix(), absent=True)
        checked = merge_scheduler_evidence(
            checked, {index: {"terminal": terminal, "receipt": None} for index, terminal in terminals.items()},
        )
        pending.append((receipt_path, receipt, terminals))
    # Validate all array identities before publishing any receipt.
    scheduler_by_index: dict[int, dict[str, object]] = {}
    for receipt_path, receipt, terminals in pending:
        write_json_exclusive(receipt_path, receipt)
        receipt_identity = artifact_identity(root, receipt_path)
        scheduler_by_index = merge_scheduler_evidence(
            scheduler_by_index,
            {index: {"terminal": terminal, "receipt": receipt_identity} for index, terminal in terminals.items()},
        )
    return scheduler_by_index


def load_scheduler_receipts(
    root: Path,
    execution: Mapping[str, object],
    arrays: Sequence[Mapping[str, object]],
    *,
    phase: str,
    source_commit: str,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[int, dict[str, object]]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    _, dispatch_identity = _scheduler_dispatch(
        root, execution, arrays, phase, source_commit, resolved_context
    )
    scheduler_root = _path(
        root,
        str(dict(execution["output_paths"])["scheduler_receipts_root"]),
    )
    scheduler_by_index: dict[int, dict[str, object]] = {}
    for array in arrays:
        job_id = str(array["job_id"])
        receipt_path = _path(root, (scheduler_root / f"{phase}_{job_id}.json").relative_to(root).as_posix())
        if not receipt_path.is_file():
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_receipt_missing: {receipt_path.relative_to(root)}"
            )
        receipt = _read_json(receipt_path)
        indices = [int(value) for value in array["command_indices"]]
        expected_argv = [
            "sacct",
            "-j",
            job_id,
            f"--format={SCHEDULER_SACCT_FORMAT}",
            "-n",
            "-P",
        ]
        expected_keys = {
            "schema_version",
            "successor_execution_binding",
            "dispatch_manifest",
            "array_submission_receipt",
            "phase",
            "job_id",
            "job_name",
            "command_indices",
            "source_commit",
            "captured_at",
            "argv",
            "stdout",
            "stdout_sha256",
            "terminal_rows",
        }
        stdout = receipt.get("stdout")
        if (
            set(receipt) != expected_keys
            or receipt.get("successor_execution_binding") != dict(resolved_context.binding)
            or receipt.get("dispatch_manifest") != dispatch_identity
            or receipt.get("array_submission_receipt") != array["submission_receipt"]
            or receipt.get("schema_version") != SCHEDULER_RECEIPT_SCHEMA_VERSION
            or receipt.get("phase") != phase
            or receipt.get("job_id") != job_id
            or receipt.get("job_name") != array.get("job_name")
            or receipt.get("command_indices") != indices
            or receipt.get("source_commit") != source_commit
            or receipt.get("argv") != expected_argv
            or not isinstance(stdout, str)
            or not stdout.strip()
            or receipt.get("stdout_sha256") != sha256_text(str(stdout or ""))
        ):
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_receipt_invalid: {receipt_path.relative_to(root)}"
            )
        parsed_rows = parse_sacct_rows(stdout)
        terminals = _array_terminal_rows(
            parsed_rows,
            job_id=job_id,
            command_indices=indices,
        )
        expected_terminal_rows = [terminals[index] for index in indices]
        if receipt.get("terminal_rows") != expected_terminal_rows:
            raise OperationalTwinProtocolError(
                f"candidate_scheduler_receipt_invalid: {receipt_path.relative_to(root)} terminal rows differ from raw stdout."
            )
        receipt_identity = artifact_identity(root, receipt_path)
        scheduler_by_index = merge_scheduler_evidence(
            scheduler_by_index,
            {index: {"terminal": terminal, "receipt": receipt_identity} for index, terminal in terminals.items()},
        )
    if phase == "recovery":
        initial = _read_json(_path(root, execution["output_paths"]["dispatch_manifest"]))
        initial_scheduler = load_scheduler_receipts(
            root, execution, initial["initial_arrays"], phase="initial",
            source_commit=initial["source_commit"], resolved_context=resolved_context,
        )
        merge_scheduler_evidence(initial_scheduler, scheduler_by_index)
    return scheduler_by_index


def _checkpoint_evidence(
    root: Path,
    assignment: Mapping[str, object],
    output_root: Path,
    terminal: Mapping[str, object],
    provenance: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    checkpoint_allowed = bool(assignment["checkpoint_creation_allowed"])
    checkpoint_dir = output_root / "checkpoints"
    if not checkpoint_allowed:
        checkpoint_paths = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []
        if checkpoint_paths:
            blockers.append(
                _block(
                    "candidate_no_fit_checkpoint_present",
                    f"{assignment['assignment_uid']} is no-fit but contains checkpoints.",
                )
            )
        if (
            terminal.get("checkpoint_created") is not False
            or int(terminal.get("trainable_parameter_count", -1)) != 0
        ):
            blockers.append(
                _block(
                    "candidate_no_fit_contract_mismatch",
                    f"{assignment['assignment_uid']} does not retain the zero-parameter no-fit contract.",
                )
            )
        return {
            "contract": "no_checkpoint",
            "best_dev": None,
            "last": None,
            "content_equal": None,
        }, blockers

    best_path = checkpoint_dir / "best_dev.pt"
    last_path = checkpoint_dir / "last.pt"
    if not best_path.is_file() or not last_path.is_file() or best_path == last_path:
        blockers.append(
            _block(
                "candidate_checkpoint_roles_missing",
                f"{assignment['assignment_uid']} lacks distinct best_dev.pt and last.pt paths.",
            )
        )
        return {
            "contract": "best_and_last",
            "best_dev": None,
            "last": None,
            "content_equal": None,
        }, blockers
    payloads: dict[str, dict[str, object]] = {}
    for role, path in (("best_dev", best_path), ("last", last_path)):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            blockers.append(
                _block(
                    "candidate_checkpoint_unloadable",
                    f"{assignment['assignment_uid']} {role} checkpoint cannot load: {error}",
                )
            )
            continue
        if not isinstance(payload, Mapping):
            blockers.append(
                _block(
                    "candidate_checkpoint_payload_invalid",
                    f"{assignment['assignment_uid']} {role} checkpoint is not a mapping.",
                )
            )
            continue
        payloads[role] = dict(payload)
        for field in (
            "candidate_lineage_uid",
            "protocol_uid",
            "assignment_uid",
            "attempt_uid",
            "method_id",
            "seed",
        ):
            expected = (
                terminal.get(field)
                if field in terminal
                else assignment.get(field)
            )
            if payload.get(field) != expected:
                blockers.append(
                    _block(
                        "candidate_checkpoint_identity_mismatch",
                        f"{assignment['assignment_uid']} {role} checkpoint {field} drifted.",
                    )
                )
        for field in (
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "step",
            "method_protocol",
            "trainable_parameter_count",
        ):
            if field not in payload:
                blockers.append(
                    _block(
                        "candidate_checkpoint_payload_invalid",
                        f"{assignment['assignment_uid']} {role} checkpoint lacks {field}.",
                    )
                )
    if "best_dev" in payloads and payloads["best_dev"].get("epoch") != terminal.get(
        "selected_checkpoint_epoch"
    ):
        blockers.append(
            _block(
                "candidate_best_checkpoint_selection_mismatch",
                f"{assignment['assignment_uid']} best checkpoint epoch differs from the terminal selection.",
            )
        )
    if "last" in payloads and (
        payloads["last"].get("epoch") != terminal.get("epochs_completed")
        or payloads["last"].get("step") != terminal.get("steps_completed")
    ):
        blockers.append(
            _block(
                "candidate_last_checkpoint_progress_mismatch",
                f"{assignment['assignment_uid']} last checkpoint progress differs from the terminal record.",
            )
        )
    calibration_eligible = bool(assignment["calibration_eligible"])
    if calibration_eligible:
        if (
            terminal.get("calibration_fitting_executed") is not True
            or provenance.get("calibration_execution_authority") is None
            or payloads.get("best_dev", {}).get("calibration_scale") is None
        ):
            blockers.append(
                _block(
                    "candidate_calibration_role_invalid",
                    f"{assignment['assignment_uid']} lacks authorized calibration on best_dev.",
                )
            )
    elif (
        terminal.get("calibration_fitting_executed") is not False
        or provenance.get("calibration_execution_authority") is not None
        or payloads.get("best_dev", {}).get("calibration_scale") is not None
    ):
        blockers.append(
            _block(
                "candidate_calibration_role_invalid",
                f"{assignment['assignment_uid']} performed unauthorized calibration fitting.",
            )
        )
    return {
        "contract": "best_and_last",
        "best_dev": artifact_identity(root, best_path),
        "last": artifact_identity(root, last_path),
        "content_equal": sha256_file(best_path) == sha256_file(last_path),
    }, blockers


def validate_scheduler_attempt_evidence(
    scheduler: Mapping[str, object],
    identity: Mapping[str, object],
    requirements: Mapping[str, object],
    phase: Mapping[str, object],
) -> list[dict[str, str]]:
    """Join raw accounting to retained allocation and the worker's process result."""
    blockers: list[dict[str, str]] = []
    fields = {
        "job_id": "resource_query_job_id", "job_id_raw": "slurm_job_id",
        "job_name": "slurm_job_name", "partition": "slurm_job_partition",
        "node_list": "slurm_job_nodelist",
    }
    if any(str(scheduler.get(key) or "") != str(identity.get(value) or "")
           for key, value in fields.items()):
        blockers.append(_block(
            "candidate_attempt_scheduler_identity_mismatch",
            "Accounting identity or placement differs from retained resource provenance.",
        ))
    try:
        allocated = _strict_tres_fields(scheduler.get("alloc_tres"), label="sacct AllocTRES")
        if int(allocated["cpu"]) != int(requirements["cpus_per_task"]) or int(allocated["node"]) != 1:
            raise ValueError("Accounting CPU or node allocation drifted.")
        # The retained Slurm 22.05 fixture omits memory/GRES in AllocTRES.
        # Their mandatory proof remains the separately authenticated scontrol
        # receipt; any accounting value that is present must agree with it.
        if "mem" in allocated and _memory_mebibytes(allocated["mem"]) != _memory_mebibytes(requirements["memory"]):
            raise ValueError("Accounting memory allocation drifted.")
        expected_gpus = 1 if requirements["gres"] == "gpu:1" else 0
        gpu_counts = [int(value) for key, value in allocated.items()
                      if key == "gres/gpu" or key.startswith("gres/gpu:")]
        if any(value != expected_gpus for value in gpu_counts):
            raise ValueError("Accounting GPU allocation drifted.")
    except (KeyError, TypeError, ValueError, OperationalTwinProtocolError) as error:
        blockers.append(_block("candidate_attempt_scheduler_resource_mismatch", str(error)))

    steps = phase.get("phases")
    expected_names = ["training_and_calibration", "open_development_calibration_evaluation"]
    valid_steps = (
        isinstance(steps, list) and len(steps) in (1, 2)
        and all(isinstance(step, Mapping) and set(step) == {"phase", "argv", "returncode"}
                and type(step["returncode"]) is int
                and isinstance(step["argv"], list) and step["argv"]
                and all(isinstance(value, str) for value in step["argv"]) for step in steps)
        and [step["phase"] for step in steps] == expected_names[:len(steps)]
    )
    if valid_steps:
        codes = [step["returncode"] for step in steps]
        valid_steps = (
            phase.get("status") == "PASS" and codes == [0, 0]
            or phase.get("status") == "FAIL" and codes[-1] != 0
            and (len(codes) == 1 or codes[0] == 0)
        )
    if not valid_steps:
        blockers.append(_block("candidate_attempt_phase_receipt_invalid", "Worker phase steps do not match its process contract."))
        return blockers

    exit_fields = str(scheduler.get("exit_code") or "").split(":")
    state = scheduler.get("state")
    valid_exit = len(exit_fields) == 2 and all(
        field.isascii() and field.isdecimal() for field in exit_fields
    )
    inconsistent = not valid_exit or state not in TERMINAL_SCHEDULER_STATES
    if valid_exit:
        exit_status, signal = map(int, exit_fields)
        if state == "COMPLETED":
            inconsistent |= phase["status"] != "PASS" or (exit_status, signal) != (0, 0)
        elif phase["status"] == "PASS":
            inconsistent = True
        elif state == "FAILED" and signal == 0:
            # The wrapper exits with Python's worker return code. Scheduler
            # signals/infrastructure states remain explicit external interruptions.
            inconsistent |= exit_status != codes[-1] % 256
    if inconsistent:
        blockers.append(_block(
            "candidate_attempt_scheduler_outcome_mismatch",
            "Accounting completion/exit evidence contradicts the retained worker process result.",
        ))
    return blockers


def validate_materialized_attempt(
    config: Mapping[str, object],
    root: Path,
    *,
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    command: Mapping[str, object],
    scheduler_evidence: Mapping[str, object],
    source_commit: str,
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    command_catalog: Mapping[str, object],
    preflight: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
    blockers: list[dict[str, str]] = []
    attempt_root = repo_path(root, str(attempt["output_root"]))
    if not attempt_root.is_dir():
        row = _empty_attempt_slot(assignment, attempt, command)
        row.update(
            failure_class="missing_attempt_root",
            validation_blockers=[
                _block(
                    "candidate_attempt_root_missing",
                    f"Required attempt root is absent: {attempt['output_root']}.",
                )
            ],
        )
        return row, list(row["validation_blockers"])
    try:
        require_candidate_attempt_ownership(config, root, resolved, str(attempt["attempt_uid"]))
    except (KeyError, TypeError, ValueError, OSError) as error:
        row = _empty_attempt_slot(assignment, attempt, command)
        row.update(materialized=True, failure_class="invalid_attempt_ownership",
                   validation_blockers=[_block("candidate_attempt_ownership_invalid", str(error))])
        return row, list(row["validation_blockers"])
    terminal_path = attempt_root / "terminal_status.json"
    phase_path = attempt_root / PHASE_RECEIPT_NAME
    if not terminal_path.is_file() or not phase_path.is_file():
        row = _empty_attempt_slot(assignment, attempt, command)
        row.update(
            materialized=True,
            failure_class="missing_terminal_evidence",
            validation_blockers=[
                _block(
                    "candidate_attempt_terminal_evidence_missing",
                    f"{attempt['attempt_uid']} lacks terminal status or phase receipt.",
                )
            ],
        )
        return row, list(row["validation_blockers"])
    terminal = _read_json(terminal_path)
    phase = _read_json(phase_path)
    expected_identity = {
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
    }
    for field, expected in expected_identity.items():
        if terminal.get(field) != expected or phase.get(field) != expected:
            blockers.append(
                _block(
                    "candidate_attempt_identity_mismatch",
                    f"{attempt['attempt_uid']} terminal or phase {field} drifted.",
                )
            )
    if (
        phase.get("schema_version") != PHASE_RECEIPT_SCHEMA_VERSION
        or phase.get("successor_execution_binding") != dict(resolved.binding)
        or phase.get("command_index") != command.get("command_index")
        or phase.get("source_commit") != source_commit
    ):
        blockers.append(
            _block(
                "candidate_attempt_phase_receipt_invalid",
                f"{attempt['attempt_uid']} phase receipt contract drifted.",
            )
        )
    if terminal.get("state_resume_used") is not False:
        blockers.append(
            _block(
                "candidate_attempt_state_reuse_detected",
                f"{attempt['attempt_uid']} does not prove state_resume_used=false.",
            )
        )
    protocol = dict(resolved.protocol)
    if terminal.get("fresh_initialization_receipt") != candidate_fresh_initialization_receipt(
        protocol, resolved
    ):
        blockers.append(
            _block(
                "candidate_attempt_initialization_mismatch",
                f"{attempt['attempt_uid']} fresh-initialization receipt drifted.",
            )
        )
    started_at: datetime | None = None
    try:
        started_at = datetime.fromisoformat(
            str(terminal["started_at"]).replace("Z", "+00:00")
        )
        paths = candidate_execution_artifact_paths(protocol, assignment, attempt)
        execution_context = require_candidate_execution_authority(
            config,
            root,
            assignment=assignment,
            attempt=attempt,
            candidate_manifest=candidate_manifest,
            assignment_manifest=assignment_manifest,
            command_catalog=command_catalog,
            preflight=preflight,
            execution_authority_path=paths["execution_authority"],
            resource_environment_receipt_path=paths[
                "resource_environment_receipt"
            ],
            cuda_available=False,
            cuda_device_count=0,
            torch_version="",
            current_source_commit=source_commit,
            now=started_at,
            validate_current_runtime=False,
            resolved_context=resolved,
        )
    except (KeyError, TypeError, ValueError, OperationalTwinProtocolError) as error:
        execution_context = None
        blockers.append(
            _block(
                "candidate_attempt_execution_gate_invalid",
                f"{attempt['attempt_uid']} execution gate is invalid: {error}",
            )
        )
    scheduler_terminal = scheduler_evidence.get("terminal")
    if not isinstance(scheduler_terminal, Mapping):
        blockers.append(
            _block(
                "candidate_attempt_scheduler_terminal_missing",
                f"{attempt['attempt_uid']} lacks scheduler terminal evidence.",
            )
        )
    else:
        try:
            scheduler_identity = terminal.get("scheduler_identity")
            if not isinstance(scheduler_identity, Mapping):
                raise OperationalTwinProtocolError(
                    "candidate_runtime_slurm_identity_invalid: terminal scheduler identity is missing."
                )
            normalized_scheduler_identity = candidate_runtime_slurm_identity(
                values=scheduler_identity
            )
            if execution_context is not None and normalized_scheduler_identity != candidate_runtime_slurm_identity(
                values=execution_context["resource_receipt_payload"]["slurm"]
            ):
                raise OperationalTwinProtocolError(
                    "candidate_runtime_slurm_identity_invalid: terminal scheduler identity differs from its resource receipt."
                )
        except OperationalTwinProtocolError as error:
            normalized_scheduler_identity = None
            blockers.append(
                _block(
                    "candidate_attempt_scheduler_identity_mismatch",
                    f"{attempt['attempt_uid']} terminal scheduler identity is invalid: {error}",
                )
            )
        if normalized_scheduler_identity is not None:
            blockers.extend(validate_scheduler_attempt_evidence(
                scheduler_terminal, normalized_scheduler_identity,
                assignment["resource_requirements"], phase,
            ))
    if phase.get("access_counters") != dict(
        config["operational_twin_candidate_execution"]
    )["access_counters"] or phase.get("closed_flags") != dict(
        config["operational_twin_candidate_execution"]
    )["closed_flags"]:
        blockers.append(
            _block(
                "candidate_attempt_access_boundary_invalid",
                f"{attempt['attempt_uid']} phase receipt opened a forbidden gate.",
            )
        )
    terminal_status = str(terminal.get("status", ""))
    phase_status = str(phase.get("status", ""))
    slot_status = (
        "COMPLETED"
        if terminal_status == "COMPLETED" and phase_status == "PASS"
        else "FAILED"
    )
    failure_class = terminal.get("failure_class")
    if terminal_status == "COMPLETED" and phase_status == "FAIL":
        failure_class = "post_training_evaluation_failure"
    provenance_path = attempt_root / "provenance/runtime_provenance.json"
    evaluation_path = attempt_root / "evaluation/summary.json"
    provenance: dict[str, object] | None = None
    checkpoint: dict[str, object] | None = None
    if terminal_status == "COMPLETED":
        if not provenance_path.is_file():
            blockers.append(
                _block(
                    "candidate_attempt_runtime_provenance_missing",
                    f"{attempt['attempt_uid']} lacks runtime provenance.",
                )
            )
        else:
            provenance = _read_json(provenance_path)
            blockers.extend(
                validate_assignment_runtime_evidence(assignment, terminal, provenance, resolved_context=resolved)
            )
            if execution_context is not None and (
                provenance.get("candidate_execution_authority")
                != execution_context["execution_authority"]
                or provenance.get("candidate_resource_environment_receipt")
                != execution_context["resource_environment_receipt"]
            ):
                blockers.append(
                    _block(
                        "candidate_attempt_runtime_binding_mismatch",
                        f"{attempt['attempt_uid']} runtime provenance differs from its execution gate.",
                    )
                )
            checkpoint, checkpoint_blockers = _checkpoint_evidence(
                root,
                assignment,
                attempt_root,
                terminal,
                provenance,
            )
            blockers.extend(checkpoint_blockers)
        if phase_status == "PASS":
            if not evaluation_path.is_file():
                blockers.append(
                    _block(
                        "candidate_attempt_evaluation_missing",
                        f"{attempt['attempt_uid']} lacks its open evaluation summary.",
                    )
                )
            else:
                evaluation = _read_json(evaluation_path)
                required_evaluation = {
                    "status": "PASS",
                    "candidate_lineage_uid": assignment["candidate_lineage_uid"],
                    "protocol_uid": assignment["protocol_uid"],
                    "assignment_uid": assignment["assignment_uid"],
                    "attempt_uid": attempt["attempt_uid"],
                    "method_id": assignment["method_id"],
                    "seed": assignment["seed"],
                    "checkpoint_reselection_performed": False,
                }
                for field, expected in required_evaluation.items():
                    if evaluation.get(field) != expected:
                        blockers.append(
                            _block(
                                "candidate_attempt_evaluation_identity_mismatch",
                                f"{attempt['attempt_uid']} evaluation {field} drifted.",
                            )
                        )
        elif evaluation_path.is_file() and _read_json(evaluation_path).get("status") == "PASS":
            blockers.append(
                _block(
                    "candidate_attempt_failed_with_success_evaluation",
                    f"{attempt['attempt_uid']} failed but retains a success-shaped evaluation summary.",
                )
            )
    elif terminal_status == "FAILED":
        blockers.extend(
            validate_candidate_recovery_predecessor(
                protocol,
                assignment,
                attempt,
                terminal,
                allowed_failure_class=terminal.get("failure_class"),
                root=root,
                candidate_manifest=candidate_manifest,
                assignment_manifest=assignment_manifest,
                command_catalog=command_catalog,
                preflight=preflight,
                resolved_context=resolved,
            )
        )
        if evaluation_path.is_file() and _read_json(evaluation_path).get("status") == "PASS":
            blockers.append(
                _block(
                    "candidate_attempt_failed_with_success_evaluation",
                    f"{attempt['attempt_uid']} failed but retains a success-shaped evaluation summary.",
                )
            )
    else:
        blockers.append(
            _block(
                "candidate_attempt_terminal_status_invalid",
                f"{attempt['attempt_uid']} has unsupported terminal status {terminal_status!r}.",
            )
        )
    row = {
        **_attempt_slot_identity(assignment, attempt, command),
        "materialized": True,
        "attempt_ownership": artifact_identity(root, attempt_root / "attempt_ownership.json"),
        "status": slot_status,
        "failure_class": failure_class,
        "terminal_reason": terminal.get("terminal_reason"),
        "scheduler_terminal": dict(scheduler_terminal or {}),
        "scheduler_receipt": scheduler_evidence.get("receipt"),
        "execution_authority": terminal.get("candidate_execution_authority"),
        "resource_environment_receipt": terminal.get(
            "candidate_resource_environment_receipt"
        ),
        "terminal_status": artifact_identity(root, terminal_path),
        "phase_receipt": artifact_identity(root, phase_path),
        "runtime_provenance": (
            artifact_identity(root, provenance_path)
            if provenance_path.is_file()
            else None
        ),
        "evaluation_summary": (
            artifact_identity(root, evaluation_path)
            if evaluation_path.is_file()
            else None
        ),
        "checkpoint": checkpoint,
        "validation_blockers": blockers,
    }
    return row, blockers


def classify_recovery_eligibility(
    slot: Mapping[str, object],
    *,
    infrastructure_failure_class: str,
) -> str:
    if slot.get("validation_blockers"):
        return "BLOCKED_INVALID_PREDECESSOR"
    if slot.get("status") == "COMPLETED":
        return "NOT_NEEDED"
    if slot.get("failure_class") == infrastructure_failure_class:
        return "ELIGIBLE"
    return "NOT_ELIGIBLE_NON_INFRASTRUCTURE_FAILURE"


def _commands_by_index(command_catalog: Mapping[str, object]) -> dict[int, dict[str, object]]:
    rows = [dict(row) for row in command_catalog["commands"]]
    result = {int(row["command_index"]): row for row in rows}
    if len(result) != len(rows):
        raise OperationalTwinProtocolError(
            "candidate_command_index_duplicate: command catalog contains duplicate indices."
        )
    return result


def build_recovery_eligibility(
    config: Mapping[str, object],
    root: Path,
    *,
    dispatch_manifest: Mapping[str, object],
    scheduler_by_index: Mapping[int, Mapping[str, object]],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
    execution = dict(resolved.execution)
    candidate, assignment_manifest, command_catalog, preflight = (
        load_candidate_protocol_artifacts(config, root, resolved_context=resolved)
    )
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import require_candidate_dispatch_manifest
    require_candidate_dispatch_manifest(config, root, dispatch_manifest, mode="initial", resolved_context=resolved)
    if dispatch_manifest.get("source_commit") is None:
        raise OperationalTwinProtocolError(
            "candidate_initial_dispatch_invalid: recovery classification requires the complete initial dispatch."
        )
    commands = _commands_by_index(command_catalog)
    failure_class = str(
        dict(resolved.protocol)[
            "scientific_contract"
        ]["failure_policy"]["interrupted_class"]
    )
    rows: list[dict[str, object]] = []
    for assignment_value in assignment_manifest["assignments"]:
        assignment = dict(assignment_value)
        initial = next(
            dict(value)
            for value in assignment["attempts"]
            if value["attempt_kind"] == "initial"
        )
        recovery = next(
            dict(value)
            for value in assignment["attempts"]
            if value["attempt_kind"] == "infrastructure_recovery_01"
        )
        command = next(
            row
            for row in commands.values()
            if row["attempt_uid"] == initial["attempt_uid"]
        )
        command_index = int(command["command_index"])
        scheduler = scheduler_by_index.get(command_index)
        if scheduler is None:
            slot = {
                "assignment_uid": assignment["assignment_uid"],
                "attempt_uid": initial["attempt_uid"],
                "status": "FAILED",
                "failure_class": "missing_scheduler_terminal",
                "validation_blockers": [
                    _block(
                        "candidate_scheduler_terminal_missing",
                        f"Initial command {command_index} lacks scheduler evidence.",
                    )
                ],
            }
        else:
            slot, _ = validate_materialized_attempt(
                config,
                root,
                assignment=assignment,
                attempt=initial,
                command=command,
                scheduler_evidence=scheduler,
                source_commit=str(dispatch_manifest["source_commit"]),
                candidate_manifest=candidate,
                assignment_manifest=assignment_manifest,
                command_catalog=command_catalog,
                preflight=preflight,
                resolved_context=resolved,
            )
        classification = classify_recovery_eligibility(
            slot,
            infrastructure_failure_class=failure_class,
        )
        recovery_command = next(
            row
            for row in commands.values()
            if row["attempt_uid"] == recovery["attempt_uid"]
        )
        rows.append(
            {
                "assignment_uid": assignment["assignment_uid"],
                "method_id": assignment["method_id"],
                "seed": assignment["seed"],
                "initial_attempt_uid": initial["attempt_uid"],
                "initial_command_index": command_index,
                "initial_scheduler_terminal": (
                    dict(scheduler["terminal"]) if scheduler is not None else None
                ),
                "initial_attempt_status": slot.get("status"),
                "initial_failure_class": slot.get("failure_class"),
                "initial_execution_authority": slot.get("execution_authority"),
                "initial_resource_environment_receipt": slot.get(
                    "resource_environment_receipt"
                ),
                "recovery_attempt_uid": recovery["attempt_uid"],
                "recovery_command_index": recovery_command["command_index"],
                "recovery_eligibility": classification,
                "validation_blockers": slot.get("validation_blockers", []),
            }
        )
    status = "PASS" if all(
        row["recovery_eligibility"] in RECOVERY_CLASSIFICATIONS for row in rows
    ) else "FAIL"
    return {
        "schema_version": RECOVERY_ELIGIBILITY_SCHEMA_VERSION,
        "successor_execution_binding": dict(resolved.binding),
        "status": status,
        "terminal_decision": (
            "OPERATIONAL_TWIN_CANDIDATE_RECOVERY_ELIGIBILITY_RESOLVED"
            if status == "PASS"
            else "OPERATIONAL_TWIN_CANDIDATE_RECOVERY_ELIGIBILITY_INVALID"
        ),
        "candidate_lineage_uid": execution["candidate_lineage_uid"],
        "protocol_uid": execution["protocol_uid"],
        "preflight_uid": execution["preflight_uid"],
        "source_commit": dispatch_manifest["source_commit"],
        "dispatch_manifest": dispatch_manifest,
        "assignment_count": len(rows),
        "assignments": rows,
        "access_counters": execution["access_counters"],
        "closed_flags": execution["closed_flags"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def require_recovery_eligibility(
    config: Mapping[str, object], root: Path, path: Path, *,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    """Recompute eligibility from retained exact initial dispatch and scheduler evidence."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
    canonical = _path(root, resolved.execution["output_paths"]["recovery_eligibility"])
    if path.absolute() != canonical:
        raise OperationalTwinProtocolError("candidate_recovery_path_not_canonical")
    payload = _read_json(canonical)
    if set(payload) != set(config["operational_twin_candidate_execution_schema"]["recovery_eligibility_fields"]):
        raise OperationalTwinProtocolError("candidate_recovery_keyset_mismatch")
    dispatch_path = _path(root, resolved.execution["output_paths"]["dispatch_manifest"])
    dispatch = _read_json(dispatch_path)
    scheduler = load_scheduler_receipts(root, resolved.execution, dispatch["initial_arrays"],
        phase="initial", source_commit=dispatch["source_commit"], resolved_context=resolved)
    expected = build_recovery_eligibility(config, root, dispatch_manifest=dispatch,
        scheduler_by_index=scheduler, resolved_context=resolved)
    # The capture timestamp is metadata, never an eligibility input.
    expected["created_at"] = payload["created_at"]
    if payload != expected:
        raise OperationalTwinProtocolError("candidate_recovery_evidence_drift")
    return payload


def _unmaterialized_recovery_slot(
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    command: Mapping[str, object],
    eligibility: Mapping[str, object],
) -> dict[str, object]:
    row = _empty_attempt_slot(assignment, attempt, command)
    row.update(
        status="NOT_MATERIALIZED_NOT_ELIGIBLE",
        terminal_reason=eligibility["recovery_eligibility"],
    )
    return row


def _extra_attempt_roots(
    root: Path,
    assignment_manifest: Mapping[str, object],
) -> list[str]:
    lineage_uid = str(assignment_manifest["candidate_lineage_uid"])
    assignments_root = (
        root
        / "Output/results/operational_twin/candidate_lineages"
        / lineage_uid
        / "assignments"
    )
    if not assignments_root.is_dir():
        return []
    expected_assignments = {
        str(row["assignment_uid"]): {
            str(attempt["attempt_uid"]) for attempt in row["attempts"]
        }
        for row in assignment_manifest["assignments"]
    }
    extras: list[str] = []
    for assignment_path in assignments_root.iterdir():
        if not assignment_path.is_dir() or assignment_path.name not in expected_assignments:
            extras.append(assignment_path.relative_to(root).as_posix())
            continue
        attempts_root = assignment_path / "attempts"
        if not attempts_root.is_dir():
            continue
        for attempt_path in attempts_root.iterdir():
            if attempt_path.name not in expected_assignments[assignment_path.name]:
                extras.append(attempt_path.relative_to(root).as_posix())
    return sorted(extras)


def build_execution_catalog(
    config: Mapping[str, object],
    root: Path,
    *,
    dispatch_manifest: Mapping[str, object],
    recovery_eligibility: Mapping[str, object],
    scheduler_by_index: Mapping[int, Mapping[str, object]],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
    require_candidate_runtime_ownership(config, root, resolved)
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import require_candidate_dispatch_manifest
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path

    require_candidate_dispatch_manifest(config, root, dispatch_manifest, mode="initial", resolved_context=resolved)
    source_by_index = {index: dispatch_manifest["source_commit"]
                       for row in dispatch_manifest["initial_arrays"] for index in row["command_indices"]}
    recovery_dispatch_path = _path(root, resolved.execution["output_paths"]["recovery_dispatch_manifest"])
    if recovery_dispatch_path.exists():
        recovery_dispatch = require_candidate_dispatch_manifest(
            config, root, _read_json(recovery_dispatch_path), mode="recovery", resolved_context=resolved,
        )
        for row in recovery_dispatch["recovery_arrays"]:
            for index in row["command_indices"]:
                if index in source_by_index:
                    raise OperationalTwinProtocolError("candidate_scheduler_index_duplicate: " + str(index))
                source_by_index[index] = recovery_dispatch["source_commit"]
    scheduler_by_index = merge_scheduler_evidence(scheduler_by_index)
    if any(index not in source_by_index for index in scheduler_by_index):
        raise OperationalTwinProtocolError("candidate_scheduler_dispatch_mismatch")
    execution = dict(resolved.execution)
    candidate, assignment_manifest, command_catalog, preflight = (
        load_candidate_protocol_artifacts(config, root, resolved_context=resolved)
    )
    commands = _commands_by_index(command_catalog)
    eligibility_by_assignment = {
        str(row["assignment_uid"]): dict(row)
        for row in recovery_eligibility["assignments"]
    }
    assignment_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    blockers: list[dict[str, str]] = []
    for assignment_value in assignment_manifest["assignments"]:
        assignment = dict(assignment_value)
        eligibility = eligibility_by_assignment.get(str(assignment["assignment_uid"]))
        if eligibility is None:
            blockers.append(
                _block(
                    "candidate_recovery_eligibility_assignment_missing",
                    f"{assignment['assignment_uid']} lacks recovery classification.",
                )
            )
            eligibility = {
                "assignment_uid": assignment["assignment_uid"],
                "recovery_eligibility": "BLOCKED_INVALID_PREDECESSOR",
                "initial_attempt_uid": assignment["attempts"][0]["attempt_uid"],
            }
        local_slots: list[dict[str, object]] = []
        for attempt_value in assignment["attempts"]:
            attempt = dict(attempt_value)
            command = next(
                row
                for row in commands.values()
                if row["attempt_uid"] == attempt["attempt_uid"]
            )
            index = int(command["command_index"])
            attempt_root = repo_path(root, str(attempt["output_root"]))
            if not attempt_root.exists() and attempt["conditional"] is True:
                if eligibility["recovery_eligibility"] == "ELIGIBLE":
                    slot = {
                        **_unmaterialized_recovery_slot(
                            assignment,
                            attempt,
                            command,
                            eligibility,
                        ),
                        "status": "FAILED",
                        "failure_class": "eligible_recovery_not_materialized",
                        "validation_blockers": [
                            _block(
                                "candidate_eligible_recovery_missing",
                                f"Eligible recovery {attempt['attempt_uid']} was not executed.",
                            )
                        ],
                    }
                    local_slots.append(slot)
                    blockers.extend(slot["validation_blockers"])
                else:
                    local_slots.append(
                        _unmaterialized_recovery_slot(
                            assignment,
                            attempt,
                            command,
                            eligibility,
                        )
                    )
                continue
            scheduler = scheduler_by_index.get(index)
            if scheduler is None:
                slot = _empty_attempt_slot(assignment, attempt, command)
                slot.update(
                    materialized=attempt_root.exists(),
                    failure_class="missing_scheduler_terminal",
                    validation_blockers=[
                        _block(
                            "candidate_scheduler_terminal_missing",
                            f"Materialized attempt {attempt['attempt_uid']} lacks scheduler evidence.",
                        )
                    ],
                )
                local_slots.append(slot)
                blockers.extend(slot["validation_blockers"])
                continue
            slot, slot_blockers = validate_materialized_attempt(
                config,
                root,
                assignment=assignment,
                attempt=attempt,
                command=command,
                scheduler_evidence=scheduler,
                source_commit=str(source_by_index[index]),
                candidate_manifest=candidate,
                assignment_manifest=assignment_manifest,
                command_catalog=command_catalog,
                preflight=preflight,
                resolved_context=resolved,
            )
            local_slots.append(slot)
            blockers.extend(slot_blockers)
        local_slots.sort(key=lambda row: int(row["attempt_ordinal"]))
        slot_rows.extend(local_slots)
        materialized = [row for row in local_slots if row["materialized"] is True]
        final_slot = materialized[-1] if materialized else local_slots[0]
        assignment_rows.append(
            {
                "candidate_lineage_uid": assignment["candidate_lineage_uid"],
                "protocol_uid": assignment["protocol_uid"],
                "assignment_uid": assignment["assignment_uid"],
                "method_id": assignment["method_id"],
                "seed": assignment["seed"],
                "final_retained_attempt_uid": final_slot["attempt_uid"],
                "final_retained_status": final_slot["status"],
                "recovery_eligibility": eligibility["recovery_eligibility"],
                "attempt_uids": [row["attempt_uid"] for row in local_slots],
                "validation_blockers": [
                    blocker
                    for row in local_slots
                    for blocker in row.get("validation_blockers", [])
                ],
            }
        )
    extras = _extra_attempt_roots(root, assignment_manifest)
    if extras:
        blockers.append(
            _block(
                "candidate_extra_attempt_root",
                "Undeclared candidate attempt roots exist: " + ", ".join(extras),
            )
        )
    if len(assignment_rows) != 15:
        blockers.append(
            _block(
                "candidate_execution_assignment_count_mismatch",
                f"Execution catalog contains {len(assignment_rows)} assignments instead of 15.",
            )
        )
    if len(slot_rows) != 30:
        blockers.append(
            _block(
                "candidate_execution_slot_count_mismatch",
                f"Execution catalog contains {len(slot_rows)} slots instead of 30.",
            )
        )
    for row in [*assignment_rows, *slot_rows]:
        row["successor_execution_binding"] = dict(resolved.binding)
    terminal = _catalog_terminal_decision(execution, blockers)
    return {
        "schema_version": EXECUTION_CATALOG_SCHEMA_VERSION,
        "successor_execution_binding": dict(resolved.binding),
        "status": "PASS" if not blockers else "FAIL",
        "terminal_decision": terminal,
        "candidate_lineage_uid": execution["candidate_lineage_uid"],
        "protocol_uid": execution["protocol_uid"],
        "preflight_uid": execution["preflight_uid"],
        "source_commit": dispatch_manifest["source_commit"],
        "candidate_manifest_hash": candidate["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "dispatch_manifest": None,
        "recovery_dispatch_manifest": None,
        "recovery_eligibility": None,
        "assignment_count": len(assignment_rows),
        "attempt_slot_count": len(slot_rows),
        "assignments": assignment_rows,
        "attempt_slots": slot_rows,
        "extra_attempt_roots": extras,
        "validation_blockers": blockers,
        "access_counters": execution["access_counters"],
        "closed_flags": execution["closed_flags"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_bound_catalog_artifact(
    root: Path,
    binding: object,
    *,
    expected_path: str,
    label: str,
    blockers: list[dict[str, str]],
) -> dict[str, object] | None:
    code_prefix = f"candidate_execution_{label}"
    if not isinstance(binding, Mapping) or set(binding) != ARTIFACT_IDENTITY_KEYS:
        blockers.append(
            _block(
                f"{code_prefix}_binding_invalid",
                f"{label} artifact binding keyset is not exact.",
            )
        )
        return None
    if binding.get("path") != expected_path:
        blockers.append(
            _block(
                f"{code_prefix}_path_mismatch",
                f"{label} artifact path differs from the execution contract.",
            )
        )
        return None
    path = repo_path(root, expected_path, must_exist=False)
    if not path.is_file():
        blockers.append(
            _block(
                f"{code_prefix}_missing",
                f"{label} artifact is absent: {expected_path}.",
            )
        )
        return None
    if dict(binding) != artifact_identity(root, path):
        blockers.append(
            _block(
                f"{code_prefix}_hash_mismatch",
                f"{label} artifact bytes differ from the catalog binding.",
            )
        )
    try:
        return _read_json(path)
    except (json.JSONDecodeError, OperationalTwinProtocolError) as error:
        blockers.append(_block(f"{code_prefix}_invalid", str(error)))
        return None


def _catalog_terminal_decision(
    execution: Mapping[str, object],
    retained_blockers: Sequence[Mapping[str, object]],
) -> str:
    terminal_policy = dict(execution["terminal_policy"])
    if any(
        blocker.get("code") in INVALID_EXECUTION_BLOCKER_CODES
        for blocker in retained_blockers
    ):
        return str(terminal_policy["invalid"])
    if retained_blockers:
        return str(terminal_policy["incomplete"])
    return str(terminal_policy["positive"])


def validate_execution_catalog(
    config: Mapping[str, object],
    root: Path,
    catalog: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
    execution = dict(resolved.execution)
    candidate, assignment_manifest, command_catalog, preflight = (
        load_candidate_protocol_artifacts(config, root, resolved_context=resolved)
    )
    required_keys = {
        "schema_version",
        "successor_execution_binding",
        "recovery_dispatch_manifest",
        "status",
        "terminal_decision",
        "candidate_lineage_uid",
        "protocol_uid",
        "preflight_uid",
        "source_commit",
        "candidate_manifest_hash",
        "assignment_manifest_hash",
        "command_catalog_hash",
        "dispatch_manifest",
        "recovery_eligibility",
        "assignment_count",
        "attempt_slot_count",
        "assignments",
        "attempt_slots",
        "extra_attempt_roots",
        "validation_blockers",
        "access_counters",
        "closed_flags",
        "created_at",
    }
    if set(catalog) != required_keys:
        blockers.append(
            _block(
                "candidate_execution_catalog_keyset_mismatch",
                "Execution catalog keyset is not exact.",
            )
        )
    if catalog.get("schema_version") != EXECUTION_CATALOG_SCHEMA_VERSION:
        blockers.append(
            _block(
                "candidate_execution_catalog_schema_mismatch",
                "Execution catalog schema version drifted.",
            )
        )

    frozen_identity = {
        "successor_execution_binding": dict(resolved.binding),
        "candidate_lineage_uid": candidate["candidate_lineage_uid"],
        "protocol_uid": candidate["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": candidate["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
    }
    for field, expected in frozen_identity.items():
        if catalog.get(field) != expected:
            blockers.append(
                _block(
                    f"candidate_execution_catalog_{field}_mismatch",
                    f"Execution catalog {field} differs from the frozen authority.",
                )
            )

    source_commit = catalog.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        blockers.append(
            _block(
                "candidate_execution_catalog_source_commit_invalid",
                "Execution catalog source commit must be a non-empty string.",
            )
        )
    output_paths = dict(execution["output_paths"])
    dispatch_path = str(output_paths["dispatch_manifest"])
    recovery_path = str(output_paths["recovery_eligibility"])
    dispatch = _load_bound_catalog_artifact(
        root,
        catalog.get("dispatch_manifest"),
        expected_path=dispatch_path,
        label="dispatch_manifest",
        blockers=blockers,
    )
    recovery = _load_bound_catalog_artifact(
        root,
        catalog.get("recovery_eligibility"),
        expected_path=recovery_path,
        label="recovery_eligibility",
        blockers=blockers,
    )
    recovery_dispatch = None
    recovery_dispatch_path = str(output_paths["recovery_dispatch_manifest"])
    if catalog.get("recovery_dispatch_manifest") is not None:
        recovery_dispatch = _load_bound_catalog_artifact(
            root, catalog["recovery_dispatch_manifest"],
            expected_path=recovery_dispatch_path, label="recovery_dispatch_manifest", blockers=blockers,
        )
    elif (root / recovery_dispatch_path).exists():
        blockers.append(_block("candidate_recovery_dispatch_binding_missing", "Recovery publication omitted."))
    common_identity = {
        "successor_execution_binding": dict(resolved.binding),
        "candidate_lineage_uid": frozen_identity["candidate_lineage_uid"],
        "protocol_uid": frozen_identity["protocol_uid"],
        "preflight_uid": frozen_identity["preflight_uid"],
        "source_commit": source_commit,
        "access_counters": execution["access_counters"],
        "closed_flags": execution["closed_flags"],
    }
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import require_candidate_dispatch_manifest
    for mode, publication in (("initial", dispatch), ("recovery", recovery_dispatch)):
        if publication is not None:
            try:
                require_candidate_dispatch_manifest(config, root, publication, mode=mode, resolved_context=resolved)
            except (KeyError, TypeError, ValueError, OSError) as error:
                blockers.append(_block("candidate_execution_dispatch_invalid", str(error)))
    if recovery is not None:
        try:
            require_recovery_eligibility(config, root, root / recovery_path, resolved_context=resolved)
        except (KeyError, TypeError, ValueError, OSError) as error:
            blockers.append(_block("candidate_execution_recovery_recomputation_failed", str(error)))
    if dispatch is not None:
        for field, expected in {
            "schema_version": DISPATCH_MANIFEST_SCHEMA_VERSION,
            **common_identity,
            "frozen_artifacts": execution["frozen_artifacts"],
        }.items():
            if dispatch.get(field) != expected:
                blockers.append(
                    _block(
                        "candidate_execution_dispatch_manifest_identity_mismatch",
                        f"Dispatch manifest {field} drifted.",
                    )
                )
        if dispatch.get("status") not in {
            "INITIAL_ARRAYS_SUBMITTED",
            "PARTIAL_SUBMISSION_FAILED",
        }:
            blockers.append(
                _block(
                    "candidate_execution_dispatch_manifest_status_invalid",
                    "Dispatch manifest is not in a complete submission state.",
                )
            )
    if recovery is not None:
        for field, expected in {
            "schema_version": RECOVERY_ELIGIBILITY_SCHEMA_VERSION,
            "status": "PASS",
            "terminal_decision": "OPERATIONAL_TWIN_CANDIDATE_RECOVERY_ELIGIBILITY_RESOLVED",
            **common_identity,
            "assignment_count": len(assignment_manifest["assignments"]),
        }.items():
            if recovery.get(field) != expected:
                blockers.append(
                    _block(
                        "candidate_execution_recovery_identity_mismatch",
                        f"Recovery eligibility {field} drifted.",
                    )
                )
        bound_dispatch = recovery.get("dispatch_manifest")
        if not isinstance(bound_dispatch, Mapping) or dispatch is None:
            blockers.append(
                _block(
                    "candidate_execution_recovery_dispatch_binding_invalid",
                    "Recovery eligibility lacks its initial dispatch binding.",
                )
            )
        else:
            for field in (
                "schema_version",
                "candidate_lineage_uid",
                "protocol_uid",
                "preflight_uid",
                "source_commit",
                "frozen_artifacts",
                "initial_arrays",
                "access_counters",
                "closed_flags",
            ):
                if bound_dispatch.get(field) != dispatch.get(field):
                    blockers.append(
                        _block(
                            "candidate_execution_recovery_dispatch_binding_invalid",
                            f"Recovery dispatch binding {field} drifted.",
                        )
                    )

    manifest_assignments = [dict(row) for row in assignment_manifest["assignments"]]
    expected_assignment_uids = [
        str(row["assignment_uid"]) for row in manifest_assignments
    ]
    commands_by_attempt = {
        str(row["attempt_uid"]): dict(row) for row in command_catalog["commands"]
    }
    expected_recovery_rows: list[dict[str, object]] = []
    for assignment in manifest_assignments:
        attempts = [dict(value) for value in assignment["attempts"]]
        initial = next(value for value in attempts if value["attempt_kind"] == "initial")
        retry = next(
            value
            for value in attempts
            if value["attempt_kind"] == "infrastructure_recovery_01"
        )
        expected_recovery_rows.append(
            {
                "assignment_uid": assignment["assignment_uid"],
                "method_id": assignment["method_id"],
                "seed": assignment["seed"],
                "initial_attempt_uid": initial["attempt_uid"],
                "initial_command_index": commands_by_attempt[str(initial["attempt_uid"])][
                    "command_index"
                ],
                "recovery_attempt_uid": retry["attempt_uid"],
                "recovery_command_index": commands_by_attempt[str(retry["attempt_uid"])][
                    "command_index"
                ],
            }
        )
    recovery_rows = recovery.get("assignments") if recovery is not None else None
    if not isinstance(recovery_rows, list) or len(recovery_rows) != len(
        expected_recovery_rows
    ):
        blockers.append(
            _block(
                "candidate_execution_recovery_assignment_count_mismatch",
                "Recovery eligibility must contain the exact frozen assignments.",
            )
        )
    else:
        if [row.get("assignment_uid") for row in recovery_rows if isinstance(row, Mapping)] != expected_assignment_uids:
            blockers.append(
                _block(
                    "candidate_execution_recovery_assignment_identity_mismatch",
                    "Recovery assignment identities or order differ from the frozen manifest.",
                )
            )
        for actual, expected in zip(recovery_rows, expected_recovery_rows, strict=True):
            if not isinstance(actual, Mapping):
                blockers.append(
                    _block(
                        "candidate_execution_recovery_assignment_invalid",
                        "A recovery assignment row is not an object.",
                    )
                )
                continue
            if any(actual.get(field) != value for field, value in expected.items()):
                blockers.append(
                    _block(
                        "candidate_execution_recovery_assignment_identity_mismatch",
                        f"Recovery assignment {expected['assignment_uid']} drifted.",
                    )
                )
            if actual.get("recovery_eligibility") not in RECOVERY_CLASSIFICATIONS:
                blockers.append(
                    _block(
                        "candidate_execution_recovery_classification_invalid",
                        f"Recovery assignment {expected['assignment_uid']} has an invalid classification.",
                    )
                )

    expected_assignment_count = len(manifest_assignments)
    expected_attempt_count = sum(
        len(assignment["attempts"]) for assignment in manifest_assignments
    )
    assignments = catalog.get("assignments")
    slots = catalog.get("attempt_slots")
    if (
        not isinstance(assignments, list)
        or len(assignments) != expected_assignment_count
        or catalog.get("assignment_count") != expected_assignment_count
    ):
        blockers.append(
            _block(
                "candidate_execution_assignment_count_mismatch",
                f"Execution catalog must contain exactly {expected_assignment_count} assignments.",
            )
        )
    elif len({row.get("assignment_uid") for row in assignments if isinstance(row, Mapping)}) != len(assignments):
        blockers.append(
            _block(
                "candidate_execution_assignment_duplicate",
                "Execution catalog contains duplicate assignment UIDs.",
            )
        )
    if (
        not isinstance(slots, list)
        or len(slots) != expected_attempt_count
        or catalog.get("attempt_slot_count") != expected_attempt_count
    ):
        blockers.append(
            _block(
                "candidate_execution_slot_count_mismatch",
                f"Execution catalog must contain exactly {expected_attempt_count} attempt slots.",
            )
        )
    elif len({row.get("attempt_uid") for row in slots if isinstance(row, Mapping)}) != len(slots):
        blockers.append(
            _block(
                "candidate_execution_attempt_duplicate",
                "Execution catalog contains duplicate attempt UIDs.",
            )
        )

    expected_catalog: dict[str, object] | None = None
    if dispatch is not None and recovery is not None:
        try:
            scheduler = load_scheduler_receipts(
                root,
                execution,
                [dict(row) for row in dispatch.get("initial_arrays", [])],
                phase="initial",
                source_commit=str(source_commit),
                resolved_context=resolved,
            )
            recovery_arrays = [
                dict(row) for row in (recovery_dispatch or {}).get("recovery_arrays", [])
            ]
            if recovery_arrays:
                scheduler = merge_scheduler_evidence(
                    scheduler, load_scheduler_receipts(
                        root,
                        execution,
                        recovery_arrays,
                        phase="recovery",
                        source_commit=str(recovery_dispatch["source_commit"]),
                        resolved_context=resolved,
                    )
                )
            expected_catalog = build_execution_catalog(
                config,
                root,
                dispatch_manifest=dispatch,
                recovery_eligibility=recovery,
                scheduler_by_index=scheduler,
                resolved_context=resolved,
            )
            expected_catalog["dispatch_manifest"] = artifact_identity(
                root, repo_path(root, dispatch_path, must_exist=True)
            )
            expected_catalog["recovery_eligibility"] = artifact_identity(
                root, repo_path(root, recovery_path, must_exist=True)
            )
            expected_catalog["recovery_dispatch_manifest"] = (
                artifact_identity(root, root / recovery_dispatch_path) if recovery_dispatch is not None else None
            )
            expected_catalog["created_at"] = catalog.get("created_at")
        except (KeyError, TypeError, ValueError, OperationalTwinProtocolError) as error:
            blockers.append(
                _block(
                    "candidate_execution_catalog_recomputation_failed",
                    f"Execution catalog could not be recomputed: {error}",
                )
            )
    if expected_catalog is not None:
        expected_assignments = expected_catalog["assignments"]
        expected_slots = expected_catalog["attempt_slots"]
        if assignments != expected_assignments:
            blockers.append(
                _block(
                    "candidate_execution_assignment_rows_mismatch",
                    "Execution assignment rows do not match the frozen evidence recomputation.",
                )
            )
        if slots != expected_slots:
            blockers.append(
                _block(
                    "candidate_execution_attempt_rows_mismatch",
                    "Execution attempt rows do not match the frozen evidence recomputation.",
                )
            )
        for field in (
            "source_commit",
            "successor_execution_binding",
            "recovery_dispatch_manifest",
            "extra_attempt_roots",
            "validation_blockers",
            "access_counters",
            "closed_flags",
        ):
            if catalog.get(field) != expected_catalog.get(field):
                blockers.append(
                    _block(
                        f"candidate_execution_catalog_{field}_mismatch",
                        f"Execution catalog {field} does not recompute from retained evidence.",
                    )
                )
        if catalog.get("status") != expected_catalog.get("status"):
            blockers.append(
                _block(
                    "candidate_execution_status_invalid",
                    "Execution catalog status does not match recomputed evidence.",
                )
            )
        if catalog.get("terminal_decision") != expected_catalog.get(
            "terminal_decision"
        ):
            blockers.append(
                _block(
                    "candidate_execution_terminal_invalid",
                    "Execution catalog terminal does not match recomputed evidence.",
                )
            )

    retained_blockers = catalog.get("validation_blockers")
    if not isinstance(retained_blockers, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"code", "message"}
        or not isinstance(row.get("code"), str)
        or not isinstance(row.get("message"), str)
        for row in retained_blockers
    ):
        blockers.append(
            _block(
                "candidate_execution_catalog_blockers_invalid",
                "Execution catalog validation_blockers are malformed.",
            )
        )
        retained_blockers = []
    expected_status = "PASS" if not retained_blockers else "FAIL"
    if catalog.get("status") != expected_status:
        blockers.append(
            _block(
                "candidate_execution_status_invalid",
                "Execution catalog status does not match its retained blockers.",
            )
        )
    if catalog.get("terminal_decision") != _catalog_terminal_decision(
        execution, retained_blockers
    ):
        blockers.append(
            _block(
                "candidate_execution_terminal_invalid",
                "Execution catalog terminal does not match its retained blockers.",
            )
        )
    return blockers

def build_execution_audit(catalog: Mapping[str, object]) -> dict[str, object]:
    slots = [dict(row) for row in catalog["attempt_slots"]]
    return {
        "schema_version": EXECUTION_AUDIT_SCHEMA_VERSION,
        "successor_execution_binding": catalog["successor_execution_binding"],
        "status": catalog["status"],
        "terminal_decision": catalog["terminal_decision"],
        "candidate_lineage_uid": catalog["candidate_lineage_uid"],
        "protocol_uid": catalog["protocol_uid"],
        "source_commit": catalog["source_commit"],
        "assignment_count": catalog["assignment_count"],
        "attempt_slot_count": catalog["attempt_slot_count"],
        "materialized_attempt_count": sum(bool(row["materialized"]) for row in slots),
        "completed_attempt_count": sum(row["status"] == "COMPLETED" for row in slots),
        "failed_attempt_count": sum(row["status"] == "FAILED" for row in slots),
        "unmaterialized_recovery_count": sum(
            row["status"] == "NOT_MATERIALIZED_NOT_ELIGIBLE" for row in slots
        ),
        "validation_blocker_count": len(catalog["validation_blockers"]),
        "validation_blockers": catalog["validation_blockers"],
        "access_counters": catalog["access_counters"],
        "closed_flags": catalog["closed_flags"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def render_execution_report(
    catalog: Mapping[str, object], audit: Mapping[str, object]
) -> str:
    return "\n".join(
        (
            "# Operational Twin candidate-lineage execution",
            "",
            f"- Terminal: `{catalog['terminal_decision']}`",
            f"- Candidate lineage: `{catalog['candidate_lineage_uid']}`",
            f"- Source commit: `{catalog['source_commit']}`",
            f"- Assignments: `{catalog['assignment_count']}`",
            f"- Frozen attempt slots: `{catalog['attempt_slot_count']}`",
            f"- Materialized attempts: `{audit['materialized_attempt_count']}`",
            f"- Completed attempts: `{audit['completed_attempt_count']}`",
            f"- Failed attempts: `{audit['failed_attempt_count']}`",
            f"- Unmaterialized ineligible recovery slots: `{audit['unmaterialized_recovery_count']}`",
            f"- Validation blockers: `{audit['validation_blocker_count']}`",
            "- Sufficient training: `false` (not adjudicated in this execution round)",
            "- Scientific Twin result: `NOT_RUN`",
            "- Sealed/protected/six-context access: `0`",
            "",
        )
    )
