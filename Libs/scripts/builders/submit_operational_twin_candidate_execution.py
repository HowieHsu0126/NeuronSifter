"""Submit successor candidate arrays under one exclusive, review-bound owner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    resolve_candidate_execution_protocol,
    require_clean_tracked_source,
    require_candidate_runtime_ownership,
    build_candidate_pre_dispatch_audit,
    validate_candidate_calibration_execution_authority,
    write_json_exclusive,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    OperationalTwinProtocolError,
    ResolvedCandidateExecutionProtocol,
    candidate_attempt_path,
    candidate_execution_artifact_paths,
    sha256_text,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


DISPATCH_SCHEMA_VERSION = "neuronsifter-operational-twin-candidate-dispatch-manifest-v2"


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
            f"candidate_submission_command_failed: {' '.join(argv)}"
        ) from error
    if not completed.stdout.strip() and argv[:2] != ["scontrol", "release"]:
        raise OperationalTwinProtocolError(
            f"candidate_submission_command_empty: {' '.join(argv)}"
        )
    return completed.stdout.strip()



def _index_spec(indices: Sequence[int]) -> str:
    values = sorted({int(value) for value in indices})
    if not values:
        raise OperationalTwinProtocolError(
            "candidate_submission_indices_empty: no command index was selected."
        )
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(value) for value in values)



def _job_name(method_id: str, mode: str) -> str:
    aliases = {
        "persistence_no_action": "persist",
        "linear_or_delta": "linear",
        "same_backbone_no_action": "noaction",
        "deterministic_action_conditioned": "detaction",
        "full_uncertainty_aware_action_conditioned": "fulluq",
    }
    return f"NrnSift_otcand_{aliases[method_id]}_{mode}"



def worker_binding(root: Path, resolved: ResolvedCandidateExecutionProtocol, source_commit: str) -> dict[str, str]:
    return {
        "NRNSIFT_HANDOFF_UID": resolved.binding["handoff_uid"],
        "NRNSIFT_SOURCE_COMMIT": source_commit,
        "NRNSIFT_SOURCE_IDENTITY": resolved.binding["executing_source_identity_hash"],
        "NRNSIFT_COMMAND_CATALOG_HASH": resolved.command_catalog["command_catalog_hash"],
        "NRNSIFT_CANDIDATE_LINEAGE_UID": resolved.protocol["candidate_lineage_uid"],
        "NRNSIFT_PROTOCOL_UID": resolved.protocol["protocol_uid"],
        "NRNSIFT_CONFIG_PATH": str(root / "Libs/configs/config_stack.yaml"),
    }


def _wrapper_command(*, gpu_requested: bool, binding: Mapping[str, str]) -> str:
    lines = [
        "set -euo pipefail",
        'task_tmp=$(mktemp -d "/tmp/NrnSift_otcand_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXXXX")',
        "trap 'rm -rf -- \"$task_tmp\"' EXIT",
        'export TMPDIR="$task_tmp"',
        "export PYTHONPATH=Libs:.",
        'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        'export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
        'export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"',
    ]
    if not gpu_requested:
        lines.append('export CUDA_VISIBLE_DEVICES=""')
    lines.extend("export " + name + "=" + shlex.quote(value) for name, value in sorted(binding.items()))
    lines.append('python -m Libs.exp.training.run_operational_twin_candidate_attempt --task-index "$SLURM_ARRAY_TASK_ID"')
    return "\n".join(lines)


def build_array_submission_argv(
    root: Path, execution: Mapping[str, object], group: Mapping[str, object],
    *, mode: str, indices: Sequence[int], binding: Mapping[str, str],
) -> list[str]:
    if mode not in {"initial", "recovery"}:
        raise OperationalTwinProtocolError("candidate_submission_mode_invalid")
    indices = list(indices)
    allowed = group[mode + "_command_indices"]
    if not indices or indices != sorted(set(indices)) or not set(indices).issubset(allowed):
        raise OperationalTwinProtocolError("candidate_submission_indices_invalid")
    log_root = root / execution["output_paths"]["execution_root"] / "logs"
    job_name = _job_name(str(group["method_id"]), mode)
    argv = [
        "sbatch", "--parsable", "--hold", f"--array={_index_spec(indices)}",
        f"--partition={group['partition']}", f"--nodelist={group['nodelist']}",
        f"--job-name={job_name}", f"--cpus-per-task={group['cpus_per_task']}",
        f"--mem={group['memory']}", f"--time={group['wall_time']}", f"--chdir={root}",
        f"--output={log_root}/{job_name}_%A_%a.out",
        f"--error={log_root}/{job_name}_%A_%a.err",
    ]
    if group["gres"] is not None:
        argv.append(f"--gres={group['gres']}")
    argv.extend(("--wrap", _wrapper_command(gpu_requested=group["gres"] is not None, binding=binding)))
    if group["exclusive"] is not False:
        raise OperationalTwinProtocolError("candidate_submission_exclusive_forbidden")
    return argv


def _required_attempts_absent(root: Path, resolved: ResolvedCandidateExecutionProtocol, indices: Sequence[int]) -> None:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    selected = set(indices)
    for row in resolved.command_catalog["commands"]:
        if row["command_index"] not in selected:
            continue
        assignment = next(item for item in resolved.assignment_manifest["assignments"] if item["assignment_uid"] == row["assignment_uid"])
        attempt = next(item for item in assignment["attempts"] if item["attempt_uid"] == row["attempt_uid"])
        candidate_attempt_path(root, attempt["output_root"], require_absent=True)
        for path in candidate_execution_artifact_paths(resolved.protocol, assignment, attempt).values():
            _path(root, path, absent=True)


def prepare_submission(
    config: Mapping[str, object], root: Path, *, mode: str,
    recovery_eligibility_path: Path | None = None, require_clean_source: bool = True,
) -> dict[str, object]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    from Libs.utils.evaluation.operational_twin_candidate_evidence import require_recovery_eligibility

    if mode not in {"initial", "recovery"}:
        raise OperationalTwinProtocolError("candidate_submission_mode_invalid")
    resolved = resolve_candidate_execution_protocol(config, root)
    source_commit = require_clean_tracked_source(root) if require_clean_source else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    execution = resolved.execution
    paths = execution["output_paths"]
    calibration_path = _path(root, paths["calibration_execution_authority"])
    blockers = validate_candidate_calibration_execution_authority(
        config, root, calibration_path, executing_source_commit=source_commit, resolved_context=resolved)
    if blockers:
        raise OperationalTwinProtocolError("; ".join(row["code"] + ": " + row["message"] for row in blockers))
    calibration = json.loads(calibration_path.read_text())
    require_candidate_runtime_ownership(config, root, resolved)
    if mode == "initial":
        if recovery_eligibility_path is not None:
            raise OperationalTwinProtocolError("candidate_initial_recovery_evidence_forbidden")
        retained_pre_dispatch = json.loads(_path(root, paths["pre_dispatch_audit"]).read_text())
        expected_pre_dispatch = build_candidate_pre_dispatch_audit(config, root)
        expected_pre_dispatch["created_at"] = retained_pre_dispatch.get("created_at")
        if retained_pre_dispatch != expected_pre_dispatch or expected_pre_dispatch["status"] != "PASS":
            raise OperationalTwinProtocolError("candidate_pre_dispatch_audit_invalid")
        _required_attempts_absent(root, resolved, list(range(30)))
        selected = list(range(15))
    else:
        canonical = _path(root, paths["recovery_eligibility"])
        if recovery_eligibility_path is not None and recovery_eligibility_path.absolute() != canonical:
            raise OperationalTwinProtocolError("candidate_recovery_path_not_canonical")
        recovery = require_recovery_eligibility(config, root, canonical, resolved_context=resolved)
        selected = [row["recovery_command_index"] for row in recovery["assignments"]
                    if row["recovery_eligibility"] == "ELIGIBLE"]
        _required_attempts_absent(root, resolved, selected)
    mode_path = paths["dispatch_manifest" if mode == "initial" else "recovery_dispatch_manifest"]
    claim_path = paths["dispatch_claim" if mode == "initial" else "recovery_dispatch_claim"]
    _path(root, mode_path, absent=True)
    _path(root, claim_path, absent=True)
    groups = [
        {**group, "selected_indices": [index for index in group[mode + "_command_indices"] if index in selected]}
        for group in execution["array_groups"]
    ]
    groups = [group for group in groups if group["selected_indices"]]
    if [index for group in groups for index in group["selected_indices"]] != selected:
        raise OperationalTwinProtocolError("candidate_submission_group_mapping_invalid")
    return {
        "resolved_protocol": resolved, "execution": execution,
        "source_commit": source_commit, "mode": mode, "groups": groups,
        "selected_indices": selected, "execution_selection": calibration["execution_selection"],
        "calibration_execution_authority": _artifact_identity(root, calibration_path),
        "binding": worker_binding(root, resolved, source_commit),
    }


def _artifact_identity(root: Path, path: Path) -> dict[str, object]:
    from Libs.utils.evaluation.operational_twin_candidate_evidence import artifact_identity
    return artifact_identity(root, path)


def _validate_submission_sinfo(stdout: str) -> None:
    rows = [row.split("|") for row in stdout.splitlines() if row.strip()]
    required = {("cpu04", "cpu"), ("gpu03", "gpu1")}
    observed: set[tuple[str, str]] = set()
    for row in rows:
        if len(row) != 4 or not row[0]:
            continue
        key = (row[0], row[1].rstrip("*"))
        if key in required:
            state = row[2].lower().rstrip("*~#+")
            if any(token in state for token in ("down", "drain", "fail", "maint")):
                raise OperationalTwinProtocolError(
                    f"candidate_submission_node_unavailable: {row[0]}/{row[1]} is {row[2]}."
                )
            if key == ("gpu03", "gpu1") and "gpu" not in row[3].lower():
                raise OperationalTwinProtocolError(
                    "candidate_submission_gpu_unavailable: gpu03 does not advertise a GPU."
                )
            observed.add(key)
    if observed != required:
        raise OperationalTwinProtocolError(
            "candidate_submission_node_unavailable: cpu04/cpu and gpu03/gpu1 are both required."
        )



def require_candidate_dispatch_manifest(
    config: Mapping[str, object], root: Path, payload: Mapping[str, object],
    *, mode: str, resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    """Reconstruct each submission from reviewed source, selection and owner."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    from Libs.utils.evaluation.operational_twin_training_protocol import _require_executing_commit
    from Libs.utils.evaluation.operational_twin_candidate_execution import require_future_execution_selection
    from Libs.utils.evaluation.operational_twin_candidate_evidence import require_recovery_eligibility

    resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
    execution = resolved.execution
    schema = config["operational_twin_candidate_execution_schema"]
    if set(payload) != set(schema["dispatch_manifest_fields"]):
        raise OperationalTwinProtocolError("candidate_dispatch_keyset_mismatch")
    required = {
        "schema_version": DISPATCH_SCHEMA_VERSION, "mode": mode,
        "candidate_lineage_uid": execution["candidate_lineage_uid"],
        "protocol_uid": execution["protocol_uid"], "preflight_uid": execution["preflight_uid"],
        "successor_execution_binding": dict(resolved.binding),
        "frozen_artifacts": execution["frozen_artifacts"],
        "release_manifest_path": execution["output_paths"]["dispatch_release_manifest" if mode == "initial" else "recovery_dispatch_release_manifest"],
        "access_counters": execution["access_counters"], "closed_flags": execution["closed_flags"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise OperationalTwinProtocolError("candidate_dispatch_binding_mismatch")
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import executing_source_identity
    if executing_source_identity(config, root, resolved.binding["executing_source_commit"]) != resolved.handoff["manifest"]["executing_source"]:
        raise OperationalTwinProtocolError("candidate_dispatch_current_source_drift")
    _require_executing_commit(root, payload["source_commit"], resolved)
    selection = payload["execution_selection"]
    selection_path = _path(root, selection["identities"]["selection_record"]["path"])
    if selection != require_future_execution_selection(config, root, selection_path.parent):
        raise OperationalTwinProtocolError("candidate_dispatch_selection_drift")
    calibration_path = _path(root, execution["output_paths"]["calibration_execution_authority"])
    if payload["calibration_execution_authority"] != _artifact_identity(root, calibration_path):
        raise OperationalTwinProtocolError("candidate_dispatch_calibration_drift")
    calibration_blockers = validate_candidate_calibration_execution_authority(
        config, root, calibration_path, executing_source_commit=payload["source_commit"], resolved_context=resolved)
    if calibration_blockers:
        raise OperationalTwinProtocolError("candidate_dispatch_calibration_invalid: " + "; ".join(row["code"] for row in calibration_blockers))
    calibration = json.loads(calibration_path.read_text())
    if (calibration["execution_selection"] != selection
            or calibration["successor_execution_binding"] != dict(resolved.binding)):
        raise OperationalTwinProtocolError("candidate_dispatch_calibration_binding_mismatch")
    claim_path = _path(root, execution["output_paths"]["dispatch_claim" if mode == "initial" else "recovery_dispatch_claim"])
    if payload["dispatch_claim"] != _artifact_identity(root, claim_path):
        raise OperationalTwinProtocolError("candidate_dispatch_claim_identity_mismatch")
    claim = json.loads(claim_path.read_text())
    if set(claim) != set(schema["dispatch_claim_fields"]):
        raise OperationalTwinProtocolError("candidate_dispatch_claim_keyset_mismatch")
    if claim["schema_version"] != "neuronsifter-operational-twin-candidate-dispatch-claim-v1":
        raise OperationalTwinProtocolError("candidate_dispatch_claim_schema_mismatch")
    for key in ("mode", "source_commit", "successor_execution_binding", "execution_selection", "selected_indices"):
        if claim[key] != payload[key]:
            raise OperationalTwinProtocolError("candidate_dispatch_claim_binding_mismatch")
    if mode == "initial":
        selected = list(range(15))
    elif mode == "recovery":
        recovery = require_recovery_eligibility(
            config, root, _path(root, execution["output_paths"]["recovery_eligibility"]),
            resolved_context=resolved)
        selected = [row["recovery_command_index"] for row in recovery["assignments"] if row["recovery_eligibility"] == "ELIGIBLE"]
    else:
        raise OperationalTwinProtocolError("candidate_dispatch_mode_invalid")
    if payload["selected_indices"] != selected:
        raise OperationalTwinProtocolError("candidate_dispatch_indices_mismatch")
    binding = worker_binding(root, resolved, payload["source_commit"])
    planned = []
    for group in execution["array_groups"]:
        indices = [index for index in group[mode + "_command_indices"] if index in selected]
        if indices:
            planned.append({"group_id": group["group_id"], "method_id": group["method_id"],
                            "command_indices": indices, "submission_argv": build_array_submission_argv(
                                root, execution, group, mode=mode, indices=indices, binding=binding)})
    if claim["groups"] != planned:
        raise OperationalTwinProtocolError("candidate_dispatch_planned_groups_mismatch")
    arrays = payload[mode + "_arrays"]
    if payload["recovery_arrays" if mode == "initial" else "initial_arrays"] != []:
        raise OperationalTwinProtocolError("candidate_dispatch_other_mode_rows_forbidden")
    if not isinstance(arrays, list) or len(arrays) > len(planned):
        raise OperationalTwinProtocolError("candidate_dispatch_array_count_invalid")
    terminal = "INITIAL_ARRAYS_SUBMITTED" if mode == "initial" else "RECOVERY_ARRAYS_SUBMITTED"
    if payload["status"] == terminal:
        if len(arrays) != len(planned) or payload["error"] is not None or payload["submission_failure"] is not None:
            raise OperationalTwinProtocolError("candidate_dispatch_incomplete_success")
    elif payload["status"] != "PARTIAL_SUBMISSION_FAILED" or not payload["error"]:
        raise OperationalTwinProtocolError("candidate_dispatch_status_invalid")
    failure = payload["submission_failure"]
    if payload["status"] == "PARTIAL_SUBMISSION_FAILED":
        if (not isinstance(failure, Mapping)
                or set(failure) != set(schema["submission_failure_fields"])
                or len(arrays) >= len(planned)
                or failure["group_id"] != planned[len(arrays)]["group_id"]
                or failure["submission_argv"] != planned[len(arrays)]["submission_argv"]
                or not isinstance(failure["error"], str) or not failure["error"]):
            raise OperationalTwinProtocolError("candidate_dispatch_submission_failure_invalid")
        if failure["stdout"] is not None and not isinstance(failure["stdout"], str):
            raise OperationalTwinProtocolError("candidate_dispatch_submission_failure_invalid")
        if failure["job_id"] is not None and (failure["stdout"] is None or failure["stdout"].split(";", 1)[0].strip() != failure["job_id"]):
            raise OperationalTwinProtocolError("candidate_dispatch_submission_failure_identity_mismatch")
    job_ids: set[str] = set()
    for row, plan in zip(arrays, planned):
        if set(row) != set(schema["dispatch_array_fields"]) or any(row[key] != value for key, value in plan.items()):
            raise OperationalTwinProtocolError("candidate_dispatch_array_binding_mismatch")
        job_id = row["job_id"]
        if not isinstance(job_id, str) or not job_id.isdigit() or job_id in job_ids:
            raise OperationalTwinProtocolError("candidate_dispatch_job_id_invalid")
        job_ids.add(job_id)
        if row["stdout"].split(";", 1)[0].strip() != job_id or row["job_name"] != _job_name(row["method_id"], mode):
            raise OperationalTwinProtocolError("candidate_dispatch_job_identity_mismatch")
        receipt_path = _path(root, execution["output_paths"]["execution_root"] + "/dispatch_receipts/" + mode + "_" + row["group_id"] + ".json")
        if row["submission_receipt"] != _artifact_identity(root, receipt_path):
            raise OperationalTwinProtocolError("candidate_dispatch_submission_receipt_drift")
        receipt = json.loads(receipt_path.read_text())
        if receipt != {key: value for key, value in row.items() if key != "submission_receipt"}:
            raise OperationalTwinProtocolError("candidate_dispatch_submission_receipt_mismatch")
    sinfo = payload["pre_submission_sinfo"]
    if (set(sinfo) != {"argv", "stdout", "stdout_sha256"}
            or sinfo["argv"] != ["sinfo", "--noheader", "--nodes=cpu04,gpu03", "--format=%N|%P|%T|%G"]
            or sinfo["stdout_sha256"] != sha256_text(sinfo["stdout"])):
        raise OperationalTwinProtocolError("candidate_dispatch_sinfo_binding_invalid")
    _validate_submission_sinfo(sinfo["stdout"])
    return dict(payload)



def _release_intent_payload(
    root: Path, resolved: ResolvedCandidateExecutionProtocol,
    manifest: Mapping[str, object], row: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    mode = manifest["mode"]
    if mode not in {"initial", "recovery"} or row not in manifest[mode + "_arrays"]:
        raise OperationalTwinProtocolError("candidate_release_array_not_owned")
    paths = resolved.execution["output_paths"]
    manifest_path = _path(root, paths["dispatch_manifest" if mode == "initial" else "recovery_dispatch_manifest"])
    intent_path = _path(root, paths["execution_root"] + "/release_intents/" + mode + "_" + row["group_id"] + ".json")
    return intent_path, {
        "schema_version": "neuronsifter-operational-twin-candidate-release-intent-v1",
        "mode": mode, "source_commit": manifest["source_commit"],
        "successor_execution_binding": dict(resolved.binding),
        "dispatch_manifest": _artifact_identity(root, manifest_path),
        "array_submission_receipt": row["submission_receipt"],
        "job_id": row["job_id"], "command_indices": row["command_indices"],
        "argv": ["scontrol", "release", row["job_id"]],
    }


def require_candidate_release_intent(
    config: Mapping[str, object], root: Path, resolved: ResolvedCandidateExecutionProtocol,
    manifest: Mapping[str, object], row: Mapping[str, object],
) -> dict[str, object]:
    """A worker may start after release intent, before the release command returns."""
    path, expected = _release_intent_payload(root, resolved, manifest, row)
    observed = json.loads(path.read_text())
    if (set(observed) != set(config["operational_twin_candidate_execution_schema"]["release_intent_fields"])
            or observed != expected):
        raise OperationalTwinProtocolError("candidate_release_intent_binding_mismatch")
    return _artifact_identity(root, path)


def require_candidate_release_manifest(
    config: Mapping[str, object], root: Path, resolved: ResolvedCandidateExecutionProtocol,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate durable release outcomes without upgrading an unknown result."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    mode = manifest["mode"]
    require_candidate_dispatch_manifest(config, root, manifest, mode=mode, resolved_context=resolved)
    terminal = "INITIAL_ARRAYS_SUBMITTED" if mode == "initial" else "RECOVERY_ARRAYS_SUBMITTED"
    if manifest["status"] != terminal:
        raise OperationalTwinProtocolError("candidate_release_requires_complete_held_submission")
    paths = resolved.execution["output_paths"]
    dispatch_path = _path(root, paths["dispatch_manifest" if mode == "initial" else "recovery_dispatch_manifest"])
    payload = json.loads(_path(root, manifest["release_manifest_path"]).read_text())
    schema = config["operational_twin_candidate_execution_schema"]
    expected = {
        "schema_version": "neuronsifter-operational-twin-candidate-release-manifest-v1",
        "mode": mode, "source_commit": manifest["source_commit"],
        "successor_execution_binding": dict(resolved.binding),
        "dispatch_manifest": _artifact_identity(root, dispatch_path),
    }
    if set(payload) != set(schema["release_manifest_fields"]) or any(payload[key] != value for key, value in expected.items()):
        raise OperationalTwinProtocolError("candidate_release_manifest_binding_mismatch")
    attempts, arrays = payload["attempts"], manifest[mode + "_arrays"]
    if not isinstance(attempts, list) or len(attempts) > len(arrays):
        raise OperationalTwinProtocolError("candidate_release_attempt_count_invalid")
    complete = payload["status"] == "RELEASED"
    if complete:
        if len(attempts) != len(arrays) or payload["error"] is not None:
            raise OperationalTwinProtocolError("candidate_release_incomplete_success")
    elif payload["status"] != "PARTIAL_RELEASE_FAILED" or not isinstance(payload["error"], str) or not payload["error"]:
        raise OperationalTwinProtocolError("candidate_release_status_invalid")
    for index, (attempt, row) in enumerate(zip(attempts, arrays)):
        if set(attempt) != set(schema["release_attempt_fields"]):
            raise OperationalTwinProtocolError("candidate_release_attempt_keyset_mismatch")
        if (attempt["job_id"] != row["job_id"]
                or attempt["argv"] != ["scontrol", "release", row["job_id"]]
                or attempt["intent"] != require_candidate_release_intent(config, root, resolved, manifest, row)):
            raise OperationalTwinProtocolError("candidate_release_attempt_binding_mismatch")
        if attempt["status"] == "RELEASED":
            if attempt["returncode"] != 0 or not isinstance(attempt["stdout"], str) or attempt["error"] is not None:
                raise OperationalTwinProtocolError("candidate_release_success_invalid")
        elif complete or index != len(attempts) - 1 or attempt["status"] not in {"FAILED", "UNKNOWN", "NOT_CALLED"}:
            raise OperationalTwinProtocolError("candidate_release_failure_position_invalid")
        elif attempt["status"] == "FAILED" and (not isinstance(attempt["returncode"], int) or attempt["returncode"] == 0):
            raise OperationalTwinProtocolError("candidate_release_failure_status_invalid")
        if attempt["status"] in {"UNKNOWN", "NOT_CALLED"} and attempt["returncode"] is not None:
            raise OperationalTwinProtocolError("candidate_release_unknown_result_invalid")
        receipt_path = _path(root, paths["execution_root"] + "/release_receipts/" + mode + "_" + row["group_id"] + ".json")
        if attempt["receipt"] is None:
            if complete or index != len(attempts) - 1:
                raise OperationalTwinProtocolError("candidate_release_receipt_missing")
        elif (attempt["receipt"] != _artifact_identity(root, receipt_path)
                or json.loads(receipt_path.read_text()) != {key: value for key, value in attempt.items() if key != "receipt"}):
            raise OperationalTwinProtocolError("candidate_release_receipt_drift")
    return payload


def _release_owned_arrays(
    config: Mapping[str, object], root: Path, resolved: ResolvedCandidateExecutionProtocol,
    manifest_path: Path, *, command_capture: Callable[[list[str], Path], str],
) -> dict[str, object]:
    """Release only reread, durably published IDs; partial outcomes remain immutable."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    initial = json.loads(manifest_path.read_text())
    mode = initial["mode"]
    manifest = require_candidate_dispatch_manifest(
        config, root, initial, mode=mode, resolved_context=resolved)
    expected_status = "INITIAL_ARRAYS_SUBMITTED" if mode == "initial" else "RECOVERY_ARRAYS_SUBMITTED"
    if manifest["status"] != expected_status:
        raise OperationalTwinProtocolError("candidate_release_requires_complete_held_submission")
    manifest_identity = _artifact_identity(root, manifest_path)
    output = _path(root, manifest["release_manifest_path"], absent=True)
    execution_root = resolved.execution["output_paths"]["execution_root"]
    result = {
        "schema_version": "neuronsifter-operational-twin-candidate-release-manifest-v1",
        "mode": mode, "status": "RELEASED", "source_commit": manifest["source_commit"],
        "successor_execution_binding": dict(resolved.binding),
        "dispatch_manifest": manifest_identity, "attempts": [], "error": None,
    }
    try:
        for row in manifest[mode + "_arrays"]:
            reread = require_candidate_dispatch_manifest(
                config, root, json.loads(manifest_path.read_text()), mode=mode, resolved_context=resolved)
            if reread != manifest or _artifact_identity(root, manifest_path) != manifest_identity:
                raise OperationalTwinProtocolError("candidate_release_dispatch_drift")
            intent_path, intent = _release_intent_payload(root, resolved, manifest, row)
            _path(root, intent_path.relative_to(root).as_posix(), absent=True)
            write_json_exclusive(intent_path, intent)
            attempt = {
                "job_id": row["job_id"], "argv": intent["argv"],
                "intent": _artifact_identity(root, intent_path), "status": "NOT_CALLED",
                "stdout": None, "stderr": None, "returncode": None, "error": None, "receipt": None,
            }
            result["attempts"].append(attempt)
            receipt_path = _path(root, execution_root + "/release_receipts/" + mode + "_" + row["group_id"] + ".json", absent=True)
            try:
                attempt["status"] = "UNKNOWN"
                attempt["stdout"] = command_capture(intent["argv"], root)
                attempt.update(status="RELEASED", returncode=0)
            except BaseException as error:
                cause = error.__cause__ or error
                returncode = getattr(cause, "returncode", None)
                attempt.update(status="FAILED" if isinstance(returncode, int) and returncode != 0 else "UNKNOWN",
                               error=f"{type(error).__name__}: {error}", returncode=returncode,
                               stdout=getattr(cause, "stdout", None), stderr=getattr(cause, "stderr", None))
                write_json_exclusive(receipt_path, {key: value for key, value in attempt.items() if key != "receipt"})
                attempt["receipt"] = _artifact_identity(root, receipt_path)
                raise
            write_json_exclusive(receipt_path, {key: value for key, value in attempt.items() if key != "receipt"})
            attempt["receipt"] = _artifact_identity(root, receipt_path)
    except BaseException as error:
        result.update(status="PARTIAL_RELEASE_FAILED", error=f"{type(error).__name__}: {error}")
        write_json_exclusive(output, result)
        raise
    write_json_exclusive(output, result)
    return result


def submit_arrays(
    config: Mapping[str, object], root: Path, *, mode: str,
    recovery_eligibility_path: Path | None = None,
    command_capture: Callable[[list[str], Path], str] = _capture,
    require_clean_source: bool = True,
) -> dict[str, object]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    prepared = prepare_submission(config, root, mode=mode,
                                  recovery_eligibility_path=recovery_eligibility_path,
                                  require_clean_source=require_clean_source)
    if mode == "recovery" and not prepared["selected_indices"]:
        raise OperationalTwinProtocolError("candidate_recovery_no_eligible_assignments")
    resolved, execution = prepared["resolved_protocol"], prepared["execution"]
    paths = execution["output_paths"]
    manifest_path = _path(root, paths["dispatch_manifest" if mode == "initial" else "recovery_dispatch_manifest"], absent=True)
    claim_path = _path(root, paths["dispatch_claim" if mode == "initial" else "recovery_dispatch_claim"], absent=True)
    sinfo_argv = ["sinfo", "--noheader", "--nodes=cpu04,gpu03", "--format=%N|%P|%T|%G"]
    sinfo_stdout = command_capture(sinfo_argv, root)
    _validate_submission_sinfo(sinfo_stdout)
    planned = [{
        "group_id": group["group_id"], "method_id": group["method_id"],
        "command_indices": group["selected_indices"],
        "submission_argv": build_array_submission_argv(
            root, execution, group, mode=mode, indices=group["selected_indices"], binding=prepared["binding"]),
    } for group in prepared["groups"]]
    claim = {
        "schema_version": "neuronsifter-operational-twin-candidate-dispatch-claim-v1",
        "mode": mode, "source_commit": prepared["source_commit"],
        "successor_execution_binding": dict(resolved.binding),
        "execution_selection": prepared["execution_selection"],
        "selected_indices": prepared["selected_indices"], "groups": planned,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_exclusive(claim_path, claim)
    manifest = {
        "schema_version": DISPATCH_SCHEMA_VERSION, "mode": mode,
        "status": "INITIAL_ARRAYS_SUBMITTED" if mode == "initial" else "RECOVERY_ARRAYS_SUBMITTED",
        "candidate_lineage_uid": execution["candidate_lineage_uid"],
        "protocol_uid": execution["protocol_uid"], "preflight_uid": execution["preflight_uid"],
        "source_commit": prepared["source_commit"],
        "successor_execution_binding": dict(resolved.binding),
        "execution_selection": prepared["execution_selection"],
        "calibration_execution_authority": prepared["calibration_execution_authority"],
        "frozen_artifacts": execution["frozen_artifacts"],
        "selected_indices": prepared["selected_indices"],
        "release_manifest_path": paths["dispatch_release_manifest" if mode == "initial" else "recovery_dispatch_release_manifest"],
        "dispatch_claim": _artifact_identity(root, claim_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "pre_submission_sinfo": {"argv": sinfo_argv, "stdout": sinfo_stdout,
                                 "stdout_sha256": sha256_text(sinfo_stdout)},
        "initial_arrays": [], "recovery_arrays": [],
        "access_counters": execution["access_counters"], "closed_flags": execution["closed_flags"],
        "error": None, "submission_failure": None,
    }
    target_key = mode + "_arrays"
    failure = None
    try:
        # The claim is retained even if dispatch stops before any job is returned.
        log_root = _path(root, paths["execution_root"] + "/logs")
        log_root.mkdir(parents=True, exist_ok=True)
        for plan in planned:
            failure = {"group_id": plan["group_id"], "submission_argv": plan["submission_argv"],
                       "stdout": None, "job_id": None, "error": None}
            stdout = command_capture(plan["submission_argv"], root)
            failure["stdout"] = stdout
            job_id = stdout.split(";", 1)[0].strip()
            failure["job_id"] = job_id if job_id.isascii() and job_id.isdigit() else None
            if not job_id.isdigit() or any(row["job_id"] == job_id for row in manifest[target_key]):
                raise OperationalTwinProtocolError("candidate_submission_job_id_invalid")
            row = {**plan, "job_id": job_id, "job_name": _job_name(plan["method_id"], mode),
                   "stdout": stdout, "submitted_at": datetime.now(timezone.utc).isoformat()}
            receipt_path = _path(root, paths["execution_root"] + "/dispatch_receipts/" + mode + "_" + plan["group_id"] + ".json")
            write_json_exclusive(receipt_path, row)
            manifest[target_key].append({**row, "submission_receipt": _artifact_identity(root, receipt_path)})
    except BaseException as error:
        manifest["status"] = "PARTIAL_SUBMISSION_FAILED"
        manifest["error"] = f"{type(error).__name__}: {error}"
        if failure is None:
            first = planned[0] if planned else {"group_id": None, "submission_argv": []}
            failure = {"group_id": first["group_id"], "submission_argv": first["submission_argv"],
                       "stdout": None, "job_id": None, "error": None}
        if failure["stdout"] is None:
            captured_stdout = getattr(error.__cause__ or error, "stdout", None)
            if isinstance(captured_stdout, str):
                failure["stdout"] = captured_stdout
        failure["error"] = manifest["error"]
        manifest["submission_failure"] = failure
        write_json_exclusive(manifest_path, manifest)
        raise
    write_json_exclusive(manifest_path, manifest)
    _release_owned_arrays(config, root, resolved, manifest_path, command_capture=command_capture)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--mode", choices=("initial", "recovery"), required=True)
    parser.add_argument("--recovery-eligibility", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.config.absolute() != REPO_ROOT / "Libs/configs/config_stack.yaml":
        parser.error("candidate execution requires the canonical config stack")
    config = load_config_stack(args.config)
    if args.dry_run:
        prepared = prepare_submission(config, REPO_ROOT, mode=args.mode,
                                      recovery_eligibility_path=args.recovery_eligibility)
        payload = {"status": "PASS", "mode": args.mode, "source_commit": prepared["source_commit"],
                   "selected_indices": prepared["selected_indices"],
                   "submission_argv": [build_array_submission_argv(
                       REPO_ROOT, prepared["execution"], group, mode=args.mode,
                       indices=group["selected_indices"], binding=prepared["binding"])
                       for group in prepared["groups"]]}
        print(json.dumps(payload, sort_keys=True))
        return 0
    manifest = submit_arrays(config, REPO_ROOT, mode=args.mode, recovery_eligibility_path=args.recovery_eligibility)
    print(json.dumps({"status": manifest["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
