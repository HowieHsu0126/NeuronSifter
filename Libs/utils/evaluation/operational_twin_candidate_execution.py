"""Build and validate immutable Operational Twin candidate execution gates."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
from typing import Callable, Mapping

import torch

from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_EXECUTION_AUTHORITY_KEYS,
    ResolvedCandidateExecutionProtocol,
    _candidate_execution_contract,
    _candidate_protocol,
    candidate_execution_binding_keys,
    candidate_runtime_artifact_identity,
    candidate_attempt_path,
    _claim_candidate_attempt_root,
    _candidate_directory_open_flags,
    _open_or_create_candidate_ancestor,
    _validate_retained_candidate_execution_bindings,
    read_repo_json,
    CANDIDATE_RESOURCE_ALLOCATION_KEYS,
    CANDIDATE_RESOURCE_ENVIRONMENT_KEYS,
    CANDIDATE_RESOURCE_RECEIPT_KEYS,
    CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
    CANDIDATE_RESOURCE_SLURM_KEYS,
    OperationalTwinProtocolError,
    _require_candidate_resource_evidence,
    _require_live_candidate_resource_placement,
    candidate_runtime_slurm_identity,
    build_candidate_assignment_manifest,
    build_candidate_lineage_manifest,
    candidate_execution_artifact_paths,
    candidate_fresh_initialization_receipt,
    candidate_source_identity_hash,
    compute_candidate_calibration_authority_uid,
    compute_candidate_execution_authority_uid,
    compute_candidate_resource_receipt_uid,
    load_candidate_protocol_artifacts,
    repo_file_identity,
    repo_path,
    require_candidate_assignment_preflight,
    require_candidate_execution_authority,
    sha256_text,
    validate_candidate_artifacts,
    validate_candidate_protocol_config,
)


EXECUTION_CONFIG_KEY = "operational_twin_candidate_execution"
EXECUTION_SCHEMA_KEY = "operational_twin_candidate_execution_schema"
EXECUTION_CONFIG_SCHEMA_VERSION = "neuronsifter-operational-twin-candidate-execution-v4"


def resolve_candidate_execution_protocol(config: Mapping[str, object], root: Path) -> ResolvedCandidateExecutionProtocol:
    """Resolve the unchanged successor through its independently committed handoff."""
    # Imported at this boundary to keep the handoff below the runtime resolver.
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import (
        require_approved_successor_execution_handoff,
    )
    execution = _execution_config(config)
    if execution.get("schema_version") != EXECUTION_CONFIG_SCHEMA_VERSION:
        raise OperationalTwinProtocolError("candidate_predecessor_execution_closed")
    handoff = require_approved_successor_execution_handoff(config, root)
    protocol = dict(handoff["manifest"]["original_protocol"])
    paths = dict(execution["frozen_artifacts"])
    artifacts = [read_repo_json(root, paths[name]["path"]) for name in
                 ("candidate_lineage_manifest", "assignment_manifest", "assignment_commands", "preflight_authority")]
    binding = {key: deepcopy(handoff[key]) for key in (
        "handoff_uid", "handoff_publication_binding_uid", "handoff_review_identity",
        "handoff_review_commit", "frozen_package_binding_uid", "frozen_source_commit",
        "executing_source_commit", "executing_source_identity_hash",
    )}
    binding["schema_version"] = execution["binding_contract"]["schema_version"]
    resolved = ResolvedCandidateExecutionProtocol(
        protocol=protocol, candidate_manifest=artifacts[0], assignment_manifest=artifacts[1],
        command_catalog=artifacts[2], preflight=artifacts[3], execution=execution,
        handoff=handoff, binding=binding,
    )
    blockers = validate_candidate_execution_config(config, root, resolved_context=resolved)
    if blockers:
        raise OperationalTwinProtocolError("; ".join(row["code"] + ": " + row["message"] for row in blockers))
    require_candidate_worker_binding(root, resolved)
    return resolved


def require_candidate_worker_binding(
    root: Path, resolved: ResolvedCandidateExecutionProtocol, *, required: bool = False,
) -> None:
    """Environment values corroborate, but never create, independently loaded authority."""
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import worker_binding
    expected = worker_binding(root, resolved, _git_commit(root))
    supplied = {name: os.environ[name] for name in expected if name in os.environ}
    if required or supplied:
        if supplied != expected:
            raise OperationalTwinProtocolError("candidate_worker_handoff_binding_mismatch")


def require_candidate_dispatch_binding(
    config: Mapping[str, object],
    root: Path,
    resolved: ResolvedCandidateExecutionProtocol,
    *,
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    slurm: Mapping[str, object] | None = None,
    source_commit: str | None = None,
    validate_current_runtime: bool = True,
) -> dict[str, object]:
    """Join the immutable dispatch and release intent before any resource capture."""
    # The submitter owns publication schemas; this import avoids a runtime cycle.
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
        require_candidate_dispatch_manifest,
        require_candidate_release_intent,
    )
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path

    context = _attempt_context(resolved, str(attempt.get("attempt_uid", "")))
    if context["assignment"] != dict(assignment) or context["attempt"] != dict(attempt):
        raise OperationalTwinProtocolError("candidate_dispatch_frozen_attempt_mismatch")
    command = context["command"]
    index = command["command_index"]
    mode = "initial" if attempt["attempt_kind"] == "initial" else "recovery"
    identity = candidate_runtime_slurm_identity(values=slurm) if slurm is not None else candidate_runtime_slurm_identity()
    commit = source_commit or _git_commit(root)
    if validate_current_runtime:
        require_candidate_worker_binding(root, resolved, required=True)
        if commit != _git_commit(root) or identity != candidate_runtime_slurm_identity():
            raise OperationalTwinProtocolError("candidate_dispatch_live_runtime_mismatch")
    if identity["slurm_array_task_id"] != str(index):
        raise OperationalTwinProtocolError("candidate_dispatch_command_index_mismatch")
    path = _path(root, resolved.execution["output_paths"][
        "dispatch_manifest" if mode == "initial" else "recovery_dispatch_manifest"])
    manifest = require_candidate_dispatch_manifest(
        config, root, _read_json(path), mode=mode, resolved_context=resolved)
    expected_status = "INITIAL_ARRAYS_SUBMITTED" if mode == "initial" else "RECOVERY_ARRAYS_SUBMITTED"
    if manifest["status"] != expected_status or manifest["source_commit"] != commit:
        raise OperationalTwinProtocolError("candidate_dispatch_runtime_source_or_status_mismatch")
    arrays = [row for row in manifest[mode + "_arrays"] if index in row["command_indices"]]
    if len(arrays) != 1:
        raise OperationalTwinProtocolError("candidate_dispatch_runtime_array_not_unique")
    row = arrays[0]
    external_id = row["job_id"] + "_" + str(index)
    if (row["method_id"] != assignment["method_id"]
            or identity["slurm_array_job_id"] != row["job_id"]
            or identity["slurm_job_name"] != row["job_name"]
            or identity["resource_query_job_id"] != external_id):
        raise OperationalTwinProtocolError("candidate_dispatch_runtime_array_mismatch")
    # Release results are necessarily unavailable when a released worker starts.
    intent = require_candidate_release_intent(config, root, resolved, manifest, row)
    binding = {
        "schema_version": "neuronsifter-operational-twin-candidate-runtime-dispatch-binding-v1",
        "candidate_lineage_uid": resolved.protocol["candidate_lineage_uid"],
        "protocol_uid": resolved.protocol["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "command_index": index,
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "source_commit": commit,
        "command_catalog_hash": resolved.command_catalog["command_catalog_hash"],
        "mode": mode,
        "dispatch_manifest": candidate_runtime_artifact_identity(root, path, resolved),
        "array_submission_receipt": row["submission_receipt"],
        "release_intent": intent,
        "external_job_id": external_id,
        "slurm_job_id": identity["slurm_job_id"],
        "array_job_id": row["job_id"],
        "array_task_id": str(index),
        "resource_query_job_id": identity["resource_query_job_id"],
        "successor_execution_binding": dict(resolved.binding),
    }
    if set(binding) != set(_execution_schema(config)["runtime_dispatch_binding_fields"]):
        raise OperationalTwinProtocolError("candidate_dispatch_runtime_binding_keyset_mismatch")
    return binding


def require_future_execution_selection(config: Mapping[str, object], root: Path, round_dir: Path) -> dict[str, object]:
    """Bind a genuine later selection and plan; a handoff round cannot execute."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _identity, _path
    expected = _execution_config(config)["execution_round"]
    relative = round_dir.relative_to(root).as_posix()
    selection_path = relative + "/selection-record.json"
    plan_record_path = relative + "/round-plan-record.json"
    plan_path = relative + "/plan.md"
    selection = _read_json(_path(root, selection_path))
    record = _read_json(_path(root, plan_record_path))
    selection_keys = {"schema_version", "round_id", "roadmap_id", "roadmap_revision", "roadmap_dir",
                      "milestone_id", "direction_id", "extracted_item_id", "summary", "scheduler"}
    plan_keys = {"schema_version", "round_id", "roadmap_id", "roadmap_revision", "roadmap_dir",
                 "milestone_id", "direction_id", "extracted_item_id", "plan_path",
                 "worker_mode", "workers", "integration"}
    if set(selection) != selection_keys or set(record) != plan_keys:
        raise OperationalTwinProtocolError("candidate_execution_selection_keyset_mismatch")
    if selection["schema_version"] != "selection-record-v1" or record["schema_version"] != "round-plan-record-v1":
        raise OperationalTwinProtocolError("candidate_execution_selection_schema_mismatch")
    for key in ("round_id", "roadmap_id", "roadmap_revision", "roadmap_dir", "milestone_id", "direction_id", "extracted_item_id"):
        if not isinstance(selection[key], str) or not selection[key] or selection[key] != record[key]:
            raise OperationalTwinProtocolError("candidate_execution_selection_lineage_mismatch")
    if (selection["direction_id"] != expected["required_direction_id"]
            or selection["milestone_id"] != expected["required_milestone_id"]
            or selection["roadmap_id"] != expected["required_roadmap_id"]
            or relative != expected["round_directory_template"].format(round_id=selection["round_id"])
            or record["plan_path"] != plan_path):
        raise OperationalTwinProtocolError("candidate_execution_future_selection_required")
    identities = {name: _identity(root, "HEAD", path, current=True) for name, path in
                  (("selection_record", selection_path), ("round_plan_record", plan_record_path), ("plan", plan_path))}
    if not _path(root, plan_path).read_text().strip():
        raise OperationalTwinProtocolError("candidate_execution_plan_empty")
    return {"selection": selection, "round_plan_record": record, "identities": identities}


def _block(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise OperationalTwinProtocolError(
            f"candidate_execution_config_invalid: {label} must be a mapping."
        )
    return dict(value)


def _execution_config(config: Mapping[str, object]) -> dict[str, object]:
    return _mapping(config.get(EXECUTION_CONFIG_KEY), label=EXECUTION_CONFIG_KEY)


def _execution_schema(config: Mapping[str, object]) -> dict[str, object]:
    return _mapping(config.get(EXECUTION_SCHEMA_KEY), label=EXECUTION_SCHEMA_KEY)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OperationalTwinProtocolError(
            f"candidate_execution_json_invalid: {path} does not contain an object."
        )
    return payload


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            "candidate_execution_source_commit_unavailable: current Git commit cannot be resolved."
        ) from error


def require_clean_tracked_source(root: Path) -> str:
    """Return HEAD only when tracked source and index are unchanged."""

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            "candidate_execution_source_status_unavailable: tracked source state cannot be inspected."
        ) from error
    if status.strip():
        raise OperationalTwinProtocolError(
            "candidate_execution_source_dirty: tracked execution source must be committed before gate creation."
        )
    return _git_commit(root)


def _git_object_id(root: Path, relative_path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", f"HEAD:{relative_path}"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def candidate_calibration_source_paths(
    config: Mapping[str, object],
    resolved: ResolvedCandidateExecutionProtocol,
    selection: Mapping[str, object],
) -> list[str]:
    """Bind calibration permission to exact reviewed sources and later selection."""
    handoff = config["operational_twin_successor_execution_handoff"]
    paths = {
        str(row["path"])
        for row in resolved.handoff["manifest"]["executing_source"]["source_files"]
    }
    paths.update(handoff["frozen_package"]["publications"].values())
    paths.update(handoff["output_paths"].values())
    paths.update(
        handoff["frozen_package"][key]
        for key in ("config_path", "schema_path", "review_path", "closeout_path")
    )
    paths.add(handoff["review_contract"]["path"])
    paths.update(row["path"] for row in selection["identities"].values())
    return sorted(paths)


def require_candidate_calibration_source_bindings(
    config: Mapping[str, object],
    root: Path,
    authority: Mapping[str, object],
    resolved: ResolvedCandidateExecutionProtocol,
    selection: Mapping[str, object],
    *,
    source_commit: str | None = None,
    executing_source_commit: str | None = None,
) -> None:
    """Reject missing, duplicate, aliased or substituted calibration source paths."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import (
        _identities, executing_source_identity,
    )

    from Libs.utils.evaluation.operational_twin_training_protocol import _require_executing_commit

    current_commit = _git_commit(root)
    expected_source = current_commit if source_commit is None else source_commit
    if authority.get("source_commit") != expected_source:
        raise OperationalTwinProtocolError("candidate_calibration_authority_source_commit_mismatch")
    if (not isinstance(expected_source, str) or len(expected_source) != 40
            or any(character not in "0123456789abcdef" for character in expected_source)):
        raise OperationalTwinProtocolError("candidate_calibration_authority_source_commit_invalid")
    _require_executing_commit(root, expected_source, resolved)
    if subprocess.run(["git", "merge-base", "--is-ancestor", expected_source, current_commit],
                      cwd=root, capture_output=True, check=False).returncode:
        raise OperationalTwinProtocolError("candidate_calibration_authority_source_not_retained")
    if executing_source_commit is not None:
        if source_commit is None:
            raise OperationalTwinProtocolError("candidate_calibration_authority_origin_required")
        if (not isinstance(executing_source_commit, str) or len(executing_source_commit) != 40
                or any(character not in "0123456789abcdef" for character in executing_source_commit)):
            raise OperationalTwinProtocolError("candidate_calibration_executing_source_commit_invalid")
        # Permission keeps its original source identity. A later dispatch must
        # independently retain the reviewed code, configuration and selection.
        reviewed = resolved.handoff["manifest"]["executing_source"]
        if executing_source_identity(config, root, str(reviewed["source_commit"])) != reviewed:
            raise OperationalTwinProtocolError("candidate_calibration_current_source_drift")
        _require_executing_commit(root, executing_source_commit, resolved)
        for ancestor, descendant in (
            (expected_source, executing_source_commit),
            (executing_source_commit, current_commit),
        ):
            if subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant],
                              cwd=root, capture_output=True, check=False).returncode:
                raise OperationalTwinProtocolError("candidate_calibration_executing_source_ancestry_mismatch")
    schema = config["operational_twin_calibration_execution_authority_schema"]
    rows = authority.get("source_files")
    expected = candidate_calibration_source_paths(config, resolved, selection)
    if executing_source_commit is not None:
        for revision in dict.fromkeys((executing_source_commit, current_commit)):
            if subprocess.run(["git", "diff", "--quiet", expected_source, revision, "--", *expected],
                              cwd=root, capture_output=True, check=False).returncode:
                raise OperationalTwinProtocolError("candidate_calibration_executing_source_drift")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise OperationalTwinProtocolError("candidate_calibration_authority_sources_invalid")
    identities = _identities(root, expected_source, expected, current=True)
    for row, path, identity in zip(rows, expected, identities, strict=True):
        if not isinstance(row, Mapping) or set(row) != set(schema["candidate_source_binding_fields"]):
            raise OperationalTwinProtocolError("candidate_calibration_authority_source_keyset_mismatch")
        if row["path"] != path:
            raise OperationalTwinProtocolError("candidate_calibration_authority_source_path_mismatch")
        if row["sha256"] != identity["sha256"]:
            raise OperationalTwinProtocolError("candidate_calibration_authority_source_drift")


def validate_candidate_calibration_execution_authority(
    config: Mapping[str, object],
    root: Path,
    authority_path: Path,
    *,
    source_commit: str | None = None,
    executing_source_commit: str | None = None,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    """Validate the immutable candidate calibration-only authorization."""

    blockers: list[dict[str, str]] = []
    try:
        resolved = resolved_context or resolve_candidate_execution_protocol(config, root)
        execution = dict(resolved.execution)
        protocol = dict(resolved.protocol)
        schema = _mapping(
            config.get("operational_twin_calibration_execution_authority_schema"),
            label="operational_twin_calibration_execution_authority_schema",
        )
        candidate, assignments, commands, preflight = load_candidate_protocol_artifacts(
            config, root, resolved_context=resolved
        )
    except (KeyError, OperationalTwinProtocolError) as error:
        return [_block("candidate_calibration_authority_context_invalid", str(error))]
    if not authority_path.is_file():
        return [
            _block(
                "candidate_calibration_authority_missing",
                f"Candidate calibration authority is absent: {authority_path}.",
            )
        ]
    try:
        authority = _read_json(authority_path)
    except (json.JSONDecodeError, OperationalTwinProtocolError) as error:
        return [_block("candidate_calibration_authority_invalid", str(error))]
    if authority.get("successor_execution_binding") != dict(resolved.binding):
        blockers.append(_block("candidate_calibration_successor_binding_mismatch", "Handoff binding drifted."))
    try:
        selection = authority["execution_selection"]
        selection_path = selection["identities"]["selection_record"]["path"]
        actual_selection = require_future_execution_selection(config, root, (root / selection_path).parent)
        if selection != actual_selection:
            blockers.append(_block("candidate_calibration_execution_selection_drift", "Future selection changed."))
    except (KeyError, ValueError, OSError) as error:
        blockers.append(_block("candidate_calibration_execution_selection_invalid", str(error)))
    required_fields = set(schema.get("candidate_required_fields", []))
    if set(authority) != required_fields:
        blockers.append(
            _block(
                "candidate_calibration_authority_keyset_mismatch",
                "Candidate calibration authority keyset is not exact.",
            )
        )
    if authority.get("schema_version") != schema.get(
        "candidate_required_schema_version"
    ):
        blockers.append(
            _block(
                "candidate_calibration_authority_schema_mismatch",
                "Candidate calibration authority schema version drifted.",
            )
        )
    for field, expected in dict(
        schema.get("candidate_required_boolean_fields", {})
    ).items():
        if authority.get(field) != expected:
            blockers.append(
                _block(
                    "candidate_calibration_authority_policy_mismatch",
                    f"Candidate calibration authority {field} drifted.",
                )
            )
    expected_source_commit = source_commit
    if expected_source_commit is None:
        # Only an explicit execution consumer may validate retained permission.
        # Missing origin is an invalid artifact, never a current-HEAD fallback.
        if executing_source_commit is not None:
            if "source_commit" not in authority:
                return [*blockers, _block("candidate_calibration_authority_source_missing", "Authority origin is absent.")]
            expected_source_commit = authority["source_commit"]
        else:
            expected_source_commit = _git_commit(root)
    expected = {
        "source_commit": expected_source_commit,
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": candidate["candidate_manifest_hash"],
        "assignment_manifest_hash": assignments["assignment_manifest_hash"],
        "command_catalog_hash": commands["command_catalog_hash"],
        "source_identity_hash": candidate_source_identity_hash(preflight),
        "approved_protocol_review_identity": repo_file_identity(
            root,
            str(dict(execution["protocol_review_contract"])["path"]),
        ),
        "allowed_method_ids": ["full_uncertainty_aware_action_conditioned"],
        "checkpoint_selection_role": "development_only",
    }
    for field, expected_value in expected.items():
        if authority.get(field) != expected_value:
            blockers.append(
                _block(
                    "candidate_calibration_authority_identity_mismatch",
                    f"Candidate calibration authority {field} drifted.",
                )
            )
    if authority.get("authority_uid") != compute_candidate_calibration_authority_uid(
        authority
    ):
        blockers.append(
            _block(
                "candidate_calibration_authority_uid_mismatch",
                "Candidate calibration authority UID does not recompute.",
            )
        )
    try:
        require_candidate_calibration_source_bindings(
            config, root, authority, resolved, actual_selection,
            source_commit=expected_source_commit,
            executing_source_commit=executing_source_commit,
        )
    except (KeyError, TypeError, ValueError, OSError, UnboundLocalError) as error:
        blockers.append(_block("candidate_calibration_authority_sources_invalid", str(error)))
    try:
        from Libs.exp.training.train_operational_twin_devcal import (
            require_calibration_execution_authority,
        )

        require_calibration_execution_authority(
            config,
            root,
            authority_path,
            method_id="full_uncertainty_aware_action_conditioned",
            candidate_lineage_uid=str(protocol["candidate_lineage_uid"]),
            protocol_uid=str(protocol["protocol_uid"]),
            resolved_context=resolved,
            source_commit=expected_source_commit,
            executing_source_commit=executing_source_commit,
        )
    except (KeyError, TypeError, ValueError, OperationalTwinProtocolError) as error:
        blockers.append(
            _block(
                "candidate_calibration_authority_runtime_invalid",
                str(error),
            )
        )
    return blockers


def validate_candidate_execution_config(
    config: Mapping[str, object],
    root: Path,
    *,
    require_materialized_artifacts: bool = True,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    """Validate the semantic execution contract against the frozen lineage."""

    blockers: list[dict[str, str]] = []
    if resolved_context is None and require_materialized_artifacts:
        try:
            resolve_candidate_execution_protocol(config, root)
            return []
        except OperationalTwinProtocolError as error:
            return [_block("candidate_execution_handoff_not_approved", str(error))]
    authenticated = None
    if resolved_context is None:
        from Libs.utils.evaluation.operational_twin_successor_execution_handoff import authenticate_frozen_successor_package
        try:
            authenticated = authenticate_frozen_successor_package(config, root)
        except OperationalTwinProtocolError as error:
            return [_block("candidate_execution_frozen_package_invalid", str(error))]
    try:
        execution = _execution_config(config)
        schema = _execution_schema(config)
        protocol = dict(resolved_context.protocol if resolved_context is not None else authenticated["protocol"])
    except OperationalTwinProtocolError as error:
        return [_block("candidate_execution_config_missing", str(error))]

    required_top_level = set(schema.get("required_top_level_fields", []))
    if set(execution) != required_top_level:
        blockers.append(
            _block(
                "candidate_execution_config_keyset_mismatch",
                "Execution config top-level keyset differs from its schema contract.",
            )
        )
    if execution.get("schema_version") != schema.get("required_schema_version"):
        blockers.append(
            _block(
                "candidate_execution_config_schema_mismatch",
                "Execution config schema version drifted.",
            )
        )
    expected_identity = {
        "candidate_lineage_uid": protocol.get("candidate_lineage_uid"),
        "protocol_uid": protocol.get("protocol_uid"),
    }
    for field, expected in expected_identity.items():
        if execution.get(field) != expected:
            blockers.append(
                _block(
                    "candidate_execution_identity_mismatch",
                    f"Execution {field} differs from the frozen protocol.",
                )
            )
    contract = _mapping(
        protocol.get("execution_authority_contract"),
        label="execution_authority_contract",
    )
    if execution.get("scheduler_evidence_contract_version") != contract.get(
        "scheduler_evidence_contract_version"
    ):
        blockers.append(
            _block(
                "candidate_execution_scheduler_contract_mismatch",
                "Execution config scheduler-evidence version differs from the candidate protocol.",
            )
        )
    if execution.get("interface_repair_contract") != schema.get(
        "required_interface_repair_contract"
    ):
        blockers.append(
            _block(
                "candidate_execution_interface_repair_contract_mismatch",
                "Execution-interface repair authority or terminal contract drifted.",
            )
        )
    if execution.get("terminal_scheduler_contract") != schema.get(
        "required_terminal_scheduler_contract"
    ):
        blockers.append(
            _block(
                "candidate_execution_terminal_scheduler_contract_mismatch",
                "Terminal scheduler JobID/JobIDRaw contract drifted.",
            )
        )
    if dict(execution.get("interface_repair_contract", {})).get(
        "candidate_dispatch_authorized"
    ) is not False:
        blockers.append(
            _block(
                "candidate_execution_interface_repair_dispatch_open",
                "Interface repair must not authorize candidate dispatch.",
            )
        )

    frozen_artifacts = execution.get("frozen_artifacts")
    required_artifacts = set(schema.get("required_frozen_artifacts", []))
    required_artifact_fields = set(schema.get("required_frozen_artifact_fields", []))
    if (
        not isinstance(frozen_artifacts, Mapping)
        or set(frozen_artifacts) != required_artifacts
    ):
        blockers.append(
            _block(
                "candidate_execution_frozen_artifacts_invalid",
                "Frozen execution artifact set is not exact.",
            )
        )
        frozen_artifacts = {}
    protocol_paths = _mapping(protocol.get("output_paths"), label="output_paths")
    expected_artifact_paths = {
        "candidate_lineage_manifest": protocol_paths.get("candidate_lineage_manifest"),
        "assignment_manifest": protocol_paths.get("assignment_manifest"),
        "assignment_commands": protocol_paths.get("assignment_commands"),
        "preflight_authority": protocol_paths.get("preflight_authority"),
        "preflight_report": protocol_paths.get("preflight_report"),
    }
    prospective_artifacts: dict[str, dict[str, object]] = {}
    loaded_artifacts: dict[str, dict[str, object]] = {}
    for name in required_artifacts:
        binding = frozen_artifacts.get(name)
        if not isinstance(binding, Mapping) or set(binding) != required_artifact_fields:
            blockers.append(
                _block(
                    "candidate_execution_frozen_artifact_binding_invalid",
                    f"Frozen artifact binding {name} is not exact.",
                )
            )
            continue
        binding = dict(binding)
        if binding.get("path") != expected_artifact_paths.get(name):
            blockers.append(
                _block(
                    "candidate_execution_frozen_artifact_path_mismatch",
                    f"Frozen artifact path drifted for {name}.",
                )
            )
            continue
        path = repo_path(root, str(binding["path"]), must_exist=False)
        if not path.is_file():
            prospective = prospective_artifacts.get(name)
            if prospective is None:
                if require_materialized_artifacts or name != "preflight_report":
                    blockers.append(
                        _block(
                            "candidate_execution_frozen_artifact_missing",
                            f"Frozen artifact is absent: {binding['path']}.",
                        )
                    )
                continue
            loaded_artifacts[name] = prospective
            internal_field = binding.get("internal_hash_field")
            if prospective.get(str(internal_field)) != binding.get("internal_hash"):
                blockers.append(
                    _block(
                        "candidate_execution_frozen_artifact_internal_identity_mismatch",
                        f"Prospective artifact internal identity drifted for {name}.",
                    )
                )
            continue
        internal_field = binding.get("internal_hash_field")
        if internal_field is not None:
            try:
                payload = _read_json(path)
            except (json.JSONDecodeError, OperationalTwinProtocolError) as error:
                blockers.append(
                    _block(
                        "candidate_execution_frozen_artifact_invalid",
                        str(error),
                    )
                )
                continue
            loaded_artifacts[name] = payload
            if payload.get(str(internal_field)) != binding.get("internal_hash"):
                blockers.append(
                    _block(
                        "candidate_execution_frozen_artifact_internal_identity_mismatch",
                        f"Frozen artifact internal identity drifted for {name}.",
                    )
                )

    review_binding = execution.get("protocol_review_contract")
    required_review_fields = set(schema.get("required_review_fields", []))
    if (
        not isinstance(review_binding, Mapping)
        or set(review_binding) != required_review_fields
    ):
        blockers.append(
            _block(
                "candidate_execution_review_binding_invalid",
                "Approved protocol review binding is not exact.",
            )
        )
    else:
        review_binding = dict(review_binding)
        if review_binding.get("path") != contract.get("approved_protocol_review_path"):
            blockers.append(
                _block(
                    "candidate_execution_review_path_mismatch",
                    "Approved protocol review path differs from the frozen authority contract.",
                )
            )
        required_review = {
            "schema_version": contract.get("required_review_schema_version"),
            "round_id": contract.get("required_review_round_id"),
            "direction_id": contract.get("required_review_direction_id"),
            "decision": contract.get("required_review_decision"),
        }
        for field, expected in required_review.items():
            if review_binding.get(field) != expected:
                blockers.append(
                    _block(
                        "candidate_execution_review_contract_mismatch",
                        f"Prospective protocol review {field} drifted.",
                    )
                )

    if execution.get("execution_round") != schema.get("required_execution_selection_contract"):
        blockers.append(_block("candidate_execution_selection_contract_mismatch", "Future execution selection contract drifted."))
    if execution.get("binding_contract") != schema.get("required_binding_contract"):
        blockers.append(_block("candidate_execution_binding_contract_mismatch", "Execution envelope contract drifted."))

    if execution.get("attempt_gate") != schema.get("required_attempt_gate"):
        blockers.append(
            _block(
                "candidate_execution_attempt_gate_mismatch",
                "Attempt-gate policy differs from the frozen execution schema.",
            )
        )
    if dict(execution.get("attempt_gate", {})).get(
        "maximum_resource_receipt_age_seconds"
    ) != contract.get("maximum_resource_receipt_age_seconds"):
        blockers.append(
            _block(
                "candidate_execution_receipt_age_mismatch",
                "Execution receipt age differs from the frozen protocol.",
            )
        )

    groups = execution.get("array_groups")
    required_group_fields = set(schema.get("required_array_group_fields", []))
    required_methods = list(schema.get("required_method_ids", []))
    initial_indices: list[int] = []
    recovery_indices: list[int] = []
    if not isinstance(groups, list) or len(groups) != len(required_methods):
        blockers.append(
            _block(
                "candidate_execution_array_groups_invalid",
                "Execution array groups do not cover the five frozen methods exactly.",
            )
        )
        groups = []
    catalog = loaded_artifacts.get("assignment_commands", {})
    commands = catalog.get("commands") if isinstance(catalog, Mapping) else None
    command_rows = (
        {
            int(row["command_index"]): dict(row)
            for row in commands
            if isinstance(row, Mapping) and isinstance(row.get("command_index"), int)
        }
        if isinstance(commands, list)
        else {}
    )
    observed_methods: list[str] = []
    for group_value in groups:
        if (
            not isinstance(group_value, Mapping)
            or set(group_value) != required_group_fields
        ):
            blockers.append(
                _block(
                    "candidate_execution_array_group_invalid",
                    "One execution array group has an inexact keyset.",
                )
            )
            continue
        group = dict(group_value)
        method_id = str(group["method_id"])
        observed_methods.append(method_id)
        group_initial = [int(value) for value in group["initial_command_indices"]]
        group_recovery = [int(value) for value in group["recovery_command_indices"]]
        initial_indices.extend(group_initial)
        recovery_indices.extend(group_recovery)
        expected_resources = {
            key: group[key]
            for key in (
                "partition",
                "nodelist",
                "cpus_per_task",
                "memory",
                "gres",
                "exclusive",
            )
        }
        for command_index, conditional in (
            *((value, False) for value in group_initial),
            *((value, True) for value in group_recovery),
        ):
            row = command_rows.get(command_index)
            if (
                row is None
                or row.get("method_id") != method_id
                or row.get("conditional") is not conditional
                or row.get("resource_requirements") != expected_resources
                or row.get("wall_time") != group.get("wall_time")
            ):
                blockers.append(
                    _block(
                        "candidate_execution_array_group_catalog_mismatch",
                        f"Command index {command_index} differs from execution group {method_id}.",
                    )
                )
    if observed_methods != required_methods:
        blockers.append(
            _block(
                "candidate_execution_method_order_mismatch",
                "Execution array method order differs from the frozen contract.",
            )
        )
    if sorted(initial_indices) != list(
        schema.get("required_initial_command_indices", [])
    ):
        blockers.append(
            _block(
                "candidate_execution_initial_index_coverage_mismatch",
                "Initial command indices do not cover 0-14 exactly once.",
            )
        )
    if sorted(recovery_indices) != list(
        schema.get("required_recovery_command_indices", [])
    ):
        blockers.append(
            _block(
                "candidate_execution_recovery_index_coverage_mismatch",
                "Recovery command indices do not cover 15-29 exactly once.",
            )
        )

    output_paths = execution.get("output_paths")
    required_output_fields = set(schema.get("required_output_path_fields", []))
    if (
        not isinstance(output_paths, Mapping)
        or set(output_paths) != required_output_fields
    ):
        blockers.append(
            _block(
                "candidate_execution_output_paths_invalid",
                "Execution output-path keyset is not exact.",
            )
        )
    else:
        lineage_uid = str(execution.get("candidate_lineage_uid", ""))
        lineage_root = f"Output/results/operational_twin/candidate_lineages/{lineage_uid}/execution"
        for field in (
            "execution_root",
            "dispatch_manifest",
            "recovery_eligibility",
            "candidate_execution_catalog",
            "scheduler_receipts_root",
        ):
            if not str(output_paths[field]).startswith(lineage_root):
                blockers.append(
                    _block(
                        "candidate_execution_output_path_mismatch",
                        f"Execution output {field} is outside the semantic lineage root.",
                    )
                )
    if execution.get("terminal_policy") != schema.get("required_terminal_policy"):
        blockers.append(
            _block(
                "candidate_execution_terminal_policy_mismatch",
                "Execution terminal policy drifted.",
            )
        )
    access = execution.get("access_counters")
    required_access = set(schema.get("required_zero_access_counters", []))
    if (
        not isinstance(access, Mapping)
        or set(access) != required_access
        or any(value != 0 for value in access.values())
        or dict(access) != dict(protocol.get("access_counters", {}))
    ):
        blockers.append(
            _block(
                "candidate_execution_access_boundary_invalid",
                "Execution access counters must match the frozen all-zero boundary.",
            )
        )
    if execution.get("closed_flags") != schema.get("required_closed_flags"):
        blockers.append(
            _block(
                "candidate_execution_closed_flags_invalid",
                "Scientific and downstream gates must remain closed during execution.",
            )
        )

    if all(
        name in loaded_artifacts
        for name in (
            "candidate_lineage_manifest",
            "assignment_manifest",
            "assignment_commands",
            "preflight_authority",
        )
    ):
        candidate_manifest = loaded_artifacts["candidate_lineage_manifest"]
        assignment_manifest = loaded_artifacts["assignment_manifest"]
        command_catalog = loaded_artifacts["assignment_commands"]
        preflight = loaded_artifacts["preflight_authority"]
        if resolved_context is not None:
            blockers.extend(
                validate_candidate_artifacts(
                    config,
                    root,
                    candidate_manifest,
                    assignment_manifest,
                    command_catalog,
                    preflight,
                    require_attempt_roots_absent=False,
                    resolved_context=resolved_context,
                )
            )
        else:
            frozen_package = authenticated["approved_package"]["package"]
            for name, observed in (("candidate_lineage_manifest", candidate_manifest),
                                   ("assignment_manifest", assignment_manifest),
                                   ("assignment_commands", command_catalog),
                                   ("preflight_authority", preflight)):
                if observed != frozen_package[name]:
                    blockers.append(_block("candidate_execution_frozen_artifact_drift", name))
        if execution.get("preflight_uid") != preflight.get("preflight_uid"):
            blockers.append(
                _block(
                    "candidate_execution_preflight_identity_mismatch",
                    "Execution preflight UID differs from the approved preflight.",
                )
            )
    return blockers


def validate_candidate_receipt_semantics_amendment(
    config: Mapping[str, object], root: Path
) -> list[dict[str, str]]:
    """Validate the v2 amendment before successor artifacts are published."""

    blockers = validate_candidate_execution_config(
        config,
        root,
        require_materialized_artifacts=False,
    )
    try:
        execution = _execution_config(config)
        protocol = _mapping(
            config.get("operational_twin_candidate_lineage_protocol"),
            label="operational_twin_candidate_lineage_protocol",
        )
        candidate = build_candidate_lineage_manifest(config)
        assignments = build_candidate_assignment_manifest(config, candidate)
    except (KeyError, OperationalTwinProtocolError) as error:
        blockers.append(
            _block("candidate_execution_amendment_context_invalid", str(error))
        )
        return blockers
    materialized: list[str] = []
    for assignment_value in assignments.get("assignments", []):
        assignment = dict(assignment_value)
        for attempt_value in assignment.get("attempts", []):
            attempt = dict(attempt_value)
            paths = candidate_execution_artifact_paths(protocol, assignment, attempt)
            for value in (
                attempt.get("output_root"),
                paths["execution_authority"],
                paths["resource_environment_receipt"],
            ):
                path = repo_path(root, str(value), must_exist=False)
                if path.exists() or path.is_symlink():
                    materialized.append(str(value))
    output_paths = _mapping(execution.get("output_paths"), label="output_paths")
    for key, value in output_paths.items():
        if (
            key == "calibration_execution_authority"
            or key.endswith("audit")
            or key.endswith("report")
            or key
            in {
                "execution_root",
                "dispatch_manifest",
                "recovery_eligibility",
                "candidate_execution_catalog",
                "scheduler_receipts_root",
            }
        ):
            path = repo_path(root, str(value), must_exist=False)
            if path.exists() or path.is_symlink():
                materialized.append(str(value))
    if materialized:
        blockers.append(
            _block(
                "candidate_execution_outputs_present_during_amendment",
                "Receipt amendment must not materialize candidate execution state: "
                + ", ".join(sorted(set(materialized))[:5]),
            )
        )
    return blockers


def validate_candidate_execution_interface_repair(
    config: Mapping[str, object],
    root: Path,
    *,
    require_materialized_artifacts: bool,
) -> list[dict[str, str]]:
    """Validate the prospective interface repair and retained resource capture."""

    blockers = validate_candidate_protocol_config(config, root)
    blockers.extend(
        validate_candidate_execution_config(
            config,
            root,
            require_materialized_artifacts=require_materialized_artifacts,
        )
    )
    try:
        protocol = _mapping(
            config.get("operational_twin_candidate_lineage_protocol"),
            label="operational_twin_candidate_lineage_protocol",
        )
        source_bindings = _mapping(
            protocol.get("source_bindings"), label="source_bindings"
        )
        fixture_path = repo_path(
            root,
            str(source_bindings["array_element_resource_fixture"]),
            must_exist=True,
        )
        fixture = _read_json(fixture_path)
        required_fixture_keys = {
            "captured_at",
            "environment",
            "resource_command",
            "resource_requirements",
            "resource_stdout",
            "resource_stdout_sha256",
            "sinfo_command",
            "sinfo_stdout",
            "sinfo_stdout_sha256",
            "slurm",
            "source_array_job_id",
            "source_array_task_id",
            "source_job_id",
        }
        if set(fixture) != required_fixture_keys:
            raise OperationalTwinProtocolError(
                "candidate_resource_fixture_keyset_mismatch: array resource fixture keyset is not exact."
            )
        if fixture.get("resource_stdout_sha256") != sha256_text(
            str(fixture.get("resource_stdout", ""))
        ) or fixture.get("sinfo_stdout_sha256") != sha256_text(
            str(fixture.get("sinfo_stdout", ""))
        ):
            raise OperationalTwinProtocolError(
                "candidate_resource_fixture_hash_mismatch: retained scheduler output hash drifted."
            )
        slurm = candidate_runtime_slurm_identity(
            values=_mapping(fixture.get("slurm"), label="fixture.slurm")
        )
        expected_query = (
            f"{fixture['source_array_job_id']}_{fixture['source_array_task_id']}"
        )
        if (
            str(fixture.get("source_job_id"))
            != str(fixture.get("source_array_job_id"))
            or slurm["resource_query_job_id"] != expected_query
        ):
            raise OperationalTwinProtocolError(
                "candidate_resource_fixture_identity_mismatch: retained array identity drifted."
            )
        receipt = {
            "schema_version": CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
            "resource_command": fixture["resource_command"],
            "resource_stdout": fixture["resource_stdout"],
            "resource_stdout_sha256": fixture["resource_stdout_sha256"],
            "resource_allocation": fixture["resource_requirements"],
            "slurm": slurm,
        }
        requirements = _mapping(
            fixture.get("resource_requirements"), label="fixture.requirements"
        )
        contract = _mapping(
            protocol.get("execution_authority_contract"),
            label="execution_authority_contract",
        )
        _require_candidate_resource_evidence(
            receipt,
            requirements,
            contract["fresh_resource_command"],
        )
        _require_live_candidate_resource_placement(
            requirements,
            slurm,
            _mapping(fixture.get("environment"), label="fixture.environment"),
        )
    except (
        KeyError,
        FileNotFoundError,
        json.JSONDecodeError,
        OperationalTwinProtocolError,
    ) as error:
        blockers.append(_block("candidate_execution_interface_fixture_invalid", str(error)))
    return blockers

def require_candidate_execution_config(config: Mapping[str, object], root: Path) -> dict[str, object]:
    return dict(resolve_candidate_execution_protocol(config, root).execution)


def _attempt_context(resolved: ResolvedCandidateExecutionProtocol, attempt_uid: str) -> dict[str, object]:
    matches = [(row, slot) for row in resolved.assignment_manifest["assignments"]
               for slot in row["attempts"] if slot["attempt_uid"] == attempt_uid]
    commands = [row for row in resolved.command_catalog["commands"] if row["attempt_uid"] == attempt_uid]
    if len(matches) != 1 or len(commands) != 1:
        raise OperationalTwinProtocolError("candidate_ownership_attempt_not_unique")
    assignment, attempt = matches[0]
    return {"resolved_protocol": resolved, "execution": resolved.execution,
            "assignment": assignment, "attempt": attempt, "command": commands[0],
            "candidate_manifest": resolved.candidate_manifest,
            "assignment_manifest": resolved.assignment_manifest,
            "command_catalog": resolved.command_catalog, "preflight": resolved.preflight}


def _check_declared_attempt_tree(config: Mapping[str, object], root: Path,
                                 assignment: Mapping[str, object], attempt: Mapping[str, object]) -> None:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    files = {value.format(method_id=assignment["method_id"])
             for value in _execution_schema(config)["attempt_owned_files"]}
    directories = {parent.as_posix() for value in files for parent in Path(value).parents if parent != Path(".")}
    base = str(attempt["output_root"])
    pending = [""]
    seen = 0
    while pending:
        relative = pending.pop()
        directory = _path(root, base + ("/" + relative if relative else ""))
        if not directory.is_dir():
            raise OperationalTwinProtocolError("candidate_owned_directory_invalid")
        for child in directory.iterdir():
            seen += 1
            if seen > len(files) + len(directories):
                raise OperationalTwinProtocolError("candidate_attempt_foreign_child")
            name = (relative + "/" if relative else "") + child.name
            safe = _path(root, base + "/" + name)
            if name in directories and safe.is_dir():
                pending.append(name)
            elif name not in files or not safe.is_file():
                raise OperationalTwinProtocolError("candidate_attempt_foreign_child: " + name)
            elif name.startswith("checkpoints/") and not assignment["checkpoint_creation_allowed"]:
                raise OperationalTwinProtocolError("candidate_no_fit_checkpoint_forbidden")


def require_candidate_attempt_ownership(
    config: Mapping[str, object], root: Path, resolved: ResolvedCandidateExecutionProtocol,
    attempt_uid: str, *, require_retained_evidence: bool = True,
) -> dict[str, object]:
    """Authenticate a claimed slot without reading checkpoint payloads."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    context = _attempt_context(resolved, attempt_uid)
    assignment, attempt, command = context["assignment"], context["attempt"], context["command"]
    base = str(attempt["output_root"])
    _check_declared_attempt_tree(config, root, assignment, attempt)
    owner = _read_json(_path(root, base + "/attempt_ownership.json"))
    if set(owner) != set(_execution_schema(config)["attempt_ownership_fields"]):
        raise OperationalTwinProtocolError("candidate_attempt_ownership_keyset_mismatch")
    expected = {
        "schema_version": "neuronsifter-operational-twin-candidate-attempt-ownership-v1",
        "candidate_lineage_uid": resolved.protocol["candidate_lineage_uid"],
        "protocol_uid": resolved.protocol["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"], "attempt_uid": attempt_uid,
        "command_index": command["command_index"], "method_id": assignment["method_id"],
        "seed": assignment["seed"], "output_root": base,
        "command_catalog_hash": resolved.command_catalog["command_catalog_hash"],
        "successor_execution_binding": dict(resolved.binding),
        "access_counters": resolved.protocol["access_counters"],
    }
    for key, value in expected.items():
        if owner.get(key) != value:
            raise OperationalTwinProtocolError("candidate_attempt_ownership_binding_mismatch: " + key)
    paths = candidate_execution_artifact_paths(resolved.protocol, assignment, attempt)
    validated = require_candidate_execution_authority(
        config, root, assignment=assignment, attempt=attempt,
        candidate_manifest=resolved.candidate_manifest, assignment_manifest=resolved.assignment_manifest,
        command_catalog=resolved.command_catalog, preflight=resolved.preflight,
        execution_authority_path=paths["execution_authority"],
        resource_environment_receipt_path=paths["resource_environment_receipt"],
        cuda_available=False, cuda_device_count=0, torch_version="",
        current_source_commit=owner["source_commit"],
        now=datetime.fromisoformat(owner["claimed_at"]), validate_current_runtime=False,
        resolved_context=resolved,
    )
    for key in ("execution_authority", "resource_environment_receipt"):
        if owner[key] != validated[key]:
            raise OperationalTwinProtocolError("candidate_attempt_ownership_gate_mismatch")
    if owner["runtime_dispatch_binding"] != validated["execution_authority"]["runtime_dispatch_binding"]:
        raise OperationalTwinProtocolError("candidate_attempt_ownership_dispatch_mismatch")
    retained = False
    terminal = _path(root, base + "/terminal_status.json")
    if terminal.is_file():
        status = _read_json(terminal)
        fields = ("candidate_lineage_uid", "protocol_uid", "assignment_uid", "attempt_uid", "method_id", "seed", "output_root")
        if any(status.get(key) != expected[key] for key in fields) or status.get("started_at") != owner["claimed_at"]:
            raise OperationalTwinProtocolError("candidate_attempt_ownership_terminal_mismatch")
        blockers = _validate_retained_candidate_execution_bindings(
            resolved.protocol, root, assignment, attempt, status,
            resolved.candidate_manifest, resolved.assignment_manifest,
            resolved.command_catalog, resolved.preflight, resolved_context=resolved,
        )
        if blockers:
            raise OperationalTwinProtocolError("; ".join(row["code"] for row in blockers))
        retained = True
    phase_path = _path(root, base + "/execution_phase_receipt.json")
    if phase_path.is_file():
        phase = _read_json(phase_path)
        if set(phase) != set(_execution_schema(config)["phase_receipt_fields"]):
            raise OperationalTwinProtocolError("candidate_owned_phase_keyset_mismatch")
        for key in ("candidate_lineage_uid", "protocol_uid", "assignment_uid", "attempt_uid",
                    "command_index", "method_id", "seed", "output_root", "source_commit",
                    "successor_execution_binding", "runtime_dispatch_binding", "execution_authority", "resource_environment_receipt", "access_counters"):
            if phase.get(key) != owner.get(key):
                raise OperationalTwinProtocolError("candidate_owned_phase_binding_mismatch: " + key)
        if (phase["schema_version"] != "neuronsifter-operational-twin-candidate-execution-phase-receipt-v2"
                or phase["attempt_kind"] != attempt["attempt_kind"]
                or phase["scheduler_identity"] != validated["resource_receipt_payload"]["slurm"]
                or phase["closed_flags"] != resolved.execution["closed_flags"]):
            raise OperationalTwinProtocolError("candidate_owned_phase_provenance_mismatch")
        for key, path in (("terminal_status", terminal), ("evaluation_summary", _path(root, base + "/evaluation/summary.json"))):
            if phase[key] != (candidate_runtime_artifact_identity(root, path, resolved) if path.is_file() else None):
                raise OperationalTwinProtocolError("candidate_owned_phase_artifact_mismatch")
        if phase["status"] not in {"PASS", "FAIL"} or not isinstance(phase["phases"], list) or not phase["phases"]:
            raise OperationalTwinProtocolError("candidate_owned_phase_status_invalid")
        retained = True
    if require_retained_evidence and not retained:
        raise OperationalTwinProtocolError("candidate_attempt_retained_provenance_missing")
    return owner


def require_candidate_runtime_ownership(
    config: Mapping[str, object], root: Path, resolved: ResolvedCandidateExecutionProtocol,
    *, selected_attempt_uid: str | None = None,
) -> None:
    """Validate the finite successor namespace without discarding crash evidence."""
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import require_candidate_dispatch_manifest

    if selected_attempt_uid is not None:
        _attempt_context(resolved, selected_attempt_uid)
    output_paths = resolved.execution["output_paths"]
    directories = {str(output_paths[key]) for key in ("execution_root", "scheduler_receipts_root")}
    directories.add(str(output_paths["execution_root"]) + "/logs")
    files = {str(value) for key, value in output_paths.items()
             if key not in {"execution_root", "scheduler_receipts_root"}}
    owned_roots: set[str] = set()
    namespaces: set[str] = set()
    lineage = str(resolved.protocol["candidate_lineage_uid"])
    for mode in ("initial", "recovery"):
        for group in resolved.execution["array_groups"]:
            for directory in ("dispatch_receipts", "release_intents", "release_receipts"):
                files.add(f"{output_paths['execution_root']}/{directory}/{mode}_{group['group_id']}.json")
        dispatch_path = _path(root, output_paths["dispatch_manifest" if mode == "initial" else "recovery_dispatch_manifest"])
        if dispatch_path.exists():
            payload = require_candidate_dispatch_manifest(
                config, root, json.loads(dispatch_path.read_text()), mode=mode, resolved_context=resolved,
            )
            for array in payload[mode + "_arrays"]:
                files.add(f"{output_paths['scheduler_receipts_root']}/{mode}_{array['job_id']}.json")
                for option in array["submission_argv"]:
                    if option.startswith(("--output=", "--error=")):
                        template = option.split("=", 1)[1]
                        for index in array["command_indices"]:
                            log_path = Path(template.replace("%A", array["job_id"]).replace("%a", str(index)))
                            files.add(log_path.relative_to(root).as_posix())

    for assignment in resolved.assignment_manifest["assignments"]:
        for attempt in assignment["attempts"]:
            uid = str(attempt["attempt_uid"])
            base = str(attempt["output_root"])
            leaf = candidate_attempt_path(root, base, require_absent=uid == selected_attempt_uid)
            directories.add(base)
            paths = candidate_execution_artifact_paths(resolved.protocol, assignment, attempt)
            files.update(paths.values())
            for value in [base, *paths.values()]:
                parts = Path(value).parts
                if lineage not in parts:
                    raise OperationalTwinProtocolError("candidate_ownership_namespace_invalid")
                namespaces.add(Path(*parts[:parts.index(lineage) + 1]).as_posix())
                _path(root, value)
            if leaf.exists():
                require_candidate_attempt_ownership(config, root, resolved, uid)
                owned_roots.add(base)
            else:
                gates = [_path(root, paths[key]) for key in (
                    "resource_environment_receipt", "execution_authority")]
                if any(os.path.lexists(gate) for gate in gates):
                    if uid != selected_attempt_uid:
                        raise OperationalTwinProtocolError("candidate_attempt_orphan_gate: " + uid)
                    if not all(gate.is_file() for gate in gates):
                        raise OperationalTwinProtocolError("candidate_attempt_partial_gate: " + uid)
                    # The selected complete pair is authenticated immediately before
                    # exclusive claim; a failed pair remains immutable evidence.

    for value in [*files, *directories]:
        _path(root, value)
        directories.update(parent.as_posix() for parent in Path(value).parents if parent != Path("."))
    seen = 0
    for namespace in namespaces:
        if not _path(root, namespace).exists():
            continue
        pending = [namespace]
        while pending:
            directory = pending.pop()
            for child in _path(root, directory).iterdir():
                seen += 1
                if seen > len(files) + len(directories):
                    raise OperationalTwinProtocolError("candidate_namespace_foreign_child")
                relative = child.relative_to(root).as_posix()
                safe = _path(root, relative)
                if relative in owned_roots and safe.is_dir():
                    continue
                if relative in directories and safe.is_dir():
                    pending.append(relative)
                elif relative not in files or not safe.is_file():
                    raise OperationalTwinProtocolError("candidate_namespace_foreign_child: " + relative)


def claim_candidate_attempt(
    config: Mapping[str, object], root: Path, resolved: ResolvedCandidateExecutionProtocol,
    attempt_uid: str, execution_context: Mapping[str, object], *, claimed_at: datetime,
) -> Path:
    """Revalidate fresh authority and acquire the single immutable attempt owner."""
    context = _attempt_context(resolved, attempt_uid)
    assignment, attempt, command = context["assignment"], context["attempt"], context["command"]
    require_candidate_runtime_ownership(config, root, resolved, selected_attempt_uid=attempt_uid)
    paths = candidate_execution_artifact_paths(resolved.protocol, assignment, attempt)
    environment = execution_context["resource_receipt_payload"]["environment"]
    validated = require_candidate_execution_authority(
        config, root, assignment=assignment, attempt=attempt,
        candidate_manifest=resolved.candidate_manifest, assignment_manifest=resolved.assignment_manifest,
        command_catalog=resolved.command_catalog, preflight=resolved.preflight,
        execution_authority_path=paths["execution_authority"],
        resource_environment_receipt_path=paths["resource_environment_receipt"],
        cuda_available=environment["cuda_available"], cuda_device_count=environment["cuda_device_count"],
        torch_version=environment["torch_version"], now=claimed_at, resolved_context=resolved,
    )
    for key in ("execution_authority", "resource_environment_receipt"):
        if execution_context[key] != validated[key]:
            raise OperationalTwinProtocolError("candidate_attempt_claim_gate_changed")
    owner = {
        "schema_version": "neuronsifter-operational-twin-candidate-attempt-ownership-v1",
        "candidate_lineage_uid": resolved.protocol["candidate_lineage_uid"],
        "protocol_uid": resolved.protocol["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"], "attempt_uid": attempt_uid,
        "command_index": command["command_index"], "method_id": assignment["method_id"],
        "seed": assignment["seed"], "output_root": attempt["output_root"],
        "source_commit": validated["execution_authority"]["source_commit"],
        "command_catalog_hash": resolved.command_catalog["command_catalog_hash"],
        "claimed_at": claimed_at.isoformat(),
        "execution_authority": validated["execution_authority"],
        "resource_environment_receipt": validated["resource_environment_receipt"],
        "successor_execution_binding": dict(resolved.binding),
        "runtime_dispatch_binding": validated["execution_authority"]["runtime_dispatch_binding"],
        "access_counters": resolved.protocol["access_counters"],
    }
    output = _claim_candidate_attempt_root(root, attempt["output_root"])
    # Once claimed, failures retain the root and gate evidence; no overwrite/resume.
    write_json_exclusive(output / "attempt_ownership.json", owner)
    return output


def resolve_candidate_attempt_context(
    config: Mapping[str, object], root: Path, task_index: int
) -> dict[str, object]:
    """Resolve one immutable command row and its matching manifest objects."""

    resolved = resolve_candidate_execution_protocol(config, root)
    execution = dict(resolved.execution)
    candidate_manifest, assignment_manifest, command_catalog, preflight = (
        load_candidate_protocol_artifacts(config, root, resolved_context=resolved)
    )
    commands = [dict(row) for row in command_catalog["commands"]]
    matches = [row for row in commands if int(row["command_index"]) == int(task_index)]
    if len(matches) != 1:
        raise OperationalTwinProtocolError(
            f"candidate_execution_command_index_invalid: {task_index} is not present exactly once."
        )
    command = matches[0]
    assignments = [
        dict(row)
        for row in assignment_manifest["assignments"]
        if row.get("assignment_uid") == command.get("assignment_uid")
    ]
    if len(assignments) != 1:
        raise OperationalTwinProtocolError(
            "candidate_execution_assignment_invalid: command assignment is not present exactly once."
        )
    assignment = assignments[0]
    attempts = [
        dict(row)
        for row in assignment["attempts"]
        if row.get("attempt_uid") == command.get("attempt_uid")
    ]
    if len(attempts) != 1:
        raise OperationalTwinProtocolError(
            "candidate_execution_attempt_invalid: command attempt is not present exactly once."
        )
    attempt = attempts[0]
    context = require_candidate_assignment_preflight(
        config,
        root,
        candidate_lineage_uid=str(command["candidate_lineage_uid"]),
        protocol_uid=str(command["protocol_uid"]),
        assignment_uid=str(command["assignment_uid"]),
        attempt_uid=str(command["attempt_uid"]),
        method_id=str(command["method_id"]),
        seed=int(command["seed"]),
        output_root=str(command["output_root"]),
        candidate_manifest_hash=str(command_catalog["candidate_manifest_hash"]),
        assignment_manifest_hash=str(command_catalog["assignment_manifest_hash"]),
        command_catalog_hash_value=str(command_catalog["command_catalog_hash"]),
        preflight_uid=str(command_catalog["preflight_uid"]),
        state_inputs={
            "resume_input": None,
            "history_input": None,
            "checkpoint_input": None,
            "optimizer_state_input": None,
            "model_state_input": None,
            "calibration_state_input": None,
        },
        require_recovery_predecessor=True,
        resolved_context=resolved,
    )
    if dict(context["assignment"]) != assignment or dict(context["attempt"]) != attempt:
        raise OperationalTwinProtocolError(
            "candidate_execution_manifest_context_mismatch: runtime preflight returned a different assignment or attempt."
        )
    return {
        "execution": execution,
        "resolved_protocol": resolved,
        "command": command,
        "assignment": assignment,
        "attempt": attempt,
        "candidate_manifest": candidate_manifest,
        "assignment_manifest": assignment_manifest,
        "command_catalog": command_catalog,
        "preflight": preflight,
    }


def _runtime_environment_identity(hostname: str) -> dict[str, object]:
    import platform

    return {
        "hostname": hostname,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _capture_command(argv: list[str], root: Path) -> str:
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
            f"candidate_execution_resource_capture_failed: {' '.join(argv)}"
        ) from error
    if not completed.stdout.strip():
        raise OperationalTwinProtocolError(
            f"candidate_execution_resource_capture_empty: {' '.join(argv)}"
        )
    return completed.stdout


def build_candidate_resource_receipt(
    config: Mapping[str, object],
    *,
    context: Mapping[str, object],
    source_commit: str,
    captured_at: datetime,
    sinfo_stdout: str,
    resource_stdout: str,
    slurm: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    """Build one exact pre-execution resource/environment receipt payload."""

    slurm_identity = candidate_runtime_slurm_identity(values=slurm)

    resolved = context.get("resolved_protocol")
    protocol = _candidate_protocol(config, resolved)
    contract = _candidate_execution_contract(protocol, resolved)
    assignment = _mapping(context.get("assignment"), label="assignment")
    attempt = _mapping(context.get("attempt"), label="attempt")
    candidate_manifest = _mapping(
        context.get("candidate_manifest"), label="candidate_manifest"
    )
    assignment_manifest = _mapping(
        context.get("assignment_manifest"), label="assignment_manifest"
    )
    command_catalog = _mapping(context.get("command_catalog"), label="command_catalog")
    preflight = _mapping(context.get("preflight"), label="preflight")
    requirements = _mapping(
        assignment.get("resource_requirements"), label="resource_requirements"
    )
    if set(requirements) != CANDIDATE_RESOURCE_ALLOCATION_KEYS:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: resource requirements keyset is not exact."
        )
    if set(slurm_identity) != CANDIDATE_RESOURCE_SLURM_KEYS:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: slurm keyset is not exact."
        )
    if set(environment) != CANDIDATE_RESOURCE_ENVIRONMENT_KEYS:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: environment keyset is not exact."
        )
    timestamp = captured_at.astimezone(timezone.utc).isoformat()
    sinfo_command = [
        str(value).format(nodelist=requirements["nodelist"])
        for value in contract["fresh_sinfo_command"]
    ]
    resource_command = [
        str(value).format(
            resource_query_job_id=slurm_identity["resource_query_job_id"]
        )
        for value in contract["fresh_resource_command"]
    ]
    receipt: dict[str, object] = {
        "schema_version": contract["resource_receipt_schema_version"],
        "status": "PASS",
        "terminal_decision": contract["positive_resource_receipt_terminal"],
        "receipt_uid": None,
        "created_at": timestamp,
        "sinfo_captured_at": timestamp,
        "sinfo_command": sinfo_command,
        "sinfo_stdout": sinfo_stdout,
        "sinfo_stdout_sha256": sha256_text(sinfo_stdout),
        "resource_captured_at": timestamp,
        "resource_command": resource_command,
        "resource_stdout": resource_stdout,
        "resource_stdout_sha256": sha256_text(resource_stdout),
        "resource_allocation": deepcopy(requirements),
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest["assignment_manifest_hash"],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "source_commit": source_commit,
        "source_identity_hash": candidate_source_identity_hash(preflight),
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "resource_profile": assignment["resource_profile"],
        "resource_requirements": deepcopy(requirements),
        "slurm": deepcopy(slurm_identity),
        "environment": deepcopy(dict(environment)),
        "fresh_sinfo": True,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "access_counters": deepcopy(protocol["access_counters"]),
    }
    if resolved is not None:
        receipt["successor_execution_binding"] = dict(resolved.binding)
        receipt["runtime_dispatch_binding"] = deepcopy(context["runtime_dispatch_binding"])
    if set(receipt) != candidate_execution_binding_keys(CANDIDATE_RESOURCE_RECEIPT_KEYS, resolved):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: generated receipt keyset is not exact."
        )
    receipt["receipt_uid"] = compute_candidate_resource_receipt_uid(receipt)
    return receipt


def build_live_candidate_resource_probe(
    config: Mapping[str, object],
    root: Path,
    *,
    resource_profile: str,
    command_capture: Callable[[list[str], Path], str] = _capture_command,
    captured_at: datetime | None = None,
) -> dict[str, object]:
    """Validate one live allocation without publishing candidate gate artifacts."""

    resolved = resolve_candidate_execution_protocol(config, root)
    source_commit = require_clean_tracked_source(root)
    protocol = dict(resolved.protocol)
    candidate, assignments, commands, preflight = load_candidate_protocol_artifacts(
        config, root, resolved_context=resolved
    )
    matching_assignments = [
        dict(row)
        for row in assignments["assignments"]
        if row.get("resource_profile") == resource_profile
    ]
    if not matching_assignments:
        raise OperationalTwinProtocolError(
            f"candidate_resource_probe_profile_invalid: {resource_profile} has no assignment."
        )
    assignment = matching_assignments[0]
    attempt = next(
        dict(row)
        for row in assignment["attempts"]
        if row.get("attempt_kind") == "initial"
    )
    context = {
        # This unpublished allocation diagnostic has no dispatch authority.
        # Runtime consumers reconstruct a non-null binding and reject this value.
        "runtime_dispatch_binding": None,
        "resolved_protocol": resolved,
        "assignment": assignment,
        "attempt": attempt,
        "candidate_manifest": candidate,
        "assignment_manifest": assignments,
        "command_catalog": commands,
        "preflight": preflight,
    }
    artifact_paths = candidate_execution_artifact_paths(
        protocol,
        assignment,
        attempt,
    )
    prohibited = [
        str(attempt["output_root"]),
        artifact_paths["execution_authority"],
        artifact_paths["resource_environment_receipt"],
    ]
    if any(
        (repo_path(root, value).exists() or repo_path(root, value).is_symlink())
        for value in prohibited
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_probe_side_effect_preexists: candidate execution path is already materialized."
        )
    hostname = socket.gethostname().split(".", 1)[0]
    slurm = candidate_runtime_slurm_identity(hostname=hostname)
    environment = _runtime_environment_identity(hostname)
    contract = _candidate_execution_contract(protocol, resolved)
    requirements = _mapping(
        assignment.get("resource_requirements"),
        label="resource_requirements",
    )
    sinfo_command = [
        str(value).format(nodelist=requirements["nodelist"])
        for value in contract["fresh_sinfo_command"]
    ]
    resource_command = [
        str(value).format(
            resource_query_job_id=slurm["resource_query_job_id"]
        )
        for value in contract["fresh_resource_command"]
    ]
    sinfo_stdout = command_capture(sinfo_command, root)
    resource_stdout = command_capture(resource_command, root)
    timestamp = captured_at or datetime.now(timezone.utc)
    receipt = build_candidate_resource_receipt(
        config,
        context=context,
        source_commit=source_commit,
        captured_at=timestamp,
        sinfo_stdout=sinfo_stdout,
        resource_stdout=resource_stdout,
        slurm=slurm,
        environment=environment,
    )
    _require_candidate_resource_evidence(
        receipt,
        requirements,
        contract["fresh_resource_command"],
    )
    _require_live_candidate_resource_placement(
        requirements,
        slurm,
        environment,
    )
    side_effects = [
        value
        for value in prohibited
        if repo_path(root, value).exists() or repo_path(root, value).is_symlink()
    ]
    if side_effects:
        raise OperationalTwinProtocolError(
            "candidate_resource_probe_side_effect_detected: no-publish validation materialized candidate state."
        )
    return {
        "schema_version": "neuronsifter-operational-twin-live-resource-probe-v1",
        "status": "PASS",
        "successor_execution_binding": dict(resolved.binding),
        "terminal_decision": "OPERATIONAL_TWIN_RESOURCE_RECEIPT_LIVE_PROBE_VALID",
        "resource_profile": resource_profile,
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "source_commit": source_commit,
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "normalized_resource_allocation": receipt["resource_allocation"],
        "resource_receipt_uid": receipt["receipt_uid"],
        "sinfo_command": receipt["sinfo_command"],
        "sinfo_stdout": receipt["sinfo_stdout"],
        "sinfo_stdout_sha256": receipt["sinfo_stdout_sha256"],
        "resource_command": receipt["resource_command"],
        "resource_stdout": receipt["resource_stdout"],
        "resource_stdout_sha256": receipt["resource_stdout_sha256"],
        "slurm": receipt["slurm"],
        "environment": receipt["environment"],
        "side_effect_count": 0,
        "access_counters": protocol["access_counters"],
    }


def build_candidate_execution_authority(
    config: Mapping[str, object],
    root: Path,
    *,
    context: Mapping[str, object],
    source_commit: str,
    issued_at: datetime,
    receipt_path: Path,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Build one review- and receipt-bound execution authority payload."""

    resolved = context.get("resolved_protocol")
    protocol = _candidate_protocol(config, resolved)
    contract = _candidate_execution_contract(protocol, resolved)
    assignment = _mapping(context.get("assignment"), label="assignment")
    attempt = _mapping(context.get("attempt"), label="attempt")
    candidate_manifest = _mapping(
        context.get("candidate_manifest"), label="candidate_manifest"
    )
    assignment_manifest = _mapping(
        context.get("assignment_manifest"), label="assignment_manifest"
    )
    command_catalog = _mapping(context.get("command_catalog"), label="command_catalog")
    preflight = _mapping(context.get("preflight"), label="preflight")
    review, review_identity = _require_approved_protocol_review(root, contract)
    receipt_identity = candidate_runtime_artifact_identity(root, receipt_path, resolved)
    authority: dict[str, object] = {
        "schema_version": contract["authority_schema_version"],
        "status": "PASS",
        "terminal_decision": contract["positive_authority_terminal"],
        "authority_uid": None,
        "issued_at": issued_at.astimezone(timezone.utc).isoformat(),
        "authority_role": contract["authority_role"],
        "approved_protocol_review_identity": review_identity,
        "approved_protocol_review_decision": review.get("decision"),
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest["assignment_manifest_hash"],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "source_commit": source_commit,
        "source_identity_hash": candidate_source_identity_hash(preflight),
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "resource_profile": assignment["resource_profile"],
        "resource_requirements": deepcopy(assignment["resource_requirements"]),
        "resource_environment_receipt_identity": {
            **receipt_identity,
            "receipt_uid": receipt["receipt_uid"],
        },
        "fresh_initialization_receipt": candidate_fresh_initialization_receipt(protocol, resolved),
        "state_resume_allowed": False,
        "training_execution_authorized": True,
        "checkpoint_creation_authorized": assignment["checkpoint_creation_allowed"],
        "calibration_fitting_authorized": assignment["calibration_eligible"],
        "access_counters": deepcopy(protocol["access_counters"]),
    }
    if resolved is not None:
        authority["successor_execution_binding"] = dict(resolved.binding)
        authority["runtime_dispatch_binding"] = deepcopy(receipt["runtime_dispatch_binding"])
    if set(authority) != candidate_execution_binding_keys(CANDIDATE_EXECUTION_AUTHORITY_KEYS, resolved):
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_schema_invalid: generated authority keyset is not exact."
        )
    authority["authority_uid"] = compute_candidate_execution_authority_uid(authority)
    return authority


def _require_approved_protocol_review(
    root: Path,
    contract: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Load and validate the committed protocol review before publication."""

    review_path = repo_path(
        root,
        str(contract["approved_protocol_review_path"]),
        must_exist=True,
    )
    try:
        review = _read_json(review_path)
    except json.JSONDecodeError as error:
        raise OperationalTwinProtocolError(
            "candidate_execution_review_invalid: approved Round 45 review is not valid JSON."
        ) from error
    review_identity = repo_file_identity(root, review_path)
    required_review = {
        "schema_version": contract["required_review_schema_version"],
        "round_id": contract["required_review_round_id"],
        "direction_id": contract["required_review_direction_id"],
        "decision": contract["required_review_decision"],
    }
    for field, expected in required_review.items():
        if review.get(field) != expected:
            raise OperationalTwinProtocolError(
                f"candidate_execution_review_not_approved: review {field} drifted."
            )
    if (root / ".git").exists() and review_identity.get("git_object_id") is None:
        raise OperationalTwinProtocolError(
            "candidate_execution_review_not_committed: approved Round 45 review must exist in the executing Git tree."
        )
    return review, review_identity


def write_bytes_exclusive(path: Path, content: bytes) -> None:
    """Create one immutable regular file through no-follow directory descriptors."""
    spelling = os.fspath(path)
    if ".." in path.parts or spelling != path.as_posix():
        raise OperationalTwinProtocolError("candidate_execution_artifact_path_alias")
    absolute = path.absolute()
    opened: list[int] = []
    parent_fd = os.open(absolute.anchor, _candidate_directory_open_flags())
    opened.append(parent_fd)
    identity = None
    ancestors: list[tuple[Path, os.stat_result]] = []
    current_parent = Path(absolute.anchor)
    try:
        for component in absolute.parts[1:-1]:
            parent_fd = _open_or_create_candidate_ancestor(parent_fd, component)
            opened.append(parent_fd)
            current_parent = current_parent / component
            ancestors.append((current_parent, os.fstat(parent_fd)))
        try:
            descriptor = os.open(
                absolute.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644, dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise OperationalTwinProtocolError("candidate_execution_artifact_exists: " + str(path)) from error
        identity = os.fstat(descriptor)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        for ancestor, original in ancestors:
            current = ancestor.lstat()
            if (not stat.S_ISDIR(current.st_mode)
                    or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)):
                raise OperationalTwinProtocolError("candidate_execution_artifact_ancestor_race")
        current = absolute.lstat()
        if (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                or (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino)):
            raise OperationalTwinProtocolError("candidate_execution_artifact_creation_race")
    except BaseException:
        if identity is not None:
            try:
                current = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    os.unlink(absolute.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    """Serialize through the shared exclusive, descriptor-owned publication path."""
    write_bytes_exclusive(path, (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode())


def build_and_publish_candidate_execution_gate(
    config: Mapping[str, object],
    root: Path,
    *,
    task_index: int,
    command_capture: Callable[[list[str], Path], str] = _capture_command,
    captured_at: datetime | None = None,
    source_commit: str | None = None,
    slurm: Mapping[str, object] | None = None,
    environment: Mapping[str, object] | None = None,
    require_clean_source: bool = True,
) -> dict[str, object]:
    """Capture, publish, and independently validate one attempt gate."""

    context = resolve_candidate_attempt_context(config, root, task_index)
    assignment = _mapping(context["assignment"], label="assignment")
    attempt = _mapping(context["attempt"], label="attempt")
    if require_clean_source:
        resolved_source_commit = require_clean_tracked_source(root)
        if source_commit is not None and source_commit != resolved_source_commit:
            raise OperationalTwinProtocolError(
                "candidate_execution_source_commit_mismatch: requested source commit differs from clean HEAD."
            )
    else:
        resolved_source_commit = source_commit or _git_commit(root)
    resolved = context["resolved_protocol"]
    context["runtime_dispatch_binding"] = require_candidate_dispatch_binding(
        config, root, resolved, assignment=assignment, attempt=attempt,
        slurm=slurm, source_commit=resolved_source_commit,
        validate_current_runtime=slurm is None and environment is None,
    )
    hostname = socket.gethostname().split(".", 1)[0]
    slurm_identity = (
        candidate_runtime_slurm_identity(hostname=hostname, values=slurm)
        if slurm is not None
        else candidate_runtime_slurm_identity(hostname=hostname)
    )
    environment_identity = (
        dict(environment)
        if environment is not None
        else _runtime_environment_identity(hostname)
    )
    resolved = context["resolved_protocol"]
    protocol = dict(resolved.protocol)
    contract = _candidate_execution_contract(protocol, resolved)
    requirements = _mapping(
        assignment.get("resource_requirements"), label="resource_requirements"
    )
    sinfo_command = [
        str(value).format(nodelist=requirements["nodelist"])
        for value in contract["fresh_sinfo_command"]
    ]
    resource_command = [
        str(value).format(
            resource_query_job_id=slurm_identity["resource_query_job_id"]
        )
        for value in contract["fresh_resource_command"]
    ]
    sinfo_stdout = command_capture(sinfo_command, root)
    resource_stdout = command_capture(resource_command, root)
    timestamp = captured_at or datetime.now(timezone.utc)
    receipt = build_candidate_resource_receipt(
        config,
        context=context,
        source_commit=resolved_source_commit,
        captured_at=timestamp,
        sinfo_stdout=sinfo_stdout,
        resource_stdout=resource_stdout,
        slurm=slurm_identity,
        environment=environment_identity,
    )
    _require_candidate_resource_evidence(
        receipt,
        requirements,
        contract["fresh_resource_command"],
    )
    _require_live_candidate_resource_placement(
        requirements,
        slurm_identity,
        environment_identity,
    )
    paths = candidate_execution_artifact_paths(protocol, assignment, attempt)
    receipt_path = repo_path(root, paths["resource_environment_receipt"])
    authority_path = repo_path(root, paths["execution_authority"])
    if repo_path(root, str(attempt["output_root"])).exists():
        raise OperationalTwinProtocolError(
            "candidate_attempt_output_exists: execution authority must precede attempt-root creation."
        )
    _require_approved_protocol_review(root, contract)
    for path in (receipt_path, authority_path):
        if os.path.lexists(path):
            raise OperationalTwinProtocolError(
                f"candidate_execution_artifact_exists: refusing to overwrite {path}."
            )

    # Retain published gate evidence on later failure; slots are never reusable.
    write_json_exclusive(receipt_path, receipt)
    authority = build_candidate_execution_authority(
        config,
        root,
        context=context,
        source_commit=resolved_source_commit,
        issued_at=timestamp,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    write_json_exclusive(authority_path, authority)
    validated = require_candidate_execution_authority(
        config,
        root,
        assignment=assignment,
        attempt=attempt,
        candidate_manifest=context["candidate_manifest"],
        assignment_manifest=context["assignment_manifest"],
        command_catalog=context["command_catalog"],
        preflight=context["preflight"],
        execution_authority_path=authority_path,
        resource_environment_receipt_path=receipt_path,
        cuda_available=bool(environment_identity["cuda_available"]),
        cuda_device_count=int(environment_identity["cuda_device_count"]),
        torch_version=str(environment_identity["torch_version"]),
        current_source_commit=resolved_source_commit,
        now=timestamp,
        validate_current_runtime=slurm is None and environment is None,
        resolved_context=resolved,
    )
    return {
        "task_index": int(task_index),
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "source_commit": resolved_source_commit,
        "successor_execution_binding": dict(resolved.binding),
        "runtime_dispatch_binding": validated["execution_authority"]["runtime_dispatch_binding"],
        "execution_authority": validated["execution_authority"],
        "resource_environment_receipt": validated["resource_environment_receipt"],
    }


def validate_published_candidate_execution_gate(
    config: Mapping[str, object],
    root: Path,
    *,
    task_index: int,
    source_commit: str | None = None,
    now: datetime | None = None,
    validate_current_runtime: bool = True,
) -> dict[str, object]:
    """Validate one already-published gate before the attempt root is claimed."""

    context = resolve_candidate_attempt_context(config, root, task_index)
    assignment = _mapping(context["assignment"], label="assignment")
    attempt = _mapping(context["attempt"], label="attempt")
    resolved = context["resolved_protocol"]
    protocol = dict(resolved.protocol)
    paths = candidate_execution_artifact_paths(protocol, assignment, attempt)
    environment = _runtime_environment_identity(socket.gethostname().split(".", 1)[0])
    validated = require_candidate_execution_authority(
        config,
        root,
        assignment=assignment,
        attempt=attempt,
        candidate_manifest=context["candidate_manifest"],
        assignment_manifest=context["assignment_manifest"],
        command_catalog=context["command_catalog"],
        preflight=context["preflight"],
        execution_authority_path=paths["execution_authority"],
        resource_environment_receipt_path=paths["resource_environment_receipt"],
        cuda_available=bool(environment["cuda_available"]),
        cuda_device_count=int(environment["cuda_device_count"]),
        torch_version=str(environment["torch_version"]),
        current_source_commit=source_commit,
        now=now,
        validate_current_runtime=validate_current_runtime,
        resolved_context=resolved,
    )
    return {
        "task_index": int(task_index),
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "execution_authority": validated["execution_authority"],
        "resource_environment_receipt": validated["resource_environment_receipt"],
    }


def build_candidate_pre_dispatch_audit(
    config: Mapping[str, object],
    root: Path,
) -> dict[str, object]:
    """Prove the clean, zero-access, all-roots-absent dispatch boundary."""

    resolved = resolve_candidate_execution_protocol(config, root)
    blockers = validate_candidate_execution_config(config, root, resolved_context=resolved)
    execution = _execution_config(config)
    try:
        source_commit = require_clean_tracked_source(root)
    except OperationalTwinProtocolError as error:
        source_commit = None
        blockers.append(_block("candidate_pre_dispatch_source_invalid", str(error)))
    output_paths = dict(execution["output_paths"])
    authority_path = repo_path(
        root,
        str(output_paths["calibration_execution_authority"]),
    )
    if source_commit is not None:
        blockers.extend(
            validate_candidate_calibration_execution_authority(
                config,
                root,
                authority_path,
                executing_source_commit=source_commit,
                resolved_context=resolved,
            )
        )
    try:
        candidate, assignments, commands, preflight = load_candidate_protocol_artifacts(
            config, root, resolved_context=resolved
        )
    except (OSError, ValueError, OperationalTwinProtocolError) as error:
        candidate = {}
        assignments = {}
        commands = {}
        preflight = {}
        blockers.append(_block("candidate_pre_dispatch_artifacts_invalid", str(error)))
    protocol = dict(resolved.protocol)
    attempt_root_count = 0
    authority_path_count = 0
    receipt_path_count = 0
    for assignment_value in assignments.get("assignments", []):
        assignment = dict(assignment_value)
        for attempt_value in assignment.get("attempts", []):
            attempt = dict(attempt_value)
            if repo_path(root, str(attempt["output_root"])).exists():
                attempt_root_count += 1
            paths = candidate_execution_artifact_paths(
                protocol,
                assignment,
                attempt,
            )
            if repo_path(root, paths["execution_authority"]).exists():
                authority_path_count += 1
            if repo_path(root, paths["resource_environment_receipt"]).exists():
                receipt_path_count += 1
    if attempt_root_count or authority_path_count or receipt_path_count:
        blockers.append(
            _block(
                "candidate_pre_dispatch_paths_not_empty",
                "All 30 attempt roots and per-attempt authority/receipt paths must be absent before initial dispatch.",
            )
        )
    for field in (
        "dispatch_manifest",
        "recovery_eligibility",
        "candidate_execution_catalog",
        "execution_audit",
        "execution_report",
    ):
        if repo_path(root, str(output_paths[field])).exists():
            blockers.append(
                _block(
                    "candidate_pre_dispatch_output_exists",
                    f"Pre-dispatch output already exists: {output_paths[field]}.",
                )
            )
    try:
        status_lines = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        status_lines = []
        blockers.append(
            _block(
                "candidate_pre_dispatch_git_status_unavailable",
                str(error),
            )
        )
    allowed_untracked = {
        str(output_paths["calibration_execution_authority"]),
    }
    unexpected_status: list[str] = []
    for line in status_lines:
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if line.startswith("?? ") and path in allowed_untracked:
            continue
        unexpected_status.append(line)
    if unexpected_status:
        blockers.append(
            _block(
                "candidate_pre_dispatch_worktree_not_clean",
                "Unexpected tracked or untracked source paths: "
                + "; ".join(unexpected_status),
            )
        )
    return {
        "schema_version": "neuronsifter-operational-twin-candidate-pre-dispatch-audit-v2",
        "successor_execution_binding": dict(resolved.binding),
        "status": "PASS" if not blockers else "FAIL",
        "terminal_decision": (
            "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_READY"
            if not blockers
            else "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_NOT_READY"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "candidate_lineage_uid": candidate.get("candidate_lineage_uid"),
        "protocol_uid": candidate.get("protocol_uid"),
        "preflight_uid": preflight.get("preflight_uid"),
        "candidate_manifest_hash": candidate.get("candidate_manifest_hash"),
        "assignment_manifest_hash": assignments.get("assignment_manifest_hash"),
        "command_catalog_hash": commands.get("command_catalog_hash"),
        "assignment_count": len(assignments.get("assignments", [])),
        "attempt_slot_count": sum(
            len(dict(row).get("attempts", []))
            for row in assignments.get("assignments", [])
        ),
        "materialized_attempt_root_count": attempt_root_count,
        "materialized_execution_authority_count": authority_path_count,
        "materialized_resource_receipt_count": receipt_path_count,
        "candidate_calibration_execution_authority": (
            repo_file_identity(root, authority_path)
            if authority_path.is_file()
            else None
        ),
        "git_status_porcelain": status_lines,
        "allowed_untracked_paths": sorted(allowed_untracked),
        "access_counters": execution["access_counters"],
        "closed_flags": execution["closed_flags"],
        "blockers": blockers,
    }
