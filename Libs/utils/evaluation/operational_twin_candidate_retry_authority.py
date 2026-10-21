"""Deterministic no-dispatch adjudication of Operational Twin retry slots."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Mapping

from Libs.scripts.builders.build_operational_twin_development_training_commands import (
    build_candidate_commands,
)
from Libs.utils.evaluation.operational_twin_candidate_evidence import (
    classify_recovery_eligibility,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_COMMAND_ROW_KEYS,
    CANDIDATE_FAILED_TERMINAL_KEYS,
    CANDIDATE_LINEAGE_ROOT,
    CANDIDATE_MANIFEST_SCHEMA_VERSION,
    CANDIDATE_PREFLIGHT_SCHEMA_VERSION,
    CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS,
    CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS,
    OperationalTwinProtocolError,
    build_candidate_assignment_manifest,
    build_candidate_lineage_manifest,
    build_candidate_preflight_authority,
    candidate_attempt_path,
    canonical_hash,
    candidate_execution_artifact_paths,
    candidate_fresh_initialization_receipt,
    command_catalog_hash,
    compute_candidate_preflight_uid,
    compute_candidate_protocol_uid,
    read_repo_json,
    validate_candidate_command_catalog,
)


CONFIG_KEY = "operational_twin_candidate_retry_authority"
POSITIVE = "OPERATIONAL_TWIN_CANDIDATE_RETRY_AUTHORITY_FROZEN"
SUCCESSOR_REQUIRED = "OPERATIONAL_TWIN_SUCCESSOR_LINEAGE_PROTOCOL_REQUIRED"
AUTHORIZED = "EXISTING_LINEAGE_SLOT_AUTHORIZED"
SUCCESSOR_ROW = "SUCCESSOR_LINEAGE_REQUIRED"
POSITIVE_PREFLIGHT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-lineage-retry-preflight-v1"
)

AUDIT_KEYS = {
    "schema_version",
    "status",
    "authoritative_date",
    "terminal_decision",
    "next_direction",
    "candidate_lineage_uid",
    "repaired_protocol_uid",
    "repaired_preflight_uid",
    "assignment_count",
    "authorized_row_count",
    "successor_required_row_count",
    "rows",
    "global_predicates",
    "round_46_candidate_tree_id",
    "round_46_audit_tree_id",
    "candidate_dispatch_authorized",
    "side_effect_count",
    "scientific_twin_result",
    "access_counters",
    "closed_flags",
    "positive_artifacts_materialized",
    "decision_uid",
}
ROW_KEYS = {
    "candidate_lineage_uid",
    "assignment_uid",
    "method_id",
    "seed",
    "resource_profile",
    "initial_attempt_uid",
    "initial_command_index",
    "initial_output_root",
    "predecessor_evidence_kind",
    "predecessor_failure_class",
    "historical_recovery_classification",
    "predecessor_identity",
    "retry_attempt_uid",
    "retry_command_index",
    "retry_attempt_kind",
    "retry_attempt_ordinal",
    "retry_output_root",
    "retry_execution_authority",
    "retry_resource_environment_receipt",
    "predicates",
    "result",
}
ROW_PREDICATE_KEYS = {
    "retry_attempt_kind",
    "retry_attempt_ordinal",
    "retry_conditional",
    "eligible_failure_class_contract",
    "predecessor_identity",
    "initial_command_index",
    "retry_command_index",
    "fresh_random_initialization",
    "state_resume_forbidden",
    "predecessor_failure_class_eligible",
    "null_resume_input",
    "null_history_input",
    "null_checkpoint_input",
    "null_optimizer_state_input",
    "null_model_state_input",
    "null_calibration_state_input",
    "output_root_canonical",
    "output_root_unused_git",
    "output_root_unused_filesystem",
    "execution_authority_canonical",
    "execution_authority_unused_git",
    "execution_authority_unused_filesystem",
    "resource_environment_receipt_canonical",
    "resource_environment_receipt_unused_git",
    "resource_environment_receipt_unused_filesystem",
}
GLOBAL_PREDICATE_KEYS = {
    "assignment_count",
    "initial_slot_count",
    "retry_slot_count",
    "unique_assignment_uid",
    "unique_retry_attempt_uid",
    "unique_retry_command_index",
    "unique_output_root",
    "unique_execution_authority",
    "unique_resource_environment_receipt",
    "all_access_counters_zero",
    "all_downstream_flags_closed",
    "candidate_dispatch_authorized_false",
    "scientific_twin_not_run",
}
APPROVED_REVIEW_KEYS = {
    "schema_version",
    "round_id",
    "roadmap_id",
    "roadmap_revision",
    "roadmap_dir",
    "milestone_id",
    "direction_id",
    "extracted_item_id",
    "decision",
    "evidence_summary",
    "roadmap_closeout",
}
PHASE_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "terminal_decision",
    "candidate_lineage_uid",
    "protocol_uid",
    "assignment_uid",
    "attempt_uid",
    "attempt_kind",
    "command_index",
    "method_id",
    "seed",
    "output_root",
    "source_commit",
    "started_at",
    "finished_at",
    "phases",
    "scheduler_identity",
    "execution_authority",
    "resource_environment_receipt",
    "terminal_status",
    "evaluation_summary",
    "error",
    "access_counters",
    "closed_flags",
}
FAILED_TERMINAL_CLAIM_KEYS = {
    "agent_trace_gate_open",
    "agent_work_open",
    "coevolution_gate_open",
    "formal_qualification",
    "mcp_gate_open",
    "memory_gate_open",
    "sufficient_training",
    "test_manifest_open",
    "tool_gate_open",
    "training_authorized",
    "twin_qualification_pass",
}
POSITIVE_ARTIFACT_KEYS = (
    "candidate_lineage_manifest",
    "assignment_manifest",
    "assignment_commands",
    "preflight_authority",
    "preflight_report",
)


def _config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get(CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise OperationalTwinProtocolError(f"missing_retry_authority_config: {CONFIG_KEY}")
    return dict(value)


def _git_object(root: Path, ref: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}:{path}"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_json(root: Path, ref: str, path: str) -> dict[str, object]:
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], cwd=root, stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"retry_authority_git_input_missing: {ref}:{path}"
        ) from error
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise OperationalTwinProtocolError(
            f"retry_authority_git_input_invalid: {ref}:{path} is not an object"
        )
    return value


def _git_bytes(root: Path, ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], cwd=root, stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"retry_authority_git_input_missing: {ref}:{path}"
        ) from error


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _decision_uid(audit: Mapping[str, object]) -> str:
    basis = deepcopy(dict(audit))
    basis.pop("decision_uid", None)
    # Materialization is publication state, not part of the scientific decision.
    # Keeping it outside the identity makes the positive decision stable before
    # and after its atomically published package appears.
    basis["positive_artifacts_materialized"] = False
    return f"operational_twin_candidate_retry_authority_{canonical_hash(basis)[:24]}"


def _canonical_relative_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    expected: str | None = None,
    allowed_root: str | Path | None = None,
) -> tuple[str, Path]:
    declared = os.fspath(value)
    if not declared:
        raise OperationalTwinProtocolError(f"{label}_path_invalid: path is empty")
    path = Path(declared)
    if path.is_absolute():
        raise OperationalTwinProtocolError(
            f"{label}_path_absolute: repository-relative path required"
        )
    if ".." in path.parts:
        raise OperationalTwinProtocolError(
            f"{label}_path_parent_reference: '..' is forbidden"
        )
    canonical = path.as_posix()
    if canonical != declared:
        raise OperationalTwinProtocolError(
            f"{label}_path_alias: canonical POSIX spelling required"
        )
    if expected is not None and canonical != expected:
        raise OperationalTwinProtocolError(
            f"{label}_path_mismatch: expected {expected}, observed {canonical}"
        )
    if allowed_root is not None:
        try:
            path.relative_to(Path(allowed_root))
        except ValueError as error:
            raise OperationalTwinProtocolError(
                f"{label}_path_outside_prefix: {canonical} is outside {Path(allowed_root).as_posix()}"
            ) from error
    repository_root = root.resolve(strict=True)
    current = repository_root
    for component in path.parts[:-1]:
        current = current / component
        if not os.path.lexists(current):
            continue
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OperationalTwinProtocolError(
                f"{label}_path_symlink: path ancestors must not be symlinks"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise OperationalTwinProtocolError(
                f"{label}_path_ancestor_not_directory: path ancestors must be directories"
            )
    resolved = repository_root.joinpath(*path.parts)
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise OperationalTwinProtocolError(
            f"{label}_path_outside_repository: {canonical}"
        ) from error
    return canonical, resolved


def _require_unused_retry_target(
    root: Path,
    value: str | Path,
    *,
    label: str,
    expected: str,
    allowed_root: str | Path,
) -> Path:
    canonical, path = _canonical_relative_path(
        root,
        value,
        label=label,
        expected=expected,
        allowed_root=allowed_root,
    )
    if os.path.lexists(path):
        metadata = path.lstat()
        suffix = "symlink" if stat.S_ISLNK(metadata.st_mode) else "collision"
        raise OperationalTwinProtocolError(
            f"{label}_path_{suffix}: retry target already exists"
        )
    if _git_object(root, "HEAD", canonical) is not None:
        raise OperationalTwinProtocolError(
            f"{label}_path_committed_collision: retry target exists in HEAD"
        )
    return path


def _expected_attempt_paths(
    protocol: Mapping[str, object],
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
) -> dict[str, str]:
    values = {
        "candidate_lineage_uid": assignment.get("candidate_lineage_uid"),
        "assignment_uid": assignment.get("assignment_uid"),
        "attempt_uid": attempt.get("attempt_uid"),
    }
    return {
        "output_root": str(dict(protocol["output_paths"])["attempt_root_template"]).format(
            **values
        ),
        **candidate_execution_artifact_paths(protocol, assignment, attempt),
    }


def _read_committed_file(
    root: Path,
    path: str,
    *,
    require_materialized: bool,
) -> tuple[bytes, dict[str, object]]:
    canonical, resolved = _canonical_relative_path(
        root,
        path,
        label="retry_authority_committed_artifact",
        expected=path,
    )
    git_object_id = _git_object(root, "HEAD", canonical)
    if git_object_id is None:
        raise OperationalTwinProtocolError(
            f"retry_authority_artifact_uncommitted: {canonical}"
        )
    committed = _git_bytes(root, "HEAD", canonical)
    if os.path.lexists(resolved):
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise OperationalTwinProtocolError(
                f"retry_authority_artifact_not_regular: {canonical}"
            )
        materialized = resolved.read_bytes()
        if materialized != committed:
            raise OperationalTwinProtocolError(
                f"retry_authority_artifact_git_mismatch: {canonical}"
            )
    elif require_materialized:
        raise OperationalTwinProtocolError(
            f"retry_authority_artifact_not_materialized: {canonical}"
        )
    return committed, {
        "path": canonical,
        "sha256": hashlib.sha256(committed).hexdigest(),
        "git_object_id": git_object_id,
    }


def _read_committed_json(
    root: Path,
    path: str,
    *,
    require_materialized: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    content, identity = _read_committed_file(
        root,
        path,
        require_materialized=require_materialized,
    )
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperationalTwinProtocolError(
            f"retry_authority_artifact_json_invalid: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise OperationalTwinProtocolError(
            f"retry_authority_artifact_json_invalid: {path} is not an object"
        )
    return payload, identity


def _positive_artifact_paths(config: Mapping[str, object]) -> dict[str, str]:
    paths = dict(_config(config)["output_paths"])
    return {key: str(paths[key]) for key in POSITIVE_ARTIFACT_KEYS}


def _positive_artifacts_materialized(config: Mapping[str, object], root: Path) -> bool:
    for value in _positive_artifact_paths(config).values():
        _, path = _canonical_relative_path(
            root,
            value,
            label="retry_authority_positive_artifact",
            expected=value,
        )
        if not os.path.lexists(path):
            return False
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return False
    return True


def _exact_infrastructure_predecessor(
    config: Mapping[str, object],
    *,
    protocol: Mapping[str, object],
    assignment: Mapping[str, object],
    initial: Mapping[str, object],
    initial_command: Mapping[str, object],
    terminal: Mapping[str, object],
    phase: Mapping[str, object],
    failure_class: str,
) -> bool:
    if set(terminal) != CANDIDATE_FAILED_TERMINAL_KEYS:
        return False
    expected_identity = {
        "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
        "candidate_lineage_uid": assignment.get("candidate_lineage_uid"),
        "protocol_uid": assignment.get("protocol_uid"),
        "assignment_uid": assignment.get("assignment_uid"),
        "attempt_uid": initial.get("attempt_uid"),
        "method_id": assignment.get("method_id"),
        "seed": assignment.get("seed"),
        "output_root": initial.get("output_root"),
        "status": "FAILED",
        "failure_class": failure_class,
        "fresh_initialization_receipt": candidate_fresh_initialization_receipt(
            protocol
        ),
        "state_resume_used": False,
    }
    if any(terminal.get(field) != expected for field, expected in expected_identity.items()):
        return False
    if not isinstance(terminal.get("terminal_reason"), str) or not terminal.get(
        "terminal_reason"
    ):
        return False
    if not isinstance(terminal.get("error"), str) or not terminal.get("error"):
        return False
    if not isinstance(terminal.get("scheduler_identity"), Mapping):
        return False
    access_flags = terminal.get("access_flags")
    if (
        not isinstance(access_flags, Mapping)
        or dict(access_flags) != dict(_config(config)["access_counters"])
    ):
        return False
    claim_flags = terminal.get("claim_flags")
    if (
        not isinstance(claim_flags, Mapping)
        or set(claim_flags) != FAILED_TERMINAL_CLAIM_KEYS
        or any(value is not False for value in claim_flags.values())
    ):
        return False
    authority = terminal.get("candidate_execution_authority")
    receipt = terminal.get("candidate_resource_environment_receipt")
    if (
        not isinstance(authority, Mapping)
        or set(authority) != CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS
        or not isinstance(receipt, Mapping)
        or set(receipt) != CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS
    ):
        return False
    expected_paths = candidate_execution_artifact_paths(
        protocol,
        assignment,
        initial,
    )
    if (
        authority.get("path") != expected_paths["execution_authority"]
        or receipt.get("path") != expected_paths["resource_environment_receipt"]
        or authority.get("source_commit") != receipt.get("source_commit")
    ):
        return False
    if set(phase) != PHASE_RECEIPT_KEYS:
        return False
    phase_expected = {
        "schema_version": "neuronsifter-operational-twin-candidate-execution-phase-receipt-v1",
        "status": "FAIL",
        "candidate_lineage_uid": assignment.get("candidate_lineage_uid"),
        "protocol_uid": assignment.get("protocol_uid"),
        "assignment_uid": assignment.get("assignment_uid"),
        "attempt_uid": initial.get("attempt_uid"),
        "attempt_kind": initial.get("attempt_kind"),
        "command_index": initial_command.get("command_index"),
        "method_id": assignment.get("method_id"),
        "seed": assignment.get("seed"),
        "output_root": initial.get("output_root"),
        "source_commit": authority.get("source_commit"),
        "scheduler_identity": terminal.get("scheduler_identity"),
        "execution_authority": authority,
        "resource_environment_receipt": receipt,
        "evaluation_summary": None,
    }
    if any(phase.get(field) != expected for field, expected in phase_expected.items()):
        return False
    if phase.get("terminal_decision") not in {
        "OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_TRAINING_FAILED",
        "OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_EVALUATION_FAILED",
    }:
        return False
    phases = phase.get("phases")
    failed_phase = False
    if isinstance(phases, list):
        for row in phases:
            if not isinstance(row, Mapping):
                continue
            try:
                failed_phase = int(row.get("returncode", 0)) != 0
            except (TypeError, ValueError):
                return False
            if failed_phase:
                break
    if not isinstance(phases, list) or not phases or not failed_phase:
        return False
    execution = config.get("operational_twin_candidate_execution")
    if not isinstance(execution, Mapping):
        return False
    if (
        phase.get("access_counters") != execution.get("access_counters")
        or phase.get("closed_flags") != execution.get("closed_flags")
    ):
        return False
    return True


def _require_object(root: Path, path: str, expected: str, ref: str = "HEAD") -> None:
    observed = _git_object(root, ref, path)
    if observed != expected:
        raise OperationalTwinProtocolError(
            f"retry_authority_git_identity_mismatch: {ref}:{path} expected {expected}, observed {observed}"
        )


def validate_retry_authority_config(config: Mapping[str, object]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    try:
        contract = _config(config)
        schema = dict(config["operational_twin_candidate_retry_authority_schema"])
        slot = dict(contract["slot_contract"])
        checks = {
            "schema_version": schema["required_schema_version"],
            "authoritative_date": schema["required_authoritative_date"],
        }
        for field, expected in checks.items():
            if contract.get(field) != expected:
                blockers.append({"code": "retry_authority_config_invalid", "message": f"{field} drifted"})
        if contract.get("terminals") != schema["required_terminals"]:
            blockers.append({"code": "retry_authority_terminals_invalid", "message": "terminal set drifted"})
        if contract.get("next_directions") != schema["required_next_directions"]:
            blockers.append({"code": "retry_authority_directions_invalid", "message": "next directions drifted"})
        if contract.get("review_contract") != schema["required_review_contract"]:
            blockers.append(
                {
                    "code": "retry_authority_review_contract_invalid",
                    "message": "review contract drifted",
                }
            )
        if (
            schema.get("required_positive_preflight_schema_version")
            != POSITIVE_PREFLIGHT_SCHEMA_VERSION
        ):
            blockers.append(
                {
                    "code": "retry_authority_positive_preflight_schema_invalid",
                    "message": "positive preflight schema version drifted",
                }
            )
        if slot.get("assignment_count") != 15 or slot.get("total_attempt_slot_count") != 30:
            blockers.append({"code": "retry_authority_slot_count_invalid", "message": "15 assignments and 30 slots are required"})
        if slot.get("initial_command_indices") != list(range(15)) or slot.get("retry_command_indices") != list(range(15, 30)):
            blockers.append({"code": "retry_authority_command_indices_invalid", "message": "command index partition drifted"})
        if any(value != 0 for value in dict(contract["access_counters"]).values()):
            blockers.append({"code": "retry_authority_access_open", "message": "all access counters must remain zero"})
        if any(value is not False for value in dict(contract["closed_flags"]).values()):
            blockers.append({"code": "retry_authority_gate_open", "message": "all downstream flags must remain false"})
        if contract.get("candidate_dispatch_authorized") is not False or contract.get("scientific_twin_result") != "NOT_RUN":
            blockers.append({"code": "retry_authority_scientific_boundary_invalid", "message": "dispatch must remain false and Twin NOT_RUN"})
        if len(list(contract.get("source_bindings", []))) != 10:
            blockers.append(
                {
                    "code": "retry_authority_source_binding_count_invalid",
                    "message": "exactly ten retry-authority source bindings are required",
                }
            )
    except (KeyError, TypeError, ValueError, OperationalTwinProtocolError) as error:
        blockers.append({"code": "retry_authority_config_invalid", "message": str(error)})
    return blockers


def _authenticate(config: Mapping[str, object], root: Path) -> dict[str, object]:
    contract = _config(config)
    blockers = validate_retry_authority_config(config)
    if blockers:
        raise OperationalTwinProtocolError(str(blockers))
    r47 = dict(contract["round_47_authority"])
    r46 = dict(contract["round_46_authority"])
    for value in (r47["review_record"], r47["closeout_record"], r46["review_record"], r46["implementation_notes"], r46["blocker_summary"]):
        row = dict(value)
        _require_object(root, str(row["path"]), str(row["git_object_id"]))
    _require_object(root, str(r47["audit_path"]), str(r47["audit_git_object_id"]))
    _require_object(root, str(r47["preflight_path"]), str(r47["preflight_git_object_id"]))
    repair_audit = _git_json(root, "HEAD", str(r47["audit_path"]))
    if (
        repair_audit.get("status") != "PASS"
        or repair_audit.get("terminal_decision")
        != "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_REPAIRED"
        or repair_audit.get("candidate_dispatch_authorized") is not False
        or repair_audit.get("scientific_twin_result") != "NOT_RUN"
    ):
        raise OperationalTwinProtocolError(
            "retry_authority_round47_audit_invalid"
        )
    if _git_object(root, "HEAD", str(r46["candidate_tree_path"])) != r46["candidate_tree_id"]:
        raise OperationalTwinProtocolError("retry_authority_round46_candidate_tree_drift")
    if _git_object(root, "HEAD", str(r46["audit_tree_path"])) != r46["audit_tree_id"]:
        raise OperationalTwinProtocolError("retry_authority_round46_audit_tree_drift")
    if _git_object(root, str(r46["merged_commit"]), str(r46["candidate_tree_path"])) != r46["candidate_tree_id"]:
        raise OperationalTwinProtocolError("retry_authority_round46_candidate_tree_history_drift")
    if _git_object(root, str(r46["merged_commit"]), str(r46["audit_tree_path"])) != r46["audit_tree_id"]:
        raise OperationalTwinProtocolError("retry_authority_round46_audit_tree_history_drift")
    artifacts = dict(contract["repaired_artifacts"])
    loaded = {
        key: _read_committed_json(
            root,
            str(artifacts[key]),
            require_materialized=False,
        )[0]
        for key in (
            "candidate_lineage_manifest",
            "assignment_manifest",
            "assignment_commands",
            "preflight_authority",
        )
    }
    candidate = loaded["candidate_lineage_manifest"]
    assignments = loaded["assignment_manifest"]
    commands = loaded["assignment_commands"]
    preflight = loaded["preflight_authority"]
    expected_candidate = build_candidate_lineage_manifest(config)
    expected_assignments = build_candidate_assignment_manifest(
        config,
        expected_candidate,
    )
    expected_commands = build_candidate_commands(
        config,
        expected_candidate,
        expected_assignments,
    )
    if candidate != expected_candidate:
        raise OperationalTwinProtocolError(
            "retry_authority_candidate_manifest_drift"
        )
    if assignments != expected_assignments:
        raise OperationalTwinProtocolError(
            "retry_authority_assignment_manifest_drift"
        )
    if commands != expected_commands:
        raise OperationalTwinProtocolError(
            "retry_authority_command_catalog_drift"
        )
    if candidate.get("candidate_manifest_hash") != canonical_hash(
        {key: value for key, value in candidate.items() if key != "candidate_manifest_hash"}
    ):
        raise OperationalTwinProtocolError(
            "retry_authority_candidate_hash_invalid"
        )
    if assignments.get("assignment_manifest_hash") != canonical_hash(
        {
            key: value
            for key, value in assignments.items()
            if key != "assignment_manifest_hash"
        }
    ):
        raise OperationalTwinProtocolError(
            "retry_authority_assignment_hash_invalid"
        )
    if commands.get("command_catalog_hash") != command_catalog_hash(commands):
        raise OperationalTwinProtocolError(
            "retry_authority_command_hash_invalid"
        )
    expected = {
        "candidate_lineage_uid": contract["candidate_lineage_uid"],
        "protocol_uid": contract["repaired_protocol_uid"],
        "preflight_uid": contract["repaired_preflight_uid"],
    }
    for payload in (candidate, assignments, commands, preflight):
        for field, value in expected.items():
            if payload.get(field) != value:
                raise OperationalTwinProtocolError(f"retry_authority_repaired_{field}_drift")
    if candidate.get("candidate_manifest_hash") != contract["repaired_candidate_manifest_hash"]:
        raise OperationalTwinProtocolError("retry_authority_candidate_hash_drift")
    if assignments.get("assignment_manifest_hash") != contract["repaired_assignment_manifest_hash"]:
        raise OperationalTwinProtocolError("retry_authority_assignment_hash_drift")
    if commands.get("command_catalog_hash") != contract["repaired_command_catalog_hash"]:
        raise OperationalTwinProtocolError("retry_authority_command_hash_drift")
    if preflight.get("source_identity_hash") != contract["repaired_source_identity_hash"]:
        raise OperationalTwinProtocolError("retry_authority_source_identity_drift")
    if (
        preflight.get("schema_version") != CANDIDATE_PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status") != "PASS"
        or preflight.get("terminal_decision")
        != "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_REPAIRED"
        or preflight.get("candidate_manifest_hash")
        != candidate["candidate_manifest_hash"]
        or preflight.get("assignment_manifest_hash")
        != assignments["assignment_manifest_hash"]
        or preflight.get("command_catalog_hash") != commands["command_catalog_hash"]
        or preflight.get("assignment_count") != 15
        or preflight.get("attempt_slot_count") != 30
    ):
        raise OperationalTwinProtocolError(
            "retry_authority_repaired_preflight_invalid"
        )
    dispatch = _git_json(root, "HEAD", str(r46["dispatch_manifest"]))
    initial_arrays = dispatch.get("initial_arrays")
    dispatched_indices = (
        [
            int(index)
            for array in initial_arrays
            for index in dict(array).get("command_indices", [])
        ]
        if isinstance(initial_arrays, list)
        else []
    )
    if (
        dispatch.get("status") != "INITIAL_ARRAYS_SUBMITTED"
        or dispatch.get("candidate_lineage_uid") != contract["candidate_lineage_uid"]
        or dispatch.get("recovery_arrays") != []
        or sorted(dispatched_indices) != list(range(15))
        or len(dispatched_indices) != len(set(dispatched_indices))
    ):
        raise OperationalTwinProtocolError("retry_authority_dispatch_evidence_invalid")
    return {
        "candidate": candidate,
        "assignments": assignments,
        "commands": commands,
        "preflight": preflight,
        "dispatch": dispatch,
    }


def adjudicate_retry_authority(config: Mapping[str, object], root: Path) -> dict[str, object]:
    contract = _config(config)
    evidence = _authenticate(config, root)
    slot_contract = dict(contract["slot_contract"])
    protocol = dict(config["operational_twin_candidate_lineage_protocol"])
    commands = {int(row["command_index"]): dict(row) for row in evidence["commands"]["commands"]}
    command_by_attempt = {str(row["attempt_uid"]): row for row in commands.values()}
    prepublication = set(dict(contract["round_46_authority"])["prepublication_command_indices"])
    rows: list[dict[str, object]] = []
    uniqueness = {"assignment_uid": [], "retry_attempt_uid": [], "retry_command_index": [], "output_root": [], "execution_authority": [], "resource_environment_receipt": []}
    for assignment_value in evidence["assignments"]["assignments"]:
        assignment = dict(assignment_value)
        attempts = {str(row["attempt_kind"]): dict(row) for row in assignment["attempts"]}
        initial = attempts["initial"]
        retry = attempts["infrastructure_recovery_01"]
        initial_command = command_by_attempt[str(initial["attempt_uid"])]
        retry_command = command_by_attempt[str(retry["attempt_uid"])]
        initial_index = int(initial_command["command_index"])
        retry_index = int(retry_command["command_index"])
        terminal_path = f"{initial['output_root']}/terminal_status.json"
        phase_path = f"{initial['output_root']}/execution_phase_receipt.json"
        terminal_object = _git_object(root, "HEAD", terminal_path)
        predicates: dict[str, bool] = {
            "retry_attempt_kind": retry.get("attempt_kind") == slot_contract["retry_attempt_kind"],
            "retry_attempt_ordinal": retry.get("attempt_ordinal") == slot_contract["retry_attempt_ordinal"],
            "retry_conditional": retry.get("conditional") is True,
            "eligible_failure_class_contract": retry.get("eligible_after_failure_class") == slot_contract["eligible_after_failure_class"],
            "predecessor_identity": retry.get("predecessor_attempt_uid") == initial.get("attempt_uid"),
            "initial_command_index": initial_index in slot_contract["initial_command_indices"],
            "retry_command_index": retry_index in slot_contract["retry_command_indices"],
            "fresh_random_initialization": retry.get("fresh_random_initialization") is True,
            "state_resume_forbidden": retry.get("state_resume_allowed") is False,
        }
        exact_infrastructure_predecessor = False
        if terminal_object is not None:
            terminal = _git_json(root, "HEAD", terminal_path)
            phase = _git_json(root, "HEAD", phase_path)
            for field, expected in {"assignment_uid": assignment["assignment_uid"], "attempt_uid": initial["attempt_uid"], "method_id": assignment["method_id"], "seed": assignment["seed"], "output_root": initial["output_root"]}.items():
                if terminal.get(field) != expected or phase.get(field) != expected:
                    raise OperationalTwinProtocolError(f"retry_authority_materialized_identity_drift:{field}")
            classification = classify_recovery_eligibility(
                {"status": terminal.get("status"), "failure_class": terminal.get("failure_class"), "validation_blockers": []},
                infrastructure_failure_class=str(slot_contract["eligible_after_failure_class"]),
            )
            observed_kind = "materialized_evaluation_failure"
            observed_failure_class = terminal.get("failure_class")
            predecessor_identity = {"terminal_status": terminal_object, "phase_receipt": _git_object(root, "HEAD", phase_path)}
            exact_infrastructure_predecessor = _exact_infrastructure_predecessor(
                config,
                protocol=protocol,
                assignment=assignment,
                initial=initial,
                initial_command=initial_command,
                terminal=terminal,
                phase=phase,
                failure_class=str(slot_contract["eligible_after_failure_class"]),
            )
        elif initial_index in prepublication:
            classification = "NOT_ELIGIBLE_UNKNOWN_PREPUBLICATION_FAILURE"
            observed_kind = "prepublication_scheduler_failure"
            observed_failure_class = None
            predecessor_identity = {"terminal_status": None, "phase_receipt": None}
        else:
            raise OperationalTwinProtocolError(f"retry_authority_initial_evidence_missing:{initial_index}")
        predicates["predecessor_failure_class_eligible"] = (
            observed_failure_class == slot_contract["eligible_after_failure_class"]
            and exact_infrastructure_predecessor
        )
        for field in ("resume_input", "history_input", "checkpoint_input", "optimizer_state_input", "model_state_input", "calibration_state_input"):
            predicates[f"null_{field}"] = dict(protocol["initialization"]).get(field) is None
        paths = {"output_root": str(retry["output_root"]), "execution_authority": str(retry["execution_authority"]), "resource_environment_receipt": str(retry["resource_environment_receipt"])}
        expected_paths = _expected_attempt_paths(
            protocol,
            assignment,
            retry,
        )
        path_roots = {
            "output_root": CANDIDATE_LINEAGE_ROOT,
            "execution_authority": Path(
                "Output/audits/operational_twin/candidate_execution_authorities"
            ),
            "resource_environment_receipt": Path(
                "Output/audits/operational_twin/candidate_resource_environment_receipts"
            ),
        }
        for label, path in paths.items():
            try:
                _, resolved_path = _canonical_relative_path(
                    root,
                    path,
                    label=label,
                    expected=str(expected_paths[label]),
                    allowed_root=path_roots[label],
                )
                if label == "output_root":
                    candidate_attempt_path(root, path, require_absent=False)
                if os.path.lexists(resolved_path) and stat.S_ISLNK(
                    resolved_path.lstat().st_mode
                ):
                    raise OperationalTwinProtocolError(
                        f"{label}_path_symlink: retry target must not be a symlink"
                    )
                canonical = True
            except OperationalTwinProtocolError:
                canonical = False
                resolved_path = None
            predicates[f"{label}_canonical"] = canonical
            predicates[f"{label}_unused_git"] = (
                canonical and _git_object(root, "HEAD", path) is None
            )
            predicates[f"{label}_unused_filesystem"] = (
                canonical
                and resolved_path is not None
                and not os.path.lexists(resolved_path)
            )
            uniqueness[label].append(path)
        uniqueness["assignment_uid"].append(str(assignment["assignment_uid"]))
        uniqueness["retry_attempt_uid"].append(str(retry["attempt_uid"]))
        uniqueness["retry_command_index"].append(retry_index)
        rows.append({
            "candidate_lineage_uid": assignment["candidate_lineage_uid"], "assignment_uid": assignment["assignment_uid"], "method_id": assignment["method_id"], "seed": assignment["seed"], "resource_profile": assignment["resource_profile"],
            "initial_attempt_uid": initial["attempt_uid"], "initial_command_index": initial_index, "initial_output_root": initial["output_root"], "predecessor_evidence_kind": observed_kind, "predecessor_failure_class": observed_failure_class, "historical_recovery_classification": classification, "predecessor_identity": predecessor_identity,
            "retry_attempt_uid": retry["attempt_uid"], "retry_command_index": retry_index, "retry_attempt_kind": retry["attempt_kind"], "retry_attempt_ordinal": retry["attempt_ordinal"], "retry_output_root": retry["output_root"], "retry_execution_authority": retry["execution_authority"], "retry_resource_environment_receipt": retry["resource_environment_receipt"],
            "predicates": predicates, "result": SUCCESSOR_ROW,
        })
    path_owners: dict[str, list[tuple[dict[str, object], str]]] = {}
    for row in rows:
        for field, predicate in (
            ("retry_output_root", "output_root_canonical"),
            ("retry_execution_authority", "execution_authority_canonical"),
            (
                "retry_resource_environment_receipt",
                "resource_environment_receipt_canonical",
            ),
        ):
            path_owners.setdefault(str(row[field]), []).append((row, predicate))
    for owners in path_owners.values():
        if len(owners) > 1:
            for row, predicate in owners:
                dict(row["predicates"])[predicate] = False
                row["predicates"][predicate] = False
    for row in rows:
        row["result"] = (
            AUTHORIZED
            if all(dict(row["predicates"]).values())
            and row["historical_recovery_classification"] == "ELIGIBLE"
            else SUCCESSOR_ROW
        )
    global_predicates = {
        "assignment_count": len(rows) == 15,
        "initial_slot_count": len({row["initial_attempt_uid"] for row in rows}) == 15,
        "retry_slot_count": len({row["retry_attempt_uid"] for row in rows}) == 15,
        **{f"unique_{key}": len(values) == len(set(values)) == 15 for key, values in uniqueness.items()},
        "all_access_counters_zero": all(value == 0 for value in dict(contract["access_counters"]).values()),
        "all_downstream_flags_closed": all(value is False for value in dict(contract["closed_flags"]).values()),
        "candidate_dispatch_authorized_false": contract["candidate_dispatch_authorized"] is False,
        "scientific_twin_not_run": contract["scientific_twin_result"] == "NOT_RUN",
    }
    positive = all(row["result"] == AUTHORIZED for row in rows) and all(global_predicates.values())
    terminal = POSITIVE if positive else SUCCESSOR_REQUIRED
    directions = dict(contract["next_directions"])
    audit = {
        "schema_version": "neuronsifter-operational-twin-candidate-retry-authority-audit-v1",
        "status": "PASS", "authoritative_date": contract["authoritative_date"],
        "terminal_decision": terminal,
        "next_direction": directions["positive" if positive else "successor_required"],
        "candidate_lineage_uid": contract["candidate_lineage_uid"],
        "repaired_protocol_uid": contract["repaired_protocol_uid"], "repaired_preflight_uid": contract["repaired_preflight_uid"],
        "assignment_count": len(rows), "authorized_row_count": sum(row["result"] == AUTHORIZED for row in rows), "successor_required_row_count": sum(row["result"] == SUCCESSOR_ROW for row in rows),
        "rows": rows, "global_predicates": global_predicates,
        "round_46_candidate_tree_id": dict(contract["round_46_authority"])["candidate_tree_id"], "round_46_audit_tree_id": dict(contract["round_46_authority"])["audit_tree_id"],
        "candidate_dispatch_authorized": False, "side_effect_count": 0, "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"], "closed_flags": contract["closed_flags"],
        "positive_artifacts_materialized": _positive_artifacts_materialized(
            config,
            root,
        ),
    }
    audit["decision_uid"] = _decision_uid(audit)
    return audit


def render_retry_authority_report(audit: Mapping[str, object]) -> str:
    return "\n".join([
        "# Operational Twin Candidate Retry Authority", "",
        f"- Status: `{audit['status']}`", f"- Terminal: `{audit['terminal_decision']}`",
        f"- Next direction: `{audit['next_direction']}`", f"- Assignments: `{audit['assignment_count']}`",
        f"- Existing-lineage slots authorized: `{audit['authorized_row_count']}`",
        f"- Successor-lineage required rows: `{audit['successor_required_row_count']}`",
        "- Candidate dispatch authorized: `false`", "- Scientific Twin result: `NOT_RUN`", "",
        "All ten access counters remain zero. No candidate job, attempt, authority, receipt, checkpoint, recovery artifact, aggregation, catalog, or scientific result was created.", "",
    ])


def _positive_protocol_config(
    config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    updated = deepcopy(dict(config))
    protocol = deepcopy(dict(updated["operational_twin_candidate_lineage_protocol"]))
    retry_contract = _config(config)
    output = dict(retry_contract["output_paths"])
    protocol["output_paths"] = {
        **dict(protocol["output_paths"]),
        **{
            key: output[key]
            for key in (
                "protocol_root",
                "candidate_lineage_manifest",
                "assignment_manifest",
                "assignment_commands",
                "preflight_authority",
                "preflight_report",
            )
        },
    }
    authority = dict(protocol["execution_authority_contract"])
    review = dict(retry_contract["review_contract"])
    authority.update(
        {
            "approved_protocol_review_path": review["path"],
            "required_review_schema_version": review["schema_version"],
            "required_review_round_id": review["round_id"],
            "required_review_direction_id": review["direction_id"],
            "required_review_decision": review["decision"],
        }
    )
    protocol["execution_authority_contract"] = authority
    source_names = (
        "retry_authority_config_stack",
        "retry_authority_contract",
        "retry_authority_schema",
        "retry_authority_utility",
        "retry_authority_protocol_builder",
        "retry_authority_validator",
        "retry_authority_execution_submitter",
        "retry_authority_tests",
        "retry_authority_protocol_tests",
        "retry_authority_execution_tests",
    )
    source_paths = list(retry_contract["source_bindings"])
    if len(source_paths) != len(source_names):
        raise OperationalTwinProtocolError(
            "retry_authority_source_binding_count_invalid"
        )
    protocol["source_bindings"] = {
        **dict(protocol["source_bindings"]),
        **dict(zip(source_names, source_paths, strict=True)),
    }
    protocol["protocol_uid"] = compute_candidate_protocol_uid(protocol)
    updated["operational_twin_candidate_lineage_protocol"] = protocol
    return updated, protocol


def _positive_preflight(
    config: Mapping[str, object],
    root: Path,
    audit: Mapping[str, object],
    candidate: Mapping[str, object],
    assignments: Mapping[str, object],
    commands: Mapping[str, object],
) -> dict[str, object]:
    preflight = build_candidate_preflight_authority(
        config,
        root,
        candidate,
        assignments,
        commands,
    )
    slot = dict(_config(config)["slot_contract"])
    preflight.update(
        {
            "schema_version": POSITIVE_PREFLIGHT_SCHEMA_VERSION,
            "terminal_decision": POSITIVE,
            "retry_authority_date": _config(config)["authoritative_date"],
            "retry_authority_decision_uid": audit["decision_uid"],
            "authorized_command_indices": list(slot["retry_command_indices"]),
            "historical_command_indices": list(slot["initial_command_indices"]),
            "attempt_roots_absent_at_freeze": False,
            "retry_attempt_roots_absent_at_freeze": True,
            "retry_execution_authorities_absent_at_freeze": True,
            "retry_resource_receipts_absent_at_freeze": True,
            "candidate_dispatch_authorized": False,
        }
    )
    return preflight


def render_positive_successor_report(preflight: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Operational Twin Candidate-Lineage Retry Preflight",
            "",
            f"- Status: `{preflight['status']}`",
            f"- Terminal: `{preflight['terminal_decision']}`",
            f"- Retry-authority date: `{preflight['retry_authority_date']}`",
            f"- Retry-authority decision UID: `{preflight['retry_authority_decision_uid']}`",
            f"- Candidate lineage UID: `{preflight['candidate_lineage_uid']}`",
            f"- Protocol UID: `{preflight['protocol_uid']}`",
            f"- Preflight UID: `{preflight['preflight_uid']}`",
            f"- Candidate manifest hash: `{preflight['candidate_manifest_hash']}`",
            f"- Assignment manifest hash: `{preflight['assignment_manifest_hash']}`",
            f"- Command catalog hash: `{preflight['command_catalog_hash']}`",
            "- Historical command indices: `0..14`",
            "- Authorized retry command indices: `15..29`",
            "- Retry attempt roots absent at freeze: `true`",
            "- Retry execution authorities absent at freeze: `true`",
            "- Retry resource receipts absent at freeze: `true`",
            "- Candidate dispatch authorized: `false`",
            "- Scientific Twin result: `NOT_RUN`",
            "",
        ]
    )


def validate_positive_retry_audit(
    config: Mapping[str, object],
    audit: Mapping[str, object],
    *,
    expected_materialized: bool,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    contract = _config(config)
    slot = dict(contract["slot_contract"])

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    if set(audit) != AUDIT_KEYS:
        block("retry_authority_positive_audit_keyset_invalid", "audit keyset drifted")
    expected_fields = {
        "schema_version": "neuronsifter-operational-twin-candidate-retry-authority-audit-v1",
        "status": "PASS",
        "authoritative_date": contract["authoritative_date"],
        "terminal_decision": POSITIVE,
        "next_direction": dict(contract["next_directions"])["positive"],
        "candidate_lineage_uid": contract["candidate_lineage_uid"],
        "repaired_protocol_uid": contract["repaired_protocol_uid"],
        "repaired_preflight_uid": contract["repaired_preflight_uid"],
        "assignment_count": 15,
        "authorized_row_count": 15,
        "successor_required_row_count": 0,
        "round_46_candidate_tree_id": dict(contract["round_46_authority"])[
            "candidate_tree_id"
        ],
        "round_46_audit_tree_id": dict(contract["round_46_authority"])[
            "audit_tree_id"
        ],
        "candidate_dispatch_authorized": False,
        "side_effect_count": 0,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"],
        "closed_flags": contract["closed_flags"],
        "positive_artifacts_materialized": expected_materialized,
    }
    for field, expected in expected_fields.items():
        if audit.get(field) != expected:
            block(
                "retry_authority_positive_audit_field_invalid",
                f"{field} drifted",
            )
    rows = audit.get("rows")
    if not isinstance(rows, list) or len(rows) != 15:
        block("retry_authority_positive_rows_invalid", "exactly 15 rows are required")
        rows = []
    initial_indices: list[int] = []
    retry_indices: list[int] = []
    assignment_uids: list[str] = []
    retry_attempt_uids: list[str] = []
    retry_paths: list[str] = []
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping) or set(value) != ROW_KEYS:
            block(
                "retry_authority_positive_row_keyset_invalid",
                f"row {index} keyset drifted",
            )
            continue
        row = dict(value)
        predicates = row.get("predicates")
        if (
            not isinstance(predicates, Mapping)
            or set(predicates) != ROW_PREDICATE_KEYS
            or any(value is not True for value in predicates.values())
        ):
            block(
                "retry_authority_positive_row_predicates_invalid",
                f"row {index} is not fully positive",
            )
        predecessor = row.get("predecessor_identity")
        if (
            not isinstance(predecessor, Mapping)
            or set(predecessor) != {"terminal_status", "phase_receipt"}
            or any(
                not isinstance(value, str) or len(value) != 40
                for value in predecessor.values()
            )
        ):
            block(
                "retry_authority_positive_predecessor_identity_invalid",
                f"row {index} predecessor identity drifted",
            )
        row_expected = {
            "candidate_lineage_uid": contract["candidate_lineage_uid"],
            "predecessor_evidence_kind": "materialized_evaluation_failure",
            "predecessor_failure_class": slot["eligible_after_failure_class"],
            "historical_recovery_classification": "ELIGIBLE",
            "retry_attempt_kind": slot["retry_attempt_kind"],
            "retry_attempt_ordinal": slot["retry_attempt_ordinal"],
            "result": AUTHORIZED,
        }
        for field, expected in row_expected.items():
            if row.get(field) != expected:
                block(
                    "retry_authority_positive_row_field_invalid",
                    f"row {index} {field} drifted",
                )
        try:
            initial_indices.append(int(row["initial_command_index"]))
            retry_indices.append(int(row["retry_command_index"]))
            assignment_uids.append(str(row["assignment_uid"]))
            retry_attempt_uids.append(str(row["retry_attempt_uid"]))
            retry_paths.extend(
                str(row[field])
                for field in (
                    "retry_output_root",
                    "retry_execution_authority",
                    "retry_resource_environment_receipt",
                )
            )
        except (KeyError, TypeError, ValueError):
            block(
                "retry_authority_positive_row_identity_invalid",
                f"row {index} identity is malformed",
            )
    if initial_indices != list(slot["initial_command_indices"]):
        block(
            "retry_authority_positive_initial_allowlist_invalid",
            "historical indices must be exactly 0..14 in order",
        )
    if retry_indices != list(slot["retry_command_indices"]):
        block(
            "retry_authority_positive_retry_allowlist_invalid",
            "retry indices must be exactly 15..29 in order",
        )
    if len(set(assignment_uids)) != 15 or len(set(retry_attempt_uids)) != 15:
        block(
            "retry_authority_positive_identity_bijection_invalid",
            "assignment and retry-attempt identities must be bijective",
        )
    if len(set(retry_paths)) != 45:
        block(
            "retry_authority_positive_path_collision",
            "all 45 retry targets must be disjoint",
        )
    predicates = audit.get("global_predicates")
    if (
        not isinstance(predicates, Mapping)
        or set(predicates) != GLOBAL_PREDICATE_KEYS
        or any(value is not True for value in predicates.values())
    ):
        block(
            "retry_authority_positive_global_predicates_invalid",
            "global predicate ledger is not fully positive",
        )
    if audit.get("decision_uid") != _decision_uid(audit):
        block(
            "retry_authority_positive_decision_uid_invalid",
            "decision UID does not recompute",
        )
    return blockers


def _validate_retry_targets(
    config: Mapping[str, object],
    root: Path,
    protocol: Mapping[str, object],
    assignments: Mapping[str, object],
    commands: Mapping[str, object],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    slot = dict(_config(config)["slot_contract"])
    commands_by_index = {
        int(row["command_index"]): dict(row) for row in commands.get("commands", [])
    }
    observed_paths: list[str] = []
    observed_indices: list[int] = []
    for assignment_value in assignments.get("assignments", []):
        assignment = dict(assignment_value)
        retry_matches = [
            dict(attempt)
            for attempt in assignment.get("attempts", [])
            if dict(attempt).get("attempt_kind") == slot["retry_attempt_kind"]
        ]
        if len(retry_matches) != 1:
            blockers.append(
                {
                    "code": "retry_authority_positive_retry_slot_invalid",
                    "message": f"{assignment.get('assignment_uid')} lacks exactly one retry slot",
                }
            )
            continue
        retry = retry_matches[0]
        command_matches = [
            row
            for row in commands_by_index.values()
            if row.get("attempt_uid") == retry.get("attempt_uid")
        ]
        if len(command_matches) != 1:
            blockers.append(
                {
                    "code": "retry_authority_positive_retry_command_invalid",
                    "message": f"{retry.get('attempt_uid')} lacks exactly one command",
                }
            )
            continue
        command = command_matches[0]
        index = int(command["command_index"])
        observed_indices.append(index)
        expected_paths = _expected_attempt_paths(
            protocol,
            assignment,
            retry,
        )
        declared_paths = {
            "output_root": str(retry.get("output_root", "")),
            "execution_authority": str(retry.get("execution_authority", "")),
            "resource_environment_receipt": str(
                retry.get("resource_environment_receipt", "")
            ),
        }
        allowed_roots = {
            "output_root": CANDIDATE_LINEAGE_ROOT,
            "execution_authority": Path(
                "Output/audits/operational_twin/candidate_execution_authorities"
            ),
            "resource_environment_receipt": Path(
                "Output/audits/operational_twin/candidate_resource_environment_receipts"
            ),
        }
        for label, declared in declared_paths.items():
            observed_paths.append(declared)
            try:
                _require_unused_retry_target(
                    root,
                    declared,
                    label=f"retry_authority_{label}",
                    expected=str(expected_paths[label]),
                    allowed_root=allowed_roots[label],
                )
                if label == "output_root":
                    candidate_attempt_path(root, declared, require_absent=True)
            except OperationalTwinProtocolError as error:
                blockers.append(
                    {
                        "code": "retry_authority_positive_retry_target_invalid",
                        "message": str(error),
                    }
                )
    if observed_indices != list(slot["retry_command_indices"]):
        blockers.append(
            {
                "code": "retry_authority_positive_retry_allowlist_invalid",
                "message": "retry commands must be exactly 15..29 in order",
            }
        )
    if len(observed_paths) != 45 or len(set(observed_paths)) != 45:
        blockers.append(
            {
                "code": "retry_authority_positive_path_collision",
                "message": "retry output, authority, and receipt paths must be disjoint",
            }
        )
    return blockers


def validate_positive_successor_package(
    config: Mapping[str, object],
    root: Path,
    audit: Mapping[str, object],
    package: Mapping[str, object],
    *,
    expected_audit_materialized: bool,
    require_publication_targets_absent: bool,
) -> list[dict[str, str]]:
    blockers = validate_positive_retry_audit(
        config,
        audit,
        expected_materialized=expected_audit_materialized,
    )
    if set(package) != set(POSITIVE_ARTIFACT_KEYS):
        blockers.append(
            {
                "code": "retry_authority_positive_package_keyset_invalid",
                "message": "positive package keyset drifted",
            }
        )
        return blockers
    updated, protocol = _positive_protocol_config(config)
    candidate = build_candidate_lineage_manifest(updated)
    assignments = build_candidate_assignment_manifest(updated, candidate)
    commands = build_candidate_commands(updated, candidate, assignments)
    preflight = _positive_preflight(
        updated,
        root,
        audit,
        candidate,
        assignments,
        commands,
    )
    expected = {
        "candidate_lineage_manifest": candidate,
        "assignment_manifest": assignments,
        "assignment_commands": commands,
        "preflight_authority": preflight,
        "preflight_report": render_positive_successor_report(preflight),
    }
    for key, expected_value in expected.items():
        if package.get(key) != expected_value:
            blockers.append(
                {
                    "code": "retry_authority_positive_artifact_mismatch",
                    "message": key,
                }
            )
    candidate_value = package.get("candidate_lineage_manifest")
    assignment_value = package.get("assignment_manifest")
    command_value = package.get("assignment_commands")
    preflight_value = package.get("preflight_authority")
    actual_candidate = (
        dict(candidate_value) if isinstance(candidate_value, Mapping) else {}
    )
    actual_assignments = (
        dict(assignment_value) if isinstance(assignment_value, Mapping) else {}
    )
    actual_commands = dict(command_value) if isinstance(command_value, Mapping) else {}
    if isinstance(candidate_value, Mapping):
        candidate_payload = dict(candidate_value)
        candidate_hash = candidate_payload.pop("candidate_manifest_hash", None)
        if (
            candidate_value.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION
            or candidate_hash != canonical_hash(candidate_payload)
        ):
            blockers.append(
                {
                    "code": "retry_authority_positive_candidate_hash_invalid",
                    "message": "candidate manifest schema or hash drifted",
                }
            )
    if isinstance(assignment_value, Mapping):
        assignment_payload = dict(assignment_value)
        assignment_hash = assignment_payload.pop("assignment_manifest_hash", None)
        if assignment_hash != canonical_hash(assignment_payload):
            blockers.append(
                {
                    "code": "retry_authority_positive_assignment_hash_invalid",
                    "message": "assignment manifest hash drifted",
                }
            )
    if isinstance(command_value, Mapping):
        command_payload = dict(command_value)
        if command_payload.get("command_catalog_hash") != command_catalog_hash(
            command_payload
        ):
            blockers.append(
                {
                    "code": "retry_authority_positive_command_hash_invalid",
                    "message": "command catalog hash drifted",
                }
            )
        for index, row in enumerate(command_payload.get("commands", [])):
            if not isinstance(row, Mapping) or set(row) != CANDIDATE_COMMAND_ROW_KEYS:
                blockers.append(
                    {
                        "code": "retry_authority_positive_command_keyset_invalid",
                        "message": f"command {index} keyset drifted",
                    }
                )
    if isinstance(preflight_value, Mapping):
        preflight_payload = dict(preflight_value)
        if (
            preflight_payload.get("schema_version")
            != POSITIVE_PREFLIGHT_SCHEMA_VERSION
            or preflight_payload.get("preflight_uid")
            != compute_candidate_preflight_uid(protocol)
            or preflight_payload.get("source_identity_hash")
            != canonical_hash(
                {
                    "config_identity": preflight_payload.get("config_identity"),
                    "schema_identity": preflight_payload.get("schema_identity"),
                    "source_identities": preflight_payload.get("source_identities"),
                }
            )
        ):
            blockers.append(
                {
                    "code": "retry_authority_positive_preflight_identity_invalid",
                    "message": "positive preflight schema, UID, or source identity drifted",
                }
            )
        slot = dict(_config(config)["slot_contract"])
        if (
            preflight_payload.get("authorized_command_indices")
            != list(slot["retry_command_indices"])
            or preflight_payload.get("historical_command_indices")
            != list(slot["initial_command_indices"])
            or preflight_payload.get("retry_authority_decision_uid")
            != audit.get("decision_uid")
            or preflight_payload.get("retry_authority_date")
            != _config(config)["authoritative_date"]
            or preflight_payload.get("retry_attempt_roots_absent_at_freeze")
            is not True
            or preflight_payload.get(
                "retry_execution_authorities_absent_at_freeze"
            )
            is not True
            or preflight_payload.get("retry_resource_receipts_absent_at_freeze")
            is not True
            or preflight_payload.get("candidate_dispatch_authorized") is not False
        ):
            blockers.append(
                {
                    "code": "retry_authority_positive_preflight_contract_invalid",
                    "message": "positive preflight allowlist or closed boundary drifted",
                }
            )
    blockers.extend(
        _validate_retry_targets(
            config,
            root,
            protocol,
            actual_assignments,
            actual_commands,
        )
    )
    predecessor_candidate = read_repo_json(
        root,
        str(dict(_config(config)["repaired_artifacts"])["candidate_lineage_manifest"]),
    )
    predecessor_assignments = read_repo_json(
        root,
        str(dict(_config(config)["repaired_artifacts"])["assignment_manifest"]),
    )
    if (
        actual_candidate.get("candidate_lineage_uid")
        != predecessor_candidate.get("candidate_lineage_uid")
        or actual_candidate.get("assignment_uids")
        != predecessor_candidate.get("assignment_uids")
        or actual_candidate.get("attempt_uids")
        != predecessor_candidate.get("attempt_uids")
    ):
        blockers.append(
            {
                "code": "retry_authority_positive_lineage_identity_drift",
                "message": "candidate lineage, assignment, or attempt identities changed",
            }
        )
    predecessor_rows = [dict(row) for row in predecessor_assignments["assignments"]]
    successor_rows = [
        dict(row) for row in actual_assignments.get("assignments", [])
    ]
    if len(predecessor_rows) != len(successor_rows):
        blockers.append(
            {
                "code": "retry_authority_positive_assignment_semantics_drift",
                "message": "assignment count changed",
            }
        )
    for predecessor, successor in zip(predecessor_rows, successor_rows):
        normalized_predecessor = deepcopy(predecessor)
        normalized_successor = deepcopy(successor)
        normalized_predecessor["protocol_uid"] = normalized_successor["protocol_uid"]
        for attempt_index, attempt in enumerate(normalized_predecessor["attempts"]):
            attempt["protocol_uid"] = normalized_successor["attempts"][attempt_index][
                "protocol_uid"
            ]
        if normalized_predecessor != normalized_successor:
            blockers.append(
                {
                    "code": "retry_authority_positive_assignment_semantics_drift",
                    "message": str(successor.get("assignment_uid")),
                }
            )
    if require_publication_targets_absent:
        publication_paths = {
            **_positive_artifact_paths(config),
            "decision_audit": str(dict(_config(config)["output_paths"])["decision_audit"]),
            "decision_report": str(dict(_config(config)["output_paths"])["decision_report"]),
        }
        for label, value in publication_paths.items():
            try:
                _, path = _canonical_relative_path(
                    root,
                    value,
                    label=f"retry_authority_{label}",
                    expected=value,
                )
                if os.path.lexists(path):
                    raise OperationalTwinProtocolError(
                        f"retry_authority_{label}_collision: publication target exists"
                    )
            except OperationalTwinProtocolError as error:
                blockers.append(
                    {
                        "code": "retry_authority_positive_publication_target_invalid",
                        "message": str(error),
                    }
                )
    return blockers


def derive_positive_successor(
    config: Mapping[str, object], root: Path, audit: Mapping[str, object]
) -> dict[str, object]:
    recomputed = adjudicate_retry_authority(config, root)
    if dict(audit) != recomputed:
        raise OperationalTwinProtocolError(
            "retry_authority_positive_audit_not_recomputed"
        )
    blockers = validate_positive_retry_audit(
        config,
        audit,
        expected_materialized=False,
    )
    if blockers:
        raise OperationalTwinProtocolError(str(blockers))
    updated, _ = _positive_protocol_config(config)
    candidate = build_candidate_lineage_manifest(updated)
    assignments = build_candidate_assignment_manifest(updated, candidate)
    commands = build_candidate_commands(updated, candidate, assignments)
    preflight = _positive_preflight(
        updated,
        root,
        audit,
        candidate,
        assignments,
        commands,
    )
    package: dict[str, object] = {
        "candidate_lineage_manifest": candidate,
        "assignment_manifest": assignments,
        "assignment_commands": commands,
        "preflight_authority": preflight,
        "preflight_report": render_positive_successor_report(preflight),
    }
    blockers = validate_positive_successor_package(
        config,
        root,
        audit,
        package,
        expected_audit_materialized=False,
        require_publication_targets_absent=False,
    )
    if blockers:
        raise OperationalTwinProtocolError(str(blockers))
    return package


def materialized_positive_audit(audit: Mapping[str, object]) -> dict[str, object]:
    payload = deepcopy(dict(audit))
    payload["positive_artifacts_materialized"] = True
    payload["decision_uid"] = _decision_uid(payload)
    return payload


def _supplied_audit_path(root: Path, supplied: Path, expected: str) -> Path:
    _, expected_path = _canonical_relative_path(
        root,
        expected,
        label="candidate_retry_authority",
        expected=expected,
    )
    if supplied.is_absolute():
        supplied_is_exact = supplied.as_posix() == expected_path.as_posix()
    else:
        supplied_is_exact = supplied.as_posix() == expected
    if not supplied_is_exact:
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_path_mismatch"
        )
    return expected_path


def positive_review_binding_uid(
    audit: Mapping[str, object],
    package: Mapping[str, object],
    artifact_identities: Mapping[str, object],
) -> str:
    preflight = dict(package["preflight_authority"])
    binding = {
        "terminal_decision": audit["terminal_decision"],
        "decision_uid": audit["decision_uid"],
        "protocol_uid": preflight["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": preflight["candidate_manifest_hash"],
        "assignment_manifest_hash": preflight["assignment_manifest_hash"],
        "command_catalog_hash": preflight["command_catalog_hash"],
        "source_identity_hash": preflight["source_identity_hash"],
        "artifacts": artifact_identities,
    }
    return f"operational_twin_candidate_retry_review_{canonical_hash(binding)[:24]}"


def validate_retry_preparation_package(
    config: Mapping[str, object],
    package: Mapping[str, object],
) -> list[dict[str, str]]:
    """Validate retry preparation without predecessor execution authorities."""
    blockers: list[dict[str, str]] = []

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    required_objects = (
        "audit",
        "protocol",
        "config",
        "candidate_lineage_manifest",
        "assignment_manifest",
        "assignment_commands",
        "preflight_authority",
        "review",
    )
    missing = [
        key
        for key in (*required_objects, "review_binding_uid")
        if key not in package
    ]
    if missing or any(
        not isinstance(package.get(key), Mapping) for key in required_objects
    ):
        block(
            "retry_preparation_package_invalid",
            f"approved retry package is missing or malformed: {missing}",
        )
        return blockers

    contract = _config(config)
    slot = dict(contract["slot_contract"])
    audit = dict(package["audit"])
    protocol = dict(package["protocol"])
    runtime_config = dict(package["config"])
    candidate = dict(package["candidate_lineage_manifest"])
    assignments = dict(package["assignment_manifest"])
    commands = dict(package["assignment_commands"])
    preflight = dict(package["preflight_authority"])
    review = dict(package["review"])
    retry_indices = [int(value) for value in slot["retry_command_indices"]]

    if runtime_config.get("operational_twin_candidate_lineage_protocol") != protocol:
        block(
            "retry_preparation_protocol_config_mismatch",
            "positive package config does not contain the validated retry protocol",
        )
    if (
        audit.get("terminal_decision") != dict(contract["terminals"])["positive"]
        or audit.get("authorized_row_count") != 15
        or audit.get("successor_required_row_count") != 0
        or preflight.get("retry_authority_decision_uid") != audit.get("decision_uid")
        or preflight.get("authorized_command_indices") != retry_indices
        or preflight.get("historical_command_indices")
        != list(slot["initial_command_indices"])
        or preflight.get("candidate_dispatch_authorized") is not False
    ):
        block(
            "retry_preparation_authority_mismatch",
            "positive retry decision or command allowlist drifted",
        )

    identity_checks = (
        (
            protocol.get("candidate_lineage_uid"),
            candidate.get("candidate_lineage_uid"),
            assignments.get("candidate_lineage_uid"),
            commands.get("candidate_lineage_uid"),
            preflight.get("candidate_lineage_uid"),
        ),
        (
            protocol.get("protocol_uid"),
            candidate.get("protocol_uid"),
            assignments.get("protocol_uid"),
            commands.get("protocol_uid"),
            preflight.get("protocol_uid"),
        ),
        (
            candidate.get("preflight_uid"),
            assignments.get("preflight_uid"),
            commands.get("preflight_uid"),
            preflight.get("preflight_uid"),
        ),
        (
            candidate.get("candidate_manifest_hash"),
            assignments.get("candidate_manifest_hash"),
            commands.get("candidate_manifest_hash"),
            preflight.get("candidate_manifest_hash"),
        ),
        (
            assignments.get("assignment_manifest_hash"),
            commands.get("assignment_manifest_hash"),
            preflight.get("assignment_manifest_hash"),
        ),
        (
            commands.get("command_catalog_hash"),
            preflight.get("command_catalog_hash"),
        ),
    )
    if any(any(value != values[0] for value in values[1:]) for values in identity_checks):
        block(
            "retry_preparation_identity_mismatch",
            "protocol, preflight, or catalog identities drifted",
        )

    scientific = protocol.get("scientific_contract")
    scientific_hash = canonical_hash(scientific)
    if (
        not isinstance(scientific, Mapping)
        or candidate.get("scientific_contract_hash") != scientific_hash
        or preflight.get("scientific_contract_hash") != scientific_hash
    ):
        block(
            "retry_preparation_scientific_contract_mismatch",
            "scientific contract hash drifted",
        )
    blockers.extend(
        {
            "code": f"retry_preparation_{row['code']}",
            "message": row["message"],
        }
        for row in validate_candidate_command_catalog(
            runtime_config,
            candidate,
            assignments,
            commands,
        )
    )

    resource_gate_value = preflight.get("resource_execution_gate")
    resource_gate = (
        dict(resource_gate_value)
        if isinstance(resource_gate_value, Mapping)
        else {}
    )
    required_gate_values = {
        "fresh_sinfo_required_before_dispatch": True,
        "fresh_resource_evidence_required_before_dispatch": True,
        "new_resource_environment_receipt_required_before_dispatch": True,
        "independent_execution_authority_required_before_dispatch": True,
        "authority_bound_in_this_round": False,
        "resource_receipt_bound_in_this_round": False,
    }
    if any(resource_gate.get(key) != value for key, value in required_gate_values.items()):
        block(
            "retry_preparation_resource_contract_invalid",
            "positive preflight resource gate drifted",
        )
    resource_profiles: dict[str, dict[str, object]] = {}
    for key in ("no_fit_placement", "trainable_placement"):
        value = resource_gate.get(key)
        if isinstance(value, Mapping):
            placement = dict(value)
            profile = placement.pop("profile", None)
            if isinstance(profile, str):
                resource_profiles[profile] = placement
    assignments_by_uid = {
        str(row["assignment_uid"]): dict(row)
        for row in assignments.get("assignments", [])
        if isinstance(row, Mapping) and isinstance(row.get("assignment_uid"), str)
    }
    retry_commands = sorted(
        (
            dict(row)
            for row in commands.get("commands", [])
            if isinstance(row, Mapping) and row.get("command_index") in retry_indices
        ),
        key=lambda row: int(row["command_index"]),
    )
    if [row.get("command_index") for row in retry_commands] != retry_indices:
        block(
            "retry_preparation_allowlist_mismatch",
            "retry commands must be exactly 15..29 in order",
        )
    for command in retry_commands:
        assignment = assignments_by_uid.get(str(command.get("assignment_uid")))
        expected_resources = (
            resource_profiles.get(str(assignment.get("resource_profile")))
            if assignment is not None
            else None
        )
        if (
            assignment is None
            or expected_resources is None
            or expected_resources.get("exclusive") is not False
            or assignment.get("resource_requirements") != expected_resources
            or command.get("resource_requirements") != expected_resources
            or command.get("wall_time") != assignment.get("wall_time")
        ):
            block(
                "retry_preparation_resource_contract_mismatch",
                f"command {command.get('command_index')} resource contract drifted",
            )

    review_binding_uid = package.get("review_binding_uid")
    evidence_summary = review.get("evidence_summary")
    if (
        review.get("decision") != "approved"
        or not isinstance(review_binding_uid, str)
        or not isinstance(evidence_summary, str)
        or review_binding_uid not in evidence_summary
    ):
        block(
            "retry_preparation_review_binding_mismatch",
            "approved review is not bound to the exact positive retry package",
        )
    return blockers


def require_positive_retry_authority(
    config: Mapping[str, object], root: Path, audit_path: Path
) -> dict[str, object]:
    contract = _config(config)
    output_paths = dict(contract["output_paths"])
    canonical_audit_path = str(output_paths["decision_audit"])
    _supplied_audit_path(root, audit_path, canonical_audit_path)
    audit_bytes, audit_identity = _read_committed_file(
        root,
        canonical_audit_path,
        require_materialized=True,
    )
    try:
        audit = json.loads(audit_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_audit_invalid"
        ) from error
    if not isinstance(audit, dict):
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_audit_invalid"
        )
    recomputed = adjudicate_retry_authority(config, root)
    if audit != recomputed or audit_bytes != _json_bytes(recomputed):
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_audit_recomputation_mismatch"
        )
    blockers = validate_positive_retry_audit(
        config,
        audit,
        expected_materialized=True,
    )
    if blockers:
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_not_positive: " + str(blockers)
        )

    package: dict[str, object] = {}
    artifact_identities: dict[str, object] = {
        "decision_audit": audit_identity,
    }
    for key in POSITIVE_ARTIFACT_KEYS:
        path = str(output_paths[key])
        content, identity = _read_committed_file(
            root,
            path,
            require_materialized=True,
        )
        artifact_identities[key] = identity
        if key == "preflight_report":
            try:
                package[key] = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise OperationalTwinProtocolError(
                    "candidate_retry_authority_positive_report_invalid"
                ) from error
        else:
            try:
                value = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise OperationalTwinProtocolError(
                    f"candidate_retry_authority_positive_artifact_invalid: {key}"
                ) from error
            if not isinstance(value, dict):
                raise OperationalTwinProtocolError(
                    f"candidate_retry_authority_positive_artifact_invalid: {key}"
                )
            if content != _json_bytes(value):
                raise OperationalTwinProtocolError(
                    f"candidate_retry_authority_positive_artifact_encoding_invalid: {key}"
                )
            package[key] = value
    decision_report_path = str(output_paths["decision_report"])
    decision_report_bytes, decision_report_identity = _read_committed_file(
        root,
        decision_report_path,
        require_materialized=True,
    )
    if decision_report_bytes.decode("utf-8") != render_retry_authority_report(audit):
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_report_mismatch"
        )
    artifact_identities["decision_report"] = decision_report_identity
    blockers = validate_positive_successor_package(
        config,
        root,
        audit,
        package,
        expected_audit_materialized=True,
        require_publication_targets_absent=False,
    )
    if blockers:
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_positive_package_invalid: " + str(blockers)
        )

    review = dict(contract["review_contract"])
    review_path = str(review["path"])
    review_payload, review_identity = _read_committed_json(
        root,
        review_path,
        require_materialized=True,
    )
    if set(review_payload) != APPROVED_REVIEW_KEYS:
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_review_keyset_mismatch"
        )
    for field in (
        "schema_version",
        "round_id",
        "roadmap_id",
        "roadmap_revision",
        "roadmap_dir",
        "milestone_id",
        "direction_id",
        "extracted_item_id",
        "decision",
    ):
        if review_payload.get(field) != review.get(field):
            raise OperationalTwinProtocolError(
                f"candidate_retry_authority_review_{field}_mismatch"
            )
    if not isinstance(review_payload.get("roadmap_closeout"), Mapping):
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_review_closeout_missing"
        )
    review_binding_uid = positive_review_binding_uid(
        audit,
        package,
        artifact_identities,
    )
    evidence_summary = review_payload.get("evidence_summary")
    if (
        not isinstance(evidence_summary, str)
        or audit["terminal_decision"] not in evidence_summary
        or audit["decision_uid"] not in evidence_summary
        or review_binding_uid not in evidence_summary
    ):
        raise OperationalTwinProtocolError(
            "candidate_retry_authority_review_identity_binding_mismatch"
        )
    updated, protocol = _positive_protocol_config(config)
    return {
        "audit": audit,
        "protocol": protocol,
        "config": updated,
        **package,
        "artifact_identities": artifact_identities,
        "review": review_payload,
        "review_identity": review_identity,
        "review_binding_uid": review_binding_uid,
    }
