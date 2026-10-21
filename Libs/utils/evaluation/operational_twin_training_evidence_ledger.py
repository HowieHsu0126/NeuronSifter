"""Canonical compact evidence ledger for Operational Twin training.

This module deliberately contains no filesystem or scheduler access.  Callers
collect source records, reduce them to the typed identities below, and then use
the pure builder and validator to produce a compact, reviewable contract.  Large
checkpoints are represented by content identity rather than embedded or copied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import re
from typing import TypedDict

from Libs.utils.evaluation.operational_twin_training_protocol import (
    METHOD_EXECUTION_CONTRACTS,
    REQUIRED_METHOD_IDS,
    REQUIRED_ZERO_ACCESS_COUNTERS,
    build_assignments,
    canonical_hash,
)


TRAINING_EVIDENCE_LEDGER_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-training-evidence-ledger-v2"
)
TRAINING_EVIDENCE_LEDGER_UID = "operational_twin_training_evidence_ledger"
REQUIRED_RECEIPT_GROUPS = (
    "training",
    "evaluation",
    "reconciliation",
    "aggregation",
    "focused_validation",
)
SCHEDULER_RECEIPT_GROUPS = (
    "training",
    "evaluation",
)
PROCESS_RECEIPT_GROUPS = (
    "reconciliation",
    "aggregation",
    "focused_validation",
)
RECEIPT_KIND_BY_GROUP = {
    **{group: "scheduler" for group in SCHEDULER_RECEIPT_GROUPS},
    **{group: "process" for group in PROCESS_RECEIPT_GROUPS},
}
RECEIPT_KINDS = frozenset(RECEIPT_KIND_BY_GROUP.values())
PROVENANCE_DISPOSITIONS = frozenset(
    {
        "complete_contributing",
        "complete_noncontributing",
        "incomplete_noncontributing",
    }
)
PROVENANCE_GAP_FIELDS = (
    "command_receipt",
    "scheduler",
    "code_commit",
)
FINAL_EVIDENCE_REFERENCE_ROLES = (
    "terminal",
    "provenance",
    "evaluation",
    "reconciliation",
    "checkpoint",
)
SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION = "scheduler_accounting_recovery"
COMMAND_IDENTITY_ALGORITHM = "sha256"
COMMAND_IDENTITY_NORMALIZATION = "sacct SubmitLine with trailing newline removed"
COMMAND_IDENTITY_SOURCE_FIELD = "sacct.SubmitLine"
REQUIRED_ASSIGNMENT_COMMIT_ROLES = (
    "training",
    "evaluation",
    "reconciliation",
)
REQUIRED_CLOSED_CLAIM_FLAGS = (
    "training_authorized",
    "sufficient_training",
    "twin_qualification_pass",
    "formal_qualification",
    "test_manifest_open",
    "tool_gate_open",
    "mcp_gate_open",
    "agent_trace_gate_open",
    "agent_work_open",
    "memory_gate_open",
    "coevolution_gate_open",
)
ATTEMPT_STATUSES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
        "RECOVERED",
        "NON_CONVERGED",
        "UNSTABLE",
        "UNDEFINED",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_ATTEMPT_UID_RE = re.compile(
    r"^operational_twin__[a-z0-9_]+__seed_-?[0-9]+__attempt_[0-9]+$"
)


class ValidationBlocker(TypedDict):
    """A stable machine-readable validation failure."""

    code: str
    message: str


class SourceIdentity(TypedDict):
    """Compact identity for a textual or otherwise tracked source artifact."""

    path: str
    sha256: str
    size_bytes: int
    tracked: bool


class SchedulerResources(TypedDict):
    """Scheduler resource fields retained without interpreting allocation policy."""

    cpus: int
    gpus: int
    memory: str
    elapsed: str
    allocation: str


class SchedulerIdentity(TypedDict):
    """Terminal scheduler identity for an attempt or receipt."""

    job_id: str
    task_id: str | None
    job_name: str
    node: str
    partition: str
    state: str
    exit_code: str
    resources: SchedulerResources


def _source_identity_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        isinstance(value.get("path"), str)
        and bool(str(value.get("path")).strip())
        and isinstance(value.get("sha256"), str)
        and _SHA256_RE.fullmatch(str(value.get("sha256"))) is not None
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and int(value.get("size_bytes", -1)) >= 0
        and value.get("tracked") is True
    )


def _scheduler_identity_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if any(
        not isinstance(value.get(field), str) or not str(value.get(field)).strip()
        for field in ("job_id", "job_name", "node", "partition", "state", "exit_code")
    ):
        return False
    task_id = value.get("task_id")
    if task_id is not None and (
        not isinstance(task_id, str) or not task_id.strip()
    ):
        return False
    resources = value.get("resources")
    if not isinstance(resources, Mapping):
        return False
    if any(
        not isinstance(resources.get(field), int)
        or isinstance(resources.get(field), bool)
        or int(resources.get(field, -1)) < 0
        for field in ("cpus", "gpus")
    ):
        return False
    return all(
        isinstance(resources.get(field), str)
        and bool(str(resources.get(field)).strip())
        for field in ("memory", "elapsed", "allocation")
    )


def _command_identity_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        set(value) == {"algorithm", "normalization", "source_field", "value"}
        and value.get("algorithm") == COMMAND_IDENTITY_ALGORITHM
        and value.get("normalization") == COMMAND_IDENTITY_NORMALIZATION
        and value.get("source_field") == COMMAND_IDENTITY_SOURCE_FIELD
        and isinstance(value.get("value"), str)
        and _SHA256_RE.fullmatch(str(value.get("value"))) is not None
    )


def _command_receipt_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    receipt_uid = value.get("receipt_uid")
    if (
        not isinstance(receipt_uid, str)
        or not receipt_uid.strip()
        or not _source_identity_complete(value.get("source"))
    ):
        return False
    keys = set(value)
    basic_keys = {"receipt_uid", "source"}
    recovered_keys = basic_keys | {
        "source_classification",
        "scheduler_job_id",
        "command_identity",
    }
    if keys == basic_keys:
        return True
    return (
        keys == recovered_keys
        and value.get("source_classification")
        == SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION
        and isinstance(value.get("scheduler_job_id"), str)
        and bool(str(value.get("scheduler_job_id")).strip())
        and _command_identity_complete(value.get("command_identity"))
    )


def attempt_provenance_gaps(
    attempt: Mapping[str, object],
) -> tuple[str, ...]:
    gaps: list[str] = []
    if not _command_receipt_complete(attempt.get("command_receipt")):
        gaps.append("command_receipt")
    if not _scheduler_identity_complete(attempt.get("scheduler")):
        gaps.append("scheduler")
    commit = attempt.get("code_commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        gaps.append("code_commit")
    return tuple(gaps)


def _attempt_uid_references(value: object) -> set[str]:
    references: set[str] = set()
    if isinstance(value, str):
        if _ATTEMPT_UID_RE.fullmatch(value) is not None:
            references.add(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and _ATTEMPT_UID_RE.fullmatch(key) is not None:
                references.add(key)
            references.update(_attempt_uid_references(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            references.update(_attempt_uid_references(child))
    return references


def collect_attempt_uid_references(
    value: object,
) -> tuple[str, ...]:
    """Return exact assignment-shaped attempt UIDs in an evidence payload."""

    return tuple(sorted(_attempt_uid_references(value)))


def _derived_provenance_summary(
    assignments: Sequence[Mapping[str, object]],
) -> tuple[list[str], bool, bool]:
    attempts: list[Mapping[str, object]] = []
    terminal_complete_by_assignment: list[bool] = []
    historical_exceptions: set[str] = set()
    for assignment in assignments:
        terminal_attempt_uid = str(assignment.get("terminal_attempt_uid", ""))
        attempts_value = assignment.get("attempts")
        local_attempts = (
            [row for row in attempts_value if isinstance(row, Mapping)]
            if isinstance(attempts_value, Sequence)
            and not isinstance(attempts_value, (str, bytes))
            else []
        )
        attempts.extend(local_attempts)
        terminal_matches = [
            attempt
            for attempt in local_attempts
            if attempt.get("attempt_uid") == terminal_attempt_uid
        ]
        terminal_complete_by_assignment.append(
            len(terminal_matches) == 1
            and _source_identity_complete(terminal_matches[0].get("source"))
            and not attempt_provenance_gaps(terminal_matches[0])
            and terminal_matches[0].get("provenance_disposition")
            == "complete_contributing"
            and terminal_matches[0].get("contributes_to_candidate") is True
        )
        for attempt in local_attempts:
            attempt_uid = str(attempt.get("attempt_uid", ""))
            gaps = attempt_provenance_gaps(attempt)
            if attempt_uid and attempt_uid != terminal_attempt_uid and gaps:
                historical_exceptions.add(attempt_uid)
    all_attempt_complete = bool(attempts) and all(
        _source_identity_complete(attempt.get("source"))
        and not attempt_provenance_gaps(attempt)
        for attempt in attempts
    )
    terminal_contributing_complete = bool(terminal_complete_by_assignment) and all(
        terminal_complete_by_assignment
    )
    return (
        sorted(historical_exceptions),
        all_attempt_complete,
        terminal_contributing_complete,
    )


def _block(code: str, message: str) -> ValidationBlocker:
    return {"code": code, "message": message}


def _protocol_payload(protocol: Mapping[str, object]) -> dict[str, object]:
    nested = protocol.get("operational_twin_training_protocol")
    if isinstance(nested, Mapping):
        return deepcopy(dict(nested))
    return deepcopy(dict(protocol))


def _protocol_matrix(
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], list[str], list[int], list[dict[str, object]]]:
    payload = _protocol_payload(protocol)
    method_rows = payload.get("methods", [])
    if not isinstance(method_rows, Sequence) or isinstance(method_rows, (str, bytes)):
        raise ValueError("protocol methods must be a sequence")
    method_ids = [str(dict(row).get("method_id", "")) for row in method_rows if isinstance(row, Mapping)]
    seed_rows = payload.get("seeds", [])
    if not isinstance(seed_rows, Sequence) or isinstance(seed_rows, (str, bytes)):
        raise ValueError("protocol seeds must be a sequence")
    seeds = [int(seed) for seed in seed_rows]
    expected_assignments = build_assignments(payload, Path("."))
    return payload, method_ids, seeds, expected_assignments


def _attempt_identity_sha256(attempt: Mapping[str, object]) -> str:
    payload = deepcopy(dict(attempt))
    payload.pop("identity_sha256", None)
    return canonical_hash(payload)


def compute_operational_twin_training_evidence_ledger_sha256(
    ledger: Mapping[str, object],
) -> str:
    """Return the canonical self-excluding SHA-256 for a ledger mapping."""

    payload = deepcopy(dict(ledger))
    payload.pop("ledger_sha256", None)
    return canonical_hash(payload)


def _normalise_assignment(
    assignment: Mapping[str, object],
) -> dict[str, object]:
    row = deepcopy(dict(assignment))
    attempts = row.get("attempts", [])
    normalised_attempts: list[dict[str, object]] = []
    if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)):
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            attempt_row = deepcopy(dict(attempt))
            attempt_row["identity_sha256"] = _attempt_identity_sha256(attempt_row)
            normalised_attempts.append(attempt_row)
    row["attempts"] = sorted(
        normalised_attempts,
        key=lambda item: (int(item.get("ordinal", 0)), str(item.get("attempt_uid", ""))),
    )
    return row


def build_operational_twin_training_evidence_ledger(
    *,
    protocol: Mapping[str, object],
    protocol_identity: Mapping[str, object],
    assignments: Sequence[Mapping[str, object]],
    receipt_groups: Mapping[str, Sequence[Mapping[str, object]]],
    access_flags: Mapping[str, int],
    claim_flags: Mapping[str, bool],
    code_commits: Mapping[str, Sequence[str]],
    dependencies: Sequence[Mapping[str, object]],
    validation_blockers: Sequence[Mapping[str, str]] = (),
) -> dict[str, object]:
    """Build a deterministic compact ledger from caller-supplied records.

    The function performs no I/O and never mutates caller-owned mappings.  It
    derives the method/seed matrix from the supplied frozen protocol, computes
    immutable attempt identities, sorts unordered collections, and binds the
    result with a canonical self-excluding digest.  Call the validator before
    treating the returned mapping as admissible.
    """

    _, method_ids, seeds, _ = _protocol_matrix(protocol)
    method_order = {method_id: index for index, method_id in enumerate(method_ids)}
    seed_order = {seed: index for index, seed in enumerate(seeds)}
    assignment_rows = [_normalise_assignment(row) for row in assignments]
    assignment_rows.sort(
        key=lambda row: (
            method_order.get(str(row.get("method_id", "")), len(method_order)),
            seed_order.get(int(row.get("seed", -1)), len(seed_order)),
            str(row.get("assignment_uid", "")),
        )
    )

    normalised_receipts: dict[str, list[dict[str, object]]] = {}
    for group, receipts in receipt_groups.items():
        rows = [deepcopy(dict(row)) for row in receipts if isinstance(row, Mapping)]
        normalised_receipts[str(group)] = sorted(rows, key=lambda row: str(row.get("receipt_uid", "")))

    normalised_commits = {
        str(group): sorted({str(commit) for commit in commits})
        for group, commits in code_commits.items()
    }
    normalised_dependencies = sorted(
        [deepcopy(dict(row)) for row in dependencies if isinstance(row, Mapping)],
        key=lambda row: str(row.get("path", "")),
    )
    blocker_rows = sorted(
        [
            {"code": str(row.get("code", "")), "message": str(row.get("message", ""))}
            for row in validation_blockers
        ],
        key=lambda row: (row["code"], row["message"]),
    )
    (
        historical_provenance_exceptions,
        all_attempt_provenance_complete,
        terminal_contributing_provenance_complete,
    ) = _derived_provenance_summary(
        assignment_rows,
    )
    ledger: dict[str, object] = {
        "schema_version": TRAINING_EVIDENCE_LEDGER_SCHEMA_VERSION,
        "ledger_uid": TRAINING_EVIDENCE_LEDGER_UID,
        "status": "BLOCKED" if blocker_rows else "VALIDATION_READY",
        "protocol_identity": deepcopy(dict(protocol_identity)),
        "matrix": {
            "method_ids": method_ids,
            "seeds": seeds,
            "assignment_count": len(method_ids) * len(seeds),
        },
        "assignments": assignment_rows,
        "historical_provenance_exceptions": historical_provenance_exceptions,
        "historical_provenance_exception_count": len(
            historical_provenance_exceptions
        ),
        "all_attempt_provenance_complete": all_attempt_provenance_complete,
        "terminal_contributing_provenance_complete": (
            terminal_contributing_provenance_complete
        ),
        "receipt_groups": normalised_receipts,
        "access_flags": {str(key): int(value) for key, value in access_flags.items()},
        "claim_flags": {str(key): bool(value) for key, value in claim_flags.items()},
        "code_commits": normalised_commits,
        "dependencies": normalised_dependencies,
        "validation_blockers": blocker_rows,
    }
    ledger["ledger_sha256"] = compute_operational_twin_training_evidence_ledger_sha256(ledger)
    return ledger


def _validate_source_identity(
    value: object,
    *,
    label: str,
    blockers: list[ValidationBlocker],
) -> None:
    if not isinstance(value, Mapping):
        blockers.append(_block("source_identity_missing", f"{label} lacks a source identity."))
        return
    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(path, str) or not path.strip():
        blockers.append(_block("source_path_missing", f"{label} lacks a non-empty source path."))
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        blockers.append(_block("source_sha256_invalid", f"{label} lacks a valid SHA-256 identity."))
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        blockers.append(_block("source_size_invalid", f"{label} lacks a non-negative byte size."))
    if value.get("tracked") is not True:
        blockers.append(_block("untracked_dependency", f"{label} is not tracked by the committed evidence package."))


def _validate_scheduler_identity(
    value: object,
    *,
    label: str,
    blockers: list[ValidationBlocker],
) -> None:
    if not isinstance(value, Mapping):
        blockers.append(_block("scheduler_identity_missing", f"{label} lacks scheduler identity."))
        return
    for field in ("job_id", "job_name", "node", "partition", "state", "exit_code"):
        if not isinstance(value.get(field), str) or not str(value.get(field)).strip():
            blockers.append(_block("scheduler_field_missing", f"{label} lacks scheduler field {field}."))
    task_id = value.get("task_id")
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        blockers.append(_block("scheduler_task_invalid", f"{label} has an invalid scheduler task identity."))
    resources = value.get("resources")
    if not isinstance(resources, Mapping):
        blockers.append(_block("scheduler_resources_missing", f"{label} lacks scheduler resource fields."))
        return
    for field in ("cpus", "gpus"):
        amount = resources.get(field)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            blockers.append(_block("scheduler_resource_invalid", f"{label} has invalid resource field {field}."))
    for field in ("memory", "elapsed", "allocation"):
        if not isinstance(resources.get(field), str) or not str(resources.get(field)).strip():
            blockers.append(_block("scheduler_resource_invalid", f"{label} lacks resource field {field}."))


def _validate_command_receipt(
    value: object,
    *,
    attempt_uid: str,
    scheduler: object,
    training_scheduler_expectations: Mapping[str, Sequence[Mapping[str, object]]],
    seen_receipt_uids: set[str],
    blockers: list[ValidationBlocker],
) -> None:
    label = attempt_uid or "<missing>"
    if not isinstance(value, Mapping):
        blockers.append(_block("command_receipt_missing", f"{label} lacks command-receipt identity."))
        return

    receipt_uid = str(value.get("receipt_uid", ""))
    if not receipt_uid or receipt_uid in seen_receipt_uids:
        blockers.append(_block("duplicate_command_receipt_identity", f"Command receipt {receipt_uid or '<missing>'} is missing or duplicated."))
    else:
        seen_receipt_uids.add(receipt_uid)
    _validate_source_identity(value.get("source"), label=f"{label} command receipt", blockers=blockers)

    recovery_fields = {
        "source_classification",
        "scheduler_job_id",
        "command_identity",
    }
    if not recovery_fields.intersection(value):
        return
    if value.get("source_classification") != SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION:
        blockers.append(
            _block(
                "command_receipt_source_classification_invalid",
                f"{label} command receipt does not declare scheduler-accounting recovery.",
            )
        )

    scheduler_job_id = ""
    if isinstance(scheduler, Mapping):
        scheduler_job_id = str(scheduler.get("job_id", ""))
    recovered_job_id = value.get("scheduler_job_id")
    if (
        not isinstance(recovered_job_id, str)
        or not recovered_job_id
        or recovered_job_id != scheduler_job_id
    ):
        blockers.append(
            _block(
                "command_receipt_job_mismatch",
                f"{label} recovered command receipt does not bind its exact scheduler job ID.",
            )
        )

    command_identity = value.get("command_identity")
    if not isinstance(command_identity, Mapping):
        blockers.append(
            _block(
                "command_identity_missing",
                f"{label} recovered command receipt lacks command identity.",
            )
        )
    else:
        if command_identity.get("algorithm") != COMMAND_IDENTITY_ALGORITHM:
            blockers.append(
                _block(
                    "command_identity_algorithm_invalid",
                    f"{label} recovered command receipt must use SHA-256.",
                )
            )
        if command_identity.get("normalization") != COMMAND_IDENTITY_NORMALIZATION:
            blockers.append(
                _block(
                    "command_identity_normalization_invalid",
                    f"{label} recovered command receipt uses the wrong SubmitLine normalization.",
                )
            )
        if command_identity.get("source_field") != COMMAND_IDENTITY_SOURCE_FIELD:
            blockers.append(
                _block(
                    "command_identity_source_field_invalid",
                    f"{label} recovered command receipt uses the wrong source field.",
                )
            )
        digest = command_identity.get("value")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            blockers.append(
                _block(
                    "command_identity_sha256_invalid",
                    f"{label} recovered command receipt lacks a valid SubmitLine SHA-256.",
                )
            )

    if isinstance(recovered_job_id, str) and recovered_job_id:
        matched_expectations = list(
            training_scheduler_expectations.get(recovered_job_id, ())
        )
        if len(matched_expectations) != 1:
            blockers.append(
                _block(
                    "command_receipt_scheduler_source_missing",
                    f"{label} recovered command receipt lacks one unique tracked training scheduler source and command identity.",
                )
            )
        else:
            expected = matched_expectations[0]
            expected_source = expected.get("source")
            actual_source = value.get("source")
            if (
                not isinstance(expected_source, Mapping)
                or not isinstance(actual_source, Mapping)
                or dict(expected_source) != dict(actual_source)
            ):
                blockers.append(
                    _block(
                        "command_receipt_scheduler_source_mismatch",
                        f"{label} recovered command receipt is not bound to its training scheduler source.",
                    )
                )
            expected_command_identity = expected.get("command_identity")
            if (
                not isinstance(expected_command_identity, Mapping)
                or not isinstance(command_identity, Mapping)
                or dict(command_identity) != dict(expected_command_identity)
            ):
                blockers.append(
                    _block(
                        "command_receipt_command_identity_mismatch",
                        f"{label} recovered command identity does not match its training scheduler SubmitLine identity.",
                    )
                )


def _validate_access_flags(
    value: object,
    *,
    label: str,
    blockers: list[ValidationBlocker],
) -> None:
    if not isinstance(value, Mapping):
        blockers.append(_block("access_flags_missing", f"{label} lacks access flags."))
        return
    missing = [key for key in REQUIRED_ZERO_ACCESS_COUNTERS if key not in value]
    if missing:
        blockers.append(_block("access_flag_missing", f"{label} lacks required access flags: {', '.join(missing)}."))
    for key, raw_value in value.items():
        if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value != 0:
            blockers.append(_block("nonzero_access_flag", f"{label} access flag {key} must be the integer zero."))


def _validate_claim_flags(
    value: object,
    *,
    label: str,
    blockers: list[ValidationBlocker],
) -> None:
    if not isinstance(value, Mapping):
        blockers.append(_block("claim_flags_missing", f"{label} lacks claim flags."))
        return
    missing = [key for key in REQUIRED_CLOSED_CLAIM_FLAGS if key not in value]
    if missing:
        blockers.append(_block("claim_flag_missing", f"{label} lacks required claim flags: {', '.join(missing)}."))
    for key, raw_value in value.items():
        if raw_value is not False:
            blockers.append(_block("open_claim_flag", f"{label} claim flag {key} must remain false."))


def _validate_commit(value: object, *, label: str, blockers: list[ValidationBlocker]) -> None:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        blockers.append(_block("code_commit_invalid", f"{label} lacks a full hexadecimal code commit identity."))


def _validate_checkpoint(
    checkpoint: object,
    *,
    method_id: str,
    assignment_uid: str,
    blockers: list[ValidationBlocker],
) -> None:
    if not isinstance(checkpoint, Mapping):
        blockers.append(_block("checkpoint_contract_missing", f"{assignment_uid} lacks a checkpoint contract."))
        return
    contract = METHOD_EXECUTION_CONTRACTS.get(method_id, {})
    checkpoint_required = contract.get("checkpoint_creation_allowed") is True
    if checkpoint_required:
        if checkpoint.get("contract") != "checkpoint_identity":
            blockers.append(_block("trainable_checkpoint_missing", f"{assignment_uid} lacks a trainable checkpoint identity."))
            return
        if not isinstance(checkpoint.get("path"), str) or not str(checkpoint.get("path")).strip():
            blockers.append(_block("trainable_checkpoint_missing", f"{assignment_uid} lacks a checkpoint path."))
        digest = checkpoint.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            blockers.append(_block("trainable_checkpoint_missing", f"{assignment_uid} lacks a checkpoint SHA-256."))
        size = checkpoint.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            blockers.append(_block("trainable_checkpoint_missing", f"{assignment_uid} lacks a positive checkpoint byte size."))
        if checkpoint.get("repository_storage") != "external_content_addressed":
            blockers.append(_block("checkpoint_storage_contract_invalid", f"{assignment_uid} checkpoint storage contract is invalid."))
    else:
        no_checkpoint = (
            checkpoint.get("contract") == "no_checkpoint"
            and checkpoint.get("path") is None
            and checkpoint.get("sha256") is None
            and checkpoint.get("size_bytes") == 0
            and checkpoint.get("repository_storage") == "not_applicable"
            and isinstance(checkpoint.get("reason"), str)
            and bool(str(checkpoint.get("reason")).strip())
        )
        if not no_checkpoint:
            blockers.append(_block("no_checkpoint_contract_invalid", f"{assignment_uid} must carry an explicit no-checkpoint contract."))


def _count_retained_attempts(attempts: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "source_attempt_count": len(attempts),
        "source_failed_count": sum(attempt.get("status") == "FAILED" for attempt in attempts),
        "source_cancelled_count": sum(attempt.get("status") == "CANCELLED" for attempt in attempts),
        "source_recovered_count": sum(
            attempt.get("status") == "RECOVERED" or attempt.get("recovered_from_attempt_uid") is not None
            for attempt in attempts
        ),
    }


def validate_operational_twin_training_evidence_ledger(
    ledger: Mapping[str, object],
    protocol: Mapping[str, object],
) -> list[ValidationBlocker]:
    """Validate a compact ledger against the caller's frozen protocol.

    Validation is fail closed: the exact Cartesian assignment matrix, complete
    attempt-retention declarations, checkpoint policy, tracked source bindings,
    closed access/claim flags, and all final receipt groups are mandatory.
    """

    blockers: list[ValidationBlocker] = []
    try:
        protocol_payload, protocol_method_ids, protocol_seeds, expected_rows = _protocol_matrix(protocol)
    except (TypeError, ValueError) as error:
        return [_block("protocol_matrix_invalid", f"The supplied protocol matrix is invalid: {error}.")]

    if ledger.get("schema_version") != TRAINING_EVIDENCE_LEDGER_SCHEMA_VERSION:
        blockers.append(_block("ledger_schema_version_invalid", "The training evidence ledger schema version is invalid."))
    if ledger.get("ledger_uid") != TRAINING_EVIDENCE_LEDGER_UID:
        blockers.append(_block("ledger_uid_invalid", "The training evidence ledger UID is invalid."))
    expected_hash = compute_operational_twin_training_evidence_ledger_sha256(ledger)
    if ledger.get("ledger_sha256") != expected_hash:
        blockers.append(_block("ledger_sha256_mismatch", "The ledger digest does not match its canonical content."))

    if protocol_method_ids != list(REQUIRED_METHOD_IDS):
        blockers.append(_block("protocol_method_contract_invalid", "The supplied protocol does not contain the five frozen method identities in order."))
    if len(protocol_seeds) != 3 or len(set(protocol_seeds)) != 3:
        blockers.append(_block("protocol_seed_contract_invalid", "The supplied protocol does not contain exactly three unique seeds."))
    protocol_identity = ledger.get("protocol_identity")
    if not isinstance(protocol_identity, Mapping):
        blockers.append(_block("protocol_identity_missing", "The ledger lacks protocol identity."))
    else:
        if protocol_identity.get("protocol_schema_version") != protocol_payload.get("schema_version"):
            blockers.append(_block("protocol_schema_identity_mismatch", "The ledger protocol schema identity does not match the supplied protocol."))
        if protocol_identity.get("protocol_semantic_sha256") != canonical_hash(protocol_payload):
            blockers.append(_block("protocol_semantic_identity_mismatch", "The ledger does not bind the supplied protocol semantics."))
        sources = protocol_identity.get("sources")
        if not isinstance(sources, Mapping):
            blockers.append(_block("protocol_sources_missing", "The ledger lacks protocol source identities."))
        else:
            for role in ("protocol", "assignment_manifest", "preflight_authority"):
                _validate_source_identity(sources.get(role), label=f"protocol source {role}", blockers=blockers)

    matrix = ledger.get("matrix")
    if not isinstance(matrix, Mapping):
        blockers.append(_block("matrix_missing", "The ledger lacks the frozen method/seed matrix."))
        matrix = {}
    if list(matrix.get("method_ids", [])) != protocol_method_ids or protocol_method_ids != list(REQUIRED_METHOD_IDS):
        blockers.append(_block("method_matrix_mismatch", "The ledger method matrix does not match the frozen protocol."))
    if list(matrix.get("seeds", [])) != protocol_seeds:
        blockers.append(_block("seed_matrix_mismatch", "The ledger seed matrix does not match the frozen protocol."))
    if matrix.get("assignment_count") != len(expected_rows) or len(expected_rows) != 15:
        blockers.append(_block("assignment_count_mismatch", "The ledger must bind exactly 15 frozen assignments."))

    expected_by_pair = {
        (str(row["method_id"]), int(row["seed"])): str(row["assignment_uid"])
        for row in expected_rows
    }
    assignments_value = ledger.get("assignments")
    if not isinstance(assignments_value, Sequence) or isinstance(assignments_value, (str, bytes)):
        blockers.append(_block("assignments_missing", "The ledger assignments collection is absent."))
        assignments: list[Mapping[str, object]] = []
    else:
        assignments = [row for row in assignments_value if isinstance(row, Mapping)]
        if len(assignments) != len(assignments_value):
            blockers.append(_block("assignment_record_invalid", "Every assignment entry must be a mapping."))

    actual_pairs: list[tuple[str, int]] = []
    actual_uids: list[str] = []
    for row in assignments:
        try:
            actual_pairs.append((str(row.get("method_id", "")), int(row.get("seed", -1))))
        except (TypeError, ValueError):
            actual_pairs.append((str(row.get("method_id", "")), -1))
        actual_uids.append(str(row.get("assignment_uid", "")))
    if len(assignments) != 15 or set(actual_pairs) != set(expected_by_pair) or len(set(actual_pairs)) != len(actual_pairs):
        blockers.append(_block("assignment_matrix_mismatch", "The ledger does not contain exactly one record for every frozen method/seed assignment."))
    if len(set(actual_uids)) != len(actual_uids):
        blockers.append(_block("duplicate_assignment_uid", "Assignment UIDs must be unique."))

    attempt_assignment_owners: dict[str, set[str]] = {}
    for assignment in assignments:
        assignment_uid = str(assignment.get("assignment_uid", ""))
        attempts_value = assignment.get("attempts")
        if not isinstance(attempts_value, Sequence) or isinstance(
            attempts_value, (str, bytes)
        ):
            continue
        for attempt in attempts_value:
            if not isinstance(attempt, Mapping):
                continue
            attempt_uid = str(attempt.get("attempt_uid", ""))
            if attempt_uid:
                attempt_assignment_owners.setdefault(attempt_uid, set()).add(
                    assignment_uid
                )

    receipt_groups = ledger.get("receipt_groups")
    training_scheduler_expectations: dict[
        str, list[dict[str, object]]
    ] = {}
    if isinstance(receipt_groups, Mapping):
        training_receipts = receipt_groups.get("training")
        if isinstance(training_receipts, Sequence) and not isinstance(
            training_receipts, (str, bytes)
        ):
            for receipt in training_receipts:
                if not isinstance(receipt, Mapping):
                    continue
                scheduler = receipt.get("scheduler")
                if not isinstance(scheduler, Mapping):
                    continue
                job_id = scheduler.get("job_id")
                source = receipt.get("source")
                command_identity = receipt.get("command_identity")
                if (
                    isinstance(job_id, str)
                    and job_id
                    and isinstance(source, Mapping)
                    and source.get("tracked") is True
                    and isinstance(command_identity, Mapping)
                ):
                    training_scheduler_expectations.setdefault(job_id, []).append(
                        {
                            "source": source,
                            "command_identity": command_identity,
                        }
                    )

    seen_attempt_uids: set[str] = set()
    seen_attempt_hashes: set[str] = set()
    seen_command_receipt_uids: set[str] = set()
    (
        expected_historical_exceptions,
        expected_all_attempt_provenance_complete,
        expected_terminal_contributing_provenance_complete,
    ) = _derived_provenance_summary(assignments)
    for assignment in assignments:
        assignment_uid = str(assignment.get("assignment_uid", ""))
        method_id = str(assignment.get("method_id", ""))
        try:
            seed = int(assignment.get("seed", -1))
        except (TypeError, ValueError):
            seed = -1
        if expected_by_pair.get((method_id, seed)) != assignment_uid:
            blockers.append(_block("assignment_uid_mismatch", f"{assignment_uid or '<missing>'} does not match its frozen method/seed identity."))

        terminal_status = assignment.get("terminal_status")
        if not isinstance(terminal_status, str) or not terminal_status.strip():
            blockers.append(_block("terminal_status_missing", f"{assignment_uid} lacks a terminal status."))
        terminal_attempt_uid = assignment.get("terminal_attempt_uid")
        if not isinstance(terminal_attempt_uid, str) or not terminal_attempt_uid.strip():
            blockers.append(_block("terminal_attempt_missing", f"{assignment_uid} lacks a terminal attempt UID."))

        evidence = assignment.get("evidence")
        if not isinstance(evidence, Mapping):
            blockers.append(_block("assignment_evidence_missing", f"{assignment_uid} lacks evidence identities."))
        else:
            for role in ("terminal", "provenance", "evaluation", "reconciliation"):
                _validate_source_identity(evidence.get(role), label=f"{assignment_uid} {role} evidence", blockers=blockers)

        _validate_checkpoint(
            assignment.get("checkpoint"),
            method_id=method_id,
            assignment_uid=assignment_uid,
            blockers=blockers,
        )
        _validate_access_flags(assignment.get("access_flags"), label=assignment_uid, blockers=blockers)
        _validate_claim_flags(assignment.get("claim_flags"), label=assignment_uid, blockers=blockers)

        assignment_commits = assignment.get("code_commits")
        if not isinstance(assignment_commits, Mapping):
            blockers.append(_block("assignment_code_commits_missing", f"{assignment_uid} lacks code commit identities."))
        else:
            for role in REQUIRED_ASSIGNMENT_COMMIT_ROLES:
                _validate_commit(assignment_commits.get(role), label=f"{assignment_uid} {role}", blockers=blockers)

        attempts_value = assignment.get("attempts")
        if not isinstance(attempts_value, Sequence) or isinstance(attempts_value, (str, bytes)):
            blockers.append(_block("attempts_missing", f"{assignment_uid} lacks retained attempts."))
            attempts: list[Mapping[str, object]] = []
        else:
            attempts = [row for row in attempts_value if isinstance(row, Mapping)]
            if len(attempts) != len(attempts_value):
                blockers.append(_block("attempt_record_invalid", f"{assignment_uid} has a non-mapping attempt record."))
        if not attempts:
            blockers.append(_block("attempts_missing", f"{assignment_uid} has no retained attempts."))

        local_attempt_uids: set[str] = set()
        local_ordinals: dict[str, int] = {}
        local_ordinal_values: set[int] = set()
        local_attempts_by_uid: dict[str, Mapping[str, object]] = {}
        contributing_attempt_uids: list[str] = []
        for attempt in attempts:
            attempt_uid = str(attempt.get("attempt_uid", ""))
            identity_sha256 = str(attempt.get("identity_sha256", ""))
            try:
                ordinal = int(attempt.get("ordinal", 0))
            except (TypeError, ValueError):
                ordinal = 0
            if (
                attempt.get("assignment_uid") != assignment_uid
                or attempt.get("method_id") != method_id
                or attempt.get("seed") != seed
            ):
                blockers.append(_block("attempt_assignment_identity_mismatch", f"{attempt_uid or '<missing>'} does not bind its assignment identity."))
            if not attempt_uid or attempt_uid in seen_attempt_uids:
                blockers.append(_block("duplicate_attempt_identity", f"Attempt UID {attempt_uid or '<missing>'} is missing or duplicated."))
            else:
                seen_attempt_uids.add(attempt_uid)
            if not identity_sha256 or identity_sha256 in seen_attempt_hashes:
                blockers.append(_block("duplicate_attempt_identity", f"Attempt hash {identity_sha256 or '<missing>'} is missing or duplicated."))
            else:
                seen_attempt_hashes.add(identity_sha256)
            if identity_sha256 != _attempt_identity_sha256(attempt):
                blockers.append(_block("attempt_identity_mismatch", f"{attempt_uid or '<missing>'} immutable identity does not match its content."))
            if ordinal <= 0 or ordinal in local_ordinal_values:
                blockers.append(_block("attempt_ordinal_invalid", f"{attempt_uid or '<missing>'} has a missing or duplicate ordinal."))
            else:
                local_ordinal_values.add(ordinal)
            local_attempt_uids.add(attempt_uid)
            local_ordinals[attempt_uid] = ordinal
            local_attempts_by_uid[attempt_uid] = attempt
            if attempt.get("status") not in ATTEMPT_STATUSES:
                blockers.append(_block("attempt_status_invalid", f"{attempt_uid or '<missing>'} has an unsupported terminal status."))
            if not isinstance(attempt.get("terminal_reason"), str) or not str(attempt.get("terminal_reason")).strip():
                blockers.append(_block("attempt_terminal_reason_missing", f"{attempt_uid or '<missing>'} lacks a terminal reason."))
            _validate_source_identity(attempt.get("source"), label=f"{attempt_uid} source", blockers=blockers)
            actual_gaps = attempt_provenance_gaps(attempt)
            attempt_complete = not actual_gaps

            gaps_value = attempt.get("provenance_gaps")
            if not isinstance(gaps_value, Sequence) or isinstance(
                gaps_value, (str, bytes)
            ):
                blockers.append(
                    _block(
                        "attempt_provenance_gaps_invalid",
                        f"{attempt_uid or '<missing>'} provenance_gaps must be an array.",
                    )
                )
                declared_gaps: list[str] = []
            else:
                declared_gaps = [str(value) for value in gaps_value]
                if (
                    len(set(declared_gaps)) != len(declared_gaps)
                    or any(value not in PROVENANCE_GAP_FIELDS for value in declared_gaps)
                ):
                    blockers.append(
                        _block(
                            "attempt_provenance_gaps_invalid",
                            f"{attempt_uid or '<missing>'} has duplicate or unsupported provenance gaps.",
                        )
                    )
            expected_gaps = [
                field for field in PROVENANCE_GAP_FIELDS if field in actual_gaps
            ]
            if declared_gaps != expected_gaps:
                blockers.append(
                    _block(
                        "attempt_provenance_gap_mismatch",
                        f"{attempt_uid or '<missing>'} provenance gaps do not match its retained values.",
                    )
                )

            is_terminal = attempt_uid == terminal_attempt_uid
            expected_disposition = (
                "complete_contributing"
                if is_terminal
                else (
                    "complete_noncontributing"
                    if attempt_complete
                    else "incomplete_noncontributing"
                )
            )
            disposition = attempt.get("provenance_disposition")
            if disposition not in PROVENANCE_DISPOSITIONS:
                blockers.append(
                    _block(
                        "attempt_provenance_disposition_invalid",
                        f"{attempt_uid or '<missing>'} lacks a supported provenance disposition.",
                    )
                )
            elif disposition != expected_disposition:
                blockers.append(
                    _block(
                        "attempt_provenance_disposition_invalid",
                        f"{attempt_uid or '<missing>'} must be {expected_disposition} for its terminal and provenance state.",
                    )
                )
            contributes = attempt.get("contributes_to_candidate")
            if contributes is True:
                contributing_attempt_uids.append(attempt_uid)
            if contributes is not is_terminal:
                blockers.append(
                    _block(
                        "attempt_contribution_mismatch",
                        f"{attempt_uid or '<missing>'} contribution flag must identify only the terminal attempt.",
                    )
                )

            if is_terminal:
                terminal_complete = (
                    attempt_complete
                    and disposition == "complete_contributing"
                    and contributes is True
                    and attempt.get("superseded_by_attempt_uid") is None
                )
                if not terminal_complete:
                    blockers.append(
                        _block(
                            "terminal_contributing_provenance_incomplete",
                            f"{attempt_uid or '<missing>'} must retain complete source, command, scheduler, and commit provenance.",
                        )
                    )
            command_receipt = attempt.get("command_receipt")
            if isinstance(command_receipt, Mapping):
                _validate_command_receipt(
                    command_receipt,
                    attempt_uid=attempt_uid,
                    scheduler=attempt.get("scheduler"),
                    training_scheduler_expectations=training_scheduler_expectations,
                    seen_receipt_uids=seen_command_receipt_uids,
                    blockers=blockers,
                )
            if "scheduler" not in actual_gaps:
                _validate_scheduler_identity(
                    attempt.get("scheduler"), label=attempt_uid, blockers=blockers
                )
            if "code_commit" not in actual_gaps:
                _validate_commit(
                    attempt.get("code_commit"),
                    label=f"{attempt_uid} code",
                    blockers=blockers,
                )

        if terminal_attempt_uid not in local_attempt_uids:
            blockers.append(_block("terminal_attempt_missing", f"{assignment_uid} terminal attempt is not present in retained attempts."))
        else:
            terminal_attempt = local_attempts_by_uid[terminal_attempt_uid]
            if terminal_attempt.get("status") != terminal_status:
                blockers.append(
                    _block(
                        "terminal_status_mismatch",
                        f"{assignment_uid} terminal status does not match its terminal attempt.",
                    )
                )
            if local_ordinals.get(str(terminal_attempt_uid), 0) != len(attempts):
                blockers.append(
                    _block(
                        "terminal_attempt_not_last",
                        f"{assignment_uid} terminal attempt must have the final retained ordinal.",
                    )
                )
        if contributing_attempt_uids != [terminal_attempt_uid]:
            blockers.append(
                _block(
                    "terminal_contributor_count_invalid",
                    f"{assignment_uid} must have exactly one contributing terminal attempt.",
                )
            )
        if local_ordinal_values != set(range(1, len(attempts) + 1)):
            blockers.append(
                _block(
                    "attempt_ordinal_sequence_incomplete",
                    f"{assignment_uid} retained attempt ordinals must be contiguous from one.",
                )
            )

        final_references = assignment.get("final_evidence_attempt_references")
        final_reference_contexts: dict[str, list[str]] = {}
        if not isinstance(final_references, Mapping):
            blockers.append(
                _block(
                    "final_attempt_references_invalid",
                    f"{assignment_uid} lacks explicit final-evidence attempt references.",
                )
            )
        else:
            missing_roles = [
                role
                for role in FINAL_EVIDENCE_REFERENCE_ROLES
                if role not in final_references
            ]
            unexpected_roles = sorted(
                set(final_references) - set(FINAL_EVIDENCE_REFERENCE_ROLES)
            )
            if missing_roles or unexpected_roles:
                blockers.append(
                    _block(
                        "final_attempt_references_invalid",
                        f"{assignment_uid} final-evidence reference roles are incomplete or unexpected.",
                    )
                )
            for role in FINAL_EVIDENCE_REFERENCE_ROLES:
                role_value = final_references.get(role)
                if not isinstance(role_value, Sequence) or isinstance(
                    role_value, (str, bytes)
                ):
                    blockers.append(
                        _block(
                            "final_attempt_references_invalid",
                            f"{assignment_uid} final-evidence role {role} must be an array.",
                        )
                    )
                    continue
                role_uids = [str(value) for value in role_value]
                if len(role_uids) != len(set(role_uids)):
                    blockers.append(
                        _block(
                            "final_attempt_references_invalid",
                            f"{assignment_uid} final-evidence role {role} has duplicate attempt UIDs.",
                        )
                    )
                if role == "terminal" and role_uids != [terminal_attempt_uid]:
                    blockers.append(
                        _block(
                            "final_terminal_attempt_reference_mismatch",
                            f"{assignment_uid} terminal evidence must reference exactly its terminal attempt.",
                        )
                    )
                for referenced_uid in role_uids:
                    final_reference_contexts.setdefault(referenced_uid, []).append(role)

        hidden_reference_scope = {
            key: value
            for key, value in assignment.items()
            if key
            not in {
                "attempts",
                "attempt_retention",
                "terminal_attempt_uid",
                "final_evidence_attempt_references",
            }
        }
        for referenced_uid in collect_attempt_uid_references(hidden_reference_scope):
            final_reference_contexts.setdefault(referenced_uid, []).append(
                "assignment"
            )
        for referenced_uid, contexts in sorted(final_reference_contexts.items()):
            context_label = ", ".join(sorted(set(contexts)))
            owners = attempt_assignment_owners.get(referenced_uid, set())
            if not owners:
                blockers.append(
                    _block(
                        "final_attempt_reference_unretained",
                        f"{assignment_uid} {context_label} references omitted attempt {referenced_uid or '<missing>'}.",
                    )
                )
            elif owners != {assignment_uid}:
                blockers.append(
                    _block(
                        "final_attempt_reference_cross_assignment",
                        f"{assignment_uid} {context_label} references non-local attempt {referenced_uid}.",
                    )
                )
            elif referenced_uid != terminal_attempt_uid:
                referenced_attempt = local_attempts_by_uid.get(referenced_uid)
                if referenced_attempt is not None and attempt_provenance_gaps(
                    referenced_attempt
                ):
                    blockers.append(
                        _block(
                            "incomplete_historical_attempt_final_reference",
                            f"{assignment_uid} {context_label} references incomplete historical attempt {referenced_uid}.",
                        )
                    )
                else:
                    blockers.append(
                        _block(
                            "noncontributing_attempt_final_reference",
                            f"{assignment_uid} {context_label} references noncontributing attempt {referenced_uid}.",
                        )
                    )

        for attempt in attempts:
            attempt_uid = str(attempt.get("attempt_uid", ""))
            recovered_from = attempt.get("recovered_from_attempt_uid")
            if attempt.get("status") == "RECOVERED" and recovered_from is None:
                blockers.append(_block("recovery_link_missing", f"{attempt_uid} is recovered but lacks its source attempt UID."))
            if recovered_from is not None:
                source_uid = str(recovered_from)
                if source_uid not in local_attempt_uids:
                    blockers.append(_block("recovery_source_missing", f"{attempt_uid} recovery source {source_uid} was omitted."))
                elif local_ordinals.get(source_uid, 0) >= local_ordinals.get(attempt_uid, 0):
                    blockers.append(_block("recovery_order_invalid", f"{attempt_uid} recovery source must precede the recovery attempt."))

            if attempt_uid == terminal_attempt_uid:
                if attempt.get("superseded_by_attempt_uid") is not None:
                    blockers.append(
                        _block(
                            "terminal_supersession_invalid",
                            f"{attempt_uid} is terminal and must not be superseded.",
                        )
                    )
                continue
            visited = {attempt_uid}
            current_uid = attempt_uid
            while current_uid != terminal_attempt_uid:
                current_attempt = local_attempts_by_uid.get(current_uid)
                if current_attempt is None:
                    blockers.append(
                        _block(
                            "supersession_terminal_unreachable",
                            f"{attempt_uid} supersession chain does not reach its terminal attempt.",
                        )
                    )
                    break
                target_uid = str(
                    current_attempt.get("superseded_by_attempt_uid") or ""
                )
                if not target_uid:
                    blockers.append(
                        _block(
                            "supersession_target_missing",
                            f"{current_uid} lacks a superseding attempt UID on the chain from {attempt_uid}.",
                        )
                    )
                    break
                owners = attempt_assignment_owners.get(target_uid, set())
                if owners != {assignment_uid}:
                    blockers.append(
                        _block(
                            "supersession_cross_assignment",
                            f"{current_uid} supersession target {target_uid} is not in the same assignment.",
                        )
                    )
                    break
                if target_uid in visited:
                    blockers.append(
                        _block(
                            "supersession_cycle",
                            f"{attempt_uid} supersession chain contains a cycle.",
                        )
                    )
                    break
                if local_ordinals.get(target_uid, 0) <= local_ordinals.get(
                    current_uid, 0
                ):
                    blockers.append(
                        _block(
                            "supersession_order_invalid",
                            f"{current_uid} supersession target must have a later ordinal.",
                        )
                    )
                    break
                visited.add(target_uid)
                current_uid = target_uid

        retention = assignment.get("attempt_retention")
        actual_counts = _count_retained_attempts(attempts)
        if not isinstance(retention, Mapping):
            blockers.append(_block("attempt_retention_missing", f"{assignment_uid} lacks an attempt-retention declaration."))
        else:
            if retention.get("retention_complete") is not True:
                blockers.append(_block("attempt_retention_incomplete", f"{assignment_uid} does not declare complete attempt retention."))
            for field, actual in actual_counts.items():
                if retention.get(field) != actual:
                    blockers.append(
                        _block(
                            "attempt_retention_count_mismatch",
                            f"{assignment_uid} {field} declares {retention.get(field)!r}, but {actual} attempts are retained.",
                        )
                    )

    historical_exceptions = ledger.get("historical_provenance_exceptions")
    if not isinstance(historical_exceptions, Sequence) or isinstance(
        historical_exceptions, (str, bytes)
    ):
        blockers.append(
            _block(
                "historical_provenance_exceptions_invalid",
                "The historical provenance exception list must be an array.",
            )
        )
    elif list(historical_exceptions) != expected_historical_exceptions:
        blockers.append(
            _block(
                "historical_provenance_exceptions_mismatch",
                "The historical provenance exception list must exactly equal the incomplete noncontributing attempts.",
            )
        )
    if ledger.get("historical_provenance_exception_count") != len(
        expected_historical_exceptions
    ):
        blockers.append(
            _block(
                "historical_provenance_exception_count_mismatch",
                "The historical provenance exception count does not match the incomplete attempt list.",
            )
        )
    if ledger.get("all_attempt_provenance_complete") is not (
        expected_all_attempt_provenance_complete
    ):
        blockers.append(
            _block(
                "all_attempt_provenance_complete_mismatch",
                "The all-attempt provenance completeness summary is incorrect.",
            )
        )
    if ledger.get("terminal_contributing_provenance_complete") is not (
        expected_terminal_contributing_provenance_complete
    ):
        blockers.append(
            _block(
                "terminal_contributing_provenance_complete_mismatch",
                "The terminal contributing provenance completeness summary is incorrect.",
            )
        )

    seen_receipt_uids: set[str] = set()
    if not isinstance(receipt_groups, Mapping):
        blockers.append(_block("receipt_groups_missing", "The final receipt groups are absent."))
    else:
        missing_groups = [group for group in REQUIRED_RECEIPT_GROUPS if group not in receipt_groups]
        unexpected_groups = sorted(set(receipt_groups) - set(REQUIRED_RECEIPT_GROUPS))
        if missing_groups:
            blockers.append(_block("receipt_group_missing", f"Missing final receipt groups: {', '.join(missing_groups)}."))
        if unexpected_groups:
            blockers.append(_block("receipt_group_unexpected", f"Unexpected receipt groups: {', '.join(unexpected_groups)}."))
        for group in REQUIRED_RECEIPT_GROUPS:
            receipts_value = receipt_groups.get(group)
            if not isinstance(receipts_value, Sequence) or isinstance(receipts_value, (str, bytes)) or not receipts_value:
                blockers.append(_block("receipt_group_missing", f"Final receipt group {group} must be non-empty."))
                continue
            for index, receipt in enumerate(receipts_value):
                label = f"{group} receipt {index}"
                if not isinstance(receipt, Mapping):
                    blockers.append(_block("receipt_identity_invalid", f"{label} is not a mapping."))
                    continue
                receipt_uid = str(receipt.get("receipt_uid", ""))
                if not receipt_uid or receipt_uid in seen_receipt_uids:
                    blockers.append(_block("duplicate_receipt_identity", f"Receipt UID {receipt_uid or '<missing>'} is missing or duplicated."))
                else:
                    seen_receipt_uids.add(receipt_uid)
                _validate_source_identity(receipt.get("source"), label=label, blockers=blockers)
                receipt_kind = receipt.get("receipt_kind")
                if receipt_kind not in RECEIPT_KINDS:
                    blockers.append(
                        _block(
                            "receipt_kind_invalid",
                            f"{label} lacks a supported scheduler or process receipt kind.",
                        )
                    )
                elif receipt_kind != RECEIPT_KIND_BY_GROUP[group]:
                    blockers.append(
                        _block(
                            "receipt_kind_mismatch",
                            f"{label} must use receipt kind {RECEIPT_KIND_BY_GROUP[group]}.",
                        )
                    )
                if group in SCHEDULER_RECEIPT_GROUPS:
                    _validate_scheduler_identity(
                        receipt.get("scheduler"),
                        label=label,
                        blockers=blockers,
                    )
                    if receipt.get("code_commit") is not None:
                        _validate_commit(
                            receipt.get("code_commit"),
                            label=f"{label} code",
                            blockers=blockers,
                        )
                else:
                    _validate_commit(
                        receipt.get("code_commit"),
                        label=f"{label} producer code",
                        blockers=blockers,
                    )

    _validate_access_flags(ledger.get("access_flags"), label="ledger", blockers=blockers)
    _validate_claim_flags(ledger.get("claim_flags"), label="ledger", blockers=blockers)

    code_commits = ledger.get("code_commits")
    if not isinstance(code_commits, Mapping):
        blockers.append(_block("code_commit_groups_missing", "The ledger lacks code commit groups."))
    else:
        missing_commit_groups = [group for group in REQUIRED_RECEIPT_GROUPS if group not in code_commits]
        if missing_commit_groups:
            blockers.append(_block("code_commit_group_missing", f"Missing code commit groups: {', '.join(missing_commit_groups)}."))
        for group in REQUIRED_RECEIPT_GROUPS:
            commits = code_commits.get(group)
            if not isinstance(commits, Sequence) or isinstance(commits, (str, bytes)) or not commits:
                blockers.append(_block("code_commit_group_missing", f"Code commit group {group} must be non-empty."))
                continue
            for commit in commits:
                _validate_commit(commit, label=f"{group} code", blockers=blockers)

    dependencies_value = ledger.get("dependencies")
    if not isinstance(dependencies_value, Sequence) or isinstance(dependencies_value, (str, bytes)) or not dependencies_value:
        blockers.append(_block("dependencies_missing", "The ledger lacks source dependencies."))
    else:
        seen_dependency_paths: set[str] = set()
        for index, dependency in enumerate(dependencies_value):
            _validate_source_identity(dependency, label=f"dependency {index}", blockers=blockers)
            if isinstance(dependency, Mapping):
                path = str(dependency.get("path", ""))
                if path in seen_dependency_paths:
                    blockers.append(_block("duplicate_dependency_path", f"Dependency path {path or '<missing>'} is duplicated."))
                seen_dependency_paths.add(path)

    reported_blockers = ledger.get("validation_blockers")
    if not isinstance(reported_blockers, Sequence) or isinstance(reported_blockers, (str, bytes)):
        blockers.append(_block("validation_blockers_invalid", "The ledger validation blockers field must be an array."))
        reported_blockers = []
    elif reported_blockers:
        blockers.append(_block("reported_validation_blockers", "The ledger carries unresolved validation blockers."))
    expected_status = "BLOCKED" if reported_blockers else "VALIDATION_READY"
    if ledger.get("status") != expected_status:
        blockers.append(_block("ledger_status_mismatch", "Ledger status does not agree with its reported validation blockers."))

    return blockers
