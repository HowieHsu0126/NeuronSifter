"""Build strict Operational Twin process receipts.

The default CLI mode preserves the deterministic aggregation producer-identity
receipt.  ``--focused-validation-contract`` executes one of the two registered
validation commands under a Slurm allocation and binds the exact committed
source state plus the combined stdout/stderr log.  Arbitrary caller-selected
commands are intentionally unsupported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / (
    "Libs/configs/schema/operational_twin_process_receipt.schema.json"
)
SCHEMA_VERSION = "neuronsifter-operational-twin-process-receipt-v1"
RECEIPT_UID = "operational_twin_aggregation_producer_identity"
RECEIPT_KIND = "process"
RECEIPT_GROUP = "aggregation"
PROCESS_RECEIPT_KIND = "producer_identity"
FOCUSED_VALIDATION_RECEIPT_GROUP = "focused_validation"
COMMAND_EXECUTION_RECEIPT_KIND = "command_execution"
COMMAND_IDENTITY_ALGORITHM = "sha256"
COMMAND_IDENTITY_NORMALIZATION = "json_argv_utf8_compact_v1"
FOCUSED_VALIDATION_NODE = "cpu04"
FOCUSED_VALIDATION_PARTITION = "cpu"
FOCUSED_VALIDATION_JOB_NAME_PREFIX = "NrnSift_"
DEFAULT_PRODUCER_SOURCE = Path(
    "Libs/scripts/builders/build_operational_twin_sufficient_training_candidate.py"
)
DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE = Path(
    "Libs/scripts/builders/build_operational_twin_process_receipt.py"
)
DEFAULT_AGGREGATION_RECEIPT_PATH = Path(
    "Output/results/operational_twin/development_training/"
    "resource_receipts/aggregation_producer_identity.json"
)
DEFAULT_FOCUSED_VALIDATION_TESTS_RECEIPT_PATH = Path(
    "Output/results/operational_twin/development_training/"
    "resource_receipts/training_evidence_tests_command_execution.json"
)
DEFAULT_FOCUSED_VALIDATION_RUFF_RECEIPT_PATH = Path(
    "Output/results/operational_twin/development_training/"
    "resource_receipts/training_evidence_ruff_command_execution.json"
)
DEFAULT_FOCUSED_VALIDATION_TESTS_LOG_PATH = Path(
    "Output/results/operational_twin/development_training/"
    "logs/training_evidence_tests_command_execution.log"
)
DEFAULT_FOCUSED_VALIDATION_RUFF_LOG_PATH = Path(
    "Output/results/operational_twin/development_training/"
    "logs/training_evidence_ruff_command_execution.log"
)
PRODUCER_IDENTITY_CLASSIFICATION = {
    "scope": "producer_identity_only",
    "output_artifact_hashes": "not_included",
    "execution_success_claim": "not_made",
}


@dataclass(frozen=True)
class FocusedValidationReceiptContract:
    """Immutable source of truth for one admissible validation receipt."""

    contract_name: str
    receipt_uid: str
    output_path: Path
    log_path: Path
    argv: tuple[str, ...]
    required_clean_scopes: tuple[Path, ...]


FOCUSED_VALIDATION_RECEIPT_CONTRACTS = {
    "tests": FocusedValidationReceiptContract(
        contract_name="tests",
        receipt_uid="operational_twin_training_evidence_tests",
        output_path=DEFAULT_FOCUSED_VALIDATION_TESTS_RECEIPT_PATH,
        log_path=DEFAULT_FOCUSED_VALIDATION_TESTS_LOG_PATH,
        argv=(
            "python",
            "-m",
            "pytest",
            "Libs/tests/test_build_operational_twin_process_receipt.py",
            "Libs/tests/test_operational_twin_training_evidence_ledger.py",
            "-q",
        ),
        required_clean_scopes=(Path("Libs"), Path("pytest.ini")),
    ),
    "ruff": FocusedValidationReceiptContract(
        contract_name="ruff",
        receipt_uid="operational_twin_training_evidence_ruff",
        output_path=DEFAULT_FOCUSED_VALIDATION_RUFF_RECEIPT_PATH,
        log_path=DEFAULT_FOCUSED_VALIDATION_RUFF_LOG_PATH,
        argv=(
            "python",
            "-m",
            "ruff",
            "check",
            "Libs/scripts/builders/build_operational_twin_process_receipt.py",
            "Libs/scripts/builders/build_operational_twin_sufficient_training_candidate.py",
            "Libs/tests/test_build_operational_twin_process_receipt.py",
            "Libs/tests/test_operational_twin_training_evidence_ledger.py",
        ),
        required_clean_scopes=(Path("Libs"),),
    ),
}
FOCUSED_VALIDATION_SOURCE_STATE_METHOD = "git_object_at_code_commit_v1"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_RECEIPT_CANDIDATE_LEDGER_OUTPUTS = frozenset(
    {
        DEFAULT_AGGREGATION_RECEIPT_PATH.as_posix(),
        (
            "Output/results/operational_twin/development_training/"
            "catalog/sufficient_training_candidate.json"
        ),
        "Output/audits/operational_twin_sufficient_training_candidate.json",
        "Output/reports/operational_twin_sufficient_training_candidate.md",
        "Output/audits/operational_twin_training_evidence_ledger.json",
        DEFAULT_FOCUSED_VALIDATION_TESTS_RECEIPT_PATH.as_posix(),
        DEFAULT_FOCUSED_VALIDATION_RUFF_RECEIPT_PATH.as_posix(),
    }
)


class ProcessReceiptError(ValueError):
    """Raised when producer identity cannot be bound without ambiguity."""


def _run_git_text(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "")
        detail = str(stderr).strip() if stderr else str(exc)
        raise ProcessReceiptError(
            f"Git provenance query failed: git {' '.join(args)}: {detail}"
        ) from exc
    return completed.stdout


def _run_git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"")
        detail = (
            stderr.decode("utf-8", errors="replace").strip()
            if stderr
            else str(exc)
        )
        raise ProcessReceiptError(
            f"Git provenance query failed: git {' '.join(args)}: {detail}"
        ) from exc
    return completed.stdout


def _repository_relative_path(
    repo_root: Path,
    value: str | Path,
    *,
    label: str,
) -> tuple[Path, str]:
    resolved_root = repo_root.resolve()
    candidate = Path(value)
    resolved = (
        candidate if candidate.is_absolute() else resolved_root / candidate
    ).resolve()
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ProcessReceiptError(
            f"{label} must resolve inside the repository: {value}"
        ) from exc
    return resolved, relative


def _validate_producer_reference(
    source_path: object,
    *,
    receipt_path: str | None = None,
) -> str:
    if not isinstance(source_path, str) or not source_path:
        raise ProcessReceiptError("producer_source.path must be a non-empty string.")
    parsed = PurePosixPath(source_path)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or source_path != parsed.as_posix()
        or source_path == "."
    ):
        raise ProcessReceiptError(
            "producer_source.path must be a normalized repository-relative path."
        )
    if (
        source_path == receipt_path
        or source_path in _KNOWN_RECEIPT_CANDIDATE_LEDGER_OUTPUTS
    ):
        raise ProcessReceiptError(
            "producer_source.path must not reference a receipt, candidate, or ledger output."
        )
    return source_path


def _validate_log_reference(
    log_path: object,
    *,
    receipt_path: str | None = None,
) -> str:
    if not isinstance(log_path, str) or not log_path:
        raise ProcessReceiptError("log_source.path must be a non-empty string.")
    parsed = PurePosixPath(log_path)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or log_path != parsed.as_posix()
        or log_path == "."
    ):
        raise ProcessReceiptError(
            "log_source.path must be a normalized repository-relative path."
        )
    if log_path == receipt_path or log_path in _KNOWN_RECEIPT_CANDIDATE_LEDGER_OUTPUTS:
        raise ProcessReceiptError(
            "log_source.path must not reference a receipt, candidate, or ledger output."
        )
    return log_path


def focused_validation_contract(
    contract_name: str,
) -> FocusedValidationReceiptContract:
    """Return one of the two registered focused-validation contracts."""

    contract = FOCUSED_VALIDATION_RECEIPT_CONTRACTS.get(contract_name)
    if contract is None:
        expected = ", ".join(sorted(FOCUSED_VALIDATION_RECEIPT_CONTRACTS))
        raise ProcessReceiptError(
            f"Unknown focused-validation contract {contract_name!r}; "
            f"expected one of: {expected}."
        )
    return contract


def focused_validation_contract_for_receipt_path(
    receipt_path: str,
) -> FocusedValidationReceiptContract | None:
    """Resolve an exact semantic receipt path to its registered contract."""

    matches = [
        contract
        for contract in FOCUSED_VALIDATION_RECEIPT_CONTRACTS.values()
        if contract.output_path.as_posix() == receipt_path
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def focused_validation_contract_for_receipt_uid(
    receipt_uid: str,
) -> FocusedValidationReceiptContract | None:
    """Resolve an exact receipt UID to its registered contract."""

    matches = [
        contract
        for contract in FOCUSED_VALIDATION_RECEIPT_CONTRACTS.values()
        if contract.receipt_uid == receipt_uid
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _source_scope_identities(
    *,
    repo_root: Path,
    code_commit: str,
    contract: FocusedValidationReceiptContract,
) -> dict[str, object]:
    """Bind a clean working tree to Git objects at ``code_commit``."""

    relative_scopes: list[str] = []
    for scope in contract.required_clean_scopes:
        _, relative = _repository_relative_path(
            repo_root,
            scope,
            label="required clean source scope",
        )
        parsed = PurePosixPath(relative)
        if relative != parsed.as_posix() or relative in {"", "."}:
            raise ProcessReceiptError(
                "Required clean source scopes must be normalized repository-relative "
                f"paths: {relative!r}."
            )
        relative_scopes.append(relative)
    if len(relative_scopes) != len(set(relative_scopes)):
        raise ProcessReceiptError(
            f"Focused-validation contract {contract.contract_name!r} repeats a source scope."
        )

    status = _run_git_text(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *relative_scopes,
    ).strip()
    if status:
        raise ProcessReceiptError(
            "Required focused-validation source scopes contain dirty or untracked "
            f"paths and the command was not executed: {status}"
        )

    scopes: list[dict[str, str]] = []
    for relative in relative_scopes:
        object_id = _run_git_text(
            repo_root,
            "rev-parse",
            "--verify",
            f"{code_commit}:{relative}",
        ).strip()
        if _COMMIT_RE.fullmatch(object_id) is None:
            raise ProcessReceiptError(
                f"Git object identity is invalid for source scope {relative}: "
                f"{object_id!r}."
            )
        object_type = _run_git_text(
            repo_root,
            "cat-file",
            "-t",
            object_id,
        ).strip()
        if object_type not in {"blob", "tree"}:
            raise ProcessReceiptError(
                f"Unsupported Git object type for source scope {relative}: "
                f"{object_type!r}."
            )
        current_object_id = _run_git_text(
            repo_root,
            "rev-parse",
            "--verify",
            f"HEAD:{relative}",
        ).strip()
        if current_object_id != object_id:
            raise ProcessReceiptError(
                "Current committed source state differs from the receipt code commit "
                f"for required scope {relative}."
            )
        scopes.append(
            {
                "path": relative,
                "git_object_id": object_id,
                "git_object_type": object_type,
            }
        )
    return {
        "identity_method": FOCUSED_VALIDATION_SOURCE_STATE_METHOD,
        "required_clean_scopes": scopes,
    }


def command_identity(command: Sequence[str]) -> dict[str, object]:
    """Return the exact, deterministic identity of an argv vector."""

    argv = list(command)
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ProcessReceiptError("command must contain non-empty string argv values.")
    if any("\x00" in value for value in argv):
        raise ProcessReceiptError("command argv values must not contain NUL bytes.")
    canonical = json.dumps(
        argv,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "argv": argv,
        "algorithm": COMMAND_IDENTITY_ALGORITHM,
        "normalization": COMMAND_IDENTITY_NORMALIZATION,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_process_receipt_payload(
    payload: Mapping[str, object],
    *,
    receipt_path: str | None = None,
) -> dict[str, object]:
    """Validate one exact process-receipt variant and its internal identities."""

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(payload)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ProcessReceiptError(
            f"Process receipt schema validation failed at {location}: {error.message}"
        )
    producer_source = payload["producer_source"]
    if not isinstance(producer_source, Mapping):  # guarded by the schema
        raise ProcessReceiptError("producer_source must be a JSON object.")
    _validate_producer_reference(
        producer_source.get("path"),
        receipt_path=receipt_path,
    )

    process_receipt_kind = payload["process_receipt_kind"]
    if process_receipt_kind == COMMAND_EXECUTION_RECEIPT_KIND:
        receipt_uid = str(payload["receipt_uid"])
        contract = focused_validation_contract_for_receipt_uid(receipt_uid)
        if contract is None:
            raise ProcessReceiptError(
                f"Focused-validation receipt UID is not registered: {receipt_uid!r}."
            )
        expected_receipt_path = contract.output_path.as_posix()
        declared_receipt_path = str(payload["receipt_path"])
        if declared_receipt_path != expected_receipt_path:
            raise ProcessReceiptError(
                "Focused-validation receipt_path does not match its registered "
                f"contract: {declared_receipt_path!r}."
            )
        if receipt_path is not None and receipt_path != expected_receipt_path:
            raise ProcessReceiptError(
                "Focused-validation receipt was loaded from the wrong semantic path: "
                f"{receipt_path!r}; expected {expected_receipt_path!r}."
            )
        if (
            producer_source.get("path")
            != DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE.as_posix()
        ):
            raise ProcessReceiptError(
                "Focused-validation producer_source.path does not match the "
                "registered receipt builder."
            )
        command = payload["command_identity"]
        if not isinstance(command, Mapping):  # guarded by the schema
            raise ProcessReceiptError("command_identity must be a JSON object.")
        expected_identity = command_identity(contract.argv)
        if dict(command) != expected_identity:
            raise ProcessReceiptError(
                "command_identity does not match the exact registered argv identity."
            )
        source_state = payload["source_state"]
        if not isinstance(source_state, Mapping):  # guarded by the schema
            raise ProcessReceiptError("source_state must be a JSON object.")
        if (
            source_state.get("identity_method")
            != FOCUSED_VALIDATION_SOURCE_STATE_METHOD
        ):
            raise ProcessReceiptError(
                "source_state.identity_method does not match the registered contract."
            )
        scopes = source_state.get("required_clean_scopes")
        if not isinstance(scopes, Sequence) or isinstance(scopes, (str, bytes)):
            raise ProcessReceiptError(
                "source_state.required_clean_scopes must be an array."
            )
        declared_scope_paths = [
            str(scope.get("path", "")) if isinstance(scope, Mapping) else ""
            for scope in scopes
        ]
        expected_scope_paths = [
            scope.as_posix() for scope in contract.required_clean_scopes
        ]
        if declared_scope_paths != expected_scope_paths:
            raise ProcessReceiptError(
                "source_state.required_clean_scopes do not match the exact "
                "registered source scopes."
            )
        execution = payload["execution"]
        if not isinstance(execution, Mapping):  # guarded by the schema
            raise ProcessReceiptError("execution must be a JSON object.")
        exit_code = execution.get("exit_code")
        expected_status = "COMPLETED" if exit_code == 0 else "FAILED"
        if execution.get("process_status") != expected_status:
            raise ProcessReceiptError(
                "execution.process_status is inconsistent with execution.exit_code."
            )
        log_source = payload["log_source"]
        if not isinstance(log_source, Mapping):  # guarded by the schema
            raise ProcessReceiptError("log_source must be a JSON object.")
        _validate_log_reference(
            log_source.get("path"),
            receipt_path=receipt_path,
        )
        if log_source.get("path") != contract.log_path.as_posix():
            raise ProcessReceiptError(
                "log_source.path does not match the exact registered command log."
            )
    return dict(payload)


def _producer_source_identity(
    *,
    repo_root: Path,
    producer_source: str | Path,
    receipt_path: str,
) -> dict[str, str]:
    source, relative = _repository_relative_path(
        repo_root,
        producer_source,
        label="producer source",
    )
    _validate_producer_reference(relative, receipt_path=receipt_path)
    if not source.is_file():
        raise ProcessReceiptError(f"Producer source is missing: {relative}")

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ProcessReceiptError(f"Producer source is not tracked by Git: {relative}")
    status = _run_git_text(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative,
    ).rstrip("\n")
    if status:
        raise ProcessReceiptError(
            f"Producer source has uncommitted content and cannot be receipted: {relative}"
        )

    working_bytes = source.read_bytes()
    head_bytes = _run_git_bytes(repo_root, "cat-file", "blob", f"HEAD:{relative}")
    if working_bytes != head_bytes:
        raise ProcessReceiptError(
            f"Producer source is not byte-identical to HEAD: {relative}"
        )
    digest = hashlib.sha256(working_bytes).hexdigest()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ProcessReceiptError("Computed producer source SHA-256 is invalid.")
    return {"path": relative, "sha256": digest}


def _slurm_execution_identity() -> dict[str, object]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    job_name = os.environ.get("SLURM_JOB_NAME", "")
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".", 1)[0]
    if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None:
        raise ProcessReceiptError(
            "focused-validation command execution requires a numeric SLURM_JOB_ID."
        )
    if not job_name.startswith(FOCUSED_VALIDATION_JOB_NAME_PREFIX):
        raise ProcessReceiptError(
            "focused-validation command execution requires an NrnSift_ Slurm job name."
        )
    if partition != FOCUSED_VALIDATION_PARTITION:
        raise ProcessReceiptError(
            "focused-validation command execution must run in the cpu partition."
        )
    if node != FOCUSED_VALIDATION_NODE:
        raise ProcessReceiptError(
            "focused-validation command execution must run on cpu04."
        )
    return {
        "scheduler": "slurm",
        "slurm_job_id": job_id,
        "job_name": job_name,
        "partition": partition,
        "node": node,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_producer_identity_receipt(
    *,
    repo_root: Path = REPO_ROOT,
    producer_source: str | Path = DEFAULT_PRODUCER_SOURCE,
    output_path: str | Path = DEFAULT_AGGREGATION_RECEIPT_PATH,
    receipt_kind: str = RECEIPT_KIND,
    receipt_group: str = RECEIPT_GROUP,
    process_receipt_kind: str = PROCESS_RECEIPT_KIND,
) -> dict[str, object]:
    """Build, but do not write, the fixed producer-identity receipt."""

    resolved_root = repo_root.resolve()
    git_root = Path(
        _run_git_text(resolved_root, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    if git_root != resolved_root:
        raise ProcessReceiptError(
            f"repo_root must be the Git worktree root: {resolved_root}"
        )
    if receipt_kind != RECEIPT_KIND:
        raise ProcessReceiptError(
            f"receipt_kind must be {RECEIPT_KIND!r}, not {receipt_kind!r}."
        )
    if receipt_group != RECEIPT_GROUP:
        raise ProcessReceiptError(
            f"receipt_group must be {RECEIPT_GROUP!r}, not {receipt_group!r}."
        )
    if process_receipt_kind != PROCESS_RECEIPT_KIND:
        raise ProcessReceiptError(
            "process_receipt_kind must be "
            f"{PROCESS_RECEIPT_KIND!r}, not {process_receipt_kind!r}."
        )

    _, output_relative = _repository_relative_path(
        resolved_root,
        output_path,
        label="receipt output",
    )
    code_commit = _run_git_text(resolved_root, "rev-parse", "HEAD").strip()
    if _COMMIT_RE.fullmatch(code_commit) is None:
        raise ProcessReceiptError(
            f"Current Git commit is not a full hexadecimal identity: {code_commit!r}"
        )
    producer_identity = _producer_source_identity(
        repo_root=resolved_root,
        producer_source=producer_source,
        receipt_path=output_relative,
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_uid": RECEIPT_UID,
        "receipt_kind": RECEIPT_KIND,
        "receipt_group": RECEIPT_GROUP,
        "process_receipt_kind": PROCESS_RECEIPT_KIND,
        "code_commit": code_commit,
        "producer_source": producer_identity,
        "classification": dict(PRODUCER_IDENTITY_CLASSIFICATION),
    }
    return validate_process_receipt_payload(
        payload,
        receipt_path=output_relative,
    )


def write_producer_identity_receipt(
    *,
    repo_root: Path = REPO_ROOT,
    producer_source: str | Path = DEFAULT_PRODUCER_SOURCE,
    output_path: str | Path = DEFAULT_AGGREGATION_RECEIPT_PATH,
    receipt_kind: str = RECEIPT_KIND,
    receipt_group: str = RECEIPT_GROUP,
    process_receipt_kind: str = PROCESS_RECEIPT_KIND,
) -> tuple[Path, dict[str, object]]:
    """Build and atomically write the deterministic receipt JSON."""

    resolved_root = repo_root.resolve()
    output, _ = _repository_relative_path(
        resolved_root,
        output_path,
        label="receipt output",
    )
    payload = build_producer_identity_receipt(
        repo_root=resolved_root,
        producer_source=producer_source,
        output_path=output,
        receipt_kind=receipt_kind,
        receipt_group=receipt_group,
        process_receipt_kind=process_receipt_kind,
    )
    _atomic_write_json(output, payload)
    return output, payload


def _run_registered_command(
    *,
    contract: FocusedValidationReceiptContract,
    repo_root: Path,
    stdout: object,
) -> int:
    """Execute one registered argv without a shell and return its exit code."""

    try:
        completed = subprocess.run(
            list(contract.argv),
            cwd=repo_root,
            check=False,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        raise ProcessReceiptError(
            f"Unable to execute focused-validation command: {exc}"
        ) from exc
    return int(completed.returncode)


def execute_registered_command_receipt(
    *,
    contract_name: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, dict[str, object], int]:
    """Execute one immutable registered command and write its exact receipt.

    Both successful and failed commands produce an honest receipt.  Callers
    must propagate the returned exit code; final evidence collection accepts
    only a zero-exit ``COMPLETED`` receipt.  Required source scopes are checked
    for dirty and untracked paths before execution and re-bound afterward.
    """

    resolved_root = repo_root.resolve()
    git_root = Path(
        _run_git_text(resolved_root, "rev-parse", "--show-toplevel").strip()
    ).resolve()
    if git_root != resolved_root:
        raise ProcessReceiptError(
            f"repo_root must be the Git worktree root: {resolved_root}"
        )
    contract = focused_validation_contract(contract_name)

    output, output_relative = _repository_relative_path(
        resolved_root,
        contract.output_path,
        label="receipt output",
    )
    log, log_relative = _repository_relative_path(
        resolved_root,
        contract.log_path,
        label="command log",
    )
    _validate_log_reference(log_relative, receipt_path=output_relative)
    exact_command_identity = command_identity(contract.argv)
    execution_identity = _slurm_execution_identity()
    code_commit = _run_git_text(resolved_root, "rev-parse", "HEAD").strip()
    if _COMMIT_RE.fullmatch(code_commit) is None:
        raise ProcessReceiptError(
            f"Current Git commit is not a full hexadecimal identity: {code_commit!r}"
        )
    producer_identity = _producer_source_identity(
        repo_root=resolved_root,
        producer_source=DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE,
        receipt_path=output_relative,
    )
    source_state = _source_scope_identities(
        repo_root=resolved_root,
        code_commit=code_commit,
        contract=contract,
    )

    log.parent.mkdir(parents=True, exist_ok=True)
    temporary_log: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=log.parent,
            prefix=f".{log.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_log = Path(handle.name)
            exit_code = _run_registered_command(
                contract=contract,
                repo_root=resolved_root,
                stdout=handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_log.replace(log)
    finally:
        if temporary_log is not None and temporary_log.exists():
            temporary_log.unlink()

    post_code_commit = _run_git_text(resolved_root, "rev-parse", "HEAD").strip()
    if post_code_commit != code_commit:
        raise ProcessReceiptError(
            "Git HEAD changed while the focused-validation command was executing."
        )
    post_source_state = _source_scope_identities(
        repo_root=resolved_root,
        code_commit=code_commit,
        contract=contract,
    )
    if post_source_state != source_state:
        raise ProcessReceiptError(
            "Required focused-validation source state changed during execution."
        )
    post_producer_identity = _producer_source_identity(
        repo_root=resolved_root,
        producer_source=DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE,
        receipt_path=output_relative,
    )
    if post_producer_identity != producer_identity:
        raise ProcessReceiptError(
            "Focused-validation receipt producer changed during execution."
        )

    log_bytes = log.read_bytes()
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_uid": contract.receipt_uid,
        "receipt_kind": RECEIPT_KIND,
        "receipt_group": FOCUSED_VALIDATION_RECEIPT_GROUP,
        "process_receipt_kind": COMMAND_EXECUTION_RECEIPT_KIND,
        "receipt_path": output_relative,
        "code_commit": code_commit,
        "producer_source": producer_identity,
        "command_identity": exact_command_identity,
        "source_state": source_state,
        "execution": {
            **execution_identity,
            "exit_code": exit_code,
            "process_status": "COMPLETED" if exit_code == 0 else "FAILED",
        },
        "log_source": {
            "path": log_relative,
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
            "size_bytes": len(log_bytes),
        },
    }
    validated = validate_process_receipt_payload(
        payload,
        receipt_path=output_relative,
    )
    _atomic_write_json(output, validated)
    return output, validated, exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--producer-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt-kind")
    parser.add_argument("--receipt-group")
    parser.add_argument("--process-receipt-kind")
    parser.add_argument(
        "--focused-validation-contract",
        choices=tuple(sorted(FOCUSED_VALIDATION_RECEIPT_CONTRACTS)),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.focused_validation_contract:
        if any(
            value is not None
            for value in (
                args.producer_source,
                args.output,
                args.receipt_kind,
                args.receipt_group,
                args.process_receipt_kind,
            )
        ):
            raise SystemExit(
                "BLOCKED: focused-validation contracts do not accept semantic "
                "overrides."
            )
        try:
            output, payload, exit_code = execute_registered_command_receipt(
                contract_name=args.focused_validation_contract,
                repo_root=args.repo_root,
            )
        except ProcessReceiptError as exc:
            raise SystemExit(f"BLOCKED: {exc}") from exc
        print(
            json.dumps(
                {
                    "status": payload["execution"]["process_status"],
                    "output": output.as_posix(),
                    "receipt_uid": payload["receipt_uid"],
                    "exit_code": exit_code,
                },
                sort_keys=True,
            )
        )
        return exit_code if 0 <= exit_code <= 255 else 1

    try:
        output, payload = write_producer_identity_receipt(
            repo_root=args.repo_root,
            producer_source=args.producer_source or DEFAULT_PRODUCER_SOURCE,
            output_path=args.output or DEFAULT_AGGREGATION_RECEIPT_PATH,
            receipt_kind=args.receipt_kind or RECEIPT_KIND,
            receipt_group=args.receipt_group or RECEIPT_GROUP,
            process_receipt_kind=args.process_receipt_kind or PROCESS_RECEIPT_KIND,
        )
    except ProcessReceiptError as exc:
        raise SystemExit(f"BLOCKED: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "PRODUCER_IDENTITY_RECEIPT_WRITTEN",
                "output": output.as_posix(),
                "receipt_uid": payload["receipt_uid"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
