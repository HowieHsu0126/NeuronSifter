"""Aggregate formal open-data Operational Twin development-training evidence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_operational_twin_devcal import (  # noqa: E402
    EXPECTED_ACCESS_POLICY,
    EXPECTED_CLAIM_BOUNDARY,
    _canonical_hash,
    require_calibration_execution_authority,
)
from Libs.scripts.builders.build_operational_twin_process_receipt import (  # noqa: E402
    COMMAND_EXECUTION_RECEIPT_KIND,
    DEFAULT_AGGREGATION_RECEIPT_PATH,
    FOCUSED_VALIDATION_RECEIPT_CONTRACTS,
    FOCUSED_VALIDATION_RECEIPT_GROUP,
    ProcessReceiptError,
    command_identity as focused_validation_command_identity,
    focused_validation_contract_for_receipt_path,
    validate_process_receipt_payload,
)
from Libs.utils.evaluation.operational_twin_development_evidence import (  # noqa: E402
    build_development_comparisons,
    training_budget_evidence,
    validate_assignment_development_evidence,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    METHOD_EXECUTION_CONTRACTS,
    REQUIRED_METHOD_IDS,
    REQUIRED_ZERO_ACCESS_COUNTERS,
    OperationalTwinProtocolError,
    canonical_hash,
    repo_path,
    validate_assignment_runtime_evidence,
)
from Libs.utils.evaluation.operational_twin_training_evidence_ledger import (  # noqa: E402
    COMMAND_IDENTITY_ALGORITHM,
    COMMAND_IDENTITY_NORMALIZATION,
    COMMAND_IDENTITY_SOURCE_FIELD,
    FINAL_EVIDENCE_REFERENCE_ROLES,
    PROCESS_RECEIPT_GROUPS,
    RECEIPT_KIND_BY_GROUP,
    REQUIRED_RECEIPT_GROUPS,
    SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION,
    SCHEDULER_RECEIPT_GROUPS,
    attempt_provenance_gaps,
    build_operational_twin_training_evidence_ledger,
    collect_attempt_uid_references,
    validate_operational_twin_training_evidence_ledger,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


DEFAULT_TRAINING_EVIDENCE_LEDGER = Path(
    "Output/audits/operational_twin_training_evidence_ledger.json"
)
FINAL_RECEIPT_SOURCE_PATHS: dict[str, tuple[Path, ...]] = {
    "training": (
        Path(
            "Output/results/operational_twin/development_training/"
            "resource_receipts/formal_attempts_round37.sacct"
        ),
    ),
    "evaluation": (
        Path(
            "Output/results/operational_twin/development_training/"
            "resource_receipts/development_evaluation_sacct.txt"
        ),
    ),
    "reconciliation": (
        Path(
            "Output/results/operational_twin/development_training/"
            "logs/training_evidence_reconciliation.log"
        ),
    ),
    "aggregation": (
        DEFAULT_AGGREGATION_RECEIPT_PATH,
    ),
    "focused_validation": (
        *(
            FOCUSED_VALIDATION_RECEIPT_CONTRACTS[name].output_path
            for name in ("tests", "ruff")
        ),
    ),
}
ASSIGNMENT_EVIDENCE_PATHS = {
    "terminal": Path("terminal_status.json"),
    "provenance": Path("provenance/runtime_provenance.json"),
    "evaluation": Path("evaluation/summary.json"),
    "reconciliation": Path("provenance/evidence_reconciliation.json"),
}
EVALUATION_DEPENDENCY_PATHS = (
    Path("evaluation/development_world_results.jsonl"),
    Path("evaluation/development_pair_results.jsonl"),
    Path("evaluation/calibration_world_results.jsonl"),
    Path("evaluation/calibration_pair_results.jsonl"),
    Path("evaluation/support_abstention_results.jsonl"),
)
TRAINABLE_DEPENDENCY_PATHS = (
    Path("config.resolved.yaml"),
    Path("hashes/config_hash.txt"),
    Path("hashes/data_manifest_hashes.json"),
    Path("hashes/seed_hash.txt"),
    Path("hashes/world_action_schema_hash.txt"),
)
STATIC_LEDGER_DEPENDENCY_PATHS = (
    Path("Libs/configs/config_stack.yaml"),
    Path("Libs/configs/model/operational_twin.yaml"),
    Path("Libs/configs/schema/operational_twin_training_evidence_ledger.schema.json"),
    Path("Libs/utils/evaluation/operational_twin_training_evidence_ledger.py"),
    Path(
        "Output/results/operational_twin/development_training/"
        "catalog/assignment_commands.json"
    ),
)
ZERO_SHA256 = "0" * 64
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
ATTEMPT_ORDINAL_RE = re.compile(r"__attempt_(\d+)$")
_MISSING_MAPPING = object()
_LEGACY_ACCESS_OMISSION = "sealed_material_access"
FAILURE_ATTEMPT_STATUSES = frozenset(
    {
        "FAILED",
        "CANCELLED",
        "INTERRUPTED",
        "NON_CONVERGED",
        "UNSTABLE",
        "UNDEFINED",
    }
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_metric(output_root: Path, method_id: str) -> dict[str, object] | None:
    method_metrics = _read_jsonl(output_root / "metrics" / f"{method_id}_metrics.jsonl")
    if method_metrics:
        return method_metrics[-1]
    dev_metrics = _read_jsonl(output_root / "metrics/dev_metrics.jsonl")
    return dev_metrics[-1] if dev_metrics else None


def _non_finite_paths(value: object, path: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            paths.extend(_non_finite_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_non_finite_paths(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
    return paths


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_blocker(
    blockers: list[dict[str, str]],
    code: str,
    message: str,
) -> None:
    blocker = {"code": code, "message": message}
    if blocker not in blockers:
        blockers.append(blocker)


def _retained_failure_attempt_uids(
    path: Path,
    *,
    blockers: list[dict[str, str]],
    assignment_uid: str,
) -> set[str]:
    retained_failures: list[Mapping[str, object]] = []
    for row_index, row in enumerate(_read_jsonl(path), start=1):
        if "retained_failures" not in row:
            retained_failures.append(row)
            continue
        nested_failures = row.get("retained_failures")
        if not isinstance(nested_failures, Sequence) or isinstance(
            nested_failures, (str, bytes)
        ):
            _append_blocker(
                blockers,
                "failure_retention_invalid",
                f"{assignment_uid} failure summary row {row_index} must contain a retained_failures array.",
            )
            continue
        for failure_index, failure in enumerate(nested_failures, start=1):
            if not isinstance(failure, Mapping):
                _append_blocker(
                    blockers,
                    "failure_retention_invalid",
                    f"{assignment_uid} failure summary row {row_index} retained failure {failure_index} must be an object.",
                )
                continue
            retained_failures.append(failure)

    attempt_uids: set[str] = set()
    for failure_index, failure in enumerate(retained_failures, start=1):
        attempt_uid = str(failure.get("attempt_uid") or "")
        if not attempt_uid:
            _append_blocker(
                blockers,
                "failure_attempt_identity_missing",
                f"{assignment_uid} retained failure {failure_index} lacks attempt_uid.",
            )
            continue
        attempt_uids.add(attempt_uid)
    return attempt_uids


def _validate_failure_attempt_retention(
    path: Path,
    *,
    retained_attempt_uids: set[str],
    blockers: list[dict[str, str]],
    assignment_uid: str,
) -> None:
    failure_attempt_uids = _retained_failure_attempt_uids(
        path,
        blockers=blockers,
        assignment_uid=assignment_uid,
    )
    for missing_attempt_uid in sorted(failure_attempt_uids - retained_attempt_uids):
        _append_blocker(
            blockers,
            "failure_attempt_source_missing",
            f"{assignment_uid} failure attempt {missing_attempt_uid} is absent from attempts/.",
        )


def _submit_line_command_identity(value: object) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    exact_submit_line = value.removesuffix("\n")
    if (
        not exact_submit_line.strip()
        or "\n" in exact_submit_line
        or "\r" in exact_submit_line
    ):
        return None
    return {
        "algorithm": COMMAND_IDENTITY_ALGORITHM,
        "normalization": COMMAND_IDENTITY_NORMALIZATION,
        "source_field": COMMAND_IDENTITY_SOURCE_FIELD,
        "value": hashlib.sha256(exact_submit_line.encode("utf-8")).hexdigest(),
    }


def _command_receipt_entry(
    *,
    attempt_uid: str,
    source: Mapping[str, object] | None,
    scheduler_receipt: Mapping[str, object] | None = None,
    required: bool = True,
    blockers: list[dict[str, str]],
) -> dict[str, object] | None:
    if source is not None:
        return {
            "receipt_uid": f"command__{attempt_uid}",
            "source": dict(source),
        }

    if isinstance(scheduler_receipt, Mapping):
        scheduler = scheduler_receipt.get("scheduler")
        scheduler_source = scheduler_receipt.get("source")
        expected_command_identity = scheduler_receipt.get("command_identity")
        job_id = scheduler.get("job_id") if isinstance(scheduler, Mapping) else None
        command_identity = _submit_line_command_identity(
            scheduler_receipt.get("submit_line")
        )
        if (
            isinstance(job_id, str)
            and job_id
            and isinstance(scheduler_source, Mapping)
            and scheduler_source.get("tracked") is True
            and command_identity is not None
            and isinstance(expected_command_identity, Mapping)
            and dict(expected_command_identity) == command_identity
        ):
            return {
                "receipt_uid": f"command__{attempt_uid}",
                "source": dict(scheduler_source),
                "source_classification": SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION,
                "scheduler_job_id": job_id,
                "command_identity": command_identity,
            }

    if required:
        _append_blocker(
            blockers,
            "command_receipt_missing",
            f"{attempt_uid} lacks a retained command receipt.",
        )
    return None


def _apply_attempt_provenance_dispositions(
    *,
    assignment_uid: str,
    terminal_attempt_uid: str,
    attempt_rows: list[dict[str, object]],
    blockers: list[dict[str, str]],
) -> None:
    terminal_positions = [
        index
        for index, attempt in enumerate(attempt_rows)
        if attempt.get("attempt_uid") == terminal_attempt_uid
    ]
    if len(terminal_positions) != 1:
        _append_blocker(
            blockers,
            "terminal_attempt_missing",
            f"{assignment_uid} does not retain exactly one terminal attempt.",
        )
    elif terminal_positions[0] != len(attempt_rows) - 1:
        _append_blocker(
            blockers,
            "terminal_attempt_not_last",
            f"{assignment_uid} terminal attempt is not the final retained ordinal.",
        )

    for index, attempt in enumerate(attempt_rows):
        attempt_uid = str(attempt.get("attempt_uid", ""))
        is_terminal = attempt_uid == terminal_attempt_uid
        provenance_gaps = list(attempt_provenance_gaps(attempt))
        attempt["provenance_gaps"] = provenance_gaps
        attempt["provenance_disposition"] = (
            "complete_contributing"
            if is_terminal
            else (
                "complete_noncontributing"
                if not provenance_gaps
                else "incomplete_noncontributing"
            )
        )
        attempt["contributes_to_candidate"] = is_terminal
        if is_terminal:
            attempt["superseded_by_attempt_uid"] = None
        elif index + 1 < len(attempt_rows):
            successor = attempt_rows[index + 1]
            successor_uid = str(successor.get("attempt_uid", ""))
            attempt["superseded_by_attempt_uid"] = successor_uid or None
            if not successor_uid:
                _append_blocker(
                    blockers,
                    "supersession_target_missing",
                    f"{attempt_uid or '<missing>'} has no identified later attempt.",
                )
            elif successor.get("assignment_uid") != assignment_uid:
                _append_blocker(
                    blockers,
                    "supersession_cross_assignment",
                    f"{attempt_uid} successor {successor_uid} is not in {assignment_uid}.",
                )
            elif int(successor.get("ordinal", 0)) <= int(attempt.get("ordinal", 0)):
                _append_blocker(
                    blockers,
                    "supersession_order_invalid",
                    f"{attempt_uid} successor {successor_uid} is not forward ordered.",
                )
        else:
            attempt["superseded_by_attempt_uid"] = None
            _append_blocker(
                blockers,
                "supersession_target_missing",
                f"{attempt_uid or '<missing>'} has no later retained attempt.",
            )


def _final_evidence_attempt_references(
    *,
    payloads: Mapping[str, Mapping[str, object]],
    checkpoint_evidence: object,
) -> dict[str, list[str]]:
    final_payloads: dict[str, object] = {
        "terminal": payloads.get("terminal"),
        "provenance": payloads.get("provenance"),
        "evaluation": payloads.get("evaluation"),
        "reconciliation": payloads.get("reconciliation"),
        "checkpoint": checkpoint_evidence,
    }
    return {
        role: list(collect_attempt_uid_references(final_payloads.get(role)))
        for role in FINAL_EVIDENCE_REFERENCE_ROLES
    }


def _repo_relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path if path.is_absolute() else resolved_root / path
    return resolved_path.resolve().relative_to(resolved_root).as_posix()


def _git_tree_state(root: Path) -> tuple[set[str], set[str]]:
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--"],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    )
    return set(tracked.splitlines()), set(changed.splitlines())


def _source_identity(
    *,
    root: Path,
    path: Path,
    committed_paths: set[str],
    changed_paths: set[str],
    blockers: list[dict[str, str]],
    label: str,
    cache: dict[str, dict[str, object]],
    expected_sha256: str | None = None,
) -> dict[str, object]:
    try:
        relative = _repo_relative(root, path)
        absolute = root / relative
    except ValueError:
        relative = path.as_posix()
        absolute = path
        _append_blocker(
            blockers,
            "source_outside_repository",
            f"{label} resolves outside the repository: {relative}.",
        )

    cached = cache.get(relative)
    if cached is None:
        if not absolute.is_file():
            identity: dict[str, object] = {
                "path": relative,
                "sha256": ZERO_SHA256,
                "size_bytes": 0,
                "tracked": False,
            }
            _append_blocker(
                blockers,
                "source_missing",
                f"{label} is missing: {relative}.",
            )
        else:
            tracked = relative in committed_paths and relative not in changed_paths
            identity = {
                "path": relative,
                "sha256": _sha256_file(absolute),
                "size_bytes": absolute.stat().st_size,
                "tracked": tracked,
            }
            if relative not in committed_paths:
                _append_blocker(
                    blockers,
                    "source_untracked",
                    f"{label} is absent from the committed Git tree: {relative}.",
                )
            elif relative in changed_paths:
                _append_blocker(
                    blockers,
                    "source_content_uncommitted",
                    f"{label} differs from the committed Git tree: {relative}.",
                )
        cache[relative] = identity
        cached = identity

    if expected_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            _append_blocker(
                blockers,
                "declared_source_sha256_invalid",
                f"{label} declares an invalid SHA-256 for {relative}.",
            )
        elif cached["sha256"] != expected_sha256:
            _append_blocker(
                blockers,
                "source_sha256_mismatch",
                f"{label} content does not match its declared SHA-256: {relative}.",
            )
    return dict(cached)


def _read_json_mapping(
    path: Path,
    *,
    blockers: list[dict[str, str]],
    label: str,
) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _append_blocker(
            blockers,
            "source_json_invalid",
            f"{label} is not valid JSON: {error}.",
        )
        return {}
    if not isinstance(value, Mapping):
        _append_blocker(
            blockers,
            "source_json_invalid",
            f"{label} must contain a JSON object.",
        )
        return {}
    return dict(value)


def _json_mappings_from_log(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= 20:
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else None


def _first_present(value: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        candidate = value.get(key)
        if candidate not in (None, ""):
            return candidate
    return None


def _normalise_scheduler(value: Mapping[str, object]) -> dict[str, object]:
    scheduler_value = value.get("scheduler")
    if not isinstance(scheduler_value, Mapping):
        scheduler_value = value.get("scheduler_identity")
    scheduler = dict(scheduler_value) if isinstance(scheduler_value, Mapping) else dict(value)

    job_id_value = _first_present(scheduler, "job_id", "slurm_job_id", "JobIDRaw", "JobID")
    task_value = _first_present(
        scheduler,
        "task_id",
        "slurm_array_task_id",
        "array_task_id",
    )
    job_id = str(job_id_value) if job_id_value is not None else ""
    if task_value is None:
        match = re.fullmatch(r"\d+_(\d+)", job_id)
        task_id = match.group(1) if match else None
    else:
        task_id = str(task_value)

    resources_value = scheduler.get("resources")
    resources = dict(resources_value) if isinstance(resources_value, Mapping) else {}
    allocation_value = _first_present(
        resources,
        "allocation",
    ) or _first_present(scheduler, "allocation", "AllocTRES", "TRES")
    allocation = str(allocation_value) if allocation_value is not None else ""

    cpus = _optional_int(_first_present(resources, "cpus"))
    if cpus is None:
        cpus = _optional_int(_first_present(scheduler, "cpus", "AllocCPUS", "NCPUS"))
    if cpus is None and allocation:
        match = re.search(r"(?:^|,)cpu=(\d+)(?:,|$)", allocation)
        cpus = int(match.group(1)) if match else None

    gpus = _optional_int(_first_present(resources, "gpus"))
    if gpus is None:
        gpus = _optional_int(
            _first_present(
                scheduler,
                "gpus",
                "slurm_gpus_on_node",
                "AllocGPUS",
            )
        )
    if gpus is None and allocation:
        match = re.search(r"(?:gres/)?gpu(?:=[^,]+:)?=(\d+)", allocation)
        if match is None:
            match = re.search(r"(?:^|,)(?:gres/)?gpu=(\d+)(?:,|$)", allocation)
        gpus = int(match.group(1)) if match else 0

    memory_value = _first_present(resources, "memory") or _first_present(
        scheduler,
        "memory",
        "ReqMem",
        "MaxRSS",
    )
    if memory_value is None and allocation:
        match = re.search(r"(?:^|,)mem=([^,]+)", allocation)
        memory_value = match.group(1) if match else None

    elapsed_value = _first_present(resources, "elapsed") or _first_present(
        scheduler,
        "elapsed",
        "Elapsed",
    )
    normalised: dict[str, object] = {
        "job_id": job_id,
        "task_id": task_id,
        "job_name": str(
            _first_present(scheduler, "job_name", "slurm_job_name", "JobName") or ""
        ),
        "node": str(
            _first_present(scheduler, "node", "hostname", "NodeList") or ""
        ),
        "partition": str(
            _first_present(scheduler, "partition", "slurm_partition", "Partition") or ""
        ),
        "state": str(_first_present(scheduler, "state", "State") or ""),
        "exit_code": str(
            _first_present(scheduler, "exit_code", "ExitCode") or ""
        ),
        "resources": {},
    }
    normalised_resources = normalised["resources"]
    assert isinstance(normalised_resources, dict)
    if cpus is not None:
        normalised_resources["cpus"] = cpus
    if gpus is not None:
        normalised_resources["gpus"] = gpus
    if memory_value is not None:
        normalised_resources["memory"] = str(memory_value)
    if elapsed_value is not None:
        normalised_resources["elapsed"] = str(elapsed_value)
    if allocation:
        normalised_resources["allocation"] = allocation
    return normalised


def _parse_sacct_rows(text: str) -> list[dict[str, object]]:
    lines = [line for line in text.splitlines() if "|" in line]
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "JobID" in line and "JobName" in line and "State" in line
        ),
        None,
    )
    if header_index is None:
        return []
    header = [field.strip() for field in lines[header_index].split("|")]
    rows: list[dict[str, object]] = []
    for line in lines[header_index + 1 :]:
        values = line.split("|")
        if len(values) != len(header):
            continue
        row = {
            field: value if field == "SubmitLine" else value.strip()
            for field, value in zip(header, values, strict=True)
        }
        job_id = str(row.get("JobIDRaw") or row.get("JobID") or "")
        if not job_id or "." in job_id:
            continue
        rows.append(row)
    return rows


def _receipt_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        receipts = value.get("receipts")
        if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)):
            return [dict(row) for row in receipts if isinstance(row, Mapping)]
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, Mapping)]

    json_rows = [
        row
        for row in _json_mappings_from_log(path)
        if any(
            key in row
            for key in (
                "receipt_uid",
                "receipt_kind",
                "scheduler",
                "scheduler_identity",
                "code_commit",
                "aggregation_code_commit",
                "evaluation_code_commit",
                "training_code_commit",
                "validation_code_commit",
            )
        )
    ]
    if json_rows:
        return json_rows
    return _parse_sacct_rows(text)


def _receipt_code_commit(group: str, row: Mapping[str, object]) -> str:
    keys_by_group = {
        "training": ("code_commit", "training_code_commit", "command_git_commit", "git_commit", "CodeCommit"),
        "evaluation": ("code_commit", "evaluation_code_commit", "git_commit", "CodeCommit"),
        "reconciliation": ("training_code_commit", "code_commit", "git_commit", "CodeCommit"),
        "aggregation": ("code_commit", "aggregation_code_commit", "git_commit", "CodeCommit"),
        "focused_validation": ("code_commit", "validation_code_commit", "focused_validation_code_commit", "git_commit", "CodeCommit"),
    }
    value = _first_present(row, *keys_by_group[group])
    return str(value) if value is not None else ""


def _git_receipt_query(root: Path, *args: str) -> str:
    """Run one collector-owned Git query for receipt source verification."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "")
        detail = str(stderr).strip() if stderr else str(exc)
        raise ProcessReceiptError(
            f"Git receipt verification failed: git {' '.join(args)}: {detail}"
        ) from exc
    return completed.stdout.strip()


def _verify_focused_validation_receipt_contract(
    *,
    root: Path,
    receipt_source_path: str,
    row: Mapping[str, object],
    blockers: list[dict[str, str]],
) -> tuple[bool, str | None]:
    """Independently verify one focused receipt against its registered contract."""

    contract = focused_validation_contract_for_receipt_path(receipt_source_path)
    if contract is None:
        _append_blocker(
            blockers,
            "focused_validation_receipt_path_invalid",
            "Focused-validation receipt is not at one of the two registered "
            f"semantic paths: {receipt_source_path}.",
        )
        return False, None

    valid = True

    def reject(message: str) -> None:
        nonlocal valid
        valid = False
        _append_blocker(
            blockers,
            "focused_validation_contract_mismatch",
            message,
        )

    if row.get("receipt_path") != contract.output_path.as_posix():
        reject(
            f"{receipt_source_path} does not declare its registered receipt path."
        )
    if row.get("receipt_uid") != contract.receipt_uid:
        reject(f"{receipt_source_path} does not declare its registered receipt UID.")
    if row.get("command_identity") != focused_validation_command_identity(
        contract.argv
    ):
        reject(f"{receipt_source_path} does not bind its exact registered argv.")

    log_source = row.get("log_source")
    if (
        not isinstance(log_source, Mapping)
        or log_source.get("path") != contract.log_path.as_posix()
    ):
        reject(f"{receipt_source_path} does not bind its registered semantic log path.")

    source_state = row.get("source_state")
    scope_rows: list[Mapping[str, object]] = []
    if isinstance(source_state, Mapping):
        raw_scopes = source_state.get("required_clean_scopes")
        if isinstance(raw_scopes, Sequence) and not isinstance(
            raw_scopes, (str, bytes)
        ):
            scope_rows = [scope for scope in raw_scopes if isinstance(scope, Mapping)]
    expected_scope_paths = [
        scope.as_posix() for scope in contract.required_clean_scopes
    ]
    declared_scope_paths = [str(scope.get("path", "")) for scope in scope_rows]
    if (
        len(scope_rows) != len(expected_scope_paths)
        or declared_scope_paths != expected_scope_paths
    ):
        reject(
            f"{receipt_source_path} does not bind its exact registered source scopes."
        )
        return valid, contract.contract_name

    code_commit = str(row.get("code_commit", ""))
    if COMMIT_RE.fullmatch(code_commit) is None:
        reject(f"{receipt_source_path} lacks a full source-state code commit.")
        return valid, contract.contract_name

    try:
        resolved_commit = _git_receipt_query(
            root,
            "rev-parse",
            "--verify",
            f"{code_commit}^{{commit}}",
        )
        if resolved_commit != code_commit:
            reject(
                f"{receipt_source_path} code commit does not resolve to its exact "
                "declared identity."
            )
        status = _git_receipt_query(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *expected_scope_paths,
        )
    except ProcessReceiptError as error:
        _append_blocker(
            blockers,
            "focused_validation_source_state_unverifiable",
            f"{receipt_source_path} source state cannot be verified: {error}",
        )
        return False, contract.contract_name
    if status:
        valid = False
        _append_blocker(
            blockers,
            "focused_validation_source_scope_dirty",
            f"{receipt_source_path} required source scopes are dirty or contain "
            f"untracked paths: {status}",
        )

    for scope, declared in zip(
        expected_scope_paths,
        scope_rows,
        strict=True,
    ):
        try:
            commit_object_id = _git_receipt_query(
                root,
                "rev-parse",
                "--verify",
                f"{code_commit}:{scope}",
            )
            object_type = _git_receipt_query(
                root,
                "cat-file",
                "-t",
                commit_object_id,
            )
            current_object_id = _git_receipt_query(
                root,
                "rev-parse",
                "--verify",
                f"HEAD:{scope}",
            )
        except ProcessReceiptError as error:
            valid = False
            _append_blocker(
                blockers,
                "focused_validation_source_state_unverifiable",
                f"{receipt_source_path} source scope {scope} cannot be verified: "
                f"{error}",
            )
            continue
        if (
            declared.get("git_object_id") != commit_object_id
            or declared.get("git_object_type") != object_type
            or current_object_id != commit_object_id
        ):
            valid = False
            _append_blocker(
                blockers,
                "focused_validation_source_state_mismatch",
                f"{receipt_source_path} source scope {scope} does not match both "
                "its receipt code commit and current committed source state.",
            )
    return valid, contract.contract_name


def _collect_receipt_groups(
    *,
    root: Path,
    committed_paths: set[str],
    changed_paths: set[str],
    blockers: list[dict[str, str]],
    source_cache: dict[str, dict[str, object]],
    receipt_source_paths: Mapping[str, Sequence[str | Path]],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, set[str]],
    list[dict[str, object]],
]:
    groups: dict[str, list[dict[str, object]]] = {
        group: [] for group in REQUIRED_RECEIPT_GROUPS
    }
    commits: dict[str, set[str]] = {
        group: set() for group in REQUIRED_RECEIPT_GROUPS
    }
    training_scheduler_rows: list[dict[str, object]] = []
    focused_validation_contracts: list[str] = []
    for group in REQUIRED_RECEIPT_GROUPS:
        expected_kind = RECEIPT_KIND_BY_GROUP[group]
        sources = tuple(receipt_source_paths.get(group, ()))
        if not sources:
            _append_blocker(
                blockers,
                "receipt_group_missing",
                f"No source path is configured for final receipt group {group}.",
            )
            continue
        for source_value in sources:
            source_path = Path(source_value)
            absolute = source_path if source_path.is_absolute() else root / source_path
            source = _source_identity(
                root=root,
                path=absolute,
                committed_paths=committed_paths,
                changed_paths=changed_paths,
                blockers=blockers,
                label=f"{group} receipt source",
                cache=source_cache,
            )
            rows = _receipt_records(absolute)
            source_is_untyped = False
            if not rows:
                if absolute.is_file():
                    rows = [{}]
                    source_is_untyped = True
                    _append_blocker(
                        blockers,
                        "receipt_source_untyped",
                        f"{source['path']} contains no typed {expected_kind} receipt.",
                    )
                else:
                    continue
            for index, row in enumerate(rows):
                declared_kind_value = row.get("receipt_kind")
                declared_kind = (
                    str(declared_kind_value)
                    if declared_kind_value not in (None, "")
                    else None
                )
                legacy_reconciliation = (
                    group == "reconciliation"
                    and declared_kind is None
                    and isinstance(row.get("training_code_commit"), str)
                    and COMMIT_RE.fullmatch(str(row["training_code_commit"])) is not None
                )
                row_is_typed = not source_is_untyped and (
                    group in SCHEDULER_RECEIPT_GROUPS
                    or declared_kind == expected_kind
                    or legacy_reconciliation
                )
                if (
                    group in {"aggregation", FOCUSED_VALIDATION_RECEIPT_GROUP}
                    and declared_kind == expected_kind
                    and not source_is_untyped
                ):
                    try:
                        validate_process_receipt_payload(
                            row,
                            receipt_path=str(source["path"]),
                        )
                    except ProcessReceiptError as error:
                        row_is_typed = False
                        _append_blocker(
                            blockers,
                            "process_receipt_invalid",
                            f"{source['path']} is not a valid {group} process "
                            f"receipt: {error}",
                        )
                    else:
                        producer_source = row["producer_source"]
                        if not isinstance(producer_source, Mapping):
                            row_is_typed = False
                        else:
                            producer_identity = _source_identity(
                                root=root,
                                path=Path(str(producer_source["path"])),
                                committed_paths=committed_paths,
                                changed_paths=changed_paths,
                                blockers=blockers,
                                label=f"{group} receipt producer source",
                                cache=source_cache,
                                expected_sha256=str(producer_source["sha256"]),
                            )
                            if (
                                producer_identity.get("tracked") is not True
                                or producer_identity.get("sha256")
                                != producer_source["sha256"]
                            ):
                                row_is_typed = False
                        if group == FOCUSED_VALIDATION_RECEIPT_GROUP:
                            log_source = row["log_source"]
                            if not isinstance(log_source, Mapping):
                                row_is_typed = False
                            else:
                                log_identity = _source_identity(
                                    root=root,
                                    path=Path(str(log_source["path"])),
                                    committed_paths=committed_paths,
                                    changed_paths=changed_paths,
                                    blockers=blockers,
                                    label="focused-validation command log",
                                    cache=source_cache,
                                    expected_sha256=str(log_source["sha256"]),
                                )
                                if (
                                    log_identity.get("tracked") is not True
                                    or log_identity.get("sha256")
                                    != log_source["sha256"]
                                    or log_identity.get("size_bytes")
                                    != log_source["size_bytes"]
                                ):
                                    row_is_typed = False
                                    if (
                                        log_identity.get("size_bytes")
                                        != log_source["size_bytes"]
                                    ):
                                        _append_blocker(
                                            blockers,
                                            "source_size_mismatch",
                                            "Focused-validation command log size does "
                                            "not match its declared identity: "
                                            f"{log_source['path']}.",
                                        )
                            execution = row["execution"]
                            if (
                                row.get("process_receipt_kind")
                                != COMMAND_EXECUTION_RECEIPT_KIND
                                or not isinstance(execution, Mapping)
                                or execution.get("exit_code") != 0
                                or execution.get("process_status") != "COMPLETED"
                            ):
                                row_is_typed = False
                                _append_blocker(
                                    blockers,
                                    "process_receipt_execution_failed",
                                    f"{source['path']} does not prove a zero-exit "
                                    "focused-validation command execution.",
                                )
                            contract_valid, contract_name = (
                                _verify_focused_validation_receipt_contract(
                                    root=root,
                                    receipt_source_path=str(source["path"]),
                                    row=row,
                                    blockers=blockers,
                                )
                            )
                            if not contract_valid:
                                row_is_typed = False
                            elif contract_name is not None:
                                focused_validation_contracts.append(contract_name)
                if (
                    group in PROCESS_RECEIPT_GROUPS
                    and declared_kind is None
                    and not legacy_reconciliation
                    and not source_is_untyped
                ):
                    _append_blocker(
                        blockers,
                        "receipt_source_untyped",
                        f"{source['path']} contains no typed {expected_kind} receipt.",
                    )
                if declared_kind is not None and declared_kind != expected_kind:
                    _append_blocker(
                        blockers,
                        "receipt_kind_invalid",
                        f"{source['path']} declares receipt kind {declared_kind!r}; expected {expected_kind!r} for {group}.",
                    )

                scheduler = _normalise_scheduler(row)
                job_id = str(scheduler.get("job_id", ""))
                receipt_uid = str(
                    row.get("receipt_uid")
                    or f"{group}__{Path(str(source['path'])).stem}__{job_id or index + 1}"
                )
                code_commit = _receipt_code_commit(group, row) if row_is_typed else ""
                if code_commit and COMMIT_RE.fullmatch(code_commit):
                    commits[group].add(code_commit)
                elif group in PROCESS_RECEIPT_GROUPS or code_commit:
                    _append_blocker(
                        blockers,
                        "receipt_code_commit_missing",
                        f"{receipt_uid} lacks a full {group} producer code commit identity.",
                    )
                receipt = {
                    "receipt_uid": receipt_uid,
                    "receipt_kind": declared_kind or expected_kind,
                    "source": source,
                }
                if group in SCHEDULER_RECEIPT_GROUPS:
                    receipt["scheduler"] = scheduler
                    command_identity = _submit_line_command_identity(
                        row.get("SubmitLine")
                    )
                    if command_identity is not None:
                        receipt["command_identity"] = command_identity
                    if code_commit:
                        receipt["code_commit"] = code_commit
                else:
                    receipt["code_commit"] = code_commit
                groups[group].append(receipt)
                if group == "training" and job_id:
                    submit_line = row.get("SubmitLine")
                    training_scheduler_rows.append(
                        {
                            "scheduler": scheduler,
                            "source": source,
                            "command_identity": receipt.get("command_identity"),
                            "submit_line": (
                                submit_line if isinstance(submit_line, str) else ""
                            ),
                        }
                    )
        if not groups[group]:
            _append_blocker(
                blockers,
                "receipt_group_missing",
                f"Final receipt group {group} has no retained receipt records.",
            )
        if group == FOCUSED_VALIDATION_RECEIPT_GROUP:
            expected_contracts = sorted(FOCUSED_VALIDATION_RECEIPT_CONTRACTS)
            if sorted(focused_validation_contracts) != expected_contracts:
                _append_blocker(
                    blockers,
                    "focused_validation_contract_set_invalid",
                    "Focused-validation receipts must contain exactly one tests "
                    "contract and one Ruff contract; observed "
                    f"{sorted(focused_validation_contracts)}.",
                )
    return groups, commits, training_scheduler_rows


def _command_log_index(logs_root: Path) -> list[tuple[Path, dict[str, object]]]:
    if not logs_root.is_dir():
        return []
    rows: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(logs_root.glob("*.out"), key=lambda item: item.as_posix()):
        for payload in _json_mappings_from_log(path):
            if payload.get("selected_assignment"):
                rows.append((path, payload))
                break
    return rows


def _match_command_log(
    *,
    assignment_uid: str,
    scheduler: Mapping[str, object],
    command_logs: Sequence[tuple[Path, Mapping[str, object]]],
) -> list[tuple[Path, Mapping[str, object]]]:
    job_id = str(scheduler.get("slurm_job_id") or "")
    array_job_id = str(scheduler.get("slurm_array_job_id") or "")
    task_id = str(scheduler.get("slurm_array_task_id") or "")
    matches: list[tuple[Path, Mapping[str, object]]] = []
    for path, payload in command_logs:
        if str(payload.get("selected_assignment") or "") != assignment_uid:
            continue
        payload_job_id = str(payload.get("slurm_job_id") or "")
        payload_task_id = str(
            payload.get("slurm_array_task_id")
            or payload.get("array_task_id")
            or ""
        )
        if job_id and payload_job_id == job_id:
            matches.append((path, payload))
        elif array_job_id and task_id and payload_task_id == task_id and array_job_id in path.name:
            matches.append((path, payload))
    return matches


def _match_scheduler_receipt(
    scheduler: Mapping[str, object],
    receipt_rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    job_id = str(scheduler.get("job_id") or scheduler.get("slurm_job_id") or "")
    array_job_id = str(scheduler.get("slurm_array_job_id") or "")
    task_id = str(
        scheduler.get("task_id") or scheduler.get("slurm_array_task_id") or ""
    )
    candidates = {job_id}
    if array_job_id and task_id:
        candidates.add(f"{array_job_id}_{task_id}")
    candidates.discard("")
    matches: list[Mapping[str, object]] = []
    for row in receipt_rows:
        receipt_scheduler = row.get("scheduler")
        scheduler_identity = (
            receipt_scheduler if isinstance(receipt_scheduler, Mapping) else row
        )
        if str(scheduler_identity.get("job_id", "")) in candidates:
            matches.append(row)
    return matches[0] if len(matches) == 1 else None


def _consistent_mapping(
    *,
    values: Sequence[tuple[str, object, bool]],
    expected: Mapping[str, object],
    blockers: list[dict[str, str]],
    assignment_uid: str,
    field: str,
    legacy_access_normalization_authorized: bool = False,
) -> dict[str, object]:
    observed: list[tuple[str, dict[str, object]]] = []
    authoritative_mapping_observed = False
    legacy_access_policy = {
        key: value
        for key, value in expected.items()
        if key != _LEGACY_ACCESS_OMISSION
    }
    for label, value, authoritative in values:
        if value is _MISSING_MAPPING:
            continue
        if not isinstance(value, Mapping):
            _append_blocker(
                blockers,
                f"{field}_invalid",
                f"{assignment_uid} {label} declares a non-mapping {field}.",
            )
            continue
        mapping = dict(value)
        if (
            field == "access_flags"
            and mapping == legacy_access_policy
            and set(mapping) == set(expected) - {_LEGACY_ACCESS_OMISSION}
            and legacy_access_normalization_authorized
        ):
            mapping = dict(expected)
        observed.append((label, mapping))
        authoritative_mapping_observed = authoritative_mapping_observed or authoritative

    if not authoritative_mapping_observed:
        _append_blocker(
            blockers,
            f"{field}_authoritative_missing",
            (
                f"{assignment_uid} has no explicit authoritative terminal, "
                f"evaluation, or attempt {field}."
            ),
        )

    selected = observed[0][1] if observed else {}
    for label, value in observed:
        if value != selected:
            _append_blocker(
                blockers,
                f"{field}_inconsistent",
                f"{assignment_uid} {label} disagrees with the retained {field}.",
            )
        if value != dict(expected):
            _append_blocker(
                blockers,
                f"{field}_policy_mismatch",
                f"{assignment_uid} {label} does not match the frozen {field} policy.",
            )
    return selected


def _authoritative_access_policy(
    *,
    protocol: Mapping[str, object],
    preflight: Mapping[str, object],
    protocol_source: Mapping[str, object],
    preflight_source: Mapping[str, object],
    blockers: list[dict[str, str]],
) -> tuple[dict[str, object], bool]:
    """Return the frozen access policy and whether legacy normalization is safe."""

    raw_protocol_policy = protocol.get("access_counters")
    protocol_policy = (
        dict(raw_protocol_policy)
        if isinstance(raw_protocol_policy, Mapping)
        else {}
    )
    required_keys = set(REQUIRED_ZERO_ACCESS_COUNTERS)
    protocol_policy_valid = (
        set(protocol_policy) == required_keys
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value == 0
            for value in protocol_policy.values()
        )
    )
    if not protocol_policy_valid:
        _append_blocker(
            blockers,
            "access_policy_authority_invalid",
            "The frozen training protocol must declare the exact ten-key zero-access policy.",
        )

    raw_preflight_policy = preflight.get("access_counters")
    preflight_policy = (
        dict(raw_preflight_policy)
        if isinstance(raw_preflight_policy, Mapping)
        else {}
    )
    preflight_matches = (
        preflight_policy == protocol_policy
        and preflight.get("protocol_uid") == protocol.get("protocol_uid")
    )
    if not preflight_matches:
        _append_blocker(
            blockers,
            "access_policy_authority_mismatch",
            "The training preflight access policy does not match the frozen protocol.",
        )

    sources_tracked = True
    for label, source in (
        ("training protocol", protocol_source),
        ("training preflight", preflight_source),
    ):
        if source.get("tracked") is not True:
            sources_tracked = False
            _append_blocker(
                blockers,
                "access_policy_authority_untracked",
                f"The {label} is not present unchanged in the committed Git tree.",
            )

    return protocol_policy, bool(
        protocol_policy_valid and preflight_matches and sources_tracked
    )


def _valid_commit(value: object) -> str:
    commit = str(value or "")
    return commit if COMMIT_RE.fullmatch(commit) else ""


def _optional_attempt_commit(value: object) -> str | None:
    commit = _valid_commit(value)
    return commit or None


def _assignment_commit(
    *,
    role: str,
    sources: Sequence[tuple[str, object]],
    blockers: list[dict[str, str]],
    assignment_uid: str,
) -> str:
    commits = {_valid_commit(value) for _, value in sources}
    commits.discard("")
    if len(commits) != 1:
        _append_blocker(
            blockers,
            f"assignment_{role}_commit_invalid",
            f"{assignment_uid} has {len(commits)} distinct valid {role} commit identities.",
        )
        return next(iter(sorted(commits)), "")
    return next(iter(commits))


def _attempt_ordinal(attempt_uid: str, payload: Mapping[str, object]) -> int:
    declared = _optional_int(payload.get("ordinal"))
    if declared is not None:
        return declared
    match = ATTEMPT_ORDINAL_RE.search(attempt_uid)
    return int(match.group(1)) if match else 0


def _dependency_candidates(
    *,
    root: Path,
    protocol: Mapping[str, object],
    assignment_sources: Sequence[Mapping[str, object]],
) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}

    def add(value: object, expected_sha256: object = None) -> None:
        if not isinstance(value, (str, Path)) or not str(value):
            return
        try:
            relative = _repo_relative(root, Path(value))
        except ValueError:
            relative = Path(value).as_posix()
        expected = str(expected_sha256 or "")
        candidates.setdefault(relative, set())
        if expected:
            candidates[relative].add(expected)

    for path in STATIC_LEDGER_DEPENDENCY_PATHS:
        add(path)
    for value in dict(protocol.get("manifests", {})).values():
        add(value)
    add(protocol.get("open_data_authority"))
    roots = dict(protocol.get("output_roots", {}))
    add(roots.get("resource_receipt"))

    for source in assignment_sources:
        output_root = Path(str(source.get("output_root", "")))
        provenance = source.get("provenance")
        if isinstance(provenance, Mapping):
            for key, value in dict(provenance.get("canonical_path_bindings", {})).items():
                if key != "output_root":
                    add(value)
            for receipt in dict(provenance.get("reference_receipt_identities", {})).values():
                if isinstance(receipt, Mapping):
                    add(receipt.get("path"))
        reconciliation = source.get("reconciliation")
        if isinstance(reconciliation, Mapping):
            for preserved in dict(reconciliation.get("preserved_originals", {})).values():
                if isinstance(preserved, Mapping):
                    add(preserved.get("path"), preserved.get("sha256"))

        runtime_contract = source.get("runtime_contract")
        if isinstance(runtime_contract, Mapping):
            metrics_value = runtime_contract.get("metrics_path")
            if metrics_value:
                metrics_path = output_root / str(metrics_value)
                if (root / metrics_path).is_file():
                    add(metrics_path)

            failure_value = runtime_contract.get("failure_retention_path")
            if failure_value:
                failure_path = output_root / str(failure_value)
                if bool(source.get("failure_evidence_required")) or (
                    root / failure_path
                ).is_file():
                    add(failure_path)
        for path in EVALUATION_DEPENDENCY_PATHS:
            add(output_root / path)
        if bool(source.get("checkpoint_required")):
            for path in TRAINABLE_DEPENDENCY_PATHS:
                add(output_root / path)
    return candidates


def _finalise_ledger(
    *,
    protocol: Mapping[str, object],
    protocol_identity: Mapping[str, object],
    assignments: Sequence[Mapping[str, object]],
    receipt_groups: Mapping[str, Sequence[Mapping[str, object]]],
    access_flags: Mapping[str, int],
    claim_flags: Mapping[str, bool],
    code_commits: Mapping[str, Sequence[str]],
    dependencies: Sequence[Mapping[str, object]],
    collector_blockers: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    blockers = [dict(row) for row in collector_blockers]
    ledger = build_operational_twin_training_evidence_ledger(
        protocol=protocol,
        protocol_identity=protocol_identity,
        assignments=assignments,
        receipt_groups=receipt_groups,
        access_flags=access_flags,
        claim_flags=claim_flags,
        code_commits=code_commits,
        dependencies=dependencies,
        validation_blockers=blockers,
    )
    semantic_blockers = validate_operational_twin_training_evidence_ledger(
        ledger,
        protocol,
    )
    for blocker in semantic_blockers:
        if blocker["code"] != "reported_validation_blockers":
            _append_blocker(blockers, blocker["code"], blocker["message"])
    if blockers != list(collector_blockers):
        ledger = build_operational_twin_training_evidence_ledger(
            protocol=protocol,
            protocol_identity=protocol_identity,
            assignments=assignments,
            receipt_groups=receipt_groups,
            access_flags=access_flags,
            claim_flags=claim_flags,
            code_commits=code_commits,
            dependencies=dependencies,
            validation_blockers=blockers,
        )
    return ledger


def collect_operational_twin_training_evidence_ledger(
    *,
    root: Path = REPO_ROOT,
    protocol: Mapping[str, object] | None = None,
    manifest: Mapping[str, object] | None = None,
    committed_paths: set[str] | None = None,
    changed_paths: set[str] | None = None,
    receipt_source_paths: Mapping[str, Sequence[str | Path]] = FINAL_RECEIPT_SOURCE_PATHS,
) -> dict[str, object]:
    """Collect a deterministic, fail-closed ledger from manifest-bound evidence.

    The collector reads only public development/calibration artifacts declared by
    the frozen protocol and its 15-assignment manifest.  Large checkpoints are
    reduced to path, SHA-256, and byte size.  A source is marked tracked only
    when its current bytes are present unchanged in ``HEAD``.
    """

    root = root.resolve()
    blockers: list[dict[str, str]] = []
    source_cache: dict[str, dict[str, object]] = {}
    if protocol is None:
        config = load_config_stack(root / "Libs/configs/config_stack.yaml")
        protocol = dict(config["operational_twin_training_protocol"])
    else:
        protocol = dict(protocol)
    roots = dict(protocol.get("output_roots", {}))
    manifest_path = repo_path(
        root,
        str(roots.get("assignment_manifest", "")),
        must_exist=False,
    )
    if manifest is None:
        manifest = _read_json_mapping(
            manifest_path,
            blockers=blockers,
            label="assignment manifest",
        )
    else:
        manifest = dict(manifest)

    if committed_paths is None:
        try:
            committed_paths, detected_changes = _git_tree_state(root)
        except (OSError, subprocess.CalledProcessError) as error:
            committed_paths = set()
            detected_changes = set()
            _append_blocker(
                blockers,
                "git_tree_unavailable",
                f"The committed Git tree could not be inspected: {error}.",
            )
        if changed_paths is None:
            changed_paths = detected_changes
    if changed_paths is None:
        changed_paths = set()

    protocol_source_path = root / "Libs/configs/model/operational_twin_training_protocol.yaml"
    preflight_path = repo_path(
        root,
        str(roots.get("preflight_authority", "")),
        must_exist=False,
    )
    protocol_source = _source_identity(
        root=root,
        path=protocol_source_path,
        committed_paths=committed_paths,
        changed_paths=changed_paths,
        blockers=blockers,
        label="training protocol",
        cache=source_cache,
    )
    preflight_source = _source_identity(
        root=root,
        path=preflight_path,
        committed_paths=committed_paths,
        changed_paths=changed_paths,
        blockers=blockers,
        label="training preflight authority",
        cache=source_cache,
    )
    preflight = _read_json_mapping(
        preflight_path,
        blockers=blockers,
        label="training preflight authority",
    )
    expected_access, legacy_access_normalization_authorized = (
        _authoritative_access_policy(
            protocol=protocol,
            preflight=preflight,
            protocol_source=protocol_source,
            preflight_source=preflight_source,
            blockers=blockers,
        )
    )
    protocol_identity = {
        "protocol_uid": str(protocol.get("protocol_uid", "")),
        "protocol_schema_version": str(protocol.get("schema_version", "")),
        "protocol_semantic_sha256": canonical_hash(protocol),
        "sources": {
            "protocol": protocol_source,
            "assignment_manifest": _source_identity(
                root=root,
                path=manifest_path,
                committed_paths=committed_paths,
                changed_paths=changed_paths,
                blockers=blockers,
                label="assignment manifest",
                cache=source_cache,
            ),
            "preflight_authority": preflight_source,
        },
    }

    manifest_payload = {
        key: value for key, value in manifest.items() if key != "assignment_manifest_hash"
    }
    if manifest.get("assignment_manifest_hash") != canonical_hash(manifest_payload):
        _append_blocker(
            blockers,
            "assignment_manifest_hash_mismatch",
            "The assignment manifest self-identity does not match its canonical content.",
        )
    if manifest.get("protocol_uid") != protocol.get("protocol_uid"):
        _append_blocker(
            blockers,
            "assignment_manifest_protocol_mismatch",
            "The assignment manifest does not bind the frozen protocol UID.",
        )

    receipt_groups, receipt_commits, training_scheduler_rows = _collect_receipt_groups(
        root=root,
        committed_paths=committed_paths,
        changed_paths=changed_paths,
        blockers=blockers,
        source_cache=source_cache,
        receipt_source_paths=receipt_source_paths,
    )
    assignment_root = repo_path(
        root,
        str(roots.get("assignment_root", "")),
        must_exist=False,
    )
    command_logs = _command_log_index(assignment_root / "logs/assignments")
    assignment_rows: list[dict[str, object]] = []
    assignment_sources: list[dict[str, object]] = []
    assignment_commit_groups: dict[str, set[str]] = {
        "training": set(),
        "evaluation": set(),
        "reconciliation": set(),
    }
    raw_assignments = manifest.get("assignments", [])
    if not isinstance(raw_assignments, Sequence) or isinstance(raw_assignments, (str, bytes)):
        raw_assignments = []
        _append_blocker(
            blockers,
            "assignment_manifest_invalid",
            "The assignment manifest assignments field is not an array.",
        )

    for raw_assignment in raw_assignments:
        if not isinstance(raw_assignment, Mapping):
            _append_blocker(
                blockers,
                "assignment_manifest_invalid",
                "The assignment manifest contains a non-object assignment.",
            )
            continue
        manifest_assignment = dict(raw_assignment)
        assignment_uid = str(manifest_assignment.get("assignment_uid", ""))
        method_id = str(manifest_assignment.get("method_id", ""))
        seed = _optional_int(manifest_assignment.get("seed"))
        if seed is None:
            seed = -1
        output_root = repo_path(
            root,
            str(manifest_assignment.get("output_root", "")),
            must_exist=False,
        )
        evidence: dict[str, dict[str, object]] = {}
        payloads: dict[str, dict[str, object]] = {}
        for role, relative_path in ASSIGNMENT_EVIDENCE_PATHS.items():
            path = output_root / relative_path
            expected_hash: str | None = None
            if role == "terminal":
                reconciliation_path = output_root / ASSIGNMENT_EVIDENCE_PATHS["reconciliation"]
                reconciliation_hint = _read_json_mapping(
                    reconciliation_path,
                    blockers=[],
                    label=f"{assignment_uid} reconciliation",
                )
                value = reconciliation_hint.get("reconciled_terminal_status_sha256")
                expected_hash = str(value) if value else None
            elif role == "provenance":
                reconciliation_path = output_root / ASSIGNMENT_EVIDENCE_PATHS["reconciliation"]
                reconciliation_hint = _read_json_mapping(
                    reconciliation_path,
                    blockers=[],
                    label=f"{assignment_uid} reconciliation",
                )
                value = reconciliation_hint.get("reconciled_runtime_provenance_sha256")
                expected_hash = str(value) if value else None
            evidence[role] = _source_identity(
                root=root,
                path=path,
                committed_paths=committed_paths,
                changed_paths=changed_paths,
                blockers=blockers,
                label=f"{assignment_uid} {role} evidence",
                cache=source_cache,
                expected_sha256=expected_hash,
            )
            payloads[role] = _read_json_mapping(
                path,
                blockers=blockers,
                label=f"{assignment_uid} {role} evidence",
            )

        for role, payload in payloads.items():
            for field, expected in (
                ("assignment_uid", assignment_uid),
                ("method_id", method_id),
                ("seed", seed),
            ):
                if payload and payload.get(field) != expected:
                    _append_blocker(
                        blockers,
                        "assignment_evidence_identity_mismatch",
                        f"{assignment_uid} {role} evidence disagrees on {field}.",
                    )

        terminal = payloads["terminal"]
        provenance = payloads["provenance"]
        evaluation = payloads["evaluation"]
        reconciliation = payloads["reconciliation"]
        terminal_attempt_uid = str(terminal.get("attempt_uid") or "")
        reconciliation_command = reconciliation.get("command_receipt")
        command_reference = (
            dict(reconciliation_command)
            if isinstance(reconciliation_command, Mapping)
            else {}
        )

        attempts_dir = output_root / "attempts"
        attempt_paths = sorted(
            attempts_dir.glob(f"{assignment_uid}__attempt_*.json"),
            key=lambda item: item.name,
        ) if attempts_dir.is_dir() else []
        if not attempt_paths:
            _append_blocker(
                blockers,
                "attempt_sources_missing",
                f"{assignment_uid} has no retained attempt files.",
            )
        attempt_rows: list[dict[str, object]] = []
        attempt_payloads: list[dict[str, object]] = []
        prior_unsuccessful_attempt_uid: str | None = None
        for attempt_path in attempt_paths:
            attempt_source = _source_identity(
                root=root,
                path=attempt_path,
                committed_paths=committed_paths,
                changed_paths=changed_paths,
                blockers=blockers,
                label=f"{assignment_uid} attempt",
                cache=source_cache,
            )
            attempt_payload = _read_json_mapping(
                attempt_path,
                blockers=blockers,
                label=f"{assignment_uid} attempt",
            )
            attempt_payloads.append(attempt_payload)
            attempt_uid = str(attempt_payload.get("attempt_uid") or attempt_path.stem)
            is_terminal_attempt = attempt_uid == terminal_attempt_uid
            ordinal = _attempt_ordinal(attempt_uid, attempt_payload)
            status = str(attempt_payload.get("status") or "")
            scheduler_payload = attempt_payload.get("scheduler")
            if not isinstance(scheduler_payload, Mapping):
                scheduler_payload = attempt_payload.get("scheduler_identity")
            scheduler_raw = (
                dict(scheduler_payload) if isinstance(scheduler_payload, Mapping) else {}
            )
            scheduler_receipt = _match_scheduler_receipt(
                scheduler_raw,
                training_scheduler_rows,
            )
            if scheduler_receipt is not None:
                retained_scheduler = scheduler_receipt.get("scheduler")
                merged_scheduler = (
                    dict(retained_scheduler)
                    if isinstance(retained_scheduler, Mapping)
                    else {}
                )
                merged_scheduler.update(scheduler_raw)
                nested_resources = (
                    retained_scheduler.get("resources")
                    if isinstance(retained_scheduler, Mapping)
                    else None
                )
                if isinstance(nested_resources, Mapping):
                    merged_scheduler["resources"] = dict(nested_resources)
                scheduler = _normalise_scheduler(merged_scheduler)
            else:
                scheduler = None
                if is_terminal_attempt:
                    _append_blocker(
                        blockers,
                        "attempt_scheduler_receipt_missing",
                        f"{attempt_uid} has no matching retained training scheduler receipt.",
                    )

            command_path: Path | None = None
            command_payload: Mapping[str, object] = {}
            expected_command_sha256: str | None = None
            if attempt_uid == terminal_attempt_uid and command_reference.get("path"):
                command_path = repo_path(
                    root,
                    str(command_reference["path"]),
                    must_exist=False,
                )
                expected_command_sha256 = str(command_reference.get("sha256") or "") or None
                rows = _json_mappings_from_log(command_path)
                command_payload = rows[0] if rows else command_reference
            else:
                command_matches = _match_command_log(
                    assignment_uid=assignment_uid,
                    scheduler=scheduler_raw,
                    command_logs=command_logs,
                )
                if len(command_matches) == 1:
                    command_path, command_payload = command_matches[0]
                elif len(command_matches) > 1:
                    _append_blocker(
                        blockers,
                        "command_receipt_ambiguous",
                        f"{attempt_uid} matches multiple retained command receipts.",
                    )
            command_source = None
            if command_path is not None:
                command_source = _source_identity(
                    root=root,
                    path=command_path,
                    committed_paths=committed_paths,
                    changed_paths=changed_paths,
                    blockers=blockers,
                    label=f"{attempt_uid} command receipt",
                    cache=source_cache,
                    expected_sha256=expected_command_sha256,
                )
            command_receipt = _command_receipt_entry(
                attempt_uid=attempt_uid,
                source=command_source,
                scheduler_receipt=scheduler_receipt,
                required=is_terminal_attempt,
                blockers=blockers,
            )
            for field, expected in (
                ("selected_assignment", assignment_uid),
                ("method_id", method_id),
                ("seed", seed),
            ):
                if command_payload and command_payload.get(field) != expected:
                    _append_blocker(
                        blockers,
                        "command_receipt_identity_mismatch",
                        f"{attempt_uid} command receipt disagrees on {field}.",
                    )

            recovered_from = attempt_payload.get("recovered_from_attempt_uid")
            if recovered_from is None and status == "COMPLETED" and prior_unsuccessful_attempt_uid:
                recovered_from = prior_unsuccessful_attempt_uid
                status = "RECOVERED"
            if status in FAILURE_ATTEMPT_STATUSES:
                prior_unsuccessful_attempt_uid = attempt_uid
            elif status == "RECOVERED":
                prior_unsuccessful_attempt_uid = None

            attempt_commit = _optional_attempt_commit(
                attempt_payload.get("code_commit")
                or attempt_payload.get("training_code_commit")
                or command_payload.get("command_git_commit")
                or command_payload.get("code_commit")
            )
            if not attempt_commit and is_terminal_attempt:
                _append_blocker(
                    blockers,
                    "attempt_code_commit_missing",
                    f"{attempt_uid} lacks a full execution code commit.",
                )
            attempt_rows.append(
                {
                    "assignment_uid": str(attempt_payload.get("assignment_uid") or ""),
                    "method_id": str(attempt_payload.get("method_id") or ""),
                    "seed": _optional_int(attempt_payload.get("seed")) or -1,
                    "attempt_uid": attempt_uid,
                    "ordinal": ordinal,
                    "status": status,
                    "terminal_reason": str(attempt_payload.get("terminal_reason") or ""),
                    "recovered_from_attempt_uid": (
                        str(recovered_from) if recovered_from is not None else None
                    ),
                    "source": attempt_source,
                    "command_receipt": command_receipt,
                    "scheduler": scheduler,
                    "code_commit": attempt_commit,
                }
            )

        failure_path = output_root / str(
            dict(manifest_assignment.get("runtime_contract", {})).get(
                "failure_retention_path",
                f"failures/{method_id}_failures.jsonl",
            )
        )
        retained_attempt_uids = {str(row["attempt_uid"]) for row in attempt_rows}
        _validate_failure_attempt_retention(
            failure_path,
            retained_attempt_uids=retained_attempt_uids,
            blockers=blockers,
            assignment_uid=assignment_uid,
        )

        checkpoint_required = bool(
            METHOD_EXECUTION_CONTRACTS.get(method_id, {}).get(
                "checkpoint_creation_allowed"
            )
        )
        checkpoint_evidence = reconciliation.get("checkpoint_evidence")
        if checkpoint_required:
            checkpoint_record = (
                dict(checkpoint_evidence)
                if isinstance(checkpoint_evidence, Mapping)
                else {}
            )
            checkpoint_path_value = checkpoint_record.get("path") or (
                output_root / "checkpoints/best_dev.pt"
            )
            checkpoint_path = repo_path(root, checkpoint_path_value, must_exist=False)
            checkpoint_sha256 = ZERO_SHA256
            checkpoint_size = 0
            if checkpoint_path.is_file():
                checkpoint_sha256 = _sha256_file(checkpoint_path)
                checkpoint_size = checkpoint_path.stat().st_size
            else:
                _append_blocker(
                    blockers,
                    "checkpoint_missing",
                    f"{assignment_uid} selected checkpoint is missing: {_repo_relative(root, checkpoint_path)}.",
                )
            declared_checkpoint_hashes = {
                str(value)
                for value in (
                    checkpoint_record.get("sha256"),
                    provenance.get("checkpoint_hash"),
                    evaluation.get("checkpoint_sha256"),
                )
                if value
            }
            if len(declared_checkpoint_hashes) != 1 or (
                checkpoint_size > 0 and checkpoint_sha256 not in declared_checkpoint_hashes
            ):
                _append_blocker(
                    blockers,
                    "checkpoint_identity_mismatch",
                    f"{assignment_uid} checkpoint source identities are missing or inconsistent.",
                )
            checkpoint = {
                "contract": "checkpoint_identity",
                "repository_storage": "external_content_addressed",
                "path": _repo_relative(root, checkpoint_path),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_size,
                "reason": None,
            }
        else:
            if checkpoint_evidence not in (None, {}):
                _append_blocker(
                    blockers,
                    "unexpected_checkpoint_evidence",
                    f"{assignment_uid} is nontrainable but carries checkpoint evidence.",
                )
            checkpoint = {
                "contract": "no_checkpoint",
                "repository_storage": "not_applicable",
                "path": None,
                "sha256": None,
                "size_bytes": 0,
                "reason": "nontrainable_method_contract",
            }

        access_flags = _consistent_mapping(
            values=[
                ("terminal", terminal.get("access_flags", _MISSING_MAPPING), True),
                (
                    "provenance",
                    provenance.get("access_flags", _MISSING_MAPPING),
                    False,
                ),
                (
                    "evaluation",
                    evaluation.get("access_flags", _MISSING_MAPPING),
                    True,
                ),
                *[
                    (
                        f"attempt {index + 1}",
                        payload.get("access_flags", _MISSING_MAPPING),
                        True,
                    )
                    for index, payload in enumerate(attempt_payloads)
                ],
            ],
            expected=expected_access,
            blockers=blockers,
            assignment_uid=assignment_uid,
            field="access_flags",
            legacy_access_normalization_authorized=(
                legacy_access_normalization_authorized
            ),
        )
        claim_flags = _consistent_mapping(
            values=[
                ("terminal", terminal.get("claim_flags", _MISSING_MAPPING), True),
                (
                    "provenance",
                    provenance.get("claim_flags", _MISSING_MAPPING),
                    False,
                ),
                (
                    "evaluation",
                    evaluation.get("claim_flags", _MISSING_MAPPING),
                    True,
                ),
                *[
                    (
                        f"attempt {index + 1}",
                        payload.get("claim_flags", _MISSING_MAPPING),
                        True,
                    )
                    for index, payload in enumerate(attempt_payloads)
                ],
            ],
            expected=EXPECTED_CLAIM_BOUNDARY,
            blockers=blockers,
            assignment_uid=assignment_uid,
            field="claim_flags",
        )

        training_commit = _assignment_commit(
            role="training",
            sources=(
                ("terminal", terminal.get("training_code_commit")),
                ("provenance", provenance.get("training_code_commit") or provenance.get("git_commit")),
                ("evaluation", evaluation.get("training_code_commit")),
                ("reconciliation", reconciliation.get("training_code_commit")),
            ),
            blockers=blockers,
            assignment_uid=assignment_uid,
        )
        evaluation_commit = _assignment_commit(
            role="evaluation",
            sources=(("evaluation", evaluation.get("evaluation_code_commit")),),
            blockers=blockers,
            assignment_uid=assignment_uid,
        )
        reconciliation_commit = _assignment_commit(
            role="reconciliation",
            sources=(
                (
                    "reconciliation",
                    reconciliation.get("training_code_commit"),
                ),
            ),
            blockers=blockers,
            assignment_uid=assignment_uid,
        )
        for role, commit in (
            ("training", training_commit),
            ("evaluation", evaluation_commit),
            ("reconciliation", reconciliation_commit),
        ):
            if commit:
                assignment_commit_groups[role].add(commit)

        attempt_rows.sort(key=lambda row: (int(row["ordinal"]), str(row["attempt_uid"])))
        _apply_attempt_provenance_dispositions(
            assignment_uid=assignment_uid,
            terminal_attempt_uid=terminal_attempt_uid,
            attempt_rows=attempt_rows,
            blockers=blockers,
        )

        final_evidence_attempt_references = _final_evidence_attempt_references(
            payloads=payloads,
            checkpoint_evidence=checkpoint_evidence,
        )
        assignment_row = {
            "assignment_uid": assignment_uid,
            "method_id": method_id,
            "seed": seed,
            "terminal_status": str(terminal.get("status") or ""),
            "terminal_attempt_uid": terminal_attempt_uid,
            "evidence": evidence,
            "checkpoint": checkpoint,
            "attempts": attempt_rows,
            "final_evidence_attempt_references": (
                final_evidence_attempt_references
            ),
            "attempt_retention": {
                "source_attempt_count": len(attempt_rows),
                "source_failed_count": sum(row["status"] == "FAILED" for row in attempt_rows),
                "source_cancelled_count": sum(row["status"] == "CANCELLED" for row in attempt_rows),
                "source_recovered_count": sum(
                    row["status"] == "RECOVERED"
                    or row["recovered_from_attempt_uid"] is not None
                    for row in attempt_rows
                ),
                "retention_complete": True,
            },
            "access_flags": access_flags,
            "claim_flags": claim_flags,
            "code_commits": {
                "training": training_commit,
                "evaluation": evaluation_commit,
                "reconciliation": reconciliation_commit,
            },
        }
        assignment_rows.append(assignment_row)
        assignment_sources.append(
            {
                "output_root": _repo_relative(root, output_root),
                "runtime_contract": dict(manifest_assignment.get("runtime_contract", {})),
                "failure_evidence_required": any(
                    str(row.get("status") or "") in FAILURE_ATTEMPT_STATUSES
                    for row in attempt_rows
                ),
                "checkpoint_required": checkpoint_required,
                "provenance": provenance,
                "reconciliation": reconciliation,
            }
        )

    dependency_rows: list[dict[str, object]] = []
    for relative, expected_hashes in sorted(
        _dependency_candidates(
            root=root,
            protocol=protocol,
            assignment_sources=assignment_sources,
        ).items()
    ):
        if len(expected_hashes) > 1:
            _append_blocker(
                blockers,
                "dependency_identity_conflict",
                f"Dependency {relative} has conflicting declared SHA-256 identities.",
            )
        dependency_rows.append(
            _source_identity(
                root=root,
                path=root / relative,
                committed_paths=committed_paths,
                changed_paths=changed_paths,
                blockers=blockers,
                label=f"dependency {relative}",
                cache=source_cache,
                expected_sha256=next(iter(sorted(expected_hashes)), None),
            )
        )

    access_flags = {
        key: int(value)
        for key, value in dict(protocol.get("access_counters", {})).items()
    }
    claim_flags = dict(EXPECTED_CLAIM_BOUNDARY)
    code_commits: dict[str, list[str]] = {}
    for group in REQUIRED_RECEIPT_GROUPS:
        values = set(receipt_commits[group])
        values.update(assignment_commit_groups.get(group, set()))
        code_commits[group] = sorted(values)

    return _finalise_ledger(
        protocol=protocol,
        protocol_identity=protocol_identity,
        assignments=assignment_rows,
        receipt_groups=receipt_groups,
        access_flags=access_flags,
        claim_flags=claim_flags,
        code_commits=code_commits,
        dependencies=dependency_rows,
        collector_blockers=blockers,
    )


def _training_evidence_ledger_summary(
    *,
    ledger: Mapping[str, object],
    protocol: Mapping[str, object],
    output_path: Path,
    root: Path,
) -> dict[str, object]:
    validation_blockers = validate_operational_twin_training_evidence_ledger(
        ledger,
        protocol,
    )
    reported_blockers = ledger.get("validation_blockers", [])
    blocker_rows = {
        (str(row.get("code", "")), str(row.get("message", "")))
        for row in reported_blockers
        if isinstance(row, Mapping)
    }
    blocker_rows.update(
        (str(row.get("code", "")), str(row.get("message", "")))
        for row in validation_blockers
        if row.get("code") != "reported_validation_blockers"
    )
    return {
        "path": _repo_relative(root, output_path),
        "ledger_uid": ledger.get("ledger_uid"),
        "ledger_sha256": ledger.get("ledger_sha256"),
        "ledger_status": ledger.get("status"),
        "validation_status": "PASS" if not validation_blockers else "BLOCKED",
        "validation_blocker_count": len(blocker_rows),
        "validation_blocker_codes": sorted(
            {code for code, _ in blocker_rows if code}
        ),
    }


def build_candidate(
    *,
    root: Path = REPO_ROOT,
    training_evidence_ledger: Mapping[str, object] | None = None,
    training_evidence_ledger_path: Path | None = None,
) -> dict[str, object]:
    root = root.resolve()
    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    protocol = dict(config["operational_twin_training_protocol"])
    method_protocols = {
        str(row["method_id"]): dict(row)
        for row in protocol["methods"]
    }
    roots = dict(protocol["output_roots"])
    manifest = json.loads(repo_path(root, roots["assignment_manifest"], must_exist=True).read_text(encoding="utf-8"))
    matrix: dict[str, dict[str, str]] = {method_id: {} for method_id in REQUIRED_METHOD_IDS}
    retained_failures = []
    rows = []
    world_family_metrics: dict[str, dict[str, list[float]]] = {}
    calibration_summary = []
    development_pair_rows: list[dict[str, object]] = []
    evaluation_summaries: list[dict[str, object]] = []
    evaluation_metrics_by_method: dict[str, dict[int, dict[str, object]]] = {}
    convergence_summary: list[dict[str, object]] = []
    training_commits: set[str] = set()
    evaluation_commits: set[str] = set()
    for row in manifest["assignments"]:
        method_id = str(row["method_id"])
        seed = int(row["seed"])
        assignment_uid = str(row["assignment_uid"])
        output_root = repo_path(root, row["output_root"], must_exist=False)
        status_path = output_root / "terminal_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {"status": "MISSING"}
        provenance_path = output_root / "provenance/runtime_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
        runtime_blockers = validate_assignment_runtime_evidence(row, status, provenance)
        metric = _latest_metric(output_root, method_id)
        development_metric_rows = _read_jsonl(output_root / "metrics/dev_metrics.jsonl")
        calibration_rows = _read_jsonl(output_root / "metrics/calibration_metrics.jsonl")
        calibration_metric = calibration_rows[-1] if calibration_rows else None
        non_finite = _non_finite_paths(metric) if metric else []
        if calibration_metric:
            non_finite.extend(_non_finite_paths(calibration_metric, "calibration"))
        if non_finite:
            runtime_blockers.append(
                {
                    "code": "non_finite_metric",
                    "message": f"{assignment_uid} has non-finite persisted metrics at {non_finite[:5]}.",
                }
            )
        if status.get("status") == "COMPLETED":
            expected_method = method_protocols[method_id]
            expected_budget = dict(expected_method["budget"])
            if status.get("budget") != expected_budget:
                runtime_blockers.append(
                    {
                        "code": "assignment_budget_mismatch",
                        "message": f"{assignment_uid} did not retain the frozen method budget.",
                    }
                )
            budget_evidence = training_budget_evidence(
                status,
                expected_budget,
                development_metric_rows,
            )
            if budget_evidence["full_budget_completed"] is not True:
                runtime_blockers.append(
                    {
                        "code": "assignment_training_budget_incomplete",
                        "message": f"{assignment_uid} did not exhaust either frozen epochs or max steps.",
                    }
                )
            training_commit = str(provenance.get("training_code_commit") or provenance.get("git_commit") or "")
            if len(training_commit) != 40:
                runtime_blockers.append(
                    {
                        "code": "training_code_commit_missing",
                        "message": f"{assignment_uid} lacks a full training Git commit.",
                    }
                )
            else:
                training_commits.add(training_commit)
            if bool(status.get("calibration_fitting_executed")) != bool(expected_method["calibration_eligible"]):
                runtime_blockers.append(
                    {
                        "code": "calibration_execution_mismatch",
                        "message": f"{assignment_uid} calibration execution does not match method eligibility.",
                    }
                )
            if bool(expected_method["calibration_eligible"]):
                try:
                    expected_authority = require_calibration_execution_authority(
                        config,
                        root,
                        root / "Output/audits/operational_twin_calibration_execution_authority.json",
                        method_id=method_id,
                    )
                except OperationalTwinProtocolError as error:
                    runtime_blockers.append(
                        {"code": "calibration_execution_authority_invalid", "message": str(error)}
                    )
                else:
                    if provenance.get("calibration_execution_authority") != expected_authority:
                        runtime_blockers.append(
                            {
                                "code": "calibration_execution_authority_mismatch",
                                "message": f"{assignment_uid} did not bind the approved calibration execution authority.",
                            }
                        )
            evaluation_root = output_root / "evaluation"
            evaluation_summary_path = evaluation_root / "summary.json"
            if not evaluation_summary_path.is_file():
                evaluation_summary = {}
                runtime_blockers.append(
                    {
                        "code": "development_evaluation_missing",
                        "message": f"{assignment_uid} lacks post-training per-world evaluation.",
                    }
                )
            else:
                evaluation_summary = json.loads(
                    evaluation_summary_path.read_text(encoding="utf-8")
                )
                assignment_development_pairs = _read_jsonl(
                    evaluation_root / "development_pair_results.jsonl"
                )
                evaluation_blockers = validate_assignment_development_evidence(
                    assignment=row,
                    summary=evaluation_summary,
                    development_world_rows=_read_jsonl(
                        evaluation_root / "development_world_results.jsonl"
                    ),
                    development_pair_rows=assignment_development_pairs,
                    calibration_world_rows=_read_jsonl(
                        evaluation_root / "calibration_world_results.jsonl"
                    ),
                    calibration_pair_rows=_read_jsonl(
                        evaluation_root / "calibration_pair_results.jsonl"
                    ),
                    support_rows=_read_jsonl(
                        evaluation_root / "support_abstention_results.jsonl"
                    ),
                )
                runtime_blockers.extend(evaluation_blockers)
                if not evaluation_blockers:
                    development_pair_rows.extend(assignment_development_pairs)
                    evaluation_summaries.append(evaluation_summary)
                    evaluation_metrics_by_method.setdefault(method_id, {})[seed] = dict(
                        dict(evaluation_summary["roles"])["development"]
                    )
                    evaluation_commit = str(evaluation_summary.get("evaluation_code_commit") or "")
                    if len(evaluation_commit) == 40:
                        evaluation_commits.add(evaluation_commit)
        effective_status = str(status.get("status"))
        if effective_status == "COMPLETED" and runtime_blockers:
            effective_status = "DISQUALIFIED_RESOURCE_EVIDENCE"
        matrix[method_id][str(seed)] = effective_status
        if effective_status != "COMPLETED":
            retained_failures.append({"assignment_uid": assignment_uid, "status": status, "runtime_evidence_blockers": runtime_blockers})
        if metric and "development_metrics" in metric:
            metric = dict(metric["development_metrics"])
        if metric and effective_status == "COMPLETED":
            for family, family_metric in dict(metric.get("family_metrics", {})).items():
                world_family_metrics.setdefault(method_id, {}).setdefault(family, []).append(float(dict(family_metric).get("reconstruction_rmse", float("nan"))))
        if (
            calibration_metric
            and effective_status == "COMPLETED"
            and bool(method_protocols[method_id]["calibration_eligible"])
        ):
            calibration_summary.append({"assignment_uid": assignment_uid, "method_id": method_id, "calibration_metrics": calibration_metric})
        rows.append({
            "assignment_uid": assignment_uid,
            "method_id": method_id,
            "seed": seed,
            "status": effective_status,
            "raw_status": status.get("status"),
            "output_root": row["output_root"],
            "attempt_count": len(list((output_root / "attempts").glob("*.json"))) if output_root.exists() else 0,
            "epochs_completed": status.get("epochs_completed"),
            "steps_completed": status.get("steps_completed"),
            "trainable_parameter_count": status.get("trainable_parameter_count"),
            "scheduler_identity": status.get("scheduler_identity"),
            "runtime_evidence_blockers": runtime_blockers,
            "budget_completion_evidence": budget_evidence if status.get("status") == "COMPLETED" else None,
            "evaluation_status": evaluation_summary.get("status") if status.get("status") == "COMPLETED" else None,
        })
        if status.get("status") == "COMPLETED":
            first_metric = development_metric_rows[0] if development_metric_rows else metric
            last_metric = development_metric_rows[-1] if development_metric_rows else metric
            convergence_summary.append(
                {
                    "assignment_uid": assignment_uid,
                    "method_id": method_id,
                    "seed": seed,
                    "budget_completion_evidence": budget_evidence,
                    "first_development_reconstruction_rmse": (
                        first_metric.get("reconstruction_rmse") if first_metric else None
                    ),
                    "last_development_reconstruction_rmse": (
                        last_metric.get("reconstruction_rmse") if last_metric else None
                    ),
                    "selected_development_reconstruction_rmse": (
                        metric.get("reconstruction_rmse") if metric else None
                    ),
                }
            )
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    if len(training_commits) != 1:
        retained_failures.append(
            {
                "assignment_uid": "aggregate",
                "status": "INVALID",
                "runtime_evidence_blockers": [
                    {
                        "code": "training_code_commit_set_invalid",
                        "message": f"Expected one training code commit, observed {sorted(training_commits)}.",
                    }
                ],
            }
        )
    if len(evaluation_commits) != 1:
        retained_failures.append(
            {
                "assignment_uid": "aggregate",
                "status": "INVALID",
                "runtime_evidence_blockers": [
                    {
                        "code": "evaluation_code_commit_set_invalid",
                        "message": f"Expected one evaluation code commit, observed {sorted(evaluation_commits)}.",
                    }
                ],
            }
        )
    statistical_policy = dict(protocol["statistics"])
    analysis_seed = int(
        hashlib.sha256(
            f"{protocol['protocol_uid']}|development_statistics".encode()
        ).hexdigest()[:8],
        16,
    )
    statistical_policy["analysis_seed"] = analysis_seed
    paired_statistics = (
        build_development_comparisons(development_pair_rows, statistical_policy)
        if len(evaluation_summaries) == 15
        else {
            "status": "UNAVAILABLE",
            "reason": "all_assignment_development_evaluations_required",
        }
    )
    metric_seed_stability = {}
    for method_id, by_seed in sorted(evaluation_metrics_by_method.items()):
        method_metrics: dict[str, object] = {}
        for metric_name in (
            "trajectory_rmse",
            "mean_action_contrast_error",
            "mean_intervention_ranking_regret",
            "effect_direction_accuracy",
            "mean_gaussian_nll",
            "coverage_95",
        ):
            values = {
                str(seed): metrics.get(metric_name)
                for seed, metrics in sorted(by_seed.items())
            }
            defined = [float(value) for value in values.values() if value is not None]
            method_metrics[metric_name] = {
                "values_by_seed": values,
                "defined_seed_count": len(defined),
                "mean": statistics.fmean(defined) if defined else None,
                "sample_standard_deviation": (
                    statistics.stdev(defined) if len(defined) > 1 else 0.0 if defined else None
                ),
                "minimum": min(defined) if defined else None,
                "maximum": max(defined) if defined else None,
            }
        metric_seed_stability[method_id] = method_metrics
    support_summaries = [dict(summary["support_abstention"]) for summary in evaluation_summaries]
    support_ood_summary = {
        "assignment_count": len(support_summaries),
        "all_status_accuracy_one": bool(support_summaries)
        and all(float(row["status_accuracy"]) == 1.0 for row in support_summaries),
        "all_abstention_compliance_one": bool(support_summaries)
        and all(float(row["abstention_compliance"]) == 1.0 for row in support_summaries),
        "learned_ood_score_available": False,
        "support_policy_source": "deterministic_world_action_support_contract",
        "limitation": "No learned OOD score is claimed by the development package.",
    }
    evaluation_scheduler_receipt_path = (
        root
        / "Output/results/operational_twin/development_training/resource_receipts/development_evaluation_sacct.txt"
    )
    if not evaluation_scheduler_receipt_path.is_file():
        retained_failures.append(
            {
                "assignment_uid": "aggregate",
                "status": "INVALID",
                "runtime_evidence_blockers": [
                    {
                        "code": "evaluation_sacct_receipt_missing",
                        "message": "Development evaluation sacct receipt is missing.",
                    }
                ],
            }
        )
        evaluation_scheduler_receipt = None
    else:
        evaluation_scheduler_receipt = {
            "path": evaluation_scheduler_receipt_path.relative_to(root).as_posix(),
            "sha256": _sha256_file(evaluation_scheduler_receipt_path),
        }
    ledger_summary = None
    if training_evidence_ledger is not None:
        ledger_summary = _training_evidence_ledger_summary(
            ledger=training_evidence_ledger,
            protocol=protocol,
            output_path=(
                training_evidence_ledger_path
                if training_evidence_ledger_path is not None
                else root / DEFAULT_TRAINING_EVIDENCE_LEDGER
            ),
            root=root,
        )
    candidate_complete = (
        len(completed) == 15
        and len(evaluation_summaries) == 15
        and not retained_failures
        and ledger_summary is not None
        and ledger_summary["validation_status"] == "PASS"
    )
    payload = {
        "schema_version": "neuronsifter-operational-twin-sufficient-training-candidate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregation_code_commit": _git_commit(root),
        "status": "COMPLETE_CANDIDATE" if candidate_complete else "INCOMPLETE_CANDIDATE",
        "candidate_complete": candidate_complete,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "sealed_material_access": 0,
        "protected_holdout_access": 0,
        "six_context_outcome_access": 0,
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "assignment_count": len(rows),
        "completed_assignment_count": len(completed),
        "terminal_matrix": matrix,
        "assignments": rows,
        "retained_failures": retained_failures,
        "family_stratified_reconstruction_rmse": {
            method_id: {
                family: {"mean": statistics.fmean(values), "count": len(values)}
                for family, values in sorted(by_family.items())
                if values
            }
            for method_id, by_family in sorted(world_family_metrics.items())
        },
        "seed_stability": {
            method_id: {
                "completed_seeds": sorted(int(seed) for seed, status in seeds.items() if status == "COMPLETED"),
                "status_values": dict(sorted(seeds.items())),
            }
            for method_id, seeds in sorted(matrix.items())
        },
        "metric_seed_stability": metric_seed_stability,
        "support_ood_abstention_summary": support_ood_summary,
        "calibration_world_only_uq_summary": calibration_summary,
        "development_paired_statistics": paired_statistics,
        "development_evaluation_summaries": evaluation_summaries,
        "convergence_summary": convergence_summary,
        "training_code_commits": sorted(training_commits),
        "evaluation_code_commits": sorted(evaluation_commits),
        "development_selection": "best_dev_checkpoint_by_development_reconstruction_rmse",
        "statistical_plan": statistical_policy,
        "scheduler_resource_index": roots.get("resource_receipt"),
        "evaluation_scheduler_receipt": evaluation_scheduler_receipt,
        "training_evidence_ledger": ledger_summary,
    }
    payload["content_identity"] = _canonical_hash(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    return payload


def render_candidate_report(payload: Mapping[str, object]) -> str:
    ledger = payload.get("training_evidence_ledger")
    ledger_lines: list[str] = []
    if isinstance(ledger, Mapping):
        ledger_lines = [
            f"- Training evidence ledger: `{ledger.get('path')}`",
            f"- Ledger identity: `{ledger.get('ledger_sha256')}`",
            f"- Ledger validation: `{ledger.get('validation_status')}` "
            f"({ledger.get('validation_blocker_count')} blockers)",
        ]
    lines = [
        "# Operational Twin Sufficient-Training Candidate",
        "",
        f"- Status: `{payload['status']}`",
        f"- Completed assignments: `{payload['completed_assignment_count']}/{payload['assignment_count']}`",
        f"- Training code commit: `{', '.join(payload['training_code_commits'])}`",
        f"- Evaluation code commit: `{', '.join(payload['evaluation_code_commits'])}`",
        f"- Aggregation code commit: `{payload['aggregation_code_commit']}`",
        *ledger_lines,
        "- `sufficient_training=false`; this artifact is a candidate for independent review only.",
        "- `twin_qualification_pass=false`; the sealed Twin partition remains unopened.",
        "",
        "## Development metrics across three fixed seeds",
        "",
        "| Method | Trajectory RMSE mean | Action-contrast error mean | Ranking regret mean | Effect-direction accuracy mean | Coverage 95 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id, metrics in sorted(dict(payload["metric_seed_stability"]).items()):
        metric_map = dict(metrics)
        values = []
        for name in (
            "trajectory_rmse",
            "mean_action_contrast_error",
            "mean_intervention_ranking_regret",
            "effect_direction_accuracy",
            "coverage_95",
        ):
            value = dict(metric_map[name]).get("mean")
            values.append("NA" if value is None else f"{float(value):.6f}")
        lines.append(f"| `{method_id}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Paired development comparisons",
            "",
            "Positive estimates favor the full uncertainty-aware action-conditioned Twin.",
            "",
            "| Baseline | Metric | Estimate | 95% CI | Holm-adjusted p | Worlds |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    comparisons = dict(payload["development_paired_statistics"]).get("comparisons", [])
    for comparison in comparisons:
        if comparison.get("status") != "DEFINED":
            continue
        interval = dict(comparison["family_stratified_bootstrap_ci95"])
        lines.append(
            "| "
            f"`{comparison['baseline_method_id']}` | `{comparison['metric']}` | "
            f"{float(interval['estimate']):.6f} | "
            f"[{float(interval['lower']):.6f}, {float(interval['upper']):.6f}] | "
            f"{float(comparison['holm_adjusted_p_value']):.6g} | "
            f"{comparison['world_count']} |"
        )
    support = dict(payload["support_ood_abstention_summary"])
    lines.extend(
        [
            "",
            "## Qualification boundary",
            "",
            f"- Deterministic support classification exact across assignments: `{support['all_status_accuracy_one']}`.",
            f"- Support-driven abstention compliance exact across assignments: `{support['all_abstention_compliance_one']}`.",
            "- Learned OOD score is not claimed; this remains an explicit limitation for independent review.",
            "- Development/calibration evidence is controlled-world evidence, not empirical therapeutic evidence.",
            "- The full Twin is better than the same-backbone no-action ablation on development ranking regret and effect-direction error, but worse on trajectory RMSE; this mixed result must remain visible during pre-access review.",
            "- No sealed/test, protected-holdout, six-context, Tool, MCP, AgentTrace, Agent, Memory, or Co-Evolution surface was opened.",
            "",
            f"Content identity: `{payload['content_identity']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "Output/audits/operational_twin_sufficient_training_candidate.json")
    parser.add_argument(
        "--training-evidence-ledger-output",
        type=Path,
        default=REPO_ROOT / DEFAULT_TRAINING_EVIDENCE_LEDGER,
        help=(
            "Semantic output path for the compact training evidence ledger. "
            "The ledger is written before candidate aggregation."
        ),
    )
    args = parser.parse_args()
    ledger_output = (
        args.training_evidence_ledger_output
        if args.training_evidence_ledger_output.is_absolute()
        else REPO_ROOT / args.training_evidence_ledger_output
    )
    ledger = collect_operational_twin_training_evidence_ledger(root=REPO_ROOT)
    ledger_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = build_candidate(
        root=REPO_ROOT,
        training_evidence_ledger=ledger,
        training_evidence_ledger_path=ledger_output,
    )
    catalog_root = REPO_ROOT / "Output/results/operational_twin/development_training/catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)
    (catalog_root / "sufficient_training_candidate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = REPO_ROOT / "Output/reports/operational_twin_sufficient_training_candidate.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_candidate_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "completed_assignment_count": payload["completed_assignment_count"],
                "output": _repo_relative(REPO_ROOT, args.output),
                "training_evidence_ledger_output": _repo_relative(
                    REPO_ROOT,
                    ledger_output,
                ),
                "training_evidence_ledger_status": ledger["status"],
                "training_evidence_ledger_sha256": ledger["ledger_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["candidate_complete"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
