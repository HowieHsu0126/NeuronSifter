"""Refresh deterministic Slurm allocation receipts for Operational Twin runs."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ASSIGNMENT_MANIFEST = Path(
    "Output/results/operational_twin/training_protocol/assignment_manifest.json"
)
DEFAULT_TRAINING_RECEIPT = Path(
    "Output/results/operational_twin/development_training/"
    "resource_receipts/formal_attempts_round37.sacct"
)
DEFAULT_EVALUATION_RECEIPT = Path(
    "Output/results/operational_twin/development_training/"
    "resource_receipts/development_evaluation_sacct.txt"
)
EXPECTED_EVALUATION_ASSIGNMENTS = 15
SACCT_FIELDS = (
    "JobID",
    "JobName",
    "Partition",
    "NodeList",
    "State",
    "ExitCode",
    "Elapsed",
    "AllocTRES",
    "ReqTRES",
    "ReqMem",
    "AllocCPUS",
    "Start",
    "End",
    "WorkDir",
    "SubmitLine",
)

_JOB_ID_PATTERN = re.compile(r"[1-9][0-9]*(?:_[0-9]+)?")
_SacctRunner = Callable[..., subprocess.CompletedProcess[str]]


class ReceiptRefreshError(ValueError):
    """Raised when scheduler receipt evidence is incomplete or ambiguous."""


def _read_json_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptRefreshError(f"Unable to read {label} at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReceiptRefreshError(f"{label} at {path} must contain a JSON object.")
    return dict(value)


def _repo_path(repo_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReceiptRefreshError(f"{label} must be a non-empty path string.")
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else repo_root / candidate
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ReceiptRefreshError(
            f"{label} escapes the repository root: {value}"
        ) from exc
    return resolved_path


def _load_assignments(
    *, repo_root: Path, assignment_manifest: Path
) -> list[dict[str, object]]:
    manifest = _read_json_mapping(assignment_manifest, label="assignment manifest")
    values = manifest.get("assignments")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ReceiptRefreshError("Assignment manifest must contain an assignments list.")
    assignments: list[dict[str, object]] = []
    assignment_uids: set[str] = set()
    output_roots: set[Path] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ReceiptRefreshError(f"Assignment {index} is not a JSON object.")
        assignment = dict(value)
        assignment_uid = assignment.get("assignment_uid")
        if not isinstance(assignment_uid, str) or not assignment_uid:
            raise ReceiptRefreshError(f"Assignment {index} lacks assignment_uid.")
        output_root = _repo_path(
            repo_root,
            assignment.get("output_root"),
            label=f"{assignment_uid} output_root",
        )
        if assignment_uid in assignment_uids:
            raise ReceiptRefreshError(
                f"Assignment manifest repeats assignment_uid {assignment_uid}."
            )
        if output_root in output_roots:
            raise ReceiptRefreshError(
                f"Assignment manifest repeats output_root {output_root}."
            )
        assignment_uids.add(assignment_uid)
        output_roots.add(output_root)
        assignment["resolved_output_root"] = output_root
        assignments.append(assignment)
    if not assignments:
        raise ReceiptRefreshError("Assignment manifest contains no assignments.")
    return assignments


def _identity_part(value: object, *, label: str) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReceiptRefreshError(f"{label} must be a string or integer.")
    text = str(value)
    if text != text.strip():
        raise ReceiptRefreshError(f"{label} contains surrounding whitespace.")
    return text


def _exact_job_id(
    scheduler: Mapping[str, object], *, context: str, allow_empty: bool
) -> str | None:
    direct_job_id = _identity_part(
        scheduler.get("slurm_job_id"), label=f"{context} slurm_job_id"
    )
    array_job_id = _identity_part(
        scheduler.get("slurm_array_job_id"),
        label=f"{context} slurm_array_job_id",
    )
    array_task_id = _identity_part(
        scheduler.get("slurm_array_task_id"),
        label=f"{context} slurm_array_task_id",
    )
    if bool(array_job_id) != bool(array_task_id):
        raise ReceiptRefreshError(
            f"{context} must provide both Slurm array job and task identities."
        )
    exact_job_id = (
        f"{array_job_id}_{array_task_id}" if array_job_id else direct_job_id
    )
    if not exact_job_id:
        if allow_empty:
            return None
        raise ReceiptRefreshError(f"{context} lacks a scheduled Slurm identity.")
    if _JOB_ID_PATTERN.fullmatch(exact_job_id) is None:
        raise ReceiptRefreshError(
            f"{context} has an invalid exact Slurm identity: {exact_job_id!r}."
        )
    return exact_job_id


def _scheduler_mapping(
    payload: Mapping[str, object], *, keys: Sequence[str], context: str
) -> Mapping[str, object]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ReceiptRefreshError(f"{context} {key} must be a JSON object.")
        return value
    return {}


def _job_id_sort_key(job_id: str) -> tuple[int, int, str]:
    base, separator, task = job_id.partition("_")
    return (int(base), int(task) if separator else -1, job_id)


def collect_training_job_ids(
    *, repo_root: Path, assignment_manifest: Path
) -> tuple[str, ...]:
    """Collect every scheduled identity from retained assignment attempt JSON."""

    identities: dict[str, Path] = {}
    for assignment in _load_assignments(
        repo_root=repo_root, assignment_manifest=assignment_manifest
    ):
        assignment_uid = str(assignment["assignment_uid"])
        output_root = Path(assignment["resolved_output_root"])
        attempts_dir = output_root / "attempts"
        attempt_paths = sorted(
            attempts_dir.glob("*__attempt_*.json"), key=lambda path: path.name
        )
        if not attempt_paths:
            raise ReceiptRefreshError(
                f"{assignment_uid} has no retained attempt JSON files."
            )
        for attempt_path in attempt_paths:
            payload = _read_json_mapping(
                attempt_path, label=f"{assignment_uid} attempt"
            )
            attempt_uid = payload.get("attempt_uid")
            if not isinstance(attempt_uid, str) or not attempt_uid:
                raise ReceiptRefreshError(f"{attempt_path} lacks attempt_uid.")
            if attempt_uid != attempt_path.stem:
                raise ReceiptRefreshError(
                    f"{attempt_path} filename and attempt_uid disagree."
                )
            if payload.get("assignment_uid") != assignment_uid:
                raise ReceiptRefreshError(
                    f"{attempt_path} is outside its declared assignment root."
                )
            scheduler = _scheduler_mapping(
                payload,
                keys=("scheduler_identity", "scheduler"),
                context=attempt_uid,
            )
            job_id = _exact_job_id(
                scheduler, context=attempt_uid, allow_empty=True
            )
            if job_id is None:
                continue
            previous = identities.get(job_id)
            if previous is not None:
                raise ReceiptRefreshError(
                    f"Training Slurm identity {job_id} is repeated by "
                    f"{previous} and {attempt_path}."
                )
            identities[job_id] = attempt_path
    if not identities:
        raise ReceiptRefreshError("No scheduled training attempt identities were found.")
    return tuple(sorted(identities, key=_job_id_sort_key))


def _receipt_allocation_job_ids(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReceiptRefreshError(
            f"Unable to read current evaluation receipt at {path}: {exc}"
        ) from exc
    if "\r" in text:
        raise ReceiptRefreshError(
            f"Current evaluation receipt {path} contains carriage returns."
        )
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or not lines[0]:
        raise ReceiptRefreshError(f"Current evaluation receipt {path} is empty.")
    header = lines[0].split("|")
    if "JobID" not in header:
        raise ReceiptRefreshError(
            f"Current evaluation receipt {path} lacks a JobID column."
        )
    job_id_index = header.index("JobID")
    allocation_ids: list[str] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            raise ReceiptRefreshError(
                f"Current evaluation receipt {path} has a blank row at line "
                f"{line_number}."
            )
        values = line.split("|")
        if len(values) != len(header):
            raise ReceiptRefreshError(
                f"Current evaluation receipt {path} has a malformed row at line "
                f"{line_number}."
            )
        job_id = values[job_id_index]
        if "." in job_id:
            continue
        if _JOB_ID_PATTERN.fullmatch(job_id) is None:
            raise ReceiptRefreshError(
                f"Current evaluation receipt {path} has invalid allocation "
                f"JobID {job_id!r}."
            )
        allocation_ids.append(job_id)
    duplicates = sorted(
        job_id
        for job_id, count in Counter(allocation_ids).items()
        if count != 1
    )
    if duplicates:
        raise ReceiptRefreshError(
            "Current evaluation receipt repeats allocation JobIDs: "
            + ", ".join(duplicates)
        )
    return tuple(sorted(allocation_ids, key=_job_id_sort_key))


def collect_evaluation_job_ids(
    *,
    repo_root: Path,
    assignment_manifest: Path,
    current_evaluation_receipt: Path,
) -> tuple[str, ...]:
    """Collect and cross-check the exact 15 evaluation allocation identities."""

    assignments = _load_assignments(
        repo_root=repo_root, assignment_manifest=assignment_manifest
    )
    if len(assignments) != EXPECTED_EVALUATION_ASSIGNMENTS:
        raise ReceiptRefreshError(
            "Evaluation identity refresh requires exactly "
            f"{EXPECTED_EVALUATION_ASSIGNMENTS} assignments; observed "
            f"{len(assignments)}."
        )
    evidence_identities: dict[str, str] = {}
    for assignment in assignments:
        assignment_uid = str(assignment["assignment_uid"])
        output_root = Path(assignment["resolved_output_root"])
        summary_path = output_root / "evaluation/summary.json"
        summary = _read_json_mapping(
            summary_path, label=f"{assignment_uid} evaluation summary"
        )
        if summary.get("assignment_uid") != assignment_uid:
            raise ReceiptRefreshError(
                f"{summary_path} assignment_uid does not match the manifest."
            )
        scheduler = _scheduler_mapping(
            summary,
            keys=("device_provenance", "scheduler_identity"),
            context=f"{assignment_uid} evaluation summary",
        )
        job_id = _exact_job_id(
            scheduler,
            context=f"{assignment_uid} evaluation summary",
            allow_empty=False,
        )
        assert job_id is not None
        previous = evidence_identities.get(job_id)
        if previous is not None:
            raise ReceiptRefreshError(
                f"Evaluation Slurm identity {job_id} is shared by "
                f"{previous} and {assignment_uid}."
            )
        evidence_identities[job_id] = assignment_uid

    receipt_identities = _receipt_allocation_job_ids(current_evaluation_receipt)
    if len(receipt_identities) != EXPECTED_EVALUATION_ASSIGNMENTS:
        raise ReceiptRefreshError(
            "Current evaluation receipt must contain exactly "
            f"{EXPECTED_EVALUATION_ASSIGNMENTS} allocation identities; observed "
            f"{len(receipt_identities)}."
        )
    evidence_set = set(evidence_identities)
    receipt_set = set(receipt_identities)
    if evidence_set != receipt_set:
        missing = sorted(evidence_set - receipt_set, key=_job_id_sort_key)
        unexpected = sorted(receipt_set - evidence_set, key=_job_id_sort_key)
        raise ReceiptRefreshError(
            "Evaluation summary and current receipt identities disagree; "
            f"missing={missing}, unexpected={unexpected}."
        )
    return tuple(sorted(evidence_set, key=_job_id_sort_key))


def _parse_sacct_output(
    text: str, *, requested_job_ids: Sequence[str]
) -> list[dict[str, str]]:
    if "\r" in text:
        raise ReceiptRefreshError("sacct output contains carriage returns.")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ReceiptRefreshError("sacct returned no output.")
    header = lines[0].split("|")
    if tuple(header) != SACCT_FIELDS:
        raise ReceiptRefreshError(
            f"sacct header mismatch: expected {SACCT_FIELDS}, observed {tuple(header)}."
        )

    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            raise ReceiptRefreshError(f"sacct returned a blank row at line {line_number}.")
        values = line.split("|")
        if len(values) != len(SACCT_FIELDS):
            raise ReceiptRefreshError(
                "sacct returned a malformed row at line "
                f"{line_number}; a field may contain a forbidden pipe or newline."
            )
        row = dict(zip(SACCT_FIELDS, values, strict=True))
        rows.append(row)

    requested = tuple(requested_job_ids)
    requested_set = set(requested)
    observed = Counter(row["JobID"] for row in rows)
    missing = sorted(requested_set - set(observed), key=_job_id_sort_key)
    unexpected = sorted(set(observed) - requested_set, key=_job_id_sort_key)
    duplicates = sorted(
        (job_id for job_id, count in observed.items() if count != 1),
        key=_job_id_sort_key,
    )
    if missing or unexpected or duplicates or len(rows) != len(requested):
        raise ReceiptRefreshError(
            "sacct allocation rows do not match requested JobIDs exactly; "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}."
        )
    return sorted(rows, key=lambda row: _job_id_sort_key(row["JobID"]))


def _serialize_receipt(rows: Sequence[Mapping[str, str]]) -> str:
    output_lines = ["|".join(SACCT_FIELDS)]
    for row in sorted(rows, key=lambda value: _job_id_sort_key(value["JobID"])):
        values = [row[field] for field in SACCT_FIELDS]
        for field, value in zip(SACCT_FIELDS, values, strict=True):
            if any(delimiter in value for delimiter in ("|", "\n", "\r")):
                raise ReceiptRefreshError(
                    f"{row['JobID']} {field} contains a forbidden PSV delimiter."
                )
        output_lines.append("|".join(values))
    return "\n".join(output_lines) + "\n"


def _query_sacct(
    job_ids: Sequence[str],
    *,
    sacct_bin: str,
    runner: _SacctRunner | None = None,
) -> str:
    if not job_ids:
        raise ReceiptRefreshError("Refusing to query sacct without exact JobIDs.")
    command = [
        sacct_bin,
        "-X",
        f"--jobs={','.join(job_ids)}",
        "--parsable2",
        f"--format={','.join(SACCT_FIELDS)}",
    ]
    execute = runner if runner is not None else subprocess.run
    completed = execute(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise ReceiptRefreshError(
            f"sacct failed with exit {completed.returncode}: {stderr or 'no stderr'}"
        )
    rows = _parse_sacct_output(
        completed.stdout, requested_job_ids=tuple(job_ids)
    )
    return _serialize_receipt(rows)


def _write_atomic(path: Path, text: str) -> None:
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
            handle.write(text)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def refresh_scheduler_receipts(
    *,
    repo_root: Path,
    assignment_manifest: Path,
    current_evaluation_receipt: Path,
    training_output: Path,
    evaluation_output: Path,
    sacct_bin: str = "sacct",
    runner: _SacctRunner | None = None,
) -> dict[str, object]:
    """Query exact allocations and write deterministic training/evaluation PSV."""

    training_job_ids = collect_training_job_ids(
        repo_root=repo_root, assignment_manifest=assignment_manifest
    )
    evaluation_job_ids = collect_evaluation_job_ids(
        repo_root=repo_root,
        assignment_manifest=assignment_manifest,
        current_evaluation_receipt=current_evaluation_receipt,
    )
    training_text = _query_sacct(
        training_job_ids, sacct_bin=sacct_bin, runner=runner
    )
    evaluation_text = _query_sacct(
        evaluation_job_ids, sacct_bin=sacct_bin, runner=runner
    )
    _write_atomic(training_output, training_text)
    _write_atomic(evaluation_output, evaluation_text)
    return {
        "status": "PASS",
        "training_job_count": len(training_job_ids),
        "evaluation_job_count": len(evaluation_job_ids),
        "training_output": str(training_output),
        "evaluation_output": str(evaluation_output),
    }


def _resolve_cli_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--assignment-manifest", type=Path, default=DEFAULT_ASSIGNMENT_MANIFEST
    )
    parser.add_argument(
        "--evaluation-identity-receipt",
        type=Path,
        default=DEFAULT_EVALUATION_RECEIPT,
    )
    parser.add_argument(
        "--training-output", type=Path, default=DEFAULT_TRAINING_RECEIPT
    )
    parser.add_argument(
        "--evaluation-output", type=Path, default=DEFAULT_EVALUATION_RECEIPT
    )
    parser.add_argument("--sacct-bin", default="sacct")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    result = refresh_scheduler_receipts(
        repo_root=repo_root,
        assignment_manifest=_resolve_cli_path(repo_root, args.assignment_manifest),
        current_evaluation_receipt=_resolve_cli_path(
            repo_root, args.evaluation_identity_receipt
        ),
        training_output=_resolve_cli_path(repo_root, args.training_output),
        evaluation_output=_resolve_cli_path(repo_root, args.evaluation_output),
        sacct_bin=args.sacct_bin,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
