"""Fail-closed storage accounting for the Public Foundation Corpus.

The storage gate is operational evidence, not a proxy for corpus quality.  A
scan reports a capacity failure only after every configured root was readable;
I/O or RPC failures remain ``pending_revalidation`` and are retained in the
ledger rather than being extrapolated from a partial byte total.
"""

from __future__ import annotations

from datetime import datetime, timezone
import errno as errno_module
import os
from pathlib import Path
from typing import Mapping, Sequence
import uuid


STORAGE_LEDGER_SCHEMA_VERSION = "neuronsifter-public-foundation-corpus-storage-ledger-v1"
STORAGE_LEDGER_PRODUCER = (
    "Libs/scripts/builders/build_public_foundation_corpus_storage_ledger.py"
)
DEFAULT_STORAGE_CAP_BYTES = 1_000_000_000_000
DEFAULT_PREVIOUS_CONFIRMED_USAGE_BYTES = 422_655_337_981
DEFAULT_STORAGE_ROOTS = {
    "raw": "Input/raw",
    "processed": "Input/processed",
    "canonical": "Input/canonical",
    "cache": "Input/cache",
    "output": "Output",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_repo_path(repo_root: Path, relative_path: str | Path) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {relative_path}")
    return candidate


def relative_repo_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def derive_storage_scan_status(
    *,
    usage_bytes: int,
    capacity_limit_bytes: int,
    partial_scan: bool,
    rpc_or_io_errors: Sequence[object],
) -> str:
    if partial_scan or rpc_or_io_errors:
        return "pending_revalidation"
    return "passed" if usage_bytes <= capacity_limit_bytes else "failed_capacity"


def storage_gate_status(ledger: Mapping[str, object]) -> str:
    """Return the evidence status admitted into G1 derivation.

    A syntactically present ledger is not sufficient for a storage pass: its
    scan and self-validation outcomes must both be explicit.
    """

    scan_status = str(ledger.get("scan_status") or ledger.get("status") or "")
    validator_status = str(ledger.get("validator_status") or "")
    if validator_status != "passed":
        return "pending_revalidation"
    if scan_status in {"passed", "failed_capacity", "pending_revalidation"}:
        return scan_status
    return "pending_revalidation"


def pending_storage_ledger(
    repo_root: Path,
    *,
    roots: Mapping[str, str] = DEFAULT_STORAGE_ROOTS,
    cap_bytes: int = DEFAULT_STORAGE_CAP_BYTES,
    exclude_relative_paths: Sequence[str] = (),
    previous_confirmed_usage_bytes: int = DEFAULT_PREVIOUS_CONFIRMED_USAGE_BYTES,
    failure_class: str = "infrastructure_scan_interrupted",
) -> dict[str, object]:
    """Materialize an explicit non-measurement state for bounded test builds."""

    started_at = utc_now()
    normalized_exclusions = sorted({str(path).replace("\\", "/") for path in exclude_relative_paths})
    error = {
        "path": "",
        "error_type": failure_class,
        "errno": None,
        "message": "No complete storage measurement is available.",
    }
    root_rows = {
        root_id: {
            "root": relative,
            "exists": False,
            "scan_complete": False,
            "file_count": 0,
            "bytes": 0,
            "excluded_file_count": 0,
            "excluded_bytes": 0,
            "skipped_symlink_count": 0,
            "skipped_non_regular_entry_count": 0,
        }
        for root_id, relative in roots.items()
    }
    return {
        "schema_version": STORAGE_LEDGER_SCHEMA_VERSION,
        "producer": STORAGE_LEDGER_PRODUCER,
        "created_at": started_at,
        "scan_uid": f"pfcs-{uuid.uuid4().hex}",
        "scan_started_at": started_at,
        "scan_completed_at": utc_now(),
        "scan_status": "pending_revalidation",
        "status": "pending_revalidation",
        "filesystem_root": repo_root.resolve().as_posix(),
        "scope_included": list(roots.values()),
        "scope_excluded": normalized_exclusions,
        "usage_bytes": 0,
        "capacity_limit_bytes": int(cap_bytes),
        "remaining_bytes": None,
        "usage_ratio": None,
        "largest_directories": [],
        "unreadable_paths": [],
        "rpc_or_io_errors": [error],
        "partial_scan": True,
        "previous_confirmed_usage_bytes": int(previous_confirmed_usage_bytes),
        "delta_from_previous": None,
        "validator_status": "passed",
        "roots": root_rows,
        "missing_roots": [],
        "total_file_count": 0,
        "decimal_byte_cap": int(cap_bytes),
        "cap_status": "pending_revalidation",
        "cap_exceeded": False,
        "measurement_exclusions": normalized_exclusions,
        "failure_class": failure_class,
        "scientific_blocker": False,
        "blocker_codes": ["storage_cap_pending_revalidation"],
        "record_count": len(root_rows),
        "measurement_scope": {
            "roots": list(roots.values()),
            "includes_output_root": "Output" in roots.values(),
            "self_reference_excluded": bool(normalized_exclusions),
        },
    }


def _error_record(repo_root: Path, path: Path, exc: OSError) -> dict[str, object]:
    try:
        relative = relative_repo_path(repo_root, path)
    except ValueError:
        relative = path.as_posix()
    error_number = exc.errno
    error_type = errno_module.errorcode.get(error_number, type(exc).__name__) if error_number else type(exc).__name__
    return {
        "path": relative,
        "error_type": error_type,
        "errno": error_number,
        "message": str(exc),
    }


def scan_storage_roots(
    repo_root: Path,
    *,
    roots: Mapping[str, str] = DEFAULT_STORAGE_ROOTS,
    cap_bytes: int = DEFAULT_STORAGE_CAP_BYTES,
    exclude_relative_paths: Sequence[str] = (),
    previous_confirmed_usage_bytes: int = DEFAULT_PREVIOUS_CONFIRMED_USAGE_BYTES,
    largest_directory_limit: int = 20,
) -> dict[str, object]:
    """Measure configured persistent roots without following symlinks.

    Missing optional roots contribute zero bytes.  Any failed directory read or
    file stat marks the whole scan partial, preserving every error detail and
    preventing capacity inference from the observed subset.
    """

    if cap_bytes <= 0:
        raise ValueError("cap_bytes must be positive")
    if largest_directory_limit <= 0:
        raise ValueError("largest_directory_limit must be positive")

    repo_root = repo_root.resolve()
    started_at = utc_now()
    normalized_exclusions = sorted({str(path).replace("\\", "/") for path in exclude_relative_paths})
    excluded = set(normalized_exclusions)
    errors: list[dict[str, object]] = []
    unreadable_paths: list[str] = []
    directory_rows: list[dict[str, object]] = []
    root_rows: dict[str, dict[str, object]] = {}
    missing_roots: list[str] = []

    def capture_error(path: Path, exc: OSError) -> None:
        errors.append(_error_record(repo_root, path, exc))
        try:
            unreadable_paths.append(relative_repo_path(repo_root, path))
        except ValueError:
            unreadable_paths.append(path.as_posix())

    def scan_directory(directory: Path) -> tuple[int, int, int, int, int, int, bool]:
        """Return included/excluded bytes/files, skipped counts, and completeness."""

        try:
            with os.scandir(directory) as handle:
                entries = sorted(handle, key=lambda entry: entry.name)
        except OSError as exc:
            capture_error(directory, exc)
            directory_rows.append(
                {
                    "path": relative_repo_path(repo_root, directory),
                    "bytes": 0,
                    "file_count": 0,
                    "scan_complete": False,
                }
            )
            return 0, 0, 0, 0, 0, 0, False

        bytes_total = 0
        file_count = 0
        excluded_bytes = 0
        excluded_file_count = 0
        skipped_symlink_count = 0
        skipped_non_regular_entry_count = 0
        complete = True
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    skipped_symlink_count += 1
                    continue
                if entry.is_dir(follow_symlinks=False):
                    child = scan_directory(path)
                    bytes_total += child[0]
                    file_count += child[1]
                    excluded_bytes += child[2]
                    excluded_file_count += child[3]
                    skipped_symlink_count += child[4]
                    skipped_non_regular_entry_count += child[5]
                    complete = complete and child[6]
                    continue
                if not entry.is_file(follow_symlinks=False):
                    skipped_non_regular_entry_count += 1
                    continue
                size = entry.stat(follow_symlinks=False).st_size
                relative_file = relative_repo_path(repo_root, path)
                if relative_file in excluded:
                    excluded_bytes += size
                    excluded_file_count += 1
                else:
                    bytes_total += size
                    file_count += 1
            except OSError as exc:
                capture_error(path, exc)
                complete = False

        directory_rows.append(
            {
                "path": relative_repo_path(repo_root, directory),
                "bytes": bytes_total,
                "file_count": file_count,
                "scan_complete": complete,
            }
        )
        return (
            bytes_total,
            file_count,
            excluded_bytes,
            excluded_file_count,
            skipped_symlink_count,
            skipped_non_regular_entry_count,
            complete,
        )

    total_bytes = 0
    total_files = 0
    for root_id, relative in roots.items():
        root = safe_repo_path(repo_root, relative)
        try:
            root.lstat()
        except FileNotFoundError:
            missing_roots.append(relative)
            root_rows[root_id] = {
                "root": relative,
                "exists": False,
                "scan_complete": True,
                "file_count": 0,
                "bytes": 0,
                "excluded_file_count": 0,
                "excluded_bytes": 0,
                "skipped_symlink_count": 0,
                "skipped_non_regular_entry_count": 0,
            }
            continue
        except OSError as exc:
            capture_error(root, exc)
            root_rows[root_id] = {
                "root": relative,
                "exists": False,
                "scan_complete": False,
                "file_count": 0,
                "bytes": 0,
                "excluded_file_count": 0,
                "excluded_bytes": 0,
                "skipped_symlink_count": 0,
                "skipped_non_regular_entry_count": 0,
            }
            continue

        try:
            root_is_directory = root.is_dir()
        except OSError as exc:
            capture_error(root, exc)
            root_rows[root_id] = {
                "root": relative,
                "exists": True,
                "scan_complete": False,
                "file_count": 0,
                "bytes": 0,
                "excluded_file_count": 0,
                "excluded_bytes": 0,
                "skipped_symlink_count": 0,
                "skipped_non_regular_entry_count": 0,
            }
            continue
        if not root_is_directory:
            exc = NotADirectoryError(errno_module.ENOTDIR, "configured storage root is not a directory", root.as_posix())
            capture_error(root, exc)
            root_rows[root_id] = {
                "root": relative,
                "exists": True,
                "scan_complete": False,
                "file_count": 0,
                "bytes": 0,
                "excluded_file_count": 0,
                "excluded_bytes": 0,
                "skipped_symlink_count": 0,
                "skipped_non_regular_entry_count": 0,
            }
            continue

        result = scan_directory(root)
        total_bytes += result[0]
        total_files += result[1]
        root_rows[root_id] = {
            "root": relative,
            "exists": True,
            "scan_complete": result[6],
            "file_count": result[1],
            "bytes": result[0],
            "excluded_file_count": result[3],
            "excluded_bytes": result[2],
            "skipped_symlink_count": result[4],
            "skipped_non_regular_entry_count": result[5],
        }

    partial_scan = bool(errors)
    scan_status = derive_storage_scan_status(
        usage_bytes=total_bytes,
        capacity_limit_bytes=cap_bytes,
        partial_scan=partial_scan,
        rpc_or_io_errors=errors,
    )
    cap_status = {
        "passed": "passed",
        "failed_capacity": "not_passed",
        "pending_revalidation": "pending_revalidation",
    }[scan_status]
    blocker_codes = {
        "passed": [],
        "failed_capacity": ["storage_cap_exceeded"],
        "pending_revalidation": ["storage_cap_pending_revalidation"],
    }[scan_status]
    largest_directories = sorted(
        directory_rows,
        key=lambda row: (-int(row["bytes"]), str(row["path"])),
    )[:largest_directory_limit]
    completed_at = utc_now()
    return {
        "schema_version": STORAGE_LEDGER_SCHEMA_VERSION,
        "producer": STORAGE_LEDGER_PRODUCER,
        "created_at": completed_at,
        "scan_uid": f"pfcs-{uuid.uuid4().hex}",
        "scan_started_at": started_at,
        "scan_completed_at": completed_at,
        "scan_status": scan_status,
        "status": scan_status,
        "filesystem_root": repo_root.as_posix(),
        "scope_included": list(roots.values()),
        "scope_excluded": normalized_exclusions,
        "usage_bytes": total_bytes,
        "capacity_limit_bytes": int(cap_bytes),
        "remaining_bytes": None if partial_scan else int(cap_bytes) - total_bytes,
        "usage_ratio": None if partial_scan else total_bytes / cap_bytes,
        "largest_directories": largest_directories,
        "unreadable_paths": sorted(set(unreadable_paths)),
        "rpc_or_io_errors": errors,
        "partial_scan": partial_scan,
        "previous_confirmed_usage_bytes": int(previous_confirmed_usage_bytes),
        "delta_from_previous": (
            None
            if partial_scan
            else total_bytes - int(previous_confirmed_usage_bytes)
        ),
        "validator_status": "passed",
        "roots": root_rows,
        "missing_roots": sorted(missing_roots),
        "total_file_count": total_files,
        "decimal_byte_cap": int(cap_bytes),
        "cap_status": cap_status,
        "cap_exceeded": total_bytes > cap_bytes,
        "measurement_exclusions": normalized_exclusions,
        "failure_class": "none" if not partial_scan else "rpc_or_io_error",
        "scientific_blocker": False,
        "blocker_codes": blocker_codes,
        "record_count": len(root_rows),
        "measurement_scope": {
            "roots": list(roots.values()),
            "includes_output_root": "Output" in roots.values(),
            "self_reference_excluded": bool(normalized_exclusions),
        },
    }
