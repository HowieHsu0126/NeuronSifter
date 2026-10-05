"""Strict, read-only generated-input linkage for Round-11E clean worktrees."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "dt-round11e-processed-input-linkage-v1"
LOGICAL_ROOT = PurePosixPath("Input/processed")
INPUT_ROOT_ENV = "NRNSIFTER_PROCESSED_INPUT_ROOT"
MANIFEST_ENV = "NRNSIFTER_PROCESSED_INPUT_MANIFEST"
MANIFEST_SHA256_ENV = "NRNSIFTER_PROCESSED_INPUT_MANIFEST_SHA256"


@dataclass(frozen=True)
class LinkedProcessedInput:
    """A verified generated input that may be mounted read-only for one process."""

    logical_path: str
    physical_path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ProcessedInputLinkage:
    """Verified linkage metadata for a bounded generated-input allowlist."""

    input_root: Path
    manifest_path: Path
    manifest_sha256: str
    entries: tuple[LinkedProcessedInput, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_logical_path(value: object) -> str:
    candidate = PurePosixPath(str(value))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Generated input logical path must stay repository-relative: {candidate}")
    if candidate.parts[:2] != LOGICAL_ROOT.parts or len(candidate.parts) <= len(LOGICAL_ROOT.parts):
        raise ValueError(f"Generated input logical path must be below {LOGICAL_ROOT}: {candidate}")
    return candidate.as_posix()


def _relative_to_input_root(logical_path: str) -> PurePosixPath:
    return PurePosixPath(logical_path).relative_to(LOGICAL_ROOT)


def _checked_input_root(value: Path) -> Path:
    root = value.expanduser()
    if root.is_symlink():
        raise ValueError(f"Generated input root must not be a symlink: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"Generated input root is unavailable: {root}")
    return root.resolve()


def _checked_source_path(input_root: Path, logical_path: str) -> Path:
    relative_path = _relative_to_input_root(logical_path)
    candidate = input_root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Generated input linkage forbids symlinked source components: {logical_path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(input_root):
        raise ValueError(f"Generated input escaped the configured root: {logical_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Generated input is unavailable: {logical_path}")
    return resolved


def _entry_for_path(input_root: Path, logical_path: str) -> dict[str, object]:
    normalized = normalize_logical_path(logical_path)
    physical_path = _checked_source_path(input_root, normalized)
    return {
        "logical_path": normalized,
        "relative_path": _relative_to_input_root(normalized).as_posix(),
        "sha256": sha256_file(physical_path),
        "size_bytes": physical_path.stat().st_size,
    }


def build_processed_input_manifest(
    input_root: Path,
    logical_paths: Iterable[str],
    *,
    discovery_trace_label: str,
    discovery_trace_sha256: str,
) -> dict[str, object]:
    """Return a deterministic manifest for an observed generated-input allowlist."""

    checked_root = _checked_input_root(input_root)
    normalized_paths = sorted({normalize_logical_path(path) for path in logical_paths})
    if not normalized_paths:
        raise ValueError("Generated input manifest must contain at least one allowlisted file")
    if len(discovery_trace_sha256) != 64:
        raise ValueError("Generated input discovery trace must have a SHA-256 digest")
    entries = [_entry_for_path(checked_root, logical_path) for logical_path in normalized_paths]
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": "round11e_processed_input_linkage_manifest",
        "logical_root": LOGICAL_ROOT.as_posix(),
        "entry_count": len(entries),
        "entry_set_sha256": _canonical_sha256(entries),
        "allowlist_discovery": {
            "trace_label": discovery_trace_label,
            "trace_sha256": discovery_trace_sha256,
        },
        "entries": entries,
        "access_policy": {
            "environment_root_variable": INPUT_ROOT_ENV,
            "environment_manifest_variable": MANIFEST_ENV,
            "environment_manifest_sha256_variable": MANIFEST_SHA256_ENV,
            "external_access": "read_only_namespace_bindings",
            "forbidden": [
                "symlinked_inputs",
                "copied_inputs",
                "parent_worktree_fallback",
                "unlisted_generated_input_access",
            ],
        },
    }


def _environment_value(environment: Mapping[str, str], variable: str) -> str:
    value = str(environment.get(variable, "")).strip()
    if not value:
        raise FileNotFoundError(f"Missing required generated-input linkage environment variable: {variable}")
    return value


def _manifest_entries(payload: Mapping[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Generated input linkage manifest schema version is invalid")
    if payload.get("logical_root") != LOGICAL_ROOT.as_posix():
        raise ValueError("Generated input linkage manifest logical root is invalid")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Generated input linkage manifest has no entries")
    if payload.get("entry_count") != len(raw_entries):
        raise ValueError("Generated input linkage manifest entry count is inconsistent")
    entries: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Generated input linkage manifest entry is not a mapping")
        logical_path = normalize_logical_path(raw_entry.get("logical_path", ""))
        relative_path = _relative_to_input_root(logical_path).as_posix()
        if str(raw_entry.get("relative_path", "")) != relative_path:
            raise ValueError(f"Generated input linkage relative path is invalid: {logical_path}")
        sha256 = str(raw_entry.get("sha256", ""))
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256.lower()):
            raise ValueError(f"Generated input linkage SHA-256 is invalid: {logical_path}")
        size_bytes = raw_entry.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"Generated input linkage size is invalid: {logical_path}")
        entries.append(
            {
                "logical_path": logical_path,
                "relative_path": relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if entries != sorted(entries, key=lambda entry: str(entry["logical_path"])):
        raise ValueError("Generated input linkage entries must be sorted by logical path")
    if len({str(entry["logical_path"]) for entry in entries}) != len(entries):
        raise ValueError("Generated input linkage entries must be unique")
    if payload.get("entry_set_sha256") != _canonical_sha256(entries):
        raise ValueError("Generated input linkage entry-set hash is invalid")
    return entries


def load_processed_input_linkage(environment: Mapping[str, str] | None = None) -> ProcessedInputLinkage:
    """Load and verify an explicit environment-bound generated-input linkage."""

    resolved_environment = dict(os.environ if environment is None else environment)
    input_root = _checked_input_root(Path(_environment_value(resolved_environment, INPUT_ROOT_ENV)))
    manifest_path = Path(_environment_value(resolved_environment, MANIFEST_ENV)).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generated input linkage manifest is unavailable: {manifest_path}")
    expected_manifest_hash = _environment_value(resolved_environment, MANIFEST_SHA256_ENV)
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "Generated input linkage manifest hash mismatch: "
            f"expected {expected_manifest_hash}, observed {observed_manifest_hash}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Generated input linkage manifest must be a JSON object")
    entries = _manifest_entries(payload)
    linked_entries: list[LinkedProcessedInput] = []
    for entry in entries:
        logical_path = str(entry["logical_path"])
        physical_path = _checked_source_path(input_root, logical_path)
        observed_size = physical_path.stat().st_size
        if observed_size != entry["size_bytes"]:
            raise ValueError(
                f"Generated input size mismatch for {logical_path}: "
                f"expected {entry['size_bytes']}, observed {observed_size}"
            )
        observed_hash = sha256_file(physical_path)
        if observed_hash != entry["sha256"]:
            raise ValueError(
                f"Generated input hash mismatch for {logical_path}: "
                f"expected {entry['sha256']}, observed {observed_hash}"
            )
        linked_entries.append(
            LinkedProcessedInput(
                logical_path=logical_path,
                physical_path=physical_path,
                sha256=observed_hash,
                size_bytes=observed_size,
            )
        )
    return ProcessedInputLinkage(
        input_root=input_root,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=observed_manifest_hash,
        entries=tuple(linked_entries),
    )


def _overlay_directories(worktree_root: Path, linkage: ProcessedInputLinkage) -> list[Path]:
    overlay_root = worktree_root / Path(LOGICAL_ROOT)
    directories = {overlay_root}
    for entry in linkage.entries:
        parent = (worktree_root / entry.logical_path).parent
        while parent != overlay_root:
            directories.add(parent)
            parent = parent.parent
        directories.add(overlay_root)
    return sorted(directories, key=lambda path: (len(path.parts), path.as_posix()))


def build_bwrap_command(
    worktree_root: Path,
    linkage: ProcessedInputLinkage,
    command: Sequence[str],
) -> list[str]:
    """Build a process-local, read-only mount namespace for allowed input files."""

    if not command:
        raise ValueError("A command is required for the generated-input linkage launcher")
    if shutil.which("bwrap") is None:
        raise RuntimeError("bubblewrap is required for the generated-input linkage launcher")
    root = worktree_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Clean worktree root is invalid: {root}")
    host_input = root / "Input"
    if host_input.is_symlink():
        raise ValueError("Clean worktree Input path must not be a symlink")
    host_processed = root / Path(LOGICAL_ROOT)
    if host_processed.exists() and (host_processed.is_symlink() or any(host_processed.iterdir())):
        raise ValueError("Clean worktree Input/processed path must be absent or empty before linkage launch")
    arguments = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        str(root),
        str(root),
        "--dir",
        str(host_input),
        "--tmpfs",
        str(host_processed),
    ]
    for directory in _overlay_directories(root, linkage):
        arguments.extend(["--dir", str(directory)])
    for entry in linkage.entries:
        arguments.extend(["--ro-bind", str(entry.physical_path), str(root / entry.logical_path)])
    arguments.extend(
        [
            "--tmpfs",
            str(linkage.input_root),
            "--unsetenv",
            INPUT_ROOT_ENV,
            "--unsetenv",
            MANIFEST_ENV,
            "--unsetenv",
            MANIFEST_SHA256_ENV,
            "--unsetenv",
            "PYTHONPATH",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            str(root),
            "--",
            *command,
        ]
    )
    return arguments


def run_with_processed_input_linkage(
    worktree_root: Path,
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute one command with no writable or unlisted generated inputs visible."""

    linkage = load_processed_input_linkage(environment)
    root = worktree_root.resolve()
    host_input = root / "Input"
    host_processed = root / Path(LOGICAL_ROOT)
    created_input_directory = False
    created_processed_directory = False
    if not host_input.exists():
        host_input.mkdir()
        created_input_directory = True
    if not host_processed.exists():
        host_processed.mkdir()
        created_processed_directory = True
    try:
        return subprocess.run(
            build_bwrap_command(root, linkage, command),
            check=False,
            text=True,
            env=None if environment is None else dict(environment),
        )
    finally:
        if created_processed_directory:
            host_processed.rmdir()
        if created_input_directory:
            host_input.rmdir()


def linkage_metadata(linkage: ProcessedInputLinkage) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": linkage.manifest_sha256,
        "entry_count": len(linkage.entries),
        "logical_paths": [entry.logical_path for entry in linkage.entries],
        "access_mode": "read_only_namespace_bindings",
    }
