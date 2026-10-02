"""Strict execution-input identity and resolution for Round-11G1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


SCHEMA_VERSION = "dt-round11g-execution-input-bundle-v1"
BUNDLE_ENV = "NRNSIFTER_EXECUTION_INPUT_BUNDLE"
BUNDLE_SHA256_ENV = "NRNSIFTER_EXECUTION_INPUT_BUNDLE_SHA256"
PROCESSED_ROOT_ENV = "NRNSIFTER_PROCESSED_INPUT_ROOT"
AUDIT_ROOT_ENV = "NRNSIFTER_AUDIT_ARTIFACT_ROOT"
OUTPUT_ROOT_ENV = "NRNSIFTER_EXECUTION_OUTPUT_ROOT"
RUNTIME_ROOT = PurePosixPath("/run/neuronsifter-round11g1-inputs")
RUNTIME_OUTPUT_ROOT = PurePosixPath("/run/neuronsifter-round11g1-outputs")
ALLOWED_ACCESS_STATES = {
    "approved_internal_read_only",
    "approved_public_source_derived_internal_read_only",
}
ALLOWED_ROLES = {
    "processed_test_input",
    "read_only_parent_audit",
}


@dataclass(frozen=True)
class ExecutionInputAsset:
    asset_uid: str
    logical_role: str
    artifact_type: str
    source_uid: str
    producer_uid: str
    version: str
    content_hash: str
    size: int
    canonical_uri: str
    license_or_access_status: str
    required_by_testcases: tuple[str, ...]
    allowed_execution_role: str
    physical_path: Path


@dataclass(frozen=True)
class ExecutionInputBundle:
    bundle_uid: str
    code_commit: str
    config_hash: str
    environment_hash: str
    manifest_path: Path
    manifest_sha256: str
    assets: tuple[ExecutionInputAsset, ...]
    raw_payload: Mapping[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required_text(mapping: Mapping[str, object], key: str, context: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{context} has no {key}")
    return value


def _sha256_value(value: object, context: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} has an invalid SHA-256")
    return digest


def _repo_uri_path(value: object) -> PurePosixPath:
    uri = str(value).strip()
    prefix = "repo://"
    if not uri.startswith(prefix):
        raise ValueError(f"Execution input canonical URI must use repo://: {uri}")
    path = PurePosixPath(uri[len(prefix) :])
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Execution input canonical URI is invalid: {uri}")
    return path


def _physical_path(entry: Mapping[str, object], environment: Mapping[str, str]) -> Path:
    canonical_path = _repo_uri_path(entry.get("canonical_uri"))
    role = str(entry.get("allowed_execution_role", ""))
    if role == "processed_test_input":
        expected_prefix = PurePosixPath("Input/processed")
        root_value = environment.get(PROCESSED_ROOT_ENV, "").strip()
        if canonical_path.parts[:2] != expected_prefix.parts:
            raise ValueError(f"Processed execution input is outside Input/processed: {canonical_path}")
        relative_path = canonical_path.relative_to(expected_prefix)
    elif role == "read_only_parent_audit":
        expected_prefix = PurePosixPath("Output")
        root_value = environment.get(AUDIT_ROOT_ENV, "").strip()
        if canonical_path.parts[:1] != expected_prefix.parts:
            raise ValueError(f"Read-only audit input is outside Output: {canonical_path}")
        relative_path = canonical_path
    else:
        raise ValueError(f"Execution input has an unsupported role: {role}")
    if not root_value:
        variable = PROCESSED_ROOT_ENV if role == "processed_test_input" else AUDIT_ROOT_ENV
        raise FileNotFoundError(f"Missing execution input root environment variable: {variable}")
    root = Path(root_value).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Execution input root is unavailable or symlinked: {root}")
    root = root.resolve()
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Execution input contains a symlinked path component: {canonical_path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Execution input escaped its configured root: {canonical_path}")
    return resolved


def load_execution_input_bundle(
    environment: Mapping[str, str] | None = None,
    *,
    config_stack_path: Path | None = None,
    expected_code_commit: str | None = None,
) -> ExecutionInputBundle:
    resolved_environment = dict(os.environ if environment is None else environment)
    manifest_value = resolved_environment.get(BUNDLE_ENV, "").strip()
    if not manifest_value:
        raise FileNotFoundError(f"Missing execution input bundle environment variable: {BUNDLE_ENV}")
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Execution input bundle is unavailable: {manifest_path}")
    expected_manifest_hash = _sha256_value(
        resolved_environment.get(BUNDLE_SHA256_ENV, ""),
        "Execution input bundle environment",
    )
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "Execution input bundle hash mismatch: "
            f"expected {expected_manifest_hash}, observed {observed_manifest_hash}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Execution input bundle must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Execution input bundle schema version is invalid")
    if payload.get("validation_status") != "passed":
        raise ValueError("Execution input bundle has not passed validation")
    if payload.get("missing_entries") != []:
        raise ValueError("Execution input bundle contains unresolved missing entries")
    code_commit = _required_text(payload, "code_commit", "Execution input bundle")
    if expected_code_commit is not None and code_commit != expected_code_commit:
        raise ValueError(f"Execution input code commit mismatch: expected {expected_code_commit}, observed {code_commit}")
    config_hash = _sha256_value(payload.get("config_hash"), "Execution input bundle")
    if config_stack_path is not None:
        observed_config_hash = sha256_file(config_stack_path)
        if observed_config_hash != config_hash:
            raise ValueError(
                "Execution input config hash mismatch: "
                f"expected {config_hash}, observed {observed_config_hash}"
            )
    raw_entries: list[object] = []
    for key in ("processed_input_entries", "external_asset_entries"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Execution input bundle field must be a list: {key}")
        raw_entries.extend(value)
    assets: list[ExecutionInputAsset] = []
    seen_uids: set[str] = set()
    seen_uris: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Execution input asset must be an object")
        asset_uid = _required_text(raw_entry, "asset_uid", "Execution input asset")
        canonical_uri = _required_text(raw_entry, "canonical_uri", f"Execution input asset {asset_uid}")
        if asset_uid in seen_uids:
            raise ValueError(f"Duplicate logical asset UID: {asset_uid}")
        if canonical_uri in seen_uris:
            raise ValueError(f"Duplicate execution input canonical URI: {canonical_uri}")
        seen_uids.add(asset_uid)
        seen_uris.add(canonical_uri)
        access_status = _required_text(raw_entry, "license_or_access_status", f"Execution input asset {asset_uid}")
        if access_status not in ALLOWED_ACCESS_STATES:
            raise PermissionError(f"Execution input access is unresolved for {asset_uid}: {access_status}")
        role = _required_text(raw_entry, "allowed_execution_role", f"Execution input asset {asset_uid}")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Execution input role is invalid for {asset_uid}: {role}")
        required_by = raw_entry.get("required_by_testcases")
        if not isinstance(required_by, list) or not required_by or not all(str(item).strip() for item in required_by):
            raise ValueError(f"Execution input has no testcase ownership: {asset_uid}")
        size = raw_entry.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Execution input size is invalid: {asset_uid}")
        physical_path = _physical_path(raw_entry, resolved_environment)
        if not physical_path.is_file():
            raise FileNotFoundError(f"Execution input is unavailable: {asset_uid}")
        observed_size = physical_path.stat().st_size
        if observed_size != size:
            raise ValueError(f"Execution input size mismatch for {asset_uid}: expected {size}, observed {observed_size}")
        content_hash = _sha256_value(raw_entry.get("content_hash"), f"Execution input asset {asset_uid}")
        observed_hash = sha256_file(physical_path)
        if observed_hash != content_hash:
            raise ValueError(
                f"Execution input hash mismatch for {asset_uid}: expected {content_hash}, observed {observed_hash}"
            )
        assets.append(
            ExecutionInputAsset(
                asset_uid=asset_uid,
                logical_role=_required_text(raw_entry, "logical_role", f"Execution input asset {asset_uid}"),
                artifact_type=_required_text(raw_entry, "artifact_type", f"Execution input asset {asset_uid}"),
                source_uid=_required_text(raw_entry, "source_uid", f"Execution input asset {asset_uid}"),
                producer_uid=_required_text(raw_entry, "producer_uid", f"Execution input asset {asset_uid}"),
                version=_required_text(raw_entry, "version", f"Execution input asset {asset_uid}"),
                content_hash=content_hash,
                size=size,
                canonical_uri=canonical_uri,
                license_or_access_status=access_status,
                required_by_testcases=tuple(str(item) for item in required_by),
                allowed_execution_role=role,
                physical_path=physical_path,
            )
        )
    content_hashes = payload.get("content_hashes")
    expected_hashes = {asset.asset_uid: asset.content_hash for asset in assets}
    if content_hashes != expected_hashes:
        raise ValueError("Execution input bundle content_hashes do not match asset entries")
    canonical_locations = payload.get("canonical_locations")
    expected_locations = {asset.asset_uid: asset.canonical_uri for asset in assets}
    if canonical_locations != expected_locations:
        raise ValueError("Execution input bundle canonical_locations do not match asset entries")
    return ExecutionInputBundle(
        bundle_uid=_required_text(payload, "bundle_uid", "Execution input bundle"),
        code_commit=code_commit,
        config_hash=config_hash,
        environment_hash=_sha256_value(payload.get("environment_hash"), "Execution input bundle"),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=observed_manifest_hash,
        assets=tuple(assets),
        raw_payload=payload,
    )


def resolve_execution_input(
    bundle: ExecutionInputBundle,
    config_stack_path: Path,
    logical_asset_uid: str,
    *,
    expected_role: str | None = None,
) -> ExecutionInputAsset:
    if sha256_file(config_stack_path) != bundle.config_hash:
        raise ValueError("Execution input resolver rejected a mismatched config stack")
    matches = [asset for asset in bundle.assets if asset.asset_uid == logical_asset_uid]
    if len(matches) != 1:
        raise KeyError(f"Execution input asset UID does not resolve exactly once: {logical_asset_uid}")
    asset = matches[0]
    if expected_role is not None and asset.allowed_execution_role != expected_role:
        raise ValueError(
            f"Execution input role mismatch for {logical_asset_uid}: "
            f"expected {expected_role}, observed {asset.allowed_execution_role}"
        )
    return asset


def _asset_source_root(asset: ExecutionInputAsset) -> Path:
    canonical_path = _repo_uri_path(asset.canonical_uri)
    if asset.allowed_execution_role == "processed_test_input":
        relative_path = canonical_path.relative_to(PurePosixPath("Input/processed"))
    elif asset.allowed_execution_role == "read_only_parent_audit":
        relative_path = canonical_path
    else:
        raise ValueError(f"Execution input has an unsupported role: {asset.allowed_execution_role}")
    source_root = asset.physical_path
    for _ in relative_path.parts:
        source_root = source_root.parent
    if source_root.joinpath(*relative_path.parts) != asset.physical_path:
        raise ValueError(f"Execution input source root cannot be derived for {asset.asset_uid}")
    return source_root


def _bundle_source_roots(bundle: ExecutionInputBundle) -> dict[str, Path]:
    roots_by_role: dict[str, set[Path]] = {}
    for asset in bundle.assets:
        roots_by_role.setdefault(asset.allowed_execution_role, set()).add(_asset_source_root(asset))
    roots: dict[str, Path] = {}
    for role, candidates in roots_by_role.items():
        if len(candidates) != 1:
            raise ValueError(f"Execution input role does not resolve to one physical source root: {role}")
        roots[role] = next(iter(candidates))
    return roots


def _strict_repo_relative_path(value: str, expected_prefix: PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Namespace probe path is invalid: {value}")
    if path.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
        raise ValueError(f"Namespace probe path is outside {expected_prefix}: {value}")
    return path


def build_execution_bwrap_command(
    worktree_root: Path,
    bundle: ExecutionInputBundle,
    command: Sequence[str],
    *,
    output_root: Path | None = None,
) -> list[str]:
    if not command:
        raise ValueError("An execution command is required")
    if shutil.which("bwrap") is None:
        raise RuntimeError("bubblewrap is required for Round-11G1 execution input isolation")
    root = worktree_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Execution worktree root is invalid: {root}")
    processed_target = root / "Input/processed"
    if processed_target.exists() and (processed_target.is_symlink() or any(processed_target.iterdir())):
        raise ValueError("Execution worktree Input/processed must be absent or empty")
    source_roots = _bundle_source_roots(bundle)
    physical_roots = tuple(source_roots.values())
    for source_root in physical_roots:
        if source_root == Path("/") or source_root == root or source_root.is_relative_to(root) or root.is_relative_to(source_root):
            raise ValueError("Execution input physical source root overlaps the clean execution worktree")
    for index, source_root in enumerate(physical_roots):
        for other_root in physical_roots[index + 1 :]:
            if source_root.is_relative_to(other_root) or other_root.is_relative_to(source_root):
                raise ValueError("Execution input physical source roots overlap")
    resolved_output_root: Path | None = None
    if output_root is not None:
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError("Execution output root is unavailable or symlinked")
        resolved_output_root = output_root.resolve()
        protected_roots = (root, *physical_roots)
        if any(
            resolved_output_root == protected
            or resolved_output_root.is_relative_to(protected)
            or protected.is_relative_to(resolved_output_root)
            for protected in protected_roots
        ):
            raise ValueError("Execution output root overlaps a protected code or input root")
    dev_root = Path("/dev")
    dev_targets = [
        path
        for path in (root, *physical_roots, resolved_output_root)
        if path is not None and path.is_relative_to(dev_root)
    ]
    dev_directories: set[Path] = set()
    for target in dev_targets:
        parent = target.parent
        while parent != dev_root:
            dev_directories.add(parent)
            parent = parent.parent
    arguments = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
    ]
    for directory in sorted(dev_directories, key=lambda path: (len(path.parts), path.as_posix())):
        arguments.extend(["--dir", str(directory)])
    arguments.extend(
        [
        "--bind",
        str(root),
        str(root),
        ]
    )
    for source_root in sorted(physical_roots, key=lambda path: path.as_posix()):
        if source_root.is_relative_to(dev_root):
            arguments.extend(["--ro-bind", str(source_root), str(source_root)])
    arguments.extend(
        [
        "--dir",
        str(root / "Input"),
        "--tmpfs",
        str(processed_target),
        "--tmpfs",
        "/run",
        "--dir",
        str(RUNTIME_ROOT),
        ]
    )
    if resolved_output_root is not None:
        arguments.extend(
            [
                "--dir",
                str(RUNTIME_OUTPUT_ROOT),
                "--bind",
                str(resolved_output_root),
                str(RUNTIME_OUTPUT_ROOT),
                "--setenv",
                OUTPUT_ROOT_ENV,
                str(RUNTIME_OUTPUT_ROOT),
            ]
        )
    else:
        arguments.extend(["--unsetenv", OUTPUT_ROOT_ENV])
    processed_assets = [asset for asset in bundle.assets if asset.allowed_execution_role == "processed_test_input"]
    directories = {processed_target}
    for asset in processed_assets:
        logical_path = Path(_repo_uri_path(asset.canonical_uri).as_posix())
        parent = (root / logical_path).parent
        while parent != processed_target:
            directories.add(parent)
            parent = parent.parent
    for directory in sorted(directories, key=lambda path: (len(path.parts), path.as_posix())):
        arguments.extend(["--dir", str(directory)])
    for asset in processed_assets:
        logical_path = Path(_repo_uri_path(asset.canonical_uri).as_posix())
        arguments.extend(["--ro-bind", str(asset.physical_path), str(root / logical_path)])
    audit_assets = [asset for asset in bundle.assets if asset.allowed_execution_role == "read_only_parent_audit"]
    for asset in audit_assets:
        runtime_target = Path(RUNTIME_ROOT.as_posix()) / _repo_uri_path(asset.canonical_uri).as_posix()
        parent = runtime_target.parent
        parents: list[Path] = []
        while parent != Path(RUNTIME_ROOT.as_posix()):
            parents.append(parent)
            parent = parent.parent
        for directory in reversed(parents):
            arguments.extend(["--dir", str(directory)])
        arguments.extend(["--ro-bind", str(asset.physical_path), str(runtime_target)])
    for source_root in sorted(physical_roots, key=lambda path: path.as_posix()):
        arguments.extend(["--tmpfs", str(source_root)])
    arguments.extend(
        [
            "--setenv",
            AUDIT_ROOT_ENV,
            RUNTIME_ROOT.as_posix(),
            "--unsetenv",
            PROCESSED_ROOT_ENV,
            "--unsetenv",
            BUNDLE_ENV,
            "--unsetenv",
            BUNDLE_SHA256_ENV,
            "--unsetenv",
            "PYTHONPATH",
            "--proc",
            "/proc",
            "--chdir",
            str(root),
            "--",
            *command,
        ]
    )
    return arguments


def run_with_execution_inputs(
    worktree_root: Path,
    bundle: ExecutionInputBundle,
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    output_root: Path | None = None,
    capture_output: bool = False,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    root = worktree_root.resolve()
    input_root = root / "Input"
    processed_root = input_root / "processed"
    created_input = False
    created_processed = False
    if not input_root.exists():
        input_root.mkdir()
        created_input = True
    if not processed_root.exists():
        processed_root.mkdir()
        created_processed = True
    try:
        return subprocess.run(
            build_execution_bwrap_command(root, bundle, command, output_root=output_root),
            check=False,
            text=True,
            env=None if environment is None else dict(environment),
            capture_output=capture_output,
            timeout=timeout_seconds,
        )
    finally:
        if created_processed:
            processed_root.rmdir()
        if created_input:
            input_root.rmdir()


def run_execution_namespace_smoke(
    worktree_root: Path,
    bundle: ExecutionInputBundle,
    *,
    tracked_input_path: str,
    unlisted_processed_path: str,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    root = worktree_root.resolve()
    tracked_path = _strict_repo_relative_path(tracked_input_path, PurePosixPath("Input/structures"))
    unlisted_path = _strict_repo_relative_path(unlisted_processed_path, PurePosixPath("Input/processed"))
    processed_assets = sorted(
        (asset for asset in bundle.assets if asset.allowed_execution_role == "processed_test_input"),
        key=lambda asset: asset.asset_uid,
    )
    audit_assets = sorted(
        (asset for asset in bundle.assets if asset.allowed_execution_role == "read_only_parent_audit"),
        key=lambda asset: asset.asset_uid,
    )
    if not processed_assets or not audit_assets:
        raise ValueError("Namespace smoke requires processed and read-only audit assets")
    processed_asset = processed_assets[0]
    audit_asset = audit_assets[0]
    unlisted_target = root / Path(unlisted_path.as_posix())
    if any(unlisted_target == asset.physical_path for asset in bundle.assets):
        raise ValueError("Namespace smoke unlisted path is present in the execution input bundle")
    probe_paths = {
        "manifested_processed_visible": str(root / Path(_repo_uri_path(processed_asset.canonical_uri).as_posix())),
        "tracked_input_visible": str(root / Path(tracked_path.as_posix())),
        "materialized_audit_visible": str(
            Path(RUNTIME_ROOT.as_posix()) / Path(_repo_uri_path(audit_asset.canonical_uri).as_posix())
        ),
        "unlisted_processed_hidden": str(unlisted_target),
        "processed_source_hidden": str(processed_asset.physical_path),
        "audit_source_hidden": str(audit_asset.physical_path),
    }
    probe_script = (
        "import json, pathlib, sys; "
        "paths=json.loads(sys.argv[1]); "
        "checks={key:(pathlib.Path(value).is_file() if key.endswith('_visible') else not pathlib.Path(value).exists()) "
        "for key,value in paths.items()}; "
        "print(json.dumps(checks, sort_keys=True)); "
        "raise SystemExit(0 if all(checks.values()) else 3)"
    )
    completed = run_with_execution_inputs(
        root,
        bundle,
        [sys.executable, "-c", probe_script, json.dumps(probe_paths, sort_keys=True)],
        capture_output=True,
        timeout_seconds=timeout_seconds,
    )
    try:
        checks = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        checks = {}
    passed = completed.returncode == 0 and checks and all(checks.get(key) is True for key in probe_paths)
    return {
        "status": "passed" if passed else "failed",
        "returncode": completed.returncode,
        "checks": checks,
        "manifested_processed_uri": processed_asset.canonical_uri,
        "tracked_input_uri": f"repo://{tracked_path.as_posix()}",
        "materialized_audit_uri": audit_asset.canonical_uri,
        "unlisted_processed_uri": f"repo://{unlisted_path.as_posix()}",
    }
