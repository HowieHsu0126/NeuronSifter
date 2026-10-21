from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ReadOnlyArtifactReference:
    artifact_id: str
    logical_path: str
    physical_path: Path
    sha256: str
    access_mode: str
    linkage_id: str | None
    environment_variable: str | None


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy(config: Mapping[str, object]) -> dict[str, object]:
    union = _mapping(config.get("iclr_design_contract_union"))
    policy = _mapping(union.get("g0_read_only_artifact_linkage"))
    if not policy:
        raise KeyError("iclr_design_contract_union.g0_read_only_artifact_linkage")
    return policy


def _relative_logical_path(value: object) -> str:
    logical_path = Path(str(value))
    if logical_path.is_absolute() or ".." in logical_path.parts:
        raise ValueError(f"Read-only artifact logical path must be repository-relative: {logical_path}")
    return logical_path.as_posix()


def resolve_read_only_artifact(
    config: Mapping[str, object],
    artifact_id: str,
    worktree_root: Path,
) -> ReadOnlyArtifactReference:
    policy = _policy(config)
    artifacts = _mapping(policy.get("artifacts"))
    artifact_policy = _mapping(artifacts.get(artifact_id))
    if not artifact_policy:
        raise KeyError(f"Unregistered read-only artifact: {artifact_id}")
    logical_path = _relative_logical_path(artifact_policy.get("logical_path", ""))
    expected_sha256 = str(artifact_policy.get("sha256", "")).strip()
    if not expected_sha256:
        raise ValueError(f"Read-only artifact has no expected SHA-256: {artifact_id}")
    local_path = worktree_root / logical_path
    access_mode = "worktree"
    linkage_id: str | None = None
    environment_variable: str | None = None
    physical_path = local_path
    if not local_path.is_file():
        environment_variable = str(policy.get("environment_variable", "")).strip()
        linkage_id = str(policy.get("linkage_id", "")).strip() or None
        if not environment_variable:
            raise RuntimeError(f"Read-only artifact root environment variable is not configured for {artifact_id}")
        artifact_root_value = os.environ.get(environment_variable, "").strip()
        if not artifact_root_value:
            raise FileNotFoundError(
                f"Missing {logical_path} in this worktree; set {environment_variable} to an approved read-only artifact root."
            )
        artifact_root = Path(artifact_root_value).resolve()
        physical_path = (artifact_root / logical_path).resolve()
        if not physical_path.is_relative_to(artifact_root):
            raise ValueError(f"Read-only artifact escaped its configured root: {logical_path}")
        access_mode = "external_read_only"
    if not physical_path.is_file():
        raise FileNotFoundError(f"Read-only artifact is unavailable: {logical_path}")
    observed_sha256 = _sha256_file(physical_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Read-only artifact hash mismatch for {logical_path}: expected {expected_sha256}, observed {observed_sha256}"
        )
    return ReadOnlyArtifactReference(
        artifact_id=artifact_id,
        logical_path=logical_path,
        physical_path=physical_path,
        sha256=observed_sha256,
        access_mode=access_mode,
        linkage_id=linkage_id,
        environment_variable=environment_variable,
    )


def artifact_metadata(reference: ReadOnlyArtifactReference) -> dict[str, object]:
    return {
        "path": reference.logical_path,
        "sha256": reference.sha256,
        "size_bytes": reference.physical_path.stat().st_size,
        "access_mode": reference.access_mode,
        "linkage_id": reference.linkage_id,
        "environment_variable": reference.environment_variable,
    }
