"""Dataset-owned locations for an explicit admitted native training input set.

Original payloads and provenance paths stay unchanged. Explicit immutable stages
use a strict local reader, not a cache or a source-registry rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil
import stat
from typing import Iterable

import yaml

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.model.screening_model_config import native_direct_mean_config

SCHEMA = "native_input_locations.v1"
MANIFEST_NAME = "native_inputs.json"
ROLES = {"train", "development", "validation", "calibration"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_input_paths(store: Path, name: str, metadata: dict, store_format: str,
                      indices: Iterable[int]) -> list[Path]:
    """Use the same row and tail chunk geometry as the maintained store readers."""
    rows = sorted(set(int(index) for index in indices))
    if store_format == "neuronsifter_chunk_store":
        relative = Path(metadata["relative_root"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Staged array root must stay within its declared store")
        size = int(metadata["chunk_size"])
        if size <= 0:
            raise ValueError("Staged array chunk size must be positive")
        return [store / relative / f"chunk_{index:06d}.npy"
                for index in sorted({row // size for row in rows})]
    if store_format != "zarr_v2":
        raise ValueError("Unsupported staged native store format: " + store_format)
    shape, chunks = tuple(metadata["shape"]), tuple(metadata["chunks"])
    if len(shape) != len(chunks) or not chunks or any(int(size) <= 0 for size in chunks):
        raise ValueError("Staged Zarr chunk geometry is invalid")
    tails = list(product(*[range(math.ceil(size / chunk)) for size, chunk in zip(shape[1:], chunks[1:])]))
    return [store / name / ".".join(str(value) for value in (first, *tail))
            for first in sorted({row // chunks[0] for row in rows}) for tail in tails]


@dataclass(frozen=True)
class NativeInputLocations:
    root: Path
    aliases: dict[str, str]
    files: dict[str, dict]
    stores: frozenset[str]
    groups: dict[str, list[str]]

    @classmethod
    def load(cls, manifest: str | Path) -> NativeInputLocations:
        path = Path(manifest).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA:
            raise ValueError("Native input location schema mismatch")
        aliases, files = payload["aliases"], payload["files"]
        stores, groups = payload["stores"], payload["groups"]
        if (not isinstance(aliases, dict) or not isinstance(files, dict) or not files
                or not isinstance(stores, list) or not groups or not set(groups) <= ROLES):
            raise ValueError("Invalid native input location manifest")
        for logical in {*files, *stores, *aliases.values()}:
            value = Path(logical)
            if not value.is_absolute() or ".." in value.parts:
                raise ValueError("Staged logical identity must be an absolute normalized path")
        for entry in files.values():
            if (type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0
                    or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64):
                raise ValueError("Invalid staged file integrity binding")
        return cls(path.parent, aliases, files, frozenset(stores), groups)

    def logical_path(self, value: str | Path) -> Path:
        key = str(value)
        logical = self.aliases.get(key, key)
        if logical not in self.files and logical not in self.stores:
            raise FileNotFoundError("Native input was not staged: " + key)
        return Path(logical)

    def file_path(self, value: str | Path) -> Path:
        logical = self.logical_path(value)
        entry = self.files.get(str(logical))
        if entry is None:
            raise FileNotFoundError("Native input is not a staged file: " + str(logical))
        path = self.root / "files" / logical.relative_to("/")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry["size_bytes"]:
            raise ValueError("Staged native file type/size changed: " + str(logical))
        return path

    def read_bytes(self, value: str | Path) -> bytes:
        logical = self.logical_path(value)
        data = self.file_path(logical).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.files[str(logical)]["sha256"]:
            raise ValueError("Staged native file bytes changed: " + str(logical))
        return data

    def read_text(self, value: str | Path) -> str:
        return self.read_bytes(value).decode("utf-8")

    def store_path(self, value: str | Path) -> Path:
        logical = self.logical_path(value)
        if str(logical) not in self.stores:
            raise FileNotFoundError("Native store was not staged: " + str(logical))
        return self.root / "files" / logical.relative_to("/")

    def check_array_files(self, store: Path, name: str, metadata: dict,
                          store_format: str, indices: Iterable[int], verified: set[Path]) -> None:
        # Compare the copy digest before the unchanged numpy loader consumes it.
        # This also rejects descriptor redirection outside the explicit file set.
        for path in array_input_paths(store, name, metadata, store_format, indices):
            if path not in verified:
                if file_digest(self.file_path(path)) != self.files[str(path)]["sha256"]:
                    raise ValueError("Staged native array bytes changed: " + str(path))
                verified.add(path)

    def require_groups(self, role: str, selected: Iterable[str]) -> None:
        if role not in self.groups or not set(selected) <= set(self.groups[role]):
            raise ValueError("Native dataset requests a group/role outside its staged input set")

    def conditioning_config(self, value: str | Path) -> dict:
        payload = yaml.safe_load(self.read_text(value)) or {}
        if not isinstance(payload, dict) or payload.get("config_includes"):
            raise ValueError("Staged conditioning requires a standalone materialized configuration")
        # Reuse the canonical loader; the preceding check prevents include fallback.
        return load_config(self.file_path(value))


def input_locations_from_config(config: dict) -> NativeInputLocations | None:
    path = config.get("native_input_staging", {}).get("manifest_path")
    return NativeInputLocations.load(path) if path else None


def stage_native_inputs(config: dict, datasets: Iterable, destination: str | Path, *,
                        include_targets: bool = True,
                        reuse_manifest: str | Path | None = None) -> Path:
    """Copy explicit role groups, optionally retaining only public prediction inputs.

    Public-only copies omit stores entirely. The existing strict location reader
    then rejects a training/target consumer before any original-store fallback.

    Reuse hard-links unchanged files from an existing immutable stage on the same
    filesystem. Original admitted bytes are still checked; only payload copies
    and their storage allocation are avoided. New scoped normalizers are copied.
    """
    # Readers import the location type, so inventory imports follow module initialization.
    from Libs.utils.data.native_diffusion_data import consumed_array_names

    if input_locations_from_config(config) is not None:
        raise ValueError("Staging requires original admitted inputs, not another staging manifest")
    aliases: dict[str, str] = {}
    files: set[Path] = set()
    stores: dict[Path, dict] = {}
    groups: dict[str, list[str]] = {}
    reference_indexes = {}
    predose_teachers = []

    def logical(value: str | Path) -> Path:
        key = str(value)
        if key not in aliases:
            resolved = str(resolve_repo_path(key))
            aliases[key] = resolved
            aliases[resolved] = resolved
        return Path(aliases[key])

    def add_file(value: str | Path) -> Path:
        path = logical(value)
        files.add(path)
        return path

    main_path = add_file(config["native_diffusion_data"]["registry_path"])
    metadata = json.loads(main_path.read_text(encoding="utf-8"))
    main = metadata.get("metadata", metadata).get("native_diffusion", metadata)
    registry_payloads = [main]
    for path in main.get("family_registries", {}).values():
        payload = json.loads(add_file(path).read_text(encoding="utf-8"))
        registry_payloads.append(payload.get("metadata", payload).get("native_diffusion", payload))
    # Family-index admission validates every family grid even for a family-scoped fit.
    for registry in registry_payloads:
        normalizer = registry.get("grid", {}).get("normalization", {}).get("artifact_ref")
        if normalizer:
            add_file(normalizer)

    def add_references(registry: dict, posteriors: set[str]) -> None:
        key = id(registry)
        if key not in reference_indexes:
            inline = {row["source_posterior_uid"] for row in registry.get("joint_sources", [])}
            references = {row["source_posterior_uid"]: row for row in registry.get("joint_source_references", [])}
            reference_indexes[key] = (registry, inline, references)
        _, inline, references = reference_indexes[key]
        if inline & set(references) or not posteriors <= inline | set(references):
            raise ValueError("Selected complete native source reference is missing or ambiguous")
        for uid in posteriors & set(references):
            add_file(references[uid]["path"])

    for dataset in datasets:
        if dataset.split_name not in ROLES or dataset.split_name in groups:
            raise ValueError("Staging requires one dataset per explicit role")
        if dataset.input_locations is not None:
            raise ValueError("Cannot stage from a staged dataset")
        groups[dataset.split_name] = sorted(dataset.group_uids)
        for child, indices, _ in dataset.groups:
            registry = child.native_registry
            if registry.get("posterior_family") != "finite_joint_particles":
                raise ValueError("Native staging requires complete joint groups")
            grid_normalizer = registry["grid"].get("normalization", {}).get("artifact_ref")
            if grid_normalizer:
                add_file(grid_normalizer)
            teachers = [child.native_records[index]["teacher"] for index in indices]
            if include_targets and config.get("native_predose_query") and dataset.split_name in {"development", "validation"}:
                predose_teachers.extend(teachers)
            posteriors = {row["posterior_uid"] for row in teachers}
            add_references(registry, posteriors)
            for teacher in teachers if include_targets else ():
                store = logical(teacher["trajectory_store_path"])
                entry = stores.setdefault(store, {"format": teacher["store_format"], "arrays": {}, "posteriors": set()})
                if entry["format"] != teacher["store_format"]:
                    raise ValueError("Native store has conflicting format declarations")
                entry["posteriors"].add(teacher["posterior_uid"])
                for name in consumed_array_names(registry).values():
                    entry["arrays"].setdefault(name, set()).add(teacher["trajectory_index"])
    if not groups or any(not selected for selected in groups.values()):
        raise ValueError("Native staging requires nonempty explicitly selected roles")
    # The query evaluator consumes admission metadata for selected evaluation
    # roles. Copy those existing receipts, never producer truth or extra arrays.
    if predose_teachers:
        fixture_flags = [row.get("software_fixture_only") is True for row in predose_teachers]
        if all(fixture_flags):
            for path in {row["qc_catalog_ref"] for row in predose_teachers}:
                add_file(path)
        elif any(fixture_flags):
            raise ValueError("Pre-dose staging cannot mix artificial and scientific source metadata")
        else:
            named_path = add_file(main_path.parent / "native_consumption.json")
            named = json.loads(named_path.read_text(encoding="utf-8"))
            add_file(named["original_consumption_ref"])
    selected_normalizer = config["native_diffusion_data"].get("field_normalization", {}).get("model_artifact_ref")
    if selected_normalizer:
        add_file(selected_normalizer)
    direct = native_direct_mean_config(config)
    if direct:
        path = add_file(direct["conditioning_config_path"])
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict) or payload.get("config_includes"):
            raise ValueError("Staged conditioning requires a standalone materialized configuration")
    for store, entry in stores.items():
        descriptor = store / ("store_manifest.json" if entry["format"] == "neuronsifter_chunk_store" else ".zattrs")
        files.add(descriptor)
        metadata = json.loads(descriptor.read_text(encoding="utf-8"))
        stored_registry = metadata.get("metadata", metadata).get("native_diffusion", {})
        add_references(stored_registry, entry["posteriors"])
        for name, indices in entry["arrays"].items():
            if entry["format"] == "neuronsifter_chunk_store":
                array = metadata["arrays"][name]
            else:
                array_path = store / name / ".zarray"
                files.add(array_path)
                array = json.loads(array_path.read_text(encoding="utf-8"))
            files.update(array_input_paths(store, name, array, entry["format"], indices))

    reuse = NativeInputLocations.load(reuse_manifest) if reuse_manifest is not None else None
    if reuse is not None:
        for role, selected in groups.items():
            reuse.require_groups(role, selected)
    reusable = {path for path in files if reuse is not None and str(path) in reuse.files}

    destination = Path(destination).resolve()
    destination.mkdir()
    try:
        required_bytes = sum(original.stat().st_size for original in files - reusable)
        free_bytes = shutil.disk_usage(destination).free
        if required_bytes > free_bytes:
            raise OSError(errno.ENOSPC,
                          f"Native input staging capacity: required_bytes={required_bytes}, free_bytes={free_bytes}",
                          str(destination))
        copied = {}
        for original in sorted(files):
            local = destination / "files" / original.relative_to("/")
            local.parent.mkdir(parents=True, exist_ok=True)
            if original in reusable:
                entry = reuse.files[str(original)]
                if (original.stat().st_size != entry["size_bytes"]
                        or file_digest(original) != entry["sha256"]):
                    raise ValueError("Original admitted file differs from staged reuse: " + str(original))
                local.hardlink_to(reuse.file_path(original))
                if file_digest(local) != entry["sha256"]:
                    raise ValueError("Reused native file bytes changed: " + str(original))
                copied[str(original)] = dict(entry)
                continue
            digest, size = hashlib.sha256(), 0
            with original.open("rb") as source, local.open("xb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    target.write(block)
            copied[str(original)] = {"size_bytes": size, "sha256": digest.hexdigest()}
            if file_digest(local) != digest.hexdigest():
                raise ValueError("Native staged copy differs from original: " + str(original))
        payload = {"schema_version": SCHEMA, "aliases": aliases, "files": copied,
                   "stores": sorted(str(path) for path in stores), "groups": groups,
                   "file_count": len(copied), "size_bytes": sum(row["size_bytes"] for row in copied.values())}
        if reuse is not None:
            payload.update(reused_file_count=len(reusable),
                           reused_bytes=sum(copied[str(path)]["size_bytes"] for path in reusable),
                           copied_bytes=sum(copied[str(path)]["size_bytes"] for path in files - reusable))
        manifest = destination / MANIFEST_NAME
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    except BaseException:
        shutil.rmtree(destination)
        raise
