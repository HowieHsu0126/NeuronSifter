"""Materialize the three deterministic controlled neural world packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS, validate_controlled_world_config  # noqa: E402
from Libs.utils.data.controlled_world_runtime import (  # noqa: E402
    build_family_manifest,
    generate_all_family_packages,
    sha256_file,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


WORLD_SCHEMA_PATH = Path("Libs/configs/schema/controlled_neural_world.schema.json")
REPLAY_SCHEMA_PATH = Path("Libs/configs/schema/controlled_world_replay_manifest.schema.json")
CONTROLLED_CONFIG_PATH = Path("Libs/configs/data/controlled_neural_world.yaml")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _validate_schema(schema_path: Path, value: Mapping[str, object]) -> None:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(dict(value))


def _resolve_artifact(artifact_root: Path, relative_path: object) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Controlled-world output path must be repository-relative: {relative}")
    return artifact_root / relative


def build_controlled_neural_world(
    *,
    config_path: Path,
    repository_root: Path,
    artifact_root: Path,
) -> dict[str, Path]:
    config = load_config(config_path)
    source = validate_controlled_world_config(config)
    _validate_schema(repository_root / WORLD_SCHEMA_PATH, source)
    config_file_identity = sha256_file(repository_root / CONTROLLED_CONFIG_PATH)
    packages = generate_all_family_packages(
        source,
        repository_root=repository_root,
        config_file_identity=config_file_identity,
    )
    family_manifest = build_family_manifest(source, packages, config_file_identity=config_file_identity)

    for family_uid in FAMILY_UIDS:
        instance, replay = packages[family_uid]
        _validate_schema(repository_root / WORLD_SCHEMA_PATH, instance)
        _validate_schema(repository_root / REPLAY_SCHEMA_PATH, replay)
    _validate_schema(repository_root / WORLD_SCHEMA_PATH, family_manifest)

    paths: dict[str, Path] = {}
    for family in source["families"]:
        family_uid = str(family["family_uid"])
        instance, replay = packages[family_uid]
        instance_path = _resolve_artifact(artifact_root, family["output_paths"]["world_instance"])
        replay_path = _resolve_artifact(artifact_root, family["output_paths"]["replay_manifest"])
        write_json(instance_path, instance)
        write_json(replay_path, replay)
        paths[f"{family_uid}.world_instance"] = instance_path
        paths[f"{family_uid}.replay_manifest"] = replay_path
    manifest_path = _resolve_artifact(artifact_root, source["family_manifest"]["output_path"])
    write_json(manifest_path, family_manifest)
    paths["family_manifest"] = manifest_path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    paths = build_controlled_neural_world(
        config_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
