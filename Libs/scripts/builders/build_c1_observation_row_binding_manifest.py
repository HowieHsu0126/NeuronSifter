"""Materialize the corpus-identity-bound C1 observation row binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_public_foundation_corpus_evidence import (
    validate_bundle_payload,
)
from Libs.utils.data.foundation_state_c1_bindings import (
    build_c1_observation_binding_artifacts,
    serialize_binding_manifest,
)
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file
from Libs.utils.data.raw_data_registry import load_config, write_json, write_text_atomic


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _derived(config: dict[str, object], repo_root: Path) -> tuple[str, dict[str, object]]:
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    bundle_path = safe_repo_path(repo_root, str(frozen["evidence_bundle_path"]))
    bundle = _load_json(bundle_path)
    report = validate_bundle_payload(
        bundle,
        repo_root,
        allow_controlled_isolation_refresh=(
            frozen.get("allowed_governance_reference_refresh") == ["controlled_isolation"]
        ),
    )
    if report.get("status") != "valid_fail_closed":
        raise ValueError(f"Frozen Phase 1 evidence is invalid: {report.get('blockers')}")
    observation_path = safe_repo_path(
        repo_root, str(bundle["evidence"]["canonical_observation_blocks"]["path"])
    )
    observation = _load_json(observation_path)
    qualified_blocks = [
        row
        for row in observation.get("blocks", [])
        if isinstance(row, dict) and row.get("observation_eligible") is True
    ]
    required = frozen["required_observation_binding"]
    row_text, manifest = build_c1_observation_binding_artifacts(
        repo_root,
        bundle,
        qualified_blocks,
        frozen["required_query_domain"],
        row_table_path=str(required["row_table_path"]),
    )
    schema = _load_json(safe_repo_path(repo_root, str(required["schema_path"])))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError("Derived binding violates schema: " + "; ".join(e.message for e in errors))
    return row_text, manifest


def _identity(row_text: str, manifest: dict[str, object]) -> dict[str, object]:
    manifest_text = serialize_binding_manifest(manifest)
    return {
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "row_table_sha256": hashlib.sha256(row_text.encode("utf-8")).hexdigest(),
        "binding_hash": manifest["binding_hash"],
        "block_count": manifest["block_count"],
        "biological_unit_count": manifest["biological_unit_count"],
        "observation_count": manifest["observation_count"],
    }


def build(config_path: str, repo_root: Path = REPO_ROOT) -> tuple[str, dict[str, object], dict[str, object]]:
    config = load_config(config_path)
    row_text, manifest = _derived(config, repo_root)
    identities = _identity(row_text, manifest)
    required = config["foundation_state_c1"]["frozen_corpus"]["required_observation_binding"]
    for key, expected in identities.items():
        if str(required[key]) != str(expected):
            raise ValueError(
                f"Derived observation binding identity differs from frozen config for {key}: "
                f"{expected} != {required[key]}"
            )
    return row_text, manifest, identities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen C1 observation row binding manifest.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--print-derived-identities", action="store_true")
    parser.add_argument("--output-manifest", default="")
    parser.add_argument("--output-row-table", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    required = frozen["required_observation_binding"]
    row_text, manifest = _derived(config, REPO_ROOT)
    identities = _identity(row_text, manifest)
    if args.print_derived_identities:
        print(json.dumps(identities, sort_keys=True))
        return
    for key, expected in identities.items():
        if str(required[key]) != str(expected):
            raise ValueError(
                f"Derived observation binding identity differs from frozen config for {key}: "
                f"{expected} != {required[key]}"
            )
    schema = _load_json(safe_repo_path(REPO_ROOT, str(required["schema_path"])))
    Draft202012Validator.check_schema(schema)
    manifest_errors = list(Draft202012Validator(schema).iter_errors(manifest))
    if manifest_errors:
        raise ValueError("Binding manifest schema errors: " + "; ".join(e.message for e in manifest_errors))
    row_path = safe_repo_path(REPO_ROOT, args.output_row_table or str(required["row_table_path"]))
    manifest_path = safe_repo_path(REPO_ROOT, args.output_manifest or str(required["manifest_path"]))
    write_text_atomic(row_path, row_text)
    write_json(manifest_path, manifest)
    if sha256_file(row_path) != str(required["row_table_sha256"]):
        raise ValueError("Written row table hash does not match the frozen binding identity")
    if sha256_file(manifest_path) != str(required["manifest_sha256"]):
        raise ValueError("Written binding manifest hash does not match the frozen binding identity")
    print(f"c1_observation_binding_hash={manifest['binding_hash']}")
    print(f"c1_observation_binding_units={manifest['biological_unit_count']}")
    print(f"c1_observation_binding_observations={manifest['observation_count']}")
    print(f"c1_observation_binding_manifest={manifest_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"c1_observation_binding_row_table={row_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
