from __future__ import annotations

import argparse
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
from Libs.utils.data.foundation_state_c1 import (
    build_qualified_action_subset_manifest,
    validate_c1_contract,
)
from Libs.utils.data.public_foundation_evidence import safe_repo_path
from Libs.utils.data.raw_data_registry import load_config, write_json


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def build(config_path: str, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    config = load_config(config_path)
    contract_blockers = validate_c1_contract(config, repo_root)
    if contract_blockers:
        raise ValueError(f"C1 contract is invalid: {contract_blockers}")
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    bundle = _load_json(safe_repo_path(repo_root, str(frozen["evidence_bundle_path"])))
    bundle_report = validate_bundle_payload(bundle, repo_root)
    if bundle_report["status"] != "valid_fail_closed":
        raise ValueError(f"Phase 1 evidence bundle is invalid: {bundle_report['blockers']}")
    evidence = bundle["evidence"]
    actions = _load_json(
        safe_repo_path(repo_root, str(evidence["structured_action_catalog"]["path"]))
    )
    manifest = build_qualified_action_subset_manifest(
        bundle,
        actions,
        repo_root,
        required_roles={str(value) for value in frozen["required_roles"]},
    )
    if manifest["record_count"] != frozen["required_qualified_action_count"]:
        raise ValueError("Qualified action count does not match the C1 contract")
    if manifest["qualified_action_subset_hash"] != frozen["qualified_action_subset_hash"]:
        raise ValueError("Qualified action subset hash does not match the C1 contract")
    schema = _load_json(
        safe_repo_path(repo_root, str(frozen["qualified_action_subset_schema_path"]))
    )
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "Qualified action manifest violates its schema: "
            + "; ".join(error.message for error in errors)
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen C1 qualified-action subset manifest."
    )
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    output_path = safe_repo_path(
        REPO_ROOT,
        args.output or str(frozen["qualified_action_subset_manifest_path"]),
    )
    manifest = build(args.config, REPO_ROOT)
    write_json(output_path, manifest)
    print(f"c1_qualified_action_count={manifest['record_count']}")
    print(f"c1_qualified_action_subset_hash={manifest['qualified_action_subset_hash']}")
    print(f"c1_qualified_action_manifest={output_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
