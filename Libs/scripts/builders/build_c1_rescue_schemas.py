"""Seal V2-specific C1 schemas without weakening the frozen V1 schemas."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


SCHEMA_SPECS = (
    (
        "Libs/configs/schema/foundation_state_c1.schema.json",
        "Libs/configs/schema/foundation_state_c1_rescue.schema.json",
        "config",
    ),
    (
        "Libs/configs/schema/c1_qualified_action_subset_manifest.schema.json",
        "Libs/configs/schema/c1_rescue_qualified_action_subset_manifest.schema.json",
        "Output/audits/c1_rescue_qualified_action_subset_manifest.json",
    ),
    (
        "Libs/configs/schema/c1_observation_row_binding_manifest.schema.json",
        "Libs/configs/schema/c1_rescue_observation_row_binding_manifest.schema.json",
        "Output/audits/c1_rescue_observation_row_binding_manifest.json",
    ),
    (
        "Libs/configs/schema/c1_vocabulary_manifest.schema.json",
        "Libs/configs/schema/c1_rescue_vocabulary_manifest.schema.json",
        "Output/audits/c1_rescue_vocabulary_manifest.json",
    ),
    (
        "Libs/configs/schema/c1_objective_capability_manifest.schema.json",
        "Libs/configs/schema/c1_rescue_objective_capability_manifest.schema.json",
        "Output/audits/c1_rescue_objective_capability_manifest.json",
    ),
    (
        "Libs/configs/schema/c1_unified_preflight.schema.json",
        "Libs/configs/schema/c1_rescue_unified_preflight.schema.json",
        "Output/audits/c1_rescue_preflight.json",
    ),
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _bind_existing_consts(
    schema: object,
    payload: object,
    *,
    root_schema: dict[str, object],
) -> None:
    if not isinstance(schema, dict):
        return
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        definition_name = reference.removeprefix("#/$defs/")
        definitions = root_schema.get("$defs")
        if not isinstance(definitions, dict) or definition_name not in definitions:
            raise ValueError(f"C1 schema has an unresolved local reference: {reference}")
        _bind_existing_consts(
            definitions[definition_name], payload, root_schema=root_schema
        )
    if "const" in schema:
        schema["const"] = deepcopy(payload)
        return
    if (
        isinstance(payload, list)
        and "minItems" in schema
        and "maxItems" in schema
    ):
        schema["minItems"] = len(payload)
        schema["maxItems"] = len(payload)
    for combinator in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(combinator)
        if isinstance(branches, list):
            for branch in branches:
                _bind_existing_consts(branch, payload, root_schema=root_schema)
    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(payload, Mapping):
        for key, child in properties.items():
            if key in payload:
                _bind_existing_consts(
                    child, payload[key], root_schema=root_schema
                )
    items = schema.get("items")
    if isinstance(items, dict) and isinstance(payload, list):
        for value in payload:
            _bind_existing_consts(items, value, root_schema=root_schema)


def _add_v2_status_contract(schema: dict[str, object], payload: Mapping[str, object]) -> None:
    if "status" not in payload:
        return
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, dict):
        raise TypeError("C1 schema root must define required/properties")
    if "status" not in required:
        required.append("status")
    properties["status"] = {"const": payload["status"]}


def _unified_preflight_bootstrap_payload(
    contract: Mapping[str, object],
) -> dict[str, object]:
    frozen = contract["frozen_corpus"]
    loader = contract["loader"]
    training = contract["training"]
    return {
        "schema_version": "neuronsifter-c1-unified-preflight-v3",
        "corpus_uid": frozen["corpus_uid"],
        "eligible_roles": list(frozen["required_roles"]),
        "sampling_policy": {
            "family": loader["sampling_policy"],
            "biological_unit": loader["weighting_policy"],
            "within_unit": loader["within_unit_aggregation"],
        },
        "training_seed_list": list(training["headline_seeds"]),
    }


def _bind_unified_preflight_cardinality(
    schema: dict[str, object],
    contract: Mapping[str, object],
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise TypeError("C1 unified preflight schema properties are missing")
    frozen = contract["frozen_corpus"]
    for key, values in (
        ("block_hashes", frozen["required_blocks"]),
        ("split_hashes", frozen["required_split_manifests"]),
    ):
        target = properties.get(key)
        if not isinstance(target, dict) or not isinstance(values, Mapping):
            raise TypeError(f"C1 unified preflight schema cannot bind {key}")
        target["minProperties"] = len(values)
        target["maxProperties"] = len(values)


def seal_schemas(
    repo_root: Path = REPO_ROOT,
    *,
    include_optional: bool = True,
) -> list[str]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / "Libs/configs/config_stack_c1_rescue.yaml")
    contract = config["foundation_state_c1"]
    written: list[str] = []
    for base_name, output_name, payload_name in SCHEMA_SPECS:
        payload_path = repo_root / payload_name
        unified_preflight = output_name.endswith("c1_rescue_unified_preflight.schema.json")
        if payload_name == "config":
            payload: Mapping[str, object] = contract
        elif not payload_path.is_file():
            if unified_preflight:
                payload = _unified_preflight_bootstrap_payload(contract)
            elif include_optional:
                continue
            else:
                raise FileNotFoundError(
                    f"Required C1 V2 schema payload is missing: {payload_name}"
                )
        else:
            payload = _load_json(payload_path)
        schema = _load_json(repo_root / base_name)
        _bind_existing_consts(schema, payload, root_schema=schema)
        if unified_preflight:
            _bind_unified_preflight_cardinality(schema, contract)
        else:
            _add_v2_status_contract(schema, payload)
        schema["$id"] = str(schema.get("$id", "")).replace(
            Path(base_name).name, Path(output_name).name
        )
        schema["title"] = f"{schema.get('title', 'NeuronSifter C1 schema')} V2 rescue"
        output_path = repo_root / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(schema, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(output_name)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail unless vocabulary, capability, and unified-preflight payloads exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = seal_schemas(REPO_ROOT, include_optional=not args.require_all)
    print(json.dumps({"status": "sealed", "schemas": written}, sort_keys=True))


if __name__ == "__main__":
    main()
