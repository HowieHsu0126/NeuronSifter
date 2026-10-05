"""Validate the deterministic, corpus-bound C1 objective capability manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    build_c1_objective_capability_manifest,
    load_c1_objective_capability_manifest,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import safe_repo_path  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    stored = load_c1_objective_capability_manifest(config, REPO_ROOT)
    rebuilt = build_c1_objective_capability_manifest(config, REPO_ROOT)
    if canonical_sha256(stored) != canonical_sha256(rebuilt):
        raise ValueError("Stored C1 objective capability manifest differs from live reconstruction")
    schema_path = safe_repo_path(
        REPO_ROOT,
        str(config["foundation_state_c1"]["artifacts"]["objective_capability_manifest_schema_path"]),
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(stored),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "C1 objective capability manifest violates its schema: "
            + "; ".join(error.message for error in errors)
        )
    print("c1_objective_capability_manifest_status=passed")
    print(f"c1_objective_capability_split_count={len(stored['splits'])}")


if __name__ == "__main__":
    main()
