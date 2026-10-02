"""Validate the deterministic C1 vocabulary manifest against live metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_loader import build_c1_vocabulary  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--manifest", default="Output/audits/c1_vocabulary_manifest.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    _, expected = build_c1_vocabulary(REPO_ROOT, config)
    manifest_path = (REPO_ROOT / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_name = (
        "c1_rescue_vocabulary_manifest.schema.json"
        if config["foundation_state_c1"]["frozen_corpus"]["corpus_uid"]
        == "public_foundation_corpus_rescue"
        else "c1_vocabulary_manifest.schema.json"
    )
    schema = json.loads(
        (REPO_ROOT / "Libs/configs/schema" / schema_name).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    if errors:
        raise ValueError(
            "C1 vocabulary manifest schema violation: "
            + "; ".join(e.message for e in errors)
        )
    if manifest != expected:
        raise ValueError(
            "C1 vocabulary manifest differs from deterministic metadata reconstruction"
        )
    print("c1_vocabulary_manifest_status=passed")
    print(f"c1_vocabulary_feature_count={manifest['feature_count']}")


if __name__ == "__main__":
    main()
