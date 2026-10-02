from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1 import build_c1_preflight_report
from Libs.utils.data.public_foundation_evidence import safe_repo_path
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen reduced-core corpus before any C1 training run.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_c1_preflight_report(config, REPO_ROOT)
    artifacts = config["foundation_state_c1"]["artifacts"]
    schema_path = safe_repo_path(REPO_ROOT, str(artifacts["preflight_schema_path"]))
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "C1 preflight report violates its schema: "
            + "; ".join(error.message for error in errors)
        )
    if report["blocker_count"] != len(report["blockers"]):
        raise ValueError("C1 preflight blocker_count does not match the blocker ledger")
    configured_output = artifacts["preflight_output_path"]
    output_path = resolve_repo_path(args.output or configured_output)
    write_json(output_path, report)
    print(f"c1_foundation_preflight_status={report['status']}")
    print(f"c1_foundation_preflight_blocker_count={report['blocker_count']}")
    print(f"c1_foundation_corpus_hash={report['corpus_hash']}")
    print(f"c1_foundation_preflight_output={output_path.relative_to(REPO_ROOT).as_posix()}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
