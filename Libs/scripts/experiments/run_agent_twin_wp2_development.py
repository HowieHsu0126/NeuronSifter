"""Run the development-only Agent-Twin WP2 sequential qualification harness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.exp.agent_twin_wp2_development import DEFAULT_CONFIG, DEFAULT_SCHEMA, run_wp2_development, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run development-only Agent-Twin WP2 qualification.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    payload = validate_config(config, resolve_repo_path(args.schema))
    output_root = resolve_repo_path(args.output_root or str(payload["output_root"]))
    decision = run_wp2_development(payload, output_root)
    print({"output_root": output_root.as_posix(), "development_coevolution_status": decision["development_coevolution_status"]})


if __name__ == "__main__":
    main()
