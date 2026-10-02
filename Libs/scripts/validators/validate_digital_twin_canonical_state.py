"""Validate one serialized partial canonical-state block without writing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.digital_twin_canonical_state import CanonicalStateContractError, load_serialized_state_block, state_block_validation_report
from Libs.utils.data.raw_data_registry import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a serialized Digital Twin canonical partial-state block.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--state-block", required=True, help="Path to one serialized state-block JSON payload.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        state_path = Path(args.state_block)
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        report = state_block_validation_report(config, payload)
        if report["status"] != "passed":
            print(json.dumps(report, sort_keys=True, ensure_ascii=True))
            return 1
        # Perform strict readback separately so the command exercises the file boundary.
        load_serialized_state_block(config, state_path)
    except (CanonicalStateContractError, OSError, json.JSONDecodeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "blocker": {"code": "validator_input_error", "message": str(error)},
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
