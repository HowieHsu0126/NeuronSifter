"""Validate the exact non-materializing Round-11 state-contract audit set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.digital_twin_state_audits import validate_round11_audits
from Libs.utils.data.raw_data_registry import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-11 DigitalTwinStateBlock contract audits.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    args = parser.parse_args()
    report = validate_round11_audits(load_config(args.config), repository_root=REPO_ROOT)
    print(f"{report['status']} blockers={report['blocker_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
