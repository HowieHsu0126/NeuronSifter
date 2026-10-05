from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.iclr_contract_union import build_config_stack_contract_report, resolve_repo_path
from Libs.utils.data.raw_data_registry import load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical config-stack composition and ownership.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_config_stack_contract_report(args.config)
    report["producer_command"] = f"python Libs/scripts/validators/validate_config_stack_contract.py --config {args.config}"
    output_path = resolve_repo_path(config["iclr_design_contract_union"]["audit_contract"]["output_paths"]["config_stack"])
    write_json(output_path, report)
    print(f"config_stack_contract_status={report['status']}")
    print(f"config_stack_contract_blocker_count={report['blocker_count']}")
    print(f"config_stack_contract_output={output_path.relative_to(REPO_ROOT).as_posix()}")
    if report["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
