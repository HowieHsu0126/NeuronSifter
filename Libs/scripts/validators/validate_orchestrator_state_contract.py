from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.iclr_contract_union import build_orchestrator_state_report, resolve_repo_path
from Libs.utils.data.raw_data_registry import load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate controller-owned orchestrator state without mutation.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_orchestrator_state_report(config)
    report["producer_command"] = f"python Libs/scripts/validators/validate_orchestrator_state_contract.py --config {args.config}"
    output_path = resolve_repo_path(config["iclr_design_contract_union"]["audit_contract"]["output_paths"]["orchestrator_state"])
    write_json(output_path, report)
    print(f"orchestrator_state_contract_status={report['status']}")
    print(f"orchestrator_state_contract_blocker_count={report['blocker_count']}")
    print(f"orchestrator_state_contract_mutated={str(report['state_mutated']).lower()}")
    print(f"orchestrator_state_contract_output={output_path.relative_to(REPO_ROOT).as_posix()}")
    if report["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
