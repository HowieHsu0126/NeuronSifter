from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_mainline_isolation import build_public_mainline_isolation_reports
from Libs.utils.data.raw_data_registry import load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate public-mainline isolation from controlled sources.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph, audit, remediation = build_public_mainline_isolation_reports(args.config)
    config = load_config(args.config)
    output_paths = config["public_mainline_isolation"]["output_paths"]
    write_json(REPO_ROOT / output_paths["config_graph"], graph)
    write_json(REPO_ROOT / output_paths["isolation_audit"], audit)
    write_json(REPO_ROOT / output_paths["remediation_report"], remediation)
    print(f"public_mainline_isolation_status={audit['status']}")
    print(f"public_mainline_controlled_reachability_count={audit['controlled_reachability_count']}")
    print(f"public_mainline_isolation_blocker_count={audit['blocker_count']}")
    if audit["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
