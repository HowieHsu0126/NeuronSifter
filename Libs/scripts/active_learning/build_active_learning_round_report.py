from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.active_learning_loop import write_round_report
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build active-learning round report.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    report_path = write_round_report(config, args.round_uid)
    complete_script_logging(pipeline_logger, pipeline_started_at, artifact=report_path, round_uid=args.round_uid)


if __name__ == "__main__":
    main()
