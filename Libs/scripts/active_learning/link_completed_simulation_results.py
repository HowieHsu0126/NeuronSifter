from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.evaluation.active_learning_loop import build_completed_result_manifest, round_path, write_frame
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link completed active-learning tool/simulator artifacts into the result manifest schema.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--request-batch", default="")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    request_path = resolve_repo_path(args.request_batch) if args.request_batch else round_path(config, "request_batch_template", args.round_uid)
    request_frame = pd.read_csv(request_path, sep="\t")
    result_frame = build_completed_result_manifest(args.round_uid, request_frame, config)
    output_path = resolve_repo_path(args.output_path) if args.output_path else round_path(config, "completed_tool_result_template", args.round_uid)
    write_frame(output_path, config["schemas"]["active_learning_simulation_result_fields"], result_frame)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        round_uid=args.round_uid,
        result_count=int(result_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
