from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.evaluation.active_learning_loop import round_path, select_request_batch, write_frame
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select active-learning simulation/tool request batch.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--candidate-pool", default="")
    parser.add_argument("--acquisition-scores", default="")
    parser.add_argument("--task-count-override", default="")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    candidate_path = resolve_repo_path(args.candidate_pool) if args.candidate_pool else round_path(config, "candidate_pool_template", args.round_uid)
    acquisition_path = resolve_repo_path(args.acquisition_scores) if args.acquisition_scores else round_path(config, "acquisition_scores_template", args.round_uid)
    candidate_frame = pd.read_csv(candidate_path, sep="\t")
    acquisition_frame = pd.read_csv(acquisition_path, sep="\t")
    task_count = int(args.task_count_override) if args.task_count_override else int(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["full_tasks_per_round"])
    request_frame = select_request_batch(args.round_uid, acquisition_frame, candidate_frame, config, task_count)
    output_path = resolve_repo_path(args.output_path) if args.output_path else round_path(config, "request_batch_template", args.round_uid)
    write_frame(output_path, config["schemas"]["active_learning_simulation_request_fields"], request_frame)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        round_uid=args.round_uid,
        request_count=int(request_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
