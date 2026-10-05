from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json
from Libs.utils.evaluation.active_learning_loop import ingest_active_learning_results, round_path, write_frame
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest completed active-learning simulation/tool results.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--request-batch", default="")
    parser.add_argument("--result-manifest", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    request_path = resolve_repo_path(args.request_batch) if args.request_batch else round_path(config, "request_batch_template", args.round_uid)
    result_path = resolve_repo_path(args.result_manifest) if args.result_manifest else round_path(config, "completed_tool_result_template", args.round_uid)
    request_frame = pd.read_csv(request_path, sep="\t")
    raw_result_frame = pd.read_csv(result_path, sep="\t")
    result_frame, teacher_update_frame, qc_summary = ingest_active_learning_results(args.round_uid, request_frame, raw_result_frame, config)
    result_output_path = round_path(config, "result_manifest_template", args.round_uid)
    teacher_output_path = round_path(config, "teacher_update_template", args.round_uid)
    qc_summary_path = round_path(config, "qc_summary_template", args.round_uid)
    write_frame(result_output_path, config["schemas"]["active_learning_simulation_result_fields"], result_frame)
    write_frame(teacher_output_path, config["schemas"]["active_learning_teacher_update_fields"], teacher_update_frame)
    write_json(qc_summary_path, qc_summary)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=result_output_path,
        round_uid=args.round_uid,
        result_count=int(result_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
