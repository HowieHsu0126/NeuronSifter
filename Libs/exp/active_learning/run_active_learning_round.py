from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.active_learning.build_active_candidate_pool import default_priority_path
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path, write_json
from Libs.utils.evaluation.active_learning_loop import (
    build_active_candidate_pool,
    ingest_active_learning_results,
    round_path,
    score_acquisition_tasks,
    select_request_batch,
    write_frame,
    write_round_report,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one NeuronSifter active-learning round.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--priority-catalog", default="")
    parser.add_argument("--task-count-override", default="")
    parser.add_argument("--result-manifest", default="")
    return parser.parse_args()


def load_round_inputs(config: dict, priority_catalog: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervention_config = config["twin"]["intervention_program_registry"]
    intervention_path = (
        resolve_repo_path(str(intervention_config["local_root"]))
        / str(intervention_config["relative_paths"]["intervention_program_catalog"])
    )
    validation_config = config["validation"]["candidate_validation_registry"]
    validation_path = (
        resolve_repo_path(str(validation_config["local_root"]))
        / str(validation_config["relative_paths"]["candidate_validation_catalog"])
    )
    priority_path = resolve_repo_path(priority_catalog) if priority_catalog else default_priority_path(config)
    if not priority_path.exists():
        raise FileNotFoundError(f"Active-learning priority catalog is required: {priority_path}")
    return (
        pd.DataFrame(read_tsv(intervention_path)),
        pd.DataFrame(read_tsv(validation_path)) if validation_path.exists() else pd.DataFrame(),
        pd.read_csv(priority_path, sep="\t"),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    intervention_frame, validation_frame, priority_frame = load_round_inputs(config, args.priority_catalog)
    candidate_frame = build_active_candidate_pool(args.round_uid, intervention_frame, validation_frame, priority_frame, config)
    acquisition_frame = score_acquisition_tasks(args.round_uid, candidate_frame, config)
    task_count = int(args.task_count_override) if args.task_count_override else int(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["full_tasks_per_round"])
    request_frame = select_request_batch(args.round_uid, acquisition_frame, candidate_frame, config, task_count)
    write_frame(round_path(config, "candidate_pool_template", args.round_uid), config["schemas"]["active_learning_candidate_pool_fields"], candidate_frame)
    write_frame(round_path(config, "acquisition_scores_template", args.round_uid), config["schemas"]["active_learning_acquisition_fields"], acquisition_frame)
    write_frame(round_path(config, "request_batch_template", args.round_uid), config["schemas"]["active_learning_simulation_request_fields"], request_frame)
    if args.result_manifest:
        raw_result_frame = pd.read_csv(resolve_repo_path(args.result_manifest), sep="\t")
        result_frame, teacher_update_frame, qc_summary = ingest_active_learning_results(args.round_uid, request_frame, raw_result_frame, config)
        write_frame(round_path(config, "result_manifest_template", args.round_uid), config["schemas"]["active_learning_simulation_result_fields"], result_frame)
        write_frame(round_path(config, "teacher_update_template", args.round_uid), config["schemas"]["active_learning_teacher_update_fields"], teacher_update_frame)
        write_json(round_path(config, "qc_summary_template", args.round_uid), qc_summary)
        report_path = write_round_report(config, args.round_uid)
    else:
        report_path = round_path(config, "request_batch_template", args.round_uid)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=report_path,
        round_uid=args.round_uid,
        request_count=int(request_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
