from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.active_learning_loop import active_learning_config, quota_counts, round_path
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate active-learning round artifacts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--require-results", action="store_true")
    return parser.parse_args()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_header(path: Path, fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    if header != fields:
        raise ValueError(f"{path} header mismatch: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    schema_pairs = [
        ("candidate_pool_template", "active_learning_candidate_pool_fields"),
        ("acquisition_scores_template", "active_learning_acquisition_fields"),
        ("request_batch_template", "active_learning_simulation_request_fields"),
    ]
    if bool(args.require_results):
        schema_pairs.extend(
            [
                ("result_manifest_template", "active_learning_simulation_result_fields"),
                ("teacher_update_template", "active_learning_teacher_update_fields"),
            ]
        )
    artifact_rows: dict[str, int] = {}
    for path_key, schema_key in schema_pairs:
        path = round_path(config, path_key, args.round_uid)
        if not path.exists():
            raise FileNotFoundError(f"Missing active-learning round artifact: {path}")
        validate_header(path, config["schemas"][schema_key])
        artifact_rows[path_key] = len(read_tsv_rows(path))
    request_path = round_path(config, "request_batch_template", args.round_uid)
    if request_path.exists():
        request_rows = read_tsv_rows(request_path)
        expected_count = int(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["full_tasks_per_round"])
        if len(request_rows) not in {
            expected_count,
            int(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["smoke_tasks_per_round"]),
        }:
            raise ValueError(f"Unexpected active-learning request count: {len(request_rows)}")
        role_counts: dict[str, int] = {}
        for row in request_rows:
            role_counts[str(row["budget_class"])] = role_counts.get(str(row["budget_class"]), 0) + 1
        if len(request_rows) == expected_count:
            expected_quota = quota_counts(expected_count, active_learning_config(config)["batch_quota"])
            if role_counts != expected_quota:
                raise ValueError(f"Active-learning request quota mismatch: expected {expected_quota}, found {role_counts}")
    report_path = round_path(config, "round_report_template", args.round_uid)
    if report_path.exists():
        json.loads(report_path.read_text(encoding="utf-8"))
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=request_path,
        round_uid=args.round_uid,
        artifact_rows=artifact_rows,
    )


if __name__ == "__main__":
    main()
