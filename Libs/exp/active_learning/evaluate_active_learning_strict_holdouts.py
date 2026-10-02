from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_tsv
from Libs.utils.evaluation.active_learning_loop import active_learning_config, round_path
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active-learning empirical deltas on strict holdouts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    rows: list[dict[str, object]] = []
    for round_config in active_learning_config(config)["active_rounds"]:
        round_uid = str(round_config["round_uid"])
        report_path = round_path(config, "round_report_template", round_uid)
        if not report_path.exists():
            raise FileNotFoundError(f"Strict-holdout evaluation requires active-learning round report: {report_path}")
        report = pd.read_json(report_path, typ="series").to_dict()
        for metric_name in [
            "source_holdout_rollout_error_delta",
            "top_k_rank_stability_delta",
            "tri_selective_risk_delta",
            "counterfactual_direction_hit_delta",
            "ood_false_accept_delta",
        ]:
            rows.append(
                {
                    "round_uid": round_uid,
                    "metric_name": metric_name,
                    "metric_value": str(report[metric_name]),
                    "bootstrap_interval": config["screening_pipeline"]["screening_benchmark"]["active_learning"]["empirical_acceptance"]["bootstrap_interval"],
                    "artifact_path": str(report_path),
                    "result_status": str(report["publication_artifact_status"]),
                }
            )
    output_path = Path(args.output_path).resolve() if args.output_path else round_path(config, "loop_summary", "unused").with_name("active_learning_strict_holdout_metrics.tsv")
    write_tsv(
        output_path,
        ["round_uid", "metric_name", "metric_value", "bootstrap_interval", "artifact_path", "result_status"],
        rows,
    )
    complete_script_logging(pipeline_logger, pipeline_started_at, artifact=output_path, metric_row_count=len(rows))


if __name__ == "__main__":
    main()
