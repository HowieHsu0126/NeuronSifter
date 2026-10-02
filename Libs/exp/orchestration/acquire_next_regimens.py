from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, write_tsv
from Libs.utils.evaluation.regimen_acquisition import build_acquisition_frame, select_acquisition_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire the next retrospective regimen batch under the active-learning contract.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--candidate-catalog", required=True, help="TSV with regimen candidates and score columns.")
    parser.add_argument("--output-path", required=True, help="TSV path for the selected regimen batch.")
    parser.add_argument("--batch-size-override", default="", help="Optional batch size override.")
    return parser.parse_args()


def required_columns() -> list[str]:
    return [
        "regimen_uid",
        "target_symbol_set",
        "route",
        "schedule",
        "pk_feasibility_score",
        "predicted_score",
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    candidate_path = (REPO_ROOT / args.candidate_catalog).resolve() if not Path(args.candidate_catalog).is_absolute() else Path(args.candidate_catalog)
    output_path = (REPO_ROOT / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)
    frame = pd.read_csv(candidate_path, sep="\t", dtype=str)
    missing_columns = [column_name for column_name in required_columns() if column_name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Candidate catalog is missing required columns: {missing_columns}")
    predicted_scores = pd.to_numeric(frame["predicted_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if "epistemic_uncertainty" in frame.columns:
        epistemic_uncertainty = pd.to_numeric(frame["epistemic_uncertainty"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        epistemic_uncertainty = np.zeros((frame.shape[0],), dtype=np.float64)
    acquisition_frame = build_acquisition_frame(frame, predicted_scores, epistemic_uncertainty, config)
    batch_size = (
        int(args.batch_size_override)
        if args.batch_size_override
        else int(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["acquisition_batch_size"])
    )
    selected = select_acquisition_batch(acquisition_frame, batch_size)
    output_fields = list(selected.columns)
    write_tsv(output_path, output_fields, selected.to_dict(orient="records"))
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        candidate_count=int(frame.shape[0]),
        selected_count=int(selected.shape[0]),
    )


if __name__ == "__main__":
    main()
