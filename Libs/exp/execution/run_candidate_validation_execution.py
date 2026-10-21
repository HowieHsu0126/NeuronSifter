from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.candidate_validation_execution import build_execution_rows, write_candidate_validation_execution_catalog
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run candidate-validation execution for docking, rescoring, and PBPK.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execution-stage", default="all", choices=["prep", "docking", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    rows = build_execution_rows(config, dry_run=bool(args.dry_run), execution_stage=str(args.execution_stage))
    catalog_path = write_candidate_validation_execution_catalog(config, rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=catalog_path,
        candidate_validation_execution_row_count=len(rows),
        dry_run=bool(args.dry_run),
        execution_stage=str(args.execution_stage),
    )


if __name__ == "__main__":
    main()
