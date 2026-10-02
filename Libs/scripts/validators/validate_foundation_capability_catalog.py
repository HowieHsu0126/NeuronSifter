from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the foundation capability catalog emitted by screening evaluation.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--benchmark-name", default="", help="Optional benchmark name override.")
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    benchmark_name = args.benchmark_name or str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["capability_catalog"]
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing foundation capability catalog: {path}")
    validate_header(path, config["schemas"]["foundation_capability_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("foundation_capability_catalog.tsv must not be empty.")
    for row in rows:
        if row["pass_flag"] not in {"yes", "no"}:
            raise ValueError(f"Foundation capability row has invalid pass_flag: {row}")
        if row["comparator"] not in {">=", "<=", "=="}:
            raise ValueError(f"Foundation capability row has invalid comparator: {row}")
        float(row["metric_value"])
        float(row["threshold_value"])
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        foundation_capability_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
