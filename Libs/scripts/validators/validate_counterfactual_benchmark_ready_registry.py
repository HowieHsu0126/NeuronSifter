from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the counterfactual benchmark registry catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
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
    root = resolve_repo_path(config["data"]["counterfactual_benchmark_ready_registry"]["local_root"])
    path = root / config["data"]["counterfactual_benchmark_ready_registry"]["relative_paths"]["counterfactual_benchmark_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing counterfactual benchmark catalog: {path}")
    validate_header(path, config["schemas"]["counterfactual_benchmark_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("counterfactual_benchmark_catalog.tsv must not be empty.")
    seen_uids = {row["counterfactual_benchmark_uid"] for row in rows}
    if len(seen_uids) != len(rows):
        raise ValueError("counterfactual_benchmark_catalog.tsv contains duplicate counterfactual_benchmark_uid values.")
    for row in rows:
        if row["baseline_regimen_uid"] == row["counterfactual_regimen_uid"]:
            raise ValueError(f"Counterfactual benchmark row must compare distinct regimens: {row}")
        if not row["validity_contract"]:
            raise ValueError(f"Counterfactual benchmark row is missing validity contract: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        counterfactual_benchmark_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
