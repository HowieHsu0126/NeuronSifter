from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the feedback-event registry catalog for online continual learning.")
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
    root = resolve_repo_path(config["data"]["feedback_ready_registry"]["local_root"])
    path = root / config["data"]["feedback_ready_registry"]["relative_paths"]["feedback_event_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing feedback-event catalog: {path}")
    validate_header(path, config["schemas"]["feedback_event_fields"])
    rows = read_tsv(path)
    for row in rows:
        json.loads(row["predicted_rollout_summary"])
        json.loads(row["realized_outcome_summary"])
        json.loads(row["discrepancy_summary"])
        if row["update_eligibility_flag"] not in {"yes", "no"}:
            raise ValueError(f"Feedback event has invalid update_eligibility_flag: {row}")
        if not row["update_decision"]:
            raise ValueError(f"Feedback event is missing update_decision: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        feedback_event_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
