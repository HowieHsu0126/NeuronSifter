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
    parser = argparse.ArgumentParser(description="Validate the candidate-validation contract catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["validation"]["candidate_validation_registry"]["local_root"])
    path = root / config["validation"]["candidate_validation_registry"]["relative_paths"]["candidate_validation_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate-validation catalog: {path}")
    validate_header(path, config["schemas"]["candidate_validation_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("candidate_validation_catalog.tsv must not be empty.")
    seen_uids = {row["candidate_validation_uid"] for row in rows}
    if len(seen_uids) != len(rows):
        raise ValueError("candidate_validation_catalog.tsv contains duplicate candidate_validation_uid values.")
    for row in rows:
        if row["affinity_evidence"]:
            json.loads(row["affinity_evidence"])
        if row["admet_summary"]:
            json.loads(row["admet_summary"])
        if row["pbpk_summary"]:
            json.loads(row["pbpk_summary"])
        if float(row["admet_readiness_score"]) < 0.0 or float(row["pbpk_readiness_score"]) < 0.0:
            raise ValueError(f"Readiness scores must be non-negative: {row}")
        if not row["validation_stage"]:
            raise ValueError(f"candidate_validation row is missing validation_stage: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        candidate_validation_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
