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
    parser = argparse.ArgumentParser(description="Validate the intervention-program contract catalog.")
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
    root = resolve_repo_path(config["twin"]["intervention_program_registry"]["local_root"])
    path = root / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing intervention-program catalog: {path}")
    validate_header(path, config["schemas"]["intervention_program_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("intervention_program_catalog.tsv must not be empty.")
    seen_uids = {row["intervention_program_uid"] for row in rows}
    if len(seen_uids) != len(rows):
        raise ValueError("intervention_program_catalog.tsv contains duplicate intervention_program_uid values.")
    for row in rows:
        payload = json.loads(row["occupancy_constraints"])
        if "occupancy_target_by_compartment" not in payload:
            raise ValueError(f"Missing occupancy_target_by_compartment payload: {row}")
        if float(row["dose_multiplier"]) <= 0.0:
            raise ValueError(f"Non-positive dose_multiplier in intervention program: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        intervention_program_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
