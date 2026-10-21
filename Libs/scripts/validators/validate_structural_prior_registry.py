from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.data.structural_prior_registry import load_structural_prior_catalogs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the structural prior registry.")
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
    root = resolve_repo_path(config["structural_prior_registry"]["local_root"])
    relative_paths = config["structural_prior_registry"]["relative_paths"]
    region_catalog_path = root / relative_paths["region_prior_catalog"]
    head_catalog_path = root / relative_paths["local_response_head_catalog"]
    validate_header(region_catalog_path, config["schemas"]["structural_prior_region_fields"])
    validate_header(head_catalog_path, config["schemas"]["structural_prior_local_response_head_fields"])
    region_rows, head_rows = load_structural_prior_catalogs(config)
    if not region_rows:
        raise ValueError("Structural prior region catalog is empty.")
    if not head_rows:
        raise ValueError("Structural prior local response head catalog is empty.")
    region_ids = {
        str(row["region_id"])
        for row in config["spatial_atlas_registry"]["regions"]
    }
    disease_programs = {
        str(value)
        for value in config["structural_prior_registry"]["disease_programs"]
    }
    region_pairs = {
        (str(row["disease_program"]), str(row["region_id"]))
        for row in region_rows
    }
    for disease_program in disease_programs:
        for region_id in region_ids:
            if (disease_program, region_id) not in region_pairs:
                raise ValueError(f"Missing structural prior row for {disease_program}:{region_id}")
    for row in region_rows:
        if str(row["region_id"]) not in region_ids:
            raise ValueError(f"Unknown region in structural prior catalog: {row}")
        if str(row["disease_program"]) not in disease_programs:
            raise ValueError(f"Unknown disease program in structural prior catalog: {row}")
    for row in head_rows:
        if str(row["region_id"]) not in region_ids:
            raise ValueError(f"Unknown region in local response head catalog: {row}")
        if str(row["disease_program"]) not in disease_programs:
            raise ValueError(f"Unknown disease program in local response head catalog: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=region_catalog_path,
        region_prior_row_count=len(region_rows),
        local_response_head_row_count=len(head_rows),
    )


if __name__ == "__main__":
    main()
