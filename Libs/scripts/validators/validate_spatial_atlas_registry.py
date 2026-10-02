from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_fields import load_spatial_atlas_catalogs, spatial_atlas_root
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the public mesoscale spatial atlas registry.")
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
    root = spatial_atlas_root(config)
    relative_paths = config["spatial_atlas_registry"]["relative_paths"]
    validate_header(root / relative_paths["region_catalog"], config["schemas"]["spatial_region_fields"])
    validate_header(root / relative_paths["region_edge_catalog"], config["schemas"]["spatial_region_edge_fields"])
    validate_header(root / relative_paths["region_compartment_catalog"], config["schemas"]["spatial_region_compartment_fields"])
    validate_header(root / relative_paths["observation_mapping_catalog"], config["schemas"]["spatial_observation_mapping_fields"])
    region_rows, edge_rows, region_compartment_rows, mapping_rows = load_spatial_atlas_catalogs(config)
    region_ids = {row["region_id"] for row in region_rows}
    if "middle_temporal_gyrus" not in region_ids:
        raise ValueError("Spatial atlas must contain middle_temporal_gyrus.")
    if "hippocampus_ca1" not in region_ids:
        raise ValueError("Spatial atlas must contain hippocampus_ca1.")
    for edge_row in edge_rows:
        if edge_row["source_region_id"] not in region_ids or edge_row["target_region_id"] not in region_ids:
            raise ValueError(f"Spatial edge references an unknown region: {edge_row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=root,
        region_row_count=len(region_rows),
        edge_row_count=len(edge_rows),
        region_compartment_row_count=len(region_compartment_rows),
        mapping_row_count=len(mapping_rows),
    )


if __name__ == "__main__":
    main()
