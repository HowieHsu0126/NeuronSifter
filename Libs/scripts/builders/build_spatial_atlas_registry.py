from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_fields import (
    build_spatial_observation_mapping_rows,
    build_spatial_region_compartment_rows,
    build_spatial_region_edge_rows,
    build_spatial_region_rows,
    spatial_atlas_root,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public mesoscale spatial atlas registry.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = spatial_atlas_root(config)
    relative_paths = config["spatial_atlas_registry"]["relative_paths"]
    ensure_directory(root)

    region_rows = build_spatial_region_rows(config)
    edge_rows = build_spatial_region_edge_rows(config)
    region_compartment_rows = build_spatial_region_compartment_rows(config)
    mapping_rows = build_spatial_observation_mapping_rows(config)

    write_tsv(root / relative_paths["region_catalog"], config["schemas"]["spatial_region_fields"], region_rows)
    write_tsv(root / relative_paths["region_edge_catalog"], config["schemas"]["spatial_region_edge_fields"], edge_rows)
    write_tsv(
        root / relative_paths["region_compartment_catalog"],
        config["schemas"]["spatial_region_compartment_fields"],
        region_compartment_rows,
    )
    write_tsv(
        root / relative_paths["observation_mapping_catalog"],
        config["schemas"]["spatial_observation_mapping_fields"],
        mapping_rows,
    )
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
