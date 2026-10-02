from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.structural_prior_registry import (
    structural_prior_local_response_head_rows,
    structural_prior_region_rows,
    write_structural_prior_catalogs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the structural prior registry from human atlas and disease vulnerability priors.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    region_rows = structural_prior_region_rows(config)
    head_rows = structural_prior_local_response_head_rows(config)
    write_structural_prior_catalogs(config, region_rows, head_rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=Path(config["structural_prior_registry"]["output_root"]),
        region_prior_row_count=len(region_rows),
        local_response_head_row_count=len(head_rows),
    )


if __name__ == "__main__":
    main()
