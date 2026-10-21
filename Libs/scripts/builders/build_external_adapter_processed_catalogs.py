from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.external_adapter_materialization import (
    build_dgidb_processed_catalog,
    build_drugcomb_processed_catalogs,
    build_moleculenet_deepchem_baseline_catalog,
    external_adapter_materialization_config,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize processed external adapter catalogs for strict M2 active-learning deployment.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--source", choices=["all", "dgidb", "drugcomb", "moleculenet_deepchem"], default="all")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    artifacts: list[str] = []
    if args.source in {"all", "dgidb"}:
        artifacts.append(build_dgidb_processed_catalog(config).as_posix())
    if args.source in {"all", "drugcomb"}:
        max_rows = None
        if bool(args.smoke):
            max_rows = int(external_adapter_materialization_config(config, "drugcomb")["smoke_row_count"])
        prior_path, matrix_dir = build_drugcomb_processed_catalogs(config, max_dose_response_rows=max_rows)
        artifacts.extend([prior_path.as_posix(), matrix_dir.as_posix()])
    if args.source in {"all", "moleculenet_deepchem"}:
        artifacts.append(build_moleculenet_deepchem_baseline_catalog(config).as_posix())
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact="|".join(artifacts),
        source=args.source,
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
