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
    parser = argparse.ArgumentParser(description="Validate the PK/PD-ready registry outputs.")
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
    root = resolve_repo_path(config["screening_pipeline"]["pkpd_ready_registry"]["local_root"])
    relative_paths = config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]
    path_map = {
        "compartment_catalog": root / relative_paths["compartment_catalog"],
        "parameter_prior_catalog": root / relative_paths["parameter_prior_catalog"],
        "compound_pk_prior_catalog": root / relative_paths["compound_pk_prior_catalog"],
        "route_catalog": root / relative_paths["route_catalog"],
        "schedule_catalog": root / relative_paths["schedule_catalog"],
        "dose_grid_catalog": root / relative_paths["dose_grid_catalog"],
    }
    for path in path_map.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing pkpd-ready output: {path}")

    validate_header(path_map["compartment_catalog"], config["schemas"]["pkpd_compartment_fields"])
    validate_header(path_map["parameter_prior_catalog"], config["schemas"]["pkpd_parameter_prior_fields"])
    validate_header(path_map["compound_pk_prior_catalog"], config["schemas"]["pkpd_compound_prior_fields"])
    validate_header(path_map["route_catalog"], config["schemas"]["pkpd_route_fields"])
    validate_header(path_map["schedule_catalog"], config["schemas"]["pkpd_schedule_fields"])
    validate_header(path_map["dose_grid_catalog"], config["schemas"]["pkpd_dose_grid_fields"])

    compartment_rows = read_tsv(path_map["compartment_catalog"])
    parameter_rows = read_tsv(path_map["parameter_prior_catalog"])
    compound_rows = read_tsv(path_map["compound_pk_prior_catalog"])
    route_rows = read_tsv(path_map["route_catalog"])
    schedule_rows = read_tsv(path_map["schedule_catalog"])
    dose_rows = read_tsv(path_map["dose_grid_catalog"])
    pipeline_logger.progress(
        "loaded_pkpd_inputs",
        compartment_row_count=len(compartment_rows),
        parameter_row_count=len(parameter_rows),
        compound_pk_row_count=len(compound_rows),
        route_row_count=len(route_rows),
        schedule_row_count=len(schedule_rows),
        dose_row_count=len(dose_rows),
    )

    compartment_ids = {row["compartment_id"] for row in compartment_rows}
    route_names = {row["route"] for row in route_rows}
    if len(compartment_ids) != 6:
        raise ValueError("PK/PD compartment catalog must contain exactly the six configured compartments.")
    if route_names != {"oral", "intravenous", "intrathecal"}:
        raise ValueError(f"Unexpected route set: {route_names}")
    if {row["schedule"] for row in schedule_rows} != {"daily", "twice_daily", "weekly"}:
        raise ValueError("Unexpected schedule set in PK/PD registry.")
    if {row["dose_multiplier"] for row in dose_rows} != {"0.25", "0.5", "1.0", "2.0"}:
        raise ValueError("Unexpected dose multiplier set in PK/PD registry.")

    for row in parameter_rows:
        if row["compartment_id"] not in compartment_ids:
            raise ValueError(f"Parameter prior references unknown compartment: {row['parameter_uid']}")
    for row in compound_rows:
        if row["route"] not in route_names:
            raise ValueError(f"Compound PK row references unknown route: {row}")
        permeability_score = float(row["permeability_score"])
        if not 0.0 <= permeability_score <= 1.0:
            raise ValueError(f"permeability_score out of range: {row}")
        if float(row["exposure_scale"]) < 0 or float(row["clearance_scale"]) < 0:
            raise ValueError(f"Negative PK scale detected: {row}")

    pipeline_logger.progress(
        "pkpd_ready_validated",
        artifact=path_map["compound_pk_prior_catalog"],
        compartment_row_count=len(compartment_rows),
        parameter_row_count=len(parameter_rows),
        compound_pk_row_count=len(compound_rows),
        route_row_count=len(route_rows),
        schedule_row_count=len(schedule_rows),
        dose_row_count=len(dose_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path_map["compound_pk_prior_catalog"],
        compartment_row_count=len(compartment_rows),
        parameter_row_count=len(parameter_rows),
        compound_pk_row_count=len(compound_rows),
        route_row_count=len(route_rows),
        schedule_row_count=len(schedule_rows),
        dose_row_count=len(dose_rows),
    )

if __name__ == "__main__":
    main()
