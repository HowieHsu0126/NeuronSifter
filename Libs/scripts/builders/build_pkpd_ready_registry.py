from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_tsv
from Libs.utils.data.screening_pipeline import stable_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PK/PD-ready registry for the AD screening system.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def descriptor_lookup(rows: list[dict[str, str]], feature_name: str) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for row in rows:
        if row["feature_name"] != feature_name:
            continue
        lookup[row["compound_uid"]] = float(row["feature_value"])
    return lookup


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    pkpd_config = config["screening_pipeline"]["pkpd_ready_registry"]
    root = resolve_repo_path(pkpd_config["local_root"])
    ensure_directory(root)
    relative_paths = pkpd_config["relative_paths"]

    compartment_rows = list(pkpd_config["compartments"])
    route_rows = list(pkpd_config["routes"])
    schedule_rows = list(pkpd_config["schedules"])
    dose_rows = list(pkpd_config["dose_grid"])

    parameter_rows = []
    for row in pkpd_config["bootstrap_parameter_priors"]:
        parameter_rows.append(
            {
                "parameter_uid": stable_uid("pkpd_parameter", row["parameter_name"], row["compartment_id"]),
                **row,
            }
        )

    literature_rows = read_tsv(resolve_repo_path("Input/raw/literature_parameter_bank/downloads/parameter_records.tsv"))
    for row in literature_rows:
        parameter_rows.append(
            {
                "parameter_uid": stable_uid("literature_parameter", row["parameter_name"], row["compartment"]),
                "parameter_name": row["parameter_name"],
                "compartment_id": row["compartment"],
                "unit": row["unit"],
                "lower_bound": row["lower_bound"],
                "upper_bound": row["upper_bound"],
                "distribution_family": row["distribution_family"],
                "central_value": row["central_value_if_reported"],
                "source_repository": "literature_parameter_bank",
                "provenance_level": row["license_status"] or "publication_curated",
                "note": row["citation"],
            }
        )

    compound_rows = read_tsv(root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"])
    compound_feature_rows = read_tsv(root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_feature_catalog"])
    pipeline_logger.progress(
        "loaded_pkpd_inputs",
        literature_parameter_row_count=len(literature_rows),
        compound_row_count=len(compound_rows),
        compound_feature_row_count=len(compound_feature_rows),
    )
    logp_lookup = descriptor_lookup(compound_feature_rows, "alogp")
    formula = pkpd_config["compound_prior_formula"]
    compound_pk_rows = []
    for compound_row in compound_rows:
        logp_value = logp_lookup.get(compound_row["compound_uid"])
        if logp_value is None:
            permeability_score = float(formula["default_permeability_score"])
        else:
            centered = (float(logp_value) - float(formula["logp_center"])) / float(formula["logp_scale"])
            permeability_score = max(0.0, min(1.0, 0.5 + centered))
        for route_row in route_rows:
            compound_pk_rows.append(
                {
                    "compound_uid": compound_row["compound_uid"],
                    "canonical_compound_id": compound_row["canonical_compound_id"],
                    "route": route_row["route"],
                    "permeability_score": permeability_score,
                    "exposure_scale": float(formula["default_exposure_scale"]) * float(route_row["absorption_scale"]),
                    "clearance_scale": float(formula["default_clearance_scale"]) / max(float(route_row["csf_scale"]), 1e-6),
                    "source_repository": "screening_pipeline",
                    "note": "descriptor_bootstrap",
                }
            )

    write_tsv(root / relative_paths["compartment_catalog"], config["schemas"]["pkpd_compartment_fields"], compartment_rows)
    write_tsv(root / relative_paths["parameter_prior_catalog"], config["schemas"]["pkpd_parameter_prior_fields"], parameter_rows)
    write_tsv(root / relative_paths["compound_pk_prior_catalog"], config["schemas"]["pkpd_compound_prior_fields"], compound_pk_rows)
    write_tsv(root / relative_paths["route_catalog"], config["schemas"]["pkpd_route_fields"], route_rows)
    write_tsv(root / relative_paths["schedule_catalog"], config["schemas"]["pkpd_schedule_fields"], schedule_rows)
    write_tsv(root / relative_paths["dose_grid_catalog"], config["schemas"]["pkpd_dose_grid_fields"], dose_rows)
    pipeline_logger.artifact(
        "pkpd_registry_written",
        artifact=root,
        compartment_row_count=len(compartment_rows),
        parameter_row_count=len(parameter_rows),
        compound_pk_row_count=len(compound_pk_rows),
        route_row_count=len(route_rows),
        schedule_row_count=len(schedule_rows),
        dose_row_count=len(dose_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=root,
        compartment_row_count=len(compartment_rows),
        parameter_row_count=len(parameter_rows),
        compound_pk_row_count=len(compound_pk_rows),
        route_row_count=len(route_rows),
        schedule_row_count=len(schedule_rows),
        dose_row_count=len(dose_rows),
    )

if __name__ == "__main__":
    main()
