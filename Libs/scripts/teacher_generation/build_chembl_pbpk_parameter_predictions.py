from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.microenvironment_teacher import json_payload, teacher_contract_version
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_tsv
from Libs.utils.model.screening_core import stable_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChEMBL/TDC/B3DB PBPK parameter predictions with uncertainty.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def numeric_feature(row: dict[str, str], feature_name: str) -> float:
    return float(row[feature_name])


def predicted_value(row: dict[str, str], parameter_name: str, parameter_config: dict[str, object]) -> float:
    coefficients = dict(parameter_config["descriptor_coefficients"][parameter_name])
    value = float(coefficients["intercept"])
    for descriptor_name in parameter_config["molecular_descriptor_fields"]:
        value += float(coefficients[descriptor_name]) * numeric_feature(row, str(descriptor_name))
    bounds = dict(parameter_config["parameter_bounds"][parameter_name])
    return max(float(bounds["lower"]), min(float(bounds["upper"]), value))


def prediction_uncertainty(row: dict[str, str], parameter_config: dict[str, object]) -> float:
    descriptor_values = [numeric_feature(row, str(descriptor_name)) for descriptor_name in parameter_config["molecular_descriptor_fields"]]
    descriptor_span = max(descriptor_values) - min(descriptor_values)
    return float(parameter_config["uncertainty_floor"]) + float(parameter_config["uncertainty_scale"]) * descriptor_span


def build_rows(config: dict, compound_pk_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parameter_config = config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]
    rows: list[dict[str, object]] = []
    for compound_row in compound_pk_rows:
        feature_payload = {
            descriptor_name: numeric_feature(compound_row, str(descriptor_name))
            for descriptor_name in parameter_config["molecular_descriptor_fields"]
        }
        for parameter_name, bounds_payload in dict(parameter_config["parameter_bounds"]).items():
            uncertainty = prediction_uncertainty(compound_row, parameter_config)
            rows.append(
                {
                    "parameter_prediction_uid": stable_uid(
                        "chembl_pbpk_parameter",
                        str(compound_row["compound_uid"]),
                        str(compound_row["route"]),
                        str(parameter_name),
                    ),
                    "contract_version": teacher_contract_version(config),
                    "compound_uid": compound_row["compound_uid"],
                    "canonical_compound_id": compound_row["canonical_compound_id"],
                    "parameter_name": parameter_name,
                    "predicted_value": predicted_value(compound_row, str(parameter_name), parameter_config),
                    "lower_bound": float(bounds_payload["lower"]),
                    "upper_bound": float(bounds_payload["upper"]),
                    "uncertainty": uncertainty,
                    "unit": str(bounds_payload["unit"]),
                    "feature_payload": json_payload(feature_payload),
                    "source_repository": parameter_config["source_repository"],
                    "provenance_level": parameter_config["provenance_level"],
                    "teacher_role": parameter_config["teacher_role"],
                    "observed_or_synthetic": "parameter_predicted",
                    "calibration_role": "synthetic_teacher_loss_weighted_by_uncertainty",
                    "note": "AD-first PBPK parameter prediction from ChEMBL/TDC/B3DB-ready molecular descriptors",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    root = resolve_repo_path(registry["local_root"])
    relative_paths = registry["relative_paths"]
    parameter_config = config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]
    source_path = resolve_repo_path(parameter_config["feature_source_catalog"])
    compound_pk_rows = read_tsv(source_path)
    rows = build_rows(config, compound_pk_rows)
    output_path = root / relative_paths["chembl_pbpk_parameter_catalog"]
    ensure_directory(output_path.parent)
    write_tsv(output_path, config["schemas"]["chembl_pbpk_parameter_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        parameter_prediction_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
