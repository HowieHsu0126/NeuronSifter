from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_store import read_zarr_array_metadata
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.microenvironment_teacher import TEACHER_ROLE_VALUES, teacher_contract_version
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the AD full microenvironment teacher registry.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"), None)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def validate_optional_header(path: Path, expected_fields: list[str]) -> None:
    if path.exists():
        validate_header(path, expected_fields)


def iter_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def validate_store_metadata(store_path_value: str, validated_store_paths: set[str]) -> None:
    if store_path_value in validated_store_paths:
        return
    store_path = REPO_ROOT / store_path_value
    trajectory_metadata = read_zarr_array_metadata(store_path, "trajectories")
    uncertainty_metadata = read_zarr_array_metadata(store_path, "teacher_uncertainty")
    if len(trajectory_metadata["shape"]) != 5:
        raise ValueError(f"Trajectory store has invalid rank: {trajectory_metadata}")
    if len(uncertainty_metadata["shape"]) != 2:
        raise ValueError(f"Uncertainty store has invalid rank: {uncertainty_metadata}")
    validated_store_paths.add(store_path_value)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    root = resolve_repo_path(registry["local_root"])
    relative_paths = registry["relative_paths"]
    teacher_path = root / relative_paths["full_microenvironment_teacher_catalog"]
    validate_header(teacher_path, config["schemas"]["full_microenvironment_teacher_fields"])
    validate_header(root / relative_paths["source_model_catalog"], config["schemas"]["teacher_source_model_fields"])
    validate_optional_header(root / relative_paths["pbpk_rollout_catalog"], config["schemas"]["pbpk_rollout_fields"])
    validate_optional_header(root / relative_paths["chembl_pbpk_parameter_catalog"], config["schemas"]["chembl_pbpk_parameter_fields"])
    validate_optional_header(root / relative_paths["neuron_rxd_protocol_catalog"], config["schemas"]["neuron_rxd_protocol_fields"])
    row_count = 0
    trajectory_backed_row_count = 0
    validated_store_paths: set[str] = set()
    contract_version = teacher_contract_version(config)
    for row in iter_tsv(teacher_path):
        row_count += 1
        if row["contract_version"] != contract_version:
            raise ValueError(f"Teacher row contract mismatch: {row}")
        if row["teacher_role"] not in TEACHER_ROLE_VALUES:
            raise ValueError(f"Unknown teacher_role: {row}")
        if row["teacher_role"] == "legacy_baseline" and row["calibration_role"] != "legacy_baseline":
            raise ValueError(f"Legacy baseline row has invalid calibration role: {row}")
        if row["teacher_role"] != "legacy_baseline" and row["calibration_role"] == "compatibility_baseline":
            raise ValueError(f"Primary teacher cannot use compatibility_baseline calibration role: {row}")
        if row["teacher_evidence_level"] not in {"empirical_measured", "validated_mechanistic", "repository_generated", "configuration_generated_provisional", "legacy_compatibility"}:
            raise ValueError(f"Unknown teacher_evidence_level: {row}")
        if row["calibration_status"] not in {"external_calibrated", "repository_protocol_validated", "not_external_calibrated", "calibration_pending"}:
            raise ValueError(f"Unknown calibration_status: {row}")
        if row["main_pretraining_eligibility"] not in {"main_pretraining", "calibration_or_compatibility_only", "reject"}:
            raise ValueError(f"Unknown main_pretraining_eligibility: {row}")
        if row["teacher_family"] == "ad_full_microenvironment_main_dynamics" and row["calibration_status"] != "calibration_pending":
            raise ValueError(f"Main dynamics teacher must remain calibration_pending until external calibration passes: {row}")
        if row["trajectory_store_path"]:
            trajectory_backed_row_count += 1
            validate_store_metadata(row["trajectory_store_path"], validated_store_paths)
    if row_count == 0:
        raise ValueError("full_microenvironment_teacher_catalog.tsv must not be empty.")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=teacher_path,
        full_microenvironment_teacher_row_count=row_count,
        trajectory_backed_row_count=trajectory_backed_row_count,
        validated_trajectory_store_count=len(validated_store_paths),
    )


if __name__ == "__main__":
    main()
