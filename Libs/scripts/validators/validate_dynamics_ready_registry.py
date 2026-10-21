from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_fields import load_all_trajectories, load_dynamics_catalogs
from Libs.utils.model.dynamics_store import read_store_manifest
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.data.structural_prior_registry import structural_prior_uid_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the public AD-first mesoscale dynamics registry.")
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
    root = resolve_repo_path(config["dynamics_ready_registry"]["local_root"])
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    validate_header(root / relative_paths["species_catalog"], config["schemas"]["dynamics_species_fields"])
    validate_header(root / relative_paths["parameter_catalog"], config["schemas"]["dynamics_parameter_fields"])
    validate_header(root / relative_paths["simulation_catalog"], config["schemas"]["dynamics_simulation_fields"])
    validate_header(root / relative_paths["sample_alignment_catalog"], config["schemas"]["dynamics_sample_alignment_fields"])
    validate_header(root / relative_paths["case_validation_catalog"], config["schemas"]["dynamics_case_validation_fields"])
    species_rows, parameter_rows, simulation_rows, sample_alignment_rows = load_dynamics_catalogs(config)
    if not species_rows:
        raise ValueError("Dynamics registry must contain species rows.")
    if not simulation_rows:
        raise ValueError("Dynamics registry must contain simulation rows.")
    store_path = root / relative_paths["trajectory_store"]
    manifest = read_store_manifest(store_path)
    trajectory_tensor = load_all_trajectories(config)
    if trajectory_tensor.shape[0] != len(simulation_rows):
        raise ValueError("Trajectory store does not match simulation catalog.")
    if list(manifest["axis_names"]) != ["simulation", "time", "region", "compartment", "species"]:
        raise ValueError("Unexpected trajectory axis ordering.")
    for row in sample_alignment_rows:
        if not (REPO_ROOT / str(row["field_alignment_bundle_path"])).exists():
            raise FileNotFoundError(f"Missing field alignment bundle: {row}")
    density_profile_ids: set[str] = set()
    for disease_program in config["structural_prior_registry"]["disease_programs"]:
        density_profile_ids.update(structural_prior_uid_lookup(config, str(disease_program)).values())
    for row in simulation_rows:
        if row["structural_prior_uid"] not in density_profile_ids:
            raise ValueError(f"Unknown structural_prior_uid in simulation catalog: {row}")
        if not row["local_response_head_uid"]:
            raise ValueError(f"local_response_head_uid missing in simulation catalog: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=store_path,
        species_row_count=len(species_rows),
        parameter_row_count=len(parameter_rows),
        simulation_row_count=len(simulation_rows),
        sample_alignment_row_count=len(sample_alignment_rows),
    )


if __name__ == "__main__":
    main()
