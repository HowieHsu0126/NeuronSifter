from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_fields import (
    build_parameter_rows,
    build_sample_alignment_rows,
    build_species_rows,
    simulation_rows_and_store,
    write_study_alignment_catalogs,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public AD-first mesoscale dynamics registry.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["dynamics_ready_registry"]["local_root"])
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    ensure_directory(root)
    structural_prior_root = resolve_repo_path(config["structural_prior_registry"]["local_root"])
    structural_prior_catalog = structural_prior_root / config["structural_prior_registry"]["relative_paths"]["region_prior_catalog"]
    if not structural_prior_catalog.exists():
        raise FileNotFoundError(f"Structural prior region catalog is missing: {structural_prior_catalog}")

    species_rows = build_species_rows(config)
    parameter_rows = build_parameter_rows(config)
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    regimen_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["regimen_catalog"])
    compound_root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    compound_rows = read_tsv(compound_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"])
    simulation_rows, store_path = simulation_rows_and_store(config, regimen_rows, compound_rows)
    sample_alignment_rows, rows_by_study = build_sample_alignment_rows(config)
    path_by_study = write_study_alignment_catalogs(config, rows_by_study)
    for row in sample_alignment_rows:
        row["field_alignment_bundle_path"] = str(path_by_study[str(row["study_uid"])].resolve().relative_to(REPO_ROOT))
    case_validation_rows = []
    for validation_row in config["dynamics_ready_registry"]["case_validation_entries"]:
        case_validation_rows.append(
            {
                "validation_uid": hashlib.sha1(
                    "::".join(
                        [
                            str(validation_row["case_study_name"]),
                            str(validation_row["intervention_label"]),
                            str(validation_row["endpoint_family"]),
                        ]
                    ).encode("utf-8")
                ).hexdigest()[:12],
                **validation_row,
            }
        )

    write_tsv(root / relative_paths["species_catalog"], config["schemas"]["dynamics_species_fields"], species_rows)
    write_tsv(root / relative_paths["parameter_catalog"], config["schemas"]["dynamics_parameter_fields"], parameter_rows)
    write_tsv(root / relative_paths["simulation_catalog"], config["schemas"]["dynamics_simulation_fields"], simulation_rows)
    write_tsv(root / relative_paths["sample_alignment_catalog"], config["schemas"]["dynamics_sample_alignment_fields"], sample_alignment_rows)
    write_tsv(root / relative_paths["case_validation_catalog"], config["schemas"]["dynamics_case_validation_fields"], case_validation_rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=store_path,
        species_row_count=len(species_rows),
        parameter_row_count=len(parameter_rows),
        simulation_row_count=len(simulation_rows),
        sample_alignment_row_count=len(sample_alignment_rows),
        case_validation_row_count=len(case_validation_rows),
    )


if __name__ == "__main__":
    main()
