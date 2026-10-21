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
from Libs.utils.evaluation.strict_case_discovery import action_ready_contamination_status, load_action_ready_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the action-ready screening registry outputs.")
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
    root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    relative_paths = config["screening_pipeline"]["action_ready_registry"]["relative_paths"]
    path_map = {
        "atlas_program_feature_mapping": root / relative_paths["atlas_program_feature_mapping"],
        "target_program_catalog": root / relative_paths["target_program_catalog"],
        "drug_program_catalog": root / relative_paths["drug_program_catalog"],
        "regimen_catalog": root / relative_paths["regimen_catalog"],
        "action_exclusion_catalog": root / relative_paths["action_exclusion_catalog"],
        "build_manifest": root / relative_paths["build_manifest"],
    }
    for path in path_map.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing action-ready output: {path}")

    validate_header(path_map["atlas_program_feature_mapping"], config["schemas"]["action_ready_atlas_mapping_fields"])
    validate_header(path_map["target_program_catalog"], config["schemas"]["action_ready_target_program_fields"])
    validate_header(path_map["drug_program_catalog"], config["schemas"]["action_ready_drug_program_fields"])
    validate_header(path_map["regimen_catalog"], config["schemas"]["action_ready_regimen_fields"])
    validate_header(path_map["action_exclusion_catalog"], config["schemas"]["action_ready_exclusion_fields"])
    manifest = load_action_ready_manifest(config)
    contamination_status = action_ready_contamination_status(config, manifest)
    if contamination_status != "clean":
        raise ValueError(f"Strict discovery action-ready manifest is not clean: {contamination_status}")

    atlas_mapping_rows = read_tsv(path_map["atlas_program_feature_mapping"])
    target_program_rows = read_tsv(path_map["target_program_catalog"])
    drug_program_rows = read_tsv(path_map["drug_program_catalog"])
    regimen_rows = read_tsv(path_map["regimen_catalog"])
    exclusion_rows = read_tsv(path_map["action_exclusion_catalog"])
    pipeline_logger.progress(
        "loaded_action_ready_inputs",
        atlas_mapping_row_count=len(atlas_mapping_rows),
        target_program_row_count=len(target_program_rows),
        drug_program_row_count=len(drug_program_rows),
        regimen_row_count=len(regimen_rows),
        exclusion_row_count=len(exclusion_rows),
        contamination_status=contamination_status,
    )

    if not atlas_mapping_rows:
        raise ValueError("atlas_program_feature_mapping.tsv must not be empty.")
    if not target_program_rows:
        raise ValueError("target_program_catalog.tsv must not be empty.")
    if not drug_program_rows:
        raise ValueError("drug_program_catalog.tsv must not be empty.")
    if not regimen_rows:
        raise ValueError("regimen_catalog.tsv must not be empty.")

    target_program_uids = {row["target_program_uid"] for row in target_program_rows}
    drug_program_uids = {row["drug_program_uid"] for row in drug_program_rows}
    if not target_program_uids:
        raise ValueError("No target_program_uids were constructed.")
    if not drug_program_uids:
        raise ValueError("No drug_program_uids were constructed.")

    for row in atlas_mapping_rows:
        coverage = float(row["coverage_fraction"])
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"coverage_fraction out of range: {row}")

    for row in drug_program_rows:
        if int(row["target_count"]) <= 0:
            raise ValueError(f"Drug program without targets detected: {row}")
        if row["pk_prior_ready_flag"] not in {"yes", "no"}:
            raise ValueError(f"Unexpected pk_prior_ready_flag: {row}")

    valid_routes = {"oral", "intravenous", "intrathecal"}
    valid_schedules = {"daily", "twice_daily", "weekly"}
    valid_doses = {"dose_025", "dose_050", "dose_100", "dose_200"}
    for row in regimen_rows:
        if row["route"] not in valid_routes:
            raise ValueError(f"Unexpected regimen route: {row}")
        if row["schedule"] not in valid_schedules:
            raise ValueError(f"Unexpected regimen schedule: {row}")
        if row["dose_level_id"] not in valid_doses:
            raise ValueError(f"Unexpected regimen dose_level_id: {row}")
        if float(row["pk_feasibility_score"]) < 0:
            raise ValueError(f"Negative pk_feasibility_score: {row}")
        if int(row["regimen_size"]) not in {1, 2}:
            raise ValueError(f"Unexpected regimen_size: {row}")
        if "murcko_scaffold" in row:
            raise ValueError("Unexpected stale murcko_scaffold field in regimen row.")

    pipeline_logger.progress(
        "action_ready_validated",
        artifact=path_map["regimen_catalog"],
        atlas_mapping_row_count=len(atlas_mapping_rows),
        target_program_row_count=len(target_program_rows),
        drug_program_row_count=len(drug_program_rows),
        regimen_row_count=len(regimen_rows),
        exclusion_row_count=len(exclusion_rows),
        contamination_status=contamination_status,
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path_map["regimen_catalog"],
        atlas_mapping_row_count=len(atlas_mapping_rows),
        target_program_row_count=len(target_program_rows),
        drug_program_row_count=len(drug_program_rows),
        regimen_row_count=len(regimen_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
