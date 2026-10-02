from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark-ready cohort, fold, and split outputs.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if header != expected_fields:
        raise ValueError(f"Header mismatch for {path}: {header}")


def fold_root_base(config: dict) -> Path:
    template_prefix = str(config["benchmark_registry"]["fold_root_template"]).split("{benchmark_name}", 1)[0].rstrip("/")
    return resolve_repo_path(template_prefix)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    benchmark_root = resolve_repo_path(config["benchmark_registry"]["local_root"])
    relative_paths = config["benchmark_registry"]["relative_paths"]

    cohort_catalog_path = benchmark_root / relative_paths["cohort_catalog"]
    fold_catalog_path = benchmark_root / relative_paths["fold_catalog"]
    training_dataset_catalog_path = benchmark_root / relative_paths["training_dataset_catalog"]
    baseline_result_catalog_path = benchmark_root / relative_paths["baseline_result_catalog"]

    validate_header(cohort_catalog_path, config["schemas"]["benchmark_cohort_fields"])
    validate_header(fold_catalog_path, config["schemas"]["benchmark_fold_fields"])
    validate_header(training_dataset_catalog_path, config["schemas"]["benchmark_training_dataset_fields"])
    validate_header(baseline_result_catalog_path, config["schemas"]["benchmark_result_fields"])

    cohort_rows = read_tsv(cohort_catalog_path)
    fold_rows = read_tsv(fold_catalog_path)
    dataset_rows = read_tsv(training_dataset_catalog_path)
    baseline_result_rows = read_tsv(baseline_result_catalog_path)
    pipeline_logger.progress(
        "loaded_benchmark_inputs",
        cohort_row_count=len(cohort_rows),
        fold_row_count=len(fold_rows),
        dataset_row_count=len(dataset_rows),
        baseline_result_row_count=len(baseline_result_rows),
    )
    for cohort_row in cohort_rows:
        if cohort_row["world_model_role"] != "disease_observation":
            raise ValueError(f"Unexpected world_model_role in benchmark cohort: {cohort_row['study_uid']}")
        if cohort_row["inclusion_flag"] == "yes" and cohort_row["subject_linkable_flag"] != "yes":
            raise ValueError(f"Unexpected subject_linkable_flag in admitted benchmark cohort: {cohort_row['study_uid']}")
        if cohort_row["inclusion_flag"] == "yes" and cohort_row["public_only_flag"] != "yes":
            raise ValueError(f"Unexpected public_only_flag in admitted benchmark cohort: {cohort_row['study_uid']}")
    disease_program_config = {
        str(program_name): dict(program_config)
        for program_name, program_config in config["benchmark_registry"].get("disease_programs", {}).items()
    }
    disease_programs = sorted(set(disease_program_config) | {row.get("disease_program", "") for row in cohort_rows if row.get("disease_program", "")})
    for disease_program in disease_programs:
        program_config = disease_program_config.get(disease_program, {})
        minimum_primary_study_count = int(program_config.get("minimum_included_primary_study_count", 1))
        contract_status = str(program_config.get("contract_status", "complete_public_contract"))
        primary_rows = [
            row
            for row in cohort_rows
            if row.get("disease_program", "") == disease_program
            and row["cohort_role"] == "primary_train_eval"
            and row["inclusion_flag"] == "yes"
        ]
        disease_fold_rows = [row for row in fold_rows if row.get("disease_program", "") == disease_program]
        if len(primary_rows) < minimum_primary_study_count:
            if disease_fold_rows:
                raise ValueError(
                    f"Benchmark contract for {disease_program} is incomplete ({contract_status}) and must not materialize folds: "
                    f"{len(disease_fold_rows)} found."
                )
            continue
        if len(disease_fold_rows) != len(primary_rows):
            raise ValueError(
                f"Fold count must equal included primary study count for {disease_program}: "
                f"{len(disease_fold_rows)} vs {len(primary_rows)}"
            )

    sample_split_schema = config["schemas"]["benchmark_sample_split_fields"]
    expected_fold_directories = {
        (str(row["benchmark_name"]), str(row["fold_name"]))
        for row in fold_rows
    }
    base_root = fold_root_base(config)
    if base_root.exists():
        for benchmark_dir in base_root.iterdir():
            if not benchmark_dir.is_dir():
                continue
            for fold_dir in benchmark_dir.iterdir():
                if not fold_dir.is_dir():
                    continue
                if (benchmark_dir.name, fold_dir.name) in expected_fold_directories:
                    continue
                raise ValueError(f"Catalog-orphan benchmark fold directory detected: {fold_dir}")
    for fold_row in fold_rows:
        matching_dataset_rows = [
            row
            for row in dataset_rows
            if row["fold_name"] == fold_row["fold_name"] and row.get("disease_program", "") == fold_row.get("disease_program", "")
        ]
        if not matching_dataset_rows:
            raise ValueError(f"Training dataset catalog lacks rows for fold {fold_row['fold_name']}")
        sample_split_path = resolve_repo_path(matching_dataset_rows[0]["sample_split_path"])
        if not sample_split_path.exists():
            raise FileNotFoundError(f"Missing fold sample split file: {sample_split_path}")
        split_frame = pd.read_parquet(sample_split_path)
        if list(split_frame.columns) != sample_split_schema:
            raise ValueError(f"Sample split schema mismatch for {sample_split_path}: {list(split_frame.columns)}")
        test_study_uids = sorted(split_frame.loc[split_frame["split_role"] == "test", "study_uid"].unique().tolist())
        if test_study_uids != [fold_row["held_out_study_uid"]]:
            raise ValueError(f"Held-out study mismatch for {fold_row['fold_name']}: {test_study_uids}")
        train_validation_overlap = set(split_frame.loc[split_frame["split_role"] == "train", "sample_uid"]) & set(
            split_frame.loc[split_frame["split_role"] == "validation", "sample_uid"]
        )
        if train_validation_overlap:
            raise ValueError(f"Train/validation overlap detected for {fold_row['fold_name']}")
        test_overlap = set(split_frame.loc[split_frame["split_role"] == "test", "sample_uid"]) & set(
            split_frame.loc[split_frame["split_role"] != "test", "sample_uid"]
        )
        if test_overlap:
            raise ValueError(f"Test split overlaps with training data for {fold_row['fold_name']}")
        supported_modalities = set(str(value) for value in str(fold_row["included_modalities"]).split("|") if str(value))
        modality_rows = [row for row in matching_dataset_rows if row["modality"] in supported_modalities]
        if not modality_rows:
            raise ValueError(f"No supported modality dataset rows found for fold {fold_row['fold_name']}")
        for modality in supported_modalities:
            modality_role_rows = [row for row in modality_rows if row["modality"] == modality]
            required_roles = {"train", "validation", "test"}
            present_roles = {str(row["dataset_role"]) for row in modality_role_rows if int(row["sample_count"]) > 0}
            if present_roles != required_roles:
                raise ValueError(
                    f"Benchmark fold {fold_row['fold_name']} modality {modality} must expose positive train/validation/test rows, "
                    f"found {sorted(present_roles)}"
                )
        for dataset_row in modality_rows:
            if int(dataset_row["feature_count"]) <= 0:
                raise ValueError(f"Benchmark dataset row has zero features: {dataset_row}")
            if int(dataset_row["sample_count"]) <= 0:
                raise ValueError(f"Benchmark dataset row has zero samples: {dataset_row}")

    pipeline_logger.progress(
        "benchmark_validated",
        artifact=fold_catalog_path,
        cohort_row_count=len(cohort_rows),
        fold_row_count=len(fold_rows),
        dataset_row_count=len(dataset_rows),
        baseline_result_row_count=len(baseline_result_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=fold_catalog_path,
        cohort_row_count=len(cohort_rows),
        fold_row_count=len(fold_rows),
        dataset_row_count=len(dataset_rows),
        baseline_result_row_count=len(baseline_result_rows),
    )

if __name__ == "__main__":
    main()
