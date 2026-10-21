from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the cohort-ready processed catalogs.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
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
    processed_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    relative_paths = config["cohort_ready_registry"]["relative_paths"]

    sample_catalog_path = processed_root / relative_paths["sample_catalog"]
    asset_catalog_path = processed_root / relative_paths["quantitative_asset_catalog"]
    study_summary_path = processed_root / relative_paths["study_summary"]

    for output_path in [sample_catalog_path, asset_catalog_path, study_summary_path]:
        if not output_path.exists():
            raise FileNotFoundError(f"Cohort-ready output is missing: {output_path}")

    validate_header(sample_catalog_path, config["schemas"]["cohort_ready_sample_fields"])
    validate_header(asset_catalog_path, config["schemas"]["cohort_ready_asset_fields"])
    validate_header(study_summary_path, config["schemas"]["cohort_ready_study_fields"])

    sample_rows = read_tsv(sample_catalog_path)
    asset_rows = read_tsv(asset_catalog_path)
    study_rows = read_tsv(study_summary_path)

    processed_registry_root = resolve_repo_path(config["processed_registry"]["local_root"])
    study_catalog_path = processed_registry_root / config["processed_registry"]["relative_paths"]["study_catalog"]
    processed_study_rows = read_tsv(study_catalog_path)
    pipeline_logger.progress(
        "loaded_cohort_ready_inputs",
        sample_row_count=len(sample_rows),
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
        processed_study_row_count=len(processed_study_rows),
    )
    expected_model_ready_studies = {
        (row["study_uid"], row["source_key"], row["source_accession"])
        for row in processed_study_rows
        if row["model_ready_flag"] == "yes" and row["source_key"] in config["cohort_ready_registry"]["enabled_sources"]
    }

    actual_studies = {(row["study_uid"], row["source_key"], row["source_accession"]) for row in study_rows}
    if actual_studies != expected_model_ready_studies:
        raise ValueError("Cohort-ready study summary does not match the model-ready study set.")

    samples_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    assets_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample_row in sample_rows:
        samples_by_study[sample_row["study_uid"]].append(sample_row)
    for asset_row in asset_rows:
        assets_by_study[asset_row["study_uid"]].append(asset_row)

    if not sample_rows:
        raise ValueError("Cohort-ready sample catalog is empty.")
    if not asset_rows:
        raise ValueError("Cohort-ready asset catalog is empty.")

    for asset_row in asset_rows:
        if asset_row["local_availability_flag"] == "yes" and not asset_row["local_path"]:
            raise ValueError(f"Local asset row lacks local_path: {asset_row['quantitative_asset_uid']}")
        if asset_row["local_availability_flag"] == "no" and not asset_row["asset_url"]:
            raise ValueError(f"Remote asset row lacks asset_url: {asset_row['quantitative_asset_uid']}")

    for study_row in study_rows:
        study_uid = study_row["study_uid"]
        study_samples = samples_by_study.get(study_uid, [])
        study_assets = assets_by_study.get(study_uid, [])
        if int(study_row["standardized_sample_count"]) != len(study_samples):
            raise ValueError(f"Sample count mismatch for study {study_uid}")
        if int(study_row["quantitative_asset_count"]) != len(study_assets):
            raise ValueError(f"Quantitative asset count mismatch for study {study_uid}")
        if int(study_row["local_quantitative_asset_count"]) != sum(
            1 for row in study_assets if row["local_availability_flag"] == "yes"
        ):
            raise ValueError(f"Local asset count mismatch for study {study_uid}")
        if int(study_row["remote_only_quantitative_asset_count"]) != sum(
            1 for row in study_assets if row["local_availability_flag"] == "no"
        ):
            raise ValueError(f"Remote asset count mismatch for study {study_uid}")
        if not study_assets:
            raise ValueError(f"Study lacks quantitative assets: {study_uid}")
        if not study_samples:
            raise ValueError(f"Study lacks standardized samples: {study_uid}")
        primary_asset_ids = {row["quantitative_asset_uid"] for row in study_assets}
        if study_row["primary_quantitative_asset_id"] not in primary_asset_ids:
            raise ValueError(f"Primary asset is not part of study asset set: {study_uid}")

    for sample_row in sample_rows:
        if sample_row["study_uid"] not in {row["study_uid"] for row in study_rows}:
            raise ValueError(f"Sample refers to unknown study: {sample_row['sample_uid']}")
        if not sample_row["primary_quantitative_asset_id"]:
            raise ValueError(f"Sample lacks primary quantitative asset id: {sample_row['sample_uid']}")

    pipeline_logger.progress(
        "cohort_ready_validated",
        artifact=study_summary_path,
        sample_row_count=len(sample_rows),
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=study_summary_path,
        sample_row_count=len(sample_rows),
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
    )

if __name__ == "__main__":
    main()
