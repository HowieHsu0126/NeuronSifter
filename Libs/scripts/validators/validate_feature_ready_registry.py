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
    parser = argparse.ArgumentParser(description="Validate the feature-ready processed catalogs.")
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
    feature_ready_root = resolve_repo_path(config["feature_ready_registry"]["local_root"])
    feature_paths = config["feature_ready_registry"]["relative_paths"]

    matrix_catalog_path = feature_ready_root / feature_paths["matrix_catalog"]
    member_catalog_path = feature_ready_root / feature_paths["member_catalog"]
    sample_axis_catalog_path = feature_ready_root / feature_paths["sample_axis_catalog"]
    feature_axis_catalog_path = feature_ready_root / feature_paths["feature_axis_catalog"]
    non_matrix_catalog_path = feature_ready_root / feature_paths["non_matrix_asset_catalog"]
    study_summary_path = feature_ready_root / feature_paths["study_summary"]

    output_paths = [
        matrix_catalog_path,
        member_catalog_path,
        sample_axis_catalog_path,
        feature_axis_catalog_path,
        non_matrix_catalog_path,
        study_summary_path,
    ]
    for output_path in output_paths:
        if not output_path.exists():
            raise FileNotFoundError(f"Feature-ready output is missing: {output_path}")

    validate_header(matrix_catalog_path, config["schemas"]["feature_ready_matrix_fields"])
    validate_header(member_catalog_path, config["schemas"]["feature_ready_member_fields"])
    validate_header(sample_axis_catalog_path, config["schemas"]["feature_ready_sample_axis_fields"])
    validate_header(feature_axis_catalog_path, config["schemas"]["feature_ready_feature_axis_fields"])
    validate_header(non_matrix_catalog_path, config["schemas"]["feature_ready_non_matrix_fields"])
    validate_header(study_summary_path, config["schemas"]["feature_ready_study_fields"])

    matrix_rows = read_tsv(matrix_catalog_path)
    member_rows = read_tsv(member_catalog_path)
    sample_axis_rows = read_tsv(sample_axis_catalog_path)
    feature_axis_rows = read_tsv(feature_axis_catalog_path)
    non_matrix_rows = read_tsv(non_matrix_catalog_path)
    study_summary_rows = read_tsv(study_summary_path)

    processed_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    cohort_sample_catalog_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["sample_catalog"]
    cohort_asset_catalog_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["quantitative_asset_catalog"]
    cohort_study_summary_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["study_summary"]

    cohort_sample_rows = read_tsv(cohort_sample_catalog_path)
    cohort_asset_rows = read_tsv(cohort_asset_catalog_path)
    cohort_study_rows = read_tsv(cohort_study_summary_path)
    pipeline_logger.progress(
        "loaded_feature_ready_inputs",
        matrix_row_count=len(matrix_rows),
        member_row_count=len(member_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        non_matrix_row_count=len(non_matrix_rows),
        study_row_count=len(study_summary_rows),
        cohort_sample_row_count=len(cohort_sample_rows),
        cohort_asset_row_count=len(cohort_asset_rows),
        cohort_study_row_count=len(cohort_study_rows),
    )
    enabled_sources = set(config["feature_ready_registry"]["enabled_sources"])

    cohort_sample_rows = [row for row in cohort_sample_rows if row["source_key"] in enabled_sources]
    cohort_asset_rows = [
        row
        for row in cohort_asset_rows
        if row["source_key"] in enabled_sources and row["local_availability_flag"] == "yes" and row["local_path"]
    ]
    cohort_study_rows = [row for row in cohort_study_rows if row["source_key"] in enabled_sources]

    expected_studies = {
        (row["study_uid"], row["source_key"], row["source_accession"])
        for row in cohort_study_rows
    }
    actual_studies = {
        (row["study_uid"], row["source_key"], row["source_accession"])
        for row in study_summary_rows
    }
    if actual_studies != expected_studies:
        raise ValueError("Feature-ready study summary does not match the cohort-ready study set.")

    cohort_samples_by_uid = {row["sample_uid"]: row for row in cohort_sample_rows}
    cohort_assets_by_uid = {row["quantitative_asset_uid"]: row for row in cohort_asset_rows}
    cohort_assets_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    cohort_study_by_uid = {row["study_uid"]: row for row in cohort_study_rows}
    for cohort_asset_row in cohort_asset_rows:
        cohort_assets_by_study[cohort_asset_row["study_uid"]].append(cohort_asset_row)

    matrix_ids = [row["matrix_uid"] for row in matrix_rows]
    if len(matrix_ids) != len(set(matrix_ids)):
        raise ValueError("Feature-ready matrix catalog contains duplicate matrix_uid values.")

    matrix_rows_by_uid = {row["matrix_uid"]: row for row in matrix_rows}
    matrix_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for matrix_row in matrix_rows:
        quantitative_asset_uid = matrix_row["quantitative_asset_uid"]
        if quantitative_asset_uid not in cohort_assets_by_uid:
            raise ValueError(f"Matrix refers to unknown cohort-ready asset: {quantitative_asset_uid}")
        matrix_rows_by_study[matrix_row["study_uid"]].append(matrix_row)

    non_matrix_asset_ids = [row["quantitative_asset_uid"] for row in non_matrix_rows]
    if len(non_matrix_asset_ids) != len(set(non_matrix_asset_ids)):
        raise ValueError("Feature-ready non-matrix asset catalog contains duplicate quantitative_asset_uid values.")
    for non_matrix_row in non_matrix_rows:
        if non_matrix_row["quantitative_asset_uid"] not in cohort_assets_by_uid:
            raise ValueError(f"Non-matrix row refers to unknown cohort-ready asset: {non_matrix_row['quantitative_asset_uid']}")

    matrix_asset_ids = {row["quantitative_asset_uid"] for row in matrix_rows}
    non_matrix_asset_id_set = set(non_matrix_asset_ids)
    if matrix_asset_ids & non_matrix_asset_id_set:
        raise ValueError("A cohort-ready asset appears in both matrix and non-matrix feature-ready catalogs.")
    if matrix_asset_ids | non_matrix_asset_id_set != set(cohort_assets_by_uid):
        raise ValueError("Feature-ready catalogs do not fully cover the local cohort-ready asset set.")

    member_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for member_row in member_rows:
        matrix_uid = member_row["matrix_uid"]
        if matrix_uid not in matrix_rows_by_uid:
            raise ValueError(f"Archive member row refers to unknown matrix_uid: {matrix_uid}")
        if member_row["sample_uid"] and member_row["sample_uid"] not in cohort_samples_by_uid:
            raise ValueError(f"Archive member row refers to unknown sample_uid: {member_row['sample_uid']}")
        member_rows_by_matrix[matrix_uid].append(member_row)

    sample_axis_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_axis_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample_axis_row in sample_axis_rows:
        matrix_uid = sample_axis_row["matrix_uid"]
        if matrix_uid not in matrix_rows_by_uid:
            raise ValueError(f"Sample axis row refers to unknown matrix_uid: {matrix_uid}")
        if sample_axis_row["sample_uid"] and sample_axis_row["sample_uid"] not in cohort_samples_by_uid:
            raise ValueError(f"Sample axis row refers to unknown sample_uid: {sample_axis_row['sample_uid']}")
        sample_axis_rows_by_matrix[matrix_uid].append(sample_axis_row)
        sample_axis_rows_by_study[sample_axis_row["study_uid"]].append(sample_axis_row)

    feature_axis_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    feature_axis_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for feature_axis_row in feature_axis_rows:
        matrix_uid = feature_axis_row["matrix_uid"]
        if matrix_uid not in matrix_rows_by_uid:
            raise ValueError(f"Feature axis row refers to unknown matrix_uid: {matrix_uid}")
        feature_axis_rows_by_matrix[matrix_uid].append(feature_axis_row)
        feature_axis_rows_by_study[feature_axis_row["study_uid"]].append(feature_axis_row)

    for matrix_row in matrix_rows:
        matrix_uid = matrix_row["matrix_uid"]
        matrix_sample_axis_rows = sample_axis_rows_by_matrix.get(matrix_uid, [])
        matrix_feature_axis_rows = feature_axis_rows_by_matrix.get(matrix_uid, [])
        matrix_member_rows = member_rows_by_matrix.get(matrix_uid, [])
        if int(matrix_row["sample_axis_count"]) != len(matrix_sample_axis_rows):
            raise ValueError(f"Sample axis count mismatch for matrix {matrix_uid}")
        if int(matrix_row["feature_axis_count"]) != len(matrix_feature_axis_rows):
            raise ValueError(f"Feature axis count mismatch for matrix {matrix_uid}")
        mapped_sample_count = sum(1 for row in matrix_sample_axis_rows if row["sample_uid"])
        unmapped_sample_count = sum(1 for row in matrix_sample_axis_rows if not row["sample_uid"])
        if int(matrix_row["mapped_sample_count"]) != mapped_sample_count:
            raise ValueError(f"Mapped sample count mismatch for matrix {matrix_uid}")
        if int(matrix_row["unmapped_sample_count"]) != unmapped_sample_count:
            raise ValueError(f"Unmapped sample count mismatch for matrix {matrix_uid}")
        if not matrix_member_rows:
            raise ValueError(f"Matrix lacks archive/member traceability rows: {matrix_uid}")
        if not matrix_row["matrix_reference"]:
            raise ValueError(f"Matrix lacks matrix_reference: {matrix_uid}")

    for study_summary_row in study_summary_rows:
        study_uid = study_summary_row["study_uid"]
        cohort_study_row = cohort_study_by_uid[study_uid]
        study_matrix_rows = matrix_rows_by_study.get(study_uid, [])
        study_sample_axis_rows = sample_axis_rows_by_study.get(study_uid, [])
        study_feature_axis_rows = feature_axis_rows_by_study.get(study_uid, [])
        study_non_matrix_rows = [row for row in non_matrix_rows if row["study_uid"] == study_uid]
        if int(study_summary_row["standardized_sample_count"]) != int(cohort_study_row["standardized_sample_count"]):
            raise ValueError(f"Standardized sample count mismatch for study {study_uid}")
        if int(study_summary_row["matrix_count"]) != len(study_matrix_rows):
            raise ValueError(f"Matrix count mismatch for study {study_uid}")
        if int(study_summary_row["matrix_asset_count"]) != len({row["quantitative_asset_uid"] for row in study_matrix_rows}):
            raise ValueError(f"Matrix asset count mismatch for study {study_uid}")
        if int(study_summary_row["non_matrix_asset_count"]) != len(study_non_matrix_rows):
            raise ValueError(f"Non-matrix asset count mismatch for study {study_uid}")
        if int(study_summary_row["feature_axis_row_count"]) != len(study_feature_axis_rows):
            raise ValueError(f"Feature axis row count mismatch for study {study_uid}")
        expected_mapped_sample_uid_count = len({row["sample_uid"] for row in study_sample_axis_rows if row["sample_uid"]})
        if int(study_summary_row["mapped_sample_uid_count"]) != expected_mapped_sample_uid_count:
            raise ValueError(f"Mapped sample uid count mismatch for study {study_uid}")
        expected_unmapped_matrix_sample_count = sum(1 for row in study_sample_axis_rows if not row["sample_uid"])
        if int(study_summary_row["unmapped_matrix_sample_count"]) != expected_unmapped_matrix_sample_count:
            raise ValueError(f"Unmapped matrix sample count mismatch for study {study_uid}")
        expected_matrix_ready_flag = "yes" if study_matrix_rows else "no"
        if study_summary_row["matrix_ready_flag"] != expected_matrix_ready_flag:
            raise ValueError(f"matrix_ready_flag mismatch for study {study_uid}")
        local_asset_count = len(cohort_assets_by_study.get(study_uid, []))
        if int(study_summary_row["matrix_asset_count"]) + int(study_summary_row["non_matrix_asset_count"]) != local_asset_count:
            raise ValueError(f"Feature-ready asset coverage mismatch for study {study_uid}")
        if study_matrix_rows:
            primary_matrix_uid = study_summary_row["primary_matrix_uid"]
            if primary_matrix_uid not in {row["matrix_uid"] for row in study_matrix_rows}:
                raise ValueError(f"Primary matrix uid is not part of the study matrix set: {study_uid}")
            if study_summary_row["primary_matrix_path"] != matrix_rows_by_uid[primary_matrix_uid]["matrix_reference"]:
                raise ValueError(f"Primary matrix path mismatch for study {study_uid}")
            continue
        if study_summary_row["primary_matrix_uid"] or study_summary_row["primary_matrix_path"]:
            raise ValueError(f"Study without matrices should not expose a primary matrix reference: {study_uid}")

    pipeline_logger.progress(
        "feature_ready_validated",
        artifact=study_summary_path,
        matrix_row_count=len(matrix_rows),
        member_row_count=len(member_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        non_matrix_row_count=len(non_matrix_rows),
        study_row_count=len(study_summary_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=study_summary_path,
        matrix_row_count=len(matrix_rows),
        member_row_count=len(member_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        non_matrix_row_count=len(non_matrix_rows),
        study_row_count=len(study_summary_rows),
    )

if __name__ == "__main__":
    main()
