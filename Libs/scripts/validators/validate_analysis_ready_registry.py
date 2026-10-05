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
    parser = argparse.ArgumentParser(description="Validate the analysis-ready study bundles.")
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


def eligible_matrix_rows(study_matrix_rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    allowed_matrix_semantics = set(config["analysis_ready_registry"]["allowed_matrix_semantics"])
    minimum_mapped_fraction = float(config["analysis_ready_registry"]["minimum_mapped_fraction"])
    minimum_mapped_sample_count = int(config["analysis_ready_registry"]["minimum_mapped_sample_count"])
    eligible_rows: list[dict[str, str]] = []
    for matrix_row in study_matrix_rows:
        mapped_sample_count = int(matrix_row["mapped_sample_count"])
        sample_axis_count = int(matrix_row["sample_axis_count"])
        mapped_fraction = mapped_sample_count / sample_axis_count if sample_axis_count else 0.0
        if matrix_row["matrix_semantics"] not in allowed_matrix_semantics:
            continue
        if mapped_sample_count < minimum_mapped_sample_count:
            continue
        if mapped_fraction < minimum_mapped_fraction:
            continue
        eligible_rows.append(matrix_row)
    return eligible_rows


def ordered_matrix_rows(source_key: str, study_matrix_rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    selection_priority = config["analysis_ready_registry"]["selection_priority"][source_key]
    return sorted(
        study_matrix_rows,
        key=lambda row: (
            selection_priority.index(row["measurement_label"])
            if row["measurement_label"] in selection_priority
            else len(selection_priority),
            -int(row["mapped_sample_count"]),
            int(row["unmapped_sample_count"]),
            -int(row["feature_axis_count"]),
            row["matrix_name"],
        ),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    analysis_ready_root = resolve_repo_path(config["analysis_ready_registry"]["local_root"])
    relative_paths = config["analysis_ready_registry"]["relative_paths"]

    study_summary_path = analysis_ready_root / relative_paths["study_summary"]
    loader_catalog_path = analysis_ready_root / relative_paths["loader_catalog"]
    sample_catalog_path = analysis_ready_root / relative_paths["sample_catalog"]
    feature_catalog_path = analysis_ready_root / relative_paths["feature_catalog"]
    exclusion_catalog_path = analysis_ready_root / relative_paths["exclusion_catalog"]

    for output_path in [study_summary_path, loader_catalog_path, sample_catalog_path, feature_catalog_path, exclusion_catalog_path]:
        if not output_path.exists():
            raise FileNotFoundError(f"Analysis-ready output is missing: {output_path}")

    validate_header(study_summary_path, config["schemas"]["analysis_ready_study_fields"])
    validate_header(loader_catalog_path, config["schemas"]["analysis_ready_loader_fields"])
    validate_header(sample_catalog_path, config["schemas"]["analysis_ready_sample_fields"])
    validate_header(feature_catalog_path, config["schemas"]["analysis_ready_feature_fields"])
    validate_header(exclusion_catalog_path, config["schemas"]["analysis_ready_exclusion_fields"])

    study_summary_rows = read_tsv(study_summary_path)
    loader_rows = read_tsv(loader_catalog_path)
    sample_rows = read_tsv(sample_catalog_path)
    feature_rows = read_tsv(feature_catalog_path)
    exclusion_rows = read_tsv(exclusion_catalog_path)

    feature_ready_root = resolve_repo_path(config["feature_ready_registry"]["local_root"])
    feature_relative_paths = config["feature_ready_registry"]["relative_paths"]
    feature_matrix_rows = read_tsv(feature_ready_root / feature_relative_paths["matrix_catalog"])
    feature_sample_axis_rows = read_tsv(feature_ready_root / feature_relative_paths["sample_axis_catalog"])
    feature_feature_axis_rows = read_tsv(feature_ready_root / feature_relative_paths["feature_axis_catalog"])
    feature_study_rows = read_tsv(feature_ready_root / feature_relative_paths["study_summary"])

    cohort_ready_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    cohort_relative_paths = config["cohort_ready_registry"]["relative_paths"]
    cohort_sample_rows = read_tsv(cohort_ready_root / cohort_relative_paths["sample_catalog"])
    cohort_study_rows = read_tsv(cohort_ready_root / cohort_relative_paths["study_summary"])
    pipeline_logger.progress(
        "loaded_analysis_ready_inputs",
        study_row_count=len(study_summary_rows),
        loader_row_count=len(loader_rows),
        sample_row_count=len(sample_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
        feature_matrix_row_count=len(feature_matrix_rows),
        cohort_sample_row_count=len(cohort_sample_rows),
        cohort_study_row_count=len(cohort_study_rows),
    )

    enabled_sources = set(config["analysis_ready_registry"]["enabled_sources"])
    feature_matrix_rows = [row for row in feature_matrix_rows if row["source_key"] in enabled_sources]
    feature_sample_axis_rows = [row for row in feature_sample_axis_rows if row["source_key"] in enabled_sources]
    feature_feature_axis_rows = [row for row in feature_feature_axis_rows if row["source_key"] in enabled_sources]
    feature_study_rows = [row for row in feature_study_rows if row["source_key"] in enabled_sources]
    cohort_sample_rows = [row for row in cohort_sample_rows if row["source_key"] in enabled_sources]
    cohort_study_rows = [row for row in cohort_study_rows if row["source_key"] in enabled_sources]

    feature_study_by_uid = {row["study_uid"]: row for row in feature_study_rows}
    feature_matrix_by_uid = {row["matrix_uid"]: row for row in feature_matrix_rows}
    cohort_sample_by_uid = {row["sample_uid"]: row for row in cohort_sample_rows}
    cohort_study_by_uid = {row["study_uid"]: row for row in cohort_study_rows}

    matrix_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_axis_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    feature_axis_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for feature_matrix_row in feature_matrix_rows:
        matrix_rows_by_study[feature_matrix_row["study_uid"]].append(feature_matrix_row)
    for sample_axis_row in feature_sample_axis_rows:
        sample_axis_rows_by_matrix[sample_axis_row["matrix_uid"]].append(sample_axis_row)
    for feature_axis_row in feature_feature_axis_rows:
        feature_axis_rows_by_matrix[feature_axis_row["matrix_uid"]].append(feature_axis_row)

    expected_studies = {(row["study_uid"], row["source_key"], row["source_accession"]) for row in feature_study_rows}
    actual_studies = {(row["study_uid"], row["source_key"], row["source_accession"]) for row in study_summary_rows}
    if actual_studies != expected_studies:
        raise ValueError("Analysis-ready study summary does not match the feature-ready study set.")

    ready_study_uids = {row["study_uid"] for row in study_summary_rows if row["analysis_ready_flag"] == "yes"}
    excluded_study_uids = {row["study_uid"] for row in study_summary_rows if row["analysis_ready_flag"] == "no"}
    if ready_study_uids & excluded_study_uids:
        raise ValueError("A study appears in both analysis-ready and excluded sets.")
    if ready_study_uids | excluded_study_uids != {row["study_uid"] for row in study_summary_rows}:
        raise ValueError("Analysis-ready study coverage is incomplete.")

    selected_matrix_uids = [row["selected_matrix_uid"] for row in study_summary_rows if row["selected_matrix_uid"]]
    if len(selected_matrix_uids) != len(set(selected_matrix_uids)):
        raise ValueError("Analysis-ready study summary contains duplicate selected_matrix_uid values.")

    loader_rows_by_matrix = {row["selected_matrix_uid"]: row for row in loader_rows}
    if set(loader_rows_by_matrix) != set(selected_matrix_uids):
        raise ValueError("Loader catalog does not match the selected matrix set.")

    sample_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample_row in sample_rows:
        if sample_row["sample_uid"] not in cohort_sample_by_uid:
            raise ValueError(f"Analysis sample row refers to unknown sample_uid: {sample_row['sample_uid']}")
        sample_rows_by_matrix[sample_row["selected_matrix_uid"]].append(sample_row)

    feature_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for feature_row in feature_rows:
        feature_rows_by_matrix[feature_row["selected_matrix_uid"]].append(feature_row)

    exclusion_rows_by_study = {row["study_uid"]: row for row in exclusion_rows}
    if set(exclusion_rows_by_study) != excluded_study_uids:
        raise ValueError("Excluded study catalog does not match the non-ready study set.")

    for study_summary_row in study_summary_rows:
        study_uid = study_summary_row["study_uid"]
        feature_study_row = feature_study_by_uid[study_uid]
        cohort_study_row = cohort_study_by_uid[study_uid]
        study_matrix_rows = matrix_rows_by_study.get(study_uid, [])
        eligible_rows = eligible_matrix_rows(study_matrix_rows, config)
        ordered_eligible_rows = ordered_matrix_rows(feature_study_row["source_key"], eligible_rows, config)

        if int(study_summary_row["standardized_sample_count"]) != int(cohort_study_row["standardized_sample_count"]):
            raise ValueError(f"Standardized sample count mismatch for study {study_uid}")
        if int(study_summary_row["local_quantitative_asset_count"]) != int(cohort_study_row["local_quantitative_asset_count"]):
            raise ValueError(f"Local quantitative asset count mismatch for study {study_uid}")
        if int(study_summary_row["candidate_matrix_count"]) != len(study_matrix_rows):
            raise ValueError(f"Candidate matrix count mismatch for study {study_uid}")
        if int(study_summary_row["eligible_matrix_count"]) != len(eligible_rows):
            raise ValueError(f"Eligible matrix count mismatch for study {study_uid}")

        if study_summary_row["analysis_ready_flag"] == "no":
            if study_summary_row["selected_matrix_uid"] or study_summary_row["selected_matrix_reference"]:
                raise ValueError(f"Excluded study should not expose a selected matrix: {study_uid}")
            if study_uid not in exclusion_rows_by_study:
                raise ValueError(f"Excluded study missing from exclusion catalog: {study_uid}")
            continue

        if not ordered_eligible_rows:
            raise ValueError(f"Analysis-ready study lacks an eligible feature-ready matrix: {study_uid}")
        selected_matrix_row = ordered_eligible_rows[0]
        if study_summary_row["selected_matrix_uid"] != selected_matrix_row["matrix_uid"]:
            raise ValueError(f"Selected matrix mismatch for study {study_uid}")
        if study_summary_row["selected_matrix_reference"] != selected_matrix_row["matrix_reference"]:
            raise ValueError(f"Selected matrix reference mismatch for study {study_uid}")
        if study_summary_row["selected_measurement_label"] != selected_matrix_row["measurement_label"]:
            raise ValueError(f"Selected measurement label mismatch for study {study_uid}")
        if study_summary_row["selected_matrix_semantics"] != selected_matrix_row["matrix_semantics"]:
            raise ValueError(f"Selected matrix semantics mismatch for study {study_uid}")

        loader_row = loader_rows_by_matrix[selected_matrix_row["matrix_uid"]]
        if loader_row["matrix_reference"] != selected_matrix_row["matrix_reference"]:
            raise ValueError(f"Loader matrix reference mismatch for study {study_uid}")
        if loader_row["selected_measurement_label"] != selected_matrix_row["measurement_label"]:
            raise ValueError(f"Loader measurement label mismatch for study {study_uid}")

        expected_mapped_sample_axis_rows = [
            row for row in sample_axis_rows_by_matrix[selected_matrix_row["matrix_uid"]] if row["sample_uid"]
        ]
        if len(sample_rows_by_matrix.get(selected_matrix_row["matrix_uid"], [])) != len(expected_mapped_sample_axis_rows):
            raise ValueError(f"Analysis sample catalog size mismatch for study {study_uid}")
        expected_feature_axis_rows = feature_axis_rows_by_matrix[selected_matrix_row["matrix_uid"]]
        if len(feature_rows_by_matrix.get(selected_matrix_row["matrix_uid"], [])) != len(expected_feature_axis_rows):
            raise ValueError(f"Analysis feature catalog size mismatch for study {study_uid}")

        expected_mapped_sample_uid_count = len({row["sample_uid"] for row in expected_mapped_sample_axis_rows})
        if int(study_summary_row["mapped_sample_uid_count"]) != expected_mapped_sample_uid_count:
            raise ValueError(f"Mapped sample uid count mismatch for study {study_uid}")
        expected_fraction = len(expected_mapped_sample_axis_rows) / int(selected_matrix_row["sample_axis_count"])
        if abs(float(study_summary_row["mapped_fraction"]) - expected_fraction) > 1e-9:
            raise ValueError(f"Mapped fraction mismatch for study {study_uid}")

    pipeline_logger.progress(
        "analysis_ready_validated",
        artifact=study_summary_path,
        study_row_count=len(study_summary_rows),
        loader_row_count=len(loader_rows),
        sample_row_count=len(sample_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=study_summary_path,
        study_row_count=len(study_summary_rows),
        loader_row_count=len(loader_rows),
        sample_row_count=len(sample_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
