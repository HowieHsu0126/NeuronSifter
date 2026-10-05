from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis-ready study bundles from the feature-ready registry.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--source-accessions",
        default="",
        help="Optional comma-separated source_accession subset for targeted rebuilds.",
    )
    return parser.parse_args()


def selected_accessions(accession_text: str) -> set[str]:
    return {value.strip() for value in accession_text.split(",") if value.strip()}


def format_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def load_filtered_rows(config: dict, source_accessions: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    enabled_sources = set(config["analysis_ready_registry"]["enabled_sources"])

    feature_root = resolve_repo_path(config["feature_ready_registry"]["local_root"])
    feature_relative_paths = config["feature_ready_registry"]["relative_paths"]
    matrix_rows = read_tsv(feature_root / feature_relative_paths["matrix_catalog"])
    sample_axis_rows = read_tsv(feature_root / feature_relative_paths["sample_axis_catalog"])
    feature_axis_rows = read_tsv(feature_root / feature_relative_paths["feature_axis_catalog"])
    feature_study_rows = read_tsv(feature_root / feature_relative_paths["study_summary"])

    cohort_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    cohort_relative_paths = config["cohort_ready_registry"]["relative_paths"]
    cohort_sample_rows = read_tsv(cohort_root / cohort_relative_paths["sample_catalog"])
    cohort_study_rows = read_tsv(cohort_root / cohort_relative_paths["study_summary"])

    matrix_rows = [row for row in matrix_rows if row["source_key"] in enabled_sources]
    sample_axis_rows = [row for row in sample_axis_rows if row["source_key"] in enabled_sources]
    feature_axis_rows = [row for row in feature_axis_rows if row["source_key"] in enabled_sources]
    feature_study_rows = [row for row in feature_study_rows if row["source_key"] in enabled_sources]
    cohort_sample_rows = [row for row in cohort_sample_rows if row["source_key"] in enabled_sources]
    cohort_study_rows = [row for row in cohort_study_rows if row["source_key"] in enabled_sources]
    if source_accessions:
        matrix_rows = [row for row in matrix_rows if row["source_accession"] in source_accessions]
        sample_axis_rows = [row for row in sample_axis_rows if row["source_accession"] in source_accessions]
        feature_axis_rows = [row for row in feature_axis_rows if row["source_accession"] in source_accessions]
        feature_study_rows = [row for row in feature_study_rows if row["source_accession"] in source_accessions]
        cohort_sample_rows = [row for row in cohort_sample_rows if row["source_accession"] in source_accessions]
        cohort_study_rows = [row for row in cohort_study_rows if row["source_accession"] in source_accessions]
    return matrix_rows, sample_axis_rows, feature_axis_rows, feature_study_rows, cohort_sample_rows, cohort_study_rows


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


def choose_selected_matrix(study_row: dict[str, str], study_matrix_rows: list[dict[str, str]], config: dict) -> dict[str, str] | None:
    eligible_rows = eligible_matrix_rows(study_matrix_rows, config)
    if not eligible_rows:
        return None
    return ordered_matrix_rows(study_row["source_key"], eligible_rows, config)[0]


def build_exclusion_reason(study_matrix_rows: list[dict[str, str]], study_row: dict[str, str], config: dict) -> str:
    if not study_matrix_rows:
        return "no_candidate_matrix"
    best_candidate = ordered_matrix_rows(study_row["source_key"], study_matrix_rows, config)[0]
    mapped_sample_count = int(best_candidate["mapped_sample_count"])
    sample_axis_count = int(best_candidate["sample_axis_count"])
    mapped_fraction = mapped_sample_count / sample_axis_count if sample_axis_count else 0.0
    minimum_mapped_fraction = float(config["analysis_ready_registry"]["minimum_mapped_fraction"])
    minimum_mapped_sample_count = int(config["analysis_ready_registry"]["minimum_mapped_sample_count"])
    allowed_matrix_semantics = set(config["analysis_ready_registry"]["allowed_matrix_semantics"])
    if best_candidate["matrix_semantics"] not in allowed_matrix_semantics:
        return f"matrix_semantics_not_allowed_{best_candidate['matrix_semantics']}"
    if mapped_sample_count < minimum_mapped_sample_count:
        return f"insufficient_mapped_samples_{mapped_sample_count}"
    if mapped_fraction < minimum_mapped_fraction:
        return f"insufficient_sample_mapping_{mapped_sample_count}_of_{sample_axis_count}"
    return "no_eligible_matrix"


def loader_kind(matrix_origin: str, config: dict) -> str:
    loader_kind_by_origin = config["analysis_ready_registry"]["loader_kind_by_origin"]
    if matrix_origin not in loader_kind_by_origin:
        raise ValueError(f"Unsupported matrix_origin for analysis-ready loader: {matrix_origin}")
    return str(loader_kind_by_origin[matrix_origin])


def build_outputs(
    matrix_rows: list[dict[str, str]],
    sample_axis_rows: list[dict[str, str]],
    feature_axis_rows: list[dict[str, str]],
    feature_study_rows: list[dict[str, str]],
    cohort_sample_rows: list[dict[str, str]],
    cohort_study_rows: list[dict[str, str]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    matrix_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_axis_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    feature_axis_rows_by_matrix: dict[str, list[dict[str, str]]] = defaultdict(list)
    cohort_sample_by_uid = {row["sample_uid"]: row for row in cohort_sample_rows}
    cohort_study_by_uid = {row["study_uid"]: row for row in cohort_study_rows}

    for matrix_row in matrix_rows:
        matrix_rows_by_study[matrix_row["study_uid"]].append(matrix_row)
    for sample_axis_row in sample_axis_rows:
        sample_axis_rows_by_matrix[sample_axis_row["matrix_uid"]].append(sample_axis_row)
    for feature_axis_row in feature_axis_rows:
        feature_axis_rows_by_matrix[feature_axis_row["matrix_uid"]].append(feature_axis_row)

    study_summary_rows: list[dict[str, object]] = []
    loader_rows: list[dict[str, object]] = []
    analysis_sample_rows: list[dict[str, object]] = []
    analysis_feature_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []

    for feature_study_row in feature_study_rows:
        study_uid = feature_study_row["study_uid"]
        cohort_study_row = cohort_study_by_uid[study_uid]
        study_matrix_rows = matrix_rows_by_study.get(study_uid, [])
        eligible_rows = eligible_matrix_rows(study_matrix_rows, config)
        selected_matrix_row = choose_selected_matrix(feature_study_row, study_matrix_rows, config)
        if selected_matrix_row is None:
            exclusion_reason = build_exclusion_reason(study_matrix_rows, feature_study_row, config)
            note = feature_study_row["primary_matrix_path"] or feature_study_row["source_accession"]
            study_summary_rows.append(
                {
                    "study_uid": study_uid,
                    "source_key": feature_study_row["source_key"],
                    "source_accession": feature_study_row["source_accession"],
                    "standardized_sample_count": cohort_study_row["standardized_sample_count"],
                    "local_quantitative_asset_count": cohort_study_row["local_quantitative_asset_count"],
                    "candidate_matrix_count": len(study_matrix_rows),
                    "eligible_matrix_count": len(eligible_rows),
                    "selected_matrix_uid": "",
                    "selected_matrix_reference": "",
                    "selected_measurement_label": "",
                    "selected_matrix_semantics": "",
                    "mapped_sample_uid_count": "0",
                    "mapped_fraction": "0.000000",
                    "loader_kind": "",
                    "analysis_ready_flag": "no",
                    "exclusion_reason": exclusion_reason,
                    "note": note,
                }
            )
            exclusion_rows.append(
                {
                    "study_uid": study_uid,
                    "source_key": feature_study_row["source_key"],
                    "source_accession": feature_study_row["source_accession"],
                    "local_quantitative_asset_count": cohort_study_row["local_quantitative_asset_count"],
                    "candidate_matrix_count": len(study_matrix_rows),
                    "exclusion_reason": exclusion_reason,
                    "note": note,
                }
            )
            continue

        selected_matrix_sample_axis_rows = sample_axis_rows_by_matrix[selected_matrix_row["matrix_uid"]]
        mapped_sample_axis_rows = [row for row in selected_matrix_sample_axis_rows if row["sample_uid"]]
        mapped_sample_uid_count = len({row["sample_uid"] for row in mapped_sample_axis_rows})
        mapped_fraction = format_fraction(len(mapped_sample_axis_rows), int(selected_matrix_row["sample_axis_count"]))
        current_loader_kind = loader_kind(selected_matrix_row["matrix_origin"], config)
        study_summary_rows.append(
            {
                "study_uid": study_uid,
                "source_key": feature_study_row["source_key"],
                "source_accession": feature_study_row["source_accession"],
                "standardized_sample_count": cohort_study_row["standardized_sample_count"],
                "local_quantitative_asset_count": cohort_study_row["local_quantitative_asset_count"],
                "candidate_matrix_count": len(study_matrix_rows),
                "eligible_matrix_count": len(eligible_rows),
                "selected_matrix_uid": selected_matrix_row["matrix_uid"],
                "selected_matrix_reference": selected_matrix_row["matrix_reference"],
                "selected_measurement_label": selected_matrix_row["measurement_label"],
                "selected_matrix_semantics": selected_matrix_row["matrix_semantics"],
                "mapped_sample_uid_count": mapped_sample_uid_count,
                "mapped_fraction": mapped_fraction,
                "loader_kind": current_loader_kind,
                "analysis_ready_flag": "yes",
                "exclusion_reason": "",
                "note": selected_matrix_row["note"],
            }
        )
        loader_rows.append(
            {
                "selected_matrix_uid": selected_matrix_row["matrix_uid"],
                "study_uid": study_uid,
                "source_key": feature_study_row["source_key"],
                "source_accession": feature_study_row["source_accession"],
                "loader_kind": current_loader_kind,
                "matrix_origin": selected_matrix_row["matrix_origin"],
                "local_path": selected_matrix_row["local_path"],
                "member_path": selected_matrix_row["member_path"],
                "matrix_reference": selected_matrix_row["matrix_reference"],
                "file_format": selected_matrix_row["file_format"],
                "sample_identifier_field": selected_matrix_row["sample_identifier_field"],
                "feature_identifier_field": selected_matrix_row["feature_identifier_field"],
                "feature_label_field": selected_matrix_row["feature_label_field"],
                "value_identifier": selected_matrix_row["value_identifier"],
                "selected_measurement_label": selected_matrix_row["measurement_label"],
                "selected_matrix_semantics": selected_matrix_row["matrix_semantics"],
                "raw_metadata_path": selected_matrix_row["raw_metadata_path"],
                "note": selected_matrix_row["note"],
            }
        )
        for sample_axis_row in mapped_sample_axis_rows:
            cohort_sample_row = cohort_sample_by_uid[sample_axis_row["sample_uid"]]
            analysis_sample_rows.append(
                {
                    "study_uid": study_uid,
                    "source_key": feature_study_row["source_key"],
                    "source_accession": feature_study_row["source_accession"],
                    "selected_matrix_uid": selected_matrix_row["matrix_uid"],
                    "column_index": sample_axis_row["column_index"],
                    "matrix_sample_id": sample_axis_row["matrix_sample_id"],
                    "sample_uid": sample_axis_row["sample_uid"],
                    "source_sample_id": cohort_sample_row["source_sample_id"],
                    "subject_id": cohort_sample_row["subject_id"],
                    "sample_title": cohort_sample_row["sample_title"],
                    "sample_diagnosis_label": cohort_sample_row["sample_diagnosis_label"],
                    "case_control_label": cohort_sample_row["case_control_label"],
                    "tissue_or_biofluid": cohort_sample_row["tissue_or_biofluid"],
                    "modality": cohort_sample_row["modality"],
                    "sample_match_status": sample_axis_row["sample_match_status"],
                    "note": cohort_sample_row["note"],
                }
            )
        for feature_axis_row in feature_axis_rows_by_matrix[selected_matrix_row["matrix_uid"]]:
            analysis_feature_rows.append(
                {
                    "study_uid": study_uid,
                    "source_key": feature_study_row["source_key"],
                    "source_accession": feature_study_row["source_accession"],
                    "selected_matrix_uid": selected_matrix_row["matrix_uid"],
                    "feature_id": feature_axis_row["feature_id"],
                    "feature_label": feature_axis_row["feature_label"],
                    "row_index": feature_axis_row["row_index"],
                    "feature_type": feature_axis_row["feature_type"],
                    "note": feature_axis_row["note"],
                }
            )

    study_summary_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    loader_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    analysis_sample_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], int(row["column_index"])))
    analysis_feature_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], int(row["row_index"])))
    exclusion_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    return study_summary_rows, loader_rows, analysis_sample_rows, analysis_feature_rows, exclusion_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accessions(args.source_accessions)
    analysis_ready_root = resolve_repo_path(config["analysis_ready_registry"]["local_root"])
    ensure_directory(analysis_ready_root)

    matrix_rows, sample_axis_rows, feature_axis_rows, feature_study_rows, cohort_sample_rows, cohort_study_rows = load_filtered_rows(config, source_accessions)
    pipeline_logger.progress(
        "loaded_analysis_ready_inputs",
        matrix_row_count=len(matrix_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        feature_study_row_count=len(feature_study_rows),
        cohort_sample_row_count=len(cohort_sample_rows),
        cohort_study_row_count=len(cohort_study_rows),
    )
    study_summary_rows, loader_rows, analysis_sample_rows, analysis_feature_rows, exclusion_rows = build_outputs(
        matrix_rows,
        sample_axis_rows,
        feature_axis_rows,
        feature_study_rows,
        cohort_sample_rows,
        cohort_study_rows,
        config,
    )

    relative_paths = config["analysis_ready_registry"]["relative_paths"]
    write_tsv(
        analysis_ready_root / relative_paths["study_summary"],
        config["schemas"]["analysis_ready_study_fields"],
        study_summary_rows,
    )
    write_tsv(
        analysis_ready_root / relative_paths["loader_catalog"],
        config["schemas"]["analysis_ready_loader_fields"],
        loader_rows,
    )
    write_tsv(
        analysis_ready_root / relative_paths["sample_catalog"],
        config["schemas"]["analysis_ready_sample_fields"],
        analysis_sample_rows,
    )
    write_tsv(
        analysis_ready_root / relative_paths["feature_catalog"],
        config["schemas"]["analysis_ready_feature_fields"],
        analysis_feature_rows,
    )
    write_tsv(
        analysis_ready_root / relative_paths["exclusion_catalog"],
        config["schemas"]["analysis_ready_exclusion_fields"],
        exclusion_rows,
    )
    pipeline_logger.artifact(
        "analysis_ready_registry_written",
        artifact=analysis_ready_root,
        study_row_count=len(study_summary_rows),
        loader_row_count=len(loader_rows),
        sample_row_count=len(analysis_sample_rows),
        feature_row_count=len(analysis_feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=analysis_ready_root,
        study_row_count=len(study_summary_rows),
        loader_row_count=len(loader_rows),
        sample_row_count=len(analysis_sample_rows),
        feature_row_count=len(analysis_feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
