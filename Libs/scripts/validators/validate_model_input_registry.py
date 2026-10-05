from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the model-input study bundles.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--source-accessions",
        default="",
        help="Optional comma-separated source_accession subset for smoke tests.",
    )
    return parser.parse_args()


def selected_accessions(accession_text: str) -> set[str]:
    return {value.strip() for value in accession_text.split(",") if value.strip()}


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def filtered_rows(rows: list[dict[str, str]], source_accessions: set[str]) -> list[dict[str, str]]:
    if not source_accessions:
        return rows
    return [row for row in rows if row["source_accession"] in source_accessions]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accessions(args.source_accessions)

    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    relative_paths = config["model_input_registry"]["relative_paths"]
    study_bundle_catalog_path = model_input_root / relative_paths["study_bundle_catalog"]
    excluded_study_catalog_path = model_input_root / relative_paths["excluded_study_catalog"]

    for output_path in [study_bundle_catalog_path, excluded_study_catalog_path]:
        if not output_path.exists():
            raise FileNotFoundError(f"Model-input output is missing: {output_path}")

    validate_header(study_bundle_catalog_path, config["schemas"]["model_input_bundle_fields"])
    validate_header(excluded_study_catalog_path, config["schemas"]["model_input_exclusion_fields"])

    bundle_rows = filtered_rows(read_tsv(study_bundle_catalog_path), source_accessions)
    exclusion_rows = filtered_rows(read_tsv(excluded_study_catalog_path), source_accessions)

    analysis_root = resolve_repo_path(config["analysis_ready_registry"]["local_root"])
    analysis_paths = config["analysis_ready_registry"]["relative_paths"]
    analysis_study_rows = filtered_rows(read_tsv(analysis_root / analysis_paths["study_summary"]), source_accessions)
    pipeline_logger.progress(
        "loaded_model_input_inputs",
        bundle_row_count=len(bundle_rows),
        exclusion_row_count=len(exclusion_rows),
        analysis_study_row_count=len(analysis_study_rows),
    )

    ready_analysis_rows = [row for row in analysis_study_rows if row["analysis_ready_flag"] == "yes"]
    excluded_analysis_rows = [row for row in analysis_study_rows if row["analysis_ready_flag"] == "no"]
    ready_analysis_by_study = {row["study_uid"]: row for row in ready_analysis_rows}
    excluded_analysis_by_study = {row["study_uid"]: row for row in excluded_analysis_rows}

    if {row["study_uid"] for row in bundle_rows} != set(ready_analysis_by_study):
        raise ValueError("Model-input bundle catalog does not match the analysis-ready study set.")
    if {row["study_uid"] for row in exclusion_rows} != set(excluded_analysis_by_study):
        raise ValueError("Model-input exclusion catalog does not match the excluded analysis-ready study set.")

    for exclusion_row in exclusion_rows:
        analysis_row = excluded_analysis_by_study[exclusion_row["study_uid"]]
        if exclusion_row["analysis_ready_flag"] != "no":
            raise ValueError(f"Excluded model-input row must keep analysis_ready_flag=no: {exclusion_row['study_uid']}")
        if exclusion_row["bundle_materialization_flag"] != "no":
            raise ValueError(f"Excluded model-input row must keep bundle_materialization_flag=no: {exclusion_row['study_uid']}")
        if exclusion_row["exclusion_reason"] != analysis_row["exclusion_reason"]:
            raise ValueError(f"Exclusion reason mismatch for study {exclusion_row['study_uid']}")

    for bundle_row in bundle_rows:
        analysis_row = ready_analysis_by_study[bundle_row["study_uid"]]
        if bundle_row["selected_matrix_uid"] != analysis_row["selected_matrix_uid"]:
            raise ValueError(f"Selected matrix mismatch for study {bundle_row['study_uid']}")
        if bundle_row["source_matrix_reference"] != analysis_row["selected_matrix_reference"]:
            raise ValueError(f"Source matrix reference mismatch for study {bundle_row['study_uid']}")
        if bundle_row["loader_kind"] != analysis_row["loader_kind"]:
            raise ValueError(f"Loader kind mismatch for study {bundle_row['study_uid']}")
        if bundle_row["selected_measurement_label"] != analysis_row["selected_measurement_label"]:
            raise ValueError(f"Selected measurement mismatch for study {bundle_row['study_uid']}")
        if bundle_row["matrix_semantics"] != analysis_row["selected_matrix_semantics"]:
            raise ValueError(f"Matrix semantics mismatch for study {bundle_row['study_uid']}")
        if not bundle_row["semantics_uid"]:
            raise ValueError(f"Model-input bundle is missing semantics_uid for study {bundle_row['study_uid']}")
        if not bundle_row["measurement_semantics_label"]:
            raise ValueError(f"Model-input bundle is missing measurement_semantics_label for study {bundle_row['study_uid']}")
        if bundle_row["scaler_fit_policy"] != config["measurement_semantics_registry"]["scaler_artifact_policy"]["production_fit_policy"]:
            raise ValueError(f"Unexpected scaler_fit_policy for study {bundle_row['study_uid']}")
        if int(bundle_row["sample_count"]) != int(analysis_row["mapped_sample_uid_count"]):
            raise ValueError(f"Sample count mismatch for study {bundle_row['study_uid']}")
        if bundle_row["world_model_role"] != "disease_observation":
            raise ValueError(f"Unexpected world_model_role for study {bundle_row['study_uid']}")
        if bundle_row["subject_linkable_flag"] != "yes":
            raise ValueError(f"Unexpected subject_linkable_flag for study {bundle_row['study_uid']}")
        if bundle_row["public_only_flag"] != "yes":
            raise ValueError(f"Unexpected public_only_flag for study {bundle_row['study_uid']}")

        bundle_root = resolve_repo_path(bundle_row["bundle_root"])
        bundle_manifest_path = resolve_repo_path(bundle_row["bundle_manifest_path"])
        sample_index_path = resolve_repo_path(bundle_row["sample_index_path"])
        feature_index_path = resolve_repo_path(bundle_row["feature_index_path"])
        matrix_raw_path = resolve_repo_path(bundle_row["matrix_raw_path"])
        matrix_norm_path = resolve_repo_path(bundle_row["matrix_norm_path"])
        scaler_artifact_path = resolve_repo_path(bundle_row["scaler_artifact_path"])

        for path in [bundle_root, bundle_manifest_path, sample_index_path, feature_index_path, matrix_raw_path, matrix_norm_path, scaler_artifact_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing model-input bundle artifact for study {bundle_row['study_uid']}: {path}")

        manifest_payload = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        if manifest_payload["study_uid"] != bundle_row["study_uid"]:
            raise ValueError(f"Manifest study_uid mismatch for study {bundle_row['study_uid']}")
        if manifest_payload["selected_matrix_uid"] != bundle_row["selected_matrix_uid"]:
            raise ValueError(f"Manifest selected_matrix_uid mismatch for study {bundle_row['study_uid']}")
        if manifest_payload["normalization_method"] != bundle_row["normalization_method"]:
            raise ValueError(f"Manifest normalization_method mismatch for study {bundle_row['study_uid']}")
        if manifest_payload["semantics_uid"] != bundle_row["semantics_uid"]:
            raise ValueError(f"Manifest semantics_uid mismatch for study {bundle_row['study_uid']}")
        if manifest_payload["measurement_semantics_label"] != bundle_row["measurement_semantics_label"]:
            raise ValueError(f"Manifest measurement_semantics_label mismatch for study {bundle_row['study_uid']}")
        if manifest_payload["scaler_fit_policy"] != bundle_row["scaler_fit_policy"]:
            raise ValueError(f"Manifest scaler_fit_policy mismatch for study {bundle_row['study_uid']}")
        if manifest_payload["scaler_artifact_path"] != bundle_row["scaler_artifact_path"]:
            raise ValueError(f"Manifest scaler_artifact_path mismatch for study {bundle_row['study_uid']}")
        for field_name in ["world_model_role", "subject_linkable_flag", "atlas_prior_ready_flag", "public_only_flag"]:
            if manifest_payload[field_name] != bundle_row[field_name]:
                raise ValueError(f"Manifest {field_name} mismatch for study {bundle_row['study_uid']}")

        sample_index = pd.read_parquet(sample_index_path)
        feature_index = pd.read_parquet(feature_index_path)
        matrix_raw = np.load(matrix_raw_path, allow_pickle=False)
        matrix_norm = np.load(matrix_norm_path, allow_pickle=False)
        validate_header(scaler_artifact_path, config["schemas"]["model_input_feature_scaler_fields"])
        scaler_rows = read_tsv(scaler_artifact_path)

        if list(sample_index.columns) != config["schemas"]["model_input_sample_index_fields"]:
            raise ValueError(f"Sample index schema mismatch for study {bundle_row['study_uid']}")
        if list(feature_index.columns) != config["schemas"]["model_input_feature_index_fields"]:
            raise ValueError(f"Feature index schema mismatch for study {bundle_row['study_uid']}")
        if sample_index.shape[0] != int(bundle_row["sample_count"]):
            raise ValueError(f"Sample index row count mismatch for study {bundle_row['study_uid']}")
        if feature_index.shape[0] != int(bundle_row["feature_count"]):
            raise ValueError(f"Feature index row count mismatch for study {bundle_row['study_uid']}")

        if matrix_raw.shape != (int(bundle_row["sample_count"]), int(bundle_row["feature_count"])):
            raise ValueError(f"Raw matrix shape mismatch for study {bundle_row['study_uid']}")
        if matrix_norm.shape != (int(bundle_row["sample_count"]), int(bundle_row["feature_count"])):
            raise ValueError(f"Normalized matrix shape mismatch for study {bundle_row['study_uid']}")
        if matrix_raw.dtype.kind != "f":
            raise ValueError(f"Raw matrix dtype must be floating-point for study {bundle_row['study_uid']}")
        if matrix_norm.dtype.kind != "f":
            raise ValueError(f"Normalized matrix dtype must be floating-point for study {bundle_row['study_uid']}")
        if len(scaler_rows) != int(bundle_row["feature_count"]):
            raise ValueError(f"Scaler artifact row count mismatch for study {bundle_row['study_uid']}")
        scaler_feature_keys = {row["feature_key"] for row in scaler_rows}
        if len(scaler_feature_keys) != len(scaler_rows):
            raise ValueError(f"Scaler artifact contains duplicate feature_key values for study {bundle_row['study_uid']}")
        if scaler_feature_keys != set(feature_index["feature_key"].astype(str).tolist()):
            raise ValueError(f"Scaler artifact feature keys do not match feature index for study {bundle_row['study_uid']}")
        for scaler_row in scaler_rows:
            if scaler_row["semantics_uid"] != bundle_row["semantics_uid"]:
                raise ValueError(f"Scaler row semantics_uid mismatch for study {bundle_row['study_uid']}")
            if scaler_row["leakage_guard"] != config["measurement_semantics_registry"]["scaler_artifact_policy"]["leakage_guard"]:
                raise ValueError(f"Scaler row leakage guard mismatch for study {bundle_row['study_uid']}")
        finite_positions = np.argwhere(np.isfinite(matrix_raw) & np.isfinite(matrix_norm))
        if finite_positions.size:
            sample_position, feature_position = [int(value) for value in finite_positions[0].tolist()]
            scaler_row = scaler_rows[feature_position]
            transformed_value = float(matrix_norm[sample_position, feature_position]) * float(scaler_row["std"]) + float(scaler_row["mean"])
            if scaler_row["transform_before_zscore"] == "log1p":
                reconstructed_value = float(np.expm1(transformed_value))
            elif scaler_row["transform_before_zscore"] == "identity":
                reconstructed_value = transformed_value
            else:
                raise ValueError(f"Unsupported scaler transform_before_zscore: {scaler_row}")
            if not np.isclose(reconstructed_value, float(matrix_raw[sample_position, feature_position]), rtol=1.0e-5, atol=1.0e-5):
                raise ValueError(f"Raw-standardized-original roundtrip failed for study {bundle_row['study_uid']}")

    pipeline_logger.progress(
        "model_input_validated",
        artifact=study_bundle_catalog_path,
        bundle_row_count=len(bundle_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=study_bundle_catalog_path,
        bundle_row_count=len(bundle_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
