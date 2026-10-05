from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.controlled_research_config import (
    assert_controlled_governance_approval,
    load_controlled_research_config,
)
from Libs.utils.data.raw_data_registry import repo_root, sha256_of_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate controlled-access ADNI registry outputs.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def adni_section(config: dict) -> dict:
    return config["adni_controlled_access"]


def resolve_configured_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def controlled_paths(config: dict) -> tuple[Path, Path]:
    paths_config = adni_section(config)["paths"]
    return (
        resolve_configured_path(paths_config["private_root"]),
        resolve_configured_path(paths_config["processed_root"]),
    )


def validate_tsv_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def read_tsv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    resolved_path = path.resolve()
    root = repo_root()
    if resolved_path.is_relative_to(root):
        return resolved_path.relative_to(root).as_posix()
    return str(resolved_path)


def validate_public_boundary(config: dict) -> None:
    if config["source_policy"]["strict_public_only_flag"] is not True:
        raise ValueError("ADNI controlled-access integration requires source_policy.strict_public_only_flag to remain true.")
    if config["source_policy"]["imaging_enabled_flag"] is not False:
        raise ValueError("ADNI imaging must remain outside the public-only imaging flag.")
    governance = adni_section(config)["governance"]
    if governance["publication_boundary"] != "preserve_public_only_mainline":
        raise ValueError("ADNI governance must preserve the public-only publication boundary.")
    if governance["public_release_policy"] != "aggregate_summary_only":
        raise ValueError("ADNI public release policy must be aggregate_summary_only.")


def validate_governance_evidence(config: dict) -> None:
    private_root, _ = controlled_paths(config)
    adni_config = adni_section(config)
    paths_config = adni_config["paths"]
    governance_config = adni_config["governance"]
    evidence_config = governance_config["evidence"]
    report_path = private_root / paths_config["access_governance_report"]
    if not report_path.exists():
        raise FileNotFoundError(f"Required ADNI access-governance report is missing: {display_path(report_path)}")
    report = read_json(report_path)
    if report["participant_level_flag"] != "aggregate_governance_only":
        raise ValueError("ADNI access-governance report must be aggregate_governance_only.")
    if report["document_status"] != "complete":
        raise ValueError(f"ADNI access-governance report is incomplete: {report['missing_required_document_paths']}")
    governance_root = private_root / str(evidence_config["governance_root"])
    for document_config in evidence_config["required_documents"]:
        document_path = governance_root / str(document_config["relative_path"])
        if bool(document_config["required"]) and not document_path.exists():
            raise FileNotFoundError(f"Missing required ADNI governance document: {display_path(document_path)}")
    reported_documents = {str(row["document_id"]): row for row in report["documents"]}
    for document_config in evidence_config["required_documents"]:
        document_id = str(document_config["document_id"])
        if document_id not in reported_documents:
            raise ValueError(f"ADNI access-governance report omits document: {document_id}")
        document_path = governance_root / str(document_config["relative_path"])
        reported_hash = str(reported_documents[document_id]["sha256"])
        if reported_hash != sha256_of_file(document_path):
            raise ValueError(f"ADNI governance document hash mismatch: {display_path(document_path)}")
    if report["public_release_policy"] != governance_config["public_release_policy"]:
        raise ValueError("ADNI access-governance report public-release policy does not match config.")


def validate_raw_manifest_checksums(config: dict) -> None:
    private_root, _ = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    manifest_path = private_root / paths_config["raw_manifest"]
    validate_tsv_header(manifest_path, config["schemas"]["adni_raw_manifest_fields"])
    frame = read_tsv_frame(manifest_path)
    if frame.empty:
        raise ValueError("ADNI raw manifest must not be empty.")
    for row in frame.to_dict(orient="records"):
        local_path = resolve_configured_path(row["local_path_private"])
        if not local_path.exists():
            raise FileNotFoundError("An ADNI raw-manifest asset is missing from the private namespace.")
        if sha256_of_file(local_path) != row["checksum_sha256"]:
            raise ValueError("An ADNI raw-manifest checksum does not match the private asset.")
        if row["participant_level_flag"] != "yes":
            raise ValueError("ADNI raw-manifest assets must be marked participant-level.")


def required_table_roles(config: dict) -> list[str]:
    sync_config = adni_section(config)["sync"]
    scoped_unavailable = {str(value) for value in sync_config.get("scoped_unavailable_table_roles", [])}
    return [str(value) for value in sync_config["required_table_roles"] if str(value) not in scoped_unavailable]


def validate_required_role_coverage(config: dict) -> None:
    private_root, _ = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    manifest_path = private_root / paths_config["raw_manifest"]
    frame = read_tsv_frame(manifest_path)
    covered_roles = set(frame["table_role"].astype(str))
    missing_roles = sorted(set(required_table_roles(config)) - covered_roles)
    if missing_roles:
        raise ValueError(f"ADNI raw manifest is missing required table roles: {missing_roles}")
    missing_audit_path = private_root / paths_config["missing_role_audit"]
    missing_audit = read_json(missing_audit_path)
    if missing_audit["missing_required_table_roles"]:
        raise ValueError(f"ADNI missing-role audit reports missing roles: {missing_audit['missing_required_table_roles']}")


def validate_processed_schemas(config: dict) -> dict[str, pd.DataFrame]:
    _, processed_root = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    sample_path = processed_root / paths_config["sample_observation_registry"]
    transition_path = processed_root / paths_config["longitudinal_transition_table"]
    imaging_path = processed_root / paths_config["imaging_inventory"]
    medication_path = processed_root / paths_config["medication_exposure"]
    domain_inventory_path = processed_root / paths_config["domain_inventory"]
    feature_path = processed_root / paths_config["feature_matrix"]
    sample_index_path = processed_root / paths_config["sample_index"]
    feature_index_path = processed_root / paths_config["feature_index"]
    transition_label_path = processed_root / paths_config["transition_label_index"]
    validate_tsv_header(sample_path, config["schemas"]["adni_sample_observation_fields"])
    validate_tsv_header(transition_path, config["schemas"]["adni_longitudinal_transition_fields"])
    validate_tsv_header(imaging_path, config["schemas"]["adni_imaging_inventory_fields"])
    validate_tsv_header(medication_path, config["schemas"]["adni_medication_exposure_fields"])
    validate_tsv_header(domain_inventory_path, config["schemas"]["adni_domain_inventory_fields"])
    sample_frame = read_tsv_frame(sample_path)
    transition_frame = read_tsv_frame(transition_path)
    imaging_frame = read_tsv_frame(imaging_path)
    medication_frame = read_tsv_frame(medication_path)
    domain_inventory_frame = read_tsv_frame(domain_inventory_path)
    feature_frame = pd.read_parquet(feature_path)
    sample_index_frame = pd.read_parquet(sample_index_path)
    feature_index_frame = pd.read_parquet(feature_index_path)
    transition_label_frame = pd.read_parquet(transition_label_path)
    expected_feature_fields = config["schemas"]["adni_feature_matrix_fields"]
    if list(feature_frame.columns) != expected_feature_fields:
        raise ValueError(f"{feature_path} columns mismatch: {list(feature_frame.columns)}")
    if list(sample_index_frame.columns) != config["schemas"]["adni_sample_observation_fields"]:
        raise ValueError(f"{sample_index_path} columns mismatch: {list(sample_index_frame.columns)}")
    if list(feature_index_frame.columns) != config["schemas"]["adni_feature_index_fields"]:
        raise ValueError(f"{feature_index_path} columns mismatch: {list(feature_index_frame.columns)}")
    if list(transition_label_frame.columns) != config["schemas"]["adni_longitudinal_transition_fields"]:
        raise ValueError(f"{transition_label_path} columns mismatch: {list(transition_label_frame.columns)}")
    if sample_frame.empty:
        raise ValueError("ADNI sample-observation registry must not be empty.")
    if feature_frame.empty:
        raise ValueError("ADNI feature matrix must not be empty.")
    return {
        "sample": sample_frame,
        "transition": transition_frame,
        "imaging": imaging_frame,
        "medication": medication_frame,
        "domain_inventory": domain_inventory_frame,
        "feature": feature_frame,
        "sample_index": sample_index_frame,
        "feature_index": feature_index_frame,
        "transition_label": transition_label_frame,
    }


def collect_raw_identifier_values(config: dict) -> set[str]:
    private_root, _ = controlled_paths(config)
    builder_config = adni_section(config)["builder"]
    identifier_columns = set()
    for column_group in [
        builder_config["column_aliases"]["subject_identifier"],
        builder_config["column_aliases"]["participant_identifier"],
        builder_config["column_aliases"]["site_identifier"],
        builder_config["imaging_inventory_columns"]["subject_identifier"],
        builder_config["imaging_inventory_columns"]["participant_identifier"],
        builder_config["imaging_inventory_columns"]["site_identifier"],
    ]:
        identifier_columns.update(str(value) for value in column_group)
    minimum_length = int(adni_section(config)["validators"]["raw_identifier_min_length"])
    identifiers: set[str] = set()
    for table_config in builder_config["table_inputs"].values():
        table_path = private_root / table_config["local_relative_path"]
        if not table_path.exists():
            continue
        frame = pd.read_csv(table_path, sep=str(table_config["delimiter"]), dtype=str, keep_default_na=False)
        for column in sorted(identifier_columns & set(frame.columns)):
            for value in frame[column].astype(str):
                stripped_value = value.strip()
                if len(stripped_value) >= minimum_length:
                    identifiers.add(stripped_value)
    return identifiers


def validate_no_raw_identifier_columns(config: dict, frames: dict[str, pd.DataFrame]) -> None:
    builder_config = adni_section(config)["builder"]
    forbidden_columns = set()
    for column_group in [
        builder_config["column_aliases"]["subject_identifier"],
        builder_config["column_aliases"]["participant_identifier"],
        builder_config["column_aliases"]["site_identifier"],
        builder_config["imaging_inventory_columns"]["subject_identifier"],
        builder_config["imaging_inventory_columns"]["participant_identifier"],
        builder_config["imaging_inventory_columns"]["site_identifier"],
    ]:
        forbidden_columns.update(str(value) for value in column_group)
    for frame_name, frame in frames.items():
        overlap = forbidden_columns & set(frame.columns)
        if overlap:
            raise ValueError(f"ADNI processed {frame_name} output exposes raw identifier columns: {sorted(overlap)}")


def validate_no_identifier_values(frames: dict[str, pd.DataFrame], raw_identifiers: set[str]) -> None:
    if not raw_identifiers:
        return
    for frame_name, frame in frames.items():
        object_frame = frame.select_dtypes(include=["object"])
        match_count = 0
        for column in object_frame.columns:
            match_count += int(object_frame[column].astype(str).isin(raw_identifiers).sum())
        if match_count:
            raise ValueError(f"ADNI processed {frame_name} output contains {match_count} raw identifier value matches.")


def validate_split_leakage(sample_frame: pd.DataFrame, split_audit: dict) -> None:
    split_count_by_subject = sample_frame.groupby("subject_uid_hash")["benchmark_split"].nunique()
    leakage_count = int((split_count_by_subject > 1).sum())
    if leakage_count:
        raise ValueError(f"ADNI subject-holdout leakage count is nonzero: {leakage_count}")
    if int(split_audit["subject_holdout_leakage_count"]) != 0:
        raise ValueError("ADNI split audit reports nonzero subject-holdout leakage.")


def validate_json_audits(config: dict) -> dict[str, dict]:
    _, processed_root = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    missingness_audit = read_json(processed_root / paths_config["missingness_audit"])
    split_audit = read_json(processed_root / paths_config["split_audit"])
    model_input_registry = read_json(processed_root / paths_config["model_input_registry"])
    controlled_application_audit = read_json(processed_root / paths_config["controlled_application_audit"])
    controlled_bundle_pointer = read_json(resolve_configured_path(paths_config["controlled_bundle_pointer"]))
    if missingness_audit["participant_level_flag"] != "aggregate_only":
        raise ValueError("ADNI missingness audit must be aggregate-only.")
    if split_audit["participant_level_flag"] != "aggregate_only":
        raise ValueError("ADNI split audit must be aggregate-only.")
    for required_key in [
        "unique_site_counts_by_split",
        "adni_phase_distribution_by_split",
        "diagnosis_transition_counts_by_split",
        "modality_missingness_by_split",
    ]:
        if required_key not in split_audit:
            raise ValueError(f"ADNI split audit is missing required controlled-bundle field: {required_key}")
    if model_input_registry["access_boundary"] != "controlled_access":
        raise ValueError("ADNI model-input registry must declare controlled_access.")
    if controlled_application_audit["participant_level_flag"] != "aggregate_only":
        raise ValueError("ADNI controlled-application audit must be aggregate-only.")
    required_roles = required_table_roles(config)
    role_counts = controlled_application_audit["table_role_row_counts"]
    missing_role_counts = [role for role in required_roles if role not in role_counts or int(role_counts[role]) <= 0]
    if missing_role_counts:
        raise ValueError(f"ADNI controlled-application audit lacks positive row counts for roles: {missing_role_counts}")
    if controlled_bundle_pointer["participant_level_flag"] != "aggregate_only":
        raise ValueError("ADNI controlled-bundle pointer must be aggregate-only.")
    if controlled_bundle_pointer["access_boundary"] != "controlled_access_aggregate_pointer":
        raise ValueError("ADNI controlled-bundle pointer must declare controlled_access_aggregate_pointer.")
    pointer_roles = controlled_bundle_pointer["role_counts"]
    pointer_missing_roles = [role for role in required_roles if role not in pointer_roles or int(pointer_roles[role]) <= 0]
    if pointer_missing_roles:
        raise ValueError(f"ADNI controlled-bundle pointer lacks positive row counts for roles: {pointer_missing_roles}")
    return {
        "missingness": missingness_audit,
        "split": split_audit,
        "model_input": model_input_registry,
        "controlled_application": controlled_application_audit,
        "controlled_bundle_pointer": controlled_bundle_pointer,
    }


def validate_model_input_bundle(config: dict, frames: dict[str, pd.DataFrame]) -> None:
    _, processed_root = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    matrix_raw = np.load(processed_root / paths_config["matrix_raw"], allow_pickle=False)
    matrix_norm = np.load(processed_root / paths_config["matrix_norm"], allow_pickle=False)
    missing_mask = np.load(processed_root / paths_config["matrix_missing_mask"], allow_pickle=False)
    expected_shape = (frames["sample_index"].shape[0], frames["feature_index"].shape[0])
    if matrix_raw.shape != expected_shape:
        raise ValueError(f"ADNI raw model-input matrix shape mismatch: {matrix_raw.shape} != {expected_shape}")
    if matrix_norm.shape != expected_shape:
        raise ValueError(f"ADNI normalized model-input matrix shape mismatch: {matrix_norm.shape} != {expected_shape}")
    if missing_mask.shape != expected_shape:
        raise ValueError(f"ADNI missing-mask matrix shape mismatch: {missing_mask.shape} != {expected_shape}")
    scaler_payload = read_json(processed_root / paths_config["train_scaler"])
    if scaler_payload["fit_split"] != "train":
        raise ValueError("ADNI controlled train-only scaler must declare fit_split=train.")
    if int(scaler_payload["feature_count"]) != expected_shape[1]:
        raise ValueError("ADNI controlled train-only scaler feature count does not match the feature index.")
    for row in scaler_payload["scaler_rows"]:
        if row["fit_split"] != "train":
            raise ValueError("Every ADNI controlled scaler row must declare fit_split=train.")
    if not np.isfinite(matrix_norm).all():
        raise ValueError("ADNI normalized model-input matrix contains non-finite values.")


def validate_world_state_application_artifacts(config: dict) -> None:
    _, processed_root = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    prediction_path = processed_root / paths_config["world_state_predictions"]
    projection_report_path = processed_root / paths_config["state_projection_report"]
    if not prediction_path.exists():
        raise FileNotFoundError(f"Required private ADNI world-state prediction artifact is missing: {prediction_path}")
    if not projection_report_path.exists():
        raise FileNotFoundError(f"Required ADNI state-projection report is missing: {projection_report_path}")
    prediction_frame = pd.read_parquet(prediction_path)
    if list(prediction_frame.columns) != config["schemas"]["adni_world_state_prediction_fields"]:
        raise ValueError(f"{prediction_path} columns mismatch: {list(prediction_frame.columns)}")
    projection_report = read_json(projection_report_path)
    if projection_report["participant_level_flag"] != "aggregate_projection_only":
        raise ValueError("ADNI state-projection report must be aggregate_projection_only.")
    if projection_report["fit_split"] != adni_section(config)["application"]["adni_tabular_adapter"]["fit_split"]:
        raise ValueError("ADNI state-projection fit split does not match the YAML-declared adapter.")


def walk_json_keys(payload: object) -> list[str]:
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.append(str(key))
            keys.extend(walk_json_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            keys.extend(walk_json_keys(value))
    return keys


def validate_controlled_application_reports(config: dict) -> None:
    validator_config = adni_section(config)["validators"]
    forbidden_keys = {str(value) for value in validator_config["forbidden_public_json_keys"]}
    paths_config = adni_section(config)["paths"]
    for report_key in [
        "state_inference_summary",
        "longitudinal_validation_summary",
        "missing_modality_robustness",
        "world_state_factor_anchoring",
        "posterior_calibration_summary",
        "controlled_bundle_pointer",
    ]:
        report_path = resolve_configured_path(paths_config[report_key])
        if not report_path.exists():
            raise FileNotFoundError(f"Required aggregate ADNI controlled report is missing: {report_path}")
        payload = read_json(report_path)
        if payload.get("participant_level_flag") != "aggregate_only":
            raise ValueError(f"ADNI controlled report must be aggregate-only: {report_path}")
        overlapping_keys = forbidden_keys & set(walk_json_keys(payload))
        if overlapping_keys:
            raise ValueError(f"ADNI controlled public report exposes participant-level keys: {sorted(overlapping_keys)}")


def validate_public_scan_paths(config: dict, raw_identifiers: set[str]) -> None:
    validator_config = adni_section(config)["validators"]
    suffixes = tuple(str(value) for value in validator_config["public_text_scan_suffixes"])
    forbidden_fragments = [str(value) for value in validator_config["forbidden_public_path_fragments"]]
    public_identifier_patterns = [re.compile(str(value)) for value in validator_config.get("public_raw_identifier_value_patterns", [])]
    public_identifiers = {
        identifier
        for identifier in raw_identifiers
        if not public_identifier_patterns or any(pattern.fullmatch(identifier) for pattern in public_identifier_patterns)
    }
    for root_value in validator_config["public_scan_roots"]:
        root_path = resolve_configured_path(str(root_value))
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            relative_text = path.relative_to(repo_root()).as_posix() if path.is_relative_to(repo_root()) else str(path)
            if any(fragment in relative_text for fragment in forbidden_fragments):
                raise ValueError(f"ADNI participant-level artifact path is present under a public scan root: {relative_text}")
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if any(identifier in line for identifier in public_identifiers):
                        raise ValueError(f"ADNI raw identifier content is present under a public scan root: {relative_text}")


def validate_adni_outputs(config: dict) -> None:
    validate_public_boundary(config)
    validate_governance_evidence(config)
    validate_raw_manifest_checksums(config)
    if bool(adni_section(config)["validators"]["require_required_role_coverage"]):
        validate_required_role_coverage(config)
    frames = validate_processed_schemas(config)
    audits = validate_json_audits(config)
    raw_identifiers = collect_raw_identifier_values(config)
    validate_no_raw_identifier_columns(config, frames)
    validate_no_identifier_values(frames, raw_identifiers)
    validate_split_leakage(frames["sample"], audits["split"])
    if bool(adni_section(config)["validators"]["require_train_only_scaler"]):
        validate_model_input_bundle(config, frames)
    if bool(adni_section(config)["validators"]["require_controlled_application_reports"]):
        validate_world_state_application_artifacts(config)
    if bool(adni_section(config)["validators"]["require_controlled_application_reports"]):
        validate_controlled_application_reports(config)
    validate_public_scan_paths(config, raw_identifiers)


def main() -> None:
    args = parse_args()
    config = load_controlled_research_config(args.config)
    assert_controlled_governance_approval(config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    validate_adni_outputs(config)
    pipeline_logger.progress(
        "adni_controlled_access_validated",
        access_boundary=adni_section(config)["governance"]["access_boundary"],
        public_release_policy=adni_section(config)["governance"]["public_release_policy"],
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        access_boundary=adni_section(config)["governance"]["access_boundary"],
    )


if __name__ == "__main__":
    main()
