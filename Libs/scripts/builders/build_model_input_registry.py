from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.measurement_scaling import fit_feature_scaler, normalize_matrix, normalization_method, zscore_matrix_with_fit
from Libs.utils.data.platform_contracts import (
    measurement_semantics_definition,
    measurement_semantics_label,
    measurement_semantics_uid,
    yes_no_text,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_json, write_tsv


SECTION_PATTERN = re.compile(r"_section_(\d+)$")
MISSING_NUMERIC_TOKENS = {"", "na", "nan", "null", "n/a", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-input Parquet bundles from the analysis-ready registry.")
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


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", newline="")


def parse_tsv_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t"))


def detect_delimiter(header_line: str) -> str:
    return "\t" if header_line.count("\t") >= header_line.count(",") else ","


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def write_parquet_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    temporary_path = path.parent / f"{path.name}.part"
    dataframe.to_parquet(temporary_path, index=False, engine="pyarrow")
    temporary_path.replace(path)


def write_npy_atomic(array: np.ndarray, path: Path) -> None:
    ensure_directory(path.parent)
    temporary_path = path.parent / f"{path.name}.part"
    with temporary_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary_path.replace(path)


def parse_numeric_value(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value.lower() in MISSING_NUMERIC_TOKENS:
            return np.nan
        return float(stripped_value)
    return float(value)


def sorted_json_values(payload: dict[str, object]) -> list[dict[str, object]]:
    return [payload[key] for key in sorted(payload, key=lambda value: int(value))]


def filtered_rows(rows: list[dict[str, str]], source_accessions: set[str]) -> list[dict[str, str]]:
    if not source_accessions:
        return rows
    return [row for row in rows if row["source_accession"] in source_accessions]


def load_registry_inputs(
    config: dict,
    source_accessions: set[str],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    enabled_sources = set(config["model_input_registry"]["enabled_sources"])

    analysis_root = resolve_repo_path(config["analysis_ready_registry"]["local_root"])
    analysis_paths = config["analysis_ready_registry"]["relative_paths"]
    analysis_study_rows = read_tsv(analysis_root / analysis_paths["study_summary"])
    analysis_loader_rows = read_tsv(analysis_root / analysis_paths["loader_catalog"])
    analysis_sample_rows = read_tsv(analysis_root / analysis_paths["sample_catalog"])
    analysis_feature_rows = read_tsv(analysis_root / analysis_paths["feature_catalog"])
    analysis_exclusion_rows = read_tsv(analysis_root / analysis_paths["exclusion_catalog"])

    feature_root = resolve_repo_path(config["feature_ready_registry"]["local_root"])
    feature_member_rows = read_tsv(feature_root / config["feature_ready_registry"]["relative_paths"]["member_catalog"])

    analysis_study_rows = [row for row in analysis_study_rows if row["source_key"] in enabled_sources]
    analysis_loader_rows = [row for row in analysis_loader_rows if row["source_key"] in enabled_sources]
    analysis_sample_rows = [row for row in analysis_sample_rows if row["source_key"] in enabled_sources]
    analysis_feature_rows = [row for row in analysis_feature_rows if row["source_key"] in enabled_sources]
    analysis_exclusion_rows = [row for row in analysis_exclusion_rows if row["source_key"] in enabled_sources]
    feature_member_rows = [row for row in feature_member_rows if row["source_key"] in enabled_sources]

    return (
        filtered_rows(analysis_study_rows, source_accessions),
        filtered_rows(analysis_loader_rows, source_accessions),
        filtered_rows(analysis_sample_rows, source_accessions),
        filtered_rows(analysis_feature_rows, source_accessions),
        filtered_rows(analysis_exclusion_rows, source_accessions),
        filtered_rows(feature_member_rows, source_accessions),
    )


def ordered_sample_rows(sample_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(sample_rows, key=lambda row: int(row["column_index"]))


def ordered_feature_rows(feature_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(feature_rows, key=lambda row: int(row["row_index"]))


def feature_key(index: int, config: dict) -> str:
    return f"{config['model_input_registry']['feature_key_prefix']}{index:0{int(config['model_input_registry']['feature_key_width'])}d}"


def add_feature_keys(feature_rows: list[dict[str, str]], config: dict) -> list[dict[str, str]]:
    keyed_rows: list[dict[str, str]] = []
    for row in ordered_feature_rows(feature_rows):
        keyed_row = dict(row)
        keyed_row["feature_key"] = feature_key(int(row["row_index"]), config)
        keyed_rows.append(keyed_row)
    return keyed_rows


def sample_index_frame(sample_rows: list[dict[str, str]], config: dict) -> pd.DataFrame:
    ordered_rows = ordered_sample_rows(sample_rows)
    frame = pd.DataFrame(ordered_rows)
    return frame[config["schemas"]["model_input_sample_index_fields"]].copy()


def feature_index_frame(feature_rows: list[dict[str, str]], config: dict) -> pd.DataFrame:
    keyed_rows = add_feature_keys(feature_rows, config)
    frame = pd.DataFrame(keyed_rows)
    return frame[config["schemas"]["model_input_feature_index_fields"]].copy()


def bundle_paths(study_row: dict[str, str], config: dict) -> dict[str, Path]:
    bundle_root = resolve_repo_path(
        config["model_input_registry"]["bundle_root_template"].format(
            source_key=study_row["source_key"],
            source_accession=study_row["source_accession"],
        )
    )
    bundle_filenames = config["model_input_registry"]["bundle_filenames"]
    return {
        "bundle_root": bundle_root,
        "bundle_manifest_path": bundle_root / bundle_filenames["manifest"],
        "sample_index_path": bundle_root / bundle_filenames["sample_index"],
        "feature_index_path": bundle_root / bundle_filenames["feature_index"],
        "matrix_raw_path": bundle_root / bundle_filenames["matrix_raw"],
        "matrix_norm_path": bundle_root / bundle_filenames["matrix_norm"],
        "scaler_artifact_path": bundle_root / bundle_filenames["feature_scaler"],
    }


def prune_bundle_root(bundle_path_map: dict[str, Path]) -> None:
    bundle_root = bundle_path_map["bundle_root"]
    ensure_directory(bundle_root)
    expected_names = {path.name for key, path in bundle_path_map.items() if key != "bundle_root"}
    for path in bundle_root.iterdir():
        if path.name in expected_names:
            continue
        if path.name.endswith(".part") or path.is_file():
            path.unlink()


def zscore_matrix(values: np.ndarray) -> np.ndarray:
    normalized, _ = zscore_matrix_with_fit(values)
    return normalized


def scalar_text(value: float | int | np.integer | np.floating) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return ""
    return f"{numeric_value:.12g}"


def feature_scaler_rows(
    study_row: dict[str, str],
    feature_rows: list[dict[str, str]],
    semantics_uid: str,
    normalization_name: str,
    scaler_payload: dict[str, np.ndarray],
    config: dict,
) -> list[dict[str, object]]:
    definition = measurement_semantics_definition(config, str(study_row["selected_matrix_semantics"]))
    policy = config["measurement_semantics_registry"]["scaler_artifact_policy"]
    transform_before_zscore = "log1p" if normalization_name == "log1p_then_zscore" else "identity"
    keyed_rows = add_feature_keys(feature_rows, config)
    rows: list[dict[str, object]] = []
    for feature_index, feature_row in enumerate(keyed_rows):
        rows.append(
            {
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "selected_matrix_uid": study_row["selected_matrix_uid"],
                "semantics_uid": semantics_uid,
                "feature_key": feature_row["feature_key"],
                "feature_id": feature_row["feature_id"],
                "matrix_semantics": study_row["selected_matrix_semantics"],
                "normalization_method": normalization_name,
                "scaler_fit_policy": str(policy["production_fit_policy"]),
                "fit_sample_count": scalar_text(scaler_payload["fit_sample_count"][feature_index]),
                "finite_count": scalar_text(scaler_payload["finite_count"][feature_index]),
                "mean": scalar_text(scaler_payload["mean"][feature_index]),
                "std": scalar_text(scaler_payload["std"][feature_index]),
                "zero_variance_flag": "yes" if bool(scaler_payload["zero_variance_flag"][feature_index]) else "no",
                "raw_unit": str(definition["raw_unit"]),
                "canonical_unit": str(definition["canonical_unit"]),
                "proxy_unit_flag": yes_no_text(definition["proxy_unit_flag"]),
                "transform_before_zscore": transform_before_zscore,
                "inverse_transform": str(definition["inverse_transform"]),
                "valid_min": str(definition["valid_min"]),
                "valid_max": str(definition["valid_max"]),
                "leakage_guard": str(policy["leakage_guard"]),
            }
        )
    return rows


def selected_sample_identifiers(sample_rows: list[dict[str, str]]) -> list[str]:
    return [row["matrix_sample_id"] for row in ordered_sample_rows(sample_rows)]


def parse_geo_series_matrix(
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> np.ndarray:
    selected_sample_ids = selected_sample_identifiers(sample_rows)
    ordered_features = ordered_feature_rows(feature_rows)
    matrix = np.full((len(selected_sample_ids), len(ordered_features)), np.nan, dtype=np.float64)
    local_path = resolve_repo_path(loader_row["local_path"])

    in_table = False
    feature_index = 0
    selected_column_positions: list[int] = []
    with open_text(local_path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue
            values = parse_tsv_line(line)
            if not selected_column_positions:
                header_sample_ids = [value.strip() for value in values[1:]]
                selected_column_positions = []
                for sample_id in selected_sample_ids:
                    if sample_id not in header_sample_ids:
                        raise ValueError(f"Missing GEO sample id {sample_id} in {local_path}")
                    selected_column_positions.append(header_sample_ids.index(sample_id) + 1)
                continue
            if feature_index >= len(ordered_features):
                raise ValueError(f"Unexpected extra GEO feature rows in {local_path}")
            expected_feature_id = ordered_features[feature_index]["feature_id"]
            source_feature_id = values[0].strip()
            if source_feature_id != expected_feature_id:
                raise ValueError(
                    f"GEO feature order mismatch in {local_path}: expected {expected_feature_id}, found {source_feature_id}"
                )
            if len(values) <= max(selected_column_positions):
                raise ValueError(f"GEO row is shorter than expected in {local_path}")
            for sample_index, column_position in enumerate(selected_column_positions):
                matrix[sample_index, feature_index] = parse_numeric_value(values[column_position])
            feature_index += 1
    if feature_index != len(ordered_features):
        raise ValueError(f"GEO feature count mismatch in {local_path}: parsed {feature_index}, expected {len(ordered_features)}")
    return matrix


def parse_arrayexpress_combined_zip(
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> np.ndarray:
    selected_sample_ids = selected_sample_identifiers(sample_rows)
    ordered_features = ordered_feature_rows(feature_rows)
    matrix = np.full((len(selected_sample_ids), len(ordered_features)), np.nan, dtype=np.float64)
    local_path = resolve_repo_path(loader_row["local_path"])

    with zipfile.ZipFile(local_path) as archive:
        with archive.open(loader_row["member_path"]) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text_handle, delimiter="\t")
            sample_header = next(reader)
            measurement_header = next(reader)
            selected_column_positions: list[int] = []
            for sample_id in selected_sample_ids:
                matching_positions = [
                    index
                    for index in range(1, len(sample_header))
                    if sample_header[index].strip() == sample_id
                    and measurement_header[index].strip() == loader_row["selected_measurement_label"]
                ]
                if len(matching_positions) != 1:
                    raise ValueError(
                        f"ArrayExpress combined matrix requires exactly one column for {sample_id} in {local_path}"
                    )
                selected_column_positions.append(matching_positions[0])

            feature_index = 0
            for values in reader:
                if feature_index >= len(ordered_features):
                    raise ValueError(f"Unexpected extra ArrayExpress feature rows in {local_path}")
                expected_feature_id = ordered_features[feature_index]["feature_id"]
                source_feature_id = values[0].strip()
                if source_feature_id != expected_feature_id:
                    raise ValueError(
                        f"ArrayExpress feature order mismatch in {local_path}: expected {expected_feature_id}, found {source_feature_id}"
                    )
                if len(values) <= max(selected_column_positions):
                    raise ValueError(f"ArrayExpress combined row is shorter than expected in {local_path}")
                for sample_index, column_position in enumerate(selected_column_positions):
                    matrix[sample_index, feature_index] = parse_numeric_value(values[column_position])
                feature_index += 1
    if feature_index != len(ordered_features):
        raise ValueError(
            f"ArrayExpress combined feature count mismatch in {local_path}: parsed {feature_index}, expected {len(ordered_features)}"
        )
    return matrix


def unique_feature_index(feature_rows: list[dict[str, str]]) -> dict[str, int]:
    feature_lookup: dict[str, int] = {}
    for feature_position, row in enumerate(ordered_feature_rows(feature_rows)):
        if row["feature_id"] in feature_lookup:
            raise ValueError(f"Feature id must be unique for sparse member loaders: {row['feature_id']}")
        feature_lookup[row["feature_id"]] = feature_position
    return feature_lookup


def arrayexpress_member_lookup(member_rows: list[dict[str, str]], selected_matrix_uid: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    selected_rows = [row for row in member_rows if row["matrix_uid"] == selected_matrix_uid]
    for row in selected_rows:
        if not row["sample_uid"]:
            continue
        if row["sample_uid"] in lookup:
            raise ValueError(f"Duplicate ArrayExpress member mapping for sample_uid {row['sample_uid']}")
        lookup[row["sample_uid"]] = row["member_path"]
    return lookup


def parse_arrayexpress_sample_table_zip(
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
    member_rows: list[dict[str, str]],
) -> np.ndarray:
    ordered_samples = ordered_sample_rows(sample_rows)
    feature_lookup = unique_feature_index(feature_rows)
    matrix = np.full((len(ordered_samples), len(feature_lookup)), np.nan, dtype=np.float64)
    member_lookup = arrayexpress_member_lookup(member_rows, loader_row["selected_matrix_uid"])
    local_path = resolve_repo_path(loader_row["local_path"])

    with zipfile.ZipFile(local_path) as archive:
        for sample_index, sample_row in enumerate(ordered_samples):
            if sample_row["sample_uid"] not in member_lookup:
                raise ValueError(f"Missing ArrayExpress sample_table member for sample_uid {sample_row['sample_uid']}")
            member_path = member_lookup[sample_row["sample_uid"]]
            with archive.open(member_path) as raw_handle:
                text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
                reader = csv.reader(text_handle, delimiter="\t")
                header = next(reader)
                if loader_row["feature_identifier_field"] not in header:
                    raise ValueError(f"Missing feature identifier column in {member_path}")
                if loader_row["value_identifier"] not in header:
                    raise ValueError(f"Missing selected value column in {member_path}")
                feature_column_index = header.index(loader_row["feature_identifier_field"])
                value_column_index = header.index(loader_row["value_identifier"])
                for values in reader:
                    source_feature_id = values[feature_column_index].strip()
                    if source_feature_id not in feature_lookup:
                        continue
                    matrix[sample_index, feature_lookup[source_feature_id]] = parse_numeric_value(values[value_column_index])
    return matrix


def parse_arrayexpress_text_matrix(
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> np.ndarray:
    selected_sample_ids = selected_sample_identifiers(sample_rows)
    ordered_features = ordered_feature_rows(feature_rows)
    matrix = np.full((len(selected_sample_ids), len(ordered_features)), np.nan, dtype=np.float64)
    local_path = resolve_repo_path(loader_row["local_path"])
    with open_text(local_path) as handle:
        header_line = handle.readline().rstrip("\n")
        if not header_line:
            raise ValueError(f"ArrayExpress text matrix is empty in {local_path}")
        delimiter = detect_delimiter(header_line)
        reader = csv.reader(io.StringIO(header_line + "\n" + handle.read()), delimiter=delimiter)
        header = next(reader, [])
        header_sample_ids = [value.strip() for value in header[1:]]
        selected_column_positions: list[int] = []
        for sample_id in selected_sample_ids:
            if sample_id not in header_sample_ids:
                raise ValueError(f"Missing ArrayExpress sample id {sample_id} in {local_path}")
            selected_column_positions.append(header_sample_ids.index(sample_id) + 1)
        feature_index = 0
        for values in reader:
            if not values or not values[0].strip():
                continue
            if feature_index >= len(ordered_features):
                raise ValueError(f"Unexpected extra ArrayExpress feature rows in {local_path}")
            expected_feature_id = ordered_features[feature_index]["feature_id"]
            source_feature_id = values[0].strip()
            if source_feature_id != expected_feature_id:
                raise ValueError(
                    f"ArrayExpress feature order mismatch in {local_path}: expected {expected_feature_id}, found {source_feature_id}"
                )
            if len(values) <= max(selected_column_positions):
                raise ValueError(f"ArrayExpress text row is shorter than expected in {local_path}")
            for sample_index, column_position in enumerate(selected_column_positions):
                matrix[sample_index, feature_index] = parse_numeric_value(values[column_position])
            feature_index += 1
    if feature_index != len(ordered_features):
        raise ValueError(
            f"ArrayExpress feature count mismatch in {local_path}: parsed {feature_index}, expected {len(ordered_features)}"
        )
    return matrix


def parse_metabolomics_data_json(
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> np.ndarray:
    selected_sample_ids = selected_sample_identifiers(sample_rows)
    ordered_features = ordered_feature_rows(feature_rows)
    matrix = np.full((len(selected_sample_ids), len(ordered_features)), np.nan, dtype=np.float64)
    payload = json.loads(resolve_repo_path(loader_row["local_path"]).read_text(encoding="utf-8"))
    data_rows = sorted_json_values(payload)

    feature_index = 0
    for data_row in data_rows:
        if feature_index >= len(ordered_features):
            raise ValueError(f"Unexpected extra Metabolomics Workbench feature rows in {loader_row['local_path']}")
        expected_feature_id = ordered_features[feature_index]["feature_id"]
        source_feature_id = str(data_row.get("metabolite_id", "")).strip() or str(data_row.get("metabolite_name", "")).strip()
        if source_feature_id != expected_feature_id:
            raise ValueError(
                f"Metabolomics feature order mismatch in {loader_row['local_path']}: expected {expected_feature_id}, found {source_feature_id}"
            )
        data_values = data_row.get("DATA", {})
        if not isinstance(data_values, dict):
            raise ValueError(f"Metabolomics DATA payload is not a dictionary in {loader_row['local_path']}")
        for sample_index, sample_id in enumerate(selected_sample_ids):
            matrix[sample_index, feature_index] = parse_numeric_value(data_values.get(sample_id))
        feature_index += 1
    if feature_index != len(ordered_features):
        raise ValueError(
            f"Metabolomics feature count mismatch in {loader_row['local_path']}: parsed {feature_index}, expected {len(ordered_features)}"
        )
    return matrix


def matrix_reference_name(loader_row: dict[str, str]) -> str:
    if "#" not in loader_row["matrix_reference"]:
        raise ValueError(f"Matrix reference lacks name fragment: {loader_row['matrix_reference']}")
    return loader_row["matrix_reference"].split("#", 1)[1]


def mwtab_section_index(loader_row: dict[str, str]) -> int:
    match = SECTION_PATTERN.search(matrix_reference_name(loader_row))
    if match is None:
        raise ValueError(f"Unable to determine mwtab section index from {loader_row['matrix_reference']}")
    return int(match.group(1))


def parse_metabolomics_mwtab_section(
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> np.ndarray:
    selected_sample_ids = selected_sample_identifiers(sample_rows)
    ordered_features = ordered_feature_rows(feature_rows)
    matrix = np.full((len(selected_sample_ids), len(ordered_features)), np.nan, dtype=np.float64)
    local_path = resolve_repo_path(loader_row["local_path"])
    target_section_index = mwtab_section_index(loader_row)
    current_section_index = 0
    current_section_name = ""
    in_target_section = False
    sample_header_seen = False
    feature_index = 0
    selected_column_positions: list[int] = []

    with local_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line in {"#MS_METABOLITE_DATA", "#NMR_METABOLITE_DATA"}:
                current_section_name = line.removeprefix("#")
                continue
            if current_section_name and line == f"{current_section_name}_START":
                current_section_index += 1
                in_target_section = current_section_index == target_section_index
                sample_header_seen = False
                selected_column_positions = []
                continue
            if current_section_name and line == f"{current_section_name}_END":
                if in_target_section:
                    break
                current_section_name = ""
                in_target_section = False
                continue
            if not in_target_section:
                continue
            values = line.split("\t")
            if not sample_header_seen:
                if not values or values[0] != "Samples":
                    raise ValueError(f"Expected Samples header in {local_path}")
                header_sample_ids = [value.strip() for value in values[1:]]
                for sample_id in selected_sample_ids:
                    if sample_id not in header_sample_ids:
                        raise ValueError(f"Missing Metabolomics sample id {sample_id} in {local_path}")
                    selected_column_positions.append(header_sample_ids.index(sample_id) + 1)
                sample_header_seen = True
                continue
            if values and values[0] == "Factors":
                continue
            if not values or not values[0].strip():
                continue
            if feature_index >= len(ordered_features):
                raise ValueError(f"Unexpected extra Metabolomics mwtab feature rows in {local_path}")
            expected_feature_id = ordered_features[feature_index]["feature_id"]
            source_feature_id = values[0].strip()
            if source_feature_id != expected_feature_id:
                raise ValueError(
                    f"Metabolomics mwtab feature order mismatch in {local_path}: expected {expected_feature_id}, found {source_feature_id}"
                )
            if len(values) <= max(selected_column_positions):
                raise ValueError(f"Metabolomics mwtab row is shorter than expected in {local_path}")
            for sample_index, column_position in enumerate(selected_column_positions):
                matrix[sample_index, feature_index] = parse_numeric_value(values[column_position])
            feature_index += 1

    if feature_index != len(ordered_features):
        raise ValueError(
            f"Metabolomics mwtab feature count mismatch in {local_path}: parsed {feature_index}, expected {len(ordered_features)}"
        )
    return matrix


def build_raw_matrix(
    study_row: dict[str, str],
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
    member_rows: list[dict[str, str]],
) -> np.ndarray:
    if loader_row["loader_kind"] == "geo_series_matrix_gz":
        return parse_geo_series_matrix(loader_row, sample_rows, feature_rows)
    if loader_row["loader_kind"] == "arrayexpress_combined_zip_tsv":
        return parse_arrayexpress_combined_zip(loader_row, sample_rows, feature_rows)
    if loader_row["loader_kind"] == "arrayexpress_sample_table_zip":
        return parse_arrayexpress_sample_table_zip(loader_row, sample_rows, feature_rows, member_rows)
    if loader_row["loader_kind"] == "arrayexpress_text_matrix_tsv":
        return parse_arrayexpress_text_matrix(loader_row, sample_rows, feature_rows)
    if loader_row["loader_kind"] == "metabolomics_data_json":
        return parse_metabolomics_data_json(loader_row, sample_rows, feature_rows)
    if loader_row["loader_kind"] == "metabolomics_mwtab_section":
        return parse_metabolomics_mwtab_section(loader_row, sample_rows, feature_rows)
    raise ValueError(f"Unsupported model-input loader_kind: {loader_row['loader_kind']} for study {study_row['study_uid']}")


def manifest_payload(
    study_row: dict[str, str],
    loader_row: dict[str, str],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
    bundle_path_map: dict[str, Path],
    semantics_uid: str,
    semantics_label: str,
    normalization_name: str,
    atlas_prior_ready_flag: str,
    scaler_fit_policy: str,
) -> dict[str, object]:
    return {
        "study_uid": study_row["study_uid"],
        "source_key": study_row["source_key"],
        "source_accession": study_row["source_accession"],
        "world_model_role": "disease_observation",
        "subject_linkable_flag": "yes",
        "atlas_prior_ready_flag": atlas_prior_ready_flag,
        "public_only_flag": "yes",
        "selected_matrix_uid": study_row["selected_matrix_uid"],
        "loader_kind": loader_row["loader_kind"],
        "selected_measurement_label": study_row["selected_measurement_label"],
        "semantics_uid": semantics_uid,
        "measurement_semantics_label": semantics_label,
        "matrix_semantics": study_row["selected_matrix_semantics"],
        "normalization_method": normalization_name,
        "scaler_fit_policy": scaler_fit_policy,
        "sample_count": len(sample_rows),
        "feature_count": len(feature_rows),
        "source_matrix_reference": study_row["selected_matrix_reference"],
        "bundle_root": relative_repo_path(bundle_path_map["bundle_root"]),
        "sample_index_path": relative_repo_path(bundle_path_map["sample_index_path"]),
        "feature_index_path": relative_repo_path(bundle_path_map["feature_index_path"]),
        "matrix_raw_path": relative_repo_path(bundle_path_map["matrix_raw_path"]),
        "matrix_norm_path": relative_repo_path(bundle_path_map["matrix_norm_path"]),
        "scaler_artifact_path": relative_repo_path(bundle_path_map["scaler_artifact_path"]),
    }


def build_bundle_row(
    study_row: dict[str, str],
    loader_row: dict[str, str],
    bundle_path_map: dict[str, Path],
    sample_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
    semantics_uid: str,
    semantics_label: str,
    normalization_name: str,
    atlas_prior_ready_flag: str,
    scaler_fit_policy: str,
) -> dict[str, object]:
    return {
        "study_uid": study_row["study_uid"],
        "source_key": study_row["source_key"],
        "source_accession": study_row["source_accession"],
        "world_model_role": "disease_observation",
        "subject_linkable_flag": "yes",
        "atlas_prior_ready_flag": atlas_prior_ready_flag,
        "public_only_flag": "yes",
        "selected_matrix_uid": study_row["selected_matrix_uid"],
        "loader_kind": loader_row["loader_kind"],
        "selected_measurement_label": study_row["selected_measurement_label"],
        "semantics_uid": semantics_uid,
        "measurement_semantics_label": semantics_label,
        "matrix_semantics": study_row["selected_matrix_semantics"],
        "normalization_method": normalization_name,
        "scaler_fit_policy": scaler_fit_policy,
        "sample_count": len(sample_rows),
        "feature_count": len(feature_rows),
        "bundle_root": relative_repo_path(bundle_path_map["bundle_root"]),
        "bundle_manifest_path": relative_repo_path(bundle_path_map["bundle_manifest_path"]),
        "sample_index_path": relative_repo_path(bundle_path_map["sample_index_path"]),
        "feature_index_path": relative_repo_path(bundle_path_map["feature_index_path"]),
        "matrix_raw_path": relative_repo_path(bundle_path_map["matrix_raw_path"]),
        "matrix_norm_path": relative_repo_path(bundle_path_map["matrix_norm_path"]),
        "scaler_artifact_path": relative_repo_path(bundle_path_map["scaler_artifact_path"]),
        "source_matrix_reference": study_row["selected_matrix_reference"],
    }


def excluded_bundle_row(exclusion_row: dict[str, str]) -> dict[str, object]:
    return {
        "study_uid": exclusion_row["study_uid"],
        "source_key": exclusion_row["source_key"],
        "source_accession": exclusion_row["source_accession"],
        "analysis_ready_flag": "no",
        "exclusion_reason": exclusion_row["exclusion_reason"],
        "bundle_materialization_flag": "no",
        "note": exclusion_row["note"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accessions(args.source_accessions)

    (
        analysis_study_rows,
        analysis_loader_rows,
        analysis_sample_rows,
        analysis_feature_rows,
        analysis_exclusion_rows,
        feature_member_rows,
    ) = load_registry_inputs(config, source_accessions)
    pipeline_logger.progress(
        "loaded_model_input_inputs",
        analysis_study_row_count=len(analysis_study_rows),
        analysis_loader_row_count=len(analysis_loader_rows),
        analysis_sample_row_count=len(analysis_sample_rows),
        analysis_feature_row_count=len(analysis_feature_rows),
        analysis_exclusion_row_count=len(analysis_exclusion_rows),
        feature_member_row_count=len(feature_member_rows),
    )

    ready_study_rows = [row for row in analysis_study_rows if row["analysis_ready_flag"] == "yes"]
    loader_by_matrix = {row["selected_matrix_uid"]: row for row in analysis_loader_rows}
    sample_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    feature_rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample_row in analysis_sample_rows:
        sample_rows_by_study[sample_row["study_uid"]].append(sample_row)
    for feature_row in analysis_feature_rows:
        feature_rows_by_study[feature_row["study_uid"]].append(feature_row)

    bundle_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    atlas_prior_root = resolve_repo_path(config["atlas_prior_registry"]["local_root"])
    atlas_prior_ready_flag = (
        "yes"
        if (atlas_prior_root / config["atlas_prior_registry"]["relative_paths"]["program_catalog"]).exists()
        else "no"
    )

    for study_row in ready_study_rows:
        sample_rows = ordered_sample_rows(sample_rows_by_study.get(study_row["study_uid"], []))
        feature_rows = ordered_feature_rows(feature_rows_by_study.get(study_row["study_uid"], []))
        if not sample_rows:
            raise ValueError(f"Missing analysis samples for study {study_row['study_uid']}")
        if not feature_rows:
            raise ValueError(f"Missing analysis features for study {study_row['study_uid']}")
        loader_row = loader_by_matrix[study_row["selected_matrix_uid"]]
        raw_matrix = build_raw_matrix(study_row, loader_row, sample_rows, feature_rows, feature_member_rows)
        if raw_matrix.shape != (len(sample_rows), len(feature_rows)):
            raise ValueError(
                f"Raw matrix shape mismatch for study {study_row['study_uid']}: {raw_matrix.shape}"
            )
        norm_matrix, normalization_name, scaler_payload = fit_feature_scaler(raw_matrix, study_row["selected_matrix_semantics"], config)
        bundle_path_map = bundle_paths(study_row, config)
        prune_bundle_root(bundle_path_map)
        semantics_uid = measurement_semantics_uid(
            {
                **study_row,
                "matrix_semantics": study_row["selected_matrix_semantics"],
            }
        )
        semantics_label = measurement_semantics_label(
            {
                **study_row,
                "matrix_semantics": study_row["selected_matrix_semantics"],
            }
        )
        scaler_fit_policy = str(config["measurement_semantics_registry"]["scaler_artifact_policy"]["production_fit_policy"])

        sample_index = sample_index_frame(sample_rows, config)
        feature_index = feature_index_frame(feature_rows, config)
        write_parquet_atomic(sample_index, bundle_path_map["sample_index_path"])
        write_parquet_atomic(feature_index, bundle_path_map["feature_index_path"])
        write_npy_atomic(raw_matrix, bundle_path_map["matrix_raw_path"])
        write_npy_atomic(norm_matrix, bundle_path_map["matrix_norm_path"])
        write_tsv(
            bundle_path_map["scaler_artifact_path"],
            config["schemas"]["model_input_feature_scaler_fields"],
            feature_scaler_rows(
                study_row,
                feature_rows,
                semantics_uid,
                normalization_name,
                scaler_payload,
                config,
            ),
        )
        write_json(
            bundle_path_map["bundle_manifest_path"],
            manifest_payload(
                study_row,
                loader_row,
                sample_rows,
                feature_rows,
                bundle_path_map,
                semantics_uid,
                semantics_label,
                normalization_name,
                atlas_prior_ready_flag,
                scaler_fit_policy,
            ),
        )
        bundle_rows.append(
            build_bundle_row(
                study_row,
                loader_row,
                bundle_path_map,
                sample_rows,
                feature_rows,
                semantics_uid,
                semantics_label,
                normalization_name,
                atlas_prior_ready_flag,
                scaler_fit_policy,
            )
        )

    for exclusion_row in analysis_exclusion_rows:
        excluded_rows.append(excluded_bundle_row(exclusion_row))

    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    ensure_directory(model_input_root)
    relative_paths = config["model_input_registry"]["relative_paths"]
    bundle_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    excluded_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    write_tsv(
        model_input_root / relative_paths["study_bundle_catalog"],
        config["schemas"]["model_input_bundle_fields"],
        bundle_rows,
    )
    write_tsv(
        model_input_root / relative_paths["excluded_study_catalog"],
        config["schemas"]["model_input_exclusion_fields"],
        excluded_rows,
    )
    pipeline_logger.artifact(
        "model_input_registry_written",
        artifact=model_input_root,
        bundle_row_count=len(bundle_rows),
        excluded_row_count=len(excluded_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=model_input_root,
        bundle_row_count=len(bundle_rows),
        excluded_row_count=len(excluded_rows),
    )

if __name__ == "__main__":
    main()
