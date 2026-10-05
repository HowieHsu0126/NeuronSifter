from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from Libs.utils.evaluation.alignment_benchmark import load_alignment_bundle_artifacts, load_alignment_catalog
from Libs.utils.data.measurement_scaling import fit_feature_scaler
from Libs.utils.data.raw_data_registry import read_tsv, resolve_repo_path


def active_disease_program(config: dict) -> str:
    return str(config["benchmark_registry"].get("active_disease_program", "ad_control"))


def benchmark_program_config(config: dict, disease_program: str | None = None) -> dict:
    program_name = disease_program or active_disease_program(config)
    programs = config["benchmark_registry"].get("disease_programs", {})
    if program_name in programs:
        merged = dict(config["benchmark_registry"])
        merged.update(programs[program_name])
        merged["disease_program"] = program_name
        merged["active_disease_program"] = program_name
        return merged
    merged = dict(config["benchmark_registry"])
    merged["disease_program"] = program_name
    merged["active_disease_program"] = program_name
    return merged


def benchmark_name_for_disease_program(config: dict, disease_program: str | None = None) -> str:
    program_config = benchmark_program_config(config, disease_program)
    return str(program_config["benchmark_name"])


def benchmark_root(config: dict) -> Path:
    return resolve_repo_path(config["benchmark_registry"]["local_root"])


def filter_rows_by_disease_program(rows: list[dict[str, str]], disease_program: str | None) -> list[dict[str, str]]:
    if not rows:
        return rows
    if "disease_program" not in rows[0]:
        return rows
    if disease_program is None:
        return rows
    return [row for row in rows if row["disease_program"] == disease_program]


def load_benchmark_catalogs(config: dict, disease_program: str | None = None) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    root = benchmark_root(config)
    relative_paths = config["benchmark_registry"]["relative_paths"]
    cohort_rows = read_tsv(root / relative_paths["cohort_catalog"])
    fold_rows = read_tsv(root / relative_paths["fold_catalog"])
    dataset_rows = read_tsv(root / relative_paths["training_dataset_catalog"])
    active_program = disease_program or active_disease_program(config)
    cohort_rows = filter_rows_by_disease_program(cohort_rows, active_program)
    fold_rows = filter_rows_by_disease_program(fold_rows, active_program)
    dataset_rows = filter_rows_by_disease_program(dataset_rows, active_program)
    return cohort_rows, fold_rows, dataset_rows


def preferred_benchmark_name(config: dict, disease_program: str | None = None) -> str:
    program_name = disease_program or active_disease_program(config)
    program_config = benchmark_program_config(config, program_name)
    cross_name = str(program_config.get("cross_study_benchmark_name", program_config["benchmark_name"]))
    single_name = str(program_config.get("single_cohort_benchmark_name", ""))
    _, fold_rows, _ = load_benchmark_catalogs(config, program_name)
    fold_names = {row["benchmark_name"] for row in fold_rows if row.get("benchmark_name", "")}
    if cross_name in fold_names:
        return cross_name
    if single_name and single_name in fold_names:
        return single_name
    return str(program_config["benchmark_name"])


def alignment_by_study(config: dict) -> dict[str, dict[str, str]]:
    return {row["study_uid"]: row for row in load_alignment_catalog(config)}


def fold_root(
    config: dict,
    fold_name: str,
    disease_program: str | None = None,
    benchmark_name_override: str | None = None,
) -> Path:
    return resolve_repo_path(
        config["benchmark_registry"]["fold_root_template"].format(
            benchmark_name=benchmark_name_override or preferred_benchmark_name(config, disease_program),
            fold_name=fold_name,
        )
    )


def fold_sample_split(
    config: dict,
    fold_name: str,
    disease_program: str | None = None,
    benchmark_name_override: str | None = None,
) -> pd.DataFrame:
    root = fold_root(config, fold_name, disease_program, benchmark_name_override)
    path = root / config["benchmark_registry"]["fold_filenames"]["sample_split"]
    return pd.read_parquet(path)


def fold_entity_index(
    config: dict,
    fold_name: str,
    modality: str,
    disease_program: str | None = None,
    benchmark_name_override: str | None = None,
) -> pd.DataFrame:
    root = fold_root(config, fold_name, disease_program, benchmark_name_override)
    if modality == "transcriptomics":
        path = root / config["benchmark_registry"]["fold_filenames"]["transcriptomics_entity_index"]
    elif modality == "metabolomics":
        path = root / config["benchmark_registry"]["fold_filenames"]["metabolomics_entity_index"]
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return pd.read_parquet(path)


def build_entity_index_from_studies(
    config: dict,
    study_uids: list[str],
    modality: str,
    minimum_source_count: int,
) -> pd.DataFrame:
    alignment_lookup = alignment_by_study(config)
    modality_study_uids = [
        study_uid
        for study_uid in study_uids
        if alignment_lookup[study_uid]["modality"] == modality
    ]
    source_count_by_entity: dict[str, int] = defaultdict(int)
    entity_label_lookup: dict[str, str] = {}
    namespace_lookup: dict[str, str] = {}
    entity_type_lookup: dict[str, str] = {}

    for study_uid in study_uids:
        alignment_row = alignment_lookup[study_uid]
        if alignment_row["modality"] != modality:
            continue
        _, entity_index, _, _ = load_alignment_bundle_artifacts(alignment_row)
        for entity_row in entity_index.itertuples(index=False):
            canonical_entity_id = str(entity_row.canonical_entity_id)
            source_count_by_entity[canonical_entity_id] += 1
            entity_label_lookup.setdefault(canonical_entity_id, str(entity_row.canonical_entity_label))
            namespace_lookup.setdefault(canonical_entity_id, str(entity_row.namespace))
            entity_type_lookup.setdefault(canonical_entity_id, str(entity_row.entity_type))

    effective_minimum_source_count = min(int(minimum_source_count), len(modality_study_uids))
    if effective_minimum_source_count <= 0:
        effective_minimum_source_count = 1

    retained_entity_ids = [
        entity_id
        for entity_id, source_count in sorted(source_count_by_entity.items())
        if source_count >= effective_minimum_source_count
    ]
    return pd.DataFrame(
        [
            {
                "row_index": index + 1,
                "canonical_entity_id": entity_id,
                "canonical_entity_label": entity_label_lookup[entity_id],
                "entity_type": entity_type_lookup[entity_id],
                "namespace": namespace_lookup[entity_id],
                "source_count": source_count_by_entity[entity_id],
            }
            for index, entity_id in enumerate(retained_entity_ids)
        ],
        columns=["row_index", "canonical_entity_id", "canonical_entity_label", "entity_type", "namespace", "source_count"],
    )


def assemble_dataset(
    config: dict,
    sample_frame: pd.DataFrame,
    modality: str,
    entity_index: pd.DataFrame,
    matrix_kind: str = "normalized",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if matrix_kind not in {"normalized", "raw"}:
        raise ValueError(f"Unsupported matrix_kind: {matrix_kind}")
    alignment_lookup = alignment_by_study(config)
    modality_frame = sample_frame.loc[sample_frame["modality"] == modality].copy()
    if modality_frame.empty:
        return np.empty((0, int(entity_index.shape[0])), dtype=np.float32), np.empty((0,), dtype=np.int64), modality_frame

    selected_entity_ids = entity_index["canonical_entity_id"].astype(str).tolist()
    selected_entity_lookup = {entity_id: index for index, entity_id in enumerate(selected_entity_ids)}
    matrix = np.zeros((modality_frame.shape[0], len(selected_entity_ids)), dtype=np.float32)
    labels = modality_frame["binary_label"].astype(int).to_numpy(dtype=np.int64)

    row_position = 0
    for study_uid in modality_frame["study_uid"].drop_duplicates().tolist():
        alignment_row = alignment_lookup[study_uid]
        sample_index, study_entity_index, raw_matrix, norm_matrix = load_alignment_bundle_artifacts(alignment_row)
        source_matrix = raw_matrix if matrix_kind == "raw" else norm_matrix
        sample_lookup = {str(sample_uid): index for index, sample_uid in enumerate(sample_index["sample_uid"].astype(str).tolist())}
        entity_lookup = {str(entity_id): index for index, entity_id in enumerate(study_entity_index["canonical_entity_id"].astype(str).tolist())}
        study_rows = modality_frame.loc[modality_frame["study_uid"] == study_uid].copy()
        for study_row in study_rows.itertuples(index=False):
            sample_matrix_position = sample_lookup[str(study_row.sample_uid)]
            for entity_id, column_index in selected_entity_lookup.items():
                if entity_id not in entity_lookup:
                    continue
                value = float(source_matrix[sample_matrix_position, entity_lookup[entity_id]])
                if np.isfinite(value):
                    matrix[row_position, column_index] = value
            row_position += 1
    return matrix, labels, modality_frame.reset_index(drop=True)


def matrix_semantics_for_modality(config: dict, modality: str) -> str:
    semantics_by_modality = config["measurement_semantics_registry"]["modality_default_matrix_semantics"]
    if modality not in semantics_by_modality:
        raise ValueError(f"Missing matrix semantics mapping for modality={modality!r}")
    return str(semantics_by_modality[modality])


def train_context_standardized_split_matrices(
    config: dict,
    modality: str,
    split_matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    train_matrix = split_matrices["train"]
    nontrain_row_count = sum(matrix.shape[0] for split_name, matrix in split_matrices.items() if split_name != "train")
    if train_matrix.shape[0] == 0:
        if nontrain_row_count > 0:
            raise ValueError(f"Train-context scaler cannot be fit without train rows for modality={modality}")
        return {split_name: matrix.astype(np.float32, copy=True) for split_name, matrix in split_matrices.items()}
    split_order = list(split_matrices)
    feature_counts = {matrix.shape[1] for matrix in split_matrices.values()}
    if len(feature_counts) != 1:
        raise ValueError(f"Split matrices have inconsistent feature counts for modality={modality}")
    combined_matrix = np.concatenate([split_matrices[split_name] for split_name in split_order], axis=0)
    fit_sample_mask = np.zeros((combined_matrix.shape[0],), dtype=bool)
    fit_sample_mask[: train_matrix.shape[0]] = True
    standardized_matrix, _, _ = fit_feature_scaler(
        combined_matrix.astype(np.float64, copy=False),
        matrix_semantics_for_modality(config, modality),
        config,
        fit_sample_mask,
    )
    standardized_splits: dict[str, np.ndarray] = {}
    row_start = 0
    for split_name in split_order:
        row_stop = row_start + split_matrices[split_name].shape[0]
        standardized_splits[split_name] = standardized_matrix[row_start:row_stop].astype(np.float32)
        row_start = row_stop
    return standardized_splits
