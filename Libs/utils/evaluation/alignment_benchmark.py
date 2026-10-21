from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from Libs.utils.data.raw_data_registry import ensure_directory, normalize_text, read_tsv, resolve_repo_path


REPO_ROOT = Path(__file__).resolve().parents[3]
MISSING_LABEL_TOKENS = {"", "na", "nan", "null", "none"}


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_input_catalog(config: dict) -> list[dict[str, str]]:
    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    catalog_path = model_input_root / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"]
    return read_tsv(catalog_path)


def load_alignment_catalog(config: dict) -> list[dict[str, str]]:
    alignment_root = resolve_repo_path(config["alignment_ready_registry"]["local_root"])
    catalog_path = alignment_root / config["alignment_ready_registry"]["relative_paths"]["aligned_matrix_catalog"]
    return read_tsv(catalog_path)


def load_bundle_artifacts(bundle_row: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sample_index = pd.read_parquet(resolve_repo_path(bundle_row["sample_index_path"]))
    feature_index = pd.read_parquet(resolve_repo_path(bundle_row["feature_index_path"]))
    raw_matrix = np.load(resolve_repo_path(bundle_row["matrix_raw_path"]), allow_pickle=False)
    norm_matrix = np.load(resolve_repo_path(bundle_row["matrix_norm_path"]), allow_pickle=False)
    return sample_index, feature_index, raw_matrix, norm_matrix


def load_alignment_bundle_artifacts(alignment_row: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sample_index = pd.read_parquet(resolve_repo_path(alignment_row["sample_index_path"]))
    entity_index = pd.read_parquet(resolve_repo_path(alignment_row["entity_index_path"]))
    raw_matrix = np.load(resolve_repo_path(alignment_row["matrix_raw_path"]), allow_pickle=False)
    norm_matrix = np.load(resolve_repo_path(alignment_row["matrix_norm_path"]), allow_pickle=False)
    return sample_index, entity_index, raw_matrix, norm_matrix


def normalize_diagnosis_label(label: str, alias_map: dict[str, list[str]]) -> str:
    normalized_label = normalize_text(label)
    if normalized_label in MISSING_LABEL_TOKENS:
        return ""
    for canonical_label, aliases in alias_map.items():
        normalized_aliases = {normalize_text(value) for value in aliases}
        normalized_aliases.add(normalize_text(canonical_label))
        if normalized_label in normalized_aliases:
            return canonical_label
    return label.strip()


def binary_label(label: str, positive_labels: set[str], negative_labels: set[str]) -> int | None:
    if label in positive_labels:
        return 1
    if label in negative_labels:
        return 0
    return None


def normalized_metabolite_label(label: str) -> str:
    stripped = label.strip()
    if not stripped:
        return ""
    normalized = normalize_text(stripped)
    normalized = normalized.replace(" o2 ", " o2 ")
    return normalized


def modality_from_sample_index(sample_index: pd.DataFrame) -> str:
    if sample_index.empty:
        raise ValueError("Sample index is empty.")
    return str(sample_index["modality"].iloc[0])


def tissue_from_sample_index(sample_index: pd.DataFrame) -> str:
    if sample_index.empty:
        raise ValueError("Sample index is empty.")
    return str(sample_index["tissue_or_biofluid"].iloc[0])


def choose_group_identifier(sample_row: pd.Series) -> str:
    subject_id = str(sample_row.get("subject_id", "")).strip()
    if subject_id and normalize_text(subject_id) not in MISSING_LABEL_TOKENS:
        return subject_id
    sample_uid = str(sample_row.get("sample_uid", "")).strip()
    if not sample_uid:
        raise ValueError("Sample row lacks both subject_id and sample_uid.")
    return sample_uid


def join_values(values: list[str]) -> str:
    ordered_values = [value for value in values if value]
    return "|".join(ordered_values)


def split_joined_values(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split("|") if item]


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Binary metrics require 1-D y_true and y_score arrays.")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("Binary metrics require y_true and y_score with identical length.")
    y_pred = (y_score >= 0.5).astype(np.int64)
    eps = np.clip(y_score, 1e-8, 1.0 - 1e-8)
    brier_score = float(np.mean((eps - y_true) ** 2))
    return {
        "auroc": safe_auroc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "brier_score": brier_score,
    }
