from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.scripts.builders.build_adni_controlled_registry import controlled_bundle_pointer_payload
from Libs.utils.data.controlled_research_config import (
    assert_controlled_governance_approval,
    load_controlled_research_config,
)
from Libs.utils.data.raw_data_registry import ensure_directory, repo_root, write_json
from Libs.utils.evaluation.screening_datasets import disease_differential_vector, patient_state_feature_matrix
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled-access ADNI aggregate validation benchmarks.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def adni_section(config: dict) -> dict:
    return config["adni_controlled_access"]


def resolve_configured_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def processed_path(config: dict, artifact_key: str) -> Path:
    paths_config = adni_section(config)["paths"]
    return resolve_configured_path(paths_config["processed_root"]) / paths_config[artifact_key]


def report_path(config: dict, artifact_key: str) -> Path:
    return resolve_configured_path(adni_section(config)["paths"][artifact_key])


def load_bundle(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sample_index = pd.read_parquet(processed_path(config, "sample_index"))
    feature_index = pd.read_parquet(processed_path(config, "feature_index"))
    transition_index = pd.read_parquet(processed_path(config, "transition_label_index"))
    matrix_norm = np.load(processed_path(config, "matrix_norm"), allow_pickle=False)
    missing_mask = np.load(processed_path(config, "matrix_missing_mask"), allow_pickle=False)
    if matrix_norm.shape[0] != sample_index.shape[0] or matrix_norm.shape[1] != feature_index.shape[0]:
        raise ValueError("ADNI controlled model-input matrix shape does not match sample and feature indices.")
    if missing_mask.shape != matrix_norm.shape:
        raise ValueError("ADNI controlled missing-mask matrix shape does not match normalized matrix.")
    return sample_index, feature_index, transition_index, matrix_norm, missing_mask


def finite_label_mask(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.len().to_numpy() > 0


def fit_centroids(matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("Centroid fitting received mismatched feature and label lengths.")
    label_values = np.array(sorted(set(int(value) for value in labels)), dtype=np.int64)
    if label_values.size == 0:
        raise ValueError("Centroid fitting requires at least one labeled train sample.")
    centroids = np.vstack([matrix[labels == label_value].mean(axis=0) for label_value in label_values])
    return label_values, centroids


def predict_centroid_probabilities(matrix: np.ndarray, label_values: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
    predictions = label_values[np.argmin(distances, axis=1)]
    scores = np.exp(-distances)
    probabilities = scores / scores.sum(axis=1, keepdims=True)
    return predictions, probabilities


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    recalls: list[float] = []
    for label_value in sorted(set(int(value) for value in y_true)):
        label_mask = y_true == label_value
        recalls.append(float(np.mean(y_pred[label_mask] == label_value)))
    return float(np.mean(recalls)) if recalls else None


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    scores: list[float] = []
    for label_value in sorted(set(int(value) for value in y_true) | set(int(value) for value in y_pred)):
        true_positive = float(np.sum((y_true == label_value) & (y_pred == label_value)))
        false_positive = float(np.sum((y_true != label_value) & (y_pred == label_value)))
        false_negative = float(np.sum((y_true == label_value) & (y_pred != label_value)))
        denominator = (2.0 * true_positive) + false_positive + false_negative
        scores.append((2.0 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores)) if scores else None


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    clipped = np.clip(probabilities, 1.0e-8, 1.0 - 1.0e-8)
    return float(np.mean((clipped - y_true.astype(np.float64)) ** 2))


def split_indices(sample_index: pd.DataFrame, split_name: str) -> np.ndarray:
    return np.where(sample_index["benchmark_split"].astype(str).to_numpy() == split_name)[0]


def state_labels(sample_index: pd.DataFrame) -> np.ndarray:
    return sample_index["diagnosis_numeric"].astype(str).astype(float).astype(np.int64).to_numpy()


def disease_positive_probabilities(label_values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    positive_columns = np.where(label_values >= 2)[0]
    if positive_columns.size == 0:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return probabilities[:, positive_columns].sum(axis=1)


def format_checkpoint_relative_path(template: str, checkpoint_config: dict) -> Path:
    seed_value = int(checkpoint_config["seed_value"])
    benchmark_name = str(checkpoint_config["benchmark_name_template"]).format(seed=seed_value)
    path_text = str(template).format(
        seed=seed_value,
        benchmark_name=benchmark_name,
        fold_name=str(checkpoint_config.get("expected_fold_name", "")),
        scope=str(checkpoint_config.get("expected_scope", "")),
    )
    return Path(benchmark_name) / path_text


def resolved_screening_checkpoint(config: dict) -> dict[str, object]:
    checkpoint_config = adni_section(config)["application"]["screening_checkpoint"]
    selection_policy = str(checkpoint_config.get("selection_policy", "explicit_path"))
    if selection_policy == "current_public_mainline_production":
        artifact_root = resolve_configured_path(str(checkpoint_config["artifact_root"]))
        checkpoint_path = artifact_root / format_checkpoint_relative_path(str(checkpoint_config["checkpoint_relative_path"]), checkpoint_config)
        manifest_path = artifact_root / format_checkpoint_relative_path(str(checkpoint_config["manifest_relative_path"]), checkpoint_config)
    elif selection_policy == "explicit_path":
        checkpoint_path = resolve_configured_path(str(checkpoint_config["path"]))
        manifest_path = resolve_configured_path(str(checkpoint_config.get("manifest_path", ""))) if checkpoint_config.get("manifest_path") else checkpoint_path.parent / "training_manifest.json"
    else:
        raise ValueError(f"Unsupported ADNI screening checkpoint selection_policy: {selection_policy}")
    if not checkpoint_path.exists():
        if bool(checkpoint_config["required"]):
            raise FileNotFoundError(f"Required ADNI screening checkpoint is missing: {checkpoint_path}")
        raise ValueError("ADNI screening checkpoint is required for controlled world-state application.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Required ADNI screening checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scope = str(checkpoint_config["expected_scope"])
    expected_fold_name = str(checkpoint_config["expected_fold_name"])
    expected_model_name = str(checkpoint_config["expected_model_name"])
    expected_model_contract_version = str(checkpoint_config["expected_model_contract_version"])
    expected_benchmark_name = str(checkpoint_config["benchmark_name_template"]).format(seed=int(checkpoint_config["seed_value"]))
    manifest_checks = {
        "scope": expected_scope,
        "fold_name": expected_fold_name,
        "model_name": expected_model_name,
        "model_contract_version": expected_model_contract_version,
        "benchmark_name": expected_benchmark_name,
    }
    for manifest_key, expected_value in manifest_checks.items():
        observed_value = str(manifest.get(manifest_key, ""))
        if observed_value != expected_value:
            raise ValueError(
                f"ADNI screening checkpoint manifest {manifest_key} mismatch: expected {expected_value!r}, found {observed_value!r}."
            )
    return {
        "selection_policy": selection_policy,
        "profile_name": str(checkpoint_config.get("profile_name", "")),
        "seed_value": int(checkpoint_config["seed_value"]),
        "checkpoint_path": checkpoint_path,
        "manifest_path": manifest_path,
        "benchmark_name": expected_benchmark_name,
        "scope": expected_scope,
        "fold_name": expected_fold_name,
        "model_name": expected_model_name,
        "model_contract_version": expected_model_contract_version,
    }


def load_screening_model(config: dict) -> tuple[ADScreeningSystem, dict[str, object]]:
    checkpoint_config = adni_section(config)["application"]["screening_checkpoint"]
    checkpoint_resolution = resolved_screening_checkpoint(config)
    checkpoint_path = checkpoint_resolution["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_contract_version = str(checkpoint_config["expected_model_contract_version"])
    checkpoint_contract_version = str(checkpoint.get("model_contract_version", ""))
    if checkpoint_contract_version != expected_contract_version:
        raise ValueError(
            f"ADNI screening checkpoint contract mismatch: expected {expected_contract_version!r}, found {checkpoint_contract_version!r}."
        )
    model_kwargs_key = str(checkpoint_config["model_kwargs_key"])
    model_state_key = str(checkpoint_config["model_state_key"])
    if model_kwargs_key not in checkpoint:
        raise KeyError(f"ADNI screening checkpoint is missing model kwargs key: {model_kwargs_key}")
    if model_state_key not in checkpoint:
        raise KeyError(f"ADNI screening checkpoint is missing model state key: {model_state_key}")
    model_kwargs = dict(checkpoint[model_kwargs_key])
    config_model_kwargs = screening_model_kwargs_from_config(config)
    model_kwargs.update(
        {
            key: value
            for key, value in config_model_kwargs.items()
            if key not in model_kwargs or model_kwargs[key] in (None, [], {})
        }
    )
    required_rollout_policy_keys = {
        "state_delta_scale",
        "control_source_scale",
        "region_prior_mix",
        "activation_checkpointing",
        "action_representation",
        "transition_backbone",
    }
    rollout_policy = model_kwargs.get("rollout_update_policy", {})
    if not isinstance(rollout_policy, dict):
        model_kwargs["rollout_update_policy"] = config_model_kwargs["rollout_update_policy"]
    else:
        completed_rollout_policy = dict(rollout_policy)
        for missing_key in sorted(required_rollout_policy_keys - set(completed_rollout_policy)):
            completed_rollout_policy[missing_key] = config_model_kwargs["rollout_update_policy"][missing_key]
        model_kwargs["rollout_update_policy"] = completed_rollout_policy
    model = ADScreeningSystem(**model_kwargs)
    model.load_state_dict(checkpoint[model_state_key], strict=bool(checkpoint_config["state_dict_strict"]))
    model.eval()
    checkpoint_report = dict(checkpoint_resolution)
    checkpoint_report["checkpoint_path"] = str(checkpoint_report["checkpoint_path"])
    checkpoint_report["manifest_path"] = str(checkpoint_report["manifest_path"])
    return model, checkpoint_report


def projection_target_latent_dim(model: ADScreeningSystem, config: dict) -> int:
    adapter_config = adni_section(config)["application"]["adni_tabular_adapter"]
    summary_feature_count = len(adapter_config["projection_policy"]["summary_feature_names"])
    target_dim = int(model.patient_state_dim) - summary_feature_count
    if target_dim <= 0:
        raise ValueError("ADNI screening state projection target dimension must be positive after summary features.")
    return target_dim


def fit_train_split_projection(
    config: dict,
    sample_index: pd.DataFrame,
    matrix_norm: np.ndarray,
    target_latent_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    adapter_config = adni_section(config)["application"]["adni_tabular_adapter"]
    projection_config = adapter_config["projection_policy"]
    fit_split = str(adapter_config["fit_split"])
    train_indices = split_indices(sample_index, fit_split)
    minimum_train_samples = int(projection_config["minimum_train_samples"])
    if train_indices.size < minimum_train_samples:
        raise ValueError(
            f"ADNI state projection requires at least {minimum_train_samples} {fit_split} samples; found {train_indices.size}."
        )
    train_matrix = matrix_norm[train_indices].astype(np.float64)
    train_center = train_matrix.mean(axis=0)
    centered_train = train_matrix - train_center
    _, _, vh = np.linalg.svd(centered_train, full_matrices=False)
    rng = np.random.default_rng(int(projection_config["random_seed"]))
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(matrix_norm.shape[1], 1)),
        size=(matrix_norm.shape[1], target_latent_dim),
    )
    svd_component_count = min(target_latent_dim, vh.shape[0])
    if svd_component_count:
        projection[:, :svd_component_count] = vh[:svd_component_count].T
    column_norm = np.linalg.norm(projection, axis=0, keepdims=True)
    projection = projection / np.maximum(column_norm, 1.0e-8)
    report = {
        "report_kind": "adni_state_projection_report",
        "participant_level_flag": "aggregate_projection_only",
        "adapter_name": adapter_config["adapter_name"],
        "projection_policy": projection_config["kind"],
        "fit_split": fit_split,
        "train_sample_count": int(train_indices.size),
        "input_feature_count": int(matrix_norm.shape[1]),
        "target_latent_dim": int(target_latent_dim),
        "svd_component_count": int(svd_component_count),
        "summary_feature_names": [str(value) for value in projection_config["summary_feature_names"]],
    }
    return train_center, projection, report


def projected_patient_states(
    config: dict,
    sample_index: pd.DataFrame,
    matrix_norm: np.ndarray,
    label_values: np.ndarray,
    centroids: np.ndarray,
    model: ADScreeningSystem,
) -> tuple[np.ndarray, dict[str, object]]:
    target_latent_dim = projection_target_latent_dim(model, config)
    train_center, projection, report = fit_train_split_projection(config, sample_index, matrix_norm, target_latent_dim)
    centered_matrix = matrix_norm.astype(np.float64) - train_center
    projected_latent = (centered_matrix @ projection).astype(np.float32)
    _, centroid_probabilities = predict_centroid_probabilities(matrix_norm, label_values, centroids)
    centroid_disease_probability = disease_positive_probabilities(label_values, centroid_probabilities).astype(np.float32)
    fit_split = str(adni_section(config)["application"]["adni_tabular_adapter"]["fit_split"])
    train_mask = sample_index["benchmark_split"].astype(str).to_numpy() == fit_split
    diagnosis_numeric = pd.to_numeric(sample_index["diagnosis_numeric"], errors="coerce").to_numpy(dtype=np.float64)
    labeled_train_mask = train_mask & np.isfinite(diagnosis_numeric)
    binary_labels = (diagnosis_numeric[labeled_train_mask] >= 2.0).astype(np.int64)
    differential = disease_differential_vector(matrix_norm[labeled_train_mask], binary_labels)
    patient_states = patient_state_feature_matrix(projected_latent, centroid_disease_probability, matrix_norm, differential)
    if patient_states.shape[1] != int(model.patient_state_dim):
        raise ValueError(
            f"ADNI projected patient-state dimension {patient_states.shape[1]} does not match screening checkpoint dimension {model.patient_state_dim}."
        )
    return patient_states, report


def tensor_mean_norm(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    detached = tensor.detach().cpu()
    flat = detached.reshape(detached.shape[0], -1)
    return flat.mean(dim=1).numpy(), torch.linalg.norm(flat, dim=1).numpy()


def empty_prediction_frame(config: dict) -> pd.DataFrame:
    return pd.DataFrame(columns=config["schemas"]["adni_world_state_prediction_fields"])


def world_state_prediction_frame(
    config: dict,
    sample_index: pd.DataFrame,
    matrix_norm: np.ndarray,
    patient_states: np.ndarray,
    model: ADScreeningSystem,
    label_values: np.ndarray,
    centroids: np.ndarray,
) -> pd.DataFrame:
    adapter_config = adni_section(config)["application"]["adni_tabular_adapter"]
    inference_splits = {str(value) for value in adapter_config["inference_splits"]}
    selected_indices = np.array(
        [
            index
            for index, split_name in enumerate(sample_index["benchmark_split"].astype(str).to_numpy())
            if split_name in inference_splits
        ],
        dtype=np.int64,
    )
    if selected_indices.size == 0:
        return empty_prediction_frame(config)
    batch_size = int(adapter_config["batch_size"])
    _, centroid_probabilities = predict_centroid_probabilities(matrix_norm, label_values, centroids)
    centroid_disease_probability = disease_positive_probabilities(label_values, centroid_probabilities)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_start in range(0, selected_indices.size, batch_size):
            batch_indices = selected_indices[batch_start:batch_start + batch_size]
            patient_tensor = torch.tensor(patient_states[batch_indices], dtype=torch.float32)
            disease_logit, _ = model.infer_state(patient_tensor)
            disease_probability = torch.sigmoid(disease_logit).detach().cpu().numpy()
            world_state_payload = model.infer_world_state(patient_tensor)
            factor_summaries = {}
            for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]:
                factor_mean, factor_norm = tensor_mean_norm(world_state_payload[factor_name])
                factor_summaries[f"{factor_name}_mean"] = factor_mean
                factor_summaries[f"{factor_name}_norm"] = factor_norm
            baseline_mean = world_state_payload["baseline_field"].detach().cpu().reshape(len(batch_indices), -1).mean(dim=1).numpy()
            parameter_mean = world_state_payload["hidden_parameter_posterior"].detach().cpu().reshape(len(batch_indices), -1).mean(dim=1).numpy()
            for offset, row_index in enumerate(batch_indices):
                sample_row = sample_index.iloc[int(row_index)]
                rows.append(
                    {
                        "sample_uid": str(sample_row["sample_uid"]),
                        "benchmark_split": str(sample_row["benchmark_split"]),
                        "diagnosis_numeric": str(sample_row["diagnosis_numeric"]),
                        "centroid_disease_probability": float(centroid_disease_probability[row_index]),
                        "screening_disease_probability": float(disease_probability[offset]),
                        "z_tx_mean": float(factor_summaries["z_tx_mean"][offset]),
                        "z_tx_norm": float(factor_summaries["z_tx_norm"][offset]),
                        "z_pk_mean": float(factor_summaries["z_pk_mean"][offset]),
                        "z_pk_norm": float(factor_summaries["z_pk_norm"][offset]),
                        "z_env_mean": float(factor_summaries["z_env_mean"][offset]),
                        "z_env_norm": float(factor_summaries["z_env_norm"][offset]),
                        "z_ephys_mean": float(factor_summaries["z_ephys_mean"][offset]),
                        "z_ephys_norm": float(factor_summaries["z_ephys_norm"][offset]),
                        "z_morph_mean": float(factor_summaries["z_morph_mean"][offset]),
                        "z_morph_norm": float(factor_summaries["z_morph_norm"][offset]),
                        "baseline_field_mean": float(baseline_mean[offset]),
                        "hidden_parameter_posterior_mean": float(parameter_mean[offset]),
                    }
                )
    return pd.DataFrame(rows, columns=config["schemas"]["adni_world_state_prediction_fields"])


def write_parquet_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    temporary_path = path.parent / f".{path.name}.{os.getpid()}.part"
    dataframe.to_parquet(temporary_path, index=False, engine="pyarrow")
    temporary_path.replace(path)


def state_metrics_for_split(
    sample_index: pd.DataFrame,
    matrix_norm: np.ndarray,
    label_values: np.ndarray,
    centroids: np.ndarray,
    split_name: str,
) -> dict[str, object]:
    indices = split_indices(sample_index, split_name)
    if indices.size == 0:
        return {"sample_count": 0, "labeled_sample_count": 0, "balanced_accuracy": None, "macro_f1": None, "brier_score": None}
    split_frame = sample_index.iloc[indices]
    labeled_indices = indices[finite_label_mask(split_frame["diagnosis_numeric"])]
    if labeled_indices.size == 0:
        return {"sample_count": int(indices.size), "labeled_sample_count": 0, "balanced_accuracy": None, "macro_f1": None, "brier_score": None}
    labels = state_labels(sample_index.iloc[labeled_indices])
    predictions, probabilities = predict_centroid_probabilities(matrix_norm[labeled_indices], label_values, centroids)
    positive_probabilities = disease_positive_probabilities(label_values, probabilities)
    binary_labels = (labels >= 2).astype(np.int64)
    return {
        "sample_count": int(indices.size),
        "labeled_sample_count": int(labeled_indices.size),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "macro_f1": macro_f1(labels, predictions),
        "brier_score": brier_score(binary_labels, positive_probabilities),
    }


def feature_column(feature_index: pd.DataFrame, feature_name: str) -> int:
    matches = np.where(feature_index["feature_name"].astype(str).to_numpy() == feature_name)[0]
    if matches.size == 0:
        raise ValueError(f"ADNI controlled model-input bundle lacks feature: {feature_name}")
    return int(matches[0])


def pathology_calibration(
    config: dict,
    sample_index: pd.DataFrame,
    feature_index: pd.DataFrame,
    matrix_norm: np.ndarray,
) -> dict[str, object]:
    thresholds = adni_section(config)["application"]["pathology_thresholds"]
    feature_specs = {
        "amyloid": ("amyloid_suvr", float(thresholds["amyloid_suvr_positive_min"])),
        "tau": ("tau_suvr", float(thresholds["tau_suvr_positive_min"])),
    }
    calibration: dict[str, object] = {}
    raw_feature_frame = pd.read_parquet(processed_path(config, "feature_matrix"))
    raw_lookup = raw_feature_frame.pivot(index="sample_uid", columns="feature_name", values="feature_value")
    for label_name, (feature_name, threshold_value) in feature_specs.items():
        column_index = feature_column(feature_index, feature_name)
        split_payload: dict[str, object] = {}
        for split_name in ["validation", "test"]:
            split_sample_index = sample_index.iloc[split_indices(sample_index, split_name)]
            observed_values = raw_lookup.reindex(split_sample_index["sample_uid"].astype(str))[feature_name].astype(str)
            observed_mask = observed_values.str.len().to_numpy() > 0
            if not observed_mask.any():
                split_payload[split_name] = {"sample_count": 0, "brier_score": None}
                continue
            labels = (observed_values[observed_mask].astype(float).to_numpy() >= threshold_value).astype(np.int64)
            logits = matrix_norm[split_indices(sample_index, split_name)][observed_mask, column_index]
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            split_payload[split_name] = {
                "sample_count": int(labels.size),
                "brier_score": brier_score(labels, probabilities),
            }
        calibration[label_name] = split_payload
    return calibration


def feature_columns_for_modality(feature_index: pd.DataFrame, modality: str) -> np.ndarray:
    modality_values = feature_index["modality_flag"].astype(str).to_numpy()
    if modality == "fluid":
        return np.where(np.isin(modality_values, ["csf", "plasma"]))[0]
    return np.where(modality_values == modality)[0]


def missingness_robustness(
    config: dict,
    sample_index: pd.DataFrame,
    feature_index: pd.DataFrame,
    matrix_norm: np.ndarray,
    label_values: np.ndarray,
    centroids: np.ndarray,
    complete_metrics: dict[str, dict[str, object]],
) -> dict[str, object]:
    task_rows: list[dict[str, object]] = []
    for task_config in adni_section(config)["application"]["missingness_tasks"]:
        modality = str(task_config["modality"])
        masked_columns = feature_columns_for_modality(feature_index, modality)
        masked_matrix = matrix_norm.copy()
        masked_matrix[:, masked_columns] = 0.0
        for split_name in ["validation", "test"]:
            indices = split_indices(sample_index, split_name)
            split_frame = sample_index.iloc[indices]
            labeled_indices = indices[finite_label_mask(split_frame["diagnosis_numeric"])]
            if labeled_indices.size:
                labels = state_labels(sample_index.iloc[labeled_indices])
                predictions, _ = predict_centroid_probabilities(masked_matrix[labeled_indices], label_values, centroids)
                task_accuracy = balanced_accuracy(labels, predictions)
            else:
                task_accuracy = None
            complete_accuracy = complete_metrics[split_name]["balanced_accuracy"]
            accuracy_delta = None if task_accuracy is None or complete_accuracy is None else float(task_accuracy - float(complete_accuracy))
            task_rows.append(
                {
                    "task_id": str(task_config["task_id"]),
                    "modality": modality,
                    "split": split_name,
                    "sample_count": int(indices.size),
                    "labeled_sample_count": int(labeled_indices.size),
                    "masked_feature_count": int(masked_columns.size),
                    "balanced_accuracy": task_accuracy,
                    "balanced_accuracy_delta_from_complete": accuracy_delta,
                }
            )
    return {
        "report_kind": "adni_missing_modality_robustness",
        "access_boundary": "controlled_access_aggregate_only",
        "participant_level_flag": "aggregate_only",
        "benchmark_id": adni_section(config)["application"]["benchmark_id"],
        "task_metrics": task_rows,
    }


def longitudinal_validation(
    config: dict,
    sample_index: pd.DataFrame,
    transition_index: pd.DataFrame,
    matrix_norm: np.ndarray,
) -> dict[str, object]:
    sample_position = {str(sample_uid): index for index, sample_uid in enumerate(sample_index["sample_uid"].astype(str))}
    train_rows = transition_index[transition_index["benchmark_split"].astype(str) == "train"]
    train_labels = (train_rows["transition_label"].astype(str) == "progression").astype(np.int64).to_numpy()
    train_positions = np.array([sample_position[str(value)] for value in train_rows["source_sample_uid"].astype(str)], dtype=np.int64)
    progression_rate_values = pd.to_numeric(train_rows["progression_rate"], errors="coerce").dropna().to_numpy()
    mean_progression_rate = float(progression_rate_values.mean()) if progression_rate_values.size else 0.0
    if not train_positions.size:
        raise ValueError("ADNI longitudinal validation requires at least one train-split transition.")
    label_values, centroids = fit_centroids(matrix_norm[train_positions], train_labels)
    metrics_by_split: dict[str, object] = {}
    for split_name in ["validation", "test"]:
        split_rows = transition_index[transition_index["benchmark_split"].astype(str) == split_name]
        if split_rows.empty:
            metrics_by_split[split_name] = {
                "transition_count": 0,
                "conversion_balanced_accuracy": None,
                "progression_rate_mae": None,
            }
            continue
        positions = np.array([sample_position[str(value)] for value in split_rows["source_sample_uid"].astype(str)], dtype=np.int64)
        labels = (split_rows["transition_label"].astype(str) == "progression").astype(np.int64).to_numpy()
        predictions, _ = predict_centroid_probabilities(matrix_norm[positions], label_values, centroids)
        rates = pd.to_numeric(split_rows["progression_rate"], errors="coerce").dropna().to_numpy()
        metrics_by_split[split_name] = {
            "transition_count": int(split_rows.shape[0]),
            "conversion_balanced_accuracy": balanced_accuracy(labels, predictions),
            "progression_rate_mae": float(np.mean(np.abs(rates - mean_progression_rate))) if rates.size else None,
        }
    return {
        "report_kind": "adni_longitudinal_validation_summary",
        "access_boundary": "controlled_access_aggregate_only",
        "participant_level_flag": "aggregate_only",
        "benchmark_id": adni_section(config)["application"]["benchmark_id"],
        "metrics_by_split": metrics_by_split,
        "train_transition_count": int(train_rows.shape[0]),
    }


def finite_numeric_array(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(numeric_values)
    return numeric_values[mask], mask


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std == 0.0 or right_std == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bin_count: int) -> float | None:
    if labels.size == 0:
        return None
    boundaries = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    for bin_index in range(bin_count):
        lower = boundaries[bin_index]
        upper = boundaries[bin_index + 1]
        if bin_index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        bin_accuracy = float(labels[mask].mean())
        bin_confidence = float(probabilities[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
    return ece


def posterior_calibration_summary(config: dict, prediction_frame: pd.DataFrame) -> dict[str, object]:
    adapter_config = adni_section(config)["application"]["adni_tabular_adapter"]
    bin_count = int(adapter_config["posterior_calibration"]["ece_bin_count"])
    metrics_by_split: dict[str, object] = {}
    for split_name in adapter_config["inference_splits"]:
        split_frame = prediction_frame[prediction_frame["benchmark_split"].astype(str) == str(split_name)]
        labels_raw, label_mask = finite_numeric_array(split_frame["diagnosis_numeric"])
        probabilities = pd.to_numeric(split_frame["screening_disease_probability"], errors="coerce").to_numpy(dtype=np.float64)
        probabilities = probabilities[label_mask]
        labels = (labels_raw >= 2.0).astype(np.int64)
        metrics_by_split[str(split_name)] = {
            "sample_count": int(labels.size),
            "brier_score": brier_score(labels, probabilities),
            "expected_calibration_error": expected_calibration_error(labels, probabilities, bin_count),
        }
    return {
        "report_kind": "adni_posterior_calibration_summary",
        "access_boundary": "controlled_access_aggregate_only",
        "participant_level_flag": "aggregate_only",
        "benchmark_id": adni_section(config)["application"]["benchmark_id"],
        "posterior_source": "ADScreeningSystem.infer_state",
        "ece_bin_count": bin_count,
        "metrics_by_split": metrics_by_split,
        "claim_policy": "observational_controlled_validation_only_no_clinical_intervention_causality_claim",
    }


def world_state_factor_anchoring(
    config: dict,
    sample_index: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> dict[str, object]:
    feature_frame = pd.read_parquet(processed_path(config, "feature_matrix"))
    raw_lookup = feature_frame.pivot(index="sample_uid", columns="feature_name", values="feature_value")
    sample_lookup = sample_index.set_index("sample_uid")
    anchor_features = [str(value) for value in adni_section(config)["application"]["adni_tabular_adapter"]["factor_anchoring"]["anchor_features"]]
    factor_columns = [
        "z_tx_norm",
        "z_pk_norm",
        "z_env_norm",
        "z_ephys_norm",
        "z_morph_norm",
        "baseline_field_mean",
        "hidden_parameter_posterior_mean",
    ]
    rows: list[dict[str, object]] = []
    for split_name in adni_section(config)["application"]["adni_tabular_adapter"]["inference_splits"]:
        split_predictions = prediction_frame[prediction_frame["benchmark_split"].astype(str) == str(split_name)]
        for factor_column in factor_columns:
            factor_values = pd.to_numeric(split_predictions[factor_column], errors="coerce").to_numpy(dtype=np.float64)
            factor_mask = np.isfinite(factor_values)
            for anchor_feature in anchor_features:
                if anchor_feature == "diagnosis_numeric":
                    anchor_series = sample_lookup.reindex(split_predictions["sample_uid"].astype(str))["diagnosis_numeric"]
                else:
                    anchor_series = raw_lookup.reindex(split_predictions["sample_uid"].astype(str))[anchor_feature]
                anchor_values = pd.to_numeric(anchor_series, errors="coerce").to_numpy(dtype=np.float64)
                anchor_mask = np.isfinite(anchor_values)
                combined_mask = factor_mask & anchor_mask
                rows.append(
                    {
                        "split": str(split_name),
                        "world_state_factor": factor_column,
                        "anchor_feature": anchor_feature,
                        "sample_count": int(combined_mask.sum()),
                        "pearson_correlation": pearson_correlation(factor_values[combined_mask], anchor_values[combined_mask]),
                    }
                )
    return {
        "report_kind": "adni_world_state_factor_anchoring",
        "access_boundary": "controlled_access_aggregate_only",
        "participant_level_flag": "aggregate_only",
        "benchmark_id": adni_section(config)["application"]["benchmark_id"],
        "anchor_metrics": rows,
        "claim_policy": "factor_anchoring_is_associational_observational_not_external_calibrated_dynamics_accuracy",
    }


def build_reports(config: dict) -> dict[str, Path]:
    sample_index, feature_index, transition_index, matrix_norm, _ = load_bundle(config)
    labeled_mask = finite_label_mask(sample_index["diagnosis_numeric"])
    train_mask = (sample_index["benchmark_split"].astype(str).to_numpy() == "train") & labeled_mask
    labels = state_labels(sample_index.loc[labeled_mask])
    train_labels = state_labels(sample_index.loc[train_mask])
    label_values, centroids = fit_centroids(matrix_norm[train_mask], train_labels)
    screening_model, checkpoint_report = load_screening_model(config)
    patient_states, projection_report = projected_patient_states(config, sample_index, matrix_norm, label_values, centroids, screening_model)
    prediction_frame = world_state_prediction_frame(
        config,
        sample_index,
        matrix_norm,
        patient_states,
        screening_model,
        label_values,
        centroids,
    )
    metrics_by_split = {
        split_name: state_metrics_for_split(sample_index, matrix_norm, label_values, centroids, split_name)
        for split_name in ["validation", "test"]
    }
    sample_counts_by_split = {
        split_name: int((sample_index["benchmark_split"].astype(str) == split_name).sum())
        for split_name in ["train", "validation", "test"]
    }
    state_summary = {
        "report_kind": "adni_state_inference_summary",
        "access_boundary": "controlled_access_aggregate_only",
        "participant_level_flag": "aggregate_only",
        "benchmark_id": adni_section(config)["application"]["benchmark_id"],
        "model_family": adni_section(config)["application"]["model_family"],
        "sample_counts_by_split": sample_counts_by_split,
        "label_count": int(labels.size),
        "train_label_count": int(train_labels.size),
        "metrics_by_split": metrics_by_split,
        "pathology_calibration": pathology_calibration(config, sample_index, feature_index, matrix_norm),
        "screening_checkpoint_path": checkpoint_report["checkpoint_path"],
        "screening_checkpoint_selection": checkpoint_report,
        "tabular_adapter_name": adni_section(config)["application"]["adni_tabular_adapter"]["adapter_name"],
        "world_state_private_prediction_count": int(prediction_frame.shape[0]),
    }
    longitudinal_summary = longitudinal_validation(config, sample_index, transition_index, matrix_norm)
    robustness_summary = missingness_robustness(config, sample_index, feature_index, matrix_norm, label_values, centroids, metrics_by_split)
    factor_anchoring_summary = world_state_factor_anchoring(config, sample_index, prediction_frame)
    posterior_summary = posterior_calibration_summary(config, prediction_frame)
    outputs = {
        "state_inference_summary": report_path(config, "state_inference_summary"),
        "longitudinal_validation_summary": report_path(config, "longitudinal_validation_summary"),
        "missing_modality_robustness": report_path(config, "missing_modality_robustness"),
        "world_state_factor_anchoring": report_path(config, "world_state_factor_anchoring"),
        "posterior_calibration_summary": report_path(config, "posterior_calibration_summary"),
        "controlled_bundle_pointer": report_path(config, "controlled_bundle_pointer"),
        "state_projection_report": processed_path(config, "state_projection_report"),
        "world_state_predictions": processed_path(config, "world_state_predictions"),
    }
    ensure_directory(outputs["state_inference_summary"].parent)
    write_json(outputs["state_inference_summary"], state_summary)
    write_json(outputs["longitudinal_validation_summary"], longitudinal_summary)
    write_json(outputs["missing_modality_robustness"], robustness_summary)
    write_json(outputs["world_state_factor_anchoring"], factor_anchoring_summary)
    write_json(outputs["posterior_calibration_summary"], posterior_summary)
    write_json(outputs["state_projection_report"], projection_report)
    write_json(
        outputs["controlled_bundle_pointer"],
        controlled_bundle_pointer_payload(
            config,
            sample_index.to_dict(orient="records"),
            transition_index.to_dict(orient="records"),
            outputs,
        ),
    )
    write_parquet_atomic(prediction_frame, outputs["world_state_predictions"])
    return outputs


def main() -> None:
    args = parse_args()
    config = load_controlled_research_config(args.config)
    assert_controlled_governance_approval(config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    output_paths = build_reports(config)
    pipeline_logger.progress(
        "adni_controlled_benchmark_completed",
        report_count=len(output_paths),
        access_boundary="controlled_access_aggregate_only",
    )
    complete_script_logging(pipeline_logger, pipeline_started_at, report_count=len(output_paths))


if __name__ == "__main__":
    main()
