from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config, load_benchmark_catalogs
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, write_json
from Libs.utils.evaluation.screening_candidate_selection import (
    diversified_candidate_indices,
    regimen_prior_scores,
    target_program_prior_scores,
)
from Libs.utils.evaluation.screening_datasets import (
    fold_patient_latents,
    patient_regimen_labels,
    patient_target_labels,
    regimen_features,
    target_program_features,
    target_uid_to_entity_lookup,
    top_columns,
)
from Libs.utils.data.screening_pipeline import pkpd_reference_tensors


def weighted_feature_context(feature_matrix: np.ndarray, label_matrix: np.ndarray) -> np.ndarray:
    if feature_matrix.size == 0:
        return np.empty((label_matrix.shape[0], 0), dtype=np.float32)
    weights = np.clip(label_matrix.astype(np.float32), a_min=0.0, a_max=None)
    denominator = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    return (weights @ feature_matrix.astype(np.float32)) / denominator


def build_proxy_transition_target(
    patient_latent: np.ndarray,
    target_context: np.ndarray,
    regimen_context: np.ndarray,
    pkpd_context: np.ndarray,
) -> np.ndarray:
    latent_dim = patient_latent.shape[1]
    target_delta = np.zeros_like(patient_latent, dtype=np.float32)
    regimen_delta = np.zeros_like(patient_latent, dtype=np.float32)
    pkpd_delta = np.zeros_like(patient_latent, dtype=np.float32)
    for source_context, output in [
        (target_context, target_delta),
        (regimen_context, regimen_delta),
        (pkpd_context, pkpd_delta),
    ]:
        if source_context.size == 0:
            continue
        repeat_count = int(np.ceil(latent_dim / source_context.shape[1]))
        tiled = np.tile(source_context, (1, repeat_count))[:, :latent_dim]
        output[:] = tiled.astype(np.float32)
    proxy = patient_latent.astype(np.float32) + 0.25 * target_delta + 0.45 * regimen_delta + 0.30 * pkpd_delta
    norms = np.linalg.norm(proxy, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return proxy / norms


def pkpd_context_vector(regimen_feature_frame: pd.DataFrame, config: dict) -> np.ndarray:
    exposure_target_tensor, occupancy_target_tensor, pk_feasibility_tensor = pkpd_reference_tensors(
        regimen_feature_frame,
        config,
        torch.device("cpu"),
    )
    pkpd_reference = torch.cat([exposure_target_tensor, occupancy_target_tensor], dim=1).cpu().numpy().astype(np.float32)
    return np.concatenate(
        [
            pkpd_reference.mean(axis=0),
            pk_feasibility_tensor.cpu().numpy().astype(np.float32),
        ],
        axis=0,
    )


def proxy_transition_targets(
    patient_latent: np.ndarray,
    target_feature_matrix: np.ndarray,
    regimen_feature_matrix: np.ndarray,
    target_label_vector: np.ndarray,
    regimen_label_vector: np.ndarray,
    pkpd_context_seed: np.ndarray,
) -> np.ndarray:
    target_context = weighted_feature_context(target_feature_matrix, target_label_vector)
    regimen_context = weighted_feature_context(regimen_feature_matrix, regimen_label_vector)
    pkpd_context = np.repeat(pkpd_context_seed[None, :], patient_latent.shape[0], axis=0).astype(np.float32)
    return build_proxy_transition_target(patient_latent, target_context, regimen_context, pkpd_context)


def select_training_actions(
    config: dict,
    disease_program: str,
    fold_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    payload = {
        modality: {
            split_name: fold_patient_latents(config, fold_name, split_name, disease_program)[modality]
            for split_name in ["train", "validation", "test"]
        }
        for modality in ["transcriptomics", "metabolomics"]
    }

    action_root = Path(config["screening_pipeline"]["action_ready_registry"]["local_root"]).resolve()
    rel = config["screening_pipeline"]["action_ready_registry"]["relative_paths"]
    target_program_rows = read_tsv(action_root / rel["target_program_catalog"])
    regimen_rows = read_tsv(action_root / rel["regimen_catalog"])
    target_rows = read_tsv(
        Path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]).resolve()
        / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"]
    )
    target_uid_to_entity = target_uid_to_entity_lookup(target_rows)
    training_cfg = config["screening_pipeline"]["screening_benchmark"]["training"]
    target_frame, target_feature_matrix = target_program_features(target_program_rows)
    regimen_frame, regimen_feature_matrix = regimen_features(regimen_rows)
    aggregate_target_scores: list[np.ndarray] = []
    aggregate_regimen_scores: list[np.ndarray] = []
    label_payload: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for modality, split_dict in payload.items():
        label_payload[modality] = {}
        train_payload = split_dict["train"]
        entity_lookup = {
            str(entity_id): index
            for index, entity_id in enumerate(train_payload["entity_index"]["canonical_entity_id"].astype(str).tolist())
        }
        train_target_labels = patient_target_labels(train_payload["matrix"], train_payload["differential"], entity_lookup, target_frame)
        train_regimen_labels = patient_regimen_labels(train_payload["matrix"], train_payload["differential"], entity_lookup, regimen_frame, target_uid_to_entity)
        aggregate_target_scores.append(train_target_labels.mean(axis=0) if train_target_labels.size else np.zeros((target_frame.shape[0],), dtype=np.float32))
        aggregate_regimen_scores.append(train_regimen_labels.mean(axis=0) if train_regimen_labels.size else np.zeros((regimen_frame.shape[0],), dtype=np.float32))
        label_payload[modality]["train"] = {
            "target_labels": train_target_labels,
            "regimen_labels": train_regimen_labels,
        }
        for split_name in ["validation", "test"]:
            split_payload = split_dict[split_name]
            split_entity_lookup = {
                str(entity_id): index
                for index, entity_id in enumerate(split_payload["entity_index"]["canonical_entity_id"].astype(str).tolist())
            }
            label_payload[modality][split_name] = {
                "target_labels": patient_target_labels(split_payload["matrix"], split_payload["differential"], split_entity_lookup, target_frame),
                "regimen_labels": patient_regimen_labels(split_payload["matrix"], split_payload["differential"], split_entity_lookup, regimen_frame, target_uid_to_entity),
            }
    mean_target_scores = np.mean(np.stack(aggregate_target_scores, axis=0), axis=0)
    mean_regimen_scores = np.mean(np.stack(aggregate_regimen_scores, axis=0), axis=0)
    target_indices = diversified_candidate_indices(
        mean_target_scores,
        target_program_prior_scores(target_frame),
        target_frame["canonical_entity_id"].astype(str).tolist(),
        int(training_cfg["training_target_count"]),
    )
    regimen_indices = diversified_candidate_indices(
        mean_regimen_scores,
        regimen_prior_scores(regimen_frame),
        regimen_frame["target_symbol_set"].astype(str).tolist(),
        int(training_cfg["training_regimen_count"]),
    )
    selected_target_frame = target_frame.iloc[target_indices].reset_index(drop=True)
    selected_regimen_frame = regimen_frame.iloc[regimen_indices].reset_index(drop=True)
    for modality, split_dict in label_payload.items():
        for split_name, current_labels in split_dict.items():
            current_labels["target_labels"] = current_labels["target_labels"][:, target_indices]
            current_labels["regimen_labels"] = current_labels["regimen_labels"][:, regimen_indices]
    return selected_target_frame, selected_regimen_frame, {
        "payload": payload,
        "labels": label_payload,
        "target_uid_to_entity": target_uid_to_entity,
    }


def build_transition_rows(
    config_path: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
    baseline_name: str = "",
) -> Path:
    config = load_config(config_path)
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    selected_target_frame, selected_regimen_frame, transition_inputs = select_training_actions(config, disease_program, fold_name)
    target_feature_frame, target_feature_matrix = target_program_features(selected_target_frame.to_dict("records"))
    regimen_feature_frame, regimen_feature_matrix = regimen_features(selected_regimen_frame.to_dict("records"))
    pkpd_context_seed = pkpd_context_vector(regimen_feature_frame, config)
    ensure_directory(output_dir)

    catalog_rows: list[dict[str, Any]] = []
    for modality, split_payload in transition_inputs["payload"].items():
        for split_name, current_payload in split_payload.items():
            patient_latent = current_payload["latent"].astype(np.float32)
            if patient_latent.shape[0] == 0:
                continue
            disease_label = current_payload["labels"].astype(np.int64)
            target_label_vector = transition_inputs["labels"][modality][split_name]["target_labels"].astype(np.float32)
            regimen_label_vector = transition_inputs["labels"][modality][split_name]["regimen_labels"].astype(np.float32)
            target_context = weighted_feature_context(target_feature_matrix, target_label_vector)
            regimen_context = weighted_feature_context(regimen_feature_matrix, regimen_label_vector)
            pkpd_context = np.repeat(pkpd_context_seed[None, :], patient_latent.shape[0], axis=0).astype(np.float32)
            proxy_transition_target = proxy_transition_targets(
                patient_latent,
                target_feature_matrix,
                regimen_feature_matrix,
                target_label_vector,
                regimen_label_vector,
                pkpd_context_seed,
            )
            array_path = output_dir / f"{disease_program}_{fold_name}_{modality}_{split_name}_latent_transition_ready.npz"
            np.savez_compressed(
                array_path,
                source_patient_latent=patient_latent,
                target_program_context=target_context.astype(np.float32),
                regimen_feature_vector=regimen_context.astype(np.float32),
                pkpd_reference=pkpd_context.astype(np.float32),
                disease_label=disease_label,
                target_label_vector=target_label_vector,
                regimen_label_vector=regimen_label_vector,
                proxy_transition_target=proxy_transition_target.astype(np.float32),
                sample_uid=current_payload["metadata"]["sample_uid"].astype(str).to_numpy(),
                study_uid=current_payload["metadata"]["study_uid"].astype(str).to_numpy(),
            )
            catalog_rows.append(
                {
                    "disease_program": disease_program,
                    "fold_name": fold_name,
                    "modality": modality,
                    "split_name": split_name,
                    "path": str(array_path),
                    "sample_count": int(patient_latent.shape[0]),
                    "patient_latent_dim": int(patient_latent.shape[1]),
                    "target_context_dim": int(target_context.shape[1]) if target_context.size else 0,
                    "regimen_context_dim": int(regimen_context.shape[1]) if regimen_context.size else 0,
                    "pkpd_context_dim": int(pkpd_context.shape[1]),
                    "target_label_dim": int(target_label_vector.shape[1]),
                    "regimen_label_dim": int(regimen_label_vector.shape[1]),
                }
            )

    manifest_path = output_dir / f"{disease_program}_{fold_name}_latent_transition_ready_manifest.json"
    write_json(
        manifest_path,
        {
            "baseline_name": baseline_name,
            "disease_program": disease_program,
            "fold_name": fold_name,
            "benchmark_name": config["benchmark_registry"]["benchmark_name"],
            "selected_target_programs": int(selected_target_frame.shape[0]),
            "selected_regimens": int(selected_regimen_frame.shape[0]),
            "catalog_rows": catalog_rows,
        },
    )
    selected_target_frame.to_parquet(output_dir / f"{disease_program}_{fold_name}_selected_target_programs.parquet", index=False, engine="pyarrow")
    selected_regimen_frame.to_parquet(output_dir / f"{disease_program}_{fold_name}_selected_regimens.parquet", index=False, engine="pyarrow")
    return manifest_path


def load_transition_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))
