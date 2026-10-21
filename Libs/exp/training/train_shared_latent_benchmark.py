from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging, get_pipeline_logger
from Libs.models.networks.shared_latent_dual_encoder import SharedLatentDualEncoder
from Libs.models.networks.tabular_jepa_auxiliary_encoder import TabularJEPAAuxiliaryEncoder
from Libs.utils.exp.experiment_tracking import capture_run_metadata, scoped_seed, seed_everything
from Libs.utils.evaluation.alignment_benchmark import (
    binary_label,
    choose_group_identifier,
    compute_binary_metrics,
    normalize_diagnosis_label,
)
from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config
from Libs.utils.evaluation.benchmark_datasets import (
    alignment_by_study,
    assemble_dataset,
    build_entity_index_from_studies,
    fold_entity_index,
    fold_sample_split,
    load_benchmark_catalogs,
    train_context_standardized_split_matrices,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical baselines and shared-latent models for the NeuronSifter benchmark.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--fold-name",
        default="",
        help="Optional benchmark fold name subset for smoke tests.",
    )
    parser.add_argument(
        "--training-scope",
        default="all",
        choices=["folds", "production", "all"],
        help="Train fold models, production model, or both.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device for the shared-latent neural model.",
    )
    parser.add_argument(
        "--disease-program",
        default="",
        help="Optional disease program override, for example ad_control.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def artifact_dir(config: dict, scope: str, fold_name: str, model_name: str) -> Path:
    root = resolve_output_root(config)
    artifact_benchmark_name = config["benchmark_registry"].get("artifact_benchmark_name", config["benchmark_registry"]["benchmark_name"])
    if scope == "production":
        return root / artifact_benchmark_name / "production" / model_name
    return root / artifact_benchmark_name / "folds" / fold_name / model_name


def resolve_output_root(config: dict) -> Path:
    return Path(config["benchmark_registry"]["artifact_root"]).resolve()


def grouped_validation_assignments(
    training_frame: pd.DataFrame,
    validation_fraction: float,
    minimum_validation_groups_per_label: int,
    random_seed: int,
) -> dict[str, str]:
    training_frame = training_frame.copy()
    group_labels = training_frame.groupby("group_id")["binary_label"].nunique()
    mixed_groups = sorted(group_labels[group_labels > 1].index.tolist())
    if mixed_groups:
        mixed_mask = training_frame["group_id"].isin(mixed_groups)
        training_frame.loc[mixed_mask, "group_id"] = training_frame.loc[mixed_mask, "sample_uid"]
    assignments = {sample_uid: "train" for sample_uid in training_frame["sample_uid"].tolist()}
    if training_frame.empty:
        return assignments
    for label_value in sorted(training_frame["binary_label"].unique()):
        label_groups = sorted(training_frame.loc[training_frame["binary_label"] == label_value, "group_id"].unique().tolist())
        if len(label_groups) <= 1:
            continue
        requested = max(minimum_validation_groups_per_label, int(round(len(label_groups) * validation_fraction)))
        requested = min(requested, len(label_groups) - 1)
        if requested <= 0:
            continue
        ordered_groups = pd.Series(label_groups).sample(frac=1.0, random_state=random_seed + int(label_value)).tolist()
        for sample_uid in training_frame.loc[training_frame["group_id"].isin(ordered_groups[:requested]), "sample_uid"].tolist():
            assignments[sample_uid] = "validation"
    return assignments


def assemble_fold_datasets(config: dict, fold_name: str) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]]:
    split_frame = fold_sample_split(config, fold_name)
    datasets: dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]] = {}
    for modality in ["transcriptomics", "metabolomics"]:
        entity_index = fold_entity_index(config, fold_name, modality)
        datasets[modality] = {}
        raw_payloads: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]] = {}
        raw_matrices: dict[str, np.ndarray] = {}
        for split_role in ["train", "validation", "test"]:
            role_frame = split_frame.loc[split_frame["split_role"] == split_role].copy()
            matrix, labels, metadata = assemble_dataset(config, role_frame, modality, entity_index, matrix_kind="raw")
            raw_payloads[split_role] = (matrix, labels, metadata, entity_index)
            raw_matrices[split_role] = matrix
        standardized_matrices = train_context_standardized_split_matrices(config, modality, raw_matrices)
        for split_role, (_, labels, metadata, current_entity_index) in raw_payloads.items():
            datasets[modality][split_role] = (standardized_matrices[split_role], labels, metadata, current_entity_index)
    return datasets


def assemble_production_datasets(config: dict) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]]:
    cohort_rows, _, _ = load_benchmark_catalogs(config)
    primary_studies = sorted(
        row["study_uid"]
        for row in cohort_rows
        if row["cohort_role"] == "primary_train_eval" and row["inclusion_flag"] == "yes"
    )
    alignment_lookup = alignment_by_study(config)
    modality_thresholds = config["benchmark_registry"]["feature_presence_threshold"]
    positive_labels = set(config["benchmark_registry"]["positive_labels"])
    negative_labels = set(config["benchmark_registry"]["negative_labels"])
    diagnosis_aliases = config["cohort_ready_registry"]["diagnosis_aliases"]
    sample_rows: list[dict[str, object]] = []
    for study_uid in primary_studies:
        alignment_row = alignment_lookup[study_uid]
        sample_index = pd.read_parquet(resolve_path(alignment_row["sample_index_path"]))
        sample_index = sample_index.copy()
        sample_index["study_uid"] = study_uid
        sample_index["source_key"] = alignment_row["source_key"]
        sample_index["source_accession"] = alignment_row["source_accession"]
        sample_index["normalized_diagnosis_label"] = sample_index["sample_diagnosis_label"].astype(str).apply(
            lambda value: normalize_diagnosis_label(value, diagnosis_aliases)
        )
        sample_index["binary_label"] = sample_index["normalized_diagnosis_label"].apply(
            lambda value: binary_label(value, positive_labels, negative_labels)
        )
        sample_index["group_id"] = sample_index.apply(choose_group_identifier, axis=1)
        sample_rows.extend(sample_index.to_dict("records"))
    full_frame = pd.DataFrame(sample_rows)
    full_frame = full_frame.loc[full_frame["binary_label"].notna()].copy()
    assignments = grouped_validation_assignments(
        full_frame,
        float(config["benchmark_registry"]["validation_fraction"]),
        int(config["benchmark_registry"]["minimum_validation_groups_per_label"]),
        int(config["benchmark_registry"]["logistic_regression"]["random_seed"]),
    )
    full_frame["split_role"] = full_frame["sample_uid"].map(assignments)

    datasets: dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]] = {}
    for modality in ["transcriptomics", "metabolomics"]:
        entity_index = build_entity_index_from_studies(
            config,
            primary_studies,
            modality,
            int(modality_thresholds[modality]),
        )
        datasets[modality] = {}
        raw_payloads: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]] = {}
        raw_matrices: dict[str, np.ndarray] = {}
        for split_role in ["train", "validation"]:
            role_frame = full_frame.loc[full_frame["split_role"] == split_role].copy()
            matrix, labels, metadata = assemble_dataset(config, role_frame, modality, entity_index, matrix_kind="raw")
            raw_payloads[split_role] = (matrix, labels, metadata, entity_index)
            raw_matrices[split_role] = matrix
        standardized_matrices = train_context_standardized_split_matrices(config, modality, raw_matrices)
        for split_role, (_, labels, metadata, current_entity_index) in raw_payloads.items():
            datasets[modality][split_role] = (standardized_matrices[split_role], labels, metadata, current_entity_index)
    return datasets


def resolve_path(path_value: str) -> Path:
    return (REPO_ROOT / path_value).resolve()


def logistic_manifest(
    model_name: str,
    status: str,
    model_path: Path | None,
    modality: str,
    train_shape: tuple[int, int],
    validation_shape: tuple[int, int],
    note: str,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "status": status,
        "modality": modality,
        "model_path": str(model_path.relative_to(REPO_ROOT)) if model_path is not None else "",
        "train_shape": list(train_shape),
        "validation_shape": list(validation_shape),
        "note": note,
    }


def validation_metric_fields(metrics: dict[str, float]) -> dict[str, object]:
    if not metrics:
        return {"validation_status": "not_available"}
    return {f"validation_{metric_name}": metric_value for metric_name, metric_value in metrics.items()}


def train_logistic_baseline(
    config: dict,
    config_path: Path,
    argv: list[str],
    fold_name: str,
    modality: str,
    datasets: dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]],
) -> None:
    model_name = "aligned_gene_logistic_regression" if modality == "transcriptomics" else "aligned_metabolite_logistic_regression"
    model_logger = get_pipeline_logger(scope="fold", fold=fold_name, modality=modality, model_name=model_name)
    train_matrix, train_labels, _, entity_index = datasets[modality]["train"]
    validation_matrix, validation_labels, _, _ = datasets[modality]["validation"]
    test_matrix, _, test_metadata, _ = datasets[modality]["test"]
    model_dir = artifact_dir(config, "fold", fold_name, model_name)
    ensure_directory(model_dir)
    model_seed = scoped_seed(int(config["benchmark_registry"]["logistic_regression"]["random_seed"]), fold_name, model_name)
    seed_everything(model_seed)
    capture_run_metadata(
        model_dir,
        config_path,
        argv,
        model_seed,
        {
            "fold_name": fold_name,
            "modality": modality,
            "model_name": model_name,
            "train_shape": list(train_matrix.shape),
            "validation_shape": list(validation_matrix.shape),
            "test_shape": list(test_matrix.shape),
        },
    )
    model_logger.progress(
        "model_training_started",
        artifact=model_dir,
        train_shape=train_matrix.shape,
        validation_shape=validation_matrix.shape,
        test_shape=test_matrix.shape,
        seed=model_seed,
    )
    manifest_path = model_dir / "training_manifest.json"
    if train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0 or test_metadata.shape[0] == 0:
        write_json(
            manifest_path,
            {
                **logistic_manifest(
                    model_name,
                    "skipped",
                    None,
                    modality,
                    train_matrix.shape,
                    validation_matrix.shape,
                    "train_or_test_split_empty",
                ),
                "seed": model_seed,
                "run_metadata_path": str((model_dir / "run_metadata.json").relative_to(REPO_ROOT)),
            },
        )
        model_logger.complete(
            "model_training_skipped",
            artifact=manifest_path,
            reason="train_or_test_split_empty",
            train_shape=train_matrix.shape,
            validation_shape=validation_matrix.shape,
            test_shape=test_matrix.shape,
        )
        return

    logistic_config = config["benchmark_registry"]["logistic_regression"]
    model = LogisticRegression(
        C=float(logistic_config["regularization_strength"]),
        max_iter=int(logistic_config["max_iter"]),
        class_weight=str(logistic_config["class_weight"]),
        random_state=model_seed,
    )
    model.fit(train_matrix, train_labels)
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    entity_index_path = model_dir / "entity_index.parquet"
    entity_index.to_parquet(entity_index_path, index=False, engine="pyarrow")

    validation_metrics = {}
    if validation_matrix.shape[0] > 0 and len(np.unique(validation_labels)) >= 2:
        validation_scores = model.predict_proba(validation_matrix)[:, 1]
        validation_metrics = compute_binary_metrics(validation_labels.astype(np.int64), validation_scores.astype(np.float64))
    write_json(
        manifest_path,
        {
            **logistic_manifest(
                model_name,
                "trained",
                model_path,
                modality,
                train_matrix.shape,
                validation_matrix.shape,
                fold_name,
            ),
            "entity_index_path": str(entity_index_path.relative_to(REPO_ROOT)),
            "seed": model_seed,
            "run_metadata_path": str((model_dir / "run_metadata.json").relative_to(REPO_ROOT)),
            "validation_metrics": validation_metrics,
            "hyperparameters": {
                "regularization_strength": float(logistic_config["regularization_strength"]),
                "max_iter": int(logistic_config["max_iter"]),
                "class_weight": str(logistic_config["class_weight"]),
            },
        },
    )
    model_logger.artifact(
        "model_training_completed",
        artifact=model_path,
        seed=model_seed,
        **validation_metric_fields(validation_metrics),
    )


def torch_tensor(matrix: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def classification_loss_terms(
    model: SharedLatentDualEncoder,
    split_payload: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]],
    device: torch.device,
    loss_function: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    payload: dict[str, dict[str, torch.Tensor]] = {}
    losses: list[torch.Tensor] = []
    for modality in ["transcriptomics", "metabolomics"]:
        features, labels, metadata, _ = split_payload[modality]
        if features.shape[0] == 0:
            continue
        feature_tensor = torch_tensor(features, device)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        logits, latents = model(modality, feature_tensor)
        losses.append(loss_function(logits, label_tensor))
        payload[modality] = {
            "logits": logits,
            "latents": latents,
            "labels": label_tensor,
            "study_codes": torch.tensor(pd.factorize(metadata["study_uid"])[0], dtype=torch.long, device=device),
        }
    if not losses:
        raise ValueError("No training payload is available for shared-latent optimization.")
    return torch.stack(losses).mean(), payload


def study_alignment_loss(payload: dict[str, dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    zero = torch.tensor(0.0, device=device)
    for modality_payload in payload.values():
        labels = modality_payload["labels"].long()
        latents = modality_payload["latents"]
        studies = modality_payload["study_codes"]
        for label_value in torch.unique(labels):
            label_mask = labels == label_value
            study_centroids: list[torch.Tensor] = []
            for study_value in torch.unique(studies[label_mask]):
                study_mask = label_mask & (studies == study_value)
                if torch.sum(study_mask) == 0:
                    continue
                study_centroids.append(torch.mean(latents[study_mask], dim=0))
            if len(study_centroids) <= 1:
                continue
            centroid_stack = torch.stack(study_centroids)
            global_centroid = torch.mean(centroid_stack, dim=0, keepdim=True)
            losses.append(torch.mean((centroid_stack - global_centroid) ** 2))
    if not losses:
        return zero
    return torch.stack(losses).mean()


def modality_alignment_loss(payload: dict[str, dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    if "transcriptomics" not in payload or "metabolomics" not in payload:
        return torch.tensor(0.0, device=device)
    transcript_payload = payload["transcriptomics"]
    metabolite_payload = payload["metabolomics"]
    losses: list[torch.Tensor] = []
    for label_value in sorted(set(transcript_payload["labels"].long().tolist()) & set(metabolite_payload["labels"].long().tolist())):
        transcript_mask = transcript_payload["labels"].long() == label_value
        metabolite_mask = metabolite_payload["labels"].long() == label_value
        if torch.sum(transcript_mask) == 0 or torch.sum(metabolite_mask) == 0:
            continue
        transcript_centroid = torch.mean(transcript_payload["latents"][transcript_mask], dim=0)
        metabolite_centroid = torch.mean(metabolite_payload["latents"][metabolite_mask], dim=0)
        losses.append(torch.mean((transcript_centroid - metabolite_centroid) ** 2))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def evaluate_shared_latent_split(
    model: SharedLatentDualEncoder,
    split_payload: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    probability_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for modality in ["transcriptomics", "metabolomics"]:
            features, labels, _, _ = split_payload[modality]
            if features.shape[0] == 0:
                continue
            logits, _ = model(modality, torch_tensor(features, device))
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            probability_blocks.append(probabilities.astype(np.float64))
            label_blocks.append(labels.astype(np.int64))
    if not probability_blocks:
        return {}, {"y_true": np.empty((0,), dtype=np.int64), "y_score": np.empty((0,), dtype=np.float64)}
    y_true = np.concatenate(label_blocks, axis=0)
    y_score = np.concatenate(probability_blocks, axis=0)
    if len(np.unique(y_true)) < 2:
        return {}, {"y_true": y_true, "y_score": y_score}
    return compute_binary_metrics(y_true, y_score), {"y_true": y_true, "y_score": y_score}


def masked_feature_batch(feature_tensor: torch.Tensor, mask_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (torch.rand_like(feature_tensor) < mask_fraction).float()
    masked_tensor = feature_tensor * (1.0 - mask)
    return masked_tensor, mask


def jepa_loss_terms(
    model: TabularJEPAAuxiliaryEncoder,
    split_payload: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]],
    device: torch.device,
    classification_loss_function: torch.nn.Module,
    mask_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    classification_losses: list[torch.Tensor] = []
    reconstruction_losses: list[torch.Tensor] = []
    payload: dict[str, dict[str, torch.Tensor]] = {}
    for modality in ["transcriptomics", "metabolomics"]:
        features, labels, metadata, _ = split_payload[modality]
        if features.shape[0] == 0:
            continue
        feature_tensor = torch_tensor(features, device)
        masked_tensor, mask = masked_feature_batch(feature_tensor, mask_fraction)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        logits, latents, reconstruction = model(modality, masked_tensor)
        reconstruction_denominator = mask.sum().clamp_min(1.0)
        classification_losses.append(classification_loss_function(logits, label_tensor))
        reconstruction_losses.append(torch.sum(((reconstruction - feature_tensor) ** 2) * mask) / reconstruction_denominator)
        payload[modality] = {
            "logits": logits,
            "latents": latents,
            "labels": label_tensor,
            "study_codes": torch.tensor(pd.factorize(metadata["study_uid"])[0], dtype=torch.long, device=device),
        }
    if not classification_losses or not reconstruction_losses:
        raise ValueError("No training payload is available for tabular JEPA optimization.")
    return torch.stack(classification_losses).mean(), torch.stack(reconstruction_losses).mean(), payload


def evaluate_tabular_jepa_split(
    model: TabularJEPAAuxiliaryEncoder,
    split_payload: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    probability_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for modality in ["transcriptomics", "metabolomics"]:
            features, labels, _, _ = split_payload[modality]
            if features.shape[0] == 0:
                continue
            logits, _, _ = model(modality, torch_tensor(features, device))
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            probability_blocks.append(probabilities.astype(np.float64))
            label_blocks.append(labels.astype(np.int64))
    if not probability_blocks:
        return {}, {"y_true": np.empty((0,), dtype=np.int64), "y_score": np.empty((0,), dtype=np.float64)}
    y_true = np.concatenate(label_blocks, axis=0)
    y_score = np.concatenate(probability_blocks, axis=0)
    if len(np.unique(y_true)) < 2:
        return {}, {"y_true": y_true, "y_score": y_score}
    return compute_binary_metrics(y_true, y_score), {"y_true": y_true, "y_score": y_score}


def train_shared_latent_fold(
    config: dict,
    config_path: Path,
    argv: list[str],
    fold_name: str,
    datasets: dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]],
    device: torch.device,
) -> None:
    model_name = "shared_latent_dual_encoder_classifier"
    model_logger = get_pipeline_logger(scope="fold", fold=fold_name, model_name=model_name)
    train_payload = {modality: datasets[modality]["train"] for modality in ["transcriptomics", "metabolomics"]}
    validation_payload = {modality: datasets[modality]["validation"] for modality in ["transcriptomics", "metabolomics"]}
    transcript_entity_index = datasets["transcriptomics"]["train"][3]
    metabolite_entity_index = datasets["metabolomics"]["train"][3]
    model_dir = artifact_dir(config, "fold", fold_name, model_name)
    ensure_directory(model_dir)

    model_config = config["benchmark_registry"]["shared_latent_model"]
    model_seed = scoped_seed(int(model_config["random_seed"]), fold_name, model_name)
    seed_everything(model_seed)
    capture_run_metadata(
        model_dir,
        config_path,
        argv,
        model_seed,
        {
            "fold_name": fold_name,
            "model_name": model_name,
            "device": str(device),
            "train_shape_by_modality": {
                modality: list(datasets[modality]["train"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
            "validation_shape_by_modality": {
                modality: list(datasets[modality]["validation"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
        },
    )
    model_logger.progress(
        "model_training_started",
        artifact=model_dir,
        device=str(device),
        transcriptomics_train_shape=datasets["transcriptomics"]["train"][0].shape,
        metabolomics_train_shape=datasets["metabolomics"]["train"][0].shape,
        transcriptomics_validation_shape=datasets["transcriptomics"]["validation"][0].shape,
        metabolomics_validation_shape=datasets["metabolomics"]["validation"][0].shape,
        seed=model_seed,
    )
    model = SharedLatentDualEncoder(
        max(int(transcript_entity_index.shape[0]), 1),
        max(int(metabolite_entity_index.shape[0]), 1),
        int(model_config["transcriptomics_hidden_dim"]),
        int(model_config["metabolomics_hidden_dim"]),
        int(model_config["latent_dim"]),
        float(model_config["dropout_rate"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    best_epoch = 0
    patience = int(model_config["patience_epochs"])
    training_history: list[dict[str, object]] = []

    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        classification_loss, payload = classification_loss_terms(model, train_payload, device, loss_function)
        study_loss = study_alignment_loss(payload, device)
        modality_loss = modality_alignment_loss(payload, device)
        total_loss = (
            float(model_config["classification_loss_weight"]) * classification_loss
            + float(model_config["study_alignment_loss_weight"]) * study_loss
            + float(model_config["modality_alignment_loss_weight"]) * modality_loss
        )
        total_loss.backward()
        optimizer.step()

        validation_metrics, validation_vectors = evaluate_shared_latent_split(model, validation_payload, device)
        if validation_metrics:
            current_metric = float(validation_metrics["auroc"])
        else:
            current_metric = float("-inf")
            if validation_vectors["y_true"].shape[0] > 0:
                current_metric = float(-torch.mean(total_loss.detach()).cpu().item())
        epoch_summary = {
            "epoch": epoch,
            "total_loss": float(total_loss.detach().cpu().item()),
            "classification_loss": float(classification_loss.detach().cpu().item()),
            "study_alignment_loss": float(study_loss.detach().cpu().item()),
            "modality_alignment_loss": float(modality_loss.detach().cpu().item()),
            "selection_metric": current_metric,
            "validation_metrics": validation_metrics,
        }
        training_history.append(epoch_summary)
        model_logger.metric(
            "training_epoch_completed",
            epoch=epoch,
            total_loss=epoch_summary["total_loss"],
            classification_loss=epoch_summary["classification_loss"],
            study_alignment_loss=epoch_summary["study_alignment_loss"],
            modality_alignment_loss=epoch_summary["modality_alignment_loss"],
            selection_metric=current_metric,
            **validation_metric_fields(validation_metrics),
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint_path = model_dir / "model_weights.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {
                "transcriptomics_input_dim": max(int(transcript_entity_index.shape[0]), 1),
                "metabolomics_input_dim": max(int(metabolite_entity_index.shape[0]), 1),
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
            },
            "feature_count_by_modality": {
                "transcriptomics": int(transcript_entity_index.shape[0]),
                "metabolomics": int(metabolite_entity_index.shape[0]),
            },
            "benchmark_name": config["benchmark_registry"]["benchmark_name"],
            "fold_name": fold_name,
        },
        checkpoint_path,
    )
    transcript_entity_path = model_dir / "transcriptomics_entity_index.parquet"
    metabolite_entity_path = model_dir / "metabolomics_entity_index.parquet"
    transcript_entity_index.to_parquet(transcript_entity_path, index=False, engine="pyarrow")
    metabolite_entity_index.to_parquet(metabolite_entity_path, index=False, engine="pyarrow")
    validation_metrics, _ = evaluate_shared_latent_split(model, validation_payload, device)
    write_json(
        model_dir / "training_manifest.json",
        {
            "model_name": model_name,
            "status": "trained",
            "device": str(device),
            "best_epoch": best_epoch,
            "seed": model_seed,
            "validation_metrics": validation_metrics,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "transcriptomics_entity_index_path": str(transcript_entity_path.relative_to(REPO_ROOT)),
            "metabolomics_entity_index_path": str(metabolite_entity_path.relative_to(REPO_ROOT)),
            "run_metadata_path": str((model_dir / "run_metadata.json").relative_to(REPO_ROOT)),
            "training_history": training_history,
            "hyperparameters": {
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
                "learning_rate": float(model_config["learning_rate"]),
                "weight_decay": float(model_config["weight_decay"]),
                "max_epochs": int(model_config["max_epochs"]),
                "patience_epochs": int(model_config["patience_epochs"]),
                "classification_loss_weight": float(model_config["classification_loss_weight"]),
                "study_alignment_loss_weight": float(model_config["study_alignment_loss_weight"]),
                "modality_alignment_loss_weight": float(model_config["modality_alignment_loss_weight"]),
            },
        },
    )
    model_logger.artifact(
        "model_training_completed",
        artifact=checkpoint_path,
        best_epoch=best_epoch,
        seed=model_seed,
        **validation_metric_fields(validation_metrics),
    )


def train_tabular_jepa_fold(
    config: dict,
    config_path: Path,
    argv: list[str],
    fold_name: str,
    datasets: dict[str, dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]]],
    device: torch.device,
) -> None:
    model_name = "tabular_jepa_auxiliary_encoder"
    model_logger = get_pipeline_logger(scope="fold", fold=fold_name, model_name=model_name)
    train_payload = {modality: datasets[modality]["train"] for modality in ["transcriptomics", "metabolomics"]}
    validation_payload = {modality: datasets[modality]["validation"] for modality in ["transcriptomics", "metabolomics"]}
    transcript_entity_index = datasets["transcriptomics"]["train"][3]
    metabolite_entity_index = datasets["metabolomics"]["train"][3]
    model_dir = artifact_dir(config, "fold", fold_name, model_name)
    ensure_directory(model_dir)

    model_config = config["benchmark_registry"]["tabular_jepa_auxiliary"]
    model_seed = scoped_seed(int(model_config["random_seed"]), fold_name, model_name)
    seed_everything(model_seed)
    capture_run_metadata(
        model_dir,
        config_path,
        argv,
        model_seed,
        {
            "fold_name": fold_name,
            "model_name": model_name,
            "device": str(device),
            "train_shape_by_modality": {
                modality: list(datasets[modality]["train"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
            "validation_shape_by_modality": {
                modality: list(datasets[modality]["validation"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
        },
    )
    model = TabularJEPAAuxiliaryEncoder(
        max(int(transcript_entity_index.shape[0]), 1),
        max(int(metabolite_entity_index.shape[0]), 1),
        int(model_config["transcriptomics_hidden_dim"]),
        int(model_config["metabolomics_hidden_dim"]),
        int(model_config["predictor_hidden_dim"]),
        int(model_config["latent_dim"]),
        float(model_config["dropout_rate"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    best_epoch = 0
    patience = int(model_config["patience_epochs"])
    training_history: list[dict[str, object]] = []

    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        classification_loss, reconstruction_loss, _ = jepa_loss_terms(
            model,
            train_payload,
            device,
            loss_function,
            float(model_config["mask_fraction"]),
        )
        total_loss = (
            float(model_config["classification_loss_weight"]) * classification_loss
            + float(model_config["masked_prediction_loss_weight"]) * reconstruction_loss
        )
        total_loss.backward()
        optimizer.step()

        validation_metrics, validation_vectors = evaluate_tabular_jepa_split(model, validation_payload, device)
        current_metric = float(validation_metrics["auroc"]) if validation_metrics else float("-inf")
        if not validation_metrics and validation_vectors["y_true"].shape[0] > 0:
            current_metric = float(-torch.mean(total_loss.detach()).cpu().item())
        epoch_summary = {
            "epoch": epoch,
            "total_loss": float(total_loss.detach().cpu().item()),
            "classification_loss": float(classification_loss.detach().cpu().item()),
            "masked_prediction_loss": float(reconstruction_loss.detach().cpu().item()),
            "selection_metric": current_metric,
            "validation_metrics": validation_metrics,
        }
        training_history.append(epoch_summary)
        model_logger.metric(
            "training_epoch_completed",
            epoch=epoch,
            total_loss=epoch_summary["total_loss"],
            classification_loss=epoch_summary["classification_loss"],
            masked_prediction_loss=epoch_summary["masked_prediction_loss"],
            selection_metric=current_metric,
            **validation_metric_fields(validation_metrics),
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint_path = model_dir / "model_weights.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {
                "transcriptomics_input_dim": max(int(transcript_entity_index.shape[0]), 1),
                "metabolomics_input_dim": max(int(metabolite_entity_index.shape[0]), 1),
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "predictor_hidden_dim": int(model_config["predictor_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
            },
            "feature_count_by_modality": {
                "transcriptomics": int(transcript_entity_index.shape[0]),
                "metabolomics": int(metabolite_entity_index.shape[0]),
            },
            "benchmark_name": config["benchmark_registry"]["benchmark_name"],
            "fold_name": fold_name,
        },
        checkpoint_path,
    )
    transcript_entity_path = model_dir / "transcriptomics_entity_index.parquet"
    metabolite_entity_path = model_dir / "metabolomics_entity_index.parquet"
    transcript_entity_index.to_parquet(transcript_entity_path, index=False, engine="pyarrow")
    metabolite_entity_index.to_parquet(metabolite_entity_path, index=False, engine="pyarrow")
    validation_metrics, _ = evaluate_tabular_jepa_split(model, validation_payload, device)
    write_json(
        model_dir / "training_manifest.json",
        {
            "model_name": model_name,
            "status": "trained",
            "device": str(device),
            "best_epoch": best_epoch,
            "seed": model_seed,
            "validation_metrics": validation_metrics,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "transcriptomics_entity_index_path": str(transcript_entity_path.relative_to(REPO_ROOT)),
            "metabolomics_entity_index_path": str(metabolite_entity_path.relative_to(REPO_ROOT)),
            "run_metadata_path": str((model_dir / "run_metadata.json").relative_to(REPO_ROOT)),
            "training_history": training_history,
            "hyperparameters": {
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "predictor_hidden_dim": int(model_config["predictor_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
                "learning_rate": float(model_config["learning_rate"]),
                "weight_decay": float(model_config["weight_decay"]),
                "max_epochs": int(model_config["max_epochs"]),
                "patience_epochs": int(model_config["patience_epochs"]),
                "classification_loss_weight": float(model_config["classification_loss_weight"]),
                "masked_prediction_loss_weight": float(model_config["masked_prediction_loss_weight"]),
                "mask_fraction": float(model_config["mask_fraction"]),
            },
        },
    )
    model_logger.artifact(
        "model_training_completed",
        artifact=checkpoint_path,
        best_epoch=best_epoch,
        seed=model_seed,
        **validation_metric_fields(validation_metrics),
    )


def train_production_shared_latent(
    config: dict,
    config_path: Path,
    argv: list[str],
    device: torch.device,
) -> None:
    datasets = assemble_production_datasets(config)
    model_logger = get_pipeline_logger(scope="production", model_name="shared_latent_dual_encoder_classifier")
    model_dir = artifact_dir(config, "production", "production", "shared_latent_dual_encoder_classifier")
    ensure_directory(model_dir)
    model_config = config["benchmark_registry"]["shared_latent_model"]
    model_seed = scoped_seed(int(model_config["random_seed"]), "production", "shared_latent_dual_encoder_classifier")
    seed_everything(model_seed)
    capture_run_metadata(
        model_dir,
        config_path,
        argv,
        model_seed,
        {
            "fold_name": "production",
            "model_name": "shared_latent_dual_encoder_classifier",
            "device": str(device),
            "train_shape_by_modality": {
                modality: list(datasets[modality]["train"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
            "validation_shape_by_modality": {
                modality: list(datasets[modality]["validation"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
        },
    )
    model_logger.progress(
        "model_training_started",
        artifact=model_dir,
        device=str(device),
        transcriptomics_train_shape=datasets["transcriptomics"]["train"][0].shape,
        metabolomics_train_shape=datasets["metabolomics"]["train"][0].shape,
        transcriptomics_validation_shape=datasets["transcriptomics"]["validation"][0].shape,
        metabolomics_validation_shape=datasets["metabolomics"]["validation"][0].shape,
        seed=model_seed,
    )
    transcript_entity_index = datasets["transcriptomics"]["train"][3]
    metabolite_entity_index = datasets["metabolomics"]["train"][3]
    model = SharedLatentDualEncoder(
        max(int(transcript_entity_index.shape[0]), 1),
        max(int(metabolite_entity_index.shape[0]), 1),
        int(model_config["transcriptomics_hidden_dim"]),
        int(model_config["metabolomics_hidden_dim"]),
        int(model_config["latent_dim"]),
        float(model_config["dropout_rate"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    best_epoch = 0
    patience = int(model_config["patience_epochs"])
    training_history: list[dict[str, object]] = []
    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        classification_loss, payload = classification_loss_terms(model, {key: value["train"] for key, value in datasets.items()}, device, loss_function)
        study_loss = study_alignment_loss(payload, device)
        modality_loss = modality_alignment_loss(payload, device)
        total_loss = (
            float(model_config["classification_loss_weight"]) * classification_loss
            + float(model_config["study_alignment_loss_weight"]) * study_loss
            + float(model_config["modality_alignment_loss_weight"]) * modality_loss
        )
        total_loss.backward()
        optimizer.step()

        validation_metrics, validation_vectors = evaluate_shared_latent_split(
            model,
            {key: value["validation"] for key, value in datasets.items()},
            device,
        )
        current_metric = float(validation_metrics["auroc"]) if validation_metrics else float("-inf")
        if not validation_metrics and validation_vectors["y_true"].shape[0] > 0:
            current_metric = float(-torch.mean(total_loss.detach()).cpu().item())
        epoch_summary = {
            "epoch": epoch,
            "total_loss": float(total_loss.detach().cpu().item()),
            "classification_loss": float(classification_loss.detach().cpu().item()),
            "study_alignment_loss": float(study_loss.detach().cpu().item()),
            "modality_alignment_loss": float(modality_loss.detach().cpu().item()),
            "selection_metric": current_metric,
            "validation_metrics": validation_metrics,
        }
        training_history.append(epoch_summary)
        model_logger.metric(
            "training_epoch_completed",
            epoch=epoch,
            total_loss=epoch_summary["total_loss"],
            classification_loss=epoch_summary["classification_loss"],
            study_alignment_loss=epoch_summary["study_alignment_loss"],
            modality_alignment_loss=epoch_summary["modality_alignment_loss"],
            selection_metric=current_metric,
            **validation_metric_fields(validation_metrics),
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint_path = model_dir / "model_weights.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {
                "transcriptomics_input_dim": max(int(transcript_entity_index.shape[0]), 1),
                "metabolomics_input_dim": max(int(metabolite_entity_index.shape[0]), 1),
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
            },
            "feature_count_by_modality": {
                "transcriptomics": int(transcript_entity_index.shape[0]),
                "metabolomics": int(metabolite_entity_index.shape[0]),
            },
            "benchmark_name": config["benchmark_registry"]["benchmark_name"],
            "fold_name": "production",
        },
        checkpoint_path,
    )
    transcript_entity_path = model_dir / "transcriptomics_entity_index.parquet"
    metabolite_entity_path = model_dir / "metabolomics_entity_index.parquet"
    transcript_entity_index.to_parquet(transcript_entity_path, index=False, engine="pyarrow")
    metabolite_entity_index.to_parquet(metabolite_entity_path, index=False, engine="pyarrow")
    validation_metrics, _ = evaluate_shared_latent_split(
        model,
        {key: value["validation"] for key, value in datasets.items()},
        device,
    )
    write_json(
        model_dir / "training_manifest.json",
        {
            "model_name": "shared_latent_dual_encoder_classifier",
            "status": "trained",
            "device": str(device),
            "best_epoch": best_epoch,
            "seed": model_seed,
            "validation_metrics": validation_metrics,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "transcriptomics_entity_index_path": str(transcript_entity_path.relative_to(REPO_ROOT)),
            "metabolomics_entity_index_path": str(metabolite_entity_path.relative_to(REPO_ROOT)),
            "run_metadata_path": str((model_dir / "run_metadata.json").relative_to(REPO_ROOT)),
            "training_history": training_history,
            "hyperparameters": {
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
                "learning_rate": float(model_config["learning_rate"]),
                "weight_decay": float(model_config["weight_decay"]),
                "max_epochs": int(model_config["max_epochs"]),
                "patience_epochs": int(model_config["patience_epochs"]),
                "classification_loss_weight": float(model_config["classification_loss_weight"]),
                "study_alignment_loss_weight": float(model_config["study_alignment_loss_weight"]),
                "modality_alignment_loss_weight": float(model_config["modality_alignment_loss_weight"]),
            },
        },
    )
    model_logger.artifact(
        "model_training_completed",
        artifact=checkpoint_path,
        best_epoch=best_epoch,
        seed=model_seed,
        **validation_metric_fields(validation_metrics),
    )


def train_production_tabular_jepa(
    config: dict,
    config_path: Path,
    argv: list[str],
    device: torch.device,
) -> None:
    datasets = assemble_production_datasets(config)
    model_logger = get_pipeline_logger(scope="production", model_name="tabular_jepa_auxiliary_encoder")
    model_dir = artifact_dir(config, "production", "production", "tabular_jepa_auxiliary_encoder")
    ensure_directory(model_dir)
    model_config = config["benchmark_registry"]["tabular_jepa_auxiliary"]
    model_seed = scoped_seed(int(model_config["random_seed"]), "production", "tabular_jepa_auxiliary_encoder")
    seed_everything(model_seed)
    capture_run_metadata(
        model_dir,
        config_path,
        argv,
        model_seed,
        {
            "fold_name": "production",
            "model_name": "tabular_jepa_auxiliary_encoder",
            "device": str(device),
            "train_shape_by_modality": {
                modality: list(datasets[modality]["train"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
            "validation_shape_by_modality": {
                modality: list(datasets[modality]["validation"][0].shape)
                for modality in ["transcriptomics", "metabolomics"]
            },
        },
    )
    transcript_entity_index = datasets["transcriptomics"]["train"][3]
    metabolite_entity_index = datasets["metabolomics"]["train"][3]
    model = TabularJEPAAuxiliaryEncoder(
        max(int(transcript_entity_index.shape[0]), 1),
        max(int(metabolite_entity_index.shape[0]), 1),
        int(model_config["transcriptomics_hidden_dim"]),
        int(model_config["metabolomics_hidden_dim"]),
        int(model_config["predictor_hidden_dim"]),
        int(model_config["latent_dim"]),
        float(model_config["dropout_rate"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    best_epoch = 0
    patience = int(model_config["patience_epochs"])
    training_history: list[dict[str, object]] = []

    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        classification_loss, reconstruction_loss, _ = jepa_loss_terms(
            model,
            {key: value["train"] for key, value in datasets.items()},
            device,
            loss_function,
            float(model_config["mask_fraction"]),
        )
        total_loss = (
            float(model_config["classification_loss_weight"]) * classification_loss
            + float(model_config["masked_prediction_loss_weight"]) * reconstruction_loss
        )
        total_loss.backward()
        optimizer.step()

        validation_metrics, validation_vectors = evaluate_tabular_jepa_split(
            model,
            {key: value["validation"] for key, value in datasets.items()},
            device,
        )
        current_metric = float(validation_metrics["auroc"]) if validation_metrics else float("-inf")
        if not validation_metrics and validation_vectors["y_true"].shape[0] > 0:
            current_metric = float(-torch.mean(total_loss.detach()).cpu().item())
        epoch_summary = {
            "epoch": epoch,
            "total_loss": float(total_loss.detach().cpu().item()),
            "classification_loss": float(classification_loss.detach().cpu().item()),
            "masked_prediction_loss": float(reconstruction_loss.detach().cpu().item()),
            "selection_metric": current_metric,
            "validation_metrics": validation_metrics,
        }
        training_history.append(epoch_summary)
        model_logger.metric(
            "training_epoch_completed",
            epoch=epoch,
            total_loss=epoch_summary["total_loss"],
            classification_loss=epoch_summary["classification_loss"],
            masked_prediction_loss=epoch_summary["masked_prediction_loss"],
            selection_metric=current_metric,
            **validation_metric_fields(validation_metrics),
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint_path = model_dir / "model_weights.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": {
                "transcriptomics_input_dim": max(int(transcript_entity_index.shape[0]), 1),
                "metabolomics_input_dim": max(int(metabolite_entity_index.shape[0]), 1),
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "predictor_hidden_dim": int(model_config["predictor_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
            },
            "feature_count_by_modality": {
                "transcriptomics": int(transcript_entity_index.shape[0]),
                "metabolomics": int(metabolite_entity_index.shape[0]),
            },
            "benchmark_name": config["benchmark_registry"]["benchmark_name"],
            "fold_name": "production",
        },
        checkpoint_path,
    )
    transcript_entity_path = model_dir / "transcriptomics_entity_index.parquet"
    metabolite_entity_path = model_dir / "metabolomics_entity_index.parquet"
    transcript_entity_index.to_parquet(transcript_entity_path, index=False, engine="pyarrow")
    metabolite_entity_index.to_parquet(metabolite_entity_path, index=False, engine="pyarrow")
    validation_metrics, _ = evaluate_tabular_jepa_split(
        model,
        {key: value["validation"] for key, value in datasets.items()},
        device,
    )
    write_json(
        model_dir / "training_manifest.json",
        {
            "model_name": "tabular_jepa_auxiliary_encoder",
            "status": "trained",
            "device": str(device),
            "best_epoch": best_epoch,
            "seed": model_seed,
            "validation_metrics": validation_metrics,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "transcriptomics_entity_index_path": str(transcript_entity_path.relative_to(REPO_ROOT)),
            "metabolomics_entity_index_path": str(metabolite_entity_path.relative_to(REPO_ROOT)),
            "run_metadata_path": str((model_dir / "run_metadata.json").relative_to(REPO_ROOT)),
            "training_history": training_history,
            "hyperparameters": {
                "transcriptomics_hidden_dim": int(model_config["transcriptomics_hidden_dim"]),
                "metabolomics_hidden_dim": int(model_config["metabolomics_hidden_dim"]),
                "predictor_hidden_dim": int(model_config["predictor_hidden_dim"]),
                "latent_dim": int(model_config["latent_dim"]),
                "dropout_rate": float(model_config["dropout_rate"]),
                "learning_rate": float(model_config["learning_rate"]),
                "weight_decay": float(model_config["weight_decay"]),
                "max_epochs": int(model_config["max_epochs"]),
                "patience_epochs": int(model_config["patience_epochs"]),
                "classification_loss_weight": float(model_config["classification_loss_weight"]),
                "masked_prediction_loss_weight": float(model_config["masked_prediction_loss_weight"]),
                "mask_fraction": float(model_config["mask_fraction"]),
            },
        },
    )
    model_logger.artifact(
        "model_training_completed",
        artifact=checkpoint_path,
        best_epoch=best_epoch,
        seed=model_seed,
        **validation_metric_fields(validation_metrics),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    program_name = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    config["benchmark_registry"].update(benchmark_program_config(config, program_name))
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    device = resolve_device(args.device)
    config_path = Path(args.config).resolve()
    argv = sys.argv.copy()
    _, fold_rows, _ = load_benchmark_catalogs(config)
    target_fold_rows = [row for row in fold_rows if not args.fold_name or row["fold_name"] == args.fold_name]
    if target_fold_rows:
        active_benchmark_name = str(target_fold_rows[0]["benchmark_name"])
        config["benchmark_registry"]["benchmark_name"] = active_benchmark_name
        config["benchmark_registry"]["artifact_benchmark_name"] = active_benchmark_name
    pipeline_logger.progress(
        "training_dispatch_started",
        scope=args.training_scope,
        fold=args.fold_name or "all",
        device=str(device),
        target_fold_count=len(target_fold_rows),
        disease_program=program_name,
    )
    if args.training_scope in {"folds", "all"}:
        for fold_row in target_fold_rows:
            datasets = assemble_fold_datasets(config, fold_row["fold_name"])
            held_out_modality = datasets["transcriptomics"]["test"][2]["modality"].iloc[0] if not datasets["transcriptomics"]["test"][2].empty else (
                datasets["metabolomics"]["test"][2]["modality"].iloc[0] if not datasets["metabolomics"]["test"][2].empty else ""
            )
            if held_out_modality == "transcriptomics":
                train_logistic_baseline(config, config_path, argv, fold_row["fold_name"], "transcriptomics", datasets)
            if held_out_modality == "metabolomics":
                train_logistic_baseline(config, config_path, argv, fold_row["fold_name"], "metabolomics", datasets)
            train_shared_latent_fold(config, config_path, argv, fold_row["fold_name"], datasets, device)
            train_tabular_jepa_fold(config, config_path, argv, fold_row["fold_name"], datasets, device)
    if args.training_scope in {"production", "all"}:
        train_production_shared_latent(config, config_path, argv, device)
        train_production_tabular_jepa(config, config_path, argv, device)
    pipeline_logger.complete(
        "training_dispatch_completed",
        scope=args.training_scope,
        fold=args.fold_name or "all",
        device=str(device),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        scope=args.training_scope,
        fold=args.fold_name or "all",
        modality=str(device),
    )

if __name__ == "__main__":
    main()
