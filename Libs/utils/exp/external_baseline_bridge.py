from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Libs.utils.evaluation.alignment_benchmark import compute_binary_metrics
from Libs.utils.evaluation.benchmark_datasets import assemble_dataset, benchmark_program_config, fold_entity_index, fold_sample_split
from Libs.utils.model.dynamics_fields import dynamics_root
from Libs.utils.model.dynamics_store import load_full_array
from Libs.utils.model.latent_transition_ready import build_transition_rows
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json, write_tsv


REPO_ROOT = Path(__file__).resolve().parents[3]


def baseline_torch_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    slurm_allocated_gpus = any(
        os.environ.get(name, "").strip()
        for name in ["SLURM_JOB_GPUS", "SLURM_STEP_GPUS"]
    )
    visible_gpu_allocation = visible_devices not in {"", "-1", "NoDevFiles"}
    slurm_context = any(
        os.environ.get(name, "").strip()
        for name in ["SLURM_JOB_ID", "SLURM_STEP_ID"]
    )
    if slurm_context:
        if slurm_allocated_gpus or visible_gpu_allocation:
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if visible_gpu_allocation else "cpu")


@contextmanager
def prepend_python_path(*paths: Path):
    injected = [str(path.resolve()) for path in paths]
    original = list(sys.path)
    for path in reversed(injected):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def bridge_hyperparameters(entry: dict[str, Any]) -> dict[str, Any]:
    return dict(entry.get("bridge_hyperparameters", {}))


def omics_classification_bundle_manifest(
    config_path: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
    baseline_name: str,
    required_modality: str | None = None,
) -> Path:
    config = load_config(config_path)
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    split_frame = fold_sample_split(config, fold_name, disease_program)
    modality_candidates = [required_modality] if required_modality else ["transcriptomics", "metabolomics"]
    selected_modality = ""
    for modality in modality_candidates:
        if modality is None:
            continue
        modality_frame = split_frame.loc[split_frame["modality"].astype(str).eq(modality)].copy()
        split_counts = modality_frame.groupby("split_role").size().to_dict()
        if all(int(split_counts.get(split_name, 0)) > 0 for split_name in ["train", "test"]):
            selected_modality = modality
            break
    if not selected_modality:
        raise ValueError(f"No eligible omics split found for baseline {baseline_name} in fold {fold_name}.")
    entity_index = fold_entity_index(config, fold_name, selected_modality, disease_program)
    ensure_directory(output_dir)
    manifest_rows: list[dict[str, object]] = []
    for split_name in ["train", "validation", "test"]:
        role_frame = split_frame.loc[split_frame["split_role"] == split_name].copy()
        matrix, labels, metadata = assemble_dataset(config, role_frame, selected_modality, entity_index)
        path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_{selected_modality}_{split_name}.npz"
        np.savez_compressed(
            path,
            matrix=matrix.astype(np.float32),
            labels=labels.astype(np.int64),
            sample_uid=metadata["sample_uid"].astype(str).to_numpy() if not metadata.empty else np.asarray([], dtype=str),
            study_uid=metadata["study_uid"].astype(str).to_numpy() if not metadata.empty else np.asarray([], dtype=str),
            canonical_entity_id=entity_index["canonical_entity_id"].astype(str).to_numpy() if not entity_index.empty else np.asarray([], dtype=str),
        )
        manifest_rows.append(
            {
                "split_name": split_name,
                "path": str(path),
                "sample_count": int(matrix.shape[0]),
                "feature_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            }
        )
    manifest_path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_external_bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "baseline_name": baseline_name,
            "disease_program": disease_program,
            "fold_name": fold_name,
            "bundle_type": "omics_classification",
            "modality": selected_modality,
            "manifest_rows": manifest_rows,
        },
    )
    return manifest_path


def latent_transition_classification_bundle_manifest(
    config_path: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
    baseline_name: str,
    required_modality: str,
) -> Path:
    transition_manifest_path = build_transition_rows(config_path, disease_program, fold_name, output_dir, baseline_name)
    transition_manifest = json.loads(transition_manifest_path.read_text(encoding="utf-8"))
    manifest_rows: list[dict[str, object]] = []
    feature_names: list[str] = []
    for row in transition_manifest["catalog_rows"]:
        if str(row["modality"]) != required_modality:
            continue
        payload = dict(np.load(row["path"], allow_pickle=True))
        feature_blocks = [
            ("patient_latent", payload["source_patient_latent"].astype(np.float32)),
            ("target_context", payload["target_program_context"].astype(np.float32)),
            ("regimen_context", payload["regimen_feature_vector"].astype(np.float32)),
            ("pkpd_context", payload["pkpd_reference"].astype(np.float32)),
        ]
        matrix = np.concatenate([block for _, block in feature_blocks], axis=1).astype(np.float32)
        if not feature_names:
            for prefix, block in feature_blocks:
                feature_names.extend([f"{prefix}_{index + 1}" for index in range(block.shape[1])])
        split_name = str(row["split_name"])
        path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_{required_modality}_{split_name}.npz"
        np.savez_compressed(
            path,
            matrix=matrix,
            labels=payload["disease_label"].astype(np.int64),
            sample_uid=payload["sample_uid"].astype(str),
            study_uid=payload["study_uid"].astype(str),
            canonical_entity_id=np.asarray(feature_names, dtype=str),
        )
        manifest_rows.append(
            {
                "split_name": split_name,
                "path": str(path),
                "sample_count": int(matrix.shape[0]),
                "feature_count": int(matrix.shape[1]),
            }
        )
    if not manifest_rows:
        raise ValueError(f"No latent-transition rows found for modality {required_modality} in fold {fold_name}.")
    manifest_path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_external_bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "baseline_name": baseline_name,
            "disease_program": disease_program,
            "fold_name": fold_name,
            "bundle_type": "latent_transition_context_classification",
            "modality": required_modality,
            "input_representation": "latent_transition_context",
            "manifest_rows": manifest_rows,
        },
    )
    return manifest_path


def load_transcriptomics_split(manifest_path: Path, split_name: str) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_row = next(row for row in manifest["manifest_rows"] if row["split_name"] == split_name)
    payload = dict(np.load(split_row["path"], allow_pickle=True))
    return {
        "x": payload["matrix"].astype(np.float32),
        "y": payload["labels"].astype(np.int64),
        "canonical_entity_id": payload["canonical_entity_id"].astype(str),
        "sample_uid": payload["sample_uid"].astype(str),
        "study_uid": payload["study_uid"].astype(str),
    }


def dynamics_operator_bundle_manifest(
    config_path: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
    baseline_name: str,
    sample_limit_per_split: int,
) -> Path:
    config = load_config(config_path)
    store_path = dynamics_root(config) / config["dynamics_ready_registry"]["relative_paths"]["trajectory_store"]
    trajectories = load_full_array(store_path, "trajectories").astype(np.float32)
    x = trajectories[:, :-1]
    y = trajectories[:, 1:]
    sample_count = x.shape[0] * x.shape[1]
    x = x.reshape(sample_count, x.shape[2], x.shape[3], x.shape[4])
    y = y.reshape(sample_count, y.shape[2], y.shape[3], y.shape[4])
    x = np.moveaxis(x, -1, 1).astype(np.float32)
    y = np.moveaxis(y, -1, 1).astype(np.float32)
    train_end = int(sample_count * 0.70)
    validation_end = int(sample_count * 0.85)
    split_slices = {
        "train": slice(0, train_end),
        "validation": slice(train_end, validation_end),
        "test": slice(validation_end, sample_count),
    }
    ensure_directory(output_dir)
    manifest_rows: list[dict[str, object]] = []
    for split_name, split_slice in split_slices.items():
        path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_dynamics_{split_name}.npz"
        split_x = x[split_slice]
        split_y = y[split_slice]
        if sample_limit_per_split > 0:
            split_x = split_x[:sample_limit_per_split]
            split_y = split_y[:sample_limit_per_split]
        np.savez_compressed(
            path,
            source_state=split_x,
            target_state=split_y,
        )
        manifest_rows.append(
            {
                "split_name": split_name,
                "path": str(path),
                "sample_count": int(split_x.shape[0]),
                "channel_count": int(x.shape[1]),
                "height": int(x.shape[2]),
                "width": int(x.shape[3]),
            }
        )
    manifest_path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_external_bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "baseline_name": baseline_name,
            "disease_program": disease_program,
            "fold_name": fold_name,
            "bundle_type": "dynamics_operator_forecasting",
            "manifest_rows": manifest_rows,
        },
    )
    return manifest_path


def load_dynamics_split(manifest_path: Path, split_name: str) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_row = next(row for row in manifest["manifest_rows"] if row["split_name"] == split_name)
    payload = dict(np.load(split_row["path"], allow_pickle=True))
    return {
        "x": payload["source_state"].astype(np.float32),
        "y": payload["target_state"].astype(np.float32),
    }


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def resized_dynamics_tensors(
    payload: dict[str, np.ndarray],
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(payload["x"], dtype=torch.float32)
    y = torch.tensor(payload["y"], dtype=torch.float32)
    if x.shape[2] != resolution or x.shape[3] != resolution:
        x = torch.nn.functional.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
        y = torch.nn.functional.interpolate(y, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return x, y


def dense_matrix_to_ragged_batch(matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices_blocks: list[torch.Tensor] = []
    values_blocks: list[torch.Tensor] = []
    offsets = [0]
    for row in matrix.astype(np.float32):
        active = np.flatnonzero(np.abs(row) > 1e-8)
        if active.size == 0:
            active = np.asarray([int(np.argmax(np.abs(row)))], dtype=np.int64)
        indices_blocks.append(torch.tensor(active + 1, dtype=torch.long))
        values_blocks.append(torch.tensor(row[active], dtype=torch.float32))
        offsets.append(offsets[-1] + int(active.size))
    return (
        torch.cat(indices_blocks, dim=0),
        torch.cat(values_blocks, dim=0),
        torch.tensor(offsets, dtype=torch.long),
    )


def write_external_result_catalog(
    baseline_name: str,
    adapter_name: str,
    adapter_mode: str,
    disease_program: str,
    fold_name: str,
    sample_count: int,
    metrics: dict[str, float],
    transition_mse: float,
    checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    result_catalog_path = output_dir / "result_catalog.tsv"
    write_tsv(
        result_catalog_path,
        [
            "baseline_name",
            "adapter_name",
            "adapter_mode",
            "disease_program",
            "fold_name",
            "sample_count",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "macro_f1",
            "brier_score",
            "transition_mse",
            "status",
            "artifact_path",
        ],
        [
            {
                "baseline_name": baseline_name,
                "adapter_name": adapter_name,
                "adapter_mode": adapter_mode,
                "disease_program": disease_program,
                "fold_name": fold_name,
                "sample_count": sample_count,
                "auroc": metrics.get("auroc", float("nan")),
                "auprc": metrics.get("auprc", float("nan")),
                "balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
                "macro_f1": metrics.get("macro_f1", float("nan")),
                "brier_score": metrics.get("brier_score", float("nan")),
                "transition_mse": transition_mse,
                "status": "evaluated",
                "artifact_path": str(checkpoint_path),
            }
        ],
    )
    return result_catalog_path


def train_tjepa_classifier(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_transcriptomics_split(manifest_path, "train")
    validation_payload = load_transcriptomics_split(manifest_path, "validation")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from src.encoder import Encoder
        from src.utils.train_utils import PositionalEncoding

    args = Namespace(
        n_cls_tokens=1,
        model_act_func="gelu",
    )
    device = baseline_torch_device()
    encoder = Encoder(
        idx_num_features=list(range(train_payload["x"].shape[1])),
        cardinalities=[],
        hidden_dim=int(hyperparameters["hidden_dim"]),
        num_layers=int(hyperparameters["num_layers"]),
        num_heads=int(hyperparameters["num_heads"]),
        p_dropout=float(hyperparameters["dropout"]),
        layer_norm_eps=float(hyperparameters["layer_norm_eps"]),
        gradient_clipping=float(hyperparameters["gradient_clipping"]),
        feature_type_embedding=False,
        feature_index_embedding=bool(hyperparameters["feature_index_embedding"]),
        dim_feedforward=int(hyperparameters["dim_feedforward"]),
        device=device,
        args=args,
    ).to(device)
    encoder.pe = PositionalEncoding(
        int(hyperparameters["hidden_dim"]),
        dropout=float(hyperparameters["dropout"]),
        max_len=int(train_payload["x"].shape[1]) + int(args.n_cls_tokens) + 8,
    ).to(device)
    classifier = nn.Linear(int(hyperparameters["hidden_dim"]), 1).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_dataset = TensorDataset(torch.tensor(train_payload["x"], dtype=torch.float32), torch.tensor(train_payload["y"], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=min(int(hyperparameters["batch_size"]), len(train_dataset)), shuffle=True)
    best_state = {
        "encoder": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
        "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
    }
    best_metric = float("-inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(hyperparameters["max_epochs"])):
        encoder.train()
        classifier.train()
        epoch_losses: list[float] = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            encoded = encoder(features)[:, 0, :]
            logits = classifier(encoded).squeeze(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            validation_x = torch.tensor(validation_payload["x"], dtype=torch.float32, device=device)
            validation_y = validation_payload["y"].astype(np.int64)
            validation_logits = classifier(encoder(validation_x)[:, 0, :]).squeeze(-1)
            validation_prob = torch.sigmoid(validation_logits).detach().cpu().numpy().astype(np.float64)
            validation_metrics = compute_binary_metrics(validation_y, validation_prob) if len(np.unique(validation_y)) >= 2 else {}
            selection_metric = float(validation_metrics.get("auroc", 0.0))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "validation_auroc": selection_metric})
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = {
                "encoder": {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()},
                "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
            }
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "encoder_state": best_state["encoder"],
            "classifier_state": best_state["classifier"],
            "training_history": history,
            "input_dim": int(train_payload["x"].shape[1]),
            "hidden_dim": int(hyperparameters["hidden_dim"]),
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_tjepa_classifier(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    test_payload = load_transcriptomics_split(manifest_path, "test")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from src.encoder import Encoder
        from src.utils.train_utils import PositionalEncoding

    args = Namespace(
        n_cls_tokens=1,
        model_act_func="gelu",
    )
    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder = Encoder(
        idx_num_features=list(range(checkpoint["input_dim"])),
        cardinalities=[],
        hidden_dim=int(hyperparameters["hidden_dim"]),
        num_layers=int(hyperparameters["num_layers"]),
        num_heads=int(hyperparameters["num_heads"]),
        p_dropout=float(hyperparameters["dropout"]),
        layer_norm_eps=float(hyperparameters["layer_norm_eps"]),
        gradient_clipping=float(hyperparameters["gradient_clipping"]),
        feature_type_embedding=False,
        feature_index_embedding=bool(hyperparameters["feature_index_embedding"]),
        dim_feedforward=int(hyperparameters["dim_feedforward"]),
        device=device,
        args=args,
    ).to(device)
    encoder.pe = PositionalEncoding(
        int(hyperparameters["hidden_dim"]),
        dropout=float(hyperparameters["dropout"]),
        max_len=int(checkpoint["input_dim"]) + int(args.n_cls_tokens) + 8,
    ).to(device)
    classifier = nn.Linear(int(hyperparameters["hidden_dim"]), 1).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        test_x = torch.tensor(test_payload["x"], dtype=torch.float32, device=device)
        probabilities = torch.sigmoid(classifier(encoder(test_x)[:, 0, :]).squeeze(-1)).detach().cpu().numpy().astype(np.float64)
    metrics = compute_binary_metrics(test_payload["y"], probabilities) if len(np.unique(test_payload["y"])) >= 2 else {}
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        json.loads(manifest_path.read_text(encoding="utf-8"))["disease_program"],
        json.loads(manifest_path.read_text(encoding="utf-8"))["fold_name"],
        int(test_payload["x"].shape[0]),
        metrics,
        float("nan"),
        checkpoint_path,
        output_dir,
    )


def train_genejepa_classifier(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_transcriptomics_split(manifest_path, "train")
    validation_payload = load_transcriptomics_split(manifest_path, "validation")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from genejepa.configs import ModelConfig
        from genejepa.models import GenePerceiverJEPA

    device = baseline_torch_device()
    model_config = ModelConfig(
        d=int(hyperparameters["embedding_dim"]),
        latents_L=int(hyperparameters["latents"]),
        blocks_D=int(hyperparameters["blocks"]),
        heads_h=int(hyperparameters["heads"]),
        predictor_depth=int(hyperparameters["predictor_depth"]),
        gene_vocab_size=int(train_payload["x"].shape[1] + 1),
        mask_ratio=float(hyperparameters["mask_ratio"]),
    )
    model = GenePerceiverJEPA(model_config).to(device)
    classifier = nn.Linear(int(hyperparameters["embedding_dim"]), 1).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_dataset = TensorDataset(torch.tensor(train_payload["x"], dtype=torch.float32), torch.tensor(train_payload["y"], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=min(int(hyperparameters["batch_size"]), len(train_dataset)), shuffle=True)
    best_state = {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
    }
    best_metric = float("-inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        classifier.train()
        epoch_losses: list[float] = []
        for features, labels in train_loader:
            indices, values, offsets = dense_matrix_to_ragged_batch(features.numpy())
            indices = indices.to(device)
            values = values.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            student_embedding = model.student_encoder(indices, values, offsets)
            target_embedding = model.teacher_encoder.ema_model(indices, values, offsets).detach()
            predicted_embedding = model.predictor(student_embedding)
            logits = classifier(student_embedding).squeeze(-1)
            loss = loss_fn(logits, labels) + float(hyperparameters["jepa_weight"]) * torch.nn.functional.mse_loss(predicted_embedding, target_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_teacher()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        classifier.eval()
        with torch.no_grad():
            validation_indices, validation_values, validation_offsets = dense_matrix_to_ragged_batch(validation_payload["x"])
            validation_embedding = model.get_embedding(
                validation_indices.to(device),
                validation_values.to(device),
                validation_offsets.to(device),
                use_teacher=True,
            )
            validation_prob = torch.sigmoid(classifier(validation_embedding).squeeze(-1)).detach().cpu().numpy().astype(np.float64)
            validation_metrics = compute_binary_metrics(validation_payload["y"], validation_prob) if len(np.unique(validation_payload["y"])) >= 2 else {}
            selection_metric = float(validation_metrics.get("auroc", 0.0))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "validation_auroc": selection_metric})
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
            }
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "model_state": best_state["model"],
            "classifier_state": best_state["classifier"],
            "training_history": history,
            "model_config": model_config.__dict__,
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_genejepa_classifier(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from genejepa.configs import ModelConfig
        from genejepa.models import GenePerceiverJEPA

    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GenePerceiverJEPA(ModelConfig(**checkpoint["model_config"])).to(device)
    classifier = nn.Linear(int(checkpoint["model_config"]["d"]), 1).to(device)
    model.load_state_dict(checkpoint["model_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    model.eval()
    classifier.eval()
    test_payload = load_transcriptomics_split(manifest_path, "test")
    with torch.no_grad():
        test_indices, test_values, test_offsets = dense_matrix_to_ragged_batch(test_payload["x"])
        embeddings = model.get_embedding(test_indices.to(device), test_values.to(device), test_offsets.to(device), use_teacher=True)
        probabilities = torch.sigmoid(classifier(embeddings).squeeze(-1)).detach().cpu().numpy().astype(np.float64)
    metrics = compute_binary_metrics(test_payload["y"], probabilities) if len(np.unique(test_payload["y"])) >= 2 else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        manifest["disease_program"],
        manifest["fold_name"],
        int(test_payload["x"].shape[0]),
        metrics,
        float("nan"),
        checkpoint_path,
        output_dir,
    )


def train_diffusionpde_operator(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_dynamics_split(manifest_path, "train")
    validation_payload = load_dynamics_split(manifest_path, "validation")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from training.networks import EDMPrecond

    device = baseline_torch_device()
    x_train = torch.tensor(train_payload["x"], dtype=torch.float32)
    y_train = torch.tensor(train_payload["y"], dtype=torch.float32)
    channel_count = int(x_train.shape[1])
    resolution = int(max(x_train.shape[2], x_train.shape[3]))
    if x_train.shape[2] != resolution or x_train.shape[3] != resolution:
        x_train = torch.nn.functional.interpolate(x_train, size=(resolution, resolution), mode="bilinear", align_corners=False)
        y_train = torch.nn.functional.interpolate(y_train, size=(resolution, resolution), mode="bilinear", align_corners=False)
    model = EDMPrecond(
        img_resolution=resolution,
        img_channels=channel_count,
        model_type="SongUNet",
        model_channels=int(hyperparameters["model_channels"]),
        channel_mult=[1, 1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hyperparameters["learning_rate"]), weight_decay=float(hyperparameters["weight_decay"]))
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=min(int(hyperparameters["batch_size"]), len(dataset)), shuffle=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_mse = float("inf")
    history: list[dict[str, float]] = []
    sigma = torch.full((1,), float(hyperparameters["sigma"]), dtype=torch.float32, device=device)
    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            pred = model(features, sigma.expand(features.shape[0]), force_fp32=True)
            loss = torch.nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_x = torch.tensor(validation_payload["x"], dtype=torch.float32, device=device)
            validation_y = torch.tensor(validation_payload["y"], dtype=torch.float32, device=device)
            if validation_x.shape[2] != resolution or validation_x.shape[3] != resolution:
                validation_x = torch.nn.functional.interpolate(validation_x, size=(resolution, resolution), mode="bilinear", align_corners=False)
                validation_y = torch.nn.functional.interpolate(validation_y, size=(resolution, resolution), mode="bilinear", align_corners=False)
            validation_pred = model(validation_x, sigma.expand(validation_x.shape[0]), force_fp32=True)
            validation_mse = float(torch.nn.functional.mse_loss(validation_pred, validation_y).detach().cpu())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(epoch_losses)), "validation_mse": validation_mse})
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "model_state": best_state,
            "training_history": history,
            "channel_count": channel_count,
            "resolution": resolution,
            "model_channels": int(hyperparameters["model_channels"]),
            "sigma": float(hyperparameters["sigma"]),
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_diffusionpde_operator(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from training.networks import EDMPrecond

    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = EDMPrecond(
        img_resolution=int(checkpoint["resolution"]),
        img_channels=int(checkpoint["channel_count"]),
        model_type="SongUNet",
        model_channels=int(checkpoint["model_channels"]),
        channel_mult=[1, 1],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_payload = load_dynamics_split(manifest_path, "test")
    test_x = torch.tensor(test_payload["x"], dtype=torch.float32, device=device)
    test_y = torch.tensor(test_payload["y"], dtype=torch.float32, device=device)
    if test_x.shape[2] != int(checkpoint["resolution"]) or test_x.shape[3] != int(checkpoint["resolution"]):
        test_x = torch.nn.functional.interpolate(test_x, size=(int(checkpoint["resolution"]), int(checkpoint["resolution"])), mode="bilinear", align_corners=False)
        test_y = torch.nn.functional.interpolate(test_y, size=(int(checkpoint["resolution"]), int(checkpoint["resolution"])), mode="bilinear", align_corners=False)
    sigma = torch.full((test_x.shape[0],), float(checkpoint["sigma"]), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(test_x, sigma, force_fp32=True)
        transition_mse = float(torch.nn.functional.mse_loss(pred, test_y).detach().cpu())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        manifest["disease_program"],
        manifest["fold_name"],
        int(test_x.shape[0]),
        {},
        transition_mse,
        checkpoint_path,
        output_dir,
    )


def train_pdediff_operator(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_dynamics_split(manifest_path, "train")
    validation_payload = load_dynamics_split(manifest_path, "validation")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from pdediff.nn.twod_unet import Unet

    device = baseline_torch_device()
    x_train = torch.tensor(train_payload["x"], dtype=torch.float32)
    y_train = torch.tensor(train_payload["y"], dtype=torch.float32)
    channel_count = int(x_train.shape[1])
    model = Unet(
        time_history=1,
        time_future=1,
        n_input_scalar_components=channel_count,
        n_input_vector_components=0,
        n_output_scalar_components=channel_count,
        n_output_vector_components=0,
        hidden_channels=int(hyperparameters["hidden_channels"]),
        activation="GELU",
        norm=False,
        ch_mults=(1, 1),
        is_attn=(False, False),
        mid_attn=False,
        n_blocks=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hyperparameters["learning_rate"]), weight_decay=float(hyperparameters["weight_decay"]))
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=min(int(hyperparameters["batch_size"]), len(dataset)), shuffle=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_mse = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            prediction = model(features.unsqueeze(1), torch.zeros((features.shape[0],), dtype=torch.float32, device=device))
            loss = torch.nn.functional.mse_loss(prediction.squeeze(1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_x = torch.tensor(validation_payload["x"], dtype=torch.float32, device=device)
            validation_y = torch.tensor(validation_payload["y"], dtype=torch.float32, device=device)
            validation_pred = model(validation_x.unsqueeze(1), torch.zeros((validation_x.shape[0],), dtype=torch.float32, device=device))
            validation_mse = float(torch.nn.functional.mse_loss(validation_pred.squeeze(1), validation_y).detach().cpu())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(epoch_losses)), "validation_mse": validation_mse})
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "model_state": best_state,
            "training_history": history,
            "channel_count": channel_count,
            "hidden_channels": int(hyperparameters["hidden_channels"]),
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_pdediff_operator(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from pdediff.nn.twod_unet import Unet

    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Unet(
        time_history=1,
        time_future=1,
        n_input_scalar_components=int(checkpoint["channel_count"]),
        n_input_vector_components=0,
        n_output_scalar_components=int(checkpoint["channel_count"]),
        n_output_vector_components=0,
        hidden_channels=int(checkpoint["hidden_channels"]),
        activation="GELU",
        norm=False,
        ch_mults=(1, 1),
        is_attn=(False, False),
        mid_attn=False,
        n_blocks=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_payload = load_dynamics_split(manifest_path, "test")
    test_x = torch.tensor(test_payload["x"], dtype=torch.float32, device=device)
    test_y = torch.tensor(test_payload["y"], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(test_x.unsqueeze(1), torch.zeros((test_x.shape[0],), dtype=torch.float32, device=device))
        transition_mse = float(torch.nn.functional.mse_loss(pred.squeeze(1), test_y).detach().cpu())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        manifest["disease_program"],
        manifest["fold_name"],
        int(test_x.shape[0]),
        {},
        transition_mse,
        checkpoint_path,
        output_dir,
    )


def train_codano_operator(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_dynamics_split(manifest_path, "train")
    validation_payload = load_dynamics_split(manifest_path, "validation")
    from neuralop.models import CODANO

    device = baseline_torch_device()
    channel_count = int(train_payload["x"].shape[1])
    n_modes = [[int(value) for value in row] for row in hyperparameters["n_modes"]]
    n_heads = [int(value) for value in hyperparameters["n_heads"]]
    n_layers = len(n_modes)
    model = CODANO(
        output_variable_codimension=1,
        lifting_channels=int(hyperparameters["lifting_token_codim"]),
        hidden_variable_codimension=int(hyperparameters["hidden_token_codim"]),
        projection_channels=int(hyperparameters["lifting_token_codim"]),
        n_layers=n_layers,
        n_modes=n_modes,
        n_heads=n_heads,
        per_layer_scaling_factors=[[1.0, 1.0] for _ in range(n_layers)],
        attention_scaling_factors=[1.0 for _ in range(n_layers)],
        variable_ids=list(range(channel_count)),
        attention_token_dim=1,
        domain_padding=0.0,
        enable_cls_token=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    train_x = torch.tensor(train_payload["x"], dtype=torch.float32)
    train_y = torch.tensor(train_payload["y"], dtype=torch.float32)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=min(int(hyperparameters["batch_size"]), len(dataset)), shuffle=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_mse = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            prediction = model(features)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_x = torch.tensor(validation_payload["x"], dtype=torch.float32, device=device)
            validation_y = torch.tensor(validation_payload["y"], dtype=torch.float32, device=device)
            validation_pred = model(validation_x)
            validation_mse = float(torch.nn.functional.mse_loss(validation_pred, validation_y).detach().cpu())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(epoch_losses)), "validation_mse": validation_mse})
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "model_state": best_state,
            "channel_count": channel_count,
            "hidden_token_codim": int(hyperparameters["hidden_token_codim"]),
            "lifting_token_codim": int(hyperparameters["lifting_token_codim"]),
            "n_modes": n_modes,
            "n_heads": n_heads,
            "training_history": history,
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_codano_operator(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    from neuralop.models import CODANO

    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CODANO(
        output_variable_codimension=1,
        lifting_channels=int(checkpoint["lifting_token_codim"]),
        hidden_variable_codimension=int(checkpoint["hidden_token_codim"]),
        projection_channels=int(checkpoint["lifting_token_codim"]),
        n_layers=len(checkpoint["n_modes"]),
        n_modes=checkpoint["n_modes"],
        n_heads=checkpoint["n_heads"],
        per_layer_scaling_factors=[[1.0, 1.0] for _ in range(len(checkpoint["n_modes"]))],
        attention_scaling_factors=[1.0 for _ in range(len(checkpoint["n_modes"]))],
        variable_ids=list(range(int(checkpoint["channel_count"]))),
        attention_token_dim=1,
        domain_padding=0.0,
        enable_cls_token=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_payload = load_dynamics_split(manifest_path, "test")
    test_x = torch.tensor(test_payload["x"], dtype=torch.float32, device=device)
    test_y = torch.tensor(test_payload["y"], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(test_x)
        transition_mse = float(torch.nn.functional.mse_loss(pred, test_y).detach().cpu())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        manifest["disease_program"],
        manifest["fold_name"],
        int(test_x.shape[0]),
        {},
        transition_mse,
        checkpoint_path,
        output_dir,
    )


def train_wdno_operator(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_dynamics_split(manifest_path, "train")
    validation_payload = load_dynamics_split(manifest_path, "validation")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root, repo_root / "burgers"):
        from ddpm_burgers.unet import Unet2D

    device = baseline_torch_device()
    channel_count = int(train_payload["x"].shape[1])
    resolution = next_power_of_two(max(int(train_payload["x"].shape[2]), int(train_payload["x"].shape[3]), 8))
    train_x, train_y = resized_dynamics_tensors(train_payload, resolution)
    model = Unet2D(
        dim=int(hyperparameters["dim"]),
        out_dim=channel_count,
        dim_mults=tuple(int(value) for value in hyperparameters["dim_muls"]),
        channels=channel_count,
        resnet_block_groups=1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=min(int(hyperparameters["batch_size"]), len(dataset)), shuffle=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_mse = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            prediction = model(features, torch.zeros((features.shape[0],), dtype=torch.float32, device=device))
            loss = torch.nn.functional.mse_loss(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_x, validation_y = resized_dynamics_tensors(validation_payload, resolution)
            validation_pred = model(
                validation_x.to(device),
                torch.zeros((validation_x.shape[0],), dtype=torch.float32, device=device),
            )
            validation_mse = float(torch.nn.functional.mse_loss(validation_pred, validation_y.to(device)).detach().cpu())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(epoch_losses)), "validation_mse": validation_mse})
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "model_state": best_state,
            "channel_count": channel_count,
            "resolution": resolution,
            "dim": int(hyperparameters["dim"]),
            "dim_muls": [int(value) for value in hyperparameters["dim_muls"]],
            "training_history": history,
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_wdno_operator(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root, repo_root / "burgers"):
        from ddpm_burgers.unet import Unet2D

    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Unet2D(
        dim=int(checkpoint["dim"]),
        out_dim=int(checkpoint["channel_count"]),
        dim_mults=tuple(int(value) for value in checkpoint["dim_muls"]),
        channels=int(checkpoint["channel_count"]),
        resnet_block_groups=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_payload = load_dynamics_split(manifest_path, "test")
    test_x, test_y = resized_dynamics_tensors(test_payload, int(checkpoint["resolution"]))
    with torch.no_grad():
        pred = model(
            test_x.to(device),
            torch.zeros((test_x.shape[0],), dtype=torch.float32, device=device),
        )
        transition_mse = float(torch.nn.functional.mse_loss(pred, test_y.to(device)).detach().cpu())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        manifest["disease_program"],
        manifest["fold_name"],
        int(test_x.shape[0]),
        {},
        transition_mse,
        checkpoint_path,
        output_dir,
    )


def train_mamba_operator(entry: dict[str, Any], manifest_path: Path, output_dir: Path) -> Path:
    hyperparameters = bridge_hyperparameters(entry)
    train_payload = load_dynamics_split(manifest_path, "train")
    validation_payload = load_dynamics_split(manifest_path, "validation")
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from MambaNOModule import MambaNO

    device = baseline_torch_device()
    channel_count = int(train_payload["x"].shape[1])
    resolution = max(int(hyperparameters["in_size"]), next_power_of_two(max(int(train_payload["x"].shape[2]), int(train_payload["x"].shape[3]))))
    train_x, train_y = resized_dynamics_tensors(train_payload, resolution)
    model = MambaNO(
        in_size=resolution,
        in_channels=channel_count,
        out_channels=channel_count,
        embed_dim=int(hyperparameters["embed_dim"]),
        depths=[int(value) for value in hyperparameters["depths"]],
        N_res_neck=int(hyperparameters["neck_depth"]),
        d_state=int(hyperparameters["d_state"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=min(int(hyperparameters["batch_size"]), len(dataset)), shuffle=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_mse = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            prediction = model(features)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_x, validation_y = resized_dynamics_tensors(validation_payload, resolution)
            validation_pred = model(validation_x.to(device))
            validation_mse = float(torch.nn.functional.mse_loss(validation_pred, validation_y.to(device)).detach().cpu())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(epoch_losses)), "validation_mse": validation_mse})
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "baseline_name": entry["baseline_name"],
            "adapter_name": entry["adapter_name"],
            "model_state": best_state,
            "channel_count": channel_count,
            "resolution": resolution,
            "embed_dim": int(hyperparameters["embed_dim"]),
            "depths": [int(value) for value in hyperparameters["depths"]],
            "neck_depth": int(hyperparameters["neck_depth"]),
            "d_state": int(hyperparameters["d_state"]),
            "training_history": history,
            "manifest_path": str(manifest_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_mamba_operator(entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    repo_root = REPO_ROOT / str(entry["source_repo_path"])
    with prepend_python_path(repo_root):
        from MambaNOModule import MambaNO

    device = baseline_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = MambaNO(
        in_size=int(checkpoint["resolution"]),
        in_channels=int(checkpoint["channel_count"]),
        out_channels=int(checkpoint["channel_count"]),
        embed_dim=int(checkpoint["embed_dim"]),
        depths=[int(value) for value in checkpoint["depths"]],
        N_res_neck=int(checkpoint["neck_depth"]),
        d_state=int(checkpoint["d_state"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_payload = load_dynamics_split(manifest_path, "test")
    test_x, test_y = resized_dynamics_tensors(test_payload, int(checkpoint["resolution"]))
    with torch.no_grad():
        pred = model(test_x.to(device))
        transition_mse = float(torch.nn.functional.mse_loss(pred, test_y.to(device)).detach().cpu())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_external_result_catalog(
        entry["baseline_name"],
        entry["adapter_name"],
        entry["adapter_mode"],
        manifest["disease_program"],
        manifest["fold_name"],
        int(test_x.shape[0]),
        {},
        transition_mse,
        checkpoint_path,
        output_dir,
    )


def prepare_external_bridge_inputs(
    config_path: str,
    baseline_name: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
) -> Path:
    config = load_config(config_path)
    entry = next(row for row in config["paper_experiments"]["runnable_baselines"] if row["baseline_name"] == baseline_name)
    input_representation = str(bridge_hyperparameters(entry).get("input_representation", "")).strip()
    required_modality = str(entry.get("fold_contract", {}).get("required_test_modality", "transcriptomics")).strip()
    if input_representation == "latent_transition_context":
        return latent_transition_classification_bundle_manifest(config_path, disease_program, fold_name, output_dir, baseline_name, required_modality)
    if baseline_name == "gene_jepa_adapter":
        return omics_classification_bundle_manifest(config_path, disease_program, fold_name, output_dir, baseline_name, required_modality="transcriptomics")
    if baseline_name == "t_jepa_adapter":
        return omics_classification_bundle_manifest(config_path, disease_program, fold_name, output_dir, baseline_name)
    if baseline_name in {
        "diffusion_pde_adapter",
        "pdediff_adapter",
        "coda_no_adapter",
        "wdno_adapter",
        "mamba_neural_operator_adapter",
    }:
        return dynamics_operator_bundle_manifest(
            config_path,
            disease_program,
            fold_name,
            output_dir,
            baseline_name,
            int(bridge_hyperparameters(entry).get("sample_limit_per_split", 0)),
        )
    raise NotImplementedError(f"External bridge input preparation is not implemented yet for baseline {baseline_name}")


def train_external_bridge(
    entry: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    if entry["baseline_name"] == "t_jepa_adapter":
        return train_tjepa_classifier(entry, manifest_path, output_dir)
    if entry["baseline_name"] == "gene_jepa_adapter":
        return train_genejepa_classifier(entry, manifest_path, output_dir)
    if entry["baseline_name"] == "coda_no_adapter":
        return train_codano_operator(entry, manifest_path, output_dir)
    if entry["baseline_name"] == "wdno_adapter":
        return train_wdno_operator(entry, manifest_path, output_dir)
    if entry["baseline_name"] == "mamba_neural_operator_adapter":
        return train_mamba_operator(entry, manifest_path, output_dir)
    if entry["baseline_name"] == "diffusion_pde_adapter":
        return train_diffusionpde_operator(entry, manifest_path, output_dir)
    if entry["baseline_name"] == "pdediff_adapter":
        return train_pdediff_operator(entry, manifest_path, output_dir)
    raise NotImplementedError(f"External bridge is not implemented yet for baseline {entry['baseline_name']}")


def evaluate_external_bridge(
    entry: dict[str, Any],
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> Path:
    if entry["baseline_name"] == "t_jepa_adapter":
        return evaluate_tjepa_classifier(entry, manifest_path, checkpoint_path, output_dir)
    if entry["baseline_name"] == "gene_jepa_adapter":
        return evaluate_genejepa_classifier(entry, manifest_path, checkpoint_path, output_dir)
    if entry["baseline_name"] == "coda_no_adapter":
        return evaluate_codano_operator(entry, manifest_path, checkpoint_path, output_dir)
    if entry["baseline_name"] == "wdno_adapter":
        return evaluate_wdno_operator(entry, manifest_path, checkpoint_path, output_dir)
    if entry["baseline_name"] == "mamba_neural_operator_adapter":
        return evaluate_mamba_operator(entry, manifest_path, checkpoint_path, output_dir)
    if entry["baseline_name"] == "diffusion_pde_adapter":
        return evaluate_diffusionpde_operator(entry, manifest_path, checkpoint_path, output_dir)
    if entry["baseline_name"] == "pdediff_adapter":
        return evaluate_pdediff_operator(entry, manifest_path, checkpoint_path, output_dir)
    raise NotImplementedError(f"External bridge is not implemented yet for baseline {entry['baseline_name']}")
