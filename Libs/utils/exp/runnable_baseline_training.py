from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Libs.utils.evaluation.alignment_benchmark import compute_binary_metrics
from Libs.utils.data.raw_data_registry import write_tsv


def adapter_hyperparameters(baseline_entry: dict[str, Any]) -> dict[str, Any]:
    return dict(baseline_entry.get("hyperparameters", {}))


def required_hyperparameter(baseline_entry: dict[str, Any], key: str) -> Any:
    hyperparameters = adapter_hyperparameters(baseline_entry)
    if key not in hyperparameters:
        raise KeyError(f"Baseline {baseline_entry.get('baseline_name', '')} is missing required hyperparameter: {key}")
    return hyperparameters[key]


def resolved_hidden_dim(baseline_entry: dict[str, Any], input_dim: int) -> int:
    hyperparameters = adapter_hyperparameters(baseline_entry)
    value = hyperparameters.get("hidden_dim", "")
    if str(value).strip():
        return int(value)
    return max(64, min(256, input_dim * 2))


def resolved_latent_dim(baseline_entry: dict[str, Any], transition_dim: int) -> int:
    hyperparameters = adapter_hyperparameters(baseline_entry)
    value = hyperparameters.get("latent_dim", "")
    if str(value).strip():
        return int(value)
    return max(32, min(128, transition_dim))


def load_transition_split(manifest_path: Path, split_name: str) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: list[dict[str, np.ndarray]] = []
    for row in manifest["catalog_rows"]:
        if row["split_name"] != split_name:
            continue
        arrays.append(dict(np.load(row["path"], allow_pickle=True)))
    if not arrays:
        return {
            "x": np.empty((0, 0), dtype=np.float32),
            "y_disease": np.empty((0,), dtype=np.int64),
            "y_transition": np.empty((0, 0), dtype=np.float32),
        }
    x_blocks = []
    disease_blocks = []
    transition_blocks = []
    for payload in arrays:
        x_blocks.append(
            np.concatenate(
                [
                    payload["source_patient_latent"],
                    payload["target_program_context"],
                    payload["regimen_feature_vector"],
                    payload["pkpd_reference"],
                ],
                axis=1,
            ).astype(np.float32)
        )
        disease_blocks.append(payload["disease_label"].astype(np.int64))
        transition_blocks.append(payload["proxy_transition_target"].astype(np.float32))
    return {
        "x": np.concatenate(x_blocks, axis=0),
        "y_disease": np.concatenate(disease_blocks, axis=0),
        "y_transition": np.concatenate(transition_blocks, axis=0),
    }


class TabularJEPAAdapter(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.disease_head = nn.Linear(latent_dim, 1)
        self.transition_head = nn.Linear(latent_dim, output_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(features)
        predicted = self.predictor(encoded)
        return {
            "encoded": encoded,
            "predicted": predicted,
            "disease_logit": self.disease_head(predicted).squeeze(-1),
            "transition_pred": self.transition_head(predicted),
        }


class GeneJEPAAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        group_count: int = 4,
        transformer_head_count: int = 4,
        transformer_layer_count: int = 2,
    ) -> None:
        super().__init__()
        self.group_count = int(group_count)
        token_dim = hidden_dim
        self.group_encoders = nn.ModuleList(
            [nn.Linear(input_dim // self.group_count + int(index < (input_dim % self.group_count)), token_dim) for index in range(self.group_count)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(transformer_head_count),
            batch_first=True,
            dim_feedforward=token_dim * 2,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layer_count))
        self.projector = nn.Linear(token_dim, latent_dim)
        self.mask_predictor = nn.Linear(latent_dim, latent_dim)
        self.disease_head = nn.Linear(latent_dim, 1)
        self.transition_head = nn.Linear(latent_dim, output_dim)
        sizes = [input_dim // self.group_count for _ in range(self.group_count)]
        for index in range(input_dim % self.group_count):
            sizes[index] += 1
        self.group_sizes = sizes

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        blocks = torch.split(features, self.group_sizes, dim=1)
        tokens = torch.stack([encoder(block) for encoder, block in zip(self.group_encoders, blocks)], dim=1)
        transformed = self.transformer(tokens)
        pooled = transformed.mean(dim=1)
        encoded = self.projector(pooled)
        predicted = self.mask_predictor(encoded)
        return {
            "encoded": encoded,
            "predicted": predicted,
            "disease_logit": self.disease_head(predicted).squeeze(-1),
            "transition_pred": self.transition_head(predicted),
        }


class MambaNeuralOperatorAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, group_count: int = 4) -> None:
        super().__init__()
        self.group_count = int(group_count)
        group_dim = hidden_dim
        self.group_encoders = nn.ModuleList(
            [nn.Linear(input_dim // self.group_count + int(index < (input_dim % self.group_count)), group_dim) for index in range(self.group_count)]
        )
        self.gru = nn.GRU(group_dim, group_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(group_dim)
        self.disease_head = nn.Linear(group_dim, 1)
        self.transition_head = nn.Linear(group_dim, output_dim)
        sizes = [input_dim // self.group_count for _ in range(self.group_count)]
        for index in range(input_dim % self.group_count):
            sizes[index] += 1
        self.group_sizes = sizes

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        blocks = torch.split(features, self.group_sizes, dim=1)
        sequence = torch.stack([encoder(block) for encoder, block in zip(self.group_encoders, blocks)], dim=1)
        output, hidden = self.gru(sequence)
        encoded = self.layer_norm(hidden[-1])
        return {
            "encoded": encoded,
            "predicted": encoded,
            "disease_logit": self.disease_head(encoded).squeeze(-1),
            "transition_pred": self.transition_head(encoded),
        }


class CoDANOAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, group_count: int = 4, attention_head_count: int = 4) -> None:
        super().__init__()
        self.group_count = int(group_count)
        self.group_encoders = nn.ModuleList(
            [nn.Linear(input_dim // self.group_count + int(index < (input_dim % self.group_count)), hidden_dim) for index in range(self.group_count)]
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=int(attention_head_count), batch_first=True)
        self.disease_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, output_dim)
        sizes = [input_dim // self.group_count for _ in range(self.group_count)]
        for index in range(input_dim % self.group_count):
            sizes[index] += 1
        self.group_sizes = sizes

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        blocks = torch.split(features, self.group_sizes, dim=1)
        tokens = torch.stack([encoder(block) for encoder, block in zip(self.group_encoders, blocks)], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens)
        encoded = attended.mean(dim=1)
        return {
            "encoded": encoded,
            "predicted": encoded,
            "disease_logit": self.disease_head(encoded).squeeze(-1),
            "transition_pred": self.transition_head(encoded),
        }


class DiffusionPDEAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.noise_head = nn.Linear(hidden_dim, output_dim)
        self.disease_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, noisy_target: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.condition(torch.cat([features, noisy_target], dim=1))
        return {
            "encoded": encoded,
            "predicted_noise": self.noise_head(encoded),
            "disease_logit": self.disease_head(encoded).squeeze(-1),
        }


class WDNOAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2)
        self.up = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.disease_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.project(features).unsqueeze(1)
        encoded = self.up(F.gelu(self.down(projected))).mean(dim=-1)
        encoded = encoded.squeeze(1) if encoded.dim() == 3 and encoded.shape[1] == 1 else encoded
        if encoded.dim() > 2:
            encoded = encoded.mean(dim=1)
        return {
            "encoded": encoded,
            "predicted": encoded,
            "disease_logit": self.disease_head(encoded).squeeze(-1),
            "transition_pred": self.transition_head(encoded),
        }


def build_model(baseline_entry: dict[str, Any], input_dim: int, transition_dim: int) -> nn.Module:
    adapter_name = str(baseline_entry["adapter_name"])
    hyperparameters = adapter_hyperparameters(baseline_entry)
    hidden_dim = resolved_hidden_dim(baseline_entry, input_dim)
    latent_dim = resolved_latent_dim(baseline_entry, transition_dim)
    if adapter_name == "t_jepa":
        return TabularJEPAAdapter(input_dim, latent_dim, hidden_dim, transition_dim)
    if adapter_name == "gene_jepa":
        return GeneJEPAAdapter(
            input_dim,
            latent_dim,
            hidden_dim,
            transition_dim,
            group_count=int(hyperparameters.get("group_count", 4)),
            transformer_head_count=int(hyperparameters.get("transformer_head_count", 4)),
            transformer_layer_count=int(hyperparameters.get("transformer_layer_count", 2)),
        )
    if adapter_name == "mamba_neural_operator":
        return MambaNeuralOperatorAdapter(
            input_dim,
            hidden_dim,
            transition_dim,
            group_count=int(hyperparameters.get("group_count", 4)),
        )
    if adapter_name == "coda_no":
        return CoDANOAdapter(
            input_dim,
            hidden_dim,
            transition_dim,
            group_count=int(hyperparameters.get("group_count", 4)),
            attention_head_count=int(hyperparameters.get("attention_head_count", 4)),
        )
    if adapter_name == "diffusion_pde":
        return DiffusionPDEAdapter(input_dim, hidden_dim, transition_dim)
    if adapter_name == "wdno":
        return WDNOAdapter(input_dim, hidden_dim, transition_dim)
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def train_wrapper(
    baseline_entry: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    adapter_name = str(baseline_entry["adapter_name"])
    max_epochs = int(required_hyperparameter(baseline_entry, "max_epochs"))
    batch_size = int(required_hyperparameter(baseline_entry, "batch_size"))
    learning_rate = float(required_hyperparameter(baseline_entry, "learning_rate"))
    weight_decay = float(required_hyperparameter(baseline_entry, "weight_decay"))
    hyperparameters = adapter_hyperparameters(baseline_entry)
    diffusion_noise_scale = float(hyperparameters.get("diffusion_noise_scale", 0.1))
    self_prediction_weight = float(hyperparameters.get("self_prediction_weight", 0.1))
    train_payload = load_transition_split(manifest_path, "train")
    validation_payload = load_transition_split(manifest_path, "validation")
    if train_payload["x"].shape[0] == 0:
        raise ValueError(f"No train rows in transition manifest: {manifest_path}")
    input_dim = train_payload["x"].shape[1]
    transition_dim = train_payload["y_transition"].shape[1]
    model = build_model(baseline_entry, input_dim, transition_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    disease_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    train_dataset = TensorDataset(
        torch.tensor(train_payload["x"], dtype=torch.float32),
        torch.tensor(train_payload["y_disease"], dtype=torch.float32),
        torch.tensor(train_payload["y_transition"], dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    history: list[dict[str, float]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for features, labels, targets in train_loader:
            if adapter_name == "diffusion_pde":
                noise = torch.randn_like(targets)
                noisy_target = targets + diffusion_noise_scale * noise
                outputs = model(features, noisy_target)
                transition_loss = mse_loss(outputs["predicted_noise"], noise)
                logits = outputs["disease_logit"]
            else:
                outputs = model(features)
                transition_loss = mse_loss(outputs["transition_pred"], targets)
                logits = outputs["disease_logit"]
                if "predicted" in outputs and "encoded" in outputs:
                    transition_loss = transition_loss + self_prediction_weight * mse_loss(outputs["predicted"], outputs["encoded"].detach())
            classification_loss = disease_loss(logits, labels)
            loss = classification_loss + transition_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_metrics: dict[str, float] = {}
        transition_mse = float("nan")
        if validation_payload["x"].shape[0] > 0:
            with torch.no_grad():
                features = torch.tensor(validation_payload["x"], dtype=torch.float32)
                labels = validation_payload["y_disease"].astype(np.int64)
                targets = torch.tensor(validation_payload["y_transition"], dtype=torch.float32)
                if adapter_name == "diffusion_pde":
                    noisy_target = targets
                    outputs = model(features, noisy_target)
                    logits = outputs["disease_logit"]
                    transition_pred = noisy_target - outputs["predicted_noise"]
                else:
                    outputs = model(features)
                    logits = outputs["disease_logit"]
                    transition_pred = outputs["transition_pred"]
                probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
                validation_metrics = compute_binary_metrics(labels, probabilities) if len(np.unique(labels)) >= 2 else {}
                transition_mse = float(mse_loss(transition_pred, targets).cpu())
        selection_metric = (validation_metrics.get("auroc", 0.0) if validation_metrics else 0.0) - transition_mse
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("inf"),
                "validation_transition_mse": transition_mse,
                "validation_auroc": validation_metrics.get("auroc", float("nan")),
                "selection_metric": selection_metric,
            }
        )
        if selection_metric > best_score:
            best_score = selection_metric
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(
        {
            "adapter_name": adapter_name,
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "transition_dim": transition_dim,
        },
        checkpoint_path,
    )
    manifest_out = output_dir / "training_manifest.json"
    manifest_out.write_text(
        json.dumps(
            {
                "adapter_name": adapter_name,
                "adapter_mode": str(baseline_entry.get("adapter_mode", "")),
                "checkpoint_path": str(checkpoint_path),
                "input_dim": input_dim,
                "transition_dim": transition_dim,
                "training_history": history,
                "best_selection_metric": best_score,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def evaluate_wrapper(baseline_entry: dict[str, Any], manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    adapter_name = str(baseline_entry["adapter_name"])
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(baseline_entry, int(payload["input_dim"]), int(payload["transition_dim"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    test_payload = load_transition_split(manifest_path, "test")
    if test_payload["x"].shape[0] == 0:
        raise ValueError(f"No test rows in transition manifest: {manifest_path}")
    with torch.no_grad():
        features = torch.tensor(test_payload["x"], dtype=torch.float32)
        labels = test_payload["y_disease"].astype(np.int64)
        targets = torch.tensor(test_payload["y_transition"], dtype=torch.float32)
        if adapter_name == "diffusion_pde":
            outputs = model(features, targets)
            logits = outputs["disease_logit"]
            transition_pred = targets - outputs["predicted_noise"]
        else:
            outputs = model(features)
            logits = outputs["disease_logit"]
            transition_pred = outputs["transition_pred"]
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        metrics = compute_binary_metrics(labels, probabilities) if len(np.unique(labels)) >= 2 else {}
        transition_mse = float(nn.MSELoss()(transition_pred, targets).cpu())
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_dir / "evaluation_manifest.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "adapter_name": adapter_name,
                "adapter_mode": str(baseline_entry.get("adapter_mode", "")),
                "baseline_name": manifest_payload.get("baseline_name", adapter_name),
                "disease_program": manifest_payload.get("disease_program", ""),
                "fold_name": manifest_payload.get("fold_name", ""),
                "sample_count": int(test_payload["x"].shape[0]),
                "disease_metrics": metrics,
                "transition_mse": transition_mse,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result_catalog_path = output_dir / "result_catalog.tsv"
    metric_payload = {
        "baseline_name": manifest_payload.get("baseline_name", adapter_name),
        "adapter_name": adapter_name,
        "adapter_mode": str(baseline_entry.get("adapter_mode", "")),
        "disease_program": manifest_payload.get("disease_program", ""),
        "fold_name": manifest_payload.get("fold_name", ""),
        "sample_count": int(test_payload["x"].shape[0]),
        "auroc": metrics.get("auroc", float("nan")),
        "auprc": metrics.get("auprc", float("nan")),
        "balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
        "macro_f1": metrics.get("macro_f1", float("nan")),
        "brier_score": metrics.get("brier_score", float("nan")),
        "transition_mse": transition_mse,
        "status": "evaluated",
        "artifact_path": str(checkpoint_path),
    }
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
        [metric_payload],
    )
    return evaluation_path
