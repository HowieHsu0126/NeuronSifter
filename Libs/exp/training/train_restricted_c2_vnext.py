"""Train the Round-23 NeuronSifter world-action model on restricted C2-S data.

Only train and validation rows from the producer-qualified current-clamp domain
are readable through this entrypoint.  The final holdout is identified from the
strict split manifest and rejected before any trajectory tensor is loaded.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
from statistics import NormalDist
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.pretrain_microenvironment_world_model import (  # noqa: E402
    MicroenvironmentPretrainingModel,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json  # noqa: E402
from Libs.utils.model.dynamics_store import load_zarr_rows  # noqa: E402
from Libs.utils.model.dynamics_fields import compartment_id_list  # noqa: E402


MODEL_SCALE_NAMES = ("Debug", "Student", "Large")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else REPO_ROOT / path
    return resolved.resolve()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C2-S CUDA device was requested but is unavailable")
    return torch.device(requested)


def _artifact_row_index(artifact_uri: str) -> tuple[Path, int]:
    if "#simulation=" not in artifact_uri:
        raise ValueError(f"C2-S trajectory URI lacks a simulation row binding: {artifact_uri}")
    store_value, index_value = artifact_uri.rsplit("#simulation=", 1)
    try:
        row_index = int(index_value)
    except ValueError as exc:
        raise ValueError(f"C2-S trajectory URI has an invalid row binding: {artifact_uri}") from exc
    if row_index < 0:
        raise ValueError(f"C2-S trajectory URI has a negative row binding: {artifact_uri}")
    return _repo_path(store_value), row_index


def _require_unique(rows: list[dict[str, str]], field: str, label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(field, "")).strip()
        if not key:
            raise ValueError(f"{label} contains an empty {field}")
        if key in output:
            raise ValueError(f"{label} contains duplicate {field}: {key}")
        output[key] = row
    return output


def _structured_action_features(
    action_rows: list[dict[str, str]],
    feature_layout: Mapping[str, object],
    *,
    train_action_uids: set[str],
) -> np.ndarray:
    indices = {str(name): int(index) for name, index in feature_layout.items()}
    width = max(indices.values()) + 1
    train_amplitudes = [
        abs(float(row["dose_value"]))
        for row in action_rows
        if str(row["action_uid"]) in train_action_uids and abs(float(row["dose_value"])) > 0.0
    ]
    amplitude_scale = max(train_amplitudes, default=1.0)
    train_durations = [
        float(row["duration_ms"])
        for row in action_rows
        if str(row["action_uid"]) in train_action_uids
        and float(row["duration_ms"]) > 0.0
    ]
    duration_scale = max(train_durations, default=1.0)
    features = np.zeros((len(action_rows), width), dtype=np.float32)
    for row_index, row in enumerate(action_rows):
        amplitude = float(row["dose_value"])
        duration = float(row["duration_ms"])
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("C2-S structured actions require a finite positive duration")
        is_intervention = str(row["action_role"]) == "intervention"
        values = {
            "aggregate_target_evidence_score_index": 1.0,
            "aggregate_affinity_score_index": 0.0,
            "clue_reversal_proxy_score_index": 0.0,
            "pk_feasibility_score_index": 1.0,
            "dose_multiplier_index": amplitude / amplitude_scale,
            "regimen_size_index": 1.0 if is_intervention else 0.0,
            "route_code_index": 1.0,
            "schedule_code_index": duration / duration_scale,
            "target_count_index": 1.0,
            "compound_count_index": 1.0 if is_intervention else 0.0,
            "single_compound_regimen_indicator_index": 1.0 if is_intervention else 0.0,
        }
        for name, value in values.items():
            if name not in indices:
                raise ValueError(f"C2-S action feature layout is missing {name}")
            features[row_index, indices[name]] = float(value)
    if not np.isfinite(features).all():
        raise ValueError("C2-S structured action features contain NaN or Inf")
    return features


def load_restricted_c2_development_data(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["neuronsifter_vnext"]["training"]["c2s_development"]
    readiness_path = _repo_path(str(contract["readiness_path"]))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("status") != "simulator_grounded_c2_restricted":
        raise ValueError("C2-S readiness status is not simulator_grounded_c2_restricted")
    if readiness.get("selected_domain") != contract["source_domain"]:
        raise ValueError("C2-S readiness domain does not match the Round-23 contract")
    if readiness.get("qualified_producer") is not True or readiness.get("final_holdout_unseen") is not True:
        raise ValueError("C2-S producer qualification or final-holdout boundary is not satisfied")

    action_path = _repo_path(str(contract["action_catalog_path"]))
    trajectory_path = _repo_path(str(contract["trajectory_manifest_path"]))
    split_path = _repo_path(str(contract["strict_split_manifest_path"]))
    actions = _require_unique(read_tsv(action_path), "action_uid", "C2-S action catalog")
    trajectories = _require_unique(read_tsv(trajectory_path), "action_uid", "C2-S trajectory manifest")
    splits = _require_unique(read_tsv(split_path), "action_uid", "C2-S strict split manifest")
    if set(actions) != set(trajectories) or set(actions) != set(splits):
        raise ValueError("C2-S action, trajectory, and strict-split manifests do not join one-to-one")

    allowed_roles = {str(value) for value in contract["allowed_split_roles"]}
    protected_role = str(contract["protected_split_role"])
    if protected_role in allowed_roles:
        raise ValueError("C2-S protected split is present in allowed_split_roles")
    records: list[dict[str, Any]] = []
    protected_indices: list[int] = []
    store_paths: set[Path] = set()
    for action_uid in actions:
        action = actions[action_uid]
        trajectory = trajectories[action_uid]
        split = splits[action_uid]
        role = str(split["split_role"])
        if role != str(action["split_role"]):
            raise ValueError(f"C2-S split-role mismatch for {action_uid}")
        store_path, simulation_index = _artifact_row_index(str(trajectory["artifact_uri"]))
        store_paths.add(store_path)
        if role == protected_role:
            protected_indices.append(simulation_index)
            continue
        if role not in allowed_roles:
            raise ValueError(f"C2-S encountered undeclared split role {role}")
        records.append(
            {
                "action_uid": action_uid,
                "split_role": role,
                "simulation_index": simulation_index,
                "store_path": store_path,
                "artifact_hash": str(trajectory["artifact_hash"]),
                "action": action,
            }
        )
    if len(store_paths) != 1:
        raise ValueError("C2-S manifests reference multiple trajectory stores")
    store_path = next(iter(store_paths))
    producer = readiness["producer"]
    if _repo_path(str(producer["trajectory_store"])) != store_path:
        raise ValueError("C2-S readiness and trajectory manifest reference different stores")
    declared_hashes = {str(record["artifact_hash"]) for record in records}
    if declared_hashes != {str(producer["trajectory_store_sha256"])}:
        raise ValueError("C2-S development rows drift from the qualified store identity")

    records.sort(key=lambda row: int(row["simulation_index"]))
    accessed_indices = np.asarray([int(row["simulation_index"]) for row in records], dtype=np.int64)
    if set(accessed_indices.tolist()) & set(protected_indices):
        raise ValueError("C2-S development selection overlaps protected final-holdout rows")
    read_store_path = store_path
    staged_store_value = os.environ.get("NRNSIFTER_C2S_STAGED_STORE", "").strip()
    if staged_store_value:
        staged_store = Path(staged_store_value).resolve()
        if not (staged_store / ".zattrs").is_file():
            raise FileNotFoundError(f"C2-S staged store is incomplete: {staged_store}")
        read_store_path = staged_store
    trajectory_array = load_zarr_rows(read_store_path, str(contract["trajectory_array_name"]), accessed_indices)
    uncertainty_array = load_zarr_rows(read_store_path, str(contract["uncertainty_array_name"]), accessed_indices)
    if trajectory_array.ndim != 5 or trajectory_array.shape[0] != len(records):
        raise ValueError(f"C2-S trajectory tensor has invalid shape {trajectory_array.shape}")
    if uncertainty_array.shape != (len(records), trajectory_array.shape[-1]):
        raise ValueError(f"C2-S uncertainty tensor has invalid shape {uncertainty_array.shape}")
    if not np.isfinite(trajectory_array).all() or not np.isfinite(uncertainty_array).all():
        raise ValueError("C2-S development tensors contain NaN or Inf")

    train_action_uids = {record["action_uid"] for record in records if record["split_role"] == "train"}
    feature_layout = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["regimen_feature_layout"]
    action_features = _structured_action_features(
        [record["action"] for record in records],
        feature_layout,
        train_action_uids=train_action_uids,
    )
    train_positions = np.asarray([index for index, record in enumerate(records) if record["split_role"] == "train"], dtype=np.int64)
    validation_positions = np.asarray([index for index, record in enumerate(records) if record["split_role"] == "validation"], dtype=np.int64)
    if len(train_positions) == 0 or len(validation_positions) == 0:
        raise ValueError("C2-S requires non-empty train and validation partitions")
    return {
        "records": records,
        "trajectories": trajectory_array.astype(np.float32, copy=False),
        "uncertainties": uncertainty_array.astype(np.float32, copy=False),
        "action_features": action_features,
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "accessed_simulation_indices": accessed_indices.tolist(),
        "protected_simulation_indices": sorted(protected_indices),
        "store_path": store_path,
        "store_read_path": read_store_path,
        "node_local_staging_used": read_store_path != store_path,
        "input_identity": {
            "readiness_sha256": _sha256_file(readiness_path),
            "action_catalog_sha256": _sha256_file(action_path),
            "trajectory_manifest_sha256": _sha256_file(trajectory_path),
            "strict_split_manifest_sha256": _sha256_file(split_path),
            "trajectory_store_declared_sha256": str(producer["trajectory_store_sha256"]),
            "producer_execution_uid": str(producer["execution_uid"]),
        },
    }


def _profile_config(config: Mapping[str, Any], scale: str) -> dict[str, Any]:
    effective = copy.deepcopy(dict(config))
    profile = effective["neuronsifter_vnext"]["training"]["c2s_development"]["profiles"][scale]
    backbone = effective["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    if backbone["kind"] != "neuronsifter_world_action":
        raise ValueError("C2-S vNext development requires neuronsifter_world_action as the mainline")
    backbone["operator_hidden_dim"] = int(profile["operator_hidden_dim"])
    backbone["operator_depth"] = int(profile["operator_depth"])
    backbone["attention_head_count"] = int(profile["attention_head_count"])
    backbone["mechanism_state_token_count"] = int(profile["mechanism_state_token_count"])
    backbone["graph_spectral_tokenizer"]["graph_mode_count"] = int(profile["graph_mode_count"])
    backbone["temporal_wavelet_tokenizer"]["wavelet_level_count"] = int(profile["wavelet_level_count"])
    sampler = backbone["diffusion_sampler"]
    sampler["training_sample_count"] = int(profile["training_sample_count"])
    sampler["training_denoising_step_count"] = int(profile["training_denoising_step_count"])
    sampler["sample_count"] = int(profile["evaluation_sample_count"])
    sampler["denoising_step_count"] = int(profile["evaluation_denoising_step_count"])
    sampler["sample_chunk_size"] = min(int(sampler["sample_chunk_size"]), int(profile["evaluation_sample_count"]))
    quantized = backbone.get("quantized_mddm_rollout")
    if isinstance(quantized, dict):
        quantized["enabled"] = False
    return effective


class RestrictedC2WorldActionModel(nn.Module):
    """Action-conditioning adapter around the existing NeuronSifter pretrainer."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.backbone = MicroenvironmentPretrainingModel(dict(config), use_smoke_dimensions=False)
        if self.backbone.operator_kind != "neuronsifter_world_action":
            raise ValueError("RestrictedC2WorldActionModel requires neuronsifter_world_action")
        feature_layout = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["regimen_feature_layout"]
        action_dim = max(int(value) for value in feature_layout.values()) + 1
        species_count = len(config["dynamics_ready_registry"]["species"])
        self.action_to_species = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, species_count),
        )
        self.region_count = len(config["spatial_atlas_registry"]["regions"])
        self.compartment_count = len(compartment_id_list(dict(config)))
        self.species_count = species_count
        node_coordinates = []
        for region_index in range(self.region_count):
            for compartment_index in range(self.compartment_count):
                node_coordinates.append(
                    [
                        region_index / max(self.region_count - 1, 1),
                        compartment_index / max(self.compartment_count - 1, 1),
                    ]
                )
        self.register_buffer(
            "node_coordinates",
            torch.tensor(node_coordinates, dtype=torch.float32).view(1, 1, -1, 2),
            persistent=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        *,
        scientific_posterior_species_condition: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if state.ndim != 4:
            raise ValueError(f"C2-S state must be batch x time x node x species, got {tuple(state.shape)}")
        if state.shape[-1] != self.species_count:
            raise ValueError("C2-S state species dimension does not match the configured state grid")
        if action_features.ndim != 2 or action_features.shape[0] != state.shape[0]:
            raise ValueError("C2-S action features must be batch-aligned rank-two tensors")
        hidden = self.backbone.state_encoder(state)
        action_state = self.action_to_species(action_features).view(state.shape[0], 1, 1, self.species_count)
        if scientific_posterior_species_condition is None:
            posterior_state = torch.zeros_like(action_state)
        else:
            posterior_state = scientific_posterior_species_condition
            expected_shape = (state.shape[0], 1, 1, self.species_count)
            if posterior_state.shape != expected_shape:
                raise ValueError(
                    "C2-S scientific-posterior species condition must have shape "
                    f"{expected_shape}, got {tuple(posterior_state.shape)}"
                )
            if not torch.is_floating_point(posterior_state) or not torch.isfinite(
                posterior_state
            ).all():
                raise ValueError(
                    "C2-S scientific-posterior species condition must be finite and floating"
                )
            posterior_state = posterior_state.to(
                device=state.device,
                dtype=state.dtype,
            )
        condition_source = (
            state.mean(dim=-2, keepdim=True).expand_as(state)
            + action_state
            + posterior_state
        )
        conditioning = self.backbone.condition_encoder(condition_source)
        coordinates = self.node_coordinates.expand(state.shape[0], state.shape[1], -1, -1)
        encoded, router_diagnostics = self.backbone.operator(
            hidden,
            conditioning,
            coordinates,
            self.backbone.node_adjacency.to(device=hidden.device, dtype=hidden.dtype),
        )
        prior_prediction = self.backbone.state_decoder(encoded)
        prior_trajectory = prior_prediction.reshape(
            state.shape[0],
            1,
            state.shape[1],
            self.region_count,
            self.compartment_count,
            self.species_count,
        )
        stochastic = self.backbone.operator.trajectory_diffusion_summary(
            prior_trajectory,
            self.training,
            self.backbone.node_adjacency.to(device=hidden.device, dtype=hidden.dtype),
        )
        return {
            "trajectory_mean": stochastic["trajectory_mean"].squeeze(1),
            "trajectory_std": stochastic["trajectory_std"].squeeze(1),
            "trajectory_quantiles": stochastic["trajectory_quantiles"].squeeze(2),
            "ranking_stability": stochastic["ranking_stability"].squeeze(1),
            "router_diagnostics": {**router_diagnostics, **stochastic["sampler_diagnostics"]},
            "scientific_posterior_condition_applied": scientific_posterior_species_condition
            is not None,
            "scientific_posterior_condition_norm": posterior_state.norm(),
        }


def _normalization_scale(train_trajectories: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(train_trajectories), axis=(0, 1, 2, 3))
    scale = np.maximum(scale, np.float32(1e-4))
    if not np.isfinite(scale).all():
        raise ValueError("C2-S train-only normalization scale is non-finite")
    return scale.astype(np.float32)


def _gaussian_crps(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sigma = std.clamp_min(torch.finfo(std.dtype).eps)
    z = (target - mean) / sigma
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _model_prediction(
    model: RestrictedC2WorldActionModel,
    trajectories: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, Any]:
    node_count = model.region_count * model.compartment_count
    source = trajectories[:, :-1].reshape(trajectories.shape[0], trajectories.shape[1] - 1, node_count, model.species_count)
    return model(source, actions)


def _evaluate(
    model: RestrictedC2WorldActionModel,
    trajectories: torch.Tensor,
    uncertainties: torch.Tensor,
    actions: torch.Tensor,
    scale: torch.Tensor,
    species_ids: list[str],
    calibration_level: float,
    mc_pass_count: int,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        base = _model_prediction(model, trajectories, actions)
    mean = base["trajectory_mean"]
    aleatoric_std = base["trajectory_std"].clamp_min(1e-6)
    target = trajectories[:, 1:]
    target_nodes = target.reshape_as(mean)
    scale_nodes = scale.view(1, 1, 1, 1, -1)
    mean_raw = mean * scale_nodes
    target_raw = target_nodes * scale_nodes
    std_raw = aleatoric_std * scale_nodes

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    mc_means = []
    with torch.no_grad():
        for _ in range(int(mc_pass_count)):
            mc_means.append(_model_prediction(model, trajectories, actions)["trajectory_mean"])
    model.eval()
    epistemic_variance = torch.stack(mc_means, dim=0).var(dim=0, unbiased=False)
    aleatoric_variance = aleatoric_std.square()
    total_std = torch.sqrt(epistemic_variance + aleatoric_variance).clamp_min(1e-6)

    teacher_std = uncertainties.view(uncertainties.shape[0], 1, 1, 1, -1).clamp_min(1e-6)
    likelihood_std = torch.sqrt(total_std.square() + teacher_std.square()).clamp_min(1e-6)
    normalized_error = target_nodes - mean
    nll = 0.5 * (torch.log(2.0 * math.pi * likelihood_std.square()) + normalized_error.square() / likelihood_std.square())
    crps = _gaussian_crps(mean, likelihood_std, target_nodes)
    z_value = NormalDist().inv_cdf((1.0 + calibration_level) / 2.0)
    covered = (torch.abs(normalized_error) <= float(z_value) * likelihood_std).float().mean()

    signed_species = {"membrane_voltage"}
    nonnegative_indices = [index for index, species_id in enumerate(species_ids) if species_id not in signed_species]
    nonnegative_prediction = mean_raw[..., nonnegative_indices]
    negative_fraction = (nonnegative_prediction < 0.0).float().mean()
    final_rmse = torch.sqrt(torch.mean((mean_raw[:, -1] - target_raw[:, -1]).square()))

    action_sensitivity = torch.tensor(float("nan"), device=mean.device)
    contrast_rmse = torch.tensor(float("nan"), device=mean.device)
    if mean.shape[0] == 2:
        predicted_contrast = mean_raw[0] - mean_raw[1]
        observed_contrast = target_raw[0] - target_raw[1]
        action_sensitivity = predicted_contrast.abs().mean()
        contrast_rmse = torch.sqrt(torch.mean((predicted_contrast - observed_contrast).square()))

    metrics = {
        "trajectory_mse_normalized": float(torch.mean(normalized_error.square()).cpu()),
        "trajectory_rmse_raw": float(torch.sqrt(torch.mean((mean_raw - target_raw).square())).cpu()),
        "long_horizon_drift_rmse_raw": float(final_rmse.cpu()),
        "action_sensitivity_mean_abs_raw": float(action_sensitivity.cpu()),
        "structured_action_contrast_rmse_raw": float(contrast_rmse.cpu()),
        "crps_normalized": float(crps.mean().cpu()),
        "nll_normalized": float(nll.mean().cpu()),
        "constraint_negative_fraction": float(negative_fraction.cpu()),
        "epistemic_variance_mean": float(epistemic_variance.mean().cpu()),
        "aleatoric_variance_mean": float(aleatoric_variance.mean().cpu()),
        "calibration_interval_level": float(calibration_level),
        "calibration_interval_coverage": float(covered.cpu()),
        "calibration_absolute_error": float(abs(float(covered.cpu()) - calibration_level)),
        "finite_trajectory": float(torch.isfinite(mean).all().item()),
        "finite_uncertainty": float(torch.isfinite(total_std).all().item()),
    }
    return metrics, mean.detach()


def _checkpoint_roundtrip(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    expected_prediction: torch.Tensor,
    trajectories: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> bool:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored = RestrictedC2WorldActionModel(config).to(device)
    restored.load_state_dict(payload["model"])
    restored.eval()
    with torch.no_grad():
        prediction = _model_prediction(restored, trajectories, actions)["trajectory_mean"]
    return bool(torch.equal(prediction, expected_prediction))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--scale", choices=MODEL_SCALE_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _repo_path(args.output_dir)
    output_dir.relative_to(REPO_ROOT.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        raise FileExistsError("C2-S run output already contains run_result.json; use a new output directory")
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "neuronsifter-round23-c2s-development-run-v1",
        "status": "failed",
        "run_uid": output_dir.name,
        "lineage_id": "neuronsifter_vnext_dev_v1",
        "domain": "c2s",
        "model_scale": args.scale,
        "seed": int(args.seed),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
        "started_at": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "host": platform.node(),
    }
    try:
        config_path = _repo_path(args.config)
        config = load_config(config_path)
        contract = config["neuronsifter_vnext"]
        if contract["dual_gates"]["development"]["development_training_allowed"] is not True:
            raise RuntimeError("Round-23 development training gate is closed")
        if any(bool(value) for value in contract["dual_gates"]["publication"].values()):
            raise RuntimeError("Round-23 publication gates must remain closed during C2-S development")
        data = load_restricted_c2_development_data(config)
        effective_config = _profile_config(config, args.scale)
        profile = effective_config["neuronsifter_vnext"]["training"]["c2s_development"]["profiles"][args.scale]
        steps = int(contract["training"]["development_steps"][args.scale])
        device = _resolve_device(args.device)
        _set_seed(int(args.seed))
        model = RestrictedC2WorldActionModel(effective_config).to(device)
        optimizer_config = contract["training"]["c2s_development"]["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        trajectories_np = data["trajectories"]
        train_positions = data["train_positions"]
        validation_positions = data["validation_positions"]
        scale_np = _normalization_scale(trajectories_np[train_positions])
        normalized = trajectories_np / scale_np.reshape(1, 1, 1, 1, -1)
        trajectories = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        uncertainties = torch.as_tensor(data["uncertainties"] / scale_np.reshape(1, -1), dtype=torch.float32, device=device)
        actions = torch.as_tensor(data["action_features"], dtype=torch.float32, device=device)
        train_index = torch.as_tensor(train_positions, dtype=torch.long, device=device)
        validation_index = torch.as_tensor(validation_positions, dtype=torch.long, device=device)
        train_trajectories = trajectories.index_select(0, train_index)
        train_uncertainties = uncertainties.index_select(0, train_index)
        train_actions = actions.index_select(0, train_index)
        losses: list[float] = []
        model.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output = _model_prediction(model, train_trajectories, train_actions)
            target = train_trajectories[:, 1:].reshape_as(output["trajectory_mean"])
            precision = torch.reciprocal(
                train_uncertainties.view(train_uncertainties.shape[0], 1, 1, 1, -1).square() + 1e-4
            )
            precision = precision / precision.mean().clamp_min(torch.finfo(precision.dtype).eps)
            trajectory_loss = torch.mean(precision * (output["trajectory_mean"] - target).square())
            router_regularization = output["router_diagnostics"].get("router_regularization_loss")
            if not isinstance(router_regularization, torch.Tensor):
                router_regularization = trajectory_loss * 0.0
            loss = trajectory_loss + 0.001 * router_regularization
            if not torch.isfinite(loss):
                raise FloatingPointError("C2-S development objective became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimizer_config["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("C2-S development gradient norm became non-finite")
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            print(json.dumps({"step": step + 1, "loss": loss_value}, sort_keys=True), flush=True)

        validation_trajectories = trajectories.index_select(0, validation_index)
        validation_uncertainties = uncertainties.index_select(0, validation_index)
        validation_actions = actions.index_select(0, validation_index)
        uncertainty_config = contract["training"]["c2s_development"]["uncertainty"]
        metrics, expected_prediction = _evaluate(
            model,
            validation_trajectories,
            validation_uncertainties,
            validation_actions,
            torch.as_tensor(scale_np, dtype=torch.float32, device=device),
            [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]],
            float(uncertainty_config["calibration_interval_level"]),
            int(uncertainty_config["mc_dropout_pass_count"]),
        )
        if not all(math.isfinite(value) for key, value in metrics.items() if key not in {"action_sensitivity_mean_abs_raw", "structured_action_contrast_rmse_raw"}):
            raise FloatingPointError("C2-S validation metrics contain NaN or Inf")
        if not math.isfinite(metrics["action_sensitivity_mean_abs_raw"]) or not math.isfinite(metrics["structured_action_contrast_rmse_raw"]):
            raise FloatingPointError("C2-S validation action-contrast metrics are unavailable")

        checkpoint_path = output_dir / "last.pt"
        checkpoint_payload = {
            "schema_version": "neuronsifter-round23-c2s-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_scale": args.scale,
            "seed": int(args.seed),
            "profile": dict(profile),
            "normalization_scale": scale_np.tolist(),
            "input_identity": data["input_identity"],
            "selection_metric": "validation_trajectory_mse_normalized",
            "selection_value": metrics["trajectory_mse_normalized"],
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        checkpoint_hash = _sha256_file(checkpoint_path)
        roundtrip = _checkpoint_roundtrip(
            checkpoint_path,
            effective_config,
            expected_prediction,
            validation_trajectories,
            validation_actions,
            device,
        )
        if not roundtrip:
            raise RuntimeError("C2-S checkpoint roundtrip changed validation predictions")

        checkpoint_pointer = _repo_path(
            f"Output/checkpoints/round23/c2s_{args.scale.lower()}_seed{int(args.seed)}.json"
        )
        write_json(
            checkpoint_pointer,
            {
                "schema_version": "neuronsifter-round23-checkpoint-pointer-v1",
                "checkpoint_path": _repo_relative(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "run_uid": output_dir.name,
                "model_scale": args.scale,
                "seed": int(args.seed),
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
            },
        )
        result.update(
            {
                "status": "passed",
                "result_status": "trained_and_validation_evaluated",
                "device": str(device),
                "config_path": _repo_relative(config_path),
                "config_sha256": _sha256_file(config_path),
                "effective_profile_sha256": _json_sha256(profile),
                "model_profile": dict(profile),
                "steps": steps,
                "losses": losses,
                "mean_loss": float(np.mean(losses)),
                "finite_loss": True,
                "finite_trajectory": bool(metrics["finite_trajectory"]),
                "finite_uncertainty": bool(metrics["finite_uncertainty"]),
                "metrics": metrics,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
                "checkpoint_path": _repo_relative(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_roundtrip": True,
                "checkpoint_pointer": _repo_relative(checkpoint_pointer),
                "input_identity": data["input_identity"],
                "node_local_staging_used": data["node_local_staging_used"],
                "accessed_simulation_indices": data["accessed_simulation_indices"],
                "protected_simulation_indices": data["protected_simulation_indices"],
                "protected_test_access": 0,
            }
        )
        return result
    except Exception as exc:
        result.update({"status": "failed", "result_status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        result["ended_at"] = _utc_now()
        result["elapsed_seconds"] = float(time.time() - started)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["resource_usage"] = {
            "user_cpu_seconds": float(usage.ru_utime),
            "system_cpu_seconds": float(usage.ru_stime),
            "max_rss_kib": int(usage.ru_maxrss),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
