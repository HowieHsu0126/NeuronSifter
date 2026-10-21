"""Bounded action-conditioned Operational Twin network for controlled worlds."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from Libs.models.modules.neuronsifter_world_action import NeuronSifterWorldActionOperator
from Libs.models.modules.world_state import ExplicitWorldStatePosterior


class OperationalTwin(nn.Module):
    """Use observable variables as graph nodes; this is not an anatomical graph claim."""

    def __init__(self, feature_count: int, action_dim: int, config: dict[str, object]) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        posterior_dim = int(config["posterior_dim"])
        self.feature_count = feature_count
        self.minimum_scale = float(config["minimum_scale"])
        self.value_projection = nn.Linear(2, hidden)
        self.feature_embedding = nn.Embedding(feature_count, hidden)
        self.action_projection = nn.Linear(action_dim, hidden)
        self.family_embedding = nn.Embedding(3, hidden)
        self.posterior_heads = nn.ModuleDict({name: nn.Linear(hidden, posterior_dim) for name in ("z_tx", "z_pk", "z_env", "z_ephys", "z_morph")})
        self.posterior_condition = nn.Linear(posterior_dim * 5, hidden)
        expert_names = [
            "pharmacology_expert", "bbb_cns_transport_expert", "protein_pathology_expert",
            "inflammation_expert", "electrophysiology_rxd_expert", "transcriptomics_expert",
            "metabolomics_expert", "morphology_expert", "solver_observation_expert", "solver_residual_expert",
        ]
        splitting = {
            "enabled": True,
            "composition_order": expert_names,
            "coupling_masks": {name: ["observable_state"] for name in expert_names},
            "expert_residual_scale": 0.1,
            "shared_coupling_residual_scale": 0.05,
            "bottleneck_divisor": 4,
        }
        self.operator = NeuronSifterWorldActionOperator(
            hidden_dim=hidden, depth=int(config["depth"]), head_count=int(config["head_count"]),
            mechanism_token_count=int(config["mechanism_token_count"]), coordinate_dim=2,
            dropout_rate=float(config["dropout_rate"]), expert_names=expert_names, top_k=2, capacity_factor=1.25,
            entropy_loss_weight=0.01, load_balance_loss_weight=0.01, expert_capacity_loss_weight=0.01,
            expert_hidden_multiplier=2, mlp_multiplier=2, graph_mode_count=min(4, feature_count), wavelet_level_count=2,
            eigenvalue_temperature=1.0, normalize_laplacian=True, spectral_residual_scale=0.1,
            mechanism_slice_temperature=1.0, diffusion_sample_count=1, denoising_step_count=1,
            sample_chunk_size=1, training_sample_count=1, training_denoising_step_count=1,
            diffusion_noise_scale=0.0, score_step_size=0.1, terminal_noise_scale=0.0,
            ranking_stability_scale=1.0, quantile_levels=[0.1, 0.5, 0.9], sampler_seed=310031,
            return_sample_trajectories=False, operator_splitting=splitting,
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)
        adjacency = torch.eye(feature_count)
        if feature_count > 1:
            indices = torch.arange(feature_count - 1)
            adjacency[indices, indices + 1] = 1.0
            adjacency[indices + 1, indices] = 1.0
        self.register_buffer("observable_adjacency", adjacency)

    def forward(self, observation: torch.Tensor, observation_mask: torch.Tensor, action_encoding: torch.Tensor, family_index: torch.Tensor) -> dict[str, object]:
        if observation.shape != observation_mask.shape or observation.ndim != 3 or observation.shape[-1] != self.feature_count:
            raise ValueError("Observation and mask must align as [batch, time, feature].")
        masked = torch.where(observation_mask, observation, torch.zeros_like(observation))
        raw = torch.stack((masked, observation_mask.to(masked.dtype)), dim=-1)
        feature_ids = torch.arange(self.feature_count, device=observation.device)
        state = self.value_projection(raw) + self.feature_embedding(feature_ids)[None, None, :, :]
        pooled = (state * observation_mask[..., None]).sum(dim=(1, 2)) / observation_mask.sum(dim=(1, 2)).clamp_min(1)[:, None]
        factors = {name: head(pooled) for name, head in self.posterior_heads.items()}
        posterior = ExplicitWorldStatePosterior(
            **factors,
            factor_mask=torch.ones(observation.shape[0], 5, dtype=torch.bool, device=observation.device),
        ).validate()
        conditioning = self.action_projection(action_encoding) + self.family_embedding(family_index) + self.posterior_condition(posterior.stacked().flatten(1))
        conditioning = conditioning[:, None, None, :].expand(-1, observation.shape[1], self.feature_count, -1)
        time_axis = torch.linspace(0.0, 1.0, observation.shape[1], device=observation.device, dtype=observation.dtype)
        feature_axis = torch.linspace(0.0, 1.0, self.feature_count, device=observation.device, dtype=observation.dtype)
        coordinates = torch.stack(torch.meshgrid(time_axis, feature_axis, indexing="ij"), dim=-1)[None].expand(observation.shape[0], -1, -1, -1)
        hidden, diagnostics = self.operator.forward_preembedded(state, conditioning, coordinates, self.observable_adjacency)
        return {
            "predictive_mean": self.mean_head(hidden).squeeze(-1),
            "predictive_scale": F.softplus(self.scale_head(hidden).squeeze(-1)) + self.minimum_scale,
            "posterior": posterior,
            "operator_diagnostics": diagnostics,
            "graph_scope": "observable_variable_adjacency_not_anatomical_graph",
        }


def last_observed_state(observation: torch.Tensor, observation_mask: torch.Tensor) -> torch.Tensor:
    """Return the last causally observed value for each feature."""
    if observation.shape != observation_mask.shape or observation.ndim != 3:
        raise ValueError("Observation and mask must align as [batch, time, feature].")
    weighted_time = observation_mask.to(torch.long) * torch.arange(
        observation.shape[1], device=observation.device
    )[None, :, None]
    gather_index = weighted_time.argmax(dim=1)[:, None, :]
    return observation.gather(1, gather_index).squeeze(1)


class PersistenceNoActionPredictor(nn.Module):
    """No-fit action-agnostic persistence predictor."""

    checkpoint_creation_allowed = False
    calibration_eligible = False
    loss_name = "masked_mse_no_fit"
    model_family = "persistence_no_action"

    def forward(self, observation: torch.Tensor, observation_mask: torch.Tensor, action_encoding: torch.Tensor, family_index: torch.Tensor) -> dict[str, object]:
        del action_encoding, family_index
        last = last_observed_state(observation, observation_mask)
        mean = last[:, None, :].expand_as(observation)
        return {
            "predictive_mean": mean,
            "predictive_scale": None,
            "uncertainty_mode": "none",
            "calibration_eligible": False,
        }


class LinearDeltaOperationalTwin(nn.Module):
    """Transparent linear baseline predicting an action-conditioned delta from the last observed state."""

    checkpoint_creation_allowed = True
    calibration_eligible = False
    loss_name = "masked_mse_linear_delta"
    model_family = "linear_or_delta"

    def __init__(self, feature_count: int, action_dim: int, family_count: int = 3) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.action_dim = action_dim
        self.family_count = family_count
        self.delta = nn.Linear(feature_count + action_dim + family_count, feature_count)

    def forward(self, observation: torch.Tensor, observation_mask: torch.Tensor, action_encoding: torch.Tensor, family_index: torch.Tensor) -> dict[str, object]:
        last = last_observed_state(observation, observation_mask)
        family = F.one_hot(family_index, num_classes=self.family_count).to(last.dtype)
        features = torch.cat((last, action_encoding.to(last.dtype), family), dim=-1)
        mean = last[:, None, :] + self.delta(features)[:, None, :]
        return {
            "predictive_mean": mean.expand_as(observation),
            "predictive_scale": None,
            "uncertainty_mode": "none",
            "calibration_eligible": False,
        }


class MeanOnlyOperationalTwin(OperationalTwin):
    """Operational Twin backbone with deterministic mean-only output contract."""

    checkpoint_creation_allowed = True
    calibration_eligible = False
    loss_name = "masked_mse_deterministic"
    model_family = "deterministic_action_conditioned"

    def __init__(self, feature_count: int, action_dim: int, config: dict[str, object]) -> None:
        super().__init__(feature_count, action_dim, config)
        for parameter in self.scale_head.parameters():
            parameter.requires_grad_(False)

    def forward(self, observation: torch.Tensor, observation_mask: torch.Tensor, action_encoding: torch.Tensor, family_index: torch.Tensor) -> dict[str, object]:
        output = super().forward(observation, observation_mask, action_encoding, family_index)
        return {
            "predictive_mean": output["predictive_mean"],
            "predictive_scale": None,
            "posterior": output["posterior"],
            "operator_diagnostics": output["operator_diagnostics"],
            "graph_scope": output["graph_scope"],
            "uncertainty_mode": "deterministic_mean_only",
            "calibration_eligible": False,
        }
