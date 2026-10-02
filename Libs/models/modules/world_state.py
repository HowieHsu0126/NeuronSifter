from __future__ import annotations

import torch
from torch import nn

from Libs.models.layers.feed_forward import FeedForwardBlock
from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES, ranking_stability_context_tensor, tri_score_from_components


class WorldStatePosterior(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        species_count: int,
        posterior_hidden_dim: int,
        posterior_dropout_rate: float,
        pk_dim: int,
        env_dim: int,
        ephys_dim: int,
    ) -> None:
        super().__init__()
        observation_stat_dim = 2
        self.tx_head = FeedForwardBlock(latent_dim + species_count + observation_stat_dim, posterior_hidden_dim, latent_dim, posterior_dropout_rate)
        self.pk_head = FeedForwardBlock(latent_dim + species_count + species_count + pk_dim + observation_stat_dim, posterior_hidden_dim, pk_dim, posterior_dropout_rate)
        self.env_head = FeedForwardBlock(latent_dim + species_count + species_count + env_dim + latent_dim + observation_stat_dim, posterior_hidden_dim, env_dim, posterior_dropout_rate)
        self.ephys_head = FeedForwardBlock(latent_dim + ephys_dim + latent_dim + observation_stat_dim, posterior_hidden_dim, ephys_dim, posterior_dropout_rate)
        self.morph_head = FeedForwardBlock(latent_dim + species_count + species_count + latent_dim + observation_stat_dim, posterior_hidden_dim, latent_dim, posterior_dropout_rate)

    def forward(
        self,
        encoded_patient: torch.Tensor,
        baseline_field: torch.Tensor,
        parameter_posterior: torch.Tensor,
        region_prior_context: torch.Tensor,
        observation_mask: torch.Tensor,
        pk_indices: torch.Tensor,
        env_indices: torch.Tensor,
        ephys_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = encoded_patient.shape[0]
        baseline_summary = baseline_field.mean(dim=(1, 2))
        prior_summary = region_prior_context.mean(dim=0, keepdim=True).expand(batch_size, -1)
        observation_stats = torch.stack(
            [
                observation_mask.mean(dim=1),
                observation_mask.sum(dim=1),
            ],
            dim=1,
        )
        pk_summary = baseline_field.index_select(-1, pk_indices).mean(dim=(1, 2))
        env_summary = baseline_field.index_select(-1, env_indices).mean(dim=(1, 2))
        ephys_summary = baseline_field.index_select(-1, ephys_indices).mean(dim=(1, 2))
        return {
            "z_tx": self.tx_head(torch.cat([encoded_patient, baseline_summary, observation_stats], dim=1)),
            "z_pk": self.pk_head(torch.cat([encoded_patient, baseline_summary, parameter_posterior, pk_summary, observation_stats], dim=1)),
            "z_env": self.env_head(torch.cat([encoded_patient, baseline_summary, parameter_posterior, env_summary, prior_summary, observation_stats], dim=1)),
            "z_ephys": self.ephys_head(torch.cat([encoded_patient, ephys_summary, prior_summary, observation_stats], dim=1)),
            "z_morph": self.morph_head(torch.cat([encoded_patient, baseline_summary, parameter_posterior, prior_summary, observation_stats], dim=1)),
        }


class ObservationAndScreeningHeads(nn.Module):
    def __init__(
        self,
        patient_state_dim: int,
        target_feature_dim: int,
        latent_dim: int,
        hidden_dim: int,
        compartment_count: int,
        species_count: int,
        tri_weight_by_component: dict[str, float],
    ) -> None:
        super().__init__()
        self.patient_state_dim = patient_state_dim
        self.compartment_count = compartment_count
        self.species_count = species_count
        self.target_encoder = FeedForwardBlock(target_feature_dim, hidden_dim, latent_dim, 0.0)
        self.disease_head = nn.Linear(latent_dim + species_count, 1)
        self.target_score_head = nn.Sequential(
            nn.Linear(latent_dim * 2 + species_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.regimen_score_head = nn.Sequential(
            nn.Linear(latent_dim * 2 + species_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pkpd_consistency_head = nn.Sequential(
            nn.Linear(compartment_count * 2 + species_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        tri_input_dim = latent_dim * 2 + species_count + len(TRI_COMPONENT_NAMES)
        self.tri_heads = nn.ModuleDict(
            {
                component_name: nn.Sequential(
                    nn.Linear(tri_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for component_name in TRI_COMPONENT_NAMES
            }
        )
        self.tri_weight_by_component = {name: float(tri_weight_by_component[name]) for name in TRI_COMPONENT_NAMES}
        self.register_buffer(
            "tri_weight_vector",
            torch.tensor([self.tri_weight_by_component[name] for name in TRI_COMPONENT_NAMES], dtype=torch.float32),
        )

    def disease_logit(self, encoded_patient: torch.Tensor, baseline_field: torch.Tensor) -> torch.Tensor:
        field_summary = baseline_field.mean(dim=(1, 2))
        return self.disease_head(torch.cat([encoded_patient, field_summary], dim=1)).squeeze(-1)

    def target_scores(
        self,
        encoded_patient: torch.Tensor,
        baseline_field: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_targets = self.target_encoder(target_features)
        patient_summary = baseline_field.mean(dim=(1, 2)).unsqueeze(1).expand(-1, encoded_targets.shape[0], -1)
        repeated_patient = encoded_patient.unsqueeze(1).expand(-1, encoded_targets.shape[0], -1)
        repeated_targets = encoded_targets.unsqueeze(0).expand(encoded_patient.shape[0], -1, -1)
        scores = self.target_score_head(torch.cat([repeated_patient, repeated_targets, patient_summary], dim=-1)).squeeze(-1)
        return scores, encoded_targets

    def regimen_summaries(self, final_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_field = final_field.mean(dim=(2, 3))
        exposure = final_field[..., 0].mean(dim=(0, 2))
        occupancy = final_field[..., 1].mean(dim=(0, 2))
        return pooled_field, exposure, occupancy

    def tri_outputs(
        self,
        encoded_patient: torch.Tensor,
        encoded_regimens: torch.Tensor,
        pooled_field: torch.Tensor,
        regimen_scores: torch.Tensor,
        pkpd_consistency: torch.Tensor,
        tri_context: dict[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        regimen_count = regimen_scores.shape[1]
        calibration_default = (
            torch.maximum(torch.sigmoid(regimen_scores), 1.0 - torch.sigmoid(regimen_scores))
            if tri_context is None or "calibration" not in tri_context
            else tri_context["calibration"]
        )
        coverage = torch.ones_like(regimen_scores) if tri_context is None or "coverage" not in tri_context else tri_context["coverage"]
        ranking_stability = ranking_stability_context_tensor(regimen_scores)
        if tri_context is not None and "ranking_stability" in tri_context:
            ranking_stability = tri_context["ranking_stability"]
        ood_score = torch.ones_like(regimen_scores) if tri_context is None or "ood_score" not in tri_context else tri_context["ood_score"]
        if tri_context is not None and "pkpd_physics_consistency" in tri_context:
            pkpd_component = tri_context["pkpd_physics_consistency"]
        else:
            pkpd_component = pkpd_consistency
        repeated_patient = encoded_patient.unsqueeze(1).expand(-1, regimen_count, -1)
        repeated_regimen = encoded_regimens.unsqueeze(0).expand(encoded_patient.shape[0], -1, -1)
        tri_input = torch.cat(
            [
                repeated_patient,
                repeated_regimen,
                pooled_field,
                torch.stack(
                    [calibration_default, coverage, pkpd_component, ranking_stability, ood_score],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        tri_components = {
            component_name: torch.sigmoid(self.tri_heads[component_name](tri_input).squeeze(-1))
            for component_name in TRI_COMPONENT_NAMES
        }
        tri_score = tri_score_from_components(tri_components, self.tri_weight_vector)
        return tri_components, tri_score


