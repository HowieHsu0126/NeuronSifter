from __future__ import annotations

import torch
from torch import nn

from Libs.models.layers.feed_forward import FeedForwardBlock, required_policy_vector
from Libs.models.modules.moe import SemanticMechanismMixtureOfExperts


class RegionPriorEncoder(nn.Module):
    def __init__(
        self,
        region_prior_feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
        compartment_count: int,
        species_count: int,
    ) -> None:
        super().__init__()
        self.compartment_count = compartment_count
        self.species_count = species_count
        self.field_prior_head = FeedForwardBlock(region_prior_feature_dim, hidden_dim, compartment_count * species_count, 0.0)
        self.region_context_head = FeedForwardBlock(region_prior_feature_dim, hidden_dim, latent_dim, 0.0)
        self.parameter_bias_head = FeedForwardBlock(region_prior_feature_dim, hidden_dim, species_count, 0.0)

    def forward(self, region_prior: torch.Tensor) -> dict[str, torch.Tensor]:
        field_prior = torch.sigmoid(self.field_prior_head(region_prior)).reshape(
            region_prior.shape[0],
            self.compartment_count,
            self.species_count,
        )
        region_context = torch.tanh(self.region_context_head(region_prior))
        parameter_bias = torch.sigmoid(self.parameter_bias_head(region_prior.mean(dim=0, keepdim=False)))
        return {
            "field_prior": field_prior,
            "region_context": region_context,
            "parameter_bias": parameter_bias,
        }


class SampleEvidenceEncoder(nn.Module):
    def __init__(
        self,
        patient_state_dim: int,
        hidden_dim: int,
        latent_dim: int,
        region_count: int,
        compartment_count: int,
        species_count: int,
        region_prior_feature_dim: int,
        dropout_rate: float,
        baseline_field_policy: dict[str, object] | None = None,
        multimodal_moe_config: dict[str, object] | None = None,
        joint_observations: bool = False,
    ) -> None:
        super().__init__()
        self.region_count = region_count
        self.compartment_count = compartment_count
        self.species_count = species_count
        self.region_prior_encoder = RegionPriorEncoder(
            region_prior_feature_dim,
            hidden_dim,
            latent_dim,
            compartment_count,
            species_count,
        )
        self.observation_encoder = FeedForwardBlock(patient_state_dim, hidden_dim, latent_dim, dropout_rate)
        self.retained_mask_projection = nn.Linear(patient_state_dim * 2, latent_dim) if joint_observations else None
        self.baseline_field_head = FeedForwardBlock(latent_dim + region_count, hidden_dim, region_count * compartment_count * species_count, dropout_rate)
        self.parameter_head = FeedForwardBlock(latent_dim + region_count, hidden_dim, species_count, dropout_rate)
        self.observation_reconstruction_head = FeedForwardBlock(region_count * species_count, hidden_dim, patient_state_dim, dropout_rate)
        if baseline_field_policy is None:
            raise ValueError("baseline_field_policy is required for SampleEvidenceEncoder.")
        self.register_buffer("baseline_direct_scale", required_policy_vector(baseline_field_policy, "direct_scale", species_count))
        self.register_buffer("baseline_offset", required_policy_vector(baseline_field_policy, "offset", species_count))
        self.register_buffer("baseline_pathology_gain", required_policy_vector(baseline_field_policy, "pathology_gain", species_count))
        self.register_buffer("baseline_raw_field_gain", required_policy_vector(baseline_field_policy, "raw_field_gain", species_count))
        if "region_prior_mix" not in baseline_field_policy:
            raise ValueError("baseline_field_policy.region_prior_mix is required.")
        self.region_prior_mix = float(baseline_field_policy["region_prior_mix"])
        self.multimodal_moe: SemanticMechanismMixtureOfExperts | None = None
        self._latest_multimodal_router_diagnostics: dict[str, torch.Tensor] = {}
        if multimodal_moe_config is not None and bool(multimodal_moe_config.get("enabled", False)):
            missing_moe_fields = [
                field_name
                for field_name in [
                    "expert_names",
                    "top_k",
                    "capacity_factor",
                    "entropy_loss_weight",
                    "load_balance_loss_weight",
                    "expert_capacity_loss_weight",
                    "expert_hidden_multiplier",
                ]
                if field_name not in multimodal_moe_config
            ]
            if missing_moe_fields:
                raise ValueError(f"Missing multimodal_moe config fields: {missing_moe_fields}")
            self.multimodal_moe = SemanticMechanismMixtureOfExperts(
                latent_dim,
                [str(expert_name) for expert_name in multimodal_moe_config["expert_names"]],
                int(multimodal_moe_config["top_k"]),
                dropout_rate,
                float(multimodal_moe_config["capacity_factor"]),
                float(multimodal_moe_config["entropy_loss_weight"]),
                float(multimodal_moe_config["load_balance_loss_weight"]),
                float(multimodal_moe_config["expert_capacity_loss_weight"]),
                int(multimodal_moe_config["expert_hidden_multiplier"]),
            )

    def encode_retained(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return encode_retained_observations(self.observation_encoder, self.retained_mask_projection, values, masks)

    def forward(
        self,
        patient_latent: torch.Tensor,
        observation_mask: torch.Tensor,
        region_prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded_patient = self.observation_encoder(patient_latent)
        if self.multimodal_moe is not None:
            routed_patient, self._latest_multimodal_router_diagnostics = self.multimodal_moe(encoded_patient)
            encoded_patient = encoded_patient + routed_patient
        else:
            self._latest_multimodal_router_diagnostics = {}
        encoder_input = torch.cat([encoded_patient, observation_mask], dim=1)
        region_prior_payload = self.region_prior_encoder(region_prior)
        raw_field = torch.sigmoid(self.baseline_field_head(encoder_input)).reshape(
            patient_latent.shape[0],
            self.region_count,
            self.compartment_count,
            self.species_count,
        )
        pathology_drive = torch.sigmoid(encoded_patient.mean(dim=1, keepdim=True)).unsqueeze(-1).unsqueeze(-1)
        baseline_field = (
            self.baseline_offset.view(1, 1, 1, -1)
            + self.baseline_pathology_gain.view(1, 1, 1, -1) * pathology_drive
            + self.baseline_raw_field_gain.view(1, 1, 1, -1) * raw_field
        )
        baseline_field = baseline_field * self.baseline_direct_scale.view(1, 1, 1, -1)
        baseline_field = baseline_field + self.region_prior_mix * region_prior_payload["field_prior"].unsqueeze(0)
        parameter_posterior = torch.sigmoid(self.parameter_head(encoder_input) + region_prior_payload["parameter_bias"].unsqueeze(0))
        observed_summary = (baseline_field.mean(dim=2) * observation_mask.unsqueeze(-1)).reshape(patient_latent.shape[0], -1)
        reconstructed_patient = self.observation_reconstruction_head(observed_summary)
        return {
            "encoded_patient": encoded_patient,
            "baseline_field": baseline_field,
            "parameter_posterior": parameter_posterior,
            "region_prior_field": region_prior_payload["field_prior"],
            "region_prior_context": region_prior_payload["region_context"],
            "reconstructed_patient": reconstructed_patient,
            "multimodal_router_diagnostics": self._latest_multimodal_router_diagnostics,
        }




def encode_retained_observations(encoder, mask_projection, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Shared observation projection for screening and native pretraining."""
    if mask_projection is None or masks.shape != (values.shape[0], values.shape[1] * 2):
        raise ValueError("Retained observation encoder requires its explicit mask/event interface.")
    return encoder(values) + mask_projection(masks)
