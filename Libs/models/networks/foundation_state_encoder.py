from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from Libs.models.modules.foundation_state_objectives import reverse_gradient


class NeuronSifterFoundationStateEncoder(nn.Module):
    unavailable_heads = (
        "transcriptomics_to_electrophysiology",
        "transcriptomics_to_imaging",
        "morphology_to_transcriptomics",
        "paired_cross_modal_reconstruction",
        "same_donor_multimodal_imputation",
    )

    def __init__(
        self,
        *,
        feature_vocabulary_size: int,
        cell_ontology_size: int,
        region_ontology_size: int,
        source_class_count: int,
        family_class_count: int,
        action_feature_dim: int,
        program_count: int,
        program_vocab_status: str = "available",
        profile: Mapping[str, object],
        posterior: Mapping[str, object],
    ) -> None:
        super().__init__()
        if feature_vocabulary_size < 2:
            raise ValueError(
                "C1 feature vocabulary must include padding and at least one real feature"
            )
        if (
            min(
                cell_ontology_size,
                region_ontology_size,
                source_class_count,
                family_class_count,
            )
            < 2
        ):
            raise ValueError(
                "C1 ontology and nuisance vocabularies must each contain at least two IDs"
            )
        if action_feature_dim <= 0:
            raise ValueError(
                "C1 action features must use a non-empty frozen vocabulary"
            )
        if program_vocab_status not in {"available", "not_available"}:
            raise ValueError(
                f"Unknown C1 program vocabulary status: {program_vocab_status}"
            )
        if program_vocab_status == "available" and program_count <= 0:
            raise ValueError("C1 available program vocabularies must be non-empty")
        if program_vocab_status == "not_available" and program_count != 0:
            raise ValueError(
                "C1 program_count must be zero when the program vocabulary is unavailable"
            )

        token_dim = int(profile["token_dim"])
        latent_dim = int(profile["latent_dim"])
        head_count = int(profile["attention_heads"])
        layer_count = int(profile["transformer_layers"])
        feedforward_dim = int(profile["feedforward_dim"])
        dropout_rate = float(profile["dropout_rate"])
        feature_token_budget = int(profile["feature_token_budget"])
        if (
            token_dim <= 0
            or latent_dim <= 0
            or layer_count <= 0
            or feedforward_dim <= 0
            or feature_token_budget <= 0
        ):
            raise ValueError(
                "C1 encoder dimensions, layer count, and token budget must be positive"
            )
        if head_count <= 0 or token_dim % head_count != 0:
            raise ValueError(
                "C1 token dimension must be divisible by the attention-head count"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("C1 encoder dropout must be in [0, 1)")

        self.feature_vocabulary_size = int(feature_vocabulary_size)
        self.cell_ontology_size = int(cell_ontology_size)
        self.region_ontology_size = int(region_ontology_size)
        self.source_class_count = int(source_class_count)
        self.family_class_count = int(family_class_count)
        self.action_feature_dim = int(action_feature_dim)
        self.program_count = int(program_count)
        self.program_vocab_status = program_vocab_status
        self.feature_token_budget = feature_token_budget
        self.log_variance_min = float(posterior["log_variance_min"])
        self.log_variance_max = float(posterior["log_variance_max"])
        if self.log_variance_min >= self.log_variance_max:
            raise ValueError("C1 posterior log-variance bounds are invalid")

        self.feature_embedding = nn.Embedding(
            feature_vocabulary_size, token_dim, padding_idx=0
        )
        self.value_projection = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=head_count,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layer_count,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        self.cell_ontology_embedding = nn.Embedding(
            cell_ontology_size, token_dim, padding_idx=0
        )
        self.region_ontology_embedding = nn.Embedding(
            region_ontology_size, token_dim, padding_idx=0
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        posterior_input_dim = token_dim * 4
        self.posterior_backbone = nn.Sequential(
            nn.Linear(posterior_input_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(feedforward_dim),
        )
        self.posterior_mean = nn.Linear(feedforward_dim, latent_dim)
        self.posterior_log_variance = nn.Linear(feedforward_dim, latent_dim)
        self.latent_to_token = nn.Linear(latent_dim, token_dim)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, 1),
        )
        self.program_head = (
            nn.Linear(latent_dim, program_count)
            if program_vocab_status == "available"
            else None
        )
        self.source_nuisance_head = nn.Linear(latent_dim, source_class_count)
        self.family_nuisance_head = nn.Linear(latent_dim, family_class_count)

    @staticmethod
    def runtime_flags(
        active_objectives: list[str] | tuple[str, ...],
    ) -> dict[str, bool]:
        active = set(active_objectives)
        return {
            "use_ontology_context": "ontology_aware_gene_cell_region_embedding"
            in active,
            "use_action_context": "action_conditioned_perturbation_representation"
            in active,
        }

    @staticmethod
    def _validate_index_vector(
        values: torch.Tensor,
        *,
        batch_size: int,
        vocabulary_size: int,
        name: str,
    ) -> None:
        if values.dtype != torch.long or values.shape != (batch_size,):
            raise ValueError(
                f"C1 {name} IDs must be int64 with one value per biological unit"
            )
        if torch.any(values < 0) or torch.any(values >= vocabulary_size):
            raise ValueError(f"C1 {name} IDs are outside the frozen vocabulary")

    def _validate_inputs(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor,
        observed_mask: torch.Tensor,
        target_mask: torch.Tensor,
        cell_ontology_ids: torch.Tensor,
        region_ontology_ids: torch.Tensor,
        observation_unit_index: torch.Tensor,
        action_features: torch.Tensor | None,
        action_available: torch.Tensor | None,
    ) -> int:
        if feature_ids.dtype != torch.long or feature_ids.ndim != 2:
            raise ValueError("C1 feature IDs must be an int64 [unit, token] tensor")
        if feature_ids.shape[1] > self.feature_token_budget:
            raise ValueError(
                "C1 feature-token width exceeds the configured profile budget"
            )
        if feature_values.shape != feature_ids.shape:
            raise ValueError("C1 feature values and IDs must have one shape")
        if (
            observed_mask.shape != feature_ids.shape
            or observed_mask.dtype != torch.bool
        ):
            raise ValueError(
                "C1 observed mask must be boolean and align with feature tokens"
            )
        if target_mask.shape != feature_ids.shape or target_mask.dtype != torch.bool:
            raise ValueError(
                "C1 target mask must be boolean and align with feature tokens"
            )
        if torch.any(target_mask & observed_mask):
            raise ValueError("C1 observed and masked-target tokens must be disjoint")
        if torch.any(observed_mask.sum(dim=1) == 0):
            raise ValueError(
                "Every C1 biological unit must contain at least one observed feature"
            )
        if torch.any(target_mask.sum(dim=1) == 0):
            raise ValueError(
                "Every C1 biological unit must contain at least one masked target"
            )
        real_feature_mask = observed_mask | target_mask
        if torch.any(feature_ids[~real_feature_mask] != 0) or torch.any(
            feature_ids[real_feature_mask] == 0
        ):
            raise ValueError(
                "C1 feature padding ID 0 must exactly match the observed/target mask union"
            )
        if torch.any(feature_ids < 0) or torch.any(
            feature_ids >= self.feature_vocabulary_size
        ):
            raise ValueError("C1 feature IDs are outside the frozen feature vocabulary")
        if not torch.isfinite(feature_values).all():
            raise ValueError(
                "C1 transcriptomic values, including padding, must be finite"
            )

        batch_size = feature_ids.shape[0]
        if batch_size <= 0:
            raise ValueError("C1 encoder requires at least one observation")
        self._validate_index_vector(
            cell_ontology_ids,
            batch_size=batch_size,
            vocabulary_size=self.cell_ontology_size,
            name="cell ontology",
        )
        self._validate_index_vector(
            region_ontology_ids,
            batch_size=batch_size,
            vocabulary_size=self.region_ontology_size,
            name="region ontology",
        )
        if (
            observation_unit_index.dtype != torch.long
            or observation_unit_index.shape != (batch_size,)
        ):
            raise ValueError(
                "C1 observation-to-biological-unit indices must be int64 and align with observations"
            )
        if torch.any(observation_unit_index < 0):
            raise ValueError("C1 biological-unit indices must be non-negative")
        biological_unit_count = int(observation_unit_index.max().item()) + 1
        observation_counts = torch.bincount(
            observation_unit_index, minlength=biological_unit_count
        )
        if torch.any(observation_counts == 0):
            raise ValueError(
                "C1 biological-unit indices must be contiguous and cover every unit"
            )
        if (action_features is None) != (action_available is None):
            raise ValueError("C1 action features require an explicit availability mask")
        if action_features is not None and action_available is not None:
            if action_features.shape != (batch_size, self.action_feature_dim):
                raise ValueError(
                    "C1 action features do not match the frozen action encoder width"
                )
            if (
                action_available.shape != (batch_size,)
                or action_available.dtype != torch.bool
            ):
                raise ValueError(
                    "C1 action availability must be boolean with one value per unit"
                )
            if not torch.isfinite(action_features).all():
                raise ValueError("C1 action features must be finite")
            if torch.any(action_features[~action_available] != 0.0):
                raise ValueError(
                    "Unavailable C1 actions must use an explicit all-zero feature vector"
                )
        return biological_unit_count

    @staticmethod
    def _aggregate_observations(
        values: torch.Tensor,
        observation_unit_index: torch.Tensor,
        biological_unit_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or values.shape[0] != observation_unit_index.shape[0]:
            raise ValueError("C1 unit aggregation requires an observation-aligned matrix")
        counts = torch.bincount(
            observation_unit_index, minlength=biological_unit_count
        )
        totals = values.new_zeros((biological_unit_count, values.shape[1]))
        totals.scatter_add_(
            0,
            observation_unit_index[:, None].expand(-1, values.shape[1]),
            values,
        )
        return totals / counts.to(dtype=values.dtype)[:, None], counts

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor,
        observed_mask: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        cell_ontology_ids: torch.Tensor,
        region_ontology_ids: torch.Tensor,
        observation_unit_index: torch.Tensor,
        action_features: torch.Tensor | None = None,
        action_available: torch.Tensor | None = None,
        use_ontology_context: bool,
        use_action_context: bool,
        sample_posterior: bool | None = None,
        nuisance_gradient_scale: float = 1.0,
        scientific_posterior_uncertainty_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        biological_unit_count = self._validate_inputs(
            feature_ids,
            feature_values,
            observed_mask,
            target_mask,
            cell_ontology_ids,
            region_ontology_ids,
            observation_unit_index,
            action_features,
            action_available,
        )
        batch_size = feature_ids.shape[0]
        real_feature_mask = observed_mask | target_mask
        encoded_values = feature_values.masked_fill(~observed_mask, 0.0).unsqueeze(-1)
        tokens = self.feature_embedding(feature_ids) + self.value_projection(
            encoded_values
        )
        tokens = tokens + target_mask.unsqueeze(-1) * self.mask_token
        tokens = tokens * real_feature_mask.unsqueeze(-1)
        encoded_tokens = self.token_encoder(
            tokens, src_key_padding_mask=~real_feature_mask
        )
        pooled = (encoded_tokens * real_feature_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / real_feature_mask.sum(dim=1, keepdim=True)

        if use_ontology_context:
            cell_context = self.cell_ontology_embedding(cell_ontology_ids)
            region_context = self.region_ontology_embedding(region_ontology_ids)
        else:
            cell_context = torch.zeros_like(pooled)
            region_context = torch.zeros_like(pooled)
        if not use_action_context:
            action_context = torch.zeros_like(pooled)
            effective_action_available = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=feature_ids.device,
            )
        elif action_features is None or action_available is None:
            raise ValueError(
                "Enabled C1 action context requires explicit action features and availability"
            )
        else:
            action_context = self.action_encoder(action_features)
            action_context = action_context * action_available.unsqueeze(-1)
            effective_action_available = action_available

        posterior_hidden = self.posterior_backbone(
            torch.cat([pooled, cell_context, region_context, action_context], dim=1)
        )
        latent_mean = self.posterior_mean(posterior_hidden)
        latent_log_variance = self.posterior_log_variance(posterior_hidden)
        if scientific_posterior_uncertainty_scale is None:
            posterior_uncertainty_scale = latent_log_variance.new_ones(batch_size)
        else:
            posterior_uncertainty_scale = scientific_posterior_uncertainty_scale
            if (
                posterior_uncertainty_scale.shape != (batch_size,)
                or not torch.is_floating_point(posterior_uncertainty_scale)
            ):
                raise ValueError(
                    "C1 scientific-posterior uncertainty scale must be a floating "
                    "tensor with one value per observation"
                )
            if (
                not torch.isfinite(posterior_uncertainty_scale).all()
                or torch.any(posterior_uncertainty_scale <= 0.0)
            ):
                raise ValueError(
                    "C1 scientific-posterior uncertainty scale must be finite and positive"
                )
            posterior_uncertainty_scale = posterior_uncertainty_scale.to(
                device=latent_log_variance.device,
                dtype=latent_log_variance.dtype,
            )
        # The external scientific posterior calibrates posterior standard
        # deviation, so its variance-space contribution is 2 * log(scale).
        latent_log_variance = (
            latent_log_variance
            + 2.0 * torch.log(posterior_uncertainty_scale).unsqueeze(1)
        ).clamp(min=self.log_variance_min, max=self.log_variance_max)
        should_sample = (
            self.training if sample_posterior is None else bool(sample_posterior)
        )
        if should_sample:
            latent = latent_mean + torch.randn_like(latent_mean) * torch.exp(
                0.5 * latent_log_variance
            )
        else:
            latent = latent_mean

        reconstruction_context = self.latent_to_token(latent).unsqueeze(1)
        reconstructed_values = self.reconstruction_head(
            encoded_tokens + reconstruction_context
        ).squeeze(-1)
        nuisance_latent = reverse_gradient(latent_mean, nuisance_gradient_scale)
        source_logits = self.source_nuisance_head(nuisance_latent)
        family_logits = self.family_nuisance_head(nuisance_latent)
        biological_unit_latent, biological_unit_observation_counts = (
            self._aggregate_observations(
                latent, observation_unit_index, biological_unit_count
            )
        )
        biological_unit_latent_mean, _ = self._aggregate_observations(
            latent_mean, observation_unit_index, biological_unit_count
        )
        posterior_second_moment = torch.exp(latent_log_variance) + latent_mean.square()
        biological_unit_second_moment, _ = self._aggregate_observations(
            posterior_second_moment,
            observation_unit_index,
            biological_unit_count,
        )
        biological_unit_latent_variance = (
            biological_unit_second_moment - biological_unit_latent_mean.square()
        ).clamp_min(torch.finfo(latent_mean.dtype).eps)
        biological_unit_source_logits, _ = self._aggregate_observations(
            source_logits, observation_unit_index, biological_unit_count
        )
        biological_unit_family_logits, _ = self._aggregate_observations(
            family_logits, observation_unit_index, biological_unit_count
        )
        return {
            "latent": latent,
            "latent_mean": latent_mean,
            "latent_log_variance": latent_log_variance,
            "latent_standard_deviation": torch.exp(0.5 * latent_log_variance),
            "biological_unit_latent": biological_unit_latent,
            "biological_unit_representation": biological_unit_latent,
            "biological_unit_latent_mean": biological_unit_latent_mean,
            "biological_unit_latent_log_variance": torch.log(
                biological_unit_latent_variance
            ),
            "biological_unit_latent_standard_deviation": torch.sqrt(
                biological_unit_latent_variance
            ),
            "biological_unit_uncertainty": torch.sqrt(
                biological_unit_latent_variance
            ),
            "biological_unit_observation_counts": biological_unit_observation_counts,
            "reconstructed_values": reconstructed_values,
            "program_predictions": (
                self.program_head(latent)
                if self.program_head is not None
                else latent.new_empty((latent.shape[0], 0))
            ),
            "source_logits": source_logits,
            "family_logits": family_logits,
            "source_diagnostic_logits": source_logits,
            "family_diagnostic_logits": family_logits,
            "biological_unit_source_diagnostic_logits": biological_unit_source_logits,
            "biological_unit_family_diagnostic_logits": biological_unit_family_logits,
            "action_available": effective_action_available,
            "scientific_posterior_uncertainty_scale": posterior_uncertainty_scale,
            "scientific_posterior_condition_applied": torch.tensor(
                scientific_posterior_uncertainty_scale is not None,
                dtype=torch.bool,
                device=feature_ids.device,
            ),
            "ontology_context_enabled": torch.tensor(
                use_ontology_context,
                dtype=torch.bool,
                device=feature_ids.device,
            ),
            "action_context_enabled": torch.tensor(
                use_action_context,
                dtype=torch.bool,
                device=feature_ids.device,
            ),
        }

    def head_status(self) -> dict[str, str]:
        statuses = {head_name: "not_available" for head_name in self.unavailable_heads}
        statuses["masked_gene_program_prediction"] = self.program_vocab_status
        return statuses
