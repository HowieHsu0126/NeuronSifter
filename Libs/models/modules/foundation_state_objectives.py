from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -float(ctx.scale) * gradient, None


def reverse_gradient(values: torch.Tensor, scale: float) -> torch.Tensor:
    if scale < 0.0:
        raise ValueError("Gradient-reversal scale must be non-negative")
    return _GradientReversal.apply(values, float(scale))


@dataclass(frozen=True)
class FoundationStateLoss:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, object]


def _weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or sample_weights.shape != values.shape:
        raise ValueError("Per-unit values and biological-unit weights must be one-dimensional and aligned")
    if not torch.isfinite(values).all() or not torch.isfinite(sample_weights).all():
        raise ValueError("Foundation-state losses and biological-unit weights must be finite")
    if torch.any(sample_weights < 0.0) or float(sample_weights.sum().detach()) <= 0.0:
        raise ValueError("Biological-unit weights must be non-negative with positive total mass")
    return torch.sum(values * sample_weights) / sample_weights.sum()


def _aggregate_observations_by_unit(
    values: torch.Tensor,
    observation_unit_index: torch.Tensor,
    biological_unit_count: int,
) -> torch.Tensor:
    if values.ndim != 1 or observation_unit_index.shape != values.shape:
        raise ValueError("Observation values and biological-unit indices must be one-dimensional and aligned")
    if observation_unit_index.dtype != torch.long:
        raise ValueError("Observation-to-biological-unit indices must be int64")
    if biological_unit_count <= 0:
        raise ValueError("C1 batches must contain at least one biological unit")
    if torch.any(observation_unit_index < 0) or torch.any(
        observation_unit_index >= biological_unit_count
    ):
        raise ValueError("Observation-to-biological-unit index is outside the sampled unit batch")
    counts = torch.bincount(observation_unit_index, minlength=biological_unit_count)
    if torch.any(counts == 0):
        raise ValueError("Every sampled C1 biological unit must contain at least one observation")
    totals = torch.zeros(
        biological_unit_count,
        dtype=values.dtype,
        device=values.device,
    )
    totals.scatter_add_(0, observation_unit_index, values)
    return totals / counts.to(dtype=values.dtype)


def _weighted_observation_mean(
    values: torch.Tensor,
    observation_unit_index: torch.Tensor,
    biological_unit_weights: torch.Tensor,
) -> torch.Tensor:
    per_unit = _aggregate_observations_by_unit(
        values,
        observation_unit_index,
        biological_unit_weights.shape[0],
    )
    return _weighted_mean(per_unit, biological_unit_weights)


def _reduce_observation_loss(
    values: torch.Tensor,
    observation_unit_index: torch.Tensor,
    biological_unit_weights: torch.Tensor,
    *,
    aggregate_by_biological_unit: bool,
) -> torch.Tensor:
    if aggregate_by_biological_unit:
        return _weighted_observation_mean(
            values, observation_unit_index, biological_unit_weights
        )
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("Observation-level ablation losses must be finite vectors")
    return values.mean()


def _aggregate_semantic_views(
    latent: torch.Tensor,
    semantic_group_ids: torch.Tensor,
    source_ids: torch.Tensor,
    observation_unit_index: torch.Tensor,
    biological_unit_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (semantic_group_ids >= 0) & (source_ids >= 0)
    if not torch.any(valid):
        empty_latent = latent.new_empty((0, latent.shape[1]))
        empty_ids = observation_unit_index.new_empty((0,))
        return empty_latent, empty_ids, empty_ids, empty_ids
    valid_units = observation_unit_index[valid]
    if torch.any(valid_units < 0) or torch.any(valid_units >= biological_unit_count):
        raise ValueError("C1 contrastive unit index is outside the sampled unit batch")
    keys = torch.stack(
        [valid_units, semantic_group_ids[valid], source_ids[valid]],
        dim=1,
    )
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    totals = torch.zeros(
        (unique_keys.shape[0], latent.shape[1]),
        dtype=latent.dtype,
        device=latent.device,
    )
    totals.scatter_add_(0, inverse[:, None].expand(-1, latent.shape[1]), latent[valid])
    counts = torch.bincount(inverse, minlength=unique_keys.shape[0]).to(dtype=latent.dtype)
    views = totals / counts[:, None]
    return views, unique_keys[:, 1], unique_keys[:, 2], unique_keys[:, 0]


def cross_source_supervised_contrastive_loss(
    latent: torch.Tensor,
    semantic_group_ids: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    temperature: float,
    observation_unit_index: torch.Tensor | None = None,
    biological_unit_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    if latent.ndim != 2:
        raise ValueError("C1 contrastive latent must have shape [observation, latent]")
    batch_size = latent.shape[0]
    if semantic_group_ids.shape != (batch_size,) or source_ids.shape != (batch_size,):
        raise ValueError("C1 semantic-group and source labels must align with observations")
    if temperature <= 0.0:
        raise ValueError("C1 contrastive temperature must be positive")
    if batch_size < 2:
        return latent.sum() * 0.0, 0

    if observation_unit_index is None and biological_unit_weights is None:
        views = latent
        view_group_ids = semantic_group_ids
        view_source_ids = source_ids
        view_unit_index = None
    elif observation_unit_index is None or biological_unit_weights is None:
        raise ValueError("C1 contrastive unit weighting requires both unit indices and unit weights")
    else:
        if observation_unit_index.shape != (batch_size,):
            raise ValueError("C1 contrastive unit indices must align with observations")
        views, view_group_ids, view_source_ids, view_unit_index = _aggregate_semantic_views(
            latent,
            semantic_group_ids,
            source_ids,
            observation_unit_index,
            biological_unit_weights.shape[0],
        )
    view_count = views.shape[0]
    if view_count < 2:
        return latent.sum() * 0.0, 0

    normalized = F.normalize(views, dim=1)
    similarities = normalized @ normalized.transpose(0, 1) / float(temperature)
    identity = torch.eye(view_count, dtype=torch.bool, device=latent.device)
    valid_labels = (view_group_ids >= 0) & (view_source_ids >= 0)
    cross_source_positive = (
        view_group_ids[:, None].eq(view_group_ids[None, :])
        & view_source_ids[:, None].ne(view_source_ids[None, :])
        & valid_labels[:, None]
        & valid_labels[None, :]
        & ~identity
    )
    if view_unit_index is not None:
        cross_source_positive &= view_unit_index[:, None].ne(view_unit_index[None, :])
    negative = (
        view_group_ids[:, None].ne(view_group_ids[None, :])
        & valid_labels[:, None]
        & valid_labels[None, :]
        & ~identity
    )
    candidates = cross_source_positive | negative
    eligible = cross_source_positive.any(dim=1) & negative.any(dim=1)
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        return latent.sum() * 0.0, 0

    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    losses: list[torch.Tensor] = []
    for index in eligible_indices.tolist():
        denominator = torch.logsumexp(similarities[index][candidates[index]], dim=0)
        positive_log_prob = similarities[index][cross_source_positive[index]] - denominator
        losses.append(-positive_log_prob.mean())
    per_anchor = torch.stack(losses)
    if view_unit_index is None or biological_unit_weights is None:
        return per_anchor.mean(), eligible_count
    eligible_unit_index = view_unit_index.index_select(0, eligible_indices)
    represented_units = torch.unique(eligible_unit_index, sorted=True)
    remapped = torch.searchsorted(represented_units, eligible_unit_index)
    represented_weights = biological_unit_weights.index_select(0, represented_units)
    return _weighted_observation_mean(per_anchor, remapped, represented_weights), eligible_count


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    observation_unit_index: torch.Tensor,
    biological_unit_weights: torch.Tensor,
    *,
    aggregate_by_biological_unit: bool,
    missing_label: int | None,
    label_name: str,
) -> tuple[torch.Tensor, int, int]:
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError(f"{label_name} logits and labels must align by observation")
    eligible = torch.ones_like(labels, dtype=torch.bool)
    if missing_label is not None:
        eligible &= labels.ne(missing_label)
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        return logits.sum() * 0.0, 0, 0
    selected_labels = labels[eligible]
    if torch.any(selected_labels < 0) or torch.any(selected_labels >= logits.shape[1]):
        raise ValueError(f"{label_name} labels are outside the configured classifier range")
    per_observation = F.cross_entropy(logits[eligible], selected_labels, reduction="none")
    selected_unit_index = observation_unit_index[eligible]
    represented_units = torch.unique(selected_unit_index, sorted=True)
    remapped = torch.searchsorted(represented_units, selected_unit_index)
    represented_weights = biological_unit_weights.index_select(0, represented_units)
    return _reduce_observation_loss(
        per_observation,
        remapped,
        represented_weights,
        aggregate_by_biological_unit=aggregate_by_biological_unit,
    ), eligible_count, int(represented_units.numel())


def foundation_state_objective(
    outputs: Mapping[str, torch.Tensor],
    target_values: torch.Tensor,
    target_mask: torch.Tensor,
    observation_unit_index: torch.Tensor,
    biological_unit_weights: torch.Tensor,
    *,
    active_objectives: Sequence[str],
    loss_weights: Mapping[str, float],
    semantic_group_ids: torch.Tensor | None = None,
    source_ids: torch.Tensor | None = None,
    family_ids: torch.Tensor | None = None,
    program_targets: torch.Tensor | None = None,
    program_target_mask: torch.Tensor | None = None,
    contrastive_temperature: float = 0.1,
) -> FoundationStateLoss:
    required_outputs = {
        "reconstructed_values",
        "latent_mean",
        "latent_log_variance",
        "source_logits",
        "family_logits",
        "program_predictions",
        "action_available",
        "ontology_context_enabled",
        "action_context_enabled",
    }
    missing_outputs = required_outputs - set(outputs)
    if missing_outputs:
        raise ValueError(f"Foundation-state output is missing: {sorted(missing_outputs)}")
    reconstructed = outputs["reconstructed_values"]
    if reconstructed.shape != target_values.shape or target_mask.shape != target_values.shape:
        raise ValueError("C1 reconstruction values, targets, and masks must have one shape")
    if target_mask.dtype != torch.bool:
        raise ValueError("C1 target mask must be boolean")
    if not torch.isfinite(target_values).all() or not torch.isfinite(reconstructed).all():
        raise ValueError("C1 reconstruction predictions and targets must be finite")
    observation_count = target_values.shape[0]
    if observation_unit_index.shape != (observation_count,):
        raise ValueError("C1 observation-to-unit indices must align with the observation batch")
    if biological_unit_weights.ndim != 1:
        raise ValueError("C1 biological-unit weights must be one-dimensional")
    _aggregate_observations_by_unit(
        torch.zeros(observation_count, dtype=target_values.dtype, device=target_values.device),
        observation_unit_index,
        biological_unit_weights.shape[0],
    )
    action_available = outputs["action_available"]
    if action_available.shape != (observation_count,) or action_available.dtype != torch.bool:
        raise ValueError("C1 action availability must align with observations")
    masked_count_by_observation = target_mask.sum(dim=1)
    if torch.any(masked_count_by_observation == 0):
        raise ValueError("Every C1 observation must contribute at least one masked target")

    active = set(active_objectives)
    allowed = {
        "masked_gene_program_prediction",
        "within_transcriptomics_feature_masking",
        "source_aware_contrastive_alignment",
        "ontology_aware_gene_cell_region_embedding",
        "family_conditioned_representation_learning",
        "biological_unit_aggregation",
        "action_conditioned_perturbation_representation",
        "source_family_nuisance_regularization",
        "uncertainty_aware_latent_posterior",
    }
    unknown = active - allowed
    if unknown:
        raise ValueError(f"Unknown C1 objectives: {sorted(unknown)}")
    required_base = {
        "within_transcriptomics_feature_masking",
        "family_conditioned_representation_learning",
        "uncertainty_aware_latent_posterior",
    }
    if not required_base.issubset(active):
        raise ValueError(
            "Every C1 encoder objective set must preserve masking, family-balanced "
            "conditioning, and posterior uncertainty"
        )
    aggregate_by_biological_unit = "biological_unit_aggregation" in active
    if family_ids is None or family_ids.shape != (observation_count,):
        raise ValueError("C1 family-balanced objectives require one family label per observation")
    if torch.any(family_ids < 0):
        raise ValueError("C1 family labels must be non-negative")
    ontology_enabled = bool(outputs["ontology_context_enabled"].item())
    action_enabled = bool(outputs["action_context_enabled"].item())
    if ontology_enabled != ("ontology_aware_gene_cell_region_embedding" in active):
        raise ValueError("C1 ontology context does not match the active objective set")
    if action_enabled != ("action_conditioned_perturbation_representation" in active):
        raise ValueError("C1 action context does not match the active objective set")
    if action_enabled and not torch.any(action_available):
        raise ValueError("Action-conditioned C1 objective has no real action-eligible observation")

    expected_loss_weights = {
        "masked_reconstruction",
        "posterior_kl",
        "cross_source_alignment",
        "source_nuisance",
        "family_nuisance",
        "program_prediction",
    }
    if set(loss_weights) != expected_loss_weights:
        raise ValueError("C1 loss weights must exactly cover the canonical objective components")
    required_positive_weights = {"masked_reconstruction", "posterior_kl"}
    if "source_aware_contrastive_alignment" in active:
        required_positive_weights.add("cross_source_alignment")
    if "source_family_nuisance_regularization" in active:
        required_positive_weights.update({"source_nuisance", "family_nuisance"})
    if "masked_gene_program_prediction" in active:
        required_positive_weights.add("program_prediction")
    non_positive = sorted(
        name for name in required_positive_weights if float(loss_weights[name]) <= 0.0
    )
    if non_positive:
        raise ValueError(
            "Active C1 objectives require strictly positive loss weights: "
            + str(non_positive)
        )

    zero = reconstructed.sum() * 0.0
    components: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, object] = {
        "masked_token_count": int(target_mask.sum().item()),
        "observation_count": observation_count,
        "biological_unit_count": int(biological_unit_weights.shape[0]),
        "within_unit_observation_aggregation_applied": int(
            aggregate_by_biological_unit
        ),
        "represented_family_count": int(torch.unique(family_ids).numel()),
        "alignment_eligible_view_count": 0,
        "alignment_status": "not_active",
        "source_nuisance_observation_count": 0,
        "source_nuisance_unit_count": 0,
        "family_nuisance_observation_count": 0,
        "family_nuisance_unit_count": 0,
        "program_target_observation_count": 0,
        "program_target_unit_count": 0,
        "action_conditioned_observation_count": int(action_available.sum().item()),
        "action_conditioned_unit_count": int(
            torch.unique(observation_unit_index[action_available]).numel()
        ),
        "action_status": (
            "not_active"
            if not action_enabled
            else ("estimable" if bool(torch.any(action_available)) else "not_estimable")
        ),
    }

    squared_error = (reconstructed - target_values).square()
    per_observation_reconstruction = (
        (squared_error * target_mask).sum(dim=1) / masked_count_by_observation
    )
    components["masked_reconstruction"] = _reduce_observation_loss(
        per_observation_reconstruction,
        observation_unit_index,
        biological_unit_weights,
        aggregate_by_biological_unit=aggregate_by_biological_unit,
    )

    latent_mean = outputs["latent_mean"]
    latent_log_variance = outputs["latent_log_variance"]
    if latent_mean.shape != latent_log_variance.shape or latent_mean.shape[0] != observation_count:
        raise ValueError("C1 posterior mean and log variance must align with the observation batch")
    per_observation_kl = -0.5 * (
        1.0
        + latent_log_variance
        - latent_mean.square()
        - latent_log_variance.exp()
    ).mean(dim=1)
    components["posterior_kl"] = _reduce_observation_loss(
        per_observation_kl,
        observation_unit_index,
        biological_unit_weights,
        aggregate_by_biological_unit=aggregate_by_biological_unit,
    )

    if "source_aware_contrastive_alignment" in active:
        if semantic_group_ids is None or source_ids is None:
            raise ValueError("Cross-source alignment requires semantic-group and source labels")
        alignment_loss, eligible_count = cross_source_supervised_contrastive_loss(
            latent_mean,
            semantic_group_ids,
            source_ids,
            temperature=contrastive_temperature,
            observation_unit_index=(
                observation_unit_index if aggregate_by_biological_unit else None
            ),
            biological_unit_weights=(
                biological_unit_weights if aggregate_by_biological_unit else None
            ),
        )
        components["cross_source_alignment"] = alignment_loss
        diagnostics["alignment_eligible_view_count"] = eligible_count
        diagnostics["alignment_status"] = (
            "estimable" if eligible_count > 0 else "not_estimable"
        )
        if eligible_count == 0:
            raise ValueError(
                "C1 source-aware alignment is not_estimable: "
                "the batch has no cross-source positive and negative anchors"
            )
    else:
        components["cross_source_alignment"] = zero

    if "source_family_nuisance_regularization" in active:
        if source_ids is None or family_ids is None:
            raise ValueError("Source/family nuisance regularization requires both labels")
        (
            components["source_nuisance"],
            diagnostics["source_nuisance_observation_count"],
            diagnostics["source_nuisance_unit_count"],
        ) = _classification_loss(
            outputs["source_logits"],
            source_ids,
            observation_unit_index,
            biological_unit_weights,
            aggregate_by_biological_unit=aggregate_by_biological_unit,
            missing_label=None,
            label_name="source",
        )
        (
            components["family_nuisance"],
            diagnostics["family_nuisance_observation_count"],
            diagnostics["family_nuisance_unit_count"],
        ) = _classification_loss(
            outputs["family_logits"],
            family_ids,
            observation_unit_index,
            biological_unit_weights,
            aggregate_by_biological_unit=aggregate_by_biological_unit,
            missing_label=None,
            label_name="family",
        )
    else:
        components["source_nuisance"] = zero
        components["family_nuisance"] = zero

    if "masked_gene_program_prediction" in active:
        if program_targets is None or program_target_mask is None:
            raise ValueError("Masked gene-program prediction requires real targets and an availability mask")
        predictions = outputs["program_predictions"]
        if predictions.shape != program_targets.shape or predictions.shape[0] != observation_count:
            raise ValueError("C1 program predictions and targets must align")
        if program_target_mask.shape != program_targets.shape or program_target_mask.dtype != torch.bool:
            raise ValueError("C1 program target availability mask must be boolean and aligned")
        if predictions.shape[1] == 0:
            raise ValueError("C1 program prediction requires a non-empty frozen program vocabulary")
        if not torch.isfinite(program_targets).all() or not torch.isfinite(predictions).all():
            raise ValueError("C1 program predictions and targets must be finite")
        target_count = program_target_mask.sum(dim=1)
        eligible_program = target_count > 0
        diagnostics["program_target_observation_count"] = int(eligible_program.sum().item())
        if torch.any(eligible_program):
            selected_unit_index = observation_unit_index[eligible_program]
            represented_units = torch.unique(selected_unit_index, sorted=True)
            remapped = torch.searchsorted(represented_units, selected_unit_index)
            represented_weights = biological_unit_weights.index_select(0, represented_units)
            per_observation_program = (
                (predictions - program_targets).square() * program_target_mask
            ).sum(dim=1)[eligible_program] / target_count[eligible_program]
            components["program_prediction"] = _reduce_observation_loss(
                per_observation_program,
                remapped,
                represented_weights,
                aggregate_by_biological_unit=aggregate_by_biological_unit,
            )
            diagnostics["program_target_unit_count"] = int(represented_units.numel())
        else:
            raise ValueError(
                "Masked gene-program prediction is not_estimable: "
                "the batch has no real program targets"
            )
    else:
        components["program_prediction"] = zero

    total = zero
    for component_name, component in components.items():
        weight = float(loss_weights[component_name])
        if not torch.isfinite(torch.tensor(weight)) or weight < 0.0:
            raise ValueError(f"C1 loss weight must be non-negative: {component_name}")
        total = total + weight * component
    if not torch.isfinite(total):
        raise ValueError("C1 foundation-state objective is non-finite")
    return FoundationStateLoss(total=total, components=components, diagnostics=diagnostics)
