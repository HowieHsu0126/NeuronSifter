from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
import torch


TRI_COMPONENT_NAMES = (
    "calibration",
    "coverage",
    "pkpd_physics_consistency",
    "ranking_stability",
    "ood_score",
)


def stable_uid(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    joined = "::".join(normalized_parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug_seed = "_".join(normalized_parts[:4]) if normalized_parts else "screen"
    return f"{normalize_text(slug_seed).replace(' ', '_')[:80]}_{digest}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", str(value).strip().lower())).strip()


def component_scale(config: dict, component_name: str) -> float:
    enabled = bool(config["screening_pipeline"]["screening_benchmark"]["component_switches"][component_name])
    return 1.0 if enabled else 0.0


def tri_weight_by_component(config: dict) -> dict[str, float]:
    tri_weights = config["screening_pipeline"]["screening_benchmark"]["tri_weights"]
    return {name: float(tri_weights[name]) for name in TRI_COMPONENT_NAMES}


def tri_component_stack(tri_components: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([tri_components[name] for name in TRI_COMPONENT_NAMES], dim=-1)


def tri_score_from_components(tri_components: dict[str, torch.Tensor], tri_weights: dict[str, float] | torch.Tensor) -> torch.Tensor:
    component_stack = tri_component_stack(tri_components)
    if isinstance(tri_weights, torch.Tensor):
        weight_vector = tri_weights.to(device=component_stack.device, dtype=component_stack.dtype)
    else:
        weight_vector = torch.tensor([float(tri_weights[name]) for name in TRI_COMPONENT_NAMES], dtype=component_stack.dtype, device=component_stack.device)
    return (component_stack * weight_vector).sum(dim=-1) / weight_vector.sum().clamp_min(1e-6)


def tri_component_loss(predicted_components: dict[str, torch.Tensor], target_components: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [
        torch.nn.functional.mse_loss(predicted_components[name], target_components[name])
        for name in TRI_COMPONENT_NAMES
    ]
    return torch.stack(losses).mean()


def expand_sample_vector(sample_vector: np.ndarray, regimen_count: int) -> np.ndarray:
    if sample_vector.size == 0:
        return np.empty((0, regimen_count), dtype=np.float32)
    return np.repeat(sample_vector[:, None], regimen_count, axis=1).astype(np.float32)


def prediction_confidence_vector(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    return np.maximum(probabilities, 1.0 - probabilities).astype(np.float32)


def calibration_target_vector(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    labels = labels.astype(np.float32)
    return (1.0 - np.abs(probabilities - labels)).astype(np.float32)


def feature_coverage_vector(patient_matrix: np.ndarray) -> np.ndarray:
    if patient_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    return (np.count_nonzero(patient_matrix, axis=1) / max(patient_matrix.shape[1], 1)).astype(np.float32)


def latent_ood_vector(latent_matrix: np.ndarray) -> np.ndarray:
    if latent_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    centroid = latent_matrix.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(latent_matrix - centroid, axis=1)
    return (1.0 - (distances / (distances.max() + 1e-6))).astype(np.float32)


def ranking_stability_target_vector(regimen_label_matrix: np.ndarray) -> np.ndarray:
    if regimen_label_matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if regimen_label_matrix.shape[1] == 0:
        return np.zeros((regimen_label_matrix.shape[0],), dtype=np.float32)
    sorted_values = np.sort(regimen_label_matrix.astype(np.float32), axis=1)
    top_values = sorted_values[:, -1]
    second_values = sorted_values[:, -2] if regimen_label_matrix.shape[1] > 1 else np.zeros_like(top_values)
    margin = np.maximum(0.0, top_values - second_values)
    return np.clip(margin / (np.abs(top_values) + 1e-6), 0.0, 1.0).astype(np.float32)


def ranking_stability_context_tensor(score_matrix: torch.Tensor) -> torch.Tensor:
    if score_matrix.shape[1] == 0:
        return torch.empty_like(score_matrix)
    if score_matrix.shape[1] == 1:
        top_values = score_matrix[:, 0]
        second_values = torch.zeros_like(top_values)
    else:
        top_pair = torch.topk(score_matrix, k=2, dim=1).values
        top_values = top_pair[:, 0]
        second_values = top_pair[:, 1]
    margin = torch.clamp(top_values - second_values, min=0.0)
    normalized_margin = torch.clamp(margin / top_values.abs().clamp_min(1e-6), min=0.0, max=1.0)
    return normalized_margin.unsqueeze(1).expand(-1, score_matrix.shape[1])


def tri_context_tensors(
    patient_probabilities: np.ndarray,
    patient_matrix: np.ndarray,
    latent_matrix: np.ndarray,
    regimen_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "calibration": torch.tensor(expand_sample_vector(prediction_confidence_vector(patient_probabilities), regimen_count), dtype=torch.float32, device=device),
        "coverage": torch.tensor(expand_sample_vector(feature_coverage_vector(patient_matrix), regimen_count), dtype=torch.float32, device=device),
        "ood_score": torch.tensor(expand_sample_vector(latent_ood_vector(latent_matrix), regimen_count), dtype=torch.float32, device=device),
    }


def tri_target_tensors(
    patient_probabilities: np.ndarray,
    patient_labels: np.ndarray,
    patient_matrix: np.ndarray,
    latent_matrix: np.ndarray,
    regimen_label_matrix: np.ndarray,
    pkpd_physics_target: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    regimen_count = regimen_label_matrix.shape[1]
    expanded_pkpd_target = pkpd_physics_target.unsqueeze(0).expand(regimen_label_matrix.shape[0], -1) if pkpd_physics_target.ndim == 1 else pkpd_physics_target
    return {
        "calibration": torch.tensor(expand_sample_vector(calibration_target_vector(patient_probabilities, patient_labels), regimen_count), dtype=torch.float32, device=device),
        "coverage": torch.tensor(expand_sample_vector(feature_coverage_vector(patient_matrix), regimen_count), dtype=torch.float32, device=device),
        "pkpd_physics_consistency": expanded_pkpd_target.to(device=device, dtype=torch.float32),
        "ranking_stability": torch.tensor(expand_sample_vector(ranking_stability_target_vector(regimen_label_matrix), regimen_count), dtype=torch.float32, device=device),
        "ood_score": torch.tensor(expand_sample_vector(latent_ood_vector(latent_matrix), regimen_count), dtype=torch.float32, device=device),
    }


def pkpd_reference_tensors(regimen_frame: pd.DataFrame, config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prior_config = config["screening_pipeline"]["screening_benchmark"]["pkpd_prior_consistency"]
    compartment_order = list(prior_config["compartment_order"])
    dose_lower_bound = float(prior_config["dose_multiplier_lower_bound"])
    dose_upper_bound = float(prior_config["dose_multiplier_upper_bound"])
    dose_exposure_exponent = float(prior_config["dose_exposure_exponent"])
    dose_occupancy_exponent = float(prior_config["dose_occupancy_exponent"])
    occupancy_lower_bound = float(prior_config["occupancy_lower_bound"])
    occupancy_upper_bound = float(prior_config["occupancy_upper_bound"])
    pk_feasibility_values = pd.to_numeric(regimen_frame["pk_feasibility_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dose_multipliers = pd.to_numeric(regimen_frame["dose_multiplier"], errors="coerce").fillna(dose_lower_bound).clip(dose_lower_bound, dose_upper_bound)

    exposure_targets: list[list[float]] = []
    occupancy_targets: list[list[float]] = []
    for regimen_row, pk_feasibility_value, dose_multiplier in zip(regimen_frame.itertuples(index=False), pk_feasibility_values.tolist(), dose_multipliers.tolist(), strict=True):
        route_name = str(regimen_row.route)
        schedule_name = str(regimen_row.schedule)
        feasibility_scale = 0.5 + 0.5 * float(pk_feasibility_value)
        exposure_scale = float(dose_multiplier) ** dose_exposure_exponent
        occupancy_scale = float(dose_multiplier) ** dose_occupancy_exponent
        exposure_targets.append(
            [
                float(prior_config["exposure_target_by_compartment"][compartment_name])
                * float(prior_config["route_exposure_multiplier"][route_name][compartment_name])
                * float(prior_config["schedule_exposure_multiplier"][schedule_name])
                * exposure_scale
                * feasibility_scale
                for compartment_name in compartment_order
            ]
        )
        occupancy_targets.append(
            [
                min(
                    occupancy_upper_bound,
                    max(
                        occupancy_lower_bound,
                        float(prior_config["occupancy_target_by_compartment"][compartment_name])
                        * float(prior_config["route_occupancy_multiplier"][route_name])
                        * float(prior_config["schedule_occupancy_multiplier"][schedule_name])
                        * occupancy_scale
                        * feasibility_scale,
                    ),
                )
                for compartment_name in compartment_order
            ]
        )
    exposure_target_tensor = torch.tensor(exposure_targets, dtype=torch.float32, device=device)
    occupancy_target_tensor = torch.tensor(occupancy_targets, dtype=torch.float32, device=device)
    pk_feasibility_tensor = torch.tensor(pk_feasibility_values.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    return exposure_target_tensor, occupancy_target_tensor, pk_feasibility_tensor


def pkpd_prior_loss(
    regimen_output: dict[str, torch.Tensor],
    exposure_target_tensor: torch.Tensor,
    occupancy_target_tensor: torch.Tensor,
    pk_feasibility_tensor: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_config = config["screening_pipeline"]["screening_benchmark"]["pkpd_prior_consistency"]
    exposure_residual = torch.log1p(regimen_output["exposure"]) - torch.log1p(exposure_target_tensor)
    occupancy_residual = regimen_output["occupancy"] - occupancy_target_tensor
    exposure_alignment = torch.mean(exposure_residual ** 2)
    occupancy_alignment = torch.mean(occupancy_residual ** 2)
    if regimen_output["exposure"].ndim != 2:
        raise ValueError("PK/PD loss expects regimen_output['exposure'] shaped [regimen, compartment].")
    if regimen_output["occupancy"].ndim != 2:
        raise ValueError("PK/PD loss expects regimen_output['occupancy'] shaped [regimen, compartment].")
    normalized_exposure = regimen_output["exposure"] / (regimen_output["exposure"].max(dim=-1, keepdim=True).values + 1e-6)
    feasibility_target = pk_feasibility_tensor.unsqueeze(-1).expand_as(normalized_exposure)
    feasibility_alignment = torch.mean((normalized_exposure - feasibility_target) ** 2)
    total_loss = (
        float(prior_config["exposure_alignment_weight"]) * exposure_alignment
        + float(prior_config["occupancy_alignment_weight"]) * occupancy_alignment
        + float(prior_config["pk_feasibility_alignment_weight"]) * feasibility_alignment
    )
    per_regimen_error = torch.mean(torch.abs(exposure_residual), dim=-1) + torch.mean(torch.abs(occupancy_residual), dim=-1)
    per_regimen_proxy = torch.exp(-per_regimen_error)
    return total_loss, per_regimen_proxy


def world_state_factorization_loss(
    world_state_payload: dict[str, torch.Tensor],
    hidden_state_support: torch.Tensor,
    missing_modality_count: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    regularization_config = config["screening_pipeline"]["screening_benchmark"]["foundation_training"]["world_state_factorization"]
    minimum_factor_std = float(regularization_config["minimum_factor_std"])
    orthogonality_weight = float(regularization_config["orthogonality_weight"])
    hidden_state_weight_scale = float(regularization_config["hidden_state_weight_scale"])
    factor_tensors = [
        world_state_payload["z_tx"],
        world_state_payload["z_pk"],
        world_state_payload["z_env"],
        world_state_payload["z_ephys"],
        world_state_payload["z_morph"],
    ]
    variance_loss_terms = []
    for factor_tensor in factor_tensors:
        factor_std = factor_tensor.std(dim=0, unbiased=False).mean()
        variance_loss_terms.append(torch.relu(factor_tensor.new_tensor(minimum_factor_std) - factor_std))
    orthogonality_loss_terms = []
    factor_signatures = []
    for factor_tensor in factor_tensors:
        flattened = factor_tensor.reshape(factor_tensor.shape[0], -1)
        sample_signature = flattened.mean(dim=1)
        centered_signature = sample_signature - sample_signature.mean()
        factor_signatures.append(torch.nn.functional.normalize(centered_signature.unsqueeze(0), dim=1).squeeze(0))
    for left_index in range(len(factor_signatures)):
        for right_index in range(left_index + 1, len(factor_signatures)):
            similarity = (factor_signatures[left_index] * factor_signatures[right_index]).sum()
            orthogonality_loss_terms.append(similarity ** 2)
    variance_loss = torch.stack(variance_loss_terms).mean()
    orthogonality_loss = torch.stack(orthogonality_loss_terms).mean() if orthogonality_loss_terms else variance_loss.new_tensor(0.0)
    hidden_state_weight = hidden_state_support.float().mean() + hidden_state_weight_scale * missing_modality_count.float().mean()
    return hidden_state_weight * (variance_loss + orthogonality_weight * orthogonality_loss)


def world_state_target_loss(
    world_state_payload: dict[str, torch.Tensor],
    factor_targets: dict[str, torch.Tensor],
    factor_masks: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    predicted_strengths = {
        "z_tx": world_state_payload["z_tx"].abs().mean(dim=1),
        "z_pk": world_state_payload["z_pk"].abs().mean(dim=1),
        "z_env": world_state_payload["z_env"].abs().mean(dim=1),
        "z_ephys": world_state_payload["z_ephys"].abs().mean(dim=1),
        "z_morph": world_state_payload["z_morph"].abs().mean(dim=1),
    }
    losses = []
    for factor_name, predicted_strength in predicted_strengths.items():
        target_strength = factor_targets[factor_name]
        if factor_masks is None:
            losses.append(torch.nn.functional.mse_loss(predicted_strength, target_strength))
            continue
        factor_mask = factor_masks[factor_name].to(device=predicted_strength.device, dtype=predicted_strength.dtype)
        if float(factor_mask.sum().item()) <= 0.0:
            continue
        squared_error = (predicted_strength - target_strength) ** 2
        losses.append((squared_error * factor_mask).sum() / factor_mask.sum())
    if not losses:
        return world_state_payload["z_tx"].new_tensor(0.0)
    return torch.stack(losses).mean()


def teacher_weighted_rollout_mse(
    prediction: torch.Tensor, target: torch.Tensor, teacher_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the native teacher weight before reducing sample/action axes."""
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise ValueError("Teacher rollout tensors require matching sample/action/time/region/compartment/species axes.")
    per_action_loss = (prediction - target).square().flatten(2).mean(-1)
    if teacher_weight is None:
        return per_action_loss.mean()
    weight = teacher_weight.to(per_action_loss).detach()
    if weight.shape != per_action_loss.shape or not bool((torch.isfinite(weight) & (weight >= 0)).all()):
        raise ValueError("Teacher weight must be finite, nonnegative and sample/action-aligned.")
    # Keep the native categorical objective's fixed sample/action denominator.
    return (per_action_loss * weight).mean()


def simulation_teacher_factor_loss(
    predicted_field_trajectory: torch.Tensor,
    simulation_targets: torch.Tensor,
    factor_masks: dict[str, torch.Tensor],
    config: dict,
    *, teacher_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher_config = config["screening_pipeline"]["screening_benchmark"]["foundation_training"]["simulation_teacher"]
    factor_to_species_index = {
        "z_pk": [0, 1],
        "z_env": [2, 3, 4, 5],
        "z_ephys": [6],
    }
    losses = []
    for factor_name, species_indices in factor_to_species_index.items():
        factor_weight = float(teacher_config[f"{factor_name}_weight"])
        factor_mask = factor_masks[factor_name].to(device=predicted_field_trajectory.device, dtype=predicted_field_trajectory.dtype)
        if float(factor_mask.sum().item()) <= 0.0:
            continue
        regimen_indices = torch.nonzero(factor_mask > 0.0, as_tuple=False).squeeze(-1)
        species_index_tensor = torch.tensor(species_indices, dtype=torch.long, device=predicted_field_trajectory.device)
        selected_prediction = predicted_field_trajectory.index_select(1, regimen_indices).index_select(-1, species_index_tensor)
        selected_target = simulation_targets.index_select(1, regimen_indices).index_select(-1, species_index_tensor)
        if selected_prediction.numel() == 0 or selected_target.numel() == 0:
            continue
        selected_weight = None if teacher_weight is None else teacher_weight.index_select(1, regimen_indices)
        losses.append(factor_weight * teacher_weighted_rollout_mse(selected_prediction, selected_target, selected_weight))
    if not losses:
        return predicted_field_trajectory.new_tensor(0.0)
    return torch.stack(losses).mean()


def counterfactual_pairwise_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    pair_indices: list[tuple[int, int]],
    config: dict,
) -> torch.Tensor:
    if not pair_indices:
        return predicted_scores.new_tensor(0.0)
    margin = float(config["screening_pipeline"]["screening_benchmark"]["foundation_training"]["counterfactual_planning"]["pairwise_margin"])
    losses = []
    for baseline_index, counterfactual_index in pair_indices:
        predicted_margin = predicted_scores[:, counterfactual_index] - predicted_scores[:, baseline_index]
        target_margin = target_scores[:, counterfactual_index] - target_scores[:, baseline_index]
        target_direction = torch.sign(target_margin)
        active_mask = target_direction.ne(0.0)
        if not bool(active_mask.any()):
            continue
        losses.append(torch.relu(margin - predicted_margin[active_mask] * target_direction[active_mask]).mean())
    if not losses:
        return predicted_scores.new_tensor(0.0)
    return torch.stack(losses).mean()
