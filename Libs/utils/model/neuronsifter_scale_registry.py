from __future__ import annotations

from copy import deepcopy
from typing import Any

from Libs.utils.model.transolver_parameter_budget import neuronsifter_world_action_active_parameters
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config


def registry(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["neuronsifter_scale_registry"])


def transition_backbone(config: dict[str, Any]) -> dict[str, Any]:
    return config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["screening_pipeline"]["screening_benchmark"]["training"]


def checkpoint_scale_config(config: dict[str, Any], checkpoint_scale: str) -> dict[str, Any]:
    scale_map = registry(config)["checkpoint_scales"]
    if checkpoint_scale not in scale_map:
        raise ValueError(f"Unknown NeuronSifter checkpoint scale: {checkpoint_scale}")
    return dict(scale_map[checkpoint_scale])


def inference_profile_config(config: dict[str, Any], inference_profile: str) -> dict[str, Any]:
    profile_map = registry(config)["inference_profiles"]
    if inference_profile not in profile_map:
        raise ValueError(f"Unknown NeuronSifter inference profile: {inference_profile}")
    return dict(profile_map[inference_profile])


def default_checkpoint_scale(config: dict[str, Any], smoke: bool = False) -> str:
    registry_config = registry(config)
    if smoke:
        return str(registry_config["smoke_checkpoint_scale"])
    return str(registry_config["default_training_checkpoint_scale"])


def default_inference_profile(config: dict[str, Any], checkpoint_scale: str) -> str:
    return str(checkpoint_scale_config(config, checkpoint_scale)["default_inference_profile"])


def canonical_model_name(config: dict[str, Any], checkpoint_scale: str | None = None) -> str:
    selected_scale = str(checkpoint_scale or default_checkpoint_scale(config))
    return str(checkpoint_scale_config(config, selected_scale)["scale_label"])


def scale_parameter_estimate(config: dict[str, Any], checkpoint_scale: str) -> int:
    backbone = transition_backbone(config)
    scale_config = checkpoint_scale_config(config, checkpoint_scale)
    mechanism_moe = backbone["mechanism_moe"]
    return neuronsifter_world_action_active_parameters(
        int(scale_config["operator_hidden_dim"]),
        int(scale_config["operator_depth"]),
        int(scale_config["attention_head_count"]),
        int(scale_config["mechanism_state_token_count"]),
        int(backbone["node_coordinate_dim"]),
        len(config["dynamics_ready_registry"]["species"]),
        9,
        int(training_config(config)["latent_dim"]) * 3,
        len(mechanism_moe["expert_names"]),
        int(mechanism_moe["top_k"]),
        int(mechanism_moe["expert_hidden_multiplier"]),
        int(backbone["dit_mlp_multiplier"]),
        int(backbone["operator_splitting"]["bottleneck_divisor"]),
    )


def apply_teacher_mixture_weights(config: dict[str, Any]) -> None:
    mixture_config = registry(config)["teacher_mixture_sweep"]
    weights = {
        str(family_key): float(weight_value)
        for family_key, weight_value in dict(mixture_config["default_family_weights"]).items()
    }
    for family in training_config(config)["teacher_family_training"]["families"]:
        family_key = str(family["family_key"])
        if family_key in weights:
            family["teacher_loss_weight"] = weights[family_key]


def apply_neuronsifter_scale(
    config: dict[str, Any],
    checkpoint_scale: str,
    inference_profile: str | None = None,
) -> dict[str, Any]:
    scaled_config = deepcopy(config)
    scale_config = checkpoint_scale_config(scaled_config, checkpoint_scale)
    selected_profile = str(inference_profile or scale_config["default_inference_profile"])
    profile_config = inference_profile_config(scaled_config, selected_profile)
    if str(profile_config["checkpoint_scale"]) != checkpoint_scale:
        raise ValueError(
            "NeuronSifter inference profile checkpoint scale mismatch: "
            f"{selected_profile} declares {profile_config['checkpoint_scale']} but requested {checkpoint_scale}."
        )
    backbone = transition_backbone(scaled_config)
    backbone["operator_hidden_dim"] = int(scale_config["operator_hidden_dim"])
    backbone["operator_depth"] = int(scale_config["operator_depth"])
    backbone["attention_head_count"] = int(scale_config["attention_head_count"])
    backbone["mechanism_state_token_count"] = int(scale_config["mechanism_state_token_count"])
    backbone["graph_spectral_tokenizer"]["graph_mode_count"] = int(scale_config["token_resolution"]["graph_mode_count"])
    backbone["temporal_wavelet_tokenizer"]["wavelet_level_count"] = int(scale_config["token_resolution"]["wavelet_level_count"])
    backbone["parameter_budget"] = dict(scale_config["parameter_budget"])
    sampler_budget = dict(profile_config["sampler_budget"])
    for key, value in sampler_budget.items():
        backbone["diffusion_sampler"][key] = value
    training_config(scaled_config)["memory_policy"] = dict(scale_config["training_memory_policy"])
    training_config(scaled_config)["batch_size"] = int(scale_config["training_memory_policy"]["effective_batch_size"])
    training_config(scaled_config)["selected_checkpoint_scale"] = checkpoint_scale
    training_config(scaled_config)["selected_inference_profile"] = selected_profile
    training_config(scaled_config)["selected_fsdp_resource_profile"] = str(scale_config["fsdp_resource_profile"])
    resource_profile = registry(scaled_config)["fsdp_resource_profiles"][str(scale_config["fsdp_resource_profile"])]
    loader_config = training_config(scaled_config)["zarr_loader"]
    if "max_epochs" in resource_profile:
        loader_config["pretraining_epoch_count"] = int(resource_profile["max_epochs"])
    if "batches_per_epoch" in resource_profile:
        loader_config["pretraining_batches_per_epoch"] = int(resource_profile["batches_per_epoch"])
    scaled_config["neuronsifter_scale_selection"] = {
        "checkpoint_scale": checkpoint_scale,
        "scale_label": str(scale_config["scale_label"]),
        "inference_profile": selected_profile,
        "profile_route_priority": str(profile_config["route_priority"]),
        "fsdp_resource_profile": str(scale_config["fsdp_resource_profile"]),
        "nominal_active_parameter_count": int(scale_config["nominal_active_parameter_count"]),
        "nominal_trainable_parameter_count": int(scale_config["nominal_trainable_parameter_count"]),
        "analytic_active_parameter_count": scale_parameter_estimate(scaled_config, checkpoint_scale),
        "analytic_parameter_scope": "transition_backbone_excludes_categorical_projections",
        "executed_capacity_status": "NOT_MEASURED",
    }
    # Expose the actual selected categorical budget alongside the existing scale report.
    effective_mddm = quantized_mddm_rollout_from_config(scaled_config, backbone)
    scaled_config["neuronsifter_scale_selection"]["quantized_mddm_rollout"] = effective_mddm
    apply_teacher_mixture_weights(scaled_config)
    return scaled_config


def fsdp_profile_config(config: dict[str, Any], checkpoint_scale: str) -> dict[str, Any]:
    scale_config = checkpoint_scale_config(config, checkpoint_scale)
    profile_key = str(scale_config["fsdp_resource_profile"])
    profile_map = registry(config)["fsdp_resource_profiles"]
    if profile_key not in profile_map:
        raise ValueError(f"NeuronSifter FSDP profile is missing: {profile_key}")
    return dict(profile_map[profile_key])


def scale_route_metadata(
    config: dict[str, Any],
    inference_profile: str,
    checkpoint_id: str,
    escalation_status: str,
    student_large_disagreement: float | None,
    large_xl_disagreement: float | None,
) -> dict[str, Any]:
    profile_config = inference_profile_config(config, inference_profile)
    checkpoint_scale = str(profile_config["checkpoint_scale"])
    scale_config = checkpoint_scale_config(config, checkpoint_scale)
    routing_policy = registry(config)["routing_policy"]
    return {
        "model_name": canonical_model_name(config, checkpoint_scale),
        "profile_used": inference_profile,
        "checkpoint_id": checkpoint_id,
        "scale_route": {
            "checkpoint_scale": checkpoint_scale,
            "scale_label": str(scale_config["scale_label"]),
            "route_priority": str(profile_config["route_priority"]),
            "sampler_budget": dict(profile_config["sampler_budget"]),
            "disagreement_source": str(routing_policy["disagreement_source_when_single_checkpoint"]),
        },
        "escalation_status": escalation_status,
        "student_large_disagreement": student_large_disagreement,
        "large_xl_disagreement": large_xl_disagreement,
    }
