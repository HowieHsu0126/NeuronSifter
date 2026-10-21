from __future__ import annotations

from copy import deepcopy

from Libs.utils.model.dynamics_fields import default_region_prior
from Libs.utils.data.joint_physical_parameters import parameter_schema_from_config
from Libs.models.modules.native_linear_dynamics import LINEAR_DYNAMICS_SPEC


def native_direct_mean_config(config: dict) -> dict:
    """Explicit direct profiles; disabled legacy Transolver remains unchanged."""
    rollout = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    backbone = rollout["transition_backbone"]
    policy = dict(backbone.get("native_direct_mean", {}))
    if not policy.get("enabled", False):
        if backbone["kind"] == "native_linear_dynamics":
            raise ValueError("Linear dynamics requires its enabled native direct profile")
        return {}
    profiles = {"transolver_microenvironment_operator": "rollout_operator_transolver",
                "native_linear_dynamics": "rollout_operator_linear_dynamics"}
    if (backbone["kind"] not in profiles or config.get("selected_experiment_family") != profiles[backbone["kind"]]
            or rollout["action_representation"]["active_policy"] != "occupancy_program"):
        raise ValueError("Native direct mean requires its explicit occupancy predictor profile")
    if backbone["kind"] == "native_linear_dynamics":
        if policy.get("linear_dynamics") != LINEAR_DYNAMICS_SPEC:
            raise ValueError("Linear dynamics equation/discretization/numerical identity mismatch")
        if policy.get("router_regularization_weight") != 0:
            raise ValueError("Linear dynamics has no router and requires zero router weight")
    if (policy.get("prediction_kind") != "mean_only"
            or policy.get("residual_likelihood_status") != "unbound"
            or policy.get("objective") != "fixed_training_normalized_mse"):
        raise ValueError("Native direct predictors support only an unbound-residual direct mean")
    width = policy.get("posterior_context_dim")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("Native direct mean requires the source-bound posterior_context_dim")
    for key in ("conditioning_config_path", "conditioning_checkpoint_path"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise ValueError("Native direct mean requires " + key)
    for key in ("loss_weight", "router_regularization_weight"):
        value = policy.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
            raise ValueError("Native direct mean requires a finite nonnegative " + key)
    return policy


def native_transolver_mean_config(config: dict) -> dict:
    """Retained Transolver-specific admission for legacy/residual consumers."""
    policy = native_direct_mean_config(config)
    kind = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"]
    if policy and kind != "transolver_microenvironment_operator":
        raise ValueError("Transolver residual/configuration authority cannot authorize linear dynamics")
    return policy


def species_ids_from_config(config: dict) -> list[str]:
    return [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]


def region_count_from_config(config: dict) -> int:
    return len(config["spatial_atlas_registry"]["regions"])


def species_count_from_config(config: dict) -> int:
    return len(config["dynamics_ready_registry"]["species"])


def baseline_field_policy_from_config(config: dict) -> dict[str, object]:
    species_ids = species_ids_from_config(config)
    policy_config = (
        config["screening_pipeline"]["screening_benchmark"]
        .get("model_architecture", {})
        .get("sample_evidence", {})
        .get("baseline_field_policy", {})
    )

    def values_from_species_map(field_name: str) -> list[float]:
        species_map = policy_config.get(field_name, {})
        missing_species = [species_id for species_id in species_ids if species_id not in species_map]
        if missing_species:
            raise ValueError(f"Missing baseline_field_policy values for {field_name}: {missing_species}")
        return [float(species_map[species_id]) for species_id in species_ids]

    if "region_prior_mix" not in policy_config:
        raise ValueError("baseline_field_policy.region_prior_mix is required in config.")

    return {
        "direct_scale": values_from_species_map("direct_scale_by_species_id"),
        "offset": values_from_species_map("offset_by_species_id"),
        "pathology_gain": values_from_species_map("pathology_gain_by_species_id"),
        "raw_field_gain": values_from_species_map("raw_field_gain_by_species_id"),
        "region_prior_mix": float(policy_config["region_prior_mix"]),
    }


def rollout_update_policy_from_config(config: dict) -> dict[str, object]:
    operator_config = (
        config["screening_pipeline"]["screening_benchmark"]
        .get("model_architecture", {})
        .get("rollout_operator", {})
    )
    required_keys = [
        "state_delta_scale",
        "control_source_scale",
        "region_prior_mix",
        "activation_checkpointing",
        "action_representation",
        "transition_backbone",
    ]
    missing_keys = [key for key in required_keys if key not in operator_config]
    if missing_keys:
        raise ValueError(f"Missing rollout_operator config fields: {missing_keys}")
    action_config = dict(operator_config["action_representation"])
    action_required_keys = ["active_policy", "feature_groups", "policies"]
    missing_action_keys = [key for key in action_required_keys if key not in action_config]
    if missing_action_keys:
        raise ValueError(f"Missing rollout_operator.action_representation fields: {missing_action_keys}")
    active_policy = str(action_config["active_policy"])
    action_policies = dict(action_config["policies"])
    if active_policy not in action_policies:
        raise ValueError(f"Unknown action representation policy: {active_policy}")
    feature_groups = {
        str(group_name): [int(index) for index in group_indices]
        for group_name, group_indices in dict(action_config["feature_groups"]).items()
    }
    policy_payload = {}
    for policy_name, policy_config in action_policies.items():
        retained_groups = [str(group_name) for group_name in dict(policy_config)["retained_feature_groups"]]
        missing_groups = sorted(set(retained_groups) - set(feature_groups))
        if missing_groups:
            raise ValueError(f"Action policy {policy_name} references missing feature groups: {missing_groups}")
        policy_payload[str(policy_name)] = {"retained_feature_groups": retained_groups}
        if "native_definition" in policy_config:
            policy_payload[str(policy_name)]["native_definition"] = dict(policy_config["native_definition"])
    transition_backbone = dict(operator_config["transition_backbone"])
    if transition_backbone["kind"] == "native_linear_dynamics":
        raise ValueError("Linear dynamics is supported only by native direct pretraining/evaluation, not screening/serving")
    required_backbone_keys = ["kind", "operator_hidden_dim", "operator_depth", "operator_dropout_rate", "operator_residual_scale"]
    missing_backbone_keys = [key for key in required_backbone_keys if key not in transition_backbone]
    if missing_backbone_keys:
        raise ValueError(f"Missing rollout_operator.transition_backbone fields: {missing_backbone_keys}")
    backbone_payload = {
        "kind": str(transition_backbone["kind"]),
        "operator_hidden_dim": int(transition_backbone["operator_hidden_dim"]),
        "operator_depth": int(transition_backbone["operator_depth"]),
        "operator_dropout_rate": float(transition_backbone["operator_dropout_rate"]),
        "operator_residual_scale": float(transition_backbone["operator_residual_scale"]),
    }
    if backbone_payload["kind"] not in {
        "factor_coupled_neural_operator",
        "neuronsifter_world_action",
        "transolver_microenvironment_operator",
    }:
        raise ValueError(f"Unsupported rollout operator kind: {backbone_payload['kind']}")
    if backbone_payload["kind"] == "transolver_microenvironment_operator":
        required_transolver_keys = [
            "attention_head_count",
            "physical_state_slice_count",
            "node_coordinate_dim",
            "parameter_budget",
            "encoder_decoder_pretraining",
            "physics_mlp_multiplier",
            "mechanism_moe",
        ]
        missing_transolver_keys = [key for key in required_transolver_keys if key not in transition_backbone]
        if missing_transolver_keys:
            raise ValueError(f"Missing transolver_microenvironment_operator config fields: {missing_transolver_keys}")
        mechanism_moe = dict(transition_backbone["mechanism_moe"])
        required_moe_keys = [
            "expert_names",
            "top_k",
            "capacity_factor",
            "router_loss_weight",
            "entropy_loss_weight",
            "load_balance_loss_weight",
            "expert_capacity_loss_weight",
            "expert_hidden_multiplier",
        ]
        missing_moe_keys = [key for key in required_moe_keys if key not in mechanism_moe]
        if missing_moe_keys:
            raise ValueError(f"Missing mechanism_moe config fields: {missing_moe_keys}")
        backbone_payload.update(
            {
                "attention_head_count": int(transition_backbone["attention_head_count"]),
                "physical_state_slice_count": int(transition_backbone["physical_state_slice_count"]),
                "node_coordinate_dim": int(transition_backbone["node_coordinate_dim"]),
                "parameter_budget": dict(transition_backbone["parameter_budget"]),
                "encoder_decoder_pretraining": dict(transition_backbone["encoder_decoder_pretraining"]),
                "physics_mlp_multiplier": int(transition_backbone["physics_mlp_multiplier"]),
                "mechanism_moe": {
                    "expert_names": [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
                    "top_k": int(mechanism_moe["top_k"]),
                    "capacity_factor": float(mechanism_moe["capacity_factor"]),
                    "router_loss_weight": float(mechanism_moe["router_loss_weight"]),
                    "entropy_loss_weight": float(mechanism_moe["entropy_loss_weight"]),
                    "load_balance_loss_weight": float(mechanism_moe["load_balance_loss_weight"]),
                    "expert_capacity_loss_weight": float(mechanism_moe["expert_capacity_loss_weight"]),
                    "expert_hidden_multiplier": int(mechanism_moe["expert_hidden_multiplier"]),
                },
            }
        )
    if backbone_payload["kind"] == "neuronsifter_world_action":
        required_neuronsifter_keys = [
            "attention_head_count",
            "mechanism_state_token_count",
            "node_coordinate_dim",
            "dit_mlp_multiplier",
            "mechanism_slice_temperature",
            "graph_spectral_tokenizer",
            "temporal_wavelet_tokenizer",
            "diffusion_sampler",
            "operator_splitting",
            "parameter_budget",
            "mechanism_moe",
        ]
        missing_neuronsifter_keys = [key for key in required_neuronsifter_keys if key not in transition_backbone]
        if missing_neuronsifter_keys:
            raise ValueError(f"Missing neuronsifter_world_action config fields: {missing_neuronsifter_keys}")
        mechanism_moe = dict(transition_backbone["mechanism_moe"])
        required_moe_keys = [
            "expert_names",
            "top_k",
            "capacity_factor",
            "router_loss_weight",
            "entropy_loss_weight",
            "load_balance_loss_weight",
            "expert_capacity_loss_weight",
            "expert_hidden_multiplier",
        ]
        missing_moe_keys = [key for key in required_moe_keys if key not in mechanism_moe]
        if missing_moe_keys:
            raise ValueError(f"Missing mechanism_moe config fields: {missing_moe_keys}")
        graph_config = dict(transition_backbone["graph_spectral_tokenizer"])
        wavelet_config = dict(transition_backbone["temporal_wavelet_tokenizer"])
        sampler_config = dict(transition_backbone["diffusion_sampler"])
        operator_splitting = dict(transition_backbone["operator_splitting"])
        required_operator_splitting_keys = [
            "enabled",
            "bottleneck_divisor",
            "expert_residual_scale",
            "shared_coupling_residual_scale",
            "composition_order",
            "coupling_masks",
        ]
        missing_operator_splitting_keys = [key for key in required_operator_splitting_keys if key not in operator_splitting]
        if missing_operator_splitting_keys:
            raise ValueError(f"Missing operator_splitting config fields: {missing_operator_splitting_keys}")
        composition_mode = str(operator_splitting.get("composition_mode", "serial"))
        if composition_mode not in {"serial", "parallel"}:
            raise ValueError(f"Unknown operator_splitting.composition_mode: {composition_mode}")
        names, order = list(mechanism_moe["expert_names"]), list(operator_splitting["composition_order"])
        if (
            not names or len(set(names)) != len(names) or len(order) != len(names) or set(order) != set(names)
        ):
            raise ValueError("Expert composition requires the complete unique configured expert roster once.")
        profile_name = config.get("selected_experiment_family", config.get("selected_paper_experiment_profile", ""))
        profile = config.get("paper_experiments", {}).get(profile_name, {})
        if profile.get("expert_order_control", False):
            declared = profile["config_overrides"]["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
            if (names != declared["mechanism_moe"]["expert_names"]
                    or order != declared["operator_splitting"]["composition_order"] or composition_mode != "serial"):
                raise ValueError("Named expert-order control differs from its declared canonical roster or exact serial order.")
        quantized_mddm_rollout = quantized_mddm_rollout_from_config(config, transition_backbone)
        backbone_payload.update(
            {
                "attention_head_count": int(transition_backbone["attention_head_count"]),
                "mechanism_state_token_count": int(transition_backbone["mechanism_state_token_count"]),
                "node_coordinate_dim": int(transition_backbone["node_coordinate_dim"]),
                "dit_mlp_multiplier": int(transition_backbone["dit_mlp_multiplier"]),
                "mechanism_slice_temperature": float(transition_backbone["mechanism_slice_temperature"]),
                "graph_spectral_tokenizer": {
                    "graph_mode_count": int(graph_config["graph_mode_count"]),
                    "eigenvalue_temperature": float(graph_config["eigenvalue_temperature"]),
                    "normalize_laplacian": bool(graph_config["normalize_laplacian"]),
                    "spectral_residual_scale": float(graph_config["spectral_residual_scale"]),
                },
                "temporal_wavelet_tokenizer": {
                    "wavelet_family": str(wavelet_config["wavelet_family"]),
                    "wavelet_level_count": int(wavelet_config["wavelet_level_count"]),
                    "boundary_mode": str(wavelet_config["boundary_mode"]),
                },
                "diffusion_sampler": {
                    "sample_count": int(sampler_config["sample_count"]),
                    "denoising_step_count": int(sampler_config["denoising_step_count"]),
                    "sample_chunk_size": int(sampler_config["sample_chunk_size"]),
                    "training_sample_count": int(sampler_config["training_sample_count"]),
                    "training_denoising_step_count": int(sampler_config["training_denoising_step_count"]),
                    "diffusion_noise_scale": float(sampler_config["diffusion_noise_scale"]),
                    "score_step_size": float(sampler_config["score_step_size"]),
                    "terminal_noise_scale": float(sampler_config["terminal_noise_scale"]),
                    "ranking_stability_scale": float(sampler_config["ranking_stability_scale"]),
                    "quantile_levels": [float(value) for value in sampler_config["quantile_levels"]],
                    "sampler_seed": int(sampler_config["sampler_seed"]),
                    "return_sample_trajectories": bool(sampler_config["return_sample_trajectories"]),
                },
                "parameter_budget": dict(transition_backbone["parameter_budget"]),
                "mechanism_moe": {
                    "expert_names": [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
                    "top_k": int(mechanism_moe["top_k"]),
                    "capacity_factor": float(mechanism_moe["capacity_factor"]),
                    "router_loss_weight": float(mechanism_moe["router_loss_weight"]),
                    "entropy_loss_weight": float(mechanism_moe["entropy_loss_weight"]),
                    "load_balance_loss_weight": float(mechanism_moe["load_balance_loss_weight"]),
                    "expert_capacity_loss_weight": float(mechanism_moe["expert_capacity_loss_weight"]),
                    "expert_hidden_multiplier": int(mechanism_moe["expert_hidden_multiplier"]),
                },
                "operator_splitting": {
                    "enabled": bool(operator_splitting["enabled"]),
                    "composition_mode": composition_mode,
                    "bottleneck_divisor": int(operator_splitting["bottleneck_divisor"]),
                    "expert_residual_scale": float(operator_splitting["expert_residual_scale"]),
                    "shared_coupling_residual_scale": float(operator_splitting["shared_coupling_residual_scale"]),
                    "composition_order": [str(expert_name) for expert_name in operator_splitting["composition_order"]],
                    "coupling_masks": {
                        str(expert_name): [str(state_name) for state_name in state_names]
                        for expert_name, state_names in dict(operator_splitting["coupling_masks"]).items()
                    },
                },
            }
        )
        if quantized_mddm_rollout is not None:
            backbone_payload["quantized_mddm_rollout"] = quantized_mddm_rollout
    return {
        "state_delta_scale": float(operator_config["state_delta_scale"]),
        "control_source_scale": float(operator_config["control_source_scale"]),
        "region_prior_mix": float(operator_config["region_prior_mix"]),
        "activation_checkpointing": bool(operator_config["activation_checkpointing"]),
        "action_representation": {
            "active_policy": active_policy,
            "feature_groups": feature_groups,
            "policies": policy_payload,
        },
        "transition_backbone": backbone_payload,
    }


def quantized_mddm_rollout_from_config(config: dict, transition_backbone: dict) -> dict[str, object] | None:
    local_config = dict(transition_backbone.get("quantized_mddm_rollout", {}))
    registry_binding = str(local_config.get("registry_binding", "quantized_mddm_main_rollout"))
    registry_payload = dict(config.get("neuronsifter_scale_registry", {}).get(registry_binding, {}))
    if not local_config and not registry_payload:
        return None
    if local_config and not registry_payload and "residual_tokenization" not in local_config:
        raise ValueError(f"Missing MDDM rollout registry binding: {registry_binding}")
    rollout_payload = deepcopy(registry_payload or local_config)
    # A bound registry owns architecture/objective. Explicit runtime overrides
    # use a separate narrow namespace; arbitrary conflicting local copies fail.
    if registry_payload:
        metadata = {"enabled", "registry_binding", "role", "continuous_residual_sampler_role", "runtime_overrides"}
        for key, value in local_config.items():
            if key not in metadata and (key not in registry_payload or value != registry_payload[key]):
                raise ValueError(f"Conflicting registry-bound MDDM field: {key}; change the canonical registry.")
    if "enabled" in local_config:
        rollout_payload["enabled"] = bool(local_config["enabled"])
    selected = config.get("neuronsifter_scale_selection", {}).get("inference_profile")
    if selected:
        budget = config["neuronsifter_scale_registry"]["inference_profiles"][selected]["sampler_budget"]
        aliases = {"sample_count": "inference_sample_count", "denoising_step_count": "inference_denoising_step_count"}
        for key in ("sample_count", "denoising_step_count", "sample_chunk_size", "training_sample_count", "training_denoising_step_count"):
            if key in budget:
                rollout_payload["partial_mask_schedule"][aliases.get(key, key)] = budget[key]
    runtime = local_config.get("runtime_overrides", {})
    allowed = {"sample_chunk_size", "denoising_token_chunk_size", "sampler_seed", "return_sample_trajectories"}
    unknown = set(runtime) - allowed
    if unknown:
        raise ValueError(f"Unsupported MDDM runtime overrides: {sorted(unknown)}")
    for key, value in runtime.items():
        section = "inverse_reconstruction_policy" if key == "return_sample_trajectories" else "partial_mask_schedule"
        rollout_payload[section][key] = value
    required_keys = [
        "enabled",
        "residual_reference_policy",
        "training_objective",
        "training_time_distribution",
        "external_reference_policy",
        "token_axes",
        "residual_tokenization",
        "partial_mask_schedule",
        "inverse_reconstruction_policy",
        "loss_weights",
    ]
    missing_keys = [key for key in required_keys if key not in rollout_payload]
    if missing_keys:
        raise ValueError(f"Missing quantized_mddm_rollout config fields: {missing_keys}")
    if rollout_payload["enabled"]:
        if rollout_payload["residual_reference_policy"] != "initial_state":
            raise ValueError("Native MDDM requires the fixed initial-state reference.")
        if rollout_payload["training_time_distribution"] != "uniform_open_unit_interval":
            raise ValueError("Native MDDM requires uniform open-unit diffusion times.")
        if rollout_payload["training_objective"] != "absorbing_masked_categorical_nonempty_importance":
            raise ValueError("Native MDDM requires the importance-corrected masked objective.")
        schedule = rollout_payload["partial_mask_schedule"]
        if schedule["schedule_kind"] != "absorbing_linear_reverse" or schedule["inference_initial_mask_probability"] != 1.0 or schedule["final_mask_probability"] != 0.0:
            raise ValueError("Native MDDM requires fully masked categorical reverse sampling ending at zero masks.")
    return rollout_payload


def solver_state_layout_from_config(config: dict) -> dict[str, int]:
    return {
        species_id: index
        for index, species_id in enumerate(species_ids_from_config(config))
    }


def solver_config_from_config(config: dict) -> dict[str, object]:
    priors = config["dynamics_ready_registry"]["field_dynamics_priors"]
    compartments = [str(row["compartment_id"]) for row in config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]]
    condition_profiles = config["dynamics_ready_registry"]["condition_profiles"]
    species_ids = species_ids_from_config(config)
    rollout_operator = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    regimen_feature_layout = {
        str(key): int(value)
        for key, value in dict(rollout_operator["regimen_feature_layout"]).items()
    }
    required_species = ["misfolded_protein_load", "aggregate_burden", "neuroinflammation", "circuit_instability"]
    missing_upper_bounds = [
        species_id
        for species_id in required_species[2:]
        if species_id not in priors["state_upper_bound_by_species"]
    ]
    if missing_upper_bounds:
        raise ValueError(f"Missing state_upper_bound_by_species entries: {missing_upper_bounds}")
    return {
        "diffusion_scale_by_compartment": [float(priors["diffusion_scale_by_compartment"][compartment_id]) for compartment_id in compartments],
        "clearance_scale_by_compartment": [float(priors["clearance_scale_by_compartment"][compartment_id]) for compartment_id in compartments],
        "route_source_by_compartment": {
            route_name: [float(priors["route_source_by_compartment"][route_name][compartment_id]) for compartment_id in compartments]
            for route_name in ["oral", "intravenous", "intrathecal"]
        },
        "schedule_frequency_multipliers": {
            schedule_name: float(priors["schedule_frequency_multiplier"][schedule_name])
            for schedule_name in ["daily", "twice_daily", "weekly"]
        },
        "target_binding_gain": float(priors["target_binding_gain"]),
        "amyloid_reduction_gain": float(priors["amyloid_reduction_gain"]),
        "plaque_clearance_gain": float(priors["plaque_clearance_gain"]),
        "electrophysiology_relief_gain": float(priors["excitability_relief_gain"]),
        "electrophysiology_amyloid_gain": float(priors["excitability_amyloid_gain"]),
        "electrophysiology_plaque_gain": float(priors["excitability_plaque_gain"]),
        "minimum_bound_fraction": float(priors["minimum_bound_fraction"]),
        "maximum_bound_fraction": float(priors["maximum_bound_fraction"]),
        "bound_decay_rate": float(priors["bound_decay_rate"]),
        "parameter_context_scale_by_species": {
            species_id: float(priors["parameter_context_scale_by_species"][species_id])
            for species_id in species_ids
            if species_id in priors["parameter_context_scale_by_species"]
        },
        "prior_context_scale_by_drive": {
            "pathology": float(priors["prior_context_scale_by_drive"]["pathology"]),
            "inflammation": float(priors["prior_context_scale_by_drive"]["inflammation"]),
        },
        "aggregate_clearance_scale": float(priors["aggregate_clearance_scale"]),
        "inflammation_coupling": {
            species_id: float(value)
            for species_id, value in dict(priors["inflammation_coupling"]).items()
        },
        "inflammation_decay": float(priors["inflammation_decay"]),
        "clearance_flux_scale": {
            scale_name: float(value)
            for scale_name, value in dict(priors["clearance_flux_scale"]).items()
        },
        "circuit_instability_scale": {
            scale_name: float(value)
            for scale_name, value in dict(priors["circuit_instability_scale"]).items()
        },
        "state_upper_bound_by_species": {
            species_id: float(priors["state_upper_bound_by_species"][species_id])
            for species_id in species_ids
            if species_id in priors["state_upper_bound_by_species"]
        },
        "mesoscopic_solver_coefficients": dict(priors["mesoscopic_solver_coefficients"]),
        "full_microenvironment_coupling": dict(priors["mesoscopic_solver_coefficients"]["full_microenvironment_coupling"]),
        "osp_teacher_coupling": dict(priors["osp_teacher_coupling"]),
        "neuron_rxd_teacher_coupling": dict(priors["neuron_rxd_teacher_coupling"]),
        "condition_rate_bounds": {
            "control": {
                "misfolded_protein_production_rate": float(condition_profiles["control"]["misfolded_protein_production_rate"]),
                "aggregate_accumulation_rate": float(condition_profiles["control"]["aggregate_accumulation_rate"]),
            },
            "ad_positive": {
                "misfolded_protein_production_rate": float(condition_profiles["ad_positive"]["misfolded_protein_production_rate"]),
                "aggregate_accumulation_rate": float(condition_profiles["ad_positive"]["aggregate_accumulation_rate"]),
            },
        },
        "regimen_feature_layout": regimen_feature_layout,
    }



def observation_feature_profile(config: dict) -> dict[str, object]:
    """Inspect a declared observation interface without source data or model allocation."""
    posterior = world_state_posterior_config_from_config(config)
    family, width = posterior["family"], posterior["observation_feature_width"]
    if family != "finite_joint_particles":
        return {"status": "not_required", "posterior_family": family, "observation_feature_width": width}
    if width is None:
        return {"status": "missing_profile", "posterior_family": family,
                "observation_feature_width": None,
                "reason": "finite_joint_particles_requires_explicit_observation_feature_width"}
    if type(width) is not int or width < 1:
        raise ValueError("Declared observation_feature_width must be a positive integer")
    manuscript = config.get("manuscript_native_protocol")
    profile = "explicit_observation_interface"
    if manuscript is not None:
        profile = "manuscript_native_protocol"
        blocks = config.get("native_state_comparison", {}).get("readout_blocks")
        if (not isinstance(blocks, list) or not blocks
                or any(not isinstance(block, list) or not block for block in blocks)):
            raise ValueError("Manuscript observation profile requires registered readout blocks")
        indices = [index for block in blocks for index in block]
        if (any(type(index) is not int for index in indices)
                or sorted(indices) != list(range(width)) or len(set(indices)) != len(indices)):
            raise ValueError("Manuscript readout blocks must partition the declared observation_feature_width")
        bridge = config.get("native_observable_bridge", {})
        if not bridge.get("enabled") or not bridge.get("quantity_profile", {}).get("families"):
            raise ValueError("Manuscript observation profile requires its native source/readout declaration")
    return {"status": "bound", "profile": profile, "posterior_family": family,
            "observation_feature_width": width,
            "authority": "selected_configuration_declared_observation_interface",
            "data_or_outcome_inference": False}


def world_state_posterior_config_from_config(config: dict) -> dict[str, object]:
    posterior_config = (
        config["screening_pipeline"]["screening_benchmark"]
        .get("model_architecture", {})
        .get("world_state_posterior", {})
    )
    required_keys = ["posterior_hidden_dim", "posterior_dropout_rate"]
    missing_keys = [key for key in required_keys if key not in posterior_config]
    if missing_keys:
        raise ValueError(f"Missing world_state_posterior config fields: {missing_keys}")
    family = posterior_config.get("family", "deterministic_compatibility")
    if family not in {"finite_joint_particles", "deterministic_compatibility"}:
        raise ValueError("Unknown world-state posterior family.")
    return {
        "posterior_hidden_dim": int(posterior_config["posterior_hidden_dim"]),
        "posterior_dropout_rate": float(posterior_config["posterior_dropout_rate"]),
        "family": family,
        "observation_feature_width": posterior_config.get("observation_feature_width"),
        "registered_occupancy_assays": dict(posterior_config.get("registered_occupancy_assays", {})),
        "joint_objectives": dict(posterior_config.get("joint_objectives", {})),
        "physical_parameter_schema": parameter_schema_from_config(config),
    }


def multimodal_moe_config_from_config(config: dict) -> dict[str, object] | None:
    moe_config = (
        config["screening_pipeline"]["screening_benchmark"]
        .get("model_architecture", {})
        .get("multimodal_moe", {})
    )
    if not moe_config:
        return None
    required_keys = [
        "enabled",
        "expert_names",
        "top_k",
        "capacity_factor",
        "router_loss_weight",
        "entropy_loss_weight",
        "load_balance_loss_weight",
        "expert_capacity_loss_weight",
        "expert_hidden_multiplier",
    ]
    missing_keys = [key for key in required_keys if key not in moe_config]
    if missing_keys:
        raise ValueError(f"Missing multimodal_moe config fields: {missing_keys}")
    return {
        "enabled": bool(moe_config["enabled"]),
        "expert_names": [str(expert_name) for expert_name in moe_config["expert_names"]],
        "top_k": int(moe_config["top_k"]),
        "capacity_factor": float(moe_config["capacity_factor"]),
        "router_loss_weight": float(moe_config["router_loss_weight"]),
        "entropy_loss_weight": float(moe_config["entropy_loss_weight"]),
        "load_balance_loss_weight": float(moe_config["load_balance_loss_weight"]),
        "expert_capacity_loss_weight": float(moe_config["expert_capacity_loss_weight"]),
        "expert_hidden_multiplier": int(moe_config["expert_hidden_multiplier"]),
    }


def screening_model_kwargs_from_config(config: dict) -> dict[str, object]:
    region_prior = default_region_prior(config)
    return {
        "region_count": region_count_from_config(config),
        "species_count": species_count_from_config(config),
        "rollout_step_count": int(config["dynamics_ready_registry"]["time_grid"]["step_count"]),
        "region_prior_feature_dim": int(region_prior.shape[1]),
        "default_region_prior": region_prior.tolist(),
        "baseline_field_policy": baseline_field_policy_from_config(config),
        "rollout_update_policy": rollout_update_policy_from_config(config),
        "solver_state_layout": solver_state_layout_from_config(config),
        "solver_config": solver_config_from_config(config),
        "world_state_posterior_config": world_state_posterior_config_from_config(config),
        "multimodal_moe_config": multimodal_moe_config_from_config(config),
    }
