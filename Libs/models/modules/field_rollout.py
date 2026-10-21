from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from Libs.models.layers.feed_forward import FeedForwardBlock
from Libs.models.modules.neuronsifter_world_action import NeuronSifterWorldActionOperator
from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.models.modules.transolver_operator import TransolverMicroenvironmentOperator


class ActionConditionedOperatorBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.state_mixer = nn.Linear(hidden_dim, hidden_dim)
        self.neighborhood_mixer = nn.Linear(hidden_dim, hidden_dim)
        self.conditioner = nn.Linear(hidden_dim, hidden_dim * 3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_state: torch.Tensor,
        neighborhood_state: torch.Tensor,
        conditioning_state: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift, gate = self.conditioner(conditioning_state).chunk(3, dim=-1)
        mixed_state = self.state_mixer(node_state) + self.neighborhood_mixer(neighborhood_state)
        conditioned_state = self.norm(mixed_state * (1.0 + torch.tanh(scale)) + shift)
        return node_state + torch.sigmoid(gate) * self.feed_forward(conditioned_state)


class FieldRolloutOperator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        regimen_feature_dim: int,
        hidden_dim: int,
        region_count: int,
        compartment_count: int,
        species_count: int,
        rollout_step_count: int,
        adjacency_matrix: list[list[float]],
        fast_steps_per_slow_step: int = 1,
        fast_state_indices: list[int] | None = None,
        slow_state_indices: list[int] | None = None,
        slow_update_scale: float = 0.25,
        rollout_update_policy: dict[str, object] | None = None,
        solver_state_layout: dict[str, int] | None = None,
        solver_config: dict[str, object] | None = None,
        field_normalization_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.region_count = region_count
        self.compartment_count = compartment_count
        self.species_count = species_count
        self.field_normalization = NativeFieldNormalization(field_normalization_config, species_count)
        self.node_count = region_count * compartment_count
        self.rollout_step_count = rollout_step_count
        self.fast_steps_per_slow_step = max(int(fast_steps_per_slow_step), 1)
        self.slow_update_scale = float(slow_update_scale)
        if rollout_update_policy is None:
            raise ValueError("rollout_update_policy is required for FieldRolloutOperator.")
        if solver_state_layout is None:
            raise ValueError("solver_state_layout is required for FieldRolloutOperator.")
        if solver_config is None:
            raise ValueError("solver_config is required for FieldRolloutOperator.")
        required_update_keys = [
            "state_delta_scale",
            "control_source_scale",
            "region_prior_mix",
            "activation_checkpointing",
            "action_representation",
            "transition_backbone",
        ]
        missing_update_keys = [key for key in required_update_keys if key not in rollout_update_policy]
        if missing_update_keys:
            raise ValueError(f"Missing rollout_update_policy fields: {missing_update_keys}")
        self.state_delta_scale = float(rollout_update_policy["state_delta_scale"])
        self.control_source_scale = float(rollout_update_policy["control_source_scale"])
        self.region_prior_mix = float(rollout_update_policy["region_prior_mix"])
        self.activation_checkpointing = bool(rollout_update_policy["activation_checkpointing"])
        action_representation = dict(rollout_update_policy["action_representation"])
        self.action_policy_name = str(action_representation["active_policy"])
        feature_groups = {
            str(group_name): [int(index) for index in group_indices]
            for group_name, group_indices in dict(action_representation["feature_groups"]).items()
        }
        policy_payload = dict(action_representation["policies"])[self.action_policy_name]
        retained_group_names = [str(group_name) for group_name in dict(policy_payload)["retained_feature_groups"]]
        retained_indices = sorted(
            {
                feature_index
                for group_name in retained_group_names
                for feature_index in feature_groups[group_name]
            }
        )
        invalid_feature_indices = [
            feature_index
            for feature_index in retained_indices
            if feature_index < 0 or feature_index >= regimen_feature_dim
        ]
        if invalid_feature_indices:
            raise ValueError(f"Action representation policy {self.action_policy_name} references invalid feature indices: {invalid_feature_indices}")
        action_feature_mask = torch.zeros(regimen_feature_dim, dtype=torch.float32)
        if retained_indices:
            action_feature_mask[torch.tensor(retained_indices, dtype=torch.long)] = 1.0
        self.register_buffer("action_feature_mask", action_feature_mask)
        transition_backbone = dict(rollout_update_policy["transition_backbone"])
        self.transition_backbone_kind = str(transition_backbone["kind"])
        supported_backbones = {"factor_coupled_neural_operator", "neuronsifter_world_action", "transolver_microenvironment_operator"}
        if self.transition_backbone_kind not in supported_backbones:
            raise ValueError(f"Unsupported transition_backbone kind: {transition_backbone['kind']}")
        self.operator_hidden_dim = int(transition_backbone["operator_hidden_dim"])
        self.operator_depth = int(transition_backbone["operator_depth"])
        self.operator_dropout_rate = float(transition_backbone["operator_dropout_rate"])
        self.operator_residual_scale = float(transition_backbone["operator_residual_scale"])
        self.regimen_encoder = FeedForwardBlock(regimen_feature_dim, hidden_dim, latent_dim, 0.0)
        self.source_modulation_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, compartment_count),
        )
        self.binding_gain_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pathology_relief_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.clearance_support_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, compartment_count),
        )
        self.patient_rate_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.operator_state_proj = nn.Linear(species_count, self.operator_hidden_dim)
        self.operator_parameter_proj = nn.Linear(species_count, self.operator_hidden_dim)
        self.operator_prior_proj = nn.Linear(species_count, self.operator_hidden_dim)
        self.operator_control_proj = nn.Linear(9, self.operator_hidden_dim)
        self.operator_context_proj = nn.Linear(latent_dim * 3, self.operator_hidden_dim)
        self.operator_blocks = nn.ModuleList()
        self.transolver_operator: TransolverMicroenvironmentOperator | None = None
        self.neuronsifter_world_action_operator: NeuronSifterWorldActionOperator | None = None
        self.quantized_mddm_rollout_operator: QuantizedMddmRolloutOperator | None = None
        if self.transition_backbone_kind == "factor_coupled_neural_operator":
            self.operator_blocks = nn.ModuleList(
                [
                    ActionConditionedOperatorBlock(
                        self.operator_hidden_dim,
                        self.operator_dropout_rate,
                    )
                    for _ in range(self.operator_depth)
                ]
            )
        elif self.transition_backbone_kind == "transolver_microenvironment_operator":
            transolver_keys = [
                "attention_head_count",
                "physical_state_slice_count",
                "node_coordinate_dim",
                "encoder_decoder_pretraining",
                "physics_mlp_multiplier",
                "mechanism_moe",
            ]
            missing_transolver_keys = [key for key in transolver_keys if key not in transition_backbone]
            if missing_transolver_keys:
                raise ValueError(f"Missing transolver_microenvironment_operator fields: {missing_transolver_keys}")
            mechanism_moe = dict(transition_backbone["mechanism_moe"])
            missing_moe_keys = [
                key
                for key in [
                    "expert_names",
                    "top_k",
                    "capacity_factor",
                    "entropy_loss_weight",
                    "load_balance_loss_weight",
                    "expert_capacity_loss_weight",
                    "expert_hidden_multiplier",
                ]
                if key not in mechanism_moe
            ]
            if missing_moe_keys:
                raise ValueError(f"Missing mechanism_moe fields: {missing_moe_keys}")
            encoder_decoder_pretraining = dict(transition_backbone["encoder_decoder_pretraining"])
            missing_pretraining_keys = [
                key
                for key in [
                    "encoder_depth",
                    "decoder_depth",
                    "checkpoint_attention_blocks",
                ]
                if key not in encoder_decoder_pretraining
            ]
            if missing_pretraining_keys:
                raise ValueError(f"Missing encoder_decoder_pretraining fields: {missing_pretraining_keys}")
            self.transolver_operator = TransolverMicroenvironmentOperator(
                self.operator_hidden_dim,
                int(encoder_decoder_pretraining["encoder_depth"]),
                int(encoder_decoder_pretraining["decoder_depth"]),
                int(transition_backbone["attention_head_count"]),
                int(transition_backbone["physical_state_slice_count"]),
                int(transition_backbone["node_coordinate_dim"]),
                self.operator_dropout_rate,
                [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
                int(mechanism_moe["top_k"]),
                float(mechanism_moe["capacity_factor"]),
                float(mechanism_moe["entropy_loss_weight"]),
                float(mechanism_moe["load_balance_loss_weight"]),
                float(mechanism_moe["expert_capacity_loss_weight"]),
                int(mechanism_moe["expert_hidden_multiplier"]),
                int(transition_backbone["physics_mlp_multiplier"]),
                bool(encoder_decoder_pretraining["checkpoint_attention_blocks"]),
            )
        else:
            neuronsifter_keys = [
                "attention_head_count",
                "mechanism_state_token_count",
                "node_coordinate_dim",
                "dit_mlp_multiplier",
                "mechanism_moe",
                "graph_spectral_tokenizer",
                "temporal_wavelet_tokenizer",
                "diffusion_sampler",
                "operator_splitting",
            ]
            missing_neuronsifter_keys = [key for key in neuronsifter_keys if key not in transition_backbone]
            if missing_neuronsifter_keys:
                raise ValueError(f"Missing neuronsifter_world_action fields: {missing_neuronsifter_keys}")
            mechanism_moe = dict(transition_backbone["mechanism_moe"])
            missing_moe_keys = [
                key
                for key in [
                    "expert_names",
                    "top_k",
                    "capacity_factor",
                    "entropy_loss_weight",
                    "load_balance_loss_weight",
                    "expert_capacity_loss_weight",
                    "expert_hidden_multiplier",
                ]
                if key not in mechanism_moe
            ]
            if missing_moe_keys:
                raise ValueError(f"Missing mechanism_moe fields: {missing_moe_keys}")
            graph_config = dict(transition_backbone["graph_spectral_tokenizer"])
            wavelet_config = dict(transition_backbone["temporal_wavelet_tokenizer"])
            sampler_config = dict(transition_backbone["diffusion_sampler"])
            operator_splitting = dict(transition_backbone["operator_splitting"])
            missing_operator_splitting_keys = [
                key
                for key in [
                    "enabled",
                    "bottleneck_divisor",
                    "expert_residual_scale",
                    "shared_coupling_residual_scale",
                    "composition_order",
                    "coupling_masks",
                ]
                if key not in operator_splitting
            ]
            if missing_operator_splitting_keys:
                raise ValueError(f"Missing operator_splitting fields: {missing_operator_splitting_keys}")
            quantile_levels = [float(value) for value in sampler_config["quantile_levels"]]
            self.neuronsifter_world_action_operator = NeuronSifterWorldActionOperator(
                self.operator_hidden_dim,
                self.operator_depth,
                int(transition_backbone["attention_head_count"]),
                int(transition_backbone["mechanism_state_token_count"]),
                int(transition_backbone["node_coordinate_dim"]),
                self.operator_dropout_rate,
                [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
                int(mechanism_moe["top_k"]),
                float(mechanism_moe["capacity_factor"]),
                float(mechanism_moe["entropy_loss_weight"]),
                float(mechanism_moe["load_balance_loss_weight"]),
                float(mechanism_moe["expert_capacity_loss_weight"]),
                int(mechanism_moe["expert_hidden_multiplier"]),
                int(transition_backbone["dit_mlp_multiplier"]),
                int(graph_config["graph_mode_count"]),
                int(wavelet_config["wavelet_level_count"]),
                float(graph_config["eigenvalue_temperature"]),
                bool(graph_config["normalize_laplacian"]),
                float(graph_config["spectral_residual_scale"]),
                float(transition_backbone["mechanism_slice_temperature"]),
                int(sampler_config["sample_count"]),
                int(sampler_config["denoising_step_count"]),
                int(sampler_config["sample_chunk_size"]),
                int(sampler_config["training_sample_count"]),
                int(sampler_config["training_denoising_step_count"]),
                float(sampler_config["diffusion_noise_scale"]),
                float(sampler_config["score_step_size"]),
                float(sampler_config["terminal_noise_scale"]),
                float(sampler_config["ranking_stability_scale"]),
                quantile_levels,
                int(sampler_config["sampler_seed"]),
                bool(sampler_config["return_sample_trajectories"]),
                operator_splitting,
            )
            quantized_mddm_config = transition_backbone.get("quantized_mddm_rollout")
            if quantized_mddm_config is not None and bool(dict(quantized_mddm_config)["enabled"]):
                self.quantized_mddm_rollout_operator = QuantizedMddmRolloutOperator(
                    dict(quantized_mddm_config), self.operator_hidden_dim, self.species_count,
                    action_policy=self.action_policy_name,
                    quantity_profile=self.field_normalization.quantity_profile,
                    scalar_exposure_definition=action_representation["policies"].get("scalar_exposure", {}).get("native_definition"))
        self.operator_delta_head = nn.Sequential(
            nn.LayerNorm(self.operator_hidden_dim),
            nn.Linear(self.operator_hidden_dim, self.operator_hidden_dim),
            nn.GELU(),
            nn.Linear(self.operator_hidden_dim, species_count),
        )
        self.register_buffer(
            "node_adjacency",
            torch.tensor(adjacency_matrix, dtype=torch.float32),
        )
        node_coordinates = []
        region_denominator = max(int(region_count) - 1, 1)
        compartment_denominator = max(int(compartment_count) - 1, 1)
        for region_index in range(int(region_count)):
            for compartment_index in range(int(compartment_count)):
                node_coordinates.append(
                    [
                        float(region_index) / float(region_denominator),
                        float(compartment_index) / float(compartment_denominator),
                    ]
                )
        self.register_buffer(
            "node_coordinates",
            torch.tensor(node_coordinates, dtype=torch.float32).view(1, 1, self.node_count, 2),
        )
        self._latest_router_diagnostics: dict[str, torch.Tensor] = {}
        if fast_state_indices is None or slow_state_indices is None:
            raise ValueError("fast_state_indices and slow_state_indices are required for FieldRolloutOperator.")
        self.register_buffer("fast_state_indices", torch.tensor([int(index) for index in fast_state_indices], dtype=torch.long))
        self.register_buffer("slow_state_indices", torch.tensor([int(index) for index in slow_state_indices], dtype=torch.long))
        required_state_names = [
            "free_drug",
            "target_bound_drug",
            "misfolded_protein_load",
            "aggregate_burden",
            "neuroinflammation",
            "clearance_flux",
            "circuit_instability",
        ]
        missing_state_names = [state_name for state_name in required_state_names if state_name not in solver_state_layout]
        if missing_state_names:
            raise ValueError(f"Missing solver_state_layout entries: {missing_state_names}")
        self.state_layout = {name: int(index) for name, index in solver_state_layout.items()}
        required_solver_keys = [
            "diffusion_scale_by_compartment",
            "clearance_scale_by_compartment",
            "route_source_by_compartment",
            "schedule_frequency_multipliers",
            "target_binding_gain",
            "amyloid_reduction_gain",
            "plaque_clearance_gain",
            "electrophysiology_relief_gain",
            "electrophysiology_amyloid_gain",
            "electrophysiology_plaque_gain",
            "minimum_bound_fraction",
            "maximum_bound_fraction",
            "bound_decay_rate",
            "parameter_context_scale_by_species",
            "prior_context_scale_by_drive",
            "aggregate_clearance_scale",
            "inflammation_coupling",
            "inflammation_decay",
            "clearance_flux_scale",
            "circuit_instability_scale",
            "state_upper_bound_by_species",
            "condition_rate_bounds",
            "regimen_feature_layout",
        ]
        missing_solver_keys = [key for key in required_solver_keys if key not in solver_config]
        if missing_solver_keys:
            raise ValueError(f"Missing solver_config fields: {missing_solver_keys}")
        self.register_buffer(
            "diffusion_scale_by_compartment",
            torch.tensor([float(value) for value in solver_config["diffusion_scale_by_compartment"]], dtype=torch.float32),
        )
        self.register_buffer(
            "clearance_scale_by_compartment",
            torch.tensor([float(value) for value in solver_config["clearance_scale_by_compartment"]], dtype=torch.float32),
        )
        route_source_payload = solver_config["route_source_by_compartment"]
        self.register_buffer(
            "route_source_oral",
            torch.tensor([float(value) for value in route_source_payload["oral"]], dtype=torch.float32),
        )
        self.register_buffer(
            "route_source_intravenous",
            torch.tensor([float(value) for value in route_source_payload["intravenous"]], dtype=torch.float32),
        )
        self.register_buffer(
            "route_source_intrathecal",
            torch.tensor([float(value) for value in route_source_payload["intrathecal"]], dtype=torch.float32),
        )
        schedule_payload = solver_config["schedule_frequency_multipliers"]
        self.register_buffer(
            "schedule_frequency_multipliers",
            torch.tensor(
                [
                    float(schedule_payload["daily"]),
                    float(schedule_payload["twice_daily"]),
                    float(schedule_payload["weekly"]),
                ],
                dtype=torch.float32,
            ),
        )
        condition_rate_bounds = solver_config["condition_rate_bounds"]
        self.register_buffer(
            "misfolded_production_bounds",
            torch.tensor(
                [
                    float(condition_rate_bounds["control"]["misfolded_protein_production_rate"]),
                    float(condition_rate_bounds["ad_positive"]["misfolded_protein_production_rate"]),
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "aggregate_accumulation_bounds",
            torch.tensor(
                [
                    float(condition_rate_bounds["control"]["aggregate_accumulation_rate"]),
                    float(condition_rate_bounds["ad_positive"]["aggregate_accumulation_rate"]),
                ],
                dtype=torch.float32,
            ),
        )
        self.target_binding_gain = float(solver_config["target_binding_gain"])
        self.amyloid_reduction_gain = float(solver_config["amyloid_reduction_gain"])
        self.plaque_clearance_gain = float(solver_config["plaque_clearance_gain"])
        self.electrophysiology_relief_gain = float(solver_config["electrophysiology_relief_gain"])
        self.electrophysiology_amyloid_gain = float(solver_config["electrophysiology_amyloid_gain"])
        self.electrophysiology_plaque_gain = float(solver_config["electrophysiology_plaque_gain"])
        self.minimum_bound_fraction = float(solver_config["minimum_bound_fraction"])
        self.maximum_bound_fraction = float(solver_config["maximum_bound_fraction"])
        self.bound_decay_rate = float(solver_config["bound_decay_rate"])
        self.parameter_context_scale_by_species = {
            species_id: float(value)
            for species_id, value in dict(solver_config["parameter_context_scale_by_species"]).items()
        }
        self.prior_context_scale_by_drive = {
            drive_name: float(value)
            for drive_name, value in dict(solver_config["prior_context_scale_by_drive"]).items()
        }
        self.aggregate_clearance_scale = float(solver_config["aggregate_clearance_scale"])
        self.inflammation_coupling = {
            species_id: float(value)
            for species_id, value in dict(solver_config["inflammation_coupling"]).items()
        }
        self.inflammation_decay = float(solver_config["inflammation_decay"])
        self.clearance_flux_scale = {
            scale_name: float(value)
            for scale_name, value in dict(solver_config["clearance_flux_scale"]).items()
        }
        self.circuit_instability_scale = {
            scale_name: float(value)
            for scale_name, value in dict(solver_config["circuit_instability_scale"]).items()
        }
        self.state_upper_bound_by_species = {
            species_id: float(value)
            for species_id, value in dict(solver_config["state_upper_bound_by_species"]).items()
        }
        full_microenvironment_species = [
            "amyloid_beta_soluble",
            "amyloid_plaque",
            "phosphorylated_tau",
            "inflammatory_mediator",
            "calcium_concentration",
            "membrane_voltage",
            "spike_rate",
            "vascular_perivascular_transport",
        ]
        full_microenvironment_coupling_payload = dict(solver_config.get("full_microenvironment_coupling", {}))
        missing_full_coupling = [
            species_id
            for species_id in full_microenvironment_species
            if species_id in self.state_layout and species_id not in full_microenvironment_coupling_payload
        ]
        if missing_full_coupling:
            raise ValueError(f"Missing full_microenvironment_coupling entries: {missing_full_coupling}")
        self.full_microenvironment_coupling = {
            species_id: {
                parameter_name: float(parameter_value) if isinstance(parameter_value, (float, int)) else str(parameter_value)
                for parameter_name, parameter_value in dict(species_config).items()
            }
            for species_id, species_config in full_microenvironment_coupling_payload.items()
        }
        feature_layout = solver_config["regimen_feature_layout"]
        required_feature_indices = [
            "dose_multiplier_index",
            "intrathecal_indicator_index",
            "weekly_indicator_index",
            "oral_route_indicator_index",
            "intravenous_route_indicator_index",
            "daily_schedule_indicator_index",
            "twice_daily_schedule_indicator_index",
        ]
        missing_feature_indices = [key for key in required_feature_indices if key not in feature_layout]
        if missing_feature_indices:
            raise ValueError(f"Missing regimen_feature_layout entries: {missing_feature_indices}")
        self.regimen_feature_layout = {key: int(value) for key, value in feature_layout.items()}
        self.register_buffer(
            "pkpd_indices",
            torch.tensor(
                [self.state_layout["free_drug"], self.state_layout["target_bound_drug"]],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "transport_indices",
            torch.tensor(
                [
                    self.state_layout["misfolded_protein_load"],
                    self.state_layout["aggregate_burden"],
                    self.state_layout["neuroinflammation"],
                    self.state_layout["clearance_flux"],
                    *[
                        self.state_layout[species_id]
                        for species_id in [
                            "amyloid_beta_soluble",
                            "amyloid_plaque",
                            "phosphorylated_tau",
                            "inflammatory_mediator",
                            "vascular_perivascular_transport",
                        ]
                        if species_id in self.state_layout
                    ],
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "electrophysiology_indices",
            torch.tensor(
                [
                    self.state_layout[species_id]
                    for species_id in ["circuit_instability", "calcium_concentration", "membrane_voltage", "spike_rate"]
                    if species_id in self.state_layout
                ],
                dtype=torch.long,
            ),
        )

    def action_conditioned_features(self, regimen_features: torch.Tensor) -> torch.Tensor:
        return regimen_features * self.action_feature_mask.to(device=regimen_features.device, dtype=regimen_features.dtype).unsqueeze(0)

    def regimen_source_prior(self, regimen_features: torch.Tensor) -> torch.Tensor:
        dose = regimen_features[:, self.regimen_feature_layout["dose_multiplier_index"]].clamp_min(0.0)
        oral_flag = regimen_features[:, self.regimen_feature_layout["oral_route_indicator_index"]]
        intravenous_flag = regimen_features[:, self.regimen_feature_layout["intravenous_route_indicator_index"]]
        intrathecal_flag = regimen_features[:, self.regimen_feature_layout["intrathecal_indicator_index"]]
        daily_flag = regimen_features[:, self.regimen_feature_layout["daily_schedule_indicator_index"]]
        twice_daily_flag = regimen_features[:, self.regimen_feature_layout["twice_daily_schedule_indicator_index"]]
        weekly_flag = regimen_features[:, self.regimen_feature_layout["weekly_indicator_index"]]
        route_source = (
            oral_flag.unsqueeze(1) * self.route_source_oral.unsqueeze(0)
            + intravenous_flag.unsqueeze(1) * self.route_source_intravenous.unsqueeze(0)
            + intrathecal_flag.unsqueeze(1) * self.route_source_intrathecal.unsqueeze(0)
        )
        schedule_scale = (
            daily_flag * self.schedule_frequency_multipliers[0]
            + twice_daily_flag * self.schedule_frequency_multipliers[1]
            + weekly_flag * self.schedule_frequency_multipliers[2]
        )
        return dose.unsqueeze(1) * schedule_scale.unsqueeze(1) * route_source

    def species_state(self, state: torch.Tensor, species_id: str) -> torch.Tensor:
        return state[..., self.state_layout[species_id]]

    def species_operator_residual(self, operator_residual: torch.Tensor, species_id: str) -> torch.Tensor:
        return operator_residual[..., self.state_layout[species_id]]

    def species_upper_bound(self, species_id: str, reference: torch.Tensor) -> torch.Tensor:
        return torch.full_like(reference, self.state_upper_bound_by_species[species_id])

    def microenvironment_config(self, species_id: str) -> dict[str, float | str]:
        return self.full_microenvironment_coupling[species_id]

    def update_full_microenvironment_species(
        self,
        next_state: torch.Tensor,
        state: torch.Tensor,
        operator_residual: torch.Tensor,
        free_update: torch.Tensor,
        bound_update: torch.Tensor,
        misfolded_update: torch.Tensor,
        aggregate_update: torch.Tensor,
        inflammation_update: torch.Tensor,
        clearance_flux: torch.Tensor,
        circuit_instability: torch.Tensor,
    ) -> torch.Tensor:
        if "amyloid_beta_soluble" in self.state_layout:
            coupling = self.microenvironment_config("amyloid_beta_soluble")
            soluble_state = self.species_state(state, "amyloid_beta_soluble")
            soluble_update = torch.clamp(
                soluble_state
                + self.state_delta_scale
                * (
                    float(coupling["production_gain"]) * misfolded_update
                    - float(coupling["clearance_gain"]) * clearance_flux * soluble_state
                    - float(coupling["target_bound_relief_gain"]) * bound_update
                    - float(coupling["plaque_conversion_gain"]) * soluble_state
                )
                + self.species_operator_residual(operator_residual, "amyloid_beta_soluble"),
                min=0.0,
                max=self.state_upper_bound_by_species["amyloid_beta_soluble"],
            )
            next_state[..., self.state_layout["amyloid_beta_soluble"]] = soluble_update
        else:
            soluble_update = misfolded_update

        if "amyloid_plaque" in self.state_layout:
            coupling = self.microenvironment_config("amyloid_plaque")
            plaque_state = self.species_state(state, "amyloid_plaque")
            plaque_update = torch.clamp(
                plaque_state
                + self.state_delta_scale
                * (
                    float(coupling["soluble_conversion_gain"]) * soluble_update
                    + float(coupling["source_species_gain"]) * aggregate_update
                    - float(coupling["clearance_gain"]) * clearance_flux * plaque_state
                    - float(coupling["target_bound_relief_gain"]) * bound_update
                )
                + self.species_operator_residual(operator_residual, "amyloid_plaque"),
                min=0.0,
                max=self.state_upper_bound_by_species["amyloid_plaque"],
            )
            next_state[..., self.state_layout["amyloid_plaque"]] = plaque_update
        else:
            plaque_update = aggregate_update

        if "phosphorylated_tau" in self.state_layout:
            coupling = self.microenvironment_config("phosphorylated_tau")
            tau_state = self.species_state(state, "phosphorylated_tau")
            tau_update = torch.clamp(
                tau_state
                + self.state_delta_scale
                * (
                    float(coupling["amyloid_gain"]) * soluble_update
                    + float(coupling["plaque_gain"]) * plaque_update
                    + float(coupling["inflammation_gain"]) * inflammation_update
                    - float(coupling["clearance_gain"]) * clearance_flux * tau_state
                    - float(coupling["target_bound_relief_gain"]) * bound_update
                )
                + self.species_operator_residual(operator_residual, "phosphorylated_tau"),
                min=0.0,
                max=self.state_upper_bound_by_species["phosphorylated_tau"],
            )
            next_state[..., self.state_layout["phosphorylated_tau"]] = tau_update
        else:
            tau_update = aggregate_update

        if "inflammatory_mediator" in self.state_layout:
            coupling = self.microenvironment_config("inflammatory_mediator")
            mediator_state = self.species_state(state, "inflammatory_mediator")
            mediator_update = torch.clamp(
                mediator_state
                + self.state_delta_scale
                * (
                    float(coupling["amyloid_gain"]) * soluble_update
                    + float(coupling["plaque_gain"]) * plaque_update
                    + float(coupling["tau_gain"]) * tau_update
                    - float(coupling["decay_gain"]) * mediator_state
                    - float(coupling["target_bound_relief_gain"]) * bound_update
                )
                + self.species_operator_residual(operator_residual, "inflammatory_mediator"),
                min=0.0,
                max=self.state_upper_bound_by_species["inflammatory_mediator"],
            )
            next_state[..., self.state_layout["inflammatory_mediator"]] = mediator_update
        else:
            mediator_update = inflammation_update

        if "calcium_concentration" in self.state_layout:
            coupling = self.microenvironment_config("calcium_concentration")
            calcium_state = self.species_state(state, "calcium_concentration")
            calcium_update = torch.clamp(
                calcium_state
                + self.state_delta_scale
                * (
                    float(coupling["circuit_gain"]) * circuit_instability
                    + float(coupling["inflammation_gain"]) * mediator_update
                    - float(coupling["decay_gain"]) * calcium_state
                )
                + self.species_operator_residual(operator_residual, "calcium_concentration"),
                min=0.0,
                max=self.state_upper_bound_by_species["calcium_concentration"],
            )
            next_state[..., self.state_layout["calcium_concentration"]] = calcium_update
        else:
            calcium_update = circuit_instability

        if "membrane_voltage" in self.state_layout:
            coupling = self.microenvironment_config("membrane_voltage")
            voltage_state = self.species_state(state, "membrane_voltage")
            voltage_update = torch.clamp(
                voltage_state
                + self.state_delta_scale
                * (
                    float(coupling["circuit_gain"]) * circuit_instability
                    + float(coupling["calcium_gain"]) * calcium_update
                    - float(coupling["decay_gain"]) * voltage_state
                )
                + self.species_operator_residual(operator_residual, "membrane_voltage"),
                min=0.0,
                max=self.state_upper_bound_by_species["membrane_voltage"],
            )
            next_state[..., self.state_layout["membrane_voltage"]] = voltage_update
        else:
            voltage_update = circuit_instability

        if "spike_rate" in self.state_layout:
            coupling = self.microenvironment_config("spike_rate")
            spike_state = self.species_state(state, "spike_rate")
            spike_update = torch.clamp(
                spike_state
                + self.state_delta_scale
                * (
                    float(coupling["voltage_gain"]) * voltage_update
                    + float(coupling["calcium_gain"]) * calcium_update
                    - float(coupling["decay_gain"]) * spike_state
                )
                + self.species_operator_residual(operator_residual, "spike_rate"),
                min=0.0,
                max=self.state_upper_bound_by_species["spike_rate"],
            )
            next_state[..., self.state_layout["spike_rate"]] = spike_update

        if "vascular_perivascular_transport" in self.state_layout:
            coupling = self.microenvironment_config("vascular_perivascular_transport")
            vascular_state = self.species_state(state, "vascular_perivascular_transport")
            vascular_update = torch.clamp(
                vascular_state
                + self.state_delta_scale
                * (
                    float(coupling["clearance_flux_gain"]) * clearance_flux
                    + float(coupling["free_drug_gain"]) * free_update
                    - float(coupling["inflammation_penalty_gain"]) * mediator_update
                    - float(coupling["decay_gain"]) * vascular_state
                )
                + self.species_operator_residual(operator_residual, "vascular_perivascular_transport"),
                min=0.0,
                max=self.state_upper_bound_by_species["vascular_perivascular_transport"],
            )
            next_state[..., self.state_layout["vascular_perivascular_transport"]] = vascular_update
        return next_state

    def operator_residual(
        self,
        state: torch.Tensor,
        patient_context: torch.Tensor,
        regimen_context_expanded: torch.Tensor,
        parameter_context: torch.Tensor,
        region_prior_node: torch.Tensor,
        region_prior_context: torch.Tensor,
        compartment_source: torch.Tensor,
        binding_gain: torch.Tensor,
        pathology_relief: torch.Tensor,
        clearance_support: torch.Tensor,
        patient_rate_bounds: torch.Tensor,
    ) -> torch.Tensor:
        control_features = torch.cat(
            [
                compartment_source.unsqueeze(-1),
                binding_gain.unsqueeze(-1),
                clearance_support.unsqueeze(-1),
                pathology_relief,
                patient_rate_bounds,
            ],
            dim=-1,
        )
        conditioning_state = self.operator_control_proj(control_features) + self.operator_context_proj(
            torch.cat([patient_context, regimen_context_expanded, region_prior_context], dim=-1)
        )
        node_state = (
            self.operator_state_proj(state)
            + self.operator_parameter_proj(parameter_context)
            + self.operator_prior_proj(region_prior_node)
            + conditioning_state
        )
        if self.transolver_operator is not None:
            node_coordinates = self.node_coordinates.expand(
                node_state.shape[0],
                node_state.shape[1],
                -1,
                -1,
            )
            node_state, self._latest_router_diagnostics = self.transolver_operator(
                node_state,
                conditioning_state,
                node_coordinates,
            )
        elif self.neuronsifter_world_action_operator is not None:
            node_coordinates = self.node_coordinates.expand(
                node_state.shape[0],
                node_state.shape[1],
                -1,
                -1,
            )
            node_state, self._latest_router_diagnostics = self.neuronsifter_world_action_operator(
                node_state.reshape(-1, 1, self.node_count, self.operator_hidden_dim),
                conditioning_state.reshape(-1, 1, self.node_count, self.operator_hidden_dim),
                node_coordinates.reshape(-1, 1, self.node_count, node_coordinates.shape[-1]),
                self.node_adjacency,
            )
            node_state = node_state.reshape_as(conditioning_state)
            activation = self._latest_router_diagnostics["operator_splitting_case_activation"]
            self._latest_router_diagnostics["operator_splitting_case_activation"] = activation.reshape(
                conditioning_state.shape[0], conditioning_state.shape[1], -1
            ).mean(dim=1)
        else:
            self._latest_router_diagnostics = {}
            for block in self.operator_blocks:
                neighborhood_state = torch.einsum("nm,brmh->brnh", self.node_adjacency, node_state)
                node_state = block(node_state, neighborhood_state, conditioning_state)
        return self.operator_residual_scale * torch.tanh(self.operator_delta_head(node_state))

    def step_update(
        self,
        state: torch.Tensor,
        patient_context: torch.Tensor,
        regimen_context_expanded: torch.Tensor,
        parameter_context: torch.Tensor,
        region_prior_node: torch.Tensor,
        region_prior_context: torch.Tensor,
        compartment_source: torch.Tensor,
        binding_gain: torch.Tensor,
        pathology_relief: torch.Tensor,
        clearance_support: torch.Tensor,
        patient_rate_bounds: torch.Tensor,
    ) -> torch.Tensor:
        neighborhood_delta = torch.einsum("nm,brms->brns", self.node_adjacency, state) - state
        compartment_diffusion = self.diffusion_scale_by_compartment.repeat(self.region_count).view(1, 1, self.node_count, 1)
        compartment_clearance = self.clearance_scale_by_compartment.repeat(self.region_count).view(1, 1, self.node_count, 1)
        free_index = self.state_layout["free_drug"]
        bound_index = self.state_layout["target_bound_drug"]
        misfolded_index = self.state_layout["misfolded_protein_load"]
        aggregate_index = self.state_layout["aggregate_burden"]
        inflammation_index = self.state_layout["neuroinflammation"]
        clearance_index = self.state_layout["clearance_flux"]
        circuit_index = self.state_layout["circuit_instability"]

        next_state = state.clone()
        free_state = state[..., free_index]
        bound_state = state[..., bound_index]
        misfolded_state = state[..., misfolded_index]
        aggregate_state = state[..., aggregate_index]
        inflammation_state = state[..., inflammation_index]
        free_delta = neighborhood_delta[..., free_index]
        misfolded_delta = neighborhood_delta[..., misfolded_index]
        aggregate_delta = neighborhood_delta[..., aggregate_index]
        operator_residual = self.operator_residual(
            state,
            patient_context,
            regimen_context_expanded,
            parameter_context,
            region_prior_node,
            region_prior_context,
            compartment_source,
            binding_gain,
            pathology_relief,
            clearance_support,
            patient_rate_bounds,
        )

        binding_term = binding_gain * free_state
        free_update = torch.relu(
            free_state
            + self.state_delta_scale
            * (
                compartment_diffusion.squeeze(-1) * free_delta
                + self.control_source_scale * compartment_source
                - (compartment_clearance.squeeze(-1) + clearance_support) * free_state
                - binding_term
            )
        )
        free_update = torch.relu(free_update + operator_residual[..., free_index])
        bound_candidate = bound_state + self.state_delta_scale * (binding_term - self.bound_decay_rate * bound_state)
        minimum_bound = self.minimum_bound_fraction * free_update
        bound_update = torch.minimum(
            torch.maximum(bound_candidate + operator_residual[..., bound_index], minimum_bound),
            torch.full_like(bound_candidate, self.maximum_bound_fraction),
        )

        pathology_drive = (
            patient_rate_bounds[..., 0]
            + self.parameter_context_scale_by_species["misfolded_protein_load"] * parameter_context[..., misfolded_index]
            + self.prior_context_scale_by_drive["pathology"] * region_prior_context[..., 0]
        )
        aggregate_drive = (
            patient_rate_bounds[..., 1]
            + self.parameter_context_scale_by_species["aggregate_burden"] * parameter_context[..., aggregate_index]
        )
        inflammation_drive = (
            patient_rate_bounds[..., 2]
            + self.prior_context_scale_by_drive["inflammation"] * region_prior_context[..., 0]
        )

        misfolded_update = torch.relu(
            misfolded_state
            + self.state_delta_scale
            * (
                compartment_diffusion.squeeze(-1) * misfolded_delta
                + pathology_drive
                - pathology_relief[..., 0] * bound_update
                - compartment_clearance.squeeze(-1) * misfolded_state
            )
        )
        misfolded_update = torch.relu(misfolded_update + operator_residual[..., misfolded_index])
        aggregate_update = torch.relu(
            aggregate_state
            + self.state_delta_scale
            * (
                compartment_diffusion.squeeze(-1) * aggregate_delta
                + aggregate_drive * misfolded_state
                - pathology_relief[..., 1] * bound_update
                - self.aggregate_clearance_scale * compartment_clearance.squeeze(-1) * aggregate_state
            )
        )
        aggregate_update = torch.relu(aggregate_update + operator_residual[..., aggregate_index])
        inflammation_update = torch.clamp(
            inflammation_state
            + self.state_delta_scale
            * (
                inflammation_drive
                * (
                    self.inflammation_coupling["misfolded_protein_load"] * misfolded_update
                    + self.inflammation_coupling["aggregate_burden"] * aggregate_update
                )
                - pathology_relief[..., 2] * bound_update
                - self.inflammation_decay * inflammation_state
            ),
            min=0.0,
            max=self.state_upper_bound_by_species["neuroinflammation"],
        )
        inflammation_update = torch.clamp(
            inflammation_update + operator_residual[..., inflammation_index],
            min=0.0,
            max=self.state_upper_bound_by_species["neuroinflammation"],
        )
        clearance_flux = torch.relu(
            self.clearance_flux_scale["free_drug"] * compartment_clearance.squeeze(-1) * free_update
            + self.clearance_flux_scale["misfolded_relief"] * torch.relu(misfolded_state - misfolded_update)
        )
        clearance_flux = torch.relu(clearance_flux + operator_residual[..., clearance_index])
        circuit_instability = torch.clamp(
            state[..., circuit_index]
            + self.state_delta_scale
            * (
                self.electrophysiology_amyloid_gain * misfolded_update
                + self.electrophysiology_plaque_gain * aggregate_update
                + self.circuit_instability_scale["neuroinflammation"] * inflammation_update
                - self.electrophysiology_relief_gain * bound_update
                - self.circuit_instability_scale["decay"] * state[..., circuit_index]
            ),
            min=0.0,
            max=self.state_upper_bound_by_species["circuit_instability"],
        )
        circuit_instability = torch.clamp(
            circuit_instability + operator_residual[..., circuit_index],
            min=0.0,
            max=self.state_upper_bound_by_species["circuit_instability"],
        )

        next_state[..., free_index] = free_update
        next_state[..., bound_index] = bound_update
        next_state[..., misfolded_index] = misfolded_update
        next_state[..., aggregate_index] = aggregate_update
        next_state[..., inflammation_index] = inflammation_update
        next_state[..., clearance_index] = clearance_flux
        next_state[..., circuit_index] = circuit_instability
        next_state = self.update_full_microenvironment_species(
            next_state,
            state,
            operator_residual,
            free_update,
            bound_update,
            misfolded_update,
            aggregate_update,
            inflammation_update,
            clearance_flux,
            circuit_instability,
        )
        if self.slow_state_indices.numel() > 0:
            slow_candidate = next_state.index_select(-1, self.slow_state_indices)
            slow_previous = state.index_select(-1, self.slow_state_indices)
            slow_blend = torch.relu(
                slow_previous
                + self.slow_update_scale * (slow_candidate - slow_previous)
            )
            next_state.index_copy_(-1, self.slow_state_indices, slow_blend)
        return next_state

    def forward(
        self,
        encoded_patient: torch.Tensor,
        baseline_field: torch.Tensor,
        parameter_posterior: torch.Tensor,
        region_prior_field: torch.Tensor,
        region_prior_context: torch.Tensor,
        regimen_features: torch.Tensor,
        *, diffusion_context: dict[str, torch.Tensor] | None = None,
        teacher_trajectory: torch.Tensor | None = None, teacher_confidence: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None, regimen_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conditioned_regimen_features = self.action_conditioned_features(regimen_features)
        if self.quantized_mddm_rollout_operator is not None and self.action_policy_name in {"drug_id", "target_id", "scalar_exposure"}:
            # Legacy counts/scaffold/case flags are not canonical drug identity.
            # Keep the observation-derived posterior action independent; the
            # admitted canonical digest reaches the shared denoiser separately.
            conditioned_regimen_features = torch.zeros_like(regimen_features)
        regimen_context = self.regimen_encoder(conditioned_regimen_features)
        batch_size = encoded_patient.shape[0]
        regimen_count = regimen_context.shape[0]
        state = baseline_field.unsqueeze(1).expand(-1, regimen_count, -1, -1, -1).reshape(
            batch_size,
            regimen_count,
            self.node_count,
            self.species_count,
        )
        region_prior_node = region_prior_field.unsqueeze(0).unsqueeze(1).expand(batch_size, regimen_count, -1, -1, -1)
        region_prior_node = region_prior_node.reshape(batch_size, regimen_count, self.node_count, self.species_count)
        state = state + self.region_prior_mix * region_prior_node
        source_prior = self.regimen_source_prior(conditioned_regimen_features)
        source_modulation = torch.sigmoid(self.source_modulation_head(regimen_context))
        compartment_source = source_prior * (0.5 + source_modulation)
        compartment_source = compartment_source.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.region_count, -1)
        compartment_source = compartment_source.reshape(batch_size, regimen_count, self.node_count)
        binding_gain = self.target_binding_gain * (0.5 + torch.sigmoid(self.binding_gain_head(regimen_context)).squeeze(-1))
        binding_gain = binding_gain.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.node_count)
        pathology_relief_base = torch.sigmoid(self.pathology_relief_head(regimen_context))
        pathology_relief_base = torch.stack(
            [
                pathology_relief_base[:, 0] * self.amyloid_reduction_gain,
                pathology_relief_base[:, 1] * self.plaque_clearance_gain,
                pathology_relief_base[:, 2] * self.electrophysiology_relief_gain,
            ],
            dim=-1,
        )
        pathology_relief = pathology_relief_base.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.node_count, -1)
        clearance_support = torch.sigmoid(self.clearance_support_head(regimen_context))
        clearance_support = clearance_support.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.region_count, -1)
        clearance_support = clearance_support.reshape(batch_size, regimen_count, self.node_count)
        patient_context = encoded_patient.unsqueeze(1).unsqueeze(2).expand(-1, regimen_count, self.node_count, -1)
        regimen_context_expanded = regimen_context.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.node_count, -1)
        parameter_context = parameter_posterior.unsqueeze(1).unsqueeze(2).expand(-1, regimen_count, self.node_count, -1)
        region_prior_context = region_prior_context.unsqueeze(0).unsqueeze(1).expand(batch_size, regimen_count, -1, -1)
        region_prior_context = region_prior_context.unsqueeze(3).expand(-1, -1, -1, self.compartment_count, -1)
        region_prior_context = region_prior_context.reshape(batch_size, regimen_count, self.node_count, regimen_context.shape[1])
        patient_rate_head = torch.sigmoid(self.patient_rate_head(encoded_patient))
        patient_rate_bounds = torch.stack(
            [
                self.misfolded_production_bounds[0]
                + patient_rate_head[:, 0] * (self.misfolded_production_bounds[1] - self.misfolded_production_bounds[0]),
                self.aggregate_accumulation_bounds[0]
                + patient_rate_head[:, 1] * (self.aggregate_accumulation_bounds[1] - self.aggregate_accumulation_bounds[0]),
                patient_rate_head[:, 2],
            ],
            dim=-1,
        )
        patient_rate_bounds = patient_rate_bounds.unsqueeze(1).unsqueeze(2).expand(batch_size, regimen_count, self.node_count, -1)
        stochastic_payload: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        if self.quantized_mddm_rollout_operator is not None:
            if diffusion_context is None:
                raise ValueError("Native rollout requires compiled occupancy, support and physical_time; regimen features alone are insufficient.")
            initial = diffusion_context.get("initial_state")
            explicit_posterior = diffusion_context.get("posterior_context")
            if (initial is None) != (explicit_posterior is None):
                raise ValueError("Native initial state and posterior context must share one explicit lineage.")
            if initial is None:
                # Existing inference callers use observation-encoded state.
                initial = baseline_field.unsqueeze(1).expand(-1, regimen_count, -1, -1, -1)
            elif initial.shape != (batch_size, regimen_count, self.region_count, self.compartment_count, self.species_count):
                raise ValueError("Native independent initial_state axes mismatch")
            posterior_context = self.operator_context_proj(
                torch.cat([patient_context, regimen_context_expanded, region_prior_context], dim=-1)
            ) + self.operator_parameter_proj(parameter_context)
            context = dict(diffusion_context)
            context.pop("initial_state", None)
            if explicit_posterior is not None and explicit_posterior.shape != (
                    batch_size, regimen_count, self.region_count, self.compartment_count, self.operator_hidden_dim):
                raise ValueError("Native observation-derived posterior_context axes mismatch")
            context["posterior_context"] = explicit_posterior if explicit_posterior is not None else posterior_context.reshape(
                batch_size, regimen_count, self.region_count, self.compartment_count, self.operator_hidden_dim)
            time_count = (context["physical_time"].shape[2] if self.quantized_mddm_rollout_operator.quantity_profile
                          else self.rollout_step_count)
            shape_carrier = initial.unsqueeze(2).expand(-1, -1, time_count, -1, -1, -1)
            stochastic_payload = self.quantized_mddm_rollout_operator(
                shape_carrier, self.training, initial_state=initial,
                teacher_trajectory=teacher_trajectory, teacher_confidence=teacher_confidence,
                conditioning=context, native_operator=self.neuronsifter_world_action_operator,
                adjacency=self.node_adjacency, coordinates=self.node_coordinates,
                sample_ids=sample_ids, regimen_ids=regimen_ids, return_samples=True,
                field_normalization=self.field_normalization,
            )
            field_trajectory = stochastic_payload["trajectory_mean"]
            slow_field_trajectory = field_trajectory
            fast_field_trajectory = field_trajectory.repeat_interleave(self.fast_steps_per_slow_step, dim=2)
        else:
            fast_trajectory_steps = []
            slow_trajectory_steps = []
            for _ in range(self.rollout_step_count):
                for _ in range(self.fast_steps_per_slow_step):
                    if self.training and self.activation_checkpointing:
                        state = checkpoint(
                            self.step_update,
                            state,
                            patient_context,
                            regimen_context_expanded,
                            parameter_context,
                            region_prior_node,
                            region_prior_context,
                            compartment_source,
                            binding_gain,
                            pathology_relief,
                            clearance_support,
                            patient_rate_bounds,
                            use_reentrant=False,
                        )
                    else:
                        state = self.step_update(
                            state,
                            patient_context,
                            regimen_context_expanded,
                            parameter_context,
                            region_prior_node,
                            region_prior_context,
                            compartment_source,
                            binding_gain,
                            pathology_relief,
                            clearance_support,
                            patient_rate_bounds,
                        )
                    fast_trajectory_steps.append(
                        state.reshape(
                            batch_size,
                            regimen_count,
                            self.region_count,
                            self.compartment_count,
                            self.species_count,
                        )
                    )
                slow_trajectory_steps.append(fast_trajectory_steps[-1])
            fast_field_trajectory = torch.stack(fast_trajectory_steps, dim=2)
            slow_field_trajectory = torch.stack(slow_trajectory_steps, dim=2)
            field_trajectory = slow_field_trajectory
            if self.neuronsifter_world_action_operator is not None:
                stochastic_payload = self.neuronsifter_world_action_operator.trajectory_diffusion_summary(field_trajectory, self.training, self.node_adjacency)
                field_trajectory = stochastic_payload["trajectory_mean"]
                slow_field_trajectory = field_trajectory
                if self.fast_steps_per_slow_step == 1:
                    fast_field_trajectory = field_trajectory
        final_field = slow_field_trajectory[:, :, -1]
        router_diagnostics = dict(self._latest_router_diagnostics)
        sampler_diagnostics = stochastic_payload.get("sampler_diagnostics", {}) if stochastic_payload else {}
        if isinstance(sampler_diagnostics, dict):
            router_diagnostics.update(sampler_diagnostics)
        if self.quantized_mddm_rollout_operator is not None:
            router_diagnostics["fast_field_is_resampled_view"] = field_trajectory.new_tensor(self.fast_steps_per_slow_step != 1)
            router_diagnostics["fast_field_view_repeat_factor"] = field_trajectory.new_tensor(self.fast_steps_per_slow_step)
        return {
            "encoded_regimens": regimen_context,
            "field_trajectory": field_trajectory,
            **({"quantity_support":stochastic_payload["quantity_support"],
                "observable_prediction":stochastic_payload["observable_prediction"]}
               if "observable_prediction" in stochastic_payload else {}),
            "fast_field_trajectory": fast_field_trajectory,
            "slow_field_trajectory": slow_field_trajectory,
            "field_training_reconstruction": stochastic_payload.get("training_reconstruction", field_trajectory),
            "field_normalization_diagnostics": stochastic_payload.get("field_normalization_diagnostics", {}),
            "action_conditioning": [stochastic_payload["action_conditioning"]] if "action_conditioning" in stochastic_payload else [],
            "field_trajectory_std": stochastic_payload.get("trajectory_std", torch.zeros_like(field_trajectory)),
            "field_trajectory_quantiles": stochastic_payload.get("trajectory_quantiles", field_trajectory.unsqueeze(0)),
            "field_trajectory_samples": stochastic_payload.get("sample_trajectories", field_trajectory.unsqueeze(0)),
            "rollout_auxiliary_loss_per_action": stochastic_payload.get(
                "rollout_auxiliary_loss_per_action", field_trajectory.new_zeros(batch_size, regimen_count)),
            "rollout_auxiliary_loss": stochastic_payload.get(
                "rollout_auxiliary_loss",
                torch.zeros((), dtype=field_trajectory.dtype, device=field_trajectory.device),
            ),
            "ranking_stability": stochastic_payload.get(
                "ranking_stability",
                torch.ones((batch_size, regimen_count), dtype=field_trajectory.dtype, device=field_trajectory.device),
            ),
            "pkpd_field_trajectory": field_trajectory.index_select(-1, self.pkpd_indices),
            "transport_field_trajectory": field_trajectory.index_select(-1, self.transport_indices),
            "electrophysiology_field_trajectory": field_trajectory.index_select(-1, self.electrophysiology_indices),
            "final_field": final_field,
            "router_diagnostics": router_diagnostics,
            "solver_family_manifest": {
                "pkpd": self.pkpd_indices.tolist(),
                "transport_reaction": self.transport_indices.tolist(),
                "electrophysiology": self.electrophysiology_indices.tolist(),
            },
        }
