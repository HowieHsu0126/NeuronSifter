from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.neuronsifter_world_action import GraphSpectralTemporalWaveletTokenizer
from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
from Libs.models.networks.screening_system import ADScreeningSystem


def solver_test_kwargs() -> dict[str, object]:
    return {
        "baseline_field_policy": {
            "direct_scale": [0.05, 0.03, 1.0, 1.0, 1.0, 1.0, 1.0],
            "offset": [0.0, 0.0, 0.15, 0.10, 0.15, 0.10, 0.10],
            "pathology_gain": [0.0, 0.0, 0.65, 0.55, 0.0, 0.55, 0.40],
            "raw_field_gain": [1.0, 1.0, 0.20, 0.20, 0.20, 0.25, 0.20],
            "region_prior_mix": 0.15,
        },
        "rollout_update_policy": {
            "state_delta_scale": 0.10,
            "control_source_scale": 0.05,
            "region_prior_mix": 0.10,
            "activation_checkpointing": False,
            "action_representation": {
                "active_policy": "occupancy_program",
                "feature_groups": {
                    "occupancy_program_features": [0, 1, 2, 3, 4, 5, 6],
                    "drug_identity": [0],
                },
                "policies": {
                    "occupancy_program": {
                        "retained_feature_groups": ["occupancy_program_features"],
                    },
                    "drug_id": {
                        "retained_feature_groups": ["drug_identity"],
                    },
                },
            },
            "transition_backbone": {
                "kind": "factor_coupled_neural_operator",
                "operator_hidden_dim": 12,
                "operator_depth": 2,
                "operator_dropout_rate": 0.0,
                "operator_residual_scale": 0.05,
            },
        },
        "solver_state_layout": {
            "free_drug": 0,
            "target_bound_drug": 1,
            "misfolded_protein_load": 2,
            "aggregate_burden": 3,
            "neuroinflammation": 4,
            "clearance_flux": 5,
            "circuit_instability": 6,
        },
        "solver_config": {
            "diffusion_scale_by_compartment": [0.18, 0.14],
            "clearance_scale_by_compartment": [0.12, 0.08],
            "route_source_by_compartment": {
                "oral": [1.0, 0.24],
                "intravenous": [1.0, 0.34],
                "intrathecal": [0.28, 0.22],
            },
            "schedule_frequency_multipliers": {
                "daily": 1.0,
                "twice_daily": 1.18,
                "weekly": 0.62,
            },
            "target_binding_gain": 0.22,
            "amyloid_reduction_gain": 0.31,
            "plaque_clearance_gain": 0.14,
            "electrophysiology_relief_gain": 0.28,
            "electrophysiology_amyloid_gain": 0.55,
            "electrophysiology_plaque_gain": 0.34,
            "minimum_bound_fraction": 0.02,
            "maximum_bound_fraction": 0.88,
            "bound_decay_rate": 0.18,
            "parameter_context_scale_by_species": {
                "misfolded_protein_load": 0.10,
                "aggregate_burden": 0.08,
            },
            "prior_context_scale_by_drive": {
                "pathology": 0.05,
                "inflammation": 0.06,
            },
            "aggregate_clearance_scale": 0.10,
            "inflammation_coupling": {
                "misfolded_protein_load": 0.25,
                "aggregate_burden": 0.18,
            },
            "inflammation_decay": 0.12,
            "clearance_flux_scale": {
                "free_drug": 0.50,
                "misfolded_relief": 0.30,
            },
            "circuit_instability_scale": {
                "neuroinflammation": 0.24,
                "decay": 0.16,
            },
            "state_upper_bound_by_species": {
                "neuroinflammation": 1.5,
                "circuit_instability": 1.5,
            },
            "condition_rate_bounds": {
                "control": {
                    "misfolded_protein_production_rate": 0.015,
                    "aggregate_accumulation_rate": 0.007,
                },
                "ad_positive": {
                    "misfolded_protein_production_rate": 0.028,
                    "aggregate_accumulation_rate": 0.014,
                },
            },
            "regimen_feature_layout": {
                "dose_multiplier_index": 0,
                "intrathecal_indicator_index": 1,
                "weekly_indicator_index": 2,
                "oral_route_indicator_index": 3,
                "intravenous_route_indicator_index": 4,
                "daily_schedule_indicator_index": 5,
                "twice_daily_schedule_indicator_index": 6,
            },
        },
        "world_state_posterior_config": {
            "posterior_hidden_dim": 10,
            "posterior_dropout_rate": 0.0,
        },
    }


def quantized_mddm_test_config() -> dict[str, object]:
    return {
        "enabled": True,
        "rollout_kind": "quantized_mddm_rollout_operator",
        "mainline_scale_binding": "neuronsifter_debug",
        "residual_reference_policy": "initial_state",
        "training_objective": "absorbing_masked_categorical_nonempty_importance",
        "training_time_distribution": "uniform_open_unit_interval",
        "external_reference_policy": {
            "reference_policy": "project_native_no_direct_import",
            "mdlm_reference_path": "Libs/external/mddm/mdlm",
            "mdm_prime_reference_path": "Libs/external/mddm/mdm-prime",
            "excluded_references": ["MDM-Prime-v2"],
        },
        "token_axes": ["time", "region", "compartment", "species"],
        "residual_tokenization": {
            "quantization_policy": "straight_through_uniform_scalar_residual_codebook",
            "codebook_size": 16,
            "residual_min_value": -0.2,
            "residual_max_value": 0.2,
            "token_hidden_dim": 8,
            "token_mlp_multiplier": 2,
            "mask_token_value": 0.0,
        },
        "partial_mask_schedule": {
            "schedule_kind": "absorbing_linear_reverse",
            "training_sample_count": 2,
            "inference_sample_count": 3,
            "sample_chunk_size": 2,
            "denoising_token_chunk_size": 5,
            "training_denoising_step_count": 2,
            "inference_denoising_step_count": 2,
            "training_initial_mask_probability": 0.55,
            "inference_initial_mask_probability": 1.0,
            "final_mask_probability": 0.0,
            "replacement_blend": 0.65,
            "sampler_seed": 3407,
        },
        "inverse_reconstruction_policy": {
            "reconstruction_policy": "inverse_haar_add_initial",
            "preserve_initial_state": True,
            "state_lower_bound": 0.0,
            "ranking_stability_scale": 4.0,
            "quantile_levels": [0.05, 0.5, 0.95],
            "return_sample_trajectories": False,
        },
        "loss_weights": {
            "token_substitution_cross_entropy": 1.0,
            "inverse_residual_reconstruction": 0.5,
            "partial_mask_consistency": 0.0,
            "ranking_stability_consistency": 0.0,
        },
    }


def neuronsifter_test_backbone() -> dict[str, object]:
    return {
        "kind": "neuronsifter_world_action",
        "operator_hidden_dim": 12,
        "operator_depth": 1,
        "operator_dropout_rate": 0.0,
        "operator_residual_scale": 0.05,
        "attention_head_count": 3,
        "mechanism_state_token_count": 5,
        "node_coordinate_dim": 2,
        "dit_mlp_multiplier": 2,
        "mechanism_slice_temperature": 0.7,
        "graph_spectral_tokenizer": {
            "graph_mode_count": 6,
            "eigenvalue_temperature": 0.0,
            "normalize_laplacian": False,
            "spectral_residual_scale": 0.1,
        },
        "temporal_wavelet_tokenizer": {
            "wavelet_family": "haar",
            "wavelet_level_count": 2,
            "boundary_mode": "replicate",
        },
        "diffusion_sampler": {
            "sample_count": 4,
            "denoising_step_count": 3,
            "sample_chunk_size": 2,
            "training_sample_count": 2,
            "training_denoising_step_count": 1,
            "diffusion_noise_scale": 0.015,
            "score_step_size": 0.18,
            "terminal_noise_scale": 0.001,
            "ranking_stability_scale": 8.0,
            "quantile_levels": [0.05, 0.5, 0.95],
            "sampler_seed": 3407,
            "return_sample_trajectories": False,
        },
        "mechanism_moe": {
            "expert_names": [
                "pharmacology_expert",
                "bbb_cns_transport_expert",
                "protein_pathology_expert",
                "electrophysiology_rxd_expert",
            ],
            "top_k": 2,
            "capacity_factor": 1.25,
            "entropy_loss_weight": 0.002,
            "load_balance_loss_weight": 0.01,
            "expert_capacity_loss_weight": 0.01,
            "expert_hidden_multiplier": 2,
        },
        "operator_splitting": {
            "enabled": True,
            "bottleneck_divisor": 2,
            "expert_residual_scale": 0.08,
            "shared_coupling_residual_scale": 0.04,
            "composition_order": [
                "pharmacology_expert",
                "bbb_cns_transport_expert",
                "protein_pathology_expert",
                "electrophysiology_rxd_expert",
            ],
            "coupling_masks": {
                "pharmacology_expert": ["free_drug", "target_bound_drug"],
                "bbb_cns_transport_expert": ["free_drug", "clearance_flux"],
                "protein_pathology_expert": ["misfolded_protein_load", "aggregate_burden"],
                "electrophysiology_rxd_expert": ["calcium_concentration", "spike_rate"],
            },
        },
    }


def test_multiscale_rollout_exposes_fast_and_slow_trajectories() -> None:
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=5,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=3,
        fast_state_indices=[0, 1, 4],
        slow_state_indices=[2, 3, 5, 6],
        slow_update_scale=0.25,
        **solver_test_kwargs(),
    )
    patient_latent = torch.randn(2, 4)
    regimen_features = torch.randn(3, 7)
    output = model.score_regimens(patient_latent, regimen_features)
    assert output["field_trajectory"].shape == (2, 3, 5, 3, 2, 7)
    assert output["slow_field_trajectory"].shape == (2, 3, 5, 3, 2, 7)
    assert output["fast_field_trajectory"].shape == (2, 3, 15, 3, 2, 7)


def test_multiscale_sequence_rollout_keeps_compatibility_output_and_new_fast_sequence() -> None:
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=4,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=2,
        fast_state_indices=[0, 1, 4],
        slow_state_indices=[2, 3, 5, 6],
        slow_update_scale=0.25,
        **solver_test_kwargs(),
    )
    patient_latent = torch.randn(1, 4)
    regimen_sequence = torch.randn(1, 3, 7)
    rollout_output = model.rollout(patient_latent, regimen_sequence)
    assert rollout_output["field_trajectory_sequence"].shape == (1, 3, 4, 3, 2, 7)
    assert rollout_output["slow_field_trajectory_sequence"].shape == (1, 3, 4, 3, 2, 7)
    assert rollout_output["fast_field_trajectory_sequence"].shape == (1, 3, 8, 3, 2, 7)


def test_regimen_chunked_scoring_matches_full_scoring() -> None:
    torch.manual_seed(7)
    solver_kwargs = solver_test_kwargs()
    solver_kwargs["rollout_update_policy"]["activation_checkpointing"] = True
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=3,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=2,
        fast_state_indices=[0, 1, 4],
        slow_state_indices=[2, 3, 5, 6],
        slow_update_scale=0.25,
        **solver_kwargs,
    )
    model.eval()
    patient_latent = torch.randn(2, 4)
    regimen_features = torch.randn(5, 7)
    with torch.no_grad():
        full_output = model.score_regimens(patient_latent, regimen_features)
        chunked_output = model.score_regimens(patient_latent, regimen_features, regimen_chunk_size=2)
    for key in ["regimen_scores", "tri_score", "pkpd_consistency", "field_trajectory"]:
        assert torch.allclose(full_output[key], chunked_output[key], atol=1e-6)


def test_infer_world_state_uses_posterior_heads_not_region_prior_mean() -> None:
    torch.manual_seed(3)
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=3,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.2, 0.4] for _ in range(3)],
        fast_steps_per_slow_step=2,
        fast_state_indices=[0, 1, 4],
        slow_state_indices=[2, 3, 5, 6],
        slow_update_scale=0.25,
        **solver_test_kwargs(),
    )
    patient_latent = torch.randn(2, 4)
    observation_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    region_prior = torch.tensor([[0.2, 0.4], [0.2, 0.4], [0.2, 0.4]], dtype=torch.float32)
    encoded_payload = model.encode_patient(patient_latent, observation_mask, region_prior)
    world_state = model.infer_world_state(patient_latent, observation_mask, region_prior)
    direct_morphology = encoded_payload["region_prior_context"].mean(dim=0, keepdim=True).expand(patient_latent.shape[0], -1)
    assert world_state["z_morph"].shape == direct_morphology.shape
    assert not torch.allclose(world_state["z_morph"], direct_morphology)


def test_graph_spectral_temporal_wavelet_tokenizer_is_invertible_at_full_rank() -> None:
    torch.manual_seed(7)
    tokenizer = GraphSpectralTemporalWaveletTokenizer(
        hidden_dim=4,
        graph_mode_count=4,
        wavelet_level_count=2,
        eigenvalue_temperature=0.0,
        normalize_laplacian=False,
    )
    adjacency_matrix = torch.eye(4, dtype=torch.float32)
    field_state = torch.randn(2, 4, 3)
    coefficients, basis, _ = tokenizer.graph_spectral_transform(field_state, adjacency_matrix)
    reconstructed_field = tokenizer.graph_spectral_inverse(coefficients, basis)
    sequence = torch.randn(2, 5, 3)
    wavelet_payload = tokenizer.temporal_wavelet_analysis(sequence)
    reconstructed_sequence = tokenizer.temporal_wavelet_synthesis(wavelet_payload)
    assert torch.allclose(reconstructed_field, field_state, atol=1e-5)
    assert torch.allclose(reconstructed_sequence, sequence, atol=1e-5)


def test_quantized_mddm_rollout_quantizes_reconstructs_and_backpropagates() -> None:
    from Libs.tests.test_native_masked_diffusion import make_native_case
    operator, carrier, kwargs = make_native_case()
    output = operator(carrier, training=True, **kwargs)
    assert output["trajectory_mean"].shape == carrier.shape
    assert output["trajectory_quantiles"].shape == (3, *carrier.shape)
    output["rollout_auxiliary_loss"].backward()
    gradients = [p.grad for p in operator.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_action_representation_policy_masks_regimen_features() -> None:
    solver_kwargs = solver_test_kwargs()
    solver_kwargs["rollout_update_policy"]["action_representation"]["active_policy"] = "drug_id"
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=2,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=1,
        fast_state_indices=[0, 1, 5],
        slow_state_indices=[2, 3, 4, 6],
        slow_update_scale=0.25,
        **solver_kwargs,
    )
    regimen_features = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    masked_features = model.field_rollout_operator.action_conditioned_features(regimen_features)
    assert torch.equal(masked_features[:, 0], regimen_features[:, 0])
    assert torch.count_nonzero(masked_features[:, 1:]) == 0


def test_neuronsifter_world_action_exposes_stochastic_diagnostics() -> None:
    torch.manual_seed(17)
    solver_kwargs = solver_test_kwargs()
    solver_kwargs["rollout_update_policy"]["transition_backbone"] = {
        "kind": "neuronsifter_world_action",
        "operator_hidden_dim": 12,
        "operator_depth": 1,
        "operator_dropout_rate": 0.0,
        "operator_residual_scale": 0.05,
        "attention_head_count": 3,
        "mechanism_state_token_count": 5,
        "node_coordinate_dim": 2,
        "dit_mlp_multiplier": 2,
        "mechanism_slice_temperature": 0.7,
        "graph_spectral_tokenizer": {
            "graph_mode_count": 6,
            "eigenvalue_temperature": 0.0,
            "normalize_laplacian": False,
            "spectral_residual_scale": 0.1,
        },
        "temporal_wavelet_tokenizer": {
            "wavelet_family": "haar",
            "wavelet_level_count": 2,
            "boundary_mode": "replicate",
        },
        "diffusion_sampler": {
            "sample_count": 4,
            "denoising_step_count": 3,
            "sample_chunk_size": 2,
            "training_sample_count": 2,
            "training_denoising_step_count": 1,
            "diffusion_noise_scale": 0.015,
            "score_step_size": 0.18,
            "terminal_noise_scale": 0.001,
            "ranking_stability_scale": 8.0,
            "quantile_levels": [0.05, 0.5, 0.95],
            "sampler_seed": 3407,
            "return_sample_trajectories": False,
        },
        "mechanism_moe": {
            "expert_names": [
                "pharmacology_expert",
                "bbb_cns_transport_expert",
                "protein_pathology_expert",
                "electrophysiology_rxd_expert",
            ],
            "top_k": 2,
            "capacity_factor": 1.25,
            "entropy_loss_weight": 0.002,
            "load_balance_loss_weight": 0.01,
            "expert_capacity_loss_weight": 0.01,
            "expert_hidden_multiplier": 2,
        },
        "operator_splitting": {
            "enabled": True,
            "bottleneck_divisor": 2,
            "expert_residual_scale": 0.08,
            "shared_coupling_residual_scale": 0.04,
            "composition_order": [
                "pharmacology_expert",
                "bbb_cns_transport_expert",
                "protein_pathology_expert",
                "electrophysiology_rxd_expert",
            ],
            "coupling_masks": {
                "pharmacology_expert": ["free_drug", "target_bound_drug"],
                "bbb_cns_transport_expert": ["free_drug", "clearance_flux"],
                "protein_pathology_expert": ["misfolded_protein_load", "aggregate_burden"],
                "electrophysiology_rxd_expert": ["calcium_concentration", "spike_rate"],
            },
        },
    }
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=3,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=1,
        fast_state_indices=[0, 1, 5],
        slow_state_indices=[2, 3, 4, 6],
        slow_update_scale=0.25,
        **solver_kwargs,
    )
    model.eval()
    patient_latent = torch.randn(2, 4)
    regimen_features = torch.randn(3, 7)
    output = model.score_regimens(patient_latent, regimen_features)
    assert output["field_trajectory"].shape == (2, 3, 3, 3, 2, 7)
    assert output["field_trajectory_std"].shape == output["field_trajectory"].shape
    assert output["field_trajectory_quantiles"].shape == (3, 2, 3, 3, 3, 2, 7)
    assert output["ranking_stability"].shape == (2, 3)
    diagnostics = output["router_diagnostics"]
    assert diagnostics["expert_utilization"].shape == (4,)
    assert diagnostics["operator_splitting_expert_utilization"].shape == (4,)
    assert diagnostics["operator_splitting_species_attribution"].shape[0] == 4
    assert diagnostics["operator_splitting_coupling_matrix"].shape == (4, 4)
    assert diagnostics["operator_splitting_score_norm_by_expert"].shape == (4,)
    assert diagnostics["operator_splitting_uncertainty_by_expert"].shape == (4,)
    assert diagnostics["operator_splitting_case_activation"].shape == (2, 4)
    assert torch.isfinite(diagnostics["mechanism_token_entropy"])
    assert torch.isfinite(diagnostics["operator_splitting_router_entropy"])
    assert torch.isfinite(diagnostics["graph_projection_error"])
    assert torch.isfinite(diagnostics["temporal_wavelet_reconstruction_error"])
    assert int(diagnostics["stochastic_sample_count"].item()) == 4
    assert int(diagnostics["denoising_step_count"].item()) == 3
    output["regimen_scores"].mean().backward()
    assert model.field_rollout_operator.neuronsifter_world_action_operator is not None


def test_field_rollout_uses_quantized_mddm_sampler_when_declared() -> None:
    torch.manual_seed(29)
    solver_kwargs = solver_test_kwargs()
    transition_backbone = neuronsifter_test_backbone()
    transition_backbone["quantized_mddm_rollout"] = quantized_mddm_test_config()
    solver_kwargs["rollout_update_policy"]["transition_backbone"] = transition_backbone
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=3,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=1,
        fast_state_indices=[0, 1, 5],
        slow_state_indices=[2, 3, 4, 6],
        slow_update_scale=0.25,
        **solver_kwargs,
    )
    model.eval()
    patient_latent = torch.randn(2, 4)
    regimen_features = torch.randn(3, 7)
    shape = (2, 3, 3, 3, 2, 1)
    context = {"occupancy": torch.rand(shape), "support": torch.ones(shape),
               "physical_time": torch.arange(3.).reshape(1, 1, 3, 1, 1, 1).expand(shape)}
    output = model.score_regimens(patient_latent, regimen_features, diffusion_context=context)
    assert output["field_trajectory"].shape == (2, 3, 3, 3, 2, 7)
    assert output["field_trajectory_quantiles"].shape == (3, 2, 3, 3, 3, 2, 7)
    diagnostics = output["router_diagnostics"]
    assert diagnostics["mddm_discrete_codebook_size"].item() == 16.0
    assert diagnostics["mddm_sample_count"].item() == 3.0
    assert output["field_trajectory_samples"].shape[0] == 3
    assert not torch.equal(output["field_trajectory_samples"][0], output["field_trajectory_samples"][1])
    assert diagnostics["mddm_teacher_loss_evaluated"] == 0
    assert "mddm_inverse_reconstruction_error" not in diagnostics
    assert torch.isfinite(output["rollout_auxiliary_loss"])
    minimal_output = model.score_regimens(patient_latent, regimen_features, include_auxiliary_outputs=False, diffusion_context=context)
    assert torch.isfinite(minimal_output["rollout_auxiliary_loss"])
    chunked = model.score_regimens(patient_latent, regimen_features, regimen_chunk_size=2, diffusion_context=context)
    assert torch.equal(output["field_trajectory"], chunked["field_trajectory"])
    assert chunked["router_diagnostics"]["operator_splitting_case_activation"].shape == (2, 3, 4)
    output["regimen_scores"].mean().backward()
    assert model.field_rollout_operator.neuronsifter_world_action_operator is not None
    assert model.field_rollout_operator.quantized_mddm_rollout_operator is not None


def test_neuronsifter_world_action_operator_does_not_import_external_neural_operator_code() -> None:
    text = (REPO_ROOT / "Libs/models/modules/neuronsifter_world_action.py").read_text(encoding="utf-8")
    mddm_text = (REPO_ROOT / "Libs/models/modules/quantized_mddm_rollout_operator.py").read_text(encoding="utf-8")
    assert "Libs.external.neural_operator" not in text
    assert "from neural_operator" not in text
    assert "Libs.external.mddm" not in mddm_text
    assert "from mdlm" not in mddm_text


def test_transolver_microenvironment_operator_exposes_router_diagnostics() -> None:
    torch.manual_seed(11)
    solver_kwargs = solver_test_kwargs()
    solver_kwargs["rollout_update_policy"]["transition_backbone"] = {
        "kind": "transolver_microenvironment_operator",
        "operator_hidden_dim": 12,
        "operator_depth": 2,
        "operator_dropout_rate": 0.0,
        "operator_residual_scale": 0.05,
        "attention_head_count": 3,
        "physical_state_slice_count": 4,
        "node_coordinate_dim": 2,
        "physics_mlp_multiplier": 2,
        "encoder_decoder_pretraining": {
            "encoder_depth": 1,
            "decoder_depth": 1,
            "checkpoint_attention_blocks": False,
        },
        "mechanism_moe": {
            "expert_names": [
                "pharmacology_expert",
                "bbb_cns_transport_expert",
                "protein_pathology_expert",
                "electrophysiology_rxd_expert",
            ],
            "top_k": 2,
            "capacity_factor": 1.25,
            "entropy_loss_weight": 0.002,
            "load_balance_loss_weight": 0.01,
            "expert_capacity_loss_weight": 0.01,
            "expert_hidden_multiplier": 2,
        },
    }
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=3,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=1,
        fast_state_indices=[0, 1, 5],
        slow_state_indices=[2, 3, 4, 6],
        slow_update_scale=0.25,
        **solver_kwargs,
    )
    patient_latent = torch.randn(2, 4)
    regimen_features = torch.randn(3, 7)
    output = model.score_regimens(patient_latent, regimen_features)
    assert output["field_trajectory"].shape == (2, 3, 3, 3, 2, 7)
    diagnostics = output["router_diagnostics"]
    assert set(diagnostics) >= {"expert_utilization", "router_entropy", "capacity_excess", "load_balance_loss", "router_regularization_loss", "slice_entropy", "slice_temperature_mean"}
    assert diagnostics["expert_utilization"].shape == (4,)
    assert torch.isfinite(diagnostics["router_entropy"])
    assert torch.isfinite(diagnostics["slice_entropy"])
    output["regimen_scores"].mean().backward()
    assert model.field_rollout_operator.transolver_operator is not None


def test_multimodal_world_state_moe_routes_patient_evidence() -> None:
    torch.manual_seed(13)
    model = ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=3,
        regimen_feature_dim=7,
        hidden_dim=8,
        latent_dim=6,
        dropout_rate=0.0,
        compartment_count=2,
        tri_weight_by_component={
            "calibration": 1.0,
            "coverage": 1.0,
            "pkpd_physics_consistency": 1.0,
            "ranking_stability": 1.0,
            "ood_score": 1.0,
        },
        region_count=3,
        species_count=7,
        rollout_step_count=2,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        region_prior_feature_dim=2,
        default_region_prior=[[0.0, 0.0] for _ in range(3)],
        fast_steps_per_slow_step=1,
        fast_state_indices=[0, 1, 5],
        slow_state_indices=[2, 3, 4, 6],
        slow_update_scale=0.25,
        multimodal_moe_config={
            "enabled": True,
            "expert_names": ["transcriptomics_expert", "metabolomics_expert", "solver_observation_expert"],
            "top_k": 2,
            "capacity_factor": 1.25,
            "entropy_loss_weight": 0.002,
            "load_balance_loss_weight": 0.01,
            "expert_capacity_loss_weight": 0.01,
            "expert_hidden_multiplier": 2,
        },
        **solver_test_kwargs(),
    )
    output = model.encode_patient(torch.randn(2, 4))
    diagnostics = output["multimodal_router_diagnostics"]
    assert diagnostics["expert_utilization"].shape == (3,)
    assert torch.isfinite(diagnostics["router_entropy"])
