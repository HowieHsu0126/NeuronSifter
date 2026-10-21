from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Libs
import Libs.exp.training.pretrain_microenvironment_world_model as pretrain_world_model
from Libs.exp.training.pretrain_microenvironment_world_model import pretraining_artifact_root
from Libs.exp.training.train_screening_system import (
    artifact_dir,
    assert_tri_component_shape_contract,
    cache_schema_version,
    freeze_signature_sha256,
    json_sha256,
    load_scope_training_cache,
    remap_counterfactual_pairs,
    resolve_training_window,
    source_artifact_hashes,
    training_cache_manifest_path,
    training_cache_tensor_path,
    training_job_cache_build_allowed,
    training_regimen_indices,
    train_one_scope,
)
from Libs.models.networks import screening_system
from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES, tri_target_tensors
from Libs.utils.data.iclr_contract_union import build_config_stack_contract_report
from Libs.utils.data.raw_data_registry import load_config, load_config_with_provenance
from Libs.utils.evaluation.formal_scale_audit import classify_scale
from Libs.utils.exp.experiment_manifests import load_experiment_family
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale, scale_parameter_estimate
from Libs.utils.model.screening_model_config import rollout_update_policy_from_config


def test_repo_local_libs_package_is_import_origin() -> None:
    assert Path(Libs.__file__).resolve() == REPO_ROOT / "Libs/__init__.py"
    assert Path(screening_system.__file__).resolve() == REPO_ROOT / "Libs/models/networks/screening_system.py"


def test_config_stack_composes_required_sections() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    for section_name in [
        "project",
        "foundation_twin",
        "schemas",
        "screening_pipeline",
        "dynamics_ready_registry",
        "validation",
        "paper_experiments",
        "cluster_registry",
    ]:
        assert section_name in config
    assert "measurement_semantics_registry" in config
    assert "digital_twin_evidence_fabric" in config
    assert "world_state_ready_registry" in config["data"]
    assert "crossmodal_anchor_ready_registry" in config["data"]
    assert "simulation_teacher_ready_registry" in config["data"]
    assert "full_microenvironment_teacher_ready_registry" in config["data"]
    assert "counterfactual_benchmark_ready_registry" in config["data"]
    assert "feedback_ready_registry" in config["data"]
    assert "adni_controlled_access" not in config
    assert "controlled_research_stack" not in config


def test_digital_twin_evidence_fabric_has_single_config_and_schema_owners() -> None:
    config, provenance = load_config_with_provenance(REPO_ROOT / "Libs/configs/config_stack.yaml")
    owner = "Libs/configs/data/digital_twin_evidence_fabric.yaml"
    assert provenance["include_order"].count(owner) == 1
    assert all(
        owners == [owner]
        for path, owners in provenance["leaf_owners"].items()
        if path.startswith("digital_twin_evidence_fabric.")
    )
    for field_key in [
        "digital_twin_evidence_record_fields",
        "digital_twin_evidence_relationship_fields",
        "digital_twin_evidence_role_matrix_fields",
    ]:
        assert provenance["leaf_owners"][f"schemas.{field_key}"] == ["Libs/configs/schema/schemas.yaml"]
        assert config["schemas"][field_key]
    assert config["digital_twin_evidence_fabric"]["version"] == "digital_twin_evidence_fabric_v1"
    adapters = {
        row["adapter_id"]: row
        for row in config["digital_twin_evidence_fabric"]["source_adapters"]
    }
    assert {row["path"] for row in adapters.values() if row["adapter_id"].startswith("public_foundation_corpus_")} == {
        "Output/audits/public_foundation_corpus_evidence_bundle.json",
        "Output/audits/public_foundation_corpus_canonical_observation_blocks.json",
        "Output/audits/public_foundation_corpus_frozen_splits.json",
        "Output/audits/public_foundation_corpus_v2_evidence_bundle.json",
        "Output/audits/public_foundation_corpus_v2_canonical_observation_blocks.json",
        "Output/audits/public_foundation_corpus_v2_frozen_splits.json",
    }
    assert adapters["public_foundation_corpus_v1_evidence_bundle"]["expected_identity"]["corpus_hash"] == (
        "06ae88daf3c44be2c2bb96e51c939554e160b205138ec96f471adb5d82eee791"
    )
    assert adapters["public_foundation_corpus_v2_evidence_bundle"]["expected_identity"]["parent_corpus_identity_hash"] == (
        "06ae88daf3c44be2c2bb96e51c939554e160b205138ec96f471adb5d82eee791"
    )
    assert not any(row["adapter_id"].endswith("_identity") for row in adapters.values())


def test_digital_twin_canonical_state_has_single_config_and_schema_owners() -> None:
    config, provenance = load_config_with_provenance(REPO_ROOT / "Libs/configs/config_stack.yaml")
    owner = "Libs/configs/data/digital_twin_canonical_state.yaml"
    assert provenance["include_order"].count(owner) == 1
    assert all(
        owners == [owner]
        for path, owners in provenance["leaf_owners"].items()
        if path.startswith("digital_twin_canonical_state.")
    )
    for field_key in [
        "digital_twin_canonical_state_block_fields",
        "digital_twin_canonical_state_axis_metadata_fields",
        "digital_twin_canonical_state_element_mask_fields",
        "digital_twin_canonical_state_value_storage_fields",
        "digital_twin_canonical_state_aggregation_semantics_fields",
        "digital_twin_canonical_state_eligibility_fields",
    ]:
        assert provenance["leaf_owners"][f"schemas.{field_key}"] == ["Libs/configs/schema/schemas.yaml"]
        assert config["schemas"][field_key]
    assert config["digital_twin_canonical_state"]["version"] == "digital_twin_canonical_state_v1"


def test_config_stack_exposes_one_owner_union_provenance() -> None:
    config, provenance = load_config_with_provenance(REPO_ROOT / "Libs/configs/config_stack.yaml")
    owner = config["iclr_design_contract_union"]["runtime_owner"]["owner_file"]
    assert provenance["include_order"].count(owner) == 1
    assert all(owners == [owner] for path, owners in provenance["leaf_owners"].items() if path.startswith("iclr_design_contract_union."))
    assert all(owners == [owner] for path, owners in provenance["leaf_owners"].items() if path.startswith("validation.design_contract_coverage."))
    report = build_config_stack_contract_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert report["status"] == "pass"
    assert report["blocker_count"] == 0
    assert report["preview_runtime_includes"] == []


def test_config_loader_rejects_duplicate_include_paths(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text("value: 1\n", encoding="utf-8")
    stack = tmp_path / "stack.yaml"
    stack.write_text("config_includes:\n- child.yaml\n- child.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="include path is repeated"):
        load_config(stack)


def test_config_loader_rejects_include_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("config_includes:\n- second.yaml\n", encoding="utf-8")
    second.write_text("config_includes:\n- first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="include cycle detected"):
        load_config(first)


def test_config_loader_rejects_missing_includes(tmp_path: Path) -> None:
    stack = tmp_path / "stack.yaml"
    stack.write_text("config_includes:\n- missing.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="include is missing"):
        load_config(stack)


def test_measurement_semantics_and_scaler_contracts_are_yaml_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    registry = config["measurement_semantics_registry"]
    assert registry["relative_paths"]["measurement_semantics_catalog"] == "measurement_semantics/measurement_semantics_catalog.tsv"
    assert registry["scaler_artifact_policy"]["fold_fit_policy"] == "train_split_only_fit_validation_test_apply"
    assert registry["scaler_artifact_policy"]["inverse_transform_required"] is True
    assert registry["modality_default_matrix_semantics"]["transcriptomics"] == "expression_signal"
    assert registry["modality_default_matrix_semantics"]["metabolomics"] == "metabolite_intensity"
    assert config["model_input_registry"]["bundle_filenames"]["feature_scaler"] == "feature_scaler_artifact.tsv"
    definitions = registry["matrix_semantics_definitions"]
    assert definitions["expression_signal"]["physical_interpretation_policy"] == "proxy_unit_not_physical_unit"
    assert definitions["expression_signal"]["unit_conversion_chain"]
    assert definitions["expression_signal"]["uncertainty_semantics"]
    assert definitions["metabolite_intensity"]["default_training_transform"] == "auto_log1p_then_zscore"


def test_scripts_default_to_config_stack() -> None:
    script_root = REPO_ROOT / "Libs/scripts"
    for path in script_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "Libs/configs/data/raw_data_collection.yaml" not in text


def test_submit_scripts_read_cluster_registry() -> None:
    for script_name in [
        "submit_candidate_validation_execution.sh",
        "submit_dynamics_mainline_ensemble_refresh.sh",
        "submit_active_learning_loop.sh",
        "submit_publication_claim_suite.sh",
        "submit_dynamics_profile_suite.sh",
        "submit_framework_hardening_suite.sh",
        "submit_framework_hardening_profile.sh",
    ]:
        text = (REPO_ROOT / "Libs/exp/submission" / script_name).read_text(encoding="utf-8")
        assert "cluster_registry" in text
        assert "SBATCH --partition=gpu1" not in text
        assert "SBATCH --nodelist=gpu03" not in text
        assert "SBATCH --cpus-per-task=4" not in text
    active_learning_submit = (REPO_ROOT / "Libs/exp/submission/submit_active_learning_loop.sh").read_text(encoding="utf-8")
    assert "--source moleculenet_deepchem" in active_learning_submit


def test_dynamics_submission_exports_torch_allocator_contract() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    training_defaults = config["cluster_registry"]["dynamics"]["profile_defaults"]["training"]
    assert training_defaults["pytorch_alloc_conf"] == "expandable_segments:True"
    assert training_defaults["cublas_workspace_config"] == ":4096:8"
    script_text = (REPO_ROOT / "Libs/exp/submission/submit_dynamics_profile_suite.sh").read_text(encoding="utf-8")
    assert "cluster_registry" in script_text
    assert "PYTORCH_ALLOC_CONF" in script_text
    assert "PYTORCH_CUDA_ALLOC_CONF" not in script_text


def test_formal_seed_values_flow_from_yaml_to_experiment_runtime() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    profile = load_experiment_family("dynamics_mainline_ad", config)
    assert profile["runtime"]["seed_values"] == [3407, 42, 26]


def test_publication_claim_suite_profiles_are_declared_with_submission_coverage() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    suite = config["paper"]["publication_claim_suite"]
    required_profiles = {
        "dynamics_mainline_ad",
        "action_conditioning_drug_identity",
        "action_conditioning_target_identity",
        "action_conditioning_scalar_exposure",
        "occupancy_shallow_reference",
        "rollout_operator_deterministic_neuronsifter",
        "rollout_operator_transolver",
        "rollout_operator_factor_coupled",
        "ad_screening_without_pkpd_prior",
        "ad_screening_without_tri_gate",
        "dynamics_without_local_response_heads",
    }
    assert required_profiles.issubset(set(suite["dynamics_profiles"]))
    for profile_name in required_profiles | set(suite["secondary_benchmark_profiles"]):
        profile = load_experiment_family(profile_name, config)
        assert profile["execution"]["train_commands"]
        assert profile["runtime"]["seed_values"] == [3407, 42, 26]
    submit_text = (REPO_ROOT / "Libs/exp/submission/submit_publication_claim_suite.sh").read_text(encoding="utf-8")
    assert "submit_dynamics_profile_suite.sh" in submit_text
    assert "submit_runnable_baseline_suite.sh" in submit_text
    assert "submit_neuronsifter_fsdp_smoke.sh" in submit_text
    assert "submit_framework_hardening_suite.sh" in submit_text
    for profile_name in suite["framework_audit_profiles"]:
        profile = load_experiment_family(profile_name, config)
        assert profile["execution"]["audit_commands"]


def test_pan_neuro_benchmark_contracts_are_declared_for_all_disease_families() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    disease_programs = config["benchmark_registry"]["disease_programs"]
    assert set(disease_programs) >= {"ad_control", "pd_control", "ftd_control", "als_control"}
    assert disease_programs["ftd_control"]["contract_status"] == "incomplete_public_contract"
    assert disease_programs["als_control"]["contract_status"] == "incomplete_public_contract"


def test_mechanistic_teacher_runtime_contracts_are_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    dynamics_config = config["dynamics_ready_registry"]
    assert dynamics_config["teacher_source_policy"]["primary_pbpk_teacher"] == "osp_pksim_rollout"
    assert dynamics_config["teacher_source_policy"]["legacy_solver_role"] == "compatibility_baseline"
    assert dynamics_config["osp_pksim_runtime"]["runtime_strategy"] == "reproducible_environment_script"
    assert dynamics_config["neuron_rxd_teacher"]["disease_scope"] == "ad_control"
    assert dynamics_config["neuron_rxd_teacher"]["teacher_role"] == "mechanism_teacher"
    assert dynamics_config["chembl_pbpk_parameterization"]["teacher_role"] == "parameter_predicted_teacher"
    assert dynamics_config["osp_pksim_runtime"]["model_library"]["local_repo_path"].endswith("OSP-PBPK-Model-Library-12.2.1")
    assert dynamics_config["osp_pksim_runtime"]["model_library"]["library_discovery"]["enabled"] is True
    assert dynamics_config["osp_pksim_runtime"]["raw_output_cache"]["reuse_existing_outputs"] is True
    assert dynamics_config["osp_pksim_runtime"]["raw_output_cache"]["regenerate_invalid_cache"] is True
    assert dynamics_config["osp_pksim_runtime"]["raw_output_cache"]["standardization_policy_uid"]
    output_mapping = dynamics_config["osp_pksim_runtime"]["output_path_mapping"]
    assert "IndividualId" in output_mapping["csv_metadata_columns"]
    assert "brain" in output_mapping["measure_column_keep_regex"]
    assert output_mapping["max_measure_columns_per_chunk"] > 0
    assert dynamics_config["neuron_rxd_teacher"]["model_sources"]["modeldb"]["source_model_id"] == "modeldb_romani_2013_ca3_ca1"
    assert dynamics_config["neuron_rxd_teacher"]["model_sources"]["allen_cell_types"]["retrieval_strategy"] == "allen_api_neuronal_model_download"


def test_neuronsifter_mainline_and_ablation_templates_are_yaml_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    benchmark_config = config["screening_pipeline"]["screening_benchmark"]
    transition_backbone = benchmark_config["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert transition_backbone["kind"] == "neuronsifter_world_action"
    assert transition_backbone["operator_registry"]["mainline_kind"] == "neuronsifter_world_action"
    assert transition_backbone["operator_registry"]["main_rollout_sampler_kind"] == "quantized_mddm_rollout_operator"
    assert set(transition_backbone["operator_registry"]["ablation_kinds"]) >= {"transolver_microenvironment_operator", "factor_coupled_neural_operator", "continuous_residual_neuronsifter_sampler"}
    assert transition_backbone["external_operator_references"]["reference_policy"] == "project_native_no_direct_import"
    assert transition_backbone["external_operator_references"]["mdlm_reference_path"] == "Libs/external/mddm/mdlm"
    assert transition_backbone["external_operator_references"]["mdm_prime_reference_path"] == "Libs/external/mddm/mdm-prime"
    assert transition_backbone["diffusion_sampler"]["sample_count"] == 64
    assert transition_backbone["diffusion_sampler"]["denoising_step_count"] == 100
    assert transition_backbone["quantized_mddm_rollout"]["enabled"] is True
    assert transition_backbone["quantized_mddm_rollout"]["registry_binding"] == "quantized_mddm_main_rollout"
    mddm_config = config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]
    assert mddm_config["rollout_kind"] == "quantized_mddm_rollout_operator"
    assert mddm_config["residual_tokenization"]["codebook_size"] == 64
    assert mddm_config["partial_mask_schedule"]["training_denoising_step_count"] == 4
    assert mddm_config["partial_mask_schedule"]["denoising_token_chunk_size"] == 1048576
    assert mddm_config["inverse_reconstruction_policy"]["reconstruction_policy"] == "inverse_haar_add_initial"
    assert mddm_config["external_reference_policy"]["reference_policy"] == "project_native_no_direct_import"
    assert "MDM-Prime-v2" in mddm_config["external_reference_policy"]["excluded_references"]
    expanded_rollout_policy = rollout_update_policy_from_config(config)
    expanded_mddm = expanded_rollout_policy["transition_backbone"]["quantized_mddm_rollout"]
    assert expanded_mddm["residual_tokenization"]["codebook_size"] == 64
    assert expanded_mddm["partial_mask_schedule"]["inference_sample_count"] == 16
    assert expanded_mddm["partial_mask_schedule"]["denoising_token_chunk_size"] == 1048576
    assert benchmark_config["training"]["pretraining_objectives"]["masked_state_ratio"] == 0.3
    assert benchmark_config["loss_weights"]["quantized_mddm_rollout_loss"] > 0.0
    memory_policy = benchmark_config["training"]["memory_policy"]
    assert benchmark_config["training"]["batch_size"] == memory_policy["effective_batch_size"]
    assert memory_policy["micro_batch_size"] * memory_policy["gradient_accumulation_steps"] == memory_policy["effective_batch_size"]
    assert memory_policy["mixed_precision"]["dtype"] == "bfloat16"
    teacher_families = {row["family_key"] for row in benchmark_config["training"]["teacher_family_training"]["families"]}
    assert teacher_families >= {"mesoscopic_solver_teacher", "full_microenvironment_teacher", "osp_pbpk_teacher", "neuron_rxd_teacher", "chembl_parameterized_teacher"}
    assert transition_backbone["graph_spectral_tokenizer"]["spectral_residual_scale"] > 0.0
    assert transition_backbone["operator_splitting"]["enabled"] is True
    assert "solver_residual_expert" in transition_backbone["operator_splitting"]["composition_order"]
    assert set(transition_backbone["operator_splitting"]["coupling_masks"]["solver_residual_expert"]) >= {"free_drug", "vascular_perivascular_transport"}
    assert transition_backbone["parameter_budget"]["training_scale"] == "single_gpu_trainable"
    assert transition_backbone["parameter_budget"]["active_parameter_lower_bound"] == 100000000
    assert "electrophysiology_rxd_expert" in transition_backbone["mechanism_moe"]["expert_names"]
    assert "solver_residual_expert" in transition_backbone["mechanism_moe"]["expert_names"]
    profiles = transition_backbone["neuronsifter_profiles"]
    assert profiles["active_screening_profile"] == "neuronsifter_large"
    assert set(profiles["profiles"]) >= {"neuronsifter_xl", "neuronsifter_large", "neuronsifter_student"}
    assert profiles["profiles"]["neuronsifter_large"]["denoising_step_count"] <= 16
    assert profiles["distillation"]["enabled"] is True
    transolver_ablation = transition_backbone["transolver_ablation_template"]
    assert transolver_ablation["kind"] == "transolver_microenvironment_operator"
    assert transolver_ablation["encoder_decoder_pretraining"]["encoder_depth"] == 12
    assert transolver_ablation["encoder_decoder_pretraining"]["decoder_depth"] == 4
    deterministic_profile = load_experiment_family("rollout_operator_deterministic_neuronsifter", config)
    deterministic_backbone_override = deterministic_profile["config_overrides"]["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert deterministic_backbone_override["quantized_mddm_rollout"]["enabled"] is False
    multimodal_moe = benchmark_config["model_architecture"]["multimodal_moe"]
    assert multimodal_moe["enabled"] is True
    assert set(multimodal_moe["expert_names"]) >= {"transcriptomics_expert", "solver_observation_expert"}


def test_screening_training_uses_yaml_declared_regimen_minibatches() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    policy = training_config["regimen_minibatch"]
    assert policy["enabled"] is True
    assert policy["candidate_count"] == 32
    assert policy["top_label_count"] > 0
    assert policy["include_counterfactual_pairs"] is True
    assert policy["counterfactual_pair_coverage"] == "stochastic_pair_window"
    assert policy["counterfactual_pair_window_size"] == 12
    assert training_config["progress_logging"]["micro_batch_interval"] > 0
    labels = np.zeros((2, 64), dtype=np.float32)
    labels[:, 10] = 1.0
    labels[:, 11] = 0.9
    indices = training_regimen_indices(
        config,
        64,
        labels,
        [(2, 3), (4, 5)],
        np.random.default_rng(3407),
    )
    assert indices.shape[0] == policy["candidate_count"]
    assert {2, 3, 4, 5, 10, 11}.issubset(set(indices.tolist()))
    assert set(remap_counterfactual_pairs([(2, 3), (4, 5), (60, 61)], indices)[:2]) == {
        (0, 1),
        (2, 3),
    }


def test_paper_grade_runtime_protocol_declares_checkpointable_profile_dag() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    protocol = config["paper_grade_full_suite"]["runtime_protocol"]
    screening_training = protocol["screening_training"]
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    assert protocol["checkpointable_split_enabled"] is True
    assert protocol["cpu_precompute"]["enabled"] is True
    assert protocol["gpu_concurrency_policy"]["exact_prefix_queue_scan_required_before_live"] is True
    assert protocol["profiling"]["record_micro_batch_per_second"] is True
    assert screening_training["max_epochs"] == 40
    assert screening_training["patience_epochs"] == 6
    assert screening_training["epoch_window_count"] == len(screening_training["epoch_window_labels"])
    assert set(protocol["commands"]) == {
        "cache_precompute",
        "pretraining",
        "screening_window",
        "evaluation",
        "active_learning_replay",
        "export",
    }
    assert training_config["cache_policy"]["enabled"] is True
    assert training_config["cache_policy"]["require_cache_for_formal"] is True
    assert training_config["cache_policy"]["allow_training_job_cache_build"] is False
    assert training_job_cache_build_allowed(config) is False
    assert training_config["data_loader"]["num_workers"] == 2
    assert training_config["data_loader"]["pin_memory"] is False
    assert config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]["formal_artifact_subdir_template"] == "{benchmark_name}/world_model_pretraining"
    assert resolve_training_window(40, 3, 8) == (10, 15)
    submit_text = (REPO_ROOT / "Libs/exp/submission/submit_dynamics_profile_suite.sh").read_text(encoding="utf-8")
    assert "PROFILE_COMPLETION_DEPENDENCIES" in submit_text
    assert "SPLIT_COMMANDS[cache_precompute]" in submit_text
    assert "SPLIT_COMMANDS[screening_window]" in submit_text
    assert "PROFILE_HAS_PRETRAINING" in submit_text
    assert "profile_complete" in submit_text
    suite_submit_text = (REPO_ROOT / "Libs/exp/submission/submit_paper_grade_full_suite.sh").read_text(encoding="utf-8")
    assert "index($2, prefix) == 1" in suite_submit_text


def test_pretraining_artifacts_are_benchmark_scoped(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["artifact_root"] = str(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] = "dynamics_twin_ad_mainline_seed_40"
    config["selected_paper_experiment_profile"] = "dynamics_mainline_ad"
    config["selected_formal_seed"] = 40
    assert pretraining_artifact_root(config, False) == tmp_path / "dynamics_twin_ad_mainline_seed_40" / "world_model_pretraining"
    assert pretraining_artifact_root(config, True) == tmp_path


def test_teacher_family_loader_groups_chunk_reads_without_reordering(monkeypatch, tmp_path: Path) -> None:
    dataset = pretrain_world_model.MicroenvironmentTeacherFamilyDataset.__new__(
        pretrain_world_model.MicroenvironmentTeacherFamilyDataset
    )
    dataset.native_registry = None
    dataset.config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    dataset.config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["quantized_mddm_rollout"]["enabled"] = False
    dataset.rows = [
        {
            "teacher_family": "full_microenvironment_teacher",
            "trajectory_store_path": str(tmp_path / "store_a"),
            "trajectory_index": 3,
            "teacher_loss_weight": 1.0,
            "store_format": "neuronsifter_chunk_store",
            "trajectory_array_name": "trajectories",
            "uncertainty_array_name": "teacher_uncertainty",
            "default_uncertainty_value": 0.25,
        },
        {
            "teacher_family": "full_microenvironment_teacher",
            "trajectory_store_path": str(tmp_path / "store_a"),
            "trajectory_index": 1,
            "teacher_loss_weight": 0.5,
            "store_format": "neuronsifter_chunk_store",
            "trajectory_array_name": "trajectories",
            "uncertainty_array_name": "teacher_uncertainty",
            "default_uncertainty_value": 0.25,
        },
        {
            "teacher_family": "mesoscopic_solver_teacher",
            "trajectory_store_path": str(tmp_path / "store_b"),
            "trajectory_index": 2,
            "teacher_loss_weight": 0.75,
            "store_format": "neuronsifter_chunk_store",
            "trajectory_array_name": "trajectories",
            "uncertainty_array_name": "teacher_uncertainty",
            "default_uncertainty_value": 0.5,
        },
    ]
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_load_chunked_rows(store_path: Path, array_name: str, row_indices: np.ndarray) -> np.ndarray:
        calls.append((Path(store_path).name, tuple(int(index) for index in row_indices.tolist())))
        return np.stack(
            [
                np.full((2, 2, 3), float(row_index), dtype=np.float32)
                for row_index in row_indices.tolist()
            ],
            axis=0,
        )

    monkeypatch.setattr(pretrain_world_model, "load_chunked_rows", fake_load_chunked_rows)
    batch = dataset.__getitems__([0, 1, 2])
    assert calls == [("store_a", (3, 1)), ("store_b", (2,))]
    assert [float(item["trajectory"][0, 0, 0]) for item in batch] == [3.0, 1.0, 2.0]
    assert [float(item["teacher_loss_weight"]) for item in batch] == [1.0, 0.5, 0.75]
    assert [float(item["uncertainty"][0]) for item in batch] == [0.25, 0.25, 0.5]


def test_missing_training_cache_fails_closed(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    with pytest.raises(FileNotFoundError, match="cache_missing"):
        load_scope_training_cache(
            config,
            REPO_ROOT / "Libs/configs/config_stack.yaml",
            tmp_path,
            "fold",
            "heldout_e_geod_12685",
            123,
        )


def test_training_scope_refuses_inline_cache_build_when_formal_cache_required(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["artifact_root"] = str(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] = "pytest_formal_cache_gate"
    with pytest.raises(FileNotFoundError, match="cache_missing"):
        train_one_scope(
            config,
            REPO_ROOT / "Libs/configs/config_stack.yaml",
            ["pytest"],
            "fold",
            "fold_alpha",
            {"transcriptomics": {}},
            [],
            [],
            [],
            [],
            [],
            {},
            "ad_control",
            torch.device("cpu"),
        )


def test_formal_cache_requirement_overrides_inline_build_flag(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["artifact_root"] = str(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] = "pytest_formal_cache_gate"
    config["screening_pipeline"]["screening_benchmark"]["training"]["cache_policy"]["allow_training_job_cache_build"] = True
    assert training_job_cache_build_allowed(config) is False
    with pytest.raises(FileNotFoundError, match="cache_missing"):
        train_one_scope(
            config,
            REPO_ROOT / "Libs/configs/config_stack.yaml",
            ["pytest"],
            "fold",
            "fold_alpha",
            {"transcriptomics": {}},
            [],
            [],
            [],
            [],
            [],
            {},
            "ad_control",
            torch.device("cpu"),
        )


def test_stale_training_cache_refuses_inline_rebuild_when_cache_build_allowed(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["artifact_root"] = str(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] = "pytest_stale_cache_gate"
    cache_policy = config["screening_pipeline"]["screening_benchmark"]["training"]["cache_policy"]
    cache_policy["require_cache_for_formal"] = False
    cache_policy["allow_training_job_cache_build"] = True
    artifact_root = artifact_dir(config, "fold", "fold_alpha")
    tensor_path = training_cache_tensor_path(artifact_root, config)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": cache_schema_version(config)}, tensor_path)
    training_cache_manifest_path(artifact_root, config).write_text(
        json.dumps(
            {
                "schema_version": cache_schema_version(config),
                "config_hash": json_sha256({"stale": True}),
                "freeze_signature_sha256": "",
                "scope": "fold",
                "fold": "fold_alpha",
                "seed": 123,
                "source_artifact_hashes": {},
                "sample_counts": {},
                "selected_target_program_count": 0,
                "selected_regimen_count": 0,
                "created_command": "pytest",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cache_stale"):
        train_one_scope(
            config,
            REPO_ROOT / "Libs/configs/config_stack.yaml",
            ["pytest"],
            "fold",
            "fold_alpha",
            {"transcriptomics": {}},
            [],
            [],
            [],
            [],
            [],
            {},
            "ad_control",
            torch.device("cpu"),
        )


def test_training_cache_manifest_requires_declared_fields(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    config_path = REPO_ROOT / "Libs/configs/config_stack.yaml"
    tensor_path = training_cache_tensor_path(tmp_path, config)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": cache_schema_version(config)}, tensor_path)
    manifest = {
        "schema_version": cache_schema_version(config),
        "config_hash": json_sha256(config),
        "freeze_signature_sha256": freeze_signature_sha256(config),
        "scope": "fold",
        "fold": "heldout_e_geod_12685",
        "seed": 123,
        "source_artifact_hashes": source_artifact_hashes(config, config_path),
        "sample_counts": {},
        "selected_target_program_count": 0,
        "selected_regimen_count": 0,
    }
    training_cache_manifest_path(tmp_path, config).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing_manifest_field:created_command"):
        load_scope_training_cache(
            config,
            config_path,
            tmp_path,
            "fold",
            "heldout_e_geod_12685",
            123,
        )


def test_stale_training_cache_manifest_fails_closed(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    tensor_path = training_cache_tensor_path(tmp_path, config)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": cache_schema_version(config)}, tensor_path)
    manifest = {
        "schema_version": cache_schema_version(config),
        "config_hash": json_sha256({"stale": True}),
        "freeze_signature_sha256": "",
        "scope": "fold",
        "fold": "heldout_e_geod_12685",
        "seed": 123,
        "source_artifact_hashes": {},
        "sample_counts": {},
        "selected_target_program_count": 0,
        "selected_regimen_count": 0,
        "created_command": "pytest",
    }
    training_cache_manifest_path(tmp_path, config).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cache_stale"):
        load_scope_training_cache(
            config,
            REPO_ROOT / "Libs/configs/config_stack.yaml",
            tmp_path,
            "fold",
            "heldout_e_geod_12685",
            123,
        )


def test_screening_training_uses_stochastic_regimen_window_for_counterfactual_pair_coverage() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    policy = config["screening_pipeline"]["screening_benchmark"]["training"]["regimen_minibatch"]
    labels = np.zeros((2, 128), dtype=np.float32)
    labels[:, 90] = 1.0
    labels[:, 91] = 0.9
    counterfactual_pairs = [(index, index + 1) for index in range(0, 72, 2)]
    indices = training_regimen_indices(
        config,
        128,
        labels,
        counterfactual_pairs,
        np.random.default_rng(3407),
    )
    assert indices.shape[0] == int(policy["candidate_count"])
    selected_counterfactual_indices = set(range(72)) & set(indices.tolist())
    assert len(selected_counterfactual_indices) >= int(policy["counterfactual_pair_window_size"]) * 2
    assert len(selected_counterfactual_indices) < 72
    assert {90, 91}.issubset(set(indices.tolist()))


def test_checkpointed_formal_training_windows_are_not_classified_as_smoke() -> None:
    assert (
        classify_scale(
            "python Libs/exp/training/train_screening_system.py --fold-name heldout_e_geod_12685 --max-epochs-override 40 --training-window-ordinal 3 --training-window-count 8",
            training_history_length=12,
            sample_count=12000,
            parameter_count=12000000,
            duration_seconds=3600.0,
        )
        == "full_scale_formal"
    )


def test_tri_targets_follow_current_regimen_window() -> None:
    full_labels = np.zeros((1, 128), dtype=np.float32)
    selected_indices = np.arange(80, dtype=np.int64)
    current_labels = full_labels[:, selected_indices]
    pkpd_target = torch.zeros(80, dtype=torch.float32)
    tri_targets = tri_target_tensors(
        np.array([0.75], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.ones((1, 4), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        current_labels,
        pkpd_target,
        torch.device("cpu"),
    )
    predicted_components = {
        component_name: torch.zeros((1, 80), dtype=torch.float32)
        for component_name in TRI_COMPONENT_NAMES
    }
    assert_tri_component_shape_contract(predicted_components, tri_targets)
    mismatched_targets = dict(tri_targets)
    mismatched_targets["calibration"] = torch.zeros((1, 128), dtype=torch.float32)
    with pytest.raises(ValueError, match="TRI component 'calibration' shape mismatch"):
        assert_tri_component_shape_contract(predicted_components, mismatched_targets)


def test_neuronsifter_scale_registry_applies_semantic_checkpoint_scales() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    scale_registry = config["neuronsifter_scale_registry"]
    required_scales = {
        "neuronsifter_debug",
        "neuronsifter_student",
        "neuronsifter_large",
        "neuronsifter_xl",
        "neuronsifter_foundation",
    }
    assert set(scale_registry["checkpoint_scales"]) == required_scales
    assert scale_registry["default_training_checkpoint_scale"] == "neuronsifter_large"
    assert scale_registry["foundation_candidate_claim_policy"]["current_status"] == "gated_future_capability"
    assert "trained foundation model" in scale_registry["foundation_candidate_claim_policy"]["forbidden_pre_gate_wording"]
    for scale_name, scale_payload in scale_registry["checkpoint_scales"].items():
        estimate = scale_parameter_estimate(config, scale_name)
        budget = scale_payload["parameter_budget"]
        assert int(budget["active_parameter_lower_bound"]) <= estimate <= int(budget["active_parameter_upper_bound"])
        profile_name = scale_payload["default_inference_profile"]
        assert scale_registry["inference_profiles"][profile_name]["checkpoint_scale"] == scale_name
    student_config = apply_neuronsifter_scale(config, "neuronsifter_student", "student_prescreen")
    student_backbone = student_config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert student_backbone["operator_hidden_dim"] == 384
    assert student_backbone["operator_depth"] == 6
    assert student_backbone["diffusion_sampler"]["sample_count"] == 4
    assert student_config["screening_pipeline"]["screening_benchmark"]["training"]["batch_size"] == 128


def test_simulation_qc_and_decision_gate_contracts_are_yaml_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    audit_config = config["ai_ready_preprocessing"]["simulation_audit"]
    assert audit_config["relative_paths"]["qc_catalog"] == "simulation_qc_catalog.tsv"
    assert audit_config["readiness_gates"]["g2_100k"]["full_min_count"] == 100000
    assert audit_config["split_audit"]["forbid_family_overlap"] is True
    assert audit_config["grading"]["qc_use_by_grade"]["A"] == "main_pretraining"
    training_families = {
        row["family_key"]: row
        for row in config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]["families"]
    }
    assert training_families["full_microenvironment_teacher"]["catalog_path"].endswith("simulation_qc_catalog.tsv")
    assert training_families["full_microenvironment_teacher"]["allowed_qc_uses"] == ["main_pretraining"]
    decision_gate = config["screening_pipeline"]["screening_benchmark"]["decision_gate"]
    assert set(decision_gate["decision_labels"].values()) == {
        "accept",
        "reject",
        "active_learning",
        "external_validation",
        "insufficient_evidence",
    }
    bridge = config["screening_pipeline"]["screening_benchmark"]["clinical_direction_bridge"]
    assert bridge["bridge_role"] == "retrospective_endpoint_direction_module"
    assert bridge["endpoint_policy"] == "literature_aggregate_direction_only"
    assert bridge["source_policy"]["disallow_sample_level_public_cohort_training"] is True
    comparison = config["screening_pipeline"]["screening_benchmark"]["case_study_comparison"]
    assert comparison["source_holdout_policy"] == "source_holdout"
    assert comparison["clinical_claim_policy"] == "retrospective_clinical_direction_consistency"
    assert comparison["anchor_score_matrix"]["VERUBECESTAT"]["external_evidence_only"] > comparison["anchor_score_matrix"]["VERUBECESTAT"]["neuronsifter_tri_gated"]


def test_publication_readiness_contracts_are_yaml_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    readiness_config = config["paper"]["publication_readiness"]
    contract = config["supplemental_milestone_contracts"]["publication_readiness"]
    assert readiness_config["research_positioning"]["method_claim"] == contract["required_method_claim"]
    assert readiness_config["research_positioning"]["disease_scope"] == "ad_control"
    assert readiness_config["research_positioning"]["data_boundary"] == "public_only_known_targets_and_compounds"
    assert readiness_config["submission_policy"]["readiness_failure_policy"] == "delay_submission_until_scientific_readiness"
    claim_ids = {row["claim_id"] for row in readiness_config["core_claims"]}
    assert set(contract["required_core_claims"]).issubset(claim_ids)
    split_names = set(readiness_config["strict_splits"]["split_names"])
    assert set(contract["required_strict_splits"]).issubset(split_names)
    g5_config = readiness_config["g5_foundation_pretraining"]
    assert g5_config["store_trajectory_min_count"] == 1000000
    assert g5_config["headline_profile"] == "neuronsifter_large"
    assert sum(float(value) for value in g5_config["expansion_quota"].values()) == g5_config["quota_sum"]
    assert g5_config["candidate_boundary"]["allow_molecule_generation"] is False
    assert g5_config["candidate_boundary"]["allow_analog_expansion"] is False
    assert g5_config["candidate_boundary"]["allow_private_patient_records"] is False
    assert {"coverage", "crps", "nll"}.issubset(set(readiness_config["metric_catalog"]["uncertainty"]))
    baseline_ids = {row["baseline_id"] for row in readiness_config["baseline_grid"]["baselines"]}
    assert set(contract["required_baselines"]).issubset(baseline_ids)
    assert readiness_config["tri_calibration"]["threshold_source_split"] == "validation"
    assert set(contract["required_report_artifacts"]).issubset(set(readiness_config["report_outputs"]["required_artifacts"]))


def test_ad_literature_anchor_directions_and_clinical_policy_are_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    anchor_rows = {
        row["intervention_label"]: row
        for row in config["dynamics_ready_registry"]["case_validation_entries"]
    }
    expected_directions = {
        "LECANEMAB": "favorable",
        "DONANEMAB": "favorable",
        "VERUBECESTAT": "weak_or_negative",
        "SOLANEZUMAB": "weak_or_negative",
    }
    for intervention_label, expected_direction in expected_directions.items():
        assert anchor_rows[intervention_label]["expected_effect_direction"] == expected_direction
        assert anchor_rows[intervention_label]["clinical_endpoint_alignment_policy"] == "qualitative_direction_only"


def test_neuron_rxd_teacher_role_is_declared_once() -> None:
    text = (REPO_ROOT / "Libs/configs/model/dynamics_solver.yaml").read_text(encoding="utf-8")
    block = text.split("  neuron_rxd_teacher:", 1)[1].split("  case_validation_entries:", 1)[0]
    assert block.count("teacher_role:") == 1


def test_local_membrane_binding_contract_uses_canonical_schema():
    from dataclasses import fields
    from Libs.utils.data.local_membrane import LocalMembraneSnapshot, MembraneSite, MembraneAcquisition, SNAPSHOT_SCHEMA
    config = load_config("Libs/configs/config_stack.yaml")
    membrane = config["schemas"]["native_joint_observation_source"]["local_membrane"]
    assert membrane["snapshot_schema_version"] == SNAPSHOT_SCHEMA
    for declared, kind in (("snapshot_fields",LocalMembraneSnapshot), ("site_fields",MembraneSite),
                           ("acquisition_fields",MembraneAcquisition)):
        assert set(membrane[declared]) == {item.name for item in fields(kind)}
    assert membrane["axes"] == ["sample","particle","site"]
    assert membrane["clock_dtype"] == "float64"
    assert membrane["free_readout_mean"] == "forbidden"
    assert membrane["missing_error_binding"] == "unbound_channel_cannot_condition"
    assert membrane["calibration_status"] == "unverified"
    assert membrane["electrical_evolution_status"] == "unbound"
