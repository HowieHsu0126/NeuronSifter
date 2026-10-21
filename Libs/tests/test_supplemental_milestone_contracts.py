from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.pretrain_microenvironment_world_model import validate_default_parameter_budget
from Libs.utils.data.raw_data_registry import load_config


def test_supplemental_milestone_contracts_are_configured() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    contract = config["supplemental_milestone_contracts"]
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    transition_backbone = benchmark["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert transition_backbone["kind"] == contract["neural_operator"]["required_kind"]
    assert transition_backbone["external_operator_references"]["reference_policy"] == contract["neural_operator"]["required_reference_policy"]
    assert transition_backbone["operator_registry"]["main_rollout_sampler_kind"] == contract["neural_operator"]["required_main_rollout_sampler_kind"]
    assert transition_backbone["quantized_mddm_rollout"]["registry_binding"] == contract["neural_operator"]["required_mddm_registry_binding"]
    assert config["neuronsifter_scale_registry"][contract["neural_operator"]["required_mddm_registry_binding"]]["enabled"] is True
    assert transition_backbone["diffusion_sampler"]["sample_count"] == contract["neural_operator"]["required_sampler_sample_count"]
    assert transition_backbone["diffusion_sampler"]["denoising_step_count"] == contract["neural_operator"]["required_sampler_denoising_step_count"]
    assert transition_backbone["operator_splitting"]["enabled"] == contract["neural_operator"]["require_operator_splitting"]
    assert set(contract["neural_operator"]["required_operator_splitting_experts"]).issubset(set(transition_backbone["operator_splitting"]["composition_order"]))
    assert transition_backbone["transolver_ablation_template"]["kind"] == contract["transolver_ablation"]["required_kind"]
    assert transition_backbone["transolver_reference"]["refactor_policy"] == contract["transolver_ablation"]["required_refactor_policy"]
    assert transition_backbone["parameter_budget"]["training_scale"] == contract["model_scale"]["required_training_scale"]
    assert config["osp_pksim_environment"]["conda_solver"] == contract["runtime"]["required_conda_solver"]
    assert benchmark["active_disease_program"] == contract["disease_scope"]["required_primary_disease_program"]


def test_dual_moe_and_full_microenvironment_species_are_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    contract = config["supplemental_milestone_contracts"]
    architecture = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]
    transition_backbone = architecture["rollout_operator"]["transition_backbone"]
    mechanism_experts = set(transition_backbone["mechanism_moe"]["expert_names"])
    multimodal_experts = set(architecture["multimodal_moe"]["expert_names"])
    species_ids = {row["species_id"] for row in config["dynamics_ready_registry"]["species"]}
    assert set(contract["moe"]["required_mechanism_experts"]).issubset(mechanism_experts)
    assert set(contract["moe"]["required_multimodal_experts"]).issubset(multimodal_experts)
    assert set(contract["full_microenvironment"]["required_species"]).issubset(species_ids)


def test_default_active_parameter_budget_matches_locked_scale() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    with pytest.raises(ValueError, match="explicit observation_feature_width"):
        validate_default_parameter_budget(config)


def test_trustworthy_screening_contracts_are_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    contract = config["supplemental_milestone_contracts"]
    assert contract["simulation_readiness_audit"]["required_readiness_gate"] == "g2_100k"
    assert contract["neuronsifter_profiles"]["required_active_screening_profile"] == "neuronsifter_large"
    assert set(config["neuronsifter_scale_registry"]["checkpoint_scales"]) >= {
        "neuronsifter_debug",
        "neuronsifter_student",
        "neuronsifter_large",
        "neuronsifter_xl",
        "neuronsifter_foundation",
    }
    assert config["neuronsifter_scale_registry"]["fsdp_resource_profiles"]["xl_four_gpu_smoke"]["distributed_strategy"] == "native_pytorch_fsdp"
    assert set(contract["decision_gate"]["required_labels"]) == {
        "accept",
        "reject",
        "active_learning",
        "external_validation",
        "insufficient_evidence",
    }
    assert contract["clinical_direction_boundary"]["required_bridge_role"] == "retrospective_endpoint_direction_module"


def test_publication_readiness_contract_is_declared() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    contract = config["supplemental_milestone_contracts"]["publication_readiness"]
    assert contract["required_disease_scope"] == "ad_control"
    assert contract["required_endpoint_policy"] == "retrospective_clinical_direction_consistency"
    assert contract["required_g5_store_trajectory_min_count"] == 1000000
    assert contract["required_g5_headline_profile"] == "neuronsifter_large"
    assert set(contract["required_uq_metrics"]) == {"coverage", "crps", "nll"}
    assert "full_neuronsifter_large" in contract["required_baselines"]
    assert "claim_metric_catalog.tsv" in contract["required_report_artifacts"]


def test_default_supplemental_report_blocks_missing_profile_before_reads_or_count(monkeypatch):
    from Libs.scripts.validators import validate_supplemental_milestone_contracts as validator
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing profile must stop before source scans or parameter count")
    monkeypatch.setattr(validator, "scan_forbidden_imports", forbidden)
    monkeypatch.setattr(validator, "validate_default_parameter_budget", forbidden)
    report = validator.build_report(config, False)
    assert report["contract_status"] == "blocked_missing_observation_profile"
    assert report["active_parameter_estimate"] is None
    assert report["observation_profile"]["observation_feature_width"] is None
    assert report["scientific_twin"] == "NOT_RUN"


def test_manuscript_observation_profile_and_parameter_count_use_declared_interface(monkeypatch):
    import torch
    from Libs.utils.model.screening_model_config import observation_feature_profile
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrainer
    config = load_config(REPO_ROOT / "Libs/configs/exp/manuscript_native_protocol.yaml")
    profile = observation_feature_profile(config)
    assert profile["status"] == "bound" and profile["observation_feature_width"] == 6
    assert profile["profile"] == "manuscript_native_protocol" and profile["data_or_outcome_inference"] is False
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    context_width = pretrainer.native_posterior_context_width(config, int(backbone["operator_hidden_dim"]))
    rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng():
        modules, width = pretrainer.joint_inference_modules(config, context_width, len(config["dynamics_ready_registry"]["species"]))
    expected = sum(p.numel() for module in modules for p in module.parameters())
    original = pretrainer.joint_inference_modules
    def count_on_meta(*args, **kwargs):
        modules, width = original(*args, **kwargs)
        assert all(parameter.is_meta for module in modules for parameter in module.parameters())
        return modules, width
    monkeypatch.setattr(pretrainer, "joint_inference_modules", count_on_meta)
    assert pretrainer.joint_inference_parameter_count(config) == expected
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.nn.Linear(1, 1).weight.device.type == "cpu"
    total = validate_default_parameter_budget(config)
    assert total == 241896002


@pytest.mark.parametrize("invalid", [True, 0, -1, 6.0, "6"])
def test_observation_profile_rejects_invalid_declared_width(invalid):
    from Libs.utils.model.screening_model_config import observation_feature_profile
    config = load_config(REPO_ROOT / "Libs/configs/exp/manuscript_native_protocol.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["observation_feature_width"] = invalid
    with pytest.raises(ValueError, match="positive integer"):
        observation_feature_profile(config)


@pytest.mark.parametrize("invalid", [[[0, 1], [2, 3], [4, 4]], [[0, 1], [2, 3]], [[0, 1], [2, 3], [4, True]]])
def test_manuscript_profile_rejects_incomplete_or_repeated_readout_blocks(invalid):
    from Libs.utils.model.screening_model_config import observation_feature_profile
    config = load_config(REPO_ROOT / "Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_state_comparison"]["readout_blocks"] = invalid
    with pytest.raises(ValueError, match="partition"):
        observation_feature_profile(config)


def test_supplemental_cli_reports_bound_profile_and_other_contract_prerequisite(tmp_path, monkeypatch):
    import argparse
    import json
    from Libs.scripts.validators import validate_supplemental_milestone_contracts as validator
    output = tmp_path / "report.json"
    monkeypatch.setattr(validator, "parse_args", lambda: argparse.Namespace(
        config=str(REPO_ROOT / "Libs/configs/exp/manuscript_native_protocol.yaml"),
        output=str(output), strict_materialized_scale=False))
    def unmet(config, strict):
        raise ValueError("Required repository-generated teacher catalog missing: retained scope")
    monkeypatch.setattr(validator, "build_report", unmet)
    with pytest.raises(SystemExit) as result:
        validator.main()
    assert result.value.code == 2
    report = json.loads(output.read_text())
    assert report["observation_profile"]["status"] == "bound"
    assert report["observation_profile"]["observation_feature_width"] == 6
    assert report["contract_status"] == "blocked_contract_prerequisite"
    assert "teacher catalog" in report["reason"]



def test_supplemental_cli_serializes_invalid_profile_without_parameter_count(tmp_path, monkeypatch):
    import argparse
    import json
    from Libs.scripts.validators import validate_supplemental_milestone_contracts as validator
    config = load_config(REPO_ROOT / "Libs/configs/exp/manuscript_native_protocol.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["observation_feature_width"] = True
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(validator, "parse_args", lambda: argparse.Namespace(
        config="explicit_invalid_fixture", output=str(output), strict_materialized_scale=False))
    monkeypatch.setattr(validator, "load_config", lambda path: config)
    def forbidden(*args, **kwargs): raise AssertionError("Invalid profile cannot reach report validation")
    monkeypatch.setattr(validator, "build_report", forbidden)
    with pytest.raises(SystemExit) as result: validator.main()
    assert result.value.code == 2
    report = json.loads(output.read_text())
    assert report["contract_status"] == "blocked_invalid_observation_profile"
    assert "positive integer" in report["reason"] and report["scientific_twin"] == "NOT_RUN"
