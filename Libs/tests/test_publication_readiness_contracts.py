from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.reports.build_paper_results import (
    active_learning_empirical_gap_audit_payload,
    active_learning_loop_summary_payload,
    foundation_readiness_gate_summary,
    publication_ablation_result_rows,
    publication_baseline_result_rows,
    publication_claim_metric_rows,
    publication_g5_effective_sample_audit,
    publication_claim_discipline_audit,
    publication_selective_risk_rows,
)
from Libs.scripts.validators.validate_design_contract_coverage import build_report as build_design_contract_report
from Libs.scripts.validators.validate_publication_readiness_contracts import validate_adni_publication_boundary
from Libs.utils.data.raw_data_registry import load_config


def test_publication_readiness_validator_accepts_yaml_contracts() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    readiness = config["paper"]["publication_readiness"]
    assert readiness["research_positioning"]["disease_scope"] == "ad_control"
    assert len(readiness["core_claims"]) == 4
    assert readiness["g5_foundation_pretraining"]["store_trajectory_min_count"] == 1000000
    assert "adni_controlled_access" not in config
    assert validate_adni_publication_boundary(config) == []


def test_publication_report_contract_rows_cover_claims_baselines_and_tri() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    claim_rows = publication_claim_metric_rows(config)
    baseline_rows = publication_baseline_result_rows(config)
    ablation_rows = publication_ablation_result_rows(config)
    selective_risk_rows = publication_selective_risk_rows(config)
    g5_audit = publication_g5_effective_sample_audit(config)
    assert {row["claim_id"] for row in claim_rows} >= {
        "occupancy_conditioned_action_interface",
        "stochastic_neuronsifter_world_model",
        "mechanism_expert_routing",
        "tri_gated_reliable_screening",
    }
    assert {row["baseline_id"] for row in baseline_rows} >= {
        "external_evidence_only",
        "drug_id_conditioned_neuronsifter",
        "transolver_microenvironment_operator",
        "diffusion_pde_adapter",
        "wdno_adapter",
        "coda_no_adapter",
        "mamba_neural_operator_adapter",
        "full_neuronsifter_large",
    }
    assert {row["replacement_axis"] for row in ablation_rows} == {
        "drug_id",
        "target_id",
        "scalar_exposure",
        "occupancy_program",
    }
    assert {row["metric_name"] for row in selective_risk_rows} >= {
        "selective_risk",
        "risk_coverage_auc",
        "aurc",
        "ood_auroc",
        "pkpd_consistency",
    }
    assert g5_audit["store_trajectory_min_count"] == 1000000
    assert g5_audit["gate_status"] == "not_passed"
    assert g5_audit["materialized_trajectory_gate_status"] == "not_passed"
    assert g5_audit["current_full_microenvironment_trajectory_count"] < 1000000
    assert g5_audit["current_mesoscopic_trajectory_count"] < 1000000
    effective_counts = g5_audit["effective_sample_counts"]
    effective_coverage = g5_audit["effective_coverage"]
    assert effective_counts["unique_trajectory_families"] > 0
    assert effective_counts["unique_regimens"] > 0
    assert effective_counts["unique_target_programs"] > 0
    assert effective_counts["high_confidence_trajectories"] == g5_audit["current_full_microenvironment_trajectory_count"]
    assert effective_counts["medium_confidence_trajectories"] == g5_audit["current_mesoscopic_trajectory_count"]
    assert effective_counts["train_eligible_grade_ab_records"] == g5_audit["current_train_eligible_record_count"]
    assert effective_counts["reject_records"] == g5_audit["current_reject_record_count"]
    assert effective_coverage["source_holdout_leakage_status"] == "pass"
    assert 0.0 < effective_coverage["counterfactual_pair_coverage"] <= 1.0
    report_outputs = config["paper"]["publication_readiness"]["report_outputs"]
    assert "measurement_semantics_audit.json" in report_outputs["required_artifacts"]
    assert "dynamics_calibration_audit.json" in report_outputs["required_artifacts"]
    assert "gpu_stochastic_smoke_summary.json" in report_outputs["required_artifacts"]
    assert "adni_controlled_bundle_pointer.json" not in report_outputs["required_artifacts"]
    assert "adni_controlled_bundle_pointer.json" in report_outputs["controlled_access_optional_artifacts"]


def test_publication_chain_keeps_adni_outside_public_formal_gate_and_runs_design_validator() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    final_commands = config["paper"]["publication_claim_suite"]["final_report_commands"]
    command_text = "\n".join(final_commands)
    assert config["paper"]["publication_claim_suite"]["controlled_access_validation_enabled"] is False
    assert "run_adni_controlled_state_inference_benchmark.py" not in command_text
    assert "validate_design_contract_coverage.py" in command_text
    supplemental_contract = config["supplemental_milestone_contracts"]["publication_readiness"]
    assert "adni_state_inference_summary.json" not in supplemental_contract["required_report_artifacts"]
    assert "adni_state_inference_summary.json" in supplemental_contract["controlled_access_optional_report_artifacts"]


def test_design_contract_coverage_maps_every_current_design_document() -> None:
    report = build_design_contract_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert report["contract_status"] == "design_contract_coverage_satisfied"
    assert report["declared_document_count"] == report["discovered_document_count"] == 39
    documents = {row["path"]: row for row in report["documents"]}
    assert documents["Docs/design/iclr/00_MASTER_ORCHESTRATION.md"]["document_status"] == "canonical"
    assert documents["Docs/design/iclr/CHANGELOG.md"]["document_status"] == "historical"
    assert documents["Docs/design/iclr/literature-search-20260713-neuronsifter-neighbor-audit/papers.md"]["document_status"] == "reference-only"
    assert all(row["coverage_status"] == "resolved" for row in documents.values())


def test_active_learning_publication_gap_reports_do_not_upgrade_pending_results() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    loop_summary = active_learning_loop_summary_payload(config)
    gap_audit = active_learning_empirical_gap_audit_payload(config)
    assert loop_summary["contract_name"] == "decision_centric_simulator_portfolio_active_learning"
    assert loop_summary["required_round_count"] == config["screening_pipeline"]["screening_benchmark"]["active_learning"]["full_round_count"]
    assert gap_audit["claim_policy"] == "no_active_learning_empirical_gain_claim_until_all_rounds_complete"
    if gap_audit["gap_status"] != "pass":
        assert loop_summary["status"] == "active_learning_empirical_results_pending"
        assert gap_audit["missing_empirical_rounds"]


def test_claim_discipline_fails_closed_when_hardening_evidence_is_unavailable(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    missing_evidence_config = deepcopy(config)
    missing_evidence_config["paper"]["paths"]["report_dir"] = str(tmp_path)
    payload = publication_claim_discipline_audit(missing_evidence_config)
    assert payload["framework_hardening_evidence_status"] == "evidence_unavailable"
    assert payload["wording_status"] == "evidence_unavailable_claims_withheld"
    assert payload["input_output_contract_status"] == "evidence_unavailable"
    assert len(payload["missing_required_evidence_artifacts"]) == 4
    assert all(not Path(identifier).is_absolute() for identifier in payload["missing_required_evidence_artifacts"])
    assert all(str(REPO_ROOT) not in identifier for identifier in payload["missing_required_evidence_artifacts"])
    assert all(str(tmp_path) not in identifier for identifier in payload["missing_required_evidence_artifacts"])
    assert payload["dynamics_accuracy_claim_scope"] == "mechanistic_teacher_pretraining_only"
    assert payload["dynamics_external_calibration_claim_status"] == "not_claimed"
    assert payload["crossscale_alignment_claim_scope"] == "paired_sample_sources_plus_distributional_weak_sources"
    assert payload["full_crossscale_alignment_claim_status"] == "not_claimed"
    assert "measurement_semantics_contract_completion" in payload["prohibited_evidence_upgrades"]
    assert "bundle_scaler_artifacts" in payload["prohibited_evidence_upgrades"]
    assert "external_calibrated_dynamics_accuracy" in payload["prohibited_evidence_upgrades"]
    assert "full_crossscale_alignment" in payload["prohibited_evidence_upgrades"]


def test_foundation_readiness_gate_public_payload_omits_forbidden_claim_text() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    payload = foundation_readiness_gate_summary(config)
    payload_text = str(payload).lower()
    for claim_fragment in config["neuronsifter_scale_registry"]["foundation_candidate_claim_policy"]["forbidden_pre_gate_wording"]:
        assert str(claim_fragment).lower() not in payload_text
    assert payload["forbidden_pre_gate_claim_count"] == 3
    assert payload["forbidden_pre_gate_wording_publication_policy"] == "withheld_from_public_reports"
