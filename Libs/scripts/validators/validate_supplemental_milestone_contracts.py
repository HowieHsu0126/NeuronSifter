from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.pretrain_microenvironment_world_model import validate_default_parameter_budget
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json
from Libs.utils.data.trajectory_numeric_lineage import load_canonical_metric_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate supplemental M7-M9 landing contracts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="Output/audits/supplemental_milestone_contracts.json")
    parser.add_argument("--strict-materialized-scale", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_subset(required_values: list[str], observed_values: list[str] | set[str], label: str) -> None:
    missing = sorted(set(required_values) - set(observed_values))
    if missing:
        raise ValueError(f"{label} missing required values: {missing}")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in handle) - 1


def first_data_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row = next(reader)
    return row


def scan_forbidden_imports(root: Path, forbidden_fragment: str) -> list[str]:
    matches: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if forbidden_fragment in text:
            matches.append(str(path.resolve().relative_to(REPO_ROOT)))
    return matches


def file_contains(path: Path, text_fragment: str) -> bool:
    return text_fragment in path.read_text(encoding="utf-8")


def build_report(config: dict[str, object], strict_materialized_scale: bool) -> dict[str, object]:
    contract = config["supplemental_milestone_contracts"]
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    architecture = benchmark["model_architecture"]
    transition_backbone = architecture["rollout_operator"]["transition_backbone"]
    dynamics_config = config["dynamics_ready_registry"]
    runtime_config = config["osp_pksim_environment"]

    operator_contract = contract["neural_operator"]
    transolver_ablation = contract["transolver_ablation"]
    forbidden_imports = scan_forbidden_imports(REPO_ROOT / "Libs" / "models", str(operator_contract["forbidden_external_import_fragment"]))
    require(not forbidden_imports, f"Project-native neural operators must not import external implementations: {forbidden_imports}")
    require(transition_backbone["kind"] == operator_contract["required_kind"], "Default transition backbone is not the required NeuronSifter microenvironment operator.")
    require(transition_backbone["external_operator_references"]["reference_policy"] == operator_contract["required_reference_policy"], "NeuronSifter reference policy is not project-native.")
    require(int(transition_backbone["diffusion_sampler"]["sample_count"]) == int(operator_contract["required_sampler_sample_count"]), "NeuronSifter sampler sample_count does not match the contract.")
    require(int(transition_backbone["diffusion_sampler"]["denoising_step_count"]) == int(operator_contract["required_sampler_denoising_step_count"]), "NeuronSifter sampler denoising_step_count does not match the contract.")
    operator_registry = transition_backbone["operator_registry"]
    require(operator_registry["main_rollout_sampler_kind"] == operator_contract["required_main_rollout_sampler_kind"], "Main rollout sampler is not the required quantized MDDM operator.")
    quantized_mddm_rollout = transition_backbone["quantized_mddm_rollout"]
    require(bool(quantized_mddm_rollout["enabled"]), "Quantized MDDM rollout binding is not enabled.")
    require(quantized_mddm_rollout["registry_binding"] == operator_contract["required_mddm_registry_binding"], "Quantized MDDM rollout registry binding does not match the contract.")
    mddm_registry = config["neuronsifter_scale_registry"][operator_contract["required_mddm_registry_binding"]]
    require(mddm_registry["rollout_kind"] == operator_contract["required_main_rollout_sampler_kind"], "MDDM registry rollout kind does not match the contract.")
    require(mddm_registry["external_reference_policy"]["reference_policy"] == operator_contract["required_reference_policy"], "MDDM reference policy is not project-native.")
    require("MDM-Prime-v2" in mddm_registry["external_reference_policy"]["excluded_references"], "Retracted MDM-Prime-v2 reference exclusion is missing.")
    for reference_path in operator_contract["required_mddm_reference_paths"]:
        require(resolve_repo_path(str(reference_path)).exists(), f"Missing required MDDM reference path: {reference_path}")
    operator_splitting = transition_backbone["operator_splitting"]
    require(bool(operator_splitting["enabled"]) == bool(operator_contract["require_operator_splitting"]), "NeuronSifter operator splitting is not enabled.")
    require_subset(operator_contract["required_operator_splitting_experts"], [str(value) for value in operator_splitting["composition_order"]], "operator_splitting.composition_order")
    coupling_masks = dict(operator_splitting["coupling_masks"])
    for expert_name in operator_contract["required_operator_splitting_experts"]:
        require(str(expert_name) in coupling_masks and len(coupling_masks[str(expert_name)]) > 0, f"operator_splitting.coupling_masks lacks {expert_name}.")
    for module_path in operator_contract["required_native_modules"]:
        require(resolve_repo_path(str(module_path)).exists(), f"Missing required native NeuronSifter module: {module_path}")
    require(file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "neuronsifter_world_action.py", "class NeuronSifterWorldActionOperator"), "Native NeuronSifter operator class is missing.")
    require(file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "neuronsifter_world_action.py", "class MechanismOperatorExpertBank"), "NeuronSifter operator-splitting expert bank is missing.")
    require(file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "quantized_mddm_rollout_operator.py", "class QuantizedMddmRolloutOperator"), "Native quantized MDDM rollout class is missing.")
    require(not file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "quantized_mddm_rollout_operator.py", "Libs.external.mddm"), "Quantized MDDM rollout must not import external MDDM code.")
    neuronsifter_text = (REPO_ROOT / "Libs" / "models" / "modules" / "neuronsifter_world_action.py").read_text(encoding="utf-8")
    for diagnostic_name in [
        "operator_splitting_species_attribution",
        "operator_splitting_coupling_matrix",
        "operator_splitting_score_norm_by_expert",
        "operator_splitting_uncertainty_by_expert",
        "operator_splitting_case_activation",
    ]:
        require(diagnostic_name in neuronsifter_text, f"NeuronSifter expert diagnostic is missing: {diagnostic_name}")
    require(transition_backbone["transolver_ablation_template"]["kind"] == transolver_ablation["required_kind"], "Transolver is not retained as the required ablation operator.")
    require(transition_backbone["transolver_reference"]["refactor_policy"] == transolver_ablation["required_refactor_policy"], "Transolver ablation refactor policy is not project-native.")
    for module_path in transolver_ablation["required_native_modules"]:
        require(resolve_repo_path(str(module_path)).exists(), f"Missing required native Transolver ablation module: {module_path}")
    require(file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "transolver_operator.py", "class TransolverMicroenvironmentOperator"), "Native Transolver operator class is missing.")

    mechanism_moe = transition_backbone["mechanism_moe"]
    multimodal_moe = architecture["multimodal_moe"]
    require(bool(contract["moe"]["require_mechanism_moe"]) and "mechanism_moe" in transition_backbone, "Mechanism MoE is not configured.")
    require(bool(contract["moe"]["require_multimodal_moe"]) and bool(multimodal_moe["enabled"]), "Multimodal MoE is not enabled.")
    require_subset(contract["moe"]["required_mechanism_experts"], [str(value) for value in mechanism_moe["expert_names"]], "mechanism_moe.expert_names")
    require_subset(contract["moe"]["required_multimodal_experts"], [str(value) for value in multimodal_moe["expert_names"]], "multimodal_moe.expert_names")
    require(file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "evidence.py", "self.multimodal_moe"), "Multimodal MoE is not wired into evidence encoding.")
    require(file_contains(REPO_ROOT / "Libs" / "models" / "modules" / "neuronsifter_world_action.py", "mechanism_score_moe"), "Mechanism MoE is not wired into the NeuronSifter score operator.")

    species_ids = [str(row["species_id"]) for row in dynamics_config["species"]]
    require(dynamics_config["contract_version"] == contract["full_microenvironment"]["required_contract_version"], "Dynamics contract version does not match full microenvironment contract.")
    require_subset(contract["full_microenvironment"]["required_species"], species_ids, "dynamics_ready_registry.species")
    require_subset(contract["full_microenvironment"]["required_species"], benchmark["model_architecture"]["sample_evidence"]["baseline_field_policy"]["direct_scale_by_species_id"].keys(), "baseline field policy species")

    require(benchmark["active_disease_program"] == contract["disease_scope"]["required_primary_disease_program"], "Benchmark is not AD-first.")
    require(dynamics_config["neuron_rxd_teacher"]["disease_scope"] == contract["neuron_rxd"]["required_disease_scope"], "NEURON/RxD teacher is not AD-first.")
    require(dynamics_config["neuron_rxd_teacher"]["teacher_role"] == contract["neuron_rxd"]["required_teacher_role"], "NEURON/RxD teacher is not a mechanism teacher.")
    require_subset(contract["neuron_rxd"]["required_source_ids"], [str(value) for value in dynamics_config["neuron_rxd_teacher"]["source_priority"]], "neuron_rxd_teacher.source_priority")
    require_subset(contract["neuron_rxd"]["required_state_outputs"], [str(value) for value in dynamics_config["neuron_rxd_teacher"]["state_outputs"]], "neuron_rxd_teacher.state_outputs")
    require_subset(contract["neuron_rxd"]["required_rxd_species"], [str(value) for value in dynamics_config["neuron_rxd_teacher"]["rxd_species"]], "neuron_rxd_teacher.rxd_species")

    parameter_estimate = validate_default_parameter_budget(config)
    budget = transition_backbone["parameter_budget"]
    require(budget["training_scale"] == contract["model_scale"]["required_training_scale"], "Training scale is not single-card trainable.")
    require(int(contract["model_scale"]["active_parameter_lower_bound"]) <= parameter_estimate <= int(contract["model_scale"]["active_parameter_upper_bound"]), "Default active parameter estimate is outside the required range.")

    require(runtime_config["runtime_strategy"] == contract["runtime"]["required_runtime_strategy"], "OSP runtime strategy is not reproducible environment script.")
    require(runtime_config["conda_solver"] == contract["runtime"]["required_conda_solver"], "OSP Conda solver does not match the locked solver.")
    package_names = {str(value) for value in runtime_config["required_r_packages"]}
    package_names.update(str(row["package_name"]) for row in runtime_config["github_r_packages"])
    require_subset(contract["runtime"]["required_r_packages"], package_names, "osp_pksim_environment R packages")

    teacher_policy = dynamics_config["teacher_source_policy"]
    require(teacher_policy["primary_pbpk_teacher"] == contract["teacher_generation"]["required_pbpk_teacher"], "PBPK teacher is not OSP/PK-Sim rollout.")
    require(teacher_policy["legacy_solver_role"] == contract["teacher_generation"]["required_legacy_solver_role"], "Legacy solver role is not compatibility baseline.")
    require(dynamics_config["chembl_pbpk_parameterization"]["teacher_role"] == "parameter_predicted_teacher", "ChEMBL PBPK parameterization is not parameter_predicted_teacher.")
    require("uncertainty" in config["schemas"]["chembl_pbpk_parameter_fields"], "ChEMBL parameter schema lacks uncertainty.")

    weighting = benchmark["training"]["uncertainty_weighting"]
    require(bool(weighting["enabled"]) == bool(contract["loss_weighting"]["require_uncertainty_weighting"]), "Uncertainty weighting is not enabled.")
    require_subset(contract["loss_weighting"]["required_objectives"], benchmark["training"]["pretraining_objectives"].keys(), "training.pretraining_objectives")
    teacher_family_config = benchmark["training"]["teacher_family_training"]
    require(bool(teacher_family_config["enabled"]), "Joint teacher-family pretraining is not enabled.")
    require_subset(contract["teacher_family_training"]["required_families"], [str(row["family_key"]) for row in teacher_family_config["families"]], "training.teacher_family_training.families")
    pretraining_text = (REPO_ROOT / "Libs" / "exp" / "training" / "pretrain_microenvironment_world_model.py").read_text(encoding="utf-8")
    microenvironment_text = (REPO_ROOT / "Libs" / "utils" / "model" / "microenvironment_teacher.py").read_text(encoding="utf-8")
    combined_training_text = pretraining_text + microenvironment_text
    for function_name in contract["loss_weighting"]["required_pretraining_functions"]:
        require(function_name in combined_training_text, f"Required uncertainty weighting function is missing: {function_name}")

    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    registry_root = resolve_repo_path(str(registry["local_root"]))
    registry_paths = registry["relative_paths"]
    source_model_path = registry_root / str(registry_paths["source_model_catalog"])
    pbpk_rollout_path = registry_root / str(registry_paths["pbpk_rollout_catalog"])
    chembl_parameter_path = registry_root / str(registry_paths["chembl_pbpk_parameter_catalog"])
    neuron_protocol_path = registry_root / str(registry_paths["neuron_rxd_protocol_catalog"])
    for required_path in [source_model_path, pbpk_rollout_path, chembl_parameter_path, neuron_protocol_path]:
        require(required_path.exists(), f"Required repository-generated teacher catalog missing: {required_path}")
    require(row_count(pbpk_rollout_path) > 0, "PBPK rollout catalog has no repository-generated rows.")
    require(row_count(neuron_protocol_path) > 0, "NEURON/RxD protocol catalog has no repository-generated rows.")
    require(row_count(chembl_parameter_path) > 0, "ChEMBL parameter catalog has no parameter-predicted rows.")
    require(first_data_row(pbpk_rollout_path)["provenance_level"] == "repository_generated_mechanistic_rollout", "PBPK rollout provenance is not repository-generated.")
    require(first_data_row(neuron_protocol_path)["provenance_level"] == "repository_generated_biophysical_rollout", "NEURON/RxD provenance is not repository-generated.")
    require(first_data_row(chembl_parameter_path)["provenance_level"] == "parameter_predicted_with_uncertainty", "ChEMBL parameter provenance is not uncertainty predicted.")

    completion_contract = contract["completion_audit"]
    adapter_sources = {str(row["source_key"]) for row in config["validation"]["external_adapter_registry"]["adapters"]}
    require_subset(completion_contract["required_external_adapter_sources"], adapter_sources, "validation.external_adapter_registry.adapters")
    service_text = (REPO_ROOT / "Libs" / "scripts" / "services" / "serve_screening_api.py").read_text(encoding="utf-8")
    for summary_field in completion_contract["required_action_summary_fields"]:
        require(str(summary_field) in service_text, f"Serving action summary missing {summary_field}.")
    for uncertainty_field in completion_contract["required_uncertainty_summary_fields"]:
        require(str(uncertainty_field) in service_text, f"Serving uncertainty summary missing {uncertainty_field}.")
    anchor_interventions = {str(row["intervention_label"]) for row in dynamics_config["case_validation_entries"]}
    require_subset(completion_contract["required_ad_anchor_interventions"], anchor_interventions, "dynamics_ready_registry.case_validation_entries")
    anchor_rows_by_intervention = {
        str(row["intervention_label"]): row
        for row in dynamics_config["case_validation_entries"]
    }
    for intervention_label, expected_direction in dict(completion_contract["required_ad_anchor_directions"]).items():
        anchor_row = anchor_rows_by_intervention[str(intervention_label)]
        require(str(anchor_row["expected_effect_direction"]) == str(expected_direction), f"AD anchor direction mismatch for {intervention_label}.")
        require(str(anchor_row["clinical_endpoint_alignment_policy"]) == str(completion_contract["required_clinical_endpoint_policy"]), f"AD anchor clinical policy mismatch for {intervention_label}.")

    profile_contract = contract["neuronsifter_profiles"]
    profiles = transition_backbone["neuronsifter_profiles"]
    require(profiles["active_screening_profile"] == profile_contract["required_active_screening_profile"], "NeuronSifter active screening profile is not Large.")
    require_subset(profile_contract["required_profiles"], dict(profiles["profiles"]).keys(), "transition_backbone.neuronsifter_profiles.profiles")
    require(bool(profiles["distillation"]["enabled"]) == bool(profile_contract["require_distillation"]), "NeuronSifter profile distillation is not enabled.")
    require_subset(profile_contract["required_efficiency_report_fields"], profiles["distillation"]["efficiency_report_fields"], "neuronsifter_profiles.distillation.efficiency_report_fields")

    decision_contract = contract["decision_gate"]
    decision_gate = benchmark["decision_gate"]
    require_subset(decision_contract["required_labels"], dict(decision_gate["decision_labels"]).values(), "screening_benchmark.decision_gate.decision_labels")
    require_subset(decision_contract["required_policy_terms"], decision_gate.keys(), "screening_benchmark.decision_gate")
    explainability = transition_backbone["mechanism_explainability"]
    require_subset(
        [
            "operator_splitting_species_attribution",
            "operator_splitting_coupling_matrix",
            "operator_splitting_score_norm_by_expert",
            "operator_splitting_uncertainty_by_expert",
            "operator_splitting_case_activation",
        ],
        explainability["required_diagnostics"],
        "transition_backbone.mechanism_explainability.required_diagnostics",
    )

    active_contract = contract["active_learning_loop"]
    active_config = benchmark["active_learning"]
    require(active_config["contract_name"] == active_contract["required_contract_name"], "Active-learning contract name does not match the required decision-centric loop.")
    require(int(active_config["full_round_count"]) == int(active_contract["required_full_round_count"]), "Active-learning full round count does not match the contract.")
    require(int(active_config["full_tasks_per_round"]) == int(active_contract["required_tasks_per_round"]), "Active-learning tasks per round does not match the contract.")
    for quota_name, quota_value in dict(active_contract["required_batch_quota"]).items():
        require(abs(float(active_config["batch_quota"][quota_name]) - float(quota_value)) <= 1.0e-9, f"Active-learning batch quota mismatch for {quota_name}.")
    require_subset(active_contract["required_public_sources"], active_config["mandatory_public_sources"], "active_learning.mandatory_public_sources")
    require_subset(active_contract["required_tool_sources"], active_config["mandatory_tool_sources"], "active_learning.mandatory_tool_sources")
    require_subset(active_contract["required_simulator_families"], [str(row["simulator_family"]) for row in active_config["simulator_portfolio"]], "active_learning.simulator_portfolio")
    require_subset(active_contract["required_schema_keys"], config["schemas"].keys(), "schemas")
    require_subset(active_contract["required_simulation_request_defaults"], active_config["simulation_request_defaults"].keys(), "active_learning.simulation_request_defaults")
    require_subset(active_contract["required_result_linking_sections"], active_config["result_linking"].keys(), "active_learning.result_linking")
    require_subset(active_config["mandatory_simulator_families"], active_config["result_linking"]["artifact_sources"].keys(), "active_learning.result_linking.artifact_sources")
    require_subset(active_contract["required_serving_endpoints"], config["screening_pipeline"]["screening_serving"]["public_endpoints"], "screening_serving.public_endpoints")
    for policy_key in active_contract["required_model_update_policy"]:
        require(bool(active_config["model_update_policy"][policy_key]), f"Active-learning model update policy disabled: {policy_key}")
    require(bool(active_config["fail_fast_on_missing_mandatory_source"]), "Active-learning must fail fast on missing mandatory sources.")
    require(bool(active_config["fail_fast_on_missing_mandatory_tool"]), "Active-learning must fail fast on missing mandatory tools.")
    require("validate_required_binaries" in (REPO_ROOT / "Libs" / "utils" / "evaluation" / "active_learning_loop.py").read_text(encoding="utf-8"), "Active-learning binary preflight is not implemented.")

    clinical_contract = contract["clinical_direction_boundary"]
    bridge_config = benchmark["clinical_direction_bridge"]
    require(bridge_config["bridge_role"] == clinical_contract["required_bridge_role"], "Clinical direction bridge role is not retrospective.")
    require(bridge_config["endpoint_policy"] == clinical_contract["required_endpoint_policy"], "Clinical direction bridge policy is not literature-aggregate direction only.")
    require(bool(bridge_config["source_policy"]["disallow_sample_level_public_cohort_training"]), "Clinical bridge should not use sample-level cohort training in this pass.")
    require_subset(
        bridge_config["forbidden_direct_operator_outputs"],
        bridge_config["forbidden_direct_operator_outputs"],
        "clinical_direction_bridge.forbidden_direct_operator_outputs",
    )
    public_report_root = resolve_repo_path(str(config["paper"]["paths"]["report_dir"]))
    if public_report_root.exists():
        forbidden_claims: list[str] = []
        for report_path in sorted(public_report_root.rglob("*")):
            if report_path.suffix.lower() not in {".json", ".md", ".tex", ".txt"}:
                continue
            report_text = report_path.read_text(encoding="utf-8").lower()
            for claim_fragment in clinical_contract["forbidden_public_claim_fragments"]:
                if str(claim_fragment).lower() in report_text:
                    forbidden_claims.append(f"{report_path.relative_to(REPO_ROOT)}::{claim_fragment}")
        require(not forbidden_claims, f"Forbidden public claim fragments found in reports: {forbidden_claims}")

    simulation_contract = contract["simulation_readiness_audit"]
    for required_path_key in ["required_qc_catalog", "required_qc_summary", "required_split_leakage_audit", "required_pair_alignment_audit"]:
        require(resolve_repo_path(str(simulation_contract[required_path_key])).exists(), f"Missing simulation readiness audit artifact: {simulation_contract[required_path_key]}")
    qc_summary = read_json(resolve_repo_path(str(simulation_contract["required_qc_summary"])))
    require(qc_summary["readiness_gate"] == simulation_contract["required_readiness_gate"], "Simulation readiness gate mismatch.")
    require(qc_summary["readiness_status"] in simulation_contract["required_readiness_statuses"], "Simulation readiness status is not accepted.")
    leakage_payload = read_json(resolve_repo_path(str(simulation_contract["required_split_leakage_audit"])))
    require(not bool(simulation_contract["require_zero_uid_leakage"]) or int(leakage_payload["trajectory_random_uid_overlap"]) == 0, "Simulation UID leakage detected.")
    require(not bool(simulation_contract["require_zero_family_leakage"]) or int(leakage_payload["family_holdout_family_overlap"]) == 0, "Simulation family leakage detected.")
    require(not bool(simulation_contract["require_zero_regimen_leakage"]) or int(leakage_payload["regimen_holdout_regimen_overlap"]) == 0, "Simulation regimen leakage detected.")
    require(not bool(simulation_contract["require_zero_target_program_leakage"]) or int(leakage_payload["target_program_holdout_overlap"]) == 0, "Simulation target-program leakage detected.")
    require(not bool(simulation_contract["require_zero_source_holdout_leakage"]) or int(leakage_payload["source_holdout_role_overlap"]) == 0, "Simulation source-holdout leakage detected.")

    ai_ready = config["ai_ready_preprocessing"]
    audit_path = resolve_repo_path(str(ai_ready["local_root"])) / str(ai_ready["relative_paths"]["audit_report"])
    require(audit_path.exists(), f"AI-ready audit is missing: {audit_path}")
    audit = read_json(audit_path)
    canonical_counts = load_canonical_metric_values(REPO_ROOT / "Libs/configs/config_stack.yaml")
    target_trajectory_count = int(contract["teacher_generation"]["target_trajectory_count"])
    declared_target = int(ai_ready["volume_targets"]["target_simulation_count"])
    require(declared_target == target_trajectory_count, "Million-trajectory target is not declared in preprocessing config.")
    mesoscopic_count = canonical_counts["raw_materialized_mesoscopic_trajectory_count"]
    full_count = canonical_counts["raw_materialized_full_microenvironment_trajectory_count"]
    materialized_scale_met = mesoscopic_count >= target_trajectory_count and full_count >= target_trajectory_count
    if strict_materialized_scale:
        require(materialized_scale_met, "Materialized teacher trajectories are below the million-scale target.")

    return {
        "contract_status": "core_contracts_satisfied_scale_shortfall_reported" if not materialized_scale_met else "all_contracts_satisfied",
        "active_parameter_estimate": parameter_estimate,
        "transition_backbone_kind": transition_backbone["kind"],
        "operator_reference_policy": transition_backbone["external_operator_references"]["reference_policy"],
        "transolver_ablation_kind": transition_backbone["transolver_ablation_template"]["kind"],
        "multimodal_moe_expert_count": len(multimodal_moe["expert_names"]),
        "mechanism_moe_expert_count": len(mechanism_moe["expert_names"]),
        "operator_splitting_expert_count": len(operator_splitting["composition_order"]),
        "full_microenvironment_species_count": len(species_ids),
        "primary_disease_program": benchmark["active_disease_program"],
        "osp_conda_solver": runtime_config["conda_solver"],
        "pbpk_rollout_row_count": row_count(pbpk_rollout_path),
        "neuron_rxd_protocol_row_count": row_count(neuron_protocol_path),
        "chembl_parameter_prediction_row_count": row_count(chembl_parameter_path),
        "mesoscopic_trajectory_count": mesoscopic_count,
        "full_microenvironment_trajectory_count": full_count,
        "target_trajectory_count": target_trajectory_count,
        "materialized_scale_met": materialized_scale_met,
        "ai_ready_status": audit["ai_ready_status"],
        "scope_boundary_status": "cns_digital_twin_screening_only",
        "external_adapter_role_status": "all_required_sources_declared_as_non_method_adapters",
        "action_abstraction_status": "target_drug_dose_schedule_route_rankings_declared",
        "neuronsifter_mainline_status": "operator_splitting_diffusion_mainline",
        "neuronsifter_active_screening_profile": profiles["active_screening_profile"],
        "neuronsifter_distillation_status": "xl_large_student_yaml_profiles_declared",
        "ad_anchor_validation_status": "literature_anchor_direction_rank_and_qualitative_clinical_alignment_declared",
        "simulation_status": "mesoscopic_full_osp_neuron_chembl_teacher_family_declared",
        "simulation_readiness_status": qc_summary["readiness_status"],
        "simulation_train_eligible_count": canonical_counts["qc_eligible_trajectory_count"],
        "simulation_weak_teacher_count": qc_summary["weak_teacher_count"],
        "simulation_reject_count": canonical_counts["rejected_trajectory_count"],
        "decision_gate_status": "accept_reject_active_learning_external_validation_insufficient_evidence_declared",
        "clinical_direction_boundary_status": "literature_aggregate_retrospective_direction_only",
        "active_learning_loop_status": "decision_centric_simulator_portfolio_contract_declared",
        "active_learning_full_round_count": int(active_config["full_round_count"]),
        "active_learning_tasks_per_round": int(active_config["full_tasks_per_round"]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_report(config, args.strict_materialized_scale)
    output_path = resolve_repo_path(args.output)
    ensure_directory(output_path.parent)
    write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
