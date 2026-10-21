from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from Libs.scripts.validators.validate_dt_round16_what_if_engine import (
    MANDATORY_NEGATIVE_TEST_NAMES,
    _full_suite_execution_blockers,
    build_round16_gate_record,
)
from Libs.utils.data.what_if_contracts import (
    ELIGIBILITY_ROLES,
    SCHEMA_VERSIONS,
    UNCERTAINTY_COMPONENTS,
    canonical_sha256,
    validate_action_binding,
    validate_attempts,
    validate_baseline,
    validate_constraint_report,
    validate_contrast,
    validate_control_pair,
    validate_eligibility,
    validate_engine,
    validate_engine_compatibility,
    validate_gap_map,
    validate_reliability,
    validate_result,
    validate_rollout_trajectory,
    validate_uncertainty,
    validate_unique_uids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64
HASH_B = "b" * 64
COMMIT = "c" * 40


def _codes(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["code"]) for row in rows}


def _world_action(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": "neuronsifter-world-action-v1",
        "action_uid": "action_round16_no_action",
        "intervention_type": "no_action_control",
        "target_set": [],
        "agent_set": [],
        "dose": None,
        "dose_unit": None,
        "schedule": {"kind": "not_applicable"},
        "route": None,
        "duration": {"semantics": "contract_scope"},
        "stimulation_or_environment": {},
        "state_scope": {"region": ["hippocampus_ca1"], "compartment": ["isf"]},
        "support_domain_ref": "domain_round16",
        "source_evidence_refs": ["round15_restricted_e4"],
        "action_provenance_ref": "round15_action_binding",
        "conflicts": [],
        "status": "pilot_only",
    }
    value.update(updates)
    return value


def _request(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["what_if_request"],
        "request_uid": "request_round16",
        "research_question_uid": "question_round16",
        "baseline_state_ref": "baseline_round16",
        "candidate_world_action_ref": "action_round16_no_action",
        "control_action_ref": "action_round16_no_action",
        "requested_outcomes": ["free_drug"],
        "requested_time_horizon": {"start": 0.0, "end": 24.0, "unit": "hour"},
        "requested_time_grid": [0.0, 24.0],
        "requested_regions": ["hippocampus_ca1"],
        "requested_compartments": ["isf"],
        "requested_feature_namespace": "round16_state",
        "rollout_engine_policy": {"fallback_engine_allowed": False},
        "ensemble_policy": {"mode": "deterministic_singleton"},
        "fidelity_policy": {"requested_fidelity": "qualification"},
        "uncertainty_requirements": list(UNCERTAINTY_COMPONENTS),
        "constraint_requirements": ["non_negativity"],
        "support_domain_ref": "domain_round16",
        "budget": {"class": "small"},
        "provenance": {"source": "test"},
        "model_based_counterfactual": True,
        "causal_effect_claim": False,
        "clinical_recommendation": False,
        "empirical_observation": False,
    }
    value.update(updates)
    return value


def _baseline(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["what_if_baseline"],
        "baseline_uid": "baseline_round16",
        "state_ref": "Output/qualification_state.npy",
        "state_hash": HASH,
        "state_time": 0.0,
        "state_uncertainty": {"status": "qualification_only"},
        "biological_or_model_context": {"context": "synthetic_qualification"},
        "region": ["hippocampus_ca1"],
        "compartment": ["isf"],
        "feature_namespace": "round16_state",
        "support_domain_ref": "domain_round16",
        "source_role": "qualification_numerical_dynamics_only",
        "qc_grade": "S",
        "missingness": {"status": "none"},
        "provenance": {"source": "round15"},
        "baseline_class": "qualification_synthetic",
        "imputation_policy": "none_all_channels_explicit",
        "empirical_state": False,
        "qualification_only": True,
        "eligible_for_biological_claims": False,
    }
    value.update(updates)
    return value


def _binding(**updates: object) -> dict[str, object]:
    action = _world_action()
    value = {
        "schema_version": SCHEMA_VERSIONS["what_if_action_binding"],
        "binding_uid": "binding_round16",
        "request_uid": "request_round16",
        "world_action": action,
        "action_program_hash": canonical_sha256(action),
        "parameter_provenance": {"status": "verified_exact_configuration", "content_hash": HASH},
        "support_domain_ref": "domain_round16",
        "spatial_scope": action["state_scope"],
        "temporal_scope": {"time_grid": [0.0, 24.0], "unit": "hour"},
        "combination_components": [],
        "interaction_assumptions": [],
        "completeness": "complete",
        "blockers": [],
        "status": "validated",
    }
    value.update(updates)
    return value


def _engine(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_engine"],
        "engine_uid": "engine_round16",
        "engine_type": "validated_replay_engine",
        "version": "engine-v1",
        "source": "Libs/scripts/builders/build_dt_round16_what_if_engine.py",
        "code_identity": HASH,
        "environment_identity": HASH_B,
        "supported_baseline_classes": ["qualification_synthetic"],
        "supported_action_classes": ["no_action_control"],
        "supported_state_variables": ["free_drug"],
        "supported_horizons": [{"start": 0.0, "end": 24.0, "unit": "hour"}],
        "supported_fidelities": ["qualification"],
        "support_domain_ref": "domain_round16",
        "uncertainty_capabilities": ["unknown_preservation"],
        "constraint_capabilities": ["non_negativity"],
        "qualification_evidence": {"round15_gate_hash": HASH},
        "known_limitations": ["qualification_only"],
        "trained": False,
        "status": "qualified_for_contract_testing",
    }
    value.update(updates)
    return value


def _rollout(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_trajectory"],
        "rollout_uid": "rollout_round16",
        "request_uid": "request_round16",
        "attempt_uid": "attempt_round16",
        "engine_uid": "engine_round16",
        "baseline_uid": "baseline_round16",
        "action_uid": "action_round16_no_action",
        "control_pair_uid": "pair_round16",
        "native_time_grid": [0.0, 24.0],
        "canonical_time_grid": [0.0, 24.0],
        "state_namespace": "round16_state",
        "state_artifact_ref": "Output/state.npy",
        "uncertainty_artifact_ref": "Output/uncertainty.json",
        "constraint_artifact_ref": "Output/constraint.json",
        "content_hashes": {"state": HASH, "uncertainty": HASH_B},
        "status": "completed",
        "eligibility": "contract_only",
        "provenance": {"source": "round15_replay"},
        "output_class": "model_rollout",
        "empirical_observation": False,
    }
    value.update(updates)
    return value


def _pair(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_control_pair"],
        "pair_uid": "pair_round16",
        "action_rollout_uid": "rollout_action",
        "control_rollout_uid": "rollout_control",
        "control_type": "no_action_control",
        "baseline_state_family_match": True,
        "engine_version_match": True,
        "parameter_family_match": True,
        "fidelity_match": True,
        "time_grid_match": True,
        "seed_policy_compatible": True,
        "state_namespace_match": True,
        "support_domain_match": True,
        "pairing_decision": "exact_deterministic_pair",
        "intervention_effect_eligible": False,
        "blockers": [],
        "status": "validated",
    }
    value.update(updates)
    return value


def _constraint(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_constraint_report"],
        "constraint_report_uid": "constraint_round16",
        "rollout_uid": "rollout_round16",
        "constraint_policy": "constraint-v1",
        "required_constraints": ["non_negativity"],
        "passed_constraints": ["non_negativity"],
        "failed_constraints": [],
        "unknown_constraints": [],
        "violation_times": {},
        "violation_magnitudes": {},
        "handling_state": "valid",
        "formal_contrast_eligible": True,
    }
    value.update(updates)
    return value


def _uncertainty(**updates: object) -> dict[str, object]:
    components = {name: {"status": "unknown", "reason": "not measured"} for name in UNCERTAINTY_COMPONENTS}
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_uncertainty"],
        "uncertainty_uid": "uncertainty_round16",
        "rollout_uids": ["rollout_round16"],
        "components": components,
        "point_estimate": {"contrast": 0.0},
        "interval_or_distribution": {"status": "unavailable_unknown_components"},
        "aggregation_policy": "preserve_components",
        "unknown_components": list(UNCERTAINTY_COMPONENTS),
        "status": "reported_with_unknown_components",
    }
    value.update(updates)
    return value


def _reliability(**updates: object) -> dict[str, object]:
    diagnostics = {
        name: {"status": "unknown"}
        for name in (
            "repeatability", "seed_stability", "parameter_sensitivity", "fidelity_consistency", "horizon_degradation",
            "constraint_violation_rate", "model_family_disagreement", "baseline_perturbation_sensitivity",
            "action_perturbation_sensitivity", "support_domain_distance",
        )
    }
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_reliability_report"],
        "reliability_uid": "reliability_round16",
        "engine_uid": "engine_round16",
        "support_domain_ref": "domain_round16",
        "action_class": "no_action_control",
        "diagnostics": diagnostics,
        "qualified_horizon": {"start": 0.0, "end": 24.0, "unit": "hour"},
        "maximum_tested_horizon": {"start": 0.0, "end": 24.0, "unit": "hour"},
        "horizon_decision": "within_qualified_horizon",
        "support_domain_decision": "restricted_in_domain",
        "disagreement_outcome": "insufficient_comparison",
        "status": "qualified_with_restrictions",
        "restrictions": ["qualification_only"],
    }
    value.update(updates)
    return value


def _contrast(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["counterfactual_contrast"],
        "contrast_uid": "contrast_round16",
        "action_rollout_uids": ["rollout_action"],
        "control_rollout_uids": ["rollout_control"],
        "pairing_decision": "exact_deterministic_pair",
        "outcome_namespace": "round16_state",
        "contrast_estimator": "paired_difference",
        "time_local_contrasts": [0.0, 0.0],
        "terminal_contrast": 0.0,
        "integrated_contrast": 0.0,
        "uncertainty": {"unknown_components_preserved": True},
        "support_domain_ref": "domain_round16",
        "extrapolation_flags": ["qualification_only"],
        "conflicts": [],
        "eligibility": "contract_only",
        "causal_effect_claim": False,
        "intervention_effect_claim": False,
    }
    value.update(updates)
    return value


def _eligibility(**updates: object) -> dict[str, object]:
    roles = {name: False for name in ELIGIBILITY_ROLES}
    roles["eligible_for_contract_validation"] = True
    value = {
        "schema_version": SCHEMA_VERSIONS["what_if_eligibility_decision"],
        "decision_uid": "eligibility_round16",
        "result_uid": "result_round16",
        "controller_uid": "neuronsifter_orchestration_controller",
        "validator_uid": "round16_governed_what_if_engine_validator",
        "roles": roles,
        "engine_status": "qualified_for_contract_testing",
        "horizon_decision": "within_qualified_horizon",
        "pairing_decision": "exact_deterministic_pair",
        "support_domain_decision": "restricted_in_domain",
        "constraint_state": "valid",
        "uncertainty_status": "reported",
        "reliability_status": "qualified_with_restrictions",
        "exact_execution_identity": True,
        "reason_codes": ["contract_only"],
        "status": "decided",
    }
    value.update(updates)
    return value


def _result(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["what_if_result"],
        "result_uid": "result_round16",
        "request_uid": "request_round16",
        "rollout_uids": ["rollout_action", "rollout_control"],
        "control_pair_uid": "pair_round16",
        "contrast_uid": "contrast_round16",
        "uncertainty_uid": "uncertainty_round16",
        "reliability_uid": "reliability_round16",
        "eligibility_decision_uid": "eligibility_round16",
        "result_type": "restricted_model_replay_contract_result",
        "claim_scope": "contract_validation_only",
        "limitations": ["qualification_only"],
        "model_based_counterfactual": True,
        "causal_effect_claim": False,
        "clinical_recommendation": False,
        "empirical_observation": False,
        "user_facing_summary": "Restricted no-action replay for contract validation only.",
        "provenance": {"code_commit": COMMIT},
    }
    value.update(updates)
    return value


def _attempt(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["rollout_attempt"],
        "attempt_uid": "attempt_round16",
        "parent_request_uid": "request_round16",
        "submission_command": "read_only_replay",
        "runtime_identity": {"code_commit": COMMIT},
        "job_identity": {"job_id": "test"},
        "stdout_ref": "stdout",
        "stderr_ref": "stderr",
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "raw_output_hashes": {"state": HASH},
        "failure_class": None,
        "supersedes_attempt_uid": None,
        "preserved": True,
        "status": "completed",
    }
    value.update(updates)
    return value


def test_missing_baseline_rejected_round16() -> None:
    assert "missing_baseline" in _codes(validate_baseline(_baseline(state_hash="")))


def test_incomplete_action_rejected_round16() -> None:
    assert "incomplete_action" in _codes(validate_action_binding(_binding(completeness="incomplete")))


def test_compound_id_used_as_complete_action_rejected_round16() -> None:
    action = _world_action(
        intervention_type="pharmacological", agent_set=["compound-x"], target_set=[], dose=None, dose_unit=None,
        route="intravenous", schedule={"kind": "interval", "interval": 1, "unit": "day"},
    )
    binding = _binding(world_action=action, action_program_hash=canonical_sha256(action))
    assert "compound_id_not_complete_action" in _codes(validate_action_binding(binding))


def test_unregistered_rollout_engine_rejected() -> None:
    assert "unregistered_rollout_engine" in _codes(validate_engine(_engine(engine_type="mystery_engine")))


def test_untrained_model_declared_qualified_rejected() -> None:
    engine = _engine(engine_type="learned_world_action_model", trained=False, status="qualified_for_formal_what_if")
    assert "untrained_model_declared_qualified" in _codes(validate_engine(engine))


def test_unsupported_action_class_rejected_round16() -> None:
    engine = _engine(supported_action_classes=["pharmacological"])
    _, _, blockers = validate_engine_compatibility(request=_request(), baseline=_baseline(), action_binding=_binding(), engine=engine)
    assert "unsupported_action_class" in _codes(blockers)


def test_unsupported_state_namespace_rejected_round16() -> None:
    _, _, blockers = validate_engine_compatibility(
        request=_request(requested_outcomes=["unknown_state"]), baseline=_baseline(), action_binding=_binding(), engine=_engine()
    )
    assert "unsupported_state_namespace" in _codes(blockers)


def test_outside_horizon_formal_use_rejected() -> None:
    horizon, _, blockers = validate_engine_compatibility(
        request=_request(requested_time_horizon={"start": 0.0, "end": 48.0, "unit": "hour"}),
        baseline=_baseline(), action_binding=_binding(), engine=_engine(),
    )
    assert horizon == "outside_qualified_horizon"
    assert "outside_qualified_horizon" in _codes(blockers)


def test_unpaired_rollout_effect_rejected_round16() -> None:
    pair = _pair(pairing_decision="unpaired_descriptive_rollout", intervention_effect_eligible=True)
    assert "unpaired_rollout_described_as_intervention_effect" in _codes(validate_control_pair(pair))


def test_constraint_failed_rollout_contrast_rejected_round16() -> None:
    report = _constraint(
        passed_constraints=[], failed_constraints=["non_negativity"], handling_state="quarantined", formal_contrast_eligible=True
    )
    assert "constraint_failed_rollout_used_in_contrast" in _codes(validate_constraint_report(report))


def test_unknown_uncertainty_zero_rejected() -> None:
    value = _uncertainty()
    value["components"]["model_parameter_uncertainty"] = {"status": "unknown", "value": 0.0}
    assert "unknown_uncertainty_replaced_by_zero" in _codes(validate_uncertainty(value))


def test_model_rollout_as_empirical_observation_rejected() -> None:
    rollout = _rollout(output_class="canonical_observation", empirical_observation=True)
    assert "model_rollout_treated_as_empirical_observation" in _codes(validate_rollout_trajectory(rollout))


def test_model_result_closes_empirical_gap_rejected() -> None:
    rows = [{"gap_uid": f"gap-{index}", "round16_status": "requires_empirical_observation", "resolved_by_model_rollout": index == 0} for index in range(20)]
    assert "model_result_closes_empirical_frozen_gap" in _codes(validate_gap_map(rows))


def test_hypothetical_result_assimilation_observation_rejected() -> None:
    result = _result(result_type="canonical_observation", empirical_observation=True)
    assert "model_rollout_treated_as_empirical_observation" in _codes(validate_result(result))


def test_support_domain_expansion_rejected_round16() -> None:
    _, _, blockers = validate_engine_compatibility(
        request=_request(support_domain_ref="other_domain"), baseline=_baseline(), action_binding=_binding(), engine=_engine()
    )
    assert "support_domain_silently_expanded" in _codes(blockers)


def test_engine_disagreement_discarded_rejected() -> None:
    assert "engine_disagreement_discarded" in _codes(validate_reliability(_reliability(disagreement_outcome="averaged_away")))


def test_failed_attempt_overwritten_rejected_round16() -> None:
    assert "failed_attempt_overwritten" in _codes(validate_attempts([_attempt(), _attempt()]))


def test_causal_effect_language_rejected_round16() -> None:
    assert "causal_effect_language_emitted" in _codes(validate_contrast(_contrast(causal_effect_claim=True), constraint_state="valid"))


def test_clinical_recommendation_rejected_round16() -> None:
    assert "clinical_recommendation_emitted" in _codes(validate_result(_result(clinical_recommendation=True)))


def test_noncanonical_formal_authority_rejected_round16() -> None:
    decision = _eligibility(controller_uid="llm_agent")
    assert "noncanonical_component_authorizes_formal_result" in _codes(
        validate_eligibility(decision, canonical_controller_uid="neuronsifter_orchestration_controller")
    )


def test_duplicate_uid_rejected_round16() -> None:
    values = [{"request_uid": "duplicate"}, {"result_uid": "duplicate"}]
    assert "duplicate_uid" in _codes(validate_unique_uids(values, ("request_uid", "result_uid")))


def test_round16_schema_registry_is_canonical() -> None:
    import yaml

    config = yaml.safe_load((REPO_ROOT / "Libs/configs/schema/schemas.yaml").read_text(encoding="utf-8"))
    registry = config["schemas"]["round16_what_if_schema_registry"]
    assert set(registry) == {
        "what_if_request", "what_if_baseline", "what_if_action_binding", "rollout_engine", "rollout_request",
        "rollout_attempt", "rollout_trajectory", "rollout_control_pair", "counterfactual_contrast", "rollout_uncertainty",
        "rollout_constraint_report", "rollout_reliability_report", "what_if_eligibility_decision", "what_if_result",
        "round16_what_if_release_candidate",
    }
    for entry in registry.values():
        schema = json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema_version"]["const"] == entry["schema_version"]


def test_round16_mandatory_negative_inventory_matches_module() -> None:
    declared = {name for name, value in globals().items() if name.startswith("test_") and callable(value)}
    assert MANDATORY_NEGATIVE_TEST_NAMES <= declared


def test_round16_blocked_gate_is_not_canonical_decision() -> None:
    gate = build_round16_gate_record(
        authority={}, assessment="blocked_execution_verification",
        blockers=[{"code": "execution_identity_missing", "message": "missing"}], input_hashes={"pilot": HASH},
        config_hash=HASH, commit=COMMIT,
        report={
            "status": "blocked", "frozen_gap_count": 20,
            "execution_identity": {"dirty_status": "dirty"},
            "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        },
    )
    assert gate["schema_version"] == "neuronsifter-dt-round16-blocked-gate-assessment-v1"
    assert gate["writer_kind"] == "approved_validator_blocked_assessment"
    assert gate["overall_gate_status"] == "blocked"
    assert gate["canonical_controller_uid"] == "neuronsifter_orchestration_controller"


def test_round16_full_suite_uses_repository_authorized_cpu_nodes() -> None:
    identity = {"exact_git_commit": COMMIT}
    execution = {
        "schema_version": "neuronsifter-dt-round16-full-suite-execution-v1",
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "node": "cpu03",
        "exact_git_commit": COMMIT,
        "mandatory_skipped": 0,
        "core_xfail": 0,
    }
    assert _full_suite_execution_blockers(execution, identity) == []
    execution["node"] = "cpu04"
    assert _full_suite_execution_blockers(execution, identity) == []
    execution["node"] = "cpu05"
    assert _full_suite_execution_blockers(execution, identity)[0]["code"] == "execution_full_suite_failed"
