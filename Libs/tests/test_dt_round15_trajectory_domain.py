from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from Libs.scripts.validators.validate_dt_round15_trajectory_domain import (
    MANDATORY_NEGATIVE_TEST_NAMES,
    build_round15_gate_record,
)
from Libs.utils.data.trajectory_contracts import (
    SCHEMA_VERSIONS,
    admit_trajectory,
    canonical_sha256,
    validate_action_binding,
    validate_admission_decision,
    validate_control_pair,
    validate_failure_ledger,
    validate_fidelity_mapping,
    validate_initial_state,
    validate_producer,
    validate_stochastic_identity,
    validate_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64
HASH_B = "b" * 64
COMMIT = "c" * 40


def _world_action(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": "neuronsifter-world-action-v1",
        "action_uid": "world_action_no_action_control",
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
        "support_domain_ref": "domain_round15",
        "source_evidence_refs": ["round13_action_contract"],
        "action_provenance_ref": "prov_round15_action",
        "conflicts": [],
        "status": "pilot_only",
    }
    value.update(updates)
    return value


def _producer(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["producer"],
        "producer_uid": "producer_round15",
        "version": "producer-v1",
        "producer_type": "network_dynamics_model",
        "source": "Libs/utils/model/dynamics_fields.py",
        "source_hash": HASH,
        "license_or_access_status": "repository_internal_governed",
        "equation_or_model_identity": "explicit_recurrence",
        "input_schema": "input-v1",
        "output_schema": SCHEMA_VERSIONS["trajectory"],
        "supported_action_classes": ["no_action_control"],
        "supported_state_variables": ["free_drug", "target_bound_drug"],
        "support_domain_ref": "domain_round15",
        "parameter_policy": {"unknown_parameters_allowed": False, "status": "verified"},
        "seed_policy": {"stochastic": False, "required_seed": 3407},
        "numerical_solver_policy": {"family": "explicit_recurrence"},
        "known_limitations": ["qualification_only"],
        "status": "registered",
    }
    value.update(updates)
    return value


def _initial_state(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["initial_state"],
        "state_uid": "state_round15",
        "state_time": 0.0,
        "time_unit": "hour",
        "region_axis": ["hippocampus_ca1"],
        "compartment_axis": ["isf"],
        "species_or_feature_axis": ["free_drug", "target_bound_drug"],
        "units": {"free_drug": "relative_concentration", "target_bound_drug": "relative_concentration"},
        "missingness": {"status": "none"},
        "uncertainty": {"status": "qualification_only"},
        "biological_or_model_context": {"disease_or_context": "ad_control_qualification_only"},
        "support_domain_ref": "domain_round15",
        "source_lineage": {"source_kind": "validated_simulator_initialization"},
        "state_values_ref": "Output/trajectories/state.npy",
        "state_hash": HASH,
        "origin": "explicit_synthetic_qualification_state",
        "imputation_policy": "none_all_channels_explicit",
        "empirical_state": False,
        "qualification_only": True,
        "eligible_for_biological_validation": False,
        "status": "validated",
    }
    value.update(updates)
    return value


def _action_binding(**updates: object) -> dict[str, object]:
    world_action = _world_action()
    value = {
        "schema_version": SCHEMA_VERSIONS["action_binding"],
        "binding_uid": "binding_round15",
        "world_action": world_action,
        "action_program_hash": canonical_sha256(world_action),
        "exposure_occupancy_program_ref": None,
        "exposure_occupancy_status": "not_applicable",
        "producer_action_parameters": {"silent_defaults_used": False, "dose_multiplier": 0.0},
        "parameter_provenance": {"status": "verified_exact_configuration", "content_hash": HASH},
        "support_domain_ref": "domain_round15",
        "completeness": "complete",
        "blockers": [],
        "status": "validated",
    }
    value.update(updates)
    return value


def _trajectory(**updates: object) -> dict[str, object]:
    binding = _action_binding()
    value = {
        "schema_version": SCHEMA_VERSIONS["trajectory"],
        "trajectory_uid": "trajectory_round15",
        "trajectory_family_uid": "family_round15",
        "episode_uid": "episode_round15",
        "producer_uid": "producer_round15",
        "producer_version": "producer-v1",
        "producer_type": "network_dynamics_model",
        "initial_state_ref": "state_round15",
        "initial_state_hash": HASH,
        "world_action_ref": binding["world_action"]["action_uid"],
        "action_program_hash": binding["action_program_hash"],
        "no_action_control_ref": binding["world_action"]["action_uid"],
        "support_domain_ref": "domain_round15",
        "state_namespace": "round15_test_state",
        "region_axis": ["hippocampus_ca1"],
        "compartment_axis": ["isf"],
        "species_or_feature_axis": ["free_drug", "target_bound_drug"],
        "channel_units": {"free_drug": "relative_concentration", "target_bound_drug": "relative_concentration"},
        "start_time": 0.0,
        "end_time": 24.0,
        "time_unit": "hour",
        "time_grid": [0.0, 24.0],
        "native_solver_time_grid": [0.0, 24.0],
        "canonical_export_time_grid": [0.0, 24.0],
        "sampling_interval": {"value": 24.0, "unit": "hour"},
        "adaptive_step_metadata": {"adaptive": False, "original_grid_preserved": True},
        "interpolation": {"method": "none_identity_grid", "error_estimate": 0.0},
        "state_values_ref": "Output/trajectories/state_values.npy",
        "state_values_hash": HASH_B,
        "uncertainty_ref": "Output/trajectories/uncertainty.json",
        "constraint_diagnostics_ref": "Output/trajectories/constraint_qc.json",
        "numerical_qc_ref": "Output/trajectories/numerical_qc.json",
        "physical_qc_ref": "Output/trajectories/constraint_qc.json",
        "producer_inputs_hash": HASH,
        "producer_parameters_hash": HASH,
        "random_seed": 3407,
        "code_commit": COMMIT,
        "environment_hash": HASH,
        "runtime_identity": {"hostname": "cpu05"},
        "status": "generated",
        "eligibility": "qualification_numerical_dynamics_only",
        "provenance": {"source_kind": "new_governed_generation"},
        "derived_from_legacy_trajectory_ref": None,
        "promotion_method": "new_governed_generation",
        "promotion_validation": {"e3_row_updated_in_place": False},
        "fidelity_uid": "qualification_relative_field",
        "conflicts": [],
    }
    value.update(updates)
    return value


def _numerical(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["numerical_qc"],
        "qc_uid": "numerical_qc_round15",
        "trajectory_uid": "trajectory_round15",
        "policy_version": "numerical-policy-v1",
        "solver_status": "completed",
        "convergence_status": "not_applicable_explicit_recurrence",
        "step_count": 1,
        "rejected_steps": 0,
        "runtime_seconds": 0.1,
        "memory": {"array_bytes": 16},
        "nan_inf_count": 0,
        "overflow_underflow_indicators": {"overflow_count": 0, "underflow_count": 0},
        "state_bound_violations": 0,
        "negative_concentration_violations": 0,
        "discontinuities": 0,
        "interpolation_error": 0.0,
        "repeatability_diagnostics": {"exact_array_equal": True},
        "status": "passed",
        "blockers": [],
    }
    value.update(updates)
    return value


def _constraint(**updates: object) -> dict[str, object]:
    value = {
        "schema_version": SCHEMA_VERSIONS["constraint_qc"],
        "constraint_qc_uid": "constraint_qc_round15",
        "trajectory_uid": "trajectory_round15",
        "policy_version": "constraint-policy-v1",
        "required_constraints": ["non_negativity"],
        "constraint_results": {"non_negativity": {"state": "passed", "evidence": {"violation_count": 0}}},
        "unsupported_monotonicity_imposed": False,
        "status": "passed",
        "blockers": [],
    }
    value.update(updates)
    return value


def _qualification(**updates: object) -> dict[str, object]:
    value = {
        "qualification_uid": "qualification_round15",
        "status": "qualified_with_restrictions",
    }
    value.update(updates)
    return value


def _codes(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["code"]) for row in rows}


def test_trajectory_without_complete_initial_state_rejected() -> None:
    codes = _codes(validate_initial_state(_initial_state(units={}), known_support_domains={"domain_round15"}))
    assert "trajectory_initial_state_incomplete" in codes


def test_trajectory_without_complete_action_rejected() -> None:
    codes = _codes(validate_action_binding(_action_binding(completeness="incomplete"), known_support_domains={"domain_round15"}))
    assert "trajectory_action_incomplete" in codes


def test_drug_id_used_as_full_action_rejected_round15() -> None:
    action = _world_action(
        intervention_type="pharmacological",
        agent_set=["lecanemab"],
        target_set=[],
        dose=None,
        dose_unit=None,
        route="intravenous",
        schedule={"kind": "interval", "interval": 2, "unit": "week"},
    )
    binding = _action_binding(world_action=action, exposure_occupancy_status="compiled")
    assert "unresolved_target_identity" in _codes(validate_action_binding(binding))


def test_missing_dose_unit_rejected_round15() -> None:
    action = _world_action(
        intervention_type="pharmacological",
        agent_set=["lecanemab"],
        target_set=["APP"],
        dose=10.0,
        dose_unit=None,
        route="intravenous",
        schedule={"kind": "interval", "interval": 2, "unit": "week"},
    )
    assert "missing_dose_unit" in _codes(validate_action_binding(_action_binding(world_action=action)))


def test_missing_required_route_rejected_round15() -> None:
    action = _world_action(
        intervention_type="pharmacological",
        agent_set=["lecanemab"],
        target_set=["APP"],
        dose=10.0,
        dose_unit="mg/kg",
        route=None,
        schedule={"kind": "interval", "interval": 2, "unit": "week"},
    )
    assert "unsupported_route" in _codes(validate_action_binding(_action_binding(world_action=action)))


def test_unknown_simulator_version_rejected() -> None:
    assert "unknown_simulator_version" in _codes(validate_producer(_producer(version="latest")))


def test_unknown_parameter_provenance_rejected_round15() -> None:
    binding = _action_binding(parameter_provenance={"status": "unknown", "content_hash": HASH})
    assert "unknown_parameter_provenance" in _codes(validate_action_binding(binding))


def test_silent_default_initial_state_rejected() -> None:
    state = _initial_state(imputation_policy="silent_zero_fill")
    assert "silent_default_initial_state" in _codes(validate_initial_state(state))


def test_silent_default_action_parameters_rejected() -> None:
    binding = _action_binding(producer_action_parameters={"silent_defaults_used": True})
    assert "silent_default_action_parameters" in _codes(validate_action_binding(binding))


def test_static_endpoint_promoted_to_trajectory_rejected_round15() -> None:
    trajectory = _trajectory(provenance={"source_kind": "static_endpoint"})
    assert "static_endpoint_promoted_to_trajectory" in _codes(validate_trajectory(trajectory))


def test_e3_promoted_without_regeneration_rejected() -> None:
    trajectory = _trajectory(derived_from_legacy_trajectory_ref="legacy-e3", promotion_method="new_governed_generation")
    assert "legacy_promoted_without_regeneration" in _codes(validate_trajectory(trajectory))


def test_hypothetical_scenario_promoted_to_e4_rejected() -> None:
    trajectory = _trajectory(provenance={"source_kind": "hypothetical_scenario"})
    assert "hypothetical_promoted_to_e4" in _codes(validate_trajectory(trajectory))


def test_failed_solver_output_admitted_as_strong_rejected() -> None:
    admission = {
        "schema_version": SCHEMA_VERSIONS["admission"],
        "admission_uid": "admission_round15",
        "trajectory_uid": "trajectory_round15",
        "producer_qualification_ref": "qualification_round15",
        "initial_state_ref": "state_round15",
        "action_binding_ref": "binding_round15",
        "support_domain_ref": "domain_round15",
        "numerical_qc_ref": "numerical_qc_round15",
        "constraint_qc_ref": "constraint_qc_round15",
        "artifact_hash": HASH,
        "blocking_conflicts": [],
        "decision": "admitted_e4_strong",
        "eligibility": "strong_clean_dynamics",
        "reason_codes": ["admitted_e4_strong"],
        "provenance": {"validator_uid": "round15"},
        "status": "decided",
    }
    assert "failed_solver_admitted_strong" in _codes(
        validate_admission_decision(admission, numerical_qc=_numerical(solver_status="failed"), constraint_qc=_constraint())
    )


def test_nan_inf_trajectory_admitted_rejected() -> None:
    decision = admit_trajectory(
        trajectory=_trajectory(),
        producer_qualification=_qualification(status="qualified"),
        initial_state=_initial_state(),
        action_binding=_action_binding(),
        numerical_qc=_numerical(nan_inf_count=1, status="failed", blockers=[{"code": "nan_inf_trajectory"}]),
        constraint_qc=_constraint(),
        known_support_domains={"domain_round15"},
    )
    assert decision["decision"] == "quarantined_numerical"


def test_constraint_failure_ignored_rejected() -> None:
    decision = admit_trajectory(
        trajectory=_trajectory(),
        producer_qualification=_qualification(status="qualified"),
        initial_state=_initial_state(),
        action_binding=_action_binding(),
        numerical_qc=_numerical(),
        constraint_qc=_constraint(
            constraint_results={"non_negativity": {"state": "failed", "evidence": {}}},
            status="failed",
            blockers=[{"code": "constraint_failure"}],
        ),
        known_support_domains={"domain_round15"},
    )
    assert decision["decision"] == "quarantined_constraint"


def test_unpaired_action_trajectory_contrast_rejected() -> None:
    pair = {
        "schema_version": SCHEMA_VERSIONS["control_pair"],
        "pair_uid": "pair_round15",
        "action_trajectory_uid": "trajectory_action",
        "control_trajectory_uid": None,
        "initial_state_family_match": False,
        "producer_version_match": False,
        "parameter_family_match": False,
        "seed_policy_compatible": False,
        "export_time_grid_match": False,
        "pairing_decision": "unpaired_trajectory",
        "action_effect_claim_eligible": True,
        "blockers": [],
        "status": "blocked",
    }
    assert "unpaired_action_contrast" in _codes(validate_control_pair(pair))


def test_different_seed_roles_treated_as_identical_samples_rejected() -> None:
    rows = [
        {"seed_family_uid": "seed-a", "seed": "3407", "sample_content_hash": HASH, "independence_group_uid": "group-a", "replicate_role": "within_action_stochastic_replicate"},
        {"seed_family_uid": "seed-a", "seed": "3407", "sample_content_hash": HASH, "independence_group_uid": "group-b", "replicate_role": "between_initial_state_replicate"},
    ]
    assert "seed_role_collision" in _codes(validate_stochastic_identity(rows))


def test_low_fidelity_silently_substituted_rejected() -> None:
    assert "low_fidelity_substitution" in _codes(
        validate_fidelity_mapping(requested_fidelity="high", actual_fidelity="low", mapping_ref=None)
    )


def test_unknown_support_domain_admitted_rejected_round15() -> None:
    assert "unknown_support_domain" in _codes(validate_trajectory(_trajectory(), known_support_domains={"other_domain"}))


def test_missing_artifact_hash_admitted_rejected_round15() -> None:
    admission = {
        "schema_version": SCHEMA_VERSIONS["admission"],
        "admission_uid": "admission_round15",
        "trajectory_uid": "trajectory_round15",
        "producer_qualification_ref": "qualification_round15",
        "initial_state_ref": "state_round15",
        "action_binding_ref": "binding_round15",
        "support_domain_ref": "domain_round15",
        "numerical_qc_ref": "numerical_qc_round15",
        "constraint_qc_ref": "constraint_qc_round15",
        "artifact_hash": "",
        "blocking_conflicts": [],
        "decision": "admitted_e4_restricted",
        "eligibility": "qualification_numerical_dynamics_only",
        "reason_codes": ["admitted_e4_restricted"],
        "provenance": {"validator_uid": "round15"},
        "status": "decided",
    }
    assert "missing_artifact_hash" in _codes(validate_admission_decision(admission))


def test_failed_attempt_overwritten_by_retry_rejected() -> None:
    rows = [
        {"attempt_uid": "attempt-1", "preserved": True, "status": "retained_failure"},
        {"attempt_uid": "attempt-1", "preserved": True, "status": "retained_failure"},
    ]
    assert "failed_attempt_overwritten" in _codes(validate_failure_ledger(rows))


def test_round15_schema_registry_is_canonical() -> None:
    import yaml

    config = yaml.safe_load((REPO_ROOT / "Libs/configs/schema/schemas.yaml").read_text(encoding="utf-8"))
    registry = config["schemas"]["round15_trajectory_schema_registry"]
    assert set(registry) == {
        "clean_action_complete_trajectory",
        "trajectory_producer",
        "trajectory_producer_qualification",
        "trajectory_initial_state",
        "trajectory_action_binding",
        "trajectory_numerical_qc",
        "trajectory_constraint_qc",
        "trajectory_control_pair",
        "trajectory_failure",
        "trajectory_admission_decision",
        "trajectory_family_manifest",
        "round15_e4_release_candidate",
    }
    for entry in registry.values():
        schema = json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema_version"]["const"] == entry["schema_version"]


def test_mandatory_negative_inventory_matches_test_module() -> None:
    declared = {name for name, value in globals().items() if name.startswith("test_") and callable(value)}
    assert MANDATORY_NEGATIVE_TEST_NAMES <= declared


def test_blocked_gate_does_not_claim_canonical_authority() -> None:
    blocker = {"code": "execution_identity_missing", "message": "Exact execution identity is missing."}
    gate = build_round15_gate_record(
        authority={},
        assessment="blocked_execution_verification",
        blockers=[blocker],
        input_hashes={"pilot_manifest": HASH},
        config_hash=HASH,
        commit=COMMIT,
        report={
            "status": "blocked",
            "execution_identity": {"dirty_status": "dirty"},
            "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        },
        materialization={"gap_count": 20, "resolved_gap_count": 0},
    )
    assert gate["schema_version"] == "neuronsifter-dt-round15-blocked-gate-assessment-v1"
    assert gate["writer_kind"] == "approved_validator_blocked_assessment"
    assert "controller_uid" not in gate
    assert gate["canonical_controller_uid"] == "neuronsifter_orchestration_controller"
    assert gate["overall_gate_status"] == "blocked"
    assert gate["blocker_classes"] == ["execution_verification"]
