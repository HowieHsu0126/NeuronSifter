from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
import yaml

from Libs.scripts.validators.validate_dt_round17_scitil_runtime import (
    MANDATORY_NEGATIVE_TEST_NAMES,
    validate_event_authority_binding,
    validate_positive_assessment_lineage,
)
from Libs.utils.data.scitil_contracts import (
    AGENT_ROLES,
    CANONICAL_CONTROLLER_UID,
    DEBT_COMPONENTS,
    ROUND17_VALIDATOR_UID,
    SCHEMA_VERSIONS,
    TwinKernel,
    belief_state_content_hash,
    canonical_sha256,
    create_belief_state_from_assimilation,
    validate_agent_identity,
    validate_agent_proposal,
    validate_attempt_preservation,
    validate_budget_contracts,
    validate_dual_action_decision,
    validate_explanation,
    validate_frozen_gap_map,
    validate_h_gate,
    validate_hypothesis,
    validate_lineage_acyclic,
    validate_loop_event,
    validate_twin_debt,
    validate_unique_uids,
    validate_w_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def _codes(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["code"]) for row in rows}


def _agent(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["agent_identity"],
        "agent_uid": "round17_scientist_agent",
        "agent_role": "ScientistAgent",
        "capabilities": ["propose_hypothesis", "propose_world_action"],
        "prohibited_capabilities": ["write_gate", "execute_tool_or_rollout", "update_twin_state"],
        "gate_writer": False,
        "tool_executor": False,
        "state_update_authority": False,
        "canonical_controller": False,
        "provenance": {"source": "test"},
    }
    value.update(updates)
    return value


def _proposal(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["agent_proposal"],
        "proposal_uid": "proposal_round17",
        "agent_uid": "round17_scientist_agent",
        "agent_role": "ScientistAgent",
        "belief_state_ref": "belief_round17",
        "proposal_type": "world_action_candidate",
        "hypothesis_ref": None,
        "world_action_ref": "action_round17",
        "information_action_ref": None,
        "what_if_request_ref": None,
        "observation_request_ref": None,
        "expected_information_gain": {"status": "represented", "relative": 0.1},
        "expected_twin_debt_change": {"status": "represented", "direction": "unknown"},
        "expected_cost": {"compute_seconds": 1.0},
        "risk_class": "restricted_computational",
        "support_domain_ref": "domain_round17",
        "evidence_refs": ["evidence_round17"],
        "assumptions": ["contract_only"],
        "uncertainties": ["biological_validity_unknown"],
        "prohibited_claim_flags": {"causal_effect_claim": False, "clinical_recommendation": False},
        "provenance": {"source": "test", "compiled": True},
    }
    value.update(updates)
    return value


def _hypothesis(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["scientific_hypothesis"],
        "hypothesis_uid": "hypothesis_round17",
        "mechanism_statement": "Restricted replay stability may be retained within its model domain.",
        "target_variables": ["repeatability"],
        "expected_outcome": {"direction": "stable"},
        "required_context": {"baseline": "qualification_synthetic"},
        "time_horizon": {"start": 0.0, "end": 1.0, "unit": "hour"},
        "potential_falsifier": {"disconfirming_prediction": "A constraint failure refutes stability."},
        "supporting_evidence": ["round16_restricted_result"],
        "conflicting_evidence": ["conflict_round17"],
        "support_domain_ref": "domain_round17",
        "uncertainty": {"biological_validity": "unknown"},
        "causal_effect_claim": False,
        "clinical_efficacy_claim": False,
        "hypothetical_as_empirical_support": False,
    }
    value.update(updates)
    return value


def _gate(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["hypothesis_gate_decision"],
        "gate_uid": "h_gate_round17",
        "gate_type": "H-Gate",
        "proposal_uid": "proposal_round17",
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "validator_uid": ROUND17_VALIDATOR_UID,
        "input_hashes": {"proposal": HASH},
        "decision": "admitted_for_testing",
        "reason_codes": ["falsifiable"],
        "approved_scope": {"execution_scope": "contract_pilot_only"},
        "budget_effect": {"increase_allowed": False},
        "twin_debt_effect": {"unknown_to_zero": False},
        "support_domain_effect": {"expansion_allowed": False},
        "provenance": {"writer": CANONICAL_CONTROLLER_UID},
    }
    value.update(updates)
    return value


def _world_action(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "neuronsifter-world-action-v1",
        "action_uid": "action_round17",
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
        "support_domain_ref": "domain_round17",
        "source_evidence_refs": ["round15_restricted_e4"],
        "action_provenance_ref": "round15_action_binding",
        "conflicts": [],
        "status": "pilot_only",
    }
    value.update(updates)
    return value


def _debt(status: str = "low", value: float | None = 0.1, **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["twin_debt"],
        "debt_uid": "debt_round17",
        "belief_state_ref": "belief_round17",
        "component_values": {name: value for name in DEBT_COMPONENTS},
        "component_status": {name: status for name in DEBT_COMPONENTS},
        "measurement_basis": {"componentwise": True},
        "unknown_components": list(DEBT_COMPONENTS) if status == "unknown" else [],
        "versioned_thresholds": {"version": "debt-v1", "low_max": 0.25, "moderate_max": 0.6},
        "allowed_decisions": ["world_action_restricted", "information_action", "stop"],
        "aggregation": None,
        "provenance": {"source": "test"},
    }
    result.update(updates)
    return result


def _budget_contracts(*, request: float = 1.0, decision: str = "authorized") -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    budget = {
        "schema_version": SCHEMA_VERSIONS["scientific_budget"],
        "budget_uid": "budget_round17",
        "dimensions": {"compute_seconds": 10.0},
        "consumed": {"compute_seconds": 0.0},
        "remaining": {"compute_seconds": 10.0},
        "iteration_limit": 2,
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "version": "budget-v1",
        "provenance": {"source": "test"},
    }
    risk = {
        "schema_version": SCHEMA_VERSIONS["risk_budget"],
        "risk_budget_uid": "risk_round17",
        "allowed_risk_classes": ["restricted_computational"],
        "maximum_risk_class": "restricted_computational",
        "production_execution_allowed": False,
        "agent_may_increase_risk": False,
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "version": "risk-v1",
        "provenance": {"source": "test"},
    }
    budget_decision = {
        "decision_uid": "budget_decision_round17",
        "proposal_uid": "proposal_round17",
        "budget_uid": "budget_round17",
        "risk_budget_uid": "risk_round17",
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "requested_cost": {"compute_seconds": request},
        "requested_risk_class": "restricted_computational",
        "decision": decision,
        "remaining_after": {"compute_seconds": 10.0},
        "reason_codes": ["test"],
    }
    return budget, risk, budget_decision


def _w_gate(**updates: object) -> dict[str, object]:
    value = _gate(
        schema_version=SCHEMA_VERSIONS["world_action_gate_decision"],
        gate_uid="w_gate_round17",
        gate_type="W-Gate",
        decision="authorized_restricted_rollout",
    )
    value.update(updates)
    return value


def _event(prior: str, next_state: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSIONS["agent_loop_event"],
        "event_uid": f"event_{prior}_{next_state}",
        "episode_uid": "episode_round17",
        "prior_state": prior,
        "next_state": next_state,
        "agent_uid": "round17_scientist_agent",
        "gate_uid": "gate_round17",
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "input_hashes": {"prior": HASH},
        "output_hashes": {"next": HASH},
        "timestamp": "2026-07-29T00:00:00+00:00",
        "decision_reason": "negative_transition_fixture",
    }


def _explanation(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["agent_explanation"],
        "explanation_uid": "explanation_round17",
        "episode_uid": "episode_round17",
        "belief_state_ref": "belief_round17",
        "gate_decision_refs": ["gate_round17"],
        "rollout_or_tool_artifact_refs": ["rollout_round17"],
        "uncertainty_report_ref": "uncertainty_round17",
        "reliability_report_ref": "reliability_round17",
        "support_domain_report_ref": "domain_round17",
        "conflicts": [],
        "claim_eligibility_ref": "eligibility_round17",
        "summary": "The restricted model result remains conditional.",
        "limitations": ["No empirical or causal conclusion"],
        "model_based_counterfactual": True,
        "empirical_observation": False,
        "causal_effect_claim": False,
        "clinical_recommendation": False,
        "provenance": {"accepted_artifacts_only": True},
    }
    value.update(updates)
    return value


def test_round17_valid_core_contracts_pass() -> None:
    budget, risk, budget_decision = _budget_contracts()
    assert validate_agent_identity(_agent()) == []
    assert validate_agent_proposal(_proposal(), agent=_agent(), known_belief_states={"belief_round17"}) == []
    assert validate_hypothesis(_hypothesis()) == []
    assert validate_twin_debt(_debt()) == []
    assert validate_budget_contracts(budget, risk, budget_decision) == []
    assert TwinKernel(controller_uid=CANONICAL_CONTROLLER_UID, allowlisted_adapters={"adapter": {}}).llm_backed is False


def test_agent_direct_state_update_rejected_round17() -> None:
    proposal = _proposal(provenance={"direct_state_update": {"posterior_write": True}})
    assert "agent_arbitrary_execution_request" in _codes(validate_agent_proposal(proposal, agent=_agent()))


def test_agent_gate_write_rejected_round17() -> None:
    gate = _gate(controller_uid="round17_scientist_agent")
    assert "agent_writes_gate" in _codes(validate_h_gate(gate, proposal=_proposal(), hypothesis=_hypothesis()))


def test_agent_arbitrary_shell_invocation_rejected_round17() -> None:
    proposal = _proposal(provenance={"shell_command": "echo forbidden"})
    assert "agent_arbitrary_execution_request" in _codes(validate_agent_proposal(proposal, agent=_agent()))


def test_agent_h_gate_bypass_rejected_round17() -> None:
    assert "illegal_agent_loop_transition" in _codes(validate_loop_event(_event("hypothesis_proposed", "action_candidates_generated")))


def test_agent_w_gate_bypass_rejected_round17() -> None:
    assert "illegal_agent_loop_transition" in _codes(validate_loop_event(_event("action_candidates_generated", "execution_authorized")))


def test_agent_i_gate_bypass_rejected_round17() -> None:
    event = _event("action_candidates_generated", "execution_completed")
    assert "illegal_agent_loop_transition" in _codes(validate_loop_event(event))


def test_agent_o_gate_bypass_rejected_round17() -> None:
    assert "illegal_agent_loop_transition" in _codes(validate_loop_event(_event("observation_normalized", "belief_updated")))


def test_tool_result_direct_belief_state_rejected_round17() -> None:
    with pytest.raises(ValueError, match="O-Gate"):
        create_belief_state_from_assimilation(
            parent={"belief_state_uid": "belief_round17"},
            o_gate={"controller_uid": CANONICAL_CONTROLLER_UID, "decision": "no_update"},
            assimilation_decision={},
            posterior_refs={},
            active_hypothesis_refs=[],
            constraint_state_ref="constraint",
            support_domain_ref="domain_round17",
            twin_debt_ref="debt_round17",
            evidence_snapshot_ref="evidence_round17",
            provenance={"source": "raw_tool_result"},
        )


def test_model_rollout_empirical_observation_rejected_round17() -> None:
    assert "model_rollout_treated_as_empirical_observation" in _codes(
        validate_explanation(_explanation(empirical_observation=True))
    )


def test_agent_evidence_qc_upgrade_rejected_round17() -> None:
    proposal = _proposal(provenance={"qc_upgrade": {"qc_grade": "A"}})
    assert "agent_arbitrary_execution_request" in _codes(validate_agent_proposal(proposal, agent=_agent()))


def test_agent_biological_pairing_creation_rejected_round17() -> None:
    proposal = _proposal(provenance={"biological_pairing": {"pairing_decision": "exact"}})
    assert "agent_arbitrary_execution_request" in _codes(validate_agent_proposal(proposal, agent=_agent()))


def test_agent_conflict_removal_rejected_round17() -> None:
    proposal = _proposal(provenance={"remove_conflict": "conflict_round17"})
    assert "agent_arbitrary_execution_request" in _codes(validate_agent_proposal(proposal, agent=_agent()))


def test_agent_frozen_gap_reasoning_closure_rejected_round17() -> None:
    rows = [{"gap_uid": "gap_round17", "round17_impact": "resolved_by_agent_reasoning", "resolved_by_agent_reasoning": True}]
    assert "agent_closes_frozen_gap_through_reasoning" in _codes(validate_frozen_gap_map(rows, expected_count=1))


def test_unknown_twin_debt_zero_rejected_round17() -> None:
    assert "unknown_twin_debt_replaced_by_zero" in _codes(validate_twin_debt(_debt(status="unknown", value=0.0)))


def test_high_twin_debt_world_action_rejected_round17() -> None:
    _, _, budget_decision = _budget_contracts()
    blockers = validate_w_gate(
        _w_gate(),
        proposal=_proposal(),
        world_action=_world_action(),
        what_if_request=None,
        twin_debt=_debt(status="high", value=0.9),
        budget_decision=budget_decision,
    )
    assert "high_twin_debt_world_action_authorized" in _codes(blockers)


def test_high_twin_debt_catalog_cannot_list_world_action_round17() -> None:
    blockers = validate_twin_debt(_debt(status="high", value=0.9))
    assert "high_twin_debt_world_action_listed" in _codes(blockers)


def test_positive_assessment_requires_linked_falsifier_and_no_update_o_gate_round17() -> None:
    ledger = {
        "hypothesis_uid": "hypothesis_round17",
        "current_status": "supported_model_conditionally",
        "falsifier_agent_ref": "round17_falsifier_agent",
        "conflicting_evidence": ["conflict_round17"],
        "unresolved_conflict_count": 1,
        "test_history": ["scenario_round17"],
    }
    proposals = [
        {
            "proposal_uid": "proposal_falsifier_round17",
            "agent_role": "FalsifierAgent",
            "proposal_type": "falsification_test",
            "hypothesis_ref": "hypothesis_round17",
        }
    ]
    gates = [
        {
            "gate_uid": "h_gate_falsifier_round17",
            "gate_type": "H-Gate",
            "proposal_uid": "proposal_falsifier_round17",
            "decision": "admitted_for_testing",
        },
        {
            "gate_uid": "o_gate_no_update_round17",
            "gate_type": "O-Gate",
            "proposal_uid": "proposal_rollout_round17",
            "decision": "no_update",
        },
    ]
    episodes = [{"episode_uid": "episode_round17", "scenario_uid": "scenario_round17"}]
    events = {
        "episode_round17": [
            {
                "agent_uid": "round17_falsifier_agent",
                "gate_uid": "h_gate_falsifier_round17",
                "next_state": "assimilation_proposed",
            },
            {
                "agent_uid": None,
                "gate_uid": "o_gate_no_update_round17",
                "next_state": "assimilation_gated",
            },
            {
                "agent_uid": "round17_falsifier_agent",
                "gate_uid": "o_gate_no_update_round17",
                "next_state": "explanation_generated",
            },
        ]
    }
    explanations = [
        {
            "episode_uid": "episode_round17",
            "gate_decision_refs": ["h_gate_falsifier_round17", "o_gate_no_update_round17"],
        }
    ]
    lineage_edges = {("proposal_falsifier_round17", "h_gate_falsifier_round17")}
    kwargs = {
        "ledger": ledger,
        "falsifier_uids": {"round17_falsifier_agent"},
        "proposals": proposals,
        "gates": gates,
        "episodes": episodes,
        "events_by_episode": events,
        "explanations": explanations,
        "lineage_edges": lineage_edges,
    }
    assert validate_positive_assessment_lineage(**kwargs) == []

    orphaned_events = {
        "episode_round17": [
            {**row, "agent_uid": "round17_scientist_agent"} for row in events["episode_round17"]
        ]
    }
    blockers = validate_positive_assessment_lineage(**{**kwargs, "events_by_episode": orphaned_events})
    assert "falsifier_or_conflict_dimension_missing" in _codes(blockers)


def test_event_gate_type_and_agent_proposal_binding_fail_closed_round17() -> None:
    event = {
        "event_uid": "event_action_gate_round17",
        "prior_state": "action_candidates_generated",
        "next_state": "action_gated",
        "agent_uid": "round17_experiment_agent",
        "gate_uid": "o_gate_wrong_round17",
    }
    gates = {
        "o_gate_wrong_round17": {
            "gate_uid": "o_gate_wrong_round17",
            "gate_type": "O-Gate",
            "proposal_uid": "proposal_scientist_round17",
        }
    }
    agents = {"round17_experiment_agent": {"agent_role": "ExperimentAgent"}}
    proposals = {"proposal_scientist_round17": {"agent_uid": "round17_scientist_agent"}}
    blockers = validate_event_authority_binding(
        event=event,
        gate_by_uid=gates,
        agent_by_uid=agents,
        proposal_by_uid=proposals,
    )
    assert "round17_event_gate_type_mismatch" in _codes(blockers)

    gates["o_gate_wrong_round17"]["gate_type"] = "I-Gate"
    blockers = validate_event_authority_binding(
        event=event,
        gate_by_uid=gates,
        agent_by_uid=agents,
        proposal_by_uid=proposals,
    )
    assert "round17_event_agent_proposal_mismatch" in _codes(blockers)


def test_budget_exceeded_silently_rejected_round17() -> None:
    budget, risk, decision = _budget_contracts(request=100.0, decision="authorized")
    assert "budget_exceeded_silently" in _codes(validate_budget_contracts(budget, risk, decision))


def test_unsupported_domain_ignored_rejected_round17() -> None:
    _, _, budget_decision = _budget_contracts()
    blockers = validate_w_gate(
        _w_gate(),
        proposal=_proposal(support_domain_ref="domain_request"),
        world_action=_world_action(support_domain_ref="domain_engine"),
        what_if_request=None,
        twin_debt=_debt(),
        budget_decision=budget_decision,
    )
    assert "w_gate_support_domain_mismatch" in _codes(blockers)


def test_causal_effect_statement_rejected_round17() -> None:
    assert "causal_effect_statement_emitted" in _codes(validate_explanation(_explanation(causal_effect_claim=True)))


def test_clinical_recommendation_rejected_round17() -> None:
    assert "clinical_recommendation_emitted" in _codes(validate_explanation(_explanation(clinical_recommendation=True)))


def test_failed_tool_attempt_overwrite_rejected_round17() -> None:
    attempt = {
        "attempt_uid": "attempt_round17",
        "terminal_state": "FAILED",
        "failure_class": "intentional_failure",
        "output_hashes": {},
        "preserved": True,
    }
    assert "failed_tool_attempt_overwritten" in _codes(validate_attempt_preservation([attempt, dict(attempt)]))


def test_blocked_loop_belief_update_rejected_round17() -> None:
    blockers = validate_loop_event(_event("blocked", "belief_updated"))
    assert {"illegal_agent_loop_transition", "blocked_loop_updates_belief"} <= _codes(blockers)


def test_noncanonical_assimilation_authority_rejected_round17() -> None:
    with pytest.raises(ValueError, match="AssimilationDecision"):
        create_belief_state_from_assimilation(
            parent={"belief_state_uid": "belief_round17"},
            o_gate={"controller_uid": CANONICAL_CONTROLLER_UID, "decision": "approved_uncertainty_only", "gate_uid": "o_gate"},
            assimilation_decision={"controller_uid": "round17_scientist_agent", "decision": "approved_uncertainty_only"},
            posterior_refs={
                "state_posterior_ref": "state",
                "parameter_posterior_ref": "parameters",
                "model_family_posterior_ref": "models",
            },
            active_hypothesis_refs=[],
            constraint_state_ref="constraint",
            support_domain_ref="domain_round17",
            twin_debt_ref="debt_round17",
            evidence_snapshot_ref="evidence_round17",
            provenance={"source": "test"},
        )


def test_round17_schema_registry_is_canonical() -> None:
    registry_document = yaml.safe_load((REPO_ROOT / "Libs/configs/schema/schemas.yaml").read_text(encoding="utf-8"))
    registry = registry_document["schemas"]["round17_scitil_schema_registry"]
    assert set(registry) == {
        "twin_belief_state",
        "agent_identity",
        "agent_proposal",
        "scientific_hypothesis",
        "hypothesis_gate_decision",
        "world_action_gate_decision",
        "information_action_gate_decision",
        "observation_gate_decision",
        "dual_action_decision",
        "twin_debt",
        "scientific_budget",
        "risk_budget",
        "agent_loop_event",
        "agent_loop_episode",
        "hypothesis_ledger_entry",
        "agent_explanation",
        "round17_scitil_release_candidate",
    }
    for key, entry in registry.items():
        schema = json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        version_key = "release" if key == "round17_scitil_release_candidate" else key
        assert schema["properties"]["schema_version"]["const"] == entry["schema_version"]
        assert entry["schema_version"] == SCHEMA_VERSIONS[version_key]
        assert schema["additionalProperties"] is False


def test_round17_agent_role_registry_is_exact() -> None:
    assert AGENT_ROLES == {"ScientistAgent", "FalsifierAgent", "ExperimentAgent", "EvidenceAgent"}


def test_round17_mandatory_negative_inventory_matches_module() -> None:
    declared = {name for name, value in globals().items() if name.startswith("test_") and callable(value)}
    assert MANDATORY_NEGATIVE_TEST_NAMES <= declared


def test_round17_controller_authority_is_canonical() -> None:
    authority = yaml.safe_load(
        (REPO_ROOT / "Libs/configs/validation/round11f_controller_authority.yaml").read_text(encoding="utf-8")
    )
    runtime = authority["scitil_runtime_authority"]
    assert runtime["controller_uid"] == CANONICAL_CONTROLLER_UID
    assert runtime["approved_validator_uid"] == ROUND17_VALIDATOR_UID
    assert runtime["gate_types"] == ["H-Gate", "W-Gate", "I-Gate", "O-Gate"]
    assert runtime["agent_tool_execution_allowed"] is False
    assert runtime["agent_state_update_allowed"] is False
    assert runtime["arbitrary_shell_execution_allowed"] is False


def test_round17_dual_action_prefers_information_under_high_debt() -> None:
    decision = {
        "schema_version": SCHEMA_VERSIONS["dual_action_decision"],
        "decision_uid": "dual_round17",
        "belief_state_ref": "belief_round17",
        "candidate_world_actions": ["action_round17"],
        "candidate_information_actions": ["information_round17"],
        "expected_utility": {"world_action": 1.0, "information_action": 0.1},
        "expected_information_gain": {"world_action": 0.0, "information_action": 0.9},
        "expected_twin_debt_change": {"world_action": "unknown", "information_action": "decrease"},
        "cost": {"bounded": True},
        "risk": {"class": "external_tool_read_only"},
        "support_domain_ref": "domain_round17",
        "selected_action_type": "information_action",
        "selected_action_ref": "information_round17",
        "selection_reason": ["high_debt"],
        "controller_uid": CANONICAL_CONTROLLER_UID,
    }
    assert validate_dual_action_decision(decision, twin_debt=_debt(status="high", value=0.9)) == []


def test_round17_uid_and_lineage_guards_fail_closed() -> None:
    assert "duplicate_uid" in _codes(validate_unique_uids([{"uid": "x"}, {"uid": "x"}], ("uid",)))
    assert "lineage_cycle" in _codes(validate_lineage_acyclic([("a", "b"), ("b", "a")]))


def test_round17_belief_hash_excludes_self_identity_fields() -> None:
    state = {
        "belief_state_uid": "pending",
        "content_hash": "pending",
        "state_posterior_ref": "state",
        "controller_uid": CANONICAL_CONTROLLER_UID,
    }
    expected = canonical_sha256({"state_posterior_ref": "state", "controller_uid": CANONICAL_CONTROLLER_UID})
    assert belief_state_content_hash(state) == expected
