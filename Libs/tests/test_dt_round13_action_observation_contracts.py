from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from Libs.scripts.builders.build_dt_round13_action_observation_contracts import (
    SCHEMA_PATHS,
    build_round13_pilot,
    write_round13_artifacts,
)
from Libs.scripts.validators.validate_dt_round13_action_observation_contracts import (
    MANDATORY_NEGATIVE_TEST_NAMES,
    build_round14_handoff,
    validate_round13_outputs,
)
from Libs.utils.data.action_observation_contracts import (
    SCHEMA_VERSIONS,
    ASSIMILATION_UPDATE_TYPES,
    AssimilationProposal,
    CanonicalObservation,
    HypotheticalActionScenario,
    InformationAction,
    ObservationQC,
    ObservationRequest,
    RawToolResult,
    ToolRequest,
    WORLD_ACTION_TYPES,
    WorldAction,
    action_support_decision,
    compile_nonpharmacological_action,
    compile_pharmacological_action,
    conflicts_preserved,
    frozen_gap_action_observation_rows,
    match_tool_capability,
    normalize_raw_tool_result,
    validate_assimilation_decision,
    validate_assimilation_proposal,
    validate_frozen_gap_action_observation_map,
    validate_hypothetical_scenario,
    validate_information_action,
    validate_observation_qc,
    validate_observation_request,
    validate_raw_tool_result,
    validate_tool_request,
    validate_world_action,
)
from Libs.utils.data.evidence_relationships import EvidenceConflict, EvidenceSupportDomain
from Libs.utils.data.evidence_union import build_evidence_union
from Libs.utils.data.raw_data_registry import load_config, sha256_of_file
from Libs.utils.exp.orchestration_controller import (
    CANONICAL_CONTROLLER_UID,
    GateAuthorityError,
    build_canonical_assimilation_decision,
    load_authority_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_PATH = REPO_ROOT / "Libs/configs/validation/round11f_controller_authority.yaml"
VALIDATOR_UID = "round13_action_observation_contract_validator"
SUPPORTED_COMPONENTS = {"neuroinflammation_uncertainty", "evidence_conflict_register"}


def _world_action(**updates: object) -> dict[str, object]:
    value = WorldAction(
        schema_version=SCHEMA_VERSIONS["world_action"],
        action_uid="world_action_test",
        intervention_type="pharmacological",
        target_set=["APP"],
        agent_set=["lecanemab"],
        dose=10.0,
        dose_unit="mg/kg",
        schedule={"kind": "interval", "interval": 2, "unit": "week"},
        route="intravenous",
        duration={"value": 18, "unit": "month"},
        stimulation_or_environment={"kind": "not_applicable"},
        state_scope={"region": ["hippocampus"], "compartment": ["isf"]},
        support_domain_ref="domain_human",
        source_evidence_refs=["evidence_ref"],
        action_provenance_ref="verified_action_provenance",
        conflicts=[],
        status="contract_valid",
    ).to_dict()
    value.update(updates)
    return value


def _information_action(**updates: object) -> dict[str, object]:
    value = InformationAction(
        schema_version=SCHEMA_VERSIONS["information_action"],
        information_action_uid="information_action_test",
        candidate_uids=["world_action_test"],
        requested_observables=["inflammatory_mediator"],
        tool_or_source_uid="measurement_tool",
        fidelity="pilot",
        measurement_context={"organism": "human"},
        budget_class="low",
        expected_cost={"value": 1.0},
        expected_runtime={"seconds": 1.0},
        failure_policy="fail_closed",
        support_domain_ref="domain_human",
        provenance_ref="verified_information_provenance",
        conflicts=[],
        status="contract_valid",
    ).to_dict()
    value.update(updates)
    return value


def _tool_request(**updates: object) -> dict[str, object]:
    value = ToolRequest(
        schema_version="neuronsifter-tool-request-v1",
        tool_request_uid="tool_request_test",
        information_action_uid="information_action_test",
        observation_request_uid="observation_request_test",
        tool_uid="measurement_tool",
        tool_version="1.0.0",
        requested_observables=["inflammatory_mediator"],
        organism_context="human",
        state_action_type="static_state_measurement",
        input_schema_version="neuronsifter-observation-request-v1",
        output_schema_version="measurement-result-v1",
        fidelity="pilot",
        budget={"maximum_cost": 2.0},
        execution_payload={"request_ref": "observation_request_test"},
        support_domain_ref="domain_human",
        provenance_ref="verified_tool_request",
        idempotency_key="round13-tool-request-test",
        sandbox_policy="read_only_adapter",
        conflicts=[],
        status="pilot_only",
    ).to_dict()
    value.update(updates)
    return value


def _capability(**updates: object) -> dict[str, object]:
    value = {
        "tool_uid": "measurement_tool",
        "tool_version": "1.0.0",
        "supported_observables": ["inflammatory_mediator"],
        "organism_scope": ["human"],
        "state_action_types": ["static_state_measurement"],
        "input_schema_versions": ["neuronsifter-observation-request-v1"],
        "output_schema_versions": ["measurement-result-v1"],
        "fidelities": ["pilot"],
        "maximum_cost": 1.0,
        "license_status": "resolved",
        "availability": "available",
        "restrictions": [],
        "provenance_ref": "verified_capability_registry",
        "conflicts": [],
    }
    value.update(updates)
    return value


def _raw_result(**updates: object) -> dict[str, object]:
    value = RawToolResult(
        schema_version=SCHEMA_VERSIONS["raw_tool_result"],
        tool_result_uid="raw_result_test",
        tool_uid="measurement_tool",
        tool_version="1.0.0",
        request_uid="tool_request_test",
        raw_artifact_hashes={"result": "a" * 64},
        exit_status={"code": 0, "state": "completed"},
        runtime={"seconds": 1.0},
        resource_usage={"cpu_seconds": 1.0},
        raw_schema={"schema_version": "measurement-result-v1"},
        failure_class="none",
        stdout_ref="contract://stdout",
        stderr_ref="contract://stderr",
        support_domain_ref="domain_human",
        provenance_ref="verified_raw_result",
        conflicts=[],
        state_effect="none",
    ).to_dict()
    value.update(updates)
    return value


def _qc(grade: str = "C", **updates: object) -> dict[str, object]:
    dimensions = {
        "schema_validity": "passed",
        "entity_mapping_quality": "passed",
        "unit_validity": "passed",
        "biological_unit_identity": "passed",
        "time_alignment": "passed",
        "action_alignment": "passed",
        "assay_quality": "passed",
        "missingness": "passed",
        "outlier_status": "passed",
        "source_reliability": "partial" if grade == "C" else "passed",
        "tool_reliability": "partial" if grade == "C" else "passed",
    }
    value = ObservationQC(
        schema_version=SCHEMA_VERSIONS["observation_qc"],
        qc_uid="qc_test",
        source_result_uid="raw_result_test",
        qc_grade=grade,
        predecessor_qc_grade=grade,
        verification_artifact_ref=None,
        reason_codes=[],
        provenance_ref="verified_qc",
        **dimensions,
    ).to_dict()
    value.update(updates)
    return value


def _mapping() -> dict[str, object]:
    return {
        "schema_validated": True,
        "entity_mapping_verified": True,
        "unit_conversion_verified": True,
        "axis_alignment_verified": True,
        "biological_unit_verified": True,
        "provenance_verified": True,
        "biological_unit_uid": "unit_001",
        "state_time": {"semantics": "static_endpoint", "value": 0, "unit": "hour"},
        "region": "hippocampus",
        "compartment": "isf",
        "cell_type": "mixed_cns",
        "modality": "protein_assay",
        "feature_namespace": "neuronsifter_state_species",
        "feature_id": "inflammatory_mediator",
        "value": 0.5,
        "unit": "relative_concentration",
        "uncertainty": {"lower": 0.4, "upper": 0.6},
        "missingness": {"state": "observed"},
    }


def _observation(**updates: object) -> dict[str, object]:
    value = CanonicalObservation(
        schema_version=SCHEMA_VERSIONS["canonical_observation"],
        observation_uid="observation_test",
        source_result_uid="raw_result_test",
        biological_unit_uid="unit_001",
        state_time={"semantics": "static_endpoint", "value": 0, "unit": "hour"},
        region="hippocampus",
        compartment="isf",
        cell_type="mixed_cns",
        modality="protein_assay",
        feature_namespace="neuronsifter_state_species",
        feature_id="inflammatory_mediator",
        value=0.5,
        unit="relative_concentration",
        uncertainty={"lower": 0.4, "upper": 0.6},
        missingness={"state": "observed"},
        qc_grade="C",
        support_domain_ref="domain_human",
        provenance_ref="verified_observation",
        train_eval_role="directional_only",
        conflicts=[],
        empirical_observation=True,
    ).to_dict()
    value.update(updates)
    return value


def _domain(uid: str, organism: str = "human", status: str = "in_domain") -> dict[str, object]:
    return EvidenceSupportDomain(
        schema_version="neuronsifter-evidence-support-domain-v1",
        support_domain_uid=uid,
        organism_scope=[organism],
        disease_scope=["ad_control"],
        region_scope=["hippocampus"],
        compartment_scope=["isf"],
        cell_type_scope=["mixed_cns"],
        modality_scope=["protein_assay"],
        assay_scope=["protein_assay"],
        action_scope=["pharmacological"],
        dose_scope=["contracted"],
        route_scope=["intravenous"],
        time_scope=["static_endpoint"],
        model_family_scope=["not_applicable"],
        tool_family_scope=["measurement_tool"],
        source_scope=["contract_test"],
        generalization_status=status,
        provenance_ref="verified_domain",
        status="pilot_only" if status == "in_domain" else "blocked",
    ).to_dict()


def _proposal(observation: Mapping[str, object], **updates: object) -> dict[str, object]:
    observation_uid = str(observation["observation_uid"])
    value = AssimilationProposal(
        schema_version=SCHEMA_VERSIONS["assimilation_proposal"],
        proposal_uid="proposal_test",
        observation_uids=[observation_uid],
        target_state_components=["neuroinflammation_uncertainty"],
        prior_state_ref="contract://prior-state",
        proposed_update_type="uncertainty_only_update",
        likelihood_or_update_model_ref="contract://update-model",
        uncertainty_change={"direction": "increase"},
        support_domain="domain_human",
        conflicts=list(observation.get("conflicts", [])),
        constraint_effects={"overwrite": False},
        expected_twin_debt_change={"direction": "unchanged"},
        provenance_ref="verified_proposal",
        train_eval_roles={observation_uid: str(observation["train_eval_role"])},
        source_kinds={observation_uid: "canonical_observation"},
        state_write_authorized=False,
        validation_status="passed",
        blockers=[],
    ).to_dict()
    value.update(updates)
    return value


def _proposal_blockers(proposal: Mapping[str, object], observation: Mapping[str, object], domains: dict[str, dict[str, object]]) -> set[str]:
    return {
        str(row["code"])
        for row in validate_assimilation_proposal(
            proposal,
            observations={str(observation["observation_uid"]): observation},
            support_domains=domains,
            supported_state_components=SUPPORTED_COMPONENTS,
        )
    }


def _handoff_fixture() -> dict[str, object]:
    node_types = ["claim"] * 14 + ["perturbation_endpoint", "structured_action", "observation", "trajectory", "tool_result", "observation"]
    gaps = [
        {
            "gap_uid": f"gap_{index:02d}",
            "missing_node_type": node_type,
            "missing_relationship_type": "derived_from",
            "required_identity_axes": "study_uid;biological_unit_uid",
            "available_evidence": "contract_test",
            "blocked_pairing_reason": "source absent",
            "affected_downstream_role": "hypothesis_generation",
            "resolution_owner": f"owner_{index:02d}",
            "status": "materialization_required",
        }
        for index, node_type in enumerate(node_types)
    ]
    domains = [
        _domain("domain_human_ad_hippocampus"),
        {**_domain("domain_simulator_stress"), "generalization_status": "stress_only", "status": "stress_only"},
        {**_domain("domain_hypothetical_unsupported"), "generalization_status": "unsupported", "status": "blocked"},
    ]
    conflict = EvidenceConflict(
        schema_version="neuronsifter-evidence-conflict-v1",
        conflict_uid="conflict_test",
        conflict_type="directional_disagreement",
        member_uids=["left", "right"],
        state="unresolved",
        typed_context_difference={},
        provenance_ref="verified_conflict",
        status="pilot_only",
    ).to_dict()
    return {
        "schema_version": "neuronsifter-dt-round13-handoff-v1",
        "status": "ready_for_structured_world_action_contracts",
        "parent_gate": "typed_evidence_relationships_resolved",
        "support_domain_rules": domains,
        "conflict_semantics": [conflict],
        "frozen_gap_ledger": gaps,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def test_all_round13_schemas_are_valid_and_registered() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    registry = config["schemas"]["round13_action_observation_schema_registry"]
    schema_aliases = {**SCHEMA_PATHS, "round13_action_observation_graph": SCHEMA_PATHS["graph"]}
    schema_aliases.pop("graph")
    assert set(registry) == set(schema_aliases)
    for family, entry in registry.items():
        version_family = "graph" if family == "round13_action_observation_graph" else family
        assert entry["schema_version"] == SCHEMA_VERSIONS[version_family]
        assert entry["path"] == schema_aliases[family]
        Draft202012Validator.check_schema(json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8")))


def test_world_and_information_actions_are_explicitly_separated() -> None:
    assert validate_world_action(_world_action()) == []
    assert validate_information_action(_information_action()) == []
    assert set(ASSIMILATION_UPDATE_TYPES) >= {"uncertainty_only_update", "conflict_register_only", "no_update"}


def test_pharmacological_compiler_uses_only_provided_parameters() -> None:
    program = compile_pharmacological_action(
        _world_action(),
        pkpd_context={
            "time_grid": [0, 24],
            "parameter_provenance": {"status": "verified", "artifact_ref": "repo://pkpd"},
            "target_program": {"targets": ["APP"]},
            "exposure_fields": {"free_drug": [0, 1]},
            "occupancy_fields": {"APP": [0, 0.5]},
            "off_target_context": {"status": "unknown"},
            "uncertainty": {"kind": "interval"},
            "organism_context": "human",
            "supported_organisms": ["human"],
        },
        baseline_state_posterior={"state_ref": "state"},
        graph_context={"graph_ref": "graph"},
    )
    assert program["status"] == "compiled_with_explicit_uncertainty"
    assert program["occupancy_fields"] == {"APP": [0, 0.5]}


@pytest.mark.parametrize("action_type", ["genetic", "electrical_stimulation", "optical_stimulation", "environmental", "state_initialization"])
def test_nonpharmacological_adapters_are_typed(action_type: str) -> None:
    action = _world_action(
        intervention_type=action_type,
        target_set=["APP"] if action_type == "genetic" else [],
        agent_set=[],
        dose=None,
        dose_unit=None,
        schedule={"kind": "not_applicable"},
        route=None,
        stimulation_or_environment={
            "affected_state_variables": ["neuroinflammation"],
            "magnitude_or_program": {"value": 0.1},
            "uncertainty": {"kind": "bounded"},
        },
    )
    field, blockers = compile_nonpharmacological_action(action)
    assert blockers == []
    assert field is not None and field["intervention_type"] == action_type


def test_no_action_control_compiles_without_occupancy() -> None:
    action = _world_action(
        intervention_type="no_action_control",
        target_set=[],
        agent_set=[],
        dose=None,
        dose_unit=None,
        schedule={"kind": "not_applicable"},
        route=None,
        stimulation_or_environment={},
    )
    field, blockers = compile_nonpharmacological_action(action)
    assert blockers == []
    assert field is not None and field["magnitude_or_program"] == {"control": True}


def test_observation_request_and_tool_capability_match_pass() -> None:
    request = ObservationRequest(
        schema_version=SCHEMA_VERSIONS["observation_request"],
        request_uid="observation_request_test",
        research_question_ref="question",
        requested_variable_namespace="neuronsifter_state_species",
        requested_feature_ids=["inflammatory_mediator"],
        region="hippocampus",
        compartment="isf",
        cell_type="mixed_cns",
        biological_unit_scope="sample_level",
        timepoint_or_time_window={"semantics": "static_endpoint"},
        assay_or_measurement_type="protein_assay",
        required_units=["relative_concentration"],
        minimum_quality="C",
        minimum_fidelity="pilot",
        uncertainty_requirement={"required": True},
        budget={"maximum_cost": 2.0},
        tool_availability_constraints={"known_version_required": True},
        support_domain_ref="domain_human",
        provenance_ref="verified_request",
        status="contract_valid",
    ).to_dict()
    assert validate_observation_request(request) == []
    match = match_tool_capability(_tool_request(), _capability())
    assert match["state"] == "supported"
    assert all(match[field] for field in match if field.endswith(("_supported", "_compatible", "_available", "_sufficient", "_resolved")))


def test_raw_result_normalizes_to_qc_c_directional_observation() -> None:
    observation, blockers = normalize_raw_tool_result(
        _raw_result(), mapping=_mapping(), qc_value=_qc("C"), provenance_status="verified"
    )
    assert blockers == []
    assert observation is not None
    assert observation["qc_grade"] == "C"
    assert observation["train_eval_role"] == "directional_only"


def test_qc_s_conflict_is_preserved_as_stress_only() -> None:
    raw = _raw_result(conflicts=["conflict_test"])
    qc = _qc("S", outlier_status="flagged", source_reliability="flagged")
    observation, blockers = normalize_raw_tool_result(raw, mapping=_mapping(), qc_value=qc, provenance_status="verified")
    assert blockers == []
    assert observation is not None
    assert observation["train_eval_role"] == "stress_only"
    assert observation["conflicts"] == ["conflict_test"]


def test_approved_uncertainty_only_assimilation_is_controller_owned() -> None:
    observation = _observation()
    proposal = _proposal(observation)
    domains = {"domain_human": _domain("domain_human")}
    assert _proposal_blockers(proposal, observation, domains) == set()
    authority = load_authority_contract(AUTHORITY_PATH)
    decision = build_canonical_assimilation_decision(
        authority,
        proposal_uid=str(proposal["proposal_uid"]),
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=VALIDATOR_UID,
        validator_report={"validator_uid": VALIDATOR_UID, "status": "passed"},
        prior_state_hash="b" * 64,
        input_observation_hashes={str(observation["observation_uid"]): "c" * 64},
        decision="approved_uncertainty_only",
        reason_codes=["directional_uncertainty_only"],
        approved_update_scope=["neuroinflammation_uncertainty"],
        conflict_handling={},
        resulting_state_ref="contract://not-materialized",
        provenance_ref="verified_decision",
    )
    assert validate_assimilation_decision(
        decision,
        proposal=proposal,
        canonical_controller_uid=CANONICAL_CONTROLLER_UID,
        approved_validator_uid=VALIDATOR_UID,
    ) == []


def test_drug_id_treated_as_complete_action_rejected() -> None:
    action = _world_action(target_set=[], dose=None, dose_unit=None, schedule={}, route=None)
    codes = {row["code"] for row in validate_world_action(action)}
    assert {"unresolved_target_identity", "missing_dose", "missing_dose_unit", "unsupported_route", "invalid_schedule"} <= codes


def test_tool_query_treated_as_world_action_rejected() -> None:
    action = {**_world_action(), "tool_or_source_uid": "measurement_tool", "requested_observables": ["x"]}
    assert "world_information_action_not_separated" in {row["code"] for row in validate_world_action(action)}


def test_tool_result_directly_updates_twin_rejected() -> None:
    result = {**_raw_result(state_effect="posterior_update"), "resulting_state_ref": "state"}
    assert "tool_result_direct_state_update" in {row["code"] for row in validate_raw_tool_result(result)}


def test_llm_proposal_directly_updates_twin_rejected() -> None:
    observation = _observation()
    proposal = _proposal(observation, state_write_authorized=True, provenance_ref="llm_proposal")
    codes = _proposal_blockers(proposal, observation, {"domain_human": _domain("domain_human")})
    assert "proposal_direct_state_update" in codes


def test_missing_dose_unit_rejected() -> None:
    assert "missing_dose_unit" in {row["code"] for row in validate_world_action(_world_action(dose_unit=None))}


def test_unsupported_route_rejected() -> None:
    assert "unsupported_route" in {row["code"] for row in validate_world_action(_world_action(route="unregistered_route"))}


def test_unknown_parameter_provenance_rejected() -> None:
    program = compile_pharmacological_action(
        _world_action(),
        pkpd_context={
            "time_grid": [0, 1],
            "parameter_provenance": {"status": "unknown"},
            "target_program": {"targets": ["APP"]},
            "exposure_fields": {"free_drug": [0, 1]},
            "occupancy_fields": {"APP": [0, 0.5]},
            "organism_context": "human",
            "supported_organisms": ["human"],
        },
        baseline_state_posterior={"state_ref": "state"},
        graph_context={"graph_ref": "graph"},
    )
    assert program["status"] == "insufficient_parameters"
    assert "unknown_parameter_provenance" in {row["code"] for row in program["blockers"]}


def test_static_endpoint_authorizes_trajectory_update_rejected() -> None:
    observation = _observation()
    proposal = _proposal(observation, target_state_components=["future_state_trajectory"])
    codes = _proposal_blockers(proposal, observation, {"domain_human": _domain("domain_human")})
    assert "static_endpoint_temporal_update" in codes


def test_hypothetical_scenario_becomes_observation_rejected() -> None:
    action = _world_action()
    scenario = HypotheticalActionScenario(
        schema_version=SCHEMA_VERSIONS["hypothetical_action_scenario"],
        scenario_uid="scenario_test",
        baseline_state_ref="state",
        candidate_world_action=action,
        assumptions=["hypothetical"],
        scenario_context={"purpose": "test"},
        requested_outputs=["state"],
        support_warnings=["not_empirical"],
        support_domain_ref="domain_human",
        provenance_ref="verified_scenario",
        model_based_counterfactual=True,
        causal_effect_claim=False,
        clinical_recommendation=False,
        empirical_observation=True,
    ).to_dict()
    assert "hypothetical_boundary_violation" in {row["code"] for row in validate_hypothetical_scenario(scenario)}


def test_unknown_provenance_becomes_strong_observation_rejected() -> None:
    observation, blockers = normalize_raw_tool_result(
        _raw_result(provenance_ref="unknown"), mapping=_mapping(), qc_value=_qc("B"), provenance_status="unknown"
    )
    assert observation is None
    assert "unknown_observation_provenance" in {row["code"] for row in blockers}


def test_qc_d_evidence_enters_assimilation_rejected() -> None:
    observation = _observation(qc_grade="D", train_eval_role="rejected")
    proposal = _proposal(observation)
    assert "rejected_qc" in _proposal_blockers(proposal, observation, {"domain_human": _domain("domain_human")})


def test_cross_domain_observation_updates_state_rejected() -> None:
    observation = _observation(support_domain_ref="domain_mouse")
    proposal = _proposal(observation)
    domains = {"domain_human": _domain("domain_human"), "domain_mouse": _domain("domain_mouse", organism="mouse", status="unsupported")}
    assert "cross_domain_assimilation" in _proposal_blockers(proposal, observation, domains)


def test_conflict_disappears_during_normalization_rejected() -> None:
    assert "conflict_dropped" in {row["code"] for row in conflicts_preserved({"conflicts": ["c1"]}, {"conflicts": []})}


def _union_member(uid: str, grade: str, eligible: bool) -> dict[str, object]:
    return {
        "artifact_uid": uid,
        "artifact_hash": ("1" if uid == "left" else "2") * 64,
        "producer_uid": "observation_normalizer",
        "schema_version": SCHEMA_VERSIONS["canonical_observation"],
        "lineage_group": "round13_observation",
        "provenance_status": "verified",
        "evidence_type": "canonical_observation",
        "evidence_class": "directional_evidence",
        "qc_grade": grade,
        "train_eval_role": "directional_only",
        "claim_eligibility": "blocked",
        "independence_group": "round13_observation",
        "status": "accepted",
        "semantic_fingerprint": "same_observation",
        "pairing_decision": "unsupported_pairing",
        "typed_conflicts": [],
        "relationship_eligibility": {"eligible_for_state_pretraining": eligible},
    }


def test_evidence_union_upgrades_observation_eligibility_rejected() -> None:
    union = build_evidence_union(
        union_uid="round13_observation_union",
        members=[_union_member("left", "B", True), _union_member("right", "C", False)],
        canonical_output_ref="contract://union",
    )
    assert union["canonical_downstream_eligibility"]["eligible_for_state_pretraining"] is False
    assert union["canonical_downstream_eligibility"]["qc_grade"] == "C"


def test_noncanonical_controller_approves_assimilation_rejected() -> None:
    authority = load_authority_contract(AUTHORITY_PATH)
    with pytest.raises(GateAuthorityError):
        build_canonical_assimilation_decision(
            authority,
            proposal_uid="proposal",
            controller_uid="llm_agent",
            validator_uid=VALIDATOR_UID,
            validator_report={"validator_uid": VALIDATOR_UID, "status": "passed"},
            prior_state_hash="a" * 64,
            input_observation_hashes={"observation": "b" * 64},
            decision="approved",
            reason_codes=[],
            approved_update_scope=["state"],
            conflict_handling={},
            resulting_state_ref="state",
            provenance_ref="provenance",
            writer_kind="llm_agent",
        )


def test_all_noncanonical_component_kinds_cannot_approve_assimilation() -> None:
    authority = load_authority_contract(AUTHORITY_PATH)
    for writer_kind in ("builder_direct", "tool_adapter", "llm_agent", "evidence_producer", "report_generator", "baseline_pipeline"):
        with pytest.raises(GateAuthorityError):
            build_canonical_assimilation_decision(
                authority,
                proposal_uid="proposal",
                controller_uid="noncanonical_component",
                validator_uid=VALIDATOR_UID,
                validator_report={"validator_uid": VALIDATOR_UID, "status": "passed"},
                prior_state_hash="a" * 64,
                input_observation_hashes={"observation": "b" * 64},
                decision="approved",
                reason_codes=[],
                approved_update_scope=["state"],
                conflict_handling={},
                resulting_state_ref="state",
                provenance_ref="provenance",
                writer_kind=writer_kind,
            )


def test_unsupported_tool_silently_replaced_rejected() -> None:
    request = _tool_request(tool_uid="unregistered_tool")
    match = match_tool_capability(request, _capability())
    assert match["state"] == "unsupported_domain"
    assert match["tool_uid"] != request["tool_uid"]


def test_arbitrary_shell_command_generated_by_action_rejected() -> None:
    request = _tool_request(execution_payload={"shell_command": "python arbitrary.py"})
    report = validate_tool_request(request)
    assert report.status == "blocked"
    assert "arbitrary_shell_command_prohibited" in {row["code"] for row in report.blockers}


def test_frozen_gap_reduced_without_evidence_rejected() -> None:
    handoff = _handoff_fixture()
    rows = frozen_gap_action_observation_rows(handoff)
    rows[0]["current_status"] = "resolved"
    assert "frozen_gap_reduced_without_evidence" in {
        row["code"] for row in validate_frozen_gap_action_observation_map(handoff, rows)
    }


def test_qc_upgrade_requires_versioned_verification_artifact() -> None:
    qc = _qc("A", predecessor_qc_grade="C", verification_artifact_ref=None)
    assert "observation_qc_upgrade_without_verification" in {row["code"] for row in validate_observation_qc(qc)}


def test_out_of_domain_action_is_retained_only_as_unsupported() -> None:
    action = _world_action(support_domain_ref="domain_mouse", status="unsupported")
    decision = action_support_decision(
        action,
        action_domain=_domain("domain_mouse", organism="mouse", status="unsupported"),
        target_domain=_domain("domain_human"),
    )
    assert decision["decision"] == "unsupported"
    assert decision["allowed_use"] == "unsupported"


def test_round13_builder_preserves_twenty_gaps_and_rejection_examples(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    gate_path = tmp_path / "gate.json"
    handoff_path.write_text(json.dumps(_handoff_fixture()), encoding="utf-8")
    gate_path.write_text(json.dumps({"status": "typed_evidence_relationships_resolved"}), encoding="utf-8")
    payloads = build_round13_pilot(
        handoff_path=handoff_path,
        expected_handoff_sha256=sha256_of_file(handoff_path),
        round12_gate_path=gate_path,
    )
    assert payloads["validation"]["status"] == "passed"
    assert len(payloads["gaps"]) == 20
    graph = payloads["graph"]
    assert [row["validation_status"] for row in graph["assimilation_proposals"]] == ["passed", "rejected", "rejected"]
    assert graph["evidence_fabric"] == {"sealed": False}
    assert graph["dt_core"] == {"status": "not_evaluated"}


def test_formal_validator_stays_blocked_until_execution_verification(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    gate_path = tmp_path / "gate.json"
    focused_path = tmp_path / "focused.xml"
    handoff_path.write_text(json.dumps(_handoff_fixture()), encoding="utf-8")
    gate_path.write_text(
        json.dumps({"status": "typed_evidence_relationships_resolved", "overall_gate_status": "accepted"}),
        encoding="utf-8",
    )
    payloads = build_round13_pilot(
        handoff_path=handoff_path,
        expected_handoff_sha256=sha256_of_file(handoff_path),
        round12_gate_path=gate_path,
    )
    write_round13_artifacts(payloads, output_root=tmp_path)
    testcases = "".join(f'<testcase classname="round13" name="{name}"/>' for name in sorted(MANDATORY_NEGATIVE_TEST_NAMES))
    focused_path.write_text(
        f'<testsuite tests="{len(MANDATORY_NEGATIVE_TEST_NAMES)}" failures="0" errors="0" skipped="0">{testcases}</testsuite>',
        encoding="utf-8",
    )
    report, gate = validate_round13_outputs(
        handoff_path=handoff_path,
        expected_handoff_sha256=sha256_of_file(handoff_path),
        round12_gate_path=gate_path,
        source_output_root=tmp_path,
        focused_junit_path=focused_path,
        prefull_validation_path=None,
        execution_identity_path=None,
        full_suite_junit_path=None,
        full_suite_execution_path=None,
        test_inventory_path=None,
    )
    assert report["mandatory_negative_cases_verified"] is True
    assert report["frozen_gap_count_preserved"] is True
    assert gate["status"] == "blocked_execution_verification"
    assert gate["overall_gate_status"] == "blocked"


def test_lineage_cycle_and_duplicate_uid_fail_closed(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    gate_path = tmp_path / "gate.json"
    handoff_path.write_text(json.dumps(_handoff_fixture()), encoding="utf-8")
    gate_path.write_text(json.dumps({"status": "typed_evidence_relationships_resolved"}), encoding="utf-8")
    payloads = build_round13_pilot(
        handoff_path=handoff_path,
        expected_handoff_sha256=sha256_of_file(handoff_path),
        round12_gate_path=gate_path,
    )
    graph = deepcopy(payloads["graph"])
    graph["lineage_edges"].append(
        {
            "source_uid": graph["assimilation_decisions"][0]["decision_uid"],
            "target_uid": graph["canonical_observations"][0]["observation_uid"],
            "relation_type": "invalid_cycle",
        }
    )
    graph["information_actions"][0]["information_action_uid"] = graph["world_actions"][0]["action_uid"]
    from Libs.utils.data.action_observation_contracts import validate_round13_graph

    report = validate_round13_graph(
        graph,
        canonical_controller_uid=CANONICAL_CONTROLLER_UID,
        approved_validator_uid=VALIDATOR_UID,
        supported_state_components={
            "neuroinflammation_uncertainty",
            "clearance_flux_uncertainty",
            "target_occupancy_parameter",
            "evidence_conflict_register",
        },
    )
    assert {"lineage_cycle", "duplicate_uid"} <= {row["code"] for row in report["blockers"]}


def test_round14_handoff_freezes_complete_world_action_vocabulary(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.json"
    gate_path = tmp_path / "gate.json"
    handoff_path.write_text(json.dumps(_handoff_fixture()), encoding="utf-8")
    gate_path.write_text(json.dumps({"status": "typed_evidence_relationships_resolved"}), encoding="utf-8")
    payloads = build_round13_pilot(
        handoff_path=handoff_path,
        expected_handoff_sha256=sha256_of_file(handoff_path),
        round12_gate_path=gate_path,
    )
    pilot_types = {str(row["intervention_type"]) for row in payloads["graph"]["world_actions"]}
    round14_handoff = build_round14_handoff(
        payloads["graph"],
        payloads["gaps"],
        {
            "status": "action_observation_contracts_resolved",
            "git_commit": "a" * 40,
        },
    )

    assert round14_handoff["action_vocabularies"] == sorted(WORLD_ACTION_TYPES)
    assert pilot_types < set(round14_handoff["action_vocabularies"])
