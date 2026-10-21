"""Build the read-only Round-13 action and observation contract pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.action_observation_contracts import (  # noqa: E402
    SCHEMA_VERSIONS,
    AssimilationProposal,
    HypotheticalActionScenario,
    InformationAction,
    ObservationQC,
    ObservationRequest,
    RawToolResult,
    ToolRequest,
    WorldAction,
    action_support_decision,
    canonical_object_hash,
    compile_nonpharmacological_action,
    compile_pharmacological_action,
    frozen_gap_action_observation_rows,
    match_tool_capability,
    normalize_raw_tool_result,
    round13_graph_uid,
    tool_availability_decision,
    validate_assimilation_proposal,
    validate_frozen_gap_action_observation_map,
    validate_round13_graph,
)
from Libs.utils.data.evidence_relationships import EvidenceSupportDomain  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, sha256_of_file, write_json, write_tsv  # noqa: E402
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_assimilation_decision,
    load_authority_contract,
)


OUTPUT_PATHS = {
    "graph": "Output/audits/dt_round13_action_observation_graph.json",
    "world_actions": "Output/audits/dt_round13_world_action_catalog.tsv",
    "information_actions": "Output/audits/dt_round13_information_action_catalog.tsv",
    "observation_requests": "Output/audits/dt_round13_observation_request_catalog.tsv",
    "tool_matches": "Output/audits/dt_round13_tool_match_catalog.tsv",
    "canonical_observations": "Output/audits/dt_round13_canonical_observation_catalog.tsv",
    "assimilation_proposals": "Output/audits/dt_round13_assimilation_proposal_catalog.tsv",
    "assimilation_decisions": "Output/audits/dt_round13_assimilation_decision_catalog.tsv",
    "gaps": "Output/audits/dt_round13_frozen_gap_action_observation_map.tsv",
}

SCHEMA_PATHS = {
    "world_action": "Libs/configs/schema/world_action.schema.json",
    "information_action": "Libs/configs/schema/information_action.schema.json",
    "observation_request": "Libs/configs/schema/observation_request.schema.json",
    "exposure_occupancy_program": "Libs/configs/schema/exposure_occupancy_program.schema.json",
    "tool_capability_match": "Libs/configs/schema/tool_capability_match.schema.json",
    "raw_tool_result": "Libs/configs/schema/raw_tool_result.schema.json",
    "canonical_observation": "Libs/configs/schema/canonical_observation.schema.json",
    "observation_qc": "Libs/configs/schema/observation_qc.schema.json",
    "assimilation_proposal": "Libs/configs/schema/assimilation_proposal.schema.json",
    "assimilation_decision": "Libs/configs/schema/assimilation_decision.schema.json",
    "hypothetical_action_scenario": "Libs/configs/schema/hypothetical_action_scenario.schema.json",
    "graph": "Libs/configs/schema/round13_action_observation_graph.schema.json",
}

AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
ASSIMILATION_VALIDATOR_UID = "round13_action_observation_contract_validator"
SUPPORTED_STATE_COMPONENTS = {
    "neuroinflammation_uncertainty",
    "clearance_flux_uncertainty",
    "target_occupancy_parameter",
    "evidence_conflict_register",
}


def _json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _world_action(
    uid: str,
    action_type: str,
    *,
    support_domain_ref: str = "domain_human_ad_hippocampus",
    status: str = "pilot_only",
) -> dict[str, object]:
    pharmacological = action_type == "pharmacological"
    no_action = action_type == "no_action_control"
    if action_type == "genetic":
        program = {
            "affected_state_variables": ["misfolded_protein_load"],
            "magnitude_or_program": {"kind": "fractional_expression_change", "value": -0.5},
            "uncertainty": {"kind": "bounded", "lower": -0.7, "upper": -0.3},
        }
    elif no_action:
        program = {}
    else:
        program = {"kind": "not_applicable"}
    return WorldAction(
        schema_version=SCHEMA_VERSIONS["world_action"],
        action_uid=uid,
        intervention_type=action_type,
        target_set=["APP"] if pharmacological or action_type == "genetic" else [],
        agent_set=["lecanemab"] if pharmacological else [],
        dose=10.0 if pharmacological else None,
        dose_unit="mg/kg" if pharmacological else None,
        schedule={"kind": "interval", "interval": 2, "unit": "week"} if pharmacological else {"kind": "not_applicable"},
        route="intravenous" if pharmacological else None,
        duration={"value": 18, "unit": "month"} if pharmacological else {"semantics": "contract_scope"},
        stimulation_or_environment=program,
        state_scope={"region": ["hippocampus"], "compartment": ["isf"]},
        support_domain_ref=support_domain_ref,
        source_evidence_refs=["round12_typed_evidence_relationships"],
        action_provenance_ref="prov_action_bundle",
        conflicts=[],
        status=status,
    ).to_dict()


def _tool_request(uid: str, tool_uid: str) -> dict[str, object]:
    return ToolRequest(
        schema_version="neuronsifter-tool-request-v1",
        tool_request_uid=uid,
        information_action_uid="information_action_measure_inflammation",
        observation_request_uid="observation_request_inflammatory_mediator",
        tool_uid=tool_uid,
        tool_version="1.0.0",
        requested_observables=["inflammatory_mediator"],
        organism_context="human",
        state_action_type="static_state_measurement",
        input_schema_version="neuronsifter-observation-request-v1",
        output_schema_version="pilot-measurement-result-v1",
        fidelity="contract_pilot",
        budget={"maximum_cost": 10.0, "currency": "internal_units"},
        execution_payload={"request_ref": "observation_request_inflammatory_mediator", "mode": "read_only_contract"},
        support_domain_ref="domain_human_ad_hippocampus",
        provenance_ref="prov_tool_registry",
        idempotency_key=f"round13-{uid}",
        sandbox_policy="read_only_adapter",
        conflicts=[],
        status="pilot_only",
    ).to_dict()


def _raw_result(
    uid: str,
    request_uid: str,
    *,
    provenance_ref: str = "prov_tool_registry",
    conflicts: Sequence[str] = (),
) -> dict[str, object]:
    return RawToolResult(
        schema_version=SCHEMA_VERSIONS["raw_tool_result"],
        tool_result_uid=uid,
        tool_uid="pilot_public_measurement_adapter",
        tool_version="1.0.0",
        request_uid=request_uid,
        raw_artifact_hashes={"result": canonical_object_hash({"uid": uid, "value": 0.25})},
        exit_status={"code": 0, "state": "completed"},
        runtime={"seconds": 0.0, "execution": "not_executed_existing_governed_input"},
        resource_usage={"cpu_seconds": 0.0, "gpu_seconds": 0.0},
        raw_schema={"schema_version": "pilot-measurement-result-v1", "validated": True},
        failure_class="none",
        stdout_ref=f"contract://{uid}/stdout-not-applicable",
        stderr_ref=f"contract://{uid}/stderr-not-applicable",
        support_domain_ref="domain_human_ad_hippocampus",
        provenance_ref=provenance_ref,
        conflicts=list(conflicts),
        state_effect="none",
    ).to_dict()


def _qc(uid: str, result_uid: str, grade: str) -> dict[str, object]:
    values = {dimension: "passed" for dimension in (
        "schema_validity",
        "entity_mapping_quality",
        "unit_validity",
        "biological_unit_identity",
        "time_alignment",
        "action_alignment",
        "assay_quality",
        "missingness",
        "outlier_status",
        "source_reliability",
        "tool_reliability",
    )}
    if grade == "C":
        values["source_reliability"] = "partial"
        values["tool_reliability"] = "partial"
    elif grade == "S":
        values["outlier_status"] = "flagged"
        values["source_reliability"] = "flagged"
    return ObservationQC(
        schema_version=SCHEMA_VERSIONS["observation_qc"],
        qc_uid=uid,
        source_result_uid=result_uid,
        qc_grade=grade,
        predecessor_qc_grade=grade,
        verification_artifact_ref=None,
        reason_codes=["directional_only"] if grade == "C" else ["typed_conflict_stress_only"] if grade == "S" else [],
        provenance_ref="prov_tool_registry",
        **values,
    ).to_dict()


def _normalization_mapping(feature_id: str, value: float) -> dict[str, object]:
    return {
        "schema_validated": True,
        "entity_mapping_verified": True,
        "unit_conversion_verified": True,
        "axis_alignment_verified": True,
        "biological_unit_verified": True,
        "provenance_verified": True,
        "biological_unit_uid": "pilot_biological_unit_001",
        "state_time": {"semantics": "static_endpoint", "value": 0.0, "unit": "hour"},
        "region": "hippocampus",
        "compartment": "isf",
        "cell_type": "mixed_cns",
        "modality": "protein_assay",
        "feature_namespace": "neuronsifter_state_species",
        "feature_id": feature_id,
        "value": value,
        "unit": "relative_concentration",
        "uncertainty": {"kind": "interval", "lower": value - 0.05, "upper": value + 0.05},
        "missingness": {"state": "observed"},
    }


def _proposal(
    uid: str,
    observation_uid: str,
    source_kind: str,
    update_type: str,
    *,
    train_eval_role: str = "",
) -> dict[str, object]:
    return AssimilationProposal(
        schema_version=SCHEMA_VERSIONS["assimilation_proposal"],
        proposal_uid=uid,
        observation_uids=[observation_uid],
        target_state_components=["neuroinflammation_uncertainty"],
        prior_state_ref="contract://pilot/prior-state",
        proposed_update_type=update_type,
        likelihood_or_update_model_ref="contract://round13/uncertainty-update-model",
        uncertainty_change={"direction": "increase", "reason": "directional_or_ineligible_input"}
        if update_type == "uncertainty_only_update"
        else {},
        support_domain="domain_human_ad_hippocampus",
        conflicts=[],
        constraint_effects={"state_value_overwrite": False},
        expected_twin_debt_change={"direction": "unchanged"},
        provenance_ref="prov_assimilation_pilot",
        train_eval_roles={observation_uid: train_eval_role} if train_eval_role else {},
        source_kinds={observation_uid: source_kind},
        state_write_authorized=False,
        validation_status="passed",
        blockers=[],
    ).to_dict()


def _record_proposal_validation(
    proposal: dict[str, object],
    *,
    observations: Mapping[str, Mapping[str, object]],
    domains: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    blockers = validate_assimilation_proposal(
        proposal,
        observations=observations,
        support_domains=domains,
        supported_state_components=SUPPORTED_STATE_COMPONENTS,
    )
    proposal["validation_status"] = "passed" if not blockers else "rejected"
    proposal["blockers"] = blockers
    return proposal


def _schema_blockers(graph: Mapping[str, object], repository_root: Path) -> list[dict[str, object]]:
    families = {
        "world_action": graph["world_actions"],
        "information_action": graph["information_actions"],
        "observation_request": graph["observation_requests"],
        "exposure_occupancy_program": graph["exposure_occupancy_programs"],
        "tool_capability_match": graph["tool_capability_matches"],
        "raw_tool_result": graph["raw_tool_results"],
        "canonical_observation": graph["canonical_observations"],
        "observation_qc": graph["observation_qc"],
        "assimilation_proposal": graph["assimilation_proposals"],
        "assimilation_decision": graph["assimilation_decisions"],
        "hypothetical_action_scenario": graph["hypothetical_scenarios"],
        "graph": [graph],
    }
    blockers: list[dict[str, object]] = []
    for family, values in families.items():
        schema = json.loads((repository_root / SCHEMA_PATHS[family]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for index, value in enumerate(values):
            for error in validator.iter_errors(value):
                blockers.append(
                    {
                        "code": "json_schema_validation",
                        "message": error.message,
                        "schema_family": family,
                        "index": index,
                        "path": list(error.path),
                    }
                )
    return blockers


def build_round13_pilot(
    *,
    handoff_path: Path,
    expected_handoff_sha256: str,
    round12_gate_path: Path,
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    observed_handoff_hash = sha256_of_file(handoff_path)
    if observed_handoff_hash != expected_handoff_sha256:
        raise ValueError("Round-13 handoff hash does not match the authorized input.")
    handoff = _load_json(handoff_path)
    round12_gate = _load_json(round12_gate_path)
    if handoff.get("status") != "ready_for_structured_world_action_contracts":
        raise ValueError("Round-13 handoff is not ready.")
    if round12_gate.get("status") != "typed_evidence_relationships_resolved":
        raise ValueError("Round-12 Gate is not accepted.")
    if handoff.get("evidence_fabric") != {"sealed": False} or handoff.get("dt_core") != {"status": "not_evaluated"}:
        raise ValueError("Round-13 starting boundary has drifted.")

    config = load_config(repository_root / "Libs/configs/config_stack.yaml")
    adapters = config["digital_twin_evidence_fabric"]["source_adapters"]
    if sum(row.get("adapter_id") == "tool_registry" for row in adapters) != 1:
        raise ValueError("Existing Tool Capability Registry adapter binding is not canonical.")

    domains = [dict(row) for row in handoff["support_domain_rules"]]
    domains.append(
        EvidenceSupportDomain(
            schema_version="neuronsifter-evidence-support-domain-v1",
            support_domain_uid="domain_mouse_ad_hippocampus_unsupported",
            organism_scope=["mouse"],
            disease_scope=["ad_model"],
            region_scope=["hippocampus"],
            compartment_scope=["isf"],
            cell_type_scope=["mixed_cns"],
            modality_scope=["not_applicable"],
            assay_scope=["not_applicable"],
            action_scope=["pharmacological"],
            dose_scope=["single_dose"],
            route_scope=["intravenous"],
            time_scope=["static_endpoint"],
            model_family_scope=["not_applicable"],
            tool_family_scope=["not_applicable"],
            source_scope=["round13_contract_pilot"],
            generalization_status="unsupported",
            provenance_ref="prov_action_bundle",
            status="blocked",
        ).to_dict()
    )
    domain_map = {str(row["support_domain_uid"]): row for row in domains}
    conflicts = [dict(row) for row in handoff["conflict_semantics"]]
    conflict_uid = str(conflicts[0]["conflict_uid"])

    world_actions = [
        _world_action("world_action_lecanemab_program", "pharmacological"),
        _world_action("world_action_no_action_control", "no_action_control"),
        _world_action("world_action_genetic_app_reduction", "genetic"),
        _world_action(
            "world_action_mouse_parameter_out_of_domain",
            "pharmacological",
            support_domain_ref="domain_mouse_ad_hippocampus_unsupported",
            status="unsupported",
        ),
    ]
    exposure_program = compile_pharmacological_action(
        world_actions[0],
        pkpd_context={
            "time_grid": [0.0, 24.0, 168.0, 336.0],
            "parameter_provenance": {"status": "verified", "artifact_ref": "repo://Input/processed/pkpd_ready_registry.tsv"},
            "target_program": {"targets": ["APP"], "semantics": "provided_contract_program"},
            "exposure_fields": {"free_drug": [0.0, 1.0, 0.4, 0.2], "unit": "relative_exposure"},
            "occupancy_fields": {"APP": [0.0, 0.7, 0.5, 0.3], "unit": "fraction"},
            "off_target_context": {"status": "not_characterized"},
            "uncertainty": {"kind": "provided_interval", "relative_width": 0.2},
            "organism_context": "human",
            "supported_organisms": ["human"],
        },
        baseline_state_posterior={"state_ref": "contract://pilot/prior-state"},
        graph_context={"graph_ref": "contract://ad-mesoscale-graph-v1"},
    )
    no_action_field, no_action_blockers = compile_nonpharmacological_action(world_actions[1])
    genetic_field, genetic_blockers = compile_nonpharmacological_action(world_actions[2])
    if no_action_blockers or genetic_blockers or no_action_field is None or genetic_field is None:
        raise ValueError("Canonical non-pharmacological pilot adapters failed.")
    action_fields = [no_action_field, genetic_field]
    action_support_decisions = [
        action_support_decision(
            world_actions[0],
            action_domain=domain_map["domain_human_ad_hippocampus"],
            target_domain=domain_map["domain_human_ad_hippocampus"],
        ),
        action_support_decision(
            world_actions[3],
            action_domain=domain_map["domain_mouse_ad_hippocampus_unsupported"],
            target_domain=domain_map["domain_human_ad_hippocampus"],
        ),
    ]

    information_actions = [
        InformationAction(
            schema_version=SCHEMA_VERSIONS["information_action"],
            information_action_uid="information_action_measure_inflammation",
            candidate_uids=[world_actions[0]["action_uid"]],
            requested_observables=["inflammatory_mediator"],
            tool_or_source_uid="pilot_public_measurement_adapter",
            fidelity="contract_pilot",
            measurement_context={"organism": "human", "region": "hippocampus", "compartment": "isf"},
            budget_class="low",
            expected_cost={"value": 5.0, "unit": "internal_units"},
            expected_runtime={"value": 0.0, "unit": "second", "execution": "not_run"},
            failure_policy="fail_closed",
            support_domain_ref="domain_human_ad_hippocampus",
            provenance_ref="prov_tool_registry",
            conflicts=[],
            status="pilot_only",
        ).to_dict()
    ]
    observation_requests = [
        ObservationRequest(
            schema_version=SCHEMA_VERSIONS["observation_request"],
            request_uid="observation_request_inflammatory_mediator",
            research_question_ref="contract://round13/inflammation-observation",
            requested_variable_namespace="neuronsifter_state_species",
            requested_feature_ids=["inflammatory_mediator"],
            region="hippocampus",
            compartment="isf",
            cell_type="mixed_cns",
            biological_unit_scope="sample_level_public_evidence",
            timepoint_or_time_window={"semantics": "static_endpoint", "value": 0.0, "unit": "hour"},
            assay_or_measurement_type="protein_assay",
            required_units=["relative_concentration"],
            minimum_quality="C",
            minimum_fidelity="contract_pilot",
            uncertainty_requirement={"required": True, "kind": "interval"},
            budget={"maximum_cost": 10.0, "currency": "internal_units"},
            tool_availability_constraints={"license_required": True, "known_version_required": True},
            support_domain_ref="domain_human_ad_hippocampus",
            provenance_ref="prov_tool_registry",
            status="pilot_only",
        ).to_dict()
    ]
    tool_requests = [
        _tool_request("tool_request_supported_measurement", "pilot_public_measurement_adapter"),
        _tool_request("tool_request_unsupported_measurement", "unregistered_measurement_adapter"),
    ]
    capability = {
        "tool_uid": "pilot_public_measurement_adapter",
        "tool_version": "1.0.0",
        "supported_observables": ["inflammatory_mediator"],
        "organism_scope": ["human"],
        "state_action_types": ["static_state_measurement"],
        "input_schema_versions": ["neuronsifter-observation-request-v1"],
        "output_schema_versions": ["pilot-measurement-result-v1"],
        "fidelities": ["contract_pilot"],
        "maximum_cost": 5.0,
        "license_status": "resolved",
        "availability": "available",
        "restrictions": [],
        "support_domain_ref": "domain_human_ad_hippocampus",
        "provenance_ref": "prov_tool_registry",
        "conflicts": [],
        "registry_adapter_id": "tool_registry",
    }
    matches = [match_tool_capability(request, capability) for request in tool_requests]
    availability = [tool_availability_decision(match) for match in matches]

    raw_results = [
        _raw_result("raw_result_directional", tool_requests[0]["tool_request_uid"]),
        _raw_result("raw_result_conflict", tool_requests[0]["tool_request_uid"], conflicts=[conflict_uid]),
        _raw_result("raw_result_unknown_provenance", tool_requests[0]["tool_request_uid"], provenance_ref="unknown"),
    ]
    qcs = [
        _qc("qc_directional", raw_results[0]["tool_result_uid"], "C"),
        _qc("qc_conflict", raw_results[1]["tool_result_uid"], "S"),
        _qc("qc_unknown_provenance", raw_results[2]["tool_result_uid"], "B"),
    ]
    directional_observation, directional_blockers = normalize_raw_tool_result(
        raw_results[0],
        mapping=_normalization_mapping("inflammatory_mediator", 0.25),
        qc_value=qcs[0],
        provenance_status="verified",
    )
    conflict_observation, conflict_blockers = normalize_raw_tool_result(
        raw_results[1],
        mapping=_normalization_mapping("inflammatory_mediator", 0.75),
        qc_value=qcs[1],
        provenance_status="verified",
    )
    unknown_observation, unknown_blockers = normalize_raw_tool_result(
        raw_results[2],
        mapping=_normalization_mapping("inflammatory_mediator", 0.5),
        qc_value=qcs[2],
        provenance_status="unknown",
    )
    if directional_blockers or conflict_blockers or directional_observation is None or conflict_observation is None:
        raise ValueError("Governed RawToolResult normalization failed.")
    if unknown_observation is not None or not any(row["code"] == "unknown_observation_provenance" for row in unknown_blockers):
        raise ValueError("Unknown-provenance normalization did not fail closed.")
    observations = [directional_observation, conflict_observation]
    observation_map = {str(row["observation_uid"]): row for row in observations}

    scenario = HypotheticalActionScenario(
        schema_version=SCHEMA_VERSIONS["hypothetical_action_scenario"],
        scenario_uid="hypothetical_scenario_lecanemab_counterfactual",
        baseline_state_ref="contract://pilot/prior-state",
        candidate_world_action=world_actions[0],
        assumptions=["contract_only_no_what_if_execution"],
        scenario_context={"purpose": "future_what_if_input_contract"},
        requested_outputs=["uncertainty_summary"],
        support_warnings=["not_empirical_evidence", "not_assimilation_eligible"],
        support_domain_ref="domain_hypothetical_unsupported",
        provenance_ref="prov_action_bundle",
        model_based_counterfactual=True,
        causal_effect_claim=False,
        clinical_recommendation=False,
        empirical_observation=False,
    ).to_dict()

    valid_proposal = _proposal(
        "assimilation_proposal_uncertainty_only",
        str(directional_observation["observation_uid"]),
        "canonical_observation",
        "uncertainty_only_update",
        train_eval_role=str(directional_observation["train_eval_role"]),
    )
    hypothetical_proposal = _proposal(
        "assimilation_proposal_rejected_hypothetical",
        str(scenario["scenario_uid"]),
        "hypothetical_scenario",
        "no_update",
    )
    unknown_proposal = _proposal(
        "assimilation_proposal_rejected_unknown_provenance",
        str(raw_results[2]["tool_result_uid"]),
        "raw_tool_result_unknown_provenance",
        "no_update",
    )
    proposals = [
        _record_proposal_validation(valid_proposal, observations=observation_map, domains=domain_map),
        _record_proposal_validation(hypothetical_proposal, observations=observation_map, domains=domain_map),
        _record_proposal_validation(unknown_proposal, observations=observation_map, domains=domain_map),
    ]
    if [row["validation_status"] for row in proposals] != ["passed", "rejected", "rejected"]:
        raise ValueError("Assimilation pilot rejection taxonomy did not fail closed.")

    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    decisions = [
        build_canonical_assimilation_decision(
            authority,
            proposal_uid=str(proposals[0]["proposal_uid"]),
            controller_uid=CANONICAL_CONTROLLER_UID,
            validator_uid=ASSIMILATION_VALIDATOR_UID,
            validator_report={"validator_uid": ASSIMILATION_VALIDATOR_UID, "status": "passed"},
            prior_state_hash="b" * 64,
            input_observation_hashes={str(directional_observation["observation_uid"]): canonical_object_hash(directional_observation)},
            decision="approved_uncertainty_only",
            reason_codes=["qc_c_directional_uncertainty_only"],
            approved_update_scope=["neuroinflammation_uncertainty"],
            conflict_handling={},
            resulting_state_ref="contract://pilot/resulting-state-not-materialized",
            provenance_ref="prov_assimilation_pilot",
        ),
        build_canonical_assimilation_decision(
            authority,
            proposal_uid=str(proposals[1]["proposal_uid"]),
            controller_uid=CANONICAL_CONTROLLER_UID,
            validator_uid=ASSIMILATION_VALIDATOR_UID,
            validator_report={"validator_uid": ASSIMILATION_VALIDATOR_UID, "status": "passed"},
            prior_state_hash="b" * 64,
            input_observation_hashes={str(scenario["scenario_uid"]): canonical_object_hash(scenario)},
            decision="insufficient_evidence",
            reason_codes=["hypothetical_scenario_not_assimilation_eligible"],
            approved_update_scope=[],
            conflict_handling={},
            resulting_state_ref=None,
            provenance_ref="prov_assimilation_pilot",
        ),
        build_canonical_assimilation_decision(
            authority,
            proposal_uid=str(proposals[2]["proposal_uid"]),
            controller_uid=CANONICAL_CONTROLLER_UID,
            validator_uid=ASSIMILATION_VALIDATOR_UID,
            validator_report={"validator_uid": ASSIMILATION_VALIDATOR_UID, "status": "passed"},
            prior_state_hash="b" * 64,
            input_observation_hashes={str(raw_results[2]["tool_result_uid"]): canonical_object_hash(raw_results[2])},
            decision="rejected_provenance",
            reason_codes=["unknown_provenance"],
            approved_update_scope=[],
            conflict_handling={},
            resulting_state_ref=None,
            provenance_ref="prov_assimilation_pilot",
        ),
    ]

    gaps = frozen_gap_action_observation_rows(handoff)
    gap_blockers = validate_frozen_gap_action_observation_map(handoff, gaps)
    graph_without_uid = {
        "schema_version": SCHEMA_VERSIONS["graph"],
        "contract_scope": "round13_contract_validation_pilot_only",
        "source_handoff": {
            "canonical_uri": "repo://Output/audits/dt_round13_handoff.json",
            "sha256": observed_handoff_hash,
            "parent_gate": "typed_evidence_relationships_resolved",
        },
        "support_domains": domains,
        "conflicts": conflicts,
        "world_actions": world_actions,
        "information_actions": information_actions,
        "observation_requests": observation_requests,
        "tool_requests": tool_requests,
        "tool_capability_matches": matches,
        "tool_availability_decisions": availability,
        "raw_tool_results": raw_results,
        "observation_qc": qcs,
        "canonical_observations": observations,
        "exposure_occupancy_programs": [exposure_program],
        "canonical_action_fields": action_fields,
        "action_support_decisions": action_support_decisions,
        "assimilation_proposals": proposals,
        "assimilation_decisions": decisions,
        "hypothetical_scenarios": [scenario],
        "lineage_edges": [
            {"source_uid": world_actions[0]["action_uid"], "target_uid": exposure_program["program_uid"], "relation_type": "compiled_to"},
            {"source_uid": information_actions[0]["information_action_uid"], "target_uid": tool_requests[0]["tool_request_uid"], "relation_type": "requests"},
            {"source_uid": observation_requests[0]["request_uid"], "target_uid": tool_requests[0]["tool_request_uid"], "relation_type": "matched_to_request"},
            {"source_uid": tool_requests[0]["tool_request_uid"], "target_uid": raw_results[0]["tool_result_uid"], "relation_type": "produced_raw_result"},
            {"source_uid": raw_results[0]["tool_result_uid"], "target_uid": directional_observation["observation_uid"], "relation_type": "normalized_to"},
            {"source_uid": directional_observation["observation_uid"], "target_uid": proposals[0]["proposal_uid"], "relation_type": "proposed_for_assimilation"},
            {"source_uid": proposals[0]["proposal_uid"], "target_uid": decisions[0]["decision_uid"], "relation_type": "decided_by_controller"},
            {"source_uid": scenario["scenario_uid"], "target_uid": proposals[1]["proposal_uid"], "relation_type": "rejected_as_assimilation_input"},
            {"source_uid": raw_results[2]["tool_result_uid"], "target_uid": proposals[2]["proposal_uid"], "relation_type": "rejected_as_assimilation_input"},
        ],
        "frozen_gap_map": gaps,
        "invariants": {
            "world_action_may_change_simulated_state": True,
            "information_action_may_reveal_observation_only": True,
            "information_action_direct_state_effect": False,
            "tool_result_direct_state_effect": False,
            "llm_proposal_direct_state_effect": False,
            "assimilation_decision_required_for_posterior_update": True,
        },
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }
    graph = {"graph_uid": round13_graph_uid(graph_without_uid), **graph_without_uid}
    validation = validate_round13_graph(
        graph,
        canonical_controller_uid=CANONICAL_CONTROLLER_UID,
        approved_validator_uid=ASSIMILATION_VALIDATOR_UID,
        supported_state_components=SUPPORTED_STATE_COMPONENTS,
    )
    blockers = [*gap_blockers, *_schema_blockers(graph, repository_root), *validation["blockers"]]
    blockers = sorted(blockers, key=lambda row: (str(row.get("code", "")), json.dumps(row, sort_keys=True)))
    if blockers:
        raise ValueError("Round-13 pilot graph failed validation: " + _json_cell(blockers))
    return {
        "graph": graph,
        "gaps": gaps,
        "unknown_provenance_normalization_blockers": unknown_blockers,
        "validation": validation,
    }


def _catalog_rows(graph: Mapping[str, object]) -> dict[str, tuple[list[str], list[dict[str, object]]]]:
    action_support = {str(row["action_uid"]): row for row in graph["action_support_decisions"]}
    programs = {str(row["action_uid"]): row for row in graph["exposure_occupancy_programs"]}
    world_rows = []
    for row in graph["world_actions"]:
        value = dict(row)
        value["support_decision"] = action_support.get(str(row["action_uid"]), {}).get("decision", "not_evaluated")
        value["compiler_status"] = programs.get(str(row["action_uid"]), {}).get("status", "not_applicable")
        world_rows.append(value)
    tool_requests = {str(row["tool_request_uid"]): row for row in graph["tool_requests"]}
    availability = {str(row["match_uid"]): row for row in graph["tool_availability_decisions"]}
    tool_rows = []
    for match in graph["tool_capability_matches"]:
        value = dict(match)
        request = tool_requests[str(match["tool_request_uid"])]
        value["requested_tool_uid"] = request["tool_uid"]
        value["availability_decision"] = availability[str(match["match_uid"])]["state"]
        tool_rows.append(value)
    return {
        "world_actions": (
            [*graph["world_actions"][0].keys(), "support_decision", "compiler_status"],
            world_rows,
        ),
        "information_actions": (list(graph["information_actions"][0].keys()), [dict(row) for row in graph["information_actions"]]),
        "observation_requests": (list(graph["observation_requests"][0].keys()), [dict(row) for row in graph["observation_requests"]]),
        "tool_matches": ([*graph["tool_capability_matches"][0].keys(), "requested_tool_uid", "availability_decision"], tool_rows),
        "canonical_observations": (list(graph["canonical_observations"][0].keys()), [dict(row) for row in graph["canonical_observations"]]),
        "assimilation_proposals": (list(graph["assimilation_proposals"][0].keys()), [dict(row) for row in graph["assimilation_proposals"]]),
        "assimilation_decisions": (list(graph["assimilation_decisions"][0].keys()), [dict(row) for row in graph["assimilation_decisions"]]),
    }


def _tsv_ready(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {key: _json_cell(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
        for row in rows
    ]


def write_round13_artifacts(payloads: Mapping[str, object], *, output_root: Path) -> None:
    graph = dict(payloads["graph"])
    write_json(output_root / OUTPUT_PATHS["graph"], graph)
    for family, (fields, rows) in _catalog_rows(graph).items():
        write_tsv(output_root / OUTPUT_PATHS[family], fields, _tsv_ready(rows))
    gap_fields = [
        "gap_uid",
        "required_action_type",
        "required_observation_type",
        "missing_identity_axis",
        "missing_support_domain",
        "missing_tool_capability",
        "missing_assimilation_prerequisite",
        "current_status",
        "downstream_blocked_capability",
    ]
    write_tsv(output_root / OUTPUT_PATHS["gaps"], gap_fields, [dict(row) for row in payloads["gaps"]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Round-13 action and observation contract pilot.")
    parser.add_argument("--round13-handoff", required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--round12-gate", required=True)
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    args = parser.parse_args()
    payloads = build_round13_pilot(
        handoff_path=Path(args.round13_handoff),
        expected_handoff_sha256=args.expected_handoff_sha256,
        round12_gate_path=Path(args.round12_gate),
    )
    write_round13_artifacts(payloads, output_root=Path(args.output_root))
    counts = payloads["validation"]["object_counts"]
    print(f"round13_contract_pilot_passed gaps={len(payloads['gaps'])} objects={sum(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
