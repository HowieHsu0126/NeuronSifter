"""Typed, fail-closed Round-13 action and observation contracts.

This module validates contract objects only. It does not execute tools, mutate a
Twin, run counterfactuals, infer PK/PD parameters, or materialize evidence.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

from Libs.utils.data.evidence_relationships import (
    CONFLICT_STATES,
    EvidenceSupportDomain,
    canonical_json,
    sorted_blockers,
    stable_uid,
    validate_conflict,
    validate_support_domain,
)


SCHEMA_VERSIONS = {
    "world_action": "neuronsifter-world-action-v1",
    "information_action": "neuronsifter-information-action-v1",
    "observation_request": "neuronsifter-observation-request-v1",
    "exposure_occupancy_program": "neuronsifter-exposure-occupancy-program-v1",
    "tool_capability_match": "neuronsifter-tool-capability-match-v1",
    "raw_tool_result": "neuronsifter-raw-tool-result-v1",
    "canonical_observation": "neuronsifter-canonical-observation-v1",
    "observation_qc": "neuronsifter-observation-qc-v1",
    "assimilation_proposal": "neuronsifter-assimilation-proposal-v1",
    "assimilation_decision": "neuronsifter-assimilation-decision-v1",
    "hypothetical_action_scenario": "neuronsifter-hypothetical-action-scenario-v1",
    "graph": "neuronsifter-round13-action-observation-graph-v1",
    "validation": "neuronsifter-dt-round13-validation-v1",
    "gate": "neuronsifter-dt-round13-gate-v1",
}

WORLD_ACTION_TYPES = frozenset(
    {
        "pharmacological",
        "genetic",
        "electrical_stimulation",
        "optical_stimulation",
        "environmental",
        "state_initialization",
        "no_action_control",
        "combination",
    }
)
NON_PHARMACOLOGICAL_ACTION_TYPES = frozenset(
    {
        "genetic",
        "electrical_stimulation",
        "optical_stimulation",
        "environmental",
        "state_initialization",
        "no_action_control",
    }
)
KNOWN_DOSE_UNITS = frozenset({"mg", "ug", "mg/kg", "ug/kg", "mmol/L", "umol/L", "nmol/L"})
SUPPORTED_ROUTES = frozenset(
    {"oral", "intravenous", "subcutaneous", "intrathecal", "intracerebroventricular", "local_cns"}
)
SCHEDULE_KINDS = frozenset({"single", "interval", "timepoints", "continuous", "program", "not_applicable"})
COMPILER_STATUSES = frozenset(
    {"compiled", "compiled_with_explicit_uncertainty", "insufficient_parameters", "unsupported_action", "invalid_action"}
)
TOOL_MATCH_STATES = frozenset(
    {
        "supported",
        "supported_with_restrictions",
        "temporarily_unavailable",
        "unsupported_schema",
        "unsupported_domain",
        "unresolved_license",
        "budget_exceeded",
    }
)
QC_GRADES = frozenset({"A", "B", "C", "D", "S"})
QC_DIMENSIONS = (
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
)
QC_DIMENSION_STATES = frozenset({"passed", "partial", "failed", "flagged", "not_applicable"})
QC_GRADE_RANK = {"D": 0, "S": 1, "C": 2, "B": 3, "A": 4}
ASSIMILATION_UPDATE_TYPES = frozenset(
    {
        "state_posterior_update",
        "parameter_posterior_update",
        "model_family_weight_update",
        "uncertainty_only_update",
        "support_domain_update",
        "conflict_register_only",
        "no_update",
    }
)
ASSIMILATION_DECISIONS = frozenset(
    {
        "approved",
        "approved_uncertainty_only",
        "approved_conflict_register_only",
        "revise_required",
        "insufficient_evidence",
        "unsupported_domain",
        "rejected_qc",
        "rejected_provenance",
        "rejected_conflict",
    }
)
OUT_OF_DOMAIN_RETENTION_STATES = frozenset({"stress_only", "hypothesis_only", "unsupported"})
CONFLICT_RESOLUTION_STATES = frozenset(
    {"context_explained", "version_resolved", "invalidated", "superseded", "retained_for_stress"}
)
PROHIBITED_EXECUTION_KEYS = frozenset(
    {"command", "commands", "shell", "shell_command", "argv", "script", "executable", "subprocess"}
)
UNSUPPORTED_CLINICAL_FEATURES = frozenset(
    {"clinical_outcome", "clinical_recommendation", "survival", "cognitive_decline_prediction", "treatment_response"}
)
STATIC_TIME_SEMANTICS = frozenset({"static", "static_endpoint", "cross_sectional", "timeless"})
TEMPORAL_STATE_COMPONENT_MARKERS = ("trajectory", "dynamics", "time_series", "rollout", "future_state")


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _blocker(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = _text(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _required_text_fields(value: Mapping[str, object], fields: Sequence[str], object_uid: str) -> list[dict[str, object]]:
    return [
        _blocker("missing_required_field", f"{field} must be explicit.", object_uid=object_uid, field=field)
        for field in fields
        if not _text(value.get(field))
    ]


@dataclass(frozen=True)
class WorldAction:
    schema_version: str
    action_uid: str
    intervention_type: str
    target_set: Sequence[str]
    agent_set: Sequence[str]
    dose: float | None
    dose_unit: str | None
    schedule: Mapping[str, object]
    route: str | None
    duration: Mapping[str, object]
    stimulation_or_environment: Mapping[str, object]
    state_scope: Mapping[str, object]
    support_domain_ref: str
    source_evidence_refs: Sequence[str]
    action_provenance_ref: str
    conflicts: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InformationAction:
    schema_version: str
    information_action_uid: str
    candidate_uids: Sequence[str]
    requested_observables: Sequence[str]
    tool_or_source_uid: str
    fidelity: str
    measurement_context: Mapping[str, object]
    budget_class: str
    expected_cost: Mapping[str, object]
    expected_runtime: Mapping[str, object]
    failure_policy: str
    support_domain_ref: str
    provenance_ref: str
    conflicts: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationRequest:
    schema_version: str
    request_uid: str
    research_question_ref: str
    requested_variable_namespace: str
    requested_feature_ids: Sequence[str]
    region: str
    compartment: str
    cell_type: str
    biological_unit_scope: str
    timepoint_or_time_window: Mapping[str, object]
    assay_or_measurement_type: str
    required_units: Sequence[str]
    minimum_quality: str
    minimum_fidelity: str
    uncertainty_requirement: Mapping[str, object]
    budget: Mapping[str, object]
    tool_availability_constraints: Mapping[str, object]
    support_domain_ref: str
    provenance_ref: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ToolRequest:
    schema_version: str
    tool_request_uid: str
    information_action_uid: str
    observation_request_uid: str
    tool_uid: str
    tool_version: str
    requested_observables: Sequence[str]
    organism_context: str
    state_action_type: str
    input_schema_version: str
    output_schema_version: str
    fidelity: str
    budget: Mapping[str, object]
    execution_payload: Mapping[str, object]
    support_domain_ref: str
    provenance_ref: str
    idempotency_key: str
    sandbox_policy: str
    conflicts: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCapabilityMatch:
    schema_version: str
    match_uid: str
    tool_request_uid: str
    tool_uid: str
    tool_version: str
    observable_supported: bool
    organism_context_supported: bool
    state_action_type_supported: bool
    input_schema_compatible: bool
    output_schema_compatible: bool
    fidelity_available: bool
    budget_sufficient: bool
    license_access_resolved: bool
    state: str
    restrictions: Sequence[str]
    support_domain_ref: str
    provenance_ref: str
    conflicts: Sequence[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ToolAvailabilityDecision:
    decision_uid: str
    match_uid: str
    state: str
    available: bool
    reason_codes: Sequence[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ToolRequestValidation:
    validation_uid: str
    tool_request_uid: str
    status: str
    blockers: Sequence[Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RawToolResult:
    schema_version: str
    tool_result_uid: str
    tool_uid: str
    tool_version: str
    request_uid: str
    raw_artifact_hashes: Mapping[str, str]
    exit_status: Mapping[str, object]
    runtime: Mapping[str, object]
    resource_usage: Mapping[str, object]
    raw_schema: Mapping[str, object]
    failure_class: str
    stdout_ref: str
    stderr_ref: str
    support_domain_ref: str
    provenance_ref: str
    conflicts: Sequence[str]
    state_effect: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationQC:
    schema_version: str
    qc_uid: str
    source_result_uid: str
    schema_validity: str
    entity_mapping_quality: str
    unit_validity: str
    biological_unit_identity: str
    time_alignment: str
    action_alignment: str
    assay_quality: str
    missingness: str
    outlier_status: str
    source_reliability: str
    tool_reliability: str
    qc_grade: str
    predecessor_qc_grade: str | None
    verification_artifact_ref: str | None
    reason_codes: Sequence[str]
    provenance_ref: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalObservation:
    schema_version: str
    observation_uid: str
    source_result_uid: str
    biological_unit_uid: str
    state_time: Mapping[str, object]
    region: str
    compartment: str
    cell_type: str
    modality: str
    feature_namespace: str
    feature_id: str
    value: object
    unit: str
    uncertainty: Mapping[str, object]
    missingness: Mapping[str, object]
    qc_grade: str
    support_domain_ref: str
    provenance_ref: str
    train_eval_role: str
    conflicts: Sequence[str]
    empirical_observation: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureOccupancyProgram:
    schema_version: str
    program_uid: str
    action_uid: str
    time_grid: Sequence[float]
    exposure_fields: Mapping[str, object]
    occupancy_fields: Mapping[str, object]
    target_program: Mapping[str, object]
    off_target_context: Mapping[str, object]
    parameter_provenance: Mapping[str, object]
    uncertainty: Mapping[str, object]
    support_domain: str
    compiler_version: str
    status: str
    blockers: Sequence[Mapping[str, object]]
    conflicts: Sequence[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalActionField:
    action_field_uid: str
    action_uid: str
    intervention_type: str
    affected_state_variables: Sequence[str]
    spatial_scope: Mapping[str, object]
    temporal_scope: Mapping[str, object]
    magnitude_or_program: Mapping[str, object]
    uncertainty: Mapping[str, object]
    provenance_ref: str
    support_domain_ref: str
    conflicts: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSupportDecision:
    decision_uid: str
    action_uid: str
    support_domain_ref: str
    decision: str
    reason_codes: Sequence[str]
    allowed_use: str
    provenance_ref: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AssimilationProposal:
    schema_version: str
    proposal_uid: str
    observation_uids: Sequence[str]
    target_state_components: Sequence[str]
    prior_state_ref: str
    proposed_update_type: str
    likelihood_or_update_model_ref: str
    uncertainty_change: Mapping[str, object]
    support_domain: str
    conflicts: Sequence[str]
    constraint_effects: Mapping[str, object]
    expected_twin_debt_change: Mapping[str, object]
    provenance_ref: str
    train_eval_roles: Mapping[str, str]
    source_kinds: Mapping[str, str]
    state_write_authorized: bool
    validation_status: str
    blockers: Sequence[Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AssimilationDecision:
    schema_version: str
    decision_uid: str
    proposal_uid: str
    controller_uid: str
    validator_uid: str
    prior_state_hash: str
    input_observation_hashes: Mapping[str, str]
    decision: str
    reason_codes: Sequence[str]
    approved_update_scope: Sequence[str]
    conflict_handling: Mapping[str, str]
    resulting_state_ref: str | None
    provenance_ref: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HypotheticalActionScenario:
    schema_version: str
    scenario_uid: str
    baseline_state_ref: str
    candidate_world_action: Mapping[str, object]
    assumptions: Sequence[str]
    scenario_context: Mapping[str, object]
    requested_outputs: Sequence[str]
    support_warnings: Sequence[str]
    support_domain_ref: str
    provenance_ref: str
    model_based_counterfactual: bool
    causal_effect_claim: bool
    clinical_recommendation: bool
    empirical_observation: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_schedule(schedule_value: object, action_uid: str) -> list[dict[str, object]]:
    schedule = _mapping(schedule_value)
    kind = _text(schedule.get("kind"))
    blockers: list[dict[str, object]] = []
    if kind not in SCHEDULE_KINDS:
        blockers.append(_blocker("invalid_schedule", "Schedule kind is missing or unsupported.", action_uid=action_uid))
    elif kind == "interval":
        if not _finite_number(schedule.get("interval")) or float(schedule["interval"]) <= 0 or not _text(schedule.get("unit")):
            blockers.append(_blocker("invalid_schedule", "Interval schedule requires a positive interval and unit.", action_uid=action_uid))
    elif kind == "timepoints":
        values = _list(schedule.get("values"))
        if not values or any(not _finite_number(value) for value in values):
            blockers.append(_blocker("invalid_schedule", "Timepoint schedule requires finite explicit values.", action_uid=action_uid))
    elif kind == "program" and not _text(schedule.get("program_ref")):
        blockers.append(_blocker("invalid_schedule", "Program schedule requires program_ref.", action_uid=action_uid))
    return blockers


def validate_world_action(value: Mapping[str, object]) -> list[dict[str, object]]:
    action = dict(value)
    uid = _text(action.get("action_uid"))
    action_type = _text(action.get("intervention_type"))
    blockers = _required_text_fields(
        action,
        ("action_uid", "intervention_type", "support_domain_ref", "action_provenance_ref", "status"),
        uid,
    )
    if action.get("schema_version") != SCHEMA_VERSIONS["world_action"]:
        blockers.append(_blocker("unknown_schema_version", "WorldAction schema is not canonical.", action_uid=uid))
    if action_type not in WORLD_ACTION_TYPES:
        blockers.append(_blocker("unknown_intervention_type", "WorldAction intervention type is unsupported.", action_uid=uid))
    forbidden = sorted(set(action) & {"tool_or_source_uid", "requested_observables", "expected_cost", "tool_request_uid"})
    if forbidden:
        blockers.append(
            _blocker(
                "world_information_action_not_separated",
                "A tool or evidence-acquisition request cannot be represented as a WorldAction.",
                action_uid=uid,
                fields=forbidden,
            )
        )
    if action_type in {"pharmacological", "combination"}:
        if not _list(action.get("agent_set")):
            blockers.append(_blocker("drug_id_not_complete_action", "Pharmacological actions require an explicit agent set.", action_uid=uid))
        if not _list(action.get("target_set")):
            blockers.append(_blocker("unresolved_target_identity", "Pharmacological actions require resolved targets.", action_uid=uid))
        if not _finite_number(action.get("dose")) or float(action.get("dose", 0)) <= 0:
            blockers.append(_blocker("missing_dose", "Pharmacological actions require a positive dose.", action_uid=uid))
        unit = _text(action.get("dose_unit"))
        if not unit:
            blockers.append(_blocker("missing_dose_unit", "Pharmacological actions require a dose unit.", action_uid=uid))
        elif unit not in KNOWN_DOSE_UNITS:
            blockers.append(_blocker("unknown_dose_unit", "Dose unit is not in the canonical unit vocabulary.", action_uid=uid, dose_unit=unit))
        route = _text(action.get("route"))
        if route not in SUPPORTED_ROUTES:
            blockers.append(_blocker("unsupported_route", "Pharmacological route is unsupported.", action_uid=uid, route=route))
        blockers.extend(validate_schedule(action.get("schedule"), uid))
    elif action_type == "no_action_control":
        if action.get("dose") is not None or action.get("dose_unit") is not None or _list(action.get("agent_set")):
            blockers.append(_blocker("invalid_no_action_control", "No-action controls cannot carry agents or dose.", action_uid=uid))
        if _text(_mapping(action.get("schedule")).get("kind")) != "not_applicable":
            blockers.append(_blocker("invalid_no_action_control", "No-action schedule must be not_applicable.", action_uid=uid))
    elif action_type in NON_PHARMACOLOGICAL_ACTION_TYPES:
        if not _mapping(action.get("stimulation_or_environment")):
            blockers.append(
                _blocker(
                    "incomplete_nonpharmacological_program",
                    "Non-pharmacological actions require a typed perturbation program.",
                    action_uid=uid,
                )
            )
    if not _mapping(action.get("state_scope")):
        blockers.append(_blocker("missing_state_scope", "WorldAction requires explicit state scope.", action_uid=uid))
    if not _list(action.get("source_evidence_refs")):
        blockers.append(_blocker("missing_action_evidence", "WorldAction requires source evidence references.", action_uid=uid))
    provenance = _text(action.get("action_provenance_ref")).lower()
    if provenance in {"unknown", "unresolved", "none"}:
        blockers.append(_blocker("unknown_action_provenance", "Unknown action provenance fails closed.", action_uid=uid))
    return sorted_blockers(blockers)


def validate_information_action(value: Mapping[str, object]) -> list[dict[str, object]]:
    action = dict(value)
    uid = _text(action.get("information_action_uid"))
    blockers = _required_text_fields(
        action,
        (
            "information_action_uid",
            "tool_or_source_uid",
            "fidelity",
            "budget_class",
            "failure_policy",
            "support_domain_ref",
            "provenance_ref",
            "status",
        ),
        uid,
    )
    if action.get("schema_version") != SCHEMA_VERSIONS["information_action"]:
        blockers.append(_blocker("unknown_schema_version", "InformationAction schema is not canonical.", information_action_uid=uid))
    if not _list(action.get("candidate_uids")) or not _list(action.get("requested_observables")):
        blockers.append(
            _blocker(
                "incomplete_information_action",
                "InformationAction requires candidates and requested observables.",
                information_action_uid=uid,
            )
        )
    forbidden = sorted(set(action) & {"dose", "dose_unit", "route", "intervention_type", "state_effect"})
    if forbidden:
        blockers.append(
            _blocker(
                "world_information_action_not_separated",
                "InformationAction cannot encode a biological intervention or state effect.",
                information_action_uid=uid,
                fields=forbidden,
            )
        )
    if action.get("state_effect") not in {None, "none"}:
        blockers.append(
            _blocker(
                "information_action_state_mutation",
                "InformationAction may reveal evidence but cannot directly alter state.",
                information_action_uid=uid,
            )
        )
    if not _mapping(action.get("measurement_context")):
        blockers.append(_blocker("missing_measurement_context", "InformationAction needs measurement context.", information_action_uid=uid))
    if not _mapping(action.get("expected_cost")) or not _mapping(action.get("expected_runtime")):
        blockers.append(_blocker("missing_information_budget", "Cost and runtime must be explicit.", information_action_uid=uid))
    return sorted_blockers(blockers)


def validate_action_separation(world_action: Mapping[str, object], information_action: Mapping[str, object]) -> list[dict[str, object]]:
    blockers = [*validate_world_action(world_action), *validate_information_action(information_action)]
    if _text(world_action.get("action_uid")) == _text(information_action.get("information_action_uid")):
        blockers.append(_blocker("world_information_uid_collision", "WorldAction and InformationAction require distinct identities."))
    return sorted_blockers(blockers)


def validate_observation_request(value: Mapping[str, object]) -> list[dict[str, object]]:
    request = dict(value)
    uid = _text(request.get("request_uid"))
    blockers = _required_text_fields(
        request,
        (
            "request_uid",
            "research_question_ref",
            "requested_variable_namespace",
            "region",
            "compartment",
            "cell_type",
            "biological_unit_scope",
            "assay_or_measurement_type",
            "minimum_quality",
            "minimum_fidelity",
            "support_domain_ref",
            "provenance_ref",
            "status",
        ),
        uid,
    )
    if request.get("schema_version") != SCHEMA_VERSIONS["observation_request"]:
        blockers.append(_blocker("unknown_schema_version", "ObservationRequest schema is not canonical.", request_uid=uid))
    features = {_text(item) for item in _list(request.get("requested_feature_ids"))}
    if not features:
        blockers.append(_blocker("observation_not_measurable", "At least one measurable feature is required.", request_uid=uid))
    unsupported = sorted(features & UNSUPPORTED_CLINICAL_FEATURES)
    if unsupported:
        blockers.append(
            _blocker(
                "unsupported_clinical_outcome",
                "The current contract cannot request unsupported clinical outcomes.",
                request_uid=uid,
                features=unsupported,
            )
        )
    time_context = _mapping(request.get("timepoint_or_time_window"))
    if not time_context or not _text(time_context.get("semantics")):
        blockers.append(_blocker("missing_time_definition", "ObservationRequest requires explicit time semantics.", request_uid=uid))
    if not _list(request.get("required_units")):
        blockers.append(_blocker("missing_required_units", "ObservationRequest requires expected units.", request_uid=uid))
    if not _mapping(request.get("uncertainty_requirement")) or not _mapping(request.get("budget")):
        blockers.append(_blocker("incomplete_observation_request", "Uncertainty and budget requirements are mandatory.", request_uid=uid))
    constraints = _mapping(request.get("tool_availability_constraints"))
    if constraints.get("hypothetical_only") is True and constraints.get("empirical_result_required") is True:
        blockers.append(
            _blocker(
                "hypothetical_source_empirical_request",
                "A hypothetical-only source cannot satisfy an empirical observation request.",
                request_uid=uid,
            )
        )
    return sorted_blockers(blockers)


def _prohibited_execution_paths(value: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in PROHIBITED_EXECUTION_KEYS:
                paths.append(path)
            paths.extend(_prohibited_execution_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_prohibited_execution_paths(child, f"{prefix}[{index}]"))
    return paths


def validate_tool_request(value: Mapping[str, object]) -> ToolRequestValidation:
    request = dict(value)
    uid = _text(request.get("tool_request_uid"))
    blockers = _required_text_fields(
        request,
        (
            "tool_request_uid",
            "information_action_uid",
            "observation_request_uid",
            "tool_uid",
            "tool_version",
            "organism_context",
            "state_action_type",
            "input_schema_version",
            "output_schema_version",
            "fidelity",
            "support_domain_ref",
            "provenance_ref",
            "idempotency_key",
            "sandbox_policy",
            "status",
        ),
        uid,
    )
    if not _list(request.get("requested_observables")):
        blockers.append(_blocker("tool_request_observable_missing", "ToolRequest needs requested observables.", tool_request_uid=uid))
    prohibited = _prohibited_execution_paths(request.get("execution_payload"))
    if prohibited:
        blockers.append(
            _blocker(
                "arbitrary_shell_command_prohibited",
                "Tool requests cannot carry commands, scripts, executables, or argv.",
                tool_request_uid=uid,
                paths=sorted(prohibited),
            )
        )
    if _text(request.get("sandbox_policy")) not in {"read_only_adapter", "isolated_contract_adapter"}:
        blockers.append(_blocker("tool_request_not_sandboxed", "ToolRequest sandbox policy is invalid.", tool_request_uid=uid))
    if not _mapping(request.get("budget")):
        blockers.append(_blocker("tool_request_budget_missing", "ToolRequest needs an explicit budget.", tool_request_uid=uid))
    blockers = sorted_blockers(blockers)
    return ToolRequestValidation(
        validation_uid=stable_uid("tool_request_validation", {"request_uid": uid, "blockers": blockers}),
        tool_request_uid=uid,
        status="passed" if not blockers else "blocked",
        blockers=blockers,
    )


def match_tool_capability(request_value: Mapping[str, object], capability_value: Mapping[str, object]) -> dict[str, object]:
    request = dict(request_value)
    capability = dict(capability_value)
    validation = validate_tool_request(request)
    request_uid = _text(request.get("tool_request_uid"))
    requested = {_text(item) for item in _list(request.get("requested_observables"))}
    supported = {_text(item) for item in _list(capability.get("supported_observables"))}
    same_tool = _text(capability.get("tool_uid")) == _text(request.get("tool_uid"))
    same_version = _text(capability.get("tool_version")) == _text(request.get("tool_version"))
    checks = {
        "observable_supported": bool(requested) and requested.issubset(supported),
        "organism_context_supported": _text(request.get("organism_context")) in {_text(item) for item in _list(capability.get("organism_scope"))},
        "state_action_type_supported": _text(request.get("state_action_type")) in {_text(item) for item in _list(capability.get("state_action_types"))},
        "input_schema_compatible": _text(request.get("input_schema_version")) in {_text(item) for item in _list(capability.get("input_schema_versions"))},
        "output_schema_compatible": _text(request.get("output_schema_version")) in {_text(item) for item in _list(capability.get("output_schema_versions"))},
        "fidelity_available": _text(request.get("fidelity")) in {_text(item) for item in _list(capability.get("fidelities"))},
        "budget_sufficient": _finite_number(_mapping(request.get("budget")).get("maximum_cost"))
        and _finite_number(capability.get("maximum_cost"))
        and float(_mapping(request.get("budget"))["maximum_cost"]) >= float(capability["maximum_cost"]),
        "license_access_resolved": capability.get("license_status") == "resolved",
    }
    restrictions = [_text(item) for item in _list(capability.get("restrictions")) if _text(item)]
    if validation.status != "passed" or not same_tool or not same_version:
        state = "unsupported_schema" if validation.status != "passed" else "unsupported_domain"
        restrictions.append("tool_identity_or_request_validation_mismatch")
    elif capability.get("availability") == "temporarily_unavailable":
        state = "temporarily_unavailable"
    elif not checks["license_access_resolved"]:
        state = "unresolved_license"
    elif not checks["budget_sufficient"]:
        state = "budget_exceeded"
    elif not checks["input_schema_compatible"] or not checks["output_schema_compatible"]:
        state = "unsupported_schema"
    elif not all(
        checks[key]
        for key in ("observable_supported", "organism_context_supported", "state_action_type_supported", "fidelity_available")
    ):
        state = "unsupported_domain"
    elif restrictions:
        state = "supported_with_restrictions"
    else:
        state = "supported"
    conflicts = sorted({*_list(request.get("conflicts")), *_list(capability.get("conflicts"))})
    return ToolCapabilityMatch(
        schema_version=SCHEMA_VERSIONS["tool_capability_match"],
        match_uid=stable_uid(
            "tool_match",
            {"request_uid": request_uid, "tool_uid": capability.get("tool_uid"), "version": capability.get("tool_version")},
        ),
        tool_request_uid=request_uid,
        tool_uid=_text(capability.get("tool_uid")),
        tool_version=_text(capability.get("tool_version")),
        **checks,
        state=state,
        restrictions=sorted(set(restrictions)),
        support_domain_ref=_text(request.get("support_domain_ref")),
        provenance_ref=_text(capability.get("provenance_ref")),
        conflicts=conflicts,
    ).to_dict()


def tool_availability_decision(match_value: Mapping[str, object]) -> dict[str, object]:
    match = dict(match_value)
    state = _text(match.get("state"))
    available = state in {"supported", "supported_with_restrictions"}
    return ToolAvailabilityDecision(
        decision_uid=stable_uid("tool_availability", {"match_uid": match.get("match_uid"), "state": state}),
        match_uid=_text(match.get("match_uid")),
        state=state,
        available=available,
        reason_codes=[] if state == "supported" else [state],
    ).to_dict()


def validate_tool_capability_match(value: Mapping[str, object]) -> list[dict[str, object]]:
    match = dict(value)
    uid = _text(match.get("match_uid"))
    blockers: list[dict[str, object]] = []
    if match.get("schema_version") != SCHEMA_VERSIONS["tool_capability_match"]:
        blockers.append(_blocker("unknown_schema_version", "ToolCapabilityMatch schema is not canonical.", match_uid=uid))
    if match.get("state") not in TOOL_MATCH_STATES:
        blockers.append(_blocker("unknown_tool_match_state", "Tool match state is invalid.", match_uid=uid))
    check_fields = (
        "observable_supported",
        "organism_context_supported",
        "state_action_type_supported",
        "input_schema_compatible",
        "output_schema_compatible",
        "fidelity_available",
        "budget_sufficient",
        "license_access_resolved",
    )
    if any(not isinstance(match.get(field), bool) for field in check_fields):
        blockers.append(_blocker("tool_capability_check_missing", "Every capability check must be boolean.", match_uid=uid))
    if match.get("state") == "supported" and not all(match.get(field) is True for field in check_fields):
        blockers.append(_blocker("unsupported_tool_silently_replaced", "A supported match requires every exact capability check.", match_uid=uid))
    return sorted_blockers(blockers)


def validate_raw_tool_result(value: Mapping[str, object]) -> list[dict[str, object]]:
    result = dict(value)
    uid = _text(result.get("tool_result_uid"))
    blockers = _required_text_fields(
        result,
        (
            "tool_result_uid",
            "tool_uid",
            "tool_version",
            "request_uid",
            "failure_class",
            "stdout_ref",
            "stderr_ref",
            "support_domain_ref",
            "provenance_ref",
            "state_effect",
        ),
        uid,
    )
    if result.get("schema_version") != SCHEMA_VERSIONS["raw_tool_result"]:
        blockers.append(_blocker("unknown_schema_version", "RawToolResult schema is not canonical.", tool_result_uid=uid))
    hashes = _mapping(result.get("raw_artifact_hashes"))
    if not hashes or any(not _is_sha256(value) for value in hashes.values()):
        blockers.append(_blocker("invalid_raw_artifact_hash", "RawToolResult requires SHA-256 artifact hashes.", tool_result_uid=uid))
    if result.get("state_effect") != "none":
        blockers.append(_blocker("tool_result_direct_state_update", "RawToolResult cannot directly alter Twin state.", tool_result_uid=uid))
    forbidden = sorted(set(result) & {"resulting_state_ref", "posterior_update", "approved_update_scope"})
    if forbidden:
        blockers.append(
            _blocker(
                "tool_result_direct_state_update",
                "RawToolResult cannot contain state transition fields.",
                tool_result_uid=uid,
                fields=forbidden,
            )
        )
    if not _mapping(result.get("exit_status")) or not _mapping(result.get("raw_schema")):
        blockers.append(_blocker("incomplete_raw_result", "Raw schema and exit status are mandatory.", tool_result_uid=uid))
    return sorted_blockers(blockers)


def validate_observation_qc(value: Mapping[str, object]) -> list[dict[str, object]]:
    qc = dict(value)
    uid = _text(qc.get("qc_uid"))
    blockers: list[dict[str, object]] = []
    if qc.get("schema_version") != SCHEMA_VERSIONS["observation_qc"]:
        blockers.append(_blocker("unknown_schema_version", "ObservationQC schema is not canonical.", qc_uid=uid))
    if qc.get("qc_grade") not in QC_GRADES:
        blockers.append(_blocker("invalid_observation_qc_grade", "Observation QC grade is invalid.", qc_uid=uid))
    for dimension in QC_DIMENSIONS:
        if qc.get(dimension) not in QC_DIMENSION_STATES:
            blockers.append(_blocker("invalid_observation_qc_dimension", "QC dimension is invalid.", qc_uid=uid, dimension=dimension))
    predecessor = qc.get("predecessor_qc_grade")
    grade = qc.get("qc_grade")
    if predecessor in QC_GRADES and grade in QC_GRADES and QC_GRADE_RANK[str(grade)] > QC_GRADE_RANK[str(predecessor)]:
        if not _text(qc.get("verification_artifact_ref")):
            blockers.append(
                _blocker(
                    "observation_qc_upgrade_without_verification",
                    "QC may improve only through an explicit versioned verification artifact.",
                    qc_uid=uid,
                )
            )
    if grade == "A" and any(qc.get(dimension) not in {"passed", "not_applicable"} for dimension in QC_DIMENSIONS):
        blockers.append(_blocker("strong_qc_without_full_verification", "QC-A requires all applicable dimensions to pass.", qc_uid=uid))
    return sorted_blockers(blockers)


def validate_canonical_observation(value: Mapping[str, object]) -> list[dict[str, object]]:
    observation = dict(value)
    uid = _text(observation.get("observation_uid"))
    blockers = _required_text_fields(
        observation,
        (
            "observation_uid",
            "source_result_uid",
            "biological_unit_uid",
            "region",
            "compartment",
            "cell_type",
            "modality",
            "feature_namespace",
            "feature_id",
            "unit",
            "qc_grade",
            "support_domain_ref",
            "provenance_ref",
            "train_eval_role",
        ),
        uid,
    )
    if observation.get("schema_version") != SCHEMA_VERSIONS["canonical_observation"]:
        blockers.append(_blocker("unknown_schema_version", "CanonicalObservation schema is not canonical.", observation_uid=uid))
    if observation.get("empirical_observation") is not True:
        blockers.append(_blocker("nonempirical_canonical_observation", "CanonicalObservation must derive from an empirical raw result.", observation_uid=uid))
    if observation.get("qc_grade") not in QC_GRADES:
        blockers.append(_blocker("invalid_observation_qc_grade", "CanonicalObservation QC grade is invalid.", observation_uid=uid))
    if not _mapping(observation.get("state_time")) or not _mapping(observation.get("uncertainty")):
        blockers.append(_blocker("incomplete_observation_context", "Time and uncertainty must be explicit.", observation_uid=uid))
    return sorted_blockers(blockers)


def normalize_raw_tool_result(
    raw_value: Mapping[str, object],
    *,
    mapping: Mapping[str, object],
    qc_value: Mapping[str, object],
    provenance_status: str,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    raw = dict(raw_value)
    qc = dict(qc_value)
    blockers = [*validate_raw_tool_result(raw), *validate_observation_qc(qc)]
    exit_status = _mapping(raw.get("exit_status"))
    if exit_status.get("code") != 0 or _text(raw.get("failure_class")) not in {"none", "not_applicable"}:
        blockers.append(_blocker("raw_tool_result_failed", "Failed raw results cannot become observations.", tool_result_uid=raw.get("tool_result_uid")))
    required_verifications = (
        "schema_validated",
        "entity_mapping_verified",
        "unit_conversion_verified",
        "axis_alignment_verified",
        "biological_unit_verified",
        "provenance_verified",
    )
    missing = [field for field in required_verifications if mapping.get(field) is not True]
    if missing:
        blockers.append(
            _blocker(
                "observation_normalization_incomplete",
                "Raw result normalization requires all verification stages.",
                tool_result_uid=raw.get("tool_result_uid"),
                stages=missing,
            )
        )
    if provenance_status != "verified":
        blockers.append(_blocker("unknown_observation_provenance", "Unknown provenance cannot become a canonical observation."))
    if qc.get("qc_grade") == "D":
        blockers.append(_blocker("rejected_qc_observation", "QC-D raw evidence is rejected before observation normalization."))
    blockers = sorted_blockers(blockers)
    if blockers:
        return None, blockers
    grade = _text(qc.get("qc_grade"))
    role = {"A": "governed_evidence", "B": "governed_incomplete", "C": "directional_only", "S": "stress_only"}[grade]
    result_uid = _text(raw.get("tool_result_uid"))
    observation = CanonicalObservation(
        schema_version=SCHEMA_VERSIONS["canonical_observation"],
        observation_uid=stable_uid(
            "observation",
            {"source_result_uid": result_uid, "feature_id": mapping.get("feature_id"), "state_time": mapping.get("state_time")},
        ),
        source_result_uid=result_uid,
        biological_unit_uid=_text(mapping.get("biological_unit_uid")),
        state_time=_mapping(mapping.get("state_time")),
        region=_text(mapping.get("region")),
        compartment=_text(mapping.get("compartment")),
        cell_type=_text(mapping.get("cell_type")),
        modality=_text(mapping.get("modality")),
        feature_namespace=_text(mapping.get("feature_namespace")),
        feature_id=_text(mapping.get("feature_id")),
        value=mapping.get("value"),
        unit=_text(mapping.get("unit")),
        uncertainty=_mapping(mapping.get("uncertainty")),
        missingness=_mapping(mapping.get("missingness")),
        qc_grade=grade,
        support_domain_ref=_text(raw.get("support_domain_ref")),
        provenance_ref=_text(raw.get("provenance_ref")),
        train_eval_role=role,
        conflicts=sorted({_text(item) for item in _list(raw.get("conflicts")) if _text(item)}),
        empirical_observation=True,
    ).to_dict()
    observation_blockers = validate_canonical_observation(observation)
    return (None, observation_blockers) if observation_blockers else (observation, [])


def compile_pharmacological_action(
    action_value: Mapping[str, object],
    *,
    pkpd_context: Mapping[str, object],
    baseline_state_posterior: Mapping[str, object],
    graph_context: Mapping[str, object],
    compiler_version: str = "round13-contract-compiler-v1",
) -> dict[str, object]:
    action = dict(action_value)
    uid = _text(action.get("action_uid"))
    blockers = validate_world_action(action)
    if _text(action.get("intervention_type")) not in {"pharmacological", "combination"}:
        blockers.append(_blocker("unsupported_action", "Occupancy compilation supports pharmacological action components only.", action_uid=uid))
        status = "unsupported_action"
    elif blockers:
        status = "invalid_action"
    else:
        required_context = {
            "time_grid": _list(pkpd_context.get("time_grid")),
            "parameter_provenance": _mapping(pkpd_context.get("parameter_provenance")),
            "target_program": _mapping(pkpd_context.get("target_program")),
            "exposure_fields": _mapping(pkpd_context.get("exposure_fields")),
            "occupancy_fields": _mapping(pkpd_context.get("occupancy_fields")),
            "organism_context": _text(pkpd_context.get("organism_context")),
        }
        if not required_context["time_grid"]:
            blockers.append(_blocker("missing_time_context", "PK/PD compilation requires an explicit time grid.", action_uid=uid))
        if not required_context["target_program"]:
            blockers.append(_blocker("unresolved_target_identity", "PK/PD target identity is unresolved.", action_uid=uid))
        provenance = required_context["parameter_provenance"]
        if not provenance or provenance.get("status") != "verified" or not _text(provenance.get("artifact_ref")):
            blockers.append(_blocker("unknown_parameter_provenance", "PK/PD parameters require verified provenance.", action_uid=uid))
        if required_context["organism_context"] not in {_text(item) for item in _list(pkpd_context.get("supported_organisms"))}:
            blockers.append(_blocker("unsupported_organism_context", "PK/PD organism/context is unsupported.", action_uid=uid))
        if not baseline_state_posterior or not _text(baseline_state_posterior.get("state_ref")):
            blockers.append(_blocker("missing_baseline_state_posterior", "Compiler requires a baseline state posterior.", action_uid=uid))
        if not graph_context or not _text(graph_context.get("graph_ref")):
            blockers.append(_blocker("missing_graph_context", "Compiler requires graph/context identity.", action_uid=uid))
        if not required_context["exposure_fields"] or not required_context["occupancy_fields"]:
            blockers.append(
                _blocker(
                    "missing_pkpd_parameterization",
                    "Contract compiler cannot fabricate exposure or occupancy fields.",
                    action_uid=uid,
                )
            )
        status = "insufficient_parameters" if blockers else (
            "compiled_with_explicit_uncertainty" if _mapping(pkpd_context.get("uncertainty")) else "compiled"
        )
    blockers = sorted_blockers(blockers)
    compiled = status in {"compiled", "compiled_with_explicit_uncertainty"}
    return ExposureOccupancyProgram(
        schema_version=SCHEMA_VERSIONS["exposure_occupancy_program"],
        program_uid=stable_uid("exposure_occupancy_program", {"action_uid": uid, "compiler_version": compiler_version}),
        action_uid=uid,
        time_grid=[float(value) for value in _list(pkpd_context.get("time_grid"))] if compiled else [],
        exposure_fields=_mapping(pkpd_context.get("exposure_fields")) if compiled else {},
        occupancy_fields=_mapping(pkpd_context.get("occupancy_fields")) if compiled else {},
        target_program=_mapping(pkpd_context.get("target_program")) if compiled else {},
        off_target_context=_mapping(pkpd_context.get("off_target_context")) if compiled else {},
        parameter_provenance=_mapping(pkpd_context.get("parameter_provenance")) if compiled else {},
        uncertainty=_mapping(pkpd_context.get("uncertainty")) if compiled else {},
        support_domain=_text(action.get("support_domain_ref")),
        compiler_version=compiler_version,
        status=status,
        blockers=blockers,
        conflicts=sorted({_text(item) for item in _list(action.get("conflicts")) if _text(item)}),
    ).to_dict()


def compile_nonpharmacological_action(action_value: Mapping[str, object]) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    action = dict(action_value)
    uid = _text(action.get("action_uid"))
    action_type = _text(action.get("intervention_type"))
    blockers = validate_world_action(action)
    if action_type not in NON_PHARMACOLOGICAL_ACTION_TYPES:
        blockers.append(_blocker("unsupported_action", "No typed non-pharmacological adapter exists.", action_uid=uid))
    program = _mapping(action.get("stimulation_or_environment"))
    if action_type != "no_action_control":
        required_program_fields = ("affected_state_variables", "magnitude_or_program", "uncertainty")
        missing = [field for field in required_program_fields if not program.get(field)]
        if missing:
            blockers.append(
                _blocker(
                    "incomplete_nonpharmacological_program",
                    "Typed non-pharmacological adapter fields are incomplete.",
                    action_uid=uid,
                    fields=missing,
                )
            )
    blockers = sorted_blockers(blockers)
    if blockers:
        return None, blockers
    no_action = action_type == "no_action_control"
    field = CanonicalActionField(
        action_field_uid=stable_uid("action_field", {"action_uid": uid, "type": action_type}),
        action_uid=uid,
        intervention_type=action_type,
        affected_state_variables=[] if no_action else [_text(item) for item in _list(program.get("affected_state_variables"))],
        spatial_scope=_mapping(action.get("state_scope")),
        temporal_scope=_mapping(action.get("duration")),
        magnitude_or_program={"control": True} if no_action else _mapping(program.get("magnitude_or_program")),
        uncertainty={"status": "not_applicable"} if no_action else _mapping(program.get("uncertainty")),
        provenance_ref=_text(action.get("action_provenance_ref")),
        support_domain_ref=_text(action.get("support_domain_ref")),
        conflicts=[_text(item) for item in _list(action.get("conflicts"))],
        status="contract_compiled",
    )
    return field.to_dict(), []


def support_domain_compatible(
    source_value: Mapping[str, object],
    target_value: Mapping[str, object],
) -> tuple[bool, list[dict[str, object]]]:
    source = dict(source_value)
    target = dict(target_value)
    blockers = [*validate_support_domain(source), *validate_support_domain(target)]
    scope_fields = (
        "organism_scope",
        "disease_scope",
        "region_scope",
        "compartment_scope",
        "cell_type_scope",
        "modality_scope",
        "assay_scope",
        "action_scope",
        "dose_scope",
        "route_scope",
        "time_scope",
        "model_family_scope",
        "tool_family_scope",
    )
    mismatches: list[str] = []
    for field in scope_fields:
        source_values = {_text(item) for item in _list(source.get(field))} - {"not_applicable"}
        target_values = {_text(item) for item in _list(target.get(field))} - {"not_applicable"}
        if source_values and target_values and source_values.isdisjoint(target_values):
            mismatches.append(field)
    if mismatches:
        blockers.append(
            _blocker(
                "support_domain_mismatch",
                "Source and target support domains are incompatible.",
                source_domain=source.get("support_domain_uid"),
                target_domain=target.get("support_domain_uid"),
                fields=mismatches,
            )
        )
    return not blockers, sorted_blockers(blockers)


def action_support_decision(
    action_value: Mapping[str, object],
    *,
    action_domain: Mapping[str, object],
    target_domain: Mapping[str, object],
) -> dict[str, object]:
    action = dict(action_value)
    compatible, blockers = support_domain_compatible(action_domain, target_domain)
    if compatible:
        decision = "in_domain"
        allowed_use = "contract_compilation"
    else:
        requested_status = _text(action.get("status"))
        decision = requested_status if requested_status in OUT_OF_DOMAIN_RETENTION_STATES else "unsupported"
        allowed_use = decision
    return ActionSupportDecision(
        decision_uid=stable_uid("action_support", {"action_uid": action.get("action_uid"), "decision": decision}),
        action_uid=_text(action.get("action_uid")),
        support_domain_ref=_text(action.get("support_domain_ref")),
        decision=decision,
        reason_codes=[_text(row.get("code")) for row in blockers],
        allowed_use=allowed_use,
        provenance_ref=_text(action.get("action_provenance_ref")),
    ).to_dict()


def validate_hypothetical_scenario(value: Mapping[str, object]) -> list[dict[str, object]]:
    scenario = dict(value)
    uid = _text(scenario.get("scenario_uid"))
    blockers = _required_text_fields(
        scenario,
        ("scenario_uid", "baseline_state_ref", "support_domain_ref", "provenance_ref"),
        uid,
    )
    if scenario.get("schema_version") != SCHEMA_VERSIONS["hypothetical_action_scenario"]:
        blockers.append(_blocker("unknown_schema_version", "HypotheticalActionScenario schema is not canonical.", scenario_uid=uid))
    required_flags = {
        "model_based_counterfactual": True,
        "causal_effect_claim": False,
        "clinical_recommendation": False,
        "empirical_observation": False,
    }
    for field, expected in required_flags.items():
        if scenario.get(field) is not expected:
            blockers.append(
                _blocker(
                    "hypothetical_boundary_violation",
                    "Hypothetical scenario boundary flags are immutable.",
                    scenario_uid=uid,
                    field=field,
                )
            )
    if validate_world_action(_mapping(scenario.get("candidate_world_action"))):
        blockers.append(_blocker("invalid_hypothetical_action", "Scenario candidate action must satisfy WorldAction syntax.", scenario_uid=uid))
    return sorted_blockers(blockers)


def validate_assimilation_proposal(
    value: Mapping[str, object],
    *,
    observations: Mapping[str, Mapping[str, object]],
    support_domains: Mapping[str, Mapping[str, object]],
    supported_state_components: set[str],
) -> list[dict[str, object]]:
    proposal = dict(value)
    uid = _text(proposal.get("proposal_uid"))
    blockers = _required_text_fields(
        proposal,
        (
            "proposal_uid",
            "prior_state_ref",
            "proposed_update_type",
            "likelihood_or_update_model_ref",
            "support_domain",
            "provenance_ref",
        ),
        uid,
    )
    if proposal.get("schema_version") != SCHEMA_VERSIONS["assimilation_proposal"]:
        blockers.append(_blocker("unknown_schema_version", "AssimilationProposal schema is not canonical.", proposal_uid=uid))
    update_type = _text(proposal.get("proposed_update_type"))
    if update_type not in ASSIMILATION_UPDATE_TYPES:
        blockers.append(_blocker("unsupported_assimilation_update", "Assimilation update type is invalid.", proposal_uid=uid))
    if proposal.get("state_write_authorized") is not False:
        blockers.append(_blocker("proposal_direct_state_update", "AssimilationProposal cannot authorize or write state.", proposal_uid=uid))
    observation_uids = [_text(item) for item in _list(proposal.get("observation_uids"))]
    source_kinds = _mapping(proposal.get("source_kinds"))
    if not observation_uids or any(item not in observations for item in observation_uids):
        blockers.append(_blocker("missing_assimilation_observation", "Proposal references missing observations.", proposal_uid=uid))
    if any(source_kinds.get(item) == "hypothetical_scenario" for item in observation_uids):
        blockers.append(_blocker("hypothetical_assimilation", "Hypothetical scenarios cannot enter assimilation.", proposal_uid=uid))
    if any(source_kinds.get(item) == "raw_tool_result_unknown_provenance" for item in observation_uids):
        blockers.append(_blocker("rejected_provenance", "Unknown raw-result provenance cannot enter assimilation.", proposal_uid=uid))
    target_components = {_text(item) for item in _list(proposal.get("target_state_components"))}
    unsupported = sorted(target_components - supported_state_components)
    if unsupported:
        blockers.append(
            _blocker(
                "unsupported_state_component",
                "Proposal targets unsupported state components.",
                proposal_uid=uid,
                components=unsupported,
            )
        )
    if update_type not in {"no_update", "conflict_register_only"} and not _mapping(proposal.get("uncertainty_change")):
        blockers.append(_blocker("state_overwrite_without_uncertainty", "Posterior updates require explicit uncertainty change.", proposal_uid=uid))
    proposal_domain = support_domains.get(_text(proposal.get("support_domain")))
    if proposal_domain is None:
        blockers.append(_blocker("missing_support_domain", "Proposal support domain is missing.", proposal_uid=uid))
    required_conflicts: set[str] = set()
    declared_roles = _mapping(proposal.get("train_eval_roles"))
    for observation_uid in observation_uids:
        observation = observations.get(observation_uid)
        if observation is None:
            continue
        if observation.get("empirical_observation") is not True or observation.get("source_kind") == "hypothetical_scenario":
            blockers.append(_blocker("hypothetical_assimilation", "Hypothetical scenarios cannot enter assimilation.", proposal_uid=uid))
        grade = _text(observation.get("qc_grade"))
        if grade == "D":
            blockers.append(_blocker("rejected_qc", "QC-D observations cannot enter assimilation.", proposal_uid=uid))
        if grade == "S" and update_type not in {"conflict_register_only", "no_update"}:
            blockers.append(_blocker("stress_evidence_assimilation", "QC-S evidence is conflict/stress only.", proposal_uid=uid))
        if grade == "C" and update_type not in {
            "uncertainty_only_update",
            "support_domain_update",
            "conflict_register_only",
            "no_update",
        }:
            blockers.append(_blocker("directional_evidence_state_update", "QC-C evidence cannot directly update state values.", proposal_uid=uid))
        provenance = _text(observation.get("provenance_status") or observation.get("provenance_ref"))
        if provenance.lower() in {"", "unknown", "unresolved", "rejected"}:
            blockers.append(_blocker("rejected_provenance", "Unknown observation provenance cannot enter assimilation.", proposal_uid=uid))
        if declared_roles.get(observation_uid) != observation.get("train_eval_role"):
            blockers.append(_blocker("train_eval_role_change", "Assimilation cannot change train/eval role.", proposal_uid=uid))
        required_conflicts.update(_text(item) for item in _list(observation.get("conflicts")) if _text(item))
        observation_domain = support_domains.get(_text(observation.get("support_domain_ref")))
        if observation_domain is None or proposal_domain is None:
            continue
        compatible, domain_blockers = support_domain_compatible(observation_domain, proposal_domain)
        if not compatible:
            blockers.extend(domain_blockers)
            blockers.append(_blocker("cross_domain_assimilation", "Cross-domain observation update is prohibited.", proposal_uid=uid))
        time_semantics = _text(_mapping(observation.get("state_time")).get("semantics"))
        temporal_target = any(marker in component for marker in TEMPORAL_STATE_COMPONENT_MARKERS for component in target_components)
        if time_semantics in STATIC_TIME_SEMANTICS and temporal_target:
            blockers.append(
                _blocker(
                    "static_endpoint_temporal_update",
                    "A static endpoint cannot authorize a trajectory or temporal state update.",
                    proposal_uid=uid,
                )
            )
    declared_conflicts = {_text(item) for item in _list(proposal.get("conflicts")) if _text(item)}
    missing_conflicts = sorted(required_conflicts - declared_conflicts)
    if missing_conflicts:
        blockers.append(
            _blocker(
                "conflict_dropped_during_assimilation",
                "All observation conflicts must propagate into the proposal.",
                proposal_uid=uid,
                conflicts=missing_conflicts,
            )
        )
    return sorted_blockers(blockers)


def validate_assimilation_decision(
    value: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    canonical_controller_uid: str,
    approved_validator_uid: str,
) -> list[dict[str, object]]:
    decision = dict(value)
    uid = _text(decision.get("decision_uid"))
    blockers: list[dict[str, object]] = []
    if decision.get("schema_version") != SCHEMA_VERSIONS["assimilation_decision"]:
        blockers.append(_blocker("unknown_schema_version", "AssimilationDecision schema is not canonical.", decision_uid=uid))
    if decision.get("decision") not in ASSIMILATION_DECISIONS:
        blockers.append(_blocker("unknown_assimilation_decision", "Assimilation decision is invalid.", decision_uid=uid))
    if decision.get("controller_uid") != canonical_controller_uid:
        blockers.append(_blocker("noncanonical_assimilation_controller", "Only the canonical controller may approve assimilation.", decision_uid=uid))
    if decision.get("validator_uid") != approved_validator_uid:
        blockers.append(_blocker("unapproved_assimilation_validator", "Assimilation validator is not approved.", decision_uid=uid))
    if decision.get("proposal_uid") != proposal.get("proposal_uid"):
        blockers.append(_blocker("assimilation_proposal_identity_mismatch", "Decision does not bind its proposal.", decision_uid=uid))
    if not _is_sha256(decision.get("prior_state_hash")):
        blockers.append(_blocker("invalid_prior_state_hash", "Assimilation decision requires prior state SHA-256.", decision_uid=uid))
    hashes = _mapping(decision.get("input_observation_hashes"))
    if not hashes or any(not _is_sha256(value) for value in hashes.values()):
        blockers.append(_blocker("invalid_observation_hash", "Assimilation inputs require SHA-256 hashes.", decision_uid=uid))
    result = _text(decision.get("decision"))
    update_type = _text(proposal.get("proposed_update_type"))
    if result == "approved_uncertainty_only" and update_type != "uncertainty_only_update":
        blockers.append(_blocker("assimilation_scope_mismatch", "Uncertainty-only approval requires uncertainty-only proposal.", decision_uid=uid))
    if result == "approved_conflict_register_only" and update_type != "conflict_register_only":
        blockers.append(_blocker("assimilation_scope_mismatch", "Conflict-only approval requires conflict-only proposal.", decision_uid=uid))
    if result.startswith("approved") and not _list(decision.get("approved_update_scope")):
        blockers.append(_blocker("approved_scope_missing", "Approved decisions require explicit update scope.", decision_uid=uid))
    required_conflicts = {_text(item) for item in _list(proposal.get("conflicts")) if _text(item)}
    handling = _mapping(decision.get("conflict_handling"))
    missing = sorted(required_conflicts - set(handling))
    invalid = sorted(key for key, state in handling.items() if state not in CONFLICT_STATES)
    if missing or invalid:
        blockers.append(
            _blocker(
                "conflict_dropped_during_decision",
                "AssimilationDecision must retain typed conflict handling.",
                decision_uid=uid,
                missing=missing,
                invalid=invalid,
            )
        )
    return sorted_blockers(blockers)


def conflicts_preserved(source_value: Mapping[str, object], target_value: Mapping[str, object]) -> list[dict[str, object]]:
    source = {_text(item) for item in _list(source_value.get("conflicts")) if _text(item)}
    target = {_text(item) for item in _list(target_value.get("conflicts")) if _text(item)}
    missing = sorted(source - target)
    return [] if not missing else [
        _blocker("conflict_dropped", "Typed conflicts cannot disappear between contract stages.", conflicts=missing)
    ]


def frozen_gap_action_observation_rows(handoff_value: Mapping[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for gap in _list(handoff_value.get("frozen_gap_ledger")):
        value = _mapping(gap)
        node_type = _text(value.get("missing_node_type"))
        if node_type in {"structured_action", "perturbation_endpoint", "trajectory", "simulator_output", "hypothetical_scenario"}:
            required_action_type = "WorldAction"
        elif node_type == "tool_result":
            required_action_type = "InformationAction"
        else:
            required_action_type = "not_applicable"
        observation_types = {
            "observation": "CanonicalObservation",
            "perturbation_endpoint": "CanonicalObservation",
            "trajectory": "CanonicalObservation_sequence_not_static_endpoint",
            "simulator_output": "CanonicalObservation_model_output",
            "tool_result": "RawToolResult_then_CanonicalObservation",
            "claim": "not_observation_claim_lineage_only",
            "structured_action": "not_applicable",
            "hypothetical_scenario": "not_eligible_as_observation",
        }
        rows.append(
            {
                "gap_uid": _text(value.get("gap_uid")),
                "required_action_type": required_action_type,
                "required_observation_type": observation_types.get(node_type, "unsupported"),
                "missing_identity_axis": _text(value.get("required_identity_axes")),
                "missing_support_domain": "materialized_object_support_domain_required",
                "missing_tool_capability": "tool_registry_capability_required" if node_type == "tool_result" else "not_applicable",
                "missing_assimilation_prerequisite": "governed_canonical_observation_absent",
                "current_status": _text(value.get("status")),
                "downstream_blocked_capability": _text(value.get("affected_downstream_role")),
            }
        )
    return rows


def validate_frozen_gap_action_observation_map(
    handoff_value: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    source = [_mapping(row) for row in _list(handoff_value.get("frozen_gap_ledger"))]
    mapped = [dict(row) for row in rows]
    blockers: list[dict[str, object]] = []
    source_uids = [_text(row.get("gap_uid")) for row in source]
    mapped_uids = [_text(row.get("gap_uid")) for row in mapped]
    if len(source) != 20 or len(mapped) != 20 or source_uids != mapped_uids:
        blockers.append(
            _blocker(
                "frozen_gap_count_drift",
                "The exact ordered 20-gap ledger must be preserved.",
                source_count=len(source),
                mapped_count=len(mapped),
            )
        )
    source_status = {_text(row.get("gap_uid")): _text(row.get("status")) for row in source}
    for row in mapped:
        uid = _text(row.get("gap_uid"))
        if row.get("current_status") != source_status.get(uid):
            blockers.append(_blocker("frozen_gap_reduced_without_evidence", "Frozen gap status changed without evidence.", gap_uid=uid))
    return sorted_blockers(blockers)


def _duplicate_uid_blockers(graph: Mapping[str, object]) -> list[dict[str, object]]:
    family_uid_fields = {
        "world_actions": "action_uid",
        "information_actions": "information_action_uid",
        "observation_requests": "request_uid",
        "tool_requests": "tool_request_uid",
        "tool_capability_matches": "match_uid",
        "raw_tool_results": "tool_result_uid",
        "observation_qc": "qc_uid",
        "canonical_observations": "observation_uid",
        "assimilation_proposals": "proposal_uid",
        "assimilation_decisions": "decision_uid",
        "hypothetical_scenarios": "scenario_uid",
        "exposure_occupancy_programs": "program_uid",
        "canonical_action_fields": "action_field_uid",
        "action_support_decisions": "decision_uid",
        "support_domains": "support_domain_uid",
        "conflicts": "conflict_uid",
    }
    blockers: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    for family, uid_field in family_uid_fields.items():
        for row in _list(graph.get(family)):
            uid = _text(_mapping(row).get(uid_field))
            if not uid:
                blockers.append(_blocker("missing_object_uid", "Graph object UID is missing.", family=family))
            elif uid in seen:
                blockers.append(_blocker("duplicate_uid", "Graph UIDs must be globally unique.", uid=uid, families=[seen[uid], family]))
            else:
                seen[uid] = family
    return blockers


def _lineage_cycle_blockers(graph: Mapping[str, object]) -> list[dict[str, object]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for edge_value in _list(graph.get("lineage_edges")):
        edge = _mapping(edge_value)
        source = _text(edge.get("source_uid"))
        target = _text(edge.get("target_uid"))
        if not source or not target:
            return [_blocker("invalid_lineage_edge", "Lineage edges require source and target UIDs.")]
        nodes.update({source, target})
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
            indegree.setdefault(source, 0)
    queue = deque(sorted(node for node in nodes if indegree[node] == 0))
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return [] if visited == len(nodes) else [_blocker("lineage_cycle", "Round-13 lineage graph must be acyclic.")]


def validate_round13_graph(
    graph_value: Mapping[str, object],
    *,
    canonical_controller_uid: str,
    approved_validator_uid: str,
    supported_state_components: set[str],
) -> dict[str, object]:
    graph = dict(graph_value)
    blockers: list[dict[str, object]] = []
    if graph.get("schema_version") != SCHEMA_VERSIONS["graph"]:
        blockers.append(_blocker("unknown_schema_version", "Round-13 graph schema is not canonical."))
    blockers.extend(_duplicate_uid_blockers(graph))
    blockers.extend(_lineage_cycle_blockers(graph))
    domains = {_text(row.get("support_domain_uid")): _mapping(row) for row in _list(graph.get("support_domains"))}
    conflicts = {_text(row.get("conflict_uid")): _mapping(row) for row in _list(graph.get("conflicts"))}
    for domain in domains.values():
        blockers.extend(validate_support_domain(domain))
    for conflict in conflicts.values():
        blockers.extend(validate_conflict(conflict))
    for family in (
        "world_actions",
        "information_actions",
        "observation_requests",
        "tool_requests",
        "tool_capability_matches",
        "raw_tool_results",
        "canonical_observations",
        "assimilation_proposals",
    ):
        for row_value in _list(graph.get(family)):
            row = _mapping(row_value)
            support_ref = _text(row.get("support_domain_ref") or row.get("support_domain"))
            if support_ref not in domains:
                blockers.append(_blocker("missing_support_domain", "Round-13 object has no registered support domain.", family=family))
            unknown_conflicts = sorted({_text(item) for item in _list(row.get("conflicts"))} - set(conflicts))
            if unknown_conflicts:
                blockers.append(_blocker("missing_typed_conflict", "Object references unknown conflicts.", family=family, conflicts=unknown_conflicts))
    for row in _list(graph.get("world_actions")):
        blockers.extend(validate_world_action(_mapping(row)))
    for row in _list(graph.get("information_actions")):
        blockers.extend(validate_information_action(_mapping(row)))
    for row in _list(graph.get("observation_requests")):
        blockers.extend(validate_observation_request(_mapping(row)))
    for row in _list(graph.get("tool_requests")):
        blockers.extend(validate_tool_request(_mapping(row)).blockers)
    for row in _list(graph.get("tool_capability_matches")):
        blockers.extend(validate_tool_capability_match(_mapping(row)))
    raw_results = {_text(row.get("tool_result_uid")): _mapping(row) for row in _list(graph.get("raw_tool_results"))}
    for raw in raw_results.values():
        blockers.extend(validate_raw_tool_result(raw))
    for row in _list(graph.get("observation_qc")):
        blockers.extend(validate_observation_qc(_mapping(row)))
    observations = {_text(row.get("observation_uid")): _mapping(row) for row in _list(graph.get("canonical_observations"))}
    for observation in observations.values():
        blockers.extend(validate_canonical_observation(observation))
        raw = raw_results.get(_text(observation.get("source_result_uid")))
        if raw is None:
            blockers.append(_blocker("raw_observation_lineage_missing", "CanonicalObservation requires a RawToolResult parent."))
        else:
            blockers.extend(conflicts_preserved(raw, observation))
    proposals = {_text(row.get("proposal_uid")): _mapping(row) for row in _list(graph.get("assimilation_proposals"))}
    for proposal in proposals.values():
        proposal_blockers = validate_assimilation_proposal(
            proposal,
            observations=observations,
            support_domains=domains,
            supported_state_components=supported_state_components,
        )
        recorded_codes = sorted(_text(row.get("code")) for row in _list(proposal.get("blockers")))
        observed_codes = sorted(_text(row.get("code")) for row in proposal_blockers)
        expected_status = "passed" if not proposal_blockers else "rejected"
        if proposal.get("validation_status") != expected_status or recorded_codes != observed_codes:
            blockers.append(
                _blocker(
                    "assimilation_validation_record_mismatch",
                    "Proposal validation status and blockers must match fail-closed validation.",
                    proposal_uid=proposal.get("proposal_uid"),
                    expected_status=expected_status,
                    expected_codes=observed_codes,
                    recorded_codes=recorded_codes,
                )
            )
    for row in _list(graph.get("assimilation_decisions")):
        decision = _mapping(row)
        proposal = proposals.get(_text(decision.get("proposal_uid")), {})
        blockers.extend(
            validate_assimilation_decision(
                decision,
                proposal=proposal,
                canonical_controller_uid=canonical_controller_uid,
                approved_validator_uid=approved_validator_uid,
            )
        )
    for row in _list(graph.get("hypothetical_scenarios")):
        blockers.extend(validate_hypothetical_scenario(_mapping(row)))
    scenario_uids = {_text(row.get("scenario_uid")) for row in _list(graph.get("hypothetical_scenarios"))}
    if any(_text(row.get("source_result_uid")) in scenario_uids for row in observations.values()):
        blockers.append(_blocker("hypothetical_scenario_became_observation", "Scenario identities cannot source observations."))
    expected_invariants = {
        "world_action_may_change_simulated_state": True,
        "information_action_may_reveal_observation_only": True,
        "information_action_direct_state_effect": False,
        "tool_result_direct_state_effect": False,
        "llm_proposal_direct_state_effect": False,
        "assimilation_decision_required_for_posterior_update": True,
    }
    if _mapping(graph.get("invariants")) != expected_invariants:
        blockers.append(_blocker("action_observation_invariant_mismatch", "Round-13 separation invariants are not exact."))
    if _mapping(graph.get("evidence_fabric")).get("sealed") is not False:
        blockers.append(_blocker("evidence_fabric_sealed", "Round-13 cannot seal the Evidence Fabric."))
    if _mapping(graph.get("dt_core")).get("status") != "not_evaluated":
        blockers.append(_blocker("dt_core_evaluated", "DT_CORE must remain not evaluated."))
    blockers = sorted_blockers(blockers)
    object_counts = {
        family: len(_list(graph.get(family)))
        for family in (
            "world_actions",
            "information_actions",
            "observation_requests",
            "tool_requests",
            "tool_capability_matches",
            "raw_tool_results",
            "canonical_observations",
            "assimilation_proposals",
            "assimilation_decisions",
            "hypothetical_scenarios",
        )
    }
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "blocker_count": len(blockers),
        "object_counts": object_counts,
    }


ROUND13_GATE_STATES = frozenset(
    {
        "action_observation_contracts_resolved",
        "blocked_action_semantics",
        "blocked_information_action_semantics",
        "blocked_support_domain",
        "blocked_tool_capability",
        "blocked_observation_qc",
        "blocked_assimilation_authority",
        "blocked_conflict_propagation",
        "blocked_hypothetical_boundary",
        "blocked_execution_verification",
        "blocked_frozen_gap_drift",
    }
)


def round13_gate_status(blockers_value: Sequence[Mapping[str, object]]) -> str:
    blockers = [dict(row) for row in blockers_value]
    if not blockers:
        return "action_observation_contracts_resolved"
    codes = {_text(row.get("code")) for row in blockers}
    priorities = (
        ("blocked_frozen_gap_drift", {"frozen_gap_count_drift", "frozen_gap_reduced_without_evidence"}),
        ("blocked_execution_verification", {"execution_identity_not_exact_clean", "full_suite_not_verified", "test_inventory_incomplete"}),
        ("blocked_assimilation_authority", {"noncanonical_assimilation_controller", "unapproved_assimilation_validator", "proposal_direct_state_update"}),
        ("blocked_conflict_propagation", {code for code in codes if "conflict" in code}),
        ("blocked_hypothetical_boundary", {code for code in codes if "hypothetical" in code}),
        ("blocked_tool_capability", {code for code in codes if "tool" in code or "shell" in code}),
        ("blocked_observation_qc", {code for code in codes if "observation" in code or "qc" in code}),
        ("blocked_support_domain", {code for code in codes if "support_domain" in code or "cross_domain" in code}),
        ("blocked_information_action_semantics", {code for code in codes if "information_action" in code}),
    )
    for state, matching_codes in priorities:
        if codes & matching_codes:
            return state
    return "blocked_action_semantics"


def canonical_object_hash(value: Mapping[str, object]) -> str:
    return _sha256(dict(value))


def round13_graph_uid(graph_without_uid: Mapping[str, object]) -> str:
    return stable_uid("round13_graph", dict(graph_without_uid))


def evidence_support_domain_from_dict(value: Mapping[str, object]) -> EvidenceSupportDomain:
    """Build the shared Round-12 support-domain type without weakening it."""

    return EvidenceSupportDomain(**dict(value))
