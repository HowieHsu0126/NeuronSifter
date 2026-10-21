"""Fail-closed contracts for governed Round-16 what-if execution.

Rollouts are model outputs. This module never promotes them to observations,
causal effects, clinical recommendations, or evidence that resolves empirical
gaps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import re
from typing import Any

from Libs.utils.data.action_observation_contracts import validate_world_action
from Libs.utils.data.trajectory_contracts import (
    blocker,
    canonical_sha256,
    is_sha256,
    sorted_blockers,
    stable_uid,
)


SCHEMA_VERSIONS = {
    "what_if_request": "neuronsifter-what-if-request-v1",
    "what_if_baseline": "neuronsifter-what-if-baseline-v1",
    "what_if_action_binding": "neuronsifter-what-if-action-binding-v1",
    "rollout_engine": "neuronsifter-rollout-engine-v1",
    "rollout_request": "neuronsifter-rollout-request-v1",
    "rollout_attempt": "neuronsifter-rollout-attempt-v1",
    "rollout_trajectory": "neuronsifter-rollout-trajectory-v1",
    "rollout_control_pair": "neuronsifter-rollout-control-pair-v1",
    "counterfactual_contrast": "neuronsifter-counterfactual-contrast-v1",
    "rollout_uncertainty": "neuronsifter-rollout-uncertainty-v1",
    "rollout_constraint_report": "neuronsifter-rollout-constraint-report-v1",
    "rollout_reliability_report": "neuronsifter-rollout-reliability-report-v1",
    "what_if_eligibility_decision": "neuronsifter-what-if-eligibility-decision-v1",
    "what_if_result": "neuronsifter-what-if-result-v1",
    "release": "neuronsifter-round16-what-if-release-candidate-v1",
}

ENGINE_TYPES = frozenset(
    {
        "mechanistic_rollout_engine",
        "qualified_simulator",
        "validated_replay_engine",
        "hybrid_multiscale_rollout",
        "learned_world_action_model",
        "ensemble_rollout_engine",
    }
)
ENGINE_STATES = frozenset(
    {
        "qualified_for_contract_testing",
        "qualified_for_restricted_what_if",
        "qualified_for_formal_what_if",
        "stress_only",
        "unsupported",
        "blocked_untrained",
        "blocked_support_domain",
        "blocked_action",
        "blocked_state_namespace",
        "blocked_reliability",
    }
)
BASELINE_CLASSES = frozenset(
    {
        "governed_observation_derived",
        "assimilated_posterior",
        "validated_mechanistic_steady_state",
        "validated_simulator_state",
        "qualification_synthetic",
    }
)
CONTROL_TYPES = frozenset(
    {"no_action_control", "vehicle_control", "sham_control", "baseline_hold", "matched_alternative_action"}
)
PAIRING_DECISIONS = frozenset(
    {"exact_deterministic_pair", "matched_stochastic_pair", "group_level_pair", "unpaired_descriptive_rollout"}
)
CONTRAST_ESTIMATORS = frozenset(
    {
        "paired_difference",
        "normalized_paired_difference",
        "ensemble_mean_difference",
        "distributional_distance",
        "time_to_event_difference",
        "area_under_trajectory_difference",
    }
)
HORIZON_DECISIONS = frozenset(
    {"within_qualified_horizon", "restricted_extrapolation", "outside_qualified_horizon", "unknown"}
)
SUPPORT_DECISIONS = frozenset(
    {"in_domain", "restricted_in_domain", "extrapolative_stress_only", "unsupported", "unknown"}
)
RELIABILITY_STATES = frozenset({"qualified", "qualified_with_restrictions", "uncertain", "unreliable", "unsupported"})
CONSTRAINT_STATES = frozenset({"valid", "valid_with_tolerance", "restricted", "quarantined", "invalid"})
UNCERTAINTY_COMPONENTS = (
    "initial_state_uncertainty",
    "action_parameter_uncertainty",
    "model_parameter_uncertainty",
    "stochastic_rollout_uncertainty",
    "numerical_uncertainty",
    "model_family_disagreement",
    "fidelity_disagreement",
    "support_domain_extrapolation",
    "observation_or_calibration_uncertainty",
)
ELIGIBILITY_ROLES = (
    "eligible_for_contract_validation",
    "eligible_for_runtime_validation",
    "eligible_for_restricted_method_result",
    "eligible_for_stress_analysis",
    "eligible_for_hypothesis_generation",
    "eligible_for_formal_model_based_claim",
    "eligible_for_empirical_validation",
    "eligible_for_causal_claim",
    "eligible_for_clinical_recommendation",
)
FROZEN_GAP_STATUSES = frozenset(
    {
        "not_addressable_by_model_rollout",
        "hypothesis_generated_only",
        "stress_information_only",
        "requires_empirical_observation",
        "requires_new_e4",
        "requires_support_domain",
        "frozen_unresolved",
    }
)
PROHIBITED_CLAIM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bthe intervention will cause\b",
        r"\bthe treatment is effective\b",
        r"\bthe patient will respond\b",
        r"\bclinically recommended\b",
        r"\bvalidated biological effect\b",
    )
)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _required(value: Mapping[str, object], fields: Sequence[str], uid: str) -> list[dict[str, object]]:
    return [
        blocker("missing_required_field", f"{field} must be explicit.", object_uid=uid, field=field)
        for field in fields
        if value.get(field) is None or value.get(field) == "" or value.get(field) == [] or value.get(field) == {}
    ]


@dataclass(frozen=True)
class WhatIfRequest:
    schema_version: str
    request_uid: str
    research_question_uid: str
    baseline_state_ref: str
    candidate_world_action_ref: str
    control_action_ref: str
    requested_outcomes: Sequence[str]
    requested_time_horizon: Mapping[str, object]
    requested_time_grid: Sequence[float]
    requested_regions: Sequence[str]
    requested_compartments: Sequence[str]
    requested_feature_namespace: str
    rollout_engine_policy: Mapping[str, object]
    ensemble_policy: Mapping[str, object]
    fidelity_policy: Mapping[str, object]
    uncertainty_requirements: Sequence[str]
    constraint_requirements: Sequence[str]
    support_domain_ref: str
    budget: Mapping[str, object]
    provenance: Mapping[str, object]
    model_based_counterfactual: bool
    causal_effect_claim: bool
    clinical_recommendation: bool
    empirical_observation: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WhatIfBaseline:
    schema_version: str
    baseline_uid: str
    state_ref: str
    state_hash: str
    state_time: float
    state_uncertainty: Mapping[str, object]
    biological_or_model_context: Mapping[str, object]
    region: Sequence[str]
    compartment: Sequence[str]
    feature_namespace: str
    support_domain_ref: str
    source_role: str
    qc_grade: str
    missingness: Mapping[str, object]
    provenance: Mapping[str, object]
    baseline_class: str
    imputation_policy: str
    empirical_state: bool
    qualification_only: bool
    eligible_for_biological_claims: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WhatIfActionBinding:
    schema_version: str
    binding_uid: str
    request_uid: str
    world_action: Mapping[str, object]
    action_program_hash: str
    parameter_provenance: Mapping[str, object]
    support_domain_ref: str
    spatial_scope: Mapping[str, object]
    temporal_scope: Mapping[str, object]
    combination_components: Sequence[Mapping[str, object]]
    interaction_assumptions: Sequence[str]
    completeness: str
    blockers: Sequence[Mapping[str, object]]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutEngine:
    schema_version: str
    engine_uid: str
    engine_type: str
    version: str
    source: str
    code_identity: str
    environment_identity: str
    supported_baseline_classes: Sequence[str]
    supported_action_classes: Sequence[str]
    supported_state_variables: Sequence[str]
    supported_horizons: Sequence[Mapping[str, object]]
    supported_fidelities: Sequence[str]
    support_domain_ref: str
    uncertainty_capabilities: Sequence[str]
    constraint_capabilities: Sequence[str]
    qualification_evidence: Mapping[str, object]
    known_limitations: Sequence[str]
    trained: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutRequest:
    schema_version: str
    rollout_request_uid: str
    what_if_request_uid: str
    engine_uid: str
    engine_version: str
    baseline_hash: str
    action_program_hash: str
    control_program_hash: str
    parameter_hash: str
    seed: int
    fidelity: str
    time_grid: Sequence[float]
    constraint_policy: Mapping[str, object]
    uncertainty_policy: Mapping[str, object]
    exact_code_commit: str
    environment_hash: str
    input_bundle_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutAttempt:
    schema_version: str
    attempt_uid: str
    parent_request_uid: str
    submission_command: str
    runtime_identity: Mapping[str, object]
    job_identity: Mapping[str, object]
    stdout_ref: str
    stderr_ref: str
    terminal_state: str
    exit_code: str
    raw_output_hashes: Mapping[str, str]
    failure_class: str | None
    supersedes_attempt_uid: str | None
    preserved: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutTrajectory:
    schema_version: str
    rollout_uid: str
    request_uid: str
    attempt_uid: str
    engine_uid: str
    baseline_uid: str
    action_uid: str
    control_pair_uid: str
    native_time_grid: Sequence[float]
    canonical_time_grid: Sequence[float]
    state_namespace: str
    state_artifact_ref: str
    uncertainty_artifact_ref: str
    constraint_artifact_ref: str
    content_hashes: Mapping[str, str]
    status: str
    eligibility: str
    provenance: Mapping[str, object]
    output_class: str
    empirical_observation: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControlRollout:
    schema_version: str
    pair_uid: str
    action_rollout_uid: str
    control_rollout_uid: str
    control_type: str
    baseline_state_family_match: bool
    engine_version_match: bool
    parameter_family_match: bool
    fidelity_match: bool
    time_grid_match: bool
    seed_policy_compatible: bool
    state_namespace_match: bool
    support_domain_match: bool
    pairing_decision: str
    intervention_effect_eligible: bool
    blockers: Sequence[Mapping[str, object]]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualContrast:
    schema_version: str
    contrast_uid: str
    action_rollout_uids: Sequence[str]
    control_rollout_uids: Sequence[str]
    pairing_decision: str
    outcome_namespace: str
    contrast_estimator: str
    time_local_contrasts: Sequence[float]
    terminal_contrast: float
    integrated_contrast: float
    uncertainty: Mapping[str, object]
    support_domain_ref: str
    extrapolation_flags: Sequence[str]
    conflicts: Sequence[str]
    eligibility: str
    causal_effect_claim: bool
    intervention_effect_claim: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutUncertainty:
    schema_version: str
    uncertainty_uid: str
    rollout_uids: Sequence[str]
    components: Mapping[str, Mapping[str, object]]
    point_estimate: Mapping[str, object]
    interval_or_distribution: Mapping[str, object]
    aggregation_policy: str
    unknown_components: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutConstraintReport:
    schema_version: str
    constraint_report_uid: str
    rollout_uid: str
    constraint_policy: str
    required_constraints: Sequence[str]
    passed_constraints: Sequence[str]
    failed_constraints: Sequence[str]
    unknown_constraints: Sequence[str]
    violation_times: Mapping[str, Sequence[float]]
    violation_magnitudes: Mapping[str, Sequence[float]]
    handling_state: str
    formal_contrast_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutReliabilityReport:
    schema_version: str
    reliability_uid: str
    engine_uid: str
    support_domain_ref: str
    action_class: str
    diagnostics: Mapping[str, Mapping[str, object]]
    qualified_horizon: Mapping[str, object]
    maximum_tested_horizon: Mapping[str, object]
    horizon_decision: str
    support_domain_decision: str
    disagreement_outcome: str
    status: str
    restrictions: Sequence[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WhatIfEligibilityDecision:
    schema_version: str
    decision_uid: str
    result_uid: str
    controller_uid: str
    validator_uid: str
    roles: Mapping[str, bool]
    engine_status: str
    horizon_decision: str
    pairing_decision: str
    support_domain_decision: str
    constraint_state: str
    uncertainty_status: str
    reliability_status: str
    exact_execution_identity: bool
    reason_codes: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WhatIfResult:
    schema_version: str
    result_uid: str
    request_uid: str
    rollout_uids: Sequence[str]
    control_pair_uid: str
    contrast_uid: str
    uncertainty_uid: str
    reliability_uid: str
    eligibility_decision_uid: str
    result_type: str
    claim_scope: str
    limitations: Sequence[str]
    model_based_counterfactual: bool
    causal_effect_claim: bool
    clinical_recommendation: bool
    empirical_observation: bool
    user_facing_summary: str
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def round16_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("round16_what_if_engine")
    if not isinstance(value, Mapping):
        raise KeyError("round16_what_if_engine")
    return dict(value)


def validate_round16_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    try:
        value = round16_config(config)
    except KeyError:
        return [blocker("missing_round16_config", "Round-16 what-if configuration is absent.")]
    blockers: list[dict[str, object]] = []
    if value.get("version") != "neuronsifter-round16-what-if-engine-v1":
        blockers.append(blocker("invalid_round16_config_version", "Round-16 configuration version is not canonical."))
    if value.get("evidence_fabric_sealed") is not False:
        blockers.append(blocker("evidence_fabric_sealed", "Round-16 must leave the Evidence Fabric unsealed."))
    if value.get("dt_core_status") != "not_evaluated":
        blockers.append(blocker("dt_core_evaluated", "Round-16 must not evaluate DT_CORE."))
    if value.get("canonical_controller_uid") != "neuronsifter_orchestration_controller":
        blockers.append(blocker("noncanonical_formal_authority", "Only the canonical controller may authorize a formal result."))
    for name, raw_path in _mapping(value.get("output_paths")).items():
        if not _text(raw_path) or str(raw_path).startswith("/"):
            blockers.append(blocker("invalid_output_path", "Round-16 output paths must be relative.", output=name))
    return sorted_blockers(blockers)


def validate_what_if_request(value: Mapping[str, object]) -> list[dict[str, object]]:
    request = dict(value)
    uid = _text(request.get("request_uid"))
    blockers = _required(
        request,
        (
            "request_uid", "research_question_uid", "baseline_state_ref", "candidate_world_action_ref",
            "control_action_ref", "requested_outcomes", "requested_time_horizon", "requested_time_grid",
            "requested_regions", "requested_compartments", "requested_feature_namespace", "rollout_engine_policy",
            "ensemble_policy", "fidelity_policy", "uncertainty_requirements", "constraint_requirements",
            "support_domain_ref", "budget", "provenance",
        ),
        uid,
    )
    if request.get("schema_version") != SCHEMA_VERSIONS["what_if_request"]:
        blockers.append(blocker("unknown_what_if_request_schema", "What-if request schema is not canonical.", request_uid=uid))
    for field, expected in (
        ("model_based_counterfactual", True),
        ("causal_effect_claim", False),
        ("clinical_recommendation", False),
        ("empirical_observation", False),
    ):
        if request.get(field) is not expected:
            code = "clinical_recommendation_request" if field == "clinical_recommendation" else "invalid_what_if_boundary"
            blockers.append(blocker(code, f"{field} must be {str(expected).lower()}.", request_uid=uid))
    horizon = _mapping(request.get("requested_time_horizon"))
    if not all(key in horizon for key in ("start", "end", "unit")) or horizon.get("end", 0) <= horizon.get("start", 0):
        blockers.append(blocker("unsupported_horizon", "Requested horizon must be explicit and increasing.", request_uid=uid))
    return sorted_blockers(blockers)


def validate_baseline(value: Mapping[str, object], *, known_support_domains: set[str] | None = None) -> list[dict[str, object]]:
    baseline = dict(value)
    uid = _text(baseline.get("baseline_uid"))
    blockers = _required(
        baseline,
        (
            "baseline_uid", "state_ref", "state_hash", "state_uncertainty", "biological_or_model_context",
            "region", "compartment", "feature_namespace", "support_domain_ref", "source_role", "qc_grade",
            "missingness", "provenance", "baseline_class", "imputation_policy",
        ),
        uid,
    )
    if baseline.get("schema_version") != SCHEMA_VERSIONS["what_if_baseline"]:
        blockers.append(blocker("unknown_baseline_schema", "What-if baseline schema is not canonical.", baseline_uid=uid))
    if not is_sha256(baseline.get("state_hash")):
        blockers.append(blocker("missing_baseline", "Baseline state requires an exact content hash.", baseline_uid=uid))
    if _text(baseline.get("baseline_class")) not in BASELINE_CLASSES:
        blockers.append(blocker("missing_baseline", "Baseline class is unsupported.", baseline_uid=uid))
    if baseline.get("imputation_policy") != "none_all_channels_explicit":
        blockers.append(blocker("silent_baseline_imputation", "Missing baseline channels cannot be silently imputed.", baseline_uid=uid))
    if baseline.get("baseline_class") == "qualification_synthetic" and not (
        baseline.get("empirical_state") is False
        and baseline.get("qualification_only") is True
        and baseline.get("eligible_for_biological_claims") is False
    ):
        blockers.append(blocker("synthetic_baseline_boundary_invalid", "Synthetic baselines are qualification-only.", baseline_uid=uid))
    domain = _text(baseline.get("support_domain_ref"))
    if known_support_domains is not None and domain not in known_support_domains:
        blockers.append(blocker("unknown_support_domain", "Baseline support domain is not registered.", baseline_uid=uid))
    return sorted_blockers(blockers)


def validate_action_binding(value: Mapping[str, object], *, known_support_domains: set[str] | None = None) -> list[dict[str, object]]:
    binding = dict(value)
    uid = _text(binding.get("binding_uid"))
    blockers = _required(
        binding,
        (
            "binding_uid", "request_uid", "world_action", "action_program_hash", "parameter_provenance",
            "support_domain_ref", "spatial_scope", "temporal_scope", "completeness", "status",
        ),
        uid,
    )
    if binding.get("schema_version") != SCHEMA_VERSIONS["what_if_action_binding"]:
        blockers.append(blocker("unknown_action_binding_schema", "What-if action binding schema is not canonical.", binding_uid=uid))
    action = _mapping(binding.get("world_action"))
    action_blockers = validate_world_action(action)
    blockers.extend(action_blockers)
    if not action or binding.get("completeness") != "complete" or action_blockers:
        blockers.append(blocker("incomplete_action", "A complete canonical WorldAction is required.", binding_uid=uid))
    provenance = _mapping(binding.get("parameter_provenance"))
    hash_matches = binding.get("action_program_hash") == canonical_sha256(action)
    parent_hash_matches = (
        provenance.get("parent_action_program_hash") == binding.get("action_program_hash")
        and provenance.get("world_action_hash") == canonical_sha256(action)
    )
    if not (hash_matches or parent_hash_matches):
        blockers.append(blocker("action_program_hash_mismatch", "Action program hash must bind the complete action.", binding_uid=uid))
    if provenance.get("status") not in {"verified_exact_configuration", "verified_governed_artifact"} or not is_sha256(provenance.get("content_hash")):
        blockers.append(blocker("unknown_action_parameter_provenance", "Action parameters require exact provenance.", binding_uid=uid))
    if binding.get("support_domain_ref") != action.get("support_domain_ref"):
        blockers.append(blocker("support_domain_mismatch", "Action and binding support domains must match.", binding_uid=uid))
    domain = _text(binding.get("support_domain_ref"))
    if known_support_domains is not None and domain not in known_support_domains:
        blockers.append(blocker("unknown_support_domain", "Action support domain is not registered.", binding_uid=uid))
    if action.get("intervention_type") == "combination" and (
        not _list(binding.get("combination_components")) or not _list(binding.get("interaction_assumptions"))
    ):
        blockers.append(blocker("incomplete_combination_action", "Combination components and assumptions are required.", binding_uid=uid))
    if any(row.get("code") in {"missing_dose", "missing_dose_unit", "unsupported_route", "unresolved_target_identity"} for row in action_blockers):
        blockers.append(blocker("compound_id_not_complete_action", "A compound ID alone is not a complete action.", binding_uid=uid))
    return sorted_blockers(blockers)


def validate_engine(value: Mapping[str, object], *, known_support_domains: set[str] | None = None) -> list[dict[str, object]]:
    engine = dict(value)
    uid = _text(engine.get("engine_uid"))
    blockers = _required(
        engine,
        (
            "engine_uid", "engine_type", "version", "source", "code_identity", "environment_identity",
            "supported_baseline_classes", "supported_action_classes", "supported_state_variables",
            "supported_horizons", "supported_fidelities", "support_domain_ref", "uncertainty_capabilities",
            "constraint_capabilities", "qualification_evidence", "known_limitations", "status",
        ),
        uid,
    )
    if engine.get("schema_version") != SCHEMA_VERSIONS["rollout_engine"]:
        blockers.append(blocker("unknown_rollout_engine_schema", "Rollout engine schema is not canonical.", engine_uid=uid))
    if _text(engine.get("engine_type")) not in ENGINE_TYPES:
        blockers.append(blocker("unregistered_rollout_engine", "Unknown rollout engines fail closed.", engine_uid=uid))
    if _text(engine.get("version")).lower() in {"", "unknown", "latest"}:
        blockers.append(blocker("unknown_rollout_engine_version", "Rollout engine version must be exact.", engine_uid=uid))
    if "llm" in (_text(engine.get("engine_type")) + _text(engine.get("source"))).lower():
        blockers.append(blocker("llm_rollout_engine_prohibited", "An LLM is not a rollout engine.", engine_uid=uid))
    if _text(engine.get("status")) not in ENGINE_STATES:
        blockers.append(blocker("invalid_engine_status", "Engine eligibility state is not canonical.", engine_uid=uid))
    if engine.get("engine_type") == "learned_world_action_model" and engine.get("trained") is not True and str(engine.get("status", "")).startswith("qualified"):
        blockers.append(blocker("untrained_model_declared_qualified", "An untrained learned model cannot be qualified.", engine_uid=uid))
    if not is_sha256(engine.get("code_identity")) or not is_sha256(engine.get("environment_identity")):
        blockers.append(blocker("engine_identity_incomplete", "Engine code and environment identities require hashes.", engine_uid=uid))
    domain = _text(engine.get("support_domain_ref"))
    if known_support_domains is not None and domain not in known_support_domains:
        blockers.append(blocker("blocked_support_domain", "Engine support domain is not registered.", engine_uid=uid))
    return sorted_blockers(blockers)


def validate_engine_compatibility(
    *, request: Mapping[str, object], baseline: Mapping[str, object], action_binding: Mapping[str, object], engine: Mapping[str, object]
) -> tuple[str, str, list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    action = _mapping(action_binding.get("world_action"))
    if baseline.get("baseline_class") not in _list(engine.get("supported_baseline_classes")):
        blockers.append(blocker("unsupported_baseline_class", "Engine does not support this baseline class."))
    if action.get("intervention_type") not in _list(engine.get("supported_action_classes")):
        blockers.append(blocker("unsupported_action_class", "Engine does not support this action class."))
    requested = set(_list(request.get("requested_outcomes")))
    if not requested.issubset(set(_list(engine.get("supported_state_variables")))):
        blockers.append(blocker("unsupported_state_namespace", "Engine does not support all requested outcomes."))
    if request.get("support_domain_ref") != engine.get("support_domain_ref") or baseline.get("support_domain_ref") != engine.get("support_domain_ref"):
        blockers.append(blocker("support_domain_silently_expanded", "Request, baseline, and engine domains must match exactly."))
    requested_horizon = _mapping(request.get("requested_time_horizon"))
    horizons = [_mapping(row) for row in _list(engine.get("supported_horizons"))]
    in_horizon = any(
        requested_horizon.get("unit") == row.get("unit")
        and requested_horizon.get("start", 0) >= row.get("start", 0)
        and requested_horizon.get("end", 0) <= row.get("end", -1)
        for row in horizons
    )
    if not in_horizon:
        blockers.append(blocker("outside_qualified_horizon", "Requested horizon is outside qualification."))
    horizon = "within_qualified_horizon" if in_horizon else "outside_qualified_horizon"
    support = "restricted_in_domain" if not any(row["code"] == "support_domain_silently_expanded" for row in blockers) else "unsupported"
    return horizon, support, sorted_blockers(blockers)


def validate_rollout_request(value: Mapping[str, object]) -> list[dict[str, object]]:
    request = dict(value)
    uid = _text(request.get("rollout_request_uid"))
    blockers = _required(
        request,
        (
            "rollout_request_uid", "what_if_request_uid", "engine_uid", "engine_version", "baseline_hash",
            "action_program_hash", "control_program_hash", "parameter_hash", "fidelity", "time_grid",
            "constraint_policy", "uncertainty_policy", "exact_code_commit", "environment_hash", "input_bundle_hash",
        ),
        uid,
    )
    if request.get("schema_version") != SCHEMA_VERSIONS["rollout_request"]:
        blockers.append(blocker("unknown_rollout_request_schema", "Rollout request schema is not canonical.", request_uid=uid))
    for field in ("baseline_hash", "action_program_hash", "control_program_hash", "parameter_hash", "environment_hash", "input_bundle_hash"):
        if not is_sha256(request.get(field)):
            blockers.append(blocker("execution_identity_incomplete", "Rollout request hashes must be exact.", field=field, request_uid=uid))
    if not re.fullmatch(r"[a-f0-9]{40}", _text(request.get("exact_code_commit"))):
        blockers.append(blocker("execution_identity_incomplete", "Exact code commit is required.", request_uid=uid))
    return sorted_blockers(blockers)


def validate_attempts(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    for attempt in values:
        uid = _text(attempt.get("attempt_uid"))
        if uid in seen:
            blockers.append(blocker("failed_attempt_overwritten", "Attempt UIDs are immutable and unique.", attempt_uid=uid))
        seen.add(uid)
        if attempt.get("schema_version") != SCHEMA_VERSIONS["rollout_attempt"]:
            blockers.append(blocker("unknown_rollout_attempt_schema", "Rollout attempt schema is not canonical.", attempt_uid=uid))
        if attempt.get("preserved") is not True:
            blockers.append(blocker("failed_attempt_overwritten", "Every attempt must remain preserved.", attempt_uid=uid))
        hashes = _mapping(attempt.get("raw_output_hashes"))
        if attempt.get("terminal_state") == "COMPLETED" and (attempt.get("exit_code") != "0:0" or not hashes or any(not is_sha256(v) for v in hashes.values())):
            blockers.append(blocker("rollout_attempt_identity_invalid", "Completed attempts require successful exit and raw hashes.", attempt_uid=uid))
        if attempt.get("terminal_state") != "COMPLETED" and not _text(attempt.get("failure_class")):
            blockers.append(blocker("rollout_failure_unclassified", "Failed attempts require a failure class.", attempt_uid=uid))
    return sorted_blockers(blockers)


def validate_rollout_trajectory(value: Mapping[str, object]) -> list[dict[str, object]]:
    rollout = dict(value)
    uid = _text(rollout.get("rollout_uid"))
    blockers = _required(
        rollout,
        (
            "rollout_uid", "request_uid", "attempt_uid", "engine_uid", "baseline_uid", "action_uid",
            "control_pair_uid", "native_time_grid", "canonical_time_grid", "state_namespace", "state_artifact_ref",
            "uncertainty_artifact_ref", "constraint_artifact_ref", "content_hashes", "status", "eligibility", "provenance",
        ),
        uid,
    )
    if rollout.get("schema_version") != SCHEMA_VERSIONS["rollout_trajectory"]:
        blockers.append(blocker("unknown_rollout_schema", "Rollout schema is not canonical.", rollout_uid=uid))
    if rollout.get("output_class") != "model_rollout" or rollout.get("empirical_observation") is not False:
        blockers.append(blocker("model_rollout_treated_as_empirical_observation", "Rollouts must remain non-empirical model outputs.", rollout_uid=uid))
    hashes = _mapping(rollout.get("content_hashes"))
    if not hashes or any(not is_sha256(value) for value in hashes.values()):
        blockers.append(blocker("rollout_content_hash_missing", "Rollout artifacts require exact hashes.", rollout_uid=uid))
    return sorted_blockers(blockers)


def validate_control_pair(value: Mapping[str, object]) -> list[dict[str, object]]:
    pair = dict(value)
    uid = _text(pair.get("pair_uid"))
    blockers = _required(pair, ("pair_uid", "action_rollout_uid", "control_rollout_uid", "control_type", "pairing_decision", "status"), uid)
    if pair.get("schema_version") != SCHEMA_VERSIONS["rollout_control_pair"]:
        blockers.append(blocker("unknown_control_pair_schema", "Control pair schema is not canonical.", pair_uid=uid))
    if pair.get("control_type") not in CONTROL_TYPES or pair.get("pairing_decision") not in PAIRING_DECISIONS:
        blockers.append(blocker("invalid_control_pair", "Control type and pairing decision must be canonical.", pair_uid=uid))
    match_fields = (
        "baseline_state_family_match", "engine_version_match", "parameter_family_match", "fidelity_match",
        "time_grid_match", "seed_policy_compatible", "state_namespace_match", "support_domain_match",
    )
    complete = all(pair.get(field) is True for field in match_fields)
    if pair.get("pairing_decision") == "unpaired_descriptive_rollout" and pair.get("intervention_effect_eligible") is True:
        blockers.append(blocker("unpaired_rollout_described_as_intervention_effect", "Unpaired rollout is descriptive only.", pair_uid=uid))
    if pair.get("pairing_decision") != "unpaired_descriptive_rollout" and not complete:
        blockers.append(blocker("control_pair_mismatch", "Matched controls must share all governed identities.", pair_uid=uid))
    return sorted_blockers(blockers)


def validate_constraint_report(value: Mapping[str, object]) -> list[dict[str, object]]:
    report = dict(value)
    blockers: list[dict[str, object]] = []
    if report.get("schema_version") != SCHEMA_VERSIONS["rollout_constraint_report"]:
        blockers.append(blocker("unknown_constraint_report_schema", "Constraint report schema is not canonical."))
    if report.get("handling_state") not in CONSTRAINT_STATES:
        blockers.append(blocker("invalid_constraint_state", "Constraint handling state is not canonical."))
    if (_list(report.get("failed_constraints")) or _list(report.get("unknown_constraints"))) and report.get("formal_contrast_eligible") is True:
        blockers.append(blocker("constraint_failed_rollout_used_in_contrast", "Required constraint failures block formal contrast."))
    return sorted_blockers(blockers)


def validate_uncertainty(value: Mapping[str, object]) -> list[dict[str, object]]:
    uncertainty = dict(value)
    blockers: list[dict[str, object]] = []
    if uncertainty.get("schema_version") != SCHEMA_VERSIONS["rollout_uncertainty"]:
        blockers.append(blocker("unknown_uncertainty_schema", "Rollout uncertainty schema is not canonical."))
    components = _mapping(uncertainty.get("components"))
    for name in UNCERTAINTY_COMPONENTS:
        component = _mapping(components.get(name))
        if not component:
            blockers.append(blocker("uncertainty_component_missing", "Every uncertainty component must be explicit.", component=name))
            continue
        if component.get("status") == "unknown" and ("value" in component or component.get("represented_as_zero") is True):
            blockers.append(blocker("unknown_uncertainty_replaced_by_zero", "Unknown uncertainty must remain unknown.", component=name))
    if not _mapping(uncertainty.get("point_estimate")) or not _mapping(uncertainty.get("interval_or_distribution")):
        blockers.append(blocker("uncertainty_output_incomplete", "Point estimate and interval/distribution are required."))
    return sorted_blockers(blockers)


def validate_reliability(value: Mapping[str, object]) -> list[dict[str, object]]:
    report = dict(value)
    blockers: list[dict[str, object]] = []
    if report.get("schema_version") != SCHEMA_VERSIONS["rollout_reliability_report"]:
        blockers.append(blocker("unknown_reliability_schema", "Reliability report schema is not canonical."))
    required = {
        "repeatability", "seed_stability", "parameter_sensitivity", "fidelity_consistency", "horizon_degradation",
        "constraint_violation_rate", "model_family_disagreement", "baseline_perturbation_sensitivity",
        "action_perturbation_sensitivity", "support_domain_distance",
    }
    if not required.issubset(_mapping(report.get("diagnostics"))):
        blockers.append(blocker("reliability_diagnostics_incomplete", "All reliability dimensions must be preserved."))
    if report.get("horizon_decision") not in HORIZON_DECISIONS:
        blockers.append(blocker("horizon_qualification_missing", "Horizon decision must be explicit."))
    if report.get("support_domain_decision") not in SUPPORT_DECISIONS:
        blockers.append(blocker("support_domain_decision_missing", "Support-domain decision must be explicit."))
    if report.get("status") not in RELIABILITY_STATES:
        blockers.append(blocker("reliability_status_invalid", "Reliability status is not canonical."))
    if report.get("disagreement_outcome") not in {"consistent", "context_explained", "retained_disagreement", "insufficient_comparison"}:
        blockers.append(blocker("engine_disagreement_discarded", "Engine disagreement must be retained or explicitly explained."))
    return sorted_blockers(blockers)


def validate_contrast(value: Mapping[str, object], *, constraint_state: str) -> list[dict[str, object]]:
    contrast = dict(value)
    blockers: list[dict[str, object]] = []
    if contrast.get("schema_version") != SCHEMA_VERSIONS["counterfactual_contrast"]:
        blockers.append(blocker("unknown_contrast_schema", "Contrast schema is not canonical."))
    if contrast.get("contrast_estimator") not in CONTRAST_ESTIMATORS:
        blockers.append(blocker("unsupported_contrast_estimator", "Contrast estimator is not registered."))
    if contrast.get("causal_effect_claim") is not False:
        blockers.append(blocker("causal_effect_language_emitted", "Model contrasts are explicitly non-causal."))
    if contrast.get("pairing_decision") == "unpaired_descriptive_rollout" and contrast.get("intervention_effect_claim") is True:
        blockers.append(blocker("unpaired_rollout_described_as_intervention_effect", "Unpaired rollout cannot claim an intervention effect."))
    if constraint_state not in {"valid", "valid_with_tolerance"} and contrast.get("eligibility") != "ineligible":
        blockers.append(blocker("constraint_failed_rollout_used_in_contrast", "Constraint-failed rollout cannot enter contrast."))
    return sorted_blockers(blockers)


def validate_eligibility(value: Mapping[str, object], *, canonical_controller_uid: str) -> list[dict[str, object]]:
    decision = dict(value)
    blockers: list[dict[str, object]] = []
    if decision.get("schema_version") != SCHEMA_VERSIONS["what_if_eligibility_decision"]:
        blockers.append(blocker("unknown_eligibility_schema", "Eligibility schema is not canonical."))
    if decision.get("controller_uid") != canonical_controller_uid or decision.get("validator_uid") != "round16_governed_what_if_engine_validator":
        blockers.append(blocker("noncanonical_component_authorizes_formal_result", "Only the canonical controller through the approved validator may authorize roles."))
    roles = _mapping(decision.get("roles"))
    if set(roles) != set(ELIGIBILITY_ROLES) or any(not isinstance(value, bool) for value in roles.values()):
        blockers.append(blocker("eligibility_roles_incomplete", "Every what-if eligibility role must be explicit."))
    for field in ("eligible_for_empirical_validation", "eligible_for_causal_claim", "eligible_for_clinical_recommendation"):
        if roles.get(field) is not False:
            blockers.append(blocker("prohibited_eligibility_upgrade", f"{field} must remain false."))
    if roles.get("eligible_for_formal_model_based_claim") is True and not (
        str(decision.get("engine_status", "")).startswith("qualified_for_formal")
        and decision.get("horizon_decision") == "within_qualified_horizon"
        and decision.get("pairing_decision") != "unpaired_descriptive_rollout"
        and decision.get("support_domain_decision") in {"in_domain", "restricted_in_domain"}
        and decision.get("constraint_state") in {"valid", "valid_with_tolerance"}
        and decision.get("uncertainty_status") == "reported"
        and decision.get("reliability_status") in {"qualified", "qualified_with_restrictions"}
        and decision.get("exact_execution_identity") is True
    ):
        blockers.append(blocker("formal_model_claim_not_qualified", "Formal model claims require every qualification condition."))
    return sorted_blockers(blockers)


def validate_result(value: Mapping[str, object]) -> list[dict[str, object]]:
    result = dict(value)
    blockers: list[dict[str, object]] = []
    if result.get("schema_version") != SCHEMA_VERSIONS["what_if_result"]:
        blockers.append(blocker("unknown_what_if_result_schema", "What-if result schema is not canonical."))
    if result.get("model_based_counterfactual") is not True or result.get("empirical_observation") is not False:
        blockers.append(blocker("model_rollout_treated_as_empirical_observation", "What-if results are non-empirical model outputs."))
    if result.get("causal_effect_claim") is not False:
        blockers.append(blocker("causal_effect_language_emitted", "What-if results cannot claim causal effects."))
    if result.get("clinical_recommendation") is not False:
        blockers.append(blocker("clinical_recommendation_emitted", "What-if results cannot make clinical recommendations."))
    text = " ".join(_text(result.get(field)) for field in ("result_type", "claim_scope", "user_facing_summary"))
    text += " " + " ".join(_text(item) for item in _list(result.get("limitations")))
    if any(pattern.search(text) for pattern in PROHIBITED_CLAIM_PATTERNS):
        blockers.append(blocker("prohibited_claim_language", "What-if result contains prohibited causal or clinical language."))
    return sorted_blockers(blockers)


def validate_gap_map(rows: Sequence[Mapping[str, object]], *, expected_gap_count: int = 20) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if len(rows) != expected_gap_count or len({_text(row.get("gap_uid")) for row in rows}) != expected_gap_count:
        blockers.append(blocker("frozen_gap_count_drift", "Exactly 20 unique frozen gaps must be preserved."))
    for row in rows:
        if row.get("round16_status") not in FROZEN_GAP_STATUSES:
            blockers.append(blocker("invalid_frozen_gap_status", "Round-16 gap status is not canonical.", gap_uid=row.get("gap_uid")))
        if str(row.get("resolved_by_model_rollout", "")).lower() not in {"false", "0"}:
            blockers.append(blocker("model_result_closes_empirical_frozen_gap", "Model rollouts cannot resolve empirical gaps.", gap_uid=row.get("gap_uid")))
    return sorted_blockers(blockers)


def validate_unique_uids(objects: Sequence[Mapping[str, object]], uid_fields: Sequence[str]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in objects:
        uid = next((_text(value.get(field)) for field in uid_fields if _text(value.get(field))), "")
        if not uid or uid in seen:
            blockers.append(blocker("duplicate_uid", "Object UIDs must be present and globally unique.", object_uid=uid))
        seen.add(uid)
    return sorted_blockers(blockers)


def validate_lineage_acyclic(edges: Sequence[tuple[str, str]]) -> list[dict[str, object]]:
    graph: dict[str, set[str]] = {}
    for parent, child in edges:
        graph.setdefault(parent, set()).add(child)
        graph.setdefault(child, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        if not all(visit(child) for child in graph[node]):
            return False
        visiting.remove(node)
        visited.add(node)
        return True

    return [] if all(visit(node) for node in graph) else [blocker("lineage_cycle", "What-if lineage must be acyclic.")]


def release_content_hash(value: Mapping[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key not in {"release_uid", "release_content_hash", "created_at"}}
    return canonical_sha256(payload)


def validate_release_candidate(value: Mapping[str, object]) -> list[dict[str, object]]:
    release = dict(value)
    blockers: list[dict[str, object]] = []
    if release.get("schema_version") != SCHEMA_VERSIONS["release"]:
        blockers.append(blocker("unknown_release_schema", "Round-16 release schema is not canonical."))
    if release.get("sealed") is not False or release.get("status") != "materialized_unsealed":
        blockers.append(blocker("evidence_fabric_sealed", "Round-16 release must remain unsealed."))
    if release.get("release_uid") == release.get("parent_release_uid"):
        blockers.append(blocker("release_uid_not_versioned", "Round-16 must create a successor release UID."))
    if release.get("release_content_hash") != release_content_hash(release):
        blockers.append(blocker("release_content_hash_mismatch", "Release candidate content hash is invalid."))
    if release.get("empirical_observation_count") != 0 or release.get("empirical_gap_resolution_count") != 0:
        blockers.append(blocker("model_output_promoted_to_empirical_evidence", "Model outputs cannot enter empirical evidence classes."))
    if release.get("dt_core_status") != "not_evaluated":
        blockers.append(blocker("dt_core_evaluated", "Round-16 must not evaluate DT_CORE."))
    return sorted_blockers(blockers)


__all__ = [
    "BASELINE_CLASSES", "CONSTRAINT_STATES", "CONTRAST_ESTIMATORS", "ControlRollout", "ELIGIBILITY_ROLES",
    "ENGINE_STATES", "ENGINE_TYPES", "FROZEN_GAP_STATUSES", "HORIZON_DECISIONS", "PAIRING_DECISIONS",
    "RELIABILITY_STATES", "RolloutAttempt", "RolloutConstraintReport", "RolloutEngine", "RolloutRequest",
    "RolloutReliabilityReport", "RolloutTrajectory", "RolloutUncertainty", "SCHEMA_VERSIONS", "SUPPORT_DECISIONS",
    "UNCERTAINTY_COMPONENTS", "WhatIfActionBinding", "WhatIfBaseline", "WhatIfEligibilityDecision", "WhatIfRequest",
    "WhatIfResult", "CounterfactualContrast", "canonical_sha256", "release_content_hash", "round16_config",
    "sorted_blockers", "stable_uid", "validate_action_binding", "validate_attempts", "validate_baseline",
    "validate_constraint_report", "validate_contrast", "validate_control_pair", "validate_eligibility", "validate_engine",
    "validate_engine_compatibility", "validate_gap_map", "validate_lineage_acyclic", "validate_release_candidate",
    "validate_reliability", "validate_result", "validate_rollout_request", "validate_rollout_trajectory",
    "validate_round16_config", "validate_uncertainty", "validate_unique_uids",
]
