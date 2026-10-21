"""Fail-closed contracts for the Round-17 Scientific Twin-in-the-Loop runtime.

Agents propose and explain. The non-LLM TwinKernel validates typed artifacts,
routes them through controller-owned Gates, and applies only explicitly approved
assimilation transitions. This module has no shell, network, model-training, or
scientific-tool execution entry point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import re
from typing import Any

from Libs.utils.data.action_observation_contracts import (
    conflicts_preserved,
    validate_assimilation_decision,
    validate_information_action,
    validate_tool_capability_match,
    validate_tool_request,
    validate_world_action,
)
from Libs.utils.data.trajectory_contracts import (
    blocker,
    canonical_sha256,
    is_sha256,
    sorted_blockers,
    stable_uid,
)
from Libs.utils.data.what_if_contracts import validate_what_if_request


CANONICAL_CONTROLLER_UID = "neuronsifter_orchestration_controller"
ROUND17_VALIDATOR_UID = "round17_scitil_twin_gated_agent_validator"

SCHEMA_VERSIONS = {
    "twin_belief_state": "neuronsifter-twin-belief-state-v1",
    "agent_identity": "neuronsifter-agent-identity-v1",
    "agent_proposal": "neuronsifter-agent-proposal-v1",
    "scientific_hypothesis": "neuronsifter-scientific-hypothesis-v1",
    "hypothesis_gate_decision": "neuronsifter-hypothesis-gate-decision-v1",
    "world_action_gate_decision": "neuronsifter-world-action-gate-decision-v1",
    "information_action_gate_decision": "neuronsifter-information-action-gate-decision-v1",
    "observation_gate_decision": "neuronsifter-observation-gate-decision-v1",
    "dual_action_decision": "neuronsifter-dual-action-decision-v1",
    "twin_debt": "neuronsifter-twin-debt-v1",
    "scientific_budget": "neuronsifter-scientific-budget-v1",
    "risk_budget": "neuronsifter-risk-budget-v1",
    "agent_loop_event": "neuronsifter-agent-loop-event-v1",
    "agent_loop_episode": "neuronsifter-agent-loop-episode-v1",
    "hypothesis_ledger_entry": "neuronsifter-hypothesis-ledger-entry-v1",
    "agent_explanation": "neuronsifter-agent-explanation-v1",
    "release": "neuronsifter-round17-scitil-release-candidate-v1",
}

AGENT_ROLES = frozenset({"ScientistAgent", "FalsifierAgent", "ExperimentAgent", "EvidenceAgent"})
PROPOSAL_TYPES = frozenset(
    {
        "new_hypothesis",
        "hypothesis_revision",
        "falsification_test",
        "world_action_candidate",
        "information_action_candidate",
        "what_if_candidate",
        "evidence_query",
        "conflict_investigation",
        "stop_recommendation",
    }
)
ROLE_PROPOSAL_TYPES = {
    "ScientistAgent": frozenset(
        {"new_hypothesis", "hypothesis_revision", "world_action_candidate", "what_if_candidate", "stop_recommendation"}
    ),
    "FalsifierAgent": frozenset({"falsification_test", "conflict_investigation", "stop_recommendation"}),
    "ExperimentAgent": frozenset({"information_action_candidate", "stop_recommendation"}),
    "EvidenceAgent": frozenset({"evidence_query", "conflict_investigation", "stop_recommendation"}),
}
ROLE_CAPABILITIES = {
    "ScientistAgent": frozenset(
        {"propose_hypothesis", "propose_mechanistic_explanation", "propose_world_action", "propose_what_if", "propose_outcomes"}
    ),
    "FalsifierAgent": frozenset(
        {"propose_disconfirming_prediction", "propose_alternative_mechanism", "propose_adversarial_control", "challenge_domain", "propose_constraint_check"}
    ),
    "ExperimentAgent": frozenset(
        {"propose_information_action", "propose_observation_request", "propose_tool_or_assay", "propose_fidelity", "propose_budget_option"}
    ),
    "EvidenceAgent": frozenset(
        {"retrieve_governed_evidence", "assemble_typed_evidence_refs", "identify_conflict", "identify_gap", "summarize_accepted_artifact"}
    ),
}

H_GATE_DECISIONS = frozenset(
    {
        "admitted_for_testing",
        "admitted_hypothesis_only",
        "revision_required",
        "insufficient_evidence",
        "not_falsifiable",
        "unsupported_domain",
        "duplicate_hypothesis",
        "rejected_provenance",
    }
)
W_GATE_DECISIONS = frozenset(
    {
        "authorized_restricted_rollout",
        "authorized_stress_rollout",
        "revision_required",
        "blocked_incomplete_action",
        "blocked_engine_qualification",
        "blocked_control_pairing",
        "blocked_horizon",
        "blocked_support_domain",
        "blocked_budget",
        "blocked_claim_boundary",
    }
)
I_GATE_DECISIONS = frozenset(
    {
        "authorized_tool_request",
        "authorized_evidence_query",
        "authorized_restricted_information_action",
        "revision_required",
        "unsupported_tool",
        "unsupported_observable",
        "blocked_license",
        "blocked_budget",
        "blocked_support_domain",
        "blocked_provenance",
    }
)
O_GATE_DECISIONS = frozenset(
    {
        "approved_state_update",
        "approved_parameter_update",
        "approved_model_weight_update",
        "approved_uncertainty_only",
        "approved_conflict_register_only",
        "no_update",
        "revision_required",
        "rejected_qc",
        "rejected_provenance",
        "rejected_support_domain",
        "rejected_conflict",
    }
)
O_GATE_APPROVALS = frozenset(
    {
        "approved_state_update",
        "approved_parameter_update",
        "approved_model_weight_update",
        "approved_uncertainty_only",
        "approved_conflict_register_only",
    }
)

DEBT_COMPONENTS = (
    "calibration_debt",
    "coverage_debt",
    "disagreement_debt",
    "provenance_debt",
    "constraint_debt",
)
DEBT_STATUSES = frozenset({"low", "moderate", "high", "unknown"})
RISK_CLASSES = frozenset(
    {"contract_only", "restricted_computational", "stress_only", "external_tool_read_only", "production_prohibited"}
)
LOOP_STATES = frozenset(
    {
        "initialized",
        "hypothesis_proposed",
        "hypothesis_gated",
        "action_candidates_generated",
        "action_gated",
        "execution_authorized",
        "execution_completed",
        "observation_normalized",
        "assimilation_proposed",
        "assimilation_gated",
        "belief_updated",
        "explanation_generated",
        "terminated",
        "blocked",
    }
)
TERMINAL_STATES = frozenset(
    {
        "belief_updated",
        "uncertainty_only_updated",
        "conflict_registered",
        "hypothesis_refuted",
        "hypothesis_retained",
        "new_information_required",
        "budget_exhausted",
        "unsupported_domain",
        "blocked_by_gate",
        "no_safe_action",
    }
)
ALLOWED_TRANSITIONS = frozenset(
    {
        ("initialized", "hypothesis_proposed"),
        ("initialized", "action_candidates_generated"),
        ("initialized", "blocked"),
        ("hypothesis_proposed", "hypothesis_gated"),
        ("hypothesis_gated", "action_candidates_generated"),
        ("hypothesis_gated", "blocked"),
        ("action_candidates_generated", "action_gated"),
        ("action_gated", "execution_authorized"),
        ("action_gated", "blocked"),
        ("execution_authorized", "execution_completed"),
        ("execution_completed", "observation_normalized"),
        ("observation_normalized", "assimilation_proposed"),
        ("observation_normalized", "explanation_generated"),
        ("assimilation_proposed", "assimilation_gated"),
        ("assimilation_gated", "belief_updated"),
        ("assimilation_gated", "explanation_generated"),
        ("assimilation_gated", "blocked"),
        ("belief_updated", "explanation_generated"),
        ("explanation_generated", "terminated"),
        ("blocked", "terminated"),
    }
)
HYPOTHESIS_STATUSES = frozenset(
    {
        "proposed",
        "admitted_for_testing",
        "under_test",
        "supported_model_conditionally",
        "refuted_in_model",
        "inconclusive",
        "requires_empirical_test",
        "unsupported_domain",
        "retired",
    }
)
FROZEN_GAP_IMPACTS = frozenset(
    {
        "no_effect",
        "hypothesis_generated",
        "information_action_proposed",
        "evidence_query_authorized",
        "requires_empirical_observation",
        "requires_new_e4",
        "requires_support_domain",
        "stress_only",
    }
)
PROHIBITED_EXECUTION_KEYS = frozenset(
    {
        "shell",
        "shell_command",
        "command",
        "python_code",
        "exec",
        "subprocess",
        "direct_state_update",
        "posterior_write",
        "gate_write",
        "gate_decision",
        "qc_grade",
        "qc_upgrade",
        "eligibility_upgrade",
        "biological_pairing",
        "pairing_decision",
        "conflict_resolution",
        "remove_conflict",
    }
)
PROHIBITED_CLAIM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwill cause\b",
        r"\bcausal(?:ly)? established\b",
        r"\btreatment is effective\b",
        r"\bpatient will respond\b",
        r"\bclinically recommended\b",
        r"\bclinical recommendation\b",
        r"\bproven\b",
        r"\bclinically validated\b",
    )
)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _present(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _required(value: Mapping[str, object], fields: Sequence[str], uid: str) -> list[dict[str, object]]:
    return [
        blocker("missing_required_field", f"{field} must be explicit.", object_uid=uid, field=field)
        for field in fields
        if not _present(value.get(field))
    ]


def _forbidden_paths(value: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in PROHIBITED_EXECUTION_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_forbidden_paths(item, f"{prefix}[{index}]"))
    return paths


@dataclass(frozen=True)
class TwinBeliefState:
    schema_version: str
    belief_state_uid: str
    parent_belief_state_uid: str | None
    state_posterior_ref: str
    parameter_posterior_ref: str
    model_family_posterior_ref: str
    evidence_snapshot_ref: str
    active_hypothesis_refs: Sequence[str]
    constraint_state_ref: str
    support_domain_ref: str
    twin_debt_ref: str
    content_hash: str
    controller_uid: str
    creation_decision_ref: str
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentIdentity:
    schema_version: str
    agent_uid: str
    agent_role: str
    capabilities: Sequence[str]
    prohibited_capabilities: Sequence[str]
    gate_writer: bool
    tool_executor: bool
    state_update_authority: bool
    canonical_controller: bool
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentProposal:
    schema_version: str
    proposal_uid: str
    agent_uid: str
    agent_role: str
    belief_state_ref: str
    proposal_type: str
    hypothesis_ref: str | None
    world_action_ref: str | None
    information_action_ref: str | None
    what_if_request_ref: str | None
    observation_request_ref: str | None
    expected_information_gain: Mapping[str, object]
    expected_twin_debt_change: Mapping[str, object]
    expected_cost: Mapping[str, object]
    risk_class: str
    support_domain_ref: str
    evidence_refs: Sequence[str]
    assumptions: Sequence[str]
    uncertainties: Sequence[str]
    prohibited_claim_flags: Mapping[str, bool]
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificHypothesis:
    schema_version: str
    hypothesis_uid: str
    mechanism_statement: str
    target_variables: Sequence[str]
    expected_outcome: Mapping[str, object]
    required_context: Mapping[str, object]
    time_horizon: Mapping[str, object]
    potential_falsifier: Mapping[str, object]
    supporting_evidence: Sequence[str]
    conflicting_evidence: Sequence[str]
    support_domain_ref: str
    uncertainty: Mapping[str, object]
    causal_effect_claim: bool
    clinical_efficacy_claim: bool
    hypothetical_as_empirical_support: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisGateDecision:
    schema_version: str
    gate_uid: str
    gate_type: str
    proposal_uid: str
    controller_uid: str
    validator_uid: str
    input_hashes: Mapping[str, str]
    decision: str
    reason_codes: Sequence[str]
    approved_scope: Mapping[str, object]
    budget_effect: Mapping[str, object]
    twin_debt_effect: Mapping[str, object]
    support_domain_effect: Mapping[str, object]
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorldActionGateDecision(HypothesisGateDecision):
    pass


@dataclass(frozen=True)
class InformationActionGateDecision(HypothesisGateDecision):
    pass


@dataclass(frozen=True)
class ObservationGateDecision(HypothesisGateDecision):
    pass


@dataclass(frozen=True)
class DualActionDecision:
    schema_version: str
    decision_uid: str
    belief_state_ref: str
    candidate_world_actions: Sequence[str]
    candidate_information_actions: Sequence[str]
    expected_utility: Mapping[str, object]
    expected_information_gain: Mapping[str, object]
    expected_twin_debt_change: Mapping[str, object]
    cost: Mapping[str, object]
    risk: Mapping[str, object]
    support_domain_ref: str
    selected_action_type: str
    selected_action_ref: str | None
    selection_reason: Sequence[str]
    controller_uid: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TwinDebt:
    schema_version: str
    debt_uid: str
    belief_state_ref: str
    component_values: Mapping[str, float | None]
    component_status: Mapping[str, str]
    measurement_basis: Mapping[str, object]
    unknown_components: Sequence[str]
    versioned_thresholds: Mapping[str, object]
    allowed_decisions: Sequence[str]
    aggregation: Mapping[str, object] | None
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificBudget:
    schema_version: str
    budget_uid: str
    dimensions: Mapping[str, float]
    consumed: Mapping[str, float]
    remaining: Mapping[str, float]
    iteration_limit: int
    controller_uid: str
    version: str
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskBudget:
    schema_version: str
    risk_budget_uid: str
    allowed_risk_classes: Sequence[str]
    maximum_risk_class: str
    production_execution_allowed: bool
    agent_may_increase_risk: bool
    controller_uid: str
    version: str
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentLoopBudgetDecision:
    decision_uid: str
    proposal_uid: str
    budget_uid: str
    risk_budget_uid: str
    controller_uid: str
    requested_cost: Mapping[str, float]
    requested_risk_class: str
    decision: str
    remaining_after: Mapping[str, float]
    reason_codes: Sequence[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentLoopEvent:
    schema_version: str
    event_uid: str
    episode_uid: str
    prior_state: str
    next_state: str
    agent_uid: str | None
    gate_uid: str | None
    controller_uid: str
    input_hashes: Mapping[str, str]
    output_hashes: Mapping[str, str]
    timestamp: str
    decision_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentLoopEpisode:
    schema_version: str
    episode_uid: str
    initial_belief_state_ref: str
    final_belief_state_ref: str | None
    event_uids: Sequence[str]
    scenario_uid: str
    terminal_state: str
    controller_uid: str
    budget_decision_refs: Sequence[str]
    conflict_refs: Sequence[str]
    frozen_gap_impacts: Mapping[str, str]
    replay_identity: Mapping[str, object]
    status: str
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisLedgerEntry:
    schema_version: str
    hypothesis_uid: str
    version: int
    parent_hypothesis: str | None
    mechanism: str
    expected_observations: Sequence[str]
    falsification_criteria: Sequence[str]
    supporting_evidence: Sequence[str]
    conflicting_evidence: Sequence[str]
    test_history: Sequence[str]
    current_status: str
    status_decision_ref: str
    support_score: float | None
    falsification_score: float | None
    unresolved_conflict_count: int
    reliability_status: str
    support_domain_status: str
    falsifier_agent_ref: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AgentExplanation:
    schema_version: str
    explanation_uid: str
    episode_uid: str
    belief_state_ref: str
    gate_decision_refs: Sequence[str]
    rollout_or_tool_artifact_refs: Sequence[str]
    uncertainty_report_ref: str
    reliability_report_ref: str
    support_domain_report_ref: str
    conflicts: Sequence[str]
    claim_eligibility_ref: str
    summary: str
    limitations: Sequence[str]
    model_based_counterfactual: bool
    empirical_observation: bool
    causal_effect_claim: bool
    clinical_recommendation: bool
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def round17_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("round17_scitil_runtime")
    if not isinstance(value, Mapping):
        raise KeyError("round17_scitil_runtime")
    return dict(value)


def validate_round17_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    try:
        value = round17_config(config)
    except KeyError:
        return [blocker("missing_round17_config", "Round-17 SciTiL configuration is absent.")]
    blockers: list[dict[str, object]] = []
    if value.get("version") != "neuronsifter-round17-scitil-runtime-v1":
        blockers.append(blocker("invalid_round17_config_version", "Round-17 configuration version is not canonical."))
    if value.get("canonical_controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("noncanonical_controller", "Only the canonical controller may own SciTiL Gates."))
    if value.get("validator_uid") != ROUND17_VALIDATOR_UID:
        blockers.append(blocker("unapproved_round17_validator", "Round-17 validator UID is not canonical."))
    if value.get("evidence_fabric_sealed") is not False:
        blockers.append(blocker("evidence_fabric_sealed", "Round-17 must leave the Evidence Fabric unsealed."))
    if value.get("dt_core_status") != "not_evaluated":
        blockers.append(blocker("dt_core_evaluated", "Round-17 must not evaluate DT_CORE."))
    if value.get("frozen_gap_count") != 20:
        blockers.append(blocker("frozen_gap_count_drift", "Round-17 must preserve 20 frozen gaps."))
    for name, raw_path in _mapping(value.get("output_paths")).items():
        if not _text(raw_path) or str(raw_path).startswith("/"):
            blockers.append(blocker("invalid_output_path", "Round-17 output paths must be relative.", output=name))
    return sorted_blockers(blockers)


def belief_state_content_hash(value: Mapping[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key not in {"belief_state_uid", "content_hash"}}
    return canonical_sha256(payload)


def validate_belief_state(value: Mapping[str, object], *, allow_initial: bool = False) -> list[dict[str, object]]:
    state = dict(value)
    uid = _text(state.get("belief_state_uid"))
    blockers = _required(
        state,
        (
            "belief_state_uid",
            "state_posterior_ref",
            "parameter_posterior_ref",
            "model_family_posterior_ref",
            "evidence_snapshot_ref",
            "constraint_state_ref",
            "support_domain_ref",
            "twin_debt_ref",
            "content_hash",
            "controller_uid",
            "creation_decision_ref",
            "provenance",
        ),
        uid,
    )
    if state.get("schema_version") != SCHEMA_VERSIONS["twin_belief_state"]:
        blockers.append(blocker("unknown_belief_state_schema", "TwinBeliefState schema is not canonical.", belief_state_uid=uid))
    if state.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("agent_directly_updates_twin_state", "Only the canonical controller may create a belief state.", belief_state_uid=uid))
    if state.get("content_hash") != belief_state_content_hash(state):
        blockers.append(blocker("belief_state_content_hash_mismatch", "TwinBeliefState content hash is invalid.", belief_state_uid=uid))
    if state.get("parent_belief_state_uid") is None and not allow_initial:
        blockers.append(blocker("belief_state_parent_missing", "Only an explicitly qualified initial state may omit a parent.", belief_state_uid=uid))
    provenance = _mapping(state.get("provenance"))
    if any(key in provenance for key in ("agent_text", "free_text_posterior", "llm_state")):
        blockers.append(blocker("agent_text_as_posterior_state", "Agent text cannot be stored as posterior state.", belief_state_uid=uid))
    return sorted_blockers(blockers)


def validate_agent_identity(value: Mapping[str, object]) -> list[dict[str, object]]:
    agent = dict(value)
    uid = _text(agent.get("agent_uid"))
    blockers = _required(agent, ("agent_uid", "agent_role", "capabilities", "prohibited_capabilities", "provenance"), uid)
    role = _text(agent.get("agent_role"))
    if agent.get("schema_version") != SCHEMA_VERSIONS["agent_identity"]:
        blockers.append(blocker("unknown_agent_identity_schema", "AgentIdentity schema is not canonical.", agent_uid=uid))
    if role not in AGENT_ROLES:
        blockers.append(blocker("unknown_agent_role", "Agent role is not registered.", agent_uid=uid))
    capabilities = {_text(item) for item in _list(agent.get("capabilities"))}
    if role in ROLE_CAPABILITIES and not capabilities.issubset(ROLE_CAPABILITIES[role]):
        blockers.append(blocker("agent_privilege_escalation", "Agent capabilities exceed its role.", agent_uid=uid))
    for field in ("gate_writer", "tool_executor", "state_update_authority", "canonical_controller"):
        if agent.get(field) is not False:
            code = "agent_writes_gate" if field == "gate_writer" else "agent_privilege_escalation"
            blockers.append(blocker(code, f"Agent {field} must remain false.", agent_uid=uid))
    return sorted_blockers(blockers)


def validate_agent_proposal(
    value: Mapping[str, object], *, agent: Mapping[str, object], known_belief_states: set[str] | None = None
) -> list[dict[str, object]]:
    proposal = dict(value)
    uid = _text(proposal.get("proposal_uid"))
    blockers = _required(
        proposal,
        (
            "proposal_uid",
            "agent_uid",
            "agent_role",
            "belief_state_ref",
            "proposal_type",
            "expected_information_gain",
            "expected_twin_debt_change",
            "expected_cost",
            "risk_class",
            "support_domain_ref",
            "evidence_refs",
            "assumptions",
            "uncertainties",
            "prohibited_claim_flags",
            "provenance",
        ),
        uid,
    )
    if proposal.get("schema_version") != SCHEMA_VERSIONS["agent_proposal"]:
        blockers.append(blocker("unknown_agent_proposal_schema", "AgentProposal schema is not canonical.", proposal_uid=uid))
    if proposal.get("agent_uid") != agent.get("agent_uid") or proposal.get("agent_role") != agent.get("agent_role"):
        blockers.append(blocker("agent_identity_mismatch", "Proposal does not bind its Agent identity.", proposal_uid=uid))
    role = _text(proposal.get("agent_role"))
    proposal_type = _text(proposal.get("proposal_type"))
    if proposal_type not in PROPOSAL_TYPES or proposal_type not in ROLE_PROPOSAL_TYPES.get(role, frozenset()):
        blockers.append(blocker("agent_role_privilege_violation", "Agent role cannot emit this proposal type.", proposal_uid=uid))
    if known_belief_states is not None and proposal.get("belief_state_ref") not in known_belief_states:
        blockers.append(blocker("unknown_belief_state", "Proposal references an unknown belief state.", proposal_uid=uid))
    if proposal.get("risk_class") not in RISK_CLASSES:
        blockers.append(blocker("unknown_risk_class", "Proposal risk class is invalid.", proposal_uid=uid))
    required_ref = {
        "new_hypothesis": "hypothesis_ref",
        "hypothesis_revision": "hypothesis_ref",
        "falsification_test": "hypothesis_ref",
        "world_action_candidate": "world_action_ref",
        "information_action_candidate": "information_action_ref",
        "what_if_candidate": "what_if_request_ref",
        "evidence_query": "observation_request_ref",
    }.get(proposal_type)
    if required_ref and not _present(proposal.get(required_ref)):
        blockers.append(blocker("proposal_typed_reference_missing", "Proposal is missing its typed artifact reference.", proposal_uid=uid, field=required_ref))
    claim_flags = _mapping(proposal.get("prohibited_claim_flags"))
    if claim_flags.get("causal_effect_claim") is not False or claim_flags.get("clinical_recommendation") is not False:
        blockers.append(blocker("agent_prohibited_claim", "Agent proposals cannot request causal or clinical claims.", proposal_uid=uid))
    forbidden = _forbidden_paths(proposal)
    if forbidden:
        blockers.append(blocker("agent_arbitrary_execution_request", "Agent proposal contains prohibited execution or authority fields.", proposal_uid=uid, paths=forbidden))
    return sorted_blockers(blockers)


def validate_hypothesis(value: Mapping[str, object]) -> list[dict[str, object]]:
    hypothesis = dict(value)
    uid = _text(hypothesis.get("hypothesis_uid"))
    blockers = _required(
        hypothesis,
        (
            "hypothesis_uid",
            "mechanism_statement",
            "target_variables",
            "expected_outcome",
            "required_context",
            "time_horizon",
            "potential_falsifier",
            "supporting_evidence",
            "conflicting_evidence",
            "support_domain_ref",
            "uncertainty",
        ),
        uid,
    )
    if hypothesis.get("schema_version") != SCHEMA_VERSIONS["scientific_hypothesis"]:
        blockers.append(blocker("unknown_hypothesis_schema", "ScientificHypothesis schema is not canonical.", hypothesis_uid=uid))
    falsifier = _mapping(hypothesis.get("potential_falsifier"))
    if not any(_present(falsifier.get(key)) for key in ("disconfirming_prediction", "alternative_mechanism", "negative_control", "domain_challenge", "reliability_challenge")):
        blockers.append(blocker("hypothesis_not_falsifiable", "Hypothesis requires an explicit falsifier.", hypothesis_uid=uid))
    if any(not _text(item) or _text(item).lower() in {"unknown", "undefined"} for item in _list(hypothesis.get("target_variables"))):
        blockers.append(blocker("hypothesis_undefined_variable", "Hypothesis target variables must be defined.", hypothesis_uid=uid))
    if hypothesis.get("causal_effect_claim") is not False:
        blockers.append(blocker("causal_hypothesis_without_identification", "Round-17 hypotheses cannot claim causality.", hypothesis_uid=uid))
    if hypothesis.get("clinical_efficacy_claim") is not False:
        blockers.append(blocker("clinical_efficacy_hypothesis", "Round-17 hypotheses cannot claim clinical efficacy.", hypothesis_uid=uid))
    if hypothesis.get("hypothetical_as_empirical_support") is not False:
        blockers.append(blocker("hypothetical_output_as_empirical_support", "Hypothetical output cannot be empirical support.", hypothesis_uid=uid))
    text = _text(hypothesis.get("mechanism_statement"))
    if any(pattern.search(text) for pattern in PROHIBITED_CLAIM_PATTERNS):
        blockers.append(blocker("prohibited_hypothesis_language", "Hypothesis contains prohibited claim language.", hypothesis_uid=uid))
    return sorted_blockers(blockers)


def _validate_gate(
    value: Mapping[str, object], *, schema_key: str, gate_type: str, decisions: frozenset[str], proposal_uid: str
) -> list[dict[str, object]]:
    gate = dict(value)
    uid = _text(gate.get("gate_uid"))
    blockers = _required(
        gate,
        (
            "gate_uid",
            "gate_type",
            "proposal_uid",
            "controller_uid",
            "validator_uid",
            "input_hashes",
            "decision",
            "reason_codes",
            "approved_scope",
            "budget_effect",
            "twin_debt_effect",
            "support_domain_effect",
            "provenance",
        ),
        uid,
    )
    if gate.get("schema_version") != SCHEMA_VERSIONS[schema_key] or gate.get("gate_type") != gate_type:
        blockers.append(blocker("gate_type_or_schema_invalid", "Gate type and schema must be canonical.", gate_uid=uid))
    if gate.get("controller_uid") != CANONICAL_CONTROLLER_UID or gate.get("validator_uid") != ROUND17_VALIDATOR_UID:
        blockers.append(blocker("agent_writes_gate", "Only the canonical controller through the approved validator may write a Gate.", gate_uid=uid))
    if gate.get("proposal_uid") != proposal_uid:
        blockers.append(blocker("gate_proposal_identity_mismatch", "Gate does not bind its proposal.", gate_uid=uid))
    hashes = _mapping(gate.get("input_hashes"))
    if not hashes or any(not is_sha256(item) for item in hashes.values()):
        blockers.append(blocker("gate_input_hash_invalid", "Gate inputs must be content-addressed.", gate_uid=uid))
    if gate.get("decision") not in decisions:
        blockers.append(blocker("unknown_gate_decision", "Gate decision is invalid.", gate_uid=uid))
    return sorted_blockers(blockers)


def validate_h_gate(value: Mapping[str, object], *, proposal: Mapping[str, object], hypothesis: Mapping[str, object]) -> list[dict[str, object]]:
    blockers = _validate_gate(
        value,
        schema_key="hypothesis_gate_decision",
        gate_type="H-Gate",
        decisions=H_GATE_DECISIONS,
        proposal_uid=_text(proposal.get("proposal_uid")),
    )
    if value.get("decision") in {"admitted_for_testing", "admitted_hypothesis_only"}:
        blockers += validate_hypothesis(hypothesis)
    return sorted_blockers(blockers)


def validate_w_gate(
    value: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    world_action: Mapping[str, object] | None,
    what_if_request: Mapping[str, object] | None,
    twin_debt: Mapping[str, object],
    budget_decision: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers = _validate_gate(
        value,
        schema_key="world_action_gate_decision",
        gate_type="W-Gate",
        decisions=W_GATE_DECISIONS,
        proposal_uid=_text(proposal.get("proposal_uid")),
    )
    authorized = value.get("decision") in {"authorized_restricted_rollout", "authorized_stress_rollout"}
    if authorized:
        if world_action is None or validate_world_action(world_action):
            blockers.append(blocker("w_gate_bypasses_action_contract", "W-Gate cannot authorize an incomplete WorldAction."))
        if what_if_request is not None and validate_what_if_request(what_if_request):
            blockers.append(blocker("w_gate_bypasses_what_if_contract", "W-Gate cannot authorize an invalid WhatIfRequest."))
        debt_statuses = set(_mapping(twin_debt.get("component_status")).values())
        if "high" in debt_statuses or "unknown" in debt_statuses:
            blockers.append(blocker("high_twin_debt_world_action_authorized", "High or unknown Twin Debt blocks world action authorization."))
        if budget_decision.get("decision") != "authorized":
            blockers.append(blocker("budget_exceeded_silently", "W-Gate requires an authorized budget decision."))
        if proposal.get("risk_class") not in {"contract_only", "restricted_computational", "stress_only"}:
            blockers.append(blocker("world_action_risk_not_allowed", "W-Gate cannot authorize production risk."))
        support_domains = {
            _text(item.get("support_domain_ref"))
            for item in (proposal, world_action or {}, what_if_request or {})
            if _present(item.get("support_domain_ref"))
        }
        if len(support_domains) != 1:
            blockers.append(blocker("w_gate_support_domain_mismatch", "W-Gate cannot silently expand support domain."))
    return sorted_blockers(blockers)


def validate_i_gate(
    value: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    information_action: Mapping[str, object],
    tool_request: Mapping[str, object] | None,
    allowlisted_adapters: set[str],
    budget_decision: Mapping[str, object],
    tool_capability_match: Mapping[str, object] | None = None,
    tool_availability_decision: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    blockers = _validate_gate(
        value,
        schema_key="information_action_gate_decision",
        gate_type="I-Gate",
        decisions=I_GATE_DECISIONS,
        proposal_uid=_text(proposal.get("proposal_uid")),
    )
    authorized = str(value.get("decision", "")).startswith("authorized_")
    if authorized:
        blockers += validate_information_action(information_action)
        adapter_uid = _text(_mapping(value.get("approved_scope")).get("adapter_uid"))
        if adapter_uid not in allowlisted_adapters:
            blockers.append(blocker("unsupported_tool", "I-Gate must bind an allowlisted adapter.", adapter_uid=adapter_uid))
        if proposal.get("support_domain_ref") != information_action.get("support_domain_ref"):
            blockers.append(blocker("i_gate_support_domain_mismatch", "I-Gate cannot silently expand support domain."))
        if tool_request is not None:
            request_validation = validate_tool_request(tool_request)
            blockers += list(request_validation.blockers)
            if tool_request.get("tool_uid") != adapter_uid:
                blockers.append(blocker("unsupported_tool", "I-Gate request and approved adapter identities differ.", adapter_uid=adapter_uid))
            if tool_request.get("information_action_uid") != information_action.get("information_action_uid"):
                blockers.append(blocker("i_gate_information_action_mismatch", "ToolRequest does not bind the InformationAction."))
            if tool_request.get("support_domain_ref") != information_action.get("support_domain_ref"):
                blockers.append(blocker("i_gate_support_domain_mismatch", "ToolRequest support domain differs from the InformationAction."))
            if tool_capability_match is None:
                blockers.append(blocker("tool_capability_match_missing", "Authorized ToolRequest requires a capability match."))
            else:
                blockers += validate_tool_capability_match(tool_capability_match)
                if (
                    tool_capability_match.get("tool_request_uid") != tool_request.get("tool_request_uid")
                    or tool_capability_match.get("tool_uid") != tool_request.get("tool_uid")
                    or tool_capability_match.get("state") not in {"supported", "supported_with_restrictions"}
                    or tool_capability_match.get("license_access_resolved") is not True
                ):
                    blockers.append(blocker("unsupported_tool", "Tool capability or license evidence does not authorize the request."))
            if tool_availability_decision is None or (
                tool_capability_match is not None
                and (
                    tool_availability_decision.get("match_uid") != tool_capability_match.get("match_uid")
                    or tool_availability_decision.get("available") is not True
                )
            ):
                blockers.append(blocker("tool_availability_not_authorized", "Authorized ToolRequest requires an available capability decision."))
        if budget_decision.get("decision") != "authorized":
            blockers.append(blocker("budget_exceeded_silently", "I-Gate requires an authorized budget decision."))
    return sorted_blockers(blockers)


def validate_o_gate(
    value: Mapping[str, object],
    *,
    proposal: Mapping[str, object],
    assimilation_proposal: Mapping[str, object],
    assimilation_decision: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers = _validate_gate(
        value,
        schema_key="observation_gate_decision",
        gate_type="O-Gate",
        decisions=O_GATE_DECISIONS,
        proposal_uid=_text(proposal.get("proposal_uid")),
    )
    if value.get("decision") in O_GATE_APPROVALS:
        blockers += validate_assimilation_decision(
            assimilation_decision,
            proposal=assimilation_proposal,
            canonical_controller_uid=CANONICAL_CONTROLLER_UID,
            approved_validator_uid=ROUND17_VALIDATOR_UID,
        )
        expected = {
            "approved_uncertainty_only": "approved_uncertainty_only",
            "approved_conflict_register_only": "approved_conflict_register_only",
        }.get(_text(value.get("decision")))
        if expected and assimilation_decision.get("decision") != expected:
            blockers.append(blocker("o_gate_assimilation_scope_mismatch", "O-Gate and AssimilationDecision scopes differ."))
    return sorted_blockers(blockers)


def validate_twin_debt(value: Mapping[str, object]) -> list[dict[str, object]]:
    debt = dict(value)
    uid = _text(debt.get("debt_uid"))
    blockers = _required(
        debt,
        (
            "debt_uid",
            "belief_state_ref",
            "component_values",
            "component_status",
            "measurement_basis",
            "versioned_thresholds",
            "allowed_decisions",
            "provenance",
        ),
        uid,
    )
    if debt.get("schema_version") != SCHEMA_VERSIONS["twin_debt"]:
        blockers.append(blocker("unknown_twin_debt_schema", "TwinDebt schema is not canonical.", debt_uid=uid))
    values = _mapping(debt.get("component_values"))
    statuses = _mapping(debt.get("component_status"))
    unknown = {_text(item) for item in _list(debt.get("unknown_components"))}
    if set(values) != set(DEBT_COMPONENTS) or set(statuses) != set(DEBT_COMPONENTS):
        blockers.append(blocker("twin_debt_components_incomplete", "All Twin Debt components must remain explicit.", debt_uid=uid))
    for component in DEBT_COMPONENTS:
        status = statuses.get(component)
        component_value = values.get(component)
        if status not in DEBT_STATUSES:
            blockers.append(blocker("twin_debt_status_invalid", "Twin Debt component status is invalid.", debt_uid=uid, component=component))
        if status == "unknown":
            if component_value is not None or component not in unknown:
                blockers.append(blocker("unknown_twin_debt_replaced_by_zero", "Unknown Twin Debt must remain null and listed.", debt_uid=uid, component=component))
        elif component_value is None or not isinstance(component_value, (int, float)):
            blockers.append(blocker("twin_debt_value_missing", "Measured Twin Debt requires a numeric value.", debt_uid=uid, component=component))
    allowed_decisions = {_text(item) for item in _list(debt.get("allowed_decisions"))}
    if ({"high", "unknown"} & set(statuses.values())) and "world_action_restricted" in allowed_decisions:
        blockers.append(
            blocker(
                "high_twin_debt_world_action_listed",
                "High or unknown Twin Debt cannot list a world action as an allowed decision.",
                debt_uid=uid,
            )
        )
    aggregation = debt.get("aggregation")
    if aggregation is not None:
        aggregate = _mapping(aggregation)
        if set(_mapping(aggregate.get("weights"))) != set(DEBT_COMPONENTS) or aggregate.get("reversible") is not True:
            blockers.append(blocker("twin_debt_aggregation_not_reversible", "Twin Debt aggregation requires versioned reversible weights.", debt_uid=uid))
    return sorted_blockers(blockers)


def validate_dual_action_decision(value: Mapping[str, object], *, twin_debt: Mapping[str, object]) -> list[dict[str, object]]:
    decision = dict(value)
    blockers = _required(
        decision,
        (
            "decision_uid",
            "belief_state_ref",
            "candidate_world_actions",
            "candidate_information_actions",
            "expected_utility",
            "expected_information_gain",
            "expected_twin_debt_change",
            "cost",
            "risk",
            "support_domain_ref",
            "selected_action_type",
            "selection_reason",
            "controller_uid",
        ),
        _text(decision.get("decision_uid")),
    )
    if decision.get("schema_version") != SCHEMA_VERSIONS["dual_action_decision"]:
        blockers.append(blocker("unknown_dual_action_schema", "DualActionDecision schema is not canonical."))
    if decision.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("noncanonical_dual_action_authority", "Only the controller may select an action type."))
    selected = decision.get("selected_action_type")
    if selected not in {"world_action", "information_action", "stop"}:
        blockers.append(blocker("dual_action_selection_invalid", "Selected action type is invalid."))
    statuses = set(_mapping(twin_debt.get("component_status")).values())
    if selected == "world_action" and statuses & {"high", "unknown"}:
        blockers.append(blocker("twin_debt_requires_information_action", "High or unknown debt requires information acquisition or stop."))
    if selected == "world_action" and "moderate" in statuses and not _mapping(decision.get("risk")).get("override_evidence_ref"):
        blockers.append(blocker("moderate_twin_debt_information_preferred", "Moderate debt requires explicit override evidence for world action."))
    return sorted_blockers(blockers)


def validate_budget_contracts(
    budget: Mapping[str, object], risk_budget: Mapping[str, object], decision: Mapping[str, object]
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if budget.get("schema_version") != SCHEMA_VERSIONS["scientific_budget"]:
        blockers.append(blocker("scientific_budget_schema_invalid", "ScientificBudget schema is not canonical."))
    if risk_budget.get("schema_version") != SCHEMA_VERSIONS["risk_budget"]:
        blockers.append(blocker("risk_budget_schema_invalid", "RiskBudget schema is not canonical."))
    if budget.get("controller_uid") != CANONICAL_CONTROLLER_UID or risk_budget.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("agent_increases_budget", "Only the controller may own budget contracts."))
    if risk_budget.get("production_execution_allowed") is not False or risk_budget.get("agent_may_increase_risk") is not False:
        blockers.append(blocker("production_risk_authorized", "Round-17 risk budget must prohibit production execution and Agent escalation."))
    limits = _mapping(budget.get("dimensions"))
    consumed = _mapping(budget.get("consumed"))
    remaining = _mapping(budget.get("remaining"))
    requested = _mapping(decision.get("requested_cost"))
    if not limits or set(consumed) != set(limits) or set(remaining) != set(limits):
        blockers.append(blocker("budget_dimensions_incomplete", "Budget dimensions must align."))
    exceeded = [key for key, value in requested.items() if not isinstance(value, (int, float)) or value > float(remaining.get(key, -1))]
    if exceeded and decision.get("decision") == "authorized":
        blockers.append(blocker("budget_exceeded_silently", "An over-budget action cannot be authorized.", dimensions=sorted(exceeded)))
    if decision.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("agent_increases_budget", "Only the controller may decide budget use."))
    requested_risk = decision.get("requested_risk_class")
    if requested_risk not in set(_list(risk_budget.get("allowed_risk_classes"))) and decision.get("decision") == "authorized":
        blockers.append(blocker("risk_budget_exceeded", "Requested risk class is not allowed."))
    return sorted_blockers(blockers)


def validate_tool_sandbox(value: Mapping[str, object], *, allowlisted_adapters: set[str]) -> list[dict[str, object]]:
    sandbox = dict(value)
    blockers = _required(
        sandbox,
        (
            "adapter_uid",
            "typed_request_schema",
            "read_only_code",
            "isolated_writable_runtime",
            "explicit_environment",
            "resource_limit",
            "timeout_seconds",
            "network_access_policy",
            "output_hash",
            "stdout_ref",
            "stderr_ref",
            "terminal_state",
            "failure_class",
        ),
        _text(sandbox.get("adapter_uid")),
    )
    if sandbox.get("adapter_uid") not in allowlisted_adapters:
        blockers.append(blocker("unregistered_adapter", "Sandbox adapter is not allowlisted."))
    if sandbox.get("read_only_code") is not True or sandbox.get("isolated_writable_runtime") is not True:
        blockers.append(blocker("tool_sandbox_isolation_missing", "Tool execution requires read-only code and isolated output."))
    if not isinstance(sandbox.get("timeout_seconds"), (int, float)) or float(sandbox.get("timeout_seconds", 0)) <= 0:
        blockers.append(blocker("tool_timeout_missing", "Tool execution requires a positive timeout."))
    if not is_sha256(sandbox.get("output_hash")):
        blockers.append(blocker("tool_output_hash_missing", "Tool output must be content-addressed."))
    forbidden = _forbidden_paths(sandbox)
    if forbidden:
        blockers.append(blocker("agent_invokes_arbitrary_shell", "Sandbox request contains arbitrary execution fields.", paths=forbidden))
    return sorted_blockers(blockers)


def validate_attempt_preservation(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Require immutable identities and explicit preservation for adapter attempts."""

    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in values:
        attempt = dict(value)
        uid = _text(attempt.get("attempt_uid"))
        if not uid or uid in seen:
            blockers.append(blocker("failed_tool_attempt_overwritten", "Tool attempt UIDs are immutable and unique.", attempt_uid=uid))
        seen.add(uid)
        if attempt.get("preserved") is not True:
            blockers.append(blocker("failed_tool_attempt_overwritten", "Every tool attempt must remain preserved.", attempt_uid=uid))
        if attempt.get("terminal_state") not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
            blockers.append(blocker("tool_attempt_terminal_state_invalid", "Tool attempt terminal state is invalid.", attempt_uid=uid))
        if attempt.get("terminal_state") != "COMPLETED" and not _text(attempt.get("failure_class")):
            blockers.append(blocker("tool_attempt_failure_unclassified", "Failed tool attempts require a failure class.", attempt_uid=uid))
        hashes = _mapping(attempt.get("output_hashes"))
        if hashes and any(not is_sha256(item) for item in hashes.values()):
            blockers.append(blocker("tool_attempt_hash_invalid", "Tool attempt output hashes must be SHA-256.", attempt_uid=uid))
    return sorted_blockers(blockers)


def validate_loop_event(value: Mapping[str, object]) -> list[dict[str, object]]:
    event = dict(value)
    uid = _text(event.get("event_uid"))
    blockers = _required(
        event,
        (
            "event_uid",
            "episode_uid",
            "prior_state",
            "next_state",
            "controller_uid",
            "input_hashes",
            "output_hashes",
            "timestamp",
            "decision_reason",
        ),
        uid,
    )
    if event.get("schema_version") != SCHEMA_VERSIONS["agent_loop_event"]:
        blockers.append(blocker("agent_loop_event_schema_invalid", "AgentLoopEvent schema is not canonical.", event_uid=uid))
    transition = (_text(event.get("prior_state")), _text(event.get("next_state")))
    if transition[0] not in LOOP_STATES or transition[1] not in LOOP_STATES or transition not in ALLOWED_TRANSITIONS:
        blockers.append(blocker("illegal_agent_loop_transition", "Agent loop transition is not allowed.", event_uid=uid, transition=list(transition)))
    if transition == ("blocked", "belief_updated"):
        blockers.append(blocker("blocked_loop_updates_belief", "A blocked loop cannot update belief state.", event_uid=uid))
    if event.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("noncanonical_transition_authority", "Only the controller may record transitions.", event_uid=uid))
    for field in ("input_hashes", "output_hashes"):
        hashes = _mapping(event.get(field))
        if any(not is_sha256(item) for item in hashes.values()):
            blockers.append(blocker("event_hash_invalid", "Event hashes must be SHA-256.", event_uid=uid, field=field))
    return sorted_blockers(blockers)


def validate_episode(value: Mapping[str, object], *, events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    episode = dict(value)
    uid = _text(episode.get("episode_uid"))
    blockers = _required(
        episode,
        (
            "episode_uid",
            "initial_belief_state_ref",
            "event_uids",
            "scenario_uid",
            "terminal_state",
            "controller_uid",
            "budget_decision_refs",
            "frozen_gap_impacts",
            "replay_identity",
            "status",
            "provenance",
        ),
        uid,
    )
    if episode.get("schema_version") != SCHEMA_VERSIONS["agent_loop_episode"]:
        blockers.append(blocker("agent_loop_episode_schema_invalid", "AgentLoopEpisode schema is not canonical.", episode_uid=uid))
    if episode.get("terminal_state") not in TERMINAL_STATES:
        blockers.append(blocker("generic_or_invalid_terminal_state", "Episode requires a canonical scientific terminal state.", episode_uid=uid))
    if episode.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker("noncanonical_episode_authority", "Only the controller may finalize an episode.", episode_uid=uid))
    declared = _list(episode.get("event_uids"))
    actual = [_text(event.get("event_uid")) for event in events]
    if declared != actual or any(_text(event.get("episode_uid")) != uid for event in events):
        blockers.append(blocker("episode_event_lineage_mismatch", "Episode event lineage is inconsistent.", episode_uid=uid))
    impacts = _mapping(episode.get("frozen_gap_impacts"))
    if any(value not in FROZEN_GAP_IMPACTS for value in impacts.values()):
        blockers.append(blocker("agent_closes_frozen_gap_through_reasoning", "Agent episode contains a forbidden frozen-gap impact.", episode_uid=uid))
    return sorted_blockers(blockers)


def validate_hypothesis_ledger_entry(value: Mapping[str, object]) -> list[dict[str, object]]:
    entry = dict(value)
    uid = _text(entry.get("hypothesis_uid"))
    blockers = _required(
        entry,
        (
            "hypothesis_uid",
            "version",
            "mechanism",
            "expected_observations",
            "falsification_criteria",
            "supporting_evidence",
            "conflicting_evidence",
            "test_history",
            "current_status",
            "status_decision_ref",
            "unresolved_conflict_count",
            "reliability_status",
            "support_domain_status",
        ),
        uid,
    )
    if entry.get("schema_version") != SCHEMA_VERSIONS["hypothesis_ledger_entry"]:
        blockers.append(blocker("hypothesis_ledger_schema_invalid", "Hypothesis ledger schema is not canonical.", hypothesis_uid=uid))
    if entry.get("current_status") not in HYPOTHESIS_STATUSES:
        blockers.append(blocker("hypothesis_status_invalid", "Hypothesis status is not allowed.", hypothesis_uid=uid))
    if entry.get("current_status") == "supported_model_conditionally" and not _present(entry.get("falsifier_agent_ref")):
        blockers.append(blocker("falsifier_skipped", "A positive model-conditional summary requires Falsifier participation.", hypothesis_uid=uid))
    if entry.get("current_status") == "supported_model_conditionally" and not _list(entry.get("falsification_criteria")):
        blockers.append(blocker("falsifier_skipped", "A positive assessment requires falsification criteria.", hypothesis_uid=uid))
    text = " ".join(_text(entry.get(field)) for field in ("mechanism", "current_status"))
    if any(pattern.search(text) for pattern in PROHIBITED_CLAIM_PATTERNS):
        blockers.append(blocker("hypothesis_claim_overreach", "Hypothesis ledger contains prohibited claim language.", hypothesis_uid=uid))
    return sorted_blockers(blockers)


def validate_explanation(value: Mapping[str, object]) -> list[dict[str, object]]:
    explanation = dict(value)
    uid = _text(explanation.get("explanation_uid"))
    blockers = _required(
        explanation,
        (
            "explanation_uid",
            "episode_uid",
            "belief_state_ref",
            "gate_decision_refs",
            "rollout_or_tool_artifact_refs",
            "uncertainty_report_ref",
            "reliability_report_ref",
            "support_domain_report_ref",
            "claim_eligibility_ref",
            "summary",
            "limitations",
            "provenance",
        ),
        uid,
    )
    if explanation.get("schema_version") != SCHEMA_VERSIONS["agent_explanation"]:
        blockers.append(blocker("agent_explanation_schema_invalid", "AgentExplanation schema is not canonical.", explanation_uid=uid))
    if explanation.get("causal_effect_claim") is not False:
        blockers.append(blocker("causal_effect_statement_emitted", "Agent explanation cannot claim a causal effect.", explanation_uid=uid))
    if explanation.get("clinical_recommendation") is not False:
        blockers.append(blocker("clinical_recommendation_emitted", "Agent explanation cannot make a clinical recommendation.", explanation_uid=uid))
    if explanation.get("model_based_counterfactual") is True and explanation.get("empirical_observation") is not False:
        blockers.append(blocker("model_rollout_treated_as_empirical_observation", "Model-based output cannot be labeled as empirical.", explanation_uid=uid))
    text = " ".join(
        [_text(explanation.get("summary"))] + [_text(item) for item in _list(explanation.get("limitations"))]
    )
    if any(pattern.search(text) for pattern in PROHIBITED_CLAIM_PATTERNS):
        blockers.append(blocker("prohibited_claim_language", "Agent explanation contains prohibited claim language.", explanation_uid=uid))
    if _list(explanation.get("conflicts")) and "conflict" not in text.lower() and "disagreement" not in text.lower():
        blockers.append(blocker("explanation_omits_conflict", "Explanation must report typed conflicts.", explanation_uid=uid))
    return sorted_blockers(blockers)


def validate_frozen_gap_map(rows: Sequence[Mapping[str, object]], *, expected_count: int = 20) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    uids = {_text(row.get("gap_uid")) for row in rows}
    if len(rows) != expected_count or len(uids) != expected_count:
        blockers.append(blocker("frozen_gap_count_drift", "Exactly 20 unique frozen gaps must remain represented."))
    for row in rows:
        if row.get("round17_impact") not in FROZEN_GAP_IMPACTS:
            blockers.append(blocker("agent_closes_frozen_gap_through_reasoning", "Round-17 gap impact is invalid.", gap_uid=row.get("gap_uid")))
        if str(row.get("resolved_by_agent_reasoning", "")).lower() not in {"false", "0"}:
            blockers.append(blocker("agent_closes_frozen_gap_through_reasoning", "Agent reasoning cannot resolve a frozen gap.", gap_uid=row.get("gap_uid")))
    return sorted_blockers(blockers)


def create_belief_state_from_assimilation(
    *,
    parent: Mapping[str, object],
    o_gate: Mapping[str, object],
    assimilation_decision: Mapping[str, object],
    posterior_refs: Mapping[str, str],
    active_hypothesis_refs: Sequence[str],
    constraint_state_ref: str,
    support_domain_ref: str,
    twin_debt_ref: str,
    evidence_snapshot_ref: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    if o_gate.get("controller_uid") != CANONICAL_CONTROLLER_UID or o_gate.get("decision") not in O_GATE_APPROVALS:
        raise ValueError("O-Gate did not authorize a belief-state transition.")
    if assimilation_decision.get("controller_uid") != CANONICAL_CONTROLLER_UID or not str(assimilation_decision.get("decision", "")).startswith("approved"):
        raise ValueError("AssimilationDecision did not authorize a belief-state transition.")
    if any(key in provenance for key in ("agent_text", "free_text_posterior", "llm_state")):
        raise ValueError("Agent text cannot become posterior state.")
    required_refs = {"state_posterior_ref", "parameter_posterior_ref", "model_family_posterior_ref"}
    if set(posterior_refs) != required_refs or any(not _text(value) for value in posterior_refs.values()):
        raise ValueError("Posterior references must be complete and typed.")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["twin_belief_state"],
        "belief_state_uid": "pending",
        "parent_belief_state_uid": parent.get("belief_state_uid"),
        **dict(posterior_refs),
        "evidence_snapshot_ref": evidence_snapshot_ref,
        "active_hypothesis_refs": list(active_hypothesis_refs),
        "constraint_state_ref": constraint_state_ref,
        "support_domain_ref": support_domain_ref,
        "twin_debt_ref": twin_debt_ref,
        "content_hash": "pending",
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "creation_decision_ref": o_gate.get("gate_uid"),
        "provenance": dict(provenance),
    }
    payload["content_hash"] = belief_state_content_hash(payload)
    payload["belief_state_uid"] = stable_uid("belief_state", payload["content_hash"])
    payload["content_hash"] = belief_state_content_hash(payload)
    return payload


def validate_unique_uids(objects: Sequence[Mapping[str, object]], uid_fields: Sequence[str]) -> list[dict[str, object]]:
    seen: set[str] = set()
    blockers: list[dict[str, object]] = []
    for value in objects:
        uid = next((_text(value.get(field)) for field in uid_fields if _text(value.get(field))), "")
        if not uid or uid in seen:
            blockers.append(blocker("duplicate_uid", "Object UIDs must be globally unique.", object_uid=uid))
        seen.add(uid)
    return sorted_blockers(blockers)


def validate_lineage_acyclic(edges: Sequence[tuple[str, str]]) -> list[dict[str, object]]:
    graph: dict[str, set[str]] = {}
    for parent, child in edges:
        if parent:
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

    return [] if all(visit(node) for node in graph) else [blocker("lineage_cycle", "SciTiL lineage must be acyclic.")]


def release_content_hash(value: Mapping[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key not in {"release_uid", "release_content_hash", "created_at"}}
    return canonical_sha256(payload)


def validate_release_candidate(value: Mapping[str, object]) -> list[dict[str, object]]:
    release = dict(value)
    blockers: list[dict[str, object]] = []
    if release.get("schema_version") != SCHEMA_VERSIONS["release"]:
        blockers.append(blocker("round17_release_schema_invalid", "Round-17 release schema is not canonical."))
    if release.get("sealed") is not False or release.get("status") != "materialized_unsealed":
        blockers.append(blocker("evidence_fabric_sealed", "Round-17 release must remain unsealed."))
    if release.get("dt_core_status") != "not_evaluated":
        blockers.append(blocker("dt_core_evaluated", "Round-17 must not evaluate DT_CORE."))
    if release.get("frozen_gap_count") != 20 or release.get("resolved_by_agent_reasoning") != 0:
        blockers.append(blocker("agent_closes_frozen_gap_through_reasoning", "Round-17 release must preserve all frozen gaps."))
    if release.get("empirical_observation_count") != 0 or release.get("autonomous_discovery_claim") is not False:
        blockers.append(blocker("agent_output_promoted_to_empirical", "SciTiL pilot output is contract evidence only."))
    if release.get("release_uid") == release.get("parent_release_uid"):
        blockers.append(blocker("release_uid_not_versioned", "Round-17 must create a successor release UID."))
    if release.get("release_content_hash") != release_content_hash(release):
        blockers.append(blocker("release_content_hash_mismatch", "Round-17 release content hash is invalid."))
    return sorted_blockers(blockers)


class TwinKernel:
    """Non-LLM controller boundary for typed SciTiL state transitions."""

    llm_backed = False

    def __init__(self, *, controller_uid: str, allowlisted_adapters: Mapping[str, Mapping[str, object]]) -> None:
        if controller_uid != CANONICAL_CONTROLLER_UID:
            raise ValueError("TwinKernel requires the canonical controller.")
        self.controller_uid = controller_uid
        self._adapters = {str(key): dict(value) for key, value in allowlisted_adapters.items()}

    @property
    def adapter_uids(self) -> frozenset[str]:
        return frozenset(self._adapters)

    def validate_proposal(self, proposal: Mapping[str, object], *, agent: Mapping[str, object], belief_states: set[str]) -> list[dict[str, object]]:
        return validate_agent_proposal(proposal, agent=agent, known_belief_states=belief_states)

    def validate_adapter_execution(self, execution_contract: Mapping[str, object]) -> list[dict[str, object]]:
        return validate_tool_sandbox(execution_contract, allowlisted_adapters=set(self._adapters))

    def apply_assimilation(
        self,
        *,
        parent: Mapping[str, object],
        o_gate: Mapping[str, object],
        assimilation_decision: Mapping[str, object],
        posterior_refs: Mapping[str, str],
        active_hypothesis_refs: Sequence[str],
        constraint_state_ref: str,
        support_domain_ref: str,
        twin_debt_ref: str,
        evidence_snapshot_ref: str,
        provenance: Mapping[str, object],
    ) -> dict[str, object]:
        return create_belief_state_from_assimilation(
            parent=parent,
            o_gate=o_gate,
            assimilation_decision=assimilation_decision,
            posterior_refs=posterior_refs,
            active_hypothesis_refs=active_hypothesis_refs,
            constraint_state_ref=constraint_state_ref,
            support_domain_ref=support_domain_ref,
            twin_debt_ref=twin_debt_ref,
            evidence_snapshot_ref=evidence_snapshot_ref,
            provenance=provenance,
        )


__all__ = [
    "AGENT_ROLES",
    "ALLOWED_TRANSITIONS",
    "AgentExplanation",
    "AgentIdentity",
    "AgentLoopBudgetDecision",
    "AgentLoopEpisode",
    "AgentLoopEvent",
    "AgentProposal",
    "CANONICAL_CONTROLLER_UID",
    "DEBT_COMPONENTS",
    "DualActionDecision",
    "FROZEN_GAP_IMPACTS",
    "H_GATE_DECISIONS",
    "HYPOTHESIS_STATUSES",
    "HypothesisGateDecision",
    "HypothesisLedgerEntry",
    "I_GATE_DECISIONS",
    "InformationActionGateDecision",
    "LOOP_STATES",
    "O_GATE_DECISIONS",
    "ObservationGateDecision",
    "PROPOSAL_TYPES",
    "RISK_CLASSES",
    "ROUND17_VALIDATOR_UID",
    "RiskBudget",
    "SCHEMA_VERSIONS",
    "ScientificBudget",
    "ScientificHypothesis",
    "TERMINAL_STATES",
    "TwinBeliefState",
    "TwinDebt",
    "TwinKernel",
    "W_GATE_DECISIONS",
    "WorldActionGateDecision",
    "belief_state_content_hash",
    "canonical_sha256",
    "conflicts_preserved",
    "create_belief_state_from_assimilation",
    "release_content_hash",
    "round17_config",
    "sorted_blockers",
    "stable_uid",
    "validate_agent_identity",
    "validate_agent_proposal",
    "validate_attempt_preservation",
    "validate_belief_state",
    "validate_budget_contracts",
    "validate_dual_action_decision",
    "validate_episode",
    "validate_explanation",
    "validate_frozen_gap_map",
    "validate_h_gate",
    "validate_hypothesis",
    "validate_hypothesis_ledger_entry",
    "validate_i_gate",
    "validate_lineage_acyclic",
    "validate_loop_event",
    "validate_o_gate",
    "validate_release_candidate",
    "validate_round17_config",
    "validate_tool_sandbox",
    "validate_twin_debt",
    "validate_unique_uids",
    "validate_w_gate",
]
