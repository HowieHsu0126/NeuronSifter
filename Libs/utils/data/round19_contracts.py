"""Fail-closed typed contracts for Round-19 public E2/E4 expansion.

The module is deliberately independent of retrieval, simulation, training, and
Evidence Fabric mutation. It validates immutable records and returns explicit
machine-readable blockers. Model-derived trajectories never become empirical
observations through these contracts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any

from Libs.utils.data.trajectory_contracts import (
    blocker,
    is_sha256,
    sorted_blockers,
    stable_uid,
)


SCHEMA_VERSIONS = {
    "scientific_contract": "neuronsifter-round19-scientific-contract-v1",
    "training_readiness_contract": "neuronsifter-round19-training-readiness-contract-v1",
    "e2_endpoint": "neuronsifter-round19-e2-endpoint-v1",
    "e2_admission": "neuronsifter-round19-e2-admission-v1",
    "e4_producer_qualification": "neuronsifter-round19-e4-producer-qualification-v1",
    "e4_trajectory": "neuronsifter-round19-e4-intervention-trajectory-v1",
    "e4_control_pair": "neuronsifter-round19-e4-control-pair-v1",
    "e2_e4_alignment": "neuronsifter-round19-e2-e4-alignment-v1",
    "frozen_gap_resolution": "neuronsifter-round19-frozen-gap-resolution-v1",
    "training_eligibility": "neuronsifter-round19-training-eligibility-v1",
}

PUBLIC_SOURCE_CLASSES = frozenset(
    {
        "open_access_article",
        "public_abstract",
        "public_supplement",
        "public_accession_data",
        "public_model_repository",
        "public_code_repository",
        "public_benchmark_database",
        "public_institutional_repository",
    }
)
ADMISSION_CLASSES = frozenset(
    {
        "admitted_e2_strong",
        "admitted_e2_weighted",
        "admitted_e2_directional",
        "admitted_stress",
        "quarantined",
        "rejected",
    }
)
E4_DECISIONS = frozenset(
    {
        "admitted_e4_strong",
        "admitted_e4_restricted",
        "admitted_e4_calibration_only",
        "admitted_stress",
        "quarantined_numerical",
        "quarantined_constraint",
        "rejected_action_incomplete",
        "rejected_initial_state",
        "rejected_provenance",
        "rejected_support_domain",
    }
)
READINESS_STATES = frozenset(
    {
        "training_ready_restricted",
        "training_ready_formal",
        "blocked_insufficient_e2",
        "blocked_insufficient_e4",
        "blocked_control_pairs",
        "blocked_validation_units",
        "blocked_split_feasibility",
        "blocked_support_domain",
    }
)
GAP_STATES = frozenset(
    {
        "resolved_by_admitted_e2",
        "resolved_by_admitted_e4",
        "partially_addressed",
        "frozen_unresolved",
    }
)
FORBIDDEN_GAP_RESOLUTION = frozenset(
    {
        "resolved_by_prediction",
        "resolved_by_agent_reasoning",
        "resolved_by_literature_narrative",
        "resolved_by_unadmitted_simulation",
        "resolved_by_row_count",
        "resolved_by_schema_completion",
    }
)
LABEL_CLASSES = frozenset({"data_derived_label", "author_claim_label", "cross_study_consensus"})
ALIGNMENT_CLASSES = frozenset({"direct", "partial", "directional_only", "stress_only", "not_comparable"})
TRAINING_ROLES = frozenset(
    {
        "state_pretraining",
        "action_endpoint_training",
        "trajectory_training",
        "calibration",
        "validation",
        "stress_only",
        "hypothesis_generation_only",
        "rejected",
    }
)
CONTROLLED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"controlled[-_ ]access",
        r"credentialed",
        r"institutional session",
        r"controlled clinical repository",
    )
)
UNKNOWN_VALUES = frozenset({"", "unknown", "unclear", "unresolved", "missing", "not_known", "n/a"})
NUMERIC_TYPES = (int, float)


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite(value: object) -> bool:
    return isinstance(value, NUMERIC_TYPES) and not isinstance(value, bool) and math.isfinite(float(value))


def _known(value: object) -> bool:
    return _text(value).lower() not in UNKNOWN_VALUES


def _string_values(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _string_values(item)]
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _string_values(item)]
    return [value] if isinstance(value, str) else []


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def record_uid(prefix: str, value: object) -> str:
    return stable_uid(prefix, value)


def as_record(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Unsupported Round-19 record type: {type(value)!r}")


@dataclass(frozen=True)
class Round19ScientificContract:
    contract_uid: str
    schema_version: str
    created_before_discovery: bool
    frozen_at: str
    minimum_strong_e2_endpoint_blocks: int
    minimum_weighted_e2_endpoint_blocks: int
    minimum_independent_studies: int
    minimum_action_classes: int
    minimum_support_domains: int
    minimum_intervention_e4_families: int
    minimum_matched_control_pairs: int
    minimum_independent_validation_units: int
    publication_holdout_required: bool = True
    source_holdout_required: bool = True
    action_holdout_required: bool = True
    support_domain_holdout_required: bool = True
    final_held_out_frozen: bool = True
    strict_public_only_flag: bool = True
    controlled_source_resolution_count: int = 0
    training_authorized: bool = False
    evidence_fabric_sealed: bool = False
    dt_core_status: str = "not_evaluated"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingReadinessContract:
    contract_uid: str
    schema_version: str
    frozen_at: str
    restricted_min_strong_e2: int
    restricted_min_weighted_e2: int
    restricted_min_e4_families: int
    restricted_min_control_pairs: int
    restricted_min_validation_units: int
    formal_min_strong_e2: int
    formal_min_weighted_e2: int
    formal_min_e4_families: int
    formal_min_control_pairs: int
    formal_min_validation_units: int
    required_holdouts: tuple[str, ...]
    training_authorized: bool = False
    training_started: bool = False

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["required_holdouts"] = list(self.required_holdouts)
        return result


@dataclass(frozen=True)
class E2Endpoint:
    endpoint_uid: str
    schema_version: str
    source_uid: str
    source_accession: str
    source_class: str
    source_version: str
    license_or_permitted_use: str
    content_hash: str
    retrieval_date: str
    publication_or_accession_identity: str
    parser_version: str
    public_availability_timestamp: str
    organism: str
    disease_or_state_context: str
    region: str
    compartment_or_cell_type: str
    action_identity: str
    target_or_agent: str
    dose: object
    dose_unit: str
    route: str
    schedule: str
    duration: str
    timepoint: str
    endpoint_identity: str
    endpoint_value: object
    endpoint_unit: str
    comparator: str
    sample_size: object
    uncertainty: object
    extraction_method: str
    label_class: str
    partition: str
    independence_groups: dict[str, str]
    admission_class: str = "quarantined"
    source_is_public: bool = True
    controlled_access_required: bool = False
    empirical_observation: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class E2AdmissionDecision:
    admission_uid: str
    endpoint_uid: str
    decision: str
    reason_codes: tuple[str, ...]
    evidence_hash: str
    eligibility: str
    empirical_observation: bool = True

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reason_codes"] = list(self.reason_codes)
        return result


@dataclass(frozen=True)
class E4ProducerQualification:
    producer_uid: str
    producer_version: str
    producer_hash: str
    equation_model_identity: str
    parameter_provenance: str
    supported_state_variables: tuple[str, ...]
    supported_action_semantics: tuple[str, ...]
    initial_state_contract: str
    control_generation_capability: str
    seed_policy: str
    fidelity_policy: str
    numerical_qc: str
    constraint_policy: str
    license_or_redistribution: str
    decision: str
    support_domain: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["supported_state_variables"] = list(self.supported_state_variables)
        result["supported_action_semantics"] = list(self.supported_action_semantics)
        return result


@dataclass(frozen=True)
class E4InterventionTrajectory:
    trajectory_uid: str
    producer_uid: str
    producer_version: str
    producer_hash: str
    initial_state_hash: str
    action_hash: str
    support_domain: str
    state_artifact_hash: str
    native_time_grid_hash: str
    canonical_time_grid_hash: str
    fidelity: str
    seed: str
    parameter_hash: str
    constraint_status: str
    numerical_status: str
    output_class: str = "model_trajectory"
    empirical_observation: bool = False
    admission_decision: str = "quarantined_numerical"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class E4ControlPair:
    pair_uid: str
    action_trajectory_uid: str
    control_trajectory_uid: str
    initial_state_family: str
    producer_version: str
    parameter_family: str
    fidelity: str
    time_grid: str
    seed_policy: str
    pairing_decision: str
    action_effect_claim_eligible: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class E2E4Alignment:
    alignment_uid: str
    endpoint_uid: str
    trajectory_uid: str
    action_identity_match: bool
    dose_match: bool
    route_match: bool
    schedule_match: bool
    timepoint_match: bool
    context_match: bool
    endpoint_mapping: str
    support_domain_match: bool
    alignment_class: str
    model_output_non_empirical: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenGapResolution:
    gap_uid: str
    status: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    resolution_basis: str = ""

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evidence_refs"] = list(self.evidence_refs)
        return result


@dataclass(frozen=True)
class TrainingEligibilityRow:
    artifact_uid: str
    role: str
    partition: str
    publication_family: str
    source_family: str
    study_family: str
    biological_unit_family: str
    action_family: str
    regimen_family: str
    support_domain: str
    eligibility_basis: str
    validation_only: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_public_boundary(
    record: Mapping[str, object], *, forbidden_source_tokens: Sequence[str] = ()
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    texts = _string_values(record)
    if any(pattern.search(text) for text in texts for pattern in CONTROLLED_PATTERNS):
        blockers.append(blocker("controlled_source", "Controlled or credentialed source route is forbidden."))
    lowered = [text.lower() for text in texts]
    if any(token.strip() and token.lower() in text for token in forbidden_source_tokens for text in lowered):
        blockers.append(blocker("controlled_source", "A policy-rejected source route is forbidden."))
    if record.get("strict_public_only_flag") is False or record.get("controlled_access_required") is True:
        blockers.append(blocker("public_only_policy", "Round-19 requires strict public-only source policy."))
    if int(record.get("controlled_source_resolution_count", 0) or 0) != 0:
        blockers.append(blocker("controlled_source_resolution", "Controlled source resolution count must remain zero."))
    if record.get("publicly_available") is not True and record.get("source_is_public") is not True:
        blockers.append(blocker("source_not_public", "Source public availability is not established."))
    if not _known(record.get("license_or_permitted_use")):
        blockers.append(blocker("unknown_license", "Unknown license or permitted-use status blocks admission."))
    return sorted_blockers(blockers)


def validate_e2_endpoint(
    record: Mapping[str, object], *, require_strong: bool = False,
    forbidden_source_tokens: Sequence[str] = (),
) -> list[dict[str, object]]:
    blockers = validate_public_boundary(record, forbidden_source_tokens=forbidden_source_tokens)
    required = (
        "source_uid", "source_accession", "source_class", "source_version",
        "license_or_permitted_use", "content_hash", "retrieval_date",
        "publication_or_accession_identity", "parser_version", "public_availability_timestamp",
        "organism", "disease_or_state_context", "region", "compartment_or_cell_type",
        "action_identity", "target_or_agent", "dose_unit", "route", "schedule",
        "duration", "timepoint", "endpoint_identity", "endpoint_unit", "comparator",
        "sample_size", "extraction_method", "label_class", "partition",
    )
    for field_name in required:
        if not _known(record.get(field_name)):
            blockers.append(blocker("e2_missing_field", f"E2 endpoint field is missing: {field_name}.", field=field_name))
    if record.get("source_class") not in PUBLIC_SOURCE_CLASSES:
        blockers.append(blocker("e2_source_class", "E2 source class is not an allowed public class."))
    if not is_sha256(record.get("content_hash")):
        blockers.append(blocker("e2_source_hash", "E2 source content hash must be SHA-256."))
    if record.get("label_class") not in LABEL_CLASSES:
        blockers.append(blocker("e2_label_class", "E2 label provenance class is not canonical."))
    if record.get("partition") not in {"development", "qualification", "final_held_out", "stress"}:
        blockers.append(blocker("e2_partition", "E2 partition is not canonical."))
    if record.get("source_class") == "public_abstract" and _finite(record.get("endpoint_value")):
        blockers.append(blocker("abstract_numeric_endpoint", "Precise numeric values may not be reconstructed from abstracts."))
    if require_strong or record.get("admission_class") == "admitted_e2_strong":
        strong_fields = ("endpoint_value", "dose", "dose_unit", "sample_size", "uncertainty")
        for field_name in strong_fields:
            if not _finite(record.get(field_name)) and field_name not in {"uncertainty"}:
                blockers.append(blocker("e2_strong_numeric_missing", f"Strong E2 endpoint lacks numeric field: {field_name}."))
        if not _known(record.get("uncertainty")):
            blockers.append(blocker("e2_uncertainty_missing", "Strong E2 endpoint requires uncertainty."))
        if record.get("extraction_method") not in {"machine_readable_supplement", "public_accession_numeric_table", "public_bioassay_table"}:
            blockers.append(blocker("e2_extraction_method", "Strong E2 requires a machine-readable numeric extraction method."))
    return sorted_blockers(blockers)


def validate_e2_admission(record: Mapping[str, object], endpoint: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    decision = _text(record.get("decision"))
    if decision not in ADMISSION_CLASSES:
        blockers.append(blocker("e2_admission_decision", "E2 admission class is not canonical."))
    if record.get("endpoint_uid") != endpoint.get("endpoint_uid"):
        blockers.append(blocker("e2_admission_lineage", "E2 admission must bind to the endpoint UID."))
    if decision in {"admitted_e2_strong", "admitted_e2_weighted"}:
        blockers.extend(validate_e2_endpoint(endpoint, require_strong=decision == "admitted_e2_strong"))
    if decision.startswith("admitted") and endpoint.get("empirical_observation") is not True:
        blockers.append(blocker("e2_not_empirical", "Admitted E2 must remain an empirical public observation."))
    if not is_sha256(record.get("evidence_hash")):
        blockers.append(blocker("e2_admission_hash", "E2 admission requires an evidence hash."))
    return sorted_blockers(blockers)


def validate_e4_action(action: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    action_class = _text(action.get("action_class"))
    required_by_class = {
        "pharmacological": ("agent", "target_set", "dose", "dose_unit", "route", "schedule", "duration", "exposure_occupancy_semantics", "context"),
        "combination": ("components", "interaction_assumption", "context"),
        "no_action_control": ("context",),
        "vehicle_control": ("vehicle", "context"),
        "sham_control": ("sham_description", "context"),
    }
    if action_class not in required_by_class:
        blockers.append(blocker("unsupported_action_class", "E4 action class is not registered.", action_class=action_class))
    for field_name in required_by_class.get(action_class, ()):
        value = action.get(field_name)
        if value is None or value == "" or value == []:
            blockers.append(blocker("incomplete_action", f"E4 action field is missing: {field_name}.", field=field_name))
    if action_class == "pharmacological":
        if not _finite(action.get("dose")):
            blockers.append(blocker("missing_dose", "Pharmacological action dose must be numeric."))
        if not _known(action.get("dose_unit")):
            blockers.append(blocker("missing_dose_unit", "Pharmacological action dose unit is required."))
        if action.get("route_supported") is False:
            blockers.append(blocker("unsupported_route", "The producer does not support the requested route."))
        if action.get("schedule_supported") is False:
            blockers.append(blocker("unsupported_schedule", "The producer does not support the requested schedule."))
    return sorted_blockers(blockers)


def validate_initial_state_claim(record: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if record.get("origin") in {"synthetic", "explicit_synthetic_qualification_state"} and record.get("biological_validation_eligible") is True:
        blockers.append(blocker("synthetic_initial_state_validation", "A synthetic initial state cannot be biological validation evidence."))
    return sorted_blockers(blockers)


def validate_intervention_effect_evidence(
    trajectory: Mapping[str, object], matched_control: Mapping[str, object] | None
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if trajectory.get("intervention_effect_claim") is True and not matched_control:
        blockers.append(blocker("missing_matched_control", "An intervention trajectory requires a matched control before any effect-evidence claim."))
    return sorted_blockers(blockers)


def validate_e4_producer(record: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    required = (
        "producer_uid", "producer_version", "producer_hash", "equation_model_identity",
        "parameter_provenance", "supported_state_variables", "supported_action_semantics",
        "initial_state_contract", "control_generation_capability", "seed_policy",
        "fidelity_policy", "numerical_qc", "constraint_policy", "license_or_redistribution",
        "support_domain", "decision",
    )
    for field_name in required:
        if not _known(record.get(field_name)) and record.get(field_name) not in ([],):
            blockers.append(blocker("producer_missing_field", f"Producer qualification field is missing: {field_name}."))
    if not is_sha256(record.get("producer_hash")):
        blockers.append(blocker("producer_hash", "Producer identity must be content hashed."))
    if record.get("decision") not in {"qualified_for_intervention_e4", "qualified_with_restrictions", "blocked_action_semantics", "blocked_parameter_provenance", "blocked_initial_state", "blocked_support_domain", "blocked_calibration", "blocked_numerical_stability", "blocked_license"}:
        blockers.append(blocker("producer_decision", "Producer qualification decision is not canonical."))
    if record.get("decision") in {"qualified_for_intervention_e4", "qualified_with_restrictions"}:
        if not record.get("supported_action_semantics"):
            blockers.append(blocker("producer_action_semantics", "Qualified producer must declare action semantics."))
        if not _known(record.get("parameter_provenance")):
            blockers.append(blocker("producer_parameter_provenance", "Qualified producer requires parameter provenance."))
        if not _known(record.get("initial_state_contract")):
            blockers.append(blocker("producer_initial_state", "Qualified producer requires an initial-state contract."))
    return sorted_blockers(blockers)


def validate_e4_trajectory(record: Mapping[str, object], action: Mapping[str, object]) -> list[dict[str, object]]:
    blockers = validate_e4_action(action)
    required_hashes = (
        "producer_hash", "initial_state_hash", "action_hash", "state_artifact_hash",
        "native_time_grid_hash", "canonical_time_grid_hash", "parameter_hash",
    )
    for field_name in required_hashes:
        if not is_sha256(record.get(field_name)):
            blockers.append(blocker("trajectory_hash", f"E4 trajectory hash is missing or invalid: {field_name}."))
    if record.get("output_class") != "model_trajectory" or record.get("empirical_observation") is not False:
        blockers.append(blocker("model_output_class", "Simulator output must remain explicitly non-empirical."))
    if record.get("source_evidence_type") == "E3" and str(record.get("admission_decision", "")).startswith("admitted_e4"):
        blockers.append(blocker("legacy_e3_promotion", "Legacy E3 cannot be promoted directly to E4."))
    if record.get("constraint_status") != "passed":
        blockers.append(blocker("constraint_failure", "Constraint-failed trajectories cannot be admitted."))
    if record.get("numerical_status") != "completed":
        blockers.append(blocker("numerical_failure", "Numerically incomplete trajectories cannot be admitted."))
    if record.get("admission_decision") not in E4_DECISIONS:
        blockers.append(blocker("e4_admission_decision", "E4 trajectory admission decision is not canonical."))
    return sorted_blockers(blockers)


def validate_control_pair(pair: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    required = ("pair_uid", "action_trajectory_uid", "control_trajectory_uid", "initial_state_family", "producer_version", "parameter_family", "fidelity", "time_grid", "seed_policy", "pairing_decision")
    for field_name in required:
        if not _known(pair.get(field_name)):
            blockers.append(blocker("control_pair_missing_field", f"Control pair field is missing: {field_name}."))
    if pair.get("action_trajectory_uid") == pair.get("control_trajectory_uid"):
        blockers.append(blocker("control_pair_identity", "Action and control trajectory UIDs must differ."))
    if pair.get("pairing_decision") not in {"exact_deterministic_pair", "matched_stochastic_pair", "group_level_pair", "unpaired_descriptive_rollout"}:
        blockers.append(blocker("control_pair_decision", "Control pairing decision is not canonical."))
    if pair.get("pairing_decision") == "unpaired_descriptive_rollout" and pair.get("action_effect_claim_eligible") is True:
        blockers.append(blocker("unpaired_effect_claim", "Unpaired rollouts cannot support intervention effect evidence."))
    return sorted_blockers(blockers)


def validate_e2_e4_alignment(record: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if record.get("alignment_class") not in ALIGNMENT_CLASSES:
        blockers.append(blocker("alignment_class", "E2/E4 alignment class is not canonical."))
    if record.get("model_output_non_empirical") is not True:
        blockers.append(blocker("alignment_empirical_boundary", "E4 alignment must mark model output non-empirical."))
    if record.get("alignment_class") == "direct":
        for field_name in ("action_identity_match", "dose_match", "route_match", "schedule_match", "timepoint_match", "context_match", "support_domain_match"):
            if record.get(field_name) is not True:
                blockers.append(blocker("direct_alignment_incomplete", f"Direct alignment requires {field_name}."))
    return sorted_blockers(blockers)


def audit_independence(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    required_fields = (
        "publication_family", "preprint_to_journal_family", "accession_family", "study",
        "biological_unit", "compound", "target_program", "action_class", "dose_regimen",
        "endpoint", "timepoint", "source_collection",
    )
    blockers: list[dict[str, object]] = []
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for field_name in required_fields:
            if not _known(row.get(field_name)):
                blockers.append(blocker("independence_field_missing", f"Independence field is missing: {field_name}."))
        group_uid = _text(row.get("study")) or _text(row.get("accession_family"))
        groups[group_uid].append(dict(row))
    independent_studies = sorted(uid for uid in groups if _known(uid))
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted_blockers(blockers),
        "independent_study_count": len(independent_studies),
        "independent_study_uids": independent_studies,
        "independence_groups": {uid: len(items) for uid, items in sorted(groups.items())},
        "rows_not_counted_as_independent": ["technical_replicates", "normalized_matrices", "figure_panels", "database_mirrors"],
    }


def audit_split_leakage(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fields = ("publication_family", "source_family", "study", "biological_unit", "action_class", "regimen", "producer_family", "support_domain")
    blockers: list[dict[str, object]] = []
    for field_name in fields:
        seen: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = _text(row.get(field_name))
            partition = _text(row.get("partition"))
            if value and partition:
                seen[value].add(partition)
        for value, partitions in seen.items():
            if "final_held_out" in partitions and len(partitions - {"final_held_out"}) > 0:
                blockers.append(blocker("split_leakage", f"{field_name} crosses final held-out partition.", field=field_name, value=value, partitions=sorted(partitions)))
    return sorted_blockers(blockers)


def evaluate_training_readiness(
    *,
    contract: Mapping[str, object],
    e2_rows: Sequence[Mapping[str, object]],
    e4_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    validation_units: Sequence[str],
    split_blockers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    strong = sum(row.get("decision") == "admitted_e2_strong" for row in e2_rows)
    weighted = sum(row.get("decision") in {"admitted_e2_strong", "admitted_e2_weighted"} for row in e2_rows)
    e4_families = {(_text(row.get("producer_uid")), _text(row.get("support_domain"))) for row in e4_rows if row.get("decision") in {"admitted_e4_strong", "admitted_e4_restricted"}}
    pairs = sum(row.get("pairing_decision") in {"exact_deterministic_pair", "matched_stochastic_pair", "group_level_pair"} for row in pair_rows)
    independent_units = len(set(_text(value) for value in validation_units if _known(value)))
    blockers: list[dict[str, object]] = []
    if strong < int(contract.get("restricted_min_strong_e2", 0)) or weighted < int(contract.get("restricted_min_weighted_e2", 0)):
        blockers.append(blocker("insufficient_e2", "E2 count thresholds are not met.", strong=strong, weighted=weighted))
    if len(e4_families) < int(contract.get("restricted_min_e4_families", 0)):
        blockers.append(blocker("insufficient_e4", "Intervention E4 family threshold is not met.", observed=len(e4_families)))
    if pairs < int(contract.get("restricted_min_control_pairs", 0)):
        blockers.append(blocker("insufficient_control_pairs", "Matched control pair threshold is not met.", observed=pairs))
    if independent_units < int(contract.get("restricted_min_validation_units", 0)):
        blockers.append(blocker("insufficient_validation_units", "Independent validation-unit threshold is not met.", observed=independent_units))
    blockers.extend(dict(item) for item in split_blockers)
    state = "training_ready_restricted" if not blockers else "blocked_insufficient_e2"
    if any(item.get("code") == "insufficient_e4" for item in blockers):
        state = "blocked_insufficient_e4"
    elif any(item.get("code") == "insufficient_control_pairs" for item in blockers):
        state = "blocked_control_pairs"
    elif any(item.get("code") == "insufficient_validation_units" for item in blockers):
        state = "blocked_validation_units"
    elif split_blockers:
        state = "blocked_split_feasibility"
    return {
        "state": state,
        "training_authorized": False,
        "training_started": False,
        "counts": {
            "strong_e2": strong,
            "weighted_e2": weighted,
            "e4_families": len(e4_families),
            "matched_control_pairs": pairs,
            "independent_validation_units": independent_units,
        },
        "blockers": sorted_blockers(blockers),
    }


def validate_frozen_gaps(rows: Sequence[Mapping[str, object]], *, expected_count: int = 20) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    uids = {_text(row.get("gap_uid")) for row in rows}
    if len(rows) != expected_count or len(uids) != expected_count or "" in uids:
        blockers.append(blocker("frozen_gap_count_drift", "Round-19 must reconcile exactly 20 inherited gaps."))
    for row in rows:
        status = _text(row.get("status"))
        if status in FORBIDDEN_GAP_RESOLUTION or _text(row.get("resolution_basis")) in FORBIDDEN_GAP_RESOLUTION:
            blockers.append(blocker("gap_fabrication", "Frozen gap cannot be closed by prediction, narrative, or unadmitted simulation.", gap_uid=row.get("gap_uid")))
        if status not in GAP_STATES:
            blockers.append(blocker("gap_status", "Frozen-gap status is not canonical.", gap_uid=row.get("gap_uid")))
        if status.startswith("resolved_by_admitted") and not row.get("evidence_refs"):
            blockers.append(blocker("gap_resolution_evidence", "Admitted gap resolution requires evidence references.", gap_uid=row.get("gap_uid")))
    return sorted_blockers(blockers)


def validate_training_eligibility(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for row in rows:
        role = _text(row.get("role"))
        if role not in TRAINING_ROLES:
            blockers.append(blocker("training_role", "Training eligibility role is not canonical."))
        if role == "validation" and row.get("validation_only") is not True:
            blockers.append(blocker("validation_in_training", "Validation rows must remain validation-only."))
        if row.get("partition") == "final_held_out" and role not in {"validation", "stress_only"}:
            blockers.append(blocker("held_out_training", "Final held-out evidence cannot enter training roles."))
    blockers.extend(audit_split_leakage(rows))
    return sorted_blockers(blockers)


def validate_round19_invariants(state: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if state.get("strict_public_only_flag") is not True:
        blockers.append(blocker("public_policy", "strict_public_only_flag must remain true."))
    if int(state.get("controlled_source_resolution_count", 0) or 0) != 0:
        blockers.append(blocker("controlled_source", "controlled_source_resolution_count must remain zero."))
    if int(state.get("final_held_out_access_count", 0) or 0) != 0:
        blockers.append(blocker("held_out_access", "final held-out access count must remain zero."))
    if state.get("training_authorized") is not False or state.get("training_started") is not False:
        blockers.append(blocker("training_started", "Training is outside Round-19 authority."))
    if state.get("evidence_fabric_sealed") is not False:
        blockers.append(blocker("fabric_sealed", "Evidence Fabric must remain unsealed."))
    if state.get("dt_core_status") != "not_evaluated":
        blockers.append(blocker("dt_core_evaluated", "DT_CORE must remain unevaluated."))
    return sorted_blockers(blockers)
