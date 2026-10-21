"""Typed, fail-closed evidence relationship contracts for Round-12.

The module validates relationship semantics only. It does not materialize
scientific evidence, infer pairings, or authorize training and evaluation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


SCHEMA_VERSIONS = {
    "node": "neuronsifter-evidence-node-v1",
    "relation": "neuronsifter-evidence-relation-v1",
    "pairing": "neuronsifter-evidence-pairing-decision-v1",
    "support_domain": "neuronsifter-evidence-support-domain-v1",
    "eligibility": "neuronsifter-evidence-eligibility-v1",
    "conflict": "neuronsifter-evidence-conflict-v1",
    "provenance": "neuronsifter-evidence-relation-provenance-v1",
    "graph": "neuronsifter-evidence-relationship-graph-v1",
    "validation": "neuronsifter-dt-round12-validation-v1",
    "gate": "neuronsifter-dt-round12-gate-v2",
}

LEGACY_ROUND12_GATE_SCHEMA_VERSION = "neuronsifter-dt-round12-gate-v1"
ROUND12_GATE_SCHEMA_VERSION = SCHEMA_VERSIONS["gate"]

NODE_TYPES = frozenset(
    {
        "publication",
        "accession",
        "study",
        "cohort",
        "biological_unit",
        "sample",
        "observation",
        "state_snapshot",
        "structured_action",
        "perturbation_endpoint",
        "trajectory",
        "mechanistic_model",
        "simulator_output",
        "tool_request",
        "tool_result",
        "hypothetical_scenario",
        "claim",
        "hypothesis",
    }
)

RELATION_TYPES = frozenset(
    {
        "same_unit_longitudinal_observation",
        "same_unit_action_endpoint",
        "cross_sectional_state_observation",
        "simulator_generated_trajectory",
        "published_mechanistic_model_output",
        "static_perturbation_endpoint",
        "tool_derived_evidence",
        "hypothetical_action_scenario",
        "semantic_alignment_only",
        "publication_reports_observation",
        "accession_contains_observation",
        "study_contains_biological_unit",
        "action_applied_to_biological_unit",
        "observation_measured_at_time",
        "supports_claim",
        "refutes_claim",
        "conflicts_with",
        "derived_from",
        "supersedes",
        "duplicates",
        "artifact_production_lineage",
        "version_successor",
    }
)

PAIRING_DECISIONS = frozenset(
    {
        "valid_longitudinal_pair",
        "valid_action_endpoint_pair",
        "valid_cross_sectional_comparison",
        "valid_simulator_trajectory_pair",
        "valid_model_output_pair",
        "semantic_alignment_only",
        "insufficient_identity",
        "incompatible_context",
        "incompatible_time",
        "incompatible_action",
        "incompatible_biological_unit",
        "unsupported_pairing",
    }
)

IDENTITY_AXES = (
    "publication_uid",
    "accession_uid",
    "study_uid",
    "cohort_uid",
    "biological_unit_uid",
    "sample_uid",
    "organism",
    "disease_state",
    "tissue",
    "brain_region",
    "compartment",
    "cell_type",
    "modality",
    "assay",
    "action_uid",
    "target_program_uid",
    "dose",
    "route",
    "schedule",
    "timepoint",
    "trajectory_family_uid",
    "simulator_family_uid",
    "tool_family_uid",
)

IDENTITY_AXIS_STATES = frozenset(
    {
        "exact_match",
        "compatible",
        "different_but_allowed",
        "different_and_blocking",
        "unknown",
        "not_applicable",
    }
)

SUPPORT_DOMAIN_FIELDS = (
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
    "source_scope",
)

ELIGIBILITY_ROLES = (
    "eligible_for_state_pretraining",
    "eligible_for_action_endpoint_training",
    "eligible_for_trajectory_training",
    "eligible_for_calibration",
    "eligible_for_validation",
    "eligible_for_stress_test",
    "eligible_for_claim_support",
    "eligible_for_hypothesis_generation",
)

CONFLICT_TYPES = frozenset(
    {
        "numeric_disagreement",
        "directional_disagreement",
        "identity_conflict",
        "unit_conflict",
        "time_conflict",
        "action_conflict",
        "model_family_disagreement",
        "source_provenance_conflict",
        "claim_interpretation_conflict",
    }
)

CONFLICT_STATES = frozenset(
    {
        "unresolved",
        "context_explained",
        "version_resolved",
        "superseded",
        "retained_for_stress",
        "invalidated",
    }
)

LINEAGE_RELATION_TYPES = frozenset(
    {"derived_from", "supersedes", "artifact_production_lineage", "version_successor"}
)
STRONG_PAIRING_DECISIONS = frozenset(
    {
        "valid_longitudinal_pair",
        "valid_action_endpoint_pair",
        "valid_simulator_trajectory_pair",
        "valid_model_output_pair",
    }
)
EMPIRICAL_ELIGIBILITY_ROLES = frozenset(
    {
        "eligible_for_state_pretraining",
        "eligible_for_action_endpoint_training",
        "eligible_for_trajectory_training",
        "eligible_for_calibration",
        "eligible_for_validation",
        "eligible_for_claim_support",
    }
)

BIOLOGICAL_UNIT_RELATIONS = frozenset(
    {"same_unit", "related_unit", "different_unit", "not_applicable", "unknown"}
)
TEMPORAL_RELATIONS = frozenset(
    {
        "same_timepoint",
        "before_after",
        "ordered_series",
        "cross_sectional",
        "static_endpoint",
        "simulated_time",
        "timeless",
        "not_applicable",
        "unknown",
    }
)
ACTION_RELATIONS = frozenset(
    {
        "same_action",
        "action_to_endpoint",
        "different_action",
        "no_action",
        "hypothetical_action",
        "not_applicable",
        "unknown",
    }
)
CONTEXT_RELATIONS = frozenset(
    {"exact", "compatible", "different_but_allowed", "extrapolative", "unsupported", "stress_only", "unknown"}
)
QC_GRADE_RANK = {"unknown": 0, "ungraded": 0, "not_applicable": 0, "S": 1, "D": 2, "C": 3, "B": 4, "A": 5}

PAIRING_REQUIRED_AXES = {
    "valid_longitudinal_pair": frozenset(
        {
            "study_uid",
            "biological_unit_uid",
            "organism",
            "disease_state",
            "tissue",
            "modality",
            "assay",
            "timepoint",
        }
    ),
    "valid_action_endpoint_pair": frozenset(
        {
            "study_uid",
            "biological_unit_uid",
            "organism",
            "disease_state",
            "tissue",
            "modality",
            "assay",
            "action_uid",
            "dose",
            "route",
            "timepoint",
        }
    ),
    "valid_cross_sectional_comparison": frozenset(
        {"study_uid", "organism", "disease_state", "tissue", "modality", "assay"}
    ),
    "valid_simulator_trajectory_pair": frozenset(
        {"action_uid", "timepoint", "trajectory_family_uid", "simulator_family_uid"}
    ),
    "valid_model_output_pair": frozenset({"timepoint", "simulator_family_uid"}),
}

REQUIRED_ACTION_ENDPOINT_EVIDENCE = frozenset(
    {
        "baseline_identity",
        "post_action_identity",
        "biological_unit_relationship",
        "action_identity",
        "dose",
        "route_or_context",
        "time_relationship",
        "measurement_compatibility",
        "source_provenance",
    }
)

NODE_CLASS_RULES = {
    "perturbation_endpoint": frozenset({"experimental_endpoint", "contract_validation_example"}),
    "trajectory": frozenset({"experimental_trajectory", "simulator_trajectory", "contract_validation_example"}),
    "simulator_output": frozenset({"simulator_output", "model_output", "contract_validation_example"}),
    "hypothetical_scenario": frozenset({"hypothetical", "contract_validation_example"}),
    "claim": frozenset({"semantic_claim", "contract_validation_example"}),
    "hypothesis": frozenset({"hypothesis", "contract_validation_example"}),
}

RELATION_PAIRING_RULES = {
    "same_unit_longitudinal_observation": frozenset({"valid_longitudinal_pair"}),
    "same_unit_action_endpoint": frozenset({"valid_action_endpoint_pair"}),
    "cross_sectional_state_observation": frozenset({"valid_cross_sectional_comparison"}),
    "simulator_generated_trajectory": frozenset({"valid_simulator_trajectory_pair"}),
    "published_mechanistic_model_output": frozenset({"valid_model_output_pair"}),
    "semantic_alignment_only": frozenset({"semantic_alignment_only"}),
    "hypothetical_action_scenario": frozenset({"unsupported_pairing"}),
}

GAP_RELATION_CONTRACTS = {
    ("real_static_state", "cross_sectional"): (
        "observation",
        "cross_sectional_state_observation",
        "study_uid;biological_unit_uid;organism;tissue;modality;assay",
        "state_pretraining",
    ),
    ("real_perturbation_endpoint", "same_unit_action_endpoint"): (
        "perturbation_endpoint",
        "same_unit_action_endpoint",
        "study_uid;biological_unit_uid;action_uid;dose;route;timepoint;modality;assay",
        "action_endpoint_training",
    ),
    ("real_longitudinal_trajectory", "same_unit_longitudinal"): (
        "observation",
        "same_unit_longitudinal_observation",
        "study_uid;biological_unit_uid;timepoint;modality;assay",
        "trajectory_training",
    ),
    ("legacy_simulator_trajectory", "simulator_generated"): (
        "trajectory",
        "simulator_generated_trajectory",
        "trajectory_family_uid;simulator_family_uid;timepoint;action_uid",
        "trajectory_training",
    ),
    ("clean_simulator_trajectory", "simulator_generated"): (
        "trajectory",
        "simulator_generated_trajectory",
        "trajectory_family_uid;simulator_family_uid;timepoint;action_uid",
        "trajectory_training",
    ),
    ("published_mechanistic_output", "simulator_generated"): (
        "simulator_output",
        "published_mechanistic_model_output",
        "simulator_family_uid;timepoint",
        "calibration",
    ),
    ("tool_result", "not_paired"): (
        "tool_result",
        "tool_derived_evidence",
        "tool_family_uid",
        "validation",
    ),
    ("hypothetical_scenario", "not_paired"): (
        "hypothetical_scenario",
        "hypothetical_action_scenario",
        "action_uid;target_program_uid;dose;route;schedule;timepoint",
        "hypothesis_generation",
    ),
}


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_uid(prefix: str, payload: object) -> str:
    return f"{prefix}_{canonical_sha256(payload)[:24]}"


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def blocker(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def sorted_blockers(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    unique = {canonical_json(dict(value)): dict(value) for value in values}
    return sorted(unique.values(), key=lambda row: (str(row.get("code", "")), canonical_json(row)))


@dataclass(frozen=True)
class EvidenceNode:
    schema_version: str
    node_uid: str
    node_type: str
    evidence_class: str
    qc_grade: str
    identity_axes: Mapping[str, object]
    provenance_ref: str
    train_eval_role: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRelation:
    schema_version: str
    relation_uid: str
    relation_type: str
    source_node_uid: str
    target_node_uid: str
    directionality: str
    biological_unit_relation: str
    temporal_relation: str
    action_relation: str
    context_relation: str
    support_domain_ref: str
    eligibility_ref: str
    provenance_ref: str
    confidence: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSupportDomain:
    schema_version: str
    support_domain_uid: str
    organism_scope: Sequence[str]
    disease_scope: Sequence[str]
    region_scope: Sequence[str]
    compartment_scope: Sequence[str]
    cell_type_scope: Sequence[str]
    modality_scope: Sequence[str]
    assay_scope: Sequence[str]
    action_scope: Sequence[str]
    dose_scope: Sequence[str]
    route_scope: Sequence[str]
    time_scope: Sequence[str]
    model_family_scope: Sequence[str]
    tool_family_scope: Sequence[str]
    source_scope: Sequence[str]
    generalization_status: str
    provenance_ref: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceEligibility:
    schema_version: str
    eligibility_uid: str
    subject_uid: str
    eligible_for_state_pretraining: bool
    eligible_for_action_endpoint_training: bool
    eligible_for_trajectory_training: bool
    eligible_for_calibration: bool
    eligible_for_validation: bool
    eligible_for_stress_test: bool
    eligible_for_claim_support: bool
    eligible_for_hypothesis_generation: bool
    reasons: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceConflict:
    schema_version: str
    conflict_uid: str
    conflict_type: str
    member_uids: Sequence[str]
    state: str
    typed_context_difference: Mapping[str, object]
    provenance_ref: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidencePairingDecision:
    schema_version: str
    pairing_uid: str
    source_node_uid: str
    target_node_uid: str
    decision: str
    identity_axis_results: Mapping[str, str]
    required_identity_axes: Sequence[str]
    pairing_evidence: Mapping[str, object]
    blockers: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRelationProvenance:
    schema_version: str
    provenance_uid: str
    source_artifact_uids: Sequence[str]
    producer_uid: str
    source_schema_versions: Sequence[str]
    content_hashes: Mapping[str, str]
    provenance_status: str
    source_uri: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_node(node_value: Mapping[str, object]) -> list[dict[str, object]]:
    node = dict(node_value)
    uid = str(node.get("node_uid", ""))
    blockers: list[dict[str, object]] = []
    if node.get("schema_version") != SCHEMA_VERSIONS["node"]:
        blockers.append(blocker("unknown_schema_version", "EvidenceNode schema_version is not canonical.", node_uid=uid))
    node_type = str(node.get("node_type", ""))
    if node_type not in NODE_TYPES:
        blockers.append(blocker("unknown_node_type", "EvidenceNode uses an unknown canonical node type.", node_uid=uid, value=node_type))
    if not uid:
        blockers.append(blocker("missing_node_uid", "EvidenceNode requires a non-empty node_uid."))
    evidence_class = str(node.get("evidence_class", ""))
    allowed_classes = NODE_CLASS_RULES.get(node_type)
    if allowed_classes is not None and evidence_class not in allowed_classes:
        blockers.append(
            blocker(
                "implicit_node_type_conversion",
                "The node evidence class is incompatible with its canonical node type.",
                node_uid=uid,
                node_type=node_type,
                evidence_class=evidence_class,
            )
        )
    if not evidence_class or not str(node.get("qc_grade", "")):
        blockers.append(blocker("missing_evidence_class_or_qc", "EvidenceNode must preserve evidence class and QC grade.", node_uid=uid))
    axes = _mapping(node.get("identity_axes"))
    unknown_axes = sorted(set(axes) - set(IDENTITY_AXES))
    if unknown_axes:
        blockers.append(blocker("unknown_identity_axis", "EvidenceNode contains an undeclared identity axis.", node_uid=uid, axes=unknown_axes))
    if not str(node.get("provenance_ref", "")):
        blockers.append(blocker("missing_node_provenance_ref", "EvidenceNode requires provenance_ref.", node_uid=uid))
    if node_type == "hypothetical_scenario" and evidence_class not in {"hypothetical", "contract_validation_example"}:
        blockers.append(blocker("hypothetical_treated_as_evidence", "A hypothetical scenario cannot become empirical evidence.", node_uid=uid))
    return sorted_blockers(blockers)


def validate_provenance(value: Mapping[str, object]) -> list[dict[str, object]]:
    provenance = dict(value)
    uid = str(provenance.get("provenance_uid", ""))
    blockers: list[dict[str, object]] = []
    if provenance.get("schema_version") != SCHEMA_VERSIONS["provenance"]:
        blockers.append(blocker("unknown_schema_version", "Relation provenance schema_version is not canonical.", provenance_uid=uid))
    if not uid:
        blockers.append(blocker("missing_provenance_uid", "Relation provenance requires a UID."))
    if not str(provenance.get("producer_uid", "")):
        blockers.append(blocker("missing_producer", "Relation provenance requires an explicit producer.", provenance_uid=uid))
    if provenance.get("provenance_status") not in {"verified", "unknown", "rejected"}:
        blockers.append(blocker("invalid_provenance_status", "Relation provenance status is invalid.", provenance_uid=uid))
    hashes = _mapping(provenance.get("content_hashes"))
    for artifact_uid, digest in hashes.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            blockers.append(blocker("invalid_provenance_hash", "Relation provenance contains an invalid SHA-256.", provenance_uid=uid, artifact_uid=artifact_uid))
    if not _list(provenance.get("source_artifact_uids")) or not hashes:
        blockers.append(blocker("incomplete_provenance", "Relation provenance requires source artifacts and hashes.", provenance_uid=uid))
    return sorted_blockers(blockers)


def validate_support_domain(value: Mapping[str, object]) -> list[dict[str, object]]:
    domain = dict(value)
    uid = str(domain.get("support_domain_uid", ""))
    blockers: list[dict[str, object]] = []
    if domain.get("schema_version") != SCHEMA_VERSIONS["support_domain"]:
        blockers.append(blocker("unknown_schema_version", "Support-domain schema_version is not canonical.", support_domain_uid=uid))
    unknown_in_domain_fields: list[str] = []
    for field in SUPPORT_DOMAIN_FIELDS:
        values = domain.get(field)
        if not isinstance(values, list) or not values or any(not str(item) for item in values):
            blockers.append(blocker("incomplete_support_domain", "Every support-domain scope must be explicit.", support_domain_uid=uid, field=field))
            continue
        if any(str(item).strip().lower() in {"unknown", "unresolved", "unspecified"} for item in values):
            unknown_in_domain_fields.append(field)
    if domain.get("generalization_status") not in {"in_domain", "extrapolative", "unsupported", "stress_only"}:
        blockers.append(blocker("invalid_support_generalization", "Support-domain generalization status is invalid.", support_domain_uid=uid))
    if domain.get("generalization_status") == "in_domain" and unknown_in_domain_fields:
        blockers.append(
            blocker(
                "unknown_in_domain_support_scope",
                "An in-domain support contract cannot contain unknown scope values.",
                support_domain_uid=uid,
                fields=sorted(unknown_in_domain_fields),
            )
        )
    if not str(domain.get("provenance_ref", "")):
        blockers.append(blocker("missing_support_domain_provenance", "Support domain requires provenance_ref.", support_domain_uid=uid))
    return sorted_blockers(blockers)


def validate_conflict(value: Mapping[str, object]) -> list[dict[str, object]]:
    conflict = dict(value)
    uid = str(conflict.get("conflict_uid", ""))
    blockers: list[dict[str, object]] = []
    if conflict.get("schema_version") != SCHEMA_VERSIONS["conflict"]:
        blockers.append(blocker("unknown_schema_version", "Conflict schema_version is not canonical.", conflict_uid=uid))
    if conflict.get("conflict_type") not in CONFLICT_TYPES:
        blockers.append(blocker("unknown_conflict_type", "Conflict uses an unknown type.", conflict_uid=uid))
    state = str(conflict.get("state", ""))
    if state not in CONFLICT_STATES:
        blockers.append(blocker("unknown_conflict_state", "Conflict uses an unknown state.", conflict_uid=uid))
    members = [str(item) for item in _list(conflict.get("member_uids"))]
    if len(set(members)) < 2:
        blockers.append(blocker("invalid_conflict_members", "Conflict requires at least two distinct members.", conflict_uid=uid))
    if state == "context_explained" and not _mapping(conflict.get("typed_context_difference")):
        blockers.append(blocker("untyped_context_explanation", "context_explained requires a typed context difference.", conflict_uid=uid))
    return sorted_blockers(blockers)


def validate_pairing_decision(value: Mapping[str, object]) -> list[dict[str, object]]:
    pairing = dict(value)
    uid = str(pairing.get("pairing_uid", ""))
    decision = str(pairing.get("decision", ""))
    blockers: list[dict[str, object]] = []
    if pairing.get("schema_version") != SCHEMA_VERSIONS["pairing"]:
        blockers.append(blocker("unknown_schema_version", "Pairing schema_version is not canonical.", pairing_uid=uid))
    if decision not in PAIRING_DECISIONS:
        blockers.append(blocker("unknown_pairing_decision", "Pairing uses an unknown canonical decision.", pairing_uid=uid, value=decision))
    axes = _mapping(pairing.get("identity_axis_results"))
    if set(axes) != set(IDENTITY_AXES):
        blockers.append(
            blocker(
                "identity_axis_inventory_mismatch",
                "Every pairing decision must record every canonical identity axis.",
                pairing_uid=uid,
                missing=sorted(set(IDENTITY_AXES) - set(axes)),
                extra=sorted(set(axes) - set(IDENTITY_AXES)),
            )
        )
    invalid_states = sorted({str(state) for state in axes.values()} - IDENTITY_AXIS_STATES)
    if invalid_states:
        blockers.append(blocker("invalid_identity_axis_state", "Pairing uses an invalid identity-axis outcome.", pairing_uid=uid, states=invalid_states))
    configured_required = {str(item) for item in _list(pairing.get("required_identity_axes"))}
    expected_required = set(PAIRING_REQUIRED_AXES.get(decision, ()))
    if configured_required != expected_required:
        blockers.append(blocker("required_identity_axis_mismatch", "Pairing required axes do not match its decision contract.", pairing_uid=uid, expected=sorted(expected_required), actual=sorted(configured_required)))
    if decision.startswith("valid_"):
        blocking_axes = sorted(
            axis
            for axis in expected_required
            if axes.get(axis) in {"unknown", "different_and_blocking", "not_applicable", None}
        )
        if blocking_axes:
            blockers.append(blocker("missing_required_identity_axis", "A required unknown or blocking identity axis prevents valid pairing.", pairing_uid=uid, axes=blocking_axes))
    if decision in {"valid_longitudinal_pair", "valid_action_endpoint_pair"} and axes.get("biological_unit_uid") != "exact_match":
        blockers.append(blocker("different_biological_unit_pairing", "Same-unit pairing requires exact biological-unit identity.", pairing_uid=uid))
    if decision == "valid_longitudinal_pair" and axes.get("timepoint") != "different_but_allowed":
        blockers.append(blocker("invalid_longitudinal_time", "Longitudinal pairing requires an ordered, different timepoint.", pairing_uid=uid))
    if decision == "valid_action_endpoint_pair":
        evidence = _mapping(pairing.get("pairing_evidence"))
        missing_evidence = sorted(field for field in REQUIRED_ACTION_ENDPOINT_EVIDENCE if not evidence.get(field))
        if missing_evidence:
            blockers.append(blocker("incomplete_action_endpoint_evidence", "Action-response pairing lacks explicit required evidence.", pairing_uid=uid, fields=missing_evidence))
    if decision in STRONG_PAIRING_DECISIONS and _list(pairing.get("blockers")):
        blockers.append(blocker("valid_pairing_has_blockers", "A strong pairing decision cannot retain blockers.", pairing_uid=uid))
    return sorted_blockers(blockers)


def validate_relation(
    value: Mapping[str, object],
    *,
    nodes: Mapping[str, Mapping[str, object]],
    domains: Mapping[str, Mapping[str, object]],
    provenances: Mapping[str, Mapping[str, object]],
    pairings: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    relation = dict(value)
    uid = str(relation.get("relation_uid", ""))
    relation_type = str(relation.get("relation_type", ""))
    source_uid = str(relation.get("source_node_uid", ""))
    target_uid = str(relation.get("target_node_uid", ""))
    blockers: list[dict[str, object]] = []
    if relation.get("schema_version") != SCHEMA_VERSIONS["relation"]:
        blockers.append(blocker("unknown_schema_version", "Relation schema_version is not canonical.", relation_uid=uid))
    if relation_type not in RELATION_TYPES:
        blockers.append(blocker("unknown_relationship_type", "Relation uses an unknown canonical type.", relation_uid=uid, value=relation_type))
    if source_uid not in nodes or target_uid not in nodes:
        blockers.append(blocker("missing_relation_node", "Relation references a missing node.", relation_uid=uid, source_node_uid=source_uid, target_node_uid=target_uid))
    if source_uid == target_uid and relation_type in LINEAGE_RELATION_TYPES:
        blockers.append(blocker("self_derived_edge", "Lineage relations cannot be self-referential.", relation_uid=uid))
    if relation.get("directionality") not in {"directed", "bidirectional"}:
        blockers.append(blocker("invalid_directionality", "Relation directionality is invalid.", relation_uid=uid))
    if relation.get("biological_unit_relation") not in BIOLOGICAL_UNIT_RELATIONS:
        blockers.append(blocker("invalid_biological_unit_relation", "Relation biological-unit semantics are invalid.", relation_uid=uid))
    if relation.get("temporal_relation") not in TEMPORAL_RELATIONS:
        blockers.append(blocker("invalid_temporal_relation", "Relation temporal semantics are invalid.", relation_uid=uid))
    if relation.get("action_relation") not in ACTION_RELATIONS:
        blockers.append(blocker("invalid_action_relation", "Relation action semantics are invalid.", relation_uid=uid))
    if relation.get("context_relation") not in CONTEXT_RELATIONS:
        blockers.append(blocker("invalid_context_relation", "Relation context relation is invalid.", relation_uid=uid))
    domain_uid = str(relation.get("support_domain_ref", ""))
    domain = domains.get(domain_uid)
    if domain is None:
        blockers.append(blocker("missing_support_domain", "Relation requires an explicit support-domain reference.", relation_uid=uid, support_domain_ref=domain_uid))
    else:
        context = str(relation.get("context_relation", ""))
        domain_status = str(domain.get("generalization_status", ""))
        expected_domain_status = context if context in {"extrapolative", "unsupported", "stress_only"} else "in_domain"
        if domain_status != expected_domain_status:
            blockers.append(
                blocker(
                    "support_domain_context_mismatch",
                    "Relation context and support-domain generalization status disagree.",
                    relation_uid=uid,
                    context_relation=context,
                    support_domain_status=domain_status,
                )
            )
    provenance_uid = str(relation.get("provenance_ref", ""))
    provenance = provenances.get(provenance_uid)
    if provenance is None:
        blockers.append(blocker("missing_relation_provenance", "Relation requires explicit provenance.", relation_uid=uid, provenance_ref=provenance_uid))
    confidence = relation.get("confidence")
    if not isinstance(confidence, (float, int)) or isinstance(confidence, bool) or not 0.0 <= float(confidence) <= 1.0:
        blockers.append(blocker("invalid_relation_confidence", "Relation confidence must be numeric in [0, 1].", relation_uid=uid))

    pairing = pairings.get((source_uid, target_uid))
    allowed_pairings = RELATION_PAIRING_RULES.get(relation_type)
    if allowed_pairings is not None:
        if pairing is None:
            blockers.append(blocker("missing_pairing_decision", "Typed relation requires an explicit pairing decision.", relation_uid=uid))
        elif str(pairing.get("decision", "")) not in allowed_pairings:
            blockers.append(blocker("relation_pairing_mismatch", "Relation type and pairing decision are incompatible.", relation_uid=uid, decision=pairing.get("decision")))

    if provenance is not None and provenance.get("provenance_status") != "verified":
        decision = str(_mapping(pairing).get("decision", ""))
        if decision in STRONG_PAIRING_DECISIONS or relation.get("status") == "active":
            blockers.append(blocker("unknown_provenance_strong_relation", "Unknown provenance cannot enter a strong active relation.", relation_uid=uid))

    source_type = str(_mapping(nodes.get(source_uid)).get("node_type", ""))
    target_type = str(_mapping(nodes.get(target_uid)).get("node_type", ""))
    target_class = str(_mapping(nodes.get(target_uid)).get("evidence_class", ""))
    if relation_type == "same_unit_longitudinal_observation" and {source_type, target_type} - {"observation", "state_snapshot"}:
        blockers.append(blocker("cross_sectional_treated_as_longitudinal", "Longitudinal relation requires observation/state-snapshot nodes.", relation_uid=uid))
    if relation_type == "same_unit_longitudinal_observation" and (
        relation.get("biological_unit_relation") != "same_unit"
        or relation.get("temporal_relation") not in {"before_after", "ordered_series"}
    ):
        blockers.append(blocker("longitudinal_relation_semantics_mismatch", "Longitudinal relation fields do not encode same-unit ordered time.", relation_uid=uid))
    if relation_type == "same_unit_action_endpoint" and (
        target_type != "perturbation_endpoint"
        or relation.get("biological_unit_relation") != "same_unit"
        or relation.get("action_relation") != "action_to_endpoint"
        or relation.get("temporal_relation") not in {"before_after", "static_endpoint"}
    ):
        blockers.append(blocker("action_endpoint_relation_semantics_mismatch", "Action-endpoint relation fields are inconsistent.", relation_uid=uid))
    if relation_type == "cross_sectional_state_observation" and relation.get("temporal_relation") != "cross_sectional":
        blockers.append(blocker("cross_sectional_relation_semantics_mismatch", "Cross-sectional relation must declare cross-sectional time.", relation_uid=uid))
    if relation_type == "static_perturbation_endpoint" and target_type != "perturbation_endpoint":
        blockers.append(blocker("static_endpoint_type_mismatch", "Static endpoint relation must target perturbation_endpoint.", relation_uid=uid))
    if relation_type == "static_perturbation_endpoint" and (
        relation.get("temporal_relation") != "static_endpoint" or relation.get("action_relation") != "action_to_endpoint"
    ):
        blockers.append(blocker("static_endpoint_relation_semantics_mismatch", "Static endpoint relation fields are inconsistent.", relation_uid=uid))
    if relation_type == "simulator_generated_trajectory" and target_type != "trajectory":
        blockers.append(blocker("static_endpoint_treated_as_trajectory", "Simulator trajectory relation must target trajectory.", relation_uid=uid))
    if relation_type == "simulator_generated_trajectory" and (
        source_type != "simulator_output"
        or target_class not in {"simulator_trajectory", "contract_validation_example"}
        or relation.get("temporal_relation") != "simulated_time"
    ):
        blockers.append(blocker("simulator_trajectory_semantics_mismatch", "Simulator trajectory relation cannot consume experimental trajectory evidence.", relation_uid=uid))
    if relation_type == "published_mechanistic_model_output" and (source_type != "mechanistic_model" or target_type != "simulator_output"):
        blockers.append(blocker("published_model_output_semantics_mismatch", "Published model output must link a mechanistic model to simulator output.", relation_uid=uid))
    if relation_type == "tool_derived_evidence" and (source_type != "tool_request" or target_type != "tool_result"):
        blockers.append(blocker("tool_relation_semantics_mismatch", "Tool-derived evidence must link tool_request to tool_result.", relation_uid=uid))
    if relation_type == "hypothetical_action_scenario" and source_type != "hypothetical_scenario":
        blockers.append(blocker("hypothetical_relation_type_mismatch", "Hypothetical relation must originate from hypothetical_scenario.", relation_uid=uid))
    if relation_type == "hypothetical_action_scenario" and relation.get("action_relation") != "hypothetical_action":
        blockers.append(blocker("hypothetical_relation_semantics_mismatch", "Hypothetical scenario must retain hypothetical action semantics.", relation_uid=uid))
    if relation_type == "semantic_alignment_only" and pairing and pairing.get("decision") != "semantic_alignment_only":
        blockers.append(blocker("semantic_alignment_upgraded", "Semantic alignment cannot become biological pairing.", relation_uid=uid))
    if relation_type in {"supports_claim", "refutes_claim"} and target_type != "claim":
        blockers.append(blocker("claim_relation_target_mismatch", "Claim support/refutation must target a claim node.", relation_uid=uid))
    return sorted_blockers(blockers)


def derive_eligibility(
    *,
    subject_uid: str,
    node: Mapping[str, object],
    relation: Mapping[str, object] | None,
    pairing: Mapping[str, object] | None,
    domain: Mapping[str, object] | None,
    provenance: Mapping[str, object] | None,
    conflicts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Derive the maximum allowed role eligibility for one typed subject."""

    values = {role: False for role in ELIGIBILITY_ROLES}
    reasons: list[str] = []
    node_type = str(node.get("node_type", ""))
    relation_type = str(_mapping(relation).get("relation_type", ""))
    pairing_decision = str(_mapping(pairing).get("decision", ""))
    provenance_verified = _mapping(provenance).get("provenance_status") == "verified"
    domain_status = str(_mapping(domain).get("generalization_status", "unsupported"))
    unresolved_conflict = any(str(conflict.get("state", "")) == "unresolved" for conflict in conflicts)
    qc_grade = str(node.get("qc_grade", ""))
    qc_training = qc_grade in {"A", "B"}
    qc_calibration = qc_grade in {"A", "B", "C"}
    qc_validation = qc_grade in {"A", "B"}

    if not provenance_verified:
        reasons.append("unknown_provenance_blocks_strong_roles")
    if domain_status in {"unsupported", "extrapolative"}:
        reasons.append("support_domain_not_in_domain")
    if unresolved_conflict:
        reasons.append("unresolved_conflict_blocks_strong_roles")

    typed_domain = _mapping(domain)
    domain_complete = all(
        isinstance(typed_domain.get(field), list)
        and typed_domain[field]
        and not any(str(item).strip().lower() in {"unknown", "unresolved", "unspecified"} for item in typed_domain[field])
        for field in SUPPORT_DOMAIN_FIELDS
    )
    strong_base = provenance_verified and domain_status == "in_domain" and domain_complete and not unresolved_conflict
    if node_type in {"observation", "state_snapshot"} and strong_base and qc_training and relation_type != "semantic_alignment_only":
        values["eligible_for_state_pretraining"] = True
    if node_type in {"observation", "state_snapshot"} and strong_base and qc_calibration and relation_type != "semantic_alignment_only":
        values["eligible_for_calibration"] = True
    if node_type in {"observation", "state_snapshot"} and strong_base and qc_validation and relation_type != "semantic_alignment_only":
        values["eligible_for_validation"] = True
    if node_type == "perturbation_endpoint" and strong_base and qc_training and pairing_decision == "valid_action_endpoint_pair":
        values["eligible_for_action_endpoint_training"] = True
    if node_type == "perturbation_endpoint" and strong_base and qc_calibration and pairing_decision == "valid_action_endpoint_pair":
        values["eligible_for_calibration"] = True
    if node_type == "perturbation_endpoint" and strong_base and qc_validation and pairing_decision == "valid_action_endpoint_pair":
        values["eligible_for_validation"] = True
        reasons.append("static_endpoint_is_terminal_or_directional_anchor_only")
    if node_type == "trajectory" and strong_base and qc_training and relation_type == "simulator_generated_trajectory":
        values["eligible_for_trajectory_training"] = True
        reasons.append("simulator_trajectory_is_not_real_experimental_truth")
    if node_type == "trajectory" and strong_base and qc_calibration and relation_type == "simulator_generated_trajectory":
        values["eligible_for_calibration"] = True
    if node_type == "hypothetical_scenario":
        values["eligible_for_hypothesis_generation"] = provenance_verified
        reasons.append("hypothetical_scenario_excluded_from_empirical_roles")
    if relation_type == "semantic_alignment_only":
        values["eligible_for_hypothesis_generation"] = provenance_verified
        reasons.append("semantic_alignment_excluded_from_endpoint_and_trajectory_training")
    if domain_status == "stress_only" or unresolved_conflict or qc_grade == "S":
        values["eligible_for_stress_test"] = provenance_verified
    if relation_type in {"supports_claim", "refutes_claim"} and strong_base and qc_validation and pairing_decision not in {"semantic_alignment_only", "unsupported_pairing", "insufficient_identity"}:
        values["eligible_for_claim_support"] = True
    if node.get("evidence_class") == "contract_validation_example":
        for role in EMPIRICAL_ELIGIBILITY_ROLES:
            values[role] = False
        reasons.append("contract_validation_example_excluded_from_empirical_roles")
    return {
        "schema_version": SCHEMA_VERSIONS["eligibility"],
        "eligibility_uid": stable_uid("eligibility", {"subject_uid": subject_uid, **values}),
        "subject_uid": subject_uid,
        **values,
        "reasons": sorted(set(reasons)) or ["role_policy_evaluated"],
        "status": "evaluated_fail_closed",
    }


def validate_eligibility(
    value: Mapping[str, object],
    *,
    maximum_allowed: Mapping[str, object],
) -> list[dict[str, object]]:
    eligibility = dict(value)
    uid = str(eligibility.get("eligibility_uid", ""))
    blockers: list[dict[str, object]] = []
    if eligibility.get("schema_version") != SCHEMA_VERSIONS["eligibility"]:
        blockers.append(blocker("unknown_schema_version", "Eligibility schema_version is not canonical.", eligibility_uid=uid))
    for role in ELIGIBILITY_ROLES:
        if not isinstance(eligibility.get(role), bool):
            blockers.append(blocker("invalid_eligibility_value", "Eligibility roles must be boolean.", eligibility_uid=uid, role=role))
        elif eligibility.get(role) and not maximum_allowed.get(role):
            blockers.append(blocker("eligibility_upgrade", "Eligibility exceeds the maximum allowed by typed evidence semantics.", eligibility_uid=uid, role=role))
    return sorted_blockers(blockers)


def validate_evidence_union_non_upgrade(union: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate conservative role propagation and typed-conflict preservation."""

    members = [_mapping(value) for value in _list(union.get("member_evidence"))]
    downstream = _mapping(union.get("canonical_downstream_eligibility"))
    blockers: list[dict[str, object]] = []
    for role in ELIGIBILITY_ROLES:
        member_values = [_mapping(member.get("relationship_eligibility")).get(role) is True for member in members]
        if downstream.get(role) is True and (not member_values or not all(member_values)):
            blockers.append(blocker("evidence_union_eligibility_upgrade", "Evidence Union upgraded role eligibility.", union_uid=union.get("union_uid"), role=role))
    member_pairings = sorted({str(member.get("pairing_decision", "unsupported_pairing")) for member in members})
    expected_pairing = member_pairings[0] if len(member_pairings) == 1 else "unsupported_pairing"
    if downstream.get("pairing_decision") != expected_pairing:
        blockers.append(blocker("evidence_union_pairing_upgrade", "Evidence Union changed or upgraded biological pairing semantics.", union_uid=union.get("union_uid")))
    member_classes = sorted({str(member.get("evidence_class", member.get("evidence_type", "unknown"))) for member in members})
    expected_class = member_classes[0] if len(member_classes) == 1 else "mixed_no_upgrade"
    if downstream.get("evidence_class") != expected_class:
        blockers.append(blocker("evidence_union_class_upgrade", "Evidence Union changed or upgraded evidence class.", union_uid=union.get("union_uid")))
    member_grades = [str(member.get("qc_grade", "unknown")) for member in members]
    expected_grade = min(member_grades, key=lambda value: QC_GRADE_RANK.get(value, 0)) if member_grades else "unknown"
    downstream_grade = str(downstream.get("qc_grade", "unknown"))
    if QC_GRADE_RANK.get(downstream_grade, 0) > QC_GRADE_RANK.get(expected_grade, 0):
        blockers.append(blocker("evidence_union_qc_upgrade", "Evidence Union upgraded the conservative QC grade.", union_uid=union.get("union_uid")))
    input_conflicts: dict[str, set[str]] = defaultdict(set)
    for member in members:
        for conflict in [_mapping(value) for value in _list(member.get("typed_conflicts"))]:
            conflict_uid = str(conflict.get("conflict_uid", ""))
            if conflict_uid:
                input_conflicts[conflict_uid].add(canonical_json(conflict))
    preserved = {
        str(conflict.get("conflict_uid")): canonical_json(conflict)
        for conflict in [_mapping(value) for value in _list(union.get("preserved_typed_conflicts"))]
        if conflict.get("conflict_uid")
    }
    collision_uids = sorted(uid for uid, payloads in input_conflicts.items() if len(payloads) > 1)
    if collision_uids:
        blockers.append(blocker("evidence_union_conflict_collision", "Evidence Union received conflicting payloads for one typed conflict UID.", union_uid=union.get("union_uid"), conflict_uids=collision_uids))
    missing = sorted(uid for uid, payloads in input_conflicts.items() if preserved.get(uid) not in payloads)
    if missing:
        blockers.append(blocker("evidence_union_conflict_loss", "Evidence Union dropped or changed typed conflicts.", union_uid=union.get("union_uid"), missing=missing))
    return sorted_blockers(blockers)


def _duplicate_uid_blockers(kind: str, rows: Sequence[Mapping[str, object]], uid_field: str) -> list[dict[str, object]]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        uid = str(row.get(uid_field, ""))
        if uid in seen:
            duplicates.add(uid)
        seen.add(uid)
    return [blocker("duplicate_uid", "Canonical graph UIDs must be unique within each object family.", object_kind=kind, uid=uid) for uid in sorted(duplicates)]


def _lineage_blockers(relations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    lineage = [dict(row) for row in relations if row.get("relation_type") in LINEAGE_RELATION_TYPES and row.get("status") == "active"]
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    successor_by_target: dict[tuple[str, str], list[str]] = defaultdict(list)
    for relation in lineage:
        source = str(relation.get("source_node_uid", ""))
        target = str(relation.get("target_node_uid", ""))
        relation_type = str(relation.get("relation_type", ""))
        nodes.update({source, target})
        if source == target:
            blockers.append(blocker("self_derived_edge", "Lineage relation cannot be self-referential.", relation_uid=relation.get("relation_uid")))
            continue
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
            indegree.setdefault(source, 0)
        if relation_type in {"supersedes", "version_successor"}:
            successor_by_target[(relation_type, target)].append(source)
    for (relation_type, target), successors in sorted(successor_by_target.items()):
        if len(set(successors)) > 1:
            blockers.append(blocker("duplicate_active_successor", "A lineage target has multiple active successors.", relation_type=relation_type, target_node_uid=target, successor_uids=sorted(set(successors))))
    queue = deque(sorted(node for node in nodes if indegree[node] == 0))
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        blockers.append(blocker("lineage_cycle", "Typed artifact/version lineage contains a cycle.", affected_node_count=len(nodes) - visited))
    return blockers


def validate_graph(graph_value: Mapping[str, object]) -> dict[str, object]:
    graph = dict(graph_value)
    nodes = [_mapping(value) for value in _list(graph.get("nodes"))]
    relations = [_mapping(value) for value in _list(graph.get("relations"))]
    domains = [_mapping(value) for value in _list(graph.get("support_domains"))]
    eligibilities = [_mapping(value) for value in _list(graph.get("eligibilities"))]
    conflicts = [_mapping(value) for value in _list(graph.get("conflicts"))]
    pairings = [_mapping(value) for value in _list(graph.get("pairing_decisions"))]
    provenances = [_mapping(value) for value in _list(graph.get("provenances"))]
    unions = [_mapping(value) for value in _list(graph.get("evidence_unions"))]

    blockers: list[dict[str, object]] = []
    if graph.get("schema_version") != SCHEMA_VERSIONS["graph"]:
        blockers.append(blocker("unknown_schema_version", "Graph schema_version is not canonical."))
    for kind, rows, uid_field in (
        ("node", nodes, "node_uid"),
        ("relation", relations, "relation_uid"),
        ("support_domain", domains, "support_domain_uid"),
        ("eligibility", eligibilities, "eligibility_uid"),
        ("conflict", conflicts, "conflict_uid"),
        ("pairing", pairings, "pairing_uid"),
        ("provenance", provenances, "provenance_uid"),
    ):
        blockers.extend(_duplicate_uid_blockers(kind, rows, uid_field))

    node_map = {str(row.get("node_uid", "")): row for row in nodes}
    domain_map = {str(row.get("support_domain_uid", "")): row for row in domains}
    eligibility_map = {str(row.get("eligibility_uid", "")): row for row in eligibilities}
    provenance_map = {str(row.get("provenance_uid", "")): row for row in provenances}
    pairing_map = {(str(row.get("source_node_uid", "")), str(row.get("target_node_uid", ""))): row for row in pairings}
    relation_map = {str(row.get("relation_uid", "")): row for row in relations}

    pairing_keys: set[tuple[str, str]] = set()
    duplicate_pairing_keys: set[tuple[str, str]] = set()
    for row in pairings:
        key = (str(row.get("source_node_uid", "")), str(row.get("target_node_uid", "")))
        if key in pairing_keys:
            duplicate_pairing_keys.add(key)
        pairing_keys.add(key)
    for source_uid, target_uid in sorted(duplicate_pairing_keys):
        blockers.append(blocker("duplicate_pairing_decision", "One node pair cannot have multiple active pairing decisions.", source_node_uid=source_uid, target_node_uid=target_uid))

    for row in nodes:
        blockers.extend(validate_node(row))
        if str(row.get("provenance_ref", "")) not in provenance_map:
            blockers.append(blocker("missing_node_provenance", "Node provenance reference does not resolve.", node_uid=row.get("node_uid")))
    for row in provenances:
        blockers.extend(validate_provenance(row))
    for row in domains:
        blockers.extend(validate_support_domain(row))
        if str(row.get("provenance_ref", "")) not in provenance_map:
            blockers.append(blocker("missing_support_domain_provenance", "Support-domain provenance reference does not resolve.", support_domain_uid=row.get("support_domain_uid")))
    for row in conflicts:
        blockers.extend(validate_conflict(row))
        if str(row.get("provenance_ref", "")) not in provenance_map:
            blockers.append(blocker("missing_conflict_provenance", "Conflict provenance reference does not resolve.", conflict_uid=row.get("conflict_uid")))
        missing_members = sorted(
            str(member_uid)
            for member_uid in _list(row.get("member_uids"))
            if str(member_uid) not in relation_map and str(member_uid) not in node_map
        )
        if missing_members:
            blockers.append(blocker("missing_conflict_member", "Conflict references missing typed graph members.", conflict_uid=row.get("conflict_uid"), member_uids=missing_members))
    for row in pairings:
        blockers.extend(validate_pairing_decision(row))
        if str(row.get("source_node_uid", "")) not in node_map or str(row.get("target_node_uid", "")) not in node_map:
            blockers.append(blocker("missing_pairing_node", "Pairing decision references a missing node.", pairing_uid=row.get("pairing_uid")))
    for row in relations:
        blockers.extend(validate_relation(row, nodes=node_map, domains=domain_map, provenances=provenance_map, pairings=pairing_map))
        eligibility_uid = str(row.get("eligibility_ref", ""))
        eligibility = eligibility_map.get(eligibility_uid)
        if eligibility is None:
            blockers.append(blocker("missing_relation_eligibility", "Relation eligibility reference does not resolve.", relation_uid=row.get("relation_uid")))
            continue
        pairing = pairing_map.get((str(row.get("source_node_uid", "")), str(row.get("target_node_uid", ""))))
        relation_conflicts = [conflict for conflict in conflicts if str(row.get("relation_uid", "")) in {str(item) for item in _list(conflict.get("member_uids"))}]
        maximum = derive_eligibility(
            subject_uid=str(row.get("relation_uid", "")),
            node=node_map.get(str(row.get("target_node_uid", "")), {}),
            relation=row,
            pairing=pairing,
            domain=domain_map.get(str(row.get("support_domain_ref", ""))),
            provenance=provenance_map.get(str(row.get("provenance_ref", ""))),
            conflicts=relation_conflicts,
        )
        blockers.extend(validate_eligibility(eligibility, maximum_allowed=maximum))
    for eligibility in eligibilities:
        subject_uid = str(eligibility.get("subject_uid", ""))
        if subject_uid not in relation_map and subject_uid not in node_map:
            blockers.append(blocker("missing_eligibility_subject", "Eligibility subject does not resolve.", eligibility_uid=eligibility.get("eligibility_uid"), subject_uid=subject_uid))
    blockers.extend(_lineage_blockers(relations))
    for union in unions:
        blockers.extend(validate_evidence_union_non_upgrade(union))

    blockers = sorted_blockers(blockers)
    return {
        "schema_version": SCHEMA_VERSIONS["validation"],
        "graph_uid": str(graph.get("graph_uid", "")),
        "object_counts": {
            "nodes": len(nodes),
            "relations": len(relations),
            "support_domains": len(domains),
            "eligibilities": len(eligibilities),
            "conflicts": len(conflicts),
            "pairing_decisions": len(pairings),
            "provenances": len(provenances),
            "evidence_unions": len(unions),
        },
        "blockers": blockers,
        "blocker_count": len(blockers),
        "status": "passed" if not blockers else "blocked",
        "evidence_fabric_sealed": False,
        "dt_core": "not_evaluated",
    }


def _gap_contract_for_adapter(adapter: Mapping[str, object]) -> tuple[str, str, str, str]:
    defaults = _mapping(adapter.get("evidence_defaults"))
    evidence_type = str(defaults.get("evidence_type", ""))
    pairing_semantics = str(defaults.get("pairing_semantics", ""))
    configured = GAP_RELATION_CONTRACTS.get((evidence_type, pairing_semantics))
    if configured is not None:
        return configured
    field_sources = _mapping(adapter.get("field_sources"))
    if evidence_type == "ontology_or_prior" and pairing_semantics == "not_paired" and "action_uid" in field_sources:
        return (
            "structured_action",
            "semantic_alignment_only",
            "action_uid;target_program_uid;dose;route;schedule",
            "hypothesis_generation",
        )
    if evidence_type == "ontology_or_prior" and pairing_semantics == "not_paired":
        return (
            "claim",
            "derived_from",
            "publication_uid;accession_uid;study_uid",
            "hypothesis_generation",
        )
    raise ValueError(
        "No explicit Round-12 gap relationship contract for adapter "
        f"{adapter.get('adapter_id', '<missing>')}: evidence_type={evidence_type}, pairing_semantics={pairing_semantics}"
    )


def frozen_gap_rows(
    validation: Mapping[str, object],
    source_adapters: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    missing = [
        _mapping(value)
        for value in _list(validation.get("blockers"))
        if _mapping(value).get("code") == "missing_required_frozen_input"
    ]
    adapter_map = {str(adapter.get("adapter_id", "")): dict(adapter) for adapter in source_adapters}
    if len(adapter_map) != len(source_adapters) or "" in adapter_map:
        raise ValueError("Evidence Fabric source adapters require unique non-empty adapter_id values.")
    rows: list[dict[str, object]] = []
    for item in sorted(missing, key=lambda row: (str(row.get("adapter_id", "")), str(row.get("path", "")))):
        path = str(item.get("path", ""))
        adapter_id = str(item.get("adapter_id", ""))
        adapter = adapter_map.get(adapter_id)
        if adapter is None:
            raise ValueError(f"Frozen gap adapter is absent from the canonical Evidence Fabric config: {adapter_id}")
        configured_path = str(adapter.get("path", ""))
        if configured_path != path:
            raise ValueError(f"Frozen gap path differs from the canonical adapter contract: {adapter_id}")
        missing_node_type, missing_relation_type, identity_axes, role = _gap_contract_for_adapter(adapter)
        defaults = _mapping(adapter.get("evidence_defaults"))
        reader_type = str(adapter.get("reader_type", ""))
        rows.append(
            {
                "gap_uid": stable_uid("round12_gap", {"adapter_id": adapter_id, "path": path}),
                "missing_node_type": missing_node_type,
                "missing_relationship_type": missing_relation_type,
                "required_identity_axes": identity_axes,
                "available_evidence": (
                    "canonical adapter contract "
                    f"evidence_type={defaults.get('evidence_type')}; "
                    f"pairing_semantics={defaults.get('pairing_semantics')}; "
                    f"time_semantics={defaults.get('time_semantics')}; reader_type={reader_type}"
                ),
                "blocked_pairing_reason": "required configured source artifact absent; no pairing inferred",
                "affected_downstream_role": role,
                "resolution_owner": adapter_id,
                "status": "materialization_required" if reader_type == "tsv_records" else "source_required",
            }
        )
    return rows


def validate_frozen_gap_preservation(
    source_validation: Mapping[str, object],
    mapped_rows: Sequence[Mapping[str, object]],
    source_adapters: Sequence[Mapping[str, object]],
    *,
    expected_count: int = 20,
) -> list[dict[str, object]]:
    source = [
        _mapping(value)
        for value in _list(source_validation.get("blockers"))
        if _mapping(value).get("code") == "missing_required_frozen_input"
    ]
    blockers: list[dict[str, object]] = []
    try:
        expected_rows = frozen_gap_rows(source_validation, source_adapters)
    except ValueError as error:
        expected_rows = []
        blockers.append(blocker("frozen_gap_adapter_contract", str(error)))
    if len(source) != expected_count:
        blockers.append(blocker("frozen_gap_source_count_drift", "Frozen source gap count changed.", expected=expected_count, actual=len(source)))
    if len(mapped_rows) != expected_count:
        blockers.append(blocker("frozen_gap_mapping_count_drift", "Frozen gap mapping count changed.", expected=expected_count, actual=len(mapped_rows)))
    source_owners = {str(row.get("adapter_id", "")) for row in source}
    mapped_owners = {str(row.get("resolution_owner", "")) for row in mapped_rows}
    if source_owners != mapped_owners:
        blockers.append(blocker("frozen_gap_owner_drift", "Frozen gap mapping does not preserve exact adapter ownership.", missing=sorted(source_owners - mapped_owners), extra=sorted(mapped_owners - source_owners)))
    normalized_rows = [{key: str(value) for key, value in row.items()} for row in mapped_rows]
    normalized_expected = [{key: str(value) for key, value in row.items()} for row in expected_rows]
    if normalized_rows != normalized_expected:
        blockers.append(blocker("frozen_gap_mapping_content_drift", "Frozen gap mapping differs from explicit canonical adapter contracts."))
    allowed_statuses = {
        "relationship_missing",
        "identity_incomplete",
        "provenance_incomplete",
        "support_domain_incomplete",
        "materialization_required",
        "source_required",
        "contract_conflict",
    }
    invalid_statuses = sorted({str(row.get("status", "")) for row in mapped_rows} - allowed_statuses)
    if invalid_statuses:
        blockers.append(blocker("invalid_frozen_gap_status", "Frozen gap mapping uses invalid statuses.", values=invalid_statuses))
    return sorted_blockers(blockers)


def gate_state_for_blockers(blockers: Sequence[Mapping[str, object]]) -> str:
    """Return the legacy v1 Gate state for compatibility with preserved artifacts."""

    codes = {str(item.get("code", "")) for item in blockers}
    if not codes:
        return "typed_evidence_relationships_resolved"
    routing = (
        ({"execution_identity", "provenance", "producer", "schema_version"}, "blocked_relation_provenance"),
        ({"identity", "pairing", "biological_unit", "longitudinal"}, "blocked_ambiguous_pairing"),
        ({"duplicate_uid", "collision"}, "blocked_identity_collision"),
        ({"support_domain", "support_generalization"}, "blocked_support_domain"),
        ({"eligibility_upgrade"}, "blocked_eligibility_upgrade"),
        ({"conflict_loss"}, "blocked_conflict_loss"),
        ({"lineage_cycle", "self_derived", "active_successor"}, "blocked_lineage_cycle"),
        ({"frozen_gap"}, "blocked_frozen_gap_drift"),
    )
    for fragments, state in routing:
        if any(any(fragment in code for fragment in fragments) for code in codes):
            return state
    return "blocked_ambiguous_pairing"


def round12_blocker_class(code: str) -> str:
    """Classify a Gate blocker without conflating execution with relationship provenance."""

    if code in {
        "execution_identity_artifact_missing",
        "execution_identity_hash_mismatch",
        "execution_identity_not_exact_clean",
        "execution_identity_schema_mismatch",
    }:
        return "execution_identity"
    if code in {
        "core_xfail_not_zero",
        "full_suite_execution_artifact_missing",
        "full_suite_execution_identity_mismatch",
        "full_suite_not_verified",
        "full_suite_terminal_evidence_invalid",
        "mandatory_skipped_not_zero",
        "test_inventory_incomplete",
        "test_inventory_missing",
    }:
        return "full_suite_verification"
    routing = (
        ({"provenance", "producer", "schema_version"}, "relationship_provenance"),
        ({"identity", "pairing", "biological_unit", "longitudinal"}, "relationship_pairing"),
        ({"duplicate_uid", "collision"}, "relationship_identity_collision"),
        ({"support_domain", "support_generalization"}, "relationship_support_domain"),
        ({"eligibility_upgrade"}, "relationship_eligibility"),
        ({"conflict_loss"}, "relationship_conflict"),
        ({"lineage_cycle", "self_derived", "active_successor"}, "relationship_lineage"),
        ({"frozen_gap"}, "relationship_frozen_gap"),
        ({"schema_registry", "json_schema"}, "relationship_schema"),
    )
    for fragments, blocker_class in routing:
        if any(fragment in code for fragment in fragments):
            return blocker_class
    return "relationship_contract"


def round12_gate_status(blockers: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Build the v2 typed Gate status while keeping the v1 state mapper intact."""

    codes = [str(item.get("code", "")) for item in blockers]
    blocker_classes = sorted({round12_blocker_class(code) for code in codes})
    relationship_blocked = any(value.startswith("relationship_") for value in blocker_classes)
    execution_blocked = "execution_identity" in blocker_classes
    full_suite_blocked = "full_suite_verification" in blocker_classes
    if not codes:
        state = "typed_evidence_relationships_resolved"
    elif not relationship_blocked and (execution_blocked or full_suite_blocked):
        state = "blocked_execution_verification"
    else:
        state = gate_state_for_blockers(blockers)
    return {
        "round12_gate_schema_version": ROUND12_GATE_SCHEMA_VERSION,
        "status": state,
        "relation_contract_status": "blocked" if relationship_blocked else "passed_local",
        "execution_identity_status": "blocked" if execution_blocked else "passed",
        "full_suite_status": "not_verified" if full_suite_blocked else "passed",
        "overall_gate_status": "blocked" if codes else "accepted",
        "blocker_classes": blocker_classes,
    }
