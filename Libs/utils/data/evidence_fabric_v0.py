"""Governed Evidence Fabric V0 admission and release contracts.

The module materializes existing evidence only.  It never executes tools,
simulators, training, or counterfactuals, and it never seals a release.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from Libs.utils.data.evidence_relationships import (
    CONFLICT_STATES,
    RELATION_TYPES,
    validate_conflict,
    validate_eligibility,
    validate_node,
    validate_pairing_decision,
    validate_support_domain,
)


ADMISSION_SCHEMA_VERSION = "neuronsifter-evidence-fabric-admission-record-v1"
RELEASE_SCHEMA_VERSION = "neuronsifter-evidence-fabric-release-manifest-v1"
FABRIC_VERSION = "evidence_fabric_v0"
LAYERS = frozenset({"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"})
ADMISSION_PHASES = (
    "discovered",
    "source_verified",
    "license_or_access_verified",
    "content_hashed",
    "parsed",
    "canonicalized",
    "typed",
    "relationship_validated",
    "support_domain_validated",
    "QC_graded",
    "eligibility_assigned",
    "admitted_or_rejected",
)
TERMINAL_STATES = frozenset(
    {
        "admitted_strong",
        "admitted_weighted",
        "admitted_weak",
        "admitted_stress",
        "admitted_legacy_restricted",
        "quarantined",
        "rejected_license",
        "rejected_provenance",
        "rejected_schema",
        "rejected_identity",
        "rejected_relationship",
        "rejected_support_domain",
        "rejected_hash",
    }
)
ADMITTED_STATES = frozenset(state for state in TERMINAL_STATES if state.startswith("admitted_"))
QC_RANK = {"D": 0, "S": 1, "C": 2, "B": 3, "A": 4, "not_applicable": 0}
LEGACY_ROLES = frozenset(
    {"legacy_dynamics_pretraining", "numerical_dynamics_only", "stress_only", "quarantined"}
)
CONFLICT_RESOLUTIONS = frozenset(
    {
        "equivalent_duplicate",
        "versioned_successor",
        "context_explained",
        "superseded",
        "retained_conflict",
        "invalidated",
        "quarantined",
    }
)
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
STRONG_STATES = frozenset({"admitted_strong", "admitted_weighted"})
STRONG_NUMERIC_ROLES = frozenset(
    {
        "numeric_ground_truth",
        "main_pretraining",
        "action_endpoint_training",
        "trajectory_training",
        "headline_training",
    }
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_uid(prefix: str, value: object) -> str:
    return f"{prefix}_{canonical_sha256(value)[:24]}"


def is_sha256(value: object) -> bool:
    return bool(SHA256_RE.fullmatch(str(value)))


def blocker(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def sorted_blockers(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    unique = {canonical_json(dict(value)): dict(value) for value in values}
    return sorted(unique.values(), key=lambda row: (str(row.get("code", "")), canonical_json(row)))


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def evidence_fabric_v0_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("evidence_fabric_v0")
    if not isinstance(value, Mapping):
        raise KeyError("evidence_fabric_v0")
    return dict(value)


def validate_v0_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    try:
        value = evidence_fabric_v0_config(config)
    except KeyError:
        return [blocker("missing_v0_config", "Evidence Fabric V0 configuration is absent.")]
    blockers: list[dict[str, object]] = []
    if value.get("version") != FABRIC_VERSION:
        blockers.append(blocker("invalid_v0_version", "Evidence Fabric V0 version is not canonical."))
    if set(_list(value.get("layers"))) != LAYERS:
        blockers.append(blocker("invalid_layer_vocabulary", "Evidence layers must be exactly E0 through E7."))
    if tuple(_list(value.get("admission_phases"))) != ADMISSION_PHASES:
        blockers.append(blocker("invalid_admission_state_machine", "Admission phases are not canonical."))
    if set(_list(value.get("terminal_states"))) != TERMINAL_STATES:
        blockers.append(blocker("invalid_terminal_state_vocabulary", "Admission terminal states are not canonical."))
    source_paths = _mapping(value.get("source_paths"))
    for name in (
        "foundation_observation_blocks",
        "legacy_trajectory_catalog",
        "mechanistic_model_catalog",
        "round12_relationship_graph",
        "round13_action_observation_graph",
    ):
        path = Path(str(source_paths.get(name, "")))
        if not str(path) or path.is_absolute():
            blockers.append(blocker("invalid_source_path", "Configured source paths must be relative.", source=name))
    output_paths = _mapping(value.get("output_paths"))
    for name, raw_path in output_paths.items():
        path = Path(str(raw_path))
        if not str(path) or path.is_absolute():
            blockers.append(blocker("invalid_output_path", "Configured output paths must be relative.", output=name))
    if value.get("sealed") is not False or value.get("status") != "materialized_unsealed":
        blockers.append(blocker("invalid_unsealed_policy", "V0 must remain materialized_unsealed with sealed=false."))
    return sorted_blockers(blockers)


def canonicalize_entity_mapping(mapping: Mapping[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    result = deepcopy(dict(mapping))
    status = str(result.get("mapping_status", ""))
    blockers: list[dict[str, object]] = []
    if status in {"one_to_many", "many_to_one", "low_confidence", "unresolved"}:
        if result.get("canonical_uid") and not result.get("resolution_artifact_ref"):
            blockers.append(
                blocker(
                    "ambiguous_mapping_collapsed",
                    "Ambiguous entity mappings cannot be collapsed without a resolution artifact.",
                    source_uid=result.get("source_uid"),
                )
            )
    elif status not in {"exact", "mapped", "not_applicable"}:
        blockers.append(blocker("invalid_mapping_status", "Entity mapping status is not canonical.", value=status))
    return result, sorted_blockers(blockers)


def validate_admission_role_boundary(value: "AdmissionInput") -> list[dict[str, object]]:
    """Reject layer/role promotions that change the scientific meaning of evidence."""

    blockers: list[dict[str, object]] = []
    role = value.train_eval_role
    if value.layer == "E1" and role in {"longitudinal_transition", "trajectory_training"}:
        blockers.append(
            blocker(
                "cross_sectional_promoted_to_longitudinal",
                "E1 static observations cannot authorize longitudinal or trajectory roles.",
            )
        )
    if value.layer == "E2" and role in {"trajectory_training", "longitudinal_transition"}:
        blockers.append(
            blocker(
                "static_endpoint_promoted_to_trajectory",
                "E2 perturbation endpoints remain static endpoints, not trajectories.",
            )
        )
    if value.layer == "E4" and value.legacy_role is not None:
        blockers.append(
            blocker(
                "legacy_trajectory_promoted_to_e4",
                "A legacy-role trajectory cannot be reclassified as action-complete E4 evidence.",
            )
        )
    if value.layer == "E3" and value.empirical_truth:
        blockers.append(
            blocker(
                "simulator_output_promoted_to_experimental_truth",
                "Simulator output cannot become experimental truth.",
            )
        )
    if value.qc_grade == "C" and role in STRONG_NUMERIC_ROLES:
        blockers.append(
            blocker(
                "grade_c_numeric_ground_truth",
                "QC-C evidence cannot enter strong numeric supervision.",
            )
        )
    if value.qc_grade == "S" and role in STRONG_NUMERIC_ROLES:
        blockers.append(
            blocker(
                "grade_s_headline_training",
                "QC-S evidence is restricted to stress/OOD/conflict roles.",
            )
        )
    if value.layer == "E7" and (value.empirical_truth or value.item_type in {"observation", "canonical_observation"}):
        blockers.append(
            blocker(
                "hypothetical_scenario_admitted_as_observation",
                "E7 hypothetical scenarios cannot become empirical observations.",
            )
        )
    return sorted_blockers(blockers)


def derive_eligibility(
    *,
    subject_uid: str,
    layer: str,
    qc_grade: str,
    terminal_state: str,
    pairing_decision: str = "unsupported_pairing",
    legacy_role: str | None = None,
    empirical_truth: bool = True,
    unresolved_conflict: bool = False,
) -> dict[str, object]:
    admitted = terminal_state in ADMITTED_STATES
    strong_qc = qc_grade in {"A", "B"}
    stress = terminal_state in {"admitted_stress", "admitted_legacy_restricted"} or qc_grade == "S"
    state = admitted and layer == "E1" and strong_qc and empirical_truth and not unresolved_conflict
    endpoint = (
        admitted
        and layer == "E2"
        and strong_qc
        and empirical_truth
        and pairing_decision in {"valid_action_endpoint_pair", "valid_group_level_action_endpoint"}
        and not unresolved_conflict
    )
    trajectory = admitted and layer == "E4" and strong_qc and not unresolved_conflict
    calibration = admitted and layer in {"E1", "E2", "E6"} and qc_grade in {"A", "B"} and empirical_truth
    reasons = [f"layer_{layer.lower()}", f"terminal_{terminal_state}", f"qc_{qc_grade}"]
    if layer == "E3":
        reasons.append(f"legacy_role_{legacy_role or 'unresolved'}")
    if layer == "E7" or not empirical_truth:
        reasons.append("not_empirical_truth")
    if unresolved_conflict:
        reasons.append("unresolved_conflict_preserved")
    return {
        "schema_version": "neuronsifter-evidence-eligibility-v1",
        "eligibility_uid": stable_uid("eligibility", {"subject_uid": subject_uid, "layer": layer, "terminal": terminal_state}),
        "subject_uid": subject_uid,
        "eligible_for_state_pretraining": state,
        "eligible_for_action_endpoint_training": endpoint,
        "eligible_for_trajectory_training": trajectory,
        "eligible_for_calibration": calibration,
        "eligible_for_validation": admitted and empirical_truth and qc_grade in {"A", "B"} and layer in {"E1", "E2"},
        "eligible_for_stress_test": admitted and (stress or unresolved_conflict),
        "eligible_for_claim_support": admitted and empirical_truth and qc_grade == "A" and layer in {"E1", "E2"} and not unresolved_conflict,
        "eligible_for_hypothesis_generation": admitted and layer in {"E0", "E5", "E7"},
        "reasons": reasons,
        "status": "evaluated_fail_closed",
    }


def validate_eligibility_non_upgrade(
    predecessor: Mapping[str, object], successor: Mapping[str, object]
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    previous_qc = str(predecessor.get("qc_grade", "not_applicable"))
    current_qc = str(successor.get("qc_grade", "not_applicable"))
    if QC_RANK.get(current_qc, -1) > QC_RANK.get(previous_qc, -1):
        blockers.append(blocker("evidence_union_qc_upgrade", "Evidence Union cannot increase QC."))
    previous = _mapping(predecessor.get("eligibility"))
    current = _mapping(successor.get("eligibility"))
    for key, value in current.items():
        if key.startswith("eligible_for_") and value is True and previous.get(key) is not True:
            blockers.append(
                blocker("evidence_union_eligibility_upgrade", "Evidence Union cannot increase eligibility.", field=key)
            )
    if successor.get("pairing_decision") != predecessor.get("pairing_decision") and successor.get("pairing_decision") in {
        "valid_longitudinal_pair",
        "valid_action_endpoint_pair",
        "valid_cross_sectional_comparison",
        "valid_simulator_trajectory_pair",
        "valid_model_output_pair",
    }:
        blockers.append(blocker("evidence_union_pairing_creation", "Evidence Union cannot create pairing."))
    return sorted_blockers(blockers)


@dataclass(frozen=True)
class AdmissionInput:
    layer: str
    item_type: str
    source_uid: str
    official_source_or_internal_producer: str
    source_version: str
    publication_or_accession: str
    license_or_access_status: str
    retrieval_or_generation_date: str
    raw_hash: str
    processed_hash: str
    parser_version: str
    canonicalization_version: str
    producer_uid: str
    producer_command: str
    train_eval_role: str
    holdout_group: str
    support_domain_ref: str
    qc_grade: str
    schema_version: str
    identity: dict[str, object]
    payload: dict[str, object]
    relationship_refs: list[str]
    conflict_refs: list[str]
    pairing_decision: str = "unsupported_pairing"
    legacy_role: str | None = None
    empirical_truth: bool = True
    schema_valid: bool = True
    identity_valid: bool = True
    relationship_valid: bool = True
    support_domain_valid: bool = True
    action_observation_valid: bool = True
    action_complete: bool = False
    provenance_complete: bool = True
    hash_verified: bool = True
    mapping_valid: bool = True


def _terminal_for_input(value: AdmissionInput, blockers: Sequence[Mapping[str, object]]) -> str:
    codes = {str(item.get("code")) for item in blockers}
    if "license_or_access_unresolved" in codes:
        return "rejected_license"
    if codes & {"missing_source", "missing_producer", "unknown_provenance", "incomplete_provenance"}:
        return "rejected_provenance"
    if codes & {"missing_hash", "hash_mismatch"}:
        return "rejected_hash"
    if "schema_invalid" in codes:
        return "rejected_schema"
    if codes & {"identity_incomplete", "ambiguous_mapping_collapsed"}:
        return "rejected_identity"
    if "relationship_invalid" in codes:
        return "rejected_relationship"
    if "support_domain_invalid" in codes:
        return "rejected_support_domain"
    if blockers:
        return "quarantined"
    if value.layer == "E3":
        return "admitted_stress" if value.legacy_role == "stress_only" else "admitted_legacy_restricted"
    if value.layer in {"E0", "E5"}:
        return {"A": "admitted_strong", "B": "admitted_weighted", "C": "admitted_weak"}.get(
            value.qc_grade, "admitted_weak"
        )
    if value.layer == "E7" or value.qc_grade == "S" or not value.empirical_truth:
        return "admitted_stress"
    return {"A": "admitted_strong", "B": "admitted_weighted", "C": "admitted_weak"}.get(
        value.qc_grade, "admitted_weak"
    )


def admit_evidence(value: AdmissionInput) -> dict[str, object]:
    blockers = validate_admission_role_boundary(value)
    if value.layer not in LAYERS:
        blockers.append(blocker("schema_invalid", "Evidence layer is not canonical.", layer=value.layer))
    if not value.source_uid or not value.official_source_or_internal_producer or not value.source_version:
        blockers.append(blocker("missing_source", "Evidence source identity is incomplete."))
    if value.license_or_access_status not in {"verified_public", "verified_internal", "not_applicable_governed"}:
        blockers.append(blocker("license_or_access_unresolved", "License or access status is unresolved."))
    if not is_sha256(value.raw_hash) or not is_sha256(value.processed_hash):
        blockers.append(blocker("missing_hash", "Raw and processed content hashes are required."))
    elif not value.hash_verified:
        blockers.append(blocker("hash_mismatch", "A declared content hash did not verify."))
    if not value.schema_version or not value.schema_valid:
        blockers.append(blocker("schema_invalid", "Evidence schema validation failed."))
    if not value.identity_valid:
        blockers.append(blocker("identity_incomplete", "Required biological or action identity is incomplete."))
    if not value.mapping_valid:
        blockers.append(blocker("ambiguous_mapping_collapsed", "An ambiguous entity mapping was silently collapsed."))
    if not value.relationship_valid:
        blockers.append(blocker("relationship_invalid", "Typed relationship validation failed."))
    if not value.support_domain_ref or not value.support_domain_valid:
        blockers.append(blocker("support_domain_invalid", "Support domain is absent or invalid."))
    if not value.producer_uid or not value.producer_command:
        blockers.append(blocker("missing_producer", "Evidence producer identity and command are required."))
    if not value.provenance_complete:
        blockers.append(blocker("incomplete_provenance", "Evidence provenance is incomplete."))
    if not value.train_eval_role or not value.holdout_group:
        blockers.append(blocker("incomplete_provenance", "Evidence role and holdout group are required."))
    if value.qc_grade not in QC_RANK:
        blockers.append(blocker("schema_invalid", "QC grade is invalid.", qc_grade=value.qc_grade))
    if value.qc_grade == "D":
        blockers.append(blocker("qc_d_rejected", "QC-D evidence cannot be admitted."))
    if value.layer == "E3" and value.legacy_role not in LEGACY_ROLES:
        blockers.append(blocker("legacy_role_invalid", "Legacy trajectories require a restricted role."))
    if value.layer == "E4" and not (value.action_complete and value.provenance_complete):
        blockers.append(blocker("e4_action_incomplete", "E4 requires an existing action-complete, provenance-complete trajectory."))
    if value.layer == "E2" and not value.action_observation_valid:
        blockers.append(blocker("action_observation_invalid", "Perturbation endpoint action/observation contract is incomplete."))
    blockers = sorted_blockers(blockers)
    terminal = _terminal_for_input(value, blockers)
    candidate_seed = {
        "layer": value.layer,
        "item_type": value.item_type,
        "source_uid": value.source_uid,
        "processed_hash": value.processed_hash,
        "identity": value.identity,
    }
    candidate_uid = stable_uid("fabric_item", candidate_seed)
    eligibility = derive_eligibility(
        subject_uid=candidate_uid,
        layer=value.layer,
        qc_grade=value.qc_grade,
        terminal_state=terminal,
        pairing_decision=value.pairing_decision,
        legacy_role=value.legacy_role,
        empirical_truth=value.empirical_truth,
        unresolved_conflict=bool(value.conflict_refs),
    )
    first_failed_index = len(ADMISSION_PHASES) - 1
    if blockers:
        code = str(blockers[0]["code"])
        phase_by_code = {
            "missing_source": 0,
            "missing_producer": 1,
            "license_or_access_unresolved": 2,
            "missing_hash": 3,
            "hash_mismatch": 3,
            "schema_invalid": 4,
            "ambiguous_mapping_collapsed": 5,
            "identity_incomplete": 5,
            "relationship_invalid": 7,
            "support_domain_invalid": 8,
            "qc_d_rejected": 9,
        }
        first_failed_index = phase_by_code.get(code, 6)
    state_history = [
        {"state": phase, "status": "passed" if index < first_failed_index or not blockers else "blocked"}
        for index, phase in enumerate(ADMISSION_PHASES[:-1])
        if index <= first_failed_index or not blockers
    ]
    state_history.append({"state": "admitted_or_rejected", "status": terminal})
    base = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "candidate_uid": candidate_uid,
        "layer": value.layer,
        "item_type": value.item_type,
        "source_uid": value.source_uid,
        "official_source_or_internal_producer": value.official_source_or_internal_producer,
        "source_version": value.source_version,
        "publication_or_accession": value.publication_or_accession,
        "license_or_access_status": value.license_or_access_status,
        "retrieval_or_generation_date": value.retrieval_or_generation_date,
        "raw_hash": value.raw_hash,
        "processed_hash": value.processed_hash,
        "parser_version": value.parser_version,
        "canonicalization_version": value.canonicalization_version,
        "producer_uid": value.producer_uid,
        "producer_command": value.producer_command,
        "train_eval_role": value.train_eval_role,
        "holdout_group": value.holdout_group,
        "support_domain_ref": value.support_domain_ref,
        "qc_grade": value.qc_grade,
        "relationship_refs": sorted(set(value.relationship_refs)),
        "conflict_refs": sorted(set(value.conflict_refs)),
        "pairing_decision": value.pairing_decision,
        "legacy_role": value.legacy_role,
        "empirical_truth": value.empirical_truth,
        "eligibility": eligibility,
        "identity": deepcopy(value.identity),
        "payload": deepcopy(value.payload),
        "state_history": state_history,
        "terminal_state": terminal,
        "blockers": blockers,
    }
    return {**base, "content_hash": canonical_sha256(base)}


def validate_admission_record(value: Mapping[str, object]) -> list[dict[str, object]]:
    record = dict(value)
    blockers: list[dict[str, object]] = []
    required = {
        "schema_version",
        "candidate_uid",
        "layer",
        "item_type",
        "source_uid",
        "official_source_or_internal_producer",
        "source_version",
        "publication_or_accession",
        "license_or_access_status",
        "retrieval_or_generation_date",
        "raw_hash",
        "processed_hash",
        "parser_version",
        "canonicalization_version",
        "producer_uid",
        "producer_command",
        "train_eval_role",
        "holdout_group",
        "support_domain_ref",
        "qc_grade",
        "relationship_refs",
        "conflict_refs",
        "pairing_decision",
        "legacy_role",
        "empirical_truth",
        "eligibility",
        "identity",
        "payload",
        "state_history",
        "terminal_state",
        "blockers",
        "content_hash",
    }
    missing = sorted(required - set(record))
    if missing:
        blockers.append(blocker("admission_schema_missing_fields", "Admission record fields are missing.", fields=missing))
        return blockers
    if record["schema_version"] != ADMISSION_SCHEMA_VERSION or record["layer"] not in LAYERS:
        blockers.append(blocker("admission_schema_invalid", "Admission record schema or layer is invalid."))
    if record["terminal_state"] not in TERMINAL_STATES:
        blockers.append(blocker("admission_terminal_invalid", "Admission terminal state is invalid."))
    expected_hash = canonical_sha256({key: record[key] for key in record if key != "content_hash"})
    if record["content_hash"] != expected_hash:
        blockers.append(blocker("admission_content_hash_mismatch", "Admission record content hash is stale."))
    history = _list(record.get("state_history"))
    if not history or history[-1] != {"state": "admitted_or_rejected", "status": record["terminal_state"]}:
        blockers.append(blocker("admission_state_machine_incomplete", "Admission state history lacks its terminal transition."))
    eligibility = _mapping(record.get("eligibility"))
    blockers.extend(validate_eligibility(eligibility, maximum_allowed=eligibility))
    if record["layer"] == "E7" and any(
        eligibility.get(field) is True
        for field in (
            "eligible_for_state_pretraining",
            "eligible_for_action_endpoint_training",
            "eligible_for_trajectory_training",
            "eligible_for_calibration",
            "eligible_for_validation",
            "eligible_for_claim_support",
        )
    ):
        blockers.append(blocker("hypothetical_empirical_eligibility", "E7 cannot enter empirical truth roles."))
    if record["layer"] == "E3" and record["terminal_state"] in ADMITTED_STATES and record.get("legacy_role") not in LEGACY_ROLES:
        blockers.append(blocker("legacy_role_invalid", "Admitted E3 evidence lacks a restricted legacy role."))
    return sorted_blockers(blockers)


def validate_typed_catalogs(
    *,
    nodes: Sequence[Mapping[str, object]],
    relations: Sequence[Mapping[str, object]],
    pairings: Sequence[Mapping[str, object]],
    support_domains: Sequence[Mapping[str, object]],
    eligibilities: Sequence[Mapping[str, object]],
    conflicts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    seen_nodes: set[str] = set()
    for node in nodes:
        uid = str(node.get("node_uid", ""))
        if not uid or uid in seen_nodes:
            blockers.append(blocker("duplicate_node_uid", "Node UIDs must be unique.", node_uid=uid))
        seen_nodes.add(uid)
        blockers.extend(validate_node(node))
    seen_relations: set[str] = set()
    for relation in relations:
        uid = str(relation.get("relation_uid", ""))
        if not uid or uid in seen_relations:
            blockers.append(blocker("duplicate_relation_uid", "Relation UIDs must be unique.", relation_uid=uid))
        seen_relations.add(uid)
        if relation.get("relation_type") not in RELATION_TYPES:
            blockers.append(blocker("unknown_relationship_type", "Relation type is not canonical.", relation_uid=uid))
        if not relation.get("source_node_uid") or not relation.get("target_node_uid") or relation.get("source_node_uid") == relation.get("target_node_uid"):
            blockers.append(blocker("invalid_relationship_endpoints", "Relations require distinct source and target UIDs.", relation_uid=uid))
    for domain in support_domains:
        blockers.extend(validate_support_domain(domain))
    for pairing in pairings:
        blockers.extend(validate_pairing_decision(pairing))
    for eligibility in eligibilities:
        blockers.extend(validate_eligibility(eligibility, maximum_allowed=eligibility))
    for conflict in conflicts:
        blockers.extend(validate_conflict(conflict))
        if conflict.get("state") not in CONFLICT_STATES:
            blockers.append(blocker("invalid_conflict_state", "Conflict state is not canonical."))
    return sorted_blockers(blockers)


def validate_conflict_preservation(
    source_conflicts: Sequence[Mapping[str, object]],
    preserved_conflicts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    source = {str(row.get("conflict_uid", "")): canonical_json(row) for row in source_conflicts}
    preserved = {str(row.get("conflict_uid", "")): canonical_json(row) for row in preserved_conflicts}
    missing = sorted(uid for uid, payload in source.items() if uid and preserved.get(uid) != payload)
    return (
        [blocker("conflict_dropped_during_deduplication", "Typed conflicts must be preserved byte-for-byte.", missing=missing)]
        if missing
        else []
    )


def validate_gap_resolution_evidence(
    gaps: Sequence[Mapping[str, object]],
    admissions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    admitted_uids = {
        str(row.get("candidate_uid", ""))
        for row in admissions
        if row.get("terminal_state") in ADMITTED_STATES
    }
    invalid: list[str] = []
    for gap in gaps:
        status = str(gap.get("current_status", ""))
        if status.startswith("resolved_"):
            resolution = str(gap.get("resolution_evidence", ""))
            if not resolution or resolution not in admitted_uids:
                invalid.append(str(gap.get("gap_uid", "")))
    return (
        [blocker("gap_closed_without_admission_evidence", "Gap resolution requires an admitted evidence UID.", gap_uids=sorted(invalid))]
        if invalid
        else []
    )


def validate_effective_sample_audit(value: Mapping[str, object]) -> list[dict[str, object]]:
    if value.get("row_count_not_used_as_effective_sample_count") is not True:
        return [blocker("row_count_used_as_effective_sample_count", "Raw row count cannot substitute for effective sample units.")]
    return []


def validate_independence_grouping(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_biological_unit: dict[str, set[str]] = {}
    for row in rows:
        biological_unit = str(row.get("biological_unit_group", ""))
        independence_group = str(row.get("independence_group", row.get("biological_unit_group", "")))
        if biological_unit:
            by_biological_unit.setdefault(biological_unit, set()).add(independence_group)
    split_units = sorted(uid for uid, groups in by_biological_unit.items() if len(groups) > 1)
    return (
        [blocker("duplicate_biological_units_split_as_independent", "One biological unit cannot span independent split groups.", biological_unit_uids=split_units)]
        if split_units
        else []
    )


def duplicate_and_conflict_audit(records: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    by_hash: dict[str, list[str]] = {}
    by_identity: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        by_hash.setdefault(str(record.get("processed_hash", "")), []).append(str(record.get("candidate_uid", "")))
        identity_key = canonical_json(_mapping(record.get("identity")))
        by_identity.setdefault(identity_key, []).append(record)
    duplicate_groups = [
        {"processed_hash": digest, "member_uids": sorted(uids), "resolution": "equivalent_duplicate"}
        for digest, uids in sorted(by_hash.items())
        if digest and len(uids) > 1
    ]
    conflicts: list[dict[str, object]] = []
    for identity_key, members in sorted(by_identity.items()):
        if len(members) < 2:
            continue
        payload_hashes = {canonical_sha256(member.get("payload")) for member in members}
        if len(payload_hashes) > 1:
            conflicts.append(
                {
                    "identity_hash": canonical_sha256(json.loads(identity_key)),
                    "member_uids": sorted(str(member["candidate_uid"]) for member in members),
                    "conflict_type": "numeric_or_context_disagreement",
                    "resolution": "retained_conflict",
                }
            )
    return (
        {
            "schema_version": "neuronsifter-dt-round14-duplicate-audit-v1",
            "duplicate_groups": duplicate_groups,
            "duplicate_group_count": len(duplicate_groups),
            "resolution_policy": "typed_no_majority_vote",
        },
        {
            "schema_version": "neuronsifter-dt-round14-conflict-audit-v1",
            "detected_conflicts": conflicts,
            "detected_conflict_count": len(conflicts),
            "resolution_policy": "typed_resolution_artifact_required",
        },
    )


def _parquet_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple, set)):
        return canonical_json(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def write_parquet_records(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_fields = {
        field
        for field in fields
        if any(isinstance(row.get(field), (Mapping, list, tuple, set)) for row in rows)
    }
    normalized = [
        {
            field: canonical_json(row.get(field)) if field in json_fields else _parquet_value(row.get(field))
            for field in fields
        }
        for row in rows
    ]
    if normalized:
        table = pa.Table.from_pylist(normalized)
    else:
        table = pa.table({field: pa.array([], type=pa.string()) for field in fields})
    pq.write_table(table, path, compression="zstd", write_statistics=True)


def read_parquet_records(path: Path) -> list[dict[str, object]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_uid(manifest_without_uid: Mapping[str, object]) -> str:
    return stable_uid("evidence_fabric_release", manifest_without_uid)


def validate_release_manifest(value: Mapping[str, object]) -> list[dict[str, object]]:
    manifest = dict(value)
    blockers: list[dict[str, object]] = []
    required = {
        "schema_version",
        "release_uid",
        "release_version",
        "parent_release",
        "created_at",
        "code_commit",
        "config_hash",
        "schema_registry_hash",
        "input_bundle_hash",
        "environment_hash",
        "source_manifest_hash",
        "node_catalog_hash",
        "relation_catalog_hash",
        "action_catalog_hash",
        "observation_catalog_hash",
        "conflict_catalog_hash",
        "gap_ledger_hash",
        "admission_summary",
        "status",
        "sealed",
    }
    missing = sorted(required - set(manifest))
    if missing:
        return [blocker("release_manifest_missing_fields", "Release manifest fields are missing.", fields=missing)]
    if manifest["schema_version"] != RELEASE_SCHEMA_VERSION or manifest["release_version"] != FABRIC_VERSION:
        blockers.append(blocker("release_manifest_schema_invalid", "Release manifest schema/version is invalid."))
    if manifest["sealed"] is not False or manifest["status"] != "materialized_unsealed":
        blockers.append(blocker("release_falsely_sealed", "Evidence Fabric V0 must remain materialized_unsealed."))
    for key in required:
        if key.endswith("_hash") and not is_sha256(manifest.get(key)):
            blockers.append(blocker("release_hash_invalid", "Release manifest hash is invalid.", field=key))
    seed = {key: manifest[key] for key in manifest if key != "release_uid"}
    if manifest["release_uid"] != release_uid(seed):
        blockers.append(blocker("release_uid_mismatch", "Release UID is not content-addressed to the manifest."))
    return sorted_blockers(blockers)


ROUND14_GATE_STATES = frozenset(
    {
        "evidence_fabric_v0_materialized",
        "blocked_source_provenance",
        "blocked_license_or_access",
        "blocked_hash_integrity",
        "blocked_canonicalization",
        "blocked_relationship_validation",
        "blocked_action_observation_validation",
        "blocked_eligibility_upgrade",
        "blocked_conflict_loss",
        "blocked_gap_fabrication",
        "blocked_independence_grouping",
        "blocked_execution_verification",
    }
)


def round14_gate_status(blockers_value: Sequence[Mapping[str, object]]) -> str:
    blockers = list(blockers_value)
    if not blockers:
        return "evidence_fabric_v0_materialized"
    codes = {str(item.get("code", "")) for item in blockers}
    ordered = (
        ("blocked_source_provenance", ("source", "producer", "provenance")),
        ("blocked_license_or_access", ("license", "access")),
        ("blocked_hash_integrity", ("hash",)),
        ("blocked_canonicalization", ("canonical", "mapping", "schema")),
        ("blocked_relationship_validation", ("relation", "pairing", "lineage")),
        ("blocked_action_observation_validation", ("action", "observation", "legacy", "hypothetical")),
        ("blocked_eligibility_upgrade", ("eligibility_upgrade", "qc_upgrade")),
        ("blocked_conflict_loss", ("conflict",)),
        ("blocked_gap_fabrication", ("gap",)),
        ("blocked_independence_grouping", ("independence", "effective_sample")),
        ("blocked_execution_verification", ("execution", "full_suite", "inventory", "focused", "prefull")),
    )
    for state, markers in ordered:
        if any(any(marker in code for marker in markers) for code in codes):
            return state
    return "blocked_execution_verification"


def admission_input_dict(value: AdmissionInput) -> dict[str, object]:
    """Return a serializable view for tests and audit debugging."""

    return asdict(value)
