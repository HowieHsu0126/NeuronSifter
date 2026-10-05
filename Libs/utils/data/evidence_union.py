"""Fail-closed artifact-level evidence union helpers.

The union accepts metadata-bearing artifacts. It never merges controller code,
inflates evidence grade, or interprets aggregate rows as independent samples.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import re
from typing import Any


EVIDENCE_UNION_SCHEMA_VERSION = "neuronsifter-evidence-union-v1"
ALLOWED_CONFLICT_STATES = {
    "no_conflict",
    "equivalent_duplicate",
    "versioned_successor",
    "superseded",
    "semantic_conflict",
    "unknown",
}
REJECTED_CONFLICT_STATES = {"semantic_conflict", "unknown"}
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
REQUIRED_MEMBER_FIELDS = {
    "artifact_uid",
    "artifact_hash",
    "producer_uid",
    "schema_version",
    "lineage_group",
    "provenance_status",
    "evidence_type",
    "qc_grade",
    "train_eval_role",
    "claim_eligibility",
    "independence_group",
    "status",
}
QC_GRADE_RANK = {"unknown": 0, "S": 1, "D": 2, "C": 3, "B": 4, "A": 5, "not_applicable": 0}
NONREAL_EVIDENCE_TYPES = {"fixture", "unit_fixture", "mock", "synthetic", "synthetic_substitute", "cached_standin"}
ACCEPTED_MEMBER_STATUSES = {"accepted", "passed", "verified"}


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _member_errors(member: Mapping[str, object]) -> list[str]:
    missing = sorted(field for field in REQUIRED_MEMBER_FIELDS if not member.get(field))
    errors = [f"missing_{field}" for field in missing]
    if member.get("artifact_hash") and not SHA256_PATTERN.fullmatch(str(member["artifact_hash"])):
        errors.append("invalid_artifact_hash")
    if member.get("provenance_status") not in {"verified", "unknown"}:
        errors.append("invalid_provenance_status")
    return errors


def _conflict_for_group(group: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    member_uids = sorted(str(member["artifact_uid"]) for member in members)
    if any(member.get("provenance_status") != "verified" for member in members):
        return {
            "lineage_group": group,
            "state": "unknown",
            "member_artifact_uids": member_uids,
            "reason": "A member has unknown or invalid provenance.",
        }
    if len(members) == 1:
        return {
            "lineage_group": group,
            "state": "no_conflict",
            "member_artifact_uids": member_uids,
            "reason": "Only one validated producer artifact is present in this lineage group.",
        }

    fingerprints = {str(member.get("semantic_fingerprint", "")) for member in members}
    if len(fingerprints) == 1 and "" not in fingerprints:
        return {
            "lineage_group": group,
            "state": "equivalent_duplicate",
            "member_artifact_uids": member_uids,
            "reason": "Multiple content-addressed artifacts declare the same semantic fingerprint.",
        }

    known_uids = set(member_uids)
    parent_links = {str(member.get("lineage_parent", "")) for member in members if member.get("lineage_parent")}
    if parent_links & known_uids:
        return {
            "lineage_group": group,
            "state": "versioned_successor",
            "member_artifact_uids": member_uids,
            "reason": "A member explicitly names another member as its lineage parent.",
        }

    superseded_uids = {
        str(uid)
        for member in members
        for uid in member.get("supersedes", [])
        if isinstance(uid, str)
    }
    if superseded_uids & known_uids:
        return {
            "lineage_group": group,
            "state": "superseded",
            "member_artifact_uids": member_uids,
            "reason": "A member explicitly declares an earlier member superseded.",
        }

    return {
        "lineage_group": group,
        "state": "semantic_conflict",
        "member_artifact_uids": member_uids,
        "reason": "Validated artifacts in the same lineage group have incompatible semantics without a version relation.",
    }


def _conservative_eligibility(members: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_types = sorted({str(member["evidence_type"]) for member in members})
    qc_grades = [str(member["qc_grade"]) for member in members]
    conservative_grade = min(qc_grades, key=lambda value: QC_GRADE_RANK.get(value, 0))
    roles = sorted({str(member["train_eval_role"]) for member in members})
    claim_eligibilities = sorted({str(member["claim_eligibility"]) for member in members})
    provenance_values = sorted({str(member["provenance_status"]) for member in members})
    independence_groups = sorted({str(member["independence_group"]) for member in members})
    contains_nonreal = any(value in NONREAL_EVIDENCE_TYPES for value in evidence_types)
    all_real = evidence_types == ["real"]
    all_claim_eligible = claim_eligibilities == ["eligible"]
    return {
        "evidence_type": "real" if all_real else "nonreal_or_mixed",
        "member_evidence_types": evidence_types,
        "fixture_or_nonreal_member_present": contains_nonreal,
        "qc_grade": conservative_grade,
        "member_qc_grades": sorted(set(qc_grades)),
        "train_eval_role": roles[0] if len(roles) == 1 else "mixed_restricted",
        "member_train_eval_roles": roles,
        "claim_eligibility": "eligible" if all_claim_eligible and all_real else "blocked",
        "member_claim_eligibilities": claim_eligibilities,
        "provenance_status": "verified" if provenance_values == ["verified"] else "unknown",
        "independence_groups": independence_groups,
        "independence_group_count": len(independence_groups),
        "aggregate_rows_expanded_to_samples": False,
    }


def build_evidence_union(
    *,
    union_uid: str,
    members: list[Mapping[str, object]],
    canonical_output_ref: str,
    conflict_policy: str = "fail_closed",
) -> dict[str, Any]:
    """Construct one union record and reduce it with conservative eligibility."""

    normalized_members = [_mapping(member) for member in members]
    validation_errors: list[dict[str, Any]] = []
    for index, member in enumerate(normalized_members):
        errors = _member_errors(member)
        if errors:
            validation_errors.append(
                {
                    "lineage_group": str(member.get("lineage_group") or "invalid_members"),
                    "state": "unknown",
                    "member_artifact_uids": [str(member.get("artifact_uid") or f"member_{index}")],
                    "reason": ", ".join(errors),
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in normalized_members:
        group = str(member.get("lineage_group") or "invalid_members")
        groups[group].append(member)
    conflicts = validation_errors + [_conflict_for_group(group, group_members) for group, group_members in sorted(groups.items())]
    rejected_conflicts = [row for row in conflicts if row["state"] in REJECTED_CONFLICT_STATES]

    blocked_member_uids = sorted(
        str(member.get("artifact_uid"))
        for member in normalized_members
        if str(member.get("status")) not in ACCEPTED_MEMBER_STATUSES
    )
    conservative_eligibility = _conservative_eligibility(normalized_members) if normalized_members else {
        "evidence_type": "nonreal_or_mixed",
        "claim_eligibility": "blocked",
        "provenance_status": "unknown",
        "aggregate_rows_expanded_to_samples": False,
    }
    resolution_status = "blocked" if rejected_conflicts or conflict_policy != "fail_closed" else "resolved"
    union_status = "blocked" if resolution_status == "blocked" or blocked_member_uids else "accepted"

    return {
        "schema_version": EVIDENCE_UNION_SCHEMA_VERSION,
        "union_uid": union_uid,
        "member_artifact_uids": [str(member.get("artifact_uid", "")) for member in normalized_members],
        "member_hashes": {str(member.get("artifact_uid", "")): str(member.get("artifact_hash", "")) for member in normalized_members},
        "producer_uids": sorted({str(member.get("producer_uid", "")) for member in normalized_members}),
        "schema_versions": sorted({str(member.get("schema_version", "")) for member in normalized_members}),
        "lineage_groups": sorted(groups),
        "conflict_policy": conflict_policy,
        "conflicts": conflicts,
        "resolution_status": resolution_status,
        "canonical_output_ref": canonical_output_ref,
        "member_evidence": normalized_members,
        "canonical_downstream_eligibility": conservative_eligibility,
        "blocked_member_artifact_uids": blocked_member_uids,
        "status": union_status,
        "valid": not rejected_conflicts and conflict_policy == "fail_closed" and bool(normalized_members),
    }


def fails_closed_for_conflicts(union: Mapping[str, object]) -> bool:
    conflicts = union.get("conflicts")
    if not isinstance(conflicts, list):
        return False
    rejected = any(_mapping(conflict).get("state") in REJECTED_CONFLICT_STATES for conflict in conflicts)
    return (not rejected or union.get("resolution_status") == "blocked") and union.get("conflict_policy") == "fail_closed"
