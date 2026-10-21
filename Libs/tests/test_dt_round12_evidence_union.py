from __future__ import annotations

from Libs.utils.data.evidence_union import RELATIONSHIP_ELIGIBILITY_ROLES, build_evidence_union


def _member(uid: str, digest: str, **overrides: object) -> dict[str, object]:
    member: dict[str, object] = {
        "artifact_uid": uid,
        "artifact_hash": digest,
        "producer_uid": f"producer_{uid}",
        "schema_version": "test-v1",
        "lineage_group": "shared_lineage",
        "provenance_status": "verified",
        "evidence_type": "real",
        "evidence_class": "experimental_endpoint",
        "qc_grade": "A",
        "train_eval_role": "evaluation_only",
        "claim_eligibility": "eligible",
        "independence_group": "shared_independence_group",
        "status": "accepted",
        "semantic_fingerprint": "shared_semantics",
        "pairing_decision": "valid_action_endpoint_pair",
        "relationship_eligibility": {role: True for role in RELATIONSHIP_ELIGIBILITY_ROLES},
        "typed_conflicts": [],
    }
    member.update(overrides)
    return member


def _union(*members: dict[str, object]) -> dict[str, object]:
    return build_evidence_union(
        union_uid="round12_test_union",
        members=list(members),
        canonical_output_ref="Output/audits/test_union.json",
    )


def test_relationship_eligibility_is_boolean_intersection() -> None:
    restricted = {role: True for role in RELATIONSHIP_ELIGIBILITY_ROLES}
    restricted["eligible_for_trajectory_training"] = False
    union = _union(
        _member("left", "1" * 64),
        _member("right", "2" * 64, relationship_eligibility=restricted),
    )
    eligibility = union["canonical_downstream_eligibility"]
    assert eligibility["eligible_for_trajectory_training"] is False
    assert eligibility["eligible_for_validation"] is True


def test_pairing_disagreement_cannot_upgrade_to_a_valid_pair() -> None:
    union = _union(
        _member("left", "1" * 64),
        _member("right", "2" * 64, pairing_decision="semantic_alignment_only"),
    )
    assert union["canonical_downstream_eligibility"]["pairing_decision"] == "unsupported_pairing"


def test_qc_and_evidence_class_use_conservative_non_upgrade_values() -> None:
    union = _union(
        _member("left", "1" * 64),
        _member("right", "2" * 64, qc_grade="C", evidence_class="simulator_output"),
    )
    eligibility = union["canonical_downstream_eligibility"]
    assert eligibility["qc_grade"] == "C"
    assert eligibility["evidence_class"] == "mixed_no_upgrade"


def test_typed_conflicts_are_preserved_exactly_once() -> None:
    conflict = {"conflict_uid": "conflict_one", "state": "unresolved", "conflict_type": "numeric_disagreement"}
    union = _union(
        _member("left", "1" * 64, typed_conflicts=[conflict]),
        _member("right", "2" * 64, typed_conflicts=[conflict]),
    )
    assert union["preserved_typed_conflicts"] == [conflict]
