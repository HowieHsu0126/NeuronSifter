from __future__ import annotations

from Libs.utils.data.evidence_relationships import (
    LEGACY_ROUND12_GATE_SCHEMA_VERSION,
    ROUND12_GATE_SCHEMA_VERSION,
    gate_state_for_blockers,
    round12_gate_status,
)


def _blocker(code: str) -> dict[str, str]:
    return {"code": code, "message": "test blocker"}


def test_legacy_v1_gate_taxonomy_remains_readable() -> None:
    assert LEGACY_ROUND12_GATE_SCHEMA_VERSION == "neuronsifter-dt-round12-gate-v1"
    assert gate_state_for_blockers([_blocker("execution_identity_not_exact_clean")]) == "blocked_relation_provenance"


def test_v2_execution_verification_blockers_are_typed_without_provenance_failure() -> None:
    status = round12_gate_status(
        [
            _blocker("execution_identity_not_exact_clean"),
            _blocker("full_suite_not_verified"),
        ]
    )
    assert status == {
        "round12_gate_schema_version": ROUND12_GATE_SCHEMA_VERSION,
        "status": "blocked_execution_verification",
        "relation_contract_status": "passed_local",
        "execution_identity_status": "blocked",
        "full_suite_status": "not_verified",
        "overall_gate_status": "blocked",
        "blocker_classes": ["execution_identity", "full_suite_verification"],
    }


def test_v2_actual_relationship_provenance_failure_remains_explicit() -> None:
    status = round12_gate_status([_blocker("unknown_provenance_strong_relation")])
    assert status["status"] == "blocked_relation_provenance"
    assert status["relation_contract_status"] == "blocked"
    assert status["blocker_classes"] == ["relationship_provenance"]


def test_v2_resolved_gate_has_no_blocker_classes() -> None:
    status = round12_gate_status([])
    assert status["status"] == "typed_evidence_relationships_resolved"
    assert status["overall_gate_status"] == "accepted"
    assert status["relation_contract_status"] == "passed_local"
    assert status["blocker_classes"] == []
