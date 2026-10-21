from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from Libs.utils.evaluation import operational_twin_candidate_retry_authority as retry_authority
from Libs.utils.evaluation import operational_twin_successor_lineage_protocol as successor
from Libs.utils.evaluation.operational_twin_candidate_retry_authority import (
    POSITIVE,
    SUCCESSOR_REQUIRED,
    adjudicate_retry_authority,
    derive_positive_successor,
    materialized_positive_audit,
    validate_positive_retry_audit,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    OperationalTwinProtocolError,
    read_repo_bytes,
)
from Libs.utils.exp_utils import load_config_stack


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return load_config_stack(ROOT / "Libs/configs/config_stack.yaml")


def _synthetic_positive_audit(config: dict[str, object]) -> dict[str, object]:
    audit = deepcopy(adjudicate_retry_authority(config, ROOT))
    contract = config["operational_twin_candidate_retry_authority"]
    for index, row in enumerate(audit["rows"]):
        row["predecessor_evidence_kind"] = "materialized_evaluation_failure"
        row["predecessor_failure_class"] = "infrastructure_interruption"
        row["historical_recovery_classification"] = "ELIGIBLE"
        row["predecessor_identity"] = {
            "terminal_status": f"{index + 1:040x}",
            "phase_receipt": f"{index + 101:040x}",
        }
        row["predicates"] = {key: True for key in row["predicates"]}
        row["result"] = "EXISTING_LINEAGE_SLOT_AUTHORIZED"
    audit.update(
        {
            "terminal_decision": POSITIVE,
            "next_direction": contract["next_directions"]["positive"],
            "authorized_row_count": 15,
            "successor_required_row_count": 0,
            "positive_artifacts_materialized": False,
        }
    )
    audit["decision_uid"] = retry_authority._decision_uid(audit)
    return audit


def test_real_evidence_requires_successor_lineage() -> None:
    audit = adjudicate_retry_authority(_config(), ROOT)
    assert audit["terminal_decision"] == SUCCESSOR_REQUIRED
    assert audit["next_direction"] == "freeze-operational-twin-successor-lineage-protocol"
    assert audit["assignment_count"] == 15
    assert audit["authorized_row_count"] == 0
    assert audit["successor_required_row_count"] == 15
    assert sum(row["predecessor_evidence_kind"] == "materialized_evaluation_failure" for row in audit["rows"]) == 10
    assert sum(row["predecessor_evidence_kind"] == "prepublication_scheduler_failure" for row in audit["rows"]) == 5
    assert all(row["result"] == "SUCCESSOR_LINEAGE_REQUIRED" for row in audit["rows"])
    assert audit["candidate_dispatch_authorized"] is False
    assert audit["side_effect_count"] == 0
    assert set(audit["access_counters"].values()) == {0}
    assert audit["positive_artifacts_materialized"] is False
    assert (
        audit["decision_uid"]
        == "operational_twin_candidate_retry_authority_05015fd1c0941e7e931cb47d"
    )


def test_successor_identity_derivation_preserves_round_48_negative_authority() -> None:
    config = _config()
    contract = config["operational_twin_candidate_retry_authority"]
    paths = contract["output_paths"]
    before = {
        key: read_repo_bytes(ROOT, paths[key])
        for key in ("decision_audit", "decision_report")
    }
    successor.compute_successor_identity_declarations(config, ROOT)
    after = {
        key: read_repo_bytes(ROOT, paths[key])
        for key in ("decision_audit", "decision_report")
    }
    assert before == after
    audit = json.loads(before["decision_audit"].decode("utf-8"))
    assert audit["terminal_decision"] == SUCCESSOR_REQUIRED
    assert audit["decision_uid"] == (
        "operational_twin_candidate_retry_authority_05015fd1c0941e7e931cb47d"
    )
    assert audit["candidate_dispatch_authorized"] is False
    assert audit["scientific_twin_result"] == "NOT_RUN"
    assert set(audit["access_counters"].values()) == {0}


def test_authentication_drift_publishes_no_decision() -> None:
    config = _config()
    config["operational_twin_candidate_retry_authority"]["round_47_authority"]["audit_git_object_id"] = "0" * 40
    with pytest.raises(OperationalTwinProtocolError, match="git_identity_mismatch"):
        adjudicate_retry_authority(config, ROOT)


def test_terminal_only_flip_cannot_derive_positive_successor() -> None:
    config = _config()
    audit = adjudicate_retry_authority(config, ROOT)
    positive = deepcopy(audit)
    positive["terminal_decision"] = POSITIVE
    with pytest.raises(
        OperationalTwinProtocolError,
        match="positive_audit_not_recomputed",
    ):
        derive_positive_successor(config, ROOT, positive)


def test_positive_audit_requires_complete_row_and_global_ledger() -> None:
    config = _config()
    audit = _synthetic_positive_audit(config)
    assert (
        validate_positive_retry_audit(
            config,
            audit,
            expected_materialized=False,
        )
        == []
    )
    materialized = materialized_positive_audit(audit)
    assert materialized["decision_uid"] == audit["decision_uid"]
    assert (
        validate_positive_retry_audit(
            config,
            materialized,
            expected_materialized=True,
        )
        == []
    )

    mutations = []
    row_predicate = deepcopy(audit)
    row_predicate["rows"][0]["predicates"]["retry_attempt_kind"] = False
    mutations.append(row_predicate)
    global_predicate = deepcopy(audit)
    global_predicate["global_predicates"]["retry_slot_count"] = False
    mutations.append(global_predicate)
    wrong_count = deepcopy(audit)
    wrong_count["authorized_row_count"] = 14
    mutations.append(wrong_count)
    wrong_date = deepcopy(audit)
    wrong_date["authoritative_date"] = "2026-09-13"
    mutations.append(wrong_date)
    wrong_allowlist = deepcopy(audit)
    wrong_allowlist["rows"][0]["retry_command_index"] = 14
    mutations.append(wrong_allowlist)
    for mutation in mutations:
        mutation["decision_uid"] = retry_authority._decision_uid(mutation)
        assert validate_positive_retry_audit(
            config,
            mutation,
            expected_materialized=False,
        )


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("output_root", "../escape"),
        (
            "execution_authority",
            "Output/results/operational_twin/not-an-authority.json",
        ),
        (
            "resource_environment_receipt",
            "Output/audits/operational_twin//candidate_resource_environment_receipts/alias.json",
        ),
    ],
)
def test_retry_target_validation_uses_each_declared_path(
    tmp_path: Path,
    field: str,
    mutated: str,
) -> None:
    config = _config()
    updated, protocol = retry_authority._positive_protocol_config(config)
    candidate = retry_authority.build_candidate_lineage_manifest(updated)
    assignments = retry_authority.build_candidate_assignment_manifest(
        updated,
        candidate,
    )
    commands = retry_authority.build_candidate_commands(updated, candidate, assignments)
    retry = assignments["assignments"][0]["attempts"][1]
    retry[field] = mutated
    blockers = retry_authority._validate_retry_targets(
        config,
        tmp_path,
        protocol,
        assignments,
        commands,
    )
    assert "retry_authority_positive_retry_target_invalid" in {
        row["code"] for row in blockers
    }


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_retry_target_collision_or_symlink_is_rejected(
    tmp_path: Path,
    kind: str,
) -> None:
    config = _config()
    updated, protocol = retry_authority._positive_protocol_config(config)
    candidate = retry_authority.build_candidate_lineage_manifest(updated)
    assignments = retry_authority.build_candidate_assignment_manifest(
        updated,
        candidate,
    )
    commands = retry_authority.build_candidate_commands(updated, candidate, assignments)
    target = tmp_path / assignments["assignments"][0]["attempts"][1]["execution_authority"]
    target.parent.mkdir(parents=True)
    if kind == "file":
        target.write_text("occupied", encoding="utf-8")
    else:
        os.symlink(tmp_path / "missing", target)
    blockers = retry_authority._validate_retry_targets(
        config,
        tmp_path,
        protocol,
        assignments,
        commands,
    )
    assert "retry_authority_positive_retry_target_invalid" in {
        row["code"] for row in blockers
    }
