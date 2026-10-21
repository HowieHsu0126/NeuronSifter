from __future__ import annotations

from Libs.utils.exp.round11g_closure import (
    R11F_SCOPE_PATHS,
    build_failure_family_summary,
    build_root_cause_graph,
    evaluate_round11g0_gate,
)


def test_round11g_scope_excludes_active_config_stack() -> None:
    assert "Libs/configs/config_stack.yaml" not in R11F_SCOPE_PATHS
    assert "Libs/configs/schema/schemas.yaml" not in R11F_SCOPE_PATHS


def test_round11g_graph_keeps_rc003_as_controller_contract_cause() -> None:
    rows = [{"root_cause_uid": "R11E-RC-003", "testcase_uid": "case-1", "repair_required": "no"}]
    graph = build_root_cause_graph(rows)
    assert graph["root_cause_graph"][0]["confidence"] == "confirmed"
    assert graph["root_cause_graph"][0]["affected_shared_fixture_or_module"] == "controller_union_roadmap_contract"


def test_round11g_summary_does_not_hide_current_nonpasses() -> None:
    rows = [{"root_cause_uid": "R11E-RC-001", "testcase_uid": "case-1", "repair_required": "yes"}]
    summary = build_failure_family_summary(rows)
    assert summary["families"][0]["current_nonpass_count"] == 1


def test_round11g0_gate_requires_every_identity_invariant() -> None:
    report = evaluate_round11g0_gate(
        storage={"storage_recovery": "passed"},
        lock_resolution={"status": "stale_lock_forensically_renamed"},
        git_health={"status": "git_index_healthy"},
        recovery={"status": "passed", "unrelated_user_files_excluded": True},
        commit_manifest={"parent_chain_valid": True, "root_index_unchanged": True},
        identity={"dirty_status": "clean", "reproduction_relation": "exact_clean_commit"},
        authority={"canonical_controller_count": 1, "canonical_gate_writer_count": 1},
        round11f_validation={
            "status": "passed",
            "negative_case_rejection_count": 11,
            "unexpected_negative_case_accept_count": 0,
        },
        handoff_validation={"status": "passed"},
        execution_index_lock_present=False,
    )
    assert report["status"] == "execution_identity_restored"


def test_round11g0_gate_fails_closed_for_dirty_execution_worktree() -> None:
    report = evaluate_round11g0_gate(
        storage={"storage_recovery": "passed"},
        lock_resolution={"status": "not_required"},
        git_health={"status": "git_index_healthy"},
        recovery={"status": "passed", "unrelated_user_files_excluded": True},
        commit_manifest={"parent_chain_valid": True, "root_index_unchanged": True},
        identity={"dirty_status": "dirty", "reproduction_relation": "exact_clean_commit"},
        authority={"canonical_controller_count": 1, "canonical_gate_writer_count": 1},
        round11f_validation={
            "status": "passed",
            "negative_case_rejection_count": 11,
            "unexpected_negative_case_accept_count": 0,
        },
        handoff_validation={"status": "passed"},
        execution_index_lock_present=False,
    )
    assert report["status"] == "blocked_full_suite"
    assert "clean_worktree_created" in report["blockers"]
