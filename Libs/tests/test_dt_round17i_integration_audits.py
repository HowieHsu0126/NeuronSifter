from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from Libs.scripts.builders.build_dt_round17i_integration_audits import (
    build_gate,
    build_identity,
    build_inventory,
    classify_dirty_path,
    collect_identities,
    integration_class,
    integration_diff_rows,
    junit_identities,
    parse_porcelain_v1_z,
)


def test_round17i_porcelain_snapshot_parser_preserves_status_and_rename() -> None:
    data = b" M tracked.txt\0?? new.txt\0R  renamed.txt\0old.txt\0"
    rows = parse_porcelain_v1_z(data)
    assert rows == [
        {
            "path": "tracked.txt",
            "original_path": "",
            "git_status": " M",
            "index_status": " ",
            "worktree_status": "M",
        },
        {
            "path": "new.txt",
            "original_path": "",
            "git_status": "??",
            "index_status": "?",
            "worktree_status": "?",
        },
        {
            "path": "renamed.txt",
            "original_path": "old.txt",
            "git_status": "R ",
            "index_status": "R",
            "worktree_status": " ",
        },
    ]


def test_round17i_dirty_path_ownership_is_fail_closed_and_complete() -> None:
    assert classify_dirty_path("Docs/design/iclr/README.md") == "ICLR workflow"
    assert classify_dirty_path("Output/audits/controlled_access_checkpoint.json") == "unrelated user work"
    assert classify_dirty_path("Output/audits/paper_grade_full_suite.xml") == "Paper-Grade workflow"
    assert classify_dirty_path("Libs/scripts/builders/build_dt_round17_scitil_runtime.py") == "NeuronSifter accepted lineage"
    assert classify_dirty_path("TODO.md") == "shared infrastructure"
    assert classify_dirty_path(".env") == "unrelated user work"
    assert classify_dirty_path("unmatched/user/path.txt") == "unrelated user work"


def test_round17i_integration_manifest_rejects_runtime_and_parallel_paths() -> None:
    accepted = {"Libs/utils/data/scitil_contracts.py"}
    assert integration_class("Libs/configs/schema/schemas.yaml", accepted_paths=accepted) == "schema registry"
    assert integration_class("Libs/tests/test_dt_round17_scitil_runtime.py", accepted_paths=accepted) == "test"
    assert integration_class("Output/audits/full_results.xml", accepted_paths=accepted) == "generated runtime artifact"
    assert integration_class("Docs/design/iclr/README.md", accepted_paths=accepted) == "contract documentation"
    assert integration_class("unexpected.bin", accepted_paths=accepted) == "unknown"


def test_round17i_integration_diff_reads_clean_commit_or_staged_candidate(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Round17I Test")
    git("config", "user.email", "round17i@example.invalid")
    tracked = repository / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")

    tracked.write_text("committed candidate\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "candidate")
    committed = integration_diff_rows(repository, base, base)
    assert [(row["path"], row["candidate_source"]) for row in committed] == [
        ("tracked.txt", "committed_head")
    ]

    staged = repository / "staged.txt"
    staged.write_text("staged candidate\n", encoding="utf-8")
    git("add", "staged.txt")
    staged_rows = integration_diff_rows(repository, base, base)
    assert {row["candidate_source"] for row in staged_rows} == {"staged_index"}
    assert {row["path"] for row in staged_rows} == {"staged.txt", "tracked.txt"}


def _write_junit(path: Path, identities: list[tuple[str, str]]) -> None:
    cases = "".join(f'<testcase classname="{classname}" name="{name}" time="0" />' for classname, name in identities)
    path.write_text(
        f'<testsuites><testsuite tests="{len(identities)}" failures="0" errors="0" skipped="0">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )


def test_round17i_dual_parent_inventory_uses_classname_and_stable_parameter_identity(tmp_path: Path) -> None:
    main_collect = tmp_path / "main.txt"
    integrated_collect = tmp_path / "integrated.txt"
    round17_junit = tmp_path / "round17.xml"
    main_collect.write_text("Libs/tests/test_a.py::test_common\n", encoding="utf-8")
    integrated_collect.write_text(
        "Libs/tests/test_a.py::test_common\nLibs/tests/test_b.py::TestB::test_added[param-1]\n",
        encoding="utf-8",
    )
    _write_junit(
        round17_junit,
        [
            ("Libs.tests.test_a", "test_common"),
            ("Libs.tests.test_b.TestB", "test_added[param-1]"),
        ],
    )
    assert collect_identities(integrated_collect) == {
        "Libs.tests.test_a::test_common",
        "Libs.tests.test_b.TestB::test_added[param-1]",
    }
    identities, counts = junit_identities(round17_junit)
    assert len(identities) == counts["tests"] == 2

    result = build_inventory(
        main_collect=main_collect,
        round17_junit=round17_junit,
        integrated_collect=integrated_collect,
        output_root=tmp_path,
    )
    assert result["status"] == "complete"
    assert result["unexplained_missing"] == 0
    assert result["mandatory_disappearance"] == 0
    written = json.loads((tmp_path / "Output/audits/dt_round17i_test_inventory_diff.json").read_text(encoding="utf-8"))
    assert written["integrated_tests"] == 2


def test_round17i_identity_accepts_scoped_commits_after_dual_parent_merge(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Round17I Test")
    git("config", "user.email", "round17i@example.invalid")
    for path in (
        "Libs/configs/config_stack.yaml",
        "Libs/configs/schema/schemas.yaml",
        "Libs/utils/exp/orchestration_controller.py",
        "Libs/configs/validation/round11f_controller_authority.yaml",
    ):
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{path}\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "main parent")
    main_parent = git("rev-parse", "HEAD")

    git("switch", "-c", "accepted")
    accepted_path = repository / "accepted.txt"
    accepted_path.write_text("accepted\n", encoding="utf-8")
    git("add", "accepted.txt")
    git("commit", "-m", "accepted parent")
    accepted_parent = git("rev-parse", "HEAD")

    git("switch", "-c", "integration", main_parent)
    git("merge", "--no-ff", accepted_parent, "-m", "dual parent merge")
    repair = repository / "repair.txt"
    repair.write_text("repair\n", encoding="utf-8")
    git("add", "repair.txt")
    git("commit", "-m", "scoped repair")

    evidence_root = tmp_path / "evidence"
    manifest = evidence_root / "Output/evidence_fabric/v0/evidence_fabric_release_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}\n", encoding="utf-8")
    result = build_identity(
        integration_worktree=repository,
        evidence_root=evidence_root,
        test_inventory=inventory,
        output_root=tmp_path,
        main_parent=main_parent,
        accepted_parent=accepted_parent,
    )

    merge_commit = git("rev-parse", "HEAD^")
    assert result["commit_parents"] == [merge_commit]
    assert result["main_parent_ancestor_verified"] is True
    assert result["accepted_parent_ancestor_verified"] is True
    assert result["dual_parent_merge_commit"] == merge_commit
    assert result["dual_parent_identity_valid"] is True

    (repository / "untracked.txt").write_text("must block exact identity\n", encoding="utf-8")
    dirty_result = build_identity(
        integration_worktree=repository,
        evidence_root=evidence_root,
        test_inventory=inventory,
        output_root=tmp_path,
        main_parent=main_parent,
        accepted_parent=accepted_parent,
    )
    assert dirty_result["dirty_status"] == "dirty"
    assert dirty_result["reproduction_relation"] == "invalid_dirty_candidate"


def test_round17i_gate_requires_full_inventory_and_public_only_diff(tmp_path: Path) -> None:
    evidence_root = tmp_path / "repository"
    evidence_root.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=evidence_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Round17I Test")
    git("config", "user.email", "round17i@example.invalid")
    tracked = evidence_root / "tracked.txt"
    tracked.write_text("main\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "integrated main")
    integration_commit = git("rev-parse", "HEAD")

    evidence_audits = evidence_root / "Output/audits"
    evidence_audits.mkdir(parents=True)
    round17_gate = {
        "status": "scitil_twin_gated_agent_loop_resolved",
        "evidence_fabric": {"sealed": False},
        "frozen_gap_count": 20,
        "dt_core": {"status": "not_evaluated"},
    }
    round17_bytes = (json.dumps(round17_gate, sort_keys=True) + "\n").encode()
    (evidence_audits / "dt_round17_gate.json").write_bytes(round17_bytes)
    fabric_manifest = evidence_root / "Output/evidence_fabric/v0/evidence_fabric_release_manifest.json"
    fabric_manifest.parent.mkdir(parents=True)
    fabric_manifest.write_text('{"sealed": false}\n', encoding="utf-8")

    output_root = tmp_path / "output"
    audit_root = output_root / "Output/audits"
    audit_root.mkdir(parents=True)

    def write(name: str, value: object) -> None:
        (audit_root / name).write_text(json.dumps(value) + "\n", encoding="utf-8")

    write("dt_round17i_repository_snapshot.json", {"dirty_root": {"unknown_ownership_count": 0}})
    write(
        "dt_round17i_accepted_lineage_map.json",
        {
            "status": "coherent_accepted_lineage",
            "rounds": [
                {
                    "round": 17,
                    "gate_artifact_hash": hashlib.sha256(round17_bytes).hexdigest(),
                }
            ],
        },
    )
    write("dt_round17i_cross_workflow_collision.json", {"blocking_collision_count": 0})
    write(
        "dt_round17i_test_inventory_diff.json",
        {
            "status": "complete",
            "integrated_tests": 2,
            "unexplained_missing": 0,
            "mandatory_disappearance": 0,
        },
    )
    write("dt_round17i_prefull_validation.json", {"status": "passed"})
    _write_junit(
        audit_root / "dt_round17i_prefull_results.xml",
        [
            (
                "Libs.tests.test_public_mainline_isolation",
                "test_public_mainline_resolves_zero_controlled_sources_in_every_route_class",
            )
        ],
    )
    identity = {
        "integration_commit": integration_commit,
        "reproduction_relation": "exact_clean_integration_commit",
        "dirty_status": "clean",
        "dual_parent_identity_valid": True,
        "config_hash": "hash",
        "schema_registry_hash": "hash",
        "controller_hash": "hash",
        "gate_authority_hash": "hash",
        "evidence_fabric_manifest_hash": "hash",
        "test_manifest_hash": "hash",
        "environment_hash": "hash",
    }
    write("dt_round17i_execution_identity.json", identity)
    write(
        "dt_round17i_full_suite_execution.json",
        {"status": "passed", "unique_test_identities": 2, "passed": 2, "core_xfail": 0},
    )
    write(
        "dt_round17i_main_promotion.json",
        {
            "post_promotion_status": "promoted_and_verified",
            "new_main_commit": integration_commit,
            "integration_commit": integration_commit,
        },
    )
    (audit_root / "dt_round17i_integration_diff_manifest.tsv").write_text(
        "path\tauthorized\nLibs/public.py\ttrue\n",
        encoding="utf-8",
    )

    validation, gate = build_gate(output_root=output_root, evidence_root=evidence_root)
    assert validation["status"] == "passed"
    assert gate["status"] == "scitil_runtime_mainline_integrated"

    write(
        "dt_round17i_full_suite_execution.json",
        {"status": "passed", "unique_test_identities": 1, "passed": 1, "core_xfail": 0},
    )
    validation, gate = build_gate(output_root=output_root, evidence_root=evidence_root)
    assert validation["status"] == "blocked_execution_verification"
    assert validation["checks"]["full_suite_passed"] is False
    assert gate["overall_gate_status"] == "blocked"
