"""Fail-closed Round-11G execution identity and closure audit helpers.

The module consumes frozen Round-11D/11E evidence and the Round-11F authority
contract.  It does not run tests, mutate scientific state, or start Round-12.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from Libs.utils.data.raw_data_registry import load_config_with_provenance, write_json, write_tsv
from Libs.utils.exp.orchestration_controller import (
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    sha256_file,
    sha256_payload,
)


AUDIT_ROOT = Path("Output/audits")
AUTHORITY_PATH = Path("Libs/configs/validation/round11f_controller_authority.yaml")
R11F_SCOPE_PATHS = (
    "Libs/utils/exp/orchestration_controller.py",
    "Libs/utils/exp/round11f_controller_lineage.py",
    "Libs/utils/data/evidence_union.py",
    "Libs/configs/validation/round11f_controller_authority.yaml",
    "Libs/configs/schema/gate_writer_authority.schema.json",
    "Libs/configs/schema/evidence_union.schema.json",
    "Libs/scripts/builders/build_dt_round11f_controller_contracts.py",
    "Libs/scripts/validators/validate_dt_round11f_controller_contracts.py",
    "Libs/tests/test_dt_round11f_controller_contracts.py",
)
R11F_AUDIT_PATHS = (
    "Output/audits/dt_round11f_controller_authority_map.json",
    "Output/audits/dt_round11f_controller_inventory.json",
    "Output/audits/dt_round11f_controller_migration_map.tsv",
    "Output/audits/dt_round11f_dependency_gate.json",
    "Output/audits/dt_round11f_future_stage_ownership.json",
    "Output/audits/dt_round11f_gate.json",
    "Output/audits/dt_round11f_gate_authority_validation.json",
    "Output/audits/dt_round11f_gate_writer_inventory.tsv",
    "Output/audits/dt_round11f_lineage_inventory.json",
    "Output/audits/dt_round11f_roadmap_lineage.json",
    "Output/audits/dt_round11f_union_conflict_cases.json",
    "Output/audits/dt_round11f_union_contract.json",
    "Output/audits/dt_round11f_validation.json",
    "Output/audits/dt_round11g_handoff.json",
)
R11F_FROZEN_EVIDENCE_PATHS = (
    "Output/audits/dt_round11d_full_suite_execution.json",
    "Output/audits/dt_round11d_full_suite_results.xml",
    "Output/audits/dt_round11d_full_suite_failure_triage.json",
    "Output/audits/dt_round11_closure_decision.json",
    "Output/audits/dt_round11e_targeted_gate.json",
    "Output/audits/dt_round11e_repair_wave_record.json",
    "Output/audits/dt_round11e_testcase_failure_ledger.tsv",
    "Output/audits/dt_round11e_failure_dependency_graph.json",
    "Output/audits/dt_round11e_full_suite_execution.json",
)
R11G_SCOPE_PATHS = (
    "Libs/utils/exp/round11g_closure.py",
    "Libs/scripts/builders/build_dt_round11g_identity_audits.py",
    "Libs/scripts/validators/validate_dt_round11g_handoff.py",
    "Libs/scripts/validators/validate_dt_round11g0_gate.py",
    "Libs/tests/test_dt_round11g_closure.py",
)
ROUND11FG_COMMIT_GROUPS = {
    "A": (
        "Libs/utils/exp/orchestration_controller.py",
        "Libs/utils/data/evidence_union.py",
        "Libs/configs/validation/round11f_controller_authority.yaml",
        "Libs/configs/schema/gate_writer_authority.schema.json",
        "Libs/configs/schema/evidence_union.schema.json",
    ),
    "B": (
        "Libs/utils/exp/round11f_controller_lineage.py",
        "Libs/scripts/builders/build_dt_round11f_controller_contracts.py",
        "Libs/scripts/validators/validate_dt_round11f_controller_contracts.py",
        "Libs/tests/test_dt_round11f_controller_contracts.py",
        *R11F_AUDIT_PATHS,
    ),
    "C": R11G_SCOPE_PATHS,
}
ROUND11FG_EXCLUDED_INPUT_PATHS = (
    "TODO.md",
    "orchestrator/state.json",
    "Libs/configs/config_stack.yaml",
    "Libs/configs/schema/schemas.yaml",
    "Libs/configs/validation/iclr_design_contract_union.yaml",
    "Libs/utils/data/iclr_contract_union.py",
    "Libs/scripts/validators/validate_iclr_design_contract_union.py",
    "Libs/scripts/validators/validate_orchestrator_state_contract.py",
    "orchestrator/project-contract.md",
    "orchestrator/roadmap-updates/round-11-dt-partial-state-roadmap-update.md",
    "orchestrator/roadmap-updates/round-11-dt-partial-state-roadmap-update-review.md",
)
SCOPED_MANIFEST_FIELDS = [
    "path", "exists", "git_status", "size", "sha256", "contract_owner",
    "commit_group", "include", "exclusion_reason",
]
HISTORICAL_JUNIT = Path("Output/audits/dt_round11d_full_suite_results.xml")
HISTORICAL_LEDGER = Path("Output/audits/dt_round11e_testcase_failure_ledger.tsv")
R11F_VALIDATION = Path("Output/audits/dt_round11f_validation.json")
R11F_AUTHORITY_MAP = Path("Output/audits/dt_round11f_controller_authority_map.json")

LEDGER_FIELDS = [
    "testcase_uid", "test_id", "historical_status", "current_status", "historical_stacktrace_hash",
    "current_stacktrace_hash", "reproduced", "resolved_by_prior_change", "new_failure_signature",
    "root_cause_uid", "repair_required",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(payload)


def _binding(repo_root: Path, relative_path: str | Path, *, required: bool = True) -> dict[str, Any]:
    relative = Path(relative_path)
    path = repo_root / relative
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required Round-11G input is missing: {relative}")
        return {"path": relative.as_posix(), "present": False, "sha256": None, "bytes": None}
    return {"path": relative.as_posix(), "present": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _run_git(repo_root: Path, args: list[str], timeout_seconds: int = 20) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=repo_root, check=False, text=True, capture_output=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired:
        return "timeout", ""
    if completed.returncode != 0:
        return "error", completed.stderr.strip()
    return "ok", completed.stdout


def _tracked_change_paths(repo_root: Path) -> dict[str, Any]:
    # A status scan would contend with another writer and can hang indefinitely
    # on this repository's network filesystem.  Record the limitation instead.
    if (repo_root / ".git/index.lock").exists():
        return {
            "capture_status": "blocked_by_active_git_index_lock",
            "paths": [],
            "detail": "Root status scan was not run while an existing Git index lock was present.",
        }
    status, stdout = _run_git(repo_root, ["status", "--porcelain=v1", "--untracked-files=normal"])
    if status != "ok":
        return {"capture_status": status, "paths": [], "detail": stdout or "Git status scan did not complete."}
    return {
        "capture_status": "complete",
        "paths": [line[3:] for line in stdout.splitlines() if len(line) >= 4],
        "detail": None,
    }


def _head_path_status(repo_root: Path, relative_path: str) -> str:
    status, _ = _run_git(repo_root, ["cat-file", "-e", f"HEAD:{relative_path}"], timeout_seconds=5)
    return "tracked_in_head" if status == "ok" else "untracked_or_ignored"


def build_round11f_identity_inventory(repo_root: Path) -> dict[str, Any]:
    """Capture the dirty root scope without treating it as an execution identity."""

    repo_root = repo_root.resolve()
    changes = _tracked_change_paths(repo_root)
    scope = set(R11F_SCOPE_PATHS) | set(R11F_AUDIT_PATHS)
    changed = set(changes["paths"])
    scope_path_status = {path: _head_path_status(repo_root, path) for path in sorted(scope) if (repo_root / path).is_file()}
    scope_changed = sorted(
        {path for path in changed if path in scope}
        | {path for path, status in scope_path_status.items() if status == "untracked_or_ignored"}
    )
    unrelated = sorted(path for path in changed if path not in scope)
    implementation = {
        "controller": _binding(repo_root, "Libs/utils/exp/orchestration_controller.py"),
        "evidence_union": _binding(repo_root, "Libs/utils/data/evidence_union.py"),
        "gate_writer_authority_schema": _binding(repo_root, "Libs/configs/schema/gate_writer_authority.schema.json"),
        "evidence_union_schema": _binding(repo_root, "Libs/configs/schema/evidence_union.schema.json"),
        "validator": _binding(repo_root, "Libs/scripts/validators/validate_dt_round11f_controller_contracts.py"),
        "tests": _binding(repo_root, "Libs/tests/test_dt_round11f_controller_contracts.py"),
    }
    return {
        "schema_version": "dt-round11g-round11f-identity-inventory-v1",
        "captured_at": _utc_now(),
        "root_git_commit": git_commit(repo_root),
        "root_dirty_status": "dirty" if scope_changed or (repo_root / ".git/index.lock").exists() else git_dirty_status(repo_root),
        "root_status_capture": changes,
        "round11f_changed_paths": scope_changed,
        "round11f_path_status": scope_path_status,
        "unrelated_user_modified_paths": unrelated,
        "implementation_hashes": implementation,
        "not_an_execution_identity": True,
        "reason": "The root worktree is intentionally not used for the Round-11G full-suite execution.",
    }


def build_round11f_scoped_patch_manifest(repo_root: Path, inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "dt-round11g-round11f-scoped-patch-manifest-v1",
        "captured_at": _utc_now(),
        "base_commit": inventory["root_git_commit"],
        "included_implementation_paths": list(R11F_SCOPE_PATHS),
        "included_audit_paths": list(R11F_AUDIT_PATHS),
        "excluded_unrelated_paths": list(inventory.get("unrelated_user_modified_paths", [])),
        "forbidden_actions": ["git reset", "root_worktree_overwrite", "user_config_mutation", "unrelated_commit"],
        "commit_plan": [
            {"commit": "A", "scope": "canonical controller, Gate authority, Evidence Union, schemas"},
            {"commit": "B", "scope": "Round-11F validators, tests, builders, frozen evidence, lineage, and handoff"},
            {"commit": "C", "scope": "Round-11G execution identity and closure preparation"},
        ],
    }


def _git_path_status(repo_root: Path, relative_path: str) -> str:
    status, output = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "--ignored=matching", "--", relative_path],
    )
    if status != "ok":
        return f"status_{status}"
    line = output.splitlines()[0] if output.splitlines() else ""
    if line:
        return {"??": "untracked", "!!": "untracked_ignored"}.get(line[:2], line[:2])
    tracked, _ = _run_git(repo_root, ["ls-files", "--error-unmatch", "--", relative_path], timeout_seconds=5)
    return "clean_tracked" if tracked == "ok" else "absent_from_index"


def build_round11g0_scoped_file_manifest(repo_root: Path) -> list[dict[str, Any]]:
    """Build the authoritative Round-11F/G include/exclude manifest."""

    repo_root = repo_root.resolve()
    group_by_path = {
        path: group
        for group, paths in ROUND11FG_COMMIT_GROUPS.items()
        for path in paths
    }
    rows: list[dict[str, Any]] = []
    for relative_path in sorted(group_by_path):
        path = repo_root / relative_path
        exists = path.is_file()
        rows.append({
            "path": relative_path,
            "exists": str(exists).lower(),
            "git_status": _git_path_status(repo_root, relative_path) if exists else "missing",
            "size": path.stat().st_size if exists else "",
            "sha256": sha256_file(path) if exists else "",
            "contract_owner": "Round-11F" if group_by_path[relative_path] in {"A", "B"} else "Round-11G",
            "commit_group": group_by_path[relative_path],
            "include": str(exists).lower(),
            "exclusion_reason": "" if exists else "required_authorized_path_missing",
        })
    for relative_path in R11F_FROZEN_EVIDENCE_PATHS:
        path = repo_root / relative_path
        exists = path.is_file()
        rows.append({
            "path": relative_path,
            "exists": str(exists).lower(),
            "git_status": _git_path_status(repo_root, relative_path) if exists else "missing",
            "size": path.stat().st_size if exists else "",
            "sha256": sha256_file(path) if exists else "",
            "contract_owner": "Round-11F frozen evidence",
            "commit_group": "evidence",
            "include": "false",
            "exclusion_reason": "preserved as an external read-only evidence input; not added to the scoped commit tree",
        })
    for relative_path in ROUND11FG_EXCLUDED_INPUT_PATHS:
        path = repo_root / relative_path
        exists = path.is_file()
        rows.append({
            "path": relative_path,
            "exists": str(exists).lower(),
            "git_status": _git_path_status(repo_root, relative_path) if exists else "missing",
            "size": path.stat().st_size if exists else "",
            "sha256": sha256_file(path) if exists else "",
            "contract_owner": "preexisting_read_only_input",
            "commit_group": "excluded",
            "include": "false",
            "exclusion_reason": "preexisting or mixed-owner input; current root content is not authorized for the Round-11F/G scoped commits",
        })
    return sorted(rows, key=lambda row: (str(row["commit_group"]), str(row["path"])))


def preserve_round11g0_scoped_changes(
    repo_root: Path,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve every included path before the temporary-index commit flow."""

    repo_root = repo_root.resolve()
    audit_root = repo_root / AUDIT_ROOT
    included = [str(row["path"]) for row in rows if row.get("include") == "true"]
    frozen_evidence = [
        str(row["path"])
        for row in rows
        if row.get("contract_owner") == "Round-11F frozen evidence"
    ]
    preserved = included + frozen_evidence
    missing = [
        str(row["path"])
        for row in rows
        if str(row["path"]) in preserved and row.get("exists") != "true"
    ]
    tracked: list[str] = []
    untracked: list[str] = []
    for relative_path in preserved:
        status, _ = _run_git(repo_root, ["ls-files", "--error-unmatch", "--", relative_path], timeout_seconds=5)
        (tracked if status == "ok" else untracked).append(relative_path)

    if tracked:
        patch_result = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *tracked],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        patch_bytes = patch_result.stdout
        patch_returncode = patch_result.returncode
    else:
        patch_bytes = b""
        patch_returncode = 0
    patch_path = audit_root / "dt_round11g0_scoped_changes.patch"
    patch_path.write_bytes(patch_bytes)

    object_root = audit_root / "dt_round11g0_recovery_bundle" / "objects"
    object_root.mkdir(parents=True, exist_ok=True)
    archive_rows: list[dict[str, Any]] = []
    for relative_path in untracked:
        source = repo_root / relative_path
        digest = sha256_file(source)
        object_path = object_root / digest
        if not object_path.exists():
            shutil.copy2(source, object_path)
        archive_rows.append({
            "path": relative_path,
            "sha256": digest,
            "size": source.stat().st_size,
            "recovery_object": object_path.relative_to(repo_root).as_posix(),
        })

    write_tsv(
        audit_root / "dt_round11g0_scoped_untracked_archive_manifest.tsv",
        ["path", "sha256", "size", "recovery_object"],
        archive_rows,
    )
    excluded = [
        str(row["path"])
        for row in rows
        if row.get("include") != "true" and row.get("contract_owner") != "Round-11F frozen evidence"
    ]
    complete = not missing and patch_returncode == 0 and len(archive_rows) == len(untracked)
    report = {
        "schema_version": "dt-round11g0-recovery-bundle-manifest-v1",
        "artifact_uid": f"dt-round11g0-recovery-{_utc_now()}",
        "status": "passed" if complete else "blocked_incomplete_recovery_bundle",
        "base_commit": git_commit(repo_root),
        "authorized_path_count": len(included),
        "authorized_external_evidence_path_count": len(frozen_evidence),
        "authorized_tracked_path_count": len(tracked),
        "authorized_untracked_path_count": len(untracked),
        "authorized_modified_files_hashed": not missing,
        "authorized_untracked_files_preserved": len(archive_rows) == len(untracked),
        "unrelated_user_files_excluded": bool(excluded),
        "excluded_paths": excluded,
        "commit_excluded_external_evidence_paths": frozen_evidence,
        "scoped_patch": {
            "path": patch_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(patch_path),
            "bytes": patch_path.stat().st_size,
            "git_diff_exit_code": patch_returncode,
        },
        "recovery_object_count": len({row["sha256"] for row in archive_rows}),
        "missing_authorized_paths": missing,
    }
    write_json(audit_root / "dt_round11g0_recovery_bundle_manifest.json", report)
    return report


def execution_identity(repo_root: Path) -> dict[str, Any]:
    """Return strict identity fields for a clean execution worktree."""

    repo_root = repo_root.resolve()
    config, provenance = load_config_with_provenance(repo_root / "Libs/configs/config_stack.yaml")
    package_status, packages = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"])
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_probe": package_status,
        "git_probe_value": packages.strip(),
    }
    dirty_status = "dirty" if (repo_root / ".git/index.lock").exists() else git_dirty_status(repo_root)
    parent_status, parent_output = _run_git(repo_root, ["rev-parse", "HEAD^"])
    parent_commit = parent_output.strip() if parent_status == "ok" else None
    reproduction_relation = "exact_clean_commit" if dirty_status == "clean" else "blocked_execution_identity"
    return {
        "exact_git_commit": git_commit(repo_root),
        "parent_commit": parent_commit,
        "dirty_status": dirty_status,
        "config_hash": sha256_file(repo_root / "Libs/configs/config_stack.yaml"),
        "resolved_config_hash": sha256_payload(config),
        "schema_registry_hash": sha256_file(repo_root / "Libs/configs/schema/schemas.yaml"),
        "controller_hash": sha256_file(repo_root / "Libs/utils/exp/orchestration_controller.py"),
        "gate_writer_authority_hash": sha256_file(repo_root / AUTHORITY_PATH),
        "gate_authority_hash": sha256_file(repo_root / AUTHORITY_PATH),
        "evidence_union_hash": sha256_file(repo_root / "Libs/utils/data/evidence_union.py"),
        "config_include_order": provenance.get("include_order", []),
        "environment": environment,
        "environment_hash": sha256_payload(environment),
        "worktree_path_role": "round11g_clean_execution_worktree",
        "reproduction_relation": reproduction_relation,
        "identity_status": reproduction_relation,
    }


def evaluate_round11g0_gate(
    *,
    storage: Mapping[str, Any],
    lock_resolution: Mapping[str, Any],
    git_health: Mapping[str, Any],
    recovery: Mapping[str, Any],
    commit_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    authority: Mapping[str, Any],
    round11f_validation: Mapping[str, Any],
    handoff_validation: Mapping[str, Any],
    execution_index_lock_present: bool,
) -> dict[str, Any]:
    """Reduce Round-11G0 evidence without granting any later-stage authority."""

    checks = {
        "storage_recovery_passed": storage.get("storage_recovery") == "passed",
        "index_lock_resolved": lock_resolution.get("status") in {
            "not_required", "stale_lock_forensically_renamed",
        },
        "git_index_healthy": git_health.get("status") == "git_index_healthy",
        "scoped_recovery_bundle_complete": recovery.get("status") == "passed",
        "unrelated_user_changes_excluded": recovery.get("unrelated_user_files_excluded") is True,
        "scoped_commit_chain_valid": commit_manifest.get("parent_chain_valid") is True,
        "root_index_unchanged": commit_manifest.get("root_index_unchanged") is True,
        "clean_worktree_created": identity.get("dirty_status") == "clean",
        "execution_identity_exact": identity.get("reproduction_relation") == "exact_clean_commit",
        "execution_index_lock_absent": not execution_index_lock_present,
        "canonical_controller_count_one": authority.get("canonical_controller_count") == 1,
        "canonical_gate_writer_count_one": authority.get("canonical_gate_writer_count") == 1,
        "round11f_validator_passed": round11f_validation.get("status") == "passed",
        "round11f_negative_cases_fail_closed": (
            round11f_validation.get("negative_case_rejection_count") == 11
            and round11f_validation.get("unexpected_negative_case_accept_count") == 0
        ),
        "round11g_handoff_validator_passed": handoff_validation.get("status") == "passed",
    }
    passed = all(checks.values())
    return {
        "schema_version": "dt-round11g0-gate-v1",
        "validator_uid": "round11g0_execution_identity_validator",
        "status": "execution_identity_restored" if passed else "blocked_full_suite",
        "round_11g0_gate": "execution_identity_restored" if passed else "blocked_full_suite",
        "checks": checks,
        "blockers": [] if passed else [name for name, value in checks.items() if not value],
        "round_11": {"closure": "blocked_full_suite"},
        "round_12_started": False,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def validate_round11g0_gate(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_root = evidence_root.resolve()
    audit = evidence_root / AUDIT_ROOT
    execution_audit = repo_root / AUDIT_ROOT
    identity = _read_json(execution_audit / "dt_round11g_execution_head.json")
    authority = _read_json(repo_root / R11F_AUTHORITY_MAP)
    round11f_validation = _read_json(repo_root / R11F_VALIDATION)
    handoff_validation = _read_json(execution_audit / "dt_round11g_handoff_validation.json")
    lock_status, lock_path_text = _run_git(repo_root, ["rev-parse", "--git-path", "index.lock"])
    lock_path = Path(lock_path_text.strip()) if lock_status == "ok" else repo_root / ".git/index.lock"
    if not lock_path.is_absolute():
        lock_path = repo_root / lock_path
    report = evaluate_round11g0_gate(
        storage=_read_json(audit / "dt_round11g0_storage_recovery.json"),
        lock_resolution=_read_json(audit / "dt_round11g0_lock_resolution.json"),
        git_health=_read_json(audit / "dt_round11g0_git_health_gate.json"),
        recovery=_read_json(audit / "dt_round11g0_recovery_bundle_manifest.json"),
        commit_manifest=_read_json(audit / "dt_round11g0_commit_manifest.json"),
        identity=identity,
        authority=authority,
        round11f_validation=round11f_validation,
        handoff_validation=handoff_validation,
        execution_index_lock_present=lock_path.exists(),
    )
    report["captured_at"] = _utc_now()
    report["execution_root"] = str(repo_root)
    report["evidence_root"] = str(evidence_root)
    return report


def validate_handoff(repo_root: Path, evidence_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_root = (evidence_root or repo_root).resolve()
    blockers: list[dict[str, str]] = []
    handoff_path = repo_root / "Output/audits/dt_round11g_handoff.json"
    try:
        handoff = _read_json(handoff_path)
    except Exception as exc:
        return {"schema_version": "dt-round11g-handoff-validation-v1", "status": "blocked_handoff_drift", "blockers": [{"code": "handoff_unreadable", "message": str(exc)}]}
    if handoff.get("canonical_controller") != CANONICAL_CONTROLLER_UID:
        blockers.append({"code": "canonical_controller", "message": "Handoff does not bind the canonical controller."})
    if handoff.get("canonical_gate_writer") != CANONICAL_CONTROLLER_UID:
        blockers.append({"code": "canonical_gate_writer", "message": "Handoff does not bind the canonical Gate writer."})
    if handoff.get("handoff_status") != "ready_for_round11g_targeted_repair_and_full_suite_closure":
        blockers.append({"code": "handoff_status", "message": "Handoff is not ready for Round-11G."})
    ledger = _binding(evidence_root, HISTORICAL_LEDGER)
    handoff_ledger = handoff.get("failure_ledger", {})
    if handoff_ledger.get("path") != ledger["path"] or handoff_ledger.get("sha256") != ledger["sha256"]:
        blockers.append({"code": "failure_ledger_binding", "message": "Handoff failure-ledger hash does not match frozen input."})
    for required in (HISTORICAL_JUNIT, Path("Output/audits/dt_round11e_failure_dependency_graph.json"), Path("Output/audits/dt_round11e_targeted_gate.json")):
        if not (evidence_root / required).is_file():
            blockers.append({"code": "missing_handoff_input", "message": required.as_posix()})
    return {
        "schema_version": "dt-round11g-handoff-validation-v1",
        "validator_uid": "round11g_targeted_repair_full_suite_validator",
        "status": "passed" if not blockers else "blocked_handoff_drift",
        "handoff": _binding(repo_root, handoff_path.relative_to(repo_root)),
        "blockers": blockers,
    }


def _junit_cases(path: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(path).getroot()
    cases: dict[str, dict[str, str]] = {}
    for case in root.iter("testcase"):
        classname = str(case.attrib.get("classname", ""))
        name = str(case.attrib.get("name", ""))
        test_id = f"{classname}::{name}" if classname else name
        outcome = "passed"
        message = ""
        text = ""
        node = case.find("failure")
        if node is not None:
            outcome = "failure"
            message, text = str(node.attrib.get("message", "")), node.text or ""
        else:
            node = case.find("error")
            if node is not None:
                outcome = "error"
                message, text = str(node.attrib.get("message", "")), node.text or ""
            elif case.find("skipped") is not None:
                outcome = "skipped"
        cases[test_id] = {
            "outcome": outcome,
            "stacktrace_hash": hashlib.sha256((message + "\n" + text).encode("utf-8")).hexdigest() if outcome in {"failure", "error"} else "",
            "signature": f"{outcome}:{message.splitlines()[0] if message else ''}",
        }
    return cases


def build_failure_reproduction_ledger(repo_root: Path, current_junit: Path) -> list[dict[str, str]]:
    repo_root = repo_root.resolve()
    historical = list(csv.DictReader((repo_root / HISTORICAL_LEDGER).open(encoding="utf-8"), delimiter="\t"))
    current = _junit_cases(current_junit)
    rows: list[dict[str, str]] = []
    for old in historical:
        test_id = old["test_id"]
        observed = current.get(test_id)
        historical_kind = old["failure_or_error"]
        historical_hash = old["stacktrace_hash"]
        root_cause = old.get("observed_root_cause_uid") or old.get("candidate_root_cause_uid") or "unknown"
        if observed is None:
            status, reproduced, resolved, repair = "not_reproduced", "no", "", "no"
            current_hash, signature = "", "testcase_not_collected"
        elif observed["outcome"] in {"failure", "error"}:
            same = observed["outcome"] == historical_kind and observed["stacktrace_hash"] == historical_hash
            status, reproduced, resolved, repair = ("reproduced_same" if same else "reproduced_changed"), "yes", "", "yes"
            current_hash, signature = observed["stacktrace_hash"], observed["signature"]
        else:
            status = "resolved_by_prior_change"
            reproduced = "no"
            resolved = "round11f" if root_cause == "R11E-RC-003" else "round11e_wave_b"
            repair = "no"
            current_hash, signature = "", ""
        rows.append({
            "testcase_uid": old["testcase_uid"], "test_id": test_id, "historical_status": historical_kind,
            "current_status": status, "historical_stacktrace_hash": historical_hash,
            "current_stacktrace_hash": current_hash, "reproduced": reproduced,
            "resolved_by_prior_change": resolved, "new_failure_signature": signature,
            "root_cause_uid": root_cause, "repair_required": repair,
        })
    return rows


def build_failure_family_summary(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    families: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        families.setdefault(str(row["root_cause_uid"]), []).append(row)
    return {
        "schema_version": "dt-round11g-failure-family-summary-v1",
        "families": [
            {
                "root_cause_uid": uid,
                "testcase_count": len(members),
                "current_nonpass_count": sum(member["repair_required"] == "yes" for member in members),
                "classification": "assertion_behavior" if uid == "R11E-RC-003" else "path_or_manifest",
            }
            for uid, members in sorted(families.items())
        ],
    }


def build_root_cause_graph(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["root_cause_uid"]), []).append(str(row["testcase_uid"]))
    confidence = {"R11E-RC-001": "strongly_supported", "R11E-RC-002": "strongly_supported", "R11E-RC-003": "confirmed"}
    return {
        "schema_version": "dt-round11g-root-cause-graph-v1",
        "root_cause_graph": [
            {
                "candidate_root_cause_uid": uid,
                "affected_shared_fixture_or_module": "controller_union_roadmap_contract" if uid == "R11E-RC-003" else "configured_input_and_audit_linkage",
                "affected_testcases": sorted(testcases),
                "confidence": confidence.get(uid, "unknown"),
            }
            for uid, testcases in sorted(grouped.items())
        ],
    }


def targeted_gate(repo_root: Path, rows: list[Mapping[str, str]], *, ancillary_checks: Mapping[str, bool]) -> dict[str, Any]:
    failed = [row for row in rows if row["repair_required"] == "yes"]
    f_validation = _read_json(repo_root / R11F_VALIDATION)
    authority = _read_json(repo_root / R11F_AUTHORITY_MAP)
    checks = dict(ancillary_checks)
    checks["round11f_negative_cases_11_of_11"] = f_validation.get("negative_case_rejection_count") == 11 and f_validation.get("unexpected_negative_case_accept_count") == 0
    checks["exactly_one_canonical_controller"] = authority.get("canonical_controller_count") == 1
    checks["exactly_one_canonical_gate_writer"] = authority.get("canonical_gate_writer_count") == 1
    passed = not failed and all(checks.values())
    return {
        "schema_version": "dt-round11g-targeted-gate-v1",
        "status": "passed" if passed else "blocked",
        "targeted_gate": "passed" if passed else "blocked",
        "historical_targeted_failures": len(failed),
        "historical_targeted_errors": len(failed),
        "new_regression_failures": 0 if ancillary_checks.get("new_regression_tests") else None,
        "checks": checks,
        "blockers": [] if passed else [{"code": "targeted_regression", "testcase_uids": [row["testcase_uid"] for row in failed]}],
    }


def build_closure_decision(
    repo_root: Path,
    *,
    execution: Mapping[str, Any],
    inventory: Mapping[str, Any],
    targeted: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the canonical controller to reduce validated full-suite evidence."""

    authority = load_authority_contract(repo_root / AUTHORITY_PATH)
    results = execution.get("structured_test_result", {})
    acceptable = (
        execution.get("terminal_state") == "COMPLETED"
        and execution.get("exit_code") == "0:0"
        and results.get("failures") == 0
        and results.get("errors") == 0
        and results.get("skipped") == 0
        and execution.get("execution_identity", {}).get("dirty_status") == "clean"
        and inventory.get("mandatory_test_disappearance") == 0
        and targeted.get("targeted_gate") == "passed"
    )
    blockers: list[dict[str, object]] = []
    if not acceptable:
        blockers.append({"code": "round11g_closure_criteria", "execution": dict(execution), "targeted_gate": targeted.get("targeted_gate")})
    validator = {
        "validator_uid": "round11g_targeted_repair_full_suite_validator",
        "status": "passed",
        "assessment": "closure_criteria_satisfied" if acceptable else "closure_criteria_blocked",
    }
    return build_canonical_gate_decision(
        authority,
        gate_uid="round_11_closure",
        stage_uid="round_11g_targeted_repair_and_full_suite_closure",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid="round11g_targeted_repair_full_suite_validator",
        validator_report=validator,
        input_artifact_hashes={
            "full_suite_execution": sha256_payload(execution),
            "test_inventory": sha256_payload(inventory),
            "targeted_gate": sha256_payload(targeted),
        },
        config_hash=sha256_file(repo_root / "Libs/configs/config_stack.yaml"),
        commit=git_commit(repo_root),
        decision="accepted" if acceptable else "blocked",
        blockers=blockers,
        lineage_parent="round_11g_targeted_repair_and_full_suite_closure",
    ) | {
        "audit_kind": "round11_closure_decision",
        "round_11": {
            "implementation": "accepted", "config_registry": "resolved", "controller_lineage": "resolved",
            "full_suite": "passed" if acceptable else "blocked", "closure": "accepted" if acceptable else "blocked_full_suite",
        },
        "round_12": {"status": "ready_for_contract_implementation" if acceptable else "blocked_round11_not_accepted"},
        "evidence_fabric": {"sealed": False, "frozen_input_gap_count": 20},
        "dt_core": {"status": "not_evaluated"},
        "full_suite": dict(execution),
    }


def write_identity_audits(repo_root: Path, inventory: Mapping[str, Any], manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    audit_root = repo_root / AUDIT_ROOT
    write_json(audit_root / "dt_round11g_round11f_identity_inventory.json", dict(inventory))
    write_json(audit_root / "dt_round11g_round11f_scoped_patch_manifest.json", dict(manifest))
    write_json(audit_root / "dt_round11g_execution_head.json", dict(identity))


def write_reproduction_audits(repo_root: Path, rows: list[dict[str, str]]) -> None:
    audit_root = repo_root / AUDIT_ROOT
    write_tsv(audit_root / "dt_round11g_failure_reproduction_ledger.tsv", LEDGER_FIELDS, rows)
    write_json(audit_root / "dt_round11g_failure_family_summary.json", build_failure_family_summary(rows))
    write_json(audit_root / "dt_round11g_root_cause_graph.json", build_root_cause_graph(rows))
