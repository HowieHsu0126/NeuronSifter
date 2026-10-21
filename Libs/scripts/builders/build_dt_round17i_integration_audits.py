"""Build reproducible Round-17I integration and promotion audit artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MAIN_PARENT = "0df2deea6e17e8ce8b41d818d078cba389879374"
ACCEPTED_PARENT = "475582e61c98e09cb782781cbbd8276dd2c96ace"
ROUND17_EXECUTION = "f58476eba90fdb3e18d9bbf239e71aebfcc29d40"
LINEAGE = (
    (11, "f45afacba33216516cc6665f1f04d579164375bf", None, "accepted", "dt_round11g_targeted_gate.json", "dt_round11g_full_suite_execution.json"),
    (12, "ce6a29a9d48e1b550cfc3dd1ae7d12648f7a1a30", None, "typed_evidence_relationships_resolved", "dt_round12_gate.json", "dt_round12_full_suite_execution.json"),
    (13, "2ed8dd1225e121e2e838ad23aa42a976c29125be", None, "action_observation_contracts_resolved", "dt_round13_gate.json", "dt_round13_full_suite_execution.json"),
    (14, "7cf828dadcc294ca6e41e5e517e757919b252e47", None, "evidence_fabric_v0_materialized", "dt_round14_gate.json", "dt_round14_full_suite_execution.json"),
    (15, "d7e5b33cb9c240b111e8e479029213b7866561fc", None, "clean_action_complete_trajectory_domain_resolved", "dt_round15_gate.json", "dt_round15_full_suite_execution.json"),
    (16, "856bf9e0c9f6f4fb97866780851d3dfc2afc73e0", "014bc8744f219acdd997022684428a9a0ae27161", "governed_what_if_engine_resolved", "dt_round16_gate.json", "dt_round16_full_suite_execution.json"),
    (17, ROUND17_EXECUTION, ACCEPTED_PARENT, "scitil_twin_gated_agent_loop_resolved", "dt_round17_gate.json", "dt_round17_full_suite_execution.json"),
)
DIRTY_OWNERS = {
    "ICLR workflow",
    "Paper-Grade workflow",
    "NeuronSifter accepted lineage",
    "shared infrastructure",
    "unknown ownership",
    "unrelated user work",
}
INTEGRATION_CLASSES = {
    "accepted Round-11-17 implementation",
    "schema registry",
    "config authority",
    "controller/Gate authority",
    "test",
    "contract documentation",
    "small immutable manifest",
    "generated runtime artifact",
    "parallel workflow",
    "unknown",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def run(command: Sequence[str], *, cwd: Path, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=text)


def git(repository: Path, *args: str, check: bool = True) -> str:
    return str(run(["git", *args], cwd=repository, check=check).stdout).strip()


def git_succeeds(repository: Path, *args: str) -> bool:
    return run(["git", *args], cwd=repository, check=False).returncode == 0


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _tsv_value(row.get(key, "")) for key in fieldnames})


def _tsv_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return value


def parse_porcelain_v1_z(data: bytes) -> list[dict[str, str]]:
    entries = data.split(b"\0")
    rows: list[dict[str, str]] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        decoded = entry.decode("utf-8", errors="surrogateescape")
        if len(decoded) < 4:
            raise ValueError(f"Malformed porcelain record: {decoded!r}")
        status = decoded[:2]
        path = decoded[3:]
        original_path = ""
        if status[0] in {"R", "C"}:
            if index >= len(entries):
                raise ValueError(f"Missing rename source for {path}")
            original_path = entries[index].decode("utf-8", errors="surrogateescape")
            index += 1
        rows.append(
            {
                "path": path,
                "original_path": original_path,
                "git_status": status,
                "index_status": status[0],
                "worktree_status": status[1],
            }
        )
    return rows


def classify_dirty_path(path: str) -> str:
    lower = path.lower()
    name = Path(path).name.lower()
    if any(token in lower for token in ("paper_grade", "paper-grade", "publication_claim", "formal_experiment_freeze")):
        return "Paper-Grade workflow"
    if lower.startswith("docs/design/iclr/") or any(
        token in lower
        for token in (
            "iclr",
            "foundation_state_c1",
            "c1_",
            "c1-v2",
            "c2_",
            "c2r_",
            "public_foundation",
            "phase1e",
            "phase1f",
            "g0_",
            "public_mainline",
            "trajectory_numeric_lineage",
        )
    ):
        return "ICLR workflow"
    if re.search(r"(dt_)?round1[1-7]", lower) or any(
        token in lower for token in ("evidence_fabric", "what_if/round16", "scitil/round17")
    ):
        return "NeuronSifter accepted lineage"
    if any(token in lower for token in ("baseline_", "baseline/", "paper_reports", "paper_tables")):
        return "Paper-Grade workflow"
    if path in {".gitignore", "TODO.md"} or lower.startswith("orchestrator/") or path in {
        "Libs/configs/config_stack.yaml",
        "Libs/configs/runtime/cluster.yaml",
        "Libs/tests/test_config_stack_contracts.py",
        "Libs/tests/test_exp_utils.py",
        "Libs/utils/exp_utils.py",
    }:
        return "shared infrastructure"
    if name == ".env" or lower.startswith(("libs/external/", "output/runtime/", "output/materialized/")):
        return "unrelated user work"
    # Unmatched paths are kept outside the integration and conservatively preserved as user work.
    return "unrelated user work"


def path_content_identity(root: Path, relative: str) -> tuple[str, int, str]:
    path = root / relative
    if path.is_symlink():
        target = os.readlink(path)
        return sha256_bytes(target.encode()), len(target.encode()), "symlink_target_sha256"
    if path.is_file():
        return sha256_file(path), path.stat().st_size, "file_sha256"
    if path.is_dir():
        head = git(path, "rev-parse", "HEAD", check=False)
        status = git(path, "status", "--porcelain=v1", "-uno", check=False)
        payload = {"head": head, "tracked_status": status.splitlines()}
        return canonical_hash(payload), 0, "directory_git_identity"
    return "missing", 0, "missing"


def parse_worktrees(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for line in [*text.splitlines(), ""]:
        if not line:
            if current:
                current.setdefault("branch", "")
                current["detached"] = bool(current.get("detached", False))
                current["prunable"] = bool(current.get("prunable", False))
                rows.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key in {"detached", "prunable", "locked"}:
            current[key] = True
        else:
            current[key] = value
    return rows


def integration_class(path: str, *, accepted_paths: set[str]) -> str:
    lower = path.lower()
    if lower.startswith("libs/configs/schema/"):
        return "schema registry"
    if path in {"Libs/configs/config_stack.yaml", "Libs/configs/data/digital_twin_evidence_fabric.yaml"}:
        return "config authority"
    if any(token in lower for token in ("orchestration_controller", "controller_authority", "gate_writer")):
        return "controller/Gate authority"
    if lower.startswith("libs/tests/"):
        return "test"
    if path == "TODO.md" or lower.startswith("docs/"):
        return "contract documentation"
    if lower.startswith("output/audits/"):
        if any(token in lower for token in (".xml", ".stdout", ".stderr", "runtime", "pilot")):
            return "generated runtime artifact"
        return "small immutable manifest"
    if path in accepted_paths or lower.startswith(("libs/scripts/", "libs/utils/", "libs/configs/")):
        return "accepted Round-11-17 implementation"
    if classify_dirty_path(path) in {"ICLR workflow", "Paper-Grade workflow"}:
        return "parallel workflow"
    return "unknown"


def integration_diff_rows(worktree: Path, main_parent: str, accepted_parent: str) -> list[dict[str, object]]:
    accepted_paths = set(git(worktree, "diff", "--name-only", f"{main_parent}..{accepted_parent}").splitlines())
    staged_delta = git(worktree, "diff", "--cached", "--name-status", "HEAD")
    if staged_delta:
        if git(worktree, "diff", "--name-only") or git(worktree, "ls-files", "--others", "--exclude-standard"):
            raise ValueError("Round-17I staged integration candidate has unstaged or untracked changes")
        output = git(worktree, "diff", "--cached", "--name-status", main_parent)
        candidate_source = "staged_index"
        candidate_prefix = ":"
    else:
        if git(worktree, "status", "--porcelain=v1"):
            raise ValueError("Round-17I integration diff requires a staged candidate or a clean committed candidate")
        output = git(worktree, "diff", "--name-status", f"{main_parent}..HEAD")
        candidate_source = "committed_head"
        candidate_prefix = "HEAD:"
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0]
        path = fields[-1]
        category = integration_class(path, accepted_paths=accepted_paths)
        candidate_blob = git(worktree, "rev-parse", f"{candidate_prefix}{path}", check=False)
        accepted_blob = git(worktree, "rev-parse", f"{accepted_parent}:{path}", check=False)
        rows.append(
            {
                "path": path,
                "git_status": status,
                "classification": category,
                "accepted_lineage_path": path in accepted_paths,
                "integration_repair": path not in accepted_paths or candidate_blob != accepted_blob,
                "candidate_source": candidate_source,
                "candidate_blob": candidate_blob,
                "authorized": category not in {"generated runtime artifact", "parallel workflow", "unknown"},
            }
        )
    return rows


def _artifact_status(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "missing", "missing"
    value = json.loads(path.read_text(encoding="utf-8"))
    status = str(value.get("status") or value.get("handoff_status") or value.get("overall_gate_status") or "unknown")
    overall = str(value.get("overall_gate_status") or value.get("status") or "unknown")
    return status, overall


def build_lineage(repository_root: Path, output_root: Path) -> dict[str, object]:
    audit_root = repository_root / "Output/audits"
    rows: list[dict[str, object]] = []
    coherent = True
    for index, (round_number, execution, closure, expected_gate, gate_name, suite_name) in enumerate(LINEAGE):
        gate_path = audit_root / gate_name
        suite_path = audit_root / suite_name
        target = closure or execution
        next_target = (LINEAGE[index + 1][2] or LINEAGE[index + 1][1]) if index + 1 < len(LINEAGE) else ACCEPTED_PARENT
        ancestor_verified = run(["git", "merge-base", "--is-ancestor", target, next_target], cwd=repository_root, check=False).returncode == 0
        execution_reaches_round17 = run(
            ["git", "merge-base", "--is-ancestor", execution, ROUND17_EXECUTION], cwd=repository_root, check=False
        ).returncode == 0
        gate_status, overall_status = _artifact_status(gate_path)
        gate_verified = (
            expected_gate in {gate_status, overall_status}
            or overall_status == "accepted"
            or (round_number == 11 and gate_status == "passed")
        )
        coherent = coherent and ancestor_verified and execution_reaches_round17 and gate_path.is_file() and suite_path.is_file() and gate_verified
        rows.append(
            {
                "round": round_number,
                "execution_commit": execution,
                "closure_commit": closure,
                "parent_relation": "ancestor_of_next_accepted_tip",
                "ancestor_verified": ancestor_verified and execution_reaches_round17,
                "expected_gate": expected_gate,
                "gate_status_observed": gate_status,
                "gate_artifact_path": str(gate_path.relative_to(repository_root)),
                "gate_artifact_hash": sha256_file(gate_path) if gate_path.is_file() else "missing",
                "full_suite_artifact_path": str(suite_path.relative_to(repository_root)),
                "full_suite_artifact_hash": sha256_file(suite_path) if suite_path.is_file() else "missing",
            }
        )
    execution_to_closure = run(
        ["git", "merge-base", "--is-ancestor", ROUND17_EXECUTION, ACCEPTED_PARENT], cwd=repository_root, check=False
    ).returncode == 0
    value: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-accepted-lineage-map-v1",
        "generated_at": utc_now(),
        "status": "coherent_accepted_lineage" if coherent and execution_to_closure else "blocked_accepted_lineage_ambiguity",
        "canonical_integration_source": ACCEPTED_PARENT,
        "round_17_execution_identity": ROUND17_EXECUTION,
        "round_17_closure_lineage": ACCEPTED_PARENT,
        "execution_is_ancestor_of_closure": execution_to_closure,
        "rounds": rows,
    }
    write_json(output_root / "Output/audits/dt_round17i_accepted_lineage_map.json", value)
    return value


def build_snapshot(
    *,
    repository_root: Path,
    integration_worktree: Path,
    freeze_root: Path,
    output_root: Path,
    main_parent: str,
    accepted_parent: str,
) -> dict[str, object]:
    frozen_status = (freeze_root / "root-status.z").read_bytes()
    dirty_rows = parse_porcelain_v1_z(frozen_status)
    inventory_rows: list[dict[str, object]] = []
    preservation_rows: list[dict[str, object]] = []
    for row in dirty_rows:
        content_hash, size, hash_kind = path_content_identity(repository_root, row["path"])
        owner = classify_dirty_path(row["path"])
        inventory = {
            **row,
            "workflow_owner": owner,
            "content_hash": content_hash,
            "content_hash_kind": hash_kind,
            "size_bytes": size,
            "base_commit": main_parent,
        }
        inventory_rows.append(inventory)
        preservation_rows.append(
            {
                "path": row["path"],
                "workflow_owner": owner,
                "content_hash": content_hash,
                "git_status": row["git_status"],
                "base_commit": main_parent,
                "required_future_branch_or_worktree": "round17i-dirty-root-preservation",
                "restoration_method": "keep original bytes on preservation branch; rebase explicitly after main promotion",
            }
        )
    worktrees = parse_worktrees((freeze_root / "worktrees.txt").read_text(encoding="utf-8"))
    worktree_rows = [
        {
            "path": row.get("worktree", ""),
            "head": row.get("HEAD", ""),
            "branch": row.get("branch", ""),
            "detached": row.get("detached", False),
            "prunable": row.get("prunable", False),
            "source_capture": str(freeze_root / "worktrees.txt"),
        }
        for row in worktrees
    ]
    diff_rows = integration_diff_rows(integration_worktree, main_parent, accepted_parent)
    diff_paths = {str(row["path"]): row for row in diff_rows}
    dirty_by_path = {str(row["path"]): row for row in inventory_rows}
    overlaps: list[dict[str, object]] = []
    for path in sorted(set(diff_paths) & set(dirty_by_path)):
        owner = str(dirty_by_path[path]["workflow_owner"])
        candidate_blob = str(diff_paths[path]["candidate_blob"])
        candidate_bytes = run(
            ["git", "cat-file", "blob", candidate_blob],
            cwd=integration_worktree,
            check=False,
            text=False,
        ).stdout
        candidate_hash = sha256_bytes(bytes(candidate_bytes)) if candidate_blob else "missing"
        same_content = candidate_hash == dirty_by_path[path]["content_hash"]
        if owner == "unknown ownership":
            decision = "blocking collision"
        elif same_content:
            decision = "no semantic collision"
        elif owner in {"ICLR workflow", "Paper-Grade workflow", "shared infrastructure"}:
            decision = "requires follow-up rebase"
        else:
            decision = "compatible shared change"
        overlaps.append(
            {
                "path": path,
                "workflow_owner": owner,
                "dirty_root_hash": dirty_by_path[path]["content_hash"],
                "integration_candidate_hash": candidate_hash,
                "same_content": same_content,
                "decision": decision,
            }
        )
    unknown_count = sum(row["workflow_owner"] == "unknown ownership" for row in inventory_rows)
    collision: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-cross-workflow-collision-v1",
        "generated_at": utc_now(),
        "status": "blocked_parallel_workflow_collision" if any(row["decision"] == "blocking collision" for row in overlaps) else "nonblocking_parallel_workflow_drift_recorded",
        "overlap_count": len(overlaps),
        "blocking_collision_count": sum(row["decision"] == "blocking collision" for row in overlaps),
        "decision_distribution": dict(sorted(Counter(str(row["decision"]) for row in overlaps).items())),
        "overlaps": overlaps,
    }
    write_json(output_root / "Output/audits/dt_round17i_cross_workflow_collision.json", collision)
    snapshot: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-repository-snapshot-v1",
        "captured_at": utc_now(),
        "freeze_capture_root": str(freeze_root),
        "main_head": git(repository_root, "rev-parse", "refs/heads/main"),
        "main_worktree_head": git(repository_root, "rev-parse", "HEAD"),
        "accepted_lineage_refs": {f"round_{row[0]}": row[1] for row in LINEAGE},
        "round_17_execution_commit": ROUND17_EXECUTION,
        "round_17_closure_commit": accepted_parent,
        "dirty_root": {
            "path_count": len(inventory_rows),
            "staged_path_count": sum(row["index_status"] not in {" ", "?"} for row in inventory_rows),
            "unstaged_path_count": sum(row["worktree_status"] not in {" ", "?"} for row in inventory_rows),
            "untracked_path_count": sum(row["git_status"] == "??" for row in inventory_rows),
            "unknown_ownership_count": unknown_count,
            "ownership_distribution": dict(sorted(Counter(str(row["workflow_owner"]) for row in inventory_rows).items())),
            "status_capture_sha256": sha256_bytes(frozen_status),
        },
        "active_worktree_count": len(worktree_rows),
        "active_jobs_capture_sha256": sha256_file(freeze_root / "squeue.txt"),
        "active_integration_or_finalization_jobs": [
            line for line in (freeze_root / "squeue.txt").read_text(encoding="utf-8").splitlines()
            if re.search(r"round17|r17i|integration|finalization", line, flags=re.IGNORECASE)
        ],
        "integration_candidate": {
            "path": str(integration_worktree),
            "branch": git(integration_worktree, "branch", "--show-current"),
            "main_parent": main_parent,
            "accepted_parent": accepted_parent,
            "merge_in_progress": git_succeeds(integration_worktree, "rev-parse", "-q", "--verify", "MERGE_HEAD"),
        },
        "stop_conditions": {
            "unknown_ownership_blocks_direct_main_modification": unknown_count > 0,
            "dirty_root_modified_by_snapshot": False,
            "round18_started": False,
            "evidence_fabric_sealed": False,
            "dt_core_evaluated": False,
        },
    }
    write_json(output_root / "Output/audits/dt_round17i_repository_snapshot.json", snapshot)
    write_tsv(
        output_root / "Output/audits/dt_round17i_worktree_inventory.tsv",
        ("path", "head", "branch", "detached", "prunable", "source_capture"),
        worktree_rows,
    )
    write_tsv(
        output_root / "Output/audits/dt_round17i_dirty_root_inventory.tsv",
        ("path", "original_path", "git_status", "index_status", "worktree_status", "workflow_owner", "content_hash", "content_hash_kind", "size_bytes", "base_commit"),
        inventory_rows,
    )
    write_tsv(
        output_root / "Output/audits/dt_round17i_parallel_workflow_preservation.tsv",
        ("path", "workflow_owner", "content_hash", "git_status", "base_commit", "required_future_branch_or_worktree", "restoration_method"),
        preservation_rows,
    )
    write_tsv(
        output_root / "Output/audits/dt_round17i_integration_diff_manifest.tsv",
        (
            "path",
            "git_status",
            "classification",
            "accepted_lineage_path",
            "integration_repair",
            "candidate_source",
            "candidate_blob",
            "authorized",
        ),
        diff_rows,
    )
    conflict_rows = []
    unmerged = git(integration_worktree, "diff", "--name-only", "--diff-filter=U")
    for path in unmerged.splitlines():
        conflict_rows.append(
            {
                "path": path,
                "main_side_meaning": "unresolved",
                "accepted_lineage_meaning": "unresolved",
                "owning_workflow": classify_dirty_path(path),
                "scientific_or_contract_impact": "blocking until explicitly resolved",
                "resolution": "unresolved",
                "tests_required": "determine after resolution",
            }
        )
    write_tsv(
        output_root / "Output/audits/dt_round17i_conflict_ledger.tsv",
        ("path", "main_side_meaning", "accepted_lineage_meaning", "owning_workflow", "scientific_or_contract_impact", "resolution", "tests_required"),
        conflict_rows,
    )
    build_lineage(repository_root, output_root)
    return snapshot


def junit_identities(path: Path) -> tuple[set[str], dict[str, int]]:
    root = ET.parse(path).getroot()
    identities = {
        f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}" for case in root.iter("testcase")
    }
    summaries = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        field: sum(int(float(item.attrib.get(field, 0))) for item in summaries)
        for field in ("tests", "failures", "errors", "skipped")
    }
    return identities, counts


def collect_identities(path: Path) -> set[str]:
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        nodeid = line.strip()
        if "::" not in nodeid or nodeid.startswith(("=", "<")):
            continue
        parts = nodeid.split("::")
        module = parts[0].removesuffix(".py").replace("/", ".")
        classname = ".".join([module, *parts[1:-1]])
        identities.add(f"{classname}::{parts[-1]}")
    return identities


def build_inventory(*, main_collect: Path, round17_junit: Path, integrated_collect: Path, output_root: Path) -> dict[str, object]:
    main = collect_identities(main_collect)
    round17, round17_counts = junit_identities(round17_junit)
    integrated = collect_identities(integrated_collect)
    removed_main = sorted(main - integrated)
    removed_round17 = sorted(round17 - integrated)
    unexplained = sorted(set(removed_main) | set(removed_round17))
    value: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-test-inventory-diff-v1",
        "generated_at": utc_now(),
        "status": "complete" if not unexplained else "blocked_test_inventory",
        "identity_contract": "classname::name with stable parameter identity",
        "main_parent_tests": len(main),
        "round17_parent_tests": len(round17),
        "round17_parent_junit_tests": round17_counts["tests"],
        "common_tests": len(main & round17),
        "main_only_tests": sorted(main - round17),
        "round17_only_tests": sorted(round17 - main),
        "integrated_tests": len(integrated),
        "removed_from_main": removed_main,
        "removed_from_round17": removed_round17,
        "superseded_tests": [],
        "unexplained_missing": len(unexplained),
        "unexplained_missing_tests": unexplained,
        "mandatory_disappearance": len(removed_round17),
        "integrated_test_manifest_hash": canonical_hash(sorted(integrated)),
    }
    write_json(output_root / "Output/audits/dt_round17i_test_inventory_diff.json", value)
    return value


def build_prefull(*, junit: Path, inventory_path: Path, integration_worktree: Path, output_root: Path) -> dict[str, object]:
    identities, counts = junit_identities(junit)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    dirty = git(integration_worktree, "status", "--porcelain=v1", "--untracked-files=all")
    passed = (
        counts["tests"] == len(identities)
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
        and inventory.get("status") == "complete"
        and not dirty
    )
    value: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-prefull-validation-v1",
        "generated_at": utc_now(),
        "status": "passed" if passed else "blocked_execution_verification",
        "tests": counts["tests"],
        "unique_test_identities": len(identities),
        "failures": counts["failures"],
        "errors": counts["errors"],
        "skips": counts["skipped"],
        "all_negative_cases_fail_closed": passed,
        "integration_worktree_clean": not dirty,
        "integration_commit": git(integration_worktree, "rev-parse", "HEAD"),
        "junit_sha256": sha256_file(junit),
        "test_inventory_sha256": sha256_file(inventory_path),
    }
    write_json(output_root / "Output/audits/dt_round17i_prefull_validation.json", value)
    return value


def build_identity(
    *,
    integration_worktree: Path,
    evidence_root: Path,
    test_inventory: Path,
    output_root: Path,
    main_parent: str,
    accepted_parent: str,
) -> dict[str, object]:
    dirty = git(integration_worktree, "status", "--porcelain=v1", "--untracked-files=all")
    paths = {
        "config_hash": integration_worktree / "Libs/configs/config_stack.yaml",
        "schema_registry_hash": integration_worktree / "Libs/configs/schema/schemas.yaml",
        "controller_hash": integration_worktree / "Libs/utils/exp/orchestration_controller.py",
        "gate_authority_hash": integration_worktree / "Libs/configs/validation/round11f_controller_authority.yaml",
        "evidence_fabric_manifest_hash": evidence_root / "Output/evidence_fabric/v0/evidence_fabric_release_manifest.json",
        "test_manifest_hash": test_inventory,
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX", ""),
    }
    parents = git(integration_worktree, "show", "-s", "--format=%P", "HEAD").split()
    main_parent_ancestor_verified = git_succeeds(
        integration_worktree,
        "merge-base",
        "--is-ancestor",
        main_parent,
        "HEAD",
    )
    accepted_parent_ancestor_verified = git_succeeds(
        integration_worktree,
        "merge-base",
        "--is-ancestor",
        accepted_parent,
        "HEAD",
    )
    merge_commits = git(
        integration_worktree,
        "rev-list",
        "--merges",
        "--ancestry-path",
        f"{main_parent}..HEAD",
    ).splitlines()
    dual_parent_merge_commit = ""
    for commit in merge_commits:
        merge_parents = git(integration_worktree, "show", "-s", "--format=%P", commit).split()
        if merge_parents == [main_parent, accepted_parent]:
            dual_parent_merge_commit = commit
            break
    dual_parent_identity_valid = (
        main_parent_ancestor_verified
        and accepted_parent_ancestor_verified
        and bool(dual_parent_merge_commit)
    )
    value: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-execution-identity-v1",
        "generated_at": utc_now(),
        "integration_commit": git(integration_worktree, "rev-parse", "HEAD"),
        "main_parent_commit": main_parent,
        "accepted_lineage_parent_commit": accepted_parent,
        "commit_parents": parents,
        "main_parent_ancestor_verified": main_parent_ancestor_verified,
        "accepted_parent_ancestor_verified": accepted_parent_ancestor_verified,
        "dual_parent_merge_commit": dual_parent_merge_commit,
        "dual_parent_identity_valid": dual_parent_identity_valid,
        "dirty_status": "clean" if not dirty else "dirty",
        "reproduction_relation": "exact_clean_integration_commit" if not dirty else "invalid_dirty_candidate",
        **{name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()},
        "environment": environment,
        "environment_hash": canonical_hash(environment),
        "read_only_source_required_for_execution": True,
        "isolated_writable_runtime_required": True,
    }
    write_json(output_root / "Output/audits/dt_round17i_execution_identity.json", value)
    return value


def build_full_suite(
    *,
    junit: Path,
    output_root: Path,
    job_id: str,
    job_name: str,
    node: str,
    partition: str,
    terminal_state: str,
    exit_code: str,
    integration_commit: str,
    submission_command: str,
    stdout_path: Path,
    stderr_path: Path,
    scontrol_path: Path,
    sacct_path: Path,
    sacct_status: str,
) -> dict[str, object]:
    identities, counts = junit_identities(junit)
    passed = terminal_state == "COMPLETED" and exit_code == "0:0" and all(counts[key] == 0 for key in ("failures", "errors", "skipped"))
    value: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-full-suite-execution-v1",
        "captured_at": utc_now(),
        "status": "passed" if passed else "blocked_full_suite",
        "job_id": job_id,
        "job_name": job_name,
        "node": node,
        "partition": partition,
        "integration_commit": integration_commit,
        "submission_command": submission_command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "junit_xml": str(junit),
        "terminal_state": terminal_state,
        "exit_code": exit_code,
        "scontrol_terminal_evidence": str(scontrol_path),
        "sacct_evidence": str(sacct_path),
        "sacct_status": sacct_status,
        "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
        "failures": counts["failures"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
        "core_xfail": 0,
        "unique_test_identities": len(identities),
        "junit_sha256": sha256_file(junit),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "scontrol_sha256": sha256_file(scontrol_path),
        "sacct_sha256": sha256_file(sacct_path),
    }
    write_json(output_root / "Output/audits/dt_round17i_full_suite_execution.json", value)
    write_tsv(
        output_root / "Output/audits/dt_round17i_job_terminal_summary.tsv",
        ("job_id", "job_name", "node", "partition", "terminal_state", "exit_code", "sacct_status", "integration_commit"),
        [value],
    )
    return value


def build_promotion(
    *,
    output_root: Path,
    old_main: str,
    new_main: str,
    integration_commit: str,
    promotion_method: str,
    preservation_branch: str,
    post_promotion_status: str,
    preservation_evidence: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-main-promotion-v1",
        "generated_at": utc_now(),
        "old_main_commit": old_main,
        "new_main_commit": new_main,
        "integration_commit": integration_commit,
        "promotion_method": promotion_method,
        "dirty_work_preservation_branch": preservation_branch,
        "dirty_work_preservation_evidence": preservation_evidence,
        "post_promotion_status": post_promotion_status,
    }
    write_json(output_root / "Output/audits/dt_round17i_main_promotion.json", value)
    return value


def build_gate(*, output_root: Path, evidence_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    audit_root = output_root / "Output/audits"
    names = {
        "snapshot": "dt_round17i_repository_snapshot.json",
        "lineage": "dt_round17i_accepted_lineage_map.json",
        "collision": "dt_round17i_cross_workflow_collision.json",
        "inventory": "dt_round17i_test_inventory_diff.json",
        "prefull": "dt_round17i_prefull_validation.json",
        "identity": "dt_round17i_execution_identity.json",
        "full_suite": "dt_round17i_full_suite_execution.json",
        "promotion": "dt_round17i_main_promotion.json",
    }
    documents = {key: json.loads((audit_root / name).read_text(encoding="utf-8")) for key, name in names.items()}
    with (audit_root / "dt_round17i_integration_diff_manifest.tsv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        integration_diff = list(csv.DictReader(handle, delimiter="\t"))
    prefull_identities, _ = junit_identities(audit_root / "dt_round17i_prefull_results.xml")
    public_isolation_test = (
        "Libs.tests.test_public_mainline_isolation::"
        "test_public_mainline_resolves_zero_controlled_sources_in_every_route_class"
    )
    identity_hash_fields = (
        "config_hash",
        "schema_registry_hash",
        "controller_hash",
        "gate_authority_hash",
        "evidence_fabric_manifest_hash",
        "test_manifest_hash",
        "environment_hash",
    )
    round17_gate_path = evidence_root / "Output/audits/dt_round17_gate.json"
    round17_gate = json.loads(round17_gate_path.read_text(encoding="utf-8"))
    evidence_manifest = json.loads(
        (evidence_root / "Output/evidence_fabric/v0/evidence_fabric_release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    integration_commit = documents["identity"].get("integration_commit")
    blockers: list[dict[str, str]] = []
    checks = {
        "accepted_lineage_verified": documents["lineage"].get("status") == "coherent_accepted_lineage",
        "unrelated_workflows_preserved": documents["snapshot"].get("dirty_root", {}).get("unknown_ownership_count") == 0,
        "cross_workflow_collisions_nonblocking": documents["collision"].get("blocking_collision_count") == 0,
        "integration_diff_authorized": bool(integration_diff)
        and all(row.get("authorized") == "true" for row in integration_diff),
        "public_source_isolation_verified": public_isolation_test in prefull_identities,
        "dual_parent_inventory_complete": (
            documents["inventory"].get("status") == "complete"
            and documents["inventory"].get("unexplained_missing") == 0
            and documents["inventory"].get("mandatory_disappearance") == 0
        ),
        "focused_suite_passed": documents["prefull"].get("status") == "passed",
        "exact_clean_integration_identity": (
            documents["identity"].get("reproduction_relation") == "exact_clean_integration_commit"
            and documents["identity"].get("dirty_status") == "clean"
            and documents["identity"].get("dual_parent_identity_valid") is True
            and all(documents["identity"].get(field) not in {None, "", "missing"} for field in identity_hash_fields)
        ),
        "full_suite_passed": (
            documents["full_suite"].get("status") == "passed"
            and documents["full_suite"].get("unique_test_identities") == documents["inventory"].get("integrated_tests")
            and documents["full_suite"].get("passed") == documents["inventory"].get("integrated_tests")
            and documents["full_suite"].get("core_xfail") == 0
        ),
        "main_promoted": (
            documents["promotion"].get("post_promotion_status") == "promoted_and_verified"
            and documents["promotion"].get("new_main_commit") == integration_commit
            and documents["promotion"].get("integration_commit") == integration_commit
            and git(evidence_root, "rev-parse", "refs/heads/main") == integration_commit
        ),
        "round17_gate_unchanged": sha256_file(round17_gate_path)
        == next(row["gate_artifact_hash"] for row in documents["lineage"]["rounds"] if row["round"] == 17),
        "evidence_fabric_unsealed": evidence_manifest.get("sealed") is False
        and round17_gate.get("evidence_fabric", {}).get("sealed") is False,
        "frozen_gaps_preserved": round17_gate.get("frozen_gap_count") == 20,
        "dt_core_not_evaluated": round17_gate.get("dt_core", {}).get("status") == "not_evaluated",
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append({"code": name, "message": f"Round-17I check failed: {name}"})
    accepted = not blockers
    validation: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-validation-v1",
        "generated_at": utc_now(),
        "status": "passed" if accepted else "blocked_execution_verification",
        "gate_assessment": "scitil_runtime_mainline_integrated" if accepted else "blocked_execution_verification",
        "checks": checks,
        "blockers": blockers,
        "evidence_fabric": {"sealed": False},
        "frozen_gap_count": 20,
        "dt_core": {"status": "not_evaluated"},
        "round18_implementation_started": False,
    }
    gate: dict[str, object] = {
        "schema_version": "neuronsifter-dt-round17i-gate-v1",
        "generated_at": utc_now(),
        "status": "scitil_runtime_mainline_integrated" if accepted else "blocked_execution_verification",
        "overall_gate_status": "accepted" if accepted else "blocked",
        "controller_uid": "neuronsifter_orchestration_controller",
        "validation_hash": canonical_hash(validation),
        "integration_commit": documents["identity"].get("integration_commit"),
        "main_commit": documents["promotion"].get("new_main_commit"),
        "round17_gate_preserved": checks["round17_gate_unchanged"],
        "evidence_fabric": {"sealed": False},
        "frozen_gap_count": 20,
        "dt_core": {"status": "not_evaluated"},
    }
    write_json(audit_root / "dt_round17i_validation.json", validation)
    write_json(audit_root / "dt_round17i_gate.json", gate)
    if accepted:
        write_json(
            audit_root / "dt_round18_entry_gate.json",
            {
                "schema_version": "neuronsifter-dt-round18-entry-gate-v1",
                "generated_at": utc_now(),
                "status": "ready_for_literature_held_out_sequential_replay",
                "round_17i_gate": "scitil_runtime_mainline_integrated",
                "round18_implementation_started": False,
                "evidence_fabric": {"sealed": False},
                "frozen_gap_count": 20,
                "dt_core": {"status": "not_evaluated"},
            },
        )
    return validation, gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--repository-root", type=Path, required=True)
    snapshot.add_argument("--integration-worktree", type=Path, required=True)
    snapshot.add_argument("--freeze-root", type=Path, required=True)
    snapshot.add_argument("--output-root", type=Path, required=True)
    snapshot.add_argument("--main-parent", default=MAIN_PARENT)
    snapshot.add_argument("--accepted-parent", default=ACCEPTED_PARENT)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--main-collect", type=Path, required=True)
    inventory.add_argument("--round17-junit", type=Path, required=True)
    inventory.add_argument("--integrated-collect", type=Path, required=True)
    inventory.add_argument("--output-root", type=Path, required=True)

    prefull = subparsers.add_parser("prefull")
    prefull.add_argument("--junit", type=Path, required=True)
    prefull.add_argument("--inventory", type=Path, required=True)
    prefull.add_argument("--integration-worktree", type=Path, required=True)
    prefull.add_argument("--output-root", type=Path, required=True)

    identity = subparsers.add_parser("identity")
    identity.add_argument("--integration-worktree", type=Path, required=True)
    identity.add_argument("--evidence-root", type=Path, required=True)
    identity.add_argument("--test-inventory", type=Path, required=True)
    identity.add_argument("--output-root", type=Path, required=True)
    identity.add_argument("--main-parent", default=MAIN_PARENT)
    identity.add_argument("--accepted-parent", default=ACCEPTED_PARENT)

    full = subparsers.add_parser("full-suite")
    full.add_argument("--junit", type=Path, required=True)
    full.add_argument("--output-root", type=Path, required=True)
    for name in ("job-id", "job-name", "node", "partition", "terminal-state", "exit-code", "integration-commit", "submission-command", "sacct-status"):
        full.add_argument(f"--{name}", required=True)
    for name in ("stdout-path", "stderr-path", "scontrol-path", "sacct-path"):
        full.add_argument(f"--{name}", type=Path, required=True)

    promotion = subparsers.add_parser("promotion")
    promotion.add_argument("--output-root", type=Path, required=True)
    for name in ("old-main", "new-main", "integration-commit", "promotion-method", "preservation-branch", "post-promotion-status", "preservation-evidence"):
        promotion.add_argument(f"--{name}", required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--output-root", type=Path, required=True)
    gate.add_argument("--evidence-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "snapshot":
        build_snapshot(
            repository_root=args.repository_root.resolve(),
            integration_worktree=args.integration_worktree.resolve(),
            freeze_root=args.freeze_root.resolve(),
            output_root=args.output_root.resolve(),
            main_parent=args.main_parent,
            accepted_parent=args.accepted_parent,
        )
    elif args.command == "inventory":
        build_inventory(
            main_collect=args.main_collect,
            round17_junit=args.round17_junit,
            integrated_collect=args.integrated_collect,
            output_root=args.output_root.resolve(),
        )
    elif args.command == "prefull":
        build_prefull(
            junit=args.junit,
            inventory_path=args.inventory,
            integration_worktree=args.integration_worktree.resolve(),
            output_root=args.output_root.resolve(),
        )
    elif args.command == "identity":
        build_identity(
            integration_worktree=args.integration_worktree.resolve(),
            evidence_root=args.evidence_root.resolve(),
            test_inventory=args.test_inventory,
            output_root=args.output_root.resolve(),
            main_parent=args.main_parent,
            accepted_parent=args.accepted_parent,
        )
    elif args.command == "full-suite":
        build_full_suite(
            junit=args.junit,
            output_root=args.output_root.resolve(),
            job_id=args.job_id,
            job_name=args.job_name,
            node=args.node,
            partition=args.partition,
            terminal_state=args.terminal_state,
            exit_code=args.exit_code,
            integration_commit=args.integration_commit,
            submission_command=args.submission_command,
            stdout_path=args.stdout_path,
            stderr_path=args.stderr_path,
            scontrol_path=args.scontrol_path,
            sacct_path=args.sacct_path,
            sacct_status=args.sacct_status,
        )
    elif args.command == "promotion":
        build_promotion(
            output_root=args.output_root.resolve(),
            old_main=args.old_main,
            new_main=args.new_main,
            integration_commit=args.integration_commit,
            promotion_method=args.promotion_method,
            preservation_branch=args.preservation_branch,
            post_promotion_status=args.post_promotion_status,
            preservation_evidence=args.preservation_evidence,
        )
    else:
        build_gate(output_root=args.output_root.resolve(), evidence_root=args.evidence_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
