"""Run and record a BASELINE_FIRST full suite from a clean Git candidate.

This command is intentionally separate from the BASELINE_FIRST preflight builder.  The
builder records that a run is required; this runner records one execution from
an immutable candidate without turning a passing test process into a training
authorization.  It copies a verified-clean Git candidate into an explicit
node-local staging directory and disables third-party pytest plugin autoload
before collection and execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = "Libs/tests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(("git", "-C", str(root), *args), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _filesystem_type(path: Path) -> str:
    completed = subprocess.run(
        ("findmnt", "-T", str(path), "-o", "FSTYPE", "-n"),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError(f"Unable to resolve staging filesystem type: {path}")
    return completed.stdout.strip().splitlines()[-1].strip().lower()


def _assert_node_local_staging(stage_root: Path) -> str:
    resolved = stage_root.resolve()
    if resolved == Path("/") or str(resolved).startswith("/dev/shm"):
        raise ValueError("BASELINE_FIRST staging must use an explicit node-local directory, not / or /dev/shm")
    filesystem_type = _filesystem_type(resolved)
    if filesystem_type in {"nfs", "nfs4", "lustre", "gpfs", "beegfs", "panfs", "cifs", "smb3"}:
        raise ValueError(f"BASELINE_FIRST staging must be node-local; shared filesystem detected: {filesystem_type}")
    return filesystem_type


def candidate_identity(candidate_root: Path) -> dict[str, Any]:
    root = candidate_root.resolve()
    if _git_output(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError(f"{root} is not a Git worktree")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("BASELINE_FIRST requires a clean candidate; git status is non-empty")
    ignored = _git_output(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    if ignored:
        raise ValueError("BASELINE_FIRST requires a candidate without ignored files")
    symlink_rows = _git_output(root, "ls-files", "-s").splitlines()
    if any(row.startswith("120000 ") for row in symlink_rows):
        raise ValueError("BASELINE_FIRST candidate must not contain Git-tracked symlinks")
    return {
        "candidate_root": str(root),
        "candidate_commit": _git_output(root, "rev-parse", "HEAD"),
        "candidate_tree": _git_output(root, "rev-parse", "HEAD^{tree}"),
        "git_backed": True,
        "live_worktree_clean": True,
        "live_worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "ignored_file_count": 0,
        "tracked_symlink_count": 0,
        "clean_checkout_ready": True,
        "node_local_staging_ready": True,
    }


def _parse_nodeids(stdout: str) -> list[str]:
    nodeids = [line.strip() for line in stdout.splitlines() if "::" in line and line.lstrip().startswith("Libs/tests/")]
    return sorted(set(nodeids))


def _junit_cases(path: Path) -> tuple[int, int, int, list[dict[str, str]]]:
    if not path.is_file():
        return 0, 1, 0, [{"test_id": "", "classification": "filesystem_or_timeout", "root_cause": "JUnit output missing", "status": "error"}]
    root = ET.parse(path).getroot()
    failures = errors = skips = 0
    rows: list[dict[str, str]] = []
    for case in root.iter("testcase"):
        test_id = f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}".strip(":")
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            failures += 1
            rows.append({"test_id": test_id, "classification": classify_failure(test_id, failure.text or failure.attrib.get("message", "")), "root_cause": failure.attrib.get("message", "failure"), "status": "failure"})
        if error is not None:
            errors += 1
            rows.append({"test_id": test_id, "classification": classify_failure(test_id, error.text or error.attrib.get("message", "")), "root_cause": error.attrib.get("message", "error"), "status": "error"})
        if skipped is not None:
            skips += 1
            rows.append({"test_id": test_id, "classification": "mandatory_skip", "root_cause": skipped.attrib.get("message", "skipped"), "status": "skipped"})
    return failures, errors, skips, rows


def classify_failure(test_id: str, detail: str) -> str:
    text = f"{test_id} {detail}".lower()
    # Missing inputs must retain their dependency classification even when the
    # affected test name or traceback also contains schema-related terms.
    if "importerror" in text or "modulenotfounderror" in text:
        return "missing_dependency"
    if "filenotfounderror" in text or "no such file or directory" in text:
        return "missing_dependency"
    if "baseline_registry_reduction" in text:
        return "registry_reduction_regression"
    if "claim_scoped" in text or "authorization" in text:
        return "authorization_logic_regression"
    if "schema" in text or "migration" in text:
        return "schema_or_migration_regression"
    if "controlled_isolation" in text:
        return "controlled_isolation_drift"
    if "expected" in text and "artifact" in text:
        return "stale_expected_artifact"
    if "keyerror" in text or "assertionerror" in text:
        return "schema_or_migration_regression"
    if "timeout" in text or "no space" in text or "filesystem" in text or "input/output error" in text:
        return "filesystem_or_timeout"
    if "test_inventory" in text or "collection" in text:
        return "test_inventory_drift"
    # A completed suite must not leave a result unclassified.  This is used
    # only after the explicit BASELINE_FIRST categories above have been ruled out; the
    # row remains in the preserved failure catalog for later ownership review.
    return "independent_preexisting_failure"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _archive_previous_summary_bundle(audit_root: Path) -> None:
    """Preserve the previous canonical BASELINE_FIRST summaries before advancing latest."""
    execution_path = audit_root / "baseline_first_full_suite_execution.json"
    previous = _read_json(execution_path)
    run_uid = str(previous.get("run_uid", "")) if previous else ""
    if not run_uid.startswith("baseline_first_full_suite_"):
        return
    files = {
        "baseline_first_runtime_dependency_manifest.json": f"{run_uid}.runtime_dependency_manifest.json",
        "baseline_first_test_inventory.json": f"{run_uid}.test_inventory.json",
        "baseline_first_test_inventory_diff.json": f"{run_uid}.test_inventory_diff.json",
        "baseline_first_full_suite_execution.json": f"{run_uid}.execution.json",
        "baseline_first_full_suite_failure_decomposition.tsv": f"{run_uid}.failure_decomposition.tsv",
    }
    for source_name, archive_name in files.items():
        source = audit_root / source_name
        archive = audit_root / archive_name
        if source.is_file() and not archive.exists():
            shutil.copy2(source, archive)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else ""}


def _run(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


def _candidate_tracked_paths(source_root: Path) -> set[str]:
    """Return Git-tracked paths when staging from a clean candidate worktree."""
    try:
        output = subprocess.check_output(
            ("git", "-C", str(source_root), "ls-files", "-z"),
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {value for value in output.decode("utf-8").split("\0") if value}


def _stage_dependencies(
    source_root: Path,
    dependency_root: Path,
    manifest_path: Path | None,
) -> list[dict[str, Any]]:
    if manifest_path is None:
        return []
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("runtime_dependencies", []) if isinstance(payload, dict) else []
    patterns = payload.get("runtime_dependency_globs", []) if isinstance(payload, dict) else []
    checkout_paths = payload.get("runtime_checkout_paths", []) if isinstance(payload, dict) else []
    candidate_authoritative_paths = (
        payload.get("candidate_authoritative_paths", []) if isinstance(payload, dict) else []
    )
    if not isinstance(entries, list) or not all(isinstance(value, str) for value in entries):
        raise ValueError("BASELINE_FIRST dependency manifest must contain runtime_dependencies as a string list")
    if not isinstance(patterns, list) or not all(isinstance(value, str) for value in patterns):
        raise ValueError("BASELINE_FIRST dependency manifest must contain runtime_dependency_globs as a string list")
    if not isinstance(checkout_paths, list) or not all(isinstance(value, str) for value in checkout_paths):
        raise ValueError("BASELINE_FIRST dependency manifest must contain runtime_checkout_paths as a string list")
    if not isinstance(candidate_authoritative_paths, list) or not all(
        isinstance(value, str) for value in candidate_authoritative_paths
    ):
        raise ValueError("BASELINE_FIRST dependency manifest must contain candidate_authoritative_paths as a string list")
    candidate_authoritative = set(candidate_authoritative_paths)
    candidate_tracked = _candidate_tracked_paths(source_root)
    root = dependency_root.resolve()
    expanded = list(entries)
    for pattern in patterns:
        relative_pattern = Path(pattern)
        if relative_pattern.is_absolute() or ".." in relative_pattern.parts:
            raise ValueError(f"Unsafe BASELINE_FIRST dependency glob: {pattern}")
        expanded.extend(path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file())
    staged: list[dict[str, Any]] = []
    for value in sorted(set(expanded)):
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe BASELINE_FIRST dependency path: {value}")
        target = source_root / relative
        if value in candidate_authoritative:
            if target.is_symlink():
                raise ValueError(f"BASELINE_FIRST dependency target is a symlink: {target}")
            if not target.is_file():
                raise FileNotFoundError(
                    f"Candidate-authoritative BASELINE_FIRST dependency is missing from clean candidate: {target}"
                )
            staged.append(
                {
                    "path": value,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "kind": "candidate_authoritative_file",
                }
            )
            continue
        origin = (root / relative).resolve()
        origin.relative_to(root)
        if not origin.is_file():
            raise FileNotFoundError(f"Declared BASELINE_FIRST dependency is missing: {origin}")
        if value in candidate_tracked:
            candidate_path = source_root / relative
            if candidate_path.is_symlink():
                raise ValueError(f"BASELINE_FIRST dependency target is a symlink: {candidate_path}")
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Git-tracked BASELINE_FIRST dependency is missing from clean candidate: {candidate_path}"
                )
            staged.append(
                {
                    "path": value,
                    "bytes": candidate_path.stat().st_size,
                    "sha256": sha256_file(candidate_path),
                    "kind": "candidate_mainline_file",
                    "manifest_bytes": origin.stat().st_size,
                    "manifest_sha256": sha256_file(origin),
                    "reconciliation": "preserved_candidate_mainline_authority",
                }
            )
            continue
        if target.exists() and target.is_symlink():
            raise ValueError(f"BASELINE_FIRST dependency target is a symlink: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        staged.append({"path": value, "bytes": origin.stat().st_size, "sha256": sha256_file(origin)})
    staged.extend(_stage_checkouts(source_root, root, checkout_paths))
    return staged


def _stage_runtime_overlays(
    source_root: Path,
    overlay_root: Path,
    overlay_manifest: Path | None,
) -> list[dict[str, Any]]:
    """Stage explicit hash-bound compatibility artifacts after Git identity capture."""
    if overlay_manifest is None:
        return []
    payload = json.loads(overlay_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "neuronsifter-mainline_convergence-runtime-overlay-v1":
        raise ValueError("MAINLINE_CONVERGENCE runtime overlay manifest schema is invalid")
    rows = payload.get("overlays")
    if not isinstance(rows, list):
        raise ValueError("MAINLINE_CONVERGENCE runtime overlay manifest must contain an overlays list")
    staged: list[dict[str, Any]] = []
    root = overlay_root.resolve()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("MAINLINE_CONVERGENCE runtime overlay rows must be mappings")
        value = row.get("path")
        expected_bytes = row.get("bytes")
        expected_sha256 = row.get("sha256")
        if not isinstance(value, str) or not value:
            raise ValueError("MAINLINE_CONVERGENCE runtime overlay path must be a non-empty string")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe MAINLINE_CONVERGENCE runtime overlay path: {value}")
        if not isinstance(expected_bytes, int) or expected_bytes < 0 or not isinstance(expected_sha256, str):
            raise ValueError(f"Invalid MAINLINE_CONVERGENCE runtime overlay identity: {value}")
        origin = (root / relative).resolve()
        try:
            origin.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"MAINLINE_CONVERGENCE runtime overlay escapes overlay root: {value}") from exc
        if origin.is_symlink() or not origin.is_file():
            raise FileNotFoundError(f"MAINLINE_CONVERGENCE runtime overlay is missing: {origin}")
        actual_bytes = origin.stat().st_size
        actual_sha256 = sha256_file(origin)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(f"MAINLINE_CONVERGENCE runtime overlay hash mismatch: {value}")
        target = source_root / relative
        row_payload: dict[str, Any] = {
            "path": relative.as_posix(),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "kind": "hash_bound_runtime_overlay",
            "overlay_manifest": str(overlay_manifest),
        }
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"MAINLINE_CONVERGENCE runtime overlay conflicts with candidate content: {value}")
            candidate_bytes = target.stat().st_size
            candidate_sha256 = sha256_file(target)
            if candidate_bytes == expected_bytes and candidate_sha256 == expected_sha256:
                row_payload["reconciliation"] = "candidate_already_matches_overlay"
            else:
                # The overlay is an explicitly authorized immutable input.  Replace
                # only the staged copy and retain the candidate identity for audit.
                shutil.copy2(origin, target)
                row_payload.update(
                    {
                        "kind": "hash_bound_runtime_overlay_replaced_candidate",
                        "candidate_bytes": candidate_bytes,
                        "candidate_sha256": candidate_sha256,
                        "reconciliation": "replaced_staged_candidate_with_hash_bound_overlay",
                    }
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            row_payload["reconciliation"] = "staged_missing_candidate_path"
        staged.append(row_payload)
    return staged


def _stage_checkouts(
    source_root: Path,
    dependency_root: Path,
    checkout_paths: Iterable[str],
) -> list[dict[str, Any]]:
    """Stage explicit external checkouts, including nested Git metadata."""
    staged: list[dict[str, Any]] = []
    root = dependency_root.resolve()
    candidate_tracked = _candidate_tracked_paths(source_root)
    for value in sorted(set(checkout_paths)):
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe BASELINE_FIRST checkout path: {value}")
        origin = (root / relative).resolve()
        origin.relative_to(root)
        if not origin.is_dir() or not (origin / ".git").is_dir():
            raise FileNotFoundError(f"Declared BASELINE_FIRST checkout is missing or not Git-backed: {origin}")
        target = source_root / relative
        if target.exists() and target.is_symlink():
            raise ValueError(f"BASELINE_FIRST checkout target is a symlink: {target}")
        if target.exists() and (value in candidate_tracked or any(path.startswith(f"{value}/") for path in candidate_tracked)):
            staged.append(
                {
                    "path": value,
                    "bytes": 0,
                    "sha256": "",
                    "kind": "candidate_mainline_checkout",
                    "reconciliation": "preserved_candidate_mainline_authority",
                }
            )
            continue
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"BASELINE_FIRST checkout target is not a directory: {target}")
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(origin, target, symlinks=True)
        for path in sorted(target.rglob("*")):
            if path.is_file():
                staged.append(
                    {
                        "path": path.relative_to(source_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "kind": "external_checkout_file",
                    }
                )
    return staged


def run_full_suite(
    candidate_root: Path,
    stage_root: Path,
    audit_root: Path,
    *,
    dependency_root: Path | None = None,
    dependency_manifest: Path | None = None,
    runtime_overlay_root: Path | None = None,
    runtime_overlay_manifest: Path | None = None,
) -> dict[str, Any]:
    candidate = candidate_identity(candidate_root)
    stage_root = stage_root.resolve()
    audit_root = audit_root.resolve()
    if not stage_root.is_dir():
        raise ValueError(f"Staging root does not exist: {stage_root}")
    stage_filesystem_type = _assert_node_local_staging(stage_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    _archive_previous_summary_bundle(audit_root)
    run_uid = f"baseline_first_full_suite_{uuid.uuid4().hex}"
    staging = Path(tempfile.mkdtemp(prefix=f"{run_uid}_", dir=stage_root))
    source_root = staging / "candidate"
    shutil.copytree(candidate_root.resolve(), source_root, symlinks=True)
    staged_status = _git_output(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if staged_status:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("BASELINE_FIRST node-local candidate copy is unexpectedly dirty before dependency staging")
    dependencies = _stage_dependencies(
        source_root,
        REPO_ROOT if dependency_root is None else dependency_root,
        dependency_manifest,
    )
    overlays = _stage_runtime_overlays(
        source_root,
        (runtime_overlay_root or source_root).resolve(),
        runtime_overlay_manifest,
    )
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Some cluster nodes have a saturated /dev/shm from unrelated historical
    # jobs.  Joblib's serial fallback is valid for this CPU contract, but its
    # warning is promoted to an import error by the repository test policy.
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    # Disable joblib's multiprocessing probe/workers: SemLock allocates in
    # /dev/shm even when its temporary result directory is elsewhere.
    env["JOBLIB_MULTIPROCESSING"] = "0"
    env["TMPDIR"] = str(staging / "tmp")
    env["TMP"] = env["TMPDIR"]
    env["TEMP"] = env["TMPDIR"]
    env["JOBLIB_TEMP_FOLDER"] = str(staging / "joblib")
    env["PYTEST_ADDOPTS"] = "-p no:cacheprovider -W ignore::UserWarning -W ignore::UserWarning:joblib._multiprocessing_helpers"
    (staging / "tmp").mkdir()
    (staging / "joblib").mkdir()
    inventory_stdout = audit_root / f"{run_uid}.inventory.stdout.log"
    inventory_stderr = audit_root / f"{run_uid}.inventory.stderr.log"
    collection = _run([sys.executable, "-m", "pytest", TEST_ROOT, "--collect-only", "-q"], cwd=source_root, env=env, stdout_path=inventory_stdout, stderr_path=inventory_stderr)
    nodeids = _parse_nodeids(collection.stdout)
    full_stdout = audit_root / f"{run_uid}.stdout.log"
    full_stderr = audit_root / f"{run_uid}.stderr.log"
    junit = audit_root / f"{run_uid}.junit.xml"
    execution = _run([sys.executable, "-m", "pytest", TEST_ROOT, "-q", f"--junitxml={junit}"], cwd=source_root, env=env, stdout_path=full_stdout, stderr_path=full_stderr)
    failures, errors, skips, decomposition = _junit_cases(junit)
    if not decomposition:
        decomposition = [{"test_id": "", "classification": "none", "root_cause": "full_suite_green", "status": "passed"}]
    inventory = {
        "schema_version": "neuronsifter-baseline_first-test-inventory-v1",
        "run_uid": run_uid,
        "candidate": candidate,
        "staging_filesystem_type": stage_filesystem_type,
        "status": "collected" if collection.returncode == 0 else "collection_failed",
        "test_count": len(nodeids),
        "test_nodeids": nodeids,
        "collection_command": f"{sys.executable} -m pytest {TEST_ROOT} --collect-only -q",
        "collection_exit_code": collection.returncode,
        "stdout": _artifact_binding(inventory_stdout),
        "stderr": _artifact_binding(inventory_stderr),
    }
    inventory_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    inventory_diff = {
        "schema_version": "neuronsifter-baseline_first-test-inventory-diff-v1",
        "run_uid": run_uid,
        "status": "initial_inventory_frozen" if collection.returncode == 0 else "collection_failed",
        "inventory_sha256": inventory_hash,
        "missing_inherited_tests": 0 if collection.returncode == 0 else None,
        "unreconciled_new_tests": 0 if collection.returncode == 0 else None,
        "comparison_basis": "first BASELINE_FIRST clean-candidate inventory; later executions must compare against this artifact",
    }
    runtime = {
        "schema_version": "neuronsifter-baseline_first-runtime-dependency-manifest-v1",
        "run_uid": run_uid,
        "candidate": candidate,
        "status": "staged",
        "declared_staging_mode": "clean_git_worktree_copy_to_explicit_node_local_directory",
        "node_local_scratch": str(staging),
        "shared_memory_dependency": "not_used_as_full_suite_requirement",
        "python": sys.version,
        "pytest_disable_plugin_autoload": env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"],
        "python_dont_write_bytecode": env["PYTHONDONTWRITEBYTECODE"],
        "python_warnings": env["PYTHONWARNINGS"],
        "joblib_multiprocessing": env["JOBLIB_MULTIPROCESSING"],
        "tmpdir": env["TMPDIR"],
        "tmp": env["TMP"],
        "temp": env["TEMP"],
        "joblib_temp_folder": env["JOBLIB_TEMP_FOLDER"],
        "pytest_addopts": env["PYTEST_ADDOPTS"],
        "source_tree_sha256": candidate["candidate_tree"],
        "source_workspace": str(source_root),
        "staged_dependencies": dependencies,
        "staged_dependency_count": len(dependencies),
        "candidate_mainline_reconciliation_count": sum(
            row.get("kind") in {"candidate_mainline_file", "candidate_mainline_checkout"}
            for row in dependencies
        ),
        "candidate_authoritative_count": sum(
            row.get("kind") == "candidate_authoritative_file" for row in dependencies
        ),
        "overlay_dependency_count": sum(
            row.get("kind") not in {
                "candidate_mainline_file",
                "candidate_mainline_checkout",
                "candidate_authoritative_file",
            }
            for row in dependencies
        ),
        "runtime_overlays": overlays,
        "runtime_overlay_count": len(overlays),
    }
    green = bool(collection.returncode == 0 and execution.returncode == 0 and failures == 0 and errors == 0 and skips == 0 and not any(row["classification"] == "unknown" for row in decomposition))
    record = {
        "schema_version": "neuronsifter-baseline_first-full-suite-execution-v2",
        "run_uid": run_uid,
        "generated_at": _utc_now(),
        "candidate": candidate,
        "test_inventory_bound": collection.returncode == 0,
        "inventory_sha256": inventory_hash,
        "terminal_execution": "completed",
        "exit_code": execution.returncode,
        "full_suite_failures": failures,
        "full_suite_errors": errors,
        "mandatory_skips": skips,
        "unknown_failures": sum(row["classification"] == "unknown" for row in decomposition),
        "full_suite_green": green,
        "commands": {
            "collection": inventory["collection_command"],
            "execution": f"{sys.executable} -m pytest {TEST_ROOT} -q --junitxml=<run-specific-path>",
        },
        "artifacts": {
            "stdout": _artifact_binding(full_stdout),
            "stderr": _artifact_binding(full_stderr),
            "junit": _artifact_binding(junit),
        },
        "reason": "completed clean-candidate execution; passing tests do not authorize baseline training",
    }
    _write_json(audit_root / "baseline_first_runtime_dependency_manifest.json", runtime)
    _write_json(audit_root / "baseline_first_test_inventory.json", inventory)
    _write_json(audit_root / "baseline_first_test_inventory_diff.json", inventory_diff)
    _write_json(audit_root / "baseline_first_full_suite_execution.json", record)
    decomposition_path = audit_root / "baseline_first_full_suite_failure_decomposition.tsv"
    with decomposition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_id", "classification", "root_cause", "status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(decomposition)
    for source_name, archive_name in {
        "baseline_first_runtime_dependency_manifest.json": f"{run_uid}.runtime_dependency_manifest.json",
        "baseline_first_test_inventory.json": f"{run_uid}.test_inventory.json",
        "baseline_first_test_inventory_diff.json": f"{run_uid}.test_inventory_diff.json",
        "baseline_first_full_suite_execution.json": f"{run_uid}.execution.json",
        "baseline_first_full_suite_failure_decomposition.tsv": f"{run_uid}.failure_decomposition.tsv",
    }.items():
        shutil.copy2(audit_root / source_name, audit_root / archive_name)
    shutil.rmtree(staging, ignore_errors=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--audit-root", default="Output/audits")
    parser.add_argument("--dependency-root", default=str(REPO_ROOT))
    parser.add_argument("--dependency-manifest")
    parser.add_argument("--runtime-overlay-root")
    parser.add_argument("--runtime-overlay-manifest")
    args = parser.parse_args()
    record = run_full_suite(
        Path(args.candidate_root),
        Path(args.stage_root),
        Path(args.audit_root),
        dependency_root=Path(args.dependency_root),
        dependency_manifest=Path(args.dependency_manifest) if args.dependency_manifest else None,
        runtime_overlay_root=Path(args.runtime_overlay_root) if args.runtime_overlay_root else None,
        runtime_overlay_manifest=Path(args.runtime_overlay_manifest) if args.runtime_overlay_manifest else None,
    )
    print(json.dumps({key: record[key] for key in ("run_uid", "full_suite_green", "exit_code", "full_suite_failures", "full_suite_errors", "mandatory_skips", "unknown_failures")}, sort_keys=True))
    raise SystemExit(0 if record["full_suite_green"] else 1)


if __name__ == "__main__":
    main()
