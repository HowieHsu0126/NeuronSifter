"""Record a source-bound Round-24 full pytest execution from JUnit XML."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_FILES = (
    "Libs/configs/config_stack.yaml",
    "Libs/configs/neuronsifter/vnext_research_contract.yaml",
    "Libs/configs/neuronsifter/round23_mechanism_pool.yaml",
    "Libs/configs/schema/neuronsifter_vnext_research_contract.schema.json",
    "Libs/configs/validation/iclr_design_contract_union.yaml",
    "Libs/models/modules/decision_consistent_runtime.py",
    "Libs/exp/training/train_foundation_state_c1.py",
    "Libs/exp/training/train_restricted_c2_vnext.py",
    "Libs/scripts/experiments/run_round23_shared_belief_replay.py",
    "Libs/scripts/experiments/run_round24_tool_dispatch.py",
    "Libs/scripts/experiments/run_round24_prospective_adaptation.py",
    "Libs/scripts/builders/build_round24_development_freeze.py",
    "Libs/scripts/builders/build_round23_vnext_artifacts.py",
    "Libs/scripts/validators/validate_round24_adaptation.py",
    "Libs/scripts/validators/record_round24_full_suite.py",
    "Libs/scripts/validators/finalize_round24_development_gate.py",
    "Libs/exp/evaluation/run_round24_adaptation.sbatch",
    "Libs/exp/evaluation/run_round24_validation.sbatch",
    "Libs/tests/test_round24_adaptation_contracts.py",
    "Libs/tests/test_round23_evidence_integrity.py",
    "Libs/tests/test_iclr_design_contract_union.py",
    "Libs/tests/test_publication_readiness_contracts.py",
    "Docs/design/iclr/ROUND24_REPRODUCTION_AND_ADAPTATION_HANDOFF.md",
    "Output/audits/round24_round23_source_successor_binding.json",
)

CHECKPOINT_FILES = (
    "Output/runs/round23/c1/large_seed3407/last.pt",
    "Output/runs/round23/c2s/large_seed3407/last.pt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _git_value(*arguments: str) -> str | None:
    process = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 and process.stdout.strip() else None


def parse_junit(junit_path: Path) -> tuple[dict[str, int | float], list[dict[str, Any]]]:
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root)
    counts: dict[str, int | float] = {
        "tests_collected": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "tests_failed": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "tests_errored": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "tests_skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "elapsed_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
    }
    counts["tests_passed"] = int(counts["tests_collected"]) - int(counts["tests_failed"]) - int(counts["tests_errored"]) - int(counts["tests_skipped"])
    inventory = [
        {
            "classname": case.attrib.get("classname", ""),
            "name": case.attrib.get("name", ""),
            "file": case.attrib.get("file", ""),
            "line": case.attrib.get("line", ""),
            "time_seconds": float(case.attrib.get("time", 0.0)),
            "failed": case.find("failure") is not None,
            "errored": case.find("error") is not None,
            "skipped": case.find("skipped") is not None,
        }
        for suite in suites
        for case in suite.iter("testcase")
    ]
    if len(inventory) != counts["tests_collected"]:
        raise ValueError("Round-24 JUnit inventory count does not match suite totals")
    return counts, inventory


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _terminal_accounting(job_id: str, *, expected_state: str) -> dict[str, Any]:
    process = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "-X",
            "--parsable2",
            "--noheader",
            "--format=JobID,JobName,Partition,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocCPUS,ReqMem,NodeList,WorkDir,SubmitLine%300",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"sacct failed for Round-24 validation job {job_id}: {process.stderr.strip()}")
    rows = [line.split("|") for line in process.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 14:
        raise ValueError(f"unexpected Round-24 sacct row for job {job_id}")
    fields = (
        "job_id",
        "job_name",
        "partition",
        "state",
        "exit_code",
        "submit",
        "start",
        "end",
        "elapsed_seconds",
        "allocated_cpus",
        "requested_memory",
        "node_list",
        "work_dir",
        "submit_line",
    )
    row = dict(zip(fields, rows[0], strict=True))
    if row["job_id"] != job_id or row["state"] != expected_state:
        raise ValueError(f"Round-24 validation job has unexpected terminal state: {row}")
    if expected_state == "COMPLETED" and row["exit_code"] != "0:0":
        raise ValueError(f"Round-24 completed validation job has a nonzero exit code: {row}")
    if expected_state == "FAILED" and row["exit_code"] == "0:0":
        raise ValueError(f"Round-24 failed validation job has a zero exit code: {row}")
    row["elapsed_seconds"] = int(row["elapsed_seconds"])
    row["allocated_cpus"] = int(row["allocated_cpus"])
    return {
        "schema_version": "neuronsifter-round24-slurm-terminal-accounting-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal_completed" if expected_state == "COMPLETED" else "terminal_failed",
        "job": row,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }


def _source_drift(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = payload.get("source_files")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("Round-24 full-suite audit has no source manifest")
    drift = []
    for row in source_rows:
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("Round-24 full-suite source identity has no path")
        path = REPO_ROOT / relative
        if not path.is_file():
            drift.append({"path": relative, "reason": "source_missing"})
            continue
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        if actual_hash != row.get("sha256") or actual_bytes != int(row.get("bytes", -1)):
            drift.append(
                {
                    "path": relative,
                    "reason": "source_identity_changed",
                    "recorded_sha256": row.get("sha256"),
                    "current_sha256": actual_hash,
                    "recorded_bytes": row.get("bytes"),
                    "current_bytes": actual_bytes,
                }
            )
    return drift


def archive_source_invalidated_execution_if_present() -> dict[str, Any]:
    current = REPO_ROOT / "Output/audits/round24_full_suite_execution.json"
    if not current.is_file():
        return {"status": "no_current_execution_audit"}
    payload = json.loads(current.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Round-24 current full-suite audit is not a JSON object")
    if payload.get("status") != "passed" or payload.get("full_suite_green") is not True:
        raise ValueError("Refusing to source-invalidate a non-green Round-24 full-suite audit")
    drift = _source_drift(payload)
    if not drift:
        raise ValueError("Refusing to archive a source-current green Round-24 full-suite audit")
    job_id = str(payload.get("slurm", {}).get("job_id", ""))
    if not job_id.isdigit():
        raise ValueError("Round-24 green full-suite audit has no valid Slurm job id")
    archive_root = REPO_ROOT / f"Output/audits/round24_full_suite_runs/job_{job_id}"
    if not archive_root.is_dir():
        raise ValueError("Round-24 green full-suite archive is missing")
    target = archive_root / "round24_full_suite_execution.json"
    accounting_path = archive_root / "slurm_terminal_accounting.json"
    if target.exists() or accounting_path.exists():
        raise FileExistsError(f"Round-24 green execution is already archived: {archive_root}")
    _write_json(accounting_path, _terminal_accounting(job_id, expected_state="COMPLETED"))
    current.replace(target)
    payload["slurm"]["terminal_accounting_status"] = "terminal_completed"
    payload["slurm"]["terminal_accounting"] = _identity(accounting_path)
    payload["candidate_binding"] = {
        "status": "invalidated_by_post_execution_source_change",
        "invalidated_at": datetime.now(timezone.utc).isoformat(),
        "execution_result_preserved": "passed",
        "drifted_source_count": len(drift),
        "drifted_sources": drift,
    }
    _write_json(target, payload)
    return {
        "status": "green_execution_archived_after_source_change",
        "job_id": job_id,
        "drifted_source_count": len(drift),
        "execution_audit": _identity(target),
        "terminal_accounting": _identity(accounting_path),
    }


def archive_noncurrent_execution_if_present() -> dict[str, Any]:
    current = REPO_ROOT / "Output/audits/round24_full_suite_execution.json"
    if not current.is_file():
        return {"status": "no_current_execution_audit"}
    payload = json.loads(current.read_text(encoding="utf-8"))
    if payload.get("status") == "failed" and payload.get("full_suite_green") is False:
        return archive_failed_execution_if_present()
    if payload.get("status") == "passed" and payload.get("full_suite_green") is True:
        return archive_source_invalidated_execution_if_present()
    raise ValueError("Round-24 current execution audit has no archivable state")


def archive_stale_terminal_accounting(*, job_id: str) -> dict[str, Any]:
    """Preserve a predecessor readback after its full-suite audit was archived.

    A fresh candidate needs the top-level accounting path. This only moves a
    completed readback after the corresponding full-suite execution has already
    been preserved under its job archive.
    """

    if not job_id.isdigit():
        raise ValueError("Round-24 stale terminal-accounting archive needs a numeric job id")
    current_execution = REPO_ROOT / "Output/audits/round24_full_suite_execution.json"
    if current_execution.exists():
        raise ValueError("cannot archive terminal accounting while a current full-suite audit exists")
    source = REPO_ROOT / "Output/audits/round24_full_suite_slurm_accounting.json"
    if not source.is_file():
        return {"status": "no_stale_terminal_accounting"}
    payload = json.loads(source.read_text(encoding="utf-8"))
    job = payload.get("job", {}) if isinstance(payload, dict) else {}
    if (
        payload.get("schema_version") != "neuronsifter-round24-slurm-terminal-accounting-v1"
        or payload.get("status") != "terminal_completed"
        or str(job.get("job_id", "")) != job_id
        or job.get("state") != "COMPLETED"
        or job.get("exit_code") != "0:0"
    ):
        raise ValueError("Round-24 stale terminal accounting is not a completed readback for its job")
    archive_root = REPO_ROOT / f"Output/audits/round24_full_suite_runs/job_{job_id}"
    execution = archive_root / "round24_full_suite_execution.json"
    if not execution.is_file():
        raise ValueError("Round-24 predecessor full-suite execution archive is missing")
    target = archive_root / "slurm_terminal_accounting_pre_source_change.json"
    manifest = archive_root / "stale_terminal_accounting_preservation.json"
    if target.exists() or manifest.exists():
        raise FileExistsError(f"Round-24 stale terminal accounting is already archived: {archive_root}")
    source.replace(target)
    preservation = {
        "schema_version": "neuronsifter-round24-stale-terminal-accounting-preservation-v1",
        "status": "preserved_predecessor_readback",
        "reason": "top_level_path_released_after_source_invalidated_full_suite_candidate",
        "predecessor_full_suite_job_id": job_id,
        "predecessor_full_suite_execution": _identity(execution),
        "terminal_accounting": _identity(target),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }
    _write_json(manifest, preservation)
    return {
        "status": "stale_terminal_accounting_archived",
        "job_id": job_id,
        "terminal_accounting": _identity(target),
        "preservation": _identity(manifest),
    }


def record_terminal_accounting(job_id: str) -> dict[str, Any]:
    current = REPO_ROOT / "Output/audits/round24_full_suite_execution.json"
    if not current.is_file():
        raise FileNotFoundError(current)
    payload = json.loads(current.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "neuronsifter-round24-full-suite-execution-v1"
        or payload.get("status") != "passed"
        or payload.get("full_suite_green") is not True
        or str(payload.get("slurm", {}).get("job_id", "")) != job_id
    ):
        raise ValueError("Round-24 current full-suite audit does not match the accounting job")
    drift = _source_drift(payload)
    if drift:
        raise ValueError(f"Round-24 full-suite source binding drifted before accounting readback: {drift}")
    accounting = _terminal_accounting(job_id, expected_state="COMPLETED")
    job = accounting["job"]
    if (
        job.get("job_name") != payload.get("slurm", {}).get("job_name")
        or job.get("node_list") != payload.get("slurm", {}).get("node")
        or not str(job.get("submit_line", "")).endswith(" full")
    ):
        raise ValueError("Round-24 terminal accounting does not match the recorded full-suite execution")
    accounting_path = REPO_ROOT / "Output/audits/round24_full_suite_slurm_accounting.json"
    if accounting_path.exists():
        raise FileExistsError(f"Round-24 terminal accounting already exists: {accounting_path}")
    _write_json(accounting_path, accounting)
    payload["slurm"]["terminal_accounting_status"] = "terminal_completed"
    payload["slurm"]["terminal_accounting"] = _identity(accounting_path)
    _write_json(current, payload)
    return {
        "status": "terminal_accounting_recorded",
        "job_id": job_id,
        "terminal_accounting": _identity(accounting_path),
        "full_suite_execution": _identity(current),
    }


def _capture_batch_script(
    job_id: str,
    destination: Path,
    *,
    parent_full_suite_job_id: str | None,
    allow_current_repository_snapshot: bool,
) -> dict[str, Any]:
    process = subprocess.run(
        ["scontrol", "write", "batch_script", job_id, str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode == 0 and destination.is_file():
        return {"method": "scontrol_write_batch_script", "source_job_id": job_id}
    if parent_full_suite_job_id and parent_full_suite_job_id.isdigit():
        parent_script = (
            REPO_ROOT
            / f"Output/audits/round24_full_suite_runs/job_{parent_full_suite_job_id}/run_round24_validation.sbatch"
        )
        if parent_script.is_file():
            shutil.copy2(parent_script, destination)
            return {
                "method": "parent_full_suite_archived_submission_script",
                "source_job_id": parent_full_suite_job_id,
                "reason": "Slurm controller no longer retained the terminal child job script",
                "scontrol_error": process.stderr.strip(),
            }
    if allow_current_repository_snapshot:
        repository_script = REPO_ROOT / "Libs/exp/evaluation/run_round24_validation.sbatch"
        if repository_script.is_file():
            shutil.copy2(repository_script, destination)
            return {
                "method": "current_repository_script_after_terminal_failure",
                "source_job_id": None,
                "exact_scheduler_snapshot": False,
                "reason": "Slurm controller no longer retained the terminal job script",
                "scontrol_error": process.stderr.strip(),
            }
    raise RuntimeError(f"scontrol could not preserve batch script for job {job_id}: {process.stderr.strip()}")


def archive_failed_validation_job(
    *, job_id: str, stage: str, parent_full_suite_job_id: str | None
) -> dict[str, Any]:
    if stage not in {"aggregate", "focused", "finalize", "final-validator"}:
        raise ValueError(f"unsupported Round-24 failed validation stage: {stage}")
    accounting = _terminal_accounting(job_id, expected_state="FAILED")
    submit_line = str(accounting["job"].get("submit_line", ""))
    if stage == "finalize" and (
        not parent_full_suite_job_id
        or not submit_line.endswith(f" finalize {parent_full_suite_job_id}")
    ):
        raise ValueError("Round-24 failed finalizer accounting does not bind its parent full-suite job")
    if stage != "finalize" and not submit_line.endswith(f" {stage}"):
        raise ValueError("Round-24 failed validation accounting does not bind its declared stage")
    job_name = str(accounting["job"]["job_name"])
    stdout = REPO_ROOT / f"Output/logs/{job_name}.{job_id}.out"
    stderr = REPO_ROOT / f"Output/logs/{job_name}.{job_id}.err"
    for path in (stdout, stderr):
        if not path.is_file():
            raise FileNotFoundError(path)
    archive_root = REPO_ROOT / f"Output/audits/round24_validation_runs/job_{job_id}"
    if archive_root.exists():
        raise FileExistsError(f"Round-24 validation failure is already archived: {archive_root}")
    with tempfile.TemporaryDirectory(prefix=f"nrnsift-round24-job-{job_id}.") as temporary:
        captured_script = Path(temporary) / "submitted_batch_script.sh"
        script_capture = _capture_batch_script(
            job_id,
            captured_script,
            parent_full_suite_job_id=parent_full_suite_job_id,
            allow_current_repository_snapshot=stage in {"aggregate", "focused", "final-validator"},
        )
        archive_root.mkdir(parents=True, exist_ok=False)
        archived_stdout = archive_root / "stdout.log"
        archived_stderr = archive_root / "stderr.log"
        archived_script = archive_root / "submitted_batch_script.sh"
        shutil.copy2(stdout, archived_stdout)
        shutil.copy2(stderr, archived_stderr)
        shutil.copy2(captured_script, archived_script)
    accounting_path = archive_root / "slurm_terminal_accounting.json"
    _write_json(accounting_path, accounting)
    payload = {
        "schema_version": "neuronsifter-round24-validation-failure-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "stage": stage,
        "job_id": job_id,
        "parent_full_suite_job_id": parent_full_suite_job_id,
        "preserved_as_failure_evidence": True,
        "stdout": _identity(archived_stdout),
        "stderr": _identity(archived_stderr),
        "submitted_batch_script": _identity(archived_script),
        "submitted_batch_script_capture": script_capture,
        "terminal_accounting": _identity(accounting_path),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }
    execution_path = archive_root / "validation_execution.json"
    _write_json(execution_path, payload)
    return {
        "status": "failed_validation_job_archived",
        "job_id": job_id,
        "stage": stage,
        "validation_execution": _identity(execution_path),
    }


def archive_failed_execution_if_present() -> dict[str, Any]:
    current = REPO_ROOT / "Output/audits/round24_full_suite_execution.json"
    if not current.is_file():
        return {"status": "no_current_execution_audit"}
    payload = json.loads(current.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Round-24 current full-suite audit is not a JSON object")
    if payload.get("status") != "failed" or payload.get("full_suite_green") is not False:
        raise ValueError("Refusing to archive a non-failed Round-24 full-suite audit")
    job_id = str(payload.get("slurm", {}).get("job_id", ""))
    if not job_id.isdigit():
        raise ValueError("Round-24 failed full-suite audit has no valid Slurm job id")
    archive_root = (
        REPO_ROOT / f"Output/audits/round24_full_suite_runs/job_{job_id}"
    )
    accounting_path = archive_root / "slurm_terminal_accounting.json"
    if not archive_root.is_dir() or not accounting_path.is_file():
        raise ValueError("Round-24 failed full-suite archive lacks terminal accounting")
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    if (
        accounting.get("status") != "terminal_failed"
        or str(accounting.get("job", {}).get("job_id", "")) != job_id
        or accounting.get("job", {}).get("state") != "FAILED"
    ):
        raise ValueError("Round-24 failed full-suite terminal accounting is invalid")
    target = archive_root / "round24_full_suite_execution.json"
    if target.exists():
        raise FileExistsError(f"Round-24 failed execution audit is already archived: {target}")
    payload["slurm"]["terminal_accounting_status"] = "terminal_failed"
    payload["slurm"]["terminal_accounting"] = _identity(accounting_path)
    _write_json(target, payload)
    current.unlink()
    return {
        "status": "failed_execution_archived",
        "job_id": job_id,
        "execution_audit": _identity(target),
        "terminal_accounting": _identity(accounting_path),
    }


def record(
    *,
    junit: Path,
    pytest_exit_code: int,
    stdout: Path,
    stderr: Path,
    archive_root: Path,
) -> dict[str, Any]:
    junit = junit.resolve()
    stdout = stdout.resolve()
    stderr = stderr.resolve()
    archive_root = (archive_root if archive_root.is_absolute() else REPO_ROOT / archive_root).resolve()
    for path in (junit, stdout, stderr):
        if not path.is_file():
            raise FileNotFoundError(path)
    if archive_root.exists():
        raise FileExistsError(f"Round-24 full-suite archive already exists: {archive_root}")

    source_rows = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        source_rows.append(_identity(path))
    counts, inventory = parse_junit(junit)
    green = bool(
        pytest_exit_code == 0
        and counts["tests_collected"] > 0
        and counts["tests_failed"] == 0
        and counts["tests_errored"] == 0
        and counts["tests_skipped"] == 0
    )

    archive_root.mkdir(parents=True, exist_ok=False)
    archived_junit = archive_root / "pytest_junit.xml"
    archived_stdout = archive_root / "pytest.stdout.log"
    archived_stderr = archive_root / "pytest.stderr.log"
    archived_sbatch = archive_root / "run_round24_validation.sbatch"
    shutil.copy2(junit, archived_junit)
    shutil.copy2(stdout, archived_stdout)
    shutil.copy2(stderr, archived_stderr)
    shutil.copy2(REPO_ROOT / "Libs/exp/evaluation/run_round24_validation.sbatch", archived_sbatch)

    dependencies = {
        "schema_version": "neuronsifter-round24-dependency-manifest-v1",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": sorted(
            (
                {"name": distribution.metadata.get("Name", distribution.name), "version": distribution.version}
                for distribution in metadata.distributions()
            ),
            key=lambda row: (str(row["name"]).lower(), str(row["version"])),
        ),
    }
    dependency_path = archive_root / "dependency_manifest.json"
    inventory_path = archive_root / "test_inventory.json"
    _write_json(dependency_path, dependencies)
    _write_json(
        inventory_path,
        {
            "schema_version": "neuronsifter-round24-test-inventory-v1",
            "test_count": len(inventory),
            "tests": inventory,
        },
    )

    source_manifest_digest = hashlib.sha256(
        json.dumps(source_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": "neuronsifter-round24-full-suite-execution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if green else "failed",
        "full_suite_green": green,
        "command": "python -m pytest Libs/tests -q -p no:cacheprovider --junitxml=<node-local-path>",
        "pytest_exit_code": int(pytest_exit_code),
        **counts,
        "junit_path": _identity(archived_junit)["path"],
        "junit_sha256": _sha256(archived_junit),
        "stdout": _identity(archived_stdout),
        "stderr": _identity(archived_stderr),
        "dependency_manifest": _identity(dependency_path),
        "test_inventory": _identity(inventory_path),
        "source_files": source_rows,
        "source_file_count": len(source_rows),
        "candidate_source_manifest_sha256": source_manifest_digest,
        "git": {
            "base_commit": _git_value("rev-parse", "HEAD"),
            "base_tree": _git_value("rev-parse", "HEAD^{tree}"),
            "candidate_commit_available": False,
            "candidate_tree_identity": f"source-manifest-sha256:{source_manifest_digest}",
            "reason": "Round-24 changes are an uncommitted working-tree candidate",
        },
        "checkpoints": [_identity(REPO_ROOT / relative) for relative in CHECKPOINT_FILES],
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "submission_script": _identity(archived_sbatch),
            "terminal_accounting_status": "pending_separate_terminal_readback",
        },
        "node_local_staging_used": str(junit).startswith("/tmp/"),
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    _write_json(REPO_ROOT / "Output/audits/round24_full_suite_execution.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-failed-execution-if-present", action="store_true")
    parser.add_argument("--archive-noncurrent-execution-if-present", action="store_true")
    parser.add_argument("--archive-stale-terminal-accounting-job-id")
    parser.add_argument("--record-terminal-accounting-job-id")
    parser.add_argument("--archive-failed-validation-job-id")
    parser.add_argument(
        "--failed-validation-stage",
        choices=("aggregate", "focused", "finalize", "final-validator"),
    )
    parser.add_argument("--parent-full-suite-job-id")
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--pytest-exit-code", type=int)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--archive-root", type=Path)
    arguments = parser.parse_args()
    if arguments.archive_failed_execution_if_present:
        print(json.dumps(archive_failed_execution_if_present(), sort_keys=True))
        return 0
    if arguments.archive_noncurrent_execution_if_present:
        print(json.dumps(archive_noncurrent_execution_if_present(), sort_keys=True))
        return 0
    if arguments.archive_stale_terminal_accounting_job_id:
        print(
            json.dumps(
                archive_stale_terminal_accounting(
                    job_id=arguments.archive_stale_terminal_accounting_job_id
                ),
                sort_keys=True,
            )
        )
        return 0
    if arguments.record_terminal_accounting_job_id:
        print(json.dumps(record_terminal_accounting(arguments.record_terminal_accounting_job_id), sort_keys=True))
        return 0
    if arguments.archive_failed_validation_job_id:
        if not arguments.failed_validation_stage:
            parser.error("--failed-validation-stage is required with --archive-failed-validation-job-id")
        print(
            json.dumps(
                archive_failed_validation_job(
                    job_id=arguments.archive_failed_validation_job_id,
                    stage=arguments.failed_validation_stage,
                    parent_full_suite_job_id=arguments.parent_full_suite_job_id,
                ),
                sort_keys=True,
            )
        )
        return 0
    required = {
        "--junit": arguments.junit,
        "--pytest-exit-code": arguments.pytest_exit_code,
        "--stdout": arguments.stdout,
        "--stderr": arguments.stderr,
        "--archive-root": arguments.archive_root,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error("missing required recording arguments: " + ", ".join(missing))
    payload = record(
        junit=arguments.junit,
        pytest_exit_code=arguments.pytest_exit_code,
        stdout=arguments.stdout,
        stderr=arguments.stderr,
        archive_root=arguments.archive_root,
    )
    print(json.dumps({"status": payload["status"], "tests_collected": payload["tests_collected"], "tests_skipped": payload["tests_skipped"]}, sort_keys=True))
    return 0 if payload["full_suite_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
