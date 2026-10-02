"""Record and terminally bind the static decision-loop full-suite execution."""

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
from typing import Any
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = REPO_ROOT / "Output/audits/round25_full_suite_execution.json"
ACCOUNTING_PATH = (
    REPO_ROOT / "Output/audits/round25_full_suite_slurm_accounting.json"
)
CONTRACT_PATH = REPO_ROOT / "Libs/configs/neuronsifter/static_decision_loop.yaml"

SOURCE_FILES = (
    "Libs/configs/config_stack.yaml",
    "Libs/configs/neuronsifter/static_decision_loop.yaml",
    "Libs/configs/neuronsifter/round24b_posterior_bound_evaluation.yaml",
    "Libs/models/modules/decision_consistent_runtime.py",
    "Libs/models/modules/round24b_posterior_adapter.py",
    "Libs/models/modules/round24b_typed_actions.py",
    "Libs/models/modules/static_information_policy.py",
    "Libs/scripts/builders/build_baseline_registry_audits.py",
    "Libs/scripts/experiments/run_round24_tool_dispatch.py",
    "Libs/scripts/experiments/run_static_decision_loop_campaign.py",
    "Libs/scripts/reports/finalize_static_decision_loop.py",
    "Libs/scripts/validators/validate_static_decision_loop.py",
    "Libs/scripts/validators/record_static_decision_loop_validation.py",
    "Libs/scripts/validators/finalize_static_decision_loop_handoff.py",
    "Libs/exp/evaluation/run_static_decision_loop_validation.sbatch",
    "Libs/tests/test_baseline_registry.py",
    "Libs/tests/test_static_decision_loop.py",
    "Docs/design/iclr/NEURONSIFTER_STATIC_DECISION_LOOP_REPRODUCTION.md",
)

CHECKPOINT_FILES = (
    "Output/runs/round23/c1/large_seed3407/last.pt",
    "Output/runs/round23/c2s/large_seed3407/last.pt",
)

EVALUATION_FILES = (
    "Input/manifests/round25/static_information_action_candidates.tsv",
    "Output/audits/round25_u0_canonical_static_twin.json",
    "Output/audits/round25_adaptation_negative_result_freeze.json",
    "Output/audits/round25_coverage_failure_taxonomy.tsv",
    "Output/audits/round25_static_support_domain.json",
    "Output/audits/round25_tri_validation_calibration.json",
    "Output/audits/round25_static_policy_freeze.json",
    "Output/audits/round25_internal_control_contract.json",
    "Output/audits/round25_prospective_round_manifest.tsv",
    "Output/audits/round25_posterior_assimilation_lineage.tsv",
    "Output/audits/round25_tool_result_catalog.tsv",
    "Output/paper_reports/round25_static_decision_loop.tsv",
    "Output/paper_reports/round25_information_action_effects.tsv",
    "Output/paper_reports/round25_belief_coherence.tsv",
    "Output/paper_reports/round25_static_failure_catalog.tsv",
    "Output/audits/round25_static_mechanism_gate.json",
    "Output/audits/round25_static_reliability_gate.json",
    "Output/audits/round25_static_loop_recovery_catalog.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_value(*arguments: str) -> str | None:
    process = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = process.stdout.strip()
    return value if process.returncode == 0 and value else None


def parse_junit(path: Path) -> tuple[dict[str, int | float], list[dict[str, Any]]]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root)
    counts: dict[str, int | float] = {
        "tests_collected": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "tests_failed": sum(
            int(suite.attrib.get("failures", 0)) for suite in suites
        ),
        "tests_errored": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "tests_skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "elapsed_seconds": sum(
            float(suite.attrib.get("time", 0.0)) for suite in suites
        ),
    }
    counts["tests_passed"] = (
        int(counts["tests_collected"])
        - int(counts["tests_failed"])
        - int(counts["tests_errored"])
        - int(counts["tests_skipped"])
    )
    inventory = []
    for suite in suites:
        for case in suite.iter("testcase"):
            inventory.append(
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
            )
    if len(inventory) != counts["tests_collected"]:
        raise ValueError("Static-loop JUnit inventory count differs from suite totals")
    return counts, inventory


def receipt_files() -> tuple[str, ...]:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    run_root = Path(contract["outputs"]["run_root"])
    return tuple(
        str(run_root / policy_uid / f"round_{round_index}" / "round_receipt.json")
        for policy_uid in contract["static_policy"]["controls"]
        for round_index in range(
            1, int(contract["static_policy"]["sequential_rounds"]) + 1
        )
    )


def record(
    *,
    junit: Path,
    pytest_exit_code: int,
    stdout: Path,
    stderr: Path,
    archive_root: Path,
) -> dict[str, Any]:
    if AUDIT_PATH.exists():
        raise FileExistsError(f"Static-loop full-suite audit already exists: {AUDIT_PATH}")
    archive_root = (
        archive_root if archive_root.is_absolute() else REPO_ROOT / archive_root
    ).resolve()
    if archive_root.exists():
        raise FileExistsError(
            f"Static-loop full-suite archive already exists: {archive_root}"
        )
    for path in (junit, stdout, stderr):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_rows = [identity(REPO_ROOT / relative) for relative in SOURCE_FILES]
    checkpoint_rows = [identity(REPO_ROOT / relative) for relative in CHECKPOINT_FILES]
    evaluation_rows = [
        identity(REPO_ROOT / relative)
        for relative in (*EVALUATION_FILES, *receipt_files())
    ]
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
    archived_script = archive_root / "run_static_decision_loop_validation.sbatch"
    shutil.copy2(junit, archived_junit)
    shutil.copy2(stdout, archived_stdout)
    shutil.copy2(stderr, archived_stderr)
    shutil.copy2(
        REPO_ROOT / "Libs/exp/evaluation/run_static_decision_loop_validation.sbatch",
        archived_script,
    )
    dependency_path = archive_root / "dependency_manifest.json"
    inventory_path = archive_root / "test_inventory.json"
    write_json(
        dependency_path,
        {
            "schema_version": "neuronsifter-static-loop-dependency-manifest-v1",
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": sorted(
                (
                    {
                        "name": distribution.metadata.get("Name", distribution.name),
                        "version": distribution.version,
                    }
                    for distribution in metadata.distributions()
                ),
                key=lambda row: (str(row["name"]).lower(), str(row["version"])),
            ),
        },
    )
    write_json(
        inventory_path,
        {
            "schema_version": "neuronsifter-static-loop-test-inventory-v1",
            "test_count": len(inventory),
            "tests": inventory,
        },
    )
    source_manifest_sha256 = hashlib.sha256(
        json.dumps(source_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload = {
        "schema_version": "neuronsifter-static-loop-full-suite-execution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if green else "failed",
        "full_suite_green": green,
        "command": (
            "python -m pytest Libs/tests -q -p no:cacheprovider "
            "--junitxml=<node-local-path>"
        ),
        "pytest_exit_code": int(pytest_exit_code),
        **counts,
        "junit": identity(archived_junit),
        "stdout": identity(archived_stdout),
        "stderr": identity(archived_stderr),
        "dependency_manifest": identity(dependency_path),
        "test_inventory": identity(inventory_path),
        "source_files": source_rows,
        "source_file_count": len(source_rows),
        "candidate_source_manifest_sha256": source_manifest_sha256,
        "checkpoints": checkpoint_rows,
        "evaluation_manifests": evaluation_rows,
        "evaluation_manifest_count": len(evaluation_rows),
        "git": {
            "base_commit": git_value("rev-parse", "HEAD"),
            "base_tree": git_value("rev-parse", "HEAD^{tree}"),
            "candidate_commit_available": False,
            "candidate_tree_identity": (
                f"source-manifest-sha256:{source_manifest_sha256}"
            ),
            "reason": "Round-25 is an uncommitted working-tree candidate",
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
            "submission_script": identity(archived_script),
            "terminal_accounting_status": "pending_separate_terminal_readback",
        },
        "node_local_staging_used": str(junit.resolve()).startswith("/tmp/"),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    write_json(AUDIT_PATH, payload)
    return payload


def terminal_accounting(job_id: str) -> dict[str, Any]:
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
        raise RuntimeError(
            f"sacct failed for static-loop full-suite job {job_id}: "
            f"{process.stderr.strip()}"
        )
    rows = [line.split("|") for line in process.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 14:
        raise ValueError(f"Unexpected static-loop sacct result for {job_id}: {rows}")
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
    if row["job_id"] != job_id or row["state"] != "COMPLETED" or row[
        "exit_code"
    ] != "0:0":
        raise ValueError(f"Static-loop full-suite job is not terminal green: {row}")
    row["elapsed_seconds"] = int(row["elapsed_seconds"])
    row["allocated_cpus"] = int(row["allocated_cpus"])
    return {
        "schema_version": "neuronsifter-static-loop-slurm-terminal-accounting-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal_completed",
        "job": row,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def record_terminal_accounting(job_id: str) -> dict[str, Any]:
    if ACCOUNTING_PATH.exists():
        raise FileExistsError(
            f"Static-loop terminal accounting already exists: {ACCOUNTING_PATH}"
        )
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "passed"
        or payload.get("full_suite_green") is not True
        or str(payload.get("slurm", {}).get("job_id", "")) != job_id
    ):
        raise ValueError(
            "Static-loop full-suite audit does not match terminal accounting request"
        )
    accounting = terminal_accounting(job_id)
    job = accounting["job"]
    if (
        job["job_name"] != payload["slurm"]["job_name"]
        or job["node_list"] != payload["slurm"]["node"]
        or job["job_name"] != "NrnSift_static_full_suite"
    ):
        raise ValueError(
            "Static-loop terminal accounting does not match the full-suite execution"
        )
    write_json(ACCOUNTING_PATH, accounting)
    payload["slurm"]["terminal_accounting_status"] = "terminal_completed"
    payload["slurm"]["terminal_accounting"] = identity(ACCOUNTING_PATH)
    write_json(AUDIT_PATH, payload)
    return accounting


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-terminal-accounting-job-id")
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--pytest-exit-code", type=int)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    if args.record_terminal_accounting_job_id:
        result = record_terminal_accounting(args.record_terminal_accounting_job_id)
        print(
            json.dumps(
                {"status": result["status"], "job_id": result["job"]["job_id"]},
                sort_keys=True,
            )
        )
        return 0
    required = (
        args.junit,
        args.pytest_exit_code,
        args.stdout,
        args.stderr,
        args.archive_root,
    )
    if any(value is None for value in required):
        parser.error(
            "JUnit, exit code, stdout, stderr, and archive root are required"
        )
    result = record(
        junit=args.junit,
        pytest_exit_code=args.pytest_exit_code,
        stdout=args.stdout,
        stderr=args.stderr,
        archive_root=args.archive_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "tests_collected": result["tests_collected"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["full_suite_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
