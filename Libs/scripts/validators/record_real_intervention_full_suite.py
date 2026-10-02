"""Record one source-bound full test execution for real-intervention closure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = REPO_ROOT / "Output/audits/round27_full_suite_execution.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_junit(path: Path) -> dict[str, int | float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root)
    counts: dict[str, int | float] = {
        "tests_collected": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "tests_failed": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "tests_errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "tests_skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "elapsed_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
    }
    counts["tests_errored"] = counts["tests_errors"]
    counts["tests_passed"] = (
        int(counts["tests_collected"])
        - int(counts["tests_failed"])
        - int(counts["tests_errors"])
        - int(counts["tests_skipped"])
    )
    return counts


def record(
    *,
    junit: Path,
    stdout: Path,
    stderr: Path,
    pytest_exit_code: int,
    slurm_job_id: str,
    slurm_job_name: str,
    slurm_state: str,
    slurm_exit_code: str,
    validation_host: str,
    archive_root: Path,
) -> dict[str, object]:
    if AUDIT_PATH.exists():
        raise FileExistsError(f"Round-27 full-suite audit already exists: {AUDIT_PATH}")
    for path in (junit, stdout, stderr):
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = parse_junit(junit)
    green = bool(
        pytest_exit_code == 0
        and counts["tests_collected"] > 0
        and counts["tests_failed"] == 0
        and counts["tests_errors"] == 0
        and counts["tests_skipped"] == 0
        and slurm_state == "COMPLETED"
        and slurm_exit_code == "0:0"
    )
    archive_root = archive_root.resolve()
    if archive_root.exists():
        raise FileExistsError(f"Full-suite archive already exists: {archive_root}")
    archive_root.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, source in (("pytest_junit.xml", junit), ("pytest.stdout.log", stdout), ("pytest.stderr.log", stderr)):
        target = archive_root / name
        shutil.copy2(source, target)
        archived[name] = identity(target)
    source_files = [
        "Libs/configs/config_stack.yaml",
        "Libs/configs/data/real_intervention_evidence.yaml",
        "Libs/scripts/builders/retrieve_real_intervention_source.py",
        "Libs/scripts/builders/materialize_real_intervention_timecourse.py",
        "Libs/scripts/builders/build_round23_vnext_artifacts.py",
        "Libs/scripts/experiments/resolve_real_intervention_world_action.py",
        "Libs/scripts/validators/validate_real_intervention_world_action.py",
        "Libs/scripts/validators/record_real_intervention_full_suite.py",
        "Libs/scripts/validators/record_real_intervention_source_successor_binding.py",
        "Libs/utils/evaluation/real_intervention_evidence.py",
        "Libs/tests/test_real_intervention_evidence.py",
    ]
    payload: dict[str, object] = {
        "schema_version": "neuronsifter-real-intervention-full-suite-execution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if green else "failed",
        "full_suite_green": green,
        "command": "python -m pytest Libs/tests -q --junitxml=<node-local-path>",
        "pytest_exit_code": int(pytest_exit_code),
        **counts,
        "junit": archived["pytest_junit.xml"],
        "stdout": archived["pytest.stdout.log"],
        "stderr": archived["pytest.stderr.log"],
        "source_files": [identity(REPO_ROOT / path) for path in source_files],
        "source_file_count": len(source_files),
        "validation_host": validation_host,
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": slurm_job_name,
        "slurm_state": slurm_state,
        "slurm_exit_code": slurm_exit_code,
        "terminal_slurm_accounting_bound": True,
        "node_local_staging_used": True,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--pytest-exit-code", type=int, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--slurm-job-name", required=True)
    parser.add_argument("--slurm-state", required=True)
    parser.add_argument("--slurm-exit-code", required=True)
    parser.add_argument("--validation-host", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    payload = record(
        junit=args.junit,
        stdout=args.stdout,
        stderr=args.stderr,
        pytest_exit_code=args.pytest_exit_code,
        slurm_job_id=args.slurm_job_id,
        slurm_job_name=args.slurm_job_name,
        slurm_state=args.slurm_state,
        slurm_exit_code=args.slurm_exit_code,
        validation_host=args.validation_host,
        archive_root=args.archive_root,
    )
    print(json.dumps({key: payload[key] for key in ("status", "tests_collected", "tests_failed", "tests_errors", "tests_skipped")}, sort_keys=True))
    return 0 if payload["full_suite_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
