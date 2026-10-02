"""Record a source-bound Round-23 full pytest execution from JUnit XML."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


SOURCE_FILES = (
    "Libs/configs/config_stack.yaml",
    "Libs/configs/data/public_foundation_corpus_rescue.yaml",
    "Libs/configs/neuronsifter/vnext_research_contract.yaml",
    "Libs/configs/neuronsifter/round23_mechanism_pool.yaml",
    "Libs/configs/schema/neuronsifter_vnext_research_contract.schema.json",
    "Libs/configs/validation/iclr_design_contract_union.yaml",
    "Libs/configs/validation/mainline_runtime_dependencies.yaml",
    "Libs/models/modules/decision_consistent_runtime.py",
    "Libs/models/networks/foundation_state_encoder.py",
    "Libs/utils/data/iclr_contract_union.py",
    "Libs/exp/training/train_foundation_state_c1.py",
    "Libs/exp/training/train_restricted_c2_vnext.py",
    "Libs/scripts/experiments/run_round23_shared_belief_replay.py",
    "Libs/scripts/experiments/run_round23_prospective_mechanism_pool.py",
    "Libs/scripts/experiments/run_round23_offline_adaptation_replay.py",
    "Libs/scripts/reports/build_round23_development_diagnostics.py",
    "Libs/scripts/builders/build_c1_repository_inventory.py",
    "Libs/scripts/builders/build_c1_rescue_evidence.py",
    "Libs/scripts/builders/build_round20m_status.py",
    "Libs/scripts/builders/build_round23_vnext_artifacts.py",
    "Libs/scripts/validators/validate_c1_repository_inventory.py",
    "Libs/scripts/validators/validate_c1_rescue_evidence.py",
    "Libs/scripts/validators/validate_round23_vnext.py",
    "Libs/scripts/validators/validate_round23_checkpoints.py",
    "Libs/scripts/validators/record_round23_full_suite.py",
    "Libs/exp/evaluation/run_round23_shared_belief_replay.sbatch",
    "Libs/exp/evaluation/run_round23_prospective_mechanism_pool.sbatch",
    "Libs/exp/evaluation/run_round23_offline_adaptation_replay.sbatch",
    "Libs/exp/evaluation/run_round23_development_diagnostics.sbatch",
    "Libs/exp/evaluation/run_round23_vnext_validation.sbatch",
    "Libs/exp/training/run_round23_c1_development.sbatch",
    "Libs/exp/training/run_round23_c2s_development.sbatch",
    "Libs/tests/test_round23_decision_consistency.py",
    "Libs/tests/test_round23_vnext_contracts.py",
    "Libs/tests/test_round23_c2s_development.py",
    "Libs/tests/test_round23_shared_belief_replay.py",
    "Libs/tests/test_round23_offline_adaptation.py",
    "Libs/tests/test_round23_evidence_integrity.py",
    "Libs/tests/test_c1_repository_inventory.py",
    "Libs/tests/test_c1_rescue_evidence.py",
    "Libs/tests/test_mainline_full_suite_runner.py",
    "Libs/tests/test_round20m_status.py",
    "Docs/design/iclr/NEURONSIFTER_VNEXT_RESEARCH_FRAMEWORK.md",
    "Docs/design/iclr/ROUND23_REPRODUCTION_AND_PUBLICATION_HANDOFF.md",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_junit_counts(junit_path: Path) -> dict[str, int | float]:
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root)
    values = {
        "tests_collected": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "tests_failed": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "tests_errored": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "tests_skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "elapsed_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
    }
    values["tests_passed"] = int(values["tests_collected"]) - int(
        values["tests_failed"]
    ) - int(values["tests_errored"]) - int(values["tests_skipped"])
    return values


def record(*, junit: Path, pytest_exit_code: int, archive_root: Path) -> dict:
    junit = junit.resolve()
    archive_root = (
        archive_root if archive_root.is_absolute() else REPO_ROOT / archive_root
    ).resolve()
    if not junit.is_file():
        raise FileNotFoundError(junit)
    if archive_root.exists():
        raise FileExistsError(f"Round-23 full-suite archive already exists: {archive_root}")
    source_rows = []
    for relative_path in SOURCE_FILES:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        source_rows.append({"path": relative_path, "sha256": _sha256_file(path)})
    counts = parse_junit_counts(junit)
    green = bool(
        pytest_exit_code == 0
        and counts["tests_collected"] > 0
        and counts["tests_failed"] == 0
        and counts["tests_errored"] == 0
        and counts["tests_skipped"] == 0
    )
    archive_root.mkdir(parents=True, exist_ok=False)
    archived_junit = archive_root / "pytest_junit.xml"
    shutil.copy2(junit, archived_junit)
    relative_junit = archived_junit.relative_to(REPO_ROOT).as_posix()
    payload = {
        "schema_version": "neuronsifter-round23-full-suite-execution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if green else "failed",
        "full_suite_green": green,
        "command": "python -m pytest Libs/tests -q -p no:cacheprovider --junitxml=<node-local-path>",
        "pytest_exit_code": int(pytest_exit_code),
        **counts,
        "junit_path": relative_junit,
        "junit_sha256": _sha256_file(archived_junit),
        "source_files": source_rows,
        "source_file_count": len(source_rows),
        "validation_host": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "node_local_staging_used": str(junit).startswith("/tmp/"),
        "protected_test_access_count": 0,
        "publication_claim_eligible": False,
    }
    write_json(REPO_ROOT / "Output/audits/round23_full_suite_execution.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--pytest-exit-code", type=int, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    arguments = parser.parse_args()
    payload = record(
        junit=arguments.junit,
        pytest_exit_code=arguments.pytest_exit_code,
        archive_root=arguments.archive_root,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "tests_collected": payload["tests_collected"],
                "tests_failed": payload["tests_failed"],
                "tests_errored": payload["tests_errored"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["full_suite_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
