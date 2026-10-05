"""Run the full pretraining regression and emit terminal machine-readable evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.exp_utils import C1_RUNTIME_GIT_PATHS, get_git_provenance  # noqa: E402


OUTPUT_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _suite_names(classname: str) -> set[str]:
    """Assign every full-suite testcase to auditable C1 regression views."""

    name = classname.lower()
    suites = {"unit"}
    if any(token in name for token in ("corpus", "crosswalk", "adapter", "binding", "loader", "contract")):
        suites.add("data_contracts")
    if any(token in name for token in ("encoder", "screening", "twin", "uq", "dynamics", "solver")):
        suites.add("models")
    if any(token in name for token in ("experiment", "training", "preflight", "entrypoint", "active_learning", "regimen")):
        suites.add("experiments")
    if any(token in name for token in ("paper", "publication", "iclr", "g0_", "milestone")):
        suites.add("publication")
    if any(token in name for token in ("shell", "platform", "python_cli", "runtime", "tool")):
        suites.add("tools")
    return suites


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Libs/tests C1 pretraining regression.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-prefix", default="c1")
    args = parser.parse_args()
    if not OUTPUT_PREFIX_PATTERN.fullmatch(args.output_prefix):
        raise ValueError("C1 regression output prefix contains unsupported characters")
    run_uid = os.environ.get("NRNSIFTER_C1_RUN_UID")
    if not run_uid:
        raise ValueError("NRNSIFTER_C1_RUN_UID is required")
    audit_root = REPO_ROOT / "Output/audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"{args.output_prefix}_pretraining_regression"
    xml_path = audit_root / f"{artifact_stem}_{run_uid}.xml"
    stdout_path = audit_root / f"{artifact_stem}_{run_uid}.stdout.log"
    stderr_path = audit_root / f"{artifact_stem}_{run_uid}.stderr.log"
    config_path = (REPO_ROOT / args.config).resolve()
    config_path.relative_to(REPO_ROOT.resolve())
    config_before = load_config(config_path)
    config_file_hash_before = sha256_file(config_path)
    resolved_config_hash_before = canonical_sha256(config_before)
    contract_hash_before = canonical_sha256(config_before["foundation_state_c1"])
    git_provenance_before = get_git_provenance(C1_RUNTIME_GIT_PATHS)
    started_at = _utc_now()
    command = [sys.executable, "-m", "pytest", "Libs/tests", "-q", f"--junitxml={xml_path}"]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False)
    ended_at = _utc_now()
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    failed_rows: list[dict[str, str]] = []
    suite_counts: dict[str, dict[str, int]] = {
        suite: {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in ("unit", "data_contracts", "models", "experiments", "publication", "tools")
    }
    if xml_path.is_file():
        root = ET.parse(xml_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            for key in counts:
                counts[key] += int(float(suite.attrib.get(key, "0")))
            for case in suite.findall("testcase"):
                memberships = _suite_names(case.attrib.get("classname", ""))
                problem = case.find("failure")
                if problem is None:
                    problem = case.find("error")
                for membership in memberships:
                    suite_counts[membership]["tests"] += 1
                    if problem is not None:
                        suite_counts[membership]["failures" if problem.tag == "failure" else "errors"] += 1
                    if case.find("skipped") is not None:
                        suite_counts[membership]["skipped"] += 1
                if problem is not None:
                    failed_rows.append({
                        "classname": case.attrib.get("classname", ""),
                        "test_name": case.attrib.get("name", ""),
                        "failure_type": problem.tag,
                        "message": problem.attrib.get("message", ""),
                    })
    config_after = load_config(config_path)
    config_file_hash_after = sha256_file(config_path)
    resolved_config_hash_after = canonical_sha256(config_after)
    contract_hash_after = canonical_sha256(config_after["foundation_state_c1"])
    git_provenance_after = get_git_provenance(C1_RUNTIME_GIT_PATHS)
    corpus_hash = str(
        config_after["foundation_state_c1"]["frozen_corpus"]["identity_hash"]
    )
    snapshot_stable = (
        config_file_hash_before == config_file_hash_after
        and resolved_config_hash_before == resolved_config_hash_after
        and contract_hash_before == contract_hash_after
        and git_provenance_before["git_full_diff_hash"]
        == git_provenance_after["git_full_diff_hash"]
        and config_before["foundation_state_c1"]["frozen_corpus"]["identity_hash"]
        == corpus_hash
    )
    wrapper_exit_code = completed.returncode or (0 if snapshot_stable else 1)
    status = (
        "passed"
        if wrapper_exit_code == 0 and not failed_rows and snapshot_stable
        else "failed"
    )
    suite_terminal_summary = {
        suite: {
            **counts_by_suite,
            "status": "passed" if counts_by_suite["failures"] == 0 and counts_by_suite["errors"] == 0 else "failed",
            "evidence_mode": "covered_by_full_libs_tests",
        }
        for suite, counts_by_suite in suite_counts.items()
    }
    summary = {
        "schema_version": "neuronsifter-c1-pretraining-regression-v1",
        "status": status,
        "run_uid": run_uid,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resource_profile": os.environ.get("NRNSIFTER_C1_RESOURCE_PROFILE"),
        "command": command,
        "start_time": started_at,
        "end_time": ended_at,
        "terminal": True,
        "exit_code": wrapper_exit_code,
        "pytest_exit_code": completed.returncode,
        "counts": counts,
        "corpus_hash": corpus_hash,
        "config_path": args.config,
        "config_sha256": config_file_hash_after,
        "resolved_config_hash": resolved_config_hash_after,
        "foundation_state_contract_hash": contract_hash_after,
        **git_provenance_after,
        "snapshot_stable_during_regression": snapshot_stable,
        "snapshot_before": {
            "config_sha256": config_file_hash_before,
            "resolved_config_hash": resolved_config_hash_before,
            "foundation_state_contract_hash": contract_hash_before,
            "git_full_diff_hash": git_provenance_before["git_full_diff_hash"],
        },
        "snapshot_after": {
            "config_sha256": config_file_hash_after,
            "resolved_config_hash": resolved_config_hash_after,
            "foundation_state_contract_hash": contract_hash_after,
            "git_full_diff_hash": git_provenance_after["git_full_diff_hash"],
        },
        "host": platform.node(),
        "python": sys.version.split()[0],
        "stdout_path": stdout_path.relative_to(REPO_ROOT).as_posix(),
        "stderr_path": stderr_path.relative_to(REPO_ROOT).as_posix(),
        "junit_path": xml_path.relative_to(REPO_ROOT).as_posix(),
        "failed_test_count": len(failed_rows),
        "suite_terminal_summary": suite_terminal_summary,
    }
    write_json(audit_root / f"{artifact_stem}_summary.json", summary)
    failed_path = audit_root / f"{args.output_prefix}_pretraining_failed_tests.tsv"
    with failed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("classname", "test_name", "failure_type", "message"), delimiter="\t")
        writer.writeheader()
        writer.writerows(failed_rows)
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(wrapper_exit_code)


if __name__ == "__main__":
    main()
