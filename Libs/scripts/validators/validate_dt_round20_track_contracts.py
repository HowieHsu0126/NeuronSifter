#!/usr/bin/env python3
"""Validate and materialize the immutable Round-20 Track A/B/C contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402
from Libs.utils.data.round20_track_contracts import (  # noqa: E402
    build_track_contract_report,
    file_sha256,
)
from Libs.utils.data.trajectory_contracts import blocker, sorted_blockers  # noqa: E402


CONFIG_RELATIVE = Path("Libs/configs/validation/round20_track_contracts.yaml")
STACK_RELATIVE = Path("Libs/configs/config_stack.yaml")
EXPECTED_BRANCH = "round20-track-contracts-20260805"
BASE_MAIN_COMMIT = "ee45298421c9dbabab952f790b4662e8c7d67f87"


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = payload.get("round20_track_contracts") if isinstance(payload, dict) else None
    if not isinstance(contract, dict):
        raise TypeError("round20_track_contracts must be a YAML object")
    return contract


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(
    *, repository: Path, config_path: Path, require_clean: bool
) -> tuple[dict[str, object], dict[str, Any]]:
    contract = _load_contract(config_path)
    report = build_track_contract_report(contract, repository=repository)
    blockers = list(report["blockers"])
    head = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    branch = _git(repository, "branch", "--show-current")
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")

    if branch != EXPECTED_BRANCH:
        blockers.append(blocker("branch_identity", "Track contract candidate branch is not exact.", expected=EXPECTED_BRANCH, actual=branch))
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_MAIN_COMMIT, "HEAD"],
        cwd=repository,
        check=False,
    ).returncode != 0:
        blockers.append(blocker("base_ancestry", "Track contract candidate does not descend from the accepted main commit."))
    if require_clean and dirty:
        blockers.append(blocker("dirty_candidate", "Exact-clean validation was requested.", status=dirty.splitlines()))

    stack = load_config(repository / STACK_RELATIVE)
    if stack.get("round20_track_contracts") != contract:
        blockers.append(blocker("config_stack_binding", "The global config stack does not load the exact Track contract."))

    report.update(
        {
            "status": "passed" if not blockers else "blocked",
            "blocker_count": len(blockers),
            "blockers": sorted_blockers(blockers),
            "candidate_commit": head,
            "candidate_tree": tree,
            "candidate_branch": branch,
            "dirty_status": "clean" if not dirty else "dirty",
            "config_path": str(config_path.relative_to(repository)),
            "config_file_sha256": file_sha256(config_path),
        }
    )
    if blockers:
        report["next_action"] = "repair_track_contract_blockers_and_rerun_with_new_identity"
    return report, contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    report, contract = validate(
        repository=repository,
        config_path=config_path.resolve(),
        require_clean=args.require_clean,
    )
    output_paths = contract["output_paths"]
    output = args.output or repository / output_paths["freeze"]
    output = output if output.is_absolute() else repository / output
    _write_json(output, report)

    tracks = contract["tracks"]
    for output_key, track_name in (
        ("track_a", "track_a_allen_full_landing"),
        ("track_b", "track_b_public_e2_e4_recovery"),
        ("track_c", "track_c_runtime_deployment"),
    ):
        track = tracks[track_name]
        payload = {
            "schema_version": "neuronsifter-round20-track-contract-v1",
            "track_uid": track_name,
            "contract_status": report["status"],
            "execution_status": track["status"],
            "candidate_commit": report["candidate_commit"],
            "config_file_sha256": report["config_file_sha256"],
            "scope": track["scope"],
            "contract": track,
            "blockers": report["blockers"],
            "authorization_boundary": track["authorization"],
            "handoff": track["handoff"],
        }
        _write_json(repository / output_paths[output_key], payload)

    print(
        json.dumps(
            {
                "status": report["status"],
                "blockers": report["blocker_count"],
                "candidate_commit": report["candidate_commit"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
