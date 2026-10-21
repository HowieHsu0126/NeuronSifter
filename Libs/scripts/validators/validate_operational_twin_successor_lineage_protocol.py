"""Validate the prospective or materialized Operational Twin successor protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_successor_lineage_protocol import (  # noqa: E402
    EXPECTED_SOURCE_GATE_CHECKS,
    INVALID_TERMINAL,
    POSITIVE_TERMINAL,
    OperationalTwinProtocolError,
    build_pre_change_identity_manifest,
    build_source_gate_evidence,
    build_successor_package,
    compute_successor_identity_declarations,
    validate_materialized_successor_package,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def _write_json_exclusive(path: Path, payload: object) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"successor_validation_output_collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _clean_tracked_source(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout == ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Libs/configs/config_stack.yaml",
    )
    parser.add_argument("--mode", choices=("adjudicate", "materialized"))
    parser.add_argument("--print-expected-identities", action="store_true")
    parser.add_argument("--write-pre-change-identity-manifest", type=Path)
    parser.add_argument("--write-source-gate-evidence", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--completed-check", action="append", default=[])
    args = parser.parse_args()

    config = load_config_stack(args.config.resolve())
    blockers: list[dict[str, str]] = []
    try:
        if args.write_pre_change_identity_manifest is not None:
            manifest = build_pre_change_identity_manifest(config, REPO_ROOT)
            if any(
                row["git_object_id_at_planner_commit"] is not None
                or not row["filesystem_absent_when_recorded"]
                for row in (
                    list(manifest["successor_paths_absent"])
                    + list(manifest["reserved_retry_paths_absent"])
                )
            ):
                raise OperationalTwinProtocolError(
                    "successor_pre_change_manifest_paths_not_absent"
                )
            _write_json_exclusive(
                args.write_pre_change_identity_manifest.resolve(), manifest
            )
        configured_manifest = REPO_ROOT / str(
            config["operational_twin_successor_lineage_protocol"][
                "evidence_inputs"
            ]["pre_change_identity_manifest"]
        )
        pre_change = None
        if configured_manifest.is_file():
            value = json.loads(configured_manifest.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                pre_change = value
        identities = compute_successor_identity_declarations(
            config, REPO_ROOT, pre_change
        )
        if args.print_expected_identities:
            print(json.dumps({"expected_identity": identities}, sort_keys=True))
            return 0
        if args.mode is None:
            parser.error("--mode is required unless --print-expected-identities is used")
        if args.mode == "materialized":
            blockers.extend(validate_materialized_successor_package(config, REPO_ROOT))
        else:
            _, _, prospective_blockers = build_successor_package(
                config,
                REPO_ROOT,
                require_publication_targets_absent=True,
                require_committed_sources=bool(args.write_source_gate_evidence),
            )
            blockers.extend(prospective_blockers)
        if args.write_source_gate_evidence is not None:
            if args.mode != "adjudicate":
                raise OperationalTwinProtocolError(
                    "successor_source_gate_requires_adjudicate_mode"
                )
            source_commit = args.source_commit or ""
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
            if source_commit != current_commit or not _clean_tracked_source(REPO_ROOT):
                blockers.append(
                    {
                        "code": "successor_source_gate_not_exact_clean",
                        "message": "Source commit must equal clean tracked HEAD.",
                    }
                )
            if sorted(args.completed_check) != EXPECTED_SOURCE_GATE_CHECKS:
                blockers.append(
                    {
                        "code": "successor_source_gate_checks_incomplete",
                        "message": "The exact focused source-gate checklist is required.",
                    }
                )
            if not str(args.slurm_job_id or "").isdigit():
                blockers.append(
                    {
                        "code": "successor_source_gate_job_id_invalid",
                        "message": "A numeric Slurm job ID is required.",
                    }
                )
            if not blockers:
                evidence = build_source_gate_evidence(
                    config,
                    REPO_ROOT,
                    completed_checks=args.completed_check,
                    source_commit=source_commit,
                    slurm_job_id=str(args.slurm_job_id),
                )
                _write_json_exclusive(
                    args.write_source_gate_evidence.resolve(), evidence
                )
    except (
        FileExistsError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        blockers.append(
            {"code": "successor_protocol_validation_failed", "message": str(error)}
        )

    payload = {
        "status": "PASS" if not blockers else "BLOCKED",
        "terminal_decision": POSITIVE_TERMINAL if not blockers else INVALID_TERMINAL,
        "candidate_dispatch_authorized": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": config.get(
            "operational_twin_successor_lineage_protocol", {}
        ).get("access_counters", {}),
        "blockers": blockers,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
