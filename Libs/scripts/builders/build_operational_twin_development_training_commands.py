"""Build manifest-bound Operational Twin development-training commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    REQUIRED_METHOD_IDS,
    command_catalog_hash,
    expected_candidate_command_argv,
    load_candidate_protocol_artifacts,
    repo_path,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def build_commands(*, dry_run: bool = False) -> dict[str, object]:
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    protocol = dict(config["operational_twin_training_protocol"])
    roots = dict(protocol["output_roots"])
    manifest_path = repo_path(REPO_ROOT, roots["assignment_manifest"], must_exist=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commands = []
    for row in manifest["assignments"]:
        cmd = [
            "env",
            "PYTHONPATH=.",
            "python",
            "Libs/exp/training/train_operational_twin_devcal.py",
            "--config",
            "Libs/configs/model/operational_twin.yaml",
            "--training-manifest",
            protocol["manifests"]["training"],
            "--development-manifest",
            protocol["manifests"]["development"],
            "--calibration-manifest",
            protocol["manifests"]["calibration"],
            "--package-receipt",
            "Input/processed/controlled_neural_world/open_action_identification/package_receipt.json",
            "--open-data-qualification",
            protocol["open_data_authority"],
            "--calibration-execution-authority",
            "Output/audits/operational_twin_calibration_execution_authority.json",
            "--output-root",
            row["output_root"],
            "--assignment-uid",
            row["assignment_uid"],
            "--method-id",
            row["method_id"],
            "--seed",
            str(row["seed"]),
        ]
        if dry_run:
            cmd.append("--dry-run")
        commands.append(
            {
                "assignment_uid": row["assignment_uid"],
                "method_id": row["method_id"],
                "seed": int(row["seed"]),
                "resource_profile": row["resource_profile"],
                "output_root": row["output_root"],
                "command": " ".join(shlex.quote(part) for part in cmd),
            }
        )
    return {
        "schema_version": "neuronsifter-operational-twin-development-training-command-catalog-v1",
        "assignment_count": len(commands),
        "method_ids": REQUIRED_METHOD_IDS,
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "commands": commands,
        "single_writer_catalog_root": roots["catalog_root"],
    }


def build_candidate_commands(
    config: dict[str, object],
    candidate_manifest: dict[str, object],
    assignment_manifest: dict[str, object],
) -> dict[str, object]:
    """Build the immutable structured argv catalog for both frozen attempt slots."""
    commands: list[dict[str, object]] = []
    for attempt_kind in ("initial", "infrastructure_recovery_01"):
        for assignment in assignment_manifest["assignments"]:
            assignment_row = dict(assignment)
            attempt = next(
                dict(row)
                for row in assignment_row["attempts"]
                if dict(row)["attempt_kind"] == attempt_kind
            )
            argv = expected_candidate_command_argv(
                config,
                candidate_manifest,
                assignment_manifest,
                assignment_row,
                attempt,
                command_catalog_hash_value="__COMMAND_CATALOG_HASH__",
            )
            commands.append(
                {
                    "command_index": len(commands),
                    "candidate_lineage_uid": assignment_row["candidate_lineage_uid"],
                    "protocol_uid": assignment_row["protocol_uid"],
                    "assignment_uid": assignment_row["assignment_uid"],
                    "attempt_uid": attempt["attempt_uid"],
                    "attempt_kind": attempt["attempt_kind"],
                    "conditional": attempt["conditional"],
                    "eligible_after_failure_class": attempt["eligible_after_failure_class"],
                    "predecessor_attempt_uid": attempt["predecessor_attempt_uid"],
                    "method_id": assignment_row["method_id"],
                    "seed": assignment_row["seed"],
                    "resource_profile": assignment_row["resource_profile"],
                    "resource_requirements": assignment_row["resource_requirements"],
                    "wall_time": assignment_row["wall_time"],
                    "output_root": attempt["output_root"],
                    "execution_authority": attempt["execution_authority"],
                    "resource_environment_receipt": attempt[
                        "resource_environment_receipt"
                    ],
                    "argv": argv,
                }
            )
    catalog: dict[str, object] = {
        "schema_version": "neuronsifter-operational-twin-candidate-command-catalog-v1",
        "candidate_lineage_uid": candidate_manifest["candidate_lineage_uid"],
        "protocol_uid": candidate_manifest["protocol_uid"],
        "preflight_uid": candidate_manifest["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest["assignment_manifest_hash"],
        "assignment_count": assignment_manifest["assignment_count"],
        "command_count": len(commands),
        "initial_command_count": 15,
        "conditional_recovery_command_count": 15,
        "commands": commands,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
    }
    digest = command_catalog_hash(catalog)
    for row in commands:
        argv = list(row["argv"])
        hash_index = argv.index("--command-catalog-hash") + 1
        argv[hash_index] = digest
        row["argv"] = argv
    catalog["command_catalog_hash"] = digest
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-scope",
        choices=("historical-v1", "candidate-lineage"),
        default="historical-v1",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.protocol_scope == "candidate-lineage":
        config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
        candidate_manifest, assignment_manifest, _, _ = load_candidate_protocol_artifacts(config, REPO_ROOT)
        payload = build_candidate_commands(config, candidate_manifest, assignment_manifest)
        default_output = REPO_ROOT / dict(config["operational_twin_candidate_lineage_protocol"])["output_paths"]["assignment_commands"]
    else:
        payload = build_commands(dry_run=args.dry_run)
        default_output = REPO_ROOT / "Output/results/operational_twin/development_training/catalog/assignment_commands.json"
    output = args.output or default_output
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.protocol_scope == "candidate-lineage":
        if os.path.lexists(output):
            raise FileExistsError(
                "candidate_command_catalog_collision: candidate command publication "
                f"is single-use and refuses an existing output path: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(content)
    else:
        if output.exists() and output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite differing command catalog: {output}")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "PASS", "assignment_count": payload["assignment_count"], "output": output.relative_to(REPO_ROOT).as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
