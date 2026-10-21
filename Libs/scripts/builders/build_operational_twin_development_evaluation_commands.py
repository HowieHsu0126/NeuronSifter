"""Build immutable per-assignment commands for open Twin evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_training_protocol import repo_path  # noqa: E402
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def build_commands() -> dict[str, object]:
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    protocol = dict(config["operational_twin_training_protocol"])
    roots = dict(protocol["output_roots"])
    manifest = json.loads(
        repo_path(REPO_ROOT, roots["assignment_manifest"], must_exist=True).read_text(
            encoding="utf-8"
        )
    )
    commands = []
    for assignment in manifest["assignments"]:
        command = [
            "env",
            "PYTHONPATH=.",
            "python",
            "Libs/exp/evaluation/evaluate_operational_twin_development.py",
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
            "--output-root",
            assignment["output_root"],
            "--assignment-uid",
            assignment["assignment_uid"],
            "--method-id",
            assignment["method_id"],
            "--seed",
            str(assignment["seed"]),
        ]
        commands.append(
            {
                "assignment_uid": assignment["assignment_uid"],
                "method_id": assignment["method_id"],
                "seed": int(assignment["seed"]),
                "resource_profile": assignment["resource_profile"],
                "output_root": assignment["output_root"],
                "command": " ".join(shlex.quote(part) for part in command),
            }
        )
    return {
        "schema_version": "neuronsifter-operational-twin-development-evaluation-command-catalog-v1",
        "assignment_count": len(commands),
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "commands": commands,
        "sealed_or_test_access": False,
        "formal_qualification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "Output/results/operational_twin/development_training/catalog/development_evaluation_commands.json",
    )
    args = parser.parse_args()
    payload = build_commands()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "assignment_count": payload["assignment_count"],
                "output": args.output.relative_to(REPO_ROOT).as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
