"""Validate the derived C1 observation row binding against frozen metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_bindings import validate_c1_observation_binding
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen C1 observation row binding.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    blockers, actual = validate_c1_observation_binding(config, REPO_ROOT)
    required = config["foundation_state_c1"]["frozen_corpus"]["required_observation_binding"]
    report = {
        "schema_version": "neuronsifter-c1-observation-row-binding-validation-v1",
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "expected": dict(required),
        "actual": actual,
        "training_binding_allowed": not blockers,
        "g3_status": "not_evaluated",
    }
    output = args.output or config["foundation_state_c1"]["artifacts"][
        "observation_binding_validation_path"
    ]
    output_path = resolve_repo_path(output)
    write_json(output_path, report)
    print(f"c1_observation_binding_status={report['status']}")
    print(f"c1_observation_binding_blocker_count={report['blocker_count']}")
    print(f"c1_observation_binding_output={output_path.relative_to(REPO_ROOT).as_posix()}")
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
