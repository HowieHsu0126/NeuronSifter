"""Validate the calibration-world-only Operational Twin execution authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_operational_twin_devcal import (  # noqa: E402
    DEFAULT_CALIBRATION_EXECUTION_AUTHORITY,
    require_calibration_execution_authority,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    _git_commit,
    validate_candidate_calibration_execution_authority,
    resolve_candidate_execution_protocol,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-scope",
        choices=("historical-v1", "candidate-lineage"),
        default="historical-v1",
    )
    parser.add_argument("--authority", type=Path)
    args = parser.parse_args()
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    candidate_mode = args.protocol_scope == "candidate-lineage"
    resolved = resolve_candidate_execution_protocol(config, REPO_ROOT) if candidate_mode else None
    authority_path = args.authority or REPO_ROOT / (
        resolved.execution["output_paths"]["calibration_execution_authority"]
        if candidate_mode
        else DEFAULT_CALIBRATION_EXECUTION_AUTHORITY
    )
    executing_source_commit = _git_commit(REPO_ROOT) if candidate_mode else None
    identity = require_calibration_execution_authority(
        config,
        REPO_ROOT,
        authority_path.resolve(),
        method_id="full_uncertainty_aware_action_conditioned",
        candidate_lineage_uid=(
            str(resolved.protocol["candidate_lineage_uid"])
            if candidate_mode
            else None
        ),
        protocol_uid=(
            str(resolved.protocol["protocol_uid"])
            if candidate_mode
            else None
        ),
        resolved_context=resolved,
        executing_source_commit=executing_source_commit,
    )
    if candidate_mode:
        blockers = validate_candidate_calibration_execution_authority(
            config,
            REPO_ROOT,
            authority_path,
            resolved_context=resolved,
            executing_source_commit=executing_source_commit,
        )
        if blockers:
            raise ValueError(
                "; ".join(f"{row['code']}: {row['message']}" for row in blockers)
            )
    print(json.dumps({"status": "PASS", "authority": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
