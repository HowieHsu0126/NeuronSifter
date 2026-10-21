"""Build one immutable Operational Twin candidate attempt execution gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    build_and_publish_candidate_execution_gate,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Libs/configs/config_stack.yaml",
    )
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    task_index = (
        int(args.task_index)
        if args.task_index is not None
        else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    config = load_config_stack(args.config.resolve())
    result = build_and_publish_candidate_execution_gate(
        config,
        REPO_ROOT,
        task_index=task_index,
    )
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
