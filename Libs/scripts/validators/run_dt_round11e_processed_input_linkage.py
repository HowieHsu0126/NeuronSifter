"""Run a bounded test command with verified, read-only generated inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.round11e_processed_input_linkage import linkage_metadata  # noqa: E402
from Libs.utils.data.round11e_processed_input_linkage import load_processed_input_linkage  # noqa: E402
from Libs.utils.data.round11e_processed_input_linkage import run_with_processed_input_linkage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Round-11E test command through the generated-input allowlist.")
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    linkage = load_processed_input_linkage()
    print(
        json.dumps(
            {
                "command": command,
                "linkage": linkage_metadata(linkage),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            sort_keys=True,
        )
    )
    completed = run_with_processed_input_linkage(Path(args.worktree_root), command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
