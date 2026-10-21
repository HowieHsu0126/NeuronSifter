"""Run a command through the frozen Round-11G1 execution-input bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.round11g_execution_inputs import load_execution_input_bundle  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import run_with_execution_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a command through the Round-11G1 frozen input bundle.")
    parser.add_argument("--worktree-root", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worktree_root = Path(args.worktree_root).resolve()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    bundle = load_execution_input_bundle(
        config_stack_path=worktree_root / "Libs/configs/config_stack.yaml",
        expected_code_commit=args.expected_code_commit,
    )
    print(
        json.dumps(
            {
                "bundle_uid": bundle.bundle_uid,
                "bundle_sha256": bundle.manifest_sha256,
                "code_commit": bundle.code_commit,
                "environment_hash": bundle.environment_hash,
                "asset_count": len(bundle.assets),
                "command": command,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            sort_keys=True,
        )
    )
    output_root = Path(args.output_root).resolve() if args.output_root else None
    return run_with_execution_inputs(worktree_root, bundle, command, output_root=output_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
