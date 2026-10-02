from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.utils.evaluation.active_learning_loop import active_learning_config, active_learning_root
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured three-round NeuronSifter active-learning loop.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--priority-catalog", default="")
    parser.add_argument("--task-count-override", default="")
    parser.add_argument("--require-result-manifests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    active_config = active_learning_config(config)
    if not args.priority_catalog:
        subprocess.run(
            [
                sys.executable,
                "Libs/scripts/active_learning/build_active_learning_priority_catalog.py",
                "--config",
                args.config,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    round_summaries: list[dict[str, object]] = []
    for round_config in active_config["active_rounds"]:
        round_uid = str(round_config["round_uid"])
        command = [
            sys.executable,
            "Libs/exp/active_learning/run_active_learning_round.py",
            "--config",
            args.config,
            "--round-uid",
            round_uid,
        ]
        if args.priority_catalog:
            command.extend(["--priority-catalog", args.priority_catalog])
        if args.task_count_override:
            command.extend(["--task-count-override", args.task_count_override])
        if bool(args.require_result_manifests):
            result_manifest = active_learning_root(config) / str(active_config["relative_paths"]["result_manifest_template"]).format(round_uid=round_uid)
            if not result_manifest.exists():
                raise FileNotFoundError(f"Required completed result manifest is missing for full active-learning loop: {result_manifest}")
            command.extend(["--result-manifest", str(result_manifest)])
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        round_summaries.append({"round_uid": round_uid, "status": "round_command_completed", "command": command})
    output_path = active_learning_root(config) / str(active_config["relative_paths"]["loop_summary"])
    write_json(output_path, {"status": "active_learning_loop_completed", "rounds": round_summaries})
    print(json.dumps({"status": "active_learning_loop_completed", "rounds": round_summaries}, indent=2))
    complete_script_logging(pipeline_logger, pipeline_started_at, artifact=output_path)


if __name__ == "__main__":
    main()
