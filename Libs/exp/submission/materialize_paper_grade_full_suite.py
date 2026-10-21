from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp_utils import build_run_metadata
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.exp_utils import parse_standard_experiment_args
from Libs.utils.exp_utils import resolve_output_dir
from Libs.utils.exp_utils import write_artifact_hashes
from Libs.utils.exp_utils import write_metrics
from Libs.utils.exp_utils import write_run_metadata
from Libs.utils.exp_utils import write_status


def materialization_summary(config: dict[str, object]) -> dict[str, object]:
    suite = dict(config["paper_grade_full_suite"])
    registry = dict(suite["entrypoint_registry"])
    return {
        "suite_id": str(suite.get("suite_id", suite["suite_name"])),
        "suite_release": str(suite["suite_release"]),
        "seeds": [int(value) for value in suite["formal_seed_values"]],
        "task_count": len(registry),
        "tasks": [
            {
                "task_id": task_name,
                "shell_entrypoint": str(dict(entry)["shell_path"]),
                "python_entrypoint": str(dict(entry)["python_path"]),
                "resource_profile": str(dict(entry)["resource_profile"]),
                "output_template": str(dict(entry)["output_template"]),
            }
            for task_name, entry in sorted(registry.items())
        ],
        "live_submission_performed": False,
    }


def main() -> None:
    args = parse_standard_experiment_args(
        "Materialize paper-grade full-suite metadata without Slurm submission.",
        "paper_report_generation",
    )
    config = load_config_stack(args.config)
    output_dir = resolve_output_dir(args, config, dict(config["paper_grade_full_suite"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = materialization_summary(config)
    status_value = "dry_run" if args.dry_run else "materialized"
    summary["status"] = status_value
    write_run_metadata(output_dir / "run_metadata.json", build_run_metadata(args, config, dict(config["paper_grade_full_suite"])))
    write_metrics(output_dir / "metrics.json", summary)
    write_status(output_dir / "status.json", {"status": status_value, "live_submission_performed": False})
    write_artifact_hashes(output_dir / "artifact_hashes.json", [output_dir / "metrics.json", output_dir / "run_metadata.json", output_dir / "status.json"])
    print(json.dumps({"output_dir": output_dir.relative_to(REPO_ROOT).as_posix(), "status": status_value}, sort_keys=True))


if __name__ == "__main__":
    main()
