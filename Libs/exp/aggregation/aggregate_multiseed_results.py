from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp_utils import aggregate_seed_metrics
from Libs.utils.exp_utils import build_run_metadata
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.exp_utils import parse_standard_experiment_args
from Libs.utils.exp_utils import resolve_output_dir
from Libs.utils.exp_utils import write_artifact_hashes
from Libs.utils.exp_utils import write_metrics
from Libs.utils.exp_utils import write_run_metadata
from Libs.utils.exp_utils import write_status


def metric_files_for_task(config: dict[str, object], task: str) -> list[Path]:
    suite = dict(config["paper_grade_full_suite"])
    manifest_template = str(dict(suite["manifest_contract"])["manifest_template"])
    paths: list[Path] = []
    for task_row in suite["tasks"]:
        task_payload = dict(task_row)
        if str(task_payload["task_name"]) != task:
            continue
        for model_key in [str(value) for value in task_payload["required_models"]]:
            for seed in [int(value) for value in suite["formal_seed_values"]]:
                manifest_path = REPO_ROOT / manifest_template.format(task_name=task, model_key=model_key, seed=seed)
                metrics_path = manifest_path.parent / "metrics.json"
                if metrics_path.exists():
                    paths.append(metrics_path)
    return sorted(paths)


def main() -> None:
    args = parse_standard_experiment_args(
        "Aggregate paper-grade multiseed metrics.",
        "paper_report_generation",
    )
    config = load_config_stack(args.config)
    output_dir = resolve_output_dir(args, config, dict(config["paper_grade_full_suite"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_files = metric_files_for_task(config, str(args.task))
    aggregate = {
        "task": str(args.task),
        "status": "dry_run" if args.dry_run else "materialized",
        "metric_files": [path.relative_to(REPO_ROOT).as_posix() for path in metric_files],
        "aggregate": aggregate_seed_metrics(metric_files) if metric_files else {"metric_count": 0, "seed_count": 0, "metrics": {}},
    }
    metadata = build_run_metadata(args, config, dict(config["paper_grade_full_suite"]))
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    write_metrics(output_dir / "metrics.json", aggregate)
    write_status(
        output_dir / "status.json",
        {
            "status": "dry_run" if args.dry_run else "materialized",
            "task_name": str(args.task),
            "metric_file_count": len(metric_files),
        },
    )
    write_artifact_hashes(output_dir / "artifact_hashes.json", metric_files + [output_dir / "metrics.json", output_dir / "run_metadata.json"])
    print(json.dumps({"output_dir": output_dir.relative_to(REPO_ROOT).as_posix(), "metric_file_count": len(metric_files)}, sort_keys=True))


if __name__ == "__main__":
    main()
