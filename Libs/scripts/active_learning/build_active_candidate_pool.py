from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path
from Libs.utils.evaluation.active_learning_loop import active_learning_config, active_learning_root, build_active_candidate_pool, round_path, write_frame
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build active-learning candidate pool for one NeuronSifter round.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--priority-catalog", default="")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def default_priority_path(config: dict) -> Path:
    return active_learning_root(config) / str(active_learning_config(config)["relative_paths"]["priority_catalog"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    intervention_config = config["twin"]["intervention_program_registry"]
    intervention_path = (
        resolve_repo_path(str(intervention_config["local_root"]))
        / str(intervention_config["relative_paths"]["intervention_program_catalog"])
    )
    validation_config = config["validation"]["candidate_validation_registry"]
    validation_path = (
        resolve_repo_path(str(validation_config["local_root"]))
        / str(validation_config["relative_paths"]["candidate_validation_catalog"])
    )
    priority_path = resolve_repo_path(args.priority_catalog) if args.priority_catalog else default_priority_path(config)
    if not priority_path.exists():
        raise FileNotFoundError(f"Active-learning priority catalog is required: {priority_path}")
    intervention_frame = pd.DataFrame(read_tsv(intervention_path))
    validation_frame = pd.DataFrame(read_tsv(validation_path)) if validation_path.exists() else pd.DataFrame()
    priority_frame = pd.read_csv(priority_path, sep="\t")
    candidate_frame = build_active_candidate_pool(args.round_uid, intervention_frame, validation_frame, priority_frame, config)
    output_path = resolve_repo_path(args.output_path) if args.output_path else round_path(config, "candidate_pool_template", args.round_uid)
    write_frame(output_path, config["schemas"]["active_learning_candidate_pool_fields"], candidate_frame)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        round_uid=args.round_uid,
        candidate_count=int(candidate_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
