from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_tsv
from Libs.utils.evaluation.active_learning_loop import round_path
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize active-learning teacher-family update manifest.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    update_path = round_path(config, "teacher_update_template", args.round_uid)
    if not update_path.exists():
        raise FileNotFoundError(f"Missing active-learning teacher update manifest: {update_path}")
    update_frame = pd.read_csv(update_path, sep="\t")
    output_path = resolve_repo_path(args.output_path) if args.output_path else update_path
    write_tsv(output_path, config["schemas"]["active_learning_teacher_update_fields"], update_frame.to_dict(orient="records"))
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        round_uid=args.round_uid,
        teacher_update_count=int(update_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
