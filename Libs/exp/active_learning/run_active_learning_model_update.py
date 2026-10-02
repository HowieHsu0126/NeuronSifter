from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.active_learning_loop import round_path
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run active-learning-triggered NeuronSifter model update.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--round-uid", required=True)
    parser.add_argument("--one-batch-smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    update_path = round_path(config, "teacher_update_template", args.round_uid)
    if not update_path.exists():
        raise FileNotFoundError(f"Active-learning model update requires teacher update manifest: {update_path}")
    update_frame = pd.read_csv(update_path, sep="\t")
    main_update_uses = set(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["model_update_policy"]["accepted_qc_uses_for_main_update"])
    if update_frame.loc[update_frame["qc_use"].astype(str).isin(main_update_uses)].empty:
        raise ValueError("Active-learning model update requires at least one Grade A/B main update row.")
    command = [
        sys.executable,
        "Libs/exp/training/pretrain_microenvironment_world_model.py",
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    if bool(args.one_batch_smoke):
        command.append("--one-batch-smoke")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=update_path,
        round_uid=args.round_uid,
        update_row_count=int(update_frame.shape[0]),
    )


if __name__ == "__main__":
    main()
