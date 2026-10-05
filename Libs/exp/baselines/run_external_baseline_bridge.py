from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.external_baseline_bridge import (
    evaluate_external_bridge,
    prepare_external_bridge_inputs,
    train_external_bridge,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config
from Libs.utils.exp.runnable_baselines import baseline_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuronSifter data bridges for vendored external baseline repositories.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--command", required=True, choices=["prepare_inputs", "train_baseline", "evaluate_baseline"])
    parser.add_argument("--disease-program", required=True)
    parser.add_argument("--fold-name", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    entry = baseline_entry(config, args.baseline_name)
    output_dir = Path(args.output_dir).resolve()
    ensure_directory(output_dir)
    manifest_path = output_dir / f"{args.baseline_name}_{args.disease_program}_{args.fold_name}_external_bundle_manifest.json"
    checkpoint_path = output_dir / "model_weights.pt"
    if args.command == "prepare_inputs":
        prepare_external_bridge_inputs(args.config, args.baseline_name, args.disease_program, args.fold_name, output_dir)
        return
    if not manifest_path.exists():
        prepare_external_bridge_inputs(args.config, args.baseline_name, args.disease_program, args.fold_name, output_dir)
    if args.command == "train_baseline":
        train_external_bridge(entry, manifest_path, output_dir)
        return
    if not checkpoint_path.exists():
        train_external_bridge(entry, manifest_path, output_dir)
    evaluate_external_bridge(entry, manifest_path, checkpoint_path, output_dir)


if __name__ == "__main__":
    main()
