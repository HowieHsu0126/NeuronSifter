from __future__ import annotations

import argparse
import subprocess
import shlex
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import ensure_directory, load_config
from Libs.utils.exp.runnable_baselines import adapter_mode, baseline_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standardized baseline adapter commands.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--command", required=True, choices=["prepare_inputs", "train_baseline", "evaluate_baseline"])
    parser.add_argument("--disease-program", default="")
    parser.add_argument("--fold-name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--baseline-output-dir", default="")
    return parser.parse_args()


def render_external_command(
    config: dict,
    entry: dict,
    command_name: str,
    config_path: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
) -> str:
    command_templates = entry.get("external_command_templates", {})
    if command_name not in command_templates:
        raise ValueError(f"Baseline {entry['baseline_name']} is marked direct_external_wrapper but has no template for {command_name}.")
    return str(command_templates[command_name]).format(
        config_path=config_path,
        disease_program=disease_program,
        fold_name=fold_name,
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        baseline_name=str(entry["baseline_name"]),
        source_repo_path=str(entry.get("source_repo_path", "")),
        python_executable=str(config["execution_environment"]["python_executable"]),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    disease_program = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    entry = baseline_entry(config, args.baseline_name)
    current_adapter_mode = adapter_mode(entry)
    if args.baseline_output_dir:
        output_dir = (REPO_ROOT / args.baseline_output_dir).resolve()
    else:
        output_root = args.output_dir or str(config["baseline_adapter_registry"]["output_roots"]["runnable_baseline_root"])
        output_dir = (REPO_ROOT / output_root / disease_program / args.baseline_name).resolve()
    if args.fold_name:
        output_dir = output_dir / args.fold_name
    ensure_directory(output_dir)

    if not args.fold_name:
        raise ValueError("--fold-name is required for runnable baseline execution.")
    manifest_path = output_dir / f"{disease_program}_{args.fold_name}_latent_transition_ready_manifest.json"
    checkpoint_path = output_dir / "model_weights.pt"
    if current_adapter_mode == "literature_only":
        raise ValueError(f"Baseline {args.baseline_name} is literature_only and cannot be executed.")
    if current_adapter_mode == "local_matched_split_wrapper":
        raise ValueError(
            f"Baseline {args.baseline_name} is a suite-level local matched-split wrapper. "
            "Run it through Libs/exp/baselines/run_runnable_baseline_suite.py."
        )
    if current_adapter_mode == "direct_external_wrapper":
        command = render_external_command(
            config,
            entry,
            args.command,
            args.config,
            disease_program,
            args.fold_name,
            output_dir,
            manifest_path,
            checkpoint_path,
        )
        subprocess.run(shlex.split(command), check=True, cwd=REPO_ROOT)
        return
    raise ValueError(
        f"Baseline {args.baseline_name} uses unsupported adapter_mode={current_adapter_mode}. "
        "Only direct_external_wrapper and literature_only are allowed."
    )


if __name__ == "__main__":
    main()
