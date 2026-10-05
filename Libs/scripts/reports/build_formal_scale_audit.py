from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.formal_scale_audit import audit_row, load_json, log_duration_seconds, write_audit_report
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a formal-scale audit report for benchmark and screening runs.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def checkpoint_parameter_count(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state", checkpoint)
    return int(sum(tensor.numel() for tensor in model_state.values()))


def checkpoint_model_name(checkpoint_path: Path, fallback: str) -> str:
    if not checkpoint_path.exists():
        return fallback
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if str(checkpoint.get("model_kwargs", {}).get("tri_weight_by_component", "")):
        return "ad_screening_system"
    return fallback


def collect_rows(root: Path, benchmark_name: str, disease_program: str, model_name: str) -> list[dict]:
    rows: list[dict] = []
    for artifact_root in sorted(root.glob(f"**/{model_name}")):
        run_metadata_path = artifact_root / "run_metadata.json"
        training_manifest_path = artifact_root / "training_manifest.json"
        if not run_metadata_path.exists() or not training_manifest_path.exists():
            continue
        run_metadata = load_json(run_metadata_path)
        training_manifest = load_json(training_manifest_path)
        checkpoint_path = artifact_root / "model_weights.pt"
        training_manifest["parameter_count"] = checkpoint_parameter_count(checkpoint_path)
        actual_model_name = checkpoint_model_name(checkpoint_path, model_name)
        log_path = artifact_root.parent.parent.parent / "logs"
        duration_seconds = None
        if isinstance(run_metadata.get("command", ""), str):
            duration_seconds = None
        rows.append(
            audit_row(
                benchmark_name=benchmark_name,
                disease_program=disease_program,
                model_name=actual_model_name,
                artifact_root=artifact_root,
                run_metadata=run_metadata,
                training_manifest=training_manifest,
                duration_seconds=duration_seconds,
            )
        )
    return rows


def collect_log_rows(log_dir: Path, rows: list[dict]) -> list[dict]:
    duration_by_command_prefix: dict[str, float] = {}
    for log_path in sorted(log_dir.glob("*.err")):
        duration_seconds = log_duration_seconds(log_path)
        if duration_seconds is None:
            continue
        duration_by_command_prefix[log_path.stem.split(".")[0]] = duration_seconds
    for row in rows:
        command = str(row.get("command", ""))
        if "train_shared_latent_benchmark.py" in command:
            row["duration_seconds"] = duration_by_command_prefix.get("NrnSift_shared_latent_formal", row["duration_seconds"])
        elif "train_screening_system.py" in command:
            for prefix in [
                "NrnSift_ad_screening_system_formal_ad_control",
                "NrnSift_neurodegeneration_without_atlas_prior",
                "NrnSift_neurodegeneration_without_clue_reversal",
                "NrnSift_neurodegeneration_without_target_ranking",
                "NrnSift_neurodegeneration_without_pkpd_prior",
                "NrnSift_neurodegeneration_without_tri_gate",
            ]:
                if prefix in duration_by_command_prefix:
                    row["duration_seconds"] = duration_by_command_prefix[prefix]
                    break
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    output_path = (REPO_ROOT / (args.output_path or config["paper"]["paths"]["formal_scale_audit_path"])).resolve()
    ensure_directory(output_path.parent)

    rows: list[dict] = []
    active_program = str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    benchmark_root = Path(config["benchmark_registry"]["artifact_root"]).resolve()
    output_roots = config["paper"]["output_roots"]
    screening_roots = [
        (REPO_ROOT / str(output_roots["screening_formal_root"])).resolve(),
        (REPO_ROOT / str(output_roots["screening_ablation_root"])).resolve(),
    ]

    rows.extend(collect_rows(benchmark_root, config["benchmark_registry"]["benchmark_name"], active_program, "shared_latent_dual_encoder_classifier"))
    rows.extend(collect_rows(benchmark_root, config["benchmark_registry"]["benchmark_name"], active_program, "tabular_jepa_auxiliary_encoder"))
    for screening_root in screening_roots:
        if screening_root.exists():
            rows.extend(collect_rows(screening_root, config["screening_pipeline"]["screening_benchmark"]["benchmark_name"], active_program, "ad_screening_system"))
    rows = collect_log_rows((REPO_ROOT / "Output/logs").resolve(), rows)
    rows.sort(key=lambda row: (row["model_name"], row["artifact_root"]))
    write_audit_report(output_path, rows)

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        audit_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
