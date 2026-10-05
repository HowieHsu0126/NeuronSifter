from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_screening_system import artifact_dir
from Libs.exp.training.train_screening_system import build_scope_training_cache_payload
from Libs.exp.training.train_screening_system import load_action_registries
from Libs.exp.training.train_screening_system import load_scope_modality_payload
from Libs.exp.training.train_screening_system import max_rss_mb
from Libs.exp.training.train_screening_system import repo_relative
from Libs.exp.training.train_screening_system import runtime_output_paths
from Libs.exp.training.train_screening_system import training_cache_manifest_path
from Libs.exp.training.train_screening_system import write_runtime_markdown
from Libs.exp.training.train_screening_system import write_scope_training_cache
from Libs.utils.data.raw_data_registry import ensure_directory
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import read_tsv
from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config
from Libs.utils.evaluation.screening_datasets import foundation_registry_frames
from Libs.utils.exp.experiment_tracking import scoped_seed
from Libs.utils.exp.logger import bootstrap_script_logging
from Libs.utils.exp.logger import complete_script_logging
from Libs.utils.evaluation.strict_case_discovery import assert_clean_action_ready_manifest
from Libs.utils.evaluation.strict_case_discovery import strict_case_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute freeze-bound NeuronSifter screening training tensor caches.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--training-scope", default="all", choices=["folds", "production", "all"])
    parser.add_argument("--fold-name", default="")
    parser.add_argument("--disease-program", default="")
    return parser.parse_args()


def target_fold_rows(config: dict, disease_program: str, fold_name: str) -> list[dict[str, str]]:
    fold_catalog = read_tsv(
        resolve_repo_path(config["benchmark_registry"]["local_root"])
        / config["benchmark_registry"]["relative_paths"]["fold_catalog"]
    )
    return [
        row
        for row in fold_catalog
        if row.get("disease_program", disease_program) == disease_program
        and (not fold_name or row["fold_name"] == fold_name)
    ]


def write_data_io_audit(config: dict, payload: dict[str, object]) -> None:
    json_path, md_path = runtime_output_paths(
        config,
        "data_io_bottleneck_audit_json",
        "data_io_bottleneck_audit_md",
    )
    write_json(json_path, payload)
    write_runtime_markdown(md_path, "Data I/O Bottleneck Audit", payload)


def build_cache_for_scope(
    config: dict,
    config_path: Path,
    argv: list[str],
    scope: str,
    split_name: str,
    disease_program: str,
    benchmark_name: str,
    target_program_rows: list[dict[str, str]],
    regimen_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    compound_rows: list[dict[str, str]],
    strict_case_rows: list[dict[str, object]],
    foundation_registries: dict[str, pd.DataFrame],
) -> dict[str, object]:
    started_at = time.perf_counter()
    modality_payload = load_scope_modality_payload(config, scope, split_name, disease_program, benchmark_name)
    cache_payload = build_scope_training_cache_payload(
        config,
        scope,
        split_name,
        modality_payload,
        target_program_rows,
        regimen_rows,
        target_rows,
        compound_rows,
        strict_case_rows,
        foundation_registries,
        disease_program,
    )
    materialization_seconds = time.perf_counter() - started_at
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    model_seed = scoped_seed(int(training_config["random_seed"]), scope, split_name, "ad_screening_system")
    output_root = artifact_dir(config, scope, split_name)
    manifest = write_scope_training_cache(
        config,
        config_path,
        argv,
        output_root,
        scope,
        split_name,
        model_seed,
        cache_payload,
        materialization_seconds,
    )
    return {
        "scope": scope,
        "fold": split_name,
        "benchmark_name": benchmark_name,
        "manifest_path": repo_relative(training_cache_manifest_path(output_root, config)),
        "data_materialization_seconds": materialization_seconds,
        "sample_counts": manifest["sample_counts"],
        "selected_target_program_count": manifest["selected_target_program_count"],
        "selected_regimen_count": manifest["selected_regimen_count"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    disease_program = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    config["screening_pipeline"]["screening_benchmark"]["active_disease_program"] = disease_program
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    config_path = Path(args.config).resolve()
    assert_clean_action_ready_manifest(config)
    strict_case_rows = strict_case_entries(config, disease_program)
    target_program_rows, regimen_rows, target_rows, compound_rows = load_action_registries(config)
    foundation_registries = foundation_registry_frames(config)
    rows: list[dict[str, object]] = []
    folds = target_fold_rows(config, disease_program, args.fold_name)
    if args.training_scope in {"folds", "all"}:
        if not folds:
            raise ValueError(f"No fold rows found for disease_program={disease_program} fold_name={args.fold_name}.")
        for fold_row in folds:
            rows.append(
                build_cache_for_scope(
                    config,
                    config_path,
                    sys.argv,
                    "fold",
                    str(fold_row["fold_name"]),
                    disease_program,
                    str(fold_row["benchmark_name"]),
                    target_program_rows,
                    regimen_rows,
                    target_rows,
                    compound_rows,
                    strict_case_rows,
                    foundation_registries,
                )
            )
    if args.training_scope in {"production", "all"}:
        rows.append(
            build_cache_for_scope(
                config,
                config_path,
                sys.argv,
                "production",
                "production",
                disease_program,
                str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"]),
                target_program_rows,
                regimen_rows,
                target_rows,
                compound_rows,
                strict_case_rows,
                foundation_registries,
            )
        )
    audit_payload = {
        "audit_name": "data_io_bottleneck_audit",
        "schema_version": "paper_grade_data_io_audit.v1",
        "status": "materialized",
        "cache_count": len(rows),
        "caches": rows,
        "total_data_materialization_seconds": float(sum(float(row["data_materialization_seconds"]) for row in rows)),
        "max_rss_mb": max_rss_mb(),
    }
    write_data_io_audit(config, audit_payload)
    ensure_directory(REPO_ROOT / "Output/audits")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=runtime_output_paths(config, "data_io_bottleneck_audit_json", "data_io_bottleneck_audit_md")[0],
        cache_count=len(rows),
    )


if __name__ == "__main__":
    main()
