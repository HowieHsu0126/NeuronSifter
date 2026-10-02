from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.experiment_manifests import deep_merge, load_experiment_family
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a paper experiment configuration from the repository YAML profile.")
    parser.add_argument("--profile", required=True, help="paper_experiments profile name in the base YAML.")
    parser.add_argument("--output-config", required=True, help="Destination path for the materialized YAML configuration.")
    parser.add_argument("--base-config", default="Libs/configs/config_stack.yaml", help="Repository YAML containing paper_experiments.")
    parser.add_argument("--seed", default="", help="Optional seed override for formal multi-seed runs.")
    return parser.parse_args()

def apply_seed_overrides(materialized_config: dict[str, object], runtime_config: dict[str, object], seed_value: int) -> None:
    benchmark_registry = materialized_config.get("benchmark_registry", {})
    shared_latent_config = benchmark_registry.get("shared_latent_model", {})
    logistic_config = benchmark_registry.get("logistic_regression", {})
    screening_registry = materialized_config.get("screening_pipeline", {}).get("screening_benchmark", {})

    if "random_seed" in shared_latent_config:
        shared_latent_config["random_seed"] = seed_value
    if "random_seed" in logistic_config:
        logistic_config["random_seed"] = seed_value
    if "training" in screening_registry and "random_seed" in screening_registry["training"]:
        screening_registry["training"]["random_seed"] = seed_value

    benchmark_name_template = runtime_config.get("benchmark_name_template", "")
    benchmark_name_scope = str(runtime_config.get("benchmark_name_scope", ""))
    if benchmark_name_template:
        benchmark_name = str(benchmark_name_template).format(seed=seed_value)
        if benchmark_name_scope == "benchmark_registry" and "benchmark_name" in benchmark_registry:
            benchmark_registry["artifact_benchmark_name"] = benchmark_name
        if benchmark_name_scope == "screening_pipeline" and "benchmark_name" in screening_registry:
            screening_registry["benchmark_name"] = benchmark_name

    result_catalog_template = runtime_config.get("result_catalog_template", "")
    if result_catalog_template:
        benchmark_registry["relative_paths"]["baseline_result_catalog"] = str(result_catalog_template).format(seed=seed_value)

    artifact_root_override = runtime_config.get("artifact_root", "")
    if artifact_root_override and "artifact_root" in screening_registry:
        screening_registry["artifact_root"] = artifact_root_override


def apply_transition_backbone_template(materialized_config: dict[str, object]) -> None:
    rollout_operator = materialized_config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    transition_backbone = dict(rollout_operator["transition_backbone"])
    template_source = str(transition_backbone.get("template_source", "")).strip()
    if not template_source:
        return
    if template_source not in transition_backbone:
        raise ValueError(f"Unknown transition_backbone template_source: {template_source}")
    rollout_operator["transition_backbone"] = copy.deepcopy(transition_backbone[template_source])


def main() -> None:
    args = parse_args()
    root_config = load_config(args.base_config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(root_config, args, Path(__file__), sys.argv)
    profile = load_experiment_family(args.profile, root_config)
    runtime_config = profile.get("runtime", {})
    base_config_path = resolve_repo_path(profile["base_config_path"])
    base_config = load_config(base_config_path)
    materialized_config = deep_merge(base_config, root_config)
    materialized_config = deep_merge(materialized_config, profile.get("config_overrides", {}))
    materialized_config["selected_paper_experiment_profile"] = args.profile
    materialized_config["selected_experiment_family"] = str(profile.get("family_name", args.profile))
    if profile.get("paper_track"):
        materialized_config["selected_paper_track"] = str(profile["paper_track"])
    if args.seed:
        seed_value = int(args.seed)
        apply_seed_overrides(materialized_config, runtime_config, seed_value)
        materialized_config["selected_formal_seed"] = seed_value
    apply_transition_backbone_template(materialized_config)

    output_path = Path(args.output_config)
    ensure_directory(output_path.parent)
    write_text_atomic(output_path, yaml.safe_dump(materialized_config, sort_keys=False))

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
