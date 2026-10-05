from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a production shared-latent checkpoint into a versioned inference bundle.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--model-version",
        default="",
        help="Optional serving model version override.",
    )
    return parser.parse_args()


def benchmark_artifact_dir(config: dict) -> Path:
    return Path(config["benchmark_registry"]["artifact_root"]).resolve()


def production_model_dir(config: dict) -> Path:
    return benchmark_artifact_dir(config) / config["benchmark_registry"]["benchmark_name"] / "production" / "shared_latent_dual_encoder_classifier"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    serving_config = config["serving_registry"]
    model_version = args.model_version or serving_config["default_model_version"]
    bundle_root = (REPO_ROOT / serving_config["production_root_template"].format(
        benchmark_name=config["benchmark_registry"]["benchmark_name"],
        model_version=model_version,
    )).resolve()
    ensure_directory(bundle_root)

    production_dir = production_model_dir(config)
    training_manifest_path = production_dir / "training_manifest.json"
    checkpoint_path = production_dir / "model_weights.pt"
    transcript_entity_path = production_dir / "transcriptomics_entity_index.parquet"
    metabolite_entity_path = production_dir / "metabolomics_entity_index.parquet"
    if not training_manifest_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Production artifact is incomplete under {production_dir}")

    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    transcript_entity_index = pd.read_parquet(transcript_entity_path)
    metabolite_entity_index = pd.read_parquet(metabolite_entity_path)
    entity_dictionary = pd.concat(
        [
            transcript_entity_index.assign(
                modality="transcriptomics",
                feature_namespace=serving_config["request_feature_namespace"]["transcriptomics"],
            ),
            metabolite_entity_index.assign(
                modality="metabolomics",
                feature_namespace=serving_config["request_feature_namespace"]["metabolomics"],
            ),
        ],
        ignore_index=True,
    )

    bundle_filenames = serving_config["bundle_filenames"]
    entity_dictionary_path = bundle_root / bundle_filenames["entity_dictionary"]
    model_weights_path = bundle_root / bundle_filenames["model_weights"]
    entity_dictionary.to_parquet(entity_dictionary_path, index=False, engine="pyarrow")
    shutil.copy2(checkpoint_path, model_weights_path)

    preprocessing_manifest = {
        "benchmark_name": config["benchmark_registry"]["benchmark_name"],
        "model_name": "shared_latent_dual_encoder_classifier",
        "model_version": model_version,
        "preprocessing_version": model_version,
        "supported_modalities": serving_config["supported_modalities"],
        "feature_namespace": serving_config["request_feature_namespace"],
        "minimum_recognized_feature_count": serving_config["minimum_recognized_feature_count"],
        "feature_count_by_modality": {
            "transcriptomics": int(transcript_entity_index.shape[0]),
            "metabolomics": int(metabolite_entity_index.shape[0]),
        },
        "training_manifest_path": str(training_manifest_path.relative_to(REPO_ROOT)),
    }
    label_dictionary = {
        "0": config["benchmark_registry"]["negative_labels"][0],
        "1": config["benchmark_registry"]["positive_labels"][0],
    }
    model_card = {
        "benchmark_name": config["benchmark_registry"]["benchmark_name"],
        "model_name": "shared_latent_dual_encoder_classifier",
        "model_version": model_version,
        "target_label_space": config["benchmark_registry"]["target_label_space"],
        "production_training": training_manifest,
        "feature_counts": preprocessing_manifest["feature_count_by_modality"],
    }

    write_json(bundle_root / bundle_filenames["preprocessing_manifest"], preprocessing_manifest)
    write_json(bundle_root / bundle_filenames["label_dictionary"], label_dictionary)
    write_json(bundle_root / bundle_filenames["model_card"], model_card)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
