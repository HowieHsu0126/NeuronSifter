from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.models.networks.shared_latent_dual_encoder import SharedLatentDualEncoder
from Libs.models.networks.tabular_jepa_auxiliary_encoder import TabularJEPAAuxiliaryEncoder
from Libs.utils.evaluation.alignment_benchmark import compute_binary_metrics
from Libs.utils.evaluation.benchmark_datasets import assemble_dataset, benchmark_program_config, fold_sample_split, load_benchmark_catalogs
from Libs.utils.data.raw_data_registry import load_config, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained benchmark models on held-out studies and write benchmark result tables.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--fold-name",
        default="",
        help="Optional benchmark fold name subset for smoke tests.",
    )
    parser.add_argument(
        "--disease-program",
        default="",
        help="Optional disease program override, for example ad_control.",
    )
    return parser.parse_args()


def resolve_output_root(config: dict) -> Path:
    return Path(config["benchmark_registry"]["artifact_root"]).resolve()


def artifact_dir(config: dict, scope: str, fold_name: str, model_name: str) -> Path:
    root = resolve_output_root(config)
    artifact_benchmark_name = config["benchmark_registry"].get("artifact_benchmark_name", config["benchmark_registry"]["benchmark_name"])
    if scope == "production":
        return root / artifact_benchmark_name / "production" / model_name
    return root / artifact_benchmark_name / "folds" / fold_name / model_name


def evaluate_logistic_model(model_path: Path, matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model = joblib.load(model_path)
    probabilities = model.predict_proba(matrix)[:, 1]
    return compute_binary_metrics(labels.astype(np.int64), probabilities.astype(np.float64)), probabilities.astype(np.float64)


def load_shared_latent_model(checkpoint_path: Path) -> SharedLatentDualEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SharedLatentDualEncoder(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_shared_latent_model(
    model_path: Path,
    modality: str,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    model = load_shared_latent_model(model_path)
    with torch.no_grad():
        logits, _ = model(modality, torch.tensor(matrix, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return compute_binary_metrics(labels.astype(np.int64), probabilities), probabilities


def load_tabular_jepa_model(checkpoint_path: Path) -> TabularJEPAAuxiliaryEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TabularJEPAAuxiliaryEncoder(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_tabular_jepa_model(
    model_path: Path,
    modality: str,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    model = load_tabular_jepa_model(model_path)
    with torch.no_grad():
        logits, _, _ = model(modality, torch.tensor(matrix, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return compute_binary_metrics(labels.astype(np.int64), probabilities), probabilities


def pooled_result_rows(
    benchmark_name: str,
    disease_program: str,
    grouped_predictions: dict[tuple[str, str], dict[str, list[np.ndarray]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model_name, evaluation_scope), payload in sorted(grouped_predictions.items()):
        y_true = np.concatenate(payload["y_true"], axis=0)
        y_score = np.concatenate(payload["y_score"], axis=0)
        metrics = compute_binary_metrics(y_true.astype(np.int64), y_score.astype(np.float64))
        unique_modalities = sorted(set(payload["modalities"]))
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "disease_program": disease_program,
                "baseline_family": (
                    "classical_linear"
                    if model_name in {"aligned_gene_logistic_regression", "aligned_metabolite_logistic_regression"}
                    else "internal_lightweight"
                ),
                "fold_name": "",
                "model_name": model_name,
                "held_out_study_uid": "",
                "evaluation_scope": evaluation_scope,
                "modality": unique_modalities[0] if len(unique_modalities) == 1 else "mixed",
                "sample_count": int(y_true.shape[0]),
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "brier_score": metrics["brier_score"],
                "headline_metric_flag": "yes" if evaluation_scope == "headline_pooled" else "no",
                "artifact_path": "",
                "case_study_name": "",
                "scale_classification": "",
                "is_case_study_anchor": "no",
                "note": "pooled_summary",
            }
        )
    return rows


def metric_field_names(metrics: dict[str, float]) -> dict[str, object]:
    return {metric_name: metric_value for metric_name, metric_value in metrics.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    disease_program = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    _, fold_rows, _ = load_benchmark_catalogs(config)
    target_fold_rows = [row for row in fold_rows if not args.fold_name or row["fold_name"] == args.fold_name]
    if target_fold_rows:
        active_benchmark_name = str(target_fold_rows[0]["benchmark_name"])
        config["benchmark_registry"]["benchmark_name"] = active_benchmark_name
        config["benchmark_registry"]["artifact_benchmark_name"] = active_benchmark_name
    benchmark_name = config["benchmark_registry"]["benchmark_name"]
    result_rows: list[dict[str, object]] = []
    grouped_predictions: dict[tuple[str, str], dict[str, list[np.ndarray] | list[str]]] = defaultdict(
        lambda: {"y_true": [], "y_score": [], "modalities": []}
    )

    for fold_row in target_fold_rows:
        fold_start_count = len(result_rows)
        fold_logger = pipeline_logger.bind(fold=fold_row["fold_name"])
        split_frame = fold_sample_split(config, fold_row["fold_name"])
        test_frame = split_frame.loc[split_frame["split_role"] == "test"].copy()
        if test_frame.empty:
            fold_logger.progress("fold_evaluation_skipped", reason="empty_test_split")
            continue
        held_out_modality = str(test_frame["modality"].iloc[0])
        headline_scope = "headline_pooled" if fold_row["headline_metric_flag"] == "yes" else "auxiliary_external_pooled"
        fold_logger.artifact(
            "fold_evaluation_started",
            held_out_study_uid=fold_row["held_out_study_uid"],
            modality=held_out_modality,
            evaluation_scope=headline_scope,
            test_sample_count=int(test_frame.shape[0]),
        )
        if held_out_modality == "transcriptomics":
            logistic_dir = artifact_dir(config, "fold", fold_row["fold_name"], "aligned_gene_logistic_regression")
            model_path = logistic_dir / "model.joblib"
            entity_index_path = logistic_dir / "entity_index.parquet"
            if model_path.exists() and entity_index_path.exists():
                entity_index = pd.read_parquet(entity_index_path)
                matrix, labels, _ = assemble_dataset(config, test_frame, "transcriptomics", entity_index)
                if matrix.shape[0] > 0:
                    metrics, probabilities = evaluate_logistic_model(model_path, matrix, labels)
                    fold_logger.metric(
                        "fold_metric_summary",
                        model_name="aligned_gene_logistic_regression",
                        modality="transcriptomics",
                        evaluation_scope="held_out_study",
                        sample_count=int(labels.shape[0]),
                        artifact=model_path,
                        **metric_field_names(metrics),
                    )
                    result_rows.append(
                        {
                            "benchmark_name": benchmark_name,
                            "disease_program": disease_program,
                            "baseline_family": "classical_linear",
                            "fold_name": fold_row["fold_name"],
                            "model_name": "aligned_gene_logistic_regression",
                            "held_out_study_uid": fold_row["held_out_study_uid"],
                            "evaluation_scope": "held_out_study",
                            "modality": "transcriptomics",
                            "sample_count": int(labels.shape[0]),
                            "auroc": metrics["auroc"],
                            "auprc": metrics["auprc"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "brier_score": metrics["brier_score"],
                            "headline_metric_flag": fold_row["headline_metric_flag"],
                            "artifact_path": str(model_path.relative_to(REPO_ROOT)),
                            "case_study_name": "",
                            "scale_classification": "",
                            "is_case_study_anchor": "no",
                            "note": fold_row["held_out_study_uid"],
                        }
                    )
                    grouped_predictions[("aligned_gene_logistic_regression", headline_scope)]["y_true"].append(labels.astype(np.int64))
                    grouped_predictions[("aligned_gene_logistic_regression", headline_scope)]["y_score"].append(probabilities.astype(np.float64))
                    grouped_predictions[("aligned_gene_logistic_regression", headline_scope)]["modalities"].append("transcriptomics")
        if held_out_modality == "metabolomics":
            logistic_dir = artifact_dir(config, "fold", fold_row["fold_name"], "aligned_metabolite_logistic_regression")
            model_path = logistic_dir / "model.joblib"
            entity_index_path = logistic_dir / "entity_index.parquet"
            if model_path.exists() and entity_index_path.exists():
                entity_index = pd.read_parquet(entity_index_path)
                matrix, labels, _ = assemble_dataset(config, test_frame, "metabolomics", entity_index)
                if matrix.shape[0] > 0:
                    metrics, probabilities = evaluate_logistic_model(model_path, matrix, labels)
                    fold_logger.metric(
                        "fold_metric_summary",
                        model_name="aligned_metabolite_logistic_regression",
                        modality="metabolomics",
                        evaluation_scope="held_out_study",
                        sample_count=int(labels.shape[0]),
                        artifact=model_path,
                        **metric_field_names(metrics),
                    )
                    result_rows.append(
                        {
                            "benchmark_name": benchmark_name,
                            "disease_program": disease_program,
                            "baseline_family": "classical_linear",
                            "fold_name": fold_row["fold_name"],
                            "model_name": "aligned_metabolite_logistic_regression",
                            "held_out_study_uid": fold_row["held_out_study_uid"],
                            "evaluation_scope": "held_out_study",
                            "modality": "metabolomics",
                            "sample_count": int(labels.shape[0]),
                            "auroc": metrics["auroc"],
                            "auprc": metrics["auprc"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "brier_score": metrics["brier_score"],
                            "headline_metric_flag": fold_row["headline_metric_flag"],
                            "artifact_path": str(model_path.relative_to(REPO_ROOT)),
                            "case_study_name": "",
                            "scale_classification": "",
                            "is_case_study_anchor": "no",
                            "note": fold_row["held_out_study_uid"],
                        }
                    )
                    grouped_predictions[("aligned_metabolite_logistic_regression", headline_scope)]["y_true"].append(labels.astype(np.int64))
                    grouped_predictions[("aligned_metabolite_logistic_regression", headline_scope)]["y_score"].append(probabilities.astype(np.float64))
                    grouped_predictions[("aligned_metabolite_logistic_regression", headline_scope)]["modalities"].append("metabolomics")

        shared_dir = artifact_dir(config, "fold", fold_row["fold_name"], "shared_latent_dual_encoder_classifier")
        model_path = shared_dir / "model_weights.pt"
        if model_path.exists():
            modality = held_out_modality
            if modality == "transcriptomics":
                entity_index_path = shared_dir / "transcriptomics_entity_index.parquet"
            else:
                entity_index_path = shared_dir / "metabolomics_entity_index.parquet"
            if entity_index_path.exists():
                entity_index = pd.read_parquet(entity_index_path)
                matrix, labels, _ = assemble_dataset(config, test_frame, modality, entity_index)
                if matrix.shape[0] > 0:
                    metrics, probabilities = evaluate_shared_latent_model(model_path, modality, matrix, labels)
                    fold_logger.metric(
                        "fold_metric_summary",
                        model_name="shared_latent_dual_encoder_classifier",
                        modality=modality,
                        evaluation_scope="held_out_study",
                        sample_count=int(labels.shape[0]),
                        artifact=model_path,
                        **metric_field_names(metrics),
                    )
                    result_rows.append(
                        {
                            "benchmark_name": benchmark_name,
                            "disease_program": disease_program,
                            "baseline_family": "internal_lightweight",
                            "fold_name": fold_row["fold_name"],
                            "model_name": "shared_latent_dual_encoder_classifier",
                            "held_out_study_uid": fold_row["held_out_study_uid"],
                            "evaluation_scope": "held_out_study",
                            "modality": modality,
                            "sample_count": int(labels.shape[0]),
                            "auroc": metrics["auroc"],
                            "auprc": metrics["auprc"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "brier_score": metrics["brier_score"],
                            "headline_metric_flag": fold_row["headline_metric_flag"],
                            "artifact_path": str(model_path.relative_to(REPO_ROOT)),
                            "case_study_name": "",
                            "scale_classification": "",
                            "is_case_study_anchor": "no",
                            "note": fold_row["held_out_study_uid"],
                        }
                    )
                    grouped_predictions[("shared_latent_dual_encoder_classifier", headline_scope)]["y_true"].append(labels.astype(np.int64))
                    grouped_predictions[("shared_latent_dual_encoder_classifier", headline_scope)]["y_score"].append(probabilities.astype(np.float64))
                    grouped_predictions[("shared_latent_dual_encoder_classifier", headline_scope)]["modalities"].append(modality)

        jepa_dir = artifact_dir(config, "fold", fold_row["fold_name"], "tabular_jepa_auxiliary_encoder")
        model_path = jepa_dir / "model_weights.pt"
        if model_path.exists():
            modality = held_out_modality
            if modality == "transcriptomics":
                entity_index_path = jepa_dir / "transcriptomics_entity_index.parquet"
            else:
                entity_index_path = jepa_dir / "metabolomics_entity_index.parquet"
            if entity_index_path.exists():
                entity_index = pd.read_parquet(entity_index_path)
                matrix, labels, _ = assemble_dataset(config, test_frame, modality, entity_index)
                if matrix.shape[0] > 0:
                    metrics, probabilities = evaluate_tabular_jepa_model(model_path, modality, matrix, labels)
                    fold_logger.metric(
                        "fold_metric_summary",
                        model_name="tabular_jepa_auxiliary_encoder",
                        modality=modality,
                        evaluation_scope="held_out_study",
                        sample_count=int(labels.shape[0]),
                        artifact=model_path,
                        **metric_field_names(metrics),
                    )
                    result_rows.append(
                        {
                            "benchmark_name": benchmark_name,
                            "disease_program": disease_program,
                            "baseline_family": "internal_lightweight",
                            "fold_name": fold_row["fold_name"],
                            "model_name": "tabular_jepa_auxiliary_encoder",
                            "held_out_study_uid": fold_row["held_out_study_uid"],
                            "evaluation_scope": "held_out_study",
                            "modality": modality,
                            "sample_count": int(labels.shape[0]),
                            "auroc": metrics["auroc"],
                            "auprc": metrics["auprc"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "brier_score": metrics["brier_score"],
                            "headline_metric_flag": fold_row["headline_metric_flag"],
                            "artifact_path": str(model_path.relative_to(REPO_ROOT)),
                            "case_study_name": "",
                            "scale_classification": "",
                            "is_case_study_anchor": "no",
                            "note": fold_row["held_out_study_uid"],
                        }
                    )
                    grouped_predictions[("tabular_jepa_auxiliary_encoder", headline_scope)]["y_true"].append(labels.astype(np.int64))
                    grouped_predictions[("tabular_jepa_auxiliary_encoder", headline_scope)]["y_score"].append(probabilities.astype(np.float64))
                    grouped_predictions[("tabular_jepa_auxiliary_encoder", headline_scope)]["modalities"].append(modality)
        fold_logger.complete(
            "fold_evaluation_completed",
            result_row_count=len(result_rows) - fold_start_count,
            held_out_study_uid=fold_row["held_out_study_uid"],
            modality=held_out_modality,
        )

    pooled_rows = pooled_result_rows(benchmark_name, disease_program, grouped_predictions)
    for pooled_row in pooled_rows:
        pipeline_logger.metric(
            "pooled_metric_summary",
            model_name=pooled_row["model_name"],
            modality=pooled_row["modality"],
            evaluation_scope=pooled_row["evaluation_scope"],
            sample_count=pooled_row["sample_count"],
            auroc=pooled_row["auroc"],
            auprc=pooled_row["auprc"],
            balanced_accuracy=pooled_row["balanced_accuracy"],
            macro_f1=pooled_row["macro_f1"],
            brier_score=pooled_row["brier_score"],
        )
    result_rows.extend(pooled_rows)
    result_rows.sort(key=lambda row: (row["model_name"], row["fold_name"], row["evaluation_scope"], row["modality"]))
    result_catalog_path = (
        (REPO_ROOT / config["benchmark_registry"]["local_root"]).resolve()
        / config["benchmark_registry"]["relative_paths"]["baseline_result_catalog"]
    )
    write_tsv(
        result_catalog_path,
        config["schemas"]["benchmark_result_fields"],
        result_rows,
    )
    pipeline_logger.artifact(
        "shared_latent_benchmark_written",
        artifact=result_catalog_path,
        result_row_count=len(result_rows),
        fold_count=len(target_fold_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=result_catalog_path,
        result_row_count=len(result_rows),
        fold_count=len(target_fold_rows),
    )

if __name__ == "__main__":
    main()
