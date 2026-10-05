from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config
from Libs.utils.model.dynamics_fields import (
    batch_simulation_targets,
    observed_region_mask,
    quantitative_case_validation_rows,
    route_ordering_exposure,
    selected_regimen_simulation_tensors,
    strict_case_dynamics_metrics,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.foundation_twin_capabilities import capability_catalog_rows
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_tsv
from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config
from Libs.utils.evaluation.screening_datasets import (
    fold_patient_latents,
    patient_regimen_labels,
    patient_target_labels,
    regimen_features,
    target_program_features,
    target_uid_to_entity_lookup,
)
from Libs.utils.model.screening_core import pkpd_prior_loss, pkpd_reference_tensors, tri_context_tensors
from Libs.utils.evaluation.screening_uq import (
    acceptance_benefit,
    adaptive_calibration_error,
    binary_brier,
    binary_nll,
    ensemble_uncertainty_summary,
    coverage_gap,
    expected_calibration_error,
    fit_temperature_from_grid,
    ood_auroc,
    risk_coverage_curve,
    sigmoid_with_temperature,
)
from Libs.utils.evaluation.strict_case_discovery import (
    assert_clean_action_ready_manifest,
    assert_no_forced_anchor_regimens,
    compute_anchor_recovery_metrics,
    compute_tri_accepted_anchor_metrics,
    regimen_anchor_mask,
    strict_case_entries,
    target_anchor_mask,
)


SCREENING_TASK_LOOKUP: dict[str, tuple[str, str]] = {}


def disease_family_name(disease_program: str) -> str:
    normalized = disease_program.strip().lower()
    if normalized.startswith("ad_"):
        return "AD"
    if normalized.startswith("pd_"):
        return "PD"
    if normalized.startswith("ftd_"):
        return "FTD"
    if normalized.startswith("als_"):
        return "ALS"
    return "pan_neuro"


def optional_tsv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the dynamics-grounded NeuronSifter AD screening system.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--fold-name", default="", help="Optional fold name subset.")
    parser.add_argument("--disease-program", default="", help="Optional disease program override.")
    parser.add_argument("--ensemble-mode", default="auto", choices=["auto", "single", "required"])
    return parser.parse_args()


def artifact_dir(config: dict, fold_name: str) -> Path:
    root = Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
    return root / config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] / "folds" / fold_name / "ad_screening_system"


def load_model(checkpoint_path: Path, config: dict) -> ADScreeningSystem:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = dict(checkpoint["model_kwargs"])
    model_kwargs.update(
        {
            key: value
            for key, value in screening_model_kwargs_from_config(config).items()
            if key not in model_kwargs or model_kwargs[key] in (None, [], {})
        }
    )
    model = ADScreeningSystem(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model


def load_training_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def discovered_ensemble_checkpoint_paths(
    config: dict,
    fold_name: str,
    current_checkpoint_path: Path,
    ensemble_mode: str,
) -> list[Path]:
    if ensemble_mode == "single":
        return [current_checkpoint_path]
    ensemble_config = config["screening_pipeline"]["screening_benchmark"]["ensemble_uq"]
    discovery_config = ensemble_config["current_mainline_discovery"]
    allowed_fold_names = {str(value) for value in discovery_config["allowed_fold_names"]}
    if fold_name not in allowed_fold_names:
        if ensemble_mode == "required":
            raise ValueError(f"Fold {fold_name} is not enabled for current-mainline ensemble discovery.")
        return [current_checkpoint_path]
    benchmark_name_template = str(discovery_config["benchmark_name_template"])
    member_seed_values = [int(value) for value in ensemble_config["member_seed_values"]]
    allowed_benchmark_names = {
        benchmark_name_template.format(seed=seed_value)
        for seed_value in member_seed_values
    }
    current_benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    if current_benchmark_name not in allowed_benchmark_names:
        if ensemble_mode == "required":
            raise ValueError(
                f"Benchmark {current_benchmark_name} does not match current-mainline discovery template {benchmark_name_template}."
            )
        return [current_checkpoint_path]
    artifact_root = (REPO_ROOT / str(discovery_config["artifact_root"])).resolve()
    manifest_relative_path = str(discovery_config["manifest_relative_path"])
    checkpoint_relative_path = str(discovery_config["checkpoint_relative_path"])
    expected_model_name = str(discovery_config["expected_model_name"])
    expected_model_contract_version = str(discovery_config["expected_model_contract_version"])
    discovered_paths: list[Path] = []
    for seed_value in member_seed_values:
        benchmark_name = benchmark_name_template.format(seed=seed_value)
        manifest_path = artifact_root / benchmark_name / manifest_relative_path.format(fold_name=fold_name)
        checkpoint_path = artifact_root / benchmark_name / checkpoint_relative_path.format(fold_name=fold_name)
        if not manifest_path.exists() or not checkpoint_path.exists():
            continue
        manifest = load_training_manifest(manifest_path)
        if str(manifest.get("fold_name", "")) != fold_name:
            continue
        if str(manifest.get("benchmark_name", "")) != benchmark_name:
            continue
        if str(manifest.get("model_name", "")) != expected_model_name:
            continue
        if str(manifest.get("model_contract_version", "")) != expected_model_contract_version:
            continue
        discovered_paths.append(checkpoint_path)
    minimum_member_count = int(ensemble_config["minimum_member_count"])
    if len(discovered_paths) >= minimum_member_count:
        return discovered_paths
    if ensemble_mode == "required":
        raise ValueError(
            f"Current-mainline ensemble discovery for fold {fold_name} found {len(discovered_paths)} members, expected at least {minimum_member_count}."
        )
    return [current_checkpoint_path]


def final_regimen_output(
    model: ADScreeningSystem,
    patient_state: torch.Tensor,
    regimen_feature_tensor: torch.Tensor,
    observation_mask: torch.Tensor,
    tri_context: dict[str, torch.Tensor],
    exposure_target_tensor: torch.Tensor,
    occupancy_target_tensor: torch.Tensor,
    pkpd_feasibility_tensor: torch.Tensor,
    config: dict,
    regimen_chunk_size: int,
) -> dict[str, torch.Tensor]:
    initial_output = model.score_regimens(
        patient_state,
        regimen_feature_tensor,
        tri_context,
        observation_mask=observation_mask,
        regimen_chunk_size=regimen_chunk_size,
        include_auxiliary_outputs=False,
    )
    _, pkpd_proxy_vector = pkpd_prior_loss(
        initial_output,
        exposure_target_tensor,
        occupancy_target_tensor,
        pkpd_feasibility_tensor,
        config,
    )
    expanded_tri_context = dict(tri_context)
    expanded_tri_context["pkpd_physics_consistency"] = pkpd_proxy_vector.unsqueeze(0).expand(
        patient_state.shape[0],
        -1,
    ).to(dtype=torch.float32)
    return model.score_regimens(
        patient_state,
        regimen_feature_tensor,
        expanded_tri_context,
        observation_mask=observation_mask,
        regimen_chunk_size=regimen_chunk_size,
        include_auxiliary_outputs=False,
    )


def ensemble_probability_summary(member_score_tensor: np.ndarray) -> dict[str, np.ndarray]:
    member_probabilities = 1.0 / (1.0 + np.exp(-member_score_tensor.astype(np.float64)))
    member_count, sample_count, candidate_count = member_probabilities.shape
    summary = ensemble_uncertainty_summary(member_probabilities.reshape(member_count, sample_count * candidate_count))
    return {
        key: value.reshape(sample_count, candidate_count)
        for key, value in summary.items()
    }


def stable_split(value: str, fraction: float) -> bool:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) % 1000) / 1000.0 < fraction


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    gains = (2 ** y_true[order] - 1) / np.log2(np.arange(order.size) + 2)
    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = (2 ** y_true[ideal_order] - 1) / np.log2(np.arange(ideal_order.size) + 2)
    denominator = ideal_gains.sum()
    if denominator <= 0:
        return 0.0
    return float(gains.sum() / denominator)


def result_row(
    benchmark_name: str,
    disease_program: str,
    split_name: str,
    evaluation_scope: str,
    model_name: str,
    modality: str,
    sample_count: int,
    metric_name: str,
    metric_value: float,
    artifact_path: str,
    note: str,
    task_name: str = "",
    task_family: str = "",
    case_study_name: str = "",
    is_case_study_anchor: str = "no",
) -> dict[str, object]:
    resolved_task_name, resolved_task_family = SCREENING_TASK_LOOKUP.get(evaluation_scope, ("", ""))
    task_name = task_name or resolved_task_name
    task_family = task_family or resolved_task_family
    return {
        "benchmark_name": benchmark_name,
        "disease_program": disease_program,
        "baseline_family": "action_conditioned_twin",
        "split_name": split_name,
        "evaluation_scope": evaluation_scope,
        "task_name": task_name,
        "task_family": task_family,
        "model_name": model_name,
        "modality": modality,
        "sample_count": sample_count,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "artifact_path": artifact_path,
        "case_study_name": case_study_name,
        "scale_classification": "",
        "is_case_study_anchor": is_case_study_anchor,
        "note": note,
    }


def strict_case_result_rows(
    benchmark_name: str,
    disease_program: str,
    split_name: str,
    evaluation_scope: str,
    model_name: str,
    modality: str,
    sample_count: int,
    artifact_path: str,
    note: str,
    case_study_name: str,
    metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        result_row(
            benchmark_name,
            disease_program,
            split_name,
            evaluation_scope,
            model_name,
            modality,
            sample_count,
            metric_name,
            metric_value,
            artifact_path,
            note,
            case_study_name=case_study_name,
            is_case_study_anchor="",
        )
        for metric_name, metric_value in metrics.items()
    ]


def normalized_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = np.maximum(np.abs(target).mean(), 1e-6)
    return float(np.sqrt(np.mean((prediction - target) ** 2)) / denominator)


def align_time_horizon(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.ndim != target_array.ndim:
        raise ValueError(
            f"Prediction and target must have the same rank for horizon alignment, got {prediction_array.shape} and {target_array.shape}."
        )
    if prediction_array.ndim < 3:
        return prediction_array, target_array
    shared_horizon = min(prediction_array.shape[2], target_array.shape[2])
    if shared_horizon <= 0:
        raise ValueError(
            f"Prediction and target must share a positive rollout horizon, got {prediction_array.shape} and {target_array.shape}."
        )
    return prediction_array[:, :, :shared_horizon, ...], target_array[:, :, :shared_horizon, ...]


def patient_state_array(payload: dict[str, object]) -> np.ndarray:
    if "state_features" not in payload:
        raise KeyError("Evaluation payload is missing state_features under the multiscale patient-state contract.")
    return np.asarray(payload["state_features"], dtype=np.float32)


def regimen_chunk_size(config: dict) -> int:
    return int(config["screening_pipeline"]["screening_benchmark"].get("execution_controls", {}).get("regimen_chunk_size", 0))


def screening_task_lookup(config: dict) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    task_catalog = config["screening_pipeline"]["screening_benchmark"].get("paper_tasks", {})
    for task_name, task_payload in task_catalog.items():
        task_family = str(task_payload.get("task_family", ""))
        for evaluation_scope in task_payload.get("evaluation_scopes", []):
            lookup[str(evaluation_scope)] = (str(task_name), task_family)
    return lookup


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    global SCREENING_TASK_LOOKUP
    SCREENING_TASK_LOOKUP = screening_task_lookup(config)
    disease_program = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    config["screening_pipeline"]["screening_benchmark"]["active_disease_program"] = disease_program
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    target_fraction = float(config["screening_pipeline"]["screening_benchmark"]["target_family_holdout_fraction"])
    scaffold_fraction = float(config["screening_pipeline"]["screening_benchmark"]["scaffold_holdout_fraction"])
    tri_threshold = float(config["screening_pipeline"]["screening_benchmark"]["tri_threshold"])
    uq_config = config["screening_pipeline"]["screening_benchmark"]["uq"]
    temperature_grid = [float(value) for value in uq_config["temperature_grid"]]
    ece_bin_count = int(uq_config["ece_bin_count"])
    ace_bin_count = int(uq_config["ace_bin_count"])
    risk_coverage_grid_size = int(uq_config["risk_coverage_grid_size"])
    target_acceptance_coverage = float(uq_config["target_acceptance_coverage"])
    uncertainty_metric_name = str(
        config["screening_pipeline"]["screening_benchmark"]["ensemble_uq"]["current_mainline_discovery"]["uncertainty_metric"]
    )
    action_chunk_size = regimen_chunk_size(config)

    action_root = Path(config["screening_pipeline"]["action_ready_registry"]["local_root"]).resolve()
    target_program_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["target_program_catalog"])
    drug_target_root = Path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]).resolve()
    target_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"])
    compound_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"])
    intervention_root = Path(config["twin"]["intervention_program_registry"]["local_root"]).resolve()
    intervention_rows = read_tsv(intervention_root / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"])
    intervention_row_by_regimen = {
        str(row["regimen_uid"]): row
        for row in intervention_rows
    }
    target_uid_to_entity = target_uid_to_entity_lookup(target_rows)
    manifest = assert_clean_action_ready_manifest(config)
    strict_case_rows = strict_case_entries(config, disease_program)
    pipeline_logger.progress(
        "action_ready_manifest_validated",
        config_hash=str(manifest["config_hash"]),
        strict_case_count=len(strict_case_rows),
    )

    fold_rows = [
        row
        for row in read_tsv(Path(config["benchmark_registry"]["local_root"]).resolve() / config["benchmark_registry"]["relative_paths"]["fold_catalog"])
        if row.get("disease_program", disease_program) == disease_program
    ]
    result_rows: list[dict[str, object]] = []
    candidate_priority_rows: list[dict[str, object]] = []
    capability_rows: list[dict[str, object]] = []
    world_state_frame = optional_tsv_frame(
        Path(config["data"]["world_state_ready_registry"]["local_root"]).resolve()
        / config["data"]["world_state_ready_registry"]["relative_paths"]["world_state_catalog"]
    )
    crossmodal_anchor_frame = optional_tsv_frame(
        Path(config["data"]["crossmodal_anchor_ready_registry"]["local_root"]).resolve()
        / config["data"]["crossmodal_anchor_ready_registry"]["relative_paths"]["crossmodal_anchor_catalog"]
    )
    simulation_teacher_frame = optional_tsv_frame(
        Path(config["data"]["simulation_teacher_ready_registry"]["local_root"]).resolve()
        / config["data"]["simulation_teacher_ready_registry"]["relative_paths"]["simulation_teacher_catalog"]
    )
    counterfactual_benchmark_frame = optional_tsv_frame(
        Path(config["data"]["counterfactual_benchmark_ready_registry"]["local_root"]).resolve()
        / config["data"]["counterfactual_benchmark_ready_registry"]["relative_paths"]["counterfactual_benchmark_catalog"]
    )
    feedback_event_frame = optional_tsv_frame(
        Path(config["data"]["feedback_ready_registry"]["local_root"]).resolve()
        / config["data"]["feedback_ready_registry"]["relative_paths"]["feedback_event_catalog"]
    )
    model_name_for_results = "ad_screening_system"
    for fold_row in fold_rows:
        if args.fold_name and fold_row["fold_name"] != args.fold_name:
            continue
        config["benchmark_registry"]["benchmark_name"] = str(fold_row["benchmark_name"])
        current_artifact_dir = artifact_dir(config, fold_row["fold_name"])
        checkpoint_path = current_artifact_dir / "model_weights.pt"
        if not checkpoint_path.exists():
            continue
        ensemble_checkpoint_paths = discovered_ensemble_checkpoint_paths(
            config,
            fold_row["fold_name"],
            checkpoint_path,
            args.ensemble_mode,
        )
        target_frame = pd.read_parquet(current_artifact_dir / "selected_target_programs.parquet")
        regimen_frame = pd.read_parquet(current_artifact_dir / "selected_regimens.parquet")
        for case_row in strict_case_rows:
            assert_no_forced_anchor_regimens(regimen_frame, case_row, compound_rows, f"{fold_row['fold_name']}:{case_row['case_study_name']}")
        target_features_frame, target_feature_matrix = target_program_features(target_frame.to_dict("records"))
        regimen_features_frame, regimen_feature_matrix = regimen_features(regimen_frame.to_dict("records"))
        target_feature_tensor = torch.tensor(target_feature_matrix, dtype=torch.float32)
        regimen_feature_tensor = torch.tensor(regimen_feature_matrix, dtype=torch.float32)
        models = [load_model(member_checkpoint_path, config) for member_checkpoint_path in ensemble_checkpoint_paths]
        ensemble_artifact_path = str(ensemble_checkpoint_paths[0].relative_to(REPO_ROOT))
        ensemble_member_count = len(models)

        simulation_targets_by_condition = selected_regimen_simulation_tensors(config, regimen_features_frame)
        exposure_target_tensor, occupancy_target_tensor, pkpd_feasibility_tensor = pkpd_reference_tensors(regimen_features_frame, config, torch.device("cpu"))
        train_payload = fold_patient_latents(config, fold_row["fold_name"], "train", disease_program, fold_row["benchmark_name"])
        validation_payload = fold_patient_latents(config, fold_row["fold_name"], "validation", disease_program, fold_row["benchmark_name"])
        test_payload = fold_patient_latents(config, fold_row["fold_name"], "test", disease_program, fold_row["benchmark_name"])

        for modality, payload in test_payload.items():
            patient_state_matrix = patient_state_array(payload)
            if patient_state_matrix.shape[0] == 0:
                continue
            patient_state = torch.tensor(patient_state_matrix, dtype=torch.float32)
            patient_labels = payload["labels"].astype(np.int64)
            observation_mask = torch.tensor(observed_region_mask(payload["metadata"], config), dtype=torch.float32)

            validation_state_matrix = patient_state_array(validation_payload[modality])
            if validation_state_matrix.shape[0] > 0:
                validation_observation_mask = torch.tensor(
                    observed_region_mask(validation_payload[modality]["metadata"], config),
                    dtype=torch.float32,
                )
                validation_logits_by_member = [
                    model.infer_state(
                        torch.tensor(validation_state_matrix, dtype=torch.float32),
                        observation_mask=validation_observation_mask,
                    )[0].detach().cpu().numpy().astype(np.float64)
                    for model in models
                ]
                validation_logits_numpy = np.mean(np.stack(validation_logits_by_member, axis=0), axis=0)
                best_temperature, _ = fit_temperature_from_grid(
                    validation_logits_numpy,
                    validation_payload[modality]["labels"],
                    temperature_grid,
                )
            else:
                best_temperature = 1.0

            tri_context = tri_context_tensors(
                payload["probabilities"],
                payload["matrix"],
                payload["latent"],
                regimen_feature_matrix.shape[0],
                torch.device("cpu"),
            )
            member_disease_logits: list[np.ndarray] = []
            member_target_scores: list[np.ndarray] = []
            member_regimen_scores: list[np.ndarray] = []
            member_field_trajectories: list[np.ndarray] = []
            member_tri_scores: list[np.ndarray] = []
            member_pkpd_consistency: list[np.ndarray] = []
            member_interactive_latency_ms: list[float] = []
            for model in models:
                started_at = time.perf_counter()
                disease_logit, _ = model.infer_state(patient_state, observation_mask=observation_mask)
                member_disease_logits.append(disease_logit.detach().cpu().numpy().astype(np.float64))
                target_scores, _ = model.score_targets(
                    patient_state,
                    target_feature_tensor,
                    observation_mask=observation_mask,
                )
                member_target_scores.append(target_scores.detach().cpu().numpy().astype(np.float64))
                regimen_output = final_regimen_output(
                    model,
                    patient_state,
                    regimen_feature_tensor,
                    observation_mask,
                    tri_context,
                    exposure_target_tensor,
                    occupancy_target_tensor,
                    pkpd_feasibility_tensor,
                    config,
                    action_chunk_size,
                )
                member_regimen_scores.append(regimen_output["regimen_scores"].detach().cpu().numpy().astype(np.float64))
                member_field_trajectories.append(regimen_output["field_trajectory"].detach().cpu().numpy().astype(np.float32))
                member_tri_scores.append(regimen_output["tri_score"].detach().cpu().numpy().astype(np.float64))
                member_pkpd_consistency.append(regimen_output["pkpd_consistency"].detach().cpu().numpy().astype(np.float64))
                member_interactive_latency_ms.append((time.perf_counter() - started_at) * 1000.0)

            disease_logit_tensor = np.stack(member_disease_logits, axis=0)
            disease_probability_tensor = 1.0 / (1.0 + np.exp(-disease_logit_tensor))
            disease_uq = ensemble_uncertainty_summary(disease_probability_tensor)
            disease_prob = disease_uq["predictive_mean"].astype(np.float32)
            calibrated_disease_prob = sigmoid_with_temperature(
                disease_logit_tensor.mean(axis=0),
                best_temperature,
            ).astype(np.float32)
            latent_ood_metric = 0.0

            if len(np.unique(patient_labels)) >= 2:
                disease_metrics = {
                    "auroc": float(roc_auc_score(patient_labels, disease_prob)),
                    "auprc": float(average_precision_score(patient_labels, disease_prob)),
                    "balanced_accuracy": float(balanced_accuracy_score(patient_labels, disease_prob >= 0.5)),
                    "macro_f1": float(f1_score(patient_labels, disease_prob >= 0.5, average="macro")),
                }
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "disease_state_inference", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in disease_metrics.items()
                    ]
                )
                calibration_metrics = {
                    "temperature": float(best_temperature),
                    "ece": expected_calibration_error(calibrated_disease_prob, patient_labels, ece_bin_count),
                    "ace": adaptive_calibration_error(calibrated_disease_prob, patient_labels, ace_bin_count),
                    "nll": binary_nll(calibrated_disease_prob, patient_labels),
                    "brier_score": binary_brier(calibrated_disease_prob, patient_labels),
                }
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "disease_state_calibration", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in calibration_metrics.items()
                    ]
                )
                disease_ensemble_metrics = {
                    "ensemble_member_count": float(ensemble_member_count),
                    "predictive_entropy_mean": float(disease_uq["predictive_entropy"].mean()),
                    "mutual_information_mean": float(disease_uq["mutual_information"].mean()),
                    "epistemic_std_mean": float(disease_uq["epistemic_std"].mean()),
                }
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "disease_state_ensemble_uq", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in disease_ensemble_metrics.items()
                    ]
                )
                latent_ood_metric = ood_auroc(
                    np.asarray(train_payload[modality]["latent"], dtype=np.float32),
                    np.asarray(validation_payload[modality]["latent"], dtype=np.float32),
                    np.asarray(payload["latent"], dtype=np.float32),
                )
                result_rows.append(
                    result_row(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "ood_robustness",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        "latent_distance_ood_auroc",
                        float(latent_ood_metric),
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                    )
                )

            entity_lookup = {
                str(entity_id): index
                for index, entity_id in enumerate(payload["entity_index"]["canonical_entity_id"].astype(str).tolist())
            }
            target_labels = patient_target_labels(payload["matrix"], payload["differential"], entity_lookup, target_features_frame)
            regimen_labels = patient_regimen_labels(payload["matrix"], payload["differential"], entity_lookup, regimen_features_frame, target_uid_to_entity)
            target_score_tensor = np.stack(member_target_scores, axis=0)
            regimen_score_tensor = np.stack(member_regimen_scores, axis=0)
            field_trajectory_tensor = np.stack(member_field_trajectories, axis=0)
            tri_score_tensor = np.stack(member_tri_scores, axis=0)
            pkpd_consistency_tensor = np.stack(member_pkpd_consistency, axis=0)
            target_scores = target_score_tensor.mean(axis=0)
            regimen_scores = regimen_score_tensor.mean(axis=0)
            predicted_field = field_trajectory_tensor.mean(axis=0)
            tri_values = tri_score_tensor.mean(axis=0).astype(np.float32)
            pkpd_consistency_values = pkpd_consistency_tensor.mean(axis=0).astype(np.float32)
            regimen_uq = ensemble_probability_summary(regimen_score_tensor)
            interactive_latency_ms = float(np.mean(member_interactive_latency_ms)) if member_interactive_latency_ms else 0.0

            target_ndcg_at_10 = 0.0
            target_score_mean = target_scores.mean(axis=0)
            target_label_mean = target_labels.mean(axis=0)
            heldout_target_mask = target_features_frame["target_class"].astype(str).apply(lambda value: stable_split(value or "unknown", target_fraction)).to_numpy()
            if heldout_target_mask.any():
                target_metrics = {
                    "ndcg_at_10": ndcg_at_k(target_label_mean[heldout_target_mask], target_score_mean[heldout_target_mask], 10),
                    "proxy_mean_score": float(target_score_mean[heldout_target_mask].mean()),
                }
                target_ndcg_at_10 = float(target_metrics["ndcg_at_10"])
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "target_retrieval", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in target_metrics.items()
                    ]
                )

            regimen_ndcg_at_10 = 0.0
            regimen_score_mean = regimen_scores.mean(axis=0)
            regimen_label_mean = regimen_labels.mean(axis=0)
            heldout_regimen_mask = regimen_features_frame["regimen_scaffold_set"].astype(str).apply(lambda value: stable_split(value or "no_scaffold", scaffold_fraction)).to_numpy()
            if heldout_regimen_mask.any():
                regimen_metrics = {
                    "ndcg_at_10": ndcg_at_k(regimen_label_mean[heldout_regimen_mask], regimen_score_mean[heldout_regimen_mask], 10),
                    "pk_feasibility_top_10": float(pd.to_numeric(regimen_features_frame.loc[heldout_regimen_mask, "pk_feasibility_score"], errors="coerce").sort_values(ascending=False).head(10).mean()),
                }
                regimen_ndcg_at_10 = float(regimen_metrics["ndcg_at_10"])
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "regimen_ranking", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in regimen_metrics.items()
                    ]
                )
            regimen_ensemble_metrics = {
                "ensemble_member_count": float(ensemble_member_count),
                "predictive_entropy_mean": float(regimen_uq["predictive_entropy"].mean()),
                "mutual_information_mean": float(regimen_uq["mutual_information"].mean()),
                "epistemic_std_mean": float(regimen_uq["epistemic_std"].mean()),
            }
            result_rows.extend(
                [
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "regimen_ensemble_uq", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                    for metric_name, metric_value in regimen_ensemble_metrics.items()
                ]
            )

            simulation_targets = batch_simulation_targets(simulation_targets_by_condition, patient_labels)
            aligned_predicted_field, aligned_simulation_targets = align_time_horizon(predicted_field, simulation_targets)
            concentration_nrmse = normalized_rmse(aligned_predicted_field[..., 0], aligned_simulation_targets[..., 0])
            occupancy_nrmse = normalized_rmse(aligned_predicted_field[..., 1], aligned_simulation_targets[..., 1])
            flux_nrmse = normalized_rmse(aligned_predicted_field[..., 4], aligned_simulation_targets[..., 4])
            electrophysiology_nrmse = normalized_rmse(aligned_predicted_field[..., 6], aligned_simulation_targets[..., 6])
            result_rows.extend(
                [
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "concentration_nrmse", concentration_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "occupancy_nrmse", occupancy_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "flux_nrmse", flux_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "electrophysiology_nrmse", electrophysiology_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                ]
            )

            route_exposure = route_ordering_exposure(predicted_field, regimen_features_frame, config)
            route_metric = 0.0
            if all(route_name in route_exposure for route_name in ["intrathecal", "intravenous", "oral"]):
                route_metric = float(route_exposure["intrathecal"] > route_exposure["intravenous"] > route_exposure["oral"])
                result_rows.append(
                    result_row(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "mechanistic_sanity",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        "route_ordering_hit",
                        route_metric,
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                    )
                )

            accept_mask = tri_values >= tri_threshold
            prediction_error = np.abs((1.0 / (1.0 + np.exp(-regimen_scores))) - np.clip(regimen_labels, 0.0, 1.0))
            flattened_curve = risk_coverage_curve(
                prediction_error.reshape(-1),
                tri_values.reshape(-1),
                risk_coverage_grid_size,
            )
            reliability_metrics = {
                "accepted_mean_error": float(prediction_error[accept_mask].mean()) if accept_mask.any() else 0.0,
                "rejected_mean_error": float(prediction_error[~accept_mask].mean()) if (~accept_mask).any() else 0.0,
                "coverage": float(accept_mask.mean()),
                "coverage_gap": coverage_gap(float(accept_mask.mean()), target_acceptance_coverage),
                "acceptance_benefit": acceptance_benefit(prediction_error.reshape(-1), accept_mask.reshape(-1)),
                "risk_coverage_auc": float(flattened_curve["auc"]),
            }
            result_rows.extend(
                [
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "reliability_abstention", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                    for metric_name, metric_value in reliability_metrics.items()
                ]
            )

            for validation_row in quantitative_case_validation_rows(config, regimen_features_frame, compound_rows, predicted_field, tri_values):
                for metric_name in [
                    "standardized_effect_error",
                    "biomarker_effect_error",
                    "direction_hit",
                    "rank_recovery_hit",
                    "tri_gate_consistency",
                    "clinical_endpoint_qualitative_presence",
                ]:
                    result_rows.append(
                        result_row(
                            benchmark_name,
                            disease_program,
                            fold_row["fold_name"],
                            "quantitative_case_validation",
                            model_name_for_results,
                            modality,
                            int(patient_labels.shape[0]),
                            metric_name,
                            float(validation_row[metric_name]),
                            ensemble_artifact_path,
                            f"{fold_row['held_out_study_uid']}|{validation_row['intervention_label']}|{validation_row['clinical_endpoint_alignment_summary']}",
                            case_study_name=str(validation_row["case_study_name"]),
                            is_case_study_anchor="yes",
                        )
                    )

            tri_mean = tri_values.mean(axis=0)
            tri_accepted_regimen_mask = tri_mean >= tri_threshold
            for case_row in strict_case_rows:
                target_metrics = compute_anchor_recovery_metrics(
                    target_score_mean.astype(np.float64),
                    target_anchor_mask(target_features_frame, case_row),
                    int(case_row["top_k"]),
                )
                result_rows.extend(
                    strict_case_result_rows(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "strict_case_target_recovery",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                        str(case_row["case_study_name"]),
                        target_metrics,
                    )
                )
                regimen_metrics = compute_anchor_recovery_metrics(
                    regimen_score_mean.astype(np.float64),
                    regimen_anchor_mask(regimen_features_frame, case_row, compound_rows),
                    int(case_row["top_k"]),
                )
                regimen_metrics.update(
                    compute_tri_accepted_anchor_metrics(
                        regimen_score_mean.astype(np.float64),
                        regimen_anchor_mask(regimen_features_frame, case_row, compound_rows),
                        tri_accepted_regimen_mask,
                        int(case_row["top_k"]),
                    )
                )
                regimen_metrics.update(
                    strict_case_dynamics_metrics(
                        config,
                        case_row,
                        regimen_features_frame,
                        compound_rows,
                        predicted_field,
                        tri_accepted_regimen_mask,
                    )
                )
                result_rows.extend(
                    strict_case_result_rows(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "strict_case_regimen_recovery",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                        str(case_row["case_study_name"]),
                        regimen_metrics,
                    )
                )

            disease_family = disease_family_name(disease_program)
            world_state_rows = world_state_frame.loc[
                (world_state_frame["disease_program"].astype(str) == disease_program)
                & (world_state_frame["observed_modality_set"].astype(str) == modality)
            ].copy() if not world_state_frame.empty else pd.DataFrame()
            crossmodal_anchor_count = int(
                crossmodal_anchor_frame.loc[crossmodal_anchor_frame["disease_family"].astype(str) == disease_family].shape[0]
            ) if not crossmodal_anchor_frame.empty else 0
            simulation_teacher_count = int(
                simulation_teacher_frame.loc[simulation_teacher_frame["disease_program"].astype(str) == disease_program].shape[0]
            ) if not simulation_teacher_frame.empty else 0
            counterfactual_pair_count = int(
                counterfactual_benchmark_frame.loc[counterfactual_benchmark_frame["disease_program"].astype(str) == disease_program].shape[0]
            ) if not counterfactual_benchmark_frame.empty else 0
            feedback_event_count = int(
                feedback_event_frame.loc[feedback_event_frame["disease_program"].astype(str) == disease_program].shape[0]
            ) if not feedback_event_frame.empty else 0
            missing_modality_count = 0.0
            if not world_state_rows.empty and "missing_modality_set" in world_state_rows.columns:
                missing_modality_count = float(
                    world_state_rows["missing_modality_set"].astype(str).apply(lambda value: len([item for item in value.split("|") if item])).mean()
                )
            capability_metrics = {
                "concentration_nrmse": float(concentration_nrmse),
                "flux_nrmse": float(flux_nrmse),
                "route_ordering_hit": float(route_metric) if "route_metric" in locals() else 0.0,
                "electrophysiology_nrmse": float(electrophysiology_nrmse),
                "pkpd_consistency_mean": float(pkpd_consistency_values.mean()),
                "regimen_ndcg_at_10": float(regimen_ndcg_at_10),
                "target_ndcg_at_10": float(target_ndcg_at_10),
                "ood_auroc": float(latent_ood_metric),
                "acceptance_benefit": float(reliability_metrics["acceptance_benefit"]),
                "interactive_latency_ms": float(interactive_latency_ms),
            }
            capability_registries = {
                "factor_count": float(len(config["foundation_twin"]["world_state_factors"])),
                "crossmodal_anchor_count": float(crossmodal_anchor_count),
                "missing_modality_count": float(missing_modality_count),
                "counterfactual_pair_count": float(counterfactual_pair_count),
                "feedback_event_count": float(feedback_event_count),
                "simulation_teacher_count": float(simulation_teacher_count),
            }
            capability_rows.extend(
                capability_catalog_rows(
                    config,
                    benchmark_name,
                    disease_program,
                    disease_family,
                    str(fold_row["fold_name"]),
                    modality,
                    ensemble_artifact_path,
                    capability_metrics,
                    capability_registries,
                )
            )

            active_learning_uncertainty_matrix = regimen_uq[uncertainty_metric_name]
            for regimen_index, regimen_row in enumerate(regimen_features_frame.to_dict(orient="records")):
                intervention_row = intervention_row_by_regimen.get(str(regimen_row["regimen_uid"]), {})
                candidate_priority_rows.append(
                    {
                        "benchmark_name": benchmark_name,
                        "fold_name": str(fold_row["fold_name"]),
                        "modality": modality,
                        "regimen_uid": str(regimen_row["regimen_uid"]),
                        "predicted_score_mean": float(regimen_score_mean[regimen_index]),
                        "tri_score_mean": float(tri_values[:, regimen_index].mean()),
                        "pkpd_consistency_mean": float(pkpd_consistency_values[:, regimen_index].mean()),
                        "predictive_entropy_mean": float(regimen_uq["predictive_entropy"][:, regimen_index].mean()),
                        "mutual_information_mean": float(regimen_uq["mutual_information"][:, regimen_index].mean()),
                        "epistemic_std_mean": float(regimen_uq["epistemic_std"][:, regimen_index].mean()),
                        "active_learning_uncertainty": float(active_learning_uncertainty_matrix[:, regimen_index].mean()),
                        "artifact_path": ensemble_artifact_path,
                        "note": str(intervention_row.get("intervention_program_uid", "")),
                    }
                )

    result_rows.sort(key=lambda row: (row["split_name"], row["evaluation_scope"], row["metric_name"], row["modality"], row["case_study_name"]))
    result_catalog_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["result_catalog"]
    )
    write_tsv(result_catalog_path, config["schemas"]["screening_result_fields"], result_rows)
    candidate_priority_rows.sort(key=lambda row: (row["fold_name"], row["modality"], row["regimen_uid"]))
    candidate_priority_catalog_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["candidate_priority_catalog"]
    )
    write_tsv(
        candidate_priority_catalog_path,
        config["schemas"]["screening_candidate_priority_fields"],
        candidate_priority_rows,
    )
    capability_rows.sort(key=lambda row: (row["fold_name"], row["modality"], row["capability_family"], row["capability_name"], row["metric_name"]))
    capability_catalog_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["capability_catalog"]
    )
    write_tsv(
        capability_catalog_path,
        config["schemas"]["foundation_capability_fields"],
        capability_rows,
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=result_catalog_path,
        result_row_count=len(result_rows),
    )


if __name__ == "__main__":
    main()
