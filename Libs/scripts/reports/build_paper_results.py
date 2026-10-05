from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.models.networks.shared_latent_dual_encoder import SharedLatentDualEncoder
from Libs.models.networks.tabular_jepa_auxiliary_encoder import TabularJEPAAuxiliaryEncoder
from Libs.utils.model.dynamics_fields import default_region_prior, node_adjacency_matrix
from Libs.utils.model.neuronsifter_scale_registry import registry as neuronsifter_scale_registry
from Libs.utils.model.neuronsifter_scale_registry import scale_parameter_estimate
from Libs.utils.model.screening_model_config import (
    baseline_field_policy_from_config,
    rollout_update_policy_from_config,
    solver_config_from_config,
    solver_state_layout_from_config,
    world_state_posterior_config_from_config,
)
from Libs.utils.exp.experiment_manifests import load_experiment_family, load_paper_track
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json, write_tsv
from Libs.utils.data.trajectory_numeric_lineage import load_canonical_metric_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures for the structural-prior dynamics paper.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--figure-dir", default="")
    parser.add_argument("--table-dir", default="")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--paper-track", default="research_hardening")
    parser.add_argument("--allow-pending", action="store_true")
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "pending"
    array = np.asarray(values, dtype=np.float64)
    return f"{float(array.mean()):.3f} $\\pm$ {float(array.std(ddof=0)):.3f}"


def format_mean(values: list[float]) -> str:
    if not values:
        return "pending"
    return f"{float(np.asarray(values, dtype=np.float64).mean()):.3f}"


def publication_disease_programs(config: dict) -> set[str]:
    values = config["paper_experiments"].get("publication_disease_programs", [])
    requested_programs = {str(value) for value in values} if values else {"ad_control"}
    benchmark_root = resolve_repo_path(config["benchmark_registry"]["local_root"])
    cohort_catalog_path = benchmark_root / config["benchmark_registry"]["relative_paths"]["cohort_catalog"]
    if not cohort_catalog_path.exists():
        return requested_programs
    cohort_frame = read_tsv_frame(cohort_catalog_path)
    publication_programs: set[str] = set()
    for disease_program in requested_programs:
        program_config = dict(config["benchmark_registry"]["disease_programs"].get(disease_program, {}))
        minimum_primary_study_count = int(program_config.get("minimum_included_primary_study_count", 2))
        contract_status = str(program_config.get("contract_status", "complete_public_contract"))
        included_primary_count = int(
            cohort_frame.loc[
                cohort_frame["disease_program"].astype(str).eq(disease_program)
                & cohort_frame["cohort_role"].astype(str).eq("primary_train_eval")
                & cohort_frame["inclusion_flag"].astype(str).eq("yes")
            ].shape[0]
        )
        if contract_status == "incomplete_public_contract":
            continue
        if included_primary_count >= minimum_primary_study_count:
            publication_programs.add(disease_program)
    return publication_programs or {"ad_control"}


def publication_disease_program_notes(config: dict) -> list[str]:
    requested_programs = {str(value) for value in config["paper_experiments"].get("publication_disease_programs", [])}
    effective_programs = publication_disease_programs(config)
    notes: list[str] = []
    for disease_program in sorted(requested_programs - effective_programs):
        notes.append(f"{disease_program} data-incomplete")
    return notes


def canonical_dynamics_spec(paper_track: dict[str, object]) -> dict[str, list[str] | str]:
    canonical_config = paper_track.get("canonical_dynamics", {})
    primary_profiles = [str(profile_name) for profile_name in paper_track.get("primary_dynamics_profiles", [])]
    context_profiles = [str(profile_name) for profile_name in paper_track.get("secondary_benchmark_profiles", [])]
    ad_mainline_profile = str(canonical_config.get("ad_mainline_profile", primary_profiles[0] if primary_profiles else "dynamics_mainline_ad"))
    ad_ablation_profiles = [
        str(profile_name)
        for profile_name in canonical_config.get("ad_ablation_profiles", primary_profiles[1:])
    ]
    context_formal_profiles = [
        str(profile_name)
        for profile_name in canonical_config.get("context_formal_profiles", context_profiles)
    ]
    return {
        "ad_mainline_profile": ad_mainline_profile,
        "ad_ablation_profiles": ad_ablation_profiles,
        "context_formal_profiles": context_formal_profiles,
    }


def required_dynamics_profiles(paper_track: dict[str, object]) -> list[str]:
    spec = canonical_dynamics_spec(paper_track)
    ordered_profiles = [
        str(spec["ad_mainline_profile"]),
        *[str(profile_name) for profile_name in spec["ad_ablation_profiles"]],
    ]
    return list(dict.fromkeys(ordered_profiles))


def profile_status_names(paper_track: dict[str, object]) -> list[str]:
    spec = canonical_dynamics_spec(paper_track)
    ordered_profiles = [
        *required_dynamics_profiles(paper_track),
        *[str(profile_name) for profile_name in paper_track.get("shared_context_profiles", [])],
        *[str(profile_name) for profile_name in spec["context_formal_profiles"]],
    ]
    return list(dict.fromkeys(ordered_profiles))


def screening_profile_paths(config: dict, profile_name: str) -> list[Path]:
    try:
        profile = load_experiment_family(profile_name, config)
    except (FileNotFoundError, ValueError):
        return []
    runtime = profile["runtime"]
    relative_result_catalog = config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["result_catalog"]
    paths = [
        (REPO_ROOT / runtime["artifact_root"]).resolve()
        / str(runtime["benchmark_name_template"]).format(seed=seed_value)
        / relative_result_catalog
        for seed_value in runtime.get("seed_values", [])
    ]
    return [path for path in paths if path.exists()]


def screening_profile_capability_paths(config: dict, profile_name: str) -> list[Path]:
    try:
        profile = load_experiment_family(profile_name, config)
    except (FileNotFoundError, ValueError):
        return []
    runtime = profile["runtime"]
    relative_capability_catalog = config["screening_pipeline"]["screening_benchmark"]["relative_paths"].get("capability_catalog", "")
    if not relative_capability_catalog:
        return []
    paths = [
        (REPO_ROOT / runtime["artifact_root"]).resolve()
        / str(runtime["benchmark_name_template"]).format(seed=seed_value)
        / relative_capability_catalog
        for seed_value in runtime.get("seed_values", [])
    ]
    return [path for path in paths if path.exists()]


def resolved_screening_profile(
    config: dict,
    primary_profile_name: str,
    fallback_profile_names: list[str] | None = None,
) -> tuple[str, list[Path]]:
    paths = screening_profile_paths(config, primary_profile_name)
    if paths:
        return primary_profile_name, paths
    for fallback_profile_name in fallback_profile_names or []:
        fallback_paths = screening_profile_paths(config, fallback_profile_name)
        if fallback_paths:
            return fallback_profile_name, fallback_paths
    return primary_profile_name, []


def shared_benchmark_paths(config: dict) -> list[Path]:
    try:
        profile = load_experiment_family("shared_latent_formal", config)
    except (FileNotFoundError, ValueError):
        return []
    runtime = profile["runtime"]
    benchmark_root = (REPO_ROOT / config["benchmark_registry"]["local_root"]).resolve()
    paths = [
        benchmark_root / str(runtime["result_catalog_template"]).format(seed=seed_value)
        for seed_value in runtime.get("seed_values", [])
    ]
    available = [path for path in paths if path.exists()]
    if available:
        return available
    fallback_path = benchmark_root / config["benchmark_registry"]["relative_paths"]["baseline_result_catalog"]
    return [fallback_path] if fallback_path.exists() else []


def current_screening_result_paths(config: dict) -> list[Path]:
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    path = (
        Path(benchmark["artifact_root"]).resolve()
        / benchmark["benchmark_name"]
        / benchmark["relative_paths"]["result_catalog"]
    )
    return [path] if path.exists() else []


def current_screening_capability_paths(config: dict) -> list[Path]:
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    relative_path = benchmark["relative_paths"].get("capability_catalog", "")
    if not relative_path:
        return []
    path = (
        Path(benchmark["artifact_root"]).resolve()
        / benchmark["benchmark_name"]
        / relative_path
    )
    return [path] if path.exists() else []


def safe_concat(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [read_tsv_frame(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def screening_task_catalog(config: dict) -> dict[str, dict[str, object]]:
    return {
        str(task_name): dict(task_payload)
        for task_name, task_payload in config["screening_pipeline"]["screening_benchmark"].get("paper_tasks", {}).items()
    }


def grouped_metric_values(
    frame: pd.DataFrame,
    evaluation_scope: str,
    metric_name: str,
    case_study_name: str = "",
) -> list[float]:
    if frame.empty:
        return []
    filtered = frame.loc[
        (frame["evaluation_scope"] == evaluation_scope)
        & (frame["metric_name"] == metric_name)
    ].copy()
    if case_study_name:
        if "case_study_name" not in filtered.columns:
            return []
        filtered = filtered.loc[filtered["case_study_name"] == case_study_name].copy()
    if filtered.empty:
        return []
    if "split_name" in filtered:
        group_columns = ["split_name"]
    elif "fold_name" in filtered:
        group_columns = ["fold_name"]
    else:
        group_columns = []
    if group_columns:
        return filtered.groupby(group_columns, sort=True)["metric_value"].mean().astype(float).tolist()
    return filtered["metric_value"].astype(float).tolist()


def grouped_task_metric_values(frame: pd.DataFrame, task_name: str, metric_name: str) -> list[float]:
    if frame.empty or "task_name" not in frame.columns:
        return []
    filtered = frame.loc[(frame["task_name"] == task_name) & (frame["metric_name"] == metric_name)].copy()
    if filtered.empty:
        return []
    if "split_name" in filtered.columns:
        return filtered.groupby(["split_name"], sort=True)["metric_value"].mean().astype(float).tolist()
    return filtered["metric_value"].astype(float).tolist()


def grouped_baseline_values(frame: pd.DataFrame, model_name: str, metric_name: str) -> list[float]:
    if frame.empty:
        return []
    if "evaluation_scope" in frame.columns:
        filtered = frame.loc[
            (frame["model_name"] == model_name)
            & (frame["evaluation_scope"] == "headline_pooled")
        ].copy()
    else:
        filtered = frame.loc[frame["model_name"] == model_name].copy()
    if filtered.empty or metric_name not in filtered:
        return []
    return filtered[metric_name].astype(float).tolist()


def grouped_runnable_baseline_values(frame: pd.DataFrame, baseline_name: str, metric_name: str) -> list[float]:
    if frame.empty:
        return []
    if {"model_name", "metric_name", "metric_value"} <= set(frame.columns):
        filtered = frame.loc[(frame["model_name"] == baseline_name) & (frame["metric_name"] == metric_name)].copy()
        if filtered.empty:
            return []
        values = filtered["metric_value"].astype(float).tolist()
        return [value for value in values if np.isfinite(value)]
    if "baseline_name" not in frame or metric_name not in frame:
        return []
    filtered = frame.loc[frame["baseline_name"] == baseline_name].copy()
    if filtered.empty:
        return []
    values = filtered[metric_name].astype(float).tolist()
    return [value for value in values if np.isfinite(value)]


def runnable_baseline_contract_satisfied(
    frame: pd.DataFrame,
    baseline_name: str,
    task_coverage: list[object],
) -> bool:
    normalized_coverage = {str(value) for value in task_coverage}
    required_metrics: set[str] = set()
    if "sample_state_inference_and_uq" in normalized_coverage:
        required_metrics.update({"auroc", "auprc", "balanced_accuracy", "macro_f1", "brier_score"})
    if "long_horizon_twin_rollout_and_active_learning" in normalized_coverage:
        required_metrics.add("transition_mse")
    if "graph_evidence_intervention_prioritization" in normalized_coverage:
        required_metrics.update({"graph_prior_score", "target_evidence_coverage", "edge_evidence_coverage"})
    if not required_metrics:
        return False
    for metric_name in required_metrics:
        if not grouped_runnable_baseline_values(frame, baseline_name, metric_name):
            return False
    return True


def dynamics_task_summary_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    profile_name = str(canonical_dynamics_spec(paper_track)["ad_mainline_profile"])
    _, paths = resolved_screening_profile(config, profile_name)
    frame = safe_concat(paths)
    rows: list[dict[str, object]] = []
    for task_name, task_payload in screening_task_catalog(config).items():
        primary_scope = str(task_payload.get("primary_scope", ""))
        primary_metric_name = str(task_payload.get("primary_metric_name", ""))
        metric_values = grouped_task_metric_values(frame, task_name, primary_metric_name)
        if not metric_values and primary_scope:
            metric_values = grouped_metric_values(frame, primary_scope, primary_metric_name)
        rows.append(
            {
                "task_name": task_name,
                "task_family": str(task_payload.get("task_family", "")),
                "primary_metric_label": str(task_payload.get("primary_metric_label", primary_metric_name)),
                "primary_metric_value": format_mean_std(metric_values),
                "status": "available" if metric_values else "pending",
            }
        )
    return rows


def foundation_capability_summary_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    profile_name = str(canonical_dynamics_spec(paper_track)["ad_mainline_profile"])
    frame = safe_concat(screening_profile_capability_paths(config, profile_name))
    if frame.empty:
        frame = safe_concat(current_screening_capability_paths(config))
    if frame.empty:
        capability_names = [
            ("hard_gate", "structured_world_state"),
            ("hard_gate", "causal_dynamics"),
            ("hard_gate", "long_horizon_action_rollout"),
            ("hard_gate", "counterfactual_planning"),
            ("engineering", "multimodal_alignment"),
            ("engineering", "online_continual_calibration"),
            ("high_order", "causal_robustness"),
            ("high_order", "hierarchical_generalization"),
            ("high_order", "interacting_subsystems"),
            ("high_order", "explanation_and_confidence"),
        ]
        return [
            {
                "capability_family": capability_family,
                "capability_name": capability_name,
                "pass_fraction": "pending",
                "status": "pending",
            }
            for capability_family, capability_name in capability_names
        ]
    filtered = frame.loc[frame["metric_name"].astype(str) == "contract_pass_fraction"].copy()
    rows: list[dict[str, object]] = []
    for (capability_family, capability_name), current_frame in filtered.groupby(["capability_family", "capability_name"], sort=True):
        pass_values = current_frame["metric_value"].astype(float).tolist()
        rows.append(
            {
                "capability_family": str(capability_family),
                "capability_name": str(capability_name),
                "pass_fraction": format_mean(pass_values),
                "status": "available" if pass_values else "pending",
            }
        )
    return rows


def dynamics_mainline_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    profile_name = str(canonical_dynamics_spec(paper_track)["ad_mainline_profile"])
    resolved_profile_name, paths = resolved_screening_profile(config, profile_name)
    frame = safe_concat(paths)
    return [
        {
            "profile_name": profile_name,
            "status": "available" if paths else "pending",
            "artifact_profile_name": resolved_profile_name if paths else "",
            "seed_count": len(paths),
            "disease_auroc": format_mean_std(grouped_metric_values(frame, "disease_state_inference", "auroc")),
            "disease_auprc": format_mean_std(grouped_metric_values(frame, "disease_state_inference", "auprc")),
            "disease_ece": format_mean_std(grouped_metric_values(frame, "disease_state_calibration", "ece")),
            "disease_nll": format_mean_std(grouped_metric_values(frame, "disease_state_calibration", "nll")),
            "disease_mutual_information": format_mean_std(grouped_metric_values(frame, "disease_state_ensemble_uq", "mutual_information_mean")),
            "target_ndcg": format_mean_std(grouped_metric_values(frame, "target_retrieval", "ndcg_at_10")),
            "regimen_ndcg": format_mean_std(grouped_metric_values(frame, "regimen_ranking", "ndcg_at_10")),
            "regimen_mutual_information": format_mean_std(grouped_metric_values(frame, "regimen_ensemble_uq", "mutual_information_mean")),
            "regimen_epistemic_std": format_mean_std(grouped_metric_values(frame, "regimen_ensemble_uq", "epistemic_std_mean")),
            "concentration_nrmse": format_mean_std(grouped_metric_values(frame, "solver_grounded_dynamics", "concentration_nrmse")),
            "flux_nrmse": format_mean_std(grouped_metric_values(frame, "solver_grounded_dynamics", "flux_nrmse")),
            "route_ordering_hit": format_mean_std(grouped_metric_values(frame, "mechanistic_sanity", "route_ordering_hit")),
            "standardized_effect_error": format_mean_std(grouped_metric_values(frame, "quantitative_case_validation", "standardized_effect_error")),
            "direction_hit": format_mean_std(grouped_metric_values(frame, "quantitative_case_validation", "direction_hit")),
            "rank_recovery_hit": format_mean_std(grouped_metric_values(frame, "quantitative_case_validation", "rank_recovery_hit")),
            "tri_gate_consistency": format_mean_std(grouped_metric_values(frame, "quantitative_case_validation", "tri_gate_consistency")),
            "coverage": format_mean_std(grouped_metric_values(frame, "reliability_abstention", "coverage")),
            "risk_coverage_auc": format_mean_std(grouped_metric_values(frame, "reliability_abstention", "risk_coverage_auc")),
            "acceptance_benefit": format_mean_std(grouped_metric_values(frame, "reliability_abstention", "acceptance_benefit")),
            "ood_auroc": format_mean_std(grouped_metric_values(frame, "ood_robustness", "latent_distance_ood_auroc")),
        }
    ]


def dynamics_ablation_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    profile_names = [str(profile_name) for profile_name in canonical_dynamics_spec(paper_track)["ad_ablation_profiles"]]
    rows: list[dict[str, object]] = []
    for profile_name in profile_names:
        resolved_profile_name, paths = resolved_screening_profile(config, profile_name)
        frame = safe_concat(paths)
        rows.append(
            {
                "profile_name": profile_name,
                "status": "available" if paths else "pending",
                "artifact_profile_name": resolved_profile_name if paths else "",
                "seed_count": len(paths),
                "disease_auroc": format_mean_std(grouped_metric_values(frame, "disease_state_inference", "auroc")),
                "disease_ece": format_mean_std(grouped_metric_values(frame, "disease_state_calibration", "ece")),
                "regimen_ndcg": format_mean_std(grouped_metric_values(frame, "regimen_ranking", "ndcg_at_10")),
                "concentration_nrmse": format_mean_std(grouped_metric_values(frame, "solver_grounded_dynamics", "concentration_nrmse")),
                "standardized_effect_error": format_mean_std(grouped_metric_values(frame, "quantitative_case_validation", "standardized_effect_error")),
                "coverage": format_mean_std(grouped_metric_values(frame, "reliability_abstention", "coverage")),
                "risk_coverage_auc": format_mean_std(grouped_metric_values(frame, "reliability_abstention", "risk_coverage_auc")),
            }
        )
    return rows


def strict_case_summary_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    profile_name = str(canonical_dynamics_spec(paper_track)["ad_mainline_profile"])
    _, paths = resolved_screening_profile(config, profile_name)
    frame = safe_concat(paths)
    case_names = ["ad_amyloid_route"]
    rows: list[dict[str, object]] = []
    for case_name in case_names:
        rows.append(
            {
                "case_study_name": case_name,
                "anchor_hit_at_top_k": format_mean_std(grouped_metric_values(frame, "strict_case_regimen_recovery", "anchor_hit_at_top_k", case_name)),
                "quantitative_anchor_hit": format_mean_std(grouped_metric_values(frame, "strict_case_regimen_recovery", "quantitative_anchor_hit", case_name)),
                "structural_plausibility_hit": format_mean_std(grouped_metric_values(frame, "strict_case_regimen_recovery", "structural_plausibility_hit", case_name)),
            }
        )
    return rows


def quantitative_case_validation_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    profile_name = str(canonical_dynamics_spec(paper_track)["ad_mainline_profile"])
    _, paths = resolved_screening_profile(config, profile_name)
    frame = safe_concat(paths)
    if frame.empty:
        return []
    filtered = frame.loc[frame["evaluation_scope"] == "quantitative_case_validation"].copy()
    if filtered.empty:
        return []
    note_parts = filtered["note"].astype(str).str.split("|", n=2, expand=True).reindex(columns=[0, 1, 2], fill_value="")
    filtered["intervention_label"] = note_parts[1].fillna("")
    filtered["clinical_endpoint_alignment_summary"] = note_parts[2].fillna("")
    metric_names = [
        "standardized_effect_error",
        "biomarker_effect_error",
        "direction_hit",
        "rank_recovery_hit",
        "tri_gate_consistency",
        "clinical_endpoint_qualitative_presence",
    ]
    rows: list[dict[str, object]] = []
    for intervention_label, current_frame in filtered.groupby("intervention_label", sort=True):
        row: dict[str, object] = {
            "case_study_name": current_frame["case_study_name"].iloc[0],
            "intervention_label": intervention_label,
            "seed_count": int(current_frame["artifact_path"].nunique()),
            "clinical_endpoint_alignment_summary": current_frame["clinical_endpoint_alignment_summary"].iloc[0],
        }
        for metric_name in metric_names:
            metric_frame = current_frame.loc[current_frame["metric_name"] == metric_name]
            row[metric_name] = format_mean_std(metric_frame["metric_value"].astype(float).tolist()) if not metric_frame.empty else "pending"
        rows.append(row)
    return rows


def clinical_direction_case_study_matrix_rows(config: dict) -> list[dict[str, object]]:
    comparison_config = config["screening_pipeline"]["screening_benchmark"]["case_study_comparison"]
    if not bool(comparison_config["enabled"]):
        return []
    score_matrix = dict(comparison_config["anchor_score_matrix"])
    validation_rows = {
        str(row["intervention_label"]): row
        for row in config["dynamics_ready_registry"]["case_validation_entries"]
    }
    comparison_stages = [str(stage) for stage in comparison_config["comparison_stages"]]
    stage_rankings: dict[str, dict[str, int]] = {}
    for stage in comparison_stages:
        ranked_labels = sorted(score_matrix, key=lambda label: float(score_matrix[label][stage]), reverse=True)
        stage_rankings[stage] = {
            label: rank_index + 1
            for rank_index, label in enumerate(ranked_labels)
        }
    rows: list[dict[str, object]] = []
    for intervention_label, score_payload in sorted(score_matrix.items()):
        validation_row = validation_rows[intervention_label]
        rows.append(
            {
                "intervention_label": intervention_label,
                "case_type": score_payload["case_type"],
                "expected_effect_direction": validation_row["expected_effect_direction"],
                "source_holdout_policy": comparison_config["source_holdout_policy"],
                "clinical_claim_policy": comparison_config["clinical_claim_policy"],
                "external_evidence_only_score": float(score_payload["external_evidence_only"]),
                "external_evidence_only_rank": stage_rankings["external_evidence_only"][intervention_label],
                "neuronsifter_ranking_score": float(score_payload["neuronsifter_ranking"]),
                "neuronsifter_ranking_rank": stage_rankings["neuronsifter_ranking"][intervention_label],
                "neuronsifter_cns_rollout_score": float(score_payload["neuronsifter_cns_rollout"]),
                "neuronsifter_cns_rollout_rank": stage_rankings["neuronsifter_cns_rollout"][intervention_label],
                "neuronsifter_tri_gated_score": float(score_payload["neuronsifter_tri_gated"]),
                "neuronsifter_tri_gated_rank": stage_rankings["neuronsifter_tri_gated"][intervention_label],
                "clinical_direction_score": float(score_payload["endpoint_direction_score"]),
                "clinical_direction_uncertainty": float(score_payload["endpoint_direction_uncertainty"]),
                "source_provenance": validation_row["clinical_endpoint_source_url"],
                "endpoint_alignment_policy": validation_row["clinical_endpoint_alignment_policy"],
            }
        )
    return rows


def publication_readiness_config(config: dict) -> dict[str, object]:
    return dict(config["paper"]["publication_readiness"])


def publication_report_path(config: dict, artifact_name: str) -> str:
    report_dir = str(publication_readiness_config(config)["report_outputs"]["report_dir"])
    return str(Path(report_dir) / artifact_name)


def publication_claim_metric_rows(config: dict) -> list[dict[str, object]]:
    readiness_config = publication_readiness_config(config)
    metric_catalog = dict(readiness_config["metric_catalog"])
    rows: list[dict[str, object]] = []
    for claim_config in readiness_config["core_claims"]:
        claim_id = str(claim_config["claim_id"])
        for metric_family in [str(value) for value in claim_config["required_metric_families"]]:
            for metric_name in [str(value) for value in metric_catalog[metric_family]]:
                for holdout_name in [str(value) for value in claim_config["required_holdouts"]]:
                    rows.append(
                        {
                            "claim_id": claim_id,
                            "metric_family": metric_family,
                            "metric_name": metric_name,
                            "holdout_name": holdout_name,
                            "baseline_id": "claim_specific_baseline_grid",
                            "comparator_id": "full_neuronsifter_large",
                            "bootstrap_interval": str(claim_config["required_bootstrap_ci"]),
                            "bootstrap_sample_count": int(readiness_config["statistical_contract"]["bootstrap_sample_count"]),
                            "acceptance_delta_policy": str(claim_config["acceptance_delta_policy"]),
                            "result_status": "requires_experiment_result",
                            "artifact_path": publication_report_path(config, "claim_metric_catalog.tsv"),
                        }
                    )
    return rows


def publication_baseline_result_rows(config: dict) -> list[dict[str, object]]:
    readiness_config = publication_readiness_config(config)
    disease_program = str(readiness_config["research_positioning"]["disease_scope"])
    holdout_names = [str(value) for value in readiness_config["strict_splits"]["split_names"]]
    rows: list[dict[str, object]] = []
    for baseline_config in readiness_config["baseline_grid"]["baselines"]:
        for holdout_name in holdout_names:
            rows.append(
                {
                    "baseline_id": str(baseline_config["baseline_id"]),
                    "baseline_family": str(baseline_config["family"]),
                    "conditioning": str(baseline_config["conditioning"]),
                    "disease_program": disease_program,
                    "holdout_name": holdout_name,
                    "primary_metric": "regimen_ndcg",
                    "bootstrap_interval": str(readiness_config["statistical_contract"]["primary_interval"]),
                    "result_status": "requires_experiment_result",
                    "artifact_path": publication_report_path(config, "baseline_result_matrix.tsv"),
                }
            )
    return rows


def publication_ablation_result_rows(config: dict) -> list[dict[str, object]]:
    readiness_config = publication_readiness_config(config)
    action_policy = readiness_config["baseline_grid"]["action_ablation_policy"]
    disease_program = str(readiness_config["research_positioning"]["disease_scope"])
    holdout_names = [str(value) for value in readiness_config["strict_splits"]["split_names"]]
    rows: list[dict[str, object]] = []
    for replacement_axis in [str(value) for value in action_policy["replacement_axis"]]:
        for holdout_name in holdout_names:
            rows.append(
                {
                    "ablation_id": f"{replacement_axis}_conditioning_ablation",
                    "ablation_family": "action_interface",
                    "shared_backbone": str(action_policy["shared_backbone"]),
                    "replacement_axis": replacement_axis,
                    "disease_program": disease_program,
                    "holdout_name": holdout_name,
                    "primary_metric": "regimen_ndcg",
                    "bootstrap_interval": str(readiness_config["statistical_contract"]["primary_interval"]),
                    "result_status": "requires_experiment_result",
                    "artifact_path": publication_report_path(config, "ablation_result_catalog.tsv"),
                }
            )
    return rows


def publication_selective_risk_rows(config: dict) -> list[dict[str, object]]:
    readiness_config = publication_readiness_config(config)
    tri_config = readiness_config["tri_calibration"]
    metric_names = [str(value) for value in readiness_config["metric_catalog"]["tri"]]
    evaluation_splits = [
        split_name
        for split_name in [str(value) for value in readiness_config["strict_splits"]["split_names"]]
        if split_name != str(tri_config["threshold_source_split"])
    ]
    rows: list[dict[str, object]] = []
    for evaluation_split in evaluation_splits:
        for metric_name in metric_names:
            rows.append(
                {
                    "gate_name": "tri_gate",
                    "threshold_source_split": str(tri_config["threshold_source_split"]),
                    "evaluation_split": evaluation_split,
                    "metric_name": metric_name,
                    "calibration_policy": str(tri_config["test_policy"]),
                    "result_status": "requires_experiment_result",
                    "artifact_path": publication_report_path(config, "selective_risk_catalog.tsv"),
                }
            )
    return rows


def pending_formal_result_surfaces(
    config: dict,
    paper_track: dict[str, object],
    status_rows: list[dict[str, object]],
    publication_claim_rows: list[dict[str, object]],
    publication_baseline_rows: list[dict[str, object]],
    publication_ablation_rows: list[dict[str, object]],
    publication_selective_risk_rows_payload: list[dict[str, object]],
) -> list[str]:
    materialization_policy = dict(
        publication_readiness_config(config)["submission_policy"]["formal_result_materialization_policy"]
    )
    if not bool(materialization_policy["fail_fast_before_main_tables"]):
        return []
    required_surfaces = {str(value) for value in materialization_policy["required_surfaces"]}
    ready_result_statuses = {str(value) for value in materialization_policy["ready_result_statuses"]}
    pending_entries: list[str] = []
    if "formal_dynamics_profiles" in required_surfaces:
        required_profiles = set(required_dynamics_profiles(paper_track))
        pending_profiles = [
            str(row["profile_name"])
            for row in status_rows
            if str(row["profile_name"]) in required_profiles and str(row["status"]) != "available"
        ]
        pending_entries.extend(f"formal_dynamics_profiles:{profile_name}" for profile_name in sorted(pending_profiles))
    publication_surface_rows = {
        "claim_metric_catalog": publication_claim_rows,
        "baseline_result_matrix": publication_baseline_rows,
        "ablation_result_catalog": publication_ablation_rows,
        "selective_risk_catalog": publication_selective_risk_rows_payload,
    }
    for surface_name, rows in publication_surface_rows.items():
        if surface_name not in required_surfaces:
            continue
        pending_count = sum(1 for row in rows if str(row["result_status"]) not in ready_result_statuses)
        if pending_count:
            pending_entries.append(f"{surface_name}:{pending_count}_rows")
    return pending_entries


def publication_figure_manifest_rows(config: dict) -> list[dict[str, object]]:
    readiness_config = publication_readiness_config(config)
    rows: list[dict[str, object]] = []
    for figure_config in readiness_config["manuscript_plan"]["figures"]:
        rows.append(
            {
                "figure_id": str(figure_config["figure_id"]),
                "output_name": str(figure_config["output_name"]),
                "claim_link": str(figure_config["claim_link"]),
                "required_inputs": ",".join(readiness_config["report_outputs"]["required_artifacts"]),
                "result_status": "requires_experiment_result",
                "artifact_path": publication_report_path(config, "figure_manifest.json"),
            }
        )
    return rows


def publication_g5_effective_sample_audit(config: dict) -> dict[str, object]:
    readiness_config = publication_readiness_config(config)
    g5_config = readiness_config["g5_foundation_pretraining"]
    simulation_audit = config["ai_ready_preprocessing"]["simulation_audit"]
    simulation_root = resolve_repo_path(str(simulation_audit["local_root"]))
    relative_paths = simulation_audit["relative_paths"]
    qc_summary_path = simulation_root / str(relative_paths["qc_summary"])
    split_leakage_path = simulation_root / str(relative_paths["split_leakage_audit"])
    pair_alignment_path = simulation_root / str(relative_paths["pair_alignment_audit"])
    qc_summary = read_json_payload(qc_summary_path)
    split_leakage = read_json_payload(split_leakage_path)
    pair_alignment = read_tsv_frame(pair_alignment_path)
    canonical_counts = load_canonical_metric_values(REPO_ROOT / "Libs/configs/config_stack.yaml")
    full_count = canonical_counts["raw_materialized_full_microenvironment_trajectory_count"]
    mesoscopic_count = canonical_counts["raw_materialized_mesoscopic_trajectory_count"]
    paired_count = canonical_counts["paired_full_mesoscopic_count"]
    train_eligible_count = canonical_counts["qc_eligible_trajectory_count"]
    store_min_count = int(g5_config["store_trajectory_min_count"])
    materialized_gate_passed = full_count >= store_min_count and mesoscopic_count >= store_min_count
    pair_status_counts = pair_alignment["paired_status"].astype(str).value_counts(dropna=False).to_dict()
    split_leakage_total = int(qc_summary["total_split_leakage_count"])
    effective_sample_counts = {
        "unique_trajectory_families": int(qc_summary["unique_trajectory_family_count"]),
        "unique_regimens": int(qc_summary["unique_regimen_count"]),
        "unique_target_programs": int(qc_summary["unique_target_program_count"]),
        "high_confidence_trajectories": int(qc_summary["high_confidence_trajectory_count"]),
        "medium_confidence_trajectories": int(qc_summary["medium_confidence_trajectory_count"]),
        "full_microenvironment_trajectories": full_count,
        "mesoscopic_trajectories": mesoscopic_count,
        "full_mesoscopic_paired_trajectories": paired_count,
        "counterfactual_pair_count": paired_count,
        "train_eligible_grade_ab_records": train_eligible_count,
        "weak_calibration_records": int(qc_summary["weak_teacher_count"]),
        "reject_records": canonical_counts["rejected_trajectory_count"],
        "stress_only_records": int(qc_summary["stress_only_count"]),
    }
    effective_coverage = {
        "full_pair_coverage": float(paired_count) / float(max(full_count, 1)),
        "mesoscopic_pair_coverage": float(paired_count) / float(max(mesoscopic_count, 1)),
        "counterfactual_pair_coverage": float(paired_count) / float(max(full_count, mesoscopic_count, 1)),
        "qc_grade_a_b_rate": float(qc_summary["train_eligible_rate"]),
        "high_or_medium_confidence_rate": float(qc_summary["high_or_medium_confidence_rate"]),
        "source_holdout_leakage_status": "pass" if split_leakage_total == 0 and str(split_leakage["leakage_status"]) == "pass" else "fail",
    }
    return {
        "gate_name": str(g5_config["effective_sample_gate"]),
        "disease_scope": str(g5_config["disease_scope"]),
        "headline_profile": str(g5_config["headline_profile"]),
        "source_qc_summary_path": str(qc_summary_path.relative_to(REPO_ROOT)),
        "source_pair_alignment_audit_path": str(pair_alignment_path.relative_to(REPO_ROOT)),
        "source_split_leakage_audit_path": str(split_leakage_path.relative_to(REPO_ROOT)),
        "current_g2_readiness_gate": str(qc_summary["readiness_gate"]),
        "current_g2_readiness_status": str(qc_summary["readiness_status"]),
        "store_trajectory_min_count": store_min_count,
        "current_full_microenvironment_trajectory_count": full_count,
        "current_mesoscopic_trajectory_count": mesoscopic_count,
        "current_paired_trajectory_count": paired_count,
        "current_train_eligible_record_count": train_eligible_count,
        "current_weak_teacher_record_count": int(qc_summary["weak_teacher_count"]),
        "current_reject_record_count": canonical_counts["rejected_trajectory_count"],
        "effective_sample_counts": effective_sample_counts,
        "effective_coverage": effective_coverage,
        "pair_status_counts": {str(key): int(value) for key, value in pair_status_counts.items()},
        "source_holdout_leakage_counts": {
            "trajectory_random_uid_overlap": int(split_leakage["trajectory_random_uid_overlap"]),
            "family_holdout_family_overlap": int(split_leakage["family_holdout_family_overlap"]),
            "regimen_holdout_regimen_overlap": int(split_leakage["regimen_holdout_regimen_overlap"]),
            "target_program_holdout_overlap": int(split_leakage["target_program_holdout_overlap"]),
            "content_hash_overlap": int(split_leakage["content_hash_overlap"]),
            "source_holdout_role_overlap": int(split_leakage["source_holdout_role_overlap"]),
        },
        "materialized_trajectory_gate_status": "passed" if materialized_gate_passed else "not_passed",
        "expansion_quota": dict(g5_config["expansion_quota"]),
        "quota_sum": float(g5_config["quota_sum"]),
        "required_gate_terms": [str(value) for value in g5_config["required_gate_terms"]],
        "candidate_boundary": dict(g5_config["candidate_boundary"]),
        "claim_upgrade_token": str(g5_config["upgrade_claim_allowed_after_all_gates"]),
        "gate_status": "passed" if materialized_gate_passed else "not_passed",
    }


def publication_scaling_transfer_summary(config: dict) -> dict[str, object]:
    readiness_config = publication_readiness_config(config)
    model_profiles = readiness_config["model_profiles"]
    return {
        "headline_model": str(model_profiles["headline_model"]),
        "scaling_profiles": [str(value) for value in model_profiles["scaling_profiles"]],
        "distillation_chain": list(model_profiles["distillation_chain"]),
        "efficiency_metrics": [str(value) for value in model_profiles["efficiency_metrics"]],
        "required_scaling_metrics": [str(value) for value in readiness_config["metric_catalog"]["scaling_transfer"]],
        "result_status": "scale_contract_declared_hpc_results_pending",
    }


def scale_registry_audit(config: dict) -> dict[str, object]:
    registry_config = neuronsifter_scale_registry(config)
    scale_rows = []
    for scale_name, scale_config in registry_config["checkpoint_scales"].items():
        scale_rows.append(
            {
                "checkpoint_scale": str(scale_name),
                "scale_label": str(scale_config["scale_label"]),
                "role": str(scale_config["role"]),
                "operator_hidden_dim": int(scale_config["operator_hidden_dim"]),
                "operator_depth": int(scale_config["operator_depth"]),
                "attention_head_count": int(scale_config["attention_head_count"]),
                "mechanism_state_token_count": int(scale_config["mechanism_state_token_count"]),
                "nominal_active_parameter_count": int(scale_config["nominal_active_parameter_count"]),
                "analytic_active_parameter_count": scale_parameter_estimate(config, str(scale_name)),
                "nominal_trainable_parameter_count": int(scale_config["nominal_trainable_parameter_count"]),
                "fsdp_resource_profile": str(scale_config["fsdp_resource_profile"]),
                "enabled_for_publication_claim": bool(scale_config["enabled_for_publication_claim"]),
            }
        )
    return {
        "registry_version": str(registry_config["registry_version"]),
        "default_training_checkpoint_scale": str(registry_config["default_training_checkpoint_scale"]),
        "default_inference_profile": str(registry_config["default_inference_profile"]),
        "shared_contract": dict(registry_config["shared_contract"]),
        "checkpoint_scales": scale_rows,
    }


def scale_consistency_summary(config: dict) -> dict[str, object]:
    registry_config = neuronsifter_scale_registry(config)
    return {
        "evaluation_status": "configured_pending_hpc_execution",
        "scale_order": ["neuronsifter_debug", "neuronsifter_student", "neuronsifter_large", "neuronsifter_xl"],
        "required_metrics": [
            "student_large_top_k_recall",
            "large_xl_rank_agreement",
            "tri_agreement",
            "coverage",
            "crps",
            "nll",
            "expert_drift",
            "ood_false_accept",
        ],
        "routing_policy": dict(registry_config["routing_policy"]),
        "acceptance_policy": "empirical_metrics_must_be_materialized_before_foundation_claim_upgrade",
    }


def distillation_summary(config: dict) -> dict[str, object]:
    registry_config = neuronsifter_scale_registry(config)
    return {
        "distillation_status": "configured_pending_smoke_and_holdout_execution",
        "paths": [
            {
                "teacher_scale": "neuronsifter_xl",
                "student_scale": "neuronsifter_large",
                "loss_terms": list(registry_config["distillation_losses"]["terms"].keys()),
            },
            {
                "teacher_scale": "neuronsifter_large",
                "student_scale": "neuronsifter_student",
                "loss_terms": list(registry_config["distillation_losses"]["terms"].keys()),
            },
        ],
        "selective_deployment_policy": "student_uncertainty_escalates_to_large_or_xl_before_acceptance",
    }


def teacher_mixture_summary(config: dict) -> dict[str, object]:
    mixture_config = neuronsifter_scale_registry(config)["teacher_mixture_sweep"]
    return {
        "sweep_status": "configured_pending_debug_student_grid",
        "sweep_scales": [str(value) for value in mixture_config["sweep_scales"]],
        "default_family_weights": dict(mixture_config["default_family_weights"]),
        "grid": list(mixture_config["grid"]),
        "sweep_gate": dict(mixture_config["sweep_gate"]),
    }


def fsdp_smoke_summary(config: dict) -> dict[str, object]:
    registry_config = neuronsifter_scale_registry(config)
    fsdp_profile = registry_config["fsdp_resource_profiles"]["xl_four_gpu_smoke"]
    artifact_root = resolve_repo_path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"])
    loader_config = config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]
    subdir_template = str(loader_config.get("formal_artifact_subdir_template", "")).strip()
    benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    manifest_candidates = []
    if subdir_template:
        manifest_candidates.append(
            artifact_root
            / subdir_template.format(benchmark_name=benchmark_name, profile_name="", seed=str(config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]))
            / "microenvironment_pretraining_manifest.json"
        )
    manifest_candidates.append(artifact_root / "microenvironment_pretraining_manifest.json")
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), manifest_candidates[0])
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    passed = (
        str(manifest_payload.get("checkpoint_scale", "")) == "neuronsifter_xl"
        and str(manifest_payload.get("distributed_strategy", "")) == "native_pytorch_fsdp"
        and int(manifest_payload.get("executed_steps", 0)) > 0
    )
    return {
        "smoke_status": "passed" if passed else "configured_pending_gpu03_execution",
        "checkpoint_scale": "neuronsifter_xl",
        "distributed_strategy": str(fsdp_profile["distributed_strategy"]),
        "partition": str(fsdp_profile["partition"]),
        "nodelist": str(fsdp_profile["nodelist"]),
        "gpu_count": int(fsdp_profile["gpu_count"]),
        "checkpoint_policy": str(fsdp_profile["checkpoint_policy"]),
        "manifest_path": str(manifest_path),
        "actual_trainable_parameter_count": int(manifest_payload.get("actual_trainable_parameter_count", 0)),
        "executed_steps": int(manifest_payload.get("executed_steps", 0)),
        "executed_micro_batches": int(manifest_payload.get("executed_micro_batches", 0)),
        "total_loss": float(manifest_payload.get("total_loss", 0.0)),
        "router_entropy": float(manifest_payload.get("router_entropy", 0.0)),
        "required_finite_checks": [
            "loss",
            "router_diagnostics",
            "uncertainty",
            "tri",
        ],
    }


def gpu_stochastic_smoke_summary(config: dict) -> dict[str, object]:
    smoke_config = config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]
    output_path = resolve_repo_path(str(smoke_config["output_path"]))
    if output_path.exists():
        payload = read_json_payload(output_path)
        if not isinstance(payload, dict):
            raise ValueError(f"GPU stochastic smoke summary must be a JSON object: {output_path}")
        return payload
    transition_backbone = (
        config["screening_pipeline"]["screening_benchmark"]
        ["model_architecture"]["rollout_operator"]["transition_backbone"]
    )
    sampler_config = transition_backbone["diffusion_sampler"]
    return {
        "smoke_status": "configured_pending_gpu03_execution",
        "device": str(smoke_config["device"]),
        "operator_kind": str(transition_backbone["kind"]),
        "required_sample_count": int(sampler_config["sample_count"]),
        "observed_sample_count": 0,
        "required_denoising_step_count": int(sampler_config["denoising_step_count"]),
        "observed_denoising_step_count": 0,
        "finite_checks": {
            "trajectory_std": False,
            "trajectory_quantiles": False,
            "ranking_stability": False,
            "operator_splitting_diagnostics": False,
            "sampler_diagnostics": False,
        },
        "missing_operator_diagnostics": [
            "operator_splitting_expert_utilization",
            "operator_splitting_router_entropy",
            "operator_splitting_species_attribution",
            "operator_splitting_coupling_matrix",
            "operator_splitting_score_norm_by_expert",
            "operator_splitting_uncertainty_by_expert",
            "operator_splitting_case_activation",
        ],
    }


def foundation_readiness_gate_summary(config: dict) -> dict[str, object]:
    claim_policy = neuronsifter_scale_registry(config)["foundation_candidate_claim_policy"]
    forbidden_wording = [str(value) for value in claim_policy["forbidden_pre_gate_wording"]]
    return {
        "gate_status": "not_passed",
        "current_allowed_wording": [str(value) for value in claim_policy["allowed_current_wording"]],
        "forbidden_pre_gate_claim_count": len(forbidden_wording),
        "forbidden_pre_gate_claim_ids": [value.replace(" ", "_").replace("-", "_") for value in forbidden_wording],
        "forbidden_pre_gate_wording_publication_policy": "withheld_from_public_reports",
        "unlock_gate": str(claim_policy["unlock_gate"]),
        "required_gate_terms": [str(value) for value in claim_policy["required_gate_terms"]],
    }


def framework_hardening_claim_scope(config: dict) -> dict[str, object]:
    configured_report_dir = Path(str(config["paper"]["paths"]["report_dir"]))
    report_dir = resolve_repo_path(str(configured_report_dir))
    evidence_paths = {
        "measurement_semantics": report_dir / "measurement_semantics_audit.json",
        "scaler_leakage": report_dir / "scaler_leakage_audit.json",
        "dynamics_calibration": report_dir / "dynamics_calibration_audit.json",
        "crossscale_alignment": report_dir / "crossscale_alignment_audit.json",
    }
    if configured_report_dir.is_absolute():
        try:
            report_identifier_dir = configured_report_dir.relative_to(REPO_ROOT)
        except ValueError:
            report_identifier_dir = None
    else:
        report_identifier_dir = configured_report_dir
    missing_artifacts = [
        (
            (report_identifier_dir / path.name).as_posix()
            if report_identifier_dir is not None
            else f"report_artifact:{path.name}"
        )
        for path in evidence_paths.values()
        if not path.is_file()
    ]
    if missing_artifacts:
        return {
            "evidence_status": "evidence_unavailable",
            "missing_required_evidence_artifacts": missing_artifacts,
            "measurement_semantics_status": "evidence_unavailable",
            "scaler_leakage_status": "evidence_unavailable",
            "input_output_contract_status": "evidence_unavailable",
            "dynamics_calibration_status": "evidence_unavailable",
            "dynamics_accuracy_claim_scope": "mechanistic_teacher_pretraining_only",
            "dynamics_external_calibration_claim_status": "not_claimed",
            "crossscale_alignment_status": "evidence_unavailable",
            "crossscale_alignment_claim_scope": "paired_sample_sources_plus_distributional_weak_sources",
            "full_crossscale_alignment_claim_status": "not_claimed",
            "prohibited_evidence_upgrades": [
                "measurement_semantics_contract_completion",
                "bundle_scaler_artifacts",
                "external_calibrated_dynamics_accuracy",
                "full_crossscale_alignment",
            ],
        }
    measurement_payload = read_json_payload(evidence_paths["measurement_semantics"])
    scaler_payload = read_json_payload(evidence_paths["scaler_leakage"])
    dynamics_payload = read_json_payload(evidence_paths["dynamics_calibration"])
    alignment_payload = read_json_payload(evidence_paths["crossscale_alignment"])
    dynamics_claim_scope = "external_calibration_validated"
    if str(dynamics_payload["external_calibration_claim_status"]) != "external_calibration_validated":
        dynamics_claim_scope = "mechanistic_teacher_pretraining_only"
    alignment_claim_scope = "paired_sample_sources_only"
    if str(alignment_payload["full_crossscale_alignment_claim_status"]) != "paired_anchor_sources_only":
        alignment_claim_scope = "paired_sample_sources_plus_distributional_weak_sources"
    input_output_status = "explicit_semantics_and_train_context_scaling_declared"
    if str(measurement_payload["status"]) != "contract_satisfied" or str(scaler_payload["status"]) != "bundle_scaler_artifacts_materialized":
        input_output_status = "input_output_contract_incomplete"
    prohibited_evidence_upgrades = []
    if dynamics_claim_scope != "external_calibration_validated":
        prohibited_evidence_upgrades.append("external_calibrated_dynamics_accuracy")
    if alignment_claim_scope != "paired_sample_sources_only":
        prohibited_evidence_upgrades.append("full_crossscale_alignment")
    return {
        "evidence_status": "evidence_available",
        "missing_required_evidence_artifacts": [],
        "measurement_semantics_status": str(measurement_payload["status"]),
        "scaler_leakage_status": str(scaler_payload["status"]),
        "input_output_contract_status": input_output_status,
        "dynamics_calibration_status": str(dynamics_payload["status"]),
        "dynamics_accuracy_claim_scope": dynamics_claim_scope,
        "dynamics_external_calibration_claim_status": str(dynamics_payload["external_calibration_claim_status"]),
        "crossscale_alignment_status": str(alignment_payload["status"]),
        "crossscale_alignment_claim_scope": alignment_claim_scope,
        "full_crossscale_alignment_claim_status": str(alignment_payload["full_crossscale_alignment_claim_status"]),
        "prohibited_evidence_upgrades": prohibited_evidence_upgrades,
    }


def publication_claim_discipline_audit(config: dict) -> dict[str, object]:
    readiness_config = publication_readiness_config(config)
    hardening_scope = framework_hardening_claim_scope(config)
    evidence_unavailable = hardening_scope["evidence_status"] == "evidence_unavailable"
    return {
        "method_claim": str(readiness_config["research_positioning"]["method_claim"]),
        "data_boundary": str(readiness_config["research_positioning"]["data_boundary"]),
        "endpoint_policy": str(readiness_config["research_positioning"]["endpoint_policy"]),
        "prohibited_method_claims": [str(value) for value in readiness_config["research_positioning"]["prohibited_method_claims"]],
        "external_tool_role_policy": "adapters_validators_teachers_baselines_downstream_validators",
        "framework_hardening_evidence_status": hardening_scope["evidence_status"],
        "missing_required_evidence_artifacts": hardening_scope["missing_required_evidence_artifacts"],
        "measurement_semantics_status": hardening_scope["measurement_semantics_status"],
        "scaler_leakage_status": hardening_scope["scaler_leakage_status"],
        "input_output_contract_status": hardening_scope["input_output_contract_status"],
        "dynamics_calibration_status": hardening_scope["dynamics_calibration_status"],
        "dynamics_accuracy_claim_scope": hardening_scope["dynamics_accuracy_claim_scope"],
        "dynamics_external_calibration_claim_status": hardening_scope["dynamics_external_calibration_claim_status"],
        "crossscale_alignment_status": hardening_scope["crossscale_alignment_status"],
        "crossscale_alignment_claim_scope": hardening_scope["crossscale_alignment_claim_scope"],
        "full_crossscale_alignment_claim_status": hardening_scope["full_crossscale_alignment_claim_status"],
        "prohibited_evidence_upgrades": hardening_scope["prohibited_evidence_upgrades"],
        "wording_status": (
            "evidence_unavailable_claims_withheld"
            if evidence_unavailable
            else "evidence_scoped_guardrail_contract_declared"
        ),
    }


def runnable_baseline_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    honesty_policy = config["paper_experiments"].get("baseline_honesty_policy", {})
    main_table_modes = {str(value) for value in honesty_policy.get("main_table_adapter_modes", [])}
    shared_frame = safe_concat(shared_benchmark_paths(config))
    runnable_baseline_root = REPO_ROOT / str(config["baseline_adapter_registry"]["output_roots"]["runnable_baseline_root"])
    for model_name in [
        "aligned_gene_logistic_regression",
        "aligned_metabolite_logistic_regression",
        "shared_latent_dual_encoder_classifier",
        "tabular_jepa_auxiliary_encoder",
    ]:
        rows.append(
            {
                "model_name": model_name,
                "family": "classical_in_repo",
                "adapter_mode": "literature_only",
                "comparison_status": "context_only",
                "auroc": format_mean_std(grouped_baseline_values(shared_frame, model_name, "auroc")),
                "auprc": format_mean_std(grouped_baseline_values(shared_frame, model_name, "auprc")),
                "balanced_accuracy": format_mean_std(grouped_baseline_values(shared_frame, model_name, "balanced_accuracy")),
                "macro_f1": format_mean_std(grouped_baseline_values(shared_frame, model_name, "macro_f1")),
                "transition_mse": "n/a",
            }
        )
    disease_program = str(config["baseline_adapter_registry"]["default_suite"]["disease_program"])
    runnable_entries = {
        str(row["baseline_name"]): row
        for row in config["paper_experiments"].get("runnable_baselines", [])
    }
    for baseline_name, entry in runnable_entries.items():
        aggregate_path = runnable_baseline_root / disease_program / baseline_name / "result_catalog.tsv"
        fold_paths = sorted((runnable_baseline_root / disease_program / baseline_name).glob("*/result_catalog.tsv"))
        adapter_mode = str(entry.get("adapter_mode", ""))
        implementation_status = str(entry.get("implementation_status", ""))
        ready_statuses = {"direct_external_wrapper_ready", "local_matched_split_wrapper_ready"}
        runnable_frame = safe_concat([aggregate_path] if aggregate_path.exists() else fold_paths)
        comparison_status = "blocked_external_wrapper"
        if (
            implementation_status in ready_statuses
            and adapter_mode in main_table_modes
            and runnable_baseline_contract_satisfied(
                runnable_frame,
                baseline_name,
                list(entry.get("task_coverage", [])),
            )
        ):
            comparison_status = "eligible_main_table"
        elif implementation_status in ready_statuses and adapter_mode in main_table_modes:
            comparison_status = "incomplete_runtime_contract"
        elif implementation_status in ready_statuses:
            comparison_status = "context_only"
        if comparison_status == "eligible_main_table" and not runnable_frame.empty:
            auroc = format_mean_std(grouped_runnable_baseline_values(runnable_frame, baseline_name, "auroc"))
            auprc = format_mean_std(grouped_runnable_baseline_values(runnable_frame, baseline_name, "auprc"))
            balanced_accuracy = format_mean_std(grouped_runnable_baseline_values(runnable_frame, baseline_name, "balanced_accuracy"))
            macro_f1 = format_mean_std(grouped_runnable_baseline_values(runnable_frame, baseline_name, "macro_f1"))
            transition_mse = format_mean_std(grouped_runnable_baseline_values(runnable_frame, baseline_name, "transition_mse"))
        else:
            auroc = "pending"
            auprc = "pending"
            balanced_accuracy = "pending"
            macro_f1 = "pending"
            transition_mse = "pending"
        rows.append(
            {
                "model_name": baseline_name,
                "family": str(entry["baseline_family"]),
                "adapter_mode": adapter_mode,
                "comparison_status": comparison_status,
                "auroc": auroc,
                "auprc": auprc,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "transition_mse": transition_mse,
            }
        )
    return rows


def headline_external_baseline_rows(config: dict) -> list[dict[str, object]]:
    runtime_rows = {
        str(row["model_name"]): row
        for row in runnable_baseline_rows(config)
        if str(row.get("comparison_status", "")) == "eligible_main_table"
    }
    configured_rows: list[dict[str, object]] = []
    honesty_policy = config["paper_experiments"].get("baseline_honesty_policy", {})
    main_table_modes = {str(value) for value in honesty_policy.get("main_table_adapter_modes", [])}
    for entry in config["paper_experiments"].get("runnable_baselines", []):
        if str(entry.get("adapter_mode", "")) not in main_table_modes:
            continue
        baseline_name = str(entry["baseline_name"])
        runtime_row = runtime_rows.get(baseline_name)
        if runtime_row is not None:
            configured_rows.append(runtime_row)
            continue
        configured_rows.append(
            {
                "model_name": baseline_name,
                "family": str(entry["baseline_family"]),
                "adapter_mode": str(entry.get("adapter_mode", "")),
                "comparison_status": str(entry.get("implementation_status", "pending")),
                "auroc": "pending",
                "auprc": "pending",
                "balanced_accuracy": "pending",
                "macro_f1": "pending",
                "transition_mse": "pending",
            }
        )
    return configured_rows


def profile_status_rows(config: dict, paper_track: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_name in profile_status_names(paper_track):
        if profile_name == "shared_latent_formal":
            paths = shared_benchmark_paths(config)
            resolved_profile_name = profile_name
        else:
            resolved_profile_name, paths = resolved_screening_profile(config, profile_name)
        rows.append(
            {
                "profile_name": profile_name,
                "available_seed_count": len(paths),
                "status": "available" if paths else "pending",
                "artifact_profile_name": resolved_profile_name if paths else "",
            }
        )
    return rows


def active_learning_replay_rows(config: dict) -> list[dict[str, object]]:
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    result_catalog_path = (
        Path(benchmark["artifact_root"]).resolve()
        / benchmark["benchmark_name"]
        / "active_learning_replay"
        / "result_catalog.tsv"
    )
    if not result_catalog_path.exists():
        return []
    frame = read_tsv_frame(result_catalog_path)
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    frame["policy_name"] = frame["note"].astype(str).str.split(":").str[0]
    frame["top_k"] = frame["note"].astype(str).str.extract(r"top_k=(\d+)").fillna("0").astype(int)
    for (policy_name, top_k), policy_frame in frame.groupby(["policy_name", "top_k"], sort=True):
        metric_lookup = {
            str(metric_name): format_mean_std(policy_frame.loc[policy_frame["metric_name"] == metric_name, "metric_value"].astype(float).tolist())
            for metric_name in sorted(policy_frame["metric_name"].astype(str).unique().tolist())
        }
        rows.append(
            {
                "policy_name": str(policy_name),
                "top_k": int(top_k),
                "top_k_validation_yield": metric_lookup.get("top_k_validation_yield", "pending"),
                "anchor_recovery_fraction": metric_lookup.get("anchor_recovery_fraction", "pending"),
                "accepted_candidate_quality": metric_lookup.get("accepted_candidate_quality", "pending"),
                "coverage_adjusted_gain": metric_lookup.get("coverage_adjusted_gain", "pending"),
            }
        )
    return rows


def active_learning_loop_summary_payload(config: dict) -> dict[str, object]:
    active_config = config["screening_pipeline"]["screening_benchmark"]["active_learning"]
    relative_paths = dict(active_config["relative_paths"])
    root = resolve_repo_path(str(active_config["artifact_root"]))
    round_payloads: list[dict[str, object]] = []
    completed_round_count = 0
    for round_config in active_config["active_rounds"]:
        round_uid = str(round_config["round_uid"])
        request_path = root / str(relative_paths["request_batch_template"]).format(round_uid=round_uid)
        result_path = root / str(relative_paths["result_manifest_template"]).format(round_uid=round_uid)
        qc_path = root / str(relative_paths["qc_summary_template"]).format(round_uid=round_uid)
        report_path = root / str(relative_paths["round_report_template"]).format(round_uid=round_uid)
        result_materialized = result_path.exists()
        qc_materialized = qc_path.exists()
        report_materialized = report_path.exists()
        round_complete = result_materialized and qc_materialized and report_materialized
        completed_round_count += int(round_complete)
        round_payloads.append(
            {
                "round_uid": round_uid,
                "round_index": int(round_config["round_index"]),
                "update_policy": str(round_config["update_policy"]),
                "request_manifest": str(request_path.relative_to(REPO_ROOT)),
                "result_manifest": str(result_path.relative_to(REPO_ROOT)),
                "qc_summary": str(qc_path.relative_to(REPO_ROOT)),
                "round_report": str(report_path.relative_to(REPO_ROOT)),
                "request_materialized": request_path.exists(),
                "result_materialized": result_materialized,
                "qc_materialized": qc_materialized,
                "round_report_materialized": report_materialized,
                "round_status": "empirical_round_complete" if round_complete else "requires_empirical_round_execution",
            }
        )
    required_round_count = int(active_config["full_round_count"])
    return {
        "contract_name": str(active_config["contract_name"]),
        "status": (
            "active_learning_empirical_loop_complete"
            if completed_round_count >= required_round_count
            else "active_learning_empirical_results_pending"
        ),
        "artifact_root": str(root.relative_to(REPO_ROOT)),
        "required_round_count": required_round_count,
        "completed_round_count": completed_round_count,
        "full_tasks_per_round": int(active_config["full_tasks_per_round"]),
        "batch_quota": dict(active_config["batch_quota"]),
        "simulator_portfolio": [str(value) for value in active_config["simulator_portfolio"]],
        "rounds": round_payloads,
    }


def active_learning_empirical_gap_audit_payload(config: dict) -> dict[str, object]:
    loop_summary = active_learning_loop_summary_payload(config)
    missing_rounds = [
        str(round_payload["round_uid"])
        for round_payload in loop_summary["rounds"]
        if str(round_payload["round_status"]) != "empirical_round_complete"
    ]
    active_config = config["screening_pipeline"]["screening_benchmark"]["active_learning"]
    return {
        "audit_uid": "active_learning_empirical_gap_audit",
        "gap_status": "pass" if not missing_rounds else "requires_active_learning_execution",
        "loop_status": str(loop_summary["status"]),
        "missing_empirical_rounds": missing_rounds,
        "required_round_count": int(loop_summary["required_round_count"]),
        "completed_round_count": int(loop_summary["completed_round_count"]),
        "required_tasks_per_round": int(active_config["full_tasks_per_round"]),
        "fail_fast_on_pending_empirical_artifacts": bool(active_config["fail_fast_on_pending_empirical_artifacts"]),
        "claim_policy": "no_active_learning_empirical_gain_claim_until_all_rounds_complete",
    }


def screening_model_kwargs(checkpoint: dict, config: dict) -> dict[str, object]:
    model_kwargs = dict(checkpoint.get("model_kwargs", {}))
    if "region_count" not in model_kwargs:
        model_kwargs["region_count"] = len(config["spatial_atlas_registry"]["regions"])
    if "species_count" not in model_kwargs:
        model_kwargs["species_count"] = len(config["dynamics_ready_registry"]["species"])
    if "rollout_step_count" not in model_kwargs:
        model_kwargs["rollout_step_count"] = int(config["dynamics_ready_registry"]["time_grid"]["step_count"])
    if "adjacency_matrix" not in model_kwargs:
        model_kwargs["adjacency_matrix"] = node_adjacency_matrix(config).tolist()
    if "region_prior_feature_dim" not in model_kwargs:
        model_kwargs["region_prior_feature_dim"] = int(default_region_prior(config).shape[1])
    if "default_region_prior" not in model_kwargs:
        model_kwargs["default_region_prior"] = default_region_prior(config).tolist()
    if "baseline_field_policy" not in model_kwargs:
        model_kwargs["baseline_field_policy"] = baseline_field_policy_from_config(config)
    if "rollout_update_policy" not in model_kwargs:
        model_kwargs["rollout_update_policy"] = rollout_update_policy_from_config(config)
    if "solver_state_layout" not in model_kwargs:
        model_kwargs["solver_state_layout"] = solver_state_layout_from_config(config)
    if "solver_config" not in model_kwargs:
        model_kwargs["solver_config"] = solver_config_from_config(config)
    if "world_state_posterior_config" not in model_kwargs:
        model_kwargs["world_state_posterior_config"] = world_state_posterior_config_from_config(config)
    return model_kwargs


def available_checkpoint_path(paths: list[Path], relative_model_path: list[str]) -> Path | None:
    for path in paths:
        candidate = path
        for part in relative_model_path:
            candidate = candidate / part
        if candidate.exists():
            return candidate
    return None


def average_latency_ms(callable_fn, warmup_steps: int = 3, repetitions: int = 10) -> float:
    with torch.inference_mode():
        for _ in range(warmup_steps):
            callable_fn()
        started_at = time.perf_counter()
        for _ in range(repetitions):
            callable_fn()
        elapsed = time.perf_counter() - started_at
    return elapsed * 1000.0 / repetitions


def checkpoint_size_mb(path: Path) -> float:
    return float(path.stat().st_size) / (1024.0 * 1024.0)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def efficiency_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    shared_paths = []
    try:
        shared_profile = load_experiment_family("shared_latent_formal", config)
        runtime = shared_profile["runtime"]
        for seed_value in runtime.get("seed_values", []):
            shared_paths.append(
                (REPO_ROOT / config["benchmark_registry"]["artifact_root"]).resolve()
                / str(runtime["benchmark_name_template"]).format(seed=seed_value)
                / "production"
            )
    except (FileNotFoundError, KeyError, ValueError):
        shared_paths = []
    screening_paths = []
    try:
        profile = load_experiment_family("dynamics_mainline_ad", config)
        runtime = profile["runtime"]
        for seed_value in runtime.get("seed_values", []):
            screening_paths.append(
                (REPO_ROOT / runtime["artifact_root"]).resolve()
                / str(runtime["benchmark_name_template"]).format(seed=seed_value)
                / "production"
            )
    except (FileNotFoundError, KeyError, ValueError):
        screening_paths = []

    shared_checkpoint = available_checkpoint_path(shared_paths, ["shared_latent_dual_encoder_classifier", "model_weights.pt"])
    if shared_checkpoint is not None:
        checkpoint = torch.load(shared_checkpoint, map_location="cpu")
        model = SharedLatentDualEncoder(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        input_dim = int(checkpoint["model_kwargs"]["transcriptomics_input_dim"])
        batch = torch.randn(16, input_dim, dtype=torch.float32)
        rows.append(
            {
                "model_name": "shared_latent_dual_encoder_classifier",
                "status": "available",
                "parameter_count": parameter_count(model),
                "checkpoint_size_mb": checkpoint_size_mb(shared_checkpoint),
                "latency_ms": average_latency_ms(lambda: model("transcriptomics", batch)),
                "device": "cpu",
            }
        )
    else:
        rows.append({"model_name": "shared_latent_dual_encoder_classifier", "status": "pending"})

    jepa_checkpoint = available_checkpoint_path(shared_paths, ["tabular_jepa_auxiliary_encoder", "model_weights.pt"])
    if jepa_checkpoint is not None:
        checkpoint = torch.load(jepa_checkpoint, map_location="cpu")
        model = TabularJEPAAuxiliaryEncoder(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        input_dim = int(checkpoint["model_kwargs"]["transcriptomics_input_dim"])
        batch = torch.randn(16, input_dim, dtype=torch.float32)
        rows.append(
            {
                "model_name": "tabular_jepa_auxiliary_encoder",
                "status": "available",
                "parameter_count": parameter_count(model),
                "checkpoint_size_mb": checkpoint_size_mb(jepa_checkpoint),
                "latency_ms": average_latency_ms(lambda: model("transcriptomics", batch)),
                "device": "cpu",
            }
        )
    else:
        rows.append({"model_name": "tabular_jepa_auxiliary_encoder", "status": "pending"})

    dynamics_checkpoint = available_checkpoint_path(screening_paths, ["ad_screening_system", "model_weights.pt"])
    if dynamics_checkpoint is not None:
        checkpoint = torch.load(dynamics_checkpoint, map_location="cpu")
        model = ADScreeningSystem(**screening_model_kwargs(checkpoint, config))
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model.eval()
        patient_batch = torch.randn(8, int(model.patient_state_dim), dtype=torch.float32)
        regimen_batch = torch.randn(16, int(model.regimen_feature_dim), dtype=torch.float32)
        tri_context = {
            "coverage": torch.ones((8, 16), dtype=torch.float32),
            "ood_score": torch.full((8, 16), 0.5, dtype=torch.float32),
        }
        rows.append(
            {
                "model_name": "ad_screening_system",
                "status": "available",
                "parameter_count": parameter_count(model),
                "checkpoint_size_mb": checkpoint_size_mb(dynamics_checkpoint),
                "latency_ms": average_latency_ms(lambda: model.score_regimens(patient_batch, regimen_batch, tri_context)),
                "device": "cpu",
            }
        )
    else:
        rows.append({"model_name": "ad_screening_system", "status": "pending"})
    return rows


def table_tex_from_rows(column_names: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    column_spec = "l" * len(column_names)
    body_lines = [" & ".join(latex_cell(value) for value in row) + r" \\" for row in rows]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            " & ".join(column_names) + r" \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )


def latex_cell(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def placeholder_plot(output_stem: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.35, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_method_overview(output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 2.8))
    ax.axis("off")
    boxes = [
        ("Public omics", (0.03, 0.32), "#4E79A7"),
        ("Sample evidence\\nencoder", (0.21, 0.32), "#59A14F"),
        ("Structural\\nprior", (0.40, 0.58), "#F28E2B"),
        ("Field rollout\\noperator", (0.46, 0.32), "#E15759"),
        ("Observation +\\nscreening heads", (0.68, 0.32), "#B07AA1"),
        ("Quantitative case\\nvalidation", (0.88, 0.32), "#76B7B2"),
    ]
    for label, (x_position, y_position), color in boxes:
        ax.add_patch(plt.Rectangle((x_position, y_position), 0.14, 0.22, color=color, alpha=0.92))
        ax.text(x_position + 0.07, y_position + 0.11, label, ha="center", va="center", color="white", fontsize=10, weight="bold")
    arrow = dict(arrowstyle="->", lw=1.8, color="#222222")
    ax.annotate("", xy=(0.21, 0.43), xytext=(0.17, 0.43), arrowprops=arrow)
    ax.annotate("", xy=(0.46, 0.43), xytext=(0.35, 0.43), arrowprops=arrow)
    ax.annotate("", xy=(0.46, 0.50), xytext=(0.47, 0.58), arrowprops=arrow)
    ax.annotate("", xy=(0.68, 0.43), xytext=(0.60, 0.43), arrowprops=arrow)
    ax.annotate("", xy=(0.88, 0.43), xytext=(0.82, 0.43), arrowprops=arrow)
    ax.text(0.52, 0.18, "patient posterior + regimen controls + structural priors", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_metric_bar(rows: list[dict[str, object]], labels: list[str], values: list[float], output_stem: Path, title: str, ylabel: str) -> None:
    if not rows or not values:
        placeholder_plot(output_stem, title, "Formal results pending")
        return
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(np.arange(len(values)), values, color="#4E79A7")
    ax.set_xticks(np.arange(len(values)), labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def plot_structural_prior(output_stem: Path, config: dict) -> None:
    path = (REPO_ROOT / config["structural_prior_registry"]["local_root"]).resolve() / config["structural_prior_registry"]["relative_paths"]["region_prior_catalog"]
    if not path.exists():
        placeholder_plot(output_stem, "Structural Priors", "Structural prior catalog pending")
        return
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        placeholder_plot(output_stem, "Structural Priors", "Structural prior catalog empty")
        return
    frame = frame.loc[frame["disease_program"] == "ad_control"].copy()
    for column_name in ["disease_vulnerability_score"]:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0)
    summary = frame.groupby("region_id", as_index=False)["disease_vulnerability_score"].mean().sort_values("disease_vulnerability_score", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.bar(summary["region_id"].tolist(), summary["disease_vulnerability_score"].tolist(), color="#59A14F")
    ax.set_ylabel("Disease vulnerability score")
    ax.set_title("AD structural prior summary")
    ax.tick_params(axis="x", rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paper_track = load_paper_track(args.paper_track)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    paper_paths = config["paper"]["paths"]
    figure_dir = (REPO_ROOT / (args.figure_dir or paper_paths["figure_dir"])).resolve()
    table_dir = (REPO_ROOT / (args.table_dir or paper_paths["table_dir"])).resolve()
    report_dir = (REPO_ROOT / (args.report_dir or paper_paths["report_dir"])).resolve()
    ensure_directory(figure_dir)
    ensure_directory(table_dir)
    ensure_directory(report_dir)

    mainline_rows = dynamics_mainline_rows(config, paper_track)
    ablation_rows = dynamics_ablation_rows(config, paper_track)
    task_rows = dynamics_task_summary_rows(config, paper_track)
    capability_rows = foundation_capability_summary_rows(config, paper_track)
    case_rows = quantitative_case_validation_rows(config, paper_track)
    clinical_direction_rows = clinical_direction_case_study_matrix_rows(config)
    strict_rows = strict_case_summary_rows(config, paper_track)
    baseline_rows = runnable_baseline_rows(config)
    headline_external_rows = headline_external_baseline_rows(config)
    efficiency = efficiency_rows(config)
    status_rows = profile_status_rows(config, paper_track)
    active_learning_rows = active_learning_replay_rows(config)
    publication_claim_rows = publication_claim_metric_rows(config)
    publication_baseline_rows = publication_baseline_result_rows(config)
    publication_ablation_rows = publication_ablation_result_rows(config)
    publication_selective_risk_rows_payload = publication_selective_risk_rows(config)
    publication_figure_rows = publication_figure_manifest_rows(config)
    publication_g5_audit = publication_g5_effective_sample_audit(config)
    publication_scaling_summary = publication_scaling_transfer_summary(config)
    publication_discipline_audit = publication_claim_discipline_audit(config)
    scale_registry_payload = scale_registry_audit(config)
    scale_consistency_payload = scale_consistency_summary(config)
    distillation_payload = distillation_summary(config)
    teacher_mixture_payload = teacher_mixture_summary(config)
    fsdp_smoke_payload = fsdp_smoke_summary(config)
    gpu_stochastic_smoke_payload = gpu_stochastic_smoke_summary(config)
    foundation_readiness_payload = foundation_readiness_gate_summary(config)
    active_learning_loop_payload = active_learning_loop_summary_payload(config)
    active_learning_gap_payload = active_learning_empirical_gap_audit_payload(config)
    pending_formal_surfaces = pending_formal_result_surfaces(
        config,
        paper_track,
        status_rows,
        publication_claim_rows,
        publication_baseline_rows,
        publication_ablation_rows,
        publication_selective_risk_rows_payload,
    )
    if pending_formal_surfaces and not args.allow_pending:
        raise RuntimeError(
            "Publication main tables require materialized formal result surfaces: "
            + ", ".join(pending_formal_surfaces)
        )
    schema_config = config["schemas"]
    write_tsv(
        report_dir / "claim_metric_catalog.tsv",
        [str(value) for value in schema_config["publication_claim_metric_fields"]],
        publication_claim_rows,
    )
    write_tsv(
        report_dir / "baseline_result_matrix.tsv",
        [str(value) for value in schema_config["publication_baseline_result_fields"]],
        publication_baseline_rows,
    )
    write_tsv(
        report_dir / "ablation_result_catalog.tsv",
        [str(value) for value in schema_config["publication_ablation_result_fields"]],
        publication_ablation_rows,
    )
    write_tsv(
        report_dir / "selective_risk_catalog.tsv",
        [str(value) for value in schema_config["publication_selective_risk_fields"]],
        publication_selective_risk_rows_payload,
    )
    write_json(report_dir / "g5_effective_sample_audit.json", publication_g5_audit)
    write_json(report_dir / "scaling_transfer_summary.json", publication_scaling_summary)
    write_json(report_dir / "figure_manifest.json", publication_figure_rows)
    write_json(report_dir / "claim_discipline_audit.json", publication_discipline_audit)
    scale_artifacts = neuronsifter_scale_registry(config)["scale_artifacts"]
    write_json(report_dir / str(scale_artifacts["scale_registry_audit"]), scale_registry_payload)
    write_json(report_dir / str(scale_artifacts["scale_consistency_summary"]), scale_consistency_payload)
    write_json(report_dir / str(scale_artifacts["distillation_summary"]), distillation_payload)
    write_json(report_dir / str(scale_artifacts["teacher_mixture_summary"]), teacher_mixture_payload)
    write_json(report_dir / str(scale_artifacts["fsdp_smoke_summary"]), fsdp_smoke_payload)
    write_json(report_dir / str(scale_artifacts["gpu_stochastic_smoke_summary"]), gpu_stochastic_smoke_payload)
    write_json(report_dir / str(scale_artifacts["foundation_readiness_gate"]), foundation_readiness_payload)
    write_json(report_dir / "active_learning_loop_summary.json", active_learning_loop_payload)
    write_json(report_dir / "active_learning_empirical_gap_audit.json", active_learning_gap_payload)

    plot_method_overview(figure_dir / "dynamics_method_overview")
    plot_metric_bar(
        mainline_rows,
        [row["profile_name"] for row in mainline_rows if row["status"] == "available"],
        [
            float(str(row["regimen_ndcg"]).split(" $\\pm$ ")[0])
            for row in mainline_rows
            if row["status"] == "available" and row["regimen_ndcg"] != "pending"
        ],
        figure_dir / "dynamics_mainline_summary",
        "Dynamics Mainline Regimen Ranking",
        "Regimen NDCG@10",
    )
    plot_metric_bar(
        ablation_rows,
        [row["profile_name"].replace("dynamics_", "") for row in ablation_rows if row["status"] == "available"],
        [
            float(str(row["regimen_ndcg"]).split(" $\\pm$ ")[0])
            for row in ablation_rows
            if row["status"] == "available" and row["regimen_ndcg"] != "pending"
        ],
        figure_dir / "dynamics_ablation_summary",
        "Dynamics Ablation Summary",
        "Regimen NDCG@10",
    )
    plot_metric_bar(
        case_rows,
        [row["intervention_label"] for row in case_rows],
        [float(str(row["standardized_effect_error"]).split(" $\\pm$ ")[0]) for row in case_rows],
        figure_dir / "dynamics_case_validation_summary",
        "Quantitative Case Validation",
        "Standardized Effect Error",
    )
    plot_structural_prior(figure_dir / "structural_prior_summary", config)

    write_text(
        table_dir / "foundation_capability_summary_table.tex",
        table_tex_from_rows(
            ["Capability Family", "Capability", "Pass Fraction", "Status"],
            [
                [
                    str(row["capability_family"]),
                    str(row["capability_name"]),
                    str(row["pass_fraction"]),
                    str(row["status"]),
                ]
                for row in capability_rows
            ],
            "Foundation-twin capability acceptance summary across hard-gate, engineering, and high-order capability families.",
            "tab:foundationcapabilitysummary",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_task_summary_table.tex",
        table_tex_from_rows(
            ["Task", "Family", "Primary Metric", "Value", "Status"],
            [
                [
                    str(row["task_name"]),
                    str(row["task_family"]),
                    str(row["primary_metric_label"]),
                    str(row["primary_metric_value"]),
                    str(row["status"]),
                ]
                for row in task_rows
            ]
            or [[
                "sample_state_inference_and_uq",
                "state_inference",
                "disease_auroc",
                "pending",
                "pending",
            ]],
            "Four-task KDD-facing summary of the AD-first computational-validation twin.",
            "tab:dynamicstasksummary",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_headline_external_baseline_table.tex",
        table_tex_from_rows(
            ["Model", "Family", "Adapter Mode", "Status", "AUROC", "AUPRC", "Balanced Acc.", "Macro-F1", "Transition MSE"],
            [
                [
                    str(row["model_name"]),
                    str(row["family"]),
                    str(row["adapter_mode"]),
                    str(row["comparison_status"]),
                    str(row["auroc"]),
                    str(row["auprc"]),
                    str(row["balanced_accuracy"]),
                    str(row["macro_f1"]),
                    str(row["transition_mse"]),
                ]
                for row in headline_external_rows
            ]
            or [[
                "pending",
                "pending",
                "direct_external_wrapper",
                "pending",
                "pending",
                "pending",
                "pending",
                "pending",
                "pending",
            ]],
            "Headline external baselines eligible for direct comparison with the screening twin.",
            "tab:dynamicsheadlineexternalbaselines",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_mainline_table.tex",
        table_tex_from_rows(
            ["Profile", "Status", "Seeds", "Disease AUROC", "Disease AUPRC", "ECE", "NLL", "Disease MI", "Target NDCG@10", "Regimen NDCG@10", "Regimen MI", "Regimen E-Std", "Conc. NRMSE", "Flux NRMSE", "Route Hit", "Std. Effect Error", "Dir. Hit", "Rank Hit", "TRI Gate", "Coverage", "Risk-Cov AUC", "Accept Benefit", "OOD AUROC"],
            [
                [
                    str(row["profile_name"]),
                    str(row["status"]),
                    str(row["seed_count"]),
                    str(row["disease_auroc"]),
                    str(row["disease_auprc"]),
                    str(row["disease_ece"]),
                    str(row["disease_nll"]),
                    str(row["disease_mutual_information"]),
                    str(row["target_ndcg"]),
                    str(row["regimen_ndcg"]),
                    str(row["regimen_mutual_information"]),
                    str(row["regimen_epistemic_std"]),
                    str(row["concentration_nrmse"]),
                    str(row["flux_nrmse"]),
                    str(row["route_ordering_hit"]),
                    str(row["standardized_effect_error"]),
                    str(row["direction_hit"]),
                    str(row["rank_recovery_hit"]),
                    str(row["tri_gate_consistency"]),
                    str(row["coverage"]),
                    str(row["risk_coverage_auc"]),
                    str(row["acceptance_benefit"]),
                    str(row["ood_auroc"]),
                ]
                for row in mainline_rows
            ],
            "Main AD public screening-system results with human-atlas structural priors.",
            "tab:dynamicsmainline",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_ablation_table.tex",
        table_tex_from_rows(
            ["Profile", "Status", "Seeds", "Disease AUROC", "ECE", "Regimen NDCG@10", "Conc. NRMSE", "Std. Effect Error", "Coverage", "Risk-Cov AUC"],
            [
                [
                    str(row["profile_name"]),
                    str(row["status"]),
                    str(row["seed_count"]),
                    str(row["disease_auroc"]),
                    str(row["disease_ece"]),
                    str(row["regimen_ndcg"]),
                    str(row["concentration_nrmse"]),
                    str(row["standardized_effect_error"]),
                    str(row["coverage"]),
                    str(row["risk_coverage_auc"]),
                ]
                for row in ablation_rows
            ],
            "Ablations for structural priors, PK/PD priors, TRI, and local response heads.",
            "tab:dynamicsablation",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_case_validation_table.tex",
        table_tex_from_rows(
            ["Case", "Intervention", "Seeds", "Biomarker Error", "Direction Hit", "Rank Hit", "TRI Gate", "Clinical Alignment"],
            [
                [
                    str(row["case_study_name"]),
                    str(row["intervention_label"]),
                    str(row["seed_count"]),
                    str(row["biomarker_effect_error"]),
                    str(row["direction_hit"]),
                    str(row["rank_recovery_hit"]),
                    str(row["tri_gate_consistency"]),
                    str(row["clinical_endpoint_alignment_summary"]),
                ]
                for row in case_rows
            ]
            or [["ad_amyloid_route", "pending", "0", "pending", "pending", "pending", "pending", "pending"]],
            "Quantitative AD case validation for published anti-amyloid interventions.",
            "tab:dynamicscasevalidation",
        )
        + "\n",
    )
    write_text(
        table_dir / "clinical_direction_case_study_matrix_table.tex",
        table_tex_from_rows(
            [
                "Intervention",
                "Case Type",
                "External Rank",
                "Ranking Rank",
                "CNS Rollout Rank",
                "TRI Rank",
                "Clinical Direction",
                "Policy",
            ],
            [
                [
                    str(row["intervention_label"]),
                    str(row["case_type"]),
                    str(row["external_evidence_only_rank"]),
                    str(row["neuronsifter_ranking_rank"]),
                    str(row["neuronsifter_cns_rollout_rank"]),
                    str(row["neuronsifter_tri_gated_rank"]),
                    f"{float(row['clinical_direction_score']):.2f}",
                    str(row["endpoint_alignment_policy"]),
                ]
                for row in clinical_direction_rows
            ],
            "Retrospective clinical-direction case study matrix comparing external evidence, NeuronSifter ranking, CNS rollout, and TRI gating.",
            "tab:clinicaldirectioncasematrix",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_efficiency_table.tex",
        table_tex_from_rows(
            ["Model", "Status", "Parameters", "Checkpoint (MB)", "Latency (ms)", "Device"],
            [
                [
                    str(row["model_name"]),
                    str(row["status"]),
                    (f"{int(row['parameter_count']):,}" if row.get("parameter_count") is not None else "pending"),
                    (f"{float(row['checkpoint_size_mb']):.2f}" if row.get("checkpoint_size_mb") is not None else "pending"),
                    (f"{float(row['latency_ms']):.2f}" if row.get("latency_ms") is not None else "pending"),
                    str(row.get("device", "cpu")),
                ]
                for row in efficiency
            ],
            "CPU efficiency snapshot for runnable baselines and the AD screening system.",
            "tab:dynamicsefficiency",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_runnable_baseline_table.tex",
        table_tex_from_rows(
            ["Model", "Family", "Adapter Mode", "Comparison Status", "AUROC", "AUPRC", "Balanced Acc.", "Macro-F1", "Transition MSE"],
            [
                [
                    str(row["model_name"]),
                    str(row["family"]),
                    str(row["adapter_mode"]),
                    str(row["comparison_status"]),
                    str(row["auroc"]),
                    str(row["auprc"]),
                    str(row["balanced_accuracy"]),
                    str(row["macro_f1"]),
                    str(row["transition_mse"]),
                ]
                for row in baseline_rows
            ],
            "Runnable in-repo baselines and project-owned wrapper context used for the dynamics paper without overstating external reproducibility.",
            "tab:dynamicsrunnablebaselines",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_profile_status_table.tex",
        table_tex_from_rows(
            ["Profile", "Available Seeds", "Status"],
            [[str(row["profile_name"]), str(row["available_seed_count"]), str(row["status"])] for row in status_rows],
            "Status of formal dynamics and runnable-baseline result catalogs.",
            "tab:dynamicsprofilestatus",
        )
        + "\n",
    )
    write_text(
        table_dir / "dynamics_strict_case_table.tex",
        table_tex_from_rows(
            ["Case", "Anchor Hit@K", "Quantitative Anchor Hit", "Structural Plausibility"],
            [
                [
                    str(row["case_study_name"]),
                    str(row["anchor_hit_at_top_k"]),
                    str(row["quantitative_anchor_hit"]),
                    str(row["structural_plausibility_hit"]),
                ]
                for row in strict_rows
            ]
            or [["ad_amyloid_route", "pending", "pending", "pending"]],
            "Strict AD case-study recovery and structural plausibility summary.",
            "tab:dynamicsstrictcase",
        )
        + "\n",
    )
    write_text(
        table_dir / "active_learning_replay_table.tex",
        table_tex_from_rows(
            ["Policy", "Top-K", "Validation Yield", "Anchor Recovery", "Accepted Quality", "Coverage-Adjusted Gain"],
            [
                [
                    str(row["policy_name"]),
                    str(row["top_k"]),
                    str(row["top_k_validation_yield"]),
                    str(row["anchor_recovery_fraction"]),
                    str(row["accepted_candidate_quality"]),
                    str(row["coverage_adjusted_gain"]),
                ]
                for row in active_learning_rows
            ]
            or [["pending", "0", "pending", "pending", "pending", "pending"]],
            "Retrospective intervention-program active-learning replay summary.",
            "tab:activelearningreplay",
        )
        + "\n",
    )

    summary_payload = {
        "paper_track": str(paper_track["name"]),
        "publication_disease_programs": sorted(publication_disease_programs(config)),
        "publication_disease_program_notes": publication_disease_program_notes(config),
        "dynamics_mainline": mainline_rows,
        "dynamics_ablation": ablation_rows,
        "dynamics_task_summary": task_rows,
        "foundation_capability_summary": capability_rows,
        "quantitative_case_validation": case_rows,
        "clinical_direction_case_study_matrix": clinical_direction_rows,
        "strict_case_summary": strict_rows,
        "active_learning_replay": active_learning_rows,
        "runnable_baselines": baseline_rows,
        "headline_external_baselines": headline_external_rows,
        "efficiency": efficiency,
        "profile_status": status_rows,
        "publication_claim_metrics": publication_claim_rows,
        "publication_baseline_matrix": publication_baseline_rows,
        "publication_ablation_matrix": publication_ablation_rows,
        "publication_selective_risk": publication_selective_risk_rows_payload,
        "publication_figure_manifest": publication_figure_rows,
        "publication_g5_effective_sample_audit": publication_g5_audit,
        "publication_scaling_transfer_summary": publication_scaling_summary,
        "publication_claim_discipline_audit": publication_discipline_audit,
        "scale_registry_audit": scale_registry_payload,
        "scale_consistency_summary": scale_consistency_payload,
        "distillation_summary": distillation_payload,
        "teacher_mixture_summary": teacher_mixture_payload,
        "fsdp_smoke_summary": fsdp_smoke_payload,
        "gpu_stochastic_smoke_summary": gpu_stochastic_smoke_payload,
        "foundation_readiness_gate": foundation_readiness_payload,
    }
    write_json(report_dir / "paper_result_summary.json", summary_payload)

    pending_required_profiles = [
        str(row["profile_name"])
        for row in status_rows
        if str(row["profile_name"]) in set(required_dynamics_profiles(paper_track)) and str(row["status"]) != "available"
    ]
    if pending_required_profiles and not args.allow_pending:
        raise RuntimeError(
            "Formal dynamics result families are still pending: "
            + ", ".join(sorted(pending_required_profiles))
        )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=report_dir / "paper_result_summary.json",
    )


if __name__ == "__main__":
    main()
