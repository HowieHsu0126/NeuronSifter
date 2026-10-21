from __future__ import annotations

from typing import Any


def _compare(metric_value: float, threshold_value: float, comparator: str) -> bool:
    if comparator == ">=":
        return metric_value >= threshold_value
    if comparator == "<=":
        return metric_value <= threshold_value
    if comparator == "==":
        return metric_value == threshold_value
    raise ValueError(f"Unsupported comparator: {comparator}")


def capability_row(
    benchmark_name: str,
    disease_program: str,
    disease_family: str,
    fold_name: str,
    modality: str,
    capability_family: str,
    capability_name: str,
    metric_name: str,
    metric_value: float,
    threshold_value: float,
    comparator: str,
    artifact_path: str,
    note: str,
) -> dict[str, object]:
    return {
        "benchmark_name": benchmark_name,
        "disease_program": disease_program,
        "disease_family": disease_family,
        "fold_name": fold_name,
        "modality": modality,
        "capability_family": capability_family,
        "capability_name": capability_name,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "threshold_value": float(threshold_value),
        "comparator": comparator,
        "pass_flag": "yes" if _compare(float(metric_value), float(threshold_value), comparator) else "no",
        "artifact_path": artifact_path,
        "note": note,
    }


def capability_summary_row(
    benchmark_name: str,
    disease_program: str,
    disease_family: str,
    fold_name: str,
    modality: str,
    capability_family: str,
    capability_name: str,
    component_rows: list[dict[str, object]],
    artifact_path: str,
) -> dict[str, object]:
    pass_fraction = 0.0 if not component_rows else sum(row["pass_flag"] == "yes" for row in component_rows) / len(component_rows)
    return {
        "benchmark_name": benchmark_name,
        "disease_program": disease_program,
        "disease_family": disease_family,
        "fold_name": fold_name,
        "modality": modality,
        "capability_family": capability_family,
        "capability_name": capability_name,
        "metric_name": "contract_pass_fraction",
        "metric_value": float(pass_fraction),
        "threshold_value": 1.0,
        "comparator": ">=",
        "pass_flag": "yes" if pass_fraction >= 1.0 else "no",
        "artifact_path": artifact_path,
        "note": "All component checks must pass for capability acceptance.",
    }


def capability_catalog_rows(
    config: dict[str, Any],
    benchmark_name: str,
    disease_program: str,
    disease_family: str,
    fold_name: str,
    modality: str,
    artifact_path: str,
    metrics: dict[str, float],
    registries: dict[str, float],
) -> list[dict[str, object]]:
    acceptance_config = config["screening_pipeline"]["screening_benchmark"]["capability_acceptance"]
    rows: list[dict[str, object]] = []

    structured_rows = [
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "structured_world_state",
            "factor_count",
            registries["factor_count"],
            float(len(acceptance_config["structured_world_state"]["required_factor_ids"])),
            ">=",
            artifact_path,
            "Five-factor world-state contract coverage.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "structured_world_state",
            "crossmodal_anchor_count",
            registries["crossmodal_anchor_count"],
            float(acceptance_config["structured_world_state"]["minimum_crossmodal_anchor_count"]),
            ">=",
            artifact_path,
            "Cross-modal anchor support for hidden-state alignment.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "structured_world_state",
            "missing_modality_count",
            registries["missing_modality_count"],
            float(acceptance_config["structured_world_state"]["minimum_missing_modality_count"]),
            ">=",
            artifact_path,
            "Missing-modality inference support.",
        ),
    ]
    rows.extend(structured_rows)
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "hard_gate", "structured_world_state", structured_rows, artifact_path))

    causal_rows = [
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "causal_dynamics",
            "concentration_nrmse",
            metrics["concentration_nrmse"],
            float(acceptance_config["causal_dynamics"]["maximum_concentration_nrmse"]),
            "<=",
            artifact_path,
            "Concentration trajectory agreement.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "causal_dynamics",
            "flux_nrmse",
            metrics["flux_nrmse"],
            float(acceptance_config["causal_dynamics"]["maximum_flux_nrmse"]),
            "<=",
            artifact_path,
            "Flux consistency with teacher dynamics.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "causal_dynamics",
            "route_ordering_hit",
            metrics["route_ordering_hit"],
            float(acceptance_config["causal_dynamics"]["minimum_route_ordering_hit"]),
            ">=",
            artifact_path,
            "Action sensitivity under route changes.",
        ),
    ]
    rows.extend(causal_rows)
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "hard_gate", "causal_dynamics", causal_rows, artifact_path))

    rollout_rows = [
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "long_horizon_action_rollout",
            "concentration_nrmse",
            metrics["concentration_nrmse"],
            float(acceptance_config["long_horizon_action_rollout"]["maximum_concentration_nrmse"]),
            "<=",
            artifact_path,
            "Long-horizon concentration rollout error.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "long_horizon_action_rollout",
            "electrophysiology_nrmse",
            metrics["electrophysiology_nrmse"],
            float(acceptance_config["long_horizon_action_rollout"]["maximum_electrophysiology_nrmse"]),
            "<=",
            artifact_path,
            "Long-horizon electrophysiology rollout error.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "long_horizon_action_rollout",
            "pkpd_consistency_mean",
            metrics["pkpd_consistency_mean"],
            float(acceptance_config["long_horizon_action_rollout"]["minimum_pkpd_consistency_mean"]),
            ">=",
            artifact_path,
            "PK/PD consistency under rollout.",
        ),
    ]
    rows.extend(rollout_rows)
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "hard_gate", "long_horizon_action_rollout", rollout_rows, artifact_path))

    counterfactual_rows = [
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "counterfactual_planning",
            "counterfactual_pair_count",
            registries["counterfactual_pair_count"],
            float(acceptance_config["counterfactual_planning"]["minimum_counterfactual_pair_count"]),
            ">=",
            artifact_path,
            "Counterfactual benchmark availability.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "hard_gate",
            "counterfactual_planning",
            "regimen_ndcg_at_10",
            metrics["regimen_ndcg_at_10"],
            float(acceptance_config["counterfactual_planning"]["minimum_regimen_ndcg_at_10"]),
            ">=",
            artifact_path,
            "Planning proxy quality from intervention ranking.",
        ),
    ]
    rows.extend(counterfactual_rows)
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "hard_gate", "counterfactual_planning", counterfactual_rows, artifact_path))

    engineering_rows = [
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "engineering",
            "multimodal_alignment",
            "crossmodal_anchor_count",
            registries["crossmodal_anchor_count"],
            float(acceptance_config["multimodal_alignment"]["minimum_crossmodal_anchor_count"]),
            ">=",
            artifact_path,
            "Cross-modal anchor availability.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "engineering",
            "interactive_rollout_latency",
            "interactive_latency_ms",
            metrics["interactive_latency_ms"],
            float(acceptance_config["multimodal_alignment"]["maximum_interactive_latency_ms"]),
            "<=",
            artifact_path,
            "Interactive analyst latency proxy.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "engineering",
            "online_continual_calibration",
            "feedback_event_count",
            registries["feedback_event_count"],
            float(acceptance_config["online_continual_calibration"]["minimum_feedback_event_count"]),
            ">=",
            artifact_path,
            "Feedback event stream availability.",
        ),
    ]
    rows.extend(engineering_rows)
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "engineering", "multimodal_alignment", engineering_rows[:2], artifact_path))
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "engineering", "online_continual_calibration", engineering_rows[2:], artifact_path))

    high_order_rows = [
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "high_order",
            "causal_robustness",
            "latent_distance_ood_auroc",
            metrics["ood_auroc"],
            float(acceptance_config["high_order"]["minimum_ood_auroc"]),
            ">=",
            artifact_path,
            "OOD discrimination for causal robustness proxy.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "high_order",
            "hierarchical_generalization",
            "target_ndcg_at_10",
            metrics["target_ndcg_at_10"],
            float(acceptance_config["high_order"]["minimum_target_ndcg_at_10"]),
            ">=",
            artifact_path,
            "Cross-family target ranking quality.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "high_order",
            "hierarchical_generalization",
            "regimen_ndcg_at_10",
            metrics["regimen_ndcg_at_10"],
            float(acceptance_config["high_order"]["minimum_regimen_ndcg_at_10"]),
            ">=",
            artifact_path,
            "Cross-family regimen ranking quality.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "high_order",
            "interacting_subsystems",
            "simulation_teacher_count",
            registries["simulation_teacher_count"],
            float(acceptance_config["high_order"]["minimum_simulation_teacher_count"]),
            ">=",
            artifact_path,
            "Availability of multiscale interacting-subsystem teachers.",
        ),
        capability_row(
            benchmark_name,
            disease_program,
            disease_family,
            fold_name,
            modality,
            "high_order",
            "explanation_and_confidence",
            "acceptance_benefit",
            metrics["acceptance_benefit"],
            float(acceptance_config["high_order"]["minimum_acceptance_benefit"]),
            ">=",
            artifact_path,
            "Reliability-aware acceptance benefit proxy.",
        ),
    ]
    rows.extend(high_order_rows)
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "high_order", "causal_robustness", high_order_rows[:1], artifact_path))
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "high_order", "hierarchical_generalization", high_order_rows[1:3], artifact_path))
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "high_order", "interacting_subsystems", high_order_rows[3:4], artifact_path))
    rows.append(capability_summary_row(benchmark_name, disease_program, disease_family, fold_name, modality, "high_order", "explanation_and_confidence", high_order_rows[4:], artifact_path))

    return rows
