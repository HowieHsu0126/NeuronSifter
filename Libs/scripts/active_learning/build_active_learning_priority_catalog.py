from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path, write_tsv
from Libs.utils.evaluation.active_learning_loop import active_learning_config, active_learning_root
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.screening_core import stable_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the full active-learning source-prior catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def priority_catalog_path(config: dict, output_path: str) -> Path:
    if output_path:
        return resolve_repo_path(output_path)
    return active_learning_root(config) / str(active_learning_config(config)["relative_paths"]["priority_catalog"])


def bounded(value: float, priority_seed: dict[str, object]) -> float:
    return min(max(float(value), float(priority_seed["score_min"])), float(priority_seed["score_max"]))


def numeric_value(value: object, priority_seed: dict[str, object]) -> float:
    if value is None or str(value).strip() == "":
        return float(priority_seed["score_min"])
    return float(value)


def evidence_payload(row: dict[str, str]) -> dict[str, object]:
    payload = json.loads(str(row["evidence_summary"]))
    if not isinstance(payload, dict):
        raise TypeError(f"Intervention evidence_summary must decode to a mapping: {row['intervention_program_uid']}")
    return payload


def validation_readiness(row: dict[str, str], priority_seed: dict[str, object]) -> float:
    weights = priority_seed["validation_readiness_weights"]
    return bounded(
        sum(
            float(weights[field_name]) * numeric_value(row.get(field_name, ""), priority_seed)
            for field_name in weights
        ),
        priority_seed,
    )


def priority_rows(config: dict) -> list[dict[str, object]]:
    active_config = active_learning_config(config)
    priority_seed = active_config["priority_seed"]
    intervention_config = config["twin"]["intervention_program_registry"]
    intervention_path = (
        resolve_repo_path(str(intervention_config["local_root"]))
        / str(intervention_config["relative_paths"]["intervention_program_catalog"])
    )
    validation_config = config["validation"]["candidate_validation_registry"]
    validation_path = (
        resolve_repo_path(str(validation_config["local_root"]))
        / str(validation_config["relative_paths"]["candidate_validation_catalog"])
    )
    intervention_rows = read_tsv(intervention_path)
    validation_by_intervention = {
        str(row["intervention_program_uid"]): row
        for row in read_tsv(validation_path)
    }
    if not intervention_rows:
        raise ValueError("Active-learning priority catalog requires non-empty intervention_program_catalog.")
    weights = active_config["weights"]
    uncertainty_weights = priority_seed["uncertainty_component_weights"]
    target_key = str(priority_seed["target_evidence_key"])
    perturbation_key = str(priority_seed["perturbation_reversal_key"])
    selected_weight_sum = (
        float(weights["expected_utility"])
        + float(weights["pkpd_feasibility"])
        + float(weights["validation_readiness"])
    )
    if selected_weight_sum <= 0.0:
        raise ValueError("Active-learning priority weights must have a positive selected_weight_sum.")
    rows: list[dict[str, object]] = []
    output_path = active_learning_root(config) / str(active_config["relative_paths"]["priority_catalog"])
    for intervention_row in intervention_rows:
        payload = evidence_payload(intervention_row)
        validation_row = validation_by_intervention.get(str(intervention_row["intervention_program_uid"]), {})
        target_evidence = numeric_value(payload.get(target_key), priority_seed)
        perturbation_reversal = numeric_value(payload.get(perturbation_key), priority_seed)
        pkpd_feasibility = numeric_value(intervention_row["pkpd_feasibility_score"], priority_seed)
        readiness = validation_readiness(validation_row, priority_seed)
        admet_readiness = numeric_value(validation_row.get("admet_readiness_score", ""), priority_seed)
        predicted_score = bounded(
            float(weights["expected_utility"]) * target_evidence
            + float(weights["pkpd_feasibility"]) * pkpd_feasibility
            + float(weights["validation_readiness"]) * readiness
            + float(weights["epistemic_uncertainty"]) * perturbation_reversal
            - float(weights["safety_penalty"]) * (float(priority_seed["score_max"]) - admet_readiness),
            priority_seed,
        )
        tri_score = bounded(
            (
                float(weights["expected_utility"]) * target_evidence
                + float(weights["pkpd_feasibility"]) * pkpd_feasibility
                + float(weights["validation_readiness"]) * readiness
            )
            / selected_weight_sum,
            priority_seed,
        )
        predictive_entropy = bounded(float(priority_seed["score_max"]) - tri_score, priority_seed)
        mutual_information = bounded(float(priority_seed["score_max"]) - readiness, priority_seed)
        epistemic_std = bounded(float(priority_seed["score_max"]) - max(target_evidence, perturbation_reversal), priority_seed)
        active_uncertainty = bounded(
            float(uncertainty_weights["predictive_entropy_mean"]) * predictive_entropy
            + float(uncertainty_weights["mutual_information_mean"]) * mutual_information
            + float(uncertainty_weights["epistemic_std_mean"]) * epistemic_std,
            priority_seed,
        )
        rows.append(
            {
                "priority_uid": stable_uid("active_learning_priority", intervention_row["regimen_uid"]),
                "regimen_uid": intervention_row["regimen_uid"],
                "predicted_score_mean": round(predicted_score, 8),
                "tri_score_mean": round(tri_score, 8),
                "pkpd_consistency_mean": round(bounded(pkpd_feasibility, priority_seed), 8),
                "predictive_entropy_mean": round(predictive_entropy, 8),
                "mutual_information_mean": round(mutual_information, 8),
                "epistemic_std_mean": round(epistemic_std, 8),
                "active_learning_uncertainty": round(active_uncertainty, 8),
                "source_holdout_role": str(priority_seed["source_holdout_role"]),
                "selection_reason": str(intervention_row["selection_reason"]),
                "evidence_summary": str(intervention_row["evidence_summary"]),
                "artifact_path": output_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "note": str(intervention_row["intervention_program_uid"]),
            }
        )
    minimum_task_capacity = int(active_config["full_tasks_per_round"])
    simulator_count = len(active_config["simulator_portfolio"])
    if len(rows) * simulator_count < minimum_task_capacity:
        raise ValueError(
            "Active-learning source-prior catalog has insufficient request capacity: "
            f"{len(rows)} candidates * {simulator_count} simulators < {minimum_task_capacity} tasks."
        )
    rows.sort(key=lambda row: (float(row["predicted_score_mean"]), str(row["regimen_uid"])), reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    output_path = priority_catalog_path(config, args.output_path)
    rows = priority_rows(config)
    write_tsv(output_path, config["schemas"]["active_learning_priority_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        priority_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
