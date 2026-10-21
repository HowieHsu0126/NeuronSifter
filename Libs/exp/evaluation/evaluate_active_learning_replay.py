from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path, write_tsv
from Libs.utils.evaluation.regimen_acquisition import build_acquisition_frame, select_acquisition_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrospective intervention-program active-learning replay policies.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def result_row(
    config: dict,
    policy_name: str,
    top_k: int,
    metric_name: str,
    metric_value: float,
    output_path: Path,
) -> dict[str, object]:
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    task_catalog = benchmark.get("paper_tasks", {})
    task_name = "long_horizon_twin_rollout_and_active_learning"
    task_family = str(task_catalog.get(task_name, {}).get("task_family", "rollout_and_active_learning"))
    return {
        "benchmark_name": benchmark["benchmark_name"],
        "disease_program": benchmark["active_disease_program"],
        "baseline_family": "active_learning_replay",
        "split_name": "intervention_pool",
        "evaluation_scope": "active_learning_replay",
        "task_name": task_name,
        "task_family": task_family,
        "model_name": "ad_screening_system",
        "modality": "intervention_program",
        "sample_count": 0,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "artifact_path": str(output_path.resolve().relative_to(REPO_ROOT)),
        "case_study_name": "",
        "scale_classification": "",
        "is_case_study_anchor": "no",
        "note": f"{policy_name}:top_k={top_k}",
    }


def readiness_score(frame: pd.DataFrame) -> np.ndarray:
    affinity = pd.to_numeric(frame["affinity_evidence_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    admet = pd.to_numeric(frame["admet_readiness_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    pbpk = pd.to_numeric(frame["pbpk_readiness_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(0.30 * affinity + 0.35 * admet + 0.35 * pbpk, 0.0, 1.0)


def evidence_utility(frame: pd.DataFrame) -> np.ndarray:
    values: list[float] = []
    for row in frame.to_dict(orient="records"):
        evidence_payload = json.loads(str(row["evidence_summary"]))
        target_evidence = float(evidence_payload["aggregate_target_evidence_score"])
        reversal = float(evidence_payload["clue_reversal_proxy_score"])
        pkpd = float(row["pkpd_feasibility_score"])
        values.append(0.45 * target_evidence + 0.15 * reversal + 0.40 * pkpd)
    return np.asarray(values, dtype=np.float64)


def selected_frame_for_policy(
    frame: pd.DataFrame,
    policy_name: str,
    top_k: int,
    random_seed: int,
    config: dict,
) -> pd.DataFrame:
    if policy_name == "random":
        order = np.random.default_rng(random_seed + int(top_k)).permutation(frame.shape[0])[:top_k]
        selected = frame.iloc[order].copy()
        selected["acquisition_rank"] = np.arange(1, selected.shape[0] + 1)
        return selected
    if policy_name == "expected_utility_only":
        return frame.assign(policy_score=frame["expected_utility_score"]).sort_values("policy_score", ascending=False).head(top_k)
    if policy_name == "epistemic_uncertainty_only":
        return frame.assign(policy_score=frame["epistemic_uncertainty_score"]).sort_values("policy_score", ascending=False).head(top_k)
    if policy_name == "full_acquisition":
        acquisition_input = frame.copy()
        acquisition_input["target_symbol_set"] = acquisition_input["target_program_label_set"].astype(str)
        acquisition_input["pk_feasibility_score"] = acquisition_input["pkpd_feasibility_score"]
        acquisition_frame = build_acquisition_frame(
            acquisition_input,
            frame["expected_utility_score"].to_numpy(dtype=np.float64),
            frame["epistemic_uncertainty_score"].to_numpy(dtype=np.float64),
            config,
        )
        return select_acquisition_batch(acquisition_frame, top_k)
    raise ValueError(f"Unsupported active-learning replay policy: {policy_name}")


def replay_metric_rows(config: dict, output_path: Path) -> list[dict[str, object]]:
    intervention_root = resolve_repo_path(config["twin"]["intervention_program_registry"]["local_root"])
    validation_root = resolve_repo_path(config["validation"]["candidate_validation_registry"]["local_root"])
    intervention_rows = read_tsv(intervention_root / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"])
    validation_rows = read_tsv(validation_root / config["validation"]["candidate_validation_registry"]["relative_paths"]["candidate_validation_catalog"])
    intervention_frame = pd.DataFrame(intervention_rows)
    validation_frame = pd.DataFrame(validation_rows)
    frame = intervention_frame.merge(
        validation_frame[
            [
                "intervention_program_uid",
                "validation_stage",
                "affinity_evidence_score",
                "admet_readiness_score",
                "pbpk_readiness_score",
            ]
        ],
        on="intervention_program_uid",
        how="left",
    )
    frame["validation_readiness_score"] = readiness_score(frame)
    frame["expected_utility_score"] = evidence_utility(frame)
    frame["epistemic_uncertainty_score"] = (
        frame["expected_utility_score"].to_numpy(dtype=np.float64)
        * (1.0 - frame["validation_readiness_score"].to_numpy(dtype=np.float64))
    )
    random_seed = int(config["screening_pipeline"]["screening_benchmark"]["active_learning_replay"]["random_seed"])
    policies = [str(value) for value in config["screening_pipeline"]["screening_benchmark"]["active_learning_replay"]["policies"]]
    top_k_values = [int(value) for value in config["screening_pipeline"]["screening_benchmark"]["active_learning_replay"]["top_k_values"]]
    rows: list[dict[str, object]] = []
    validated_stages = {
        "knowledge_profiled",
        "admet_pbpk_profiled",
        "structure_profiled",
        "docked",
        "rescored",
    }
    for top_k in top_k_values:
        capped_top_k = min(top_k, frame.shape[0])
        for policy_name in policies:
            selected = selected_frame_for_policy(frame, policy_name, capped_top_k, random_seed, config)
            selected_readiness = pd.to_numeric(selected["validation_readiness_score"], errors="coerce").fillna(0.0)
            validated_mask = selected["validation_stage"].astype(str).isin(validated_stages)
            validation_coverage = validated_mask.mean()
            anchor_mask = selected["selection_reason"].astype(str).str.contains("case_study_seed", regex=False)
            metric_payload = {
                "top_k_validation_yield": float(selected_readiness.mean()) if selected.shape[0] else 0.0,
                "anchor_recovery_fraction": float(anchor_mask.mean()) if selected.shape[0] else 0.0,
                "accepted_candidate_quality": float(selected_readiness[validated_mask].mean()) if validation_coverage > 0 else 0.0,
                "coverage_adjusted_gain": float(selected_readiness.mean() * validation_coverage) if selected.shape[0] else 0.0,
            }
            rows.extend(
                result_row(config, policy_name, capped_top_k, metric_name, metric_value, output_path)
                for metric_name, metric_value in metric_payload.items()
            )
    for row in rows:
        row["sample_count"] = int(frame.shape[0])
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    default_output_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / config["screening_pipeline"]["screening_benchmark"]["benchmark_name"]
        / "active_learning_replay"
        / "result_catalog.tsv"
    )
    output_path = Path(args.output_path).resolve() if args.output_path else default_output_path
    rows = replay_metric_rows(config, output_path)
    write_tsv(output_path, config["schemas"]["screening_result_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        result_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
