from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.empty((0,), dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-6:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def _diversity_bonus(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0,), dtype=np.float64)
    target_column = "target_symbol_set" if "target_symbol_set" in frame.columns else "target_program_label_set"
    group_key = (
        frame[target_column].astype(str)
        + "::"
        + frame["route"].astype(str)
        + "::"
        + frame["schedule"].astype(str)
    )
    group_frequency = group_key.map(group_key.value_counts()).astype(float).to_numpy(dtype=np.float64)
    return _normalize(1.0 / np.sqrt(np.maximum(group_frequency, 1.0)))


def build_acquisition_frame(
    regimen_frame: pd.DataFrame,
    predicted_scores: np.ndarray,
    epistemic_uncertainty: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    if regimen_frame.empty:
        return regimen_frame.copy()
    weights = config["screening_pipeline"]["screening_benchmark"]["active_learning"]["weights"]
    frame = regimen_frame.copy().reset_index(drop=True)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
    epistemic_uncertainty = np.asarray(epistemic_uncertainty, dtype=np.float64).reshape(-1)
    if predicted_scores.shape[0] != frame.shape[0]:
        raise ValueError("Predicted scores must align with regimen_frame rows.")
    if epistemic_uncertainty.shape[0] != frame.shape[0]:
        raise ValueError("Epistemic uncertainty must align with regimen_frame rows.")
    utility = _normalize(predicted_scores)
    uncertainty = _normalize(epistemic_uncertainty)
    pkpd = _normalize(pd.to_numeric(frame["pk_feasibility_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    if "validation_readiness_score" in frame.columns:
        validation_readiness_values = pd.to_numeric(frame["validation_readiness_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        validation_readiness_values = np.zeros((frame.shape[0],), dtype=np.float64)
    validation_readiness = _normalize(validation_readiness_values)
    diversity = _diversity_bonus(frame)
    safety_penalty = 1.0 - pkpd
    frame["expected_utility"] = utility
    frame["epistemic_uncertainty"] = uncertainty
    frame["diversity_bonus"] = diversity
    frame["pkpd_component"] = pkpd
    frame["validation_readiness"] = validation_readiness
    frame["safety_penalty"] = safety_penalty
    frame["acquisition_score"] = (
        float(weights["expected_utility"]) * utility
        + float(weights["epistemic_uncertainty"]) * uncertainty
        + float(weights["diversity"]) * diversity
        + float(weights["pkpd_feasibility"]) * pkpd
        + float(weights.get("validation_readiness", 0.0)) * validation_readiness
        - float(weights["safety_penalty"]) * safety_penalty
    )
    return frame


def select_acquisition_batch(
    acquisition_frame: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    if acquisition_frame.empty or batch_size <= 0:
        return acquisition_frame.iloc[0:0].copy()
    remaining = acquisition_frame.copy().reset_index(drop=True)
    selected_indices: list[int] = []
    route_usage: dict[str, int] = defaultdict(int)
    schedule_usage: dict[str, int] = defaultdict(int)
    target_usage: dict[str, int] = defaultdict(int)
    target_column = "target_symbol_set" if "target_symbol_set" in remaining.columns else "target_program_label_set"

    while remaining.shape[0] > 0 and len(selected_indices) < int(batch_size):
        route_penalty = remaining["route"].astype(str).map(lambda value: route_usage[value]).to_numpy(dtype=np.float64)
        schedule_penalty = remaining["schedule"].astype(str).map(lambda value: schedule_usage[value]).to_numpy(dtype=np.float64)
        target_penalty = remaining[target_column].astype(str).map(lambda value: target_usage[value]).to_numpy(dtype=np.float64)
        adjusted_score = remaining["acquisition_score"].to_numpy(dtype=np.float64) - 0.05 * (
            route_penalty + schedule_penalty + target_penalty
        )
        best_position = int(np.argmax(adjusted_score))
        selected_row = remaining.iloc[best_position]
        selected_indices.append(int(selected_row.name))
        route_usage[str(selected_row["route"])] += 1
        schedule_usage[str(selected_row["schedule"])] += 1
        target_usage[str(selected_row[target_column])] += 1
        remaining = remaining.drop(index=selected_row.name)
    selected = acquisition_frame.loc[selected_indices].copy()
    selected["acquisition_rank"] = np.arange(1, selected.shape[0] + 1, dtype=np.int64)
    return selected.sort_values("acquisition_rank").reset_index(drop=True)
