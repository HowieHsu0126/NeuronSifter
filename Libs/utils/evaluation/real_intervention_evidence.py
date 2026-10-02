"""Decision-sensitivity and split-integrity utilities for real interventions."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("RMS requires a non-empty finite array")
    return float(np.sqrt(np.mean(np.square(array))))


def transformed_expression(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"Expression matrix must be non-empty and rank two, got {array.shape}")
    if not np.isfinite(array).all() or float(array.min()) < 0.0:
        raise ValueError("Expression matrix must contain finite non-negative values")
    return np.log2(array + 1.0)


def control_variability(control: np.ndarray) -> float:
    array = np.asarray(control, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("Control variability requires at least two biological replicates")
    return rms(array - array.mean(axis=0, keepdims=True))


def single_action_effect(action: np.ndarray, control: np.ndarray) -> dict[str, float | str]:
    action_array = np.asarray(action, dtype=np.float64)
    control_array = np.asarray(control, dtype=np.float64)
    if action_array.ndim != 2 or control_array.ndim != 2 or action_array.shape[1] != control_array.shape[1]:
        raise ValueError("Action and control matrices must have matching feature axes")
    if action_array.shape[0] < 2 or control_array.shape[0] < 2:
        raise ValueError("Effect diagnostics require at least two biological replicates per arm")
    effect = action_array.mean(axis=0) - control_array.mean(axis=0)
    noise = control_variability(control_array)
    magnitude = rms(effect)
    return {
        "effect_magnitude": magnitude,
        "effect_direction": "positive" if float(effect.mean()) > 0.0 else "negative" if float(effect.mean()) < 0.0 else "zero",
        "within_control_variability": noise,
        "effect_to_control_noise": magnitude / noise if noise > 0.0 else float("inf"),
    }


def _bootstrap_effect_magnitudes(
    actions: Mapping[str, np.ndarray],
    control: np.ndarray,
    resamples: int,
    seed: int,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    control_array = np.asarray(control, dtype=np.float64)
    action_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in actions.items()}
    feature_count = control_array.shape[1]
    if any(value.ndim != 2 or value.shape[1] != feature_count for value in action_arrays.values()):
        raise ValueError("All action matrices must share the control feature axis")
    output = {key: np.empty(resamples, dtype=np.float64) for key in action_arrays}
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        size = stop - start
        control_indices = rng.integers(0, control_array.shape[0], size=(size, control_array.shape[0]))
        control_means = control_array[control_indices].mean(axis=1)
        for key, action in action_arrays.items():
            action_indices = rng.integers(0, action.shape[0], size=(size, action.shape[0]))
            action_means = action[action_indices].mean(axis=1)
            output[key][start:stop] = np.sqrt(np.mean(np.square(action_means - control_means), axis=1))
    return output


def candidate_set_diagnostics(
    actions: Mapping[str, np.ndarray],
    control: np.ndarray,
    *,
    resamples: int,
    seed: int,
    effect_noise_floor: float,
    contrast_noise_floor: float,
) -> dict[str, object]:
    if len(actions) < 2:
        raise ValueError("Candidate-set diagnostics require at least two actions")
    control_array = np.asarray(control, dtype=np.float64)
    noise = control_variability(control_array)
    effect_vectors = {
        key: np.asarray(value, dtype=np.float64).mean(axis=0) - control_array.mean(axis=0)
        for key, value in actions.items()
    }
    effect_magnitudes = {key: rms(value) for key, value in effect_vectors.items()}
    effect_directions = {
        key: "positive" if float(value.mean()) > 0.0 else "negative" if float(value.mean()) < 0.0 else "zero"
        for key, value in effect_vectors.items()
    }
    pairwise_contrasts = {
        f"{left}__vs__{right}": rms(effect_vectors[left] - effect_vectors[right])
        for left, right in combinations(sorted(effect_vectors), 2)
    }
    ordered = sorted(effect_magnitudes, key=lambda key: (-effect_magnitudes[key], key))
    top_action, runner_up_action = ordered[:2]
    observed_margin = effect_magnitudes[top_action] - effect_magnitudes[runner_up_action]
    bootstrap = _bootstrap_effect_magnitudes(actions, control_array, resamples, seed)
    bootstrap_margin = bootstrap[top_action] - bootstrap[runner_up_action]
    lower, upper = np.quantile(bootstrap_margin, [0.025, 0.975]).tolist()
    max_effect_ratio = max(effect_magnitudes.values()) / noise if noise > 0.0 else float("inf")
    max_contrast = max(pairwise_contrasts.values())
    max_contrast_ratio = max_contrast / noise if noise > 0.0 else float("inf")
    rank_separable = bool(lower > 0.0 or upper < 0.0)
    if max_effect_ratio < effect_noise_floor:
        classification = "near_null"
    elif max_contrast_ratio >= contrast_noise_floor and rank_separable:
        classification = "decision_sensitive"
    else:
        classification = "scientifically_informative_but_not_rank_sensitive"
    return {
        "effect_magnitudes": effect_magnitudes,
        "effect_directions": effect_directions,
        "within_control_variability": noise,
        "pairwise_action_contrasts": pairwise_contrasts,
        "maximum_effect_to_control_noise": max_effect_ratio,
        "maximum_action_contrast": max_contrast,
        "maximum_action_contrast_to_control_noise": max_contrast_ratio,
        "top_action_uid": top_action,
        "runner_up_action_uid": runner_up_action,
        "decision_margin": observed_margin,
        "decision_margin_ci95_lower": float(lower),
        "decision_margin_ci95_upper": float(upper),
        "rank_separable": rank_separable,
        "outcome_rank_classification": classification,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def overlap_values(rows_by_split: Mapping[str, Sequence[Mapping[str, object]]], field: str) -> dict[str, list[str]]:
    normalized: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        values = {
            str(row.get(field, "")).strip()
            for row in rows
            if str(row.get(field, "")).strip()
            and not str(row.get(field, "")).startswith(("explicit_unavailable", "not_applicable"))
        }
        normalized[split] = values
    overlaps: dict[str, list[str]] = {}
    for left, right in combinations(sorted(normalized), 2):
        shared = sorted(normalized[left] & normalized[right])
        if shared:
            overlaps[f"{left}__vs__{right}"] = shared
    return overlaps
