from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


LOG2 = float(np.log(2.0))


def clipped_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def binary_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    safe_probabilities = clipped_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.float64)
    return float(
        -np.mean(
            labels * np.log(safe_probabilities)
            + (1.0 - labels) * np.log(1.0 - safe_probabilities)
        )
    )


def binary_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))


def sigmoid_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    scaled_logits = logits / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def fit_temperature_from_grid(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    temperature_grid: list[float],
) -> tuple[float, np.ndarray]:
    if validation_logits.size == 0 or validation_labels.size == 0:
        default_probabilities = sigmoid_with_temperature(validation_logits, 1.0)
        return 1.0, default_probabilities
    candidate_temperatures = [float(value) for value in temperature_grid] or [1.0]
    best_temperature = candidate_temperatures[0]
    best_probabilities = sigmoid_with_temperature(validation_logits, best_temperature)
    best_nll = binary_nll(best_probabilities, validation_labels)
    for candidate_temperature in candidate_temperatures[1:]:
        candidate_probabilities = sigmoid_with_temperature(validation_logits, candidate_temperature)
        candidate_nll = binary_nll(candidate_probabilities, validation_labels)
        if candidate_nll < best_nll:
            best_temperature = candidate_temperature
            best_probabilities = candidate_probabilities
            best_nll = candidate_nll
    return best_temperature, best_probabilities


def _bin_edges_from_quantiles(probabilities: np.ndarray, bin_count: int) -> np.ndarray:
    if probabilities.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(int(bin_count), 1) + 1)
    edges = np.quantile(probabilities, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    return np.unique(edges)


def calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bin_count: int,
    adaptive: bool = False,
) -> float:
    probabilities = clipped_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.size == 0:
        return 0.0
    if adaptive:
        bin_edges = _bin_edges_from_quantiles(probabilities, bin_count)
    else:
        bin_edges = np.linspace(0.0, 1.0, max(int(bin_count), 1) + 1)
    if bin_edges.shape[0] <= 1:
        return 0.0
    total = 0.0
    sample_count = float(probabilities.shape[0])
    for left_edge, right_edge in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if right_edge >= 1.0:
            mask = (probabilities >= left_edge) & (probabilities <= right_edge)
        else:
            mask = (probabilities >= left_edge) & (probabilities < right_edge)
        if not np.any(mask):
            continue
        mean_confidence = float(probabilities[mask].mean())
        mean_accuracy = float(labels[mask].mean())
        total += (float(mask.sum()) / sample_count) * abs(mean_confidence - mean_accuracy)
    return float(total)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bin_count: int) -> float:
    return calibration_error(probabilities, labels, bin_count=bin_count, adaptive=False)


def adaptive_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bin_count: int) -> float:
    return calibration_error(probabilities, labels, bin_count=bin_count, adaptive=True)


def distance_ood_scores(
    reference_latents: np.ndarray,
    query_latents: np.ndarray,
) -> np.ndarray:
    reference_latents = np.asarray(reference_latents, dtype=np.float64)
    query_latents = np.asarray(query_latents, dtype=np.float64)
    if query_latents.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if reference_latents.shape[0] == 0:
        return np.ones((query_latents.shape[0],), dtype=np.float64)
    centroid = reference_latents.mean(axis=0, keepdims=True)
    reference_distances = np.linalg.norm(reference_latents - centroid, axis=1)
    radius = max(float(np.percentile(reference_distances, 90)), 1e-6)
    query_distances = np.linalg.norm(query_latents - centroid, axis=1)
    return (query_distances / radius).astype(np.float64)


def ood_auroc(
    reference_latents: np.ndarray,
    id_latents: np.ndarray,
    ood_latents: np.ndarray,
) -> float:
    id_scores = distance_ood_scores(reference_latents, id_latents)
    ood_scores = distance_ood_scores(reference_latents, ood_latents)
    if id_scores.size == 0 or ood_scores.size == 0:
        return 0.0
    labels = np.concatenate(
        [
            np.zeros((id_scores.shape[0],), dtype=np.int64),
            np.ones((ood_scores.shape[0],), dtype=np.int64),
        ]
    )
    scores = np.concatenate([id_scores, ood_scores], axis=0)
    if np.unique(labels).shape[0] < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def risk_coverage_curve(
    errors: np.ndarray,
    confidence_scores: np.ndarray,
    grid_size: int,
) -> dict[str, np.ndarray | float]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    confidence_scores = np.asarray(confidence_scores, dtype=np.float64).reshape(-1)
    if errors.size == 0 or confidence_scores.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "coverage": empty,
            "risk": empty,
            "auc": 0.0,
        }
    order = np.argsort(-confidence_scores, kind="mergesort")
    sorted_errors = errors[order]
    sorted_confidence = confidence_scores[order]
    grid_points = np.linspace(1, sorted_errors.shape[0], max(int(grid_size), 2), dtype=np.int64)
    grid_points = np.unique(np.clip(grid_points, 1, sorted_errors.shape[0]))
    coverage = grid_points.astype(np.float64) / float(sorted_errors.shape[0])
    risk = np.asarray([sorted_errors[:count].mean() for count in grid_points], dtype=np.float64)
    auc = float(np.trapezoid(risk, coverage))
    return {
        "coverage": coverage,
        "risk": risk,
        "auc": auc,
        "max_confidence": float(sorted_confidence[0]),
    }


def coverage_gap(coverage: float, target_coverage: float) -> float:
    return float(abs(float(coverage) - float(target_coverage)))


def acceptance_benefit(errors: np.ndarray, accept_mask: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=np.float64)
    accept_mask = np.asarray(accept_mask, dtype=bool)
    if errors.size == 0 or not np.any(accept_mask):
        return 0.0
    return float(errors.mean() - errors[accept_mask].mean())


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    safe_probabilities = clipped_probabilities(probabilities)
    return -(
        safe_probabilities * np.log(safe_probabilities)
        + (1.0 - safe_probabilities) * np.log(1.0 - safe_probabilities)
    )


def ensemble_prediction_summary(prediction_matrix: np.ndarray) -> dict[str, np.ndarray]:
    prediction_matrix = np.asarray(prediction_matrix, dtype=np.float64)
    if prediction_matrix.ndim != 2:
        raise ValueError("Ensemble prediction summary expects [ensemble, candidate] predictions.")
    mean_prediction = prediction_matrix.mean(axis=0)
    std_prediction = prediction_matrix.std(axis=0)
    return {
        "mean": mean_prediction.astype(np.float64),
        "std": std_prediction.astype(np.float64),
    }


def ensemble_uncertainty_summary(prediction_matrix: np.ndarray) -> dict[str, np.ndarray]:
    member_probabilities = clipped_probabilities(prediction_matrix)
    if member_probabilities.ndim != 2:
        raise ValueError("Ensemble uncertainty summary expects [ensemble, candidate] probabilities.")
    predictive_mean = member_probabilities.mean(axis=0)
    epistemic_std = member_probabilities.std(axis=0)
    aleatoric_variance = np.mean(
        member_probabilities * (1.0 - member_probabilities),
        axis=0,
    )
    predictive_entropy = binary_entropy(predictive_mean)
    member_entropy = binary_entropy(member_probabilities)
    expected_entropy = member_entropy.mean(axis=0)
    mutual_information = np.clip(predictive_entropy - expected_entropy, a_min=0.0, a_max=None)
    confidence = np.clip(1.0 - mutual_information / LOG2, a_min=0.0, a_max=1.0)
    return {
        "predictive_mean": predictive_mean.astype(np.float64),
        "epistemic_std": epistemic_std.astype(np.float64),
        "aleatoric_variance": aleatoric_variance.astype(np.float64),
        "predictive_entropy": predictive_entropy.astype(np.float64),
        "expected_entropy": expected_entropy.astype(np.float64),
        "mutual_information": mutual_information.astype(np.float64),
        "confidence": confidence.astype(np.float64),
    }
