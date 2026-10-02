from __future__ import annotations

import numpy as np


def normalization_method(
    matrix_semantics: str,
    raw_matrix: np.ndarray,
    config: dict,
    fit_sample_mask: np.ndarray | None = None,
) -> str:
    semantics_config = config["model_input_registry"]["normalization_by_semantics"][matrix_semantics]
    transform = semantics_config["transform"]
    if transform == "zscore":
        return str(semantics_config["normalization_method"])
    if transform == "auto_log1p_then_zscore":
        fit_matrix = raw_matrix if fit_sample_mask is None else raw_matrix[fit_sample_mask]
        finite_mask = np.isfinite(fit_matrix)
        if np.any(finite_mask) and np.min(fit_matrix[finite_mask]) >= 0:
            return str(semantics_config["nonnegative_method"])
        return str(semantics_config["signed_method"])
    raise ValueError(f"Unsupported normalization transform: {transform}")


def zscore_matrix_with_fit(values: np.ndarray, fit_sample_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if fit_sample_mask is None:
        fit_sample_mask = np.ones(values.shape[0], dtype=bool)
    if fit_sample_mask.shape != (values.shape[0],):
        raise ValueError(f"fit_sample_mask has invalid shape: {fit_sample_mask.shape}")
    finite_mask = np.isfinite(values)
    fit_finite_mask = finite_mask & fit_sample_mask[:, None]
    counts = fit_finite_mask.sum(axis=0).astype(np.int64)
    sums = np.where(fit_finite_mask, values, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(values.shape[1], dtype=np.float64), where=counts > 0)
    fit_centered = np.where(fit_finite_mask, values - means, 0.0)
    variances = np.divide((fit_centered**2).sum(axis=0), counts, out=np.zeros(values.shape[1], dtype=np.float64), where=counts > 0)
    stds = np.sqrt(variances)

    normalized = np.full(values.shape, np.nan, dtype=np.float64)
    nonzero_std_mask = (counts > 0) & (stds > 0)
    zero_std_mask = (counts > 0) & (stds == 0)

    if np.any(nonzero_std_mask):
        normalized[:, nonzero_std_mask] = (values[:, nonzero_std_mask] - means[nonzero_std_mask]) / stds[nonzero_std_mask]
    if np.any(zero_std_mask):
        normalized[:, zero_std_mask] = np.where(finite_mask[:, zero_std_mask], 0.0, np.nan)
    normalized[~finite_mask] = np.nan
    return normalized, {
        "fit_sample_count": np.asarray([int(fit_sample_mask.sum())] * values.shape[1], dtype=np.int64),
        "finite_count": counts,
        "mean": means,
        "std": stds,
        "zero_variance_flag": zero_std_mask,
    }


def fit_feature_scaler(
    raw_matrix: np.ndarray,
    matrix_semantics: str,
    config: dict,
    fit_sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str, dict[str, np.ndarray]]:
    selected_method = normalization_method(matrix_semantics, raw_matrix, config, fit_sample_mask)
    transformed = raw_matrix.copy()
    finite_mask = np.isfinite(transformed)
    if selected_method == "log1p_then_zscore":
        transformed[finite_mask] = np.log1p(transformed[finite_mask])
    if selected_method in {"zscore", "log1p_then_zscore"}:
        normalized, scaler_payload = zscore_matrix_with_fit(transformed, fit_sample_mask)
        return normalized, selected_method, scaler_payload
    raise ValueError(f"Unsupported normalization method: {selected_method}")


def normalize_matrix(raw_matrix: np.ndarray, matrix_semantics: str, config: dict) -> tuple[np.ndarray, str]:
    normalized, selected_method, _ = fit_feature_scaler(raw_matrix, matrix_semantics, config)
    return normalized, selected_method
