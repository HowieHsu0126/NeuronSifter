from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_model_input_registry import fit_feature_scaler
from Libs.utils.evaluation.benchmark_datasets import train_context_standardized_split_matrices
from Libs.utils.data.raw_data_registry import load_config


def test_train_context_scaler_uses_train_rows_for_fit_and_applies_to_holdout() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    raw_matrix = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 10.0],
            [100.0, 10.0],
        ],
        dtype=np.float64,
    )
    train_mask = np.asarray([True, True, False])
    normalized, method, scaler_payload = fit_feature_scaler(raw_matrix, "expression_signal", config, train_mask)
    assert method == "zscore"
    assert np.isclose(scaler_payload["mean"][0], 2.0)
    assert np.isclose(scaler_payload["std"][0], 1.0)
    assert np.isclose(normalized[2, 0], 98.0)
    assert bool(scaler_payload["zero_variance_flag"][1])
    assert np.isclose(normalized[2, 1], 0.0)


def test_log1p_scaler_roundtrips_metabolomics_proxy_units() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    raw_matrix = np.asarray(
        [
            [0.0],
            [3.0],
            [8.0],
        ],
        dtype=np.float64,
    )
    normalized, method, scaler_payload = fit_feature_scaler(raw_matrix, "metabolite_intensity", config)
    assert method == "log1p_then_zscore"
    reconstructed = np.expm1(normalized[:, 0] * scaler_payload["std"][0] + scaler_payload["mean"][0])
    assert np.allclose(reconstructed, raw_matrix[:, 0], rtol=1.0e-6, atol=1.0e-6)


def test_benchmark_split_standardization_fits_train_context_only() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    standardized = train_context_standardized_split_matrices(
        config,
        "transcriptomics",
        {
            "train": np.asarray([[1.0], [3.0]], dtype=np.float32),
            "validation": np.asarray([[100.0]], dtype=np.float32),
            "test": np.asarray([[2.0]], dtype=np.float32),
        },
    )
    assert np.isclose(standardized["train"][0, 0], -1.0)
    assert np.isclose(standardized["train"][1, 0], 1.0)
    assert np.isclose(standardized["validation"][0, 0], 98.0)
    assert np.isclose(standardized["test"][0, 0], 0.0)
