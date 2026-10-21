from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def test_temperature_search_and_calibration_metrics_are_finite() -> None:
    validation_logits = np.asarray([-2.0, -1.0, 0.5, 1.5], dtype=np.float64)
    validation_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    best_temperature, validation_probabilities = fit_temperature_from_grid(
        validation_logits,
        validation_labels,
        [0.5, 1.0, 1.5, 2.0],
    )
    assert best_temperature in {0.5, 1.0, 1.5, 2.0}
    assert np.isfinite(binary_nll(validation_probabilities, validation_labels))
    assert np.isfinite(binary_brier(validation_probabilities, validation_labels))
    assert np.isfinite(expected_calibration_error(validation_probabilities, validation_labels, 5))
    assert np.isfinite(adaptive_calibration_error(validation_probabilities, validation_labels, 5))
    calibrated_test = sigmoid_with_temperature(np.asarray([0.25, -0.25], dtype=np.float64), best_temperature)
    assert calibrated_test.shape == (2,)


def test_ood_and_risk_coverage_metrics_behave_reasonably() -> None:
    reference_latents = np.asarray([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.05]], dtype=np.float64)
    id_latents = np.asarray([[0.05, 0.0], [0.0, 0.05]], dtype=np.float64)
    ood_latents = np.asarray([[2.0, 2.0], [2.2, 1.8]], dtype=np.float64)
    assert ood_auroc(reference_latents, id_latents, ood_latents) > 0.9

    errors = np.asarray([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
    confidence = np.asarray([0.95, 0.8, 0.6, 0.2], dtype=np.float64)
    curve = risk_coverage_curve(errors, confidence, 5)
    assert curve["coverage"].shape[0] >= 2
    assert np.isfinite(curve["auc"])
    accept_mask = confidence >= 0.6
    assert coverage_gap(float(accept_mask.mean()), 0.75) >= 0.0
    assert np.isfinite(acceptance_benefit(errors, accept_mask))


def test_deep_ensemble_summary_reports_zero_epistemic_for_single_member() -> None:
    single_member = np.asarray([[0.2, 0.8, 0.6]], dtype=np.float64)
    summary = ensemble_uncertainty_summary(single_member)
    assert summary["predictive_mean"].shape == (3,)
    assert np.allclose(summary["epistemic_std"], 0.0)
    assert np.all(summary["mutual_information"] >= 0.0)


def test_deep_ensemble_summary_reports_finite_uncertainty_for_multiple_members() -> None:
    member_predictions = np.asarray(
        [
            [0.2, 0.8, 0.6],
            [0.4, 0.7, 0.2],
            [0.3, 0.9, 0.4],
        ],
        dtype=np.float64,
    )
    summary = ensemble_uncertainty_summary(member_predictions)
    assert np.all(np.isfinite(summary["predictive_mean"]))
    assert np.all(np.isfinite(summary["epistemic_std"]))
    assert np.all(np.isfinite(summary["aleatoric_variance"]))
    assert np.all(np.isfinite(summary["predictive_entropy"]))
    assert np.all(np.isfinite(summary["mutual_information"]))
    assert np.all(summary["epistemic_std"] > 0.0)
