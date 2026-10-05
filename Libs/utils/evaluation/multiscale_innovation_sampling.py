"""Randomized acquisition of balanced-Haar population mean hinge risk.

The supplied envelopes bound expected orthonormal detail energy and the RMS
errors of each paired child mean. They are assumptions, not fitted calibration.
The confidence statement additionally requires independent, identically
distributed acquisition draws (location, source field, paired measurement).
Dependence between the two errors within a draw is allowed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InnovationSamplingDesign:
    horizon: int
    energy_constant: float
    smoothness: float
    mean_error_rms: float
    error_decay: float
    signal_envelopes: tuple[float, ...]
    measurement_envelopes: tuple[float, ...]
    level_probabilities: tuple[float, ...]
    normalizer: float
    measurement_bias_bound: float

    @property
    def level_count(self) -> int:
        return self.horizon.bit_length() - 1

    @property
    def query_free(self) -> bool:
        return self.normalizer == 0.0

    @property
    def second_moment_bound(self) -> float:
        return self.normalizer * self.normalizer


@dataclass(frozen=True)
class InnovationLocations:
    levels: np.ndarray
    node_indices: np.ndarray
    probabilities: np.ndarray


def _finite_scalar(value, name, *, nonnegative=False, positive=False):
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar.") from error
    if (
        not math.isfinite(value)
        or (nonnegative and value < 0)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{name} must be finite and satisfy its sign constraint.")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def compile_innovation_design(
    horizon: int,
    energy_constant: float,
    smoothness: float,
    *,
    mean_error_rms: float = 0.0,
    error_decay: float = 1.0,
) -> InnovationSamplingDesign:
    """Compile O(log horizon) probabilities without enumerating tree nodes.

    For level ell, require E||d_ell||_2^2 <= C*N*2**(-2*s*ell),
    where d contains orthonormal Haar details. Each child-mean error requires
    RMS <= e0*2**(-r*ell), uniformly over nodes at that level. Both decays are
    positive. Signed int64 node indices limit the horizon to 2**63.
    """
    horizon = _integer(horizon, "horizon")
    if horizon < 2 or horizon & (horizon - 1) or horizon > 2**63:
        raise ValueError("horizon must be a power of two in [2, 2**63].")
    constant = _finite_scalar(energy_constant, "energy_constant", nonnegative=True)
    smoothness = _finite_scalar(smoothness, "smoothness", positive=True)
    error = _finite_scalar(mean_error_rms, "mean_error_rms", nonnegative=True)
    decay = _finite_scalar(error_decay, "error_decay", positive=True)
    levels = np.arange(horizon.bit_length() - 1, dtype=np.int64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        signal = math.sqrt(constant) / 2 * np.exp2(-smoothness * levels)
        measurement = error / math.sqrt(2) * np.exp2(-decay * levels)
        envelope = signal + measurement
        normalizer = float(envelope.sum())
        bias = float(measurement.sum())
    if (
        (constant > 0 and np.any(signal == 0))
        or (error > 0 and np.any(measurement == 0))
        or not np.isfinite(envelope).all()
        or not math.isfinite(normalizer)
        or normalizer > math.sqrt(np.finfo(float).max)
    ):
        raise ValueError("Envelope dynamic range exceeds finite float64 arithmetic.")
    probabilities = envelope / normalizer if normalizer else np.zeros_like(envelope)
    if normalizer and np.any(np.ldexp(probabilities, -levels) == 0):
        raise ValueError("Node probabilities underflow float64 arithmetic.")
    return InnovationSamplingDesign(
        horizon,
        constant,
        smoothness,
        error,
        decay,
        tuple(map(float, signal)),
        tuple(map(float, measurement)),
        tuple(map(float, probabilities)),
        normalizer,
        bias,
    )


def sample_innovation_locations(
    design: InnovationSamplingDesign,
    count: int,
    *,
    rng: np.random.Generator,
) -> InnovationLocations:
    """Draw a level and then a uniform node, independently for each query."""
    if not isinstance(design, InnovationSamplingDesign):
        raise ValueError("Supply a compiled innovation design.")
    count = _integer(count, "count")
    if count < 0 or (design.query_free and count != 0):
        raise ValueError(
            "Use a nonnegative count, and zero queries for a zero-envelope design."
        )
    if not isinstance(rng, np.random.Generator):
        raise ValueError("Supply an explicit numpy Generator.")
    if count == 0:
        return InnovationLocations(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
        )
    levels = rng.choice(design.level_count, size=count, p=design.level_probabilities)
    widths = np.left_shift(np.int64(1), levels)
    nodes = rng.integers(0, widths, size=count, dtype=np.int64)
    probabilities = np.ldexp(np.asarray(design.level_probabilities)[levels], -levels)
    return InnovationLocations(levels, nodes, probabilities)


def _validated_locations(design, locations):
    if not isinstance(design, InnovationSamplingDesign) or not isinstance(
        locations, InnovationLocations
    ):
        raise ValueError("Supply a compiled design and InnovationLocations.")
    levels = np.asarray(locations.levels)
    nodes = np.asarray(locations.node_indices)
    probabilities = np.asarray(locations.probabilities, dtype=float)
    if (
        levels.ndim != 1
        or nodes.shape != levels.shape
        or probabilities.shape != levels.shape
    ):
        raise ValueError(
            "Location levels, node indices and probabilities must be aligned vectors."
        )
    if not np.issubdtype(levels.dtype, np.integer) or not np.issubdtype(
        nodes.dtype, np.integer
    ):
        raise ValueError("Location levels and node indices must be integers.")
    if (
        np.any(levels < 0)
        or np.any(levels >= design.level_count)
        or np.any(nodes < 0)
        or np.any(nodes >= np.left_shift(np.int64(1), levels))
    ):
        raise ValueError("Location level or node index lies outside the balanced tree.")
    expected = np.ldexp(np.asarray(design.level_probabilities)[levels], -levels)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0)
        or not np.allclose(probabilities, expected, rtol=1e-12, atol=0)
    ):
        raise ValueError(
            "Location probabilities must equal the compiled per-node design probabilities."
        )
    if design.query_free and len(levels):
        raise ValueError("A zero-envelope design requires zero queries.")
    return levels, probabilities


def innovation_importance_values(
    design: InnovationSamplingDesign,
    locations: InnovationLocations,
    left_means,
    right_means,
    *,
    threshold: float,
) -> np.ndarray:
    """Return observed g_v/p_v without estimating any noise or energy envelope.

    The hinge Jensen gap is min(|left-b|, |right-b|) when the centered child
    means have opposite signs, and zero otherwise. This equivalent expression
    avoids cancellation between large positive hinge terms.
    """
    levels, _ = _validated_locations(design, locations)
    threshold = _finite_scalar(threshold, "threshold")
    left = np.asarray(left_means, dtype=float)
    right = np.asarray(right_means, dtype=float)
    if (
        left.shape != levels.shape
        or right.shape != levels.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError(
            "Paired child means must be finite vectors aligned with locations."
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        left, right = left - threshold, right - threshold
        opposite = ((left > 0) & (right < 0)) | ((left < 0) & (right > 0))
        gap = np.where(opposite, np.minimum(np.abs(left), np.abs(right)), 0.0)
        # g = gap/2**(ell+1), p = q_ell/2**ell.
        values = gap / (2 * np.asarray(design.level_probabilities)[levels])
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Paired means or importance values exceed finite float64 arithmetic."
        )
    return values


def estimate_innovation_risk(
    design: InnovationSamplingDesign,
    locations: InnovationLocations,
    left_means,
    right_means,
    *,
    threshold: float,
    root_risk: float,
    alpha: float,
    root_error_bound: float = 0.0,
) -> dict:
    """Median-of-means estimate with a conditional supplied-envelope guarantee.

    Use the smallest odd k >= 8*log(1/alpha) and m=floor(n/k). The first k*m
    draws form k consecutive groups; remaining draws are explicitly unused.
    The radius is B + 2*K/sqrt(m) plus a supplied root approximation allowance
    exactly once. Independence and the supplied envelopes are never inferred
    from the observed arrays. For K=0 only an empty, query-free input is valid.
    """
    root = _finite_scalar(root_risk, "root_risk", nonnegative=True)
    root_error = _finite_scalar(root_error_bound, "root_error_bound", nonnegative=True)
    alpha = _finite_scalar(alpha, "alpha", positive=True)
    if alpha >= 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    values = innovation_importance_values(
        design, locations, left_means, right_means, threshold=threshold
    )
    count = len(values)
    if design.query_free:
        groups = size = used = 0
        estimate, sampling_error, confidence = root, 0.0, 1.0
    else:
        groups = math.ceil(-8 * math.log(alpha))
        if groups % 2 == 0:
            groups += 1
        size = count // groups
        if size < 1:
            raise ValueError(
                f"At least {groups} IID acquisition draws are required for alpha={alpha}."
            )
        used = groups * size
        with np.errstate(over="ignore", invalid="ignore"):
            group_means = (values[:used].reshape(groups, size) / size).sum(axis=1)
        if not np.isfinite(group_means).all():
            raise ValueError("Group means exceed finite float64 arithmetic.")
        estimate = root + float(np.median(group_means))
        sampling_error = 2 * design.normalizer / math.sqrt(size)
        confidence = 1 - alpha
    radius = design.measurement_bias_bound + sampling_error + root_error
    if (
        not math.isfinite(estimate)
        or not math.isfinite(radius)
        or not math.isfinite(estimate + radius)
    ):
        raise ValueError(
            "Risk estimate or confidence endpoints exceed finite float64 arithmetic."
        )
    return {
        "estimate": estimate,
        "lower": max(0.0, estimate - radius),
        "upper": estimate + radius,
        "error_radius": radius,
        "measurement_bias_bound": design.measurement_bias_bound,
        "root_error_bound": root_error,
        "sampling_error_bound": sampling_error,
        "normalizer": design.normalizer,
        "second_moment_bound": design.second_moment_bound,
        "confidence_level": confidence,
        "sample_count": count,
        "used_sample_count": used,
        "group_count": groups,
        "group_size": size,
        "query_free": design.query_free,
        "scope": "balanced_haar_population_mean_hinge_risk",
        "calibration_established": False,
        "iid_validated": False,
        "assumptions": (
            "Expected orthonormal Haar level energies satisfy the supplied envelope.",
            "Each paired child-mean RMS error satisfies the supplied level envelope uniformly over nodes.",
            "Acquisition draws are IID under the compiled location design and the target population law.",
            "Locations are independent of fresh source draws from the target population law.",
            "The supplied root risk is within root_error_bound of the target root expectation.",
        ),
    }
