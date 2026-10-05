"""Frozen summary-statistic conversions for public retrospective evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class SummaryArm:
    """One published arm summary; no individual-replicate claim is implied."""

    mean: float
    n: int
    sd: float
    uncertainty_source: str

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.mean)):
            raise ValueError("Arm mean must be finite")
        if int(self.n) < 2:
            raise ValueError("At least two biological replicates are required")
        if not math.isfinite(float(self.sd)) or float(self.sd) <= 0.0:
            raise ValueError("Arm SD must be finite and positive")


def resolve_sd(
    *,
    n: int,
    sd: float | None = None,
    sem: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    confidence_level: float = 0.95,
) -> tuple[float, str]:
    """Convert exactly one reported uncertainty form to SD using the frozen policy."""

    if int(n) < 2:
        raise ValueError("At least two biological replicates are required")
    candidates = [sd is not None, sem is not None, ci_low is not None or ci_high is not None]
    if sum(candidates) != 1:
        raise ValueError("Provide exactly one of SD, SEM, or a complete confidence interval")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("Confidence level must lie strictly between zero and one")
    if sd is not None:
        value = float(sd)
        source = "source_reported_sd"
    elif sem is not None:
        value = float(sem) * math.sqrt(int(n))
        source = "source_reported_sem"
    else:
        if ci_low is None or ci_high is None or not math.isfinite(float(ci_low)) or not math.isfinite(float(ci_high)):
            raise ValueError("Confidence interval must have finite lower and upper bounds")
        if float(ci_high) <= float(ci_low):
            raise ValueError("Confidence interval upper bound must exceed lower bound")
        critical = float(t.ppf((1.0 + float(confidence_level)) / 2.0, int(n) - 1))
        value = (float(ci_high) - float(ci_low)) * math.sqrt(int(n)) / (2.0 * critical)
        source = "source_reported_ci"
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Converted SD must be finite and positive")
    return value, source


def build_summary_arm(
    *,
    mean: float,
    n: int,
    sd: float | None = None,
    sem: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    confidence_level: float = 0.95,
) -> SummaryArm:
    resolved_sd, source = resolve_sd(
        n=n,
        sd=sd,
        sem=sem,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
    )
    return SummaryArm(mean=float(mean), n=int(n), sd=resolved_sd, uncertainty_source=source)


def control_referenced_difference(treatment: SummaryArm, control: SummaryArm) -> tuple[float, float]:
    """Return treatment-minus-control effect and independent-arm variance."""

    effect = float(treatment.mean) - float(control.mean)
    variance = float(treatment.sd) ** 2 / int(treatment.n) + float(control.sd) ** 2 / int(control.n)
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("Summary contrast variance must be finite and positive")
    return effect, variance


def summary_contrast_bootstrap(
    treatment: SummaryArm,
    control: SummaryArm,
    *,
    resamples: int = 2000,
    seed: int = 3407,
) -> dict[str, float | int | str]:
    """Propagate arm-level uncertainty without presenting draws as replicates."""

    if int(resamples) < 100:
        raise ValueError("Summary bootstrap requires at least 100 resamples")
    effect, variance = control_referenced_difference(treatment, control)
    rng = np.random.default_rng(int(seed))
    draws = rng.normal(effect, math.sqrt(variance), size=int(resamples))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "statistic_family": "parametric_normal_summary_bootstrap",
        "resamples": int(resamples),
        "seed": int(seed),
        "effect_estimate": float(effect),
        "variance": float(variance),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "benefit_probability_for_positive_direction": float(np.mean(draws > 0.0)),
        "individual_replicates_claimed": False,
    }


def rank_probability(effects: Sequence[float], *, tie_tolerance: float = 1e-12) -> list[float]:
    """Return deterministic rank probabilities from sampled effects."""

    values = np.asarray(list(effects), dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Effects must be a two-dimensional action-by-sample array")
    winners = np.max(values, axis=0)
    return [float(np.mean(np.abs(values[index] - winners) <= float(tie_tolerance))) for index in range(values.shape[0])]
