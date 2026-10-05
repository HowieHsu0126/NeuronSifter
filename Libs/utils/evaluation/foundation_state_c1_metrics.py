"""Unit-aware C1 metrics and paired bootstrap utilities.

All inferential summaries operate on the preregistered biological unit.  Raw
observation weighting is retained as an explicitly auxiliary view and never
drives the headline estimate or bootstrap.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


FAMILY_BOOTSTRAP_UNITS = {
    "cns_disease_aging_state_transcriptomics": "subject_specimen",
    "cns_cell_type_region_atlas_transcriptomics": "donor",
    "compound_dose_time_perturbation_transcriptomics": "independent_condition_or_action",
}


def _validated_rows(
    records: Iterable[Mapping[str, object]],
    *,
    family_key: str,
    unit_key: str,
    unit_type_key: str,
    value_key: str,
) -> list[Mapping[str, object]]:
    rows = list(records)
    if not rows:
        raise ValueError("C1 metric requires at least one biological-unit result")
    for row in rows:
        family = str(row[family_key])
        if family not in FAMILY_BOOTSTRAP_UNITS:
            raise ValueError(f"Unknown C1 family for unit-aware inference: {family}")
        expected_type = FAMILY_BOOTSTRAP_UNITS[family]
        actual_type = str(row.get(unit_type_key, ""))
        if actual_type != expected_type:
            raise ValueError(
                f"C1 bootstrap unit mismatch for {family}: expected {expected_type}, got {actual_type or 'missing'}"
            )
        unit = str(row.get(unit_key, ""))
        if not unit:
            raise ValueError("C1 biological-unit identifier is required")
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("C1 metric values must be finite")
    return rows


def _unit_values(
    records: Iterable[Mapping[str, object]],
    *,
    family_key: str = "family_uid",
    unit_key: str = "biological_unit_uid",
    unit_type_key: str = "bootstrap_unit",
    value_key: str = "value",
) -> tuple[list[Mapping[str, object]], dict[str, dict[str, list[float]]]]:
    rows = _validated_rows(
        records,
        family_key=family_key,
        unit_key=unit_key,
        unit_type_key=unit_type_key,
        value_key=value_key,
    )
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        family = str(row[family_key])
        unit = str(row[unit_key])
        grouped.setdefault(family, {}).setdefault(unit, []).append(float(row[value_key]))
    return rows, grouped


def aggregate_unit_metric(
    records: Iterable[Mapping[str, object]],
    *,
    ess_by_family: Mapping[str, float],
    family_key: str = "family_uid",
    unit_key: str = "biological_unit_uid",
    unit_type_key: str = "bootstrap_unit",
    value_key: str = "value",
    observation_weight_key: str = "observation_count",
) -> dict[str, object]:
    rows, grouped = _unit_values(
        records,
        family_key=family_key,
        unit_key=unit_key,
        unit_type_key=unit_type_key,
        value_key=value_key,
    )
    per_family = {
        family: float(np.mean([np.mean(values) for values in units.values()]))
        for family, units in grouped.items()
        if units
    }
    if not per_family:
        raise ValueError("C1 metric requires at least one biological-unit result")
    macro = float(np.mean(list(per_family.values())))
    total_ess = sum(float(ess_by_family.get(family, 0.0)) for family in per_family)
    if total_ess <= 0.0:
        raise ValueError("C1 ESS-weighted metric requires positive family ESS")
    ess_weighted = float(
        sum(per_family[family] * float(ess_by_family.get(family, 0.0)) for family in per_family)
        / total_ess
    )
    raw_values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    raw_weights = np.asarray(
        [float(row.get(observation_weight_key, 1.0)) for row in rows], dtype=np.float64
    )
    if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights <= 0.0):
        raise ValueError("C1 observation weights must be finite and positive")
    raw_auxiliary = float(np.average(raw_values, weights=raw_weights))
    return {
        "per_family": per_family,
        "macro_family_average": macro,
        "ess_weighted": ess_weighted,
        "raw_observation_weighted_auxiliary": raw_auxiliary,
        "family_count": len(per_family),
        "biological_unit_count": sum(len(units) for units in grouped.values()),
        "raw_observation_count": len(rows),
    }


def paired_bootstrap_ci(
    paired_records: Sequence[Mapping[str, object]],
    *,
    value_key: str = "delta",
    family_key: str = "family_uid",
    unit_key: str = "biological_unit_uid",
    unit_type_key: str = "bootstrap_unit",
    resamples: int = 2000,
    seed: int = 3407,
) -> dict[str, object]:
    if resamples != 2000 or seed != 3407:
        raise ValueError("C1 paired bootstrap is frozen at 2000 resamples and seed 3407")
    _, grouped = _unit_values(
        paired_records,
        family_key=family_key,
        unit_key=unit_key,
        unit_type_key=unit_type_key,
        value_key=value_key,
    )
    unit_values = {
        family: np.asarray([np.mean(values) for values in units.values()], dtype=np.float64)
        for family, units in grouped.items()
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        family_means = []
        for values in unit_values.values():
            sample = rng.choice(values, size=len(values), replace=True)
            family_means.append(float(np.mean(sample)))
        draws[index] = float(np.mean(family_means))
    point = float(np.mean([float(np.mean(values)) for values in unit_values.values()]))
    return {
        "estimate": point,
        "estimate_view": "macro_family_average",
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "seed": seed,
        "bootstrap_unit": "family_then_biological_unit",
        "family_count": len(unit_values),
        "unit_count": int(sum(len(values) for values in unit_values.values())),
    }


def chance_normalized_balanced_accuracy(accuracy: float, class_count: int) -> float:
    if class_count < 2:
        raise ValueError("Shortcut diagnostics require at least two classes")
    if not math.isfinite(float(accuracy)):
        raise ValueError("Shortcut accuracy must be finite")
    chance = 1.0 / float(class_count)
    return float((float(accuracy) - chance) / (1.0 - chance))


def deterministic_label_permutation(labels: Sequence[object], *, seed: int = 3407) -> list[object]:
    if seed != 3407:
        raise ValueError("C1 diagnostic permutation seed is frozen at 3407")
    values = list(labels)
    order = sorted(
        range(len(values)),
        key=lambda index: hashlib.sha256(f"{seed}|{index}".encode("utf-8")).hexdigest(),
    )
    return [values[index] for index in order]
