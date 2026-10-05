"""Shared physical-exposure semantics for qualified C1 actions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class C1PhysicalExposure:
    dose_value: float
    duration_hours: float
    dose_unit: str
    schedule: str
    route: str

    @property
    def canonical_key(self) -> tuple[object, ...]:
        """Identity of the information visible to the C1 action encoder."""

        return (
            self.dose_value,
            self.duration_hours,
            self.dose_unit,
            self.schedule,
            self.route,
        )


def action_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def normalized_action_unit(value: object) -> str:
    return action_text(value).lower().replace("µ", "u").replace("μ", "u")


def action_duration_hours(value: object, unit: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"C1 action duration is not numeric: {value!r}") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"C1 action duration is invalid: {value!r}")
    factors = {
        "h": 1.0,
        "hr": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "d": 24.0,
        "day": 24.0,
        "days": 24.0,
        "week": 24.0 * 7.0,
        "weeks": 24.0 * 7.0,
        "month": 24.0 * 30.0,
        "months": 24.0 * 30.0,
    }
    normalized = normalized_action_unit(unit)
    if normalized not in factors:
        raise ValueError(f"C1 action duration unit is not supported: {unit!r}")
    return numeric * factors[normalized]


def action_dose_value(row: Mapping[str, object]) -> float:
    raw = row.get("physical_dose")
    if raw is None or action_text(raw) == "":
        raw = row.get("dose_value")
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"C1 action dose is not numeric: {raw!r}") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"C1 action dose is invalid: {raw!r}")
    return numeric


def qualified_physical_exposure(
    row: Mapping[str, object],
) -> C1PhysicalExposure:
    """Return the exact physical fields visible to the C1 action encoder."""

    dose_unit = normalized_action_unit(row.get("dose_unit"))
    schedule = action_text(row.get("schedule"))
    route = action_text(row.get("route")) or action_text(
        row.get("route_or_not_applicable")
    )
    if not dose_unit or not schedule or not route:
        raise ValueError("C1 qualified action has an empty exposure field")
    return C1PhysicalExposure(
        dose_value=action_dose_value(row),
        duration_hours=action_duration_hours(
            row.get("duration"), row.get("duration_unit")
        ),
        dose_unit=dose_unit,
        schedule=schedule,
        route=route,
    )
