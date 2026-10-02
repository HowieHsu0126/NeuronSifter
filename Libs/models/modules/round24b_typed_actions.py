"""Round-24B typed current-clamp actions.

Current-clamp actions use the canonical ``WorldAction`` boundary, but their
numeric field is an electrical amplitude, not a pharmacological dose.  The
explicit protocol payload in ``schedule`` preserves delay and duration in the
content-derived action identity.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from Libs.models.modules.decision_consistent_runtime import WorldAction


@dataclass(frozen=True)
class CurrentClampAction:
    action_uid: str
    pair_uid: str
    role: str
    amplitude_nA: float
    duration_ms: float
    delay_ms: float
    source_model_id: str
    split_role: str
    schema_version: str = "neuronsifter-round24b-current-clamp-action-v1"

    def __post_init__(self) -> None:
        if not self.action_uid or not self.pair_uid or not self.source_model_id:
            raise ValueError("CurrentClampAction identities are required")
        if self.role not in {"intervention", "matched_control"}:
            raise ValueError("CurrentClampAction.role must be intervention or matched_control")
        if self.split_role not in {"train", "validation", "development_diagnostic"}:
            raise ValueError("CurrentClampAction.split_role is outside development scope")
        for name, value in (("amplitude_nA", self.amplitude_nA), ("duration_ms", self.duration_ms), ("delay_ms", self.delay_ms)):
            if not math.isfinite(float(value)):
                raise ValueError(f"CurrentClampAction.{name} must be finite")
        if self.duration_ms <= 0.0 or self.delay_ms < 0.0:
            raise ValueError("CurrentClampAction timing is invalid")
        if self.role == "matched_control" and abs(self.amplitude_nA) > 1e-12:
            raise ValueError("Matched current-clamp controls must have zero amplitude")

    @classmethod
    def from_pool_record(cls, record: Mapping[str, Any], source_model_id: str) -> "CurrentClampAction":
        action = record["action"]
        return cls(
            action_uid=str(record["action_uid"]),
            pair_uid=str(record["pair_uid"]),
            role=str(action["action_role"]),
            amplitude_nA=float(action["dose_value"]),
            duration_ms=float(action["duration_ms"]),
            delay_ms=float(action["delay_ms"]),
            source_model_id=source_model_id,
            split_role=str(record["split_role"]),
        )

    def schedule_payload(self) -> dict[str, Any]:
        return {
            "protocol": "single_pulse_after_resting_state",
            "delay_ms": float(self.delay_ms),
            "duration_ms": float(self.duration_ms),
            "domain": "current_clamp",
        }

    def to_world_action(self) -> WorldAction:
        return WorldAction(
            target_set=("soma_membrane",),
            drug_set=("electrical_current" if self.role == "intervention" else "no_external_agent",),
            dose={"value": float(self.amplitude_nA), "unit": "nA", "domain": "current_clamp_amplitude"},
            schedule=json.dumps(self.schedule_payload(), sort_keys=True, separators=(",", ":")),
            route="intracellular_soma_current_clamp",
        )

    def to_record(self) -> dict[str, Any]:
        world = self.to_world_action()
        return {
            "schema_version": self.schema_version,
            "action_type": "world",
            "action_uid": self.action_uid,
            "world_action_uid": world.action_uid,
            "pair_uid": self.pair_uid,
            "role": self.role,
            "source_model_id": self.source_model_id,
            "split_role": self.split_role,
            "amplitude_nA": float(self.amplitude_nA),
            "duration_ms": float(self.duration_ms),
            "delay_ms": float(self.delay_ms),
            "target_set": list(world.target_set),
            "agent_set": list(world.drug_set),
            "route": world.route,
            "schedule": self.schedule_payload(),
        }


def pair_from_pool_records(records: list[Mapping[str, Any]], source_model_id: str, split_role: str = "validation") -> list[tuple[CurrentClampAction, CurrentClampAction]]:
    interventions: dict[str, CurrentClampAction] = {}
    controls: dict[str, CurrentClampAction] = {}
    for record in records:
        if str(record["split_role"]) != split_role:
            continue
        action = CurrentClampAction.from_pool_record(record, source_model_id)
        target = interventions if action.role == "intervention" else controls
        if action.pair_uid in target:
            raise ValueError(f"Duplicate current-clamp role for pair {action.pair_uid}")
        target[action.pair_uid] = action
    pairs = []
    for pair_uid in sorted(interventions):
        if pair_uid not in controls:
            raise ValueError(f"Missing matched current-clamp control for {pair_uid}")
        pairs.append((interventions[pair_uid], controls[pair_uid]))
    if not pairs:
        raise ValueError("No typed current-clamp validation pairs are available")
    return pairs


__all__ = ["CurrentClampAction", "pair_from_pool_records"]
