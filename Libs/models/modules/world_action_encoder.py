"""Collision-resistant deterministic encoding of canonical controlled WorldActions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Mapping

import torch

from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS


WORLD_ACTION_SCHEMA_VERSION = "neuronsifter-operational-world-action-encoding-v2"
CATEGORICAL_FIELDS = (
    "intervention_type",
    "source",
    "target_set",
    "agent_set",
    "dose_unit",
    "schedule",
    "route",
    "context",
    "stimulation_or_environment",
    "modality",
    "combination",
    "support_domain",
)
NUMERIC_FIELDS = (
    "dose",
    "dose_or_intensity",
    "duration",
    "controlled_duration",
    "response_delay",
)
DIGEST_WIDTH = 4
WORLD_ACTION_FIELDS = tuple(
    [f"{name}_digest_{index}" for name in CATEGORICAL_FIELDS for index in range(DIGEST_WIDTH)]
    + list(NUMERIC_FIELDS)
    + [f"{name}_valid" for name in (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS)]
    + [f"family_{name}" for name in FAMILY_UIDS]
)


def world_action_schema_hash() -> str:
    payload = {"schema_version": WORLD_ACTION_SCHEMA_VERSION, "fields": WORLD_ACTION_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _digest_vector(value: object) -> list[float]:
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()
    return [int.from_bytes(digest[index * 8:(index + 1) * 8], "big") / float(2**63) - 1.0 for index in range(DIGEST_WIDTH)]


def _canonical_set(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("Set-valued WorldAction identity fields must be sequences.")
    return sorted(value, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))


NATIVE_ACTION_IDENTITY_SCHEMA = "canonical_agent_set_sha256_float4.v1"


def canonical_agent_identity(action: Mapping[str, object]) -> list[float]:
    """Encode only canonical agent membership, independently of program or RNG IDs.

    Reuse the controlled encoder's fixed digest without altering its complete
    WorldAction semantics. No fitted vocabulary or row-dependent identity exists.
    """
    if not isinstance(action, Mapping):
        raise ValueError("Native drug_id requires canonical program.action.agent_set.")
    agents = action.get("agent_set")
    if (not isinstance(agents, (list, tuple)) or not agents
            or any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in agents)):
        raise ValueError("Native drug_id requires a nonempty canonical program.action.agent_set of identifiers.")
    return _digest_vector(_canonical_set(sorted(set(agents))))


NATIVE_TARGET_IDENTITY_SCHEMA = "canonical_target_set_sha256_float4.v1"


def canonical_target_identity(action: Mapping[str, object]) -> list[float]:
    """Encode only explicit canonical target membership; admission owns control semantics."""
    if not isinstance(action, Mapping):
        raise ValueError("Native target_id requires canonical program.action.target_set.")
    targets = action.get("target_set")
    if (not isinstance(targets, (list, tuple))
            or any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in targets)):
        raise ValueError("Native target_id requires an explicit canonical program.action.target_set of identifiers.")
    return _digest_vector(_canonical_set(sorted(set(targets))))


def native_action_identity_spec(action_policy: str):
    """Select the native identity contract without changing legacy agent encodings."""
    if action_policy == "target_id":
        return canonical_target_identity, NATIVE_TARGET_IDENTITY_SCHEMA
    if action_policy in {"occupancy_program", "drug_id", "scalar_exposure"}:
        return canonical_agent_identity, NATIVE_ACTION_IDENTITY_SCHEMA
    raise ValueError(f"Native action policy {action_policy!r} is unimplemented.")


def _scaled_numeric(value: object) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    numeric = float(value)
    return math.copysign(math.log1p(abs(numeric)) / 10.0, numeric), 1.0


class WorldActionEncoder:
    """Fixed and versioned: no vocabulary is fit from development or calibration data."""

    output_dim = len(WORLD_ACTION_FIELDS)

    @staticmethod
    def no_action_binding(binding: Mapping[str, object]) -> dict[str, object]:
        value = deepcopy(dict(binding))
        canonical = value["canonical_world_action"]
        extension = value["controlled_extension"]
        canonical.update({
            "action_uid": "canonical_no_action", "intervention_type": "no_action", "target_set": [],
            "agent_set": [], "dose": None, "dose_unit": None, "route": None,
            "schedule": {"kind": "none", "program_ref": "no_action"},
            "duration": {"value": 0.0, "unit": extension["duration"]["unit"]},
            "stimulation_or_environment": {"modality": "none"},
        })
        extension.update({
            "modality": "none",
            "dose_or_intensity": {"value": 0.0, "unit": extension["dose_or_intensity"]["unit"]},
            "duration": {"value": 0.0, "unit": extension["duration"]["unit"]},
            "combination": [],
        })
        extension["action_semantics"].update({
            "target_set": [], "agent_set": [], "schedule": {"kind": "none", "program_ref": "no_action"}, "route": None,
        })
        value["response_delay_steps"] = 0
        value["mapping_status"] = "canonical_no_action"
        return value

    def encode_no_action(self, binding: Mapping[str, object], family_uid: str) -> torch.Tensor:
        return self.encode(self.no_action_binding(binding), family_uid)

    def encode(self, binding: Mapping[str, object], family_uid: str) -> torch.Tensor:
        if family_uid not in FAMILY_UIDS:
            raise ValueError(f"Unsupported family: {family_uid}")
        canonical = dict(binding["canonical_world_action"])
        extension = dict(binding["controlled_extension"])
        semantics = dict(extension["action_semantics"])
        context = dict(extension["biological_context"])
        canonical_duration = dict(canonical["duration"])
        controlled_duration = dict(extension["duration"])
        dose_or_intensity = dict(extension["dose_or_intensity"])
        categorical = {
            "intervention_type": canonical.get("intervention_type"), "source": canonical.get("action_provenance_ref"),
            "target_set": _canonical_set(canonical.get("target_set")),
            "agent_set": _canonical_set(canonical.get("agent_set")),
            "dose_unit": canonical.get("dose_unit"), "schedule": canonical.get("schedule"),
            "route": canonical.get("route"),
            "context": {"state_scope": canonical.get("state_scope"), "biological_context": context},
            "stimulation_or_environment": canonical.get("stimulation_or_environment"),
            "modality": extension.get("modality"),
            "combination": _canonical_set(extension.get("combination")),
            "support_domain": semantics.get("support_domain_ref"),
        }
        numeric_raw = {
            "dose": canonical.get("dose"),
            "dose_or_intensity": dose_or_intensity.get("value"),
            "duration": canonical_duration.get("value"),
            "controlled_duration": controlled_duration.get("value"),
            "response_delay": binding.get("response_delay_steps"),
        }
        values = [component for name in CATEGORICAL_FIELDS for component in _digest_vector(categorical[name])]
        numeric = [_scaled_numeric(numeric_raw[name]) for name in NUMERIC_FIELDS]
        values.extend(value for value, _ in numeric)
        values.extend(float(categorical[name] is not None) for name in CATEGORICAL_FIELDS)
        values.extend(valid for _, valid in numeric)
        values.extend(float(family_uid == expected) for expected in FAMILY_UIDS)
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.shape != (self.output_dim,) or not torch.isfinite(tensor).all():
            raise ValueError("WorldAction encoding is malformed or non-finite.")
        return tensor
