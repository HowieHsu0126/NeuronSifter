"""Numerical typed stimulation on registered native observable clocks."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from Libs.utils.data.action_observation_contracts import compile_nonpharmacological_action
from Libs.utils.data.native_observable_profile import validate_profile_grid, quantity_incidence

STIMULATION_SCHEMA = "native_numerical_stimulation_program.v1"
FORCING_ARRAY_NAMES = (
    "electrical_current_uA", "electrical_applicability", "optical_power_mW", "optical_applicability",
)


def compile_stimulation_record(action: dict, grid: dict, binding: dict) -> dict:
    field, blockers = compile_nonpharmacological_action(action)
    if blockers or field is None:
        raise ValueError("Typed stimulation action failed canonical admission: " + str(blockers))
    kind = action["intervention_type"]
    if kind not in {"electrical_stimulation", "optical_stimulation"}:
        raise ValueError("Numerical native stimulation requires electrical or optical action semantics.")
    validate_profile_grid(grid, grid["quantity_profile"])
    required = {"source_uid", "posterior_uid", "sample_uid", "support_uid", "particle_uid", "particle_basis", "regimen_uid"}
    if not required <= set(binding) or any(not binding[k] for k in required):
        raise ValueError("Stimulation requires complete source/particle/program binding.")
    waveform = field["magnitude_or_program"]
    unit = "uA" if kind == "electrical_stimulation" else "mW"
    query = grid["query_time"]
    if waveform.get("time_unit") != query["unit"] or waveform.get("value_unit") != unit or waveform.get("interpolation") != "left_hold":
        raise ValueError("Stimulation units/clock/interpolation do not match the typed family program.")
    knots = np.asarray(waveform.get("time_values"), dtype=np.float64)
    values = np.asarray(waveform.get("values"), dtype=np.float64)
    if (knots.ndim != 1 or not len(knots) or knots.shape != values.shape
            or not np.isfinite(knots).all() or not np.isfinite(values).all()
            or knots[0] != 0 or not (np.diff(knots) > 0).all()
            or knots[-1] > query["values"][-1] or (kind == "optical_stimulation" and (values < 0).any())):
        raise ValueError("Stimulation requires a finite, causal, ordered numerical waveform.")
    site = waveform.get("incidence", {})
    registered = [quantity_incidence(grid["quantity_profile"], q, binding["source_uid"]) for q in grid["quantity_profile"]["quantities"]
                  if q.get("family_uid") == grid["family_uid"]]
    if site not in registered or site["anchor_region_id"] not in action["state_scope"]["regions"]:
        raise ValueError("Stimulation incidence is outside the registered physical action site.")
    shape = (len(query["values"]), len(grid["region_id"]), len(grid["compartment_id"]), 1)
    arrays = {name: np.zeros(shape, dtype=np.float64) for name in (*FORCING_ARRAY_NAMES, "occupancy", "support")}
    ri = grid["region_id"].index(site["anchor_region_id"])
    ci = grid["compartment_id"].index(site["anchor_compartment_id"])
    selected = np.searchsorted(knots, np.asarray(query["values"]), side="right") - 1
    amplitude, applicability = FORCING_ARRAY_NAMES[:2] if kind == "electrical_stimulation" else FORCING_ARRAY_NAMES[2:]
    arrays[amplitude][:, ri, ci, 0] = values[selected]
    arrays[applicability][:, ri, ci, 0] = 1.
    arrays["physical_time"] = np.broadcast_to(np.asarray(grid["time_hours"])[:, None, None, None], shape).copy()
    identity = {"schema_version": STIMULATION_SCHEMA, "action_field": field,
                "grid": grid, "binding": binding}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    compiled = {"schema_version": STIMULATION_SCHEMA, "program_uid": "stimulation:" + digest,
                "action_uid": action["action_uid"], "time_grid": grid["time_hours"],
                "status": "compiled", "action_field": field,
                "quantity_profile_sha256": grid["quantity_profile_sha256"],
                "arrays": {key: value.tolist() for key, value in arrays.items()}}
    return {"representation": "explicit_stimulation_grid", **copy.deepcopy(binding),
            "action_uid": action["action_uid"], "action": copy.deepcopy(action),
            "program_uid": compiled["program_uid"], "compiled_program": compiled,
            "stimulation_grid": copy.deepcopy(grid), "stimulation_binding": copy.deepcopy(binding)}


def validate_stimulation_record(record: dict) -> dict:
    expected = compile_stimulation_record(record["action"], record["stimulation_grid"], record["stimulation_binding"])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Numerical stimulation program/binding changed after compilation.")
    timing = record.get("source_action_timing")
    if timing is not None:
        wave = record["action"]["stimulation_or_environment"]["magnitude_or_program"]
        times, levels = np.asarray(wave["time_values"]), np.asarray(wave["values"])
        active = np.flatnonzero(levels[:-1])
        if (timing.get("schema_version") != "native_source_action_timing.v1" or not len(active)
                or timing["actual_onset"] != float(times[active[0]])
                or timing["actual_stop"] != float(times[active[-1]+1])
                or timing["effective_duration"] != timing["actual_stop"]-timing["actual_onset"]
                or timing["time_unit"] != wave["time_unit"]
                or timing["cumulative_stimulation"] != float(np.dot(levels[:-1], np.diff(times)))
                or timing["cumulative_stimulation_unit"] != wave["value_unit"]+"*"+wave["time_unit"]
                or timing["requested_duration_exact"] != (timing["source_requested_duration"] == timing["effective_duration"])):
            raise ValueError("Source action timing/burden differs from the actual compiled intervals")
    return expected["compiled_program"]
