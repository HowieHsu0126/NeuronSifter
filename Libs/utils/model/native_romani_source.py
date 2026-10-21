"""Vm-only Romani source primitives; no trajectory is produced by initialization.

The maintained ModelDB loader and staging/build path are reused. The only staged
HOC adaptation disables ExperimentControl's nonphysical variable-dump cleanup.
All mechanism code, morphology, current_balance and membrane equations remain
unchanged. NEURON is optional until an explicit staged cell is loaded.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess

import numpy as np

SCHEMA = "native_romani_vm_source.v1"
PARAMETER_UNITS = {
    "somatic_sodium_conductance": "S/cm2",
    "somatic_delayed_rectifier_conductance": "S/cm2",
    "somatic_membrane_resistance": "ohm*cm2",
    "somatic_axial_resistivity": "ohm*cm",
    "somatic_membrane_capacitance": "uF/cm2",
    "initial_membrane_voltage": "mV",
    "temperature": "degC",
}
_LOADED_LIBRARIES: set[str] = set()
_CLEANUP_BODY = """proc clear_variable_dump() {
  sprint(tmp_str, "system(\\"rm -rf %s/variable_dump\\")", data_dir)
  execute1(tmp_str)
}"""
_SAFE_CLEANUP_BODY = """proc clear_variable_dump() {
  // Native Vm source: variable-dump cleanup disabled in task-owned staged copy.
}"""


def definition(config: dict) -> dict:
    value = config["native_romani_source"]
    if value.get("schema_version") != SCHEMA:
        raise ValueError("Unknown native Romani source schema")
    declarations = value["parameters"]
    if set(declarations) != set(PARAMETER_UNITS):
        raise ValueError("Romani source requires the complete typed HOC parameter tuple")
    for name, unit in PARAMETER_UNITS.items():
        row = declarations[name]
        if row.get("unit") != unit:
            raise ValueError("Romani parameter unit mismatch: " + name)
        if set(row) != {"value", "unit", "seed_jitter", "factor_binding"}:
            raise ValueError("Romani parameter declaration is incomplete: " + name)
        nominal, jitter = row["value"], row["seed_jitter"]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (nominal, jitter)) or jitter < 0:
            raise ValueError("Romani parameter nominal/jitter must be finite")
        factor, coordinate = row["factor_binding"]
        if not isinstance(factor, str) or type(coordinate) is not int or coordinate < 0:
            raise ValueError("Romani parameter factor binding is invalid")
    validate_parameters({key: declarations[key]["value"] for key in declarations})
    site = value["site"]
    if (site["section_name"] != "soma[0]" or site["segment_x"] != .5
            or value["somatic_sections"] != [f"soma[{i}]" for i in range(5)]):
        raise ValueError("Romani contract requires the five real somata and soma[0](.5)")
    solver = value["solver"]
    steps = solver["dt_ms"]
    if (len(steps) != 3 or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                  and math.isfinite(x) and x > 0 for x in steps)
            or not steps[0] > steps[1] > steps[2]
            or solver["cvode_active"] is not False or solver["secondorder"] != 0
            or solver["threads"] != 1):
        raise ValueError("Romani solver requires three decreasing fixed steps and one thread")
    return value


def validate_parameters(parameters: dict) -> None:
    if set(parameters) != set(PARAMETER_UNITS):
        raise ValueError("Romani parameters must match all source-declared physical names")
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Romani parameters must be finite numbers: " + name)
        if name not in {"initial_membrane_voltage", "temperature"} and value <= 0:
            raise ValueError("Romani conductance/resistance/capacitance must be positive: " + name)
        if name == "temperature" and value <= -273.15:
            raise ValueError("Romani temperature must exceed absolute zero")


def parameter_candidates(config: dict, factors: dict, rng: np.random.Generator) -> list[dict]:
    """Build every named tuple before the caller selects a hidden atom."""
    declarations = definition(config)["parameters"]
    counts = {np.asarray(value).shape[0] for value in factors.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise ValueError("Romani parameters require aligned complete factor tables")
    count = next(iter(counts))
    if count != definition(config)["prior_particle_count"]:
        raise ValueError("Romani source requires all32 complete prior particles")
    jitter = {name: float(rng.uniform(0., 1.)) * row["seed_jitter"] for name, row in declarations.items()}
    result = []
    for index in range(count):
        current = {}
        for name, row in declarations.items():
            factor, coordinate = row["factor_binding"]
            values = np.asarray(factors[factor])
            if values.ndim != 2 or coordinate >= values.shape[1] or not np.isfinite(values).all():
                raise ValueError("Romani parameter factor coordinate is unavailable")
            current[name] = float(row["value"]) + (-1. if values[index, coordinate] < 1. else 1.) * jitter[name]
        validate_parameters(current)
        result.append(current)
    return result


def disable_staged_cleanup(path: Path) -> dict:
    """Disable only the known nonphysical deletion, after copying source to scratch."""
    original = path.read_text()
    if original.count(_CLEANUP_BODY) != 1:
        raise ValueError("Romani staged cleanup body differs from the inspected source")
    path.write_text(original.replace(_CLEANUP_BODY, _SAFE_CLEANUP_BODY, 1))
    return {"path": str(path), "adaptation": "disable_ExperimentControl_variable_dump_cleanup",
            "physical_equations_changed": False}


def stage_source(config: dict, source_root: Path, scratch: Path) -> dict:
    """Use a fresh, caller-owned node-local workspace and the existing serial build."""
    policy = definition(config)
    source_root = source_root.resolve(strict=True)
    scratch = scratch.resolve(strict=True)
    if scratch == source_root or source_root in scratch.parents or scratch in source_root.parents:
        raise ValueError("Romani staging must be disjoint from the external source")
    filesystem = subprocess.run(["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"],
                                capture_output=True, text=True, check=True).stdout.strip()
    if not filesystem or filesystem.startswith(("nfs", "cifs", "lustre", "gpfs")):
        raise ValueError("Romani source staging requires node-local storage")
    if any(scratch.iterdir()):
        raise ValueError("Romani staging requires a fresh task-owned directory")
    # Optional NEURON import and compile occur only through this explicit entry.
    from Libs.scripts.teacher_generation.run_neuron_rxd_microenvironment_teacher import _stage_registration_source
    prepared = _stage_registration_source(source_root, scratch, {
        "source_dependency_directories": policy["source_dependency_directories"],
        "recording_mode": "endpoint_v1",
    })
    staged = Path(prepared["build"]["staged_root"])
    prepared["native_adaptations"] = [disable_staged_cleanup(staged / "template/ExperimentControl.hoc")]
    prepared["schema_version"] = SCHEMA
    prepared["purpose"] = "native_vm_source_preparation_no_trajectory"
    return prepared


def current_program(amplitude_uA: float, delay_ms: float, duration_ms: float) -> dict:
    """Physical units at the compiler/NEURON boundary; no amplitude rescaling."""
    values = (amplitude_uA, delay_ms, duration_ms)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("IClamp amplitude and times must be finite numbers")
    if amplitude_uA < 0 or delay_ms < 0 or duration_ms <= 0:
        raise ValueError("Declared depolarizing IClamp requires nonnegative current/delay and positive duration")
    return {"amplitude_uA": float(amplitude_uA), "amplitude_nA": float(amplitude_uA) * 1000.,
            "delay_ms": float(delay_ms), "duration_ms": float(duration_ms),
            "delay_hours": float(delay_ms) / 3_600_000., "duration_hours": float(duration_ms) / 3_600_000.}


class RomaniCell:
    """One staged morphology; each candidate is reapplied and initialized at t=0."""

    def __init__(self, config: dict, prepared: dict):
        self.policy = copy.deepcopy(definition(config))
        self.prepared = prepared
        staged = Path(prepared["build"]["staged_root"]).resolve(strict=True)
        source = Path(prepared["source_root"]).resolve(strict=True)
        cleanup = staged / "template/ExperimentControl.hoc"
        if (source == staged or source in staged.parents or
                _SAFE_CLEANUP_BODY not in cleanup.read_text() or _CLEANUP_BODY in cleanup.read_text()):
            raise ValueError("Only the explicitly adapted staged Romani source may be loaded")
        from Libs.scripts.teacher_generation.run_neuron_rxd_microenvironment_teacher import h, load_modeldb_source_cell
        self.h = h
        h.load_file("stdrun.hoc")
        h.v_init = self.policy["parameters"]["initial_membrane_voltage"]["value"]
        h.dt = self.policy["solver"]["dt_ms"][0]
        library = str(staged / prepared["build"]["library_relative_path"])
        prior_cwd = os.getcwd()
        try:
            load_modeldb_source_cell(config, self.policy["source_model_id"], source_root=staged,
                                     load_mechanisms=library not in _LOADED_LIBRARIES, force_reload=True)
            _LOADED_LIBRARIES.add(library)
        finally:
            os.chdir(prior_cwd)
        h.cvode_active(0)
        h.secondorder = self.policy["solver"]["secondorder"]
        h.ParallelContext().nthread(self.policy["solver"]["threads"])
        somata = {sec.name(): sec for sec in h.allsec() if sec.name().startswith("soma[")}
        if set(somata) != set(self.policy["somatic_sections"]):
            raise ValueError("Loaded Romani source must expose exactly the five declared soma sections")
        self.somata = somata
        self.section = somata[self.policy["site"]["section_name"]]
        self.segment = self.section(self.policy["site"]["segment_x"])
        if float(self.segment.x) != self.policy["site"]["segment_x"]:
            raise ValueError("Romani recording site does not resolve to the registered segment")
        self.stimulus = h.IClamp(self.segment)
        self.stimulus.amp, self.stimulus.delay, self.stimulus.dur = 0., 0., 0.
        self.initialization_count = 1  # loadBasicModel -> current_balance
        self.readback_count = 0
        if float(h.t) != 0.:
            raise ValueError("Romani HOC startup unexpectedly advanced time")

    def initialize(self, parameters: dict) -> dict:
        validate_parameters(parameters)
        h = self.h
        h.celsius = parameters["temperature"]
        h.v_init = parameters["initial_membrane_voltage"]
        for section in self.somata.values():
            section.Ra = parameters["somatic_axial_resistivity"]
            for segment in section:
                segment.cm = parameters["somatic_membrane_capacitance"]
                segment.gbar_na3 = parameters["somatic_sodium_conductance"]
                segment.gkdrbar_kdr = parameters["somatic_delayed_rectifier_conductance"]
                segment.g_pas = 1. / parameters["somatic_membrane_resistance"]
        # Recompute the original source's derived e_pas after every complete tuple.
        # This initializes gating states/ionic currents but never advances time.
        h.current_balance(h.v_init)
        self.initialization_count += 1
        if float(h.t) != 0.:
            raise ValueError("Romani parameter initialization unexpectedly advanced time")
        readback = self.readback()
        for row in readback["somatic_segments"]:
            actual = {
                "somatic_sodium_conductance": row["gbar_na3_S_cm2"],
                "somatic_delayed_rectifier_conductance": row["gkdrbar_kdr_S_cm2"],
                "somatic_membrane_resistance": 1. / row["g_pas_S_cm2"],
                "somatic_axial_resistivity": row["Ra_ohm_cm"],
                "somatic_membrane_capacitance": row["cm_uF_cm2"],
                "initial_membrane_voltage": row["Vm_mV"],
                "temperature": readback["temperature_degC"],
            }
            if any(not math.isclose(actual[name], value, rel_tol=1e-12, abs_tol=1e-12)
                   for name, value in parameters.items()):
                raise ValueError("Romani HOC parameter application/readback mismatch")
        readback["parameters"] = copy.deepcopy(parameters)
        return readback

    def readback(self) -> dict:
        rows = []
        for name, section in self.somata.items():
            for segment in section:
                row = {"section_name": name, "segment_x": float(segment.x),
                       "gbar_na3_S_cm2": float(segment.gbar_na3), "gkdrbar_kdr_S_cm2": float(segment.gkdrbar_kdr),
                       "g_pas_S_cm2": float(segment.g_pas), "Ra_ohm_cm": float(section.Ra),
                       "cm_uF_cm2": float(segment.cm), "derived_e_pas_mV": float(segment.e_pas),
                       "Vm_mV": float(segment.v), "area_um2": float(segment.area())}
                if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError("Nonfinite Romani parameter readback")
                rows.append(row)
        self.readback_count += 1
        return {"time_ms": float(self.h.t), "temperature_degC": float(self.h.celsius),
                "recording_site": copy.deepcopy(self.policy["site"]), "Vm_mV": float(self.segment.v),
                "somatic_segments": rows, "initialization_count": self.initialization_count}

    def configure_current(self, amplitude_uA: float, delay_ms: float, duration_ms: float) -> dict:
        program = current_program(amplitude_uA, delay_ms, duration_ms)
        self.stimulus.amp = program["amplitude_nA"]
        self.stimulus.delay = program["delay_ms"]
        self.stimulus.dur = program["duration_ms"]
        actual = {name: float(getattr(self.stimulus, attr)) for name, attr in
                  (("amplitude_nA", "amp"), ("delay_ms", "delay"), ("duration_ms", "dur"))}
        if any(actual[name] != program[name] for name in actual):
            raise ValueError("Romani IClamp parameter readback mismatch")
        return {**program, "IClamp_readback": actual, "time_ms": float(self.h.t)}

    def trajectory(self, parameters: dict, action: dict, grid: dict, dt_ms: float) -> dict:
        """Advance the actual source only when explicitly called by its producer/solver."""
        times = np.asarray(grid["query_time"]["values"], dtype=float)
        if grid["query_time"]["unit"] != "ms" or times[0] != 0 or not (np.diff(times) > 0).all():
            raise ValueError("Romani trajectory needs the registered increasing millisecond clock")
        if dt_ms not in self.policy["solver"]["dt_ms"]:
            raise ValueError("Romani timestep is outside the prospective refinement contract")
        ticks = times / dt_ms
        if not np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-8):
            raise ValueError("Romani output times must coincide with actual solver steps")
        extension = action["extension"]
        if extension["dose_or_intensity"]["unit"] != "uA" or extension["duration"]["unit"] != "ms":
            raise ValueError("Romani IClamp units differ from the compiler contract")
        self.stimulus.amp = 0.
        self.h.dt = dt_ms
        readback = self.initialize(parameters)
        program = self.configure_current(extension["dose_or_intensity"]["value"],
            float(times[action["response_delay_steps"]]), extension["duration"]["value"])
        values, actual_times, steps = [[float(self.segment.v)]], [float(self.h.t)], 0
        for target_tick in np.rint(ticks[1:]).astype(int):
            while steps < target_tick:
                self.h.fadvance()
                steps += 1
            actual_times.append(float(self.h.t))
            values.append([float(self.segment.v)])
        if not np.allclose(actual_times, times, rtol=0, atol=1e-8) or not np.isfinite(values).all():
            raise ValueError("Romani recorded Vm/time differs from registered native support")
        return {"values": np.asarray(values, dtype=np.float64), "feature_names":["membrane_voltage"],
                "time_ms": actual_times, "site": copy.deepcopy(self.policy["site"]),
                "dt_ms": dt_ms, "solver": "NEURON_fixed_step_source_HOC",
                "parameter_readback": readback, "current_program": program,
                "time_advance_calls": steps}

    def verify_candidates(self, candidates: list[dict]) -> dict:
        if len(candidates) != self.policy["prior_particle_count"]:
            raise ValueError("Romani source requires all32 complete prior particles")
        rows = [self.initialize(candidate) for candidate in candidates]
        return {"schema_version": SCHEMA, "candidate_count": len(rows), "readbacks": rows,
                "hoc_startup_initializations": 1, "candidate_initializations": len(rows),
                "time_advance_calls": 0, "final_time_ms": float(self.h.t),
                "scientific_source_admission": "NOT_RUN", "trajectory_count": 0}
