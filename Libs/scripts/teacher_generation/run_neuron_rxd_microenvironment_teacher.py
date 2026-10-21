from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
import sys

import numpy as np

import neuron
from neuron import h

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation import source_membrane_charge_registration as registration
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NEURON/RxD AD-first full microenvironment teacher rollout.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--source-model-id", default="modeldb_romani_2013_ca3_ca1")
    parser.add_argument("--operation", choices=("legacy", "source-registration", "source-registration-case", "analyze-source-registration"), default="legacy")
    parser.add_argument("--run-id")
    parser.add_argument("--source-root")
    parser.add_argument("--scratch-root")
    parser.add_argument("--result-dir")
    parser.add_argument("--case-id")
    parser.add_argument("--dispatch-token", default="")
    parser.add_argument("--registration-recording-mode", choices=("endpoint_v1", registration.LEDGER_MODE))
    return parser.parse_args()


def reset_neuron() -> None:
    h("forall delete_section()")


def load_modeldb_source_cell(config: dict, source_model_id: str, *, source_root: Path | None = None,
                             load_mechanisms: bool = True, force_reload: bool = False):
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    modeldb_config = teacher_config["model_sources"]["modeldb"]
    if source_model_id != str(modeldb_config["source_model_id"]):
        raise ValueError(f"Unsupported ModelDB source_model_id {source_model_id}; expected {modeldb_config['source_model_id']}.")
    mechanism_root = resolve_repo_path(str(modeldb_config["mechanism_root"]))
    simulation_hoc_path = resolve_repo_path(str(modeldb_config["simulation_hoc_path"]))
    if source_root is not None:
        source_layout = Path(str(modeldb_config["extracted_root"]))
        mechanism_root = source_root / Path(str(modeldb_config["mechanism_root"])).relative_to(source_layout)
        simulation_hoc_path = source_root / Path(str(modeldb_config["simulation_hoc_path"])).relative_to(source_layout)
    compiled_library = mechanism_root / str(modeldb_config["compiled_library_relpath"])
    required_paths = [mechanism_root, simulation_hoc_path, compiled_library]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"ModelDB NEURON source is not compiled/materialized: {missing_paths}")
    reset_neuron()
    if load_mechanisms:
        h.nrn_load_dll(str(compiled_library))
    h.chdir(str(mechanism_root))
    h.load_file("stdrun.hoc")
    loaded = h.load_file(1, str(simulation_hoc_path)) if force_reload else h.load_file(str(simulation_hoc_path))
    if int(loaded) != 1:
        raise RuntimeError(f"NEURON failed to load source-backed ModelDB HOC file: {simulation_hoc_path}")
    sections = list(h.allsec())
    if not sections:
        raise RuntimeError(f"ModelDB source {source_model_id} did not create any NEURON sections.")
    soma_sections = [section for section in sections if "soma" in section.name().lower()]
    if not soma_sections:
        raise RuntimeError(f"ModelDB source {source_model_id} did not expose a soma section for AD microenvironment recording.")
    return soma_sections[0]


def configure_rxd(soma, runtime: dict[str, object]):
    from neuron import rxd

    cytosol = rxd.Region([soma], name="cytosol", nrn_region="i")
    calcium = rxd.Species(cytosol, name="calcium", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["calcium_initial_millimolar"]))
    amyloid = rxd.Species(cytosol, name="amyloid_beta", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["amyloid_initial_millimolar"]))
    tau = rxd.Species(cytosol, name="tau", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["tau_initial_millimolar"]))
    inflammatory = rxd.Species(cytosol, name="inflammatory_mediator", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["inflammatory_initial_millimolar"]))
    rxd.Rate(calcium, -float(runtime["rxd_decay_per_ms"]) * calcium)
    rxd.Rate(tau, float(runtime["amyloid_tau_coupling_per_ms"]) * amyloid - float(runtime["rxd_decay_per_ms"]) * tau)
    rxd.Rate(inflammatory, float(runtime["inflammation_amyloid_coupling_per_ms"]) * amyloid - float(runtime["rxd_decay_per_ms"]) * inflammatory)
    return calcium, amyloid, tau, inflammatory


def run_protocol(config: dict, source_model_id: str) -> dict[str, np.ndarray | float]:
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    runtime = teacher_config["protocol_runtime"]
    soma = load_modeldb_source_cell(config, source_model_id)
    calcium, amyloid, tau, inflammatory = configure_rxd(soma, runtime)
    stimulus = h.IClamp(soma(0.5))
    stimulus.delay = float(runtime["stimulus_delay_ms"])
    stimulus.dur = float(runtime["stimulus_duration_ms"])
    stimulus.amp = float(runtime["stimulus_amplitude_nA"])
    voltage_vector = h.Vector().record(soma(0.5)._ref_v)
    time_vector = h.Vector().record(h._ref_t)
    calcium_vector = h.Vector().record(calcium.nodes(soma)(0.5)._ref_concentration)
    amyloid_vector = h.Vector().record(amyloid.nodes(soma)(0.5)._ref_concentration)
    tau_vector = h.Vector().record(tau.nodes(soma)(0.5)._ref_concentration)
    inflammatory_vector = h.Vector().record(inflammatory.nodes(soma)(0.5)._ref_concentration)
    h.dt = float(runtime["h_dt_ms"])
    h.finitialize(float(runtime["resting_voltage_millivolt"]))
    h.continuerun(float(runtime["simulation_duration_ms"]))
    time_ms = np.asarray(time_vector, dtype=np.float32)
    voltage = np.asarray(voltage_vector, dtype=np.float32)
    calcium_trace = np.asarray(calcium_vector, dtype=np.float32)
    amyloid_trace = np.asarray(amyloid_vector, dtype=np.float32)
    tau_trace = np.asarray(tau_vector, dtype=np.float32)
    inflammatory_trace = np.asarray(inflammatory_vector, dtype=np.float32)
    spike_threshold = float(teacher_config["voltage_normalization"]["depolarized_millivolt"])
    spike_count = float(np.count_nonzero((voltage[1:] >= spike_threshold) & (voltage[:-1] < spike_threshold)))
    return {
        "time_ms": time_ms,
        "voltage": voltage,
        "calcium": calcium_trace,
        "amyloid": amyloid_trace,
        "tau": tau_trace,
        "inflammatory": inflammatory_trace,
        "spike_count": spike_count,
    }


def normalize_voltage(config: dict, voltage: np.ndarray) -> np.ndarray:
    normalization = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["voltage_normalization"]
    resting = float(normalization["resting_millivolt"])
    depolarized = float(normalization["depolarized_millivolt"])
    return np.clip((voltage - resting) / max(depolarized - resting, 1.0e-12), 0.0, 1.5).astype(np.float32)


def normalize_positive(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / maximum, 0.0, 1.5).astype(np.float32)


def protocol_to_trajectory(config: dict, payload: dict[str, np.ndarray | float]) -> np.ndarray:
    from Libs.utils.model.dynamics_fields import compartment_id_list, region_id_list, species_id_list, time_grid_hours

    time_values = time_grid_hours(config)
    time_ms = np.asarray(payload["time_ms"], dtype=np.float32)
    region_ids = region_id_list(config)
    compartment_ids = compartment_id_list(config)
    species_ids = species_id_list(config)
    species_lookup = {species_id: index for index, species_id in enumerate(species_ids)}
    trajectory = np.zeros((time_values.shape[0], len(region_ids), len(compartment_ids), len(species_ids)), dtype=np.float32)
    target_time_ms = np.linspace(float(time_ms.min()), float(time_ms.max()), time_values.shape[0], dtype=np.float32)
    voltage = normalize_voltage(config, np.interp(target_time_ms, time_ms, np.asarray(payload["voltage"], dtype=np.float32)).astype(np.float32))
    calcium = normalize_positive(np.interp(target_time_ms, time_ms, np.asarray(payload["calcium"], dtype=np.float32)).astype(np.float32))
    amyloid = normalize_positive(np.interp(target_time_ms, time_ms, np.asarray(payload["amyloid"], dtype=np.float32)).astype(np.float32))
    tau = normalize_positive(np.interp(target_time_ms, time_ms, np.asarray(payload["tau"], dtype=np.float32)).astype(np.float32))
    inflammatory = normalize_positive(np.interp(target_time_ms, time_ms, np.asarray(payload["inflammatory"], dtype=np.float32)).astype(np.float32))
    spike_rate = np.full_like(voltage, float(payload["spike_count"]) / max(float(target_time_ms[-1] - target_time_ms[0]), 1.0))
    for region_index in range(len(region_ids)):
        for compartment_index in range(len(compartment_ids)):
            trajectory[:, region_index, compartment_index, species_lookup["membrane_voltage"]] = voltage
            trajectory[:, region_index, compartment_index, species_lookup["calcium_concentration"]] = calcium
            trajectory[:, region_index, compartment_index, species_lookup["spike_rate"]] = spike_rate
            trajectory[:, region_index, compartment_index, species_lookup["amyloid_beta_soluble"]] = amyloid
            trajectory[:, region_index, compartment_index, species_lookup["phosphorylated_tau"]] = tau
            trajectory[:, region_index, compartment_index, species_lookup["inflammatory_mediator"]] = inflammatory
            trajectory[:, region_index, compartment_index, species_lookup["circuit_instability"]] = np.clip(voltage + calcium + spike_rate + inflammatory, 0.0, 1.5)
    return trajectory


def sanity_check(config: dict, payload: dict[str, np.ndarray | float]) -> None:
    bounds = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["sanity_bounds"]
    resting_voltage = float(np.asarray(payload["voltage"], dtype=np.float32)[0])
    if resting_voltage < float(bounds["resting_voltage_millivolt_min"]) or resting_voltage > float(bounds["resting_voltage_millivolt_max"]):
        raise ValueError(f"NEURON resting voltage {resting_voltage} mV outside configured sanity bounds.")
    if float(np.min(np.asarray(payload["calcium"], dtype=np.float32))) < float(bounds["calcium_concentration_min"]):
        raise ValueError("NEURON/RxD calcium concentration is below configured minimum.")
    if float(payload["spike_count"]) < float(bounds["spike_count_min"]):
        raise ValueError("NEURON spike count is below configured minimum.")



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_registration_source(source: Path, scratch: Path, policy: dict) -> dict:
    staged = scratch / "source"
    identities = {}
    for subdirectory in policy["source_dependency_directories"]:
        for path in sorted((source / subdirectory).rglob("*")):
            if path.is_file() and path.suffix in (".hoc", ".mod"):
                relative = str(path.relative_to(source))
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = path.read_bytes()
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError(f"staged source differs: {relative}")
                if not target.exists():
                    target.write_bytes(payload)
                identities[relative] = hashlib.sha256(payload).hexdigest()
    if not identities:
        raise ValueError("configured source dependencies are absent")
    library = staged / "experiment/x86_64/libnrnmech.so"
    receipt_path = scratch / "build_receipt.json"
    # Reuse this task's already completed staged build only after its source
    # copies and recorded successful command are checked; never use ROOT's binary.
    if receipt_path.exists() and library.exists():
        receipt = json.loads(receipt_path.read_text())
        old_sources = json.loads((scratch / "source_hashes.json").read_text())
        if receipt["exit_code"] != 0 or old_sources != identities:
            raise ValueError("prepared build/source binding mismatch")
    else:
        shim = scratch / "build_tools"
        shim.mkdir(exist_ok=True)
        make = shutil.which("make")
        compiler = shutil.which("nrnivmodl")
        if not make or not compiler:
            raise FileNotFoundError("nrnivmodl and make are required")
        # Installed NEURON 9.0.1 requests -j4; the final make option limits the
        # actual compilation to the one allocated CPU without changing MOD code.
        launcher = shim / "make"
        launcher.write_text("#!/bin/sh\nexec " + shlex.quote(make) + ' "$@" -j1\n')
        launcher.chmod(0o755)
        environment = _registration_environment(scratch)
        environment["PATH"] = str(shim) + os.pathsep + environment["PATH"]
        started = time.monotonic()
        with (scratch / "build.log").open("w") as log:
            process = subprocess.run([compiler], cwd=staged / "experiment",
                                     env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt = {"command": [compiler], "exit_code": process.returncode,
                   "elapsed_seconds": time.monotonic() - started,
                   "maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                   "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID"),
                   "make_parallelism": 1}
        registration.write_json(receipt_path, receipt)
        registration.write_json(scratch / "source_hashes.json", identities)
        if process.returncode:
            raise RuntimeError("staged source build failed; see build.log")
    if not library.exists():
        raise FileNotFoundError("staged compiled library is absent")
    neuron_root = str(Path(neuron.__file__).parent.resolve())
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if neuron_root in line and ".so" in line and line.split()[-1].startswith("/")}
    runtime_libraries = {path: _sha256(Path(path)) for path in sorted(loaded)}
    if not runtime_libraries:
        raise ValueError("actual NEURON runtime libraries were not identified")
    receipt["identity"] = {
        "neuron_runtime_libraries": runtime_libraries,
        "library_sha256": _sha256(library),
        "source_files_uid": registration.content_uid(identities),
        "neuron_version": neuron.__version__, "nrnversion": h.nrnversion(),
        "python_executable": sys.executable, "python_executable_sha256": _sha256(Path(sys.executable)),
    }
    receipt["compiler_script_sha256"] = _sha256(Path(neuron.__file__).parent / ".data/bin/nrnivmodl")
    receipt["make_shim"] = (scratch / "build_tools/make").read_text()
    receipt["library_relative_path"] = "experiment/x86_64/libnrnmech.so"
    receipt["staged_root"] = str(staged)
    prepared = {"source_root": str(source), "files": identities, "build": receipt}
    if registration.recording_version(policy) == registration.LEDGER_VERSION:
        original = source / library.relative_to(staged)
        prepared["original_library"] = {"relative_path": str(library.relative_to(staged)),
                                         "sha256": _sha256(original) if original.exists() else None}
    return prepared


def _registration_environment(scratch: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch), CUDA_VISIBLE_DEVICES="")
    return environment


def _load_registration_cell(config: dict, source_model_id: str, staged: Path, policy: dict):
    h.load_file("stdrun.hoc")
    h.dt = policy["dt_ms"][0]
    h.v_init = policy["initial_voltage_mV"]
    load_modeldb_source_cell(config, source_model_id, source_root=staged)
    matches = [section for section in h.allsec() if section.name() == policy["section_name"]]
    if len(matches) != 1:
        raise ValueError("prospective soma[0] site does not resolve uniquely")
    section = matches[0]
    segment = section(policy["segment_x"])
    # This is the actual source discretization, not a substituted cylinder.
    # Endpoint ri mappings below are defined for this one root segment.
    if section.nseg != 1 or section.parentseg() is not None or section.orientation() != 0 or segment.x != policy["segment_x"]:
        raise ValueError("source site no longer has the declared single-root-segment support")
    site = {
        "section_name": section.name(), "segment_x": segment.x, "nseg": section.nseg,
        "area_um2": segment.area(), "cm_uF_cm2": segment.cm, "Ra_ohm_cm": section.Ra,
        "capacitance_nF": registration.capacitance_nf(segment.cm, segment.area()),
        "length_um": section.L, "diameter_um": segment.diam,
        "center_node_index": segment.node_index(),
        "left_node": {"section_name": section.name(), "x": 0.0, "node_index": section(0).node_index()},
        "right_node": {"section_name": section.name(), "x": 1.0, "node_index": section(1).node_index()},
        "left_ri_MOhm": h.ri(segment.x, sec=section),
        "right_ri_MOhm": h.ri(1.0, sec=section),
        "ri_mapping": "left: ri(0.5), center-to-proximal endpoint; right: ri(1), distal endpoint-to-center",
        "support": "one source root membrane segment, not a whole-soma average",
    }
    for sec in h.allsec():
        if not np.isfinite([sec.L, sec.Ra]).all() or min(sec.L, sec.Ra) <= 0:
            raise ValueError(f"invalid source geometry/resistivity: {sec.name()}")
        for sg in sec:
            registration.capacitance_nf(sg.cm, sg.area())
        for mechanism in sec.psection()["density_mechs"].values():
            for name, values in mechanism.items():
                a = np.asarray(values, dtype=float)
                if not np.isfinite(a).all():
                    raise ValueError(f"nonfinite source mechanism field: {sec.name()}/{name}")
                if name.startswith("g") and name != "ggk" and (a < 0).any():
                    raise ValueError(f"negative source conductance: {sec.name()}/{name}")
    for key in ("left_ri_MOhm", "right_ri_MOhm"):
        if not np.isfinite(site[key]) or site[key] <= 0:
            raise ValueError("invalid source axial resistance")
    time_constants = {
        "h.taun": segment.h.taun, "cat.taum": segment.cat.taum, "cat.tauh": segment.cat.tauh,
        "kca.tau_m": segment.kca.tau_m, "somacar.tau0": segment.somacar.tau[0],
        "somacar.tau1": segment.somacar.tau[1],
    }
    for name in ("mtau_na3", "htau_na3", "taus_na3", "taun_kdr", "tau_km",
                 "taul_kap", "taun_kap", "taum_cal", "tau_mykca", "taur_cad"):
        time_constants[name] = float(getattr(h, name))
    if not np.isfinite(list(time_constants.values())).all() or min(time_constants.values()) <= 0:
        raise ValueError(f"invalid initialized source kinetic time constant: {time_constants}")
    psection = section.psection()
    expected_ions = {"na", "k", "ca", "Ca"}
    if set(psection["ions"]) != expected_ions or psection["point_processes"]:
        raise ValueError("source current partition differs from the predeclared base-cell accounting")
    for name in ("ina", "ik", "ica", "iCa", "i_pas"):
        float(getattr(segment, name))
    states = {"state_cai": "cai"}
    reference = h.ref("")
    parameters = {}
    for name in sorted(psection["density_mechs"]):
        state = h.MechanismStandard(name, 3)
        for i in range(int(state.count())):
            state.name(reference, i)
            variable = str(reference[0])
            states["state_" + variable] = variable
        parameter = h.MechanismStandard(name, 1)
        parameters[name] = {}
        for i in range(int(parameter.count())):
            parameter.name(reference, i)
            variable = str(reference[0])
            try:
                parameters[name][variable] = float(getattr(segment, variable))
            except AttributeError:
                parameters[name][variable] = float(getattr(h, variable))
    return section, site, states, {
        "celsius": float(h.celsius), "section_count": len(list(h.allsec())),
        "density_parameters": parameters, "initialized_time_constants_ms": time_constants,
        "time_constant_scope": "initialized source kinetics; not a guarantee over every subsequent voltage/state",
        "ions": {name: {k: v for k, v in values.items() if k in ("ena", "ek", "eca", "eCa", "cao", "Cao")}
                 for name, values in psection["ions"].items()},
    }


def _verify_registration_inputs(declaration: dict) -> None:
    registration.validate_declaration(declaration)
    staged = Path(declaration["build"]["staged_root"])
    for relative, expected in declaration["source"]["files"].items():
        if _sha256(staged / relative) != expected:
            raise ValueError(f"staged source changed: {relative}")
    if _sha256(staged / declaration["build"]["library_relative_path"]) != declaration["build"]["identity"]["library_sha256"]:
        raise ValueError("staged library changed")
    if neuron.__version__ != declaration["build"]["identity"]["neuron_version"] or h.nrnversion() != declaration["build"]["identity"]["nrnversion"]:
        raise ValueError("NEURON build changed")
    for path, expected in declaration["build"]["identity"]["neuron_runtime_libraries"].items():
        if _sha256(Path(path)) != expected:
            raise ValueError("NEURON runtime library changed")
    for path, expected in declaration["implementation"].items():
        if _sha256(REPO_ROOT / path) != expected:
            raise ValueError(f"executing registration implementation changed: {path}")


def _empty_registration_trace(declaration: dict, condition: dict) -> dict:
    trace = {
        "schema_version": declaration["schema_version"],
        "declaration_uid": declaration["declaration_uid"], "condition": condition,
        "site": declaration["site"], "units": declaration["units"],
        "build_identity": declaration["build"]["identity"], "status": "NOT_RUN",
        "attempt_started": False, "samples": {}, "observed_mask": [],
        "source_time_advanced_count": 0, "error": None, "cost": {},
    }
    if declaration["schema_version"] == registration.LEDGER_VERSION:
        trace["step_ledger"] = registration.empty_step_ledger(declaration)
    return trace



def _validate_step_range_variables(segment) -> None:
    for name in ("i_membrane_", "i_cap", *[n.removesuffix("_mA_cm2") for n in registration.ION_COLUMNS]):
        if not np.isfinite(float(getattr(segment, name))):
            raise ValueError(f"unavailable finite source-step recording interface: {name}")


def _validate_registration_step_interface(section, policy: dict) -> None:
    cvode = h.CVode()
    previous = cvode.use_fast_imem()
    try:
        cvode.use_fast_imem(1)
        _validate_step_range_variables(section(policy["segment_x"]))
        for name in ("extra_scatter_gather", "extra_scatter_gather_remove"):
            if not callable(getattr(cvode, name)):
                raise ValueError(f"unavailable source-step callback interface: {name}")
    finally:
        cvode.use_fast_imem(previous)


class SourceStepCurrentRecorder:
    """Read source-assigned currents at the verified post-voltage/pre-state phase."""

    def __init__(self, section, stimuli: list, trace: dict):
        self.section = section
        self.segment = section(trace["site"]["segment_x"])
        self.stimuli = stimuli
        self.trace = trace
        self.cvode = h.CVode()
        self.pc = h.ParallelContext()
        self.previous_fast_imem = self.cvode.use_fast_imem()
        self.cvode.use_fast_imem(1)
        self.callback = self.capture
        self.registered = False
        self.last_time = None

    def install(self) -> None:
        # Resolve the real exported range variables before the first advance.
        _validate_step_range_variables(self.segment)
        self.last_time = float(self.pc.t(0))
        self.cvode.extra_scatter_gather(0, self.callback)
        self.registered = True

    def capture(self) -> None:
        # h.t is not the authoritative thread clock inside fixed-step callbacks.
        end = float(self.pc.t(0))
        if self.last_time is None or not np.isclose(end - self.last_time, self.trace["condition"]["dt_ms"], rtol=0, atol=1e-8):
            raise ValueError("source-step callback phase/cardinality mismatch")
        sample = {
            "end_time_ms": end, "voltage_mV": float(self.segment.v),
            "left_voltage_mV": float(self.section(0).v), "right_voltage_mV": float(self.section(1).v),
            "membrane_current_nA": float(self.segment.i_membrane_),
            "capacitive_current_mA_cm2": float(self.segment.i_cap),
            "electrode_current_nA": sum(float(s.i) for s in self.stimuli),
            "assembled_ionic_current_mA_cm2": sum(float(getattr(self.segment, n.removesuffix("_mA_cm2")))
                                                   for n in registration.ION_COLUMNS),
        }
        if not np.isfinite(list(sample.values())).all():
            raise ValueError("nonfinite source-step current sample")
        for name, value in sample.items():
            self.trace["step_ledger"]["samples"][name].append(value)
        self.last_time = end

    def close(self) -> None:
        if self.registered:
            self.cvode.extra_scatter_gather_remove(self.callback)
            self.registered = False
        self.cvode.use_fast_imem(self.previous_fast_imem)


def _capture_interval_current_and_refresh(stimuli: list) -> float:
    current = sum(float(stimulus.i) for stimulus in stimuli)
    h.fcurrent()
    return current


def _run_registration_case(config: dict, root: Path, case_id: str, source_model_id: str, dispatch_token: str = "") -> None:
    declaration = json.loads((root / "declaration.json").read_text())
    _verify_registration_inputs(declaration)
    selected_policy = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"]
    if registration.recording_version(selected_policy) != declaration["schema_version"]:
        raise ValueError("selected recording mode differs from the bound declaration")
    if config["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"] != declaration["source"]["model_config"]:
        raise ValueError("configured source model changed after declaration")
    condition = next((c for c in registration.condition_matrix(declaration["policy"]) if c["case_id"] == case_id), None)
    if condition is None:
        raise ValueError("case outside predeclared matrix")
    path = root / (case_id + ".json")
    trace = json.loads(path.read_text())
    registration.validate_trace(trace, declaration)
    if dispatch_token:
        if trace["attempt_started"] is not True or trace["status"] != "DISPATCHED" or trace.get("dispatch_token") != dispatch_token:
            raise ValueError("source dispatch slot already consumed or token mismatch")
    elif trace["attempt_started"] is not False or trace["status"] != "NOT_RUN":
        raise ValueError("source case already attempted; no automatic scientific retry")
    trace["child_started"] = True
    trace["attempt_started"] = True
    trace["status"] = "RUNNING"
    registration.write_json(path, trace)
    started = time.monotonic()
    recorder = None
    try:
        policy = declaration["policy"]
        section, site, states, _ = _load_registration_cell(config, source_model_id, Path(declaration["build"]["staged_root"]), policy)
        if site != declaration["site"] or states != declaration["state_variables"]:
            raise ValueError("loaded site/state pointers differ from declaration")
        segment = section(policy["segment_x"])
        h.CVode().active(0)
        h.ParallelContext().nthread(policy["solver_threads"])
        h.secondorder = policy["secondorder"]
        h.dt = condition["dt_ms"]
        h.steps_per_ms = 1.0 / condition["dt_ms"]
        stimuli = []
        for active, key in ((condition["prehistory"] == "conditioned", "prehistory_pulse_ms"),
                            (condition["continuation"] == "test", "test_pulse_ms")):
            stimulus = h.IClamp(segment)
            a, b = policy[key]
            stimulus.delay, stimulus.dur = a, b - a
            stimulus.amp = policy["stimulus_amplitude_nA"] if active else 0.0
            stimuli.append(stimulus)
        if declaration["schema_version"] == registration.LEDGER_VERSION:
            recorder = SourceStepCurrentRecorder(section, stimuli, trace)
        h.finitialize(policy["initial_voltage_mV"])
        h.fcurrent()
        if recorder is not None:
            recorder.install()
        columns = ["time_ms", "voltage_mV", "electrode_nA", "left_voltage_mV", "right_voltage_mV",
                   "interval_electrode_nA", *registration.ION_COLUMNS, *states]
        data = {name: [] for name in columns}
        trace["samples"] = data
        steps = round(policy["duration_ms"] / condition["dt_ms"])
        for index in range(steps + 1):
            interval_current = 0.0  # no completed interval at initialization
            if index:
                before = float(h.ParallelContext().t(0))
                trace["source_advance_calls"] = trace.get("source_advance_calls", 0) + 1
                try:
                    h.fadvance()
                finally:
                    after = float(h.ParallelContext().t(0))
                    trace["source_time_advanced_count"] += int(after > before)
                    trace["source_final_time_ms"] = after
                if recorder is not None and len(trace["step_ledger"]["samples"]["end_time_ms"]) != index:
                    raise ValueError("source-step ledger missing after advance")
                # Capture the completed-interval forcing before endpoint
                # fcurrent overwrites assigned point-process currents.
                interval_current = _capture_interval_current_and_refresh(stimuli)
            sample = {
                "time_ms": float(h.t), "voltage_mV": float(segment.v),
                "electrode_nA": sum(float(stimulus.i) for stimulus in stimuli),
                "interval_electrode_nA": interval_current,
                "left_voltage_mV": float(section(0).v), "right_voltage_mV": float(section(1).v),
            }
            for name in registration.ION_COLUMNS:
                sample[name] = float(getattr(segment, name.removesuffix("_mA_cm2")))
            for name, variable in states.items():
                sample[name] = float(getattr(segment, variable))
            if not np.isfinite(list(sample.values())).all():
                raise ValueError(f"nonfinite source output at {h.t} ms")
            for name, value in sample.items():
                data[name].append(value)
            trace["observed_mask"].append(True)
        trace["status"] = "COMPLETE"
        u = np.asarray(data["electrode_nA"])
        switches = np.flatnonzero(np.diff(u) != 0) + 1
        trace["delivered_current_switches"] = [
            {"time_ms": data["time_ms"][int(i)], "current_nA": data["electrode_nA"][int(i)]} for i in switches
        ]
        trace["initial_source_parameters"] = {"celsius": float(h.celsius), "dt_ms": float(h.dt), "secondorder": int(h.secondorder)}
        registration.validate_trace(trace, declaration)
    except Exception as error:
        trace["status"] = "FAILED"
        trace["error"] = f"{type(error).__name__}: {error}"
    finally:
        if recorder is not None:
            recorder.close()
        trace["cost"] = {"elapsed_seconds": time.monotonic() - started,
                         "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID")}
        registration.write_json(path, trace)
    if trace["status"] != "COMPLETE":
        raise RuntimeError(trace["error"])



def _dispatch_registration_cases(args: argparse.Namespace, declaration: dict, root: Path, scratch: Path) -> None:
    """Reserve each finite slot before dispatch; a failed launch consumes that slot."""
    registration.validate_declaration(declaration)
    conditions = registration.condition_matrix(declaration["policy"])
    for condition in conditions:
        trace = json.loads((root / (condition["case_id"] + ".json")).read_text())
        registration.validate_trace(trace, declaration)
        if trace["attempt_started"] is not False or trace["status"] != "NOT_RUN":
            raise ValueError("matrix already attempted or outcome unresolved; no replacement/retry")
    for condition in conditions:
        case_id = condition["case_id"]
        trace_path = root / (case_id + ".json")
        trace = json.loads(trace_path.read_text())
        token = uuid.uuid4().hex
        trace.update(attempt_started=True, status="DISPATCHED", dispatch_token=token, child_started=False)
        command = [sys.executable, str(Path(__file__).resolve()), "--operation", "source-registration-case",
                   "--config", str(Path(args.config).resolve()), "--source-model-id", args.source_model_id,
                   "--result-dir", str(root), "--case-id", case_id, "--dispatch-token", token,
                   "--registration-recording-mode", declaration["policy"].get("recording_mode", "endpoint_v1")]
        trace["dispatch_receipt"] = {"command": command, "job_id": os.environ.get("SLURM_JOB_ID"),
                                     "step_id": os.environ.get("SLURM_STEP_ID"), "node": socket.gethostname(),
                                     "return_code": None, "error": None}
        registration.write_json(trace_path, trace)
        local_log = scratch / (case_id + ".log")
        started = time.monotonic()
        return_code, launch_error = None, None
        try:
            with local_log.open("w") as log:
                process = subprocess.run(command, cwd=REPO_ROOT, env=_registration_environment(scratch),
                                         stdout=log, stderr=subprocess.STDOUT)
            return_code = process.returncode
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"
        try:
            observed = json.loads(trace_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            observed = registration.unavailable_trace(declaration, condition, f"missing child evidence: {error}")
            observed["attempt_started"] = True
            observed["dispatch_token"] = token
        if observed["status"] == "RUNNING":
            observed["last_recorded_source_time_advanced_count"] = observed.get("source_time_advanced_count")
            observed["source_time_advanced_count"] = None
        observed["attempt_started"] = True
        observed["dispatch_receipt"] = {**trace["dispatch_receipt"], "return_code": return_code,
                                        "error": launch_error, "elapsed_seconds": time.monotonic() - started}
        if launch_error or return_code != 0 or observed["status"] != "COMPLETE":
            observed["status"] = "FAILED"
            observed["error"] = observed.get("error") or launch_error or f"child exit {return_code} without successful terminal evidence; no retry"
        registration.write_json(trace_path, observed)
        if local_log.exists():
            shutil.copyfile(local_log, root / local_log.name)
        print(case_id, observed["status"], flush=True)


def source_registration_operation(args: argparse.Namespace) -> None:
    args.config = str(Path(args.config).resolve())
    config = load_config(args.config)
    policy = dict(config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"])
    if getattr(args, "registration_recording_mode", None):
        policy["recording_mode"] = args.registration_recording_mode
    config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"] = policy
    registration.validate_policy(policy)
    registration.validate_ledger_contract(policy, config["schemas"]["source_membrane_charge_registration"])
    if args.operation == "analyze-source-registration":
        result = registration.analyze_directory(Path(args.result_dir).resolve())
        print(json.dumps({"status": result["status"], "completed_cases": result["completed_cases"]}))
        return
    if args.operation == "source-registration-case":
        _run_registration_case(config, Path(args.result_dir).resolve(), args.case_id, args.source_model_id, getattr(args, "dispatch_token", ""))
        return
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id) or not args.source_root or not args.scratch_root:
        raise ValueError("source-registration requires a safe run-id, existing source-root and compute-local scratch-root")
    root = resolve_repo_path(policy["output_root"]) / args.run_id
    if root.exists():
        raise FileExistsError("registration output exists; analyze it without rerunning source conditions")
    root.mkdir(parents=True)
    scratch = Path(args.scratch_root).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"], text=True).strip() in ("nfs", "nfs4"):
        raise ValueError("source build must use compute-local storage")
    try:
        args.source_root = str(Path(args.source_root).resolve())
        prepared = _stage_registration_source(Path(args.source_root), scratch, policy)
        prepared["model_config"] = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"]
        section, site, states, parameters = _load_registration_cell(config, args.source_model_id, scratch / "source", policy)
        if registration.recording_version(policy) == registration.LEDGER_VERSION:
            _validate_registration_step_interface(section, policy)
        del section
        schema = config["schemas"]["source_membrane_charge_registration"]
        declaration = {
            "schema_version": registration.recording_version(policy), "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy": policy, "units": schema["units"], "schema": schema, "site": site,
            "state_variables": states, "source_parameters": parameters,
            "source": {key: value for key, value in prepared.items() if key != "build"},
            "build": prepared["build"],
            "implementation": {path: _sha256(REPO_ROOT / path) for path in (
                "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py",
                "Libs/utils/evaluation/source_membrane_charge_registration.py")},
            "execution": {"node": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                          "step_id": os.environ.get("SLURM_STEP_ID"), "cpu_limit": 1, "memory_limit_gib": 2},
            "recording_semantics": {"electrode_nA": "endpoint assigned current after fcurrent", "interval_electrode_nA": "completed interval current before fcurrent", "ionic_densities": "post-state endpoint fcurrent; not retained solver-linearized currents"},
            "source_parameter_scope": "source-derived local parameters plus pinned source dependencies; not observed native factors",
        }
        if declaration["schema_version"] == registration.LEDGER_VERSION:
            declaration["step_current_contract"] = policy["source_step_ledger"]
        declaration["declaration_uid"] = registration.content_uid(declaration)
        registration.validate_declaration(declaration)
        registration.write_json(root / "declaration.json", declaration)
        shutil.copyfile(scratch / "build.log", root / "build.log")
        if (scratch / "resources.json").exists():
            shutil.copyfile(scratch / "resources.json", root / "resources.json")
        for condition in registration.condition_matrix(policy):
            registration.write_json(root / (condition["case_id"] + ".json"), _empty_registration_trace(declaration, condition))
        _dispatch_registration_cases(args, declaration, root, scratch)
        result = registration.analyze_directory(root)
        # This comparison is consumed by the result and detects forbidden changes
        # to the source originals as well as changes to the staged library.
        _verify_registration_inputs(declaration)
        for relative, expected in declaration["source"]["files"].items():
            if _sha256(Path(args.source_root) / relative) != expected:
                raise ValueError("original source changed during study")
        if "original_library" in declaration["source"]:
            original = Path(args.source_root) / declaration["source"]["original_library"]["relative_path"]
            current = _sha256(original) if original.exists() else None
            if current != declaration["source"]["original_library"]["sha256"]:
                raise ValueError("original compiled source library changed during study")
            result["original_compiled_library_preserved"] = True
        result["original_source_preserved"] = True
        registration.write_json(root / "result.json", result)
        print(json.dumps({"status": result["status"], "completed_cases": result["completed_cases"], "output": str(root)}))
    except Exception as error:
        registration.write_json(root / "operation_failure.json", {"error": f"{type(error).__name__}: {error}",
                                  "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID")})
        raise


def main() -> None:
    args = parse_args()
    if args.operation != "legacy":
        try:
            source_registration_operation(args)
        except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as error:
            print(f"source registration error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        return
    from Libs.utils.model.dynamics_fields import species_id_list
    from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
    from Libs.utils.model.microenvironment_teacher import json_payload, teacher_contract_version, write_microenvironment_zarr
    from Libs.utils.model.screening_core import stable_uid

    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    root = resolve_repo_path(registry["local_root"])
    relative_paths = registry["relative_paths"]
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    payload = run_protocol(config, args.source_model_id)
    sanity_check(config, payload)
    trajectory = protocol_to_trajectory(config, payload)
    uncertainty = np.full((1, len(species_id_list(config))), float(config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]["uncertainty_floor"]), dtype=np.float32)
    store_path = root / relative_paths["neuron_rxd_trajectory_store"]
    write_microenvironment_zarr(store_path, trajectory[None, ...], uncertainty, config, {"teacher_family": "neuron_rxd_microenvironment", "source_model_id": args.source_model_id})
    protocol_parameters = teacher_config["protocol_runtime"]
    rows = [
        {
            "neuron_rxd_protocol_uid": stable_uid("neuron_rxd_protocol", args.source_model_id),
            "contract_version": teacher_contract_version(config),
            "disease_program": teacher_config["disease_scope"],
            "source_model_id": args.source_model_id,
            "source_repository": teacher_config["source_repository"],
            "provenance_level": teacher_config["provenance_level"],
            "teacher_role": teacher_config["teacher_role"],
            "observed_or_synthetic": teacher_config["observed_or_synthetic"],
            "protocol_name": "ad_full_microenvironment_current_clamp_rxd",
            "protocol_parameter_payload": json_payload(protocol_parameters),
            "trajectory_store_path": str(store_path.resolve().relative_to(REPO_ROOT)),
            "trajectory_index": 0,
            "resting_voltage_millivolt": float(np.asarray(payload["voltage"], dtype=np.float32)[0]),
            "spike_count": float(payload["spike_count"]),
            "calcium_peak": float(np.max(np.asarray(payload["calcium"], dtype=np.float32))),
            "amyloid_beta_final": float(np.asarray(payload["amyloid"], dtype=np.float32)[-1]),
            "tau_final": float(np.asarray(payload["tau"], dtype=np.float32)[-1]),
            "inflammatory_mediator_final": float(np.asarray(payload["inflammatory"], dtype=np.float32)[-1]),
            "circuit_instability_final": float(trajectory[-1, :, :, species_id_list(config).index("circuit_instability")].mean()),
            "uncertainty_payload": json_payload({species_id: float(value) for species_id, value in zip(species_id_list(config), uncertainty[0].tolist())}),
            "note": "Repository-generated source-backed ModelDB NEURON/RxD full microenvironment teacher trajectory",
        }
    ]
    output_path = root / relative_paths["neuron_rxd_protocol_catalog"]
    write_tsv(output_path, config["schemas"]["neuron_rxd_protocol_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        neuron_rxd_protocol_row_count=len(rows),
        trajectory_store_path=store_path,
    )


if __name__ == "__main__":
    main()
