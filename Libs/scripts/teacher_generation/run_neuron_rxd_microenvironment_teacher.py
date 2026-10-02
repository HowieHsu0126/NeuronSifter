from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from neuron import h, rxd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_fields import compartment_id_list, region_id_list, species_id_list, time_grid_hours
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.microenvironment_teacher import json_payload, teacher_contract_version, write_microenvironment_zarr
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_tsv
from Libs.utils.model.screening_core import stable_uid


_LOADED_COMPILED_LIBRARIES: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NEURON/RxD AD-first full microenvironment teacher rollout.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--source-model-id", default="modeldb_romani_2013_ca3_ca1")
    return parser.parse_args()


def reset_neuron() -> None:
    h("forall delete_section()")


def load_modeldb_source_cell(config: dict, source_model_id: str):
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    modeldb_config = teacher_config["model_sources"]["modeldb"]
    if source_model_id != str(modeldb_config["source_model_id"]):
        raise ValueError(f"Unsupported ModelDB source_model_id {source_model_id}; expected {modeldb_config['source_model_id']}.")
    mechanism_root = resolve_repo_path(str(modeldb_config["mechanism_root"]))
    simulation_hoc_path = resolve_repo_path(str(modeldb_config["simulation_hoc_path"]))
    compiled_library = mechanism_root / str(modeldb_config["compiled_library_relpath"])
    required_paths = [mechanism_root, simulation_hoc_path, compiled_library]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"ModelDB NEURON source is not compiled/materialized: {missing_paths}")
    reset_neuron()
    compiled_library_key = str(compiled_library.resolve())
    if compiled_library_key not in _LOADED_COMPILED_LIBRARIES:
        h.nrn_load_dll(compiled_library_key)
        _LOADED_COMPILED_LIBRARIES.add(compiled_library_key)
    h.chdir(str(mechanism_root))
    h.load_file("stdrun.hoc")
    if int(h.load_file(str(simulation_hoc_path))) != 1:
        raise RuntimeError(f"NEURON failed to load source-backed ModelDB HOC file: {simulation_hoc_path}")
    sections = list(h.allsec())
    if not sections:
        raise RuntimeError(f"ModelDB source {source_model_id} did not create any NEURON sections.")
    soma_sections = [section for section in sections if "soma" in section.name().lower()]
    if not soma_sections:
        raise RuntimeError(f"ModelDB source {source_model_id} did not expose a soma section for AD microenvironment recording.")
    return soma_sections[0]


def configure_rxd(soma, runtime: dict[str, object]):
    cytosol = rxd.Region([soma], name="cytosol", nrn_region="i")
    calcium = rxd.Species(cytosol, name="calcium", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["calcium_initial_millimolar"]))
    amyloid = rxd.Species(cytosol, name="amyloid_beta", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["amyloid_initial_millimolar"]))
    tau = rxd.Species(cytosol, name="tau", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["tau_initial_millimolar"]))
    inflammatory = rxd.Species(cytosol, name="inflammatory_mediator", d=float(runtime["rxd_diffusion_um2_per_ms"]), initial=float(runtime["inflammatory_initial_millimolar"]))
    rxd.Rate(calcium, -float(runtime["rxd_decay_per_ms"]) * calcium)
    rxd.Rate(tau, float(runtime["amyloid_tau_coupling_per_ms"]) * amyloid - float(runtime["rxd_decay_per_ms"]) * tau)
    rxd.Rate(inflammatory, float(runtime["inflammation_amyloid_coupling_per_ms"]) * amyloid - float(runtime["rxd_decay_per_ms"]) * inflammatory)
    return calcium, amyloid, tau, inflammatory


def _run_loaded_protocol(
    config: dict,
    soma,
    calcium,
    amyloid,
    tau,
    inflammatory,
    runtime: dict[str, object],
) -> dict[str, np.ndarray | float]:
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
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


def run_protocol(
    config: dict,
    source_model_id: str,
    protocol_runtime: dict[str, object] | None = None,
) -> dict[str, np.ndarray | float]:
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    runtime = dict(teacher_config["protocol_runtime"])
    if protocol_runtime:
        runtime.update(protocol_runtime)
    soma = load_modeldb_source_cell(config, source_model_id)
    calcium, amyloid, tau, inflammatory = configure_rxd(soma, runtime)
    return _run_loaded_protocol(config, soma, calcium, amyloid, tau, inflammatory, runtime)


def run_protocol_batch(
    config: dict,
    source_model_id: str,
    protocol_runtimes: list[dict[str, object]],
) -> list[dict[str, np.ndarray | float]]:
    """Run multiple action programs against one loaded source model.

    Reinitializing NEURON between protocols preserves the matched initial state
    while avoiding a second HOC load, which the ModelDB source does not support
    reliably in one process.
    """
    if not protocol_runtimes:
        raise ValueError("protocol_runtimes must not be empty")
    teacher_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    base_runtime = dict(teacher_config["protocol_runtime"])
    soma = load_modeldb_source_cell(config, source_model_id)
    calcium, amyloid, tau, inflammatory = configure_rxd(soma, base_runtime)
    payloads = []
    for protocol_runtime in protocol_runtimes:
        runtime = dict(base_runtime)
        runtime.update(protocol_runtime)
        payloads.append(_run_loaded_protocol(config, soma, calcium, amyloid, tau, inflammatory, runtime))
    return payloads


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


def main() -> None:
    args = parse_args()
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
