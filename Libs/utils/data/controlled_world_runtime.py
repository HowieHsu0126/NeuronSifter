"""Deterministic closed-dispatch runtime for three controlled neural worlds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import random

from Libs.utils.data.controlled_world_contracts import (
    FAMILY_MANIFEST_SCHEMA_VERSION,
    FAMILY_UIDS,
    INSTANCE_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    ControlledWorldContractError,
    GeneratorProvenance,
    WorldIdentity,
    action_sequence_identity,
    authoritative_family_manifest_row,
    bind_information_action,
    bind_world_action,
    classify_world_action,
    controlled_world_config,
    ensure_finite_payload,
    information_sequence_identity,
    select_family_config,
    validate_controlled_world_config,
    validate_family_manifest,
    validate_replay_manifest,
    validate_runtime_instance,
)
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter(family: Mapping[str, object], name: str, expected_unit: str) -> tuple[float, float]:
    row = dict(family["generator"]["parameters"][name])
    if row.get("unit") != expected_unit:
        raise ControlledWorldContractError("invalid_unit", "Generator parameter unit is not canonical.", field=name)
    value = float(row["value"])
    jitter = float(row.get("seed_jitter", 0.0))
    ensure_finite_payload({"value": value, "seed_jitter": jitter})
    if jitter < 0:
        raise ControlledWorldContractError("invalid_parameter", "Seed jitter cannot be negative.", field=name)
    return value, jitter


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def load_harness_oracle(
    config: Mapping[str, object],
    *,
    family_uid: str,
    reader_role: str,
) -> dict[str, object]:
    """Build family-specific hidden causal state only for the benchmark harness."""

    source = validate_controlled_world_config(config)
    family = select_family_config(source, family_uid)
    if reader_role != "benchmark_harness":
        raise ControlledWorldContractError(
            "oracle_access_forbidden",
            "Only the benchmark harness may load controlled-world hidden state.",
            reader_role=reader_role,
        )
    generator = dict(family["generator"])
    rng = random.Random(int(dict(generator["subseeds"])["parameter_jitter"]))

    def fixed(name: str, unit: str) -> float:
        return _parameter(family, name, unit)[0]

    def jittered(name: str, unit: str) -> float:
        center, width = _parameter(family, name, unit)
        return center + rng.uniform(-width, width)

    common = {
        "authority_role": "benchmark_harness",
        "family_uid": family_uid,
    }
    if family_uid == "electrophysiology_conductance_world":
        return {
            **common,
            "hidden_optimum": {
                "peak_spike_rate": {"value": 60.0, "unit": "Hz"},
                "terminal_membrane_voltage": {"value": -60.0, "unit": "mV"},
                "calcium_ceiling": {"value": 0.30, "unit": "umol/L"},
            },
            "oracle_state": {
                "initial_membrane_voltage": fixed("initial_membrane_voltage", "mV"),
                "initial_calcium_concentration": fixed("initial_calcium_concentration", "umol/L"),
                "initial_spike_rate": 0.0,
            },
            "oracle_parameters": {
                "resting_membrane_voltage": fixed("resting_membrane_voltage", "mV"),
                "membrane_leak_rate": jittered("membrane_leak_rate", "per_ms"),
                "stimulation_gain": jittered("stimulation_gain", "mV_per_uA_ms"),
                "calcium_decay_rate": jittered("calcium_decay_rate", "per_ms"),
                "spike_threshold": jittered("spike_threshold", "mV"),
                "spike_rate_gain": fixed("spike_rate_gain", "Hz_per_mV"),
                "spike_calcium_gain": fixed("spike_calcium_gain", "umol_L_per_Hz_ms"),
            },
        }
    if family_uid == "calcium_synaptic_world":
        return {
            **common,
            "hidden_optimum": {
                "peak_postsynaptic_current": {"value": 22.0, "unit": "pA"},
                "terminal_release_probability": {"value": 0.65, "unit": "probability"},
                "terminal_releasable_vesicle_fraction": {"value": 0.70, "unit": "fraction"},
                "calcium_ceiling": {"value": 0.30, "unit": "umol/L"},
            },
            "oracle_state": {
                "initial_intracellular_calcium": fixed("initial_intracellular_calcium", "umol/L"),
                "initial_release_probability": fixed("initial_release_probability", "probability"),
                "initial_releasable_vesicle_fraction": fixed("initial_releasable_vesicle_fraction", "fraction"),
                "initial_postsynaptic_current": fixed("initial_postsynaptic_current", "pA"),
            },
            "oracle_parameters": {
                "baseline_intracellular_calcium": fixed("baseline_intracellular_calcium", "umol/L"),
                "calcium_clearance_rate": jittered("calcium_clearance_rate", "per_ms"),
                "presynaptic_calcium_gain": jittered("presynaptic_calcium_gain", "umol_L_per_mW_ms"),
                "release_coupling": jittered("release_coupling", "per_umol_L"),
                "vesicle_release_rate": fixed("vesicle_release_rate", "per_ms"),
                "vesicle_recovery_rate": jittered("vesicle_recovery_rate", "per_ms"),
                "postsynaptic_current_gain": fixed("postsynaptic_current_gain", "pA_per_fraction"),
                "postsynaptic_decay_rate": jittered("postsynaptic_decay_rate", "per_ms"),
            },
        }
    if family_uid == "molecular_pharmacological_causal_world":
        return {
            **common,
            "hidden_optimum": {
                "terminal_target_occupancy": {"value": 0.65, "unit": "fraction"},
                "terminal_downstream_signal": {"value": 0.70, "unit": "normalized_signal"},
                "terminal_biologic_response": {"value": 0.60, "unit": "normalized_response"},
                "peak_free_agent_exposure_ceiling": {"value": 100.0, "unit": "nmol/L"},
            },
            "oracle_state": {
                "initial_free_agent_exposure": fixed("initial_free_agent_exposure", "nmol/L"),
                "initial_target_occupancy": fixed("initial_target_occupancy", "fraction"),
                "initial_downstream_signal": fixed("initial_downstream_signal", "normalized_signal"),
                "initial_biologic_response": fixed("initial_biologic_response", "normalized_response"),
            },
            "oracle_parameters": {
                "dose_to_exposure_gain": jittered("dose_to_exposure_gain", "nmol_L_per_mg_kg_hour"),
                "exposure_clearance_rate": jittered("exposure_clearance_rate", "per_hour"),
                "binding_affinity": jittered("binding_affinity", "nmol/L"),
                "occupancy_relaxation_rate": fixed("occupancy_relaxation_rate", "per_hour"),
                "signal_activation_rate": jittered("signal_activation_rate", "per_hour"),
                "signal_decay_rate": fixed("signal_decay_rate", "per_hour"),
                "biologic_response_rate": jittered("biologic_response_rate", "per_hour"),
                "biologic_clearance_rate": fixed("biologic_clearance_rate", "per_hour"),
                "feedback_rate": jittered("feedback_rate", "per_hour"),
            },
        }
    raise ControlledWorldContractError("unsupported_world_family", "World family is outside the closed dispatch.")


def _action_level(times: Sequence[float], index: int, delay_steps: int, duration: float, value: float) -> float:
    action_start = times[delay_steps]
    return value if index >= delay_steps and times[index] < action_start + duration else 0.0


def _target_value(oracle: Mapping[str, object], name: str, unit: str) -> float:
    targets = dict(oracle["hidden_optimum"])
    target = dict(targets[name])
    if target.get("unit") != unit:
        raise ControlledWorldContractError("invalid_unit", "Harness utility target unit is not canonical.", field=name)
    value = float(target["value"])
    ensure_finite_payload({name: value})
    return value


def _normalized_action_burden(family: Mapping[str, object], action_binding: Mapping[str, object]) -> float:
    extension = dict(action_binding["controlled_extension"])
    intensity = float(dict(extension["dose_or_intensity"])["value"])
    duration = float(dict(extension["duration"])["value"])
    support = dict(family["support_domain"])
    maximum_intensity = float(dict(support["supported_intensity"])["maximum"])
    maximum_duration = float(dict(support["supported_duration"])["maximum"])
    burden = (intensity / maximum_intensity) * (duration / maximum_duration)
    ensure_finite_payload({"normalized_action_burden": burden})
    return burden


def _electrophysiology_true_utility(
    family: Mapping[str, object],
    oracle: Mapping[str, object],
    action_binding: Mapping[str, object],
    trajectory: Sequence[Mapping[str, float]],
) -> float:
    delay = int(action_binding["response_delay_steps"])
    post_action = list(trajectory[delay:])
    if not post_action:
        raise ControlledWorldContractError("invalid_harness_utility", "Electrophysiology utility needs post-action trajectory rows.")
    target_spike_rate = _target_value(oracle, "peak_spike_rate", "Hz")
    target_voltage = _target_value(oracle, "terminal_membrane_voltage", "mV")
    calcium_ceiling = _target_value(oracle, "calcium_ceiling", "umol/L")
    peak_spike_rate = max(float(row["spike_rate"]) for row in post_action)
    terminal_voltage = float(post_action[-1]["membrane_voltage"])
    peak_calcium = max(float(row["calcium_concentration"]) for row in post_action)
    spike_error = abs(peak_spike_rate - target_spike_rate) / target_spike_rate
    voltage_error = abs(terminal_voltage - target_voltage) / 10.0
    calcium_excess = max(0.0, peak_calcium - calcium_ceiling) / calcium_ceiling
    return -(spike_error + 0.5 * voltage_error + calcium_excess + 0.1 * _normalized_action_burden(family, action_binding))


def _calcium_synaptic_true_utility(
    family: Mapping[str, object],
    oracle: Mapping[str, object],
    action_binding: Mapping[str, object],
    trajectory: Sequence[Mapping[str, float]],
) -> float:
    delay = int(action_binding["response_delay_steps"])
    post_action = list(trajectory[delay:])
    if not post_action:
        raise ControlledWorldContractError("invalid_harness_utility", "Calcium-synaptic utility needs post-action trajectory rows.")
    target_current = _target_value(oracle, "peak_postsynaptic_current", "pA")
    target_release = _target_value(oracle, "terminal_release_probability", "probability")
    target_vesicles = _target_value(oracle, "terminal_releasable_vesicle_fraction", "fraction")
    calcium_ceiling = _target_value(oracle, "calcium_ceiling", "umol/L")
    peak_current = max(float(row["postsynaptic_current"]) for row in post_action)
    terminal_release = float(post_action[-1]["release_probability"])
    terminal_vesicles = float(post_action[-1]["releasable_vesicle_fraction"])
    peak_calcium = max(float(row["intracellular_calcium"]) for row in post_action)
    current_error = abs(peak_current - target_current) / target_current
    release_error = abs(terminal_release - target_release) / target_release
    vesicle_error = abs(terminal_vesicles - target_vesicles) / target_vesicles
    calcium_excess = max(0.0, peak_calcium - calcium_ceiling) / calcium_ceiling
    return -(
        current_error
        + 0.5 * release_error
        + 0.5 * vesicle_error
        + calcium_excess
        + 0.1 * _normalized_action_burden(family, action_binding)
    )


def _molecular_pharmacological_true_utility(
    family: Mapping[str, object],
    oracle: Mapping[str, object],
    action_binding: Mapping[str, object],
    trajectory: Sequence[Mapping[str, float]],
) -> float:
    delay = int(action_binding["response_delay_steps"])
    post_action = list(trajectory[delay:])
    if not post_action:
        raise ControlledWorldContractError("invalid_harness_utility", "Molecular utility needs post-action trajectory rows.")
    target_occupancy = _target_value(oracle, "terminal_target_occupancy", "fraction")
    target_signal = _target_value(oracle, "terminal_downstream_signal", "normalized_signal")
    target_response = _target_value(oracle, "terminal_biologic_response", "normalized_response")
    exposure_ceiling = _target_value(oracle, "peak_free_agent_exposure_ceiling", "nmol/L")
    terminal = post_action[-1]
    peak_exposure = max(float(row["free_agent_exposure"]) for row in post_action)
    occupancy_error = abs(float(terminal["target_occupancy"]) - target_occupancy) / target_occupancy
    signal_error = abs(float(terminal["downstream_signal"]) - target_signal) / target_signal
    response_error = abs(float(terminal["biologic_response"]) - target_response) / target_response
    exposure_excess = max(0.0, peak_exposure - exposure_ceiling) / exposure_ceiling
    return -(
        occupancy_error
        + signal_error
        + response_error
        + exposure_excess
        + 0.1 * _normalized_action_burden(family, action_binding)
    )


def evaluate_harness_true_utility(
    family: Mapping[str, object],
    oracle: Mapping[str, object],
    action_binding: Mapping[str, object],
    trajectory: Sequence[Mapping[str, float]],
    *,
    reader_role: str,
) -> float:
    """Evaluate hidden family-specific utility from the executed action and causal trajectory."""

    if reader_role != "benchmark_harness" or oracle.get("authority_role") != "benchmark_harness":
        raise ControlledWorldContractError(
            "oracle_access_forbidden",
            "Only the benchmark harness may evaluate controlled-world true utility.",
            reader_role=reader_role,
        )
    family_uid = str(family["family_uid"])
    if oracle.get("family_uid") != family_uid:
        raise ControlledWorldContractError("oracle_family_mismatch", "Harness utility oracle and family identities differ.")
    binding_value = dict(action_binding)
    canonical_binding = bind_world_action(
        family,
        dict(binding_value["canonical_world_action"]),
        dict(binding_value["controlled_extension"]),
    ).to_dict()
    if binding_value != canonical_binding:
        raise ControlledWorldContractError(
            "action_mapping_mismatch",
            "Harness utility requires the authoritative bound action and delay.",
        )
    expected_trajectory = simulate_harness_trajectory(family, oracle, canonical_binding)
    observed_trajectory = [dict(row) for row in trajectory]
    if observed_trajectory != expected_trajectory:
        raise ControlledWorldContractError(
            "harness_trajectory_mismatch",
            "Harness utility requires the causal trajectory produced by the validated action and oracle state.",
            expected_identity=canonical_sha256(expected_trajectory),
            observed_identity=canonical_sha256(observed_trajectory),
        )
    if family_uid == "electrophysiology_conductance_world":
        utility = _electrophysiology_true_utility(family, oracle, canonical_binding, observed_trajectory)
    elif family_uid == "calcium_synaptic_world":
        utility = _calcium_synaptic_true_utility(family, oracle, canonical_binding, observed_trajectory)
    elif family_uid == "molecular_pharmacological_causal_world":
        utility = _molecular_pharmacological_true_utility(family, oracle, canonical_binding, observed_trajectory)
    else:
        raise ControlledWorldContractError("unsupported_world_family", "World family is outside the closed dispatch.")
    ensure_finite_payload({"true_utility": utility})
    return utility


def step_electrophysiology(values: tuple[float, ...], parameters: Mapping[str, object],
        step: float, stimulation: float) -> tuple[float, ...]:
    """One original source transition; caller owns clock and actuation binding."""
    voltage, calcium, spike_rate = values
    voltage += step * (
        -float(parameters["membrane_leak_rate"]) * (voltage - float(parameters["resting_membrane_voltage"]))
        + float(parameters["stimulation_gain"]) * stimulation
    )
    spike_rate = max(0.0, (voltage - float(parameters["spike_threshold"])) * float(parameters["spike_rate_gain"]))
    calcium += step * (
        -float(parameters["calcium_decay_rate"]) * (calcium - 0.10)
        + float(parameters["spike_calcium_gain"]) * spike_rate
    )
    return voltage, calcium, spike_rate


def step_calcium_synaptic(values: tuple[float, ...], parameters: Mapping[str, object],
        step: float, stimulation: float, *, initial_release_probability: float) -> tuple[float, ...]:
    """One original source transition; caller owns clock and actuation binding."""
    calcium, release_probability, vesicle_fraction, postsynaptic_current = values
    baseline = float(parameters["baseline_intracellular_calcium"])
    calcium += step * (
        -float(parameters["calcium_clearance_rate"]) * (calcium - baseline)
        + float(parameters["presynaptic_calcium_gain"]) * stimulation
    )
    release_probability = _clamp(
        float(initial_release_probability)
        + float(parameters["release_coupling"]) * max(0.0, calcium - baseline)
    )
    released_fraction = min(
        vesicle_fraction,
        step * float(parameters["vesicle_release_rate"]) * release_probability * vesicle_fraction,
    )
    vesicle_fraction = _clamp(
        vesicle_fraction
        + step * float(parameters["vesicle_recovery_rate"]) * (1.0 - vesicle_fraction)
        - released_fraction
    )
    postsynaptic_current += (
        float(parameters["postsynaptic_current_gain"]) * released_fraction
        - step * float(parameters["postsynaptic_decay_rate"]) * postsynaptic_current
    )
    return calcium, release_probability, vesicle_fraction, postsynaptic_current


def step_molecular_pharmacological(values: tuple[float, ...], parameters: Mapping[str, object],
        step: float, active_dose: float) -> tuple[float, ...]:
    """One original source transition; caller owns clock and actuation binding."""
    exposure, occupancy, signal, response = values
    exposure += step * (
        float(parameters["dose_to_exposure_gain"]) * active_dose
        - float(parameters["exposure_clearance_rate"]) * exposure
        - float(parameters["feedback_rate"]) * response * exposure
    )
    exposure = max(0.0, exposure)
    equilibrium_occupancy = exposure / (exposure + float(parameters["binding_affinity"])) if exposure > 0 else 0.0
    occupancy = _clamp(
        occupancy + step * float(parameters["occupancy_relaxation_rate"]) * (equilibrium_occupancy - occupancy)
    )
    signal = _clamp(
        signal + step * (
            float(parameters["signal_activation_rate"]) * occupancy * (1.0 - signal)
            - float(parameters["signal_decay_rate"]) * signal
        )
    )
    response = _clamp(
        response + step * (
            float(parameters["biologic_response_rate"]) * signal * (1.0 - response)
            - float(parameters["biologic_clearance_rate"]) * response
        )
    )
    return exposure, occupancy, signal, response


def _simulate_electrophysiology(
    family: Mapping[str, object], oracle: Mapping[str, object], action_binding: Mapping[str, object]
) -> list[dict[str, float]]:
    times = [float(value) for value in family["world"]["time_grid"]["values"]]
    state = dict(oracle["oracle_state"])
    parameters = dict(oracle["oracle_parameters"])
    extension = dict(action_binding["controlled_extension"])
    intensity = float(extension["dose_or_intensity"]["value"])
    duration = float(extension["duration"]["value"])
    delay = int(action_binding["response_delay_steps"])
    voltage = float(state["initial_membrane_voltage"])
    calcium = float(state["initial_calcium_concentration"])
    spike_rate = float(state["initial_spike_rate"])
    trajectory = [{"time": times[0], "membrane_voltage": voltage, "calcium_concentration": calcium, "spike_rate": spike_rate}]
    for index in range(1, len(times)):
        step = times[index] - times[index - 1]
        stimulation = _action_level(times, index, delay, duration, intensity)
        voltage, calcium, spike_rate = step_electrophysiology(
            (voltage, calcium, spike_rate), parameters, step, stimulation)
        row = {"time": times[index], "membrane_voltage": voltage, "calcium_concentration": calcium, "spike_rate": spike_rate}
        ensure_finite_payload(row)
        trajectory.append(row)
    return trajectory


def _simulate_calcium_synaptic(
    family: Mapping[str, object], oracle: Mapping[str, object], action_binding: Mapping[str, object]
) -> list[dict[str, float]]:
    times = [float(value) for value in family["world"]["time_grid"]["values"]]
    state = dict(oracle["oracle_state"])
    parameters = dict(oracle["oracle_parameters"])
    extension = dict(action_binding["controlled_extension"])
    intensity = float(extension["dose_or_intensity"]["value"])
    duration = float(extension["duration"]["value"])
    delay = int(action_binding["response_delay_steps"])
    calcium = float(state["initial_intracellular_calcium"])
    release_probability = float(state["initial_release_probability"])
    vesicle_fraction = float(state["initial_releasable_vesicle_fraction"])
    postsynaptic_current = float(state["initial_postsynaptic_current"])
    trajectory = [{
        "time": times[0],
        "intracellular_calcium": calcium,
        "release_probability": release_probability,
        "releasable_vesicle_fraction": vesicle_fraction,
        "postsynaptic_current": postsynaptic_current,
    }]
    for index in range(1, len(times)):
        step = times[index] - times[index - 1]
        stimulation = _action_level(times, index, delay, duration, intensity)
        calcium, release_probability, vesicle_fraction, postsynaptic_current = step_calcium_synaptic(
            (calcium, release_probability, vesicle_fraction, postsynaptic_current), parameters, step, stimulation, initial_release_probability=float(state["initial_release_probability"]))
        row = {
            "time": times[index],
            "intracellular_calcium": calcium,
            "release_probability": release_probability,
            "releasable_vesicle_fraction": vesicle_fraction,
            "postsynaptic_current": postsynaptic_current,
        }
        ensure_finite_payload(row)
        trajectory.append(row)
    return trajectory


def _simulate_molecular_pharmacological(
    family: Mapping[str, object], oracle: Mapping[str, object], action_binding: Mapping[str, object]
) -> list[dict[str, float]]:
    times = [float(value) for value in family["world"]["time_grid"]["values"]]
    state = dict(oracle["oracle_state"])
    parameters = dict(oracle["oracle_parameters"])
    extension = dict(action_binding["controlled_extension"])
    dose = float(extension["dose_or_intensity"]["value"])
    duration = float(extension["duration"]["value"])
    delay = int(action_binding["response_delay_steps"])
    exposure = float(state["initial_free_agent_exposure"])
    occupancy = float(state["initial_target_occupancy"])
    signal = float(state["initial_downstream_signal"])
    response = float(state["initial_biologic_response"])
    trajectory = [{
        "time": times[0],
        "free_agent_exposure": exposure,
        "target_occupancy": occupancy,
        "downstream_signal": signal,
        "biologic_response": response,
    }]
    for index in range(1, len(times)):
        step = times[index] - times[index - 1]
        active_dose = _action_level(times, index, delay, duration, dose)
        exposure, occupancy, signal, response = step_molecular_pharmacological(
            (exposure, occupancy, signal, response), parameters, step, active_dose)
        row = {
            "time": times[index],
            "free_agent_exposure": exposure,
            "target_occupancy": occupancy,
            "downstream_signal": signal,
            "biologic_response": response,
        }
        ensure_finite_payload(row)
        trajectory.append(row)
    return trajectory


def simulate_harness_trajectory(
    family: Mapping[str, object], oracle: Mapping[str, object], action_binding: Mapping[str, object]
) -> list[dict[str, float]]:
    family_uid = str(family["family_uid"])
    if family_uid == "electrophysiology_conductance_world":
        return _simulate_electrophysiology(family, oracle, action_binding)
    if family_uid == "calcium_synaptic_world":
        return _simulate_calcium_synaptic(family, oracle, action_binding)
    if family_uid == "molecular_pharmacological_causal_world":
        return _simulate_molecular_pharmacological(family, oracle, action_binding)
    raise ControlledWorldContractError("unsupported_world_family", "World family is outside the closed dispatch.")


def _observation_view(family: Mapping[str, object], trajectory: Sequence[Mapping[str, float]]) -> list[dict[str, object]]:
    units = {str(row["variable"]): str(row["unit"]) for row in family["world"]["observable_variables"]}
    policy = dict(family["world"]["observation_policy"])
    observed_indices = {name: set(indices) for name, indices in policy["observed_indices"].items()}
    noise_scale = {name: float(row["value"]) for name, row in policy["noise_scale"].items()}
    rng = random.Random(int(family["generator"]["subseeds"]["observation_noise"]))
    observations: list[dict[str, object]] = []
    for index, row in enumerate(trajectory):
        measurements: dict[str, object] = {}
        for variable, unit in units.items():
            if index not in observed_indices[variable]:
                continue
            value = float(row[variable]) + rng.uniform(-noise_scale[variable], noise_scale[variable])
            measurements[variable] = {"value": value, "unit": unit}
        observations.append(
            {
                "time": {"value": float(row["time"]), "unit": str(family["world"]["time_grid"]["unit"])},
                "measurements": measurements,
                "missing_observables": sorted(set(units) - set(measurements)),
            }
        )
    ensure_finite_payload(observations)
    return observations


def _information_results(
    family: Mapping[str, object], trajectory: Sequence[Mapping[str, float]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    units = {str(row["variable"]): str(row["unit"]) for row in family["world"]["observable_variables"]}
    results: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    support_ref = str(family["support_domain"]["support_domain_uid"])
    for raw_spec in family["information_actions"]:
        binding = bind_information_action(family, dict(raw_spec)).to_dict()
        bindings.append(binding)
        action = dict(binding["canonical_information_action"])
        outcome = dict(binding["outcome_policy"])
        time_index = int(dict(action["measurement_context"])["time_index"])
        measurements: dict[str, object] = {}
        if outcome["status"] == "partial":
            offsets = dict(outcome["noise_offsets"])
            for variable in outcome["measured_observables"]:
                name = str(variable)
                measurements[name] = {
                    "value": float(trajectory[time_index][name]) + float(offsets.get(name, 0.0)),
                    "unit": units[name],
                }
        results.append(
            {
                "information_action_uid": action["information_action_uid"],
                "status": outcome["status"],
                "cost": dict(action["expected_cost"]),
                "measurements": measurements,
                "missing_observables": list(outcome["missing_observables"]),
                "failure_code": outcome["failure_code"],
                "support_status": {"status": "supported", "reason_codes": [], "support_domain_ref": support_ref},
                "provenance_ref": action["provenance_ref"],
            }
        )
    ensure_finite_payload(results)
    return results, bindings


def _generate_runtime_instance_unvalidated(
    config: Mapping[str, object],
    *,
    family_uid: str,
    repository_root: Path,
    config_file_identity: str,
    canonical_action: Mapping[str, object] | None = None,
    action_extension: Mapping[str, object] | None = None,
    force_execute: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source = validate_controlled_world_config(config)
    family = select_family_config(source, family_uid)
    action_binding = bind_world_action(family, canonical_action, action_extension)
    support = classify_world_action(family, action_binding.canonical_world_action, action_binding.controlled_extension)
    if support.status == "unsupported":
        raise ControlledWorldContractError("unsupported_action", "The requested intervention is unsupported.", reasons=list(support.reason_codes))
    if support.status == "out_of_distribution":
        code = "forced_ood_execution" if force_execute else "out_of_distribution_action"
        raise ControlledWorldContractError(code, "Out-of-distribution actions cannot execute.", reasons=list(support.reason_codes))
    oracle = load_harness_oracle(source, family_uid=family_uid, reader_role="benchmark_harness")
    trajectory = simulate_harness_trajectory(family, oracle, action_binding.to_dict())
    evaluate_harness_true_utility(
        family,
        oracle,
        action_binding.to_dict(),
        trajectory,
        reader_role="benchmark_harness",
    )
    observations = _observation_view(family, trajectory)
    information_results, information_bindings = _information_results(family, trajectory)
    generator = dict(family["generator"])
    implementation_path = repository_root / str(generator["implementation_path"])
    if not implementation_path.is_file():
        raise ControlledWorldContractError("generator_source_missing", "Generator implementation path does not exist.")
    config_content_identity = canonical_sha256(family)
    seed_identity = canonical_sha256(
        {
            "family_uid": family_uid,
            "world_uid": family["world"]["world_uid"],
            "seed": generator["seed"],
            "seed_policy": generator["seed_policy"],
            "subseeds": generator["subseeds"],
        }
    )
    provenance = GeneratorProvenance(
        implementation_uid=str(generator["implementation_uid"]),
        implementation_version=str(generator["implementation_version"]),
        implementation_path=str(generator["implementation_path"]),
        code_identity=sha256_file(implementation_path),
        config_content_identity=config_content_identity,
        config_file_identity=config_file_identity,
        seed=int(generator["seed"]),
        seed_identity=seed_identity,
    )
    identity = WorldIdentity(
        world_uid=str(family["world"]["world_uid"]),
        world_family_uid=family_uid,
        latent_generator_lineage_uid=str(generator["latent_generator_lineage_uid"]),
        biological_context=dict(family["world"]["biological_context"]),
    )
    total_cost = sum(float(dict(row["cost"])["value"]) for row in information_results)
    instance = {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "authority_role": "runtime_observable",
        "evidence_class": "controlled_world_non_empirical",
        "world_identity": identity.to_dict(),
        "generator_provenance": provenance.to_dict(),
        "action_binding": action_binding.to_dict(),
        "support_status": support.to_dict(),
        "observations": observations,
        "information_results": information_results,
        "information_cost_total": {"value": total_cost, "unit": "cost_unit"},
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0},
        "claim_boundary": dict(source["claim_boundary"]),
    }
    return instance, information_bindings


def generate_runtime_instance(
    config: Mapping[str, object],
    *,
    family_uid: str,
    repository_root: Path,
    config_file_identity: str,
    canonical_action: Mapping[str, object] | None = None,
    action_extension: Mapping[str, object] | None = None,
    force_execute: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    instance, bindings = _generate_runtime_instance_unvalidated(
        config,
        family_uid=family_uid,
        repository_root=repository_root,
        config_file_identity=config_file_identity,
        canonical_action=canonical_action,
        action_extension=action_extension,
        force_execute=force_execute,
    )
    if canonical_action is None and action_extension is None:
        validate_runtime_instance(
            instance,
            config=config,
            repository_root=repository_root,
            config_file_identity=config_file_identity,
        )
    else:
        validate_runtime_instance(instance)
    return instance, bindings


def build_replay_manifest(
    config: Mapping[str, object],
    family_uid: str,
    instance: Mapping[str, object],
    information_bindings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source = controlled_world_config(config)
    family = select_family_config(source, family_uid)
    provenance = dict(instance["generator_provenance"])
    identity = dict(instance["world_identity"])
    binding = dict(instance["action_binding"])
    output_identity = canonical_sha256(instance)
    manifest_basis = {
        "world_uid": identity["world_uid"],
        "generator_code_identity": provenance["code_identity"],
        "config_content_identity": provenance["config_content_identity"],
        "seed_identity": provenance["seed_identity"],
        "action_sequence_identity": action_sequence_identity(binding),
        "information_sequence_identity": information_sequence_identity(information_bindings),
        "observable_output_identity": output_identity,
    }
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_uid": stable_uid("controlled_world_replay", manifest_basis),
        "world_uid": identity["world_uid"],
        "world_family_uid": identity["world_family_uid"],
        "latent_generator_lineage_uid": identity["latent_generator_lineage_uid"],
        "generator_implementation_uid": provenance["implementation_uid"],
        "generator_implementation_version": provenance["implementation_version"],
        "generator_code_identity": provenance["code_identity"],
        "config_content_identity": provenance["config_content_identity"],
        "config_file_identity": provenance["config_file_identity"],
        "seed": provenance["seed"],
        "seed_identity": provenance["seed_identity"],
        "action_sequence_identity": manifest_basis["action_sequence_identity"],
        "information_sequence_identity": manifest_basis["information_sequence_identity"],
        "observable_output_identity": output_identity,
        "input_lineage": [
            {"role": "generator_implementation", "artifact_ref": provenance["implementation_path"], "content_identity": provenance["code_identity"]},
            {"role": "controlled_world_config", "artifact_ref": "Libs/configs/data/controlled_neural_world.yaml", "content_identity": provenance["config_file_identity"]},
            {"role": "family_seed_policy", "artifact_ref": f"controlled_neural_world.families.{family_uid}.generator.seed", "content_identity": provenance["seed_identity"]},
        ],
        "output_lineage": [
            {"role": "runtime_observable", "artifact_ref": family["output_paths"]["world_instance"], "content_identity": output_identity}
        ],
        "access_counters": dict(instance["access_counters"]),
        "deterministic_reproduction": {"required": True, "byte_equivalent_canonical_json": True, "replay_count": 1, "status": "replay_identity_frozen"},
    }


def generate_runtime_and_manifest(
    config: Mapping[str, object],
    *,
    family_uid: str,
    repository_root: Path,
    config_file_identity: str,
) -> tuple[dict[str, object], dict[str, object]]:
    instance, bindings = generate_runtime_instance(
        config,
        family_uid=family_uid,
        repository_root=repository_root,
        config_file_identity=config_file_identity,
    )
    manifest = build_replay_manifest(config, family_uid, instance, bindings)
    validate_replay_manifest(manifest)
    return instance, manifest


def generate_all_family_packages(
    config: Mapping[str, object],
    *,
    repository_root: Path,
    config_file_identity: str,
    family_uids: Sequence[str] = FAMILY_UIDS,
) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    source = validate_controlled_world_config(config)
    requested = list(family_uids)
    if len(requested) != 3 or set(requested) != set(FAMILY_UIDS):
        raise ControlledWorldContractError("unsupported_world_family", "Generation requires exactly the closed three-family set.")
    packages: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for family_uid in requested:
        packages[family_uid] = generate_runtime_and_manifest(
            source,
            family_uid=family_uid,
            repository_root=repository_root,
            config_file_identity=config_file_identity,
        )
    return packages


def build_family_manifest(
    config: Mapping[str, object],
    packages: Mapping[str, tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    config_file_identity: str,
) -> dict[str, object]:
    source = validate_controlled_world_config(config)
    if set(packages) != set(FAMILY_UIDS):
        raise ControlledWorldContractError("family_manifest_incomplete", "All three family packages are required.")
    family_rows: list[dict[str, object]] = []
    for family_uid in FAMILY_UIDS:
        family = select_family_config(source, family_uid)
        instance, replay = packages[family_uid]
        validate_runtime_instance(instance, config=source, config_file_identity=config_file_identity)
        validate_replay_manifest(replay)
        family_rows.append(
            authoritative_family_manifest_row(
                family,
                instance,
                replay,
                config_file_identity=config_file_identity,
            )
        )
    basis = {"source_config_identity": config_file_identity, "families": family_rows}
    manifest_spec = dict(source["family_manifest"])
    manifest = {
        "schema_version": FAMILY_MANIFEST_SCHEMA_VERSION,
        "manifest_uid": stable_uid("controlled_world_family_manifest", basis),
        "evidence_class": "controlled_world_non_empirical",
        "family_count": 3,
        "family_uids": list(FAMILY_UIDS),
        "source_config_ref": "Libs/configs/data/controlled_neural_world.yaml",
        "source_config_identity": config_file_identity,
        "families": family_rows,
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0},
        "real_world_formal_agent_evaluation": False,
        "claim_boundary": dict(source["claim_boundary"]),
        "benchmark_split_assignment": manifest_spec["benchmark_split_assignment"],
        "sealed_test_partitions": manifest_spec["sealed_test_partitions"],
        "information_headroom_qualification": manifest_spec["information_headroom_qualification"],
        "benchmark_qualification": manifest_spec["benchmark_qualification"],
    }
    validate_family_manifest(
        manifest,
        config=source,
        config_file_identity=config_file_identity,
        authoritative_packages=packages,
    )
    return manifest


def replay_runtime_instance(
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    repository_root: Path,
) -> dict[str, object]:
    source = validate_controlled_world_config(config)
    manifest_value = validate_replay_manifest(manifest)
    family_uid = str(manifest_value["world_family_uid"])
    config_path = repository_root / "Libs/configs/data/controlled_neural_world.yaml"
    if not config_path.is_file():
        raise ControlledWorldContractError("config_source_missing", "Controlled-world source config is missing.")
    instance, bindings = generate_runtime_instance(
        source,
        family_uid=family_uid,
        repository_root=repository_root,
        config_file_identity=sha256_file(config_path),
    )
    expected = build_replay_manifest(source, family_uid, instance, bindings)
    if manifest_value != expected:
        mismatches = sorted(field for field in set(manifest_value) | set(expected) if manifest_value.get(field) != expected.get(field))
        raise ControlledWorldContractError(
            "replay_identity_mismatch",
            "Replay manifest does not exactly match the authoritative regenerated manifest.",
            fields=mismatches,
            expected_replay_uid=expected["replay_uid"],
            observed_replay_uid=manifest_value.get("replay_uid"),
        )
    if canonical_sha256(instance) != str(manifest_value["observable_output_identity"]):
        raise ControlledWorldContractError("replay_output_mismatch", "Replay output identity changed.")
    return instance
