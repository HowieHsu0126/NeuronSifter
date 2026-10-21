"""Consumed, source-only membrane charge closure diagnostics; no native inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "source-membrane-charge-registration-v1"
ION_COLUMNS = ("ina_mA_cm2", "ik_mA_cm2", "ica_mA_cm2", "iCa_mA_cm2", "i_pas_mA_cm2")
CURRENT_ORIGINS = {
    "ionic": "source_ion_aggregates_plus_pas",
    "axial": "source_endpoint_voltage_over_ri",
}


def content_uid(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def capacitance_nf(cm_uf_cm2: float, area_um2: float) -> float:
    if not np.isfinite([cm_uf_cm2, area_um2]).all() or min(cm_uf_cm2, area_um2) <= 0:
        raise ValueError("segment area and specific capacitance must be finite and positive")
    return 1.0e-5 * cm_uf_cm2 * area_um2


def condition_matrix(policy: dict[str, Any]) -> list[dict[str, Any]]:
    steps = policy["dt_ms"]
    if len(steps) != 2 or not np.isfinite(steps).all() or steps[0] <= 0 or steps[1] != steps[0] / 2:
        raise ValueError("exactly one base step and its single half-step refinement are required")
    return [
        {"case_id": f"{history}_{continuation}_dt{str(dt).replace('.', 'p')}",
         "prehistory": history, "continuation": continuation, "dt_ms": dt}
        for history in ("rest", "conditioned")
        for continuation in ("control", "test")
        for dt in steps
    ]


def validate_policy(policy: dict[str, Any]) -> None:
    condition_matrix(policy)
    validate_ledger_contract(policy)
    if policy["candidate"] != "Gamma=(Q,C,site); J=F(Q,C) at fixed source parameters":
        raise ValueError("unrecognized prospective charge-only candidate")
    if policy["current_origins"] != CURRENT_ORIGINS:
        raise ValueError("independent source current accounting required; circular u-C*dV/dt is inadmissible")
    positive = ("duration_ms", "continuation_start_ms", "stimulus_amplitude_nA",
                "refinement_envelope_factor", "current_atol_nA", "balance_atol_pC",
                "balance_relative_tolerance")
    if any(not np.isfinite(policy[k]) or policy[k] <= 0 for k in positive):
        raise ValueError("invalid positive registration policy value")
    if not np.isfinite(policy["initial_voltage_mV"]) or policy["matching_charge_points"] < 2:
        raise ValueError("invalid initialization or matching count")
    if not 0 < policy["segment_x"] < 1 or policy["secondorder"] != 0:
        raise ValueError("unsupported segment location or integration order")
    if policy["solver_threads"] != 1 or policy["cvode_active"]:
        raise ValueError("source challenge requires serial fixed-step execution")
    starts = [phase["start_ms"] for phase in policy["matching_phases"]]
    if starts != sorted(starts):
        raise ValueError("matching phases must be chronological")
    for phase in policy["matching_phases"]:
        if not policy["continuation_start_ms"] <= phase["start_ms"] < phase["end_ms"] <= policy["duration_ms"]:
            raise ValueError("matching phase outside continuation")
    for pulse in ("prehistory_pulse_ms", "test_pulse_ms"):
        a, b = policy[pulse]
        if not 0 <= a < b <= policy["duration_ms"]:
            raise ValueError("invalid pulse")


def validate_declaration(declaration: dict[str, Any]) -> None:
    schema = declaration["schema"]
    if declaration["schema_version"] != recording_version(declaration["policy"]):
        raise ValueError("declaration version mismatch")
    for key in schema["declaration_required_fields"]:
        if key not in declaration:
            raise ValueError(f"declaration missing {key}")
    validate_policy(declaration["policy"])
    validate_ledger_contract(declaration["policy"], schema, declaration["build"]["identity"])
    if declaration["schema_version"] == LEDGER_VERSION:
        if declaration.get("step_current_contract") != declaration["policy"]["source_step_ledger"]:
            raise ValueError("selected solver-step mode requires its consumed ledger contract before advance")
    copy = dict(declaration)
    uid = copy.pop("declaration_uid")
    if content_uid(copy) != uid:
        raise ValueError("declaration identity mismatch")
    if declaration["units"] != schema["units"]:
        raise ValueError("declaration unit mismatch")
    site = declaration["site"]
    expected = capacitance_nf(site["cm_uF_cm2"], site["area_um2"])
    if site["capacitance_nF"] != expected:
        raise ValueError("capacitance identity mismatch")
    if site["section_name"] != declaration["policy"]["section_name"] or site["segment_x"] != declaration["policy"]["segment_x"]:
        raise ValueError("site/policy mismatch")
    for key in ("left_ri_MOhm", "right_ri_MOhm"):
        if not np.isfinite(site[key]) or site[key] <= 0:
            raise ValueError("invalid source axial resistance")


def validate_trace(trace: dict[str, Any], declaration: dict[str, Any]) -> None:
    for key in declaration["schema"]["trace_required_fields"]:
        if key not in trace:
            raise ValueError(f"trace missing {key}")
    if trace["schema_version"] != declaration["schema_version"] or trace["declaration_uid"] != declaration["declaration_uid"]:
        raise ValueError("trace declaration/source binding mismatch")
    for key in ("units", "site", "build_identity"):
        expected = declaration["build"]["identity"] if key == "build_identity" else declaration[key]
        if trace[key] != expected:
            raise ValueError(f"trace {key} mismatch")
    expected = {x["case_id"]: x for x in condition_matrix(declaration["policy"])}
    if trace["condition"] != expected.get(trace["condition"].get("case_id")):
        raise ValueError("condition not in the prospective eight-run matrix")
    status = trace["status"]
    if status not in ("COMPLETE", "FAILED", "NOT_RUN", "RUNNING", "DISPATCHED", "MISSING"):
        raise ValueError("unfinished or unknown source case")
    # Validate all records before the incomplete return or authoritative aggregation.
    attempt = trace["attempt_started"]
    advances = trace["source_time_advanced_count"]
    if attempt is not None and type(attempt) is not bool:
        raise ValueError("attempt_started must be boolean or unknown")
    if advances is not None and (type(advances) is not int or advances < 0):
        raise ValueError("source advance count must be a nonnegative integer or unknown")
    planned_advances = round(declaration["policy"]["duration_ms"] / trace["condition"]["dt_ms"])
    if advances is not None and advances > planned_advances:
        raise ValueError("source advance count exceeds the declared interval count")
    if attempt is not True and advances not in (None, 0):
        raise ValueError("positive source advances require attempt_started=True")
    if attempt is False and advances != 0:
        raise ValueError("attempt_started=False requires zero source advances")
    if status == "NOT_RUN" and attempt is not False:
        raise ValueError("NOT_RUN requires attempt_started=False and zero source advances")
    if status in ("DISPATCHED", "RUNNING", "COMPLETE") and attempt is not True:
        raise ValueError(f"{status} requires attempt_started=True")
    if status == "DISPATCHED" and advances != 0:
        raise ValueError("DISPATCHED requires zero source advances before child start")
    if status == "COMPLETE" and advances != planned_advances:
        raise ValueError("COMPLETE requires the exact known source advance count")
    if status != "COMPLETE":
        validate_step_ledger(trace, declaration)
        return
    channels = ["time_ms", "voltage_mV", "electrode_nA", "left_voltage_mV",
                "right_voltage_mV", *ION_COLUMNS, *declaration["state_variables"]]
    data = trace["samples"]
    for optional in declaration["schema"].get("optional_sample_columns", []):
        if optional in data:
            channels.append(optional)
    if set(data) != set(channels):
        raise ValueError("recorded channel schema mismatch")
    n = len(data["time_ms"])
    if n < 2 or len(trace["observed_mask"]) != n or not all(trace["observed_mask"]):
        raise ValueError("complete trace requires explicitly observed finite samples")
    for name in channels:
        a = np.asarray(data[name], dtype=np.float64)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"nonfinite or malformed channel {name}")
    t = np.asarray(data["time_ms"])
    dt = trace["condition"]["dt_ms"]
    duration = declaration["policy"]["duration_ms"]
    if n != round(duration / dt) + 1 or not np.allclose(np.diff(t), dt, rtol=0, atol=1e-8):
        raise ValueError("source clock/cardinality mismatch")
    if abs(t[0]) > 1e-8 or abs(t[-1] - duration) > 1e-7:
        raise ValueError("incomplete source clock")
    if abs(data["voltage_mV"][0] - declaration["policy"]["initial_voltage_mV"]) > 1e-8:
        raise ValueError("initial voltage differs from prospective source initialization")
    expected_u = np.zeros(n)
    safe = np.ones(n, dtype=bool)
    policy = declaration["policy"]
    for active, pulse in ((trace["condition"]["prehistory"] == "conditioned", "prehistory_pulse_ms"),
                          (trace["condition"]["continuation"] == "test", "test_pulse_ms")):
        a, b = policy[pulse]
        safe &= (abs(t - a) > dt + 1e-8) & (abs(t - b) > dt + 1e-8)
        if active:
            expected_u[(t >= a) & (t < b)] += policy["stimulus_amplitude_nA"]
    if np.any(abs(np.asarray(data["electrode_nA"])[safe] - expected_u[safe]) > policy["current_atol_nA"]):
        raise ValueError("actual current waveform differs away from recorded switching uncertainty")
    validate_step_ledger(trace, declaration)


def current_components(trace: dict[str, Any]) -> dict[str, np.ndarray]:
    data, site = trace["samples"], trace["site"]
    density_scale = site["area_um2"] * 0.01  # mA/cm2 * um2 -> nA
    result = {name.removesuffix("_mA_cm2"): np.asarray(data[name]) * density_scale for name in ION_COLUMNS}
    v = np.asarray(data["voltage_mV"])
    result["axial_left"] = (v - np.asarray(data["left_voltage_mV"])) / site["left_ri_MOhm"]
    result["axial_right"] = (v - np.asarray(data["right_voltage_mV"])) / site["right_ri_MOhm"]
    result["ionic"] = sum(result[name.removesuffix("_mA_cm2")] for name in ION_COLUMNS)
    result["axial"] = result["axial_left"] + result["axial_right"]
    result["total_outward"] = result["ionic"] + result["axial"]
    return result


def interval_electrode_current(trace: dict[str, Any], policy: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Current used for each completed interval; never infer it from voltage."""
    data = trace["samples"]
    if "interval_electrode_nA" in data:
        return np.asarray(data["interval_electrode_nA"], dtype=float), "recorded_before_fcurrent"
    # Explicit migration for this retained NEURON 9.0.1 fixed-step study.
    # fadvance evaluates IClamp at the midpoint; fcurrent then overwrote its
    # assigned i at the endpoint. The immutable pulse and clock recover the
    # former independently of V, Q, or any ionic/axial current.
    if trace["build_identity"].get("neuron_version") != "9.0.1" or policy["secondorder"] != 0 or policy["cvode_active"]:
        raise ValueError("missing interval current requires the pinned NEURON 9.0.1 fixed-step IClamp semantics")
    t = np.asarray(data["time_ms"])
    midpoint = (t[:-1] + t[1:]) / 2
    current = np.zeros(len(t))
    for active, name in ((trace["condition"]["prehistory"] == "conditioned", "prehistory_pulse_ms"),
                         (trace["condition"]["continuation"] == "test", "test_pulse_ms")):
        if active:
            start, stop = policy[name]
            current[1:] += policy["stimulus_amplitude_nA"] * ((midpoint >= start) & (midpoint <= stop))
    return current, "reconstructed_from_pinned_IClamp_midpoint_law_not_recorded"


def charge_balance(trace: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if policy["current_origins"] != CURRENT_ORIGINS:
        raise ValueError("circular reconstructed current cannot validate charge balance")
    data = trace["samples"]
    t = np.asarray(data["time_ms"])
    q = np.asarray(data["voltage_mV"]) * trace["site"]["capacitance_nF"]
    u = np.asarray(data["electrode_nA"])
    interval_u, forcing_origin = interval_electrode_current(trace, policy)
    current = current_components(trace)["total_outward"]
    # Right-endpoint source readouts share the solved endpoint voltages used
    # for axial current. Ionic currents remain post-state fcurrent values,
    # not the solver's unavailable linearized current: this is a first-order
    # numerical diagnostic, not an exact discrete solver conservation identity.
    integral = np.r_[0.0, np.cumsum((interval_u[1:] - current[1:]) * np.diff(t))]
    residual = q - q[0] - integral
    point_flux = u - current
    old_integral = np.r_[0.0, np.cumsum(0.5 * (point_flux[:-1] + point_flux[1:]) * np.diff(t))]
    old_residual = q - q[0] - old_integral
    scale = max(float(np.ptp(q)), float(np.sum(0.5 * (abs(u[:-1]) + abs(u[1:]) + abs(current[:-1]) + abs(current[1:])) * np.diff(t))))
    tolerance = max(policy["balance_atol_pC"], policy["balance_relative_tolerance"] * scale)
    return {
        "independent_current_accounting": True,
        "forcing_origin": forcing_origin,
        "point_sample_trapezoidal_max_residual_pC": float(np.max(abs(old_residual))),
        "point_sample_trapezoidal_terminal_residual_pC": float(old_residual[-1]),
        "max_abs_integrated_residual_pC": float(np.max(abs(residual))),
        "terminal_residual_pC": float(residual[-1]),
        "charge_range_pC": float(np.ptp(q)),
        "integrated_absolute_current_scale_pC": scale,
        "prospective_screen_tolerance_pC": tolerance,
        "numerical_screen_met": bool(np.max(abs(residual)) <= tolerance),
        "interpretation": "interval forcing/right-endpoint readout diagnostic; post-state ionic currents are not the retained solver linearization; not a convergence proof",
    }


def _phase_indices(trace: dict[str, Any], phase: dict[str, Any], policy: dict[str, Any]) -> np.ndarray:
    data = trace["samples"]
    t, u = np.asarray(data["time_ms"]), np.asarray(data["electrode_nA"])
    expected = policy["stimulus_amplitude_nA"] if trace["condition"]["continuation"] == "test" and phase["pulse_on"] else 0.0
    return np.flatnonzero((t >= phase["start_ms"]) & (t <= phase["end_ms"]) & (abs(u - expected) <= policy["current_atol_nA"]))


def _crossing(trace: dict[str, Any], indices: np.ndarray, charge: float) -> dict[str, float] | None:
    """Earliest eligible piecewise-linear crossing; never extrapolate."""
    data = trace["samples"]
    q = np.asarray(data["voltage_mV"]) * trace["site"]["capacitance_nF"]
    currents = current_components(trace)
    values = {**data, **currents}
    eligible = set(indices.tolist())
    for i in indices:
        if q[i] == charge:
            return {key: float(value[i]) for key, value in values.items()}
        if i + 1 not in eligible or q[i + 1] == q[i]:
            continue
        if min(q[i], q[i + 1]) <= charge <= max(q[i], q[i + 1]):
            fraction = (charge - q[i]) / (q[i + 1] - q[i])
            return {key: float(value[i] + fraction * (value[i + 1] - value[i])) for key, value in values.items()}
    return None


def compare_histories(traces: list[dict[str, Any]], declaration: dict[str, Any]) -> dict[str, Any]:
    policy = declaration["policy"]
    lookup = {(x["condition"]["prehistory"], x["condition"]["continuation"], x["condition"]["dt_ms"]): x for x in traces}
    base, fine = policy["dt_ms"]
    results, unsupported = [], []
    for continuation in ("control", "test"):
        four = [lookup[h, continuation, dt] for h, dt in (("rest", base), ("conditioned", base), ("rest", fine), ("conditioned", fine))]
        for phase in policy["matching_phases"]:
            indices = [_phase_indices(t, phase, policy) for t in four]
            if any(len(i) < 2 for i in indices):
                unsupported.append({"continuation": continuation, "phase": phase["name"], "reason": "insufficient phase/current support"})
                continue
            charges = [np.asarray(t["samples"]["voltage_mV"])[i] * t["site"]["capacitance_nF"] for t, i in zip(four, indices)]
            low, high = max(float(x.min()) for x in charges), min(float(x.max()) for x in charges)
            if low > high:
                unsupported.append({"continuation": continuation, "phase": phase["name"], "reason": "no common charge support"})
                continue
            for q in np.unique(np.linspace(low, high, policy["matching_charge_points"])):
                matches = [_crossing(t, i, float(q)) for t, i in zip(four, indices)]
                if any(m is None for m in matches):
                    unsupported.append({"continuation": continuation, "phase": phase["name"], "reason": "no eligible crossing", "charge_pC": float(q)})
                    continue
                a, b, ar, br = matches
                components = {}
                for name in current_components(four[0]):
                    contrast = br[name] - ar[name]
                    coarse = b[name] - a[name]
                    envelope = policy["refinement_envelope_factor"] * (abs(a[name] - ar[name]) + abs(b[name] - br[name])) + policy["current_atol_nA"]
                    components[name] = {
                        "conditioned_minus_rest_nA": contrast,
                        "base_contrast_nA": coarse,
                        "empirical_refinement_envelope_nA": envelope,
                        "exceeds_empirical_envelope": bool(abs(contrast) > envelope and contrast * coarse > 0),
                    }
                results.append({
                    "continuation": continuation, "phase": phase["name"], "charge_pC": float(q),
                    "refined_rest_time_ms": ar["time_ms"], "refined_conditioned_time_ms": br["time_ms"],
                    "components": components,
                    "refined_state_differences": {name: br[name] - ar[name] for name in declaration["state_variables"]},
                    "refined_boundary_voltage_differences_mV": {name: br[name] - ar[name] for name in ("left_voltage_mV", "right_voltage_mV")},
                })
    return {"matched_points": results, "unsupported_strata": unsupported,
            "numerical_interpretation": "one refinement gives an empirical comparison, not a rigorous error bound"}



def paired_numerical_summaries(traces: list[dict[str, Any]], declaration: dict[str, Any]) -> dict[str, Any]:
    policy = declaration["policy"]
    lookup = {(x["condition"]["prehistory"], x["condition"]["continuation"], x["condition"]["dt_ms"]): x for x in traces}
    base, fine = policy["dt_ms"]
    resolution, effects = {}, {}
    for history in ("rest", "conditioned"):
        for continuation in ("control", "test"):
            a, b = lookup[history, continuation, base], lookup[history, continuation, fine]
            t = np.asarray(a["samples"]["time_ms"])
            differences = {}
            for name in ("voltage_mV", "electrode_nA"):
                d = np.asarray(a["samples"][name]) - np.interp(t, b["samples"]["time_ms"], b["samples"][name])
                differences[name] = {"max_abs_base_minus_refined": float(np.max(abs(d))),
                                     "rms_base_minus_refined": float(np.sqrt(np.mean(d * d)))}
            j = current_components(a)["total_outward"] - np.interp(t, b["samples"]["time_ms"], current_components(b)["total_outward"])
            differences["total_outward_nA"] = {"max_abs_base_minus_refined": float(np.max(abs(j))),
                                               "rms_base_minus_refined": float(np.sqrt(np.mean(j * j)))}
            resolution[f"{history}_{continuation}"] = differences
        for dt in (base, fine):
            control, test = lookup[history, "control", dt], lookup[history, "test", dt]
            t = np.asarray(test["samples"]["time_ms"])
            keep = t >= policy["continuation_start_ms"]
            times = t[keep]
            delta = np.asarray(test["samples"]["voltage_mV"])[keep] - np.asarray(control["samples"]["voltage_mV"])[keep]
            effects[f"{history}_dt{dt}"] = {
                "continuation_test_minus_control_voltage_integral_mV_ms": float(np.sum(0.5 * (delta[:-1] + delta[1:]) * np.diff(times))),
                "max_abs_test_minus_control_voltage_mV": float(np.max(abs(delta))),
                "terminal_test_minus_control_voltage_mV": float(delta[-1]),
            }
    return {"time_aligned_base_refined": resolution, "paired_test_effects": effects,
            "interpretation": "paired source trajectories and one resolution comparison; no independent biological replicates"}


def analyze_traces(declaration: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, Any]:
    validate_declaration(declaration)
    expected = {x["case_id"] for x in condition_matrix(declaration["policy"])}
    ids = [x["condition"]["case_id"] for x in traces]
    if len(ids) != 8 or len(set(ids)) != 8 or set(ids) != expected:
        raise ValueError("exact eight-condition completeness and uniqueness required")
    for trace in traces:
        validate_trace(trace, declaration)
    result: dict[str, Any] = {
        "schema_version": declaration["schema_version"], "declaration_uid": declaration["declaration_uid"],
        "attempted_cases": sum(x["attempt_started"] is True for x in traces),
        "attempt_unknown_cases": sum(x["attempt_started"] is None for x in traces),
        "not_attempted_cases": sum(x["attempt_started"] is False for x in traces),
        "source_time_advanced_count": sum(x["source_time_advanced_count"] for x in traces
                                        if x["source_time_advanced_count"] is not None and x["status"] != "RUNNING"),
        "source_advance_unknown_cases": sum(x["source_time_advanced_count"] is None or x["status"] == "RUNNING" for x in traces),
        "case_errors": {x["condition"]["case_id"]: x["error"] for x in traces if x["error"]},
        "case_costs": {x["condition"]["case_id"]: x["cost"] for x in traces},
        "candidate": declaration["policy"]["candidate"], "planned_cases": 8,
        "completed_cases": sum(x["status"] == "COMPLETE" for x in traces),
        "case_statuses": {x["condition"]["case_id"]: x["status"] for x in traces},
        "scope": "one deterministic source/site; native and biological qualification unavailable",
        "ScientificTwin": "NOT_RUN", "governed_access_counters": [0] * 10,
        "real_world_formal_agent_evaluation": False,
    }
    if result["completed_cases"] != 8:
        result["status"] = "INCOMPLETE_SOURCE_EXECUTION"
        return result
    policy = declaration["policy"]
    if declaration["schema_version"] == LEDGER_VERSION:
        result["solver_step_accounting"] = {x["condition"]["case_id"]: solver_step_balance(x, declaration) for x in traces}
        result["solver_step_numerical_screen_met"] = all(x["numerical_screen_met"] for x in result["solver_step_accounting"].values())
    result["charge_balance"] = {x["condition"]["case_id"]: charge_balance(x, policy) for x in traces}
    result["numerical_and_paired_effects"] = paired_numerical_summaries(traces, declaration)
    result["matching"] = compare_histories(traces, declaration)
    rows = result["matching"]["matched_points"]
    physical_ok = all(x["numerical_screen_met"] for x in result["charge_balance"].values())
    result["physical_endpoint_numerical_screen_met"] = physical_ok
    balance_ok = physical_ok and result.get("solver_step_numerical_screen_met", True)
    challenged = [r for r in rows if r["components"]["total_outward"]["exceeds_empirical_envelope"]]
    if not rows:
        result["status"] = "INCONCLUSIVE_NO_COMMON_SUPPORT"
    elif not balance_ok:
        result["status"] = "INCONCLUSIVE_NUMERICAL_ATTRIBUTION"
    elif challenged:
        result["status"] = "CHARGE_ONLY_DISCREPANCY_EXCEEDS_EMPIRICAL_REFINEMENT"
    else:
        result["status"] = "NOT_CONTRADICTED_ON_TESTED_SUPPORT"
    result["matched_point_count"] = len(rows)
    result["charge_only_discrepancy_count"] = len(challenged)
    result["component_contrasts"] = {}
    for name in ("ionic", "axial", "ina", "ik", "ica", "iCa", "i_pas"):
        supported = [r for r in rows if r["components"][name]["exceeds_empirical_envelope"]]
        strongest = max(supported, key=lambda r: abs(r["components"][name]["conditioned_minus_rest_nA"]), default=None)
        result["component_contrasts"][name] = {
            "count_exceeding_empirical_envelope": len(supported),
            "strongest_point": strongest,
        }
    result["analysis_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["native_implication"] = {
        "numerical_screen_met": balance_ok,
        "readout": "Gamma=(Q,C,site) supplies source Q=C*V; no native latent index has been identified",
        "electrophysiology": ("Numerical attribution remains unresolved; the recorded current/state contrasts do not qualify the charge-only closure and do not establish a missing sufficient statistic. A native interpretation would need a source-supported ionic/history and neighboring-electrical-state law, whose identification remains open." if not balance_ok else "Recorded current contrasts test only this source charge-only candidate under its empirical numerical assumptions. Gates/calcium are measured correlates, not an isolated channel cause or an identified native sufficient statistic."),
        "morphology": "Membrane area/cm fix C; the recorded endpoint potentials and ri fix axial currents. Static geometry alone does not specify the neighboring electrical state. Their joint native responsibilities require an explicit source-to-native support and coupling law.",
        "site": declaration["site"],
        "limits": "No universal hidden-state insufficiency, global closure proof, biological replication, error calibration, native registration or full-grid teacher admission.",
    }
    return result


def analyze_directory(root: Path) -> dict[str, Any]:
    declaration = json.loads((root / "declaration.json").read_text())
    traces = []
    for condition in condition_matrix(declaration["policy"]):
        path = root / (condition["case_id"] + ".json")
        try:
            trace = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            trace = unavailable_trace(declaration, condition, f"{type(error).__name__}: {error}")
        traces.append(trace)
    result = analyze_traces(declaration, traces)
    write_json(root / "result.json", result)
    return result


LEDGER_VERSION = "source-membrane-charge-registration-v2"
LEDGER_MODE = "solver_step_v2"
LEDGER_DEFINITION = "neuron_9_0_1_fast_imem_minus_capacitance_v1"
LEDGER_PHASE = "after_voltage_update_before_nonvint_state_update"
LEDGER_COLUMNS = (
    "end_time_ms", "voltage_mV", "left_voltage_mV", "right_voltage_mV",
    "membrane_current_nA", "capacitive_current_mA_cm2", "electrode_current_nA",
    "assembled_ionic_current_mA_cm2",
)
LEDGER_ORIGINS = {
    "membrane_current_nA": "NEURON_i_membrane_includes_capacitance_excludes_ELECTRODE_CURRENT",
    "capacitive_current_mA_cm2": "NEURON_i_cap_from_solved_voltage_increment",
    "electrode_current_nA": "IClamp_i_assigned_at_interval_midpoint",
    "assembled_ionic_current_mA_cm2": "source_ion_aggregates_plus_pas_before_state_update",
    "axial_current_nA": "solved_endpoint_voltage_differences_over_source_ri",
    "effective_ionic_current_nA": "derived_i_membrane_minus_area_scaled_i_cap",
}


def recording_version(policy: dict[str, Any]) -> str:
    mode = policy.get("recording_mode", "endpoint_v1")
    if mode not in ("endpoint_v1", LEDGER_MODE):
        raise ValueError("unsupported source current recording mode")
    return LEDGER_VERSION if mode == LEDGER_MODE else CONTRACT_VERSION


def validate_ledger_contract(policy: dict[str, Any], schema: dict[str, Any] | None = None,
                             build: dict[str, Any] | None = None) -> None:
    if recording_version(policy) == CONTRACT_VERSION:
        return
    ledger = policy.get("source_step_ledger")
    if not isinstance(ledger, dict):
        raise ValueError("selected solver-step mode requires its independent ledger definition before advance")
    if (ledger.get("definition_id") != LEDGER_DEFINITION or ledger.get("phase") != LEDGER_PHASE
            or ledger.get("origins") != LEDGER_ORIGINS):
        raise ValueError("invalid/circular solver-step ledger origins or phase")
    if ledger.get("primary_revision") != "b12a541" or not ledger.get("primary_sources"):
        raise ValueError("solver-step ledger requires its exact primary implementation references")
    if ledger.get("required_neuron_version") != "9.0.1" or ledger.get("required_nrnversion") != "NEURON -- VERSION 9.0.1 HEAD (b12a541+) 2025-11-14":
        raise ValueError("solver-step recording definition is bound to the inspected runtime")
    required = ledger.get("required_build_identity")
    if not isinstance(required, dict) or set(required) != {"source_files_uid", "neuron_runtime_libraries", "python_executable_sha256"}:
        raise ValueError("selected solver ledger requires the fixed source/runtime identity")
    if schema is not None:
        specification = schema.get("step_ledger")
        if not isinstance(specification, dict) or specification.get("sample_columns") != list(LEDGER_COLUMNS):
            raise ValueError("selected solver-step mode requires its explicit ledger schema")
        if specification.get("units") != {
            "end_time_ms": "ms", "voltage_mV": "mV", "left_voltage_mV": "mV",
            "right_voltage_mV": "mV", "membrane_current_nA": "nA",
            "capacitive_current_mA_cm2": "mA/cm2", "electrode_current_nA": "nA",
            "assembled_ionic_current_mA_cm2": "mA/cm2",
        }:
            raise ValueError("solver-step ledger unit mismatch")
    if build is not None:
        if any(build.get(key) != value for key, value in required.items()):
            raise ValueError("solver-step source/runtime identity differs from the fixed source study")
        for key in ("neuron_version", "nrnversion"):
            if build.get(key) != ledger["required_" + key]:
                raise ValueError("solver-step ledger/runtime identity mismatch")


def empty_step_ledger(declaration: dict[str, Any]) -> dict[str, Any]:
    definition = declaration["step_current_contract"]
    return {"definition_id": definition["definition_id"], "phase": definition["phase"],
            "units": declaration["schema"]["step_ledger"]["units"],
            "samples": {name: [] for name in LEDGER_COLUMNS}}


def validate_step_ledger(trace: dict[str, Any], declaration: dict[str, Any]) -> None:
    if declaration["schema_version"] != LEDGER_VERSION:
        return
    ledger = trace.get("step_ledger")
    expected = empty_step_ledger(declaration)
    if not isinstance(ledger, dict):
        raise ValueError("selected solver-step mode requires ledger before advance")
    for key in ("definition_id", "phase", "units"):
        if ledger.get(key) != expected[key]:
            raise ValueError("step ledger definition/phase/unit mismatch")
    data = ledger.get("samples")
    if not isinstance(data, dict) or set(data) != set(LEDGER_COLUMNS):
        raise ValueError("step ledger columns missing or unexpected")
    lengths = set()
    for name, values in data.items():
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError(f"nonfinite or malformed step ledger {name}")
        lengths.add(len(a))
    if len(lengths) != 1:
        raise ValueError("step ledger arrays have inconsistent lengths")
    if trace["status"] != "COMPLETE":
        return
    n = len(trace["samples"]["time_ms"]) - 1
    if lengths != {n}:
        raise ValueError("incomplete solver-step ledger")
    for field, sample in (("end_time_ms", "time_ms"), ("voltage_mV", "voltage_mV"),
                          ("left_voltage_mV", "left_voltage_mV"), ("right_voltage_mV", "right_voltage_mV"),
                          ("electrode_current_nA", "interval_electrode_nA")):
        if sample not in trace["samples"] or not np.allclose(data[field], trace["samples"][sample][1:], rtol=0, atol=1e-8):
            raise ValueError("step ledger phase/index alignment mismatch")


def solver_step_balance(trace: dict[str, Any], declaration: dict[str, Any]) -> dict[str, Any]:
    """Use the source's assembled membrane output, never a forcing-minus-charge residual."""
    validate_step_ledger(trace, declaration)
    data, site, policy = trace["step_ledger"]["samples"], trace["site"], declaration["policy"]
    scale = site["area_um2"] * 0.01
    t = np.asarray(trace["samples"]["time_ms"])
    dt = np.diff(t)
    v = np.asarray(data["voltage_mV"])
    membrane = np.asarray(data["membrane_current_nA"])
    capacitive = np.asarray(data["capacitive_current_mA_cm2"]) * scale
    ionic_effective = membrane - capacitive
    electrode = np.asarray(data["electrode_current_nA"])
    left = (v - np.asarray(data["left_voltage_mV"])) / site["left_ri_MOhm"]
    right = (v - np.asarray(data["right_voltage_mV"])) / site["right_ri_MOhm"]
    axial = left + right
    balance = electrode - membrane - axial
    delta_q = np.diff(np.asarray(trace["samples"]["voltage_mV"])) * site["capacitance_nF"]
    capacitive_error = delta_q - capacitive * dt
    integrated_error = np.cumsum(delta_q - (electrode - ionic_effective - axial) * dt)
    absolute_scale = float(np.sum(dt * (abs(electrode) + abs(ionic_effective) + abs(left) + abs(right))))
    tolerance = max(policy["balance_atol_pC"], policy["balance_relative_tolerance"] * absolute_scale)
    instantaneous_ok = bool(np.max(abs(balance)) <= policy["current_atol_nA"])
    cap_ok = bool(np.max(abs(capacitive_error)) <= policy["balance_atol_pC"])
    integrated_ok = bool(np.max(abs(integrated_error)) <= tolerance)
    endpoint = current_components(trace)["ionic"][1:]
    assembled = np.asarray(data["assembled_ionic_current_mA_cm2"]) * scale
    endpoint_components = current_components(trace)
    endpoint_total = endpoint_components["total_outward"]
    interval_u, _ = interval_electrode_current(trace, policy)
    q = np.asarray(trace["samples"]["voltage_mV"]) * site["capacitance_nF"]
    endpoint_residual = q[1:] - q[0] - np.cumsum(dt * (interval_u[1:] - endpoint_total[1:]))
    endpoint_shift = np.cumsum(dt * (endpoint - assembled))
    effective_correction = np.cumsum(dt * (ionic_effective - assembled))
    spatial_forcing_difference = np.cumsum(dt * (
        electrode - interval_u[1:] + endpoint_components["axial"][1:] - axial))
    decomposition_error = endpoint_residual - integrated_error - (
        endpoint_shift - effective_correction + spatial_forcing_difference)
    point_u = np.asarray(trace["samples"]["electrode_nA"])
    forcing_clock = -np.cumsum(dt * (0.5 * (point_u[:-1] + point_u[1:]) - interval_u[1:]))
    current_quadrature = np.cumsum(dt * (
        0.5 * (endpoint_total[:-1] + endpoint_total[1:]) - endpoint_total[1:]))
    point_residual = q[1:] - q[0] - np.cumsum(dt * 0.5 * (
        point_u[:-1] + point_u[1:] - endpoint_total[:-1] - endpoint_total[1:]))
    def summary(values):
        return {"max_abs_pC": float(np.max(abs(values))), "terminal_pC": float(values[-1])}
    return {
        "definition_id": LEDGER_DEFINITION, "completed_intervals": len(dt),
        "effective_ionic_origin": LEDGER_ORIGINS["effective_ionic_current_nA"],
        "effective_ionic_interpretation": "derived solver-linearized total, not an instantaneous measured channel current",
        "max_abs_current_balance_residual_nA": float(np.max(abs(balance))),
        "current_screen_tolerance_nA": policy["current_atol_nA"],
        "current_screen_met": instantaneous_ok,
        "max_abs_capacitive_charge_residual_pC": float(np.max(abs(capacitive_error))),
        "capacitive_screen_met": cap_ok,
        "max_abs_integrated_residual_pC": float(np.max(abs(integrated_error))),
        "terminal_residual_pC": float(integrated_error[-1]),
        "integrated_screen_tolerance_pC": tolerance,
        "integrated_screen_met": integrated_ok,
        "numerical_screen_met": instantaneous_ok and cap_ok and integrated_ok,
        "max_abs_effective_minus_assembled_ionic_nA": float(np.max(abs(ionic_effective - assembled))),
        "max_abs_effective_minus_endpoint_ionic_nA": float(np.max(abs(ionic_effective - endpoint))),
        "max_abs_endpoint_minus_assembled_ionic_nA": float(np.max(abs(endpoint - assembled))),
        "endpoint_residual_decomposition": {
            "endpoint_minus_assembled": summary(endpoint_shift),
            "effective_minus_assembled": summary(effective_correction),
            "spatial_and_forcing_alignment": summary(spatial_forcing_difference),
            "reconciliation_error": summary(decomposition_error),
            "identity": "R_endpoint = R_ledger + integral(endpoint-assembled) - integral(effective-assembled) + spatial/forcing alignment",
            "scope": "endpoint-minus-assembled combines voltage/time/state changes; not an isolated channel-state or biological effect",
        },
        "point_quadrature_decomposition": {
            "endpoint_forcing_clock": summary(forcing_clock),
            "endpoint_current_quadrature": summary(current_quadrature),
            "reconciliation_error": summary(point_residual - endpoint_residual - forcing_clock - current_quadrature),
            "identity": "R_point_trapezoid = R_endpoint + forcing_clock + current_quadrature",
        },
        "cancellation": {
            "max_sum_absolute_terms_nA": float(np.max(abs(electrode) + abs(membrane) + abs(left) + abs(right))),
            "max_abs_axial_left_nA": float(np.max(abs(left))),
            "max_abs_axial_right_nA": float(np.max(abs(right))),
            "max_absolute_to_net_scale_ratio": float(np.max(
                (abs(electrode) + abs(membrane) + abs(left) + abs(right)) /
                np.maximum(abs(balance), policy["current_atol_nA"]))),
        },
        "scope": "discrete source accounting only; original physical endpoint screens remain independently required",
    }


def unavailable_trace(declaration: dict[str, Any], condition: dict[str, Any], error: str) -> dict[str, Any]:
    trace = {
        "schema_version": declaration["schema_version"], "declaration_uid": declaration["declaration_uid"],
        "condition": condition, "site": declaration["site"], "units": declaration["units"],
        "build_identity": declaration["build"]["identity"], "status": "MISSING",
        "attempt_started": None, "samples": {}, "observed_mask": [],
        "source_time_advanced_count": None, "error": error, "cost": {},
    }
    if declaration["schema_version"] == LEDGER_VERSION:
        trace["step_ledger"] = empty_step_ledger(declaration)
    return trace
