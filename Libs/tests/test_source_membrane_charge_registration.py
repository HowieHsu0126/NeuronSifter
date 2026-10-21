from __future__ import annotations

from copy import deepcopy
import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from utils.data.raw_data_registry import load_config
from utils.evaluation import source_membrane_charge_registration as registration

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py"


@pytest.fixture(scope="module")
def configuration():
    return load_config(ROOT / "Libs/configs/config_stack.yaml")


@pytest.fixture
def declaration(configuration):
    config = configuration
    policy = deepcopy(config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"])
    policy.update(duration_ms=4.0, continuation_start_ms=0.5,
                  prehistory_pulse_ms=[0.1, 0.4], test_pulse_ms=[2.4, 3.6],
                  matching_charge_points=5,
                  matching_phases=[{"name": "off", "start_ms": 0.6, "end_ms": 2.3, "pulse_on": False}])
    schema = config["schemas"]["source_membrane_charge_registration"]
    site = {"section_name": "soma[0]", "segment_x": 0.5, "area_um2": 100.0,
            "cm_uF_cm2": 1.0, "capacitance_nF": 0.001,
            "left_ri_MOhm": 1.0, "right_ri_MOhm": 1.0}
    value = {"schema_version": registration.CONTRACT_VERSION, "policy": policy, "units": schema["units"],
             "schema": schema, "site": site, "source": {"fixture": True},
             "build": {"identity": {"fixture": True}},
             "state_variables": {"state_m_na3": "m_na3"}, "source_parameters": {},
             "implementation": {}, "execution": {}}
    value["declaration_uid"] = registration.content_uid(value)
    return value


def traces_for(declaration, conditioned_slope=1.0):
    records = []
    policy = declaration["policy"]
    for condition in registration.condition_matrix(policy):
        dt = condition["dt_ms"]
        t = np.arange(round(policy["duration_ms"] / dt) + 1) * dt
        slope = conditioned_slope if condition["prehistory"] == "conditioned" else 1.0
        v = policy["initial_voltage_mV"] + slope * t
        u = np.zeros_like(t)
        for active, name in ((condition["prehistory"] == "conditioned", "prehistory_pulse_ms"),
                             (condition["continuation"] == "test", "test_pulse_ms")):
            start, stop = policy[name]
            if active:
                u[(t >= start) & (t < stop)] += policy["stimulus_amplitude_nA"]
        # Analytic fixture: independently specified axial flux yields the known
        # linear charge solution; production never obtains J from dV/dt.
        axial = u - declaration["site"]["capacitance_nF"] * slope
        data = {"time_ms": t.tolist(), "voltage_mV": v.tolist(), "electrode_nA": u.tolist(), "interval_electrode_nA": u.tolist(),
                "left_voltage_mV": (v - axial).tolist(), "right_voltage_mV": v.tolist(),
                "state_m_na3": np.full_like(t, 0.1 * slope).tolist()}
        data.update({name: np.zeros_like(t).tolist() for name in registration.ION_COLUMNS})
        records.append({"schema_version": registration.CONTRACT_VERSION,
                        "declaration_uid": declaration["declaration_uid"], "condition": condition,
                        "site": deepcopy(declaration["site"]), "units": declaration["units"],
                        "build_identity": declaration["build"]["identity"], "status": "COMPLETE",
                        "attempt_started": True, "samples": data, "observed_mask": [True] * len(t),
                        "source_time_advanced_count": len(t) - 1, "error": None, "cost": {}})
    return records


def test_capacitance_units_and_positive_inward_sign():
    assert registration.capacitance_nf(1.0, 400.0) == pytest.approx(0.004)
    # nA*ms / nF = mV
    assert 0.15 * 1.0 / registration.capacitance_nf(1.0, 400.0) == pytest.approx(37.5)
    with pytest.raises(ValueError):
        registration.capacitance_nf(0, 400)


def test_analytic_balance_and_perturbed_independent_current(declaration):
    trace = traces_for(declaration)[0]
    balance = registration.charge_balance(trace, declaration["policy"])
    assert balance["max_abs_integrated_residual_pC"] < 1e-12
    trace["samples"]["ina_mA_cm2"] = [0.01] * len(trace["samples"]["time_ms"])
    changed = registration.charge_balance(trace, declaration["policy"])
    assert changed["terminal_residual_pC"] == pytest.approx(0.04)
    assert not changed["numerical_screen_met"]


def test_circular_current_cannot_validate_balance(declaration):
    policy = deepcopy(declaration["policy"])
    policy["current_origins"]["axial"] = "u-C*dV/dt"
    with pytest.raises(ValueError, match="circular"):
        registration.charge_balance(traces_for(declaration)[0], policy)
    with pytest.raises(ValueError, match="independent"):
        registration.validate_policy(policy)


def test_history_difference_has_scoped_axial_attribution(declaration):
    result = registration.analyze_traces(declaration, traces_for(declaration, 2.0))
    assert result["status"] == "CHARGE_ONLY_DISCREPANCY_EXCEEDS_EMPIRICAL_REFINEMENT"
    assert result["charge_only_discrepancy_count"] > 0
    assert result["component_contrasts"]["axial"]["count_exceeding_empirical_envelope"] > 0
    assert result["component_contrasts"]["ionic"]["count_exceeding_empirical_envelope"] == 0
    assert "not a rigorous error bound" in result["matching"]["numerical_interpretation"]


def test_no_overlap_is_inconclusive(declaration):
    result = registration.analyze_traces(declaration, traces_for(declaration, 10.0))
    assert result["status"] == "INCONCLUSIVE_NO_COMMON_SUPPORT"
    assert result["matched_point_count"] == 0
    assert len(result["matching"]["unsupported_strata"]) == 2


def test_equal_closure_only_not_contradicted(declaration):
    result = registration.analyze_traces(declaration, traces_for(declaration))
    assert result["status"] == "NOT_CONTRADICTED_ON_TESTED_SUPPORT"
    assert result["ScientificTwin"] == "NOT_RUN"
    assert result["governed_access_counters"] == [0] * 10


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_eight_case_denominator(declaration, mutation):
    traces = traces_for(declaration)
    if mutation == "missing":
        traces.pop()
    else:
        traces[-1] = traces[0]
    with pytest.raises(ValueError, match="eight-condition"):
        registration.analyze_traces(declaration, traces)


@pytest.mark.parametrize("key", ["site", "units", "build_identity", "declaration_uid"])
def test_identity_rejection(declaration, key):
    trace = traces_for(declaration)[0]
    trace[key] = {} if key != "declaration_uid" else "changed"
    with pytest.raises(ValueError, match="mismatch"):
        registration.validate_trace(trace, declaration)


@pytest.mark.parametrize("mutation", ["nonfinite", "time", "mask", "channel", "waveform"])
def test_trace_errors(declaration, mutation):
    trace = traces_for(declaration)[0]
    if mutation == "nonfinite":
        trace["samples"]["voltage_mV"][4] = float("nan")
    elif mutation == "time":
        trace["samples"]["time_ms"][4] += 0.01
    elif mutation == "mask":
        trace["observed_mask"][4] = False
    elif mutation == "channel":
        del trace["samples"]["ica_mA_cm2"]
    else:
        trace["samples"]["electrode_nA"][20] = 1.0
    with pytest.raises(ValueError):
        registration.validate_trace(trace, declaration)


def test_failed_case_is_retained(declaration):
    traces = traces_for(declaration)
    traces[3]["status"] = "FAILED"
    traces[3]["error"] = "source failure"
    result = registration.analyze_traces(declaration, traces)
    assert result["status"] == "INCOMPLETE_SOURCE_EXECUTION"
    assert result["planned_cases"] == 8 and result["completed_cases"] == 7


def test_declaration_drift_is_rejected(declaration):
    declaration["policy"]["stimulus_amplitude_nA"] *= 2
    with pytest.raises(ValueError, match="identity"):
        registration.validate_declaration(declaration)


def test_actual_cli_analysis_and_error_exit(declaration, tmp_path):
    registration.write_json(tmp_path / "declaration.json", declaration)
    for trace in traces_for(declaration, 2.0):
        registration.write_json(tmp_path / (trace["condition"]["case_id"] + ".json"), trace)
    process = subprocess.run([sys.executable, str(RUNNER), "--operation", "analyze-source-registration",
                              "--config", str(ROOT / "Libs/configs/config_stack.yaml"),
                              "--result-dir", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert process.returncode == 0, process.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["completed_cases"] == 8
    assert result["status"] == "CHARGE_ONLY_DISCREPANCY_EXCEEDS_EMPIRICAL_REFINEMENT"
    (tmp_path / "declaration.json").write_text("{}")
    bad = subprocess.run([sys.executable, str(RUNNER), "--operation", "analyze-source-registration",
                          "--config", str(ROOT / "Libs/configs/config_stack.yaml"),
                          "--result-dir", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 2
    assert "source registration error" in bad.stderr


def test_legacy_default_and_normalization_are_preserved(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("source_runner_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    assert module.parse_args().operation == "legacy"
    config = load_config(ROOT / "Libs/configs/config_stack.yaml")
    values = module.normalize_voltage(config, np.array([-70.0, -40.0, 0.0]))
    np.testing.assert_array_equal(values, [0.0, 1.0, 1.5])

def test_pinned_midpoint_reconstruction_removes_point_clock_impulse(declaration):
    trace = traces_for(declaration)[4]
    trace["samples"].pop("interval_electrode_nA")
    trace["build_identity"] = {"neuron_version": "9.0.1"}
    current, origin = registration.interval_electrode_current(trace, declaration["policy"])
    t = np.asarray(trace["samples"]["time_ms"])
    # At the first endpoint exactly at pulse start, point current is on but
    # the completed interval midpoint is still before the pulse.
    index = int(np.flatnonzero(np.isclose(t, 0.1))[0])
    assert trace["samples"]["electrode_nA"][index] == 0.15
    assert current[index] == 0.0
    assert "reconstructed" in origin
    v = np.asarray(trace["samples"]["voltage_mV"])
    axial = current - trace["site"]["capacitance_nF"]
    trace["samples"]["left_voltage_mV"] = (v - axial).tolist()
    balance = registration.charge_balance(trace, declaration["policy"])
    assert balance["max_abs_integrated_residual_pC"] < 1e-12
    assert balance["point_sample_trapezoidal_max_residual_pC"] > 0.001


def test_unpinned_missing_interval_current_rejects(declaration):
    trace = traces_for(declaration)[0]
    trace["samples"].pop("interval_electrode_nA")
    with pytest.raises(ValueError, match="pinned"):
        registration.interval_electrode_current(trace, declaration["policy"])


def test_balance_failure_withholds_native_omission_claim(declaration):
    traces = traces_for(declaration, 2.0)
    traces[0]["samples"]["ina_mA_cm2"] = [0.01] * len(traces[0]["samples"]["time_ms"])
    result = registration.analyze_traces(declaration, traces)
    assert result["status"] == "INCONCLUSIVE_NUMERICAL_ATTRIBUTION"
    assert result["native_implication"]["numerical_screen_met"] is False


def test_interval_capture_precedes_current_refresh(monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("source_runner_ordering", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stimulus = SimpleNamespace(i=0.15)
    monkeypatch.setattr(module, "h", SimpleNamespace(fcurrent=lambda: setattr(stimulus, "i", 0.0)))
    assert module._capture_interval_current_and_refresh([stimulus]) == 0.15
    assert stimulus.i == 0.0


@pytest.fixture
def ledger_declaration(declaration, configuration):
    value = deepcopy(declaration)
    value["schema_version"] = registration.LEDGER_VERSION
    value["policy"]["recording_mode"] = registration.LEDGER_MODE
    value["step_current_contract"] = deepcopy(value["policy"]["source_step_ledger"])
    value["build"]["identity"] = {
        **deepcopy(value["step_current_contract"]["required_build_identity"]),
        "neuron_version": value["step_current_contract"]["required_neuron_version"],
        "nrnversion": value["step_current_contract"]["required_nrnversion"],
    }
    value["source"]["model_config"] = configuration["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"]
    value.pop("declaration_uid")
    value["declaration_uid"] = registration.content_uid(value)
    return value


def analytic_ledger_traces(declaration):
    records = traces_for(declaration)
    for trace in records:
        trace["schema_version"] = declaration["schema_version"]
        data = trace["samples"]
        v, u = np.asarray(data["voltage_mV"]), np.asarray(data["interval_electrode_nA"])
        # Independent analytic circuit: C=.001 nF, slope=1 mV/ms,
        # ionic outward current=.0002 nA, prescribed boundary current=u-.0012.
        data["ina_mA_cm2"] = np.full_like(v, 0.0002).tolist()
        data["left_voltage_mV"] = (v - (u - 0.0012)).tolist()
        trace["step_ledger"] = registration.empty_step_ledger(declaration)
        ledger = trace["step_ledger"]["samples"]
        ledger.update(
            end_time_ms=data["time_ms"][1:], voltage_mV=data["voltage_mV"][1:],
            left_voltage_mV=data["left_voltage_mV"][1:], right_voltage_mV=data["right_voltage_mV"][1:],
            membrane_current_nA=[0.0012] * (len(v) - 1),
            capacitive_current_mA_cm2=[0.001] * (len(v) - 1),
            electrode_current_nA=u[1:].tolist(),
            assembled_ionic_current_mA_cm2=[0.0001] * (len(v) - 1))
    return records


@pytest.fixture
def source_runner():
    spec = importlib.util.spec_from_file_location("source_step_runner_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_solver_step_analytic_balance(ledger_declaration):
    for trace in analytic_ledger_traces(ledger_declaration):
        registration.validate_trace(trace, ledger_declaration)
        result = registration.solver_step_balance(trace, ledger_declaration)
        assert result["numerical_screen_met"]
        assert result["max_abs_current_balance_residual_nA"] < 1e-12
        assert result["max_abs_integrated_residual_pC"] < 1e-12
        assert result["max_abs_effective_minus_assembled_ionic_nA"] == pytest.approx(0.0001)
        assert result["effective_ionic_origin"] == "derived_i_membrane_minus_area_scaled_i_cap"


@pytest.mark.parametrize("mutation", ["membrane_sign", "double_capacitance", "electrode_sign"])
def test_ledger_detects_independent_current_sign_and_double_counting(ledger_declaration, mutation):
    trace = analytic_ledger_traces(ledger_declaration)[-1]
    data = trace["step_ledger"]["samples"]
    key, factor = {"membrane_sign": ("membrane_current_nA", -1),
                   "double_capacitance": ("capacitive_current_mA_cm2", 2),
                   "electrode_sign": ("electrode_current_nA", -1)}[mutation]
    data[key] = (np.asarray(data[key]) * factor).tolist()
    if mutation == "electrode_sign":
        # Preserve phase agreement while testing the independent balance equation.
        trace["samples"]["interval_electrode_nA"][1:] = data[key]
    assert not registration.solver_step_balance(trace, ledger_declaration)["numerical_screen_met"]


@pytest.mark.parametrize("mutation", ["missing", "column", "nonfinite", "length", "time", "phase", "units"])
def test_selected_ledger_rejects_missing_or_misaligned_data(ledger_declaration, mutation):
    trace = analytic_ledger_traces(ledger_declaration)[0]
    ledger = trace["step_ledger"]
    if mutation == "missing":
        del trace["step_ledger"]
    elif mutation == "column":
        del ledger["samples"]["membrane_current_nA"]
    elif mutation == "nonfinite":
        ledger["samples"]["membrane_current_nA"][0] = float("nan")
    elif mutation == "length":
        ledger["samples"]["membrane_current_nA"].pop()
    elif mutation == "time":
        ledger["samples"]["end_time_ms"] = (np.asarray(ledger["samples"]["end_time_ms"]) + 0.05).tolist()
    else:
        ledger[mutation] = "incorrect"
    with pytest.raises(ValueError):
        registration.validate_trace(trace, ledger_declaration)


def test_ledger_contract_rejects_circularity_and_runtime_drift(ledger_declaration):
    for mutation in ("origins", "runtime", "missing_contract", "schema"):
        value = deepcopy(ledger_declaration)
        if mutation == "origins":
            value["policy"]["source_step_ledger"]["origins"]["effective_ionic_current_nA"] = "u-C*dV/dt"
            value["step_current_contract"] = value["policy"]["source_step_ledger"]
        elif mutation == "runtime":
            value["build"]["identity"]["neuron_version"] = "uninspected"
        elif mutation == "missing_contract":
            value.pop("step_current_contract")
        else:
            value["schema"].pop("step_ledger")
        value.pop("declaration_uid")
        value["declaration_uid"] = registration.content_uid(value)
        with pytest.raises(ValueError):
            registration.validate_declaration(value)


def test_solver_ledger_pass_cannot_replace_physical_endpoint_screen(ledger_declaration):
    records = analytic_ledger_traces(ledger_declaration)
    records[0]["samples"]["ina_mA_cm2"] = [0.01] * len(records[0]["samples"]["time_ms"])
    result = registration.analyze_traces(ledger_declaration, records)
    assert result["solver_step_numerical_screen_met"]
    assert not result["physical_endpoint_numerical_screen_met"]
    assert result["status"] == "INCONCLUSIVE_NUMERICAL_ATTRIBUTION"
    assert not result["native_implication"]["numerical_screen_met"]


def test_callback_uses_thread_clock_skips_initialization_and_cleans_up(
    ledger_declaration, source_runner, monkeypatch
):
    trace = analytic_ledger_traces(ledger_declaration)[0]
    trace["step_ledger"] = registration.empty_step_ledger(ledger_declaration)
    segment = SimpleNamespace(v=-64.9, i_membrane_=0.0012, i_cap=0.001,
                              ina=0.0001, ik=0, ica=0, iCa=0, i_pas=0)
    endpoint = SimpleNamespace(v=-64.901)
    def section(x):
        return segment if x == 0.5 else endpoint
    callbacks, fast = [], [False]
    clock = [0.0]
    events = []
    def fast_imem(value=None):
        old = fast[0]
        if value is not None:
            fast[0] = value
        return old
    def add(direction, callback):
        assert direction == 0
        callbacks.append(callback)
    cv = SimpleNamespace(use_fast_imem=fast_imem, extra_scatter_gather=add,
                         extra_scatter_gather_remove=lambda f: callbacks.remove(f))
    fake = SimpleNamespace(t=-999, CVode=lambda: cv,
                           ParallelContext=lambda: SimpleNamespace(t=lambda _: clock[0]))
    monkeypatch.setattr(source_runner, "h", fake)
    recorder = source_runner.SourceStepCurrentRecorder(section, [SimpleNamespace(i=0.15)], trace)
    assert callbacks == []  # initialization happens before install
    recorder.install()
    clock[0] = 0.1
    events.append("voltage_updated")
    callbacks[0]()
    events.append("state_update")
    segment.i_cap = 999
    segment.i_membrane_ = 999
    ledger = trace["step_ledger"]["samples"]
    assert ledger["end_time_ms"] == [0.1] and fake.t == -999
    assert ledger["capacitive_current_mA_cm2"] == [0.001]
    assert ledger["membrane_current_nA"] == [0.0012]
    assert events == ["voltage_updated", "state_update"]
    with pytest.raises(ValueError, match="phase/cardinality"):
        callbacks[0]()  # duplicated callback is not another interval
    recorder.close()
    assert callbacks == [] and fast[0] is False


@pytest.mark.parametrize("operation", ["source-registration", "source-registration-case"])
def test_actual_cli_missing_ledger_definition_rejects_before_source(
    configuration, tmp_path, operation
):
    config = deepcopy(configuration)
    config["logging_registry"].update(file_enabled=False, console_enabled=False,
                                      root_dir=str(tmp_path / "logs"))
    config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"].pop("source_step_ledger")
    path = tmp_path / "no-ledger.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    result = subprocess.run([sys.executable, str(RUNNER), "--config", str(path),
                             "--operation", operation, "--registration-recording-mode", "solver_step_v2",
                             "--run-id", "must_not_run", "--result-dir", str(tmp_path / "absent"),
                             "--source-root", str(tmp_path / "absent-source"),
                             "--scratch-root", str(tmp_path / "absent-scratch"),
                             "--case-id", "rest_control_dt0p1"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires its independent ledger definition before advance" in result.stderr
    assert not (tmp_path / "absent-scratch").exists()


def test_actual_dispatch_retains_eight_attempts_including_prechild_failures(
    ledger_declaration, source_runner, tmp_path, monkeypatch, configuration
):
    root, scratch = tmp_path / "results", tmp_path / "scratch"
    root.mkdir(); scratch.mkdir()
    registration.write_json(root / "declaration.json", ledger_declaration)
    records = {r["condition"]["case_id"]: r for r in analytic_ledger_traces(ledger_declaration)}
    for condition in registration.condition_matrix(ledger_declaration["policy"]):
        registration.write_json(root / (condition["case_id"] + ".json"),
                                source_runner._empty_registration_trace(ledger_declaration, condition))
    calls = []
    def dispatch(command, **kwargs):
        case = command[command.index("--case-id") + 1]
        path = root / (case + ".json")
        before = json.loads(path.read_text())
        assert before["attempt_started"] is True and before["status"] == "DISPATCHED"
        assert command[command.index("--dispatch-token") + 1] == before["dispatch_token"]
        assert case not in calls
        calls.append(case)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1)  # before child marker
        if len(calls) == 2:
            raise OSError("fixture: child could not be launched")
        if len(calls) == 3:
            path.unlink()
            return subprocess.CompletedProcess(command, 1)
        if len(calls) == 4:
            before.update(status="RUNNING", child_started=True, source_time_advanced_count=0)  # stale initial marker is not a final count
            registration.write_json(path, before)
            return subprocess.CompletedProcess(command, 0)  # no terminal evidence
        before.update(deepcopy(records[case]))
        before["child_started"] = True
        registration.write_json(path, before)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(source_runner.subprocess, "run", dispatch)
    args = SimpleNamespace(config=str(tmp_path / "fixture.yaml"), source_model_id="modeldb_romani_2013_ca3_ca1")
    source_runner._dispatch_registration_cases(args, ledger_declaration, root, scratch)
    result = registration.analyze_directory(root)
    assert result["planned_cases"] == result["attempted_cases"] == len(calls) == 8
    assert result["completed_cases"] == 4 and result["source_advance_unknown_cases"] == 2
    assert result["status"] == "INCOMPLETE_SOURCE_EXECUTION"
    assert len(result["case_errors"]) == 4
    with pytest.raises(ValueError, match="already attempted"):
        source_runner._dispatch_registration_cases(args, ledger_declaration, root, scratch)
    assert len(calls) == 8
    config = deepcopy(configuration)
    config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"]["recording_mode"] = "solver_step_v2"
    monkeypatch.setattr(source_runner, "_verify_registration_inputs", registration.validate_declaration)
    monkeypatch.setattr(source_runner, "_load_registration_cell", lambda *a: pytest.fail("reattempt reached source"))
    with pytest.raises(ValueError, match="already attempted"):
        source_runner._run_registration_case(config, root, calls[0], args.source_model_id)


def test_missing_trace_retains_unknown_advance_and_full_denominator(ledger_declaration, tmp_path):
    registration.write_json(tmp_path / "declaration.json", ledger_declaration)
    records = analytic_ledger_traces(ledger_declaration)
    for trace in records[:-1]:
        registration.write_json(tmp_path / (trace["condition"]["case_id"] + ".json"), trace)
    result = registration.analyze_directory(tmp_path)
    assert result["planned_cases"] == 8 and result["completed_cases"] == 7
    assert result["attempted_cases"] == 7 and result["attempt_unknown_cases"] == 1
    assert result["source_advance_unknown_cases"] == 1
    assert result["case_statuses"][records[-1]["condition"]["case_id"]] == "MISSING"
    assert result["status"] == "INCOMPLETE_SOURCE_EXECUTION"


def test_coordinator_consumes_prechild_exit_slots_without_loading_source(
    ledger_declaration, source_runner, tmp_path, monkeypatch, configuration
):
    config = deepcopy(configuration)
    policy = deepcopy(ledger_declaration["policy"])
    policy["output_root"] = str(tmp_path / "results")
    config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"] = policy
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "build.log").write_text("source-free coordinator fixture\n")
    monkeypatch.setattr(source_runner, "load_config", lambda path: deepcopy(config))
    monkeypatch.setattr(source_runner, "_validate_registration_step_interface", lambda *a: None)
    monkeypatch.setattr(source_runner, "_stage_registration_source", lambda *a: {
        "source_root": str(tmp_path), "files": {}, "build": deepcopy(ledger_declaration["build"])})
    monkeypatch.setattr(source_runner, "_load_registration_cell", lambda *a: (
        None, ledger_declaration["site"], ledger_declaration["state_variables"], {}))
    monkeypatch.setattr(source_runner, "_verify_registration_inputs", registration.validate_declaration)
    monkeypatch.setattr(source_runner.subprocess, "check_output", lambda *a, **k: "tmpfs\n")
    calls = []
    def fail_before_child(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(source_runner.subprocess, "run", fail_before_child)
    args = SimpleNamespace(config=str(tmp_path / "fixture.yaml"), operation="source-registration",
                           registration_recording_mode="solver_step_v2", run_id="coordinator_fixture",
                           source_root=str(tmp_path), scratch_root=str(scratch),
                           source_model_id="modeldb_romani_2013_ca3_ca1")
    source_runner.source_registration_operation(args)
    result = json.loads((tmp_path / "results/coordinator_fixture/result.json").read_text())
    assert len(calls) == result["attempted_cases"] == 8
    assert result["completed_cases"] == result["source_time_advanced_count"] == 0
    assert result["status"] == "INCOMPLETE_SOURCE_EXECUTION"
    with pytest.raises(FileExistsError):
        source_runner.source_registration_operation(args)
    assert len(calls) == 8


@pytest.mark.parametrize("invalid_request", ["missing_ledger", "wrong_token", "outside_matrix"])
def test_direct_child_rejects_before_source_loading(
    ledger_declaration, source_runner, configuration, tmp_path, monkeypatch, invalid_request
):
    registration.write_json(tmp_path / "declaration.json", ledger_declaration)
    condition = registration.condition_matrix(ledger_declaration["policy"])[0]
    trace = source_runner._empty_registration_trace(ledger_declaration, condition)
    if invalid_request == "missing_ledger":
        trace.pop("step_ledger")
    registration.write_json(tmp_path / (condition["case_id"] + ".json"), trace)
    config = deepcopy(configuration)
    config["dynamics_ready_registry"]["neuron_rxd_teacher"]["source_membrane_charge_registration"]["recording_mode"] = "solver_step_v2"
    monkeypatch.setattr(source_runner, "_verify_registration_inputs", registration.validate_declaration)
    monkeypatch.setattr(source_runner, "_load_registration_cell", lambda *a: pytest.fail("invalid direct request reached source"))
    case_id = "outside" if invalid_request == "outside_matrix" else condition["case_id"]
    token = "unbound-token" if invalid_request == "wrong_token" else ""
    with pytest.raises(ValueError):
        source_runner._run_registration_case(config, tmp_path, case_id, "modeldb_romani_2013_ca3_ca1", token)
    retained = json.loads((tmp_path / (condition["case_id"] + ".json")).read_text())
    assert retained["attempt_started"] is False and retained["source_time_advanced_count"] == 0


def test_shared_interface_rejection_restores_fast_imem_without_advance(source_runner, monkeypatch):
    fast = [False]
    def fast_imem(value=None):
        previous = fast[0]
        if value is not None:
            fast[0] = value
        return previous
    cv = SimpleNamespace(use_fast_imem=fast_imem,
                         extra_scatter_gather=lambda *a: None,
                         extra_scatter_gather_remove=lambda *a: None)
    monkeypatch.setattr(source_runner, "h", SimpleNamespace(
        CVode=lambda: cv, fadvance=lambda: pytest.fail("preflight advanced source")))
    with pytest.raises(AttributeError):
        source_runner._validate_registration_step_interface(lambda x: SimpleNamespace(i_cap=0), {"segment_x": 0.5})
    assert fast[0] is False


def test_running_marker_never_supplies_a_final_zero_advance_count(ledger_declaration):
    records = analytic_ledger_traces(ledger_declaration)
    records[0].update(status="RUNNING", source_time_advanced_count=0)
    result = registration.analyze_traces(ledger_declaration, records)
    assert result["source_advance_unknown_cases"] == 1
    assert result["source_time_advanced_count"] == sum(r["source_time_advanced_count"] for r in records[1:])
    assert result["status"] == "INCOMPLETE_SOURCE_EXECUTION"


@pytest.mark.parametrize("identity_key", ["source_files_uid", "neuron_runtime_libraries", "python_executable_sha256"])
def test_solver_step_rejects_drift_from_fixed_source_identity(ledger_declaration, identity_key):
    registration.validate_declaration(ledger_declaration)
    broken = deepcopy(ledger_declaration)
    broken["build"]["identity"][identity_key] = "changed"
    broken.pop("declaration_uid")
    broken["declaration_uid"] = registration.content_uid(broken)
    with pytest.raises(ValueError, match="differs from the fixed source study"):
        registration.validate_declaration(broken)


def test_endpoint_residual_decomposition_retains_combined_readout_effects(ledger_declaration):
    trace = analytic_ledger_traces(ledger_declaration)[0]
    trace["samples"]["ina_mA_cm2"] = [0.0007] * len(trace["samples"]["time_ms"])
    result = registration.solver_step_balance(trace, ledger_declaration)
    terms = result["endpoint_residual_decomposition"]
    assert terms["endpoint_minus_assembled"]["terminal_pC"] == pytest.approx(0.0024)
    assert terms["effective_minus_assembled"]["terminal_pC"] == pytest.approx(0.0004)
    assert terms["reconciliation_error"]["max_abs_pC"] < 1e-14
    assert result["point_quadrature_decomposition"]["reconciliation_error"]["max_abs_pC"] < 1e-14
    assert "voltage/time/state" in terms["scope"]


@pytest.mark.parametrize("use_ledger", [False, True], ids=["endpoint_v1", "solver_step_v2"])
@pytest.mark.parametrize("mutation", [
    "not_attempted", "nonboolean_attempt", "float_advance_count",
    "unknown_attempt", "unknown_advance_count",
])
def test_analyzer_rejects_complete_attempt_metadata(
    declaration, ledger_declaration, use_ledger, mutation, tmp_path
):
    selected = ledger_declaration if use_ledger else declaration
    records = analytic_ledger_traces(selected) if use_ledger else traces_for(selected)
    trace = records[0]
    changes = {
        "not_attempted": {"attempt_started": False},
        "nonboolean_attempt": {"attempt_started": 1},
        "float_advance_count": {"source_time_advanced_count": float(trace["source_time_advanced_count"])},
        "unknown_attempt": {"attempt_started": None},
        "unknown_advance_count": {"source_time_advanced_count": None},
    }
    trace.update(changes[mutation])
    with pytest.raises(ValueError, match="attempt_started|advance"):
        registration.analyze_traces(selected, records)
    registration.write_json(tmp_path / "declaration.json", selected)
    for record in records:
        registration.write_json(tmp_path / (record["condition"]["case_id"] + ".json"), record)
    retained = tmp_path / "result.json"
    prior = b'{"status": "retained_result"}\n'
    retained.write_bytes(prior)
    with pytest.raises(ValueError, match="attempt_started|advance"):
        registration.analyze_directory(tmp_path)
    assert retained.read_bytes() == prior


@pytest.mark.parametrize(("status", "attempt", "advances"), [
    ("FAILED", 1, 0),
    ("FAILED", "true", 0),
    ("FAILED", True, 0.0),
    ("FAILED", True, True),
    ("FAILED", True, -1),
    ("FAILED", True, 1000000),
    ("FAILED", False, None),
    ("FAILED", False, 1),
    ("MISSING", None, 1),
    ("NOT_RUN", True, 0),
    ("NOT_RUN", None, None),
    ("DISPATCHED", False, 0),
    ("DISPATCHED", True, None),
    ("DISPATCHED", True, 1),
    ("RUNNING", False, 0),
    ("RUNNING", None, None),
])
def test_analyzer_rejects_incomplete_attempt_metadata(ledger_declaration, status, attempt, advances):
    records = analytic_ledger_traces(ledger_declaration)
    trace = registration.unavailable_trace(
        ledger_declaration, records[0]["condition"], "source-free invalid metadata fixture")
    trace.update(status=status, attempt_started=attempt, source_time_advanced_count=advances)
    records[0] = trace
    with pytest.raises(ValueError, match="attempt_started|advance"):
        registration.analyze_traces(ledger_declaration, records)


@pytest.mark.parametrize("use_ledger", [False, True], ids=["endpoint_v1", "solver_step_v2"])
@pytest.mark.parametrize(("status", "attempt", "advances"), [
    ("NOT_RUN", False, 0),
    ("DISPATCHED", True, 0),
    ("RUNNING", True, 0),
    ("RUNNING", True, 7),
    ("RUNNING", True, None),
    ("FAILED", True, 0),
    ("FAILED", True, 7),
    ("FAILED", True, None),
    ("FAILED", False, 0),  # Historical pre-child failure with no started attempt.
    ("FAILED", None, None),
    ("MISSING", None, None),
    ("MISSING", True, None),
    ("MISSING", None, 0),
    ("MISSING", True, 7),
])
def test_analyzer_preserves_valid_incomplete_attempt_accounting(
    declaration, ledger_declaration, use_ledger, status, attempt, advances
):
    selected = ledger_declaration if use_ledger else declaration
    records = analytic_ledger_traces(selected) if use_ledger else traces_for(selected)
    known_other_advances = sum(t["source_time_advanced_count"] for t in records[1:])
    trace = registration.unavailable_trace(
        selected, records[0]["condition"], "source-free incomplete metadata fixture")
    trace.update(status=status, attempt_started=attempt, source_time_advanced_count=advances)
    records[0] = trace
    result = registration.analyze_traces(selected, records)
    assert result["status"] == "INCOMPLETE_SOURCE_EXECUTION"
    assert result["planned_cases"] == 8 and result["completed_cases"] == 7
    assert result["attempted_cases"] == 7 + int(attempt is True)
    assert result["attempt_unknown_cases"] == int(attempt is None)
    assert result["not_attempted_cases"] == int(attempt is False)
    assert (result["attempted_cases"] + result["attempt_unknown_cases"]
            + result["not_attempted_cases"]) == 8
    unknown = advances is None or status == "RUNNING"
    assert result["source_advance_unknown_cases"] == int(unknown)
    assert result["source_time_advanced_count"] == known_other_advances + (0 if unknown else advances)


@pytest.mark.parametrize("use_ledger", [False, True], ids=["endpoint_v1", "solver_step_v2"])
def test_analyzer_preserves_complete_attempt_accounting(declaration, ledger_declaration, use_ledger):
    selected = ledger_declaration if use_ledger else declaration
    records = analytic_ledger_traces(selected) if use_ledger else traces_for(selected)
    result = registration.analyze_traces(selected, records)
    assert result["planned_cases"] == result["attempted_cases"] == result["completed_cases"] == 8
    assert result["attempt_unknown_cases"] == result["not_attempted_cases"] == 0
    assert result["source_advance_unknown_cases"] == 0
    assert result["source_time_advanced_count"] == sum(t["source_time_advanced_count"] for t in records)
