"""Numerical primitives of the target-free frozen-primary forecast adapter."""
import math

import pytest
import torch

from Libs.models.modules.native_frozen_rhs_solver import rk4_interval


def test_rk4_original_clock_and_convergence():
    initial = torch.tensor([1., 2.], dtype=torch.float64)
    rhs = lambda time, state: -.2*state
    expected = initial*math.exp(-.2*3.)
    coarse = rk4_interval(rhs, initial, 7., 10., 3)
    fine = rk4_interval(rhs, initial, 7., 10., 6)
    assert torch.max(torch.abs(fine-expected)) < torch.max(torch.abs(coarse-expected))/10
    assert torch.max(torch.abs(fine-expected)) < 1e-6
    clocks = []
    result = rk4_interval(lambda time, state:(clocks.append(time) or torch.ones_like(state)),
                         initial, 2e-6, 4e-6, 2)
    assert torch.allclose(result, initial+2e-6, atol=1e-15, rtol=0)
    assert len(clocks) == 8 and min(clocks) == 2e-6 and max(clocks) == pytest.approx(4e-6,abs=1e-20)


@pytest.mark.parametrize("left,right,steps", [(0., 0., 1), (1., 0., 1), (0., 1., 0), (0., 1., True), (0., float("inf"), 1)])
def test_rk4_rejects_invalid_clock_without_work(left, right, steps):
    def forbidden(time, state):
        raise AssertionError("Invalid clocks cannot call the RHS")
    with pytest.raises(ValueError, match="increasing finite interval"):
        rk4_interval(forbidden, torch.ones(1), left, right, steps)


def _exercise_retained_solver(tmp_path, monkeypatch, *, full_horizon):
    """Reuse an actual selected tiny belief; no source generation or teacher arrays."""
    import json
    import os
    from pathlib import Path

    from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
    from Libs.models.modules import native_frozen_rhs_solver as solver_module
    from Libs.utils.data import native_diffusion_data as native
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.native_intervention_decisions import compiled_action_burden
    from Libs.utils.exp.native_comparator_training import load_frozen_native_belief
    from Libs.utils.model import dynamics_store
    from Libs.utils.model.native_comparator_inputs import build_native_comparator_inputs
    from Libs.utils.model.native_scientific_source import selected_group_definitions

    location = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:
        pytest.skip("Requires the retained artificial selected-belief fixture, supplied explicitly")
    fixture = Path(location)
    cfg = load_config(str(fixture/"config.yaml"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure-RHS model forecast must not read future target arrays")
    monkeypatch.setattr(native, "load_records", forbidden)
    monkeypatch.setattr(dynamics_store, "load_zarr_rows", forbidden)
    monkeypatch.setattr(dynamics_store, "load_chunked_rows", forbidden)
    belief, binding = load_frozen_native_belief(cfg, fixture/"config.yaml", fixture/"native_fit/manifest.json", "cpu")
    registries = native.load_family_registries(cfg)
    declarations = selected_group_definitions(cfg, "validation")
    reference = json.loads((fixture/"public_cns_reference.json").read_text())
    settings = {"equation_authority":"frozen_primary_training_law", "maximum_step_hours":.25,
                "maximum_rhs_particle_action_evaluations":2_000_000}
    original_rk4 = solver_module.rk4_interval
    witnessed_calls = [0]
    def observed_interval(rhs, state, left, right, steps):
        def observed_rhs(now, value):
            witnessed_calls[0] += 1
            return rhs(now, value)
        return original_rk4(observed_rhs, state, left, right, steps)
    monkeypatch.setattr(solver_module, "rk4_interval", observed_interval)
    solver = NativeFrozenRHSSolver(cfg, settings)
    assert sum(parameter.numel() for parameter in solver.parameters()) == 0
    rows = []
    for group in declarations:
        registry = registries[group["family_uid"]]
        raw = native.load_joint_sources(registry, [group["posterior_uid"]])[group["posterior_uid"]]
        posterior = belief(raw)
        inputs = build_native_comparator_inputs(raw, posterior, belief.field_normalization, cfg)
        initial_before = inputs.posterior.support.initial_field.clone()
        parameters_before = inputs.posterior.support.physical_parameter_hypotheses["values"].clone()
        count = len(inputs.grid["time_hours"]) if full_horizon else 3
        before = witnessed_calls[0]
        result = solver.forecast_native(inputs, belief.field_normalization, query_count=count)
        fields, execution = result["particle_fields"], result["execution"]
        assert fields.shape == (1, 32, 3, count, 6, 6, 26) and torch.isfinite(fields).all()
        assert execution["time_values_hours"] == inputs.grid["time_hours"][:count]
        assert execution["forecast_query_count"] == count and execution["horizon_complete"] is full_horizon
        assert witnessed_calls[0]-before == execution["rhs_python_calls"]
        assert execution["rhs_tuple_action_state_evaluations"] == execution["rhs_python_calls"]*32*3
        assert execution["rhs_tuple_action_state_evaluations"] <= settings["maximum_rhs_particle_action_evaluations"]
        assert execution["source_provider_calls"] == 0 and execution["target_access"] is False
        assert torch.equal(inputs.posterior.support.initial_field, initial_before)
        assert torch.equal(inputs.posterior.support.physical_parameter_hypotheses["values"], parameters_before)
        expected_initial = initial_before[:, :, None].expand(-1, -1, 3, -1, -1, -1).double()
        assert torch.equal(fields[:, :, :, 0], expected_initial)
        assert torch.equal(result["particle_weights"], inputs.weights)
        assert fields.dtype == torch.float64
        if group["family_uid"] == "cns_long_horizon_world":
            program = inputs.programs["records"][0][0][0]["physical_program"]
            expected_free = torch.tensor(program["free_concentration_mol_per_l"][:count], dtype=torch.float64)[..., 0]
            actual_free = fields[0, 0, 0, ..., inputs.grid["species_id"].index("free_drug")]
            assert expected_free[1:].abs().sum() > 0
            assert torch.allclose(actual_free, expected_free, atol=1e-20, rtol=1e-10)

        validation = {}
        if full_horizon:
            before = witnessed_calls[0]
            limited = NativeFrozenRHSSolver(cfg, dict(settings,
                maximum_rhs_particle_action_evaluations=execution["rhs_tuple_action_state_evaluations"]-1))
            with pytest.raises(ValueError, match="exceeds the declared RHS work budget"):
                limited.forecast_native(inputs, belief.field_normalization)
            assert witnessed_calls[0] == before
            fine_step = min(settings["maximum_step_hours"]/2,
                            min(b-a for a, b in zip(inputs.grid["time_hours"], inputs.grid["time_hours"][1:]))/2)
            fine_solver = NativeFrozenRHSSolver(cfg, dict(settings, maximum_step_hours=fine_step,
                maximum_rhs_particle_action_evaluations=3_000_000))
            before = witnessed_calls[0]
            fine = fine_solver.forecast_native(inputs, belief.field_normalization)
            assert fine["execution"]["horizon_complete"] is True
            assert witnessed_calls[0]-before == fine["execution"]["rhs_python_calls"]
            difference = (fine["particle_fields"]-fields).abs().amax((0,1,2,3,4,5))
            validation.update(rhs_budget_refused_before_any_rhs_call=True,
                rhs_python_calls_witnessed=execution["rhs_python_calls"],
                refinement={"maximum_step_hours":fine_step, "execution":fine["execution"],
                    "scope":"continuous_CNS_RK4_substeps_with_original_discrete_auxiliary_clock_unchanged",
                    "difference_by_quantity":dict(zip(inputs.grid["species_id"], difference.tolist())),
                    "interpretation":"step_refinement_difference_not_an_error_bound_or_biological_validation"})
            del fine
            if group["family_uid"] == "electrophysiology_conductance_world":
                from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters, parameter_schema_from_config
                declared = raw["support"]["physical_parameter_hypotheses"]
                parameter_names = [row["name"] for row in declared["declarations"]]
                candidates = [dict(zip(parameter_names, row)) for row in declared["values"][0]]
                before_gain = candidates[0]["stimulation_gain"]
                candidates[0]["stimulation_gain"] *= 1.01
                producer = {"parameters":candidates, "scope":"artificial_parameter_response_fixture"}
                parameter_path = tmp_path/"perturbed_electrical_parameters.json"
                parameter_path.write_text(json.dumps(producer))
                perturbed_raw = bind_candidate_parameters(raw, group, producer, parameter_schema_from_config(cfg),
                    record_ref=str(parameter_path), producer_commit="artificial_parameter_response_fixture")
                perturbed_posterior = belief(perturbed_raw)
                perturbed_inputs = build_native_comparator_inputs(
                    perturbed_raw, perturbed_posterior, belief.field_normalization, cfg)
                changed = solver.forecast_native(perturbed_inputs, belief.field_normalization)
                assert torch.equal(changed["particle_fields"][:,1:], fields[:,1:])
                vm = next(i for i, row in enumerate(inputs.grid["quantity_profile"]["quantities"])
                          if row.get("family_uid") == group["family_uid"] and row["observable_name"] == "membrane_voltage")
                gain_response = float((changed["particle_fields"][:,0,...,vm]-fields[:,0,...,vm]).max())
                assert gain_response > 0
                assert torch.equal(inputs.posterior.support.physical_parameter_hypotheses["values"], parameters_before)
                validation["parameter_response"] = {"parameter":"stimulation_gain", "particle_index":0,
                    "before":before_gain, "after":candidates[0]["stimulation_gain"], "maximum_vm_increase_mV":gain_response,
                    "other_31_particle_fields_identical":True,
                    "scope":"new_artificial_parameter_table_bound_to_a_copy_original_artifacts_unchanged"}
                del perturbed_raw, perturbed_posterior, perturbed_inputs, changed

        names = list(cfg["native_public_cns_loss"]["endpoint_weights"])
        ci = inputs.grid["compartment_id"].index("isf")
        indices = [inputs.grid["species_id"].index(name) for name in names]
        mean = (fields[0]*inputs.weights[0, :, None, None, None, None, None]).sum(0)
        readouts = mean[..., ci, :][..., indices]
        raw_span = (readouts.amax(0)-readouts.amin(0)).amax((0, 1))
        serialization = (readouts-readouts.float().double()).abs().amax((0, 1, 2))
        low = readouts.new_tensor([reference["endpoints"][name]["low"] for name in names])
        high = readouts.new_tensor([reference["endpoints"][name]["high"] for name in names])
        endpoint_weights = readouts.new_tensor(list(cfg["native_public_cns_loss"]["endpoint_weights"].values()))
        normalized_harm = (((readouts-low)/(high-low)).clamp(0,1)*endpoint_weights).sum(-1)
        harm_span = float((normalized_harm.amax(0)-normalized_harm.amin(0)).max())
        burdens = [compiled_action_burden(record, group["family_uid"], cfg)
                   for record in inputs.programs["records"][0][0]]
        assert all(row["status"] == "defined" for row in burdens)
        action_costs = [cfg["native_public_cns_loss"]["intervention_weight"]*row["normalized_burden"] for row in burdens]
        row = {"family_uid":group["family_uid"], "execution":execution,
            "public_endpoint_raw_action_span":dict(zip(names, raw_span.tolist())),
            "float32_roundtrip_error":dict(zip(names, serialization.tolist())),
            "maximum_pointwise_reference_scaled_harm_gap":harm_span,
            "intervention_costs":action_costs, "intervention_cost_span":max(action_costs)-min(action_costs),
            "validation":validation, "horizon_complete":full_horizon,
            "interpretation":"artificial_primary_input_model_forecast_not_source_accuracy_or_biological_validation"}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del raw, posterior, inputs, fields, result
    assert len(rows) == 4
    receipt = {"scope":"artificial_pure_primary_rhs_model_forecast", "status":"software_full_horizon_pass" if full_horizon else "software_prefix_pass",
        "selected_belief_binding":binding, "groups":rows, "target_array_reads":0,
        "source_provider_calls":0, "optimizer_updates":0, "full_horizon_verified":full_horizon,
        "vm_only_scope":"unsupported_missing_primary_auxiliary_gating_state",
        "scientific_twin":"NOT_RUN", "formal_qualification":False}
    destination = Path(os.environ.get("NRNSIFT_RHS_EVIDENCE", str(tmp_path/"result.json")))
    destination.write_text(json.dumps(receipt, indent=2))


def test_frozen_primary_prefix_from_retained_inputs_without_targets(tmp_path, monkeypatch):
    _exercise_retained_solver(tmp_path, monkeypatch, full_horizon=False)


def test_frozen_primary_full_horizon_response_refinement_and_budget(tmp_path, monkeypatch):
    _exercise_retained_solver(tmp_path, monkeypatch, full_horizon=True)


def test_supported_physical_domains_are_validated_without_clipping():
    from Libs.models.modules.native_frozen_rhs_solver import validate_physical_forecast
    quantities = [{"quantity_id":"free_drug","signed":False},
                  {"quantity_id":"voltage","signed":True},
                  {"quantity_id":"occupancy","signed":False,"maximum":1.}]
    value = torch.tensor([[0., -65., .5]], dtype=torch.float64)
    support = torch.ones_like(value, dtype=torch.bool)
    validate_physical_forecast(value, support, quantities)
    for index, invalid, message in ((0, -1e-9, "free_drug"), (2, 1.00001, "occupancy"),
                                    (0, float("nan"), "Nonfinite")):
        changed = value.clone(); changed[0,index] = invalid
        with pytest.raises(FloatingPointError, match=message):
            validate_physical_forecast(changed, support, quantities)
        assert (torch.isnan(changed[0,index]) if math.isnan(invalid) else changed[0,index] == invalid)
    unsupported = support.clone(); unsupported[0,0] = False
    missing = value.clone(); missing[0,0] = float("nan")
    validate_physical_forecast(missing, unsupported, quantities)
    assert torch.isnan(missing[0,0])


@pytest.mark.parametrize("values,weights,target,expected", [
    ([0.,2.],[.25,.75],1.,.625),
    ([4.,-1.,2.],[.5,.2,.3],0.,1.82),
])
def test_deterministic_conditional_crps_uses_exact_unequal_tuple_weights(values,weights,target,expected):
    from Libs.utils.evaluation.native_comparator_evaluation import weighted_native_scores
    class Identity:
        def normalize(self, value, mask):
            return value
    samples=torch.tensor(values,dtype=torch.float64).reshape(-1,1,1,1,1,1,1)
    truth=torch.full((1,1,1,1,1),target,dtype=torch.float64)
    mask=torch.ones_like(truth,dtype=torch.bool)
    mass=torch.tensor(weights,dtype=torch.float64)
    score=weighted_native_scores(samples,mass,truth,mask,Identity(),[.5,.9],deterministic_conditional=True)
    assert score["crps"] == pytest.approx(expected,abs=1e-12)
    with pytest.raises(ValueError,match="two draws"):
        weighted_native_scores(samples,mass,truth,mask,Identity(),[.5])
    with pytest.raises(ValueError,match="one path"):
        weighted_native_scores(samples.expand(-1,2,-1,-1,-1,-1,-1),mass,truth,mask,Identity(),[.5],
                               deterministic_conditional=True)


def test_actual_frozen_rhs_cli_full_horizon_and_zero_fit(tmp_path):
    import copy
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    from Libs.utils.data.raw_data_registry import load_config

    location=os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:pytest.skip("Requires retained artificial selected-belief fixture")
    fixture=Path(location)
    root=Path(__file__).resolve().parents[2]
    config=load_config(str(fixture/"config.yaml"))
    current=load_config(str(root/"Libs/configs/exp/manuscript_native_protocol.yaml"))
    config["native_controlled_comparators"]["references"]["same_compiler_solver"]=copy.deepcopy(
        current["native_controlled_comparators"]["references"]["same_compiler_solver"])
    config_path=tmp_path/"solver_evaluation_config.yaml"
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    output=tmp_path/"solver_evaluation"
    command=[sys.executable,str(root/"Libs/exp/baselines/run_external_baseline_bridge.py"),
        "--protocol-scope","manuscript-native","--config",str(config_path),
        "--baseline-name","same_compiler_solver","--command","evaluate_baseline",
        "--conditioning-config",str(fixture/"config.yaml"),
        "--conditioning-manifest",str(fixture/"native_fit/manifest.json"),
        "--decision-reference",str(fixture/"public_cns_reference.json"),
        "--output-dir",str(output),"--evaluation-role","validation","--device","cpu"]
    completed=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (tmp_path/"solver_cli.log").write_text(completed.stdout+completed.stderr)
    assert completed.returncode==0,completed.stdout+completed.stderr
    report=json.loads((output/"native_comparator_evaluation.json").read_text())
    assert report["status"]=="complete",report
    assert report["declared_group_count"]==report["fully_supported_group_count"]==4
    assert report["binding"]["checkpoint_path"] is None
    assert report["binding"]["training_manifest"] is None and report["binding"]["optimizer_updates"]==0
    assert report["numerical_accuracy_qualified"] is False and report["reported_uncertainty_calibrated"] is False
    assert report["scientific_twin"]=="NOT_RUN" and report["formal_qualification"] is False
    counts={"electrophysiology_conductance_world":7,"calcium_synaptic_world":8,
        "molecular_pharmacological_causal_world":7,"cns_long_horizon_world":28}
    for row in report["groups"]:
        assert row["status"]=="defined" and row["particle_count"]==32 and row["conditional_draws_per_particle"]==1
        assert len(row["posterior_weights"])==32 and math.isclose(sum(row["posterior_weights"]),1.,abs_tol=1e-6)
        assert row["execution"]["horizon_complete"] is True
        assert row["execution"]["forecast_query_count"]==counts[row["family_uid"]]
        assert row["execution"]["source_provider_calls"]==0 and row["execution"]["target_access"] is False
        assert row["public_decision"]["selected_regimen_uid"] in row["regimen_uids"]
        assert math.isfinite(row["metrics"]["crps"])
    rejected=subprocess.run(command+["--training-manifest",str(fixture/"native_fit/manifest.json")],
                            cwd=root,text=True,capture_output=True)
    assert rejected.returncode!=0 and "no transition checkpoint" in rejected.stderr
    receipt={"scope":"actual_artificial_frozen_rhs_cli","cli_exit_code":completed.returncode,
        "transition_checkpoint_override_rejected":True,"source_provider_calls":0,"optimizer_updates":0,
        "report":report,"scientific_twin":"NOT_RUN","formal_qualification":False}
    Path(os.environ.get("NRNSIFT_RHS_EVIDENCE",str(tmp_path/"result.json"))).write_text(json.dumps(receipt,indent=2))


def test_frozen_rhs_catalog_is_zero_fit_and_uses_shared_public_reference():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile,command_catalog
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    selected,profile=training_profile(config,"same_compiler_solver",41)
    assert profile["training_manifest"] is None and profile["additional_optimizer_updates"]==0
    assert profile["conditional_paths_per_particle"]==1
    commands=command_catalog(config,[profile])
    rows={row["command_uid"]:row for row in commands}
    evaluation=rows[profile["profile_id"]+":evaluate"]
    assert "public_cns_reference" in rows and "public_cns_reference" in evaluation["dependencies"]
    assert not any(uid.endswith((":fit",":calibrate")) for uid in rows)
    assert "--training-manifest" not in evaluation["argv"] and "--residual-artifact" not in evaluation["argv"]
    assert evaluation["argv"][evaluation["argv"].index("--device")+1]=="cpu"
    assert "native_wam_fraction_100_seed_41:fit" in evaluation["dependencies"]
    assert "same_compiler_solver" not in selected["native_controlled_comparators"]["conditional_mean_methods"]


def test_deterministic_crps_normalizes_float32_roundoff_without_mutating_weights():
    from Libs.utils.evaluation.native_comparator_evaluation import weighted_native_scores
    class Identity:
        def normalize(self,value,mask):return value
    weights=torch.tensor([.2,.8],dtype=torch.float32)
    original=weights.clone()
    assert weights.double().sum()>1
    samples=torch.ones(2,1,1,1,1,1,1,dtype=torch.float64)
    target=torch.zeros(1,1,1,1,1,dtype=torch.float64)
    mask=torch.ones_like(target,dtype=torch.bool)
    result=weighted_native_scores(samples,weights,target,mask,Identity(),[.5,.9],deterministic_conditional=True)
    assert result["crps"]==pytest.approx(1.,abs=1e-14)
    assert result["normalized_rmse"]==pytest.approx(1.,abs=1e-14)
    assert torch.equal(weights,original)


@pytest.fixture(scope="module")
def retained_continuation_cases():
    """Public retained candidate/posterior inputs only; no new source worlds."""
    import os
    from pathlib import Path
    from Libs.utils.data import native_diffusion_data as native
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.exp.native_comparator_training import load_frozen_native_belief
    from Libs.utils.model.native_comparator_inputs import build_native_comparator_inputs
    from Libs.utils.model.native_scientific_source import selected_group_definitions

    location = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:
        pytest.skip("Requires retained artificial selected-belief fixture")
    fixture = Path(location)
    config = load_config(str(fixture/"config.yaml"))
    belief, _ = load_frozen_native_belief(config, fixture/"config.yaml", fixture/"native_fit/manifest.json", "cpu")
    registries = native.load_family_registries(config)
    cases = {}
    for group in selected_group_definitions(config, "validation"):
        raw = native.load_joint_sources(registries[group["family_uid"]], [group["posterior_uid"]])[group["posterior_uid"]]
        inputs = build_native_comparator_inputs(raw, belief(raw), belief.field_normalization, config)
        cases[group["family_uid"]] = (config, inputs, belief.field_normalization)
    assert len(cases) == 4
    return cases


_CONTINUATION_FAMILIES = [
    "electrophysiology_conductance_world", "calcium_synaptic_world",
    "molecular_pharmacological_causal_world", "cns_long_horizon_world",
]


def _continuation_solver(config):
    from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
    return NativeFrozenRHSSolver(config, dict(
        equation_authority="frozen_primary_training_law", maximum_step_hours=.25,
        maximum_rhs_particle_action_evaluations=2_000_000))


@pytest.mark.parametrize("family", _CONTINUATION_FAMILIES)
def test_continuation_matches_uninterrupted_original_clocks(
        retained_continuation_cases, monkeypatch, family):
    import importlib.util
    import json
    import os
    from pathlib import Path
    from Libs.models.modules import native_frozen_rhs_solver as module
    from Libs.utils.data import native_diffusion_data as native
    from Libs.utils.model import dynamics_store

    prior_path = os.environ.get("NRNSIFT_RHS_PRIOR_MODULE")
    prior = None
    if prior_path:
        # Optional one-time refactor witness injected by the bounded runner.
        spec = importlib.util.spec_from_file_location("prior_native_rhs_solver", prior_path)
        prior = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prior)
    config, inputs, normalization = retained_continuation_cases[family]
    def forbidden(*args, **kwargs):
        raise AssertionError("Continuation cannot read target arrays or rebuild its initial guidance")
    monkeypatch.setattr(native, "load_records", forbidden)
    monkeypatch.setattr(dynamics_store, "load_zarr_rows", forbidden)
    monkeypatch.setattr(dynamics_store, "load_chunked_rows", forbidden)
    settings = dict(equation_authority="frozen_primary_training_law", maximum_step_hours=.25,
                    maximum_rhs_particle_action_evaluations=2_000_000)
    initial_before = inputs.posterior.support.initial_field.clone()
    old = prior.NativeFrozenRHSSolver(config, settings).forecast_native(inputs, normalization) if prior else None
    solver = _continuation_solver(config)
    full = solver.forecast_native(inputs, normalization)
    if old is not None:
        assert torch.equal(full["particle_fields"], old["particle_fields"])
        for key in ("rhs_python_calls", "auxiliary_tuple_action_updates", "quantity_readout_tuple_action_evaluations"):
            assert full["execution"][key] == old["execution"][key]
    state = solver.initialize_continuation(inputs, normalization)
    monkeypatch.setattr(module, "FrozenNativePhysicsGuidance", forbidden)
    observed_intervals = []
    original_interval = module.rk4_interval
    def observe(rhs, current, left, right, steps):
        observed_intervals.append((left, right, steps))
        return original_interval(rhs, current, left, right, steps)
    monkeypatch.setattr(module, "rk4_interval", observe)
    chunks, execution = [], []
    for count in range(2, len(inputs.grid["time_hours"])+1):
        previous_time = state.current_time_hours
        before = len(observed_intervals)
        chunk = solver.advance_continuation(state, query_count=count)
        assert all(left >= previous_time for left, _, _ in observed_intervals[before:])
        assert state.query_index == count-1
        assert state.current_time_hours == inputs.grid["time_hours"][count-1]
        assert chunk["execution"]["start_query_index"] == (0 if count == 2 else count-1)
        chunks.append(chunk["particle_fields"]); execution.append(chunk["execution"])
        free_index = inputs.grid["species_id"].index("free_drug")
        assert torch.equal(state.cns_state[..., -1],
            chunk["particle_fields"][:, :, :, -1, ..., free_index].reshape(state._guide.n, -1))
        if state.applied_events is not None:
            assert torch.equal(state.applied_events, state.event_times <= state.current_time_hours)
    joined = torch.cat(chunks, 3)
    assert torch.equal(joined, full["particle_fields"])
    assert sum(row["rhs_python_calls"] for row in execution) == state.rhs_python_calls == full["execution"]["rhs_python_calls"]
    assert sum(row["auxiliary_tuple_action_updates"] for row in execution) == full["execution"]["auxiliary_tuple_action_updates"]
    assert len(observed_intervals) == len(state._intervals)
    assert torch.equal(inputs.posterior.support.initial_field, initial_before)
    evidence = {"family_uid": family, "shape": list(joined.shape),
        "original_time_values_hours": inputs.grid["time_hours"],
        "prior_solver_bitwise_equal": True if old is not None else None, "split_uninterrupted_bitwise_equal": True,
        "current_free_state_matches_emitted_field": True,
        "advance_calls": len(chunks), "rhs_python_calls": state.rhs_python_calls,
        "integration_intervals": len(observed_intervals),
        "auxiliary_updates": full["execution"]["auxiliary_tuple_action_updates"],
        "applied_dose_events": int(state.applied_events.sum()) if state.applied_events is not None else None,
        "guidance_reinitialized_after_start": False, "target_array_access": False}
    print(json.dumps(evidence), flush=True)
    destination = os.environ.get("NRNSIFT_RHS_CONTINUATION_EVIDENCE")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else {
            "scope": "retained_artificial_fixed_program_continuation",
            "groups": [], "source_provider_calls": 0, "optimizer_updates": 0,
            "scientific_twin": "NOT_RUN", "formal_qualification": False}
        report["groups"].append(evidence)
        path.write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize("family", _CONTINUATION_FAMILIES)
def test_continuation_consumes_current_cns_auxiliary_and_dose_history(retained_continuation_cases, family):
    from dataclasses import replace
    config, inputs, normalization = retained_continuation_cases[family]
    solver = _continuation_solver(config)
    current = solver.initialize_continuation(inputs, normalization)
    solver.advance_continuation(current, query_count=3)
    def fork():
        return replace(current, cns_state=current.cns_state.clone(),
            auxiliary_state=current.auxiliary_state.clone() if current.auxiliary_state is not None else None,
            applied_events=current.applied_events.clone() if current.applied_events is not None else None)
    unchanged, changed_cns = fork(), fork()
    changed_cns.cns_state[..., 1] *= 1.01
    expected = solver.advance_continuation(unchanged, query_count=4)["particle_fields"]
    actual = solver.advance_continuation(changed_cns, query_count=4)["particle_fields"]
    assert not torch.equal(expected, actual)
    if current.auxiliary_state is not None:
        changed_aux = fork()
        column = {"electrophysiology_conductance_world": 1, "calcium_synaptic_world": 2,
                  "molecular_pharmacological_causal_world": 3}[family]
        changed_aux.auxiliary_state[:, column] *= .99
        actual = solver.advance_continuation(changed_aux, query_count=4)["particle_fields"]
        assert not torch.equal(expected, actual)
    else:
        missing_history = fork()
        assert missing_history.applied_events.any()
        missing_history.applied_events.zero_()
        before = missing_history.rhs_python_calls
        with pytest.raises(ValueError, match="applied dose history"):
            solver.advance_continuation(missing_history, query_count=4)
        assert missing_history.rhs_python_calls == before


def test_continuation_invalid_requests_state_failure_and_cumulative_budget(retained_continuation_cases, monkeypatch):
    from dataclasses import replace
    family = "electrophysiology_conductance_world"
    config, inputs, normalization = retained_continuation_cases[family]
    solver = _continuation_solver(config)
    state = solver.initialize_continuation(inputs, normalization)
    solver.advance_continuation(state, query_count=3)
    original = state.cns_state.clone()
    before = state.rhs_python_calls
    for count in (True, 0, 1, 2, 3, len(inputs.grid["time_hours"])+1):
        with pytest.raises(ValueError, match="later original query"):
            solver.advance_continuation(state, query_count=count)
    assert state.rhs_python_calls == before and torch.equal(state.cns_state, original)
    for altered, error in (
        (replace(state, cns_state=state.cns_state[..., :-1]), "CNS state shape"),
        (replace(state, auxiliary_state=None), "auxiliary state shape"),
        (replace(state, query_index=0), "clock and cumulative"),
    ):
        with pytest.raises(ValueError, match=error):
            solver.advance_continuation(altered, query_count=4)
    invalid = replace(state, cns_state=state.cns_state.clone())
    invalid.cns_state[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="finite and nonnegative"):
        solver.advance_continuation(invalid, query_count=4)
    with pytest.raises(ValueError, match="different solver"):
        _continuation_solver(config).advance_continuation(state, query_count=4)
    all_work = 4*sum(steps for _, _, steps in state._intervals)*state._guide.n
    limited = _continuation_solver(config)
    limited.maximum_rhs_work = all_work-1
    budgeted = limited.initialize_continuation(inputs, normalization)
    limited.advance_continuation(budgeted, query_count=3)
    before = budgeted.rhs_python_calls
    with pytest.raises(ValueError, match="exceeds the declared RHS work budget"):
        limited.advance_continuation(budgeted, query_count=len(inputs.grid["time_hours"]))
    assert budgeted.rhs_python_calls == before
    def fail(*args):
        raise FloatingPointError("intentional invalid numerical stage")
    monkeypatch.setattr(solver, "_derivative", fail)
    with pytest.raises(FloatingPointError, match="intentional"):
        solver.advance_continuation(state, query_count=4)
    assert state.failed
    with pytest.raises(ValueError, match="Failed continuation"):
        solver.advance_continuation(state, query_count=4)


def test_continuation_preserves_vm_only_rejection(retained_continuation_cases, monkeypatch):
    from types import SimpleNamespace
    from Libs.models.modules import native_frozen_rhs_solver as module
    config, inputs, normalization = retained_continuation_cases["electrophysiology_conductance_world"]
    monkeypatch.setattr(module, "FrozenNativePhysicsGuidance", lambda *args: SimpleNamespace(vm_only=True))
    with pytest.raises(ValueError, match="Vm-only tuples lack"):
        _continuation_solver(config).initialize_continuation(inputs, normalization)


# Reuse the retained public-support loader; it does not load a fitted checkpoint.
from Libs.tests.test_native_mechanistic_residual import cases, FAMILIES


def _future_setup(cases, family, *, maximum=None):
    import copy
    from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
    from Libs.utils.data.raw_data_registry import load_config
    cfg, inputs, norm = cases["validation", family]
    cfg = copy.deepcopy(cfg)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    cfg["native_future_controls"] = copy.deepcopy(current["native_future_controls"])
    cfg["schemas"]["native_future_controls"] = copy.deepcopy(current["schemas"]["native_future_controls"])
    settings = copy.deepcopy(cfg["native_controlled_comparators"]["references"]["same_compiler_solver"]["hyperparameters"])
    if maximum is not None:
        settings["maximum_rhs_particle_action_evaluations"] = maximum
    engine = NativeFrozenRHSSolver(cfg, settings)
    return engine, engine.initialize_continuation(inputs, norm), inputs, norm


def _future_snapshot(state):
    import copy
    names = ("cns_state", "auxiliary_state", "event_times", "event_doses", "applied_events",
             "_event_doses_mg_per_kg", "_weights", "_log_weights")
    return {"tensors": {name: None if getattr(state, name) is None else getattr(state, name).clone() for name in names},
        "query": state.query_index, "interval": state.interval_index, "calls": state.rhs_python_calls,
        "binding": copy.deepcopy(state._filter_binding), "program_context": copy.deepcopy(state.program_context),
        "revision": state.filter_revision, "observation_uids": state.filter_observation_uids}


def _assert_future_snapshot(state, before):
    for name, value in before["tensors"].items():
        assert getattr(state, name) is None if value is None else torch.equal(getattr(state, name), value)
    assert (state.query_index, state.interval_index, state.rhs_python_calls) == (before["query"], before["interval"], before["calls"])
    assert state._filter_binding == before["binding"] and state.program_context == before["program_context"]
    assert state.filter_revision == before["revision"] and state.filter_observation_uids == before["observation_uids"]


def _future_decision_point(state):
    guide = state._guide
    if state._registered_stimulus is None:
        # Pick only from public event metadata, before inspecting any outcomes.
        for k in range(1, len(state._times)-2):
            prefix = (state._event_doses_mg_per_kg*(state.event_times <= state._times[k])).reshape(1, 32, guide.a, -1).sum(-1)[0, 0]
            if (prefix > 0).all() and (state.event_times > state._times[k]).logical_and(torch.isfinite(state.event_times)).any():
                return k, int(prefix.argmin())
        raise AssertionError("Retained event library has no nonzero common-prefix/future decision point")
    offset = 0 if guide.bind_source_intervals and guide.family != "molecular_pharmacological_causal_world" else 1
    controls = guide.stimulus.reshape(1, 32, guide.a, -1)[0, 0]
    for k in range(1, len(state._times)-2):
        prefix = (controls[:, offset:offset+k].abs()*guide.clock.diff()[:k]).sum(-1)
        first = k+offset
        if (prefix > 0).all() and any(not torch.equal(controls[0, first:], row[first:]) for row in controls[1:]):
            return k, int(prefix.argmin())
    raise AssertionError("Retained stimulation library has no nonzero prefix and different future controls")


@pytest.mark.parametrize("family", FAMILIES)
def test_future_controls_execute_and_rebranch_from_selected_composite_history(cases, family, monkeypatch, tmp_path):
    import json
    import os
    from pathlib import Path
    from Libs.utils.model import native_scientific_source as source
    from Libs.utils.model.native_comparator_physics import auxiliary_step
    engine, state, inputs, norm = _future_setup(cases, family)
    k, executed = _future_decision_point(state)
    def forbidden(*args, **kwargs):
        raise AssertionError("Future planning cannot generate a source or load future teacher targets")
    monkeypatch.setattr(source, "draw_group", forbidden)
    monkeypatch.setattr("Libs.utils.data.native_diffusion_data.load_records", forbidden)
    monkeypatch.setattr("Libs.utils.model.dynamics_store.load_zarr_rows", forbidden)
    monkeypatch.setattr("Libs.utils.model.dynamics_store.load_chunked_rows", forbidden)
    engine.advance_continuation(state, query_count=k+1)
    before = _future_snapshot(state)
    if state.event_times is not None:
        event = min(float(t) for t in state.event_times.flatten() if state.current_time_hours < t < state._times[-1])
        stop = next(index for index, t in enumerate(state._times) if t >= event)
    else:
        stop = k+1
    assert stop < len(state._times)-1
    reference = engine.fork_continuation(state)
    expected = engine.advance_continuation(reference, query_count=stop+1)["particle_fields"]
    old_spent = state._work_ledger.spent
    planned = engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[executed])
    assert planned._guide is not state._guide and planned._guide.inputs is None and planned._guide.planning_only
    assert planned._guide.free is planned._guide.occupancy is None
    assert planned._work_ledger is state._work_ledger and planned._work_ledger.spent == old_spent
    assert planned._work_ledger.planning_active and planned._guide.a == len(inputs.regimen_uids)
    original_fields = engine.current_particle_fields(state)[:, :, executed:executed+1]
    assert torch.equal(engine.current_particle_fields(planned), original_fields.expand_as(engine.current_particle_fields(planned)))
    for name, values in state._guide.parameters.items():
        expected_params = values.reshape(1, 32, state._guide.a)[:, :, executed:executed+1].expand(1, 32, planned._guide.a)
        assert torch.equal(planned._guide.parameters[name].reshape_as(expected_params), expected_params)
    assert torch.equal(planned._log_weights, state._log_weights) and planned.filter_observation_uids == state.filter_observation_uids
    assert all(uid not in inputs.regimen_uids for uid in planned._filter_binding["regimen_uids"])
    if family == "molecular_pharmacological_causal_world":
        expected_aux = auxiliary_step(family, planned.auxiliary_state, planned._guide.parameters,
            planned._guide.clock[k+1]-planned._guide.clock[k], planned._guide.stimulus[:, k+1])
        no_feedback = dict(planned._guide.parameters, feedback_rate=torch.zeros_like(planned._guide.parameters["feedback_rate"]))
        counterfactual = auxiliary_step(family, planned.auxiliary_state, no_feedback,
            planned._guide.clock[k+1]-planned._guide.clock[k], planned._guide.stimulus[:, k+1])
        assert not torch.equal(expected_aux, counterfactual)
    if planned.event_times is not None:
        selected_past = state.applied_events.reshape(1, 32, state._guide.a, -1)[:, :, executed].sum(-1)
        assert torch.equal(planned.applied_events.reshape(1, 32, planned._guide.a, -1).sum(-1), selected_past[..., None].expand(1, 32, planned._guide.a))
        assert not ((planned.event_times <= planned.current_time_hours) & ~planned.applied_events).any()
    result = engine.advance_continuation(planned, query_count=stop+1)
    actual = result["particle_fields"]
    assert torch.equal(actual[:, :, executed], expected[:, :, executed])
    assert any(not torch.equal(actual[:, :, 0], actual[:, :, ai]) for ai in range(1, planned._guide.a))
    if family == "molecular_pharmacological_causal_world":
        assert torch.equal(planned.auxiliary_state, expected_aux)
    _assert_future_snapshot(state, before)
    ci = (executed+1) % planned._guide.a
    parent_uid = planned._filter_binding["regimen_uids"][ci]
    selected_fields = engine.current_particle_fields(planned)[:, :, ci:ci+1]
    first_before = _future_snapshot(planned)
    second = engine.branch_future_controls(planned, inputs, norm, executed_regimen_uid=parent_uid,
                                          candidate_regimen_uids=(inputs.regimen_uids[ci],))
    assert second._guide.a == 1 and second._work_ledger is state._work_ledger
    assert second._planning_origin["parent_model_regimen_uid"] == parent_uid
    assert second._filter_binding["regimen_uids"][0] != parent_uid
    assert second._registered_program_uids == state._registered_program_uids
    assert torch.equal(engine.current_particle_fields(second), selected_fields)
    assert torch.equal(second._log_weights, state._log_weights)
    second_result = engine.advance_continuation(second, query_count=stop+2)
    assert torch.isfinite(second_result["particle_fields"]).all()
    _assert_future_snapshot(planned, first_before)
    _assert_future_snapshot(state, before)
    for branch in (planned, second):
        for budget in branch._planning_origin["intervention_budgets"]:
            assert budget["prefix"] > 0 and budget["total"] <= budget["library_maximum"]
        with pytest.raises(ValueError, match="planning"):
            engine.filter_context(branch)
        with pytest.raises(ValueError, match="planning"):
            engine.update_filter_weights(branch, branch._log_weights, query_index=branch.query_index,
                parent_revision=branch.filter_revision, observation_uid="must_not_admit")
        with pytest.raises(ValueError, match="guidance"):
            branch._guide(torch.empty(0))
    evidence = {"family": family, "scope": "model_side_future_controls_software", "source_calls": 0,
        "source_readback": False, "scientific_twin": "NOT_RUN", "first_query": k,
        "second_query": stop, "executed_original_regimen_uid": inputs.regimen_uids[executed],
        "particle_count": 32, "first_candidate_count": planned._guide.a, "second_candidate_count": 1,
        "unchanged_suffix_exact": True, "different_suffixes_change_later_fields": True,
        "selected_composite_carry_preserved_on_rebranch": True,
        "parent_model_regimen_uid": parent_uid, "second_model_regimen_uid": second._filter_binding["regimen_uids"][0],
        "shared_actual_rhs_work": state._work_ledger.spent, "rhs_ceiling": state._work_ledger.maximum,
        "first_budgets": planned._planning_origin["intervention_budgets"],
        "second_budgets": second._planning_origin["intervention_budgets"],
        "physical_activation": "NOT_IMPLEMENTED", "cross_switch_H3": "NOT_IMPLEMENTED"}
    root = Path(os.environ.get("NRNSIFT_FUTURE_CONTROLS_EVIDENCE", str(tmp_path)))
    root.mkdir(exist_ok=True, parents=True)
    (root/(family+".json")).write_text(json.dumps(evidence, indent=2)+"\n")


@pytest.mark.parametrize("kind", ["source", "units", "route", "prefix", "parameters", "static_control", "vm_only", "unknown_candidate"])
def test_future_invalid_proposals_reject_without_work_or_planning_activation(cases, kind):
    import copy
    from dataclasses import replace
    engine, state, inputs, norm = _future_setup(cases, FAMILIES[0])
    engine.advance_continuation(state, query_count=2)
    proposal = inputs
    kwargs = {"executed_regimen_uid": inputs.regimen_uids[0]}
    if kind == "source":
        proposal = replace(inputs, posterior_uid="different_origin")
    elif kind == "units":
        grid = copy.deepcopy(inputs.grid); grid["query_time"]["unit"] = "hour"
        proposal = replace(inputs, grid=grid)
    elif kind == "route":
        programs = copy.deepcopy(inputs.programs)
        programs["records"][0][0][0]["action"]["route"] = "changed_route"
        proposal = replace(inputs, programs=programs)
    elif kind == "prefix":
        state._filter_executed_regimen_uid = inputs.regimen_uids[1]
    elif kind == "parameters":
        state._guide.parameters[next(iter(state._guide.parameters))] += 1
    elif kind == "static_control":
        state._guide.stimulus[:, 0] += 1
    elif kind == "vm_only":
        state._guide.vm_only = True
    else:
        kwargs["candidate_regimen_uids"] = ("unknown",)
    before = _future_snapshot(state); spent = state._work_ledger.spent
    with pytest.raises(ValueError):
        engine.branch_future_controls(state, proposal, norm, **kwargs)
    assert not state._work_ledger.planning_active and state._work_ledger.spent == spent
    _assert_future_snapshot(state, before)


def test_future_shared_budget_counts_old_action_width_once_and_fork_cannot_reset(cases):
    engine, template, inputs, norm = _future_setup(cases, FAMILIES[0])
    k = 1
    past_calls = 4*sum(steps for _, right, steps in template._intervals if right <= template._times[k])
    later_calls = 4*sum(steps for left, _, steps in template._intervals if left >= template._times[k])
    ceiling = past_calls*96+later_calls*32
    engine, state, inputs, norm = _future_setup(cases, FAMILIES[0], maximum=ceiling)
    engine.advance_continuation(state, query_count=k+1)
    spent = state._work_ledger.spent
    with pytest.raises(ValueError, match="shared actual RHS"):
        engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0])
    assert not state._work_ledger.planning_active and state._work_ledger.spent == spent
    branch = engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0],
                                          candidate_regimen_uids=(inputs.regimen_uids[0],))
    fork = engine.fork_continuation(branch)
    engine.advance_continuation(branch, query_count=len(state._times))
    assert state._work_ledger.spent == ceiling
    assert state._work_ledger.spent != branch.rhs_python_calls*branch._guide.n
    with pytest.raises(ValueError, match="RHS work budget"):
        engine.advance_continuation(fork, query_count=len(state._times))
    assert state._work_ledger.spent == ceiling


def test_future_molecular_drive_budget_cannot_supply_physical_dose_admission(cases):
    engine, state, inputs, norm = _future_setup(cases, FAMILIES[2])
    engine.advance_continuation(state, query_count=2)
    with pytest.raises(ValueError, match="no delivered-dose mapping"):
        engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0],
                                      require_delivered_dose=True)
    assert not state._work_ledger.planning_active


def test_future_planning_branch_rejects_old_residual_conditioning_before_network(cases):
    from Libs.models.modules.native_mechanistic_residual import NativeMechanisticResidual, NativeResidualContinuation
    engine, state, inputs, norm = _future_setup(cases, FAMILIES[0])
    engine.advance_continuation(state, query_count=2)
    branch = engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0])
    owner = object()
    continuation = NativeResidualContinuation(branch, torch.empty(0), torch.empty(0), norm.identity, owner)
    with pytest.raises(ValueError, match="cross-switch H3"):
        NativeMechanisticResidual.advance_continuation(owner, continuation, norm, query_count=3)


def test_future_absolute_stimulation_integral_uses_signed_current_magnitude():
    from Libs.utils.model.native_future_controls import _row_integral
    assert _row_integral([-2., 1.], [3., 4.]) == 10.



def test_future_stitch_cannot_double_registered_input_burden(cases):
    """Crossing artificial waveforms use only the retained library's zero/max levels."""
    import copy
    from dataclasses import replace
    from Libs.utils.data.native_stimulation import compile_stimulation_record, FORCING_ARRAY_NAMES
    engine, _, inputs, norm = _future_setup(cases, FAMILIES[0])
    programs = copy.deepcopy(inputs.programs)
    arrays = {name: torch.as_tensor(programs[name]).clone() for name in FORCING_ARRAY_NAMES}
    magnitude = float(arrays["electrical_current_uA"].abs().max())
    assert magnitude > 0 and len(inputs.grid["time_hours"]) == 7
    for particle in programs["records"][0]:
        for ai, record in enumerate(particle):
            action = copy.deepcopy(record["action"])
            wave = action["stimulation_or_environment"]["magnitude_or_program"]
            wave["values"] = [magnitude if 2*ai <= index < 2*ai+2 else 0. for index in range(7)]
            compiled = compile_stimulation_record(action, record["stimulation_grid"], record["stimulation_binding"])
            record.update(compiled)
            record.pop("source_action_timing", None)
    for ji, particle in enumerate(programs["records"][0]):
        for ai, record in enumerate(particle):
            for name in FORCING_ARRAY_NAMES:
                arrays[name][0, ji, ai] = torch.as_tensor(record["compiled_program"]["arrays"][name])
    programs.update(arrays)
    artificial = replace(inputs, programs=programs)
    state = engine.initialize_continuation(artificial, norm)
    engine.advance_continuation(state, query_count=3)
    before = _future_snapshot(state)
    with pytest.raises(ValueError, match="remaining registered-library intervention budget"):
        engine.branch_future_controls(state, artificial, norm, executed_regimen_uid=inputs.regimen_uids[0],
                                      candidate_regimen_uids=(inputs.regimen_uids[1],))
    assert not state._work_ledger.planning_active
    _assert_future_snapshot(state, before)


def test_future_failed_rhs_call_remains_charged_and_failed(cases, monkeypatch):
    engine, state, inputs, norm = _future_setup(cases, FAMILIES[0])
    engine.advance_continuation(state, query_count=2)
    branch = engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0])
    before = state._work_ledger.spent
    def failed_rhs(*args, **kwargs):
        raise FloatingPointError("injected_rhs_failure")
    monkeypatch.setattr(engine, "_terms", failed_rhs)
    with pytest.raises(FloatingPointError, match="injected_rhs_failure"):
        engine.advance_continuation(branch, query_count=3)
    assert branch.failed and state._work_ledger.spent == before+branch._guide.n
    with pytest.raises(ValueError, match="Failed continuation"):
        engine.advance_continuation(branch, query_count=3)
    assert state._work_ledger.spent == before+branch._guide.n



@pytest.mark.parametrize("source_intervals", [True, False])
def test_future_stimulation_keeps_source_left_and_legacy_right_endpoint_semantics(cases, source_intervals):
    from Libs.utils.model.native_comparator_physics import auxiliary_step
    engine, _, inputs, norm = _future_setup(cases, FAMILIES[0])
    engine.config["native_observable_bridge"]["bind_source_update_intervals"] = source_intervals
    state = engine.initialize_continuation(inputs, norm)
    engine.advance_continuation(state, query_count=2)
    branch = engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0])
    first = 1 if source_intervals else 2
    assert branch._planning_origin["first_future_control_index"] == first
    original = state._guide.stimulus.reshape(1, 32, 3, -1)
    compiled = branch._guide.stimulus.reshape(1, 32, 3, -1)
    assert torch.equal(compiled[..., :first], original[:, :, 0:1, :first].expand_as(compiled[..., :first]))
    assert torch.equal(compiled[..., first:], original[..., first:])
    expected = auxiliary_step(FAMILIES[0], branch.auxiliary_state, branch._guide.parameters,
                              branch._guide.clock[2]-branch._guide.clock[1], branch._guide.stimulus[:, first])
    engine.advance_continuation(branch, query_count=3)
    assert torch.equal(branch.auxiliary_state, expected)


def test_future_changed_event_prefix_rejects_without_planning_activation(cases):
    engine, state, inputs, norm = _future_setup(cases, FAMILIES[3])
    engine.advance_continuation(state, query_count=2)
    state.event_doses[0, 0] *= 2
    before = _future_snapshot(state)
    with pytest.raises(ValueError, match="static controls changed"):
        engine.branch_future_controls(state, inputs, norm, executed_regimen_uid=inputs.regimen_uids[0])
    assert not state._work_ledger.planning_active
    _assert_future_snapshot(state, before)



def test_future_semantic_signature_distinguishes_drug_context_from_typed_waveforms(cases):
    import copy
    from Libs.utils.model.native_future_controls import _semantic_signature
    for family in FAMILIES[:2]:
        record = cases["validation", family][1].programs["records"][0][0][0]
        before = copy.deepcopy(record)
        signature = _semantic_signature(record)
        wave = signature["stimulation_or_environment"]["magnitude_or_program"]
        assert "time_values" not in wave and "values" not in wave
        assert wave["interpolation"] == "left_hold"
        assert record == before
        for malformed in (None, {"modality": "stimulation"}, {"magnitude_or_program": {"values": [1.]}}):
            broken = copy.deepcopy(record)
            broken["action"]["stimulation_or_environment"] = malformed
            with pytest.raises(ValueError, match="Typed stimulation"):
                _semantic_signature(broken)
    record = cases["validation", FAMILIES[2]][1].programs["records"][0][0][0]
    before = copy.deepcopy(record)
    assert record["action"]["intervention_type"] == "pharmacological"
    assert "magnitude_or_program" not in record["action"]["stimulation_or_environment"]
    assert _semantic_signature(record)["stimulation_or_environment"] == record["action"]["stimulation_or_environment"]
    assert record == before
