"""Native arrived measurements on retained public candidates; no NN/source execution."""
from dataclasses import fields, is_dataclass, replace
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
from Libs.models.modules.world_state import (
    JointWorldStatePosterior, _normalized_log_weights, _posterior_uid,
)
from Libs.scripts.builders.build_sample_observation_registry import produce_native_arrived_observation
from Libs.utils.data.joint_observation import retained_log_likelihood
from Libs.utils.model.native_arrived_observation import (
    NativeArrivedObservation, admitted_window_binding, arrived_payload_fingerprint,
    filter_native_observation, native_acquisition_mask, native_cns_readout,
    predict_native_readout, project_cns_readout,
)

FAMILIES = ("electrophysiology_conductance_world", "calcium_synaptic_world",
            "molecular_pharmacological_causal_world", "cns_long_horizon_world")


@pytest.fixture(scope="module", autouse=True)
def prohibit_nn_source_and_targets():
    from Libs.utils.data import native_diffusion_data as native
    from Libs.utils.model import dynamics_store, native_scientific_source as source
    def forbidden(*args, **kwargs):
        raise AssertionError("Native filtering fixture cannot execute NN/source/latent selection or target loaders")
    with patch.object(torch.nn.Module, "__call__", forbidden), patch.object(torch, "multinomial", forbidden), \
            patch.object(source, "draw_group", forbidden), patch.object(native, "load_records", forbidden), \
            patch.object(dynamics_store, "load_zarr_rows", forbidden), patch.object(dynamics_store, "load_chunked_rows", forbidden):
        yield


@pytest.fixture(scope="module")
def retained_filter_cases(prohibit_nn_source_and_targets):
    from Libs.utils.data.native_diffusion_data import load_family_registries, load_joint_sources, parse_joint_source
    from Libs.utils.data.native_field_normalization import NativeFieldNormalization, normalization_from_config
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.model.native_comparator_inputs import build_native_comparator_inputs
    from Libs.utils.model.native_scientific_source import selected_group_definitions
    location = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:
        pytest.skip("Requires retained artificial public-support fixture")
    config = load_config(str(Path(location)/"config.yaml"))
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_arrived_observation_filter"] = copy.deepcopy(current["native_arrived_observation_filter"])
    normalization = NativeFieldNormalization(normalization_from_config(config), 26)
    registries = load_family_registries(config)
    cases = {}
    for group in selected_group_definitions(config, "validation"):
        raw = load_joint_sources(registries[group["family_uid"]], [group["posterior_uid"]])[group["posterior_uid"]]
        support, observations = parse_joint_source(raw)
        # Test-only prior mass and unused zero neural context; no learned posterior is claimed.
        logs = _normalized_log_weights(support.prior_weights.double().log())
        context = torch.zeros(*support.initial_field.shape[:-1], 1)
        uids = tuple(row.observation_uid for row in observations)
        posterior = JointWorldStatePosterior(support, logs, context, torch.zeros(len(support.sample_uids), 6),
            uids, len(uids)-1, _posterior_uid(support, uids, None, logs, context), None,
            torch.zeros_like(logs), "test_only_retained_prior_no_neural_execution")
        inputs = build_native_comparator_inputs(raw, posterior, normalization, config)
        assert torch.equal(inputs.posterior.support.initial_field, support.initial_field)
        cases[group["family_uid"]] = (config, group, inputs, normalization)
    assert set(cases) == set(FAMILIES)
    return cases


def prepared(cases, family, count=2):
    config, group, inputs, normalization = cases[family]
    solver = NativeFrozenRHSSolver(config, {"equation_authority": "frozen_primary_training_law",
        "maximum_step_hours": .25, "maximum_rhs_particle_action_evaluations": 2_000_000})
    state = solver.initialize_continuation(inputs, normalization)
    solver.advance_continuation(state, query_count=count)
    readout = native_cns_readout(config, group)
    return solver, state, readout, inputs


def physical_snapshot(state):
    result = {name: getattr(state, name).clone() if torch.is_tensor(getattr(state, name)) else getattr(state, name)
              for name in ("cns_state", "auxiliary_state", "event_times", "event_doses", "applied_events",
                           "query_index", "interval_index", "rhs_python_calls", "failed")}
    result["parameters"] = {key: value.clone() for key, value in state._guide.parameters.items()}
    result["factors"] = {key: value.clone() for key, value in state._guide.factors.items()}
    result["pk_parameters"] = {key: value.clone() for key, value in state._pk_parameters.items()}
    return result


def assert_snapshot(state, original):
    current = physical_snapshot(state)
    for key, value in original.items():
        if isinstance(value, dict):
            assert all(torch.equal(value[name], current[key][name]) for name in value)
        elif torch.is_tensor(value):
            assert torch.equal(value, current[key])
        else:
            assert value == current[key]


def private_episode_field(inputs, readout, multiplier=1.15):
    # An explicit synthetic current-field fixture, kept only on the evaluator side.
    # Reuses an existing public tuple's initial field; no latent/source world is drawn.
    value = inputs.posterior.support.initial_field[:, 0].clone().double()
    for r, c, q in readout.selector:
        value[:, r, c, q] *= multiplier
    return value


def arrive(prediction, private, generator, uid):
    a = prediction.acquisition
    return produce_native_arrived_observation(private, a,
        episode_source_uid=a.readout.source_uid, episode_source_snapshot_ref=a.readout.source_snapshot_ref,
        episode_sample_uids=a.sample_uids, episode_query_index=a.query_index,
        episode_regimen_uid=a.executed_regimen_uid, measurement_generator=generator, observation_uid=uid)


def likelihood(prediction, observation):
    return retained_log_likelihood(observation,
        {"law": "gaussian", "units": prediction.acquisition.readout.units,
         "mean": prediction.mean, "variance": prediction.variance}, prediction.nuisance, {})


@pytest.mark.parametrize("family", FAMILIES)
def test_actual_continuation_arrival_fresh_filter_preserves_all_physics(retained_filter_cases, family):
    solver, state, readout, inputs = prepared(retained_filter_cases, family)
    before = physical_snapshot(state)
    initial_before = inputs.posterior.support.initial_field.clone()
    prediction = predict_native_readout(solver, state, readout, executed_regimen_uid=inputs.regimen_uids[0])
    assert prediction.mean.shape == (1, 32, 6)
    private = private_episode_field(inputs, readout)
    private_before = private.clone()
    generator = torch.Generator().manual_seed(451)
    arrived = arrive(prediction, private, generator, family+":arrived:1")
    old_logs = solver.current_log_weights(state)
    expected = _normalized_log_weights(old_logs+likelihood(prediction, arrived.observation).double())
    result = filter_native_observation(solver, state, prediction, arrived)
    assert torch.equal(result["log_weights"], expected)
    assert not torch.equal(result["log_weights"], old_logs)
    assert state.filter_revision == 1
    assert_snapshot(state, before)
    assert torch.equal(private, private_before)
    assert torch.equal(inputs.posterior.support.initial_field, initial_before)
    assert torch.equal(inputs.posterior.log_weights, old_logs)
    assert torch.equal(state._weights, expected.exp().to(state._weights.dtype))
    evidence = {"family_uid": family, "query_index": state.query_index,
        "physical_time_hours": state.current_time_hours, "particle_count": 32,
        "readout_features": 6, "filtering_revision": 1, "weights_changed": True,
        "current_physics_and_events_unchanged": True,
        "retained_prior_fixture_no_learned_context": True, "source_calls": 0, "nn_calls": 0}
    destination = os.environ.get("NRNSIFT_NATIVE_FILTER_EVIDENCE")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else {
            "scope": "native_arrived_observation_software", "groups": [],
            "source_calls": 0, "nn_calls": 0, "optimizer_updates": 0, "formal_qualification": False}
        report["groups"].append(evidence)
        path.write_text(json.dumps(report, indent=2))
    print(json.dumps(evidence), flush=True)


def test_sequential_filter_uses_current_means_and_current_log_mass(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[-1])
    initial_logs = solver.current_log_weights(state)
    generator = torch.Generator().manual_seed(452)
    first = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    packet1 = arrive(first, private_episode_field(inputs, spec, 1.3), generator, "sequential:1")
    filter_native_observation(solver, state, first, packet1)
    after_first = solver.current_log_weights(state)
    solver.advance_continuation(state, query_count=3)
    second = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    packet2 = arrive(second, private_episode_field(inputs, spec, 1.6), generator, "sequential:2")
    ll = likelihood(second, packet2.observation)
    expected = _normalized_log_weights(after_first+ll.double())
    restarted = _normalized_log_weights(initial_logs+ll.double())
    old_mean = project_cns_readout(inputs.posterior.support.initial_field, spec)
    stale = replace(second, mean=old_mean)
    stale_posterior = _normalized_log_weights(after_first+likelihood(stale, packet2.observation).double())
    assert not torch.allclose(expected, restarted, rtol=0, atol=1e-10)
    assert not torch.allclose(expected, stale_posterior, rtol=0, atol=1e-10)
    before = physical_snapshot(state)
    with pytest.raises(ValueError, match="Stale predicted readout"):
        filter_native_observation(solver, state, stale, packet2)
    assert torch.equal(solver.current_log_weights(state), after_first)
    filtered = filter_native_observation(solver, state, second, packet2)
    assert torch.equal(filtered["log_weights"], expected)
    assert state.filter_revision == 2
    assert_snapshot(state, before)


def test_masks_mixing_scales_and_group_scoped_noise(retained_filter_cases):
    config, group, inputs, _ = retained_filter_cases[FAMILIES[0]]
    held = dict(group, source_variant="held", observation_noise_multiplier=2.,
                observation_mask=[True, False, True, False, False, True])
    spec = native_cns_readout(config, held)
    assert spec.standard_deviation == .12
    assert spec.observed == (True, False, True, False, False, True)
    assert spec.matrix[0][:2] == (.9, .1)
    value = torch.zeros(1, 6, 6, 26, dtype=torch.float64)
    for index, ((r,c,q), scale) in enumerate(zip(spec.selector, spec.scales)):
        value[0,r,c,q] = (index+1)*scale
    actual = project_cns_readout(value, spec)
    expected = torch.arange(1,7,dtype=torch.float64)[None] @ torch.tensor(spec.matrix,dtype=torch.float64).T
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    with pytest.raises(ValueError, match="six finite"):
        project_cns_readout(value, replace(spec, standard_deviation=0.))


def test_producer_keeps_episode_rng_and_private_fields_out_of_packet(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[0])
    prediction = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    private = private_episode_field(inputs, spec)
    generator = torch.Generator().manual_seed(453)
    witness = torch.Generator(); witness.set_state(generator.get_state())
    global_before = torch.random.get_rng_state().clone()
    mean = project_cns_readout(private, spec)
    expected1 = mean+spec.standard_deviation*torch.randn(mean.shape,dtype=mean.dtype,generator=witness)
    expected2 = mean+spec.standard_deviation*torch.randn(mean.shape,dtype=mean.dtype,generator=witness)
    first = arrive(prediction, private, generator, "noise:1")
    second = arrive(prediction, private, generator, "noise:2")
    mask = first.observation.observed
    assert torch.equal(first.observation.values[mask], expected1[mask])
    assert torch.equal(second.observation.values[mask], expected2[mask])
    assert not torch.equal(first.observation.values[mask], second.observation.values[mask])
    assert torch.equal(generator.get_state(), witness.get_state())
    assert torch.equal(global_before, torch.random.get_rng_state())
    forbidden = {"generating_fields","generating_indices","latent_seed","observation_seed","measurement_generator","private_field"}
    def inspect(value):
        if is_dataclass(value):
            assert not forbidden.intersection(field.name for field in fields(value))
            for field in fields(value): inspect(getattr(value, field.name))
    inspect(first)
    assert set(vars(first)) == {"acquisition", "observation"}
    assert first.observation.physical_acquisition is None
    before = generator.get_state().clone()
    a = prediction.acquisition
    with pytest.raises(ValueError, match="Evaluator episode"):
        produce_native_arrived_observation(private, a, episode_source_uid=spec.source_uid,
            episode_source_snapshot_ref=spec.source_snapshot_ref, episode_sample_uids=("wrong",),
            episode_query_index=a.query_index, episode_regimen_uid=a.executed_regimen_uid,
            measurement_generator=generator, observation_uid="wrong_episode")
    assert torch.equal(generator.get_state(), before)


def test_stale_clock_program_readout_lineage_and_duplicate_reject(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[0])
    prediction = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    packet = arrive(prediction, private_episode_field(inputs,spec), torch.Generator().manual_seed(454), "reject:1")
    a = prediction.acquisition
    reversed_ids = (tuple(reversed(a.particle_uids[0])),)
    variants = [
        replace(a,query_index=a.query_index+1), replace(a,time_hours=a.time_hours*2),
        replace(a,source_time=a.source_time+1), replace(a,source_time_unit="hour"),
        replace(a,executed_regimen_uid=inputs.regimen_uids[1]),
        replace(a,particle_uids=reversed_ids), replace(a,support_uid="wrong"),
        replace(a,origin_posterior_uid="wrong"), replace(a,filtering_revision=1),
        replace(a,readout=replace(spec,standard_deviation=spec.standard_deviation*2)),
    ]
    before = physical_snapshot(state)
    logs = solver.current_log_weights(state)
    for changed in variants:
        with pytest.raises(ValueError, match="Stale or changed"):
            filter_native_observation(solver,state,prediction,replace(packet,acquisition=changed))
        assert torch.equal(solver.current_log_weights(state), logs)
    with pytest.raises(ValueError, match="readout changed"):
        predict_native_readout(solver,state,replace(spec,standard_deviation=spec.standard_deviation*2),
            executed_regimen_uid=inputs.regimen_uids[0])
    with pytest.raises(ValueError, match="fixed executed program"):
        predict_native_readout(solver,state,spec,executed_regimen_uid=inputs.regimen_uids[1])
    assert_snapshot(state,before)
    filter_native_observation(solver,state,prediction,packet)
    with pytest.raises(ValueError,match="Duplicate"):
        filter_native_observation(solver,state,prediction,packet)
    stale = predict_native_readout(solver,state,spec,executed_regimen_uid=inputs.regimen_uids[0])
    pending = arrive(stale,private_episode_field(inputs,spec),torch.Generator().manual_seed(455),"reject:2")
    solver.advance_continuation(state,query_count=3)
    with pytest.raises(ValueError,match="Stale or changed"):
        filter_native_observation(solver,state,stale,pending)


def test_filter_preserves_extreme_finite_logs_and_ignores_masked_values(retained_filter_cases):
    solver,state,spec,inputs=prepared(retained_filter_cases,FAMILIES[0])
    logs=torch.full_like(solver.current_log_weights(state),-10000.)
    logs[:,0]=0.
    logs=_normalized_log_weights(logs)
    solver.update_filter_weights(state,logs,query_index=state.query_index,parent_revision=0,observation_uid="extreme_setup")
    assert state._weights[0,1]==0 and torch.isfinite(solver.current_log_weights(state)[0,1])
    prediction=predict_native_readout(solver,state,spec,executed_regimen_uid=inputs.regimen_uids[0])
    packet=arrive(prediction,private_episode_field(inputs,spec),torch.Generator().manual_seed(456),"extreme:1")
    changed=packet.observation.values.clone()
    changed[~packet.observation.observed]=float("inf")
    altered=replace(packet,observation=replace(packet.observation,values=changed))
    assert torch.equal(likelihood(prediction,packet.observation),likelihood(prediction,altered.observation))
    expected=_normalized_log_weights(logs+likelihood(prediction,altered.observation).double())
    result=filter_native_observation(solver,state,prediction,altered)
    assert torch.equal(result["log_weights"],expected) and torch.isfinite(result["log_weights"]).all()


def test_predose_and_malformed_measurements_reject_without_physical_work(retained_filter_cases):
    config, group, inputs, normalization = retained_filter_cases[FAMILIES[0]]
    solver = NativeFrozenRHSSolver(config, {"equation_authority": "frozen_primary_training_law",
        "maximum_step_hours": .25, "maximum_rhs_particle_action_evaluations": 2_000_000})
    state = solver.initialize_continuation(inputs, normalization)
    spec = native_cns_readout(config, group)
    with pytest.raises(ValueError, match="postzero original query"):
        predict_native_readout(solver,state,spec,executed_regimen_uid=inputs.regimen_uids[0])
    assert state.rhs_python_calls == 0 and state.filter_revision == 0
    solver.advance_continuation(state,query_count=2)
    prediction=predict_native_readout(solver,state,spec,executed_regimen_uid=inputs.regimen_uids[0])
    packet=arrive(prediction,private_episode_field(inputs,spec),torch.Generator().manual_seed(457),"malformed")
    before=physical_snapshot(state)
    logs=solver.current_log_weights(state)
    for observation in [
        replace(packet.observation,source_uid="wrong"),
        replace(packet.observation,normalization_ref="wrong"),
        replace(packet.observation,sequence_number=20),
        replace(packet.observation,observed=~packet.observation.observed),
    ]:
        with pytest.raises(ValueError):
            filter_native_observation(solver,state,prediction,replace(packet,observation=observation))
    values=packet.observation.values.clone();values[packet.observation.observed]=float("nan")
    with pytest.raises(ValueError,match="Retained values"):
        filter_native_observation(solver,state,prediction,replace(packet,observation=replace(packet.observation,values=values)))
    assert state.filter_revision==0 and torch.equal(solver.current_log_weights(state),logs)
    assert_snapshot(state,before)



def test_successful_admission_binds_payload_and_failure_preserves_history(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[0])
    prediction = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    arrived = arrive(prediction, private_episode_field(inputs, spec), torch.Generator().manual_seed(451), "bound_readout")
    filter_native_observation(solver, state, prediction, arrived)
    assert state._filter_binding["native_admitted_readout_payload_sha256"]["bound_readout"] == arrived_payload_fingerprint(arrived)
    history, logs, revision, uids = copy.deepcopy(state._filter_binding), solver.current_log_weights(state), state.filter_revision, state.filter_observation_uids
    prediction = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    bad = arrive(prediction, private_episode_field(inputs, spec), torch.Generator().manual_seed(452), "failed_readout")
    values = bad.observation.values.clone()
    values[0, bad.observation.observed[0].nonzero()[0, 0]] = float("nan")
    bad = replace(bad, observation=replace(bad.observation, values=values))
    with pytest.raises(ValueError):
        filter_native_observation(solver, state, prediction, bad)
    assert state._filter_binding == history and torch.equal(solver.current_log_weights(state), logs)
    assert state.filter_revision == revision and state.filter_observation_uids == uids


def test_selected_mixed_coordinate_preserves_full_draw_and_all_tuple_likelihood(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[0])
    config, group, _, _ = retained_filter_cases[FAMILIES[0]]
    # Author the test channel before binding. The default fixture is identity;
    # reuse the existing held-scope mixing while retaining this source identity.
    mixed = native_cns_readout(config, dict(group, source_variant="held", observation_noise_multiplier=2.,
                                           observation_mask=[True, False, True, False, False, True]))
    spec = replace(spec, matrix=mixed.matrix)
    available = [index for index, retained in enumerate(spec.observed) if retained]
    coordinate = next(index for index in available
                      if any(value != 0 for other, value in enumerate(spec.matrix[index]) if other != index))
    full = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0])
    selected = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0],
                                     selected_coordinate=coordinate)
    assert selected.acquisition.readout == spec and selected.mean.shape == (1, 32, 6)
    assert torch.equal(selected.mean, full.mean) and torch.equal(selected.variance, full.variance)
    private = private_episode_field(inputs, spec)
    full_rng = torch.Generator().manual_seed(921)
    selected_rng = torch.Generator().manual_seed(921)
    full_packet = arrive(full, private, full_rng, "full_mixed_readout")
    packet = arrive(selected, private, selected_rng, "selected_mixed_readout")
    assert torch.equal(full_rng.get_state(), selected_rng.get_state())
    assert int(packet.observation.observed.sum()) == 1
    assert packet.observation.values[0, coordinate] == full_packet.observation.values[0, coordinate]
    assert torch.count_nonzero(packet.observation.values[~packet.observation.observed]) == 0
    variance = full.variance[..., coordinate]
    difference = packet.observation.values[:, coordinate, None]-full.mean[..., coordinate]
    expected_likelihood = -.5*(difference.square()/variance+torch.log(2*torch.pi*variance))
    torch.testing.assert_close(likelihood(selected, packet.observation), expected_likelihood, rtol=1e-12, atol=1e-12)
    expected = _normalized_log_weights(solver.current_log_weights(state)+expected_likelihood.double())
    before = physical_snapshot(state)
    other = next(index for index in available if index != coordinate)
    with pytest.raises(ValueError, match="Stale or changed"):
        filter_native_observation(solver, state, selected,
            replace(packet, acquisition=replace(packet.acquisition, selected_coordinate=other)))
    leaked = packet.observation.values.clone()
    leaked[0, other] = 1.
    with pytest.raises(ValueError, match="must be zero"):
        filter_native_observation(solver, state, selected,
            replace(packet, observation=replace(packet.observation, values=leaked)))
    result = filter_native_observation(solver, state, selected, packet)
    torch.testing.assert_close(result["log_weights"], expected, rtol=1e-12, atol=1e-12)
    assert result["log_weights"].shape == (1, 32)
    assert_snapshot(state, before)


def test_coordinate_availability_and_type_reject_before_effect(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[0])
    before = physical_snapshot(state)
    for choice in (-1, 6, True, 0., "0"):
        with pytest.raises(ValueError, match="available mixed readout"):
            predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0],
                                   selected_coordinate=choice)
    unavailable = replace(spec, observed=(False, *spec.observed[1:]))
    with pytest.raises(ValueError, match="available mixed readout"):
        native_acquisition_mask(unavailable, 0)
    assert native_acquisition_mask(unavailable) == unavailable.observed
    assert_snapshot(state, before)
    assert state.filter_revision == 0


def test_changing_coordinate_keeps_consecutive_admitted_window(retained_filter_cases):
    solver, state, spec, inputs = prepared(retained_filter_cases, FAMILIES[0])
    solver.bind_filter_readout(state, spec, inputs.regimen_uids[0])
    start = solver.fork_continuation(state)
    available = [index for index, retained in enumerate(spec.observed) if retained]
    arrivals = []
    generator = torch.Generator().manual_seed(922)
    # Start at the already advanced query1 and admit the next three clocks.
    for query in range(2, 5):
        solver.advance_continuation(state, query_count=query+1)
        prediction = predict_native_readout(solver, state, spec, executed_regimen_uid=inputs.regimen_uids[0],
                                           selected_coordinate=available[query % len(available)])
        packet = arrive(prediction, private_episode_field(inputs, spec), generator, f"coordinate_window_{query}")
        filter_native_observation(solver, state, prediction, packet)
        arrivals.append(packet)
    binding = admitted_window_binding(solver, start, state, tuple(arrivals))
    assert binding["status"] == "ready" and binding["query_indices"] == [2, 3, 4]
    assert state._filter_readout == spec and state.filter_revision == 3
    last = arrivals[-1]
    choice = next(index for index in available if index != last.acquisition.selected_coordinate)
    mask = torch.tensor(native_acquisition_mask(spec, choice))[None]
    changed = replace(last, acquisition=replace(last.acquisition, selected_coordinate=choice),
                      observation=replace(last.observation, observed=mask, values=torch.zeros_like(last.observation.values)))
    with pytest.raises(ValueError, match="payload or acquisition changed"):
        admitted_window_binding(solver, start, state, (*arrivals[:-1], changed))
