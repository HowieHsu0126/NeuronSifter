"""Independent-state hybrid transitions on retained artificial public supports."""
import copy
from dataclasses import asdict, replace
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
from Libs.models.modules.native_mechanistic_residual import (
    NativeMechanisticResidual, NativeResidualOnlineEpisode, native_residual_online_policy,
)
from Libs.models.modules.world_state import JointWorldStatePosterior, _normalized_log_weights, _posterior_uid
from Libs.utils.data.native_diffusion_data import load_family_registries, load_joint_sources, parse_joint_source
from Libs.utils.data.native_field_normalization import NativeFieldNormalization, normalization_from_config
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.native_input_staging import input_locations_from_config
from Libs.utils.model.native_comparator_inputs import build_native_comparator_inputs
from Libs.utils.model import native_scientific_source as source
from Libs.utils.evaluation.native_predose_evaluation import evaluate_native_predose_queries
from Libs.models.modules import native_mechanistic_residual as residual_module
from Libs.scripts.builders.build_sample_observation_registry import produce_native_arrived_observation
from Libs.utils.model.native_conditional_readout import AdaptiveReadoutQuery
from Libs.utils.model.native_arrived_observation import (
    admitted_window_binding, filter_native_observation, native_acquisition_mask, native_cns_readout, predict_native_readout,
)

FAMILIES = ("electrophysiology_conductance_world", "calcium_synaptic_world",
            "molecular_pharmacological_causal_world", "cns_long_horizon_world")


@pytest.fixture(scope="module")
def cases():
    location = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:
        pytest.skip("Requires retained artificial four-domain public-support fixture")
    config = load_config(str(Path(location)/"config.yaml"))
    manifest = os.environ.get("NRNSIFT_RHS_INPUT_MANIFEST")
    if manifest:
        config["native_input_staging"] = {"manifest_path": manifest}
    locations = input_locations_from_config(config)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_controlled_comparators"]["references"]["mechanistic_residual"] = copy.deepcopy(
        current["native_controlled_comparators"]["references"]["mechanistic_residual"])
    config["native_controlled_comparators"]["references"]["same_compiler_solver"] = copy.deepcopy(
        current["native_controlled_comparators"]["references"]["same_compiler_solver"])
    normalizer = NativeFieldNormalization(normalization_from_config(config), 26, input_locations=locations)
    registries = load_family_registries(config)
    result = {}
    for role in ("train", "validation"):
        for group in source.selected_group_definitions(config, role):
            raw = load_joint_sources(registries[group["family_uid"]], [group["posterior_uid"]],
                input_locations=locations)[group["posterior_uid"]]
            support, observations = parse_joint_source(raw)
            logs = _normalized_log_weights(support.prior_weights.double().log())
            context = torch.zeros(*support.initial_field.shape[:-1], 1)
            ids = tuple(item.observation_uid for item in observations)
            posterior = JointWorldStatePosterior(support, logs, context, torch.zeros(1, 6), ids, 0,
                _posterior_uid(support, ids, None, logs, context), None, torch.zeros_like(logs), "retained_prior_software_fixture")
            result[role, group["family_uid"]] = (config, build_native_comparator_inputs(raw, posterior, normalizer, config), normalizer)
    return result


def solver(config):
    return NativeFrozenRHSSolver(config, config["native_controlled_comparators"]["references"]["same_compiler_solver"]["hyperparameters"])


def fixed_history(state):
    names = ("event_times", "event_doses", "applied_events", "query_index", "interval_index", "rhs_python_calls")
    result = {name: getattr(state, name).clone() if torch.is_tensor(getattr(state, name)) else getattr(state, name) for name in names}
    result["free_carry"] = state.cns_state[..., -1].clone()
    result["pk"] = {key: value.clone() for key, value in state._pk_parameters.items()}
    return result


def equal_history(state, previous):
    actual = fixed_history(state)
    for key, value in previous.items():
        if isinstance(value, dict): assert all(torch.equal(value[name], actual[key][name]) for name in value)
        elif torch.is_tensor(value): assert torch.equal(value, actual[key])
        else: assert value == actual[key]


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_hybrid_is_exact_solver_on_original_prefix_without_target_access(cases, family):
    config, inputs, norm = cases["validation", family]
    model = NativeMechanisticResidual(config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"], inputs, config).eval()
    expected = solver(config).forecast_native(inputs, norm, query_count=3)["particle_fields"].reshape(-1, 3, *model.field_shape)
    def forbidden(*args, **kwargs): raise AssertionError("Hybrid inference must not draw sources or load teacher arrays")
    with patch.object(source, "draw_group", forbidden), patch("Libs.utils.data.native_diffusion_data.load_records", forbidden):
        actual = model.predict_native(inputs, norm, query_count=3)
    assert torch.equal(actual, expected)
    assert actual.shape == (96, 3, 6, 6, 26)
    assert torch.equal(actual[:, 0], inputs.posterior.support.initial_field[:, :, None].expand(1, 32, 3, 6, 6, 26).reshape(96, 6, 6, 26))
    assert model.output.weight.count_nonzero() == model.output.bias.count_nonzero() == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_nonzero_correction_changes_next_mechanical_transition_not_PK_history(cases, family):
    config, inputs, norm = cases["validation", family]
    engine = solver(config)
    baseline, altered = [engine.initialize_continuation(inputs, norm) for _ in range(2)]
    engine.advance_continuation(baseline, query_count=2)
    engine.advance_continuation(altered, query_count=2)
    previous = fixed_history(altered)
    original_aux = None if altered.auxiliary_state is None else altered.auxiliary_state.clone()
    before = engine.current_particle_fields(altered)
    delta = torch.zeros_like(before)
    coordinate = inputs.grid["species_id"].index("calcium_concentration")
    delta[..., coordinate] = norm.scale[coordinate]*.001
    after = engine.apply_state_residual(altered, delta)
    assert not torch.equal(before, after)
    equal_history(altered, previous)
    if original_aux is not None: assert torch.equal(altered.auxiliary_state, original_aux)
    base_next = engine.advance_continuation(baseline, query_count=3)["particle_fields"]
    changed_next = engine.advance_continuation(altered, query_count=3)["particle_fields"]
    assert not torch.equal(base_next, changed_next)
    for name in ("free_drug", "target_bound_drug"):
        index = inputs.grid["species_id"].index(name)
        assert torch.equal(base_next[..., index], changed_next[..., index])
    equal_history(altered, fixed_history(baseline))


@pytest.mark.parametrize("family", FAMILIES)
def test_teacher_force_only_train_legal_fields_and_atomic_domain_rejection(cases, family):
    config, inputs, norm = cases["train", family]
    engine = solver(config)
    state = engine.initialize_continuation(inputs, norm)
    original = inputs.posterior.support.initial_field.clone()
    fields = engine.current_particle_fields(state)
    altered = fields.clone()
    index = inputs.grid["species_id"].index("amyloid_plaque")
    altered[..., index] *= 1.01
    altered[..., inputs.grid["species_id"].index("free_drug")] = 1000.
    prior = fixed_history(state)
    before_aux = None if state.auxiliary_state is None else state.auxiliary_state.clone()
    observed = torch.ones_like(fields, dtype=torch.bool)
    missing = observed.clone(); missing[..., index] = False
    with pytest.raises(ValueError, match="every independent current state"):
        engine.teacher_force_independent_state(state, altered, observed=missing)
    equal_history(state, prior)
    updated = engine.teacher_force_independent_state(state, altered, observed=observed)
    assert torch.equal(updated[..., index], altered[..., index])
    equal_history(state, prior)
    assert torch.equal(inputs.posterior.support.initial_field, original)
    if family == FAMILIES[2]: assert torch.equal(state.auxiliary_state, before_aux)
    with pytest.raises(ValueError, match="t0"):
        engine.apply_state_residual(state, torch.zeros_like(fields))
    _, validation, _ = cases["validation", family]
    with pytest.raises(ValueError, match="TRAIN"):
        engine.teacher_force_independent_state(engine.initialize_continuation(validation, norm), fields, observed=observed)
    engine.advance_continuation(state, query_count=2)
    current = engine.current_particle_fields(state)
    forbidden = torch.zeros_like(current)
    forbidden[..., inputs.grid["species_id"].index("free_drug")] = .1
    with pytest.raises(ValueError, match="outside permitted"):
        engine.apply_state_residual(state, forbidden)
    invalid = torch.zeros_like(current); invalid[..., index] = -current[..., index]-1
    with pytest.raises(FloatingPointError):
        engine.apply_state_residual(state, invalid)
    assert torch.equal(engine.current_particle_fields(state), current)


def test_neural_auxiliary_derivation_and_molecular_exclusion(cases):
    for family in FAMILIES[:3]:
        config, inputs, norm = cases["validation", family]
        engine = solver(config); state = engine.initialize_continuation(inputs, norm)
        engine.advance_continuation(state, query_count=2)
        delta = torch.zeros_like(engine.current_particle_fields(state))
        mask = engine.independent_state_mask(state).reshape_as(state._initial)
        guide = state._guide
        if family == FAMILIES[2]:
            assert all(not mask[:, node, index].any() for node, index in zip(guide.auxiliary_nodes, guide.auxiliary_indices))
            continue
        flat = delta.reshape_as(state._initial)
        flat[:, guide.auxiliary_nodes[0], guide.auxiliary_indices[0]] = .00001
        engine.apply_state_residual(state, delta)
        if family == FAMILIES[0]:
            expected = ((state.auxiliary_state[:, 0]-guide.parameters["spike_threshold"])*guide.parameters["spike_rate_gain"]).clamp_min(0)
            assert torch.equal(state.auxiliary_state[:, 2], expected)
        else:
            expected = (guide.parameters["initial_release_probability"]+guide.parameters["release_coupling"]
                *(state.auxiliary_state[:, 0]-guide.parameters["baseline_intracellular_calcium"]).clamp_min(0)).clamp(0, 1)
            assert torch.equal(state.auxiliary_state[:, 1], expected)


def test_uncalibrated_hybrid_cannot_open_scientific_query_evaluation(tmp_path):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    model = SimpleNamespace(prediction_kind="offline_one_step_mechanistic_residual_conditional_mean")
    with patch("Libs.utils.evaluation.native_predose_evaluation.predose_source_admission",
               return_value={"scope": "admitted_controlled_source"}):
        with pytest.raises(ValueError, match="independently admitted residual calibration"):
            evaluate_native_predose_queries(config, model, None, {}, None, {}, {}, tmp_path)


@pytest.mark.parametrize("family", FAMILIES[:2])
def test_masked_independent_carries_survive_other_supported_corrections(cases, family):
    config, inputs, norm = cases["validation", family]
    engine = solver(config)
    state = engine.initialize_continuation(inputs, norm)
    engine.advance_continuation(state, query_count=2)
    guide = state._guide
    hidden_cns = inputs.grid["species_id"].index("amyloid_plaque")
    cns_column = guide.dynamic_indices.index(hidden_cns)
    hidden_aux_node, hidden_aux_index = guide.auxiliary_nodes[0], guide.auxiliary_indices[0]
    # Disposable support fixture: the carried physical states remain present
    # even when their rendered target/readout coordinates are not supported.
    state._validity = state._validity.clone()
    flat_support = state._validity[:, :, :, state.query_index].reshape_as(state._initial)
    flat_support[:, :, hidden_cns] = False
    flat_support[:, hidden_aux_node, hidden_aux_index] = False
    original_cns = state.cns_state[..., cns_column].clone()
    original_aux = state.auxiliary_state.clone()
    before = engine.current_particle_fields(state)
    assert not before[..., hidden_cns].count_nonzero()
    delta = torch.zeros_like(before)
    altered = inputs.grid["species_id"].index("calcium_concentration")
    delta[..., altered] = norm.scale[altered]*.001
    engine.apply_state_residual(state, delta)
    assert torch.equal(state.cns_state[..., cns_column], original_cns)
    assert torch.equal(state.auxiliary_state, original_aux)
    assert torch.equal(engine.current_particle_fields(state)[..., altered], before[..., altered]+delta[..., altered])


@pytest.fixture(scope="module")
def prior_hybrid_class():
    """Exact prior two source blobs, staged explicitly; no historical data load."""
    location = os.environ.get("NRNSIFT_RHS_BASELINE_SOURCE")
    if not location:
        pytest.skip("Requires the exact pre-change two-file source snapshot")
    modules = []
    for suffix in ("native_frozen_rhs_solver", "native_mechanistic_residual"):
        name = "_nrnsift_prior_"+suffix
        spec = importlib.util.spec_from_file_location(name, Path(location)/(suffix+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    modules[1].NativeFrozenRHSSolver = modules[0].NativeFrozenRHSSolver
    yield modules[1].NativeMechanisticResidual
    for module in modules:
        sys.modules.pop(module.__name__, None)


def _retained_hybrid(cases, family):
    config, inputs, norm = cases["validation", family]
    model = NativeMechanisticResidual(config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"], inputs, config).eval()
    return model, config, inputs, norm


@pytest.mark.parametrize("family", FAMILIES)
def test_hybrid_continuation_matches_prior_source_and_segmented_inference(cases, prior_hybrid_class, family):
    model, config, inputs, norm = _retained_hybrid(cases, family)
    legacy = prior_hybrid_class(model.settings, inputs, config).eval()
    with torch.no_grad():
        model.output.bias[inputs.grid["species_id"].index("calcium_concentration")] = .001
    legacy.load_state_dict(model.state_dict(), strict=True)
    old_states = []
    old_apply = legacy.solver.apply_state_residual
    def capture_old_state(state, residual):
        value = old_apply(state, residual)
        old_states.append(state)
        return value
    legacy.solver.apply_state_residual = capture_old_state
    with torch.no_grad():
        expected = legacy.predict_native(inputs, norm, query_count=4)
        actual = model.predict_native(inputs, norm, query_count=4)
        initial = model.initialize_continuation(inputs, norm)
        partial, first = model.advance_continuation(initial, norm, query_count=2)
        completed, rest = model.advance_continuation(partial, norm, query_count=4)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cat((first, rest), 1), expected)
    assert actual.shape == (96, 4, 6, 6, 26) and not actual.requires_grad
    assert initial.rhs.query_index == 0 and partial.rhs.query_index == 1
    assert completed.rhs.query_index == 3
    assert completed.rhs._filter_binding["particle_uids"] == tuple(tuple(row) for row in inputs.particle_uids)
    assert completed.rhs._times == tuple(inputs.grid["time_hours"])
    assert torch.equal(completed.rhs.cns_state, old_states[-1].cns_state)
    if completed.rhs.auxiliary_state is not None:
        assert torch.equal(completed.rhs.auxiliary_state, old_states[-1].auxiliary_state)
    equal_history(completed.rhs, fixed_history(old_states[-1]))
    assert torch.equal(completed.rhs._log_weights, initial.rhs._log_weights)
    assert completed.rhs.program_context == initial.rhs.program_context


def test_zero_head_has_three_interval_recursive_residual_gradients_only(cases, monkeypatch):
    model, _, inputs, norm = _retained_hybrid(cases, FAMILIES[0])
    continuation = model.initialize_continuation(inputs, norm)
    before = fixed_history(continuation.rhs)
    deltas = []
    original = model.normalized_residual
    def observe_delta(*args, **kwargs):
        delta = original(*args, **kwargs)
        delta.retain_grad()
        deltas.append(delta)
        return delta
    monkeypatch.setattr(model, "normalized_residual", observe_delta)
    def forbidden(*args, **kwargs):
        raise AssertionError("Recursive gradient probe cannot access source or teacher arrays")
    with patch.object(source, "draw_group", forbidden), patch("Libs.utils.data.native_diffusion_data.load_records", forbidden):
        completed, fields = model.advance_continuation(continuation, norm, query_count=4)
    indices = [inputs.grid["species_id"].index(name) for name in
               ("calcium_concentration", "electrical_membrane_voltage")]
    loss = sum((fields[:, -1, ..., index]/norm.scale[index]).mean() for index in indices)
    loss.backward()
    assert len(deltas) == 3 and completed.rhs.query_index == 3
    assert not any(torch.count_nonzero(delta.detach()) for delta in deltas)
    for index in indices:
        assert torch.isfinite(deltas[0].grad[..., index]).all()
        assert torch.count_nonzero(deltas[0].grad[..., index]) > 0
    assert model.output.bias.grad is not None and torch.isfinite(model.output.bias.grad).all()
    assert torch.count_nonzero(model.output.bias.grad) > 0
    constants = [*completed.rhs._guide.parameters.values(), *completed.rhs._guide.factors.values(),
                 *completed.rhs._pk_parameters.values(), *norm.buffers(), continuation.program,
                 continuation.static, continuation.rhs._initial, continuation.rhs._weights]
    assert all(not value.requires_grad and value.grad is None for value in constants)
    assert sum(parameter.numel() for parameter in model.solver.parameters()) == 0
    assert not continuation.rhs.cns_state.requires_grad
    equal_history(continuation.rhs, before)
    assert fields.shape == (96, 4, 6, 6, 26)


def test_hybrid_functional_branches_keep_distinct_carry_and_fixed_history(cases):
    model, _, inputs, norm = _retained_hybrid(cases, FAMILIES[3])
    with torch.no_grad():
        initial = model.initialize_continuation(inputs, norm)
        prefix, _ = model.advance_continuation(initial, norm, query_count=2)
        before = fixed_history(prefix.rhs)
        left, left_fields = model.advance_continuation(prefix, norm, query_count=4)
        right, right_fields = model.advance_continuation(prefix, norm, query_count=4)
    assert torch.equal(left_fields, right_fields)
    equal_history(prefix.rhs, before)
    assert left.rhs.cns_state.data_ptr() != right.rhs.cns_state.data_ptr()
    assert left.rhs.applied_events.data_ptr() != right.rhs.applied_events.data_ptr()
    expected = right.rhs.cns_state.clone()
    left.rhs.cns_state.add_(.001)
    assert torch.equal(right.rhs.cns_state, expected)
    assert left.rhs.program_context == right.rhs.program_context == prefix.rhs.program_context
    assert left.rhs._filter_binding == right.rhs._filter_binding == prefix.rhs._filter_binding
    assert left.rhs._times == right.rhs._times == prefix.rhs._times
    assert left_fields.shape[:2] == (96, 2)


def test_hybrid_functional_continuation_rejects_invalid_or_foreign_state(cases):
    model, _, inputs, norm = _retained_hybrid(cases, FAMILIES[0])
    initial = model.initialize_continuation(inputs, norm)
    with pytest.raises(ValueError, match="later original query"):
        model.advance_continuation(initial, norm, query_count=1)
    with pytest.raises(ValueError, match="owner or fixed normalization"):
        model.advance_continuation(replace(initial, owner=object()), norm, query_count=2)
    broken = model.solver.fork_continuation(initial.rhs)
    broken.cns_state[..., 0] = -1
    with pytest.raises(FloatingPointError, match="finite and nonnegative"):
        model.advance_continuation(replace(initial, rhs=broken), norm, query_count=2)
    assert initial.rhs.query_index == 0 and not initial.rhs.failed
    assert (initial.rhs.cns_state >= 0).all()


@pytest.fixture(scope="module")
def received_h3_window(cases):
    config, inputs, norm = cases["validation", FAMILIES[0]]
    config = copy.deepcopy(config)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_arrived_observation_filter"] = copy.deepcopy(current["native_arrived_observation_filter"])
    group = next(row for row in source.selected_group_definitions(config, "validation")
                 if row["family_uid"] == FAMILIES[0])
    readout = native_cns_readout(config, group)
    model = NativeMechanisticResidual(config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"], inputs, config).eval()
    start = model.initialize_continuation(inputs, norm)
    live, arrivals = start, []
    generator = torch.Generator().manual_seed(451)
    for index in range(1, 4):
        with torch.no_grad():
            live, _ = model.advance_continuation(live, norm, query_count=index+1)
        prediction = predict_native_readout(model.solver, live.rhs, readout,
                                            executed_regimen_uid=inputs.regimen_uids[0])
        # Existing provider receives an explicit private current-field software
        # fixture, not a new source trajectory or an exposed hidden tuple index.
        private = inputs.posterior.support.initial_field[:, 0].clone().double()
        for r, c, q in readout.selector:
            private[:, r, c, q] *= 1.15
        acquisition = prediction.acquisition
        arrived = produce_native_arrived_observation(private, acquisition,
            episode_source_uid=readout.source_uid, episode_source_snapshot_ref=readout.source_snapshot_ref,
            episode_sample_uids=acquisition.sample_uids, episode_query_index=index,
            episode_regimen_uid=acquisition.executed_regimen_uid,
            measurement_generator=generator, observation_uid=f"h3_window_observation_{index}")
        result = filter_native_observation(model.solver, live.rhs, prediction, arrived)
        assert result["status"] == "weight_filter_complete"
        arrivals.append(arrived)
    return model, inputs, norm, start, live, tuple(arrivals)


def test_actual_provider_filter_h3_joint_likelihood_and_phi_update(received_h3_window, monkeypatch, tmp_path):
    model, inputs, norm, start, live, arrivals = received_h3_window
    prior = start.rhs._log_weights.clone()
    live_logs = live.rhs._log_weights.clone()
    live_history = fixed_history(live.rhs)
    live_binding = copy.deepcopy(live.rhs._filter_binding)
    live_uids, live_revision = live.rhs.filter_observation_uids, live.rhs.filter_revision
    assert live_revision == start.rhs.filter_revision+3
    assert not torch.equal(prior, live_logs)
    assert not any(row.observation.observation_uid in start.rhs.filter_observation_uids for row in arrivals)
    captured = []
    original = residual_module.retained_log_likelihood
    def observe_likelihood(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value)
        return value
    monkeypatch.setattr(residual_module, "retained_log_likelihood", observe_likelihood)
    def forbidden(*args, **kwargs):
        raise AssertionError("H3 cannot generate sources, load future targets, or filter again")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # Explicit software-test step, not a protocol default.
    with patch.object(source, "draw_group", forbidden), patch("Libs.utils.data.native_diffusion_data.load_records", forbidden), \
            patch("Libs.utils.model.dynamics_store.load_zarr_rows", forbidden), \
            patch("Libs.utils.model.dynamics_store.load_chunked_rows", forbidden), \
            patch.object(model.solver, "update_filter_weights", forbidden):
        loss, diagnostic = model.partial_observation_window_loss(start, live, arrivals, norm)
        assert loss is not None and loss.requires_grad and len(captured) == 3
        expected = -torch.logsumexp(prior+torch.stack(captured).sum(0), dim=-1).sum()
        assert torch.equal(loss, expected)
        nll_before = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert model.output.bias.grad is not None and torch.isfinite(model.output.bias.grad).all()
        gradient_norm = float(model.output.bias.grad.norm())
        assert gradient_norm > 0
        optimizer.step()
        with torch.no_grad():
            after_loss, _ = model.partial_observation_window_loss(start, live, arrivals, norm)
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(before[name], parameter)]
    assert changed and all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert diagnostic["particle_count"] == 32 and diagnostic["observed_counts"] == [int(row.observation.observed.sum()) for row in arrivals]
    assert diagnostic["online_calibration_qualified"] is False and diagnostic["scope"] == "partial_observation_H3"
    assert torch.equal(start.rhs._log_weights, prior) and torch.equal(live.rhs._log_weights, live_logs)
    assert live.rhs.filter_revision == live_revision and live.rhs.filter_observation_uids == live_uids
    assert live.rhs._filter_binding == live_binding
    equal_history(live.rhs, live_history)
    constants = [*live.rhs._guide.parameters.values(), *live.rhs._guide.factors.values(),
                 *live.rhs._pk_parameters.values(), *norm.buffers(), start.rhs.cns_state,
                 start.program, start.static, start.rhs._log_weights]
    assert all(not value.requires_grad and value.grad is None for value in constants)
    assert sum(parameter.numel() for parameter in model.solver.parameters()) == 0
    evidence = dict(diagnostic, provider_calls=3, actual_filter_updates=3, optimizer_steps=1,
        test_only_learning_rate=1e-6, nll_before=nll_before, nll_after=float(after_loss),
        residual_head_gradient_norm=gradient_norm, changed_phi_parameters=changed,
        live_state_and_filter_unchanged=True, source_calls=0, future_target_loads=0,
        private_current_field_kind="explicit_software_fixture_not_source_trajectory")
    destination = Path(os.environ.get("NRNSIFT_NATIVE_H3_EVIDENCE", str(tmp_path/"h3_window.json")))
    destination.write_text(json.dumps(evidence, indent=2)+"\n")


@pytest.mark.parametrize("kind", ["unadmitted", "duplicate", "future", "nonconsecutive", "source", "support", "program", "value", "mask", "noise", "trajectory", "accepted_bit"])
def test_h3_rejects_unadmitted_changed_or_future_payload_before_model(received_h3_window, monkeypatch, kind):
    model, inputs, norm, start, live, original = received_h3_window
    arrivals = list(original)
    row = arrivals[1]
    if kind == "unadmitted":
        row = replace(row, observation=replace(row.observation, observation_uid="unadmitted_uid"))
    elif kind == "duplicate":
        row = arrivals[0]
    elif kind == "future":
        row = replace(row, acquisition=replace(row.acquisition, query_index=4))
    elif kind == "nonconsecutive":
        row = replace(row, acquisition=replace(row.acquisition, query_index=1))
    elif kind == "source":
        row = replace(row, observation=replace(row.observation, source_snapshot_ref="other_source"))
    elif kind == "support":
        row = replace(row, acquisition=replace(row.acquisition, support_uid="other_support"))
    elif kind == "program":
        row = replace(row, acquisition=replace(row.acquisition, executed_regimen_uid=inputs.regimen_uids[1]))
    elif kind == "value":
        values = row.observation.values.clone()
        index = row.observation.observed[0].nonzero()[0, 0]
        values[0, index] += 1
        row = replace(row, observation=replace(row.observation, values=values))
    elif kind == "mask":
        mask = row.observation.observed.clone()
        mask[0, 0] = ~mask[0, 0]
        row = replace(row, observation=replace(row.observation, observed=mask))
    elif kind == "noise":
        row = replace(row, acquisition=replace(row.acquisition, readout=replace(row.acquisition.readout,
            standard_deviation=2*row.acquisition.readout.standard_deviation)))
    elif kind == "trajectory":
        row = replace(row, observation=replace(row.observation, values=torch.empty(1, 4, 6, 6, 26)))
    else:
        row = {"accepted": True, "observation": row.observation}
    arrivals[1] = row
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid H3 packets must reject before model execution")
    monkeypatch.setattr(model, "advance_continuation", forbidden)
    with pytest.raises(ValueError):
        model.partial_observation_window_loss(start, live, tuple(arrivals), norm)


def test_h3_missing_window_or_legacy_binding_is_not_ready(received_h3_window, monkeypatch):
    model, _, norm, start, live, arrivals = received_h3_window
    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete H3 evidence must not execute the model")
    monkeypatch.setattr(model, "advance_continuation", forbidden)
    for count in range(3):
        loss, diagnostic = model.partial_observation_window_loss(start, live, arrivals[:count], norm)
        assert loss is None and diagnostic["status"] == "not_ready"
    legacy_rhs = model.solver.fork_continuation(live.rhs)
    legacy_rhs._filter_binding.pop("native_admitted_readout_payload_sha256")
    loss, diagnostic = model.partial_observation_window_loss(start, replace(live, rhs=legacy_rhs), arrivals, norm)
    assert loss is None and diagnostic["reason"] == "historical_admission_payload_binding_missing"
    assert len(diagnostic["unbound_observation_uids"]) == 3
    with pytest.raises(ValueError):
        model.partial_observation_window_loss(live, live, arrivals, norm)


@pytest.fixture(scope="module")
def online_warmup(cases):
    """Four actual public-fixture transitions; no source calls or optimizer steps."""
    config, inputs, norm = cases["validation", FAMILIES[0]]
    config = copy.deepcopy(config)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    for key in ("native_arrived_observation_filter", "native_mechanistic_residual_online"):
        config[key] = copy.deepcopy(current[key])
    config["schemas"]["native_mechanistic_residual_online"] = copy.deepcopy(
        current["schemas"]["native_mechanistic_residual_online"])
    settings = config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"]
    # Software capacity only; online policy, complete32 support and clocks stay exact.
    settings.update(encoder_widths=[4, 8, 12], bottleneck_width=16)
    with torch.random.fork_rng():
        torch.manual_seed(1703)
        original = NativeMechanisticResidual(settings, inputs, config).train()
    group = next(row for row in source.selected_group_definitions(config, "validation")
                 if row["family_uid"] == FAMILIES[0])
    readout = native_cns_readout(config, group)
    episode = NativeResidualOnlineEpisode(original, inputs, norm, readout,
                                          executed_regimen_uid=inputs.regimen_uids[0])
    private = inputs.posterior.support.initial_field[:, 0].clone().double()
    for r, c, q in readout.selector:
        private[:, r, c, q] *= 1.15
    generator = torch.Generator().manual_seed(451)
    def provider(acquisition):
        return produce_native_arrived_observation(private, acquisition,
            episode_source_uid=readout.source_uid, episode_source_snapshot_ref=readout.source_snapshot_ref,
            episode_sample_uids=acquisition.sample_uids, episode_query_index=acquisition.query_index,
            episode_regimen_uid=acquisition.executed_regimen_uid, measurement_generator=generator,
            observation_uid=f"online_observation_{acquisition.query_index}")
    def forbidden(*args, **kwargs):
        raise AssertionError("Online integration cannot draw a source or load future targets")
    with patch.object(source, "draw_group", forbidden), patch("Libs.utils.data.native_diffusion_data.load_records", forbidden), \
            patch("Libs.utils.model.dynamics_store.load_zarr_rows", forbidden), \
            patch("Libs.utils.model.dynamics_store.load_chunked_rows", forbidden):
        for index in range(1, 5):
            result = episode.step(provider)
            assert result["adaptation"]["status"] == "warmup"
            assert result["filtering"]["filtering_revision"] == index
            assert result["adaptation"]["optimizer_updates_completed"] == 0
    assert original.training and not episode.model.training
    assert episode.model is not original
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(original.parameters(), episode.model.parameters()))
    assert episode.optimizer_updates == 0 and episode.observations_seen == 4
    return original, episode, inputs, private, generator.get_state()


def _online_provider(episode, private, generator_state, requests, packets):
    generator = torch.Generator()
    generator.set_state(generator_state)
    def provide(acquisition):
        # This callable receives exactly one public request; private values and
        # measurement RNG exist only in the evaluator closure.
        assert type(acquisition) is residual_module.NativeAcquisition
        assert not {"private_field", "generator", "seed", "latent_index", "particle_fields"} & set(asdict(acquisition))
        assert acquisition.query_index == episode.continuation.rhs.query_index
        requests.append(acquisition)
        result = produce_native_arrived_observation(private, acquisition,
            episode_source_uid=acquisition.readout.source_uid,
            episode_source_snapshot_ref=acquisition.readout.source_snapshot_ref,
            episode_sample_uids=acquisition.sample_uids, episode_query_index=acquisition.query_index,
            episode_regimen_uid=acquisition.executed_regimen_uid, measurement_generator=generator,
            observation_uid=f"online_observation_{acquisition.query_index}")
        packets.append(result)
        return result
    return provide


def _online_live_snapshot(episode):
    state = episode.continuation.rhs
    return {"history": fixed_history(state), "cns": state.cns_state.clone(),
        "auxiliary": None if state.auxiliary_state is None else state.auxiliary_state.clone(),
        "weights": state._weights.clone(), "log_weights": state._log_weights.clone(),
        "binding": copy.deepcopy(state._filter_binding), "revision": state.filter_revision,
        "uids": state.filter_observation_uids}


def _assert_online_live_snapshot(episode, before):
    state = episode.continuation.rhs
    equal_history(state, before["history"])
    assert torch.equal(state.cns_state, before["cns"])
    if before["auxiliary"] is not None:
        assert torch.equal(state.auxiliary_state, before["auxiliary"])
    assert torch.equal(state._weights, before["weights"])
    assert torch.equal(state._log_weights, before["log_weights"])
    assert state._filter_binding == before["binding"]
    assert state.filter_revision == before["revision"] and state.filter_observation_uids == before["uids"]


def test_online_automatic_H3_update_reaches_later_prediction_without_live_rewind(online_warmup, monkeypatch, tmp_path):
    original, warm, inputs, private, generator_state = online_warmup
    episode = copy.deepcopy(warm)
    baseline_parameters = {name: value.detach().clone() for name, value in original.named_parameters()}
    requests, packets, controls, seen_windows = [], [], {}, []
    provider = _online_provider(episode, private, generator_state, requests, packets)
    raw_adapt = episode._adapt
    raw_loss = episode.model.partial_observation_window_loss
    raw_filter = episode.model.solver.update_filter_weights
    updating, filter_calls = False, []
    def count_filter(*args, **kwargs):
        assert not updating, "H3 updating must not re-filter history"
        filter_calls.append(kwargs["query_index"])
        return raw_filter(*args, **kwargs)
    def inspect_window(start, live, arrivals, normalization):
        assert updating and len(arrivals) == 3
        expected = list(episode._buffer)[-3]
        assert start is expected.start
        assert torch.equal(start.rhs._log_weights, expected.start.rhs._log_weights)
        seen_windows.append([row.acquisition.query_index for row in arrivals])
        return raw_loss(start, live, arrivals, normalization)
    def observe_adaptation(record):
        nonlocal updating
        before = _online_live_snapshot(episode)
        if record["query_index"] == 5:
            # Same actual post-filter carry, before optimizer updates, and no
            # sixth observation exists. This forecast is a software control.
            assert len(requests) == 1 and len(packets) == 1
            with torch.no_grad():
                _, controls["without_update"] = episode.model.advance_continuation(
                    episode.continuation, episode.normalization, query_count=7)
        updating = True
        try:
            raw_adapt(record)
        finally:
            updating = False
        _assert_online_live_snapshot(episode, before)
        if record["query_index"] == 5:
            with torch.no_grad():
                _, controls["with_update"] = episode.model.advance_continuation(
                    episode.continuation, episode.normalization, query_count=7)
            _assert_online_live_snapshot(episode, before)
            assert len(requests) == 1 and len(packets) == 1
    monkeypatch.setattr(episode, "_adapt", observe_adaptation)
    monkeypatch.setattr(episode.model, "partial_observation_window_loss", inspect_window)
    monkeypatch.setattr(episode.model.solver, "update_filter_weights", count_filter)
    def forbidden(*args, **kwargs):
        raise AssertionError("Online integration cannot draw a source or load future targets")
    with patch.object(source, "draw_group", forbidden), patch("Libs.utils.data.native_diffusion_data.load_records", forbidden), \
            patch("Libs.utils.model.dynamics_store.load_zarr_rows", forbidden), \
            patch("Libs.utils.model.dynamics_store.load_chunked_rows", forbidden):
        fifth = episode.step(provider)
        assert fifth["adaptation"]["status"] == "updated"
        assert fifth["adaptation"]["optimizer_updates_completed"] == 10
        assert fifth["adaptation"]["window_query_indices"] == [3, 4, 5]
        assert fifth["adaptation"]["window_start_filter_revision"] == 2
        assert not torch.equal(controls["with_update"], controls["without_update"])
        assert torch.isfinite(controls["with_update"]).all()
        retained = episode._buffer[-1].arrived.observation.values.clone()
        observed = packets[0].observation.observed
        packets[0].observation.values[observed] += 1000
        torch.testing.assert_close(episode._buffer[-1].arrived.observation.values, retained, rtol=0, atol=0, equal_nan=True)
        sixth = episode.step(provider)
    assert torch.equal(sixth["particle_fields"], controls["with_update"])
    assert sixth["adaptation"]["window_query_indices"] == [4, 5, 6]
    assert sixth["adaptation"]["window_start_filter_revision"] == 3
    assert filter_calls == [5, 6] and seen_windows == [[3, 4, 5]]*10+[[4, 5, 6]]*10
    assert episode.optimizer_updates == 20 and episode.observations_seen == 6
    assert episode._buffer.maxlen == 60 and len(episode._buffer) == 6
    assert episode.policy["optimizer"] == {"name": "Adam", "learning_rate": 1e-5,
        "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": 0}
    assert not episode.model.training and all(not module.training for module in episode.model.modules())
    changed = [name for name, value in episode.model.named_parameters() if not torch.equal(value, baseline_parameters[name])]
    assert changed
    assert all(torch.equal(value, baseline_parameters[name]) for name, value in original.named_parameters())
    assert all(torch.equal(value, baseline_parameters[name]) for name, value in warm.model.named_parameters())
    assert all(value.grad is None for value in episode.model.parameters())
    assert tuple(episode.model.solver.parameters()) == ()
    assert episode.continuation.rhs._log_weights.shape == (1, 32)
    assert sixth["particle_fields"].shape == (96, 1, 6, 6, 26)
    with pytest.raises(ValueError, match="exhausted"):
        episode.step(forbidden)
    evidence = {"scope": "actual_partial_observation_online_consumer_software",
        "source_generation_calls": 0, "future_target_loads": 0, "scientific_twin": "NOT_RUN",
        "full_MPC": "NOT_IMPLEMENTED", "online_calibration_qualified": False,
        "parameter_count": sum(value.numel() for value in episode.model.parameters()),
        "encoder_widths": [4, 8, 12], "bottleneck_width": 16,
        "particle_count": 32, "action_count": 3, "provider_calls": 6, "filter_updates": 6,
        "actual_original_source_times": [float(value) for value in episode.continuation.rhs._guide.clock],
        "source_time_unit": episode.continuation.rhs._guide.grid["query_time"]["unit"],
        "optimizer_updates": episode.optimizer_updates, "online_policy": episode.policy,
        "steps": episode.trace, "changed_phi_parameters": changed,
        "later_prediction_max_absolute_change": float((controls["with_update"]-controls["without_update"]).abs().max()),
        "later_prediction_exactly_reaches_next_consumer": True,
        "post_update_live_carry_and_filter_unchanged": True, "original_and_other_episode_unchanged": True,
        "future_accuracy_or_online_benefit_evaluated": False}
    destination = Path(os.environ.get("NRNSIFT_NATIVE_ONLINE_EVIDENCE", str(tmp_path/"online_consumer.json")))
    destination.write_text(json.dumps(evidence, indent=2)+"\n")


@pytest.mark.parametrize("failure", ["optimizer_exception", "nonfinite_parameters", "invalid_arrival"])
def test_online_failure_retains_progress_and_blocks_retry(online_warmup, monkeypatch, failure):
    _, warm, _, private, generator_state = online_warmup
    episode = copy.deepcopy(warm)
    original = {name: value.detach().clone() for name, value in episode.model.named_parameters()}
    requests, packets, state = [], [], {}
    provider = _online_provider(episode, private, generator_state, requests, packets)
    raw_adapt, raw_step = episode._adapt, episode.optimizer.step
    step_count = 0
    def optimizer_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        if failure == "optimizer_exception" and step_count == 2:
            raise RuntimeError("injected_after_one_real_optimizer_update")
        value = raw_step(*args, **kwargs)
        if failure == "nonfinite_parameters":
            with torch.no_grad():
                episode.model.output.bias[0] = float("nan")
        return value
    def snapshot_adapt(record):
        state.update(_online_live_snapshot(episode))
        raw_adapt(record)
    monkeypatch.setattr(episode, "_adapt", snapshot_adapt)
    monkeypatch.setattr(episode.optimizer, "step", optimizer_step)
    if failure == "invalid_arrival":
        provider = lambda request: episode._buffer[-1].arrived  # stale admitted query4
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        episode.step(provider)
    assert episode.status == episode.trace[-1]["status"] == "failed"
    assert episode.failed_arrival is not None
    assert episode.continuation.rhs.query_index == 5
    if failure == "invalid_arrival":
        assert episode.continuation.rhs.filter_revision == 4 and episode.optimizer_updates == 0
        assert all(torch.equal(value, original[name]) for name, value in episode.model.named_parameters())
    else:
        assert episode.optimizer_updates == episode.trace[-1]["optimizer_updates_completed"] == 1
        assert episode.continuation.rhs.filter_revision == 5
        _assert_online_live_snapshot(episode, state)
        assert any(not torch.equal(value, original[name]) for name, value in episode.model.named_parameters())
        if failure == "nonfinite_parameters":
            assert not torch.isfinite(episode.model.output.bias).all()
    before = _online_live_snapshot(episode)
    def forbidden(*args, **kwargs):
        raise AssertionError("A failed episode cannot issue another provider request")
    with pytest.raises(ValueError, match="Failed online"):
        episode.step(forbidden)
    _assert_online_live_snapshot(episode, before)


def test_online_policy_defaults_detached_and_offline_unbound_compatibility(online_warmup):
    original, warm, inputs, _, _ = online_warmup
    config = copy.deepcopy(original.config)
    policy = native_residual_online_policy(config)
    assert (policy["horizon_transitions"], policy["warmup_transitions"],
            policy["buffer_transitions"], policy["updates_per_observation"]) == (3, 5, 60, 10)
    config["native_mechanistic_residual_online"]["optimizer"]["learning_rate"] = .1
    assert policy["optimizer"]["learning_rate"] == 1e-5
    config.pop("native_mechanistic_residual_online")
    settings = config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"]
    assert settings["online_adaptation"] == "unbound"
    legacy = NativeMechanisticResidual(settings, inputs, config)
    assert legacy.prediction_kind == "offline_one_step_mechanistic_residual_conditional_mean"
    legacy.load_state_dict(original.state_dict(), strict=True)
    assert set(legacy.state_dict()) == set(original.state_dict())
    with pytest.raises(ValueError, match="missing required"):
        NativeResidualOnlineEpisode(legacy, inputs, warm.normalization, warm.readout,
                                    executed_regimen_uid=warm.executed_regimen_uid)


@pytest.fixture(scope="module")
def execution_cases(cases):
    def forbidden(*args,**kwargs):
        raise AssertionError("Actual execution software checks cannot draw datasets or read future teacher targets")
    with patch.object(source,"draw_group",forbidden), \
            patch("Libs.utils.data.native_diffusion_data.load_records",forbidden), \
            patch("Libs.utils.model.dynamics_store.load_zarr_rows",forbidden), \
            patch("Libs.utils.model.dynamics_store.load_chunked_rows",forbidden):
        yield cases


def _actual_execution_case(cases, family, *, initial_control_index=0, zero_mass=False,
                           molecular_administration=False, model_seed=1729, measurement_seed=9151):
    """Retained public model inputs plus an independent explicit nominal source world."""
    import numpy as np
    from Libs.utils.data.native_actuation import NativeObservationIdentity, native_actuation_library
    from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
    from Libs.utils.model import controlled_cns_source as cns
    from Libs.utils.evaluation.native_physical_plant import NativePhysicalPlant
    config,inputs,norm=cases["validation",family]
    if zero_mass:
        logs = inputs.posterior.log_weights.clone()
        logs[:, (0, 7, 31)] = -float("inf")
        logs = _normalized_log_weights(logs)
        posterior = replace(inputs.posterior, log_weights=logs, posterior_uid=_posterior_uid(
            inputs.posterior.support, inputs.posterior.observation_uids,
            inputs.posterior.parent_posterior_uid, logs, inputs.posterior.particle_context))
        posterior.assert_current_binding()
        programs = dict(inputs.programs, posterior_uid=posterior.posterior_uid,
            source_posterior_uid=inputs.programs.get("source_posterior_uid", inputs.posterior_uid))
        inputs = replace(inputs, posterior=posterior, posterior_uid=posterior.posterior_uid,
            weights=posterior.weights, programs=programs)
    config=copy.deepcopy(config)
    current=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    for key in ("native_arrived_observation_filter","native_mechanistic_residual_online","native_future_controls"):
        config[key]=copy.deepcopy(current[key])
        if key in current["schemas"]:config["schemas"][key]=copy.deepcopy(current["schemas"][key])
    if molecular_administration:
        opt_in=load_config("Libs/configs/exp/native_molecular_administered_rate.yaml")
        config["native_molecular_administration"]=copy.deepcopy(opt_in["native_molecular_administration"])
    settings=config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"]
    settings.update(encoder_widths=[4,8,12],bottleneck_width=16)
    with torch.random.fork_rng():
        torch.manual_seed(model_seed)
        original=NativeMechanisticResidual(settings,inputs,config).eval()
    group=next(row for row in source.selected_group_definitions(config,"validation") if row["family_uid"]==family)
    # Never rewrite source group IDs to force the consumer/plant binding to pass.
    assert tuple(group["particle_uids"])==inputs.particle_uids[0]
    assert group["support_uid"]==inputs.posterior.support.support_uid
    readout=native_cns_readout(config,group)
    episode=NativeResidualOnlineEpisode(original,inputs,norm,readout,
        executed_regimen_uid=inputs.regimen_uids[initial_control_index])
    identity=NativeObservationIdentity.from_context(episode.model.solver.filter_context(episode.continuation.rhs))
    library=native_actuation_library(config,group)
    assert tuple(row.regimen_uid for row in library)==inputs.regimen_uids
    scoped=source.scoped_group_config(config,group)
    background=source.long_cns_definition(scoped) if family==FAMILIES[-1] else scoped["controlled_cns_source"]
    factors={key:np.ones(len(value)) for key,value in background["factor_basis"].items()}
    initial=np.zeros((6,6,26),dtype=np.float32)
    names=background["grid"]["species_id"]
    for name in cns.DYNAMIC:initial[...,names.index(name)]=.1
    law=cns.ControlledLaw(background,factors,initial[...,:15])
    dynamic=initial[...,:15].reshape(1,36,15)[:,:,[names.index(name) for name in cns.DYNAMIC]]
    initial[...,:15]=law.quantities([0.],dynamic,background["actions"][0])[0]
    parameters=None if family==FAMILIES[-1] else {
        key:float(value["value"]) for key,value in scoped["native_observable_bridge"]["physical_families"][family]["parameters"].items()}
    grid=family_grid(observable_profile(scoped),family)
    mask=quantity_support(grid,readout.source_uid)[0]
    for qi,quantity in enumerate(grid["quantity_profile"]["quantities"]):
        if quantity.get("family_uid") != family:continue
        site=quantity["incidence"]
        ri=grid["region_id"].index(site["anchor_region_id"]);ci=grid["compartment_id"].index(site["anchor_compartment_id"])
        if mask[ri,ci,qi]:initial[ri,ci,qi]=parameters.get("initial_"+quantity["observable_name"],0.)
    plant=NativePhysicalPlant(config,group,private_initial_field=initial,private_factors=factors,
        private_physical_parameters=parameters,initial_actuation=library[initial_control_index],
        measurement_generator=torch.Generator().manual_seed(measurement_seed),observation_identity=identity)
    episode.bind_execution(library,library[initial_control_index],plant.observation_identity)
    return original,episode,plant,library


@pytest.mark.parametrize("family",FAMILIES)
def test_verified_execution_unchanged_conditioning_keeps_original_program(execution_cases,family):
    from Libs.utils.data.native_actuation import NativeActuationRequest
    original,episode,plant,library=_actual_execution_case(execution_cases,family)
    episode.step(plant.acquire)
    before=episode.continuation
    expected=before.program.reshape(1,32,before.rhs._guide.a,*before.program.shape[1:])[:,:,0].reshape(32,*before.program.shape[1:]).clone()
    branch=episode.plan_controls([library[0].regimen_uid])
    episode.activate_selected_control(branch,NativeActuationRequest(plant.prefix,library[0]),plant.switch)
    assert torch.equal(episode.continuation.program,expected)
    assert episode.continuation.rhs._execution_model_uid != episode.executed_regimen_uid
    assert episode.continuation.rhs._public_control.regimen_uid==plant.prefix.executed_regimen_uid
    assert episode.observations_seen==1 and episode.optimizer_updates==0
    assert episode.continuation.rhs._work_ledger is before.rhs._work_ledger
    result=episode.step(plant.acquire)
    assert result["filtering"]["filtering_revision"]==2 and plant._query==2
    assert plant._source_prior_uid==plant._group["posterior_uid"]
    assert plant.observation_identity.origin_posterior_uid==episode._execution_inputs.posterior_uid


def test_actual_switch_H3_last_time_gradient_and_online_update_reach_next_query(execution_cases,monkeypatch,tmp_path):
    from Libs.utils.data.native_actuation import NativeActuationRequest, compose_actuation, executed_prefix
    from Libs.utils.model.native_future_controls import conditioning_identity
    original,episode,plant,library=_actual_execution_case(execution_cases,FAMILIES[0],initial_control_index=2)
    parent_weights={key:value.clone() for key,value in original.state_dict().items()}
    other=copy.deepcopy(original)
    # Fix and validate both public requests before seeing any arrived outcomes.
    initial,first,second=library[2],library[1],library[2]
    first_request=NativeActuationRequest(executed_prefix(initial,1),first)
    first_program=compose_actuation(first_request,library)
    second_request=NativeActuationRequest(executed_prefix(first_program,3),second)
    second_program=compose_actuation(second_request,library)
    assert initial.source_times==(0.,2.,4.,6.,8.,10.,12.)
    assert initial.interval_values==(0.,3.2,3.2,3.2,0.,0.)
    assert first_program.interval_values==(0.,4.8,4.8,0.,0.,0.)
    assert second_program.interval_values==(0.,4.8,4.8,3.2,0.,0.)
    evidence={"status":"public_schedule_validated_before_observations",
        "public_planned_programs":[asdict(item) for item in (initial,first_program,second_program)],
        "actual_public_programs":[dict(activated_at_query=0,**asdict(initial))]}
    destination=Path(os.environ.get("NRNSIFT_EXECUTED_H3_EVIDENCE",tmp_path))/"actual_execution_h3.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(evidence,indent=2)+"\n")
    requests=[];packets=[];predictions=[]
    def provider(request):
        requests.append(request);packet=plant.acquire(request);packets.append(packet);return packet
    actual_predict=residual_module.predict_native_readout
    def observe_predict(*args,**kwargs):
        result=actual_predict(*args,**kwargs);predictions.append(result);return result
    monkeypatch.setattr(residual_module,"predict_native_readout",observe_predict)
    original_adapt=episode._adapt
    next_before=[];next_after=[]
    def forecast():
        carry,_=episode.model.advance_continuation(episode.continuation,episode.normalization,query_count=7)
        return actual_predict(episode.model.solver,carry.rhs,episode.readout,
                              executed_regimen_uid=episode.executed_regimen_uid).mean.detach().clone()
    def adapt(record):
        if record["query_index"]==5:
            snapshot=_online_live_snapshot(episode);next_before.append(forecast())
            source_state=torch.from_numpy(plant._state).clone()
            source_aux=torch.from_numpy(plant._physical).clone()
            source_rng=plant._generator.get_state().clone()
            source_clock=(plant._query,plant._clock,plant.prefix)
            original_adapt(record)
            _assert_online_live_snapshot(episode,snapshot);next_after.append(forecast())
            assert torch.equal(torch.from_numpy(plant._state),source_state)
            assert torch.equal(torch.from_numpy(plant._physical),source_aux)
            assert torch.equal(plant._generator.get_state(),source_rng)
            assert (plant._query,plant._clock,plant.prefix)==source_clock
        else:original_adapt(record)
    monkeypatch.setattr(episode,"_adapt",adapt)
    history_versions=[]
    for query in range(1,6):
        episode.step(provider)
        if query==2:
            assert plant.prefix.interval_values[1]==first_program.interval_values[1] != initial.interval_values[1]
        if query==4:
            assert plant.prefix.interval_values[3]==second_program.interval_values[3] != first_program.interval_values[3]
        if query in (1,3):
            request=first_request if query==1 else second_request
            assert request.prefix==plant.prefix
            old=episode.continuation
            old_program=old.program.clone()
            branch=episode.plan_controls([row.regimen_uid for row in library])
            episode.activate_selected_control(branch,request,plant.switch)
            active=episode.continuation.rhs._public_control
            assert active==(first_program if query==1 else second_program)
            evidence["actual_public_programs"].append(dict(activated_at_query=query,**asdict(active)))
            evidence["actual_consumed_interval_values"]=list(plant.prefix.interval_values)
            evidence["status"]="executing"
            destination.write_text(json.dumps(evidence,indent=2)+"\n")
            assert torch.equal(old.program,old_program)
            assert episode.continuation._conditioning_identity==conditioning_identity(episode.continuation.program,episode.continuation.static)
            history_versions.append((old,old_program))
    assert episode.optimizer_updates==10 and episode.observations_seen==5
    assert [row.arrived.acquisition.query_index for row in list(episode._buffer)[-3:]]==[3,4,5]
    assert packets[2].acquisition.executed_regimen_uid != packets[3].acquisition.executed_regimen_uid
    assert not torch.equal(next_before[0],next_after[0])
    # Prove the final time alone reaches the earliest pre-switch residual. A
    # nonzero early-time loss cannot conceal a detach at the real switch.
    window=list(episode._buffer)[-3:]
    deltas=[];ells=[]
    residual=episode.model.normalized_residual
    likelihood_fn=residual_module.retained_log_likelihood
    def capture_delta(*args,**kwargs):
        value=residual(*args,**kwargs);deltas.append(value);return value
    def capture_likelihood(*args,**kwargs):
        value=likelihood_fn(*args,**kwargs);ells.append(value);return value
    monkeypatch.setattr(episode.model,"normalized_residual",capture_delta)
    monkeypatch.setattr(residual_module,"retained_log_likelihood",capture_likelihood)
    live=_online_live_snapshot(episode)
    loss,diagnostic=episode.model.partial_observation_window_loss(window[0].start,episode.continuation,
        tuple(row.arrived for row in window),episode.normalization)
    prior=window[0].start.rhs._log_weights.detach()
    assert torch.equal(loss,-torch.logsumexp(prior+torch.stack(ells).sum(0),-1).sum())
    last_only=-torch.logsumexp(prior+ells[-1],-1).sum()
    gradient=torch.autograd.grad(last_only,deltas[0])[0]
    assert torch.isfinite(gradient).all() and gradient.count_nonzero()>0
    _assert_online_live_snapshot(episode,live)
    monkeypatch.setattr(episode.model,"normalized_residual",residual)
    monkeypatch.setattr(residual_module,"retained_log_likelihood",likelihood_fn)
    recorded=episode.continuation.rhs._execution_history[-1]
    saved_program=recorded["program"].clone()
    recorded["program"][0,0,0,0]+=1.
    def forbidden(*args,**kwargs):raise AssertionError("Changed execution history reached the residual network")
    monkeypatch.setattr(episode.model,"normalized_residual",forbidden)
    with pytest.raises(ValueError,match="history|conditioning"):
        episode.model.partial_observation_window_loss(window[0].start,episode.continuation,
            tuple(row.arrived for row in window),episode.normalization)
    recorded["program"].copy_(saved_program)
    monkeypatch.setattr(episode.model,"normalized_residual",residual)
    episode.step(provider)
    assert torch.equal(predictions[-1].mean,next_after[0])
    assert len(requests)==len(packets)==plant._query==episode.continuation.rhs.filter_revision==6
    assert episode.optimizer_updates==20 and episode.observations_seen==6
    assert all(torch.equal(original.state_dict()[key],value) and torch.equal(other.state_dict()[key],value)
               for key,value in parent_weights.items())
    for old,program in history_versions:assert torch.equal(old.program,program)
    evidence.update({"status":"component_check_passed",
        "scope":"artificial_same_world_selected_execution_H3_not_full_MPC",
        "particle_count":32,"source_query_indices":list(range(7)),"switch_queries":[1,3],
        "source_query_values":list(plant._source_times),"source_time_unit":plant._grid["query_time"]["unit"],
        "model_parameter_count":sum(value.numel() for value in episode.model.parameters()),
        "selection":"fixed_public_library_software_check_not_optimized_policy",
        "selected_regimen_uids":[initial.regimen_uid,first.regimen_uid,second.regimen_uid],
        "actual_consumed_interval_values":list(plant.prefix.interval_values),
        "provider_calls":len(requests),"filter_calls":episode.continuation.rhs.filter_revision,
        "optimizer_updates":episode.optimizer_updates,"H3_query_indices":diagnostic["query_indices"],
        "last_time_only_to_earliest_delta_gradient_nonzero":int(gradient.count_nonzero()),
        "source_rhs_calls":plant._work["nfev"],"model_rhs_tuple_action_work":episode.continuation.rhs._work_ledger.spent,
        "scientific_qualification":"NOT_EVALUATED","original_dataset_generation":False,"protected_reads":False})
    destination.write_text(json.dumps(evidence,indent=2)+"\n")


@pytest.mark.parametrize("fault",["prefix","forecast","state","observation_identity"])
def test_actual_activation_rejects_bad_origin_before_actuator(execution_cases,fault):
    from Libs.utils.data.native_actuation import NativeActuationRequest
    _,episode,plant,library=_actual_execution_case(execution_cases,FAMILIES[0])
    episode.step(plant.acquire)
    branch=episode.plan_controls([library[0].regimen_uid])
    request=NativeActuationRequest(plant.prefix,library[0])
    if fault=="prefix":request=replace(request,prefix=replace(request.prefix,interval_values=(99.,)))
    elif fault=="forecast":episode.model.solver.advance_continuation(branch,query_count=3)
    elif fault=="state":branch.cns_state=branch.cns_state+.001
    else:branch.filter_observation_uids=(*branch.filter_observation_uids,"fabricated")
    old_prefix=plant.prefix;old_rng=plant._generator.get_state().clone();work=plant._work.copy()
    calls=[]
    def actuator(value):calls.append(value);return plant.switch(value)
    with pytest.raises(ValueError):episode.activate_selected_control(branch,request,actuator)
    assert not calls and plant.prefix==old_prefix and plant._work==work
    assert torch.equal(plant._generator.get_state(),old_rng)


def test_actual_committed_actuator_failure_cannot_repeat_effect(execution_cases):
    from Libs.utils.data.native_actuation import NativeActuationRequest
    _,episode,plant,library=_actual_execution_case(execution_cases,FAMILIES[0])
    episode.step(plant.acquire)
    branch=episode.plan_controls([library[0].regimen_uid]);request=NativeActuationRequest(plant.prefix,library[0])
    calls=[]
    def actuator(value):
        calls.append(value);plant.switch(value);raise RuntimeError("injected after actual actuation commit")
    with pytest.raises(RuntimeError,match="after actual"):episode.activate_selected_control(branch,request,actuator)
    committed=plant.prefix
    assert episode.status=="failed" and committed.executed_regimen_uid != request.prefix.executed_regimen_uid
    with pytest.raises(ValueError):episode.activate_selected_control(branch,request,actuator)
    with pytest.raises(ValueError):episode.step(plant.acquire)
    assert len(calls)==1 and plant.prefix==committed


def test_source_observation_identity_preserves_original_group_and_tuple_order(execution_cases):
    import numpy as np
    from Libs.utils.evaluation.native_physical_plant import NativePhysicalPlant
    _,episode,plant,library=_actual_execution_case(execution_cases,FAMILIES[0])
    identity=plant.observation_identity
    variants=[replace(identity,source_uid="foreign"),replace(identity,support_uid="foreign"),
        replace(identity,particle_uids=(tuple(reversed(identity.particle_uids[0])),)),
        replace(identity,family_uid=FAMILIES[1])]
    for invalid in variants:
        with pytest.raises(ValueError,match="original source group"):
            NativePhysicalPlant(episode.model.config,plant._group,
                private_initial_field=plant._history[0]["field"],
                private_factors={key:np.asarray(value) for key,value in plant._law.law.factors.items()},
                private_physical_parameters=plant._parameters,initial_actuation=library[0],
                measurement_generator=torch.Generator(),observation_identity=invalid)


def test_selected_coordinates_run_fifth_and_sixth_actual_H3_updates(execution_cases, monkeypatch, tmp_path):
    original, episode, plant, _ = _actual_execution_case(execution_cases, FAMILIES[0])
    original_parameters = {name: value.detach().clone() for name, value in original.named_parameters()}
    available = [index for index, retained in enumerate(episode.readout.observed) if retained]
    assert len(available) >= 2
    choices = available[-2:]
    raw_loss = episode.model.partial_observation_window_loss
    windows = []
    def inspect_loss(start, live, arrivals, normalization):
        loss, diagnostic = raw_loss(start, live, arrivals, normalization)
        assert diagnostic["particle_count"] == 32 and diagnostic["observed_counts"] == [1, 1, 1]
        assert diagnostic["observed_count"] == 3
        assert torch.isfinite(loss) and loss.requires_grad
        windows.append({"indices": diagnostic["query_indices"],
                        "coordinates": [arrival.acquisition.selected_coordinate for arrival in arrivals]})
        return loss, diagnostic
    monkeypatch.setattr(episode.model, "partial_observation_window_loss", inspect_loss)
    before = _online_live_snapshot(episode)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid coordinate must reject before provider or model advance")
    with pytest.raises(ValueError, match="available mixed readout"):
        episode.step(forbidden, selected_coordinate=99)
    _assert_online_live_snapshot(episode, before)
    assert not episode.trace and episode.status == "ready" and plant._query == 0
    for query in range(1, 7):
        result = episode.step(plant.acquire, selected_coordinate=choices[(query-1) % 2])
        assert result["prediction"].acquisition.readout == episode.readout
        assert result["filtering"]["filtering_revision"] == query
        assert result["adaptation"]["optimizer_updates_completed"] == (10 if query >= 5 else 0)
        assert result["adaptation"]["status"] == ("updated" if query >= 5 else "warmup")
        packet = episode._buffer[-1].arrived
        assert int(packet.observation.observed.sum()) == 1
        assert torch.count_nonzero(packet.observation.values[~packet.observation.observed]) == 0
    assert [row["indices"] for row in windows] == [[3, 4, 5]]*10+[[4, 5, 6]]*10
    assert windows[0]["coordinates"] == [choices[0], choices[1], choices[0]]
    assert windows[-1]["coordinates"] == [choices[1], choices[0], choices[1]]
    assert episode.observations_seen == plant._query == 6 and episode.optimizer_updates == 20
    assert episode.continuation.rhs._log_weights.shape == (1, 32)
    assert any(not torch.equal(value, original_parameters[name]) for name, value in episode.model.named_parameters())
    assert all(torch.equal(value, original_parameters[name]) for name, value in original.named_parameters())
    window = list(episode._buffer)[-3:]
    arrivals = tuple(item.arrived for item in window)
    last = arrivals[-1]
    altered_coordinate = choices[0]
    mask = torch.tensor(native_acquisition_mask(episode.readout, altered_coordinate))[None]
    altered = replace(last, acquisition=replace(last.acquisition, selected_coordinate=altered_coordinate),
                      observation=replace(last.observation, observed=mask, values=torch.zeros_like(last.observation.values)))
    with pytest.raises(ValueError, match="payload or acquisition changed"):
        admitted_window_binding(episode.model.solver, window[0].start.rhs, episode.continuation.rhs,
                                (*arrivals[:-1], altered))
    evidence = {"scope": "selected_coordinate_explicit_world_H3_software_primitive",
        "particle_count": 32, "observations_seen": episode.observations_seen,
        "optimizer_updates": episode.optimizer_updates, "observed_counts_per_H3": [1, 1, 1],
        "windows": windows, "source_rhs_calls": plant._work["nfev"],
        "model_rhs_tuple_action_work": episode.continuation.rhs._work_ledger.spent,
        "postzero_joint_policy": "NOT_IMPLEMENTED", "scientific_qualification": "NOT_EVALUATED"}
    destination = Path(os.environ.get("NRNSIFT_NATIVE_COORDINATE_EVIDENCE", str(tmp_path/"selected_coordinate_h3.json")))
    destination.write_text(json.dumps(evidence, indent=2)+"\n")


@pytest.fixture(scope="module")
def adaptive_executed_h3(execution_cases):
    """Actual controlled advances/filtering; synthetic scales, no fit or optimizer."""
    from Libs.utils.data.native_actuation import NativeActuationRequest, executed_prefix
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference, FixedReadoutReference
    _, episode, plant, library = _actual_execution_case(execution_cases, FAMILIES[0], initial_control_index=2)
    # This interface check freezes H3 during observation collection. Production
    # warmup/Adam settings remain unchanged. No training or calibration is done.
    with patch.object(episode, "_adapt", lambda record: record.update(status="software_history_capture_only")):
        for query in range(1, 6):
            episode.step(plant.acquire)
            if query in (1, 3):
                action = library[1] if query == 1 else library[2]
                request = NativeActuationRequest(executed_prefix(episode.continuation.rhs._public_control, query), action)
                branch = episode.plan_controls([row.regimen_uid for row in library])
                episode.activate_selected_control(branch, request, plant.switch)
    first = episode._buffer[0]
    context = episode.model.solver.filter_context(first.start.rhs)
    binding = {"normalization_id": episode.normalization.identity, "checkpoint_sha256": "synthetic_H3_software_predictor"}
    fixed = FixedReadoutReference(episode.readout, context["support_uid"], context["origin_posterior_uid"],
        tuple(context["sample_uids"]), tuple(tuple(row) for row in context["particle_uids"]), library[2].regimen_uid,
        json.dumps(binding, sort_keys=True), episode.normalization.identity, "synthetic_inherited_fixed_residual_scale",
        (.01,) * 26, tuple(episode.normalization.scale.tolist()), max(first.start.rhs._times),
        tuple(library[2].source_times), tuple(first.start.rhs._times), library[2].source_time_unit, 0)
    reference = AdaptiveReadoutReference(fixed, library, library[2].regimen_uid)
    window = episode.prospective_adaptive_h3_window(reference, trial_predictor_binding=binding)
    return episode, plant, window


def test_adaptive_h3_consumer_uses_joint_conditional_law_and_frozen_preupdate_history(adaptive_executed_h3, monkeypatch):
    from Libs.utils.model.native_conditional_readout import adaptive_readout_window
    episode, plant, window = adaptive_executed_h3
    buffer = list(episode._buffer)[-3:]
    arrivals = tuple(row.arrived for row in buffer)
    live = _online_live_snapshot(episode)
    source_clock = (plant._query, plant._clock, plant.prefix)
    rng = plant._generator.get_state().clone()
    frozen = copy.deepcopy(window)
    means, deltas = [], []
    real_projection, real_residual = residual_module.project_cns_readout, episode.model.normalized_residual
    def project(*args, **kwargs):
        mean = real_projection(*args, **kwargs)
        means.append(mean)
        return mean
    def residual(*args, **kwargs):
        delta = real_residual(*args, **kwargs)
        deltas.append(delta)
        return delta
    def forbidden(*args, **kwargs):
        raise AssertionError("Adaptive H3 must not refilter, fit scales, load targets or use independent sensor likelihoods")
    monkeypatch.setattr(residual_module, "project_cns_readout", project)
    monkeypatch.setattr(episode.model, "normalized_residual", residual)
    monkeypatch.setattr(residual_module, "retained_log_likelihood", forbidden)
    monkeypatch.setattr(episode.model.solver, "update_filter_weights", forbidden)
    loss, diagnostic = episode.model.partial_observation_window_loss(buffer[0].start, episode.continuation,
        arrivals, episode.normalization, adaptive_window=window)
    assert len(means) == 3 and loss.requires_grad
    queries = tuple(AdaptiveReadoutQuery(replace(row.context, predictor_binding=window.trial_predictor_binding),
        row.query_index, row.filtering_revision, tuple(arrived.observation.observed[0].tolist()), mean[0],
        arrived.observation.values[0]) for row, arrived, mean in zip(window.accepted, arrivals, means))
    joint = adaptive_readout_window(window.reference, window.history, queries)
    prior = _normalized_log_weights((joint.history_log_likelihood.new_tensor(window.initial_log_weights)
                                     + joint.history_log_likelihood)[None])
    expected = -torch.logsumexp(prior + joint.log_likelihood[None], -1).sum()
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(prior, buffer[0].start.rhs._log_weights, rtol=1e-7, atol=1e-7)
    gradient = torch.autograd.grad(loss, deltas[0], retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
    trial_gradients = torch.autograd.grad(loss, means)
    assert all(torch.isfinite(value).all() and value.count_nonzero() > 0 for value in trial_gradients)
    assert diagnostic["adaptive_scientific_qualification"] == "UNQUALIFIED"
    assert diagnostic["history_observation_count"] == 2 and diagnostic["online_calibration_qualified"] is False
    assert "live_policy_uses_adaptive_law" not in diagnostic
    assert diagnostic["prior_authority"] == "initial_tuple_weights_plus_same_law_frozen_history"
    assert [row.context.original_action.regimen_uid for row in window.accepted] == [
        window.reference.original_actions[1].regimen_uid, window.reference.initial_action_uid, window.reference.initial_action_uid]
    assert window == frozen and episode.optimizer_updates == 0
    _assert_online_live_snapshot(episode, live)
    assert (plant._query, plant._clock, plant.prefix) == source_clock
    assert torch.equal(plant._generator.get_state(), rng)
    # Changing only the trial predictor leaves all historical pre-update means,
    # values, filter mass and physical carry intact.
    with torch.no_grad():
        episode.model.output.bias[0] += .001
    updated_binding = dict(json.loads(window.trial_predictor_binding), residual_optimizer_updates=1)
    updated = episode.prospective_adaptive_h3_window(window.reference, trial_predictor_binding=updated_binding)
    assert updated.history == window.history and updated.accepted == window.accepted
    with torch.no_grad():
        episode.model.output.bias[0] -= .001
    _assert_online_live_snapshot(episode, live)


@pytest.mark.parametrize("fault", ["tuple", "clock", "scale", "history", "weights", "packet", "switch", "binding", "qualification", "older_mean", "older_value", "accepted_mean", "normalized_weights", "nan_prior", "positive_inf_prior", "all_impossible_prior"])
def test_adaptive_h3_rejects_changed_bindings_before_residual_execution(adaptive_executed_h3, monkeypatch, fault):
    episode, _, window = adaptive_executed_h3
    buffer = list(episode._buffer)[-3:]
    arrivals = tuple(row.arrived for row in buffer)
    if fault in {"tuple", "clock", "scale"}:
        fixed = window.reference.fixed
        if fault == "tuple": fixed = replace(fixed, particle_uids=(tuple(reversed(fixed.particle_uids[0])),))
        if fault == "clock": fixed = replace(fixed, time_hours=tuple(t * 2 for t in fixed.time_hours))
        if fault == "scale": fixed = replace(fixed, inverse_normalizer_scales=(2.,) * 26)
        window = replace(window, reference=replace(window.reference, fixed=fixed))
    if fault == "history": window = replace(window, history=window.history[:-1])
    if fault == "weights": window = replace(window, initial_log_weights=(0.,) * 32)
    if fault.endswith("_prior"):
        weights = window.initial_log_weights
        if fault == "nan_prior": weights = (float("nan"), *weights[1:])
        elif fault == "positive_inf_prior": weights = (float("inf"), *weights[1:])
        elif fault == "all_impossible_prior": weights = (-float("inf"),) * 32
        window = replace(window, initial_log_weights=weights)
    if fault == "normalized_weights":
        wrong = _normalized_log_weights(torch.arange(32, dtype=torch.float64)[None] / 32)[0]
        window = replace(window, initial_log_weights=tuple(wrong.tolist()))
    if fault == "packet":
        changed = replace(window.accepted[0], values=tuple(v + .1 for v in window.accepted[0].values))
        window = replace(window, accepted=(changed, *window.accepted[1:]))
    if fault in {"older_mean", "older_value", "accepted_mean"}:
        row = window.accepted[0] if fault == "accepted_mean" else window.history[0]
        changed = replace(row, values=tuple(v + .1 for v in row.values)) if fault == "older_value" else replace(
            row, mean=tuple(tuple(v + .1 for v in values) for values in row.mean))
        if fault == "accepted_mean": window = replace(window, accepted=(changed, *window.accepted[1:]))
        else: window = replace(window, history=(changed, *window.history[1:]))
    if fault == "switch":
        changed = replace(window.accepted[0], context=replace(window.accepted[0].context, switches=()))
        window = replace(window, accepted=(changed, *window.accepted[1:]))
    if fault == "binding": window = replace(window, trial_predictor_binding=json.dumps({"normalization_id": "wrong"}))
    if fault == "qualification": window = replace(window, reference=replace(window.reference, scientific_qualification="QUALIFIED"))
    spent = episode.continuation.rhs._work_ledger.spent
    before = _online_live_snapshot(episode)
    parameters = {name: value.detach().clone() for name, value in episode.model.named_parameters()}
    optimizer = copy.deepcopy(episode.optimizer.state_dict())
    def forbidden(*args, **kwargs): raise AssertionError("Invalid adaptive binding reached residual physics")
    monkeypatch.setattr(episode.model, "normalized_residual", forbidden)
    with pytest.raises(ValueError):
        episode.model.partial_observation_window_loss(buffer[0].start, episode.continuation,
            arrivals, episode.normalization, adaptive_window=window)
    assert episode.continuation.rhs._work_ledger.spent == spent
    _assert_online_live_snapshot(episode, before)
    assert all(torch.equal(value, parameters[name]) for name, value in episode.model.named_parameters())
    assert episode.optimizer.state_dict() == optimizer


def _episode_field_projection(fixed):
    readout = fixed.readout
    compartments, species = len(readout.compartment_ids), len(readout.species_ids)
    projection = torch.zeros(6, len(readout.region_ids)*compartments*species, dtype=torch.float64)
    for output in range(6):
        for selected, (region, compartment, coordinate) in enumerate(readout.selector):
            column = (region*compartments+compartment)*species+coordinate
            projection[output,column] += (readout.matrix[output][selected]
                * fixed.inverse_normalizer_scales[coordinate]*fixed.residual_scales[coordinate]/readout.scales[selected])
    return projection


def _episode_dense_log_density(reference, rows):
    projection = _episode_field_projection(reference.fixed)
    base = projection @ projection.T
    blocks = [(row.coordinates,reference.fixed.time_hours[row.query_index],row.context.original_action.regimen_uid) for row in rows]
    covariance = torch.cat([torch.cat([base[list(a)][:,list(b)]*
        (min(t,s)/reference.fixed.reference_time_hours if action == other else 0.) for b,s,other in blocks],1)
        for a,t,action in blocks],0)
    covariance += torch.eye(len(covariance),dtype=torch.float64)*reference.fixed.readout.standard_deviation**2
    mean = torch.cat([torch.tensor(row.mean,dtype=torch.float64) for row in rows],1)
    values = torch.cat([torch.tensor(row.values,dtype=torch.float64) for row in rows])
    return torch.distributions.MultivariateNormal(mean,covariance_matrix=covariance).log_prob(values)


@pytest.fixture(scope="module")
def adaptive_online_episode(execution_cases):
    return _build_adaptive_online_episode(execution_cases)


@pytest.fixture(scope="module")
def zero_mass_adaptive_online_episode(execution_cases):
    return _build_adaptive_online_episode(execution_cases, zero_mass=True)


def _build_adaptive_online_episode(execution_cases, *, zero_mass=False):
    """Actual electrical A/B/A acquisitions with the prescribed two10-update H3 events."""
    from Libs.utils.data.native_actuation import NativeActuationRequest, executed_prefix
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference, FixedReadoutReference
    original, legacy, plant, library = _actual_execution_case(execution_cases, FAMILIES[0], initial_control_index=2,
        zero_mass=zero_mass)
    state = legacy.continuation.rhs
    context = legacy.model.solver.filter_context(state)
    binding = {"normalization_id": legacy.normalization.identity, "checkpoint_sha256": "declared_synthetic_episode_predictor"}
    fixed = FixedReadoutReference(legacy.readout, context["support_uid"], context["origin_posterior_uid"],
        tuple(context["sample_uids"]), tuple(tuple(row) for row in context["particle_uids"]), library[2].regimen_uid,
        json.dumps(binding, sort_keys=True), legacy.normalization.identity, "synthetic_software_scale_not_scientific_fit",
        (.01,) * 26, tuple(legacy.normalization.scale.tolist()), max(state._times),
        tuple(library[2].source_times), tuple(state._times), library[2].source_time_unit, 0)
    reference = AdaptiveReadoutReference(fixed, library, library[2].regimen_uid)
    episode = NativeResidualOnlineEpisode(original, legacy._execution_inputs, legacy.normalization, legacy.readout,
        executed_regimen_uid=library[2].regimen_uid, adaptive_reference=reference)
    episode.bind_execution(library, library[2], plant.observation_identity)
    packets, results, updates, windows, live_snapshots, dense_checks = [], [], [], [], [], []
    raw_loss, raw_adapt, raw_filter = episode.model.partial_observation_window_loss, episode._adapt, episode.model.solver.update_filter_weights
    def inspect_loss(start, live, arrivals, normalization, *, adaptive_window=None):
        assert adaptive_window is not None
        windows.append(adaptive_window)
        loss, diagnostic = raw_loss(start, live, arrivals, normalization, adaptive_window=adaptive_window)
        assert diagnostic["readout_law"] == reference.law
        return loss, diagnostic
    def inspect_adapt(record):
        before = _online_live_snapshot(episode)
        saved = episode.continuation._accepted_readout_snapshots
        raw_adapt(record)
        _assert_online_live_snapshot(episode, before)
        assert episode.continuation._accepted_readout_snapshots == saved
        live_snapshots.append(before)
    def inspect_filter(*args, **kwargs):
        updates.append(kwargs["query_index"])
        return raw_filter(*args, **kwargs)
    episode.model.partial_observation_window_loss = inspect_loss
    episode._adapt = inspect_adapt
    episode.model.solver.update_filter_weights = inspect_filter
    def provider(acquisition):
        packet = plant.acquire(acquisition)
        packets.append(packet)
        return packet
    for index in range(1, 7):
        selected = None if index % 2 == 0 else next(i for i, keep in enumerate(episode.readout.observed) if keep)
        result = episode.step(provider, selected_coordinate=selected)
        results.append(result)
        history = episode._adaptive_history()
        expected = _normalized_log_weights((torch.tensor(episode._adaptive_initial_log_weights, dtype=torch.float64)
            + _episode_dense_log_density(reference, history))[None])
        torch.testing.assert_close(episode.continuation.rhs._log_weights, expected, atol=2e-10, rtol=2e-10)
        finite = torch.isfinite(expected)
        assert torch.equal(torch.isneginf(episode.continuation.rhs._log_weights), torch.isneginf(expected))
        dense_checks.append(float((episode.continuation.rhs._log_weights[finite]-expected[finite]).abs().max()))
        if index in (1, 3):
            action = library[1] if index == 1 else library[2]
            request = NativeActuationRequest(executed_prefix(episode.continuation.rhs._public_control, index), action)
            branch = episode.plan_controls([action.regimen_uid])
            old_prior = episode.continuation._initial_readout_log_weights
            snapshots = episode.continuation._accepted_readout_snapshots
            episode.activate_selected_control(branch, request, plant.switch)
            assert episode.continuation._initial_readout_log_weights == old_prior
            assert episode.continuation._accepted_readout_snapshots == snapshots
    episode.model.partial_observation_window_loss, episode._adapt = raw_loss, raw_adapt
    episode.model.solver.update_filter_weights = raw_filter
    return original, episode, plant, reference, results, packets, updates, windows, dense_checks


@pytest.mark.parametrize("episode_fixture", ["adaptive_online_episode", "zero_mass_adaptive_online_episode"])
def test_adaptive_actual_episode_two_H3_events_weights_and_dense_cross_switch_gradients(request, episode_fixture, monkeypatch, tmp_path):
    original, episode, plant, reference, results, packets, updates, windows, errors = request.getfixturevalue(episode_fixture)
    assert episode.optimizer_updates == 20 and episode.observations_seen == plant._query == 6
    assert updates == list(range(1, 7)) and len(packets) == 6
    assert [row["adaptation"]["optimizer_updates_completed"] for row in results] == [0, 0, 0, 0, 10, 10]
    assert len(windows) == 20 and all(row is windows[0] for row in windows[:10])
    assert all(row is windows[10] for row in windows[10:])
    assert windows[0].history == windows[10].history[:-1]
    assert [snapshot.predictor_update_count for snapshot in episode.continuation._accepted_readout_snapshots] == [0, 0, 0, 0, 0, 10]
    assert episode._buffer.maxlen == 60 and len(episode._buffer) == 6
    assert all(row["filtering"]["adaptive_scientific_qualification"] == "UNQUALIFIED" for row in results)
    assert tuple(episode.model.solver.parameters()) == () and all(value.grad is None for value in episode.model.parameters())
    assert not torch.equal(original.output.bias, episode.model.output.bias)
    # Re-evaluate fifth H3 across B/A using the actual frozen accepted packet window.
    transition = list(episode._buffer)[-4:-1]
    # The fifth post-filter continuation is the sixth transition's exact start.
    post = list(episode._buffer)[-1].start
    window = windows[0]
    arrivals = tuple(row.arrived for row in transition)
    means, deltas = [], []
    raw_project, raw_residual = residual_module.project_cns_readout, episode.model.normalized_residual
    def project(*args, **kwargs):
        value = raw_project(*args, **kwargs); means.append(value); return value
    def residual(*args, **kwargs):
        value = raw_residual(*args, **kwargs); deltas.append(value); return value
    monkeypatch.setattr(residual_module, "project_cns_readout", project)
    monkeypatch.setattr(episode.model, "normalized_residual", residual)
    before = _online_live_snapshot(episode)
    loss, diagnostic = episode.model.partial_observation_window_loss(transition[0].start, post, arrivals,
        episode.normalization, adaptive_window=window)
    projection = _episode_field_projection(reference.fixed)
    base = projection @ projection.T
    all_rows = window.history + window.accepted
    blocks = [(row.coordinates, reference.fixed.time_hours[row.query_index], row.context.original_action.regimen_uid) for row in all_rows]
    covariance = torch.cat([torch.cat([base[list(a)][:, list(b)] *
        (min(t, s)/reference.fixed.reference_time_hours if action == other else 0.) for b,s,other in blocks],1)
        for a,t,action in blocks],0)
    covariance += torch.eye(len(covariance), dtype=torch.float64)*episode.readout.standard_deviation**2
    historical_mean = torch.cat([torch.tensor(row.mean, dtype=torch.float64) for row in window.history],1)
    trial_mean = torch.cat([value[0,:,list(row.coordinates)] for value,row in zip(means,window.accepted)],1)
    values = torch.cat([torch.tensor(row.values, dtype=torch.float64) for row in all_rows])
    joint = torch.distributions.MultivariateNormal(torch.cat((historical_mean, trial_mean),1), covariance_matrix=covariance).log_prob(values)
    size = historical_mean.shape[1]
    history = torch.distributions.MultivariateNormal(historical_mean, covariance_matrix=covariance[:size,:size]).log_prob(values[:size])
    initial = torch.tensor(window.initial_log_weights, dtype=torch.float64)
    dense_loss = -torch.logsumexp(initial+joint,0)+torch.logsumexp(initial+history,0)
    torch.testing.assert_close(loss,dense_loss,atol=2e-10,rtol=2e-10)
    actual_grad = torch.autograd.grad(loss, (*means, deltas[0]), retain_graph=True)
    dense_grad = torch.autograd.grad(dense_loss, (*means, deltas[0]))
    for actual, expected in zip(actual_grad,dense_grad):
        torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-10)
        assert actual.count_nonzero() and torch.isfinite(actual).all()
    zero_mass = torch.isneginf(initial)
    if zero_mass.any():
        assert zero_mass.nonzero().flatten().tolist() == [0, 7, 31]
        assert all(torch.count_nonzero(gradient[:, zero_mass]) == 0 for gradient in actual_grad[:-1])
        assert torch.equal(torch.isneginf(episode.continuation.rhs._log_weights[0]), zero_mass)
    from Libs.utils.model.native_conditional_readout import (
        adaptive_history_log_likelihood, reload_adaptive_readout_history,
    )
    context = {"group_uid": episode._execution_inputs.sample_uids[0],
        "family_uid": reference.fixed.readout.family_uid, "binding": json.loads(reference.fixed.predictor_binding),
        "reference_id": "artificial_adaptive_H3_history", "pre_execution_policy_snapshot": {
            "native_residual_control": {}, "native_temporal_query": {},
            "native_arrived_observation_filter": {}, "native_mechanistic_residual_online": episode.policy}}
    rng = plant._generator.get_state().clone()
    payload = json.loads(json.dumps(episode.actual_adaptive_readout_history(
        context, episode_status="component_complete"), allow_nan=False))
    restored = reload_adaptive_readout_history(payload, expected_reference=reference, expected_report_context=context)
    assert restored.initial_log_weights == episode._adaptive_initial_log_weights
    assert restored.history == episode._adaptive_history()
    assert payload["initial_log_weights"].count("-inf") == int(zero_mass.sum())
    score = adaptive_history_log_likelihood(reference, restored.history, restored.initial_log_weights)
    dense_score = torch.logsumexp(initial + _episode_dense_log_density(reference, restored.history), 0)
    assert torch.isfinite(score)
    torch.testing.assert_close(score, dense_score, atol=2e-10, rtol=2e-10)
    assert torch.equal(plant._generator.get_state(), rng)
    _assert_online_live_snapshot(episode,before)
    evidence = {"scope":"prospective_actual_adaptive_episode_software", "scientific_qualification":"UNQUALIFIED",
        "particle_count":32,"acquisitions":6,"switch_queries":[1,3],"optimizer_updates":20,
        "updates_per_H3":10,"buffer_transitions":60,"source_independent_fit":False,"scale_source":"declared_synthetic",
        "encoder_widths":[4,8,12],"bottleneck_width":16,"posterior_dense_max_errors":errors,
        "dense_H3_loss":float(dense_loss.detach()),"actual_H3_loss":float(loss.detach()),
        "all_trial_and_cross_switch_gradients_match_dense":True,"history_window_frozen_once_per_acquisition":True,
        "source_times":list(plant._source_times),"source_time_unit":reference.fixed.source_time_unit,
        "model_rhs_tuple_action_work":episode.continuation.rhs._work_ledger.spent,
        "source_rhs_calls":plant._work["nfev"],"filter_admissions":updates,"full_scientific_fit":False,
        "adaptive_calibration":False,"full_joint_policy":"CLOSED"}
    destination = Path(os.environ.get("NRNSIFT_NATIVE_ADAPTIVE_EPISODE_EVIDENCE",str(tmp_path/"adaptive_episode.json")))
    destination.write_text(json.dumps(evidence,indent=2)+"\n")


@pytest.mark.parametrize("fault", ["missing_history", "packet", "mean", "update_count", "weights", "library", "prior", "reference"])
def test_adaptive_episode_rejects_changed_history_before_advance(adaptive_online_episode, monkeypatch, fault):
    _, original, _, _, _, _, _, _, _ = adaptive_online_episode
    episode = copy.deepcopy(original)
    if fault == "missing_history": episode.continuation._accepted_readout_snapshots = episode.continuation._accepted_readout_snapshots[:-1]
    elif fault in {"packet", "mean", "update_count"}:
        snapshot = episode.continuation._accepted_readout_snapshots[0]
        if fault == "packet": snapshot = replace(snapshot,arrived=replace(snapshot.arrived,observation=replace(snapshot.arrived.observation,values=snapshot.arrived.observation.values+1)))
        if fault == "mean": snapshot = replace(snapshot,mean=tuple(tuple(value+1 for value in row) for row in snapshot.mean))
        if fault == "update_count": snapshot = replace(snapshot,predictor_update_count=10)
        episode.continuation._accepted_readout_snapshots = (snapshot,*episode.continuation._accepted_readout_snapshots[1:])
    elif fault == "weights": episode.continuation.rhs._log_weights = torch.full((1,32),-__import__('math').log(32),dtype=torch.float64)
    elif fault == "library": episode.continuation.rhs._execution_library = tuple(reversed(episode.continuation.rhs._execution_library))
    elif fault == "prior": episode.continuation._initial_readout_log_weights = (0.,)*32
    else: episode._adaptive_reference = replace(episode._adaptive_reference,fixed=replace(episode._adaptive_reference.fixed,support_uid="foreign"))
    before = _online_live_snapshot(episode)
    def forbidden(*args,**kwargs): raise AssertionError("Invalid adaptive history reached model or provider")
    monkeypatch.setattr(episode.model,"advance_continuation",forbidden)
    # The actual pure history boundary runs before each physical advance.
    with pytest.raises(ValueError):
        episode._adaptive_history()
    _assert_online_live_snapshot(episode,before)


@pytest.fixture
def adaptive_instrument_episode(execution_cases):
    from Libs.utils.model.native_arrived_observation import NativeArrivedObservation
    from Libs.utils.model.native_conditional_readout import filter_adaptive_native_observation
    from Libs.utils.data.joint_observation import RetainedObservation
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference,FixedReadoutReference
    original,accepted,plant,library = _actual_execution_case(execution_cases,FAMILIES[0],initial_control_index=2)
    state = accepted.continuation.rhs
    context = accepted.model.solver.filter_context(state)
    binding = {"normalization_id":accepted.normalization.identity,"checkpoint_sha256":"independent_instrument_software"}
    fixed = FixedReadoutReference(accepted.readout,context["support_uid"],context["origin_posterior_uid"],
        tuple(context["sample_uids"]),tuple(tuple(row) for row in context["particle_uids"]),library[2].regimen_uid,
        json.dumps(binding,sort_keys=True),accepted.normalization.identity,"synthetic_instrument_scale",(.01,)*26,
        tuple(accepted.normalization.scale.tolist()),max(state._times),tuple(library[2].source_times),tuple(state._times),
        library[2].source_time_unit,0)
    reference = AdaptiveReadoutReference(fixed,library,library[2].regimen_uid)
    episode = NativeResidualOnlineEpisode(original, accepted._execution_inputs, accepted.normalization, accepted.readout,
        executed_regimen_uid=reference.initial_action_uid, adaptive_reference=reference)
    episode.bind_execution(reference.original_actions, reference.original_actions[2], plant.observation_identity)
    requests = []
    def provide(acquisition):
        requests.append(acquisition)
        mask = torch.tensor(native_acquisition_mask(acquisition.readout, acquisition.selected_coordinate))[None]
        # A source-independent instrument stub returns values and public metadata
        # only. It has no solver/source object, full-state target or tuple draw.
        values = torch.where(mask, torch.full((1,6), .2, dtype=torch.float64), 0.)
        spec = acquisition.readout
        observation = RetainedObservation(channel_uid=spec.channel_uid, assay_uid=spec.assay_uid,
            observation_uid=f"independent_{acquisition.query_index}",sequence_number=acquisition.filtering_revision+1,
            source_uid=spec.source_uid,source_snapshot_ref=spec.source_snapshot_ref,sample_uids=acquisition.sample_uids,
            role=spec.role,modality="state",law="gaussian",units=spec.units,
            normalization_ref=spec.normalization_ref,values=values,observed=mask,
            left_censored=torch.zeros_like(mask),encoder_indices=tuple(range(6)))
        return NativeArrivedObservation(acquisition,observation)
    return episode,provide,requests,reference


@pytest.mark.parametrize("fault", ["nan", "positive_inf", "all_impossible", "unnormalized"])
def test_adaptive_filter_rejects_malformed_initial_prior_before_admission(adaptive_instrument_episode, monkeypatch, fault):
    from Libs.utils.model.native_conditional_readout import filter_adaptive_native_observation
    episode, provide, _, reference = adaptive_instrument_episode
    episode.step(provide)
    state = episode.continuation.rhs
    prediction = predict_native_readout(episode.model.solver, state, episode.readout,
        executed_regimen_uid=episode.executed_regimen_uid)
    packet = episode._buffer[-1].arrived
    packet = replace(packet, acquisition=prediction.acquisition, observation=replace(packet.observation,
        observation_uid="unadmitted_invalid_prior_packet", sequence_number=state.filter_revision+1))
    weights = episode._adaptive_initial_log_weights
    if fault == "nan": weights = (float("nan"), *weights[1:])
    elif fault == "positive_inf": weights = (float("inf"), *weights[1:])
    elif fault == "all_impossible": weights = (-float("inf"),) * 32
    elif fault == "unnormalized": weights = tuple(value+1 for value in weights)
    before = _online_live_snapshot(episode)
    accepted = episode.continuation._accepted_readout_snapshots
    optimizer = copy.deepcopy(episode.optimizer.state_dict())
    parameters = {name: value.detach().clone() for name, value in episode.model.named_parameters()}
    def forbidden(*args, **kwargs): raise AssertionError("Malformed prior reached admission or Adam")
    monkeypatch.setattr(episode.model.solver, "update_filter_weights", forbidden)
    monkeypatch.setattr(episode.optimizer, "step", forbidden)
    with pytest.raises(ValueError, match="initial32-tuple prior"):
        filter_adaptive_native_observation(episode.model.solver, state, prediction, packet,
            reference=reference, history=episode._adaptive_history(), initial_log_weights=weights,
            context=episode._adaptive_history()[-1].context)
    _assert_online_live_snapshot(episode, before)
    assert episode.continuation._accepted_readout_snapshots is accepted
    assert episode.optimizer.state_dict() == optimizer and episode.optimizer_updates == 0
    assert all(torch.equal(value, parameters[name]) for name, value in episode.model.named_parameters())


def test_adaptive_source_independent_provider_selected_packets_and_no_double_admission(adaptive_instrument_episode):
    from Libs.utils.model.native_conditional_readout import filter_adaptive_native_observation
    episode,provide,requests,reference = adaptive_instrument_episode
    choice = next(i for i,keep in enumerate(episode.readout.observed) if keep)
    first = episode.step(provide,selected_coordinate=choice)
    assert len(requests) == 1 and first["filtering"]["filtering_revision"] == 1
    before = _online_live_snapshot(episode)
    history = episode._adaptive_history()
    packet = episode._buffer[-1].arrived
    with pytest.raises(ValueError,match="Duplicate"):
        filter_adaptive_native_observation(episode.model.solver,episode.continuation.rhs,first["prediction"],packet,
            reference=reference,history=history,initial_log_weights=episode._adaptive_initial_log_weights,context=history[-1].context)
    _assert_online_live_snapshot(episode,before)
    second = episode.step(provide,selected_coordinate=choice)
    assert len(requests) == 2 and second["filtering"]["filtering_revision"] == 2
    assert episode.observations_seen == 2 and episode.optimizer_updates == 0


def test_native_packet_validation_is_pure_and_rejects_forged_current_prediction(adaptive_online_episode):
    from Libs.utils.model.native_arrived_observation import validate_native_observation
    _, original, _, _, _, _, _, _, _ = adaptive_online_episode
    episode = copy.deepcopy(original)
    state = episode.continuation.rhs
    prediction = predict_native_readout(episode.model.solver,state,episode.readout,
        executed_regimen_uid=episode.executed_regimen_uid,selected_coordinate=episode._buffer[-1].arrived.acquisition.selected_coordinate)
    packet = episode._buffer[-1].arrived
    # A fresh revision packet tests validation without admitting it; its sequence
    # is current, while the public original query and fields stay unchanged.
    packet = replace(packet,acquisition=prediction.acquisition,observation=replace(packet.observation,
        observation_uid="unadmitted_validation_packet",sequence_number=state.filter_revision+1))
    before = _online_live_snapshot(episode)
    validate_native_observation(episode.model.solver,state,prediction,packet)
    _assert_online_live_snapshot(episode,before)
    with pytest.raises(ValueError,match="Stale predicted"):
        validate_native_observation(episode.model.solver,state,replace(prediction,mean=prediction.mean+.01),packet)
    _assert_online_live_snapshot(episode,before)


def test_adaptive_activation_rejects_changed_history_before_actuator(adaptive_instrument_episode):
    from Libs.utils.data.native_actuation import NativeActuationRequest,executed_prefix
    episode,provider,_,reference = adaptive_instrument_episode
    for _ in range(2):
        episode.step(provider)
    state = episode.continuation.rhs
    request = NativeActuationRequest(executed_prefix(state._public_control,state.query_index),reference.original_actions[1])
    branch = episode.plan_controls([reference.original_actions[1].regimen_uid])
    episode.continuation._accepted_readout_snapshots = episode.continuation._accepted_readout_snapshots[:-1]
    before = _online_live_snapshot(episode)
    calls = []
    def actuator(value):
        calls.append(value)
        raise AssertionError("Changed adaptive history reached actuator")
    with pytest.raises(ValueError,match="complete actual admitted"):
        episode.activate_selected_control(branch,request,actuator)
    assert not calls
    _assert_online_live_snapshot(episode,before)


@pytest.mark.parametrize("replacement",["equal_content","changed_mean"])
def test_adaptive_provider_cannot_replace_owned_history_before_admission(adaptive_instrument_episode,replacement):
    episode,provider,requests,_ = adaptive_instrument_episode
    episode.step(provider)
    before_revision = episode.continuation.rhs.filter_revision
    original = episode.continuation._accepted_readout_snapshots
    def corrupt(acquisition):
        packet = provider(acquisition)
        first = replace(original[0])
        if replacement == "changed_mean":
            first = replace(first,mean=tuple(tuple(value+1 for value in row) for row in first.mean))
        episode.continuation._accepted_readout_snapshots = (first,)
        return packet
    with pytest.raises(ValueError,match="complete actual admitted"):
        episode.step(corrupt)
    assert len(requests) == 2 and episode.status == "failed"
    assert episode.continuation.rhs.query_index == 2
    assert episode.continuation.rhs.filter_revision == before_revision == 1
    assert episode.observations_seen == 1 and episode.optimizer_updates == 0
    assert len(episode.continuation.rhs._filter_binding["native_admitted_readout_payload_sha256"]) == 1
    assert episode.failed_arrival is not None


def test_molecular_opt_in_actual_activation_preserves_prefix_and_matches_numeric_forecast(execution_cases):
    """Actual zero-residual execution on retained public tuples; no full MPC or fitting."""
    from Libs.utils.data.native_actuation import NativeActuationRequest, compose_actuation, executed_prefix
    from Libs.utils.model.native_future_controls import compile_future_controls

    original, episode, plant, library = _actual_execution_case(
        execution_cases, FAMILIES[2], molecular_administration=True,
        model_seed=41, measurement_seed=41)
    inputs, norm = episode._execution_inputs, episode.normalization
    assert inputs.group_shape == (1, 32, 3)
    assert original.output.weight.count_nonzero() == original.output.bias.count_nonzero() == 0
    assert all(action.administration is not None for action in library)
    assert library[0].administered_mass(1) > 0
    parameters = {name: value.detach().clone() for name, value in original.named_parameters()}
    episode.step(plant.acquire)
    previous = episode.continuation
    state = previous.rhs
    request = NativeActuationRequest(plant.prefix, library[2])
    public = compose_actuation(request, library)
    assert request.prefix == executed_prefix(library[0], 1)
    assert request.prefix.administered_mass_mg_per_kg == library[0].administered_mass(1)
    assert public.interval_values[:1] == request.prefix.interval_values
    assert public.interval_values[1:] == library[2].interval_values[1:]
    assert public.interval_values != library[0].interval_values
    assert public.administered_mass() == pytest.approx(
        request.prefix.administered_mass_mg_per_kg +
        library[2].administered_mass() - library[2].administered_mass(1))
    assert public.source_times == library[0].source_times == tuple(inputs.grid["time_hours"])
    assert not public.dose_events

    compiled = compile_future_controls(state, inputs, episode.model.config,
        executed_regimen_uid=state._execution_model_uid,
        candidate_regimen_uids=[library[2].regimen_uid], require_delivered_dose=True)
    branch = episode.model.solver.branch_future_controls(state, inputs, norm,
        executed_regimen_uid=state._execution_model_uid,
        candidate_regimen_uids=[library[2].regimen_uid], require_delivered_dose=True)
    assert torch.equal(branch._guide.stimulus, compiled["stimulus"])
    # Preserve the actual inverse-encoded per-tuple driver, including roundoff.
    expected_drive = state._guide.stimulus.reshape(1, 32, 3, -1)[0, :, 0].clone()
    first = compiled["first_future_control_index"]
    assert first == state.query_index + 1 == 2
    expected_drive[:, first:] = state._registered_stimulus.reshape(1, 32, 3, -1)[0, :, 2, first:]
    assert torch.equal(branch._guide.stimulus, expected_drive)
    public_drive = expected_drive.new_tensor([0., *public.interval_values])[None].expand(32, -1)
    maximum_drive_discrepancy = float((expected_drive - public_drive).abs().max())
    print(json.dumps({"public_drive_maximum_absolute_discrepancy": maximum_drive_discrepancy}))
    # The original frozen compiler accepts inverse-input roundoff at 1e-5.
    # Exact recompiled concentration authentication remains in the real consumer.
    assert maximum_drive_discrepancy <= 1e-5
    assert len(compiled["metadata"]["intervention_budgets"]) == 32
    for budget in compiled["metadata"]["intervention_budgets"]:
        assert budget["delivered_dose_bound"] and budget["unit"] == "mg/kg"
        assert budget["prefix"] == request.prefix.administered_mass_mg_per_kg
        assert budget["total"] == public.administered_mass()
        assert budget["future"] == public.administered_mass() - budget["prefix"]
    before = _online_live_snapshot(episode)
    forecast = episode.model.prepare_forecast_control(previous, inputs, norm, branch, request)
    assert forecast.rhs.query_index == state.query_index == 1
    assert forecast.rhs.current_time_hours == state.current_time_hours
    assert forecast._forecast_control == public
    forecast_end, expected = episode.model.advance_forecast(forecast, norm, query_count=3)
    assert expected.shape == (32, 1, 6, 6, 26) and torch.isfinite(expected).all()
    _assert_online_live_snapshot(episode, before)

    calls = []
    plant_prefix, plant_work = plant.prefix, plant._work.copy()
    plant_rng = plant._generator.get_state().clone()
    program, static = previous.program.clone(), previous.static.clone()
    def actuator(value):
        calls.append(value)
        return plant.switch(value)
    invalid = replace(request, prefix=replace(request.prefix,
        administered_mass_mg_per_kg=request.prefix.administered_mass_mg_per_kg + .125))
    with pytest.raises(ValueError):
        episode.activate_selected_control(branch, invalid, actuator)
    saved_law = episode.model.config["native_molecular_administration"]["reference_time_hours"]
    episode.model.config["native_molecular_administration"]["reference_time_hours"] = 2.
    try:
        with pytest.raises(ValueError):
            episode.activate_selected_control(branch, request, actuator)
    finally:
        episode.model.config["native_molecular_administration"]["reference_time_hours"] = saved_law
    _assert_online_live_snapshot(episode, before)
    assert episode.continuation is previous and episode.status == "active"
    assert torch.equal(previous.program, program) and torch.equal(previous.static, static)
    assert not calls and plant.prefix == plant_prefix and plant._work == plant_work
    assert torch.equal(plant._generator.get_state(), plant_rng)

    carry = forecast.rhs.cns_state.clone()
    auxiliary = None if forecast.rhs.auxiliary_state is None else forecast.rhs.auxiliary_state.clone()
    history = fixed_history(forecast.rhs)
    result = episode.activate_selected_control(branch, request, actuator)
    active = episode.continuation
    assert len(calls) == 1 and calls[0] == request
    assert result["scope"] == "actual_control_execution_component_not_full_MPC"
    assert active.rhs._public_control == public and plant.prefix == executed_prefix(public, 1)
    assert torch.equal(active.rhs.cns_state, carry)
    if auxiliary is not None:
        assert torch.equal(active.rhs.auxiliary_state, auxiliary)
    equal_history(active.rhs, history)
    assert active.rhs._work_ledger is state._work_ledger
    assert active.rhs._guide.j == 32
    assert active.rhs._filter_binding["particle_uids"] == inputs.particle_uids
    assert torch.equal(active.rhs._weights, state._weights)
    assert torch.equal(active.rhs._log_weights, state._log_weights)
    assert torch.equal(active.program, forecast.program) and torch.equal(active.static, forecast.static)
    assert torch.equal(active.rhs._guide.stimulus, expected_drive)
    assert active.rhs._times == state._times and torch.equal(active.rhs._guide.clock, state._guide.clock)
    assert plant._work == plant_work and torch.equal(plant._generator.get_state(), plant_rng)
    with torch.no_grad():
        actual_end, actual = episode.model.advance_continuation(active, norm, query_count=3)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_end.rhs.cns_state, forecast_end.rhs.cns_state)
    assert actual_end.rhs.query_index == forecast_end.rhs.query_index == 2
    assert actual_end.rhs.current_time_hours == forecast_end.rhs.current_time_hours == public.source_times[2]
    assert episode.optimizer_updates == 0 and episode.observations_seen == 1
    assert all(torch.equal(value, parameters[name]) for name, value in original.named_parameters())
    print(json.dumps({"actual_activation_calls": len(calls), "successor_forecast_bitwise_equal": True,
        "model_and_measurement_seed": 41, "scientific_twin": "NOT_RUN"}))
