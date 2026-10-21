"""Prospective adaptive Gaussian algebra only; no source, calibration or live wiring."""
import copy
from dataclasses import replace
import inspect
import json

import pytest
import torch

from Libs.tests.test_native_conditional_readout import (
    dense_history, explicit_field_projection, predictor_mean, query, record, reference,
)
from Libs.utils.data.native_actuation import (
    NativeActuation, NativeActuationRequest, compose_actuation, executed_prefix,
)
from Libs.utils.model.native_arrived_observation import NativeArrivedObservation
from Libs.utils.model.native_conditional_readout import (
    ADAPTIVE_READOUT_LAW, AdaptiveReadoutContext, AdaptiveReadoutQuery, AdaptiveReadoutRecord,
    AdaptiveReadoutReference, adaptive_readout_window, adaptive_history_log_likelihood,
    conditional_readout, projected_brownian_covariance,
    export_adaptive_readout_history, reload_adaptive_readout_history,
)
from Libs.utils.model.native_observable_source import ELECTRICAL
from Libs.utils.data.native_observable_profile import family_grid, original_query_time_hours


def adaptive_reference(*, duplicate=False, mask=(True,) * 6):
    fixed = reference(duplicate=duplicate, mask=mask)
    fixed = replace(fixed, readout=replace(fixed.readout, family_uid=ELECTRICAL),
                    source_time_unit="ms", time_hours=original_query_time_hours(fixed.source_times, "ms"),
                    reference_time_hours=fixed.reference_time_hours / 3600000)
    actions = tuple(NativeActuation(fixed.readout.source_uid, fixed.readout.source_snapshot_ref,
        ELECTRICAL, uid, "ms", fixed.source_times, "stimulation", "uA", (value,) * 3)
        for uid, value in (("fixed_action", 1.), ("original_B", .2)))
    return AdaptiveReadoutReference(fixed, actions, "fixed_action")


def contexts(ref):
    first, second = ref.original_actions
    a = AdaptiveReadoutContext(first, first, (), ref.fixed.predictor_binding)
    switch_b = NativeActuationRequest(executed_prefix(first, 1), second)
    composite_b = compose_actuation(switch_b, ref.original_actions)
    b = AdaptiveReadoutContext(second, composite_b, (switch_b,), ref.fixed.predictor_binding)
    switch_a = NativeActuationRequest(executed_prefix(composite_b, 2), first)
    composite_a = compose_actuation(switch_a, ref.original_actions)
    revisit = AdaptiveReadoutContext(first, composite_a, (switch_b, switch_a), ref.fixed.predictor_binding)
    return a, b, revisit


def packet(ref, context, index, revision=0, **kwargs):
    fixed = replace(ref.fixed, executed_regimen_uid=context.executed_program.regimen_uid,
                    predictor_binding=context.predictor_binding)
    return record(fixed, index, revision, **kwargs)


def capture(ref, context, index, revision=0, **kwargs):
    row = packet(ref, context, index, revision, **kwargs)
    return AdaptiveReadoutRecord.capture(ref, context, row.arrived, row.mean)


def future(ref, context, index, revision=0, *, selected=None, mean=None, values=None):
    observed = ref.fixed.readout.observed if selected is None else tuple(i == selected for i in range(6))
    return AdaptiveReadoutQuery(context, index, revision, observed,
                               predictor_mean(index) if mean is None else mean, values)


def query_for_row(row):
    values = torch.zeros(6, dtype=torch.float64)
    values[list(row.coordinates)] = torch.tensor(row.values, dtype=torch.float64)
    mean = torch.zeros(32, 6, dtype=torch.float64)
    mean[:, list(row.coordinates)] = torch.tensor(row.mean, dtype=torch.float64)
    return AdaptiveReadoutQuery(row.context, row.query_index, row.filtering_revision,
                               tuple(i in row.coordinates for i in range(6)), mean, values)


def dense_adaptive(ref, rows):
    projection = explicit_field_projection(ref.fixed)
    base = projection @ projection.T
    blocks = [(row.coordinates, ref.fixed.time_hours[row.query_index], row.context.original_action.regimen_uid)
              for row in rows]
    covariance = torch.cat([torch.cat([
        base[list(a)][:, list(b)] * (min(t, s) / ref.fixed.reference_time_hours if action == other else 0.)
        for b, s, other in blocks], 1) for a, t, action in blocks], 0)
    covariance += torch.eye(len(covariance), dtype=torch.float64) * ref.fixed.readout.standard_deviation ** 2
    means = torch.cat([torch.tensor(row.mean, dtype=torch.float64) for row in rows], 1)
    values = torch.cat([torch.tensor(row.values, dtype=torch.float64) for row in rows])
    return torch.distributions.MultivariateNormal(means, covariance_matrix=covariance).log_prob(values)


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("equal_clock", [False, True])
def test_exact_fixed_action_reduction(duplicate, equal_clock):
    ref = adaptive_reference(duplicate=duplicate)
    context = contexts(ref)[0]
    fixed_rows = (packet(ref, context, 1, selected=0), packet(ref, context, 1 if equal_clock else 2, 1))
    rows = tuple(AdaptiveReadoutRecord.capture(ref, context, row.arrived, row.mean) for row in fixed_rows)
    desired = future(ref, context, 3, 2, values=torch.ones(6, dtype=torch.float64))
    actual = adaptive_readout_window(ref, rows, (desired,))
    expected = conditional_readout(ref.fixed, fixed_rows, query(ref.fixed, values=desired.values))
    for field in ("mean", "covariance", "residual_covariance", "history_log_likelihood", "log_likelihood"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(actual.history_log_likelihood, dense_history(ref.fixed, fixed_rows))


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("equal_clock", [False, True])
def test_dense_joint_equals_sequential_innovations_across_action_revisits(duplicate, equal_clock):
    ref = adaptive_reference(duplicate=duplicate)
    a, b, revisit = contexts(ref)
    rows = (capture(ref, a, 1, selected=0), capture(ref, b, 1 if equal_clock else 2, 1),
            capture(ref, revisit, 2 if equal_clock else 3, 2, selected=2))
    innovations = torch.zeros(32, dtype=torch.float64)
    for offset, row in enumerate(rows):
        result = adaptive_readout_window(ref, rows[:offset], (query_for_row(row),))
        innovations += result.log_likelihood
        assert result.log_likelihood.shape == (32,)
    expected = dense_adaptive(ref, rows)
    torch.testing.assert_close(innovations, expected, atol=2e-10, rtol=2e-10)
    joint = adaptive_readout_window(ref, (), tuple(query_for_row(row) for row in rows))
    torch.testing.assert_close(joint.log_likelihood, expected, atol=2e-10, rtol=2e-10)
    final = adaptive_readout_window(ref, rows, (future(ref, revisit, 3, 3),))
    torch.testing.assert_close(final.history_log_likelihood, expected, atol=2e-10, rtol=2e-10)


def test_independent_actions_and_revisit_keep_absolute_clock_covariance():
    ref = adaptive_reference()
    a, b, revisit = contexts(ref)
    result = adaptive_readout_window(ref, (), (future(ref, a, 1, selected=0),
        future(ref, b, 2, 1, selected=0), future(ref, revisit, 3, 2, selected=0)))
    base = projected_brownian_covariance(ref.fixed)[0, 0]
    expected = base * torch.tensor([[1., 0., 1.], [0., 3., 0.], [1., 0., 5.]], dtype=torch.float64) / 5
    torch.testing.assert_close(result.residual_covariance, expected)
    torch.testing.assert_close(result.covariance, expected + torch.eye(3, dtype=torch.float64) * .09)
    history = (capture(ref, a, 1, selected=0),)
    independent = adaptive_readout_window(ref, history, (future(ref, b, 2, 1, selected=0),))
    torch.testing.assert_close(independent.mean, predictor_mean(2)[:, :1])
    revisited = adaptive_readout_window(ref, history, (future(ref, revisit, 3, 1, selected=0),))
    assert not torch.equal(revisited.mean, predictor_mean(3)[:, :1])


def test_equal_clock_duplicate_selectors_share_residual_and_independent_noise():
    ref = adaptive_reference(duplicate=True)
    a = contexts(ref)[0]
    result = adaptive_readout_window(ref, (), (future(ref, a, 2, selected=0),
        future(ref, a, 2, 1, selected=1), future(ref, a, 2, 2, selected=0)))
    base = projected_brownian_covariance(ref.fixed)
    coordinates = [0, 1, 0]
    expected = base[coordinates][:, coordinates] * 3 / 5
    torch.testing.assert_close(result.residual_covariance, expected)
    torch.testing.assert_close(result.covariance, expected + torch.eye(3, dtype=torch.float64) * .09)
    assert result.residual_covariance[0, 1] != 0


def test_accepted_mean_and_values_remain_immutable_after_h3_predictor_update():
    ref = adaptive_reference()
    a = contexts(ref)[0]
    mean = predictor_mean(1).requires_grad_()
    row = packet(ref, a, 1, mean=mean, values=torch.arange(6, dtype=torch.float64) / 3)
    accepted = AdaptiveReadoutRecord.capture(ref, a, row.arrived, mean)
    snapshot = (accepted.mean, accepted.values)
    before = adaptive_readout_window(ref, (accepted,), (future(ref, a, 3, 1),))
    with torch.no_grad():
        mean.add_(100.)
        row.arrived.observation.values.add_(200.)
    assert (accepted.mean, accepted.values) == snapshot
    assert all(type(values) is tuple for values in accepted.mean)
    after = adaptive_readout_window(ref, (accepted,), (future(ref, a, 3, 1),))
    torch.testing.assert_close(after.mean, before.mean)
    updated = replace(a, predictor_binding=json.dumps({"normalization_id":ref.fixed.normalization_id,
                                                      "checkpoint_sha256":"updated_H3_predictor"}))
    trial = predictor_mean(3).requires_grad_()
    result = adaptive_readout_window(ref, (accepted,), (future(ref, updated, 3, 1, mean=trial),))
    altered = adaptive_readout_window(ref, (accepted,), (future(ref, updated, 3, 1, mean=trial + 2),))
    torch.testing.assert_close(result.history_log_likelihood, before.history_log_likelihood)
    torch.testing.assert_close(altered.mean, result.mean + 2)
    torch.testing.assert_close(torch.autograd.grad(result.mean.sum(), trial)[0], torch.ones_like(trial))
    assert mean.grad is None


def test_joint_h3_window_preserves_every_trial_prediction_gradient_and_complete_tuple():
    ref = adaptive_reference()
    a, b, revisit = contexts(ref)
    history = (capture(ref, a, 1, selected=0),)
    means = tuple(predictor_mean(index).requires_grad_() for index in (1, 2, 3))
    window = (future(ref, b, 1, 1, mean=means[0], values=torch.ones(6, dtype=torch.float64)),
              future(ref, revisit, 2, 2, mean=means[1], values=torch.ones(6, dtype=torch.float64)),
              future(ref, revisit, 3, 3, mean=means[2], values=torch.ones(6, dtype=torch.float64)))
    result = adaptive_readout_window(ref, history, window)
    weights = torch.arange(1, 33, dtype=torch.float64); weights /= weights.sum()
    loss = -torch.logsumexp(weights.log() + result.log_likelihood, 0)
    gradients = torch.autograd.grad(loss, means)
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) > 0 for value in gradients)
    for position in range(3):
        plus, minus = list(window), list(window)
        delta = torch.zeros_like(means[position]); delta[7, 2] = 1e-5
        plus[position] = replace(window[position], mean=means[position].detach() + delta)
        minus[position] = replace(window[position], mean=means[position].detach() - delta)
        forward = adaptive_readout_window(ref, history, tuple(plus)).log_likelihood
        backward = adaptive_readout_window(ref, history, tuple(minus)).log_likelihood
        numerical = (-torch.logsumexp(weights.log() + forward, 0)
                     + torch.logsumexp(weights.log() + backward, 0)) / 2e-5
        torch.testing.assert_close(gradients[position][7, 2], numerical, atol=2e-7, rtol=2e-7)
    trial_rows = tuple(AdaptiveReadoutRecord.capture(ref, item.context,
        packet(ref, item.context, item.query_index, item.filtering_revision,
               mean=item.mean, values=item.values).arrived, item.mean) for item in window)
    torch.testing.assert_close(result.log_likelihood, dense_adaptive(ref, (*history, *trial_rows))
                               - dense_adaptive(ref, history), atol=2e-10, rtol=2e-10)
    assert "weights" not in inspect.signature(adaptive_readout_window).parameters


@pytest.mark.parametrize("kind", ["missing", "unknown", "relabel", "forged_program", "wrong_prefix",
                                  "rewritten_switches", "missing_predictor", "normalizer"])
def test_original_action_and_predictor_lineage_rejections(kind):
    ref = adaptive_reference()
    a, b, revisit = contexts(ref)
    history, desired = (), future(ref, revisit, 3)
    if kind == "missing": desired = replace(desired, context=None)
    elif kind == "unknown":
        desired = replace(desired, context=replace(revisit, original_action=replace(a.original_action, regimen_uid="unknown")))
    elif kind == "relabel": desired = replace(desired, context=replace(revisit, original_action=b.original_action))
    elif kind == "forged_program":
        desired = replace(desired, context=replace(revisit, executed_program=replace(revisit.executed_program, interval_values=(.1,) * 3)))
    elif kind == "wrong_prefix":
        request = revisit.switches[-1]
        wrong = replace(request, prefix=replace(request.prefix, interval_values=(.9, .9)))
        desired = replace(desired, context=replace(revisit, switches=(*revisit.switches[:-1], wrong)))
    elif kind == "rewritten_switches": history, desired = (capture(ref, b, 2),), future(ref, a, 3, 1)
    elif kind == "missing_predictor": desired = replace(desired, context=replace(revisit, predictor_binding="{}"))
    elif kind == "normalizer":
        desired = replace(desired, context=replace(revisit, predictor_binding=json.dumps({"normalization_id":"other", "checkpoint_sha256":"H3"})))
    with pytest.raises(ValueError): adaptive_readout_window(ref, history, (desired,))


@pytest.mark.parametrize("field,value", [("time_hours", (0., 1., 3., 5.)),
    ("source_times", (0., 1., 2., 5.)), ("source_time_unit", "hour"),
    ("particle_uids", (tuple(f"changed{i}" for i in range(32)),)),
    ("sample_uids", ("changed_sample",)), ("support_uid", "changed_support")])
def test_changed_complete_history_reference_and_clocks_rejected(field, value):
    ref = adaptive_reference(); a = contexts(ref)[0]
    row = capture(ref, a, 1)
    changed = replace(ref, fixed=replace(ref.fixed, **{field:value}))
    with pytest.raises(ValueError): adaptive_readout_window(changed, (row,), (future(changed, a, 3, 1),))


@pytest.mark.parametrize("field,value", [("executed_regimen_uid", "unknown_composite"),
    ("source_time", 1.5), ("time_hours", 1.5), ("particle_uids", (("changed",) * 32,)),
    ("sample_uids", ("other",)), ("support_uid", "other"), ("source_time_unit", "hour")])
def test_snapshot_rejects_changed_arrived_packet_bindings(field, value):
    ref = adaptive_reference(); a = contexts(ref)[0]
    row = packet(ref, a, 1)
    changed = NativeArrivedObservation(replace(row.arrived.acquisition, **{field:value}), row.arrived.observation)
    with pytest.raises(ValueError): AdaptiveReadoutRecord.capture(ref, a, changed, row.mean)


@pytest.mark.parametrize("kind", ["duplicate_uid", "stale_revision", "decreasing_clock", "query_before_past",
    "mutable_history", "mutable_snapshot", "empty_window", "masked_query", "mixed_values", "observed_nan",
    "future_nan", "t0_history", "wrong_tuple_shape"])
def test_invalid_histories_and_trial_windows_rejected(kind):
    ref = adaptive_reference(mask=(True, False, True, True, True, True)); a = contexts(ref)[0]
    history, desired = (capture(ref, a, 1, selected=0),), (future(ref, a, 3, 1),)
    if kind == "duplicate_uid":
        history += (replace(capture(ref, a, 2, 1, selected=0), observation_uid=history[0].observation_uid),)
        desired = (future(ref, a, 3, 2),)
    elif kind == "stale_revision": desired = (replace(desired[0], filtering_revision=2),)
    elif kind == "decreasing_clock":
        history, desired = (capture(ref, a, 2, selected=0), capture(ref, a, 1, 1, selected=0)), (future(ref, a, 3, 2),)
    elif kind == "query_before_past": history, desired = (capture(ref, a, 2, selected=0),), (future(ref, a, 1, 1),)
    elif kind == "mutable_history": history = list(history)
    elif kind == "mutable_snapshot": history = (replace(history[0], mean=torch.zeros(32, 1)),)
    elif kind == "empty_window": desired = ()
    elif kind == "masked_query": desired = (future(ref, a, 3, 1, selected=1),)
    elif kind == "mixed_values": desired += (future(ref, a, 3, 2, values=torch.zeros(6)),)
    elif kind == "observed_nan": history = (replace(history[0], values=(float("nan"),)),)
    elif kind == "future_nan": desired = (replace(desired[0], mean=torch.full((32, 6), float("nan"))),)
    elif kind == "t0_history": history = (replace(history[0], query_index=0),)
    elif kind == "wrong_tuple_shape": history = (replace(history[0], mean=history[0].mean[:-1]),)
    with pytest.raises(ValueError): adaptive_readout_window(ref, history, desired)


def test_masked_missing_values_excluded_and_t0_is_noise_only():
    ref = adaptive_reference(mask=(True, False, True, False, False, True)); a = contexts(ref)[0]
    values = torch.tensor([.1, float("nan"), .3, float("inf"), float("nan"), .8], dtype=torch.float64)
    result = adaptive_readout_window(ref, (), (future(ref, a, 0, values=values),))
    assert result.coordinates == ((0, 2, 5),)
    assert torch.count_nonzero(result.residual_covariance) == 0
    torch.testing.assert_close(result.covariance, torch.eye(3, dtype=torch.float64) * .09)
    expected = torch.distributions.Normal(predictor_mean(0)[:, [0, 2, 5]], .3).log_prob(values[[0, 2, 5]]).sum(-1)
    torch.testing.assert_close(result.log_likelihood, expected)


def test_fixed_action_fit_never_qualifies_prospective_adaptive_law():
    ref = adaptive_reference()
    assert ref.law == ADAPTIVE_READOUT_LAW and ref.scientific_qualification == "UNQUALIFIED"
    assert "not_adaptive_calibration" in ref.scale_authority
    ref.validate()
    for field, value in (("scientific_qualification", "FIT_COMPLETE"), ("law", "fixed_action"),
                         ("scale_authority", "fixed_action_calibration_transferred")):
        with pytest.raises(ValueError, match="UNQUALIFIED"): replace(ref, **{field:value}).validate()


@pytest.mark.parametrize("kind", ["missing_acquisition", "foreign_acquisition", "boolean_revision", "string_revision"])
def test_snapshot_rejects_malformed_acquisition_type_and_revision(kind):
    ref = adaptive_reference(); a = contexts(ref)[0]
    row = packet(ref, a, 1)
    acquisition = row.arrived.acquisition
    if kind == "missing_acquisition": acquisition = None
    elif kind == "foreign_acquisition": acquisition = object()
    elif kind == "boolean_revision": acquisition = replace(acquisition, filtering_revision=True)
    elif kind == "string_revision": acquisition = replace(acquisition, filtering_revision="0")
    arrived = NativeArrivedObservation(acquisition, row.arrived.observation)
    with pytest.raises(ValueError, match="native acquisition"):
        AdaptiveReadoutRecord.capture(ref, a, arrived, row.mean)



def test_registered_source_clock_conversion_keeps_existing_grid_values():
    # The10ms value exposes division-versus-multiplication rounding. These are
    # the original registered7-query hours; exact equality protects every caller.
    source = (0., 2., 4., 6., 8., 10., 12.)
    golden = (0., 5.555555555555555e-7, 1.111111111111111e-6, 1.6666666666666667e-6,
              2.222222222222222e-6, 2.7777777777777775e-6, 3.3333333333333333e-6)
    profile = {"families": {ELECTRICAL: {"time_values": source, "time_unit": "ms"}},
               "region_id": [], "compartment_id": [], "quantities": []}
    assert original_query_time_hours(source, "ms") == golden
    assert tuple(family_grid(profile, ELECTRICAL)["time_hours"]) == golden
    assert golden[5] != source[5] / 3600000
    assert original_query_time_hours((0., 672.), "hour") == (0., 672.)
    with pytest.raises(ValueError):
        original_query_time_hours(source, "second")

def _trial_history(*, duplicate=False, masked=False):
    ref = adaptive_reference(duplicate=duplicate, mask=(True, False, True, True, False, True)
                             if masked else (True,)*6)
    first, second, revisit = contexts(ref)
    rows = (capture(ref, first, 1, selected=0), capture(ref, second, 2, 1, selected=2),
            capture(ref, revisit, 3, 2))
    initial = torch.arange(1, 33, dtype=torch.float64)
    initial[0] = 0.
    initial /= initial.sum()
    return ref, rows, tuple(initial.log().tolist())


def _dense_trial_history(ref, rows, initial_log_weights, scales):
    # Independent full-field incidence and dense covariance. This reference does
    # not call projected_brownian_covariance, the window, or its Schur complement.
    shape = (len(ref.fixed.readout.region_ids), len(ref.fixed.readout.compartment_ids),
             len(ref.fixed.readout.species_ids))
    index = torch.arange(shape[0]*shape[1]*shape[2]).reshape(shape)
    selector = torch.zeros(6, index.numel(), dtype=torch.float64)
    for channel, coordinate in enumerate(ref.fixed.readout.selector):
        selector[channel, int(index[coordinate])] = (ref.fixed.inverse_normalizer_scales[coordinate[-1]]
            * scales[coordinate[-1]] / ref.fixed.readout.scales[channel])
    projection = torch.tensor(ref.fixed.readout.matrix, dtype=torch.float64) @ selector
    base = projection @ projection.T
    blocks = [(row.coordinates, ref.fixed.time_hours[row.query_index],
               row.context.original_action.regimen_uid) for row in rows]
    covariance = torch.cat([torch.cat([
        base[list(left)][:, list(right)] * (min(t, u)/ref.fixed.reference_time_hours if action == other else 0.)
        for right, u, other in blocks], 1) for left, t, action in blocks], 0)
    covariance += torch.eye(len(covariance), dtype=torch.float64)*ref.fixed.readout.standard_deviation**2
    raw_means = torch.cat([torch.tensor(row.mean, dtype=torch.float64) for row in rows], 1)
    values = torch.cat([torch.tensor(row.values, dtype=torch.float64) for row in rows])
    component = torch.distributions.MultivariateNormal(raw_means, covariance_matrix=covariance).log_prob(values)
    return torch.logsumexp(torch.tensor(initial_log_weights, dtype=torch.float64)+component, 0)


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_trial_scale_complete_history_matches_independent_dense_mixture(duplicate, masked):
    ref, rows, initial = _trial_history(duplicate=duplicate, masked=masked)
    scales = torch.linspace(.2, 1.4, 26, dtype=torch.float64, requires_grad=True)
    before = (ref, rows, initial)
    actual = adaptive_history_log_likelihood(ref, rows, initial, trial_scales=scales)
    expected = _dense_trial_history(ref, rows, initial, scales)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    assert (ref, rows, initial) == before
    assert rows[0].context.original_action == rows[2].context.original_action
    assert rows[1].context.original_action != rows[0].context.original_action
    assert initial[0] == -float("inf") and ref.scientific_qualification == "UNQUALIFIED"
    wrong = torch.tensor(rows[-1].mean, dtype=torch.float64)+3.
    changed = (*rows[:-1], replace(rows[-1], mean=tuple(tuple(row) for row in wrong.tolist())))
    assert not torch.isclose(actual, adaptive_history_log_likelihood(ref, changed, initial, trial_scales=scales))


def test_trial_scale_gradient_matches_dense_and_finite_difference_on_observed_subspace():
    ref, rows, initial = _trial_history()
    scales = torch.linspace(.2, 1.4, 26, dtype=torch.float64, requires_grad=True)
    score = adaptive_history_log_likelihood(ref, rows, initial, trial_scales=scales)
    gradient, = torch.autograd.grad(score, scales)
    dense_gradient, = torch.autograd.grad(_dense_trial_history(ref, rows, initial, scales), scales)
    torch.testing.assert_close(gradient, dense_gradient, atol=2e-10, rtol=2e-10)
    selected_species = {coordinate[-1] for coordinate in ref.fixed.readout.selector}
    for species in selected_species:
        step = torch.zeros_like(scales); step[species] = 1e-5
        positive = adaptive_history_log_likelihood(ref, rows, initial, trial_scales=scales.detach()+step)
        negative = adaptive_history_log_likelihood(ref, rows, initial, trial_scales=scales.detach()-step)
        torch.testing.assert_close(gradient[species], (positive-negative)/2e-5, atol=2e-7, rtol=2e-7)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient[list(selected_species)]) > 0
    assert torch.count_nonzero(gradient[[i for i in range(26) if i not in selected_species]]) == 0


def test_trial_scale_defaults_keep_covariance_window_and_full_history_unchanged():
    ref, rows, initial = _trial_history(duplicate=True)
    scales = torch.tensor(ref.fixed.residual_scales, dtype=torch.float64, requires_grad=True)
    fixed = projected_brownian_covariance(ref.fixed)
    trial = projected_brownian_covariance(ref.fixed, trial_scales=scales)
    torch.testing.assert_close(trial, fixed, atol=0, rtol=0)
    query = query_for_row(rows[-1])
    old = adaptive_readout_window(ref, rows[:-1], (query,))
    new = adaptive_readout_window(ref, rows[:-1], (query,), trial_scales=scales)
    for field in ("mean", "covariance", "residual_covariance", "history_log_likelihood", "log_likelihood"):
        torch.testing.assert_close(getattr(new, field), getattr(old, field), atol=0, rtol=0)
    old_score = adaptive_history_log_likelihood(ref, rows, initial)
    new_score = adaptive_history_log_likelihood(ref, rows, initial, trial_scales=scales)
    torch.testing.assert_close(new_score, old_score, atol=0, rtol=0)
    gradient, = torch.autograd.grad(new.mean.sum()+new.log_likelihood.sum(), scales)
    assert torch.isfinite(gradient).all()
    float_scales = scales.detach().float().requires_grad_()
    float_score = adaptive_history_log_likelihood(ref, rows, initial, trial_scales=float_scales)
    float_gradient, = torch.autograd.grad(float_score, float_scales)
    assert float_gradient.dtype == torch.float32 and torch.isfinite(float_gradient).all()


@pytest.mark.parametrize("fault", ["shape", "integer", "nan", "inf", "zero", "negative", "list", "device"])
def test_trial_scales_reject_unbound_values_before_density(fault):
    ref, rows, initial = _trial_history()
    scales = torch.ones(26, dtype=torch.float64)
    if fault == "shape": scales = scales[:-1]
    elif fault == "integer": scales = scales.long()
    elif fault == "nan": scales[0] = float("nan")
    elif fault == "inf": scales[0] = float("inf")
    elif fault == "zero": scales[0] = 0.
    elif fault == "negative": scales[0] = -.1
    elif fault == "list": scales = scales.tolist()
    elif fault == "device": scales = torch.empty(26, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="Trial discrepancy scales"):
        projected_brownian_covariance(ref.fixed, device="cpu", trial_scales=scales)
    with pytest.raises(ValueError, match="Trial discrepancy scales"):
        adaptive_readout_window(ref, rows[:-1], (query_for_row(rows[-1]),), trial_scales=scales)


@pytest.mark.parametrize("fault", ["empty", "list", "record_type", "duplicate_final_uid", "mutable_final_mean",
    "nan_final_value", "stale_final_revision", "wrong_final_reference", "masked_final_coordinate",
    "final_t0", "prefix_mutable_uid", "prefix_nan", "rewritten_action"])
def test_complete_history_likelihood_rejects_invalid_immutable_snapshots(fault):
    ref, rows, initial = _trial_history(masked=True)
    if fault == "empty": rows = ()
    elif fault == "list": rows = list(rows)
    elif fault == "record_type": rows = (*rows[:-1], object())
    elif fault == "duplicate_final_uid": rows = (*rows[:-1], replace(rows[-1], observation_uid=rows[0].observation_uid))
    elif fault == "mutable_final_mean": rows = (*rows[:-1], replace(rows[-1], mean=list(rows[-1].mean)))
    elif fault == "nan_final_value": rows = (*rows[:-1], replace(rows[-1], values=(float("nan"),)*len(rows[-1].values)))
    elif fault == "stale_final_revision": rows = (*rows[:-1], replace(rows[-1], filtering_revision=7))
    elif fault == "wrong_final_reference":
        changed = replace(ref, initial_action_uid="unknown")
        rows = (*rows[:-1], replace(rows[-1], reference=changed))
    elif fault == "masked_final_coordinate":
        rows = (*rows[:-1], replace(rows[-1], coordinates=(1,), mean=((0.,),)*32, values=(0.,)))
    elif fault == "final_t0": rows = (*rows[:-1], replace(rows[-1], query_index=0))
    elif fault == "prefix_mutable_uid": rows = (replace(rows[0], observation_uid=[]), *rows[1:])
    elif fault == "prefix_nan": rows = (replace(rows[0], values=(float("nan"),)), *rows[1:])
    elif fault == "rewritten_action":
        context = replace(rows[-1].context, switches=())
        rows = (*rows[:-1], replace(rows[-1], context=context))
    with pytest.raises(ValueError):
        adaptive_history_log_likelihood(ref, rows, initial)


@pytest.mark.parametrize("fault", ["shape", "list", "nan", "positive_inf", "all_impossible", "unnormalized", "string"])
def test_complete_history_likelihood_rejects_invalid_initial_prior(fault):
    ref, rows, initial = _trial_history()
    if fault == "shape": initial = initial[:-1]
    elif fault == "list": initial = list(initial)
    elif fault == "nan": initial = (float("nan"), *initial[1:])
    elif fault == "positive_inf": initial = (float("inf"), *initial[1:])
    elif fault == "all_impossible": initial = (-float("inf"),)*32
    elif fault == "unnormalized": initial = tuple(value+1 for value in initial)
    elif fault == "string": initial = ("-inf", *initial[1:])
    with pytest.raises(ValueError, match="initial32-tuple prior"):
        adaptive_history_log_likelihood(ref, rows, initial)


def test_complete_history_likelihood_uses_one_initial_tuple_across_all_records():
    ref, rows, initial = _trial_history()
    actual = adaptive_history_log_likelihood(ref, rows, initial)
    # Summing marginal mixtures redraws the latent tuple and is a different law.
    reset_scores = []
    for row in rows:
        one = replace(row, filtering_revision=0)
        reset_scores.append(_dense_trial_history(ref, (one,), initial,
                            torch.tensor(ref.fixed.residual_scales, dtype=torch.float64)))
    assert not torch.isclose(actual, sum(reset_scores))
    singleton = (rows[0],)
    torch.testing.assert_close(adaptive_history_log_likelihood(ref, singleton, initial),
        _dense_trial_history(ref, singleton, initial, torch.tensor(ref.fixed.residual_scales, dtype=torch.float64)))


def test_complete_history_prior_uses_existing_float32_input_double_normalization():
    from Libs.models.modules.world_state import _normalized_log_weights
    ref, rows, _ = _trial_history()
    logits = torch.linspace(-7., 3., 32, dtype=torch.float32)
    logits[0] = -float("inf")
    logs = _normalized_log_weights(logits)
    assert logs.dtype == torch.float64 and logs[0] == -float("inf")
    initial = tuple(logs.tolist())
    actual = adaptive_history_log_likelihood(ref, rows, initial)
    expected = _dense_trial_history(ref, rows, initial, torch.tensor(ref.fixed.residual_scales, dtype=torch.float64))
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


def test_complete_history_rejects_malformed_fixed_reference_before_snapshot_access():
    ref, rows, initial = _trial_history()
    with pytest.raises(ValueError, match="bound fixed starting reference"):
        adaptive_history_log_likelihood(replace(ref, fixed=object()), rows, initial)



def transport_context(ref):
    return {"group_uid": "artificial_transport_group", "family_uid": ref.fixed.readout.family_uid,
        "binding": json.loads(ref.fixed.predictor_binding), "reference_id": "artificial_public_reference",
        "pre_execution_policy_snapshot": {"native_residual_control": {}, "native_temporal_query": {"integration_seed": 7},
            "native_arrived_observation_filter": {},
            "native_mechanistic_residual_online": {"warmup_transitions": 9, "updates_per_observation": 1}}}


def transport_rows(ref):
    a, b, revisit = contexts(ref)
    contexts_with_revision = tuple(replace(context, predictor_binding=json.dumps(
        dict(json.loads(ref.fixed.predictor_binding), residual_optimizer_updates=0), sort_keys=True))
        for context in (a, b, revisit))
    return tuple(capture(ref, context, index, index-1, selected=coordinate)
                 for index, context, coordinate in zip((1, 2, 3), contexts_with_revision, (0, 2, 0)))


def artificial_transport(ref, rows, weights, *, status="component_complete"):
    payload = export_adaptive_readout_history(ref, rows, weights,
        report_context=transport_context(ref), episode_status=status)
    # Algebra fixture only. Reload cannot authenticate this label; production
    # obtains it solely from the actual episode accessor tested separately.
    payload["origin"] = "episode_authenticated_actual_acceptance"
    return json.loads(json.dumps(payload, allow_nan=False))


def test_actual_history_transport_json_preserves_raw_revisit_likelihood_and_zero_mass():
    ref = adaptive_reference(); rows = transport_rows(ref)
    logits = torch.linspace(-2., 1., 32, dtype=torch.float64); logits[4] = -torch.inf
    weights = tuple(logits.log_softmax(0).tolist())
    payload = artificial_transport(ref, rows, weights)
    assert payload["initial_log_weights"][4] == "-inf"
    restored = reload_adaptive_readout_history(payload, expected_reference=ref,
        expected_report_context=transport_context(ref))
    assert restored.history == rows and restored.initial_log_weights == weights
    assert restored.authentication == "structural_validation_against_expected_report_context"
    assert not restored.calibration_eligible and restored.scientific_readiness == "NOT_ADMITTED"
    assert payload["report_context"]["pre_execution_policy_snapshot"]["native_temporal_query"]["integration_seed"] == 7
    torch.testing.assert_close(adaptive_history_log_likelihood(ref, rows, weights),
        adaptive_history_log_likelihood(restored.reference, restored.history, restored.initial_log_weights), rtol=0, atol=0)
    raw_means = copy.deepcopy(payload["records"])
    payload["records"][0]["mean"][0][0] += 100
    assert restored.history == rows and raw_means[0]["mean"][0][0] == rows[0].mean[0][0]


@pytest.mark.parametrize("fault", ["imagined", "group", "policy", "reference", "revision", "duplicate", "clock",
    "coordinates", "mean_shape", "mean_nan", "values_inf", "weight_nan", "weight_inf", "weight_string", "weight_null",
    "weight_unnormalized", "action", "prefix", "predictor", "status", "likelihood", "extra_private", "private_context"])
def test_actual_history_transport_rejects_binding_lineage_and_malformed_data(fault):
    ref = adaptive_reference(); rows = transport_rows(ref)
    weights = tuple(torch.zeros(32, dtype=torch.float64).log_softmax(0).tolist())
    payload = artificial_transport(ref, rows, weights); expected = transport_context(ref)
    if fault == "imagined": payload["origin"] = "hypothetical_branch"
    elif fault == "group": payload["report_context"]["group_uid"] = "foreign"
    elif fault == "policy": payload["report_context"]["pre_execution_policy_snapshot"]["native_temporal_query"]["integration_seed"] = 8
    elif fault == "reference": payload["reference_binding"]["fixed"]["particle_uids"][0].reverse()
    elif fault == "revision": payload["records"][1]["filtering_revision"] = 0
    elif fault == "duplicate": payload["records"][1]["observation_uid"] = payload["records"][0]["observation_uid"]
    elif fault == "clock": payload["records"][1]["query_index"] = 1
    elif fault == "coordinates": payload["records"][0]["coordinates"] = [6]
    elif fault == "mean_shape": payload["records"][0]["mean"].pop()
    elif fault == "mean_nan": payload["records"][0]["mean"][0][0] = float("nan")
    elif fault == "values_inf": payload["records"][0]["values"][0] = float("inf")
    elif fault == "weight_nan": payload["initial_log_weights"][0] = float("nan")
    elif fault == "weight_inf": payload["initial_log_weights"][0] = -float("inf")
    elif fault == "weight_string": payload["initial_log_weights"][0] = "-1"
    elif fault == "weight_null": payload["initial_log_weights"][0] = None
    elif fault == "weight_unnormalized": payload["initial_log_weights"][0] += 1
    elif fault == "action": payload["records"][1]["original_action_uid"] = "unregistered"
    elif fault == "prefix": payload["records"][1]["switches"][0]["prefix"]["interval_values"][0] += 1
    elif fault == "predictor": payload["records"][1]["predictor_binding"] = json.dumps(dict(expected["binding"], residual_optimizer_updates=9))
    elif fault == "status": payload["status"] = "empty"
    elif fault == "likelihood": payload["likelihood_data_available"] = False
    elif fault == "extra_private": payload["private_fields"] = [1]
    else: expected["pre_execution_policy_snapshot"]["native_temporal_query"]["measurement_seed"] = 99
    with pytest.raises(ValueError):
        reload_adaptive_readout_history(payload, expected_reference=ref, expected_report_context=expected)


@pytest.mark.parametrize("status", ["ready", "active", "failed", "component_complete"])
def test_actual_history_transport_empty_or_failed_prefix_is_unadmitted(status):
    ref = adaptive_reference(); weights = tuple(torch.zeros(32, dtype=torch.float64).log_softmax(0).tolist())
    rows = () if status in {"ready", "component_complete"} else transport_rows(ref)[:1]
    payload = artificial_transport(ref, rows, weights, status=status)
    restored = reload_adaptive_readout_history(payload, expected_reference=ref, expected_report_context=transport_context(ref))
    assert restored.history == rows and not restored.calibration_eligible
    assert restored.status == ("failed" if status == "failed" else "empty" if not rows else "partial")
    if not rows:
        assert payload["likelihood_data_available"] is False
        with pytest.raises(ValueError, match="nonempty"):
            adaptive_history_log_likelihood(ref, restored.history, restored.initial_log_weights)


def test_supplied_algebra_records_cannot_establish_actual_transport_origin():
    ref = adaptive_reference(); rows = transport_rows(ref)
    weights = tuple(torch.zeros(32, dtype=torch.float64).log_softmax(0).tolist())
    payload = export_adaptive_readout_history(ref, rows, weights, report_context=transport_context(ref), episode_status="active")
    assert payload["origin"] == "supplied_records_without_episode_authentication"
    with pytest.raises(ValueError, match="origin"):
        reload_adaptive_readout_history(payload, expected_reference=ref, expected_report_context=transport_context(ref))


# Numeric fixtures isolate known scale recovery from biological qualification.
def _calibration_numeric_episodes(*, coordinate=0, missing=False, collapsed=False):
    from Libs.utils.model.native_conditional_readout import AdaptiveCalibrationEpisode
    ref = adaptive_reference(mask=(True,)*6 if missing else tuple(i == coordinate for i in range(6)))
    readout = replace(ref.fixed.readout, matrix=tuple(tuple(float(i == j) for j in range(6)) for i in range(6)),
                      scales=(1.,)*6, standard_deviation=.1)
    ref = replace(ref, fixed=replace(ref.fixed, readout=readout, inverse_normalizer_scales=(1.,)*26))
    a = contexts(ref)[0]
    # Balanced observations make the known Gaussian variance exactly attainable.
    amplitude = (1.7**2 * ref.fixed.time_hours[1]/ref.fixed.reference_time_hours + .1**2)**.5
    initial = (-__import__("math").log(32),)*32
    episodes = []
    for index, sign in enumerate((-1., 1., -1., 1.)):
        values = torch.zeros(6, dtype=torch.float64)
        values[coordinate] = 0. if collapsed else sign*amplitude
        row = capture(ref, a, 1, selected=coordinate, uid=f"numeric_{index}",
                      mean=torch.zeros((32,6), dtype=torch.float64), values=values)
        episodes.append(AdaptiveCalibrationEpisode(f"numeric_group_{index}", ref, (row,), initial))
    return tuple(episodes)


def _calibration_numerical():
    from Libs.utils.data.raw_data_registry import load_config
    return load_config("Libs/configs/exp/manuscript_native_protocol.yaml")["native_controlled_comparators"]["residual_calibration"]["numerical"]


def test_adaptive_fit_recovers_known_scale_and_preserves_unobserved_full_field():
    from Libs.utils.model.native_conditional_readout import fit_adaptive_readout_scales
    episodes = _calibration_numeric_episodes()
    before = copy.deepcopy(episodes)
    fitted = fit_adaptive_readout_scales(episodes, tuple(row.group_uid for row in episodes), _calibration_numerical())
    assert fitted["status"] == "FIT_COMPLETE", fitted
    assert fitted["fitted_species_indices"] == [0]
    assert fitted["scales"][0] == pytest.approx(1.7, rel=1e-5)
    assert fitted["scales"][1:] == list(episodes[0].reference.fixed.residual_scales[1:])
    assert fitted["diagnostics"]["hessian_rank"] == fitted["diagnostics"]["fitted_dimension"] == 1
    assert fitted["diagnostics"]["gradient_max_abs"] <= _calibration_numerical()["gradient_tolerance"]
    assert fitted["whole_26_field_adaptive_qualification"] == "UNQUALIFIED"
    assert episodes == before


@pytest.mark.parametrize("kind", ["missing_coordinate_information", "collapsed", "below_sensor_noise", "repeated_group", "incomplete_group"])
def test_adaptive_fit_rejects_insufficient_or_collapsed_information(kind):
    from Libs.utils.model.native_conditional_readout import fit_adaptive_readout_scales
    episodes = _calibration_numeric_episodes(missing=kind == "missing_coordinate_information", collapsed=kind == "collapsed")
    if kind == "below_sensor_noise":
        episodes = tuple(replace(row,history=tuple(replace(record,values=(.05,)) for record in row.history)) for row in episodes)
    groups = tuple(row.group_uid for row in episodes)
    if kind == "repeated_group": episodes = episodes+(episodes[0],)
    if kind == "incomplete_group": groups += ("missing_group",)
    fitted = fit_adaptive_readout_scales(episodes, groups, _calibration_numerical())
    assert fitted["status"] == "FIT_FAILED", fitted
    assert fitted["scales"] is None and not fitted["observable_fit_valid"]


def test_adaptive_fit_uses_complete_history_mixture_without_coordinate_divisor():
    from Libs.utils.model.native_conditional_readout import AdaptiveCalibrationEpisode, adaptive_calibration_objective, adaptive_observable_species
    ref, rows, weights = _trial_history()
    episodes = (AdaptiveCalibrationEpisode("whole_history", ref, rows, weights),)
    indices = adaptive_observable_species(episodes)
    eta = torch.tensor([ref.fixed.residual_scales[i] for i in indices], dtype=torch.float64).log().requires_grad_()
    loss = adaptive_calibration_objective(eta, episodes, indices)
    torch.testing.assert_close(loss, -adaptive_history_log_likelihood(ref, rows, weights))
    gradient, = torch.autograd.grad(loss, eta)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
