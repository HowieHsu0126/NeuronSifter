"""Synthetic fixed-action Gaussian algebra; no source or learned-model evaluation."""
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

from Libs.utils.data.joint_observation import RetainedObservation
from Libs.utils.model.native_arrived_observation import (
    NativeAcquisition, NativeArrivedObservation, NativeCNSReadout,
)
from Libs.utils.model.native_conditional_readout import (
    FixedReadoutQuery, FixedReadoutRecord, FixedReadoutReference,
    conditional_readout, projected_brownian_covariance,
)
from Libs.utils.model.native_transolver_residual import CONTROLLED_LAW


def reference(*, duplicate=False, mask=(True,) * 6, noise=.3):
    selector = ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3), (0, 0, 4), (1, 1, 5))
    if duplicate:
        selector = (selector[0], selector[0], *selector[2:])
    matrix = tuple(tuple(float(i == j) + (.2 if j == (i + 1) % 6 else 0.)
                         for j in range(6)) for i in range(6))
    readout = NativeCNSReadout(
        "synthetic_channel", "synthetic_assay", "synthetic_source", "synthetic_snapshot",
        "validation", "synthetic_family", "synthetic_fixed_source_scales",
        "source_reference_standardized", ("r0", "r1"), ("c0", "c1"),
        tuple(f"s{i}" for i in range(26)), selector, (1., 2., 3., 4., 5., 6.),
        matrix, mask, noise)
    return FixedReadoutReference(
        readout, "synthetic_support", "synthetic_origin", ("synthetic_sample",),
        (tuple(f"tuple{i}" for i in range(32)),), "fixed_action",
        json.dumps({"normalization_id": "synthetic_train_scale", "checkpoint_sha256": "synthetic_fixed_predictor"},
                   sort_keys=True), "synthetic_train_scale", "synthetic_residual",
        tuple(.5 + i / 30 for i in range(26)), tuple(2. + i / 10 for i in range(26)),
        5., (0., 1., 3., 5.), (0., 1., 3., 5.), "hour", 0)


def predictor_mean(index):
    return torch.arange(192, dtype=torch.float64).reshape(32, 6) / 100 + index / 10


def record(ref, index, revision=0, *, selected=None, uid=None, mean=None, values=None):
    acquisition = NativeAcquisition(
        ref.readout, ref.support_uid, ref.origin_posterior_uid, ref.sample_uids, ref.particle_uids,
        revision, index, ref.source_times[index], ref.source_time_unit, ref.time_hours[index],
        ref.executed_regimen_uid, selected_coordinate=selected)
    mask = torch.tensor(ref.readout.observed if selected is None else
                        tuple(i == selected for i in range(6))).unsqueeze(0)
    if values is None:
        values = torch.arange(6, dtype=torch.float64) / 7 + index / 5
        if selected is not None:
            values = values.masked_fill(~mask[0], 0.)
    observation = RetainedObservation(
        uid or f"acquisition_{revision}", ref.readout.source_uid, ref.readout.source_snapshot_ref,
        ref.sample_uids, ref.readout.role, ref.readout.channel_uid, "gaussian", ref.readout.units,
        values.unsqueeze(0), mask, torch.zeros_like(mask), tuple(range(6)), revision + 1,
        normalization_ref=ref.readout.normalization_ref, assay_uid=ref.readout.assay_uid)
    return FixedReadoutRecord(ref, NativeArrivedObservation(acquisition, observation),
                              predictor_mean(index) if mean is None else mean)


def query(ref, index=3, *, mask=None, mean=None, values=None):
    return FixedReadoutQuery(ref, index, ref.readout.observed if mask is None else mask,
                            predictor_mean(index) if mean is None else mean, values)


def explicit_field_projection(ref):
    # Construct the full independent physical grid, rather than the utility's
    # reduced duplicate-selector representation.
    projection = torch.zeros(6, 2 * 2 * 26, dtype=torch.float64)
    for output in range(6):
        for selected, (region, compartment, species) in enumerate(ref.readout.selector):
            column = (region * 2 + compartment) * 26 + species
            projection[output, column] += (ref.readout.matrix[output][selected]
                * ref.inverse_normalizer_scales[species] * ref.residual_scales[species]
                / ref.readout.scales[selected])
    return projection


def dense_history(ref, rows):
    base = explicit_field_projection(ref) @ explicit_field_projection(ref).T
    coordinates = [tuple(i for i, keep in enumerate(row.arrived.observation.observed[0]) if keep)
                   for row in rows]
    times = [row.arrived.acquisition.time_hours for row in rows]
    covariance = torch.cat([torch.cat([
        base[list(a)][:, list(b)] * min(t, s) / ref.reference_time_hours
        for b, s in zip(coordinates, times)], 1) for a, t in zip(coordinates, times)], 0)
    covariance += torch.eye(len(covariance), dtype=torch.float64) * ref.readout.standard_deviation ** 2
    means = torch.cat([row.mean[:, list(selected)] for row, selected in zip(rows, coordinates)], 1)
    values = torch.cat([row.arrived.observation.values[0, list(selected)]
                        for row, selected in zip(rows, coordinates)])
    return torch.distributions.MultivariateNormal(means, covariance_matrix=covariance).log_prob(values)


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("equal_clock", [False, True])
def test_dense_joint_density_equals_sequential_conditional_innovations(duplicate, equal_clock):
    ref = reference(duplicate=duplicate)
    rows = (record(ref, 1, selected=0), record(ref, 2, 1),
            record(ref, 2 if equal_clock else 3, 2, selected=2))
    innovations = torch.zeros(32, dtype=torch.float64)
    for offset, row in enumerate(rows):
        packet = row.arrived
        result = conditional_readout(ref, rows[:offset], query(
            ref, packet.acquisition.query_index, mask=tuple(packet.observation.observed[0].tolist()),
            mean=row.mean, values=packet.observation.values[0]))
        innovations += result.log_likelihood
        assert result.log_likelihood.shape == (32,)
    final = conditional_readout(ref, rows, query(ref))
    expected = dense_history(ref, rows)
    torch.testing.assert_close(innovations, expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(final.history_log_likelihood, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("duplicate", [False, True])
def test_mixing_and_inverse_train_scaler_match_full_physical_projection(duplicate):
    ref = reference(duplicate=duplicate)
    projection = explicit_field_projection(ref)
    expected = projection @ projection.T
    torch.testing.assert_close(projected_brownian_covariance(ref), expected)
    result = conditional_readout(ref, (), query(ref, 2))
    torch.testing.assert_close(result.residual_covariance, expected * 3 / 5)
    torch.testing.assert_close(result.covariance, expected * 3 / 5 + torch.eye(6) * .09)
    assert expected[0, 1] != 0  # Mixing creates correlated readout coordinates.


def test_equal_clock_acquisitions_share_brownian_error_with_independent_sensor_noise():
    ref = reference()
    rows = (record(ref, 2, selected=0), record(ref, 2, 1, selected=0))
    selected = (True, False, False, False, False, False)
    result = conditional_readout(ref, rows, query(ref, 2, mask=selected))
    prior = projected_brownian_covariance(ref)[0, 0] * 3 / 5
    noise = ref.readout.standard_deviation ** 2
    expected_residual = prior * noise / (noise + 2 * prior)
    expected_mean = predictor_mean(2)[:, :1] + prior / (noise + 2 * prior) * sum(
        row.arrived.observation.values[:, :1] - row.mean[:, :1] for row in rows)
    torch.testing.assert_close(result.mean, expected_mean)
    torch.testing.assert_close(result.residual_covariance, expected_residual.reshape(1, 1))
    torch.testing.assert_close(result.covariance, (expected_residual + noise).reshape(1, 1))
    assert result.covariance[0, 0] > noise


def test_original_t0_has_zero_brownian_error_and_sensor_noise_only():
    ref = reference()
    result = conditional_readout(ref, (), query(ref, 0, values=torch.zeros(6, dtype=torch.float64)))
    assert torch.count_nonzero(result.residual_covariance) == 0
    torch.testing.assert_close(result.covariance, torch.eye(6, dtype=torch.float64) * .09)
    torch.testing.assert_close(result.mean, predictor_mean(0))
    torch.testing.assert_close(result.history_log_likelihood, torch.zeros(32, dtype=torch.float64))
    expected = torch.distributions.Normal(predictor_mean(0), .3).log_prob(torch.zeros(6)).sum(-1)
    torch.testing.assert_close(result.log_likelihood, expected)
    with pytest.raises(ValueError, match="postzero"):
        conditional_readout(ref, (record(ref, 0),), query(ref))


def test_missing_masked_values_are_excluded_and_observed_nan_is_rejected():
    mask = (True, False, True, False, False, True)
    ref = reference(mask=mask)
    values = torch.tensor([.1, float("nan"), .3, float("inf"), float("nan"), .8])
    rows = (record(ref, 1, values=values),)
    result = conditional_readout(ref, rows, query(ref, values=values))
    sanitized = values.clone().masked_fill(~torch.tensor(mask), 0.)
    clean = conditional_readout(ref, (record(ref, 1, values=sanitized),), query(ref, values=sanitized))
    assert result.coordinates == (0, 2, 5)
    torch.testing.assert_close(result.log_likelihood, clean.log_likelihood)
    torch.testing.assert_close(result.history_log_likelihood, clean.history_log_likelihood)
    values[0] = float("nan")
    with pytest.raises(ValueError, match="Retained values must be finite"):
        conditional_readout(ref, rows, query(ref))
    with pytest.raises(ValueError, match="finite on selected"):
        conditional_readout(ref, (), query(ref, values=values))


@pytest.mark.parametrize("field,value", [
    ("executed_regimen_uid", "switched_action"), ("normalization_id", "other_scaler"),
    ("residual_artifact_ref", "other_residual"), ("support_uid", "other_support"),
    ("predictor_binding", json.dumps({"normalization_id": "synthetic_train_scale", "version": "changed"})),
    ("readout", replace(reference().readout, source_snapshot_ref="other_snapshot")),
])
def test_changed_reference_is_rejected_for_history_and_query(field, value):
    ref = reference()
    changed = replace(ref, **{field: value})
    with pytest.raises(ValueError, match="Changed action|changed action"):
        conditional_readout(ref, (replace(record(ref, 1), reference=changed),), query(ref))
    with pytest.raises(ValueError, match="changed action"):
        conditional_readout(ref, (), query(changed))


@pytest.mark.parametrize("field,value", [
    ("executed_regimen_uid", "switched_action"), ("source_time", 1.5),
    ("time_hours", 1.5), ("filtering_revision", 2),
    ("particle_uids", (tuple(f"changed{i}" for i in range(32)),)),
])
def test_stale_or_switched_acquisition_packet_is_rejected(field, value):
    ref = reference()
    row = record(ref, 1)
    acquisition = replace(row.arrived.acquisition, **{field: value})
    if field == "filtering_revision":
        observation = replace(row.arrived.observation, sequence_number=value + 1)
    else:
        observation = row.arrived.observation
    changed = replace(row, arrived=NativeArrivedObservation(acquisition, observation))
    with pytest.raises(ValueError, match="Stale or switched"):
        conditional_readout(ref, (changed,), query(ref))


@pytest.mark.parametrize("kind", ["duplicate_uid", "decreasing_clock", "query_before_history", "mutable_history"])
def test_invalid_history_order_and_container_are_rejected(kind):
    ref = reference()
    if kind == "duplicate_uid":
        history = (record(ref, 1, uid="repeated"), record(ref, 2, 1, uid="repeated"))
    elif kind == "decreasing_clock":
        history = (record(ref, 2), record(ref, 1, 1))
    else:
        history = (record(ref, 2),)
    if kind == "mutable_history":
        history = list(history)
    with pytest.raises(ValueError):
        conditional_readout(ref, history, query(ref, 1 if kind == "query_before_history" else 3))


def test_tuple_likelihoods_keep_one_tuple_for_complete_history_and_weights_are_caller_owned():
    ref = reference()
    rows = (record(ref, 1, selected=0), record(ref, 2, 1, selected=2))
    result = conditional_readout(ref, rows, query(ref))
    assert result.history_log_likelihood.shape == (32,)
    assert "weights" not in inspect.signature(conditional_readout).parameters
    weights = torch.arange(1, 33, dtype=torch.float64)
    weights /= weights.sum()
    expected = dense_history(ref, rows)
    torch.testing.assert_close(result.history_log_likelihood, expected)
    torch.testing.assert_close(torch.logsumexp(weights.log() + result.history_log_likelihood, 0),
                               torch.logsumexp(weights.log() + expected, 0))
    torch.testing.assert_close(conditional_readout(ref, rows, query(ref)).history_log_likelihood, expected)


def test_conditional_means_and_log_densities_preserve_fixed_reference_gradients():
    ref = reference()
    historical = predictor_mean(1).requires_grad_()
    desired = predictor_mean(3).requires_grad_()
    row = record(ref, 1, mean=historical)
    result = conditional_readout(ref, (row,), query(ref, mean=desired, values=torch.ones(6, dtype=torch.float64)))
    gradients = torch.autograd.grad(result.mean.sum() + result.log_likelihood.sum()
                                    + result.history_log_likelihood.sum(), (historical, desired))
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) > 0 for value in gradients)
    changed = desired.detach().clone()
    changed[7, 2] += 1e-5
    forward = conditional_readout(ref, (row,), query(ref, mean=changed, values=torch.ones(6, dtype=torch.float64)))
    changed[7, 2] -= 2e-5
    backward = conditional_readout(ref, (row,), query(ref, mean=changed, values=torch.ones(6, dtype=torch.float64)))
    numerical = ((forward.mean.sum() + forward.log_likelihood.sum())
                 - (backward.mean.sum() + backward.log_likelihood.sum())) / 2e-5
    torch.testing.assert_close(gradients[1][7, 2], numerical, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("field,value", [
    ("residual_scales", (0.,) * 26), ("inverse_normalizer_scales", (float("nan"),) * 26),
    ("reference_time_hours", 0.), ("time_hours", (0., 1., 1., 5.)),
    ("source_times", (0., 1., 3.)), ("particle_uids", (("repeated",) * 32,)),
    ("source_time_unit", "second"),
])
def test_invalid_reference_scales_clocks_and_tuple_identities_are_rejected(field, value):
    ref = replace(reference(), **{field: value})
    with pytest.raises(ValueError):
        conditional_readout(ref, (), query(ref))


@pytest.mark.parametrize("field,value", [
    ("units", "unbound"), ("role", "protected"), ("source_uid", ""),
    ("observed", (1,) * 6), ("standard_deviation", 0.),
    ("standard_deviation", 1e-200), ("standard_deviation", 1e200),
])
def test_invalid_native_metadata_mask_and_noise_are_rejected(field, value):
    ref = reference()
    ref = replace(ref, readout=replace(ref.readout, **{field: value}))
    with pytest.raises(ValueError):
        conditional_readout(ref, (), query(ref))


@pytest.mark.parametrize("mean", [torch.zeros(31, 6), torch.zeros(32, 5), torch.ones(32, 6, dtype=torch.int64),
                                  torch.full((32, 6), float("nan"))])
def test_invalid_predictor_mean_shapes_and_values_are_rejected(mean):
    ref = reference()
    with pytest.raises(ValueError, match="finite floating32x6"):
        conditional_readout(ref, (), query(ref, mean=mean))


def test_singular_finite_covariance_is_rejected_without_jitter():
    ref = reference(noise=1e-10)
    duplicate_outputs = ((1., 0., 0., 0., 0., 0.),) * 6
    ref = replace(ref, readout=replace(ref.readout, matrix=duplicate_outputs))
    with pytest.raises(ValueError, match="positive definite; no numerical jitter"):
        conditional_readout(ref, (), query(ref))


def test_from_validated_copies_only_bound_normalizer_and_residual_metadata():
    ref = reference()
    binding = json.loads(ref.predictor_binding)
    # This artificial envelope tests the metadata-copy boundary. The full
    # residual artifact validator and scientific fit/check remain caller-owned.
    normalization = SimpleNamespace(active=True, identity=ref.normalization_id,
                                    scale=torch.tensor(ref.inverse_normalizer_scales, dtype=torch.float64))
    artifact = {"law": CONTROLLED_LAW, "schema_version": "native_comparator_residual.v1",
                "status": "FIT_COMPLETE", "calibration_check_complete": True, "binding": binding,
                "scales": list(ref.residual_scales), "reference_time_hours": ref.reference_time_hours}
    arguments = dict(readout=ref.readout, normalization=normalization, residual_artifact=artifact,
        residual_artifact_ref=ref.residual_artifact_ref, predictor_binding=binding, support_uid=ref.support_uid,
        origin_posterior_uid=ref.origin_posterior_uid, sample_uids=ref.sample_uids, particle_uids=ref.particle_uids,
        executed_regimen_uid=ref.executed_regimen_uid, source_times=ref.source_times, time_hours=ref.time_hours,
        source_time_unit=ref.source_time_unit)
    actual = FixedReadoutReference.from_validated(**arguments)
    assert actual == ref
    normalization.scale.add_(100.)
    artifact["scales"][0] += 100.
    assert actual == ref
    with pytest.raises(ValueError, match="TRAIN-normalizer predictor binding"):
        FixedReadoutReference.from_validated(**{**arguments, "predictor_binding": {"normalization_id": "changed"}})
