"""Meaningful complete-32-tuple tests of conditional decision-risk queries."""
from dataclasses import fields, replace
import copy

import pytest
import torch

from Libs.scripts.builders.build_sample_observation_registry import NativePredoseMeasurementProvider
from Libs.utils.data.joint_observation import RetainedObservation, retained_log_likelihood
from Libs.utils.data.raw_data_registry import load_config
from Libs.models.modules.world_state import _normalized_log_weights
from Libs.utils.model.native_arrived_observation import NativeCNSReadout
from Libs.utils.model.native_predose_query import (
    NativePredoseChannel, admit_predose_observation, plan_predose_query,
    predose_query_policy, run_native_predose_episode,
)


@pytest.fixture(scope="module")
def config():
    full = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    result = {"native_predose_query": copy.deepcopy(full["native_predose_query"]),
              "schemas": {"native_predose_episode": full["schemas"]["native_predose_episode"]}}
    return result


def channel_and_loss(*, xor=False):
    spec = NativeCNSReadout("native_arrived_cns_selector", "controlled_native_monitoring_gaussian",
        "synthetic", "synthetic:v1", "validation", "synthetic_family", "synthetic:v1:fixed_source_scales",
        "source_reference_standardized", ("r",), ("c",), tuple("abcdef"),
        tuple((0, 0, i) for i in range(6)), (1.,)*6,
        tuple(tuple(float(i == j) for j in range(6)) for i in range(6)),
        (True, True, True, False, False, False), .2)
    a, b = (torch.arange(32)//8)%2, torch.arange(32)//16
    mean = torch.zeros(1, 32, 6)
    mean[0, :, 0], mean[0, :, 1] = (a*2-1)*8., (b*2-1)*8.
    original = RetainedObservation("initial", spec.source_uid, spec.source_snapshot_ref, ("sample",),
        spec.role, "controlled_predose_selector", "gaussian", spec.units, torch.zeros(1, 6),
        torch.tensor([spec.observed]), torch.zeros(1, 6, dtype=torch.bool), tuple(range(6)), 0,
        normalization_ref=spec.normalization_ref, assay_uid="controlled_predose_gaussian")
    channel = NativePredoseChannel(spec, "support", tuple("tuple"+str(i) for i in range(32)), original,
        {"law": "gaussian", "units": spec.units, "mean": mean, "variance": torch.full_like(mean, .2**2)},
        torch.zeros_like(mean))
    target = (a ^ b) if xor else a
    loss = torch.stack((target.float(), 1.-target.float(), torch.full((32,), .8)), -1)[:, None]
    return channel, loss, torch.full((1, 32), -torch.tensor(32.).double().log(), dtype=torch.float64)


def outcome(request, value):
    values = request.observation.values.clone()
    values[0, request.coordinate] = value
    return replace(request.observation, values=values)


def test_informative_not_merely_uncertain_coordinate_changes_terminal_action(config):
    channel, loss, logs = channel_and_loss()
    original = logs.clone()
    plan = plan_predose_query(channel, logs, loss, (), predose_query_policy(config))
    assert plan["selected_coordinate"] == 0
    # Coordinate1 strongly identifies an independent bit but cannot change the
    # optimal intervention. Its apparent utility comes only from later query0,
    # and two costs make it worse than measuring coordinate0 directly.
    values = {row["coordinate"]: row for row in plan["candidates"]}
    assert values[0]["estimated_net_benefit"] > values[1]["estimated_net_benefit"]
    updated, observations, report = run_native_predose_episode(channel, logs, loss, ("a0", "a1", "a2"),
        lambda request: outcome(request, 8.), config, origin_posterior_uid="posterior0")
    assert report["status"] == "component_complete" and report["accepted_query_count"] == 1
    assert report["initial_selected_regimen_uid"] == "a0" and report["selected_regimen_uid"] == "a1"
    assert report["measurement_units_spent"] == 1 and report["total_measurement_loss_penalty"] == .01
    assert report["stop_reason"] == "no_positive_estimated_net_risk_reduction"
    assert len(observations) == 1 and updated.shape == (1, 32) and torch.isfinite(updated).all()
    assert torch.equal(logs, original) and not report["physical_time_advanced"]


def test_dominant_action_has_exact_zero_information_value(config):
    channel, _, logs = channel_and_loss()
    loss = torch.arange(3.).expand(32, 1, 3).clone()
    plan = plan_predose_query(channel, logs, loss, (), predose_query_policy(config))
    assert plan["selected_coordinate"] is None
    assert all(row["estimated_immediate_risk_reduction"] == 0 for row in plan["candidates"])
    def forbidden(request):
        raise AssertionError("A zero-value channel must never be issued")
    _, _, report = run_native_predose_episode(channel, logs, loss, ("a0", "a1", "a2"),
        forbidden, config, origin_posterior_uid="posterior0")
    assert report["measurement_units_spent"] == 0


def test_depth_two_complementary_queries_replan_and_use_current_weights(config):
    channel, loss, logs = channel_and_loss(xor=True)
    initial = logs.clone()
    def actual(request):
        return outcome(request, -8. if request.coordinate == 0 else 8.)
    updated, observations, report = run_native_predose_episode(channel, logs, loss, ("a0", "a1", "a2"),
        actual, config, origin_posterior_uid="posterior0")
    assert report["accepted_query_count"] == 2 and report["measurement_units_spent"] == 2
    assert set(report["completed_coordinate_queries"]) == {0, 1}
    assert report["selected_regimen_uid"] == "a1"
    expected = logs
    for observation in observations:
        likelihood = retained_log_likelihood(observation, channel.parameters, channel.nuisance, {})
        expected = _normalized_log_weights(expected+likelihood.double())
    restarted = _normalized_log_weights(initial+likelihood.double())
    assert torch.equal(updated, expected) and not torch.equal(updated, restarted)
    assert len(report["steps"]) == 3 and report["steps"][1]["prior_log_weights"] != report["initial_log_weights"]


@pytest.mark.parametrize("kind", ["runtime", "nan", "wrong_source", "wrong_mask", "wrong_units", "admission"])
def test_real_failure_keeps_cost_quarantines_and_withholds_update(config, kind):
    channel, loss, logs = channel_and_loss()
    def failed(request):
        if kind == "runtime":
            raise RuntimeError("actual controlled provider failure")
        observation = outcome(request, float("nan") if kind == "nan" else 8.)
        if kind == "wrong_source": observation = replace(observation, source_uid="alien")
        if kind == "wrong_mask": observation = replace(observation, observed=~observation.observed)
        if kind == "wrong_units": observation = replace(observation, units="native_Vm")
        if kind == "admission": observation = replace(observation, informative_admission=True)
        return observation
    updated, observations, report = run_native_predose_episode(channel, logs, loss, ("a0", "a1", "a2"),
        failed, config, origin_posterior_uid="posterior0")
    assert report["status"] == "failed" and report["measurement_units_spent"] == 1
    assert report["selected_regimen_uid"] is None and not observations
    assert torch.equal(updated, logs)
    assert report["steps"][0]["biological_update_withheld"]
    assert (report["steps"][0]["quarantined_observation"] is not None) == (kind != "runtime")


def test_duplicate_unsupported_and_changed_original_channel_reject(config):
    channel, _, logs = channel_and_loss()
    for coordinate, sequence, used in [(0, 2, (0,)), (4, 1, ()), (1, 1, (0,))]:
        with pytest.raises(ValueError, match="Unsupported, duplicate"):
            channel.request(coordinate, sequence, "p", used)
    for original in [replace(channel.original, source_uid="alien"),
                     replace(channel.original, units="native_Vm"),
                     replace(channel.original, observed=~channel.original.observed)]:
        with pytest.raises(ValueError, match="original source/mask/units"):
            replace(channel, original=original).validate()
    req = channel.request(0, 1, "p", ())
    with pytest.raises(ValueError, match="acquisition"):
        admit_predose_observation(channel, logs, replace(req, variance=req.variance*2), outcome(req, 8.))
    bad = copy.deepcopy(config)
    bad["native_predose_query"]["integration_noise_replicates"] = 0
    with pytest.raises(ValueError, match="finite integration"):
        predose_query_policy(bad)


def test_provider_same_world_independent_noise_and_truth_isolation(config):
    channel, _, logs = channel_and_loss()
    private = torch.tensor([[[[8., -8., 0., 0., 0., 0.]]]])
    before = private.clone()
    generator = torch.Generator().manual_seed(83)
    witness = torch.Generator().manual_seed(83)
    global_state = torch.random.get_rng_state().clone()
    provider = NativePredoseMeasurementProvider(private, channel, generator)
    first = channel.request(0, 1, "p", ())
    one = provider(first)
    second = channel.request(1, 2, "p", (0,))
    two = provider(second)
    scale = channel.parameters["variance"][0, 0, 0].sqrt().double()
    expected1 = 8.+scale*torch.randn((), dtype=torch.float64, generator=witness)
    expected2 = -8.+scale*torch.randn((), dtype=torch.float64, generator=witness)
    assert one.values[0, 0] == expected1 and two.values[0, 1] == expected2
    assert torch.equal(private, before) and torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.equal(generator.get_state(), witness.get_state())
    forbidden = {"private_field", "generating_fields", "generating_indices", "measurement_generator", "rng_seed"}
    assert not forbidden.intersection(field.name for field in fields(one))
    assert one.observed.sum() == 1 and torch.isnan(one.values[~one.observed]).all()
    # A rejected duplicate cannot silently acquire another replicate.
    with pytest.raises(ValueError, match="duplicate"):
        provider(first)
    expected = _normalized_log_weights(logs+retained_log_likelihood(one, channel.parameters, channel.nuisance, {}).double())
    assert torch.equal(admit_predose_observation(channel, logs, first, one), expected)
