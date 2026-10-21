from __future__ import annotations

import copy
import json

import pytest
import torch

from Libs.tests.test_native_diffusion_data import producer_case
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_field_normalization import (
    NativeFieldNormalization, bind_artifact, fit_transform, normalization_spec)
from Libs.utils.evaluation.native_rollout_metrics import (
    metric_policy, metric_protocol, score_native_episode, build_native_metric_payload,
    validate_native_metric_payload)


@pytest.fixture(scope="module")
def metric_case(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native_metric_coordinates")
    config, registry, arrays, _ = producer_case(directory)
    records = native.select_records(registry, config, "train")
    aligned, _ = native.load_records(registry, records, config)
    artifact = fit_transform(registry["grid"], records, aligned, config)
    path = directory / "training-normalization.json"
    path.write_text(json.dumps(artifact))
    grid = bind_artifact(registry["grid"], path, config)
    normalization = NativeFieldNormalization(normalization_spec(grid, config), 15)
    policy = metric_policy(config)
    protocol = metric_protocol(policy, [.05, .5, .95], normalization.identity)
    return config, grid, normalization, protocol, path


def identity(episode="sample-a", family="family-a", actions=1):
    return {"episode_uid": episode, "sample_uid": episode, "family_uid": family,
            "group_kind": "action_family", "source_uid": "fixture-source",
            "regimen_uids": [f"regimen-{i}" for i in range(actions)],
            "action_uids": [f"action-{i}" for i in range(actions)],
            "trajectory_uids": [f"trajectory-{episode}-{i}" for i in range(actions)],
            "posterior_uids": [f"posterior-{episode}"] * actions}


def fields(case, draws, target, actions=1):
    _, grid, normalization, _, _ = case
    shape = (actions, len(grid["time_hours"]), len(grid["region_id"]),
             len(grid["compartment_id"]), len(grid["species_id"]))
    x = torch.tensor(draws, dtype=torch.float64).reshape(-1, *([1] * 5)).expand(-1, *shape)
    y = torch.full(shape, float(target), dtype=torch.float64)
    return normalization.inverse(x), normalization.inverse(y), torch.ones(shape, dtype=torch.bool)


@pytest.mark.parametrize("draws,target,expected", [([0.,1.,3.],2.,1/3), ([0.,2.],3.,1.), ([1.,1.,1.],3.,2.)])
def test_off_diagonal_hand_examples_and_degenerate_draws(metric_case,draws,target,expected):
    _,grid,norm,protocol,path = metric_case
    original = path.read_bytes()
    x,y,mask = fields(metric_case,draws,target)
    row = score_native_episode(x,y,mask,identity(),norm,grid,protocol)
    assert row["metrics"]["crps"] == pytest.approx(expected,abs=1e-12)
    assert row["metrics"]["normalized_rmse"] == pytest.approx(abs(sum(draws)/len(draws)-target))
    assert path.read_bytes() == original


def test_sorted_pair_algebra_permutation_and_action_reassembly(metric_case):
    _,grid,norm,protocol,_ = metric_case
    x,y,mask = fields(metric_case,[-2.,.3,1.,4.,7.],1.2,actions=2)
    # Distinct species source scales cannot become a target-derived denominator.
    x[:,1] += norm.scale * .7
    reference_x, reference_y = norm.normalize(x), norm.normalize(y)
    pair = sum((reference_x[i]-reference_x[j]).abs() for i in range(5) for j in range(i+1,5))
    expected = ((reference_x-reference_y).abs().mean(0)-pair/(5*4)).mean()
    row = score_native_episode(x,y,mask,identity(actions=2),norm,grid,protocol)
    changed = score_native_episode(torch.cat([x[[4,2,0,1,3],1:],x[[4,2,0,1,3],:1]],dim=1),
        y.flip(0),mask.flip(0),identity(actions=2),norm,grid,protocol)
    assert row["metrics"]["crps"] == pytest.approx(float(expected),abs=1e-12)
    assert changed["metrics"] == pytest.approx(row["metrics"])


def test_equal_episode_denominators_and_post_aggregate_calibration(metric_case):
    config,grid,norm,protocol,_ = metric_case
    rows = []
    for index, target in enumerate([1.,3.]):
        x,y,mask = fields(metric_case,[1.,1.,1.],target)
        mask[:] = False
        mask.flatten()[:1 if index == 0 else 16] = True
        rows.append(score_native_episode(x,y,mask,identity(f"sample-{index}",f"family-{index}"),norm,grid,protocol))
    payload = build_native_metric_payload(rows,protocol,{},config)
    assert payload["metrics"]["normalized_rmse"]["mean"] == pytest.approx(1.)
    assert payload["metrics"]["predictive_coverage"]["mean"] == .5
    assert payload["metrics"]["predictive_calibration_error"]["mean"] == pytest.approx(.4)
    assert payload["counts"] == {"episode_count":2,"defined_episode_count":2,"undefined_episode_count":0,
        "failed_episode_count":0,"group_count":2,"defined_group_count":2,"valid_cell_count":17,"realized_draw_count":3}


@pytest.mark.parametrize("target", [1.,19.])
def test_linear_central_quantiles_include_boundary_ties(metric_case,target):
    _,grid,norm,protocol,_ = metric_case
    x,y,mask = fields(metric_case,list(range(21)),target)
    row = score_native_episode(x,y,mask,identity(),norm,grid,protocol)
    assert row["metrics"]["predictive_coverage"] == 1.


@pytest.mark.parametrize("kind", ["masked","one_draw"])
def test_undefined_episodes_are_retained(metric_case,kind):
    config,grid,norm,protocol,_ = metric_case
    x,y,mask = fields(metric_case,[1.] if kind == "one_draw" else [1.,2.],1.)
    if kind == "masked":
        mask[:] = False
        y[:] = float("nan")  # Source-declared missing cells are not scored.
    row = score_native_episode(x,y,mask,identity(),norm,grid,protocol)
    assert row["status"] == ("fully_masked" if kind == "masked" else "insufficient_draws")
    payload = build_native_metric_payload([row],protocol,{},config)
    assert payload["status"] == "incomplete" and payload["counts"]["undefined_episode_count"] == 1
    assert all(value == {"mean":None,"count":0} for value in payload["metrics"].values())
    json.dumps(payload,allow_nan=False)


@pytest.mark.parametrize("mutation", ["shape","mask","nan","time","units","actions"])
def test_metric_contract_rejects_mismatches(metric_case,mutation):
    _,grid,norm,protocol,_ = metric_case
    grid = copy.deepcopy(grid)
    x,y,mask = fields(metric_case,[0.,1.],.5)
    ids = identity()
    if mutation == "shape":
        y = y[:,:-1]
    elif mutation == "mask":
        mask = mask.float()
    elif mutation == "nan":
        x[0] = float("nan")
    elif mutation == "time":
        grid["time_hours"][-1] += 1
    elif mutation == "units":
        grid["species_units"][grid["species_id"][0]] = "wrong"
    else:
        ids["regimen_uids"] = []
    with pytest.raises(ValueError):
        score_native_episode(x,y,mask,ids,norm,grid,protocol)


def test_payload_validator_rejects_forged_completeness(metric_case):
    config,grid,norm,protocol,_ = metric_case
    x,y,mask = fields(metric_case,[0.,1.],.5)
    mask[:] = False
    row = score_native_episode(x,y,mask,identity(),norm,grid,protocol)
    payload = build_native_metric_payload([row],protocol,{},config)
    payload["status"] = "complete"
    with pytest.raises(ValueError,match="denominator"):
        validate_native_metric_payload(payload,config)
