"""Complete named physical hypotheses on disposable source-format inputs."""
from __future__ import annotations

import copy

import pytest
import torch

from Libs.tests.test_native_observable_source import fixture
from Libs.utils.model import native_observable_source as source
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.joint_physical_parameters import (
    CONDITIONING_KIND, bind_candidate_parameters, candidate_conditioning,
    parameter_schema_from_config, select_candidate_samples,
)
from Libs.models.modules.world_state import JointParticleSupport
from Libs.exp.training.pretrain_microenvironment_world_model import joint_inference_modules

FAMILIES = [source.ELECTRICAL, source.SYNAPTIC, source.MOLECULAR]


def named_case(tmp_path, family):
    config, group, generated = fixture(family)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"][
        "physical_parameter_conditioning"] = CONDITIONING_KIND
    records = []
    for atom in range(4):
        row = []
        for index in range(2):
            action = source.action_definition(config, family, index)
            estimate = source.estimated_exposure(config, generated["parameters"][atom], action) if family == source.MOLECULAR else None
            row.append(source.compiled_program(config, group, generated["support"], atom,
                generated["parameters"][atom], action, "0" * 64, estimate))
        records.append(row)
    raw = source.joint_source(config, group, generated, records)
    producer = {"parameters": generated["parameters"], "hidden_index": generated["hidden_index"]}
    schema = parameter_schema_from_config(config)
    kwargs = {"record_ref": str(tmp_path / "producer.json"), "producer_commit": "disposable_source_definition"}
    bound = bind_candidate_parameters(raw, group, producer, schema, **kwargs)
    return config, group, producer, raw, bound, schema, kwargs


@pytest.mark.parametrize("family", FAMILIES)
def test_named_parameters_roundtrip_selection_and_program_identity(tmp_path, family):
    config, group, producer, raw, bound, schema, _ = named_case(tmp_path, family)
    support, _ = native.parse_joint_source(bound)
    named = support.physical_parameter_hypotheses
    assert bound["particle_programs"] == raw["particle_programs"]
    assert support.to_payload() == bound["support"]
    assert named["values"].dtype == torch.float64
    names = [row["name"] for row in named["declarations"]]
    assert names == sorted(config["native_observable_bridge"]["physical_families"][family]["parameters"])
    assert named["particle_uids"] == [group["particle_uids"]]
    for atom in range(4):
        assert named["values"][0, atom].tolist() == [producer["parameters"][atom][name] for name in names]
    reordered = native.select_joint_source(bound, [group["sample_uid"]],
        list(reversed(bound["particle_programs"]["regimen_uids"])))
    assert reordered["support"]["physical_parameter_hypotheses"] == bound["support"]["physical_parameter_hypotheses"]
    features = candidate_conditioning(support, schema)
    assert features.shape == (1, 4, 68)
    assert features[..., 34:].sum(-1).tolist() == [[len(names)] * 4]


@pytest.mark.parametrize("family", FAMILIES)
def test_each_named_value_reaches_actual_posterior_conditioning(tmp_path, family):
    config, _, _, _, bound, schema, _ = named_case(tmp_path, family)
    support, observations = native.parse_joint_source(bound)
    torch.manual_seed(47191)
    (_, _, model), width = joint_inference_modules(config, 12, 26)
    encoded = torch.zeros(1, model.factor_dimensions["z_tx"])
    features = torch.zeros(1, width)
    baseline = model.infer_joint(encoded, features, support, observations)
    declarations = schema["families"][family]
    for index, declaration in enumerate(declarations):
        changed = copy.deepcopy(bound["support"])
        changed["physical_parameter_hypotheses"]["values"][0][2][index] += declaration["reference_scale"] * .05
        alternative = JointParticleSupport.from_payload(changed)
        actual = model.infer_joint(encoded, features, alternative, observations)
        assert not torch.equal(actual.particle_context[:, 2], baseline.particle_context[:, 2]), declaration["name"]
        assert torch.equal(actual.particle_context[:, [0, 1, 3]], baseline.particle_context[:, [0, 1, 3]])
        assert torch.equal(actual.log_weights, baseline.log_weights)
        assert actual.posterior_uid != baseline.posterior_uid
    actual.particle_context.square().mean().backward()
    assert model.joint_candidate_parameter_projection.weight.grad is not None
    assert torch.isfinite(model.joint_candidate_parameter_projection.weight.grad).all()
    draws = baseline.resample(12, torch.Generator().manual_seed(47199))
    expected = support.physical_parameter_hypotheses["values"][0, draws["indices"][0]]
    assert draws["physical_parameter_hypotheses"]["values"][0] == expected.tolist()
    assert draws["physical_parameter_hypotheses"]["particle_uids"] == [list(row) for row in draws["particle_uids"]]
    stale = copy.deepcopy(support.physical_parameter_hypotheses)
    support.physical_parameter_hypotheses["values"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="changed support"):
        baseline.assert_current_binding()
    support.physical_parameter_hypotheses["values"].copy_(stale["values"])


@pytest.mark.parametrize("family", FAMILIES)
def test_hidden_and_future_metadata_do_not_change_admitted_conditions_or_rng(tmp_path, family):
    config, group, producer, raw, bound, schema, kwargs = named_case(tmp_path, family)
    altered = copy.deepcopy(producer)
    altered.update(hidden_index=(producer["hidden_index"] + 1) % 4,
                   selected_truth={"not_an_input": 1e99}, future_targets=[-1e99, 1e99])
    changed = bind_candidate_parameters(raw, group, altered, schema, **kwargs)
    assert changed == bound
    assert native._source_digest(changed) == native._source_digest(bound)
    torch.manual_seed(49211)
    (_, _, model), width = joint_inference_modules(config, 12, 26)
    posteriors = []
    for current in (bound, changed):
        support, observations = native.parse_joint_source(current)
        posteriors.append(model.infer_joint(torch.zeros(1, model.factor_dimensions["z_tx"]),
                                           torch.zeros(1, width), support, observations))
    assert posteriors[0].posterior_uid == posteriors[1].posterior_uid
    assert torch.equal(posteriors[0].particle_context, posteriors[1].particle_context)
    assert torch.equal(posteriors[0].resample(20, torch.Generator().manual_seed(38))["indices"],
                       posteriors[1].resample(20, torch.Generator().manual_seed(38))["indices"])


@pytest.mark.parametrize("fault", ["unit", "name", "group", "particle", "bool", "missing"])
def test_named_parameter_binding_rejects_malformed_inputs(tmp_path, fault):
    _, _, _, _, bound, schema, _ = named_case(tmp_path, source.ELECTRICAL)
    changed = copy.deepcopy(bound["support"])
    named = changed["physical_parameter_hypotheses"]
    if fault == "unit":
        named["declarations"][0]["unit"] = "wrong"
    elif fault == "name":
        named["declarations"][0]["name"] = "wrong"
    elif fault == "group":
        named["group_uids"][0] = "other"
    elif fault == "particle":
        named["particle_uids"][0][0] = "other"
    elif fault == "bool":
        named["values"][0][0][0] = True
    else:
        del changed["physical_parameter_hypotheses"]
    with pytest.raises(ValueError):
        candidate_conditioning(JointParticleSupport.from_payload(changed), schema)


def test_named_PK_subset_cannot_disagree_with_retained_program(tmp_path):
    _, _, _, _, bound, _, _ = named_case(tmp_path, source.MOLECULAR)
    names = [row["name"] for row in bound["support"]["physical_parameter_hypotheses"]["declarations"]]
    for name in ("initial_free_agent_exposure", "dose_to_exposure_gain", "exposure_clearance_rate", "binding_affinity"):
        changed = copy.deepcopy(bound)
        changed["support"]["physical_parameter_hypotheses"]["values"][0][1][names.index(name)] += .1
        with pytest.raises(ValueError, match="estimated-PK"):
            native.parse_joint_source(changed)


def test_named_parameter_checkpoint_schema_cannot_be_reordered(tmp_path):
    config, _, _, _, _, _, _ = named_case(tmp_path, source.ELECTRICAL)
    (_, _, model), _ = joint_inference_modules(config, 12, 26)
    state = model.state_dict()
    state["joint_parameter_schema_identity"] = state["joint_parameter_schema_identity"].roll(1)
    with pytest.raises(ValueError, match="checkpoint schema"):
        model.load_state_dict(state)


def test_candidate_sample_selection_retains_source_and_values(tmp_path):
    _, _, _, _, bound, _, _ = named_case(tmp_path, source.SYNAPTIC)
    value = copy.deepcopy(bound["support"]["physical_parameter_hypotheses"])
    for key in ("sample_uids", "group_uids", "particle_uids", "source_record_refs", "values"):
        value[key].append(copy.deepcopy(value[key][0]))
    value["sample_uids"][1] = value["group_uids"][1] = "second"
    value["particle_uids"][1] = ["second:" + str(i) for i in range(4)]
    value["values"][1][0][0] += .03
    selected = select_candidate_samples(value, [1, 0])
    for key in ("sample_uids", "group_uids", "particle_uids", "source_record_refs", "values"):
        assert selected[key] == [value[key][1], value[key][0]]
