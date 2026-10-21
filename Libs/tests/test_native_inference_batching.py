"""Actual neural batching, trajectory RNG identity and singleton diagnostics."""
import copy
import math

import pytest
import torch

from Libs.models.modules.moe import SemanticMechanismMixtureOfExperts
from Libs.tests.test_native_masked_diffusion import make_native_case


def inference_case():
    sampler, carrier, kwargs = make_native_case(time_count=5, batch=2, actions=2)
    kwargs = {key: value for key, value in kwargs.items()
              if key not in ("teacher_trajectory", "teacher_confidence")}
    kwargs.update(sample_ids=torch.tensor([11, 27]), regimen_ids=torch.tensor([7, 19]), draw_offset=5)
    sampler.inference_sample_count = 3
    sampler.inference_denoising_step_count = 2
    return sampler, carrier, kwargs


def assert_diagnostics_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        if key == "operator_splitting_identity":
            assert actual[key] == expected[key]
        elif key != "mddm_sample_chunk_size":
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6, msg=key)


def test_heterogeneous_casewise_diagnostics_match_singletons():
    _, _, kwargs = make_native_case()
    native = kwargs["native_operator"]
    hidden = torch.randn(3, 4, 6, native.hidden_dim)
    hidden[1] = hidden[1] * 3 + 4
    hidden[2] = hidden[2] * .1 - 2
    condition = torch.randn_like(hidden)
    coords = kwargs["coordinates"].reshape(1, 1, 6, -1).expand(3, 4, -1, -1)
    # Exercise nonzero attention and feed-forward gates, beyond AdaLN initialization.
    with torch.no_grad():
        for block in native.blocks:
            for adaln in (block.self_adaln, block.cross_adaln, block.feed_forward_adaln):
                adaln.condition_projection.bias[-native.hidden_dim:].fill_(.2)
        state = {key: value.clone() for key, value in native.state_dict().items()}
        output, records = native(hidden, condition, coords, kwargs["adjacency"], casewise_diagnostics=True)
        for index in range(3):
            expected, diagnostic = native(hidden[index:index+1], condition[index:index+1],
                                          coords[index:index+1], kwargs["adjacency"])
            torch.testing.assert_close(output[index:index+1], expected, rtol=2e-5, atol=2e-6)
            assert_diagnostics_equal({key: value if key == "operator_splitting_identity" else value[index]
                                      for key, value in records.items()}, diagnostic)
    assert all(torch.equal(value, native.state_dict()[key]) for key, value in state.items())


def test_casewise_moe_preserves_nonlinear_reduction_and_default_gradients():
    moe = SemanticMechanismMixtureOfExperts(4, ["a", "b"], 1, 0., 1., .1, .2, .3).eval()
    with torch.no_grad():
        moe.router.weight.zero_(); moe.router.bias.zero_()
        moe.router.weight[0, 0] = 1.; moe.router.weight[1, 0] = -1.
    hidden = torch.ones(2, 5, 4, requires_grad=True)
    with torch.no_grad(): hidden[1, :, 0] = -1.
    control = copy.deepcopy(moe)
    output, pooled = moe(hidden)
    case_output, case = control(hidden.detach().clone().requires_grad_(), casewise_diagnostics=True)
    torch.testing.assert_close(output, case_output, rtol=0, atol=0)
    assert pooled["capacity_excess"] == 0 and pooled["load_balance_loss"] == 0
    assert torch.equal(case["capacity_excess"], torch.full((2,), .25))
    assert torch.equal(case["load_balance_loss"], torch.ones(2))
    # The optional diagnostic axis never changes the learned routed output.
    output.square().sum().backward(); case_output.square().sum().backward()
    for parameter, other in zip(moe.parameters(), control.parameters()):
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


@pytest.mark.parametrize("chunk", [4, 8])
def test_seeded_batch_streams_calls_diagnostics_and_merge(chunk, monkeypatch):
    sampler, carrier, kwargs = inference_case()
    native = kwargs["native_operator"]
    generators = {}
    original_generator = sampler.generator
    def record_generator(device, sample, action, draw):
        generator = original_generator(device, sample, action, draw)
        generators[(sample, action, draw)] = generator
        return generator
    monkeypatch.setattr(sampler, "generator", record_generator)
    observed = []
    hook = native.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape[0]))
    try:
        sampler.sample_chunk_size = 1
        before = torch.random.get_rng_state().clone()
        expected = sampler(carrier, False, **kwargs)
        assert torch.equal(before, torch.random.get_rng_state())
        reference_states = {key: gen.get_state().clone() for key, gen in generators.items()}
        assert observed == [1] * 24
        observed.clear(); generators.clear()
        sampler.sample_chunk_size = chunk
        actual = sampler(carrier + 91, False, **kwargs)
        assert torch.equal(before, torch.random.get_rng_state())
        assert len(observed) == 2 * math.ceil(12 / chunk) and max(observed) == chunk
        assert sum(observed) == 24
        assert generators.keys() == reference_states.keys()
        assert all(torch.equal(gen.get_state(), reference_states[key]) for key, gen in generators.items())
        assert torch.equal(actual["sample_trajectories"], expected["sample_trajectories"])
        assert_diagnostics_equal(actual["sampler_diagnostics"], expected["sampler_diagnostics"])
        pieces = []
        for index in (1, 0):
            part = dict(kwargs, initial_state=kwargs["initial_state"][index:index+1],
                        sample_ids=kwargs["sample_ids"][index:index+1],
                        conditioning={key: value[index:index+1] for key, value in kwargs["conditioning"].items()})
            pieces.append(sampler(carrier[index:index+1], False, **part)["sample_trajectories"])
        assert torch.equal(torch.cat(list(reversed(pieces)), dim=1), expected["sample_trajectories"])
        assert torch.equal(actual["sample_trajectories"][:, :, :, 0],
                           kwargs["initial_state"][None].expand_as(actual["sample_trajectories"][:, :, :, 0]))
    finally:
        hook.remove()


def test_unseeded_global_rng_path_remains_sequential():
    sampler, carrier, kwargs = inference_case()
    sampler.sampler_seed = -1
    torch.manual_seed(59)
    sampler.sample_chunk_size = 1
    expected = sampler(carrier, False, **kwargs)
    after = torch.random.get_rng_state().clone()
    torch.manual_seed(59)
    sampler.sample_chunk_size = 8
    observed = []
    hook = kwargs["native_operator"].register_forward_pre_hook(lambda module, args: observed.append(args[0].shape[0]))
    try:
        actual = sampler(carrier, False, **kwargs)
    finally:
        hook.remove()
    assert observed == [1] * 24
    assert torch.equal(actual["sample_trajectories"], expected["sample_trajectories"])
    assert torch.equal(after, torch.random.get_rng_state())


@pytest.mark.parametrize("policy", ["occupancy_program", "drug_id", "target_id", "scalar_exposure"])
def test_batched_logits_preserve_all_conditioning_policies(policy):
    sampler, carrier, kwargs = inference_case()
    sampler.action_policy_name = policy  # Probe the learned conditioning branches only.
    native = kwargs["native_operator"]
    template, _ = sampler.pack(carrier[0, 0], native)
    contexts = []
    for index in range(2):
        row = {key: value[index, 0] for key, value in kwargs["conditioning"].items()}
        row.update(initial_state=kwargs["initial_state"][index, 0],
                   coefficient_support=torch.ones_like(template, dtype=torch.bool),
                   action_identity=template.new_tensor([.1, .2 + index, .3, .4]),
                   scalar_exposure_auc=template.new_full((3, 2, 1), .6 + index))
        contexts.append(row)
    values = torch.randn(2, *template.shape)
    masks = values > .2
    context = {key: torch.stack([row[key] for row in contexts]) for key in contexts[0]}
    with torch.no_grad():
        logits, diagnostics = sampler.logits(values, masks, template.new_tensor(.4), context,
                                               native, kwargs["adjacency"], kwargs["coordinates"])
        for index in range(2):
            expected, record = sampler.logits(values[index], masks[index], template.new_tensor(.4), contexts[index],
                                                native, kwargs["adjacency"], kwargs["coordinates"])
            torch.testing.assert_close(logits[index], expected, rtol=2e-5, atol=2e-6)
            assert_diagnostics_equal({key: value if key == "operator_splitting_identity" else value[index]
                                      for key, value in diagnostics.items()}, record)
