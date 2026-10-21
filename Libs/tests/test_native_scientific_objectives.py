"""Exact small-law checks of CRPS value and score-function gradients."""
import itertools

import pytest
import torch

from Libs.utils.model.native_scientific_objectives import stratified_mixture_crps, paired_action_score


def test_crps_score_gradient_matches_exact_bernoulli_law():
    theta = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    probability = theta.sigmoid()
    expected_value, expected_gradient = 0., 0.
    for values in itertools.product((0., 1.), repeat=2):
        x = torch.tensor(values, dtype=torch.float64)
        logp = x * probability.log() + (1 - x) * (1 - probability).log()
        score = stratified_mixture_crps(x.reshape(1, 2, 1, 1), logp.reshape(1, 2, 1),
                                        theta.new_ones(1), theta.new_zeros(1, 1),
                                        torch.ones(1, 1, dtype=torch.bool)).sum()
        gradient = torch.autograd.grad(score, theta, retain_graph=True)[0]
        mass = logp.sum().exp().detach()
        expected_value += mass * score.detach()
        expected_gradient += mass * gradient
    exact = probability.square()  # CRPS(Bernoulli(p), y=0).
    assert torch.allclose(expected_value, exact)
    assert torch.allclose(expected_gradient, torch.autograd.grad(exact, theta)[0])


def test_crps_exact_mixture_weight_gradient_and_mask():
    theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    weights = torch.stack((theta.sigmoid(), 1 - theta.sigmoid()))
    samples = torch.tensor([[[[0., 999.]], [[0., -999.]]],
                            [[[1., 777.]], [[1., -777.]]]], dtype=torch.float64)
    score = stratified_mixture_crps(samples, torch.zeros(2, 2, 1, dtype=torch.float64),
                                    weights, torch.zeros(1, 2, dtype=torch.float64),
                                    torch.tensor([[True, False]])).sum()
    exact = weights[1].square()
    assert torch.allclose(score, exact)
    assert torch.allclose(torch.autograd.grad(score, theta, retain_graph=True)[0],
                          torch.autograd.grad(exact, theta)[0])


def test_action_contrast_uses_both_targets_and_has_exact_gradient():
    theta = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    p = theta.sigmoid()
    expected, derivative = 0., 0.
    # Two independent action draws; repeat the same law experiment twice.
    for a, b in itertools.product((0., 1.), repeat=2):
        sample = theta.new_tensor([[[[a], [b]], [[a], [b]]]])
        action = theta.new_tensor([a, b])
        logp = (action * p.log() + (1-action) * (1-p).log()).repeat(1, 2, 1)
        score = paired_action_score(sample, logp, theta.new_ones(1),
                                    theta.new_tensor([[1.], [0.]]),
                                    torch.ones(2, 1, dtype=torch.bool), theta.new_ones(2))
        grad = torch.autograd.grad(score, theta, retain_graph=True)[0]
        mass = (action * p.log() + (1-action) * (1-p).log()).sum().exp().detach()
        expected += mass * score.detach()
        derivative += mass * grad
    exact = 1 + 2*p*(1-p)
    assert torch.allclose(expected, exact)
    assert torch.allclose(derivative, torch.autograd.grad(exact, theta)[0])


def test_scores_reject_missing_draws_and_support():
    with pytest.raises(ValueError, match="two draws"):
        stratified_mixture_crps(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2),
                                torch.ones(1), torch.zeros(2, 3), torch.ones(2, 3, dtype=torch.bool))


def test_development_selection_retains_independent_best_and_last_and_caps():
    from Libs.utils.model.native_training_selection import NativeTrainingSelection
    policy = {"min_delta":.01, "minimum_epochs":2, "minimum_steps":2,
              "max_epochs":10, "max_steps":20, "max_seconds":100., "patience":2}
    state = NativeTrainingSelection(policy)
    assert not state.update(role="development", epoch=1, steps=1, training_loss=2.,
                            development_loss=1., checkpoint_path="epoch1.pt", elapsed_seconds=1.)
    assert not state.update(role="development", epoch=2, steps=2, training_loss=1.,
                            development_loss=1.1, checkpoint_path="epoch2.pt", elapsed_seconds=2.)
    assert state.update(role="development", epoch=3, steps=3, training_loss=.5,
                        development_loss=1.2, checkpoint_path="epoch3.pt", elapsed_seconds=3.)
    assert state.best["checkpoint_path"] == "epoch1.pt"
    assert state.last["checkpoint_path"] == "epoch3.pt"
    assert state.best is not state.last and not state.payload()["sufficient_training"]
    with pytest.raises(ValueError, match="development"):
        state.update(role="calibration", epoch=4, steps=4, training_loss=.5,
                     development_loss=.1, checkpoint_path="bad.pt", elapsed_seconds=4.)
    cap = NativeTrainingSelection(policy)
    cap.update(role="development", epoch=1, steps=1, training_loss=1.,
               development_loss=1., checkpoint_path="last.pt", elapsed_seconds=100.)
    assert cap.terminal == "INCOMPLETE_BUDGET_CAP"
