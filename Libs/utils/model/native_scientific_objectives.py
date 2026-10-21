"""Proper sample scores and action contrasts for complete native posterior groups."""
from __future__ import annotations

import torch


def _law(samples: torch.Tensor, log_probabilities: torch.Tensor, weights: torch.Tensor,
         target: torch.Tensor, support: torch.Tensor) -> tuple[int, int]:
    if samples.ndim < 4 or samples.shape[2:] != target.shape or support.shape != target.shape:
        raise ValueError("Scores require particle/draw/action/field axes and aligned targets/support")
    j, k = samples.shape[:2]
    if k < 2 or log_probabilities.shape != (j, k, target.shape[0]) or weights.shape != (j,):
        raise ValueError("Proper sample scoring requires two draws and complete posterior weights")
    if support.dtype != torch.bool or not support.flatten(1).any(-1).all():
        raise ValueError("Every scored action requires explicit nonempty Boolean support")
    if not all(torch.isfinite(x).all() for x in (samples, log_probabilities, weights, target)):
        raise ValueError("Sample score contains nonfinite values")
    if (weights < 0).any() or not torch.allclose(weights.sum(), weights.new_tensor(1.)):
        raise ValueError("Sample score posterior weights must sum to one")
    return j, k


def _supported_mean(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(values.ndim - support.ndim + 1, values.ndim))
    return (values * support).sum(axes) / support.flatten(1).sum(-1)


def stratified_mixture_crps(samples: torch.Tensor, log_probabilities: torch.Tensor,
                            weights: torch.Tensor, target: torch.Tensor,
                            support: torch.Tensor) -> torch.Tensor:
    """Unbiased CRPS per action, with exact categorical score-function gradients.

    Conditional draws are independent within each particle. The within-particle
    U statistic excludes identical draws; cross-particle terms use independent
    strata. Posterior weights differentiate the mixture quadrature directly.
    Samples are detached and path log probabilities differentiate their law.
    """
    j, k = _law(samples, log_probabilities, weights, target, support)
    pool = samples.detach().flatten(0, 1)
    mass = weights.repeat_interleave(k) / k
    first = _supported_mean((pool - target).abs(), support)
    distances = torch.stack([_supported_mean((row - pool).abs(), support) for row in pool])
    strata = torch.arange(j, device=weights.device).repeat_interleave(k)
    same = strata[:, None] == strata[None, :]
    correction = torch.where(same, weights.new_tensor(k / (k - 1)), weights.new_tensor(1.))
    correction = correction.masked_fill(torch.eye(j * k, dtype=torch.bool, device=weights.device), 0.)
    pairs = mass[:, None] * mass[None, :] * correction
    value = (mass[:, None] * first).sum(0) - .5 * (pairs[:, :, None] * distances).sum((0, 1))
    # Each pair depends on both sampled paths, so its derivative coefficient
    # contains the complete row sum, not half that row.
    costs = mass[:, None] * first - (pairs[:, :, None] * distances).sum(1)
    surrogate = (costs.detach() * log_probabilities.flatten(0, 1)).sum(0)
    return value + surrogate - surrogate.detach()


def paired_action_score(samples: torch.Tensor, log_probabilities: torch.Tensor,
                        weights: torch.Tensor, target: torch.Tensor,
                        support: torch.Tensor, action_weights: torch.Tensor) -> torch.Tensor:
    """E ||(X^a-X^b)-(Y^a-Y^b)||^2 over shared complete particles and draws."""
    j, k = _law(samples, log_probabilities, weights, target, support)
    if target.shape[0] < 2 or action_weights.shape != (target.shape[0],):
        raise ValueError("Paired targets require at least two actions and aligned confidence")
    terms = []
    for a in range(target.shape[0]):
        for b in range(a + 1, target.shape[0]):
            mask = support[a] & support[b]
            if not mask.any():
                raise ValueError("Paired actions have no shared supported quantities")
            error = ((samples[:, :, a].detach() - samples[:, :, b].detach())
                     - (target[a] - target[b])).square()
            cost = error[..., mask].mean(-1)
            value = (weights[:, None] * cost).sum() / k
            surrogate = (weights.detach()[:, None] * cost
                         * (log_probabilities[:, :, a] + log_probabilities[:, :, b])).sum() / k
            terms.append((action_weights[a] + action_weights[b]) / 2
                         * (value + surrogate - surrogate.detach()))
    return torch.stack(terms).mean()


def expected_constraint_score(costs: torch.Tensor, log_probabilities: torch.Tensor,
                              weights: torch.Tensor) -> torch.Tensor:
    """Differentiate a prespecified constraint cost on unprojected sampled fields."""
    if costs.shape != log_probabilities.shape or costs.shape[0] != len(weights):
        raise ValueError("Constraint costs must align to particle/draw/action paths")
    value = (weights[:, None, None] * costs.detach()).sum(0).mean()
    surrogate = (weights.detach()[:, None, None] * costs.detach() * log_probabilities).sum(0).mean()
    return value + surrogate - surrogate.detach()
