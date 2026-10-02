"""Decision expectations from token marginals, without assuming independence.

Intervals concern all couplings of the supplied marginals. Only an explicit
external marginal-error allowance extends them to a reference distribution.
"""

from __future__ import annotations

import math

import torch


def _validate_risk_inputs(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    synthesis: torch.Tensor | None,
    reference: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    *,
    lower_bound: float | None,
) -> None:
    """Validate the shared finite-marginal/field contract without densifying a codec."""
    items = {
        "probabilities": probabilities,
        "codebook": codebook,
        "reference": reference,
        "weights": weights,
        "offset": offset,
    }
    if synthesis is not None:
        items["synthesis"] = synthesis
    for name, value in items.items():
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating-point values.")
        if value.device != probabilities.device or value.dtype != probabilities.dtype:
            raise ValueError("All risk inputs must have a common dtype and device.")
    if (
        probabilities.ndim < 2
        or codebook.ndim != 1
        or reference.ndim < 1
        or (synthesis is not None and synthesis.ndim != 2)
    ):
        raise ValueError(
            "Expected categorical probabilities, a codebook, and a synthesis matrix."
        )
    prefix, m, k = (
        probabilities.shape[:-2],
        probabilities.shape[-2],
        probabilities.shape[-1],
    )
    n = reference.shape[-1] if synthesis is None else synthesis.shape[0]
    if (
        min(n, m, k) < 1
        or codebook.shape != (k,)
        or (synthesis is not None and synthesis.shape[1] != m)
    ):
        raise ValueError("Codebook/coefficient/synthesis dimensions disagree.")
    if (
        reference.shape != (*prefix, n)
        or weights.shape != reference.shape
        or offset.shape != prefix
    ):
        raise ValueError("Reference, objective, and probability batch shapes disagree.")
    if (probabilities < 0).any() or not torch.allclose(
        probabilities.sum(-1),
        torch.ones_like(probabilities.sum(-1)),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "Categorical probabilities must be normalized and nonnegative."
        )
    if k > 1 and not (codebook.diff() > 0).all():
        raise ValueError("The codebook must be strictly increasing.")
    if lower_bound is not None and not math.isfinite(lower_bound):
        raise ValueError("The clipping threshold must be finite.")


def clipped_affine_marginal_bounds(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    synthesis: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    *,
    lower_bound: float | None,
    marginal_tv_bounds: torch.Tensor | None = None,
    reference_loss_error: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | str | None]:
    """Bound E[offset + weights dot max(lower, reference + S z)].

    Shapes are probabilities [...,M,K], S [N,M], reference/weights [...,N],
    offset [...]. Reference allowances cover errors outside the finite-codebook
    law and are never inferred from confidence.
    """
    _validate_risk_inputs(
        probabilities,
        codebook,
        synthesis,
        reference,
        weights,
        offset,
        lower_bound=lower_bound,
    )
    mean = (probabilities * codebook).sum(-1)
    variance = (probabilities * (codebook - mean[..., None]).square()).sum(-1)
    physical_mean = reference + torch.einsum("nm,...m->...n", synthesis, mean)
    risk = offset + (weights * physical_mean).sum(-1)
    adjoint_weights = torch.einsum("nm,...n->...m", synthesis, weights)
    if lower_bound is None:
        lower = upper = risk
        sensitivity = adjoint_weights.abs()
    else:
        positive = synthesis.clamp_min(0)
        negative = synthesis.clamp_max(0)
        support_lower = reference + (
            positive * codebook[0] + negative * codebook[-1]
        ).sum(-1)
        support_upper = reference + (
            positive * codebook[-1] + negative * codebook[0]
        ).sum(-1)
        variance_bound = torch.einsum(
            "nm,...m->...n", synthesis.abs(), variance.sqrt()
        ).square()
        distance = lower_bound - physical_mean
        hinge_lower = distance.clamp_min(0)
        span = support_upper - support_lower
        safe_span = torch.where(span > 0, span, torch.ones_like(span))
        support_chord = (
            (support_upper - physical_mean) * (lower_bound - support_lower).clamp_min(0)
            + (physical_mean - support_lower)
            * (lower_bound - support_upper).clamp_min(0)
        ) / safe_span
        support_chord = torch.where(
            span > 0, support_chord, (lower_bound - support_lower).clamp_min(0)
        )
        hinge_upper = torch.minimum(
            support_chord,
            (torch.sqrt(distance.square() + variance_bound) + distance) / 2,
        )
        # The two sign-definite regions have exact clipping expectations.
        hinge_upper = torch.where(
            support_lower >= lower_bound, torch.zeros_like(hinge_upper), hinge_upper
        )
        hinge_upper = torch.where(support_upper <= lower_bound, distance, hinge_upper)
        hinge_lower = torch.where(support_upper <= lower_bound, distance, hinge_lower)
        if (hinge_upper < hinge_lower - 1e-6).any():
            raise ArithmeticError(
                "Inconsistent clipping bounds; inspect probability precision."
            )
        lower = risk + torch.where(
            weights >= 0, weights * hinge_lower, weights * hinge_upper
        ).sum(-1)
        upper = risk + torch.where(
            weights >= 0, weights * hinge_upper, weights * hinge_lower
        ).sum(-1)
        sensitivity = torch.einsum("nm,...n->...m", synthesis.abs(), weights.abs())

    result: dict[str, torch.Tensor | str | None] = {
        "unclipped_marginal_risk": risk,
        "coupling_lower": lower,
        "coupling_upper": upper,
        "reference_lower": None,
        "reference_upper": None,
        "marginal_error_allowance": None,
        "status": "supplied_marginal_coupling_bounds",
    }
    return _extend_reference_bounds(
        result,
        probabilities,
        codebook,
        sensitivity,
        marginal_tv_bounds,
        reference_loss_error,
    )


def _extend_reference_bounds(
    result,
    probabilities,
    codebook,
    sensitivity,
    marginal_tv_bounds,
    reference_loss_error,
):
    """Apply supplied prediction/reference allowances to either coupling engine."""
    prefix, m = probabilities.shape[:-2], probabilities.shape[-2]
    if (marginal_tv_bounds is None) != (reference_loss_error is None):
        raise ValueError(
            "Supply both marginal TV bounds and the external reference-loss allowance."
        )
    if marginal_tv_bounds is not None:
        for name, value, shape in (
            ("marginal_tv_bounds", marginal_tv_bounds, (*prefix, m)),
            ("reference_loss_error", reference_loss_error, prefix),
        ):
            if (
                value.shape != shape
                or value.dtype != probabilities.dtype
                or value.device != probabilities.device
            ):
                raise ValueError(f"{name} shape/dtype/device mismatch.")
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative.")
        if (marginal_tv_bounds > 1).any():
            raise ValueError("Marginal TV bounds must lie in [0,1].")
        allowance = (codebook[-1] - codebook[0]) * (
            sensitivity * marginal_tv_bounds
        ).sum(-1)
        result.update(
            reference_lower=result["coupling_lower"] - allowance - reference_loss_error,
            reference_upper=result["coupling_upper"] + allowance + reference_loss_error,
            marginal_error_allowance=allowance,
            status="conditional_on_supplied_error_bounds",
        )
    return result


def temporal_synthesis_matrix(
    codec, time_count: int, template: torch.Tensor
) -> torch.Tensor:
    """Extract only the small T-by-L temporal map from the actual crop-aware codec."""
    if time_count < 1:
        raise ValueError("A nonempty temporal grid is required.")
    packed, metadata = codec.pack_temporal_wavelets(
        template.new_zeros((1, time_count, 1, 1))
    )
    length = packed.shape[1]
    basis = torch.eye(length, dtype=template.dtype, device=template.device).reshape(
        length, length, 1, 1
    )
    return (
        codec.unpack_temporal_wavelets(basis, metadata)[:, :, 0, 0]
        .transpose(0, 1)
        .contiguous()
    )


@torch.no_grad()
def _wavelet_goal_predictions(
    sampler,
    *,
    initial_state: torch.Tensor,
    occupancy_program: torch.Tensor,
    occupancy_support: torch.Tensor,
    posterior_context: torch.Tensor,
    time_hours: torch.Tensor,
    node_coordinates: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    backbone,
    goal_weights: torch.Tensor,
    goal_offset: torch.Tensor,
) -> tuple:
    """Shared all-mask prediction and physical-goal mapping for both evaluators."""
    predicted = sampler.predict_clean_marginals(
        initial_state=initial_state,
        occupancy_program=occupancy_program,
        occupancy_support=occupancy_support,
        posterior_context=posterior_context,
        time_hours=time_hours,
        node_coordinates=node_coordinates,
        adjacency_matrix=adjacency_matrix,
        backbone=backbone,
    )
    probabilities = predicted["probabilities"].to(torch.float64)
    b, a, length, r, c, species, _ = probabilities.shape
    t = time_hours.shape[-1]
    if goal_weights.shape != (b, a, t, r, c, species) or goal_offset.shape != (b, a):
        raise ValueError(
            "Goal weights must match the physical field and offsets the candidate batch."
        )
    if not goal_weights.is_floating_point() or not goal_offset.is_floating_point():
        raise ValueError("Goal parameters must be floating-point tensors.")
    if not torch.isfinite(goal_weights).all() or not torch.isfinite(goal_offset).all():
        raise ValueError("Goal parameters must be finite.")
    weights = goal_weights.to(device=probabilities.device, dtype=torch.float64)
    reference = initial_state.to(torch.float64)[:, :, None].expand(
        b, a, t, r, c, species
    )
    fixed = time_hours.eq(0)[:, :, :, None, None, None] & sampler.preserve_initial_state
    constant = (weights * reference * fixed).sum(dim=(2, 3, 4, 5)) + goal_offset.to(
        probabilities
    )
    weights = torch.where(fixed, torch.zeros_like(weights), weights)
    _, metadata = backbone.tokenizer.pack_temporal_wavelets(
        probabilities.new_zeros((1, t, 1, 1))
    )
    return (
        probabilities.permute(0, 1, 3, 4, 5, 2, 6),
        sampler.codebook_values.to(probabilities),
        metadata,
        reference.permute(0, 1, 3, 4, 5, 2),
        weights.permute(0, 1, 3, 4, 5, 2),
        constant,
    )


@torch.no_grad()
def evaluate_wavelet_marginal_goal(sampler, **goal_inputs) -> dict:
    """Evaluate a declared field goal; never substitute it for a nonlinear head."""
    probabilities, codebook, synthesis, reference, weights, constant = (
        _wavelet_goal_predictions(sampler, **goal_inputs)
    )
    synthesis = temporal_synthesis_matrix(
        goal_inputs["backbone"].tokenizer, synthesis["time_count"], probabilities
    )
    bounds = clipped_affine_marginal_bounds(
        probabilities,
        codebook,
        synthesis,
        reference,
        weights,
        probabilities.new_zeros(probabilities.shape[:-2]),
        lower_bound=sampler.state_lower_bound,
    )
    b, a = constant.shape
    return {
        "unclipped_marginal_risk": constant
        + bounds["unclipped_marginal_risk"].sum((-1, -2, -3)),
        "coupling_lower": constant + bounds["coupling_lower"].sum((-1, -2, -3)),
        "coupling_upper": constant + bounds["coupling_upper"].sum((-1, -2, -3)),
        "reference_lower": None,
        "reference_upper": None,
        "status": "uncalibrated_marginal_goal_bounds",
        "denoiser_evaluations": b * a,
        "scope": "all_mask_clean_predictions_not_sequential_sampler_marginals",
    }
