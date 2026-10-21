"""Frozen, anchored Brownian teacher-readout discrepancy; never latent-state QC."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import torch

LAW = "native_transolver_teacher_readout_brownian.v1"
SPECIES_COUNT = 15
CONTROLLED_LAW = "native_controlled_teacher_readout_brownian.v1"


@dataclass(frozen=True)
class EpisodeStatistics:
    episode_uid: str
    group_uid: str
    squared_increments: torch.Tensor  # J,S; divided by retained physical gaps/Tref
    log_gap_sum: torch.Tensor         # S
    species_counts: torch.Tensor     # S
    weights: torch.Tensor             # J, original frozen prior/posterior weights

    @property
    def count(self) -> int:
        return int(self.species_counts.sum())


def elapsed_times(times, reference_time: float) -> torch.Tensor:
    values = torch.as_tensor(times, dtype=torch.float64)
    if (values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all()
            or not math.isfinite(reference_time) or reference_time <= 0
            or not torch.all(values[1:] > values[:-1])):
        raise ValueError("Finite strictly increasing physical times and positive reference required")
    return values - values[0]


def validate_fields(means, weights, target, mask, times, reference_time):
    means = torch.as_tensor(means, dtype=torch.float64).detach()
    weights = torch.as_tensor(weights, dtype=torch.float64).detach()
    target = torch.as_tensor(target, dtype=torch.float64).detach()
    mask = torch.as_tensor(mask)
    elapsed = elapsed_times(times, reference_time)
    if (means.ndim != 6 or target.shape != means.shape[1:] or mask.shape != target.shape
            or mask.dtype != torch.bool or means.shape[2] != len(elapsed)
            or weights.shape != (means.shape[0],) or not torch.isfinite(means).all()
            or not torch.isfinite(target).all() or not torch.isfinite(weights).all()
            or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.), atol=1e-6, rtol=0)):
        raise ValueError("Aligned finite J,A,T,R,C,S means, original weights and bool mask required")
    return means, weights, target, mask, elapsed


def episode_statistics(means, weights, target, mask, times, reference_time,
                       episode_uid: str, group_uid: str) -> EpisodeStatistics:
    means, weights, target, mask, elapsed = validate_fields(means, weights, target, mask, times, reference_time)
    j, a, t, r, c, s = means.shape
    residual = (target.unsqueeze(0) - means).permute(0, 2, 1, 3, 4, 5).reshape(j, t, a*r*c, s)
    observed = mask.permute(1, 0, 2, 3, 4).reshape(t, a*r*c, s)
    previous = torch.zeros_like(residual[:, 0])
    previous_time = torch.zeros_like(previous[0])
    squared, logs = torch.zeros((j, s), dtype=torch.float64), torch.zeros(s, dtype=torch.float64)
    counts = torch.zeros(s, dtype=torch.int64)
    # t0 is an atom at the original particle initial field, not a likelihood term.
    for ti in range(1, t):
        keep = observed[ti]
        gap = (elapsed[ti] - previous_time) / reference_time
        change = residual[:, ti] - previous
        squared += torch.where(keep, change.square()/gap, 0.).sum(1)
        logs += torch.where(keep, gap.log(), 0.).sum(0)
        counts += keep.sum(0)
        previous = torch.where(keep, residual[:, ti], previous)
        previous_time = torch.where(keep, elapsed[ti], previous_time)
    return EpisodeStatistics(episode_uid, group_uid, squared, logs, counts, weights)


def episode_log_likelihood(eta: torch.Tensor, row: EpisodeStatistics) -> torch.Tensor:
    if row.count == 0:
        raise ValueError("Empty episode has no future-time likelihood")
    component = -.5 * (row.squared_increments * torch.exp(-2*eta)
        + row.species_counts * (2*eta + math.log(2*math.pi)) + row.log_gap_sum).sum(-1)
    # One outer mixture for the complete episode/action set.
    return torch.logsumexp(row.weights.log() + component, dim=0)


def residual_objective(eta: torch.Tensor, rows: list[EpisodeStatistics]) -> torch.Tensor:
    groups = {}
    for row in rows:
        if row.count:
            groups.setdefault(row.group_uid, []).append(-episode_log_likelihood(eta, row)/row.count)
    if not groups:
        raise ValueError("No eligible group/episode likelihood; no artificial denominator")
    return torch.stack([torch.stack(values).mean() for values in groups.values()]).mean()


def information_counts(rows: list[EpisodeStatistics], declared_groups: list[str], *, species_count: int = SPECIES_COUNT) -> dict:
    eligible = [row for row in rows if row.count]
    by_species = (torch.stack([row.species_counts for row in eligible]).sum(0).tolist()
                  if eligible else [0]*species_count)
    return {"declared_group_count": len(declared_groups), "declared_episode_count": len(rows),
        "eligible_group_count": len({row.group_uid for row in eligible}),
        "eligible_episode_count": len(eligible), "future_coordinate_count": sum(row.count for row in eligible),
        "per_species_future_coordinate_count": by_species,
        "groups": [{"group_uid": group, "declared_episode_count": sum(row.group_uid == group for row in rows),
                    "eligible_episode_count": sum(row.group_uid == group for row in eligible)}
                   for group in declared_groups],
        "episodes": [{"episode_uid": row.episode_uid, "group_uid": row.group_uid,
                      "future_coordinate_count": row.count, "eligible": row.count > 0} for row in rows]}


class EvaluationBudgetExceeded(RuntimeError):
    pass


def fit_scales(rows: list[EpisodeStatistics], declared_groups: list[str], numerical: dict, *,
               species_count: int = SPECIES_COUNT) -> dict:
    started = time.perf_counter()
    counts = information_counts(rows, declared_groups, species_count=species_count)
    result = {"status": "FIT_FAILED", "scales": None, "counts": counts, "diagnostics": {},
              "cost": {"optimizer_closure_evaluations": 0, "fit_wall_seconds": 0.}}
    def fail(reason):
        result["reason"] = reason
        result["cost"]["fit_wall_seconds"] = time.perf_counter()-started
        return result
    if (len(counts["per_species_future_coordinate_count"]) != species_count
            or not all(counts["per_species_future_coordinate_count"])
            or counts["eligible_group_count"] != counts["declared_group_count"]):
        return fail("incomplete_group_or_species_information")
    if any(not torch.isfinite(row.squared_increments).all() for row in rows):
        return fail("nonfinite_residual_information")
    if (torch.stack([row.squared_increments.sum(0) for row in rows]).sum(0) == 0).any():
        return fail("collapsed_species_residual")
    fitted = minimize_log_scales(lambda eta: residual_objective(eta, rows),
                                 torch.zeros(species_count, dtype=torch.float64), numerical)
    result.update(fitted)
    result["cost"]["fit_wall_seconds"] = time.perf_counter()-started
    return result


def minimize_log_scales(objective, initial_eta: torch.Tensor, numerical: dict) -> dict:
    """Bounded LBFGS on the supplied likelihood; certify only local numeric rank.

    Fixed-field and adaptive-readout fits share numerical settings, not sufficient
    statistics or objectives. No positive floor rescues a collapsed direction.
    """
    started = time.perf_counter()
    result = {"status": "FIT_FAILED", "scales": None, "diagnostics": {},
              "cost": {"optimizer_closure_evaluations": 0, "fit_wall_seconds": 0.}}
    def fail(reason):
        result["reason"] = reason
        result["cost"]["fit_wall_seconds"] = time.perf_counter()-started
        return result
    if (initial_eta.ndim != 1 or not len(initial_eta) or initial_eta.dtype != torch.float64
            or not torch.isfinite(initial_eta).all()):
        return fail("invalid_log_scale_initialization")
    eta = initial_eta.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([eta], lr=1., max_iter=numerical["max_iterations"],
        max_eval=numerical["max_closure_evaluations"], tolerance_grad=numerical["gradient_tolerance"],
        tolerance_change=numerical["change_tolerance"], history_size=numerical["history_size"],
        line_search_fn=numerical["line_search"])
    def closure():
        if result["cost"]["optimizer_closure_evaluations"] >= numerical["max_closure_evaluations"]:
            raise EvaluationBudgetExceeded("optimizer_closure_budget_exhausted")
        result["cost"]["optimizer_closure_evaluations"] += 1
        optimizer.zero_grad()
        loss = objective(eta)
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError("nonfinite_or_nonscalar_objective")
        loss.backward()
        if eta.grad is None or not torch.isfinite(eta.grad).all():
            raise ValueError("nonfinite_gradient")
        return loss
    try:
        optimizer.step(closure)
        loss = objective(eta)
        gradient, = torch.autograd.grad(loss, eta)
        hessian = torch.autograd.functional.hessian(objective, eta)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        scale = eta.detach().exp()
        diagnostics = {"objective": float(loss.detach()), "gradient_max_abs": float(gradient.abs().max()),
            "eta": eta.detach().tolist(), "hessian_eigenvalues": eigenvalues.tolist(),
            "iterations": int(optimizer.state[eta]["n_iter"]),
            "identifiability_scope": "local_numerical_only_not_global_or_scientific"}
        result["diagnostics"] = diagnostics
        if not all(torch.isfinite(value).all() for value in (loss, gradient, hessian, eigenvalues, scale)) or not (scale > 0).all():
            return fail("nonfinite_or_nonpositive_optimum")
        if diagnostics["gradient_max_abs"] > numerical["gradient_tolerance"]:
            return fail("final_gradient_not_converged")
        threshold = max(numerical["hessian_absolute_threshold"],
                        numerical["hessian_relative_threshold"]*float(eigenvalues[-1]))
        diagnostics["hessian_required_minimum"] = threshold
        diagnostics["hessian_rank"] = int((eigenvalues > threshold).sum())
        diagnostics["fitted_dimension"] = len(eta)
        if float(eigenvalues[0]) <= threshold:
            return fail("local_numerical_identifiability_failed")
    except (ValueError, RuntimeError) as error:
        return fail(str(error))
    result.update(status="FIT_COMPLETE", scales=scale.tolist(), reason=None)
    result["cost"]["fit_wall_seconds"] = time.perf_counter()-started
    return result


def residual_draws(means, scales, times, reference_time, particle_indices, *,
                   episode_uid: str, residual_seed: int, draw_ids: list[int],
                   law: str = LAW) -> torch.Tensor:
    """Complete paths per absolute draw ID; action/time chunking happens after this call."""
    means = torch.as_tensor(means, dtype=torch.float64)
    scales = torch.as_tensor(scales, dtype=torch.float64)
    elapsed = elapsed_times(times, reference_time)
    indices = torch.as_tensor(particle_indices)
    if (means.ndim != 6 or means.shape[2] != len(elapsed) or scales.shape != means.shape[-1:]
            or not torch.isfinite(means).all() or not torch.isfinite(scales).all() or not (scales > 0).all()
            or indices.shape != (len(draw_ids),) or indices.dtype != torch.int64
            or (indices < 0).any() or (indices >= len(means)).any()
            or len(set(draw_ids)) != len(draw_ids) or any(d < 0 for d in draw_ids)):
        raise ValueError("Complete finite mean paths and explicit particle/draw identities required")
    if law not in {LAW, CONTROLLED_LAW}:
        raise ValueError("Unknown declared residual law")
    draws = []
    for index, draw_id in zip(indices.tolist(), draw_ids):
        key = f"{law}|{residual_seed}|{episode_uid}|{draw_id}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63-1)
        generator = torch.Generator().manual_seed(seed)
        residual = torch.zeros_like(means[index])
        # Draw one complete spatial/action increment at each time; horizon extension
        # preserves this conditional residual prefix, not an unverified mean prefix.
        for ti in range(1, len(elapsed)):
            noise = torch.randn(means.shape[1:2]+means.shape[3:], generator=generator, dtype=torch.float64)
            residual[:, ti] = residual[:, ti-1] + noise*scales*((elapsed[ti]-elapsed[ti-1])/reference_time).sqrt()
        draws.append(means[index] + residual)
    return torch.stack(draws)


def check_diagnostics(means, weights, target, mask, times, reference_time, scales,
                      adjacency, species_ids: list[str]) -> dict:
    means, weights, target, mask, elapsed = validate_fields(means, weights, target, mask, times, reference_time)
    scale = torch.as_tensor(scales, dtype=torch.float64)
    j, a, t, r, c, s = means.shape
    mu = means.reshape(j, a, t, r*c, s)
    y, valid = target.reshape(a, t, r*c, s), mask.reshape(a, t, r*c, s)
    average = (weights[:, None, None, None, None]*mu).sum(0)
    centered_mu = mu-average
    centered_y = y-average
    sd = scale[None, None, None, :]*(elapsed[None, 1:, None, None]/reference_time).sqrt()
    cdf = .5*(1+torch.erf((y[None, :, 1:]-mu[:, :, 1:])/(sd*math.sqrt(2))))
    pit = (weights[:, None, None, None, None]*cdf).sum(0)
    def summary(values):
        return {"count": values.numel(), "mean": float(values.mean()) if values.numel() else None}
    pit_rows = [{"species_id": species_ids[si], **summary(pit[..., si][valid[:, 1:, :, si]])} for si in range(s)]
    strata = {"adjacent_retained_times": [], "graph_edges": [], "cross_species_pairs": []}
    def add(kind, label, left, right, keep, kernel):
        lx, rx = centered_y[left], centered_y[right]
        lm, rm = centered_mu[(slice(None), *left)], centered_mu[(slice(None), *right)]
        expected = (weights.reshape(j, *([1]*(lm.ndim-1)))*lm*rm).sum(0)+kernel
        products, covariance = (lx*rx)[keep], expected[keep]
        strata[kind].append(dict(label, count=products.numel(),
            status="observed" if products.numel() else "missing",
            centered_product_mean=float(products.mean()) if products.numel() else None,
            total_covariance_mean=float(covariance.mean()) if products.numel() else None))
    # Pair adjacent observed future times per block, marginalizing missing times.
    for si in range(s):
        for ti in range(1, t-1):
            for ui in range(ti+1, t):
                keep = valid[:, ti, :, si] & valid[:, ui, :, si]
                if ui > ti+1:
                    keep = keep & ~valid[:, ti+1:ui, :, si].any(1)
                add("adjacent_retained_times", {"species_id": species_ids[si], "time_indices": [ti, ui]},
                    (slice(None), ti, slice(None), si), (slice(None), ui, slice(None), si),
                    keep, scale[si].square()*elapsed[ti]/reference_time)
    graph = torch.as_tensor(adjacency)
    if graph.shape != (r*c, r*c):
        raise ValueError("Native graph adjacency/domain mismatch")
    for ni, nj in torch.nonzero(torch.triu(graph != 0, diagonal=1)).tolist():
        add("graph_edges", {"node_indices": [ni, nj]},
            (slice(None), slice(1,None), ni, slice(None)), (slice(None), slice(1,None), nj, slice(None)),
            valid[:, 1:, ni] & valid[:, 1:, nj], 0.)
    for si in range(s):
        for sj in range(si+1, s):
            add("cross_species_pairs", {"species_ids": [species_ids[si], species_ids[sj]]},
                (slice(None), slice(1,None), slice(None), si), (slice(None), slice(1,None), slice(None), sj),
                valid[:, 1:, :, si] & valid[:, 1:, :, sj], 0.)
    return {"pit": {"future": summary(pit[valid[:, 1:]]), "by_species": pit_rows,
                    "t0_continuous_pit_count": 0, "kind": "exact_marginal_mixture_cdf"},
            "covariance": strata, "scientific_check_pass": None,
            "interpretation": "unbounded_teacher_readout_approximation_not_physical_state_certification"}
