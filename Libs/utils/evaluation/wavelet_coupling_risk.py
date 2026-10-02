"""Exact amplitude reduction and an ancestral-bag comparator for Haar risk.

Both LPs range over coefficient couplings with supplied singleton marginals.
The amplitude quotient compresses histories using uniform-codebook arithmetic;
its common-radius envelopes account for binning and codebook storage deviation.
This class need not equal laws realizable by a learned reverse sampler.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from Libs.utils.evaluation.exact_transform_state import (
    ExactTransformDescriptor,
    ExactTransformEdge,
    _assemble_exact_transform_problem,
    compile_exact_transform_topology,
)
from Libs.utils.evaluation.marginal_intervention_risk import (
    _extend_reference_bounds,
    _validate_risk_inputs,
    _wavelet_goal_predictions,
    clipped_affine_marginal_bounds,
)


def _bags(synthesis: np.ndarray, weights: np.ndarray):
    bags, rows = [], []
    seen, previous = set(), set()
    for i in np.flatnonzero(weights):
        scope = tuple(np.flatnonzero(synthesis[i]).tolist())
        if not scope:
            continue
        current = set(scope)
        if (current - previous) & seen:
            raise ValueError(
                "Synthesis rows violate interval support/running intersection."
            )
        seen |= current
        previous = current
        if bags and scope == bags[-1]:
            rows[-1].append(int(i))
        else:
            bags.append(scope)
            rows.append([int(i)])
    return bags, rows


def _transport_problem(
    probabilities,
    codebook,
    synthesis,
    reference,
    weights,
    lower_bound,
    partition,
    max_variables,
):
    bags, rows = _bags(synthesis, weights)
    bin_count = len(partition) - 1
    sizes = [bin_count ** len(bag) for bag in bags]
    variable_count = sum(sizes)
    if variable_count > max_variables:
        raise ValueError(
            "Transport variable budget exceeded; use a coarser codebook partition."
        )
    starts = np.cumsum([0, *sizes])
    slices = [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(bags))]
    assignments = [
        np.asarray(
            list(itertools.product(range(bin_count), repeat=len(bag))), dtype=np.int64
        )
        for bag in bags
    ]
    low_values = codebook[np.asarray(partition[:-1])]
    high_values = codebook[np.asarray(partition[1:]) - 1]
    masses = np.stack(
        [
            probabilities[:, left:right].sum(axis=-1)
            for left, right in zip(partition[:-1], partition[1:], strict=True)
        ],
        axis=-1,
    )
    cost_lower, cost_upper = np.zeros(variable_count), np.zeros(variable_count)
    matrix_rows, matrix_cols, matrix_values, rhs = [], [], [], []

    def constraint(columns, values, value):
        matrix_rows.extend([len(rhs)] * len(columns))
        matrix_cols.extend(columns)
        matrix_values.extend(values)
        rhs.append(value)

    assigned = set()
    for index, (bag, row_ids, assignment, sl) in enumerate(
        zip(bags, rows, assignments, slices, strict=True)
    ):
        cols = np.arange(sl.start, sl.stop)
        constraint(cols, np.ones(len(cols)), 1.0)
        for position, coefficient in enumerate(bag):
            if coefficient not in assigned:
                for value in range(bin_count):
                    selected = cols[assignment[:, position] == value]
                    constraint(
                        selected, np.ones(len(selected)), masses[coefficient, value]
                    )
                assigned.add(coefficient)
        lo, hi = low_values[assignment], high_values[assignment]
        for row in row_ids:
            coefficients = synthesis[row, list(bag)]
            field_lo = reference[row] + np.minimum(
                lo * coefficients, hi * coefficients
            ).sum(-1)
            field_hi = reference[row] + np.maximum(
                lo * coefficients, hi * coefficients
            ).sum(-1)
            a = weights[row] * np.maximum(lower_bound, field_lo)
            b = weights[row] * np.maximum(lower_bound, field_hi)
            cost_lower[sl] += np.minimum(a, b)
            cost_upper[sl] += np.maximum(a, b)
        if index:
            separator = tuple(sorted(set(bags[index - 1]) & set(bag)))
            if separator:
                left = assignments[index - 1]
                left_positions = [bags[index - 1].index(j) for j in separator]
                right_positions = [bag.index(j) for j in separator]
                powers = bin_count ** np.arange(len(separator) - 1, -1, -1)
                left_ids = left[:, left_positions] @ powers
                right_ids = assignment[:, right_positions] @ powers
                left_cols = np.arange(slices[index - 1].start, slices[index - 1].stop)
                for value in range(bin_count ** len(separator)):
                    lcols, rcols = (
                        left_cols[left_ids == value],
                        cols[right_ids == value],
                    )
                    constraint(
                        np.concatenate([lcols, rcols]),
                        np.concatenate([np.ones(len(lcols)), -np.ones(len(rcols))]),
                        0.0,
                    )
    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_cols)), shape=(len(rhs), variable_count)
    ).tocsr()
    return matrix, np.asarray(rhs), cost_lower, cost_upper, slices, bags


def _dual_lower_bound(
    cost,
    matrix,
    rhs,
    slices,
    *,
    inequality_matrix=None,
    inequality_rhs=None,
    scalar_radius=None,
):
    """Outward dual bound for blocks of mass at most one and a bounded scalar."""
    bounds = (0, None)
    if scalar_radius is not None:
        if not math.isfinite(scalar_radius) or scalar_radius <= 0:
            raise ValueError("The auxiliary scalar needs a finite positive radius.")
        bounds = [(0, None)] * (len(cost) - 1) + [(-scalar_radius, scalar_radius)]
    result = linprog(
        cost,
        A_eq=matrix,
        b_eq=rhs,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs,
        bounds=bounds,
        method="highs",
        # Solve more tightly than the independently checked 1e-8 residual gate.
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise ArithmeticError(f"Coupling LP did not converge: {result.message}")
    dual = np.asarray(result.eqlin.marginals)
    if (
        not np.isfinite(result.x).all()
        or not np.isfinite(dual).all()
        or not np.isfinite(result.fun)
    ):
        raise ArithmeticError("Coupling LP returned nonfinite primal/dual values.")
    primal_residual = max(
        float(np.abs(matrix @ result.x - rhs).max(initial=0)),
        float(
            np.maximum(-result.x if scalar_radius is None else -result.x[:-1], 0).max(
                initial=0
            )
        ),
    )
    reduced = cost - matrix.T @ dual
    dual_value = float(rhs @ dual)
    inequality_scale = 0.0
    if inequality_matrix is not None:
        inequality_dual = np.minimum(np.asarray(result.ineqlin.marginals), 0.0)
        if not np.isfinite(inequality_dual).all():
            raise ArithmeticError("Coupling LP returned nonfinite inequality duals.")
        reduced -= inequality_matrix.T @ inequality_dual
        dual_value += float(inequality_rhs @ inequality_dual)
        inequality_scale = float(np.abs(inequality_rhs * inequality_dual).sum())
        primal_residual = max(
            primal_residual,
            float(
                np.maximum(inequality_matrix @ result.x - inequality_rhs, 0).max(
                    initial=0
                )
            ),
        )
    # Each nonnegative block has mass at most one. Correct negative residuals,
    # rather than treating a minimization primal value as a lower certificate.
    repair = sum(min(0.0, float(reduced[sl].min())) for sl in slices)
    if scalar_radius is not None:
        repair -= scalar_radius * abs(float(reduced[-1]))
        primal_residual = max(primal_residual, abs(float(result.x[-1])) - scalar_radius)
    scale = 1.0 + float(np.abs(rhs * dual).sum()) + inequality_scale
    scale += sum(float(np.abs(cost[sl]).max()) for sl in slices)
    if scalar_radius is not None:
        scale += scalar_radius * abs(float(cost[-1]))
    padding = 64 * np.finfo(np.float64).eps * scale
    bound = float(np.nextafter(dual_value + repair - padding, -np.inf))
    gap = float(result.fun - bound)
    if primal_residual > 1e-8 or gap < -1e-8 * scale or gap > 1e-7 * scale:
        raise ArithmeticError(
            "Coupling LP failed primal/dual numerical verification: "
            f"residual={primal_residual:.17g}, gap={gap:.17g}, scale={scale:.17g}."
        )
    return bound, primal_residual, max(gap, 0.0)


@torch.no_grad()
def clipped_wavelet_coupling_bounds(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    synthesis: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    *,
    lower_bound: float | None,
    partition: tuple[int, ...],
    max_variables: int,
    marginal_tv_bounds: torch.Tensor | None = None,
    reference_loss_error: torch.Tensor | None = None,
) -> dict:
    """Bound clipped field risk using local tables with full separator consistency.

    Tensor shapes match ``clipped_affine_marginal_bounds``. This CPU float64
    evaluator requires normalized probabilities. ``partition`` contains bin
    edges as codebook indices, starting at 0 and ending at K. Singleton bins
    recover the finite-product sharp interval, up to recorded numerical slack.
    The explicit table budget is checked before Cartesian tables are allocated.
    No temporal Markov or coefficient-independence assumption is made.
    """
    if probabilities.device.type != "cpu" or probabilities.dtype != torch.float64:
        raise ValueError("Coupling optimization requires CPU float64 inputs.")
    bounds = clipped_affine_marginal_bounds(
        probabilities,
        codebook,
        synthesis,
        reference,
        weights,
        offset,
        lower_bound=lower_bound,
        marginal_tv_bounds=marginal_tv_bounds,
        reference_loss_error=reference_loss_error,
    )
    if not torch.allclose(
        probabilities.sum(-1),
        torch.ones_like(probabilities.sum(-1)),
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("Coupling probabilities must sum to one at float64 precision.")
    if (
        not isinstance(max_variables, int)
        or isinstance(max_variables, bool)
        or max_variables < 1
    ):
        raise ValueError("max_variables must be a positive integer.")
    if (
        not isinstance(partition, tuple)
        or len(partition) < 2
        or any(type(edge) is not int for edge in partition)
        or partition[0] != 0
        or partition[-1] != codebook.numel()
        or any(b <= a for a, b in zip(partition[:-1], partition[1:], strict=True))
    ):
        raise ValueError("partition must be strictly increasing codebook index edges.")
    lower, upper = bounds["coupling_lower"].clone(), bounds["coupling_upper"].clone()
    variables = torch.zeros_like(offset, dtype=torch.int64)
    widths = torch.zeros_like(variables)
    residuals, gaps = torch.zeros_like(offset), torch.zeros_like(offset)
    if lower_bound is not None:
        arrays = [
            value.detach().numpy()
            for value in (probabilities, codebook, synthesis, reference, weights)
        ]
        p, code, matrix_s, r, w = arrays
        for index in np.ndindex(tuple(offset.shape)):
            if not np.any(w[index]):
                continue
            # Rows with no random coefficient are deterministic, including
            # restored observations; all remaining rows keep their actual scope.
            fixed = ~np.any(matrix_s != 0, axis=1)
            constant = float(offset[index]) + float(
                (w[index][fixed] * np.maximum(lower_bound, r[index][fixed])).sum()
            )
            problem = _transport_problem(
                p[index],
                code,
                matrix_s,
                r[index],
                w[index],
                lower_bound,
                partition,
                max_variables,
            )
            matrix, rhs, lo_cost, hi_cost, slices, bags = problem
            if not slices:
                continue
            lo, lo_res, lo_gap = _dual_lower_bound(lo_cost, matrix, rhs, slices)
            neg_hi, hi_res, hi_gap = _dual_lower_bound(-hi_cost, matrix, rhs, slices)
            lower[index] = max(float(lower[index]), constant + lo)
            upper[index] = min(float(upper[index]), constant - neg_hi)
            variables[index] = len(lo_cost)
            widths[index] = max(map(len, bags))
            residuals[index], gaps[index] = max(lo_res, hi_res), max(lo_gap, hi_gap)
    if (lower > upper).any():
        raise ArithmeticError(
            "Moment and transport intervals have an empty intersection."
        )
    bounds.update(
        coupling_lower=lower,
        coupling_upper=upper,
        transport_variables=variables,
        path_width=widths,
        primal_residual=residuals,
        numerical_duality_gap=gaps,
        partition=partition,
        resolution=(
            "singleton_exact_transport"
            if len(partition) == codebook.numel() + 1
            else "coarsened_outer_transport"
        ),
        numerical_status="floating_point_primal_dual_checked_not_interval_arithmetic",
    )
    if marginal_tv_bounds is not None:
        allowance = bounds["marginal_error_allowance"] + reference_loss_error
        bounds.update(
            reference_lower=lower - allowance, reference_upper=upper + allowance
        )
    return bounds


def _haar_metadata(metadata, time_count, coefficient_count):
    lengths = tuple(metadata["lengths"])
    current = time_count
    for length in lengths:
        if type(length) is not int or length != current or length < 2:
            raise ValueError("Invalid Haar crop metadata.")
        current = (current + 1) // 2
    sizes = (current, *reversed(tuple((length + 1) // 2 for length in lengths)))
    if (
        metadata["time_count"] != time_count
        or tuple(metadata["component_lengths"]) != sizes
        or sum(sizes) != coefficient_count
    ):
        raise ValueError("Haar metadata does not match the coefficient/field shapes.")
    return lengths, sizes


def _haar_pullback(weights, lengths, *, absolute=False):
    """Actual crop transpose: missing odd children receive zero, not replication."""
    approximation, details = weights, []
    for length in lengths:
        if length % 2:
            approximation = np.append(approximation, 0.0)
        even, odd = approximation[0::2], approximation[1::2]
        details.append((even + odd if absolute else even - odd) / math.sqrt(2))
        approximation = (even + odd) / math.sqrt(2)
    return np.concatenate([approximation, *reversed(details)])


@dataclass(frozen=True)
class HaarAmplitudeTable:
    depth: int
    position: int
    columns: slice
    amplitudes: np.ndarray
    details: np.ndarray


@dataclass(frozen=True)
class HaarCouplingProblem:
    matrix: csr_matrix
    rhs: np.ndarray
    lower_cost: np.ndarray
    upper_cost: np.ndarray
    slices: tuple[slice, ...]
    tables: tuple[HaarAmplitudeTable, ...]


def _haar_transform_descriptor(lengths):
    multiplication = ((0, 2), (1, 0))
    left = ExactTransformEdge(multiplication, (0, 1))
    right = ExactTransformEdge(multiplication, (0, -1))
    return ExactTransformDescriptor(
        q0=2,
        radicand=2,
        edge_maps=tuple(
            tuple(left if child % 2 == 0 else right for child in range(child_count))
            for child_count in reversed(lengths)
        ),
    )


def _haar_state_problem(
    masses,
    lower,
    step,
    radius,
    lengths,
    sizes,
    reference,
    weights,
    floor,
    max_variables,
):
    """Compile Haar transport through the certified exact-transform backend."""
    topology = compile_exact_transform_topology(
        _haar_transform_descriptor(lengths),
        tuple(lengths),
        tuple(sizes),
        np.asarray(weights != 0, dtype=np.bool_),
        masses.shape[-1],
        max_variables=max_variables,
    )
    problem = _assemble_exact_transform_problem(
        topology,
        masses,
        lower,
        step,
        radius,
        reference,
        weights,
        floor,
        allow_certified_radius=True,
    )
    return HaarCouplingProblem(
        problem.matrix,
        problem.rhs,
        problem.lower_cost,
        problem.upper_cost,
        problem.slices,
        tuple(
            HaarAmplitudeTable(
                table.depth,
                table.position,
                table.columns,
                table.amplitudes,
                table.details,
            )
            for table in problem.tables
        ),
    )


@torch.no_grad()
def clipped_haar_coupling_bounds(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    metadata: dict,
    reference: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    *,
    lower_bound: float | None,
    bin_count: int,
    max_variables: int,
    marginal_tv_bounds: torch.Tensor | None = None,
    reference_loss_error: torch.Tensor | None = None,
) -> dict:
    """Exact amplitude-state transport for an affine uniform Haar codebook.

    Equal-size bins use exact marginal masses and outward leaf envelopes.
    Deviations of the supplied codebook from its affine grid are included in
    the reported radius. No dense synthesis matrix or identity is constructed.
    The variable budget covers the sum of all requested nonzero-loss streams;
    inputs, sparse solver workspace and denoiser storage are additional memory.
    """
    _validate_risk_inputs(
        probabilities,
        codebook,
        None,
        reference,
        weights,
        offset,
        lower_bound=lower_bound,
    )
    if probabilities.device.type != "cpu" or probabilities.dtype != torch.float64:
        raise ValueError("Coupling optimization requires CPU float64 inputs.")
    if not torch.allclose(
        probabilities.sum(-1),
        torch.ones_like(probabilities.sum(-1)),
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("Coupling probabilities must sum to one at float64 precision.")
    k = codebook.numel()
    if type(bin_count) is not int or bin_count < 1 or k % bin_count:
        raise ValueError("bin_count must be a positive divisor of the codebook size.")
    if type(max_variables) is not int or max_variables < 1:
        raise ValueError("max_variables must be a positive integer.")
    lengths, sizes = _haar_metadata(
        metadata, reference.shape[-1], probabilities.shape[-2]
    )
    code = codebook.detach().numpy()
    step = (code[-1] - code[0]) / (k - 1) if k > 1 else 0.0
    block_size = k // bin_count
    center_lower = code[0] + step * (block_size - 1) / 2
    center_step = step * block_size
    grid_error = float(np.abs(code - (code[0] + step * np.arange(k))).max())
    radius = float(step * (block_size - 1) / 2 + grid_error)
    p = probabilities.detach().numpy()
    masses = p.reshape(*p.shape[:-1], bin_count, block_size).sum(-1)
    lo, hi, affine = (
        torch.zeros_like(offset),
        torch.zeros_like(offset),
        torch.zeros_like(offset),
    )
    variables = torch.zeros_like(offset, dtype=torch.int64)
    residuals, gaps = torch.zeros_like(offset), torch.zeros_like(offset)
    sensitivity = torch.zeros_like(probabilities[..., 0])
    used = 0
    for index in np.ndindex(tuple(offset.shape)):
        r, w = reference[index].detach().numpy(), weights[index].detach().numpy()
        adjoint = _haar_pullback(w, lengths)
        sensitivity[index] = torch.from_numpy(
            np.abs(adjoint)
            if lower_bound is None
            else _haar_pullback(np.abs(w), lengths, absolute=True)
        )
        affine[index] = float(offset[index]) + w @ r + adjoint @ (p[index] @ code)
        if lower_bound is None or not np.any(w):
            lo[index] = hi[index] = affine[index]
            continue
        if not lengths:
            exact = float(offset[index]) + np.sum(
                w * (p[index] * np.maximum(lower_bound, r[:, None] + code)).sum(-1)
            )
            lo[index] = hi[index] = exact
            continue
        problem = _haar_state_problem(
            masses[index],
            center_lower,
            center_step,
            radius,
            lengths,
            sizes,
            r,
            w,
            lower_bound,
            max_variables - used,
        )
        lower, lr, lg = _dual_lower_bound(
            problem.lower_cost, problem.matrix, problem.rhs, problem.slices
        )
        neg_upper, ur, ug = _dual_lower_bound(
            -problem.upper_cost, problem.matrix, problem.rhs, problem.slices
        )
        lo[index], hi[index] = (
            float(offset[index]) + lower,
            float(offset[index]) - neg_upper,
        )
        variables[index] = len(problem.lower_cost)
        residuals[index], gaps[index] = max(lr, ur), max(lg, ug)
        used += len(problem.lower_cost)
    if (
        not all(torch.isfinite(value).all() for value in (lo, hi, affine))
        or (lo > hi).any()
    ):
        raise ArithmeticError("Nonfinite or inconsistent Haar risk endpoints.")
    result = {
        "unclipped_marginal_risk": affine,
        "coupling_lower": lo,
        "coupling_upper": hi,
        "reference_lower": None,
        "reference_upper": None,
        "marginal_error_allowance": None,
        "status": "supplied_marginal_coupling_bounds",
        "transport_variables": variables,
        "primal_residual": residuals,
        "numerical_duality_gap": gaps,
        "executed_haar_levels": len(lengths),
        "bin_count": bin_count,
        "coefficient_bin_radius": radius,
        "affine_codebook_max_deviation": grid_error,
        "resolution": "affine_grid_transport"
        if bin_count == k
        else "coarsened_outer_transport",
        "numerical_status": "floating_point_primal_dual_checked_not_interval_arithmetic",
        "budget_scope": "total_table_entries_across_requested_streams",
    }
    return _extend_reference_bounds(
        result,
        probabilities,
        codebook,
        sensitivity,
        marginal_tv_bounds,
        reference_loss_error,
    )


def _mixing_error_envelope(codebook, metadata, time_count, mixing_by_level):
    """Bound a real-arithmetic decoder perturbation without enumerating histories."""
    lengths, sizes = _haar_metadata(
        metadata, time_count, sum(metadata["component_lengths"])
    )
    if not isinstance(mixing_by_level, (tuple, list)) or len(mixing_by_level) != len(
        lengths
    ):
        raise ValueError("Supply one mixing tensor per executed inverse Haar level.")
    root_bound = float(codebook.abs().max())
    amplitude_bound = np.full(sizes[0], root_bound)
    error = np.zeros(sizes[0])
    inverse_sqrt_two = 1 / math.sqrt(2)
    nominal = np.array([[1, 1], [1, -1]]) * inverse_sqrt_two
    deviation = 0.0
    for depth, mixing in enumerate(mixing_by_level):
        if (
            not isinstance(mixing, torch.Tensor)
            or mixing.device.type != "cpu"
            or mixing.dtype != torch.float64
            or tuple(mixing.shape) != (sizes[depth + 1], 2, 2)
            or not torch.isfinite(mixing).all()
        ):
            raise ValueError(
                "Mixing tensors must be finite CPU float64 [nodes, 2, 2] arrays."
            )
        matrix = mixing.detach().numpy()
        deviation = max(deviation, float(np.abs(matrix - nominal).max()))
        child_count = lengths[-1 - depth]
        error = (
            np.abs(matrix[:, :, 0]) * error[:, None]
            + np.abs(matrix[:, :, 0] - nominal[:, 0]) * amplitude_bound[:, None]
            + np.abs(matrix[:, :, 1] - nominal[:, 1]) * root_bound
        ).reshape(-1)[:child_count]
        amplitude_bound = np.repeat(
            (amplitude_bound + root_bound) * inverse_sqrt_two, 2
        )[:child_count]
        if not np.isfinite(error).all():
            raise ArithmeticError("Decoder perturbation envelope is nonfinite.")
    contraction = inverse_sqrt_two + deviation
    uniform_bound = (
        deviation
        * (2 + math.sqrt(2))
        * root_bound
        * (1 - contraction ** len(lengths))
        / (1 - contraction)
        if contraction < 1
        else None
    )
    return torch.from_numpy(error), deviation, uniform_bound


@torch.no_grad()
def clipped_perturbed_haar_coupling_bounds(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    metadata: dict,
    reference: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    *,
    mixing_by_level: tuple[torch.Tensor, ...],
    lower_bound: float | None,
    bin_count: int,
    max_variables: int,
    marginal_tv_bounds: torch.Tensor | None = None,
    reference_loss_error: torch.Tensor | None = None,
) -> dict:
    """Haar proxy LP with explicit outer bounds for perturbed tree mixing.

    Each inverse level supplies [nodes, 2, 2] matrices acting on (amplitude,
    detail), in the tokenizer's packed coarse-to-fine order. One decoder is
    shared across the requested streams. No exact merging of its perturbed
    amplitude histories is assumed. Additional floating-point evaluation and
    physical-model errors remain outside this real-arithmetic mixing bound.
    """
    _validate_risk_inputs(
        probabilities,
        codebook,
        None,
        reference,
        weights,
        offset,
        lower_bound=lower_bound,
    )
    if probabilities.device.type != "cpu" or probabilities.dtype != torch.float64:
        raise ValueError("Coupling optimization requires CPU float64 inputs.")
    field_error, deviation, uniform_bound = _mixing_error_envelope(
        codebook, metadata, reference.shape[-1], mixing_by_level
    )
    allowance = (weights.abs() * field_error).sum(-1)
    if not torch.isfinite(allowance).all():
        raise ArithmeticError("Decoder mixing loss allowance is nonfinite.")
    result = clipped_haar_coupling_bounds(
        probabilities,
        codebook,
        metadata,
        reference,
        weights,
        offset,
        lower_bound=lower_bound,
        bin_count=bin_count,
        max_variables=max_variables,
        marginal_tv_bounds=marginal_tv_bounds,
        reference_loss_error=reference_loss_error,
    )
    components = torch.split(
        probabilities @ codebook, metadata["component_lengths"], dim=-1
    )
    field_mean = components[0]
    for depth, mixing in enumerate(mixing_by_level):
        field_mean = (
            mixing[:, :, 0] * field_mean.unsqueeze(-1)
            + mixing[:, :, 1] * components[depth + 1].unsqueeze(-1)
        ).flatten(-2)[..., : metadata["lengths"][-1 - depth]]
    affine_risk = offset + (weights * (reference + field_mean)).sum(-1)
    result.update(
        nominal_unclipped_marginal_risk=result["unclipped_marginal_risk"],
        unclipped_marginal_risk=affine_risk,
        nominal_coupling_lower=result["coupling_lower"],
        nominal_coupling_upper=result["coupling_upper"],
        coupling_lower=result["coupling_lower"] - allowance,
        coupling_upper=result["coupling_upper"] + allowance,
        mixing_field_error_bound=field_error,
        mixing_loss_allowance=allowance,
        max_mixing_coefficient_deviation=deviation,
        uniform_mixing_field_error_bound=uniform_bound,
        status="supplied_marginal_perturbed_decoder_outer_bounds",
        resolution="haar_proxy_outer_transport",
    )
    if lower_bound is None:
        result.update(
            coupling_lower=affine_risk,
            coupling_upper=affine_risk,
            resolution="affine_direct_mean",
        )
    if result["reference_lower"] is not None:
        result.update(
            reference_lower=result["reference_lower"] - allowance,
            reference_upper=result["reference_upper"] + allowance,
        )
    if (
        not torch.isfinite(result["coupling_lower"]).all()
        or not torch.isfinite(result["coupling_upper"]).all()
    ):
        raise ArithmeticError("Perturbed decoder bounds are nonfinite.")
    return result


@torch.no_grad()
def evaluate_wavelet_coupling_goal(
    sampler, *, bin_count: int, max_variables: int, **goal_inputs
) -> dict:
    """One all-mask prediction followed by CPU coupling optimization.

    Goal inputs are those of ``evaluate_wavelet_marginal_goal``. This remains
    an explicit, uncalibrated evaluation API, not a serving acceptance gate.
    """
    p, codebook, metadata, reference, weights, constant = _wavelet_goal_predictions(
        sampler, **goal_inputs
    )
    p, codebook, reference, weights, constant = [
        item.detach().to(device="cpu", dtype=torch.float64)
        for item in (p, codebook, reference, weights, constant)
    ]
    # Explicitly remove float32 softmax summation roundoff before equality LPs.
    normalized = p / p.sum(-1, keepdim=True)
    normalization_tv = (normalized - p).abs().sum(-1).mul(0.5).max()
    batch_shape = p.shape[:-2]
    result = clipped_haar_coupling_bounds(
        normalized,
        codebook,
        metadata,
        reference,
        weights,
        p.new_zeros(batch_shape),
        lower_bound=sampler.state_lower_bound,
        bin_count=bin_count,
        max_variables=max_variables,
    )
    b, a = constant.shape
    return {
        "coupling_lower": constant + result["coupling_lower"].sum((-1, -2, -3)),
        "coupling_upper": constant + result["coupling_upper"].sum((-1, -2, -3)),
        "reference_lower": None,
        "reference_upper": None,
        "status": "uncalibrated_wavelet_coupling_goal_bounds",
        "denoiser_evaluations": b * a,
        "transport_variables": int(result["transport_variables"].sum()),
        "executed_haar_levels": result["executed_haar_levels"],
        "numerical_duality_gap": float(result["numerical_duality_gap"].max()),
        "primal_residual": float(result["primal_residual"].max()),
        "softmax_roundoff_normalization_tv": float(normalization_tv),
        "resolution": result["resolution"],
        "coefficient_bin_radius": result["coefficient_bin_radius"],
        "affine_codebook_max_deviation": result["affine_codebook_max_deviation"],
        "budget_scope": result["budget_scope"],
        "numerical_status": result["numerical_status"],
        "scope": "all_mask_clean_predictions_not_sequential_sampler_marginals",
    }
