"""Loss-local moment queries on the existing Haar coupling polytope.

Reports concern population moments under one fixed conditioning context. They
are not individual assay values or draws from the learned reverse sampler.
Finite report radii are supplied by the caller; no calibration is inferred.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack

from Libs.utils.evaluation.haar_query_preserving_compression import (
    QueryPreservingHaarProblem,
    compile_query_preserving_haar,
)
from Libs.utils.evaluation.marginal_intervention_risk import (
    _validate_risk_inputs,
    _wavelet_goal_predictions,
)
from Libs.utils.evaluation.wavelet_coupling_risk import (
    HaarCouplingProblem,
    _dual_lower_bound,
    _haar_metadata,
    _haar_state_problem,
)


@dataclass(frozen=True)
class HaarDependenceProblem:
    coupling: HaarCouplingProblem
    innovations: csr_matrix
    cross_moments: csr_matrix
    centered_innovations: csr_matrix
    centered_energy_moments: csr_matrix
    node_ids: tuple[tuple[int, int], ...]
    ranges: np.ndarray
    cross_moment_ranges: np.ndarray
    root_risk: float
    offset: float
    probabilities: np.ndarray
    codebook: np.ndarray
    source_codebook: np.ndarray
    lengths: tuple[int, ...]
    sizes: tuple[int, ...]
    reference: np.ndarray
    weights: np.ndarray
    lower_bound: float
    conditioning_key: str
    affine_grid_deviation: float
    coefficient_bin_radius: float

    @property
    def variable_count(self) -> int:
        return self.coupling.matrix.shape[1]


def _zero_detail_cost(amplitude, depth, position, levels, reference, weights, floor):
    """Cost below a node when all its descendants' details are set to zero."""
    width = 2 ** (levels - depth)
    start, stop = position * width, min((position + 1) * width, len(weights))
    value = np.zeros_like(amplitude, dtype=np.float64)
    for leaf in range(start, stop):
        value += weights[leaf] * np.maximum(
            floor, reference[leaf] + amplitude / math.sqrt(width)
        )
    return value


def _innovation(a, d, depth, position, levels, reference, weights, floor):
    value = -_zero_detail_cost(a, depth, position, levels, reference, weights, floor)
    for side, sign in ((0, 1), (1, -1)):
        value = value + _zero_detail_cost(
            (a + sign * d) / math.sqrt(2),
            depth + 1,
            2 * position + side,
            levels,
            reference,
            weights,
            floor,
        )
    return value


def compile_haar_dependence_queries(
    probabilities: torch.Tensor,
    codebook: torch.Tensor,
    metadata: dict,
    reference: torch.Tensor,
    weights: torch.Tensor,
    offset: torch.Tensor,
    *,
    lower_bound: float,
    conditioning_key: str,
    max_variables: int,
    bin_count: int | None = None,
) -> HaarDependenceProblem:
    """Compile one stream's exact-grid risk and its decision-sufficient moments.

    Queries always refer to the nominal bin-center statistic. Coarsening and
    nonuniform source-codebook deviations widen the field costs, not the query
    semantics. ``max_variables`` bounds the single-law table compilation; paired
    query selection enforces twice this count plus one.
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
        raise ValueError("Dependence queries require CPU float64 inputs.")
    if probabilities.ndim != 2 or not bool((weights != 0).any()):
        raise ValueError("Supply one temporal stream with nonzero goal weights.")
    if (
        lower_bound is None
        or not isinstance(conditioning_key, str)
        or not conditioning_key
    ):
        raise ValueError("Supply a finite clipping floor and a fixed conditioning key.")
    if type(max_variables) is not int or max_variables < 1:
        raise ValueError("max_variables must be a positive integer.")
    p = probabilities.detach().numpy().copy()
    if not np.allclose(p.sum(-1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Marginals must sum to one at float64 precision.")
    code = codebook.detach().numpy().copy()
    if len(code) < 2:
        raise ValueError("Dependence queries require at least two code values.")
    spacing = (code[-1] - code[0]) / (len(code) - 1)
    grid = code[0] + spacing * np.arange(len(code))
    deviation = float(np.abs(code - grid).max())
    bins = len(code) if bin_count is None else bin_count
    if type(bins) is not int or bins < 2 or len(code) % bins:
        raise ValueError(
            "bin_count must divide the source codebook and be at least two."
        )
    block = len(code) // bins
    radius = float(spacing * (block - 1) / 2 + deviation)
    centers = code[0] + spacing * ((block - 1) / 2 + block * np.arange(bins))
    masses = p.reshape(len(p), bins, block).sum(-1)
    lengths, sizes = _haar_metadata(metadata, len(reference), len(p))
    if not lengths:
        raise ValueError("Dependence queries require at least one inverse Haar level.")
    r, w = reference.detach().numpy().copy(), weights.detach().numpy().copy()
    coupling = _haar_state_problem(
        masses,
        float(centers[0]),
        float(spacing * block),
        radius,
        lengths,
        sizes,
        r,
        w,
        lower_bound,
        max_variables,
    )
    (
        rows,
        cols,
        values,
        cross_values,
        centered_values,
        energy_values,
        ranges,
        cross_ranges,
        node_ids,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, table in enumerate(coupling.tables):
        a = np.repeat(table.amplitudes, bins)
        d = np.tile(table.details, len(table.amplitudes))
        innovation = _innovation(
            a, d, table.depth, table.position, len(lengths), r, w, lower_bound
        )
        rows.extend([index] * len(innovation))
        cols.extend(range(table.columns.start, table.columns.stop))
        values.extend(innovation)
        cross_values.extend(a * d)
        raw = innovation.reshape(-1, bins)
        marginal_index = sum(sizes[: table.depth + 1]) + table.position
        centered_values.extend((raw - (raw @ masses[marginal_index])[:, None]).ravel())
        energy_values.extend(
            a**2 * (d**2 - float(masses[marginal_index] @ table.details**2))
        )
        ranges.append((float(innovation.min()), float(innovation.max())))
        cross_ranges.append((float((a * d).min()), float((a * d).max())))
        node_ids.append((table.depth, table.position))
    queries = coo_matrix(
        (values, (rows, cols)), shape=(len(node_ids), coupling.matrix.shape[1])
    ).tocsr()
    queries.eliminate_zeros()
    cross_moments = coo_matrix(
        (cross_values, (rows, cols)), shape=queries.shape
    ).tocsr()
    cross_moments.eliminate_zeros()
    centered_queries = coo_matrix(
        (centered_values, (rows, cols)), shape=queries.shape
    ).tocsr()
    centered_queries.eliminate_zeros()
    energy_queries = coo_matrix(
        (energy_values, (rows, cols)), shape=queries.shape
    ).tocsr()
    energy_queries.eliminate_zeros()
    root_risk = float(offset)
    for root in range(sizes[0]):
        root_cost = _zero_detail_cost(centers, 0, root, len(lengths), r, w, lower_bound)
        root_risk += float(masses[root] @ root_cost)
    return HaarDependenceProblem(
        coupling,
        queries,
        cross_moments,
        centered_queries,
        energy_queries,
        tuple(node_ids),
        np.asarray(ranges),
        np.asarray(cross_ranges),
        root_risk,
        float(offset),
        masses,
        centers,
        code,
        lengths,
        sizes,
        r,
        w,
        float(lower_bound),
        conditioning_key,
        deviation,
        radius,
    )


def centered_haar_innovations(problem: HaarDependenceProblem) -> csr_matrix:
    """Threshold innovations minus their independent-detail conditional mean.

    This is a distinct population-moment query. For every incoming amplitude,
    averaging over the prescribed detail marginal gives zero. It must not be
    substituted for an uncentered innovation or a full local observation law.
    """
    return problem.centered_innovations


def haar_resampling_influences(
    problem: HaarDependenceProblem, *, current_report: dict | None = None
) -> dict:
    """Loss changes from independently resampling one original Haar detail.

    Requires exact singleton coefficient bins and a balanced full-depth tree.
    Existing local-innovation reports protect their nodes and ancestors. No
    assertion is made about repair under additional, unrepresented constraints.
    """
    n = len(problem.reference)
    levels = len(problem.lengths)
    if (
        n < 2
        or n & (n - 1)
        or 2**levels != n
        or problem.sizes[0] != 1
        or problem.coefficient_bin_radius != 0
        or problem.affine_grid_deviation != 0
    ):
        raise ValueError(
            "Resampling certificates require balanced exact-grid singleton Haar states."
        )
    _report_constraints(problem, current_report, problem.conditioning_key)
    protected = set()
    for index in () if not current_report else current_report["intervals"]:
        depth, position = problem.node_ids[index]
        while True:
            protected.add((depth, position))
            if depth == 0:
                break
            depth, position = depth - 1, position // 2
    return _balanced_haar_influences(
        problem.probabilities, problem.codebook, problem.weights,
        problem.node_ids, protected,
    )


def _balanced_haar_influences(probabilities, codebook, weights, node_ids, protected=()):
    """Linear-time detail-redraw bounds after the balanced input contract is checked."""
    n = len(weights)
    absolute_weights = abs(weights)
    prefix = np.r_[0.0, np.cumsum(absolute_weights)]
    distance = abs(codebook[:, None] - codebook[None, :])
    deviations, masses, influences = [], [], []
    for depth, position in node_ids:
        size = n // 2**depth
        mass = float(prefix[(position + 1) * size] - prefix[position * size])
        marginal = probabilities[2**depth + position]
        eta = float(marginal @ distance @ marginal)
        masses.append(mass)
        deviations.append(eta)
        influences.append(eta * mass / math.sqrt(size))
    total_weight = float(absolute_weights.sum())
    coefficient_bound = float(abs(codebook).max())
    return {
        "influences": np.asarray(influences),
        "subtree_weight_masses": np.asarray(masses),
        "independent_detail_deviations": np.asarray(deviations),
        "repair_allowed": np.asarray([node not in protected for node in node_ids]),
        "protected_node_ids": tuple(sorted(protected)),
        "total_lipschitz_weight": total_weight,
        "coefficient_bound": coefficient_bound,
        "summability_bound": coefficient_bound * total_weight / (math.sqrt(2) - 1),
        "uniform_absolute_weights": bool(
            np.all(absolute_weights == absolute_weights[0])
        ),
        "scope": "centered_local_moments_and_existing_local_innovation_reports",
    }


def _statistic_rows(problem: HaarDependenceProblem, statistic: str) -> csr_matrix:
    if statistic == "innovation":
        return problem.innovations
    if statistic == "centered_innovation":
        return centered_haar_innovations(problem)
    if statistic == "centered_energy":
        return problem.centered_energy_moments
    if statistic == "cross_moment":
        return problem.cross_moments
    if statistic == "direct_goal":
        # The nominal goal's expectation differs from the innovation sum only
        # by the known root term. Its difference between two laws is identical.
        return csr_matrix(problem.innovations.sum(axis=0))
    raise ValueError("Unknown query statistic.")


def _query_ids(
    problem: HaarDependenceProblem, query_ids, statistic="innovation"
) -> tuple[int, ...]:
    ids = tuple(query_ids)
    count = _statistic_rows(problem, statistic).shape[0]
    if len(set(ids)) != len(ids) or any(
        type(index) is not int or not 0 <= index < count for index in ids
    ):
        raise ValueError("Query IDs must be distinct valid integer indices.")
    return ids


def _statistic_definition(problem: HaarDependenceProblem) -> dict:
    return {
        "statistic": "haar_zero_detail_clipped_loss_innovation",
        "conditioning_key": problem.conditioning_key,
        "node_ids": [list(node) for node in problem.node_ids],
        "crop_lengths": list(problem.lengths),
        "nominal_codebook": problem.codebook.tolist(),
        "source_codebook": problem.source_codebook.tolist(),
        "reference": problem.reference.tolist(),
        "weights": problem.weights.tolist(),
        "lower_bound": problem.lower_bound,
    }


def bind_haar_moment_intervals(problem: HaarDependenceProblem, intervals: dict) -> dict:
    """Attach an explicit statistic definition to externally justified intervals.

    Binding records the declared meaning; it does not validate calibration or
    authorize transferring measurements from another compiled statistic.
    """
    return {
        "intervals": dict(intervals),
        "statistic_definition": _statistic_definition(problem),
    }


def _report_constraints(problem, report, conditioning_key):
    if conditioning_key != problem.conditioning_key:
        raise ValueError(
            "The moment report belongs to a different conditioning context."
        )
    if report is None or not report:
        return csr_matrix((0, problem.variable_count)), np.empty(0)
    if report.get("statistic_definition") != _statistic_definition(problem):
        raise ValueError(
            "The report statistic definition does not match the compiled goal/codebook."
        )
    values = report["intervals"]
    ids = _query_ids(problem, tuple(values))
    if not ids:
        return csr_matrix((0, problem.variable_count)), np.empty(0)
    intervals = np.asarray([values[index] for index in ids], dtype=np.float64)
    if intervals.shape != (len(ids), 2) or not np.isfinite(intervals).all():
        raise ValueError("Each report interval must have two finite endpoints.")
    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ValueError("Moment intervals must be ordered.")
    rows = problem.innovations[list(ids)]
    return vstack([rows, -rows], format="csr"), np.r_[intervals[:, 1], -intervals[:, 0]]


def condition_haar_risk(
    problem: HaarDependenceProblem,
    report: dict | None,
    *,
    conditioning_key: str,
) -> dict:
    """Condition the population-law class on local moment intervals.

    Infeasible evidence raises explicitly, including disagreement between a
    reported moment and the supplied marginal class.
    """
    matrix, rhs = _report_constraints(problem, report, conditioning_key)
    p = problem.coupling
    lower, lr, lg = _dual_lower_bound(
        p.lower_cost,
        p.matrix,
        p.rhs,
        p.slices,
        inequality_matrix=matrix,
        inequality_rhs=rhs,
    )
    negative_upper, ur, ug = _dual_lower_bound(
        -p.upper_cost,
        p.matrix,
        p.rhs,
        p.slices,
        inequality_matrix=matrix,
        inequality_rhs=rhs,
    )
    return {
        "lower": problem.offset + lower,
        "upper": problem.offset - negative_upper,
        "status": "supplied_marginal_moment_conditioned_bounds",
        "conditioning_key": problem.conditioning_key,
        "transport_variables": problem.variable_count,
        "primal_residual": max(lr, ur),
        "numerical_duality_gap": max(lg, ug),
        "reference_risk_qualified": False,
        "resolution": "nominal_bin_moments_with_field_envelopes",
        "coefficient_bin_radius": problem.coefficient_bin_radius,
    }


def worst_report_decision_regret(
    problem: HaarDependenceProblem,
    query_ids: tuple[int, ...],
    report_radii: np.ndarray,
    *,
    alternative_loss: float,
    conditioning_key: str,
    max_variables: int,
    current_report: dict | None = None,
    statistic: str = "innovation",
) -> dict:
    """Worst-report deterministic minimax regret for one precommitted query set.

    The comparison is with a fixed known loss. It is neither Bayesian VOI nor
    optimal adaptive-policy value. A common report for two laws exists exactly
    when their moments differ by at most twice each component's error radius.
    """
    ids = _query_ids(problem, query_ids, statistic)
    radii = np.asarray(report_radii, dtype=np.float64)
    if radii.shape != (len(ids),) or not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("Supply a finite nonnegative radius per queried moment.")
    if not math.isfinite(alternative_loss):
        raise ValueError("The alternative loss must be finite.")
    if type(max_variables) is not int or max_variables < 2 * problem.variable_count + 1:
        raise ValueError("Paired transport variable budget exceeded before allocation.")
    observed, observed_rhs = _report_constraints(
        problem, current_report, conditioning_key
    )
    p = problem.coupling
    result = _paired_regret_bound(
        p.matrix, p.rhs, p.slices, p.lower_cost, p.upper_cost, problem.offset,
        _statistic_rows(problem, statistic)[list(ids)], radii, alternative_loss,
        observed, observed_rhs,
    )
    result.update(
        query_ids=ids,
        statistic=statistic,
        status="uncalibrated_deterministic_worst_report_regret",
        conditioning_key=problem.conditioning_key,
        scope="fixed_context_population_moments_and_fixed_alternative",
        coefficient_bin_radius=problem.coefficient_bin_radius,
        regret_scope="exact_for_ideal_singleton_bins_otherwise_outer_envelope",
    )
    return result


def _paired_regret_bound(
    matrix, targets, mass_slices, lower_cost, upper_cost, offset,
    query, radii, alternative_loss, observed, observed_rhs,
):
    """Shared paired-law LP for exact and query-preserving state compilers."""
    n = matrix.shape[1]
    equalities = hstack(
        [block_diag([matrix, matrix]), csr_matrix((2 * len(targets), 1))],
        format="csr",
    )
    rhs = np.r_[targets, targets]
    constraints, bounds = [], []
    if observed.shape[0]:
        constraints.append(
            hstack(
                [
                    block_diag([observed, observed]),
                    csr_matrix((2 * len(observed_rhs), 1)),
                ]
            )
        )
        bounds.extend(np.r_[observed_rhs, observed_rhs])
    difference = hstack([query, -query, csr_matrix((query.shape[0], 1))], format="csr")
    constraints.extend([difference, -difference])
    bounds.extend(np.r_[2 * radii, 2 * radii])
    rows = np.zeros((2, 2 * n + 1))
    rows[0, :n], rows[1, n : 2 * n], rows[:, -1] = -upper_cost, lower_cost, 1.0
    constraints.append(csr_matrix(rows))
    bounds.extend(
        [offset - alternative_loss, alternative_loss - offset]
    )
    inequalities = vstack(constraints, format="csr")
    objective = np.zeros(2 * n + 1)
    objective[-1] = -1.0
    radius = 1.0 + abs(offset - alternative_loss)
    radius += sum(
        float(np.maximum(np.abs(lower_cost[sl]), np.abs(upper_cost[sl])).max())
        for sl in mass_slices
    )
    slices = (*mass_slices, *(slice(sl.start + n, sl.stop + n) for sl in mass_slices))
    negative_regret, residual, gap = _dual_lower_bound(
        objective,
        equalities,
        rhs,
        slices,
        inequality_matrix=inequalities,
        inequality_rhs=np.asarray(bounds),
        scalar_radius=radius,
    )
    return {
        "worst_report_regret": max(0.0, -negative_regret),
        "transport_variables": 2 * n + 1,
        "primal_residual": residual,
        "numerical_duality_gap": gap,
    }


def select_haar_dependence_query(
    problem: HaarDependenceProblem,
    candidates: tuple[tuple[int, ...], ...],
    report_radii: tuple[np.ndarray, ...],
    query_costs: np.ndarray,
    *,
    alternative_loss: float,
    conditioning_key: str,
    max_variables: int,
    current_report: dict | None = None,
) -> dict:
    """Compare a finite declared menu and the no-query option by regret plus cost."""
    costs = np.asarray(query_costs, dtype=np.float64)
    if len(candidates) != len(report_radii) or costs.shape != (len(candidates),):
        raise ValueError("Query sets, radii, and costs must align.")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Query costs must be finite, nonnegative loss-unit costs.")
    results = []
    for ids, radii, cost in zip(
        ((), *candidates), (np.empty(0), *report_radii), (0.0, *costs), strict=True
    ):
        result = worst_report_decision_regret(
            problem,
            ids,
            radii,
            alternative_loss=alternative_loss,
            conditioning_key=conditioning_key,
            max_variables=max_variables,
            current_report=current_report,
        )
        result.update(
            query_cost=float(cost),
            objective=result["worst_report_regret"] + float(cost),
        )
        results.append(result)
    selected = min(range(len(results)), key=lambda index: results[index]["objective"])
    return {
        "selected": results[selected],
        "options": results,
        "status": "finite_menu_precommitted_query_selection",
    }


def select_resampling_screened_query(
    problem: HaarDependenceProblem,
    report_radii: np.ndarray,
    query_costs: np.ndarray,
    *,
    alternative_loss: float,
    conditioning_key: str,
    max_variables: int,
    optimality_tolerance: float,
    current_report: dict | None = None,
    statistic: str = "centered_innovation",
) -> dict:
    """Select a singleton centered query using resampling lower bounds.

    The no-query comparator is always available. Report costs are in loss units.
    Returned optimality certificates use checked float64 LP bounds; this routine
    does not provide interval-arithmetic certification or observation-law/VOI
    guarantees. Its hypothesis is a population centered-moment query menu.
    """
    if statistic not in ("centered_innovation", "centered_energy"):
        raise ValueError(
            "Resampling screening requires a supported centered moment statistic."
        )
    count = len(problem.node_ids)
    radii, costs = (
        np.asarray(report_radii, dtype=float),
        np.asarray(query_costs, dtype=float),
    )
    if (
        radii.shape != (count,)
        or costs.shape != (count,)
        or not np.isfinite(radii).all()
        or not np.isfinite(costs).all()
        or (radii < 0).any()
        or (costs < 0).any()
        or not math.isfinite(optimality_tolerance)
        or optimality_tolerance < 0
    ):
        raise ValueError(
            "Supply finite nonnegative radii, costs and optimality tolerance."
        )
    influence = haar_resampling_influences(problem, current_report=current_report)

    def solve(ids):
        result = worst_report_decision_regret(
            problem,
            ids,
            radii[list(ids)],
            alternative_loss=alternative_loss,
            conditioning_key=conditioning_key,
            max_variables=max_variables,
            current_report=current_report,
            statistic=statistic,
        )
        upper = result["worst_report_regret"]
        padding = 128 * np.finfo(float).eps * (1 + upper)
        lower = max(0.0, upper - result["numerical_duality_gap"] - padding)
        cost = 0.0 if not ids else float(costs[ids[0]])
        result.update(
            objective_lower=lower + cost, objective_upper=upper + cost, query_cost=cost
        )
        return result

    baseline = solve(())
    base_lower = baseline["objective_lower"]
    lower = (
        np.where(
            influence["repair_allowed"],
            np.maximum(0.0, base_lower - influence["influences"]),
            0.0,
        )
        + costs
    )
    selected = baseline
    solved, remaining = [], set(range(count))
    final_lower = np.r_[baseline["objective_lower"], lower]
    trace = []
    while remaining:
        viable = [
            i
            for i in remaining
            if lower[i] < selected["objective_upper"] - optimality_tolerance
        ]
        if not viable:
            break
        index = min(viable, key=lambda i: (lower[i], -influence["influences"][i], i))
        result = solve((index,))
        if lower[index] > result["objective_upper"] + 1e-8 * (
            1 + abs(result["objective_upper"])
        ):
            raise ArithmeticError(
                "Resampling lower bound contradicts the checked query LP."
            )
        final_lower[index + 1] = max(lower[index], result["objective_lower"])
        solved.append(index)
        remaining.remove(index)
        if result["objective_upper"] < selected["objective_upper"]:
            selected = result
        trace.append(
            {
                "query_id": index,
                "objective_lower": result["objective_lower"],
                "objective_upper": result["objective_upper"],
                "incumbent_upper": selected["objective_upper"],
            }
        )
    gap = max(0.0, selected["objective_upper"] - float(final_lower.min()))
    if gap > optimality_tolerance + 1e-8 * (1 + selected["objective_upper"]):
        raise ArithmeticError(
            "Screened query selection has an unresolved numerical gap."
        )
    minimum_cost = float(costs.min())
    improvement = base_lower + minimum_cost - selected["objective_upper"]
    count_bound = None
    if improvement > 0:
        count_bound = influence["summability_bound"] / improvement
    return {
        "selected": selected,
        "no_query": baseline,
        "solved_query_ids": solved,
        "screened_query_ids": sorted(remaining),
        "solve_count": 1 + len(solved),
        "dictionary_size": count,
        "optimality_gap": gap,
        "optimality_tolerance": optimality_tolerance,
        "query_objective_lower_bounds": final_lower[1:].tolist(),
        "resampling_influences": influence["influences"].tolist(),
        "repair_allowed": influence["repair_allowed"].tolist(),
        "protected_node_ids": influence["protected_node_ids"],
        "summability_bound": influence["summability_bound"],
        "checked_incumbent_improvement": improvement,
        "repairable_solve_count_bound": count_bound,
        "uniform_weight_solve_count_bound": (
            2
            * (influence["coefficient_bound"] * influence["total_lipschitz_weight"])
            ** 2
            / (len(problem.reference) * improvement**2)
            if improvement > 0 and influence["uniform_absolute_weights"]
            else None
        ),
        "statistic": statistic,
        "trace": trace,
        "floating_point_interval_certified": False,
        "status": "numerically_checked_centered_query_selection",
        "scope": "finite_precommitted_centered_population_moment_menu",
    }


def compressed_haar_query_regret(
    problem: QueryPreservingHaarProblem,
    query_node: tuple[int, int] | None,
    *,
    report_radius: float,
    alternative_loss: float,
    max_variables: int,
    statistic: str,
) -> dict:
    """Bound a centered-moment query while preserving its exact amplitude path.

    Original coefficient singleton marginals are retained. This interface has
    no prior reports. Its regret allowance is the loss error e, not twice e.
    LP endpoint checks remain float64 checks, not exact feasible-law certificates.
    """
    if statistic not in ("centered_innovation", "centered_energy"):
        raise ValueError("A supported centered population-moment statistic is required.")
    if (
        not math.isfinite(report_radius) or report_radius < 0
        or not math.isfinite(alternative_loss)
    ):
        raise ValueError("Supply a finite alternative and nonnegative report radius.")
    n = problem.variable_count
    if type(max_variables) is not int or max_variables < 2 * n + 1:
        raise ValueError("Paired transport variable budget exceeded before allocation.")
    if query_node is None:
        query, radii = csr_matrix((0, n)), np.empty(0)
    else:
        if query_node not in problem.protected_nodes:
            raise ValueError("The query node and its ancestors must be exact.")
        index = problem.node_ids.index(query_node)
        table = problem.tables[index]
        depth, position = query_node
        a = np.repeat(table.states_numeric, len(problem.codebook))
        d = np.tile(problem.codebook, len(table.states_numeric))
        marginal = problem.probabilities[2**depth + position]
        if statistic == "centered_energy":
            values = a**2 * (d**2 - float(marginal @ problem.codebook**2))
        else:
            raw = _innovation(
                a, d, depth, position, len(problem.reference).bit_length() - 1,
                problem.reference, problem.weights, problem.lower_bound,
            ).reshape(-1, len(problem.codebook))
            values = (raw - (raw @ marginal)[:, None]).ravel()
        query = csr_matrix(
            (values, (np.zeros(len(values), dtype=int),
                      np.arange(table.columns.start, table.columns.stop))),
            shape=(1, n),
        )
        radii = np.asarray([report_radius])
    result = _paired_regret_bound(
        problem.matrix, problem.rhs, problem.slices,
        problem.surrogate_cost, problem.surrogate_cost, problem.offset,
        query, radii, alternative_loss, csr_matrix((0, n)), np.empty(0),
    )
    upper = result["worst_report_regret"]
    padding = 128 * np.finfo(float).eps * (1 + upper)
    surrogate_lower = max(0.0, upper - result["numerical_duality_gap"] - padding)
    error = float(abs(problem.weights) @ problem.leaf_error_radii)
    result.update(
        regret_lower=max(0.0, surrogate_lower - error),
        regret_upper=upper + error,
        surrogate_regret_lower=surrogate_lower,
        loss_error_allowance=error,
        query_node=query_node,
        statistic=statistic,
        base_variables=n,
        protected_nodes=problem.protected_nodes,
        epsilon=problem.epsilon,
        rounding=problem.rounding,
        floating_point_interval_certified=False,
        status="numerically_checked_compressed_regret_interval",
        scope="original_singleton_marginals_exact_centered_query_no_prior_reports",
    )
    return result


def select_compressed_resampling_query(
    probabilities, codebook, reference, weights, report_radii, query_costs,
    *, lower_bound, offset, alternative_loss, epsilon, rounding, max_variables,
    max_paired_variables, optimality_tolerance, max_refinements,
    refinement_factor, statistic="centered_innovation", search="resampling",
) -> dict:
    """Refine per-query compressed LP intervals until a finite-menu gap closes.

    Each compilation protects one query path. Screening uses a shared no-query
    lower endpoint and the original-coefficient resampling influences. Exhaustive
    search uses identical compilations, error envelopes and refinement budgets.
    The budget limits refinements per option; unresolved gaps are returned.
    """
    if search not in ("resampling", "exhaustive"):
        raise ValueError("search must be resampling or exhaustive.")
    if statistic not in ("centered_innovation", "centered_energy"):
        raise ValueError("A supported centered population-moment statistic is required.")
    if (
        type(max_refinements) is not int or max_refinements < 0
        or not math.isfinite(refinement_factor) or not 0 < refinement_factor < 1
        or not math.isfinite(optimality_tolerance) or optimality_tolerance <= 0
    ):
        raise ValueError("Supply positive accuracy and a finite geometric refinement budget.")
    started = time.perf_counter()
    base = compile_query_preserving_haar(
        probabilities, codebook, reference, weights, lower_bound=lower_bound,
        offset=offset, epsilon=epsilon, protected_nodes=(), rounding=rounding,
        max_variables=max_variables,
    )
    construction_seconds = time.perf_counter() - started
    count = len(base.node_ids)
    radii, costs = np.asarray(report_radii, dtype=float), np.asarray(query_costs, dtype=float)
    if (
        radii.shape != (count,) or costs.shape != (count,)
        or not np.isfinite(radii).all() or not np.isfinite(costs).all()
        or (radii < 0).any() or (costs < 0).any()
    ):
        raise ValueError("Supply a finite nonnegative radius and cost for every query.")
    node_ids = base.node_ids
    influence = _balanced_haar_influences(
        base.probabilities, base.codebook, base.weights, node_ids,
    )
    # Finish each option before visiting the next; keep the initial bounds fixed.
    lower, upper = np.r_[0.0, costs], np.full(count + 1, np.inf)
    attempts = np.zeros(count + 1, dtype=int)
    trace, results = [], {}
    lp_seconds = 0.0
    peak_variables = total_variables = 0
    selected = 0

    def refine(pending):
        nonlocal base, construction_seconds, lp_seconds, peak_variables
        nonlocal total_variables, selected
        while attempts[pending] <= max_refinements:
            step = int(attempts[pending])
            if not (pending == 0 and step == 0):
                # Drop the previous state tables before constructing the next.
                base = None
                started = time.perf_counter()
                base = compile_query_preserving_haar(
                    probabilities, codebook, reference, weights, lower_bound=lower_bound,
                    offset=offset, epsilon=epsilon * refinement_factor**step,
                    protected_nodes=() if pending == 0 else (node_ids[pending - 1],),
                    rounding=rounding, max_variables=max_variables,
                )
                construction_seconds += time.perf_counter() - started
            started = time.perf_counter()
            result = compressed_haar_query_regret(
                base, None if pending == 0 else node_ids[pending - 1],
                report_radius=0.0 if pending == 0 else float(radii[pending - 1]),
                alternative_loss=alternative_loss, max_variables=max_paired_variables,
                statistic=statistic,
            )
            lp_seconds += time.perf_counter() - started
            cost = 0.0 if pending == 0 else float(costs[pending - 1])
            lower[pending] = max(lower[pending], result["regret_lower"] + cost)
            value = result["regret_upper"] + cost
            if value < upper[pending]:
                upper[pending] = value
                results[pending] = result
            attempts[pending] += 1
            numerical_slack = 1e-8 * (1 + np.max(np.abs(upper[np.isfinite(upper)])))
            if np.any(lower > upper + numerical_slack):
                raise ArithmeticError("Compressed/refined regret intervals are inconsistent.")
            selected = int(np.argmin(upper))
            peak_variables = max(peak_variables, result["transport_variables"])
            total_variables += result["transport_variables"]
            trace.append({
                "option_index": pending, "refinement": step,
                "epsilon": base.epsilon, "loss_error_allowance": result["loss_error_allowance"],
                "objective_lower": float(lower[pending]), "objective_upper": float(upper[pending]),
                "optimality_gap": max(0.0, float(upper[selected] - lower.min())),
                "transport_variables": result["transport_variables"],
                "primal_residual": result["primal_residual"],
                "numerical_duality_gap": result["numerical_duality_gap"],
            })
            if upper[pending] - lower[pending] <= optimality_tolerance:
                break
            if pending and lower[pending] >= upper[selected] - optimality_tolerance:
                break

    refine(0)
    base_lower = float(lower[0])
    if search == "resampling":
        lower[1:] = np.maximum(0.0, base_lower - influence["influences"]) + costs
    frozen_lower = lower[1:].copy()
    order = sorted(range(count), key=lambda i: (frozen_lower[i], -influence["influences"][i], i))
    for index in order:
        if search == "resampling" and frozen_lower[index] >= upper[selected] - optimality_tolerance:
            break
        refine(index + 1)
    gap = max(0.0, float(upper[selected] - lower.min()))
    status = (
        "numerically_checked_compressed_query_selection" if gap <= optimality_tolerance
        else "unresolved_compression_accuracy_budget"
    )
    improvement = base_lower + float(costs.min()) - float(upper[selected])
    chosen = dict(results[selected])
    chosen.update(
        option_index=selected,
        query_ids=() if selected == 0 else (selected - 1,),
        objective_lower=float(lower[selected]), objective_upper=float(upper[selected]),
        query_cost=0.0 if selected == 0 else float(costs[selected - 1]),
    )
    return {
        "selected": chosen, "objective_lower_bounds": lower.tolist(),
        "objective_upper_bounds": [float(v) if math.isfinite(v) else None for v in upper],
        "solve_count": int(attempts.sum()), "attempts_per_option": attempts.tolist(),
        "solved_query_ids": np.flatnonzero(attempts[1:] > 0).tolist(),
        "screened_query_ids": np.flatnonzero(attempts[1:] == 0).tolist(),
        "optimality_gap": gap, "optimality_tolerance": optimality_tolerance,
        "selection_resolved": gap <= optimality_tolerance, "status": status,
        "dictionary_size": count, "search": search, "statistic": statistic,
        "frozen_query_lower_bounds": frozen_lower.tolist(),
        "checked_incumbent_improvement": improvement,
        "distinct_query_count_bound": (
            min(count, influence["summability_bound"] / improvement)
            if improvement > 0 and search == "resampling" else None
        ),
        "resampling_influences": influence["influences"].tolist(),
        "summability_bound": influence["summability_bound"],
        "construction_seconds": construction_seconds, "lp_seconds": lp_seconds,
        "peak_paired_variables": peak_variables, "total_paired_variables": total_variables,
        "trace": trace, "floating_point_interval_certified": False,
        "scope": "finite_centered_population_moment_menu_no_prior_reports",
    }


def evaluate_haar_innovations(
    problem: HaarDependenceProblem,
    coefficient_indices: np.ndarray,
    *,
    conditioning_key: str,
) -> np.ndarray:
    """Compute the query dictionary on matched coefficient samples, without fitting."""
    if conditioning_key != problem.conditioning_key:
        raise ValueError("Samples must use the compiled conditioning context.")
    indices = np.asarray(coefficient_indices)
    if indices.ndim != 2 or indices.shape[1] != sum(problem.sizes):
        raise ValueError("Expected samples by packed coefficient index.")
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or (indices < 0).any()
        or (indices >= len(problem.source_codebook)).any()
    ):
        raise ValueError("Sampled coefficients must be valid integer code indices.")
    coefficients = problem.codebook[
        indices // (len(problem.source_codebook) // len(problem.codebook))
    ]
    components = np.split(coefficients, np.cumsum(problem.sizes)[:-1], axis=1)
    amplitudes = components[0]
    values = np.zeros((len(indices), len(problem.node_ids)))
    node_lookup = {node: index for index, node in enumerate(problem.node_ids)}
    for depth, details in enumerate(components[1:]):
        for position in range(amplitudes.shape[1]):
            if (depth, position) in node_lookup:
                values[:, node_lookup[(depth, position)]] = _innovation(
                    amplitudes[:, position],
                    details[:, position],
                    depth,
                    position,
                    len(problem.lengths),
                    problem.reference,
                    problem.weights,
                    problem.lower_bound,
                )
        amplitudes = np.stack(
            [
                (amplitudes + details) / math.sqrt(2),
                (amplitudes - details) / math.sqrt(2),
            ],
            axis=-1,
        ).reshape(len(indices), -1)[:, : problem.lengths[-1 - depth]]
    return values


def haar_energy_tail_bound(problem: HaarDependenceProblem, observed_depth: int) -> dict:
    """Unmeasured innovation bound for a complete tree and uniform clipped burden."""
    n, levels = len(problem.weights), len(problem.lengths)
    if n != 2**levels or problem.sizes[0] != 1:
        raise ValueError("The energy-tail certificate requires a full balanced tree.")
    if not np.allclose(problem.weights, np.full(n, 1 / n), rtol=0, atol=1e-14):
        raise ValueError(
            "The energy-tail certificate requires uniform normalized weights."
        )
    if not np.all(problem.reference == problem.reference[0]):
        raise ValueError("The energy-tail certificate requires a common reference.")
    if type(observed_depth) is not int or not 0 <= observed_depth <= levels:
        raise ValueError("observed_depth must lie within the executed tree.")
    energy = problem.probabilities @ problem.codebook**2
    groups = np.split(energy, np.cumsum(problem.sizes)[:-1])
    level_energy = np.array([part.sum() for part in groups[1:]])
    tail = float(np.sqrt(level_energy[observed_depth:]).sum() / (2 * math.sqrt(n)))
    ids = tuple(
        index
        for index, (depth, _) in enumerate(problem.node_ids)
        if depth < observed_depth
    )
    return {
        "tail_lower": 0.0,
        "tail_upper": tail,
        "required_query_ids": ids,
        "level_detail_energy": level_energy.tolist(),
        "status": "singleton_energy_tail_bound",
        "scope": "nominal_bin_innovations_only",
    }


@torch.no_grad()
def compile_wavelet_dependence_goal(
    sampler,
    *,
    conditioning_key: str,
    bin_count: int,
    max_variables: int,
    goal_threshold: float,
    **goal_inputs,
) -> dict:
    """Compile a one-stream declared goal from the actual all-mask model interface.

    The footprint check prevents silently discarding other species, regions,
    candidates, or samples. Broader field goals retain the existing rollout and
    marginal interfaces. The supplied context key labels, not validates, evidence.
    The loss is ``goal_offset + sum w * max(goal_threshold, X)``. For an
    exceedance loss, subtract ``goal_threshold * sum w`` in ``goal_offset``,
    including the weights of restored observations. Alternatives use these units.
    """
    if not math.isfinite(goal_threshold) or goal_threshold < sampler.state_lower_bound:
        raise ValueError(
            "The declared goal threshold must be finite and at least the physical floor."
        )
    p, code, metadata, reference, weights, constant = _wavelet_goal_predictions(
        sampler, **goal_inputs
    )
    active = weights.abs().sum(-1).ne(0).nonzero()
    if constant.numel() != 1 or len(active) != 1:
        raise ValueError(
            "Dependence goal compilation requires one sample/action and one active temporal stream."
        )
    index = tuple(int(x) for x in active[0])
    probabilities = p[index].detach().to(device="cpu", dtype=torch.float64)
    normalized = probabilities / probabilities.sum(-1, keepdim=True)
    # Recompute restored observations in the same thresholded loss as the future.
    fixed = goal_inputs["time_hours"].eq(0) & sampler.preserve_initial_state
    raw_weights = goal_inputs["goal_weights"]
    restored = goal_inputs["initial_state"][:, :, None].expand_as(raw_weights)
    constant = goal_inputs["goal_offset"].to(torch.float64) + (
        raw_weights * restored.clamp_min(goal_threshold) * fixed[..., None, None, None]
    ).sum(dim=(2, 3, 4, 5)).to(torch.float64)
    problem = compile_haar_dependence_queries(
        normalized,
        code.detach().cpu(),
        metadata,
        reference[index].detach().cpu(),
        weights[index].detach().cpu(),
        constant.reshape(()).detach().cpu(),
        lower_bound=goal_threshold,
        conditioning_key=conditioning_key,
        max_variables=max_variables,
        bin_count=bin_count,
    )
    return {
        "problem": problem,
        "denoiser_evaluations": 1,
        "softmax_roundoff_normalization_tv": float(
            (probabilities - normalized).abs().sum(-1).mul(0.5).max()
        ),
        "status": "uncalibrated_all_mask_dependence_goal",
        "scope": "all_mask_clean_predictions_not_sequential_sampler_marginals",
    }


def report_from_matched_blocks(
    problem: HaarDependenceProblem,
    query_ids: tuple[int, ...],
    field_block_means: np.ndarray,
    *,
    measurement_error: np.ndarray,
    coefficient_error_bound: float,
    failure_probability: float,
    conditioning_key: str,
) -> dict:
    """Fixed-sample simultaneous intervals from paired physical block means.

    Each row is an independent matched trajectory replicate in the declared
    source context. The encoded clean field must be this measured field, not an
    inaccessible pre-clipping field. The caller must justify the coefficient
    error bound, including saturation and binning. These assumptions are not
    inferred from the supplied arrays or from feasible moment constraints.
    """
    haar_energy_tail_bound(problem, 0)  # Enforce the balanced uniform-goal contract.
    if conditioning_key != problem.conditioning_key:
        raise ValueError("The block observations use a different conditioning context.")
    ids = _query_ids(problem, query_ids)
    means, error = (
        np.asarray(field_block_means, dtype=np.float64),
        np.asarray(measurement_error, dtype=np.float64),
    )
    if (
        not ids
        or means.ndim != 3
        or means.shape[1:] != (len(ids), 2)
        or means.shape[0] < 1
    ):
        raise ValueError("Supply replicate by query by two matched block means.")
    if (
        error.shape != (len(ids), 2)
        or not np.isfinite(error).all()
        or (error < 0).any()
        or not np.isfinite(means).all()
    ):
        raise ValueError(
            "Block means and nonnegative measurement error bounds must be finite and aligned."
        )
    if not math.isfinite(coefficient_error_bound) or coefficient_error_bound < 0:
        raise ValueError(
            "A finite nonnegative coefficient representation error is required."
        )
    if not math.isfinite(failure_probability) or not 0 < failure_probability < 1:
        raise ValueError("failure_probability must lie strictly between zero and one.")
    n = len(problem.weights)
    observations, allowances = [], []
    for column, query in enumerate(ids):
        depth, _ = problem.node_ids[query]
        width = n // 2**depth
        centered = means[:, column] - problem.reference[0]
        amplitude = math.sqrt(width) * centered.sum(-1) / 2
        detail = math.sqrt(width) * (centered[:, 0] - centered[:, 1]) / 2
        factor = math.sqrt(width) / (2 * n)
        observed = factor * np.maximum(
            0,
            np.abs(detail)
            - np.abs(
                amplitude
                - (problem.lower_bound - problem.reference[0]) * math.sqrt(width)
            ),
        )
        row_norm = 1 + math.sqrt(2) - math.sqrt(2) * 2 ** (-depth / 2)
        allowance = width / (2 * n) * float(error[column].max())
        allowance += factor * (row_norm + 1) * coefficient_error_bound
        lo, hi = problem.ranges[query]
        if np.any(observed < lo - allowance - 1e-10) or np.any(
            observed > hi + allowance + 1e-10
        ):
            raise ValueError(
                "Block observation violates the declared nominal support/error bound."
            )
        observations.append(observed)
        allowances.append(allowance)
    observed_means = np.asarray(observations).mean(axis=1)
    span = np.diff(problem.ranges[list(ids)], axis=1).ravel()
    radii = np.asarray(allowances) + span * np.sqrt(
        math.log(2 * len(ids) / failure_probability) / (2 * len(means))
    )
    intervals = {
        query: (float(mean - radius), float(mean + radius))
        for query, mean, radius in zip(ids, observed_means, radii, strict=True)
    }
    return {
        "intervals": intervals,
        "means": observed_means.tolist(),
        "radii": radii.tolist(),
        "statistic_definition": _statistic_definition(problem),
        "observation_allowances": allowances,
        "replicate_count": len(means),
        "failure_probability": failure_probability,
        "conditioning_key": conditioning_key,
        "status": "conditional_fixed_sample_simultaneous_moment_report",
        "assumptions": "iid_matched_replicates_fixed_dictionary_same_encoded_field_certified_representation_error",
        "marginal_membership_verified": False,
    }
