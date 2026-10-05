"""Finite-context Haar coupling with shared uncertain posterior weights.

Conditional amplitude programs are reused unchanged. Their probability tables
are scaled by context mass, so pooled observations and candidate loss contrasts
remain linear. The returned bounds do not establish source calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack

from Libs.utils.evaluation.wavelet_coupling_risk import _dual_lower_bound
from Libs.utils.evaluation.wavelet_dependence_queries import (
    HaarDependenceProblem,
    _statistic_definition,
)


@dataclass(frozen=True)
class PosteriorHaarProblem:
    conditional: tuple[tuple[HaarDependenceProblem, ...], ...]
    context_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    posterior_key: str
    nominal_weights: np.ndarray
    tv_radius: float
    matrix: csr_matrix
    rhs: np.ndarray
    inequality_matrix: csr_matrix
    inequality_rhs: np.ndarray
    slices: tuple[slice, ...]
    cell_columns: tuple[tuple[slice, ...], ...]
    weight_columns: slice
    lower_costs: np.ndarray
    upper_costs: np.ndarray

    @property
    def variable_count(self) -> int:
        return self.matrix.shape[1]


@dataclass(frozen=True)
class PosteriorMomentQuery:
    action_id: str
    node_index: int | None = None
    statistic: str = "innovation"


def _identifiers(values, count, name):
    values = tuple(values)
    if (
        len(values) != count
        or len(set(values)) != count
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValueError(f"Supply {count} distinct nonempty {name} identifiers.")
    return values


def compile_posterior_haar(
    conditional,
    nominal_weights,
    *,
    posterior_tv_radius: float,
    posterior_key: str,
    context_ids,
    action_ids,
    max_variables: int,
) -> PosteriorHaarProblem:
    """Lift context-by-action programs without imposing cross-action dependence.

    All candidates share context weights. Each conditional action law can be
    coupled arbitrarily with the others given that context. Additional shared
    physical disturbances or cross-action causal constraints are not supplied.
    """
    cells = tuple(tuple(row) for row in conditional)
    if not cells or not cells[0]:
        raise ValueError("Supply at least one context and one action.")
    count, actions = len(cells), len(cells[0])
    if any(len(row) != actions for row in cells) or any(
        not isinstance(cell, HaarDependenceProblem) for row in cells for cell in row
    ):
        raise ValueError("Conditional programs must form a context-by-action grid.")
    context_ids = _identifiers(context_ids, count, "context")
    action_ids = _identifiers(action_ids, actions, "action")
    weights = np.asarray(nominal_weights, dtype=float)
    if (
        weights.shape != (count,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Posterior weights must be finite nonnegative context masses.")
    if not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Posterior weights must sum to one.")
    if not np.isfinite(posterior_tv_radius) or not 0 <= posterior_tv_radius <= 1:
        raise ValueError("Posterior TV radius must lie in [0,1].")
    if not isinstance(posterior_key, str) or not posterior_key:
        raise ValueError("Supply a nonempty posterior conditioning key.")
    entries = sum(cell.variable_count for row in cells for cell in row)
    variables = entries + 2 * count
    if type(max_variables) is not int or max_variables < variables:
        raise ValueError("Posterior variable budget exceeded before allocation.")
    # Half absolute deviations have total mass <= TV radius <= 1. Together
    # with weights and scaled tables, every correction block has mass <= 1.
    wcols, dcols = slice(entries, entries + count), slice(entries + count, variables)
    matrices, weights_rows, slices, columns = [], [], [], []
    lower, upper = np.zeros((actions, variables)), np.zeros((actions, variables))
    offset = 0
    for z, row in enumerate(cells):
        row_columns = []
        for a, cell in enumerate(row):
            c = cell.coupling
            sl = slice(offset, offset + cell.variable_count)
            row_columns.append(sl)
            matrices.append(c.matrix)
            block = np.zeros((len(c.rhs), 2 * count))
            block[:, z] = -c.rhs
            weights_rows.append(csr_matrix(block))
            lower[a, sl], upper[a, sl] = c.lower_cost, c.upper_cost
            lower[a, entries + z] = upper[a, entries + z] = cell.offset
            slices.extend(slice(offset + t.start, offset + t.stop) for t in c.slices)
            offset = sl.stop
        columns.append(tuple(row_columns))
    matrix = hstack([block_diag(matrices), vstack(weights_rows)], format="csr")
    normalization = np.zeros((1, variables))
    normalization[0, wcols] = 1
    matrix = vstack([matrix, csr_matrix(normalization)], format="csr")
    rhs = np.r_[np.zeros(matrix.shape[0] - 1), 1.0]
    # Explicit scaled table normalization makes the zero-context face auditable.
    rows, cols, vals, mass_row = [], [], [], 0
    for z, row in enumerate(cells):
        for a, cell in enumerate(row):
            shift = columns[z][a].start
            for table in cell.coupling.slices:
                ids = list(range(shift + table.start, shift + table.stop))
                rows.extend([mass_row] * (len(ids) + 1))
                cols.extend([*ids, entries + z])
                vals.extend([*np.ones(len(ids)), -1.0])
                mass_row += 1
    mass = coo_matrix((vals, (rows, cols)), shape=(mass_row, variables)).tocsr()
    matrix, rhs = vstack([matrix, mass], format="csr"), np.r_[rhs, np.zeros(mass_row)]
    inequalities = np.zeros((2 * count + 1, variables))
    inequalities[:count, wcols] = 0.5 * np.eye(count)
    inequalities[count : 2 * count, wcols] = -0.5 * np.eye(count)
    inequalities[:count, dcols] = inequalities[count : 2 * count, dcols] = -np.eye(
        count
    )
    inequalities[-1, dcols] = 1
    inequality_rhs = np.r_[0.5 * weights, -0.5 * weights, posterior_tv_radius]
    return PosteriorHaarProblem(
        cells,
        context_ids,
        action_ids,
        posterior_key,
        weights.copy(),
        float(posterior_tv_radius),
        matrix,
        rhs,
        csr_matrix(inequalities),
        inequality_rhs,
        (*slices, wcols, dcols),
        tuple(columns),
        wcols,
        lower,
        upper,
    )


def posterior_moment_row(problem: PosteriorHaarProblem, query: PosteriorMomentQuery):
    """Pool the same observable statistic over contexts with unknown masses.

    A pooled report cannot silently use a different block, goal, reference or
    codebook in an unobserved context. Such a channel needs context labels and a
    separately specified observation model.
    """
    if (
        not isinstance(query, PosteriorMomentQuery)
        or query.action_id not in problem.action_ids
    ):
        raise ValueError("The query must name a declared posterior action.")
    if query.statistic not in {"innovation", "direct_goal"}:
        raise ValueError("Unknown pooled statistic.")
    if query.statistic == "direct_goal" and query.node_index is not None:
        raise ValueError("A direct-goal query has no node index.")
    a = problem.action_ids.index(query.action_id)
    row = np.zeros(problem.variable_count)
    definition = None
    for z, cells in enumerate(problem.conditional):
        cell = cells[a]
        current = _statistic_definition(cell)
        current.pop("conditioning_key")
        if query.statistic == "direct_goal":
            current["offset"] = cell.offset
        if definition is not None and current != definition:
            raise ValueError(
                "Pooled contexts must share the same observable statistic definition."
            )
        definition = current
        if query.statistic == "innovation":
            if type(query.node_index) is not int or not 0 <= query.node_index < len(
                cell.node_ids
            ):
                raise ValueError("Supply a valid common innovation node index.")
            row[problem.cell_columns[z][a]] = (
                cell.innovations[query.node_index].toarray().ravel()
            )
        else:
            row[problem.cell_columns[z][a]] = np.asarray(
                cell.innovations.sum(axis=0)
            ).ravel()
            row[problem.weight_columns.start + z] = cell.root_risk
    definition = {
        "posterior_key": problem.posterior_key,
        "context_ids": list(problem.context_ids),
        "action_id": query.action_id,
        "statistic": query.statistic,
        "node_index": query.node_index,
        "conditional_statistic": definition,
    }
    return csr_matrix(row[None]), definition


def _query_arrays(problem, queries, radii, conditional_errors):
    queries = tuple(queries)
    radii = np.asarray(radii, dtype=float)
    if (
        radii.shape != (len(queries),)
        or not np.isfinite(radii).all()
        or np.any(radii < 0)
    ):
        raise ValueError("Supply finite nonnegative radii for every pooled query.")
    if len(set(queries)) != len(queries):
        raise ValueError("Pooled queries must be distinct.")
    errors = (
        np.zeros((len(queries), len(problem.context_ids)))
        if conditional_errors is None
        else np.asarray(conditional_errors, dtype=float)
    )
    if (
        errors.shape != (len(queries), len(problem.context_ids))
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
    ):
        raise ValueError(
            "Conditional query errors must be nonnegative query-by-context bounds."
        )
    pairs = [posterior_moment_row(problem, q) for q in queries]
    rows = (
        vstack([x[0] for x in pairs], format="csr")
        if pairs
        else csr_matrix((0, problem.variable_count))
    )
    allowances = np.zeros((len(queries), problem.variable_count))
    allowances[:, problem.weight_columns] = errors
    return rows, csr_matrix(allowances), [x[1] for x in pairs], radii


def bind_posterior_report(
    problem, queries, intervals, *, conditional_query_errors=None
):
    """Bind externally justified pooled intervals; do not infer calibration."""
    queries = tuple(queries)
    values = np.asarray(intervals, dtype=float)
    if (
        values.shape != (len(queries), 2)
        or not np.isfinite(values).all()
        or np.any(values[:, 0] > values[:, 1])
    ):
        raise ValueError(
            "Pooled report intervals must be ordered finite endpoint pairs."
        )
    _, allowances, definitions, _ = _query_arrays(
        problem, queries, np.zeros(len(queries)), conditional_query_errors
    )
    return {
        "statistic_definitions": definitions,
        "intervals": values.tolist(),
        "conditional_query_errors": allowances[:, problem.weight_columns]
        .toarray()
        .tolist(),
    }


def _posterior_constraints(problem, report):
    if report is None:
        return problem.inequality_matrix, problem.inequality_rhs
    if not isinstance(report, dict) or "statistic_definitions" not in report:
        raise ValueError("A posterior report needs its bound statistic definitions.")
    definitions = report["statistic_definitions"]
    queries = tuple(
        PosteriorMomentQuery(x["action_id"], x["node_index"], x["statistic"])
        for x in definitions
    )
    bound = bind_posterior_report(
        problem,
        queries,
        report["intervals"],
        conditional_query_errors=report["conditional_query_errors"],
    )
    if bound["statistic_definitions"] != definitions:
        raise ValueError(
            "Posterior report statistic definitions do not match this population."
        )
    rows, error, _, _ = _query_arrays(
        problem, queries, np.zeros(len(queries)), report["conditional_query_errors"]
    )
    endpoints = np.asarray(report["intervals"], dtype=float).reshape(-1, 2)
    return (
        vstack([problem.inequality_matrix, rows - error, -rows - error], format="csr"),
        np.r_[problem.inequality_rhs, endpoints[:, 1], -endpoints[:, 0]],
    )


def _contrast_costs(problem, action_pair, conditional_loss_errors):
    if (
        len(action_pair) != 2
        or action_pair[0] == action_pair[1]
        or any(a not in problem.action_ids for a in action_pair)
    ):
        raise ValueError("Compare two distinct declared actions.")
    a, b = (problem.action_ids.index(value) for value in action_pair)
    errors = (
        np.zeros((len(problem.context_ids), len(problem.action_ids)))
        if conditional_loss_errors is None
        else np.asarray(conditional_loss_errors, dtype=float)
    )
    if (
        errors.shape != (len(problem.context_ids), len(problem.action_ids))
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
    ):
        raise ValueError(
            "Loss errors must be finite nonnegative context-by-action bounds."
        )
    lower = problem.lower_costs[a] - problem.upper_costs[b]
    upper = problem.upper_costs[a] - problem.lower_costs[b]
    lower = lower.copy()
    upper = upper.copy()
    lower[problem.weight_columns] -= errors[:, a] + errors[:, b]
    upper[problem.weight_columns] += errors[:, a] + errors[:, b]
    return lower, upper


def posterior_contrast_bounds(
    problem, action_pair, *, report=None, conditional_loss_errors=None
):
    """Bound J_first-J_second while retaining their common posterior weights."""
    constraints, rhs = _posterior_constraints(problem, report)
    lower, upper = _contrast_costs(problem, action_pair, conditional_loss_errors)
    lo, lr, lg = _dual_lower_bound(
        lower,
        problem.matrix,
        problem.rhs,
        problem.slices,
        inequality_matrix=constraints,
        inequality_rhs=rhs,
    )
    neg_hi, ur, ug = _dual_lower_bound(
        -upper,
        problem.matrix,
        problem.rhs,
        problem.slices,
        inequality_matrix=constraints,
        inequality_rhs=rhs,
    )
    hi = -neg_hi
    decision = action_pair[0] if hi < 0 else action_pair[1] if lo > 0 else "unresolved"
    return {
        "lower": lo,
        "upper": hi,
        "certified_order": decision,
        "conditional_minimax_regret": min(max(0.0, hi), max(0.0, -lo)),
        "primal_residual": max(lr, ur),
        "numerical_duality_gap": max(lg, ug),
        "posterior_key": problem.posterior_key,
        "reference_risk_qualified": False,
        "scope": "finite_posterior_shared_context_pairwise_contrast",
    }


def posterior_worst_report_regret(
    problem,
    action_pair,
    queries,
    report_radii,
    *,
    max_variables,
    current_report=None,
    conditional_query_errors=None,
    conditional_loss_errors=None,
):
    """Two feasible posterior worlds, each sharing weights across interventions."""
    n = problem.variable_count
    if type(max_variables) is not int or max_variables < 2 * n + 1:
        raise ValueError("Paired posterior variable budget exceeded before allocation.")
    rows, allowances, _, radii = _query_arrays(
        problem, queries, report_radii, conditional_query_errors
    )
    old, old_rhs = _posterior_constraints(problem, current_report)
    lower, upper = _contrast_costs(problem, action_pair, conditional_loss_errors)
    equalities = hstack(
        [
            block_diag([problem.matrix, problem.matrix]),
            csr_matrix((2 * len(problem.rhs), 1)),
        ],
        format="csr",
    )
    inequalities = hstack(
        [block_diag([old, old]), csr_matrix((2 * len(old_rhs), 1))], format="csr"
    )
    common = hstack(
        [rows - allowances, -rows - allowances, csr_matrix((len(radii), 1))],
        format="csr",
    )
    opposite = hstack(
        [-rows - allowances, rows - allowances, csr_matrix((len(radii), 1))],
        format="csr",
    )
    witness = np.zeros((2, 2 * n + 1))
    witness[0, :n], witness[1, n : 2 * n], witness[:, -1] = -upper, lower, 1
    inequalities = vstack(
        [inequalities, common, opposite, csr_matrix(witness)], format="csr"
    )
    objective = np.zeros(2 * n + 1)
    objective[-1] = -1
    radius = 1 + sum(
        float(np.maximum(np.abs(lower[sl]), np.abs(upper[sl])).max())
        for sl in problem.slices
    )
    slices = (
        *problem.slices,
        *(slice(sl.start + n, sl.stop + n) for sl in problem.slices),
    )
    value, residual, gap = _dual_lower_bound(
        objective,
        equalities,
        np.tile(problem.rhs, 2),
        slices,
        inequality_matrix=inequalities,
        inequality_rhs=np.r_[old_rhs, old_rhs, 2 * radii, 2 * radii, 0.0, 0.0],
        scalar_radius=radius,
    )
    return {
        "worst_report_regret": max(0.0, -value),
        "transport_variables": 2 * n + 1,
        "primal_residual": residual,
        "numerical_duality_gap": gap,
        "posterior_key": problem.posterior_key,
        "reference_risk_qualified": False,
        "scope": "deterministic_two_action_precommitted_population_query",
        "exactness": "nominal_exact_class_otherwise_outer_cost_and_report_envelopes",
    }


def select_posterior_query(
    problem,
    action_pair,
    query_sets,
    report_radii,
    query_costs,
    *,
    max_variables,
    current_report=None,
    conditional_query_errors=None,
    conditional_loss_errors=None,
):
    """Minimize worst-report pairwise regret plus cost over an explicit menu."""
    costs = np.asarray(query_costs, dtype=float)
    if (
        len(query_sets) != len(report_radii)
        or costs.shape != (len(query_sets),)
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
    ):
        raise ValueError("Align the finite query menu, radii and nonnegative costs.")
    errors = (
        (None,) * len(query_sets)
        if conditional_query_errors is None
        else tuple(conditional_query_errors)
    )
    if len(errors) != len(query_sets):
        raise ValueError("Align conditional query errors with the finite menu.")
    options = []
    for queries, radii, cost, query_errors in zip(
        ((), *query_sets),
        (np.empty(0), *report_radii),
        (0.0, *costs),
        (None, *errors),
        strict=True,
    ):
        item = posterior_worst_report_regret(
            problem,
            action_pair,
            queries,
            radii,
            max_variables=max_variables,
            current_report=current_report,
            conditional_query_errors=query_errors,
            conditional_loss_errors=conditional_loss_errors,
        )
        item.update(
            query_cost=float(cost), objective=item["worst_report_regret"] + float(cost)
        )
        options.append(item)
    selected = min(range(len(options)), key=lambda j: options[j]["objective"])
    return {
        "selected_index": selected,
        "selected": options[selected],
        "options": options,
    }
