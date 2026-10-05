"""Finite categorical experiments on exact Haar amplitude-flow tables.

Observation kernels and partitions are supplied explicitly; this module never
infers a physical likelihood or merges nearby floating-point statistics.
Count-based risk bounds use a conservative Pinsker coordinate-box outer
approximation to the anytime mixture-KL confidence class, not that exact class.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, vstack
from scipy.special import rel_entr

from Libs.utils.evaluation.wavelet_coupling_risk import _dual_lower_bound
from Libs.utils.evaluation.wavelet_dependence_queries import (
    HaarDependenceProblem,
    _statistic_definition,
)


@dataclass(frozen=True)
class HaarObservationChannel:
    node_index: int
    outcome_ids: tuple[str, ...]
    operator: csr_matrix
    definition: dict


def _exact_problem(problem):
    if not isinstance(problem, HaarDependenceProblem):
        raise ValueError("Supply a compiled Haar dependence problem.")
    if (
        problem.coefficient_bin_radius != 0
        or problem.affine_grid_deviation != 0
        or not np.array_equal(problem.codebook, problem.source_codebook)
    ):
        raise ValueError(
            "An exact observation likelihood cannot use coarsened or nonuniform nominal codebooks."
        )


def _outcomes(values, count):
    values = tuple(values)
    if (
        len(values) != count
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != count
    ):
        raise ValueError(
            "Supply distinct nonempty outcome identifiers for every kernel row."
        )
    return values


def _stochastic_kernel(kernel, cells):
    kernel = csr_matrix(kernel, dtype=float)
    kernel.sum_duplicates()
    kernel.eliminate_zeros()
    kernel.sort_indices()
    if (
        kernel.ndim != 2
        or kernel.shape[1] != cells
        or kernel.shape[0] < 1
        or not np.isfinite(kernel.data).all()
        or np.any(kernel.data < 0)
        or not np.allclose(
            np.asarray(kernel.sum(axis=0)).ravel(), 1, atol=1e-12, rtol=0
        )
    ):
        raise ValueError(
            "The known likelihood kernel must be finite, nonnegative and column-stochastic."
        )
    return kernel


def _definition(problem, node, outcomes, kernel):
    table = problem.coupling.tables[node]
    coordinates = kernel.tocoo()
    return {
        "problem": _statistic_definition(problem),
        "singleton_probabilities": problem.probabilities.tolist(),
        "variable_count": problem.variable_count,
        "node_index": node,
        "node_id": list(problem.node_ids[node]),
        "table_amplitudes": table.amplitudes.tolist(),
        "table_details": table.details.tolist(),
        "outcome_ids": list(outcomes),
        "known_kernel": {
            "shape": list(kernel.shape),
            "rows": coordinates.row.tolist(),
            "columns": coordinates.col.tolist(),
            "values": coordinates.data.tolist(),
        },
        "observation_scope": "one_declared_channel_on_each_fresh_replica",
    }


def compile_haar_observation_channel(
    problem: HaarDependenceProblem,
    node_index: int,
    *,
    labels=None,
    kernel=None,
    outcome_ids=None,
) -> HaarObservationChannel:
    """Map one table into categorical observations using a declared experiment.

    Default: observe its exact amplitude/detail cell. ``labels`` supplies one
    explicit deterministic outcome identifier per cell. Alternatively supply a
    known kernel R[outcome, cell] and its outcome identifiers. Numeric noise
    scales alone do not define a likelihood. No float-tolerance grouping occurs.
    """
    _exact_problem(problem)
    if type(node_index) is not int or not 0 <= node_index < len(problem.node_ids):
        raise ValueError("Supply a valid integer node index.")
    table = problem.coupling.tables[node_index]
    cells = table.columns.stop - table.columns.start
    if labels is not None and (kernel is not None or outcome_ids is not None):
        raise ValueError(
            "Supply either deterministic labels or a known kernel and outcome IDs."
        )
    if labels is not None:
        labels = tuple(labels)
        if len(labels) != cells or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError(
                "Supply one nonempty categorical label for every table cell."
            )
        outcomes = tuple(dict.fromkeys(labels))
        indices = {label: index for index, label in enumerate(outcomes)}
        block = coo_matrix(
            (np.ones(cells), ([indices[label] for label in labels], np.arange(cells))),
            shape=(len(outcomes), cells),
        ).tocsr()
    elif kernel is None:
        if outcome_ids is not None:
            raise ValueError(
                "Outcome identifiers without a declared kernel do not define a likelihood."
            )
        outcomes = tuple(f"cell_{index}" for index in range(cells))
        block = eye(cells, format="csr")
    else:
        block = _stochastic_kernel(kernel, cells)
        if outcome_ids is None:
            raise ValueError("A known kernel requires explicit outcome identifiers.")
        outcomes = _outcomes(outcome_ids, block.shape[0])
    coordinates = block.tocoo()
    operator = coo_matrix(
        (coordinates.data, (coordinates.row, coordinates.col + table.columns.start)),
        shape=(len(outcomes), problem.variable_count),
    ).tocsr()
    return HaarObservationChannel(
        node_index,
        outcomes,
        operator,
        _definition(problem, node_index, outcomes, block),
    )


def _validate_channel(problem, channel):
    _exact_problem(problem)
    if not isinstance(channel, HaarObservationChannel):
        raise ValueError("Supply a compiled Haar observation channel.")
    node = channel.node_index
    if type(node) is not int or not 0 <= node < len(problem.node_ids):
        raise ValueError("The channel node does not belong to this problem.")
    table = problem.coupling.tables[node]
    operator = csr_matrix(channel.operator)
    if (
        operator.shape != (len(channel.outcome_ids), problem.variable_count)
        or operator[:, : table.columns.start].nnz
        or operator[:, table.columns.stop :].nnz
    ):
        raise ValueError(
            "The observation operator must select only its declared table."
        )
    block = _stochastic_kernel(
        operator[:, table.columns], table.columns.stop - table.columns.start
    )
    outcomes = _outcomes(channel.outcome_ids, block.shape[0])
    if channel.definition != _definition(problem, node, outcomes, block):
        raise ValueError(
            "Channel binding differs in context, goal, codebook, topology, marginals or likelihood."
        )


def observation_probabilities(problem, channel, coupling_vector) -> np.ndarray:
    """Evaluate the declared experiment on a feasible compressed coupling law."""
    _validate_channel(problem, channel)
    law = np.asarray(coupling_vector, dtype=float)
    if (
        law.shape != (problem.variable_count,)
        or not np.isfinite(law).all()
        or np.any(law < 0)
        or np.max(
            np.abs(problem.coupling.matrix @ law - problem.coupling.rhs), initial=0
        )
        > 1e-10
    ):
        raise ValueError(
            "Supply a finite nonnegative law satisfying the amplitude-flow constraints."
        )
    result = np.asarray(channel.operator @ law).ravel()
    _probability_vector(result)
    return result


def _probability_vector(values):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Categorical probabilities must be finite, nonnegative and sum to one."
        )
    return values


def categorical_kl(probabilities, alternative) -> float:
    """Categorical KL; positive mass against a zero alternative gives infinity."""
    probabilities, alternative = (
        _probability_vector(probabilities),
        _probability_vector(alternative),
    )
    if probabilities.shape != alternative.shape:
        raise ValueError("Categorical probability alphabets must have the same shape.")
    return max(0.0, float(rel_entr(probabilities, alternative).sum()))


def _counts(counts, alphabet_size):
    values = np.asarray(counts)
    if (
        values.shape != (alphabet_size,)
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError(
            "Observed counts must be a nonnegative integer vector, not expectation counts."
        )
    result = [int(value) for value in values]
    total = sum(result)
    if total > 2**53:
        raise ValueError(
            "Count totals exceed exact integer representation in the float64 confidence calculation."
        )
    return result, total


def bind_haar_observation_counts(problem, channel, counts) -> dict:
    """Bind actual integer outcome counts to a fully declared observation channel."""
    _validate_channel(problem, channel)
    counts, total = _counts(counts, len(channel.outcome_ids))
    return {
        "channel_definition": copy.deepcopy(channel.definition),
        "counts": counts,
        "sample_count": total,
        "source": "declared_fresh_replica_observation_counts",
    }


def haar_observation_risk_bounds(
    problem: HaarDependenceProblem,
    channels,
    reports,
    *,
    alpha: float,
    decision_threshold: float | None = None,
) -> dict:
    """Outward LP risk bounds for a conservative anytime confidence outer set.

    The exact mixture-KL class uses sum_q n_q KL(p_hat_q || P_q x) <= beta,
    beta=log(1/alpha)+sum_q log binomial(n_q+K_q-1,K_q-1). Pinsker implies
    each coordinate lies within sqrt(beta/(2*n_q)) of its empirical frequency.
    Intersect these boxes with the original exact amplitude-flow constraints.
    Adaptive query selection is allowed; each query needs a fresh independent
    replica with the declared stationary likelihood. Those assumptions are not
    checked or calibrated by counts. Infeasible outer sets raise explicitly.
    """
    _exact_problem(problem)
    if np.ndim(alpha) != 0 or not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite scalar strictly between zero and one.")
    channels, reports = tuple(channels), tuple(reports)
    if len(channels) != len(reports):
        raise ValueError("Supply one bound count report per declared channel.")
    threshold = None
    if decision_threshold is not None:
        if np.ndim(decision_threshold) != 0 or not math.isfinite(
            float(decision_threshold)
        ):
            raise ValueError("The decision threshold must be finite.")
        threshold = float(decision_threshold)
    totals, count_vectors, definitions = [], [], []
    beta = -math.log(float(alpha))
    for channel, report in zip(channels, reports, strict=True):
        _validate_channel(problem, channel)
        if any(channel.definition == definition for definition in definitions):
            raise ValueError(
                "Channels must be distinct; aggregate repeated observations before binding."
            )
        definitions.append(channel.definition)
        if (
            not isinstance(report, dict)
            or report.get("channel_definition") != channel.definition
        ):
            raise ValueError(
                "Observation report binding does not match the declared channel."
            )
        counts, total = _counts(report.get("counts"), len(channel.outcome_ids))
        if (
            type(report.get("sample_count")) is not int
            or report["sample_count"] != total
        ):
            raise ValueError(
                "Observation report sample_count does not match its integer counts."
            )
        # Stable log binomial, avoiding subtraction of large log-factorials.
        beta += math.fsum(math.log1p(total / j) for j in range(1, len(counts)))
        totals.append(total)
        count_vectors.append(counts)
    beta = float(
        np.nextafter(beta + 64 * np.finfo(float).eps * (1 + abs(beta)), np.inf)
    )
    rows, rhs, intervals = [], [], []
    for channel, counts, total in zip(channels, count_vectors, totals, strict=True):
        if total:
            empirical = np.asarray(counts, dtype=float) / total
            radius = math.sqrt(beta / (2 * total))
            radius += 64 * np.finfo(float).eps * (1 + radius)
            lower = np.maximum(0.0, np.nextafter(empirical - radius, -np.inf))
            upper = np.minimum(1.0, np.nextafter(empirical + radius, np.inf))
        else:
            lower, upper = np.zeros(len(counts)), np.ones(len(counts))
        intervals.append(np.column_stack((lower, upper)).tolist())
        rows.extend((channel.operator, -channel.operator))
        rhs.extend((upper, -lower))
    inequalities = (
        vstack(rows, format="csr") if rows else csr_matrix((0, problem.variable_count))
    )
    inequality_rhs = np.concatenate(rhs) if rhs else np.empty(0)
    coupling = problem.coupling
    lower, lr, lg = _dual_lower_bound(
        coupling.lower_cost,
        coupling.matrix,
        coupling.rhs,
        coupling.slices,
        inequality_matrix=inequalities,
        inequality_rhs=inequality_rhs,
    )
    negative_upper, ur, ug = _dual_lower_bound(
        -coupling.upper_cost,
        coupling.matrix,
        coupling.rhs,
        coupling.slices,
        inequality_matrix=inequalities,
        inequality_rhs=inequality_rhs,
    )
    lower = float(np.nextafter(problem.offset + lower, -np.inf))
    upper = float(np.nextafter(problem.offset - negative_upper, np.inf))
    decision = (
        "not_requested"
        if threshold is None
        else "below"
        if upper < threshold
        else "above"
        if lower > threshold
        else "unresolved"
    )
    return {
        "lower": lower,
        "upper": upper,
        "decision": decision,
        "confidence_level": 1 - float(alpha),
        "mixture_kl_threshold": beta,
        "sample_counts": totals,
        "coordinate_intervals": intervals,
        "confidence_set": "conservative_pinsker_coordinate_outer_of_anytime_mixture_kl",
        "exact_kl_region_solved": False,
        "primal_residual": max(lr, ur),
        "numerical_duality_gap": max(lg, ug),
        "calibration_established": False,
        "iid_validated": False,
        "scope": "finite_exact_haar_fresh_replica_observation_experiments",
        "assumptions": (
            "Fixed singleton marginals and the exact Haar amplitude-flow model are valid.",
            "Each selected channel observes a fresh independent replica under its declared stationary likelihood.",
            "Reported counts are actual categorical observations, not expected or fractional counts.",
            "Known kernels and outcome labels represent the actual physical observation experiment.",
        ),
    }
