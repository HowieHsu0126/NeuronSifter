"""Query-preserving Haar compression for finite integer coefficient alphabets.

A declared ancestral query skeleton uses exact Q(sqrt(2)) amplitude keys.
Other internal nodes use deterministic nearest-grid states; leaves are never
rounded. Local flows keep the original coefficient marginals. A finite memory
lift retains repeated deterministic readouts of one coefficient realization.
The solver supplies numerically checked LP bounds, not interval-arithmetic
certificates or learned-model calibration.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack

from Libs.utils.evaluation.wavelet_coupling_risk import _dual_lower_bound


@dataclass(frozen=True)
class QueryHaarTable:
    """Raw-amplitude states and deterministic child maps.

    Internal transitions hold child-state indices; final-level transitions hold
    unrounded leaf fluctuations before adding the supplied physical reference.
    """

    node_id: tuple[int, int]
    state_keys: tuple
    states_numeric: np.ndarray
    mean_values: np.ndarray
    details: np.ndarray
    columns: slice
    transitions: tuple[np.ndarray, np.ndarray]
    exact: bool
    grid_width: float
    mean_error_radius: float


@dataclass(frozen=True)
class QueryPreservingHaarProblem:
    probabilities: np.ndarray
    codebook: np.ndarray
    reference: np.ndarray
    weights: np.ndarray
    lower_bound: float
    offset: float
    epsilon: float
    rounding: str
    protected_nodes: tuple[tuple[int, int], ...]
    tables: tuple[QueryHaarTable, ...]
    matrix: csr_matrix
    rhs: np.ndarray
    flow_matrix: csr_matrix
    slices: tuple[slice, ...]
    surrogate_cost: np.ndarray
    error_cost: np.ndarray
    leaf_error_radii: np.ndarray

    @property
    def variable_count(self):
        return self.matrix.shape[1]

    @property
    def node_ids(self):
        return tuple(table.node_id for table in self.tables)


@dataclass(frozen=True)
class QueryThresholdReadout:
    problem: QueryPreservingHaarProblem
    node_id: tuple[int, int]
    amplitude_coefficient: Fraction
    detail_coefficient: Fraction
    threshold: Fraction
    labels: np.ndarray


@dataclass(frozen=True)
class QueryMemoryProblem:
    problem: QueryPreservingHaarProblem
    readouts: tuple[QueryThresholdReadout, ...]
    memory_ids: tuple[tuple[int, ...], ...]
    memory_columns: tuple[slice, ...]
    matrix: csr_matrix
    rhs: np.ndarray
    slices: tuple[slice, ...]
    probability_rows: csr_matrix

    @property
    def variable_count(self):
        return self.matrix.shape[1]

    @property
    def memory_count(self):
        return len(self.memory_ids)


def _readonly(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _readonly_sparse(matrix):
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return result


def _rational(value, name):
    if isinstance(value, Fraction):
        return value
    _finite_scalar(value, name)
    # Decimal spellings define threshold coefficients; state keys stay rational.
    return Fraction(str(value))


def _q_value(key):
    return float(key[0]) + math.sqrt(2) * float(key[1])


def _q_nonnegative(a, b):
    if b == 0:
        return a >= 0
    if a == 0:
        return b > 0
    if (a > 0) == (b > 0):
        return a > 0
    return (a > 0) if a * a > 2 * b * b else (b > 0)


def _node_id(value, levels):
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(part) is not int for part in value)
        or not 0 <= value[0] < levels
        or not 0 <= value[1] < 2 ** value[0]
    ):
        raise ValueError("A query node must be an internal (depth, position) pair.")
    return value


def _grid_key(value, width):
    ratio = value / width
    if not math.isfinite(ratio):
        raise ValueError(
            "The rounding grid cannot represent the requested state scale."
        )
    # Halfway values round upward; each grid cell has radius width/2.
    return math.floor(ratio + 0.5)


def compile_query_preserving_haar(
    probabilities,
    codebook,
    reference,
    weights,
    *,
    lower_bound,
    offset,
    epsilon,
    protected_nodes,
    rounding,
    max_variables,
):
    """Compile one balanced Haar stream with original singleton marginals.

    Coefficients are ordered [root amplitude, details in breadth-first order].
    The alphabet is uniformly spaced, strictly increasing, and integer valued.
    protected_nodes closes upward to the root. With physical_mean rounding,
    h_l=2*epsilon*gamma**l/sum_{j=1}^{L-1} gamma**j, gamma=2**(3/4).
    The amplitude grid uses the finite-depth contraction sum, with maximum
    leaf error epsilon. Returned error radii are
    analytic real-arithmetic envelopes evaluated in float64.
    """
    p = np.asarray(probabilities, dtype=float)
    source_code = np.asarray(codebook)
    if (
        p.ndim != 2
        or p.shape[0] < 2
        or p.shape[0] & (p.shape[0] - 1)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Probabilities require a normalized finite nonnegative balanced N-by-K array."
        )
    n, k = p.shape
    if source_code.shape != (k,) or k < 2:
        raise ValueError(
            "The codebook must have at least two uniformly spaced exact integer values."
        )
    rational_code = tuple(_rational(value, "codebook") for value in source_code)
    if (
        any(value.denominator != 1 or abs(value) > 2**53 for value in rational_code)
        or rational_code[1] <= rational_code[0]
        or any(
            right - left != rational_code[1] - rational_code[0]
            for left, right in zip(rational_code[:-1], rational_code[1:], strict=True)
        )
    ):
        raise ValueError(
            "The codebook must have at least two uniformly spaced exact integer values."
        )
    code = np.asarray(rational_code, dtype=float)
    ref, w = np.asarray(reference, dtype=float), np.asarray(weights, dtype=float)
    if (
        ref.shape != (n,)
        or w.shape != (n,)
        or not np.isfinite(ref).all()
        or not np.isfinite(w).all()
    ):
        raise ValueError("Reference and weights must be finite N-vectors.")
    floor = _finite_scalar(lower_bound, "lower_bound")
    constant = _finite_scalar(offset, "offset")
    eps = _finite_scalar(epsilon, "epsilon")
    if eps <= 0:
        raise ValueError("epsilon must be positive.")
    if rounding not in ("amplitude", "physical_mean"):
        raise ValueError("rounding must be amplitude or physical_mean.")
    if type(max_variables) is not int or max_variables < k * k + (n - 2) * k:
        raise ValueError("Haar variable budget exceeded before allocation.")
    levels = n.bit_length() - 1
    exact = {(0, 0)}
    try:
        declared = tuple(protected_nodes)
    except TypeError as exc:
        raise ValueError("protected_nodes must contain internal node pairs.") from exc
    for value in declared:
        depth, position = _node_id(value, levels)
        while depth:
            exact.add((depth, position))
            depth, position = depth - 1, position // 2
    gamma = 2**0.75
    normalizer = sum(gamma**depth for depth in range(1, levels))
    amplitude_grid = (
        2 * eps * (math.sqrt(2) - 1) / (1 - 2 ** (-(levels - 1) / 2))
        if levels > 1
        else 0.0
    )
    grid = {
        depth: amplitude_grid
        if rounding == "amplitude"
        else 2 * eps * gamma**depth / normalizer
        for depth in range(1, levels)
    }
    if any(not math.isfinite(width) or width <= 0 for width in grid.values()):
        raise ValueError(
            "The rounding grid must have finite positive representable widths."
        )
    integers = tuple(int(value) for value in code)
    states = {(0, 0): tuple((Fraction(value), Fraction(0)) for value in integers)}
    errors = {(0, 0): 0.0}
    tables, width, planned_width = [], 0, k * k
    leaf_errors = np.zeros(n)
    for depth in range(levels):
        size = n // 2**depth
        for position in range(2**depth):
            node = (depth, position)
            keys = states[node]
            cells = len(keys) * k
            if cells > max_variables - width:
                raise ValueError(
                    "Haar variable budget exceeded before table allocation."
                )
            is_exact = node in exact
            if is_exact:
                raw = np.array([_q_value(key) for key in keys])
                means = raw / math.sqrt(size)
            elif rounding == "amplitude":
                raw = np.array(keys, dtype=float) * grid[depth]
                means = raw / math.sqrt(size)
            else:
                means = np.array(keys, dtype=float) * grid[depth]
                raw = means * math.sqrt(size)
            if not np.isfinite(raw).all() or not np.isfinite(means).all():
                raise ValueError("Haar state values exceed the finite numerical range.")
            transitions = []
            for branch, sign in enumerate((1, -1)):
                child = (depth + 1, 2 * position + branch)
                child_values = []
                for index, key in enumerate(keys):
                    for detail in integers:
                        if depth + 1 == levels:
                            child_values.append(
                                means[index] + sign * detail / math.sqrt(size)
                            )
                        elif child in exact:
                            a, b = key
                            child_values.append((b, (a + sign * detail) / 2))
                        elif rounding == "amplitude":
                            value = (raw[index] + sign * detail) / math.sqrt(2)
                            child_values.append(_grid_key(value, grid[depth + 1]))
                        else:
                            value = means[index] + sign * detail / math.sqrt(size)
                            child_values.append(_grid_key(value, grid[depth + 1]))
                if depth + 1 == levels:
                    transitions.append(
                        _readonly(np.asarray(child_values).reshape(len(keys), k))
                    )
                    leaf_errors[child[1]] = errors[node]
                else:
                    distinct = set()
                    for value in child_values:
                        if value not in distinct:
                            if planned_width + k > max_variables:
                                raise ValueError(
                                    "Haar variable budget exceeded before reachable-state allocation."
                                )
                            distinct.add(value)
                            planned_width += k
                    child_keys = tuple(sorted(distinct))
                    states[child] = child_keys
                    lookup = {value: index for index, value in enumerate(child_keys)}
                    transitions.append(
                        _readonly(
                            np.array([lookup[value] for value in child_values]).reshape(
                                len(keys), k
                            ),
                            int,
                        )
                    )
                    if child in exact:
                        errors[child] = 0.0
                    elif rounding == "amplitude":
                        errors[child] = errors[node] + grid[depth + 1] / (
                            2 * math.sqrt(size / 2)
                        )
                    else:
                        errors[child] = errors[node] + grid[depth + 1] / 2
            table = QueryHaarTable(
                node,
                keys,
                _readonly(raw),
                _readonly(means),
                _readonly(code),
                slice(width, width + cells),
                tuple(transitions),
                is_exact,
                0.0 if is_exact else grid[depth],
                errors[node],
            )
            tables.append(table)
            width += cells
    if np.max(leaf_errors) > eps * (1 + 64 * np.finfo(float).eps):
        raise ArithmeticError(
            "The leaf error envelope exceeds the requested tolerance."
        )
    rows, columns, values, rhs = [], [], [], []

    def add(indices, coefficients, target):
        row = len(rhs)
        indices, coefficients = list(indices), list(coefficients)
        rows.extend([row] * len(indices))
        columns.extend(indices)
        values.extend(coefficients)
        rhs.append(target)

    root = tables[0]
    for index in range(k):
        indices = range(
            root.columns.start + index * k, root.columns.start + (index + 1) * k
        )
        add(indices, [1.0] * k, p[0, index])
    for table in tables:
        depth, position = table.node_id
        coefficient = 2**depth + position
        for index in range(k):
            indices = range(table.columns.start + index, table.columns.stop, k)
            add(indices, [1.0] * len(table.state_keys), p[coefficient, index])
    flow_start = len(rhs)
    by_node = {table.node_id: table for table in tables}
    for table in tables[1:]:
        depth, position = table.node_id
        parent = by_node[(depth - 1, position // 2)]
        mapping = parent.transitions[position % 2].ravel()
        # Group incoming cells once; scanning all parent cells per state is quadratic.
        order = np.argsort(mapping, kind="stable")
        boundaries = np.r_[0, np.cumsum(np.bincount(mapping, minlength=len(table.state_keys)))]
        for state in range(len(table.state_keys)):
            own = list(
                range(
                    table.columns.start + state * k,
                    table.columns.start + (state + 1) * k,
                )
            )
            incoming = (
                order[boundaries[state] : boundaries[state + 1]] + parent.columns.start
            ).tolist()
            add(own + incoming, [1.0] * k + [-1.0] * len(incoming), 0.0)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(rhs), width)).tocsr()
    cost, error_cost = np.zeros(width), np.zeros(width)
    for table in tables:
        depth, position = table.node_id
        if depth + 1 == levels:
            for branch in range(2):
                leaf = 2 * position + branch
                cost[table.columns] += w[leaf] * np.maximum(
                    floor, ref[leaf] + table.transitions[branch].ravel()
                )
                error_cost[table.columns] += abs(w[leaf]) * leaf_errors[leaf]
    if not np.isfinite(cost).all() or not np.isfinite(error_cost).all():
        raise ValueError(
            "The loss or error envelope exceeds the finite numerical range."
        )
    return QueryPreservingHaarProblem(
        _readonly(p),
        _readonly(code),
        _readonly(ref),
        _readonly(w),
        floor,
        constant,
        eps,
        rounding,
        tuple(sorted(exact)),
        tuple(tables),
        _readonly_sparse(matrix),
        _readonly(rhs),
        _readonly_sparse(matrix[flow_start:]),
        tuple(table.columns for table in tables),
        _readonly(cost),
        _readonly(error_cost),
        _readonly(leaf_errors),
    )


def compile_query_readout(
    problem,
    node_id,
    *,
    amplitude_coefficient=1.0,
    detail_coefficient=0.0,
    threshold=0.0,
):
    """Bind 1{a*A_v+b*D_v >= threshold} to an exact retained query node.

    Decimal spellings of supplied finite numbers define rational coefficients.
    Comparisons use exact rational arithmetic in Q(sqrt(2)), including ties.
    """
    if not isinstance(problem, QueryPreservingHaarProblem):
        raise ValueError("Supply a compiled query-preserving Haar problem.")
    node = _node_id(node_id, len(problem.reference).bit_length() - 1)
    if node not in problem.protected_nodes:
        raise ValueError(
            "Readouts must belong to the declared exact protected query skeleton."
        )
    a = _rational(amplitude_coefficient, "amplitude_coefficient")
    b = _rational(detail_coefficient, "detail_coefficient")
    threshold = _rational(threshold, "threshold")
    table = problem.tables[problem.node_ids.index(node)]
    labels = [
        int(_q_nonnegative(a * state[0] + b * int(detail) - threshold, a * state[1]))
        for state in table.state_keys
        for detail in table.details
    ]
    return QueryThresholdReadout(problem, node, a, b, threshold, _readonly(labels, int))


def compile_query_memory(problem, readouts, *, max_variables):
    """Replicate the compressed flow once per deterministic binary query memory.

    Coefficient marginals are imposed after aggregation over memories. Each
    memory has its own homogeneous flows, equal table masses, and readout
    support; conditional coefficient marginals remain free.
    """
    if not isinstance(problem, QueryPreservingHaarProblem):
        raise ValueError("Supply a compiled query-preserving Haar problem.")
    width = problem.variable_count
    if type(max_variables) is not int or max_variables < width:
        raise ValueError("Query memory variable budget exceeded before allocation.")
    readouts = tuple(readouts)
    count = 1
    for readout in readouts:
        if (
            not isinstance(readout, QueryThresholdReadout)
            or readout.problem is not problem
        ):
            raise ValueError("A readout must be bound to this same Haar problem.")
        expected = compile_query_readout(
            problem,
            readout.node_id,
            amplitude_coefficient=readout.amplitude_coefficient,
            detail_coefficient=readout.detail_coefficient,
            threshold=readout.threshold,
        )
        if not np.array_equal(expected.labels, readout.labels):
            raise ValueError(
                "Readout labels disagree with its exact threshold binding."
            )
        count *= 2
        if count > max_variables // width:
            raise ValueError("Query memory variable budget exceeded before allocation.")
    ids = tuple(itertools.product((0, 1), repeat=len(readouts)))
    memory_columns = tuple(
        slice(index * width, (index + 1) * width) for index in range(count)
    )
    root = problem.tables[0]
    mass_rows, mass_cols, mass_values = [], [], []
    for row, table in enumerate(problem.tables[1:]):
        for selected, sign in ((table.columns, 1), (root.columns, -1)):
            cols = list(range(selected.start, selected.stop))
            mass_rows.extend([row] * len(cols))
            mass_cols.extend(cols)
            mass_values.extend([sign] * len(cols))
    mass = coo_matrix(
        (mass_values, (mass_rows, mass_cols)), shape=(len(problem.tables) - 1, width)
    ).tocsr()
    local = vstack((problem.flow_matrix, mass), format="csr")
    aggregate = hstack([problem.matrix] * count, format="csr")
    conditional = block_diag([local] * count, format="csr")
    forbidden = []
    for index, memory in enumerate(ids):
        allowed = np.ones(width, dtype=bool)
        for readout, observed in zip(readouts, memory, strict=True):
            table = problem.tables[problem.node_ids.index(readout.node_id)]
            allowed[table.columns] &= readout.labels == observed
        forbidden.extend((np.flatnonzero(~allowed) + index * width).tolist())
    support = coo_matrix(
        (np.ones(len(forbidden)), (np.arange(len(forbidden)), forbidden)),
        shape=(len(forbidden), count * width),
    ).tocsr()
    matrix = vstack((aggregate, conditional, support), format="csr")
    rhs = np.r_[problem.rhs, np.zeros(conditional.shape[0] + support.shape[0])]
    p_rows, p_cols = [], []
    for index in range(count):
        columns = list(
            range(index * width + root.columns.start, index * width + root.columns.stop)
        )
        p_rows.extend([index] * len(columns))
        p_cols.extend(columns)
    probabilities = coo_matrix(
        (np.ones(len(p_cols)), (p_rows, p_cols)), shape=(count, count * width)
    ).tocsr()
    slices = tuple(
        slice(index * width + sl.start, index * width + sl.stop)
        for index in range(count)
        for sl in problem.slices
    )
    return QueryMemoryProblem(
        problem,
        readouts,
        ids,
        memory_columns,
        _readonly_sparse(matrix),
        _readonly(rhs),
        slices,
        _readonly_sparse(probabilities),
    )


def query_memory_bounds(memory, event_memory_ids=None):
    """Bound event mass and unnormalized surrogate/true event-weighted losses.

    The loss is offset + sum_i w_i max(lower_bound, reference_i + X_i).
    True-loss envelopes optimize surrogate cost plus/minus the event-weighted
    rounding envelope jointly. They do not multiply separately optimized event
    and loss bounds. Float64 state arithmetic and LP repair are not rigorous
    interval arithmetic.
    """
    if not isinstance(memory, QueryMemoryProblem):
        raise ValueError("Supply a compiled query memory problem.")
    selected = tuple(range(memory.memory_count))
    if event_memory_ids is not None:
        supplied = tuple(event_memory_ids)
        if any(
            not isinstance(value, tuple)
            or any(type(part) is not int for part in value)
            or value not in memory.memory_ids
            for value in supplied
        ):
            raise ValueError(
                "Event identifiers must be declared full binary memory tuples."
            )
        selected = tuple(memory.memory_ids.index(value) for value in supplied)
        if len(set(selected)) != len(selected):
            raise ValueError("Event memory identifiers must be distinct.")
    probability = np.asarray(
        memory.probability_rows[list(selected)].sum(axis=0)
    ).ravel()
    cost, error = np.zeros(memory.variable_count), np.zeros(memory.variable_count)
    for index in selected:
        cost[memory.memory_columns[index]] = memory.problem.surrogate_cost
        error[memory.memory_columns[index]] = memory.problem.error_cost
    cost += memory.problem.offset * probability
    objectives = (probability, -probability, cost, -cost, cost - error, -cost - error)
    solved = [
        _dual_lower_bound(value, memory.matrix, memory.rhs, memory.slices)
        for value in objectives
    ]
    endpoints = [value[0] for value in solved]
    return {
        "probability_lower": max(0.0, endpoints[0]),
        "probability_upper": min(1.0, -endpoints[1]),
        "surrogate_event_loss_lower": endpoints[2],
        "surrogate_event_loss_upper": -endpoints[3],
        "true_event_loss_lower": endpoints[4],
        "true_event_loss_upper": -endpoints[5],
        "primal_residual": max(value[1] for value in solved),
        "numerical_duality_gap": max(value[2] for value in solved),
        "leaf_error_max": float(memory.problem.leaf_error_radii.max()),
        "loss_error_radius": float(
            np.abs(memory.problem.weights) @ memory.problem.leaf_error_radii
        ),
        "transport_variables": memory.variable_count,
        "memory_count": memory.memory_count,
        "conditional_coefficient_marginals_imposed": False,
        "floating_point_interval_certified": False,
        "scope": "balanced_integer_haar_query_skeleton_with_unobserved_state_rounding",
    }
