"""Finite memory lift for repeated deterministic queries on one Haar replica.

Original coefficient marginals hold only after summing over query memories.
Each memory slice obeys homogeneous amplitude flows and its deterministic
readout support. Conditional-on-memory coefficient marginals are not imposed.
The implementation covers one balanced exact-grid temporal stream.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix, eye, hstack, vstack

from Libs.utils.evaluation.haar_observation_experiments import (
    HaarObservationChannel,
    _exact_problem,
    _validate_channel,
)
from Libs.utils.evaluation.wavelet_coupling_risk import _dual_lower_bound
from Libs.utils.evaluation.wavelet_dependence_queries import HaarDependenceProblem


@dataclass(frozen=True)
class JointHaarObservationProblem:
    problem: HaarDependenceProblem
    channels: tuple[HaarObservationChannel, ...]
    memory_ids: tuple[tuple[str, ...], ...]
    memory_columns: tuple[slice, ...]
    matrix: csr_matrix
    rhs: np.ndarray
    slices: tuple[slice, ...]
    probability_rows: csr_matrix
    root_columns: slice

    @property
    def variable_count(self) -> int:
        return self.matrix.shape[1]

    @property
    def memory_count(self) -> int:
        return len(self.memory_ids)


def _balanced_root(problem):
    _exact_problem(problem)
    count = len(problem.reference)
    tables = problem.coupling.tables
    roots = [table for table in tables if table.depth == 0]
    if (
        count < 2
        or count & (count - 1)
        or problem.sizes[0] != 1
        or len(roots) != 1
        or roots[0].position != 0
        or problem.lengths
        != tuple(count // 2**level for level in range(count.bit_length() - 1))
    ):
        raise ValueError(
            "Joint observation requires one balanced temporal stream with exactly one root."
        )
    if tuple((table.depth, table.position) for table in tables) != problem.node_ids:
        raise ValueError("Haar table topology and node identities do not agree.")
    return roots[0]


def compile_joint_observation(
    problem: HaarDependenceProblem,
    channels,
    *,
    max_variables: int,
) -> JointHaarObservationProblem:
    """Compile D copies of the M-variable flow model, guarded before allocation.

    D is the product of all declared deterministic outcome alphabet sizes.
    Multiple channels may read the same node, including repeated identical
    channels; incompatible memory combinations acquire zero probability.
    Stochastic kernels are rejected because one fixed memory cannot represent
    unspecified repeated noise draws.
    """
    root = _balanced_root(problem)
    channels = tuple(channels)
    original = problem.coupling
    width = problem.variable_count
    if type(max_variables) is not int or max_variables < width:
        raise ValueError(
            "Joint observation variable budget exceeded before allocation."
        )
    count, label_indices = 1, []
    for channel in channels:
        _validate_channel(problem, channel)
        count *= len(channel.outcome_ids)
        if count > max_variables // width:
            raise ValueError(
                "Joint observation variable budget exceeded before allocation."
            )
        table = original.tables[channel.node_index]
        block = channel.operator[:, table.columns].tocsc()
        if np.any(np.diff(block.indptr) != 1) or np.any(block.data != 1.0):
            raise ValueError(
                "Joint memory requires deterministic one-unit-probability readouts; stochastic likelihoods are unsupported."
            )
        label_indices.append(block.indices.copy())
    variables = count * width
    memory_indices = tuple(
        itertools.product(*(range(len(channel.outcome_ids)) for channel in channels))
    )
    memory_ids = tuple(
        tuple(
            channel.outcome_ids[index]
            for channel, index in zip(channels, memory, strict=True)
        )
        for memory in memory_indices
    )
    columns = tuple(slice(index * width, (index + 1) * width) for index in range(count))
    aggregate = hstack([original.matrix] * count, format="csr")
    coordinates = original.matrix.tocoo()
    flow_ids = np.unique(coordinates.row[coordinates.data < 0])
    if np.any(original.rhs[flow_ids] != 0):
        raise ValueError("Negative-entry rows must be homogeneous amplitude flows.")
    flow = original.matrix[flow_ids]
    # True single-root projections give every table the same per-memory mass.
    # These redundant normalizations make each dual-correction block's mass
    # bound explicit, including zero-probability memory slices.
    mass_rows, mass_cols, mass_values, row = [], [], [], 0
    for table in original.tables:
        if table.columns == root.columns:
            continue
        for selected, sign in ((table.columns, 1.0), (root.columns, -1.0)):
            indices = range(selected.start, selected.stop)
            mass_rows.extend([row] * len(indices))
            mass_cols.extend(indices)
            mass_values.extend([sign] * len(indices))
        row += 1
    mass = coo_matrix((mass_values, (mass_rows, mass_cols)), shape=(row, width)).tocsr()
    local = vstack((flow, mass), format="csr")
    per_memory = block_diag([local] * count, format="csr")
    forbidden = []
    for index, memory in enumerate(memory_indices):
        allowed = np.ones(width, dtype=bool)
        for channel, labels, observed in zip(
            channels, label_indices, memory, strict=True
        ):
            table = original.tables[channel.node_index]
            allowed[table.columns] &= labels == observed
        forbidden.extend((np.flatnonzero(~allowed) + index * width).tolist())
    support = coo_matrix(
        (np.ones(len(forbidden)), (np.arange(len(forbidden)), forbidden)),
        shape=(len(forbidden), variables),
    ).tocsr()
    matrix = vstack((aggregate, per_memory, support), format="csr")
    rhs = np.r_[original.rhs, np.zeros(per_memory.shape[0] + support.shape[0])]
    probability_rows, probability_cols = [], []
    for index in range(count):
        ids = range(
            index * width + root.columns.start, index * width + root.columns.stop
        )
        probability_rows.extend([index] * len(ids))
        probability_cols.extend(ids)
    probabilities = coo_matrix(
        (np.ones(len(probability_cols)), (probability_rows, probability_cols)),
        shape=(count, variables),
    ).tocsr()
    slices = tuple(
        slice(index * width + sl.start, index * width + sl.stop)
        for index in range(count)
        for sl in original.slices
    )
    return JointHaarObservationProblem(
        problem,
        channels,
        memory_ids,
        columns,
        matrix,
        rhs,
        slices,
        probabilities,
        root.columns,
    )


def _memory_index(lift, value):
    if not isinstance(value, tuple) or value not in lift.memory_ids:
        raise ValueError(
            "A memory identifier must be a declared full tuple of outcome identifiers."
        )
    return lift.memory_ids.index(value)


def _joint_constraints(lift, prior_table_law, joint_probability_intervals):
    matrix, rhs = lift.matrix, lift.rhs
    original = lift.problem.coupling
    if prior_table_law is not None:
        prior = np.asarray(prior_table_law, dtype=float)
        if (
            prior.shape != (lift.problem.variable_count,)
            or not np.isfinite(prior).all()
            or np.any(prior < 0)
            or np.max(np.abs(original.matrix @ prior - original.rhs), initial=0) > 1e-10
        ):
            raise ValueError(
                "Known aggregate tables must be a feasible finite nonnegative original table law."
            )
        aggregate = hstack(
            [eye(lift.problem.variable_count, format="csr")] * lift.memory_count,
            format="csr",
        )
        matrix, rhs = vstack((matrix, aggregate), format="csr"), np.r_[rhs, prior]
    rows, endpoints = [], []
    if joint_probability_intervals is not None:
        if not isinstance(joint_probability_intervals, dict):
            raise ValueError(
                "Joint probability intervals must map declared memory tuples to endpoint pairs."
            )
        for memory, interval in joint_probability_intervals.items():
            index = _memory_index(lift, memory)
            values = np.asarray(interval, dtype=float)
            if (
                values.shape != (2,)
                or not np.isfinite(values).all()
                or not 0 <= values[0] <= values[1] <= 1
            ):
                raise ValueError(
                    "Joint probability intervals must be ordered finite pairs in [0,1]."
                )
            rows.extend((lift.probability_rows[index], -lift.probability_rows[index]))
            endpoints.extend((values[1], -values[0]))
    inequalities = (
        vstack(rows, format="csr") if rows else csr_matrix((0, lift.variable_count))
    )
    return matrix, rhs, inequalities, np.asarray(endpoints)


def joint_observation_bounds(
    lift: JointHaarObservationProblem,
    event_memory_ids=None,
    *,
    prior_table_law=None,
    joint_probability_intervals=None,
) -> dict:
    """Bound P(event) and the unnormalized E[F * 1_event] over the exact lift.

    None selects all memories; an empty tuple selects the empty event. A known
    original table vector constrains only sum_r z^r. Joint probability intervals
    are externally supplied constraints, not confidence intervals inferred here.
    The offset is weighted by event probability. The unconditional root-risk
    expectation must not be reused as a conditional root expectation.
    """
    if not isinstance(lift, JointHaarObservationProblem):
        raise ValueError("Supply a compiled joint observation problem.")
    if event_memory_ids is None:
        selected = tuple(range(lift.memory_count))
    else:
        event_memory_ids = tuple(event_memory_ids)
        selected = tuple(_memory_index(lift, value) for value in event_memory_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("Event memory identifiers must be distinct.")
    matrix, rhs, inequalities, bounds = _joint_constraints(
        lift, prior_table_law, joint_probability_intervals
    )
    probability = np.asarray(lift.probability_rows[list(selected)].sum(axis=0)).ravel()
    lower_loss, upper_loss = (
        np.zeros(lift.variable_count),
        np.zeros(lift.variable_count),
    )
    for index in selected:
        sl = lift.memory_columns[index]
        lower_loss[sl] = lift.problem.coupling.lower_cost
        upper_loss[sl] = lift.problem.coupling.upper_cost
    lower_loss += lift.problem.offset * probability
    upper_loss += lift.problem.offset * probability

    def bound(cost):
        return _dual_lower_bound(
            cost,
            matrix,
            rhs,
            lift.slices,
            inequality_matrix=inequalities,
            inequality_rhs=bounds,
        )

    probability_lower, plr, plg = bound(probability)
    probability_negative_upper, pur, pug = bound(-probability)
    loss_lower, llr, llg = bound(lower_loss)
    loss_negative_upper, lur, lug = bound(-upper_loss)
    return {
        "probability_lower": max(0.0, probability_lower),
        "probability_upper": min(1.0, -probability_negative_upper),
        "event_loss_lower": loss_lower,
        "event_loss_upper": -loss_negative_upper,
        "primal_residual": max(plr, pur, llr, lur),
        "numerical_duality_gap": max(plg, pug, llg, lug),
        "memory_count": lift.memory_count,
        "transport_variables": lift.variable_count,
        "known_aggregate_tables": prior_table_law is not None,
        "conditional_coefficient_marginals_imposed": False,
        "event_loss_interpretation": "unnormalized_expectation_F_times_event_indicator",
        "scope": "exact_single_root_balanced_haar_same_replica_deterministic_query_memory",
        "reference_risk_qualified": False,
        "calibration_established": False,
    }
