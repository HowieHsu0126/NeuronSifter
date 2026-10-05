"""Certified exact-state compilation for scalar binary transform forests.

The compiler tracks rational coordinates or coordinates in one quadratic
field ``Q(sqrt(D))``.  It builds the crop-aware active topology and sparse
transport equalities once; marginal right-hand sides and clipped leaf costs
are assembled separately for each evaluation instance.  Exact integer keys,
never floating-point values, determine state equality.  Objective evaluation
remains floating point.  The initial generic path is full-resolution with zero
coefficient radius; nonzero radius is restricted to an explicit supported
legacy uncertainty path such as the existing Haar envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


ExactCoordinate = tuple[int, ...]


class ExactTransformCompilationError(ValueError):
    """Fail-closed compiler error with a machine-readable status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ExactTransformEdge:
    """Integer certificate for one edge map.

    ``matrix`` is multiplication by ``q0 * a_e`` in the fixed power basis
    ``(1, sqrt(D))``, and ``vector`` contains the coordinates of ``q0 * b_e``.
    Rational descriptors use the one-dimensional basis ``(1,)``.
    """

    matrix: tuple[tuple[int, ...], ...]
    vector: ExactCoordinate


@dataclass(frozen=True)
class ExactTransformDescriptor:
    """One exact field and deterministic edge certificate per retained child."""

    q0: int
    radicand: int | None
    edge_maps: tuple[tuple[ExactTransformEdge, ...], ...]


@dataclass(frozen=True)
class ExactTransformTableTopology:
    """Exact states and incoming pushforward map for one active internal node."""

    depth: int
    position: int
    columns: slice
    state_keys: tuple[ExactCoordinate, ...]
    constant_key: ExactCoordinate
    detail_index: int
    uncertainty_factor: float
    parent_table_index: int | None
    incoming_state_ids: tuple[int, ...] | None
    root_code_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class ExactTransformTransition:
    """Exact map from every parent table entry to one active child."""

    parent_table_index: int
    child_table_index: int | None
    child_depth: int
    child_position: int
    leaf_index: int | None
    state_keys: tuple[ExactCoordinate, ...]
    state_ids: tuple[int, ...] | None
    constant_key: ExactCoordinate
    uncertainty_factor: float


@dataclass(frozen=True)
class ExactTransformTopology:
    """Reusable exact state topology and marginal-independent constraints."""

    descriptor: ExactTransformDescriptor
    lengths: tuple[int, ...]
    sizes: tuple[int, ...]
    active_leaves: tuple[bool, ...]
    bin_count: int
    matrix: csr_matrix
    rhs_constants: np.ndarray
    rhs_marginal_rows: tuple[tuple[int, int, int], ...]
    slices: tuple[slice, ...]
    tables: tuple[ExactTransformTableTopology, ...]
    transitions: tuple[ExactTransformTransition, ...]
    table_order: tuple[int, ...]
    node_order: tuple[tuple[int, int], ...]
    state_counts: tuple[int, ...]
    variable_count: int
    max_coordinate_bit_length: int
    status: str


@dataclass(frozen=True)
class ExactAmplitudeTable:
    """Floating evaluation of one exact table for a supplied affine codebook."""

    depth: int
    position: int
    columns: slice
    amplitudes: np.ndarray
    details: np.ndarray


@dataclass(frozen=True)
class ExactTransformProblem:
    """One marginal/cost instance assembled on a reusable exact topology."""

    topology: ExactTransformTopology
    matrix: csr_matrix
    rhs: np.ndarray
    lower_cost: np.ndarray
    upper_cost: np.ndarray
    slices: tuple[slice, ...]
    tables: tuple[ExactAmplitudeTable, ...]


def _compilation_error(status: str, message: str) -> None:
    raise ExactTransformCompilationError(status, message)


def _strict_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        _compilation_error(
            "invalid_exact_transform_certificate",
            f"{name} must be a positive Python integer.",
        )
    return value


def _validate_descriptor(
    descriptor: ExactTransformDescriptor,
    child_counts: tuple[int, ...],
) -> int:
    if not isinstance(descriptor, ExactTransformDescriptor):
        _compilation_error(
            "invalid_exact_transform_certificate",
            "Expected an ExactTransformDescriptor.",
        )
    _strict_positive_integer(descriptor.q0, "q0")
    if descriptor.radicand is None:
        dimension = 1
    else:
        radicand = descriptor.radicand
        if (
            type(radicand) is not int
            or radicand <= 0
            or math.isqrt(radicand) ** 2 == radicand
        ):
            _compilation_error(
                "invalid_exact_transform_certificate",
                "Quadratic radicand D must be a positive nonsquare Python integer.",
            )
        dimension = 2
    if not isinstance(descriptor.edge_maps, tuple) or len(descriptor.edge_maps) != len(
        child_counts
    ):
        _compilation_error(
            "invalid_exact_transform_certificate",
            "Edge-map levels must match the executed transform depth.",
        )
    for depth, (edges, child_count) in enumerate(
        zip(descriptor.edge_maps, child_counts, strict=True)
    ):
        if not isinstance(edges, tuple) or len(edges) != child_count:
            _compilation_error(
                "invalid_exact_transform_certificate",
                f"Edge-map level {depth} must contain one edge per child.",
            )
        for edge in edges:
            if not isinstance(edge, ExactTransformEdge):
                _compilation_error(
                    "invalid_exact_transform_certificate",
                    "Every child needs an ExactTransformEdge certificate.",
                )
            if (
                not isinstance(edge.matrix, tuple)
                or len(edge.matrix) != dimension
                or any(
                    not isinstance(row, tuple) or len(row) != dimension
                    for row in edge.matrix
                )
                or not isinstance(edge.vector, tuple)
                or len(edge.vector) != dimension
            ):
                _compilation_error(
                    "invalid_exact_transform_certificate",
                    "Edge matrices and vectors must match the exact field dimension.",
                )
            values = (*edge.vector, *(value for row in edge.matrix for value in row))
            if any(type(value) is not int for value in values):
                _compilation_error(
                    "invalid_exact_transform_certificate",
                    "Edge matrices and vectors must contain only Python integers.",
                )
            if dimension == 2:
                matrix = edge.matrix
                if (
                    matrix[0][0] != matrix[1][1]
                    or matrix[0][1] != descriptor.radicand * matrix[1][0]
                ):
                    _compilation_error(
                        "invalid_exact_transform_certificate",
                        "Quadratic edge matrices must certify field multiplication.",
                    )
    return dimension


def _field_value(coordinate: ExactCoordinate, radicand: int | None) -> float:
    try:
        rational = float(coordinate[0])
        if radicand is None or coordinate[1] == 0:
            value = rational
        else:
            value = rational + math.sqrt(radicand) * float(coordinate[1])
    except (OverflowError, ValueError) as error:
        raise ExactTransformCompilationError(
            "exact_transform_numeric_overflow",
            "Exact field evaluation overflowed floating-point arithmetic.",
        ) from error
    if not math.isfinite(value):
        _compilation_error(
            "exact_transform_numeric_overflow",
            "Exact field evaluation is nonfinite in floating-point arithmetic.",
        )
    return value


def _matrix_vector(
    matrix: tuple[tuple[int, ...], ...], coordinate: ExactCoordinate
) -> ExactCoordinate:
    return tuple(
        sum(value * coordinate[column] for column, value in enumerate(row))
        for row in matrix
    )


def compile_exact_transform_topology(
    descriptor: ExactTransformDescriptor,
    lengths: tuple[int, ...],
    sizes: tuple[int, ...],
    active_leaves,
    bin_count: int,
    *,
    max_variables: int,
    max_coordinate_magnitude: int = (1 << 255) - 1,
    max_coordinate_bits: int = 255,
) -> ExactTransformTopology:
    """Compile exact active-subtree states and reusable transport equalities."""

    bin_count = _strict_positive_integer(bin_count, "bin_count")
    max_variables = _strict_positive_integer(max_variables, "max_variables")
    max_coordinate_magnitude = _strict_positive_integer(
        max_coordinate_magnitude, "max_coordinate_magnitude"
    )
    max_coordinate_bits = _strict_positive_integer(
        max_coordinate_bits, "max_coordinate_bits"
    )
    if (
        not isinstance(lengths, tuple)
        or not lengths
        or any(type(length) is not int or length < 2 for length in lengths)
        or not isinstance(sizes, tuple)
        or any(type(size) is not int or size < 1 for size in sizes)
    ):
        _compilation_error(
            "invalid_exact_transform_topology",
            "Lengths and sizes must be nonempty positive integer tuples.",
        )
    leaves = np.asarray(active_leaves)
    if leaves.ndim != 1 or leaves.dtype != np.bool_ or len(leaves) != lengths[0]:
        _compilation_error(
            "invalid_exact_transform_topology",
            "active_leaves must be one Boolean value per physical leaf.",
        )
    current = len(leaves)
    for length in lengths:
        if length != current:
            _compilation_error(
                "invalid_exact_transform_topology",
                "Transform lengths must follow repeated binary ceiling crops.",
            )
        current = (current + 1) // 2
    expected_sizes = (
        current,
        *reversed(tuple((length + 1) // 2 for length in lengths)),
    )
    if sizes != expected_sizes:
        _compilation_error(
            "invalid_exact_transform_topology",
            "Component sizes do not match the transform lengths.",
        )
    child_counts = tuple(reversed(lengths))
    dimension = _validate_descriptor(descriptor, child_counts)

    max_bit_length = 0

    def checked_integer(value: int) -> int:
        nonlocal max_bit_length
        magnitude = abs(value)
        if magnitude > max_coordinate_magnitude:
            _compilation_error(
                "exact_transform_coordinate_overflow",
                "Exact transform coordinate magnitude budget exceeded.",
            )
        bit_length = magnitude.bit_length()
        if bit_length > max_coordinate_bits:
            _compilation_error(
                "exact_transform_bit_length_overflow",
                "Exact transform coordinate bit-length budget exceeded.",
            )
        max_bit_length = max(max_bit_length, bit_length)
        return value

    checked_integer(descriptor.q0)
    if descriptor.radicand is not None:
        checked_integer(descriptor.radicand)
    for edges in descriptor.edge_maps:
        for edge in edges:
            for value in (*edge.vector, *(item for row in edge.matrix for item in row)):
                checked_integer(value)

    active_levels = [tuple(bool(value) for value in leaves)]
    for length in lengths:
        previous = active_levels[-1]
        if length % 2:
            previous = (*previous, False)
        active_levels.append(
            tuple(
                previous[index] or previous[index + 1]
                for index in range(0, len(previous), 2)
            )
        )
    active_levels = list(reversed(active_levels))
    active_root_count = sum(active_levels[0])
    if active_root_count * bin_count**2 > max_variables:
        _compilation_error(
            "exact_transform_state_budget_exceeded",
            "Total transport variable budget exceeded before root-state allocation.",
        )
    detail_offsets = []
    offset = sizes[0]
    for detail_count in sizes[1:]:
        detail_offsets.append(offset)
        offset += detail_count

    one = (1, *([0] * (dimension - 1)))
    root_states = (
        tuple(
            tuple(checked_integer(digit * coordinate) for coordinate in one)
            for digit in range(bin_count)
        )
        if active_root_count
        else ()
    )
    tables: list[dict] = []
    transitions: list[ExactTransformTransition] = []
    slices: list[slice] = []
    total = 0

    def checked_coordinate(values) -> ExactCoordinate:
        coordinate = tuple(checked_integer(value) for value in values)
        if len(coordinate) != dimension:
            _compilation_error(
                "invalid_exact_transform_topology",
                "Generated coordinate has the wrong exact field dimension.",
            )
        return coordinate

    def visit(
        depth: int,
        position: int,
        state_keys: tuple[ExactCoordinate, ...],
        constant_key: ExactCoordinate,
        uncertainty_factor: float,
        parent_table_index: int | None = None,
        incoming_state_ids: tuple[int, ...] | None = None,
    ) -> int:
        nonlocal total
        count = len(state_keys) * bin_count
        if total + count > max_variables:
            _compilation_error(
                "exact_transform_state_budget_exceeded",
                "Total transport variable budget exceeded before allocation.",
            )
        columns = slice(total, total + count)
        total += count
        table_index = len(tables)
        slices.append(columns)
        tables.append(
            {
                "depth": depth,
                "position": position,
                "columns": columns,
                "state_keys": state_keys,
                "constant_key": constant_key,
                "detail_index": detail_offsets[depth] + position,
                "uncertainty_factor": uncertainty_factor,
                "parent_table_index": parent_table_index,
                "incoming_state_ids": incoming_state_ids,
                "root_code_ids": (
                    tuple(range(bin_count)) if parent_table_index is None else None
                ),
            }
        )
        scale = checked_integer(descriptor.q0**depth)
        for side in (0, 1):
            child_position = 2 * position + side
            if (
                child_position >= child_counts[depth]
                or not active_levels[depth + 1][child_position]
            ):
                continue
            edge = descriptor.edge_maps[depth][child_position]
            child_constant_base = _matrix_vector(edge.matrix, constant_key)
            child_constant = checked_coordinate(
                child_constant_base[index] + scale * edge.vector[index]
                for index in range(dimension)
            )
            child_keys = []
            for state_key in state_keys:
                base = _matrix_vector(edge.matrix, state_key)
                for digit in range(bin_count):
                    child_keys.append(
                        checked_coordinate(
                            base[index] + scale * edge.vector[index] * digit
                            for index in range(dimension)
                        )
                    )
            multiplier = (
                _field_value(tuple(row[0] for row in edge.matrix), descriptor.radicand)
                / descriptor.q0
            )
            detail_multiplier = (
                _field_value(edge.vector, descriptor.radicand) / descriptor.q0
            )
            child_uncertainty = abs(multiplier) * uncertainty_factor + abs(
                detail_multiplier
            )
            if not math.isfinite(child_uncertainty):
                _compilation_error(
                    "exact_transform_numeric_overflow",
                    "Exact transform uncertainty factor is nonfinite.",
                )
            child_depth = depth + 1
            checked_integer(descriptor.q0**child_depth)
            if child_depth == len(lengths):
                transition = ExactTransformTransition(
                    table_index,
                    None,
                    child_depth,
                    child_position,
                    child_position,
                    tuple(child_keys),
                    None,
                    child_constant,
                    child_uncertainty,
                )
            else:
                unique_keys = tuple(sorted(set(child_keys)))
                state_lookup = {
                    state_key: state_id
                    for state_id, state_key in enumerate(unique_keys)
                }
                state_ids = tuple(state_lookup[state_key] for state_key in child_keys)
                child_table_index = visit(
                    child_depth,
                    child_position,
                    unique_keys,
                    child_constant,
                    child_uncertainty,
                    table_index,
                    state_ids,
                )
                transition = ExactTransformTransition(
                    table_index,
                    child_table_index,
                    child_depth,
                    child_position,
                    None,
                    tuple(child_keys),
                    state_ids,
                    child_constant,
                    child_uncertainty,
                )
            transitions.append(transition)
        return table_index

    for root in np.flatnonzero(active_levels[0]):
        visit(0, int(root), root_states, one, 1.0)

    frozen_tables = tuple(ExactTransformTableTopology(**table) for table in tables)
    row_chunks: list[np.ndarray] = []
    column_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    rhs_constants: list[float] = []
    rhs_marginal_rows: list[tuple[int, int, int]] = []

    def entries(rows, columns, values) -> None:
        row_chunks.append(np.asarray(rows, dtype=np.int64))
        column_chunks.append(np.asarray(columns, dtype=np.int64))
        value_chunks.append(np.asarray(values, dtype=np.float64))

    for table in frozen_tables:
        columns = np.arange(table.columns.start, table.columns.stop)
        state_ids = np.repeat(np.arange(len(table.state_keys)), bin_count)
        code_ids = np.tile(np.arange(bin_count), len(table.state_keys))
        entries(
            np.full(len(columns), len(rhs_constants)),
            columns,
            np.ones(len(columns)),
        )
        rhs_constants.append(1.0)
        row_start = len(rhs_constants)
        entries(row_start + code_ids, columns, np.ones(len(columns)))
        for code_id in range(bin_count):
            rhs_marginal_rows.append((row_start + code_id, table.detail_index, code_id))
            rhs_constants.append(0.0)
        row_start = len(rhs_constants)
        entries(row_start + state_ids, columns, np.ones(len(columns)))
        if table.parent_table_index is None:
            if table.root_code_ids is None:
                raise AssertionError("Root code IDs were not compiled.")
            for state_id, code_id in enumerate(table.root_code_ids):
                rhs_marginal_rows.append(
                    (row_start + state_id, table.position, code_id)
                )
                rhs_constants.append(0.0)
        else:
            if table.incoming_state_ids is None:
                raise AssertionError("Internal incoming states were not compiled.")
            parent = frozen_tables[table.parent_table_index]
            parent_columns = np.arange(parent.columns.start, parent.columns.stop)
            entries(
                row_start + np.asarray(table.incoming_state_ids),
                parent_columns,
                -np.ones(len(parent_columns)),
            )
            rhs_constants.extend(np.zeros(len(table.state_keys)))

    if total:
        matrix = coo_matrix(
            (
                np.concatenate(value_chunks),
                (np.concatenate(row_chunks), np.concatenate(column_chunks)),
            ),
            shape=(len(rhs_constants), total),
        ).tocsr()
    else:
        matrix = csr_matrix((0, 0), dtype=np.float64)
    return ExactTransformTopology(
        descriptor=descriptor,
        lengths=lengths,
        sizes=sizes,
        active_leaves=tuple(bool(value) for value in leaves),
        bin_count=bin_count,
        matrix=matrix,
        rhs_constants=np.asarray(rhs_constants),
        rhs_marginal_rows=tuple(rhs_marginal_rows),
        slices=tuple(slices),
        tables=frozen_tables,
        transitions=tuple(transitions),
        table_order=tuple(range(len(frozen_tables))),
        node_order=tuple((table.depth, table.position) for table in frozen_tables),
        state_counts=tuple(len(table.state_keys) for table in frozen_tables),
        variable_count=total,
        max_coordinate_bit_length=max_bit_length,
        status="exact_transform_topology_compiled",
    )


def _coordinate_amplitude(
    state_key: ExactCoordinate,
    constant_key: ExactCoordinate,
    *,
    lower: float,
    step: float,
    denominator: int,
    radicand: int | None,
) -> float:
    return (
        lower * _field_value(constant_key, radicand)
        + step * _field_value(state_key, radicand)
    ) / denominator


def _supports_certified_haar_radius(topology: ExactTransformTopology) -> bool:
    descriptor = topology.descriptor
    if descriptor.q0 != 2 or descriptor.radicand != 2:
        return False
    matrix = ((0, 2), (1, 0))
    return all(
        edge.matrix == matrix and edge.vector == (0, 1 if child % 2 == 0 else -1)
        for edges in descriptor.edge_maps
        for child, edge in enumerate(edges)
    )


def _assemble_exact_transform_problem(
    topology: ExactTransformTopology,
    masses,
    lower: float,
    step: float,
    radius: float,
    reference,
    weights,
    floor: float,
    *,
    allow_certified_radius: bool = False,
) -> ExactTransformProblem:
    """Bind one floating objective to exact states and reusable constraints.

    Generic rational/quadratic assembly is currently certified only at zero
    coefficient radius.  ``allow_certified_radius`` is an internal opt-in for
    an independently supported envelope, presently the legacy Haar path.
    """

    if not isinstance(topology, ExactTransformTopology):
        raise ValueError("Expected a compiled exact transform topology.")
    masses_array = np.asarray(masses, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    expected_masses = (sum(topology.sizes), topology.bin_count)
    if masses_array.shape != expected_masses:
        raise ValueError("Marginal masses do not match the compiled topology.")
    if reference_array.shape != (topology.lengths[0],) or weights_array.shape != (
        topology.lengths[0],
    ):
        raise ValueError("Leaf reference and weights do not match the topology.")
    scalars = np.asarray([lower, step, radius, floor], dtype=np.float64)
    if (
        not np.isfinite(masses_array).all()
        or not np.isfinite(reference_array).all()
        or not np.isfinite(weights_array).all()
        or not np.isfinite(scalars).all()
        or (masses_array < 0).any()
        or not np.allclose(masses_array.sum(-1), 1.0, atol=1e-10, rtol=0)
        or step < 0
        or radius < 0
    ):
        raise ValueError("Exact transform problem inputs must be finite and valid.")
    if radius > 0 and (
        allow_certified_radius is not True
        or not _supports_certified_haar_radius(topology)
    ):
        _compilation_error(
            "exact_transform_uncertainty_not_certified",
            "Positive coefficient radius is certified only for the exact Haar descriptor.",
        )
    if np.any((weights_array != 0) & ~np.asarray(topology.active_leaves)):
        raise ValueError("Goal weights activate a leaf pruned from the topology.")
    rhs = topology.rhs_constants.copy()
    for row, coefficient, code_id in topology.rhs_marginal_rows:
        rhs[row] = masses_array[coefficient, code_id]
    lower_cost = np.zeros(topology.variable_count)
    upper_cost = np.zeros(topology.variable_count)
    amplitude_tables = []
    radicand = topology.descriptor.radicand
    q0 = topology.descriptor.q0
    for table in topology.tables:
        denominator = q0**table.depth
        amplitudes = np.asarray(
            [
                _coordinate_amplitude(
                    state_key,
                    table.constant_key,
                    lower=lower,
                    step=step,
                    denominator=denominator,
                    radicand=radicand,
                )
                for state_key in table.state_keys
            ]
        )
        amplitude_tables.append(
            ExactAmplitudeTable(
                table.depth,
                table.position,
                table.columns,
                amplitudes,
                lower + step * np.arange(topology.bin_count),
            )
        )
    for transition in topology.transitions:
        if transition.leaf_index is None:
            continue
        denominator = q0**transition.child_depth
        amplitude = np.asarray(
            [
                _coordinate_amplitude(
                    state_key,
                    transition.constant_key,
                    lower=lower,
                    step=step,
                    denominator=denominator,
                    radicand=radicand,
                )
                for state_key in transition.state_keys
            ]
        )
        uncertainty = radius * transition.uncertainty_factor
        leaf = transition.leaf_index
        lo = weights_array[leaf] * np.maximum(
            floor, reference_array[leaf] + amplitude - uncertainty
        )
        hi = weights_array[leaf] * np.maximum(
            floor, reference_array[leaf] + amplitude + uncertainty
        )
        columns = topology.tables[transition.parent_table_index].columns
        lower_cost[columns] += np.minimum(lo, hi)
        upper_cost[columns] += np.maximum(lo, hi)
    if not all(
        np.isfinite(value).all() for value in (rhs, lower_cost, upper_cost)
    ) or any(not np.isfinite(table.amplitudes).all() for table in amplitude_tables):
        raise ArithmeticError("Exact transform assembly produced nonfinite values.")
    return ExactTransformProblem(
        topology=topology,
        matrix=topology.matrix,
        rhs=rhs,
        lower_cost=lower_cost,
        upper_cost=upper_cost,
        slices=topology.slices,
        tables=tuple(amplitude_tables),
    )
