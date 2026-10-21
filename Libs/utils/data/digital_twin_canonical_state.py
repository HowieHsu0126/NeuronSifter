"""Typed, sparse-safe state blocks for the NeuronSifter Digital Twin.

The contract is intentionally a boundary, not a data materializer.  It accepts
only explicit coordinates and logical values, preserves source storage shape,
and refuses to turn missingness into a numerical observation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterator, Mapping, Sequence


CANONICAL_AXES = ("time", "region", "compartment", "species")
AXES = (*CANONICAL_AXES, "modality")
STATE_BLOCK_FIELDS = (
    "schema_version", "state_block_uid", "evidence_uid", "evidence_type", "source_uid",
    "publication_accession", "biological_unit_uid", "state_uid", "state_time",
    "time_semantics", "organism", "axis_metadata", "feature_namespace", "feature_ids",
    "value_storage", "units", "biological_context", "uncertainty", "qc_grade",
    "pairing_semantics", "aggregation_semantics", "element_masks", "provenance",
    "allowed_train_roles", "allowed_eval_roles", "claim_eligibility", "loss_eligibility",
    "content_hash",
)
RAW_STATE_BLOCK_FIELDS = tuple(field for field in STATE_BLOCK_FIELDS if field not in {"state_block_uid", "content_hash", "loss_eligibility"})
AXIS_METADATA_FIELDS = ("axis_status", "axis_present", "axis_values", "axis_mapping_version", "axis_uncertainty", "axis_mask")
ELEMENT_MASK_FIELDS = ("observed", "missing", "unavailable", "not_applicable", "quarantined")
VALUE_STORAGE_FIELDS = ("format", "source_axis_order", "shape", "dense_values", "sparse_entries", "ragged_entries", "reference")
AGGREGATION_FIELDS = (
    "aggregation_level", "aggregation_method", "source_record_count", "independent_unit_count",
    "aggregation_weights", "aggregation_provenance", "headline_eligible",
)
AGGREGATION_PROVENANCE_FIELDS = (
    "status", "source_record_uids", "source_artifact_sha256", "method_version", "reason",
)
ELIGIBILITY_FIELDS = (
    "eligible_for_state_reconstruction", "eligible_for_temporal_dynamics",
    "eligible_for_action_endpoint", "eligible_for_uncertainty_calibration",
    "eligible_for_constraint_evaluation", "eligible_for_what_if_initialization",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanonicalStateContractError(ValueError):
    """A state block violates a semantic boundary that cannot be repaired safely."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class DigitalTwinStateBlock:
    """A fully normalized, deterministic, JSON-safe state block."""

    payload: dict[str, object]

    @property
    def state_block_uid(self) -> str:
        return str(self.payload["state_block_uid"])

    @property
    def state_uid(self) -> str:
        return str(self.payload["state_uid"])

    @property
    def content_hash(self) -> str:
        return str(self.payload["content_hash"])

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.payload)


# Retained as a compatibility alias for the original isolated Round-11 API.
CanonicalStateBlock = DigitalTwinStateBlock


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash(payload: object) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _mapping(value: object, code: str, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CanonicalStateContractError(code, f"{name} must be a mapping.")
    return dict(value)


def _list(value: object, code: str, name: str) -> list[object]:
    if not isinstance(value, list):
        raise CanonicalStateContractError(code, f"{name} must be a list.")
    return list(value)


def _exact_keys(value: Mapping[str, object], fields: Sequence[str], code: str, name: str) -> None:
    missing = sorted(set(fields) - set(value))
    unknown = sorted(set(value) - set(fields))
    if missing or unknown:
        raise CanonicalStateContractError(code, f"{name} fields mismatch; missing={missing}, unknown={unknown}.")


def _string(value: object, code: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalStateContractError(code, f"{name} must be a non-empty string.")
    return value


def _optional_string(value: object, code: str, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, code, name)


def _finite(value: object, code: str, name: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CanonicalStateContractError(code, f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise CanonicalStateContractError(code, f"{name} must be finite; NaN and infinity are ambiguous observations.")
    if nonnegative and result < 0:
        raise CanonicalStateContractError(code, f"{name} must be non-negative.")
    return result


def _dotted(config: Mapping[str, object], path: str) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise CanonicalStateContractError("missing_config_owner", f"Missing configured owner {path}.")
        current = current[part]
    return current


def canonical_state_config(config: Mapping[str, object]) -> dict[str, object]:
    return _mapping(config.get("digital_twin_canonical_state"), "missing_canonical_state_config", "digital_twin_canonical_state")


def _axis_coordinates(config: Mapping[str, object]) -> dict[str, list[object]]:
    time_grid = _mapping(_dotted(config, "dynamics_ready_registry.time_grid"), "invalid_time_axis_owner", "time grid")
    start = _finite(time_grid.get("start_hours"), "invalid_time_axis_owner", "start_hours")
    stop = _finite(time_grid.get("stop_hours"), "invalid_time_axis_owner", "stop_hours")
    count = time_grid.get("step_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 2 or stop <= start:
        raise CanonicalStateContractError("invalid_time_axis_owner", "Configured time grid must have an increasing integer grid.")
    def ids(path: str, field: str, code: str) -> list[str]:
        rows = _list(_dotted(config, path), code, path)
        result = [_string(_mapping(row, code, "axis owner").get(field), code, field) for row in rows]
        if not result or len(set(result)) != len(result):
            raise CanonicalStateContractError(code, f"{path} must contain unique values.")
        return result
    return {
        "time": [start + (stop - start) * index / (count - 1) for index in range(count)],
        "region": ids("spatial_atlas_registry.regions", "region_id", "invalid_region_axis_owner"),
        "compartment": ids("screening_pipeline.pkpd_ready_registry.compartments", "compartment_id", "invalid_compartment_axis_owner"),
        "species": ids("dynamics_ready_registry.species", "species_id", "invalid_species_axis_owner"),
        "modality": [_string(item, "invalid_modality_owner", "supported modality") for item in _list(_dotted(config, "serving_registry.supported_modalities"), "invalid_modality_owner", "supported modalities")],
    }


def canonical_axis_coordinates(config: Mapping[str, object]) -> dict[str, list[object]]:
    validate_canonical_state_config(config)
    return {axis: values for axis, values in _axis_coordinates(config).items() if axis in CANONICAL_AXES}


def _species_units(config: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _list(_dotted(config, "dynamics_ready_registry.species"), "invalid_species_unit_owner", "species"):
        item = _mapping(row, "invalid_species_unit_owner", "species")
        name = _string(item.get("species_id"), "invalid_species_unit_owner", "species_id")
        if name in result:
            raise CanonicalStateContractError("invalid_species_unit_owner", "Duplicate state-variable unit.")
        result[name] = _string(item.get("unit"), "invalid_species_unit_owner", "unit")
    return result


def validate_canonical_state_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    """Return deterministic config blockers without reading state or evidence artifacts."""
    try:
        contract = canonical_state_config(config)
        if contract.get("version") != "digital_twin_canonical_state_v1":
            raise CanonicalStateContractError("invalid_canonical_state_version", "Expected digital_twin_canonical_state_v1.")
        if contract.get("serialization_version") != "neuronsifter-digital-twin-state-block-v2":
            raise CanonicalStateContractError("invalid_serialization_version", "State serialization version is not v2.")
        if tuple(_list(contract.get("canonical_axis_order"), "invalid_axis_order", "canonical_axis_order")) != CANONICAL_AXES:
            raise CanonicalStateContractError("invalid_axis_order", "Canonical axes must be time, region, compartment, species.")
        schema_keys = _mapping(contract.get("schema_field_keys"), "invalid_schema_field_keys", "schema_field_keys")
        schemas = _mapping(config.get("schemas"), "missing_schema_owner", "schemas")
        expected = {
            "state_block": list(STATE_BLOCK_FIELDS),
            "axis_metadata": list(AXIS_METADATA_FIELDS),
            "element_masks": list(ELEMENT_MASK_FIELDS),
            "value_storage": list(VALUE_STORAGE_FIELDS),
            "aggregation_semantics": list(AGGREGATION_FIELDS),
            "eligibility": list(ELIGIBILITY_FIELDS),
        }
        for family, fields in expected.items():
            key = _string(schema_keys.get(family), "invalid_schema_field_keys", family)
            if schemas.get(key) != fields:
                raise CanonicalStateContractError("invalid_schema_field_owner", f"schemas.{key} does not own the expected {family} fields.")
        if list(contract.get("element_mask_states", [])) != list(ELEMENT_MASK_FIELDS):
            raise CanonicalStateContractError("invalid_element_mask_policy", "Element-mask states must be exhaustive and ordered.")
        if set(contract.get("value_storage_formats", [])) != {"dense", "sparse", "ragged", "reference"}:
            raise CanonicalStateContractError("invalid_storage_formats", "Storage formats are incomplete.")
        required_aggregation_methods = {
            "none", "cell_to_donor", "sample_to_subject", "technical_replicate",
            "pseudobulk", "source_aggregate", "legacy_aggregate_unknown",
        }
        if set(contract.get("supported_aggregation_methods", [])) != required_aggregation_methods:
            raise CanonicalStateContractError("invalid_aggregation_method_policy", "Aggregation methods must match the canonical seven-value vocabulary.")
        if set(contract.get("supported_aggregation_semantics", [])) != required_aggregation_methods:
            raise CanonicalStateContractError("invalid_aggregation_method_policy", "Aggregation semantics compatibility values must match canonical methods.")
        if set(contract.get("aggregation_levels", [])) != {"none", "donor", "subject", "technical_replicate", "pseudobulk", "source", "unknown"}:
            raise CanonicalStateContractError("invalid_aggregation_level_policy", "Aggregation levels are incomplete.")
        required_time_semantics = {
            "absolute_time", "relative_to_action", "pseudotime", "cross_sectional",
            "ordered_endpoint", "simulator_time", "unknown", "not_applicable",
        }
        if not required_time_semantics.issubset(set(contract.get("supported_time_semantics", []))):
            raise CanonicalStateContractError("invalid_time_semantics_policy", "Canonical time semantics are incomplete.")
        if _mapping(contract.get("axis_owners"), "invalid_axis_owners", "axis_owners") != {
            "time": "dynamics_ready_registry.time_grid", "region": "spatial_atlas_registry.regions",
            "compartment": "screening_pipeline.pkpd_ready_registry.compartments", "species": "dynamics_ready_registry.species",
        }:
            raise CanonicalStateContractError("invalid_axis_owners", "Canonical axes must reuse existing owners.")
        _axis_coordinates(config)
        _species_units(config)
    except CanonicalStateContractError as error:
        return [{"code": error.code, "message": str(error).split(": ", 1)[-1]}]
    return []


def _contract(config: Mapping[str, object]) -> dict[str, object]:
    blockers = validate_canonical_state_config(config)
    if blockers:
        raise CanonicalStateContractError(str(blockers[0]["code"]), str(blockers[0]["message"]))
    return canonical_state_config(config)


def _normal_axis_value(axis: str, value: object, available: Sequence[object]) -> object:
    if axis == "time":
        candidate = _finite(value, "invalid_axis_value", "time coordinate")
        matches = [entry for entry in available if math.isclose(float(entry), candidate, rel_tol=0.0, abs_tol=1e-8)]
        if len(matches) != 1:
            raise CanonicalStateContractError("invalid_axis_value", "Time coordinate is not configured.")
        return float(matches[0])
    candidate = _string(value, "invalid_axis_value", f"{axis} coordinate")
    if candidate not in available:
        raise CanonicalStateContractError("invalid_axis_value", f"{axis} coordinate is not configured.")
    return candidate


def _one_hot_masks(raw: object, count: int, name: str) -> dict[str, list[bool]]:
    masks = _mapping(raw, "invalid_axis_mask", name)
    _exact_keys(masks, ELEMENT_MASK_FIELDS, "invalid_axis_mask", name)
    result: dict[str, list[bool]] = {}
    for state in ELEMENT_MASK_FIELDS:
        values = _list(masks[state], "invalid_axis_mask", f"{name}.{state}")
        if len(values) != count or any(not isinstance(value, bool) for value in values):
            raise CanonicalStateContractError("invalid_axis_mask", f"{name}.{state} must have {count} booleans.")
        result[state] = [bool(value) for value in values]
    if any(sum(result[state][index] for state in ELEMENT_MASK_FIELDS) != 1 for index in range(count)):
        raise CanonicalStateContractError("invalid_axis_mask", f"{name} must be mutually exclusive and exhaustive.")
    return result


def _axis_metadata(config: Mapping[str, object], raw: object) -> dict[str, dict[str, object]]:
    metadata = _mapping(raw, "invalid_axis_metadata", "axis_metadata")
    _exact_keys(metadata, AXES, "invalid_axis_metadata", "axis_metadata")
    available = _axis_coordinates(config)
    normalized: dict[str, dict[str, object]] = {}
    for axis in AXES:
        item = _mapping(metadata[axis], "invalid_axis_metadata", axis)
        _exact_keys(item, AXIS_METADATA_FIELDS, "invalid_axis_metadata", axis)
        axis_status = _string(item.get("axis_status"), "invalid_axis_status", axis)
        if axis_status not in ELEMENT_MASK_FIELDS:
            raise CanonicalStateContractError("invalid_axis_status", f"{axis}.axis_status is not a canonical five-state value.")
        present = item.get("axis_present")
        if not isinstance(present, bool):
            raise CanonicalStateContractError("invalid_axis_presence", f"{axis}.axis_present must be boolean.")
        values = [_normal_axis_value(axis, value, available[axis]) for value in _list(item.get("axis_values"), "invalid_axis_values", axis)]
        if len(set(values)) != len(values):
            raise CanonicalStateContractError("duplicate_axis_value", f"{axis} contains duplicate values.")
        if present != bool(values):
            raise CanonicalStateContractError("inconsistent_axis_presence", f"{axis}.axis_present must match non-empty axis_values.")
        mapping_version = _string(item.get("axis_mapping_version"), "invalid_axis_mapping_version", axis)
        if present and axis_status != "observed":
            raise CanonicalStateContractError("invalid_axis_status", f"Present {axis} must use axis_status=observed.")
        if not present:
            if axis_status not in {"missing", "unavailable", "not_applicable"}:
                raise CanonicalStateContractError("invalid_absent_axis_status", f"Absent {axis} must remain explicitly missing, unavailable, or not_applicable.")
            if mapping_version != f"{axis_status}_axis":
                raise CanonicalStateContractError("invalid_missing_axis_mapping", f"Absent {axis} mapping must retain its {axis_status} semantic status.")
        uncertainty = _mapping(item.get("axis_uncertainty"), "invalid_axis_uncertainty", axis)
        _exact_keys(uncertainty, ("status", "values", "reason"), "invalid_axis_uncertainty", axis)
        status = uncertainty.get("status")
        if status not in {"provided", "missing"}:
            raise CanonicalStateContractError("invalid_axis_uncertainty", f"{axis} uncertainty status is invalid.")
        if status == "provided":
            numbers = [_finite(value, "invalid_axis_uncertainty", f"{axis} uncertainty", nonnegative=True) for value in _list(uncertainty.get("values"), "invalid_axis_uncertainty", axis)]
            if len(numbers) != len(values):
                raise CanonicalStateContractError("axis_uncertainty_shape_mismatch", f"{axis} uncertainty must align with axis values.")
            reason: str | None = _optional_string(uncertainty.get("reason"), "invalid_axis_uncertainty", axis)
        else:
            if uncertainty.get("values") is not None:
                raise CanonicalStateContractError("invalid_missing_axis_uncertainty", f"Missing {axis} uncertainty cannot contain substituted values.")
            reason = _string(uncertainty.get("reason"), "invalid_missing_axis_uncertainty", axis)
            numbers = []
        axis_mask = _one_hot_masks(item.get("axis_mask"), len(values), f"{axis}.axis_mask")
        normalized[axis] = {
            "axis_status": axis_status, "axis_present": present, "axis_values": values, "axis_mapping_version": mapping_version,
            "axis_uncertainty": {"status": status, "values": numbers if status == "provided" else None, "reason": reason},
            "axis_mask": axis_mask,
        }
    return normalized


def _shape(value: object, rank: int) -> list[int]:
    shape = _list(value, "invalid_storage_shape", "value_storage.shape")
    if len(shape) != rank or any(not isinstance(entry, int) or isinstance(entry, bool) or entry < 1 for entry in shape):
        raise CanonicalStateContractError("invalid_storage_shape", "Storage shape must use positive dimensions matching source_axis_order.")
    return [int(entry) for entry in shape]


def _visit_dense(values: object, shape: Sequence[int], path: str = "dense_values") -> Iterator[tuple[tuple[int, ...], object]]:
    def walk(value: object, depth: int, prefix: tuple[int, ...], label: str) -> Iterator[tuple[tuple[int, ...], object]]:
        if depth == len(shape):
            yield prefix, value
            return
        row = _list(value, "invalid_dense_values", label)
        if len(row) != shape[depth]:
            raise CanonicalStateContractError("dense_shape_mismatch", f"{label} does not match value_storage.shape.")
        for index, item in enumerate(row):
            yield from walk(item, depth + 1, (*prefix, index), f"{label}[{index}]")
    yield from walk(values, 0, (), path)


def _relative_reference(path: object) -> str:
    value = _string(path, "invalid_reference_path", "reference.uri")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or value.startswith("~") or "://" in value:
        raise CanonicalStateContractError("private_absolute_reference_path", "Reference URI must be a repository-relative public-safe path.")
    return value


def _value_storage(config: Mapping[str, object], raw: object, metadata: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, object], list[object] | None, int]:
    contract = canonical_state_config(config)
    storage = _mapping(raw, "invalid_value_storage", "value_storage")
    _exact_keys(storage, VALUE_STORAGE_FIELDS, "invalid_value_storage", "value_storage")
    fmt = _string(storage.get("format"), "invalid_storage_format", "value_storage.format")
    if fmt not in contract.get("value_storage_formats", []):
        raise CanonicalStateContractError("invalid_storage_format", "Unknown value storage format.")
    source_axes = [_string(axis, "invalid_source_axis_order", "source axis") for axis in _list(storage.get("source_axis_order"), "invalid_source_axis_order", "source_axis_order")]
    if not source_axes or len(set(source_axes)) != len(source_axes) or any(axis not in CANONICAL_AXES for axis in source_axes):
        raise CanonicalStateContractError("invalid_source_axis_order", "Source axes must be a non-empty canonical-axis subset.")
    if source_axes != [axis for axis in CANONICAL_AXES if axis in source_axes]:
        raise CanonicalStateContractError("invalid_source_axis_order", "Source axes must retain canonical order.")
    if any(not metadata[axis]["axis_present"] for axis in source_axes):
        raise CanonicalStateContractError("source_axis_not_present", "Source storage may not use a missing axis.")
    shape = _shape(storage.get("shape"), 1 if fmt in {"ragged", "reference"} else len(source_axes))
    expected = [len(metadata[axis]["axis_values"]) for axis in source_axes]
    if fmt in {"dense", "sparse"} and shape != expected:
        raise CanonicalStateContractError("storage_axis_shape_mismatch", "Storage shape must match explicitly declared axis coverage.")
    entries: list[dict[str, object]] = []
    logical_values: list[object] = []
    if fmt == "dense":
        if storage.get("sparse_entries") is not None or storage.get("ragged_entries") is not None or storage.get("reference") is not None:
            raise CanonicalStateContractError("invalid_storage_payload", "Dense storage cannot carry another storage payload.")
        for index, value in _visit_dense(storage.get("dense_values"), shape):
            entries.append({"index": list(index), "value": value})
            logical_values.append(value)
        dense_values = deepcopy(storage.get("dense_values"))
        sparse_entries: object = None
        ragged_entries: object = None
        reference: object = None
    elif fmt == "sparse":
        if storage.get("dense_values") is not None or storage.get("ragged_entries") is not None or storage.get("reference") is not None:
            raise CanonicalStateContractError("invalid_storage_payload", "Sparse storage cannot carry another storage payload.")
        rows = _list(storage.get("sparse_entries"), "invalid_sparse_entries", "sparse_entries")
        seen: set[tuple[int, ...]] = set()
        for position, row in enumerate(rows):
            item = _mapping(row, "invalid_sparse_entries", f"sparse entry {position}")
            _exact_keys(item, ("index", "value"), "invalid_sparse_entries", f"sparse entry {position}")
            index = _list(item.get("index"), "invalid_sparse_index", "sparse index")
            if len(index) != len(shape) or any(not isinstance(entry, int) or isinstance(entry, bool) or entry < 0 or entry >= shape[offset] for offset, entry in enumerate(index)):
                raise CanonicalStateContractError("invalid_sparse_index", "Sparse index is out of range.")
            key = tuple(int(entry) for entry in index)
            if key in seen:
                raise CanonicalStateContractError("duplicate_sparse_index", "Sparse entries may not duplicate an index.")
            seen.add(key)
            entries.append({"index": list(key), "value": item.get("value")})
            logical_values.append(item.get("value"))
        dense_values = None
        sparse_entries = sorted(entries, key=lambda entry: tuple(entry["index"]))
        entries = list(sparse_entries)
        ragged_entries = None
        reference = None
    elif fmt == "ragged":
        if storage.get("dense_values") is not None or storage.get("sparse_entries") is not None or storage.get("reference") is not None:
            raise CanonicalStateContractError("invalid_storage_payload", "Ragged storage cannot carry another storage payload.")
        if shape != [len(_list(storage.get("ragged_entries"), "invalid_ragged_entries", "ragged_entries"))]:
            raise CanonicalStateContractError("invalid_ragged_shape", "Ragged shape must be its logical entry count.")
        seen_coordinates: set[str] = set()
        for position, row in enumerate(_list(storage.get("ragged_entries"), "invalid_ragged_entries", "ragged_entries")):
            item = _mapping(row, "invalid_ragged_entries", f"ragged entry {position}")
            _exact_keys(item, ("coordinates", "value"), "invalid_ragged_entries", f"ragged entry {position}")
            coordinates = _mapping(item.get("coordinates"), "invalid_ragged_coordinates", "ragged coordinates")
            _exact_keys(coordinates, source_axes, "invalid_ragged_coordinates", "ragged coordinates")
            normalized = {axis: _normal_axis_value(axis, coordinates[axis], metadata[axis]["axis_values"]) for axis in source_axes}
            key = _json(normalized)
            if key in seen_coordinates:
                raise CanonicalStateContractError("duplicate_ragged_coordinates", "Ragged coordinates may not duplicate.")
            seen_coordinates.add(key)
            entries.append({"coordinates": normalized, "value": item.get("value")})
            logical_values.append(item.get("value"))
        dense_values = None
        sparse_entries = None
        ragged_entries = entries
        reference = None
    else:
        if storage.get("dense_values") is not None or storage.get("sparse_entries") is not None or storage.get("ragged_entries") is not None:
            raise CanonicalStateContractError("invalid_storage_payload", "Reference storage cannot carry inline values.")
        item = _mapping(storage.get("reference"), "invalid_reference", "reference")
        _exact_keys(item, ("uri", "content_sha256", "entry_count"), "invalid_reference", "reference")
        uri = _relative_reference(item.get("uri"))
        digest = _string(item.get("content_sha256"), "invalid_reference_hash", "reference.content_sha256")
        count = item.get("entry_count")
        if not SHA256_RE.fullmatch(digest) or not isinstance(count, int) or isinstance(count, bool) or count < 1 or shape != [count]:
            raise CanonicalStateContractError("invalid_reference", "Reference hash, count, or shape is invalid.")
        logical_values = None
        dense_values = None
        sparse_entries = None
        ragged_entries = None
        reference = {"uri": uri, "content_sha256": digest, "entry_count": count}
    return {
        "format": fmt, "source_axis_order": source_axes, "shape": shape, "dense_values": dense_values,
        "sparse_entries": sparse_entries, "ragged_entries": ragged_entries, "reference": reference,
    }, logical_values, count if fmt == "reference" else len(logical_values)


def _element_masks(raw: object, count: int, values: Sequence[object] | None, *, fmt: str) -> dict[str, list[bool]]:
    masks = _one_hot_masks(raw, count, "element_masks")
    for index in range(count):
        value = None if values is None else values[index]
        state = next(name for name in ELEMENT_MASK_FIELDS if masks[name][index])
        if value is not None:
            _finite(value, "invalid_state_value", f"logical value {index}")
        if state == "observed" and value is None:
            raise CanonicalStateContractError("observed_value_missing", "Observed elements require finite values; zero remains observed.")
        if state != "observed" and value is not None:
            raise CanonicalStateContractError("masked_value_present", "Missing, unavailable, not-applicable, and quarantined elements cannot carry numeric values.")
        if fmt == "reference" and state == "observed":
            raise CanonicalStateContractError("reference_observation_unavailable", "Reference-only values cannot be marked observed without materialization.")
    return masks


def _uncertainty(raw: object, count: int, masks: Mapping[str, Sequence[bool]]) -> dict[str, object]:
    value = _mapping(raw, "invalid_uncertainty", "uncertainty")
    _exact_keys(value, ("source", "mapping"), "invalid_uncertainty", "uncertainty")
    result: dict[str, object] = {}
    for field in ("source", "mapping"):
        item = _mapping(value[field], "invalid_uncertainty", field)
        _exact_keys(item, ("status", "values", "reason"), "invalid_uncertainty", field)
        status = item.get("status")
        if status == "missing":
            if item.get("values") is not None:
                raise CanonicalStateContractError("invalid_missing_uncertainty", f"{field} uncertainty cannot substitute values.")
            result[field] = {"status": "missing", "values": None, "reason": _string(item.get("reason"), "invalid_missing_uncertainty", field)}
            continue
        if status != "provided":
            raise CanonicalStateContractError("invalid_uncertainty_status", f"{field} uncertainty status is invalid.")
        values = _list(item.get("values"), "invalid_uncertainty", field)
        if len(values) != count:
            raise CanonicalStateContractError("uncertainty_shape_mismatch", f"{field} uncertainty must align with logical values.")
        normalized: list[object] = []
        for index, entry in enumerate(values):
            observed = bool(masks["observed"][index])
            if observed:
                normalized.append(_finite(entry, "invalid_uncertainty", f"{field} uncertainty", nonnegative=True))
            elif entry is not None:
                raise CanonicalStateContractError("uncertainty_for_unobserved", f"{field} uncertainty cannot describe an unobserved value.")
            else:
                normalized.append(None)
        result[field] = {"status": "provided", "values": normalized, "reason": _optional_string(item.get("reason"), "invalid_uncertainty", field)}
    return result


def _metadata(config: Mapping[str, object], raw: Mapping[str, object], axis_metadata: Mapping[str, Mapping[str, object]], count: int) -> tuple[dict[str, object], str, str | None, str | None, str | None]:
    evidence_type = _string(raw.get("evidence_type"), "invalid_evidence_type", "evidence_type")
    contract = canonical_state_config(config)
    if evidence_type not in contract.get("supported_evidence_types", []):
        raise CanonicalStateContractError("invalid_evidence_type", "Unknown evidence type.")
    time_semantics = _string(raw.get("time_semantics"), "invalid_time_semantics", "time_semantics")
    if time_semantics not in contract.get("supported_time_semantics", []):
        raise CanonicalStateContractError("invalid_time_semantics", "Unknown time semantics.")
    pairing = _string(raw.get("pairing_semantics"), "invalid_pairing_semantics", "pairing_semantics")
    if pairing not in contract.get("supported_pairing_semantics", []):
        raise CanonicalStateContractError("invalid_pairing_semantics", "Unknown pairing semantics.")
    if evidence_type == "real_static_state" and time_semantics != "cross_sectional":
        raise CanonicalStateContractError("static_time_semantics", "Real static state must use cross_sectional time semantics.")
    if evidence_type == "real_perturbation_endpoint" and (time_semantics != "ordered_endpoint" or pairing != "same_unit_action_endpoint"):
        raise CanonicalStateContractError("static_endpoint_marked_longitudinal", "Perturbation endpoints require endpoint/same-unit-action pairing, never longitudinal semantics.")
    if evidence_type == "real_longitudinal_trajectory" and (time_semantics != "absolute_time" or pairing != "same_unit_longitudinal"):
        raise CanonicalStateContractError("invalid_longitudinal_semantics", "Real longitudinal trajectory requires absolute-time/same-unit pairing.")
    if evidence_type in {"legacy_simulator_trajectory", "clean_simulator_trajectory"} and (time_semantics != "simulator_time" or pairing != "simulator_generated"):
        raise CanonicalStateContractError("invalid_simulator_trajectory_semantics", "Simulator trajectories require simulator-time/generated pairing.")
    feature_namespace = _string(raw.get("feature_namespace"), "invalid_feature_namespace", "feature_namespace")
    modality_values = axis_metadata["modality"]["axis_values"]
    if len(modality_values) != 1:
        raise CanonicalStateContractError("invalid_modality_axis", "A state block must declare exactly one measurement modality.")
    namespaces = _mapping(_dotted(config, "serving_registry.request_feature_namespace"), "invalid_feature_namespace_owner", "feature namespaces")
    if feature_namespace != namespaces.get(modality_values[0]):
        raise CanonicalStateContractError("invalid_feature_namespace", "Feature namespace must match declared modality.")
    feature_ids = [_string(value, "invalid_feature_id", "feature id") for value in _list(raw.get("feature_ids"), "invalid_feature_ids", "feature_ids")]
    if len(feature_ids) != count:
        raise CanonicalStateContractError("feature_value_mismatch", "Feature IDs must align exactly with logical values.")
    organism = _string(raw.get("organism"), "invalid_organism", "organism")
    if organism in axis_metadata["species"]["axis_values"]:
        raise CanonicalStateContractError("organism_species_conflation", "Organism must not be encoded as a state-variable species.")
    context = _mapping(raw.get("biological_context"), "invalid_biological_context", "biological_context")
    for field in canonical_state_config(config).get("required_biological_context_fields", []):
        _string(context.get(str(field)), "missing_biological_context", str(field))
    if context.get("organism") != organism:
        raise CanonicalStateContractError("organism_context_mismatch", "Top-level organism and biological context must agree.")
    return context, evidence_type, time_semantics, pairing, organism


def _units(config: Mapping[str, object], raw: object, metadata: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    value = _mapping(raw, "invalid_units", "units")
    _exact_keys(value, ("by_state_variable", "conversion_kind"), "invalid_units", "units")
    by_species = _mapping(value.get("by_state_variable"), "invalid_units", "units.by_state_variable")
    observed_species = [str(item) for item in metadata["species"]["axis_values"]]
    if set(by_species) != set(observed_species):
        raise CanonicalStateContractError("required_unit_missing", "Units must cover exactly declared state-variable species.")
    configured = _species_units(config)
    for species in observed_species:
        if by_species[species] != configured[species]:
            raise CanonicalStateContractError("incompatible_species_unit", f"Unit for {species} is not the configured state-variable unit.")
    if value.get("conversion_kind") != canonical_state_config(config).get("unit_policy", {}).get("conversion_kind"):
        raise CanonicalStateContractError("incompatible_unit_conversion", "Unit conversion is not canonical.")
    return {"by_state_variable": dict(sorted(by_species.items())), "conversion_kind": value["conversion_kind"]}


def _aggregation(raw: object, evidence_type: str, biological_unit_uid: str | None, supported_methods: Sequence[object]) -> dict[str, object]:
    value = _mapping(raw, "invalid_aggregation_semantics", "aggregation_semantics")
    _exact_keys(value, AGGREGATION_FIELDS, "invalid_aggregation_semantics", "aggregation_semantics")
    level = _string(value.get("aggregation_level"), "invalid_aggregation_semantics", "aggregation_level")
    method = _string(value.get("aggregation_method"), "invalid_aggregation_semantics", "aggregation_method")
    if method not in supported_methods:
        raise CanonicalStateContractError("invalid_aggregation_semantics", "Aggregation method is not configured.")
    if level not in {"none", "donor", "subject", "technical_replicate", "pseudobulk", "source", "unknown"}:
        raise CanonicalStateContractError("invalid_aggregation_semantics", "Aggregation level is not configured.")
    source_count = value.get("source_record_count")
    count = value.get("independent_unit_count")
    for field, value_count in (("source_record_count", source_count), ("independent_unit_count", count)):
        if value_count is not None and (not isinstance(value_count, int) or isinstance(value_count, bool) or value_count < 1):
            raise CanonicalStateContractError("invalid_aggregation_count", f"{field} must be null or a positive integer.")
    weights = value.get("aggregation_weights")
    if weights is not None:
        weights = [_finite(item, "invalid_aggregation_weights", "aggregation weight", nonnegative=True) for item in _list(weights, "invalid_aggregation_weights", "aggregation_weights")]
        if source_count is None or len(weights) != source_count or not weights or sum(weights) <= 0:
            raise CanonicalStateContractError("invalid_aggregation_weights", "Aggregation weights must align with source_record_count and contain positive mass.")
    provenance = _mapping(value.get("aggregation_provenance"), "invalid_aggregation_provenance", "aggregation_provenance")
    _exact_keys(provenance, AGGREGATION_PROVENANCE_FIELDS, "invalid_aggregation_provenance", "aggregation_provenance")
    provenance_status = provenance.get("status")
    if provenance_status not in {"provided", "unavailable", "not_applicable"}:
        raise CanonicalStateContractError("invalid_aggregation_provenance", "Aggregation provenance status is invalid.")
    source_uids = provenance.get("source_record_uids")
    if provenance_status == "provided":
        source_uids = [_string(item, "invalid_aggregation_provenance", "source record UID") for item in _list(source_uids, "invalid_aggregation_provenance", "source_record_uids")]
        if source_count is None or len(source_uids) != source_count:
            raise CanonicalStateContractError("aggregation_provenance_count_mismatch", "Provided aggregation source UIDs must align with source_record_count.")
        artifact_hash = _string(provenance.get("source_artifact_sha256"), "invalid_aggregation_provenance", "source_artifact_sha256")
        if not SHA256_RE.fullmatch(artifact_hash):
            raise CanonicalStateContractError("invalid_aggregation_provenance", "Aggregation provenance artifact hash must be SHA-256.")
        method_version = _string(provenance.get("method_version"), "invalid_aggregation_provenance", "method_version")
        reason = _optional_string(provenance.get("reason"), "invalid_aggregation_provenance", "reason")
    else:
        if provenance.get("source_record_uids") is not None or provenance.get("source_artifact_sha256") is not None or provenance.get("method_version") is not None:
            raise CanonicalStateContractError("invalid_unavailable_aggregation_provenance", "Unavailable aggregation provenance cannot substitute source identifiers or hashes.")
        source_uids, artifact_hash, method_version = None, None, None
        reason = _string(provenance.get("reason"), "invalid_aggregation_provenance", "reason")
    headline = value.get("headline_eligible")
    if not isinstance(headline, bool):
        raise CanonicalStateContractError("invalid_aggregation_semantics", "headline_eligible must be boolean.")
    if method == "legacy_aggregate_unknown":
        if evidence_type != "legacy_simulator_trajectory" or biological_unit_uid is not None or count is not None or headline or level != "unknown" or source_count is None:
            raise CanonicalStateContractError("invalid_legacy_aggregate", "Legacy aggregate must not invent biological units, independent counts, or headline eligibility.")
    else:
        if source_count is None or count is None:
            raise CanonicalStateContractError("invalid_aggregation_count", "Non-legacy aggregation needs explicit source and independent counts.")
        if method == "none" and (level != "none" or source_count != 1 or count != 1 or weights != [1.0] or biological_unit_uid is None):
            raise CanonicalStateContractError("invalid_none_aggregation", "No aggregation requires one source, one independent unit, and unit weight.")
        if method != "none" and biological_unit_uid is None:
            raise CanonicalStateContractError("missing_biological_unit", "Non-legacy aggregation requires a biological unit.")
    return {
        "aggregation_level": level,
        "aggregation_method": method,
        "source_record_count": source_count,
        "independent_unit_count": count,
        "aggregation_weights": weights,
        "aggregation_provenance": {
            "status": provenance_status,
            "source_record_uids": source_uids,
            "source_artifact_sha256": artifact_hash,
            "method_version": method_version,
            "reason": reason,
        },
        "headline_eligible": headline,
    }


def _provenance(raw: object) -> dict[str, object]:
    value = _mapping(raw, "invalid_provenance", "provenance")
    required = ("source_record_uid", "source_artifact_sha256", "mapping_version", "serialization_origin")
    for field in required:
        item = _string(value.get(field), "missing_provenance", field)
        if field == "source_artifact_sha256" and not SHA256_RE.fullmatch(item):
            raise CanonicalStateContractError("invalid_source_artifact_sha256", "Source artifact hash must be SHA-256.")
    for key, item in value.items():
        if "path" in key or key.endswith("uri"):
            _relative_reference(item)
    return dict(sorted(value.items()))


def derive_loss_eligibility(config: Mapping[str, object], *, evidence_type: str, time_semantics: str, pairing_semantics: str, qc_grade: str, masks: Mapping[str, Sequence[bool]], allowed_train_roles: Sequence[str]) -> dict[str, bool]:
    """Derive the six user-facing eligibility fields from evidence semantics."""
    policy = _mapping(canonical_state_config(config).get("loss_eligibility_policy"), "invalid_loss_policy", "loss_eligibility_policy")
    observed = sum(bool(value) for value in masks["observed"])
    quarantined = any(bool(value) for value in masks["quarantined"])
    train = bool(observed and not quarantined and qc_grade in policy["train_qc_grades"] and allowed_train_roles and evidence_type in policy["state_reconstruction_evidence_types"])
    temporal = bool(
        train and evidence_type in policy["temporal_evidence_types"] and time_semantics in policy["temporal_time_semantics"]
        and pairing_semantics in policy["temporal_pairing_semantics"] and qc_grade in policy["temporal_qc_grades"]
    )
    return {
        "eligible_for_state_reconstruction": train,
        "eligible_for_temporal_dynamics": temporal,
        "eligible_for_action_endpoint": bool(train and evidence_type in policy["temporal_action_endpoint_evidence_types"] and pairing_semantics in {"same_unit_action_endpoint", "simulator_generated"}),
        "eligible_for_uncertainty_calibration": bool(observed and not quarantined and qc_grade in policy["uncertainty_calibration_qc_grades"] and evidence_type in policy["uncertainty_calibration_evidence_types"]),
        "eligible_for_constraint_evaluation": bool(observed and not quarantined and qc_grade in policy["constraint_evaluation_qc_grades"] and evidence_type in policy["constraint_evaluation_evidence_types"]),
        "eligible_for_what_if_initialization": bool(train and evidence_type in policy["what_if_initialization_evidence_types"] and pairing_semantics in policy["what_if_initialization_pairing_semantics"] and time_semantics in {"cross_sectional", "absolute_time", "relative_to_action", "simulator_time"}),
    }


def _normal_roles(config: Mapping[str, object], raw: object, name: str) -> list[str]:
    values = [_string(item, "invalid_role_eligibility", name) for item in _list(raw, "invalid_role_eligibility", name)]
    if len(set(values)) != len(values):
        raise CanonicalStateContractError("invalid_role_eligibility", f"{name} may not duplicate roles.")
    fabric = _mapping(config.get("digital_twin_evidence_fabric"), "missing_evidence_role_owner", "digital_twin_evidence_fabric")
    allowed_key = "allowed_train_roles" if name == "allowed_train_roles" else "allowed_eval_roles"
    allowed = set(_list(fabric.get(allowed_key), "invalid_evidence_role_owner", allowed_key))
    if not set(values).issubset(allowed):
        raise CanonicalStateContractError("invalid_role_eligibility", f"{name} includes unknown evidence roles.")
    return sorted(values)


def _normal_state_time(raw: object, metadata: Mapping[str, Mapping[str, object]], time_semantics: str) -> dict[str, object]:
    value = _mapping(raw, "invalid_state_time", "state_time")
    _exact_keys(value, ("unit", "values"), "invalid_state_time", "state_time")
    physical_semantics = {"absolute_time", "relative_to_action", "simulator_time"}
    if time_semantics in physical_semantics:
        if not metadata["time"]["axis_present"]:
            raise CanonicalStateContractError("time_semantics_axis_mismatch", "Physical and simulator time semantics require an explicit time axis.")
        if value.get("unit") != "hours":
            raise CanonicalStateContractError("invalid_state_time", "Physical and simulator time axes require hours.")
        times = [_finite(item, "invalid_state_time", "state time") for item in _list(value.get("values"), "invalid_state_time", "state_time.values")]
        if times != metadata["time"]["axis_values"]:
            raise CanonicalStateContractError("state_time_axis_mismatch", "state_time must equal declared time-axis values.")
        return {"unit": "hours", "values": times}
    if time_semantics == "pseudotime":
        if metadata["time"]["axis_present"] or value.get("unit") != "pseudotime":
            raise CanonicalStateContractError("pseudotime_axis_mismatch", "Pseudotime must not be mapped onto the physical canonical time axis.")
        values = [_finite(item, "invalid_state_time", "pseudotime") for item in _list(value.get("values"), "invalid_state_time", "state_time.values")]
        if not values:
            raise CanonicalStateContractError("invalid_state_time", "Pseudotime requires explicit source ordering values.")
        return {"unit": "pseudotime", "values": values}
    if metadata["time"]["axis_present"] or value.get("unit") is not None or value.get("values") != []:
        raise CanonicalStateContractError("state_time_missing_axis_mismatch", "Cross-sectional, ordered-endpoint, unknown, and not-applicable time semantics require a missing canonical time axis and non-numeric state_time.")
    return {"unit": None, "values": []}


def _content_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {field: deepcopy(payload[field]) for field in STATE_BLOCK_FIELDS if field not in {"state_block_uid", "content_hash"}}


def compile_partial_state_block(config: Mapping[str, object], raw_payload: Mapping[str, object]) -> DigitalTwinStateBlock:
    """Compile a logical state representation without densifying sparse/ragged/reference inputs."""
    contract = _contract(config)
    raw = _mapping(raw_payload, "invalid_state_block", "state block")
    allowed = set(RAW_STATE_BLOCK_FIELDS) | {"state_block_uid", "content_hash", "loss_eligibility"}
    missing = sorted(set(RAW_STATE_BLOCK_FIELDS) - set(raw))
    unknown = sorted(set(raw) - allowed)
    if missing or unknown:
        raise CanonicalStateContractError("invalid_state_block_fields", f"State block fields mismatch; missing={missing}, unknown={unknown}.")
    if raw.get("schema_version") != contract["serialization_version"]:
        raise CanonicalStateContractError("invalid_state_schema_version", "State block schema version does not match configured v2 contract.")
    axis_metadata = _axis_metadata(config, raw.get("axis_metadata"))
    storage, logical_values, logical_count = _value_storage(config, raw.get("value_storage"), axis_metadata)
    masks = _element_masks(raw.get("element_masks"), logical_count, logical_values, fmt=str(storage["format"]))
    context, evidence_type, time_semantics, pairing, organism = _metadata(config, raw, axis_metadata, logical_count)
    units = _units(config, raw.get("units"), axis_metadata)
    uncertainty = _uncertainty(raw.get("uncertainty"), logical_count, masks)
    qc_grade = _string(raw.get("qc_grade"), "invalid_qc_grade", "qc_grade")
    if qc_grade not in {"A", "B", "C", "D", "S"}:
        raise CanonicalStateContractError("invalid_qc_grade", "QC grade is unknown.")
    biological_unit_uid = _optional_string(raw.get("biological_unit_uid"), "invalid_biological_unit_uid", "biological_unit_uid")
    aggregation = _aggregation(raw.get("aggregation_semantics"), evidence_type, biological_unit_uid, contract["supported_aggregation_methods"])
    train_roles = _normal_roles(config, raw.get("allowed_train_roles"), "allowed_train_roles")
    eval_roles = _normal_roles(config, raw.get("allowed_eval_roles"), "allowed_eval_roles")
    if any(masks["quarantined"]) and train_roles:
        raise CanonicalStateContractError("quarantined_train_eligibility", "Quarantined elements cannot have training eligibility.")
    claim = _string(raw.get("claim_eligibility"), "invalid_claim_eligibility", "claim_eligibility")
    allowed_claims = set(_list(_mapping(config.get("digital_twin_evidence_fabric"), "missing_evidence_role_owner", "digital_twin_evidence_fabric").get("claim_eligibility_values"), "invalid_claim_eligibility", "claim_eligibility_values"))
    if claim not in allowed_claims:
        raise CanonicalStateContractError("invalid_claim_eligibility", "Claim eligibility is not configured.")
    if aggregation["aggregation_method"] == "legacy_aggregate_unknown" and claim != "ineligible":
        raise CanonicalStateContractError("legacy_aggregate_claim_promotion", "Legacy aggregate claim eligibility must remain ineligible.")
    loss = derive_loss_eligibility(config, evidence_type=evidence_type, time_semantics=time_semantics, pairing_semantics=pairing, qc_grade=qc_grade, masks=masks, allowed_train_roles=train_roles)
    supplied_loss = raw.get("loss_eligibility")
    if supplied_loss is not None and supplied_loss != loss:
        raise CanonicalStateContractError("loss_eligibility_mismatch", "Loss eligibility is derived and cannot be row-hardcoded.")
    normalized: dict[str, object] = {
        "schema_version": contract["serialization_version"],
        "evidence_uid": _string(raw.get("evidence_uid"), "invalid_evidence_uid", "evidence_uid"),
        "evidence_type": evidence_type,
        "source_uid": _string(raw.get("source_uid"), "invalid_source_uid", "source_uid"),
        "publication_accession": _optional_string(raw.get("publication_accession"), "invalid_publication_accession", "publication_accession"),
        "biological_unit_uid": biological_unit_uid,
        "state_uid": _string(raw.get("state_uid"), "invalid_state_uid", "state_uid"),
        "state_time": _normal_state_time(raw.get("state_time"), axis_metadata, time_semantics),
        "time_semantics": time_semantics,
        "organism": organism,
        "axis_metadata": axis_metadata,
        "feature_namespace": _string(raw.get("feature_namespace"), "invalid_feature_namespace", "feature_namespace"),
        "feature_ids": [_string(item, "invalid_feature_id", "feature id") for item in _list(raw.get("feature_ids"), "invalid_feature_ids", "feature_ids")],
        "value_storage": storage,
        "units": units,
        "biological_context": context,
        "uncertainty": uncertainty,
        "qc_grade": qc_grade,
        "pairing_semantics": pairing,
        "aggregation_semantics": aggregation,
        "element_masks": masks,
        "provenance": _provenance(raw.get("provenance")),
        "allowed_train_roles": train_roles,
        "allowed_eval_roles": eval_roles,
        "claim_eligibility": claim,
        "loss_eligibility": loss,
    }
    digest = _hash(normalized)
    uid = f"dt_state_block_{digest[:24]}"
    if raw.get("content_hash") is not None and raw["content_hash"] != digest:
        raise CanonicalStateContractError("content_hash_mismatch", "Supplied content hash does not match canonical state content.")
    if raw.get("state_block_uid") is not None and raw["state_block_uid"] != uid:
        raise CanonicalStateContractError("state_block_uid_mismatch", "Supplied state block UID does not match canonical content.")
    serialized = {"schema_version": normalized["schema_version"], "state_block_uid": uid}
    serialized.update({field: normalized[field] for field in STATE_BLOCK_FIELDS if field not in {"schema_version", "state_block_uid", "content_hash"}})
    serialized["content_hash"] = digest
    _json(serialized)
    return DigitalTwinStateBlock(serialized)


def deserialize_state_block(config: Mapping[str, object], payload: Mapping[str, object]) -> DigitalTwinStateBlock:
    serialized = _mapping(payload, "invalid_serialized_state_block", "serialized state block")
    _exact_keys(serialized, STATE_BLOCK_FIELDS, "invalid_serialized_state_block_fields", "serialized state block")
    return compile_partial_state_block(config, serialized)


def load_serialized_state_block(config: Mapping[str, object], path: str | Path) -> DigitalTwinStateBlock:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CanonicalStateContractError("unreadable_state_block", f"Unable to read serialized state block: {error}") from error
    return deserialize_state_block(config, _mapping(payload, "invalid_serialized_state_block", "serialized state block"))


def serialize_state_block(block: DigitalTwinStateBlock | Mapping[str, object]) -> str:
    payload = block.payload if isinstance(block, DigitalTwinStateBlock) else _mapping(block, "invalid_state_block", "state block")
    return _json(payload)


def state_block_content_hash(block: DigitalTwinStateBlock | Mapping[str, object]) -> str:
    payload = block.payload if isinstance(block, DigitalTwinStateBlock) else _mapping(block, "invalid_state_block", "state block")
    return _string(payload.get("content_hash"), "missing_content_hash", "content_hash")


def state_blocks_equal(left: DigitalTwinStateBlock | Mapping[str, object], right: DigitalTwinStateBlock | Mapping[str, object]) -> bool:
    left_payload = left.payload if isinstance(left, DigitalTwinStateBlock) else _mapping(left, "invalid_state_block", "left state block")
    right_payload = right.payload if isinstance(right, DigitalTwinStateBlock) else _mapping(right, "invalid_state_block", "right state block")
    return _json(left_payload) == _json(right_payload)


def logical_state_view(block: DigitalTwinStateBlock | Mapping[str, object]) -> dict[str, object]:
    payload = block.payload if isinstance(block, DigitalTwinStateBlock) else _mapping(block, "invalid_state_block", "state block")
    storage = _mapping(payload.get("value_storage"), "invalid_value_storage", "value_storage")
    fmt = _string(storage.get("format"), "invalid_storage_format", "value_storage.format")
    if fmt == "dense":
        count = math.prod(_list(storage.get("shape"), "invalid_storage_shape", "shape"))
    elif fmt == "sparse":
        count = len(_list(storage.get("sparse_entries"), "invalid_sparse_entries", "sparse_entries"))
    elif fmt == "ragged":
        count = len(_list(storage.get("ragged_entries"), "invalid_ragged_entries", "ragged_entries"))
    else:
        count = _mapping(storage.get("reference"), "invalid_reference", "reference")["entry_count"]
    return {"format": fmt, "source_axis_order": deepcopy(storage["source_axis_order"]), "logical_element_count": count, "densified": False}


def iter_logical_elements(block: DigitalTwinStateBlock | Mapping[str, object]) -> Iterator[dict[str, object]]:
    """Yield logical elements lazily; no dense projection is constructed for sparse/ragged/reference storage."""
    payload = block.payload if isinstance(block, DigitalTwinStateBlock) else _mapping(block, "invalid_state_block", "state block")
    storage = _mapping(payload["value_storage"], "invalid_value_storage", "value_storage")
    masks = _mapping(payload["element_masks"], "invalid_element_masks", "element_masks")
    ids = _list(payload["feature_ids"], "invalid_feature_ids", "feature_ids")
    fmt = storage["format"]
    if fmt == "dense":
        raw_entries = ({"index": list(index), "value": value} for index, value in _visit_dense(storage["dense_values"], storage["shape"]))
    elif fmt == "sparse":
        raw_entries = _list(storage["sparse_entries"], "invalid_sparse_entries", "sparse_entries")
    elif fmt == "ragged":
        raw_entries = _list(storage["ragged_entries"], "invalid_ragged_entries", "ragged_entries")
    else:
        count = _mapping(storage["reference"], "invalid_reference", "reference")["entry_count"]
        raw_entries = ({"reference_index": index, "value": None} for index in range(int(count)))
    for index, entry in enumerate(raw_entries):
        state = next(name for name in ELEMENT_MASK_FIELDS if masks[name][index])
        yield {"logical_index": index, "feature_id": ids[index], "mask_state": state, "storage_entry": deepcopy(entry)}


def state_block_validation_report(config: Mapping[str, object], payload: Mapping[str, object]) -> dict[str, object]:
    try:
        block = deserialize_state_block(config, payload)
    except CanonicalStateContractError as error:
        return {"schema_version": canonical_state_config(config).get("validation_version"), "status": "failed", "blocker": {"code": error.code, "message": str(error).split(": ", 1)[-1]}}
    masks = block.payload["element_masks"]
    return {
        "schema_version": canonical_state_config(config).get("validation_version"), "status": "passed",
        "state_block_uid": block.state_block_uid, "content_hash": block.content_hash,
        "logical_view": logical_state_view(block),
        "mask_counts": {state: sum(masks[state]) for state in ELEMENT_MASK_FIELDS},
        "loss_eligibility": block.payload["loss_eligibility"],
    }
