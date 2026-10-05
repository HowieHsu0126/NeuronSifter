"""Hash-bound, ESS-aware loader for the frozen C1 transcriptomics core.

The loader deliberately keeps within-biological-unit observations separate.  The
objective layer is responsible for reducing observation losses to a biological
unit before applying family-balanced weights.  This is important for the atlas
blocks, where a donor has many cells but an effective sample size of one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

from Libs.utils.data.foundation_state_c1 import (
    C1LivePreflightAttestation,
    C1Unit,
    attest_live_c1_preflight,
    build_qualified_action_subset_manifest,
    family_balanced_unit_weights,
    load_c1_split_units,
    require_live_c1_preflight,
    validate_c1_preflight_attestation,
    validate_c1_contract,
)
from Libs.utils.data.foundation_state_c1_semantics import (
    qualified_semantic_alignment_key,
)
from Libs.utils.data.foundation_state_c1_actions import qualified_physical_exposure
from Libs.utils.data.foundation_state_c1_sampler import (
    C1_REPLAY_COORDINATES,
    C1_REPLAY_POLICY,
    C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT,
    C1_SAMPLER_BATCH_SIZE,
    OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY,
    SAMPLER_CONTRACT_VERSION,
    C1ObjectiveRequirementRef,
    plan_objective_requirements,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file
from Libs.utils.data.raw_data_registry import load_config


FEATURE_NORMALIZATION_VERSION = "neuronsifter-c1-feature-token-v1"
FAMILY_AND_UNIT_BALANCED_POLICY = "family_and_biological_unit_balanced"
FAMILY_BALANCED_POLICY = "family_balanced"
BIOLOGICAL_UNIT_BALANCED_POLICY = "biological_unit_balanced"
ROW_BALANCED_POLICY = "row_balanced"
FEATURE_VALUE_NORMALIZATION_VERSION = (
    "neuronsifter-c1-observation-zscore-leave-one-masked-out-v2"
)
ACTION_FEATURE_SCHEMA_VERSION = "neuronsifter-c1-action-vector-v1"
ACTION_FEATURE_DIM = 8
ACTION_DESCRIPTOR_VERSION = "neuronsifter-c1-physical-exposure-descriptor-v1"
OBJECTIVE_AWARE_SAMPLER_VERSION = SAMPLER_CONTRACT_VERSION
C1_LOADER_IMPLEMENTATION_VERSION = "neuronsifter-c1-transcriptomics-loader-v2"
_MISSING = "__missing__"
_NAMESPACE_ALIASES = {
    "NCBI_Entrez_Gene": "entrez_gene_id",
    "NCBI_Entrez_gene": "entrez_gene_id",
    "entrez_gene_id": "entrez_gene_id",
    "Ensembl_Gene": "ensembl_gene_id",
    "ensembl_gene": "ensembl_gene_id",
    "ensembl_gene_id": "ensembl_gene_id",
}
_ACTION_DESCRIPTOR_COLUMNS = (
    "action_uid",
    "physical_dose",
    "dose_value",
    "dose_unit",
    "duration",
    "duration_unit",
    "schedule",
    "route",
    "route_or_not_applicable",
    "target_set",
    "target_set_or_explicit_unknown",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def _canonical_feature_key(namespace: object, value: object) -> str:
    ns = _text(namespace)
    raw = _text(value)
    if not raw:
        raise ValueError("C1 feature metadata contains an empty canonical feature ID")
    if ":" in raw and not ns:
        prefix, candidate = raw.split(":", 1)
        ns = prefix
        raw = candidate
    ns = _NAMESPACE_ALIASES.get(ns, ns.lower() if ns else "")
    if not ns:
        raise ValueError(f"C1 feature metadata has no namespace: {value!r}")
    if ns == "entrez_gene_id" and raw.startswith("NCBIGene:"):
        raw = raw.split(":", 1)[1]
    return f"{ns}|{raw}"


def _feature_columns(path: Path) -> tuple[str, str, str]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    if {"canonical_gene_id", "canonical_namespace"} <= names:
        return "canonical_gene_id", "canonical_namespace", "canonical_gene_id"
    if {"canonical_feature_id", "canonical_namespace"} <= names:
        return "canonical_feature_id", "canonical_namespace", "canonical_feature_id"
    if {"feature_id", "feature_namespace"} <= names:
        return "feature_id", "feature_namespace", "feature_id"
    raise ValueError(f"C1 feature table has no canonical ID/namespace columns: {path}")


def _feature_records(path: Path) -> list[tuple[int, str]]:
    id_column, namespace_column, _ = _feature_columns(path)
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [id_column, namespace_column]
    if "feature_index" in names:
        columns.append("feature_index")
    table = pq.read_table(path, columns=columns)
    values = table.to_pydict()
    ids = values[id_column]
    namespaces = values[namespace_column]
    indices = values.get("feature_index", list(range(len(ids))))
    records: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index, namespace, value in zip(indices, namespaces, ids, strict=True):
        key = _canonical_feature_key(namespace, value)
        if key in seen:
            raise ValueError(f"Duplicate canonical feature key in {path}: {key}")
        seen.add(key)
        records.append((int(index), key))
    if not records:
        raise ValueError(f"C1 feature table is empty: {path}")
    return records


def _metadata_projection(path: Path, key_column: str) -> list[dict[str, object]]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [key_column]
    for candidate in (
        "biological_unit_id",
        "biological_unit_uid",
        "donor_id",
        "donor_label",
        "value_row_index",
        "cell_type_id",
        "cell_type",
        "cell_type_label",
        "cell_or_tissue_context",
        "brain_region_ontology_id",
        "brain_region_id",
        "brain_region_canonical",
        "brain_region_label",
        "brain_region_original",
        "disease_state_id",
        "disease_state_canonical",
        "disease_state",
        "observation_uid",
        "action_uid",
        "dose_value",
        "time_value",
        "duration",
        "route",
        "schedule",
        "compound_uid",
        "agent_uid",
    ):
        if candidate in names and candidate not in columns:
            columns.append(candidate)
    table = pq.read_table(path, columns=columns)
    data = table.to_pydict()
    row_count = table.num_rows
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        row = {column: data[column][index] for column in columns}
        row["__ordinal"] = index
        rows.append(row)
    return rows


def _first_label(row: Mapping[str, object], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        value = _text(row.get(candidate))
        if value:
            return value
    return ""


def _standardize_selected_feature_values(
    values: Sequence[float], *, excluded_index: int | None = None
) -> list[float]:
    """Remove assay-scale magnitude while preserving within-observation signal."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("C1 selected feature values must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError("C1 selected feature values must be finite")
    fit_values = array
    if excluded_index is not None:
        if excluded_index < 0 or excluded_index >= array.size:
            raise ValueError("C1 excluded feature index is outside the selected vector")
        fit_values = np.delete(array, excluded_index)
        if fit_values.size == 0:
            raise ValueError(
                "C1 masked normalization requires at least one visible feature"
            )
    center = float(fit_values.mean())
    scale = float(fit_values.std())
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    normalized = (array - center) / scale
    return normalized.astype(np.float32).tolist()


@dataclass(frozen=True)
class C1Vocabulary:
    feature_to_id: Mapping[str, int]
    cell_to_id: Mapping[str, int]
    region_to_id: Mapping[str, int]
    source_to_id: Mapping[str, int]
    family_to_id: Mapping[str, int]
    feature_mapping_sha256: str
    cell_mapping_sha256: str
    region_mapping_sha256: str
    source_mapping_sha256: str
    family_mapping_sha256: str

    @property
    def feature_count(self) -> int:
        return len(self.feature_to_id)

    @property
    def feature_vocabulary_size(self) -> int:
        """Embedding rows, including reserved padding ID 0."""

        return max(self.feature_to_id.values(), default=0) + 1

    @property
    def cell_count(self) -> int:
        return len(self.cell_to_id)

    @property
    def cell_ontology_size(self) -> int:
        """Embedding rows, including reserved missing-ID 0."""

        return max(self.cell_to_id.values(), default=0) + 1

    @property
    def region_count(self) -> int:
        return len(self.region_to_id)

    @property
    def region_ontology_size(self) -> int:
        """Embedding rows, including reserved missing-ID 0."""

        return max(self.region_to_id.values(), default=0) + 1

    @property
    def source_count(self) -> int:
        return len(self.source_to_id)

    @property
    def family_count(self) -> int:
        return len(self.family_to_id)


@dataclass(frozen=True)
class _ActionDescriptor:
    vector: tuple[float, ...]
    target_set_available: bool
    agent_identity_available: bool


@dataclass(frozen=True)
class C1LoaderContext:
    """Reusable in-process bindings created by one complete live attestation."""

    attestation: C1LivePreflightAttestation
    binding: Mapping[str, object]
    binding_digest: str
    vocabulary: C1Vocabulary
    vocabulary_digest: str
    action_descriptors: Mapping[str, _ActionDescriptor]
    action_descriptor_digest: str


@dataclass(frozen=True)
class C1RunScopeAttestation:
    """Live run-preflight capability bound to one exact loader scope."""

    run_preflight_report: Mapping[str, object]
    run_preflight_digest: str
    run_scope_binding: Mapping[str, object]
    run_scope_binding_digest: str
    _capability_id: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True)
class C1ReplayScopeAttestation:
    """Preflight-only capability authorizing exact loader replay."""

    run_scope_binding: Mapping[str, object]
    run_scope_binding_digest: str
    _capability_id: object = dataclass_field(repr=False, compare=False)


@dataclass
class _ReplayCapabilityRecord:
    attestation: C1ReplayScopeAttestation
    run_scope_binding_digest: str
    observed_calls: list[tuple[int, int, str]]


@dataclass(frozen=True)
class _RunScopeCapabilityRecord:
    attestation: C1RunScopeAttestation
    run_preflight_digest: str
    run_scope_binding_digest: str


_REPLAY_CAPABILITIES: dict[object, _ReplayCapabilityRecord] = {}
_RUN_SCOPE_CAPABILITIES: dict[object, _RunScopeCapabilityRecord] = {}
_EXPECTED_REPLAY_CALL_SEQUENCE = tuple(
    coordinate for coordinate in C1_REPLAY_COORDINATES for _ in range(2)
)


_RUN_SCOPE_BINDING_FIELDS = {
    "corpus_uid",
    "corpus_hash",
    "observation_binding_hash",
    "split_uid",
    "split_manifest_hash",
    "split_role",
    "role_binding_path",
    "role_binding_sha256",
    "source_scope",
    "source_id",
    "family_scope",
    "family_uid",
    "baseline_id",
    "model_profile",
    "objective_set",
    "active_objectives",
    "sampling_policy",
    "weighting_policy",
    "sampler_contract_version",
    "batch_size",
    "max_observations_per_biological_unit",
    "seed",
    "contract_digest",
    "live_preflight_digest",
    "loader_implementation_version",
    "vocabulary_manifest_sha256",
    "action_descriptor_sha256",
    "qualified_action_subset_hash",
    "objective_capability_manifest_sha256",
    "frozen_block_binding_hash",
    "sampler_profile_hash",
    "run_scope_binding_hash",
}
_SAMPLER_PROFILE_FIELDS = (
    "sampler_contract_version",
    "source_scope",
    "source_id",
    "baseline_id",
    "model_profile",
    "objective_set",
    "active_objectives",
    "sampling_policy",
    "weighting_policy",
    "batch_size",
    "max_observations_per_biological_unit",
)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def c1_sampler_profile_hash(binding: Mapping[str, object]) -> str:
    missing = set(_SAMPLER_PROFILE_FIELDS) - set(binding)
    if missing:
        raise ValueError(f"C1 sampler profile is missing fields: {sorted(missing)}")
    return canonical_sha256(
        {field: binding[field] for field in _SAMPLER_PROFILE_FIELDS}
    )


def _run_scope_binding_hash(binding: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in binding.items()
            if key != "run_scope_binding_hash"
        }
    )


def _validate_run_scope_binding_shape(binding: Mapping[str, object]) -> None:
    # Preserve read-only compatibility with pre-family-scope development attestations.
    # Newly issued bindings always include these fields; old all-family bindings are
    # normalized in place before the fail-closed shape check.
    if "family_scope" not in binding and "family_uid" not in binding:
        if isinstance(binding, dict):
            binding["family_scope"] = "all_families"
            binding["family_uid"] = None
            binding["run_scope_binding_hash"] = _run_scope_binding_hash(binding)
    if set(binding) != _RUN_SCOPE_BINDING_FIELDS:
        raise ValueError(
            "C1 run scope binding fields do not match the sampler contract: "
            f"missing={sorted(_RUN_SCOPE_BINDING_FIELDS - set(binding))}, "
            f"extra={sorted(set(binding) - _RUN_SCOPE_BINDING_FIELDS)}"
        )
    text_fields = _RUN_SCOPE_BINDING_FIELDS - {
        "source_id",
        "family_uid",
        "active_objectives",
        "batch_size",
        "max_observations_per_biological_unit",
        "seed",
    }
    for field in text_fields:
        if type(binding[field]) is not str or not str(binding[field]).strip():
            raise TypeError(f"C1 run scope binding {field} must be a non-empty string")
    for field in (
        "corpus_hash",
        "observation_binding_hash",
        "split_manifest_hash",
        "role_binding_sha256",
        "contract_digest",
        "live_preflight_digest",
        "vocabulary_manifest_sha256",
        "action_descriptor_sha256",
        "qualified_action_subset_hash",
        "objective_capability_manifest_sha256",
        "frozen_block_binding_hash",
        "sampler_profile_hash",
        "run_scope_binding_hash",
    ):
        if not _is_sha256(binding[field]):
            raise ValueError(f"C1 run scope binding {field} is not a SHA-256 digest")
    if binding["split_role"] != "train":
        raise ValueError("C1 run scope binding must attest the frozen train role")
    if binding["sampler_contract_version"] != OBJECTIVE_AWARE_SAMPLER_VERSION:
        raise ValueError("C1 run scope binding sampler contract version drift")
    if binding["source_scope"] not in {"all_sources", "single_source"}:
        raise ValueError("C1 run scope binding source scope is invalid")
    source_id = binding["source_id"]
    if binding["source_scope"] == "single_source":
        if type(source_id) is not str or not source_id.strip():
            raise TypeError("C1 single-source run scope requires a source_id")
    elif source_id is not None:
        raise ValueError("C1 all-sources run scope must use source_id=null")
    if binding["family_scope"] not in {"all_families", "single_family"}:
        raise ValueError("C1 run scope binding family scope is invalid")
    family_uid = binding["family_uid"]
    if binding["family_scope"] == "single_family":
        if type(family_uid) is not str or not family_uid.strip():
            raise TypeError("C1 single-family run scope requires a family_uid")
    elif family_uid is not None:
        raise ValueError("C1 all-families run scope must use family_uid=null")
    active = binding["active_objectives"]
    if type(active) is not list or any(
        type(value) is not str or not value for value in active
    ):
        raise TypeError("C1 run scope active_objectives must be a string list")
    if len(set(active)) != len(active):
        raise ValueError("C1 run scope active_objectives contains duplicates")
    for field in ("batch_size", "max_observations_per_biological_unit"):
        if type(binding[field]) is not int or int(binding[field]) <= 0:
            raise TypeError(f"C1 run scope binding {field} must be a positive integer")
    if binding["batch_size"] != C1_SAMPLER_BATCH_SIZE:
        raise ValueError(
            "C1 run scope binding batch size is outside the frozen sampler"
        )
    if (
        binding["max_observations_per_biological_unit"]
        != C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT
    ):
        raise ValueError("C1 run scope observation cap is outside the frozen sampler")
    if type(binding["seed"]) is not int or int(binding["seed"]) < 0:
        raise TypeError("C1 run scope binding seed must be a non-negative integer")
    if binding["loader_implementation_version"] != C1_LOADER_IMPLEMENTATION_VERSION:
        raise ValueError("C1 run scope loader implementation version drift")
    if c1_sampler_profile_hash(binding) != binding["sampler_profile_hash"]:
        raise ValueError("C1 run scope sampler profile hash drift")
    if _run_scope_binding_hash(binding) != binding["run_scope_binding_hash"]:
        raise ValueError("C1 run scope binding hash drift")


def _validate_runtime_replay_evidence(runtime_replay: Mapping[str, object]) -> None:
    expected_replay_fields = {
        "status",
        "policy",
        "coordinates",
        "replay_verified",
        "batch_guarantee",
        "blocker",
    }
    if set(runtime_replay) != expected_replay_fields:
        raise ValueError("C1 runtime replay fields drift")
    if runtime_replay.get("status") != "verified":
        raise ValueError("C1 runtime replay status is not verified")
    if runtime_replay.get("policy") != C1_REPLAY_POLICY:
        raise ValueError("C1 runtime replay policy drift")
    coordinates = runtime_replay.get("coordinates")
    if type(coordinates) is not list:
        raise TypeError("C1 runtime replay coordinates must be a list")
    actual_coordinates: list[tuple[int, int]] = []
    for record in coordinates:
        if not isinstance(record, Mapping):
            raise TypeError("C1 runtime replay coordinate must be a mapping")
        if set(record) != {
            "epoch",
            "batch_index",
            "batch_hash",
            "repeated_batch_hash",
            "deterministic",
            "postconditions",
        }:
            raise ValueError("C1 runtime replay coordinate fields drift")
        epoch = record.get("epoch")
        batch_index = record.get("batch_index")
        if type(epoch) is not int or type(batch_index) is not int:
            raise TypeError("C1 runtime replay coordinate indices must be integers")
        actual_coordinates.append((epoch, batch_index))
        batch_hash = record.get("batch_hash")
        repeated_batch_hash = record.get("repeated_batch_hash")
        if not _is_sha256(batch_hash) or not _is_sha256(repeated_batch_hash):
            raise ValueError("C1 runtime replay coordinate has an invalid batch hash")
        if batch_hash != repeated_batch_hash:
            raise ValueError("C1 runtime replay double-replay hash mismatch")
        if record.get("deterministic") is not True:
            raise ValueError("C1 runtime replay coordinate is not deterministic")
        postconditions = record.get("postconditions")
        if not isinstance(postconditions, Mapping):
            raise TypeError("C1 runtime replay coordinate has no postconditions")
        expected_postcondition_fields = {
            "all_passed",
            "observation_count",
            "biological_unit_count",
            "non_empty_observation_batch",
            "exact_biological_unit_batch_size",
            "observation_unit_index_coverage",
            "within_unit_observation_cap_respected",
            "tensor_shape_dtype_contract",
            "finite_numeric_tensors",
            "feature_id_range_valid",
            "ontology_and_nuisance_id_ranges_valid",
            "mask_partition_valid",
            "padding_contract_valid",
            "unit_weights_normalized",
            "unit_weights_nonnegative",
            "unavailable_action_zero_vector",
            "target_mask_present_per_observation",
            "action_requirement_satisfied",
            "alignment_requirement_satisfied",
            "family_balance_requirement_satisfied",
        }
        if set(postconditions) != expected_postcondition_fields:
            raise ValueError("C1 runtime replay postcondition fields drift")
        for field in ("observation_count", "biological_unit_count"):
            if type(postconditions[field]) is not int or postconditions[field] <= 0:
                raise ValueError(f"C1 runtime replay {field} is invalid")
        boolean_fields = expected_postcondition_fields - {
            "observation_count",
            "biological_unit_count",
        }
        if any(postconditions[field] is not True for field in boolean_fields):
            raise ValueError("C1 runtime replay postconditions have not all passed")
    if tuple(actual_coordinates) != C1_REPLAY_COORDINATES:
        raise ValueError("C1 runtime replay coordinates drift")
    if (
        runtime_replay.get("replay_verified") is not True
        or runtime_replay.get("batch_guarantee") is not True
        or runtime_replay.get("blocker") != "none"
    ):
        raise ValueError("C1 runtime replay has not established a batch guarantee")


def attest_c1_run_scope(
    run_preflight_report: Mapping[str, object],
    replay_scope_attestation: C1ReplayScopeAttestation | None = None,
) -> C1RunScopeAttestation:
    """Convert one fully replay-verified report into a training capability."""

    if not isinstance(run_preflight_report, Mapping):
        raise TypeError("C1 run preflight report must be a mapping")
    if run_preflight_report.get("status") != "passed":
        raise ValueError("C1 run preflight report has not passed")
    if run_preflight_report.get("training_started") is not False:
        raise ValueError("C1 run preflight attestation requires training_started=false")
    runtime_replay = run_preflight_report.get("runtime_replay")
    if not isinstance(runtime_replay, Mapping):
        raise TypeError("C1 run preflight report has no runtime_replay result")
    _validate_runtime_replay_evidence(runtime_replay)
    if (
        run_preflight_report.get("replay_verified") is not True
        or run_preflight_report.get("batch_guarantee") is not True
    ):
        raise ValueError(
            "C1 run preflight report has not verified runtime replay and batch guarantees"
        )
    binding = run_preflight_report.get("run_scope_binding")
    if not isinstance(binding, Mapping):
        raise TypeError("C1 run preflight report has no run_scope_binding")
    _validate_run_scope_binding_shape(binding)
    if run_preflight_report.get("run_scope_binding_hash") != binding.get(
        "run_scope_binding_hash"
    ):
        raise ValueError("C1 run preflight top-level run scope hash drift")
    if run_preflight_report.get("baseline_id") != binding.get("baseline_id"):
        raise ValueError("C1 run preflight top-level baseline identity drift")
    if run_preflight_report.get("training_authorization_status") != (
        "runtime_replay_verified"
    ):
        raise ValueError("C1 run preflight is not a training authorization")
    if type(replay_scope_attestation) is not C1ReplayScopeAttestation:
        raise TypeError(
            "C1 training attestation requires the live opaque replay capability"
        )
    replay_record = _REPLAY_CAPABILITIES.get(replay_scope_attestation._capability_id)
    if replay_record is None:
        raise ValueError("C1 replay capability is unknown, consumed, or forged")
    if (
        replay_record.attestation is not replay_scope_attestation
        or replay_record.run_scope_binding_digest
        != replay_scope_attestation.run_scope_binding_digest
        or canonical_sha256(replay_scope_attestation.run_scope_binding)
        != replay_scope_attestation.run_scope_binding_digest
        or canonical_sha256(binding)
        != replay_scope_attestation.run_scope_binding_digest
    ):
        raise ValueError("C1 replay capability does not match the run scope")
    expected_calls: list[tuple[int, int, str]] = []
    for coordinate in runtime_replay["coordinates"]:
        expected_calls.extend(
            [
                (
                    int(coordinate["epoch"]),
                    int(coordinate["batch_index"]),
                    str(coordinate["batch_hash"]),
                ),
                (
                    int(coordinate["epoch"]),
                    int(coordinate["batch_index"]),
                    str(coordinate["repeated_batch_hash"]),
                ),
            ]
        )
    if replay_record.observed_calls != expected_calls:
        raise ValueError(
            "C1 runtime replay report is not backed by the live loader calls"
        )
    del _REPLAY_CAPABILITIES[replay_scope_attestation._capability_id]
    report_copy = json.loads(
        json.dumps(run_preflight_report, sort_keys=True, separators=(",", ":"))
    )
    binding_copy = report_copy["run_scope_binding"]
    capability_id = object()
    attestation = C1RunScopeAttestation(
        run_preflight_report=report_copy,
        run_preflight_digest=canonical_sha256(report_copy),
        run_scope_binding=binding_copy,
        run_scope_binding_digest=canonical_sha256(binding_copy),
        _capability_id=capability_id,
    )
    _RUN_SCOPE_CAPABILITIES[capability_id] = _RunScopeCapabilityRecord(
        attestation=attestation,
        run_preflight_digest=attestation.run_preflight_digest,
        run_scope_binding_digest=attestation.run_scope_binding_digest,
    )
    return attestation


def authorize_c1_run_scope_replay(
    run_scope_binding: Mapping[str, object],
) -> C1ReplayScopeAttestation:
    """Authorize one exact loader construction for run-preflight replay only."""

    if not isinstance(run_scope_binding, Mapping):
        raise TypeError("C1 replay run scope binding must be a mapping")
    _validate_run_scope_binding_shape(run_scope_binding)
    binding_copy = json.loads(
        json.dumps(run_scope_binding, sort_keys=True, separators=(",", ":"))
    )
    capability_id = object()
    attestation = C1ReplayScopeAttestation(
        run_scope_binding=binding_copy,
        run_scope_binding_digest=canonical_sha256(binding_copy),
        _capability_id=capability_id,
    )
    _REPLAY_CAPABILITIES[capability_id] = _ReplayCapabilityRecord(
        attestation=attestation,
        run_scope_binding_digest=attestation.run_scope_binding_digest,
        observed_calls=[],
    )
    return attestation


def release_c1_run_scope_attestation(
    attestation: C1RunScopeAttestation,
) -> None:
    """Release an unused live training capability without making it portable."""

    if type(attestation) is not C1RunScopeAttestation:
        raise TypeError("C1 run scope release requires an exact live capability")
    record = _RUN_SCOPE_CAPABILITIES.get(attestation._capability_id)
    if record is None or record.attestation is not attestation:
        raise ValueError("C1 run scope capability is unknown, released, or forged")
    del _RUN_SCOPE_CAPABILITIES[attestation._capability_id]


def _vocabulary_digest(vocabulary: C1Vocabulary) -> str:
    return canonical_sha256(
        {
            "feature_to_id": sorted(vocabulary.feature_to_id.items()),
            "cell_to_id": sorted(vocabulary.cell_to_id.items()),
            "region_to_id": sorted(vocabulary.region_to_id.items()),
            "source_to_id": sorted(vocabulary.source_to_id.items()),
            "family_to_id": sorted(vocabulary.family_to_id.items()),
            "mapping_hashes": [
                vocabulary.feature_mapping_sha256,
                vocabulary.cell_mapping_sha256,
                vocabulary.region_mapping_sha256,
                vocabulary.source_mapping_sha256,
                vocabulary.family_mapping_sha256,
            ],
        }
    )


def _action_descriptor_digest(
    descriptors: Mapping[str, _ActionDescriptor],
) -> str:
    return canonical_sha256(
        [
            {
                "action_uid": uid,
                "vector": list(descriptor.vector),
                "target_set_available": descriptor.target_set_available,
                "agent_identity_available": descriptor.agent_identity_available,
            }
            for uid, descriptor in sorted(descriptors.items())
        ]
    )


def _action_feature_spec(
    records: Sequence[Mapping[str, object]],
    *,
    action_feature_dim: int = ACTION_FEATURE_DIM,
    descriptor_version: str = ACTION_DESCRIPTOR_VERSION,
) -> tuple[dict[str, _ActionDescriptor], tuple[str, ...], str, dict[str, list[str]]]:
    if not records:
        raise ValueError("C1 qualified action catalog is empty")
    by_uid: dict[str, Mapping[str, object]] = {}
    for row in records:
        uid = _text(row.get("action_uid"))
        if not uid or uid in by_uid:
            raise ValueError(
                f"C1 qualified action UID is missing or duplicated: {uid!r}"
            )
        by_uid[uid] = row
    exposures = {uid: qualified_physical_exposure(row) for uid, row in by_uid.items()}
    categories = {
        "dose_unit": sorted({exposure.dose_unit for exposure in exposures.values()}),
        "schedule": sorted({exposure.schedule for exposure in exposures.values()}),
        "route": sorted({exposure.route for exposure in exposures.values()}),
    }
    if any(not values or "" in values for values in categories.values()):
        raise ValueError("C1 qualified actions contain an empty categorical field")
    # The descriptor dimension is frozen per corpus version. V2 adds qualified
    # schedule and route categories and therefore uses a distinct contract.
    if 2 + sum(len(values) for values in categories.values()) != action_feature_dim:
        raise ValueError(
            "C1 action categorical cardinalities do not match the frozen descriptor"
        )
    feature_names = ["log1p_dose_value", "log1p_duration_hours"]
    for field in ("dose_unit", "schedule", "route"):
        feature_names.extend(f"{field}={value}" for value in categories[field])
    descriptors: dict[str, _ActionDescriptor] = {}
    for uid, row in by_uid.items():
        exposure = exposures[uid]
        vector = [0.0] * action_feature_dim
        vector[0] = math.log1p(exposure.dose_value)
        vector[1] = math.log1p(exposure.duration_hours)
        offset = 2
        for field, value in (
            ("dose_unit", exposure.dose_unit),
            ("schedule", exposure.schedule),
            ("route", exposure.route),
        ):
            index = categories[field].index(value)
            vector[offset + index] = 1.0
            offset += len(categories[field])
        target = _text(row.get("target_set")) or _text(
            row.get("target_set_or_explicit_unknown")
        )
        target_available = target not in {
            "",
            "explicit_unknown",
            "explicit_unavailable",
        }
        # Qualified catalogs only provide source-native agent IDs. Keep
        # this fact explicit rather than feeding a source-prefixed hash.
        descriptors[uid] = _ActionDescriptor(
            vector=tuple(vector),
            target_set_available=target_available,
            agent_identity_available=False,
        )
    spec_payload = {
        "version": descriptor_version,
        "feature_names": feature_names,
        "categories": categories,
        "target_set_available_count": sum(
            int(value.target_set_available) for value in descriptors.values()
        ),
    }
    return (
        descriptors,
        tuple(feature_names),
        canonical_sha256(spec_payload),
        categories,
    )


def _mapping(values: Iterable[str], *, missing_id: bool) -> tuple[dict[str, int], str]:
    unique = sorted({value for value in values if value})
    start = 1 if missing_id else 0
    mapping = {value: index + start for index, value in enumerate(unique)}
    payload = [[key, mapping[key]] for key in unique]
    return mapping, canonical_sha256(payload)


def _binding_manifest(
    config: Mapping[str, object], repo_root: Path
) -> dict[str, object]:
    frozen = config["foundation_state_c1"]["frozen_corpus"]
    required = frozen["required_observation_binding"]
    manifest_path = safe_repo_path(repo_root, str(required["manifest_path"]))
    row_path = safe_repo_path(repo_root, str(required["row_table_path"]))
    if sha256_file(manifest_path) != str(required["manifest_sha256"]):
        raise ValueError("C1 observation binding manifest hash drift")
    if sha256_file(row_path) != str(required["row_table_sha256"]):
        raise ValueError("C1 observation binding row-table hash drift")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("corpus_hash") != frozen["identity_hash"]:
        raise ValueError("C1 observation binding corpus hash drift")
    if manifest.get("binding_hash") != required["binding_hash"]:
        raise ValueError("C1 observation binding hash drift")
    return manifest


def _qualified_action_records(
    repo_root: Path, config: Mapping[str, object]
) -> list[dict[str, object]]:
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    manifest_path = safe_repo_path(
        repo_root, str(frozen["qualified_action_subset_manifest_path"])
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    evidence_path = safe_repo_path(repo_root, str(frozen["evidence_bundle_path"]))
    with evidence_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    action_catalog_path = safe_repo_path(
        repo_root, str(bundle["evidence"]["structured_action_catalog"]["path"])
    )
    with action_catalog_path.open("r", encoding="utf-8") as handle:
        action_catalog = json.load(handle)
    rebuilt = build_qualified_action_subset_manifest(bundle, action_catalog, repo_root)
    if (
        manifest != rebuilt
        or manifest.get("qualified_action_subset_hash")
        != frozen["qualified_action_subset_hash"]
    ):
        raise ValueError("C1 qualified action subset hash drift")
    records: list[dict[str, object]] = []
    for catalog in manifest["qualified_catalogs"]:
        path = safe_repo_path(repo_root, str(catalog["path"]))
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = set(_ACTION_DESCRIPTOR_COLUMNS) - names
        if missing:
            raise ValueError(
                f"C1 qualified action catalog lacks descriptor fields: {path}:{sorted(missing)}"
            )
        table = pq.read_table(path, columns=list(_ACTION_DESCRIPTOR_COLUMNS))
        if table.num_rows != int(catalog["record_count"]):
            raise ValueError(f"C1 qualified action record-count drift: {path}")
        records.extend(table.to_pylist())
    if len(records) != int(frozen["required_qualified_action_count"]):
        raise ValueError("C1 qualified action record total drift")
    return records


def _build_c1_vocabulary_attested(
    repo_root: Path,
    config: Mapping[str, object],
    binding: Mapping[str, object],
) -> tuple[C1Vocabulary, dict[str, object]]:
    """Derive the vocabulary after the caller has completed live attestation."""

    repo_root = repo_root.resolve()
    contract = config.get("foundation_state_c1")
    if not isinstance(contract, Mapping):
        raise ValueError("foundation_state_c1 contract is missing")
    frozen = contract["frozen_corpus"]
    crosswalk = frozen["required_crosswalk"]
    crosswalk_path = safe_repo_path(repo_root, str(crosswalk["table_path"]))
    if sha256_file(crosswalk_path) != str(crosswalk["table_sha256"]):
        raise ValueError("C1 vocabulary crosswalk hash drift")
    with crosswalk_path.open("r", encoding="utf-8", newline="") as handle:
        crosswalk_record_count = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    if crosswalk_record_count != int(crosswalk["record_count"]):
        raise ValueError("C1 vocabulary crosswalk record-count drift")
    action_records = _qualified_action_records(repo_root, config)
    input_contract = contract["encoder"]["input_contract"]
    action_feature_dim = int(input_contract["action_feature_dim"])
    descriptor_version = str(input_contract["action_descriptor_version"])
    action_descriptors, action_feature_names, action_spec_hash, action_categories = (
        _action_feature_spec(
            action_records,
            action_feature_dim=action_feature_dim,
            descriptor_version=descriptor_version,
        )
    )
    feature_keys: set[str] = set()
    block_feature_hashes: dict[str, str] = {}
    cell_labels: set[str] = set()
    region_labels: set[str] = set()
    for block in binding["blocks"]:
        feature_path = safe_repo_path(repo_root, str(block["features"]["path"]))
        feature_keys.update(key for _, key in _feature_records(feature_path))
        block_feature_hashes[str(block["block_uid"])] = str(block["features"]["sha256"])
        metadata_path = safe_repo_path(repo_root, str(block["metadata"]["path"]))
        rows = _metadata_projection(metadata_path, str(block["metadata"]["key_column"]))
        cell_labels.update(
            _first_label(
                row,
                (
                    "cell_type_id",
                    "cell_type",
                    "cell_type_label",
                    "cell_or_tissue_context",
                ),
            )
            for row in rows
        )
        region_labels.update(
            _first_label(
                row,
                (
                    "brain_region_ontology_id",
                    "brain_region_id",
                    "brain_region_canonical",
                    "brain_region_label",
                    "brain_region_original",
                ),
            )
            for row in rows
        )

    query_path = safe_repo_path(
        repo_root, str(contract["frozen_corpus"]["required_query_domain"]["path"])
    )
    units = []
    with query_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        units = list(reader)
    source_to_id, source_hash = _mapping(
        (str(row["source_id"]) for row in units), missing_id=False
    )
    family_to_id, family_hash = _mapping(
        (str(row["family_uid"]) for row in units), missing_id=False
    )
    cell_to_id, cell_hash = _mapping(cell_labels, missing_id=True)
    region_to_id, region_hash = _mapping(region_labels, missing_id=True)
    feature_order = sorted(feature_keys)
    feature_to_id = {key: index + 1 for index, key in enumerate(feature_order)}
    feature_hash = canonical_sha256(
        [[key, feature_to_id[key]] for key in feature_order]
    )
    vocabulary = C1Vocabulary(
        feature_to_id=feature_to_id,
        cell_to_id=cell_to_id,
        region_to_id=region_to_id,
        source_to_id=source_to_id,
        family_to_id=family_to_id,
        feature_mapping_sha256=feature_hash,
        cell_mapping_sha256=cell_hash,
        region_mapping_sha256=region_hash,
        source_mapping_sha256=source_hash,
        family_mapping_sha256=family_hash,
    )
    is_v2 = contract["frozen_corpus"]["corpus_uid"] == "public_foundation_corpus_rescue"
    summary: dict[str, object] = {
        "schema_version": "neuronsifter-c1-vocabulary-manifest-v1",
        "corpus_uid": contract["frozen_corpus"]["corpus_uid"],
        "corpus_hash": contract["frozen_corpus"]["identity_hash"],
        "scope": contract["frozen_corpus"]["scope"],
        "derivation_policy": (
            "read_only_derived_artifact_for_frozen_rescue"
            if is_v2
            else "read_only_derived_artifact_outside_frozen_v1"
        ),
        "feature_normalization_version": FEATURE_NORMALIZATION_VERSION,
        "feature_value_normalization_version": FEATURE_VALUE_NORMALIZATION_VERSION,
        "feature_tokenization_policy": (
            "namespace_preserving_gene_token_union_no_cross_namespace_merge"
        ),
        "crosswalk_usage": "provenance_binding_and_identity_audit_only",
        "feature_namespace_aliases": dict(sorted(_NAMESPACE_ALIASES.items())),
        "feature_count": vocabulary.feature_count,
        "feature_vocabulary_size": vocabulary.feature_vocabulary_size,
        "feature_mapping_sha256": feature_hash,
        "feature_namespace_counts": {
            namespace: sum(key.startswith(namespace + "|") for key in feature_order)
            for namespace in sorted({key.split("|", 1)[0] for key in feature_order})
        },
        "crosswalk": {
            "mapping_version": crosswalk["mapping_version"],
            "path": crosswalk["table_path"],
            "sha256": crosswalk["table_sha256"],
            "record_count": crosswalk_record_count,
        },
        "cell_ontology_count": vocabulary.cell_count,
        "cell_ontology_size": vocabulary.cell_ontology_size,
        "cell_context_policy": "source_vocabulary_context_ids_not_graph_ontology_embeddings",
        "cell_mapping_sha256": cell_hash,
        "region_ontology_count": vocabulary.region_count,
        "region_ontology_size": vocabulary.region_ontology_size,
        "region_context_policy": "source_vocabulary_context_ids_not_graph_ontology_embeddings",
        "region_mapping_sha256": region_hash,
        "source_class_count": vocabulary.source_count,
        "source_mapping_sha256": source_hash,
        "family_class_count": vocabulary.family_count,
        "family_mapping_sha256": family_hash,
        "action_feature_schema_version": (
            "neuronsifter-c1-action-vector-v2"
            if descriptor_version == "neuronsifter-c1-physical-exposure-descriptor-v2"
            else ACTION_FEATURE_SCHEMA_VERSION
        ),
        "action_descriptor_version": descriptor_version,
        "action_feature_dim": action_feature_dim,
        "action_feature_names": list(action_feature_names),
        "action_category_values": action_categories,
        "action_descriptor_sha256": action_spec_hash,
        "action_descriptor_record_count": len(action_descriptors),
        "action_identity_scope": "physical_exposure_only_without_agent_target_or_occupancy_identity",
        "action_target_set_available_count": sum(
            int(value.target_set_available) for value in action_descriptors.values()
        ),
        "program_vocab_status": "not_available",
        "observation_binding_hash": binding["binding_hash"],
        "block_feature_hashes": dict(sorted(block_feature_hashes.items())),
    }
    return vocabulary, summary


def build_c1_vocabulary(
    repo_root: Path, config: Mapping[str, object]
) -> tuple[C1Vocabulary, dict[str, object]]:
    """Derive the deterministic vocabulary from a live-attested frozen corpus."""

    repo_root = repo_root.resolve()
    require_live_c1_preflight(config, repo_root)
    binding = _binding_manifest(config, repo_root)
    return _build_c1_vocabulary_attested(repo_root, config, binding)


def build_c1_vocabulary_from_frozen_bindings(
    repo_root: Path, config: Mapping[str, object]
) -> tuple[C1Vocabulary, dict[str, object]]:
    """Derive a vocabulary for preflight artifact construction only.

    This validates the frozen observation-binding and action-subset inputs but
    intentionally does not attest that a unified preflight already exists.
    Runtime loaders must continue to use :func:`build_c1_vocabulary`, which
    requires the live preflight authorization before data access.
    """

    repo_root = repo_root.resolve()
    binding = _binding_manifest(config, repo_root)
    return _build_c1_vocabulary_attested(repo_root, config, binding)


def _load_c1_vocabulary_attested(
    repo_root: Path,
    config: Mapping[str, object],
    binding: Mapping[str, object],
) -> C1Vocabulary:
    vocabulary, summary = _build_c1_vocabulary_attested(repo_root, config, binding)
    input_contract = config["foundation_state_c1"]["encoder"]["input_contract"]
    manifest_path_value = input_contract.get("vocabulary_manifest_path")
    manifest_hash = input_contract.get("vocabulary_manifest_sha256")
    if not manifest_path_value or not manifest_hash:
        raise ValueError(
            "C1 vocabulary manifest path/hash is not frozen in the encoder contract"
        )
    manifest_path = safe_repo_path(repo_root, str(manifest_path_value))
    if sha256_file(manifest_path) != str(manifest_hash):
        raise ValueError("C1 vocabulary manifest hash drift")
    with manifest_path.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored != summary:
        raise ValueError(
            "C1 vocabulary manifest differs from deterministic reconstruction"
        )
    return vocabulary


def load_c1_vocabulary(repo_root: Path, config: Mapping[str, object]) -> C1Vocabulary:
    repo_root = repo_root.resolve()
    require_live_c1_preflight(config, repo_root)
    binding = _binding_manifest(config, repo_root)
    return _load_c1_vocabulary_attested(repo_root, config, binding)


def build_c1_loader_context(
    repo_root: Path, config: Mapping[str, object] | None = None
) -> C1LoaderContext:
    """Create reusable loader bindings after one complete live corpus scan."""

    repo_root = repo_root.resolve()
    resolved_config = (
        dict(config)
        if config is not None
        else load_config(repo_root / "Libs/configs/config_stack.yaml")
    )
    errors = validate_c1_contract(resolved_config, repo_root)
    if errors:
        raise ValueError("C1 contract is invalid: " + str(errors))
    attestation = attest_live_c1_preflight(resolved_config, repo_root)
    binding = _binding_manifest(resolved_config, repo_root)
    vocabulary = _load_c1_vocabulary_attested(repo_root, resolved_config, binding)
    action_records = _qualified_action_records(repo_root, resolved_config)
    input_contract = resolved_config["foundation_state_c1"]["encoder"][
        "input_contract"
    ]
    action_descriptors = _action_feature_spec(
        action_records,
        action_feature_dim=int(input_contract["action_feature_dim"]),
        descriptor_version=str(input_contract["action_descriptor_version"]),
    )[0]
    return C1LoaderContext(
        attestation=attestation,
        binding=binding,
        binding_digest=canonical_sha256(binding),
        vocabulary=vocabulary,
        vocabulary_digest=_vocabulary_digest(vocabulary),
        action_descriptors=action_descriptors,
        action_descriptor_digest=_action_descriptor_digest(action_descriptors),
    )


@dataclass(frozen=True)
class C1Batch:
    feature_ids: torch.Tensor
    feature_values: torch.Tensor
    observed_mask: torch.Tensor
    target_mask: torch.Tensor
    cell_ontology_ids: torch.Tensor
    region_ontology_ids: torch.Tensor
    action_features: torch.Tensor
    action_available: torch.Tensor
    observation_unit_index: torch.Tensor
    biological_unit_weights: torch.Tensor
    source_ids: torch.Tensor
    family_ids: torch.Tensor
    semantic_group_ids: torch.Tensor
    biological_unit_uids: tuple[str, ...]
    canonical_uids: tuple[str, ...]

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "feature_ids": self.feature_ids,
            "feature_values": self.feature_values,
            "observed_mask": self.observed_mask,
            "target_mask": self.target_mask,
            "cell_ontology_ids": self.cell_ontology_ids,
            "region_ontology_ids": self.region_ontology_ids,
            "observation_unit_index": self.observation_unit_index,
            "action_features": self.action_features,
            "action_available": self.action_available,
        }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def c1_batch_hash(batch: C1Batch) -> str:
    """Hash every model-visible tensor and unit identity in one C1 batch."""

    tensor_fields = (
        "feature_ids",
        "feature_values",
        "observed_mask",
        "target_mask",
        "cell_ontology_ids",
        "region_ontology_ids",
        "action_features",
        "action_available",
        "observation_unit_index",
        "biological_unit_weights",
        "source_ids",
        "family_ids",
        "semantic_group_ids",
    )
    return canonical_sha256(
        {
            "tensor_sha256": {
                name: _tensor_sha256(getattr(batch, name)) for name in tensor_fields
            },
            "biological_unit_uids": list(batch.biological_unit_uids),
            "canonical_uids": list(batch.canonical_uids),
        }
    )


def _alignment_postcondition(batch: C1Batch) -> bool:
    views = {
        (int(group), int(source), int(unit))
        for group, source, unit in zip(
            batch.semantic_group_ids.tolist(),
            batch.source_ids.tolist(),
            batch.observation_unit_index.tolist(),
            strict=True,
        )
        if int(group) >= 0
    }
    return any(
        any(
            other_group == group and other_source != source and other_unit != unit
            for other_group, other_source, other_unit in views
        )
        and any(other_group != group for other_group, _, _ in views)
        for group, source, unit in views
    )


def _family_balance_postcondition(batch: C1Batch) -> bool:
    unit_families: dict[int, int] = {}
    for unit_index, family_id in zip(
        batch.observation_unit_index.tolist(), batch.family_ids.tolist(), strict=True
    ):
        existing = unit_families.setdefault(int(unit_index), int(family_id))
        if existing != int(family_id):
            return False
    counts: dict[int, int] = {}
    for family_id in unit_families.values():
        counts[family_id] = counts.get(family_id, 0) + 1
    return bool(counts) and len(set(counts.values())) == 1


def validate_c1_batch_contract(
    batch: C1Batch,
    *,
    active_objectives: Sequence[str],
    batch_size: int,
    token_budget: int,
    max_observations_per_unit: int,
    feature_vocabulary_size: int,
    cell_ontology_size: int,
    region_ontology_size: int,
    source_class_count: int,
    family_class_count: int,
    action_feature_dim: int,
    family_balance_required: bool,
) -> dict[str, object]:
    """Fail closed on every model and sampling invariant before batch release."""

    if batch_size <= 0 or token_budget <= 0 or max_observations_per_unit <= 0:
        raise ValueError("C1 batch validator dimensions must be positive")
    observation_count = (
        int(batch.feature_ids.shape[0]) if batch.feature_ids.ndim == 2 else 0
    )
    vector_shape = (observation_count,)
    tensor_shape_dtype_contract = (
        observation_count > 0
        and batch.feature_ids.shape == (observation_count, token_budget)
        and batch.feature_values.shape == batch.feature_ids.shape
        and batch.observed_mask.shape == batch.feature_ids.shape
        and batch.target_mask.shape == batch.feature_ids.shape
        and batch.action_features.shape == (observation_count, action_feature_dim)
        and batch.cell_ontology_ids.shape == vector_shape
        and batch.region_ontology_ids.shape == vector_shape
        and batch.action_available.shape == vector_shape
        and batch.observation_unit_index.shape == vector_shape
        and batch.source_ids.shape == vector_shape
        and batch.family_ids.shape == vector_shape
        and batch.semantic_group_ids.shape == vector_shape
        and batch.biological_unit_weights.shape == (batch_size,)
        and batch.feature_ids.dtype == torch.long
        and batch.feature_values.dtype == torch.float32
        and batch.observed_mask.dtype == torch.bool
        and batch.target_mask.dtype == torch.bool
        and batch.cell_ontology_ids.dtype == torch.long
        and batch.region_ontology_ids.dtype == torch.long
        and batch.action_features.dtype == torch.float32
        and batch.action_available.dtype == torch.bool
        and batch.observation_unit_index.dtype == torch.long
        and batch.biological_unit_weights.dtype == torch.float32
        and batch.source_ids.dtype == torch.long
        and batch.family_ids.dtype == torch.long
        and batch.semantic_group_ids.dtype == torch.long
    )
    if not tensor_shape_dtype_contract:
        raise ValueError("C1 batch failed tensor shape/dtype contract")

    unit_indices = [int(value) for value in batch.observation_unit_index.tolist()]
    unit_counts = {
        unit_index: unit_indices.count(unit_index) for unit_index in set(unit_indices)
    }
    expected_unit_indices = set(range(batch_size))
    nonpadding = batch.feature_ids.ne(0)
    padding = ~nonpadding
    active = set(active_objectives)
    alignment_required = "source_aware_contrastive_alignment" in active
    action_required = "action_conditioned_perturbation_representation" in active
    finite_numeric_tensors = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            batch.feature_values,
            batch.action_features,
            batch.biological_unit_weights,
        )
    )
    feature_id_range_valid = bool(
        torch.all(batch.feature_ids >= 0)
        and torch.all(batch.feature_ids < feature_vocabulary_size)
    )
    ontology_and_nuisance_id_ranges_valid = bool(
        torch.all(batch.cell_ontology_ids >= 0)
        and torch.all(batch.cell_ontology_ids < cell_ontology_size)
        and torch.all(batch.region_ontology_ids >= 0)
        and torch.all(batch.region_ontology_ids < region_ontology_size)
        and torch.all(batch.source_ids >= 0)
        and torch.all(batch.source_ids < source_class_count)
        and torch.all(batch.family_ids >= 0)
        and torch.all(batch.family_ids < family_class_count)
        and torch.all(batch.semantic_group_ids >= -1)
    )
    mask_partition_valid = bool(
        not torch.any(batch.observed_mask & batch.target_mask)
        and torch.equal(batch.observed_mask | batch.target_mask, nonpadding)
        and torch.all(batch.target_mask.sum(dim=1) >= 1)
    )
    padding_contract_valid = bool(
        not torch.any(batch.observed_mask[padding])
        and not torch.any(batch.target_mask[padding])
        and torch.all(batch.feature_values[padding] == 0.0)
    )
    unavailable_action_zero_vector = bool(
        torch.all(batch.action_features[~batch.action_available] == 0.0)
    )
    checks = {
        "non_empty_observation_batch": observation_count > 0,
        "exact_biological_unit_batch_size": (
            len(batch.biological_unit_uids) == batch_size
            and len(batch.canonical_uids) == batch_size
            and len(set(batch.biological_unit_uids)) == batch_size
        ),
        "observation_unit_index_coverage": set(unit_indices) == expected_unit_indices,
        "within_unit_observation_cap_respected": (
            bool(unit_counts) and max(unit_counts.values()) <= max_observations_per_unit
        ),
        "tensor_shape_dtype_contract": tensor_shape_dtype_contract,
        "finite_numeric_tensors": finite_numeric_tensors,
        "feature_id_range_valid": feature_id_range_valid,
        "ontology_and_nuisance_id_ranges_valid": ontology_and_nuisance_id_ranges_valid,
        "mask_partition_valid": mask_partition_valid,
        "padding_contract_valid": padding_contract_valid,
        "unit_weights_normalized": math.isclose(
            float(batch.biological_unit_weights.sum()),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ),
        "unit_weights_nonnegative": bool(
            torch.all(batch.biological_unit_weights >= 0.0)
        ),
        "unavailable_action_zero_vector": unavailable_action_zero_vector,
        "target_mask_present_per_observation": bool(batch.target_mask.any(dim=1).all()),
        "action_requirement_satisfied": (
            not action_required or bool(batch.action_available.any())
        ),
        "alignment_requirement_satisfied": (
            not alignment_required or _alignment_postcondition(batch)
        ),
        "family_balance_requirement_satisfied": (
            not family_balance_required or _family_balance_postcondition(batch)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("C1 batch failed runtime contract: " + ",".join(failed))
    return {
        "all_passed": True,
        "observation_count": observation_count,
        "biological_unit_count": len(batch.biological_unit_uids),
        **checks,
    }


@dataclass
class _BlockState:
    binding: Mapping[str, object]
    metadata_rows: list[dict[str, object]]
    metadata_by_key: dict[str, list[dict[str, object]]]
    feature_array_to_token: dict[int, int]
    feature_count: int
    values: object


@dataclass(frozen=True)
class _UnitObjectiveCapability:
    semantic_refs: tuple[C1ObjectiveRequirementRef, ...]
    action_refs: tuple[C1ObjectiveRequirementRef, ...]


def _select_observation_rows(
    rows: Sequence[dict[str, object]],
    *,
    unit_key: str,
    cap: int,
    seed: str,
    required_ordinals: Sequence[int] = (),
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError(f"C1 biological unit has no metadata rows: {unit_key}")
    if cap <= 0:
        raise ValueError("C1 max_observations_per_biological_unit must be positive")
    by_ordinal = {int(row["__ordinal"]): row for row in rows}
    if len(by_ordinal) != len(rows):
        raise ValueError(f"C1 metadata ordinals are duplicated: {unit_key}")
    required = {int(value) for value in required_ordinals}
    missing = required - set(by_ordinal)
    if missing:
        raise ValueError(
            f"C1 objective-required observations are absent: {unit_key}:{sorted(missing)}"
        )
    if len(required) > cap:
        raise ValueError(
            "C1 objective-aware observation requirements exceed the per-unit cap: "
            f"{unit_key}:{len(required)}>{cap}"
        )
    if len(rows) <= cap:
        return list(rows)
    ranked = sorted(
        (row for row in rows if int(row["__ordinal"]) not in required),
        key=lambda row: hashlib.sha256(
            f"{seed}|{unit_key}|{row['__ordinal']}".encode("utf-8")
        ).hexdigest(),
    )
    selected = [by_ordinal[ordinal] for ordinal in required]
    selected.extend(ranked[: cap - len(selected)])
    return sorted(selected, key=lambda row: int(row["__ordinal"]))


def _top_tokens(
    pairs: Iterable[tuple[int, float]],
    *,
    token_budget: int,
    feature_array_to_token: Mapping[int, int],
    unit_key: str,
) -> tuple[list[int], list[float]]:
    values_by_token: dict[int, float] = {}
    for array_index, raw_value in pairs:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"C1 non-finite transcriptomic value for {unit_key}")
        token = feature_array_to_token.get(int(array_index))
        if token is None:
            raise ValueError(
                f"C1 feature index is outside the frozen vocabulary: {array_index}"
            )
        if token in values_by_token:
            raise ValueError(
                f"C1 duplicate canonical feature in observation: {unit_key}:{token}"
            )
        if value != 0.0:
            values_by_token[token] = value
    if not values_by_token:
        raise ValueError(
            f"C1 observation has no nonzero finite transcriptomic feature: {unit_key}"
        )
    selected = sorted(
        values_by_token, key=lambda token: (-abs(values_by_token[token]), token)
    )[:token_budget]
    selected.sort()
    return selected, [values_by_token[token] for token in selected]


def _action_vector(
    row: Mapping[str, object],
    descriptors: Mapping[str, _ActionDescriptor],
    *,
    action_feature_dim: int = ACTION_FEATURE_DIM,
) -> tuple[np.ndarray, bool]:
    action_uid = _text(row.get("action_uid"))
    if not action_uid:
        return np.zeros(action_feature_dim, dtype=np.float32), False
    descriptor = descriptors.get(action_uid)
    if descriptor is None:
        raise ValueError(
            f"C1 observation action is not in the qualified action subset: {action_uid}"
        )
    vector = np.asarray(descriptor.vector, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError(f"C1 action feature vector is non-finite: {action_uid}")
    # Source-native action identity is deliberately excluded from semantic
    # alignment until an exact cross-source action mapping is qualified.
    return vector, True


class C1TranscriptomicsLoader:
    """Deterministic unit sampler and adapter-backed C1 observation loader."""

    def __init__(
        self,
        repo_root: Path,
        config: Mapping[str, object] | None = None,
        *,
        role_binding_path: Path,
        expected_role: str = "train",
        batch_size: int = 6,
        profile: str = "Student",
        seed: int = 3407,
        sampling_policy: str | None = None,
        max_observations_per_unit: int | None = None,
        objective_set: str = "masked_only",
        source_id: str | None = None,
        family_uid: str | None = None,
        run_scope_attestation: (
            C1RunScopeAttestation | C1ReplayScopeAttestation | None
        ) = None,
        run_scope_mode: Literal["training", "preflight_replay"] = "training",
        loader_context: C1LoaderContext | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = (
            dict(config)
            if config is not None
            else load_config(self.repo_root / "Libs/configs/config_stack.yaml")
        )
        contract = self.config["foundation_state_c1"]
        errors = validate_c1_contract(self.config, self.repo_root)
        if errors:
            raise ValueError("C1 contract is invalid: " + str(errors))
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("C1 loader batch_size must be positive")
        if profile not in contract["encoder"]["profiles"]:
            raise ValueError(f"Unknown C1 encoder profile: {profile}")
        if expected_role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown C1 expected split role: {expected_role}")
        if run_scope_mode not in {"training", "preflight_replay"}:
            raise ValueError(f"Unknown C1 run scope mode: {run_scope_mode}")
        self.run_scope_mode = run_scope_mode
        objective_sets = contract["encoder"].get("objective_sets")
        if (
            not isinstance(objective_sets, Mapping)
            or objective_set not in objective_sets
        ):
            raise ValueError(f"Unknown C1 objective set: {objective_set}")
        configured_objectives = objective_sets[objective_set]
        if not isinstance(configured_objectives, Sequence) or isinstance(
            configured_objectives, (str, bytes)
        ):
            raise ValueError(f"C1 objective set is malformed: {objective_set}")
        self.objective_set = objective_set
        self.active_objectives = tuple(str(value) for value in configured_objectives)
        self._alignment_required = (
            "source_aware_contrastive_alignment" in self.active_objectives
        )
        self._action_required = (
            "action_conditioned_perturbation_representation" in self.active_objectives
        )
        if source_id is not None and (
            type(source_id) is not str or not source_id.strip()
        ):
            raise TypeError("C1 loader source_id must be a non-empty string or null")
        self.source_id = source_id
        if family_uid is not None and (
            type(family_uid) is not str or not family_uid.strip()
        ):
            raise TypeError("C1 loader family_uid must be a non-empty string or null")
        self.family_uid = family_uid
        role_binding_path = (
            role_binding_path.resolve()
            if role_binding_path.is_absolute()
            else safe_repo_path(self.repo_root, role_binding_path.as_posix())
        )
        self._role_binding_path = role_binding_path
        loader_contract = contract["loader"]
        sampler_contract = loader_contract.get("sampler_contract")
        if (
            not isinstance(sampler_contract, Mapping)
            or sampler_contract.get("version") != OBJECTIVE_AWARE_SAMPLER_VERSION
        ):
            raise ValueError("C1 objective-aware sampler contract version drift")
        requested_sampling_policy = sampling_policy or str(
            loader_contract["sampling_policy"]
        )
        if requested_sampling_policy == ROW_BALANCED_POLICY and (
            self._alignment_required or self._action_required
        ):
            self.sampling_policy = OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY
        else:
            self.sampling_policy = requested_sampling_policy
        if self.sampling_policy not in {
            FAMILY_AND_UNIT_BALANCED_POLICY,
            FAMILY_BALANCED_POLICY,
            BIOLOGICAL_UNIT_BALANCED_POLICY,
            ROW_BALANCED_POLICY,
            OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY,
        }:
            raise ValueError(f"Unsupported C1 sampling policy: {self.sampling_policy}")
        self.batch_size = int(batch_size)
        self.weighting_policy = (
            "biological_unit_balanced"
            if self.sampling_policy in {
                FAMILY_AND_UNIT_BALANCED_POLICY,
                FAMILY_BALANCED_POLICY,
                BIOLOGICAL_UNIT_BALANCED_POLICY,
            }
            else (
                "objective_constrained_row_proportional_weighting"
                if self.sampling_policy == OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY
                else "observation_row_balanced"
            )
        )
        self.profile = profile
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("C1 loader seed must be non-negative")
        self.token_budget = int(
            contract["encoder"]["profiles"][profile]["feature_token_budget"]
        )
        self.action_feature_dim = int(
            contract["encoder"]["input_contract"]["action_feature_dim"]
        )
        if self.action_feature_dim <= 0:
            raise ValueError("C1 action feature dimension must be positive")
        configured_cap = loader_contract.get("max_observations_per_biological_unit", 8)
        if (
            max_observations_per_unit is not None
            and type(max_observations_per_unit) is not int
        ):
            raise TypeError(
                "C1 max_observations_per_biological_unit must be an integer"
            )
        self.max_observations_per_unit = int(
            configured_cap
            if max_observations_per_unit is None
            else max_observations_per_unit
        )
        if self.max_observations_per_unit <= 0:
            raise ValueError("C1 max_observations_per_biological_unit must be positive")
        if loader_context is None:
            loader_context = build_c1_loader_context(self.repo_root, self.config)
        else:
            if type(loader_context) is not C1LoaderContext:
                raise TypeError(
                    "C1 loader_context must be a live in-process capability"
                )
            validate_c1_preflight_attestation(
                self.config, self.repo_root, loader_context.attestation
            )
        binding = _binding_manifest(self.config, self.repo_root)
        if canonical_sha256(binding) != loader_context.binding_digest:
            raise ValueError("C1 loader context observation binding drift")
        if (
            _vocabulary_digest(loader_context.vocabulary)
            != loader_context.vocabulary_digest
        ):
            raise ValueError("C1 loader context vocabulary was mutated")
        if (
            _action_descriptor_digest(loader_context.action_descriptors)
            != loader_context.action_descriptor_digest
        ):
            raise ValueError("C1 loader context action descriptors were mutated")
        self.loader_context = loader_context
        frozen = contract["frozen_corpus"]
        query_path = safe_repo_path(
            self.repo_root, str(frozen["required_query_domain"]["path"])
        )
        (
            self.split_uid,
            self.split_role,
            self.split_manifest_hash,
            self.role_binding_sha256,
        ) = self._validate_split_binding(role_binding_path, contract)
        if self.split_role != expected_role:
            raise ValueError(
                "C1 loader role mismatch: "
                f"expected {expected_role}, received {self.split_role}"
            )
        self._validate_run_scope_attestation(
            run_scope_attestation,
            role_binding_path=role_binding_path,
            contract=contract,
        )
        self._run_scope_attestation = run_scope_attestation
        self.vocabulary = loader_context.vocabulary
        self.units = load_c1_split_units(query_path, role_binding_path)
        if self.source_id is not None:
            self.units = [
                unit for unit in self.units if unit.source_id == self.source_id
            ]
        if self.family_uid is not None:
            self.units = [
                unit for unit in self.units if unit.family_uid == self.family_uid
            ]
        if not self.units:
            scope = self.source_id or self.family_uid or "all_sources_all_families"
            raise ValueError(f"C1 loader split contains no units for scope: {scope}")
        available_family_count = len({unit.family_uid for unit in self.units})
        if self.sampling_policy in {
            FAMILY_AND_UNIT_BALANCED_POLICY,
            FAMILY_BALANCED_POLICY,
        } and (
            self.batch_size < available_family_count
            or self.batch_size % available_family_count != 0
        ):
            raise ValueError(
                "Family-balanced C1 batch size must be divisible by the available family count"
            )
        self._binding_rows = self._load_binding_rows(binding)
        required_block_uids = {
            str(self._binding_rows[unit.canonical_uid]["block_uid"])
            for unit in self.units
        }
        self._blocks = self._load_blocks(binding, required_block_uids)
        self._qualified_action_descriptors = loader_context.action_descriptors
        self._unit_groups = self._group_units()
        self._objective_capabilities = self._build_objective_capabilities()

    def _validate_split_binding(
        self, role_binding_path: Path, contract: Mapping[str, object]
    ) -> tuple[str, str, str, str]:
        """Require the supplied role table to be one of the frozen split files."""

        frozen = contract["frozen_corpus"]
        evidence_path = safe_repo_path(
            self.repo_root, str(frozen["evidence_bundle_path"])
        )
        with evidence_path.open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        split_ref = bundle["evidence"]["frozen_splits"]
        split_path = safe_repo_path(self.repo_root, str(split_ref["path"]))
        if sha256_file(split_path) != str(split_ref["sha256"]):
            raise ValueError("C1 frozen split artifact hash drift")
        with split_path.open("r", encoding="utf-8") as handle:
            split_artifact = json.load(handle)
        supplied = role_binding_path.resolve()
        matches: list[tuple[str, str, str, Mapping[str, object]]] = []
        for manifest in split_artifact.get("manifests", []):
            split_uid = str(manifest.get("split_uid") or "")
            for role in ("train", "validation", "test"):
                binding = manifest.get("role_bindings", {}).get(role, {})
                if not isinstance(binding, Mapping):
                    continue
                path = safe_repo_path(self.repo_root, str(binding.get("path") or ""))
                if path == supplied:
                    matches.append(
                        (
                            split_uid,
                            str(manifest.get("manifest_hash") or ""),
                            role,
                            binding,
                        )
                    )
        if len(matches) != 1:
            raise ValueError("C1 role binding path is outside the frozen split domain")
        split_uid, manifest_hash, role, binding = matches[0]
        if sha256_file(supplied) != str(binding["sha256"]):
            raise ValueError("C1 role binding file hash drift")
        expected_hash = frozen["required_split_manifests"].get(split_uid)
        if expected_hash is None or manifest_hash != expected_hash:
            raise ValueError(f"C1 split manifest identity is not frozen: {split_uid}")
        return split_uid, role, manifest_hash, str(binding["sha256"])

    def _validate_run_scope_attestation(
        self,
        attestation: C1RunScopeAttestation | C1ReplayScopeAttestation | None,
        *,
        role_binding_path: Path,
        contract: Mapping[str, object],
    ) -> None:
        if self.split_role != "train":
            if self.run_scope_mode != "training":
                raise ValueError(
                    "C1 preflight replay mode is valid only for the frozen train role"
                )
            if attestation is not None:
                raise ValueError(
                    "C1 validation/test loaders must not carry a run scope attestation"
                )
            return

        if self.run_scope_mode == "training":
            if type(attestation) is not C1RunScopeAttestation:
                raise TypeError(
                    "C1 train loader requires an exact live in-process "
                    "C1RunScopeAttestation capability"
                )
            registered = _RUN_SCOPE_CAPABILITIES.get(attestation._capability_id)
            if (
                registered is None
                or registered.attestation is not attestation
                or registered.run_preflight_digest != attestation.run_preflight_digest
                or registered.run_scope_binding_digest
                != attestation.run_scope_binding_digest
            ):
                raise ValueError("C1 run scope capability is unknown or forged")
            if (
                canonical_sha256(attestation.run_preflight_report)
                != attestation.run_preflight_digest
                or canonical_sha256(attestation.run_scope_binding)
                != attestation.run_scope_binding_digest
            ):
                raise ValueError("C1 run scope attestation was mutated")
            report_binding = attestation.run_preflight_report.get("run_scope_binding")
            if not isinstance(report_binding, Mapping) or canonical_sha256(
                report_binding
            ) != canonical_sha256(attestation.run_scope_binding):
                raise ValueError("C1 run scope attestation report/binding drift")
            runtime_replay = attestation.run_preflight_report.get("runtime_replay")
            if not isinstance(runtime_replay, Mapping):
                raise TypeError("C1 run scope attestation has no runtime replay")
            _validate_runtime_replay_evidence(runtime_replay)
            if (
                attestation.run_preflight_report.get("training_authorization_status")
                != "runtime_replay_verified"
            ):
                raise ValueError("C1 run scope capability is not training-authorized")
        else:
            if type(attestation) is not C1ReplayScopeAttestation:
                raise TypeError(
                    "C1 preflight replay requires an exact live in-process "
                    "C1ReplayScopeAttestation capability"
                )
            replay_record = _REPLAY_CAPABILITIES.get(attestation._capability_id)
            if replay_record is None:
                raise ValueError("C1 replay scope capability is unknown or forged")
            if (
                replay_record.attestation is not attestation
                or canonical_sha256(attestation.run_scope_binding)
                != attestation.run_scope_binding_digest
                or replay_record.run_scope_binding_digest
                != attestation.run_scope_binding_digest
            ):
                raise ValueError("C1 replay scope attestation was mutated")
        binding = attestation.run_scope_binding
        _validate_run_scope_binding_shape(binding)
        role_relative_path = (
            role_binding_path.resolve().relative_to(self.repo_root).as_posix()
        )
        frozen = contract["frozen_corpus"]
        input_contract = contract["encoder"]["input_contract"]
        vocabulary_path = safe_repo_path(
            self.repo_root, str(input_contract["vocabulary_manifest_path"])
        )
        with vocabulary_path.open("r", encoding="utf-8") as handle:
            vocabulary_manifest = json.load(handle)
        expected = {
            "corpus_uid": frozen["corpus_uid"],
            "corpus_hash": frozen["identity_hash"],
            "observation_binding_hash": frozen["required_observation_binding"][
                "binding_hash"
            ],
            "split_uid": self.split_uid,
            "split_manifest_hash": self.split_manifest_hash,
            "split_role": self.split_role,
            "role_binding_path": role_relative_path,
            "role_binding_sha256": self.role_binding_sha256,
            "source_scope": (
                "single_source" if self.source_id is not None else "all_sources"
            ),
            "source_id": self.source_id,
            "family_scope": (
                "single_family" if self.family_uid is not None else "all_families"
            ),
            "family_uid": self.family_uid,
            "model_profile": self.profile,
            "objective_set": self.objective_set,
            "active_objectives": list(self.active_objectives),
            "sampling_policy": self.sampling_policy,
            "weighting_policy": self.weighting_policy,
            "sampler_contract_version": OBJECTIVE_AWARE_SAMPLER_VERSION,
            "batch_size": self.batch_size,
            "max_observations_per_biological_unit": (self.max_observations_per_unit),
            "seed": self.seed,
            "contract_digest": canonical_sha256(contract),
            "live_preflight_digest": self.loader_context.attestation.preflight_digest,
            "loader_implementation_version": C1_LOADER_IMPLEMENTATION_VERSION,
            "vocabulary_manifest_sha256": input_contract["vocabulary_manifest_sha256"],
            "action_descriptor_sha256": vocabulary_manifest["action_descriptor_sha256"],
            "qualified_action_subset_hash": frozen["qualified_action_subset_hash"],
            "objective_capability_manifest_sha256": contract["artifacts"][
                "objective_capability_manifest_sha256"
            ],
            "frozen_block_binding_hash": canonical_sha256(
                dict(frozen["required_blocks"])
            ),
        }
        mismatches = [
            field
            for field, value in expected.items()
            if type(binding.get(field)) is not type(value)
            or binding.get(field) != value
        ]
        if mismatches:
            raise ValueError(
                "C1 run scope attestation does not match loader runtime: "
                f"{sorted(mismatches)}"
            )
        baselines = contract["baselines"]
        baseline_id = str(binding["baseline_id"])
        runtime_profiles = baselines.get("runtime_profiles")
        if (
            baseline_id not in baselines.get("required", [])
            or not isinstance(runtime_profiles, Mapping)
            or baseline_id not in runtime_profiles
        ):
            raise ValueError(
                f"C1 run scope attests an unknown required baseline: {baseline_id}"
            )
        baseline_profile = runtime_profiles[baseline_id]
        if not isinstance(baseline_profile, Mapping):
            raise TypeError(f"C1 baseline runtime profile is malformed: {baseline_id}")
        pretraining_scope = baseline_profile.get("pretraining_scope")
        if pretraining_scope == "one_train_source_per_run":
            baseline_source_scope = "single_source"
            baseline_family_scope = "all_families"
        elif pretraining_scope == "one_train_family_per_run":
            baseline_source_scope = "all_sources"
            baseline_family_scope = "single_family"
        elif pretraining_scope == "all_train_sources":
            baseline_source_scope = "all_sources"
            baseline_family_scope = "all_families"
        else:
            raise ValueError(
                f"C1 baseline has no loader-backed pretraining scope: {baseline_id}"
            )
        baseline_expected = {
            "source_scope": baseline_source_scope,
            "family_scope": baseline_family_scope,
            "objective_set": baseline_profile.get("objective_set"),
            "sampling_policy": baseline_profile.get("sampling_policy"),
            "weighting_policy": baseline_profile.get("weighting_policy"),
        }
        baseline_mismatches = [
            field
            for field, value in baseline_expected.items()
            if type(binding.get(field)) is not type(value)
            or binding.get(field) != value
        ]
        if baseline_mismatches:
            raise ValueError(
                "C1 run scope attestation does not match the frozen baseline profile: "
                f"{baseline_id}:{sorted(baseline_mismatches)}"
            )

    def _load_binding_rows(
        self, binding: Mapping[str, object]
    ) -> dict[str, Mapping[str, object]]:
        path = safe_repo_path(self.repo_root, str(binding["row_binding_table"]["path"]))
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        by_uid: dict[str, Mapping[str, object]] = {}
        for row in rows:
            uid = _text(row.get("canonical_uid"))
            if not uid or uid in by_uid:
                raise ValueError(
                    f"C1 row binding table has duplicate/empty canonical UID: {uid}"
                )
            by_uid[uid] = row
        return by_uid

    def _load_blocks(
        self,
        binding: Mapping[str, object],
        required_block_uids: set[str],
    ) -> dict[str, _BlockState]:
        states: dict[str, _BlockState] = {}
        for block in binding["blocks"]:
            block_uid = str(block["block_uid"])
            if block_uid not in required_block_uids:
                continue
            metadata_path = safe_repo_path(
                self.repo_root, str(block["metadata"]["path"])
            )
            key_column = str(block["metadata"]["key_column"])
            rows = _metadata_projection(metadata_path, key_column)
            by_key: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                key = _text(row.get(key_column))
                if not key:
                    raise ValueError(
                        f"C1 metadata has an empty selector key: {metadata_path}"
                    )
                by_key.setdefault(key, []).append(row)
            feature_path = safe_repo_path(
                self.repo_root, str(block["features"]["path"])
            )
            feature_records = _feature_records(feature_path)
            array_to_token: dict[int, int] = {}
            for array_index, key in feature_records:
                token = self.vocabulary.feature_to_id.get(key)
                if token is None:
                    raise ValueError(
                        f"C1 feature key missing from frozen vocabulary: {key}"
                    )
                if array_index in array_to_token:
                    raise ValueError(
                        f"C1 feature array index is duplicated: {block['block_uid']}:{array_index}"
                    )
                array_to_token[array_index] = token
            value_path = safe_repo_path(self.repo_root, str(block["values"]["path"]))
            values: object
            if str(block["adapter"]) == "zarr_dense_indexed":
                group = zarr.open(value_path, mode="r")
                if "values" not in group:
                    raise ValueError(f"C1 dense values array is missing: {value_path}")
                values = group["values"]
            elif str(block["adapter"]) == "zarr_csr_ordinal":
                group = zarr.open(value_path, mode="r")
                if not {"data", "indices", "indptr"} <= set(group.keys()):
                    raise ValueError(
                        f"C1 CSR values arrays are incomplete: {value_path}"
                    )
                values = (group["data"], group["indices"], group["indptr"])
            elif str(block["adapter"]) == "parquet_long_keyed":
                value_key = str(block["values"]["unit_key_column"])
                feature_key = str(block["values"]["feature_key_column"])
                table = pq.read_table(
                    value_path, columns=[value_key, feature_key, "value"]
                )
                data = table.to_pydict()
                grouped: dict[str, list[tuple[int, float]]] = {}
                for unit_key, feature_value, expression in zip(
                    data[value_key], data[feature_key], data["value"], strict=True
                ):
                    feature_token = self.vocabulary.feature_to_id.get(
                        _canonical_feature_key(
                            "entrez_gene_id"
                            if feature_key == "canonical_gene_id"
                            else "",
                            feature_value,
                        )
                    )
                    if feature_token is None:
                        raise ValueError(
                            f"C1 parquet feature is absent from frozen vocabulary: {feature_value}"
                        )
                    grouped.setdefault(_text(unit_key), []).append(
                        (feature_token, float(expression))
                    )
                values = grouped
            else:
                raise ValueError(f"Unsupported C1 adapter: {block['adapter']}")
            states[block_uid] = _BlockState(
                binding=block,
                metadata_rows=rows,
                metadata_by_key=by_key,
                feature_array_to_token=array_to_token,
                feature_count=int(block["matrix_shape"][1]),
                values=values,
            )
        missing = required_block_uids - set(states)
        if missing:
            raise ValueError(
                f"C1 split references unknown frozen blocks: {sorted(missing)}"
            )
        return states

    def _group_units(self) -> dict[str, list[C1Unit]]:
        groups: dict[str, list[C1Unit]] = {}
        families: dict[str, str] = {}
        for unit in self.units:
            family = families.setdefault(unit.biological_unit_uid, unit.family_uid)
            if family != unit.family_uid:
                raise ValueError(
                    "C1 biological unit occurs in multiple semantic families"
                )
            groups.setdefault(unit.biological_unit_uid, []).append(unit)
        return groups

    def _build_objective_capabilities(
        self,
    ) -> dict[str, _UnitObjectiveCapability]:
        if not (self._alignment_required or self._action_required):
            return {}
        capabilities: dict[str, _UnitObjectiveCapability] = {}
        for biological_uid, group in self._unit_groups.items():
            semantic_refs: dict[tuple[str, str], C1ObjectiveRequirementRef] = {}
            action_refs: list[C1ObjectiveRequirementRef] = []
            for unit in group:
                binding_row = self._binding_rows.get(unit.canonical_uid)
                if binding_row is None:
                    raise ValueError(
                        "C1 objective sampler unit is absent from row binding: "
                        f"{unit.canonical_uid}"
                    )
                state = self._blocks[str(binding_row["block_uid"])]
                selector_key = str(binding_row["metadata_key_value"])
                rows = state.metadata_by_key.get(selector_key, [])
                if not rows:
                    raise ValueError(
                        "C1 objective sampler unit has no metadata observations: "
                        f"{unit.canonical_uid}"
                    )
                for row in rows:
                    ordinal = int(row["__ordinal"])
                    semantic_key = qualified_semantic_alignment_key(row)
                    if semantic_key:
                        ref = C1ObjectiveRequirementRef(
                            canonical_uid=unit.canonical_uid,
                            biological_unit_uid=biological_uid,
                            family_uid=unit.family_uid,
                            source_id=unit.source_id,
                            ordinal=ordinal,
                            semantic_key=semantic_key,
                        )
                        view_key = (semantic_key, unit.source_id)
                        current = semantic_refs.get(view_key)
                        if current is None or ref.locator < current.locator:
                            semantic_refs[view_key] = ref
                    action_uid = _text(row.get("action_uid"))
                    if action_uid:
                        if action_uid not in self._qualified_action_descriptors:
                            raise ValueError(
                                "C1 objective sampler found an action outside the "
                                f"qualified subset: {action_uid}"
                            )
                        action_refs.append(
                            C1ObjectiveRequirementRef(
                                canonical_uid=unit.canonical_uid,
                                biological_unit_uid=biological_uid,
                                family_uid=unit.family_uid,
                                source_id=unit.source_id,
                                ordinal=ordinal,
                                action_available=True,
                            )
                        )
            capabilities[biological_uid] = _UnitObjectiveCapability(
                semantic_refs=tuple(
                    sorted(
                        semantic_refs.values(),
                        key=lambda ref: (
                            ref.semantic_key,
                            ref.source_id,
                            ref.canonical_uid,
                            ref.ordinal,
                        ),
                    )
                ),
                action_refs=tuple(
                    sorted(
                        action_refs,
                        key=lambda ref: (
                            ref.biological_unit_uid,
                            ref.canonical_uid,
                            ref.ordinal,
                        ),
                    )
                ),
            )
        return capabilities

    def _family_quotas(self) -> dict[str, int]:
        family_count = len(
            {group[0].family_uid for group in self._unit_groups.values()}
        )
        if self.sampling_policy not in {
            FAMILY_AND_UNIT_BALANCED_POLICY,
            FAMILY_BALANCED_POLICY,
        }:
            return {}
        base, remainder = divmod(self.batch_size, family_count)
        return {
            family: base + (1 if index < remainder else 0)
            for index, family in enumerate(
                sorted({group[0].family_uid for group in self._unit_groups.values()})
            )
        }

    def _objective_rank(
        self,
        purpose: str,
        value: object,
        *,
        epoch: int,
        batch_index: int,
    ) -> str:
        return hashlib.sha256(
            (
                f"{self.seed}|{epoch}|{batch_index}|{self.objective_set}|"
                f"{purpose}|{value}"
            ).encode("utf-8")
        ).hexdigest()

    def _objective_observation_requirements(
        self, *, epoch: int, batch_index: int
    ) -> list[C1ObjectiveRequirementRef]:
        if not (self._alignment_required or self._action_required):
            return []
        family_quotas = self._family_quotas()
        semantic_refs = [
            ref
            for capability in self._objective_capabilities.values()
            for ref in capability.semantic_refs
        ]
        action_refs = [
            ref
            for capability in self._objective_capabilities.values()
            for ref in capability.action_refs
        ]
        requirements = plan_objective_requirements(
            semantic_refs,
            action_refs,
            alignment_required=self._alignment_required,
            action_required=self._action_required,
            batch_size=self.batch_size,
            max_observations_per_unit=self.max_observations_per_unit,
            family_quotas=family_quotas,
            ranker=lambda purpose, value: self._objective_rank(
                purpose,
                value,
                epoch=epoch,
                batch_index=batch_index,
            ),
        )
        if requirements is not None:
            return requirements
        if not self._alignment_required:
            raise ValueError(
                f"C1 objective-aware sampler is not_estimable for objective_set "
                f"{self.objective_set!r}: no qualified action observation fits the "
                "batch sampling and observation-cap constraints"
            )
        joint = " and a qualified action row" if self._action_required else ""
        raise ValueError(
            f"C1 objective-aware sampler is not_estimable for objective_set "
            f"{self.objective_set!r}: no batch satisfies a cross-source same-key "
            f"positive, a distinct-key negative{joint}, and the sampling/observation-cap "
            "constraints"
        )

    def _sample_unit_groups(
        self,
        *,
        epoch: int,
        batch_index: int,
        required_group_uids: set[str] | None = None,
    ) -> list[tuple[str, list[C1Unit]]]:
        groups = self._unit_groups
        required = set(required_group_uids or ())
        unknown_required = required - set(groups)
        if unknown_required:
            raise ValueError(
                "C1 objective sampler requires units outside the split: "
                f"{sorted(unknown_required)}"
            )
        if self.sampling_policy in {
            FAMILY_AND_UNIT_BALANCED_POLICY,
            FAMILY_BALANCED_POLICY,
        }:
            family_groups: dict[str, list[tuple[str, list[C1Unit]]]] = {}
            for key, value in groups.items():
                family_groups.setdefault(value[0].family_uid, []).append((key, value))
            families = sorted(family_groups)
            if self.batch_size < len(families):
                raise ValueError(
                    "Family-balanced C1 batch cannot cover available families"
                )
            base, remainder = divmod(self.batch_size, len(families))
            selected: list[tuple[str, list[C1Unit]]] = []
            for family_index, family in enumerate(families):
                count = base + (1 if family_index < remainder else 0)
                candidates = sorted(
                    family_groups[family],
                    key=lambda item: hashlib.sha256(
                        f"{self.seed}|{epoch}|{batch_index}|{family}|{item[0]}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                )
                if count > len(candidates):
                    raise ValueError(
                        f"Family-balanced C1 batch requests {count} units from {family}, "
                        f"but only {len(candidates)} are available"
                    )
                required_candidates = [
                    item for item in candidates if item[0] in required
                ]
                if len(required_candidates) > count:
                    raise ValueError(
                        "C1 objective requirements exceed a family-balanced quota: "
                        f"{family}"
                    )
                remaining = [item for item in candidates if item[0] not in required]
                family_selected = list(required_candidates)
                if self.sampling_policy == FAMILY_AND_UNIT_BALANCED_POLICY:
                    family_selected.extend(remaining[: count - len(family_selected)])
                else:
                    rng = random.Random(
                        f"{self.seed}|{epoch}|{batch_index}|{family}|row_weighted"
                    )
                    while len(family_selected) < count:
                        weights = [
                            sum(
                                int(self._binding_rows[u.canonical_uid]["observation_count"])
                                for u in value
                            )
                            for _, value in remaining
                        ]
                        threshold = rng.random() * sum(weights)
                        cursor = 0.0
                        selected_index = len(remaining) - 1
                        for index, weight in enumerate(weights):
                            cursor += weight
                            if threshold <= cursor:
                                selected_index = index
                                break
                        family_selected.append(remaining.pop(selected_index))
                selected.extend(family_selected)
            return selected

        candidates = list(groups.items())
        if self.batch_size > len(candidates):
            raise ValueError(
                "Row-balanced C1 sampler cannot draw more unique units than the split contains"
            )
        if self.sampling_policy == BIOLOGICAL_UNIT_BALANCED_POLICY:
            ranked = sorted(
                candidates,
                key=lambda item: hashlib.sha256(
                    f"{self.seed}|{epoch}|{batch_index}|unit|{item[0]}".encode("utf-8")
                ).hexdigest(),
            )
            selected = [item for item in ranked if item[0] in required]
            selected.extend(
                item for item in ranked if item[0] not in required
            )
            return selected[: self.batch_size]
        rng = random.Random(f"{self.seed}|{epoch}|{batch_index}")
        selected = [item for item in candidates if item[0] in required]
        remaining = [item for item in candidates if item[0] not in required]
        for _ in range(self.batch_size - len(selected)):
            weights = [
                sum(
                    int(self._binding_rows[u.canonical_uid]["observation_count"])
                    for u in value
                )
                for _, value in remaining
            ]
            total = sum(weights)
            if total <= 0:
                raise ValueError(
                    "C1 row-balanced sampler encountered non-positive observation counts"
                )
            threshold = rng.random() * total
            cursor = 0.0
            selected_index = len(remaining) - 1
            for index, weight in enumerate(weights):
                cursor += weight
                if threshold <= cursor:
                    selected_index = index
                    break
            selected.append(remaining.pop(selected_index))
        return selected

    def _observation_pairs(
        self, state: _BlockState, row: Mapping[str, object], unit_key: str
    ) -> tuple[list[int], list[float]]:
        adapter = str(state.binding["adapter"])
        if adapter == "parquet_long_keyed":
            raw_pairs = state.values.get(unit_key)
            if not raw_pairs:
                raise ValueError(f"C1 parquet observation is missing: {unit_key}")
            # parquet values already carry canonical feature IDs, so the map is token keyed.
            values_by_token: dict[int, float] = {}
            for token, value in raw_pairs:
                if token in values_by_token:
                    raise ValueError(
                        f"C1 duplicate parquet feature token: {unit_key}:{token}"
                    )
                if not math.isfinite(value):
                    raise ValueError(f"C1 parquet value is non-finite: {unit_key}")
                if value != 0.0:
                    values_by_token[token] = value
            if not values_by_token:
                raise ValueError(
                    f"C1 parquet observation has no nonzero value: {unit_key}"
                )
            selected = sorted(
                values_by_token, key=lambda token: (-abs(values_by_token[token]), token)
            )[: self.token_budget]
            selected.sort()
            return selected, [values_by_token[token] for token in selected]
        ordinal = int(row["__ordinal"])
        if adapter == "zarr_dense_indexed":
            value_row = int(row.get("value_row_index", ordinal))
            matrix = state.values
            if value_row < 0 or value_row >= matrix.shape[0]:
                raise ValueError(
                    f"C1 dense value row is outside the canonical matrix: {unit_key}:{value_row}"
                )
            dense = np.asarray(matrix[value_row, :], dtype=np.float32)
            pairs = (
                (index, float(value))
                for index, value in enumerate(dense)
                if value != 0.0
            )
            return _top_tokens(
                pairs,
                token_budget=self.token_budget,
                feature_array_to_token=state.feature_array_to_token,
                unit_key=unit_key,
            )
        data, indices, indptr = state.values
        if ordinal < 0 or ordinal + 1 >= indptr.shape[0]:
            raise ValueError(
                f"C1 CSR metadata ordinal is outside the canonical matrix: {unit_key}:{ordinal}"
            )
        start = int(indptr[ordinal])
        end = int(indptr[ordinal + 1])
        row_indices = np.asarray(indices[start:end], dtype=np.int64)
        row_values = np.asarray(data[start:end], dtype=np.float32)
        return _top_tokens(
            zip(row_indices.tolist(), row_values.tolist(), strict=True),
            token_budget=self.token_budget,
            feature_array_to_token=state.feature_array_to_token,
            unit_key=unit_key,
        )

    def _load_observation(
        self,
        unit: C1Unit,
        binding_row: Mapping[str, object],
        row: Mapping[str, object],
        state: _BlockState,
        unit_key: str,
        *,
        unit_index: int,
        mask_context: str,
    ) -> dict[str, object]:
        tokens, values = self._observation_pairs(state, row, unit_key)
        if len(tokens) < 2:
            raise ValueError(
                f"C1 masked observation has fewer than two tokens: {unit.canonical_uid}"
            )
        mask_index = int(
            hashlib.sha256(
                f"c1-mask-v2|{self.seed}|{unit.canonical_uid}|{mask_context}".encode(
                    "utf-8"
                )
            ).hexdigest(),
            16,
        ) % len(tokens)
        values = _standardize_selected_feature_values(values, excluded_index=mask_index)
        width = self.token_budget
        feature_ids = np.zeros(width, dtype=np.int64)
        feature_values = np.zeros(width, dtype=np.float32)
        observed_mask = np.zeros(width, dtype=bool)
        target_mask = np.zeros(width, dtype=bool)
        feature_ids[: len(tokens)] = np.asarray(tokens, dtype=np.int64)
        feature_values[: len(values)] = np.asarray(values, dtype=np.float32)
        observed_mask[: len(tokens)] = True
        if not tokens:
            raise ValueError(f"C1 observation has no tokens: {unit.canonical_uid}")
        observed_mask[mask_index] = False
        target_mask[mask_index] = True
        cell_label = _first_label(
            row,
            ("cell_type_id", "cell_type", "cell_type_label", "cell_or_tissue_context"),
        )
        region_label = _first_label(
            row,
            (
                "brain_region_ontology_id",
                "brain_region_id",
                "brain_region_canonical",
                "brain_region_label",
                "brain_region_original",
            ),
        )
        action_features, action_available = _action_vector(
            row,
            self._qualified_action_descriptors,
            action_feature_dim=self.action_feature_dim,
        )
        semantic_group = qualified_semantic_alignment_key(row)
        semantic_group_id = (
            int(hashlib.sha256(semantic_group.encode("utf-8")).hexdigest()[:12], 16)
            % (2**31 - 1)
            if semantic_group
            else -1
        )
        return {
            "feature_ids": feature_ids,
            "feature_values": feature_values,
            "observed_mask": observed_mask,
            "target_mask": target_mask,
            "cell_ontology_id": self.vocabulary.cell_to_id.get(cell_label, 0),
            "region_ontology_id": self.vocabulary.region_to_id.get(region_label, 0),
            "action_features": action_features,
            "action_available": action_available,
            "observation_unit_index": unit_index,
            "source_id": self.vocabulary.source_to_id[unit.source_id],
            "family_id": self.vocabulary.family_to_id[unit.family_uid],
            "semantic_group": semantic_group,
            "semantic_group_id": semantic_group_id,
        }

    def _validate_objective_observations(
        self, observations: Sequence[Mapping[str, object]]
    ) -> None:
        if self._action_required and not any(
            bool(row["action_available"]) for row in observations
        ):
            raise ValueError(
                "C1 objective-aware sampler failed to retain a qualified action row"
            )
        if not self._alignment_required:
            return
        views = {
            (
                str(row["semantic_group"]),
                int(row["semantic_group_id"]),
                int(row["source_id"]),
                int(row["observation_unit_index"]),
            )
            for row in observations
            if str(row["semantic_group"])
        }
        eligible = False
        for semantic_key, semantic_id, source_id, unit_index in views:
            positive = any(
                other_key == semantic_key
                and other_id == semantic_id
                and other_source != source_id
                and other_unit != unit_index
                for other_key, other_id, other_source, other_unit in views
            )
            negative = any(
                other_key != semantic_key and other_id != semantic_id
                for other_key, other_id, _, _ in views
            )
            if positive and negative:
                eligible = True
                break
        if not eligible:
            raise ValueError(
                "C1 objective-aware sampler failed to retain a cross-source "
                "same-key positive and distinct-key negative"
            )

    def batch(self, *, epoch: int = 0, batch_index: int = 0) -> C1Batch:
        # Revalidate the live capability immediately before sampling so a
        # constructed loader cannot outlive revocation or drift from its
        # attested seed, dimensions, objectives, or source scope.
        self._validate_run_scope_attestation(
            self._run_scope_attestation,
            role_binding_path=self._role_binding_path,
            contract=self.config["foundation_state_c1"],
        )
        replay_record: _ReplayCapabilityRecord | None = None
        if self.run_scope_mode == "preflight_replay":
            attestation = self._run_scope_attestation
            if type(attestation) is not C1ReplayScopeAttestation:
                raise TypeError("C1 replay loader lost its opaque replay capability")
            replay_record = _REPLAY_CAPABILITIES.get(attestation._capability_id)
            if replay_record is None:
                raise ValueError("C1 replay capability is unknown, consumed, or forged")
            if replay_record.attestation is not attestation:
                raise ValueError("C1 replay capability instance is forged or copied")
            call_index = len(replay_record.observed_calls)
            if call_index >= len(_EXPECTED_REPLAY_CALL_SEQUENCE):
                raise ValueError(
                    "C1 replay capability has already consumed all coordinates"
                )
            if (epoch, batch_index) != _EXPECTED_REPLAY_CALL_SEQUENCE[call_index]:
                raise ValueError(
                    "C1 replay loader may only consume the fixed double-replay "
                    f"coordinate sequence; expected={_EXPECTED_REPLAY_CALL_SEQUENCE[call_index]}, "
                    f"received={(epoch, batch_index)}"
                )
        requirements = self._objective_observation_requirements(
            epoch=epoch, batch_index=batch_index
        )
        selected_groups = self._sample_unit_groups(
            epoch=epoch,
            batch_index=batch_index,
            required_group_uids={ref.biological_unit_uid for ref in requirements},
        )
        required_ordinals: dict[str, set[int]] = {}
        for ref in requirements:
            required_ordinals.setdefault(ref.canonical_uid, set()).add(ref.ordinal)
        representatives = [group[1][0] for group in selected_groups]
        if self.sampling_policy == FAMILY_AND_UNIT_BALANCED_POLICY:
            representative_weights = family_balanced_unit_weights(representatives)
        else:
            # Both row-proportional modes use uniform post-selection weights.
            # The constrained mode first reserves objective-required units, so
            # it must not be described as an unbiased row-balanced sample.
            representative_weights = [1.0 / len(selected_groups)] * len(selected_groups)
        observations: list[dict[str, object]] = []
        biological_uids: list[str] = []
        canonical_uids: list[str] = []
        for unit_index, (biological_uid, group) in enumerate(selected_groups):
            biological_uids.append(biological_uid)
            canonical_uids.append(group[0].canonical_uid)
            for unit in group:
                binding_row = self._binding_rows.get(unit.canonical_uid)
                if binding_row is None:
                    raise ValueError(
                        f"C1 canonical unit is absent from row binding table: {unit.canonical_uid}"
                    )
                for field in (
                    "canonical_uid",
                    "family_uid",
                    "source_id",
                    "source_accession",
                    "biological_unit_uid",
                ):
                    if str(binding_row[field]) != getattr(unit, field):
                        raise ValueError(
                            f"C1 split/observation binding mismatch: {unit.canonical_uid}:{field}"
                        )
                block_uid = str(binding_row["block_uid"])
                state = self._blocks[block_uid]
                if str(binding_row["block_sha256"]) != str(
                    state.binding["block_sha256"]
                ) or str(binding_row["adapter"]) != str(state.binding["adapter"]):
                    raise ValueError(
                        f"C1 observation binding block identity drift: {unit.canonical_uid}"
                    )
                # Dense matrices are selected by a physical row index, while the
                # metadata lookup is always keyed by the canonical metadata key.
                selector_key = str(binding_row["metadata_key_value"])
                rows = _select_observation_rows(
                    state.metadata_by_key.get(selector_key, []),
                    unit_key=biological_uid,
                    cap=self.max_observations_per_unit,
                    seed=f"{self.seed}|{epoch}|{batch_index}",
                    required_ordinals=required_ordinals.get(unit.canonical_uid, ()),
                )
                for row in rows:
                    observations.append(
                        self._load_observation(
                            unit,
                            binding_row,
                            row,
                            state,
                            biological_uid,
                            unit_index=unit_index,
                            mask_context=(
                                f"{epoch}|{batch_index}|{block_uid}|{row['__ordinal']}"
                            ),
                        )
                    )
        if not observations:
            raise ValueError("C1 loader produced an empty observation batch")
        self._validate_objective_observations(observations)
        batch = C1Batch(
            feature_ids=torch.from_numpy(
                np.stack([row["feature_ids"] for row in observations])
            ).long(),
            feature_values=torch.from_numpy(
                np.stack([row["feature_values"] for row in observations])
            ).float(),
            observed_mask=torch.from_numpy(
                np.stack([row["observed_mask"] for row in observations])
            ).bool(),
            target_mask=torch.from_numpy(
                np.stack([row["target_mask"] for row in observations])
            ).bool(),
            cell_ontology_ids=torch.tensor(
                [row["cell_ontology_id"] for row in observations], dtype=torch.long
            ),
            region_ontology_ids=torch.tensor(
                [row["region_ontology_id"] for row in observations], dtype=torch.long
            ),
            action_features=torch.from_numpy(
                np.stack([row["action_features"] for row in observations])
            ).float(),
            action_available=torch.tensor(
                [row["action_available"] for row in observations], dtype=torch.bool
            ),
            observation_unit_index=torch.tensor(
                [row["observation_unit_index"] for row in observations],
                dtype=torch.long,
            ),
            biological_unit_weights=torch.tensor(
                representative_weights, dtype=torch.float32
            ),
            source_ids=torch.tensor(
                [row["source_id"] for row in observations], dtype=torch.long
            ),
            family_ids=torch.tensor(
                [row["family_id"] for row in observations], dtype=torch.long
            ),
            semantic_group_ids=torch.tensor(
                [row["semantic_group_id"] for row in observations], dtype=torch.long
            ),
            biological_unit_uids=tuple(biological_uids),
            canonical_uids=tuple(canonical_uids),
        )
        validate_c1_batch_contract(
            batch,
            active_objectives=self.active_objectives,
            batch_size=self.batch_size,
            token_budget=self.token_budget,
            max_observations_per_unit=self.max_observations_per_unit,
            feature_vocabulary_size=self.vocabulary.feature_vocabulary_size,
            cell_ontology_size=self.vocabulary.cell_ontology_size,
            region_ontology_size=self.vocabulary.region_ontology_size,
            source_class_count=self.vocabulary.source_count,
            family_class_count=self.vocabulary.family_count,
            action_feature_dim=self.action_feature_dim,
            family_balance_required=self.sampling_policy in {
                FAMILY_AND_UNIT_BALANCED_POLICY,
                FAMILY_BALANCED_POLICY,
            },
        )
        if replay_record is not None:
            replay_record.observed_calls.append(
                (epoch, batch_index, c1_batch_hash(batch))
            )
        return batch

    def iter_batches(self, *, steps: int, epoch: int = 0) -> Iterable[C1Batch]:
        if self.run_scope_mode == "preflight_replay":
            raise ValueError(
                "C1 replay-only loaders cannot be used as an iterable training source"
            )
        if steps <= 0:
            raise ValueError("C1 loader steps must be positive")
        for batch_index in range(steps):
            yield self.batch(epoch=epoch, batch_index=batch_index)
