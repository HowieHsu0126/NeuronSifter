"""Canonical observation-row bindings for the frozen C1 transcriptomics corpus."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from jsonschema import Draft202012Validator
import pyarrow.parquet as pq

from Libs.utils.data.public_foundation_evidence import (
    canonical_hash,
    relative_repo_path,
    safe_repo_path,
    sha256_file,
)
from Libs.utils.data.public_foundation_corpus import hash_path


BINDING_SCHEMA_VERSION = "neuronsifter-c1-observation-row-binding-v1"
BINDING_FIELDS = (
    "canonical_uid",
    "content_hash",
    "family_uid",
    "source_id",
    "source_accession",
    "biological_unit_uid",
    "block_uid",
    "block_sha256",
    "adapter",
    "metadata_key_column",
    "metadata_key_value",
    "selector_kind",
    "selector_column",
    "selector_value",
    "observation_count",
    "feature_count",
    "matrix_row_count",
    "matrix_column_count",
)
QUERY_FIELDS = (
    "canonical_uid",
    "content_hash",
    "family_uid",
    "source_id",
    "source_accession",
    "publication_accession_group",
    "mirror_group",
    "biological_unit_uid",
    "regimen_uid",
    "target_program_proxy",
    "source_holdout_role",
    "split_role",
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_query_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_FIELDS:
            raise ValueError(f"Unexpected frozen query-domain fields: {reader.fieldnames}")
        rows = [{field: str(row[field]) for field in QUERY_FIELDS} for row in reader]
    if (
        not rows
        or len({row["canonical_uid"] for row in rows}) != len(rows)
        or len({row["biological_unit_uid"] for row in rows}) != len(rows)
        or any(not row["biological_unit_uid"].strip() for row in rows)
    ):
        raise ValueError("Frozen query domain is empty or contains duplicate or blank unit identities")
    return rows


def serialize_binding_rows(rows: Sequence[Mapping[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(BINDING_FIELDS),
        delimiter="\t",
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def serialize_binding_manifest(manifest: Mapping[str, object]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=True) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _block_file_path(
    repo_root: Path,
    manifest_path: Path,
    record: Mapping[str, object],
) -> Path:
    if str(record.get("relative_path") or ""):
        return safe_repo_path(manifest_path.parent, str(record["relative_path"]))
    if str(record.get("path") or ""):
        return safe_repo_path(repo_root, str(record["path"]))
    raise ValueError(f"Block file {record.get('name')} has no path binding")


def _file_binding(
    repo_root: Path,
    manifest_path: Path,
    files: Mapping[str, Mapping[str, object]],
    name: str,
    *,
    verify_file: bool = True,
) -> dict[str, object]:
    if name not in files:
        raise ValueError(f"Frozen block is missing required file: {name}")
    record = files[name]
    path = _block_file_path(repo_root, manifest_path, record)
    binding = {
        "path": relative_repo_path(repo_root, path),
        "sha256": str(record["sha256"]),
        "bytes": int(record["bytes"]),
    }
    if verify_file:
        if path.is_dir():
            count, byte_count, digest = hash_path(path)
            if count != int(record.get("file_count", 1)):
                raise ValueError(f"File count drift: {binding['path']}")
            if byte_count != binding["bytes"] or digest != binding["sha256"]:
                raise ValueError(f"Directory content drift: {binding['path']}")
        else:
            if path.stat().st_size != binding["bytes"]:
                raise ValueError(f"File byte count drift: {binding['path']}")
            if sha256_file(path) != binding["sha256"]:
                raise ValueError(f"File hash drift: {binding['path']}")
    return binding


def _feature_key_column(columns: Sequence[str]) -> str:
    for candidate in ("canonical_gene_id", "canonical_feature_id", "feature_id"):
        if candidate in columns:
            return candidate
    raise ValueError(f"No canonical feature key column in {list(columns)}")


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _validate_feature_keys(path: Path, key_column: str) -> int:
    frame = pd.read_parquet(path, columns=[key_column])
    keys = frame[key_column].astype(str)
    if keys.eq("").any() or keys.duplicated().any():
        raise ValueError(f"Feature metadata key is blank or duplicated: {path}")
    return len(frame)


def _zarray_shape(zarr_path: Path, relative: str) -> list[int]:
    payload = _load_json(zarr_path / relative / ".zarray")
    shape = payload.get("shape")
    if not isinstance(shape, list) or not shape or any(int(value) < 1 for value in shape):
        raise ValueError(f"Invalid Zarr structural shape: {zarr_path / relative}")
    return [int(value) for value in shape]


def _file_records(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = manifest.get("block_files")
    if not isinstance(raw, list):
        raise ValueError("Frozen block manifest has no block_files list")
    records = {
        str(row["name"]): row
        for row in raw
        if isinstance(row, Mapping) and str(row.get("name") or "")
    }
    if len(records) != len(raw):
        raise ValueError("Frozen block file names are missing or duplicated")
    return records


def _validate_query_identity(row: Mapping[str, str], block: Mapping[str, object]) -> None:
    biological_uid = row["biological_unit_uid"]
    expected_uid = f"pfc_unit:{canonical_hash([block['canonical_uid'], biological_uid])[:24]}"
    expected_content = canonical_hash([block["block_sha256"], biological_uid])
    if row["canonical_uid"] != expected_uid or row["content_hash"] != expected_content:
        raise ValueError(f"Frozen query identity mismatch: {row['canonical_uid']}")
    for key, block_key in (
        ("family_uid", "family_id"),
        ("source_id", "source_id"),
        ("source_accession", "source_accession"),
    ):
        if row[key] != str(block[block_key]):
            raise ValueError(f"Frozen query/block mismatch: {row['canonical_uid']}:{key}")


def _common_descriptor(
    repo_root: Path,
    block: Mapping[str, object],
    manifest_path: Path,
    manifest: Mapping[str, object],
    adapter: str,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    files = _file_records(manifest)
    effective_sample_size = float(block["effective_sample_size"])
    if not effective_sample_size.is_integer():
        raise ValueError(f"Frozen ESS is not integral for C1 unit binding: {block['canonical_uid']}")
    descriptor = {
        "block_uid": str(block["canonical_uid"]),
        "block_sha256": str(block["block_sha256"]),
        "family_uid": str(block["family_id"]),
        "source_id": str(block["source_id"]),
        "source_accession": str(block["source_accession"]),
        "adapter": adapter,
        "manifest": {
            "path": relative_repo_path(repo_root, manifest_path),
            "sha256": str(block["manifest_sha256"]),
        },
        "biological_unit_count": int(effective_sample_size),
        "observation_count": int(block["observation_count"]),
    }
    if sha256_file(manifest_path) != descriptor["manifest"]["sha256"]:
        raise ValueError(f"Frozen block manifest hash drift: {manifest_path}")
    return descriptor, files


def _bulk_bindings(
    repo_root: Path,
    block: Mapping[str, object],
    manifest_path: Path,
    manifest: Mapping[str, object],
    query_rows: Sequence[Mapping[str, str]],
    *,
    verify_files: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    descriptor, files = _common_descriptor(
        repo_root, block, manifest_path, manifest, "parquet_long_keyed"
    )
    metadata_name = "sample_metadata.parquet"
    feature_name = "feature_metadata.parquet"
    metadata = _file_binding(
        repo_root, manifest_path, files, metadata_name, verify_file=verify_files
    )
    values = _file_binding(
        repo_root, manifest_path, files, "values.parquet", verify_file=verify_files
    )
    features = _file_binding(
        repo_root, manifest_path, files, feature_name, verify_file=verify_files
    )
    metadata_path = safe_repo_path(repo_root, str(metadata["path"]))
    values_path = safe_repo_path(repo_root, str(values["path"]))
    features_path = safe_repo_path(repo_root, str(features["path"]))
    metadata_frame = pd.read_parquet(metadata_path, columns=["biological_unit_id"])
    feature_columns = _parquet_columns(features_path)
    feature_key = _feature_key_column(feature_columns)
    feature_frame = pd.read_parquet(features_path, columns=[feature_key])
    feature_keys = feature_frame[feature_key].astype(str)
    if feature_keys.eq("").any() or feature_keys.duplicated().any():
        raise ValueError(f"Bulk feature metadata key is blank or duplicated: {descriptor['block_uid']}")
    value_frame = pd.read_parquet(
        values_path, columns=["biological_unit_id", "canonical_gene_id"]
    )
    metadata_units = metadata_frame["biological_unit_id"].astype(str)
    if metadata_units.duplicated().any():
        raise ValueError(f"Bulk metadata biological units are not unique: {descriptor['block_uid']}")
    if value_frame.duplicated(["biological_unit_id", "canonical_gene_id"]).any():
        raise ValueError(f"Bulk value keys are not unique: {descriptor['block_uid']}")
    feature_ids = set(feature_frame[feature_key].astype(str))
    if set(value_frame["canonical_gene_id"].astype(str)) != feature_ids:
        raise ValueError(f"Bulk values do not cover the frozen feature set: {descriptor['block_uid']}")
    per_unit_features = value_frame.groupby("biological_unit_id")["canonical_gene_id"].nunique()
    feature_count = len(feature_frame)
    if len(value_frame) != len(metadata_frame) * feature_count:
        raise ValueError(f"Bulk logical matrix is incomplete: {descriptor['block_uid']}")
    if set(per_unit_features.index.astype(str)) != set(metadata_units):
        raise ValueError(f"Bulk value/metadata unit mismatch: {descriptor['block_uid']}")
    if set(map(int, per_unit_features.tolist())) != {feature_count}:
        raise ValueError(f"Bulk unit feature coverage is incomplete: {descriptor['block_uid']}")

    query_units = {row["biological_unit_uid"] for row in query_rows}
    if query_units != set(metadata_units):
        raise ValueError(f"Bulk query/metadata unit mismatch: {descriptor['block_uid']}")
    rows = [
        {
            "canonical_uid": row["canonical_uid"],
            "content_hash": row["content_hash"],
            "family_uid": row["family_uid"],
            "source_id": row["source_id"],
            "source_accession": row["source_accession"],
            "biological_unit_uid": row["biological_unit_uid"],
            "block_uid": descriptor["block_uid"],
            "block_sha256": descriptor["block_sha256"],
            "adapter": descriptor["adapter"],
            "metadata_key_column": "biological_unit_id",
            "metadata_key_value": row["biological_unit_uid"],
            "selector_kind": "keyed_value_join",
            "selector_column": "biological_unit_id",
            "selector_value": row["biological_unit_uid"],
            "observation_count": 1,
            "feature_count": feature_count,
            "matrix_row_count": len(metadata_frame),
            "matrix_column_count": feature_count,
        }
        for row in query_rows
    ]
    descriptor.update(
        {
            "matrix_shape": [len(metadata_frame), feature_count],
            "stored_value_row_count": len(value_frame),
            "metadata": {**metadata, "record_count": len(metadata_frame), "key_column": "biological_unit_id"},
            "values": {**values, "format": "parquet_long", "unit_key_column": "biological_unit_id", "feature_key_column": "canonical_gene_id"},
            "features": {**features, "record_count": feature_count, "key_column": feature_key},
            "matrix_manifest": None,
            "selector_contract": {
                "kind": "keyed_value_join",
                "column": "biological_unit_id",
                "canonical_row_index_source": "stable_key_not_physical_ordinal",
                "forbidden_columns": [],
            },
        }
    )
    return rows, descriptor


def _matrix_manifest_binding(
    repo_root: Path,
    manifest_path: Path,
    files: Mapping[str, Mapping[str, object]],
    *,
    verify_files: bool = True,
) -> tuple[dict[str, object], list[int], dict[str, object]]:
    binding = _file_binding(
        repo_root,
        manifest_path,
        files,
        "matrix_manifest.json",
        verify_file=verify_files,
    )
    payload = _load_json(safe_repo_path(repo_root, str(binding["path"])))
    shape = payload.get("matrix_shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"Invalid matrix shape: {binding['path']}")
    return binding, [int(shape[0]), int(shape[1])], payload


def _atlas_bindings(
    repo_root: Path,
    block: Mapping[str, object],
    manifest_path: Path,
    manifest: Mapping[str, object],
    query_rows: Sequence[Mapping[str, str]],
    *,
    verify_files: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    descriptor, files = _common_descriptor(
        repo_root, block, manifest_path, manifest, "zarr_csr_ordinal"
    )
    metadata = _file_binding(
        repo_root,
        manifest_path,
        files,
        "cell_metadata.parquet",
        verify_file=verify_files,
    )
    values = _file_binding(
        repo_root,
        manifest_path,
        files,
        "sparse_counts.zarr",
        verify_file=verify_files,
    )
    features = _file_binding(
        repo_root,
        manifest_path,
        files,
        "feature_metadata.parquet",
        verify_file=verify_files,
    )
    matrix_binding, matrix_shape, matrix_payload = _matrix_manifest_binding(
        repo_root, manifest_path, files, verify_files=verify_files
    )
    metadata_path = safe_repo_path(repo_root, str(metadata["path"]))
    features_path = safe_repo_path(repo_root, str(features["path"]))
    columns = _parquet_columns(metadata_path)
    donor_column = "donor_id" if "donor_id" in columns else "donor_label"
    selected_columns = [donor_column]
    forbidden_columns: list[str] = []
    if "source_raw_row_index" in columns:
        selected_columns.append("source_raw_row_index")
        forbidden_columns.append("source_raw_row_index")
    metadata_frame = pd.read_parquet(metadata_path, columns=selected_columns)
    feature_columns = _parquet_columns(features_path)
    feature_key = _feature_key_column(feature_columns)
    feature_count = _validate_feature_keys(features_path, feature_key)
    if matrix_shape != [len(metadata_frame), feature_count]:
        raise ValueError(f"Atlas matrix/metadata/feature shape mismatch: {descriptor['block_uid']}")
    zarr_path = safe_repo_path(repo_root, str(values["path"]))
    data_shape = _zarray_shape(zarr_path, "data")
    indices_shape = _zarray_shape(zarr_path, "indices")
    indptr_shape = _zarray_shape(zarr_path, "indptr")
    expected_nnz = int(matrix_payload.get("nnz", -1))
    if data_shape != indices_shape or data_shape != [expected_nnz]:
        raise ValueError(f"CSR data/indices shape mismatch: {descriptor['block_uid']}")
    if indptr_shape != [matrix_shape[0] + 1]:
        raise ValueError(f"CSR indptr shape mismatch: {descriptor['block_uid']}")
    donor_counts = Counter(metadata_frame[donor_column].astype(str))
    query_units = {row["biological_unit_uid"] for row in query_rows}
    if query_units != set(donor_counts):
        raise ValueError(f"Atlas query/donor mismatch: {descriptor['block_uid']}")
    forbidden_mismatch_count = 0
    if forbidden_columns:
        raw_indices = metadata_frame["source_raw_row_index"].astype(int).tolist()
        forbidden_mismatch_count = sum(index != value for index, value in enumerate(raw_indices))
        if (
            descriptor["block_uid"] == "cellxgene_amygdala_2021_phase1e_v1"
            and forbidden_mismatch_count != 14030
        ):
            raise ValueError(
                "CELLxGENE source_raw_row_index mismatch count drifted: "
                f"{forbidden_mismatch_count} != 14030"
            )

    rows = [
        {
            "canonical_uid": row["canonical_uid"],
            "content_hash": row["content_hash"],
            "family_uid": row["family_uid"],
            "source_id": row["source_id"],
            "source_accession": row["source_accession"],
            "biological_unit_uid": row["biological_unit_uid"],
            "block_uid": descriptor["block_uid"],
            "block_sha256": descriptor["block_sha256"],
            "adapter": descriptor["adapter"],
            "metadata_key_column": donor_column,
            "metadata_key_value": row["biological_unit_uid"],
            "selector_kind": "metadata_ordinal_where",
            "selector_column": donor_column,
            "selector_value": row["biological_unit_uid"],
            "observation_count": donor_counts[row["biological_unit_uid"]],
            "feature_count": feature_count,
            "matrix_row_count": matrix_shape[0],
            "matrix_column_count": matrix_shape[1],
        }
        for row in query_rows
    ]
    descriptor.update(
        {
            "matrix_shape": matrix_shape,
            "stored_value_row_count": matrix_shape[0],
            "metadata": {**metadata, "record_count": len(metadata_frame), "key_column": donor_column},
            "values": {**values, "format": "zarr_csr", "unit_key_column": donor_column, "feature_key_column": "feature_index"},
            "features": {**features, "record_count": feature_count, "key_column": feature_key},
            "matrix_manifest": matrix_binding,
            "selector_contract": {
                "kind": "metadata_ordinal_where",
                "column": donor_column,
                "canonical_row_index_source": "cell_metadata_parquet_ordinal",
                "forbidden_columns": forbidden_columns,
            },
            "forbidden_row_index_mismatch_count": forbidden_mismatch_count,
        }
    )
    return rows, descriptor


def _dense_bindings(
    repo_root: Path,
    block: Mapping[str, object],
    manifest_path: Path,
    manifest: Mapping[str, object],
    query_rows: Sequence[Mapping[str, str]],
    *,
    verify_files: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    descriptor, files = _common_descriptor(
        repo_root, block, manifest_path, manifest, "zarr_dense_indexed"
    )
    metadata = _file_binding(
        repo_root,
        manifest_path,
        files,
        "observations.parquet",
        verify_file=verify_files,
    )
    values = _file_binding(
        repo_root,
        manifest_path,
        files,
        "expression.zarr",
        verify_file=verify_files,
    )
    features = _file_binding(
        repo_root,
        manifest_path,
        files,
        "features.parquet",
        verify_file=verify_files,
    )
    matrix_binding, matrix_shape, _ = _matrix_manifest_binding(
        repo_root, manifest_path, files, verify_files=verify_files
    )
    metadata_path = safe_repo_path(repo_root, str(metadata["path"]))
    features_path = safe_repo_path(repo_root, str(features["path"]))
    metadata_frame = pd.read_parquet(
        metadata_path, columns=["biological_unit_uid", "value_row_index"]
    )
    feature_columns = _parquet_columns(features_path)
    feature_key = _feature_key_column(feature_columns)
    feature_count = _validate_feature_keys(features_path, feature_key)
    if matrix_shape != [len(metadata_frame), feature_count]:
        raise ValueError(f"Dense matrix/metadata/feature shape mismatch: {descriptor['block_uid']}")
    zarr_shape = _zarray_shape(safe_repo_path(repo_root, str(values["path"])), "values")
    if zarr_shape != matrix_shape:
        raise ValueError(f"Dense Zarr shape mismatch: {descriptor['block_uid']}")
    units = metadata_frame["biological_unit_uid"].astype(str)
    indices = metadata_frame["value_row_index"].astype(int)
    if units.duplicated().any():
        raise ValueError(f"Dense biological units are not one-to-one: {descriptor['block_uid']}")
    if sorted(indices.tolist()) != list(range(len(metadata_frame))):
        raise ValueError(f"Dense value_row_index is not a complete bijection: {descriptor['block_uid']}")
    query_units = {row["biological_unit_uid"] for row in query_rows}
    if query_units != set(units):
        raise ValueError(f"Dense query/metadata unit mismatch: {descriptor['block_uid']}")
    index_by_unit = dict(zip(units, indices, strict=True))
    rows = [
        {
            "canonical_uid": row["canonical_uid"],
            "content_hash": row["content_hash"],
            "family_uid": row["family_uid"],
            "source_id": row["source_id"],
            "source_accession": row["source_accession"],
            "biological_unit_uid": row["biological_unit_uid"],
            "block_uid": descriptor["block_uid"],
            "block_sha256": descriptor["block_sha256"],
            "adapter": descriptor["adapter"],
            "metadata_key_column": "biological_unit_uid",
            "metadata_key_value": row["biological_unit_uid"],
            "selector_kind": "explicit_value_row_index",
            "selector_column": "value_row_index",
            "selector_value": int(index_by_unit[row["biological_unit_uid"]]),
            "observation_count": 1,
            "feature_count": feature_count,
            "matrix_row_count": matrix_shape[0],
            "matrix_column_count": matrix_shape[1],
        }
        for row in query_rows
    ]
    descriptor.update(
        {
            "matrix_shape": matrix_shape,
            "stored_value_row_count": matrix_shape[0],
            "metadata": {**metadata, "record_count": len(metadata_frame), "key_column": "biological_unit_uid"},
            "values": {**values, "format": "zarr_dense", "unit_key_column": "value_row_index", "feature_key_column": "feature_index"},
            "features": {**features, "record_count": feature_count, "key_column": feature_key},
            "matrix_manifest": matrix_binding,
            "selector_contract": {
                "kind": "explicit_value_row_index",
                "column": "value_row_index",
                "canonical_row_index_source": "observations_parquet_value_row_index",
                "forbidden_columns": [],
            },
        }
    )
    return rows, descriptor


def build_c1_observation_binding_artifacts(
    repo_root: Path,
    bundle: Mapping[str, object],
    qualified_blocks: Sequence[Mapping[str, object]],
    query_domain: Mapping[str, object],
    *,
    row_table_path: str,
    verify_files: bool = True,
) -> tuple[str, dict[str, object]]:
    repo_root = repo_root.resolve()
    query_path = safe_repo_path(repo_root, str(query_domain["path"]))
    if sha256_file(query_path) != str(query_domain["sha256"]):
        raise ValueError("Frozen query-domain hash drift")
    query_rows = _read_query_rows(query_path)
    if len(query_rows) != int(query_domain["record_count"]):
        raise ValueError("Frozen query-domain record-count drift")
    all_rows: list[dict[str, object]] = []
    descriptors: list[dict[str, object]] = []
    used_query_uids: set[str] = set()
    for block in sorted(qualified_blocks, key=lambda value: str(value["canonical_uid"])):
        block_query_rows = [
            row
            for row in query_rows
            if row["source_id"] == str(block["source_id"])
            and row["source_accession"] == str(block["source_accession"])
            and row["family_uid"] == str(block["family_id"])
            and row["canonical_uid"]
            == f"pfc_unit:{canonical_hash([block['canonical_uid'], row['biological_unit_uid']])[:24]}"
        ]
        block_query_uids = {row["canonical_uid"] for row in block_query_rows}
        if not block_query_rows or used_query_uids & block_query_uids:
            raise ValueError(
                "Frozen block does not resolve a unique query-domain unit set: "
                f"{block['canonical_uid']}"
            )
        used_query_uids.update(block_query_uids)
        for query_row in block_query_rows:
            _validate_query_identity(query_row, block)
        manifest_path = safe_repo_path(repo_root, str(block["manifest_path"]))
        manifest = _load_json(manifest_path)
        family = str(block["family_id"])
        if family == "cns_disease_aging_state_transcriptomics":
            rows, descriptor = _bulk_bindings(
                repo_root,
                block,
                manifest_path,
                manifest,
                block_query_rows,
                verify_files=verify_files,
            )
        elif family == "cns_cell_type_region_atlas_transcriptomics":
            rows, descriptor = _atlas_bindings(
                repo_root,
                block,
                manifest_path,
                manifest,
                block_query_rows,
                verify_files=verify_files,
            )
        elif family == "compound_dose_time_perturbation_transcriptomics":
            rows, descriptor = _dense_bindings(
                repo_root,
                block,
                manifest_path,
                manifest,
                block_query_rows,
                verify_files=verify_files,
            )
        else:
            raise ValueError(f"Unsupported frozen C1 family: {family}")
        all_rows.extend(rows)
        descriptors.append(descriptor)
    if used_query_uids != {row["canonical_uid"] for row in query_rows}:
        raise ValueError("Frozen query domain contains an unbound canonical unit")
    all_rows.sort(key=lambda row: str(row["canonical_uid"]))
    if [row["canonical_uid"] for row in all_rows] != sorted(
        row["canonical_uid"] for row in query_rows
    ):
        raise ValueError("Observation bindings do not cover the exact frozen query UID set")

    row_text = serialize_binding_rows(all_rows)
    row_sha256 = _sha256_text(row_text)
    adapter_unit_counts = dict(sorted(Counter(str(row["adapter"]) for row in all_rows).items()))
    adapter_observation_counts: Counter[str] = Counter()
    for row in all_rows:
        adapter_observation_counts[str(row["adapter"])] += int(row["observation_count"])
    observation_count = sum(int(row["observation_count"]) for row in all_rows)
    binding_payload = {
        "corpus_hash": str(bundle["corpus_hash"]),
        "query_domain_sha256": str(query_domain["sha256"]),
        "row_binding_table_sha256": row_sha256,
        "blocks": descriptors,
    }
    is_v2 = str(bundle["corpus_uid"]) == "public_foundation_corpus_rescue"
    manifest = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "corpus_uid": str(bundle["corpus_uid"]),
        "corpus_hash": str(bundle["corpus_hash"]),
        "scope": "reduced_transcriptomics_core",
        "derivation_policy": (
            "read_only_derived_artifact_for_frozen_rescue"
            if is_v2
            else "read_only_derived_artifact_outside_frozen_v1"
        ),
        "query_domain": {
            "path": str(query_domain["path"]),
            "sha256": str(query_domain["sha256"]),
            "record_count": int(query_domain["record_count"]),
        },
        "row_binding_table": {
            "path": row_table_path,
            "sha256": row_sha256,
            "record_count": len(all_rows),
        },
        "block_count": len(descriptors),
        "biological_unit_count": len(all_rows),
        "observation_count": observation_count,
        "adapter_unit_counts": adapter_unit_counts,
        "adapter_observation_counts": dict(sorted(adapter_observation_counts.items())),
        "within_unit_aggregation_required": True,
        "blocks": descriptors,
        "binding_hash": canonical_hash(binding_payload),
    }
    return row_text, manifest


def _blocker(code: str, path: object, message: object) -> dict[str, str]:
    return {"code": code, "path": str(path), "message": str(message)}


def validate_c1_observation_binding(
    config: Mapping[str, object],
    repo_root: Path,
    *,
    bundle: Mapping[str, object] | None = None,
    qualified_blocks: Sequence[Mapping[str, object]] | None = None,
    verify_files: bool = True,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    repo_root = repo_root.resolve()
    actual: dict[str, object] = {
        "manifest_path": None,
        "manifest_sha256": None,
        "row_table_path": None,
        "row_table_sha256": None,
        "binding_hash": None,
        "block_count": None,
        "biological_unit_count": None,
        "observation_count": None,
    }
    blockers: list[dict[str, str]] = []
    try:
        contract = config["foundation_state_c1"]
        frozen = contract["frozen_corpus"]
        required = frozen["required_observation_binding"]
        bundle_payload = dict(bundle) if bundle is not None else _load_json(
            safe_repo_path(repo_root, str(frozen["evidence_bundle_path"]))
        )
        if qualified_blocks is None:
            observation_artifact = _load_json(
                safe_repo_path(
                    repo_root,
                    str(bundle_payload["evidence"]["canonical_observation_blocks"]["path"]),
                )
            )
            qualified = [
                row
                for row in observation_artifact["blocks"]
                if isinstance(row, Mapping) and row.get("observation_eligible") is True
            ]
        else:
            qualified = list(qualified_blocks)
        row_text, rebuilt = build_c1_observation_binding_artifacts(
            repo_root,
            bundle_payload,
            qualified,
            frozen["required_query_domain"],
            row_table_path=str(required["row_table_path"]),
            verify_files=verify_files,
        )
        manifest_path = safe_repo_path(repo_root, str(required["manifest_path"]))
        row_path = safe_repo_path(repo_root, str(required["row_table_path"]))
        stored = _load_json(manifest_path)
        schema = _load_json(safe_repo_path(repo_root, str(required["schema_path"])))
        Draft202012Validator.check_schema(schema)
        schema_errors = sorted(
            Draft202012Validator(schema).iter_errors(stored),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        blockers.extend(
            _blocker(
                "c1_observation_binding_schema_violation",
                ".".join(map(str, error.absolute_path)) or "$",
                error.message,
            )
            for error in schema_errors
        )
        actual.update(
            {
                "manifest_path": str(required["manifest_path"]),
                "manifest_sha256": sha256_file(manifest_path),
                "row_table_path": str(required["row_table_path"]),
                "row_table_sha256": sha256_file(row_path),
                "binding_hash": stored.get("binding_hash"),
                "block_count": stored.get("block_count"),
                "biological_unit_count": stored.get("biological_unit_count"),
                "observation_count": stored.get("observation_count"),
            }
        )
        expected_actual = {
            "manifest_path": required["manifest_path"],
            "manifest_sha256": required["manifest_sha256"],
            "row_table_path": required["row_table_path"],
            "row_table_sha256": required["row_table_sha256"],
            "binding_hash": required["binding_hash"],
            "block_count": required["block_count"],
            "biological_unit_count": required["biological_unit_count"],
            "observation_count": required["observation_count"],
        }
        if actual != expected_actual:
            blockers.append(
                _blocker("c1_observation_binding_identity_mismatch", required["manifest_path"], actual)
            )
        if stored != rebuilt:
            blockers.append(
                _blocker(
                    "c1_observation_binding_manifest_drift",
                    required["manifest_path"],
                    "stored manifest differs from canonical metadata reconstruction",
                )
            )
        if row_path.read_text(encoding="utf-8") != row_text:
            blockers.append(
                _blocker(
                    "c1_observation_binding_row_table_drift",
                    required["row_table_path"],
                    "stored row table differs from canonical metadata reconstruction",
                )
            )
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_observation_binding_reconstruction_failed", "$", exc))
    return blockers, actual
