"""Materialize the first qualified CELLxGENE amygdala atlas observation block.

The builder reads the official sparse H5AD without densifying it. Cells whose
cell-type label or ontology identifier is ``unknown`` or missing are written to
an explicit quarantine table and excluded from the canonical mainline matrix.
The donor axis remains the only ESS axis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.materialize_phase1e_allen_whb_nonneurons_mainline_block import (
    _h5ad_dataframe_index,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import upsert_accession_requalification_delta
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic
from Libs.utils.model.dynamics_store import initialize_zarr_store, write_zarr_array_from_loader


BUILDER_ID = "Libs/scripts/builders/materialize_phase1e_cellxgene_amygdala_mainline_block.py"
MANIFEST_VERSION = "neuronsifter-phase1e-cellxgene-atlas-mainline-block-v1"
ADMISSION_VERSION = "neuronsifter-phase1e-cellxgene-atlas-mainline-admission-v1"
SOURCE_CARD_VERSION = "neuronsifter-phase1e-cellxgene-atlas-source-card-v1"
RETRIEVAL_RECORD_SCHEMA_VERSION = "neuronsifter-phase1e-cellxgene-retrieval-record-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str) -> Path:
    return resolve_repo_path(value)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _decode(values: Iterable[object]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _obs_column(group: h5py.Group, name: str) -> list[str]:
    item = group.get(name)
    if isinstance(item, h5py.Dataset):
        return _decode(item[:])
    if not isinstance(item, h5py.Group):
        raise ValueError(f"CELLxGENE H5AD obs has no supported column: {name}")
    codes = np.asarray(item["codes"][:])
    categories = _decode(item["categories"][:])
    values: list[str] = []
    for code in codes:
        index = int(code)
        values.append(categories[index] if index >= 0 else "")
    return values


def _matrix_layout(raw_path: Path) -> tuple[tuple[int, int], str, str, str, list[str], list[str]]:
    with h5py.File(raw_path, "r") as handle:
        x_group = handle.get("X")
        obs_group = handle.get("obs")
        var_group = handle.get("var")
        if not isinstance(x_group, h5py.Group) or not isinstance(obs_group, h5py.Group) or not isinstance(var_group, h5py.Group):
            raise ValueError("CELLxGENE H5AD lacks sparse X, obs, or var groups")
        encoding = x_group.attrs.get("encoding-type")
        encoding = encoding.decode("utf-8") if isinstance(encoding, bytes) else str(encoding)
        if encoding != "csr_matrix" or not {"data", "indices", "indptr"}.issubset(x_group.keys()):
            raise ValueError("CELLxGENE H5AD X must be a CSR matrix with data, indices, and indptr")
        shape = tuple(int(value) for value in x_group.attrs.get("shape", ()))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"CELLxGENE H5AD sparse shape is invalid: {shape}")
        obs_names = _h5ad_dataframe_index(obs_group, "obs")
        var_names = _h5ad_dataframe_index(var_group, "var")
        if len(obs_names) != shape[0] or len(var_names) != shape[1]:
            raise ValueError("CELLxGENE H5AD axis names do not match X shape")
        if len(set(obs_names)) != len(obs_names) or len(set(var_names)) != len(var_names):
            raise ValueError("CELLxGENE H5AD observation or feature identifiers are not unique")
        return shape, encoding, str(x_group["data"].dtype), str(x_group["indices"].dtype), obs_names, var_names


def _cell_tables(raw_path: Path, obs_names: list[str], policy: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(raw_path, "r") as handle:
        obs = handle["obs"]
        columns = {
            name: _obs_column(obs, name)
            for name in (
                "donor_id", "cell_type", "cell_type_ontology_term_id", "tissue", "tissue_ontology_term_id",
                "disease", "disease_ontology_term_id", "assay", "assay_ontology_term_id", "suspension_type",
                "is_primary_data", "development_stage", "development_stage_ontology_term_id", "sex", "sex_ontology_term_id",
            )
        }
    raw = pd.DataFrame(columns)
    raw.insert(0, "source_observation_id", obs_names)
    raw.insert(1, "source_raw_row_index", np.arange(len(raw), dtype=np.int64))
    quarantine_values = {str(value).strip().lower() for value in policy["parser"]["cell_type_quarantine_values"]}
    label = raw["cell_type"].astype(str).str.strip()
    ontology = raw["cell_type_ontology_term_id"].astype(str).str.strip()
    quarantine = label.str.lower().isin(quarantine_values) | ontology.str.lower().isin(quarantine_values)
    raw["quarantine_reason"] = np.where(quarantine, "cell_type_label_or_ontology_unknown_or_missing", "")
    quarantined = raw.loc[quarantine].copy()
    qualified = raw.loc[~quarantine].copy()
    required = ("source_observation_id", "donor_id", "cell_type", "cell_type_ontology_term_id", "tissue", "tissue_ontology_term_id", "disease", "disease_ontology_term_id", "assay", "assay_ontology_term_id", "is_primary_data")
    if any(qualified[name].astype(str).str.strip().eq("").any() for name in required):
        raise ValueError("CELLxGENE qualified cells have missing required semantic metadata")
    if not qualified["is_primary_data"].astype(str).str.lower().eq("true").all():
        raise ValueError("CELLxGENE block contains a non-primary-data cell")
    if qualified["donor_id"].nunique() != int(policy["qc"]["minimum_donor_count"]):
        raise ValueError("CELLxGENE qualified donor count does not equal the locked first-block contract")
    samples = pd.DataFrame(
        {
            "biological_unit_id": qualified["source_observation_id"].astype(str),
            "source_observation_id": qualified["source_observation_id"].astype(str),
            "source_raw_row_index": qualified["source_raw_row_index"].astype(np.int64),
            "donor_id": qualified["donor_id"].astype(str),
            "cell_type_id": qualified["cell_type_ontology_term_id"].astype(str),
            "cell_type_label": qualified["cell_type"].astype(str),
            "cell_type_namespace": "Cell_Ontology",
            "brain_region_id": qualified["tissue_ontology_term_id"].astype(str),
            "brain_region_label": qualified["tissue"].astype(str),
            "brain_region_namespace": "UBERON",
            "disease_state_id": qualified["disease_ontology_term_id"].astype(str),
            "disease_state_label": qualified["disease"].astype(str),
            "species_taxonomy_id": "NCBITaxon:9606",
            "species_label": "Homo sapiens",
            "assay_id": qualified["assay_ontology_term_id"].astype(str),
            "assay_label": qualified["assay"].astype(str),
            "modality": "transcriptomics",
            "suspension_type": qualified["suspension_type"].astype(str),
            "development_stage": qualified["development_stage"].astype(str),
            "development_stage_id": qualified["development_stage_ontology_term_id"].astype(str),
            "sex": qualified["sex"].astype(str),
            "sex_id": qualified["sex_ontology_term_id"].astype(str),
            "technical_replicate_group": qualified["source_observation_id"].astype(str),
            "time_status": "not_applicable_cross_sectional_reference_atlas",
            "dose_status": "not_applicable_observational_reference_atlas",
        }
    )
    if samples["biological_unit_id"].duplicated().any() or samples["donor_id"].eq("").any():
        raise ValueError("CELLxGENE canonical sample identifiers are invalid")
    return samples, quarantined


def _feature_table(raw_path: Path, var_names: list[str]) -> pd.DataFrame:
    with h5py.File(raw_path, "r") as handle:
        var = handle["var"]
        feature_names = _obs_column(var, "feature_name")
        feature_reference = _obs_column(var, "feature_reference")
        feature_biotype = _obs_column(var, "feature_biotype")
    features = pd.DataFrame(
        {
            "feature_index": np.arange(len(var_names), dtype=np.int64),
            "canonical_feature_id": var_names,
            "canonical_gene_symbol": feature_names,
            "canonical_namespace": "Ensembl_Gene",
            "source_feature_id": var_names,
            "source_feature_name": feature_names,
            "feature_reference": feature_reference,
            "feature_biotype": feature_biotype,
            "mapping_method": "official_cellxgene_h5ad_var_index_exact_identifier_join",
            "mapping_confidence": "exact",
        }
    )
    if features["canonical_feature_id"].duplicated().any() or features["canonical_feature_id"].eq("").any():
        raise ValueError("CELLxGENE canonical Ensembl feature IDs are invalid")
    if not features["canonical_feature_id"].str.fullmatch(r"ENSG\d+").all():
        raise ValueError("CELLxGENE feature IDs are not stable Ensembl gene identifiers")
    if set(features["feature_reference"]) != {"NCBITaxon:9606"}:
        raise ValueError("CELLxGENE feature reference does not bind human genes")
    return features


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    return [(int(group[0]), int(group[-1]) + 1) for group in np.split(indices, splits)]


def _write_filtered_sparse_store(
    raw_path: Path,
    destination: Path,
    *,
    selected_rows: np.ndarray,
    feature_count: int,
    chunk_size: int,
) -> dict[str, object]:
    with h5py.File(raw_path, "r") as handle:
        x_group = handle["X"]
        source_data = x_group["data"]
        source_indices = x_group["indices"]
        source_indptr = np.asarray(x_group["indptr"][:], dtype=np.int64)
        if source_indptr[0] != 0 or np.any(np.diff(source_indptr) < 0) or source_indptr[-1] != len(source_data):
            raise ValueError("CELLxGENE sparse CSR pointers are inconsistent")
        row_lengths = source_indptr[selected_rows + 1] - source_indptr[selected_rows]
        indptr = np.concatenate(([0], np.cumsum(row_lengths, dtype=np.int64)))
        nnz = int(indptr[-1])
        data = np.empty(nnz, dtype=source_data.dtype)
        indices = np.empty(nnz, dtype=source_indices.dtype)
        offset = 0
        for start_row, stop_row in _runs(selected_rows):
            start = int(source_indptr[start_row])
            stop = int(source_indptr[stop_row])
            width = stop - start
            data[offset:offset + width] = source_data[start:stop]
            indices[offset:offset + width] = source_indices[start:stop]
            offset += width
        if offset != nnz:
            raise ValueError("CELLxGENE filtered CSR copy did not cover the expected nonzero values")
    if len(indices) and (indices.min() < 0 or indices.max() >= feature_count):
        raise ValueError("CELLxGENE sparse feature indices are out of bounds")
    initialize_zarr_store(
        destination,
        {
            "format_name": "neuronsifter_sparse_csr_zarr_v1",
            "matrix_encoding": "csr_matrix",
            "matrix_shape": [int(len(selected_rows)), int(feature_count)],
            "source_value_kind": "official_cellxgene_raw_count_matrix_filtered_to_nonquarantined_cell_types",
        },
    )
    for name, array, axis in (("data", data, "nonzero_entry"), ("indices", indices, "nonzero_entry"), ("indptr", indptr, "compressed_axis_pointer")):
        write_zarr_array_from_loader(
            destination,
            name,
            (len(array),),
            (min(chunk_size, max(len(array), 1)),),
            array.dtype,
            lambda slices, payload=array: payload[slices[0]],
            attrs={"axis_names": [axis]},
        )
    finite = True
    integer_like = True
    minimum: float | int | None = None
    maximum: float | int | None = None
    for start in range(0, len(data), chunk_size):
        chunk = data[start:min(start + chunk_size, len(data))]
        if not np.isfinite(chunk).all():
            finite = False
            break
        integer_like = integer_like and bool(np.equal(chunk, np.floor(chunk)).all())
        low = chunk.min().item() if len(chunk) else None
        high = chunk.max().item() if len(chunk) else None
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
    if not finite or minimum is None or minimum <= 0 or not integer_like:
        raise ValueError("CELLxGENE retained sparse values are not finite positive integer-like raw counts")
    return {
        "matrix_shape": [int(len(selected_rows)), int(feature_count)],
        "matrix_encoding": "csr_matrix",
        "nnz": nnz,
        "data_dtype": str(data.dtype),
        "indices_dtype": str(indices.dtype),
        "indptr_dtype": str(indptr.dtype),
        "finite_nonzero_values": finite,
        "integer_like_nonzero_values": integer_like,
        "minimum_nonzero_value": minimum,
        "maximum_nonzero_value": maximum,
        "value_unit_original": "raw count matrix",
        "value_unit_canonical": "raw count matrix",
        "unit_conversion_formula": "identity",
    }


def _record(path: Path, name: str) -> dict[str, object]:
    file_count, byte_count, digest = hash_path(path)
    return {"name": name, "relative_path": name, "file_count": file_count, "bytes": byte_count, "sha256": digest}


def _content_hash(records: list[Mapping[str, object]]) -> str:
    return canonical_sha256([{key: row[key] for key in ("name", "file_count", "bytes", "sha256")} for row in records])


def _semantic_axes(candidate: Mapping[str, object], parser: Mapping[str, object]) -> dict[str, dict[str, str]]:
    return {
        "study": {"status": "observed", "value": "CELLxGENE reward circuitry collection 22ccff12-9a18-4b01-ba10-302df57a01f8; AmyG region"},
        "biological_unit": {"status": "observed", "value": "single nucleus; exact CELLxGENE H5AD obs index"},
        "species": {"status": "mapped", "value": "Homo sapiens; NCBITaxon:9606"},
        "disease_state": {"status": "mapped", "value": "normal; PATO:0000461"},
        "tissue": {"status": "observed", "value": "amygdala"},
        "brain_region": {"status": "mapped", "value": "amygdala; UBERON:0001876"},
        "cell_type": {"status": "mapped", "value": "Cell Ontology annotations; unknown or missing labels quarantined"},
        "assay": {"status": "mapped", "value": "10x 3 prime v3; EFO:0009922"},
        "modality": {"status": "observed", "value": "transcriptomics"},
        "feature": {"status": "mapped", "value": "H5AD var index; Ensembl Gene identifiers"},
        "time": {"status": "not_applicable", "reason": "cross-sectional reference atlas has no longitudinal time axis"},
        "dose": {"status": "not_applicable", "reason": "observational reference atlas has no intervention dose"},
        "unit": {"status": "mapped", "value": str(parser["unit_mapping_version"])},
        "provenance": {"status": "observed", "value": f"{candidate['source_id']}:{candidate['source_accession']}"},
    }


def _retrieval_record(policy: Mapping[str, object], candidate: Mapping[str, object], raw_binding: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    path = _path(str(policy["retrieval_record_path"]))
    record = _load_json(path)
    if record.get("schema_version") != RETRIEVAL_RECORD_SCHEMA_VERSION or record.get("validation_status") != "passed":
        raise ValueError("CELLxGENE retrieval record is not an accepted verified record")
    if record.get("official_value_url") != candidate["official_value_url"] or record.get("raw_value_binding") != dict(raw_binding):
        raise ValueError("CELLxGENE retrieval record does not bind the configured raw asset")
    job = record.get("download_job")
    completion = record.get("download_log_completion")
    log_binding = record.get("download_log_binding")
    if not isinstance(job, Mapping) or job.get("state") != "COMPLETED" or job.get("exit_code") != "0:0":
        raise ValueError("CELLxGENE retrieval record does not bind a successful Slurm job")
    if not isinstance(completion, Mapping) or completion.get("remote_sha256") != raw_binding["sha256"] or completion.get("local_sha256") != raw_binding["sha256"]:
        raise ValueError("CELLxGENE retrieval log does not bind the raw SHA-256")
    if not isinstance(log_binding, Mapping) or _binding(_path(str(log_binding.get("path", "")))) != dict(log_binding):
        raise ValueError("CELLxGENE retrieval log binding has drifted")
    return record, _binding(path)


def _source_card(candidate: Mapping[str, object], input_bindings: list[dict[str, object]], raw_binding: Mapping[str, object], retrieval: Mapping[str, object], retrieval_binding: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": SOURCE_CARD_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "official_value_url": candidate["official_value_url"],
        "collection_id": candidate["collection_id"],
        "dataset_id": candidate["dataset_id"],
        "publication_doi": candidate["publication_doi"],
        "terms": {
            "license": "CC-BY-4.0",
            "use_scope": "public data may be downloaded shared and used with attribution to the original contributor",
            "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
            "access_class": "public_with_attribution",
            "citation_required": True,
        },
        "raw_value_binding": dict(raw_binding),
        "official_evidence_bindings": input_bindings,
        "retrieval_record": {"binding": dict(retrieval_binding), "download_job": retrieval["download_job"], "download_log_binding": retrieval["download_log_binding"]},
    }


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_cellxgene_atlas_materialization"]
    candidate = dict(policy["candidate"])
    raw_path = _path(str(candidate["raw_value_path"]))
    raw_binding = _binding(raw_path)
    if raw_binding["bytes"] != candidate["expected_raw_bytes"] or raw_binding["sha256"] != candidate["expected_raw_sha256"]:
        raise ValueError("CELLxGENE raw H5AD does not match configured release bytes and SHA-256")
    retrieval, retrieval_binding = _retrieval_record(policy, candidate, raw_binding)
    matrix_shape, encoding, _, _, obs_names, var_names = _matrix_layout(raw_path)
    samples, quarantined = _cell_tables(raw_path, obs_names, policy)
    features = _feature_table(raw_path, var_names)
    if len(features) != matrix_shape[1] or len(samples) + len(quarantined) != matrix_shape[0]:
        raise ValueError("CELLxGENE canonical axes do not reconstruct the official H5AD axes")
    qc = policy["qc"]
    donor_count = int(samples["donor_id"].nunique())
    if donor_count < int(qc["minimum_donor_count"]) or len(samples) < int(qc["minimum_cell_count"]) or len(features) < int(qc["minimum_mapped_gene_count"]):
        raise ValueError("CELLxGENE first atlas block is below a locked QC threshold")
    root = _path(str(policy["output_root"]))
    staging_root = root / ".phase1e_cellxgene_atlas_staging"
    ensure_directory(staging_root)
    selected_rows = samples["source_raw_row_index"].to_numpy(dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="cellxgene-amygdala-", dir=staging_root) as temporary:
        stage = Path(temporary)
        matrix_stats = _write_filtered_sparse_store(raw_path, stage / "sparse_counts.zarr", selected_rows=selected_rows, feature_count=len(features), chunk_size=int(policy["parser"]["sparse_chunk_size"]))
        write_parquet_atomic(samples, stage / "cell_metadata.parquet")
        write_parquet_atomic(quarantined, stage / "cell_type_quarantine.parquet")
        write_parquet_atomic(features, stage / "feature_metadata.parquet")
        write_json(stage / "matrix_manifest.json", matrix_stats)
        output_names = ("sparse_counts.zarr", "cell_metadata.parquet", "cell_type_quarantine.parquet", "feature_metadata.parquet", "matrix_manifest.json")
        records = [_record(stage / name, name) for name in output_names]
        block_sha256 = _content_hash(records)
        destination = root / "blocks" / str(candidate["source_id"]) / str(candidate["source_accession"]) / block_sha256[:16]
        input_paths = [raw_path, *[_path(str(value)) for value in candidate["release_evidence_paths"]]]
        input_bindings = [_binding(path) for path in input_paths]
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": "cellxgene_amygdala_2021_phase1e_v1",
            "content_address": f"sha256:{block_sha256}",
            "block_sha256": block_sha256,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "cns_source": True,
            "role": "qualified_for_mainline",
            "admission_allowed": True,
            "value_block_format": "zarr",
            "value_block_path": "sparse_counts.zarr",
            "sample_metadata_path": "cell_metadata.parquet",
            "cell_type_quarantine_path": "cell_type_quarantine.parquet",
            "feature_metadata_path": "feature_metadata.parquet",
            "matrix_manifest_path": "matrix_manifest.json",
            "observation_count": int(len(samples)),
            "value_row_count": int(matrix_stats["nnz"]),
            "biological_unit_count": int(samples["biological_unit_id"].nunique()),
            "effective_sample_size": float(donor_count),
            "effective_sample_size_formula": policy["biological_unit_policy"]["effective_sample_size_formula"],
            "biological_unit_policy": policy["biological_unit_policy"],
            "qc_grade": qc["grade"],
            "qc_grade_rationale": qc["grade_rationale"],
            "qc": {"cell_count": int(len(samples)), "quarantined_cell_count": int(len(quarantined)), "minimum_cell_count": qc["minimum_cell_count"], "donor_count": donor_count, "minimum_donor_count": qc["minimum_donor_count"], "mapped_gene_count": int(len(features)), "minimum_mapped_gene_count": qc["minimum_mapped_gene_count"], "finite_nonzero_values": matrix_stats["finite_nonzero_values"], "integer_like_nonzero_values": matrix_stats["integer_like_nonzero_values"], "status": "passed"},
            "mapping_confidence": "official_h5ad_obs_and_var_exact_identifier_axes_with_explicit_cell_type_quarantine",
            "mapping_version": policy["parser"]["mapping_version"],
            "unit_mapping_version": policy["parser"]["unit_mapping_version"],
            "ontology_versions": {"species": "NCBITaxon:9606", "cell_type": "Cell_Ontology", "brain_region": "UBERON:0001876", "disease": "PATO:0000461", "assay": "EFO:0009922", "gene": "Ensembl_Gene"},
            "semantic_axes": _semantic_axes(candidate, policy["parser"]),
            "split_role": "unassigned_pending_nested_split_freeze",
            "semantic_axes_complete": True,
            "source_release": {"official_url": candidate["official_value_url"], "collection_id": candidate["collection_id"], "dataset_id": candidate["dataset_id"], "dataset_version_id": candidate["source_accession"], "publication_doi": candidate["publication_doi"], "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution", "access_class": "public_with_attribution", "reuse_evidence": "Official CELLxGENE policy applies CC-BY 4.0 to public collections, allowing download, sharing, and use with attribution."},
            "parser": policy["parser"],
            "retrieval_record": dict(retrieval_binding),
            "matrix_statistics": matrix_stats,
            "input_bindings": input_bindings,
            "block_files": records,
            "blocker_codes": ["second_independent_disease_source_missing", "second_independent_atlas_source_missing", "compound_dose_time_perturbation_first_block_missing", "nested_split_freeze_pending", "real_domain_leakage_audit_pending"],
        }
        write_json(stage / "block_manifest.json", manifest)
        if destination.exists():
            existing = _load_json(destination / "block_manifest.json")
            if existing.get("block_sha256") != block_sha256:
                raise ValueError(f"Content-address collision at {destination}")
            manifest = existing
        else:
            ensure_directory(destination.parent)
            stage.replace(destination)
    manifest_path = destination / "block_manifest.json"
    ledger_path = _path(str(policy["source_ledger_path"]))
    delta_row = {"delta_version": "phase1e-source-qualification-delta-v1", "entry_id": "phase1e_cellxgene_amygdala_mainline", "source_id": candidate["source_id"], "source_accession": candidate["source_accession"], "prior_decision": "deferred", "decision": "qualified_for_mainline", "admission_allowed": "true", "scope": "source_accession_only", "frozen_ledger_path": _relative(ledger_path), "frozen_ledger_sha256": sha256_file(ledger_path), "block_manifest_path": _relative(manifest_path), "block_manifest_sha256": sha256_file(manifest_path), "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution", "access_class": "public_with_attribution", "status": "accepted"}
    delta_path = _path(str(policy["qualification_delta_path"]))
    upsert_accession_requalification_delta(delta_path, delta_row)
    source_card = _source_card(candidate, manifest["input_bindings"], raw_binding, retrieval, retrieval_binding)
    source_card_path = _path(str(policy["source_card_path"]))
    write_json(source_card_path, source_card)
    admission = {
        "schema_version": ADMISSION_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "status": "accepted_first_qualified_block",
        "candidate_id": "atlas_cellxgene_amygdala_2021",
        "candidate_role_before": "deferred_source_not_consumed",
        "candidate_role_after": "qualified_for_mainline",
        "admission_allowed": True,
        "scope": "source_accession_only",
        "source_wide_phase1a_decision_unchanged": True,
        "frozen_source_ledger": _binding(ledger_path),
        "source_qualification_delta": _binding(delta_path),
        "canonical_block_manifest": _binding(manifest_path),
        "source_card": _binding(source_card_path),
        "block_content_address": manifest["content_address"],
        "qc_grade": manifest["qc_grade"],
        "effective_sample_size": manifest["effective_sample_size"],
        "semantic_axes_complete": manifest["semantic_axes_complete"],
        "independence_status": "first_qualified_atlas_source_only; configured accession is a non-mirror primary release, but the G2 pair requirement remains unmet",
        "remaining_g2_blockers": manifest["blocker_codes"],
        "license_release_checksum_evidence": manifest["input_bindings"],
        "parser_mapping_provenance_evidence": {"parser": policy["parser"], "raw_value_origin": candidate["raw_value_path"], "source_card_path": policy["source_card_path"], "cell_type_quarantine_policy": "unknown_or_missing_cell_type_labels_are_excluded_from_canonical_mainline_matrix"},
        "independence_graph_delta": {"node": f"cellxgene::{candidate['source_accession']}", "mirror_group": candidate["mirror_group"], "relation": "official_native_primary_atlas_release", "independent_for_g2": False},
        "role_transition_justification": "Official CC-BY CELLxGENE release, checksum-verified remote retrieval, complete native observation axes, explicit unknown-cell-type quarantine, donor-level ESS, and Grade B QC support an accession-scoped mainline transition only.",
    }
    write_json(_path(str(policy["admission_audit_path"])), admission)
    return manifest, delta_row, admission


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the CELLxGENE amygdala Phase 1E canonical atlas block.")
    parser.add_argument("--output", default="Output/audits/phase1e_cellxgene_amygdala_mainline_admission.json")
    args = parser.parse_args()
    manifest, _, admission = build(REPO_ROOT)
    output = _path(args.output)
    configured = _path("Output/audits/phase1e_cellxgene_amygdala_mainline_admission.json")
    if output != configured:
        write_json(output, admission)
    print(json.dumps({"status": admission["status"], "canonical_uid": manifest["canonical_uid"], "block_sha256": manifest["block_sha256"], "effective_sample_size": manifest["effective_sample_size"]}, sort_keys=True))


if __name__ == "__main__":
    main()
