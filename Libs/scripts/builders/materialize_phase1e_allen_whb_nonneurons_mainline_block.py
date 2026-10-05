"""Materialize a native Allen WHB cell-level observation block.

The input is the official WHB-10Xv3 non-neuronal raw-count H5AD release.
It is copied into a content-addressed sparse Zarr store without densifying or
inventing zero observations.  Official cell, donor, region, taxonomy, and gene
metadata are joined by their source-native identifiers before any accession can
be re-admitted to the mainline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import (
    canonical_sha256,
    sha256_file,
)
from Libs.utils.data.public_foundation_corpus import hash_path
from Libs.utils.data.public_foundation_roles import (
    upsert_accession_requalification_delta,
)
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    load_config,
    resolve_repo_path,
    write_json,
)
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic
from Libs.utils.model.dynamics_store import (
    initialize_zarr_store,
    write_zarr_array_from_loader,
)


BUILDER_ID = "Libs/scripts/builders/materialize_phase1e_allen_whb_nonneurons_mainline_block.py"
MANIFEST_VERSION = "neuronsifter-phase1e-canonical-atlas-mainline-block-v1"
ADMISSION_VERSION = "neuronsifter-phase1e-atlas-mainline-admission-v1"
SOURCE_CARD_VERSION = "neuronsifter-phase1e-atlas-source-card-v1"
MATRIX_NAME = "WHB-10Xv3-Nonneurons"
RETRIEVAL_RECORD_SCHEMA_VERSION = "neuronsifter-phase1e-allen-whb-retrieval-record-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str) -> Path:
    return resolve_repo_path(value)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object JSON payload: {path}")
    return payload


def _pair_independence_evidence(
    policy: Mapping[str, object],
    candidate: Mapping[str, object],
    samples: pd.DataFrame,
) -> dict[str, object]:
    reference = policy["phase1f_pair_reference"]
    admission_path = _path(str(reference["admission_audit_path"]))
    admission = _load_json(admission_path)
    manifest_binding = admission.get("canonical_block_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise ValueError("Atlas M1 admission lacks a canonical manifest binding")
    manifest_path = _path(str(manifest_binding.get("path", "")))
    if _binding(manifest_path) != dict(manifest_binding):
        raise ValueError("Atlas M1 canonical manifest binding has drifted")
    manifest = _load_json(manifest_path)
    expected = {
        "canonical_uid": reference["expected_canonical_uid"],
        "source_id": reference["expected_source_id"],
        "source_accession": reference["expected_source_accession"],
        "publication_accession_group": reference["expected_publication_accession_group"],
        "mirror_group": reference["expected_mirror_group"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Atlas M1 manifest identity does not match the frozen pair reference")
    if manifest.get("role") != "qualified_for_mainline" or manifest.get("qc_grade") not in {"A", "B"}:
        raise ValueError("Atlas M1 pair reference is not a qualified Grade A/B block")
    m1_samples = pd.read_parquet(manifest_path.parent / str(manifest["sample_metadata_path"]))
    required_m1 = {"biological_unit_id", "donor_id"}
    required_m2 = {"biological_unit_id", "donor_label"}
    if required_m1 - set(m1_samples.columns) or required_m2 - set(samples.columns):
        raise ValueError("Atlas pair metadata lacks biological-unit or donor identity columns")
    raw_biological_overlap = sorted(
        set(m1_samples["biological_unit_id"].astype(str))
        & set(samples["biological_unit_id"].astype(str))
    )
    raw_donor_overlap = sorted(
        set(m1_samples["donor_id"].astype(str))
        & set(samples["donor_label"].astype(str))
    )
    different_identity_axes = {
        "source_id": manifest["source_id"] != candidate["source_id"],
        "source_accession": manifest["source_accession"] != candidate["source_accession"],
        "publication_accession_group": manifest["publication_accession_group"] != candidate["publication_accession_group"],
        "mirror_group": manifest["mirror_group"] != candidate["mirror_group"],
    }
    if not all(different_identity_axes.values()) or raw_biological_overlap or raw_donor_overlap:
        raise ValueError("Atlas M1/M2 independence contract is not satisfied")
    return {
        "status": "passed",
        "independent_for_g2": True,
        "m1_manifest": _binding(manifest_path),
        "m1_identity": expected,
        "m2_identity": {
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
        },
        "different_identity_axes": different_identity_axes,
        "raw_biological_unit_token_overlap_count": 0,
        "raw_donor_token_overlap_count": 0,
        "biological_unit_overlap_count": 0,
        "mirror_overlap_count": 0,
        "identity_scope": "source-native cell and donor identifiers compared before source namespacing",
    }


def _retrieval_record(
    policy: Mapping[str, object], candidate: Mapping[str, object], raw_binding: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    record_path = _path(str(policy["retrieval_record_path"]))
    record = _load_json(record_path)
    if record.get("schema_version") != RETRIEVAL_RECORD_SCHEMA_VERSION:
        raise ValueError("WHB retrieval record schema version is not accepted")
    if record.get("validation_status") != "passed":
        raise ValueError("WHB retrieval record does not certify a successful official download")
    if record.get("official_value_url") != candidate["official_value_url"]:
        raise ValueError("WHB retrieval record official URL does not equal configured source URL")
    if record.get("expected_raw_bytes") != candidate["expected_raw_bytes"]:
        raise ValueError("WHB retrieval record expected byte count does not equal configured contract")
    if record.get("raw_value_binding") != dict(raw_binding):
        raise ValueError("WHB retrieval record raw checksum binding does not equal the materializer input")
    job = record.get("download_job")
    log_binding = record.get("download_log_binding")
    completion = record.get("download_log_completion")
    if not isinstance(job, Mapping) or job.get("state") != "COMPLETED" or job.get("exit_code") != "0:0":
        raise ValueError("WHB retrieval record does not bind a successful terminal Slurm job")
    if not isinstance(log_binding, Mapping) or not isinstance(completion, Mapping):
        raise ValueError("WHB retrieval record lacks a bound completion log")
    log_path = _path(str(log_binding.get("path", "")))
    if _binding(log_path) != dict(log_binding):
        raise ValueError("WHB retrieval log binding has drifted")
    if completion.get("sha256") != raw_binding["sha256"] or int(str(completion.get("bytes", "-1"))) != raw_binding["bytes"]:
        raise ValueError("WHB retrieval completion log does not bind current raw bytes")
    return record, _binding(record_path)


def _decode(values: Iterable[object]) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _metadata_paths(candidate: Mapping[str, object]) -> dict[str, Path]:
    release_paths = [_path(str(value)) for value in candidate["release_evidence_paths"]]
    by_name = {path.name: path for path in release_paths}
    required = {
        "cell_metadata.csv",
        "donor.csv",
        "gene.csv",
        "region_of_interest_structure_map.csv",
        "cluster.csv",
        "cluster_annotation_term.csv",
    }
    missing = required - set(by_name)
    if missing:
        raise ValueError("Missing required WHB metadata evidence: " + ",".join(sorted(missing)))
    return by_name


def _load_cells(cell_path: Path, obs_names: list[str]) -> pd.DataFrame:
    required_columns = [
        "cell_label",
        "barcoded_cell_sample_label",
        "feature_matrix_label",
        "donor_label",
        "region_of_interest_label",
        "anatomical_division_label",
        "cluster_alias",
        "library_label",
        "brain_section_label",
        "entity",
        "library_method",
    ]
    pieces: list[pd.DataFrame] = []
    for frame in pd.read_csv(
        cell_path,
        usecols=required_columns,
        dtype=str,
        chunksize=250_000,
        keep_default_na=False,
    ):
        selected = frame.loc[frame["feature_matrix_label"] == MATRIX_NAME].copy()
        if not selected.empty:
            pieces.append(selected)
    if not pieces:
        raise ValueError("Official WHB cell metadata has no non-neuronal cell rows")
    cells = pd.concat(pieces, ignore_index=True)
    if cells["cell_label"].duplicated().any():
        raise ValueError("Official WHB cell labels are not unique")
    source_ids = set(cells["cell_label"])
    h5ad_ids = set(obs_names)
    if source_ids != h5ad_ids:
        missing_in_metadata = len(h5ad_ids - source_ids)
        missing_in_h5ad = len(source_ids - h5ad_ids)
        raise ValueError(
            "H5AD obs-name to official cell-label join is incomplete: "
            f"missing_in_metadata={missing_in_metadata}, missing_in_h5ad={missing_in_h5ad}"
        )
    return cells.set_index("cell_label", verify_integrity=True).loc[obs_names].reset_index()


def _load_gene_metadata(gene_path: Path, var_names: list[str]) -> pd.DataFrame:
    genes = pd.read_csv(gene_path, dtype=str, keep_default_na=False)
    required = {"gene_identifier", "gene_symbol", "biotype", "name"}
    if missing := required - set(genes.columns):
        raise ValueError("Official WHB gene metadata is missing columns: " + ",".join(sorted(missing)))
    if genes["gene_identifier"].duplicated().any():
        raise ValueError("Official WHB gene identifiers are not unique")
    source_ids = set(genes["gene_identifier"])
    h5ad_ids = set(var_names)
    if source_ids != h5ad_ids:
        missing_in_gene_csv = len(h5ad_ids - source_ids)
        missing_in_h5ad = len(source_ids - h5ad_ids)
        raise ValueError(
            "H5AD var-name to official gene-identifier join is incomplete: "
            f"missing_in_gene_csv={missing_in_gene_csv}, missing_in_h5ad={missing_in_h5ad}"
        )
    return genes.set_index("gene_identifier", verify_integrity=True).loc[var_names].reset_index()


def _load_taxonomy(metadata: Mapping[str, Path]) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, str]]]:
    clusters = pd.read_csv(metadata["cluster.csv"], dtype=str, keep_default_na=False)
    if {"cluster_alias", "label"} - set(clusters.columns):
        raise ValueError("WHB cluster metadata lacks cluster_alias/label")
    cluster_labels = dict(zip(clusters["cluster_alias"], clusters["label"], strict=True))
    terms = pd.read_csv(metadata["cluster_annotation_term.csv"], dtype=str, keep_default_na=False)
    if {"label", "name", "cluster_annotation_term_set_label"} - set(terms.columns):
        raise ValueError("WHB taxonomy terms lack required columns")
    cluster_names: dict[str, str] = {}
    for row in terms.itertuples(index=False):
        if str(row.cluster_annotation_term_set_label).endswith("_CLUS"):
            cluster_names.setdefault(str(row.label), str(row.name))
    regions = pd.read_csv(metadata["region_of_interest_structure_map.csv"], dtype=str, keep_default_na=False)
    if {"region_of_interest_label", "structure_identifier", "structure_name"} - set(regions.columns):
        raise ValueError("WHB region map lacks required columns")
    region_map = {
        str(row.region_of_interest_label): (str(row.structure_identifier), str(row.structure_name))
        for row in regions.itertuples(index=False)
        if str(row.region_of_interest_label)
    }
    return cluster_labels, cluster_names, region_map


def _sample_metadata(
    cells: pd.DataFrame,
    donor_path: Path,
    cluster_labels: Mapping[str, str],
    cluster_names: Mapping[str, str],
    region_map: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    donors = pd.read_csv(donor_path, dtype=str, keep_default_na=False)
    required_donor_columns = {"donor_label", "donor_age", "donor_sex", "donor_species"}
    if missing := required_donor_columns - set(donors.columns):
        raise ValueError("WHB donor metadata is missing columns: " + ",".join(sorted(missing)))
    if donors["donor_label"].duplicated().any():
        raise ValueError("WHB donor labels are not unique")
    donor_lookup = donors.set_index("donor_label", verify_integrity=True)
    if not set(cells["donor_label"]).issubset(set(donor_lookup.index)):
        raise ValueError("Official cell metadata contains donors absent from donor.csv")
    missing_clusters = sorted(set(cells["cluster_alias"]) - set(cluster_labels))
    if missing_clusters:
        raise ValueError(
            "Official cell metadata contains cluster aliases absent from WHB taxonomy: "
            + ",".join(missing_clusters[:10])
        )
    missing_regions = sorted(set(cells["region_of_interest_label"]) - set(region_map))
    if missing_regions:
        raise ValueError(
            "Official cell metadata contains regions absent from WHB region map: "
            + ",".join(missing_regions[:10])
        )

    rows: list[dict[str, object]] = []
    for row in cells.itertuples(index=False):
        donor = donor_lookup.loc[str(row.donor_label)]
        raw_cluster = str(row.cluster_alias)
        cell_type_id = cluster_labels[raw_cluster]
        cell_type_label = cluster_names.get(cell_type_id, cell_type_id)
        raw_region = str(row.region_of_interest_label)
        region_id, region_label = region_map[raw_region]
        rows.append(
            {
                "biological_unit_id": str(row.cell_label),
                "source_cell_label": str(row.cell_label),
                "source_sample_label": str(row.barcoded_cell_sample_label) if hasattr(row, "barcoded_cell_sample_label") else str(row.cell_label),
                "donor_label": str(row.donor_label),
                "donor_age": str(donor["donor_age"]),
                "donor_sex": str(donor["donor_sex"]),
                "species_taxonomy_id": str(donor["donor_species"]),
                "feature_matrix_label": str(row.feature_matrix_label),
                "cell_type_id": cell_type_id,
                "cell_type_label": cell_type_label,
                "cell_type_namespace": "Allen_WHB_CCN202210140",
                "brain_region_id": region_id,
                "brain_region_label": region_label,
                "brain_region_namespace": "DHBA_or_Allen_WHB_region_of_interest",
                "region_of_interest_label": raw_region,
                "anatomical_division_label": str(row.anatomical_division_label),
                "source_cluster_alias": raw_cluster,
                "source_library_label": str(row.library_label),
                "source_brain_section_label": str(row.brain_section_label),
                "source_entity": str(row.entity),
                "assay": "10x Genomics 3 prime v3 single-nucleus RNA sequencing",
                "modality": "transcriptomics",
                "disease_state_status": "not_applicable_reference_atlas_without_disease_stratification",
                "time_status": "not_applicable_cross_sectional_reference_atlas",
                "dose_status": "not_applicable_observational_reference_atlas",
                "technical_replicate_group": str(row.cell_label),
            }
        )
    metadata = pd.DataFrame(rows)
    if metadata["biological_unit_id"].duplicated().any() or metadata["donor_label"].eq("").any():
        raise ValueError("Canonical WHB cell metadata has invalid biological-unit or donor identities")
    return metadata


def _feature_metadata(genes: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "feature_index": np.arange(len(genes), dtype=np.int64),
            "canonical_feature_id": genes["gene_identifier"].astype(str),
            "canonical_gene_symbol": genes["gene_symbol"].astype(str),
            "canonical_namespace": "Ensembl_Gene",
            "source_gene_identifier": genes["gene_identifier"].astype(str),
            "source_gene_symbol": genes["gene_symbol"].astype(str),
            "biotype": genes["biotype"].astype(str),
            "feature_name": genes["name"].astype(str),
            "mapping_method": "official_allen_whb_gene_csv_exact_identifier_join",
            "mapping_confidence": "exact",
        }
    )
    if output["canonical_feature_id"].duplicated().any() or output["canonical_feature_id"].eq("").any():
        raise ValueError("Canonical WHB feature identifiers are invalid")
    return output


def _h5ad_dataframe_index(group: h5py.Group, axis_name: str) -> list[str]:
    """Read the AnnData-declared dataframe index without assuming a legacy path."""

    index_name = group.attrs.get("_index")
    if isinstance(index_name, bytes):
        index_name = index_name.decode("utf-8")
    if not isinstance(index_name, str) or not index_name:
        raise ValueError(f"WHB H5AD {axis_name} dataframe lacks a declared _index column")
    index = group.get(index_name)
    if not isinstance(index, h5py.Dataset):
        raise ValueError(f"WHB H5AD {axis_name} index column is not a dataset: {index_name!r}")
    encoding = index.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if str(encoding) != "string-array":
        raise ValueError(
            f"WHB H5AD {axis_name} index column has unsupported encoding: {encoding!r}"
        )
    return _decode(index[:])


def _matrix_layout(raw_path: Path) -> tuple[tuple[int, int], str, str, str, list[str], list[str]]:
    with h5py.File(raw_path, "r") as handle:
        if "X" not in handle or not isinstance(handle["X"], h5py.Group):
            raise ValueError("WHB H5AD lacks a sparse X group")
        x_group = handle["X"]
        encoding = x_group.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if str(encoding) not in {"csr_matrix", "csc_matrix"}:
            raise ValueError(f"WHB H5AD X encoding is not a supported sparse matrix: {encoding!r}")
        if not {"data", "indices", "indptr"}.issubset(x_group.keys()):
            raise ValueError("WHB H5AD sparse X group lacks data/indices/indptr")
        shape_raw = x_group.attrs.get("shape")
        if shape_raw is None:
            raise ValueError("WHB H5AD sparse X group lacks shape")
        shape = tuple(int(value) for value in shape_raw)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"WHB H5AD sparse X shape is invalid: {shape}")
        obs_group = handle.get("obs")
        var_group = handle.get("var")
        if not isinstance(obs_group, h5py.Group) or not isinstance(var_group, h5py.Group):
            raise ValueError("WHB H5AD lacks AnnData obs or var dataframe groups")
        obs_names = _h5ad_dataframe_index(obs_group, "obs")
        var_names = _h5ad_dataframe_index(var_group, "var")
        if len(obs_names) != shape[0] or len(var_names) != shape[1]:
            raise ValueError("WHB H5AD observation or feature axis does not match X shape")
        if len(set(obs_names)) != len(obs_names) or len(set(var_names)) != len(var_names):
            raise ValueError("WHB H5AD observation or feature identifiers are not unique")
        return (
            shape,
            str(encoding),
            str(x_group["data"].dtype),
            str(x_group["indices"].dtype),
            obs_names,
            var_names,
        )


def _write_sparse_store(
    raw_path: Path,
    destination: Path,
    *,
    chunk_size: int,
    matrix_shape: tuple[int, int],
    encoding: str,
    source_value_kind: str = "official_allen_whb_raw_umi_count",
) -> dict[str, object]:
    with h5py.File(raw_path, "r") as handle:
        x_group = handle["X"]
        data = x_group["data"]
        indices = x_group["indices"]
        indptr = x_group["indptr"]
        if int(indptr[-1]) != len(data) or len(indices) != len(data):
            raise ValueError("WHB sparse matrix data/indices/indptr dimensions are inconsistent")
        expected_indptr = matrix_shape[0] + 1 if encoding == "csr_matrix" else matrix_shape[1] + 1
        if len(indptr) != expected_indptr:
            raise ValueError("WHB sparse indptr length does not match matrix orientation")
        pointer = np.asarray(indptr[:])
        if pointer[0] != 0 or np.any(np.diff(pointer) < 0):
            raise ValueError("WHB sparse indptr is not a monotonic zero-based pointer")
        index_limit = matrix_shape[1] if encoding == "csr_matrix" else matrix_shape[0]
        initialize_zarr_store(
            destination,
            {
                "format_name": "neuronsifter_sparse_csr_or_csc_zarr_v1",
                "matrix_encoding": encoding,
                "matrix_shape": list(matrix_shape),
                "source_value_kind": source_value_kind,
            },
        )

        def copy_dataset(name: str, dataset: h5py.Dataset, attrs: dict[str, object]) -> None:
            write_zarr_array_from_loader(
                destination,
                name,
                (len(dataset),),
                (min(chunk_size, max(len(dataset), 1)),),
                dataset.dtype,
                lambda slices: dataset[slices[0]],
                attrs=attrs,
            )

        copy_dataset("data", data, {"axis_names": ["nonzero_entry"]})
        copy_dataset("indices", indices, {"axis_names": ["nonzero_entry"]})
        copy_dataset("indptr", indptr, {"axis_names": ["compressed_axis_pointer"]})
        finite = True
        minimum: float | int | None = None
        maximum: float | int | None = None
        for start in range(0, len(data), chunk_size):
            stop = min(start + chunk_size, len(data))
            chunk = np.asarray(data[start:stop])
            index_chunk = np.asarray(indices[start:stop])
            if not np.isfinite(chunk).all():
                finite = False
                break
            if len(index_chunk) and (index_chunk.min() < 0 or index_chunk.max() >= index_limit):
                raise ValueError("WHB sparse indices are outside the declared matrix shape")
            chunk_min = chunk.min().item() if len(chunk) else None
            chunk_max = chunk.max().item() if len(chunk) else None
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        if not finite or minimum is None or minimum < 0:
            raise ValueError("WHB raw UMI sparse values are not finite non-negative counts")
        return {
            "matrix_shape": list(matrix_shape),
            "matrix_encoding": encoding,
            "nnz": int(len(data)),
            "data_dtype": str(data.dtype),
            "indices_dtype": str(indices.dtype),
            "indptr_dtype": str(indptr.dtype),
            "finite_nonzero_values": finite,
            "minimum_nonzero_value": minimum,
            "maximum_nonzero_value": maximum,
            "value_unit_original": "raw UMI count",
            "value_unit_canonical": "raw UMI count",
            "unit_conversion_formula": "identity",
        }


def _record(path: Path, name: str) -> dict[str, object]:
    file_count, byte_count, digest = hash_path(path)
    return {
        "name": name,
        "relative_path": name,
        "file_count": file_count,
        "bytes": byte_count,
        "sha256": digest,
    }


def _content_hash(records: list[Mapping[str, object]]) -> str:
    return canonical_sha256(
        [
            {
                "name": row["name"],
                "file_count": row["file_count"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in records
        ]
    )


def _semantic_axes(candidate: Mapping[str, object], parser: Mapping[str, object]) -> dict[str, dict[str, str]]:
    return {
        "study": {"status": "observed", "value": "Allen Brain Cell Atlas WHB-10Xv3"},
        "biological_unit": {"status": "observed", "value": "cell; exact official cell_label identity"},
        "species": {"status": "mapped", "value": "Homo sapiens; NCBITaxon:9606"},
        "disease_state": {"status": "not_applicable", "reason": "reference atlas has no disease-state stratification"},
        "tissue": {"status": "observed", "value": "human whole brain nuclei"},
        "brain_region": {"status": "mapped", "value": "official WHB region_of_interest mapped to DHBA where available"},
        "cell_type": {"status": "mapped", "value": "official WHB CCN202210140 cluster taxonomy"},
        "assay": {"status": "observed", "value": "10x Genomics 3 prime v3 single-nucleus RNA sequencing"},
        "modality": {"status": "observed", "value": "transcriptomics"},
        "feature": {"status": "mapped", "value": "official WHB Ensembl gene identifier"},
        "time": {"status": "not_applicable", "reason": "cross-sectional reference atlas has no longitudinal time axis"},
        "dose": {"status": "not_applicable", "reason": "observational reference atlas has no intervention dose"},
        "unit": {"status": "mapped", "value": str(parser["unit_mapping_version"])},
        "provenance": {"status": "observed", "value": f"{candidate['source_id']}:{candidate['source_accession']}"},
    }


def _source_card(
    candidate: Mapping[str, object],
    input_bindings: list[dict[str, object]],
    raw_binding: Mapping[str, object],
    retrieval_record: Mapping[str, object],
    retrieval_record_binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SOURCE_CARD_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "official_value_url": candidate["official_value_url"],
        "expression_release_version": candidate["expression_release_version"],
        "metadata_release_version": candidate["metadata_release_version"],
        "terms": {
            "use_scope": "research_or_other_noncommercial_purposes",
            "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
            "access_class": "public_with_attribution",
            "citation_required": True,
            "redistribution_of_raw_assets_allowed": False,
        },
        "raw_value_binding": dict(raw_binding),
        "official_evidence_bindings": input_bindings,
        "retrieval_record": {
            "binding": dict(retrieval_record_binding),
            "download_job": retrieval_record["download_job"],
            "download_log_binding": retrieval_record["download_log_binding"],
            "retrieval_verification": "successful_slurm_job_and_completion_log_bind_current_official_raw_sha256",
        },
    }


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_atlas_materialization"]
    candidate = dict(policy["candidate"])
    raw_path = _path(str(candidate["raw_value_path"]))
    if not raw_path.is_file():
        raise FileNotFoundError(f"Official WHB H5AD has not been materialized: {raw_path}")
    raw_binding = _binding(raw_path)
    if raw_binding["bytes"] != candidate["expected_raw_bytes"]:
        raise ValueError("Official WHB H5AD byte count does not equal configured release contract")
    retrieval_record, retrieval_record_binding = _retrieval_record(policy, candidate, raw_binding)
    metadata_paths = _metadata_paths(candidate)
    matrix_shape, encoding, _, _, obs_names, var_names = _matrix_layout(raw_path)
    cells = _load_cells(metadata_paths["cell_metadata.csv"], obs_names)
    genes = _load_gene_metadata(metadata_paths["gene.csv"], var_names)
    cluster_labels, cluster_names, region_map = _load_taxonomy(metadata_paths)
    samples = _sample_metadata(
        cells,
        metadata_paths["donor.csv"],
        cluster_labels,
        cluster_names,
        region_map,
    )
    features = _feature_metadata(genes)
    if len(samples) != matrix_shape[0] or len(features) != matrix_shape[1]:
        raise ValueError("Canonical WHB metadata axes do not match official raw matrix shape")
    qc = policy["qc"]
    donor_count = int(samples["donor_label"].nunique())
    if donor_count < int(qc["minimum_donor_count"]):
        raise ValueError("WHB donor ESS is below configured minimum")
    if donor_count != int(qc["expected_donor_count"]):
        raise ValueError("WHB donor ESS does not equal the official matrix-specific expectation")
    if len(samples) < int(qc["minimum_cell_count"]):
        raise ValueError("WHB cell count is below configured minimum")
    if len(samples) != int(qc["expected_cell_count"]):
        raise ValueError("WHB cell count does not equal the official matrix-specific expectation")
    if len(features) < int(qc["minimum_mapped_gene_count"]):
        raise ValueError("WHB mapped gene count is below configured minimum")
    if len(features) != int(qc["expected_mapped_gene_count"]):
        raise ValueError("WHB gene count does not equal the official matrix-specific expectation")
    independence_evidence = _pair_independence_evidence(policy, candidate, samples)

    root = _path(str(policy["output_root"]))
    staging_root = root / ".phase1e_atlas_staging"
    ensure_directory(staging_root)
    with tempfile.TemporaryDirectory(prefix="allen-whb-", dir=staging_root) as temporary:
        stage = Path(temporary)
        sparse_store = stage / "sparse_counts.zarr"
        matrix_stats = _write_sparse_store(
            raw_path,
            sparse_store,
            chunk_size=int(policy["parser"]["sparse_chunk_size"]),
            matrix_shape=matrix_shape,
            encoding=encoding,
        )
        write_parquet_atomic(samples, stage / "cell_metadata.parquet")
        write_parquet_atomic(features, stage / "feature_metadata.parquet")
        write_json(stage / "matrix_manifest.json", matrix_stats)
        output_names = (
            "sparse_counts.zarr",
            "cell_metadata.parquet",
            "feature_metadata.parquet",
            "matrix_manifest.json",
        )
        records = [_record(stage / name, name) for name in output_names]
        block_sha256 = _content_hash(records)
        destination = root / "blocks" / str(candidate["source_id"]) / str(candidate["source_accession"]) / block_sha256[:16]
        input_paths = [raw_path, *[_path(str(value)) for value in candidate["release_evidence_paths"]]]
        input_bindings: list[dict[str, object]] = []
        seen_paths: set[Path] = set()
        for path in input_paths:
            if path not in seen_paths:
                input_bindings.append(_binding(path))
                seen_paths.add(path)
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": "whb_allen_nonneurons_20240330_phase1e_v1",
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
            "qc": {
                "cell_count": int(len(samples)),
                "minimum_cell_count": qc["minimum_cell_count"],
                "donor_count": donor_count,
                "minimum_donor_count": qc["minimum_donor_count"],
                "expected_donor_count": qc["expected_donor_count"],
                "mapped_gene_count": int(len(features)),
                "minimum_mapped_gene_count": qc["minimum_mapped_gene_count"],
                "expected_mapped_gene_count": qc["expected_mapped_gene_count"],
                "expected_cell_count": qc["expected_cell_count"],
                "finite_nonzero_values": matrix_stats["finite_nonzero_values"],
                "nonnegative_raw_count_values": matrix_stats["minimum_nonzero_value"] >= 0,
                "status": "passed",
            },
            "mapping_confidence": "official_cell_donor_region_taxonomy_and_ensembl_identifier_exact_joins",
            "mapping_version": policy["parser"]["mapping_version"],
            "unit_mapping_version": policy["parser"]["unit_mapping_version"],
            "ontology_versions": {
                "species": "NCBITaxon:9606",
                "cell_type": "Allen_WHB_CCN202210140_20240330",
                "brain_region": "DHBA_20241115_or_Allen_WHB_source_native",
                "gene": "Ensembl_Gene_20241115",
            },
            "semantic_axes": _semantic_axes(candidate, policy["parser"]),
            "split_role": "unassigned_pending_nested_split_freeze",
            "semantic_axes_complete": True,
            "phase1f_pair_independence": independence_evidence,
            "source_release": {
                "official_url": candidate["official_value_url"],
                "expression_release_version": candidate["expression_release_version"],
                "metadata_release_version": candidate["metadata_release_version"],
                "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
                "access_class": "public_with_attribution",
                "reuse_evidence": "Allen Terms permit use, copy, and derivative works for research or other noncommercial purposes with attribution.",
            },
            "parser": policy["parser"],
            "retrieval_record": dict(retrieval_record_binding),
            "matrix_statistics": matrix_stats,
            "input_bindings": input_bindings,
            "block_files": records,
            "blocker_codes": [
                "nested_split_freeze_pending",
                "real_domain_leakage_audit_pending",
            ],
        }
        write_json(stage / "block_manifest.json", manifest)
        if destination.exists():
            existing_path = destination / "block_manifest.json"
            if not existing_path.is_file():
                raise ValueError(f"Content-address destination is incomplete: {destination}")
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("block_sha256") != block_sha256:
                raise ValueError(f"Content-address collision at {destination}")
            manifest = existing
        else:
            ensure_directory(destination.parent)
            stage.replace(destination)

    manifest_path = destination / "block_manifest.json"
    ledger_path = _path(str(policy["source_ledger_path"]))
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    delta_row = {
        "delta_version": "phase1e-source-qualification-delta-v1",
        "entry_id": "phase1e_allen_whb_nonneurons_mainline",
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "prior_decision": "qualified_as_context_prior",
        "decision": "qualified_for_mainline",
        "admission_allowed": "true",
        "scope": "source_accession_only",
        "frozen_ledger_path": _relative(ledger_path),
        "frozen_ledger_sha256": sha256_file(ledger_path),
        "block_manifest_path": _relative(manifest_path),
        "block_manifest_sha256": sha256_file(manifest_path),
        "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
        "access_class": "public_with_attribution",
        "status": "accepted",
    }
    delta_path = _path(str(policy["qualification_delta_path"]))
    upsert_accession_requalification_delta(delta_path, delta_row)
    source_card = _source_card(
        candidate,
        manifest["input_bindings"],
        raw_binding,
        retrieval_record,
        retrieval_record_binding,
    )
    write_json(_path(str(policy["source_card_path"])), source_card)
    admission = {
        "schema_version": ADMISSION_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "status": "accepted_second_independent_block",
        "candidate_id": "atlas_allen_whb_nonneurons_20240330",
        "candidate_role_before": "qualified_as_context_prior_source",
        "candidate_role_after": "qualified_for_mainline",
        "admission_allowed": True,
        "scope": "source_accession_only",
        "source_wide_phase1a_decision_unchanged": True,
        "frozen_source_ledger": _binding(ledger_path),
        "source_qualification_delta": _binding(delta_path),
        "canonical_block_manifest": _binding(manifest_path),
        "source_card": _binding(_path(str(policy["source_card_path"]))),
        "block_content_address": manifest["content_address"],
        "qc_grade": manifest["qc_grade"],
        "effective_sample_size": manifest["effective_sample_size"],
        "semantic_axes_complete": manifest["semantic_axes_complete"],
        "independence_status": "atlas_independent_source_pair_closed",
        "phase1f_pair_independence": manifest["phase1f_pair_independence"],
        "remaining_g2_blockers": [
            "second_independent_disease_source_missing",
            "second_independent_perturbation_source_missing",
            "g2_action_catalog_not_frozen",
            "nested_split_freeze_pending",
            "real_domain_leakage_audit_pending",
        ],
        "license_release_checksum_evidence": manifest["input_bindings"],
        "parser_mapping_provenance_evidence": {
            "parser": policy["parser"],
            "raw_value_origin": candidate["raw_value_path"],
            "source_card_path": policy["source_card_path"],
        },
        "independence_graph_delta": {
            "node": f"allen::{candidate['source_accession']}",
            "mirror_group": candidate["mirror_group"],
            "relation": "independent_official_native_atlas_release",
            "paired_with": manifest["phase1f_pair_independence"]["m1_identity"],
            "independent_for_g2": True,
        },
        "role_transition_justification": "Official Allen raw counts, exact metadata joins, donor-level ESS, source-card terms evidence, Grade B QC, and zero M1/M2 identity overlap support an accession-scoped mainline transition and close the atlas independent-source pair.",
    }
    write_json(_path(str(policy["admission_audit_path"])), admission)
    return manifest, delta_row, admission


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the Allen WHB Phase 1E canonical atlas block.")
    parser.add_argument("--output", default="Output/audits/phase1e_allen_whb_nonneurons_mainline_admission.json")
    args = parser.parse_args()
    manifest, _, admission = build(REPO_ROOT)
    output = _path(args.output)
    configured = _path("Output/audits/phase1e_allen_whb_nonneurons_mainline_admission.json")
    if output != configured:
        write_json(output, admission)
    print(json.dumps({
        "status": admission["status"],
        "canonical_uid": manifest["canonical_uid"],
        "block_sha256": manifest["block_sha256"],
        "effective_sample_size": manifest["effective_sample_size"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
