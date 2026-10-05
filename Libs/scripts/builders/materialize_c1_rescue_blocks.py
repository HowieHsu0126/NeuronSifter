"""Materialize the two real-value blocks admitted by the C1 rescue V2 plan."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.materialize_phase1f_perturbation_pair_block import (  # noqa: E402
    _canonicalize as canonicalize_e_mtab,
    _read_adf as read_e_mtab_adf,
    _read_assay as read_e_mtab_assay,
    _read_sdrf as read_e_mtab_sdrf,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    ensure_directory,
    load_config,
    write_json,
)
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic  # noqa: E402
from Libs.utils.model.dynamics_store import (  # noqa: E402
    initialize_zarr_store,
    write_zarr_array,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/data/public_foundation_corpus_rescue.yaml"
GSE_CANONICAL_UID = "pfc_geo_gse262992_dex_control_rescue"
CONTROL_CANONICAL_UID = "pfc_arrayexpress_e_mtab_6772_controls_rescue"
BUILDER_ID = "Libs/scripts/builders/materialize_c1_rescue_blocks.py"
ENSEMBL_MOUSE_GENE = re.compile(r"^ENSMUSG[0-9]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_policy() -> Mapping[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = payload.get("public_foundation_corpus_rescue")
    if not isinstance(policy, Mapping):
        raise ValueError("V2 rescue config is malformed")
    return policy


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
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


def _write_content_addressed_block(
    stage: Path,
    output_root: Path,
    *,
    source_id: str,
    accession_path: str,
    output_names: tuple[str, ...],
    manifest: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    records = [_record(stage / name, name) for name in output_names]
    block_hash = _content_hash(records)
    manifest["block_sha256"] = block_hash
    manifest["content_address"] = f"sha256:{block_hash}"
    manifest["block_files"] = records
    write_json(stage / "block_manifest.json", manifest)
    destination = output_root / "blocks" / source_id / accession_path / block_hash[:16]
    if destination.exists():
        existing = _load_json(destination / "block_manifest.json")
        if existing.get("block_sha256") != block_hash:
            raise ValueError(f"Content-address collision: {destination}")
        shutil.rmtree(stage)
        return destination, existing
    ensure_directory(destination.parent)
    shutil.move(str(stage), str(destination))
    return destination, manifest


def _series_matrix_metadata(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    rows: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    accessions: list[str] = []
    titles: list[str] = []
    treatments: list[str] = []
    supplementary: list[str] = []
    series: dict[str, str] = {}
    for row in rows:
        if not row:
            continue
        key = row[0]
        values = [value.strip() for value in row[1:]]
        if key == "!Sample_geo_accession":
            accessions = values
        elif key == "!Sample_title":
            titles = values
        elif key == "!Sample_characteristics_ch1" and values and all(
            value.lower().startswith("treatment:") for value in values
        ):
            treatments = [value.split(":", 1)[1].strip() for value in values]
        elif key == "!Sample_supplementary_file_1":
            supplementary = [PurePosixPath(value).name for value in values]
        elif key.startswith("!Series_") and len(values) == 1:
            series[key.removeprefix("!Series_")] = values[0]
    lengths = {len(accessions), len(titles), len(treatments), len(supplementary)}
    if lengths != {12} or len(set(accessions)) != 12:
        raise ValueError("GSE262992 sample metadata is incomplete or duplicated")
    records = {
        accession: {
            "sample_accession": accession,
            "title": title,
            "treatment": treatment,
            "supplementary_file": filename,
        }
        for accession, title, treatment, filename in zip(
            accessions, titles, treatments, supplementary, strict=True
        )
    }
    return records, series


def _safe_tar_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if member.isdir() or path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError(f"Unsafe or unexpected GSE262992 archive member: {member.name}")
        if not member.isfile() or not member.name.endswith("_FPKM.txt.gz"):
            raise ValueError(f"Unexpected GSE262992 archive member type: {member.name}")
        if member.name in members:
            raise ValueError(f"Duplicate GSE262992 archive member: {member.name}")
        members[member.name] = member
    if len(members) != 12:
        raise ValueError("GSE262992 raw archive does not contain exactly 12 FPKM tables")
    return members


def _read_gse_expression(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> tuple[pd.Series, dict[str, str], dict[str, int]]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Unable to read archive member: {member.name}")
    with gzip.GzipFile(fileobj=extracted) as compressed:
        frame = pd.read_csv(
            io.BytesIO(compressed.read()),
            sep="\t",
            dtype={"gene_id": str},
            low_memory=False,
        )
    required = {
        "t_id",
        "gene_id",
        "gene_name",
        "FPKM",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GSE262992 FPKM table lacks columns {sorted(missing)}: {member.name}")
    values = pd.to_numeric(frame["FPKM"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"GSE262992 FPKM values are non-finite or negative: {member.name}")
    if frame["t_id"].astype(str).duplicated().any():
        raise ValueError(f"GSE262992 transcript IDs are duplicated: {member.name}")
    genes = frame["gene_id"].astype(str).str.split(".", n=1, regex=False).str[0]
    eligible = genes.str.fullmatch(ENSEMBL_MOUSE_GENE.pattern)
    eligible_frame = pd.DataFrame(
        {
            "gene_id": genes[eligible].to_numpy(),
            "gene_name": frame.loc[eligible, "gene_name"].fillna("").astype(str).to_numpy(),
            "FPKM": values[eligible.to_numpy()],
        }
    )
    grouped = eligible_frame.groupby("gene_id", sort=True, observed=True)
    expression = grouped["FPKM"].sum()
    symbols = grouped["gene_name"].agg(
        lambda values: "|".join(sorted({value.strip() for value in values if value.strip()}))
    )
    diagnostics = {
        "source_transcript_count": int(len(frame)),
        "exact_ensembl_transcript_count": int(eligible.sum()),
        "quarantined_transcript_count": int((~eligible).sum()),
        "canonical_gene_count": int(len(expression)),
    }
    return expression.astype(np.float64), symbols.to_dict(), diagnostics


def _gse_condition(accession: str, candidate: Mapping[str, object]) -> str:
    admitted = candidate["admitted_conditions"]
    assert isinstance(admitted, Mapping)
    for condition, specification in admitted.items():
        assert isinstance(specification, Mapping)
        if accession in specification["sample_accessions"]:
            return str(condition)
    if accession in {"GSM8182845", "GSM8182846", "GSM8182847"}:
        return "bmp4_only"
    if accession in {"GSM8182848", "GSM8182849", "GSM8182850"}:
        return "dexamethasone_plus_bmp4"
    raise ValueError(f"GSE262992 sample is outside the frozen condition map: {accession}")


def _gse_observations(candidate: Mapping[str, object]) -> pd.DataFrame:
    action_uid = "geo_action:GSE262992:dexamethasone_1uM_72h"
    common = {
        "source_uid": "geo",
        "publication_uid": "geo:GSE262992",
        "accession_uid": "GSE262992",
        "subject_or_donor_uid": "explicit_unavailable_condition_aggregate",
        "cell_or_tissue_context": "mouse_hippocampal_neural_stem_cell_model",
        "cns_context_class": candidate["cns_context_class"],
        "species": "Mus musculus",
        "species_taxonomy_id": "NCBITaxon:10090",
        "tissue": "hippocampal_neural_stem_cell_model",
        "brain_region_ontology_id": candidate["region_ontology_id"],
        "disease_state": "not_applicable_reference_cell_model",
        "genotype_class": "wild_type",
        "assay_uid": "geo_assay:GPL24247_rna_seq",
        "modality": "transcriptomics",
        "feature_namespace": "Ensembl_Gene",
        "value_unit": "FPKM_condition_replicate_median",
        "technical_replicate_status": "three_source_declared_replicates_aggregated",
        "qc_grade": candidate["qc_grade_candidate"],
    }
    rows: list[dict[str, object]] = []
    for index, condition in enumerate(("control", "dexamethasone")):
        specification = candidate["admitted_conditions"][condition]
        accessions = [str(value) for value in specification["sample_accessions"]]
        is_action = condition == "dexamethasone"
        rows.append(
            {
                **common,
                "source_sample_name": "|".join(accessions),
                "sample_uid": f"geo:GSE262992:condition:{condition}",
                "source_sample_accessions": "|".join(accessions),
                "observation_uid": f"geo_observation:GSE262992:{condition}",
                "biological_unit_uid": f"geo:GSE262992:condition:{condition}",
                "replicate_group_uid": f"geo:GSE262992:condition:{condition}",
                "compound_uid": "pubchem:5743" if is_action else "",
                "compound_name": "dexamethasone" if is_action else "",
                "dose_value": 1.0 if is_action else np.nan,
                "dose_unit": "uM" if is_action else "",
                "time_value": 72.0,
                "time_unit": "h",
                "schedule": "continuous_72h" if is_action else "not_applicable_control",
                "route": "in_vitro_media" if is_action else "not_applicable_control",
                "action_uid": action_uid if is_action else "",
                "control_group_uid": "geo:GSE262992:untreated_control",
                "value_row_index": index,
            }
        )
    return pd.DataFrame(rows)


def _gse_action(candidate: Mapping[str, object]) -> pd.DataFrame:
    observation_uid = "geo_observation:GSE262992:dexamethasone"
    return pd.DataFrame(
        [
            {
                "action_uid": "geo_action:GSE262992:dexamethasone_1uM_72h",
                "intervention_type": "compound_perturbation",
                "agent_uid": "pubchem:5743",
                "agent_set": "pubchem:5743",
                "target_set_or_explicit_unknown": "explicit_unknown",
                "target_set": "explicit_unknown",
                "physical_dose": 1.0,
                "dose_value": 1.0,
                "dose_unit": "uM",
                "duration": 72.0,
                "duration_unit": "h",
                "schedule": "continuous_72h",
                "route_or_not_applicable": "in_vitro_media",
                "route": "in_vitro_media",
                "cell_or_tissue_context": "mouse_hippocampal_neural_stem_cell_model",
                "assay_uid": "geo_assay:GPL24247_rna_seq",
                "assay_context": "mouse_hippocampal_neural_stem_cell_model|geo_assay:GPL24247_rna_seq",
                "state_scope": candidate["cns_context_class"],
                "publication_uid": "geo:GSE262992",
                "source_publication": "geo:GSE262992",
                "accession_uid": "GSE262992",
                "source_accession": "GSE262992",
                "exposure_program_ref_or_explicit_unavailable": "explicit_unavailable",
                "exposure_program_ref": "explicit_unavailable",
                "occupancy_program_ref_or_explicit_unavailable": "explicit_unavailable",
                "occupancy_program_ref": "explicit_unavailable",
                "observation_uid": observation_uid,
                "control_group_uid": "geo:GSE262992:untreated_control",
                "cns_context_class": candidate["cns_context_class"],
                "qc_grade": candidate["qc_grade_candidate"],
            }
        ]
    )


def _validate_retrieval_archive(
    retrieval: Mapping[str, object],
    candidate: Mapping[str, object],
    archive_path: Path,
) -> None:
    """Bind the local snapshot to a verified official retrieval by content identity.

    A snapshot may be exposed through a new semantic storage path without
    changing its official object identity.  The retrieval record therefore
    binds the official URL, byte count, and digest; its historical local path
    is provenance, not an additional identity requirement.
    """
    retrieval_assets = retrieval.get("assets")
    retrieval_archive = (
        retrieval_assets.get("raw_archive") if isinstance(retrieval_assets, Mapping) else None
    )
    if not isinstance(retrieval_archive, Mapping):
        raise ValueError("GSE262992 retrieval evidence lacks the raw archive binding")
    expected_bytes = archive_path.stat().st_size
    expected_hash = sha256_file(archive_path)
    if retrieval.get("status") != "passed":
        raise ValueError("GSE262992 retrieval evidence is not passed")
    if retrieval.get("source_accession") != candidate["source_accession"]:
        raise ValueError("GSE262992 retrieval evidence has a different source accession")
    if retrieval.get("official_release") != candidate["official_release"]:
        raise ValueError("GSE262992 retrieval evidence has a different official release")
    if retrieval_archive.get("official_url") != candidate["raw_archive_url"]:
        raise ValueError("GSE262992 retrieval evidence has a different official archive URL")
    if (
        int(retrieval_archive.get("bytes", -1)) != expected_bytes
        or retrieval_archive.get("sha256") != expected_hash
    ):
        raise ValueError("GSE262992 retrieval evidence does not match the local archive identity")


def materialize_gse262992(
    policy: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    candidate = policy["candidates"]["gse262992"]
    assert isinstance(candidate, Mapping)
    snapshot_root = REPO_ROOT / str(policy["paths"]["raw_root"]) / "geo/GSE262992/official_snapshots"
    archive_path = snapshot_root / "GSE262992_RAW.tar"
    if archive_path.stat().st_size != int(candidate["expected_raw_archive_bytes"]):
        raise ValueError("GSE262992 raw archive byte count drift")
    if sha256_file(archive_path) != str(candidate["expected_raw_archive_sha256"]):
        raise ValueError("GSE262992 raw archive checksum drift")
    retrieval_path = REPO_ROOT / "Output/audits/c1_rescue_gse262992_retrieval.json"
    retrieval = _load_json(retrieval_path)
    _validate_retrieval_archive(retrieval, candidate, archive_path)

    metadata_path = snapshot_root / "GSE262992_series_matrix.txt.gz"
    sample_metadata, series = _series_matrix_metadata(metadata_path)
    if series.get("geo_accession") != "GSE262992" or series.get("status") != candidate["official_release"]:
        raise ValueError("GSE262992 official accession or public release drift")
    expected_members = {row["supplementary_file"] for row in sample_metadata.values()}
    expression_by_sample: dict[str, pd.Series] = {}
    symbol_map: dict[str, set[str]] = {}
    diagnostics_by_sample: dict[str, dict[str, int]] = {}
    with tarfile.open(archive_path, mode="r:") as archive:
        members = _safe_tar_members(archive)
        if set(members) != expected_members:
            raise ValueError("GSE262992 archive member names differ from sample metadata")
        reference_genes: tuple[str, ...] | None = None
        for accession in sorted(sample_metadata):
            row = sample_metadata[accession]
            expression, symbols, diagnostics = _read_gse_expression(
                archive, members[row["supplementary_file"]]
            )
            genes = tuple(expression.index.astype(str))
            if reference_genes is None:
                reference_genes = genes
            elif genes != reference_genes:
                raise ValueError(f"GSE262992 canonical gene axis differs for {accession}")
            expression_by_sample[accession] = expression
            diagnostics_by_sample[accession] = diagnostics
            for gene_id, symbol_text in symbols.items():
                symbol_map.setdefault(gene_id, set()).update(
                    value for value in symbol_text.split("|") if value
                )
    if reference_genes is None:
        raise ValueError("GSE262992 contains no eligible Ensembl genes")

    observations = _gse_observations(candidate)
    condition_values: list[np.ndarray] = []
    lineage_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []
    for accession in sorted(sample_metadata):
        condition = _gse_condition(accession, candidate)
        row = sample_metadata[accession]
        lineage_rows.append({**row, "condition": condition, **diagnostics_by_sample[accession]})
        if condition in {"bmp4_only", "dexamethasone_plus_bmp4"}:
            quarantine_rows.append(
                {
                    **row,
                    "condition": condition,
                    "quarantine_reason": "outside_frozen_minimal_dex_control_admission",
                }
            )
    for condition in ("control", "dexamethasone"):
        accessions = candidate["admitted_conditions"][condition]["sample_accessions"]
        replicate_matrix = np.vstack(
            [expression_by_sample[str(accession)].to_numpy(dtype=np.float64) for accession in accessions]
        )
        condition_values.append(np.median(replicate_matrix, axis=0).astype(np.float32))
    matrix = np.vstack(condition_values)
    if matrix.shape != (2, len(reference_genes)) or not np.isfinite(matrix).all():
        raise ValueError("GSE262992 canonical condition matrix is malformed")
    features = pd.DataFrame(
        {
            "feature_index": np.arange(len(reference_genes), dtype=np.int64),
            "feature_namespace": "Ensembl_Gene",
            "feature_id": list(reference_genes),
            "canonical_gene_id": list(reference_genes),
            "source_gene_symbols": [
                "|".join(sorted(symbol_map.get(gene_id, set()))) or "explicit_unavailable"
                for gene_id in reference_genes
            ],
            "mapping_method": "exact_mm10_Ensembl_gene_id_strip_version_then_sum_transcript_FPKM",
            "mapping_confidence": "exact_ensembl_gene_identifier",
            "aggregation_rule": "sum_transcript_FPKM_then_median_three_declared_condition_replicates",
        }
    )
    actions = _gse_action(candidate)
    output_root = REPO_ROOT / str(policy["paths"]["canonical_root"])
    staging_root = output_root / ".c1_rescue_staging"
    ensure_directory(staging_root)
    with tempfile.TemporaryDirectory(prefix="gse262992-", dir=staging_root) as temporary:
        stage = Path(temporary)
        store = stage / "expression.zarr"
        initialize_zarr_store(
            store,
            {
                "format": "zarr_v2_uncompressed",
                "source_accession": "GSE262992",
                "row_axis": "observations.parquet:value_row_index",
                "column_axis": "features.parquet:feature_index",
                "value_semantics": "FPKM_condition_replicate_median",
            },
        )
        write_zarr_array(
            store,
            "values",
            matrix,
            (2, len(features)),
            attrs={"feature_namespace": "Ensembl_Gene", "unit": "FPKM_condition_replicate_median"},
        )
        write_parquet_atomic(observations, stage / "observations.parquet")
        write_parquet_atomic(features, stage / "features.parquet")
        write_parquet_atomic(actions, stage / "actions.parquet")
        write_parquet_atomic(pd.DataFrame(lineage_rows), stage / "replicate_lineage.parquet")
        write_parquet_atomic(pd.DataFrame(quarantine_rows), stage / "condition_quarantine.parquet")
        matrix_manifest = {
            "matrix_shape": list(matrix.shape),
            "source_sample_count": len(sample_metadata),
            "admitted_source_sample_count": 6,
            "quarantined_source_sample_count": 6,
            "condition_observation_count": 2,
            "feature_count": len(features),
            "scalar_value_count": int(matrix.size),
            "finite_values": True,
            "minimum_value": float(matrix.min()),
            "maximum_value": float(matrix.max()),
            "sample_mapping_conflict": {
                "status": "retained_nonblocking_filename_shorthand_conflict",
                "official_treatment_source": "GEO_sample_characteristics_ch1",
                "conflict": "Dex accessions GSM8182842-44 have B_20 raw filename shorthand; no treatment was inferred from filename",
            },
        }
        write_json(stage / "matrix_manifest.json", matrix_manifest)
        manifest = {
            "schema_version": "neuronsifter-c1-rescue-v2-observation-block-v1",
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": GSE_CANONICAL_UID,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "cns_source": True,
            "cns_context_class": candidate["cns_context_class"],
            "role": "qualified_for_mainline",
            "admission_allowed": True,
            "value_block_format": "zarr",
            "value_block_path": "expression.zarr",
            "sample_metadata_path": "observations.parquet",
            "feature_metadata_path": "features.parquet",
            "action_catalog_path": "actions.parquet",
            "replicate_lineage_path": "replicate_lineage.parquet",
            "quarantine_path": "condition_quarantine.parquet",
            "matrix_manifest_path": "matrix_manifest.json",
            "observation_count": 2,
            "source_sample_count": 12,
            "biological_unit_count": 2,
            "effective_sample_size": 2.0,
            "effective_sample_size_formula": "two independent condition units after within-condition replicate aggregation",
            "biological_unit_policy": candidate["biological_unit_policy"],
            "action_count": 1,
            "action_linkage_complete": True,
            "real_value_matrix": True,
            "compound_dose_time_axes_complete": True,
            "qc_grade": candidate["qc_grade_candidate"],
            "qc_grade_rationale": candidate["qc_grade_rationale"],
            "qc": {**matrix_manifest, "status": "passed"},
            "mapping_confidence": "official_GEO_sample_accession_and_exact_mm10_Ensembl_gene_identifier",
            "mapping_version": "gse262992_stringtie_exact_ensembl_gene_sum_v1",
            "unit_mapping_version": "geo_1uM_72h_FPKM_v1",
            "ontology_versions": {
                "species": "NCBITaxon:10090",
                "brain_region": "UBERON:0002421",
                "feature": "Ensembl_Gene_mm10_source_native",
                "compound": "PubChem:5743",
                "unit": "uM|h|FPKM",
            },
            "semantic_axes_complete": True,
            "observation_value_join_contract": {
                "observation_axis": "observations.parquet:value_row_index",
                "feature_axis": "features.parquet:feature_index|feature_id",
                "value_axis": "expression.zarr:values[value_row_index,feature_index]",
                "feature_namespace": "Ensembl_Gene",
                "value_unit": "FPKM_condition_replicate_median",
            },
            "source_release": {
                "official_url": candidate["official_series_url"],
                "release_version": candidate["official_release"],
                "access_class": "public",
                "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
                "terms_snapshot": _binding(snapshot_root / "geo_download_terms.html"),
            },
            "parser": {
                "parser_id": "materialize_c1_rescue_gse262992_v1",
                "parser_version": "1.0.0",
                "condition_aggregation": "median_three_declared_replicates",
                "gene_aggregation": "sum_transcript_FPKM_by_exact_Ensembl_gene_id",
            },
            "input_bindings": [
                _binding(snapshot_root / name)
                for name in (
                    "GSE262992_RAW.tar",
                    "GSE262992_series_matrix.txt.gz",
                    "GSE262992.soft.txt",
                    "filelist.txt",
                    "geo_download_terms.html",
                )
            ]
            + [_binding(retrieval_path)],
            "split_role": "unassigned_pending_v2_nested_split_freeze",
            "blocker_codes": ["v2_nested_split_freeze_pending", "v2_real_domain_leakage_audit_pending"],
        }
        destination, final_manifest = _write_content_addressed_block(
            stage,
            output_root,
            source_id="geo",
            accession_path="GSE262992",
            output_names=(
                "expression.zarr",
                "observations.parquet",
                "features.parquet",
                "actions.parquet",
                "replicate_lineage.parquet",
                "condition_quarantine.parquet",
                "matrix_manifest.json",
            ),
            manifest=manifest,
        )
    return destination, final_manifest


def _e_mtab_control_observations(controls: pd.DataFrame) -> pd.DataFrame:
    frame = controls.copy()
    frame["observation_uid"] = frame["control_uid"].str.replace(
        "arrayexpress_control:", "arrayexpress_control_observation:", regex=False
    )
    frame["biological_unit_uid"] = (
        "arrayexpress:E-MTAB-6772:control_sample:" + frame["source_sample_name"].astype(str)
    )
    frame["compound_uid"] = ""
    frame["compound_name"] = ""
    frame["dose_value"] = np.nan
    frame["dose_unit"] = ""
    frame["time_value"] = 5.5
    frame["time_unit"] = "month"
    frame["schedule"] = "not_applicable_vehicle_control"
    frame["route"] = "intraperitoneal_vehicle_control"
    frame["action_uid"] = ""
    frame["qc_grade"] = "B"
    frame["value_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["counted_in_observation_block"] = True
    frame["counted_in_ess"] = True
    return frame


def materialize_e_mtab_controls(
    policy: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    candidate = policy["candidates"]["e_mtab_6772_controls"]
    assert isinstance(candidate, Mapping)
    parent_manifest_path = REPO_ROOT / str(candidate["parent_v1_block_manifest"])
    parent_manifest = _load_json(parent_manifest_path)
    if parent_manifest.get("block_sha256") != candidate["parent_v1_block_sha256"]:
        raise ValueError("E-MTAB-6772 parent V1 block hash drift")
    project_config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    phase1_policy = project_config["public_foundation_corpus"][
        "phase1f_perturbation_pair_materialization"
    ]
    phase1_candidate = phase1_policy["candidate"]
    phase1_qc = phase1_policy["qc"]
    snapshot_root = REPO_ROOT / str(phase1_candidate["official_snapshot_root"])
    assay_paths = sorted(snapshot_root.glob("[0-9]*_?.txt"))
    treated, controls = read_e_mtab_sdrf(
        REPO_ROOT / str(phase1_candidate["sample_metadata_path"]), phase1_policy
    )
    sample_order = pd.concat([treated, controls], ignore_index=True)
    available = {path.name: path for path in assay_paths}
    probe_ids: list[str] | None = None
    rows: list[np.ndarray] = []
    for filename in sample_order["source_value_file"]:
        current_probes, current_values = read_e_mtab_assay(available[str(filename)])
        if probe_ids is None:
            probe_ids = current_probes
        elif current_probes != probe_ids:
            raise ValueError(f"E-MTAB-6772 probe axis differs in {filename}")
        rows.append(current_values)
    if probe_ids is None:
        raise ValueError("E-MTAB-6772 assays are empty")
    features, feature_quarantine, matrix, diagnostics = canonicalize_e_mtab(
        probe_ids,
        np.vstack(rows),
        read_e_mtab_adf(REPO_ROOT / str(phase1_candidate["platform_annotation_path"])),
    )
    expected_diagnostics = {
        "source_probe_count": int(phase1_qc["expected_source_probe_count"]),
        "exact_entrez_probe_count": int(phase1_qc["expected_exact_entrez_probe_count"]),
        "canonical_gene_count": int(phase1_qc["expected_canonical_gene_count"]),
        "unmatched_probe_count": int(phase1_qc["expected_unmatched_probe_count"]),
        "non_numeric_entrez_count": int(phase1_qc["expected_non_numeric_entrez_count"]),
    }
    if diagnostics != expected_diagnostics:
        raise ValueError("E-MTAB-6772 control reconstruction mapping drift")
    control_matrix = matrix[len(treated) :]
    observations = _e_mtab_control_observations(controls)
    if control_matrix.shape != (6, 14042) or not np.isfinite(control_matrix).all():
        raise ValueError("E-MTAB-6772 control matrix shape or finite-value contract failed")
    output_root = REPO_ROOT / str(policy["paths"]["canonical_root"])
    staging_root = output_root / ".c1_rescue_staging"
    ensure_directory(staging_root)
    with tempfile.TemporaryDirectory(prefix="e-mtab-6772-controls-", dir=staging_root) as temporary:
        stage = Path(temporary)
        store = stage / "expression.zarr"
        initialize_zarr_store(
            store,
            {
                "format": "zarr_v2_uncompressed",
                "source_accession": "E-MTAB-6772",
                "block_variant": "controls_v2",
                "row_axis": "observations.parquet:value_row_index",
                "column_axis": "features.parquet:feature_index",
                "value_semantics": "normalized_Illumina_AVG_Signal",
            },
        )
        write_zarr_array(
            store,
            "values",
            control_matrix.astype(np.float32),
            (6, len(features)),
            attrs={"feature_namespace": "NCBI_Entrez_Gene", "unit": "normalized_Illumina_AVG_Signal"},
        )
        write_parquet_atomic(observations, stage / "observations.parquet")
        write_parquet_atomic(features, stage / "features.parquet")
        write_parquet_atomic(feature_quarantine, stage / "feature_quarantine.parquet")
        overlap_declaration = {
            "status": "declared_same_study_matrix_not_counted_as_independent_accession",
            "parent_v1_block": _binding(parent_manifest_path),
            "parent_v1_block_sha256": parent_manifest["block_sha256"],
            "shared_source_accession": "E-MTAB-6772",
            "shared_publication_accession_group": candidate["publication_accession_group"],
            "shared_mirror_group": candidate["mirror_group"],
            "sample_uid_overlap_count": 0,
            "biological_unit_uid_overlap_count": 0,
            "independence_claim_allowed": False,
            "cross_source_anchor_role": "non_action_control_context_only",
        }
        write_json(stage / "source_overlap_declaration.json", overlap_declaration)
        matrix_manifest = {
            "matrix_shape": list(control_matrix.shape),
            "source_assay_count": len(assay_paths),
            "scalar_value_count": int(control_matrix.size),
            "finite_values": True,
            "minimum_value": float(control_matrix.min()),
            "maximum_value": float(control_matrix.max()),
            "mapping_diagnostics": diagnostics,
            "parent_v1_treated_observation_count": len(treated),
            "control_observation_count": len(observations),
        }
        write_json(stage / "matrix_manifest.json", matrix_manifest)
        manifest = {
            "schema_version": "neuronsifter-c1-rescue-v2-observation-block-v1",
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": CONTROL_CANONICAL_UID,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "block_variant": candidate["block_variant"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "cns_source": True,
            "cns_context_class": "mouse_hippocampus_control_context",
            "role": "qualified_for_mainline",
            "observation_role": "qualified_non_action_control_context",
            "admission_allowed": True,
            "value_block_format": "zarr",
            "value_block_path": "expression.zarr",
            "sample_metadata_path": "observations.parquet",
            "feature_metadata_path": "features.parquet",
            "feature_quarantine_path": "feature_quarantine.parquet",
            "source_overlap_declaration_path": "source_overlap_declaration.json",
            "matrix_manifest_path": "matrix_manifest.json",
            "observation_count": 6,
            "biological_unit_count": 6,
            "effective_sample_size": 6.0,
            "effective_sample_size_formula": "six distinct source-declared control specimens; no invented mouse identities",
            "biological_unit_policy": candidate["biological_unit_policy"],
            "action_count": 0,
            "action_linkage_complete": True,
            "action_status": "not_applicable_control",
            "real_value_matrix": True,
            "compound_dose_time_axes_complete": False,
            "qc_grade": candidate["qc_grade_candidate"],
            "qc_grade_rationale": "Real official assay values with exact ADF mapping; source pooling ambiguity keeps Grade B and declared specimens are not promoted to mouse-level independence.",
            "qc": {**matrix_manifest, "status": "passed"},
            "mapping_confidence": "exact_A-MEXP-932_reporter_to_numeric_Entrez_with_median_probe_aggregation",
            "mapping_version": "a_mexp_932_exact_reporter_numeric_entrez_median_v1",
            "unit_mapping_version": "normalized_Illumina_AVG_Signal_not_ucum_v1",
            "ontology_versions": {
                "species": "NCBITaxon:10090",
                "brain_region": "UBERON:0002421",
                "feature": "NCBI_Entrez_Gene_A-MEXP-932",
            },
            "semantic_axes_complete": True,
            "observation_value_join_contract": {
                "observation_axis": "observations.parquet:value_row_index",
                "feature_axis": "features.parquet:feature_index|feature_id",
                "value_axis": "expression.zarr:values[value_row_index,feature_index]",
                "feature_namespace": "NCBI_Entrez_Gene",
                "value_unit": "normalized_Illumina_AVG_Signal",
            },
            "source_independence": overlap_declaration,
            "input_bindings": [_binding(path) for path in assay_paths]
            + [
                _binding(REPO_ROOT / str(phase1_candidate["sample_metadata_path"])),
                _binding(REPO_ROOT / str(phase1_candidate["study_metadata_path"])),
                _binding(REPO_ROOT / str(phase1_candidate["platform_annotation_path"])),
                _binding(parent_manifest_path),
            ],
            "parser": {
                "parser_id": "materialize_c1_rescue_e_mtab_6772_controls_v1",
                "parser_version": "1.0.0",
                "mapping_version": "a_mexp_932_exact_reporter_numeric_entrez_median_v1",
            },
            "split_role": "unassigned_pending_v2_nested_split_freeze",
            "blocker_codes": ["v2_nested_split_freeze_pending", "v2_real_domain_leakage_audit_pending"],
        }
        destination, final_manifest = _write_content_addressed_block(
            stage,
            output_root,
            source_id="arrayexpress",
            accession_path="E-MTAB-6772/controls_v2",
            output_names=(
                "expression.zarr",
                "observations.parquet",
                "features.parquet",
                "feature_quarantine.parquet",
                "source_overlap_declaration.json",
                "matrix_manifest.json",
            ),
            manifest=manifest,
        )
    return destination, final_manifest


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("C1 rescue V2 materialization is repository-root bound")
    policy = _load_policy()
    parent = policy["parent_corpus"]
    for key in ("evidence_bundle", "canonical_blocks", "frozen_splits", "leakage_audit"):
        path = REPO_ROOT / str(parent[f"{key}_path"])
        if sha256_file(path) != str(parent[f"{key}_sha256"]):
            raise ValueError(f"Frozen V1 parent artifact drift: {key}")
    gse_path, gse_manifest = materialize_gse262992(policy)
    controls_path, controls_manifest = materialize_e_mtab_controls(policy)
    report = {
        "schema_version": "neuronsifter-c1-rescue-v2-materialization-v1",
        "status": "passed",
        "created_at": _now(),
        "parent_corpus_uid": parent["corpus_uid"],
        "parent_corpus_identity_hash": parent["identity_hash"],
        "v1_mutation_performed": False,
        "blocks": [
            {
                "canonical_uid": gse_manifest["canonical_uid"],
                "manifest": _binding(gse_path / "block_manifest.json"),
                "block_sha256": gse_manifest["block_sha256"],
                "observation_count": gse_manifest["observation_count"],
                "effective_sample_size": gse_manifest["effective_sample_size"],
            },
            {
                "canonical_uid": controls_manifest["canonical_uid"],
                "manifest": _binding(controls_path / "block_manifest.json"),
                "block_sha256": controls_manifest["block_sha256"],
                "observation_count": controls_manifest["observation_count"],
                "effective_sample_size": controls_manifest["effective_sample_size"],
            },
        ],
        "total_new_observation_count": int(
            gse_manifest["observation_count"] + controls_manifest["observation_count"]
        ),
        "total_new_effective_sample_size": float(
            gse_manifest["effective_sample_size"] + controls_manifest["effective_sample_size"]
        ),
        "blockers": [],
    }
    write_json(REPO_ROOT / "Output/audits/c1_rescue_materialization.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build(REPO_ROOT)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
