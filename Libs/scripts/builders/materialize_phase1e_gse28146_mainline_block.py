"""Materialize the first real, accession-scoped Phase 1E mainline block.

The input is the official GEO GSE28146 SOFT family record.  Legacy NPY/TSV
bundles are deliberately not read by this builder.  ArrayExpress assets are
kept only as a checksum-bound mirror provenance record for the same study.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_alignment_ready_registry import (
    gene_reference_maps,
    read_adf_rows,
    transcript_mapping_from_rows,
)
from Libs.utils.data.public_foundation_evidence import canonical_sha256, sha256_file
from Libs.utils.data.public_foundation_roles import REQUALIFICATION_DELTA_FIELDS
from Libs.utils.data.public_foundation_roles import upsert_accession_requalification_delta
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    load_config,
    resolve_repo_path,
    write_json,
    write_tsv,
)
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic


BUILDER_ID = "Libs/scripts/builders/materialize_phase1e_gse28146_mainline_block.py"
MANIFEST_VERSION = "neuronsifter-phase1e-canonical-mainline-block-v1"
ADMISSION_VERSION = "neuronsifter-phase1e-mainline-admission-v1"
MISSING = {"", "na", "n/a", "none", "not provided"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str) -> Path:
    return resolve_repo_path(value)


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parse_characteristic(value: str) -> tuple[str, str]:
    if ":" not in value:
        return "", value.strip()
    key, parsed = value.split(":", 1)
    return key.strip().casefold(), parsed.strip()


def _parse_age_years(value: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


def _parse_soft(path: Path) -> tuple[list[dict[str, object]], list[str], np.ndarray, np.ndarray]:
    """Read source sample tables directly from a GEO family SOFT archive."""

    samples: list[dict[str, object]] = []
    probe_order: list[str] | None = None
    signals: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    current: dict[str, object] | None = None
    in_table = False
    header: list[str] = []
    current_probe_ids: list[str] = []
    current_values: list[float] = []
    current_probabilities: list[float] = []

    def finish_current() -> None:
        nonlocal current, current_probe_ids, current_values, current_probabilities
        if current is None:
            return
        accession = str(current.get("source_sample_accession", ""))
        if not accession:
            raise ValueError("GEO SOFT sample section lacks !Sample_geo_accession")
        if not current_values:
            raise ValueError(f"GEO SOFT sample {accession} lacks value rows")
        if probe_order is None:
            raise AssertionError("probe_order must be established before finishing a sample")
        if current_probe_ids != probe_order:
            raise ValueError(f"GEO SOFT feature order differs for sample {accession}")
        samples.append(current)
        signals.append(np.asarray(current_values, dtype=np.float64))
        probabilities.append(np.asarray(current_probabilities, dtype=np.float64))
        current = None
        current_probe_ids = []
        current_values = []
        current_probabilities = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if in_table:
                    raise ValueError("GEO SOFT begins a new sample before closing the prior table")
                finish_current()
                current = {
                    "source_sample_accession": line.split("=", 1)[1].strip(),
                    "characteristics": [],
                }
                continue
            if current is None:
                continue
            if line == "!sample_table_begin":
                if in_table:
                    raise ValueError("Nested GEO SOFT sample table")
                in_table = True
                header = []
                continue
            if line == "!sample_table_end":
                if not in_table:
                    raise ValueError("GEO SOFT closes a table that was not open")
                in_table = False
                if probe_order is None:
                    probe_order = list(current_probe_ids)
                continue
            if in_table:
                if not header:
                    header = next(csv.reader([line], delimiter="\t"))
                    if header != ["ID_REF", "VALUE", "PROBABILITY"]:
                        raise ValueError(f"Unexpected GEO SOFT table header: {header}")
                    continue
                values = next(csv.reader([line], delimiter="\t"))
                if len(values) != 3:
                    raise ValueError(f"Malformed GEO SOFT value row in {current['source_sample_accession']}: {values}")
                current_probe_ids.append(values[0].strip())
                try:
                    current_values.append(float(values[1]))
                    current_probabilities.append(float(values[2]))
                except ValueError as exc:
                    raise ValueError(f"Non-numeric GEO SOFT value row: {values}") from exc
                continue
            if line.startswith("!Sample_title = "):
                current["sample_title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch1 = "):
                current["source_name"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_organism_ch1 = "):
                current["organism"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_taxid_ch1 = "):
                current["taxid"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_platform_id = "):
                current["platform_accession"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_data_processing = "):
                current["data_processing"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1 = "):
                characteristics = current["characteristics"]
                if not isinstance(characteristics, list):
                    raise TypeError("characteristics must be a list")
                characteristics.append(line.split("=", 1)[1].strip())
    if in_table:
        raise ValueError("GEO SOFT ends with an open sample table")
    finish_current()
    if probe_order is None or not samples:
        raise ValueError("GEO SOFT contains no sample tables")
    return samples, probe_order, np.vstack(signals), np.vstack(probabilities)


def _sample_metadata(samples: list[dict[str, object]], candidate: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        characteristics = sample.get("characteristics", [])
        parsed = dict(_parse_characteristic(str(value)) for value in characteristics if str(value).strip())
        disease_raw = parsed.get("disease status", "")
        disease_key = disease_raw.casefold()
        if disease_key == "control":
            disease_state = "normal_control"
            disease_stage = "not_applicable_control"
        elif disease_key in {"incipient", "moderate", "severe"}:
            disease_state = "Alzheimer disease"
            disease_stage = disease_key
        else:
            raise ValueError(f"Unexpected GSE28146 disease status: {disease_raw!r}")
        accession = str(sample["source_sample_accession"])
        biological_unit_id = f"geo::GSE28146::specimen::{accession}"
        rows.append(
            {
                "sample_index": index,
                "source_sample_accession": accession,
                "biological_unit_id": biological_unit_id,
                "biological_unit_type": "specimen",
                "subject_or_donor_id": "not_provided",
                "subject_or_donor_status": "not_provided",
                "sample_id": accession,
                "cell_id": "not_applicable_bulk_laser_capture",
                "replicate_id": "not_applicable_one_array_per_specimen",
                "technical_replicate_group": biological_unit_id,
                "sample_title": str(sample.get("sample_title", "")),
                "sex": parsed.get("sex", "not_provided"),
                "age_years": _parse_age_years(parsed.get("age", "")),
                "age_raw": parsed.get("age", "not_provided"),
                "disease_state_original": disease_raw,
                "disease_state_canonical": disease_state,
                "disease_stage": disease_stage,
                "tissue_original": "CA1 field of hippocampus",
                "brain_region_original": "CA1 field of hippocampus",
                "brain_region_canonical": "hippocampal CA1 field",
                "brain_region_ontology_id": "UBERON:0001954",
                "cell_type": "not_applicable_laser_capture_gray_matter_bulk",
                "species": str(sample.get("organism", "")),
                "species_taxonomy_id": str(sample.get("taxid", "")),
                "assay": "Affymetrix HG-U133 Plus 2.0 expression microarray",
                "platform_accession": str(sample.get("platform_accession", "")),
                "modality": "transcriptomics",
                "source_id": str(candidate["source_id"]),
                "source_accession": str(candidate["source_accession"]),
                "publication_accession_group": str(candidate["publication_accession_group"]),
                "mirror_group": str(candidate["mirror_group"]),
                "value_unit_original": "MAS5-calculated Signal intensity",
                "value_unit_canonical": "MAS5-calculated Signal intensity",
                "unit_conversion_formula": "identity",
                "time_status": "not_applicable_cross_sectional_post_mortem",
                "dose_status": "not_applicable_observational_tissue_study",
                "source_sample_description": str(sample.get("source_name", "")),
                "data_processing": str(sample.get("data_processing", "")),
            }
        )
    return pd.DataFrame(rows)


def _feature_mapping(config: dict, annotation_path: Path) -> dict[str, dict[str, str]]:
    gene_symbol_lookup, gene_id_to_symbol = gene_reference_maps(config)
    header, rows = read_adf_rows(annotation_path)
    mapping = transcript_mapping_from_rows(
        header,
        rows,
        config,
        "official_arrayexpress_a_affy_44_adf",
        gene_symbol_lookup,
        gene_id_to_symbol,
    )
    if not mapping:
        raise ValueError("Official A-AFFY-44 annotation produced no probe-to-gene mappings")
    return mapping


def _canonicalize_matrix(
    probe_ids: list[str],
    signal: np.ndarray,
    probability: np.ndarray,
    mapping: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if signal.shape != probability.shape or signal.shape[1] != len(probe_ids):
        raise ValueError("GEO SOFT signal/probability matrix dimensions are inconsistent")
    positions_by_gene: dict[str, list[int]] = defaultdict(list)
    mapping_rows: list[dict[str, object]] = []
    metadata_by_gene: dict[str, dict[str, str]] = {}
    for position, probe_id in enumerate(probe_ids):
        record = mapping.get(probe_id)
        if record is None:
            continue
        gene_id = record["canonical_entity_id"]
        positions_by_gene[gene_id].append(position)
        metadata_by_gene.setdefault(gene_id, record)
        mapping_rows.append(
            {
                "source_probe_id": probe_id,
                "canonical_gene_id": gene_id,
                "canonical_gene_symbol": record["canonical_entity_label"],
                "mapping_method": record["mapping_method"],
                "mapping_confidence": record["mapping_confidence"],
                "aggregation_rule": record["aggregation_rule"],
            }
        )
    if not positions_by_gene:
        raise ValueError("No GSE28146 probes mapped to canonical genes")
    ordered_gene_ids = sorted(positions_by_gene, key=lambda gene: min(positions_by_gene[gene]))
    signal_by_gene = np.empty((signal.shape[0], len(ordered_gene_ids)), dtype=np.float64)
    probability_by_gene = np.empty_like(signal_by_gene)
    feature_rows: list[dict[str, object]] = []
    for gene_index, gene_id in enumerate(ordered_gene_ids, start=1):
        positions = positions_by_gene[gene_id]
        signal_by_gene[:, gene_index - 1] = np.nanmedian(signal[:, positions], axis=1)
        probability_by_gene[:, gene_index - 1] = np.nanmedian(probability[:, positions], axis=1)
        record = metadata_by_gene[gene_id]
        feature_rows.append(
            {
                "feature_index": gene_index,
                "canonical_gene_id": gene_id,
                "canonical_gene_symbol": record["canonical_entity_label"],
                "canonical_namespace": "entrez_gene_id",
                "source_probe_ids": "|".join(probe_ids[position] for position in positions),
                "source_probe_count": len(positions),
                "mapping_method": record["mapping_method"],
                "mapping_confidence": record["mapping_confidence"],
                "aggregation_rule": record["aggregation_rule"],
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(mapping_rows), signal_by_gene, probability_by_gene


def _value_rows(samples: pd.DataFrame, features: pd.DataFrame, signal: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    if signal.shape != probability.shape:
        raise ValueError("Canonical signal and probability matrices differ in shape")
    if signal.shape != (len(samples), len(features)):
        raise ValueError("Canonical matrix dimensions do not match sample/feature metadata")
    row_count = signal.size
    return pd.DataFrame(
        {
            "biological_unit_id": np.repeat(samples["biological_unit_id"].to_numpy(), len(features)),
            "source_sample_accession": np.repeat(samples["source_sample_accession"].to_numpy(), len(features)),
            "canonical_gene_id": np.tile(features["canonical_gene_id"].to_numpy(), len(samples)),
            "value": signal.reshape(row_count),
            "detection_p_value": probability.reshape(row_count),
            "value_unit_original": "MAS5-calculated Signal intensity",
            "value_unit_canonical": "MAS5-calculated Signal intensity",
            "unit_conversion_formula": "identity",
            "source_value_kind": "official_geo_soft_sample_table",
        }
    )


def _file_records(root: Path, names: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"name": name, **_binding(root / name)} for name in names]


def _block_content_hash(records: list[dict[str, object]]) -> str:
    return canonical_sha256(
        [
            {"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in records
        ]
    )


def _semantic_axes(candidate: dict[str, object], parser: dict[str, object]) -> dict[str, dict[str, str]]:
    return {
        "study": {"status": "observed", "value": "GSE28146"},
        "biological_unit": {"status": "observed", "value": "specimen; source accession is the unit identity"},
        "species": {"status": "mapped", "value": "Homo sapiens; NCBI:txid:9606"},
        "disease_state": {"status": "observed", "value": "Control, Incipient, Moderate, Severe"},
        "tissue": {"status": "observed", "value": "laser-capture CA1 hippocampal gray matter"},
        "brain_region": {"status": "mapped", "value": "hippocampal CA1 field; UBERON:0001954"},
        "cell_type": {"status": "not_applicable", "reason": "laser-capture gray-matter bulk specimen has no cell-level axis"},
        "assay": {"status": "observed", "value": "Affymetrix HG-U133 Plus 2.0; GPL570"},
        "modality": {"status": "observed", "value": "transcriptomics"},
        "feature": {"status": "mapped", "value": "Entrez Gene ID from official A-AFFY-44 annotation"},
        "time": {"status": "not_applicable", "reason": "cross-sectional post-mortem study; no real longitudinal time axis"},
        "dose": {"status": "not_applicable", "reason": "observational tissue study has no intervention dose"},
        "unit": {"status": "mapped", "value": str(parser["unit_mapping_version"])},
        "provenance": {"status": "observed", "value": f"{candidate['source_id']}:{candidate['source_accession']}"},
    }


def _manifest(
    *,
    candidate: dict[str, object],
    policy: dict[str, object],
    samples: pd.DataFrame,
    features: pd.DataFrame,
    values: pd.DataFrame,
    files: list[dict[str, object]],
    block_sha256: str,
    input_bindings: list[dict[str, object]],
) -> dict[str, object]:
    qc = policy["qc"]
    biological = policy["biological_unit_policy"]
    parser = policy["parser"]
    finite_fraction = float(np.isfinite(values["value"].to_numpy(dtype=np.float64)).mean())
    stage_counts = samples["disease_stage"].value_counts().sort_index().to_dict()
    return {
        "schema_version": MANIFEST_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "canonical_uid": "pfc_geo_gse28146_phase1e_v1",
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
        "value_block_format": "parquet",
        "value_block_path": "values.parquet",
        "sample_metadata_path": "sample_metadata.parquet",
        "feature_metadata_path": "feature_metadata.parquet",
        "source_probe_mapping_path": "source_probe_mapping.parquet",
        "observation_count": int(len(samples)),
        "value_row_count": int(len(values)),
        "biological_unit_count": int(samples["biological_unit_id"].nunique()),
        "effective_sample_size": float(samples["biological_unit_id"].nunique()),
        "effective_sample_size_formula": biological["effective_sample_size_formula"],
        "biological_unit_policy": biological,
        "qc_grade": qc["grade"],
        "qc_grade_rationale": qc["grade_rationale"],
        "qc": {
            "sample_count": int(len(samples)),
            "expected_sample_count": qc["expected_sample_count"],
            "mapped_gene_count": int(len(features)),
            "minimum_mapped_gene_count": qc["minimum_mapped_gene_count"],
            "finite_value_fraction": finite_fraction,
            "minimum_finite_fraction": qc["minimum_finite_fraction"],
            "detection_p_value_in_unit_interval": bool(values["detection_p_value"].between(0.0, 1.0).all()),
            "disease_stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
            "status": "passed",
        },
        "mapping_confidence": "official_platform_annotation_gene_aggregation",
        "mapping_version": parser["mapping_version"],
        "unit_mapping_version": parser["unit_mapping_version"],
        "ontology_versions": {
            "species": "NCBI_Taxonomy:9606",
            "brain_region": "UBERON:0001954",
            "gene": "NCBI_Entrez_Gene via official A-AFFY-44 annotation",
        },
        "semantic_axes": _semantic_axes(candidate, parser),
        "split_role": "unassigned_pending_nested_split_freeze",
        "semantic_axes_complete": True,
        "source_release": {
            "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE28146",
            "study_release_date": candidate["study_release_date"],
            "publication_pubmed_id": candidate["publication_pubmed_id"],
            "publication_doi": candidate["publication_doi"],
            "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
            "access_class": "public_with_attribution",
            "reuse_evidence": "Official GEO FAQ states released records are available for anyone to access, download and re-use.",
        },
        "mirror_provenance": {
            "mirror_source_id": candidate["mirror_source_id"],
            "mirror_source_accession": candidate["mirror_source_accession"],
            "independence_status": "same_study_mirror_not_independent",
            "counts_as_independent_source": False,
        },
        "parser": parser,
        "input_bindings": input_bindings,
        "block_files": files,
        "blocker_codes": ["second_independent_disease_source_missing", "nested_split_freeze_pending"],
    }


def _existing_manifest(destination: Path, block_sha256: str) -> dict[str, object] | None:
    manifest_path = destination / "block_manifest.json"
    if not manifest_path.is_file():
        return None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("block_sha256") != block_sha256:
        raise ValueError(f"Content-address path collision at {destination}")
    return existing


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_materialization"]
    candidate = dict(policy["candidate"])
    raw_path = _path(str(candidate["raw_value_path"]))
    annotation_path = _path(str(candidate["platform_annotation_path"]))
    samples_raw, probe_ids, signal, probability = _parse_soft(raw_path)
    samples = _sample_metadata(samples_raw, candidate)
    mapping = _feature_mapping(config, annotation_path)
    features, probe_mapping, signal_by_gene, probability_by_gene = _canonicalize_matrix(
        probe_ids, signal, probability, mapping
    )
    values = _value_rows(samples, features, signal_by_gene, probability_by_gene)
    qc = policy["qc"]
    if len(samples) != int(qc["expected_sample_count"]):
        raise ValueError(f"Expected {qc['expected_sample_count']} samples, found {len(samples)}")
    if len(features) < int(qc["minimum_mapped_gene_count"]):
        raise ValueError(f"Mapped gene count {len(features)} is below minimum {qc['minimum_mapped_gene_count']}")
    if not np.isfinite(values["value"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Canonical value block contains non-finite values")
    if not values["detection_p_value"].between(0.0, 1.0).all():
        raise ValueError("Detection probabilities are outside [0, 1]")

    root = _path(str(policy["output_root"]))
    staging_root = root / ".phase1e_staging"
    ensure_directory(staging_root)
    file_names = (
        "values.parquet",
        "sample_metadata.parquet",
        "feature_metadata.parquet",
        "source_probe_mapping.parquet",
    )
    with tempfile.TemporaryDirectory(prefix="gse28146-", dir=staging_root) as temporary:
        stage = Path(temporary)
        write_parquet_atomic(values, stage / "values.parquet")
        write_parquet_atomic(samples, stage / "sample_metadata.parquet")
        write_parquet_atomic(features, stage / "feature_metadata.parquet")
        write_parquet_atomic(probe_mapping, stage / "source_probe_mapping.parquet")
        records = _file_records(stage, file_names)
        block_sha256 = _block_content_hash(records)
        destination = root / "blocks" / str(candidate["source_id"]) / str(candidate["source_accession"]) / block_sha256[:16]
        existing = _existing_manifest(destination, block_sha256) if destination.exists() else None
        if existing is None:
            ensure_directory(destination.parent)
            stage.replace(destination)
        else:
            for record in records:
                existing_path = destination / str(record["name"])
                if not existing_path.is_file() or sha256_file(existing_path) != record["sha256"]:
                    raise ValueError(f"Existing content-addressed block differs: {existing_path}")

    input_paths = [str(candidate["raw_value_path"]), str(candidate["platform_annotation_path"])]
    input_paths.extend(str(value) for value in candidate["release_evidence_paths"])
    input_paths.extend(str(value) for value in candidate["raw_mirror_paths"])
    input_bindings = []
    seen_paths: set[str] = set()
    for relative in input_paths:
        if relative not in seen_paths:
            input_bindings.append(_binding(_path(relative)))
            seen_paths.add(relative)
    records = _file_records(destination, file_names)
    manifest = _manifest(
        candidate=candidate,
        policy=policy,
        samples=samples,
        features=features,
        values=values,
        files=records,
        block_sha256=block_sha256,
        input_bindings=input_bindings,
    )
    manifest_path = destination / "block_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ledger_path = _path(str(policy["source_ledger_path"]))
    delta_path = _path(str(policy["qualification_delta_path"]))
    delta_row = {
        "delta_version": "phase1e-source-qualification-delta-v1",
        "entry_id": "phase1e_geo_gse28146_mainline",
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "prior_decision": "deferred",
        "decision": "qualified_for_mainline",
        "admission_allowed": "true",
        "scope": "source_accession_only",
        "frozen_ledger_path": ledger_path.relative_to(repo_root).as_posix(),
        "frozen_ledger_sha256": sha256_file(ledger_path),
        "block_manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "block_manifest_sha256": sha256_file(manifest_path),
        "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
        "access_class": "public_with_attribution",
        "status": "accepted",
    }
    upsert_accession_requalification_delta(delta_path, delta_row)
    admission = {
        "schema_version": ADMISSION_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "status": "accepted_first_qualified_block",
        "candidate_id": "disease_geo_gse28146",
        "candidate_role_before": "deferred_source_not_consumed",
        "candidate_role_after": "qualified_for_mainline",
        "admission_allowed": True,
        "scope": "source_accession_only",
        "source_wide_phase1a_decision_unchanged": True,
        "frozen_source_ledger": _binding(ledger_path),
        "source_qualification_delta": _binding(delta_path),
        "canonical_block_manifest": _binding(manifest_path),
        "block_content_address": manifest["content_address"],
        "qc_grade": manifest["qc_grade"],
        "effective_sample_size": manifest["effective_sample_size"],
        "semantic_axes_complete": manifest["semantic_axes_complete"],
        "independence_status": "first_qualified_source_only; G2 pair requirement remains unmet",
        "remaining_g2_blockers": [
            "second_independent_disease_source_missing",
            "cell_type_region_atlas_first_block_missing",
            "compound_dose_time_perturbation_first_block_missing",
            "nested_split_freeze_pending",
            "real_domain_leakage_audit_pending",
        ],
        "license_release_checksum_evidence": input_bindings,
        "parser_mapping_provenance_evidence": {
            "parser": policy["parser"],
            "platform_equivalence": candidate["platform_equivalence"],
            "raw_value_origin": candidate["raw_value_path"],
        },
        "independence_graph_delta": {
            "node": f"geo::{candidate['source_accession']}",
            "mirror_group": candidate["mirror_group"],
            "relation": "same_study_mirror_with_arrayexpress_e_geod_28146",
            "independent_for_g2": False,
        },
        "role_transition_justification": "Official GEO release/reuse evidence, direct official value tables, complete sample semantics, official platform mapping, and Grade B QC support an accession-scoped role transition only.",
    }
    write_json(_path(str(policy["admission_audit_path"])), admission)
    return manifest, delta_row, admission


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the GSE28146 Phase 1E canonical mainline block.")
    parser.add_argument("--output", default="Output/audits/phase1e_gse28146_mainline_admission.json")
    args = parser.parse_args()
    manifest, _, admission = build(REPO_ROOT)
    requested = _path(args.output)
    configured = _path("Output/audits/phase1e_gse28146_mainline_admission.json")
    if requested != configured:
        write_json(requested, admission)
    print(
        json.dumps(
            {
                "status": admission["status"],
                "canonical_uid": manifest["canonical_uid"],
                "block_sha256": manifest["block_sha256"],
                "effective_sample_size": manifest["effective_sample_size"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
