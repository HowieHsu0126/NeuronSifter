"""Materialize the independent GSE37263 disease/aging mainline block."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping
from zipfile import ZipFile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_alignment_ready_registry import gene_reference_maps
from Libs.utils.data.public_foundation_evidence import canonical_sha256, sha256_file
from Libs.utils.data.public_foundation_roles import upsert_accession_requalification_delta
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic


BUILDER_ID = "Libs/scripts/builders/materialize_phase1f_gse37263_mainline_block.py"
MANIFEST_VERSION = "neuronsifter-phase1f-disease-pair-mainline-block-v1"
ADMISSION_VERSION = "neuronsifter-phase1f-disease-pair-admission-v1"


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


def _read_sdrf(path: Path, candidate: Mapping[str, object]) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    parsed: list[dict[str, object]] = []
    subject_pattern = re.compile(r"^(op[0-9]+)_(?:AD|CNT)\b", re.IGNORECASE)
    for index, row in enumerate(rows, start=1):
        source_name = str(row.get("Source Name", "")).strip()
        accession = source_name.split()[0]
        title = str(row.get("Comment [Sample_title]", "")).strip()
        subject_match = subject_pattern.match(title)
        if not subject_match:
            raise ValueError(f"Cannot recover source-native subject token from {title!r}")
        subject_token = subject_match.group(1).lower()
        disease_raw = str(row.get("Characteristics[disease status]", "")).strip()
        if disease_raw.casefold() == "control":
            disease = "normal_control"
        elif disease_raw.casefold() == "alzheimer's disease (ad)":
            disease = "Alzheimer disease"
        else:
            raise ValueError(f"Unexpected GSE37263 disease label: {disease_raw!r}")
        array_file = str(row.get("Derived Array Data File", "")).strip()
        biological_uid = f"arrayexpress::E-GEOD-37263::subject::{subject_token}"
        parsed.append(
            {
                "sample_index": index,
                "source_sample_accession": accession,
                "source_native_subject_token": subject_token,
                "biological_unit_id": biological_uid,
                "biological_unit_type": "subject_specimen",
                "subject_or_donor_id": subject_token,
                "subject_or_donor_status": "observed_source_native_subject_token",
                "sample_id": accession,
                "cell_id": "not_applicable_bulk_tissue",
                "replicate_id": "not_applicable_one_array_per_subject_specimen",
                "technical_replicate_group": biological_uid,
                "source_sample_title": title,
                "source_value_file": array_file,
                "sex": str(row.get("Characteristics[Sex]", "")).strip(),
                "age_years": float(str(row.get("Characteristics[age]", "")).strip()),
                "postmortem_interval_hours": float(str(row.get("Characteristics[pmi]", "")).strip()),
                "disease_state_original": disease_raw,
                "disease_state_canonical": disease,
                "tissue_original": "neocortex; source description specifies temporal cortex BA22",
                "brain_region_original": "temporal cortex BA22",
                "brain_region_canonical": "temporal lobe",
                "brain_region_ontology_id": "UBERON:0001871",
                "cell_type": "not_applicable_bulk_neocortex",
                "species": str(row.get("Characteristics[Organism]", "")).strip(),
                "species_taxonomy_id": "9606",
                "assay": "Affymetrix Human Exon 1.0 ST expression microarray",
                "platform_accession": str(row.get("Array Design REF", "")).strip(),
                "modality": "transcriptomics",
                "source_id": str(candidate["source_id"]),
                "source_accession": str(candidate["source_accession"]),
                "publication_accession_group": str(candidate["publication_accession_group"]),
                "mirror_group": str(candidate["mirror_group"]),
                "value_unit_original": "quantile-normalized log2 signal",
                "value_unit_canonical": "quantile-normalized log2 signal",
                "unit_conversion_formula": "identity",
                "time_status": "not_applicable_cross_sectional_post_mortem",
                "dose_status": "not_applicable_observational_tissue_study",
            }
        )
    frame = pd.DataFrame(parsed)
    if frame["source_native_subject_token"].duplicated().any():
        raise ValueError("GSE37263 source-native subject tokens are not unique")
    return frame


def _read_value_matrix(path: Path, samples: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    vectors: dict[str, tuple[list[str], np.ndarray]] = {}
    with ZipFile(path) as archive:
        for member in archive.infolist():
            with archive.open(member) as handle:
                header = handle.readline().decode("utf-8", errors="strict").rstrip("\r\n").split("\t")
                if header != ["ID_REF", "VALUE"]:
                    raise ValueError(f"Unexpected GSE37263 value header in {member.filename}: {header}")
                probe_ids: list[str] = []
                values: list[float] = []
                for raw in handle:
                    columns = raw.decode("utf-8", errors="strict").rstrip("\r\n").split("\t")
                    if len(columns) != 2:
                        raise ValueError(f"Malformed GSE37263 value row in {member.filename}")
                    probe_ids.append(columns[0].strip())
                    values.append(float(columns[1]))
            vectors[member.filename] = (probe_ids, np.asarray(values, dtype=np.float64))
    feature_order: list[str] | None = None
    matrices: list[np.ndarray] = []
    for source_file in samples["source_value_file"]:
        if source_file not in vectors:
            raise ValueError(f"SDRF value file absent from processed archive: {source_file}")
        probe_ids, values = vectors[source_file]
        if feature_order is None:
            feature_order = probe_ids
        elif feature_order != probe_ids:
            raise ValueError(f"GSE37263 feature order differs in {source_file}")
        matrices.append(values)
    if feature_order is None:
        raise ValueError("GSE37263 processed archive contains no value vectors")
    return feature_order, np.vstack(matrices)


def _platform_mapping(path: Path, requested_ids: set[str]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    in_table = False
    header: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            values = line.split("\t")
            if not header:
                header = values
                continue
            if values[0] in requested_ids:
                mapping[values[0]] = dict(zip(header, values))
    if set(mapping) != requested_ids:
        missing = sorted(requested_ids - set(mapping))[:10]
        raise ValueError(f"GPL5175 lacks exact GSE37263 feature IDs: {missing}")
    return mapping


def _unique_entrez_assignment(value: str) -> tuple[str, str, str]:
    pairs: list[tuple[str, str]] = []
    for chunk in value.split(" /// "):
        columns = [token.strip() for token in chunk.split(" // ")]
        if len(columns) >= 5 and columns[4].isdigit():
            pairs.append((columns[4], columns[1]))
    gene_ids = sorted({gene_id for gene_id, _ in pairs})
    if not gene_ids:
        return "", "", "unassigned_official_platform_feature"
    if len(gene_ids) > 1:
        return "", "", "multi_gene_official_platform_conflict"
    gene_id = gene_ids[0]
    symbols = sorted({symbol for candidate, symbol in pairs if candidate == gene_id and symbol})
    return gene_id, symbols[0] if symbols else "", "accepted_unique_entrez_assignment"


def _canonicalize(
    probe_ids: list[str], matrix: np.ndarray, platform: Mapping[str, Mapping[str, str]], config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    _, gene_id_to_symbol = gene_reference_maps(config)
    positions_by_gene: dict[str, list[int]] = defaultdict(list)
    source_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []
    for position, probe_id in enumerate(probe_ids):
        annotation = platform[probe_id]
        gene_id, source_symbol, status = _unique_entrez_assignment(annotation.get("gene_assignment", ""))
        if status == "accepted_unique_entrez_assignment" and gene_id not in gene_id_to_symbol:
            status = "entrez_id_absent_from_canonical_gene_backbone"
        row = {
            "source_feature_id": probe_id,
            "canonical_gene_id": gene_id if status == "accepted_unique_entrez_assignment" else "",
            "source_gene_symbol": source_symbol,
            "canonical_gene_symbol": gene_id_to_symbol.get(gene_id, "") if status == "accepted_unique_entrez_assignment" else "",
            "mapping_status": status,
            "mapping_method": "official_gpl5175_exact_feature_join_unique_entrez_assignment",
            "mapping_confidence": "exact" if status == "accepted_unique_entrez_assignment" else "quarantined",
        }
        if status == "accepted_unique_entrez_assignment":
            positions_by_gene[gene_id].append(position)
            source_rows.append(row)
        else:
            quarantine_rows.append(row)
    ordered_genes = sorted(positions_by_gene, key=lambda gene: min(positions_by_gene[gene]))
    canonical = np.empty((matrix.shape[0], len(ordered_genes)), dtype=np.float64)
    features: list[dict[str, object]] = []
    for feature_index, gene_id in enumerate(ordered_genes, start=1):
        positions = positions_by_gene[gene_id]
        canonical[:, feature_index - 1] = np.nanmedian(matrix[:, positions], axis=1)
        features.append(
            {
                "feature_index": feature_index,
                "canonical_gene_id": gene_id,
                "canonical_gene_symbol": gene_id_to_symbol[gene_id],
                "canonical_namespace": "entrez_gene_id",
                "source_feature_ids": "|".join(probe_ids[position] for position in positions),
                "source_feature_count": len(positions),
                "mapping_method": "official_gpl5175_exact_feature_join_unique_entrez_assignment",
                "mapping_confidence": "exact",
                "aggregation_rule": "median_across_unique_gene_assigned_transcript_clusters",
            }
        )
    return pd.DataFrame(features), pd.DataFrame(source_rows), pd.DataFrame(quarantine_rows), canonical


def _value_rows(samples: pd.DataFrame, features: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    if values.shape != (len(samples), len(features)):
        raise ValueError("Canonical GSE37263 matrix dimensions are inconsistent")
    return pd.DataFrame(
        {
            "biological_unit_id": np.repeat(samples["biological_unit_id"].to_numpy(), len(features)),
            "source_sample_accession": np.repeat(samples["source_sample_accession"].to_numpy(), len(features)),
            "canonical_gene_id": np.tile(features["canonical_gene_id"].to_numpy(), len(samples)),
            "value": values.reshape(values.size),
            "value_unit_original": "quantile-normalized log2 signal",
            "value_unit_canonical": "quantile-normalized log2 signal",
            "unit_conversion_formula": "identity",
            "source_value_kind": "official_arrayexpress_gene_level_processed_matrix",
        }
    )


def _pair_independence_evidence(policy: Mapping[str, object], candidate: Mapping[str, object], samples: pd.DataFrame) -> dict[str, object]:
    reference = policy["phase1f_pair_reference"]
    if not isinstance(reference, Mapping):
        raise TypeError("phase1f_pair_reference must be an object")
    admission = _load_json(_path(str(reference["admission_audit_path"])))
    binding = admission.get("canonical_block_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError("Disease M1 admission lacks canonical manifest binding")
    manifest_path = _path(str(binding.get("path", "")))
    if _binding(manifest_path) != dict(binding):
        raise ValueError("Disease M1 manifest binding is stale")
    manifest = _load_json(manifest_path)
    for field in ("canonical_uid", "source_id", "source_accession", "publication_accession_group", "mirror_group"):
        expected = reference[f"expected_{field}"]
        if manifest.get(field) != expected:
            raise ValueError(f"Disease M1 reference mismatch for {field}")
    identity_fields = ("source_id", "source_accession", "publication_accession_group", "mirror_group")
    conflicts = [field for field in identity_fields if str(manifest[field]) == str(candidate[field])]
    m1_samples = pd.read_parquet(manifest_path.parent / str(manifest["sample_metadata_path"]))
    m1_sample_tokens = set(m1_samples["source_sample_accession"].astype(str))
    m2_sample_tokens = set(samples["source_sample_accession"].astype(str))
    m1_biological = set(m1_samples["biological_unit_id"].astype(str))
    m2_biological = set(samples["biological_unit_id"].astype(str))
    sample_overlap = sorted(m1_sample_tokens & m2_sample_tokens)
    biological_overlap = sorted(m1_biological & m2_biological)
    if conflicts or sample_overlap or biological_overlap:
        raise ValueError(
            "Disease M1/M2 independence contract failed: "
            f"identity={conflicts}, sample_overlap={sample_overlap[:5]}, biological_overlap={biological_overlap[:5]}"
        )
    return {
        "status": "passed",
        "m1_manifest": _binding(manifest_path),
        "identity_equal_fields": [],
        "source_sample_token_overlap_count": 0,
        "biological_unit_overlap_count": 0,
        "m1_observation_count": int(manifest["observation_count"]),
        "m2_observation_count": int(len(samples)),
    }


def _block_hash(records: list[dict[str, object]]) -> str:
    return canonical_sha256([{"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in records])


def _file_records(root: Path, names: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"name": name, **_binding(root / name)} for name in names]


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object]]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1f_disease_pair_materialization"]
    candidate = policy["candidate"]
    samples = _read_sdrf(_path(candidate["sample_metadata_path"]), candidate)
    probe_ids, raw_matrix = _read_value_matrix(_path(candidate["raw_value_path"]), samples)
    platform = _platform_mapping(_path(candidate["platform_annotation_path"]), set(probe_ids))
    features, mapping, quarantine, canonical = _canonicalize(probe_ids, raw_matrix, platform, config)
    values = _value_rows(samples, features, canonical)
    independence = _pair_independence_evidence(policy, candidate, samples)
    qc = policy["qc"]
    reason_counts = quarantine["mapping_status"].value_counts().to_dict()
    checks = {
        "sample_count": len(samples) == int(qc["expected_sample_count"]),
        "source_feature_count": len(probe_ids) == int(qc["expected_source_feature_count"]),
        "unique_gene_assignment_count": (
            len(mapping) + reason_counts.get("entrez_id_absent_from_canonical_gene_backbone", 0)
            == int(qc["expected_unique_gene_assignment_count"])
        ),
        "backbone_missing_quarantine_count": reason_counts.get("entrez_id_absent_from_canonical_gene_backbone", 0) == int(qc["expected_backbone_missing_quarantine_count"]),
        "accepted_source_mapping_count": len(mapping) == int(qc["expected_accepted_source_mapping_count"]),
        "canonical_gene_count": len(features) == int(qc["expected_canonical_gene_count"]),
        "multi_gene_quarantine_count": reason_counts.get("multi_gene_official_platform_conflict", 0) == int(qc["expected_multi_gene_quarantine_count"]),
        "unassigned_quarantine_count": reason_counts.get("unassigned_official_platform_feature", 0) == int(qc["expected_unassigned_quarantine_count"]),
        "mapped_gene_count": len(features) >= int(qc["minimum_mapped_gene_count"]),
        "finite_fraction": float(np.isfinite(canonical).mean()) >= float(qc["minimum_finite_fraction"]),
        "disease_balance": samples["disease_state_canonical"].value_counts().to_dict() == {"normal_control": 8, "Alzheimer disease": 8},
    }
    if not all(checks.values()):
        raise ValueError(
            "GSE37263 QC contract failed: "
            f"checks={checks}, accepted_mapping_count={len(mapping)}, "
            f"canonical_gene_count={len(features)}, quarantine_counts={reason_counts}"
        )

    output_root = _path(policy["output_root"])
    staging_root = output_root / ".phase1f_disease_staging"
    ensure_directory(staging_root)
    names = ("values.parquet", "sample_metadata.parquet", "feature_metadata.parquet", "source_feature_mapping.parquet", "feature_quarantine.parquet")
    with tempfile.TemporaryDirectory(prefix="gse37263-", dir=staging_root) as temporary:
        stage = Path(temporary)
        for frame, name in zip((values, samples, features, mapping, quarantine), names, strict=True):
            write_parquet_atomic(frame, stage / name)
        records = _file_records(stage, names)
        block_sha256 = _block_hash(records)
        destination = output_root / "blocks" / candidate["source_id"] / candidate["source_accession"] / block_sha256[:16]
        if destination.exists():
            existing = _load_json(destination / "block_manifest.json")
            if existing.get("block_sha256") != block_sha256:
                raise ValueError(f"Content-address collision at {destination}")
        else:
            ensure_directory(destination.parent)
            stage.replace(destination)

    records = _file_records(destination, names)
    bindings = [_binding(_path(candidate[key])) for key in ("raw_value_path", "sample_metadata_path", "study_metadata_path", "platform_annotation_path", "terms_snapshot_path")]
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "canonical_uid": "pfc_arrayexpress_e_geod_37263_phase1f_v1",
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
        "source_feature_mapping_path": "source_feature_mapping.parquet",
        "feature_quarantine_path": "feature_quarantine.parquet",
        "observation_count": int(len(samples)),
        "value_row_count": int(len(values)),
        "biological_unit_count": int(samples["source_native_subject_token"].nunique()),
        "effective_sample_size": float(samples["source_native_subject_token"].nunique()),
        "effective_sample_size_formula": policy["biological_unit_policy"]["effective_sample_size_formula"],
        "biological_unit_policy": policy["biological_unit_policy"],
        "qc_grade": qc["grade"],
        "qc_grade_rationale": qc["grade_rationale"],
        "qc": {
            **{key: bool(value) for key, value in checks.items()},
            "mapped_gene_count": int(len(features)),
            "accepted_source_feature_count": int(len(mapping)),
            "quarantined_source_feature_count": int(len(quarantine)),
            "quarantine_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
            "status": "passed",
        },
        "mapping_confidence": "official_platform_exact_unique_entrez_assignment",
        "mapping_version": policy["parser"]["mapping_version"],
        "unit_mapping_version": policy["parser"]["unit_mapping_version"],
        "ontology_versions": {"species": "NCBI_Taxonomy:9606", "brain_region": "UBERON:0001871", "gene": "GPL5175_build32_Entrez_plus_canonical_backbone"},
        "semantic_axes": {
            "study": {"status": "observed", "value": "E-GEOD-37263 / GSE37263"},
            "biological_unit": {"status": "observed", "value": "source-native subject specimen"},
            "species": {"status": "mapped", "value": "Homo sapiens; NCBI:txid:9606"},
            "disease_state": {"status": "observed", "value": "Alzheimer disease or control"},
            "tissue": {"status": "observed", "value": "neocortex; temporal cortex BA22 in study description"},
            "brain_region": {"status": "mapped", "value": "temporal lobe; UBERON:0001871; no MTG inference"},
            "cell_type": {"status": "not_applicable", "reason": "bulk neocortex specimen has no cell-level axis"},
            "assay": {"status": "observed", "value": "Affymetrix Human Exon 1.0 ST; GPL5175"},
            "modality": {"status": "observed", "value": "transcriptomics"},
            "feature": {"status": "mapped", "value": "unique Entrez Gene assignment from official GPL5175"},
            "time": {"status": "not_applicable", "reason": "cross-sectional post-mortem study"},
            "dose": {"status": "not_applicable", "reason": "observational tissue study"},
            "unit": {"status": "mapped", "value": policy["parser"]["unit_mapping_version"]},
            "provenance": {"status": "observed", "value": "arrayexpress:E-GEOD-37263; geo:GSE37263 mirror"},
        },
        "semantic_axes_complete": True,
        "split_role": "unassigned_pending_nested_split_freeze",
        "source_release": {
            "official_url": candidate["official_study_url"], "study_release_date": candidate["study_release_date"],
            "publication_pubmed_id": candidate["publication_pubmed_id"], "publication_doi": candidate["publication_doi"],
            "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
            "access_class": "public_with_attribution", "reuse_evidence": "BioStudies ArrayExpress provides public data for reuse by the research community.",
        },
        "pair_independence": independence,
        "parser": policy["parser"],
        "input_bindings": bindings,
        "block_files": records,
        "blocker_codes": ["atlas_second_independent_source_missing", "perturbation_second_independent_source_missing", "nested_split_freeze_pending"],
    }
    manifest_path = destination / "block_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    else:
        manifest = _load_json(manifest_path)

    ledger_path = _path(policy["source_ledger_path"])
    delta_path = _path(policy["qualification_delta_path"])
    delta_row = {
        "delta_version": "phase1f-source-qualification-delta-v1", "entry_id": "phase1f_arrayexpress_e_geod_37263_mainline",
        "source_id": candidate["source_id"], "source_accession": candidate["source_accession"], "prior_decision": "deferred",
        "decision": "qualified_for_mainline", "admission_allowed": "true", "scope": "source_accession_only",
        "frozen_ledger_path": _relative(ledger_path), "frozen_ledger_sha256": sha256_file(ledger_path),
        "block_manifest_path": _relative(manifest_path), "block_manifest_sha256": sha256_file(manifest_path),
        "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
        "access_class": "public_with_attribution", "status": "accepted",
    }
    upsert_accession_requalification_delta(delta_path, delta_row)
    admission = {
        "schema_version": ADMISSION_VERSION, "producer": BUILDER_ID, "created_at": _now(),
        "status": "accepted_second_independent_block", "candidate_id": "disease_arrayexpress_e_geod_37263",
        "candidate_role_before": "deferred_source_not_consumed", "candidate_role_after": "qualified_for_mainline",
        "admission_allowed": True, "scope": "source_accession_only", "source_wide_phase1a_decision_unchanged": True,
        "frozen_source_ledger": _binding(ledger_path), "source_qualification_delta": _binding(delta_path),
        "canonical_block_manifest": _binding(manifest_path), "block_content_address": manifest["content_address"],
        "qc_grade": manifest["qc_grade"], "effective_sample_size": manifest["effective_sample_size"],
        "semantic_axes_complete": True, "independence_status": "passed_independent_pair_with_gse28146",
        "pair_independence": independence,
        "remaining_g2_blockers": ["atlas_second_independent_source_missing", "perturbation_second_independent_source_missing", "nested_split_freeze_pending", "real_domain_leakage_audit_pending"],
        "license_release_checksum_evidence": bindings,
        "parser_mapping_provenance_evidence": {"parser": policy["parser"], "platform_accession": candidate["platform_accession"], "raw_value_origin": candidate["raw_value_path"]},
        "independence_graph_delta": {"node": "arrayexpress::E-GEOD-37263", "mirror_group": candidate["mirror_group"], "relation": "independent_disease_pair_with_geo_gse28146", "independent_for_g2": True},
        "role_transition_justification": "Official gene-level values, exact official GPL5175 mapping, complete subject/CNS metadata, Grade B QC, and zero M1 identity overlap support accession-scoped re-admission.",
    }
    write_json(_path(policy["admission_audit_path"]), admission)
    return manifest, admission


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the independent GSE37263 disease block.")
    parser.add_argument("--output", default="Output/audits/phase1f_gse37263_mainline_admission.json")
    args = parser.parse_args()
    _, admission = build(REPO_ROOT)
    requested = _path(args.output)
    configured = _path("Output/audits/phase1f_gse37263_mainline_admission.json")
    if requested != configured:
        write_json(requested, admission)
    print(json.dumps(admission, sort_keys=True))


if __name__ == "__main__":
    main()
