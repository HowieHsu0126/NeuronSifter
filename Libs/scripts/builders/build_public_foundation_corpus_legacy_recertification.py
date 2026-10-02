from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import read_tsv, write_json, write_tsv


REPORT_PATH = REPO_ROOT / "Output/audits/public_foundation_corpus_legacy_recertification.json"
LEDGER_PATH = REPO_ROOT / "Output/audits/public_foundation_corpus_legacy_recertification.tsv"

STUDY_CATALOG = "Input/processed/model_input/study_bundle_catalog.tsv"
ALIGNED_CATALOG = "Input/processed/alignment_ready/aligned_matrix_catalog.tsv"
OBSERVATION_CATALOG = "Input/processed/sample_observation/sample_observation_catalog.tsv"
SAMPLE_CATALOG = "Input/processed/cohort_ready/sample_catalog.tsv"
ISOLATION_AUDIT = "Output/audits/public_mainline_isolation_audit.json"

EXPECTED_BUNDLE_COUNT = 13
EXPECTED_BUNDLE_SAMPLE_COUNT = 862
EXPECTED_ALIGNED_BUNDLE_COUNT = 5
EXPECTED_ALIGNED_SAMPLE_COUNT = 171
EXPECTED_ACTION_COUNTS = {
    "regimen_catalog": 13_896,
    "drug_program_catalog": 169_024,
    "compound_pk_prior_catalog": 507_867,
}

BUNDLE_FILE_FIELDS = (
    "bundle_manifest_path",
    "sample_index_path",
    "feature_index_path",
    "matrix_raw_path",
    "matrix_norm_path",
    "scaler_artifact_path",
)
ALIGNED_FILE_FIELDS = (
    "manifest_path",
    "sample_index_path",
    "entity_index_path",
    "matrix_raw_path",
    "matrix_norm_path",
)

EXTERNAL_SOURCE_SPECS = (
    {
        "source_id": "allen_brain_cell_atlas",
        "source_manifest": "Input/raw/allen_brain_cell_atlas/manifests/source_manifest.json",
        "checksum_ledger": "Input/raw/allen_brain_cell_atlas/checksums/sha256.tsv",
        "terms_snapshot": "Input/raw/allen_brain_cell_atlas/terms/allen_terms_snapshot.html",
        "anchor": "downloads/releases/20250531/SEAAD/precomputed_stats.20231120.sea_ad.MTG.h5",
        "candidate_family": "cns_cell_type_region_atlas",
        "intended_role": "core_candidate_after_requalification",
    },
    {
        "source_id": "biccn_public",
        "source_manifest": "Input/raw/biccn_public/manifests/source_manifest.json",
        "checksum_ledger": "Input/raw/biccn_public/checksums/sha256.tsv",
        "terms_snapshot": None,
        "anchor": "downloads/releases/public/biccn_data_page.html",
        "candidate_family": "cns_cell_type_region_atlas",
        "intended_role": "core_candidate_after_native_matrix_materialization",
    },
    {
        "source_id": "clue_lincs",
        "source_manifest": "Input/raw/clue_lincs/manifests/source_manifest.json",
        "checksum_ledger": "Input/raw/clue_lincs/checksums/sha256.tsv",
        "terms_snapshot": "Input/raw/clue_lincs/terms/clue_terms_snapshot.html",
        "anchor": "downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        "candidate_family": "compound_dose_time_perturbation",
        "intended_role": "core_candidate_for_qualified_cns_subset_only",
    },
    {
        "source_id": "drugcomb",
        "source_manifest": "Input/raw/drugcomb/manifests/source_manifest.json",
        "checksum_ledger": "Input/raw/drugcomb/checksums/sha256.tsv",
        "terms_snapshot": "Input/raw/drugcomb/terms/drugcomb_publication_terms_snapshot.html",
        "anchor": "downloads/releases/zenodo/18449193/drugcomb_data_v1.4.csv",
        "candidate_family": "auxiliary_pharmacology",
        "intended_role": "weak_calibration_after_requalification",
    },
    {
        "source_id": "bindingdb",
        "source_manifest": "Input/raw/bindingdb/manifests/source_manifest.json",
        "checksum_ledger": "Input/raw/bindingdb/checksums/sha256.tsv",
        "terms_snapshot": "Input/raw/bindingdb/terms/bindingdb_home_snapshot.html",
        "anchor": "downloads/releases/BindingDB_All_202604_tsv.zip",
        "candidate_family": "auxiliary_pharmacology",
        "intended_role": "context_prior_after_requalification",
    },
)

ACTION_SPECS = (
    ("regimen_catalog", "Input/processed/action_ready/regimen_catalog.tsv"),
    ("drug_program_catalog", "Input/processed/action_ready/drug_program_catalog.tsv"),
    ("compound_pk_prior_catalog", "Input/processed/pkpd_ready/compound_pk_prior_catalog.tsv"),
)

TEACHER_SPECS = (
    ("mesoscopic_teacher_catalog", "Input/processed/simulation_teacher_ready/simulation_teacher_catalog.tsv"),
    ("full_microenvironment_teacher_catalog", "Input/processed/full_microenvironment_teacher_ready/full_microenvironment_teacher_catalog.tsv"),
    ("pbpk_rollout_catalog", "Input/processed/full_microenvironment_teacher_ready/pbpk_rollout_catalog.tsv"),
    ("simulation_qc_catalog", "Input/processed/simulation_audit/simulation_qc_catalog.tsv"),
)

LEDGER_FIELDS = [
    "asset_id",
    "asset_class",
    "source_id",
    "source_accession",
    "modality",
    "record_count",
    "sample_count",
    "candidate_core_family",
    "current_disposition",
    "core_g2_eligible",
    "bounded_file_count",
    "bounded_bytes",
    "binding_sha256",
    "issue_codes",
]


def _repo_path(repo_root: Path, value: str) -> Path:
    root = repo_root.resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes repository root: {value}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_and_count_rows(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    newline_count = 0
    final_byte = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            final_byte = chunk[-1:]
    line_count = newline_count + (1 if path.stat().st_size and final_byte != b"\n" else 0)
    return digest.hexdigest(), max(0, line_count - 1), path.stat().st_size


def _bind_file(repo_root: Path, relative_path: str) -> dict[str, object]:
    path = _repo_path(repo_root, relative_path)
    if not path.is_file():
        return {
            "path": relative_path,
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
            "verification_mode": "missing",
        }
    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "verification_mode": "content_sha256",
    }


def _binding_digest(bindings: Iterable[dict[str, object]]) -> str:
    payload = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
            "verification_mode": row["verification_mode"],
        }
        for row in bindings
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _issue(blockers: list[dict[str, object]], code: str, scope: str, count: int, message: str) -> None:
    blockers.append({"code": code, "scope": scope, "count": count, "message": message})


def analyze_sample_semantics(
    observation_rows: list[dict[str, str]],
    cohort_rows: list[dict[str, str]],
    study_uids: set[str],
) -> dict[str, object]:
    observations = [row for row in observation_rows if row.get("study_uid") in study_uids]
    samples = [row for row in cohort_rows if row.get("study_uid") in study_uids]
    whole_blood_brain = [
        row
        for row in observations
        if row.get("tissue_or_biofluid", "").strip().lower() in {"blood", "whole blood"}
        and row.get("region_scope", "").strip().lower()
        not in {"", "unknown", "not_applicable", "systemic", "systemic_circulation"}
    ]
    cns_metabolomics_systemic = [
        row
        for row in observations
        if row.get("modality", "").strip().lower() == "metabolomics"
        and row.get("tissue_or_biofluid", "").strip().lower()
        in {"brain", "brain tissue", "csf", "cerebrospinal fluid"}
        and row.get("region_scope", "").strip().lower() in {"systemic", "systemic_circulation"}
    ]
    ad_case_control = [
        row
        for row in samples
        if "alzheimer" in row.get("sample_diagnosis_label", "").lower()
        and row.get("case_control_label", "").strip().lower() != "case"
    ]
    longitudinal_claims = [
        row
        for row in observations
        if "longitud" in " ".join(
            [row.get("observation_role", ""), row.get("note", "")]
        ).lower()
    ]
    per_accession: dict[str, Counter[str]] = defaultdict(Counter)
    for row in whole_blood_brain:
        per_accession[row.get("source_accession", "")]["whole_blood_to_brain_region"] += 1
    for row in cns_metabolomics_systemic:
        per_accession[row.get("source_accession", "")]["cns_metabolomics_to_systemic"] += 1
    for row in ad_case_control:
        per_accession[row.get("source_accession", "")]["ad_case_control_inconsistency"] += 1
    for row in longitudinal_claims:
        per_accession[row.get("source_accession", "")]["unsupported_longitudinal_claim"] += 1

    e6613_observations = [row for row in observations if row.get("source_accession") == "E-GEOD-6613"]
    e6613_samples = [row for row in samples if row.get("source_accession") == "E-GEOD-6613"]
    e6613_pd_program_count = sum(row.get("disease_program") == "pd_control" for row in e6613_observations)
    e6613_bad_study_label_count = sum(
        "alzheimer" in row.get("study_diagnosis_label", "").lower() for row in e6613_samples
    )
    if e6613_observations and e6613_pd_program_count != len(e6613_observations):
        per_accession["E-GEOD-6613"]["pd_program_mismatch"] += len(e6613_observations) - e6613_pd_program_count
    if e6613_bad_study_label_count:
        per_accession["E-GEOD-6613"]["pd_study_mislabeled_as_ad"] += e6613_bad_study_label_count

    return {
        "observation_row_count": len(observations),
        "cohort_sample_row_count": len(samples),
        "whole_blood_to_brain_region_count": len(whole_blood_brain),
        "cns_metabolomics_to_systemic_count": len(cns_metabolomics_systemic),
        "ad_case_control_inconsistency_count": len(ad_case_control),
        "unsupported_longitudinal_claim_count": len(longitudinal_claims),
        "longitudinal_evidence_axis_present": False,
        "longitudinal_eligible_record_count": 0,
        "e_geod_6613": {
            "observation_count": len(e6613_observations),
            "pd_control_observation_count": e6613_pd_program_count,
            "incorrect_ad_study_label_sample_count": e6613_bad_study_label_count,
            "required_canonical_program": "pd_control",
            "current_disposition": "quarantine_semantic_conflict",
        },
        "issue_counts_by_accession": {
            accession: dict(sorted(counts.items()))
            for accession, counts in sorted(per_accession.items())
        },
    }


def _build_bundle_inventory(
    repo_root: Path, blockers: list[dict[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    catalog_paths = [STUDY_CATALOG, ALIGNED_CATALOG, OBSERVATION_CATALOG, SAMPLE_CATALOG]
    catalog_bindings = [_bind_file(repo_root, value) for value in catalog_paths]
    missing_catalogs = [row["path"] for row in catalog_bindings if not row["exists"]]
    if missing_catalogs:
        _issue(blockers, "legacy_catalog_missing", "legacy_bundles", len(missing_catalogs), str(missing_catalogs))
        return {"catalog_bindings": catalog_bindings}, [], []

    bundles = read_tsv(_repo_path(repo_root, STUDY_CATALOG))
    aligned_rows = read_tsv(_repo_path(repo_root, ALIGNED_CATALOG))
    observation_rows = read_tsv(_repo_path(repo_root, OBSERVATION_CATALOG))
    cohort_rows = read_tsv(_repo_path(repo_root, SAMPLE_CATALOG))
    aligned_by_uid = {row["study_uid"]: row for row in aligned_rows}
    study_uids = {row["study_uid"] for row in bundles}
    semantics = analyze_sample_semantics(observation_rows, cohort_rows, study_uids)
    issue_counts = semantics["issue_counts_by_accession"]

    bundle_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for row in bundles:
        accession = row["source_accession"]
        modality = "transcriptomics" if row.get("matrix_semantics") == "expression_signal" else "metabolomics"
        bindings = [_bind_file(repo_root, row[field]) for field in BUNDLE_FILE_FIELDS]
        aligned = aligned_by_uid.get(row["study_uid"])
        aligned_bindings: list[dict[str, object]] = []
        if aligned:
            aligned_bindings = [_bind_file(repo_root, aligned[field]) for field in ALIGNED_FILE_FIELDS]
        missing = [binding["path"] for binding in bindings + aligned_bindings if not binding["exists"]]
        issues = set(issue_counts.get(accession, {}).keys())
        if missing:
            issues.add("listed_bundle_file_missing")
        if modality == "metabolomics":
            issues.add("metabolite_crosswalk_missing")
            disposition = "deferred_auxiliary_quarantine"
            candidate_family = ""
        elif accession == "E-GEOD-6613":
            disposition = "quarantine_semantic_conflict"
            candidate_family = "cns_disease_aging_state"
        else:
            disposition = "quarantine_pending_source_and_semantic_recertification"
            candidate_family = "cns_disease_aging_state"
            if not aligned:
                issues.add("gene_alignment_missing")
        all_bindings = bindings + aligned_bindings
        bundle_entry = {
            "study_uid": row["study_uid"],
            "source_id": row["source_key"],
            "source_accession": accession,
            "modality": modality,
            "sample_count": int(row["sample_count"]),
            "feature_count": int(row["feature_count"]),
            "gene_alignment_status": "aligned" if aligned else "not_aligned",
            "aligned_sample_count": int(aligned["sample_count"]) if aligned else 0,
            "candidate_core_family": candidate_family,
            "current_disposition": disposition,
            "core_g2_eligible": False,
            "counts_toward_core_ess": False,
            "issue_codes": sorted(issues),
            "semantic_issue_counts": issue_counts.get(accession, {}),
            "file_bindings": bindings,
            "aligned_file_bindings": aligned_bindings,
            "bounded_binding_sha256": _binding_digest(all_bindings),
        }
        bundle_rows.append(bundle_entry)
        ledger_rows.append(
            {
                "asset_id": row["study_uid"],
                "asset_class": "legacy_study_bundle",
                "source_id": row["source_key"],
                "source_accession": accession,
                "modality": modality,
                "record_count": "",
                "sample_count": row["sample_count"],
                "candidate_core_family": candidate_family,
                "current_disposition": disposition,
                "core_g2_eligible": "no",
                "bounded_file_count": len(all_bindings),
                "bounded_bytes": sum(int(binding["size_bytes"]) for binding in all_bindings),
                "binding_sha256": bundle_entry["bounded_binding_sha256"],
                "issue_codes": "|".join(sorted(issues)),
            }
        )

    bundle_count = len(bundle_rows)
    sample_count = sum(int(row["sample_count"]) for row in bundle_rows)
    aligned_count = sum(row["gene_alignment_status"] == "aligned" for row in bundle_rows)
    aligned_samples = sum(int(row["aligned_sample_count"]) for row in bundle_rows)
    if bundle_count != EXPECTED_BUNDLE_COUNT:
        _issue(blockers, "legacy_bundle_count_mismatch", "legacy_bundles", bundle_count, f"expected {EXPECTED_BUNDLE_COUNT}")
    if sample_count != EXPECTED_BUNDLE_SAMPLE_COUNT:
        _issue(blockers, "legacy_bundle_sample_count_mismatch", "legacy_bundles", sample_count, f"expected {EXPECTED_BUNDLE_SAMPLE_COUNT}")
    if aligned_count != EXPECTED_ALIGNED_BUNDLE_COUNT or aligned_samples != EXPECTED_ALIGNED_SAMPLE_COUNT:
        _issue(
            blockers,
            "legacy_alignment_count_mismatch",
            "legacy_bundles",
            aligned_count,
            f"expected {EXPECTED_ALIGNED_BUNDLE_COUNT} bundles/{EXPECTED_ALIGNED_SAMPLE_COUNT} samples",
        )
    missing_files = sum(
        not binding["exists"]
        for bundle in bundle_rows
        for binding in bundle["file_bindings"] + bundle["aligned_file_bindings"]
    )
    if missing_files:
        _issue(blockers, "listed_bundle_file_missing", "legacy_bundles", missing_files, "One or more catalog-bound files are absent")

    transcriptomic = [row for row in bundle_rows if row["modality"] == "transcriptomics"]
    metabolomic = [row for row in bundle_rows if row["modality"] == "metabolomics"]
    summary = {
        "catalog_bindings": catalog_bindings,
        "bundle_count": bundle_count,
        "sample_count": sample_count,
        "transcriptomic_bundle_count": len(transcriptomic),
        "transcriptomic_sample_count": sum(int(row["sample_count"]) for row in transcriptomic),
        "metabolomics_bundle_count": len(metabolomic),
        "metabolomics_sample_count": sum(int(row["sample_count"]) for row in metabolomic),
        "aligned_transcriptomic_bundle_count": aligned_count,
        "aligned_transcriptomic_sample_count": aligned_samples,
        "qualified_core_bundle_count": 0,
        "qualified_core_sample_count": 0,
        "paired_crossmodal_biological_unit_count": 0,
        "semantic_audit": semantics,
        "bundles": bundle_rows,
    }
    return summary, bundle_rows, ledger_rows


def _checksum_anchor_binding(
    repo_root: Path, source_manifest_path: str, checksum_path: str, anchor: str
) -> dict[str, object]:
    manifest = json.loads(_repo_path(repo_root, source_manifest_path).read_text(encoding="utf-8"))
    local_root = _repo_path(repo_root, str(manifest["local_root"]))
    rows = read_tsv(_repo_path(repo_root, checksum_path))
    matching = [row for row in rows if row.get("relative_path") == anchor]
    anchor_path = (local_root / anchor).resolve()
    if local_root.resolve() not in anchor_path.parents:
        raise ValueError(f"external anchor escapes local root: {anchor}")
    declared_sha = matching[0].get("sha256") if len(matching) == 1 else None
    declared_size = int(matching[0]["size_bytes"]) if len(matching) == 1 else None
    exists = anchor_path.is_file()
    size_matches = bool(exists and declared_size == anchor_path.stat().st_size)
    declared_sha_valid = bool(declared_sha and re.fullmatch(r"[0-9a-f]{64}", declared_sha))
    return {
        "relative_path": anchor,
        "exists": exists,
        "declared_size_bytes": declared_size,
        "actual_size_bytes": anchor_path.stat().st_size if exists else 0,
        "declared_sha256": declared_sha,
        "declared_sha256_valid": declared_sha_valid,
        "size_matches": size_matches,
        "verification_mode": "declared_sha256_bound_to_content_hashed_checksum_ledger_plus_size",
        "content_rehashed_by_bounded_audit": False,
        "binding_status": "declared_checksum_bound" if size_matches and declared_sha_valid else "invalid",
    }


def _external_value_summary(
    repo_root: Path, source_id: str, anchor: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if source_id == "allen_brain_cell_atlas":
        evidence_paths = [
            "Input/processed/atlas_prior/allen_biccn/source_catalog.tsv",
            "Input/processed/atlas_prior/allen_biccn/program_catalog.tsv",
        ]
        evidence_bindings = [_bind_file(repo_root, path) for path in evidence_paths]
        rows = read_tsv(_repo_path(repo_root, evidence_paths[1]))
        count = sum(row.get("atlas_name") == "ALLEN_ABC" for row in rows)
        return {
            "aggregate_program_count": count,
            "native_observation_block_eligible": False,
        }, evidence_bindings
    if source_id == "biccn_public":
        evidence_paths = [
            "Input/processed/atlas_prior/allen_biccn/source_catalog.tsv",
            "Input/processed/atlas_prior/allen_biccn/program_catalog.tsv",
        ]
        evidence_bindings = [_bind_file(repo_root, path) for path in evidence_paths]
        rows = read_tsv(_repo_path(repo_root, evidence_paths[1]))
        count = sum(row.get("atlas_name") == "BICCN_PUBLIC" for row in rows)
        return {
            "aggregate_program_count": count,
            "native_observation_block_eligible": False,
        }, evidence_bindings
    if source_id == "clue_lincs":
        evidence_paths = [
            "Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.h5dump.txt",
            "Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_sig_info.txt.gz",
            "Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_pert_info.txt.gz",
        ]
        evidence_bindings = [_bind_file(repo_root, path) for path in evidence_paths]
        match = re.search(r"n(\d+)x(\d+)", anchor)
        shape = [int(match.group(1)), int(match.group(2))] if match else None
        return {
            "level5_matrix_shape": shape,
            "real_numeric_asset_present": shape is not None,
            "qualified_cns_signature_count": 0,
        }, evidence_bindings
    if source_id == "drugcomb":
        manifest_path = "Input/processed/external_adapters/drugcomb_dose_response_manifest.json"
        prior_path = "Input/processed/external_adapters/drugcomb_combination_prior_catalog.tsv"
        manifest_binding = _bind_file(repo_root, manifest_path)
        prior_sha, prior_count, prior_bytes = hash_and_count_rows(_repo_path(repo_root, prior_path))
        evidence_bindings = [
            manifest_binding,
            {
                "path": prior_path,
                "exists": True,
                "size_bytes": prior_bytes,
                "sha256": prior_sha,
                "verification_mode": "content_sha256",
            },
        ]
        manifest = json.loads(
            _repo_path(repo_root, manifest_path).read_text(encoding="utf-8")
        )
        return {
            "dose_response_row_count": int(manifest["dose_response_row_count"]),
            "combination_prior_count": prior_count,
            "core_transcriptomics_observation_eligible": False,
        }, evidence_bindings
    if source_id == "bindingdb":
        edge_path = "Input/processed/drug_target_ready/drug_target_edge_catalog.tsv"
        count = 0
        with _repo_path(repo_root, edge_path).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("source_repository") == "bindingdb" and row.get("affinity_unit") == "nM":
                    count += 1
        evidence_bindings = [_bind_file(repo_root, edge_path)]
        return {
            "measured_nm_affinity_edge_count": count,
            "observation_corpus_eligible": False,
        }, evidence_bindings
    raise ValueError(source_id)


def _build_external_sources(
    repo_root: Path, blockers: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for spec in EXTERNAL_SOURCE_SPECS:
        governance_paths = [spec["source_manifest"], spec["checksum_ledger"]]
        if spec["terms_snapshot"]:
            governance_paths.append(spec["terms_snapshot"])
        bindings = [_bind_file(repo_root, str(path)) for path in governance_paths]
        required_missing = [binding["path"] for binding in bindings if not binding["exists"]]
        if required_missing:
            anchor_binding = {
                "relative_path": spec["anchor"],
                "exists": False,
                "binding_status": "invalid",
            }
            manifest: dict[str, object] = {}
            value_summary: dict[str, object] = {}
            evidence_bindings: list[dict[str, object]] = []
        else:
            manifest = json.loads(
                _repo_path(repo_root, str(spec["source_manifest"])).read_text(encoding="utf-8")
            )
            anchor_binding = _checksum_anchor_binding(
                repo_root,
                str(spec["source_manifest"]),
                str(spec["checksum_ledger"]),
                str(spec["anchor"]),
            )
            value_summary, evidence_bindings = _external_value_summary(
                repo_root, str(spec["source_id"]), str(spec["anchor"])
            )
        missing_evidence = [binding["path"] for binding in evidence_bindings if not binding["exists"]]
        if missing_evidence:
            required_missing.extend(missing_evidence)
        issues = ["license_qualification_pending", "release_version_qualification_pending"]
        if required_missing:
            issues.append("required_source_evidence_missing")
        if not spec["terms_snapshot"]:
            issues.append("terms_snapshot_missing")
        if anchor_binding["binding_status"] != "declared_checksum_bound":
            issues.append("anchor_checksum_binding_invalid")
        if spec["source_id"] == "allen_brain_cell_atlas":
            issues.append("aggregate_programs_not_native_observation_blocks")
        elif spec["source_id"] == "biccn_public":
            issues.append("native_value_matrix_not_materialized")
        elif spec["source_id"] == "clue_lincs":
            issues.extend(["qualified_cns_subset_missing", "dose_time_canonicalization_pending"])
        elif spec["source_id"] == "drugcomb":
            issues.extend(["non_transcriptomic_auxiliary_only", "cns_role_pending"])
        elif spec["source_id"] == "bindingdb":
            issues.append("pharmacology_context_prior_only")
        entry = {
            "source_id": spec["source_id"],
            "candidate_family": spec["candidate_family"],
            "intended_role_after_requalification": spec["intended_role"],
            "current_disposition": "quarantine_pending_recertification",
            "core_g2_eligible": False,
            "counts_toward_core_ess": False,
            "official_url": manifest.get("source_url"),
            "release_label": manifest.get("release_label"),
            "release_date": manifest.get("release_date"),
            "retrieval_date": manifest.get("download_date"),
            "access_mode": manifest.get("access_mode"),
            "license_or_terms_claim": manifest.get("license_or_terms"),
            "license_qualification_status": "pending_manual_legal_review",
            "governance_bindings": bindings,
            "processed_evidence_bindings": evidence_bindings,
            "anchor_binding": anchor_binding,
            "value_summary": value_summary,
            "issue_codes": sorted(issues),
            "bounded_binding_sha256": _binding_digest(bindings + evidence_bindings),
        }
        source_rows.append(entry)
        ledger_rows.append(
            {
                "asset_id": spec["source_id"],
                "asset_class": "legacy_external_source",
                "source_id": spec["source_id"],
                "source_accession": manifest.get("release_label", ""),
                "modality": "",
                "record_count": "",
                "sample_count": "",
                "candidate_core_family": spec["candidate_family"],
                "current_disposition": entry["current_disposition"],
                "core_g2_eligible": "no",
                "bounded_file_count": len(bindings) + len(evidence_bindings),
                "bounded_bytes": sum(
                    int(binding["size_bytes"])
                    for binding in bindings + evidence_bindings
                ),
                "binding_sha256": entry["bounded_binding_sha256"],
                "issue_codes": "|".join(sorted(issues)),
            }
        )
    invalid = sum(
        row["anchor_binding"]["binding_status"] != "declared_checksum_bound"
        or "required_source_evidence_missing" in row["issue_codes"]
        for row in source_rows
    )
    if invalid:
        _issue(blockers, "external_anchor_binding_invalid", "external_sources", invalid, "Anchor missing, malformed, or size mismatch")
    return source_rows, ledger_rows


def assess_action_headers(headers_by_asset: dict[str, set[str]]) -> dict[str, object]:
    union = set().union(*headers_by_asset.values()) if headers_by_asset else set()
    required_groups = {
        "agent_and_target": {"compound_uid", "target_uid_set"},
        "physical_dose_or_concentration": {"physical_dose_value", "physical_concentration_value"},
        "physical_unit": {"physical_dose_unit", "physical_concentration_unit"},
        "time_or_duration": {"duration_value", "timepoint_value"},
        "route_or_assay_context": {"route", "assay_context"},
        "publication_or_accession": {"publication_id", "source_accession"},
        "exposure_or_occupancy_reference": {"exposure_reference", "occupancy_reference"},
    }
    group_status: dict[str, bool] = {}
    for name, alternatives in required_groups.items():
        if name == "agent_and_target":
            group_status[name] = alternatives.issubset(union)
        else:
            group_status[name] = bool(alternatives & union)
    return {
        "required_group_status": group_status,
        "missing_required_groups": sorted(name for name, passed in group_status.items() if not passed),
        "complete_contract_possible_from_legacy_columns": all(group_status.values()),
    }


def _build_actions(
    repo_root: Path, blockers: list[dict[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    catalog_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    headers: dict[str, set[str]] = {}
    for asset_id, relative_path in ACTION_SPECS:
        path = _repo_path(repo_root, relative_path)
        digest, count, size = hash_and_count_rows(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            headers[asset_id] = set(next(csv.reader(handle, delimiter="\t")))
        expected = EXPECTED_ACTION_COUNTS[asset_id]
        if count != expected:
            _issue(blockers, "legacy_action_count_mismatch", asset_id, count, f"expected {expected}")
        entry = {
            "asset_id": asset_id,
            "path": relative_path,
            "record_count": count,
            "expected_record_count": expected,
            "size_bytes": size,
            "sha256": digest,
            "current_disposition": "migration_input_only",
            "structured_action_main_catalog_eligible": False,
            "core_observation_eligible": False,
        }
        catalog_rows.append(entry)
        ledger_rows.append(
            {
                "asset_id": asset_id,
                "asset_class": "legacy_action_catalog",
                "source_id": "legacy_derived",
                "source_accession": "",
                "modality": "structured_action",
                "record_count": count,
                "sample_count": "",
                "candidate_core_family": "",
                "current_disposition": "migration_input_only",
                "core_g2_eligible": "no",
                "bounded_file_count": 1,
                "bounded_bytes": size,
                "binding_sha256": digest,
                "issue_codes": "physical_action_contract_incomplete",
            }
        )
    header_assessment = assess_action_headers(headers)
    return {
        "catalogs": catalog_rows,
        "header_contract_assessment": header_assessment,
        "complete_structured_action_count": 0,
        "current_disposition": "migration_input_only",
        "reason": "Legacy rows omit physical dose/concentration units, duration/time, publication/accession, and exposure/occupancy references.",
    }, ledger_rows


def _simulation_qc_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counts[row.get("source_collection", "unknown")] += 1
    return counts


def _build_teachers(repo_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    catalog_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for asset_id, relative_path in TEACHER_SPECS:
        digest, count, size = hash_and_count_rows(_repo_path(repo_root, relative_path))
        entry = {
            "asset_id": asset_id,
            "path": relative_path,
            "record_count": count,
            "size_bytes": size,
            "sha256": digest,
            "current_disposition": "teacher_only",
            "observation_eligible": False,
            "core_g2_eligible": False,
            "counts_toward_modality_coverage": False,
        }
        catalog_rows.append(entry)
        ledger_rows.append(
            {
                "asset_id": asset_id,
                "asset_class": "synthetic_teacher_catalog",
                "source_id": "repository_generated_teacher",
                "source_accession": "",
                "modality": "synthetic_trajectory",
                "record_count": count,
                "sample_count": "",
                "candidate_core_family": "",
                "current_disposition": "teacher_only",
                "core_g2_eligible": "no",
                "bounded_file_count": 1,
                "bounded_bytes": size,
                "binding_sha256": digest,
                "issue_codes": "synthetic_teacher_not_observation",
            }
        )
    qc_counts = _simulation_qc_counts(
        _repo_path(repo_root, "Input/processed/simulation_audit/simulation_qc_catalog.tsv")
    )
    return {
        "catalogs": catalog_rows,
        "trajectory_counts": {
            "mesoscopic_solver_teacher": qc_counts["mesoscopic_solver_teacher"],
            "full_microenvironment_teacher": qc_counts["full_microenvironment_teacher"],
            "total": sum(qc_counts.values()),
        },
        "policy": "teacher_only_never_public_observation",
        "observation_count": 0,
        "core_g2_eligible_count": 0,
        "trajectory_store_bytes_recounted": False,
        "trajectory_store_note": "This bounded recertification hashes teacher catalogs, not multi-hundred-GB trajectory trees; storage accounting is a separate Phase-1 audit.",
    }, ledger_rows


def _controlled_boundary(repo_root: Path, blockers: list[dict[str, object]]) -> dict[str, object]:
    binding = _bind_file(repo_root, ISOLATION_AUDIT)
    if not binding["exists"]:
        _issue(blockers, "controlled_isolation_audit_missing", "controlled_boundary", 1, ISOLATION_AUDIT)
        return {"audit_binding": binding, "controlled_reachability_count": None, "status": "blocked"}
    payload = json.loads(_repo_path(repo_root, ISOLATION_AUDIT).read_text(encoding="utf-8"))
    count = int(payload.get("controlled_reachability_count", -1))
    status = "pass" if payload.get("status") == "pass" and count == 0 else "blocked"
    if status != "pass":
        _issue(blockers, "controlled_public_mainline_reachability", "controlled_boundary", count, "Controlled reachability must be zero")
    return {
        "audit_binding": binding,
        "status": status,
        "controlled_reachability_count": count,
        "public_mainline_dependency_allowed": False,
        "excluded_source_classes": [
            "ADNI",
            "dbGaP",
            "UK_Biobank",
            "ROSMAP_AMP_AD",
            "restricted_Synapse",
            "other_controlled_access_sources",
        ],
    }


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], list[dict[str, object]]]:
    repo_root = repo_root.resolve()
    blockers: list[dict[str, object]] = []
    legacy, _, bundle_ledger = _build_bundle_inventory(repo_root, blockers)
    sources, source_ledger = _build_external_sources(repo_root, blockers)
    actions, action_ledger = _build_actions(repo_root, blockers)
    teachers, teacher_ledger = _build_teachers(repo_root)
    controlled = _controlled_boundary(repo_root, blockers)

    semantic = legacy.get("semantic_audit", {})
    for code, count, message in (
        ("whole_blood_to_brain_region", int(semantic.get("whole_blood_to_brain_region_count", 0)), "Whole-blood rows carry a brain-region label"),
        ("cns_metabolomics_to_systemic", int(semantic.get("cns_metabolomics_to_systemic_count", 0)), "CSF/brain metabolomics rows carry systemic-circulation labels"),
        ("ad_case_control_inconsistency", int(semantic.get("ad_case_control_inconsistency_count", 0)), "AD diagnosis and case/control labels conflict or are missing"),
        ("pd_study_mislabeled_as_ad", int(semantic.get("e_geod_6613", {}).get("incorrect_ad_study_label_sample_count", 0)), "E-GEOD-6613 is a PD study but its cohort-level study diagnosis is AD"),
    ):
        if count:
            _issue(blockers, code, "legacy_observation_semantics", count, message)
    if legacy.get("metabolomics_bundle_count", 0):
        _issue(
            blockers,
            "metabolite_crosswalk_missing",
            "legacy_bundles",
            int(legacy["metabolomics_bundle_count"]),
            "Legacy metabolomics bundles remain deferred auxiliary data",
        )
    unaligned = int(legacy.get("transcriptomic_bundle_count", 0)) - int(legacy.get("aligned_transcriptomic_bundle_count", 0))
    if unaligned:
        _issue(blockers, "gene_alignment_missing", "legacy_bundles", unaligned, "Transcriptomic bundles lack canonical gene alignment")
    _issue(
        blockers,
        "legacy_sources_require_requalification",
        "external_sources",
        len(sources),
        "Official license, release, role, CNS relevance, and canonicalization qualification is incomplete",
    )
    _issue(
        blockers,
        "legacy_action_physical_contract_incomplete",
        "structured_actions",
        sum(int(row["record_count"]) for row in actions["catalogs"]),
        "Legacy action rows are migration inputs only",
    )

    ledger_rows = bundle_ledger + source_ledger + action_ledger + teacher_ledger
    core_families = {
        "cns_disease_aging_state": {
            "legacy_candidate_bundle_count": int(legacy.get("transcriptomic_bundle_count", 0)),
            "legacy_aligned_bundle_count": int(legacy.get("aligned_transcriptomic_bundle_count", 0)),
            "legacy_aligned_sample_count": int(legacy.get("aligned_transcriptomic_sample_count", 0)),
            "qualified_source_count": 0,
            "qualified_observation_block_count": 0,
            "g2_status": "not_passed",
        },
        "cns_cell_type_region_atlas": {
            "legacy_candidate_source_ids": ["allen_brain_cell_atlas", "biccn_public"],
            "qualified_source_count": 0,
            "qualified_observation_block_count": 0,
            "g2_status": "not_passed",
        },
        "compound_dose_time_perturbation": {
            "legacy_candidate_source_ids": ["clue_lincs"],
            "qualified_source_count": 0,
            "qualified_observation_block_count": 0,
            "g2_status": "not_passed",
        },
    }
    fatal_audit_codes = {
        "legacy_catalog_missing",
        "legacy_bundle_count_mismatch",
        "legacy_bundle_sample_count_mismatch",
        "legacy_alignment_count_mismatch",
        "listed_bundle_file_missing",
        "external_anchor_binding_invalid",
        "legacy_action_count_mismatch",
        "controlled_isolation_audit_missing",
        "controlled_public_mainline_reachability",
    }
    report = {
        "schema_version": "neuronsifter-public-foundation-corpus-legacy-recertification-v1",
        "producer": "Libs/scripts/builders/build_public_foundation_corpus_legacy_recertification.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "reduced_transcriptomics_core",
        "audit_execution_status": "pass"
        if not any(row["code"] in fatal_audit_codes for row in blockers)
        else "blocked",
        "qualification_status": "blocked",
        "public_observation_count": 0,
        "legacy_recertified_core_observation_count": 0,
        "hash_policy": {
            "catalogs_and_listed_bundle_value_blocks": "content_sha256",
            "large_external_anchors": "declared_sha256_bound_to_content_hashed_checksum_ledger_plus_size",
            "large_external_anchor_content_rehash": False,
            "recursive_tree_hashing": False,
            "reason": "Bounded audit avoids full-tree hashing while preserving explicit, fail-closed bindings.",
        },
        "legacy_study_bundles": legacy,
        "external_source_requalification": sources,
        "legacy_structured_actions": actions,
        "synthetic_teachers": teachers,
        "controlled_boundary": controlled,
        "core_family_assessment": core_families,
        "paired_benchmark": {
            "real_paired_biological_unit_count": 0,
            "cross_modal_prediction_claim_allowed": False,
        },
        "gates": {
            "iclr_v2.public_corpus.G1": {
                "status": "not_passed",
                "reason": "Source requalification and critical mapping conflicts remain open",
            },
            "iclr_v2.public_corpus.G2": {
                "status": "not_passed",
                "passing_status_when_complete": "PASS_WITH_SCOPE_LIMITATION",
                "scope": "reduced_transcriptomics_core",
            },
            "G2_full_multimodal": {"status": "not_passed"},
        },
        "blocker_count": len(blockers),
        "blockers": blockers,
    }
    return report, ledger_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bounded legacy public-corpus recertification evidence.")
    parser.add_argument("--output", default=str(REPORT_PATH))
    parser.add_argument("--ledger", default=str(LEDGER_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    ledger = Path(args.ledger)
    report, ledger_rows = build()
    write_tsv(ledger, LEDGER_FIELDS, ledger_rows)
    report["ledger"] = {
        "path": ledger.resolve().relative_to(REPO_ROOT).as_posix(),
        "row_count": len(ledger_rows),
        "sha256": sha256_file(ledger),
    }
    write_json(output, report)
    print(
        json.dumps(
            {
                "audit_execution_status": report["audit_execution_status"],
                "qualification_status": report["qualification_status"],
                "bundle_count": report["legacy_study_bundles"].get("bundle_count", 0),
                "blocker_count": report["blocker_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
