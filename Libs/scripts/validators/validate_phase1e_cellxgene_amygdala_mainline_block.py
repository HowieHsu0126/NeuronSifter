"""Validate the accession-scoped CELLxGENE amygdala Phase 1E atlas block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path, validate_semantic_axes
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import allowed_for_accession_role, load_accession_requalification_delta, load_phase1a_ledger
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.utils.model.dynamics_store import read_zarr_array_metadata


VALIDATOR_ID = "Libs/scripts/validators/validate_phase1e_cellxgene_amygdala_mainline_block.py"
EXPECTED_MANIFEST_VERSION = "neuronsifter-phase1e-cellxgene-atlas-mainline-block-v1"
EXPECTED_RETRIEVAL_RECORD_SCHEMA_VERSION = "neuronsifter-phase1e-cellxgene-retrieval-record-v1"


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _blocker(blockers: list[dict[str, str]], code: str, path: str, message: str) -> None:
    blockers.append({"code": code, "path": path, "message": message})


def _content_hash(records: list[Mapping[str, object]]) -> str:
    return canonical_sha256([
        {"name": row["name"], "file_count": row["file_count"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in records
    ])


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    config = load_config(root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_cellxgene_atlas_materialization"]
    candidate = policy["candidate"]
    blockers: list[dict[str, str]] = []
    manifest: dict[str, object] = {}
    try:
        admission_path = _path(str(policy["admission_audit_path"]))
        admission = _load_json(admission_path)
        delta_path = _path(str(policy["qualification_delta_path"]))
        delta = load_accession_requalification_delta(delta_path)
        ledger_path = _path(str(policy["source_ledger_path"]))
        ledger = load_phase1a_ledger(ledger_path)
        manifest_binding = admission.get("canonical_block_manifest")
        if not isinstance(manifest_binding, Mapping):
            raise ValueError("Admission audit lacks canonical_block_manifest binding")
        manifest_path = _path(str(manifest_binding.get("path", "")))
        if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_binding.get("sha256"):
            raise ValueError("Admission manifest binding checksum mismatch")
        manifest = _load_json(manifest_path)
        expected_identity = {
            "family_id": candidate["family_id"], "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"], "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"], "qc_grade": policy["qc"]["grade"],
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                _blocker(blockers, "manifest_identity_mismatch", field, f"expected {expected!r}, got {manifest.get(field)!r}")
        if manifest.get("schema_version") != EXPECTED_MANIFEST_VERSION:
            _blocker(blockers, "manifest_schema_version", "schema_version", "Unexpected CELLxGENE block manifest schema version.")
        if manifest.get("role") != "qualified_for_mainline" or manifest.get("admission_allowed") is not True:
            _blocker(blockers, "mainline_admission_missing", "role", "Manifest must declare accession-scoped mainline admission.")
        if manifest.get("split_role") != "unassigned_pending_nested_split_freeze":
            _blocker(blockers, "premature_split_assignment", "split_role", "First block must remain unassigned before nested split freeze.")
        for code in validate_semantic_axes(manifest.get("semantic_axes", {})):
            _blocker(blockers, "semantic_axis_invalid", "semantic_axes", code)

        delta_row = delta.get((str(candidate["source_id"]), str(candidate["source_accession"])))
        if delta_row is None:
            _blocker(blockers, "requalification_delta_missing", "source_qualification_delta.tsv", "CELLxGENE accession delta is missing.")
        else:
            if delta_row.get("prior_decision") != "deferred":
                _blocker(blockers, "delta_prior_decision_mismatch", "source_qualification_delta.tsv", "Delta must preserve the frozen deferred source decision.")
            if delta_row.get("frozen_ledger_path") != ledger_path.relative_to(root).as_posix() or delta_row.get("frozen_ledger_sha256") != sha256_file(ledger_path):
                _blocker(blockers, "frozen_ledger_binding_mismatch", "source_qualification_delta.tsv", "Delta must bind frozen ledger bytes.")
            if delta_row.get("block_manifest_path") != manifest_path.relative_to(root).as_posix() or delta_row.get("block_manifest_sha256") != sha256_file(manifest_path):
                _blocker(blockers, "delta_manifest_binding_mismatch", "source_qualification_delta.tsv", "Delta must bind current canonical manifest bytes.")
        if ledger[str(candidate["source_id"])]["decision"] != "deferred":
            _blocker(blockers, "frozen_ledger_unexpected_source_decision", "source_qualification_ledger.tsv", "CELLxGENE source-wide Phase 1A decision must remain deferred.")
        qualified = {"source_id": candidate["source_id"], "source_accession": candidate["source_accession"]}
        unqualified = {"source_id": candidate["source_id"], "source_accession": "unmaterialized"}
        weak = {"source_id": "clue_lincs", "source_accession": "GSE92742-TouchstoneV1.1"}
        if allowed_for_accession_role(qualified, ledger, "mainline_observation"):
            _blocker(blockers, "default_loader_permits_deferred_source", "role_loader", "Deferred CELLxGENE source was accepted without a delta.")
        if not allowed_for_accession_role(qualified, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "scoped_delta_not_consumable", "role_loader", "Qualified CELLxGENE accession cannot enter mainline with its delta.")
        for name, row in (("unqualified", unqualified), ("weak", weak)):
            if allowed_for_accession_role(row, ledger, "mainline_observation", requalification_delta=delta):
                _blocker(blockers, "unscoped_source_promoted", f"role_loader.{name}", "Delta promoted an unqualified accession or weak source.")

        input_bindings = manifest.get("input_bindings")
        if not isinstance(input_bindings, list):
            _blocker(blockers, "input_bindings_invalid", "input_bindings", "Input bindings must be a list.")
            input_bindings = []
        required_paths = {str(candidate["raw_value_path"]), *map(str, candidate["release_evidence_paths"])}
        paths = {str(row.get("path", "")) for row in input_bindings if isinstance(row, Mapping)}
        if not required_paths.issubset(paths):
            _blocker(blockers, "input_provenance_incomplete", "input_bindings", "Manifest lacks official raw or evidence bindings.")
        for binding in input_bindings:
            if not isinstance(binding, Mapping):
                _blocker(blockers, "input_binding_invalid", "input_bindings", "Input binding is not an object.")
                continue
            path = _path(str(binding.get("path", "")))
            if not path.is_file() or binding.get("bytes") != path.stat().st_size or binding.get("sha256") != sha256_file(path):
                _blocker(blockers, "input_binding_mismatch", str(binding.get("path", "")), "Bound input bytes or checksum have drifted.")

        retrieval_binding = manifest.get("retrieval_record")
        if not isinstance(retrieval_binding, Mapping):
            _blocker(blockers, "retrieval_record_missing", "retrieval_record", "Manifest must bind a verified retrieval record.")
        else:
            retrieval_path = _path(str(retrieval_binding.get("path", "")))
            if not retrieval_path.is_file() or retrieval_binding.get("bytes") != retrieval_path.stat().st_size or retrieval_binding.get("sha256") != sha256_file(retrieval_path):
                _blocker(blockers, "retrieval_record_binding_mismatch", "retrieval_record", "Retrieval record binding has drifted.")
            else:
                retrieval = _load_json(retrieval_path)
                raw_binding = retrieval.get("raw_value_binding")
                completion = retrieval.get("download_log_completion")
                job = retrieval.get("download_job")
                if retrieval.get("schema_version") != EXPECTED_RETRIEVAL_RECORD_SCHEMA_VERSION or retrieval.get("validation_status") != "passed":
                    _blocker(blockers, "retrieval_record_unverified", "retrieval_record", "Retrieval record is not in the accepted verified state.")
                if raw_binding != next((row for row in input_bindings if isinstance(row, Mapping) and row.get("path") == candidate["raw_value_path"]), None):
                    _blocker(blockers, "retrieval_raw_binding_mismatch", "retrieval_record", "Retrieval record must bind the canonical raw H5AD input.")
                if not isinstance(job, Mapping) or job.get("job_id") != candidate["retrieval_job_id"] or job.get("state") != "COMPLETED" or job.get("exit_code") != "0:0":
                    _blocker(blockers, "retrieval_job_not_successful", "retrieval_record", "Retrieval record must bind successful job 486183 only.")
                if not isinstance(completion, Mapping) or completion.get("remote_sha256") != candidate["expected_raw_sha256"] or completion.get("local_sha256") != candidate["expected_raw_sha256"]:
                    _blocker(blockers, "retrieval_completion_mismatch", "retrieval_record", "Verified remote and local SHA-256 must equal configured raw SHA-256.")

        records = manifest.get("block_files")
        expected_names = {"sparse_counts.zarr", "cell_metadata.parquet", "cell_type_quarantine.parquet", "feature_metadata.parquet", "matrix_manifest.json"}
        if not isinstance(records, list) or {str(row.get("name", "")) for row in records if isinstance(row, Mapping)} != expected_names:
            _blocker(blockers, "block_file_set_invalid", "block_files", "Block must bind sparse matrix, sample, quarantine, feature, and matrix files.")
            records = []
        for record in records:
            if not isinstance(record, Mapping):
                _blocker(blockers, "block_file_record_invalid", "block_files", "Block file record is not an object.")
                continue
            payload_path = manifest_path.parent / str(record.get("relative_path", record.get("name", "")))
            file_count, byte_count, digest = hash_path(payload_path)
            if record.get("file_count") != file_count or record.get("bytes") != byte_count or record.get("sha256") != digest:
                _blocker(blockers, "block_file_binding_mismatch", str(record.get("name", "")), "Canonical block file binding has drifted.")
        if records and manifest.get("block_sha256") != _content_hash([row for row in records if isinstance(row, Mapping)]):
            _blocker(blockers, "block_content_hash_mismatch", "block_sha256", "Block hash must derive from all canonical payloads.")

        samples = pd.read_parquet(manifest_path.parent / "cell_metadata.parquet")
        quarantined = pd.read_parquet(manifest_path.parent / "cell_type_quarantine.parquet")
        features = pd.read_parquet(manifest_path.parent / "feature_metadata.parquet")
        matrix = _load_json(manifest_path.parent / "matrix_manifest.json")
        if len(samples) < int(policy["qc"]["minimum_cell_count"]) or samples["biological_unit_id"].nunique() != len(samples):
            _blocker(blockers, "biological_unit_axis_invalid", "cell_metadata.parquet", "Qualified cells must remain unique biological units above threshold.")
        if samples["donor_id"].nunique() != int(policy["qc"]["minimum_donor_count"]) or manifest.get("effective_sample_size") != float(policy["qc"]["minimum_donor_count"]):
            _blocker(blockers, "donor_ess_invalid", "effective_sample_size", "ESS must be exactly the five unique qualified donor IDs.")
        if not (samples["technical_replicate_group"] == samples["biological_unit_id"]).all():
            _blocker(blockers, "technical_replicate_policy_invalid", "cell_metadata.parquet", "Technical replicate groups must use unique source observation IDs.")
        if not set(samples["cell_type_namespace"]) == {"Cell_Ontology"} or not samples["cell_type_id"].astype(str).str.startswith("CL:").all():
            _blocker(blockers, "cell_type_mapping_invalid", "cell_metadata.parquet", "Qualified cell types must bind Cell Ontology IDs.")
        if (samples["cell_type_label"].astype(str).str.lower() == "unknown").any() or (samples["cell_type_id"].astype(str).str.lower() == "unknown").any():
            _blocker(blockers, "unknown_cell_type_not_quarantined", "cell_metadata.parquet", "Unknown cell types cannot remain in the qualified block.")
        if quarantined.empty or not (quarantined["quarantine_reason"] == "cell_type_label_or_ontology_unknown_or_missing").all():
            _blocker(blockers, "cell_type_quarantine_invalid", "cell_type_quarantine.parquet", "Unknown cell types must be explicitly quarantined.")
        if not ((quarantined["cell_type"].astype(str).str.lower() == "unknown") | (quarantined["cell_type_ontology_term_id"].astype(str).str.lower() == "unknown") | quarantined["cell_type"].astype(str).eq("") | quarantined["cell_type_ontology_term_id"].astype(str).eq("")).all():
            _blocker(blockers, "cell_type_quarantine_overreach", "cell_type_quarantine.parquet", "Quarantine rows must explain an unknown or missing cell-type axis.")
        if set(samples["brain_region_id"]) != {"UBERON:0001876"} or set(samples["disease_state_id"]) != {"PATO:0000461"} or set(samples["assay_id"]) != {"EFO:0009922"}:
            _blocker(blockers, "semantic_metadata_mismatch", "cell_metadata.parquet", "Qualified CELLxGENE semantic axes do not match official amygdala normal 10x data.")
        if len(features) < int(policy["qc"]["minimum_mapped_gene_count"]) or features["canonical_feature_id"].duplicated().any() or not features["canonical_feature_id"].astype(str).str.fullmatch(r"ENSG\d+").all():
            _blocker(blockers, "feature_mapping_invalid", "feature_metadata.parquet", "Features must be unique Ensembl gene IDs above the QC threshold.")
        if set(features["feature_reference"]) != {"NCBITaxon:9606"}:
            _blocker(blockers, "feature_species_mismatch", "feature_metadata.parquet", "Feature reference must be human NCBITaxon:9606.")
        store = manifest_path.parent / "sparse_counts.zarr"
        data_metadata = read_zarr_array_metadata(store, "data")
        indices_metadata = read_zarr_array_metadata(store, "indices")
        indptr_metadata = read_zarr_array_metadata(store, "indptr")
        expected_shape = [len(samples), len(features)]
        if matrix.get("matrix_shape") != expected_shape or matrix.get("nnz", 0) <= 0 or matrix.get("finite_nonzero_values") is not True or matrix.get("integer_like_nonzero_values") is not True:
            _blocker(blockers, "matrix_qc_invalid", "matrix_manifest.json", "Filtered sparse values must be finite integer-like counts with positive nnz.")
        if data_metadata.get("shape") != [matrix.get("nnz")] or indices_metadata.get("shape") != [matrix.get("nnz")] or indptr_metadata.get("shape") != [len(samples) + 1]:
            _blocker(blockers, "sparse_store_shape_invalid", "sparse_counts.zarr", "Sparse store arrays do not match filtered canonical axes.")
        if manifest.get("qc", {}).get("status") != "passed" or manifest.get("qc", {}).get("quarantined_cell_count") != len(quarantined):
            _blocker(blockers, "qc_not_passed", "qc", "Manifest QC must pass and report exact quarantine count.")
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as exc:
        _blocker(blockers, "validation_exception", "$", str(exc))
    return {
        "schema_version": "neuronsifter-phase1e-cellxgene-atlas-mainline-block-validation-v1",
        "producer": VALIDATOR_ID,
        "validated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers), "blockers": blockers,
        "canonical_uid": manifest.get("canonical_uid"), "source_id": manifest.get("source_id"),
        "source_accession": manifest.get("source_accession"), "qc_grade": manifest.get("qc_grade"),
        "effective_sample_size": manifest.get("effective_sample_size"), "g2_status": "not_passed",
        "g2_remaining_blockers": manifest.get("blocker_codes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CELLxGENE amygdala Phase 1E canonical atlas block.")
    parser.add_argument("--output", default="Output/audits/phase1e_cellxgene_amygdala_mainline_block_validation.json")
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(_path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
