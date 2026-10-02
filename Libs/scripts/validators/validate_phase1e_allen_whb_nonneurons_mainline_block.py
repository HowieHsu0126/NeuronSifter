"""Validate the real Allen WHB Phase 1E canonical atlas mainline block."""

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

from Libs.utils.data.public_foundation_corpus import (
    canonical_sha256,
    hash_path,
    validate_semantic_axes,
)
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import (
    allowed_for_accession_role,
    load_accession_requalification_delta,
    load_phase1a_ledger,
)
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.utils.model.dynamics_store import read_zarr_array_metadata


VALIDATOR_ID = "Libs/scripts/validators/validate_phase1e_allen_whb_nonneurons_mainline_block.py"
EXPECTED_MANIFEST_VERSION = "neuronsifter-phase1e-canonical-atlas-mainline-block-v1"
EXPECTED_RETRIEVAL_RECORD_SCHEMA_VERSION = "neuronsifter-phase1e-allen-whb-retrieval-record-v1"


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


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    config = load_config(root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_atlas_materialization"]
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

        if admission.get("status") != "accepted_second_independent_block" or admission.get("independence_status") != "atlas_independent_source_pair_closed":
            _blocker(blockers, "phase1f_pair_admission_invalid", "admission.status", "Allen WHB must be admitted as the second independent atlas block.")

        expected = {
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "qc_grade": policy["qc"]["grade"],
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                _blocker(blockers, "manifest_identity_mismatch", field, f"expected {value!r}, got {manifest.get(field)!r}")
        if manifest.get("schema_version") != EXPECTED_MANIFEST_VERSION:
            _blocker(blockers, "manifest_schema_version", "schema_version", "Unexpected atlas canonical-block manifest version.")
        if manifest.get("role") != "qualified_for_mainline" or manifest.get("admission_allowed") is not True:
            _blocker(blockers, "mainline_admission_missing", "role", "Manifest does not declare mainline admission.")
        if manifest.get("split_role") != "unassigned_pending_nested_split_freeze":
            _blocker(blockers, "premature_split_assignment", "split_role", "Atlas block must remain unassigned before nested split freeze.")
        for code in validate_semantic_axes(manifest.get("semantic_axes", {})):
            _blocker(blockers, "semantic_axis_invalid", "semantic_axes", code)
        pair = manifest.get("phase1f_pair_independence")
        if not isinstance(pair, Mapping):
            _blocker(blockers, "pair_independence_missing", "phase1f_pair_independence", "Atlas M2 must bind M1/M2 independence evidence.")
        else:
            zero_overlap_fields = (
                "raw_biological_unit_token_overlap_count",
                "raw_donor_token_overlap_count",
                "biological_unit_overlap_count",
                "mirror_overlap_count",
            )
            if pair.get("status") != "passed" or pair.get("independent_for_g2") is not True:
                _blocker(blockers, "pair_independence_not_passed", "phase1f_pair_independence", "Atlas pair independence is not passed.")
            if any(pair.get(field) != 0 for field in zero_overlap_fields):
                _blocker(blockers, "pair_overlap_nonzero", "phase1f_pair_independence", "Atlas M1/M2 biological-unit, donor, or mirror overlap is nonzero.")
            axes = pair.get("different_identity_axes")
            if not isinstance(axes, Mapping) or set(axes) != {"source_id", "source_accession", "publication_accession_group", "mirror_group"} or not all(value is True for value in axes.values()):
                _blocker(blockers, "pair_identity_axes_not_independent", "phase1f_pair_independence.different_identity_axes", "Atlas M1/M2 source and accession groups are not all independent.")
            m1_binding = pair.get("m1_manifest")
            if not isinstance(m1_binding, Mapping):
                _blocker(blockers, "pair_m1_binding_missing", "phase1f_pair_independence.m1_manifest", "Atlas pair lacks a bound M1 manifest.")
            else:
                m1_path = _path(str(m1_binding.get("path", "")))
                if not m1_path.is_file() or m1_binding.get("bytes") != m1_path.stat().st_size or m1_binding.get("sha256") != sha256_file(m1_path):
                    _blocker(blockers, "pair_m1_binding_mismatch", "phase1f_pair_independence.m1_manifest", "Atlas M1 manifest binding has drifted.")
        if admission.get("phase1f_pair_independence") != pair:
            _blocker(blockers, "admission_pair_evidence_mismatch", "admission.phase1f_pair_independence", "Admission and canonical manifest pair evidence differ.")
        graph_delta = admission.get("independence_graph_delta")
        if not isinstance(graph_delta, Mapping) or graph_delta.get("independent_for_g2") is not True:
            _blocker(blockers, "independence_graph_delta_invalid", "admission.independence_graph_delta", "Atlas M2 independence graph delta is not eligible for G2.")

        delta_row = delta.get((str(candidate["source_id"]), str(candidate["source_accession"])))
        if delta_row is None:
            _blocker(blockers, "requalification_delta_missing", "source_qualification_delta.tsv", "The Atlas accession-scoped re-admission row is missing.")
        else:
            if delta_row.get("prior_decision") != "qualified_as_context_prior":
                _blocker(blockers, "delta_prior_decision_mismatch", "source_qualification_delta.tsv", "Delta must preserve the frozen context-prior source decision.")
            if delta_row.get("frozen_ledger_path") != ledger_path.relative_to(root).as_posix() or delta_row.get("frozen_ledger_sha256") != sha256_file(ledger_path):
                _blocker(blockers, "frozen_ledger_binding_mismatch", "source_qualification_delta.tsv", "Delta is not bound to frozen Phase 1A ledger bytes.")
            if delta_row.get("block_manifest_path") != manifest_path.relative_to(root).as_posix() or delta_row.get("block_manifest_sha256") != sha256_file(manifest_path):
                _blocker(blockers, "delta_manifest_binding_mismatch", "source_qualification_delta.tsv", "Delta is not bound to current canonical manifest bytes.")
        if ledger[str(candidate["source_id"])]["decision"] != "qualified_as_context_prior":
            _blocker(blockers, "frozen_ledger_unexpected_source_decision", "source_qualification_ledger.tsv", "Allen Phase 1A source decision must stay context-prior-only.")

        qualified = {"source_id": candidate["source_id"], "source_accession": candidate["source_accession"]}
        other_allen = {"source_id": candidate["source_id"], "source_accession": "SEAAD-20250531"}
        other_geo = {"source_id": "geo", "source_accession": "GSE13214"}
        weak = {"source_id": "clue_lincs", "source_accession": "GSE92742-TouchstoneV1.1"}
        if allowed_for_accession_role(qualified, ledger, "mainline_observation"):
            _blocker(blockers, "default_loader_permits_context_prior", "role_loader", "Allen context prior was accepted without a delta.")
        if not allowed_for_accession_role(qualified, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "scoped_delta_not_consumable", "role_loader", "Qualified Allen accession cannot enter mainline with its delta.")
        for name, row in (("other_allen", other_allen), ("other_geo", other_geo), ("weak", weak)):
            if allowed_for_accession_role(row, ledger, "mainline_observation", requalification_delta=delta):
                _blocker(blockers, "unscoped_source_promoted", f"role_loader.{name}", "Delta promoted a non-qualified accession or weak source.")

        input_bindings = manifest.get("input_bindings")
        if not isinstance(input_bindings, list):
            _blocker(blockers, "input_bindings_invalid", "input_bindings", "Input bindings must be a list.")
        else:
            required_paths = {str(candidate["raw_value_path"]), *map(str, candidate["release_evidence_paths"])}
            paths = {str(row.get("path", "")) for row in input_bindings if isinstance(row, Mapping)}
            if not required_paths.issubset(paths):
                _blocker(blockers, "input_provenance_incomplete", "input_bindings", "Manifest lacks official raw or release evidence bindings.")
            for binding in input_bindings:
                if not isinstance(binding, Mapping):
                    _blocker(blockers, "input_binding_invalid", "input_bindings", "Input binding is not an object.")
                    continue
                bound_path = _path(str(binding.get("path", "")))
                if not bound_path.is_file() or binding.get("bytes") != bound_path.stat().st_size or binding.get("sha256") != sha256_file(bound_path):
                    _blocker(blockers, "input_binding_mismatch", str(binding.get("path", "")), "Official input checksum or size does not match.")

        retrieval_binding = manifest.get("retrieval_record")
        if not isinstance(retrieval_binding, Mapping):
            _blocker(blockers, "retrieval_record_missing", "retrieval_record", "Canonical manifest must bind a successful retrieval record.")
        else:
            retrieval_path = _path(str(retrieval_binding.get("path", "")))
            if not retrieval_path.is_file() or retrieval_binding.get("bytes") != retrieval_path.stat().st_size or retrieval_binding.get("sha256") != sha256_file(retrieval_path):
                _blocker(blockers, "retrieval_record_binding_mismatch", "retrieval_record", "Canonical manifest retrieval binding has drifted.")
            else:
                retrieval = _load_json(retrieval_path)
                raw_binding = retrieval.get("raw_value_binding")
                job = retrieval.get("download_job")
                completion = retrieval.get("download_log_completion")
                if retrieval.get("schema_version") != EXPECTED_RETRIEVAL_RECORD_SCHEMA_VERSION or retrieval.get("validation_status") != "passed":
                    _blocker(blockers, "retrieval_record_unverified", "retrieval_record", "Retrieval record must be in the accepted verified state.")
                if raw_binding != next((row for row in input_bindings if isinstance(row, Mapping) and row.get("path") == candidate["raw_value_path"]), None):
                    _blocker(blockers, "retrieval_raw_binding_mismatch", "retrieval_record", "Retrieval record must bind the official raw input checksum.")
                if not isinstance(job, Mapping) or job.get("state") != "COMPLETED" or job.get("exit_code") != "0:0":
                    _blocker(blockers, "retrieval_job_not_successful", "retrieval_record", "Retrieval must bind a completed zero-exit Slurm job.")
                completion_matches_raw = (
                    isinstance(completion, Mapping)
                    and isinstance(raw_binding, Mapping)
                    and completion.get("sha256") == raw_binding.get("sha256")
                )
                if not completion_matches_raw:
                    _blocker(blockers, "retrieval_completion_mismatch", "retrieval_record", "Completion log must bind the retrieval raw checksum.")

        source_card_binding = admission.get("source_card")
        if not isinstance(source_card_binding, Mapping):
            _blocker(blockers, "source_card_missing", "source_card", "Admission must bind a source card.")
        else:
            source_card_path = _path(str(source_card_binding.get("path", "")))
            if not source_card_path.is_file() or source_card_binding.get("bytes") != source_card_path.stat().st_size or source_card_binding.get("sha256") != sha256_file(source_card_path):
                _blocker(blockers, "source_card_binding_mismatch", "source_card", "Admission source-card binding has drifted.")
            else:
                source_card = _load_json(source_card_path)
                card_retrieval = source_card.get("retrieval_record")
                if not isinstance(card_retrieval, Mapping) or card_retrieval.get("binding") != retrieval_binding:
                    _blocker(blockers, "source_card_retrieval_binding_mismatch", "source_card", "Source card must bind the canonical retrieval record.")

        records = manifest.get("block_files")
        expected_names = {"sparse_counts.zarr", "cell_metadata.parquet", "feature_metadata.parquet", "matrix_manifest.json"}
        if not isinstance(records, list) or {str(row.get("name", "")) for row in records if isinstance(row, Mapping)} != expected_names:
            _blocker(blockers, "block_file_set_invalid", "block_files", "Atlas block must bind sparse store, metadata, and matrix manifest.")
            records = []
        for record in records:
            if not isinstance(record, Mapping):
                _blocker(blockers, "block_file_record_invalid", "block_files", "Block file record is not an object.")
                continue
            payload_path = manifest_path.parent / str(record.get("relative_path", record.get("name", "")))
            count, byte_count, digest = hash_path(payload_path)
            if not payload_path.exists() or record.get("file_count") != count or record.get("bytes") != byte_count or record.get("sha256") != digest:
                _blocker(blockers, "block_file_binding_mismatch", str(record.get("name", "")), "Canonical block binding does not match payload bytes.")
        if records and manifest.get("block_sha256") != _content_hash([row for row in records if isinstance(row, Mapping)]):
            _blocker(blockers, "block_content_hash_mismatch", "block_sha256", "Block hash is not derived from content-addressed records.")

        samples = pd.read_parquet(manifest_path.parent / "cell_metadata.parquet")
        features = pd.read_parquet(manifest_path.parent / "feature_metadata.parquet")
        matrix_stats = _load_json(manifest_path.parent / "matrix_manifest.json")
        store = manifest_path.parent / "sparse_counts.zarr"
        data_meta = read_zarr_array_metadata(store, "data")
        indices_meta = read_zarr_array_metadata(store, "indices")
        indptr_meta = read_zarr_array_metadata(store, "indptr")
        shape = tuple(int(value) for value in matrix_stats.get("matrix_shape", []))
        if len(shape) != 2 or shape != (len(samples), len(features)):
            _blocker(blockers, "matrix_shape_invalid", "matrix_manifest.json", "Sparse matrix shape does not equal cell and feature metadata axes.")
        if int(data_meta["shape"][0]) != int(matrix_stats.get("nnz", -1)) or indices_meta["shape"] != data_meta["shape"]:
            _blocker(blockers, "sparse_nnz_axis_invalid", "sparse_counts.zarr", "Sparse data and indices do not match recorded nnz.")
        expected_indptr = shape[0] + 1 if matrix_stats.get("matrix_encoding") == "csr_matrix" else shape[1] + 1
        if int(indptr_meta["shape"][0]) != expected_indptr:
            _blocker(blockers, "sparse_indptr_axis_invalid", "sparse_counts.zarr", "Sparse indptr length does not match matrix orientation.")
        if samples["biological_unit_id"].duplicated().any() or samples["source_cell_label"].duplicated().any():
            _blocker(blockers, "cell_identity_invalid", "cell_metadata.parquet", "Cell biological-unit IDs must be unique.")
        if samples["donor_label"].eq("").any():
            _blocker(blockers, "donor_identity_missing", "cell_metadata.parquet", "Every cell must retain an observed donor label.")
        donor_count = int(samples["donor_label"].nunique())
        if donor_count < int(policy["qc"]["minimum_donor_count"]) or donor_count != int(policy["qc"]["expected_donor_count"]) or manifest.get("effective_sample_size") != float(donor_count):
            _blocker(blockers, "biological_unit_ess_invalid", "effective_sample_size", "ESS must be unique donors, not cell count.")
        if len(samples) < int(policy["qc"]["minimum_cell_count"]) or len(features) < int(policy["qc"]["minimum_mapped_gene_count"]):
            _blocker(blockers, "atlas_qc_threshold_not_met", "qc", "Configured cell or feature QC threshold is not met.")
        if len(samples) != int(policy["qc"]["expected_cell_count"]) or len(features) != int(policy["qc"]["expected_mapped_gene_count"]):
            _blocker(blockers, "atlas_official_shape_mismatch", "qc", "Cell or feature count differs from the official matrix-specific contract.")
        if features["canonical_feature_id"].duplicated().any() or features["canonical_feature_id"].eq("").any():
            _blocker(blockers, "feature_mapping_invalid", "feature_metadata.parquet", "Canonical Ensembl features must be unique and nonempty.")
        if matrix_stats.get("value_unit_canonical") != "raw UMI count" or matrix_stats.get("finite_nonzero_values") is not True or matrix_stats.get("minimum_nonzero_value", -1) < 0:
            _blocker(blockers, "raw_count_value_invalid", "matrix_manifest.json", "Raw count sparse values must remain finite and nonnegative.")
        if manifest.get("qc", {}).get("status") != "passed":
            _blocker(blockers, "qc_not_passed", "qc.status", "Atlas manifest QC status must be passed.")
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as exc:
        _blocker(blockers, "validation_exception", "$", str(exc))

    return {
        "schema_version": "neuronsifter-phase1e-atlas-mainline-block-validation-v1",
        "producer": VALIDATOR_ID,
        "validated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "canonical_uid": manifest.get("canonical_uid"),
        "source_id": manifest.get("source_id"),
        "source_accession": manifest.get("source_accession"),
        "qc_grade": manifest.get("qc_grade"),
        "effective_sample_size": manifest.get("effective_sample_size"),
        "g2_status": "not_passed",
        "g2_remaining_blockers": manifest.get("blocker_codes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Allen WHB Phase 1E canonical atlas mainline block.")
    parser.add_argument("--output", default="Output/audits/phase1e_allen_whb_nonneurons_mainline_block_validation.json")
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(_path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
