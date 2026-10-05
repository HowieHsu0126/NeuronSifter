"""Validate the independent GSE37263 disease/aging mainline block."""

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

from Libs.utils.data.public_foundation_corpus import validate_semantic_axes
from Libs.utils.data.public_foundation_evidence import canonical_sha256, sha256_file
from Libs.utils.data.public_foundation_roles import allowed_for_accession_role, load_accession_requalification_delta, load_phase1a_ledger
from Libs.utils.data.raw_data_registry import load_config, write_json


VALIDATOR_ID = "Libs/scripts/validators/validate_phase1f_gse37263_mainline_block.py"


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _blocker(blockers: list[dict[str, str]], code: str, path: str, message: str) -> None:
    blockers.append({"code": code, "path": path, "message": message})


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    blockers: list[dict[str, str]] = []
    manifest: dict[str, object] = {}
    try:
        config = load_config(repo_root / "Libs/configs/config_stack.yaml")
        policy = config["public_foundation_corpus"]["phase1f_disease_pair_materialization"]
        candidate = policy["candidate"]
        admission = _load_json(_path(policy["admission_audit_path"]))
        if admission.get("status") != "accepted_second_independent_block" or admission.get("independence_status") != "passed_independent_pair_with_gse28146":
            _blocker(blockers, "pair_admission_status_invalid", "admission", "Disease M2 is not explicitly accepted as the independent pair.")
        binding = admission.get("canonical_block_manifest")
        if not isinstance(binding, Mapping):
            raise ValueError("Admission lacks canonical_block_manifest binding")
        manifest_path = _path(str(binding.get("path", "")))
        if not manifest_path.is_file() or binding.get("bytes") != manifest_path.stat().st_size or binding.get("sha256") != sha256_file(manifest_path):
            raise ValueError("Canonical manifest binding is stale")
        manifest = _load_json(manifest_path)
        expected = {
            "schema_version": "neuronsifter-phase1f-disease-pair-mainline-block-v1",
            "canonical_uid": "pfc_arrayexpress_e_geod_37263_phase1f_v1",
            "family_id": candidate["family_id"], "source_id": candidate["source_id"], "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"], "mirror_group": candidate["mirror_group"], "qc_grade": "B",
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                _blocker(blockers, "manifest_identity_mismatch", field, f"expected {value!r}, got {manifest.get(field)!r}")
        if manifest.get("role") != "qualified_for_mainline" or manifest.get("admission_allowed") is not True:
            _blocker(blockers, "mainline_role_missing", "role", "Disease M2 lacks mainline role.")
        if manifest.get("split_role") != "unassigned_pending_nested_split_freeze":
            _blocker(blockers, "premature_split_assignment", "split_role", "Disease M2 was assigned before final split freeze.")
        for code in validate_semantic_axes(manifest.get("semantic_axes", {})):
            _blocker(blockers, "semantic_axis_invalid", "semantic_axes", code)

        input_bindings = manifest.get("input_bindings", [])
        required_inputs = {str(candidate[key]) for key in ("raw_value_path", "sample_metadata_path", "study_metadata_path", "platform_annotation_path", "terms_snapshot_path")}
        bound_inputs = {str(row.get("path", "")) for row in input_bindings if isinstance(row, Mapping)}
        if required_inputs != bound_inputs:
            _blocker(blockers, "input_provenance_incomplete", "input_bindings", "Required official inputs are not bound exactly.")
        for row in input_bindings:
            if not isinstance(row, Mapping):
                _blocker(blockers, "input_binding_invalid", "input_bindings", "Input binding is not an object.")
                continue
            path = _path(str(row.get("path", "")))
            if not path.is_file() or row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
                _blocker(blockers, "input_binding_mismatch", str(row.get("path", "")), "Official input checksum mismatch.")

        block_files = manifest.get("block_files", [])
        expected_names = {"values.parquet", "sample_metadata.parquet", "feature_metadata.parquet", "source_feature_mapping.parquet", "feature_quarantine.parquet"}
        if {str(row.get("name", "")) for row in block_files if isinstance(row, Mapping)} != expected_names:
            _blocker(blockers, "block_file_set_invalid", "block_files", "Canonical block file set is incomplete.")
        for row in block_files:
            if not isinstance(row, Mapping):
                continue
            path = manifest_path.parent / str(row.get("name", ""))
            if not path.is_file() or row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
                _blocker(blockers, "block_file_binding_mismatch", str(row.get("name", "")), "Canonical block file binding mismatch.")
        expected_hash = canonical_sha256([{"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]} for row in block_files if isinstance(row, Mapping)])
        if manifest.get("block_sha256") != expected_hash:
            _blocker(blockers, "block_hash_mismatch", "block_sha256", "Block hash is not derived from its files.")

        samples = pd.read_parquet(manifest_path.parent / "sample_metadata.parquet")
        features = pd.read_parquet(manifest_path.parent / "feature_metadata.parquet")
        values = pd.read_parquet(manifest_path.parent / "values.parquet")
        mapping = pd.read_parquet(manifest_path.parent / "source_feature_mapping.parquet")
        quarantine = pd.read_parquet(manifest_path.parent / "feature_quarantine.parquet")
        qc = policy["qc"]
        if len(samples) != int(qc["expected_sample_count"]) or samples["source_native_subject_token"].nunique() != len(samples):
            _blocker(blockers, "subject_ess_invalid", "sample_metadata.parquet", "Sixteen unique source-native subjects are required.")
        if samples["source_sample_accession"].duplicated().any() or samples["technical_replicate_group"].duplicated().any():
            _blocker(blockers, "technical_replicate_policy_invalid", "sample_metadata.parquet", "One array per subject specimen is not preserved.")
        if samples["disease_state_canonical"].value_counts().to_dict() != {"normal_control": 8, "Alzheimer disease": 8}:
            _blocker(blockers, "disease_balance_invalid", "sample_metadata.parquet", "Expected 8 AD and 8 controls.")
        if set(samples["brain_region_ontology_id"]) != {"UBERON:0001871"} or samples["brain_region_canonical"].str.contains("middle temporal gyrus", case=False).any():
            _blocker(blockers, "brain_region_overmapping", "sample_metadata.parquet", "BA22 source evidence must not be coerced to MTG.")
        if len(mapping) != int(qc["expected_accepted_source_mapping_count"]):
            _blocker(blockers, "accepted_mapping_count_invalid", "source_feature_mapping.parquet", "Accepted GPL5175 source mapping count differs from frozen expectation.")
        reasons = quarantine["mapping_status"].value_counts().to_dict()
        if reasons.get("multi_gene_official_platform_conflict", 0) != int(qc["expected_multi_gene_quarantine_count"]):
            _blocker(blockers, "multi_gene_quarantine_invalid", "feature_quarantine.parquet", "Multi-gene conflicts are not explicitly quarantined.")
        if reasons.get("unassigned_official_platform_feature", 0) != int(qc["expected_unassigned_quarantine_count"]):
            _blocker(blockers, "unassigned_quarantine_invalid", "feature_quarantine.parquet", "Unassigned features are not explicitly quarantined.")
        if reasons.get("entrez_id_absent_from_canonical_gene_backbone", 0) != int(qc["expected_backbone_missing_quarantine_count"]):
            _blocker(blockers, "backbone_missing_quarantine_invalid", "feature_quarantine.parquet", "Unique Entrez IDs absent from the frozen backbone are not explicitly quarantined.")
        if len(mapping) + reasons.get("entrez_id_absent_from_canonical_gene_backbone", 0) != int(qc["expected_unique_gene_assignment_count"]):
            _blocker(blockers, "unique_gene_assignment_partition_invalid", "feature mapping", "Unique platform assignments do not partition into accepted and backbone-missing records.")
        if len(mapping) + len(quarantine) != int(qc["expected_source_feature_count"]):
            _blocker(blockers, "source_feature_partition_invalid", "feature mapping", "Accepted and quarantined features do not partition the official source axis.")
        if len(features) != int(qc["expected_canonical_gene_count"]) or len(features) < int(qc["minimum_mapped_gene_count"]) or features["canonical_gene_id"].duplicated().any():
            _blocker(blockers, "canonical_feature_axis_invalid", "feature_metadata.parquet", "Canonical gene axis is insufficient or duplicated.")
        if len(values) != len(samples) * len(features) or not np.isfinite(values["value"].to_numpy(dtype=np.float64)).all():
            _blocker(blockers, "canonical_value_block_invalid", "values.parquet", "Value rows are incomplete or non-finite.")
        if manifest.get("effective_sample_size") != float(len(samples)):
            _blocker(blockers, "manifest_ess_invalid", "effective_sample_size", "ESS must equal unique source-native subject count.")

        independence = manifest.get("pair_independence", {})
        if independence.get("status") != "passed" or independence.get("source_sample_token_overlap_count") != 0 or independence.get("biological_unit_overlap_count") != 0:
            _blocker(blockers, "pair_independence_invalid", "pair_independence", "Disease M1/M2 independence is not closed.")
        m1_binding = independence.get("m1_manifest", {})
        m1_path = _path(str(m1_binding.get("path", "")))
        if not m1_path.is_file() or m1_binding.get("sha256") != sha256_file(m1_path) or m1_binding.get("bytes") != m1_path.stat().st_size:
            _blocker(blockers, "m1_binding_invalid", "pair_independence.m1_manifest", "Disease M1 binding is stale.")

        ledger_path = _path(policy["source_ledger_path"])
        delta_path = _path(policy["qualification_delta_path"])
        ledger = load_phase1a_ledger(ledger_path)
        delta = load_accession_requalification_delta(delta_path)
        row = {"source_id": candidate["source_id"], "source_accession": candidate["source_accession"]}
        if allowed_for_accession_role(row, ledger, "mainline_observation"):
            _blocker(blockers, "deferred_source_default_promotion", "role_loader", "Deferred source entered mainline without a scoped delta.")
        if not allowed_for_accession_role(row, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "scoped_role_not_consumable", "role_loader", "Disease M2 scoped delta is not consumable.")
        delta_row = delta.get((candidate["source_id"], candidate["source_accession"]))
        if delta_row is None or delta_row.get("block_manifest_path") != manifest_path.relative_to(repo_root).as_posix() or delta_row.get("block_manifest_sha256") != sha256_file(manifest_path):
            _blocker(blockers, "delta_manifest_binding_invalid", "source_qualification_delta.tsv", "Disease M2 delta does not bind current manifest.")
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as exc:
        _blocker(blockers, "validation_exception", "$", str(exc))

    return {
        "schema_version": "neuronsifter-phase1f-disease-pair-mainline-validation-v1",
        "producer": VALIDATOR_ID,
        "validated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked", "blocker_count": len(blockers), "blockers": blockers,
        "canonical_uid": manifest.get("canonical_uid"), "source_id": manifest.get("source_id"), "source_accession": manifest.get("source_accession"),
        "qc_grade": manifest.get("qc_grade"), "effective_sample_size": manifest.get("effective_sample_size"),
        "independent_disease_source_count": 2 if not blockers else 1,
        "g2_status": "not_passed",
        "g2_remaining_blockers": ["atlas_second_independent_source_missing", "perturbation_second_independent_source_missing", "nested_split_freeze_pending", "real_domain_leakage_audit_pending"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the independent GSE37263 disease block.")
    parser.add_argument("--output", default="Output/audits/phase1f_gse37263_mainline_block_validation.json")
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(_path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
