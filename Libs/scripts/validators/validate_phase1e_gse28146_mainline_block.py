"""Validate the real GSE28146 Phase 1E canonical mainline block."""

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
from Libs.utils.data.public_foundation_roles import (
    allowed_for_accession_role,
    load_accession_requalification_delta,
    load_phase1a_ledger,
)
from Libs.utils.data.raw_data_registry import load_config, write_json


VALIDATOR_ID = "Libs/scripts/validators/validate_phase1e_gse28146_mainline_block.py"
EXPECTED_MANIFEST_VERSION = "neuronsifter-phase1e-canonical-mainline-block-v1"


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


def _content_hash(files: list[Mapping[str, object]]) -> str:
    return canonical_sha256(
        [
            {"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in files
        ]
    )


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_materialization"]
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
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if sha256_file(manifest_path) != manifest_binding.get("sha256"):
            raise ValueError("Admission manifest binding checksum mismatch")
        manifest = _load_json(manifest_path)

        expected_identity = {
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "qc_grade": policy["qc"]["grade"],
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                _blocker(blockers, "manifest_identity_mismatch", field, f"expected {expected!r}, got {manifest.get(field)!r}")
        if manifest.get("schema_version") != EXPECTED_MANIFEST_VERSION:
            _blocker(blockers, "manifest_schema_version", "schema_version", "Unexpected canonical block manifest schema version.")
        if manifest.get("role") != "qualified_for_mainline" or manifest.get("admission_allowed") is not True:
            _blocker(blockers, "mainline_admission_missing", "role", "Manifest does not declare mainline admission.")
        if manifest.get("split_role") != "unassigned_pending_nested_split_freeze":
            _blocker(blockers, "premature_split_assignment", "split_role", "First qualified block must remain unassigned until nested split freeze.")
        semantic_blockers = validate_semantic_axes(manifest.get("semantic_axes", {}))
        for code in semantic_blockers:
            _blocker(blockers, "semantic_axis_invalid", "semantic_axes", code)

        delta_row = delta.get((str(candidate["source_id"]), str(candidate["source_accession"])))
        if delta_row is None:
            _blocker(blockers, "requalification_delta_missing", "source_qualification_delta.tsv", "The accession-scoped re-admission row is missing.")
        else:
            if delta_row.get("frozen_ledger_path") != ledger_path.relative_to(repo_root).as_posix():
                _blocker(blockers, "frozen_ledger_path_mismatch", "source_qualification_delta.tsv", "Delta does not bind the frozen source ledger path.")
            if delta_row.get("frozen_ledger_sha256") != sha256_file(ledger_path):
                _blocker(blockers, "frozen_ledger_hash_mismatch", "source_qualification_delta.tsv", "Delta does not bind the frozen source ledger bytes.")
            if delta_row.get("block_manifest_path") != manifest_path.relative_to(repo_root).as_posix():
                _blocker(blockers, "delta_manifest_path_mismatch", "source_qualification_delta.tsv", "Delta does not bind this manifest.")
            if delta_row.get("block_manifest_sha256") != sha256_file(manifest_path):
                _blocker(blockers, "delta_manifest_hash_mismatch", "source_qualification_delta.tsv", "Delta does not bind current manifest bytes.")
        if ledger[str(candidate["source_id"])]["decision"] != "deferred":
            _blocker(blockers, "frozen_ledger_unexpected_source_decision", "source_qualification_ledger.tsv", "Phase 1A source decision must remain deferred.")

        first_row = {"source_id": candidate["source_id"], "source_accession": candidate["source_accession"]}
        mirror_row = {"source_id": candidate["mirror_source_id"], "source_accession": candidate["mirror_source_accession"]}
        weak_row = {"source_id": "clue_lincs", "source_accession": "GSE92742-TouchstoneV1.1"}
        if allowed_for_accession_role(first_row, ledger, "mainline_observation"):
            _blocker(blockers, "default_loader_permits_deferred_source", "role_loader", "Deferred source was accepted without a delta.")
        if not allowed_for_accession_role(first_row, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "scoped_delta_not_consumable", "role_loader", "Qualified accession cannot enter mainline with its delta.")
        if allowed_for_accession_role(mirror_row, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "mirror_source_promoted", "role_loader", "ArrayExpress mirror was promoted by the GEO accession delta.")
        if allowed_for_accession_role(weak_row, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "weak_source_promoted", "role_loader", "Weak-calibration CLUE source entered mainline.")

        input_bindings = manifest.get("input_bindings", [])
        if not isinstance(input_bindings, list):
            _blocker(blockers, "input_bindings_invalid", "input_bindings", "Input bindings must be a list.")
        else:
            bound_paths = {str(row.get("path", "")) for row in input_bindings if isinstance(row, Mapping)}
            required_paths = {str(candidate["raw_value_path"]), str(candidate["platform_annotation_path"])}
            required_paths.update(str(value) for value in candidate["release_evidence_paths"])
            required_paths.update(str(value) for value in candidate["raw_mirror_paths"])
            if not required_paths.issubset(bound_paths):
                _blocker(blockers, "input_provenance_incomplete", "input_bindings", "Manifest lacks required official input bindings.")
            for binding in input_bindings:
                if not isinstance(binding, Mapping):
                    _blocker(blockers, "input_binding_invalid", "input_bindings", "Input binding is not an object.")
                    continue
                path = _path(str(binding.get("path", "")))
                if not path.is_file() or binding.get("sha256") != sha256_file(path) or binding.get("bytes") != path.stat().st_size:
                    _blocker(blockers, "input_binding_mismatch", str(binding.get("path", "")), "Official input checksum or size does not match.")

        block_files = manifest.get("block_files", [])
        if not isinstance(block_files, list) or len(block_files) != 4:
            _blocker(blockers, "block_file_set_invalid", "block_files", "Expected four canonical Parquet block files.")
            block_files = []
        for record in block_files:
            if not isinstance(record, Mapping):
                _blocker(blockers, "block_file_record_invalid", "block_files", "Block file record is not an object.")
                continue
            path = manifest_path.parent / str(record.get("name", ""))
            if not path.is_file() or record.get("sha256") != sha256_file(path) or record.get("bytes") != path.stat().st_size:
                _blocker(blockers, "block_file_binding_mismatch", str(record.get("name", "")), "Canonical block file checksum or size does not match.")
        if block_files and manifest.get("block_sha256") != _content_hash([row for row in block_files if isinstance(row, Mapping)]):
            _blocker(blockers, "block_content_hash_mismatch", "block_sha256", "Block hash is not derived from its Parquet files.")

        values = pd.read_parquet(manifest_path.parent / "values.parquet")
        samples = pd.read_parquet(manifest_path.parent / "sample_metadata.parquet")
        features = pd.read_parquet(manifest_path.parent / "feature_metadata.parquet")
        mapping = pd.read_parquet(manifest_path.parent / "source_probe_mapping.parquet")
        expected_samples = int(policy["qc"]["expected_sample_count"])
        if len(samples) != expected_samples or samples["biological_unit_id"].nunique() != expected_samples:
            _blocker(blockers, "biological_unit_ess_invalid", "sample_metadata.parquet", "ESS must count 30 unique biological specimens.")
        if not (samples["subject_or_donor_status"] == "not_provided").all():
            _blocker(blockers, "subject_identity_inferred", "sample_metadata.parquet", "Unprovided donor IDs must remain explicit.")
        if not (samples["technical_replicate_group"] == samples["biological_unit_id"]).all():
            _blocker(blockers, "technical_replicate_policy_invalid", "sample_metadata.parquet", "Technical replicate grouping differs from source-declared one-array-per-specimen policy.")
        if len(features) < int(policy["qc"]["minimum_mapped_gene_count"]):
            _blocker(blockers, "mapped_gene_count_below_threshold", "feature_metadata.parquet", "Mapped gene count is below QC threshold.")
        if features["canonical_gene_id"].duplicated().any() or features["canonical_gene_id"].astype(str).str.fullmatch(r"\d+").eq(False).any():
            _blocker(blockers, "feature_mapping_invalid", "feature_metadata.parquet", "Features must be unique Entrez Gene IDs.")
        if mapping.empty or mapping["canonical_gene_id"].isna().any():
            _blocker(blockers, "probe_mapping_missing", "source_probe_mapping.parquet", "Official probe-to-gene mapping is missing.")
        expected_rows = len(samples) * len(features)
        if len(values) != expected_rows:
            _blocker(blockers, "value_row_count_invalid", "values.parquet", "Canonical value row count does not equal samples times genes.")
        if not np.isfinite(values["value"].to_numpy(dtype=np.float64)).all():
            _blocker(blockers, "nonfinite_expression_values", "values.parquet", "Canonical expression values must be finite.")
        if not values["detection_p_value"].between(0.0, 1.0).all():
            _blocker(blockers, "detection_probability_invalid", "values.parquet", "Detection p values must be in [0, 1].")
        if set(values["value_unit_canonical"].unique()) != {"MAS5-calculated Signal intensity"}:
            _blocker(blockers, "unit_mapping_invalid", "values.parquet", "Canonical measurement unit is not preserved.")
        if not set(values["biological_unit_id"].unique()) == set(samples["biological_unit_id"]):
            _blocker(blockers, "value_sample_axis_invalid", "values.parquet", "Value rows do not cover exactly the declared biological units.")
        if manifest.get("effective_sample_size") != float(expected_samples):
            _blocker(blockers, "manifest_ess_mismatch", "effective_sample_size", "Manifest ESS must equal unique biological specimen count.")
        if manifest.get("qc", {}).get("status") != "passed":
            _blocker(blockers, "qc_not_passed", "qc.status", "Manifest QC status must be passed.")
        if manifest.get("mirror_provenance", {}).get("counts_as_independent_source") is not False:
            _blocker(blockers, "mirror_independence_overclaim", "mirror_provenance", "Same-study mirror cannot count as an independent source.")
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as exc:
        _blocker(blockers, "validation_exception", "$", str(exc))

    return {
        "schema_version": "neuronsifter-phase1e-mainline-block-validation-v1",
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
    parser = argparse.ArgumentParser(description="Validate the GSE28146 Phase 1E canonical mainline block.")
    parser.add_argument("--output", default="Output/audits/phase1e_gse28146_mainline_block_validation.json")
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(_path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
