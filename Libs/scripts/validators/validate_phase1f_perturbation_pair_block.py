"""Validate the independent E-MTAB-6772 perturbation mainline block."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator, FormatChecker
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.materialize_phase1f_perturbation_pair_block import (
    CANONICAL_UID,
    MANIFEST_VERSION,
    _canonicalize,
    _content_hash,
    _read_adf,
    _read_assay,
)
from Libs.utils.data.public_foundation_corpus import hash_path, validate_semantic_axes
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import (
    allowed_for_accession_role,
    load_accession_requalification_delta,
    load_phase1a_ledger,
)
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.utils.model.dynamics_store import load_zarr_full_array, read_zarr_array_metadata


VALIDATOR_ID = "Libs/scripts/validators/validate_phase1f_perturbation_pair_block.py"


def _path(value: str | Path, root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _blocker(blockers: list[dict[str, str]], code: str, path: str, message: str) -> None:
    blockers.append({"code": code, "path": path, "message": message})


def _nonblank(frame: pd.DataFrame, fields: list[str]) -> bool:
    return all(field in frame and not frame[field].astype(str).str.strip().eq("").any() for field in fields)


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    blockers: list[dict[str, str]] = []
    manifest: dict[str, object] = {}
    try:
        config = load_config(root / "Libs/configs/config_stack.yaml")
        policy = config["public_foundation_corpus"]["phase1f_perturbation_pair_materialization"]
        candidate = policy["candidate"]
        qc = policy["qc"]
        action = policy["action"]
        admission = _load_json(_path(str(policy["admission_audit_path"]), root))
        if admission.get("status") != "accepted_second_independent_perturbation_block" or admission.get("independence_status") != "passed_independent_pair_with_clue_m1":
            _blocker(blockers, "pair_admission_invalid", "admission", "E-MTAB-6772 is not accepted as the independent perturbation pair")
        binding = admission.get("canonical_block_manifest")
        if not isinstance(binding, Mapping):
            raise ValueError("Admission lacks canonical_block_manifest binding")
        manifest_path = _path(str(binding.get("path", "")), root)
        if not manifest_path.is_file() or binding.get("bytes") != manifest_path.stat().st_size or binding.get("sha256") != sha256_file(manifest_path):
            raise ValueError("Admission manifest binding is stale")
        manifest = _load_json(manifest_path)

        schema = _load_json(_path(str(policy["manifest_schema_path"]), root))
        Draft202012Validator.check_schema(schema)
        for error in sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(manifest), key=lambda item: list(item.absolute_path)):
            _blocker(blockers, "manifest_schema_invalid", ".".join(map(str, error.absolute_path)) or "$", error.message)
        expected_identity = {
            "schema_version": MANIFEST_VERSION,
            "canonical_uid": CANONICAL_UID,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "qc_grade": "B",
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                _blocker(blockers, "manifest_identity_mismatch", field, f"Expected {expected!r}, got {manifest.get(field)!r}")
        for field in ("admission_allowed", "real_value_matrix", "compound_dose_time_axes_complete", "action_linkage_complete", "semantic_axes_complete", "cns_source"):
            if manifest.get(field) is not True:
                _blocker(blockers, "mainline_contract_false", field, f"{field} must be true")
        if manifest.get("role") != "qualified_for_mainline" or manifest.get("split_role") != "unassigned_pending_nested_split_freeze":
            _blocker(blockers, "role_or_split_invalid", "role", "Block must be mainline with splits still unassigned")
        for code in validate_semantic_axes(manifest.get("semantic_axes", {})):
            _blocker(blockers, "semantic_axis_invalid", "semantic_axes", code)

        input_bindings = manifest.get("input_bindings", [])
        if not isinstance(input_bindings, list):
            raise ValueError("input_bindings must be a list")
        for row in input_bindings:
            if not isinstance(row, Mapping):
                _blocker(blockers, "input_binding_invalid", "input_bindings", "Binding is not an object")
                continue
            path = _path(str(row.get("path", "")), root)
            if not path.is_file() or row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
                _blocker(blockers, "input_binding_drift", str(row.get("path", "")), "Source binding differs from current bytes")
        adf_path = _path(str(candidate["platform_annotation_path"]), root)
        if adf_path.stat().st_size != int(candidate["expected_platform_annotation_bytes"]) or sha256_file(adf_path) != candidate["expected_platform_annotation_sha256"]:
            _blocker(blockers, "adf_release_binding_invalid", str(candidate["platform_annotation_path"]), "ADF bytes or checksum differ from frozen source contract")

        records = manifest.get("block_files", [])
        expected_files = {"expression.zarr", "observations.parquet", "features.parquet", "controls.parquet", "actions.parquet", "feature_quarantine.parquet", "matrix_manifest.json"}
        if not isinstance(records, list) or {str(row.get("name", "")) for row in records if isinstance(row, Mapping)} != expected_files:
            _blocker(blockers, "block_file_set_invalid", "block_files", "Canonical payload set is incomplete")
            records = []
        for row in records:
            if not isinstance(row, Mapping):
                continue
            file_count, byte_count, digest = hash_path(manifest_path.parent / str(row.get("relative_path", "")))
            if row.get("file_count") != file_count or row.get("bytes") != byte_count or row.get("sha256") != digest:
                _blocker(blockers, "block_file_binding_drift", str(row.get("name", "")), "Canonical file binding differs")
        if records and manifest.get("block_sha256") != _content_hash([row for row in records if isinstance(row, Mapping)]):
            _blocker(blockers, "block_hash_invalid", "block_sha256", "Block hash does not derive from bound files")

        observations = pd.read_parquet(manifest_path.parent / "observations.parquet")
        controls = pd.read_parquet(manifest_path.parent / "controls.parquet")
        features = pd.read_parquet(manifest_path.parent / "features.parquet")
        actions = pd.read_parquet(manifest_path.parent / "actions.parquet")
        quarantine = pd.read_parquet(manifest_path.parent / "feature_quarantine.parquet")
        matrix_manifest = _load_json(manifest_path.parent / "matrix_manifest.json")
        observation_fields = ["observation_uid", "source_uid", "publication_uid", "accession_uid", "biological_unit_uid", "sample_uid", "cell_or_tissue_context", "compound_uid", "dose_value", "dose_unit", "time_value", "time_unit", "schedule", "route", "assay_uid", "feature_namespace", "qc_grade", "action_uid", "replicate_group_uid", "control_group_uid", "value_row_index"]
        if not _nonblank(observations, observation_fields):
            _blocker(blockers, "observation_contract_incomplete", "observations.parquet", "Required observation fields are blank or absent")
        if len(observations) != int(qc["expected_treated_count"]) or observations["observation_uid"].duplicated().any() or observations["biological_unit_uid"].duplicated().any():
            _blocker(blockers, "treated_identity_invalid", "observations.parquet", "Six unique treated observations and biological units are required")
        if len(controls) != int(qc["expected_control_count"]) or controls["counted_in_ess"].any() or controls["counted_in_observation_block"].any():
            _blocker(blockers, "control_lineage_invalid", "controls.parquet", "Six vehicle controls must remain lineage-only")
        if set(observations["control_group_uid"]) != set(controls["control_group_uid"]):
            _blocker(blockers, "control_pairing_invalid", "controls.parquet", "Treated samples lack genotype-matched control groups")
        if set(observations["compound_uid"]) != {candidate["compound_uid"]} or set(observations["dose_value"]) != {float(action["physical_dose"])} or set(observations["dose_unit"]) != {action["dose_unit"]}:
            _blocker(blockers, "compound_dose_invalid", "observations.parquet", "Pantethine 15 mg per-administration semantics differ")
        if "mg/kg" in set(observations["dose_unit"].astype(str)) or set(observations["time_value"]) != {float(action["duration"])} or set(observations["time_unit"]) != {action["duration_unit"]}:
            _blocker(blockers, "dose_time_coercion", "observations.parquet", "Dose must remain mg and duration must remain 5.5 months")
        if set(observations["schedule"]) != {"three_times_per_week"} or set(observations["route"]) != {"intraperitoneal"}:
            _blocker(blockers, "schedule_route_invalid", "observations.parquet", "Peer-reviewed schedule and route are not preserved")
        if len(features) != int(qc["expected_canonical_gene_count"]) or features["feature_id"].duplicated().any() or not features["feature_id"].str.fullmatch(r"NCBIGene:\d+").all():
            _blocker(blockers, "feature_axis_invalid", "features.parquet", "Exact Entrez median feature axis is invalid")
        reasons = quarantine["mapping_status"].value_counts().to_dict()
        if reasons != {"unmatched_adf_reporter": int(qc["expected_unmatched_probe_count"]), "non_numeric_or_missing_adf_entrez": int(qc["expected_non_numeric_entrez_count"])}:
            _blocker(blockers, "feature_quarantine_invalid", "feature_quarantine.parquet", "Unmatched/non-numeric source features are not partitioned exactly")

        action_fields = ["action_uid", "agent_uid", "target_set_or_explicit_unknown", "physical_dose", "dose_unit", "duration", "duration_unit", "schedule", "route_or_not_applicable", "cell_or_tissue_context", "assay_uid", "publication_uid", "accession_uid", "exposure_program_ref_or_explicit_unavailable", "occupancy_program_ref_or_explicit_unavailable"]
        if len(actions) != 1 or not _nonblank(actions, action_fields) or actions.iloc[0]["linked_observation_count"] != len(observations):
            _blocker(blockers, "action_contract_invalid", "actions.parquet", "One complete action linked to six observations is required")
        if set(actions["target_set_or_explicit_unknown"]) != {"explicit_unknown"} or set(actions["exposure_program_ref_or_explicit_unavailable"]) != {"explicit_unavailable"} or set(actions["occupancy_program_ref_or_explicit_unavailable"]) != {"explicit_unavailable"}:
            _blocker(blockers, "action_unavailable_semantics_invalid", "actions.parquet", "Target/exposure/occupancy unavailable values must be explicit")
        if set(actions["action_uid"]) != set(observations["action_uid"]) or set(actions["schedule"]) != {"three_times_per_week"} or set(actions["route"]) != {"intraperitoneal"}:
            _blocker(blockers, "action_observation_linkage_invalid", "actions.parquet", "Action identity, schedule or route differs from observations")

        conflicts = manifest.get("protocol_conflicts", [])
        conflict_ids = {row.get("conflict_id") for row in conflicts if isinstance(row, Mapping)}
        if conflict_ids != {"treatment_schedule_frequency", "pooled_sample_wording"}:
            _blocker(blockers, "protocol_conflicts_missing", "protocol_conflicts", "Both source conflicts must remain explicit")
        schedule_conflict = next((row for row in conflicts if isinstance(row, Mapping) and row.get("conflict_id") == "treatment_schedule_frequency"), {})
        if schedule_conflict.get("idf_value") != "two_times_per_week" or schedule_conflict.get("publication_methods_value") != "three_times_per_week":
            _blocker(blockers, "schedule_conflict_invalid", "protocol_conflicts", "IDF/publication schedule conflict was hidden or altered")

        pair = manifest.get("pair_independence", {})
        if not isinstance(pair, Mapping) or pair.get("status") != "passed" or pair.get("sample_uid_overlap_count") != 0 or pair.get("biological_unit_overlap_count") != 0 or pair.get("mirror_overlap") is not False or pair.get("identity_equal_fields") != []:
            _blocker(blockers, "pair_independence_invalid", "pair_independence", "CLUE/E-MTAB-6772 identity independence is not closed")
        m1_binding = pair.get("m1_manifest", {}) if isinstance(pair, Mapping) else {}
        m1_path = _path(str(m1_binding.get("path", "")), root)
        if not m1_path.is_file() or m1_binding.get("bytes") != m1_path.stat().st_size or m1_binding.get("sha256") != sha256_file(m1_path):
            _blocker(blockers, "m1_binding_invalid", "pair_independence.m1_manifest", "CLUE M1 binding is stale")

        ledger_path = _path(str(policy["source_ledger_path"]), root)
        delta_path = _path(str(policy["qualification_delta_path"]), root)
        ledger = load_phase1a_ledger(ledger_path)
        delta = load_accession_requalification_delta(delta_path)
        qualified = {"source_id": candidate["source_id"], "source_accession": candidate["source_accession"]}
        unrelated = {"source_id": candidate["source_id"], "source_accession": "E-MTAB-UNQUALIFIED"}
        if ledger[str(candidate["source_id"])]["decision"] != "deferred" or allowed_for_accession_role(qualified, ledger, "mainline_observation"):
            _blocker(blockers, "source_wide_role_changed", "role_loader", "ArrayExpress source-wide deferred role must remain closed")
        if not allowed_for_accession_role(qualified, ledger, "mainline_observation", requalification_delta=delta) or allowed_for_accession_role(unrelated, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "accession_scope_invalid", "role_loader", "Only E-MTAB-6772 may be mainline-readable")
        delta_row = delta.get((str(candidate["source_id"]), str(candidate["source_accession"])))
        if delta_row is None or delta_row.get("block_manifest_path") != manifest_path.relative_to(root).as_posix() or delta_row.get("block_manifest_sha256") != sha256_file(manifest_path):
            _blocker(blockers, "delta_manifest_binding_invalid", "source_qualification_delta.tsv", "Accession delta does not bind current manifest")

        assay_paths = sorted(_path(str(candidate["official_snapshot_root"]), root).glob("[0-9]*_?.txt"))
        assay_by_name = {path.name: path for path in assay_paths}
        raw_rows: list[np.ndarray] = []
        probe_ids: list[str] | None = None
        for value_file in observations["source_value_file"]:
            current_probes, current_values = _read_assay(assay_by_name[str(value_file)])
            if probe_ids is None:
                probe_ids = current_probes
            elif probe_ids != current_probes:
                raise ValueError("Treated assay probe axes differ")
            raw_rows.append(current_values)
        assert probe_ids is not None
        _, _, expected_values, diagnostics = _canonicalize(probe_ids, np.vstack(raw_rows), _read_adf(_path(str(candidate["platform_annotation_path"]), root)))
        stored_values = load_zarr_full_array(manifest_path.parent / "expression.zarr", "values")
        if not np.array_equal(stored_values, expected_values) or not np.isfinite(stored_values).all():
            _blocker(blockers, "matrix_value_lineage_invalid", "expression.zarr", "Stored values do not equal independently reconstructed exact-Entrez medians")
        zarray = read_zarr_array_metadata(manifest_path.parent / "expression.zarr", "values")
        if zarray.get("shape") != [len(observations), len(features)] or zarray.get("dtype") != "<f4" or matrix_manifest.get("mapping_diagnostics") != diagnostics:
            _blocker(blockers, "matrix_layout_invalid", "expression.zarr", "Zarr layout or matrix diagnostics differ")
        if manifest.get("effective_sample_size") != float(observations["biological_unit_uid"].nunique()) or manifest.get("effective_sample_size") <= 0:
            _blocker(blockers, "ess_invalid", "effective_sample_size", "ESS must be six unique treated biological samples")

        supporting = {
            "candidate_audit_path": "passed_candidate_selected",
            "qc_audit_path": "passed",
            "ess_audit_path": "passed",
            "action_linkage_path": "passed",
            "pair_independence_path": "passed",
        }
        for key, expected_status in supporting.items():
            audit = _load_json(_path(str(policy[key]), root))
            if audit.get("status") != expected_status or audit.get("canonical_uid") != CANONICAL_UID:
                _blocker(blockers, "supporting_audit_invalid", str(policy[key]), f"Expected {expected_status}")
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, pd.errors.ParserError) as exc:
        _blocker(blockers, "validation_exception", "$", str(exc))

    return {
        "schema_version": "neuronsifter-phase1f-perturbation-pair-validation-v1",
        "producer": VALIDATOR_ID,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "canonical_uid": manifest.get("canonical_uid"),
        "source_id": manifest.get("source_id"),
        "source_accession": manifest.get("source_accession"),
        "qc_grade": manifest.get("qc_grade"),
        "observation_count": manifest.get("observation_count"),
        "effective_sample_size": manifest.get("effective_sample_size"),
        "action_count": manifest.get("action_count"),
        "independent_perturbation_source_count": 2 if not blockers else 1,
        "g2_status": "not_passed",
        "g2_remaining_blockers": ["nested_split_freeze_pending", "real_domain_leakage_audit_pending"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the E-MTAB-6772 perturbation independent pair block.")
    parser.add_argument("--output", default="Output/audits/phase1f_perturbation_block_validation.json")
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(_path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
