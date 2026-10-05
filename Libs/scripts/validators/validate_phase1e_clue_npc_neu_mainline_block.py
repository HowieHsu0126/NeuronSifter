"""Validate the accession-scoped CLUE NPC/NEU perturbation M1 block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.materialize_phase1e_clue_npc_neu_mainline_block import (
    CANONICAL_UID,
    MANIFEST_VERSION,
    MATRIX_DATASET,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path, validate_semantic_axes
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import (
    allowed_for_accession_role,
    load_accession_requalification_delta,
    load_phase1a_ledger,
)
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.utils.model.dynamics_store import load_zarr_rows, read_zarr_array_metadata


VALIDATOR_ID = "Libs/scripts/validators/validate_phase1e_clue_npc_neu_mainline_block.py"


def _path(value: str | Path) -> Path:
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
        {
            "name": row["name"], "file_count": row["file_count"],
            "bytes": row["bytes"], "sha256": row["sha256"],
        }
        for row in records
    ])


def _nonblank(frame: pd.DataFrame, fields: list[str]) -> bool:
    return all(
        field in frame.columns and not frame[field].astype(str).str.strip().eq("").any()
        for field in fields
    )


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    config = load_config(root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_perturbation_materialization"]
    candidate = policy["candidate"]
    qc = policy["qc"]
    blockers: list[dict[str, str]] = []
    manifest: dict[str, object] = {}
    try:
        admission_path = _path(str(policy["admission_audit_path"]))
        admission = _load_json(admission_path)
        manifest_binding = admission.get("canonical_block_manifest")
        if not isinstance(manifest_binding, Mapping):
            raise ValueError("Admission decision lacks canonical block manifest binding")
        manifest_path = _path(str(manifest_binding.get("path", "")))
        if not manifest_path.is_file() or manifest_binding.get("bytes") != manifest_path.stat().st_size or manifest_binding.get("sha256") != sha256_file(manifest_path):
            raise ValueError("Admission decision canonical manifest binding has drifted")
        manifest = _load_json(manifest_path)

        expected_identity = {
            "schema_version": MANIFEST_VERSION,
            "canonical_uid": CANONICAL_UID,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "parent_accession": candidate["parent_accession"],
            "source_subset_id": candidate["source_subset_id"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "qc_grade": qc["grade"],
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                _blocker(blockers, "manifest_identity_mismatch", field, f"expected {expected!r}, got {manifest.get(field)!r}")
        for field in ("real_value_matrix", "compound_dose_time_axes_complete", "action_linkage_complete", "semantic_axes_complete", "admission_allowed", "cns_source"):
            if manifest.get(field) is not True:
                _blocker(blockers, "mainline_contract_false", field, f"{field} must be true")
        if manifest.get("role") != "qualified_for_mainline":
            _blocker(blockers, "mainline_role_missing", "role", "Perturbation subset must be accession-scoped mainline")
        if manifest.get("cns_context_class") != "CNS_derived_model":
            _blocker(blockers, "cns_context_class_invalid", "cns_context_class", "Only the preregistered CNS-derived model class is admitted")
        if manifest.get("split_role") != "unassigned_pending_nested_split_freeze":
            _blocker(blockers, "premature_split_assignment", "split_role", "M1 cannot assign formal splits")
        for code in validate_semantic_axes(manifest.get("semantic_axes", {})):
            _blocker(blockers, "semantic_axis_invalid", "semantic_axes", code)

        ledger_path = _path(str(policy["source_ledger_path"]))
        delta_path = _path(str(policy["qualification_delta_path"]))
        ledger = load_phase1a_ledger(ledger_path)
        delta = load_accession_requalification_delta(delta_path)
        source_row = ledger[str(candidate["source_id"])]
        if source_row["decision"] != "qualified_for_weak_calibration":
            _blocker(blockers, "source_wide_role_changed", "source_qualification_ledger.tsv", "Source-wide CLUE must remain weak calibration")
        delta_row = delta.get((str(candidate["source_id"]), str(candidate["source_accession"])))
        if delta_row is None:
            _blocker(blockers, "accession_delta_missing", "source_qualification_delta.tsv", "Qualified subset delta is absent")
        else:
            if delta_row.get("prior_decision") != "qualified_for_weak_calibration" or delta_row.get("decision") != "qualified_for_mainline":
                _blocker(blockers, "accession_delta_role_invalid", "source_qualification_delta.tsv", "CLUE subset must transition weak to mainline only")
            if delta_row.get("frozen_ledger_path") != str(policy["source_ledger_path"]) or delta_row.get("frozen_ledger_sha256") != sha256_file(ledger_path):
                _blocker(blockers, "frozen_ledger_binding_mismatch", "source_qualification_delta.tsv", "Delta does not bind frozen Phase 1A ledger")
            relative_manifest = manifest_path.relative_to(root).as_posix()
            if delta_row.get("block_manifest_path") != relative_manifest or delta_row.get("block_manifest_sha256") != sha256_file(manifest_path):
                _blocker(blockers, "delta_manifest_binding_mismatch", "source_qualification_delta.tsv", "Delta does not bind canonical manifest")
        qualified = {"source_id": candidate["source_id"], "source_accession": candidate["source_accession"]}
        parent = {"source_id": candidate["source_id"], "source_accession": candidate["parent_accession"]}
        unrelated = {"source_id": candidate["source_id"], "source_accession": "GSE92742-TouchstoneV1.1::UNQUALIFIED"}
        if allowed_for_accession_role(qualified, ledger, "mainline_observation"):
            _blocker(blockers, "weak_source_promoted_without_delta", "role_loader", "Source-wide weak role entered mainline")
        if not allowed_for_accession_role(qualified, ledger, "mainline_observation", requalification_delta=delta):
            _blocker(blockers, "qualified_subset_not_loadable", "role_loader", "Qualified perturbation subset is not mainline-readable")
        for label, row in (("parent", parent), ("unrelated", unrelated)):
            if allowed_for_accession_role(row, ledger, "mainline_observation", requalification_delta=delta):
                _blocker(blockers, "unscoped_clue_promotion", f"role_loader.{label}", "Accession delta widened beyond the qualified subset")
        if not allowed_for_accession_role(parent, ledger, "weak_calibration", requalification_delta=delta):
            _blocker(blockers, "source_wide_weak_role_lost", "role_loader.parent", "Parent CLUE source must remain weak-readable")

        audit_statuses = {
            "candidate_audit_path": "passed_candidate_selected",
            "qc_audit_path": "passed",
            "ess_audit_path": "passed",
            "action_linkage_path": "passed",
        }
        supporting_audits: dict[str, dict[str, object]] = {}
        for key, expected_status in audit_statuses.items():
            audit = _load_json(_path(str(policy[key])))
            supporting_audits[key] = audit
            if audit.get("status") != expected_status or audit.get("canonical_uid") != CANONICAL_UID:
                _blocker(blockers, "supporting_audit_invalid", str(policy[key]), f"Expected status {expected_status}")
        diagnostics = supporting_audits["candidate_audit_path"].get("diagnostics")
        expected_diagnostics = {
            "source_subset_signature_count": int(qc["expected_source_subset_signature_count"]),
            "replicate_control_valid_signature_count": int(qc["expected_replicate_control_valid_signature_count"]),
            "duplicate_condition_signature_count": int(qc["expected_duplicate_condition_signature_count"]),
            "selected_condition_count": int(qc["expected_selected_condition_count"]),
            "selected_context_counts": dict(qc["expected_selected_context_counts"]),
            "quarantine_count": int(qc["expected_quarantine_count"]),
            "quarantine_reason_counts": dict(qc["expected_quarantine_reason_counts"]),
            "shsy5y_signature_count": int(qc["expected_shsy5y_signature_count"]),
            "shsy5y_compound_signature_count": int(qc["expected_shsy5y_compound_signature_count"]),
        }
        if not isinstance(diagnostics, Mapping) or {
            key: diagnostics.get(key) for key in expected_diagnostics
        } != expected_diagnostics:
            _blocker(
                blockers,
                "selection_count_drift",
                str(policy["candidate_audit_path"]),
                "Accession-scoped signature, quarantine, or condition counts differ from the frozen real-source selection",
            )
        else:
            cns_evidence = diagnostics.get("cns_applicability_evidence")
            if not isinstance(cns_evidence, Mapping):
                _blocker(blockers, "cns_evidence_missing", "diagnostics.cns_applicability_evidence", "Official cell metadata evidence is required")
            else:
                npc = cns_evidence.get("NPC")
                neu = cns_evidence.get("NEU")
                if not isinstance(npc, Mapping) or npc.get("primary_site") != "central nervous system" or "iPSC" not in str(npc.get("modification", "")):
                    _blocker(blockers, "npc_cns_evidence_invalid", "diagnostics.cns_applicability_evidence.NPC", "NPC CNS-derived identity must come from official cell metadata")
                if not isinstance(neu, Mapping) or neu.get("cell_type") != "differentiated" or "neurons" not in str(neu.get("modification", "")):
                    _blocker(blockers, "neu_cns_evidence_invalid", "diagnostics.cns_applicability_evidence.NEU", "NEU neuronal identity must come from official cell metadata")
        audit_manifest = _load_json(_path(str(policy["block_manifest_audit_path"])))
        if audit_manifest != manifest:
            _blocker(blockers, "block_manifest_audit_drift", str(policy["block_manifest_audit_path"]), "Audit copy does not equal canonical manifest")
        if admission.get("status") != "accepted_first_qualified_perturbation_block" or admission.get("source_family_role_unchanged") != "qualified_for_weak_calibration":
            _blocker(blockers, "admission_decision_invalid", str(policy["admission_audit_path"]), "Admission must remain subset-scoped")
        license_role = admission.get("license_role")
        if license_role != policy["license_role"]:
            _blocker(blockers, "license_role_drift", "license_role", "Admission license role differs from frozen config")

        input_bindings = manifest.get("input_bindings")
        if not isinstance(input_bindings, list):
            raise ValueError("Manifest input_bindings must be a list")
        required_inputs = {
            str(candidate["raw_value_path"]), str(candidate["raw_archive_path"]),
            *map(str, candidate["metadata_paths"]), *map(str, candidate["release_evidence_paths"]),
        }
        bound_paths = {str(row.get("path", "")) for row in input_bindings if isinstance(row, Mapping)}
        if required_inputs != bound_paths:
            _blocker(blockers, "input_binding_set_mismatch", "input_bindings", "Raw, archive, metadata, and release binding set must be exact")
        for binding in input_bindings:
            if not isinstance(binding, Mapping):
                _blocker(blockers, "input_binding_invalid", "input_bindings", "Input binding is not an object")
                continue
            path = _path(str(binding.get("path", "")))
            if not path.is_file() or binding.get("bytes") != path.stat().st_size or binding.get("sha256") != sha256_file(path):
                _blocker(blockers, "input_binding_drift", str(binding.get("path", "")), "Bound source bytes have drifted")

        records = manifest.get("block_files")
        expected_files = {"expression.zarr", "observations.parquet", "features.parquet", "actions.parquet", "quarantine.parquet", "matrix_manifest.json"}
        if not isinstance(records, list) or {str(row.get("name", "")) for row in records if isinstance(row, Mapping)} != expected_files:
            _blocker(blockers, "block_file_set_invalid", "block_files", "Canonical perturbation payload set is incomplete")
            records = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            path = manifest_path.parent / str(record.get("relative_path", record.get("name", "")))
            file_count, byte_count, digest = hash_path(path)
            if record.get("file_count") != file_count or record.get("bytes") != byte_count or record.get("sha256") != digest:
                _blocker(blockers, "block_file_binding_drift", str(record.get("name", "")), "Canonical payload binding has drifted")
        if records and manifest.get("block_sha256") != _content_hash([row for row in records if isinstance(row, Mapping)]):
            _blocker(blockers, "block_content_hash_mismatch", "block_sha256", "Block hash does not derive from canonical payloads")

        observations = pd.read_parquet(manifest_path.parent / "observations.parquet")
        features = pd.read_parquet(manifest_path.parent / "features.parquet")
        actions = pd.read_parquet(manifest_path.parent / "actions.parquet")
        quarantine = pd.read_parquet(manifest_path.parent / "quarantine.parquet")
        matrix = _load_json(manifest_path.parent / "matrix_manifest.json")
        observation_fields = [
            "observation_uid", "source_uid", "publication_uid", "accession_uid",
            "biological_unit_uid", "subject_or_donor_uid", "sample_uid", "cell_or_tissue_context", "compound_uid",
            "dose_value", "dose_unit", "time_value", "time_unit", "assay_uid",
            "feature_namespace", "value_unit", "qc_grade", "action_uid",
            "replicate_group_uid", "source_instance_ids", "control_group_uid",
            "control_instance_ids", "rna_plate_ids", "gctx_column_index", "value_row_index",
        ]
        if not _nonblank(observations, observation_fields):
            _blocker(blockers, "observation_contract_incomplete", "observations.parquet", "Required observation fields are absent or blank")
        join_contract = manifest.get("observation_value_join_contract")
        if not isinstance(join_contract, Mapping) or join_contract.get("value_axis") != "expression.zarr:values[value_row_index,feature_index]" or join_contract.get("feature_axis") != "features.parquet:feature_index|feature_id":
            _blocker(blockers, "observation_value_join_contract_invalid", "observation_value_join_contract", "Matrix-native feature/value linkage must be explicit")
        if len(observations) < int(qc["minimum_selected_condition_count"]):
            _blocker(blockers, "condition_count_below_threshold", "observations.parquet", "Qualified condition count is below threshold")
        if observations["observation_uid"].duplicated().any() or observations["biological_unit_uid"].duplicated().any() or observations["action_uid"].duplicated().any():
            _blocker(blockers, "condition_identity_not_unique", "observations.parquet", "Observation, biological unit, and action IDs must be unique after collapse")
        if set(observations["cell_id"]) != set(candidate["eligible_cell_ids"]) or set(observations["cns_context_class"]) != {"CNS_derived_model"}:
            _blocker(blockers, "unqualified_context_present", "observations.parquet", "Only NPC/NEU CNS-derived contexts are permitted")
        if not observations["dose_value"].gt(0).all() or set(observations["dose_unit"]) != {qc["canonical_dose_unit"]}:
            _blocker(blockers, "dose_axis_invalid", "observations.parquet", "Physical positive uM dose is required")
        if not observations["time_value"].gt(0).all() or set(observations["time_unit"]) != {qc["canonical_time_unit"]}:
            _blocker(blockers, "time_axis_invalid", "observations.parquet", "Physical positive treatment duration in h is required")
        if not observations["source_instance_count"].ge(int(qc["minimum_source_instance_count"])).all():
            _blocker(blockers, "replicate_group_invalid", "observations.parquet", "Each Level5 condition needs at least two source instances")
        if not observations["gctx_column_index"].is_monotonic_increasing or observations["gctx_column_index"].duplicated().any():
            _blocker(blockers, "matrix_index_invalid", "observations.parquet", "GCTX row indices must be sorted and unique")
        expected_rows = np.arange(len(observations), dtype=np.int64)
        if not np.array_equal(observations["value_row_index"].to_numpy(dtype=np.int64), expected_rows):
            _blocker(blockers, "value_row_index_invalid", "observations.parquet", "Canonical value row index must be contiguous")

        action_fields = [
            "action_uid", "intervention_type", "agent_uid", "agent_set",
            "target_set_or_explicit_unknown", "target_set", "physical_dose", "dose_value",
            "dose_unit", "duration", "duration_unit", "schedule", "route_or_not_applicable",
            "route", "cell_or_tissue_context", "assay_uid", "assay_context", "state_scope",
            "publication_uid", "source_publication", "accession_uid", "source_accession",
            "exposure_program_ref_or_explicit_unavailable",
            "exposure_program_ref", "occupancy_program_ref_or_explicit_unavailable",
            "occupancy_program_ref", "observation_uid", "control_group_uid",
        ]
        if not _nonblank(actions, action_fields) or len(actions) != len(observations):
            _blocker(blockers, "action_contract_incomplete", "actions.parquet", "Action contract must be complete and one-to-one")
        if actions["action_uid"].duplicated().any() or set(actions["action_uid"]) != set(observations["action_uid"]):
            _blocker(blockers, "action_linkage_mismatch", "actions.parquet", "Actions do not match observation action IDs")
        action_observation = actions.merge(
            observations[
                [
                    "action_uid", "compound_uid", "dose_value", "dose_unit", "time_value", "time_unit",
                    "cell_or_tissue_context", "assay_uid", "publication_uid", "accession_uid", "control_group_uid",
                ]
            ],
            on="action_uid",
            how="inner",
            suffixes=("_action", "_observation"),
            validate="one_to_one",
        )
        aligned_fields = (
            ("agent_uid", "compound_uid"),
            ("physical_dose", "dose_value_observation"),
            ("dose_unit_action", "dose_unit_observation"),
            ("duration", "time_value"),
            ("duration_unit", "time_unit"),
            ("cell_or_tissue_context_action", "cell_or_tissue_context_observation"),
            ("assay_uid_action", "assay_uid_observation"),
            ("publication_uid_action", "publication_uid_observation"),
            ("accession_uid_action", "accession_uid_observation"),
            ("control_group_uid_action", "control_group_uid_observation"),
        )
        if len(action_observation) != len(observations) or any(
            not action_observation[left].astype(str).equals(action_observation[right].astype(str))
            for left, right in aligned_fields
        ):
            _blocker(blockers, "action_semantics_mismatch", "actions.parquet", "Action dose, duration, agent, context, provenance, or control linkage differs from its observation")
        if set(actions["target_set_or_explicit_unknown"]) != {"explicit_unknown"}:
            _blocker(blockers, "target_unknown_not_explicit", "actions.parquet", "Unavailable targets must be explicit_unknown")
        for field in ("exposure_program_ref_or_explicit_unavailable", "occupancy_program_ref_or_explicit_unavailable"):
            if set(actions[field]) != {"explicit_unavailable"}:
                _blocker(blockers, "program_reference_not_explicit", f"actions.parquet:{field}", "Unavailable program reference must be explicit")
        if set(actions["schedule"]) != {"explicit_unavailable"} or set(actions["route_or_not_applicable"]) != {"not_applicable_in_vitro"}:
            _blocker(blockers, "schedule_route_semantics_invalid", "actions.parquet", "Schedule and route may not be guessed")

        if len(features) != int(qc["expected_feature_count"]) or not features["feature_id"].astype(str).str.fullmatch(r"NCBIGene:\d+").all() or features["feature_id"].duplicated().any():
            _blocker(blockers, "feature_axis_invalid", "features.parquet", "All 12,328 unique Entrez features are required")
        if "duplicate_condition_signature_not_counted" not in set(quarantine["quarantine_reason"]):
            _blocker(blockers, "duplicate_condition_quarantine_missing", "quarantine.parquet", "Duplicate conditions must be explicit quarantine rows")
        if "SHSY5Y" in set(observations["cell_id"]):
            _blocker(blockers, "shsy5y_leaked", "observations.parquet", "SHSY5Y genetic-only signatures cannot enter compound M1")

        ess = float(observations["biological_unit_uid"].nunique())
        if manifest.get("effective_sample_size") != ess or manifest.get("biological_unit_count") != int(ess) or ess <= 0:
            _blocker(blockers, "ess_invalid", "effective_sample_size", "ESS must equal unique condition biological units")
        if manifest.get("observation_count") != len(observations) or manifest.get("action_count") != len(actions):
            _blocker(blockers, "manifest_count_mismatch", "observation_count", "Manifest counts differ from canonical tables")

        store = manifest_path.parent / "expression.zarr"
        zarray = read_zarr_array_metadata(store, "values")
        expected_shape = [len(observations), len(features)]
        if zarray.get("shape") != expected_shape or zarray.get("chunks") != [int(policy["parser"]["row_chunk_size"]), len(features)] or zarray.get("dtype") != "<f4":
            _blocker(blockers, "zarr_layout_invalid", "expression.zarr", "Zarr shape, chunks, or dtype differ from contract")
        if matrix.get("matrix_shape") != expected_shape or matrix.get("scalar_value_count") != len(observations) * len(features) or matrix.get("finite_values") is not True:
            _blocker(blockers, "matrix_manifest_invalid", "matrix_manifest.json", "Matrix statistics do not match axes")
        sample_rows = np.array([0, len(observations) - 1], dtype=np.int64)
        stored = load_zarr_rows(store, "values", sample_rows)
        with h5py.File(_path(str(candidate["raw_value_path"])), "r") as handle:
            source = np.asarray(handle[MATRIX_DATASET][observations.iloc[sample_rows]["gctx_column_index"].to_numpy(dtype=np.int64), :], dtype=np.float32)
        if not np.array_equal(stored, source) or not np.isfinite(stored).all():
            _blocker(blockers, "matrix_value_lineage_mismatch", "expression.zarr", "Stored values do not equal source GCTX rows")

        sig = pd.read_csv(_path(str(candidate["metadata_paths"][0])), sep="\t", dtype=str, keep_default_na=False)
        selected_sig_ids = set(observations["source_sig_id"].astype(str))
        selected_sig = sig[sig["sig_id"].isin(selected_sig_ids)]
        if len(selected_sig) != len(observations) or set(selected_sig["pert_type"]) != {"trt_cp"} or set(selected_sig["cell_id"]) != set(candidate["eligible_cell_ids"]):
            _blocker(blockers, "signature_metadata_lineage_invalid", "sig_info", "Selected signatures do not reconstruct trt_cp NPC/NEU observations")
        shsy5y = sig[sig["cell_id"].eq("SHSY5Y")]
        if shsy5y["pert_type"].eq("trt_cp").any():
            _blocker(blockers, "shsy5y_candidate_assumption_drift", "sig_info", "SHSY5Y unexpectedly acquired compound signatures")

        inst = pd.read_csv(_path(str(candidate["metadata_paths"][1])), sep="\t", dtype=str, keep_default_na=False)
        control_ids = {
            value
            for group in observations["control_instance_ids"].astype(str)
            for value in group.split("|")
            if value
        }
        controls = inst[inst["inst_id"].isin(control_ids)].copy()
        if len(controls) != len(control_ids) or set(controls["pert_type"]) != {"ctl_vehicle"}:
            _blocker(blockers, "control_instance_lineage_invalid", "inst_info", "Every bound control must resolve uniquely to a vehicle-control instance")
        else:
            controls_by_id = controls.set_index("inst_id", drop=False)
            for observation in observations.to_dict(orient="records"):
                expected_time = float(observation["time_value"])
                expected_unit = str(observation["time_unit"])
                expected_cell = str(observation["cell_id"])
                expected_plates = set(str(observation["rna_plate_ids"]).split("|"))
                for control_id in str(observation["control_instance_ids"]).split("|"):
                    control = controls_by_id.loc[control_id]
                    if (
                        str(control["cell_id"]) != expected_cell
                        or str(control["rna_plate"]) not in expected_plates
                        or str(control["pert_time_unit"]) != expected_unit
                        or not np.isclose(float(control["pert_time"]), expected_time)
                    ):
                        _blocker(blockers, "control_pairing_mismatch", f"inst_info:{control_id}", "Vehicle control must match treatment cell, RNA plate, time, and time unit")
                        break
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        pd.errors.MergeError,
        pd.errors.ParserError,
    ) as exc:
        _blocker(blockers, "validation_exception", "$", str(exc))

    return {
        "schema_version": "neuronsifter-phase1e-clue-npc-neu-perturbation-validation-v1",
        "producer": VALIDATOR_ID,
        "validated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
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
        "g2_status": "not_passed",
        "g2_remaining_blockers": manifest.get("blocker_codes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CLUE NPC/NEU perturbation M1 block.")
    parser.add_argument("--output", default="Output/audits/g2_m1_perturbation_block_validation.json")
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(_path(args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
