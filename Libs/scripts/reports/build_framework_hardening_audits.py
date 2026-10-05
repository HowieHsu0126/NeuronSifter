from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.platform_contracts import measurement_semantics_rows, sample_observation_rows, world_state_ready_rows, crossmodal_anchor_rows, simulation_teacher_rows
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build framework hardening audit artifacts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def measurement_semantics_audit(config: dict) -> dict[str, object]:
    registry = config["measurement_semantics_registry"]
    semantics_rows = measurement_semantics_rows(config)
    observation_rows = sample_observation_rows(config)
    world_state_rows = world_state_ready_rows(config)
    observation_semantics_uids = {str(row["semantics_uid"]) for row in observation_rows}
    world_state_semantics_uids = {str(row["semantics_uid"]) for row in world_state_rows}
    semantics_uids = {str(row["semantics_uid"]) for row in semantics_rows}
    return {
        "audit_uid": "measurement_semantics_registry_audit",
        "catalog_path": str(resolve_repo_path(registry["local_root"]) / registry["relative_paths"]["measurement_semantics_catalog"]),
        "semantics_row_count": len(semantics_rows),
        "sample_observation_row_count": len(observation_rows),
        "world_state_row_count": len(world_state_rows),
        "unique_semantics_uid_count": len(semantics_uids),
        "sample_semantics_coverage_count": len(observation_semantics_uids),
        "world_state_semantics_coverage_count": len(world_state_semantics_uids),
        "coverage_status": "complete" if semantics_uids.issuperset(observation_semantics_uids | world_state_semantics_uids) else "partial",
        "leakage_guard": str(registry["scaler_artifact_policy"]["leakage_guard"]),
        "inverse_transform_required": bool(registry["scaler_artifact_policy"]["inverse_transform_required"]),
        "status": "contract_satisfied" if semantics_rows and observation_rows and world_state_rows else "incomplete",
    }


def scaler_leakage_audit(config: dict) -> dict[str, object]:
    registry = config["model_input_registry"]
    bundle_root = resolve_repo_path(registry["local_root"])
    bundle_rows = read_tsv(bundle_root / registry["relative_paths"]["study_bundle_catalog"])
    bundle_filenames = registry["bundle_filenames"]
    all_scalers_exist = True
    all_inverse_transform_artifacts_declared = True
    all_leakage_guards_match_policy = True
    leakage_guard = str(config["measurement_semantics_registry"]["scaler_artifact_policy"]["leakage_guard"])
    bundle_summaries = []
    for row in bundle_rows:
        scaler_artifact_path = str(row.get("scaler_artifact_path", "")).strip()
        if not scaler_artifact_path:
            scaler_artifact_path = str(Path(row["bundle_root"]) / bundle_filenames["feature_scaler"])
        scaler_path = resolve_repo_path(scaler_artifact_path)
        scaler_exists = scaler_path.exists()
        all_scalers_exist &= scaler_exists
        inverse_transform_declared = False
        leakage_guard_matches_policy = False
        if scaler_exists:
            scaler_rows = read_tsv(scaler_path)
            inverse_transform_declared = bool(scaler_rows) and all(str(scaler_row.get("inverse_transform", "")).strip() for scaler_row in scaler_rows)
            leakage_guard_matches_policy = bool(scaler_rows) and all(str(scaler_row.get("leakage_guard", "")).strip() == leakage_guard for scaler_row in scaler_rows)
        all_inverse_transform_artifacts_declared &= inverse_transform_declared
        all_leakage_guards_match_policy &= leakage_guard_matches_policy
        bundle_summaries.append(
            {
                "study_uid": row["study_uid"],
                "semantics_uid": row.get("semantics_uid", ""),
                "scaler_fit_policy": row.get("scaler_fit_policy", "legacy_missing_scaler_fit_policy"),
                "scaler_exists": scaler_exists,
                "inverse_transform_declared": inverse_transform_declared,
                "leakage_guard_matches_policy": leakage_guard_matches_policy,
                "scaler_artifact_path": scaler_artifact_path,
                "bundle_root": row["bundle_root"],
            }
        )
    return {
        "audit_uid": "model_input_scaler_leakage_audit",
        "bundle_count": len(bundle_rows),
        "all_scaler_artifacts_exist": all_scalers_exist,
        "all_inverse_transform_artifacts_declared": all_inverse_transform_artifacts_declared,
        "all_leakage_guards_match_policy": all_leakage_guards_match_policy,
        "declared_fold_fit_policy": str(config["measurement_semantics_registry"]["scaler_artifact_policy"]["fold_fit_policy"]),
        "declared_production_fit_policy": str(config["measurement_semantics_registry"]["scaler_artifact_policy"]["production_fit_policy"]),
        "leakage_guard": leakage_guard,
        "bundle_summaries": bundle_summaries,
        "status": "bundle_scaler_artifacts_materialized" if all_scalers_exist and all_inverse_transform_artifacts_declared and all_leakage_guards_match_policy else "scaler_contract_incomplete",
    }


def dynamics_calibration_audit(config: dict) -> dict[str, object]:
    simulation_rows = simulation_teacher_rows(config)
    teacher_frame = pd.DataFrame(read_tsv(resolve_repo_path(config["data"]["full_microenvironment_teacher_ready_registry"]["local_root"]) / config["data"]["full_microenvironment_teacher_ready_registry"]["relative_paths"]["full_microenvironment_teacher_catalog"]))
    simulation_teacher_frame = pd.DataFrame(simulation_rows)
    schema_status = "current_schema"
    if "teacher_evidence_level" not in teacher_frame.columns:
        teacher_frame["teacher_evidence_level"] = "configuration_generated_provisional"
        schema_status = "legacy_schema_requires_rebuild"
    if "calibration_status" not in teacher_frame.columns:
        teacher_frame["calibration_status"] = "calibration_pending"
        schema_status = "legacy_schema_requires_rebuild"
    if "main_pretraining_eligibility" not in teacher_frame.columns:
        teacher_frame["main_pretraining_eligibility"] = "calibration_or_compatibility_only"
        schema_status = "legacy_schema_requires_rebuild"
    mainline_rows = teacher_frame.loc[teacher_frame["teacher_family"].astype(str) == "ad_full_microenvironment_main_dynamics"].copy()
    calibration_pending_count = int((teacher_frame["calibration_status"].astype(str) == "calibration_pending").sum()) if not teacher_frame.empty else 0
    mainline_calibration_statuses = sorted(set(mainline_rows["calibration_status"].astype(str).tolist())) if not mainline_rows.empty else []
    external_calibration_claim_status = "not_claimed" if calibration_pending_count else "external_calibration_validated"
    return {
        "audit_uid": "dynamics_calibration_audit",
        "simulation_teacher_row_count": int(simulation_teacher_frame.shape[0]),
        "full_microenvironment_teacher_row_count": int(teacher_frame.shape[0]),
        "mainline_teacher_row_count": int(mainline_rows.shape[0]),
        "calibration_pending_count": calibration_pending_count,
        "mean_teacher_loss_weight": float(pd.to_numeric(teacher_frame["teacher_loss_weight"], errors="coerce").fillna(0.0).mean()) if not teacher_frame.empty else 0.0,
        "teacher_evidence_levels": sorted(set(teacher_frame["teacher_evidence_level"].astype(str).tolist())) if not teacher_frame.empty else [],
        "calibration_statuses": sorted(set(teacher_frame["calibration_status"].astype(str).tolist())) if not teacher_frame.empty else [],
        "mainline_teacher_evidence_levels": sorted(set(mainline_rows["teacher_evidence_level"].astype(str).tolist())) if not mainline_rows.empty else [],
        "mainline_calibration_statuses": mainline_calibration_statuses,
        "main_pretraining_eligibility_values": sorted(set(teacher_frame["main_pretraining_eligibility"].astype(str).tolist())) if not teacher_frame.empty else [],
        "teacher_loss_weight_field_status": "present" if "teacher_loss_weight" in teacher_frame.columns else "missing",
        "external_calibration_claim_status": external_calibration_claim_status,
        "model_validity_scope": "mechanistic_teacher_pretraining_only" if calibration_pending_count else "external_calibration_validated",
        "schema_status": schema_status,
        "status": "weak_teacher_only" if calibration_pending_count else "requires_external_calibration_validation",
        "note": "Full microenvironment main dynamics remain provisional until external calibration thresholds are passed.",
    }


def crossscale_alignment_audit(config: dict) -> dict[str, object]:
    anchor_rows = crossmodal_anchor_rows(config)
    world_state_rows = world_state_ready_rows(config)
    paired_anchor_count = sum(1 for row in anchor_rows if str(row["anchor_type"]) == "paired_sample_cell")
    weak_anchor_count = sum(1 for row in anchor_rows if str(row["anchor_type"]) != "paired_sample_cell")
    paired_anchor_sources = sorted({str(row["anchor_source"]) for row in anchor_rows if str(row["anchor_type"]) == "paired_sample_cell"})
    weak_anchor_sources = sorted({str(row["anchor_source"]) for row in anchor_rows if str(row["anchor_type"]) != "paired_sample_cell"})
    return {
        "audit_uid": "crossscale_alignment_audit",
        "anchor_count": len(anchor_rows),
        "paired_anchor_count": paired_anchor_count,
        "weak_anchor_count": weak_anchor_count,
        "paired_anchor_sources": paired_anchor_sources,
        "weak_anchor_sources": weak_anchor_sources,
        "world_state_row_count": len(world_state_rows),
        "sample_observation_row_count": len(sample_observation_rows(config)),
        "alignment_status": "partial_registry_alignment" if weak_anchor_count and paired_anchor_count else "incomplete_alignment_registry",
        "paired_anchor_policy": "supervised_contrastive_retrieval" if paired_anchor_count else "none",
        "weak_anchor_policy": "distributional_alignment_only" if weak_anchor_count else "none",
        "full_crossscale_alignment_claim_status": "not_claimed" if weak_anchor_count else "paired_anchor_sources_only",
        "alignment_validity_scope": "paired_sample_sources_plus_distributional_weak_sources" if weak_anchor_count else "paired_sample_sources_only",
        "status": "partial_crossscale_registry_alignment" if anchor_rows and world_state_rows else "incomplete",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    report_dir = resolve_repo_path(config["paper"]["paths"]["report_dir"])
    ensure_directory(report_dir)
    payloads = {
        "measurement_semantics_audit.json": measurement_semantics_audit(config),
        "scaler_leakage_audit.json": scaler_leakage_audit(config),
        "dynamics_calibration_audit.json": dynamics_calibration_audit(config),
        "crossscale_alignment_audit.json": crossscale_alignment_audit(config),
    }
    for filename, payload in payloads.items():
        write_json(report_dir / filename, payload)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=report_dir,
        audit_file_count=len(payloads),
    )


if __name__ == "__main__":
    main()
