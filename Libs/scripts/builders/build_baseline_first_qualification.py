from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json, write_tsv  # noqa: E402

from Libs.scripts.builders.build_baseline_registry_reduction import (  # noqa: E402
    CONDITIONAL as REDUCED_CONDITIONAL,
    MANDATORY as REDUCED_MANDATORY,
    UPPER_BOUNDS as REDUCED_UPPER_BOUNDS,
    MIGRATION as REDUCED_MIGRATION,
    validate_reduction,
)


DEFAULT_REGISTRY = "Libs/configs/experiments/baseline_registry.yaml"
DEFAULT_SCHEMA = "Libs/configs/schema/baseline_registry.schema.json"
DEFAULT_AUDIT_ROOT = "Output/audits"
DEFAULT_REPORT_ROOT = "Output/paper_reports"

REQUIRED_UIDS = {
    "C1": {
        "feature_mean", "feature_median", "pca_ridge_logistic", "pca_nearest_neighbor", "scvi",
        "generic_encoder_from_scratch", "single_source_encoder", "no_ontology_encoder", "geneformer",
        "scgpt", "uce", "scfoundation",
    },
    "C2": {
        "identity_no_action", "persistence", "linear_delta", "linear_state_space_var", "shallow_mlp", "gru",
        "deeponet", "fno_tfno", "transolver_plus", "diffusionpde", "wdno", "deterministic_neuronsifter",
        "scgen", "cpa", "chemcpa", "cellot", "conditional_monge_gap", "gears", "cellflow", "bliss_loewe_hsa",
    },
    "C3": {
        "random_acquisition", "round_robin", "cheapest_first", "highest_fidelity_only", "fixed_cascade",
        "mechanism_rule", "uncertainty_acquisition", "entropy_bald", "expected_error_reduction", "infobax",
        "ps_bax", "mf_mes", "cost_aware_bo", "oracle_acquisition",
    },
    "C4": {"raw_score", "uncertainty_only", "calibration_only", "ood_only", "pkpd_only", "no_tri"},
    "C5": {"large_only", "student_without_distillation", "fixed_escalation", "no_escalation", "no_cascade"},
}

EXPECTED_CLASSES = {
    "feature_mean": "B0_naive_lower_bound",
    "feature_median": "B0_naive_lower_bound",
    "oracle_acquisition": "B8_oracle_upper_bound",
    "deterministic_neuronsifter": "B2_same_backbone_ablation",
    "no_tri": "B2_same_backbone_ablation",
    "student_without_distillation": "B2_same_backbone_ablation",
}

EXPECTED_CLASS_BY_UID = {
    **{uid: "B0_naive_lower_bound" for uid in {"feature_mean", "feature_median", "identity_no_action", "persistence", "random_acquisition", "round_robin", "cheapest_first", "highest_fidelity_only"}},
    **{uid: "B1_simple_strong" for uid in {"pca_ridge_logistic", "pca_nearest_neighbor", "generic_encoder_from_scratch", "linear_delta", "linear_state_space_var", "shallow_mlp", "gru", "bliss_loewe_hsa", "fixed_cascade", "mechanism_rule"}},
    **{uid: "B2_same_backbone_ablation" for uid in {"single_source_encoder", "no_ontology_encoder", "deterministic_neuronsifter", "no_tri", "student_without_distillation"}},
    **{uid: "B3_external_pretrained_model" for uid in {"geneformer", "scgpt", "uce", "scfoundation"}},
    **{uid: "B4_external_task_specific_model" for uid in {"scvi", "deeponet", "fno_tfno", "transolver_plus", "diffusionpde", "wdno", "scgen", "cpa", "chemcpa", "cellot", "conditional_monge_gap", "gears", "cellflow"}},
    **{uid: "B5_tool_acquisition_policy" for uid in {"uncertainty_acquisition", "entropy_bald", "expected_error_reduction", "infobax", "ps_bax", "mf_mes", "cost_aware_bo"}},
    **{uid: "B6_reliability_baseline" for uid in {"raw_score", "uncertainty_only", "calibration_only", "ood_only", "pkpd_only"}},
    **{uid: "B7_scale_deployment_baseline" for uid in {"large_only", "fixed_escalation", "no_escalation", "no_cascade"}},
    "oracle_acquisition": "B8_oracle_upper_bound",
}
CONDITIONAL_UIDS = {"scgen", "cpa", "chemcpa", "cellot", "conditional_monge_gap", "gears", "cellflow", "bliss_loewe_hsa", "oracle_acquisition"}
V31_MAIN_BASELINES = ("persistence_or_no_action", "linear_delta_or_state_space", "gru", "chemcpa", "cellot", "fno", "transolver_plus", "wdno", "deep_ensemble", "conformal_calibration")
V31_MAIN_ABLATIONS = ("neuronsifter_deterministic", "neuronsifter_drug_id_conditioning", "neuronsifter_scalar_exposure", "neuronsifter_without_constraints_and_reliability", "neuronsifter_without_assimilation_and_what_if_pairing")

EXTERNAL_LEGACY_PATHS = {
    "diffusionpde": "Libs/external/diffusionpde",
    "wdno": "Libs/external/neural_operator/wdno",
    "transolver_plus": "Libs/external/neural_operator/Transolver_plus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fail-closed NeuronSifter Baseline-First qualification artifacts.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--audit-root", default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_registry(registry_path: Path, schema_path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(registry))
    if errors:
        detail = "; ".join(error.message for error in errors)
        raise ValueError(f"Baseline registry schema validation failed: {detail}")
    return registry


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _evidence_binding_state(binding: Mapping[str, Any]) -> tuple[Path, dict[str, Any] | None, bool]:
    path = repo_path(str(binding["path"]))
    payload = _read_json(path)
    provenance_valid = bool(
        payload is not None
        and all(payload.get(field) not in (None, "", [], {}) for field in binding["required_provenance_fields"])
    )
    valid = bool(
        payload is not None
        and path.is_file()
        and sha256_file(path) == binding["expected_sha256"]
        and payload.get("schema_version") == binding["expected_schema_version"]
        and payload.get("status") == binding["expected_status"]
        and provenance_valid
    )
    return path, payload, valid


def _task_gate(claim_id: str, evidence_binding: Mapping[str, Any], artifact_bindings: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    path, evidence, evidence_binding_valid = _evidence_binding_state(evidence_binding)
    evidence = _read_json(path)
    status = evidence.get("status") if evidence else "missing_or_invalid_evidence"
    reason_by_claim = {
        "C1": "C1 full-objective task and holdout coverage remain insufficient.",
        "C2": "C2 has no action-complete, independently held-out trajectory domain.",
        "C3": "No frozen tool-outcome matrix is bound to an acquisition evaluation protocol.",
        "C4": "No frozen validation/test prediction bundle exists for reliability comparison.",
        "C5": "No frozen comparable Large and Student checkpoint bundle exists.",
    }
    evidence = evidence or {}
    hash_keys = {
        "C1": ("corpus_identity_sha256", "split_manifest_sha256", "leakage_audit_sha256"),
        "C2": ("trajectory_registry_sha256", "split_manifest_sha256", "leakage_audit_sha256"),
        "C3": ("tool_outcome_matrix_sha256", "split_manifest_sha256"),
        "C4": ("prediction_bundle_sha256", "split_manifest_sha256"),
        "C5": ("checkpoint_bundle_sha256", "split_manifest_sha256"),
    }
    bound_artifacts: dict[str, dict[str, Any]] = {}
    hashes_valid = True
    for key in hash_keys[claim_id]:
        binding = artifact_bindings.get(key, {})
        artifact_path = repo_path(str(binding.get("path", ""))) if binding.get("path") else Path("")
        expected_hash = str(binding.get("sha256", ""))
        declared_hash = evidence.get(key)
        observed_hash = sha256_file(artifact_path) if artifact_path.is_file() else ""
        valid = (
            _sha256_digest(declared_hash)
            and _sha256_digest(expected_hash)
            and artifact_path.is_file()
            and declared_hash == expected_hash == observed_hash
        )
        hashes_valid = hashes_valid and valid
        bound_artifacts[key] = {
            "path": str(binding.get("path", "")),
            "expected_sha256": expected_hash,
            "declared_sha256": declared_hash or "",
            "observed_sha256": observed_hash,
            "valid": valid,
        }
    eligibility_rules = {
        "C1": status == "c1_baseline_data_ready" and evidence.get("full_objective_task_family_count", 0) >= 2 and evidence.get("full_objective_holdout_family_count", 0) >= 2 and evidence.get("frozen_splits") is True and evidence.get("leakage_audit_passed") is True,
        "C2": status == "c2_baseline_data_ready" and evidence.get("action_complete") is True and evidence.get("independent_trajectory_domain") is True and evidence.get("strict_holdouts_available") is True and evidence.get("frozen_splits") is True and evidence.get("leakage_audit_passed") is True,
        "C3": status == "c3_tool_outcome_matrix_frozen" and evidence.get("frozen_tool_outcome_matrix") is True and evidence.get("costs_frozen") is True and evidence.get("outcomes_observed") is True,
        "C4": status == "c4_prediction_bundle_frozen" and evidence.get("frozen_validation_test_predictions") is True and evidence.get("paired_raw_units_available") is True and evidence.get("test_labels_unseen_during_selection") is True,
        "C5": status == "c5_checkpoint_bundle_frozen" and evidence.get("frozen_comparable_checkpoint_bundle") is True and evidence.get("same_deployment_domain") is True and evidence.get("test_labels_unseen_during_selection") is True,
    }
    eligible = evidence_binding_valid and hashes_valid and eligibility_rules[claim_id]
    return {
        "claim_id": claim_id,
        "evidence_path": evidence_binding["path"],
        "evidence_exists": path.exists(),
        "evidence_sha256": sha256_file(path) if path.exists() else "",
        "evidence_binding_valid": evidence_binding_valid,
        "observed_status": status,
        "formal_run_eligible": eligible,
        "bound_artifacts": bound_artifacts,
        "reason": "eligible" if eligible else reason_by_claim[claim_id],
    }


def _external(entry: Mapping[str, Any]) -> bool:
    return entry["official_or_authoritative"] not in {"project_owned", "related_work_only"}


def _command(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return completed.returncode, completed.stdout.strip()


def _git_revision(path: Path, revision: str) -> str:
    code, output = _command("git", "-C", str(path), "rev-parse", revision)
    return output if code == 0 else ""


def _verified_registry_artifact(path_text: str, expected_sha256: str) -> bool:
    path = repo_path(path_text)
    return bool(path.is_file() and _sha256_digest(expected_sha256) and sha256_file(path) == expected_sha256)


def _term_evidence_state(entry: Mapping[str, Any], field: str) -> tuple[str, dict[str, Any]]:
    declared_status = str(entry[field])
    path = str(entry[f"{field}_evidence_path"])
    expected_sha256 = str(entry[f"{field}_evidence_sha256"])
    provenance = str(entry[f"{field}_evidence_provenance"])
    if not _external(entry):
        return "not_applicable", {"path": path, "expected_sha256": expected_sha256, "provenance": provenance, "verified": True}
    if field == "weight_terms" and declared_status == "not_applicable" and str(entry["checkpoint_identity"]).startswith("from_scratch"):
        return "not_applicable", {"path": path, "expected_sha256": expected_sha256, "provenance": provenance, "verified": True}
    verified = (
        declared_status == "qualified"
        and provenance not in {"", "not_available", "pending", "unverified"}
        and _verified_registry_artifact(path, expected_sha256)
    )
    return "qualified" if verified else "not_resolved", {
        "path": path,
        "expected_sha256": expected_sha256,
        "provenance": provenance,
        "verified": verified,
    }


def _source_state(entry: Mapping[str, Any]) -> dict[str, Any]:
    uid = str(entry["baseline_uid"])
    managed_path = REPO_ROOT / "Libs" / "external" / uid
    legacy_path_text = EXTERNAL_LEGACY_PATHS.get(uid, "")
    legacy_path = REPO_ROOT / legacy_path_text if legacy_path_text else None
    exact_pin = str(entry["tag_or_commit"])
    managed_git = managed_path.is_dir() and (managed_path / ".git").exists()
    resolved_pin = _git_revision(managed_path, exact_pin) if managed_git and exact_pin and "pending" not in exact_pin else ""
    head_commit = _git_revision(managed_path, "HEAD") if managed_git else ""
    pin_verified = bool(resolved_pin and head_commit and resolved_pin == head_commit)
    checkpoint_identity = str(entry["checkpoint_identity"])
    checkpoint_resolved = checkpoint_identity in {"not_applicable", "not_required_from_scratch"} or checkpoint_identity.startswith("sha256:")
    terms = f"{entry['license']} {entry['redistribution_policy']}"
    license_resolved = not any(token in terms.lower() for token in ("pending", "unknown", "unverified", "not_qualified"))
    term_evidence = {
        field: _term_evidence_state(entry, field)
        for field in ("data_terms", "weight_terms", "package_terms", "anonymous_package_permission")
    }
    term_status = {field: value[0] for field, value in term_evidence.items()}
    term_evidence_records = {field: value[1] for field, value in term_evidence.items()}
    terms_resolved = all(status in {"qualified", "not_applicable"} for status in term_status.values())
    environment_definition_path = str(entry["environment_lock_path"])
    environment_hash = str(entry["environment_lock_sha256"])
    environment_lock_resolved = _verified_registry_artifact(environment_definition_path, environment_hash)
    patch_diff_path = str(entry["patch_diff_path"])
    patch_diff_sha256 = str(entry["patch_diff_sha256"])
    patch_integrity_resolved = _verified_registry_artifact(patch_diff_path, patch_diff_sha256)
    source_locked = bool(entry["repository_or_package"]) and managed_git and pin_verified and checkpoint_resolved and license_resolved and terms_resolved and environment_lock_resolved and patch_integrity_resolved
    return {
        "baseline_uid": uid,
        "external": _external(entry),
        "mandatory_or_conditional": entry["mandatory_or_conditional"],
        "managed_path": f"Libs/external/{uid}",
        "managed_path_exists": managed_path.exists(),
        "managed_git": managed_git,
        "head_commit": head_commit,
        "resolved_pin": resolved_pin,
        "pin_verified": pin_verified,
        "legacy_discovered_path": legacy_path_text,
        "legacy_path_exists": bool(legacy_path and legacy_path.exists()),
        "repository_or_package": entry["repository_or_package"],
        "tag_or_commit": exact_pin,
        "source_locked": source_locked,
        "license": entry["license"],
        "license_resolved": license_resolved,
        "term_status": term_status,
        "term_evidence": term_evidence_records,
        "terms_resolved": terms_resolved,
        "environment_lock_resolved": environment_lock_resolved,
        "patch_integrity_resolved": patch_integrity_resolved,
        "environment_definition_path": environment_definition_path,
        "environment_hash": environment_hash or "not_computed",
        "patch_diff_path": patch_diff_path,
        "patch_diff": patch_diff_path if patch_integrity_resolved else "not_applicable_without_qualified_diff",
        "patch_diff_sha256": patch_diff_sha256,
        "adapter_qualification_sha256": entry["adapter_qualification_sha256"],
        "upstream_qualification_sha256": entry["upstream_qualification_sha256"],
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_resolved": checkpoint_resolved,
        "qualification_status": "source_locked" if _external(entry) and source_locked else "not_acquired_or_unqualified",
    }


def _source_artifacts(entries: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for entry in entries:
        state = _source_state(entry)
        if not state["external"]:
            continue
        uid = state["baseline_uid"]
        base = f"Output/audits/baselines/{uid}"
        artifacts[uid] = {
            "source_manifest": {
                "schema_version": "baseline_source_manifest_v1",
                "baseline_uid": uid,
                "qualification_status": state["qualification_status"],
                "managed_path": state["managed_path"],
                "managed_path_exists": state["managed_path_exists"],
                "managed_git": state["managed_git"],
                "legacy_discovered_path": state["legacy_discovered_path"],
                "repository_or_package": state["repository_or_package"],
                "tag_or_commit": state["tag_or_commit"],
                "repository_hash": state["head_commit"] or "not_available_without_managed_checkout",
                "resolved_pin": state["resolved_pin"],
                "pin_verified": state["pin_verified"],
                "source_locked": state["source_locked"],
                "data_terms_status": state["term_status"]["data_terms"],
                "weight_terms_status": state["term_status"]["weight_terms"],
                "package_terms_status": state["term_status"]["package_terms"],
                "anonymous_package_permission_status": state["term_status"]["anonymous_package_permission"],
                "submodule_hashes": "not_available_without_managed_checkout",
                "download_date": "not_acquired_or_unrecorded",
                "official_checkpoint_hash": state["checkpoint_identity"] if state["checkpoint_identity"].startswith("sha256:") else "not_available",
            },
            "license_audit": {
                "schema_version": "baseline_license_audit_v1",
                "baseline_uid": uid,
                "license": state["license"],
                "redistribution_policy": entry["redistribution_policy"],
                "data_terms": {"status": state["term_status"]["data_terms"], "declared_status": entry["data_terms"], **state["term_evidence"]["data_terms"]},
                "weight_terms": {"status": state["term_status"]["weight_terms"], "declared_status": entry["weight_terms"], **state["term_evidence"]["weight_terms"]},
                "package_terms": {"status": state["term_status"]["package_terms"], "declared_status": entry["package_terms"], **state["term_evidence"]["package_terms"]},
                "anonymous_package_permission": {"status": state["term_status"]["anonymous_package_permission"], "declared_status": entry["anonymous_package_permission"], **state["term_evidence"]["anonymous_package_permission"]},
                "resolved": state["license_resolved"],
                "status": "qualified" if state["source_locked"] else "not_qualified",
            },
            "environment_lock": {
                "schema_version": "baseline_environment_lock_v1",
                "baseline_uid": uid,
                "environment": entry["environment"],
                "environment_definition_path": state["environment_definition_path"],
                "environment_hash": state["environment_hash"],
                "status": "locked" if state["environment_lock_resolved"] else "not_built",
            },
            "patch_manifest": {
                "schema_version": "baseline_patch_manifest_v1",
                "baseline_uid": uid,
                "core_algorithm_modified": False,
                "patch_diff_path": state["patch_diff_path"],
                "patch_diff": state["patch_diff"],
                "patch_diff_sha256": state["patch_diff_sha256"],
                "status": "verified" if state["patch_integrity_resolved"] else "not_acquired",
            },
            "paths": {
                "source_manifest": f"{base}/source_manifest.json",
                "license_audit": f"{base}/license_audit.json",
                "environment_lock": f"{base}/environment_lock.json",
                "patch_manifest": f"{base}/patch_manifest.json",
            },
        }
    return artifacts


def _validate_program(program: Mapping[str, Any], registry: Mapping[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    entries = list(program["mandatory_baselines"])
    ids = [str(entry["baseline_uid"]) for entry in entries]
    if len(ids) != len(set(ids)):
        blockers.append({"code": "duplicate_baseline_uid", "message": "baseline_uid values must be unique."})
    counts = Counter(entry["mandatory_or_conditional"] for entry in entries)
    cardinality = program["frozen_cardinality"]
    if len(entries) != cardinality["baseline_count"]:
        blockers.append({"code": "baseline_count_mismatch", "message": "Baseline entry count differs from the frozen cardinality."})
    if counts["mandatory"] != cardinality["mandatory_count"] or counts["conditional"] != cardinality["conditional_count"]:
        blockers.append({"code": "mandatory_conditional_count_mismatch", "message": "Mandatory and conditional baseline counts differ from the freeze."})
    class_counts = Counter(entry["baseline_class"] for entry in entries)
    if dict(class_counts) != dict(cardinality["class_counts"]):
        blockers.append({"code": "baseline_class_count_mismatch", "message": "Baseline class counts differ from the freeze."})
    by_id = {str(entry["baseline_uid"]): entry for entry in entries}
    for claim_id, required_ids in REQUIRED_UIDS.items():
        missing = sorted(required_ids - set(ids))
        unexpected = sorted(uid for uid, entry in by_id.items() if claim_id in entry["claim_ids"] and uid not in required_ids)
        if missing:
            blockers.append({"code": "missing_mandatory_baseline", "message": f"{claim_id}: {','.join(missing)}"})
        if unexpected:
            blockers.append({"code": "unexpected_baseline_for_claim", "message": f"{claim_id}: {','.join(unexpected)}"})
    if set(EXPECTED_CLASS_BY_UID) != set(ids):
        blockers.append({"code": "baseline_class_mapping_coverage_mismatch", "message": "The frozen class mapping must cover exactly the baseline registry."})
    for uid, expected_class in EXPECTED_CLASS_BY_UID.items():
        entry = by_id.get(uid)
        if entry and entry["baseline_class"] != expected_class:
            blockers.append({"code": "baseline_class_mismatch", "message": f"{uid} must be {expected_class}."})
    conditional = {uid for uid, entry in by_id.items() if entry["mandatory_or_conditional"] == "conditional"}
    if conditional != CONDITIONAL_UIDS:
        blockers.append({"code": "conditional_baseline_membership_mismatch", "message": "Conditional baseline membership differs from the freeze."})
    profiles = registry["active_set_profiles"]
    profile_ids = [row["active_set_id"] for row in profiles]
    expected_profiles = list(V31_MAIN_BASELINES + V31_MAIN_ABLATIONS)
    if profile_ids != expected_profiles:
        blockers.append({"code": "v31_active_profile_mismatch", "message": "V3.1 profile order or membership drifted."})
    if _canonical_sha256(profiles) != program["v31_binding"]["profile_mapping_sha256"]:
        blockers.append({"code": "v31_profile_mapping_hash_mismatch", "message": "The complete normalized V3.1 profile mapping drifted."})
    active_contract = registry["active_set_contract"]
    binding = program["v31_binding"]
    if tuple(active_contract["main_baseline_ids"]) != V31_MAIN_BASELINES or tuple(binding["main_baseline_ids"]) != V31_MAIN_BASELINES:
        blockers.append({"code": "v31_main_baseline_binding_mismatch", "message": "The frozen V3.1 main baseline binding drifted."})
    if tuple(active_contract["main_ablation_ids"]) != V31_MAIN_ABLATIONS or tuple(binding["main_ablation_ids"]) != V31_MAIN_ABLATIONS:
        blockers.append({"code": "v31_main_ablation_binding_mismatch", "message": "The frozen V3.1 main ablation binding drifted."})
    if any(entry["baseline_class"] == "B8_oracle_upper_bound" and entry["mandatory_or_conditional"] == "mandatory" for entry in entries):
        blockers.append({"code": "oracle_ranking_violation", "message": "Oracle baselines must remain conditional and separate from deployable ranking."})
    return blockers


def _validated_historical_baseline_qualification(path: Path, expected_hash: str) -> tuple[dict[str, Any] | None, bool, str]:
    payload = _read_json(path)
    if payload is None:
        return None, False, "missing_or_invalid_historical_baseline_qualification_gate"
    # These fields bind the preserved historical document. They are not a
    # current baseline authorization state and must retain its frozen bytes.
    expected = {
        "schema_version": "baseline_r1_gate",
        "status": "blocked_source_or_license",
        "round11_closure": "blocked_full_suite",
        "round11_ceiling": "upstream_smoke_wave1_complete",
        "active_set_version": "baseline_registry_v3.1",
        "external_source_required_count": 7,
        "internal_no_external_source_count": 8,
        "historical_record_count": 36,
        "main_baseline_count": 10,
        "main_ablation_count": 5,
        "source_license_passed_external_count": 0,
        "clean_clone_performed_count": 0,
        "upstream_smoke_run_count": 0,
        "native_reproduction_count": 0,
        "matched_training_count": 0,
        "headline_ready_count": 0,
        "DT_CORE_allowed": False,
        "evidence_fabric_sealing_allowed": False,
        "baseline_native_reproduction_allowed": False,
        "baseline_adapter_training_smoke_allowed": False,
        "matched_protocol_training_allowed": False,
        "blockers": [
            "round11_closure_blocked_full_suite",
            "no_external_active_baseline_has_complete_source_license_weight_data_package_terms",
            "no_clean_r1_clone_performed",  # historical qualification artifact field
            "no_upstream_smoke_run",
        ],
    }
    invalid = [key for key, value in expected.items() if payload.get(key) != value]
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        invalid.append("sha256")
    return (
        payload,
        not invalid,
        "historical_baseline_qualification_gate_valid"
        if not invalid
        else f"historical_baseline_qualification_gate_mismatch:{','.join(invalid)}",
    )


def _validated_current_baseline_qualification(path: Path, expected_hash: str) -> tuple[dict[str, Any] | None, bool, str]:
    payload = _read_json(path)
    if payload is None:
        return None, False, "missing_or_invalid_current_baseline_qualification_gate"
    expected = {
        "schema_version": "baseline_registry_qualification_gate",
        "active_set_version": "baseline_registry.1",
        "historical_record_count": 36,
        "main_baseline_count": 10,
        "main_ablation_count": 5,
        "internal_no_external_source_count": 8,
        "external_source_required_count": 7,
        "source_license_passed_external_count": 0,
        "clean_clone_performed_count": 0,
        "upstream_smoke_run_count": 0,
        "native_reproduction_count": 0,
        "matched_training_count": 0,
        "headline_ready_count": 0,
        "round11_closure": "blocked_full_suite",
        "round11_ceiling": "upstream_smoke_wave1_complete",
        "baseline_native_reproduction_allowed": False,
        "baseline_adapter_training_smoke_allowed": False,
        "matched_protocol_training_allowed": False,
        "evidence_fabric_sealing_allowed": False,
        "DT_CORE_allowed": False,
        "status": "blocked_source_or_license",
        "blockers": [
            "round11_closure_blocked_full_suite",
            "no_external_active_baseline_has_complete_source_license_weight_data_package_terms",
            "no_clean_external_baseline_clone_performed",
            "no_upstream_smoke_run",
        ],
    }
    invalid = [key for key, value in expected.items() if payload.get(key) != value]
    if sha256_file(path) != expected_hash:
        invalid.append("sha256")
    return (
        payload,
        not invalid,
        "current_baseline_qualification_gate_valid"
        if not invalid
        else f"current_baseline_qualification_gate_mismatch:{','.join(invalid)}",
    )


def _evidence_status(binding: Mapping[str, Any]) -> bool:
    _, _, valid = _evidence_binding_state(binding)
    return valid


def _test_access_unseen(binding: Mapping[str, Any]) -> bool:
    _, payload, valid = _evidence_binding_state(binding)
    return bool(
        valid
        and payload
        and payload.get("status") == "sealed"
        and payload.get("test_data_accessed") is False
        and payload.get("formal_test_result_count") == 0
        and _sha256_digest(payload.get("identity_sha256"))
    )


def _baseline_evidence_status(uid: str, kind: str, expected_sha256: str, expected_status: str) -> bool:
    path = REPO_ROOT / "Output" / "audits" / "baselines" / uid / f"{kind}.json"
    payload = _read_json(path)
    return bool(
        payload
        and path.is_file()
        and sha256_file(path) == expected_sha256
        and payload.get("schema_version") == f"baseline_{kind}_v1"
        and payload.get("status") == expected_status
        and payload.get("baseline_uid") == uid
        and payload.get("producer")
        and payload.get("created_at")
        and payload.get("artifact_manifest")
    )


def _upstream_rows(external_states: list[Mapping[str, Any]], qualified_by_uid: Mapping[str, bool]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in external_states:
        uid = str(state["baseline_uid"])
        qualified = bool(qualified_by_uid.get(uid, False))
        rows.append({
            "baseline_uid": uid,
            "mandatory_or_conditional": state["mandatory_or_conditional"],
            "upstream_status": "upstream_reproduced_within_declared_tolerance" if qualified else "not_run",
            "reported_reference_result": "qualified_evidence_bundle" if qualified else "not_collected_before_source_lock",
            "reproduced_result": "qualified_evidence_bundle" if qualified else "not_run",
            "metric_definition": "frozen" if qualified else "not_frozen",
            "dataset_version": "frozen" if qualified else "not_frozen",
            "split": "frozen" if qualified else "not_frozen",
            "seed": "qualified" if qualified else "not_run",
            "hardware": "qualified" if qualified else "not_run",
            "runtime": "qualified" if qualified else "not_run",
            "deviation": "within_declared_tolerance" if qualified else "not_available",
        })
    return rows


def authorization_state(
    gate_flags: Mapping[str, bool],
    contract_blockers: list[dict[str, str]],
    source_locked: bool,
    licenses_resolved: bool,
    c1_c2_ready: bool,
    adapters_complete: bool,
    current_baseline_qualification_valid: bool,
    historical_baseline_qualification_valid: bool,
) -> tuple[str, bool]:
    # `test_accessed` is a negative safety predicate: the resolved state
    # requires it to remain false, unlike the positive readiness flags.
    resolved = all(value for name, value in gate_flags.items() if name != "test_accessed") and gate_flags.get("test_accessed", False) is False
    if resolved:
        return "baseline_challenge_suite_resolved", True
    if (
        contract_blockers
        or not current_baseline_qualification_valid
        or not historical_baseline_qualification_valid
    ):
        return "blocked_baseline_registry", False
    if not source_locked:
        return "blocked_baseline_source", False
    if not licenses_resolved:
        return "blocked_baseline_license", False
    if not c1_c2_ready:
        return "blocked_data_readiness", False
    if not adapters_complete:
        return "blocked_baseline_adapter", False
    return "blocked_strong_baseline_execution", False


def build_all(registry_path: str | Path = DEFAULT_REGISTRY, schema_path: str | Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    registry_file = repo_path(registry_path)
    schema_file = repo_path(schema_path)
    registry = _load_registry(registry_file, schema_file)
    program = registry["baseline_first_program"]
    entries = list(program["mandatory_baselines"])
    contract_blockers = _validate_program(program, registry)
    source_states = [_source_state(entry) for entry in entries]
    task_gates = [
        _task_gate(claim_id, program["data_gate_evidence"][claim_id.lower()], program["data_gate_artifact_bindings"][claim_id])
        for claim_id in program["claim_ids"]
    ]
    task_by_claim = {row["claim_id"]: row for row in task_gates}
    external_states = [row for row in source_states if row["external"]]
    mandatory_external_states = [row for row in external_states if row["mandatory_or_conditional"] == "mandatory"]
    source_locked = all(row["source_locked"] for row in mandatory_external_states)
    licenses_resolved = all(
        row["license_resolved"]
        and row["terms_resolved"]
        and row["environment_lock_resolved"]
        and row["patch_integrity_resolved"]
        for row in mandatory_external_states
    )
    adapters_complete = all(
        not row["external"] or (
            (REPO_ROOT / "Libs" / "exp" / "baselines" / "adapters" / f"{row['baseline_uid']}.py").exists()
            and _baseline_evidence_status(
                row["baseline_uid"], "adapter_qualification", row["adapter_qualification_sha256"], "adapter_qualified"
            )
        )
        for row in source_states if row["mandatory_or_conditional"] == "mandatory"
    )
    data_and_splits_frozen = all(row["formal_run_eligible"] for row in task_gates)
    c1_c2_ready = task_by_claim["C1"]["formal_run_eligible"] and task_by_claim["C2"]["formal_run_eligible"]
    evidence_paths = program["qualification_evidence"]
    qualification_evidence_bundle = {
        name: {
            "path": binding["path"],
            "expected_sha256": binding["expected_sha256"],
            "observed_sha256": sha256_file(repo_path(binding["path"])) if repo_path(binding["path"]).is_file() else "",
            "binding_valid": _evidence_binding_state(binding)[2],
        }
        for name, binding in evidence_paths.items()
    }
    qualification_evidence_bundle["task_gates"] = {
        "sha256": _canonical_sha256(task_gates),
        "claim_ids": list(program["claim_ids"]),
    }
    qualification_evidence_bundle_sha256 = _canonical_sha256(qualification_evidence_bundle)
    current_baseline_qualification_binding = evidence_paths["current_baseline_qualification_gate"]
    current_baseline_qualification_path = repo_path(current_baseline_qualification_binding["path"])
    current_baseline_qualification, current_baseline_qualification_valid, current_baseline_qualification_reason = _validated_current_baseline_qualification(
        current_baseline_qualification_path,
        current_baseline_qualification_binding["expected_sha256"],
    )
    historical_baseline_qualification_binding = evidence_paths["historical_baseline_qualification_gate"]
    historical_baseline_qualification_path = repo_path(historical_baseline_qualification_binding["path"])
    historical_baseline_qualification, historical_baseline_qualification_valid, historical_baseline_qualification_reason = _validated_historical_baseline_qualification(
        historical_baseline_qualification_path,
        historical_baseline_qualification_binding["expected_sha256"],
    )
    upstream_qualification_by_uid = {
        row["baseline_uid"]: _baseline_evidence_status(
            row["baseline_uid"], "upstream_qualification", row["upstream_qualification_sha256"], "upstream_reproduced_within_declared_tolerance"
        )
        for row in external_states
    }
    gate_flags = {
        "baseline_registry_frozen": not contract_blockers,
        "mandatory_baseline_sources_locked": source_locked,
        "mandatory_licenses_resolved": licenses_resolved,
        "upstream_qualification_complete": all(upstream_qualification_by_uid[row["baseline_uid"]] for row in mandatory_external_states),
        "task_compatible_adapters_complete": adapters_complete,
        "fairness_contract_frozen": True,
        "data_and_splits_frozen": data_and_splits_frozen,
        "simple_baseline_sanity_passed": _evidence_status(evidence_paths["simple_baseline_sanity"]),
        "strong_baseline_results_available": _evidence_status(evidence_paths["strong_baseline_results"]),
        "challenge_thresholds_frozen": _evidence_status(evidence_paths["challenge_thresholds"]),
        "test_data_unseen": _test_access_unseen(evidence_paths["test_access_ledger"]),
        "current_baseline_qualification_bound": current_baseline_qualification_valid,
        "historical_baseline_qualification_lineage_bound": historical_baseline_qualification_valid,
    }
    reduction_validation = validate_reduction(registry)
    reduction = registry.get("reduction_registry", {})
    reduced_ids = {
        str(row["baseline_uid"])
        for section in ("mandatory", "conditional", "upper_bounds")
        for row in reduction.get(section, [])
    }
    composition_path = REPO_ROOT / "Libs/configs/experiments/coevolution_comparison_registry.yaml"
    composition_complete = False
    if composition_path.is_file():
        try:
            import yaml as _yaml
            composition = _yaml.safe_load(composition_path.read_text(encoding="utf-8"))
            composition_complete = {
                str(row.get("comparison_uid")) for row in composition.get("entries", [])
            } == {
                "static_twin_static_agent", "static_twin_adaptive_agent", "adaptive_twin_static_agent",
                "full_agent_twin_coevolution", "naive_latest_batch_joint_update", "full_replay_joint_update",
            }
        except Exception:
            composition_complete = False
    old_entries_by_uid = {str(entry["baseline_uid"]): entry for entry in entries}
    reduced_mandatory_qualified = reduction_validation["status"] == "passed" and len(reduction.get("mandatory", [])) == 20
    for reduced_row in reduction.get("mandatory", []):
        reduced_uid = str(reduced_row["baseline_uid"])
        source_uids = [
            old_uid for old_uid, decision in REDUCED_MIGRATION.items()
            if decision[0] == "baseline" and decision[1] == reduced_uid and decision[3] == "mandatory"
        ]
        if not source_uids:
            reduced_mandatory_qualified = False
            continue
        source_qualification = []
        for source_uid in source_uids:
            source_state = _source_state(old_entries_by_uid[source_uid])
            base = REPO_ROOT / "Output/audits/baselines" / reduced_uid
            source_manifest = _read_json(base / "source_manifest.json")
            adapter = REPO_ROOT / "Libs/exp/baselines/adapters" / f"{reduced_uid}.py"
            upstream = _read_json(base / "upstream_qualification.json")
            adapter_evidence = _read_json(base / "adapter_qualification.json")
            source_qualification.append(bool(
                not source_state["external"]
                or (
                    source_state["source_locked"]
                    and source_state["license_resolved"]
                    and source_state["environment_lock_resolved"]
                    and source_state["patch_integrity_resolved"]
                    and source_manifest and source_manifest.get("status") in {"qualified", "source_locked"}
                    and upstream and upstream.get("status") in {"qualified", "upstream_reproduced_within_declared_tolerance"}
                    and adapter.is_file()
                    and adapter_evidence and adapter_evidence.get("status") == "adapter_qualified"
                )
            ))
        reduced_mandatory_qualified = reduced_mandatory_qualified and any(source_qualification)
    # Conditional slots remain inactive until their decision artifacts are frozen.
    inactive_conditional_proofs = all(
        (REPO_ROOT / str(row.get("incompatibility_proof_when_inactive", ""))).is_file()
        for row in reduction.get("conditional", [])
    )
    conditional_qualification = all(
        bool(row.get("activation_frozen_before_test"))
        and (
            (REPO_ROOT / str(row.get("decision_artifact", ""))).is_file()
            or (REPO_ROOT / str(row.get("incompatibility_proof_when_inactive", ""))).is_file()
        )
        for row in reduction.get("conditional", [])
    )
    gate_flags.update({
        "reduced_registry_frozen": reduction_validation["status"] == "passed",
        "mandatory_qualified": reduced_mandatory_qualified,
        "mandatory_qualified_20_of_20": reduced_mandatory_qualified,
        "activated_conditionals_qualified": conditional_qualification,
        "inactive_conditionals_have_incompatibility_proof": inactive_conditional_proofs,
        "upper_bounds_separated": set(str(row.get("baseline_uid")) for row in reduction.get("upper_bounds", [])) == REDUCED_UPPER_BOUNDS,
        "coevolution_compositions_complete": composition_complete,
        "test_accessed": not gate_flags["test_data_unseen"],
    })
    authorization_status, headline_training_authorized = authorization_state(
        gate_flags,
        contract_blockers,
        source_locked,
        licenses_resolved,
        c1_c2_ready,
        adapters_complete,
        current_baseline_qualification_valid,
        historical_baseline_qualification_valid,
    )
    source_artifacts = _source_artifacts(entries)
    registry_hash = sha256_file(registry_file)
    freeze = {
        "schema_version": "baseline_registry_freeze_v1",
        "freeze_name": program["freeze_name"],
        "registry_path": str(registry_file.relative_to(REPO_ROOT)) if registry_file.is_relative_to(REPO_ROOT) else str(registry_file),
        "registry_sha256": registry_hash,
        "baseline_count": len(entries),
        "claim_counts": {claim_id: sum(claim_id in entry["claim_ids"] for entry in entries) for claim_id in program["claim_ids"]},
        "baseline_classes": list(program["baseline_classes"]),
        "mandatory_baseline_uids": [entry["baseline_uid"] for entry in entries if entry["mandatory_or_conditional"] == "mandatory"],
        "conditional_baseline_uids": [entry["baseline_uid"] for entry in entries if entry["mandatory_or_conditional"] == "conditional"],
        "ranking_separation": {
            "oracle_class": "B8_oracle_upper_bound",
            "oracle_must_not_rank_with": ["B0_naive_lower_bound", "B1_simple_strong", "B2_same_backbone_ablation", "B3_external_pretrained_model", "B4_external_task_specific_model", "B5_tool_acquisition_policy", "B6_reliability_baseline", "B7_scale_deployment_baseline"],
            "same_rank_column_forbidden": True,
        },
        "contract_blockers": contract_blockers,
        "status": "frozen" if not contract_blockers else "invalid",
    }
    fairness = {
        "schema_version": "baseline_fairness_contract_v1",
        "status": "frozen_contract_only",
        "shared_inputs": program["fairness_contract"]["shared_inputs"],
        "same_backbone_controls": program["fairness_contract"]["same_backbone_controls"],
        "failure_accounting": program["fairness_contract"]["failure_accounting"],
        "pretrained_reporting": program["fairness_contract"]["pretrained_reporting"],
        "tuning_contract": program["tuning_contract"],
        "empirical_fairness_result_available": False,
    }
    thresholds = {
        "schema_version": "neuronsifter_challenge_thresholds_v1",
        "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256,
        "status": "frozen" if gate_flags["challenge_thresholds_frozen"] else "not_frozen_no_strong_baseline_results",
        "freeze_before_neuronsifter_test": program["challenge_threshold_contract"]["freeze_before_neuronsifter_test"],
        "primary_metric": "frozen_primary_metric" if gate_flags["challenge_thresholds_frozen"] else "not_frozen",
        "secondary_metrics": ["frozen_secondary_metrics"] if gate_flags["challenge_thresholds_frozen"] else [],
        "strongest_compatible_baseline_per_task": {},
        "best_simple_baseline_per_task": {},
        "lower_bound_per_task": {},
        "oracle_upper_bound_per_task": {},
        "minimum_meaningful_delta": "frozen" if gate_flags["challenge_thresholds_frozen"] else "not_frozen",
        "compute_budget": "frozen" if gate_flags["challenge_thresholds_frozen"] else "not_frozen",
        "coverage_floor": program["challenge_threshold_contract"]["coverage_floor"],
        "blockers": [] if gate_flags["challenge_thresholds_frozen"] else ["strong_baseline_results_unavailable", "simple_baseline_sanity_not_passed"],
    }
    authorization = {
        "schema_version": "neuronsifter_headline_training_authorization_v1",
        "gate_name": program["training_authorization_contract"]["gate_name"],
        "status": authorization_status,
        "baseline_challenge_suite_resolved": headline_training_authorized,
        "headline_training_authorized": headline_training_authorized,
        "authorization_conditions": gate_flags,
        "qualification_evidence_bundle": qualification_evidence_bundle,
        "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256,
        "task_gates": task_gates,
        "current_baseline_qualification_gate": current_baseline_qualification,
        "current_baseline_qualification_gate_sha256": (
            sha256_file(current_baseline_qualification_path)
            if current_baseline_qualification_valid
            else ""
        ),
        "current_baseline_qualification_status": current_baseline_qualification_reason,
        "historical_baseline_qualification_gate": historical_baseline_qualification,
        "historical_baseline_qualification_gate_sha256": (
            sha256_file(historical_baseline_qualification_path)
            if historical_baseline_qualification_valid
            else ""
        ),
        "historical_baseline_qualification_lineage_status": historical_baseline_qualification_reason,
        "allowed_pre_gate_work": program["training_authorization_contract"]["allowed_pre_gate_work"],
        "blockers": [name for name, passed in gate_flags.items() if not passed],
        "next_action": "proceed_to_neuronsifter_headline_training" if headline_training_authorized else "recover_action_complete_independent_C2_trajectory_domain_and_complete_external_source_license_qualification",
    }
    task_compatibility = {
        "schema_version": "baseline_task_compatibility_v1",
        "status": "formal_data_gates_passed" if data_and_splits_frozen else "blocked_pending_formal_data_gates",
        "task_gates": task_gates,
        "baselines": [
            {
                "baseline_uid": entry["baseline_uid"],
                "claim_ids": entry["claim_ids"],
                "task_compatibility": entry["task_compatibility"],
                "required_modalities": entry["required_modalities"],
                "required_action_fields": entry["required_action_fields"],
                "required_trajectory_semantics": entry["required_trajectory_semantics"],
                "formal_task_run_allowed": all(task_by_claim[claim_id]["formal_run_eligible"] for claim_id in entry["claim_ids"]),
            }
            for entry in entries
        ],
    }
    upstream = {
        "schema_version": "baseline_upstream_reproduction_summary_v1",
        "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256,
        "qualification_scope": "mandatory_external_baselines",
        "status": "complete" if gate_flags["upstream_qualification_complete"] else "not_started_source_and_data_gated",
        "rows": _upstream_rows(external_states, upstream_qualification_by_uid),
    }
    return {
        "freeze": freeze,
        "fairness": fairness,
        "information_access": [
            {
                "baseline_uid": entry["baseline_uid"],
                "claim_ids": ",".join(entry["claim_ids"]),
                "task_compatibility": entry["task_compatibility"],
                "required_modalities": ",".join(entry["required_modalities"]),
                "required_action_fields": ",".join(entry["required_action_fields"]),
                "required_trajectory_semantics": ",".join(entry["required_trajectory_semantics"]),
                "input_adapter": entry["input_adapter"],
                "output_adapter": entry["output_adapter"],
            }
            for entry in entries
        ],
        "compute_budget": [
            {
                "baseline_uid": entry["baseline_uid"],
                "baseline_class": entry["baseline_class"],
                "tuning_space": entry["tuning_space"],
                "tuning_budget": entry["tuning_budget"],
                "seed_policy": entry["seed_policy"],
                "compute_profile": entry["compute_profile"],
                "pretraining_provenance_required": str(entry["baseline_class"] == "B3_external_pretrained_model").lower(),
            }
            for entry in entries
        ],
        "task_compatibility": task_compatibility,
        "upstream": upstream,
        "source_artifacts": source_artifacts,
        "task_sanity": {
            "schema_version": "baseline_task_sanity_v1",
            "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256,
            "status": "passed" if gate_flags["simple_baseline_sanity_passed"] else "not_run_data_gate_blocked",
            "simple_baseline_sanity_passed": gate_flags["simple_baseline_sanity_passed"],
            "stop_or_adjust_conditions": [
                "constant_or_majority_reaches_primary_metric_ceiling", "linear_statistically_equivalent_to_complex_model",
                "source_only_exceeds_biology_only", "source_shortcut_detected", "strict_holdout_gap_unexplained", "metric_task_mismatch",
            ],
            "task_gates": task_gates,
        },
        "source_shortcut": [
            {"claim_id": row["claim_id"], "status": "passed" if gate_flags["simple_baseline_sanity_passed"] else "not_run_data_gate_blocked", "metric": "validated" if gate_flags["simple_baseline_sanity_passed"] else "not_available", "reason": row["reason"], "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256}
            for row in task_gates
        ],
        "floor_matrix": [
            {"claim_id": row["claim_id"], "baseline_stage": "B0_B1", "result_status": "available" if gate_flags["simple_baseline_sanity_passed"] else "not_run_data_gate_blocked", "primary_metric": "validated" if gate_flags["simple_baseline_sanity_passed"] else "not_available", "value": "evidence_bundle" if gate_flags["simple_baseline_sanity_passed"] else "", "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256}
            for row in task_gates
        ],
        "result_matrix": [
            {"claim_id": row["claim_id"], "baseline_stage": "B3_B4_B5", "result_status": "available" if gate_flags["strong_baseline_results_available"] else "not_run_data_gate_blocked", "primary_metric": "validated" if gate_flags["strong_baseline_results_available"] else "not_available", "value": "evidence_bundle" if gate_flags["strong_baseline_results_available"] else "", "paired_bootstrap_ci": "available" if gate_flags["strong_baseline_results_available"] else "not_available", "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256}
            for row in task_gates
        ],
        "resource_matrix": [
            {"claim_id": row["claim_id"], "result_status": "available" if gate_flags["strong_baseline_results_available"] else "not_run_data_gate_blocked", "runtime": "qualified" if gate_flags["strong_baseline_results_available"] else "not_available", "peak_memory": "qualified" if gate_flags["strong_baseline_results_available"] else "not_available", "compute_cost": "qualified" if gate_flags["strong_baseline_results_available"] else "not_available", "qualification_evidence_bundle_sha256": qualification_evidence_bundle_sha256}
            for row in task_gates
        ],
        "failure_catalog": [
            {"claim_id": row["claim_id"], "failure_type": "data_gate_blocked", "detail": row["reason"], "preserved": "true"}
            for row in task_gates if not row["formal_run_eligible"]
        ],
        "thresholds": thresholds,
        "authorization": authorization,
    }


def write_all(payload: Mapping[str, Any], audit_root: str | Path = DEFAULT_AUDIT_ROOT, report_root: str | Path = DEFAULT_REPORT_ROOT) -> None:
    audits = repo_path(audit_root)
    reports = repo_path(report_root)
    write_json(audits / "baseline_registry_freeze.json", payload["freeze"])
    write_json(audits / "baseline_fairness_contract.json", payload["fairness"])
    write_tsv(audits / "baseline_information_access_matrix.tsv", list(payload["information_access"][0]), payload["information_access"])
    write_tsv(audits / "baseline_compute_budget_matrix.tsv", list(payload["compute_budget"][0]), payload["compute_budget"])
    write_json(audits / "baseline_task_compatibility.json", payload["task_compatibility"])
    write_json(audits / "baseline_upstream_reproduction_summary.json", payload["upstream"])
    write_json(audits / "baseline_task_sanity.json", payload["task_sanity"])
    write_tsv(audits / "baseline_source_shortcut_diagnostics.tsv", list(payload["source_shortcut"][0]), payload["source_shortcut"])
    write_json(audits / "neuronsifter_challenge_thresholds.json", payload["thresholds"])
    write_json(audits / "neuronsifter_headline_training_authorization.json", payload["authorization"])
    write_tsv(reports / "baseline_floor_matrix.tsv", list(payload["floor_matrix"][0]), payload["floor_matrix"])
    write_tsv(reports / "baseline_result_matrix.tsv", list(payload["result_matrix"][0]), payload["result_matrix"])
    write_tsv(reports / "baseline_resource_matrix.tsv", list(payload["resource_matrix"][0]), payload["resource_matrix"])
    write_tsv(reports / "baseline_failure_catalog.tsv", list(payload["failure_catalog"][0]), payload["failure_catalog"])
    for uid, artifact in payload["source_artifacts"].items():
        base = audits / "baselines" / uid
        write_json(base / "source_manifest.json", artifact["source_manifest"])
        write_json(base / "license_audit.json", artifact["license_audit"])
        write_json(base / "environment_lock.json", artifact["environment_lock"])
        write_json(base / "patch_manifest.json", artifact["patch_manifest"])


def main() -> None:
    args = parse_args()
    payload = build_all(args.registry, args.schema)
    write_all(payload, args.audit_root, args.report_root)
    authorization = payload["authorization"]
    print(json.dumps({"status": authorization["status"], "headline_training_authorized": authorization["headline_training_authorized"]}, sort_keys=True))


if __name__ == "__main__":
    main()
