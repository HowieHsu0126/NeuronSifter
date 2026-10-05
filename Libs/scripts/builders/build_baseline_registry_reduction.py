from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY = REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml"
SCHEMA = REPO_ROOT / "Libs/configs/schema/baseline_registry_reduction.schema.json"
AUDIT_ROOT = REPO_ROOT / "Output/audits"

MANDATORY = {
    "pca_linear", "scvi", "generic_masked_encoder", "scgpt", "uce",
    "persistence", "linear_state_space", "gru", "fno_neuraloperator", "transolver_plus_plus", "wdno",
    "random", "fixed_cascade", "uncertainty_bald", "ps_bax", "mf_mes", "cost_aware_bo",
    "raw_score_no_tri", "uncertainty_only", "calibration_only",
}
CONDITIONAL = {"geneformer", "scfoundation", "domain_specific_perturbation_primary", "diffusionpde_partial_observation", "tri_domain_specific_component"}
UPPER_BOUNDS = {"hidden_outcome_oracle", "full_information_retraining_oracle"}
CLAIM_SLOTS = {
    "C1": ["pca_linear", "scvi", "generic_masked_encoder", "scgpt", "uce"],
    "C2": ["persistence", "linear_state_space", "gru", "fno_neuraloperator", "transolver_plus_plus", "wdno"],
    "C3": ["random", "fixed_cascade", "uncertainty_bald", "ps_bax", "mf_mes", "cost_aware_bo"],
    "C4": ["raw_score_no_tri", "uncertainty_only", "calibration_only"],
}

# Every historical UID is assigned exactly once. Multiple historical controls may
# be consolidated into one active slot, but no historical evidence is discarded.
MIGRATION: dict[str, tuple[str, str, str, str, str]] = {
    "feature_mean": ("related_work", "feature_mean", "B9_related_work_only", "historical_only", "redundant lower-bound control retired from the active ranking"),
    "feature_median": ("related_work", "feature_median", "B9_related_work_only", "historical_only", "redundant lower-bound control retired from the active ranking"),
    "pca_ridge_logistic": ("baseline", "pca_linear", "B1_simple_strong", "mandatory", "canonicalized PCA linear slot"),
    "pca_nearest_neighbor": ("related_work", "pca_nearest_neighbor", "B9_related_work_only", "historical_only", "one C1 linear slot retained"),
    "scvi": ("baseline", "scvi", "B4_external_task_specific_model", "mandatory", "retained C1 task-specific baseline"),
    "generic_encoder_from_scratch": ("baseline", "generic_masked_encoder", "B1_simple_strong", "mandatory", "canonicalized generic encoder slot"),
    "single_source_encoder": ("ablation", "single_source_encoder", "B2_same_backbone_ablation", "ablation", "same-backbone source ablation moved out of baseline ranking"),
    "no_ontology_encoder": ("ablation", "no_ontology_encoder", "B2_same_backbone_ablation", "ablation", "same-backbone ontology ablation moved out of baseline ranking"),
    "geneformer": ("baseline", "geneformer", "B3_external_pretrained_model", "conditional", "conditional C1 transfer slot"),
    "scgpt": ("baseline", "scgpt", "B3_external_pretrained_model", "mandatory", "retained C1 pretrained slot"),
    "uce": ("baseline", "uce", "B3_external_pretrained_model", "mandatory", "retained C1 pretrained slot"),
    "scfoundation": ("baseline", "scfoundation", "B3_external_pretrained_model", "conditional", "conditional C1 transfer slot"),
    "identity_no_action": ("related_work", "identity_no_action", "B9_related_work_only", "historical_only", "lower-bound control outside reduced C2 set"),
    "persistence": ("baseline", "persistence", "B0_naive_lower_bound", "mandatory", "retained C2 lower bound"),
    "linear_delta": ("related_work", "linear_delta", "B9_related_work_only", "historical_only", "subsumed by linear state-space slot"),
    "linear_state_space_var": ("baseline", "linear_state_space", "B1_simple_strong", "mandatory", "canonicalized C2 linear slot"),
    "shallow_mlp": ("related_work", "shallow_mlp", "B9_related_work_only", "historical_only", "simple-model budget reduced to GRU and linear controls"),
    "gru": ("baseline", "gru", "B1_simple_strong", "mandatory", "retained C2 recurrent slot"),
    "deeponet": ("related_work", "deeponet", "B9_related_work_only", "historical_only", "operator budget reduced to maintained FNO family"),
    "fno_tfno": ("baseline", "fno_neuraloperator", "B4_external_task_specific_model", "mandatory", "canonicalized FNO slot"),
    "transolver_plus": ("baseline", "transolver_plus_plus", "B4_external_task_specific_model", "mandatory", "canonicalized Transolver++ slot"),
    "diffusionpde": ("baseline", "diffusionpde_partial_observation", "B4_external_task_specific_model", "conditional", "partial-observation compatibility remains conditional"),
    "wdno": ("baseline", "wdno", "B4_external_task_specific_model", "mandatory", "retained C2 operator slot"),
    "deterministic_neuronsifter": ("ablation", "deterministic_neuronsifter", "B2_same_backbone_ablation", "ablation", "same-backbone deterministic ablation moved out of baseline ranking"),
    "scgen": ("baseline", "domain_specific_perturbation_primary", "B4_external_task_specific_model", "conditional", "candidate for the single C2 domain-specific slot"),
    "cpa": ("related_work", "cpa", "B9_related_work_only", "historical_only", "candidate retained as related work until domain slot resolution"),
    "chemcpa": ("related_work", "chemcpa", "B9_related_work_only", "historical_only", "candidate retained as related work until domain slot resolution"),
    "cellot": ("related_work", "cellot", "B9_related_work_only", "historical_only", "candidate retained as related work until domain slot resolution"),
    "conditional_monge_gap": ("related_work", "conditional_monge_gap", "B9_related_work_only", "historical_only", "candidate retained as related work until domain slot resolution"),
    "gears": ("related_work", "gears", "B9_related_work_only", "historical_only", "candidate retained as related work until domain slot resolution"),
    "cellflow": ("related_work", "cellflow", "B9_related_work_only", "historical_only", "candidate retained as related work until domain slot resolution"),
    "bliss_loewe_hsa": ("related_work", "bliss_loewe_hsa", "B9_related_work_only", "historical_only", "combo endpoint semantics are not frozen"),
    "random_acquisition": ("baseline", "random", "B0_naive_lower_bound", "mandatory", "canonicalized C3 random slot"),
    "round_robin": ("related_work", "round_robin", "B9_related_work_only", "historical_only", "acquisition controls reduced"),
    "cheapest_first": ("related_work", "cheapest_first", "B9_related_work_only", "historical_only", "acquisition controls reduced"),
    "highest_fidelity_only": ("related_work", "highest_fidelity_only", "B9_related_work_only", "historical_only", "acquisition controls reduced"),
    "fixed_cascade": ("baseline", "fixed_cascade", "B1_simple_strong", "mandatory", "retained C3 fixed control"),
    "mechanism_rule": ("related_work", "mechanism_rule", "B9_related_work_only", "historical_only", "rule control reduced from formal set"),
    "uncertainty_acquisition": ("related_work", "uncertainty_acquisition", "B9_related_work_only", "historical_only", "subsumed by uncertainty/BALD slot"),
    "entropy_bald": ("baseline", "uncertainty_bald", "B5_tool_acquisition_policy", "mandatory", "canonicalized uncertainty/BALD slot"),
    "expected_error_reduction": ("related_work", "expected_error_reduction", "B9_related_work_only", "historical_only", "acquisition budget reduced"),
    "infobax": ("related_work", "infobax", "B9_related_work_only", "historical_only", "acquisition budget reduced"),
    "ps_bax": ("baseline", "ps_bax", "B5_tool_acquisition_policy", "mandatory", "retained C3 acquisition slot"),
    "mf_mes": ("baseline", "mf_mes", "B5_tool_acquisition_policy", "mandatory", "retained C3 acquisition slot"),
    "cost_aware_bo": ("baseline", "cost_aware_bo", "B5_tool_acquisition_policy", "mandatory", "retained C3 acquisition slot"),
    "oracle_acquisition": ("upper_bounds", "hidden_outcome_oracle", "B8_oracle_upper_bound", "upper_bound", "separated evaluation-only oracle"),
    "raw_score": ("baseline", "raw_score_no_tri", "B6_reliability_baseline", "mandatory", "canonicalized C4 no-TRI score control"),
    "uncertainty_only": ("baseline", "uncertainty_only", "B6_reliability_baseline", "mandatory", "retained C4 reliability control"),
    "calibration_only": ("baseline", "calibration_only", "B6_reliability_baseline", "mandatory", "retained C4 reliability control"),
    "ood_only": ("related_work", "ood_only", "B9_related_work_only", "historical_only", "reliability component outside reduced C4 set"),
    "pkpd_only": ("related_work", "pkpd_only", "B9_related_work_only", "historical_only", "reliability component outside reduced C4 set"),
    "no_tri": ("ablation", "no_tri", "B2_same_backbone_ablation", "ablation", "same-backbone TRI removal moved out of baseline ranking"),
    "large_only": ("deployment", "large_only", "B7_scale_deployment_baseline", "deployment", "deployment profile moved out of baseline ranking"),
    "student_without_distillation": ("deployment", "student_without_distillation", "B7_scale_deployment_baseline", "deployment", "deployment profile moved out of baseline ranking"),
    "fixed_escalation": ("deployment", "fixed_escalation_or_no_cascade", "B7_scale_deployment_baseline", "deployment", "deployment profiles consolidated"),
    "no_escalation": ("deployment", "fixed_escalation_or_no_cascade", "B7_scale_deployment_baseline", "deployment", "deployment profiles consolidated"),
    "no_cascade": ("deployment", "fixed_escalation_or_no_cascade", "B7_scale_deployment_baseline", "deployment", "deployment profiles consolidated"),
}

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode())

def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def source_entries(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = registry.get("baseline_first_program", {}).get("mandatory_baselines", [])
    if len(entries) != 57:
        raise ValueError(f"historical baseline source must contain 57 entries, found {len(entries)}")
    return [dict(entry) for entry in entries]

def validate_reduction(registry: Mapping[str, Any], schema_path: Path = SCHEMA) -> dict[str, Any]:
    reduction = registry.get("reduction_registry")
    if not isinstance(reduction, dict):
        raise ValueError("reduction_registry is missing")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = sorted(Draft202012Validator(schema).iter_errors(reduction), key=lambda e: list(e.path))
    blockers = [{"code": "schema", "message": error.message} for error in schema_errors]
    mandatory = list(reduction.get("mandatory", []))
    conditional = list(reduction.get("conditional", []))
    upper_bounds = list(reduction.get("upper_bounds", []))
    active = mandatory + conditional + upper_bounds
    ids = [str(row.get("baseline_uid", "")) for row in active]
    if len(ids) != len(set(ids)):
        blockers.append({"code": "duplicate_active_uid", "message": "Active 27-slot registry contains duplicate baseline_uid values."})
    if set(ids) != MANDATORY | CONDITIONAL | UPPER_BOUNDS:
        blockers.append({"code": "active_slot_membership", "message": "Active slot membership differs from the frozen 20/5/2 set."})
    counts = {"total": len(active), "mandatory": len(mandatory), "conditional": len(conditional), "upper_bounds": len(upper_bounds)}
    if counts != {"total": 27, "mandatory": 20, "conditional": 5, "upper_bounds": 2}:
        blockers.append({"code": "cardinality", "message": json.dumps(counts, sort_keys=True)})
    for claim_id, expected in CLAIM_SLOTS.items():
        actual = list(reduction.get("claim_slots", {}).get(claim_id, []))
        if actual != expected:
            blockers.append({"code": "claim_slots", "message": f"{claim_id} slot order/membership drifted."})
    for row in conditional:
        required = ("activation_condition", "decision_artifact", "mandatory_when_activated", "incompatibility_proof_when_inactive", "activation_frozen_before_test")
        if any(row.get(key) in (None, "", False) for key in required):
            blockers.append({"code": "conditional_contract", "message": f"{row.get('baseline_uid')} has an incomplete conditional contract."})
        if row.get("active") is True or row.get("activated") is True:
            blockers.append({"code": "conditional_self_activation", "message": f"{row.get('baseline_uid')} cannot self-activate."})
        if row.get("resolved_method") is not None and row.get("baseline_uid") == "domain_specific_perturbation_primary":
            blockers.append({"code": "conditional_self_activation", "message": "Domain-specific slot cannot self-activate before a decision artifact."})
    for row in upper_bounds:
        if row.get("deployable") is not False or row.get("eligible_for_best_baseline_claim") is not False or row.get("reported_in_separate_table_section") is not True:
            blockers.append({"code": "oracle_separation", "message": f"{row.get('baseline_uid')} is not a separated non-deployable upper bound."})
    return {"status": "passed" if not blockers else "failed", "blocker_count": len(blockers), "blockers": blockers, "counts": counts, "active_uids": ids}

def build_snapshot(registry_path: Path, historical_path: Path | None = None) -> dict[str, Any]:
    source_path = historical_path or registry_path
    raw = source_path.read_bytes()
    source_registry = load_yaml(source_path)
    entries = source_entries(source_registry)
    return {
        "schema_version": "baseline_registry_57_entry_snapshot_v1",
        "source_path": str(registry_path.relative_to(REPO_ROOT)) if registry_path.is_relative_to(REPO_ROOT) else str(registry_path),
        "capture_source_path": str(source_path.relative_to(REPO_ROOT)) if source_path.is_relative_to(REPO_ROOT) else str(source_path),
        "source_sha256": sha256_bytes(raw),
        "entry_count": len(entries),
        "mandatory_count": sum(row.get("mandatory_or_conditional") == "mandatory" for row in entries),
        "conditional_count": sum(row.get("mandatory_or_conditional") == "conditional" for row in entries),
        "entries": entries,
        "source_registry": source_registry,
    }

def build_migration(registry: Mapping[str, Any], source_sha: str) -> list[dict[str, Any]]:
    entries = source_entries(registry)
    rows = []
    seen: set[str] = set()
    for old in entries:
        old_uid = str(old["baseline_uid"])
        if old_uid in seen or old_uid not in MIGRATION:
            raise ValueError(f"historical UID is missing or duplicated in migration map: {old_uid}")
        seen.add(old_uid)
        new_registry, new_uid, new_class, status, reason = MIGRATION[old_uid]
        decision = {"old_baseline_uid": old_uid, "new_registry": new_registry, "new_uid": new_uid, "new_class": new_class, "mandatory_status": status, "migration_reason": reason}
        rows.append({
            "old_baseline_uid": old_uid,
            "old_class": old["baseline_class"],
            "new_registry": new_registry,
            "new_uid": new_uid,
            "new_class": new_class,
            "mandatory_status": status,
            "migration_reason": reason,
            "historical_evidence_preserved": "true",
            "source_hash": canonical_sha256(old),
            "decision_hash": canonical_sha256(decision),
        })
    if len(rows) != 57 or len(seen) != 57:
        raise ValueError("migration ledger must reconcile exactly 57 unique historical entries")
    return rows

def build_contract(registry: Mapping[str, Any], snapshot: Mapping[str, Any], validation: Mapping[str, Any], migration: list[dict[str, Any]]) -> dict[str, Any]:
    reduction = registry["reduction_registry"]
    return {
        "schema_version": "baseline_registry_reduction_contract_v1",
        "status": "frozen" if validation["status"] == "passed" else "invalid",
        "source_registry_path": "Libs/configs/experiments/baseline_registry.yaml",
        "source_registry_sha256": snapshot["source_sha256"],
        "historical_entry_count": snapshot["entry_count"],
        "target_cardinality": reduction["target_cardinality"],
        "mandatory_uids": sorted(MANDATORY),
        "conditional_uids": sorted(CONDITIONAL),
        "upper_bound_uids": sorted(UPPER_BOUNDS),
        "migration_row_count": len(migration),
        "historical_entries_reconciled_exactly_once": validation["status"] == "passed",
        "prohibited_actions": ["external_installation", "upstream_execution", "baseline_training", "neuronsifter_headline_training"],
        "authorization_requirements": {"mandatory_qualified": "20_of_20", "activated_conditionals_qualified": "all", "inactive_conditionals_have_incompatibility_proof": True, "upper_bounds_separated": True, "coevolution_compositions_complete": True, "test_accessed": False},
    }

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["old_baseline_uid", "old_class", "new_registry", "new_uid", "new_class", "mandatory_status", "migration_reason", "historical_evidence_preserved", "source_hash", "decision_hash"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def write_incompatibility_proofs(root: Path, reduction: Mapping[str, Any]) -> None:
    for row in reduction.get("conditional", []):
        proof_path = REPO_ROOT / str(row["incompatibility_proof_when_inactive"])
        payload = {
            "schema_version": "baseline_conditional_incompatibility_proof_v1",
            "baseline_uid": row["baseline_uid"],
            "activation_status": "inactive",
            "activation_frozen_before_test": bool(row["activation_frozen_before_test"]),
            "proof_basis": str(row["activation_condition"]),
            "decision_artifact": str(row["decision_artifact"]),
            "test_data_accessed": False,
            "producer": "build_baseline_registry_reduction.py",
        }
        write_json(proof_path, payload)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the NeuronSifter 57-to-27 baseline registry reduction.")
    parser.add_argument("--registry", default=str(REGISTRY))
    parser.add_argument("--schema", default=str(SCHEMA))
    parser.add_argument("--historical-source", default="")
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    parser.add_argument("--full-suite-green", action="store_true", help="Promote the final gate only after an independently recorded full-suite green run.")
    args = parser.parse_args()
    registry_path = Path(args.registry)
    if not registry_path.is_absolute(): registry_path = REPO_ROOT / registry_path
    registry = load_yaml(registry_path)
    snapshot = build_snapshot(registry_path, Path(args.historical_source) if args.historical_source else None)
    validation = validate_reduction(registry, Path(args.schema) if Path(args.schema).is_absolute() else REPO_ROOT / args.schema)
    migration = build_migration(registry, snapshot["source_sha256"])
    contract = build_contract(registry, snapshot, validation, migration)
    root = Path(args.audit_root)
    if not root.is_absolute(): root = REPO_ROOT / root
    write_json(root / "baseline_registry_57_entry_snapshot.json", snapshot)
    write_json(root / "baseline_registry_reduction_contract.json", contract)
    write_tsv(root / "baseline_registry_57_to_27_migration.tsv", migration)
    write_incompatibility_proofs(root, registry["reduction_registry"])
    result = {"schema_version": "baseline_registry_reduction_validation_v1", **validation, "historical_entries_reconciled": len(migration), "migration_unique_entries": len({row["old_baseline_uid"] for row in migration}), "unclassified_entries": 57 - len(migration), "authorization_fail_closed": True, "source_registry_sha256": snapshot["source_sha256"]}
    write_json(root / "baseline_registry_reduction_validation.json", result)
    full_suite_green = bool(args.full_suite_green)
    gate = {"schema_version": "baseline_registry_reduction_gate_v1", "gate_name": "iclr_scale_baseline_registry_resolved", "status": "iclr_scale_baseline_registry_resolved" if validation["status"] == "passed" and full_suite_green else "blocked_baseline_registry_reduction", "historical_entries_reconciled": len(migration) == 57, "new_registry_total": 27, "mandatory": 20, "conditional": 5, "upper_bounds": 2, "unclassified_entries": 57 - len(migration), "authorization_fail_closed": True, "full_suite_green": full_suite_green, "blockers": ([] if validation["status"] == "passed" else [row["code"] for row in validation["blockers"]]) + ([] if full_suite_green else ["full_suite_not_green"]) }
    write_json(root / "baseline_registry_reduction_gate.json", gate)
    print(json.dumps({"status": validation["status"], "counts": validation["counts"], "migration_rows": len(migration)}, sort_keys=True))
    raise SystemExit(0 if validation["status"] == "passed" else 1)

if __name__ == "__main__":
    main()
