from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_registry_reduction import (  # noqa: E402
    AUDIT_ROOT,
    MANDATORY,
    CONDITIONAL,
    UPPER_BOUNDS,
    build_contract,
    build_migration,
    build_snapshot,
    load_yaml,
    validate_reduction,
)

REQUIRED_MIGRATION_FIELDS = ["old_baseline_uid", "old_class", "new_registry", "new_uid", "new_class", "mandatory_status", "migration_reason", "historical_evidence_preserved", "source_hash", "decision_hash"]

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

def validate(
    registry_path: str | Path = REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml",
    audit_root: str | Path = AUDIT_ROOT,
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    if not registry_file.is_absolute(): registry_file = REPO_ROOT / registry_file
    root = Path(audit_root)
    if not root.is_absolute(): root = REPO_ROOT / root
    blockers: list[dict[str, str]] = []
    try:
        registry = load_yaml(registry_file)
        reduction_validation = validate_reduction(registry)
    except Exception as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "registry_invalid", "message": str(exc)}]}
    blockers.extend(reduction_validation.get("blockers", []))
    reduction = registry["reduction_registry"]
    snapshot_path = root / "baseline_registry_57_entry_snapshot.json"
    migration_path = root / "baseline_registry_57_to_27_migration.tsv"
    contract_path = root / "baseline_registry_reduction_contract.json"
    validation_path = root / "baseline_registry_reduction_validation.json"
    gate_path = root / "baseline_registry_reduction_gate.json"
    for path in (snapshot_path, migration_path, contract_path, validation_path, gate_path):
        if not path.is_file(): blockers.append({"code": "missing_artifact", "message": str(path.relative_to(REPO_ROOT))})
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        expected_sha = str(reduction.get("historical_source_sha256", ""))
        if snapshot.get("schema_version") != "baseline_registry_57_entry_snapshot_v1" or snapshot.get("source_sha256") != expected_sha or snapshot.get("entry_count") != 57:
            blockers.append({"code": "snapshot_binding", "message": "The 57-entry snapshot is not bound to the frozen source SHA or count."})
        current_historical = registry["baseline_first_program"]["mandatory_baselines"]
        if snapshot.get("entries") != current_historical:
            blockers.append({"code": "historical_source_drift", "message": "Current historical entries differ from the frozen 57-entry snapshot."})
    else:
        snapshot = {}
    expected_migration = build_migration(registry, str(reduction.get("historical_source_sha256", "")))
    if migration_path.is_file():
        actual_migration = read_tsv(migration_path)
        if list(actual_migration[0]) != REQUIRED_MIGRATION_FIELDS if actual_migration else True:
            blockers.append({"code": "migration_schema", "message": "Migration ledger fields are incomplete or missing."})
        if actual_migration != [{key: str(value) for key, value in row.items()} for row in expected_migration]:
            blockers.append({"code": "migration_reconciliation", "message": "Migration ledger does not exactly match the 57-entry source mapping."})
        if len({row.get("old_baseline_uid") for row in actual_migration}) != 57 or len(actual_migration) != 57:
            blockers.append({"code": "migration_exactly_once", "message": "Every historical UID must occur exactly once."})
    if contract_path.is_file() and snapshot:
        actual_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_contract = build_contract(registry, snapshot, reduction_validation, expected_migration)
        if actual_contract != expected_contract:
            blockers.append({"code": "contract_stale_or_tampered", "message": "Reduction contract does not match the registry and migration ledger."})
    for section, expected_ids in (("mandatory", MANDATORY), ("conditional", CONDITIONAL), ("upper_bounds", UPPER_BOUNDS)):
        for row in reduction.get(section, []):
            if section == "mandatory" and row["baseline_uid"] in {"deterministic_neuronsifter", "no_tri", "student_without_distillation"}:
                blockers.append({"code": "ablation_in_baseline", "message": row["baseline_uid"]})
            if section == "upper_bounds" and row.get("deployable") is not False:
                blockers.append({"code": "oracle_deployable", "message": row["baseline_uid"]})
    # The six compositions are a separate comparison registry and must not be active methods.
    composition_path = REPO_ROOT / "Libs/configs/experiments/coevolution_comparison_registry.yaml"
    if composition_path.is_file():
        composition = yaml.safe_load(composition_path.read_text(encoding="utf-8"))
        composition_ids = {str(row.get("comparison_uid")) for row in composition.get("entries", [])}
        expected_compositions = {"static_twin_static_agent", "static_twin_adaptive_agent", "adaptive_twin_static_agent", "full_agent_twin_coevolution", "naive_latest_batch_joint_update", "full_replay_joint_update"}
        if composition_ids != expected_compositions or len(composition.get("entries", [])) != 6:
            blockers.append({"code": "coevolution_composition_count", "message": "Exactly six co-evolution compositions are required."})
    else:
        blockers.append({"code": "missing_coevolution_registry", "message": str(composition_path)})
    active_ids = set(MANDATORY) | set(CONDITIONAL) | set(UPPER_BOUNDS)
    split_ids: dict[str, set[str]] = {}
    for registry_name, expected_count in (("ablation_registry.yaml", 0), ("deployment_comparison_registry.yaml", 3), ("related_work_registry.yaml", 0)):
        path = REPO_ROOT / "Libs/configs/experiments" / registry_name
        if not path.is_file(): blockers.append({"code": "missing_split_registry", "message": registry_name})
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            schema_name = registry_name.replace(".yaml", ".schema.json")
            schema_path = REPO_ROOT / "Libs/configs/schema" / schema_name
            if schema_path.is_file():
                schema_errors = list(Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).iter_errors(payload))
                blockers.extend({"code": "split_registry_schema", "message": f"{registry_name}: {error.message}"} for error in schema_errors)
            key = {"ablation_registry.yaml": "ablation_uid", "deployment_comparison_registry.yaml": "deployment_uid", "related_work_registry.yaml": "related_uid"}[registry_name]
            split_ids[registry_name] = {str(row.get(key)) for row in payload.get("entries", [])}
    for filename, ids in split_ids.items():
        overlap = sorted(active_ids & ids)
        if overlap:
            blockers.append({"code": "active_split_overlap", "message": f"{filename}: {','.join(overlap)}"})
    if split_ids.get("deployment_comparison_registry.yaml") != {"large_only", "student_without_distillation", "fixed_escalation_or_no_cascade"}:
        blockers.append({"code": "deployment_slot_membership", "message": "Deployment comparison registry must contain exactly the three frozen profiles."})
    if split_ids.get("related_work_registry.yaml") and any(row.get("experimentally_compared") is not False for row in yaml.safe_load((REPO_ROOT / "Libs/configs/experiments/related_work_registry.yaml").read_text(encoding="utf-8")).get("entries", [])):
        blockers.append({"code": "related_work_experimental_claim", "message": "Related-work methods cannot be described as experimentally compared."})
    if validation_path.is_file():
        materialized = json.loads(validation_path.read_text(encoding="utf-8"))
        if materialized.get("status") != "passed": blockers.append({"code": "validation_not_passed", "message": "Materialized reduction validation is not passed."})
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("authorization_fail_closed") is not True: blockers.append({"code": "authorization_not_fail_closed", "message": "Reduction gate must be fail-closed."})
        if gate.get("unclassified_entries") != 0: blockers.append({"code": "unclassified_entries", "message": "Historical migration leaves unclassified entries."})
    auth_path = root / "neuronsifter_headline_training_authorization.json"
    if auth_path.is_file():
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        conditions = auth.get("authorization_conditions", {})
        if conditions.get("mandatory_qualified_20_of_20") and not conditions.get("mandatory_qualified"):
            blockers.append({"code": "authorization_count_alias", "message": "20-of-20 qualification flag is inconsistent."})
        if auth.get("headline_training_authorized") and not all(conditions.get(key) is True for key in ("mandatory_qualified_20_of_20", "activated_conditionals_qualified", "inactive_conditionals_have_incompatibility_proof", "upper_bounds_separated", "coevolution_compositions_complete", "test_accessed")):
            blockers.append({"code": "authorization_fail_open", "message": "Headline authorization bypasses reduced registry conditions."})
    return {"status": "passed" if not blockers else "failed", "blocker_count": len(blockers), "blockers": blockers, "historical_entries_reconciled": 57 if not blockers else 0, "new_registry_total": 27, "mandatory": 20, "conditional": 5, "upper_bounds": 2, "authorization_fail_closed": True}

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the 57-to-27 baseline registry reduction.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml"))
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    args = parser.parse_args()
    report = validate(args.registry, args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)

if __name__ == "__main__":
    main()
