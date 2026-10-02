"""Write current C2 and baseline readiness artifacts from immutable gate inputs.

No candidate is selected, no manifest is fabricated, and no baseline run is
created when the underlying data, rights, or split gates remain blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv


BASELINE_SOURCE_FIELDS = ["baseline_uid", "display_name", "claim_ids", "repository_or_package", "tag_or_commit", "license", "source_decision", "license_decision", "qualification_status"]
BASELINE_ENVIRONMENT_FIELDS = ["baseline_uid", "environment", "environment_path_exists", "environment_lock_path", "environment_lock_hash", "environment_decision"]
BASELINE_REPRODUCTION_FIELDS = ["baseline_uid", "upstream_reproduction_status", "reason"]
BASELINE_ADAPTER_FIELDS = ["baseline_uid", "input_adapter", "output_adapter", "adapter_readiness", "reason"]
C2_SCORECARD_FIELDS = ["candidate_uid", "source", "publication_accession", "action_complete", "matched_controls_complete", "independent", "strict_split_available", "status", "reason"]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _record_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    program = dict(config["baseline_first_program"])
    reduction = dict(config["reduction_registry"])
    for collection in (program["mandatory_baselines"], reduction["mandatory"], reduction["conditional"]):
        for raw in collection:
            record = dict(raw)
            uid = str(record["baseline_uid"])
            records[uid] = {**records.get(uid, {}), **record}
    return records


def _source_decision(record: Mapping[str, Any]) -> str:
    repository = str(record.get("repository_or_package", ""))
    pin = str(record.get("tag_or_commit", ""))
    if repository.startswith("project://"):
        return "resolved_project_owned"
    if pin and "pending" not in pin:
        return "source_pin_declared_evidence_pending"
    return "blocked_source_pin_missing"


def _license_decision(record: Mapping[str, Any]) -> str:
    repository = str(record.get("repository_or_package", ""))
    license_name = str(record.get("license", ""))
    if repository.startswith("project://"):
        return "resolved_project_owned"
    if license_name and "pending" not in license_name.lower():
        return "license_declared_evidence_pending"
    return "blocked_license_evidence_missing"


def build(repo_root: Path, output_root: Path) -> dict[str, Any]:
    load_config(repo_root / "Libs/configs/config_stack.yaml")
    with (repo_root / "Libs/configs/experiments/baseline_registry.yaml").open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    records = _record_index(registry)
    reduction = registry["reduction_registry"]
    mandatory = [records[str(item["baseline_uid"])] for item in reduction["mandatory"]]
    conditional = [records[str(item["baseline_uid"])] for item in reduction["conditional"]]

    source_rows, environment_rows, reproduction_rows, adapter_rows = [], [], [], []
    for record in mandatory:
        source = _source_decision(record)
        license_decision = _license_decision(record)
        source_rows.append({
            "baseline_uid": record["baseline_uid"], "display_name": record["display_name"],
            "claim_ids": "|".join(record.get("claim_ids", [])), "repository_or_package": record.get("repository_or_package", ""),
            "tag_or_commit": record.get("tag_or_commit", ""), "license": record.get("license", ""),
            "source_decision": source, "license_decision": license_decision,
            "qualification_status": "blocked_data_or_source_gate",
        })
        environment = str(record.get("environment", ""))
        lock_path = str(record.get("environment_lock_path", ""))
        environment_rows.append({
            "baseline_uid": record["baseline_uid"], "environment": environment,
            "environment_path_exists": (repo_root / environment).exists(), "environment_lock_path": lock_path,
            "environment_lock_hash": record.get("environment_lock_sha256", ""),
            "environment_decision": "blocked_environment_lock_missing" if lock_path == "not_acquired" else "environment_recorded",
        })
        reproduction_rows.append({"baseline_uid": record["baseline_uid"], "upstream_reproduction_status": "not_run_data_gate_blocked", "reason": "Frozen C1/C2 challenge data and splits are not qualified."})
        adapter_rows.append({"baseline_uid": record["baseline_uid"], "input_adapter": record.get("input_adapter", ""), "output_adapter": record.get("output_adapter", ""), "adapter_readiness": "not_run_data_gate_blocked", "reason": "No qualified frozen task contract is available."})

    legacy_gate = _read_json(repo_root / "Output/audits/c2r_entry_gate_decision.json")
    legacy_gap = _read_json(repo_root / "Output/audits/c2r_gap_decomposition.json")
    candidates = _read_tsv(repo_root / "Output/audits/c2r_real_trajectory_candidate_audit.tsv")
    scorecard_rows = [{
        "candidate_uid": row["candidate_uid"], "source": row["source"], "publication_accession": row["publication_accession"],
        "action_complete": False, "matched_controls_complete": False, "independent": False, "strict_split_available": False,
        "status": "not_qualified", "reason": row["reason"],
    } for row in candidates]
    c2_selected = {"status": "blocked_no_qualified_domain", "selected_domain": None, "manifest_creation_allowed": False, "reason": legacy_gate["claim_boundary"]}
    c2_readiness = {
        "schema_version": "neuronsifter-c2-domain-readiness",
        "status": "blocked", "action_complete": False, "qualified_producer": False,
        "matched_controls_complete": False, "independent_trajectory_units_nonzero": False,
        "parameter_provenance_complete": False, "strict_split_overlap": None,
        "final_holdout_unseen": True, "baseline_challenge_allowed": False,
        "selected_domain": None, "legacy_gate_status": legacy_gate["status"],
        "blockers": [item["requirement_id"] for item in legacy_gap["requirements"] if item["shortfall"] == "unmet"],
        "next_action": "obtain_a_public_action_complete_domain_with_matched_controls_before_creating_c2_manifests",
    }
    conditional_payload = {
        "status": "blocked", "conditional_count": len(conditional), "test_access_allowed": False,
        "decisions": [{
            "baseline_uid": record["baseline_uid"], "status": "inactive_without_incompatibility_proof",
            "activation_condition": record.get("activation_condition", ""),
            "reason": "The C1/C2 frozen data and split gates are blocked; activation and incompatibility evidence cannot be finalized.",
        } for record in conditional],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_tsv(output_root / "baseline_source_license_matrix.tsv", BASELINE_SOURCE_FIELDS, source_rows)
    write_tsv(output_root / "baseline_environment_matrix.tsv", BASELINE_ENVIRONMENT_FIELDS, environment_rows)
    write_tsv(output_root / "baseline_upstream_reproduction.tsv", BASELINE_REPRODUCTION_FIELDS, reproduction_rows)
    write_tsv(output_root / "baseline_adapter_readiness.tsv", BASELINE_ADAPTER_FIELDS, adapter_rows)
    write_json(output_root / "baseline_conditional_activation.json", conditional_payload)
    write_tsv(output_root / "c2_domain_scorecard.tsv", C2_SCORECARD_FIELDS, scorecard_rows)
    write_json(output_root / "c2_selected_domain.json", c2_selected)
    write_json(output_root / "c2_domain_readiness.json", c2_readiness)
    return {"mandatory_count": len(mandatory), "conditional_count": len(conditional), "c2_readiness": c2_readiness}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fail-closed Round-20 C2 and baseline readiness artifacts.")
    parser.add_argument("--output-root", default="Output/audits")
    args = parser.parse_args()
    result = build(REPO_ROOT, (REPO_ROOT / args.output_root).resolve())
    print(json.dumps({"mandatory_count": result["mandatory_count"], "conditional_count": result["conditional_count"], "c2_status": result["c2_readiness"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
