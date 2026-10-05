"""Materialize BASELINE_FIRST's fail-closed pre-execution and handoff audits.

This builder records only evidence available at invocation time.  It is not a
test runner and cannot promote a partial, dirty, or historical test execution
into an exact-clean BASELINE_FIRST full-suite result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "Libs/configs/experiments/baseline_first.yaml"
DEFAULT_SCHEMA = "Libs/configs/schema/baseline_first_contract.schema.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    path.relative_to(root.resolve())
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(("git", "-C", str(root), *args), text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def file_binding(root: Path, value: str) -> dict[str, Any]:
    path = repo_path(root, value)
    return {
        "path": value,
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def load_contract(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_path = REPO_ROOT / DEFAULT_SCHEMA
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(payload))
    if errors:
        raise ValueError("BASELINE_FIRST config schema validation failed: " + "; ".join(error.message for error in errors))
    if not isinstance(payload, dict) or not isinstance(payload.get("baseline_first"), dict):
        raise ValueError("BASELINE_FIRST config must contain baseline_first.")
    return dict(payload["baseline_first"])


def candidate_identity(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    # The live tree is deliberately not scanned: BASELINE_FIRST's formal identity is a
    # staged Git candidate, and a shared-filesystem scan is neither bounded nor
    # evidence for that candidate's cleanliness.
    status_lines: list[str] = []
    return {
        "candidate_commit": git_output(root, "rev-parse", "HEAD"),
        "candidate_tree": git_output(root, "rev-parse", "HEAD^{tree}"),
        "git_backed": bool(git_output(root, "rev-parse", "--is-inside-work-tree") == "true"),
        "live_worktree_clean": False,
        "live_worktree_status_scan_completed": False,
        "live_worktree_change_count": None,
        "live_worktree_status_sha256": canonical_sha256(status_lines),
        "clean_checkout_ready": False,
        "node_local_staging_ready": False,
        "precondition_blockers": [
            "live_worktree_not_scanned_formal_cleanliness_requires_staged_candidate",
            "baseline_first_candidate_commit_not_materialized",
            "node_local_runtime_dependency_manifest_not_materialized",
        ],
        "forbidden_runtime_actions": list(contract["candidate"]["forbidden_runtime_actions"]),
    }


def _registry_counts(root: Path) -> dict[str, int]:
    registry_path = root / "Libs/configs/experiments/baseline_registry.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    reduced = payload.get("reduction_registry", {})
    return {
        "mandatory": len(reduced.get("mandatory", [])),
        "conditional": len(reduced.get("conditional", [])),
        "upper_bounds": len(reduced.get("upper_bounds", [])),
    }


def _historical_reconciliation(root: Path, ledger_path: str) -> int:
    path = repo_path(root, ledger_path)
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return len(list(csv.DictReader(handle, delimiter="\t")))


def build(config: str | Path = DEFAULT_CONFIG, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    root = repo_root.resolve()
    config_path = Path(config)
    config_path = config_path if config_path.is_absolute() else root / config_path
    contract = load_contract(config_path)
    freeze_bindings = {
        key: file_binding(root, value)
        for key, value in dict(contract["historical_freeze_inputs"]).items()
    }
    inherited = {
        key: read_json(repo_path(root, value))
        for key, value in dict(contract["inherited_audits"]).items()
    }
    counts = _registry_counts(root)
    expected_counts = dict(contract["expected_counts"])
    registry_valid = (
        counts == {key: expected_counts[key] for key in counts}
        and _historical_reconciliation(root, contract["historical_freeze_inputs"]["migration_ledger"])
        == expected_counts["historical_entries"]
        and bool(inherited["reduction_validation"] and inherited["reduction_validation"].get("status") == "passed")
    )
    c1 = inherited["c1_preflight"] or {}
    c2 = inherited["c2_preflight"] or {}
    c2r = inherited["c2r_entry_gate"] or {}
    authorization = inherited["baseline_authorization"] or {}
    scoped = inherited["claim_scoped_authorization"] or {}
    scoped_semantic = inherited["claim_scoped_semantic_validation"] or {}
    scoped_claims = {
        str(row.get("claim_id")): row
        for row in scoped.get("claim_scoped_authorization", [])
        if isinstance(row, dict)
    }
    identity = candidate_identity(root, contract)
    full_suite = {
        "schema_version": "neuronsifter-baseline_first-full-suite-execution-v1",
        "candidate": identity,
        "test_inventory_bound": False,
        "terminal_execution": "not_started",
        "exit_code": None,
        "full_suite_failures": None,
        "full_suite_errors": None,
        "mandatory_skips": None,
        "full_suite_green": False,
        "reason": "BASELINE_FIRST requires a newly materialized Git-backed clean checkout and node-local dependency staging before full-suite execution.",
    }
    decomposition = [
        {
            "record_id": "baseline_first_preexecution_001",
            "record_kind": "execution_precondition",
            "test_id": "",
            "classification": "filesystem_or_timeout",
            "root_cause": "A current clean Git checkout and node-local runtime dependency manifest have not been materialized.",
            "status": "blocked_before_test_execution",
            "evidence_path": "Output/audits/baseline_first_full_suite_execution.json",
        }
    ]
    runtime_dependencies = {
        "schema_version": "neuronsifter-baseline_first-runtime-dependency-manifest-v1",
        "candidate": identity,
        "status": "blocked_pre_execution",
        "declared_staging_mode": "git_archive_plus_hash_verified_read_only_dependencies",
        "pytest_runtime": dict(contract["candidate"]["pytest_runtime"]),
        "staged_dependency_count": 0,
        "node_local_scratch": "not_materialized",
        "shared_memory_dependency": "not_used_as_full_suite_requirement",
        "blockers": identity["precondition_blockers"],
    }
    inventory = {
        "schema_version": "neuronsifter-baseline_first-test-inventory-v1",
        "candidate": identity,
        "status": "not_collected_no_clean_candidate",
        "test_count": None,
        "collection_command": "python -m pytest Libs/tests --collect-only -q",
        "reason": "Collection must execute from the same staged clean candidate as the full suite.",
    }
    inventory_diff = {
        "schema_version": "neuronsifter-baseline_first-test-inventory-diff-v1",
        "status": "not_comparable_no_clean_candidate",
        "missing_inherited_tests": None,
        "unreconciled_new_tests": None,
        "reason": "No BASELINE_FIRST clean-candidate inventory exists yet.",
    }
    semantic = {
        "schema_version": "neuronsifter-baseline_first-registry-semantic-validation-v1",
        "status": "passed" if registry_valid else "failed",
        "registry_counts_valid": registry_valid,
        "historical_reconciliation": f"{_historical_reconciliation(root, contract['historical_freeze_inputs']['migration_ledger'])}_of_{expected_counts['historical_entries']}",
        "authorization_dependency_cycles": 0 if scoped_semantic.get("dependency_graph_valid") is True else None,
        "self_attested_evidence": len(scoped_semantic.get("self_attested_evidence_records", [])),
        "test_access_before_threshold_freeze": None,
    }
    c1_resolved = bool(c1.get("headline_training_allowed") is True and scoped_claims.get("C1", {}).get("status") == "authorized")
    c2_ready = bool(c2.get("status") == "c2_baseline_data_ready" and c2r.get("status") == "qualified")
    c2_resolved = bool(c2_ready and scoped_claims.get("C2", {}).get("status") == "authorized")
    validation = {
        "schema_version": "neuronsifter-baseline_first-validation-v1",
        "status": "blocked",
        "full_suite_green": False,
        "registry_valid": registry_valid,
        "mandatory_baselines": {"source_license_decisions": "0_of_20"},
        "conditional_baselines": {"activation_decisions": "0_of_5"},
        "c1": {"challenge_resolved": c1_resolved, "thresholds_frozen": False},
        "c2": {
            "action_complete_domain": c2_ready,
            "challenge_resolved": c2_resolved,
            "thresholds_frozen": False,
            "current_domain_status": c2r.get("status", "missing"),
        },
        "final_holdout_accessed": bool(authorization.get("authorization_conditions", {}).get("test_data_unseen") is False),
        "neuronsifter_training_started": bool(authorization.get("headline_training_authorized") is True),
        "coevolution_started": False,
        "blockers": [
            "baseline_first_clean_candidate_not_materialized",
            "full_suite_not_executed",
            "mandatory_source_license_qualification_incomplete",
            "conditional_activation_decisions_incomplete",
            "c1_headline_capability_unreachable",
            "c2_action_complete_independent_domain_unavailable",
            "protected_final_holdout_not_established",
        ],
    }
    gate = {
        "schema_version": "neuronsifter-baseline_first-gate-v1",
        "gate_name": contract["gate_name"],
        "status": contract["status_when_blocked"],
        "baseline_first_c1_c2_challenge_resolved": False,
        "validation": validation,
        "freeze_inputs": freeze_bindings,
    }
    handoff = {
        "schema_version": "neuronsifter-baseline_first-mainline-handoff-v1",
        "status": "blocked",
        "next_action": "materialize_clean_baseline_first_candidate_then_acquire_provenance_complete_C2_domain_before_any_baseline_challenge",
        "headline_training_authorized": False,
        "prohibited_actions": identity["forbidden_runtime_actions"],
        "gate_name": contract["gate_name"],
    }
    scoped_path = repo_path(root, contract["inherited_audits"]["claim_scoped_authorization"])
    baseline_first_scoped_authorization = {
        "schema_version": "neuronsifter-baseline_first-claim-scoped-authorization-v1",
        "status": scoped.get("status", "missing"),
        "source_artifact": file_binding(root, contract["inherited_audits"]["claim_scoped_authorization"]),
        "source_artifact_sha256": sha256_file(scoped_path) if scoped_path.is_file() else "",
        "claim_scoped_authorization": scoped.get("claim_scoped_authorization", []),
        "global_authorized": False,
    }
    return {
        "full_suite": full_suite,
        "decomposition": decomposition,
        "runtime_dependencies": runtime_dependencies,
        "inventory": inventory,
        "inventory_diff": inventory_diff,
        "semantic": semantic,
        "validation": validation,
        "gate": gate,
        "handoff": handoff,
        "baseline_first_scoped_authorization": baseline_first_scoped_authorization,
    }


def write_all(payload: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    audits = repo_root / "Output/audits"
    audits.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "baseline_first_full_suite_execution.json": payload["full_suite"],
        "baseline_first_runtime_dependency_manifest.json": payload["runtime_dependencies"],
        "baseline_first_test_inventory.json": payload["inventory"],
        "baseline_first_test_inventory_diff.json": payload["inventory_diff"],
        "baseline_first_validation.json": payload["validation"],
        "baseline_first_gate.json": payload["gate"],
        "baseline_first_mainline_handoff.json": payload["handoff"],
        "baseline_first_claim_scoped_authorization.json": payload["baseline_first_scoped_authorization"],
    }
    for filename, value in json_outputs.items():
        (audits / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (audits / "baseline_first_full_suite_failure_decomposition.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["decomposition"][0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(payload["decomposition"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed BASELINE_FIRST pre-execution audits.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args.config)
    write_all(payload)
    print(json.dumps({"status": payload["gate"]["status"], "resolved": False}, sort_keys=True))


if __name__ == "__main__":
    main()
