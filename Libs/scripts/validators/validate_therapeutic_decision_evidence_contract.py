"""Validate the frozen therapeutic evidence admission contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_therapeutic_decision_evidence_contract import (  # noqa: E402
    CONTRACT_CONFIG_PATH,
    OUTPUT_PATH,
    SCHEMA_PATH,
    _artifact,
    _canonical_hash,
    _load_contract,
    build,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        contract = _load_contract()
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    for error in Draft202012Validator(schema).iter_errors(contract):
        blockers.append(f"contract_schema:{list(error.absolute_path)}:{error.message}")
    if payload.get("schema_version") != "neuronsifter-therapeutic-decision-evidence-contract-audit-v2":
        blockers.append("audit_schema_version")
    if payload.get("contract_digest") != _canonical_hash(contract):
        blockers.append("contract_digest")
    if payload.get("contract_binding") != _artifact(CONTRACT_CONFIG_PATH):
        blockers.append("contract_binding")
    if payload.get("schema_binding") != _artifact(SCHEMA_PATH):
        blockers.append("schema_binding")
    if payload.get("implementation_binding") != _artifact(
        REPO_ROOT / "Libs/scripts/builders/build_therapeutic_decision_evidence_contract.py"
    ):
        blockers.append("implementation_binding")
    for name, binding in payload.get("gate_bindings", {}).items():
        path = REPO_ROOT / str(binding.get("path", ""))
        if not path.is_file() or _artifact(path) != binding:
            blockers.append(f"gate_binding:{name}")
    if payload.get("admission_contract_frozen") is not True:
        blockers.append("contract_not_frozen")
    if payload.get("objective_selection_outcome_blind") is not True:
        blockers.append("objective_not_outcome_blind")
    if payload.get("protected_holdout_access") != 0:
        blockers.append("protected_holdout_access")
    if payload.get("historical_round21_authority_recreated") is not False:
        blockers.append("historical_round21_recreated")
    if payload.get("old_round22_tier_c_replay_used") is not False:
        blockers.append("old_tier_c_replay_used")
    if payload.get("observed_gate", {}).get("targeted_therapeutic_qualified_sets") != 0:
        blockers.append("qualified_sets_baseline_drift")
    if payload.get("observed_gate", {}).get("current_gate") != (
        "blocked_insufficient_decision_sensitive_therapeutic_outcomes"
    ):
        blockers.append("current_gate_drift")
    if payload.get("observed_gate", {}).get("benchmark_split_created") is not False:
        blockers.append("benchmark_split_prequalified")
    expected_downstream = contract["downstream_authorization"]
    if payload.get("downstream_authorization") != expected_downstream:
        blockers.append("downstream_authorization")
    if recompute and not blockers:
        rebuilt = build(write_output=False)
        for key in ("contract_digest", "gate_bindings", "observed_gate", "downstream_authorization"):
            if rebuilt.get(key) != payload.get(key):
                blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-therapeutic-decision-evidence-contract-validation-v2",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "current_gate": payload.get("observed_gate", {}).get("current_gate"),
        "qualified_therapeutic_sets": payload.get("observed_gate", {}).get("targeted_therapeutic_qualified_sets"),
        "protected_holdout_access": payload.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
