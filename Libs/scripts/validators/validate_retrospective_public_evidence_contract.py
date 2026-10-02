"""Validate the frozen public-only retrospective evidence contract."""

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

from Libs.scripts.builders.build_retrospective_public_evidence_contract import (  # noqa: E402
    CONTRACT_CONFIG_PATH,
    LEGACY_CONTRACT_PATH,
    OUTPUT_PATH,
    SCHEMA_PATH,
    artifact,
    build,
    canonical_hash,
    load_contract,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        contract = load_contract()
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.absolute_path))
    blockers.extend(f"contract_schema:{list(error.absolute_path)}:{error.message}" for error in errors)
    if payload.get("schema_version") != "neuronsifter-retrospective-public-evidence-contract-audit-v1":
        blockers.append("audit_schema_version")
    if payload.get("contract_digest") != canonical_hash(contract):
        blockers.append("contract_digest")
    if payload.get("contract_binding") != artifact(CONTRACT_CONFIG_PATH):
        blockers.append("contract_binding")
    if payload.get("schema_binding") != artifact(SCHEMA_PATH):
        blockers.append("schema_binding")
    builder_path = REPO_ROOT / "Libs/scripts/builders/build_retrospective_public_evidence_contract.py"
    if payload.get("implementation_binding") != artifact(builder_path):
        blockers.append("implementation_binding")
    if payload.get("legacy_contract_binding") != artifact(LEGACY_CONTRACT_PATH):
        blockers.append("legacy_contract_binding")
    expected = {
        "admission_contract_frozen": True,
        "summary_statistics_frozen": True,
        "objective_selection_outcome_blind": True,
        "author_prose_as_ground_truth": False,
        "simulator_or_docking_as_empirical_truth": False,
        "protected_holdout_access": 0,
        "publication_claim_eligible": False,
        "legacy_contract_modified": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            blockers.append(key)
    if payload.get("formal_evidence_grades") != ["A", "B", "C"]:
        blockers.append("formal_evidence_grades")
    if payload.get("non_formal_evidence_grades") != ["D", "M"]:
        blockers.append("non_formal_evidence_grades")
    if recompute and not blockers:
        rebuilt = build(write_output=False)
        for key in ("contract_digest", "contract_binding", "schema_binding", "implementation_binding", "legacy_contract_binding"):
            if rebuilt.get(key) != payload.get(key):
                blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-evidence-contract-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "formal_evidence_grades": payload.get("formal_evidence_grades"),
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
