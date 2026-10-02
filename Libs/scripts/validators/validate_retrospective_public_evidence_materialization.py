"""Validate public retrospective evidence materialization and its fail-closed gate."""

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

from Libs.scripts.builders.build_retrospective_public_evidence_materialization import (  # noqa: E402
    AUDIT_OUTPUT_PATH,
    BUILDER_PATH,
    CONTRACT_PATH,
    EVIDENCE_OUTPUT_PATH,
    MATERIALIZATION_SCHEMA_PATH,
    REJECTION_OUTPUT_PATH,
    build,
    _artifact,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        payload = json.loads(AUDIT_OUTPUT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(MATERIALIZATION_SCHEMA_PATH.read_text(encoding="utf-8"))
        schema_errors = list(Draft202012Validator(schema).iter_errors(payload))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    if schema_errors:
        blockers.extend(f"schema:{list(error.absolute_path)}:{error.message}" for error in schema_errors[:8])
    for key, path in (
        ("contract_binding", CONTRACT_PATH),
        ("materialization_schema_binding", MATERIALIZATION_SCHEMA_PATH),
        ("implementation_binding", BUILDER_PATH),
    ):
        if payload.get(key) != _artifact(path):
            blockers.append(key)
    if payload.get("protected_holdout_access") != 0:
        blockers.append("protected_holdout_access")
    for key in ("world_action_authorized", "agent_in_twin_authorized", "coevolution_authorized"):
        if payload.get(key) is not False:
            blockers.append(key)
    if not EVIDENCE_OUTPUT_PATH.is_file() or not REJECTION_OUTPUT_PATH.is_file():
        blockers.append("materialized_outputs_missing")
    status = str(payload.get("status", ""))
    if status == "qualified_public_candidates_but_split_insufficient":
        if payload.get("decision_sensitive_candidate_set_count", 0) < 1:
            blockers.append("split_insufficient_without_qualified_candidate")
        if payload.get("independent_decision_sensitive_group_count", 0) >= payload.get(
            "minimum_independent_replay_groups", 3
        ):
            blockers.append("split_insufficient_with_enough_independent_decision_groups")
        if payload.get("replay_roles_feasible") is not False:
            blockers.append("split_feasibility_flag")
    if payload.get("replay_roles_feasible") is True and payload.get(
        "independent_decision_sensitive_group_count", 0
    ) < payload.get("minimum_independent_replay_groups", 3):
        blockers.append("replay_feasible_without_enough_independent_decision_groups")
    if payload.get("candidate_set_count", 0) < payload.get("legal_candidate_set_count", 0):
        blockers.append("candidate_count_below_legal_count")
    if payload.get("legal_candidate_set_count", 0) < payload.get("decision_sensitive_candidate_set_count", 0):
        blockers.append("legal_count_below_decision_count")
    if recompute and not blockers:
        rebuilt = build(write_output=False)
        for key in (
            "status", "accepted_record_count", "rejected_record_count", "candidate_set_count",
            "legal_candidate_set_count", "decision_sensitive_candidate_set_count", "replay_roles_feasible",
            "independent_decision_sensitive_group_count",
        ):
            if rebuilt.get(key) != payload.get(key):
                blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-evidence-materialization-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "materialization_status": payload.get("status"),
        "accepted_record_count": payload.get("accepted_record_count"),
        "rejected_record_count": payload.get("rejected_record_count"),
        "candidate_set_count": payload.get("candidate_set_count"),
        "legal_candidate_set_count": payload.get("legal_candidate_set_count"),
        "decision_sensitive_candidate_set_count": payload.get("decision_sensitive_candidate_set_count"),
        "independent_decision_sensitive_group_count": payload.get("independent_decision_sensitive_group_count"),
        "replay_roles_feasible": payload.get("replay_roles_feasible"),
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
