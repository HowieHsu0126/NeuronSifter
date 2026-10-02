"""Validate sealed retrospective replay and fail closed on leakage or missing evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_replay_benchmark import (  # noqa: E402
    BENCHMARK_CONFIG_PATH,
    BENCHMARK_SCHEMA_PATH,
    BUILDER_PATH,
    OUTPUT_PATH,
    artifact,
    build,
    load_contract,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        contract = load_contract()
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    if payload.get("schema_version") != "neuronsifter-retrospective-replay-benchmark-audit-v1":
        blockers.append("audit_schema_version")
    if payload.get("contract_version") != contract["contract_version"]:
        blockers.append("contract_version")
    if payload.get("contract_binding") != artifact(BENCHMARK_CONFIG_PATH):
        blockers.append("contract_binding")
    if payload.get("schema_binding") != artifact(BENCHMARK_SCHEMA_PATH):
        blockers.append("schema_binding")
    if payload.get("implementation_binding") != artifact(BUILDER_PATH):
        blockers.append("implementation_binding")
    for key, expected in {
        "split_policy_enforced": True,
        "objective_selection_outcome_blind": True,
        "current_live_web_search_allowed": False,
        "protected_holdout_access": 0,
        "publication_claim_eligible": False,
        "world_action_authorized": False,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
    }.items():
        if payload.get(key) != expected:
            blockers.append(key)
    leakage = payload.get("leakage_audit")
    if not isinstance(leakage, dict) or leakage.get("status") != "passed":
        blockers.append("leakage_audit")
    if payload.get("split_files_created") is True:
        if payload.get("decision_sensitive_candidate_set_count", 0) < 1:
            blockers.append("split_without_qualified_candidate")
        if payload.get("independent_decision_sensitive_group_count", 0) < payload.get(
            "minimum_independent_decision_sensitive_groups", 3
        ):
            blockers.append("split_without_enough_independent_decision_groups")
        if payload.get("tool_access_snapshot_present") is not True or payload.get("hidden_outcome_manifest_present") is not True:
            blockers.append("split_without_sealed_manifests")
    if payload.get("status") == "qualified_public_candidates_but_split_insufficient" and payload.get(
        "independent_decision_sensitive_group_count", 0
    ) >= payload.get("minimum_independent_decision_sensitive_groups", 3):
        if "split_role_assignment_missing" not in payload.get("blockers", []):
            blockers.append("split_insufficient_status_without_independence_or_role_blocker")
    if payload.get("status") == "qualified_sealed_retrospective_benchmark" and payload.get("split_files_created") is not True:
        blockers.append("qualified_status_without_split")
    if recompute and not blockers:
        rebuilt = build(write_output=False)
        for key in (
            "contract_binding", "schema_binding", "implementation_binding", "candidate_set_count",
            "decision_sensitive_candidate_set_count", "independent_decision_sensitive_group_count",
            "split_files_created", "status",
        ):
            if rebuilt.get(key) != payload.get(key):
                blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-replay-benchmark-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "benchmark_status": payload.get("status"),
        "candidate_set_count": payload.get("candidate_set_count"),
        "decision_sensitive_candidate_set_count": payload.get("decision_sensitive_candidate_set_count"),
        "independent_decision_sensitive_group_count": payload.get("independent_decision_sensitive_group_count"),
        "split_files_created": payload.get("split_files_created"),
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
