"""Validate the empty-queue source-recovery terminal audit."""

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

from Libs.scripts.builders.build_retrospective_public_design_enriched_source_recovery import (  # noqa: E402
    OUTPUT_PATH,
    SCHEMA_PATH,
    build,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    blockers.extend(f"schema:{list(error.absolute_path)}:{error.message}" for error in errors[:8])
    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            for key in (
                "status",
                "intake_uid",
                "preregistration_digest",
                "queue_digest",
                "metadata_record_count",
                "metadata_eligible_count",
                "queue_record_count",
                "source_recovered_count",
                "formal_abc_admitted_count",
                "new_legal_candidate_count",
                "new_decision_sensitive_group_count",
                "unchanged_artifact_bindings",
                "final_state",
                "split_authorized",
                "world_action_authorized",
                "gpu_authorized",
                "tri_authorized",
                "agent_in_twin_authorized",
                "coevolution_authorized",
                "protected_holdout_access",
                "next_action",
            ):
                if rebuilt.get(key) != payload.get(key):
                    blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-design-enriched-source-recovery-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "recovery_status": payload.get("status"),
        "metadata_record_count": payload.get("metadata_record_count"),
        "metadata_eligible_count": payload.get("metadata_eligible_count"),
        "queue_record_count": payload.get("queue_record_count"),
        "source_recovered_count": payload.get("source_recovered_count"),
        "independent_decision_sensitive_group_count": payload.get("final_state", {}).get("independent_decision_sensitive_groups"),
        "protected_holdout_access": payload.get("protected_holdout_access"),
        "next_action": payload.get("next_action"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
