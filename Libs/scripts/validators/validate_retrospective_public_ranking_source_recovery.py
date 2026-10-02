"""Validate the persisted post-preregistration ranking-source recovery audit."""

from __future__ import annotations

import argparse
import json
from typing import Any

from jsonschema import Draft202012Validator

from Libs.scripts.builders.build_retrospective_public_ranking_source_recovery import (
    AUDIT_PATH,
    SCHEMA_PATH,
    build,
)


def validate(*, recompute: bool = False, write_output: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    if write_output:
        try:
            build(write_output=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"status": "blocked", "passed": False, "blockers": [f"build:{exc}"]}
    try:
        payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.absolute_path))
    blockers.extend(f"schema:{list(error.absolute_path)}:{error.message}" for error in errors[:8])
    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            for key in (
                "status", "publication_uid", "source_accession", "parent_frozen_queue_digest",
                "refresh_queue_digest", "metadata_snapshot_binding", "source_manifest_binding",
                "source_artifact_binding", "xml_structure", "qualification",
                "protected_holdout_access", "formal_evidence_admitted", "candidate_set_created",
                "next_action",
            ):
                if rebuilt.get(key) != payload.get(key):
                    blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-ranking-source-recovery-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "recovery_status": payload.get("status"),
        "publication_uid": payload.get("publication_uid"),
        "formal_evidence_admitted": payload.get("formal_evidence_admitted"),
        "candidate_set_created": payload.get("candidate_set_created"),
        "protected_holdout_access": payload.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute, write_output=args.write)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
