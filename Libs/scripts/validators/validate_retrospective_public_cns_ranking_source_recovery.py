"""Validate the frozen public CNS ranking intake source recovery."""

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

from Libs.scripts.builders.build_retrospective_public_cns_ranking_source_recovery import (  # noqa: E402
    AUDIT_PATH,
    EXPECTED_QUEUE_DIGEST,
    SCHEMA_PATH,
    SOURCE_MANIFEST_PATH,
    SOURCE_PATH,
    _sha256,
    build,
)


def validate(*, recompute: bool = False) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    blockers.extend(f"schema:{list(error.absolute_path)}:{error.message}" for error in errors[:8])
    if payload.get("queue_digest") != EXPECTED_QUEUE_DIGEST:
        blockers.append("queue_digest")
    if not SOURCE_PATH.is_file() or payload.get("source_artifact_binding", {}).get("sha256") != _sha256(SOURCE_PATH):
        blockers.append("source_artifact_binding")
    if not SOURCE_MANIFEST_PATH.is_file() or payload.get("source_manifest_binding", {}).get("sha256") != _sha256(SOURCE_MANIFEST_PATH):
        blockers.append("source_manifest_binding")
    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            for key in (
                "status",
                "queue_digest",
                "publication_uid",
                "source_accession",
                "source_artifact_binding",
                "source_structure",
                "qualification",
                "citation_substitution_performed",
                "protected_holdout_access",
                "formal_evidence_admitted",
                "candidate_set_created",
                "independent_decision_sensitive_group_count",
                "required_group_count",
                "next_action",
            ):
                if rebuilt.get(key) != payload.get(key):
                    blockers.append(f"recompute:{key}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-source-recovery-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "recovery_status": payload.get("status"),
        "publication_uid": payload.get("publication_uid"),
        "formal_evidence_admitted": payload.get("formal_evidence_admitted"),
        "candidate_set_created": payload.get("candidate_set_created"),
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
