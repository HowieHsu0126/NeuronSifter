"""Validate the persisted active-contract migration audit against live inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_active_contract_migration import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build,
    validate_schema,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


DEFAULT_VALIDATION_PATH = "Output/audits/active_contract_migration_validation.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_payload(
    payload: Mapping[str, object], repo_root: Path = REPO_ROOT
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    try:
        validate_schema(payload, repo_root)
    except (TypeError, ValueError) as exc:
        blockers.append(
            {
                "code": "active_contract_migration_schema_invalid",
                "path": "Libs/configs/schema/active_contract_migration.schema.json",
                "message": str(exc),
            }
        )
    try:
        expected = build(repo_root)
        if canonical_sha256(payload) != canonical_sha256(expected):
            blockers.append(
                {
                    "code": "active_contract_migration_live_recompute_mismatch",
                    "path": DEFAULT_OUTPUT_PATH,
                    "message": "The persisted audit differs from the current contract and artifact state.",
                }
            )
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(
            {
                "code": "active_contract_migration_live_recompute_failed",
                "path": DEFAULT_OUTPUT_PATH,
                "message": str(exc),
            }
        )
        expected = {}
    return {
        "schema_version": "neuronsifter-active-contract-migration-validation",
        "status": "valid_fail_closed" if not blockers else "blocked",
        "audit_status": payload.get("status", "missing"),
        "migration_binding_valid": payload.get("migration_binding_valid", False),
        "execution_artifacts_ready": payload.get("execution_artifacts_ready", False),
        "audit_sha256": canonical_sha256(payload),
        "live_recompute_sha256": canonical_sha256(expected) if expected else None,
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    audit_path = safe_repo_path(repo_root, DEFAULT_OUTPUT_PATH)
    payload = _load_json(audit_path)
    report = validate_payload(payload, repo_root)
    report["audit_path"] = DEFAULT_OUTPUT_PATH
    report["audit_file_sha256"] = sha256_file(audit_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_VALIDATION_PATH)
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "valid_fail_closed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
