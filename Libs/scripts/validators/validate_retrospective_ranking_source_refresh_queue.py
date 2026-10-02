"""Validate the append-only metadata-only public-source refresh queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_ranking_source_refresh_queue import (  # noqa: E402
    AUDIT_PATH,
    FROZEN_PARENT_QUEUE_DIGEST,
    QUEUE_PATH,
    SCHEMA_PATH,
    build,
)


def validate(*, write_output: bool = False) -> dict[str, object]:
    audit = build(write_output=write_output)
    expected_queue_path = QUEUE_PATH.relative_to(REPO_ROOT).as_posix()
    if audit["queue_path"] != expected_queue_path:
        raise ValueError(f"Ranking source refresh queue path drifted: {audit['queue_path']}")
    if audit["parent_frozen_queue_digest"] != FROZEN_PARENT_QUEUE_DIGEST:
        raise ValueError("Ranking source refresh parent digest drifted")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(audit),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Ranking source refresh queue validation failed: {detail}")
    return {
        "status": "passed",
        "queue_digest": audit["queue_digest"],
        "record_count": audit["record_count"],
        "parent_frozen_queue_digest": audit["parent_frozen_queue_digest"],
        "parent_queue_unchanged": audit["parent_queue_unchanged"],
        "protected_holdout_access": audit["protected_holdout_access"],
        "formal_evidence_admitted": audit["formal_evidence_admitted"],
        "audit_path": AUDIT_PATH.relative_to(REPO_ROOT).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(write_output=args.write), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
