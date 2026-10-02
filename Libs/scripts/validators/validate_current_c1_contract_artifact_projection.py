"""Validate the current C1 contract artifact projection against live inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_current_c1_contract_artifact_projection import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_projection,
    validate_schema,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import safe_repo_path  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_payload(
    payload: Mapping[str, object], repo_root: Path = REPO_ROOT
) -> dict[str, object]:
    blockers: list[str] = []
    try:
        validate_schema(payload, repo_root)
    except (TypeError, ValueError) as exc:
        blockers.append(f"projection_schema_invalid:{exc}")
    try:
        expected, _ = build_projection(repo_root)
        if canonical_sha256(payload) != canonical_sha256(expected):
            blockers.append("projection_live_recompute_mismatch")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(f"projection_live_recompute_failed:{exc}")
        expected = {}
    return {
        "status": "valid_fail_closed" if not blockers else "blocked",
        "projection_status": payload.get("status", "missing"),
        "blockers": blockers,
        "live_recompute_matches": bool(expected)
        and canonical_sha256(payload) == canonical_sha256(expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    payload = _load_json(safe_repo_path(REPO_ROOT, args.projection))
    report = validate_payload(payload, REPO_ROOT)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "valid_fail_closed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
