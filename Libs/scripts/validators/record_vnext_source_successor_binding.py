"""Record an append-only source successor for the vNext provenance chain.

The Round-23 full-suite audit and its Round-24/24B successor bindings are
historical evidence.  This utility records only the newly observed checkout
transitions and refuses to rewrite an existing binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


ROUND23_AUDIT = REPO_ROOT / "Output/audits/round23_full_suite_execution.json"
ROUND24_BINDING = REPO_ROOT / "Output/audits/round24_round23_source_successor_binding.json"
ROUND24B_BINDING = REPO_ROOT / "Output/audits/round24b_round23_source_successor_binding.json"
ROUND25_BINDING = REPO_ROOT / "Output/audits/round25_round23_source_successor_binding.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("path", "")): row
        for row in payload.get("source_transitions", [])
        if isinstance(row, dict)
    }


def _require_predecessor(
    audit: dict[str, Any],
    binding: dict[str, Any],
    *,
    schema_version: str,
    predecessor_path: Path | None = None,
) -> dict[str, str]:
    source_rows = {
        str(row.get("path", "")): row
        for row in audit.get("source_files", [])
        if isinstance(row, dict)
    }
    transitions = _rows(binding)
    if (
        audit.get("schema_version") != "neuronsifter-round23-full-suite-execution-v1"
        or audit.get("status") != "passed"
        or binding.get("schema_version") != schema_version
        or binding.get("status") != "bound_successor_development"
        or binding.get("round23_full_suite_audit_sha256") != sha256_file(ROUND23_AUDIT)
        or not set(transitions).issubset(source_rows)
        or binding.get("publication_claim_eligible") is not False
        or binding.get("protected_test_access") != 0
    ):
        raise ValueError("Historical vNext source binding is invalid")
    effective = {
        path: str(row.get("sha256", "")) for path, row in source_rows.items()
    }
    for path, row in transitions.items():
        predecessor_hash = str(row.get("predecessor_sha256", row.get("round23_sha256", "")))
        successor_hash = str(
            row.get("round24b_sha256", row.get("round24_sha256", ""))
        )
        if (
            row.get("round23_sha256") != source_rows[path].get("sha256")
            or predecessor_hash != effective[path]
            or not successor_hash
        ):
            raise ValueError(f"Historical vNext source transition is invalid: {path}")
        effective[path] = successor_hash
    if predecessor_path is not None:
        declared = binding.get("predecessor_binding", {})
        if (
            not isinstance(declared, dict)
            or declared.get("path") != predecessor_path.relative_to(REPO_ROOT).as_posix()
            or declared.get("sha256") != sha256_file(predecessor_path)
        ):
            raise ValueError("Historical vNext predecessor binding hash is invalid")
    return effective


def build_binding() -> dict[str, Any]:
    audit = load_json(ROUND23_AUDIT)
    round24 = load_json(ROUND24_BINDING)
    round24b = load_json(ROUND24B_BINDING)
    source_rows = {
        str(row.get("path", "")): row
        for row in audit.get("source_files", [])
        if isinstance(row, dict)
    }
    effective = _require_predecessor(
        audit,
        round24,
        schema_version="neuronsifter-round24-round23-source-successor-binding-v1",
    )
    effective_before_round24b = dict(effective)
    declared_round24b = round24b.get("predecessor_binding", {})
    if (
        declared_round24b.get("path") != ROUND24_BINDING.relative_to(REPO_ROOT).as_posix()
        or declared_round24b.get("sha256") != sha256_file(ROUND24_BINDING)
    ):
        raise ValueError("Round-24B predecessor binding hash is invalid")
    round24b_rows = _rows(round24b)
    if (
        round24b.get("schema_version")
        != "neuronsifter-round24b-round23-source-successor-binding-v1"
        or round24b.get("status") != "bound_successor_development"
        or not set(round24b_rows).issubset(source_rows)
    ):
        raise ValueError("Round-24B source binding is invalid")
    for path, row in round24b_rows.items():
        if (
            row.get("round23_sha256") != source_rows[path].get("sha256")
            or row.get("predecessor_sha256") != effective_before_round24b[path]
            or not row.get("round24b_sha256")
        ):
            raise ValueError(f"Round-24B transition is invalid: {path}")
        effective[path] = str(row["round24b_sha256"])

    current_hashes = {
        path: sha256_file(REPO_ROOT / path) for path in source_rows
    }
    current_source_drift = sorted(
        path
        for path, row in source_rows.items()
        if current_hashes[path] != row.get("sha256")
    )
    residual = sorted(
        path for path in source_rows if current_hashes[path] != effective[path]
    )
    transitions = [
        {
            "path": path,
            "round23_sha256": source_rows[path]["sha256"],
            "predecessor_sha256": effective[path],
            "round25_sha256": current_hashes[path],
            "transition_reason": "round25_static_loop_source_or_validation_extension",
        }
        for path in residual
    ]
    if not transitions:
        raise ValueError("No new source transition requires a Round-25 binding")
    return {
        "schema_version": "neuronsifter-round25-round23-source-successor-binding-v1",
        "status": "bound_successor_development",
        "round23_full_suite_audit_path": "Output/audits/round23_full_suite_execution.json",
        "round23_full_suite_audit_sha256": sha256_file(ROUND23_AUDIT),
        "predecessor_binding": {
            "path": "Output/audits/round24b_round23_source_successor_binding.json",
            "sha256": sha256_file(ROUND24B_BINDING),
        },
        "source_transitions": transitions,
        "current_source_drift": current_source_drift,
        "residual_source_drift_after_predecessors": residual,
        "historical_round23_audit_rewritten": False,
        "historical_round24_binding_rewritten": False,
        "historical_round24b_binding_rewritten": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROUND25_BINDING),
        help="Append-only successor binding path.",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    payload = build_binding()
    if output.exists():
        if load_json(output) != payload:
            raise FileExistsError(f"Refusing to rewrite successor binding: {output}")
        return
    write_json(output, payload)


if __name__ == "__main__":
    main()
