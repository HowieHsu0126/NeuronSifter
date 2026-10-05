"""Record an append-only successor for source drift after the real-domain binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
ROUND23_AUDIT = REPO_ROOT / "Output/audits/round23_full_suite_execution.json"
ROUND24_BINDING = REPO_ROOT / "Output/audits/round24_round23_source_successor_binding.json"
ROUND24B_BINDING = REPO_ROOT / "Output/audits/round24b_round23_source_successor_binding.json"
ROUND25_BINDING = REPO_ROOT / "Output/audits/round25_round23_source_successor_binding.json"
PREDECESSOR_BINDING = REPO_ROOT / (
    "Output/audits/real_intervention_round23_source_successor_binding_v3.json"
)
OUTPUT_PATH = REPO_ROOT / "Output/audits/round23_current_source_successor_binding_v4.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _rows(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(row.get("path", "")): row
        for row in payload.get("source_transitions", [])
        if isinstance(row, Mapping)
    }


def _apply_binding(
    *,
    audit: Mapping[str, Mapping[str, object]],
    effective: dict[str, str],
    path: Path,
    schema_version: str,
    successor_key: str,
    predecessor_path: Path | None,
) -> None:
    binding = load_json(path)
    rows = _rows(binding)
    if (
        binding.get("schema_version") != schema_version
        or binding.get("status") != "bound_successor_development"
        or binding.get("round23_full_suite_audit_sha256") != sha256_file(ROUND23_AUDIT)
        or not set(rows).issubset(audit)
        or binding.get("publication_claim_eligible") is not False
        or binding.get("protected_test_access") != 0
    ):
        raise ValueError(f"Invalid historical source binding: {path}")
    if predecessor_path is not None:
        declared = binding.get("predecessor_binding", {})
        if (
            not isinstance(declared, Mapping)
            or declared.get("path") != predecessor_path.relative_to(REPO_ROOT).as_posix()
            or declared.get("sha256") != sha256_file(predecessor_path)
        ):
            raise ValueError(f"Invalid predecessor binding: {path}")
    for relative, row in rows.items():
        predecessor_sha256 = str(row.get("predecessor_sha256", row.get("round23_sha256", "")))
        if (
            row.get("round23_sha256") != audit[relative].get("sha256")
            or predecessor_sha256 != effective[relative]
            or not row.get(successor_key)
        ):
            raise ValueError(f"Invalid source transition: {relative}")
        effective[relative] = str(row[successor_key])


def build_binding() -> dict[str, object]:
    audit = load_json(ROUND23_AUDIT)
    predecessor = load_json(PREDECESSOR_BINDING)
    source_rows = _rows({"source_transitions": audit.get("source_files", [])})
    if audit.get("schema_version") != "neuronsifter-round23-full-suite-execution-v1":
        raise ValueError("Round-23 full-suite audit schema is invalid")
    if (
        predecessor.get("schema_version")
        != "neuronsifter-real-intervention-round23-source-successor-binding-v3"
        or predecessor.get("status") != "bound_successor_development"
        or predecessor.get("round23_full_suite_audit_sha256") != sha256_file(ROUND23_AUDIT)
        or predecessor.get("publication_claim_eligible") is not False
        or predecessor.get("protected_test_access") != 0
    ):
        raise ValueError("Real-intervention predecessor binding is invalid")
    declared_predecessor = predecessor.get("predecessor_binding", {})
    if not isinstance(declared_predecessor, Mapping) or not declared_predecessor.get("path"):
        raise ValueError("Real-intervention predecessor chain is incomplete")

    effective = {relative: str(row.get("sha256", "")) for relative, row in source_rows.items()}
    _apply_binding(
        audit=source_rows,
        effective=effective,
        path=ROUND24_BINDING,
        schema_version="neuronsifter-round24-round23-source-successor-binding-v1",
        successor_key="round24_sha256",
        predecessor_path=None,
    )
    _apply_binding(
        audit=source_rows,
        effective=effective,
        path=ROUND24B_BINDING,
        schema_version="neuronsifter-round24b-round23-source-successor-binding-v1",
        successor_key="round24b_sha256",
        predecessor_path=ROUND24_BINDING,
    )
    _apply_binding(
        audit=source_rows,
        effective=effective,
        path=ROUND25_BINDING,
        schema_version="neuronsifter-round25-round23-source-successor-binding-v1",
        successor_key="round25_sha256",
        predecessor_path=ROUND24B_BINDING,
    )
    _apply_binding(
        audit=source_rows,
        effective=effective,
        path=PREDECESSOR_BINDING,
        schema_version="neuronsifter-real-intervention-round23-source-successor-binding-v3",
        successor_key="real_intervention_sha256",
        predecessor_path=ROUND25_BINDING,
    )

    current_hashes = {relative: sha256_file(REPO_ROOT / relative) for relative in source_rows}
    current_source_drift = sorted(
        relative
        for relative, row in source_rows.items()
        if current_hashes[relative] != row.get("sha256")
    )
    residual = sorted(
        relative for relative in source_rows if current_hashes[relative] != effective[relative]
    )
    if not residual:
        raise ValueError("No current source transition requires binding")
    return {
        "schema_version": "neuronsifter-round23-current-source-successor-binding-v4",
        "status": "bound_successor_development",
        "round23_full_suite_audit_path": ROUND23_AUDIT.relative_to(REPO_ROOT).as_posix(),
        "round23_full_suite_audit_sha256": sha256_file(ROUND23_AUDIT),
        "predecessor_binding": {
            "path": PREDECESSOR_BINDING.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(PREDECESSOR_BINDING),
        },
        "source_transitions": [
            {
                "path": relative,
                "round23_sha256": source_rows[relative]["sha256"],
                "predecessor_sha256": effective[relative],
                "current_sha256": current_hashes[relative],
                "transition_reason": "targeted_therapeutic_wp3_provenance_repair",
            }
            for relative in residual
        ],
        "current_source_drift": current_source_drift,
        "residual_source_drift_after_predecessor": residual,
        "historical_round23_audit_rewritten": False,
        "historical_real_intervention_binding_rewritten": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(REPO_ROOT.resolve())
    payload = build_binding()
    if output.exists():
        if load_json(output) != payload:
            raise FileExistsError(f"Refusing to rewrite successor binding: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
