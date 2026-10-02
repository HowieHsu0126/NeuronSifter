from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_registry_audits import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    build_all,
    repo_path,
)


AUDIT_FILES = {
    "inventory": "baseline_existing_inventory.json",
    "duplicates": "baseline_duplicate_implementation_audit.json",
    "sources": "baseline_source_audit.tsv",
    "clones": "baseline_clone_manifest.tsv",
    "licenses": "baseline_license_audit.tsv",
    "native": "baseline_native_reproduction_status.tsv",
    "adapters": "baseline_adapter_compatibility.tsv",
    "fairness": "baseline_fairness_matrix.tsv",
    "gate": "baseline_reproduction_gate.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Baseline Registry V3 and generated audits.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--lock", dest="lock_path", default=DEFAULT_LOCK)
    parser.add_argument("--audit-root", default="Output/audits")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tsv_serialized(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]


def validate(registry: str | Path, schema: str | Path, lock_path: str | Path, audit_root: str | Path) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    try:
        expected = build_all(repo_path(registry), repo_path(schema), repo_path(lock_path))
    except Exception as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "registry_or_lock_invalid", "message": str(exc)}]}
    root = repo_path(audit_root)
    for key, filename in AUDIT_FILES.items():
        path = root / filename
        if not path.exists():
            blockers.append({"code": "missing_required_audit", "message": filename})
            continue
        actual = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else read_tsv(path)
        expected_value = expected[key] if path.suffix == ".json" else tsv_serialized(expected[key])
        if actual != expected_value:
            blockers.append({"code": "stale_or_tampered_audit", "message": filename})
    gate = expected["gate"]
    if gate["entry_gate"] not in {"registry_complete", "source_audit_complete", "clone_and_pin_complete", "upstream_smoke_complete", "reproduction_program_ready", "blocked_source_or_license", "blocked_round11_full_suite"}:
        blockers.append({"code": "invalid_entry_gate", "message": str(gate["entry_gate"])})
    if gate["headline_ready_count"] != 0:
        blockers.append({"code": "unsupported_headline_promotion", "message": "Round-11 is blocked and matched protocol evidence is absent."})
    if any((gate[name] for name in ("formal_matched_training_allowed", "headline_comparison_allowed", "evidence_fabric_sealing_allowed", "DT_CORE_allowed"))):
        blockers.append({"code": "frozen_boundary_violation", "message": "A prohibited execution flag was enabled."})
    return {"status": "passed" if not blockers else "failed", "blocker_count": len(blockers), "blockers": blockers, "entry_gate": gate["entry_gate"], "headline_ready_count": gate["headline_ready_count"]}


def main() -> None:
    args = parse_args()
    report = validate(args.registry, args.schema, args.lock_path, args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
