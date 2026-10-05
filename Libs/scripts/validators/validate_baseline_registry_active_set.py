from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_registry_active_set_audits import (  # noqa: E402
    AUDIT_FILES,
    DEFAULT_LOCK,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    build_all,
    repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Baseline Registry V3.1 active-set audits.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--lock", dest="lock_path", default=DEFAULT_LOCK)
    parser.add_argument("--audit-root", default="Output/audits")
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _serialize(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]


def validate(registry: str | Path, schema: str | Path, lock: str | Path, audit_root: str | Path) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    try:
        expected = build_all(repo_path(registry), repo_path(schema), repo_path(lock))
    except Exception as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "registry_or_schema_invalid", "message": str(exc)}]}
    root = repo_path(audit_root)
    for key, filename in AUDIT_FILES.items():
        path = root / filename
        if not path.exists():
            blockers.append({"code": "missing_required_audit", "message": filename})
            continue
        actual = _read_tsv(path) if path.suffix == ".tsv" else json.loads(path.read_text(encoding="utf-8"))
        expected_value = _serialize(expected[key]) if path.suffix == ".tsv" else expected[key]
        if actual != expected_value:
            blockers.append({"code": "stale_or_tampered_audit", "message": filename})
    gate = expected["gate"]
    if gate["headline_ready_count"] != 0:
        blockers.append({"code": "unsupported_headline_promotion", "message": "V3.1 active-set reduction cannot promote headline results."})
    if any(gate[name] for name in ("formal_matched_training_allowed", "headline_comparison_allowed", "evidence_fabric_sealing_allowed", "DT_CORE_allowed")):
        blockers.append({"code": "frozen_boundary_violation", "message": "A prohibited execution flag was enabled."})
    if gate["main_baseline_count"] != 10 or gate["main_ablation_count"] != 5:
        blockers.append({"code": "active_count_mismatch", "message": "Expected 10 main baselines and 5 main ablations."})
    if gate["status"] != "active_set_frozen":
        blockers.append({"code": gate["status"], "message": "V3.1 active-set gate is not frozen."})
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "gate_status": gate["status"],
        "main_baseline_count": gate["main_baseline_count"],
        "main_ablation_count": gate["main_ablation_count"],
    }


def main() -> None:
    args = parse_args()
    report = validate(args.registry, args.schema, args.lock_path, args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
