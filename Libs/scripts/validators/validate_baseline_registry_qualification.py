from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_registry_qualification import (  # noqa: E402
    AUDIT_FILES,
    ENVIRONMENT_SPECS,
    QUALIFICATION_STATUSES,
    build_all,
)
from Libs.scripts.builders.build_baseline_registry_audits import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fail-closed baseline qualification audits.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--lock", dest="lock_path", default=DEFAULT_LOCK)
    parser.add_argument("--audit-root", default="Output/audits")
    parser.add_argument("--environment-root", default="Libs/external/environments")
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _serialize(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]


def validate(
    registry: str | Path,
    schema: str | Path,
    lock: str | Path,
    audit_root: str | Path,
    environment_root: str | Path,
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    try:
        expected = build_all(repo_path(registry), repo_path(schema), repo_path(lock))
    except Exception as exc:
        return {
            "status": "failed",
            "blocker_count": 1,
            "blockers": [{"code": "registry_or_builder_invalid", "message": str(exc)}],
        }

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

    env_root = repo_path(environment_root)
    for baseline_id, expected_spec in ENVIRONMENT_SPECS.items():
        path = env_root / f"{baseline_id}.yaml"
        if not path.exists():
            blockers.append({"code": "missing_environment_record", "message": str(path)})
            continue
        actual_spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if actual_spec != expected_spec:
            blockers.append({"code": "stale_or_tampered_environment_record", "message": str(path)})

    source_rows = expected["source_resolution"]
    if any(row["qualification_status"] not in QUALIFICATION_STATUSES for row in source_rows):
        blockers.append({"code": "invalid_qualification_status", "message": "Source audit has an unsupported status."})
    gate = expected["gate"]
    required_zero = (
        "clean_clone_performed_count",
        "upstream_smoke_run_count",
        "native_reproduction_count",
        "matched_training_count",
        "headline_ready_count",
    )
    if any(gate[key] != 0 for key in required_zero):
        blockers.append({"code": "unsupported_execution_promotion", "message": "Baseline qualification must preserve zero execution counts."})
    if gate["round11_closure"] != "blocked_full_suite":
        blockers.append({"code": "round11_boundary_changed", "message": "Baseline qualification is bounded by blocked Round-11 closure."})
    if gate["status"] != "blocked_source_or_license":
        blockers.append({"code": "invalid_gate_status", "message": str(gate["status"])})

    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "gate_status": gate["status"],
        "external_source_required_count": gate["external_source_required_count"],
        "upstream_smoke_run_count": gate["upstream_smoke_run_count"],
    }


def main() -> None:
    args = parse_args()
    report = validate(args.registry, args.schema, args.lock_path, args.audit_root, args.environment_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
