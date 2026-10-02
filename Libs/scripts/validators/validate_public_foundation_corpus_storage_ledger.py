"""Validate a completed or interrupted Public Foundation Corpus storage scan."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator, FormatChecker


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_storage import (
    derive_storage_scan_status,
    safe_repo_path,
    scan_storage_roots,
)
from Libs.utils.data.raw_data_registry import load_config, write_json


DEFAULT_CONFIG = "Libs/configs/config_stack.yaml"
DEFAULT_LEDGER = "Output/audits/public_foundation_corpus_storage_ledger.json"
DEFAULT_SCHEMA = "Libs/configs/schema/public_foundation_corpus_storage_ledger.schema.json"
DEFAULT_OUTPUT = "Output/audits/public_foundation_corpus_storage_validation.json"


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def validate_ledger_payload(
    payload: object,
    *,
    storage_policy: Mapping[str, object],
    schema: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: (list(error.absolute_path), error.message))
    for error in errors:
        blockers.append(
            {
                "code": "json_schema_violation",
                "path": ".".join(map(str, error.absolute_path)) or "$",
                "message": error.message,
            }
        )
    if errors or not isinstance(payload, dict):
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": blockers}

    roots = payload["roots"]
    expected_roots = storage_policy["scan_root_paths"]
    expected_exclusions = sorted(storage_policy["scan_excluded_paths"])
    if roots.keys() != expected_roots.keys():
        blockers.append({"code": "storage_root_set_mismatch", "path": "roots", "message": "Ledger roots differ from the frozen storage policy."})
    for root_id, relative in expected_roots.items():
        root = roots.get(root_id, {})
        if root.get("root") != relative:
            blockers.append({"code": "storage_root_path_mismatch", "path": f"roots.{root_id}.root", "message": "Ledger root path differs from policy."})
    if payload["scope_included"] != list(expected_roots.values()):
        blockers.append({"code": "storage_scope_included_mismatch", "path": "scope_included", "message": "Included roots differ from policy."})
    if payload["scope_excluded"] != expected_exclusions or payload["measurement_exclusions"] != expected_exclusions:
        blockers.append({"code": "storage_scope_excluded_mismatch", "path": "scope_excluded", "message": "Excluded paths differ from policy."})
    if payload["capacity_limit_bytes"] != storage_policy["decimal_byte_cap"] or payload["decimal_byte_cap"] != storage_policy["decimal_byte_cap"]:
        blockers.append({"code": "storage_capacity_limit_mismatch", "path": "capacity_limit_bytes", "message": "Ledger capacity limit differs from policy."})
    if payload["previous_confirmed_usage_bytes"] != storage_policy["previous_confirmed_usage_bytes"]:
        blockers.append({"code": "storage_previous_usage_mismatch", "path": "previous_confirmed_usage_bytes", "message": "Ledger baseline differs from policy."})
    if payload["record_count"] != len(expected_roots):
        blockers.append({"code": "storage_record_count_mismatch", "path": "record_count", "message": "Ledger root record count is invalid."})

    root_bytes = sum(int(row["bytes"]) for row in roots.values())
    root_files = sum(int(row["file_count"]) for row in roots.values())
    if payload["usage_bytes"] != root_bytes:
        blockers.append({"code": "storage_usage_sum_mismatch", "path": "usage_bytes", "message": "Usage bytes do not equal the sum of root measurements."})
    if payload["total_file_count"] != root_files:
        blockers.append({"code": "storage_file_count_sum_mismatch", "path": "total_file_count", "message": "File count does not equal the sum of root measurements."})
    errors_present = bool(payload["rpc_or_io_errors"])
    expected_status = derive_storage_scan_status(
        usage_bytes=int(payload["usage_bytes"]),
        capacity_limit_bytes=int(payload["capacity_limit_bytes"]),
        partial_scan=bool(payload["partial_scan"]),
        rpc_or_io_errors=payload["rpc_or_io_errors"],
    )
    if bool(payload["partial_scan"]) != errors_present:
        blockers.append({"code": "storage_partial_scan_mismatch", "path": "partial_scan", "message": "Partial scan must exactly reflect captured I/O errors."})
    if payload["scan_status"] != expected_status or payload["status"] != expected_status:
        blockers.append({"code": "storage_status_derivation_mismatch", "path": "scan_status", "message": "Storage outcome is not derived from completeness and capacity."})
    expected_cap_status = {
        "passed": "passed",
        "failed_capacity": "not_passed",
        "pending_revalidation": "pending_revalidation",
    }[expected_status]
    if payload["cap_status"] != expected_cap_status or bool(payload["cap_exceeded"]) != (payload["usage_bytes"] > payload["capacity_limit_bytes"]):
        blockers.append({"code": "storage_cap_derivation_mismatch", "path": "cap_status", "message": "Capacity summary is not derived from the scan."})
    expected_blockers = {
        "passed": [],
        "failed_capacity": ["storage_cap_exceeded"],
        "pending_revalidation": ["storage_cap_pending_revalidation"],
    }[expected_status]
    if payload["blocker_codes"] != expected_blockers:
        blockers.append({"code": "storage_blocker_derivation_mismatch", "path": "blocker_codes", "message": "Storage blockers are not derived from the scan state."})
    if payload["validator_status"] != "passed":
        blockers.append({"code": "storage_self_validation_not_passed", "path": "validator_status", "message": "Ledger producer did not record successful self-validation."})
    if sorted({str(row.get("path", "")) for row in payload["rpc_or_io_errors"] if str(row.get("path", ""))}) != payload["unreadable_paths"]:
        blockers.append({"code": "storage_unreadable_path_mismatch", "path": "unreadable_paths", "message": "Unreadable paths must match captured I/O errors."})
    if expected_status == "pending_revalidation":
        if payload["remaining_bytes"] is not None or payload["usage_ratio"] is not None or payload["delta_from_previous"] is not None:
            blockers.append({"code": "storage_partial_measurement_extrapolated", "path": "usage_bytes", "message": "Incomplete scans must not extrapolate remaining capacity, ratio, or delta."})
    else:
        if payload["remaining_bytes"] != payload["capacity_limit_bytes"] - payload["usage_bytes"]:
            blockers.append({"code": "storage_remaining_bytes_mismatch", "path": "remaining_bytes", "message": "Remaining bytes are not derived from usage and capacity."})
        expected_ratio = payload["usage_bytes"] / payload["capacity_limit_bytes"]
        if abs(float(payload["usage_ratio"]) - expected_ratio) > 1e-15:
            blockers.append({"code": "storage_usage_ratio_mismatch", "path": "usage_ratio", "message": "Usage ratio is not derived from usage and capacity."})
        if payload["delta_from_previous"] != payload["usage_bytes"] - payload["previous_confirmed_usage_bytes"]:
            blockers.append({"code": "storage_delta_mismatch", "path": "delta_from_previous", "message": "Delta does not match the frozen baseline."})
    try:
        started = datetime.fromisoformat(payload["scan_started_at"])
        completed = datetime.fromisoformat(payload["scan_completed_at"])
        if completed < started:
            blockers.append({"code": "storage_scan_time_order_invalid", "path": "scan_completed_at", "message": "Scan completion precedes scan start."})
    except ValueError:
        blockers.append({"code": "storage_scan_time_invalid", "path": "scan_started_at", "message": "Scan timestamps must be ISO-8601 datetimes."})

    return {
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "storage_gate_status": payload.get("scan_status"),
        "usage_bytes": payload.get("usage_bytes"),
        "capacity_limit_bytes": payload.get("capacity_limit_bytes"),
        "partial_scan": payload.get("partial_scan"),
    }


def build_report(
    ledger_path: str = DEFAULT_LEDGER,
    *,
    repo_root: Path = REPO_ROOT,
    config_path: str = DEFAULT_CONFIG,
    schema_path: str = DEFAULT_SCHEMA,
    verify_current_scan: bool = False,
) -> dict[str, object]:
    config = load_config(safe_repo_path(repo_root, config_path))
    policy = config["public_foundation_corpus"]["storage_policy"]
    payload = _load_json(safe_repo_path(repo_root, ledger_path))
    schema = _load_json(safe_repo_path(repo_root, schema_path))
    if not isinstance(schema, dict):
        raise TypeError("Storage ledger schema must be a JSON object")
    Draft202012Validator.check_schema(schema)
    report = validate_ledger_payload(payload, storage_policy=policy, schema=schema)
    if verify_current_scan and isinstance(payload, dict):
        current = scan_storage_roots(
            repo_root,
            roots=policy["scan_root_paths"],
            cap_bytes=int(policy["decimal_byte_cap"]),
            exclude_relative_paths=policy["scan_excluded_paths"],
            previous_confirmed_usage_bytes=int(policy["previous_confirmed_usage_bytes"]),
            largest_directory_limit=int(policy["largest_directory_limit"]),
        )
        report["current_scan"] = {
            key: current[key]
            for key in ("scan_status", "usage_bytes", "total_file_count", "partial_scan")
        }
        expected = {
            key: payload.get(key)
            for key in ("scan_status", "usage_bytes", "total_file_count", "partial_scan")
        }
        if report["current_scan"] != expected:
            report["blockers"].append(
                {
                    "code": "storage_current_scan_mismatch",
                    "path": "current_scan",
                    "message": "A fresh full scan differs from the ledger measurement.",
                }
            )
            report["blockers"] = [{key: str(value) for key, value in row.items()} for row in report["blockers"]]
            report["blocker_count"] = len(report["blockers"])
            report["status"] = "blocked"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Public Foundation Corpus storage accounting.")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-current-scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.ledger,
        repo_root=REPO_ROOT,
        config_path=args.config,
        schema_path=args.schema,
        verify_current_scan=args.verify_current_scan,
    )
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
