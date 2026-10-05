"""Validate append-only Round20QR v3 source-qualification evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
C2B_BLOCKED = "blocked_insufficient_real_e2_e4"
BLOCKED_STATUS = "blocked_round20qr_v3_source_qualification"
REQUIRED = (
    "source_qualification.tsv",
    "adapter_qualification.tsv",
    "environment_lock.json",
    "source_pin_manifest.json",
    "producer_identity.json",
    "gate.json",
    "root_manifest.json",
)
ROOT_BINDINGS = tuple(name for name in REQUIRED if name != "root_manifest.json")
SOURCE_MANIFEST_FIELDS = (
    "baseline_uid",
    "source_kind",
    "declared_source",
    "observed_path",
    "version_or_commit",
    "tree_or_record_sha256",
    "license_sha256",
    "record_integrity_status",
    "record_hash_entry_count",
    "record_verified_entry_count",
    "module_spec_origin",
    "module_spec_origin_status",
    "import_origin",
    "import_origin_status",
    "official_artifact_status",
    "official_artifact_path",
    "official_artifact_sha256",
    "official_artifact_url",
    "source_identity_status",
    "upstream_smoke_status",
    "upstream_smoke_detail",
    "source_identity_known",
    "version_pin_known",
    "license_decision_known",
    "adapter_status",
    "formal_challenge_eligible",
    "reason",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "artifact_uid"}
    return hashlib.sha256(
        json.dumps(
            canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _tsv_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    return str(value if value is not None else "")


def validate(root: Path, registry: Path) -> dict[str, Any]:
    blockers: list[str] = []
    paths = {name: root / name for name in REQUIRED}
    for name, path in paths.items():
        if not path.is_file():
            blockers.append(f"missing:{name}")
    if blockers:
        return {
            "status": "failed",
            "blockers": blockers,
            "blocker_count": len(blockers),
        }
    try:
        with paths["source_qualification.tsv"].open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        with paths["adapter_qualification.tsv"].open(
            encoding="utf-8", newline=""
        ) as handle:
            adapters = list(csv.DictReader(handle, delimiter="\t"))
        source = json.loads(
            paths["source_pin_manifest.json"].read_text(encoding="utf-8")
        )
        environment = json.loads(
            paths["environment_lock.json"].read_text(encoding="utf-8")
        )
        producer = json.loads(
            paths["producer_identity.json"].read_text(encoding="utf-8")
        )
        gate = json.loads(paths["gate.json"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["root_manifest.json"].read_text(encoding="utf-8"))
        registry_rows = list(
            yaml.safe_load(registry.read_text(encoding="utf-8"))["reduction_registry"][
                "mandatory"
            ]
        )
    except (
        OSError,
        ValueError,
        KeyError,
        csv.Error,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        return {"status": "failed", "blockers": [f"parse:{exc}"], "blocker_count": 1}
    expected_uids = {str(row["baseline_uid"]) for row in registry_rows}
    source_uids = {row.get("baseline_uid", "") for row in rows}
    if source_uids != expected_uids or len(rows) != 20:
        blockers.append("source_rows_registry_mismatch")
    if {row.get("baseline_uid", "") for row in adapters} != expected_uids:
        blockers.append("adapter_rows_registry_mismatch")
    if any(row.get("formal_challenge_eligible") != "False" for row in rows):
        blockers.append("source_row_fail_open")
    if any(row.get("formal_challenge_eligible") != "False" for row in adapters):
        blockers.append("adapter_row_fail_open")
    if source.get("registry_sha256") != _sha256(registry):
        blockers.append("registry_hash_mismatch")
    if source.get("environment") != environment:
        blockers.append("environment_lock_source_manifest_mismatch")
    artifact_pins = source.get("package_artifact_pin_manifest")
    if not isinstance(artifact_pins, dict) or artifact_pins.get("status") not in {
        "not_supplied",
        "loaded",
    }:
        blockers.append("package_artifact_pin_manifest_invalid")
    elif artifact_pins.get("status") == "loaded":
        pin_records = artifact_pins.get("artifact_pins")
        if not isinstance(pin_records, list) or not pin_records:
            blockers.append("package_artifact_pin_manifest_missing_records")
        elif (
            any(
                not isinstance(pin, dict)
                or pin.get("authority_manifest_bound") is not True
                for pin in pin_records
            )
            or len({str(pin.get("baseline_uid", "")) for pin in pin_records})
            != len(pin_records)
        ):
            blockers.append("package_artifact_pin_manifest_records_invalid")
    manifest_rows = source.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(rows):
        blockers.append("source_manifest_rows_missing")
    else:
        by_uid = {str(row.get("baseline_uid", "")): row for row in manifest_rows}
        for row in rows:
            expected = by_uid.get(str(row.get("baseline_uid", "")))
            if expected is None:
                blockers.append("source_manifest_tsv_uid_mismatch")
                break
            if any(
                _tsv_scalar(expected.get(field, "")) != row.get(field, "")
                for field in SOURCE_MANIFEST_FIELDS
            ):
                blockers.append("source_manifest_tsv_row_mismatch")
                break
    if (
        producer.get("git_bound") is not True
        or producer.get("worktree_clean") is not True
        or producer.get("tracked_clean") is not True
        or producer.get("staged_clean") is not True
        or producer.get("untracked_check_succeeded") is not True
        or producer.get("untracked_file_count") != 0
        or not producer.get("commit")
        or not producer.get("tree")
    ):
        blockers.append("producer_not_clean_git_bound")
    if gate.get("status") != BLOCKED_STATUS:
        blockers.append("gate_status_drift")
    for key in (
        "formal_challenge_eligible",
        "simple_baseline_sanity_passed",
        "strong_baseline_results_available",
        "challenge_thresholds_frozen",
        "round21r_replay_authorized",
        "c3_c4_execution_allowed",
        "coevolution_execution_allowed",
    ):
        if gate.get(key) is not False:
            blockers.append(f"gate_fail_open:{key}")
    if gate.get("c2b_status") != C2B_BLOCKED:
        blockers.append("c2b_status_drift")
    if gate.get("source_identity_known") != sum(
        row.get("source_identity_known") == "True" for row in rows
    ):
        blockers.append("source_identity_count_mismatch")
    if gate.get("version_pin_known") != sum(
        row.get("version_pin_known") == "True" for row in rows
    ):
        blockers.append("version_pin_count_mismatch")
    if gate.get("license_decision_known") != sum(
        row.get("license_decision_known") == "True" for row in rows
    ):
        blockers.append("license_count_mismatch")
    if manifest.get("artifact_uid") != _canonical_sha256(manifest):
        blockers.append("root_uid_mismatch")
    if manifest.get("status") != BLOCKED_STATUS or manifest.get("status") != gate.get(
        "status"
    ):
        blockers.append("root_status_mismatch")
    if manifest.get("c2b_status") != C2B_BLOCKED:
        blockers.append("root_c2b_drift")
    if manifest.get("formal_challenge_eligible") is not False:
        blockers.append("root_fail_open")
    bindings = manifest.get("files")
    if not isinstance(bindings, dict) or set(bindings) != set(ROOT_BINDINGS):
        blockers.append("root_binding_set_mismatch")
    else:
        for name in ROOT_BINDINGS:
            binding = bindings[name]
            path = root / str(binding.get("path", ""))
            if binding.get("path") != name or not path.is_file() or binding.get(
                "sha256"
            ) != _sha256(path):
                blockers.append("root_binding_mismatch")
                break
    for row in rows:
        if row.get("source_identity_status") != "package_identity_bound":
            continue
        if (
            row.get("source_kind") != "python_distribution"
            or row.get("record_integrity_status") != "verified"
            or row.get("module_spec_origin_status") != "verified_record_entry"
            or row.get("official_artifact_status") != "verified_wheel_binding"
            or row.get("import_origin_status") not in {
                "not_imported",
                "verified_record_entry",
            }
        ):
            blockers.append("package_identity_binding_incomplete")
            break
    return {
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "blocker_count": len(blockers),
        "source_rows": len(rows),
        "formal_challenge_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--registry",
        default=str(REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml"),
    )
    args = parser.parse_args()
    report = validate(Path(args.root).resolve(), Path(args.registry).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
