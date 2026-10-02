"""Validate the append-only Round-20Q-R minimal challenge audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "Output/audits/round20qr_minimal_challenge_v1"
REQUIRED_SOURCE_COUNT = 20
REQUIRED_SIMPLE_UIDS = {"persistence", "linear_state_space", "gru"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    value = {key: item for key, item in payload.items() if key != "artifact_uid"}
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    manifest_path = root / "round20qr_minimal_challenge_root_manifest.json"
    gate_path = root / "round20qr_minimal_challenge_gate.json"
    source_path = root / "mandatory_source_catalog.tsv"
    results_path = root / "simple_adapter_results.tsv"
    failure_path = root / "failure_catalog.tsv"
    c2_path = root / "c2_baseline_qualification.json"
    required = (
        manifest_path,
        gate_path,
        source_path,
        results_path,
        failure_path,
        c2_path,
    )
    for path in required:
        if not path.is_file():
            blockers.append(
                {
                    "code": "missing_artifact",
                    "message": str(path.relative_to(REPO_ROOT)),
                }
            )
    if blockers:
        return {
            "status": "failed",
            "blocker_count": len(blockers),
            "blockers": blockers,
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        c2 = json.loads(c2_path.read_text(encoding="utf-8"))
        source_rows = _read_tsv(source_path)
        result_rows = _read_tsv(results_path)
        failure_rows = _read_tsv(failure_path)
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        return {
            "status": "failed",
            "blocker_count": 1,
            "blockers": [{"code": "invalid_artifact", "message": str(exc)}],
        }

    if manifest.get("schema_version") != "neuronsifter-round20qr-minimal-challenge-v1":
        blockers.append(
            {
                "code": "schema_version",
                "message": "root manifest schema version is not bound to the minimal challenge validator",
            }
        )
    if manifest.get("lineage_uid") != root.name:
        blockers.append(
            {
                "code": "lineage_uid",
                "message": "root manifest lineage UID must match its append-only output directory",
            }
        )
    if manifest.get("artifact_uid") != _canonical_sha256(manifest):
        blockers.append(
            {
                "code": "artifact_uid",
                "message": "root manifest artifact UID does not match canonical content",
            }
        )
    if gate.get("status") != "blocked_round20qr_baseline_challenge":
        blockers.append(
            {
                "code": "gate_status",
                "message": "minimal challenge must remain fail-closed until formal prerequisites exist",
            }
        )
    for key in (
        "round20qr_baseline_challenge_resolved",
        "simple_baseline_sanity_passed",
        "strong_baseline_results_available",
        "challenge_thresholds_frozen",
        "formal_challenge_started",
        "headline_training_authorized",
    ):
        if gate.get(key) is not False:
            blockers.append(
                {"code": "gate_fail_open", "message": f"{key} must remain false"}
            )
    if gate.get("test_data_unseen") is not True:
        blockers.append(
            {
                "code": "test_access",
                "message": "minimal challenge must prove that final test data was not used",
            }
        )
    if gate.get("c2b_status") != "blocked_insufficient_real_e2_e4":
        blockers.append({"code": "c2b_status", "message": "C2-B status drifted"})
    if len(source_rows) != REQUIRED_SOURCE_COUNT:
        blockers.append(
            {
                "code": "source_count",
                "message": f"expected {REQUIRED_SOURCE_COUNT} source rows, found {len(source_rows)}",
            }
        )
    source_uids = [row.get("baseline_uid", "") for row in source_rows]
    if len(source_uids) != len(set(source_uids)):
        blockers.append(
            {
                "code": "source_uid_duplicate",
                "message": "mandatory source catalog contains duplicate baseline UIDs",
            }
        )
    if any(row.get("formal_challenge_eligible") != "False" for row in source_rows):
        blockers.append(
            {
                "code": "source_fail_open",
                "message": "source rows cannot be formal challenge eligible in the restricted route",
            }
        )
    result_uids = {row.get("baseline_uid", "") for row in result_rows}
    if result_uids != REQUIRED_SIMPLE_UIDS:
        blockers.append(
            {
                "code": "simple_adapter_set",
                "message": f"expected simple adapters {sorted(REQUIRED_SIMPLE_UIDS)}, found {sorted(result_uids)}",
            }
        )
    if any(
        row.get("final_holdout_accessed") != "False"
        or row.get("formal_challenge_eligible") != "False"
        for row in result_rows
    ):
        blockers.append(
            {
                "code": "result_fail_open",
                "message": "simple adapter results must be validation-only and non-formal",
            }
        )
    if c2.get("status") != "passed" or c2.get("formal_challenge_started") is not False:
        blockers.append(
            {
                "code": "c2_qualification",
                "message": "restricted C2 qualification is not a clean smoke result",
            }
        )
    split_access = c2.get("split_access", {})
    if (
        split_access.get("final_holdout_prediction_materialized") is not False
        or split_access.get("final_holdout_metrics_computed") is not False
    ):
        blockers.append(
            {
                "code": "c2_holdout_access",
                "message": "restricted C2 output reports final-holdout access",
            }
        )
    if manifest.get("c2_qualification", {}).get("sha256") != _sha256(c2_path):
        blockers.append(
            {"code": "c2_binding", "message": "root manifest C2 output hash is stale"}
        )
    if not failure_rows:
        blockers.append(
            {
                "code": "failure_catalog",
                "message": "blocked source/data reasons must be preserved",
            }
        )
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "source_rows": len(source_rows),
        "simple_result_rows": len(result_rows),
        "failure_rows": len(failure_rows),
        "formal_challenge_resolved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the fail-closed Round-20Q-R minimal challenge audit."
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    report = validate(Path(args.root).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
