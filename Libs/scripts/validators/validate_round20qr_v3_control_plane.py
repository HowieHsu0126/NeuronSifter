"""Validate the blocked Round-20QR v3 control-plane audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_BASELINES = 20
EXPECTED_C2B_STATUS = "blocked_insufficient_real_e2_e4"
EXPECTED_STATUS = "blocked_round20qr_v3_source_qualification"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "Libs/configs/experiments/baseline_registry.yaml"
IDENTITY_FIELDS = ("claim_ids", "baseline_class", "repository_or_package")


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


def _claim_ids_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return ",".join(item.strip() for item in str(value).split(",") if item.strip())


def _registry_identity(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = list(registry.get("reduction_registry", {}).get("mandatory", []))
    if len(entries) != EXPECTED_BASELINES:
        raise ValueError(f"registry_count:{len(entries)}")
    identity = {
        str(entry.get("baseline_uid", "")): {
            "claim_ids": _claim_ids_text(entry.get("claim_ids", [])),
            "baseline_class": str(entry.get("baseline_class", "")).strip(),
            "repository_or_package": str(
                entry.get("repository_or_package", "")
            ).strip(),
        }
        for entry in entries
    }
    if len(identity) != EXPECTED_BASELINES or "" in identity:
        raise ValueError("registry_duplicate_or_empty_uid")
    return identity, _sha256(path)


def validate(root: Path, registry_path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    blockers: list[str] = []
    required = (
        "producer_identity.json",
        "runtime_dependency_manifest.json",
        "network_route_manifest.json",
        "source_access_preflight.json",
        "mandatory_source_matrix.tsv",
        "gate.json",
        "root_manifest.json",
        "round21r_replay_authorization.json",
    )
    paths = {name: root / name for name in required}
    for name, path in paths.items():
        if not path.is_file():
            blockers.append(f"missing_artifact:{name}")
    if blockers:
        return {
            "status": "failed",
            "blocker_count": len(blockers),
            "blockers": blockers,
        }
    try:
        producer = json.loads(
            paths["producer_identity.json"].read_text(encoding="utf-8")
        )
        source_access = json.loads(
            paths["source_access_preflight.json"].read_text(encoding="utf-8")
        )
        gate = json.loads(paths["gate.json"].read_text(encoding="utf-8"))
        root_manifest = json.loads(
            paths["root_manifest.json"].read_text(encoding="utf-8")
        )
        replay = json.loads(
            paths["round21r_replay_authorization.json"].read_text(encoding="utf-8")
        )
        expected_identity, registry_sha256 = _registry_identity(registry_path)
        with paths["mandatory_source_matrix.tsv"].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (
        OSError,
        ValueError,
        yaml.YAMLError,
        json.JSONDecodeError,
        csv.Error,
    ) as exc:
        return {
            "status": "failed",
            "blocker_count": 1,
            "blockers": [f"parse_error:{exc}"],
        }

    if (
        producer.get("git_bound") is not True
        or producer.get("worktree_clean") is not True
    ):
        blockers.append("producer_not_clean_git_bound")
    if producer.get("dirty_patch_required") is not False:
        blockers.append("dirty_patch_required")
    if producer.get("unrelated_user_files_modified") is not False:
        blockers.append("producer_unrelated_files_modified")
    if producer.get("untracked_check_succeeded") is not True:
        blockers.append("producer_untracked_check_missing")
    if producer.get("untracked_file_count") != 0:
        blockers.append("producer_untracked_files")
    registry_input = producer.get("inputs", {}).get("baseline_registry", {})
    if registry_input.get("sha256") != registry_sha256:
        blockers.append("frozen_registry_hash_mismatch")
    if len(rows) != EXPECTED_BASELINES:
        blockers.append(f"source_matrix_count:{len(rows)}")
    row_by_uid = {row.get("baseline_uid", ""): row for row in rows}
    if len(row_by_uid) != len(rows):
        blockers.append("source_matrix_duplicate_uid")
    if set(row_by_uid) != set(expected_identity):
        blockers.append("source_matrix_registry_uid_mismatch")
    for uid in sorted(set(row_by_uid).intersection(expected_identity)):
        for field in IDENTITY_FIELDS:
            if row_by_uid[uid].get(field, "").strip() != expected_identity[uid][field]:
                blockers.append(f"source_matrix_registry_identity:{uid}:{field}")
    if any(row.get("formal_challenge_eligible") != "False" for row in rows):
        blockers.append("source_matrix_fail_open")
    if (
        source_access.get("network_failures_distinguished_from_source_failures")
        is not True
    ):
        blockers.append("network_failure_classification_missing")
    route_known = source_access.get("official_source_route_known") is True
    expected_source_access_status = (
        "source_route_available_pending_qualification"
        if route_known
        else "blocked_external_source_route"
    )
    if source_access.get("status") != expected_source_access_status:
        blockers.append("source_access_status_mismatch")
    if gate.get("source_access_status") != source_access.get("status"):
        blockers.append("gate_source_access_status_mismatch")
    gate_blockers = set(gate.get("blockers", []))
    if route_known and "official_source_route_unresolved" in gate_blockers:
        blockers.append("gate_route_blocker_stale")
    if not route_known and "official_source_route_unresolved" not in gate_blockers:
        blockers.append("gate_route_blocker_missing")
    if gate.get("round20qr_v3_baseline_challenge_resolved") is not False:
        blockers.append("gate_fail_open:baseline_challenge")
    for key in (
        "simple_baseline_sanity_passed",
        "strong_baseline_results_available",
        "challenge_thresholds_frozen",
        "headline_training_authorized",
        "round21r_replay_authorized",
        "c3_c4_execution_allowed",
        "coevolution_execution_allowed",
    ):
        if gate.get(key) is not False:
            blockers.append(f"gate_fail_open:{key}")
    if (
        gate.get("c2b_status") != EXPECTED_C2B_STATUS
        or replay.get("c2b_status") != EXPECTED_C2B_STATUS
    ):
        blockers.append("c2b_status_drift")
    if root_manifest.get("artifact_uid") != _canonical_sha256(root_manifest):
        blockers.append("root_manifest_uid")
    if root_manifest.get("status") != EXPECTED_STATUS:
        blockers.append("root_status_drift")
    if root_manifest.get("status") != gate.get("status"):
        blockers.append("root_gate_status_mismatch")
    if root_manifest.get("c2b_status") != EXPECTED_C2B_STATUS:
        blockers.append("root_c2b_status_drift")
    if root_manifest.get("c2b_status") != gate.get("c2b_status"):
        blockers.append("root_gate_c2b_mismatch")
    if root_manifest.get("c2b_status") != replay.get("c2b_status"):
        blockers.append("root_replay_c2b_mismatch")
    if root_manifest.get("round21r_replay_authorized") is not False:
        blockers.append("root_replay_fail_open")
    if root_manifest.get("round21r_replay_authorized") != gate.get(
        "round21r_replay_authorized"
    ):
        blockers.append("root_gate_replay_mismatch")
    if root_manifest.get("round21r_replay_authorized") != replay.get(
        "round21r_replay_authorized"
    ):
        blockers.append("root_replay_authorization_mismatch")
    for field in (
        "producer",
        "network_route",
        "source_access",
        "source_matrix",
        "runtime",
        "gate",
    ):
        binding = root_manifest.get(field, {})
        path = root / str(binding.get("path", ""))
        if not path.is_file() or binding.get("sha256") != _sha256(path):
            blockers.append(f"root_binding:{field}")
    if (
        replay.get("round21r_replay_authorized") is not False
        or replay.get("gpu_allowed") is not False
    ):
        blockers.append("replay_fail_open")
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "source_rows": len(rows),
        "formal_challenge_resolved": False,
        "round21r_replay_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()
    report = validate(Path(args.root).resolve(), Path(args.registry).resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
