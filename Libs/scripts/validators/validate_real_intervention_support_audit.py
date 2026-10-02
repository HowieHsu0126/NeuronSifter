"""Fail-closed validation for the real-intervention U0 support audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}


def _repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_matches(root: Path, expected_path: Path, binding: object) -> bool:
    if not isinstance(binding, Mapping):
        return False
    try:
        bound_path = _repo_path(root, str(binding.get("path", "")))
        expected_bytes = int(binding.get("bytes", -1))
    except (TypeError, ValueError):
        return False
    return (
        bound_path == expected_path.resolve()
        and expected_path.is_file()
        and expected_path.stat().st_size == expected_bytes
        and _sha256(expected_path) == binding.get("sha256")
    )


def validate(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Validate the materialized support matrix and its source bindings."""

    root = repo_root.resolve()
    blockers: list[str] = []
    try:
        config = load_config(root / "Libs/configs/config_stack.yaml")
        contract = config["real_intervention_evidence"]["support_audit"]
        input_paths = {
            name: _repo_path(root, str(value))
            for name, value in contract["inputs"].items()
        }
        matrix_path = _repo_path(root, str(contract["outputs"]["support_matrix"]))
        root_cause_path = _repo_path(root, str(contract["outputs"]["root_cause"]))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {"status": "blocked", "blocker_count": 1, "blockers": [f"contract:{exc}"]}

    for name, path in {**input_paths, "support_matrix": matrix_path, "root_cause": root_cause_path}.items():
        if not path.is_file():
            blockers.append(f"missing_artifact:{name}")
    if blockers:
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": sorted(set(blockers))}

    try:
        source_rows = _read_tsv(input_paths["source_manifest"])
        episode_rows = _read_tsv(input_paths["episode_catalog"])
        matrix_rows = _read_tsv(matrix_path)
        summary = json.loads(root_cause_path.read_text(encoding="utf-8"))
    except (OSError, csv.Error, json.JSONDecodeError, TypeError) as exc:
        return {"status": "blocked", "blocker_count": 1, "blockers": [f"read:{exc}"]}
    if not isinstance(summary, dict):
        blockers.append("root_cause:not_object")
        summary = {}

    expected_source_count = int(contract["expected_source_count"])
    expected_episode_count = int(contract["expected_episode_count"])
    expected_u0_support = int(contract["expected_u0_executable_support_count"])
    if len(source_rows) != expected_source_count:
        blockers.append(f"source_count:{len(source_rows)}")
    if len(episode_rows) != expected_episode_count:
        blockers.append(f"episode_input_count:{len(episode_rows)}")
    if len(matrix_rows) != expected_episode_count:
        blockers.append(f"matrix_row_count:{len(matrix_rows)}")
    if summary.get("source_count") != expected_source_count:
        blockers.append("summary_source_count")
    if summary.get("episode_count") != expected_episode_count:
        blockers.append("summary_episode_count")
    if summary.get("u0_executable_support_count") != expected_u0_support:
        blockers.append("summary_u0_support_count")

    allowed_failures = set(contract["primary_failure_classes"])
    allowed_dispositions = set(contract["dispositions"])
    episode_uids = [row.get("episode_uid", "") for row in episode_rows]
    matrix_uids = [row.get("episode_uid", "") for row in matrix_rows]
    if not all(episode_uids) or len(set(episode_uids)) != len(episode_uids):
        blockers.append("episode_input_identity")
    if matrix_uids != episode_uids:
        blockers.append("matrix_episode_identity")
    for index, row in enumerate(matrix_rows):
        primary = row.get("primary_failure_class", "")
        disposition = row.get("support_disposition", "")
        if primary not in allowed_failures:
            blockers.append(f"row_{index}:unknown_primary_failure")
        if disposition not in allowed_dispositions:
            blockers.append(f"row_{index}:unknown_disposition")
        if row.get("u0_executable_support") != "unsupported":
            blockers.append(f"row_{index}:u0_support_not_zero")

    if summary.get("status") != "resolved":
        blockers.append("summary_status")
    if summary.get("unknown_primary_failure_count") != 0:
        blockers.append("summary_unknown_primary")
    if summary.get("unknown_disposition_count") != 0:
        blockers.append("summary_unknown_disposition")
    if summary.get("all_episode_support_dispositions_known") is not True:
        blockers.append("summary_disposition_resolution")
    for key, expected in BOUNDARY.items():
        if summary.get(key) != expected:
            blockers.append(f"boundary:{key}")

    bindings = summary.get("input_bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("input_bindings:missing")
    else:
        for name, path in input_paths.items():
            if not _binding_matches(root, path, bindings.get(name)):
                blockers.append(f"input_binding:{name}")

    unique_blockers = sorted(set(blockers))
    return {
        "status": "passed" if not unique_blockers else "blocked",
        "blocker_count": len(unique_blockers),
        "blockers": unique_blockers,
        "source_count": len(source_rows),
        "episode_count": len(matrix_rows),
        "u0_executable_support_count": sum(
            row.get("u0_executable_support") == "supported" for row in matrix_rows
        ),
        **BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    report = validate(args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
