"""Pure evidence-layer helpers for Public Foundation Corpus V1.

The evidence layer binds small JSON audits to content hashes and derives gates
from the referenced records.  It deliberately never treats a legacy registry
or a generated trajectory as a public observation without an explicit,
qualified value block.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from Libs.utils.data.public_foundation_corpus import (
    CORE_FAMILIES,
    CORE_QC_GRADES,
    canonical_sha256,
    family_qualification,
)


EVIDENCE_SCHEMA_VERSION = "neuronsifter-public-foundation-corpus-evidence-bundle-v1"
EVIDENCE_PRODUCER = "Libs/scripts/builders/build_public_foundation_corpus_evidence.py"
EVIDENCE_KEYS = (
    "source_qualification",
    "legacy_disposition",
    "entity_unit_crosswalk",
    "canonical_observation_blocks",
    "structured_action_catalog",
    "tool_model_registry",
    "frozen_splits",
    "leakage_audit",
    "qc_ess_statistics",
    "storage_ledger",
    "controlled_isolation",
)
STORAGE_ROOTS = {
    "raw": "Input/raw",
    "processed": "Input/processed",
    "canonical": "Input/canonical",
    "cache": "Input/cache",
    "output": "Output",
}
STORAGE_CAP_BYTES = 1_000_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    return canonical_sha256(payload)


def safe_repo_path(repo_root: Path, relative_path: str | Path) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {relative_path}")
    return candidate


def relative_repo_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def count_tsv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        first = handle.readline()
        if not first:
            return 0
        return sum(1 for _ in handle)


def count_json_records(payload: object) -> int:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, Mapping):
        return 0
    for key in ("records", "rows", "blocks", "sources", "catalogs", "tools"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    value = payload.get("record_count")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def evidence_reference(
    repo_root: Path,
    path: Path,
    *,
    schema_version: str,
    producer: str,
    record_count: int,
    status: str,
) -> dict[str, object]:
    relative = relative_repo_path(repo_root, path)
    if not relative.startswith("Output/audits/") or path.suffix != ".json":
        raise ValueError(f"Evidence reference must be an Output/audits JSON file: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "schema_version": schema_version,
        "producer": producer,
        "record_count": int(record_count),
        "status": status,
    }


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if not (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                yield path


def storage_ledger(
    repo_root: Path,
    *,
    roots: Mapping[str, str] = STORAGE_ROOTS,
    cap_bytes: int = STORAGE_CAP_BYTES,
    exclude_relative_paths: Sequence[str] = (),
) -> dict[str, object]:
    excluded = {str(value).replace("\\", "/") for value in exclude_relative_paths}
    classes: dict[str, dict[str, object]] = {}
    total_bytes = 0
    total_files = 0
    missing_roots: list[str] = []
    for class_name, relative in roots.items():
        path = safe_repo_path(repo_root, relative)
        if not path.exists():
            missing_roots.append(relative)
            classes[class_name] = {
                "root": relative,
                "exists": False,
                "file_count": 0,
                "bytes": 0,
                "excluded_file_count": 0,
                "excluded_bytes": 0,
            }
            continue
        file_count = 0
        byte_count = 0
        excluded_count = 0
        excluded_bytes = 0
        for file_path in _iter_files(path):
            relative_file = relative_repo_path(repo_root, file_path)
            size = file_path.stat().st_size
            if relative_file in excluded:
                excluded_count += 1
                excluded_bytes += size
                continue
            file_count += 1
            byte_count += size
        total_files += file_count
        total_bytes += byte_count
        classes[class_name] = {
            "root": relative,
            "exists": True,
            "file_count": file_count,
            "bytes": byte_count,
            "excluded_file_count": excluded_count,
            "excluded_bytes": excluded_bytes,
        }
    return {
        "roots": classes,
        "missing_roots": missing_roots,
        "total_file_count": total_files,
        "total_bytes": total_bytes,
        "decimal_byte_cap": cap_bytes,
        "cap_status": "passed" if total_bytes <= cap_bytes else "not_passed",
        "cap_exceeded": total_bytes > cap_bytes,
        "measurement_exclusions": sorted(excluded),
    }


def family_qualification_summary(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Apply the locked two-source/two-group/CNS/value-block contract."""

    return family_qualification(rows)


def overlap_counts(rows: Iterable[Mapping[str, object]], keys: Sequence[str]) -> dict[str, int]:
    rows_list = list(rows)
    counts: dict[str, int] = {}
    for key in keys:
        values = [str(row.get(key) or "") for row in rows_list]
        values = [value for value in values if value]
        counts[key] = len(values) - len(set(values))
    return counts


def adversarial_leakage_test(keys: Sequence[str]) -> dict[str, object]:
    base = {key: f"base-{key}" for key in keys}
    injected = dict(base)
    injected[keys[0]] = base[keys[0]]
    detected = overlap_counts([base, injected], keys)[keys[0]] > 0
    return {
        "tested": True,
        "injected_key": keys[0],
        "overlap_detected": detected,
        "status": "passed" if detected else "not_passed",
    }


def family_assessment_from_rows(
    family_id: str,
    rows: Iterable[Mapping[str, object]],
    *,
    query_domain_status: str,
    nested_split_status: str,
    leakage_status: str,
) -> dict[str, object]:
    if family_id not in CORE_FAMILIES:
        raise ValueError(f"unknown core family: {family_id}")
    summary = family_qualification(row for row in rows if row.get("family_id") == family_id)
    blockers = list(summary["blockers"])
    if query_domain_status != "frozen":
        blockers.append("query_domain_not_frozen")
    if nested_split_status != "frozen":
        blockers.append("nested_splits_not_frozen")
    if leakage_status != "passed":
        blockers.append("leakage_not_passed")
    blockers = sorted(set(blockers))
    qualified = not blockers
    eligible_rows = [
        row
        for row in rows
        if row.get("family_id") == family_id
        and row.get("observation_eligible") is True
        and str(row.get("qc_grade")) in CORE_QC_GRADES
    ]
    return {
        "status": "qualified" if qualified else "not_passed",
        "qualified_source_count": int(summary["independent_source_count"]),
        "independent_publication_accession_group_count": int(summary["publication_accession_group_count"]),
        "cns_source_count": int(summary["cns_source_count"]),
        "real_value_block_count": int(summary["eligible_block_count"]),
        "grade_ab_observation_count": sum(int(row.get("observation_count") or 0) for row in eligible_rows),
        "grade_ab_effective_sample_size": float(sum(float(row.get("effective_sample_size") or 0) for row in eligible_rows)),
        "query_domain_status": query_domain_status,
        "nested_split_status": nested_split_status,
        "leakage_status": leakage_status,
        "blocker_codes": blockers,
    }


def corpus_hash(payload: Mapping[str, object]) -> str:
    without_hash = {key: value for key, value in payload.items() if key != "corpus_hash"}
    return canonical_hash(without_hash)
