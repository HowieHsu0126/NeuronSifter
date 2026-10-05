"""Shared primitives for the public foundation corpus V1 evidence layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


CORE_FAMILIES = (
    "cns_disease_aging_state_transcriptomics",
    "cns_cell_type_region_atlas_transcriptomics",
    "compound_dose_time_perturbation_transcriptomics",
)
SEMANTIC_AXES = (
    "study",
    "biological_unit",
    "species",
    "disease_state",
    "tissue",
    "brain_region",
    "cell_type",
    "assay",
    "modality",
    "feature",
    "time",
    "dose",
    "unit",
    "provenance",
)
CORE_QC_GRADES = frozenset({"A", "B"})
CONTROLLED_SOURCE_IDS = frozenset(
    {"adni", "dbgap", "uk_biobank", "rosmap", "amp_ad", "restricted_synapse"}
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def hash_path(path: Path) -> tuple[int, int, str]:
    """Return file count, byte count, and a path-bound deterministic hash."""

    if path.is_file():
        return 1, path.stat().st_size, sha256_file(path)
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        byte_count += size
    return file_count, byte_count, digest.hexdigest()


def unique_nonempty(values: Iterable[object]) -> set[str]:
    return {str(value).strip() for value in values if str(value or "").strip()}


def is_complete_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_semantic_axes(axes: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    missing = [axis for axis in SEMANTIC_AXES if axis not in axes]
    if missing:
        blockers.append("missing_semantic_axes:" + ",".join(missing))
    for axis in SEMANTIC_AXES:
        payload = axes.get(axis)
        if not isinstance(payload, Mapping):
            if axis not in missing:
                blockers.append(f"invalid_semantic_axis:{axis}")
            continue
        status = str(payload.get("status") or "")
        if status not in {"observed", "mapped", "not_applicable"}:
            blockers.append(f"invalid_semantic_axis_status:{axis}:{status or 'missing'}")
        if status in {"observed", "mapped"} and not str(payload.get("value") or "").strip():
            blockers.append(f"missing_semantic_axis_value:{axis}")
        if status == "not_applicable" and not str(payload.get("reason") or "").strip():
            blockers.append(f"missing_not_applicable_reason:{axis}")
    return blockers


def family_qualification(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    family_rows = list(rows)
    eligible = [
        row
        for row in family_rows
        if row.get("observation_eligible") is True
        and str(row.get("qc_grade")) in CORE_QC_GRADES
        and int(row.get("observation_count") or 0) > 0
        and int(row.get("value_bytes") or 0) > 0
        and is_complete_sha256(row.get("block_sha256"))
    ]
    sources = unique_nonempty(row.get("source_id") for row in eligible)
    groups = unique_nonempty(row.get("publication_accession_group") for row in eligible)
    cns_sources = unique_nonempty(
        row.get("source_id") for row in eligible if row.get("cns_source") is True
    )
    mirror_groups = unique_nonempty(row.get("mirror_group") for row in eligible)
    blockers: list[str] = []
    if len(sources) < 2:
        blockers.append("minimum_independent_source_ids_not_met")
    if len(groups) < 2:
        blockers.append("nonoverlapping_publication_accession_groups_not_met")
    if not cns_sources:
        blockers.append("minimum_cns_sources_not_met")
    if len(mirror_groups) < len(sources):
        blockers.append("source_independence_not_proven")
    return {
        "qualified": not blockers,
        "eligible_block_count": len(eligible),
        "independent_source_count": len(sources),
        "publication_accession_group_count": len(groups),
        "cns_source_count": len(cns_sources),
        "observation_count": sum(int(row.get("observation_count") or 0) for row in eligible),
        "value_bytes": sum(int(row.get("value_bytes") or 0) for row in eligible),
        "blockers": blockers,
    }
