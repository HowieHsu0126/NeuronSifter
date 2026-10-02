"""Role-aware source consumption boundaries for Phase 1A.

The ledger is the authority.  This module makes weak calibration and context
prior records available to permitted consumers while preventing accidental
promotion into mainline observations, Grade A/B ESS, or foundation coverage.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from Libs.utils.data.raw_data_registry import write_tsv


TERMINAL_DECISIONS = frozenset({
    "qualified_for_mainline",
    "qualified_for_weak_calibration",
    "qualified_as_context_prior",
    "deferred",
    "formally_excluded",
})


REQUALIFICATION_DELTA_FIELDS = (
    "delta_version",
    "entry_id",
    "source_id",
    "source_accession",
    "prior_decision",
    "decision",
    "admission_allowed",
    "scope",
    "frozen_ledger_path",
    "frozen_ledger_sha256",
    "block_manifest_path",
    "block_manifest_sha256",
    "training_evaluation_role",
    "access_class",
    "status",
)


def load_phase1a_ledger(path: str | Path) -> dict[str, dict[str, str]]:
    """Load the frozen Phase 1A ledger and fail closed on malformed rows."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        decision = str(row.get("decision", ""))
        if not source_id or decision not in TERMINAL_DECISIONS or row.get("terminal_status") != "terminal":
            raise ValueError(f"Invalid Phase 1A source ledger row: {source_id or '<missing>'}")
        if source_id in result:
            raise ValueError(f"Duplicate Phase 1A source: {source_id}")
        result[source_id] = row
    return result


def allowed_for_role(row: Mapping[str, str], role: str) -> bool:
    decision = row.get("decision")
    if role == "mainline_observation":
        return decision == "qualified_for_mainline"
    if role == "grade_ab_ess":
        return decision == "qualified_for_mainline"
    if role == "foundation_transfer":
        return decision == "qualified_for_mainline"
    if role == "weak_calibration":
        return decision in {"qualified_for_mainline", "qualified_for_weak_calibration"}
    if role == "context_prior":
        return decision in {"qualified_for_mainline", "qualified_as_context_prior"}
    if role == "holdout_extension":
        return decision in {"qualified_for_mainline", "qualified_for_weak_calibration", "qualified_as_context_prior"}
    raise ValueError(f"Unknown Phase 1 source-consumption role: {role}")


def load_accession_requalification_delta(path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load an append-only, accession-scoped re-admission delta.

    The source-level Phase 1A ledger stays immutable.  A delta may grant a
    role only to the explicitly named accession after its real value block has
    been qualified and bound to the frozen ledger.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(REQUALIFICATION_DELTA_FIELDS):
            raise ValueError(f"Unexpected requalification-delta fields: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("Requalification delta must contain at least one row.")
    entries: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("source_accession", "")))
        if not all(key):
            raise ValueError(f"Requalification delta has a blank source identity: {row}")
        if key in entries:
            raise ValueError(f"Requalification delta has duplicate source/accession entry: {key}")
        if row.get("prior_decision") not in TERMINAL_DECISIONS:
            raise ValueError(f"Requalification delta has invalid prior decision: {row}")
        if row.get("decision") not in TERMINAL_DECISIONS:
            raise ValueError(f"Requalification delta has invalid decision: {row}")
        if row.get("admission_allowed") not in {"true", "false"}:
            raise ValueError(f"Requalification delta has invalid admission flag: {row}")
        entries[key] = row
    return entries


def upsert_accession_requalification_delta(
    path: str | Path,
    row: Mapping[str, object],
) -> None:
    """Add or replace one accession-scoped decision without dropping peers.

    The Phase 1A ledger is frozen, but re-admission deltas are intentionally
    append-only across independently qualified accessions.  Builders must not
    recreate a single-row delta, because that would silently revoke a prior
    validated accession when a later family is materialized.
    """

    delta_path = Path(path)
    normalized = {field: str(row.get(field, "")) for field in REQUALIFICATION_DELTA_FIELDS}
    key = (normalized["source_id"], normalized["source_accession"])
    if not all(key):
        raise ValueError("Requalification delta row has a blank source/accession identity")

    ordered_rows: list[dict[str, str]] = []
    if delta_path.is_file():
        with delta_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != list(REQUALIFICATION_DELTA_FIELDS):
                raise ValueError(f"Unexpected requalification-delta fields: {reader.fieldnames}")
            ordered_rows = [
                {field: str(existing.get(field, "")) for field in REQUALIFICATION_DELTA_FIELDS}
                for existing in reader
            ]

    replaced = False
    for index, existing in enumerate(ordered_rows):
        existing_key = (existing["source_id"], existing["source_accession"])
        if existing_key == key:
            ordered_rows[index] = normalized
            replaced = True
            break
    if not replaced:
        ordered_rows.append(normalized)

    seen: set[tuple[str, str]] = set()
    for existing in ordered_rows:
        existing_key = (existing["source_id"], existing["source_accession"])
        if not all(existing_key) or existing_key in seen:
            raise ValueError(f"Requalification delta contains duplicate identity: {existing_key}")
        seen.add(existing_key)
    write_tsv(delta_path, list(REQUALIFICATION_DELTA_FIELDS), ordered_rows)


def allowed_for_accession_role(
    row: Mapping[str, object],
    ledger: Mapping[str, Mapping[str, str]],
    role: str,
    *,
    requalification_delta: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
) -> bool:
    """Resolve a consumption role without widening source-wide permissions."""

    source_id = str(row.get("source_id", ""))
    ledger_row = ledger.get(source_id)
    if ledger_row is None:
        raise ValueError(f"Source is not present in frozen Phase 1A ledger: {source_id}")
    if allowed_for_role(ledger_row, role):
        return True
    if requalification_delta is None:
        return False
    accession = str(row.get("source_accession", ""))
    delta = requalification_delta.get((source_id, accession))
    if delta is None:
        return False
    if delta.get("scope") != "source_accession_only":
        return False
    if delta.get("status") != "accepted":
        return False
    if delta.get("admission_allowed") != "true":
        return False
    return allowed_for_role(delta, role)


def filter_source_rows(
    rows: Iterable[Mapping[str, object]],
    ledger: Mapping[str, Mapping[str, str]],
    *,
    role: str,
    requalification_delta: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
) -> list[Mapping[str, object]]:
    """Return only rows allowed for *role*; missing ledger bindings fail closed."""
    selected: list[Mapping[str, object]] = []
    for row in rows:
        source_id = str(row.get("source_id", ""))
        ledger_row = ledger.get(source_id)
        if ledger_row is None:
            raise ValueError(f"Source is not present in frozen Phase 1A ledger: {source_id}")
        if allowed_for_accession_role(
            row,
            ledger,
            role,
            requalification_delta=requalification_delta,
        ):
            selected.append(row)
    return selected


def assert_not_mainline(rows: Iterable[Mapping[str, object]], ledger: Mapping[str, Mapping[str, str]]) -> None:
    """Raise if weak/context records are about to enter a mainline surface."""
    leaked = filter_source_rows(rows, ledger, role="mainline_observation")
    source_ids = {str(row.get("source_id", "")) for row in rows}
    allowed_ids = {str(row.get("source_id", "")) for row in leaked}
    disallowed = sorted(source_ids - allowed_ids)
    if disallowed:
        raise ValueError("Non-mainline sources cannot enter mainline surfaces: " + ",".join(disallowed))
