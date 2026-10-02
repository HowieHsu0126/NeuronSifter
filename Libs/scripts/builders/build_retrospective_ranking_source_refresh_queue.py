"""Build the append-only, outcome-blind public-source refresh queue audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_ranking_source_queue import (  # noqa: E402
    build as build_parent_queue,
)
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402

QUEUE_PATH = REPO_ROOT / "Input/manifests/retrospective_public/ranking_source_refresh_queue.tsv"
PARENT_QUEUE_PATH = REPO_ROOT / "Input/manifests/retrospective_public/ranking_informative_source_queue.tsv"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_public_ranking_source_refresh_queue.schema.json"
AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_ranking_source_refresh_queue.json"
FROZEN_PARENT_QUEUE_DIGEST = "da26b83d34ee00b417bf7dcb4a75680a05add3a57f75f50b7eef09b9f67c08fd"

FIELDS = (
    "refresh_uid", "refresh_sequence", "parent_queue_digest", "publication_uid",
    "source_identifier", "publication_date", "journal", "title", "source_accession",
    "context_hint", "objective_uid", "study_design_class", "therapeutic_arm_structure",
    "shared_control_expected", "same_endpoint_time_expected", "replicate_structure_expected",
    "uncertainty_source_expected", "independence_key", "ranking_informativeness_prior",
    "priority_order", "queue_status", "metadata_basis", "metadata_snapshot_date",
    "metadata_snapshot_path", "metadata_snapshot_sha256", "preexposed_sources_excluded",
    "abstracts_retrieved_before_priority", "full_text_retrieved_before_priority",
    "figures_retrieved_before_priority", "outcome_values_used_for_priority",
    "observed_effect_used_for_priority", "p_values_used_for_priority", "protected_holdout_access",
)

FALSE_FLAGS = (
    "abstracts_retrieved_before_priority", "full_text_retrieved_before_priority",
    "figures_retrieved_before_priority", "outcome_values_used_for_priority",
    "observed_effect_used_for_priority", "p_values_used_for_priority",
)
FORBIDDEN_PRIORITY_TERMS = (
    "effect", "p-value", "pvalue", "significant", "winner", "strongest", "observed",
    "improv", "mitigat", "rescu", "benefit", "protect", "identif", "increase",
    "decrease", "enhanc", "reduc", "restore", "attenuat", "efficac", "promot",
    "inhibit", "facilitat", "prevent", "ameliorat", "reverse",
)
OUTCOME_BLIND_TEXT_FIELDS = (
    "journal", "title", "context_hint", "objective_uid", "study_design_class",
    "therapeutic_arm_structure", "replicate_structure_expected",
    "uncertainty_source_expected", "independence_key", "metadata_basis",
)
ALLOWED_PRIORS = {"high", "medium_high", "medium", "low"}
ALLOWED_QUEUE_STATUSES = {"preregistered_for_public_artifact_recovery"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Ranking source refresh queue columns drifted")
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("Ranking source refresh queue is empty")
    return rows


def _digest(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_binding(row: dict[str, str]) -> dict[str, str]:
    label = row["metadata_snapshot_path"]
    if not label.startswith("Input/raw/public/retrospective/"):
        raise ValueError(f"Metadata snapshot is outside the public retrospective boundary: {label}")
    path = (REPO_ROOT / label).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Metadata snapshot escapes the repository: {label}") from error
    if not path.is_file():
        raise ValueError(f"Metadata snapshot is missing: {label}")
    observed_hash = _file_sha256(path)
    if observed_hash != row["metadata_snapshot_sha256"]:
        raise ValueError(f"Metadata snapshot hash drift: {label}")

    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != "neuronsifter-pubmed-metadata-snapshot-v1":
        raise ValueError(f"Metadata snapshot schema drift: {label}")
    if snapshot.get("retrieval_mode") != "metadata_only":
        raise ValueError(f"Metadata snapshot is not metadata-only: {label}")
    boundaries = snapshot.get("source_boundaries", {})
    forbidden_access = (
        "abstracts_retrieved", "full_text_retrieved", "figures_retrieved",
        "outcome_values_retrieved", "p_values_retrieved",
    )
    if any(boundaries.get(key) is not False for key in forbidden_access):
        raise ValueError(f"Metadata snapshot crossed the outcome-blind boundary: {label}")
    if boundaries.get("protected_holdout_access") != 0:
        raise ValueError(f"Metadata snapshot accessed the protected holdout: {label}")

    selection = snapshot.get("selection", {})
    if selection.get("parent_frozen_queue_digest") != FROZEN_PARENT_QUEUE_DIGEST:
        raise ValueError(f"Metadata snapshot parent queue binding drift: {label}")
    if any(selection.get(key) is not False for key in (
        "outcome_values_used_for_priority", "observed_effect_used_for_priority",
        "p_values_used_for_priority",
    )):
        raise ValueError(f"Metadata snapshot used outcome-bearing priority inputs: {label}")
    selected = snapshot.get("selected_record", {})
    expected_record = {
        "pmid": row["publication_uid"].removeprefix("pubmed:"),
        "publication_date": row["publication_date"],
        "journal": row["journal"],
        "title": row["title"],
        "pmcid": row["source_accession"],
    }
    for key, expected in expected_record.items():
        if selected.get(key) != expected:
            raise ValueError(f"Metadata snapshot {key} drift for {row['refresh_uid']}")
    if not str(snapshot.get("retrieved_at", "")).startswith(row["metadata_snapshot_date"]):
        raise ValueError(f"Metadata snapshot date drift for {row['refresh_uid']}")
    exclusions = snapshot.get("exclusions", {})
    if "10.7554/eLife.88279" not in exclusions.get("preexposed_source_dois", []):
        raise ValueError("Preexposed eLife 88279 source is not explicitly excluded")
    if exclusions.get("outcome_bearing_titles_excluded_before_priority") is not True:
        raise ValueError("Outcome-bearing titles were not excluded before priority")
    search_plan = snapshot.get("search_plan", {})
    if search_plan.get("bounded_query_count") != len(search_plan.get("queries", [])):
        raise ValueError("Metadata snapshot query count drift")
    if not 1 <= int(search_plan.get("bounded_query_count", 0)) <= 4:
        raise ValueError("Metadata snapshot search was not bounded")
    maximum_ids = int(search_plan.get("maximum_ids_per_query", 0))
    if not 1 <= maximum_ids <= 100:
        raise ValueError("Metadata snapshot per-query limit drift")
    if any(len(query.get("returned_pmids", [])) > maximum_ids for query in search_plan["queries"]):
        raise ValueError("Metadata snapshot query exceeded its declared bound")
    return {"path": label, "sha256": observed_hash}


def _validate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    parent_audit = build_parent_queue(write_output=False)
    if parent_audit["queue_digest"] != FROZEN_PARENT_QUEUE_DIGEST:
        raise ValueError("Frozen parent ranking queue digest drifted")
    seen_refresh: set[str] = set()
    seen_publication: set[str] = set()
    seen_independence: set[str] = set()
    bindings: list[dict[str, str]] = []
    for expected_sequence, row in enumerate(rows, start=1):
        missing = [field for field in FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"Refresh queue row {expected_sequence} is missing fields: {missing}")
        if int(row["refresh_sequence"]) != expected_sequence:
            raise ValueError("Refresh sequence must be contiguous and append ordered")
        if int(row["priority_order"]) != expected_sequence:
            raise ValueError("Refresh priority order must be contiguous and sorted")
        if row["refresh_uid"] in seen_refresh or row["publication_uid"] in seen_publication:
            raise ValueError(f"Refresh queue identifiers are not unique at row {expected_sequence}")
        if row["independence_key"] in seen_independence:
            raise ValueError(f"Refresh queue independence keys are not unique at row {expected_sequence}")
        seen_refresh.add(row["refresh_uid"])
        seen_publication.add(row["publication_uid"])
        seen_independence.add(row["independence_key"])
        if row["parent_queue_digest"] != FROZEN_PARENT_QUEUE_DIGEST:
            raise ValueError("Refresh row parent queue digest drifted")
        if row["ranking_informativeness_prior"] not in ALLOWED_PRIORS:
            raise ValueError(f"Unsupported refresh ranking prior: {row['ranking_informativeness_prior']}")
        if row["queue_status"] not in ALLOWED_QUEUE_STATUSES:
            raise ValueError(f"Unsupported refresh queue status: {row['queue_status']}")
        for field in ("shared_control_expected", "same_endpoint_time_expected", *FALSE_FLAGS):
            if row[field] not in {"true", "false"}:
                raise ValueError(f"Non-boolean refresh queue flag {field}: {row[field]}")
        if any(row[field] != "false" for field in FALSE_FLAGS):
            raise ValueError("Outcome-bearing access and priority flags must remain false")
        if row["preexposed_sources_excluded"] != "true":
            raise ValueError("Preexposed sources must remain excluded")
        if row["protected_holdout_access"] != "0":
            raise ValueError("Protected holdout access must remain zero")
        for field in OUTCOME_BLIND_TEXT_FIELDS:
            text = row[field].lower()
            if any(term in text for term in FORBIDDEN_PRIORITY_TERMS):
                raise ValueError(f"Outcome-bearing term in refresh queue field {field} at row {expected_sequence}")
        bindings.append(_snapshot_binding(row))
    return bindings


def build(*, write_output: bool = True, queue_path: Path = QUEUE_PATH) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    if write_output and queue_path != QUEUE_PATH.resolve():
        raise ValueError("Only the canonical refresh queue may produce the persisted audit")
    rows = _read_rows(queue_path)
    snapshot_bindings = _validate_rows(rows)
    try:
        queue_path_label = queue_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        queue_path_label = queue_path.as_posix()
    audit: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-ranking-source-refresh-queue-v1",
        "status": "metadata_only_refresh_queue_preregistered",
        "queue_path": queue_path_label,
        "queue_digest": _digest(rows),
        "record_count": len(rows),
        "parent_frozen_queue_path": PARENT_QUEUE_PATH.relative_to(REPO_ROOT).as_posix(),
        "parent_frozen_queue_digest": FROZEN_PARENT_QUEUE_DIGEST,
        "parent_queue_unchanged": True,
        "priority_order": [row["publication_uid"] for row in rows],
        "metadata_snapshot_bindings": snapshot_bindings,
        "preexposed_sources_excluded": True,
        "abstracts_retrieved_before_priority": False,
        "full_text_retrieved_before_priority": False,
        "figures_retrieved_before_priority": False,
        "outcome_values_used_for_priority": False,
        "observed_effect_used_for_priority": False,
        "p_values_used_for_priority": False,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": 0,
        "next_action": "recover_preregistered_public_artifacts_and_qualify_immediately",
        "records": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(audit),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Ranking source refresh queue audit schema error: {detail}")
    if write_output:
        write_json(AUDIT_PATH, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    audit = build(write_output=not args.no_write)
    print(json.dumps({key: audit[key] for key in (
        "status", "record_count", "queue_digest", "parent_frozen_queue_digest", "next_action",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
