"""Build an outcome-blind priority audit for ranking-informative public sources."""

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

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402

QUEUE_PATH = REPO_ROOT / "Input/manifests/retrospective_public/ranking_informative_source_queue.tsv"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_public_ranking_source_queue.schema.json"
AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_ranking_source_queue.json"

FIELDS = (
    "queue_uid", "publication_uid", "source_identifier", "publication_date", "journal",
    "title", "source_accession", "context_hint", "objective_uid", "study_design_class",
    "therapeutic_arm_structure", "shared_control_expected", "same_endpoint_time_expected",
    "replicate_structure_expected", "uncertainty_source_expected", "independence_key",
    "ranking_informativeness_prior", "priority_order", "queue_status", "metadata_basis",
    "metadata_snapshot_date", "outcome_values_used_for_priority",
    "observed_effect_used_for_priority", "p_values_used_for_priority",
    "protected_holdout_access",
)
FORBIDDEN_PRIORITY_TERMS = ("effect", "p-value", "pvalue", "significant", "winner", "strongest", "observed")
FORBIDDEN_TITLE_TERMS = (
    "effect", "p-value", "pvalue", "significant", "winner", "strongest",
    "observed", "improv", "mitigat", "rescu", "benefit", "protect",
)
OUTCOME_BLIND_TEXT_FIELDS = (
    "journal", "title", "context_hint", "study_design_class",
    "therapeutic_arm_structure", "replicate_structure_expected",
    "uncertainty_source_expected", "independence_key", "metadata_basis",
)
ALLOWED_PRIORS = {"high", "medium_high", "medium", "low"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Ranking source queue columns drifted")
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("Ranking source queue is empty")
    return rows


def _digest(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_rows(rows: list[dict[str, str]]) -> None:
    seen_queue: set[str] = set()
    seen_publication: set[str] = set()
    for expected_rank, row in enumerate(rows, start=1):
        missing = [field for field in FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"Queue row {expected_rank} is missing fields: {missing}")
        if row["queue_uid"] in seen_queue or row["publication_uid"] in seen_publication:
            raise ValueError(f"Queue identifiers are not unique at row {expected_rank}")
        seen_queue.add(row["queue_uid"])
        seen_publication.add(row["publication_uid"])
        if int(row["priority_order"]) != expected_rank:
            raise ValueError("Priority order must be contiguous and sorted")
        if row["ranking_informativeness_prior"] not in ALLOWED_PRIORS:
            raise ValueError(f"Unsupported ranking prior: {row['ranking_informativeness_prior']}")
        for field in (
            "shared_control_expected", "same_endpoint_time_expected",
            "outcome_values_used_for_priority", "observed_effect_used_for_priority",
            "p_values_used_for_priority",
        ):
            if row[field] not in {"true", "false"}:
                raise ValueError(f"Non-boolean queue flag {field}: {row[field]}")
        if any(row[field] != "false" for field in (
            "outcome_values_used_for_priority", "observed_effect_used_for_priority",
            "p_values_used_for_priority",
        )):
            raise ValueError("Outcome, effect, and p-value priority flags must remain false")
        if row["protected_holdout_access"] != "0":
            raise ValueError("Protected holdout access must remain zero")
        priority_basis = " ".join((row["metadata_basis"], row["study_design_class"], row["ranking_informativeness_prior"])).lower()
        if any(term in priority_basis for term in FORBIDDEN_PRIORITY_TERMS):
            raise ValueError(f"Outcome-bearing term in priority metadata at row {expected_rank}")
        for field in OUTCOME_BLIND_TEXT_FIELDS:
            text = row[field].lower()
            if any(term in text for term in FORBIDDEN_TITLE_TERMS):
                if field == "title":
                    raise ValueError(f"Outcome-bearing term in queue title at row {expected_rank}")
                raise ValueError(f"Outcome-bearing term in queue field {field} at row {expected_rank}")


def build(*, write_output: bool = True, queue_path: Path = QUEUE_PATH) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    if write_output and queue_path != QUEUE_PATH.resolve():
        raise ValueError("Only the canonical ranking source queue may produce the persisted audit")
    rows = _read_rows(queue_path)
    _validate_rows(rows)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        queue_path_label = queue_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        queue_path_label = queue_path.as_posix()
    audit: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-ranking-source-queue-v1",
        "status": "ranking_informative_queue_frozen",
        "queue_path": queue_path_label,
        "queue_digest": _digest(rows),
        "record_count": len(rows),
        "priority_order": [row["publication_uid"] for row in rows],
        "ranking_informativeness_priors": {
            key: sum(row["ranking_informativeness_prior"] == key for row in rows)
            for key in sorted(ALLOWED_PRIORS)
        },
        "outcome_values_used_for_priority": False,
        "observed_effect_used_for_priority": False,
        "p_values_used_for_priority": False,
        "abstract_outcome_claims_used_for_priority": False,
        "current_live_web_search_used_for_formal_evidence": False,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": 0,
        "next_action": "materialize_sources_in_priority_order_and_qualify_immediately",
        "records": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    errors = sorted(Draft202012Validator(schema).iter_errors(audit), key=lambda error: list(error.absolute_path))
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Ranking source queue audit schema error: {detail}")
    if write_output:
        write_json(AUDIT_PATH, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    audit = build(write_output=not args.no_write)
    print(json.dumps({key: audit[key] for key in ("status", "record_count", "queue_digest", "next_action")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
