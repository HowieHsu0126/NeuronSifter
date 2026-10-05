"""Validate the design-enriched outcome-blind public CNS ranking intake."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_design_enriched_ranking_intake import (  # noqa: E402
    INTAKE_SCHEMA_PATH,
    QUEUE_FIELDS,
    _assert_parent_state,
    _load_policy,
    _output_paths,
    _queue_digest,
    _sha256,
    _validate_preregistration,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUEUE_FIELDS:
            raise ValueError("Design-enriched ranking queue columns drifted")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def validate() -> dict[str, Any]:
    blockers: list[str] = []
    policy = _load_policy()
    paths = _output_paths(policy)
    try:
        _assert_parent_state(policy)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"parent_state:{exc}")
    if not paths["preregistration_manifest"].is_file():
        blockers.append("missing:preregistration_manifest")
        preregistration: dict[str, Any] = {}
    else:
        try:
            preregistration = _read_json(paths["preregistration_manifest"])
            _validate_preregistration(preregistration, policy)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"preregistration:{exc}")
            preregistration = {}
    if not paths["intake_audit"].is_file():
        if paths["metadata_snapshot"].exists() or paths["source_queue"].exists():
            blockers.append("partial_outputs_without_intake_audit")
        unique = sorted(set(blockers))
        return {
            "schema_version": "neuronsifter-retrospective-public-design-enriched-ranking-intake-validation-v1",
            "status": "preregistered" if not unique else "blocked",
            "passed": not unique,
            "blocker_count": len(unique),
            "blockers": unique,
            "record_count": 0,
            "source_count_cap": 5,
            "metadata_record_cap": 150,
            "protected_holdout_access": 0,
        }
    try:
        audit = _read_json(paths["intake_audit"])
        schema = _read_json(INTAKE_SCHEMA_PATH)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(audit),
            key=lambda error: list(error.absolute_path),
        )
        blockers.extend(f"schema:{list(error.absolute_path)}:{error.message}" for error in errors[:8])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"audit:{exc}")
        audit = {}
    if not paths["metadata_snapshot"].is_file():
        blockers.append("missing:metadata_snapshot")
        snapshot: dict[str, Any] = {}
    else:
        snapshot = _read_json(paths["metadata_snapshot"])
    if not paths["source_queue"].is_file():
        blockers.append("missing:source_queue")
        rows: list[dict[str, str]] = []
    else:
        rows = _read_queue(paths["source_queue"])
    if audit:
        if audit.get("record_count") != len(rows):
            blockers.append("record_count_mismatch")
        if audit.get("queue_digest") != _queue_digest(rows, preregistration):
            blockers.append("queue_digest_mismatch")
        if audit.get("queue_digest_contract") != "intake_envelope_v1":
            blockers.append("queue_digest_contract")
        if audit.get("priority_order") != [row.get("publication_uid") for row in rows]:
            blockers.append("priority_order_mismatch")
        if audit.get("preregistration_digest") != preregistration.get("preregistration_digest"):
            blockers.append("preregistration_binding_mismatch")
        binding = audit.get("metadata_snapshot_binding", {})
        if snapshot and binding.get("sha256") != _sha256(paths["metadata_snapshot"]):
            blockers.append("metadata_snapshot_hash_mismatch")
        if audit.get("selection_summary", {}).get("metadata_record_count", 0) > 150:
            blockers.append("metadata_record_cap_exceeded")
    if snapshot:
        if snapshot.get("retrieval_mode") != "metadata_only":
            blockers.append("metadata_snapshot_retrieval_mode")
        boundaries = snapshot.get("source_boundaries", {})
        for key in (
            "abstracts_retrieved",
            "full_text_retrieved",
            "figures_retrieved",
            "supplementary_retrieved",
            "outcome_values_retrieved",
            "p_values_retrieved",
            "author_winner_statements_retrieved",
        ):
            if boundaries.get(key) is not False:
                blockers.append(f"metadata_boundary:{key}")
        if boundaries.get("protected_holdout_access") != 0:
            blockers.append("metadata_boundary:protected_holdout_access")
        if snapshot.get("exclusions", {}).get("prior_metadata_universe_count") != 100:
            blockers.append("prior_metadata_universe_exclusion_count")
    prior_pmids = set(preregistration.get("exclusions", {}).get("prior_metadata_universe_pmids", []))
    preexposed_pmids = set(preregistration.get("exclusions", {}).get("preexposed_pmids", []))
    preexposed_dois = set(preregistration.get("exclusions", {}).get("preexposed_dois", []))
    for expected_index, row in enumerate(rows, start=1):
        pmid = row.get("publication_uid", "").removeprefix("pubmed:")
        if row.get("intake_sequence") != str(expected_index) or row.get("priority_order") != str(expected_index):
            blockers.append(f"queue_order:{expected_index}")
        if pmid in prior_pmids or pmid in preexposed_pmids:
            blockers.append(f"excluded_pmid:{pmid}")
        if row.get("doi", "").lower() in preexposed_dois:
            blockers.append(f"excluded_doi:{row.get('doi')}")
        if int(row.get("named_intervention_entity_count", "0")) < 2:
            blockers.append(f"named_intervention_count:{pmid}")
        if row.get("experimental_numeric_evidence_status") != "unknown_pre_recovery":
            blockers.append(f"numeric_evidence_prejudged:{pmid}")
        for key in (
            "abstracts_retrieved_before_priority",
            "full_text_retrieved_before_priority",
            "figures_retrieved_before_priority",
            "supplementary_retrieved_before_priority",
            "outcome_values_used_for_priority",
            "observed_effect_used_for_priority",
            "p_values_used_for_priority",
            "author_winner_statement_used_for_priority",
        ):
            if row.get(key) != "false":
                blockers.append(f"queue_boundary:{pmid}:{key}")
        if row.get("protected_holdout_access") != "0":
            blockers.append(f"queue_boundary:{pmid}:protected_holdout_access")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-design-enriched-ranking-intake-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "record_count": len(rows),
        "source_count_cap": 5,
        "metadata_record_cap": 150,
        "queue_digest": audit.get("queue_digest"),
        "protected_holdout_access": audit.get("protected_holdout_access"),
        "next_action": audit.get("next_action"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
