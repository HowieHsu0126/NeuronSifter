"""Validate the bounded source-data-first metadata universe and queue."""

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

from Libs.scripts.builders.build_retrospective_public_source_data_first_intake import (  # noqa: E402
    INTAKE_SCHEMA_PATH,
    QUEUE_FIELDS,
    UNIVERSE_FIELDS,
    _assert_parent_state,
    _load_policy,
    _output_paths,
    _queue_digest,
    _sha256,
    _universe_digest,
    _validate_preregistration,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_tsv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"TSV columns drifted: {path}")
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
        if any(paths[key].exists() for key in ("metadata_universe", "source_queue", "metadata_snapshot")):
            blockers.append("partial_outputs_without_intake_audit")
        unique = sorted(set(blockers))
        return {
            "schema_version": "neuronsifter-retrospective-public-source-data-first-intake-validation-v1",
            "status": "preregistered" if not unique else "blocked",
            "passed": not unique,
            "blocker_count": len(unique),
            "blockers": unique,
            "repository_family_count": 4,
            "metadata_asset_cap": 160,
            "queue_record_cap": 8,
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
    try:
        universe_rows = _read_tsv(paths["metadata_universe"], UNIVERSE_FIELDS)
    except (OSError, ValueError, TypeError) as exc:
        blockers.append(f"metadata_universe:{exc}")
        universe_rows = []
    try:
        queue_rows = _read_tsv(paths["source_queue"], QUEUE_FIELDS)
    except (OSError, ValueError, TypeError) as exc:
        blockers.append(f"source_queue:{exc}")
        queue_rows = []
    if not paths["metadata_snapshot"].is_file():
        blockers.append("missing:metadata_snapshot")
        snapshot: dict[str, Any] = {}
    else:
        snapshot = _read_json(paths["metadata_snapshot"])
    if audit:
        if audit.get("metadata_asset_count") != len(universe_rows):
            blockers.append("metadata_asset_count")
        if audit.get("queue_record_count") != len(queue_rows):
            blockers.append("queue_record_count")
        if audit.get("metadata_universe_digest") != _universe_digest(universe_rows, preregistration):
            blockers.append("metadata_universe_digest")
        if audit.get("metadata_universe_digest_contract") != "source_data_universe_envelope_v1":
            blockers.append("metadata_universe_digest_contract")
        if audit.get("queue_digest") != _queue_digest(queue_rows, preregistration):
            blockers.append("queue_digest")
        if audit.get("queue_digest_contract") != "source_data_queue_envelope_v1":
            blockers.append("queue_digest_contract")
        if len(universe_rows) > int(policy["authorization"]["metadata_asset_cap"]):
            blockers.append("metadata_asset_cap")
        if len(queue_rows) > int(policy["authorization"]["preregistered_source_cap"]):
            blockers.append("queue_record_cap")
        if audit.get("preregistration_digest") != preregistration.get("preregistration_digest"):
            blockers.append("preregistration_binding")
        binding = audit.get("metadata_snapshot_binding", {})
        if snapshot and binding.get("sha256") != _sha256(paths["metadata_snapshot"]):
            blockers.append("metadata_snapshot_hash")
        statuses = [item.get("status") for item in audit.get("repository_family_status", [])]
        if audit.get("status") == "source_data_queue_exhausted" and any(
            status != "resolved" for status in statuses
        ):
            blockers.append("exhaustion_with_unresolved_repository_family")
    if snapshot:
        if snapshot.get("retrieval_mode") != "repository_metadata_only":
            blockers.append("snapshot_retrieval_mode")
        boundaries = snapshot.get("source_boundaries", {})
        for key in (
            "source_asset_content_retrieved",
            "numeric_outcomes_retrieved",
            "effect_direction_used_for_priority",
            "p_values_used_for_priority",
            "winner_statements_used_for_priority",
        ):
            if boundaries.get(key) is not False:
                blockers.append(f"snapshot_boundary:{key}")
        if boundaries.get("protected_holdout_access") != 0:
            blockers.append("snapshot_boundary:protected_holdout_access")
    eligible_uids = {
        row["asset_uid"] for row in universe_rows if row["eligibility_status"] == "eligible"
    }
    queue_uids = {row["asset_uid"] for row in queue_rows}
    if not queue_uids.issubset(eligible_uids):
        blockers.append("queue_contains_ineligible_asset")
    if len(queue_uids) != len(queue_rows):
        blockers.append("duplicate_queue_asset")
    for expected_index, row in enumerate(queue_rows, start=1):
        if row["queue_sequence"] != str(expected_index) or row["ranking_priority"] != str(expected_index):
            blockers.append(f"queue_order:{expected_index}")
        if row["source_asset_content_retrieved_before_priority"] != "false":
            blockers.append(f"queue_content_access:{row['asset_uid']}")
        if row["protected_holdout_access"] != "0":
            blockers.append(f"queue_holdout_access:{row['asset_uid']}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-source-data-first-intake-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "intake_status": audit.get("status"),
        "repository_family_count": audit.get("repository_family_count"),
        "metadata_asset_count": len(universe_rows),
        "metadata_eligible_count": audit.get("metadata_eligible_count"),
        "queue_record_count": len(queue_rows),
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
