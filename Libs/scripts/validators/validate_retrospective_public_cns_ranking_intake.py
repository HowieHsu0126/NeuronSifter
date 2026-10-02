"""Validate the bounded metadata-only public CNS ranking intake artifacts."""

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

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    CONFIG_PATH,
    FROZEN_QUEUE_AUDIT_PATH,
    INTAKE_SCHEMA_PATH,
    MATERIALIZATION_AUDIT_PATH,
    QUEUE_FIELDS,
    REFRESH_QUEUE_AUDIT_PATH,
    REPLAY_AUDIT_PATH,
    _assert_parent_state,
    _canonical_digest,
    _load_policy,
    _output_paths,
    _sha256,
    _validate_preregistration,
)

RECOVERY_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_source_recovery.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _schema_blockers(payload: dict[str, Any], schema_path: Path, prefix: str) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{prefix}:{list(error.absolute_path)}:{error.message}" for error in errors]


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUEUE_FIELDS:
            raise ValueError("CNS ranking intake queue columns drifted")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def validate() -> dict[str, Any]:
    blockers: list[str] = []
    policy = _load_policy()
    paths = _output_paths(policy)
    recovery_payload: dict[str, Any] = {}
    if RECOVERY_AUDIT_PATH.is_file():
        try:
            recovery_payload = _read_json(RECOVERY_AUDIT_PATH)
            materialization = _read_json(MATERIALIZATION_AUDIT_PATH)
            replay = _read_json(REPLAY_AUDIT_PATH)
            frozen = _read_json(FROZEN_QUEUE_AUDIT_PATH)
            refresh = _read_json(REFRESH_QUEUE_AUDIT_PATH)
            parent = policy["parent_state"]
            expected_current = {
                "frozen_queue_digest": parent["frozen_queue_digest"],
                "refresh_queue_digest": parent["refresh_queue_digest"],
                "status": parent["status"],
                "accepted_records": parent["accepted_records"],
                "rejected_records": int(parent["rejected_records"]) + 1,
                "independent_decision_sensitive_groups": parent["independent_decision_sensitive_groups"],
                "required_groups": parent["required_groups"],
                "split_files_created": False,
                "protected_holdout_access": 0,
                "recovery_status": "queue_exhausted_without_new_decision_sensitive_group",
            }
            observed_current = {
                "frozen_queue_digest": frozen.get("queue_digest"),
                "refresh_queue_digest": refresh.get("queue_digest"),
                "status": materialization.get("status"),
                "accepted_records": materialization.get("accepted_record_count"),
                "rejected_records": materialization.get("rejected_record_count"),
                "independent_decision_sensitive_groups": replay.get("independent_decision_sensitive_group_count"),
                "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
                "split_files_created": replay.get("split_files_created"),
                "protected_holdout_access": replay.get("protected_holdout_access"),
                "recovery_status": recovery_payload.get("status"),
            }
            if observed_current != expected_current:
                blockers.append(
                    f"completed_state:observed={observed_current}:expected={expected_current}"
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"completed_state:{exc}")
    else:
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
    audit_exists = paths["intake_audit"].is_file()
    snapshot_exists = paths["metadata_snapshot"].is_file()
    queue_exists = paths["source_queue"].is_file()
    if not audit_exists:
        if snapshot_exists or queue_exists:
            blockers.append("partial_outputs_without_intake_audit")
        unique = sorted(set(blockers))
        return {
            "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-validation-v1",
            "status": "preregistered" if not unique else "blocked",
            "passed": not unique,
            "blocker_count": len(unique),
            "blockers": unique,
            "config_path": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
            "record_count": 0,
            "source_count_cap": int(policy["authorization"]["source_count_cap"]),
            "protected_holdout_access": 0,
        }
    try:
        audit = _read_json(paths["intake_audit"])
        blockers.extend(_schema_blockers(audit, INTAKE_SCHEMA_PATH, "audit_schema"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"audit:{exc}")
        audit = {}
    if not snapshot_exists:
        blockers.append("missing:metadata_snapshot")
        snapshot: dict[str, Any] = {}
    else:
        try:
            snapshot = _read_json(paths["metadata_snapshot"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"metadata_snapshot:{exc}")
            snapshot = {}
    if not queue_exists:
        blockers.append("missing:source_queue")
        rows: list[dict[str, str]] = []
    else:
        try:
            rows = _read_queue(paths["source_queue"])
        except (OSError, ValueError, TypeError) as exc:
            blockers.append(f"source_queue:{exc}")
            rows = []
    if audit:
        if audit.get("record_count") != len(rows):
            blockers.append("record_count_mismatch")
        if int(audit.get("record_count", -1)) > int(policy["authorization"]["source_count_cap"]):
            blockers.append("source_count_cap_exceeded")
        if audit.get("queue_digest") != _canonical_digest(rows):
            blockers.append("queue_digest_mismatch")
        if audit.get("priority_order") != [row.get("publication_uid") for row in rows]:
            blockers.append("priority_order_mismatch")
        if preregistration and audit.get("preregistration_digest") != preregistration.get("preregistration_digest"):
            blockers.append("preregistration_binding_mismatch")
        if snapshot_exists:
            binding = audit.get("metadata_snapshot_binding", {})
            if binding.get("sha256") != _sha256(paths["metadata_snapshot"]):
                blockers.append("metadata_snapshot_hash_mismatch")
            if binding.get("bytes") != paths["metadata_snapshot"].stat().st_size:
                blockers.append("metadata_snapshot_size_mismatch")
    if snapshot:
        if snapshot.get("schema_version") != "neuronsifter-pubmed-metadata-snapshot-v2":
            blockers.append("metadata_snapshot_schema_version")
        if snapshot.get("retrieval_mode") != "metadata_only":
            blockers.append("metadata_snapshot_retrieval_mode")
        boundaries = snapshot.get("source_boundaries", {})
        for key in (
            "abstracts_retrieved",
            "full_text_retrieved",
            "figures_retrieved",
            "outcome_values_retrieved",
            "p_values_retrieved",
        ):
            if boundaries.get(key) is not False:
                blockers.append(f"metadata_boundary:{key}")
        if boundaries.get("protected_holdout_access") != 0:
            blockers.append("metadata_boundary:protected_holdout_access")
    excluded_uids = set(preregistration.get("exclusions", {}).get("preexposed_publication_uids", []))
    excluded_pmids = set(preregistration.get("exclusions", {}).get("preexposed_pmids", []))
    excluded_dois = set(preregistration.get("exclusions", {}).get("preexposed_dois", []))
    seen_publications: set[str] = set()
    for expected_sequence, row in enumerate(rows, start=1):
        publication_uid = row.get("publication_uid", "")
        if publication_uid in seen_publications:
            blockers.append(f"duplicate_publication:{publication_uid}")
        seen_publications.add(publication_uid)
        if row.get("intake_sequence") != str(expected_sequence):
            blockers.append(f"intake_sequence:{expected_sequence}")
        if row.get("priority_order") != str(expected_sequence):
            blockers.append(f"priority_order:{expected_sequence}")
        if publication_uid in excluded_uids:
            blockers.append(f"preexposed_publication:{publication_uid}")
        if publication_uid.removeprefix("pubmed:") in excluded_pmids:
            blockers.append(f"preexposed_pmid:{publication_uid}")
        if row.get("doi", "").lower() in excluded_dois:
            blockers.append(f"preexposed_doi:{row.get('doi')}")
        for key in (
            "abstracts_retrieved_before_priority",
            "full_text_retrieved_before_priority",
            "figures_retrieved_before_priority",
            "outcome_values_used_for_priority",
            "observed_effect_used_for_priority",
            "p_values_used_for_priority",
        ):
            if row.get(key) != "false":
                blockers.append(f"queue_boundary:{publication_uid}:{key}")
        if row.get("protected_holdout_access") != "0":
            blockers.append(f"queue_boundary:{publication_uid}:protected_holdout_access")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-retrospective-public-cns-ranking-intake-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "config_path": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "record_count": len(rows),
        "source_count_cap": int(policy["authorization"]["source_count_cap"]),
        "queue_digest": audit.get("queue_digest"),
        "protected_holdout_access": audit.get("protected_holdout_access"),
        "next_action": recovery_payload.get("next_action", audit.get("next_action")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
