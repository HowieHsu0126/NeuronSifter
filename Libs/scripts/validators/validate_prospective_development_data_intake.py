"""Validate the bounded, outcome-blind prospective development-data intake."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_prospective_development_data_intake import (  # noqa: E402
    ACCESS_BLOCKED_INVALIDATION_PATH,
    ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH,
    EXCLUSION_SCHEMA_PATH,
    INTAKE_SCHEMA_PATH,
    QUEUE_FIELDS,
    UNIVERSE_FIELDS,
    _assert_parent_state,
    _validate_implementation_freeze,
    _load_policy,
    _output_paths,
    _queue_digest,
    _read_json,
    _universe_digest,
    _validate_preregistration,
    IMPLEMENTATION_FREEZE_PATH,
    IMPLEMENTATION_FREEZE_INVALIDATION_PATH,
    IMPLEMENTATION_FREEZE_INVALIDATION_SCHEMA_PATH,
    IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH,
    IMPLEMENTATION_FREEZE_V2_INVALIDATION_SCHEMA_PATH,
    TERMINAL_INTAKE_STATUSES,
)
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


VALIDATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_data_intake_validation.schema.json"
)


def _read_tsv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"TSV columns drifted: {path}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def _schema_blockers(
    payload: Mapping[str, Any], schema_path: Path, prefix: str
) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{prefix}:{list(error.absolute_path)}:{error.message}" for error in errors]


def validate(*, write_output: bool = False) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    blockers: list[str] = []
    try:
        _assert_parent_state(policy)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"parent_state:{exc}")
    try:
        preregistration = _read_json(paths["preregistration_manifest"])
        _validate_preregistration(preregistration, policy)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"preregistration:{exc}")
        preregistration = {}
    try:
        requirements = yaml.safe_load(
            paths["role_requirements_snapshot"].read_text(encoding="utf-8")
        )
        if requirements != {
            "prospective_development_role_requirements": policy["role_requirements"]
        }:
            blockers.append("role_requirements_snapshot")
    except (OSError, TypeError, yaml.YAMLError) as exc:
        blockers.append(f"role_requirements:{exc}")
    try:
        exclusion = _read_json(paths["exclusion_snapshot"])
        blockers.extend(
            _schema_blockers(exclusion, EXCLUSION_SCHEMA_PATH, "exclusion_schema")
        )
        if len(exclusion.get("evaluation_context_uids", [])) != 6:
            blockers.append("six_context_count")
        if exclusion.get("six_context_outcome_access") != 0:
            blockers.append("six_context_outcome_access")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"exclusion_snapshot:{exc}")
        exclusion = {}
    terminal_intake_exists = False
    if paths["intake_audit"].is_file():
        try:
            terminal_intake_exists = (
                _read_json(paths["intake_audit"]).get("status")
                in TERMINAL_INTAKE_STATUSES
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            terminal_intake_exists = False
    try:
        implementation_freeze = _read_json(IMPLEMENTATION_FREEZE_PATH)
        _validate_implementation_freeze(
            implementation_freeze,
            preregistration,
            verify_current_implementation=not terminal_intake_exists,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"implementation_freeze:{exc}")
    try:
        invalidation = _read_json(IMPLEMENTATION_FREEZE_INVALIDATION_PATH)
        blockers.extend(
            _schema_blockers(
                invalidation,
                IMPLEMENTATION_FREEZE_INVALIDATION_SCHEMA_PATH,
                "implementation_invalidation_schema",
            )
        )
        if invalidation.get("metadata_discovery_started") is not False:
            blockers.append("v1_invalidation_after_metadata_access")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"implementation_invalidation:{exc}")
    try:
        retry_invalidation = _read_json(ACCESS_BLOCKED_INVALIDATION_PATH)
        blockers.extend(
            _schema_blockers(
                retry_invalidation,
                ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH,
                "access_blocked_invalidation_schema",
            )
        )
        if retry_invalidation.get("source_asset_content_retrieved") is not False:
            blockers.append("access_blocked_source_content_retrieved")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"access_blocked_invalidation:{exc}")
    try:
        v2_invalidation = _read_json(IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH)
        blockers.extend(
            _schema_blockers(
                v2_invalidation,
                IMPLEMENTATION_FREEZE_V2_INVALIDATION_SCHEMA_PATH,
                "implementation_v2_invalidation_schema",
            )
        )
        if v2_invalidation.get("metadata_assets_retrieved") != 0:
            blockers.append("v2_invalidation_metadata_assets")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"implementation_v2_invalidation:{exc}")

    audit_exists = paths["intake_audit"].is_file()
    if not audit_exists:
        if any(
            paths[key].exists()
            for key in ("metadata_universe", "source_queue", "metadata_snapshot")
        ):
            blockers.append("partial_outputs_without_intake_audit")
        unique = sorted(set(blockers))
        result = {
            "schema_version": "neuronsifter-prospective-development-data-intake-validation-v1",
            "status": "preregistered" if not unique else "blocked",
            "passed": not unique,
            "blocker_count": len(unique),
            "blockers": unique,
            "intake_status": None,
            "repository_family_count": len(policy["repository_families"]),
            "metadata_asset_count": 0,
            "metadata_eligible_count": None,
            "queue_lineage_count": 0,
            "queue_digest": None,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
            "validation_host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "validation_scope": [
                "preregistration_digest",
                "implementation_freeze",
                "role_requirements_freeze",
                "six_context_exclusion_snapshot",
                "outcome_access_guard",
            ],
        }
        schema_errors = _schema_blockers(
            result, VALIDATION_SCHEMA_PATH, "validation_schema"
        )
        if schema_errors:
            raise ValueError("; ".join(schema_errors))
        if write_output:
            write_json(paths["validation_audit"], result)
        return result

    try:
        audit = _read_json(paths["intake_audit"])
        blockers.extend(_schema_blockers(audit, INTAKE_SCHEMA_PATH, "intake_schema"))
        universe_rows = _read_tsv(paths["metadata_universe"], UNIVERSE_FIELDS)
        queue_rows = _read_tsv(paths["source_queue"], QUEUE_FIELDS)
        snapshot = _read_json(paths["metadata_snapshot"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        blockers.append(f"intake_artifacts:{exc}")
        audit, snapshot, universe_rows, queue_rows = {}, {}, [], []

    if audit:
        if audit.get("metadata_asset_count") != len(universe_rows):
            blockers.append("metadata_asset_count")
        if audit.get("queue_lineage_count") != len(queue_rows):
            blockers.append("queue_lineage_count")
        if audit.get("metadata_universe_digest") != _universe_digest(
            universe_rows, preregistration
        ):
            blockers.append("metadata_universe_digest")
        if audit.get("queue_digest") != _queue_digest(queue_rows, preregistration):
            blockers.append("queue_digest")
        if len(universe_rows) > int(policy["authorization"]["metadata_asset_cap"]):
            blockers.append("metadata_asset_cap")
        if len(queue_rows) > int(
            policy["authorization"]["preregistered_source_lineage_cap"]
        ):
            blockers.append("queue_lineage_cap")
        if audit.get("preregistration_digest") != preregistration.get(
            "preregistration_digest"
        ):
            blockers.append("preregistration_binding")
        if audit.get("queue_frozen_before_new_development_outcome_access") is not True:
            blockers.append("queue_not_frozen_before_outcome_access")
        if (
            audit.get("development_numeric_outcomes_retrieved_before_queue_freeze")
            is not False
        ):
            blockers.append("development_outcome_access_before_freeze")
        if audit.get("six_context_outcome_access") != 0:
            blockers.append("audit_six_context_outcome_access")
        if (
            audit.get("training_authorized") is not False
            or audit.get("gpu_authorized") is not False
        ):
            blockers.append("downstream_authority_open")
    if snapshot:
        if snapshot.get("retrieval_mode") != "repository_metadata_only":
            blockers.append("metadata_snapshot_mode")
        boundaries = snapshot.get("source_boundaries", {})
        if boundaries != {
            "source_asset_content_retrieved": False,
            "development_numeric_outcomes_retrieved": False,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        }:
            blockers.append("metadata_snapshot_boundaries")
    eligible_assets = {
        row["asset_uid"]
        for row in universe_rows
        if row["eligibility_status"] == "eligible"
    }
    queue_assets = [row["asset_uid"] for row in queue_rows]
    if not set(queue_assets).issubset(eligible_assets):
        blockers.append("queue_contains_ineligible_asset")
    if len(queue_assets) != len(set(queue_assets)):
        blockers.append("duplicate_queue_asset")
    queue_lineages = [row["lineage_uid"] for row in queue_rows]
    if len(queue_lineages) != len(set(queue_lineages)):
        blockers.append("duplicate_queue_lineage")
    for index, row in enumerate(queue_rows, start=1):
        if row["queue_sequence"] != str(index) or row["queue_priority"] != str(index):
            blockers.append(f"queue_order:{index}")
        if row["source_asset_content_retrieved_before_priority"] != "false":
            blockers.append(f"queue_content_access:{row['asset_uid']}")
        if row["development_outcomes_retrieved_before_priority"] != "false":
            blockers.append(f"queue_development_outcome_access:{row['asset_uid']}")
        if (
            row["six_context_outcome_access"] != "0"
            or row["protected_holdout_access"] != "0"
        ):
            blockers.append(f"queue_protected_access:{row['asset_uid']}")
    unique = sorted(set(blockers))
    result = {
        "schema_version": "neuronsifter-prospective-development-data-intake-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "intake_status": audit.get("status"),
        "repository_family_count": len(policy["repository_families"]),
        "metadata_asset_count": len(universe_rows),
        "metadata_eligible_count": audit.get("metadata_eligible_count"),
        "queue_lineage_count": len(queue_rows),
        "queue_digest": audit.get("queue_digest"),
        "six_context_outcome_access": audit.get("six_context_outcome_access", 0),
        "protected_holdout_access": audit.get("protected_holdout_access", 0),
        "validation_host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_scope": [
            "preregistration_digest",
            "historical_generation_implementation_freeze",
            "terminal_intake_reentry_guard",
            "metadata_universe_digest",
            "frozen_lineage_queue",
            "candidate_role_capability_metadata",
            "six_context_exclusion",
            "outcome_access_guard",
        ],
    }
    schema_errors = _schema_blockers(
        result, VALIDATION_SCHEMA_PATH, "validation_schema"
    )
    if schema_errors:
        raise ValueError("; ".join(schema_errors))
    if write_output:
        write_json(paths["validation_audit"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()
    result = validate(write_output=args.write_output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
