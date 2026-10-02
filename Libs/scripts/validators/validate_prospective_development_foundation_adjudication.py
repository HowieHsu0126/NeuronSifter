"""Validate the terminal prospective development-foundation adjudication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_prospective_development_foundation_adjudication import (  # noqa: E402
    AUTHORIZATION_SCHEMA_PATH,
    DECISION_SCHEMA_PATH,
    RECOVERY_SCHEMA_PATH,
    ROLE_SCHEMA_PATH,
    TAXONOMY_SCHEMA_PATH,
    VALIDATION_SCHEMA_PATH,
    build,
)
from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    _binding,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
VALIDATOR_PATH = (
    REPO_ROOT
    / "Libs/scripts/validators/validate_prospective_development_foundation_adjudication.py"
)
TEST_PATH = (
    REPO_ROOT / "Libs/tests/test_prospective_development_foundation_adjudication.py"
)
VALIDATION_WRAPPER_PATH = (
    REPO_ROOT
    / "Libs/exp/evaluation/run_prospective_development_foundation_adjudication.sbatch"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _schema_blockers(
    payload: Mapping[str, Any], schema_path: Path, prefix: str
) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{prefix}:{list(error.absolute_path)}:{error.message}" for error in errors]


def _check_bindings(
    bindings: Mapping[str, Any], prefix: str, blockers: list[str]
) -> None:
    for name, binding in bindings.items():
        path_value = binding.get("path") if isinstance(binding, dict) else None
        if not path_value:
            blockers.append(f"{prefix}_binding_missing:{name}")
            continue
        path = (REPO_ROOT / str(path_value)).resolve()
        if not path.is_file() or _binding(path) != binding:
            blockers.append(f"{prefix}_binding_drift:{name}")


def validate(*, recompute: bool = False, write_output: bool = False) -> dict[str, Any]:
    policy = load_config(CONFIG_PATH)["prospective_development_foundation_adjudication"]
    outputs = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    blockers: list[str] = []
    try:
        recovery = _read_json(outputs["source_recovery_audit"])
        taxonomy = _read_json(outputs["failure_taxonomy"])
        assignment = _read_json(outputs["role_assignment"])
        decision = _read_json(outputs["foundation_decision"])
        authorization = _read_json(outputs["conditional_authorization"])
        report = outputs["failure_analysis_report"].read_text(encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "neuronsifter-prospective-development-foundation-validation-v1",
            "status": "blocked",
            "passed": False,
            "blocker_count": 1,
            "blockers": [f"artifact_read:{exc}"],
        }
    for payload, schema_path, prefix in (
        (recovery, RECOVERY_SCHEMA_PATH, "recovery_schema"),
        (taxonomy, TAXONOMY_SCHEMA_PATH, "taxonomy_schema"),
        (assignment, ROLE_SCHEMA_PATH, "assignment_schema"),
        (decision, DECISION_SCHEMA_PATH, "decision_schema"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization_schema"),
    ):
        blockers.extend(_schema_blockers(payload, schema_path, prefix))

    if recovery.get("status") != "NOT_RUN_QUEUE_EMPTY":
        blockers.append("source_recovery_status")
    if recovery.get("metadata_asset_count") != 153:
        blockers.append("source_recovery_metadata_count")
    if (
        recovery.get("metadata_eligible_count") != 0
        or recovery.get("queue_lineage_count") != 0
    ):
        blockers.append("source_recovery_queue_not_empty")
    if (
        recovery.get("source_payloads_recovered") != 0
        or recovery.get("empirical_action_response_units") != 0
    ):
        blockers.append("source_recovery_executed")

    rejection_counts = taxonomy.get("rejection_reason_counts", {})
    if sum(int(value) for value in rejection_counts.values()) != 153:
        blockers.append("failure_taxonomy_total")
    if taxonomy.get("status") != "bounded_metadata_universe_no_eligible_lineage":
        blockers.append("failure_taxonomy_status")
    if any(taxonomy.get("capability_resolution", {}).values()):
        blockers.append("capability_resolution_overclaim")

    if assignment.get("status") != "NOT_ASSIGNABLE_QUEUE_EMPTY":
        blockers.append("role_assignment_status")
    if assignment.get("assignment_performed") is not False:
        blockers.append("role_assignment_performed")
    if any(
        record.get("lineage_uids")
        or record.get("source_count")
        or record.get("unit_count")
        for record in assignment.get("roles", {}).values()
    ):
        blockers.append("role_assignment_nonempty")

    if decision.get("status") != "PUBLIC_PROSPECTIVE_DEVELOPMENT_DATA_INSUFFICIENT":
        blockers.append("foundation_decision_status")
    for key in (
        "source_payload_recovery",
        "empirical_action_response_materialization",
    ):
        if decision.get("work_package_status", {}).get(key) != "NOT_RUN_QUEUE_EMPTY":
            blockers.append(f"work_package_status:{key}")
    if decision.get("work_package_status", {}).get("four_role_assignment") != (
        "NOT_ASSIGNABLE_QUEUE_EMPTY"
    ):
        blockers.append("work_package_status:four_role_assignment")
    if (
        decision.get("work_package_status", {}).get("terminal_intake_reentry_guard")
        != "ENFORCED_NO_SECOND_INTAKE"
    ):
        blockers.append("work_package_status:terminal_intake_reentry_guard")
    if not any(
        "screening candidates only" in str(limitation)
        for limitation in decision.get("interpretation_limitations", [])
    ):
        blockers.append("candidate_role_semantics_missing")
    for key, value in decision.get("execution_authority", {}).items():
        if key in {
            "development_outcome_access",
            "six_context_outcome_access",
            "protected_holdout_access",
        }:
            if value != 0:
                blockers.append(f"decision_authority:{key}")
        elif value is not False:
            blockers.append(f"decision_authority:{key}")
    if authorization.get("status") != "not_issuable":
        blockers.append("authorization_status")
    if authorization.get("allowed_scope"):
        blockers.append("authorization_scope")
    if (
        authorization.get("training_authorized") is not False
        or authorization.get("gpu_authorized") is not False
    ):
        blockers.append("authorization_training_or_gpu")
    if decision["status"] not in report or "153" not in report:
        blockers.append("failure_analysis_report")

    _check_bindings(recovery.get("artifact_bindings", {}), "recovery", blockers)
    _check_bindings(taxonomy.get("artifact_bindings", {}), "taxonomy", blockers)
    _check_bindings(decision.get("artifact_bindings", {}), "decision", blockers)
    _check_bindings(
        {"decision": authorization.get("decision_binding", {})},
        "authorization",
        blockers,
    )

    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            for label, current, expected in (
                ("recovery", recovery, rebuilt["source_recovery"]),
                ("taxonomy", taxonomy, rebuilt["failure_taxonomy"]),
                ("assignment", assignment, rebuilt["role_assignment"]),
                ("decision", decision, rebuilt["foundation_decision"]),
                (
                    "authorization",
                    authorization,
                    rebuilt["conditional_authorization"],
                ),
            ):
                if current != expected:
                    blockers.append(f"recompute:{label}")
            if report != rebuilt["failure_analysis_report"]:
                blockers.append("recompute:failure_analysis_report")
    unique = sorted(set(blockers))
    result = {
        "schema_version": "neuronsifter-prospective-development-foundation-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "foundation_status": decision.get("status"),
        "metadata_asset_count": recovery.get("metadata_asset_count", 0),
        "queue_lineage_count": recovery.get("queue_lineage_count", 0),
        "source_payloads_recovered": recovery.get("source_payloads_recovered", 0),
        "formal_roles_assigned": sum(
            record.get("source_count", 0)
            for record in assignment.get("roles", {}).values()
        ),
        "six_context_outcome_access": decision.get("execution_authority", {}).get(
            "six_context_outcome_access"
        ),
        "training_authorized": authorization.get("training_authorized"),
        "gpu_authorized": authorization.get("gpu_authorized"),
        "protected_holdout_access": authorization.get("protected_holdout_access"),
        "recompute": recompute,
        "validation_host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_scope": [
            "schema_validation",
            "artifact_hash_bindings",
            "semantic_recompute" if recompute else "persisted_artifact_consistency",
            "bounded_intake_exhaustion",
            "empty_queue_hard_stop",
            "terminal_intake_reentry_guard",
            "four_role_assignment_gate",
            "outcome_access_guard",
        ],
        "implementation_bindings": {
            "validator": _binding(VALIDATOR_PATH),
            "tests": _binding(TEST_PATH),
            "validation_wrapper": _binding(VALIDATION_WRAPPER_PATH),
        },
    }
    schema_errors = _schema_blockers(
        result, VALIDATION_SCHEMA_PATH, "validation_schema"
    )
    if schema_errors:
        raise ValueError("; ".join(schema_errors))
    if write_output:
        write_json(outputs["validation_audit"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute, write_output=args.write_output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
