"""Validate zero-shot method authority and the checkpoint hard-stop package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    _binding,
)
from Libs.scripts.builders.build_zero_shot_method_authority import (  # noqa: E402
    AUTHORITY_SCHEMA_PATH,
    AUTHORIZATION_SCHEMA_PATH,
    DECISION_SCHEMA_PATH,
    INDEPENDENCE_SCHEMA_PATH,
    build,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
VALIDATION_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/zero_shot_method_authority_validation.schema.json"
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
    policy = load_config(CONFIG_PATH)["zero_shot_method_authority"]
    outputs = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    blockers: list[str] = []
    try:
        authority = _read_json(outputs["method_authority"])
        independence = _read_json(outputs["independence_audit"])
        decision = _read_json(outputs["executability_decision"])
        authorization = _read_json(outputs["conditional_authorization"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "neuronsifter-zero-shot-method-authority-validation-v1",
            "status": "blocked",
            "passed": False,
            "blocker_count": 1,
            "blockers": [f"artifact_read:{exc}"],
        }
    for payload, schema_path, prefix in (
        (authority, AUTHORITY_SCHEMA_PATH, "authority_schema"),
        (independence, INDEPENDENCE_SCHEMA_PATH, "independence_schema"),
        (decision, DECISION_SCHEMA_PATH, "decision_schema"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization_schema"),
    ):
        blockers.extend(_schema_blockers(payload, schema_path, prefix))

    if authority.get("status") != "NO_FORMAL_ZERO_SHOT_CHECKPOINT":
        blockers.append("method_authority_terminal_status")
    candidates = authority.get("candidate_authority_matrix", [])
    if len(candidates) != 2:
        blockers.append("candidate_authority_count")
    if any(
        not record.get("checkpoint", {}).get("hash_matches") for record in candidates
    ):
        blockers.append("candidate_checkpoint_integrity")
    if any(
        record.get("prospective_zero_shot_eligible") is not False
        for record in candidates
    ):
        blockers.append("candidate_formal_eligibility")
    if any(
        record.get("selection", {}).get("lineage_complete") is not False
        for record in candidates
    ):
        blockers.append("candidate_selection_lineage")
    if any(
        record.get("training", {}).get("lineage_complete") is not False
        for record in candidates
    ):
        blockers.append("candidate_training_lineage")
    historical = authority.get("historical_freeze", {})
    if historical.get("current_matches_expected") is not False:
        blockers.append("historical_config_unexpected_match")
    if historical.get("recoverable_from_bounded_authorized_sources") is not False:
        blockers.append("historical_config_recovery_overclaim")
    if historical.get("global_absence_claimed") is not False:
        blockers.append("historical_config_global_absence_overclaim")
    if historical.get("new_prospective_method_version_created") is not False:
        blockers.append("new_method_version_created_after_hard_stop")
    hard_stop = authority.get("hard_stop", {})
    if hard_stop.get("triggered") is not True:
        blockers.append("checkpoint_hard_stop_not_triggered")
    for key in (
        "deterministic_intervention_adapter",
        "cpu_inference_uncertainty_abstention",
        "baseline_implementation_binding",
        "non_evaluation_end_to_end_qualification",
    ):
        if hard_stop.get(key) != "NOT_RUN_HARD_STOP":
            blockers.append(f"downstream_work_executed:{key}")
    if independence.get("status") != "formal_independence_not_proven":
        blockers.append("independence_status")
    if independence.get("formal_independence_proven") is not False:
        blockers.append("independence_overclaim")
    if independence.get("bounded_exact_overlap_check", {}).get("match_count") != 0:
        blockers.append("exact_evaluation_overlap")
    if independence.get("six_context_outcome_access") != 0:
        blockers.append("six_context_outcome_access")
    if decision.get("status") != "ZERO_SHOT_PROTOCOL_REMAINS_NON_EXECUTABLE":
        blockers.append("executability_decision")
    if decision.get("terminal_blocker") != "NO_FORMAL_ZERO_SHOT_CHECKPOINT":
        blockers.append("executability_terminal_blocker")
    if (
        decision.get("work_package_status", {}).get("formal_method_identity")
        != "COMPLETE_NEGATIVE"
    ):
        blockers.append("formal_method_identity_status")
    for key, value in decision.get("execution_authority", {}).items():
        if key in {"six_context_outcome_access", "protected_holdout_access"}:
            if value != 0:
                blockers.append(f"decision_authority:{key}")
        elif value is not False:
            blockers.append(f"decision_authority:{key}")
    if authorization.get("status") != "not_issuable":
        blockers.append("authorization_status")
    if authorization.get("protocol_executable") is not False:
        blockers.append("authorization_protocol_executable")
    if authorization.get("execution_authorized") is not False:
        blockers.append("authorization_execution_authorized")
    if authorization.get("allowed_future_scope"):
        blockers.append("authorization_allowed_scope")
    if authorization.get("protected_holdout_access") != 0:
        blockers.append("authorization_protected_holdout_access")

    _check_bindings(authority.get("artifact_bindings", {}), "authority", blockers)
    _check_bindings(independence.get("artifact_bindings", {}), "independence", blockers)
    _check_bindings(decision.get("artifact_bindings", {}), "decision", blockers)
    decision_binding = authorization.get("decision_binding", {})
    _check_bindings({"decision": decision_binding}, "authorization", blockers)

    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            for label, current, expected in (
                ("authority", authority, rebuilt["method_authority"]),
                ("independence", independence, rebuilt["independence_audit"]),
                ("decision", decision, rebuilt["executability_decision"]),
                (
                    "authorization",
                    authorization,
                    rebuilt["conditional_authorization"],
                ),
            ):
                if current != expected:
                    blockers.append(f"recompute:{label}")
    unique = sorted(set(blockers))
    result = {
        "schema_version": "neuronsifter-zero-shot-method-authority-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "method_authority_status": authority.get("status"),
        "protocol_status": decision.get("status"),
        "terminal_blocker": decision.get("terminal_blocker"),
        "candidate_count": len(candidates),
        "eligible_candidate_count": sum(
            record.get("prospective_zero_shot_eligible") is True
            for record in candidates
        ),
        "downstream_work_executed": False,
        "six_context_outcome_access": independence.get("six_context_outcome_access"),
        "execution_authorized": authorization.get("execution_authorized"),
        "protected_holdout_access": authorization.get("protected_holdout_access"),
        "recompute": recompute,
        "validation_host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_scope": [
            "schema_validation",
            "artifact_hash_bindings",
            "semantic_recompute" if recompute else "persisted_artifact_consistency",
            "hard_stop_enforcement",
            "outcome_access_guard",
        ],
    }
    validation_schema_errors = _schema_blockers(
        result, VALIDATION_SCHEMA_PATH, "validation_schema"
    )
    if validation_schema_errors:
        raise ValueError("; ".join(validation_schema_errors))
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
