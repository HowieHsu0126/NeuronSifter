"""Validate prospective development-domain authority and hard-stop artifacts."""

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

from Libs.scripts.builders.build_prospective_development_domain import (  # noqa: E402
    AUTHORITY_SCHEMA_PATH,
    AUTHORIZATION_SCHEMA_PATH,
    DECISION_SCHEMA_PATH,
    EXCLUSION_SCHEMA_PATH,
    PREREGISTRATION_SCHEMA_PATH,
    VALIDATION_SCHEMA_PATH,
    build,
)
from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    _binding,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
VALIDATOR_PATH = (
    REPO_ROOT / "Libs/scripts/validators/validate_prospective_development_domain.py"
)
TEST_PATH = REPO_ROOT / "Libs/tests/test_prospective_development_domain.py"
VALIDATION_WRAPPER_PATH = (
    REPO_ROOT
    / "Libs/exp/evaluation/run_prospective_development_domain_validation.sbatch"
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
    policy = load_config(CONFIG_PATH)["prospective_development_domain"]
    outputs = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    blockers: list[str] = []
    try:
        authority = _read_json(outputs["development_data_authority"])
        exclusion = _read_json(outputs["semantic_exclusion_audit"])
        preregistration = _read_json(outputs["method_preregistration"])
        decision = _read_json(outputs["reconstruction_decision"])
        authorization = _read_json(outputs["conditional_authorization"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "neuronsifter-prospective-development-domain-validation-v1",
            "status": "blocked",
            "passed": False,
            "blocker_count": 1,
            "blockers": [f"artifact_read:{exc}"],
        }
    for payload, schema_path, prefix in (
        (authority, AUTHORITY_SCHEMA_PATH, "authority_schema"),
        (exclusion, EXCLUSION_SCHEMA_PATH, "exclusion_schema"),
        (preregistration, PREREGISTRATION_SCHEMA_PATH, "preregistration_schema"),
        (decision, DECISION_SCHEMA_PATH, "decision_schema"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization_schema"),
    ):
        blockers.extend(_schema_blockers(payload, schema_path, prefix))

    if authority.get("status") != "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN":
        blockers.append("development_domain_terminal_status")
    if authority.get("four_role_partition_possible") is not False:
        blockers.append("four_role_partition_overclaim")
    if authority.get("training_authorized") is not False:
        blockers.append("training_authority_open")
    if authority.get("new_data_intake_authorized") is not False:
        blockers.append("new_data_intake_authority_open")
    if authority.get("six_context_outcome_access") != 0:
        blockers.append("authority_six_context_outcome_access")
    role_feasibility = authority.get("role_feasibility", {})
    if set(role_feasibility) != set(policy["required_roles"]):
        blockers.append("required_role_set")
    if any(record.get("status") == "qualified" for record in role_feasibility.values()):
        blockers.append("unexpected_qualified_development_role")
    modalities = authority.get("target_modality_coverage", {})
    if (
        modalities.get("protein_biologic_delivery", {}).get("qualified_action_count")
        != 0
    ):
        blockers.append("protein_biologic_domain_count")
    if (
        modalities.get("small_molecule_compound", {}).get(
            "action_complete_dynamic_domain"
        )
        is not False
    ):
        blockers.append("compound_dynamic_domain_overclaim")

    if exclusion.get("status") != "failed_incomplete_semantic_independence":
        blockers.append("semantic_exclusion_status")
    if exclusion.get("exact_exclusion", {}).get("match_count") != 0:
        blockers.append("exact_evaluation_identifier_overlap")
    if (
        exclusion.get("exact_exclusion", {}).get("complete_for_all_development_assets")
        is not False
    ):
        blockers.append("exact_exclusion_scope_overclaim")
    if (
        exclusion.get("semantic_exclusion", {}).get("action_family_independence")
        != "not_resolved_compound_overlap"
    ):
        blockers.append("compound_semantic_overlap")
    if (
        exclusion.get("semantic_exclusion", {}).get(
            "protein_biologic_development_domain_present"
        )
        is not False
    ):
        blockers.append("protein_biologic_semantic_domain")
    if exclusion.get("six_context_outcome_access") != 0:
        blockers.append("exclusion_six_context_outcome_access")

    if preregistration.get("status") != "NOT_ISSUED_TRAINING_DOMAIN_UNQUALIFIED":
        blockers.append("method_preregistration_status")
    if preregistration.get("method_uid") is not None:
        blockers.append("method_uid_created_without_domain")
    if preregistration.get("training_authorized") is not False:
        blockers.append("preregistration_training_authority")
    if any(
        value is not None for value in preregistration.get("freeze_fields", {}).values()
    ):
        blockers.append("method_freeze_fields_fabricated")

    if decision.get("status") != "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN":
        blockers.append("reconstruction_terminal_status")
    if decision.get("terminal_blocker") != "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN":
        blockers.append("reconstruction_terminal_blocker")
    for key in (
        "structured_multimodal_world_action_compiler",
        "formal_training_selection_calibration",
        "formal_method_freeze",
        "non_evaluation_end_to_end_qualification",
    ):
        if decision.get("work_package_status", {}).get(key) != "NOT_RUN_HARD_STOP":
            blockers.append(f"downstream_work_executed:{key}")
    for key, value in decision.get("execution_authority", {}).items():
        if key in {"six_context_outcome_access", "protected_holdout_access"}:
            if value != 0:
                blockers.append(f"decision_authority:{key}")
        elif value is not False:
            blockers.append(f"decision_authority:{key}")

    if authorization.get("status") != "not_issuable":
        blockers.append("authorization_status")
    if authorization.get("development_domain_qualified") is not False:
        blockers.append("authorization_development_domain")
    if authorization.get("method_preregistered") is not False:
        blockers.append("authorization_method_preregistered")
    if authorization.get("training_authorized") is not False:
        blockers.append("authorization_training")
    if authorization.get("gpu_authorized") is not False:
        blockers.append("authorization_gpu")
    if authorization.get("allowed_scope"):
        blockers.append("authorization_allowed_scope")
    if authorization.get("six_context_outcome_access") != 0:
        blockers.append("authorization_six_context_outcome_access")

    _check_bindings(authority.get("artifact_bindings", {}), "authority", blockers)
    _check_bindings(exclusion.get("artifact_bindings", {}), "exclusion", blockers)
    _check_bindings(decision.get("artifact_bindings", {}), "decision", blockers)
    _check_bindings(
        {"authority": preregistration.get("authority_binding", {})},
        "preregistration",
        blockers,
    )
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
                ("authority", authority, rebuilt["development_data_authority"]),
                ("exclusion", exclusion, rebuilt["semantic_exclusion_audit"]),
                ("preregistration", preregistration, rebuilt["method_preregistration"]),
                ("decision", decision, rebuilt["reconstruction_decision"]),
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
        "schema_version": "neuronsifter-prospective-development-domain-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "development_domain_status": authority.get("status"),
        "reconstruction_status": decision.get("status"),
        "terminal_blocker": decision.get("terminal_blocker"),
        "four_role_partition_possible": authority.get("four_role_partition_possible"),
        "downstream_work_executed": False,
        "six_context_outcome_access": authority.get("six_context_outcome_access"),
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
            "four_role_partition_gate",
            "semantic_exclusion_guard",
            "hard_stop_enforcement",
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
