"""Validate the fail-closed prospective protocol scientific review package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    AUTHORIZATION_SCHEMA_PATH,
    CLAIM_SCHEMA_PATH,
    DECISION_SCHEMA_PATH,
    EVALUATION_MATRIX_FIELDS,
    METHOD_SCHEMA_PATH,
    REVIEW_SCHEMA_PATH,
    TOOL_SCHEMA_PATH,
    _binding,
    build,
)
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


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


def _read_matrix(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    blockers: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EVALUATION_MATRIX_FIELDS:
            blockers.append("evaluation_matrix_header")
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    return rows, blockers


def _semantic_equal(
    current: Any, rebuilt: Any, label: str, blockers: list[str]
) -> None:
    if current != rebuilt:
        blockers.append(f"recompute:{label}")


def validate(*, recompute: bool = False) -> dict[str, Any]:
    policy = load_config(CONFIG_PATH)["abstention_selective_risk_contract"]
    outputs = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    blockers: list[str] = []
    try:
        review = _read_json(outputs["scientific_review"])
        method = _read_json(outputs["method_manifest"])
        matrix_rows, matrix_blockers = _read_matrix(outputs["evaluation_matrix"])
        tool = _read_json(outputs["tool_headroom_audit"])
        claim = _read_json(outputs["claim_to_evidence_matrix"])
        decision = _read_json(outputs["go_no_go_decision"])
        authorization = _read_json(outputs["conditional_authorization"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "neuronsifter-prospective-protocol-scientific-review-validation-v1",
            "status": "blocked",
            "passed": False,
            "blocker_count": 1,
            "blockers": [f"artifact_read:{exc}"],
        }
    blockers.extend(matrix_blockers)
    for payload, schema_path, prefix in (
        (review, REVIEW_SCHEMA_PATH, "review_schema"),
        (method, METHOD_SCHEMA_PATH, "method_schema"),
        (tool, TOOL_SCHEMA_PATH, "tool_schema"),
        (claim, CLAIM_SCHEMA_PATH, "claim_schema"),
        (decision, DECISION_SCHEMA_PATH, "decision_schema"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization_schema"),
    ):
        blockers.extend(_schema_blockers(payload, schema_path, prefix))

    expected_positive = set(policy["context_roles"]["positive_ranking"])
    expected_ambiguous = set(policy["context_roles"]["ambiguous_abstention"])
    observed_positive = {
        row["context_uid"]
        for row in matrix_rows
        if row["evaluation_role"] == "positive_ranking"
    }
    observed_ambiguous = {
        row["context_uid"]
        for row in matrix_rows
        if row["evaluation_role"] == "ambiguous_abstention"
    }
    if len(matrix_rows) != 6 or len({row["context_uid"] for row in matrix_rows}) != 6:
        blockers.append("evaluation_matrix_context_count_or_uniqueness")
    if observed_positive != expected_positive:
        blockers.append("evaluation_matrix_positive_roles")
    if observed_ambiguous != expected_ambiguous:
        blockers.append("evaluation_matrix_ambiguous_roles")
    prohibited_matrix_fields = {
        "winner",
        "numeric_outcome",
        "p_value",
        "effect_direction",
    }
    if prohibited_matrix_fields & set(EVALUATION_MATRIX_FIELDS):
        blockers.append("evaluation_matrix_contains_outcome_field")
    for row in matrix_rows:
        if any(
            row[field] != "prohibited"
            for field in (
                "training_role",
                "calibration_role",
                "model_selection_role",
                "threshold_fit_role",
            )
        ):
            blockers.append(f"evaluation_role_leakage:{row['context_uid']}")
        if (
            row["execution_status"] != "NOT_RUN"
            or row["outcome_access_during_review"] != "0"
        ):
            blockers.append(f"evaluation_executed_during_review:{row['context_uid']}")
        if (
            row["evaluation_role"] == "ambiguous_abstention"
            and row["ranking_truth_role"] != "unresolved_no_ranking_truth"
        ):
            blockers.append(f"ambiguous_promoted_to_truth:{row['context_uid']}")

    if method.get("evaluation_independence", {}).get("identifier_overlap_count") != 0:
        blockers.append("method_evaluation_identifier_overlap")
    if (
        method.get("evaluation_independence", {}).get("formal_independence_proven")
        is not False
    ):
        blockers.append("method_independence_overstated")
    if (
        method.get("evaluation_independence", {}).get(
            "evaluation_outcome_access_during_review"
        )
        != 0
    ):
        blockers.append("method_review_outcome_access")
    if not all(
        item.get("hash_matches") for item in method.get("checkpoint_components", [])
    ):
        blockers.append("method_checkpoint_hash")
    if method.get("status") == "complete_and_executable" and method.get(
        "blocking_findings"
    ):
        blockers.append("method_status_blocker_inconsistency")
    if method.get("status") == "incomplete_not_executable" and not method.get(
        "blocking_findings"
    ):
        blockers.append("method_missing_blockers")
    if (
        method.get("abstention_identity", {}).get(
            "evaluation_context_threshold_fit_allowed"
        )
        is not False
    ):
        blockers.append("evaluation_threshold_fit_not_prohibited")
    if tool.get("status") == "TOOL_INFORMATION_HEADROOM_NOT_SUPPORTED":
        if tool.get("formal_agent_in_twin_headline_authorized") is not False:
            blockers.append("agent_headline_open_without_tool_headroom")
    if any(
        context.get("direct_hidden_truth_reveal_allowed") is not False
        for context in tool.get("ambiguous_contexts", [])
    ):
        blockers.append("tool_hidden_truth_reveal")
    if decision.get("decision") != review.get("status"):
        blockers.append("review_decision_mismatch")
    if decision.get("execution_authorized") is not False:
        blockers.append("decision_execution_authority")
    if authorization.get("execution_authorized") is not False:
        blockers.append("authorization_self_authorized")
    if decision.get("decision") == "PROTOCOL_NOT_FORMALLY_EXECUTABLE":
        if authorization.get("status") != "not_issuable":
            blockers.append("authorization_status_for_no_go")
        if authorization.get("authorized_scope"):
            blockers.append("authorization_scope_nonempty_for_no_go")
    authority = review.get("execution_authority", {})
    for key, value in authority.items():
        if key == "protected_holdout_access":
            if value != 0:
                blockers.append("review_protected_holdout_access")
        elif value is not False:
            blockers.append(f"review_authority:{key}")
    if review.get("immutable_parent_state") != {
        "status": "qualified_public_candidates_but_split_insufficient",
        "accepted_records": 36,
        "rejected_records": 7,
        "legal_candidate_sets": 6,
        "independent_public_contexts": 6,
        "independent_decision_sensitive_groups": 2,
        "required_groups": 3,
        "split_files_created": False,
        "leakage_status": "passed",
        "protected_holdout_access": 0,
    }:
        blockers.append("immutable_parent_state")
    for name, binding in review.get("artifact_bindings", {}).items():
        path_value = binding.get("path") if isinstance(binding, dict) else None
        if not path_value:
            blockers.append(f"review_binding_missing:{name}")
            continue
        path = (REPO_ROOT / str(path_value)).resolve()
        if not path.is_file() or _binding(path) != binding:
            blockers.append(f"review_binding_drift:{name}")
    for name, binding in decision.get("evidence_bindings", {}).items():
        path_value = binding.get("path") if isinstance(binding, dict) else None
        if not path_value:
            blockers.append(f"decision_binding_missing:{name}")
            continue
        path = (REPO_ROOT / str(path_value)).resolve()
        if not path.is_file() or _binding(path) != binding:
            blockers.append(f"decision_binding_drift:{name}")
    authorization_binding = authorization.get("decision_binding", {})
    decision_path_value = (
        authorization_binding.get("path")
        if isinstance(authorization_binding, dict)
        else None
    )
    if not decision_path_value:
        blockers.append("authorization_decision_binding_missing")
    else:
        decision_path = (REPO_ROOT / str(decision_path_value)).resolve()
        if (
            not decision_path.is_file()
            or _binding(decision_path) != authorization_binding
        ):
            blockers.append("authorization_decision_binding_drift")

    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            _semantic_equal(
                review, rebuilt["scientific_review"], "scientific_review", blockers
            )
            _semantic_equal(
                method, rebuilt["method_manifest"], "method_manifest", blockers
            )
            current_matrix_text = outputs["evaluation_matrix"].read_text(
                encoding="utf-8"
            )
            _semantic_equal(
                current_matrix_text,
                rebuilt["evaluation_matrix_text"],
                "evaluation_matrix",
                blockers,
            )
            _semantic_equal(
                tool, rebuilt["tool_headroom_audit"], "tool_headroom", blockers
            )
            _semantic_equal(
                claim, rebuilt["claim_to_evidence_matrix"], "claim_matrix", blockers
            )
            _semantic_equal(
                decision, rebuilt["go_no_go_decision"], "go_no_go", blockers
            )
            _semantic_equal(
                authorization,
                rebuilt["conditional_authorization"],
                "conditional_authorization",
                blockers,
            )
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-prospective-protocol-scientific-review-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "protocol_decision": decision.get("decision"),
        "scientific_blocker_count": len(decision.get("blocking_findings", [])),
        "method_status": method.get("status"),
        "tool_headroom_status": tool.get("status"),
        "evaluation_context_count": len(matrix_rows),
        "positive_ranking_context_count": len(observed_positive),
        "ambiguous_context_count": len(observed_ambiguous),
        "execution_authorized": authorization.get("execution_authorized"),
        "protected_holdout_access": authorization.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
