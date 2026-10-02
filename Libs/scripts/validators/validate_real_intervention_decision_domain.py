"""Validate the fail-closed real-intervention decision-domain qualification."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _binding_matches(binding: Mapping[str, object]) -> bool:
    path = _repo_path(str(binding["path"]))
    return bool(
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and sha256_file(path) == str(binding["sha256"])
    )


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Decision-domain validation is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["decision_domain_qualification"]
    if not isinstance(policy, Mapping):
        raise TypeError("Decision-domain qualification policy must be a mapping")
    outputs = {name: _repo_path(str(path)) for name, path in policy["outputs"].items()}
    summary = _json(outputs["domain_summary"])
    summary_schema = _json(_repo_path(str(policy["summary_schema_path"])))
    split_audit = _json(outputs["split_audit"])
    context_rows = read_tsv(outputs["context_group_qualification"])
    episode_rows = read_tsv(_repo_path(str(policy["input_episode_catalog"])))
    action_rows = read_tsv(_repo_path(str(policy["input_action_catalog"])))
    bridge_quality = _json(_repo_path(str(policy["input_bridge_quality"])))
    outcome_summary = _json(_repo_path(str(policy["input_outcome_sensitivity_summary"])))

    blockers: list[str] = []

    def require(condition: bool, blocker: str) -> None:
        if not condition:
            blockers.append(blocker)

    Draft202012Validator.check_schema(summary_schema)
    for error in sorted(
        Draft202012Validator(summary_schema).iter_errors(summary),
        key=lambda item: list(item.absolute_path),
    ):
        location = ".".join(str(value) for value in error.absolute_path) or "root"
        blockers.append(f"summary_schema_error:{location}:{error.message}")

    allowed_classifications = {str(value) for value in policy["classification_values"]}
    classifications = [str(row["decision_domain_classification"]) for row in context_rows]
    candidate_rows = [row for row in context_rows if row["structural_multi_action_set"] == "True"]
    qualified_rows = [row for row in context_rows if row["qualified_decision_candidate_set"] == "True"]
    split_eligible_rows = [row for row in context_rows if row["split_eligible"] == "True"]
    context_counts = dict(sorted(Counter(classifications).items()))
    candidate_counts = dict(
        sorted(Counter(str(row["decision_domain_classification"]) for row in candidate_rows).items())
    )
    source_context_counts = dict(sorted(Counter(str(row["source_key"]) for row in context_rows).items()))
    source_candidate_counts = dict(sorted(Counter(str(row["source_key"]) for row in candidate_rows).items()))

    decoded_action_members: list[str] = []
    decoded_control_members: list[str] = []
    membership_valid = True
    for row in context_rows:
        try:
            action_members = json.loads(row["structured_action_uids"])
            control_members = json.loads(row["control_uids"])
        except (json.JSONDecodeError, TypeError):
            membership_valid = False
            continue
        if not isinstance(action_members, list) or not isinstance(control_members, list):
            membership_valid = False
            continue
        membership_valid = membership_valid and len(action_members) == int(row["action_count"])
        membership_valid = membership_valid and len(control_members) == int(row["control_uid_count"])
        membership_valid = membership_valid and (int(row["action_count"]) >= 2) == (
            row["structural_multi_action_set"] == "True"
        )
        decoded_action_members.extend(str(value) for value in action_members)
        decoded_control_members.extend(str(value) for value in control_members)

    input_bindings = summary.get("input_bindings")
    implementation_bindings = summary.get("implementation_bindings")
    output_bindings = summary.get("output_bindings")
    require(isinstance(input_bindings, Mapping), "input_bindings_missing")
    require(isinstance(implementation_bindings, Mapping), "implementation_bindings_missing")
    require(isinstance(output_bindings, Mapping), "output_bindings_missing")
    for prefix, bindings in (
        ("input", input_bindings),
        ("implementation", implementation_bindings),
        ("output", output_bindings),
    ):
        if isinstance(bindings, Mapping):
            for name, binding in bindings.items():
                require(
                    isinstance(binding, Mapping) and _binding_matches(binding),
                    f"{prefix}_binding_mismatch:{name}",
                )

    require(summary.get("schema_version") == policy["schema_version"], "schema_version_mismatch")
    require(summary.get("status") == policy["blocked_status"], "status_mismatch")
    require(summary.get("gate_resolved") is True, "gate_not_resolved")
    require(summary.get("decision_sensitive_domain_available") is False, "decision_domain_incorrectly_open")
    require(summary.get("model_training_domain_qualified") is False, "training_domain_incorrectly_open")
    require(summary.get("decision_sensitivity_known") is True, "decision_sensitivity_unknown")
    require(int(summary.get("episode_count", -1)) == len(episode_rows) == 7189, "episode_count_mismatch")
    require(int(summary.get("action_count", -1)) == len(action_rows) == 7136, "action_count_mismatch")
    require(int(summary.get("context_group_count", -1)) == len(context_rows), "context_group_count_mismatch")
    require(
        int(summary.get("structural_multi_action_set_count", -1)) == len(candidate_rows),
        "candidate_set_count_mismatch",
    )
    require(
        int(summary.get("single_action_context_group_count", -1)) == len(context_rows) - len(candidate_rows),
        "single_action_group_count_mismatch",
    )
    require(
        int(summary.get("multi_action_episode_count", -1))
        == sum(int(row["episode_count"]) for row in candidate_rows),
        "multi_action_episode_count_mismatch",
    )
    require(sum(int(row["episode_count"]) for row in context_rows) == len(episode_rows), "episode_membership_incomplete")
    require(len({row["context_group_uid"] for row in context_rows}) == len(context_rows), "duplicate_context_group_uid")
    require(membership_valid, "invalid_context_group_membership")
    require(set(decoded_action_members) == {row["structured_action_uid"] for row in action_rows}, "action_coverage_mismatch")
    require(len(decoded_control_members) == len(context_rows), "control_group_cardinality_mismatch")
    require(set(classifications) <= allowed_classifications, "unknown_decision_domain_classification")
    require(summary.get("context_classification_counts") == context_counts, "context_classification_counts_mismatch")
    require(summary.get("candidate_classification_counts") == candidate_counts, "candidate_classification_counts_mismatch")
    require(summary.get("source_context_group_counts") == source_context_counts, "source_context_counts_mismatch")
    require(summary.get("source_candidate_set_counts") == source_candidate_counts, "source_candidate_counts_mismatch")
    require(not qualified_rows and int(summary.get("qualified_decision_candidate_set_count", -1)) == 0, "qualified_set_present")
    require(not split_eligible_rows, "split_eligible_set_present")
    require(not ({"decision_sensitive", "weakly_decision_sensitive"} & set(classifications)), "decision_sensitive_classification_present")
    require(all(row["primary_blocker"] not in {"", "unknown"} for row in context_rows), "unknown_primary_blocker")
    require(all(row["real_outcome_diagnostic_bound"] == "True" for row in context_rows), "outcome_diagnostic_unbound")
    require(all(row["model_predictions_inspected"] == "False" for row in context_rows), "model_prediction_inspection_detected")
    require(all(row["evidence_role"] == "development_only" for row in context_rows), "evidence_role_violation")
    require(all(row["publication_claim_eligible"] == "False" for row in context_rows), "publication_claim_violation")
    require(all(row["protected_holdout_access"] == "0" for row in context_rows), "protected_holdout_violation")
    require(bridge_quality.get("model_training_domain_qualified") is False, "bridge_training_gate_open")
    require(bridge_quality.get("model_predictions_inspected") is False, "bridge_model_results_inspected")
    require(outcome_summary.get("classification_is_based_on_real_outcomes") is True, "outcome_basis_invalid")
    require(outcome_summary.get("model_predictions_inspected") is False, "historical_model_results_inspected")
    require(summary.get("historical_outcome_diagnostics_mutated") is False, "historical_outcomes_mutated")
    require(summary.get("world_action_target_claim_retained") is True, "world_action_target_claim_not_retained")
    require(summary.get("agent_in_twin_target_claim_retained") is True, "agent_in_twin_target_claim_not_retained")
    require(summary.get("coevolution_target_claim_retained") is True, "coevolution_target_claim_not_retained")
    require(summary.get("agent_in_twin_authorized") is False, "agent_in_twin_incorrectly_authorized")
    require(summary.get("coevolution_authorized") is False, "coevolution_incorrectly_authorized")
    require(summary.get("publication_comparison_open") is False, "publication_comparison_incorrectly_open")
    require(summary.get("protected_holdout_access") == 0, "summary_protected_holdout_violation")
    require(split_audit.get("status") == "not_created_no_qualified_candidate_sets", "split_audit_status_mismatch")
    require(split_audit.get("split_files_created") is False, "split_files_incorrectly_created")
    require(split_audit.get("development_splits_frozen") is False, "empty_splits_incorrectly_frozen")
    require(split_audit.get("model_training_authorized") is False, "split_audit_training_authorized")
    require(split_audit.get("protected_holdout_access") == 0, "split_audit_protected_holdout_violation")
    planned_split_paths = {
        name: _repo_path(str(path)) for name, path in policy["planned_split_paths"].items()
    }
    require(not any(path.exists() for path in planned_split_paths.values()), "decision_benchmark_split_file_exists")

    unique_blockers = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-real-intervention-decision-domain-validation-v1",
        "status": "passed" if not unique_blockers else "failed",
        "blocker_count": len(unique_blockers),
        "blockers": unique_blockers,
        "episode_count": len(episode_rows),
        "action_count": len(action_rows),
        "context_group_count": len(context_rows),
        "structural_multi_action_set_count": len(candidate_rows),
        "qualified_decision_candidate_set_count": len(qualified_rows),
        "model_training_authorized": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def main() -> int:
    result = validate()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
