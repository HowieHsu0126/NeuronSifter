"""Build the fail-closed Agent-Twin co-evolution qualification contract audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


DEFAULT_CONFIG = "Libs/configs/experiments/agent_twin_coevolution.yaml"
DEFAULT_SCHEMA = "Libs/configs/schema/agent_twin_coevolution.schema.json"
DEFAULT_OUTPUT = "Output/audits/agent_twin_update_contract.json"
FINAL_GATE = "agent_twin_coevolution_evidence_resolved"
REQUIRED_PREREQUISITES = {
    "public_corpus_frozen",
    "allen_full_landing",
    "intervention_e4_available",
    "strict_sequential_splits_frozen",
    "static_twin_baselines_resolved",
    "static_agent_baselines_resolved",
    "tool_outcome_environment_available",
    "tri_calibration_protocol_frozen",
    "final_holdout_accessed",
}
REQUIRED_BASELINES = {
    "static_twin_static_agent",
    "static_twin_adaptive_agent",
    "evolving_twin_static_agent",
    "naive_joint_online_update",
    "full_replay_twin_static_agent",
    "continual_twin_static_agent",
    "static_twin_bax_agent",
    "static_twin_cost_aware_bo_agent",
    "governed_agent_twin_coevolution",
    "full_information_oracle",
}
REQUIRED_ABLATIONS = {
    "no_twin_debt",
    "no_falsifier",
    "no_evidence_quality_weighting",
    "no_replay",
    "no_support_domain_gate",
    "no_calibration_preservation",
    "no_rollback",
    "no_sentinel_allocation",
    "no_failure_aware_acquisition",
}
REQUIRED_EVALUATION_SYSTEMS = {
    "initial_frozen_twin_initial_agent",
    "initial_frozen_twin_current_agent",
    "current_twin_initial_agent",
    "current_twin_current_agent",
}
REQUIRED_ARTIFACT_KEYS = {
    "input_manifest",
    "action_catalog",
    "evidence_admission",
    "twin_update",
    "agent_update",
    "regression_audit",
    "promotion_or_rollback",
    "report",
}


def repo_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def relative_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_path_value(payload: object, path: str) -> tuple[bool, object | None]:
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def load_contract(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    contract = config.get("agent_twin_coevolution")
    if not isinstance(contract, dict):
        raise ValueError("Agent-Twin co-evolution config is missing agent_twin_coevolution.")
    return contract


def validate_contract(contract: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:10])
        raise ValueError(f"Agent-Twin co-evolution schema validation failed: {detail}")
    if str(contract["final_gate"]) != FINAL_GATE:
        raise ValueError("Agent-Twin final gate does not match the required gate.")

    prerequisite_ids = [str(row["prerequisite_id"]) for row in contract["prerequisites"]]
    if set(prerequisite_ids) != REQUIRED_PREREQUISITES or len(prerequisite_ids) != len(REQUIRED_PREREQUISITES):
        raise ValueError("Agent-Twin prerequisites must be the exact required unique set.")

    update_contract = contract["update_contract"]
    required_update_fields = {
        "round_count",
        "initial_checkpoints",
        "initial_policy",
        "evidence_cutoff",
        "admission_rules",
        "twin_update_trigger",
        "agent_update_trigger",
        "update_order",
        "replay_buffer_policy",
        "retention_policy",
        "regularization",
        "checkpoint_selection",
        "promotion_thresholds",
        "rollback_thresholds",
        "compute_budget",
        "tool_budget",
        "stopping_conditions",
    }
    missing_update_fields = sorted(required_update_fields - set(update_contract))
    if missing_update_fields:
        raise ValueError(f"Agent-Twin update contract is incomplete: {missing_update_fields}")
    if int(update_contract["round_count"]) < 3:
        raise ValueError("Agent-Twin protocol requires at least three real sequential rounds.")
    if str(update_contract["checkpoint_selection"]["split"]) != "validation_only":
        raise ValueError("Agent-Twin checkpoint selection must be validation-only.")
    if not bool(update_contract["checkpoint_selection"]["final_holdout_forbidden"]):
        raise ValueError("Agent-Twin checkpoint selection must forbid final-holdout access.")
    if bool(update_contract["evidence_cutoff"]["final_sequential_holdout_forbidden_for_selection"]) is not True:
        raise ValueError("Agent-Twin evidence cutoff must forbid final sequential holdout selection.")
    if list(update_contract["update_order"]) != [
        "twin_candidate",
        "twin_promotion_or_rollback",
        "agent_candidate",
        "agent_identity_freeze",
    ]:
        raise ValueError("Agent-Twin update order must promote or roll back Twin before Agent update.")

    baseline_ids = [str(row["baseline_id"]) for row in contract["baselines"]]
    if set(baseline_ids) != REQUIRED_BASELINES or len(baseline_ids) != len(REQUIRED_BASELINES):
        raise ValueError("Agent-Twin baseline registry must contain the exact required unique baseline set.")
    oracle = next(row for row in contract["baselines"] if row["baseline_id"] == "full_information_oracle")
    if "evaluation_only" not in str(oracle["compute_tool_budget"]):
        raise ValueError("Full-information oracle must remain evaluation-only.")

    round_protocol = contract["round_protocol"]
    if int(round_protocol["minimum_real_round_count"]) < 3:
        raise ValueError("Round protocol must retain at least three real sequential rounds.")
    if set(round_protocol["required_artifacts"]) != REQUIRED_ARTIFACT_KEYS:
        raise ValueError("Round protocol required artifact keys are incomplete.")
    if set(round_protocol["required_json_fields"]) != REQUIRED_ARTIFACT_KEYS - {"action_catalog", "evidence_admission"}:
        raise ValueError("Round protocol JSON field declarations are incomplete.")
    if set(round_protocol["required_tsv_fields"]) != {"action_catalog", "evidence_admission"}:
        raise ValueError("Round protocol TSV field declarations are incomplete.")

    evaluation = contract["evaluation"]
    systems = {str(value) for value in evaluation["frozen_vs_evolving_systems"]}
    if systems != REQUIRED_EVALUATION_SYSTEMS:
        raise ValueError("Frozen-vs-evolving evaluation must retain the required 2x2 systems.")
    invariants = {str(value) for value in evaluation["comparison_invariants"]}
    if {"identical_available_evidence", "identical_budget", "identical_tool_availability"} - invariants:
        raise ValueError("Frozen-vs-evolving comparisons must share evidence, budget, and tools.")
    if str(evaluation["statistics"]["confidence_interval"]) != "paired_bootstrap_95":
        raise ValueError("Agent-Twin statistics must use paired bootstrap confidence intervals.")

    ablation_ids = [str(row["ablation_id"]) for row in contract["ablations"]]
    if set(ablation_ids) != REQUIRED_ABLATIONS or len(ablation_ids) != len(REQUIRED_ABLATIONS):
        raise ValueError("Agent-Twin ablation set is incomplete or duplicated.")
    acceptance = contract["acceptance"]
    required_conditions = {str(value) for value in acceptance["required_conditions"]}
    expected_conditions = {
        "full_coevolution_beats_strongest_static_baseline",
        "full_coevolution_beats_both_one_sided_adaptive_baselines",
        "fixed_budget_and_fixed_risk_directions_agree",
        "new_region_gains_preserve_old_region_retention",
        "calibration_and_support_domain_within_frozen_tolerances",
        "all_updates_evidence_gated_and_reproducible",
        "rollback_demonstrated_on_injected_or_real_failed_update",
        "final_holdout_unused_for_updates_or_selection",
    }
    if required_conditions != expected_conditions:
        raise ValueError("Agent-Twin acceptance conditions do not match the prospective gate.")
    final_evidence = contract["final_evidence"]
    required_final_fields = {
        "schema_version",
        "final_gate",
        "status",
        "real_round_count",
        "prerequisite_snapshot_hash",
        "baseline_comparisons",
        "baseline_results",
        "continual_learning_controls",
        "fixed_budget_direction",
        "fixed_risk_direction",
        "old_domain_retention",
        "calibration_and_support_domain",
        "evidence_reproducibility",
        "rollback",
        "final_holdout_evaluation",
        "ablations",
        "statistics",
    }
    if set(final_evidence["required_fields"]) != required_final_fields:
        raise ValueError("Agent-Twin final evidence fields do not match the required gate surface.")


def evidence_binding(path_value: str, repo_root: Path) -> dict[str, object]:
    path = repo_path(path_value, repo_root)
    return {
        "path": relative_path(path, repo_root),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def evaluate_prerequisites(contract: Mapping[str, Any], repo_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prerequisite in contract["prerequisites"]:
        path_value = str(prerequisite["evidence_path"])
        binding = evidence_binding(path_value, repo_root)
        assertions: list[dict[str, object]] = []
        if binding["exists"]:
            try:
                payload = json.loads(repo_path(path_value, repo_root).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                payload = None
                parse_error = str(exc)
            else:
                parse_error = ""
            for assertion in prerequisite["assertions"]:
                found, observed = json_path_value(payload, str(assertion["json_path"])) if payload is not None else (False, None)
                assertions.append(
                    {
                        "json_path": str(assertion["json_path"]),
                        "expected": assertion["equals"],
                        "observed": observed,
                        "passed": bool(found and observed == assertion["equals"]),
                    }
                )
        else:
            parse_error = "evidence_artifact_missing"
            assertions = [
                {
                    "json_path": str(assertion["json_path"]),
                    "expected": assertion["equals"],
                    "observed": None,
                    "passed": False,
                }
                for assertion in prerequisite["assertions"]
            ]
        passed = bool(binding["exists"] and not parse_error and all(bool(row["passed"]) for row in assertions))
        rows.append(
            {
                "prerequisite_id": str(prerequisite["prerequisite_id"]),
                "passed": passed,
                "failure_blocker": str(prerequisite["failure_blocker"]),
                "evidence": binding,
                "parse_error": parse_error,
                "assertions": assertions,
            }
        )
    return rows


def round_artifact_paths(contract: Mapping[str, Any], artifact_root: Path, round_index: int) -> dict[str, Path]:
    required_artifacts = contract["round_protocol"]["required_artifacts"]
    return {
        str(key): artifact_root / str(template).format(round_index=round_index)
        for key, template in required_artifacts.items()
    }


def build_contract(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    config_file = repo_path(config_path, repo_root)
    schema_file = repo_path(schema_path, repo_root)
    contract = load_contract(config_file)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validate_contract(contract, schema)
    prerequisites = evaluate_prerequisites(contract, repo_root)
    prerequisite_snapshot_hash = canonical_sha256(prerequisites)
    unmet = [row for row in prerequisites if not bool(row["passed"])]
    blockers = list(dict.fromkeys(str(row["failure_blocker"]) for row in unmet))
    prerequisites_passed = not unmet
    final_holdout_policy = contract["formal_execution_policy"]["final_holdout_policy"]
    return {
        "schema_version": "neuronsifter-agent-twin-coevolution-update-contract-audit-v1",
        "track_id": str(contract["track_id"]),
        "final_gate": str(contract["final_gate"]),
        "source_contract": {
            **evidence_binding(relative_path(config_file, repo_root), repo_root),
            "contract_sha256": canonical_sha256(contract),
        },
        "source_schema": evidence_binding(relative_path(schema_file, repo_root), repo_root),
        "contract_status": "frozen",
        "implementation_allowed": bool(contract["formal_execution_policy"]["implementation_allowed_before_prerequisites"]),
        "formal_execution_allowed": bool(prerequisites_passed),
        "empirical_execution_status": "not_authorized" if not prerequisites_passed else "authorized_not_started",
        "final_gate_passed": False,
        "status": "blocked_preconditions" if not prerequisites_passed else "ready_for_formal_execution",
        "blockers": blockers,
        "prerequisites": prerequisites,
        # This binds later final evidence to the resolved evidence records, not
        # merely to the static YAML that declared the prerequisite policy.
        "prerequisite_snapshot_hash": prerequisite_snapshot_hash,
        "final_holdout_policy": final_holdout_policy,
        "round_protocol": {
            "minimum_real_round_count": int(contract["round_protocol"]["minimum_real_round_count"]),
            "artifact_root": str(contract["round_protocol"]["artifact_root"]),
            "required_artifacts": contract["round_protocol"]["required_artifacts"],
            "required_order": contract["round_protocol"]["required_order"],
        },
        "baseline_ids": [str(row["baseline_id"]) for row in contract["baselines"]],
        "ablation_ids": [str(row["ablation_id"]) for row in contract["ablations"]],
        "evaluation_systems": contract["evaluation"]["frozen_vs_evolving_systems"],
        "comparison_invariants": contract["evaluation"]["comparison_invariants"],
        "acceptance_conditions": contract["acceptance"]["required_conditions"],
        "final_evidence": {
            "required": evidence_binding(str(contract["final_evidence"]["artifact_path"]), repo_root),
            "required_fields": contract["final_evidence"]["required_fields"],
        },
        "next_action": (
            "resolve_prerequisite_evidence_and_rebuild_agent_twin_update_contract"
            if not prerequisites_passed
            else "run_round_1_with_registered_tools_and_frozen_final_holdout"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Agent-Twin co-evolution update-contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_contract(args.config, args.schema)
    output_path = repo_path(args.output)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
