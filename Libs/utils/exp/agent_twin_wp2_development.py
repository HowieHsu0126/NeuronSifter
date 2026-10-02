"""Development-only Agent-Twin WP2 sequential qualification harness.

The harness intentionally uses transparent, configured development generators.
It is not a substitute for public sequential evidence, a Core-13 static
baseline, a formal holdout, or a scientific co-evolution performance claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from Libs.utils.data.raw_data_registry import ensure_directory, resolve_repo_path, temporary_output_path, write_json, write_text_atomic
from Libs.utils.model.screening_core import stable_uid


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "Libs/configs/experiments/agent_twin_wp2_development.yaml"
DEFAULT_SCHEMA = "Libs/configs/schema/agent_twin_wp2_development.schema.json"
ALLOWED_OUTCOMES = {
    "supporting",
    "falsified",
    "inconclusive",
    "requires_observation",
    "twin_mismatch_suspected",
    "resolved",
}
BEHAVIOR_CHANGING_OUTCOMES = {
    "falsified",
    "inconclusive",
    "requires_observation",
    "twin_mismatch_suspected",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_identity() -> dict[str, Any]:
    """Bind a development rehearsal to the exact files that define it."""
    relative_paths = (
        DEFAULT_CONFIG,
        DEFAULT_SCHEMA,
        "Libs/utils/exp/agent_twin_wp2_development.py",
        "Libs/scripts/experiments/run_agent_twin_wp2_development.py",
        "Libs/scripts/validators/validate_agent_twin_wp2.py",
    )
    files: list[dict[str, str]] = []
    for relative_path in relative_paths:
        path = repo_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"WP2 execution identity file is missing: {path}")
        files.append({"path": relative_path, "sha256": sha256_file(path)})
    return {
        "identity_kind": "worktree_file_hashes_not_exact_clean_candidate",
        "files": files,
        "canonical_sha256": canonical_hash({"files": files}),
    }


def repo_path(path_value: str) -> Path:
    return resolve_repo_path(path_value)


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_config(config: Mapping[str, Any], schema_path: Path) -> dict[str, Any]:
    if "agent_twin_wp2_development" not in config:
        raise ValueError("WP2 configuration is missing agent_twin_wp2_development.")
    payload = dict(config["agent_twin_wp2_development"])
    schema = load_json_object(schema_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        raise ValueError("WP2 configuration schema error: " + "; ".join(error.message for error in errors))
    world_ids = [str(row["world_id"]) for row in payload["development_worlds"]]
    if len(world_ids) != len(set(world_ids)):
        raise ValueError("WP2 development world ids must be unique.")
    if len(world_ids) != int(payload["protocol"]["paired_development_world_count"]):
        raise ValueError("WP2 development world count does not match the paired protocol.")
    rounds = int(payload["protocol"]["rounds_per_run"])
    for world in payload["development_worlds"]:
        schedule = [str(value) for value in world["outcome_schedule"]]
        if len(schedule) < rounds or any(value not in ALLOWED_OUTCOMES for value in schedule[:rounds]):
            raise ValueError(f"WP2 world {world['world_id']} has an invalid outcome schedule.")
    fixture = payload["protocol"]["negative_rollback_fixture"]
    if str(fixture["world_id"]) not in set(world_ids) or not 1 <= int(fixture["round_index"]) <= rounds:
        raise ValueError("WP2 negative rollback fixture is outside the configured development worlds or rounds.")
    return payload


def validate_wp1_evidence(payload: Mapping[str, Any]) -> dict[str, str]:
    evidence = payload["wp1_evidence"]
    full_suite = load_json_object(repo_path(str(evidence["full_suite_audit"])))
    focused = load_json_object(repo_path(str(evidence["focused_audit"])))
    inventory = load_json_object(repo_path(str(evidence["inventory_audit"])))
    expected_commit = str(evidence["candidate_commit"])
    expected_tree = str(evidence["candidate_tree"])
    if any(str(row.get("candidate_commit")) != expected_commit for row in (full_suite, focused, inventory)):
        raise ValueError("WP2 must bind every WP1 audit to the frozen candidate commit.")
    if any(str(row.get("candidate_tree_hash")) != expected_tree for row in (full_suite, focused, inventory)):
        raise ValueError("WP2 must bind every WP1 audit to the frozen candidate tree.")
    if focused.get("status") != "passed" or int(focused["result"]["passed"]) != 7:
        raise ValueError("WP2 requires the frozen WP1 focused validation result.")
    full = full_suite.get("full_suite", {})
    if full_suite.get("status") != "passed" or int(full.get("passed", -1)) != 614 or any(int(full.get(key, -1)) != 0 for key in ("failures", "errors", "skips")):
        raise ValueError("WP2 requires the frozen WP1 614/614 exact-clean suite.")
    if inventory.get("status") != "passed" or int(inventory.get("candidate_test_count", -1)) != 614 or int(inventory.get("inherited_test_count", -1)) != 607:
        raise ValueError("WP2 requires the frozen WP1 test inventory reconciliation.")
    return {
        "candidate_commit": expected_commit,
        "candidate_tree": expected_tree,
        "full_suite_audit_sha256": sha256_file(repo_path(str(evidence["full_suite_audit"]))),
        "focused_audit_sha256": sha256_file(repo_path(str(evidence["focused_audit"]))),
        "inventory_audit_sha256": sha256_file(repo_path(str(evidence["inventory_audit"]))),
    }


def inspect_core13_frontier(payload: Mapping[str, Any]) -> dict[str, Any]:
    join = payload["core13_join"]
    path = repo_path(str(join["frontier_path"]))
    if not path.is_file():
        return {
            "frontier_path": str(join["frontier_path"]),
            "frontier_exists": False,
            "core13_frontier_bound": False,
            "status": "missing",
            "reason": "core13_frontier_missing",
        }
    frontier = load_json_object(path)
    status = str(frontier.get("status", ""))
    selected_key = str(join["required_selected_backbone_field"])
    selected_backbone = frontier.get(selected_key)
    required_identity_fields = [str(value) for value in join["required_identity_fields"]]
    missing_identity_fields = [field for field in required_identity_fields if not isinstance(frontier.get(field), str) or not str(frontier[field]).strip()]
    hash_identity_fields = (
        "selected_twin_checkpoint_sha256",
        "selected_backbone_config_sha256",
        "provider_tool_profile_sha256",
    )
    invalid_identity_hash_fields = [
        field
        for field in hash_identity_fields
        if field not in missing_identity_fields
        and (not isinstance(frontier.get(field), str) or len(str(frontier[field])) != 64 or any(character not in "0123456789abcdef" for character in str(frontier[field]).lower()))
    ]
    identity_hash_errors: list[str] = []
    for path_field, hash_field in (
        ("selected_twin_checkpoint_path", "selected_twin_checkpoint_sha256"),
        ("selected_backbone_config_path", "selected_backbone_config_sha256"),
    ):
        if path_field in missing_identity_fields or hash_field in missing_identity_fields:
            continue
        try:
            artifact_path = repo_path(str(frontier[path_field]))
            if not artifact_path.is_file() or sha256_file(artifact_path) != str(frontier[hash_field]):
                identity_hash_errors.append(path_field)
        except OSError:
            identity_hash_errors.append(path_field)
    qualified = (
        status in set(str(value) for value in join["acceptable_frontier_statuses"])
        and isinstance(selected_backbone, str)
        and bool(selected_backbone.strip())
        and not missing_identity_fields
        and not invalid_identity_hash_fields
        and not identity_hash_errors
    )
    return {
        "frontier_path": str(join["frontier_path"]),
        "frontier_exists": True,
        "frontier_sha256": sha256_file(path),
        "status": status,
        "selected_backbone_field": selected_key,
        "selected_backbone_id": selected_backbone if isinstance(selected_backbone, str) else None,
        "selected_twin_checkpoint_id": frontier.get("selected_twin_checkpoint_id"),
        "selected_twin_checkpoint_path": frontier.get("selected_twin_checkpoint_path"),
        "selected_twin_checkpoint_sha256": frontier.get("selected_twin_checkpoint_sha256"),
        "selected_backbone_config_path": frontier.get("selected_backbone_config_path"),
        "selected_backbone_config_sha256": frontier.get("selected_backbone_config_sha256"),
        "provider_tool_profile_sha256": frontier.get("provider_tool_profile_sha256"),
        "required_identity_fields": required_identity_fields,
        "missing_identity_fields": missing_identity_fields,
        "invalid_identity_hash_fields": invalid_identity_hash_fields,
        "identity_hash_errors": identity_hash_errors,
        "core13_frontier_bound": qualified,
        "reason": "qualified" if qualified else "frontier_identity_not_fully_bound",
    }


def resolved_initial_twin_checkpoint_id(payload: Mapping[str, Any], frontier: Mapping[str, Any]) -> str:
    """Use the immutable Core-13 Twin identity only after its full binding validates."""
    if bool(frontier.get("core13_frontier_bound")):
        checkpoint_id = frontier.get("selected_twin_checkpoint_id")
        if isinstance(checkpoint_id, str) and checkpoint_id:
            return checkpoint_id
        raise ValueError("A qualified Core-13 frontier is missing its selected Twin checkpoint identity.")
    return str(payload["protocol"]["initial_twin_checkpoint_id"])


def validate_formal_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    boundary = payload["formal_boundary"]
    contract = load_json_object(repo_path(str(boundary["formal_update_contract"])))
    if contract.get("formal_execution_allowed") is not False or contract.get("final_gate_passed") is not False:
        raise ValueError("WP2 development-only execution requires formal execution and the final gate to remain closed.")
    if int(boundary["formal_holdout_access_count"]) != 0:
        raise ValueError("WP2 development-only execution forbids formal holdout access.")
    return {
        "formal_update_contract_path": str(boundary["formal_update_contract"]),
        "formal_update_contract_sha256": sha256_file(repo_path(str(boundary["formal_update_contract"]))),
        "formal_contract_status": str(contract.get("status")),
        "formal_execution_allowed": False,
        "formal_claim_allowed": False,
        "formal_holdout_access_count": 0,
    }


def append_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    ensure_directory(path.parent)
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def action_for_previous_outcome(previous_outcome: str | None) -> str:
    if previous_outcome == "falsified":
        return "falsifier_follow_up_action"
    if previous_outcome == "inconclusive":
        return "observation_disambiguation_action"
    if previous_outcome == "requires_observation":
        return "observation_enrichment_action"
    if previous_outcome == "twin_mismatch_suspected":
        return "twin_diagnostic_action"
    if previous_outcome == "supporting":
        return "replication_resolution_action"
    return "initial_competing_hypothesis_action"


def checkpoint_descriptor(
    path: Path,
    *,
    checkpoint_id: str,
    branch: str,
    world_id: str,
    round_index: int,
    parent_checkpoint_id: str,
    parameter_update: bool,
    outcome_hash: str,
) -> str:
    payload = {
        "schema_version": "neuronsifter-agent-twin-wp2-development-checkpoint-v1",
        "checkpoint_id": checkpoint_id,
        "branch": branch,
        "world_id": world_id,
        "round_index": round_index,
        "parent_checkpoint_id": parent_checkpoint_id,
        "parameter_update": parameter_update,
        "outcome_hash": outcome_hash,
        "development_only": True,
        "scientific_training_label": False,
    }
    write_json(path, payload)
    return sha256_file(path)


def promotion_metrics(branch: str, is_negative_fixture: bool) -> dict[str, float]:
    if branch == "static":
        return {
            "new_region_decision_risk_delta": 0.0,
            "calibration_drift": 0.0,
            "support_domain_drift": 0.0,
            "old_domain_retention": 1.0,
            "ood_selective_risk_delta": 0.0,
            "constraint_risk_delta": 0.0,
        }
    if is_negative_fixture:
        return {
            "new_region_decision_risk_delta": 0.01,
            "calibration_drift": 0.04,
            "support_domain_drift": 0.01,
            "old_domain_retention": 0.985,
            "ood_selective_risk_delta": 0.01,
            "constraint_risk_delta": 0.01,
        }
    return {
        "new_region_decision_risk_delta": 0.01,
        "calibration_drift": 0.01,
        "support_domain_drift": 0.01,
        "old_domain_retention": 0.985,
        "ood_selective_risk_delta": 0.01,
        "constraint_risk_delta": 0.01,
    }


def promotion_eligible(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["new_region_decision_risk_delta"]) >= float(thresholds["minimum_new_region_decision_risk_delta"])
        and float(metrics["calibration_drift"]) <= float(thresholds["maximum_calibration_drift"])
        and float(metrics["support_domain_drift"]) <= float(thresholds["maximum_support_domain_drift"])
        and float(metrics["old_domain_retention"]) >= float(thresholds["minimum_old_domain_retention"])
        and float(metrics["ood_selective_risk_delta"]) <= float(thresholds["maximum_ood_selective_risk_delta"])
        and float(metrics["constraint_risk_delta"]) <= float(thresholds["maximum_constraint_risk_delta"])
    )


def run_branch(
    payload: Mapping[str, Any],
    output_root: Path,
    world: Mapping[str, Any],
    branch: str,
    initial_twin_checkpoint_id: str,
) -> dict[str, Any]:
    protocol = payload["protocol"]
    world_id = str(world["world_id"])
    run_uid = stable_uid("agent_twin_wp2", branch, world_id, str(world["seed"]))
    run_root = output_root / "sequential_evidence" / "sequential_runs" / run_uid
    run_root.mkdir(parents=True, exist_ok=False)
    branch_root = output_root / ("static_baseline/static_runs" if branch == "static" else "twin_updates/coevolution_runs") / run_uid
    branch_root.mkdir(parents=True, exist_ok=False)
    rounds = int(protocol["rounds_per_run"])
    previous_outcome: str | None = None
    previous_action: str | None = None
    twin_id = initial_twin_checkpoint_id
    agent_id = str(protocol["initial_agent_checkpoint_id"])
    memory_id = str(protocol["initial_memory_id"])
    memory_hash = canonical_hash({"memory_id": memory_id, "world_id": world_id, "seed": int(world["seed"])})
    run_events: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    evidence_updates: list[dict[str, Any]] = []
    mioy_versions: list[dict[str, Any]] = []
    emp_versions: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    retentions: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    rollbacks: list[dict[str, Any]] = []
    update_candidates: list[dict[str, Any]] = []
    behavior_changed = False

    for round_index in range(1, rounds + 1):
        action_name = action_for_previous_outcome(previous_outcome)
        if previous_outcome in BEHAVIOR_CHANGING_OUTCOMES and previous_action is not None:
            behavior_changed = behavior_changed or action_name != previous_action
        hypothesis_uid = stable_uid("wp2_hypothesis", world_id, str(round_index), memory_hash)
        action_uid = stable_uid("wp2_action", run_uid, str(round_index), action_name)
        selection_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-selection-v1",
            "event_type": "selection_freeze",
            "event_sequence": len(run_events) + 1,
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "hypothesis_uid": hypothesis_uid,
            "structured_action_uid": action_uid,
            "selected_action": action_name,
            "gate_h": "pass",
            "gate_w": "pass",
            "gate_i": "pass",
            "eligibility_passed": True,
            "validity_passed": True,
            "selection_frozen_before_outcome": True,
            "formal_holdout_visible": False,
            "twin_id": twin_id,
            "agent_id": agent_id,
            "memory_hash": memory_hash,
            "budget_units": 1,
        }
        selection_payload["selection_hash"] = canonical_hash(selection_payload)
        selections.append(selection_payload)
        run_events.append(selection_payload)

        outcome_label = str(world["outcome_schedule"][round_index - 1])
        source_outcome_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-development-source-outcome-v1",
            "world_id": world_id,
            "round_index": round_index,
            "selected_action": action_name,
            "outcome_label": outcome_label,
            "outcome_origin": str(protocol["outcome_origin"]),
            "outcome_generator_registration": str(protocol["outcome_generator_registration"]),
            "generator_family": str(world["generator_family"]),
            "generator_seed": int(world["seed"]),
            "fixture_rehearsal_only": True,
        }
        source_outcome_hash = canonical_hash(source_outcome_payload)
        outcome_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-development-outcome-v1",
            "event_type": "development_outcome",
            "event_sequence": len(run_events) + 1,
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "action_uid": action_uid,
            "selection_hash": selection_payload["selection_hash"],
            "outcome_label": outcome_label,
            "outcome_origin": str(protocol["outcome_origin"]),
            "source_outcome_uid": stable_uid("wp2_source_outcome", world_id, str(round_index), source_outcome_hash),
            "source_outcome_hash": source_outcome_hash,
            "outcome_generator_registration": str(protocol["outcome_generator_registration"]),
            "verified_update_label": True,
            "twin_rollout_used_as_label": False,
            "formal_holdout_visible": False,
            "evaluator_only_field_visible": False,
            "generator_family": str(world["generator_family"]),
            "generator_seed": int(world["seed"]),
            "fixture_rehearsal_only": True,
        }
        outcome_payload["outcome_hash"] = canonical_hash(outcome_payload)
        outcomes.append(outcome_payload)
        run_events.append(outcome_payload)

        evidence_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-evidence-update-v1",
            "event_type": "evidence_update",
            "event_sequence": len(run_events) + 1,
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "evidence_uid": stable_uid("wp2_evidence", run_uid, str(round_index), outcome_payload["outcome_hash"]),
            "outcome_hash": outcome_payload["outcome_hash"],
            "source_outcome_uid": outcome_payload["source_outcome_uid"],
            "source_outcome_hash": outcome_payload["source_outcome_hash"],
            "outcome_origin": str(protocol["outcome_origin"]),
            "observation_assimilation_gate": "pass",
            "admission_decision": "admitted",
            "evidence_quality_weight": 1.0,
            "twin_rollout_used_as_label": False,
            "formal_holdout_visible": False,
        }
        evidence_payload["evidence_hash"] = canonical_hash(evidence_payload)
        evidence_updates.append(evidence_payload)
        run_events.append(evidence_payload)

        next_memory_hash = canonical_hash({"previous_memory_hash": memory_hash, "evidence_hash": evidence_payload["evidence_hash"]})
        mioy_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-mioy-v1",
            "event_type": "mioy_version",
            "event_sequence": len(run_events) + 1,
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "previous_memory_hash": memory_hash,
            "memory_hash": next_memory_hash,
            "admitted_evidence_hash": evidence_payload["evidence_hash"],
            "source_outcome_hash": outcome_payload["source_outcome_hash"],
            "append_only": True,
        }
        mioy_versions.append(mioy_payload)
        run_events.append(mioy_payload)
        emp_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-emp-v1",
            "event_type": "emp_version",
            "event_sequence": len(run_events) + 1,
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "previous_memory_hash": memory_hash,
            "memory_hash": next_memory_hash,
            "posterior_outcome_label": outcome_label,
            "append_only": True,
        }
        emp_versions.append(emp_payload)
        run_events.append(emp_payload)
        memory_hash = next_memory_hash

        fixture = protocol["negative_rollback_fixture"]
        is_negative_fixture = branch == "coevolution" and world_id == str(fixture["world_id"]) and round_index == int(fixture["round_index"])
        metrics = promotion_metrics(branch, is_negative_fixture)
        candidate_twin_id = twin_id if branch == "static" else stable_uid("wp2_twin_candidate", run_uid, str(round_index), evidence_payload["evidence_hash"])
        twin_descriptor = branch_root / f"twin_candidate_round_{round_index}.json"
        twin_descriptor_hash = checkpoint_descriptor(
            twin_descriptor,
            checkpoint_id=candidate_twin_id,
            branch=branch,
            world_id=world_id,
            round_index=round_index,
            parent_checkpoint_id=twin_id,
            parameter_update=branch == "coevolution",
            outcome_hash=outcome_payload["outcome_hash"],
        )
        eligible = promotion_eligible(metrics, payload["promotion_thresholds"])
        if branch == "static":
            twin_decision = "static_no_update"
        else:
            twin_decision = "promote" if eligible else "rollback"
        candidate_agent_id = stable_uid("wp2_agent_candidate", run_uid, str(round_index), memory_hash)
        agent_descriptor = branch_root / f"agent_candidate_round_{round_index}.json"
        agent_descriptor_hash = checkpoint_descriptor(
            agent_descriptor,
            checkpoint_id=candidate_agent_id,
            branch=branch,
            world_id=world_id,
            round_index=round_index,
            parent_checkpoint_id=agent_id,
            parameter_update=True,
            outcome_hash=evidence_payload["evidence_hash"],
        )
        previous_twin_id = twin_id
        previous_agent_id = agent_id
        if twin_decision == "promote":
            twin_id = candidate_twin_id
        if branch != "static":
            agent_id = candidate_agent_id
        else:
            agent_id = candidate_agent_id

        update_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-twin-update-v1",
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "input_twin_checkpoint_id": previous_twin_id,
            "candidate_twin_checkpoint_id": candidate_twin_id,
            "candidate_twin_checkpoint_path": twin_descriptor.relative_to(REPO_ROOT).as_posix() if twin_descriptor.is_relative_to(REPO_ROOT) else str(twin_descriptor),
            "candidate_twin_checkpoint_sha256": twin_descriptor_hash,
            "twin_parameter_update": branch == "coevolution",
            "update_label_origin": str(protocol["outcome_origin"]),
            "twin_rollout_used_as_label": False,
            "admitted_evidence_hash": evidence_payload["evidence_hash"],
            "source_outcome_hash": outcome_payload["source_outcome_hash"],
            "promotion_eligible": eligible if branch == "coevolution" else None,
            "twin_decision": twin_decision,
            **metrics,
        }
        update_candidates.append(update_payload)
        promotion_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-promotion-v1",
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "twin_decision": twin_decision,
            "agent_decision": "promote",
            "input_twin_checkpoint_id": previous_twin_id,
            "candidate_twin_checkpoint_id": candidate_twin_id,
            "promoted_twin_checkpoint_id": twin_id,
            "input_agent_checkpoint_id": previous_agent_id,
            "candidate_agent_checkpoint_id": candidate_agent_id,
            "promoted_agent_checkpoint_id": agent_id,
            "rollback_executed": twin_decision == "rollback",
            "rollback_reason": "injected_calibration_drift" if twin_decision == "rollback" else "",
            "restored_twin_checkpoint_id": previous_twin_id if twin_decision == "rollback" else "",
            "restored_agent_checkpoint_id": previous_agent_id if twin_decision == "rollback" else "",
            "formal_holdout_used": False,
        }
        promotions.append(promotion_payload)
        if twin_decision == "rollback":
            rollbacks.append(copy.deepcopy(promotion_payload))
        calibrations.append({"run_uid": run_uid, "branch": branch, "world_id": world_id, "round_index": round_index, "pre_post_complete": True, "calibration_error_pre": 0.10, "calibration_error_post": 0.10 + metrics["calibration_drift"], "calibration_drift": metrics["calibration_drift"], "promotion_decision": twin_decision})
        retentions.append({"run_uid": run_uid, "branch": branch, "world_id": world_id, "round_index": round_index, "old_domain_retention": metrics["old_domain_retention"], "new_region_decision_risk_delta": metrics["new_region_decision_risk_delta"], "constraint_risk_delta": metrics["constraint_risk_delta"], "promotion_decision": twin_decision})
        validity_rows.append({"run_uid": run_uid, "branch": branch, "world_id": world_id, "round_index": round_index, "validity_domain_precision": 1.0, "validity_domain_recall": 1.0, "illegally_expanded": False, "constraint_risk_delta": metrics["constraint_risk_delta"]})
        ood_rows.append({"run_uid": run_uid, "branch": branch, "world_id": world_id, "round_index": round_index, "ood_selective_risk_delta": metrics["ood_selective_risk_delta"], "support_domain_drift": metrics["support_domain_drift"], "abstention_valid": True})
        round_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-round-v1",
            "run_uid": run_uid,
            "branch": branch,
            "world_id": world_id,
            "round_index": round_index,
            "selection": selection_payload,
            "outcome": outcome_payload,
            "evidence_update": evidence_payload,
            "mioy": mioy_payload,
            "emp": emp_payload,
            "twin_update": update_payload,
            "promotion": promotion_payload,
            "formal_holdout_accessed": False,
        }
        write_json(run_root / f"round_{round_index}.json", round_payload)
        previous_action = action_name
        previous_outcome = outcome_label

    run_manifest = {
        "schema_version": "neuronsifter-agent-twin-wp2-run-v1",
        "run_uid": run_uid,
        "branch": branch,
        "world_id": world_id,
        "seed": int(world["seed"]),
        "generator_family": str(world["generator_family"]),
        "round_count": rounds,
        "initial_twin_checkpoint_id": initial_twin_checkpoint_id,
        "initial_agent_checkpoint_id": str(protocol["initial_agent_checkpoint_id"]),
        "initial_memory_id": str(protocol["initial_memory_id"]),
        "outcome_schedule": list(world["outcome_schedule"])[:rounds],
        "outcome_origin": str(protocol["outcome_origin"]),
        "outcome_generator_registration": str(protocol["outcome_generator_registration"]),
        "fixture_rehearsal_only": True,
        "twin_rollout_used_as_label": False,
        "formal_holdout_access_count": 0,
        "behavior_changed_after_evidence": behavior_changed,
        "final_twin_checkpoint_id": twin_id,
        "final_agent_checkpoint_id": agent_id,
        "final_memory_hash": memory_hash,
        "rollback_count": len(rollbacks),
    }
    write_json(run_root / "run_manifest.json", run_manifest)
    return {
        "run_manifest": run_manifest,
        "events": run_events,
        "selections": selections,
        "outcomes": outcomes,
        "evidence_updates": evidence_updates,
        "mioy_versions": mioy_versions,
        "emp_versions": emp_versions,
        "calibrations": calibrations,
        "retentions": retentions,
        "validity_rows": validity_rows,
        "ood_rows": ood_rows,
        "promotions": promotions,
        "rollbacks": rollbacks,
        "update_candidates": update_candidates,
    }


def write_parquet(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write machine-readable metric rows through the repository parquet helper."""
    import pandas as pd

    ensure_directory(path.parent)
    temporary_path = temporary_output_path(path)
    pd.DataFrame(rows).to_parquet(temporary_path, index=False, engine="pyarrow")
    temporary_path.replace(path)


def paired_statistics(paired_rows: list[Mapping[str, Any]], statistics_config: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray([float(row["hypothesis_resolution_delta"]) for row in paired_rows], dtype=np.float64)
    rng = np.random.default_rng(int(statistics_config["bootstrap_seed"]))
    sample_means = np.asarray([float(rng.choice(values, size=values.size, replace=True).mean()) for _ in range(int(statistics_config["bootstrap_replicates"]))])
    sign_means = np.asarray([float((values * signs).mean()) for signs in np.array(np.meshgrid(*[[-1.0, 1.0]] * values.size)).T.reshape(-1, values.size)])
    observed = abs(float(values.mean()))
    permutation_p = float(np.mean(np.abs(sign_means) >= observed))
    return {
        "schema_version": "neuronsifter-agent-twin-wp2-paired-statistics-v1",
        "independent_unit": str(statistics_config["independent_unit"]),
        "paired_world_count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)),
        "paired_effect_size": 0.0 if math.isclose(float(values.std(ddof=1)), 0.0) else float(values.mean() / values.std(ddof=1)),
        "bootstrap_95_ci": [float(np.quantile(sample_means, 0.025)), float(np.quantile(sample_means, 0.975))],
        "paired_permutation_two_sided_p": permutation_p,
        "bootstrap_replicates": int(statistics_config["bootstrap_replicates"]),
        "bootstrap_seed": int(statistics_config["bootstrap_seed"]),
        "inference_status": "not_interpretable_core13_frontier_unbound",
        "performance_claim": "not_evaluated",
    }


def write_checksums(root: Path) -> str:
    rows: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    content = "\n".join(rows) + "\n"
    write_text_atomic(root / "checksums.sha256", content)
    return sha256_file(root / "checksums.sha256")


def run_wp2_development(payload: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"WP2 output root already exists; preserve the append-only record: {output_root}")
    wp1 = validate_wp1_evidence(payload)
    formal = validate_formal_boundary(payload)
    frontier = inspect_core13_frontier(payload)
    source_identity = execution_identity()
    initial_twin_checkpoint_id = resolved_initial_twin_checkpoint_id(payload, frontier)
    output_root.mkdir(parents=True, exist_ok=False)
    for relative in ("sequential_evidence", "static_baseline", "twin_updates", "calibration_drift", "retention", "rollback", "paired_comparison", "efficiency", "failures"):
        ensure_directory(output_root / relative)
    all_results = [
        run_branch(payload, output_root, world, branch, initial_twin_checkpoint_id)
        for world in payload["development_worlds"]
        for branch in payload["protocol"]["branches"]
    ]

    event_rows = [row for result in all_results for row in result["events"]]
    selection_rows = [row for result in all_results for row in result["selections"]]
    outcome_rows = [row for result in all_results for row in result["outcomes"]]
    evidence_rows = [row for result in all_results for row in result["evidence_updates"]]
    mioy_rows = [row for result in all_results for row in result["mioy_versions"]]
    emp_rows = [row for result in all_results for row in result["emp_versions"]]
    calibration_rows = [row for result in all_results for row in result["calibrations"]]
    retention_rows = [row for result in all_results for row in result["retentions"]]
    validity_rows = [row for result in all_results for row in result["validity_rows"]]
    ood_rows = [row for result in all_results for row in result["ood_rows"]]
    promotion_rows = [row for result in all_results for row in result["promotions"]]
    rollback_rows = [row for result in all_results for row in result["rollbacks"]]
    update_rows = [row for result in all_results for row in result["update_candidates"]]
    manifests = [result["run_manifest"] for result in all_results]

    sequential_root = output_root / "sequential_evidence"
    append_jsonl(sequential_root / "sequential_event_logs.jsonl", event_rows)
    append_jsonl(sequential_root / "selection_freeze_records.jsonl", selection_rows)
    append_jsonl(sequential_root / "development_outcomes.jsonl", outcome_rows)
    append_jsonl(sequential_root / "evidence_updates.jsonl", evidence_rows)
    append_jsonl(sequential_root / "mioy_versions.jsonl", mioy_rows)
    append_jsonl(sequential_root / "emp_versions.jsonl", emp_rows)
    adaptation_report = {
        "schema_version": "neuronsifter-agent-twin-wp2-sequential-adaptation-report-v1",
        "mode": "development_only",
        "paired_runs_complete": len(manifests) // 2,
        "multi_round_execution": all(int(row["round_count"]) >= 3 for row in manifests),
        "selection_before_outcome": True,
        "verified_update_labels_only": all(row["outcome_origin"] == payload["protocol"]["outcome_origin"] for row in outcome_rows),
        "twin_rollout_used_as_label": False,
        "falsification_or_inconclusive_case": any(row["outcome_label"] in BEHAVIOR_CHANGING_OUTCOMES for row in outcome_rows),
        "agent_behavior_changes_after_evidence": any(bool(row["behavior_changed_after_evidence"]) for row in manifests),
        "fixed_seed_replay": True,
        "formal_holdout_access_count": 0,
        "outcome_origin": payload["protocol"]["outcome_origin"],
        "outcome_generator_registration": payload["protocol"]["outcome_generator_registration"],
        "fixture_rehearsal_only": True,
        "empirical_precondition_resolution_allowed": False,
    }
    write_json(sequential_root / "sequential_adaptation_report.json", adaptation_report)

    static_binding = {
        "schema_version": "neuronsifter-agent-twin-wp2-static-baseline-binding-v1",
        "mode": "development_only",
        "core13_join": frontier,
        "core13_frontier_bound": bool(frontier["core13_frontier_bound"]),
        "static_baseline_qualified": bool(frontier["core13_frontier_bound"]),
        "temporary_backbone_used_for_interface_smoke": not bool(frontier["core13_frontier_bound"]),
        "operational_twin_checkpoint_id": initial_twin_checkpoint_id,
        "operational_twin_checkpoint_path": frontier.get("selected_twin_checkpoint_path") if frontier["core13_frontier_bound"] else None,
        "operational_twin_checkpoint_sha256": frontier.get("selected_twin_checkpoint_sha256") if frontier["core13_frontier_bound"] else None,
        "operational_backbone_kind": payload["protocol"]["operational_backbone_kind"],
        "blocked_static_baseline": not bool(frontier["core13_frontier_bound"]),
        "performance_conclusion_allowed": False,
    }
    write_json(output_root / "static_baseline" / "static_baseline_binding.json", static_binding)
    paired_initialization = {
        "schema_version": "neuronsifter-agent-twin-wp2-paired-initialization-v1",
        "same_initialization": True,
        "same_outcome_order": True,
        "same_budget": True,
        "same_provider_tool_configuration": True,
        "only_registered_difference_is_twin_update": True,
        "initial_twin_checkpoint_id": initial_twin_checkpoint_id,
        "initial_agent_checkpoint_id": payload["protocol"]["initial_agent_checkpoint_id"],
        "initial_memory_id": payload["protocol"]["initial_memory_id"],
        "world_ids": [str(row["world_id"]) for row in payload["development_worlds"]],
        "frontier_bound": bool(frontier["core13_frontier_bound"]),
    }
    write_json(output_root / "static_baseline" / "paired_initialization_manifest.json", paired_initialization)
    paired_budget = {
        "schema_version": "neuronsifter-agent-twin-wp2-paired-budget-v1",
        "action_budget_per_round": payload["protocol"]["action_budget_per_round"],
        "rounds_per_run": payload["protocol"]["rounds_per_run"],
        "compute_ceiling_gpu_hours_per_run": payload["protocol"]["compute_ceiling_gpu_hours_per_run"],
        "actual_gpu_hours": 0.0,
        "matched": True,
        "development_only": True,
    }
    write_json(output_root / "static_baseline" / "paired_budget_report.json", paired_budget)

    append_jsonl(output_root / "twin_updates" / "promotion_decisions.jsonl", promotion_rows)
    append_jsonl(output_root / "rollback" / "rollback_reports.jsonl", rollback_rows)
    for row in update_rows:
        write_json(output_root / "twin_updates" / "twin_update_candidates" / f"{row['run_uid']}_round_{row['round_index']}.json", row)
    write_parquet(output_root / "calibration_drift" / "calibration_drift_metrics.parquet", calibration_rows)
    write_parquet(output_root / "retention" / "retention_metrics.parquet", retention_rows)
    write_parquet(output_root / "calibration_drift" / "validity_metrics.parquet", validity_rows)
    write_parquet(output_root / "calibration_drift" / "ood_metrics.parquet", ood_rows)

    static_by_world = {row["world_id"]: row for row in manifests if row["branch"] == "static"}
    coevolution_by_world = {row["world_id"]: row for row in manifests if row["branch"] == "coevolution"}
    paired_rows: list[dict[str, Any]] = []
    for world_id in sorted(static_by_world):
        static_run = static_by_world[world_id]
        coevolution_run = coevolution_by_world[world_id]
        resolution = float(sum(value == "resolved" for value in static_run["outcome_schedule"]) / int(payload["protocol"]["rounds_per_run"]))
        paired_rows.append(
            {
                "world_id": world_id,
                "generator_family": next(str(row["generator_family"]) for row in payload["development_worlds"] if str(row["world_id"]) == world_id),
                "static_run_uid": static_run["run_uid"],
                "coevolution_run_uid": coevolution_run["run_uid"],
                "static_hypothesis_resolution_per_experiment": resolution,
                "coevolution_hypothesis_resolution_per_experiment": resolution,
                "hypothesis_resolution_delta": 0.0,
                "same_outcome_schedule": static_run["outcome_schedule"] == coevolution_run["outcome_schedule"],
                "same_budget": True,
                "formal_holdout_access_count": 0,
                "development_only": True,
            }
        )
    write_parquet(output_root / "paired_comparison" / "paired_results.parquet", paired_rows)
    write_parquet(output_root / "paired_comparison" / "sequential_metrics.parquet", paired_rows)
    statistics = paired_statistics(paired_rows, payload["statistics"])
    write_json(output_root / "paired_comparison" / "paired_statistics.json", statistics)
    efficiency_rows = [
        {
            "run_uid": row["run_uid"],
            "branch": row["branch"],
            "world_id": row["world_id"],
            "verified_outcomes_consumed": int(row["round_count"]),
            "twin_update_count": int(row["round_count"]) if row["branch"] == "coevolution" else 0,
            "gpu_hours": 0.0,
            "agent_provider_cost": 0.0,
            "rollout_count": 0,
            "wall_clock_seconds": 0.0,
            "failed_updates": int(row["rollback_count"]),
            "rollback_count": int(row["rollback_count"]),
            "development_only": True,
        }
        for row in manifests
    ]
    write_parquet(output_root / "efficiency" / "efficiency_results.parquet", efficiency_rows)
    failure_analysis = {
        "schema_version": "neuronsifter-agent-twin-wp2-failure-analysis-v1",
        "failed_or_rolled_back_update_count": len(rollback_rows),
        "rollback_count": len(rollback_rows),
        "failure_categories": sorted({str(row["rollback_reason"]) for row in rollback_rows}),
        "failed_runs_deleted": False,
        "performance_conclusion": "not_evaluated_core13_frontier_unbound",
    }
    write_json(output_root / "failures" / "failure_analysis.json", failure_analysis)

    sequential_rehearsal_complete = bool(adaptation_report["paired_runs_complete"] >= 5 and adaptation_report["multi_round_execution"] and adaptation_report["falsification_or_inconclusive_case"] and adaptation_report["agent_behavior_changes_after_evidence"])
    drift_rehearsal_complete = bool(len(calibration_rows) == 30 and len(retention_rows) == 30 and len(validity_rows) == 30 and len(ood_rows) == 30 and len(rollback_rows) >= 1)
    decision = {
        "schema_version": "neuronsifter-agent-twin-wp2-precondition-decision-v1",
        "track_id": payload["track_id"],
        "mode": payload["mode"],
        "wp1_exact_clean_preserved": True,
        "wp1_binding": wp1,
        "execution_identity": source_identity,
        "development_sequential_evidence": {"passed": False, "technical_rehearsal_complete": sequential_rehearsal_complete, "reason": "fixture_outcomes_are_not_registered_external_development_evidence", "run_count": len(manifests), "paired_world_count": len(manifests) // 2, "rounds_per_run": payload["protocol"]["rounds_per_run"]},
        "core13_frontier": frontier,
        "static_baseline": {"core13_frontier_bound": bool(frontier["core13_frontier_bound"]), "matched_initialization_and_budget": True, "blocked_static_baseline": not bool(frontier["core13_frontier_bound"])},
        "calibration_drift": {"pre_post_calibration": False, "technical_rehearsal_complete": drift_rehearsal_complete, "reason": "fixture_descriptors_and_constant_metrics_are_not_measured_twin_evaluations", "old_domain_retention": False, "ood_validity": False, "rollback_verified": len(rollback_rows) >= 1, "invalid_candidate_not_promoted": len(rollback_rows) >= 1},
        "comparison": {"paired_statistics_complete": True, "failed_runs_reported": True, "efficiency_reported": True, "performance_conclusion": "not_evaluated_core13_frontier_unbound"},
        "integrity": {**formal, "coevolution_claim_upgraded": False, "formal_holdout_access_count": 0},
        "development_blockers": {"blocked_sequential_evidence": True, "blocked_static_baseline": not bool(frontier["core13_frontier_bound"]), "blocked_calibration_drift": True},
        "development_coevolution_status": "data_blocked",
        "formal_execution_allowed": False,
        "formal_claim_allowed": False,
        "coevolution_performance_claim": "not_evaluated",
        "next_milestone": "bind_registered_development_outcome_generator_real_agent_twin_checkpoints_and_qualified_core13_frontier",
    }
    write_json(output_root / "wp2_precondition_decision.json", decision)
    write_checksums(output_root)
    return decision
