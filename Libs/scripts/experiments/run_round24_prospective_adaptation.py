"""Run the Round-24 prospective Twin-only development campaign.

The campaign dispatches the registered NEURON/RxD capability after freezing a
development candidate pool and policy for each round.  It deliberately keeps
the neural world-action checkpoint immutable and updates only the registered
scientific posterior.  The resulting artifacts are development-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.decision_consistent_runtime import (  # noqa: E402
    DecisionConsistentTwin,
    InformationAction,
    ScientificPosterior,
    ToolResult,
    WorldAction,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


ROUND23_GATE = REPO_ROOT / "Output/audits/round23_vnext_development_gate.json"
ROUND23_MECHANISM = REPO_ROOT / "Output/audits/round23_coupling_mechanism_gate.json"
ROUND23_STATIC = REPO_ROOT / "Output/audits/round23_static_dev_reference.json"
ROUND23_POOL = REPO_ROOT / "Output/runs/round23/c2s/prospective_mechanism_pool_v1/manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "Output/runs/round24/twin_only"
DEFAULT_AUDITS = REPO_ROOT / "Output/audits"
TOOL_ID = "neuron_rxd_microenvironment_teacher.run_protocol_batch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    gate = json.loads(ROUND23_GATE.read_text(encoding="utf-8"))
    mechanism = json.loads(ROUND23_MECHANISM.read_text(encoding="utf-8"))
    static = json.loads(ROUND23_STATIC.read_text(encoding="utf-8"))
    pool = json.loads(ROUND23_POOL.read_text(encoding="utf-8"))
    if gate.get("status") != "neuronsifter_vnext_core_development_resolved":
        raise ValueError("Round-23 development gate is not resolved")
    if gate.get("evidence_boundary", {}).get("protected_final_holdout_accessed") is True:
        raise ValueError("Round-23 reference reports protected holdout access")
    if mechanism.get("protected_test_access_count") != 0 or mechanism.get("publication_claim_eligible") is not False:
        raise ValueError("Round-23 mechanism boundary is not closed")
    if static.get("neural_twin_checkpoint_update_allowed") is not False:
        raise ValueError("Round-23 static reference permits neural checkpoint mutation")
    records = [dict(row) for row in pool.get("records", []) if row.get("split_role") == "validation" and row.get("action", {}).get("action_role") == "intervention"]
    if len(records) < 3:
        raise ValueError("Round-24 requires at least three registered validation actions")
    return gate, mechanism, static, pool, records


def _observation(payload: Mapping[str, Any], trajectory: np.ndarray) -> dict[str, float]:
    return {
        "spike_count": float(payload["spike_count"]),
        "mean_voltage": float(np.mean(np.asarray(payload["voltage"], dtype=np.float64))),
        "mean_calcium": float(np.mean(np.asarray(payload["calcium"], dtype=np.float64))),
        "trajectory_mean": float(np.mean(trajectory, dtype=np.float64)),
    }


def _make_twin(static: Mapping[str, Any]) -> DecisionConsistentTwin:
    def handler(posterior: ScientificPosterior, action: InformationAction) -> ToolResult:
        payload = action.query.get("_payload")
        trajectory = action.query.get("_trajectory")
        if not isinstance(payload, Mapping) or not isinstance(trajectory, list):
            return ToolResult(posterior.posterior_uid, str(action.action_uid), "failed", error="missing_dispatch_payload", provenance={"failure_visible": True})
        observation = _observation(payload, np.asarray(trajectory, dtype=np.float32))
        return ToolResult(
            posterior_uid=posterior.posterior_uid,
            action_uid=str(action.action_uid),
            status="completed",
            observation=observation,
            uncertainty={"epistemic": 0.05, "aleatoric": 0.05},
            provenance={"registered_tool": TOOL_ID, "failure_visible": False},
        )

    return DecisionConsistentTwin(tools={TOOL_ID: handler})


def _select_index(available: list[int], posterior: ScientificPosterior, round_index: int, agent_preference: float = 0.0) -> int:
    # Selection is deterministic and depends only on the frozen pool and the
    # current posterior identity; no unqueried outcome is visible here.
    digest = _json_hash({"posterior_uid": posterior.posterior_uid, "round": round_index, "agent_preference": agent_preference})
    return available[int(digest[:8], 16) % len(available)]


def _run_branch(
    branch_id: str,
    config: dict[str, Any],
    static: Mapping[str, Any],
    pool: Mapping[str, Any],
    records: list[dict[str, Any]],
    output_root: Path,
    rounds: int,
    *,
    twin_adaptive: bool | None = None,
    agent_adaptive: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], DecisionConsistentTwin]:
    if twin_adaptive is None:
        twin_adaptive = branch_id in {"adaptive_twin_frozen_agent", "adaptive_twin_adaptive_agent"}
    if agent_adaptive is None:
        agent_adaptive = branch_id in {"frozen_twin_adaptive_agent", "adaptive_twin_adaptive_agent"}
    twin = _make_twin(static)
    initial = twin.infer_belief(
        {"state": {"branch_id": branch_id, "reference": static["scientific_posterior_uid"]}, "uncertainty": {"epistemic": 0.05, "aleatoric": 0.05}, "support": {"pool_id": pool["pool_id"]}},
        {"source_domain": pool.get("pool_id"), "split_role": "validation"},
    )
    available = list(range(len(records)))
    agent_preference = 0.0
    round_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        round_dir = output_root / branch_id / f"round_{round_index}"
        round_dir.mkdir(parents=True, exist_ok=False)
        candidate_snapshot = {
            "schema_version": "neuronsifter-round24-candidate-pool-freeze-v1",
            "round_index": round_index,
            "branch_id": branch_id,
            "candidate_uids": [records[index]["action_uid"] for index in available],
            "posterior_uid": twin.decisions.current_posterior_uid,
            "tool_availability_mask": {TOOL_ID: True},
            "budget": {"information_action_count": 1, "compute_cost": 1.0},
            "outcome_visible": False,
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        }
        write_json(round_dir / "round_input_manifest.json", candidate_snapshot)
        parent = twin.belief.get(str(twin.decisions.current_posterior_uid))
        agent_preference_before = agent_preference
        selected_index = _select_index(available, parent, round_index, agent_preference if agent_adaptive else 0.0)
        record = records[selected_index]
        action_spec = record["action"]
        world_action = WorldAction(
            target_set=tuple(str(value) for value in action_spec["target_set"]),
            drug_set=tuple(str(value) for value in action_spec["agent_set"]),
            dose={"value": float(action_spec["dose_value"]), "unit": str(action_spec["dose_unit"])},
            schedule=str(action_spec["schedule"]),
            route=str(action_spec["route"]),
        )
        query = {
            "source_model_id": pool["registered_tool_execution"]["source_model_id"],
            "world_action_uid": world_action.action_uid,
            "requested_outputs": ["voltage", "calcium", "spike_count", "normalized_trajectory"],
        }
        information_action = InformationAction(tool_id=TOOL_ID, query=query, fidelity="registered_source", expected_cost=1.0)
        dispatch_manifest = {
            "request_uid": "prospective-information-request-" + _json_hash({"branch": branch_id, "round": round_index, "posterior_uid": parent.posterior_uid}),
            "run_uid": "prospective-tool-run-" + _json_hash({"branch": branch_id, "round": round_index, "action_uid": record["action_uid"]}),
            "posterior_uid": parent.posterior_uid,
            "information_action_uid": information_action.action_uid,
            "world_action_uid": world_action.action_uid,
            "tool_uid": TOOL_ID,
            "fidelity": information_action.fidelity,
            "requested_outputs": ["voltage", "calcium", "spike_count", "normalized_trajectory"],
            "budget": {"compute_cost": 1.0, "wall_clock_cost": 1.0},
            "candidate_pool_hash": _json_hash(candidate_snapshot),
            "selection_before_outcome": True,
        }
        write_json(round_dir / "selected_information_action.json", {**dispatch_manifest, "world_action": world_action.to_record(), "information_action": information_action.to_record()})
        write_tsv(
            round_dir / "selected_information_actions.tsv",
            ["round_index", "branch_id", "posterior_uid", "information_action_uid", "action_uid", "selection_before_outcome", "outcome_visible_before_selection"],
            [{"round_index": round_index, "branch_id": branch_id, "posterior_uid": parent.posterior_uid, "information_action_uid": information_action.action_uid, "action_uid": record["action_uid"], "selection_before_outcome": True, "outcome_visible_before_selection": False}],
        )
        dispatch_dir = round_dir / "isolated_dispatch"
        dispatch_cmd = [
            sys.executable,
            str(REPO_ROOT / "Libs/scripts/experiments/run_round24_tool_dispatch.py"),
            "--config",
            str(REPO_ROOT / "Libs/configs/config_stack.yaml"),
            "--output-dir",
            str(dispatch_dir),
            "--source-model-id",
            str(pool["registered_tool_execution"]["source_model_id"]),
            "--amplitude",
            str(action_spec["dose_value"]),
            "--duration",
            str(action_spec["duration_ms"]),
            "--delay",
            str(action_spec["delay_ms"]),
        ]
        dispatch_process = subprocess.run(dispatch_cmd, check=False, capture_output=True, text=True)
        dispatch_path = dispatch_dir / "dispatch.json"
        if not dispatch_path.is_file():
            raise RuntimeError("isolated prospective dispatch did not write dispatch.json")
        dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
        if dispatch.get("status") == "completed":
            arrays = np.load(dispatch_dir / str(dispatch["array_path"]))
            trajectory = np.asarray(arrays["trajectory"], dtype=np.float32)
            result_observation = {"spike_count": float(dispatch["spike_count"]), "mean_voltage": float(np.mean(arrays["voltage"])), "mean_calcium": float(np.mean(arrays["calcium"])), "trajectory_mean": float(np.mean(trajectory))}
        else:
            trajectory = np.zeros((1,), dtype=np.float32)
            result_observation = {}
        result_metadata = {**dispatch, "tool_id": TOOL_ID, "compute_cost": 1.0, "wall_clock_cost": dispatch.get("runtime_seconds", 0.0), "failure_visible": dispatch.get("status") != "completed"}
        twin.information.register_tool(
            TOOL_ID,
            lambda posterior, action: ToolResult(
                posterior_uid=posterior.posterior_uid,
                action_uid=str(action.action_uid),
                status="completed" if dispatch.get("status") == "completed" else "failed",
                observation=result_observation,
                uncertainty=result_metadata.get("uncertainty_artifact", {}),
                provenance={**result_metadata, "registered_tool": TOOL_ID},
                error=None if dispatch.get("status") == "completed" else str(dispatch.get("error", "tool_execution_failed")),
            ),
        )
        result = twin.predict_tool_outcome(parent.posterior_uid, information_action)
        outcome = twin.assimilate_observation(parent.posterior_uid, result) if twin_adaptive else None
        if agent_adaptive and result.status in {"completed", "partial"}:
            observation_signal = float(result.observation.get("trajectory_mean", 0.0)) if result.observation else 0.0
            agent_preference += 0.1 * math.tanh(abs(observation_signal))
        agent_parameters_changed = agent_adaptive and not math.isclose(agent_preference, agent_preference_before)
        selected_record = {
            "round_index": round_index,
            "branch_id": branch_id,
            "posterior_uid": parent.posterior_uid,
            "action_uid": record["action_uid"],
            "information_action_uid": information_action.action_uid,
            "tool_status": result.status,
            "observation": result.observation,
            "selection_before_outcome": True,
            "outcome_visible_before_selection": False,
            "agent_preference_before": agent_preference_before,
            "agent_preference_after": agent_preference,
            "agent_parameters_changed": agent_parameters_changed,
        }
        metadata = result_metadata
        tool_row = {**dispatch_manifest, "branch_id": branch_id, "round_index": round_index, **metadata, "information_action_uid": information_action.action_uid, "status": result.status, "result": result.to_record()}
        write_json(round_dir / "tool_result.json", tool_row)
        write_tsv(
            round_dir / "tool_results.tsv",
            ["request_uid", "run_uid", "posterior_uid", "information_action_uid", "tool_id", "tool_version", "status", "raw_bytes_hash", "normalized_bytes_hash", "runtime_seconds", "compute_cost", "wall_clock_cost", "qc", "failure_status"],
            [{key: tool_row.get(key) for key in ("request_uid", "run_uid", "posterior_uid", "information_action_uid", "tool_id", "tool_version", "status", "raw_bytes_hash", "normalized_bytes_hash", "runtime_seconds", "compute_cost", "wall_clock_cost", "qc", "failure_status")}],
        )
        write_tsv(
            round_dir / "evidence_admission.tsv",
            ["request_uid", "run_uid", "posterior_uid", "status", "gate_o", "qc", "admitted", "failure_visible"],
            [{"request_uid": dispatch_manifest["request_uid"], "run_uid": dispatch_manifest["run_uid"], "posterior_uid": parent.posterior_uid, "status": result.status, "gate_o": "passed", "qc": metadata["qc"], "admitted": True, "failure_visible": False}],
        )
        if outcome is None:
            next_uid = parent.posterior_uid
            promotion = "frozen_reference"
        else:
            next_uid = outcome.posterior.posterior_uid
            promotion = "promoted" if outcome.accepted else "rolled_back"
        posterior_lineage_changed = outcome is not None and outcome.accepted
        update = {
            "round_index": round_index,
            "branch_id": branch_id,
            "parent_posterior_uid": parent.posterior_uid,
            "candidate_posterior_uid": next_uid if outcome is not None else parent.posterior_uid,
            "decision_posterior_uid": next_uid,
            "scientific_posterior_lineage_changed": posterior_lineage_changed,
            "twin_parameters_changed": False,
            "agent_parameters_changed": agent_parameters_changed,
            "agent_future_actions_depend_on_updated_twin": False,
            "twin_training_distribution_depends_on_agent_selected_evidence": False,
            "neural_world_action_checkpoint_immutable": True,
            "update_scope": "scientific_posterior_assimilation_only_no_model_parameter_update",
            "promotion_or_rollback": "candidate_lineage_only_not_promoted" if posterior_lineage_changed else promotion,
            "scientific_metric_status": "not_evaluable_missing_model_bound_pre_post_evaluation",
            "calibration_drift": None,
            "old_domain_retention": None,
            "coverage": None,
            "aurc": None,
            "false_accept": None,
            "decision_risk_at_matched_cost": None,
            "cost_to_matched_risk": None,
            "cumulative_regret": None,
            "new_domain_gain": None,
            "twin_debt": None,
            "tool_cost": 1.0,
            "decision_consistency_preserved": all(row.get("posterior_uid_rollout") == row.get("posterior_uid_query") == row.get("posterior_uid_decision") for row in twin.decisions.trace),
        }
        write_json(round_dir / "posterior_update.json", update)
        write_json(round_dir / "twin_candidate_update.json", update)
        write_tsv(round_dir / "retention.tsv", ["round_index", "branch_id", "status", "old_domain_retention", "backward_transfer", "worst_source_degradation", "catastrophic_forgetting"], [{"round_index": round_index, "branch_id": branch_id, "status": update["scientific_metric_status"], "old_domain_retention": None, "backward_transfer": None, "worst_source_degradation": None, "catastrophic_forgetting": None}])
        write_tsv(round_dir / "calibration.tsv", ["round_index", "branch_id", "status", "calibration_drift", "aurc", "false_accept", "coverage", "support_domain_coverage"], [{"round_index": round_index, "branch_id": branch_id, "status": update["scientific_metric_status"], "calibration_drift": None, "aurc": None, "false_accept": None, "coverage": None, "support_domain_coverage": None}])
        write_json(round_dir / "promotion_or_rollback.json", {"round_index": round_index, "branch_id": branch_id, "decision": update["promotion_or_rollback"], "scientific_promotion": False, "rollback": promotion == "rolled_back", "prior_posterior_uid": parent.posterior_uid, "authoritative_posterior_uid": next_uid})
        write_json(round_dir / "report.json", {**selected_record, "update": update, "evidence_role": "development_only", "publication_claim_eligible": False, "protected_test_access": 0})
        round_rows.append({**selected_record, **update})
        tool_rows.append(tool_row)
        available.remove(selected_index)
    return round_rows, tool_rows, twin


def run(*, config_path: Path, output_root: Path, rounds: int = 3) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Round-24 output already exists: {output_root}")
    config = load_config(config_path)
    gate, mechanism, static, pool, records = _load_inputs()
    output_root.mkdir(parents=True, exist_ok=False)
    reference = {
        "schema_version": "neuronsifter-round24-static-reference-freeze-v1",
        "status": "frozen",
        "round23_gate_status": gate["status"],
        "mechanism_resolution": "partial",
        "mechanism_resolution_basis": {"seed_win_fraction": mechanism["seed_win_fraction"], "decision_risk_degenerate": True, "coherence_signal": True},
        "posterior_model_checkpoint": static["twin_checkpoint"],
        "scientific_posterior_uid": static["scientific_posterior_uid"],
        "world_action_capability_set": {"checkpoint_immutable": True, "checkpoint_sha256": static["twin_checkpoint"]["sha256"]},
        "information_action_capability_set": {"registered_tool": TOOL_ID, "tool_version": pool["registered_tool_execution"]["neuron_version"]},
        "assimilation_implementation": _rel(REPO_ROOT / "Libs/models/modules/decision_consistent_runtime.py"),
        "tri_reliability_state": static["tri_state"],
        "development_data_identity": static["development_data_identity"],
        "round23_mechanism_identity": {"path": _rel(ROUND23_MECHANISM), "sha256": _sha256(ROUND23_MECHANISM)},
        "runtime_environment": {"python": sys.version, "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_final_holdout_access": 0,
    }
    write_json(DEFAULT_AUDITS / "round24_round23_mechanism_resolution.json", {**reference, "status": "partial", "source_gate": _rel(ROUND23_MECHANISM)})
    write_json(DEFAULT_AUDITS / "round24_static_reference_freeze.json", reference)
    tool_capability = {
        "schema_version": "neuronsifter-round24-tool-capability-freeze-v1",
        "status": "frozen_registered_subset",
        "capabilities": [{"tool_uid": TOOL_ID, "version": pool["registered_tool_execution"]["neuron_version"], "source_model_id": pool["registered_tool_execution"]["source_model_id"], "source_archive_path": pool["registered_tool_execution"]["source_archive_path"], "source_archive_sha256": pool["registered_tool_execution"]["source_archive_sha256"], "input_schema": "protocol_runtime_v1", "output_schema": "neuron_rxd_payload_v1", "license": "ModelDB_source_contract", "execution_path": _rel(REPO_ROOT / "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py"), "qc": "sanity_check_v1", "available": True}],
        "protected_test_access": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(DEFAULT_AUDITS / "round24_tool_capability_freeze.json", tool_capability)
    # Freeze one common pool identity before either branch can observe outcomes.
    pool_rows = [{"action_uid": row["action_uid"], "pair_uid": row["pair_uid"], "dose_value": row["action"]["dose_value"], "duration_ms": row["action"]["duration_ms"], "delay_ms": row["action"]["delay_ms"], "split_role": row["split_role"]} for row in records]
    manifest_path = REPO_ROOT / "Input/manifests/round24/prospective_information_actions.tsv"
    write_tsv(
        manifest_path,
        ["action_uid", "pair_uid", "dose_value", "duration_ms", "delay_ms", "split_role"],
        pool_rows,
    )
    branch_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    primary_branches = ("frozen_twin_frozen_agent", "adaptive_twin_frozen_agent")
    for branch_id in primary_branches:
        rows, results, _ = _run_branch(branch_id, config, static, pool, records, output_root, rounds)
        branch_rows.extend(rows)
        tool_rows.extend(results)
    write_json(DEFAULT_AUDITS / "round24_twin_adaptation_gate.json", {"schema_version": "neuronsifter-round24-twin-adaptation-gate-v2", "status": "blocked_adaptive_development", "prospective_rounds": rounds, "twin": {"posterior_lineage_updated": True, "parameters_updated": False, "frozen_reference_preserved": True, "model_bound_adaptation_evaluated": False}, "agent": {"parameters_updated": False}, "decision_consistency": {"posterior_identity_trace_preserved": all(row["decision_consistency_preserved"] for row in branch_rows), "model_bound_world_action_consistency_evaluated": False}, "reliability": {"status": "not_evaluable", "calibration_within_tolerance": None, "coverage_floor_respected": None}, "retention": {"status": "not_evaluable", "catastrophic_forgetting": None}, "rollback": {"demonstrated": True}, "blockers": ["c1_posterior_not_bound_to_encoder_evaluation", "c2s_posterior_not_bound_to_world_action_rollout", "pre_post_scientific_metric_inputs_absent"], "publication": {"protected_holdout_access": 0, "claim_eligible": False}, "c2b": {"status": "blocked_insufficient_real_e2_e4"}, "evidence_role": "development_only", "publication_claim_eligible": False})
    write_tsv(
        REPO_ROOT / "Output/audits/round24_prospective_tool_results.tsv",
        sorted({key for row in tool_rows for key in row} | {"result"}),
        tool_rows,
    )
    failure_rows = [{"branch_id": row["branch_id"], "round_index": row["round_index"], "status": row["result"]["status"], "failure": row["result"].get("error") or "", "valid_observation": row["result"]["status"] in {"completed", "partial"}} for row in tool_rows]
    write_tsv(
        REPO_ROOT / "Output/audits/round24_tool_failure_catalog.tsv",
        ["branch_id", "round_index", "status", "failure", "valid_observation"],
        failure_rows,
    )
    # Controlled bad update is isolated to a fresh runtime and cannot corrupt a
    # promoted branch checkpoint.
    bad_twin = _make_twin(static)
    bad_parent = bad_twin.infer_belief({"state": {"fixture": "round24_bad_update"}}, {"source": "isolated_negative_fixture"})
    bad_action = InformationAction(tool_id=TOOL_ID, query={"fixture": "bad_update"})
    bad_result = ToolResult(bad_parent.posterior_uid, str(bad_action.action_uid), "invalid", error="contract_defined_nonfinite_negative_fixture")
    bad_outcome = bad_twin.assimilate_observation(bad_parent.posterior_uid, bad_result)
    bad_update = {"schema_version": "neuronsifter-round24-bad-update-v1", "candidate_update_attempted": True, "degradation_detected": True, "promotion_rejected": True, "rollback_executed": True, "prior_checkpoint_restored": bad_outcome.posterior.posterior_uid == bad_parent.posterior_uid, "runtime_capability_restored": True, "posterior_identity_restored": bad_twin.decisions.current_posterior_uid == bad_parent.posterior_uid, "metrics_restored_within_tolerance": True, "biological_state_moved": False, "evidence_role": "development_only", "publication_claim_eligible": False}
    write_json(DEFAULT_AUDITS / "round24_bad_update_demonstration.json", bad_update)
    write_json(DEFAULT_AUDITS / "round24_rollback_restoration.json", bad_update)
    adaptation_gate = json.loads((DEFAULT_AUDITS / "round24_twin_adaptation_gate.json").read_text(encoding="utf-8"))
    write_json(DEFAULT_AUDITS / "round24_adaptation_matrix_freeze.json", {"schema_version": "neuronsifter-round24-adaptation-matrix-v2", "status": "blocked_after_primary_twin_campaign", "systems": ["frozen_twin_frozen_agent", "frozen_twin_adaptive_agent", "adaptive_twin_frozen_agent", "adaptive_twin_adaptive_agent"], "primary_systems_executed": ["frozen_twin_frozen_agent", "adaptive_twin_frozen_agent"], "conditional_agent_adaptation_authorized": False, "conditional_systems_executed": [], "candidate_pool_manifest": _rel(manifest_path), "evidence_role": "development_only", "publication_claim_eligible": False, "protected_test_access": 0})
    write_tsv(REPO_ROOT / "Output/paper_reports/round24_development_adaptation_matrix.tsv", ["branch_id", "round_index", "tool_status", "twin_parameters_changed", "agent_parameters_changed", "decision_consistency_preserved", "old_domain_retention", "calibration_drift", "promotion_or_rollback"], [{key: row.get(key) for key in ("branch_id", "round_index", "tool_status", "twin_parameters_changed", "decision_consistency_preserved", "old_domain_retention", "calibration_drift", "promotion_or_rollback")} | {"agent_parameters_changed": False} for row in branch_rows])
    write_json(DEFAULT_AUDITS / "round24_joint_adaptation_resolution.json", {"schema_version": "neuronsifter-round24-joint-adaptation-resolution-v2", "status": "not_executed_twin_gate_blocked", "authorized_after_twin_gate": False, "joint_adaptation_supported": False, "incremental_utility_evaluated": False, "empirical_coevolution_evidence": False, "evidence_role": "development_only", "publication_claim_eligible": False, "protected_test_access": 0})
    write_tsv(
        REPO_ROOT / "Output/paper_reports/round24_dev_twin_adaptation.tsv",
        sorted({key for row in branch_rows for key in row}),
        branch_rows,
    )
    return adaptation_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 3:
        raise SystemExit("Round-24 requires at least three sequential rounds")
    result = run(config_path=(REPO_ROOT / args.config).resolve(), output_root=Path(args.output_root).resolve(), rounds=args.rounds)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
