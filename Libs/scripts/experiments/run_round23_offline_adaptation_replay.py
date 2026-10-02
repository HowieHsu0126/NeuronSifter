"""Run the frozen Round-23 Agent-Twin adaptation matrix as offline replay.

The source outcomes were generated prospectively for the Round-23 mechanism
pool. This runner replays them without new tool dispatches, keeps the neural
Twin checkpoint immutable, and limits Twin adaptation to the scientific
posterior calibration layer. The resulting evidence is development-only and
is not empirical Co-Evolution evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.experiments.run_round23_shared_belief_replay import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    GaussianBelief,
    _action_design_matrix,
    _expected_decision_risk_after_query,
    _fit_prior,
    _load_candidate_data,
    _load_world_action_outputs,
    _predictive_variance,
    _response_values,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    write_json,
    write_tsv,
)


DEFAULT_POOL_MANIFEST = (
    "Output/runs/round23/c2s/prospective_mechanism_pool_v1/manifest.json"
)
DEFAULT_MECHANISM_GATE = "Output/audits/round23_coupling_mechanism_gate.json"
DEFAULT_OUTPUT_ROOT = "Output/runs/round23/adaptation/offline_replay_v1"
STATIC_REFERENCE_PATH = "Output/audits/round23_static_dev_reference.json"
ADAPTATION_MATRIX_PATH = "Output/audits/round23_adaptation_matrix.json"
ROLLBACK_PATH = "Output/audits/round23_rollback_validation.json"
ADAPTATION_REPORT_PATH = "Output/paper_reports/round23_dev_adaptation_results.tsv"


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    twin_adaptive: bool
    agent_adaptive: bool
    update_mode: str
    branch_role: str


BRANCHES = (
    BranchSpec("frozen_twin_frozen_agent", False, False, "none", "primary_2x2"),
    BranchSpec("frozen_twin_adaptive_agent", False, True, "none", "primary_2x2"),
    BranchSpec("adaptive_twin_frozen_agent", True, False, "incremental", "primary_2x2"),
    BranchSpec("adaptive_twin_adaptive_agent", True, True, "incremental", "primary_2x2"),
    BranchSpec("latest_batch_joint_update", True, True, "latest_batch", "control"),
    BranchSpec("full_replay_joint_update", True, True, "full_replay", "control"),
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _belief_uid(parent_uid: str, event: Mapping[str, Any], belief: GaussianBelief) -> str:
    return "posterior_" + _json_hash(
        {
            "parent_uid": parent_uid,
            "event": dict(event),
            "mean": np.asarray(belief.mean, dtype=np.float64).tolist(),
            "covariance": np.asarray(belief.covariance, dtype=np.float64).tolist(),
        }
    )[:24]


def _load_mechanism_identity(
    gate_path: Path,
    pool_path: Path,
    *,
    expected_attempt_id: str,
) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "neuronsifter-round23-coupling-mechanism-gate-v2",
        "attempt_id": expected_attempt_id,
        "development_gate_passed": True,
        "execution_mode": "prospective_development_tool_replay",
        "fixture_rows_used": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access_count": 0,
        "world_action_checkpoint_identity_verified": True,
    }
    for field, expected in required.items():
        if gate.get(field) != expected:
            raise ValueError(
                f"Round-23 adaptation source mechanism mismatch: {field}={gate.get(field)!r}"
            )
    pool_identity = gate.get("candidate_pool_identity", {})
    if (
        pool_identity.get("prospective_dispatch") is not True
        or pool_identity.get("manifest_sha256") != _sha256_file(pool_path)
        or _repo_path(str(pool_identity.get("manifest_path"))) != pool_path
    ):
        raise ValueError("Round-23 adaptation pool identity drift")
    report_path = _repo_path(str(gate["report_path"]))
    coherence_path = _repo_path(str(gate["coherence_report_path"]))
    if gate.get("report_sha256") != _sha256_file(report_path):
        raise ValueError("Round-23 mechanism report hash drift")
    if gate.get("coherence_report_sha256") != _sha256_file(coherence_path):
        raise ValueError("Round-23 coherence report hash drift")
    attempt_gate = _repo_path(str(gate["attempt_gate_path"]))
    attempt_root = attempt_gate.parent
    attempt_manifest_path = attempt_root / "manifest.json"
    attempt_manifest = json.loads(attempt_manifest_path.read_text(encoding="utf-8"))
    if attempt_manifest.get("attempt_id") != gate["attempt_id"]:
        raise ValueError("Round-23 mechanism attempt lineage drift")
    artifacts = attempt_manifest.get("artifacts", {})
    if artifacts.get("gate.json") != _sha256_file(attempt_gate):
        raise ValueError("Round-23 mechanism attempt gate hash drift")
    if attempt_gate.read_bytes() != gate_path.read_bytes():
        raise ValueError("Authoritative and attempt-local mechanism gates differ")
    for name, expected in (
        ("shared_vs_decoupled_dev.tsv", gate["report_sha256"]),
        ("belief_coherence_diagnostics.tsv", gate["coherence_report_sha256"]),
    ):
        attempt_path = attempt_root / name
        if artifacts.get(name) != expected or _sha256_file(attempt_path) != expected:
            raise ValueError(f"Round-23 mechanism attempt artifact drift: {name}")
    return {
        "gate": gate,
        "gate_path": _repo_relative(gate_path),
        "gate_sha256": _sha256_file(gate_path),
        "attempt_manifest_path": _repo_relative(attempt_manifest_path),
        "attempt_manifest_sha256": _sha256_file(attempt_manifest_path),
        "pool_manifest_path": _repo_relative(pool_path),
        "pool_manifest_sha256": _sha256_file(pool_path),
    }


def _normalized_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    spread = float(np.ptp(values))
    if not math.isfinite(spread) or spread <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.min(values))) / spread


def _policy_entropy(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        return 0.0
    shifted = scores - float(np.max(scores))
    probabilities = np.exp(np.clip(shifted, -50.0, 50.0))
    probabilities /= float(np.sum(probabilities))
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return entropy / math.log(scores.size)


def _select_query(
    belief: GaussianBelief,
    candidate_designs: np.ndarray,
    available_indices: Sequence[int],
    agent_preferences: np.ndarray,
    *,
    agent_adaptive: bool,
    noise_variance: float,
    seed: int,
) -> tuple[int, np.ndarray, float]:
    """Select from belief and policy state only; outcomes are intentionally absent."""

    available = np.asarray(available_indices, dtype=np.int64)
    designs = candidate_designs[available]
    rng = np.random.default_rng(seed)
    prior_risk = float(
        np.mean(
            np.maximum(
                0.0,
                np.max(designs @ belief.mean)
                - designs @ belief.mean,
            )
        )
    )
    decision_utility = np.asarray(
        [
            prior_risk
            - _expected_decision_risk_after_query(
                belief, design, candidate_designs, noise_variance, rng
            )
            for design in designs
        ],
        dtype=np.float64,
    )
    scores = _normalized_scores(decision_utility)
    if agent_adaptive:
        scores = scores + _normalized_scores(designs @ agent_preferences)
    selected_local = int(np.argmax(scores))
    return int(available[selected_local]), scores, _policy_entropy(scores)


def _update_agent_preferences(
    preferences: np.ndarray,
    design: np.ndarray,
    observation: float,
    prediction: float,
    *,
    learning_rate: float,
    noise_scale: float,
) -> np.ndarray:
    direction = np.asarray(design, dtype=np.float64).copy()
    direction[0] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm > 0.0:
        direction /= norm
    surprise = math.tanh(abs(float(observation) - float(prediction)) / noise_scale)
    updated = np.asarray(preferences, dtype=np.float64) + learning_rate * surprise * direction
    if not np.isfinite(updated).all():
        raise ValueError("Round-23 adaptive Agent produced non-finite preferences")
    return updated


def _coverage(
    belief: GaussianBelief,
    designs: np.ndarray,
    truth: np.ndarray,
    noise_variance: float,
    interval_level: float,
) -> float:
    z_value = NormalDist().inv_cdf((1.0 + interval_level) / 2.0)
    mean = designs @ belief.mean
    standard_deviation = np.sqrt(_predictive_variance(belief, designs, noise_variance))
    return float(np.mean(np.abs(truth - mean) <= z_value * standard_deviation))


def _support_coverage(
    candidate_designs: np.ndarray, evidence_designs: np.ndarray
) -> float:
    lower = np.min(evidence_designs, axis=0) - 1e-12
    upper = np.max(evidence_designs, axis=0) + 1e-12
    supported = np.all(
        (candidate_designs >= lower[None, :])
        & (candidate_designs <= upper[None, :]),
        axis=1,
    )
    return float(np.mean(supported))


def _evaluate_belief(
    belief: GaussianBelief,
    *,
    old_designs: np.ndarray,
    old_truth: np.ndarray,
    new_designs: np.ndarray,
    new_truth: np.ndarray,
    evidence_designs: np.ndarray,
    noise_variance: float,
    interval_level: float,
    reliability_accept_threshold: float,
    baseline_old_rmse: float,
    baseline_old_coverage: float,
) -> dict[str, float | int]:
    old_prediction = old_designs @ belief.mean
    old_rmse = float(np.sqrt(np.mean(np.square(old_truth - old_prediction))))
    old_retention = min(1.0, baseline_old_rmse / max(old_rmse, 1e-12))
    old_coverage = _coverage(
        belief, old_designs, old_truth, noise_variance, interval_level
    )
    calibration_drift = abs(old_coverage - baseline_old_coverage)

    new_prediction = new_designs @ belief.mean
    new_variance = _predictive_variance(belief, new_designs, noise_variance)
    new_standard_deviation = np.sqrt(new_variance)
    response_range = max(float(np.ptp(new_truth)), 1e-8)
    selected = int(np.argmax(new_prediction))
    decision_risk = float((np.max(new_truth) - new_truth[selected]) / response_range)
    normalized_error = np.abs(new_truth - new_prediction) / response_range
    confidence_order = np.argsort(new_standard_deviation, kind="stable")
    cumulative_risk = np.cumsum(normalized_error[confidence_order]) / np.arange(
        1, len(new_truth) + 1
    )
    aurc = float(np.mean(cumulative_risk))
    reliability = np.clip(1.0 - new_standard_deviation / response_range, 0.0, 1.0)
    accepted = reliability >= reliability_accept_threshold
    z_value = NormalDist().inv_cdf((1.0 + interval_level) / 2.0)
    interval_miss = np.abs(new_truth - new_prediction) > z_value * new_standard_deviation
    false_accept = float(np.mean(interval_miss[accepted])) if np.any(accepted) else 0.0
    support_coverage = _support_coverage(new_designs, evidence_designs)
    new_coverage = float(np.mean(~interval_miss))
    twin_debt = float(
        np.mean(normalized_error) + abs(new_coverage - interval_level)
    )
    return {
        "new_domain_utility": 1.0 - decision_risk,
        "decision_risk": decision_risk,
        "selected_intervention_index": selected,
        "old_domain_rmse": old_rmse,
        "old_domain_retention": old_retention,
        "old_domain_calibration_coverage": old_coverage,
        "calibration_drift": calibration_drift,
        "aurc": aurc,
        "false_accept": false_accept,
        "accepted_action_count": int(np.sum(accepted)),
        "support_coverage": support_coverage,
        "twin_debt": twin_debt,
        "new_domain_calibration_coverage": new_coverage,
    }


def _candidate_belief(
    update_mode: str,
    *,
    prior: GaussianBelief,
    current: GaussianBelief,
    selected_designs: Sequence[np.ndarray],
    selected_observations: Sequence[float],
    noise_variance: float,
) -> GaussianBelief:
    if update_mode == "incremental":
        return current.update(
            selected_designs[-1], selected_observations[-1], noise_variance
        )
    if update_mode == "latest_batch":
        return prior.update(
            selected_designs[-1], selected_observations[-1], noise_variance
        )
    if update_mode == "full_replay":
        candidate = GaussianBelief(prior.mean.copy(), prior.covariance.copy())
        for design, observation in zip(
            selected_designs, selected_observations, strict=True
        ):
            candidate = candidate.update(design, observation, noise_variance)
        return candidate
    if update_mode == "none":
        return current
    raise ValueError(f"Unsupported Round-23 Twin update mode: {update_mode}")


def _run_branch(
    spec: BranchSpec,
    *,
    prior: GaussianBelief,
    prior_uid: str,
    old_designs: np.ndarray,
    old_truth: np.ndarray,
    candidate_designs: np.ndarray,
    candidate_truth: np.ndarray,
    candidate_action_uids: Sequence[str],
    candidate_pair_uids: Sequence[str],
    protocol: Mapping[str, Any],
    noise_variance: float,
    baseline_metrics: Mapping[str, float | int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    round_count = int(protocol["sequential_round_count"])
    if round_count < 3 or round_count >= len(candidate_action_uids):
        raise ValueError("Round-23 adaptation requires three bounded query rounds")
    current = GaussianBelief(prior.mean.copy(), prior.covariance.copy())
    current_uid = prior_uid
    preferences = np.zeros(candidate_designs.shape[1], dtype=np.float64)
    available = list(range(len(candidate_action_uids)))
    selected_indices: list[int] = []
    selected_designs: list[np.ndarray] = []
    selected_observations: list[float] = []
    rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    policy_entropies: list[float] = []
    promotion_count = 0
    promotion_rollback_count = 0
    invalid_probe_count = 0
    invalid_probe_pass_count = 0
    minimum_retention = float(protocol["minimum_old_domain_retention"])
    maximum_drift = float(protocol["maximum_calibration_drift"])
    interval_level = float(protocol["calibration_interval_level"])
    accept_threshold = float(protocol["reliability_accept_threshold"])
    learning_rate = float(protocol["agent_preference_learning_rate"])
    noise_scale = math.sqrt(noise_variance)

    for round_index in range(1, round_count + 1):
        query_index, _query_scores, entropy = _select_query(
            current,
            candidate_designs,
            available,
            preferences,
            agent_adaptive=spec.agent_adaptive,
            noise_variance=noise_variance,
            seed=23_000 + round_index,
        )
        policy_entropies.append(entropy)
        design = candidate_designs[query_index]
        observation = float(candidate_truth[query_index])
        prediction_before = float(design @ current.mean)
        parent_uid = current_uid
        selected_indices.append(query_index)
        selected_designs.append(design.copy())
        selected_observations.append(observation)
        available.remove(query_index)

        if spec.agent_adaptive:
            preferences = _update_agent_preferences(
                preferences,
                design,
                observation,
                prediction_before,
                learning_rate=learning_rate,
                noise_scale=noise_scale,
            )

        invalid_probe_passed = False
        if spec.twin_adaptive and round_index == round_count:
            invalid_probe_count += 1
            before_mean = current.mean.copy()
            before_covariance = current.covariance.copy()
            try:
                current.update(design, float("nan"), noise_variance)
            except ValueError:
                invalid_probe_passed = bool(
                    np.array_equal(current.mean, before_mean)
                    and np.array_equal(current.covariance, before_covariance)
                    and current_uid == parent_uid
                )
            if invalid_probe_passed:
                invalid_probe_pass_count += 1
            rollback_rows.append(
                {
                    "branch_id": spec.branch_id,
                    "round_index": round_index,
                    "rollback_type": "invalid_observation_probe",
                    "reason": "nonfinite_observation",
                    "parent_posterior_uid": parent_uid,
                    "restored_posterior_uid": current_uid,
                    "biological_state_moved": False,
                    "rollback_correct": invalid_probe_passed,
                    "evidence_role": "development_only",
                    "publication_claim_eligible": False,
                }
            )

        update_attempted = spec.twin_adaptive
        update_promoted = False
        rollback_reason = "not_applicable_frozen_twin"
        candidate_uid = current_uid
        candidate_metrics: dict[str, float | int] | None = None
        if update_attempted:
            candidate = _candidate_belief(
                spec.update_mode,
                prior=prior,
                current=current,
                selected_designs=selected_designs,
                selected_observations=selected_observations,
                noise_variance=noise_variance,
            )
            candidate_uid = _belief_uid(
                parent_uid,
                {
                    "branch_id": spec.branch_id,
                    "round_index": round_index,
                    "update_mode": spec.update_mode,
                    "action_uid": candidate_action_uids[query_index],
                    "observation": observation,
                },
                candidate,
            )
            evidence_designs = np.concatenate(
                [old_designs, np.asarray(selected_designs)], axis=0
            )
            candidate_metrics = _evaluate_belief(
                candidate,
                old_designs=old_designs,
                old_truth=old_truth,
                new_designs=candidate_designs,
                new_truth=candidate_truth,
                evidence_designs=evidence_designs,
                noise_variance=noise_variance,
                interval_level=interval_level,
                reliability_accept_threshold=accept_threshold,
                baseline_old_rmse=float(baseline_metrics["old_domain_rmse"]),
                baseline_old_coverage=float(
                    baseline_metrics["old_domain_calibration_coverage"]
                ),
            )
            finite_candidate = bool(
                np.isfinite(candidate.mean).all()
                and np.isfinite(candidate.covariance).all()
                and all(
                    math.isfinite(float(value))
                    for value in candidate_metrics.values()
                )
            )
            update_promoted = bool(
                finite_candidate
                and float(candidate_metrics["old_domain_retention"])
                >= minimum_retention
                and float(candidate_metrics["calibration_drift"])
                <= maximum_drift
            )
            if update_promoted:
                current = candidate
                current_uid = candidate_uid
                promotion_count += 1
                rollback_reason = "none"
            else:
                promotion_rollback_count += 1
                reasons = []
                if not finite_candidate:
                    reasons.append("nonfinite_candidate")
                if float(candidate_metrics["old_domain_retention"]) < minimum_retention:
                    reasons.append("old_domain_retention")
                if float(candidate_metrics["calibration_drift"]) > maximum_drift:
                    reasons.append("calibration_drift")
                rollback_reason = "+".join(reasons) or "promotion_gate_failed"
                rollback_rows.append(
                    {
                        "branch_id": spec.branch_id,
                        "round_index": round_index,
                        "rollback_type": "candidate_promotion_rollback",
                        "reason": rollback_reason,
                        "candidate_posterior_uid": candidate_uid,
                        "parent_posterior_uid": parent_uid,
                        "restored_posterior_uid": current_uid,
                        "biological_state_moved": False,
                        "rollback_correct": current_uid == parent_uid,
                        "evidence_role": "development_only",
                        "publication_claim_eligible": False,
                    }
                )

        evidence_designs = np.concatenate(
            [old_designs, np.asarray(selected_designs)], axis=0
        )
        metrics = _evaluate_belief(
            current,
            old_designs=old_designs,
            old_truth=old_truth,
            new_designs=candidate_designs,
            new_truth=candidate_truth,
            evidence_designs=evidence_designs,
            noise_variance=noise_variance,
            interval_level=interval_level,
            reliability_accept_threshold=accept_threshold,
            baseline_old_rmse=float(baseline_metrics["old_domain_rmse"]),
            baseline_old_coverage=float(
                baseline_metrics["old_domain_calibration_coverage"]
            ),
        )
        rows.append(
            {
                "branch_id": spec.branch_id,
                "branch_role": spec.branch_role,
                "twin": "adaptive" if spec.twin_adaptive else "frozen",
                "agent": "adaptive" if spec.agent_adaptive else "frozen",
                "update_mode": spec.update_mode,
                "round_index": round_index,
                "selected_action_uid": candidate_action_uids[query_index],
                "selected_pair_uid": candidate_pair_uids[query_index],
                "selection_before_outcome": True,
                "unqueried_outcomes_used_for_policy": False,
                "revealed_observation": observation,
                "prediction_before_observation": prediction_before,
                "parent_posterior_uid": parent_uid,
                "candidate_posterior_uid": candidate_uid,
                "decision_posterior_uid": current_uid,
                "update_attempted": update_attempted,
                "update_promoted": update_promoted,
                "rollback_reason": rollback_reason,
                "invalid_probe_passed": invalid_probe_passed,
                "policy_entropy": entropy,
                "agent_preference_norm": float(np.linalg.norm(preferences)),
                **metrics,
                "compute_tool_cost": float(round_index)
                * float(protocol["information_action_cost"]),
                "prospective_tool_dispatch_count": 0,
                "offline_replayed_outcome_count": round_index,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_test_access": 0,
            }
        )

    final = dict(rows[-1])
    final.update(
        {
            "status": "completed_offline_replay",
            "round_count": round_count,
            "selected_action_uids": ",".join(
                candidate_action_uids[index] for index in selected_indices
            ),
            "policy_entropy": float(np.mean(policy_entropies)),
            "promotion_count": promotion_count,
            "promotion_rollback_count": promotion_rollback_count,
            "invalid_probe_count": invalid_probe_count,
            "invalid_probe_pass_count": invalid_probe_pass_count,
            "rollback_correct": all(
                bool(row["rollback_correct"])
                for row in rollback_rows
                if row["branch_id"] == spec.branch_id
            ),
        }
    )
    return rows, final, rollback_rows


def _guard_required_output_replacement() -> None:
    matrix_path = _repo_path(ADAPTATION_MATRIX_PATH)
    if not matrix_path.is_file():
        return
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if payload.get("status") != "offline_replay_only_prospective_tool_execution_unavailable":
        raise FileExistsError(
            "Authoritative Round-23 adaptation evidence already exists and cannot be overwritten"
        )


def run(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT_PATH,
    checkpoint_identity: str = DEFAULT_CHECKPOINT_PATH,
    device: str = "cpu",
    pool_manifest: str = DEFAULT_POOL_MANIFEST,
    pool_identity: str = DEFAULT_POOL_MANIFEST,
    mechanism_gate: str = DEFAULT_MECHANISM_GATE,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    _guard_required_output_replacement()
    output_root_path = _repo_path(output_root)
    if output_root_path.exists():
        raise FileExistsError(f"Round-23 adaptation run already exists: {output_root}")
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    protocol = config["neuronsifter_vnext"]["adaptation"]["offline_protocol"]
    source_attempt_id = str(protocol.get("source_mechanism_attempt_id", ""))
    if not source_attempt_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in source_attempt_id
    ):
        raise ValueError("Round-23 adaptation source attempt is not frozen")
    if protocol.get("neural_twin_checkpoint_update_allowed") is not False:
        raise ValueError("Round-23 adaptation must keep the neural checkpoint immutable")
    if protocol.get("twin_update_scope") != "scientific_posterior_calibration_layer_only":
        raise ValueError("Round-23 Twin adaptation scope drift")

    pool_path = _repo_path(pool_manifest)
    pool_identity_path = _repo_path(pool_identity)
    gate_path = _repo_path(mechanism_gate)
    source_identity = _load_mechanism_identity(
        gate_path,
        pool_identity_path,
        expected_attempt_id=source_attempt_id,
    )
    if _sha256_file(pool_path) != _sha256_file(pool_identity_path):
        raise ValueError("Staged Round-23 adaptation pool manifest changed identity")
    pool_payload = json.loads(pool_identity_path.read_text(encoding="utf-8"))
    protected = pool_payload.get("protected_boundary", {})
    if any(
        protected.get(field) is not False
        for field in (
            "canonical_c2s_trajectory_store_accessed",
            "canonical_final_holdout_result_accessed",
            "protected_action_uids_referenced",
        )
    ) or int(protected.get("protected_test_access_count", -1)) != 0:
        raise ValueError("Round-23 adaptation source pool crossed the protected boundary")

    data = _load_candidate_data(
        config,
        candidate_pool=str(pool_path),
        candidate_pool_identity=str(pool_identity_path),
    )
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_identity_path = _repo_path(checkpoint_identity)
    checkpoint_hash_before = _sha256_file(checkpoint_identity_path)
    truth, prediction, prediction_std, checkpoint_metadata = _load_world_action_outputs(
        config,
        data,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        device=device,
    )
    if checkpoint_metadata["checkpoint_sha256"] != source_identity["gate"][
        "world_action_checkpoint_sha256"
    ]:
        raise ValueError("Round-23 adaptation checkpoint differs from the mechanism reference")
    species_ids = [
        str(row["species_id"])
        for row in config["dynamics_ready_registry"]["species"]
    ]
    (
        response_truth,
        response_prediction,
        _response_prediction_std,
        action_uids,
        pair_uids,
    ) = _response_values(
        data["records"], truth, prediction, prediction_std, species_ids
    )
    split_role_by_action = {
        str(row["action"]["action_uid"]): str(row["split_role"])
        for row in data["records"]
        if str(row["action"]["action_role"]) == "intervention"
    }
    train_action_uids = {
        action_uid
        for action_uid, role in split_role_by_action.items()
        if role == "train"
    }
    designs, _, _ = _action_design_matrix(
        data["records"], action_uids, train_action_uids
    )
    train_positions = [
        index for index, uid in enumerate(action_uids) if split_role_by_action[uid] == "train"
    ]
    validation_positions = [
        index
        for index, uid in enumerate(action_uids)
        if split_role_by_action[uid] == "validation"
    ]
    if len(train_positions) != 6 or len(validation_positions) != 6:
        raise ValueError("Round-23 adaptation requires six train and six validation pairs")
    old_designs = designs[train_positions]
    old_truth = response_truth[train_positions]
    candidate_designs = designs[validation_positions]
    candidate_truth = response_truth[validation_positions]
    candidate_action_uids = [action_uids[index] for index in validation_positions]
    candidate_pair_uids = [pair_uids[index] for index in validation_positions]
    noise_scale = max(float(np.std(old_truth)) * 0.25, 1e-3)
    noise_variance = noise_scale**2
    prior = _fit_prior(
        old_designs, response_prediction[train_positions], noise_variance
    )
    prior_uid = _belief_uid(
        "none",
        {
            "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
            "pool_manifest_sha256": source_identity["pool_manifest_sha256"],
            "role": "frozen_static_scientific_posterior",
        },
        prior,
    )
    baseline_metrics = _evaluate_belief(
        prior,
        old_designs=old_designs,
        old_truth=old_truth,
        new_designs=candidate_designs,
        new_truth=candidate_truth,
        evidence_designs=old_designs,
        noise_variance=noise_variance,
        interval_level=float(protocol["calibration_interval_level"]),
        reliability_accept_threshold=float(
            protocol["reliability_accept_threshold"]
        ),
        baseline_old_rmse=float(
            np.sqrt(
                np.mean(
                    np.square(old_truth - old_designs @ prior.mean)
                )
            )
        ),
        baseline_old_coverage=_coverage(
            prior,
            old_designs,
            old_truth,
            noise_variance,
            float(protocol["calibration_interval_level"]),
        ),
    )

    round_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    for spec in BRANCHES:
        branch_rounds, branch_final, branch_rollbacks = _run_branch(
            spec,
            prior=prior,
            prior_uid=prior_uid,
            old_designs=old_designs,
            old_truth=old_truth,
            candidate_designs=candidate_designs,
            candidate_truth=candidate_truth,
            candidate_action_uids=candidate_action_uids,
            candidate_pair_uids=candidate_pair_uids,
            protocol=protocol,
            noise_variance=noise_variance,
            baseline_metrics=baseline_metrics,
        )
        round_rows.extend(branch_rounds)
        final_rows.append(branch_final)
        rollback_rows.extend(branch_rollbacks)

    checkpoint_hash_after = _sha256_file(checkpoint_identity_path)
    checkpoint_immutable = checkpoint_hash_before == checkpoint_hash_after
    if not checkpoint_immutable:
        raise RuntimeError("Round-23 adaptation mutated the neural Twin checkpoint")
    output_root_path.mkdir(parents=True, exist_ok=False)
    round_path = output_root_path / "branch_rounds.tsv"
    rollback_rows_path = output_root_path / "rollback_events.tsv"
    write_tsv(round_path, list(round_rows[0]), round_rows)
    write_tsv(
        rollback_rows_path,
        list(rollback_rows[0]),
        rollback_rows,
    )

    static_reference = {
        "schema_version": "neuronsifter-round23-static-development-reference-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_development_reference",
        "frozen_twin": True,
        "frozen_agent": True,
        "frozen_tri": True,
        "frozen_development_data_identity": True,
        "frozen_tool_registry": True,
        "neural_twin_checkpoint_immutable": checkpoint_immutable,
        "neural_twin_checkpoint_update_allowed": False,
        "twin_checkpoint": {
            "path": checkpoint_metadata["checkpoint_path"],
            "sha256": checkpoint_metadata["checkpoint_sha256"],
            "run_uid": checkpoint_metadata["checkpoint_run_uid"],
        },
        "scientific_posterior_uid": prior_uid,
        "scientific_posterior_schema": "round23_gaussian_calibration_posterior_v1",
        "agent_policy": {
            "policy_id": "decision_risk_utility_with_post_observation_preferences_v1",
            "policy_hash": _json_hash(
                {
                    "learning_rate": protocol["agent_preference_learning_rate"],
                    "query_without_replacement": protocol["query_without_replacement"],
                }
            ),
        },
        "tri_state": {
            "reliability_accept_threshold": protocol[
                "reliability_accept_threshold"
            ],
            "calibration_interval_level": protocol[
                "calibration_interval_level"
            ],
        },
        "development_data_identity": {
            "pool_id": pool_payload["pool_id"],
            "manifest_path": source_identity["pool_manifest_path"],
            "manifest_sha256": source_identity["pool_manifest_sha256"],
            "arrays_sha256": pool_payload["arrays"]["sha256"],
            "train_action_uids": sorted(train_action_uids),
            "validation_action_uids": candidate_action_uids,
        },
        "tool_registry": pool_payload["registered_tool_execution"],
        "source_mechanism": {
            "attempt_id": source_identity["gate"]["attempt_id"],
            "gate_path": source_identity["gate_path"],
            "gate_sha256": source_identity["gate_sha256"],
            "attempt_manifest_path": source_identity["attempt_manifest_path"],
            "attempt_manifest_sha256": source_identity[
                "attempt_manifest_sha256"
            ],
        },
        "baseline_metrics": baseline_metrics,
        "offline_replay_only": True,
        "prospective_sequential_dispatch": False,
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(_repo_path(STATIC_REFERENCE_PATH), static_reference)

    report_path = _repo_path(ADAPTATION_REPORT_PATH)
    write_tsv(report_path, list(final_rows[0]), final_rows)
    invalid_rows = [
        row for row in rollback_rows if row["rollback_type"] == "invalid_observation_probe"
    ]
    rollback_payload = {
        "schema_version": "neuronsifter-round23-rollback-validation-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_offline_adaptation_rollback_validation"
        if invalid_rows and all(bool(row["rollback_correct"]) for row in rollback_rows)
        else "failed_offline_adaptation_rollback_validation",
        "rollback_validated": bool(
            invalid_rows and all(bool(row["rollback_correct"]) for row in rollback_rows)
        ),
        "invalid_assimilation_moves_biological_state": False,
        "adaptive_twin_updates_run": True,
        "invalid_probe_count": len(invalid_rows),
        "invalid_probe_pass_count": sum(bool(row["rollback_correct"]) for row in invalid_rows),
        "candidate_promotion_rollback_count": sum(
            row["rollback_type"] == "candidate_promotion_rollback"
            for row in rollback_rows
        ),
        "rollback_event_path": _repo_relative(rollback_rows_path),
        "rollback_event_sha256": _sha256_file(rollback_rows_path),
        "neural_twin_checkpoint_immutable": checkpoint_immutable,
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(_repo_path(ROLLBACK_PATH), rollback_payload)

    static_path = _repo_path(STATIC_REFERENCE_PATH)
    rollback_path = _repo_path(ROLLBACK_PATH)
    manifest_path = output_root_path / "manifest.json"
    manifest = {
        "schema_version": "neuronsifter-round23-offline-adaptation-replay-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_uid": output_root_path.name,
        "status": "completed",
        "execution_host": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "branch_count": len(final_rows),
        "primary_branch_count": sum(
            row["branch_role"] == "primary_2x2" for row in final_rows
        ),
        "control_branch_count": sum(row["branch_role"] == "control" for row in final_rows),
        "rounds_per_branch": int(protocol["sequential_round_count"]),
        "source_identity": {
            key: value for key, value in source_identity.items() if key != "gate"
        },
        "artifacts": {
            _repo_relative(round_path): _sha256_file(round_path),
            _repo_relative(rollback_rows_path): _sha256_file(rollback_rows_path),
            STATIC_REFERENCE_PATH: _sha256_file(static_path),
            ROLLBACK_PATH: _sha256_file(rollback_path),
            ADAPTATION_REPORT_PATH: _sha256_file(report_path),
        },
        "fixture_rows_used": False,
        "neural_twin_checkpoint_updated": False,
        "prospective_tool_dispatch_count": 0,
        "source_outcomes_prospectively_generated": True,
        "offline_replay_only": True,
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(manifest_path, manifest)

    primary_rows = [row for row in final_rows if row["branch_role"] == "primary_2x2"]
    eligible_reference_rows = [
        row
        for row in primary_rows
        if row["branch_id"] != "adaptive_twin_adaptive_agent"
    ]
    best_reference = min(
        eligible_reference_rows,
        key=lambda row: (
            -float(row["new_domain_utility"]),
            float(row["aurc"]),
            float(row["calibration_drift"]),
            -float(row["old_domain_retention"]),
        ),
    )
    joint = next(
        row for row in primary_rows if row["branch_id"] == "adaptive_twin_adaptive_agent"
    )
    one_sided = [
        row
        for row in primary_rows
        if row["branch_id"]
        in {"frozen_twin_adaptive_agent", "adaptive_twin_frozen_agent"}
    ]
    joint_better_than_one_sided = all(
        float(joint["new_domain_utility"]) > float(row["new_domain_utility"])
        and float(joint["aurc"]) < float(row["aurc"])
        for row in one_sided
    )
    matrix = {
        "schema_version": "neuronsifter-round23-adaptation-matrix-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_offline_replay_of_prospectively_generated_outcomes",
        "source_outcomes_prospectively_generated": True,
        "prospective_sequential_dispatch": False,
        "prospective_round_count": 0,
        "offline_replay_round_count": int(protocol["sequential_round_count"]),
        "branch_count": len(final_rows),
        "primary_branch_count": len(primary_rows),
        "control_branch_count": len(final_rows) - len(primary_rows),
        "matrix": final_rows,
        "status_known": True,
        "rollback_validated_if_updates_run": rollback_payload[
            "rollback_validated"
        ],
        "joint_adaptation_better_than_one_sided": joint_better_than_one_sided,
        "joint_adaptation_direction_status": "supported_in_offline_replay"
        if joint_better_than_one_sided
        else "not_supported_by_offline_replay",
        "recommended_static_or_one_sided_reference": best_reference["branch_id"],
        "joint_adaptation_claim_allowed": False,
        "empirical_coevolution_evidence": False,
        "neural_twin_checkpoint_updated": False,
        "scientific_posterior_calibration_layer_only": True,
        "unqueried_outcomes_used_for_policy": False,
        "full_pool_outcomes_used_for_posthoc_development_metrics": True,
        "fixture_rows_used": False,
        "report_path": ADAPTATION_REPORT_PATH,
        "report_sha256": _sha256_file(report_path),
        "round_log_path": _repo_relative(round_path),
        "round_log_sha256": _sha256_file(round_path),
        "static_reference_path": STATIC_REFERENCE_PATH,
        "static_reference_sha256": _sha256_file(static_path),
        "rollback_validation_path": ROLLBACK_PATH,
        "rollback_validation_sha256": _sha256_file(rollback_path),
        "run_manifest_path": _repo_relative(manifest_path),
        "run_manifest_sha256": _sha256_file(manifest_path),
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(_repo_path(ADAPTATION_MATRIX_PATH), matrix)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-identity", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pool-manifest", default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--pool-identity", default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--mechanism-gate", default=DEFAULT_MECHANISM_GATE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    result = run(
        checkpoint=arguments.checkpoint,
        checkpoint_identity=arguments.checkpoint_identity,
        device=arguments.device,
        pool_manifest=arguments.pool_manifest,
        pool_identity=arguments.pool_identity,
        mechanism_gate=arguments.mechanism_gate,
        output_root=arguments.output_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "branch_count": result["branch_count"],
                "rollback_validated": result[
                    "rollback_validated_if_updates_run"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
