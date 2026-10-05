"""Run the Round-23 matched shared-belief mechanism replay.

This is an offline development mechanism test.  It reuses the trained C2-S
world-action checkpoint and development-only train/validation trajectories; it
does not access the protected final holdout or execute an external tool.
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
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_restricted_c2_vnext import (  # noqa: E402
    RestrictedC2WorldActionModel,
    _json_sha256,
    _model_prediction,
    _profile_config,
    _structured_action_features,
    load_restricted_c2_development_data,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


VARIANTS = (
    "S_shared_posterior",
    "D_decoupled_posteriors",
    "P_shared_point",
    "U_uncertainty_query",
)
DEFAULT_CHECKPOINT_PATH = "Output/runs/round23/c2s/large_seed3407/last.pt"
DEFAULT_ATTEMPT_ID = "attempt_01_aggregate_ephys_response"


@dataclass
class GaussianBelief:
    mean: np.ndarray
    covariance: np.ndarray

    def update(self, design: np.ndarray, observation: float, noise_variance: float) -> "GaussianBelief":
        if not math.isfinite(float(observation)):
            raise ValueError("Gaussian belief observations must be finite")
        if not math.isfinite(float(noise_variance)) or noise_variance <= 0.0:
            raise ValueError("Gaussian belief noise variance must be finite and positive")
        design = np.asarray(design, dtype=np.float64).reshape(1, -1)
        covariance_design = self.covariance @ design.T
        innovation_variance = float((design @ covariance_design)[0, 0] + noise_variance)
        gain = covariance_design[:, 0] / max(innovation_variance, 1e-12)
        residual = float(observation - (design @ self.mean.reshape(-1, 1))[0, 0])
        mean = self.mean + gain * residual
        covariance = self.covariance - np.outer(gain, design[0] @ self.covariance)
        covariance = 0.5 * (covariance + covariance.T)
        return GaussianBelief(mean, covariance)


def _repo_path(value: str) -> Path:
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


def _predictive_variance(
    belief: GaussianBelief,
    candidate_designs: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    designs = np.asarray(candidate_designs, dtype=np.float64)
    covariance = np.asarray(belief.covariance, dtype=np.float64)
    if designs.ndim != 2 or covariance.shape != (designs.shape[1], designs.shape[1]):
        raise ValueError("Candidate designs and belief covariance have incompatible shapes")
    if not math.isfinite(float(noise_variance)) or noise_variance <= 0.0:
        raise ValueError("Predictive noise variance must be finite and positive")
    latent_variance = np.einsum("ij,jk,ik->i", designs, covariance, designs)
    if np.any(~np.isfinite(latent_variance)) or np.any(latent_variance < -1e-10):
        raise ValueError("Candidate predictive variances are invalid")
    return np.maximum(latent_variance, 0.0) + float(noise_variance)


def _load_candidate_data(
    config: dict[str, Any],
    *,
    candidate_pool: str | None,
    candidate_pool_identity: str | None,
) -> dict[str, Any]:
    if not candidate_pool:
        data = load_restricted_c2_development_data(config)
        data.update(
            {
                "mechanism_candidate_split_roles": ["train", "validation"],
                "execution_mode": "offline_development_replay",
                "prospective_tool_execution_available": False,
                "tool_execution_status": "offline_replay_only",
                "mechanism_output_origin": "checkpoint_backed_c2s_development_replay",
                "candidate_pool_identity": {
                    "pool_id": "modeldb_romani_current_clamp_restricted_train_validation",
                    "manifest_path": config["neuronsifter_vnext"]["training"]
                    ["c2s_development"]["action_catalog_path"],
                    "prospective_dispatch": False,
                },
            }
        )
        return data

    if not candidate_pool_identity:
        raise ValueError("Prospective candidate pools require a canonical identity path")
    manifest_path = _repo_path(candidate_pool)
    identity_path = _repo_path(candidate_pool_identity)
    if not manifest_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("Round-23 prospective mechanism pool manifest is unavailable")
    if _sha256_file(manifest_path) != _sha256_file(identity_path):
        raise ValueError("Staged Round-23 candidate pool manifest changed identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = config["round23_mechanism_pool"]
    if manifest.get("schema_version") != "neuronsifter-round23-prospective-mechanism-pool-result-v1":
        raise ValueError("Round-23 prospective mechanism pool result schema drift")
    if manifest.get("pool_id") != contract["pool_id"]:
        raise ValueError("Round-23 prospective mechanism pool identity drift")
    if manifest.get("pool_contract_sha256") != _json_sha256(contract):
        raise ValueError("Round-23 prospective mechanism pool contract changed after dispatch")
    if manifest.get("status") != "prospective_registered_tool_execution_complete":
        raise ValueError("Round-23 prospective mechanism tool execution is incomplete")
    if manifest.get("evidence_role") != "development_only" or manifest.get(
        "publication_claim_eligible"
    ) is not False:
        raise ValueError("Round-23 prospective mechanism pool crossed its evidence boundary")
    boundary = manifest.get("protected_boundary", {})
    if (
        boundary.get("canonical_c2s_trajectory_store_accessed") is not False
        or boundary.get("canonical_final_holdout_result_accessed") is not False
        or boundary.get("protected_action_uids_referenced") is not False
        or int(boundary.get("protected_test_access_count", -1)) != 0
    ):
        raise ValueError("Round-23 prospective mechanism pool crossed the protected boundary")
    arrays_path = manifest_path.parent / str(manifest["arrays"]["path"])
    if not arrays_path.is_file() or _sha256_file(arrays_path) != manifest["arrays"]["sha256"]:
        raise ValueError("Round-23 prospective mechanism pool array hash drift")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        trajectories = np.asarray(arrays["trajectories"], dtype=np.float32)
        uncertainties = np.asarray(arrays["uncertainties"], dtype=np.float32)
        action_features = np.asarray(arrays["action_features"], dtype=np.float32)
    records = [dict(row) for row in manifest["records"]]
    if trajectories.shape[0] != len(records) or action_features.shape[0] != len(records):
        raise ValueError("Round-23 prospective mechanism pool array/record count drift")
    if uncertainties.shape != (len(records), trajectories.shape[-1]):
        raise ValueError("Round-23 prospective mechanism uncertainty shape drift")
    if not all(
        np.isfinite(values).all()
        for values in (trajectories, uncertainties, action_features)
    ):
        raise ValueError("Round-23 prospective mechanism pool contains NaN or Inf")
    candidate_role = str(manifest["candidate_split_role"])
    if candidate_role != str(contract["candidate_split_role"]):
        raise ValueError("Round-23 prospective mechanism candidate split drift")
    feature_layout = config["screening_pipeline"]["screening_benchmark"][
        "model_architecture"
    ]["rollout_operator"]["regimen_feature_layout"]
    train_action_uids = {
        str(row["action_uid"]) for row in records if row["split_role"] == "train"
    }
    recomputed_features = _structured_action_features(
        [row["action"] for row in records],
        feature_layout,
        train_action_uids=train_action_uids,
    )
    if not np.array_equal(action_features, recomputed_features):
        raise ValueError("Round-23 prospective mechanism action compiler drift")
    return {
        "records": records,
        "trajectories": trajectories,
        "uncertainties": uncertainties,
        "action_features": action_features,
        "accessed_simulation_indices": [
            int(row["simulation_index"]) for row in records
        ],
        "protected_simulation_indices": [],
        "mechanism_candidate_split_roles": [candidate_role],
        "execution_mode": "prospective_development_tool_replay",
        "prospective_tool_execution_available": True,
        "tool_execution_status": "fresh_registered_tool_outcomes",
        "mechanism_output_origin": "checkpoint_backed_c2s_prospective_tool_replay",
        "candidate_pool_identity": {
            "pool_id": manifest["pool_id"],
            "manifest_path": _repo_relative(identity_path),
            "manifest_sha256": _sha256_file(identity_path),
            "prospective_dispatch": True,
            "registered_tool_execution": manifest["registered_tool_execution"],
        },
    }


def _rank_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    left_order = np.argsort(-np.asarray(left), kind="stable")
    right_order = np.argsort(-np.asarray(right), kind="stable")
    left_position = {int(value): index for index, value in enumerate(left_order.tolist())}
    right_position = {int(value): index for index, value in enumerate(right_order.tolist())}
    inversions = 0
    pairs = 0
    for index in range(len(left_order)):
        for other in range(index + 1, len(left_order)):
            pairs += 1
            left_ordered = left_position[index] < left_position[other]
            right_ordered = right_position[index] < right_position[other]
            inversions += int(left_ordered != right_ordered)
    return float(inversions / max(pairs, 1))


def _fit_prior(designs: np.ndarray, values: np.ndarray, noise_variance: float) -> GaussianBelief:
    precision = np.eye(designs.shape[1], dtype=np.float64) * 1e-2
    precision += designs.T @ designs / max(noise_variance, 1e-8)
    covariance = np.linalg.inv(precision)
    mean = covariance @ designs.T @ values / max(noise_variance, 1e-8)
    return GaussianBelief(mean, 0.5 * (covariance + covariance.T))


def _action_design_matrix(
    records: list[dict[str, Any]],
    action_uids: list[str],
    train_action_uids: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_by_uid = {str(row["action"]["action_uid"]): row["action"] for row in records}
    amplitudes = np.asarray(
        [float(action_by_uid[action_uid]["dose_value"]) for action_uid in action_uids],
        dtype=np.float64,
    )
    durations = np.asarray(
        [float(action_by_uid[action_uid]["duration_ms"]) for action_uid in action_uids],
        dtype=np.float64,
    )
    train_positions = [
        index for index, action_uid in enumerate(action_uids) if action_uid in train_action_uids
    ]
    if len(train_positions) < 2:
        raise ValueError("Round-23 mechanism belief requires at least two train actions")
    amplitude_scale = max(float(np.max(np.abs(amplitudes[train_positions]))), 1e-6)
    duration_scale = max(float(np.max(durations[train_positions])), 1e-6)
    signed_charge = amplitudes * durations
    charge_scale = max(float(np.max(np.abs(signed_charge[train_positions]))), 1e-6)
    designs = np.stack(
        [
            np.ones(len(action_uids), dtype=np.float64),
            amplitudes / amplitude_scale,
            durations / duration_scale,
            signed_charge / charge_scale,
        ],
        axis=1,
    )
    if not np.isfinite(designs).all():
        raise ValueError("Round-23 mechanism action designs contain NaN or Inf")
    return designs, amplitudes, durations


def _sample_risk(
    belief: GaussianBelief,
    candidate_designs: np.ndarray,
    true_values: np.ndarray | None,
    rng: np.random.Generator,
    sample_count: int = 128,
) -> float:
    covariance = belief.covariance + np.eye(belief.covariance.shape[0]) * 1e-10
    samples = rng.multivariate_normal(belief.mean, covariance, size=sample_count)
    sampled_values = samples @ candidate_designs.T
    selected = int(np.argmax(candidate_designs @ belief.mean))
    if true_values is not None:
        return float(np.max(true_values) - true_values[selected])
    expected_best = sampled_values.max(axis=1)
    selected_values = sampled_values[:, selected]
    return float(np.mean(expected_best - selected_values))


def _decision_risk(belief: GaussianBelief, candidate_designs: np.ndarray, true_values: np.ndarray) -> float:
    selected = int(np.argmax(candidate_designs @ belief.mean))
    scale = max(float(np.ptp(true_values)), 1e-6)
    return float((np.max(true_values) - true_values[selected]) / scale)


def _expected_decision_risk_after_query(
    belief: GaussianBelief,
    query_design: np.ndarray,
    candidate_designs: np.ndarray,
    noise_variance: float,
    rng: np.random.Generator,
) -> float:
    query_mean = float(query_design @ belief.mean)
    query_variance = float(query_design @ belief.covariance @ query_design + noise_variance)
    observations = rng.normal(query_mean, math.sqrt(max(query_variance, 1e-10)), size=32)
    return float(
        np.mean(
            [
                _sample_risk(
                    belief.update(query_design, float(observation), noise_variance),
                    candidate_designs,
                    None,
                    rng,
                    sample_count=64,
                )
                for observation in observations
            ]
        )
    )


def _load_world_action_outputs(
    config: dict[str, Any],
    data: dict[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_identity_path: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("C2-S Large seed 3407 checkpoint is required for the mechanism replay")
    identity_result_path = checkpoint_identity_path.parent / "run_result.json"
    if not identity_result_path.is_file():
        raise FileNotFoundError("C2-S checkpoint identity requires its run_result.json")
    identity = json.loads(identity_result_path.read_text(encoding="utf-8"))
    checkpoint_hash = _sha256_file(checkpoint_path)
    declared_hash = str(identity.get("checkpoint_sha256") or identity.get("checkpoint_hash") or "")
    if not declared_hash or declared_hash != checkpoint_hash:
        raise ValueError("C2-S mechanism checkpoint hash does not match its run identity")
    if identity.get("status") != "passed" or identity.get("checkpoint_roundtrip") is not True:
        raise ValueError("C2-S mechanism checkpoint is not a passed round-trip-validated run")
    if identity.get("evidence_role") != "development_only" or identity.get("publication_claim_eligible") is not False:
        raise ValueError("C2-S mechanism checkpoint crossed its development evidence boundary")
    if int(identity.get("protected_test_access", -1)) != 0:
        raise ValueError("C2-S mechanism checkpoint accessed a protected holdout")
    if str(identity.get("checkpoint_path")) != _repo_relative(checkpoint_identity_path):
        raise ValueError("C2-S checkpoint identity path does not match the selected run")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_scale") != "Large":
        raise ValueError("Round-23 mechanism replay requires the selected Large checkpoint")
    if (
        payload.get("evidence_role") != "development_only"
        or payload.get("publication_claim_eligible") is not False
    ):
        raise ValueError("C2-S checkpoint payload is missing its development-only boundary")
    effective_config = _profile_config(config, "Large")
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the Round-23 mechanism replay but is unavailable")
    model = RestrictedC2WorldActionModel(effective_config).to(device_object)
    model.load_state_dict(payload["model"])
    model.eval()
    normalization_scale = np.asarray(payload["normalization_scale"], dtype=np.float32)
    normalized = data["trajectories"] / normalization_scale.reshape(1, 1, 1, 1, -1)
    with torch.no_grad():
        prediction_output = _model_prediction(
            model,
            torch.as_tensor(normalized, dtype=torch.float32, device=device_object),
            torch.as_tensor(data["action_features"], dtype=torch.float32, device=device_object),
        )
        prediction = prediction_output["trajectory_mean"].cpu().numpy()
        prediction_std = prediction_output["trajectory_std"].cpu().numpy()
    return (
        normalized.astype(np.float64),
        prediction.astype(np.float64),
        prediction_std.astype(np.float64),
        {
            "checkpoint_path": _repo_relative(checkpoint_identity_path),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_identity_verified": True,
            "checkpoint_run_uid": str(identity["run_uid"]),
            "checkpoint_seed": int(identity["seed"]),
            "device": str(device_object),
        },
    )


def _response_values(
    records: list[dict[str, Any]],
    trajectories: np.ndarray,
    predictions: np.ndarray,
    prediction_std: np.ndarray,
    species_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    ephys_ids = {"calcium_concentration", "membrane_voltage", "spike_rate", "circuit_instability"}
    ephys_indices = [index for index, species_id in enumerate(species_ids) if species_id in ephys_ids]
    if not ephys_indices:
        raise ValueError("C2-S mechanism replay requires electrophysiology species")
    control_by_pair = {
        str(row["action"]["pair_uid"]): index
        for index, row in enumerate(records)
        if str(row["action"]["action_role"]) == "matched_control"
    }
    intervention_indices = [
        index
        for index, row in enumerate(records)
        if str(row["action"]["action_role"]) == "intervention"
    ]
    intervention_indices.sort(key=lambda index: float(records[index]["action"]["dose_value"]))
    true_values: list[float] = []
    predicted_values: list[float] = []
    predicted_standard_deviations: list[float] = []
    action_uids: list[str] = []
    pair_uids: list[str] = []
    for intervention_index in intervention_indices:
        pair_uid = str(records[intervention_index]["action"]["pair_uid"])
        control_index = control_by_pair[pair_uid]
        true_response = np.abs(
            trajectories[intervention_index, 1:, ..., ephys_indices]
            - trajectories[control_index, 1:, ..., ephys_indices]
        ).mean()
        predicted_response = np.abs(
            predictions[intervention_index, ..., ephys_indices]
            - predictions[control_index, ..., ephys_indices]
        ).mean()
        predicted_response_std = np.sqrt(
            np.square(prediction_std[intervention_index, ..., ephys_indices])
            + np.square(prediction_std[control_index, ..., ephys_indices])
        ).mean()
        true_values.append(float(true_response))
        predicted_values.append(float(predicted_response))
        predicted_standard_deviations.append(float(predicted_response_std))
        action_uids.append(str(records[intervention_index]["action"]["action_uid"]))
        pair_uids.append(pair_uid)
    return (
        np.asarray(true_values),
        np.asarray(predicted_values),
        np.asarray(predicted_standard_deviations),
        action_uids,
        pair_uids,
    )


def _endpoint_response_catalog(
    records: list[dict[str, Any]],
    trajectories: np.ndarray,
    predictions: np.ndarray,
    prediction_std: np.ndarray,
    species_ids: list[str],
    candidate_split_roles: set[str],
) -> list[dict[str, Any]]:
    ephys_ids = {
        "calcium_concentration",
        "membrane_voltage",
        "spike_rate",
        "circuit_instability",
    }
    control_by_pair = {
        str(row["action"]["pair_uid"]): index
        for index, row in enumerate(records)
        if str(row["action"]["action_role"]) == "matched_control"
    }
    intervention_indices = [
        index
        for index, row in enumerate(records)
        if str(row["action"]["action_role"]) == "intervention"
        and str(row["split_role"]) in candidate_split_roles
    ]
    intervention_indices.sort(
        key=lambda index: float(records[index]["action"]["dose_value"])
    )
    catalog: list[dict[str, Any]] = []
    for species_index, species_id in enumerate(species_ids):
        if species_id not in ephys_ids:
            continue
        for region_index in range(trajectories.shape[2]):
            true_values: list[float] = []
            predicted_values: list[float] = []
            predicted_std_values: list[float] = []
            for intervention_index in intervention_indices:
                pair_uid = str(records[intervention_index]["action"]["pair_uid"])
                control_index = control_by_pair[pair_uid]
                true_values.append(
                    float(
                        np.abs(
                            trajectories[
                                intervention_index,
                                1:,
                                region_index,
                                :,
                                species_index,
                            ]
                            - trajectories[
                                control_index,
                                1:,
                                region_index,
                                :,
                                species_index,
                            ]
                        ).mean()
                    )
                )
                predicted_values.append(
                    float(
                        np.abs(
                            predictions[
                                intervention_index, :, region_index, :, species_index
                            ]
                            - predictions[
                                control_index, :, region_index, :, species_index
                            ]
                        ).mean()
                    )
                )
                predicted_std_values.append(
                    float(
                        np.sqrt(
                            np.square(
                                prediction_std[
                                    intervention_index,
                                    :,
                                    region_index,
                                    :,
                                    species_index,
                                ]
                            )
                            + np.square(
                                prediction_std[
                                    control_index,
                                    :,
                                    region_index,
                                    :,
                                    species_index,
                                ]
                            )
                        ).mean()
                    )
                )
            true_array = np.asarray(true_values, dtype=np.float64)
            predicted_array = np.asarray(predicted_values, dtype=np.float64)
            std_array = np.asarray(predicted_std_values, dtype=np.float64)
            selected = int(np.argmax(predicted_array))
            selected_lower = predicted_array[selected] - 1.6448536269514722 * std_array[selected]
            other_upper = max(
                predicted_array[index] + 1.6448536269514722 * std_array[index]
                for index in range(len(predicted_array))
                if index != selected
            )
            scale = max(float(np.ptp(true_array)), 1e-6)
            catalog.append(
                {
                    "endpoint_uid": f"{species_id}|region_{region_index}",
                    "species_id": species_id,
                    "region_index": region_index,
                    "true_response_values": true_values,
                    "checkpoint_predicted_response_values": predicted_values,
                    "checkpoint_predictive_standard_deviations": predicted_std_values,
                    "rank_disagreement": _rank_disagreement(
                        predicted_array, true_array
                    ),
                    "initial_decision_risk": float(
                        (np.max(true_array) - true_array[selected]) / scale
                    ),
                    "decision_uncertainty_overlap": bool(other_upper >= selected_lower),
                }
            )
    return catalog


def run(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT_PATH,
    checkpoint_identity: str = DEFAULT_CHECKPOINT_PATH,
    device: str = "cpu",
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    candidate_pool: str | None = None,
    candidate_pool_identity: str | None = None,
) -> dict[str, Any]:
    if not attempt_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in attempt_id
    ):
        raise ValueError("Round-23 mechanism attempt IDs must use lowercase snake case")
    attempt_root = REPO_ROOT / "Output/audits/round23_mechanism_attempts" / attempt_id
    if attempt_root.exists():
        raise FileExistsError(f"Round-23 mechanism attempt already exists: {attempt_id}")
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    data = _load_candidate_data(
        config,
        candidate_pool=candidate_pool,
        candidate_pool_identity=candidate_pool_identity,
    )
    accessed_indices = {int(value) for value in data["accessed_simulation_indices"]}
    protected_indices = {int(value) for value in data["protected_simulation_indices"]}
    if accessed_indices & protected_indices:
        raise ValueError("C2-S mechanism replay crossed the protected split boundary")
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_identity_path = _repo_path(checkpoint_identity)
    normalized_truth, predicted, prediction_std, checkpoint_metadata = _load_world_action_outputs(
        config,
        data,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        device=device,
    )
    species_ids = [
        str(row["species_id"])
        for row in config["dynamics_ready_registry"]["species"]
    ]
    (
        all_true_values,
        all_predicted_values,
        all_predicted_standard_deviations,
        all_action_uids,
        all_pair_uids,
    ) = _response_values(
        data["records"],
        normalized_truth,
        predicted,
        prediction_std,
        species_ids,
    )
    candidate_split_roles = {
        str(value) for value in data["mechanism_candidate_split_roles"]
    }
    endpoint_catalog = _endpoint_response_catalog(
        data["records"],
        normalized_truth,
        predicted,
        prediction_std,
        species_ids,
        candidate_split_roles,
    )
    split_role_by_action = {
        str(row["action"]["action_uid"]): str(row["split_role"])
        for row in data["records"]
        if str(row["action"]["action_role"]) == "intervention"
    }
    train_action_uids = {
        action_uid
        for action_uid, split_role in split_role_by_action.items()
        if split_role == "train"
    }
    all_designs, all_amplitudes, all_durations = _action_design_matrix(
        data["records"], all_action_uids, train_action_uids
    )
    train_positions = [
        index
        for index, action_uid in enumerate(all_action_uids)
        if action_uid in train_action_uids
    ]
    candidate_positions = [
        index
        for index, action_uid in enumerate(all_action_uids)
        if split_role_by_action[action_uid] in candidate_split_roles
    ]
    if len(train_positions) < 2 or len(candidate_positions) < 2:
        raise ValueError(
            "C2-S mechanism replay requires at least two calibration and candidate actions"
        )
    candidate_designs = all_designs[candidate_positions]
    candidate_amplitudes = all_amplitudes[candidate_positions]
    candidate_durations = all_durations[candidate_positions]
    true_values = all_true_values[candidate_positions]
    predicted_values = all_predicted_values[candidate_positions]
    predicted_standard_deviations = all_predicted_standard_deviations[
        candidate_positions
    ]
    action_uids = [all_action_uids[index] for index in candidate_positions]
    pair_uids = [all_pair_uids[index] for index in candidate_positions]
    noise_scale = max(float(np.std(all_true_values[train_positions])) * 0.25, 1e-3)
    noise_variance = noise_scale**2
    prior = _fit_prior(
        all_designs[train_positions],
        all_predicted_values[train_positions],
        noise_variance,
    )
    protocol = config["neuronsifter_vnext"]["mechanism_test"]["protocol"]
    rows: list[dict[str, Any]] = []
    seed_count = int(protocol["seed_count"])
    for seed in range(3407, 3407 + seed_count):
        utility_rng = np.random.default_rng(seed)
        prior_risk = _sample_risk(prior, candidate_designs, None, utility_rng)
        query_ideal = np.asarray(
            [
                prior_risk
                - _expected_decision_risk_after_query(
                    prior, design, candidate_designs, noise_variance, utility_rng
                )
                for design in candidate_designs
            ]
        )
        observation_rng = np.random.default_rng(seed + 1_000_000)
        candidate_observations = true_values + observation_rng.normal(
            0.0, noise_scale, size=len(true_values)
        )
        for variant in VARIANTS:
            predictive = GaussianBelief(prior.mean.copy(), prior.covariance.copy())
            acquisition = GaussianBelief(prior.mean.copy(), prior.covariance.copy())
            point_mean = prior.mean.copy()
            pre_ranking = candidate_designs @ predictive.mean
            pre_selected = int(np.argmax(pre_ranking))
            if variant == "S_shared_posterior":
                query_scores = query_ideal.copy()
            elif variant == "U_uncertainty_query":
                query_scores = np.asarray(
                    [
                        float(
                            np.trace(predictive.covariance)
                            - np.trace(
                                predictive.update(
                                    design,
                                    float(design @ predictive.mean),
                                    noise_variance,
                                ).covariance
                            )
                        )
                        for design in candidate_designs
                    ]
                )
            elif variant == "D_decoupled_posteriors":
                query_scores = np.asarray(
                    [
                        float(
                            np.trace(acquisition.covariance)
                            - np.trace(
                                acquisition.update(
                                    design,
                                    float(design @ acquisition.mean),
                                    noise_variance,
                                ).covariance
                            )
                        )
                        for design in candidate_designs
                    ]
                )
            else:
                query_scores = np.zeros(len(candidate_designs), dtype=np.float64)
            query_index = int(np.argmax(query_scores))
            observation = float(candidate_observations[query_index])
            if variant in {"S_shared_posterior", "U_uncertainty_query"}:
                predictive = predictive.update(candidate_designs[query_index], observation, noise_variance)
                acquisition = predictive
            elif variant == "D_decoupled_posteriors":
                acquisition = acquisition.update(candidate_designs[query_index], observation, noise_variance)
            else:
                point_mean = point_mean + (
                    candidate_designs[query_index]
                    * (observation - candidate_designs[query_index] @ point_mean)
                ) / (
                    float(candidate_designs[query_index] @ candidate_designs[query_index])
                    + 1e-6
                )
            decision_belief = predictive
            if variant == "P_shared_point":
                decision_belief = GaussianBelief(point_mean, np.zeros_like(prior.covariance))
            post_ranking = candidate_designs @ decision_belief.mean
            post_selected = int(np.argmax(post_ranking))
            posterior_discrepancy = float(
                np.linalg.norm(predictive.mean - acquisition.mean)
                / max(np.linalg.norm(predictive.mean), 1e-6)
            )
            if variant == "P_shared_point":
                posterior_discrepancy = float(
                    np.linalg.norm(point_mean - prior.mean)
                    / max(np.linalg.norm(prior.mean), 1e-6)
                )
            true_rank = true_values
            risk = _decision_risk(decision_belief, candidate_designs, true_values)
            baseline_risk = _decision_risk(
                GaussianBelief(prior.mean, np.zeros_like(prior.covariance)),
                candidate_designs,
                true_values,
            )
            predictive_variance = _predictive_variance(
                decision_belief, candidate_designs, noise_variance
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "posterior_discrepancy_proxy": posterior_discrepancy,
                    "query_rank_disagreement": _rank_disagreement(query_scores, query_ideal),
                    "intervention_rank_disagreement": _rank_disagreement(post_ranking, true_rank),
                    "observation_induced_intervention_switch_rate": float(pre_selected != post_selected),
                    "utility_calibration": float(
                        np.mean(
                            np.abs(true_values - post_ranking)
                            <= 1.6448536269514722 * np.sqrt(predictive_variance)
                        )
                    ),
                    "utility_calibration_interval_level": 0.90,
                    "top_query_regret": float(np.max(query_ideal) - query_ideal[query_index]),
                    "decision_risk_at_matched_cost": risk,
                    "cost_to_matched_decision_risk": float(
                        1.0 if risk <= max(baseline_risk * 0.5, 1e-6) else 2.0
                    ),
                    "posterior_tv_available": False,
                    "proxy_name": "normalized_gaussian_belief_mean_covariance_discrepancy",
                    "evidence_role": "development_only",
                    "publication_claim_eligible": False,
                    "execution_mode": data["execution_mode"],
                    "tool_execution_status": data["tool_execution_status"],
                    "checkpoint_path": checkpoint_metadata["checkpoint_path"],
                    "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
                    "checkpoint_identity_verified": True,
                    "candidate_action_uids": ",".join(action_uids),
                    "candidate_pair_uids": ",".join(pair_uids),
                    "protected_test_access": 0,
                }
            )
    fieldnames = list(rows[0])
    write_tsv(REPO_ROOT / "Output/paper_reports/round23_shared_vs_decoupled_dev.tsv", fieldnames, rows)
    coherence_fields = [
        "seed",
        "variant",
        "posterior_discrepancy_proxy",
        "query_rank_disagreement",
        "intervention_rank_disagreement",
        "observation_induced_intervention_switch_rate",
        "utility_calibration",
        "top_query_regret",
        "decision_risk_at_matched_cost",
        "cost_to_matched_decision_risk",
        "posterior_tv_available",
        "proxy_name",
        "evidence_role",
        "publication_claim_eligible",
    ]
    write_tsv(
        REPO_ROOT / "Output/paper_reports/round23_belief_coherence_diagnostics.tsv",
        coherence_fields,
        [{field: row[field] for field in coherence_fields} for row in rows],
    )
    means: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        means[variant] = {
            field: float(np.mean([float(row[field]) for row in subset]))
            for field in (
                "posterior_discrepancy_proxy",
                "query_rank_disagreement",
                "intervention_rank_disagreement",
                "observation_induced_intervention_switch_rate",
                "utility_calibration",
                "top_query_regret",
                "decision_risk_at_matched_cost",
                "cost_to_matched_decision_risk",
            )
        }
    shared = means["S_shared_posterior"]
    controls = [means[variant] for variant in VARIANTS if variant != "S_shared_posterior"]
    seed_win_fields = tuple(str(value) for value in protocol["seed_win_metric_order"])
    tie_tolerance = float(protocol["seed_win_tie_tolerance"])

    def _lexicographic_win(shared_row: dict[str, Any], control_row: dict[str, Any]) -> bool:
        for field in seed_win_fields:
            difference = float(shared_row[field]) - float(control_row[field])
            if difference < -tie_tolerance:
                return True
            if difference > tie_tolerance:
                return False
        return False

    seed_wins = []
    for seed in range(3407, 3407 + seed_count):
        shared_row = next(
            row for row in rows if row["variant"] == "S_shared_posterior" and row["seed"] == seed
        )
        control_rows = [
            row
            for row in rows
            if row["variant"] != "S_shared_posterior" and row["seed"] == seed
        ]
        seed_wins.append(all(_lexicographic_win(shared_row, row) for row in control_rows))
    seed_win_fraction = float(np.mean(seed_wins))
    best_control_risk = min(
        control["decision_risk_at_matched_cost"] for control in controls
    )
    decoupled_rank = means["D_decoupled_posteriors"]["intervention_rank_disagreement"]
    pass_gate = (
        len(rows) >= int(protocol["minimum_valid_episode_count"])
        and seed_win_fraction >= float(protocol["required_control_win_fraction"])
        and shared["decision_risk_at_matched_cost"]
        <= float(protocol["maximum_decision_risk_ratio_vs_best_control"])
        * max(best_control_risk, 1e-6)
        and shared["intervention_rank_disagreement"]
        <= float(protocol["maximum_rank_disagreement_ratio_vs_decoupled"])
        * max(decoupled_rank, 1e-6)
    )
    gate = {
        "schema_version": "neuronsifter-round23-coupling-mechanism-gate-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "status": "passed_development_mechanism_replay" if pass_gate else "failed_development_mechanism_replay",
        "development_gate_passed": bool(pass_gate),
        "execution_mode": data["execution_mode"],
        "prospective_tool_execution_available": data[
            "prospective_tool_execution_available"
        ],
        "matched_capacity": True,
        "matched_development_budget": True,
        "matched_candidate_pools": True,
        "matched_world_action_checkpoint": True,
        "mechanism_output_origin": data["mechanism_output_origin"],
        "fixture_rows_used": False,
        "candidate_pool_identity": data["candidate_pool_identity"],
        "world_action_checkpoint_path": checkpoint_metadata["checkpoint_path"],
        "world_action_checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "world_action_checkpoint_identity_verified": checkpoint_metadata[
            "checkpoint_identity_verified"
        ],
        "world_action_checkpoint_run_uid": checkpoint_metadata["checkpoint_run_uid"],
        "execution_device": checkpoint_metadata["device"],
        "accessed_simulation_indices": sorted(accessed_indices),
        "protected_simulation_indices": sorted(protected_indices),
        "protected_test_access_count": 0,
        "candidate_action_uids": action_uids,
        "candidate_pair_uids": pair_uids,
        "candidate_dose_values": candidate_amplitudes.tolist(),
        "candidate_duration_ms_values": candidate_durations.tolist(),
        "candidate_true_response_values": true_values.tolist(),
        "candidate_checkpoint_predicted_response_values": predicted_values.tolist(),
        "candidate_checkpoint_predictive_standard_deviations": predicted_standard_deviations.tolist(),
        "initial_true_vs_predicted_rank_disagreement": _rank_disagreement(
            predicted_values, true_values
        ),
        "initial_decision_risk": _decision_risk(
            GaussianBelief(prior.mean, np.zeros_like(prior.covariance)),
            candidate_designs,
            true_values,
        ),
        "decision_task_degenerate": bool(
            all(
                float(row["decision_risk_at_matched_cost"]) == 0.0
                and float(row["intervention_rank_disagreement"]) == 0.0
                for row in rows
            )
        ),
        "endpoint_decision_diagnostic": {
            "endpoint_count": len(endpoint_catalog),
            "endpoint_with_nonzero_initial_decision_risk_count": sum(
                float(row["initial_decision_risk"]) > 0.0
                for row in endpoint_catalog
            ),
            "endpoint_with_uncertainty_overlap_count": sum(
                bool(row["decision_uncertainty_overlap"])
                for row in endpoint_catalog
            ),
            "rows": endpoint_catalog,
        },
        "posterior_tv_available": False,
        "proxy_declared_separately": True,
        "seed_count": seed_count,
        "episode_count": len(rows),
        "seed_win_fraction": seed_win_fraction,
        "seed_win_metric_order": list(seed_win_fields),
        "seed_win_tie_tolerance": tie_tolerance,
        "variant_means": means,
        "protocol": dict(protocol),
        "primary_question": "Does S outperform D/P/U on downstream decision consistency under matched capacity and matched development budget?",
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "report_path": "Output/paper_reports/round23_shared_vs_decoupled_dev.tsv",
        "coherence_report_path": "Output/paper_reports/round23_belief_coherence_diagnostics.tsv",
        "attempt_gate_path": (
            f"Output/audits/round23_mechanism_attempts/{attempt_id}/gate.json"
        ),
        "report_sha256": _sha256_file(
            REPO_ROOT / "Output/paper_reports/round23_shared_vs_decoupled_dev.tsv"
        ),
        "coherence_report_sha256": _sha256_file(
            REPO_ROOT / "Output/paper_reports/round23_belief_coherence_diagnostics.tsv"
        ),
        "stop_rule_if_failed": "diagnose_posterior_mismatch_assimilation_and_decision_utility_before_scaling",
    }
    write_json(REPO_ROOT / "Output/audits/round23_coupling_mechanism_gate.json", gate)
    attempt_root.mkdir(parents=True, exist_ok=False)
    attempt_report_path = attempt_root / "shared_vs_decoupled_dev.tsv"
    attempt_coherence_path = attempt_root / "belief_coherence_diagnostics.tsv"
    attempt_gate_path = attempt_root / "gate.json"
    write_tsv(attempt_report_path, fieldnames, rows)
    write_tsv(
        attempt_coherence_path,
        coherence_fields,
        [{field: row[field] for field in coherence_fields} for row in rows],
    )
    write_json(attempt_gate_path, gate)
    write_json(
        attempt_root / "manifest.json",
        {
            "schema_version": "neuronsifter-round23-mechanism-attempt-manifest-v1",
            "attempt_id": attempt_id,
            "status": gate["status"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "artifacts": {
                "shared_vs_decoupled_dev.tsv": _sha256_file(attempt_report_path),
                "belief_coherence_diagnostics.tsv": _sha256_file(
                    attempt_coherence_path
                ),
                "gate.json": _sha256_file(attempt_gate_path),
            },
            "immutable_append_only_attempt": True,
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
        },
    )
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-identity", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument("--candidate-pool", default="")
    parser.add_argument("--candidate-pool-identity", default="")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                checkpoint=arguments.checkpoint,
                checkpoint_identity=arguments.checkpoint_identity,
                device=arguments.device,
                attempt_id=arguments.attempt_id,
                candidate_pool=arguments.candidate_pool or None,
                candidate_pool_identity=arguments.candidate_pool_identity or None,
            ),
            sort_keys=True,
        )
    )
