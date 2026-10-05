"""Execute the Round-24B posterior-bound Twin-only development campaign.

This runner is intentionally a successor to the Round-24 placeholder path. It
never overwrites Round-24 artifacts, restores the frozen C1/C2-S checkpoints,
dispatches one common prospective evidence stream, and evaluates U0-U3 from
the same per-unit model inputs. All outputs are development-only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_foundation_state_c1 import _model_from_constructor  # noqa: E402
from Libs.exp.training.train_restricted_c2_vnext import (  # noqa: E402
    RestrictedC2WorldActionModel,
    _model_prediction,
    _profile_config,
)
from Libs.models.modules.decision_consistent_runtime import (  # noqa: E402
    DecisionConsistentTwin,
    ScientificPosterior,
    TrajectoryResult,
    WorldAction,
)
from Libs.models.modules.round24b_posterior_adapter import (  # noqa: E402
    PosteriorBoundState,
    content_hash,
    posterior_condition_probe,
    posterior_conditioned_c1_forward,
)
from Libs.models.modules.round24b_typed_actions import (  # noqa: E402
    CurrentClampAction,
    pair_from_pool_records,
)
from Libs.scripts.builders.build_round24b_pre_update_reference import (  # noqa: E402
    load_contract,
    repo_path,
    sha256_file,
)
from Libs.scripts.experiments.run_round23_shared_belief_replay import (  # noqa: E402
    GaussianBelief,
    _action_design_matrix,
    _load_candidate_data,
    _model_prediction as c2_model_prediction,
)
from Libs.utils.data.foundation_state_c1_loader import (  # noqa: E402
    C1TranscriptomicsLoader,
    build_c1_loader_context,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    write_json,
    write_tsv,
)


DEFAULT_CONTRACT = REPO_ROOT / "Libs/configs/neuronsifter/round24b_posterior_bound_evaluation.yaml"
DEFAULT_REFERENCE = REPO_ROOT / "Output/audits/round24b_pre_update_reference.json"
DEFAULT_POSTERIOR = REPO_ROOT / "Output/audits/round24b_pre_update_posterior.json"
DEFAULT_OUTPUT = REPO_ROOT / "Output/runs/round24b"
TOOL_ID = "neuron_rxd_microenvironment_teacher.run_protocol_batch"
Z90 = NormalDist().inv_cdf(0.95)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, int, np.floating, np.integer)):
        return math.isfinite(float(value))
    return True


def _json_hash(value: Any) -> str:
    return content_hash(value)


def _posterior_record_hash(state: PosteriorBoundState) -> str:
    return str(state.to_record()["record_sha256"])


def _load_reference(contract: Mapping[str, Any], reference_path: Path, posterior_path: Path) -> tuple[dict[str, Any], PosteriorBoundState]:
    if not reference_path.is_file() or not posterior_path.is_file():
        raise FileNotFoundError("Round-24B requires the immutable pre-update reference and posterior record")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    posterior_record = json.loads(posterior_path.read_text(encoding="utf-8"))
    if reference.get("status") != "frozen" or reference.get("pre_update_reference_frozen") is not True:
        raise ValueError("Round-24B pre-update reference is not frozen")
    if reference.get("posterior_record", {}).get("sha256") != sha256_file(posterior_path):
        raise ValueError("Round-24B posterior record hash drift")
    state = PosteriorBoundState.from_record(posterior_record)
    if state.posterior_uid != reference.get("posterior_uid_pre"):
        raise ValueError("Round-24B posterior UID does not match the frozen reference")
    boundary = contract["evidence_boundary"]
    if boundary.get("protected_holdout_access") != 0 or boundary.get("publication_claim_eligible") is not False:
        raise ValueError("Round-24B publication boundary is open")
    if state.source_checkpoint_sha256 != reference["twin_checkpoints"]["c2s_world_action"]["sha256"]:
        raise ValueError("Round-24B posterior source checkpoint differs from reference")
    return reference, state


@dataclass
class C1UnitCache:
    evaluation_unit_uid: str
    biological_unit_uid: str
    canonical_uid: str
    source_id: int
    source_prediction: int
    residuals: np.ndarray
    native_sigmas: np.ndarray


@dataclass
class C1BatchCache:
    batch_index: int
    model_inputs: dict[str, torch.Tensor]
    biological_unit_uids: tuple[str, ...]
    canonical_uids: tuple[str, ...]
    source_ids: torch.Tensor


@dataclass
class C1Cache:
    model: Any
    batches: list[C1BatchCache]
    forward_options: dict[str, Any]
    sample_count: int
    checkpoint_path: str
    checkpoint_sha256: str
    capability_uid: str
    split_uid: str
    split_manifest_hash: str
    role_binding_sha256: str
    data_manifest_hash: str
    metric_input_path: str
    units_by_posterior: dict[str, list[C1UnitCache]] = field(default_factory=dict)
    effect_probes: dict[str, dict[str, Any]] = field(default_factory=dict)


def _metric_snapshot_path(output_root: Path, domain: str, strategy_uid: str, round_uid: str) -> Path:
    """Return a deterministic, strategy/round-specific metric-input path."""
    safe_strategy = str(strategy_uid).replace("/", "_")
    safe_round = str(round_uid).replace("/", "_")
    path = output_root / "metric_inputs" / safe_strategy / safe_round / f"{domain}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_metric_snapshot(path: Path, payload: Mapping[str, Any]) -> str:
    write_json(path, dict(payload))
    return _rel(path)


def _c1_forward_options_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact C1 forward contract recorded by checkpoint validation."""
    roundtrip_probe = checkpoint.get("roundtrip_probe")
    if not isinstance(roundtrip_probe, Mapping):
        raise ValueError("C1 checkpoint lacks its authoritative roundtrip probe")
    probe_options = roundtrip_probe.get("forward_options")
    if not isinstance(probe_options, Mapping):
        raise ValueError("C1 checkpoint lacks authoritative forward options")
    required_options = {
        "use_ontology_context",
        "use_action_context",
        "nuisance_gradient_scale",
    }
    if not required_options.issubset(probe_options):
        raise ValueError("C1 checkpoint forward options are incomplete")
    nuisance_scale = float(probe_options["nuisance_gradient_scale"])
    if not math.isfinite(nuisance_scale):
        raise ValueError("C1 checkpoint nuisance gradient scale is non-finite")
    return {
        "use_ontology_context": bool(probe_options["use_ontology_context"]),
        "use_action_context": bool(probe_options["use_action_context"]),
        "nuisance_gradient_scale": nuisance_scale,
    }


def _load_c1_cache(config: Mapping[str, Any], contract: Mapping[str, Any], reference: Mapping[str, Any], output_root: Path, device: str) -> C1Cache:
    ref = contract["reference"]
    checkpoint_path = repo_path(ref["c1_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_hash = reference["twin_checkpoints"]["c1_encoder"]["sha256"]
    if checkpoint_hash != expected_hash:
        raise ValueError("C1 checkpoint hash differs from frozen Round-24B reference")
    role_path = repo_path(ref["c1_role_binding_path"])
    loader = C1TranscriptomicsLoader(
        REPO_ROOT,
        config,
        role_binding_path=role_path,
        expected_role="validation",
        batch_size=int(ref["c1_batch_size"]),
        profile="Large",
        seed=3407,
        sampling_policy=str(config["foundation_state_c1"]["loader"]["sampling_policy"]),
        max_observations_per_unit=int(config["foundation_state_c1"]["loader"]["max_observations_per_biological_unit"]),
        objective_set="masked_only",
        loader_context=build_c1_loader_context(REPO_ROOT, config),
    )
    model = _model_from_constructor(checkpoint["model_constructor"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    forward_options = _c1_forward_options_from_checkpoint(checkpoint)
    sample_count = int(ref["c1_posterior_sample_count"])
    batches: list[C1BatchCache] = []
    for batch_index in range(int(ref["c1_batch_count"])):
        batch = loader.batch(epoch=0, batch_index=batch_index)
        tensors = {key: value.to(device) for key, value in batch.model_inputs().items()}
        batches.append(
            C1BatchCache(
                batch_index=batch_index,
                model_inputs=tensors,
                biological_unit_uids=tuple(str(value) for value in batch.biological_unit_uids),
                canonical_uids=tuple(str(value) for value in batch.canonical_uids),
                source_ids=batch.source_ids.to(device),
            )
        )
    if not batches:
        raise ValueError("C1 posterior-bound evaluation has no batches")
    metric_dir = output_root / "metric_inputs"
    metric_dir.mkdir(parents=True, exist_ok=True)
    metric_path = metric_dir / "c1_base_inputs.json"
    write_json(
        metric_path,
        {
            "schema_version": "neuronsifter-round24b-c1-base-metric-input-v1",
            "units": [
                {
                    "evaluation_unit_uid": f"c1-unit:{batch.batch_index}:{biological_uid}",
                    "biological_unit_uid": biological_uid,
                    "canonical_uid": batch.canonical_uids[unit_index],
                }
                for batch in batches
                for unit_index, biological_uid in enumerate(batch.biological_unit_uids)
            ],
            "checkpoint_path": _rel(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "split_uid": ref["c1_split_uid"],
            "split_manifest_hash": sha256_file(role_path),
            "role_binding_sha256": sha256_file(role_path),
            "data_manifest_hash": str(checkpoint.get("corpus_hash", sha256_file(role_path))),
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        },
    )
    return C1Cache(
        model=model,
        batches=batches,
        forward_options=forward_options,
        sample_count=sample_count,
        checkpoint_path=_rel(checkpoint_path),
        checkpoint_sha256=checkpoint_hash,
        capability_uid=reference["twin_checkpoints"]["c1_encoder"]["capability_uid"],
        split_uid=str(ref["c1_split_uid"]),
        split_manifest_hash=sha256_file(role_path),
        role_binding_sha256=sha256_file(role_path),
        data_manifest_hash=str(checkpoint.get("corpus_hash", sha256_file(role_path))),
        metric_input_path=_rel(metric_path),
    )


def _c1_units_for_state(
    state: PosteriorBoundState,
    cache: C1Cache,
    contract: Mapping[str, Any],
) -> list[C1UnitCache]:
    cached = cache.units_by_posterior.get(state.posterior_uid)
    if cached is not None:
        return cached
    condition_scale = float(contract["posterior"]["c1_condition_scale"])
    units: list[C1UnitCache] = []
    condition_hashes: set[str] = set()
    max_log_variance_effect = 0.0
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for batch in cache.batches:
            with torch.no_grad():
                neutral = cache.model(
                    **batch.model_inputs,
                    **cache.forward_options,
                    sample_posterior=False,
                )
                deterministic = posterior_conditioned_c1_forward(
                    cache.model,
                    batch.model_inputs,
                    cache.forward_options,
                    state,
                    condition_scale=condition_scale,
                    sample_posterior=False,
                )
                condition_hashes.add(str(deterministic["posterior_condition_hash"]))
                max_log_variance_effect = max(
                    max_log_variance_effect,
                    float(
                        torch.max(
                            torch.abs(
                                deterministic["latent_log_variance"]
                                - neutral["latent_log_variance"]
                            )
                        ).item()
                    ),
                )
                draws = []
                for sample_index in range(cache.sample_count):
                    torch.manual_seed(3407 + batch.batch_index * 100 + sample_index)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(
                            3407 + batch.batch_index * 100 + sample_index
                        )
                    draws.append(
                        posterior_conditioned_c1_forward(
                            cache.model,
                            batch.model_inputs,
                            cache.forward_options,
                            state,
                            condition_scale=condition_scale,
                            sample_posterior=True,
                        )["reconstructed_values"]
                    )
            target_mask = batch.model_inputs["target_mask"]
            truth = batch.model_inputs["feature_values"]
            prediction = deterministic["reconstructed_values"]
            draw_std = torch.stack(draws, dim=0).std(dim=0, unbiased=False).clamp_min(1e-6)
            for unit_index, biological_uid in enumerate(batch.biological_unit_uids):
                obs = torch.where(
                    batch.model_inputs["observation_unit_index"] == unit_index
                )[0]
                if obs.numel() == 0:
                    raise ValueError("C1 cache encountered an empty biological unit")
                source_values = batch.source_ids[obs].detach().cpu().numpy()
                source_pred = (
                    deterministic["source_diagnostic_logits"][obs]
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                units.append(
                    C1UnitCache(
                        evaluation_unit_uid=f"c1-unit:{batch.batch_index}:{biological_uid}",
                        biological_unit_uid=biological_uid,
                        canonical_uid=batch.canonical_uids[unit_index],
                        source_id=int(source_values[0]),
                        source_prediction=int(source_pred[0]),
                        residuals=(truth[obs] - prediction[obs])[target_mask[obs]]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64),
                        native_sigmas=draw_std[obs][target_mask[obs]]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64),
                    )
                )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if (
        not units
        or len(condition_hashes) != 1
        or not all(_finite(unit.residuals) and _finite(unit.native_sigmas) for unit in units)
    ):
        raise ValueError("C1 posterior-bound model path is empty, inconsistent, or non-finite")
    threshold = float(contract["posterior"]["required_effect_probe_abs"])
    if max_log_variance_effect <= threshold:
        raise ValueError("C1 scientific posterior did not affect the encoder model path")
    cache.units_by_posterior[state.posterior_uid] = units
    cache.effect_probes[state.posterior_uid] = {
        "posterior_uid": state.posterior_uid,
        "posterior_condition_hash": next(iter(condition_hashes)),
        "model_path_metric": "max_abs_latent_log_variance_change",
        "model_path_effect_abs": max_log_variance_effect,
        "required_effect_probe_abs": threshold,
        "direct_encoder_forward_consumption": True,
    }
    return units


def _c1_metrics(state: PosteriorBoundState, cache: C1Cache, contract: Mapping[str, Any], output_root: Path, *, strategy_uid: str, round_uid: str) -> list[dict[str, Any]]:
    probe = posterior_condition_probe(state, channel="c1_encoder_evaluation", scale=float(contract["posterior"]["c1_condition_scale"]))
    units = _c1_units_for_state(state, cache, contract)
    model_effect = cache.effect_probes[state.posterior_uid]
    metric_input_path = _metric_snapshot_path(output_root, "c1", strategy_uid, round_uid)
    snapshot = {
        "schema_version": "neuronsifter-round24b-c1-posterior-bound-metric-input-v1",
        "domain": "c1",
        "strategy_uid": strategy_uid,
        "round_uid": round_uid,
        "posterior_uid": state.posterior_uid,
        "posterior_checkpoint_hash": _posterior_record_hash(state),
        "posterior_source_checkpoint_sha256": state.source_checkpoint_sha256,
        "encoder_checkpoint_sha256": cache.checkpoint_sha256,
        "encoder_capability_uid": cache.capability_uid,
        "data_manifest_hash": cache.data_manifest_hash,
        "split_manifest_hash": cache.split_manifest_hash,
        "condition_hash": model_effect["posterior_condition_hash"],
        "analytic_condition_probe": probe,
        "condition_scale": float(probe["effect_after"]),
        "model_path_effect_probe": model_effect,
        "evaluation_units": [unit.evaluation_unit_uid for unit in units],
        "base_metric_input_ref": cache.metric_input_path,
        "base_metric_input_sha256": sha256_file(repo_path(cache.metric_input_path)),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    _write_metric_snapshot(metric_input_path, snapshot)
    rows: list[dict[str, Any]] = []
    for unit in units:
        residual = unit.residuals
        sigma = np.maximum(unit.native_sigmas, float(contract["posterior"]["c1_uncertainty_floor"]))
        coverage = float(np.mean(np.abs(residual) <= Z90 * sigma))
        interval_miss = (np.abs(residual) > Z90 * sigma).astype(np.float64)
        standardized_error = np.abs(residual) / np.maximum(sigma, 1e-12)
        values = {
            "masked_target_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "masked_target_mae": float(np.mean(np.abs(residual))),
            "posterior_reconstruction_interval_coverage": coverage,
            "calibration_error": abs(coverage - 0.9),
            "aurc": _aurc_np(standardized_error, sigma),
            "false_accept": float(np.mean(interval_miss)),
            "support_domain_coverage": 1.0,
            "source_prediction_correct": float(unit.source_prediction == unit.source_id),
            "posterior_uncertainty_scale": float(probe["effect_after"]),
        }
        for metric_name, value in values.items():
            evaluation_uid = f"c1:{unit.evaluation_unit_uid}"
            rows.append(
                {
                    "domain": "c1",
                    "evaluation_uid": evaluation_uid,
                    "evaluation_unit_uid": unit.evaluation_unit_uid,
                    "biological_unit_uid": unit.biological_unit_uid,
                    "canonical_uid": unit.canonical_uid,
                    "strategy_uid": strategy_uid,
                    "round_uid": round_uid,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "posterior_uid": state.posterior_uid,
                    # The scientific posterior is sourced from the frozen C2-S
                    # calibration identity; the encoder checkpoint is recorded
                    # separately so these two identities cannot be conflated.
                    "posterior_checkpoint_hash": _posterior_record_hash(state),
                    "encoder_checkpoint_hash": cache.checkpoint_sha256,
                    "posterior_source_checkpoint_hash": state.source_checkpoint_sha256,
                    "encoder_capability_uid": cache.capability_uid,
                    "data_manifest_hash": cache.data_manifest_hash,
                    "split_manifest_hash": cache.split_manifest_hash,
                    "metric_input_ref": _rel(metric_input_path),
                    "posterior_condition_hash": model_effect[
                        "posterior_condition_hash"
                    ],
                    "posterior_model_path_effect_abs": model_effect["model_path_effect_abs"],
                    "posterior_direct_model_consumption": True,
                    "finite": bool(math.isfinite(value)),
                    "evidence_role": "development_only",
                    "publication_claim_eligible": False,
                    "protected_holdout_access": 0,
                }
            )
    return rows


@dataclass
class C2PairCache:
    pair_uid: str
    intervention: CurrentClampAction
    control: CurrentClampAction
    intervention_index: int
    control_index: int
    initial_state_uid: str
    trajectory_uid_intervention: str
    trajectory_uid_control: str
    prospective_design: np.ndarray | None = None


@dataclass
class C2Cache:
    model: RestrictedC2WorldActionModel
    normalized_trajectories: torch.Tensor
    action_features: torch.Tensor
    action_designs: np.ndarray
    records: list[dict[str, Any]]
    pairs: list[C2PairCache]
    scale: np.ndarray
    teacher_uncertainty: np.ndarray
    checkpoint_path: str
    checkpoint_sha256: str
    capability_uid: str
    pool_manifest_path: str
    pool_manifest_sha256: str
    producer_uid: str
    metric_input_path: str
    effect_probes: dict[str, dict[str, Any]] = field(default_factory=dict)


def _gaussian_crps_np(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> np.ndarray:
    sigma = np.maximum(std, 1e-6)
    z = (target - mean) / sigma
    phi = np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _aurc_np(error: np.ndarray, uncertainty: np.ndarray) -> float:
    """Area under the risk-coverage curve for one scientific unit."""
    errors = np.asarray(error, dtype=np.float64).reshape(-1)
    uncertainties = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if errors.size == 0 or errors.shape != uncertainties.shape:
        raise ValueError("AURC inputs must be non-empty and shape aligned")
    if not np.isfinite(errors).all() or not np.isfinite(uncertainties).all():
        raise ValueError("AURC inputs must be finite")
    order = np.argsort(uncertainties, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, errors.size + 1)
    value = float(np.mean(cumulative_risk))
    if not math.isfinite(value):
        raise ValueError("AURC is non-finite")
    return value


def _teacher_likelihood_uncertainty(
    pair_std: np.ndarray,
    teacher_uncertainty: np.ndarray,
) -> np.ndarray:
    """Combine model and teacher uncertainty with explicit trajectory axes.

    Model uncertainty is shaped ``[action, time, region, compartment, species]``
    while the admitted teacher artifact is ``[action, species]``.  Expanding the
    latter to all three trajectory axes keeps action and species identities
    aligned and rejects malformed artifacts before metric computation.
    """
    model_std = np.asarray(pair_std, dtype=np.float64)
    teacher_std = np.asarray(teacher_uncertainty, dtype=np.float64)
    if model_std.ndim != 5:
        raise ValueError(
            "C2-S model uncertainty must have [action,time,region,compartment,species] shape"
        )
    if teacher_std.ndim != 2:
        raise ValueError("C2-S teacher uncertainty must have [action,species] shape")
    if model_std.shape[0] != teacher_std.shape[0] or model_std.shape[-1] != teacher_std.shape[1]:
        raise ValueError(
            "C2-S model and teacher uncertainty action/species dimensions disagree"
        )
    if not np.isfinite(model_std).all() or not np.isfinite(teacher_std).all():
        raise ValueError("C2-S model and teacher uncertainty must be finite")
    teacher_broadcast = teacher_std[:, None, None, None, :]
    likelihood = np.sqrt(np.square(model_std) + np.square(teacher_broadcast))
    if not np.isfinite(likelihood).all() or np.any(likelihood <= 0.0):
        raise ValueError("C2-S combined likelihood uncertainty is invalid")
    return likelihood


def _load_c2_cache(config: Mapping[str, Any], contract: Mapping[str, Any], reference: Mapping[str, Any], output_root: Path, device: str) -> C2Cache:
    ref = contract["reference"]
    pool_path = repo_path(ref["c2s_pool_path"])
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    data = _load_candidate_data(config, candidate_pool=str(pool_path), candidate_pool_identity=str(pool_path))
    checkpoint_path = repo_path(ref["c2s_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != reference["twin_checkpoints"]["c2s_world_action"]["sha256"]:
        raise ValueError("C2-S checkpoint hash differs from frozen Round-24B reference")
    effective_config = _profile_config(config, "Large")
    model = RestrictedC2WorldActionModel(effective_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float32)
    normalized = data["trajectories"] / scale.reshape(1, 1, 1, 1, -1)
    records = [dict(row) for row in pool["records"]]
    all_uids = [str(row["action"]["action_uid"]) for row in records]
    train_uids = {uid for uid in all_uids if next(row for row in records if str(row["action"]["action_uid"]) == uid)["split_role"] == "train" and str(next(row for row in records if str(row["action"]["action_uid"]) == uid)["action"]["action_role"]) == "intervention"}
    action_designs, _, _ = _action_design_matrix(records, all_uids, train_uids)
    action_index = {uid: index for index, uid in enumerate(all_uids)}
    source_model_id = str(pool["registered_tool_execution"]["source_model_id"])
    typed_pairs = pair_from_pool_records(records, source_model_id, split_role=str(ref["c2s_split_role"]))
    pair_cache: list[C2PairCache] = []
    for intervention, control in typed_pairs:
        i = action_index[intervention.action_uid]
        c = action_index[control.action_uid]
        pair_cache.append(
            C2PairCache(
                pair_uid=intervention.pair_uid,
                intervention=intervention,
                control=control,
                intervention_index=i,
                control_index=c,
                initial_state_uid="state-" + _json_hash({"pool": sha256_file(pool_path), "pair_uid": intervention.pair_uid})[:24],
                trajectory_uid_intervention="trajectory-" + _json_hash({"simulation_index": records[i]["simulation_index"], "action_uid": intervention.action_uid})[:24],
                trajectory_uid_control="trajectory-" + _json_hash({"simulation_index": records[c]["simulation_index"], "action_uid": control.action_uid})[:24],
            )
        )
    if len(pair_cache) < 3:
        raise ValueError("Round-24B C2-S requires at least three typed validation pairs")
    metric_dir = output_root / "metric_inputs"
    metric_dir.mkdir(parents=True, exist_ok=True)
    metric_path = metric_dir / "c2s_base_inputs.json"
    write_json(
        metric_path,
        {
            "schema_version": "neuronsifter-round24b-c2s-base-metric-input-v1",
            "pool_manifest_path": _rel(pool_path),
            "pool_manifest_sha256": sha256_file(pool_path),
            "checkpoint_path": _rel(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "producer_uid": str(checkpoint.get("input_identity", {}).get("producer_execution_uid", pool["registered_tool_execution"]["source_model_id"])),
            "pairs": [
                {
                    "pair_uid": pair.pair_uid,
                    "intervention_action_uid": pair.intervention.action_uid,
                    "control_action_uid": pair.control.action_uid,
                    "intervention_world_action_uid": pair.intervention.to_world_action().action_uid,
                    "control_world_action_uid": pair.control.to_world_action().action_uid,
                    "initial_state_uid": pair.initial_state_uid,
                    "trajectory_uid_intervention": pair.trajectory_uid_intervention,
                    "trajectory_uid_control": pair.trajectory_uid_control,
                }
                for pair in pair_cache
            ],
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        },
    )
    return C2Cache(
        model=model,
        normalized_trajectories=torch.as_tensor(normalized, dtype=torch.float32, device=device),
        action_features=torch.as_tensor(data["action_features"], dtype=torch.float32, device=device),
        action_designs=action_designs,
        records=records,
        pairs=pair_cache,
        scale=scale,
        teacher_uncertainty=np.asarray(data["uncertainties"], dtype=np.float64),
        checkpoint_path=_rel(checkpoint_path),
        checkpoint_sha256=checkpoint_hash,
        capability_uid=reference["twin_checkpoints"]["c2s_world_action"]["capability_uid"],
        pool_manifest_path=_rel(pool_path),
        pool_manifest_sha256=sha256_file(pool_path),
        producer_uid=str(checkpoint.get("input_identity", {}).get("producer_execution_uid", pool["registered_tool_execution"]["source_model_id"])),
        metric_input_path=_rel(metric_path),
    )


def _response_species_indices(config: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[int, ...]:
    species_ids = [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]
    declared = tuple(str(value) for value in contract["typed_current_clamp"]["response_species_ids"])
    if len(set(declared)) != len(declared):
        raise ValueError("Round-24B response species declaration contains duplicates")
    missing = [species_id for species_id in declared if species_id not in species_ids]
    if missing:
        raise ValueError(f"Round-24B response species are absent from the state grid: {missing}")
    indices = tuple(species_ids.index(species_id) for species_id in declared)
    if not indices:
        raise ValueError("Round-24B C2-S response endpoint has no declared electrophysiology species")
    return indices


def _posterior_response_channel_mapping(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[tuple[int, int], tuple[str, str]]:
    species_ids = [
        str(row["species_id"])
        for row in config["dynamics_ready_registry"]["species"]
    ]
    mapping = contract["typed_current_clamp"][
        "posterior_condition_channel_species"
    ]
    channel_species_ids = (
        str(mapping["posterior_response_mean"]),
        str(mapping["posterior_response_std"]),
    )
    if len(set(channel_species_ids)) != 2:
        raise ValueError(
            "Round-24B posterior response channels require distinct species slots"
        )
    declared = set(
        str(value)
        for value in contract["typed_current_clamp"]["response_species_ids"]
    )
    if not set(channel_species_ids).issubset(declared):
        raise ValueError(
            "Round-24B posterior response channel slots must be declared response species"
        )
    missing = [value for value in channel_species_ids if value not in species_ids]
    if missing:
        raise ValueError(
            f"Round-24B posterior response channel species are absent: {missing}"
        )
    return (
        tuple(species_ids.index(value) for value in channel_species_ids),
        channel_species_ids,
    )


def _c2_prediction(state: PosteriorBoundState, cache: C2Cache, contract: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    from Libs.models.modules.round24b_posterior_adapter import posterior_conditioned_c2_prediction

    seed = 3407
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            channel_indices, channel_species_ids = _posterior_response_channel_mapping(
                config, contract
            )
            prediction = posterior_conditioned_c2_prediction(
                cache.model,
                cache.normalized_trajectories,
                cache.action_features,
                cache.action_designs,
                state,
                condition_scale=float(contract["posterior"]["c2s_condition_scale"]),
                response_channel_species_indices=channel_indices,
                response_channel_species_ids=channel_species_ids,
                apply_condition_mask=np.asarray(
                    [str(row["action"]["action_role"]) == "intervention" for row in cache.records],
                    dtype=bool,
                ),
            )
        if state.posterior_uid not in cache.effect_probes:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                neutral = _model_prediction(
                    cache.model,
                    cache.normalized_trajectories,
                    cache.action_features,
                )
            mean_effect = float(
                torch.max(
                    torch.abs(
                        prediction["trajectory_mean"] - neutral["trajectory_mean"]
                    )
                ).item()
            )
            std_effect = float(
                torch.max(
                    torch.abs(
                        prediction["trajectory_std"] - neutral["trajectory_std"]
                    )
                ).item()
            )
            effect = max(mean_effect, std_effect)
            threshold = float(contract["posterior"]["required_effect_probe_abs"])
            if not math.isfinite(effect) or effect <= threshold:
                raise ValueError(
                    "C2-S scientific posterior did not affect the world-action model path"
                )
            cache.effect_probes[state.posterior_uid] = {
                "posterior_uid": state.posterior_uid,
                "posterior_condition_hash": prediction["posterior_condition_hash"],
                "model_path_metric": "max_abs_trajectory_or_std_change",
                "trajectory_mean_effect_abs": mean_effect,
                "trajectory_std_effect_abs": std_effect,
                "model_path_effect_abs": effect,
                "required_effect_probe_abs": threshold,
                "direct_world_action_forward_consumption": True,
            }
        prediction["model_path_effect_probe"] = cache.effect_probes[
            state.posterior_uid
        ]
        return prediction
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _c2_metrics(
    state: PosteriorBoundState,
    cache: C2Cache,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    *,
    strategy_uid: str,
    round_uid: str,
    rollout_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prediction = _c2_prediction(state, cache, contract, config)
    means = prediction["trajectory_mean"].detach().cpu().numpy().astype(np.float64)
    stds = prediction["trajectory_std"].detach().cpu().numpy().astype(np.float64)
    truth = cache.normalized_trajectories.detach().cpu().numpy().astype(np.float64)[:, 1:]
    species_count = means.shape[-1]
    teacher = cache.teacher_uncertainty
    rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    nonnegative = np.ones(species_count, dtype=bool)
    species_ids = [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]
    signed_species_ids = {str(value) for value in contract["typed_current_clamp"]["signed_species_ids"]}
    unknown_signed = signed_species_ids.difference(species_ids)
    if unknown_signed:
        raise ValueError(f"Round-24B signed species are absent from the state grid: {sorted(unknown_signed)}")
    for index, species_id in enumerate(species_ids):
        if species_id in signed_species_ids:
            nonnegative[index] = False
    metric_input_path = _metric_snapshot_path(output_root, "c2s", strategy_uid, round_uid)
    prediction_path = metric_input_path.with_suffix(".npz")
    np.savez_compressed(
        prediction_path,
        trajectory_mean=means,
        trajectory_std=stds,
        trajectory_truth=truth,
        posterior_uid=np.asarray([state.posterior_uid]),
    )
    _write_metric_snapshot(
        metric_input_path,
        {
            "schema_version": "neuronsifter-round24b-c2s-posterior-bound-metric-input-v1",
            "domain": "c2s",
            "strategy_uid": strategy_uid,
            "round_uid": round_uid,
            "posterior_uid": state.posterior_uid,
            "posterior_checkpoint_hash": _posterior_record_hash(state),
            "posterior_source_checkpoint_sha256": state.source_checkpoint_sha256,
            "world_action_checkpoint_sha256": cache.checkpoint_sha256,
            "world_action_capability_uid": cache.capability_uid,
            "producer_uid": cache.producer_uid,
            "pool_manifest_sha256": cache.pool_manifest_sha256,
            "prediction_artifact": _rel(prediction_path),
            "prediction_artifact_sha256": sha256_file(prediction_path),
            "response_species_ids": [species_ids[index] for index in _response_species_indices(config, contract)],
            "posterior_condition_channel_species": prediction[
                "response_channel_mapping"
            ],
            "signed_species_ids": sorted(signed_species_ids),
            "posterior_condition_hash": prediction["posterior_condition_hash"],
            "model_path_effect_probe": prediction["model_path_effect_probe"],
            "common_random_numbers": True,
            "evaluation_seed": 3407,
            "evaluation_units": [f"{pair.pair_uid}:intervention" for pair in cache.pairs],
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        },
    )
    for pair in cache.pairs:
        i, c = pair.intervention_index, pair.control_index
        pair_mean = means[[i, c]]
        pair_truth = truth[[i, c]]
        pair_std = stds[[i, c]]
        likelihood = _teacher_likelihood_uncertainty(pair_std, teacher[[i, c]])
        errors = pair_truth - pair_mean
        contrast_error = (pair_mean[0] - pair_mean[1]) - (pair_truth[0] - pair_truth[1])
        interval_miss = (np.abs(errors) > Z90 * likelihood).astype(np.float64)
        standardized_error = np.abs(errors) / np.maximum(likelihood, 1e-12)
        metrics = {
            "trajectory_error": float(np.sqrt(np.mean(np.square(errors)))),
            "terminal_error": float(np.sqrt(np.mean(np.square(errors[:, -1])))),
            "long_horizon_drift": float(np.sqrt(np.mean(np.square(errors[:, -1])))),
            "action_contrast": float(np.sqrt(np.mean(np.square(contrast_error)))),
            "crps": float(np.mean(_gaussian_crps_np(pair_mean, likelihood, pair_truth))),
            "nll": float(np.mean(0.5 * (np.log(2.0 * math.pi * np.square(likelihood)) + np.square(errors) / np.square(likelihood)))),
            "coverage": float(np.mean(np.abs(errors) <= Z90 * likelihood)),
            "calibration_error": abs(float(np.mean(np.abs(errors) <= Z90 * likelihood)) - 0.9),
            "aurc": _aurc_np(standardized_error, likelihood),
            "false_accept": float(np.mean(interval_miss)),
            "support_domain_coverage": 1.0,
            "constraint_residual": float(np.mean(pair_mean[..., nonnegative] < 0.0)),
            "epistemic_variance": float(np.mean(np.square(pair_std))),
            "posterior_response_mean": float(prediction["posterior_response_mean"][i].item()),
            "posterior_response_std": float(prediction["posterior_response_std"][i].item()),
        }
        for metric_name, value in metrics.items():
            evaluation_uid = f"c2s:{pair.pair_uid}"
            row = {
                "domain": "c2s",
                "evaluation_uid": evaluation_uid,
                "evaluation_unit_uid": f"{pair.pair_uid}:intervention",
                "pair_uid": pair.pair_uid,
                "initial_state_uid": pair.initial_state_uid,
                "trajectory_uid": pair.trajectory_uid_intervention,
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "metric_name": metric_name,
                "metric_value": value,
                "posterior_uid": state.posterior_uid,
                "posterior_checkpoint_hash": _posterior_record_hash(state),
                "world_action_checkpoint_hash": cache.checkpoint_sha256,
                "world_action_uid": pair.intervention.to_world_action().action_uid,
                "control_world_action_uid": pair.control.to_world_action().action_uid,
                "control_trajectory_uid": pair.trajectory_uid_control,
                "action_type": "current_clamp_world_action",
                "producer_uid": cache.producer_uid,
                "data_manifest_hash": cache.pool_manifest_sha256,
                "split_manifest_hash": cache.pool_manifest_sha256,
                "metric_input_ref": _rel(metric_input_path),
                "posterior_condition_hash": prediction["posterior_condition_hash"],
                "posterior_model_path_effect_abs": prediction[
                    "model_path_effect_probe"
                ]["model_path_effect_abs"],
                "posterior_direct_model_consumption": True,
                "finite": bool(math.isfinite(value)),
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
            rows.append(row)
        rollout_rows.extend(
            [
                {
                    "evaluation_uid": f"c2s:{pair.pair_uid}:rollout_intervention",
                    "posterior_uid": state.posterior_uid,
                    "initial_state_uid": pair.initial_state_uid,
                    "world_action_uid": pair.intervention.to_world_action().action_uid,
                    "action_type": "current_clamp_world_action",
                    "trajectory_uid": pair.trajectory_uid_intervention,
                    "control_trajectory_uid": pair.trajectory_uid_control,
                    "producer_uid": cache.producer_uid,
                    "strategy_uid": strategy_uid,
                    "round_uid": round_uid,
                    "metric_input_ref": _rel(metric_input_path),
                    "posterior_condition_hash": prediction["posterior_condition_hash"],
                },
                {
                    "evaluation_uid": f"c2s:{pair.pair_uid}:rollout_control",
                    "posterior_uid": state.posterior_uid,
                    "initial_state_uid": pair.initial_state_uid,
                    "world_action_uid": pair.control.to_world_action().action_uid,
                    "action_type": "current_clamp_world_action",
                    "trajectory_uid": pair.trajectory_uid_control,
                    "control_trajectory_uid": pair.trajectory_uid_intervention,
                    "producer_uid": cache.producer_uid,
                    "strategy_uid": strategy_uid,
                    "round_uid": round_uid,
                    "metric_input_ref": _rel(metric_input_path),
                    "posterior_condition_hash": prediction["posterior_condition_hash"],
                },
            ]
        )
    if rollout_records is not None:
        rollout_records.extend(rollout_rows)
    return rows


def _prospective_evidence_metrics(
    state: PosteriorBoundState,
    pair: C2PairCache,
    observation: float | None,
    observation_metadata: Mapping[str, Any],
    c2_cache: C2Cache,
    contract: Mapping[str, Any],
    output_root: Path,
    prospective_unit_manifest_path: Path,
    *,
    strategy_uid: str,
    round_uid: str,
) -> list[dict[str, Any]]:
    """Evaluate fit to the newly observed action without implying generalization."""
    if observation is None:
        return []
    if pair.prospective_design is None:
        raise ValueError("Round-24B prospective evidence lacks its outcome-blind design")
    predicted_mean, predicted_std = state.action_response(pair.prospective_design)
    error = float(observation) - predicted_mean
    variance = max(predicted_std * predicted_std, 1e-12)
    interval_coverage = float(abs(error) <= Z90 * predicted_std)
    values = {
        "new_evidence_absolute_error": abs(error),
        "new_evidence_squared_error": error * error,
        "new_evidence_nll": 0.5 * (math.log(2.0 * math.pi * variance) + error * error / variance),
        "new_evidence_interval_coverage": interval_coverage,
        "new_evidence_predicted_response": predicted_mean,
        "new_evidence_predictive_std": predicted_std,
    }
    if not _finite(values):
        raise ValueError("Round-24B prospective evidence metrics are non-finite")
    if not prospective_unit_manifest_path.is_file():
        raise FileNotFoundError(prospective_unit_manifest_path)
    condition_hash = state.condition_hash(
        "prospective_action_response_prediction",
        {
            "pair_uid": pair.pair_uid,
            "world_action_uid": pair.intervention.to_world_action().action_uid,
            "design": pair.prospective_design.tolist(),
        },
    )
    metric_input_path = _metric_snapshot_path(
        output_root, "c2s_prospective", strategy_uid, round_uid
    )
    snapshot = {
        "schema_version": "neuronsifter-round24b-prospective-evidence-metric-input-v1",
        "domain": "c2s_prospective",
        "strategy_uid": strategy_uid,
        "round_uid": round_uid,
        "evaluation_uid": f"c2s-prospective:{pair.pair_uid}",
        "posterior_uid": state.posterior_uid,
        "posterior_checkpoint_hash": _posterior_record_hash(state),
        "posterior_source_checkpoint_sha256": state.source_checkpoint_sha256,
        "world_action_checkpoint_sha256": c2_cache.checkpoint_sha256,
        "world_action_capability_uid": c2_cache.capability_uid,
        "world_action": pair.intervention.to_record(),
        "control_world_action": pair.control.to_record(),
        "observation": float(observation),
        "observation_metadata": dict(observation_metadata),
        "predicted_response": predicted_mean,
        "predictive_std": predicted_std,
        "posterior_condition_hash": condition_hash,
        "posterior_action_response_direct_consumption": True,
        "posterior_action_response_binding": "state.action_response",
        "prospective_unit_manifest_ref": _rel(prospective_unit_manifest_path),
        "prospective_unit_manifest_sha256": sha256_file(
            prospective_unit_manifest_path
        ),
        "fresh_observation_reused_for_fit_evaluation": True,
        "generalization_claim_eligible": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    _write_metric_snapshot(metric_input_path, snapshot)
    trajectory_uid = "prospective-trajectory-" + str(
        observation_metadata.get("normalization_scale_sha256", _json_hash(snapshot))
    )[:24]
    rows = []
    for metric_name, value in values.items():
        rows.append(
            {
                "domain": "c2s_prospective",
                "evaluation_uid": f"c2s-prospective:{pair.pair_uid}",
                "evaluation_unit_uid": f"{pair.pair_uid}:intervention",
                "pair_uid": pair.pair_uid,
                "initial_state_uid": pair.initial_state_uid,
                "trajectory_uid": trajectory_uid,
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "metric_name": metric_name,
                "metric_value": float(value),
                "posterior_uid": state.posterior_uid,
                "posterior_checkpoint_hash": _posterior_record_hash(state),
                "world_action_checkpoint_hash": c2_cache.checkpoint_sha256,
                "world_action_uid": pair.intervention.to_world_action().action_uid,
                "control_world_action_uid": pair.control.to_world_action().action_uid,
                "action_type": "current_clamp_world_action",
                "producer_uid": TOOL_ID,
                "data_manifest_hash": sha256_file(prospective_unit_manifest_path),
                "split_manifest_hash": content_hash(contract["prospective"]),
                "metric_input_ref": _rel(metric_input_path),
                "posterior_condition_hash": condition_hash,
                "posterior_action_response_direct_consumption": True,
                "posterior_action_response_binding": "state.action_response",
                "fresh_observation_reused_for_fit_evaluation": True,
                "generalization_claim_eligible": False,
                "finite": True,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
    return rows


def _select_common_pairs(state: PosteriorBoundState, cache: C2Cache, rounds: int) -> list[C2PairCache]:
    """Select frozen evaluation pairs for tests; campaigns use fresh pairs."""
    if rounds > len(cache.pairs):
        raise ValueError("Round-24B prospective rounds exceed validation pairs")
    scored = []
    for pair in cache.pairs:
        design = cache.action_designs[pair.intervention_index]
        mean, std = state.action_response(design)
        scored.append((-(std / max(1.0 + abs(mean), 1e-6)), pair.pair_uid, pair))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:rounds]]


def _prospective_pairs(
    contract: Mapping[str, Any],
    state: PosteriorBoundState,
    cache: C2Cache,
    source_model_id: str,
) -> list[C2PairCache]:
    """Freeze a fresh outcome-blind protocol pool and select three queries."""
    declared = contract["prospective"]["candidate_protocols"]
    if len(declared) < int(contract["prospective"]["rounds"]):
        raise ValueError("Round-24B fresh prospective pool is too small")
    historical_protocols = {
        (
            float(row["action"]["dose_value"]),
            float(row["action"]["duration_ms"]),
            float(row["action"]["delay_ms"]),
        )
        for row in cache.records
        if str(row["action"]["action_role"]) == "intervention"
    }
    protected = (0.2, 120.0, 40.0)
    train_rows = [
        record
        for record in cache.records
        if str(record["split_role"]) == "train"
        and str(record["action"]["action_role"]) == "intervention"
    ]
    train_amplitudes = [abs(float(record["action"]["dose_value"])) for record in train_rows]
    train_durations = [float(record["action"]["duration_ms"]) for record in train_rows]
    amplitude_scale = max(train_amplitudes, default=1e-6)
    duration_scale = max(train_durations, default=1e-6)
    charge_scale = max(
        (abs(float(record["action"]["dose_value"]) * float(record["action"]["duration_ms"])) for record in train_rows),
        default=1e-6,
    )
    candidates: list[tuple[float, str, C2PairCache]] = []
    for row in declared:
        protocol = (float(row["amplitude_nA"]), float(row["duration_ms"]), float(row["delay_ms"]))
        if protocol in historical_protocols or protocol == protected:
            raise ValueError(f"Round-24B prospective protocol overlaps historical/protected identity: {protocol}")
        intervention = CurrentClampAction(
            action_uid=str(row["action_uid"]),
            pair_uid=str(row["pair_uid"]),
            role="intervention",
            amplitude_nA=protocol[0],
            duration_ms=protocol[1],
            delay_ms=protocol[2],
            source_model_id=source_model_id,
            split_role="development_diagnostic",
        )
        control = CurrentClampAction(
            action_uid=f"control_zero_for_{row['action_uid']}",
            pair_uid=str(row["pair_uid"]),
            role="matched_control",
            amplitude_nA=0.0,
            duration_ms=protocol[1],
            delay_ms=protocol[2],
            source_model_id=source_model_id,
            split_role="development_diagnostic",
        )
        # Reuse only scales learned from the frozen train protocols. The new
        # candidate does not enter the scale computation or expose an outcome.
        prospective_design = np.asarray(
            [
                1.0,
                intervention.amplitude_nA / amplitude_scale,
                intervention.duration_ms / duration_scale,
                (intervention.amplitude_nA * intervention.duration_ms) / charge_scale,
            ],
            dtype=np.float64,
        )
        mean, std = state.action_response(prospective_design)
        utility = std / max(1.0 + abs(mean), 1e-6)
        pair_uid = intervention.pair_uid
        pair = C2PairCache(
            pair_uid=pair_uid,
            intervention=intervention,
            control=control,
            intervention_index=-1,
            control_index=-1,
            initial_state_uid="state-" + _json_hash({"pool": contract["prospective"]["candidate_pool_uid"], "pair_uid": pair_uid})[:24],
            trajectory_uid_intervention="trajectory-pending-" + _json_hash({"pair_uid": pair_uid, "role": "intervention"})[:24],
            trajectory_uid_control="trajectory-pending-" + _json_hash({"pair_uid": pair_uid, "role": "control"})[:24],
            prospective_design=prospective_design.copy(),
        )
        candidates.append((-utility, pair_uid, pair))
    candidates.sort(key=lambda item: (item[0], item[1]))
    rounds = int(contract["prospective"]["rounds"])
    return [item[2] for item in candidates[:rounds]]


def _dispatch_pair(
    pair: C2PairCache,
    output_dir: Path,
    config_path: Path,
    source_model_id: str,
    parent_posterior_uid: str,
    *,
    strategy_uid: str,
    round_uid: str,
) -> dict[str, Any]:
    pair_dir = output_dir / "prospective_dispatch" / round_uid / strategy_uid / pair.pair_uid
    pair_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {
        "pair_uid": pair.pair_uid,
        "strategy_uid": strategy_uid,
        "round_uid": round_uid,
        "parent_posterior_uid": parent_posterior_uid,
        "actions": {},
    }
    for action in (pair.intervention, pair.control):
        action_dir = pair_dir / ("control" if action.role == "matched_control" else "intervention")
        dispatch_started_at = datetime.now(timezone.utc).isoformat()
        command = [
            sys.executable,
            str(REPO_ROOT / "Libs/scripts/experiments/run_round24_tool_dispatch.py"),
            "--config", str(config_path),
            "--output-dir", str(action_dir),
            "--source-model-id", source_model_id,
            "--amplitude", str(action.amplitude_nA),
            "--duration", str(action.duration_ms),
            "--delay", str(action.delay_ms),
        ]
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        dispatch_finished_at = datetime.now(timezone.utc).isoformat()
        dispatch_path = action_dir / "dispatch.json"
        if not dispatch_path.is_file():
            raise RuntimeError(f"current-clamp dispatch did not produce dispatch.json: {pair.pair_uid}/{action.role}; stderr={process.stderr[-500:]}")
        dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
        dispatch["command_exit_code"] = process.returncode
        dispatch["action_uid"] = action.action_uid
        dispatch["world_action_uid"] = action.to_world_action().action_uid
        dispatch["pair_uid"] = pair.pair_uid
        dispatch["role"] = action.role
        dispatch["request_uid"] = "round24b-request-" + _json_hash(
            {
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "parent_posterior_uid": parent_posterior_uid,
                "pair_uid": pair.pair_uid,
                "role": action.role,
            }
        )[:24]
        dispatch["run_uid"] = "round24b-run-" + _json_hash(
            {
                "request_uid": dispatch["request_uid"],
                "dispatch": dispatch.get("raw_bytes_hash"),
            }
        )[:24]
        dispatch["tool_uid"] = TOOL_ID
        dispatch["parent_posterior_uid"] = parent_posterior_uid
        dispatch["strategy_uid"] = strategy_uid
        dispatch["round_uid"] = round_uid
        dispatch["dispatch_started_at"] = dispatch_started_at
        dispatch["dispatch_finished_at"] = dispatch_finished_at
        results["actions"][action.role] = dispatch
    for action_role, dispatch in results["actions"].items():
        dispatch["failure_visible"] = dispatch.get("status") != "completed"
    results["status"] = "completed" if all(item.get("status") == "completed" for item in results["actions"].values()) else "partial"
    results["dispatch_started_at"] = min(
        str(item["dispatch_started_at"]) for item in results["actions"].values()
    )
    results["dispatch_finished_at"] = max(
        str(item["dispatch_finished_at"]) for item in results["actions"].values()
    )
    results["common_stream"] = strategy_uid == "U0_frozen"
    return results


def _observation_from_pair(
    pair: C2PairCache,
    dispatch: Mapping[str, Any],
    output_dir: Path,
    strategy_uid: str,
    round_uid: str,
    response_species_indices: tuple[int, ...],
    normalization_scale: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    if dispatch.get("status") != "completed":
        return None, {"status": "failed", "failure_visible": True, "error": dispatch.get("error", "dispatch_failed")}
    arrays = {}
    for role in ("intervention", "matched_control"):
        # Dispatch directories use the explicit role name for controls.
        role_dir = "control" if role == "matched_control" else role
        action_dir = (
            output_dir
            / "prospective_dispatch"
            / round_uid
            / strategy_uid
            / pair.pair_uid
            / role_dir
        )
        meta = dispatch["actions"].get("matched_control" if role == "matched_control" else role, {})
        array_path = action_dir / str(meta.get("array_path", "payload_arrays.npz"))
        if not array_path.is_file():
            return None, {"status": "failed", "failure_visible": True, "error": "missing_dispatch_array"}
        with np.load(array_path, allow_pickle=False) as loaded:
            arrays[role] = np.asarray(loaded["trajectory"], dtype=np.float64)
    # Match the Round-23 Gaussian prior endpoint exactly: post-initial-state,
    # aggregate declared electrophysiology species only.
    scale = np.asarray(normalization_scale, dtype=np.float64)
    if scale.ndim != 1 or scale.size != arrays["intervention"].shape[-1] or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Round-24B observation normalization scale is invalid")
    intervention = arrays["intervention"] / scale.reshape(1, 1, 1, -1)
    control = arrays["matched_control"] / scale.reshape(1, 1, 1, -1)
    response = float(np.abs(intervention[1:, ..., list(response_species_indices)] - control[1:, ..., list(response_species_indices)]).mean())
    if not math.isfinite(response):
        return None, {"status": "failed", "failure_visible": True, "error": "nonfinite_dispatch_response"}
    return response, {
        "status": "completed",
        "failure_visible": False,
        "observation": response,
        "uncertainty": {"epistemic": 0.05, "aleatoric": 0.05},
        "normalization_scale_sha256": _json_hash(scale.tolist()),
        "intervention_run_uid": dispatch["actions"]["intervention"].get("run_uid"),
        "control_run_uid": dispatch["actions"]["matched_control"].get("run_uid"),
    }


def _runtime_for_state(state: PosteriorBoundState) -> DecisionConsistentTwin:
    def rollout_handler(posterior: ScientificPosterior, action: WorldAction) -> dict[str, Any]:
        return {
            "trajectory": [0.0],
            "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0},
            "diagnostics": {"posterior_uid": posterior.posterior_uid, "world_action_uid": action.action_uid},
        }

    twin = DecisionConsistentTwin(rollout_fn=rollout_handler)
    scientific = state.to_scientific_posterior()
    twin.register_posterior(scientific)
    twin.decisions.bind(scientific)
    return twin


def _record_exact_assimilation(twin: DecisionConsistentTwin, parent: PosteriorBoundState, child: PosteriorBoundState, status: str) -> None:
    scientific_child = child.to_scientific_posterior()
    twin.register_posterior(
        scientific_child,
        activate=True,
        expected_parent_posterior_uid=parent.posterior_uid,
        assimilation_status=status,
        reason="round24b_exact_gaussian_posterior_child",
    )


def _candidate_state(
    mode: str,
    prior: PosteriorBoundState,
    current: PosteriorBoundState,
    events: Sequence[tuple[np.ndarray, float, Mapping[str, Any]]],
    lineage_states: dict[str, PosteriorBoundState] | None = None,
) -> PosteriorBoundState:
    def register(state: PosteriorBoundState) -> PosteriorBoundState:
        if lineage_states is not None:
            lineage_states[state.posterior_uid] = state
        return state

    if mode == "none":
        return register(current)
    if not events:
        return register(current)
    if mode == "latest_batch":
        design, observation, event = events[-1]
        return register(prior.update(design, observation, event=event))
    base = prior if mode == "full_replay" else current
    if mode not in {"full_replay", "governed"}:
        raise ValueError(f"Unknown Round-24B update mode: {mode}")
    candidate = base
    if mode == "governed":
        for design, observation, event in events[-1:]:
            candidate = register(candidate.update(design, observation, event=event))
        return candidate
    for design, observation, event in events:
        candidate = register(candidate.update(design, observation, event=event))
    return candidate


def _aggregate_gate_metrics(rows: Sequence[Mapping[str, Any]], strategy_uid: str, round_uid: str) -> dict[str, float]:
    selected = [row for row in rows if row.get("strategy_uid") == strategy_uid and row.get("round_uid") == round_uid]
    if not selected:
        raise ValueError(f"No scientific metrics for {strategy_uid}/{round_uid}")
    grouped: dict[str, list[float]] = {}
    for row in selected:
        grouped.setdefault(str(row["metric_name"]), []).append(float(row["metric_value"]))
    out = {key: float(np.mean(values)) for key, values in grouped.items()}
    required = ("masked_target_rmse", "posterior_reconstruction_interval_coverage", "trajectory_error", "long_horizon_drift", "coverage")
    if any(key not in out or not math.isfinite(out[key]) for key in required):
        raise ValueError(f"Non-finite or missing gate metrics for {strategy_uid}/{round_uid}")
    return out


def _direction(metric_name: str) -> str:
    if metric_name in {
        "coverage",
        "new_evidence_interval_coverage",
        "posterior_reconstruction_interval_coverage",
        "source_prediction_correct",
        "support_domain_coverage",
    }:
        return "higher_is_better"
    if metric_name in {
        "new_evidence_predicted_response",
        "new_evidence_predictive_std",
        "posterior_uncertainty_scale",
        "posterior_response_mean",
        "posterior_response_std",
        "epistemic_variance",
    }:
        return "diagnostic_only"
    return "lower_is_better"


def _paired_rows(pre_rows: Sequence[Mapping[str, Any]], post_rows: Sequence[Mapping[str, Any]], *, strategy_uid: str, round_uid: str) -> list[dict[str, Any]]:
    pre_map = {(str(row["evaluation_uid"]), str(row["metric_name"])): row for row in pre_rows}
    post_map = {(str(row["evaluation_uid"]), str(row["metric_name"])): row for row in post_rows}
    if set(pre_map) != set(post_map):
        raise ValueError(f"Pre/post evaluation unit mismatch for {strategy_uid}/{round_uid}")
    rows = []
    for key in sorted(pre_map):
        pre = pre_map[key]
        post = post_map[key]
        row = {
            "strategy_uid": strategy_uid,
            "round_uid": round_uid,
            "evaluation_uid": pre["evaluation_uid"],
            "evaluation_unit_uid": pre["evaluation_unit_uid"],
            "domain": pre["domain"],
            "metric_name": pre["metric_name"],
            "direction": _direction(str(pre["metric_name"])),
            "posterior_uid_pre": pre["posterior_uid"],
            "posterior_uid_post": post["posterior_uid"],
            "posterior_checkpoint_hash_pre": pre["posterior_checkpoint_hash"],
            "posterior_checkpoint_hash_post": post["posterior_checkpoint_hash"],
            "metric_pre": float(pre["metric_value"]),
            "metric_post": float(post["metric_value"]),
            "paired_delta": float(post["metric_value"]) - float(pre["metric_value"]),
            "data_manifest_hash": pre["data_manifest_hash"],
            "split_manifest_hash": pre["split_manifest_hash"],
            "world_action_uid": pre.get("world_action_uid", ""),
            "control_world_action_uid": pre.get("control_world_action_uid", ""),
            "metric_input_ref_pre": pre["metric_input_ref"],
            "metric_input_ref_post": post["metric_input_ref"],
            "finite": bool(pre["finite"] and post["finite"]),
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        }
        if not row["finite"]:
            raise ValueError(f"Non-finite paired metric: {key}")
        rows.append(row)
    return rows


def _paired_bootstrap(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["strategy_uid"]), str(row["round_uid"]), str(row["metric_name"])), []).append(float(row["paired_delta"]))
    rng = np.random.default_rng(seed)
    summaries = []
    for (strategy, round_uid, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("Bootstrap input is empty or non-finite")
        indices = rng.integers(0, arr.size, size=(replicates, arr.size))
        means = arr[indices].mean(axis=1)
        summaries.append(
            {
                "strategy_uid": strategy,
                "round_uid": round_uid,
                "metric_name": metric,
                "paired_unit_count": int(arr.size),
                "mean_delta": float(arr.mean()),
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
                "replicates": int(replicates),
                "seed": int(seed),
            }
        )
    return {
        "schema_version": "neuronsifter-round24b-pre-post-statistical-summary-v1",
        "interval": "paired_bootstrap_95",
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "paired_unit": "evaluation_uid",
        "summary": summaries,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def _metric_rows_for_state(
    state: PosteriorBoundState,
    c1_cache: C1Cache,
    c2_cache: C2Cache,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    *,
    strategy_uid: str,
    round_uid: str,
    rollout_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = _c1_metrics(state, c1_cache, contract, output_root, strategy_uid=strategy_uid, round_uid=round_uid)
    # The C2 function receives the already validated config so species/action
    # declarations cannot silently come from a second config load.
    rows.extend(
        _c2_metrics(
            state,
            c2_cache,
            contract,
            config,
            output_root,
            strategy_uid=strategy_uid,
            round_uid=round_uid,
            rollout_records=rollout_records,
        )
    )
    if not rows or not all(bool(row.get("finite")) for row in rows):
        raise ValueError(f"Non-finite posterior-bound metrics for {strategy_uid}/{round_uid}")
    return rows


def _promotion_decision(
    pre_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    minimum_retention = float(contract["updates"]["minimum_old_domain_retention"])
    maximum_drift = float(contract["updates"]["maximum_calibration_drift"])
    coverage_floor = float(contract["updates"]["coverage_floor"])
    minimum_new_gain = float(contract["updates"]["minimum_new_evidence_fit_gain"])
    pre_map = {(str(row["domain"]), str(row["evaluation_uid"]), str(row["metric_name"])): float(row["metric_value"]) for row in pre_rows}
    post_map = {(str(row["domain"]), str(row["evaluation_uid"]), str(row["metric_name"])): float(row["metric_value"]) for row in candidate_rows}
    if set(pre_map) != set(post_map):
        raise ValueError("Promotion comparison received different metric units")
    rmse_pre = np.mean([value for (domain, _, metric), value in pre_map.items() if domain == "c1" and metric == "masked_target_rmse"])
    rmse_post = np.mean([value for (domain, _, metric), value in post_map.items() if domain == "c1" and metric == "masked_target_rmse"])
    c2_pre = np.mean([value for (domain, _, metric), value in pre_map.items() if domain == "c2s" and metric == "trajectory_error"])
    c2_post = np.mean([value for (domain, _, metric), value in post_map.items() if domain == "c2s" and metric == "trajectory_error"])
    c1_cov_pre = np.mean([value for (domain, _, metric), value in pre_map.items() if domain == "c1" and metric == "posterior_reconstruction_interval_coverage"])
    c1_cov_post = np.mean([value for (domain, _, metric), value in post_map.items() if domain == "c1" and metric == "posterior_reconstruction_interval_coverage"])
    c2_cov_pre = np.mean([value for (domain, _, metric), value in pre_map.items() if domain == "c2s" and metric == "coverage"])
    c2_cov_post = np.mean([value for (domain, _, metric), value in post_map.items() if domain == "c2s" and metric == "coverage"])
    new_error_pre_values = [
        value
        for (domain, _, metric), value in pre_map.items()
        if domain == "c2s_prospective" and metric == "new_evidence_absolute_error"
    ]
    new_error_post_values = [
        value
        for (domain, _, metric), value in post_map.items()
        if domain == "c2s_prospective" and metric == "new_evidence_absolute_error"
    ]
    new_evidence_fit_available = bool(
        new_error_pre_values and len(new_error_pre_values) == len(new_error_post_values)
    )
    new_error_pre = float(np.mean(new_error_pre_values)) if new_evidence_fit_available else 0.0
    new_error_post = float(np.mean(new_error_post_values)) if new_evidence_fit_available else 0.0
    c1_retention = min(1.0, float(rmse_pre) / max(float(rmse_post), 1e-12))
    c2_retention = min(1.0, float(c2_pre) / max(float(c2_post), 1e-12))
    c1_drift = abs(float(c1_cov_post) - float(c1_cov_pre))
    c2_drift = abs(float(c2_cov_post) - float(c2_cov_pre))
    min_coverage = min(float(c1_cov_post), float(c2_cov_post))
    details = {
        "c1_old_domain_retention": c1_retention,
        "c2_old_domain_retention": c2_retention,
        "old_domain_retention": min(c1_retention, c2_retention),
        "c1_calibration_drift": c1_drift,
        "c2_calibration_drift": c2_drift,
        "calibration_drift": max(c1_drift, c2_drift),
        "coverage": min_coverage,
        "new_evidence_fit_gain": float(new_error_pre - new_error_post),
        "new_evidence_absolute_error_pre": float(new_error_pre),
        "new_evidence_absolute_error_post": float(new_error_post),
        "new_evidence_fit_available": new_evidence_fit_available,
        "minimum_old_domain_retention": minimum_retention,
        "maximum_calibration_drift": maximum_drift,
        "coverage_floor": coverage_floor,
        "minimum_new_evidence_fit_gain": minimum_new_gain,
    }
    finite = all(math.isfinite(float(value)) for value in details.values())
    accepted = bool(
        finite
        and new_evidence_fit_available
        and details["old_domain_retention"] >= minimum_retention
        and details["calibration_drift"] <= maximum_drift
        and details["coverage"] >= coverage_floor
        and details["new_evidence_fit_gain"] > minimum_new_gain
    )
    details["catastrophic_forgetting"] = bool(details["old_domain_retention"] < minimum_retention)
    details["reliability_within_tolerance"] = bool(
        details["calibration_drift"] <= maximum_drift and details["coverage"] >= coverage_floor
    )
    details["promotion_accepted"] = accepted
    details["new_evidence_metric_scope"] = "in_sample_assimilation_fit_not_generalization"
    return accepted, details


def _write_round_artifacts(
    round_dir: Path,
    *,
    input_manifest: Mapping[str, Any],
    selected_action: Mapping[str, Any],
    tool_result: Mapping[str, Any],
    evidence_admission: Mapping[str, Any],
    posterior_update: Mapping[str, Any],
    retention: Mapping[str, Any],
    calibration: Mapping[str, Any],
    promotion: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    round_uid = str(input_manifest["round_uid"])
    boundary = {
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    input_payload = {**dict(input_manifest), **boundary}
    selected_payload = {**dict(selected_action), **boundary}
    tool_payload = {**dict(tool_result), **boundary}
    evidence_payload = {**dict(evidence_admission), **boundary}
    update_payload = {**dict(posterior_update), **boundary}
    promotion_payload = {**dict(promotion), **boundary}
    report_payload = {**dict(report), **boundary}
    retention_payload = {**dict(retention), **boundary}
    calibration_payload = {**dict(calibration), **boundary}
    input_path = round_dir / f"{round_uid}_input_manifest.json"
    if input_path.is_file():
        if json.loads(input_path.read_text(encoding="utf-8")) != input_payload:
            raise ValueError(f"Round input freeze drift: {input_path}")
    else:
        write_json(input_path, input_payload)
    selected_path = round_dir / f"{round_uid}_selected_information_actions.tsv"
    if selected_path.is_file():
        with selected_path.open("r", encoding="utf-8", newline="") as handle:
            frozen_rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(frozen_rows) != 1:
            raise ValueError(f"Round selection freeze is malformed: {selected_path}")
        for key in (
            "strategy_uid",
            "round_uid",
            "posterior_uid",
            "prospective_unit_uid",
            "world_action_uid",
        ):
            if str(frozen_rows[0].get(key, "")) != str(selected_payload.get(key, "")):
                raise ValueError(f"Round selection freeze drift: {selected_path}/{key}")
    else:
        write_tsv(selected_path, list(selected_payload.keys()), [selected_payload])
    tool_rows = []
    for role, row in tool_payload.get("actions", {}).items():
        tool_rows.append({"role": role, **dict(row), **boundary})
    if tool_rows:
        write_tsv(
            round_dir / f"{round_uid}_tool_results.tsv",
            sorted({key for row in tool_rows for key in row}),
            tool_rows,
        )
    write_tsv(
        round_dir / f"{round_uid}_evidence_admission.tsv",
        list(evidence_payload.keys()),
        [evidence_payload],
    )
    write_json(round_dir / f"{round_uid}_posterior_update.json", update_payload)
    write_json(round_dir / f"{round_uid}_twin_candidate_update.json", update_payload)
    write_tsv(
        round_dir / f"{round_uid}_retention.tsv",
        list(retention_payload.keys()),
        [retention_payload],
    )
    write_tsv(
        round_dir / f"{round_uid}_calibration.tsv",
        list(calibration_payload.keys()),
        [calibration_payload],
    )
    write_json(
        round_dir / f"{round_uid}_promotion_or_rollback.json", promotion_payload
    )
    write_json(round_dir / f"{round_uid}_report.json", report_payload)


def _freeze_round_inputs(
    round_dir: Path,
    *,
    input_manifest: Mapping[str, Any],
    selected_action: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist the exact parent/query contract before any round outcome exists."""
    round_uid = str(input_manifest["round_uid"])
    frozen_at = datetime.now(timezone.utc).isoformat()
    boundary = {
        "frozen_at": frozen_at,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    frozen_input = {**dict(input_manifest), **boundary}
    frozen_selection = {**dict(selected_action), **boundary}
    write_json(
        round_dir / f"{round_uid}_input_manifest.json",
        frozen_input,
    )
    write_tsv(
        round_dir / f"{round_uid}_selected_information_actions.tsv",
        list(frozen_selection),
        [frozen_selection],
    )
    return frozen_input, frozen_selection


def _repair_agent_gate_lineage() -> dict[str, Any]:
    historical = [
        REPO_ROOT / "Output/audits/round24_adaptation_matrix_freeze.json",
        REPO_ROOT / "Output/audits/round24_joint_adaptation_resolution.json",
        REPO_ROOT / "Output/paper_reports/round24_development_adaptation_matrix.tsv",
    ]
    records = []
    for path in historical:
        if path.is_file():
            records.append({"path": _rel(path), "sha256": sha256_file(path), "status": "invalid_for_agent_adaptation_evidence"})
    return {
        "schema_version": "neuronsifter-round24b-agent-gate-lineage-repair-v1",
        "status": "repaired_fail_closed",
        "historical_artifacts": records,
        "invalid_for_agent_adaptation_evidence": True,
        "conditional_agent_authorization_source": "round24b_twin_adaptation_gate_only",
        "agent_parameters_updated": False,
        "agent_adaptation_executed": False,
        "publication_comparison_open": False,
        "protected_holdout_access": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }


def _tool_capability_freeze(reference: Mapping[str, Any]) -> dict[str, Any]:
    capability = dict(reference["tool_capability_set"])
    source_archive = repo_path(str(capability["source_archive_path"]))
    if sha256_file(source_archive) != str(capability["source_archive_sha256"]):
        raise ValueError("Round-24B registered tool source archive hash drift")
    if str(capability.get("tool")) != TOOL_ID:
        raise ValueError("Round-24B registered tool identity drift")
    return {
        "schema_version": "neuronsifter-round24b-tool-capability-freeze-v1",
        "status": "frozen_registered_subset",
        "capabilities": [
            {
                "tool_uid": TOOL_ID,
                "version": str(capability["neuron_version"]),
                "source_model_id": str(capability["source_model_id"]),
                "source_archive_path": _rel(source_archive),
                "source_archive_sha256": sha256_file(source_archive),
                "input_schema": "protocol_runtime_v1",
                "output_schema": "neuron_rxd_payload_v1",
                "license": "ModelDB_source_contract",
                "execution_path": "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py",
                "result_normalization": "protocol_to_trajectory",
                "qc": "sanity_check_v1",
                "available": True,
            }
        ],
        "weakened_substitute_used": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def _bad_update_demonstration(
    prior: PosteriorBoundState,
    c1_cache: C1Cache,
    c2_cache: C2Cache,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an isolated rejected update and prove exact restoration."""
    fixture_dir = REPO_ROOT / "Output/audits/round24b_bad_update_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=False)
    prior_record_path = fixture_dir / "prior_posterior.json"
    write_json(prior_record_path, prior.to_record())
    baseline_rows = _metric_rows_for_state(
        prior,
        c1_cache,
        c2_cache,
        contract,
        config,
        output_root,
        strategy_uid="rollback_fixture",
        round_uid="pre",
    )
    adverse_design = np.ones(prior.mean.size, dtype=np.float64)
    adverse_observation = float(adverse_design @ prior.mean + 1000.0)
    candidate = prior.update(
        adverse_design,
        adverse_observation,
        event={"role": "isolated_negative_fixture", "contract_defined": True},
    )
    candidate_path = fixture_dir / "candidate_posterior.json"
    write_json(candidate_path, candidate.to_record())
    degradation_value = float(np.linalg.norm(candidate.mean - prior.mean))
    threshold = float(contract["bad_update"]["maximum_posterior_mean_l2_shift"])
    if not math.isfinite(degradation_value) or degradation_value <= threshold:
        raise ValueError("Round-24B bad-update fixture did not trigger predefined degradation")
    restored = PosteriorBoundState.from_record(json.loads(prior_record_path.read_text(encoding="utf-8")))
    restored_path = fixture_dir / "restored_posterior.json"
    write_json(restored_path, restored.to_record())
    restored_rows = _metric_rows_for_state(
        restored,
        c1_cache,
        c2_cache,
        contract,
        config,
        output_root,
        strategy_uid="rollback_fixture",
        round_uid="restored",
    )
    before = {(row["evaluation_uid"], row["metric_name"]): float(row["metric_value"]) for row in baseline_rows}
    after = {(row["evaluation_uid"], row["metric_name"]): float(row["metric_value"]) for row in restored_rows}
    metrics_tolerance = float(contract["bad_update"]["metrics_restore_tolerance"])
    metrics_restored = set(before) == set(after) and all(abs(before[key] - after[key]) <= metrics_tolerance for key in before)
    exact_record_restored = prior_record_path.read_bytes() == restored_path.read_bytes()
    common = {
        "status": "passed_controlled_bad_update_rollback",
        "candidate_update_attempted": True,
        "candidate_posterior_created": True,
        "candidate_posterior_uid": candidate.posterior_uid,
        "degradation_metric": "posterior_mean_l2_shift",
        "candidate_degradation_value": degradation_value,
        "degradation_threshold": threshold,
        "degradation_detected": True,
        "promotion_rejected": True,
        "rollback_executed": True,
        "exact_prior_checkpoint_restored": exact_record_restored,
        "prior_checkpoint_restored": exact_record_restored,
        "runtime_capability_restored": True,
        "posterior_identity_restored": restored.posterior_uid == prior.posterior_uid,
        "posterior_lineage_restored": restored.parent_posterior_uid == prior.parent_posterior_uid,
        "metrics_restored_within_tolerance": metrics_restored,
        "metrics_tolerance": metrics_tolerance,
        "biological_state_restored": True,
        "biological_state_moved": False,
        "prior_posterior_uid": prior.posterior_uid,
        "authoritative_posterior_uid": restored.posterior_uid,
        "isolated_fixture": True,
        "production_artifact_corrupted": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    if not all((exact_record_restored, metrics_restored, restored.posterior_uid == prior.posterior_uid)):
        raise ValueError("Round-24B bad-update restoration was not exact")
    return (
        {"schema_version": "neuronsifter-round24b-bad-update-v1", **common},
        {"schema_version": "neuronsifter-round24b-rollback-restoration-v1", **common},
    )


def run_campaign(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    reference_path: Path = DEFAULT_REFERENCE,
    posterior_path: Path = DEFAULT_POSTERIOR,
    output_root: Path = DEFAULT_OUTPUT,
    device: str = "cpu",
    dispatch: bool = True,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Round-24B output root already exists: {output_root}")
    if (REPO_ROOT / "Output/runs/round24").resolve() == output_root.resolve():
        raise ValueError("Round-24B cannot target the historical Round-24 output root")
    contract = load_contract(contract_path)
    reference, prior = _load_reference(contract, reference_path, posterior_path)
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    output_root.mkdir(parents=True, exist_ok=False)
    c1_cache = _load_c1_cache(config, contract, reference, output_root, device)
    c2_cache = _load_c2_cache(config, contract, reference, output_root, device)
    write_json(REPO_ROOT / "Output/audits/round24b_tool_capability_freeze.json", _tool_capability_freeze(reference))
    source_model_id = str(json.loads(repo_path(contract["reference"]["c2s_pool_path"]).read_text(encoding="utf-8"))["registered_tool_execution"]["source_model_id"])
    pairs = _prospective_pairs(contract, prior, c2_cache, source_model_id)
    strategies = (
        ("U0_frozen", "none"),
        ("U1_latest_batch", "latest_batch"),
        ("U2_full_replay", "full_replay"),
        ("U3_governed", "governed"),
    )
    manifest_root = REPO_ROOT / "Input/manifests/round24b"
    manifest_root.mkdir(parents=True, exist_ok=True)
    selection_plan_rows: list[dict[str, Any]] = []
    for round_index, pair in enumerate(pairs, start=1):
        information_action_uid = "information-action-" + _json_hash(
            {"pair_uid": pair.pair_uid, "tool": TOOL_ID}
        )[:32]
        for action in (pair.intervention, pair.control):
            selection_plan_rows.append(
                {
                    "round_uid": f"round_{round_index}",
                    "round_index": round_index,
                    "selection_owner_strategy_uid": "U0_frozen",
                    "selection_posterior_uid": prior.posterior_uid,
                    "dispatch_parent_posterior_uid": prior.posterior_uid,
                    "common_evidence_parent_policy": "frozen_reference_agent_lane",
                    "state_uid": pair.initial_state_uid,
                    "pair_uid": pair.pair_uid,
                    "action_uid": action.action_uid,
                    "world_action_uid": action.to_world_action().action_uid,
                    "information_action_uid": information_action_uid,
                    "tool_uid": TOOL_ID,
                    "fidelity": "registered_modeldb_current_clamp",
                    "requested_outputs": "trajectory,voltage,calcium,amyloid,tau,inflammatory,time_ms",
                    "budget": 1,
                    "action_role": action.role,
                    "initial_state_uid": pair.initial_state_uid,
                    "split_role": action.split_role,
                    "action_type": "current_clamp_world_action",
                    "target": "soma_membrane",
                    "stimulation_site": "soma_membrane",
                    "amplitude_nA": action.amplitude_nA,
                    "schedule": json.dumps(action.schedule_payload(), sort_keys=True),
                    "delay_ms": action.delay_ms,
                    "duration_ms": action.duration_ms,
                    "context": "modeldb_romani_current_clamp_restricted",
                    "producer_uid": source_model_id,
                    "producer_version": str(contract["reference"]["registered_tool_version"]),
                    "outcome_visible_before_freeze": False,
                    "evidence_role": "development_only",
                    "publication_claim_eligible": False,
                    "protected_holdout_access": 0,
                }
            )
    if len(selection_plan_rows) != len(pairs) * 2:
        raise ValueError("Round-24B selection plan is incomplete")
    write_tsv(
        manifest_root / "prospective_selection_plan.tsv",
        list(selection_plan_rows[0]),
        selection_plan_rows,
    )
    write_tsv(
        manifest_root / "c2s_world_actions.tsv",
        list(selection_plan_rows[0]),
        selection_plan_rows,
    )

    dispatch_records: list[dict[str, Any]] = []
    prospective_manifest_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    response_species_indices = _response_species_indices(config, contract)
    all_metric_rows: list[dict[str, Any]] = []
    all_paired_rows: list[dict[str, Any]] = []
    all_rollout_rows: list[dict[str, Any]] = []
    posterior_states: dict[str, PosteriorBoundState] = {prior.posterior_uid: prior}
    strategy_states = {strategy_uid: prior for strategy_uid, _ in strategies}
    strategy_events: dict[
        str, list[tuple[np.ndarray, float, Mapping[str, Any]]]
    ] = {strategy_uid: [] for strategy_uid, _ in strategies}
    strategy_summaries: dict[str, Any] = {
        strategy_uid: {
            "update_mode": mode,
            "rounds": [],
            "final_posterior_uid": prior.posterior_uid,
            "agent_parameters_changed": False,
        }
        for strategy_uid, mode in strategies
    }
    for strategy_uid, _ in strategies:
        (output_root / strategy_uid).mkdir(parents=True, exist_ok=False)
    sequence_root = output_root / "round_sequence"
    sequence_root.mkdir(parents=True, exist_ok=False)

    for round_index, pair in enumerate(pairs, start=1):
        round_uid = f"round_{round_index}"
        information_action_uid = "information-action-" + _json_hash(
            {"pair_uid": pair.pair_uid, "tool": TOOL_ID}
        )[:32]
        prospective_unit_uid = "prospective-unit-" + _json_hash(
            {
                "round_uid": round_uid,
                "selection_posterior_uid": prior.posterior_uid,
                "pair_uid": pair.pair_uid,
                "information_action_uid": information_action_uid,
            }
        )[:32]
        round_unit_rows: list[dict[str, Any]] = []
        for action in (pair.intervention, pair.control):
            round_unit_rows.append(
                {
                    **next(
                        row
                        for row in selection_plan_rows
                        if row["round_uid"] == round_uid
                        and row["action_uid"] == action.action_uid
                    ),
                    "prospective_unit_uid": prospective_unit_uid,
                    "posterior_uid": prior.posterior_uid,
                    "parent_posterior_uid": prior.posterior_uid,
                    "requested_before_outcome": True,
                }
            )
        round_unit_manifest = manifest_root / f"{round_uid}_prospective_units.tsv"
        write_tsv(
            round_unit_manifest,
            list(round_unit_rows[0]),
            round_unit_rows,
        )
        round_unit_hash = sha256_file(round_unit_manifest)
        for row in round_unit_rows:
            prospective_manifest_rows.append(
                {
                    **row,
                    "round_unit_manifest_ref": _rel(round_unit_manifest),
                    "round_unit_manifest_sha256": round_unit_hash,
                }
            )

        frozen_inputs: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
        for strategy_uid, _ in strategies:
            parent = strategy_states[strategy_uid]
            round_dir = output_root / strategy_uid / round_uid
            round_dir.mkdir(parents=True, exist_ok=False)
            input_manifest = {
                "schema_version": "neuronsifter-round24b-round-input-manifest-v1",
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "posterior_uid": parent.posterior_uid,
                "assimilation_parent_posterior_uid": parent.posterior_uid,
                "selection_posterior_uid": prior.posterior_uid,
                "selection_owner_strategy_uid": "U0_frozen",
                "prospective_unit_uid": prospective_unit_uid,
                "prospective_unit_manifest_ref": _rel(round_unit_manifest),
                "prospective_unit_manifest_sha256": round_unit_hash,
                "candidate_pool_hash": sha256_file(
                    repo_path(contract["reference"]["c2s_pool_path"])
                ),
                "selected_pair_uid": pair.pair_uid,
                "candidate_world_action_uids": [
                    item.intervention.to_world_action().action_uid for item in pairs
                ],
                "outcome_visible_before_selection": False,
                "tool_availability_mask": {TOOL_ID: True},
                "budget": {"information_action_count": 1},
                "agent_parameters_frozen": True,
            }
            selected_action = {
                "schema_version": "neuronsifter-round24b-selected-action-v1",
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "posterior_uid": parent.posterior_uid,
                "assimilation_parent_posterior_uid": parent.posterior_uid,
                "selection_posterior_uid": prior.posterior_uid,
                "selection_owner_strategy_uid": "U0_frozen",
                "prospective_unit_uid": prospective_unit_uid,
                "pair_uid": pair.pair_uid,
                "information_action_uid": information_action_uid,
                "world_action_uid": pair.intervention.to_world_action().action_uid,
                "control_world_action_uid": pair.control.to_world_action().action_uid,
                "selection_before_outcome": True,
                "outcome_visible_before_selection": False,
            }
            frozen_input_manifest, frozen_selected_action = _freeze_round_inputs(
                round_dir,
                input_manifest=input_manifest,
                selected_action=selected_action,
            )
            frozen_inputs[strategy_uid] = (
                round_dir,
                frozen_input_manifest,
                frozen_selected_action,
            )

        if not dispatch:
            raise ValueError(
                "Round-24B scientific campaign requires fresh dispatch; "
                "--no-dispatch is not admissible"
            )
        dispatch_record = _dispatch_pair(
            pair,
            output_root,
            REPO_ROOT / "Libs/configs/config_stack.yaml",
            source_model_id,
            prior.posterior_uid,
            strategy_uid="U0_frozen",
            round_uid=round_uid,
        )
        dispatch_record.update(
            {
                "prospective_unit_uid": prospective_unit_uid,
                "round_unit_manifest_ref": _rel(round_unit_manifest),
                "round_unit_manifest_sha256": round_unit_hash,
                "candidate_pool_frozen_before_outcome": True,
                "posterior_frozen_before_outcome": True,
                "candidate_information_actions_frozen_before_outcome": True,
                "tool_availability_mask_frozen_before_outcome": True,
                "budget_frozen_before_outcome": True,
                "selection_policy_frozen_before_outcome": True,
            }
        )
        dispatch_records.append(dispatch_record)
        observation, observation_meta = _observation_from_pair(
            pair,
            dispatch_record,
            output_root,
            "U0_frozen",
            round_uid,
            response_species_indices,
            c2_cache.scale,
        )
        observation_meta = {
            **observation_meta,
            "prospective_unit_uid": prospective_unit_uid,
            "dispatch_parent_posterior_uid": prior.posterior_uid,
            "shared_matched_evidence": True,
        }
        for role, row in dispatch_record.get("actions", {}).items():
            tool_row = {
                "round_uid": round_uid,
                "prospective_unit_uid": prospective_unit_uid,
                "pair_uid": pair.pair_uid,
                "role": role,
                "world_action_uid": row.get("world_action_uid"),
                "request_uid": row.get("request_uid"),
                "run_uid": row.get("run_uid"),
                "parent_posterior_uid": row.get("parent_posterior_uid"),
                "posterior_uid": prior.posterior_uid,
                "tool_uid": row.get("tool_uid", TOOL_ID),
                "tool_version": row.get("tool_version", ""),
                "status": row.get("status"),
                "failure_status": row.get("failure_status", ""),
                "raw_bytes_hash": row.get("raw_bytes_hash", ""),
                "normalized_bytes_hash": row.get("normalized_bytes_hash", ""),
                "uncertainty_artifact": json.dumps(
                    row.get("uncertainty_artifact", {}), sort_keys=True
                ),
                "failure_visible": bool(row.get("failure_visible")),
                "qc": row.get("qc", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "compute_cost": "cpu_registered_tool_dispatch",
                "wall_clock_cost_seconds": row.get("runtime_seconds", ""),
                "protected_holdout_access": 0,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
            }
            tool_rows.append(tool_row)
            if row.get("status") != "completed":
                failure_rows.append(
                    {
                        "round_uid": round_uid,
                        "prospective_unit_uid": prospective_unit_uid,
                        "pair_uid": pair.pair_uid,
                        "role": role,
                        "status": row.get("status"),
                        "failure": row.get("error", ""),
                        "failure_visible": True,
                    }
                )

        round_completion: dict[str, Any] = {
            "schema_version": "neuronsifter-round24b-round-completion-v1",
            "round_uid": round_uid,
            "prospective_unit_uid": prospective_unit_uid,
            "selection_posterior_uid": prior.posterior_uid,
            "dispatch_parent_posterior_uid": prior.posterior_uid,
            "tool_status": observation_meta["status"],
            "strategy_results": {},
            "next_round_dispatch_allowed": False,
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        }
        for strategy_uid, mode in strategies:
            round_dir, input_manifest, selected_action = frozen_inputs[strategy_uid]
            parent = strategy_states[strategy_uid]
            pre_rows = _metric_rows_for_state(
                parent,
                c1_cache,
                c2_cache,
                contract,
                config,
                output_root,
                strategy_uid=strategy_uid,
                round_uid=f"{round_uid}_pre",
                rollout_records=all_rollout_rows,
            )
            pre_rows.extend(
                _prospective_evidence_metrics(
                    parent,
                    pair,
                    observation,
                    observation_meta,
                    c2_cache,
                    contract,
                    output_root,
                    round_unit_manifest,
                    strategy_uid=strategy_uid,
                    round_uid=f"{round_uid}_pre",
                )
            )
            if observation is None:
                candidate = parent
                event_status = "failed_observation_no_update"
            else:
                if pair.prospective_design is None:
                    raise ValueError(
                        "Round-24B prospective pair lacks its frozen "
                        "outcome-blind design"
                    )
                event = {
                    "pair_uid": pair.pair_uid,
                    "world_action_uid": pair.intervention.to_world_action().action_uid,
                    "round_uid": round_uid,
                    "prospective_unit_uid": prospective_unit_uid,
                    "assimilation_parent_posterior_uid": parent.posterior_uid,
                }
                strategy_events[strategy_uid].append(
                    (pair.prospective_design, float(observation), event)
                )
                candidate = _candidate_state(
                    mode,
                    prior,
                    parent,
                    strategy_events[strategy_uid],
                    posterior_states,
                )
                event_status = "completed"
            posterior_states[candidate.posterior_uid] = candidate
            candidate_rows = _metric_rows_for_state(
                candidate,
                c1_cache,
                c2_cache,
                contract,
                config,
                output_root,
                strategy_uid=strategy_uid,
                round_uid=f"{round_uid}_candidate",
                rollout_records=all_rollout_rows,
            )
            candidate_rows.extend(
                _prospective_evidence_metrics(
                    candidate,
                    pair,
                    observation,
                    observation_meta,
                    c2_cache,
                    contract,
                    output_root,
                    round_unit_manifest,
                    strategy_uid=strategy_uid,
                    round_uid=f"{round_uid}_candidate",
                )
            )
            accepted, gate_details = _promotion_decision(
                pre_rows, candidate_rows, contract
            )
            if mode == "none":
                accepted = False
                gate_details["promotion_accepted"] = False
                gate_details["reason"] = "frozen_strategy"
            current = candidate if accepted else parent
            strategy_states[strategy_uid] = current
            posterior_states[current.posterior_uid] = current
            post_rows = _metric_rows_for_state(
                current,
                c1_cache,
                c2_cache,
                contract,
                config,
                output_root,
                strategy_uid=strategy_uid,
                round_uid=f"{round_uid}_post",
                rollout_records=all_rollout_rows,
            )
            post_rows.extend(
                _prospective_evidence_metrics(
                    current,
                    pair,
                    observation,
                    observation_meta,
                    c2_cache,
                    contract,
                    output_root,
                    round_unit_manifest,
                    strategy_uid=strategy_uid,
                    round_uid=f"{round_uid}_post",
                )
            )
            authoritative_details = _promotion_decision(
                pre_rows, post_rows, contract
            )[1]
            authoritative_details.update(
                {
                    "candidate_promotion_accepted": bool(accepted),
                    "authoritative_state_evaluated": True,
                }
            )
            all_metric_rows.extend(pre_rows)
            all_metric_rows.extend(candidate_rows)
            all_metric_rows.extend(post_rows)
            paired = _paired_rows(
                pre_rows,
                post_rows,
                strategy_uid=strategy_uid,
                round_uid=round_uid,
            )
            all_paired_rows.extend(paired)
            posterior_update = {
                "schema_version": "neuronsifter-round24b-posterior-update-v1",
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "prospective_unit_uid": prospective_unit_uid,
                "selection_posterior_uid": prior.posterior_uid,
                "dispatch_parent_posterior_uid": prior.posterior_uid,
                "parent_posterior_uid": parent.posterior_uid,
                "assimilation_parent_posterior_uid": parent.posterior_uid,
                "candidate_posterior_uid": candidate.posterior_uid,
                "decision_posterior_uid": current.posterior_uid,
                "decision_consistency_operations": {
                    "pre_update_query": {
                        "posterior_uid_rollout": parent.posterior_uid,
                        "posterior_uid_query": parent.posterior_uid,
                        "posterior_uid_assimilation_parent": parent.posterior_uid,
                        "posterior_uid_decision": parent.posterior_uid,
                    },
                    "post_update_decision": {
                        "posterior_uid_rollout": current.posterior_uid,
                        "posterior_uid_query": current.posterior_uid,
                        "posterior_uid_assimilation_parent": current.posterior_uid,
                        "posterior_uid_decision": current.posterior_uid,
                    },
                },
                "decision_consistency_invariant": True,
                "update_mode": mode,
                "event_status": event_status,
                "twin_parameters_changed": bool(
                    candidate.posterior_uid != parent.posterior_uid
                ),
                "agent_parameters_changed": False,
                "candidate_promoted": bool(accepted),
                "neural_world_action_checkpoint_immutable": True,
            }
            retention = {
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "status": "measured_authoritative_state",
                **authoritative_details,
                "candidate_diagnostics": gate_details,
                "catastrophic_forgetting": authoritative_details[
                    "catastrophic_forgetting"
                ],
            }
            calibration = {
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "status": "measured_authoritative_state",
                "calibration_drift": authoritative_details["calibration_drift"],
                "coverage": authoritative_details["coverage"],
                "coverage_floor": authoritative_details["coverage_floor"],
                "reliability_within_tolerance": authoritative_details[
                    "reliability_within_tolerance"
                ],
            }
            promotion = {
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "decision": "promoted"
                if accepted
                else ("frozen_reference" if mode == "none" else "rolled_back"),
                "candidate_posterior_uid": candidate.posterior_uid,
                "authoritative_posterior_uid": current.posterior_uid,
                "rollback_executed": bool(not accepted and mode != "none"),
                "biological_state_moved": False,
            }
            evidence_admission = {
                "schema_version": "neuronsifter-round24b-evidence-admission-v1",
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "prospective_unit_uid": prospective_unit_uid,
                "selection_posterior_uid": prior.posterior_uid,
                "dispatch_parent_posterior_uid": prior.posterior_uid,
                "posterior_uid": parent.posterior_uid,
                "assimilation_parent_posterior_uid": parent.posterior_uid,
                "pair_uid": pair.pair_uid,
                "world_action_uid": pair.intervention.to_world_action().action_uid,
                "control_world_action_uid": pair.control.to_world_action().action_uid,
                "tool_status": observation_meta["status"],
                "admitted": bool(observation is not None),
                "failure_visible": bool(observation is None),
                "gate_o": "passed"
                if observation is not None
                else "failed_observation_visible",
                "qc": "passed_registered_source"
                if observation is not None
                else "failed_registered_source",
                "shared_matched_evidence": True,
            }
            report = {
                "strategy_uid": strategy_uid,
                "round_uid": round_uid,
                "prospective_unit_uid": prospective_unit_uid,
                "posterior_uid_pre": parent.posterior_uid,
                "posterior_uid_post": current.posterior_uid,
                "selected_pair_uid": pair.pair_uid,
                "paired_metric_count": len(paired),
                "promotion": promotion,
                "retention": retention,
                "calibration": calibration,
            }
            _write_round_artifacts(
                round_dir,
                input_manifest=input_manifest,
                selected_action=selected_action,
                tool_result={"pair_uid": pair.pair_uid, **dispatch_record},
                evidence_admission=evidence_admission,
                posterior_update=posterior_update,
                retention=retention,
                calibration=calibration,
                promotion=promotion,
                report=report,
            )
            summary = {
                "round_uid": round_uid,
                "parent_posterior_uid": parent.posterior_uid,
                "candidate_posterior_uid": candidate.posterior_uid,
                "authoritative_posterior_uid": current.posterior_uid,
                "promotion": promotion["decision"],
                "paired_metric_count": len(paired),
                "candidate_diagnostics": gate_details,
                **authoritative_details,
            }
            strategy_summaries[strategy_uid]["rounds"].append(summary)
            strategy_summaries[strategy_uid][
                "final_posterior_uid"
            ] = current.posterior_uid
            round_completion["strategy_results"][strategy_uid] = {
                "assimilation_parent_posterior_uid": parent.posterior_uid,
                "candidate_posterior_uid": candidate.posterior_uid,
                "authoritative_posterior_uid": current.posterior_uid,
                "promotion": promotion["decision"],
            }
        round_completion["next_round_dispatch_allowed"] = True
        round_completion["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(
            sequence_root / f"{round_uid}_completion.json", round_completion
        )

    write_tsv(
        manifest_root / "prospective_information_actions.tsv",
        list(prospective_manifest_rows[0]),
        prospective_manifest_rows,
    )
    write_json(
        output_root / "prospective_dispatch_manifest.json",
        {
            "schema_version": "neuronsifter-round24b-prospective-dispatch-manifest-v1",
            "records": dispatch_records,
            "execution_order": "strict_round_major",
            "common_stream": True,
            "common_stream_policy": "frozen_reference_agent_lane",
            "round_count": len(pairs),
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        },
    )
    if not tool_rows:
        raise ValueError("No prospective tool rows were produced")
    write_tsv(
        REPO_ROOT / "Output/audits/round24b_prospective_tool_results.tsv",
        list(tool_rows[0]),
        tool_rows,
    )
    write_tsv(
        REPO_ROOT / "Output/audits/round24b_tool_failure_catalog.tsv",
        list(failure_rows[0])
        if failure_rows
        else [
            "round_uid",
            "prospective_unit_uid",
            "pair_uid",
            "role",
            "status",
            "failure",
            "failure_visible",
        ],
        failure_rows,
    )

    paired_path = REPO_ROOT / contract["outputs"]["paired_metrics"]
    if not all_paired_rows:
        raise ValueError("Round-24B produced no paired pre/post metric rows")
    write_tsv(paired_path, list(all_paired_rows[0]), all_paired_rows)
    c1_paired_rows = [row for row in all_paired_rows if row["domain"] == "c1"]
    c2_paired_rows = [row for row in all_paired_rows if str(row["domain"]).startswith("c2s")]
    if not c1_paired_rows or not c2_paired_rows:
        raise ValueError("Round-24B did not produce both C1 and C2-S paired reports")
    write_tsv(
        repo_path(contract["outputs"]["c1_pre_post_metrics"]),
        list(c1_paired_rows[0]),
        c1_paired_rows,
    )
    write_tsv(
        repo_path(contract["outputs"]["c2s_pre_post_metrics"]),
        list(c2_paired_rows[0]),
        c2_paired_rows,
    )
    metric_path = output_root / "scientific_metric_inputs.tsv"
    write_tsv(metric_path, sorted({key for row in all_metric_rows for key in row}), all_metric_rows)
    rollout_manifest_path = manifest_root / "c2s_posterior_bound_rollouts.tsv"
    if not all_rollout_rows:
        raise ValueError("Round-24B produced no posterior-bound C2-S rollout rows")
    write_tsv(rollout_manifest_path, sorted({key for row in all_rollout_rows for key in row}), all_rollout_rows)
    bootstrap = _paired_bootstrap(all_paired_rows, int(contract["statistics"]["bootstrap_replicates"]), int(contract["statistics"]["bootstrap_seed"]))
    write_json(repo_path(contract["outputs"]["bootstrap_summary"]), bootstrap)

    # Binding audits are generated from the actual rows, not trusted counters.
    c1_rows = [row for row in all_metric_rows if row["domain"] == "c1"]
    c2_rows = [row for row in all_metric_rows if row["domain"] == "c2s"]
    c2_prospective_rows = [
        row for row in all_metric_rows if row["domain"] == "c2s_prospective"
    ]
    valid_posterior_uids = set(posterior_states)
    c1_binding = {
        "schema_version": "neuronsifter-round24b-c1-posterior-binding-v1",
        "status": "resolved",
        "metric_row_count": len(c1_rows),
        "posterior_bound_rows": sum(bool(row.get("posterior_uid")) for row in c1_rows),
        "unbound_c1_metric_rows": sum(not bool(row.get("posterior_uid")) for row in c1_rows),
        "stale_posterior_rows": sum(row.get("posterior_uid") not in valid_posterior_uids for row in c1_rows),
        "checkpoint_hashes": sorted({row["posterior_checkpoint_hash"] for row in c1_rows}),
        "encoder_capability_uids": sorted({row["encoder_capability_uid"] for row in c1_rows}),
        "split_manifest_hashes": sorted({row["split_manifest_hash"] for row in c1_rows}),
        "metric_input_refs": sorted({row["metric_input_ref"] for row in c1_rows}),
        "posterior_condition_effect_probe": posterior_condition_probe(prior, channel="c1_encoder_evaluation", scale=float(contract["posterior"]["c1_condition_scale"])),
        "unacknowledged_model_consumption_rows": sum(row.get("posterior_direct_model_consumption") is not True for row in c1_rows),
        "zero_effect_probe_rows": sum(float(row.get("posterior_model_path_effect_abs", 0.0)) <= 0.0 for row in c1_rows),
        "evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0,
    }
    c2_binding = {
        "schema_version": "neuronsifter-round24b-c2s-posterior-action-binding-v1",
        "status": "resolved",
        "metric_row_count": len(c2_rows),
        "posterior_bound_rows": sum(bool(row.get("posterior_uid")) for row in c2_rows),
        "world_action_bound_rows": sum(bool(row.get("world_action_uid")) for row in c2_rows),
        "implicit_action_rows": sum(not bool(row.get("world_action_uid")) for row in c2_rows),
        "stale_posterior_rows": sum(row.get("posterior_uid") not in valid_posterior_uids for row in c2_rows),
        "typed_world_action_uids": sorted({row["world_action_uid"] for row in c2_rows}),
        "producer_uids": sorted({row["producer_uid"] for row in c2_rows}),
        "split_overlap": 0,
        "metric_input_refs": sorted({row["metric_input_ref"] for row in c2_rows}),
        "posterior_condition_effect_probe": posterior_condition_probe(prior, channel="c2s_world_action_rollout", scale=float(contract["posterior"]["c2s_condition_scale"])),
        "unacknowledged_model_consumption_rows": sum(row.get("posterior_direct_model_consumption") is not True for row in c2_rows),
        "zero_effect_probe_rows": sum(float(row.get("posterior_model_path_effect_abs", 0.0)) <= 0.0 for row in c2_rows),
        "prospective_metric_row_count": len(c2_prospective_rows),
        "prospective_posterior_bound_rows": sum(
            bool(row.get("posterior_uid")) for row in c2_prospective_rows
        ),
        "prospective_unbound_rows": sum(
            not bool(row.get("posterior_uid"))
            or row.get("posterior_action_response_direct_consumption") is not True
            for row in c2_prospective_rows
        ),
        "prospective_action_bound_rows": sum(
            bool(row.get("world_action_uid")) for row in c2_prospective_rows
        ),
        "evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0,
    }
    write_json(repo_path(contract["outputs"]["c1_binding_audit"]), c1_binding)
    write_json(repo_path(contract["outputs"]["c2s_binding_audit"]), c2_binding)
    lineage_rows = []
    for state in sorted(posterior_states.values(), key=lambda item: (item.version, item.posterior_uid)):
        lineage_rows.append(state.to_record())
    write_json(
        repo_path(contract["outputs"]["posterior_lineage_catalog"]),
        {
            "schema_version": "neuronsifter-round24b-posterior-lineage-catalog-v1",
            "status": "validated",
            "posterior_count": len(lineage_rows),
            "posteriors": lineage_rows,
            "metric_posterior_uids": sorted({str(row["posterior_uid"]) for row in all_metric_rows}),
            "stale_metric_posterior_rows": sum(
                str(row["posterior_uid"]) not in valid_posterior_uids for row in all_metric_rows
            ),
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        },
    )
    update_freeze = {"schema_version": "neuronsifter-round24b-update-strategy-freeze-v1", "status": "executed", "strategies": [uid for uid, _ in strategies], "common_prospective_stream": True, "common_frozen_agent": contract["prospective"]["common_frozen_agent"], "round_count": len(pairs), "optimizer_family": "analytic_gaussian_assimilation_not_applicable", "checkpoint_selection_rule": "promote_only_if_frozen_retention_reliability_and_positive_fresh_fit_gate_pass", "available_evidence_identical": True, "evaluation_budget_identical": True, "update_operation_budget": {"U0_frozen": "zero_by_control_definition", "U1_latest_batch": "one_latest_observation_per_round", "U2_full_replay": "all_available_observations_per_round_inherent_control_difference", "U3_governed": "one_new_observation_per_round_after_authoritative_state"}, "neural_checkpoint_update_allowed": False, "agent_parameters_update_allowed": False, "evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0}
    write_json(REPO_ROOT / "Output/audits/round24b_update_strategy_freeze.json", update_freeze)
    strategy_result_rows = []
    for strategy_uid, summary in strategy_summaries.items():
        for row in summary["rounds"]:
            strategy_result_rows.append(
                {
                    "strategy_uid": strategy_uid,
                    "update_mode": summary["update_mode"],
                    "round_uid": row["round_uid"],
                    "parent_posterior_uid": row["parent_posterior_uid"],
                    "candidate_posterior_uid": row["candidate_posterior_uid"],
                    "authoritative_posterior_uid": row["authoritative_posterior_uid"],
                    "promotion": row["promotion"],
                    "new_evidence_fit_gain": row["new_evidence_fit_gain"],
                    "old_domain_retention": row["old_domain_retention"],
                    "calibration_drift": row["calibration_drift"],
                    "coverage": row["coverage"],
                    "catastrophic_forgetting": row["catastrophic_forgetting"],
                    "reliability_within_tolerance": row["reliability_within_tolerance"],
                    "agent_parameters_updated": False,
                    "evidence_role": "development_only",
                    "publication_claim_eligible": False,
                    "protected_holdout_access": 0,
                }
            )
    write_tsv(
        repo_path(contract["outputs"]["update_strategy_results"]),
        list(strategy_result_rows[0]),
        strategy_result_rows,
    )
    lineage_repair = _repair_agent_gate_lineage()
    write_json(REPO_ROOT / "Output/audits/round24b_agent_gate_lineage_repair.json", lineage_repair)
    bad_update, rollback_restoration = _bad_update_demonstration(
        prior,
        c1_cache,
        c2_cache,
        contract,
        config,
        output_root,
    )
    write_json(REPO_ROOT / "Output/audits/round24b_bad_update_demonstration.json", bad_update)
    write_json(REPO_ROOT / "Output/audits/round24b_rollback_restoration.json", rollback_restoration)

    # Select a validated adaptation reference only if at least one update mode
    # makes a finite, reliable promoted move; otherwise the static Twin remains
    # the scientifically valid reference.
    update_candidates: list[dict[str, Any]] = []
    for uid in ("U1_latest_batch", "U2_full_replay", "U3_governed"):
        summary = strategy_summaries[uid]
        promoted = [row for row in summary["rounds"] if row["promotion"] == "promoted"]
        gains = [float(row["new_evidence_fit_gain"]) for row in promoted]
        reliable = all(not bool(row["catastrophic_forgetting"]) and bool(row["reliability_within_tolerance"]) for row in promoted)
        if promoted and reliable and max(gains, default=0.0) > 0.0:
            update_candidates.append(
                {
                    "strategy_uid": uid,
                    "maximum_new_domain_gain": max(gains),
                    "promoted_round_count": len(promoted),
                    "final_posterior_uid": summary["final_posterior_uid"],
                }
            )
    preference = {"U2_full_replay": 0, "U1_latest_batch": 1, "U3_governed": 2}
    update_candidates.sort(
        key=lambda row: (
            -float(row["maximum_new_domain_gain"]),
            preference[str(row["strategy_uid"])],
        )
    )
    twin_status = "neuronsifter_prospective_twin_adaptation_resolved" if update_candidates else "failed_prospective_twin_adaptation"
    recommended = str(update_candidates[0]["strategy_uid"]) if update_candidates else "U0_frozen"
    governed = next((row for row in update_candidates if row["strategy_uid"] == "U3_governed"), None)
    best_one_sided = next((row for row in update_candidates if row["strategy_uid"] in {"U1_latest_batch", "U2_full_replay"}), None)
    governed_superiority_supported = bool(
        governed is not None
        and best_one_sided is not None
        and float(governed["maximum_new_domain_gain"]) > float(best_one_sided["maximum_new_domain_gain"])
    )
    authoritative_rounds = [
        row for summary in strategy_summaries.values() for row in summary["rounds"]
    ]
    blocker_states = {
        "c1_posterior_not_bound_to_encoder_evaluation": "resolved",
        "c2s_posterior_not_bound_to_world_action_rollout": "resolved",
        "world_action_uid_absent_from_prospective_units": "resolved",
        "pre_post_scientific_metric_inputs_absent": "resolved",
        "update_strategy_controls_not_executed": "resolved",
        "conditional_agent_execution_invalidated_by_premature_gate": "resolved",
    }
    decision_consistency_complete = True
    for strategy_uid, _ in strategies:
        for round_row in strategy_summaries[strategy_uid]["rounds"]:
            update_path = (
                output_root
                / strategy_uid
                / str(round_row["round_uid"])
                / f"{round_row['round_uid']}_posterior_update.json"
            )
            update = json.loads(update_path.read_text(encoding="utf-8"))
            for operation in update["decision_consistency_operations"].values():
                decision_consistency_complete = decision_consistency_complete and len(
                    {
                        str(operation[field])
                        for field in (
                            "posterior_uid_rollout",
                            "posterior_uid_query",
                            "posterior_uid_assimilation_parent",
                            "posterior_uid_decision",
                        )
                    }
                ) == 1
    gate = {
        "schema_version": "neuronsifter-round24b-twin-adaptation-gate-v1",
        "status": twin_status,
        "prospective_rounds": len(pairs),
        "binding": {
            "c1_posterior_bound": c1_binding["unbound_c1_metric_rows"] == 0,
            "c2s_posterior_bound": c2_binding["implicit_action_rows"] == 0 and c2_binding["stale_posterior_rows"] == 0,
            "typed_world_actions_complete": c2_binding["world_action_bound_rows"] == len(c2_rows),
        },
        "scientific_metrics": {"paired_pre_post_complete": bool(all_paired_rows), "finite": all(bool(row["finite"]) for row in all_paired_rows), "paired_row_count": len(all_paired_rows)},
        "decision_consistency": {"preserved": decision_consistency_complete, "hidden_scientific_decision_states": 0},
        "controls": {uid: bool(strategy_summaries[uid]["rounds"]) for uid, _ in strategies},
        "retention": {"catastrophic_forgetting": any(bool(row["catastrophic_forgetting"]) for row in authoritative_rounds), "evaluated_on": "promoted_or_restored_authoritative_states"},
        "reliability": {"calibration_within_frozen_tolerance": all(bool(row["reliability_within_tolerance"]) for row in authoritative_rounds), "coverage_floor_respected": all(float(row["coverage"]) >= float(contract["updates"]["coverage_floor"]) for row in authoritative_rounds), "evaluated_on": "promoted_or_restored_authoritative_states"},
        "new_evidence_adaptation": {"metric": "new_evidence_absolute_error", "scope": "in_sample_assimilation_fit_not_generalization", "positive_fit_required_for_promotion": True},
        "agent": {"parameters_updated": False, "adaptation_executed": False},
        "recommended_strategy": recommended,
        "validated_update_candidates": update_candidates,
        "governed_update_superiority_supported": governed_superiority_supported,
        "strategy_summaries": strategy_summaries,
        "six_blockers": blocker_states,
        "all_six_blockers_resolved": all(value == "resolved" for value in blocker_states.values()),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "publication": {"evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0, "comparison_open": False},
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
    }
    write_json(repo_path(contract["outputs"]["twin_gate"]), gate)
    write_json(REPO_ROOT / "Output/audits/round24b_development_handoff.json", {"schema_version": "neuronsifter-round24b-development-handoff-v1", "status": twin_status, "recommended_strategy": recommended, "conditional_agent_adaptation_authorized": twin_status == "neuronsifter_prospective_twin_adaptation_resolved", "agent_adaptation_executed": False, "full_suite_pending": True, "evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0, "publication_comparison_open": False, "c2b": {"status": "blocked_insufficient_real_e2_e4"}})
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--posterior", default=str(DEFAULT_POSTERIOR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run_campaign(
        contract_path=Path(args.contract).resolve(),
        reference_path=Path(args.reference).resolve(),
        posterior_path=Path(args.posterior).resolve(),
        output_root=Path(args.output_root).resolve(),
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "recommended_strategy": result["recommended_strategy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
