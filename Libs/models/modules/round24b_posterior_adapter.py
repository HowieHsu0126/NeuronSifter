"""Posterior-bound conditioning for the Round-24B development evaluation.

The frozen neural checkpoints remain unchanged.  This module makes the
scientific posterior an explicit, versioned input to the evaluation channels:
the C1 path consumes its reliability/uncertainty channel and the restricted
C2-S path consumes its typed current-clamp action-response channel.  The
calibration layer is deliberately small and auditable; it is not a second
scientific state or a replacement neural model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from Libs.models.modules.decision_consistent_runtime import ScientificPosterior
from Libs.scripts.experiments.run_round23_shared_belief_replay import GaussianBelief


SCHEMA_VERSION = "round24b_gaussian_calibration_posterior-v1"
SOURCE_SCHEMA_VERSION = "round23_gaussian_calibration_posterior_v1"


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and non-empty")
    return array


@dataclass(frozen=True)
class PosteriorBoundState:
    """Immutable Gaussian scientific posterior with a content-derived UID."""

    posterior_uid: str
    version: int
    mean: np.ndarray
    covariance: np.ndarray
    noise_variance: float
    source_posterior_uid: str
    parent_posterior_uid: str | None
    source_checkpoint_sha256: str
    source_pool_manifest_sha256: str
    lineage_event: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    source_schema_version: str = SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        mean = _finite_array(self.mean, "posterior mean", 1)
        covariance = _finite_array(self.covariance, "posterior covariance", 2)
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("Posterior covariance shape does not match mean")
        if not self.posterior_uid.startswith("posterior-"):
            raise ValueError("Round-24B posterior UID must use the canonical versioned prefix")
        if int(self.version) < 1 or not self.source_posterior_uid:
            raise ValueError("Posterior version and source identity are required")
        if not math.isfinite(float(self.noise_variance)) or self.noise_variance <= 0.0:
            raise ValueError("Posterior noise variance must be finite and positive")
        if not self.source_checkpoint_sha256 or not self.source_pool_manifest_sha256:
            raise ValueError("Posterior source hashes are required")
        if not isinstance(self.lineage_event, Mapping):
            raise TypeError("Posterior lineage_event must be a mapping")
        canonical_event = _canonical(self.lineage_event)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) < -1e-8:
            raise ValueError("Posterior covariance must be positive semidefinite")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "lineage_event", MappingProxyType(canonical_event))
        expected_uid = self._content_uid(
            mean=mean,
            covariance=covariance,
            noise_variance=float(self.noise_variance),
            source_posterior_uid=self.source_posterior_uid,
            parent_posterior_uid=self.parent_posterior_uid,
            version=int(self.version),
            source_checkpoint_sha256=self.source_checkpoint_sha256,
            source_pool_manifest_sha256=self.source_pool_manifest_sha256,
            lineage_event=canonical_event,
        )
        if self.posterior_uid != expected_uid:
            raise ValueError("Round-24B posterior UID does not match its serialized scientific content")

    @staticmethod
    def _content_uid(
        *,
        mean: np.ndarray,
        covariance: np.ndarray,
        noise_variance: float,
        source_posterior_uid: str,
        parent_posterior_uid: str | None,
        version: int,
        source_checkpoint_sha256: str,
        source_pool_manifest_sha256: str,
        lineage_event: Mapping[str, Any],
    ) -> str:
        identity = {
            "schema_version": SCHEMA_VERSION,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "source_posterior_uid": source_posterior_uid,
            "parent_posterior_uid": parent_posterior_uid,
            "version": int(version),
            "mean": mean,
            "covariance": covariance,
            "noise_variance": float(noise_variance),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_pool_manifest_sha256": source_pool_manifest_sha256,
            "event": lineage_event,
        }
        return "posterior-" + content_hash(identity)[:32]

    @classmethod
    def from_gaussian(
        cls,
        belief: GaussianBelief,
        *,
        source_posterior_uid: str,
        source_checkpoint_sha256: str,
        source_pool_manifest_sha256: str,
        noise_variance: float,
        version: int = 1,
        parent_posterior_uid: str | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> "PosteriorBoundState":
        mean = _finite_array(belief.mean, "posterior mean", 1)
        covariance = _finite_array(belief.covariance, "posterior covariance", 2)
        lineage_event = dict(event or {})
        uid = cls._content_uid(
            mean=mean,
            covariance=covariance,
            noise_variance=float(noise_variance),
            source_posterior_uid=source_posterior_uid,
            parent_posterior_uid=parent_posterior_uid,
            version=int(version),
            source_checkpoint_sha256=source_checkpoint_sha256,
            source_pool_manifest_sha256=source_pool_manifest_sha256,
            lineage_event=lineage_event,
        )
        return cls(
            posterior_uid=uid,
            version=int(version),
            mean=mean,
            covariance=covariance,
            noise_variance=float(noise_variance),
            source_posterior_uid=source_posterior_uid,
            parent_posterior_uid=parent_posterior_uid,
            source_checkpoint_sha256=source_checkpoint_sha256,
            source_pool_manifest_sha256=source_pool_manifest_sha256,
            lineage_event=lineage_event,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PosteriorBoundState":
        return cls(
            posterior_uid=str(record["posterior_uid"]),
            version=int(record["version"]),
            mean=np.asarray(record["mean"], dtype=np.float64),
            covariance=np.asarray(record["covariance"], dtype=np.float64),
            noise_variance=float(record["noise_variance"]),
            source_posterior_uid=str(record["source_posterior_uid"]),
            parent_posterior_uid=(str(record["parent_posterior_uid"]) if record.get("parent_posterior_uid") else None),
            source_checkpoint_sha256=str(record["source_checkpoint_sha256"]),
            source_pool_manifest_sha256=str(record["source_pool_manifest_sha256"]),
            lineage_event=dict(record.get("lineage_event", {})),
            schema_version=str(record.get("schema_version", SCHEMA_VERSION)),
            source_schema_version=str(record.get("source_schema_version", SOURCE_SCHEMA_VERSION)),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_schema_version": self.source_schema_version,
            "posterior_uid": self.posterior_uid,
            "version": int(self.version),
            "parent_posterior_uid": self.parent_posterior_uid,
            "source_posterior_uid": self.source_posterior_uid,
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "noise_variance": float(self.noise_variance),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_pool_manifest_sha256": self.source_pool_manifest_sha256,
            "lineage_event": dict(self.lineage_event),
            "record_sha256": content_hash({
                "posterior_uid": self.posterior_uid,
                "mean": self.mean,
                "covariance": self.covariance,
                "noise_variance": self.noise_variance,
                "lineage_event": self.lineage_event,
            }),
        }

    def to_scientific_posterior(self) -> ScientificPosterior:
        state = {
            "gaussian_mean": self.mean.tolist(),
            "gaussian_covariance": self.covariance.tolist(),
            "calibration_layer": "round24b_posterior_bound",
        }
        uncertainty = {
            "noise_variance": float(self.noise_variance),
            "epistemic": float(np.trace(self.covariance) / self.mean.size),
            "aleatoric": float(self.noise_variance),
        }
        support = {
            "source_posterior_uid": self.source_posterior_uid,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_pool_manifest_sha256": self.source_pool_manifest_sha256,
        }
        mechanism = {
            "posterior_family": "current_clamp_action_response_calibration",
            "schema_version": SCHEMA_VERSION,
        }
        reliability = {
            "status": "unavailable_development_evaluation",
            "tri_score_available": False,
            "decision_gate_eligible": False,
            "reason": "round24b_posterior_has_no_independent_tri_measurement",
        }
        return ScientificPosterior(
            posterior_uid=self.posterior_uid,
            version=self.version,
            parent_posterior_uid=self.parent_posterior_uid,
            state=MappingProxyType(state),
            uncertainty=MappingProxyType(uncertainty),
            support=MappingProxyType(support),
            mechanism=MappingProxyType(mechanism),
            reliability=MappingProxyType(reliability),
        )

    def update(self, design: np.ndarray, observation: float, *, event: Mapping[str, Any] | None = None) -> "PosteriorBoundState":
        design = _finite_array(design, "posterior design", 1)
        if design.size != self.mean.size:
            raise ValueError("Posterior design dimension mismatch")
        if not math.isfinite(float(observation)):
            raise ValueError("Posterior observation must be finite")
        column = self.covariance @ design
        innovation = float(design @ column + self.noise_variance)
        gain = column / max(innovation, 1e-12)
        residual = float(observation - design @ self.mean)
        mean = self.mean + gain * residual
        covariance = self.covariance - np.outer(gain, design @ self.covariance)
        covariance = 0.5 * (covariance + covariance.T)
        belief = GaussianBelief(mean, covariance)
        return self.from_gaussian(
            belief,
            source_posterior_uid=self.source_posterior_uid,
            source_checkpoint_sha256=self.source_checkpoint_sha256,
            source_pool_manifest_sha256=self.source_pool_manifest_sha256,
            noise_variance=self.noise_variance,
            version=self.version + 1,
            parent_posterior_uid=self.posterior_uid,
            event=event or {"observation": float(observation), "design": design.tolist()},
        )

    def action_response(self, design: np.ndarray) -> tuple[float, float]:
        design = _finite_array(design, "posterior action design", 1)
        if design.size != self.mean.size:
            raise ValueError("Posterior action design dimension mismatch")
        mean = float(design @ self.mean)
        variance = float(design @ self.covariance @ design + self.noise_variance)
        if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("Posterior action response is invalid")
        return mean, math.sqrt(variance)

    def c1_uncertainty_scale(self, *, scale: float = 0.05) -> float:
        mean_signal = float(np.linalg.norm(self.mean) / math.sqrt(self.mean.size))
        covariance_signal = float(
            math.sqrt(
                max(
                    np.trace(self.covariance) / self.mean.size
                    + self.noise_variance,
                    0.0,
                )
            )
        )
        value = float(np.tanh(mean_signal + covariance_signal))
        factor = math.exp(float(np.clip(scale * value, -0.25, 0.25)))
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("C1 posterior uncertainty scale is invalid")
        return factor

    def condition_hash(self, channel: str, payload: Mapping[str, Any]) -> str:
        return content_hash({"channel": channel, "posterior_uid": self.posterior_uid, "payload": payload})


def posterior_condition_probe(state: PosteriorBoundState, *, channel: str, scale: float) -> dict[str, Any]:
    if channel not in {"c1_encoder_evaluation", "c2s_world_action_rollout"}:
        raise ValueError("Unknown Round-24B posterior channel")
    if channel == "c1_encoder_evaluation":
        before = 1.0
        after = state.c1_uncertainty_scale(scale=scale)
    else:
        before = 0.0
        after = float(np.linalg.norm(state.mean) * scale)
    effect = abs(after - before)
    if not math.isfinite(effect):
        raise ValueError("Posterior condition effect probe is non-finite")
    return {
        "channel": channel,
        "posterior_uid": state.posterior_uid,
        "condition_hash": state.condition_hash(channel, {"scale": scale, "effect": effect}),
        "effect_before": before,
        "effect_after": after,
        "effect_abs": effect,
    }


def posterior_conditioned_c1_forward(
    model: Any,
    model_inputs: Mapping[str, torch.Tensor],
    forward_options: Mapping[str, Any],
    posterior: PosteriorBoundState,
    *,
    condition_scale: float,
    sample_posterior: bool,
) -> dict[str, Any]:
    """Run the frozen C1 encoder with the exact scientific posterior.

    The calibration factor enters the encoder before latent sampling and
    reconstruction.  This is deliberately an explicit forward input rather
    than a metric-layer adjustment, so every emitted C1 prediction can be
    traced to the posterior that produced it.
    """
    feature_ids = model_inputs.get("feature_ids")
    if feature_ids is None or feature_ids.ndim != 2:
        raise ValueError("C1 posterior-bound forward requires rank-two feature IDs")
    uncertainty_scale = posterior.c1_uncertainty_scale(scale=condition_scale)
    observation_scale = torch.full(
        (feature_ids.shape[0],),
        uncertainty_scale,
        dtype=torch.float32,
        device=feature_ids.device,
    )
    output = model(
        **model_inputs,
        **forward_options,
        sample_posterior=sample_posterior,
        scientific_posterior_uncertainty_scale=observation_scale,
    )
    if output.get("scientific_posterior_condition_applied").item() is not True:
        raise RuntimeError("C1 encoder did not acknowledge scientific-posterior conditioning")
    if not torch.allclose(
        output["scientific_posterior_uncertainty_scale"], observation_scale
    ):
        raise RuntimeError("C1 encoder returned a different posterior condition")
    for key in ("reconstructed_values", "latent_log_variance"):
        if not torch.isfinite(output[key]).all():
            raise FloatingPointError(f"C1 posterior-conditioned output is non-finite: {key}")
    output["posterior_uid"] = posterior.posterior_uid
    output["posterior_condition_hash"] = posterior.condition_hash(
        "c1_encoder_evaluation",
        {
            "condition_scale": condition_scale,
            "uncertainty_scale": uncertainty_scale,
            "sample_posterior": bool(sample_posterior),
        },
    )
    return output


def posterior_conditioned_c2_prediction(
    model: Any,
    trajectories: torch.Tensor,
    action_features: torch.Tensor,
    action_designs: np.ndarray,
    posterior: PosteriorBoundState,
    *,
    condition_scale: float = 1.0,
    response_channel_species_indices: tuple[int, int],
    response_channel_species_ids: tuple[str, str],
    apply_condition_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the frozen C2-S backbone with an explicit posterior condition.

    The condition enters the same physical state-conditioning tensor as the
    typed action.  Its two distinct channels are the posterior-predicted
    response mean and standard deviation for the current-clamp design.  Their
    configured species slots are explicit and ordered, so the two quantities
    cannot be collapsed or broadcast ambiguously.  No action feature is
    overwritten, so amplitude and duration retain their canonical semantics.
    """
    if trajectories.ndim != 5:
        raise ValueError("C2 posterior-conditioned trajectories must be [batch,time,region,compartment,species]")
    if action_features.ndim != 2 or action_features.shape[0] != trajectories.shape[0]:
        raise ValueError("C2 posterior-conditioned action features are not batch aligned")
    designs = np.asarray(action_designs, dtype=np.float64)
    if designs.shape[0] != trajectories.shape[0] or designs.shape[1] != posterior.mean.size:
        raise ValueError("C2 posterior action designs do not match the rollout batch")
    response = np.asarray([posterior.action_response(row) for row in designs], dtype=np.float32)
    if apply_condition_mask is None:
        raise ValueError("C2 posterior conditioning requires an explicit typed world-action mask")
    mask = np.asarray(apply_condition_mask, dtype=bool)
    if mask.shape != (trajectories.shape[0],):
        raise ValueError("C2 posterior condition mask is not batch aligned")
    response_tensor = torch.as_tensor(response, dtype=trajectories.dtype, device=trajectories.device)
    source = trajectories[:, :-1].reshape(
        trajectories.shape[0], trajectories.shape[1] - 1,
        model.region_count * model.compartment_count, model.species_count,
    )
    if len(response_channel_species_indices) != 2 or len(response_channel_species_ids) != 2:
        raise ValueError("C2 posterior conditioning requires exactly two response channel slots")
    if len(set(response_channel_species_indices)) != 2 or len(set(response_channel_species_ids)) != 2:
        raise ValueError("C2 posterior mean and uncertainty channels require distinct species slots")
    if any(index < 0 or index >= model.species_count for index in response_channel_species_indices):
        raise ValueError("C2 posterior response species index is outside the state grid")
    posterior_state = trajectories.new_zeros(
        (trajectories.shape[0], 1, 1, model.species_count)
    )
    mask_tensor = torch.as_tensor(mask, dtype=trajectories.dtype, device=trajectories.device).view(-1, 1, 1)
    response_mean = response_tensor[:, 0].view(-1, 1, 1) * float(condition_scale) * mask_tensor
    response_std = response_tensor[:, 1].view(-1, 1, 1) * float(condition_scale) * mask_tensor
    mean_index, std_index = response_channel_species_indices
    posterior_state[..., mean_index] = response_mean
    posterior_state[..., std_index] = response_std
    output = model(
        source,
        action_features,
        scientific_posterior_species_condition=posterior_state,
    )
    output.update({
        "posterior_uid": posterior.posterior_uid,
        "posterior_condition_hash": posterior.condition_hash(
            "c2s_world_action_rollout",
            {
                "designs": designs.tolist(),
                "condition_scale": condition_scale,
                "apply_condition_mask": mask.tolist(),
                "response_channel_mapping": {
                    "posterior_response_mean": response_channel_species_ids[0],
                    "posterior_response_std": response_channel_species_ids[1],
                },
            },
        ),
        "posterior_response_mean": response[:, 0],
        "posterior_response_std": response[:, 1],
        "response_channel_species_indices": list(response_channel_species_indices),
        "response_channel_species_ids": list(response_channel_species_ids),
        "response_channel_mapping": {
            "posterior_response_mean": response_channel_species_ids[0],
            "posterior_response_std": response_channel_species_ids[1],
        },
        "condition_applied_mask": mask.tolist(),
    })
    if output.get("scientific_posterior_condition_applied") is not True:
        raise RuntimeError("C2-S model did not acknowledge scientific-posterior conditioning")
    for key in ("trajectory_mean", "trajectory_std"):
        if not torch.isfinite(output[key]).all():
            raise FloatingPointError(f"C2 posterior-conditioned output is non-finite: {key}")
    return output


__all__ = [
    "PosteriorBoundState",
    "SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "content_hash",
    "posterior_condition_probe",
    "posterior_conditioned_c1_forward",
    "posterior_conditioned_c2_prediction",
]
