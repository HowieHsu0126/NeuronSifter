from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import json

import torch
from torch import nn

from Libs.utils.data.native_observable_profile import family_grid, quantity_support
from Libs.utils.data.joint_observation import RetainedObservation, observation_distribution, retained_log_likelihood
from Libs.utils.data.joint_physical_parameters import (
    candidate_conditioning, parameter_conditioning_width, candidate_from_payload, candidate_payload,
    parameter_schema_identity, validate_candidate_parameters,
)
from Libs.utils.data.local_membrane import (
    LocalMembraneSnapshot, ResolvedMembraneReadout, parse_readouts,
    physical_readout_bound, validate_physical_readout, payload as membrane_payload,
)
from Libs.models.modules.world_action_encoder import _digest_vector
from Libs.models.layers.feed_forward import FeedForwardBlock
from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES, ranking_stability_context_tensor, tri_score_from_components


@dataclass(frozen=True)
class ExplicitWorldStatePosterior:
    z_tx: torch.Tensor
    z_pk: torch.Tensor
    z_env: torch.Tensor
    z_ephys: torch.Tensor
    z_morph: torch.Tensor
    factor_mask: torch.Tensor

    def validate(self) -> "ExplicitWorldStatePosterior":
        factors = (self.z_tx, self.z_pk, self.z_env, self.z_ephys, self.z_morph)
        if any(value.ndim != 2 for value in factors):
            raise ValueError("World-state posterior factors must be rank-2 tensors.")
        if len({value.shape for value in factors}) != 1:
            raise ValueError("World-state posterior factors must have aligned shapes.")
        if self.factor_mask.shape != (factors[0].shape[0], 5):
            raise ValueError("World-state posterior factor mask must have shape [batch, 5].")
        if not all(torch.isfinite(value).all() for value in factors):
            raise ValueError("World-state posterior contains NaN or Inf.")
        return self

    def stacked(self) -> torch.Tensor:
        self.validate()
        return torch.stack((self.z_tx, self.z_pk, self.z_env, self.z_ephys, self.z_morph), dim=1)


class WorldStatePosterior(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        species_count: int,
        posterior_hidden_dim: int,
        posterior_dropout_rate: float,
        pk_dim: int,
        env_dim: int,
        ephys_dim: int,
        joint_config: dict | None = None,
        native_context_width: int | None = None,
    ) -> None:
        super().__init__()
        observation_stat_dim = 2
        self.tx_head = FeedForwardBlock(latent_dim + species_count + observation_stat_dim, posterior_hidden_dim, latent_dim, posterior_dropout_rate)
        self.pk_head = FeedForwardBlock(latent_dim + species_count + species_count + pk_dim + observation_stat_dim, posterior_hidden_dim, pk_dim, posterior_dropout_rate)
        self.env_head = FeedForwardBlock(latent_dim + species_count + species_count + env_dim + latent_dim + observation_stat_dim, posterior_hidden_dim, env_dim, posterior_dropout_rate)
        self.ephys_head = FeedForwardBlock(latent_dim + ephys_dim + latent_dim + observation_stat_dim, posterior_hidden_dim, ephys_dim, posterior_dropout_rate)
        self.morph_head = FeedForwardBlock(latent_dim + species_count + species_count + latent_dim + observation_stat_dim, posterior_hidden_dim, latent_dim, posterior_dropout_rate)
        self.joint_config = dict(joint_config or {})
        self.family = self.joint_config.get("family", "deterministic_compatibility")
        if self.family not in {"deterministic_compatibility", "finite_joint_particles"}:
            raise ValueError("Unknown world-state posterior family.")
        self.factor_dimensions = dict(zip(FACTOR_NAMES, (latent_dim, pk_dim, env_dim, ephys_dim, latent_dim)))
        if self.family == "finite_joint_particles":
            if native_context_width is None:
                raise ValueError("Joint posterior requires native context width.")
            particle_width = sum(self.factor_dimensions.values()) + 4
            self.joint_score = FeedForwardBlock(latent_dim + particle_width, posterior_hidden_dim, 1, posterior_dropout_rate)
            self.joint_factor_projection = nn.Linear(particle_width, native_context_width)
            self.joint_field_projection = nn.Linear(species_count, native_context_width)
            self.joint_parameter_projection = nn.Linear(species_count, native_context_width)
            self.physical_parameter_schema = self.joint_config.get("physical_parameter_schema")
            if self.physical_parameter_schema is not None:
                width = parameter_conditioning_width(self.physical_parameter_schema)
                self.joint_candidate_parameter_projection = nn.Linear(2 * width, native_context_width, bias=False)
                self.register_buffer("joint_parameter_schema_identity", torch.tensor(
                    list(bytes.fromhex(parameter_schema_identity(self.physical_parameter_schema))), dtype=torch.uint8))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "joint_parameter_schema_identity"
        if hasattr(self, "joint_parameter_schema_identity") and key in state_dict:
            if not torch.equal(state_dict[key].cpu(), self.joint_parameter_schema_identity.cpu()):
                raise ValueError("Joint physical parameter checkpoint schema identity mismatch")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def infer_joint(self, encoded_observation: torch.Tensor, observation_features: torch.Tensor,
                    support: "JointParticleSupport", observations: tuple[RetainedObservation, ...], field_normalization=None) -> "JointWorldStatePosterior":
        if self.family != "finite_joint_particles":
            raise ValueError("Joint posterior family was not selected.")
        support.validate()
        for name, dimension in self.factor_dimensions.items():
            if support.factors[name].shape[-1] != dimension:
                raise ValueError(f"Joint factor width mismatch: {name}.")
        b, j = support.prior_weights.shape
        regime = encoded_observation.new_tensor([[_digest_vector(value) for value in row] for row in support.regime_ids])
        particle_features = torch.cat([*[support.factors[name] for name in FACTOR_NAMES], regime], -1)
        score = self.joint_score(torch.cat([encoded_observation.unsqueeze(1).expand(-1, j, -1), particle_features], -1)).squeeze(-1)
        likelihood = score.new_zeros(b, j)
        seen = []
        retained = torch.zeros(b, dtype=torch.bool, device=score.device)
        for index, observation in enumerate(observations):
            if observation.observation_uid in seen or observation.sequence_number != index:
                raise ValueError("Initial observations require unique ordered identities.")
            assays = self.joint_config.get("registered_occupancy_assays", {})
            parameters, nuisance = support.validate_observation(observation, assays)
            likelihood = likelihood + retained_log_likelihood(observation, parameters, nuisance, assays)
            retained |= (observation.observed | observation.left_censored).any(-1)
            seen.append(observation.observation_uid)
        log_weights = _normalized_log_weights(
            support.prior_weights.double().log() + likelihood.double() + score.double() * retained.unsqueeze(-1))
        field_view = support.initial_field
        if field_normalization is not None:
            field_normalization.assert_support(support)
            mask = None
            if support.quantity_profile is not None:
                mask = torch.as_tensor(quantity_support(family_grid(
                    support.quantity_profile, support.source_family_uid), support.source_uid)[0], device=field_view.device)
            field_view = field_normalization.normalize(field_view, mask)
        context = (self.joint_factor_projection(particle_features).unsqueeze(2).unsqueeze(3)
                   + self.joint_field_projection(field_view)
                   + self.joint_parameter_projection(support.physical_parameters).unsqueeze(2).unsqueeze(3))
        if self.physical_parameter_schema is not None:
            named = candidate_conditioning(support, self.physical_parameter_schema)
            context = context + self.joint_candidate_parameter_projection(named).unsqueeze(2).unsqueeze(3)
        elif support.physical_parameter_hypotheses is not None:
            raise ValueError("Named physical parameters cannot be silently omitted by an unbound model")
        uids = tuple(seen)
        return JointWorldStatePosterior(support, log_weights, context, observation_features, uids, len(uids)-1,
            _posterior_uid(support, uids, None, log_weights, context), None, likelihood,
            "observation_conditioned" if retained.any() else "prior_only_no_retained_observation")


    def forward(
        self,
        encoded_patient: torch.Tensor,
        baseline_field: torch.Tensor,
        parameter_posterior: torch.Tensor,
        region_prior_context: torch.Tensor,
        observation_mask: torch.Tensor,
        pk_indices: torch.Tensor,
        env_indices: torch.Tensor,
        ephys_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = encoded_patient.shape[0]
        baseline_summary = baseline_field.mean(dim=(1, 2))
        prior_summary = region_prior_context.mean(dim=0, keepdim=True).expand(batch_size, -1)
        observation_stats = torch.stack(
            [
                observation_mask.mean(dim=1),
                observation_mask.sum(dim=1),
            ],
            dim=1,
        )
        pk_summary = baseline_field.index_select(-1, pk_indices).mean(dim=(1, 2))
        env_summary = baseline_field.index_select(-1, env_indices).mean(dim=(1, 2))
        ephys_summary = baseline_field.index_select(-1, ephys_indices).mean(dim=(1, 2))
        return {
            "z_tx": self.tx_head(torch.cat([encoded_patient, baseline_summary, observation_stats], dim=1)),
            "z_pk": self.pk_head(torch.cat([encoded_patient, baseline_summary, parameter_posterior, pk_summary, observation_stats], dim=1)),
            "z_env": self.env_head(torch.cat([encoded_patient, baseline_summary, parameter_posterior, env_summary, prior_summary, observation_stats], dim=1)),
            "z_ephys": self.ephys_head(torch.cat([encoded_patient, ephys_summary, prior_summary, observation_stats], dim=1)),
            "z_morph": self.morph_head(torch.cat([encoded_patient, baseline_summary, parameter_posterior, prior_summary, observation_stats], dim=1)),
        }


class ObservationAndScreeningHeads(nn.Module):
    def __init__(
        self,
        patient_state_dim: int,
        target_feature_dim: int,
        latent_dim: int,
        hidden_dim: int,
        compartment_count: int,
        species_count: int,
        tri_weight_by_component: dict[str, float],
    ) -> None:
        super().__init__()
        self.patient_state_dim = patient_state_dim
        self.compartment_count = compartment_count
        self.species_count = species_count
        self.target_encoder = FeedForwardBlock(target_feature_dim, hidden_dim, latent_dim, 0.0)
        self.disease_head = nn.Linear(latent_dim + species_count, 1)
        self.target_score_head = nn.Sequential(
            nn.Linear(latent_dim * 2 + species_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.regimen_score_head = nn.Sequential(
            nn.Linear(latent_dim * 2 + species_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pkpd_consistency_head = nn.Sequential(
            nn.Linear(compartment_count * 2 + species_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        tri_input_dim = latent_dim * 2 + species_count + len(TRI_COMPONENT_NAMES)
        self.tri_heads = nn.ModuleDict(
            {
                component_name: nn.Sequential(
                    nn.Linear(tri_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for component_name in TRI_COMPONENT_NAMES
            }
        )
        self.tri_weight_by_component = {name: float(tri_weight_by_component[name]) for name in TRI_COMPONENT_NAMES}
        self.register_buffer(
            "tri_weight_vector",
            torch.tensor([self.tri_weight_by_component[name] for name in TRI_COMPONENT_NAMES], dtype=torch.float32),
        )

    def disease_logit(self, encoded_patient: torch.Tensor, baseline_field: torch.Tensor) -> torch.Tensor:
        field_summary = baseline_field.mean(dim=(1, 2))
        return self.disease_head(torch.cat([encoded_patient, field_summary], dim=1)).squeeze(-1)

    def target_scores(
        self,
        encoded_patient: torch.Tensor,
        baseline_field: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_targets = self.target_encoder(target_features)
        patient_summary = baseline_field.mean(dim=(1, 2)).unsqueeze(1).expand(-1, encoded_targets.shape[0], -1)
        repeated_patient = encoded_patient.unsqueeze(1).expand(-1, encoded_targets.shape[0], -1)
        repeated_targets = encoded_targets.unsqueeze(0).expand(encoded_patient.shape[0], -1, -1)
        scores = self.target_score_head(torch.cat([repeated_patient, repeated_targets, patient_summary], dim=-1)).squeeze(-1)
        return scores, encoded_targets

    def regimen_summaries(self, final_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_field = final_field.mean(dim=(2, 3))
        exposure = final_field[..., 0].mean(dim=(0, 2))
        occupancy = final_field[..., 1].mean(dim=(0, 2))
        return pooled_field, exposure, occupancy

    def tri_outputs(
        self,
        encoded_patient: torch.Tensor,
        encoded_regimens: torch.Tensor,
        pooled_field: torch.Tensor,
        regimen_scores: torch.Tensor,
        pkpd_consistency: torch.Tensor,
        tri_context: dict[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        regimen_count = regimen_scores.shape[1]
        calibration_default = (
            torch.maximum(torch.sigmoid(regimen_scores), 1.0 - torch.sigmoid(regimen_scores))
            if tri_context is None or "calibration" not in tri_context
            else tri_context["calibration"]
        )
        coverage = torch.ones_like(regimen_scores) if tri_context is None or "coverage" not in tri_context else tri_context["coverage"]
        ranking_stability = ranking_stability_context_tensor(regimen_scores)
        if tri_context is not None and "ranking_stability" in tri_context:
            ranking_stability = tri_context["ranking_stability"]
        ood_score = torch.ones_like(regimen_scores) if tri_context is None or "ood_score" not in tri_context else tri_context["ood_score"]
        if tri_context is not None and "pkpd_physics_consistency" in tri_context:
            pkpd_component = tri_context["pkpd_physics_consistency"]
        else:
            pkpd_component = pkpd_consistency
        repeated_patient = encoded_patient.unsqueeze(1).expand(-1, regimen_count, -1)
        repeated_regimen = encoded_regimens.unsqueeze(0).expand(encoded_patient.shape[0], -1, -1)
        tri_input = torch.cat(
            [
                repeated_patient,
                repeated_regimen,
                pooled_field,
                torch.stack(
                    [calibration_default, coverage, pkpd_component, ranking_stability, ood_score],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        tri_components = {
            component_name: torch.sigmoid(self.tri_heads[component_name](tri_input).squeeze(-1))
            for component_name in TRI_COMPONENT_NAMES
        }
        tri_score = tri_score_from_components(tri_components, self.tri_weight_vector)
        return tri_components, tri_score



FACTOR_NAMES = ("z_tx", "z_pk", "z_env", "z_ephys", "z_morph")
JOINT_POSTERIOR_SCHEMA = "observation_conditioned_joint_particles.v1"


@dataclass(frozen=True)
class JointParticleSupport:
    support_uid: str
    source_uid: str
    source_snapshot_ref: str
    source_kind: str
    role: str
    sample_uids: tuple[str, ...]
    particle_uids: tuple[tuple[str, ...], ...]
    factors: dict[str, torch.Tensor]
    regime_ids: tuple[tuple[str, ...], ...]
    initial_field: torch.Tensor
    # Preserve the legacy species-aligned bindings used by frozen numerical programs.
    physical_parameters: torch.Tensor
    prior_weights: torch.Tensor
    readout_parameters: dict[str, dict[str, object]]
    source_nuisance: dict[str, torch.Tensor]
    species_units: tuple[str, ...]
    axes: tuple[str, ...] = ("time", "region", "compartment", "species")
    local_membrane: LocalMembraneSnapshot | None = None
    quantity_profile: dict | None = None
    source_family_uid: str | None = None
    physical_parameter_hypotheses: dict | None = None

    def validate(self) -> "JointParticleSupport":
        if not all((self.support_uid, self.source_uid, self.source_snapshot_ref)):
            raise ValueError("Joint support requires source/snapshot/support identity.")
        if self.source_kind not in {"controlled_simulated", "public_observation"}:
            raise ValueError("Joint support source kind is not declared.")
        if self.role not in {"train", "development", "validation", "calibration", "inference"}:
            raise ValueError("Joint support role is not admitted.")
        if self.prior_weights.ndim != 2 or min(self.prior_weights.shape) < 1:
            raise ValueError("Joint support requires nonempty sample/particle axes.")
        b, j = self.prior_weights.shape
        if len(self.sample_uids) != b or len(set(self.sample_uids)) != b:
            raise ValueError("Joint support sample identity mismatch.")
        for rows in (self.particle_uids, self.regime_ids):
            if len(rows) != b or any(len(row) != j or not all(row) for row in rows):
                raise ValueError("Particle/regime identity cardinality mismatch.")
        if any(len(set(row)) != j for row in self.particle_uids):
            raise ValueError("Joint source particle identities must be unique.")
        if set(self.factors) != set(FACTOR_NAMES):
            raise ValueError("Joint support requires all five complete factors.")
        for name, value in self.factors.items():
            if value.ndim != 3 or value.shape[:2] != (b, j) or value.shape[-1] < 1 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid complete particle factor {name}.")
        if self.initial_field.ndim != 5 or self.initial_field.shape[:2] != (b, j) or not torch.isfinite(self.initial_field).all():
            raise ValueError("Joint initial field requires sample/particle/region/compartment/species axes.")
        s = self.initial_field.shape[-1]
        if self.quantity_profile is not None:
            grid = family_grid(self.quantity_profile, self.source_family_uid)
            expected_shape = tuple(len(grid[k]) for k in ("region_id", "compartment_id", "species_id"))
            if tuple(self.initial_field.shape[-3:]) != expected_shape:
                raise ValueError("Joint observable profile/initial field mismatch")
            supported = torch.as_tensor(quantity_support(grid, self.source_uid)[0], device=self.initial_field.device)
            if not (self.initial_field.masked_select(~supported.expand_as(self.initial_field)) == 0).all():
                raise ValueError("Unsupported initial quantities must remain missing masked placeholders")
        elif self.source_family_uid is not None:
            raise ValueError("Source family metadata requires its quantity profile")
        if self.physical_parameters.shape != (b, j, s) or not torch.isfinite(self.physical_parameters).all():
            raise ValueError("Particle physical parameter bindings must align to species.")
        if len(self.species_units) != s or not all(self.species_units) or self.axes != ("time", "region", "compartment", "species"):
            raise ValueError("Joint support requires explicit field units and axes.")
        if (not torch.isfinite(self.prior_weights).all() or (self.prior_weights < 0).any()
                or not (self.prior_weights.sum(-1) > 0).all()):
            raise ValueError("Joint prior support has invalid or zero total mass.")
        if self.physical_parameter_hypotheses is not None:
            validate_candidate_parameters(self.physical_parameter_hypotheses, self)
        if self.local_membrane is not None:
            self.local_membrane.validate(self)
        bound_channels = set()
        for channel, parameters in self.readout_parameters.items():
            if "physical_readout" not in parameters or validate_physical_readout(self, channel):
                bound_channels.add(channel)
        if bound_channels != set(self.source_nuisance):
            raise ValueError("Source nuisance must be separate and complete for every bound readout.")
        for channel in bound_channels:
            parameters, nuisance = self.resolve_readout(channel)
            distribution = observation_distribution(parameters, nuisance)
            if distribution.batch_shape[:2] != (b, j):
                raise ValueError("Readout support is not particle aligned.")
        return self

    def resolve_readout(self, channel: str) -> tuple[dict, torch.Tensor]:
        parameters = self.readout_parameters[channel]
        if "physical_readout" in parameters:
            if not validate_physical_readout(self, channel):
                raise ValueError("local membrane: error model is unbound; conditioning is withheld")
            parameters = ResolvedMembraneReadout(parameters, self.local_membrane)
        return parameters, self.source_nuisance[channel]

    def validate_observation(self, observation: RetainedObservation, assays: dict) -> tuple[dict, torch.Tensor]:
        _validate_observation_binding(self, observation)
        parameters, nuisance = self.resolve_readout(observation.channel_uid)
        observation.validate(parameters, assays)
        return parameters, nuisance

    def local_binding_payload(self) -> dict | None:
        if self.local_membrane is None:
            return None
        channels = {key: value for key, value in self.readout_parameters.items() if "physical_readout" in value}
        return {"snapshot": self.local_membrane.to_payload(), "readout_parameters": membrane_payload(channels),
                "source_nuisance": membrane_payload({key: self.source_nuisance[key] for key in channels if key in self.source_nuisance})}

    def local_membrane_status(self) -> dict | None:
        if self.local_membrane is None:
            return None
        channels = {key: "source_declared_unverified" if physical_readout_bound(value) else "unbound"
                    for key, value in self.readout_parameters.items() if "physical_readout" in value}
        return {"coordinate_status": "current_snapshot_bound", "likelihood_status_by_channel": channels,
                "likelihood_status": "source_declared_unverified" if channels and all(value != "unbound" for value in channels.values()) else "unbound",
                "electrical_evolution_status": "unbound", "calibration_status": "unverified"}

    @property
    def particle_count(self) -> int:
        return self.prior_weights.shape[1]

    def to_payload(self) -> dict:
        def encode(value):
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            return value
        return {key: (value.to_payload() if key == "local_membrane" else encode(value))
                for key, value in vars(self).items() if key not in {"local_membrane", "quantity_profile", "source_family_uid", "physical_parameter_hypotheses"} or value is not None}

    @classmethod
    def from_payload(cls, value: dict, device="cpu") -> "JointParticleSupport":
        payload = dict(value)
        if payload.get("physical_parameter_hypotheses") is not None:
            payload["physical_parameter_hypotheses"] = candidate_from_payload(payload["physical_parameter_hypotheses"], device)
        for key in ("initial_field", "physical_parameters", "prior_weights"):
            payload[key] = torch.as_tensor(payload[key], dtype=torch.float32, device=device)
        payload["factors"] = {key: torch.as_tensor(item, dtype=torch.float32, device=device) for key, item in payload["factors"].items()}
        payload["source_nuisance"] = {key: torch.as_tensor(item, dtype=torch.float32, device=device) for key, item in payload["source_nuisance"].items()}
        payload["readout_parameters"] = parse_readouts(payload["readout_parameters"], device)
        if payload.get("local_membrane") is not None:
            payload["local_membrane"] = LocalMembraneSnapshot.from_payload(payload["local_membrane"], device)
        for key in ("sample_uids", "species_units", "axes"):
            if key in payload:
                payload[key] = tuple(payload[key])
        for key in ("particle_uids", "regime_ids"):
            payload[key] = tuple(tuple(row) for row in payload[key])
        return cls(**payload).validate()


def _posterior_uid(support: JointParticleSupport, observation_uids: tuple[str, ...], parent: str | None,
                   log_weights: torch.Tensor, context: torch.Tensor) -> str:
    payload = [support.support_uid, support.source_snapshot_ref, support.sample_uids, observation_uids, parent,
               _log_weight_payload(log_weights),
               hashlib.sha256(context.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()]
    if support.physical_parameter_hypotheses is not None:
        payload.append(candidate_payload(support.physical_parameter_hypotheses))
    if support.local_membrane is not None:
        payload.append(support.local_binding_payload())
    return "joint_posterior_" + hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class JointWorldStatePosterior:
    support: JointParticleSupport
    log_weights: torch.Tensor
    particle_context: torch.Tensor
    observation_features: torch.Tensor
    observation_uids: tuple[str, ...]
    sequence_number: int
    posterior_uid: str
    parent_posterior_uid: str | None
    log_likelihood: torch.Tensor
    status: str
    weights: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        normalizer = torch.logsumexp(self.log_weights, dim=-1)
        if (self.log_weights.shape != self.support.prior_weights.shape
                or torch.isnan(self.log_weights).any() or torch.isposinf(self.log_weights).any()
                or not torch.allclose(normalizer, torch.zeros_like(normalizer),
                                      atol=4 * torch.finfo(self.log_weights.dtype).eps)):
            raise ValueError("Joint posterior log weights must be normalized on the shared support.")
        # Keep this derived tensor for existing sampling/loss consumers and their
        # gradient hooks. It never supplies the state of a later update.
        object.__setattr__(self, "weights", self.log_weights.exp().to(self.support.prior_weights.dtype))
        if self.particle_context.shape[:4] != self.support.initial_field.shape[:4] or not torch.isfinite(self.particle_context).all():
            raise ValueError("Joint posterior context is not particle/grid aligned.")

    def assert_current_binding(self) -> None:
        if self.support.local_membrane is not None or self.support.physical_parameter_hypotheses is not None:
            self.support.validate()
            if self.posterior_uid != _posterior_uid(self.support, self.observation_uids, self.parent_posterior_uid,
                                                    self.log_weights, self.particle_context):
                raise ValueError("joint support/local membrane: changed support cannot reuse the consumed posterior identity")

    @property
    def ess(self) -> torch.Tensor:
        return self.weights.square().sum(-1).reciprocal()

    def categorical_kl(self) -> torch.Tensor:
        prior_log_weights = _normalized_log_weights(self.support.prior_weights.double().log())
        finite_mass = torch.isfinite(self.log_weights)
        if (finite_mass & ~torch.isfinite(prior_log_weights)).any():
            raise ValueError("Variational mass lies outside the shared prior support.")
        # Zero mass has exactly zero contribution; mask before subtracting infinities.
        log_ratio = (torch.where(finite_mass, self.log_weights, 0.)
                     - torch.where(finite_mass, prior_log_weights, 0.))
        return (self.log_weights.exp() * log_ratio).sum(-1).to(self.weights.dtype)

    def assimilate(self, observation: RetainedObservation, *, parent_posterior_uid: str, assays: dict[str, str]) -> "JointWorldStatePosterior":
        self.assert_current_binding()
        if parent_posterior_uid != self.posterior_uid:
            raise ValueError("Out-of-order posterior lineage.")
        if observation.observation_uid in self.observation_uids or observation.sequence_number != self.sequence_number + 1:
            raise ValueError("Duplicate or out-of-order observation assimilation.")
        parameters, nuisance = self.support.validate_observation(observation, assays)
        likelihood = retained_log_likelihood(observation, parameters, nuisance, assays)
        log_weights = _normalized_log_weights(self.log_weights + likelihood.double())
        uids = (*self.observation_uids, observation.observation_uid)
        return JointWorldStatePosterior(self.support, log_weights, self.particle_context, self.observation_features, uids,
            observation.sequence_number, _posterior_uid(self.support, uids, self.posterior_uid, log_weights, self.particle_context),
            self.posterior_uid, self.log_likelihood + likelihood, "observation_updated")

    def resample(self, count: int, generator: torch.Generator) -> dict:
        self.assert_current_binding()
        if count < 1:
            raise ValueError("Whole-particle resampling count must be positive.")
        indices = torch.multinomial(self.weights, count, replacement=True, generator=generator)
        def select(value):
            index = indices.reshape(*indices.shape, *([1] * (value.ndim - 2))).expand(*indices.shape, *value.shape[2:])
            return value.gather(1, index)
        result = {"indices": indices, "factors": {name: select(value) for name, value in self.support.factors.items()},
                "initial_field": select(self.support.initial_field), "physical_parameters": select(self.support.physical_parameters),
                "particle_context": select(self.particle_context),
                "particle_uids": tuple(tuple(self.support.particle_uids[b][i] for i in row.tolist()) for b, row in enumerate(indices)),
                "regime_ids": tuple(tuple(self.support.regime_ids[b][i] for i in row.tolist()) for b, row in enumerate(indices)),
                "posterior_uid": self.posterior_uid}
        if self.support.physical_parameter_hypotheses is not None:
            named = candidate_payload(self.support.physical_parameter_hypotheses)
            named["values"] = select(self.support.physical_parameter_hypotheses["values"]).detach().cpu().tolist()
            named["particle_uids"] = [list(row) for row in result["particle_uids"]]
            result["physical_parameter_hypotheses"] = named
        if self.support.local_membrane is not None:
            result["local_membrane"] = self.support.local_membrane.resampled_payload(indices)
            result["readout_parameters"] = {channel: {key: select(value) if torch.is_tensor(value) else value
                                                       for key, value in parameters.items()}
                                            for channel, parameters in self.support.readout_parameters.items()}
            result["source_nuisance"] = {channel: select(value) for channel, value in self.support.source_nuisance.items()}
        return result

    def to_payload(self) -> dict:
        self.assert_current_binding()
        result = {"schema_version": JOINT_POSTERIOR_SCHEMA, "posterior_uid": self.posterior_uid,
                "parent_posterior_uid": self.parent_posterior_uid, "support_uid": self.support.support_uid,
                "sample_uids": list(self.support.sample_uids), "particle_uids": [list(row) for row in self.support.particle_uids],
                "log_weights": _log_weight_payload(self.log_weights),
                "weights": self.weights.detach().cpu().tolist(), "ess": self.ess.detach().cpu().tolist(),
                "particle_count": self.support.particle_count, "observation_uids": list(self.observation_uids),
                "sequence_number": self.sequence_number, "status": self.status,
                "factors": {name: value.detach().cpu().tolist() for name, value in self.support.factors.items()},
                "regime_ids": [list(row) for row in self.support.regime_ids],
                "initial_field": self.support.initial_field.detach().cpu().tolist(),
                "particle_context": self.particle_context.detach().cpu().tolist(),
                "source_kind": self.support.source_kind, "calibration_status": "unverified"}
        if self.support.physical_parameter_hypotheses is not None:
            result["physical_parameter_hypotheses"] = candidate_payload(self.support.physical_parameter_hypotheses)
        if self.support.local_membrane is not None:
            result["local_membrane"] = self.support.local_binding_payload()
            result["local_membrane_status"] = self.support.local_membrane_status()
        return result


def _log_weight_payload(log_weights: torch.Tensor) -> list[list[float | str]]:
    return [["-inf" if value == -float("inf") else value for value in row]
            for row in log_weights.detach().cpu().tolist()]


def _normalized_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    if torch.isnan(log_weights).any() or torch.isposinf(log_weights).any():
        raise ValueError("Joint log weights cannot contain NaN or positive infinity.")
    if not torch.isfinite(log_weights).any(-1).all():
        raise ValueError("All joint particles are impossible under retained evidence.")
    # log_softmax subtracts the maximum before normalizing. Keeping double log
    # state also preserves finite admission logs beyond the float32 range.
    return torch.log_softmax(log_weights.double(), dim=-1)


def _validate_observation_binding(support: JointParticleSupport, observation: RetainedObservation) -> None:
    if (observation.source_uid != support.source_uid or observation.source_snapshot_ref != support.source_snapshot_ref
            or observation.sample_uids != support.sample_uids or observation.role != support.role
            or observation.channel_uid not in support.readout_parameters):
        raise ValueError("Observation source/sample/role/support binding mismatch.")


def joint_belief_objective(posterior, config: dict, *, retained_feature_count: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    policy = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    coefficients = policy.get("joint_objectives", {})
    names = ("observation_likelihood_weight", "categorical_kl_weight")
    if any(isinstance(coefficients.get(key), bool) or not isinstance(coefficients.get(key), (int, float))
           or not math.isfinite(coefficients[key]) or coefficients[key] < 0 for key in names):
        raise ValueError("Joint likelihood/KL objective coefficients must be explicitly configured; scientific defaults are unbound.")
    retained_likelihood = torch.where(posterior.weights > 0, posterior.log_likelihood, torch.zeros_like(posterior.log_likelihood))
    expected_nll = -(posterior.weights * retained_likelihood).sum(-1)
    if coefficients.get("reduction") == "retained_features_mean":
        if retained_feature_count is None or retained_feature_count.shape != expected_nll.shape or (retained_feature_count <= 0).any():
            raise ValueError("Scientific belief loss requires positive retained feature counts")
        expected_nll = expected_nll / retained_feature_count
    value = (float(coefficients[names[0]]) * expected_nll + float(coefficients[names[1]]) * posterior.categorical_kl()).mean()
    return value, {"retained_observation_likelihood": "provided", "categorical_kl": "same_prior_support",
                   "held_modality": "not_provided", "ontology_alignment": "not_provided",
                   "source_transfer": "not_provided", "calibration": "unverified"}

