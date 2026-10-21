"""Observation-only state probes on the admitted complete physical prior."""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import torch
from torch import nn

from Libs.exp.training.pretrain_microenvironment_world_model import joint_inference_modules
from Libs.models.modules.evidence import encode_retained_observations
from Libs.models.modules.world_state import FACTOR_NAMES
from Libs.utils.data.joint_observation import retained_encoder_inputs, retained_log_likelihood
from Libs.utils.data.native_diffusion_data import parse_joint_source


def observation_view(observations, indices):
    """Remove values and both retention masks before an encoder or likelihood sees them."""
    visible = set(indices)
    result = []
    for observation in observations:
        keep = torch.tensor([index in visible for index in observation.encoder_indices],
                            dtype=torch.bool, device=observation.values.device)
        result.append(replace(observation,
            values=torch.where(observation.observed & keep, observation.values,
                               torch.zeros_like(observation.values)),
            observed=observation.observed & keep,
            left_censored=observation.left_censored & keep))
    return tuple(result)


def held_readout_log_likelihood(support, observations, log_weights, indices, assays):
    held = observation_view(observations, indices)
    likelihood = torch.zeros_like(log_weights)
    counts = torch.zeros(log_weights.shape[0], device=log_weights.device, dtype=torch.long)
    for observation in held:
        parameters, nuisance = support.validate_observation(observation, assays)
        likelihood = likelihood + retained_log_likelihood(observation, parameters, nuisance, assays)
        counts += (observation.observed | observation.left_censored).sum(-1)
    if not (counts > 0).all():
        raise ValueError("State probe has no observed held readout in the declared block")
    return torch.logsumexp(log_weights.double() + likelihood.double(), -1) / counts, counts


class LinearCandidateScore(nn.Module):
    """Bilinear observation/candidate correction with no hidden nonlinear encoder."""
    def __init__(self, latent_width, particle_width):
        super().__init__()
        self.latent_width = latent_width
        self.candidate = nn.Linear(particle_width, latent_width, bias=False)

    def forward(self, value):
        observation, particle = value.split(
            (self.latent_width, value.shape[-1] - self.latent_width), -1)
        return (observation * self.candidate(particle)).sum(-1, keepdim=True) / math.sqrt(self.latent_width)


class LinearObservationMap(nn.Module):
    def __init__(self, width, latent_width, components=None):
        super().__init__()
        self.width = width
        self.components = components
        feature_width = 3 * width
        if components is not None:
            if not 1 <= components <= feature_width:
                raise ValueError("PCA dimension must fit the observation and retention representation")
            self.register_buffer("center", torch.zeros(feature_width))
            self.register_buffer("basis", torch.zeros(feature_width, components))
            self.register_buffer("fitted", torch.tensor(False))
        self.projection = nn.Linear(components or feature_width, latent_width)

    def forward(self, values, masks):
        features = torch.cat((values, masks), -1)
        if self.components is not None:
            if not bool(self.fitted):
                raise ValueError("PCA state probe requires a training-only fitted transform")
            features = (features - self.center) @ self.basis
        return self.projection(features)


class NativeStateProbe(nn.Module):
    """Fine-tune observation inference only; no trajectory or oracle target is consumed."""
    def __init__(self, method, belief, config):
        super().__init__()
        policy = config["native_state_comparison"]
        if method not in policy["fitted_methods"]:
            raise ValueError("Unknown fitted native state probe")
        self.method = method
        self.width = belief.observation_feature_width
        self.blocks = tuple(tuple(block) for block in policy["readout_blocks"])
        flat = [index for block in self.blocks for index in block]
        if sorted(flat) != list(range(self.width)) or len(set(flat)) != len(flat):
            raise ValueError("State readout blocks must partition the complete registered observation width")
        self.unimodal_block = int(policy["unimodal_block"])
        if not 0 <= self.unimodal_block < len(self.blocks):
            raise ValueError("Unimodal state probe selects an unregistered readout block")
        self.assays = belief.world_state_posterior.joint_config.get("registered_occupancy_assays", {})
        self.normalization = copy.deepcopy(belief.field_normalization)
        self.normalization.requires_grad_(False)
        branches = len(self.blocks) if method == "state_late_fusion" else 1
        self.encoders = nn.ModuleList()
        self.masks = nn.ModuleList()
        self.posteriors = nn.ModuleList()
        self.linear_maps = nn.ModuleList()
        for _ in range(branches):
            if method == "state_scratch":
                hidden, species = belief.world_state_posterior.joint_field_projection.weight.shape
                (encoder, mask, posterior), width = joint_inference_modules(config, hidden, species)
                if width != self.width:
                    raise ValueError("Scratch observation width differs from the selected common checkpoint")
            else:
                encoder = copy.deepcopy(belief.observation_encoder)
                mask = copy.deepcopy(belief.retained_mask_projection)
                posterior = copy.deepcopy(belief.world_state_posterior)
            posterior.requires_grad_(False)
            encoder.requires_grad_(True)
            mask.requires_grad_(True)
            posterior.joint_score.requires_grad_(True)
            if method in {"state_pca", "state_linear"}:
                latent = posterior.factor_dimensions["z_tx"]
                components = int(policy["pca_components"]) if method == "state_pca" else None
                self.linear_maps.append(LinearObservationMap(self.width, latent, components))
                posterior.joint_score = LinearCandidateScore(
                    latent, sum(posterior.factor_dimensions.values()) + 4)
                encoder = nn.Identity()
                mask = nn.Identity()
            self.encoders.append(encoder)
            self.masks.append(mask)
            self.posteriors.append(posterior)
        self.objectives = copy.deepcopy(policy["objectives"])
        self.pca_fit_count = 0

    def _parsed(self, source):
        return parse_joint_source(source, next(self.parameters()).device)

    def infer(self, source, *, visible_indices=None):
        support, observations = self._parsed(source)
        allowed = set(range(self.width)) if visible_indices is None else set(visible_indices)
        if not allowed.issubset(range(self.width)):
            raise ValueError("State view selects an unregistered observation index")
        if self.method == "state_unimodal":
            allowed &= set(self.blocks[self.unimodal_block])
        branch_logs, active = [], []
        for i, posterior in enumerate(self.posteriors):
            indices = allowed & set(self.blocks[i]) if self.method == "state_late_fusion" else allowed
            view = observation_view(observations, indices)
            values, masks = retained_encoder_inputs(view, self.width, support=support, assays=self.assays)
            if self.linear_maps:
                encoded = self.linear_maps[i](values, masks)
            else:
                encoded = encode_retained_observations(self.encoders[i], self.masks[i], values, masks)
            inferred = posterior.infer_joint(encoded, values, support, view, self.normalization)
            branch_logs.append(inferred.log_weights)
            active.append(masks.sum(-1) > 0)
        if len(branch_logs) == 1:
            return support, observations, branch_logs[0]
        # Late fusion combines independently encoded unimodal posterior laws only.
        logs = torch.stack(branch_logs, 1)
        admitted = torch.stack(active, 1)
        count = admitted.sum(1)
        masked = logs.masked_fill(~admitted[:, :, None], -torch.inf)
        mixture = torch.logsumexp(masked, 1) - count.clamp_min(1).log()[:, None]
        prior = support.prior_weights.double().log()
        prior -= torch.logsumexp(prior, -1, keepdim=True)
        return support, observations, torch.where(count[:, None] > 0, mixture, prior)

    def held_scores(self, source):
        scores = []
        for block in self.blocks:
            visible = set(range(self.width)) - set(block)
            support, observations, logs = self.infer(source, visible_indices=visible)
            if not any((observation.observed | observation.left_censored)[:, [
                    i for i, index in enumerate(observation.encoder_indices) if index in block
                    ]].any() for observation in observations):
                continue
            score, _ = held_readout_log_likelihood(support, observations, logs, block, self.assays)
            scores.append(score.mean())
        if not scores:
            raise ValueError("State comparison has no held readout evidence")
        return torch.stack(scores)

    def loss(self, source):
        support, observations, logs = self.infer(source)
        visible = self.blocks[self.unimodal_block] if self.method == "state_unimodal" else range(self.width)
        retained, _ = held_readout_log_likelihood(support, observations, logs, visible, self.assays)
        weights = logs.exp()
        prior = support.prior_weights.double().log()
        prior -= torch.logsumexp(prior, -1, keepdim=True)
        finite = torch.isfinite(logs)
        kl = (weights * (torch.where(finite, logs, 0.) - torch.where(finite, prior, 0.))).sum(-1).mean()
        held = -self.held_scores(source).mean()
        terms = {"retained_readout": -retained.mean(), "held_readout": held, "categorical_kl": kl}
        return sum(float(self.objectives[key]) * value for key, value in terms.items())

    @torch.no_grad()
    def fit_pca(self, dataset):
        if self.method != "state_pca":
            return
        transform = self.linear_maps[0]
        if bool(transform.fitted):
            raise ValueError("A fitted PCA transform cannot be refitted")
        n = 0
        total = torch.zeros(3 * self.width, dtype=torch.float64)
        square = torch.zeros(3 * self.width, 3 * self.width, dtype=torch.float64)
        for index in range(len(dataset)):
            source = dataset[index]["native_joint_group"]["joint_source"]
            support, observations = self._parsed(source)
            if any(observation.role != "train" for observation in observations):
                raise ValueError("PCA may fit only admitted training observations")
            values, masks = retained_encoder_inputs(observations, self.width, support=support, assays=self.assays)
            features = torch.cat((values, masks), -1).double().cpu()
            n += features.shape[0]
            total += features.sum(0)
            square += features.T @ features
        if n < 2:
            raise ValueError("PCA needs at least two independent training observations")
        center = total / n
        covariance = (square - n * center[:, None] * center[None, :]) / (n - 1)
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        if not torch.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0:
            raise ValueError("PCA training observations have no finite variation")
        basis = vectors[:, -transform.components:].flip(-1)
        # Fix eigenvector signs so serialized transforms do not change by convention.
        pivots = basis.abs().argmax(0)
        signs = torch.sign(basis[pivots, torch.arange(basis.shape[1])])
        transform.center.copy_(center.to(transform.center))
        transform.basis.copy_((basis * signs).to(transform.basis))
        transform.fitted.fill_(True)
        self.pca_fit_count = n

    @torch.no_grad()
    def posterior_factor_summary(self, source):
        support, _, logs = self.infer(source)
        weights = logs.exp()
        return {name: (weights[:, :, None] * support.factors[name]).sum(1)
                for name in FACTOR_NAMES}
