"""Source-bound retained observation laws for finite joint-particle inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import torch

from Libs.utils.data.digital_twin_canonical_state import CANONICAL_AXES
from Libs.utils.data.local_membrane import MembraneAcquisition, validate_physical_observation

LIKELIHOOD_NAMES = ("negative_binomial", "log_normal", "gaussian", "categorical")


@dataclass(frozen=True)
class RetainedObservation:
    observation_uid: str
    source_uid: str
    source_snapshot_ref: str
    sample_uids: tuple[str, ...]
    role: str
    channel_uid: str
    law: str
    units: str
    values: torch.Tensor
    observed: torch.Tensor
    left_censored: torch.Tensor
    encoder_indices: tuple[int, ...]
    sequence_number: int
    library_offset: torch.Tensor | None = None
    detection_limit: torch.Tensor | None = None
    normalization_ref: str | None = None
    modality: str = "state"
    assay_uid: str | None = None
    informative_admission: bool = False
    admission_log_probability: torch.Tensor | None = None
    physical_acquisition: MembraneAcquisition | None = None

    def validate(self, parameters: Mapping[str, object], assays: Mapping[str, str]) -> None:
        validate_physical_observation(self, parameters)
        if not all((self.observation_uid, self.source_uid, self.source_snapshot_ref, self.channel_uid, self.units)):
            raise ValueError("Observation requires source/snapshot/channel/unit identity.")
        if self.role not in {"train", "development", "validation", "calibration", "inference"}:
            raise ValueError("Observation source role is not admitted.")
        if self.law not in LIKELIHOOD_NAMES or parameters.get("law") != self.law or parameters.get("units") != self.units:
            raise ValueError("Observation likelihood/unit binding mismatch.")
        if self.modality == "occupancy" and (not self.assay_uid or assays.get(self.assay_uid) != self.law):
            raise ValueError("Direct occupancy requires an explicitly registered assay law.")
        if self.values.ndim != 2 or min(self.values.shape) < 1 or len(self.sample_uids) != self.values.shape[0]:
            raise ValueError("Observation values require nonempty sample/feature axes.")
        if len(set(self.sample_uids)) != len(self.sample_uids) or len(self.encoder_indices) != self.values.shape[1]:
            raise ValueError("Observation sample/encoder identity mismatch.")
        if self.sequence_number < 0 or any(index < 0 for index in self.encoder_indices):
            raise ValueError("Observation sequence and encoder indices must be nonnegative.")
        for mask in (self.observed, self.left_censored):
            if mask.shape != self.values.shape or mask.dtype != torch.bool:
                raise ValueError("Observation masks must be aligned booleans.")
        if (self.observed & self.left_censored).any():
            raise ValueError("A value cannot be both observed and censored.")
        if not torch.isfinite(self.values[self.observed]).all():
            raise ValueError("Retained values must be finite; rejected values are not zero observations.")
        if self.informative_admission and self.admission_log_probability is None:
            raise ValueError("Biological update withheld: unmodeled informative admission event.")
        if self.left_censored.any() and self.law != "log_normal":
            raise ValueError("Censoring law is unimplemented for this channel.")
        if self.law == "negative_binomial":
            if self.library_offset is None or self.library_offset.shape != self.values.shape:
                raise ValueError("Counts require an explicit library offset.")
            if not torch.isfinite(self.library_offset).all() or not (self.library_offset > 0).all():
                raise ValueError("Library offsets must be finite and positive.")
            value = self.values[self.observed]
            if (value < 0).any() or not torch.equal(value, value.floor()):
                raise ValueError("Count observations must be nonnegative integers.")
        if self.law == "log_normal":
            if (self.values[self.observed] <= 0).any():
                raise ValueError("Concentrations must be positive.")
            if self.detection_limit is None or self.detection_limit.shape != self.values.shape:
                raise ValueError("Log-normal channels require explicit detection limits.")
            if not torch.isfinite(self.detection_limit).all() or not (self.detection_limit > 0).all():
                raise ValueError("Detection limits must be finite and positive.")
        if self.law == "gaussian" and not self.normalization_ref:
            raise ValueError("Scaled morphology/electrophysiology requires a normalization binding.")


def observation_distribution(parameters: Mapping[str, object], nuisance: torch.Tensor):
    law = parameters["law"]
    if not torch.isfinite(nuisance).all():
        raise ValueError("Source nuisance must be finite and supplied separately.")
    if law == "categorical":
        probabilities = parameters["probabilities"]
        if (probabilities.ndim != 4 or probabilities.shape != nuisance.shape
                or not torch.isfinite(probabilities).all() or not (probabilities >= 0).all()
                or not (probabilities.sum(-1) > 0).all()):
            raise ValueError("Categorical support requires finite nonnegative class probabilities.")
        return torch.distributions.Categorical(logits=probabilities.log() + nuisance)
    mean = parameters["mean"]
    if mean.ndim != 3 or mean.shape != nuisance.shape or not torch.isfinite(mean).all():
        raise ValueError("Particle readout means require finite sample/particle/feature axes.")
    if not torch.isfinite(nuisance).all():
        raise ValueError("Source nuisance must be finite and supplied separately.")
    if law == "negative_binomial":
        dispersion = parameters["dispersion"]
        if dispersion.shape != mean.shape or not torch.isfinite(dispersion).all() or not (dispersion > 0).all() or not (mean > 0).all():
            raise ValueError("Negative-binomial mean/dispersion must be positive and aligned.")
        return torch.distributions.NegativeBinomial(total_count=dispersion, logits=(mean * nuisance.exp()).log() - dispersion.log())
    variance = parameters["variance"]
    if variance.shape != mean.shape or not torch.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError("Heteroscedastic variance must be finite, positive and particle aligned.")
    if law == "gaussian":
        return torch.distributions.Normal(mean + nuisance, variance.sqrt())
    if law == "log_normal":
        if not (mean > 0).all():
            raise ValueError("Log-normal concentrations must be positive.")
        return torch.distributions.LogNormal(mean.log() + nuisance, variance.sqrt())
    raise ValueError(f"Unimplemented likelihood law: {law}")


def retained_log_likelihood(observation: RetainedObservation, parameters: Mapping[str, object],
                            nuisance: torch.Tensor, assays: Mapping[str, str]) -> torch.Tensor:
    observation.validate(parameters, assays)
    payload = dict(parameters)
    if observation.law == "negative_binomial":
        payload["mean"] = parameters["mean"] * observation.library_offset.unsqueeze(1)
    distribution = observation_distribution(payload, nuisance)
    # Missing values are marginalized. The placeholders only avoid invalid
    # distribution arithmetic and never enter a retained likelihood term.
    safe = torch.where(observation.observed, observation.values, torch.ones_like(observation.values))
    if observation.law == "categorical":
        values = observation.values[observation.observed]
        if not torch.equal(values, values.floor()) or (values < 0).any() or (values >= parameters["probabilities"].shape[-1]).any():
            raise ValueError("Categorical observation is outside its declared support.")
        safe = torch.where(observation.observed, observation.values, torch.zeros_like(observation.values)).long()
    log_probability = distribution.log_prob(safe.unsqueeze(1))
    terms = torch.where(observation.observed.unsqueeze(1), log_probability, torch.zeros_like(log_probability))
    if observation.law == "log_normal":
        limit = observation.detection_limit.unsqueeze(1)
        z = (limit.log() - distribution.loc) / distribution.scale
        terms = terms + torch.where(observation.left_censored.unsqueeze(1), torch.special.log_ndtr(z), torch.zeros_like(z))
    result = terms.sum(-1)
    if observation.informative_admission:
        admission = observation.admission_log_probability
        if admission.shape != result.shape or torch.isnan(admission).any() or (admission > 0).any():
            raise ValueError("Admission-event log probabilities must align to whole particles.")
        result = result + admission
    if torch.isnan(result).any() or torch.isposinf(result).any():
        raise ValueError("Invalid observation likelihood.")
    return result


def retained_encoder_inputs(observations: tuple[RetainedObservation, ...], width: int, *,
                            support=None, assays: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if support is not None:
        support.validate()
        seen = set()
        for index, observation in enumerate(observations):
            if observation.observation_uid in seen or observation.sequence_number != index:
                raise ValueError("Initial observations require unique ordered identities.")
            support.validate_observation(observation, assays or {})
            seen.add(observation.observation_uid)
    elif any(observation.physical_acquisition is not None for observation in observations):
        raise ValueError("Physical observations require joint support validation before the evidence encoder.")
    if not observations:
        raise ValueError("Retained observation payload is required.")
    first = observations[0]
    values = first.values.new_zeros(len(first.sample_uids), width)
    masks = values.new_zeros(len(first.sample_uids), width * 2)
    assigned = set()
    for observation in observations:
        if observation.sample_uids != first.sample_uids or observation.source_uid != first.source_uid:
            raise ValueError("Observation encoder sample/source mismatch.")
        for feature, index in enumerate(observation.encoder_indices):
            if index >= width or index in assigned:
                raise ValueError("Observation encoder slots must be unique and source bound.")
            assigned.add(index)
            retained = observation.observed[:, feature]
            censored = observation.left_censored[:, feature]
            raw = torch.where(retained, observation.values[:, feature], torch.zeros_like(values[:, index]))
            if observation.law == "negative_binomial":
                raw = torch.log1p(raw / observation.library_offset[:, feature])
            elif observation.law == "log_normal":
                raw = torch.where(retained, raw.clamp_min(torch.finfo(raw.dtype).tiny).log(), torch.zeros_like(raw))
                raw = torch.where(censored, observation.detection_limit[:, feature].log(), raw)
            values[:, index] = raw
            masks[:, index] = retained
            masks[:, width + index] = censored
    return values, masks


def observation_to_payload(observation: RetainedObservation) -> dict:
    def encode(value):
        if torch.is_tensor(value):
            return encode(value.detach().cpu().tolist())
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    payload = {name: (value.to_payload() if name == "physical_acquisition" else encode(value))
               for name, value in vars(observation).items()
               if name != "admission_log_probability" and (name != "physical_acquisition" or value is not None)}
    admission = observation.admission_log_probability
    if admission is None:
        payload["admission_log_probability"] = None
    else:
        if admission.ndim != 2 or torch.isnan(admission).any() or (admission > 0).any():
            raise ValueError("Admission-event log probabilities must be nonpositive and particle aligned.")
        payload["admission_log_probability"] = [
            ["-inf" if value == -float("inf") else value for value in row]
            for row in admission.detach().cpu().tolist()]
    return payload


def observation_from_payload(payload: dict, device: torch.device | str = "cpu") -> RetainedObservation:
    fields = dict(payload)
    if fields.get("physical_acquisition") is not None:
        fields["physical_acquisition"] = MembraneAcquisition.from_payload(fields["physical_acquisition"], device)
    def numeric(value):
        if isinstance(value, list):
            return [numeric(item) for item in value]
        return float("nan") if value is None else value
    def boolean(value):
        if isinstance(value, list):
            return all(boolean(item) for item in value)
        return isinstance(value, bool)
    for name in ("values", "library_offset", "detection_limit"):
        if fields.get(name) is not None:
            fields[name] = torch.as_tensor(numeric(fields[name]), dtype=torch.float32, device=device)
    admission = fields.get("admission_log_probability")
    if admission is not None:
        if (not isinstance(admission, list) or not all(isinstance(row, list) for row in admission)
                or any(not ((isinstance(value, (int, float)) and not isinstance(value, bool)) or value == "-inf")
                       for row in admission for value in row)):
            raise ValueError("Admission-event log probabilities require numbers or the explicit '-inf' token, never null.")
        fields["admission_log_probability"] = torch.as_tensor(
            [[-float("inf") if value == "-inf" else value for value in row] for row in admission],
            dtype=torch.float64, device=device)
        if (torch.isnan(fields["admission_log_probability"]).any()
                or (fields["admission_log_probability"] > 0).any()):
            raise ValueError("Admission-event log probabilities cannot contain NaN, positive infinity or positive values.")
    for name in ("observed", "left_censored"):
        if not boolean(fields[name]):
            raise ValueError("Serialized observation masks must contain explicit booleans.")
        fields[name] = torch.as_tensor(fields[name], dtype=torch.bool, device=device)
    for name in ("sample_uids", "encoder_indices"):
        fields[name] = tuple(fields[name])
    return RetainedObservation(**fields)


STATE_AXES = CANONICAL_AXES
