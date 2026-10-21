"""Fixed-action Gaussian readouts of the existing anchored Brownian residual law.

Callers supply an already validated residual artifact and TRAIN normalizer.
This utility conditions one unchanged predictor/action and complete tuple history.
The separate prospective adaptive window extends the readout law across original
actions and predictor updates. It is scientifically UNQUALIFIED and has no live
adapter or calibration transfer. No biological tuple is redrawn; mixture weights
remain caller-owned.
"""
from __future__ import annotations

from Libs.utils.data.native_actuation import actuation_payload

from dataclasses import dataclass, replace
import json
import math
from typing import TYPE_CHECKING

import torch

from Libs.utils.data.native_actuation import (
    NativeActuation, NativeActuationRequest, NativeExecutedPrefix, compose_actuation, executed_prefix,
)
from Libs.utils.model.native_arrived_observation import (
    NativeAcquisition, NativeArrivedObservation, NativeCNSReadout, NativePredictedReadout,
    _validate_arrived_payload, validate_native_observation, _admit_native_observation,
)
from Libs.utils.model.native_transolver_residual import CONTROLLED_LAW, minimize_log_scales
from Libs.utils.data.native_observable_profile import original_query_time_hours

if TYPE_CHECKING:
    from Libs.utils.data.native_field_normalization import NativeFieldNormalization


@dataclass(frozen=True)
class FixedReadoutReference:
    readout: NativeCNSReadout
    support_uid: str
    origin_posterior_uid: str
    sample_uids: tuple[str, ...]
    particle_uids: tuple[tuple[str, ...], ...]
    executed_regimen_uid: str
    predictor_binding: str  # Existing predictor metadata, copied as canonical JSON.
    normalization_id: str
    residual_artifact_ref: str
    residual_scales: tuple[float, ...]
    inverse_normalizer_scales: tuple[float, ...]
    reference_time_hours: float
    source_times: tuple[float, ...]
    time_hours: tuple[float, ...]
    source_time_unit: str
    initial_filtering_revision: int

    @classmethod
    def from_validated(cls, *, readout: NativeCNSReadout, normalization: NativeFieldNormalization,
                       residual_artifact: dict, residual_artifact_ref: str, predictor_binding: dict,
                       support_uid: str, origin_posterior_uid: str, sample_uids: tuple[str, ...],
                       particle_uids: tuple[tuple[str, ...], ...], executed_regimen_uid: str,
                       source_times: tuple[float, ...], time_hours: tuple[float, ...],
                       source_time_unit: str, initial_filtering_revision: int = 0) -> FixedReadoutReference:
        """Bind validated existing inputs without reading their files or fitting scales."""
        readout.validate()
        if (residual_artifact.get("law") != CONTROLLED_LAW
                or residual_artifact.get("schema_version") != "native_comparator_residual.v1"
                or residual_artifact.get("status") != "FIT_COMPLETE"
                or residual_artifact.get("calibration_check_complete") is not True
                or residual_artifact.get("binding") != predictor_binding
                or not normalization.active
                or predictor_binding.get("normalization_id") != normalization.identity):
            raise ValueError("Validated controlled residual and TRAIN-normalizer predictor binding required")
        result = cls(readout, support_uid, origin_posterior_uid, sample_uids, particle_uids,
                     executed_regimen_uid, json.dumps(predictor_binding, sort_keys=True, allow_nan=False),
                     normalization.identity, residual_artifact_ref,
                     tuple(residual_artifact["scales"]), tuple(normalization.scale.detach().cpu().tolist()),
                     residual_artifact["reference_time_hours"], source_times, time_hours,
                     source_time_unit, initial_filtering_revision)
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.readout, NativeCNSReadout):
            raise ValueError("Fixed readout requires the existing native readout type")
        self.readout.validate()
        identities = (self.readout.channel_uid, self.readout.assay_uid, self.readout.source_uid,
                      self.readout.source_snapshot_ref, self.readout.family_uid, self.readout.normalization_ref,
                      *self.readout.region_ids, *self.readout.compartment_ids, *self.readout.species_ids)
        noise = self.readout.standard_deviation * self.readout.standard_deviation
        if (self.readout.units != "source_reference_standardized"
                or self.readout.role not in {"train", "development", "validation", "calibration", "inference"}
                or not all(isinstance(value, str) and value for value in identities)
                or any(len(set(axis)) != len(axis) for axis in
                       (self.readout.region_ids, self.readout.compartment_ids, self.readout.species_ids))
                or not math.isfinite(noise) or noise <= 0):
            raise ValueError("Fixed readout requires bound native metadata and positive finite sensor variance")
        species = len(self.readout.species_ids)
        if (species != 26 or len(self.residual_scales) != species
                or len(self.inverse_normalizer_scales) != species
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in (*self.residual_scales, *self.inverse_normalizer_scales, self.reference_time_hours))
                or self.source_time_unit not in {"ms", "hour"}
                or len(self.source_times) != len(self.time_hours) or len(self.time_hours) < 2
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                       for value in (*self.source_times, *self.time_hours))
                or self.source_times[0] != 0 or self.time_hours[0] != 0
                or any(b <= a for a, b in zip(self.source_times, self.source_times[1:]))
                or any(b <= a for a, b in zip(self.time_hours, self.time_hours[1:]))
                or len(self.sample_uids) != 1 or len(self.particle_uids) != 1
                or len(self.particle_uids[0]) != 32 or len(set(self.particle_uids[0])) != 32
                or not all(isinstance(value, str) and value for value in (*self.sample_uids, *self.particle_uids[0],
                       self.support_uid, self.origin_posterior_uid, self.executed_regimen_uid,
                       self.normalization_id, self.residual_artifact_ref, self.predictor_binding))
                or type(self.initial_filtering_revision) is not int or self.initial_filtering_revision < 0):
            raise ValueError("Fixed readout requires original clocks, positive bound scales and all32 tuple identities")
        binding = json.loads(self.predictor_binding)
        json.dumps(binding, allow_nan=False)
        if not isinstance(binding, dict) or binding.get("normalization_id") != self.normalization_id:
            raise ValueError("Fixed predictor and normalizer identity differ")


@dataclass(frozen=True)
class FixedReadoutRecord:
    reference: FixedReadoutReference
    arrived: NativeArrivedObservation
    mean: torch.Tensor  # 32,6; frozen fixed-reference readout mean at this acquisition.


@dataclass(frozen=True)
class FixedReadoutQuery:
    reference: FixedReadoutReference
    query_index: int
    observed: tuple[bool, ...]
    mean: torch.Tensor  # 32,6, in the physical selector/scaled/mixed readout coordinates.
    values: torch.Tensor | None = None  # 6; only selected coordinates enter a density.


@dataclass(frozen=True)
class ConditionalReadout:
    mean: torch.Tensor  # 32,D, one conditional mean for each unchanged biological tuple.
    covariance: torch.Tensor  # D,D, conditional residual plus new independent measurement noise.
    residual_covariance: torch.Tensor  # D,D, conditional Brownian readout without new noise.
    history_log_likelihood: torch.Tensor  # 32; one tuple retained for the complete history.
    log_likelihood: torch.Tensor | None  # 32, if query values were supplied.
    coordinates: tuple[int, ...]


def _mean(value: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.shape != (32, 6) or not torch.isfinite(value).all()
            or device is not None and value.device != device):
        raise ValueError("Fixed readout means require finite floating32x6 tensors on one device")
    return value.double()


def projected_brownian_covariance(reference: FixedReadoutReference, *, device=None,
                                  trial_scales: torch.Tensor | None = None, _allow_zero_boundary=False) -> torch.Tensor:
    """Readout covariance; optional trial scales keep the bound reference unchanged.

    Trial scales are a likelihood interface, not a fitted calibration. Species
    absent from the selected projection have no identifiable scale information.
    """
    reference.validate()
    readout = reference.readout
    unique = tuple(dict.fromkeys(readout.selector))
    projection = torch.zeros((6, len(unique)), dtype=torch.float64, device=device)
    matrix = torch.tensor(readout.matrix, dtype=torch.float64, device=device)
    scales = reference.residual_scales
    if trial_scales is not None:
        if (not isinstance(trial_scales, torch.Tensor) or not trial_scales.is_floating_point()
                or trial_scales.shape != (len(readout.species_ids),)
                or trial_scales.device != projection.device
                or not torch.isfinite(trial_scales).all()
                or not (trial_scales >= 0 if _allow_zero_boundary else trial_scales > 0).all()):
            raise ValueError("Trial discrepancy scales require matching-device positive finite floating26 tensors")
        scales = trial_scales.double()
    for selected, coordinate in enumerate(readout.selector):
        species = coordinate[-1]
        coefficient = (reference.inverse_normalizer_scales[species]
                       * scales[species] / readout.scales[selected])
        projection[:, unique.index(coordinate)] += matrix[:, selected] * coefficient
    covariance = projection @ projection.T
    if not torch.isfinite(covariance).all():
        raise ValueError("Projected Brownian covariance is not finite")
    return covariance


def _record(reference: FixedReadoutReference, row: FixedReadoutRecord, offset: int,
            device: torch.device) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, float]:
    if not isinstance(row, FixedReadoutRecord) or row.reference != reference:
        raise ValueError("Changed action, predictor, normalizer, readout or residual reference history")
    arrived = row.arrived
    if not isinstance(arrived, NativeArrivedObservation):
        raise ValueError("Fixed readout history requires existing accepted acquisition packets")
    acquisition = arrived.acquisition
    acquisition.validate()
    _validate_arrived_payload(arrived, acquisition)
    arrived.observation.validate({"law": "gaussian", "units": reference.readout.units}, {})
    index = acquisition.query_index
    if (index >= len(reference.time_hours) or acquisition.readout != reference.readout
            or acquisition.executed_regimen_uid != reference.executed_regimen_uid
            or acquisition.support_uid != reference.support_uid
            or acquisition.origin_posterior_uid != reference.origin_posterior_uid
            or acquisition.sample_uids != reference.sample_uids or acquisition.particle_uids != reference.particle_uids
            or acquisition.source_time_unit != reference.source_time_unit
            or acquisition.source_time != reference.source_times[index]
            or acquisition.time_hours != reference.time_hours[index]
            or acquisition.filtering_revision != reference.initial_filtering_revision + offset):
        raise ValueError("Stale or switched fixed acquisition source/clock/tuple/program binding")
    observation = arrived.observation
    if observation.values.device != device:
        raise ValueError("Fixed acquisition values and predictor means must share a device")
    coordinates = tuple(index for index, keep in enumerate(observation.observed[0].tolist()) if keep)
    if not coordinates:
        raise ValueError("Fixed acquisition must retain a measured coordinate")
    return coordinates, _mean(row.mean, device)[:, list(coordinates)], observation.values[0, list(coordinates)].double(), acquisition.time_hours


def _covariance(base: torch.Tensor, blocks: list[tuple[tuple[int, ...], float]], reference_time: float) -> torch.Tensor:
    if not blocks:
        return base.new_empty((0, 0))
    return torch.cat([torch.cat([base[list(a)][:, list(b)] * min(t, s) / reference_time
                                for b, s in blocks], dim=1) for a, t in blocks], dim=0)


def _cholesky(covariance: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(covariance).all():
        raise ValueError("Conditional covariance is not finite")
    factor, info = torch.linalg.cholesky_ex(covariance)
    if torch.any(info != 0) or not torch.isfinite(factor).all():
        raise ValueError("Conditional covariance is not positive definite; no numerical jitter is admitted")
    return factor


def _log_density(values: torch.Tensor, means: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    residual = values.unsqueeze(0) - means
    solved = torch.cholesky_solve(residual.T, factor).T
    answer = -.5 * ((residual * solved).sum(-1) + 2 * factor.diagonal().log().sum()
                    + values.numel() * math.log(2 * math.pi))
    if not torch.isfinite(answer).all():
        raise ValueError("Fixed conditional readout likelihood is not finite")
    return answer


def conditional_readout(reference: FixedReadoutReference, history: tuple[FixedReadoutRecord, ...],
                        query: FixedReadoutQuery) -> ConditionalReadout:
    """Condition all readouts jointly within each tuple, with independent sensor noise.

    Distinct acquisition UIDs at an equal original clock share the Brownian error
    and have independent measurement noise. Caller-owned mixture weights are not
    an input. Histories remain immutable and can be copied for hypothetical branches.
    """
    reference.validate()
    if not isinstance(history, tuple) or not isinstance(query, FixedReadoutQuery) or query.reference != reference:
        raise ValueError("Fixed query/history has a changed action or predictor reference")
    mean = _mean(query.mean)
    if (type(query.query_index) is not int or not 0 <= query.query_index < len(reference.time_hours)
            or len(query.observed) != 6 or any(type(value) is not bool for value in query.observed)
            or any(keep and not available for keep, available in zip(query.observed, reference.readout.observed))):
        raise ValueError("Query requires original-clock available boolean readout coordinates")
    coordinates = tuple(index for index, keep in enumerate(query.observed) if keep)
    if not coordinates:
        raise ValueError("Query must select an available readout coordinate")
    device = mean.device
    base = projected_brownian_covariance(reference, device=device)
    blocks, means, values, uids = [], [], [], []
    previous_index = 0
    for offset, row in enumerate(history):
        selected, predicted, measured, time_hours = _record(reference, row, offset, device)
        index = row.arrived.acquisition.query_index
        uid = row.arrived.observation.observation_uid
        if uid in uids or index < previous_index:
            raise ValueError("Duplicate or nonchronological fixed readout history")
        blocks.append((selected, time_hours))
        means.append(predicted)
        values.append(measured)
        uids.append(uid)
        previous_index = index
    if query.query_index < previous_index:
        raise ValueError("Desired readout cannot precede the accepted history")
    desired = (coordinates, reference.time_hours[query.query_index])
    joint = _covariance(base, [*blocks, desired], reference.reference_time_hours)
    count = sum(len(selected) for selected, _ in blocks)
    noise = reference.readout.standard_deviation * reference.readout.standard_deviation
    covariance = joint + torch.eye(len(joint), dtype=joint.dtype, device=device) * noise
    desired_mean = mean[:, list(coordinates)]
    residual_covariance = joint[count:, count:]
    history_likelihood = mean.new_zeros(32)
    if count:
        historical_mean, historical_values = torch.cat(means, dim=1), torch.cat(values)
        factor = _cholesky(covariance[:count, :count])
        cross = joint[count:, :count]
        gain = torch.cholesky_solve(cross.T, factor).T
        desired_mean = desired_mean + (historical_values.unsqueeze(0) - historical_mean) @ gain.T
        residual_covariance = residual_covariance - gain @ cross.T
        history_likelihood = _log_density(historical_values, historical_mean, factor)
    conditional_covariance = residual_covariance + torch.eye(len(coordinates), dtype=mean.dtype, device=device) * noise
    factor = _cholesky(conditional_covariance)
    likelihood = None
    if query.values is not None:
        values = query.values
        if (not isinstance(values, torch.Tensor) or not values.is_floating_point() or values.shape != (6,)
                or values.device != device or not torch.isfinite(values[list(coordinates)]).all()):
            raise ValueError("Desired acquisition values must be finite on selected coordinates")
        likelihood = _log_density(values[list(coordinates)].double(), desired_mean, factor)
    if not torch.isfinite(desired_mean).all() or not torch.isfinite(residual_covariance).all():
        raise ValueError("Fixed conditional readout result is not finite")
    return ConditionalReadout(desired_mean, conditional_covariance, residual_covariance,
                              history_likelihood, likelihood, coordinates)


ADAPTIVE_READOUT_LAW = "native_original_action_adaptive_readout_brownian.v1"


@dataclass(frozen=True)
class AdaptiveReadoutReference:
    """Prospective extension; fixed-action fit/check is only a starting scale.

    Each original action has an independent Brownian process on the absolute
    original clock. Revisits resume that process. Predictor changes affect future
    means only. The complete biological tuple is fixed for the whole history.
    """

    fixed: FixedReadoutReference
    original_actions: tuple[NativeActuation, ...]
    initial_action_uid: str
    law: str = ADAPTIVE_READOUT_LAW
    scientific_qualification: str = "UNQUALIFIED"
    scale_authority: str = "inherited_fixed_action_starting_model_not_adaptive_calibration"
    adaptive_artifact_id: str | None = None
    fitted_species_indices: tuple[int, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.fixed, FixedReadoutReference):
            raise ValueError("Adaptive readout requires a bound fixed starting reference")
        self.fixed.validate()
        fixed = self.fixed
        if (self.law != ADAPTIVE_READOUT_LAW or self.scientific_qualification != "UNQUALIFIED"
                or self.scale_authority not in {"inherited_fixed_action_starting_model_not_adaptive_calibration",
                                                "authenticated_observable_adaptive_fit"}
                or type(self.original_actions) is not tuple or not self.original_actions
                or any(type(action) is not NativeActuation for action in self.original_actions)):
            raise ValueError("Prospective adaptive readout is UNQUALIFIED with exact supported original actions")
        fitted = self.scale_authority == "authenticated_observable_adaptive_fit"
        if (fitted != bool(self.adaptive_artifact_id) or type(self.fitted_species_indices) is not tuple
                or fitted != bool(self.fitted_species_indices)
                or len(set(self.fitted_species_indices)) != len(self.fitted_species_indices)
                or len(self.fitted_species_indices) > len(fixed.readout.observed)
                or any(type(index) is not int or not 0 <= index < 26 for index in self.fitted_species_indices)):
            raise ValueError("Adaptive fitted reference requires an authenticated observable-scale artifact")
        frozen_sequences = (fixed.sample_uids, fixed.particle_uids, *fixed.particle_uids,
                            fixed.residual_scales, fixed.inverse_normalizer_scales,
                            fixed.source_times, fixed.time_hours, fixed.readout.region_ids,
                            fixed.readout.compartment_ids, fixed.readout.species_ids,
                            fixed.readout.selector, *fixed.readout.selector, fixed.readout.scales,
                            fixed.readout.matrix, *fixed.readout.matrix, fixed.readout.observed)
        if any(type(value) is not tuple for value in frozen_sequences):
            raise ValueError("Adaptive reference metadata must remain immutable tuple snapshots")
        uids = tuple(action.regimen_uid for action in self.original_actions)
        if (len(set(uids)) != len(uids) or self.initial_action_uid not in uids
                or self.initial_action_uid != fixed.executed_regimen_uid):
            raise ValueError("Adaptive original action identities and initial fixed binding differ")
        expected_hours = original_query_time_hours(fixed.source_times, fixed.source_time_unit)
        if fixed.time_hours != expected_hours:
            raise ValueError("Adaptive readout requires the original absolute source/hour clock conversion")
        for action in self.original_actions:
            action.__post_init__()
            if (action.source_uid != fixed.readout.source_uid
                    or action.source_snapshot_ref != fixed.readout.source_snapshot_ref
                    or action.family_uid != fixed.readout.family_uid
                    or action.source_times != fixed.source_times
                    or action.source_time_unit != fixed.source_time_unit):
                raise ValueError("Supported original action source or absolute clock differs")


@dataclass(frozen=True)
class AdaptiveReadoutContext:
    """Exact public execution lineage, separate from the residual process key.

    Switches contain all executed original-library requests up to this readout.
    Composite UIDs alone cannot establish which original process applies.
    """

    original_action: NativeActuation
    executed_program: NativeActuation
    switches: tuple[NativeActuationRequest, ...]
    predictor_binding: str

    def validate(self, reference: AdaptiveReadoutReference, query_index: int) -> None:
        fixed = reference.fixed
        if (type(query_index) is not int or not 0 <= query_index < len(fixed.time_hours)
                or type(self.original_action) is not NativeActuation
                or self.original_action not in reference.original_actions
                or type(self.executed_program) is not NativeActuation
                or type(self.switches) is not tuple):
            raise ValueError("Adaptive query requires exact supported original action lineage and clock")
        self.executed_program.__post_init__()
        current = next(action for action in reference.original_actions
                       if action.regimen_uid == reference.initial_action_uid)
        original, previous_index = current, 0
        for request in self.switches:
            if (type(request) is not NativeActuationRequest
                    or type(request.prefix) is not NativeExecutedPrefix
                    or request.actuation not in reference.original_actions
                    or type(request.prefix.query_index) is not int
                    or not previous_index < request.prefix.query_index <= query_index
                    or request.prefix != executed_prefix(current, request.prefix.query_index)):
                raise ValueError("Adaptive execution switches lack the exact consumed original prefix")
            current = compose_actuation(request, reference.original_actions)
            original, previous_index = request.actuation, request.prefix.query_index
        if current != self.executed_program or original != self.original_action:
            raise ValueError("Adaptive original action cannot relabel the executed composite program")
        if type(self.predictor_binding) is not str or not self.predictor_binding:
            raise ValueError("Adaptive readout requires its actual pre-update predictor binding")
        binding = json.loads(self.predictor_binding)
        json.dumps(binding, allow_nan=False)
        if (not isinstance(binding, dict) or binding.get("normalization_id") != fixed.normalization_id
                or type(binding.get("checkpoint_sha256")) is not str or not binding["checkpoint_sha256"]):
            raise ValueError("Adaptive predictor requires a bound checkpoint and unchanged TRAIN normalizer")


@dataclass(frozen=True)
class AdaptiveReadoutRecord:
    """Immutable accepted pre-update prediction and observed-value snapshots.

    Numeric tuples, rather than mutable tensors, retain past innovations across
    H3 updates. Hypothetical branches can capture their own pre-update means in
    the same format. Differentiable trial windows use AdaptiveReadoutQuery.
    """

    reference: AdaptiveReadoutReference
    context: AdaptiveReadoutContext
    query_index: int
    filtering_revision: int
    observation_uid: str
    coordinates: tuple[int, ...]
    mean: tuple[tuple[float, ...], ...]  # 32,D, never recentered by a later predictor.
    values: tuple[float, ...]  # D, selected arrived values only.

    @classmethod
    def capture(cls, reference: AdaptiveReadoutReference, context: AdaptiveReadoutContext,
                arrived: NativeArrivedObservation, mean: torch.Tensor) -> AdaptiveReadoutRecord:
        reference.validate()
        if not isinstance(context, AdaptiveReadoutContext) or not isinstance(arrived, NativeArrivedObservation):
            raise ValueError("Adaptive snapshot requires accepted native packet and public execution context")
        acquisition = arrived.acquisition
        if (type(acquisition) is not NativeAcquisition
                or type(acquisition.filtering_revision) is not int
                or acquisition.filtering_revision < reference.fixed.initial_filtering_revision):
            raise ValueError("Adaptive snapshot requires a native acquisition with a current integer revision")
        acquisition.validate()
        context.validate(reference, acquisition.query_index)
        fixed = replace(reference.fixed, executed_regimen_uid=context.executed_program.regimen_uid,
                        predictor_binding=context.predictor_binding)
        offset = acquisition.filtering_revision - fixed.initial_filtering_revision
        if offset < 0:
            raise ValueError("Adaptive snapshot filtering revision precedes the starting history")
        coordinates, predicted, measured, _ = _record(
            fixed, FixedReadoutRecord(fixed, arrived, mean), offset, _mean(mean).device)
        return cls(reference, context, acquisition.query_index, acquisition.filtering_revision,
                   arrived.observation.observation_uid, coordinates,
                   tuple(tuple(row) for row in predicted.detach().cpu().tolist()),
                   tuple(measured.detach().cpu().tolist()))


@dataclass(frozen=True)
class AdaptiveReadoutQuery:
    context: AdaptiveReadoutContext
    query_index: int
    filtering_revision: int
    observed: tuple[bool, ...]
    mean: torch.Tensor  # 32,6; all future/window means keep their gradients.
    values: torch.Tensor | None = None  # 6; either every window query has values or none does.


@dataclass(frozen=True)
class ConditionalReadoutWindow:
    mean: torch.Tensor  # 32,sum(Dk), one complete tuple throughout the future window.
    covariance: torch.Tensor  # sum(Dk),sum(Dk), with independent acquisition noise.
    residual_covariance: torch.Tensor  # Same shape, without new acquisition noise.
    history_log_likelihood: torch.Tensor  # 32; frozen past means and observations.
    log_likelihood: torch.Tensor | None  # 32; joint future window, conditional on history.
    coordinates: tuple[tuple[int, ...], ...]


def _adaptive_coordinates(reference: AdaptiveReadoutReference, observed: tuple[bool, ...]) -> tuple[int, ...]:
    if (type(observed) is not tuple or len(observed) != 6
            or any(type(keep) is not bool for keep in observed)
            or any(keep and not available for keep, available in zip(observed, reference.fixed.readout.observed))):
        raise ValueError("Adaptive readout requires available boolean coordinates")
    coordinates = tuple(index for index, keep in enumerate(observed) if keep)
    if not coordinates:
        raise ValueError("Adaptive readout must select an available coordinate")
    return coordinates


def _adaptive_order(reference, context, index, revision, offset, previous_context, previous_index):
    if not isinstance(context, AdaptiveReadoutContext):
        raise ValueError("Adaptive readout requires its public original action context")
    context.validate(reference, index)
    if (type(revision) is not int or revision != reference.fixed.initial_filtering_revision + offset
            or index < previous_index):
        raise ValueError("Adaptive readout has stale revisions or nonchronological original clocks")
    if previous_context is not None:
        previous_switches = previous_context.switches
        if (context.switches[:len(previous_switches)] != previous_switches
                or any(request.prefix.query_index < previous_index
                       for request in context.switches[len(previous_switches):])):
            raise ValueError("Adaptive readout cannot rewrite an executed action history")



def _adaptive_record_coordinates(reference, row, previous_uids):
    """Validate one immutable accepted snapshot without replaying its predictor."""
    if type(row) is not AdaptiveReadoutRecord or row.reference != reference:
        raise ValueError("Adaptive history changed its complete tuple, scales or original support reference")
    if (type(row.query_index) is not int or row.query_index <= 0
            or type(row.observation_uid) is not str or not row.observation_uid
            or row.observation_uid in previous_uids or type(row.coordinates) is not tuple
            or any(type(index) is not int or not 0 <= index < 6 for index in row.coordinates)
            or row.coordinates != tuple(sorted(set(row.coordinates)))):
        raise ValueError("Adaptive history requires unique postzero acquisition identities and coordinates")
    coordinates = _adaptive_coordinates(reference, tuple(i in row.coordinates for i in range(6)))
    if (type(row.mean) is not tuple or len(row.mean) != 32
            or any(type(values) is not tuple or len(values) != len(coordinates) for values in row.mean)
            or type(row.values) is not tuple or len(row.values) != len(coordinates)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for values in row.mean for value in values)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in row.values)):
        raise ValueError("Adaptive history requires immutable finite complete-tuple mean/value snapshots")
    return coordinates

def _adaptive_covariance(base, blocks, reference_time):
    # Independent processes are keyed by the ORIGINAL action, never composite UID
    # or predictor revision. min(t,s) uses absolute source time even on a revisit.
    return torch.cat([torch.cat([
        base[list(a)][:, list(b)] * (min(t, s) / reference_time if action == other else 0.)
        for b, s, other in blocks], dim=1) for a, t, action in blocks], dim=0)


def adaptive_readout_window(reference: AdaptiveReadoutReference,
                            history: tuple[AdaptiveReadoutRecord, ...],
                            queries: tuple[AdaptiveReadoutQuery, ...], *,
                            trial_scales: torch.Tensor | None = None, _allow_zero_boundary=False) -> ConditionalReadoutWindow:
    """Joint future/H3 density conditioned only on the frozen window-start history.

    H3 callers must freeze both starting tuple weights (owned by the caller) and
    this discrepancy history. Every trial mean remains differentiable; no trial
    acquisition is detached and fed back as accepted history within this window.
    The same covariance supplies hypothetical readouts and terminal residual
    draws. Discrepancy never updates physical state, exposure, dose or truth.
    This explicit software candidate remains UNQUALIFIED until independent adaptive
    calibration-fit/check and policy qualification.
    """
    reference.validate()
    if (type(history) is not tuple or type(queries) is not tuple or not queries
            or any(type(query) is not AdaptiveReadoutQuery for query in queries)):
        raise ValueError("Adaptive readout requires immutable history and a nonempty trial window")
    device = _mean(queries[0].mean).device
    fixed = reference.fixed
    base = projected_brownian_covariance(fixed, device=device, trial_scales=trial_scales,
                                         _allow_zero_boundary=_allow_zero_boundary)
    blocks, historical_means, historical_values, future_means, future_values = [], [], [], [], []
    previous_context, previous_index, uids = None, 0, set()
    for offset, row in enumerate(history):
        coordinates = _adaptive_record_coordinates(reference, row, uids)
        _adaptive_order(reference, row.context, row.query_index, row.filtering_revision, offset,
                        previous_context, previous_index)
        historical_means.append(torch.tensor(row.mean, dtype=torch.float64, device=device))
        historical_values.append(torch.tensor(row.values, dtype=torch.float64, device=device))
        blocks.append((coordinates, fixed.time_hours[row.query_index], row.context.original_action.regimen_uid))
        previous_context, previous_index = row.context, row.query_index
        uids.add(row.observation_uid)
    count = sum(len(coordinates) for coordinates, _, _ in blocks)
    selected_queries = []
    has_values = queries[0].values is not None
    for position, query in enumerate(queries):
        _adaptive_order(reference, query.context, query.query_index, query.filtering_revision, len(history) + position,
                        previous_context, previous_index)
        coordinates = _adaptive_coordinates(reference, query.observed)
        future_means.append(_mean(query.mean, device)[:, list(coordinates)])
        if (query.values is not None) != has_values:
            raise ValueError("Adaptive joint window requires values for every query or none")
        if has_values:
            values = query.values
            if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
                    or values.shape != (6,) or values.device != device
                    or not torch.isfinite(values[list(coordinates)]).all()):
                raise ValueError("Adaptive trial values must be finite on selected coordinates")
            future_values.append(values[list(coordinates)].double())
        blocks.append((coordinates, fixed.time_hours[query.query_index], query.context.original_action.regimen_uid))
        selected_queries.append(coordinates)
        previous_context, previous_index = query.context, query.query_index
    joint = _adaptive_covariance(base, blocks, fixed.reference_time_hours)
    noise = fixed.readout.standard_deviation ** 2
    future_mean = torch.cat(future_means, dim=1)
    residual_covariance = joint[count:, count:]
    history_likelihood = future_mean.new_zeros(32)
    if count:
        historical_mean, values = torch.cat(historical_means, dim=1), torch.cat(historical_values)
        past_covariance = joint[:count, :count] + torch.eye(count, dtype=joint.dtype, device=device) * noise
        factor = _cholesky(past_covariance)
        cross = joint[count:, :count]
        gain = torch.cholesky_solve(cross.T, factor).T
        future_mean = future_mean + (values.unsqueeze(0) - historical_mean) @ gain.T
        residual_covariance = residual_covariance - gain @ cross.T
        history_likelihood = _log_density(values, historical_mean, factor)
    covariance = residual_covariance + torch.eye(future_mean.shape[1], dtype=joint.dtype, device=device) * noise
    factor = _cholesky(covariance)
    likelihood = _log_density(torch.cat(future_values), future_mean, factor) if has_values else None
    if not torch.isfinite(future_mean).all() or not torch.isfinite(residual_covariance).all():
        raise ValueError("Adaptive conditional readout result is not finite")
    return ConditionalReadoutWindow(future_mean, covariance, residual_covariance, history_likelihood,
                                    likelihood, tuple(selected_queries))



@dataclass(frozen=True)
class ReloadedAdaptiveReadoutHistory:
    """Structurally checked transport; external JSON is not acquisition authority."""

    reference: AdaptiveReadoutReference
    history: tuple[AdaptiveReadoutRecord, ...]
    initial_log_weights: tuple[float, ...]
    status: str
    authentication: str = "structural_validation_against_expected_report_context"
    scientific_readiness: str = "NOT_ADMITTED"
    calibration_eligible: bool = False


def _history_json(value):
    # Only public reference/policy metadata and selected observations enter here.
    return json.loads(json.dumps(value, allow_nan=False))


def _history_reference_binding(reference: AdaptiveReadoutReference) -> dict:
    reference.validate()
    return {"fixed": _history_json(actuation_payload(reference.fixed)),
            "original_action_uids": [action.regimen_uid for action in reference.original_actions],
            "initial_action_uid": reference.initial_action_uid, "law": reference.law}


def _history_report_context(reference: AdaptiveReadoutReference, context: dict) -> dict:
    if type(reference) is not AdaptiveReadoutReference:
        raise ValueError("Actual history requires the expected typed adaptive reference")
    reference.validate()
    required = {"group_uid", "family_uid", "binding", "reference_id", "pre_execution_policy_snapshot"}
    if (type(context) is not dict or set(context) != required
            or any(type(context[name]) is not str or not context[name]
                   for name in ("group_uid", "family_uid", "reference_id"))
            or context["family_uid"] != reference.fixed.readout.family_uid
            or context["binding"] != json.loads(reference.fixed.predictor_binding)
            or type(context["pre_execution_policy_snapshot"]) is not dict
            or set(context["pre_execution_policy_snapshot"]) != {
                "native_residual_control", "native_temporal_query",
                "native_mechanistic_residual_online", "native_arrived_observation_filter"}):
        raise ValueError("Actual history requires its existing group, predictor and policy report context")
    result = _history_json(context)
    # Declared training/integration seeds are public provenance. Evaluator state
    # and its private measurement RNG are never transport fields.
    forbidden = {"measurement_seed", "measurement_generator", "rng_state", "private_fields",
                 "private_initial_field", "private_factors", "latent_index", "hidden_tuple_index"}
    def check(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("Private evaluator state cannot enter actual history transport")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(result)
    policy = result["pre_execution_policy_snapshot"]["native_mechanistic_residual_online"]
    if (type(policy) is not dict
            or any(type(policy.get(name)) is not int or policy[name] < 0
                   for name in ("warmup_transitions", "updates_per_observation"))):
        raise ValueError("Actual history requires the declared residual update policy")
    return result


def normalized_adaptive_initial_weights(weights: tuple[float, ...], *, device=None,
                             error_message="Actual history requires the immutable normalized initial32-tuple prior") -> torch.Tensor:
    """Validate the exact normalized prior; preserve zero mass without flooring."""
    if (type(weights) is not tuple or len(weights) != 32
            or any(type(value) not in (int, float) or math.isnan(value) or value == math.inf for value in weights)
            or not any(math.isfinite(value) for value in weights)):
        raise ValueError(error_message)
    initial = torch.tensor(weights, dtype=torch.float64, device=device)
    if not torch.allclose(initial.logsumexp(0), initial.new_zeros(()), atol=1e-12, rtol=0):
        raise ValueError(error_message)
    return initial


def _validate_transport_records(reference, history, context):
    if type(history) is not tuple:
        raise ValueError("Actual history requires immutable accepted records")
    uids, previous, previous_index = set(), None, 0
    policy = context["pre_execution_policy_snapshot"]["native_mechanistic_residual_online"]
    base = json.loads(reference.fixed.predictor_binding)
    for offset, row in enumerate(history):
        _adaptive_record_coordinates(reference, row, uids)
        _adaptive_order(reference, row.context, row.query_index, row.filtering_revision,
                        offset, previous, previous_index)
        updates = max(0, offset-policy["warmup_transitions"]+1)*policy["updates_per_observation"]
        if (row.query_index != offset+1
                or json.loads(row.context.predictor_binding) != dict(base, residual_optimizer_updates=updates)):
            raise ValueError("Actual history changed its acquisition clock or pre-update predictor revision")
        uids.add(row.observation_uid)
        previous, previous_index = row.context, row.query_index


def export_adaptive_readout_history(reference: AdaptiveReadoutReference,
                                    history: tuple[AdaptiveReadoutRecord, ...],
                                    initial_log_weights: tuple[float, ...], *,
                                    report_context: dict, episode_status: str) -> dict:
    """Encode supplied records; only the episode accessor establishes actual origin.

    Exact public predictor and policy bindings retain training/integration seeds.
    Private evaluator fields, latent indices and measurement RNG remain excluded.
    """
    context = _history_report_context(reference, report_context)
    normalized_adaptive_initial_weights(initial_log_weights)
    _validate_transport_records(reference, history, context)
    if type(episode_status) is not str or episode_status not in {"ready", "active", "failed", "component_complete", "blocked"}:
        raise ValueError("Unknown actual history episode status")
    status = ("failed" if episode_status == "failed" else "empty" if not history
              else "complete" if episode_status == "component_complete" else "partial")
    rows = []
    for row in history:
        rows.append({"query_index": row.query_index, "filtering_revision": row.filtering_revision,
            "observation_uid": row.observation_uid, "coordinates": list(row.coordinates),
            "mean": [list(values) for values in row.mean], "values": list(row.values),
            "original_action_uid": row.context.original_action.regimen_uid,
            "executed_regimen_uid": row.context.executed_program.regimen_uid,
            "switches": [{"action_uid": request.actuation.regimen_uid,
                          "prefix": _history_json(actuation_payload(request.prefix))} for request in row.context.switches],
            "predictor_binding": row.context.predictor_binding})
    return {"schema_version": "native_residual_control_history.v1",
        "origin": "supplied_records_without_episode_authentication", "status": status,
        "episode_status": episode_status, "scientific_readiness": "NOT_ADMITTED",
        "adaptive_scientific_qualification": "UNQUALIFIED", "calibration_eligible": False,
        "likelihood_data_available": bool(history), "report_context": context,
        "reference_binding": _history_reference_binding(reference),
        "initial_log_weights": ["-inf" if value == -math.inf else value for value in initial_log_weights],
        "records": rows, "record_count": len(rows)}


def reload_adaptive_readout_history(payload: dict, *, expected_reference: AdaptiveReadoutReference,
                                   expected_report_context: dict) -> ReloadedAdaptiveReadoutHistory:
    """Check structure and caller bindings; never authenticate arbitrary JSON."""
    context = _history_report_context(expected_reference, expected_report_context)
    fields = {"schema_version", "origin", "status", "episode_status", "scientific_readiness",
              "adaptive_scientific_qualification", "calibration_eligible", "likelihood_data_available",
              "report_context", "reference_binding", "initial_log_weights", "records", "record_count"}
    if (type(payload) is not dict or set(payload) != fields
            or payload["schema_version"] != "native_residual_control_history.v1"
            or payload["origin"] != "episode_authenticated_actual_acceptance"
            or payload["scientific_readiness"] != "NOT_ADMITTED"
            or payload["adaptive_scientific_qualification"] != "UNQUALIFIED"
            or payload["calibration_eligible"] is not False
            or payload["report_context"] != context
            or payload["reference_binding"] != _history_reference_binding(expected_reference)
            or type(payload["records"]) is not list
            or type(payload["record_count"]) is not int or payload["record_count"] != len(payload["records"])
            or type(payload["initial_log_weights"]) is not list):
        raise ValueError("Actual history transport origin, schema or expected report binding differs")
    raw_weights = payload["initial_log_weights"]
    if any((type(value) is str and value != "-inf")
           or type(value) not in (int, float, str)
           or type(value) in (int, float) and not math.isfinite(value) for value in raw_weights):
        raise ValueError("Only explicit '-inf' may represent zero initial tuple mass")
    weights = tuple(-math.inf if value == "-inf" else value for value in raw_weights)
    normalized_adaptive_initial_weights(weights)
    actions = {action.regimen_uid: action for action in expected_reference.original_actions}
    rows = []
    row_fields = {"query_index", "filtering_revision", "observation_uid", "coordinates", "mean", "values",
                  "original_action_uid", "executed_regimen_uid", "switches", "predictor_binding"}
    for raw in payload["records"]:
        if (type(raw) is not dict or set(raw) != row_fields
                or type(raw["switches"]) is not list or type(raw["coordinates"]) is not list
                or type(raw["mean"]) is not list or any(type(values) is not list for values in raw["mean"])
                or type(raw["values"]) is not list or type(raw["original_action_uid"]) is not str
                or raw["original_action_uid"] not in actions):
            raise ValueError("Malformed actual accepted readout record")
        program = actions[expected_reference.initial_action_uid]
        switches = []
        for item in raw["switches"]:
            if (type(item) is not dict or set(item) != {"action_uid", "prefix"}
                    or type(item["action_uid"]) is not str or item["action_uid"] not in actions
                    or type(item["prefix"]) is not dict or type(item["prefix"].get("query_index")) is not int):
                raise ValueError("Malformed actual original-action switch")
            prefix = executed_prefix(program, item["prefix"]["query_index"])
            if item["prefix"] != _history_json(actuation_payload(prefix)):
                raise ValueError("Actual action switch changed its consumed original prefix")
            request = NativeActuationRequest(prefix, actions[item["action_uid"]])
            program = compose_actuation(request, expected_reference.original_actions)
            switches.append(request)
        if raw["executed_regimen_uid"] != program.regimen_uid:
            raise ValueError("Actual readout changed its composed execution identity")
        execution = AdaptiveReadoutContext(actions[raw["original_action_uid"]], program,
                                           tuple(switches), raw["predictor_binding"])
        rows.append(AdaptiveReadoutRecord(expected_reference, execution, raw["query_index"],
            raw["filtering_revision"], raw["observation_uid"], tuple(raw["coordinates"]),
            tuple(tuple(values) for values in raw["mean"]), tuple(raw["values"])))
    history = tuple(rows)
    rebuilt = export_adaptive_readout_history(expected_reference, history, weights,
        report_context=context, episode_status=payload["episode_status"])
    if (payload["status"] != rebuilt["status"]
            or payload["likelihood_data_available"] is not rebuilt["likelihood_data_available"]):
        raise ValueError("Actual history status differs from its accepted record prefix")
    return ReloadedAdaptiveReadoutHistory(expected_reference, history, weights, payload["status"])


def adaptive_history_log_likelihood(reference: AdaptiveReadoutReference,
                                    history: tuple[AdaptiveReadoutRecord, ...],
                                    initial_log_weights: tuple[float, ...], *,
                                    trial_scales: torch.Tensor | None = None, _allow_zero_boundary=False) -> torch.Tensor:
    """One complete-history mixture of raw accepted pre-update readout means.

    The final accepted snapshot becomes a masked query in the existing window
    likelihood. Its joint density is the prefix density plus the conditional
    final density. One initial biological tuple supplies the complete history.
    Trial scales do not replay predictors, actions, acquisition or stopping.
    This likelihood prerequisite supplies no scale fit, policy qualification,
    selected-decision calibration, or identification of unobserved species.
    """
    if (type(reference) is not AdaptiveReadoutReference or type(history) is not tuple or not history
            or any(type(row) is not AdaptiveReadoutRecord for row in history)):
        raise ValueError("Adaptive likelihood requires a nonempty immutable accepted history")
    reference.validate()
    device = (trial_scales.device if isinstance(trial_scales, torch.Tensor)
              and trial_scales.device.type != "meta" else None)
    initial = normalized_adaptive_initial_weights(initial_log_weights, device=device,
        error_message="Adaptive likelihood requires the normalized immutable initial32-tuple prior")
    final = history[-1]
    coordinates = _adaptive_record_coordinates(reference, final, tuple(row.observation_uid for row in history[:-1]))
    mean, values = initial.new_zeros((32, 6)), initial.new_zeros(6)
    mean[:, list(coordinates)] = mean.new_tensor(final.mean)
    values[list(coordinates)] = values.new_tensor(final.values)
    query = AdaptiveReadoutQuery(final.context, final.query_index, final.filtering_revision,
        tuple(i in coordinates for i in range(6)), mean, values)
    window = adaptive_readout_window(reference, history[:-1], (query,), trial_scales=trial_scales,
                                     _allow_zero_boundary=_allow_zero_boundary)
    return torch.logsumexp(initial + window.history_log_likelihood + window.log_likelihood, 0)


@dataclass(frozen=True)
class AdaptiveCalibrationEpisode:
    group_uid: str
    reference: AdaptiveReadoutReference
    history: tuple[AdaptiveReadoutRecord, ...]
    initial_log_weights: tuple[float, ...]


def adaptive_observable_species(episodes: tuple[AdaptiveCalibrationEpisode, ...]) -> tuple[int, ...]:
    """Species with a nonzero projection into an actually accepted coordinate."""
    indices = set()
    maximum = 0
    for episode in episodes:
        episode.reference.validate()
        readout = episode.reference.fixed.readout
        coordinates = {i for i, available in enumerate(readout.observed) if available}
        maximum = max(maximum, len(readout.observed))
        for column, selector in enumerate(readout.selector):
            if any(readout.matrix[coordinate][column] != 0 for coordinate in coordinates):
                indices.add(selector[-1])
    if not indices or len(indices) > maximum:
        raise ValueError("Adaptive calibration requires one to six observable species")
    return tuple(sorted(indices))


def adaptive_calibration_objective(eta: torch.Tensor, episodes: tuple[AdaptiveCalibrationEpisode, ...],
                                   indices: tuple[int, ...]) -> torch.Tensor:
    """Mean complete-history NLL per independent group; no coordinate divisor."""
    losses = []
    for episode in episodes:
        base = eta.new_tensor(episode.reference.fixed.residual_scales)
        trial = base.index_copy(0, torch.tensor(indices, device=eta.device), eta.exp())
        losses.append(-adaptive_history_log_likelihood(episode.reference, episode.history,
                         episode.initial_log_weights, trial_scales=trial))
    return torch.stack(losses).mean()



def adaptive_scale_boundary_diagnostics(episodes, indices, scales, tolerance):
    """Compare coordinate zero boundaries at fixed other scales; no second fit.

    Sensor noise supplies a positive covariance at zero discrepancy. This local
    coordinate check does not certify the global mixture or joint boundary.
    Ordinary trial and live reference scales remain strictly positive.
    """
    trial = torch.tensor(scales, dtype=torch.float64)
    def loss(value, boundary=False):
        return torch.stack([-adaptive_history_log_likelihood(row.reference, row.history, row.initial_log_weights,
            trial_scales=value, _allow_zero_boundary=boundary) for row in episodes]).mean()
    optimum = float(loss(trial))
    rows = []
    for index in indices:
        boundary = trial.clone();boundary[index] = 0.
        objective = float(loss(boundary, True))
        rows.append({"species_index": index, "zero_boundary_nll": objective,
                     "interior_improvement": objective-optimum})
    return {"scope": "coordinate_zero_boundary_other_scales_fixed_not_global_identifiability",
        "numerical_tolerance": tolerance, "comparisons": rows,
        "interior_separated": all(math.isfinite(row["interior_improvement"])
                                  and row["interior_improvement"] > tolerance for row in rows)}

def fit_adaptive_readout_scales(episodes: tuple[AdaptiveCalibrationEpisode, ...],
                               declared_groups: tuple[str, ...], numerical: dict) -> dict:
    """Fit accepted immutable histories after the collection consumer admits them."""
    groups = tuple(row.group_uid for row in episodes)
    if (not episodes or len(set(groups)) != len(groups) or set(groups) != set(declared_groups)
            or len(set(declared_groups)) != len(declared_groups)
            or any(not row.history for row in episodes)):
        return {"status": "FIT_FAILED", "reason": "incomplete_or_repeated_independent_group_information",
                "scales": None, "observable_fit_valid": False}
    indices = adaptive_observable_species(episodes)
    offline = episodes[0].reference.fixed.residual_scales
    if any(row.reference.fixed.residual_scales != offline for row in episodes):
        raise ValueError("Adaptive episodes must share their immutable offline scale authority")
    initial = torch.tensor([offline[index] for index in indices], dtype=torch.float64).log()
    result = minimize_log_scales(lambda eta: adaptive_calibration_objective(eta, episodes, indices), initial, numerical)
    result.update(fitted_species_indices=list(indices), inherited_species_indices=[i for i in range(26) if i not in indices],
        group_count=len(groups), accepted_record_count=sum(len(row.history) for row in episodes),
        objective_kind="mean_complete_history_mixture_nll_per_independent_group",
        whole_26_field_adaptive_qualification="UNQUALIFIED", unobserved_scale_validation="NOT_VALIDATED",
        observable_fit_valid=result["status"] == "FIT_COMPLETE")
    if result["status"] == "FIT_COMPLETE":
        complete = list(offline)
        for index, scale in zip(indices, result["scales"]):
            complete[index] = scale
        boundary = adaptive_scale_boundary_diagnostics(episodes,indices,complete,numerical["gradient_tolerance"])
        result["diagnostics"]["boundary"] = boundary
        if not boundary["interior_separated"]:
            result.update(status="FIT_FAILED", scales=None, observable_fit_valid=False,
                          reason="collapsed_or_indistinguishable_zero_scale_boundary")
        else:
            result["scales"] = complete
    return result


@torch.no_grad()
def filter_adaptive_native_observation(solver, state, prediction: NativePredictedReadout,
        arrived: NativeArrivedObservation, *, reference: AdaptiveReadoutReference,
        history: tuple[AdaptiveReadoutRecord, ...], initial_log_weights: tuple[float, ...],
        context: AdaptiveReadoutContext) -> dict:
    """Prospective weight-only candidate under one complete original-action law.

    The episode authenticates continuation-owned historical snapshots. This
    boundary checks the actual current packet before any admission mutation.
    """
    fresh = validate_native_observation(solver, state, prediction, arrived)
    initial = normalized_adaptive_initial_weights(initial_log_weights, device=fresh.mean.device,
        error_message="Adaptive filter requires its normalized immutable initial32-tuple prior")
    if (len(history) != state.filter_revision
            or tuple(row.observation_uid for row in history) !=
                state.filter_observation_uids[len(state.filter_observation_uids)-state.filter_revision:]):
        raise ValueError("Adaptive filter requires the complete actual accepted history")
    accepted = AdaptiveReadoutRecord.capture(reference, context, arrived, fresh.mean[0])
    query = AdaptiveReadoutQuery(context, accepted.query_index, accepted.filtering_revision,
        tuple(bool(value) for value in arrived.observation.observed[0].tolist()),
        fresh.mean[0], arrived.observation.values[0])
    conditioned = adaptive_readout_window(reference, history, (query,))
    prior = torch.log_softmax(initial + conditioned.history_log_likelihood, 0)[None]
    if not torch.allclose(prior, solver.current_log_weights(state), atol=1e-12, rtol=1e-12):
        raise ValueError("Adaptive filter current posterior differs from the SAME-law complete accepted history")
    updated = torch.log_softmax(initial + conditioned.history_log_likelihood + conditioned.log_likelihood, 0)[None]
    result = _admit_native_observation(solver, state, fresh.acquisition, arrived, updated,
                                     conditioned.log_likelihood[None])
    result.update(readout_law=reference.law, adaptive_scientific_qualification="UNQUALIFIED",
        scope="prospective_adaptive_episode_software",
        tuple_prior_authority="immutable_initial_weights_plus_same_law_complete_preupdate_history")
    return result
