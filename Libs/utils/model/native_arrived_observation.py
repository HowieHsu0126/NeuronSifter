"""Original-query native readouts and weight-only filtering on carried physics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import TYPE_CHECKING

import torch

from Libs.models.modules.world_state import _normalized_log_weights
from Libs.utils.data.joint_observation import RetainedObservation, observation_to_payload, retained_log_likelihood
from Libs.utils.data.native_observable_profile import family_grid, observable_profile

if TYPE_CHECKING:
    from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver, NativeRHSContinuation


@dataclass(frozen=True)
class NativeCNSReadout:
    channel_uid: str
    assay_uid: str
    source_uid: str
    source_snapshot_ref: str
    role: str
    family_uid: str
    normalization_ref: str
    units: str
    region_ids: tuple[str, ...]
    compartment_ids: tuple[str, ...]
    species_ids: tuple[str, ...]
    selector: tuple[tuple[int, int, int], ...]
    scales: tuple[float, ...]
    matrix: tuple[tuple[float, ...], ...]
    observed: tuple[bool, ...]
    standard_deviation: float

    def validate(self) -> None:
        count = len(self.selector)
        if (count != 6 or len(self.scales) != count or len(self.matrix) != count
                or any(len(row) != count for row in self.matrix)
                or len(self.observed) != count or any(type(v) is not bool for v in self.observed)
                or any(not math.isfinite(v) or v <= 0 for v in self.scales)
                or any(not math.isfinite(v) for row in self.matrix for v in row)
                or not math.isfinite(self.standard_deviation) or self.standard_deviation <= 0):
            raise ValueError("Native readout requires its six finite scaled coordinates, mixing, mask and positive noise")
        shape = (len(self.region_ids), len(self.compartment_ids), len(self.species_ids))
        if any(len(row) != 3 or any(type(i) is not int or not 0 <= i < size
                                   for i, size in zip(row, shape)) for row in self.selector):
            raise ValueError("Native readout selector is outside the bound physical grid")


def native_acquisition_mask(readout: NativeCNSReadout,
                            selected_coordinate: int | None = None) -> tuple[bool, ...]:
    """Select one available output of the fixed public mixing transform."""
    readout.validate()
    if selected_coordinate is None:
        return readout.observed
    if (type(selected_coordinate) is not int
            or not 0 <= selected_coordinate < len(readout.observed)
            or not readout.observed[selected_coordinate]):
        raise ValueError("Native acquisition coordinate must select an available mixed readout")
    return tuple(index == selected_coordinate for index in range(len(readout.observed)))


@dataclass(frozen=True)
class NativeAcquisition:
    readout: NativeCNSReadout
    support_uid: str
    origin_posterior_uid: str
    sample_uids: tuple[str, ...]
    particle_uids: tuple[tuple[str, ...], ...]
    filtering_revision: int
    query_index: int
    source_time: float
    source_time_unit: str
    time_hours: float
    executed_regimen_uid: str
    event_phase: str = "post_event"
    selected_coordinate: int | None = None

    def validate(self) -> None:
        native_acquisition_mask(self.readout, self.selected_coordinate)
        if (type(self.query_index) is not int or self.query_index <= 0
                or type(self.filtering_revision) is not int or self.filtering_revision < 0
                or self.event_phase != "post_event"
                or self.source_time_unit not in {"ms", "hour"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (self.source_time, self.time_hours))
                or not all((self.support_uid, self.origin_posterior_uid, self.executed_regimen_uid))):
            raise ValueError("Native acquisition requires a postzero original query, current revision and post-event phase")


@dataclass(frozen=True)
class NativePredictedReadout:
    acquisition: NativeAcquisition
    mean: torch.Tensor
    variance: torch.Tensor
    nuisance: torch.Tensor


@dataclass(frozen=True)
class NativeArrivedObservation:
    """Inference packet: measurement and public binding only; no evaluator state or seeds."""

    acquisition: NativeAcquisition
    observation: RetainedObservation


def native_cns_readout(config: dict, group: dict) -> NativeCNSReadout:
    # The source module imports the observation producer; delay this metadata-only
    # import to avoid a producer/readout/source import cycle.
    from Libs.utils.model.native_scientific_source import scoped_group_config
    policy = config["native_arrived_observation_filter"]
    expected = {
        "schema_version": "native_arrived_observation_filter.v1",
        "query_policy": "original_postzero_queries",
        "event_phase": "post_event",
        "temporal_noise": "independent_across_acquisitions_and_coordinates_after_mixing",
        "update": "current_log_weight_filter_only",
        "measurement_kind": "controlled_simulated",
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Native arrived observation policy differs from its prospective definition")
    scoped = scoped_group_config(config, group)
    settings, source = scoped["native_observable_bridge"], scoped["controlled_cns_source"]
    grid = family_grid(observable_profile(scoped), group["family_uid"])
    selector = source["observation"]["selector"]
    count = len(selector)
    matrix = settings.get("observation_matrix")
    if matrix is None:
        raise ValueError("Native monitoring requires an explicitly scoped public measurement matrix")
    spec = NativeCNSReadout(
        policy["channel_uid"], policy["assay_uid"], settings["source_uid"], settings["snapshot_ref"],
        group["role"], group["family_uid"], settings["snapshot_ref"]+":fixed_source_scales",
        "source_reference_standardized", tuple(grid["region_id"]), tuple(grid["compartment_id"]),
        tuple(grid["species_id"]),
        tuple((row["region_index"], row["compartment_index"], grid["species_id"].index(row["species"]))
              for row in selector),
        tuple(float(source["reference_scales"][row["species"]]) for row in selector),
        tuple(tuple(float(v) for v in row) for row in matrix),
        tuple(source["observation"]["mask"]), float(source["observation"]["standard_deviation"]))
    spec.validate()
    if count != 6:
        raise ValueError("Native monitoring requires the existing six-coordinate selector")
    return spec


def project_cns_readout(fields: torch.Tensor, readout: NativeCNSReadout) -> torch.Tensor:
    """Shared numeric projection; callers separately own predicted versus private fields."""
    readout.validate()
    shape = (len(readout.region_ids), len(readout.compartment_ids), len(readout.species_ids))
    if fields.ndim < 4 or tuple(fields.shape[-3:]) != shape:
        raise ValueError("Native readout field does not match its bound physical grid")
    selected = torch.stack([fields[..., r, c, q] for r, c, q in readout.selector], -1).double()
    if not torch.isfinite(selected).all():
        raise ValueError("Native readout selected coordinates must be finite")
    scaled = selected/selected.new_tensor(readout.scales)
    return scaled @ selected.new_tensor(readout.matrix).T


def _acquisition_for_context(context: dict, readout: NativeCNSReadout,
                             executed_regimen_uid: str, *,
                             selected_coordinate: int | None = None) -> NativeAcquisition:
    readout.validate()
    if (readout.source_uid != context["source_uid"]
            or readout.source_snapshot_ref != context["source_snapshot_ref"]
            or readout.role != context["role"] or readout.family_uid != context["family_uid"]
            or readout.region_ids != context["region_ids"] or readout.compartment_ids != context["compartment_ids"]
            or readout.species_ids != context["species_ids"]
            or executed_regimen_uid not in ((context["public_execution_uid"],)
                if "public_execution_uid" in context else context["regimen_uids"])):
        raise ValueError("Native readout source/grid/executed-program binding differs from the continuation")
    acquisition = NativeAcquisition(readout, context["support_uid"], context["origin_posterior_uid"],
        context["sample_uids"], context["particle_uids"], context["filtering_revision"],
        context["query_index"], context["source_time"], context["source_time_unit"],
        context["time_hours"], executed_regimen_uid, selected_coordinate=selected_coordinate)
    acquisition.validate()
    return acquisition


def _current_acquisition(solver: NativeFrozenRHSSolver, state: NativeRHSContinuation,
                         readout: NativeCNSReadout, executed_regimen_uid: str, *,
                         selected_coordinate: int | None = None) -> NativeAcquisition:
    acquisition = _acquisition_for_context(solver.filter_context(state), readout, executed_regimen_uid,
                                           selected_coordinate=selected_coordinate)
    solver.bind_filter_readout(state, readout, executed_regimen_uid)
    return acquisition


@torch.no_grad()
def predict_native_readout(solver: NativeFrozenRHSSolver, state: NativeRHSContinuation,
                           readout: NativeCNSReadout, *, executed_regimen_uid: str,
                           selected_coordinate: int | None = None) -> NativePredictedReadout:
    acquisition = _current_acquisition(solver, state, readout, executed_regimen_uid,
                                       selected_coordinate=selected_coordinate)
    fields = solver.current_particle_fields(state)
    context = solver.filter_context(state)
    action_index = context["regimen_uids"].index(context.get("executed_model_regimen_uid",executed_regimen_uid))
    mean = project_cns_readout(fields[:, :, action_index], readout)
    return NativePredictedReadout(acquisition, mean,
        torch.full_like(mean, readout.standard_deviation**2), torch.zeros_like(mean))


def _validate_arrived_payload(arrived: NativeArrivedObservation, expected: NativeAcquisition) -> None:
    if not isinstance(arrived, NativeArrivedObservation) or arrived.acquisition != expected:
        raise ValueError("Stale or changed native acquisition clock/program/readout/particle lineage")
    observation = arrived.observation
    shape = (len(expected.sample_uids), len(expected.readout.selector))
    if not isinstance(observation, RetainedObservation) or observation.values.shape != shape:
        raise ValueError("Native arrived values must retain only sample/readout axes")
    spec = expected.readout
    if (observation.source_uid != spec.source_uid or observation.source_snapshot_ref != spec.source_snapshot_ref
            or observation.sample_uids != expected.sample_uids or observation.role != spec.role
            or observation.channel_uid != spec.channel_uid or observation.assay_uid != spec.assay_uid
            or observation.sequence_number != expected.filtering_revision+1
            or observation.normalization_ref != spec.normalization_ref
            or observation.modality != "state" or observation.physical_acquisition is not None
            or observation.encoder_indices != tuple(range(len(spec.selector)))
            or observation.informative_admission or observation.admission_log_probability is not None):
        raise ValueError("Native arrived measurement identity/sequence/channel differs from its acquisition")
    mask = torch.tensor(native_acquisition_mask(spec, expected.selected_coordinate),
                        device=observation.values.device, dtype=torch.bool).expand(len(expected.sample_uids), -1)
    if not torch.equal(observation.observed, mask) or observation.left_censored.any():
        raise ValueError("Native arrived measurement mask/censoring differs from its public readout")
    if expected.selected_coordinate is not None and torch.any(observation.values[~mask] != 0):
        raise ValueError("Unselected native acquisition values must be zero and unobserved")


@torch.no_grad()
def validate_native_observation(solver: NativeFrozenRHSSolver, state: NativeRHSContinuation,
                                prediction: NativePredictedReadout,
                                arrived: NativeArrivedObservation) -> NativePredictedReadout:
    """Validate the current packet and physical prediction without filter mutation."""
    if not isinstance(arrived, NativeArrivedObservation) or not isinstance(prediction, NativePredictedReadout):
        raise ValueError("Native filtering requires a predicted readout and arrived packet")
    observation = arrived.observation
    if not isinstance(observation, RetainedObservation):
        raise ValueError("Native filtering requires a retained observation")
    if observation.observation_uid in state.filter_observation_uids:
        raise ValueError("Duplicate native arrived observation")
    prediction.acquisition.validate()
    arrived.acquisition.validate()
    context = solver.filter_context(state)
    expected = _acquisition_for_context(context, prediction.acquisition.readout,
        prediction.acquisition.executed_regimen_uid, selected_coordinate=prediction.acquisition.selected_coordinate)
    if (state._filter_readout is not None and state._filter_readout != expected.readout
            or state._filter_executed_regimen_uid is not None
            and state._filter_executed_regimen_uid != expected.executed_regimen_uid):
        raise ValueError("Native filtering readout or executed program changed after binding")
    if prediction.acquisition != expected or arrived.acquisition != expected:
        raise ValueError("Stale or changed native acquisition clock/program/readout/particle lineage")
    _validate_arrived_payload(arrived, expected)
    spec = expected.readout
    fields = solver.current_particle_fields(state)
    action_index = context["regimen_uids"].index(context.get("executed_model_regimen_uid", expected.executed_regimen_uid))
    mean = project_cns_readout(fields[:, :, action_index], spec)
    fresh = NativePredictedReadout(expected, mean, torch.full_like(mean, spec.standard_deviation**2),
                                  torch.zeros_like(mean))
    if any(not torch.equal(getattr(prediction, name), getattr(fresh, name)) for name in ("mean", "variance", "nuisance")):
        raise ValueError("Stale predicted readout differs from current carried physical fields")
    observation.validate({"law": "gaussian", "units": spec.units}, {})
    if observation.values.device != fresh.mean.device:
        raise ValueError("Native measurement and prediction must share a device")
    return fresh


@torch.no_grad()
def filter_native_observation(solver: NativeFrozenRHSSolver, state: NativeRHSContinuation,
                              prediction: NativePredictedReadout, arrived: NativeArrivedObservation) -> dict:
    fresh = validate_native_observation(solver, state, prediction, arrived)
    spec = fresh.acquisition.readout
    parameters = {"law": "gaussian", "units": spec.units, "mean": fresh.mean, "variance": fresh.variance}
    likelihood = retained_log_likelihood(arrived.observation, parameters, fresh.nuisance, {})
    updated = _normalized_log_weights(solver.current_log_weights(state)+likelihood.double())
    return _admit_native_observation(solver, state, fresh.acquisition, arrived, updated, likelihood)


@torch.no_grad()
def _admit_native_observation(solver: NativeFrozenRHSSolver, state: NativeRHSContinuation,
                             expected: NativeAcquisition, arrived: NativeArrivedObservation,
                             updated: torch.Tensor, likelihood: torch.Tensor) -> dict:
    """Commit one already-validated packet through the existing UID/revision authority."""
    observation = arrived.observation
    history = state._filter_binding.get(_ADMITTED_PAYLOADS, {})
    if not isinstance(history, dict) or observation.observation_uid in history:
        raise ValueError("Native accepted-payload history is inconsistent")
    # All potentially failing serialization occurs before the live filter update.
    accepted = dict(history, **{observation.observation_uid: arrived_payload_fingerprint(arrived)})
    solver.bind_filter_readout(state, expected.readout, expected.executed_regimen_uid)
    solver.update_filter_weights(state, updated, query_index=expected.query_index,
        parent_revision=expected.filtering_revision, observation_uid=observation.observation_uid)
    state._filter_binding[_ADMITTED_PAYLOADS] = accepted
    return {"status": "weight_filter_complete", "filtering_revision": state.filter_revision,
        "observation_uid": observation.observation_uid, "query_index": state.query_index,
        "log_likelihood": likelihood, "log_weights": solver.current_log_weights(state),
        "physical_state_changed": False, "resampled": False, "neural_context_updated": False}


_ADMITTED_PAYLOADS = "native_admitted_readout_payload_sha256"


def arrived_payload_fingerprint(arrived: NativeArrivedObservation) -> str:
    """Bind the accepted values/masks and fixed channel using existing serialization."""
    payload = {"acquisition": asdict(arrived.acquisition),
               "observation": observation_to_payload(arrived.observation)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def admitted_window_binding(solver: NativeFrozenRHSSolver, start: NativeRHSContinuation,
                            post_filter: NativeRHSContinuation,
                            arrivals: tuple[NativeArrivedObservation, ...]) -> dict:
    """Validate accepted historical readouts without filtering or changing any state."""
    solver._validate_continuation(start)
    solver._validate_continuation(post_filter)
    if any(not isinstance(row, NativeArrivedObservation) or not isinstance(row.observation, RetainedObservation)
           for row in arrivals):
        raise ValueError("Observation windows accept arrived readout packets, never trajectory targets")
    uids = tuple(row.observation.observation_uid for row in arrivals)
    if len(set(uids)) != len(uids) or any(uid in start.filter_observation_uids for uid in uids):
        raise ValueError("Observation window contains repeated or already-prior observation UIDs")
    if (post_filter.filter_observation_uids != start.filter_observation_uids+uids
            or post_filter.filter_revision != start.filter_revision+len(arrivals)):
        raise ValueError("Observation window is not the actual admitted filter-history suffix")
    if start._public_control is not None or post_filter._public_control is not None:
        return _executed_window_binding(solver,start,post_filter,arrivals,uids)
    source = lambda state: {key: value for key, value in state._filter_binding.items()
                            if key != _ADMITTED_PAYLOADS}
    if (source(start) != source(post_filter) or start._times != post_filter._times
            or start.program_context != post_filter.program_context):
        raise ValueError("Observation window source/support/program history changed")
    if post_filter.query_index != start.query_index+len(arrivals):
        raise ValueError("Observation window must end at its actual consecutive post-filter query")
    readout, executed = post_filter._filter_readout, post_filter._filter_executed_regimen_uid
    if (readout is None or executed is None
            or start._filter_readout is not None and start._filter_readout != readout
            or start._filter_executed_regimen_uid is not None and start._filter_executed_regimen_uid != executed):
        raise ValueError("Observation window has no unchanged admitted channel/program binding")
    history = post_filter._filter_binding.get(_ADMITTED_PAYLOADS, {})
    if not isinstance(history, dict):
        raise ValueError("Native accepted-payload history is inconsistent")
    context = solver.filter_context(start)
    missing = []
    for offset, arrived in enumerate(arrivals, 1):
        index = start.query_index+offset
        expected_context = dict(context, query_index=index, filtering_revision=start.filter_revision+offset-1,
            time_hours=start._times[index], source_time=float(start._guide.clock[index]))
        expected = _acquisition_for_context(expected_context, readout, executed,
                                           selected_coordinate=arrived.acquisition.selected_coordinate)
        arrived.acquisition.validate()
        _validate_arrived_payload(arrived, expected)
        uid = arrived.observation.observation_uid
        if uid not in history:
            missing.append(uid)
        elif history[uid] != arrived_payload_fingerprint(arrived):
            raise ValueError("Accepted native readout payload or acquisition changed under its UID")
    return {"status": "not_ready" if missing else "ready",
            "reason": "historical_admission_payload_binding_missing" if missing else None,
            "unbound_observation_uids": missing, "observation_uids": list(uids),
            "executed_regimen_uid": executed, "start_query_index": start.query_index,
            "end_query_index": post_filter.query_index,
            "query_indices": [row.acquisition.query_index for row in arrivals],
            "source_times": [row.acquisition.source_time for row in arrivals],
            "source_time_unit": context["source_time_unit"],
            "time_hours": [row.acquisition.time_hours for row in arrivals]}


def _executed_window_binding(solver,start,post,arrivals,uids):
    # Delayed imports avoid the existing observation-producer/source import cycle.
    from Libs.utils.model.native_future_controls import validate_execution_state, execution_constants_identity
    from Libs.utils.data.native_actuation import compose_actuation, executed_prefix
    if start._public_control is None or post._public_control is None:
        raise ValueError("H3 cannot mix an unbound history with an actual execution")
    validate_execution_state(start);validate_execution_state(post)
    if (execution_constants_identity(start) != execution_constants_identity(post)
            or post.query_index != start.query_index+len(arrivals)
            or start._filter_readout != post._filter_readout or post._filter_readout is None):
        raise ValueError("Executed H3 source/tuple/parameters/clock/readout changed")
    prior_history=tuple(row["identity"] for row in start._execution_history)
    if prior_history != tuple(row["identity"] for row in post._execution_history[:len(prior_history)]):
        raise ValueError("Executed H3 has no immutable actual switch-history prefix")
    records=post._execution_history[len(prior_history):]
    public,model_uid=start._public_control,start._execution_model_uid
    control_by_query={}
    initial_observations=len(post.filter_observation_uids)-post.filter_revision
    for record in records:
        query=record["query_index"]
        if (query in control_by_query or not start.query_index <= query <= post.query_index
                or record["request"].prefix != executed_prefix(public,query)
                or record["parent_model_uid"] != model_uid
                or record["public"] != compose_actuation(record["request"],post._execution_library)
                or record["filter_revision"] != start.filter_revision+query-start.query_index
                or record["observation_uids"] != post.filter_observation_uids[:initial_observations+record["filter_revision"]]):
            raise ValueError("Program identity change lacks a legal actual current-query execution record")
        control_by_query[query]=record
        public,model_uid=record["public"],record["model_uid"]
    history=post._filter_binding.get(_ADMITTED_PAYLOADS,{})
    context=solver.filter_context(start)
    public=start._public_control
    missing=[];executed=[]
    for offset,arrived in enumerate(arrivals,1):
        query=start.query_index+offset
        if query-1 in control_by_query:public=control_by_query[query-1]["public"]
        expected_context=dict(context,query_index=query,filtering_revision=start.filter_revision+offset-1,
            time_hours=start._times[query],source_time=float(start._guide.clock[query]),
            public_execution_uid=public.regimen_uid)
        expected=_acquisition_for_context(expected_context,post._filter_readout,public.regimen_uid,
                                         selected_coordinate=arrived.acquisition.selected_coordinate)
        arrived.acquisition.validate();_validate_arrived_payload(arrived,expected)
        uid=arrived.observation.observation_uid
        if uid not in history:missing.append(uid)
        elif history[uid] != arrived_payload_fingerprint(arrived):
            raise ValueError("Accepted execution observation payload or acquisition changed")
        executed.append(public.regimen_uid)
    return {"status":"not_ready" if missing else "ready",
        "reason":"historical_admission_payload_binding_missing" if missing else None,
        "unbound_observation_uids":missing,"observation_uids":list(uids),
        "executed_regimen_uid":executed[-1],"executed_regimen_uids":executed,
        "start_query_index":start.query_index,"end_query_index":post.query_index,
        "query_indices":[row.acquisition.query_index for row in arrivals],
        "source_times":[row.acquisition.source_time for row in arrivals],
        "source_time_unit":context["source_time_unit"],"time_hours":[row.acquisition.time_hours for row in arrivals],
        "execution_records":tuple(record for record in records if record["query_index"] < post.query_index)}
