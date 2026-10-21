"""Target-free CPU forecasts from the maintained frozen-primary CNS equations.

This is a model-side equation adapter. It never dispatches a source producer,
selects a hidden tuple, or calls NEURON/ModelDB. A Vm-only source with unavailable
gating state is explicitly unsupported by this pure-RHS path.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import math
import time

import torch
from torch import nn

from Libs.utils.model.controlled_cns_source import DYNAMIC
from Libs.utils.model.native_comparator_physics import FrozenNativePhysicsGuidance, auxiliary_step
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, MOLECULAR
from Libs.utils.model.native_future_controls import compile_future_controls, model_control_identity

from Libs.utils.model.native_future_controls import (
    execution_constants_identity, execution_library_identity, validate_execution_library, future_control_policy,
    validate_execution_state, execution_record_identity,
)
from Libs.utils.data.native_actuation import NativeObservationIdentity, NativeActuationRequest, executed_prefix, compose_actuation


def rk4_interval(rhs, state: torch.Tensor, left: float, right: float, steps: int) -> torch.Tensor:
    """Integrate one event-free interval without changing its physical clock."""
    if (not math.isfinite(left) or not math.isfinite(right) or right <= left
            or type(steps) is not int or steps < 1):
        raise ValueError("RK4 requires an increasing finite interval and positive integer step count")
    step = (right-left)/steps
    for index in range(steps):
        now = left+index*step
        k1 = rhs(now, state)
        k2 = rhs(now+step/2, state+step*k1/2)
        k3 = rhs(now+step/2, state+step*k2/2)
        k4 = rhs(now+step, state+step*k3)
        state = state+step*(k1+2*k2+2*k3+k4)/6
        if not torch.isfinite(state).all():
            raise FloatingPointError("Frozen-primary RK4 produced a nonfinite state")
    return state



def validate_physical_forecast(values: torch.Tensor, support: torch.Tensor, quantities: list[dict]) -> None:
    """Check supported physical domains without replacing or clipping values."""
    if values.shape != support.shape or support.dtype != torch.bool or values.shape[-1] != len(quantities):
        raise ValueError("Physical forecast values/support/profile shapes differ")
    if not torch.isfinite(values[support]).all():
        raise FloatingPointError("Nonfinite supported physical forecast")
    signed = torch.tensor([row["signed"] for row in quantities], device=values.device, dtype=torch.bool)
    maximum = values.new_tensor([float(row["maximum"]) if row.get("maximum") is not None else float("inf")
                                for row in quantities])
    for invalid, description in (((values < 0) & ~signed & support, "Negative"),
                                 ((values > maximum) & support, "Above-maximum")):
        selected = invalid.reshape(-1, len(quantities)).any(0).nonzero().flatten().tolist()
        if selected:
            names = ", ".join(quantities[index]["quantity_id"] for index in selected)
            raise FloatingPointError(description+" supported physical quantity: "+names)


@dataclass
class NativeRHSWorkLedger:
    """Actual work shared by forks; the planning ceiling becomes active once."""
    maximum: int
    spent: int = 0
    planning_active: bool = False


@dataclass
class NativeRHSContinuation:
    """Model carry; physical use requires a verified public execution binding."""

    cns_state: torch.Tensor
    auxiliary_state: torch.Tensor | None
    event_times: torch.Tensor | None
    event_doses: torch.Tensor | None
    applied_events: torch.Tensor | None
    program_context: tuple[dict, ...]
    _guide: FrozenNativePhysicsGuidance = field(repr=False)
    _owner: NativeFrozenRHSSolver = field(repr=False)
    _pk_parameters: dict = field(repr=False)
    _times: tuple[float, ...] = field(repr=False)
    _intervals: tuple[tuple[float, float, int], ...] = field(repr=False)
    _initial: torch.Tensor = field(repr=False)
    _validity: torch.Tensor = field(repr=False)
    _weights: torch.Tensor = field(repr=False)
    _maximum_rhs_work: int = field(repr=False)
    _log_weights: torch.Tensor = field(repr=False)
    _filter_binding: dict = field(repr=False)
    query_index: int = 0
    interval_index: int = 0
    rhs_python_calls: int = 0
    failed: bool = False
    filter_revision: int = 0
    filter_observation_uids: tuple[str, ...] = ()
    _filter_readout: object | None = field(default=None, repr=False)
    _filter_executed_regimen_uid: str | None = field(default=None, repr=False)
    _work_ledger: NativeRHSWorkLedger | None = field(default=None, repr=False)
    _registered_program_uids: tuple[str, ...] = field(default=(), repr=False)
    _registered_regimen_uids: tuple[str, ...] = field(default=(), repr=False)
    _registered_stimulus: torch.Tensor | None = field(default=None, repr=False)
    _event_doses_mg_per_kg: torch.Tensor | None = field(default=None, repr=False)
    _planning_origin: dict | None = field(default=None, repr=False)
    _control_identity: str | None = field(default=None, repr=False)
    _public_control: object | None = field(default=None, repr=False)
    _execution_model_uid: str | None = field(default=None, repr=False)
    _execution_library: tuple = field(default=(), repr=False)
    _execution_history: tuple = field(default=(), repr=False)
    _execution_constants_identity: str | None = field(default=None, repr=False)
    _execution_library_identity: str | None = field(default=None, repr=False)

    @property
    def current_time_hours(self) -> float:
        return self._times[self.query_index]


class NativeFrozenRHSSolver(nn.Module):
    """Forecast every inferred complete tuple under its own compiled actions."""

    prediction_kind = "frozen_primary_solver_posterior_pushforward"

    def __init__(self, config: dict, settings: dict):
        super().__init__()
        if settings.get("equation_authority") != "frozen_primary_training_law":
            raise ValueError("Native solver requires the frozen-primary equation authority")
        maximum = settings.get("maximum_step_hours")
        budget = settings.get("maximum_rhs_particle_action_evaluations")
        if (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0
                or type(budget) is not int or budget < 1):
            raise ValueError("Native solver requires an explicit step bound and positive RHS work budget")
        self.config = config
        self.maximum_step_hours = float(maximum)
        self.maximum_rhs_work = budget
        self.register_buffer("device_anchor", torch.empty(0), persistent=False)

    @torch.no_grad()
    def initialize_continuation(self, inputs, normalization) -> NativeRHSContinuation:
        """Bind public inputs once, without advancing any physical equation."""
        if inputs.normalization_id != normalization.identity:
            raise ValueError("Solver input/normalization identity mismatch")
        normalization.assert_domain(inputs.grid)
        guide = FrozenNativePhysicsGuidance(inputs, normalization, self.config)
        if guide.vm_only:
            raise ValueError("Pure-RHS forecast unavailable: Vm-only tuples lack frozen-primary auxiliary/gating state")
        # Subsequent calls consume this owned state, not mutable caller tensors.
        guide.grid = copy.deepcopy(guide.grid)
        guide.parameters = {key: value.clone() for key, value in guide.parameters.items()}
        guide.factors = {key: value.clone() for key, value in guide.factors.items()}
        if hasattr(guide, "stimulus"):
            guide.stimulus = guide.stimulus.clone()
        times = tuple(float(value) for value in guide.times)
        if len(times) < 2 or times[0] != 0 or any(not math.isfinite(t) for t in times):
            raise ValueError("Continuation needs the original finite clock starting at zero")
        n, nodes = guide.n, guide.initial.shape[1]*guide.initial.shape[2]
        initial = guide.initial.reshape(n, nodes, -1).double().clone()
        current = torch.zeros(n, nodes, len(DYNAMIC)+1, device=guide.device, dtype=torch.float64)
        current[..., :len(DYNAMIC)] = initial[..., guide.dynamic_indices]
        current[..., -1] = initial[..., guide.grid["species_id"].index("free_drug")]
        auxiliary = (torch.stack([initial[:, node, index] for node, index in
            zip(guide.auxiliary_nodes, guide.auxiliary_indices)], -1) if guide.auxiliary_indices else None)
        records = [record for sample in inputs.programs["records"] for particle in sample for record in particle]
        context = tuple(copy.deepcopy({key: row["action"].get(key) for key in
            ("action_uid", "route", "agent_set", "target_set", "schedule")}) for row in records)
        event_time = event_dose = event_dose_mg = applied = None
        parameters, events = {}, []
        if auxiliary is None:
            snapshots = [row["physical_program"]["inputs"]["run_snapshot"] for row in records]
            schedules = [row["action"]["schedule"] for row in records]
            events = sorted({float(t) for schedule in schedules for t in schedule["values"] if 0 <= t <= times[-1]})
            columns = max(len(schedule["values"]) for schedule in schedules)
            event_time = torch.full((n, columns), float("inf"), device=guide.device, dtype=torch.float64)
            event_dose = torch.zeros_like(event_time)
            event_dose_mg = torch.zeros_like(event_time)
            for row, (snapshot, schedule) in enumerate(zip(snapshots, schedules)):
                definition = snapshot["pk_definition"]
                molecular_dose = (definition["total_dose_mg_per_kg"]*definition["body_mass_kg"]
                                  *1e-3/definition["molecular_weight_g_per_mol"])
                width = len(schedule["values"])
                event_time[row, :width] = guide.tensor(schedule["values"])
                event_dose[row, :width] = molecular_dose*guide.tensor(schedule["dose_fractions"])
                event_dose_mg[row, :width] = definition["total_dose_mg_per_kg"]*guide.tensor(schedule["dose_fractions"])
            parameters = {name: guide.tensor([row["parameters"][name] for row in snapshots]) for name in
                ("volume_l", "delivery_fraction", "elimination_per_hour", "kd_mol_per_l", "bmax_mol_per_l")}
            if any(value.shape != (n, nodes) or not torch.isfinite(value).all() for value in parameters.values()):
                raise ValueError("Compiled long-CNS parameters do not cover each retained particle/action/site")
            if (parameters["volume_l"] <= 0).any() or (parameters["kd_mol_per_l"] <= 0).any():
                raise ValueError("Compiled long-CNS volume/affinity must remain positive")
            applied = torch.zeros_like(event_time, dtype=torch.bool)
        boundaries = sorted(set(times+tuple(events)))
        intervals = tuple((left, right, math.ceil((right-left)/self.maximum_step_hours))
                          for left, right in zip(boundaries, boundaries[1:]))
        validity = inputs.validity.reshape(guide.b, guide.j, guide.a, len(times), *guide.initial.shape[1:]).clone()
        support = inputs.posterior.support
        binding = {"source_uid": support.source_uid, "source_snapshot_ref": support.source_snapshot_ref,
            "role": support.role, "support_uid": support.support_uid, "family_uid": guide.family,
            "sample_uids": tuple(inputs.sample_uids), "particle_uids": tuple(tuple(row) for row in inputs.particle_uids),
            "regimen_uids": tuple(inputs.regimen_uids), "origin_posterior_uid": inputs.posterior_uid,
            "region_ids": tuple(guide.grid["region_id"]), "compartment_ids": tuple(guide.grid["compartment_id"]),
            "species_ids": tuple(guide.grid["species_id"])}
        state = NativeRHSContinuation(current, auxiliary, event_time, event_dose, applied, context,
            guide, self, parameters, times, intervals, initial, validity, inputs.weights.clone(), self.maximum_rhs_work,
            inputs.posterior.log_weights.double().clone(), binding,
            filter_observation_uids=tuple(inputs.posterior.observation_uids),
            _work_ledger=NativeRHSWorkLedger(self.maximum_rhs_work),
            _registered_program_uids=tuple(row["program_uid"] for row in records),
            _registered_regimen_uids=tuple(inputs.regimen_uids),
            _registered_stimulus=guide.stimulus.clone() if hasattr(guide, "stimulus") else None,
            _event_doses_mg_per_kg=event_dose_mg)
        state._control_identity = model_control_identity(state)
        return state

    def _validate_continuation(self, state: NativeRHSContinuation) -> None:
        if not isinstance(state, NativeRHSContinuation) or state._owner is not self:
            raise ValueError("Continuation belongs to a different solver")
        if state.failed:
            raise ValueError("Failed continuation cannot resume an incomplete numerical interval")
        guide = state._guide
        if (type(state.query_index) is not int or not 0 <= state.query_index < len(state._times)
                or type(state.interval_index) is not int or not 0 <= state.interval_index <= len(state._intervals)):
            raise ValueError("Continuation original clock position is invalid")
        expected_intervals = sum(right <= state.current_time_hours for _, right, _ in state._intervals)
        expected_calls = 4*sum(steps for _, _, steps in state._intervals[:state.interval_index])
        if state.interval_index != expected_intervals or state.rhs_python_calls != expected_calls:
            raise ValueError("Continuation clock and cumulative RHS history disagree")
        expected_shape = (guide.n, state._initial.shape[1], len(DYNAMIC)+1)
        if (state.cns_state.shape != expected_shape or state.cns_state.dtype != torch.float64
                or state.cns_state.device != guide.device):
            raise ValueError("Continuation CNS state shape, dtype or device changed")
        if not torch.isfinite(state.cns_state).all() or (state.cns_state < 0).any():
            raise FloatingPointError("Continuation CNS state must remain finite and nonnegative")
        auxiliary = state.auxiliary_state
        if guide.auxiliary_indices:
            if (auxiliary is None or auxiliary.shape != (guide.n, len(guide.auxiliary_indices))
                    or auxiliary.dtype != torch.float64 or auxiliary.device != guide.device):
                raise ValueError("Continuation auxiliary state shape, dtype or device changed")
            quantities = [guide.grid["quantity_profile"]["quantities"][i] for i in guide.auxiliary_indices]
            validate_physical_forecast(auxiliary, torch.ones_like(auxiliary, dtype=torch.bool), quantities)
        elif auxiliary is not None:
            raise ValueError("Long-CNS continuation cannot acquire an auxiliary state")
        if state.event_times is not None:
            finite = torch.isfinite(state.event_times)
            if (state.event_times.ndim != 2 or state.event_times.shape[0] != guide.n
                    or state.event_doses.shape != state.event_times.shape
                    or state._event_doses_mg_per_kg.shape != state.event_times.shape
                    or torch.isnan(state.event_times).any() or torch.isneginf(state.event_times).any()
                    or (state.event_times[finite] < 0).any()
                    or not torch.isfinite(state.event_doses).all() or (state.event_doses < 0).any()
                    or not torch.isfinite(state._event_doses_mg_per_kg).all() or (state._event_doses_mg_per_kg < 0).any()
                    or state.event_doses[~finite].any() or state._event_doses_mg_per_kg[~finite].any()
                    or any(not (row[mask].diff() > 0).all() for row, mask in zip(state.event_times, finite))):
                raise ValueError("Continuation dose events/units/order are invalid")
            expected = state.event_times <= state.current_time_hours
            if state.query_index == 0:
                expected = torch.zeros_like(expected)
            if (state.applied_events is None or state.applied_events.dtype != torch.bool
                    or not torch.equal(state.applied_events, expected)):
                raise ValueError("Continuation applied dose history differs from its fixed program and clock")

    def fork_continuation(self, state: NativeRHSContinuation) -> NativeRHSContinuation:
        """Copy mutable carry/history while retaining its existing action histories.

        Cloning the dynamic tensors preserves their residual-network graph. The
        frozen guide is shared read-only; no coefficients or original inputs are
        made trainable. This does not compile a new action or reset the clock.
        """
        self._validate_continuation(state)
        def clone_optional(value):
            return None if value is None else value.clone()
        return replace(state,
            cns_state=state.cns_state.clone(),
            auxiliary_state=clone_optional(state.auxiliary_state),
            event_times=clone_optional(state.event_times),
            event_doses=clone_optional(state.event_doses),
            _event_doses_mg_per_kg=clone_optional(state._event_doses_mg_per_kg),
            applied_events=clone_optional(state.applied_events),
            program_context=copy.deepcopy(state.program_context),
            _pk_parameters={key: value.detach().clone() for key, value in state._pk_parameters.items()},
            _initial=state._initial.detach().clone(), _validity=state._validity.clone(),
            _weights=state._weights.detach().clone(), _log_weights=state._log_weights.detach().clone(),
            _filter_binding=copy.deepcopy(state._filter_binding))

    @torch.no_grad()
    def prepare_future_control_branch(self, state: NativeRHSContinuation, inputs, normalization, *,
                               executed_regimen_uid: str, candidate_regimen_uids=None,
                               require_delivered_dose: bool = False, public_control=None) -> NativeRHSContinuation:
        """Compile controls without advancing or reserving an entire forecast.

        Complete-task budgeting belongs to the caller. Existing execution callers
        retain the full-library preflight below; every branch remains unactivated
        and ineligible for actual observation filtering.
        """
        self._validate_continuation(state)
        if inputs.normalization_id != normalization.identity or normalization.identity != state._guide.normalization.identity:
            raise ValueError("Future-control fixed normalization identity changed")
        control = compile_future_controls(state, inputs, self.config,
            executed_regimen_uid=executed_regimen_uid, candidate_regimen_uids=candidate_regimen_uids,
            require_delivered_dose=require_delivered_dose, public_control=public_control)
        guide, executed, count = state._guide, control["executed_index"], len(control["candidate_indices"])
        def select(value):
            if value is None:
                return None
            return value.reshape(guide.b, guide.j, guide.a, *value.shape[1:])[:, :, executed:executed+1].expand(
                guide.b, guide.j, count, *value.shape[1:]).reshape(guide.b*guide.j*count, *value.shape[1:]).clone()
        new_guide = guide.fork_executed_columns(executed, count)
        if control["stimulus"] is not None:
            new_guide.stimulus = control["stimulus"]
        metadata = control["metadata"]
        context = tuple({"model_regimen_uid": metadata["model_regimen_uids"][ci],
            "source_candidate_regimen_uid": metadata["source_candidate_regimen_uids"][ci],
            "executed_prefix": copy.deepcopy(state.program_context[ji*guide.a+executed])}
            for ji in range(guide.j) for ci in range(count))
        future_events = (() if control["event_times"] is None else tuple(float(t) for t in
            control["event_times"].flatten().tolist() if state.current_time_hours < t <= state._times[-1]))
        boundaries = sorted({state.current_time_hours, *state._times[state.query_index+1:], *future_events})
        remaining = tuple((left, right, math.ceil((right-left)/self.maximum_step_hours))
                          for left, right in zip(boundaries, boundaries[1:]))
        branch = replace(state, cns_state=select(state.cns_state), auxiliary_state=select(state.auxiliary_state),
            event_times=control["event_times"], event_doses=control["event_doses"],
            _event_doses_mg_per_kg=control["event_doses_mg_per_kg"], applied_events=control["applied_events"],
            program_context=context, _guide=new_guide,
            _pk_parameters={key: select(value) for key, value in state._pk_parameters.items()},
            _initial=select(state._initial), _validity=state._validity[:, :, executed:executed+1].expand(
                guide.b, guide.j, count, *state._validity.shape[3:]).clone(),
            _weights=state._weights.clone(), _log_weights=state._log_weights.clone(),
            _intervals=state._intervals[:state.interval_index]+remaining,
            _filter_binding=copy.deepcopy(state._filter_binding), _planning_origin=metadata,
            _public_control=None, _execution_model_uid=None)
        branch._filter_binding["regimen_uids"] = tuple(metadata["model_regimen_uids"])
        branch._control_identity = model_control_identity(branch)
        metadata["static_control_sha256"] = branch._control_identity
        self._validate_continuation(branch)
        return branch

    @torch.no_grad()
    def branch_future_controls(self, state: NativeRHSContinuation, inputs, normalization, *,
                               executed_regimen_uid: str, candidate_regimen_uids=None,
                               require_delivered_dose: bool = False, public_control=None) -> NativeRHSContinuation:
        """Preserve the full remaining-library forecast preflight for existing callers."""
        branch = self.prepare_future_control_branch(state, inputs, normalization,
            executed_regimen_uid=executed_regimen_uid, candidate_regimen_uids=candidate_regimen_uids,
            require_delivered_dose=require_delivered_dose, public_control=public_control)
        ledger = state._work_ledger
        planned = self.forecast_work(branch, query_count=len(branch._times))
        ceiling = ledger.maximum if ledger.planning_active else state._maximum_rhs_work
        if ledger.spent+planned > ceiling:
            raise ValueError("Complete future-control library exceeds the shared actual RHS work budget")
        if not ledger.planning_active:
            ledger.maximum, ledger.planning_active = ceiling, True
        return branch

    def forecast_work(self, state: NativeRHSContinuation, *, query_count: int) -> int:
        """Exact forward RHS work for this compiled interval union and action width."""
        self._validate_continuation(state)
        if type(query_count) is not int or not state.query_index+1 <= query_count <= len(state._times):
            raise ValueError("RHS work requires a current or later original query count")
        final_time = state._times[query_count-1]
        stop = sum(right <= final_time for _, right, _ in state._intervals)
        return 4*state._guide.n*sum(steps for _, _, steps in state._intervals[state.interval_index:stop])

    def bind_public_execution(self, state, inputs, library, initial_control, observation_identity):
        """Bind one original public execution at t0 without changing a physical tensor."""
        self._validate_continuation(state)
        if state.query_index != 0 or state._planning_origin is not None or state._public_control is not None:
            raise ValueError("Execution identity must be bound once at the original initial query")
        if future_control_policy(self.config)["activation"] != "requires_verified_public_execution_binding":
            raise ValueError("Legacy model-planning-only policy does not authorize public execution binding")
        expected=NativeObservationIdentity.from_context(self.filter_context(state))
        if observation_identity != expected or initial_control not in library:
            raise ValueError("Controller/source observation identity or initial public action differs")
        validate_execution_library(state,inputs,library,self.config)
        uid=initial_control.regimen_uid
        if uid != state._filter_executed_regimen_uid:
            raise ValueError("Initial physical action differs from the actual admitted model column")
        constants, controls=execution_constants_identity(state),execution_library_identity(state)
        state._public_control, state._execution_model_uid = initial_control, uid
        state._execution_library=library
        state._execution_constants_identity, state._execution_library_identity = constants,controls

    def _select_column(self, state, index):
        """Select a model column while preserving ambient state gradients."""
        guide=state._guide
        def select(value):
            if value is None:return None
            return value.reshape(guide.b,guide.j,guide.a,*value.shape[1:])[:,:,index:index+1].reshape(
                guide.b*guide.j,*value.shape[1:]).clone()
        new_guide=guide.fork_executed_columns(index,1)
        if hasattr(guide,"stimulus"):
            new_guide.stimulus=select(guide.stimulus)
        uid=state._filter_binding["regimen_uids"][index]
        origin=copy.deepcopy(state._planning_origin)
        if origin is not None:
            origin["model_regimen_uids"]=[uid]
            origin["source_candidate_regimen_uids"]=[origin["source_candidate_regimen_uids"][index]]
        result=replace(state,cns_state=select(state.cns_state),auxiliary_state=select(state.auxiliary_state),
            event_times=select(state.event_times),event_doses=select(state.event_doses),
            applied_events=select(state.applied_events),_event_doses_mg_per_kg=select(state._event_doses_mg_per_kg),
            _guide=new_guide,_initial=select(state._initial),
            _validity=state._validity[:,:,index:index+1].clone(),
            _pk_parameters={key:select(value) for key,value in state._pk_parameters.items()},
            program_context=tuple(copy.deepcopy(state.program_context[ji*guide.a+index]) for ji in range(guide.j)),
            _filter_binding=dict(copy.deepcopy(state._filter_binding),regimen_uids=(uid,)),_planning_origin=origin)
        if result.event_times is not None:
            width=int(torch.isfinite(result.event_times).sum(-1).max())
            for name in ("event_times","event_doses","_event_doses_mg_per_kg","applied_events"):
                setattr(result,name,getattr(result,name)[:,:width].clone())
        result._control_identity=model_control_identity(result)
        if origin is not None:origin["static_control_sha256"]=result._control_identity
        return result

    def prepare_initial_control(self, state, inputs, library, initial_control, readout):
        """Select a pristine public t0 column without executing or applying its events."""
        self._validate_continuation(state)
        if (state.query_index or state.interval_index or state.rhs_python_calls
                or state._public_control is not None or state._planning_origin is not None
                or state.filter_revision or initial_control not in library):
            raise ValueError("Initial selection requires the pristine original public model origin")
        validate_execution_library(state, inputs, library, self.config)
        if future_control_policy(self.config)["activation"] != "requires_verified_public_execution_binding":
            raise ValueError("Initial selection requires verified public execution activation")
        expected = self.filter_context(state)
        if (readout.source_uid != expected["source_uid"]
                or readout.source_snapshot_ref != expected["source_snapshot_ref"]
                or readout.role != expected["role"] or readout.family_uid != expected["family_uid"]):
            raise ValueError("Initial selection readout differs from original public support")
        index = state._registered_regimen_uids.index(initial_control.regimen_uid)
        selected = self._select_column(state, index)
        selected._execution_library = library
        selected._execution_constants_identity = execution_constants_identity(state)
        selected._execution_library_identity = execution_library_identity(state)
        selected._filter_readout = readout
        selected._filter_executed_regimen_uid = None
        selected._planning_origin = {"model_regimen_uids": [initial_control.regimen_uid],
            "source_candidate_regimen_uids": [initial_control.regimen_uid],
            "initial_public_selection": True}
        return selected

    def install_initial_control(self, origin, selected, inputs, library, initial_control, readout):
        """Install only an unchanged unadvanced initial candidate; share all spent work."""
        expected = self.prepare_initial_control(origin, inputs, library, initial_control, readout)
        self._validate_continuation(selected)
        if (selected.query_index or selected.interval_index or selected.rhs_python_calls
                or selected._work_ledger is not origin._work_ledger
                or selected._planning_origin != expected._planning_origin
                or selected._public_control is not None or selected._execution_model_uid is not None
                or execution_constants_identity(selected) != execution_constants_identity(expected)
                or model_control_identity(selected) != model_control_identity(expected)
                or selected._intervals != expected._intervals
                or selected._filter_binding != expected._filter_binding
                or selected.filter_revision != expected.filter_revision
                or selected.filter_observation_uids != expected.filter_observation_uids):
            raise ValueError("Advanced or changed initial forecast cannot become physical execution")
        for name in ("cns_state", "auxiliary_state", "event_times", "event_doses", "applied_events",
                     "_weights", "_log_weights", "_event_doses_mg_per_kg"):
            left, right = getattr(selected, name), getattr(expected, name)
            if (left is None) != (right is None) or left is not None and not torch.equal(left, right):
                raise ValueError("Initial forecast differs from its pristine public action column")
        return replace(selected, _public_control=initial_control,
            _execution_model_uid=initial_control.regimen_uid, _planning_origin=None,
            _filter_executed_regimen_uid=initial_control.regimen_uid)

    def prepare_public_execution(self, live, branch, inputs, request, *, hypothetical_parent=None):
        """Validate the actual unadvanced selected branch before any actuator effect."""
        if type(request) is not NativeActuationRequest:
            raise ValueError("Activation requires a typed parameter-free public actuation request")
        self._validate_continuation(live);self._validate_continuation(branch)
        if hypothetical_parent is None:
            validate_execution_state(live)
            parent_public, parent_uid = live._public_control, live._execution_model_uid
            if parent_public is None:
                raise ValueError("Activation requires a bound live carry and an unactivated planning branch")
        else:
            # Residual forecasting separately binds this model-only public plan
            # to its immutable numeric conditioning; this does not admit execution.
            parent_public, parent_uid = hypothetical_parent
            if (live._public_control is not None or live._planning_origin is None
                    or parent_uid not in live._filter_binding["regimen_uids"]
                    or parent_public.source_uid != live._filter_binding["source_uid"]
                    or parent_public.source_snapshot_ref != live._filter_binding["source_snapshot_ref"]
                    or parent_public.family_uid != live._guide.family):
                raise ValueError("Hypothetical control requires its own unexecuted public prefix")
        if branch._planning_origin is None or branch._public_control is not None:
            raise ValueError("Activation requires an unactivated planning branch")
        origin=branch._planning_origin
        if (request.prefix != executed_prefix(parent_public,live.query_index)
                or branch.query_index != live.query_index or origin["original_query_index"] != live.query_index
                or origin.get("origin_filter_revision") != live.filter_revision
                or tuple(origin.get("origin_observation_uids",())) != live.filter_observation_uids
                or origin["executed_regimen_uid"] != parent_uid
                or branch._work_ledger is not live._work_ledger
                or model_control_identity(branch) != origin["static_control_sha256"]):
            raise ValueError("Stale, forecasted or foreign branch cannot become the actual execution")
        public=compose_actuation(request,live._execution_library)
        candidates=origin["source_candidate_regimen_uids"]
        if request.actuation.regimen_uid not in candidates:
            raise ValueError("Selected public action is not present in this model branch")
        index=candidates.index(request.actuation.regimen_uid)
        selected=self._select_column(branch,index)
        current=self._select_column(live,live._filter_binding["regimen_uids"].index(parent_uid))
        if (execution_constants_identity(selected) != live._execution_constants_identity
                or selected.filter_revision != live.filter_revision
                or selected.filter_observation_uids != live.filter_observation_uids
                or selected._filter_readout != live._filter_readout
                or {key:value for key,value in selected._filter_binding.items() if key != "regimen_uids"}
                   != {key:value for key,value in live._filter_binding.items() if key != "regimen_uids"}):
            raise ValueError("Selected branch changed its source/tuple/static or filtering origin")
        for name in ("cns_state","auxiliary_state","_weights","_log_weights"):
            left,right=getattr(selected,name),getattr(current,name)
            if (left is None) != (right is None) or left is not None and not torch.equal(left,right):
                raise ValueError("Selected branch does not carry the actual current executed state")
        expected=compile_future_controls(live,inputs,self.config,executed_regimen_uid=parent_uid,
                                        candidate_regimen_uids=[request.actuation.regimen_uid],public_control=parent_public)
        for attribute,key in (("event_times","event_times"),("event_doses","event_doses"),
                ("_event_doses_mg_per_kg","event_doses_mg_per_kg"),("applied_events","applied_events")):
            value=getattr(selected,attribute);check=expected[key]
            if (value is None) != (check is None) or value is not None and not torch.equal(value,check):
                raise ValueError("Selected model events differ from the actual-prefix public candidate")
        if expected["stimulus"] is not None and not torch.equal(selected._guide.stimulus,expected["stimulus"]):
            raise ValueError("Selected model drive differs from its admitted original compiler representation")
        if selected._filter_binding["regimen_uids"] != tuple(expected["metadata"]["model_regimen_uids"]):
            raise ValueError("Selected branch identity differs from its actual current origin")
        return selected,public

    def install_execution(self, selected, record, prior):
        """Install a prepared, content-checked execution without a physical advance."""
        return replace(selected,_public_control=record["public"],_execution_model_uid=record["model_uid"],
            _execution_history=prior._execution_history+(record,),
            _filter_executed_regimen_uid=record["public"].regimen_uid,
            _filter_binding=copy.deepcopy(selected._filter_binding))

    def replay_execution(self, cursor, record):
        """Replay only exogenous controls on this differentiable predicted cursor."""
        if record["identity"] != execution_record_identity(record) or cursor.query_index != record["query_index"]:
            raise ValueError("Historical execution content or replay clock changed")
        parent=record["parent_model_uid"]
        if parent not in cursor._filter_binding["regimen_uids"]:
            raise ValueError("Historical execution has a different actual model parent")
        selected=self._select_column(cursor,cursor._filter_binding["regimen_uids"].index(parent))
        template=record["template"]
        binding=copy.deepcopy(selected._filter_binding)
        binding["regimen_uids"]=(record["model_uid"],)
        # Carry values are replay predictions, never the saved live switch state.
        result=replace(template,cns_state=selected.cns_state,auxiliary_state=selected.auxiliary_state,
            _weights=selected._weights,_log_weights=selected._log_weights,
            query_index=selected.query_index,interval_index=selected.interval_index,
            rhs_python_calls=selected.rhs_python_calls,_work_ledger=selected._work_ledger,
            filter_revision=selected.filter_revision,filter_observation_uids=selected.filter_observation_uids,
            _filter_binding=binding,_filter_readout=selected._filter_readout,
            _intervals=selected._intervals[:selected.interval_index]+template._intervals[template.interval_index:])
        return self.install_execution(result,record,selected)

    def _pk(self, state: NativeRHSContinuation, now: float, *, predose: bool = False):
        parameters = state._pk_parameters
        lag = now-state.event_times
        available = state.applied_events & (lag >= 0)
        safe_lag = torch.where(available, lag, 0.)
        amount = (state.event_doses[..., None]*torch.exp(-safe_lag[..., None]
            *parameters["elimination_per_hour"][:, None])*available[..., None]).sum(1)
        free = amount*parameters["delivery_fraction"]/parameters["volume_l"]
        if predose and now == 0:
            free = torch.zeros_like(free)
        occupancy = free/(free+parameters["kd_mol_per_l"])
        return free, occupancy, parameters["bmax_mol_per_l"]*occupancy

    def _terms(self, state: NativeRHSContinuation, current: torch.Tensor, now: float):
        guide, auxiliary = state._guide, state.auxiliary_state
        n, nodes = guide.n, state._initial.shape[1]
        x = current[..., :len(DYNAMIC)]
        if not torch.isfinite(current).all() or (x < 0).any() or (current[..., -1] < 0).any():
            raise FloatingPointError("Frozen-primary integration left its nonnegative CNS state domain")
        if auxiliary is None:
            _, occupancy, _ = self._pk(state, now)
        else:
            occupancy = torch.zeros(n, nodes, device=guide.device, dtype=torch.float64)
            if guide.family == MOLECULAR:
                occupancy[:, guide.ca1] = auxiliary[:, 1]
        production, removal, paired = (value[:, 0] for value in guide.reaction(x[:, None], occupancy[:, None]))
        if guide.family in {ELECTRICAL, SYNAPTIC}:
            production = production.clone()
            rates, coupling = guide.source["base_rates_per_hour"], guide.coupling
            if guide.family == ELECTRICAL:
                production[:, guide.ca1, DYNAMIC.index("calcium_concentration")] += (
                    rates["calcium_concentration_relaxation"]*coupling["gain"]*auxiliary[:, 1]
                    /coupling["calcium_reference_umol_per_l"])
                production[:, guide.ca1, DYNAMIC.index("spike_rate")] += (
                    rates["spike_rate_relaxation"]*coupling["gain"]*auxiliary[:, 2]/coupling["spike_reference_hz"])
            else:
                production[:, guide.ca3, DYNAMIC.index("calcium_concentration")] += (
                    rates["calcium_concentration_relaxation"]*coupling["gain"]*auxiliary[:, 0]
                    /coupling["calcium_reference_umol_per_l"])
                production[:, guide.ca1, DYNAMIC.index("membrane_voltage")] += (
                    rates["membrane_voltage_relaxation"]*coupling["gain"]*auxiliary[:, 3]
                    /coupling["current_reference_pa"])
        diffusion, flux = guide.transport(x[:, None])
        return production-removal+paired+diffusion[:, 0], removal, flux[:, 0]

    def _derivative(self, state: NativeRHSContinuation, now: float, current: torch.Tensor):
        state.rhs_python_calls += 1
        state._work_ledger.spent += state._guide.n
        guide, auxiliary = state._guide, state.auxiliary_state
        n, nodes = guide.n, state._initial.shape[1]
        dynamics, _, _ = self._terms(state, current, now)
        free_rhs = torch.zeros(n, nodes, device=guide.device, dtype=torch.float64)
        if guide.family == MOLECULAR:
            free = current[..., -1].clone()
            free[:, guide.ca1] = auxiliary[:, 0]*1e-9
            transport, _ = guide.transport(free[:, None, :, None], guide.tensor([1.]))
            free_rhs = transport[:, 0, :, 0]-guide.elimination*free
            free_rhs[:, guide.ca1] = 0.
        return torch.cat((dynamics, free_rhs[..., None]), -1)

    def _field(self, state: NativeRHSContinuation):
        guide, auxiliary, current = state._guide, state.auxiliary_state, state.cns_state
        initial, now = state._initial, state.current_time_hours
        n, nodes = guide.n, initial.shape[1]
        result = torch.zeros_like(initial)
        result[..., guide.dynamic_indices] = current[..., :len(DYNAMIC)]
        free_index = guide.grid["species_id"].index("free_drug")
        bound_index = guide.grid["species_id"].index("target_bound_drug")
        if auxiliary is None:
            free, _, bound = self._pk(state, now, predose=True)
            result[..., free_index], result[..., bound_index] = free, bound
        elif guide.family == MOLECULAR:
            result[..., free_index] = current[..., -1]
            result[:, guide.ca1, free_index] = auxiliary[:, 0]*1e-9
            result[:, guide.ca1, bound_index] = guide.bmax[:, guide.ca1]*auxiliary[:, 1]
        _, removal, flux = self._terms(state, current, now)
        result[..., guide.grid["species_id"].index("clearance_flux")] = (guide.volumes[..., None]*removal).sum(-1)
        absolute = torch.zeros(n, nodes, device=guide.device, dtype=torch.float64)
        amount = flux.abs().sum(-1)/2
        absolute.index_add_(1, guide.edges[:, 0], amount)
        absolute.index_add_(1, guide.edges[:, 1], amount)
        result[..., guide.grid["species_id"].index("vascular_perivascular_transport")] = absolute
        if auxiliary is not None:
            for column, (node, species) in enumerate(zip(guide.auxiliary_nodes, guide.auxiliary_indices)):
                result[:, node, species] = auxiliary[:, column]
        return result

    @torch.no_grad()
    def advance_continuation(self, state: NativeRHSContinuation, *, query_count: int) -> dict:
        """Preserved inference API: mutate the owned carry without an autograd graph."""
        return self._advance_continuation(state, query_count=query_count)

    def advance_continuation_functional(self, state: NativeRHSContinuation, *,
                                       query_count: int) -> tuple[NativeRHSContinuation, dict]:
        """Advance a private branch, preserving ambient gradients through its state."""
        branch = self.fork_continuation(state)
        return branch, self._advance_continuation(branch, query_count=query_count)

    def _advance_continuation(self, state: NativeRHSContinuation, *, query_count: int) -> dict:
        """Advance to an absolute original prefix count and return only new fields.

        The first advance also returns t=0. Later calls never replay earlier
        intervals. Query boundaries use the unchanged auxiliary/event ordering.
        """
        started = time.monotonic()
        self._validate_continuation(state)
        guide = state._guide
        total_times, count = len(state._times), query_count
        if type(count) is not int or not state.query_index+1 < count <= total_times:
            raise ValueError("Continuation requires a later original query count within its declared horizon")
        final_time = state._times[count-1]
        stop_interval = sum(right <= final_time for _, right, _ in state._intervals)
        planned = 4*sum(steps for _, _, steps in state._intervals[state.interval_index:stop_interval])
        ledger = state._work_ledger
        required = ledger.spent+planned*guide.n if ledger.planning_active else (state.rhs_python_calls+planned)*guide.n
        ceiling = ledger.maximum if ledger.planning_active else state._maximum_rhs_work
        if required > ceiling:
            raise ValueError("Complete posterior/action forecast exceeds the declared RHS work budget")
        start_index = 0 if state.query_index == 0 else state.query_index+1
        old_query, old_calls = state.query_index, state.rhs_python_calls
        outputs = [state._initial.clone()] if start_index == 0 else []
        try:
            while state.interval_index < stop_interval:
                left, right, steps = state._intervals[state.interval_index]
                if state.applied_events is not None:
                    state.applied_events = state.applied_events | (state.event_times <= left)
                if guide.family == MOLECULAR:
                    current = state.cns_state.clone()
                    current[:, guide.ca1, -1] = state.auxiliary_state[:, 0]*1e-9
                    state.cns_state = current
                state.cns_state = rk4_interval(lambda now, value: self._derivative(state, now, value),
                                              state.cns_state, left, right, steps)
                state.interval_index += 1
                if right == state._times[state.query_index+1]:
                    index = state.query_index+1
                    if state.auxiliary_state is not None:
                        stimulus_index = index-1 if guide.bind_source_intervals and guide.family != MOLECULAR else index
                        state.auxiliary_state = auxiliary_step(guide.family, state.auxiliary_state, guide.parameters,
                            guide.clock[index]-guide.clock[index-1], guide.stimulus[:, stimulus_index])
                        if not torch.isfinite(state.auxiliary_state).all():
                            raise FloatingPointError("Frozen-primary auxiliary forecast is nonfinite")
                    state.query_index = index
                    if state.applied_events is not None:
                        state.applied_events = state.applied_events | (state.event_times <= right)
                    current_field = self._field(state)
                    # Keep the carried coordinate synchronized; readout-only calls are pure.
                    free_index = guide.grid["species_id"].index("free_drug")
                    current = state.cns_state.clone()
                    current[..., -1] = current_field[..., free_index]
                    state.cns_state = current
                    outputs.append(current_field)
            calls = state.rhs_python_calls-old_calls
            if len(outputs) != count-start_index or calls != planned:
                raise RuntimeError("Solver clock or executed RHS accounting differs from the declared prefix")
            value = torch.stack(outputs, 1).reshape(guide.b, guide.j, guide.a,
                len(outputs), *guide.initial.shape[1:])
            mask = state._validity[:, :, :, start_index:count]
            validate_physical_forecast(value, mask, guide.grid["quantity_profile"]["quantities"])
            value = torch.where(mask, value, 0.)
        except (ValueError, RuntimeError, FloatingPointError):
            state.failed = True
            raise
        updates = state.query_index-old_query
        return {"particle_fields": value, "particle_weights": state._weights,
            "execution": {"equation_authority": "frozen_primary_training_law",
                "source_provider_calls": 0, "target_access": False, "hidden_selector_access": False,
                "source_variant_equation_selection": False, "trainable_transition_parameters": 0,
                "rhs_python_calls": calls, "rhs_tuple_action_state_evaluations": calls*guide.n,
                "rhs_budget_unit": "tuple_action_state_evaluations",
                "auxiliary_tuple_action_updates": updates*guide.n if state.auxiliary_state is not None else 0,
                "quantity_readout_tuple_action_evaluations": updates*guide.n,
                "declared_query_count": total_times, "forecast_query_count": count,
                "returned_query_count": len(outputs), "start_query_index": start_index,
                "horizon_complete": count == total_times, "physical_time_unit": "hour",
                "integration_dtype": "float64", "numerical_accuracy_qualified": False,
                "time_values_hours": list(state._times[start_index:count]),
                "original_query_time_unit": guide.grid["query_time"]["unit"],
                "splitting_scheme": ("primary_discrete_auxiliary_with_left_held_CNS_RK4"
                    if state.auxiliary_state is not None else "dose_event_split_continuous_primary_PK_CNS_RK4"),
                "cumulative_rhs_python_calls": state.rhs_python_calls,
                "shared_actual_rhs_tuple_action_evaluations": state._work_ledger.spent,
                "shared_planning_budget_active": state._work_ledger.planning_active,
                "elapsed_seconds": time.monotonic()-started}}

    def current_particle_fields(self, state: NativeRHSContinuation) -> torch.Tensor:
        """Read current predicted fields without advancing or replacing physical state."""
        self._validate_continuation(state)
        value = state._initial.clone() if state.query_index == 0 else self._field(state)
        guide = state._guide
        value = value.reshape(guide.b, guide.j, guide.a, *guide.initial.shape[1:])
        mask = state._validity[:, :, :, state.query_index]
        validate_physical_forecast(value, mask, guide.grid["quantity_profile"]["quantities"])
        return torch.where(mask, value, 0.)

    def independent_state_mask(self, state: NativeRHSContinuation) -> torch.Tensor:
        """Correctable CNS/neural states; excludes all PK and molecular auxiliary states."""
        self._validate_continuation(state)
        guide = state._guide
        mask = torch.zeros_like(state._initial, dtype=torch.bool)
        mask[..., guide.dynamic_indices] = True
        # Native electrical spike and synaptic release probability are algebraic
        # readouts. Molecular signal/response feed exposure and stay mechanistic.
        columns = {ELECTRICAL: (0, 1), SYNAPTIC: (0, 2, 3)}.get(guide.family, ())
        for column in columns:
            mask[:, guide.auxiliary_nodes[column], guide.auxiliary_indices[column]] = True
        mask = mask.reshape(guide.b, guide.j, guide.a, *guide.initial.shape[1:])
        return mask & state._validity[:, :, :, state.query_index]

    def _replace_independent_state(self, state: NativeRHSContinuation, fields: torch.Tensor) -> torch.Tensor:
        """Atomically replace allowed coordinates and reconstruct all derived readouts."""
        self._validate_continuation(state)
        guide = state._guide
        expected = (guide.b, guide.j, guide.a, *guide.initial.shape[1:])
        if fields.shape != expected:
            raise ValueError("Independent-state injection must retain complete tuple/action/grid axes")
        values = fields.reshape_as(state._initial).double()
        allowed = self.independent_state_mask(state).reshape_as(state._initial)
        candidate_cns = state.cns_state.clone()
        candidate_cns[..., :len(DYNAMIC)] = torch.where(allowed[..., guide.dynamic_indices],
            values[..., guide.dynamic_indices], candidate_cns[..., :len(DYNAMIC)])
        candidate_aux = None if state.auxiliary_state is None else state.auxiliary_state.clone()
        columns = {ELECTRICAL: (0, 1), SYNAPTIC: (0, 2, 3)}.get(guide.family, ())
        for column in columns:
            node, quantity = guide.auxiliary_nodes[column], guide.auxiliary_indices[column]
            candidate_aux[:, column] = torch.where(allowed[:, node, quantity],
                values[:, node, quantity], candidate_aux[:, column])
        if guide.family == ELECTRICAL:
            changed = candidate_aux[:, 0] != state.auxiliary_state[:, 0]
            spike = ((candidate_aux[:, 0]-guide.parameters["spike_threshold"])
                     *guide.parameters["spike_rate_gain"]).clamp_min(0)
            candidate_aux[:, 2] = torch.where(changed, spike, candidate_aux[:, 2])
        elif guide.family == SYNAPTIC:
            changed = candidate_aux[:, 0] != state.auxiliary_state[:, 0]
            release = (guide.parameters["initial_release_probability"]+guide.parameters["release_coupling"]
                *(candidate_aux[:, 0]-guide.parameters["baseline_intracellular_calcium"]).clamp_min(0)).clamp(0, 1)
            candidate_aux[:, 1] = torch.where(changed, release, candidate_aux[:, 1])
        candidate = replace(state, cns_state=candidate_cns, auxiliary_state=candidate_aux)
        self._validate_continuation(candidate)
        rendered = self._field(candidate).reshape(expected)
        mask = state._validity[:, :, :, state.query_index]
        validate_physical_forecast(rendered, mask, guide.grid["quantity_profile"]["quantities"])
        # Neither the free-drug carry nor any exposure, PK parameter, dose/event,
        # clock, work counter or original initial field is assigned by this API.
        state.cns_state, state.auxiliary_state = candidate_cns, candidate_aux
        return torch.where(mask, rendered, 0.)

    @torch.no_grad()
    def teacher_force_independent_state(self, state: NativeRHSContinuation,
                                         training_fields: torch.Tensor, *, observed: torch.Tensor) -> torch.Tensor:
        """Offline training-only current-state supervision; no hidden-index matching."""
        self._validate_continuation(state)
        if state._filter_binding["role"] != "train":
            raise ValueError("Teacher forcing is allowed only for TRAIN-role current states")
        required = self.independent_state_mask(state)
        if (observed.shape != required.shape or observed.dtype != torch.bool
                or not observed[required].all()):
            raise ValueError("Teacher forcing requires every independent current state to be observed")
        return self._replace_independent_state(state, training_fields)

    @torch.no_grad()
    def apply_state_residual(self, state: NativeRHSContinuation, residual: torch.Tensor) -> torch.Tensor:
        """Preserved inference API: apply a correction without an autograd graph."""
        return self._apply_state_residual(state, residual)

    def apply_state_residual_functional(self, state: NativeRHSContinuation,
                                        residual: torch.Tensor) -> tuple[NativeRHSContinuation, torch.Tensor]:
        """Correct a private branch while retaining residual-to-future gradients."""
        branch = self.fork_continuation(state)
        return branch, self._apply_state_residual(branch, residual)

    def _apply_state_residual(self, state: NativeRHSContinuation, residual: torch.Tensor) -> torch.Tensor:
        """Use unchanged native domains instead of the tumor paper's [0,1] projection."""
        mask = self.independent_state_mask(state)
        if state.query_index == 0:
            raise ValueError("Runtime residual cannot replace the original 32-tuple t0 state")
        if (residual.shape != mask.shape or not torch.isfinite(residual).all()
                or torch.count_nonzero(residual[~mask])):
            raise ValueError("Residual must be finite and zero outside permitted independent CNS/neural states")
        current = self.current_particle_fields(state)
        if not residual.requires_grad and not torch.count_nonzero(residual):
            return current
        return self._replace_independent_state(state, current+residual)

    def current_log_weights(self, state: NativeRHSContinuation) -> torch.Tensor:
        self._validate_continuation(state)
        return state._log_weights.clone()

    def filter_context(self, state: NativeRHSContinuation) -> dict:
        self._validate_continuation(state)
        validate_execution_state(state)
        execution={} if state._public_control is None else {
            "public_execution_uid":state._public_control.regimen_uid,
            "executed_model_regimen_uid":state._execution_model_uid}
        return dict(state._filter_binding, **execution, filtering_revision=state.filter_revision,
            query_index=state.query_index, time_hours=state.current_time_hours,
            source_time=float(state._guide.clock[state.query_index]),
            source_time_unit=state._guide.grid["query_time"]["unit"])

    def bind_filter_readout(self, state: NativeRHSContinuation, readout,
                            executed_regimen_uid: str) -> None:
        self._validate_continuation(state)
        validate_execution_state(state)
        if state._filter_readout is not None and state._filter_readout != readout:
            raise ValueError("Native filtering readout changed after its initial binding")
        if (state._filter_executed_regimen_uid is not None
                and state._filter_executed_regimen_uid != executed_regimen_uid):
            raise ValueError("Native filtering cannot switch its fixed executed program")
        state._filter_readout = readout
        state._filter_executed_regimen_uid = executed_regimen_uid

    def update_filter_weights(self, state: NativeRHSContinuation, log_weights: torch.Tensor, *,
                              query_index: int, parent_revision: int, observation_uid: str) -> None:
        """Update finite-support mass only; current physical coordinates remain untouched."""
        self._validate_continuation(state)
        validate_execution_state(state)
        if (type(query_index) is not int or query_index != state.query_index or query_index <= 0
                or type(parent_revision) is not int or parent_revision != state.filter_revision
                or not isinstance(observation_uid, str) or not observation_uid
                or observation_uid in state.filter_observation_uids):
            raise ValueError("Native filter update has a stale query/revision or duplicate observation")
        if (log_weights.shape != state._log_weights.shape or log_weights.dtype != torch.float64
                or log_weights.device != state._log_weights.device or torch.isnan(log_weights).any()
                or torch.isposinf(log_weights).any() or not torch.isfinite(log_weights).any(-1).all()
                or not torch.allclose(torch.logsumexp(log_weights, -1),
                    torch.zeros_like(log_weights[:, 0]), atol=4*torch.finfo(torch.float64).eps)):
            raise ValueError("Native filtering requires normalized double log weights on the unchanged tuples")
        state._log_weights = log_weights.clone()
        state._weights = log_weights.exp().to(state._weights.dtype)
        state.filter_revision += 1
        state.filter_observation_uids += (observation_uid,)

    @torch.no_grad()
    def forecast_native(self, inputs, normalization, *, query_count: int | None = None) -> dict:
        """Existing full/prefix API, using the same maintained continuation path."""
        started = time.monotonic()
        state = self.initialize_continuation(inputs, normalization)
        count = len(state._times) if query_count is None else query_count
        result = self.advance_continuation(state, query_count=count)
        result["execution"]["elapsed_seconds"] = time.monotonic()-started
        return result
