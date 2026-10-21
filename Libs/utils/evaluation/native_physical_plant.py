"""Evaluator-private continuation of one original native physical world.

Only public, parameter-independent actuation and acquisition requests cross the
interface. Current truth, source factors, auxiliary state, work and RNG remain
owned here. Public execution receipts bind model filtering and residual H3
without exposing the evaluator-private world.
"""
from __future__ import annotations

from Libs.utils.data.native_actuation import actuation_payload

import copy
import math

import numpy as np
import torch

from Libs.scripts.builders.build_sample_observation_registry import produce_native_arrived_observation
from Libs.utils.data.controlled_world_runtime import (
    step_electrophysiology, step_calcium_synaptic, step_molecular_pharmacological,
)
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
from Libs.utils.model import controlled_cns_source as cns
from Libs.utils.model.native_arrived_observation import (
    NativeAcquisition, NativeArrivedObservation, native_cns_readout,
)
from Libs.utils.model.native_observable_source import (
    ELECTRICAL, SYNAPTIC, MOLECULAR, CoupledObservableLaw, action_levels, integrate_coupled_interval,
)
from Libs.utils.model.native_scientific_source import is_long_cns, long_cns_definition, scoped_group_config
from Libs.utils.data.native_actuation import (
    NativeDoseEvent, NativeActuation, NativeExecutedPrefix, NativeActuationRequest,
    NativeObservationIdentity, native_actuation_library, compose_actuation,
    NativeActuationPlan, compile_actuation_plan, executed_prefix,
)




class NativePhysicalPlant:
    """One private initial world, physical history and measurement stream.

    Construction belongs to the evaluator. A caller supplies an explicit true
    initial condition and true source parameters, never a candidate tuple array.
    Public methods accept only registered actuation content and acquisition
    metadata. They never accept model continuation state or forecast trajectories.
    """

    def __init__(self, config: dict, group: dict, *, private_initial_field: np.ndarray,
                 private_factors: dict, private_physical_parameters: dict | None,
                 initial_actuation: NativeActuation, measurement_generator: torch.Generator,
                 observation_identity: NativeObservationIdentity | None = None):
        self._library = native_actuation_library(config, group)
        if type(initial_actuation) is not NativeActuation or initial_actuation not in self._library:
            raise ValueError("Initial actuation content is outside the public source library")
        self._config = scoped_group_config(config, group)
        self._group = copy.deepcopy(group)
        self._readout = native_cns_readout(config, group)
        self._source_prior_uid = group["posterior_uid"]
        expected_identity = NativeObservationIdentity(self._readout.source_uid,self._readout.source_snapshot_ref,
            group["family_uid"],group["role"],(group["sample_uid"],),group["support_uid"],
            (tuple(group["particle_uids"]),),group["posterior_uid"])
        self._observation_identity = expected_identity if observation_identity is None else observation_identity
        if observation_identity is not None:
            if (type(observation_identity) is not NativeObservationIdentity
                    or not isinstance(observation_identity.origin_posterior_uid,str) or not observation_identity.origin_posterior_uid
                    or any(getattr(observation_identity,name) != getattr(expected_identity,name)
                           for name in NativeObservationIdentity.__dataclass_fields__ if name != "origin_posterior_uid")
                    or len(observation_identity.particle_uids) != 1 or len(observation_identity.particle_uids[0]) != 32):
                raise ValueError("Controller source/sample/support/complete32 identity differs from the original source group")
        self._family = group["family_uid"]
        self._grid = family_grid(observable_profile(self._config), self._family)
        self._times = np.asarray(self._grid["time_hours"], dtype=np.float64)
        self._source_times = initial_actuation.source_times
        initial = np.asarray(private_initial_field, dtype=np.float64).copy()
        if initial.shape != (6,6,26) or not np.isfinite(initial).all():
            raise ValueError("Plant requires one finite explicit native initial field")
        source = self._config["controlled_cns_source"]
        factors = {key: np.asarray(value, dtype=np.float64).copy() for key,value in private_factors.items()}
        if (set(factors) != set(source["factor_basis"])
                or any(value.shape != (len(source["factor_basis"][key]),)
                       or not np.isfinite(value).all() or (value <= 0).any() for key,value in factors.items())):
            raise ValueError("Plant requires one complete private source-factor vector per factor")
        self._long = is_long_cns(self._config, self._family)
        names = source["grid"]["species_id"]
        dynamic = initial[..., :15].reshape(36,15)[:, [names.index(name) for name in cns.DYNAMIC]]
        if (dynamic < 0).any():
            raise ValueError("Negative true source dynamic initial state")
        self._physical = None
        self._parameters = None
        if self._long:
            if private_physical_parameters is not None:
                raise ValueError("Long-CNS source does not accept a discrete auxiliary parameter interface")
            self._law = cns.ControlledLaw(long_cns_definition(self._config), factors, initial[..., :15])
            self._state = np.concatenate([dynamic.ravel(), np.zeros(11)])
        else:
            declarations = self._config["native_observable_bridge"]["physical_families"][self._family]["parameters"]
            if (not isinstance(private_physical_parameters, dict) or set(private_physical_parameters) != set(declarations)
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in private_physical_parameters.values())):
                raise ValueError("Plant requires one finite private primary-source parameter mapping")
            self._parameters = copy.deepcopy(private_physical_parameters)
            self._law = CoupledObservableLaw.from_initial_state(self._config, self._group, initial, factors,
                                                               initial_actuation.source_uid)
            self._physical = np.asarray([self._parameters.get("initial_"+name, 0.)
                for name in self._law.physical_names], dtype=np.float64)
            self._state = np.concatenate([dynamic.ravel(), np.zeros(36+11+2)])
            # Original producer writes supported family readouts to a float32
            # initial field after its core pre-dose quantities. Preserve that
            # rounding and the independent historical t0 core-derived values.
            for qi, quantity in enumerate(self._grid["quantity_profile"]["quantities"]):
                if quantity.get("family_uid") != self._family:
                    continue
                site = quantity["incidence"]
                ri = self._grid["region_id"].index(site["anchor_region_id"])
                ci = self._grid["compartment_id"].index(site["anchor_compartment_id"])
                if self._law.quantity_support[ri,ci,qi]:
                    expected = self._parameters.get("initial_"+quantity["observable_name"],0.)
                    if np.float32(initial[ri,ci,qi]) != np.float32(expected):
                        raise ValueError("Initial supported source readout differs from its original float32 auxiliary state")
        if not isinstance(measurement_generator, torch.Generator) or measurement_generator.device.type != "cpu":
            raise ValueError("Plant requires a CPU measurement generator")
        self._generator = torch.Generator(device="cpu")
        self._generator.set_state(measurement_generator.get_state().clone())
        self._public_program = initial_actuation
        self._intervals = initial_actuation.interval_values
        self._events = initial_actuation.dose_events
        self._active_uid = initial_actuation.regimen_uid
        self._query = 0
        self._clock = 0.
        self._last_switch_query = None
        self._acquired_queries: tuple[int, ...] = ()
        self._tail_started = self._tail_complete = False
        self._committed_tail_plan = None
        self._failed = None
        self._work = {"nfev":0}
        self._input_jumps = self._prior_c = self._prior_bound = 0.
        self._executed_intervals: tuple[float, ...] = ()
        self._executed_events: tuple[NativeDoseEvent, ...] = ()
        if self._long:
            # The long-CNS producer pads the original integrator's pre-dose
            # quantities; unlike the coupled producer it does not replace t0
            # by its float32 persisted initial field. Keep the dynamic initial
            # state unchanged while preserving that original readout arithmetic.
            self._field = np.zeros((6,6,26),dtype=np.float64)
            self._field[...,:15] = self._law.quantities([0.],
                self._state[:396].reshape(1,36,11),self._action())[0]
        else:
            self._field = initial
        self._history = []
        if not self._long:
            self._apply_molecular_boundary()
        self._save_private_history()

    @property
    def observation_identity(self):
        return self._observation_identity

    @property
    def prefix(self) -> NativeExecutedPrefix:
        self._require_live()
        prefix = executed_prefix(self._public_program,self._query)
        if prefix.interval_values != self._executed_intervals or prefix.dose_events != self._executed_events:
            raise ValueError("Physical prefix does not retain its actually consumed public intervals/events")
        return prefix

    def _require_live(self):
        if self._failed is not None:
            raise RuntimeError("Physical plant failed after execution began; same-query retry is prohibited: "+self._failed)

    def _require_interactive(self):
        self._require_live()
        if self._tail_started:
            raise RuntimeError("Physical plant committed its unobserved tail; acquisition/control use is closed")

    def switch(self, request: NativeActuationRequest) -> NativeActuation:
        self._require_interactive()
        if (type(request) is not NativeActuationRequest or type(request.prefix) is not NativeExecutedPrefix
                or type(request.actuation) is not NativeActuation or request.actuation not in self._library
                or request.prefix != self.prefix or self._clock != self._times[self._query]
                or not 0 < self._query < len(self._times)-1 or self._last_switch_query == self._query):
            raise ValueError("Switch requires current consumed content/clock and a registered future actuation")
        program = compose_actuation(request,self._library)
        self._intervals, self._events, self._active_uid = program.interval_values, program.dose_events, program.regimen_uid
        self._public_program = program
        self._last_switch_query = self._query
        return program

    def _validate_acquisition(self, acquisition, *, query: int):
        if type(acquisition) is not NativeAcquisition:
            raise ValueError("Physical plant accepts only public native acquisition metadata")
        acquisition.validate()
        if (not 0 < query < len(self._times) or acquisition.query_index != query
                or query in self._acquired_queries
                or acquisition.readout != self._readout
                or acquisition.sample_uids != (self._group["sample_uid"],)
                or acquisition.support_uid != self._group["support_uid"]
                or acquisition.origin_posterior_uid != self.observation_identity.origin_posterior_uid
                or acquisition.particle_uids != (tuple(self._group["particle_uids"]),)
                or acquisition.filtering_revision != len(self._acquired_queries)
                or acquisition.source_time != self._source_times[query]
                or acquisition.source_time_unit != self._grid["query_time"]["unit"]
                or acquisition.time_hours != self._times[query]
                or acquisition.executed_regimen_uid != self._active_uid
                or self._clock != self._times[self._query]):
            raise ValueError("Acquisition differs from the current source/sample/readout/program/query binding or repeats a measurement")

    def _retain(self, now, state):
        self._clock = float(now)
        self._state = state.copy()
        if not np.isfinite(state).all() or (state[:396] < 0).any():
            raise FloatingPointError("Invalid accepted source state")

    def _action(self):
        return {"regimen_uid":self._active_uid,
            "dose_times_hours":[event.source_time for event in self._events],
            "fractions":[event.fraction for event in self._events],
            "event_total_dose_mg_per_kg":[event.total_dose_mg_per_kg for event in self._events]}

    def _advance_long(self, deadline):
        source = self._law.source
        right = self._times[self._query+1]
        action = self._action()
        scales = np.asarray([source["reference_scales"][name] for name in cns.DYNAMIC])
        atol = np.concatenate([np.tile(scales,36),scales*self._law.volumes.sum()])*source["numerics"]["reference_atol_scale"]
        boundaries = sorted({self._clock,float(right),*(event.source_time for event in self._events
                                                       if self._clock < event.source_time < right)})
        for left, endpoint in zip(boundaries,boundaries[1:]):
            self._executed_events = tuple(event for event in self._events if event.source_time <= left)
            rhs = lambda t,state: self._law.rhs(t,state,action,left,deadline)
            self._state, _ = cns.integrate_source_interval(rhs,left,endpoint,self._state,"DOP853",
                source["numerics"]["reference_max_step_hours"],rtol=source["numerics"]["reference_rtol"],
                atol=atol,progress=self._retain,work=self._work,accumulated_clock=True)
        self._clock = float(right)
        self._executed_events = tuple(event for event in self._events if event.source_time <= right)
        self._field = np.zeros((6,6,26),dtype=np.float64)
        self._field[...,:15] = self._law.quantities([right],self._state[:396].reshape(1,36,11),action)[0]

    def _apply_molecular_boundary(self):
        if self._family != MOLECULAR:
            return
        current_c = self._physical[0]*1e-9
        current_bound = self._law.law.pk["bmax_mol_per_l"][self._law.ca1]*self._physical[1]
        self._input_jumps += (current_c-self._prior_c+current_bound-self._prior_bound)*self._law.law.volumes[self._law.ca1]
        self._state[396+self._law.ca1] = current_c
        self._prior_c, self._prior_bound = current_c, current_bound

    def _advance_coupled(self, deadline):
        numerics = self._config["native_observable_bridge"]["numerics"]
        self._state, _ = integrate_coupled_interval(self._law,self._state,self._query,self._physical,
            numerics["reference_method"],numerics["cns_max_step_interval_fractions"][-1],deadline,
            progress=self._retain,work=self._work)
        index = self._query
        step = self._source_times[index+1]-self._source_times[index]
        value = self._intervals[index]
        values = tuple(float(v) for v in self._physical)
        if self._family == ELECTRICAL:
            next_physical = step_electrophysiology(values,self._parameters,step,value)
        elif self._family == SYNAPTIC:
            next_physical = step_calcium_synaptic(values,self._parameters,step,value,
                initial_release_probability=self._parameters["initial_release_probability"])
        else:
            next_physical = step_molecular_pharmacological(values,self._parameters,step,value)
        self._physical = np.asarray(next_physical,dtype=np.float64)
        self._executed_intervals = (*self._executed_intervals,value)
        self._clock = float(self._times[index+1])
        self._apply_molecular_boundary()
        self._field = self._law.quantities(self._state,physical=self._physical)
        if not np.isfinite(self._field).all():
            raise FloatingPointError("Nonfinite physical plant field")

    def _save_private_history(self):
        state = self._state.copy()
        if not self._long:
            state[-2] += self._input_jumps
        self._history.append({"query_index":self._query,"time_hours":self._clock,
            "state":state,"field":self._field.copy(),
            "physical":None if self._physical is None else self._physical.copy(),
            "regimen_uid":self._active_uid})

    def _evaluation_history(self):
        """Evaluator-only history and work; never included in an acquisition packet."""
        indices = [row["query_index"] for row in self._history]
        times = [row["time_hours"] for row in self._history]
        complete = (self._failed is None and self._query == len(self._times)-1
                    and indices == list(range(len(self._times))) and times == self._times.tolist())
        return {"source_rhs_calls": self._work["nfev"], "source_query_indices": indices,
                "source_solver_elapsed_seconds": None, "source_solver_elapsed_status": "not_separately_measured",
                "source_time_values_hours": times, "original_query_values": list(self._source_times),
                "original_time_unit": self._grid["query_time"]["unit"],
                "source_clock_time_hours": self._clock, "source_horizon_complete": complete,
                "source_failure": self._failed,
                "committed_control_plan": (None if self._committed_tail_plan is None else actuation_payload(self._committed_tail_plan)),
                "consumed_interval_values": list(self._executed_intervals),
                "consumed_dose_events": self._executed_events, "final_public_program": self._public_program,
                "grid": self._grid, "support": quantity_support(self._grid, self._readout.source_uid),
                "private_fields": np.stack([row["field"] for row in self._history]) if complete else None}

    @staticmethod
    def _validate_deadline(deadline):
        if type(deadline) not in (float, int) or math.isnan(deadline):
            raise ValueError("Invalid private source deadline")

    def _validate_advance(self, prefix: NativeExecutedPrefix, deadline):
        self._require_live()
        self._validate_deadline(deadline)
        if (type(prefix) is not NativeExecutedPrefix or prefix != self.prefix
                or self._clock != self._times[self._query]
                or self._query+1 >= len(self._times)):
            raise ValueError("Advance requires the current original clock and committed control prefix")

    def advance(self, prefix: NativeExecutedPrefix, *, deadline=math.inf) -> NativeExecutedPrefix:
        """Advance one original interval without acquiring or consuming measurement RNG."""
        self._require_interactive()
        self._validate_advance(prefix, deadline)
        self._advance_physics(deadline)
        return self.prefix

    def _advance_physics(self, deadline):
        try:
            if self._long:
                self._advance_long(deadline)
            else:
                self._advance_coupled(deadline)
            self._query += 1
            self._save_private_history()
        except Exception as error:
            self._failed = type(error).__name__+": "+str(error)
            raise

    def acquire_current(self, acquisition: NativeAcquisition) -> NativeArrivedObservation:
        """Acquire once at the current original clock, without physical advancement."""
        self._require_interactive()
        self._validate_acquisition(acquisition, query=self._query)
        try:
            arrived = produce_native_arrived_observation(torch.from_numpy(self._field[None]), acquisition,
                episode_source_uid=self._readout.source_uid,
                episode_source_snapshot_ref=self._readout.source_snapshot_ref,
                episode_sample_uids=(self._group["sample_uid"],),episode_query_index=self._query,
                episode_regimen_uid=self._active_uid,measurement_generator=self._generator,
                observation_uid=self._group["sample_uid"]+":physical-monitor:"+str(self._query))
            self._acquired_queries = (*self._acquired_queries, self._query)
            return arrived
        except Exception as error:
            self._failed = type(error).__name__+": "+str(error)
            raise

    def acquire(self, acquisition: NativeAcquisition, *, deadline=math.inf) -> NativeArrivedObservation:
        """Legacy combined next-clock advance and full/selected acquisition."""
        self._require_interactive()
        self._validate_acquisition(acquisition, query=self._query+1)
        self._validate_deadline(deadline)
        self.advance(self.prefix, deadline=deadline)
        return self.acquire_current(acquisition)

    def _complete_unobserved_tail(self, prefix: NativeExecutedPrefix, *, deadline=math.inf) -> None:
        """Evaluator-only physical tail under the already committed full program.

        This does not choose STOP or assess the model episode's status. The
        evaluator must authorize a tail before calling; no packet or model
        feedback is produced, and failed private physics remains failed.
        """
        self._require_interactive()
        self._validate_deadline(deadline)
        if type(prefix) is not NativeExecutedPrefix or prefix != self.prefix:
            raise ValueError("Unobserved tail requires the current committed control prefix")
        self._tail_started = True
        while self._query+1 < len(self._times):
            self._advance_physics(deadline)
        self._tail_complete = True


    def _complete_planned_tail(self, plan: NativeActuationPlan, *, deadline=math.inf) -> None:
        """Execute a fully committed schedule; no future observations, decisions or model updates."""
        self._require_interactive()
        self._validate_deadline(deadline)
        if (type(plan) is not NativeActuationPlan or plan.origin != self.prefix
                or plan != compile_actuation_plan(self._public_program, self.prefix, plan.steps, self._library)):
            raise ValueError("Unobserved planned tail differs from the current original-library program")
        self._tail_started = True
        self._committed_tail_plan = plan
        scheduled = {request.prefix.query_index: request for request in plan.steps}
        while self._query+1 < len(self._times):
            request = scheduled.get(self._query)
            if request is not None:
                if request.prefix != self.prefix or self._last_switch_query == self._query:
                    raise ValueError("Committed switch no longer matches its consumed physical prefix")
                program = compose_actuation(request, self._library)
                self._public_program = program
                self._intervals, self._events = program.interval_values, program.dose_events
                self._active_uid = program.regimen_uid
                self._last_switch_query = self._query
            self._advance_physics(deadline)
        if self._public_program != plan.final_program:
            raise ValueError("Committed physical schedule did not reach its validated final program")
        self._tail_complete = True
