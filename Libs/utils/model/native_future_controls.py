"""Model-owned future controls joined to an actual fixed-program prefix.

This compiler consumes registered public candidate programs. It never creates
source readbacks or physical-plant observations. Molecular drive integrals are
not delivered drug doses. Verified public execution binds selected controls,
frozen residual conditioning and cross-switch H3 replay.
"""
from __future__ import annotations

from Libs.utils.data.native_actuation import actuation_payload

import copy
import math
import hashlib

import numpy as np

import torch

from Libs.utils.data.joint_physical_parameters import (
    validate_candidate_parameters, validate_program_parameter_subset,
)
from Libs.utils.data.native_diffusion_data import joint_particle_contexts
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, MOLECULAR, estimated_exposure_on_levels, action_levels
from Libs.utils.model.controlled_cns_source import pk_values
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.data.native_observable_profile import observable_profile, family_grid
from Libs.utils.model.native_scientific_source import action_definition
from Libs.utils.data.native_actuation import (
    NativeActuation, NativeActuationRequest, NativeObservationIdentity, executed_prefix, compose_actuation,
    molecular_administration_policy, molecular_administration_bound, validate_administration_library,
)


def future_control_policy(config: dict) -> dict:
    policy = config.get("native_future_controls")
    required = config["schemas"]["native_future_controls"]["required_fields"]
    expected = {"schema_version": "native_future_controls.v1",
        "library": "registered_current_input_programs_only",
        "decision_phase": "postzero_original_query_post_event",
        "horizon": "remaining_original_grid", "prefix": "actual_executed_action_column",
        "stimulation_budget": "library_maximum_absolute_input_integral",
        "long_cns_budget": "library_maximum_total_event_dose_mg_per_kg",
        "molecular_budget": "library_maximum_model_drive_integral_not_delivered_dose",
        "rhs_budget": "shared_actual_work_original_declared_ceiling"}
    if (not isinstance(policy, dict) or set(required)-set(policy)
            or policy.get("activation") not in {"unsupported_model_planning_only","requires_verified_public_execution_binding"}
            or any(policy.get(key) != value for key, value in expected.items())):
        raise ValueError("Native future-control policy or boundary is unbound")
    if (policy["activation"] == "requires_verified_public_execution_binding"
            and config["schemas"]["native_future_controls"].get("observation_activation")
                != "requires_verified_public_execution_and_admitted_actual_switch_history"):
        raise ValueError("Canonical execution activation description is not bound to actual switch history")
    return copy.deepcopy(policy)


def _semantic_signature(record: dict) -> dict:
    action = copy.deepcopy(record["action"])
    for name in ("action_uid", "dose", "schedule", "duration"):
        action.pop(name, None)
    if action["intervention_type"] in {"electrical_stimulation", "optical_stimulation"}:
        field = action.get("stimulation_or_environment")
        if not isinstance(field, dict) or not isinstance(field.get("magnitude_or_program"), dict):
            raise ValueError("Typed stimulation requires its numerical waveform")
        wave = field["magnitude_or_program"]
        if not {"time_values", "values"} <= set(wave):
            raise ValueError("Typed stimulation waveform lacks its original time/value arrays")
        for name in ("time_values", "values"):
            wave.pop(name)
    physical = record.get("physical_program")
    if physical is not None:
        inputs = physical["inputs"]
        definition = copy.deepcopy(inputs["run_snapshot"].get("pk_definition"))
        if definition is not None:
            definition.pop("total_dose_mg_per_kg")
        action["physical_identity"] = {"sites": inputs["binding"]["site_mappings"],
            "parameters": inputs["run_snapshot"]["parameters"], "pk_definition": definition,
            "region_ids": inputs["region_ids"], "compartment_ids": inputs["compartment_ids"]}
    return action


def _row_integral(values, durations) -> float:
    return math.fsum(abs(float(value))*float(dt) for value, dt in zip(values, durations))


def model_control_identity(state) -> str:
    """Consumed on rebranch to reject altered static controls under old identities."""
    if state._registered_stimulus is not None:
        value = {"stimulus": state._guide.stimulus.cpu().tolist(),
                 "source_interval_binding": state._guide.bind_source_intervals}
    else:
        value = {"event_times": [], "event_doses_mol": [], "event_doses_mg_per_kg": []}
        for times, dose, amount in zip(state.event_times, state.event_doses, state._event_doses_mg_per_kg):
            valid = torch.isfinite(times)
            value["event_times"].append(times[valid].cpu().tolist())
            value["event_doses_mol"].append(dose[valid].cpu().tolist())
            value["event_doses_mg_per_kg"].append(amount[valid].cpu().tolist())
    return digest(value)


def compile_future_controls(state, inputs, config: dict, *, executed_regimen_uid: str,
                            candidate_regimen_uids=None, require_delivered_dose: bool = False,
                            public_control: NativeActuation | None = None) -> dict:
    """Validate and compile without changing state, identities or work accounting."""
    policy = future_control_policy(config)
    guide, k = state._guide, state.query_index
    if model_control_identity(state) != state._control_identity:
        raise ValueError("Executed prefix/static controls changed under their retained identity")
    if not 0 < k < len(state._times)-1 or guide.vm_only:
        raise ValueError("Future control requires a supported postzero query with a remaining interval")
    if inputs.grid != guide.grid or inputs.group_shape != (guide.b, guide.j, len(state._registered_regimen_uids)) or (guide.b, guide.j) != (1, 32):
        raise ValueError("Future control requires the original complete32-tuple grid and clock")
    binding = state._filter_binding
    support = inputs.posterior.support
    identities = {"source_uid": support.source_uid, "source_snapshot_ref": support.source_snapshot_ref,
        "support_uid": support.support_uid, "role": support.role, "origin_posterior_uid": inputs.posterior_uid,
        "sample_uids": tuple(inputs.sample_uids), "particle_uids": tuple(tuple(row) for row in inputs.particle_uids),
        }
    if tuple(inputs.regimen_uids) != state._registered_regimen_uids:
        raise ValueError("Future-control registered library identity changed")
    if any(binding[key] != value for key, value in identities.items()):
        raise ValueError("Future-control original source/support/particle/program identity changed")
    if (executed_regimen_uid not in binding["regimen_uids"]
            or state._planning_origin is None and state._filter_executed_regimen_uid is not None
            and (state._execution_model_uid or state._filter_executed_regimen_uid) != executed_regimen_uid):
        raise ValueError("Future-control executed prefix differs from the admitted action")
    original_initial = support.initial_field[:, :, None].expand(guide.b, guide.j, guide.a, *support.initial_field.shape[2:])
    if not torch.equal(original_initial.double().reshape_as(state._initial), state._initial):
        raise ValueError("Future-control initial field differs from its executed prefix")
    joint_particle_contexts(inputs.posterior, inputs.programs, inputs.regimen_uids)
    validate_candidate_parameters(support.physical_parameter_hypotheses, support)
    validate_program_parameter_subset(support, inputs.programs)
    for index, declaration in enumerate(support.physical_parameter_hypotheses["declarations"]):
        actual = guide.parameters[declaration["name"]].reshape(guide.b, guide.j, guide.a)
        expected = support.physical_parameter_hypotheses["values"][..., index, None].expand_as(actual)
        if not torch.equal(actual, expected):
            raise ValueError("Future-control named physical tuple parameters changed")
    records = [row for sample in inputs.programs["records"] for particle in sample for row in particle]
    if tuple(row["program_uid"] for row in records) != state._registered_program_uids:
        raise ValueError("Future controls must retain the registered original program contents")
    candidates = tuple(inputs.regimen_uids if candidate_regimen_uids is None else candidate_regimen_uids)
    if not candidates or len(set(candidates)) != len(candidates) or any(uid not in inputs.regimen_uids for uid in candidates):
        raise ValueError("Future-control candidate library is empty, duplicated or unregistered")
    executed = binding["regimen_uids"].index(executed_regimen_uid)
    prefix_source_uid = (state._planning_origin["source_candidate_regimen_uids"][executed]
                         if state._planning_origin is not None else executed_regimen_uid)
    prefix_source_index = inputs.regimen_uids.index(prefix_source_uid)
    indices = [inputs.regimen_uids.index(uid) for uid in candidates]
    for particle in inputs.programs["records"][0]:
        signature = _semantic_signature(particle[prefix_source_index])
        if any(_semantic_signature(particle[index]) != signature for index in indices):
            raise ValueError("Future suffix changes an unrepresentable actuator/agent/route/site/PK identity")
    count, b, j, a = len(indices), guide.b, guide.j, guide.a
    library_a = len(inputs.regimen_uids)
    result = {"executed_index": executed, "candidate_indices": indices, "candidate_regimen_uids": candidates,
        "stimulus": None, "event_times": None, "event_doses": None, "event_doses_mg_per_kg": None, "applied_events": None}
    budgets = []
    if hasattr(guide, "stimulus"):
        mapped = guide.family == MOLECULAR and molecular_administration_policy(config) is not None
        public_budgets = []
        if mapped:
            parent_public = state._public_control if public_control is None else public_control
            if state._public_control is not None and parent_public != state._public_control:
                raise ValueError("Mapped future controls cannot replace the actually executed public prefix")
            if parent_public is None or not molecular_administration_bound(parent_public,config):
                raise ValueError("Mapped future controls require the actual or explicitly retained hypothetical public prefix")
            library = state._execution_library
            validate_administration_library(library,config)
            prefix_public = executed_prefix(parent_public,k)
            for candidate_uid in candidates:
                action = next((item for item in library if item.regimen_uid == candidate_uid),None)
                if action is None:
                    raise ValueError("Mapped future action is outside its public registered library")
                joined_public = compose_actuation(NativeActuationRequest(prefix_public,action),library)
                prefix_mass = parent_public.administered_mass(k)
                total_mass = joined_public.administered_mass()
                public_budgets.append({"unit":"mg/kg","prefix":prefix_mass,"future":total_mass-prefix_mass,
                    "total":total_mass,"library_maximum":action.administration.maximum_supported_total_mg_per_kg,
                    "delivered_dose_bound":True,"administration_law_uid":action.administration.law_uid})
        elif guide.family == MOLECULAR and require_delivered_dose:
            raise ValueError("Molecular model-drive integral has no delivered-dose mapping; physical dosing is unbound")
        first = k if guide.bind_source_intervals and guide.family != MOLECULAR else k+1
        old = guide.stimulus.reshape(b, j, a, -1)
        joined = old[:, :, executed:executed+1].expand(b, j, count, -1).clone()
        library_stimulus = state._registered_stimulus.reshape(b, j, library_a, -1)
        joined[..., first:] = library_stimulus[:, :, indices, first:]
        offset = 0 if guide.bind_source_intervals and guide.family != MOLECULAR else 1
        durations = guide.clock.diff().cpu().tolist()
        unit = {ELECTRICAL: "uA*"+guide.grid["query_time"]["unit"],
                SYNAPTIC: "mW*"+guide.grid["query_time"]["unit"], MOLECULAR: "mg/kg*hour"}[guide.family]
        if guide.family == MOLECULAR:
            durations = guide.times.diff().cpu().tolist()
        for particle in range(j):
            full = [_row_integral(library_stimulus[0, particle, ai, offset:offset+len(durations)], durations) for ai in range(library_a)]
            limit = max(full)
            prefix = _row_integral(old[0, particle, executed, offset:offset+k], durations[:k])
            for ci in range(count):
                total = _row_integral(joined[0, particle, ci, offset:offset+len(durations)], durations)
                suffix = _row_integral(joined[0, particle, ci, offset+k:offset+len(durations)], durations[k:])
                if mapped:
                    budgets.append({"particle_index":particle,"candidate_index":ci,**public_budgets[ci]})
                else:
                    if total > limit:
                        raise ValueError("Future suffix exceeds the remaining registered-library intervention budget")
                    budgets.append({"particle_index": particle, "candidate_index": ci, "unit": unit,
                        "prefix": prefix, "future": suffix, "total": total, "library_maximum": limit,
                        "delivered_dose_bound": False})
        result.update(stimulus=joined.reshape(b*j*count, -1), first_future_control_index=first)
    else:
        if any(row["action"].get("dose_unit") != "mg/kg" or row["action"]["schedule"].get("unit") != "hour" for row in records):
            raise ValueError("Long-CNS planning requires registered mg/kg doses and hour events")
        event_rows, dose_rows, dose_mg_rows, flags = [], [], [], []
        now = state.current_time_hours
        for particle in range(j):
            row_index = particle*a+executed
            past = state.applied_events[row_index]
            if not torch.equal(past, state.event_times[row_index] <= now):
                raise ValueError("Long-CNS executed event prefix is inconsistent")
            library = inputs.programs["records"][0][particle]
            total_doses = [float(row["physical_program"]["inputs"]["run_snapshot"]["pk_definition"]["total_dose_mg_per_kg"]) for row in library]
            limit = max(total_doses)
            prefix_amounts = state._event_doses_mg_per_kg[row_index, past].cpu().tolist()
            prefix = math.fsum(prefix_amounts)
            for ci, ai in enumerate(indices):
                action = library[ai]["action"]
                schedule = action["schedule"]
                future = [(float(t), total_doses[ai]*float(f)) for t, f in zip(schedule["values"], schedule["dose_fractions"]) if float(t) > now]
                suffix = math.fsum(value for _, value in future)
                if math.fsum([*prefix_amounts, *[value for _, value in future]]) > limit:
                    raise ValueError("Future dose exceeds the remaining registered-library intervention budget")
                definition = library[ai]["physical_program"]["inputs"]["run_snapshot"]["pk_definition"]
                total_mol = definition["total_dose_mg_per_kg"]*definition["body_mass_kg"]*1e-3/definition["molecular_weight_g_per_mol"]
                future_mol = [total_mol*float(f) for t, f in zip(schedule["values"], schedule["dose_fractions"]) if float(t) > now]
                event_rows.append([*state.event_times[row_index, past].cpu().tolist(), *[t for t, _ in future]])
                dose_rows.append([*state.event_doses[row_index, past].cpu().tolist(), *future_mol])
                dose_mg_rows.append([*prefix_amounts, *[amount for _, amount in future]])
                flags.append([True]*int(past.sum())+[False]*len(future))
                budgets.append({"particle_index": particle, "candidate_index": ci, "unit": "mg/kg",
                    "prefix": prefix, "future": suffix, "total": prefix+suffix,
                    "library_maximum": limit, "delivered_dose_bound": True})
        width = max(map(len, event_rows))
        times = torch.full((b*j*count, width), float("inf"), dtype=torch.float64, device=guide.device)
        doses = torch.zeros_like(times); doses_mg = torch.zeros_like(times); applied = torch.zeros_like(times, dtype=torch.bool)
        for row, (ts, ds, fs) in enumerate(zip(event_rows, dose_rows, flags)):
            times[row, :len(ts)] = guide.tensor(ts); doses[row, :len(ds)] = guide.tensor(ds)
            doses_mg[row, :len(ds)] = guide.tensor(dose_mg_rows[row])
            applied[row, :len(fs)] = torch.tensor(fs, device=guide.device)
        result.update(event_times=times, event_doses=doses, event_doses_mg_per_kg=doses_mg, applied_events=applied, first_future_control_index=None)
    identities = []
    for ci, candidate in enumerate(candidates):
        content = {"schema_version": "native_model_future_control.v1", "source": copy.deepcopy(binding),
            "executed_regimen_uid": executed_regimen_uid, "candidate_regimen_uid": candidate,
            "query_index": k, "source_time": float(guide.clock[k]), "filter_revision": state.filter_revision,
            "admitted_observation_uids": state.filter_observation_uids,
            "original_program_uids": list(state._registered_program_uids),
            "control": (result["stimulus"].reshape(b, j, count, -1)[:, :, ci].cpu().tolist() if result["stimulus"] is not None
                        else [{"times": event_rows[ji*count+ci], "doses_mol": dose_rows[ji*count+ci]} for ji in range(j)])}
        if guide.family == MOLECULAR and molecular_administration_policy(config) is not None:
            content["administration"] = public_budgets[ci]
        identities.append("model-suffix:"+digest(content))
    result["metadata"] = {"schema_version": "native_model_future_control.v1", "policy": policy,
        "model_regimen_uids": identities, "source_candidate_regimen_uids": list(candidates),
        "executed_regimen_uid": executed_regimen_uid, "parent_model_regimen_uid": executed_regimen_uid if state._planning_origin is not None else None,
        "original_query_index": k, "origin_filter_revision":state.filter_revision,
        "origin_observation_uids":state.filter_observation_uids,
        "first_future_control_index": result["first_future_control_index"],
        "original_program_uids": list(state._registered_program_uids), "intervention_budgets": budgets,
        "source_readback": False, "physical_activation": "REQUIRES_VERIFIED_PUBLIC_EXECUTION_BINDING",
        "cross_switch_H3": "REQUIRES_ADMITTED_ACTUAL_EXECUTION_HISTORY", "scientific_twin": "NOT_RUN"}
    return result


def _tensor_identity(value):
    if value is None:
        return None
    array = value.detach().contiguous().cpu().numpy()
    return {"dtype":str(value.dtype),"shape":list(value.shape),
            "sha256":hashlib.sha256(array.tobytes()).hexdigest()}


def execution_constants_identity(state):
    """Constant whole-tuple/source identity, independent of the current action width."""
    guide = state._guide
    def particle(value):
        grouped=value.reshape(guide.b,guide.j,guide.a,*value.shape[1:])
        if not torch.equal(grouped,grouped[:,:,:1].expand_as(grouped)):
            raise ValueError("Execution changes static whole-tuple parameters across actions")
        return _tensor_identity(grouped[:,:,0])
    binding={key:value for key,value in state._filter_binding.items()
             if key not in {"regimen_uids","native_admitted_readout_payload_sha256"}}
    validity=state._validity
    if not torch.equal(validity,validity[:,:,:1].expand_as(validity)):
        raise ValueError("Execution changed the original supported quantity mask")
    return digest({"source":binding,"clock":state._times,"grid":guide.grid,
        "equation_definition":guide.source,"coupling":guide.coupling,
        "validity":_tensor_identity(validity[:,:,0]),
        "graph_and_rates":{key:_tensor_identity(getattr(guide,key)) for key in
            ("edges","transport_scale")},
        "derived_tuple_constants":{key:particle(getattr(guide,key)) for key in
            ("volumes","conductance","elimination","bmax")},
        "parameters":{key:particle(value) for key,value in guide.parameters.items()},
        "factors":{key:particle(value) for key,value in guide.factors.items()},
        "pk":{key:particle(value) for key,value in state._pk_parameters.items()},
        "initial":particle(state._initial)})


def execution_library_identity(state):
    return digest({"uids":state._registered_regimen_uids,"programs":state._registered_program_uids,
                   "registered_model_input":_tensor_identity(state._registered_stimulus)})


def _model_estimated_occupancy(inputs, particle, candidate, public, config):
    """Original compiler arithmetic on model-estimated parameters, never source truth."""
    record=inputs.programs["records"][0][particle][candidate]
    snapshot=record["physical_program"]["inputs"]["run_snapshot"]
    support=np.asarray(record["physical_program"]["support"],dtype=bool).reshape(len(public.source_times),36)
    if public.family_uid == MOLECULAR:
        estimate=estimated_exposure_on_levels(config,snapshot["parameters"],public.source_times,
            (0.,*public.interval_values))
        concentration=estimate["exposure_nmol_per_l"]*1e-9
        kd=snapshot["parameters"]["binding_affinity"]*1e-9
        occupancy=concentration/(concentration+kd)
        return np.where(support,occupancy[:,None],0.),np.where(support,concentration[:,None],0.)
    action={"dose_times_hours":[event.source_time for event in public.dose_events],
        "fractions":[event.fraction for event in public.dose_events],
        "event_total_dose_mg_per_kg":[event.total_dose_mg_per_kg for event in public.dose_events]}
    definition={"pk":snapshot["pk_definition"],"numerics":config["controlled_cns_source"]["numerics"]}
    concentration,occupancy,_=pk_values(inputs.grid["time_hours"],snapshot["parameters"],action,definition)
    return np.where(support,occupancy,0.),np.where(support,concentration,0.)


def validate_mapped_molecular_record(public, record, config):
    """Check the forward rate mapping against original public action and compiled occupancy."""
    if not molecular_administration_bound(public,config):
        raise ValueError("Mapped molecular record requires its explicit public administration law")
    action = record["action"]
    snapshot = record["physical_program"]["inputs"]["run_snapshot"]
    settings = snapshot["settings"]
    policy = molecular_administration_policy(config)
    if (settings["route"] != policy["route"] or settings["dose_unit"] != "mg/kg"
            or settings["schedule"] != {"kind":"interval","interval":4.0,"unit":"hour"}
            or action["route"] != settings["route"] or action["schedule"] != settings["schedule"]):
        raise ValueError("Mapped molecular record route/schedule differs from the public law")
    # Registered regimen identity picks the original public source program before any target read.
    grid = family_grid(observable_profile(config),MOLECULAR)
    source = config["native_observable_bridge"]
    count = len(source.get("action_variants",source.get("action_multipliers",[])))
    matched = [action_definition(config,MOLECULAR,index) for index in range(count)]
    matched = [item for item in matched if item["regimen_uid"] == public.regimen_uid]
    if len(matched) != 1:
        raise ValueError("Mapped molecular record has no original public program identity")
    expected_levels = tuple(action_levels(grid,matched[0])[1:])
    if public.interval_values != expected_levels or tuple(grid["query_time"]["values"]) != public.source_times:
        raise ValueError("Mapped molecular rates do not preserve the original consumed driving intervals")
    if tuple(rate*public.administration.reference_time_hours for rate in public.administration_rates_mg_per_kg_hour) != expected_levels:
        raise ValueError("Mapped rate-to-drive arithmetic differs from its original numerical lineage")
    estimate = estimated_exposure_on_levels(config,snapshot["parameters"],public.source_times,(0.,*public.interval_values))
    original = np.asarray(record["physical_program"]["free_concentration_mol_per_l"])
    support = np.asarray(record["physical_program"]["support"],dtype=bool).reshape(len(public.source_times),36)
    expected = np.where(support,estimate["exposure_nmol_per_l"][:,None]*1e-9,0.)
    if not np.array_equal(original.reshape(expected.shape),expected):
        raise ValueError("Mapped rate-to-exposure arithmetic does not preserve original compiled concentration")


def validate_execution_library(state, inputs, library, config):
    """Tie parameter-free requests to existing admitted records and numerical encodings."""
    if (type(library) is not tuple or not library or any(type(item) is not NativeActuation for item in library)
            or tuple(item.regimen_uid for item in library) != state._registered_regimen_uids
            or tuple(inputs.regimen_uids) != state._registered_regimen_uids or inputs.group_shape[:2] != (1,32)):
        raise ValueError("Execution requires the original public action library and complete32 support")
    validate_administration_library(library,config)
    joint_particle_contexts(inputs.posterior,inputs.programs,inputs.regimen_uids)
    validate_candidate_parameters(inputs.posterior.support.physical_parameter_hypotheses,inputs.posterior.support)
    validate_program_parameter_subset(inputs.posterior.support,inputs.programs)
    binding=state._filter_binding
    for ai,public in enumerate(library):
        if (public.source_uid != binding["source_uid"] or public.source_snapshot_ref != binding["source_snapshot_ref"]
                or public.family_uid != binding["family_uid"]
                or public.source_times != tuple(float(t) for t in state._guide.clock)
                or public.source_time_unit != state._guide.grid["query_time"]["unit"]):
            raise ValueError("Public execution source/family/original clock differs from the model")
        for ji in range(32):
            record=inputs.programs["records"][0][ji][ai]
            if public.input_kind == "stimulation":
                offset=0 if state._guide.bind_source_intervals else 1
                waveform=record["action"]["stimulation_or_environment"]["magnitude_or_program"]
                levels=tuple(float(v) for v in waveform["values"])
                if (waveform["value_unit"] != public.input_unit or waveform["time_unit"] != public.source_time_unit
                        or tuple(waveform["time_values"]) != public.source_times
                        or levels[offset:offset+len(public.interval_values)] != public.interval_values):
                    raise ValueError("Public waveform differs from its admitted original compiler record")
                registered=state._registered_stimulus.reshape(1,32,len(library),-1)[0,ji,ai]
                if not torch.equal(registered,registered.new_tensor(levels)):
                    raise ValueError("Registered model stimulus changed from its admitted public waveform")
            else:
                occupancy,free=_model_estimated_occupancy(inputs,ji,ai,public,config)
                physical=record["physical_program"]
                original=np.asarray(physical["free_concentration_mol_per_l"]).reshape(free.shape)
                if not np.array_equal(free,original):
                    raise ValueError("Public controls do not exactly reconstruct the admitted frozen-PK compiler record")
                if public.family_uid == MOLECULAR:
                    if public.administration is not None:
                        validate_mapped_molecular_record(public,record,config)
                    # This inverse check authenticates the existing numerical encoding only.
                    # Public administered mass is derived forward from declared driving intervals.
                    guide=state._guide
                    exposure=torch.as_tensor(original,device=guide.device,dtype=torch.float64)[:,guide.ca1]*1e9
                    parameters={name:value.reshape(1,32,guide.a)[0,ji,0] for name,value in guide.parameters.items()}
                    level=((exposure[1:]-exposure[:-1])/guide.clock.diff()
                           +parameters["exposure_clearance_rate"]*exposure[:-1])/parameters["dose_to_exposure_gain"]
                    if (level < -1e-5).any():
                        raise ValueError("Admitted molecular estimate violates its original inverse input rule")
                    expected=torch.cat((level.new_zeros(1),level.clamp_min(0)))
                    actual=state._registered_stimulus.reshape(1,32,len(library),-1)[0,ji,ai]
                    if not torch.equal(actual,expected):
                        raise ValueError("Molecular numerical driver differs from its admitted inverse representation")
                else:
                    snapshot=physical["inputs"]["run_snapshot"]
                    schedule=record["action"]["schedule"]
                    expected=tuple((float(t),float(snapshot["pk_definition"]["total_dose_mg_per_kg"]),float(f))
                        for t,f in zip(schedule["values"],schedule["dose_fractions"]))
                    if expected != tuple((event.source_time,event.total_dose_mg_per_kg,event.fraction) for event in public.dose_events):
                        raise ValueError("Public dose events differ from the original model input record")


def execution_program(previous, selected, public, candidate, inputs, config):
    """New model-conditioning version; previously consumed versions remain immutable."""
    guide=selected._guide
    contexts=joint_particle_contexts(inputs.posterior,inputs.programs,inputs.regimen_uids)
    scales=config["native_controlled_comparators"]["conditioning"]["program_reference_scales"]
    names=("occupancy","support","physical_time",*FORCING_ARRAY_NAMES)
    arrays=[]
    for name in names:
        value=torch.stack([row[name][:,candidate] for row in contexts],1).reshape(32,len(public.source_times),36,1).clone()
        if name=="occupancy" and public.input_kind != "stimulation":
            values=np.stack([_model_estimated_occupancy(inputs,ji,candidate,public,config)[0] for ji in range(32)])
            value=value.new_tensor(values[...,None])
        if name in {"electrical_current_uA","optical_power_mW"} and public.input_kind=="stimulation":
            active="electrical_current_uA" if public.family_uid==ELECTRICAL else "optical_power_mW"
            if name==active:
                offset=0 if guide.bind_source_intervals else 1
                node=guide.ca1 if public.family_uid==ELECTRICAL else guide.ca3
                value[:,offset:offset+len(public.interval_values),node,0]=value.new_tensor(public.interval_values)
        arrays.append(value/scales[name])
    program=torch.cat(arrays,-1).to(previous.program)
    first=selected.query_index if guide.bind_source_intervals and public.family_uid in {ELECTRICAL,SYNAPTIC} else selected.query_index+1
    parent=selected._planning_origin["executed_regimen_uid"]
    parent_index=previous.rhs._filter_binding["regimen_uids"].index(parent)
    old=previous.program.reshape(1,32,previous.rhs._guide.a,*previous.program.shape[1:])[:,:,parent_index].reshape_as(program)
    program[:,:first]=old[:,:first]
    return program.detach().clone()


def execution_record_identity(record):
    template=record["template"]
    return digest({"public":actuation_payload(record["public"]),"request":actuation_payload(record["request"]),
        "parent_model_uid":record["parent_model_uid"],"model_uid":record["model_uid"],
        "conditioning_authority":record["conditioning_authority"],
        "query_index":record["query_index"],"filter_revision":record["filter_revision"],
        "observation_uids":record["observation_uids"],"constants":execution_constants_identity(template),
        "controls":model_control_identity(template),"intervals":template._intervals,
        "program":_tensor_identity(record["program"]),"static":_tensor_identity(record["static"])})


def validate_execution_state(state):
    if state._public_control is None:
        if state._planning_origin is not None:
            raise ValueError("Unactivated model-side planning branch has no physical execution binding")
        return
    if (execution_constants_identity(state) != state._execution_constants_identity
            or execution_library_identity(state) != state._execution_library_identity
            or model_control_identity(state) != state._control_identity
            or state._execution_model_uid not in state._filter_binding["regimen_uids"]):
        raise ValueError("Executed source/tuple/parameter/control binding changed")
    for record in state._execution_history:
        if record["identity"] != execution_record_identity(record):
            raise ValueError("Executed control or model-conditioning history was modified")
    if state._execution_history:
        last=state._execution_history[-1]
        if last["public"] != state._public_control or last["model_uid"] != state._execution_model_uid:
            raise ValueError("Current execution does not follow its actual switch history")


def conditioning_identity(program, static):
    return digest({"program":_tensor_identity(program),"static":_tensor_identity(static)})


def public_suffix_candidates(state):
    """Enumerate the original ordered library; retain each infeasible public budget result."""
    validate_execution_state(state)
    if state._public_control is None:
        raise ValueError("Public suffix selection requires an actual executed control")
    prefix = executed_prefix(state._public_control, state.query_index)
    candidates = []
    for action in state._execution_library:
        request = NativeActuationRequest(prefix, action)
        try:
            public = compose_actuation(request, state._execution_library)
        except ValueError as error:
            if str(error) not in {
                "Composite actuation exceeds the registered total public dose budget",
                "Composite actuation exceeds the registered original-clock input integral",
            }:
                raise
            candidates.append({"regimen_uid": action.regimen_uid, "status": "outside_public_execution_budget",
                               "reason": str(error), "request": request, "public": None})
        else:
            candidates.append({"regimen_uid": action.regimen_uid, "status": "legal",
                               "request": request, "public": public})
    return candidates
