"""Explicit artificial source-law inputs; no generated/held-out world or model fit."""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
from Libs.utils.model import controlled_cns_source as cns
from Libs.utils.model import native_observable_source as observable
from Libs.utils.model.native_arrived_observation import NativeAcquisition, NativeArrivedObservation
from Libs.utils.model.native_scientific_source import action_definition, long_cns_definition
from Libs.utils.evaluation import native_physical_plant as plant_module
from Libs.utils.evaluation.native_physical_plant import (
    NativePhysicalPlant, NativeActuationRequest, native_actuation_library,
)

FAMILIES = [observable.ELECTRICAL, observable.SYNAPTIC, observable.MOLECULAR, "cns_long_horizon_world"]


def explicit_world(family, *, action_index=0):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    settings = config["native_observable_bridge"]
    settings.pop("source_variants",None)
    settings.update(source_uid="artificial_plant_source", snapshot_ref="artificial_source_definition",
        active_source_law="original_discrete_source", observation_matrix=np.eye(6).tolist())
    group = {"family_uid":family, "sample_uid":"explicit_sample", "group_uid":"explicit_world",
        "support_uid":"public_support", "posterior_uid":"public_origin", "role":"train",
        "particle_uids":["public_hypothesis_"+str(i) for i in range(32)]}
    source = config["controlled_cns_source"]
    # Nominal, explicitly constructed software state; no latent draw or tuple selection.
    initial = np.zeros((6,6,26),dtype=np.float32)
    for name in cns.DYNAMIC:
        initial[...,source["grid"]["species_id"].index(name)] = 0.1
    factors = {name:np.ones(len(basis),dtype=np.float64) for name,basis in source["factor_basis"].items()}
    params = (None if family == FAMILIES[-1] else
              {name:float(row["value"]) for name,row in settings["physical_families"][family]["parameters"].items()})
    # Match the original producer order exactly: explicit private DYNAMIC,
    # background pre-dose core quantities, then supported source-native readouts;
    # every write passes through the original float32 initial-field storage.
    background = long_cns_definition(config) if params is None else source
    law = cns.ControlledLaw(background,factors,initial[...,:15])
    dynamic = initial[...,:15].reshape(1,36,15)[:,:,[source["grid"]["species_id"].index(name) for name in cns.DYNAMIC]]
    initial[...,:15] = law.quantities([0.],dynamic,background["actions"][0])[0]
    grid = family_grid(observable_profile(config),family)
    source_mask = quantity_support(grid,settings["source_uid"])[0]
    for qi,quantity in enumerate(grid["quantity_profile"]["quantities"]):
        if quantity.get("family_uid") != family:
            continue
        site=quantity["incidence"]
        ri=grid["region_id"].index(site["anchor_region_id"])
        ci=grid["compartment_id"].index(site["anchor_compartment_id"])
        if source_mask[ri,ci,qi]:
            initial[ri,ci,qi] = params.get("initial_"+quantity["observable_name"],0.)
    library = native_actuation_library(config,group)
    generator = torch.Generator().manual_seed(9107)
    plant = NativePhysicalPlant(config,group,private_initial_field=initial,private_factors=factors,
        private_physical_parameters=params,initial_actuation=library[action_index],measurement_generator=generator)
    return config,group,initial,factors,params,library,plant


def acquisition(plant):
    query = plant._query+1
    return NativeAcquisition(plant._readout,plant._group["support_uid"],plant._group["posterior_uid"],
        (plant._group["sample_uid"],),(tuple(plant._group["particle_uids"]),),plant._query,query,
        plant._source_times[query],plant._grid["query_time"]["unit"],float(plant._times[query]),plant._active_uid)


@pytest.fixture(scope="module")
def original_source_modules():
    path = os.environ.get("NRNSIFT_PLANT_BASELINE")
    if not path:
        pytest.skip("Opt-in source equivalence requires NRNSIFT_PLANT_BASELINE pinned to commit 36de3306")
    modules = {}
    for name in ("runtime","cns","observable"):
        spec = importlib.util.spec_from_file_location("_plant_original_"+name,Path(path)/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    old = modules["observable"]
    old.ControlledLaw = modules["cns"].ControlledLaw
    for name in ("electrophysiology","calcium_synaptic","molecular_pharmacological"):
        setattr(old,"_simulate_"+name,getattr(modules["runtime"],"_simulate_"+name))
    return modules


@pytest.mark.parametrize("family",FAMILIES)
def test_fixed_full_original_clock_matches_prechange_source_exactly(family,original_source_modules):
    config,group,initial,factors,params,library,plant = explicit_world(family)
    action = action_definition(config,family,0,group)
    if params is None:
        definition = long_cns_definition(config)
        definition["pk"]["total_dose_mg_per_kg"] *= action.get("total_dose_multiplier",1.)
        law = original_source_modules["cns"].ControlledLaw(definition,factors,initial[...,:15])
        reference = original_source_modules["cns"].integrate(law,action,"DOP853",
            definition["numerics"]["reference_max_step_hours"])
        expected = np.zeros((len(plant._times),6,6,26))
        expected[...,:15] = reference["trajectory"]
    else:
        execution = copy.deepcopy(action)
        if execution.get("physical_update_clock") == "left_interval_current":
            execution["response_delay_steps"] += 1
        old = original_source_modules["observable"]
        physical = old.physical_trajectory(config,family,params,execution)
        support = SimpleNamespace(initial_field=torch.from_numpy(initial[None,None]),
            factors={key:torch.from_numpy(value[None,None]) for key,value in factors.items()},
            source_uid=library[0].source_uid)
        law = old.CoupledObservableLaw(config,group,support,0,physical)
        numerics = config["native_observable_bridge"]["numerics"]
        reference = old.integrate_coupled(law,"DOP853",numerics["cns_max_step_interval_fractions"][-1])
        expected = reference["trajectory"]
    packets=[]
    for _ in plant._times[1:]:
        packets.append(plant.acquire(acquisition(plant)))
    actual = np.stack([item["field"] for item in plant._history])
    np.testing.assert_array_equal(actual,expected)
    assert plant._work["nfev"] == reference["nfev"]
    if params is not None:
        np.testing.assert_array_equal(np.stack([row["physical"] for row in plant._history]),physical["values"])
        np.testing.assert_array_equal([row["state"][-2] for row in plant._history],reference["boundary_mass_input"])
        np.testing.assert_array_equal([row["state"][-1] for row in plant._history],reference["eliminated_mass"])
    assert all(type(packet) is NativeArrivedObservation and set(vars(packet)) == {"acquisition","observation"}
               for packet in packets)
    out = os.environ.get("NRNSIFT_PLANT_EVIDENCE")
    if out:
        Path(out).mkdir(parents=True,exist_ok=True)
        (Path(out)/(family+".json")).write_text(json.dumps({"scope":"explicit_artificial_software_world",
            "original_query_count":len(plant._times),"packet_count":len(packets),"nfev":plant._work["nfev"],
            "prechange_full_field_comparison":"BITWISE_EQUAL",
            "reference_initial_replaced":False,
            "persisted_initial_to_source_t0_max_abs":float(np.max(np.abs(initial.astype(np.float64)-expected[0]))),
            "original_dataset_generation":False,"protected_reads":False,
            "scientific_qualification":"NOT_EVALUATED"},indent=2)+"\n")


@pytest.mark.parametrize("family",FAMILIES)
def test_two_switches_preserve_one_world_and_consumed_content(family,original_source_modules):
    config,group,initial,factors,params,library,plant = explicit_world(family)
    # Start with the maximum-input library row so legal suffix replacement never
    # needs a larger budget. Subsequent choices are bounded by actual composition.
    if params is None:
        amount=lambda item:sum(event.amount_mg_per_kg for event in item.dose_events)
    else:
        amount=lambda item:sum(abs(v)*dt for v,dt in zip(item.interval_values,np.diff(item.source_times)))
    largest=max(library,key=amount)
    plant=NativePhysicalPlant(config,group,private_initial_field=initial,private_factors=factors,
        private_physical_parameters=params,initial_actuation=largest,measurement_generator=torch.Generator().manual_seed(5))
    plant.acquire(acquisition(plant))
    before_state=plant._state.copy();before_rng=plant._generator.get_state().clone()
    prefix=plant.prefix
    lowest=min(library,key=amount)
    plant.switch(NativeActuationRequest(prefix,lowest))
    first_uid=plant.prefix.executed_regimen_uid
    np.testing.assert_array_equal(plant._state,before_state)
    assert torch.equal(plant._generator.get_state(),before_rng)
    assert plant.prefix.interval_values==prefix.interval_values and plant.prefix.dose_events==prefix.dose_events
    with pytest.raises(ValueError):plant.switch(NativeActuationRequest(plant.prefix,lowest))
    plant.acquire(acquisition(plant))
    second_prefix=plant.prefix
    earlier=[row["field"].copy() for row in plant._history]
    # A second switch uses the first composite's actual prefix, not a library restart.
    if params is None:
        second = next(item for item in library if item != lowest and item != largest
            and any(event.source_time > plant._clock for event in item.dose_events)
            and sum(event.amount_mg_per_kg for event in second_prefix.dose_events)
                +sum(event.amount_mg_per_kg for event in item.dose_events if event.source_time > plant._clock)
                <= amount(largest))
    else:
        durations = np.diff(plant._source_times)
        second = next(item for item in library if item != lowest
            and item.interval_values[plant._query:] != lowest.interval_values[plant._query:]
            and sum(abs(v)*dt for v,dt in zip(second_prefix.interval_values+item.interval_values[plant._query:],durations))
                <= amount(largest))
    plant.switch(NativeActuationRequest(second_prefix,second))
    assert plant.prefix.executed_regimen_uid!=first_uid
    while plant._query+1 < len(plant._times):plant.acquire(acquisition(plant))
    for old,new in zip(earlier,plant._history):np.testing.assert_array_equal(old,new["field"])
    assert plant._query==len(plant._times)-1 and plant._clock==plant._times[-1]
    if params is not None:
        assert plant.prefix.interval_values[:2]==second_prefix.interval_values
        old_runtime = original_source_modules["runtime"]
        old_action_level = old_runtime._action_level
        old_simulate = {observable.ELECTRICAL:old_runtime._simulate_electrophysiology,
            observable.SYNAPTIC:old_runtime._simulate_calcium_synaptic,
            observable.MOLECULAR:old_runtime._simulate_molecular_pharmacological}[family]
        original_action = action_definition(config,family,library.index(largest),group)
        oracle_state = {name:value for name,value in params.items() if name.startswith("initial_")}
        if family == observable.ELECTRICAL:
            oracle_state["initial_spike_rate"] = 0.
        try:
            old_runtime._action_level = lambda times,index,delay,duration,intensity: plant.prefix.interval_values[index-1]
            composite_rows = old_simulate({"world":{"time_grid":{"values":plant._source_times}}},
                {"oracle_state":oracle_state,"oracle_parameters":params},
                {"controlled_extension":original_action["extension"],"response_delay_steps":original_action["response_delay_steps"]})
        finally:
            old_runtime._action_level = old_action_level
        composite_physical = np.asarray([[row[name] for name in plant._law.physical_names] for row in composite_rows])
        np.testing.assert_array_equal(np.stack([row["physical"] for row in plant._history]),composite_physical)
        support = SimpleNamespace(initial_field=torch.from_numpy(initial[None,None]),
            factors={key:torch.from_numpy(value[None,None]) for key,value in factors.items()},
            source_uid=library[0].source_uid)
        carried_physical = {"values":composite_physical}
        original_law = original_source_modules["observable"].CoupledObservableLaw(config,group,support,0,carried_physical)
        reference = original_source_modules["observable"].integrate_coupled(original_law,"DOP853",
            config["native_observable_bridge"]["numerics"]["cns_max_step_interval_fractions"][-1])
        np.testing.assert_array_equal(np.stack([row["field"] for row in plant._history]),reference["trajectory"])
        fixed_action=copy.deepcopy(original_action)
        if fixed_action.get("physical_update_clock") == "left_interval_current":
            fixed_action["response_delay_steps"] += 1
        fixed_physical = original_source_modules["observable"].physical_trajectory(config,family,params,fixed_action)
        fixed_law = original_source_modules["observable"].CoupledObservableLaw(config,group,support,0,fixed_physical)
        fixed_reference = original_source_modules["observable"].integrate_coupled(fixed_law,"DOP853",
            config["native_observable_bridge"]["numerics"]["cns_max_step_interval_fractions"][-1])
        np.testing.assert_array_equal(reference["trajectory"][:2],fixed_reference["trajectory"][:2])
        assert not np.array_equal(reference["trajectory"][2:],fixed_reference["trajectory"][2:])
    else:
        assert plant.prefix.dose_events[:len(second_prefix.dose_events)]==second_prefix.dose_events
        assert len(plant.prefix.dose_events)>len(second_prefix.dose_events)
        assert plant._action()["fractions"] == [event.fraction for event in plant.prefix.dose_events]
        # Independent original PK kernels, each with its true original total and
        # fraction, must sum to the carried mixed history without renormalization.
        expected = np.zeros((len(plant._times),36))
        for event in plant.prefix.dose_events:
            definition=copy.deepcopy(plant._law.source)
            definition["pk"]["total_dose_mg_per_kg"]=event.total_dose_mg_per_kg
            original_action={"dose_times_hours":[event.source_time],"fractions":[event.fraction]}
            expected += original_source_modules["cns"].pk_values(plant._times,plant._law.pk,original_action,definition)[0]
        np.testing.assert_array_equal(cns.pk_values(plant._times,plant._law.pk,plant._action(),plant._law.source)[0],expected)
        # Old integrate supplies the independent full-clock event splitter;
        # ControlledLaw consumes the already independently checked mixed PK.
        reference = original_source_modules["cns"].integrate(plant._law,plant._action(),"DOP853",
            plant._law.source["numerics"]["reference_max_step_hours"])
        actual = np.stack([row["field"] for row in plant._history])
        expected_field = np.zeros_like(actual)
        expected_field[...,:15] = reference["trajectory"]
        np.testing.assert_array_equal(actual,expected_field)
        original_action=action_definition(config,family,library.index(largest),group)
        definition=long_cns_definition(config)
        definition["pk"]["total_dose_mg_per_kg"]*=original_action.get("total_dose_multiplier",1.)
        original_law=original_source_modules["cns"].ControlledLaw(definition,factors,initial[...,:15])
        fixed_reference=original_source_modules["cns"].integrate(original_law,original_action,"DOP853",
            definition["numerics"]["reference_max_step_hours"])
        np.testing.assert_array_equal(actual[1:2,...,:15],fixed_reference["trajectory"][1:2])
        assert not np.array_equal(actual[2:,...,:15],fixed_reference["trajectory"][2:])


@pytest.mark.parametrize("fault",["uid_content","prefix","clock","readout","query","model_object"])
def test_bad_requests_rejected_without_work_or_noise(fault):
    *_,library,plant=explicit_world(observable.ELECTRICAL)
    plant.acquire(acquisition(plant))
    state=plant._state.copy();rng=plant._generator.get_state().clone();work=plant._work.copy();prefix=plant.prefix
    with pytest.raises((ValueError,TypeError)):
        if fault=="uid_content":plant.switch(NativeActuationRequest(prefix,replace(library[0],interval_values=tuple(v+1 for v in library[0].interval_values))))
        elif fault=="prefix":plant.switch(NativeActuationRequest(replace(prefix,interval_values=(99.,)),library[0]))
        elif fault=="clock":plant.switch(NativeActuationRequest(replace(prefix,source_time=prefix.source_time+1),library[0]))
        elif fault=="readout":plant.acquire(replace(acquisition(plant),readout=replace(plant._readout,standard_deviation=99.)))
        elif fault=="query":plant.acquire(replace(acquisition(plant),query_index=plant._query))
        else:plant.switch(SimpleNamespace(prefix=prefix,actuation=library[0],parameters=torch.ones(32,5)))
    np.testing.assert_array_equal(plant._state,state)
    assert torch.equal(plant._generator.get_state(),rng) and plant._work==work and plant.prefix==prefix
    assert plant._failed is None


def test_failed_provider_keeps_progress_and_consumed_noise(monkeypatch):
    *_,plant=explicit_world(observable.MOLECULAR)
    original=plant_module.produce_native_arrived_observation
    rng=plant._generator.get_state().clone()
    def failing(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError("injected after measurement noise")
    monkeypatch.setattr(plant_module,"produce_native_arrived_observation",failing)
    request=acquisition(plant)
    with pytest.raises(RuntimeError,match="after measurement"):plant.acquire(request)
    assert plant._query==1 and plant._clock==plant._times[1] and plant._work["nfev"]>0
    assert not torch.equal(plant._generator.get_state(),rng)
    work=plant._work.copy();rng=plant._generator.get_state().clone();state=plant._state.copy()
    with pytest.raises(RuntimeError,match="retry is prohibited"):plant.acquire(request)
    assert plant._work==work and torch.equal(plant._generator.get_state(),rng)
    np.testing.assert_array_equal(plant._state,state)


def test_failed_physics_keeps_accepted_progress_and_does_not_measure(monkeypatch):
    *_,plant=explicit_world(observable.ELECTRICAL)
    original=plant._law.rhs
    state=plant._state.copy();rng=plant._generator.get_state().clone()
    def failing(*args,**kwargs):
        if plant._clock>0:raise RuntimeError("injected after accepted step")
        return original(*args,**kwargs)
    monkeypatch.setattr(plant._law,"rhs",failing)
    request=acquisition(plant)
    with pytest.raises(RuntimeError,match="accepted step"):plant.acquire(request)
    assert plant._clock>0 and plant._work["nfev"]>0 and not np.array_equal(plant._state,state)
    assert torch.equal(plant._generator.get_state(),rng) and plant._query==0
    work=plant._work.copy()
    with pytest.raises(RuntimeError,match="retry is prohibited"):plant.acquire(request)
    assert plant._work==work


@pytest.mark.parametrize("family",FAMILIES[:3])
def test_same_query_cannot_repeat_measurement(family):
    *_,plant=explicit_world(family)
    request=acquisition(plant);plant.acquire(request)
    rng=plant._generator.get_state().clone();work=plant._work.copy()
    with pytest.raises(ValueError):plant.acquire(request)
    assert torch.equal(plant._generator.get_state(),rng) and plant._work==work


def test_per_event_totals_preserve_original_pk_operation_order():
    config,_,initial,factors,_,_,_=explicit_world(FAMILIES[-1])
    source=long_cns_definition(config);law=cns.ControlledLaw(source,factors,initial[...,:15])
    action=source["actions"][-1]
    total=source["pk"]["total_dose_mg_per_kg"]*action.get("total_dose_multiplier",1.)
    scaled=copy.deepcopy(source);scaled["pk"]["total_dose_mg_per_kg"]=total
    composite=dict(action,event_total_dose_mg_per_kg=[total]*len(action["fractions"]))
    clock=sorted(set([0.,1.,672.,*action["dose_times_hours"]]))
    for precise in [False,True]:
        for actual,expected in zip(cns.pk_values(clock,law.pk,composite,source,high_precision=precise),
                                   cns.pk_values(clock,law.pk,action,scaled,high_precision=precise)):
            np.testing.assert_array_equal(actual,expected)
    # Later events do not leak into a preceding interval endpoint stage.
    event=action["dose_times_hours"][-1]
    if event>0:
        before=cns.pk_values([event],law.pk,composite,source,initial_pre_dose=False,active_through=event-1)[0]
        after=cns.pk_values([event],law.pk,composite,source,initial_pre_dose=False,active_through=event)[0]
        assert (after>before).all()


def test_shared_dop853_core_matches_original_solve_ivp_dense_endpoint():
    rhs=lambda t,y:np.asarray([-0.13*y[0]+0.003*t,-0.21*y[1]+0.7*y[0]])
    initial=np.asarray([0.2,0.8]);atol=np.asarray([1e-12,1e-12])
    original=solve_ivp(rhs,(0.,7.),initial,method="DOP853",max_step=.25,rtol=1e-10,atol=atol,t_eval=[7.])
    actual,nfev=cns.integrate_source_interval(rhs,0.,7.,initial,"DOP853",.25,rtol=1e-10,atol=atol)
    np.testing.assert_array_equal(actual,original.y[:,-1])
    assert nfev==original.nfev


@pytest.mark.parametrize("law",["cooperative_exponential_source","romani_vm_source","invented_source"])
def test_unsupported_law_is_rejected(law):
    config,group,*_=explicit_world(observable.ELECTRICAL)
    config["native_observable_bridge"]["active_source_law"]=law
    with pytest.raises(ValueError,match="original primary source"):native_actuation_library(config,group)


def test_molecular_public_input_stays_model_drive_without_delivered_dose():
    *_,library,plant=explicit_world(observable.MOLECULAR)
    assert all(item.input_kind=="model_driving_level" and not item.dose_events for item in library)
    assert not hasattr(plant,"forecast") and plant._law.physical is None
    assert all(not hasattr(item,name) for item in library for name in ["parameters","occupancy","guide","initial_state"])


@pytest.mark.parametrize("family",FAMILIES)
def test_original_rk4_source_wrapper_preserves_full_clock(family,original_source_modules):
    config,group,initial,factors,params,_,plant=explicit_world(family)
    action=action_definition(config,family,0,group)
    if params is None:
        definition=long_cns_definition(config)
        definition["pk"]["total_dose_mg_per_kg"]*=action.get("total_dose_multiplier",1.)
        old_law=original_source_modules["cns"].ControlledLaw(definition,factors,initial[...,:15])
        new_law=cns.ControlledLaw(definition,factors,initial[...,:15])
        old=original_source_modules["cns"].integrate(old_law,action,"RK4",definition["numerics"]["reference_max_step_hours"])
        new=cns.integrate(new_law,action,"RK4",definition["numerics"]["reference_max_step_hours"])
    else:
        support=SimpleNamespace(initial_field=torch.from_numpy(initial[None,None]),
            factors={key:torch.from_numpy(value[None,None]) for key,value in factors.items()},source_uid=plant._readout.source_uid)
        physical=original_source_modules["observable"].physical_trajectory(config,family,params,action)
        old_law=original_source_modules["observable"].CoupledObservableLaw(config,group,support,0,physical)
        new_law=observable.CoupledObservableLaw(config,group,support,0,physical)
        fraction=config["native_observable_bridge"]["numerics"]["cns_max_step_interval_fractions"][0]
        old=original_source_modules["observable"].integrate_coupled(old_law,"RK4",fraction)
        new=observable.integrate_coupled(new_law,"RK4",fraction)
    for key in old:
        np.testing.assert_array_equal(new[key],old[key])


def test_original_signed_current_library_and_other_input_sign_boundaries():
    config,group,initial,factors,params,_,_=explicit_world(observable.ELECTRICAL)
    settings=config["native_observable_bridge"]["physical_families"][observable.ELECTRICAL]
    settings["action"]["extension"]["dose_or_intensity"]["value"] *= -1
    library=native_actuation_library(config,group)
    assert any(value<0 for item in library for value in item.interval_values)
    plant=NativePhysicalPlant(config,group,private_initial_field=initial,private_factors=factors,
        private_physical_parameters=params,initial_actuation=library[0],measurement_generator=torch.Generator().manual_seed(8))
    while not any(plant.prefix.interval_values):
        plant.acquire(acquisition(plant))
    assert any(value<0 for value in plant.prefix.interval_values)
    for family in [observable.SYNAPTIC,observable.MOLECULAR]:
        config,group,*_=explicit_world(family)
        config["native_observable_bridge"]["physical_families"][family]["action"]["extension"]["dose_or_intensity"]["value"] *= -1
        with pytest.raises(ValueError,match="nonnegative-drive"):
            native_actuation_library(config,group)
        # A typed payload also rejects negative optical or molecular drive.
        *_,registered,_=explicit_world(family)
        with pytest.raises(ValueError,match="typed units"):
            replace(registered[0],interval_values=tuple(-1. for _ in registered[0].interval_values))


def test_supported_initial_auxiliary_uses_original_float32_consistency():
    config,group,initial,factors,params,library,plant=explicit_world(observable.ELECTRICAL)
    qi=plant._grid["species_id"].index("electrical_membrane_voltage")
    quantity=plant._grid["quantity_profile"]["quantities"][qi]
    ri=plant._grid["region_id"].index(quantity["incidence"]["anchor_region_id"])
    ci=plant._grid["compartment_id"].index(quantity["incidence"]["anchor_compartment_id"])
    assert plant._law.quantity_support[ri,ci,qi]
    equivalent=initial.astype(np.float64)
    equivalent[ri,ci,qi]=np.nextafter(equivalent[ri,ci,qi],np.inf)
    # Preserve the original independent t0 derived-field convention; do not
    # impose a new coupled-source derived-value constraint on this stored field.
    equivalent[...,plant._grid["species_id"].index("clearance_flux")]+=1.
    NativePhysicalPlant(config,group,private_initial_field=equivalent,private_factors=factors,
        private_physical_parameters=params,initial_actuation=library[0],measurement_generator=torch.Generator())
    bad=initial.copy()
    bad[ri,ci,qi]=np.nextafter(bad[ri,ci,qi],np.float32(np.inf))
    with pytest.raises(ValueError,match="float32 auxiliary state"):
        NativePhysicalPlant(config,group,private_initial_field=bad,private_factors=factors,
            private_physical_parameters=params,initial_actuation=library[0],measurement_generator=torch.Generator())


@pytest.mark.parametrize("family", FAMILIES)
def test_split_advance_acquisition_matches_combined_physics_and_noise(family):
    *_, combined = explicit_world(family)
    split = copy.deepcopy(combined)
    request = acquisition(split)
    expected = combined.acquire(request)
    prefix, rng = split.prefix, split._generator.get_state().clone()
    receipt = split.advance(prefix)
    assert receipt == split.prefix and receipt.query_index == 1
    assert not split._acquired_queries and torch.equal(split._generator.get_state(), rng)
    work = split._work.copy()
    with pytest.raises(ValueError, match="current original clock"):
        split.advance(prefix)
    assert split._work == work and split._failed is None
    actual = split.acquire_current(request)
    assert actual.acquisition == expected.acquisition
    torch.testing.assert_close(actual.observation.values, expected.observation.values, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(actual.observation.observed, expected.observation.observed)
    assert torch.equal(split._generator.get_state(), combined._generator.get_state())
    np.testing.assert_array_equal(split._state, combined._state)
    np.testing.assert_array_equal(split._field, combined._field)
    assert split.prefix == combined.prefix and split._work == combined._work
    assert split._acquired_queries == (1,)
    rng = split._generator.get_state().clone()
    with pytest.raises(ValueError, match="repeats a measurement"):
        split.acquire_current(request)
    assert torch.equal(split._generator.get_state(), rng) and split._work == work


@pytest.mark.parametrize("family", FAMILIES)
def test_unobserved_committed_tail_matches_full_acquired_source_without_noise(family):
    config, group, initial, factors, parameters, library, _ = explicit_world(family)
    def amount(row):
        return (sum(event.amount_mg_per_kg for event in row.dose_events) if row.dose_events else
                sum(abs(value)*dt for value, dt in zip(row.interval_values, np.diff(row.source_times))))
    largest, lowest = max(library, key=amount), min(library, key=amount)
    plant = NativePhysicalPlant(config, group, private_initial_field=initial, private_factors=factors,
        private_physical_parameters=parameters, initial_actuation=largest,
        measurement_generator=torch.Generator().manual_seed(923))
    plant.acquire(acquisition(plant))
    committed = plant.switch(NativeActuationRequest(plant.prefix, lowest))
    observed = copy.deepcopy(plant)
    pending = acquisition(plant)
    prefix, rng = plant.prefix, plant._generator.get_state().clone()
    with pytest.raises(ValueError, match="current committed control prefix"):
        plant._complete_unobserved_tail(replace(prefix, source_time=prefix.source_time+1))
    assert not plant._tail_started and plant._failed is None
    plant._complete_unobserved_tail(prefix)
    while observed._query+1 < len(observed._times):
        observed.acquire(acquisition(observed))
    assert plant._tail_complete and plant._public_program == committed
    assert plant._acquired_queries == (1,) and torch.equal(plant._generator.get_state(), rng)
    assert plant.prefix == observed.prefix and plant._work == observed._work
    actual, expected = plant._evaluation_history(), observed._evaluation_history()
    assert actual["source_horizon_complete"] and actual["source_failure"] is None
    np.testing.assert_array_equal(actual["private_fields"], expected["private_fields"])
    assert actual["source_query_indices"] == list(range(len(plant._times)))
    for operation in (lambda: plant.advance(plant.prefix), lambda: plant.acquire(pending),
                      lambda: plant.acquire_current(pending),
                      lambda: plant.switch(NativeActuationRequest(plant.prefix, lowest)),
                      lambda: plant._complete_unobserved_tail(plant.prefix)):
        with pytest.raises(RuntimeError, match="use is closed"):
            operation()
    assert torch.equal(plant._generator.get_state(), rng) and plant._work == observed._work


def test_selected_request_rejects_before_advance_and_can_be_corrected():
    *_, plant = explicit_world(observable.ELECTRICAL)
    request, prefix = acquisition(plant), plant.prefix
    rng = plant._generator.get_state().clone()
    with pytest.raises(ValueError, match="available mixed readout"):
        plant.acquire(replace(request, selected_coordinate=99))
    assert plant.prefix == prefix and plant._work["nfev"] == 0
    assert plant._failed is None and torch.equal(plant._generator.get_state(), rng)
    coordinate = next(index for index, retained in enumerate(plant._readout.observed) if retained)
    packet = plant.acquire(replace(request, selected_coordinate=coordinate))
    assert int(packet.observation.observed.sum()) == 1
    assert torch.count_nonzero(packet.observation.values[~packet.observation.observed]) == 0


def test_unobserved_tail_failure_keeps_progress_and_cannot_complete(monkeypatch):
    *_, plant = explicit_world(observable.ELECTRICAL)
    original = plant._law.rhs
    rng = plant._generator.get_state().clone()
    def failing(*args, **kwargs):
        if plant._clock > 0:
            raise RuntimeError("injected tail failure after accepted step")
        return original(*args, **kwargs)
    monkeypatch.setattr(plant._law, "rhs", failing)
    prefix = plant.prefix
    with pytest.raises(RuntimeError, match="injected tail failure"):
        plant._complete_unobserved_tail(prefix)
    assert plant._clock > 0 and plant._work["nfev"] > 0
    assert plant._tail_started and not plant._tail_complete
    assert not plant._evaluation_history()["source_horizon_complete"]
    assert plant._evaluation_history()["private_fields"] is None
    work = plant._work.copy()
    with pytest.raises(RuntimeError, match="retry is prohibited"):
        plant._complete_unobserved_tail(prefix)
    assert plant._work == work and torch.equal(plant._generator.get_state(), rng)
