"""Explicit controlled-rate metadata and causal public mass accounting; no model/source run."""
import copy
from dataclasses import replace, asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.native_actuation import (
    NativeActuation, NativeMolecularAdministration, NativeActuationRequest,
    molecular_administration_policy, native_actuation_library, validate_administration_library,
    executed_prefix, compose_actuation, actuation_payload, actuation_from_payload,
)
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.evaluation.native_intervention_decisions import composed_action_burden, compiled_action_burden
from Libs.utils.model.native_future_controls import validate_mapped_molecular_record
from Libs.utils.model.native_observable_source import MOLECULAR


@pytest.fixture
def config():
    return load_config("Libs/configs/exp/native_molecular_administered_rate.yaml")


def group(action_indices=None):
    value = {"family_uid":MOLECULAR,"source_variant":"primary"}
    if action_indices is not None:
        value["action_indices"] = action_indices
    return value


def test_forward_rates_actual_totals_and_pre_post_interval_clock(config):
    library = native_actuation_library(config,group())
    expected_totals = (1.6,2.4,1.6,2.4,1.6,1.8)
    for public,total in zip(library,expected_totals):
        assert public.administered_mass() == pytest.approx(total)
        assert public.administration.maximum_supported_total_mg_per_kg == 2.4
        assert public.administration.numerical_source_uid == public.source_uid
        assert public.administration_rates_mg_per_kg_hour == tuple(u/4. for u in public.interval_values)
        assert executed_prefix(public,0).administered_mass_mg_per_kg == 0.
    assert library[0].interval_values == (1.6,1.6,1.6,1.6,0.,0.)
    assert library[4].interval_values == (0.,0.,3.2,3.2,0.,0.)
    assert [library[4].administered_mass(k) for k in range(7)] == [0.,0.,0.,.8,1.6,1.6,1.6]
    # Distinct legacy amplitudes with equal forward mass are not relabeled as equal amplitude.
    assert max(library[4].interval_values) == 3.2 and library[4].administered_mass() == 1.6


def test_support_uses_actual_selected_library_not_amplitude_maximum(config):
    library = native_actuation_library(config,group([4,5]))
    assert len(library) == 2
    assert library[0].administration.maximum_supported_total_mg_per_kg == 1.8
    assert max(library[1].interval_values) == 3.6
    validate_administration_library(library,config)
    wrong = tuple(replace(item,administration=replace(item.administration,maximum_supported_total_mg_per_kg=4.)) for item in library)
    with pytest.raises(ValueError,match="action-set"):
        validate_administration_library(wrong,config)


def test_repeated_switch_retains_actual_prefix_mass_and_future_absolute_intervals(config):
    library = native_actuation_library(config,group())
    first = compose_actuation(NativeActuationRequest(executed_prefix(library[0],1),library[4]),library)
    assert first.interval_values == (1.6,0.,3.2,3.2,0.,0.)
    assert first.administered_mass(1) == .4 and first.administered_mass() == 2.
    second = compose_actuation(NativeActuationRequest(executed_prefix(first,3),library[0]),library)
    assert second.interval_values == (1.6,0.,3.2,1.6,0.,0.)
    assert second.administered_mass(3) == first.administered_mass(3) == pytest.approx(1.2)
    assert second.administered_mass() == pytest.approx(1.6)
    assert second.administration == first.administration
    assert first.interval_values == (1.6,0.,3.2,3.2,0.,0.)


@pytest.mark.parametrize("fault",["refund","law","limit"])
def test_switch_rejects_modified_mass_law_or_actual_library_support_before_effect(config,fault):
    library = native_actuation_library(config,group())
    prefix = executed_prefix(library[0],1)
    if fault == "refund":prefix = replace(prefix,administered_mass_mg_per_kg=0.)
    elif fault == "law":prefix = replace(prefix,administration=None)
    else:
        library = tuple(replace(item,administration=replace(item.administration,maximum_supported_total_mg_per_kg=5.)) for item in library)
    with pytest.raises(ValueError,match="preserve"):
        compose_actuation(NativeActuationRequest(prefix,library[4]),library)
    assert prefix.interval_values == (1.6,)


def test_atomic_mass_budget_rejects_hybrid_without_refund_or_redose(config):
    library = native_actuation_library(config,group())
    prefix = executed_prefix(library[1],2)
    before = actuation_payload(prefix)
    with pytest.raises(ValueError,match="total public dose budget"):
        compose_actuation(NativeActuationRequest(prefix,library[5]),library)
    assert actuation_payload(prefix) == before and prefix.administered_mass_mg_per_kg == 1.2


def test_forward_burden_uses_mapped_total_and_default_remains_unbound(config):
    library = native_actuation_library(config,group())
    result = composed_action_burden(library[4],config)
    assert result["unit"] == "mg/kg" and result["actual_amount"] == 1.6
    assert result["normalized_burden"] == pytest.approx(2/3)
    assert result["fixed_support_maximum"] == 2.4 and not result["clinical_cost_calibration"]
    legacy = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    old = native_actuation_library(legacy,group())[4]
    assert old.interval_values == library[4].interval_values
    assert old.administration is None
    assert composed_action_burden(old,legacy)["status"] == "molecular_delivered_dose_mapping_unbound"
    with pytest.raises(ValueError,match="forward public actuation"):
        compiled_action_burden({},MOLECULAR,config)
    with pytest.raises(ValueError,match="explicit prospective"):
        validate_administration_library(library,legacy)


def test_legacy_serialized_identity_exact_and_opt_in_identity_distinct(config):
    legacy = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    old_library = native_actuation_library(legacy,group())
    prefix = executed_prefix(old_library[0],1)
    expected_prefix = asdict(prefix)
    expected_prefix.pop("administration");expected_prefix.pop("administered_mass_mg_per_kg")
    expected_action = asdict(old_library[4]);expected_action.pop("administration")
    intervals = prefix.interval_values+old_library[4].interval_values[1:]
    expected_uid = old_library[4].source_uid+":executed:"+digest({"prefix":expected_prefix,
        "actuation":expected_action,"interval_values":intervals,"dose_events":[]})
    old_joined = compose_actuation(NativeActuationRequest(prefix,old_library[4]),old_library)
    assert old_joined.regimen_uid == expected_uid
    assert actuation_payload(prefix) == expected_prefix and actuation_payload(old_library[4]) == expected_action
    mapped = native_actuation_library(config,group())
    new_joined = compose_actuation(NativeActuationRequest(executed_prefix(mapped[0],1),mapped[4]),mapped)
    assert new_joined.regimen_uid != old_joined.regimen_uid
    assert actuation_from_payload(json.loads(json.dumps(actuation_payload(new_joined)))) == new_joined


@pytest.mark.parametrize("field,value",[("reference_time_hours",2.),("rate_unit","mg/kg"),("route","oral"),
    ("schedule","periodic"),("enabled",False),("numerical_source_law","replacement")])
def test_missing_or_changed_prospective_law_rejects(config,field,value):
    changed = copy.deepcopy(config);changed["native_molecular_administration"][field] = value
    with pytest.raises(ValueError,match="law"):
        molecular_administration_policy(changed)


@pytest.mark.parametrize("fault",["route","schedule"])
def test_action_route_and_schedule_must_match_explicit_controlled_law(config,fault):
    changed = copy.deepcopy(config)
    action = changed["native_observable_bridge"]["physical_families"][MOLECULAR]["action"]
    if fault == "route":action["extension"]["action_semantics"]["route"] = "oral"
    else:action["extension"]["action_semantics"]["schedule"]["interval"] = 2.
    with pytest.raises(ValueError,match="route or schedule"):
        native_actuation_library(changed,group())


@pytest.mark.parametrize("fault",[None,"drive","concentration"])
def test_explicit_forward_rate_mapping_preserves_original_compiled_values(config,fault):
    public = native_actuation_library(config,group())[4]
    # Tiny compiler metadata fixture. No source trajectory, private world or model is run.
    p = {"initial_free_agent_exposure":0.,"dose_to_exposure_gain":18.,"exposure_clearance_rate":.3,"binding_affinity":24.}
    exposure = [0.]
    for u in public.interval_values:
        exposure.append(max(0.,exposure[-1]+18.*u-.3*exposure[-1]))
    support = np.zeros((7,36),dtype=bool);support[:,0] = True
    concentration = np.where(support,np.asarray(exposure)[:,None]*1e-9,0.)
    settings = {"route":"intrathecal","schedule":{"kind":"interval","interval":4.,"unit":"hour"},"dose_unit":"mg/kg"}
    record = {"action":copy.deepcopy(settings),"physical_program":{"support":support,
        "free_concentration_mol_per_l":concentration,"inputs":{"run_snapshot":{"parameters":p,"settings":settings}}}}
    if fault == "drive":public = replace(public,interval_values=(0.,0.,2.,3.2,0.,0.))
    elif fault == "concentration":record["physical_program"]["free_concentration_mol_per_l"][3,0] += 1e-9
    if fault is None:
        validate_mapped_molecular_record(public,record,config)
        # κr = (G*Tref)*(u/Tref) uses the same original exposure forcing.
        assert [p["dose_to_exposure_gain"]*4.*rate for rate in public.administration_rates_mg_per_kg_hour] == [18.*u for u in public.interval_values]
    else:
        with pytest.raises(ValueError,match="Mapped"):
            validate_mapped_molecular_record(public,record,config)


def test_physical_prefix_forwards_actual_mass_without_new_ledger(config):
    from Libs.utils.evaluation.native_physical_plant import NativePhysicalPlant
    public = native_actuation_library(config,group())[4]
    plant = object.__new__(NativePhysicalPlant)
    plant._failed = None;plant._public_program = public;plant._query = 3
    plant._executed_intervals = public.interval_values[:3];plant._executed_events = ()
    assert plant.prefix == executed_prefix(public,3)
    assert plant.prefix.administered_mass_mg_per_kg == .8
    plant._executed_intervals = (0.,0.,0.)
    with pytest.raises(ValueError,match="actually consumed"):
        plant.prefix


@pytest.mark.parametrize("fault",[None,"budget","missing_prefix"])
def test_real_future_control_compiler_uses_public_mass_budget_and_keeps_numeric_input(config,monkeypatch,fault):
    from Libs.utils.model import native_future_controls as controls
    library = native_actuation_library(config,group())
    initial = 1 if fault == "budget" else 0
    query = 2 if fault == "budget" else 1
    candidate = 5 if fault == "budget" else 4
    uids = tuple(item.regimen_uid for item in library)
    clock = torch.tensor(library[0].source_times,dtype=torch.float64)
    stimulus = torch.tensor([[0.,*item.interval_values] for item in library],dtype=torch.float64)
    stimulus = stimulus[None].expand(32,-1,-1).clone().reshape(32*len(library),-1)
    initial_field = torch.zeros(1,32,6,6,26)
    particles = (tuple("atom"+str(j) for j in range(32)),)
    support = SimpleNamespace(source_uid=library[0].source_uid,source_snapshot_ref=library[0].source_snapshot_ref,
        support_uid="support",role="validation",initial_field=initial_field,
        physical_parameter_hypotheses={"declarations":[],"values":torch.empty(1,32,0)})
    grid = {"family_uid":MOLECULAR,"query_time":{"unit":"hour"}}
    binding = {"source_uid":support.source_uid,"source_snapshot_ref":support.source_snapshot_ref,
        "support_uid":"support","role":"validation","origin_posterior_uid":"posterior",
        "sample_uids":("sample",),"particle_uids":particles,"regimen_uids":uids}
    records = [[[{"program_uid":"program:"+str(j)+":"+str(i)} for i in range(len(library))] for j in range(32)]]
    inputs = SimpleNamespace(grid=grid,group_shape=(1,32,len(library)),posterior_uid="posterior",
        sample_uids=("sample",),particle_uids=particles,regimen_uids=uids,
        posterior=SimpleNamespace(support=support),programs={"records":records})
    guide = SimpleNamespace(b=1,j=32,a=len(library),family=MOLECULAR,vm_only=False,
        grid=grid,clock=clock,times=clock,bind_source_intervals=False,stimulus=stimulus.clone(),parameters={})
    state = SimpleNamespace(_guide=guide,query_index=query,_times=tuple(clock.tolist()),
        _registered_regimen_uids=uids,_filter_binding=binding,_planning_origin=None,
        _filter_executed_regimen_uid=uids[initial],_execution_model_uid=uids[initial],
        _registered_stimulus=stimulus.clone(),_registered_program_uids=tuple(row["program_uid"] for particle in records[0] for row in particle),
        _initial=initial_field[:,:,None].expand(1,32,len(library),6,6,26).double().reshape(32*len(library),6,6,26),
        _public_control=None if fault == "missing_prefix" else library[initial],
        _execution_library=library,filter_revision=0,filter_observation_uids=())
    state._control_identity = controls.model_control_identity(state)
    monkeypatch.setattr(controls,"joint_particle_contexts",lambda *args:None)
    monkeypatch.setattr(controls,"validate_candidate_parameters",lambda *args:None)
    monkeypatch.setattr(controls,"validate_program_parameter_subset",lambda *args:None)
    monkeypatch.setattr(controls,"_semantic_signature",lambda record:{"fixture":"same"})
    before = guide.stimulus.clone()
    if fault:
        with pytest.raises(ValueError,match="dose budget|public prefix"):
            controls.compile_future_controls(state,inputs,config,executed_regimen_uid=uids[initial],
                candidate_regimen_uids=[uids[candidate]],require_delivered_dose=True)
    else:
        result = controls.compile_future_controls(state,inputs,config,executed_regimen_uid=uids[initial],
            candidate_regimen_uids=[uids[candidate]],require_delivered_dose=True)
        assert len(result["metadata"]["intervention_budgets"]) == 32
        for row in result["metadata"]["intervention_budgets"]:
            assert row["delivered_dose_bound"] and row["unit"] == "mg/kg"
            assert row["prefix"] == .4 and row["future"] == 1.6 and row["total"] == 2.
            assert row["library_maximum"] == 2.4
        assert result["stimulus"].shape == (32,7)
        assert torch.equal(result["stimulus"][0],torch.tensor([0.,1.6,0.,3.2,3.2,0.,0.],dtype=torch.float64))
    assert torch.equal(guide.stimulus,before) and state.query_index == query


def test_mapped_residual_component_guard_removes_only_missing_mass_block(config):
    from Libs.utils.evaluation.native_residual_control_evaluation import run_native_residual_control_episode
    library = native_actuation_library(config,group())
    state = SimpleNamespace(_public_control=library[0],query_index=0,_guide=SimpleNamespace(family=MOLECULAR),
        _execution_library=library,_times=[0.],_work_ledger=SimpleNamespace(spent=0))
    episode = SimpleNamespace(continuation=SimpleNamespace(rhs=state),observations_seen=0,optimizer_updates=0)
    report = run_native_residual_control_episode(episode,None,None,None,config)
    assert report["status"] == "component_complete"
    assert not report["full_residual_mpc_eligible"] and not report["formal_qualification"]
    assert report["scientific_twin"] == "NOT_RUN"
