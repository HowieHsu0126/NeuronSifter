"""Prospective source integration fixtures; no NEURON trajectory is executed."""
import copy
import json
from collections import Counter

import numpy as np
import pytest
import torch

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data import native_diffusion_data as data
from Libs.utils.data.joint_physical_parameters import (
    parameter_schema_from_config, parameter_conditioning_width, bind_candidate_parameters,
    candidate_conditioning, parameter_declarations,
)
from Libs.utils.data.native_observable_profile import (
    observable_profile, family_grid, quantity_support, observable_readout,
)
from Libs.utils.model import native_scientific_source as source
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, action_levels
from Libs.models.modules.world_state import JointParticleSupport


def config():
    return load_config("Libs/configs/exp/manuscript_native_protocol.yaml")


def romani_group(cfg):
    return next((i,g) for i,g in enumerate(source.group_definitions(cfg)) if g.get("source_variant") == "romani")


def initialize_fixture(candidates):
    return [{"parameters":dict(row), "time_ms":0., "Vm_mV":row["initial_membrane_voltage"]}
            for row in candidates]


def test_source_assignment_support_and_parameter_blocks():
    cfg=config();groups=source.group_definitions(cfg)
    train=[g for g in groups if g["role"] == "train" and g["family_uid"] == ELECTRICAL]
    assert Counter(g["source_stratum"] for g in train) == {
        "original_primary":500, "shared_primary":250, "romani":250}
    assert all(Counter(g["condition"] for g in train if g["source_stratum"] == name) ==
               {"control":n//2,"ad_positive":n//2} for name,n in
               (("original_primary",500),("shared_primary",250),("romani",250)))
    counts=source.planned_operation_counts(cfg)
    assert (counts["groups"],counts["cases"],counts["programs"]) == (22144,63232,2023424)
    assert counts["parameter_binding"] == 8000 and counts["source_prepare"] == 1
    assert counts["physical_refinement"] == 750 and counts["operations"] == 3350575
    assert all(g["family_uid"] == ELECTRICAL and g["role"] == "train"
               for g in groups if g.get("source_variant") == "romani")
    schema=parameter_schema_from_config(cfg)
    assert parameter_conditioning_width(schema) == 221
    uid=cfg["native_romani_source"]["source_uid"]
    assert len(parameter_declarations(schema,uid,ELECTRICAL)[1]) == 7
    with pytest.raises(ValueError,match="not defined"):
        parameter_declarations(schema,uid,SYNAPTIC)
    grid=family_grid(observable_profile(cfg),ELECTRICAL)
    actual=quantity_support(grid,uid)
    assert actual[...,:15].all() and actual[...,15].sum() == len(grid["time_hours"])
    assert not actual[...,16:].any()
    mean=torch.ones((1,*actual.shape));mask=torch.from_numpy(actual).unsqueeze(0)
    readout=observable_readout(mean,mean,grid,mask)
    assert readout["feature_mask"].tolist() == [[True,False,False]]
    assert (readout["predictive_mean"][...,1:] == 0).all()


def test_complete_prior_precedes_selector_and_future_vm_cannot_change_inputs(monkeypatch):
    cfg=config();gi,group=romani_group(cfg);scoped=source.scoped_group_config(cfg,group)
    events=[]
    original=source.produce_controlled_observations
    def initializer(candidates):
        assert len(candidates) == 32
        events.append("all_prior_parameters_applied")
        return initialize_fixture(candidates)
    def selector(*args,**kwargs):
        assert events == ["all_prior_parameters_applied"]
        events.append("hidden_selection")
        return original(*args,**kwargs)
    monkeypatch.setattr(source,"produce_controlled_observations",selector)
    generated=source.draw_group(scoped,group,gi,physical_initializer=initializer)
    assert events == ["all_prior_parameters_applied","hidden_selection"]
    schema=parameter_schema_from_config(cfg)
    records=[[source.compiled_program(scoped,group,generated["support"],atom,
              generated["parameters"][atom],source.action_definition(scoped,ELECTRICAL,ai,group),"fixture")
              for ai in range(3)] for atom in range(32)]
    raw=source.joint_source(scoped,group,generated,records)
    bound=bind_candidate_parameters(raw,group,{"parameters":generated["parameters"]},schema,
                                    record_ref="software_fixture",producer_commit="fixture")
    support=JointParticleSupport.from_payload(bound["support"])
    condition=candidate_conditioning(support,schema)
    assert condition.shape == (1,32,442) and condition[...,221:].sum(-1).eq(7).all()
    baseline=copy.deepcopy(bound)
    teacher=np.tile(np.array([[generated["parameters"][generated["hidden_index"]]["initial_membrane_voltage"]]]),(7,1))
    law=source.CoupledObservableLaw(scoped,group,generated["support"],generated["hidden_index"],
                                    {"values":teacher,"feature_names":["membrane_voltage"]})
    initial=law.initial[...,:15].reshape(36,15)[:,[law.names.index(name) for name in source.DYNAMIC]]
    before=law.reaction(initial,0)[0].copy()
    teacher[1:]=100.
    assert np.array_equal(law.reaction(initial,0)[0],before)
    assert not np.array_equal(law.reaction(initial,1)[0],before)
    rebuilt=source.joint_source(scoped,group,generated,records)
    rebound=bind_candidate_parameters(rebuilt,group,{"parameters":generated["parameters"]},schema,
                                     record_ref="software_fixture",producer_commit="fixture")
    assert rebound == baseline
    assert torch.equal(candidate_conditioning(JointParticleSupport.from_payload(rebound["support"]),schema),condition)
    for record in records[0]:
        data.validate_compiled_program(record)
        assert record["action"]["state_scope"]["physical_site_uid"] == "modeldb_romani_n123_soma_0_midpoint"
        assert record["source_action_timing"]["effective_duration"] == 6.


@pytest.mark.parametrize("variant",["primary","held"])
@pytest.mark.parametrize("family",[ELECTRICAL,SYNAPTIC])
def test_all_discrete_actions_compile_actual_first_last_intervals(variant,family):
    cfg=config()
    group=next(g for g in source.group_definitions(cfg) if g["family_uid"]==family and g["source_variant"]==variant)
    group=copy.deepcopy(group);group.pop("action_indices",None);group.pop("action_stratum",None)
    scoped=source.scoped_group_config(cfg,group)
    generated=source.draw_group(scoped,group,0)
    grid=family_grid(observable_profile(cfg),family)
    for ai in range(6):
        action=source.action_definition(scoped,family,ai,group)
        record=source.compiled_program(scoped,group,generated["support"],0,generated["parameters"][0],action,"fixture")
        data.validate_compiled_program(record)
        wave=record["action"]["stimulation_or_environment"]["magnitude_or_program"]
        expected=np.r_[np.asarray(action_levels(grid,action))[1:],0.]
        assert np.array_equal(wave["values"],expected)
        timing=record["source_action_timing"]
        active=np.flatnonzero(expected[:-1]);times=np.asarray(grid["query_time"]["values"])
        assert timing["actual_onset"] == times[active[0]]
        assert timing["actual_stop"] == times[active[-1]+1]
        assert timing["cumulative_stimulation"] == np.dot(expected[:-1],np.diff(times))
        if ai >= 4:
            assert timing["source_requested_duration"] < timing["effective_duration"]
            assert timing["requested_duration_exact"] is False


def test_common_source_action_intervals_match_and_romani_never_falls_back(monkeypatch):
    cfg=config();gi,group=romani_group(cfg)
    rom=source.scoped_group_config(cfg,group)
    primary_group=next(g for g in source.group_definitions(cfg) if g.get("source_stratum")=="shared_primary")
    primary=source.scoped_group_config(cfg,primary_group)
    generated=source.draw_group(rom,group,gi,physical_initializer=initialize_fixture)
    other=source.draw_group(primary,primary_group,0)
    grid=family_grid(observable_profile(cfg),ELECTRICAL)
    class Runtime:
        def trajectory(self,parameters,action,query,dt):
            return {"values":np.zeros((7,1)), "feature_names":["membrane_voltage"],"actual_test_dt":dt}
    def forbidden(*args,**kwargs):
        raise AssertionError("Romani incorrectly dispatched to the stylized source")
    for ai in range(3):
        action=source.action_definition(rom,ELECTRICAL,ai,group)
        equivalent=source.action_definition(primary,ELECTRICAL,ai,primary_group)
        a=source.compiled_program(rom,group,generated["support"],0,generated["parameters"][0],action,"fixture")
        b=source.compiled_program(primary,primary_group,other["support"],0,other["parameters"][0],equivalent,"fixture")
        left=a["action"]["stimulation_or_environment"]["magnitude_or_program"]
        right=b["action"]["stimulation_or_environment"]["magnitude_or_program"]
        assert left["values"] == right["values"]
        assert a["source_action_timing"]["actual_onset"] == [2.,2.,4.][ai]
        assert b["source_action_timing"]["actual_stop"] == [8.,8.,10.][ai]
        assert a["source_action_timing"]["cumulative_stimulation"] == b["source_action_timing"]["cumulative_stimulation"]
    monkeypatch.setattr(source,"original_physical_trajectory",forbidden)
    for label,dt in zip(("coarse","fine","reference"),(.1,.05,.025)):
        result=source.physical_trajectory(rom,ELECTRICAL,generated["parameters"][0],action,
                                         refinement=label,source_runtime=Runtime())
        assert result["actual_test_dt"] == dt and result["values"].shape == (7,1)
    with pytest.raises(ValueError,match="actual source runtime"):
        source.physical_trajectory(rom,ELECTRICAL,generated["parameters"][0],action)


def test_shared_current_rejects_off_grid_stop_and_reports_actual_equivalence():
    cfg=config()
    groups=source.group_definitions(cfg,role_filter="train")
    group=next(g for g in groups if g.get("source_stratum")=="shared_primary")
    # Reject a fractional stop that would otherwise differ between IClamp and the discrete source.
    bad=copy.deepcopy(cfg)
    bad["native_romani_source"]["common_current"]["action_variants"][0]["duration_ms"]=5.
    with pytest.raises(ValueError,match="onset and stop"):
        source.action_definition(bad,ELECTRICAL,0,group)
    bad["native_romani_source"]["common_current"]["action_variants"][0]["duration_ms"]=0.
    with pytest.raises(ValueError,match="positive duration"):
        source.action_definition(bad,ELECTRICAL,0,group)
    report=source.effective_stimulation_actions(cfg,groups)
    assert all(row["declared_action_count"]==row["effective_program_count"] and not row["collapsed_action_indices"]
               for row in report)
    common=[row for row in report if row["action_stratum"]=="shared_current"]
    assert len(common)==2
    assert [row["values"] for row in common[0]["effective_programs"]]==[
        row["values"] for row in common[1]["effective_programs"]]
    assert [[a["actual_onset"],a["actual_stop"]] for row in common[0]["effective_programs"]
            for a in row["source_requests"]]==[[2.,8.],[2.,8.],[4.,10.]]
    changed=copy.deepcopy(cfg)
    changed["native_observable_bridge"]["action_variants"][1]=copy.deepcopy(
        changed["native_observable_bridge"]["action_variants"][0])
    duplicate=source.effective_stimulation_actions(changed,groups)
    assert all([0,1] in row["collapsed_action_indices"] for row in duplicate if row["action_stratum"]!="shared_current")


def test_prospective_code_identity_binds_romani_dependency_set_and_legacy_hash():
    from Libs.scripts.builders.build_controlled_native_source import observable_code_identity, verify_observable_source
    cfg=config();settings=cfg["native_observable_bridge"]
    required={"Libs/configs/exp/native_romani_vm_source.yaml","Libs/utils/model/native_romani_source.py",
              "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py"}
    assert required<=set(settings["freeze_paths"]) and required<=set(settings["producer_freeze_paths"])
    bound={path:str(index) for index,path in enumerate(settings["freeze_paths"])}
    entry,identity=observable_code_identity(settings,bound)
    changed=dict(bound);changed["Libs/utils/model/native_romani_source.py"]="changed_runtime"
    assert observable_code_identity(settings,changed)[1]!=identity
    with pytest.raises(ValueError,match="code identity"):
        verify_observable_source({"source_identity":{"bound_files":changed,"source_file":entry,
                                  "source_code_sha256":identity}},cfg)
    legacy={"scientific_profile":False}
    entry,identity=observable_code_identity(legacy,bound)
    assert entry=="Libs/utils/model/native_observable_source.py" and identity==bound[entry]


def test_romani_preparation_and_parameter_binding_persist_correct_group(tmp_path):
    from Libs.scripts.builders.build_controlled_native_source import ObservableSourceLedger
    cfg=config()
    settings=cfg["native_observable_bridge"]
    settings["group_partitions"]=[dict(settings["group_partitions"][0],groups_per_family=8)]
    settings["budget"].update(source.planned_operation_counts(cfg))
    ledger=ObservableSourceLedger(tmp_path/"source_receipt.json",cfg,{},clock=lambda:0.)
    gi=next(i for i,g in enumerate(ledger.data["groups"]) if g["source_variant"]=="romani")
    assert gi!=ledger.data["cases"][gi]["group_index"]
    for key in ("source_prepare:"+str(gi),"parameter_binding:"+str(gi)+":0"):
        def operation():
            row=json.loads((tmp_path/("group_"+str(gi)+"_receipt.json")).read_text())
            assert row["group_index"]==gi and row["operations"][key]["status"]=="started"
            return {"software_fixture":True,"time_advanced":False}
        ledger.execute(key,operation)
        row=json.loads((tmp_path/("group_"+str(gi)+"_receipt.json")).read_text())
        assert row["operations"][key]["status"]=="complete"
        with pytest.raises(RuntimeError,match="already consumed"):
            ledger.execute(key,operation)
