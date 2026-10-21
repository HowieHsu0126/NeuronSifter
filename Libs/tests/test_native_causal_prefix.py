"""Source-free formula checks and actual native causal-prefix consumers."""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from Libs.utils.data.native_field_normalization import NativeFieldNormalization, source_domain
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_causal_prefix import (
    PREFIX_SPEC, score_prefix_pair, match_prepared_pair, contrast_row,
    build_prefix_payload, validate_prefix_payload, causal_prefix_policy, declared_contrast)


# Existing producer/checkpoint fixtures retain all native field axes and experts.
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from Libs.tests.test_expert_composition import ten_expert_case
from Libs.tests.test_native_field_normalization import joint_normalized_case
from Libs.tests.test_native_rollout_evaluation import development_case
from Libs.tests.test_joint_world_posterior import local_membrane_case, raw_joint_source
from Libs.tests.test_physical_occupancy import make_packet
from Libs.models.modules.world_state import JointParticleSupport
from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config
from Libs.exp.evaluation import evaluate_screening_system as evaluation
from Libs.exp.evaluation import evaluate_rollout_prediction as cli

def arithmetic_case(x, y):
    x,y=torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64)
    species=x.shape[-1]
    grid={"time_hours":[0.,1.,2.,3.],"region_id":["region"],"compartment_id":["compartment"],
          "species_id":[f"s{i}" for i in range(species)],"physical_time_unit":"hour",
          "species_units":{f"s{i}":"fixture_unit" for i in range(species)},
          "normalization":{"kind":"training_scaler","sha256":"fixture-fixed-transform"}}
    norm=NativeFieldNormalization({"kind":"physical"},species)
    norm.active=True;norm.identity="fixture-fixed-transform";norm.artifact={"domain":source_domain(grid)}
    norm.mean=torch.arange(species,dtype=torch.float64)+10
    norm.scale=torch.arange(species,dtype=torch.float64)+2
    values=torch.zeros(len(x),2,4,1,1,species,dtype=torch.float64)
    values[:,0,1,0,0]=x;values[:,1,1,0,0]=y
    mask=torch.ones(2,4,1,1,species,dtype=torch.bool)
    return norm.inverse(values),mask,grid,norm


def explicit_energy(x,y):
    count=len(x);m=len(x[0])
    def distance(a,b):return math.sqrt(sum((u-v)**2 for u,v in zip(a,b))/m)
    return sum(distance(x[i],y[j])+distance(y[i],x[j])-distance(x[i],x[j])-distance(y[i],y[j])
               for i in range(count) for j in range(count) if i!=j)/(count*(count-1))


@pytest.mark.parametrize("x,y",[
    ([[0.,0.]]*4,[[1.,1.]]*4),
    ([[-1.,0.],[1.,2.],[0.,-2.]], [[-1.,0.],[1.,2.],[0.,-2.]]),
    ([[-1.,-1.],[1.,1.]], [[1.,1.],[-1.,-1.]]),
    ([[-1.,-1.],[1.,1.]]*4,[[-1.,1.],[1.,-1.]]*4),
    ([[.2,1.],[-1.,2.],[3.,.4],[2.,-2.]], [[1.,-.2],[2.,1.],[-.4,2.],[-2.,3.]])
])
def test_independent_energy_and_paired_jackknife_formulas(x,y):
    samples,mask,grid,norm=arithmetic_case(x,y)
    result=score_prefix_pair(samples,mask,grid,2.,norm)
    assert result["status"]=="defined" and result["valid_coordinate_count"]==2 and result["prefix_time_count"]==1
    assert result["energy_u_statistic"]==pytest.approx(explicit_energy(x,y),abs=1e-12)
    if len(x)>=3:
        leave=[explicit_energy(x[:i]+x[i+1:],y[:i]+y[i+1:]) for i in range(len(x))]
        average=sum(leave)/len(leave)
        expected=math.sqrt((len(x)-1)/len(x)*sum((value-average)**2 for value in leave))
        assert result["mc_standard_error"]==pytest.approx(expected,abs=1e-12)
    else:
        assert result["mc_standard_error"] is None and result["mc_uncertainty_status"]=="unavailable_fewer_than_three_draws"


def test_signed_coupling_and_joint_dependence_are_not_causal_verdicts():
    # Same exact marginal laws, opposite coupling: paths disagree and the
    # off-diagonal estimate can be negative. Neither is a causal-failure test.
    samples,mask,grid,norm=arithmetic_case([[-1.],[1.]],[[1.],[-1.]])
    result=score_prefix_pair(samples,mask,grid,2.,norm)
    assert result["energy_u_statistic"]==pytest.approx(-4.)
    assert result["paired_path_rms"]==pytest.approx(2.)
    assert not any(key in result for key in ("causal_pass","causal_failure","p_value"))
    x=[[-1.,-1.],[1.,1.]]*4;y=[[-1.,1.],[1.,-1.]]*4
    assert sorted(row[0] for row in x)==sorted(row[0] for row in y)
    assert sorted(row[1] for row in x)==sorted(row[1] for row in y)
    samples,mask,grid,norm=arithmetic_case(x,y)
    assert score_prefix_pair(samples,mask,grid,2.,norm)["energy_u_statistic"]>0


def test_swap_paired_permutation_fixed_units_and_excluded_cells():
    samples,mask,grid,norm=arithmetic_case([[0.,1.],[2.,3.],[4.,2.]],[[3.,-1.],[1.,2.],[0.,3.]])
    expected=score_prefix_pair(samples,mask,grid,2.,norm)
    for changed in (samples.flip(1),samples[[2,0,1]]):
        actual=score_prefix_pair(changed,mask,grid,2.,norm)
        for key in ("energy_u_statistic","mc_standard_error","paired_path_rms"):
            assert actual[key]==pytest.approx(expected[key],abs=1e-12)
    changed=samples.clone();changed[:,:,[0,2,3]]+=10000
    assert score_prefix_pair(changed,mask,grid,2.,norm)==expected
    scaled=copy.deepcopy(norm);scaled.mean*=1000;scaled.scale*=1000
    actual=score_prefix_pair(samples*1000,mask,grid,2.,scaled)
    assert actual["energy_u_statistic"]==pytest.approx(expected["energy_u_statistic"],abs=1e-12)


@pytest.mark.parametrize("bad",["nan","inf","shape","mask_type","mask_shape","mask_mismatch","cutoff","normalization"])
def test_numerical_and_prefix_contract_rejection(bad):
    samples,mask,grid,norm=arithmetic_case([[0.,0.],[1.,2.]],[[1.,0.],[0.,2.]])
    cutoff=2.
    if bad=="nan":samples[0,0,1,0,0,0]=float("nan")
    elif bad=="inf":samples[0,0,1,0,0,0]=float("inf")
    elif bad=="shape":samples=samples[:,:,:-1]
    elif bad=="mask_type":mask=mask.float()
    elif bad=="mask_shape":mask=mask[:,:-1]
    elif bad=="mask_mismatch":mask[1,1,0,0,0]=False
    elif bad=="cutoff":cutoff=1.5
    else:norm.active=False
    with pytest.raises(ValueError):score_prefix_pair(samples,mask,grid,cutoff,norm)


@pytest.mark.parametrize("kind",["one_draw","no_draws","initial_only","empty_mask"])
def test_undefined_denominator_without_budget_increase(kind):
    samples,mask,grid,norm=arithmetic_case([[0.,0.],[1.,2.]],[[1.,0.],[0.,2.]])
    cutoff=2.
    if kind=="one_draw":samples=samples[:1]
    elif kind=="no_draws":samples=samples[:0]
    elif kind=="initial_only":cutoff=1.
    else:mask.zero_()
    result=score_prefix_pair(samples,mask,grid,cutoff,norm)
    assert result["status"]=="undefined" and result["energy_u_statistic"] is None and result["mc_standard_error"] is None
    assert result["realized_draw_count"]==len(samples)


def prepared_point_case():
    _,mask,grid,norm=arithmetic_case([[0.,0.],[1.,2.]],[[1.,0.],[0.,2.]])
    context={"initial_state":torch.ones(1,2,1,1,2),"posterior_context":torch.ones(1,2,1,1,2),
             "occupancy":torch.full((1,2,4,1,1,1),.2),"support":torch.ones(1,2,4,1,1,1),
             "physical_time":torch.tensor(grid["time_hours"]).reshape(1,1,4,1,1,1).expand(1,2,4,1,1,1).clone()}
    context["occupancy"][:,1,2:]=.8
    observation={"posterior_uid":"posterior"}
    records=[{"teacher":{"sample_uid":"sample","regimen_uid":uid,"posterior_uid":"posterior","program_uid":"program-"+uid},
              "observation":observation} for uid in ("a","b")]
    prepared=(torch.zeros(1,2),{"sample_ids":torch.tensor([native.stable_numeric_id("sample")]),
        "regimen_ids":torch.tensor([native.stable_numeric_id(uid) for uid in ("a","b")]),"diffusion_context":context})
    return prepared,records,mask,grid,SimpleNamespace(field_normalization=norm,operator_hidden_dim=2)


@pytest.mark.parametrize("bad",["initial_state","posterior_context","past_program","clock","future_missing","ids","mask"])
def test_prepared_conditions_reject_before_any_sampling(bad):
    prepared,records,mask,grid,field=prepared_point_case()
    context=prepared[1]["diffusion_context"]
    if bad in ("initial_state","posterior_context"):context[bad][:,1]+=1
    elif bad=="past_program":context["occupancy"][:,1,1]=.5
    elif bad=="clock":context["physical_time"][:,1,1]=.5
    elif bad=="future_missing":context["occupancy"].fill_(.2)
    elif bad=="ids":prepared[1]["regimen_ids"][1]+=1
    else:mask[1,1,0,0,0]=False
    with pytest.raises(ValueError):match_prepared_pair(prepared,records,["a","b"],grid,2.,mask,field)


def test_prepared_pair_keeps_original_rng_ids_and_exact_prefix():
    prepared,records,mask,grid,field=prepared_point_case()
    matched=match_prepared_pair(prepared,records,["a","b"],grid,2.,mask,field)
    assert matched["posterior_uid"]=="posterior" and matched["valid_coordinate_count"]==2
    assert prepared[1]["regimen_ids"].tolist()==[native.stable_numeric_id(uid) for uid in ("a","b")]


def test_canonical_policy_equal_episode_aggregation_and_unbound_schema():
    config=load_config("Libs/configs/config_stack.yaml")
    assert {key:value for key,value in config["paper_grade_full_suite"]["native_causal_prefix_evaluation"].items()
            if key!="contrasts"}==PREFIX_SPEC
    rows=[]
    for index,(sample,value) in enumerate((("a",1.),("a",3.),("b",9.),("b",None))):
        row=contrast_row(index,{"contrast_uid":str(index),"sample_uid":sample,"episode_uid":sample,
             "regimen_uids":["x","y"],"future_knot_hours":2.},2)
        row.update(source_uid="source",family_uid="family")
        if value is not None:
            row.update(status="defined",reason=None,energy_u_statistic=value,paired_path_rms=1.,
                       realized_draw_count=2,prefix_time_count=1,valid_coordinate_count=2,
                       mc_uncertainty_status="unavailable_fewer_than_three_draws")
        rows.append(row)
    payload=build_prefix_payload(rows,{}, {},config)
    assert payload["metrics"]["energy_u_statistic"]["mean"]==5.5
    assert payload["counts"]["declared_contrasts"]==4 and payload["counts"]["defined_contrasts"]==3
    assert payload["counts"]["complete_episodes"]==1
    assert payload["status"]==payload["diagnostic_status"]=="incomplete"
    assert payload["causal_decision_rule_status"]=="unbound"
    bad=copy.deepcopy(payload);bad["counts"]["declared_contrasts"]=3
    with pytest.raises(ValueError):validate_prefix_payload(bad,config)
    bad=copy.deepcopy(payload);bad["causal_decision_rule_status"]="pass"
    with pytest.raises(ValueError):validate_prefix_payload(bad,config)
    # A forged episode plus a consistently forged overall mean must not bypass
    # the contrast-to-episode denominator and reduction authority.
    for key in ("declared_contrast_count","defined_contrast_count","energy_u_statistic","paired_path_rms","status","episode_uid"):
        bad=copy.deepcopy(payload)
        bad["episodes"][0][key] = "forged" if isinstance(bad["episodes"][0][key],str) else 99
        for metric in bad["metrics"]:
            bad["metrics"][metric]["mean"]=sum(row[metric] for row in bad["episodes"])/2
        with pytest.raises(ValueError):validate_prefix_payload(bad,config)
    for key in ("declared_episodes","numerically_defined_episodes","complete_episodes","group_count","mc_uncertainty_available_contrasts"):
        bad=copy.deepcopy(payload);bad["counts"][key]+=1
        with pytest.raises(ValueError):validate_prefix_payload(bad,config)
    bad=copy.deepcopy(payload);bad["metrics"].pop("paired_path_rms")
    with pytest.raises(ValueError):validate_prefix_payload(bad,config)
    json.dumps(payload,allow_nan=False)


ROOT=Path(__file__).resolve().parents[2]


def matched_native_case(case,directory,point):
    case=list(case)
    case[1]=copy.deepcopy(case[1])
    backbone=case[1]["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    binding=backbone["quantized_mddm_rollout"]["registry_binding"]
    case[1]["neuronsifter_scale_registry"][binding]["partial_mask_schedule"]["inference_sample_count"]=3
    selected=case[1].get("neuronsifter_scale_selection",{}).get("inference_profile")
    if selected:
        case[1]["neuronsifter_scale_registry"]["inference_profiles"][selected]["sampler_budget"]["sample_count"]=3
    # Bind unequal draw/action axes through the maintained resolver before checkpoint construction.
    assert quantized_mddm_rollout_from_config(case[1],backbone)["partial_mask_schedule"]["inference_sample_count"]==3
    config,registry,arrays,store,fmt,checkpoint=development_case(case,directory,point=point)
    source=copy.deepcopy(native.load_registry(case[1])["joint_sources"][0] if point else registry["joint_sources"][0])
    source["support"]["role"]="development"
    for obs in source["observations"]:obs["role"]="development"
    support=JointParticleSupport.from_payload(source["support"])
    keys=("channel_uid","observation_uid","observed","encoder_indices","library_offset","detection_limit","normalization_ref","modality","assay_uid")
    channels=[{key:row[key] for key in keys if row.get(key) is not None} for row in source["observations"]]
    if not point:
        support,channels=local_membrane_case(support,channels,registry["grid"])
    production=produce_controlled_observations(support,channels,latent_seed=41,observation_seed=42)
    raw=raw_joint_source(support,production)
    exports=directory/"matched-source-exports"
    packet=make_packet(exports,config,raw)
    for entry in packet["exports"]:
        bi=list(support.sample_uids).index(entry["sample_uid"])
        ji=list(support.particle_uids[bi]).index(entry["particle_uid"])
        path=exports/entry["csv_path"]
        frame=pd.read_csv(path)
        frame.loc[frame["Time"]==0.,"simulationValues"]=0.
        frame.loc[frame["Time"]==60.,"simulationValues"]=10.*2**ji
        frame.to_csv(path,index=False)
        snapshot_path=exports/entry["run_snapshot_path"]
        snapshot=json.loads(snapshot_path.read_text())
        snapshot["export_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot_path.write_text(json.dumps(snapshot))
    source=build_native_physical_source(config,packet,exports)
    if point:
        registry["programs"]=copy.deepcopy(source["particle_programs"]["records"][0][0])
        for index,program in enumerate(registry["programs"]):
            program.update(store_path=str(store),store_format=fmt,program_index=index)
        registry["observations"][0]["posterior_uid"]=registry["programs"][0]["posterior_uid"]
        for key in ("occupancy","support","physical_time"):
            arrays[key]=np.asarray([row["physical_program"][key] for row in registry["programs"]],dtype=np.float32)
    else:
        registry["joint_sources"]=[source]
    for index,teacher in enumerate(registry["teachers"]):
        program=registry["programs"][index] if point else source["particle_programs"]["records"][index//2][0][index%2]
        teacher.update({key:program[key] for key in ("program_uid","posterior_uid","action_uid","target_program_uid")})
    write_native_microenvironment_store(store,config,registry,arrays,fmt)
    contrasts=[]
    for sample in config["paper_grade_full_suite"]["native_rollout_evaluation"]["sample_uids"]:
        contrasts.append({"contrast_uid":sample+"-future","sample_uid":sample,"episode_uid":sample,
                          "regimen_uids":["regimen-a","regimen-b"],"future_knot_hours":2.})
    contrasts.append(dict(contrasts[0],contrast_uid="same-episode-reversed",regimen_uids=["regimen-b","regimen-a"]))
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=contrasts
    return {"config":config,"checkpoint":checkpoint,"registry":registry,"directory":directory}


@pytest.fixture(scope="module")
def joint_prefix_case(ten_expert_case,tmp_path_factory):
    return matched_native_case(ten_expert_case,tmp_path_factory.mktemp("joint_prefix"),False)


@pytest.fixture(scope="module")
def point_prefix_case(ten_expert_case,tmp_path_factory):
    return matched_native_case(ten_expert_case,tmp_path_factory.mktemp("point_prefix"),True)


def forbid_teacher_reads(monkeypatch,registry):
    names=native.consumed_array_names(registry)
    forbidden={names[key] for key in native.TEACHER_ARRAY_NAMES if key in names}
    accesses={"array_names":[],"conditioning_calls":0}
    for name in ("load_zarr_rows","load_chunked_rows"):
        original=getattr(native.dynamics_store,name)
        def watched(path,array_name,*args,_original=original,**kwargs):
            assert array_name not in forbidden,"teacher outcome array was read"
            accesses["array_names"].append(array_name)
            return _original(path,array_name,*args,**kwargs)
        monkeypatch.setattr(native.dynamics_store,name,watched)
    original=native.load_records
    def read(*args,**kwargs):
        assert kwargs.get("include_teacher",True) is False
        accesses["conditioning_calls"]+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(native,"load_records",read)
    return accesses


@pytest.mark.parametrize("point",[False,True])
def test_actual_native_cli_full_field_reuse_and_reordering(request,tmp_path,monkeypatch,point):
    case=request.getfixturevalue("point_prefix_case" if point else "joint_prefix_case")
    config,checkpoint=copy.deepcopy(case["config"]),case["checkpoint"]
    accesses=forbid_teacher_reads(monkeypatch,case["registry"])
    samples,operator_calls,sampler_calls,scopes=[],[],[],[]
    loader=evaluation.load_native_conditioning_checkpoint
    handles=[]
    def loaded(*args,**kwargs):
        result=loader(*args,**kwargs);model=result[0];field=model.field_rollout_operator
        operator=field.neuronsifter_world_action_operator;sampler=field.quantized_mddm_rollout_operator
        assert len(operator.blocks[0].operator_splitting_expert_bank.expert_names)==10
        handles.append(operator.register_forward_hook(lambda module,args,out:operator_calls.append(type(module).__name__)))
        def before(module,args,kwargs):
            assert kwargs["teacher_trajectory"] is None and kwargs["teacher_confidence"] is None
            assert kwargs["initial_state"].shape[-3:]==(6,6,15)
            sampler_calls.append((kwargs["sample_ids"].tolist(),kwargs["regimen_ids"].tolist()))
        handles.append(sampler.register_forward_pre_hook(before,with_kwargs=True))
        scopes.append({"parameters":sum(p.numel() for p in model.parameters()),"grid":[3,6,6,15],
                       "draws":sampler.inference_sample_count,"steps":sampler.inference_denoising_step_count})
        return result
    monkeypatch.setattr(evaluation,"load_native_conditioning_checkpoint",loaded)
    predictor=evaluation.predict_native_episode
    def captured(*args,**kwargs):
        output=predictor(*args,**kwargs)
        assert float(output["router_diagnostics"]["mddm_teacher_loss_evaluated"])==0
        samples.append((list(kwargs["prepared"][1].get("regimen_uids",["regimen-a","regimen-b"])),
                        output["field_trajectory_samples"].clone(),
                        output.get("particle_draw_indices"),output.get("local_membrane_status")))
        return output
    monkeypatch.setattr(evaluation,"predict_native_episode",captured)
    cfg=tmp_path/"config.yaml";cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    output=tmp_path/"written"
    monkeypatch.setattr(sys,"argv",["evaluate_rollout_prediction.py","--execution-mode","native",
        "--native-operation","causal_prefix","--config",str(cfg),"--checkpoint",str(checkpoint),
        "--output-dir",str(output),"--freeze-manifest",str(tmp_path/"absent-freeze.json")])
    with pytest.raises(SystemExit) as stopped:cli.main()
    assert stopped.value.code==2
    payload=json.loads((output/"metrics.json").read_text())
    validate_prefix_payload(payload,config)
    assert payload["status"]=="incomplete" and payload["diagnostic_status"]=="complete",payload
    assert payload["causal_decision_rule_status"]=="unbound"
    episodes=1 if point else 2
    assert len(samples)==episodes
    assert len(sampler_calls)==episodes*(1 if point else 2)
    assert operator_calls and accesses["conditioning_calls"]==episodes
    if point:assert accesses["array_names"]
    assert all(len(ids)==2 for _,ids in sampler_calls)
    assert payload["counts"]["defined_contrasts"]==episodes+1
    assert all(row["cost"]["native_prediction_calls"]==1 for row in payload["episodes"])
    assert all(row["realized_draw_count"]==row["configured_draw_count"]==3 for row in payload["contrasts"])
    assert all(row["mc_standard_error"] is not None for row in payload["contrasts"])
    if not point:
        assert all(item[3]["electrical_evolution_status"]=="unbound" for item in samples)
        assert all(row["prepared_binding"]["local_membrane"] is not None for row in payload["contrasts"])
    assert json.loads((output/"artifact_manifest.json").read_text())["postrun_status"]=="incomplete"
    assert json.loads((output/"status.json").read_text())["status"]=="incomplete"
    assert payload["provenance"]["teacher_outcome_arrays_read"] is False
    first={row["contrast_uid"]:row["energy_u_statistic"] for row in payload["contrasts"]}
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"].reverse()
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["regimen_uids"].reverse()
    samples.clear();sampler_calls.clear()
    reordered,_=evaluation.evaluate_native_causal_prefix(config,checkpoint)
    assert {row["contrast_uid"]:row["energy_u_statistic"] for row in reordered["contrasts"]}==pytest.approx(first,abs=1e-12)
    assert len(samples)==episodes and len(sampler_calls)==episodes*(1 if point else 2)
    print("ACTUAL_NATIVE_PREFIX_SCOPE",json.dumps({"point":point,"scope":scopes[0],
        "energy_u_by_contrast":first,"scientific_status":payload["status"],"decision":"unbound"}))
    for handle in handles:handle.remove()


@pytest.mark.parametrize("bad",["initial","context","past","clock","future_missing","rng_ids"])
def test_real_point_prepared_mismatch_rejects_before_sampler(point_prefix_case,monkeypatch,bad):
    config=copy.deepcopy(point_prefix_case["config"])
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"][:1]
    original=evaluation.prepare_native_episode
    def changed(*args,**kwargs):
        prepared=original(*args,**kwargs);context=prepared[1]["diffusion_context"]
        if bad=="initial":context["initial_state"][:,1]+=1
        elif bad=="context":context["posterior_context"][:,1]+=1
        elif bad=="past":context["occupancy"][:,1,1]=.1
        elif bad=="clock":context["physical_time"][:,1,1]=.5
        elif bad=="future_missing":context["occupancy"][:,1]=context["occupancy"][:,0]
        else:prepared[1]["regimen_ids"][1]+=1
        return prepared
    monkeypatch.setattr(evaluation,"prepare_native_episode",changed)
    monkeypatch.setattr(evaluation,"predict_native_episode",lambda *a,**k:pytest.fail("invalid contrast reached native sampling"))
    payload,_=evaluation.evaluate_native_causal_prefix(config,point_prefix_case["checkpoint"])
    assert payload["counts"]["failed_contrasts"]==1
    assert payload["counts"]["defined_contrasts"]==0
    assert payload["episodes"][0]["cost"]["native_prediction_calls"]==0


@pytest.mark.parametrize("bad",["cached_weights","particle_context","particle_identity","membrane_clock"])
def test_real_joint_particle_and_snapshot_mismatch_precedes_sampler(joint_prefix_case,monkeypatch,bad):
    config=copy.deepcopy(joint_prefix_case["config"])
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"][:1]
    original=evaluation.prepare_native_episode
    def changed(*args,**kwargs):
        prepared=original(*args,**kwargs);posterior=prepared[1]["joint_posterior"]
        if bad=="cached_weights":posterior.weights[0,0]+=.1
        elif bad=="particle_context":posterior.particle_context.add_(1)
        elif bad=="particle_identity":
            object.__setattr__(posterior.support,"particle_uids",(("changed",*posterior.support.particle_uids[0][1:]),))
        else:
            support=posterior.support.to_payload()
            def alter(value):
                if isinstance(value,dict):
                    if "current_time_ms" in value:
                        value["current_time_ms"][0]+=.05
                        return True
                    return any(alter(item) for item in value.values())
                return False
            assert alter(support["local_membrane"])
            object.__setattr__(posterior,"support",JointParticleSupport.from_payload(support))
        return prepared
    monkeypatch.setattr(evaluation,"prepare_native_episode",changed)
    monkeypatch.setattr(evaluation,"predict_native_episode",lambda *a,**k:pytest.fail("invalid posterior reached native sampling"))
    payload,_=evaluation.evaluate_native_causal_prefix(config,joint_prefix_case["checkpoint"])
    assert payload["counts"]["failed_contrasts"]==1 and payload["counts"]["defined_contrasts"]==0


@pytest.mark.parametrize("future_dependent",[False,True])
def test_known_controls_through_consumed_evaluator_are_separate_from_native(point_prefix_case,monkeypatch,future_dependent):
    config=copy.deepcopy(point_prefix_case["config"]);contrast=config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"][0]
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=[contrast]
    expected=[]
    def controlled(model,records,config,arrays,features,*,prepared):
        context=prepared[1]["diffusion_context"];norm=model.field_rollout_operator.field_normalization
        initial=context["initial_state"]
        shifts=context["occupancy"][:,:,2].mean((2,3,4)) if future_dependent else torch.zeros(1,len(records))
        expected.append(2*abs(float(shifts[0,0]-shifts[0,1])))
        values=norm.normalize(initial).unsqueeze(2).expand(-1,-1,3,-1,-1,-1).clone()
        values[:,:,1:]+=shifts[:,:,None,None,None,None]
        physical=norm.inverse(values)
        physical[:,:,0]=initial
        draws=physical.unsqueeze(0).expand(model.field_rollout_operator.quantized_mddm_rollout_operator.inference_sample_count,-1,-1,-1,-1,-1,-1).clone()
        return {"field_trajectory_samples":draws,"field_trajectory_std":torch.zeros_like(physical),
                "router_diagnostics":{"status":"labelled_synthetic_control"}}
    monkeypatch.setattr(evaluation,"predict_native_episode",controlled)
    payload,_=evaluation.evaluate_native_causal_prefix(config,point_prefix_case["checkpoint"])
    assert payload["diagnostic_status"]=="complete",payload
    assert payload["contrasts"][0]["energy_u_statistic"]==pytest.approx(expected[0],abs=1e-6)
    assert (expected[0]>0)==future_dependent
    assert payload["causal_decision_rule_status"]=="unbound" and payload["status"]=="incomplete"


def test_failed_contrasts_keep_complete_denominator_and_do_not_duplicate_union(point_prefix_case,monkeypatch):
    config=copy.deepcopy(point_prefix_case["config"])
    valid=config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"][0]
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=[
        valid,dict(valid,contrast_uid="bad-action",regimen_uids=["regimen-a","missing"]),
        dict(valid,contrast_uid="bad-cutoff",future_knot_hours=1.5),
        dict(valid,contrast_uid="empty-prefix",future_knot_hours=1.)]
    calls=[];original=evaluation.predict_native_episode
    def run(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(evaluation,"predict_native_episode",run)
    payload,_=evaluation.evaluate_native_causal_prefix(config,point_prefix_case["checkpoint"])
    assert len(calls)==1
    assert payload["counts"]["declared_contrasts"]==4
    assert payload["counts"]["defined_contrasts"]==1 and payload["counts"]["failed_contrasts"]==2
    assert payload["counts"]["undefined_contrasts"]==1
    assert payload["diagnostic_status"]=="incomplete" and payload["counts"]["complete_episodes"]==0


def test_unsupported_policy_namespace_and_occupied_output_before_model(point_prefix_case,tmp_path,monkeypatch):
    config=copy.deepcopy(point_prefix_case["config"])
    for predictor in ("initial_state_persistence","transolver_direct_mean","transolver_fitted_residual","linear_dynamics_direct_mean"):
        bad=copy.deepcopy(config);bad["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"]=predictor
        with pytest.raises(ValueError):causal_prefix_policy(bad)
    bad=copy.deepcopy(config);bad["paper_grade_full_suite"]["native_rollout_evaluation"]["source_role"]="calibration"
    with pytest.raises(ValueError,match="development"):causal_prefix_policy(bad)
    args=cli.parse_args(["--execution-mode","native","--native-operation","causal_prefix"])
    cli.resolve_native_identity(args,config)
    assert args.output_dir.endswith("dynamics_mainline_ad_causal_prefix") and args.run_id.endswith("_causal_prefix")
    explicit=cli.parse_args(["--execution-mode","native","--native-operation","causal_prefix",
        "--output-dir",str(tmp_path/"explicit"),"--run-id","declared-id"])
    cli.resolve_native_identity(explicit,config)
    assert explicit.output_dir==str(tmp_path/"explicit") and explicit.run_id=="declared-id"
    invalid=cli.parse_args(["--execution-mode","native","--native-operation","causal_prefix","--residual-artifact","unused"])
    with pytest.raises(ValueError,match="residual"):cli.resolve_native_identity(invalid,config)
    cfg=tmp_path/"config.yaml";cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    output=tmp_path/"occupied";output.mkdir();(output/"metrics.json").write_text("original-evidence")
    monkeypatch.setattr(evaluation,"evaluate_native_causal_prefix",lambda *a,**k:pytest.fail("occupied output reached model"))
    monkeypatch.setattr(sys,"argv",["evaluate_rollout_prediction.py","--execution-mode","native","--native-operation","causal_prefix",
        "--config",str(cfg),"--checkpoint",str(point_prefix_case["checkpoint"]),"--output-dir",str(output)])
    with pytest.raises(FileExistsError):cli.main()
    assert (output/"metrics.json").read_text()=="original-evidence"


def test_actual_causal_prefix_subprocess_cli(point_prefix_case,tmp_path):
    config=copy.deepcopy(point_prefix_case["config"])
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"][:1]
    cfg=tmp_path/"config.yaml";cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    output=tmp_path/"subprocess"
    result=subprocess.run([sys.executable,str(ROOT/"Libs/exp/evaluation/evaluate_rollout_prediction.py"),
        "--execution-mode","native","--native-operation","causal_prefix","--config",str(cfg),
        "--checkpoint",str(point_prefix_case["checkpoint"]),"--output-dir",str(output),
        "--freeze-manifest",str(tmp_path/"absent-freeze.json")],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==2,result.stdout+result.stderr
    payload=json.loads((output/"metrics.json").read_text())
    assert payload["diagnostic_status"]=="complete",payload
    assert payload["status"]=="incomplete" and payload["causal_decision_rule_status"]=="unbound"
    validate_prefix_payload(payload,config)


@pytest.mark.parametrize("cutoff,label",[
    (float("nan"),"nan"),(float("inf"),"inf"),(float("-inf"),"-inf"),("bad-cutoff",None)])
def test_rejected_cutoff_keeps_strict_json_evidence(cutoff,label):
    config=load_config("Libs/configs/config_stack.yaml")
    config["paper_grade_full_suite"]["native_rollout_evaluation"].update(
        sample_uids=["sample"],regimen_uids=["a","b"])
    declaration={"contrast_uid":"rejected","sample_uid":"sample","episode_uid":"sample",
                 "regimen_uids":["a","b"],"future_knot_hours":cutoff}
    row=contrast_row(0,declaration,None)
    with pytest.raises(ValueError,match="finite source-grid hour") as rejection:
        declared_contrast(declaration,config,set())
    row["reason"]=str(rejection.value)
    payload=build_prefix_payload([row],{},{},config)
    expected={"invalid_value_type":"nonfinite_float","value":label} if label else cutoff
    assert row["future_knot_hours"]==row["declaration"]["future_knot_hours"]==expected
    assert row["declaration"] is not declaration
    assert math.isnan(declaration["future_knot_hours"]) if label=="nan" else declaration["future_knot_hours"]==cutoff
    with pytest.raises(ValueError,match="finite source-grid hour"):
        declared_contrast(row["declaration"],config,set())
    assert payload["counts"]["declared_contrasts"]==payload["counts"]["failed_contrasts"]==1
    assert payload["counts"]["defined_contrasts"]==0
    assert payload["status"]==payload["diagnostic_status"]=="incomplete"
    assert payload["causal_decision_rule_status"]=="unbound"
    assert json.loads(json.dumps(payload,allow_nan=False))==payload


@pytest.mark.parametrize("include_defined",[False,True])
def test_nonfinite_rejections_survive_actual_cli_writer(tmp_path,monkeypatch,include_defined):
    # Reporting-only numerical fixture: real declaration/matching/scoring/CLI
    # writer; source I/O, checkpoint preparation and prediction are substitutes.
    # No native model or stochastic sampler is constructed or run.
    config=load_config("Libs/configs/config_stack.yaml")
    config["selected_experiment_family"]="dynamics_mainline_ad"
    config["selected_formal_seed"]=41
    config["screening_pipeline"]["screening_benchmark"]["training"].update(
        seed_policy="manuscript_native_five",random_seed=41)
    base=config["paper_grade_full_suite"]["native_rollout_evaluation"]
    base.update(source_role="development",sample_uids=["sample","rejected-sample"],
                regimen_uids=["a","b"],checkpoint_scope="folds",checkpoint_split_name="fixture")
    valid={"contrast_uid":"defined","sample_uid":"sample","episode_uid":"sample",
           "regimen_uids":["a","b"],"future_knot_hours":2.}
    rejected=[dict(valid,contrast_uid=label,future_knot_hours=float(label))
              for label in ("nan","inf","-inf")]
    rejected[-1].update(sample_uid="rejected-sample",episode_uid="rejected-sample")
    config["paper_grade_full_suite"]["native_causal_prefix_evaluation"]["contrasts"]=(
        [valid] if include_defined else [])+rejected

    prepared,records,mask,grid,field=prepared_point_case()
    samples,_,_,_=arithmetic_case([[0.,1.],[2.,3.],[4.,2.]],[[3.,-1.],[1.,2.],[0.,3.]])
    expected=score_prefix_pair(samples,mask,grid,2.,field.field_normalization)
    norm_path=tmp_path/"normalization.json"
    norm_path.write_text(json.dumps(field.field_normalization.artifact))
    field.field_normalization.spec["artifact_ref"]=str(norm_path)
    units={"episode_field":base["episode_field"],"group_field":base["group_field"],
           "group_kind":base["group_kind"],"independent_episode_units":True,
           "evidence_ref":"labelled-reporting-fixture"}
    for index,row in enumerate(records):
        row["teacher"].update(source_uid="fixture-source",family_uid="fixture-family",
            action_uid="action-"+str(index),trajectory_uid="trajectory-"+str(index),
            validity_mask={"kind":"all_valid","evidence_ref":"labelled-reporting-fixture"})
    registry={"grid":grid,"sources":[{"source_uid":"fixture-source","predictive_evaluation_units":units}]}
    registry_path=tmp_path/"reporting-fixture.json";registry_path.write_text(json.dumps(registry))
    config["native_diffusion_data"]["registry_path"]=str(registry_path)
    monkeypatch.setattr(native,"load_registry",lambda *a,**k:registry)
    monkeypatch.setattr(native,"select_records",lambda *a,**k:records)
    calls={"conditioning":0,"prediction":0}
    def load_records(*args,**kwargs):
        assert kwargs["include_teacher"] is False
        calls["conditioning"]+=1
        return {},[]
    monkeypatch.setattr(native,"load_records",load_records)
    field.quantized_mddm_rollout_operator=SimpleNamespace(inference_sample_count=3,
        inference_denoising_step_count=2,sample_chunk_size=3,denoising_token_chunk_size=1,
        sampler_seed=3407,quantile_levels=[.05,.5,.95])
    field.neuronsifter_world_action_operator=object()
    model=SimpleNamespace(field_rollout_operator=field,parameters=lambda:iter(()))
    monkeypatch.setattr(evaluation,"load_native_conditioning_checkpoint",
        lambda *a,**k:(model,{"kind":"reporting_fixture"},torch.zeros(2,1),{},[]))
    monkeypatch.setattr(evaluation,"prepare_native_episode",lambda *a,**k:prepared)
    def predict(*args,**kwargs):
        calls["prediction"]+=1
        return {"field_trajectory_samples":samples.unsqueeze(1),
                "field_trajectory_std":samples.std(0).unsqueeze(0),
                "router_diagnostics":{"status":"labelled_reporting_fixture"}}
    monkeypatch.setattr(evaluation,"predict_native_episode",predict)
    cfg=tmp_path/"config.yaml";cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    output=tmp_path/"written"
    monkeypatch.setattr(sys,"argv",["evaluate_rollout_prediction.py","--execution-mode","native",
        "--native-operation","causal_prefix","--config",str(cfg),
        "--checkpoint",str(tmp_path/"absent-unused-checkpoint.pt"),"--output-dir",str(output),
        "--freeze-manifest",str(tmp_path/"absent-freeze.json")])
    with pytest.raises(SystemExit) as stopped:cli.main()
    assert stopped.value.code==2
    def reject_constant(value):raise AssertionError("Nonstandard JSON token: "+value)
    payload=json.loads((output/"metrics.json").read_text(),parse_constant=reject_constant)
    validate_prefix_payload(payload,config)
    assert payload["counts"]["declared_contrasts"]==3+int(include_defined)
    assert payload["counts"]["failed_contrasts"]==3
    assert payload["counts"]["defined_contrasts"]==int(include_defined)
    assert payload["counts"]["declared_episodes"]==2 and payload["counts"]["complete_episodes"]==0
    assert payload["status"]==payload["diagnostic_status"]=="incomplete"
    assert payload["causal_decision_rule_status"]=="unbound"
    assert calls=={"conditioning":int(include_defined),"prediction":int(include_defined)}
    for row,label in zip(payload["contrasts"][-3:],("nan","inf","-inf")):
        evidence={"invalid_value_type":"nonfinite_float","value":label}
        assert row["status"]=="failed" and row["reason"]=="Future knot must be a finite source-grid hour"
        assert row["declaration"]["future_knot_hours"]==row["future_knot_hours"]==evidence
    if include_defined:
        row=payload["contrasts"][0]
        assert row["future_knot_hours"]==row["declaration"]["future_knot_hours"]==2.
        for key in expected:assert row[key]==expected[key]
        for metric in ("energy_u_statistic","paired_path_rms"):
            assert payload["metrics"][metric]["mean"]==expected[metric]
            assert payload["metrics"][metric]["episode_count"]==1
            assert payload["metrics"][metric]["declared_contrast_count"]==4
    else:assert payload["metrics"]["energy_u_statistic"]["mean"] is None
    for name in ("status.json","artifact_manifest.json","run_metadata.json","artifact_hashes.json"):
        artifact=json.loads((output/name).read_text(),parse_constant=reject_constant)
        if name=="status.json":assert artifact["status"]=="incomplete"
        elif name in ("artifact_manifest.json","run_metadata.json"):assert artifact["postrun_status"]=="incomplete"
    assert json.loads(json.dumps(payload,allow_nan=False))==payload
