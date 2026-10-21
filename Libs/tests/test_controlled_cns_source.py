"""Source-free unit inputs only; the production cohort never runs in pytest."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model import controlled_cns_source as source
from Libs.scripts.builders import build_controlled_native_source as builder
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store,teacher_precision_from_uncertainty
from Libs.exp.training.pretrain_microenvironment_world_model import (
    MicroenvironmentZarrDataset,MicroenvironmentTeacherFamilyDataset,collate_native_sources,uncertainty_weighted_mse)
from Libs.tests.test_native_diffusion_data import producer_case


@pytest.fixture(scope="module")
def config():
    return load_config("Libs/configs/exp/controlled_native_source.yaml")


@pytest.fixture
def unit_source(config):
    s=source.definition(config)
    s["source_uid"]="unit_controlled_source"
    s["snapshot_ref"]="unit_controlled_definition"
    s["grid"]["time_hours"]=[0.,1.,2.]
    return s


def unit_group(s):
    return {"sample_uid":"unit_sample","group_uid":"unit_group","support_uid":"unit_support",
            "posterior_uid":"unit_predose","particle_uids":["unit_atom_"+str(i) for i in range(4)],
            "role":"train","condition":"control","replicate":0}


def unit_law(s):
    initial=np.full((6,6,15),.1,dtype=np.float32)
    initial[...,:2]=0.
    factors={k:np.ones(len(v)) for k,v in s["factor_basis"].items()}
    return source.ControlledLaw(s,factors,initial)


def test_full_definition_and_five_factor_coordinates(config):
    s=source.definition(config)
    assert len(s["grid"]["time_hours"])==28 and s["grid"]["time_hours"][-1]==672.
    assert len(s["sites"])==36 and len(s["channel_definitions"])==15
    assert {k:len(v) for k,v in s["factor_basis"].items()}=={"z_tx":32,"z_pk":2,"z_env":9,"z_ephys":4,"z_morph":32}
    for basis in s["factor_basis"].values():
        assert sorted(i for row in basis for i in row["sites"])==list(range(36))
    assert s["baseline_intervention_history"]==[]


@pytest.mark.parametrize("fault",["negative_rate","nan_scale","grid","budget","factor","event"])
def test_definition_rejects_invalid_contract(config,fault):
    c=copy.deepcopy(config);s=c["controlled_cns_source"]
    if fault=="negative_rate":s["base_rates_per_hour"]["aggregation"]=-1
    elif fault=="nan_scale":s["reference_scales"]["free_drug"]=float("nan")
    elif fault=="grid":s["grid"]["time_hours"]=[0.,1.]
    elif fault=="budget":s["budget"]["integrations"]+=1
    elif fault=="factor":s["factor_basis"]["z_pk"].pop()
    else:s["event_convention"]["initial_output"]="post_dose"
    with pytest.raises(ValueError):source.definition(c)


def test_two_site_transport_and_paired_transfers(unit_source):
    x=np.array([[2.,4.],[6.,1.]]);vol=np.array([2.,3.])
    derivative,flux=source.transport(x,vol,np.array([[0,1]]),np.array([.5]),np.ones(2))
    np.testing.assert_array_equal((derivative*vol[:,None]).sum(0),[0.,0.])
    np.testing.assert_array_equal(flux,[[2.,-1.5]])
    law=unit_law(unit_source)
    state=np.full((36,len(source.DYNAMIC)),.25)
    production,removal,paired=law.reaction(state,np.full(36,.3))
    assert np.isfinite(production).all() and (production>=0).all() and (removal>=0).all()
    np.testing.assert_array_equal(paired[:,0]+paired[:,1],np.zeros(36))
    np.testing.assert_array_equal(paired[:,4]+paired[:,5],np.zeros(36))


def test_causal_pk_zero_input_precision_and_equal_total_history(unit_source):
    law=unit_law(unit_source);a,b=unit_source["actions"]
    times=np.array([0.,1.,335.,336.,337.])
    C,O,B=source.pk_values(times,law.pk,a,unit_source)
    np.testing.assert_array_equal(C[0],np.zeros(36))
    np.testing.assert_allclose(C[1],np.asarray(law.pk["delivery_fraction"])*.01*.025*.001/500/np.asarray(law.pk["volume_l"])*np.exp(-np.asarray(law.pk["elimination_per_hour"])))
    np.testing.assert_allclose(B,np.asarray(law.pk["bmax_mol_per_l"])*O)
    split=source.pk_values(times,law.pk,b,unit_source)[0]
    np.testing.assert_allclose(split[1:3],C[1:3]/2)
    assert (split[3]>C[3]).all()
    precise=source.pk_values(times,law.pk,b,unit_source,high_precision=True)
    np.testing.assert_allclose(split,precise[0],rtol=1e-14,atol=0)
    zero=copy.deepcopy(unit_source);zero["pk"]["total_dose_mg_per_kg"]=0.
    for value in source.pk_values(times,law.pk,a,zero):assert np.count_nonzero(value)==0
    altered=dict(b,dose_times_hours=[0.,500.])
    np.testing.assert_array_equal(source.pk_values([1.,335.],law.pk,b,unit_source)[0],
                                  source.pk_values([1.,335.],law.pk,altered,unit_source)[0])
    with pytest.raises(ValueError):source.pk_values([float("nan")],law.pk,a,unit_source)


@pytest.mark.parametrize("method",["RK4","DOP853"])
def test_event_endpoint_stages_use_left_then_right_limit(unit_source,method):
    # Instrument a constant RHS, retaining the real PK kernel and event splitter.
    # This is not a source-cohort integration or a source-law numerical claim.
    unit_source["grid"]["time_hours"]=[0.,336.,337.]
    law=unit_law(unit_source);action=unit_source["actions"][1];calls=[]
    def rhs(t,y,action,active_through,deadline):
        C=source.pk_values([t],law.pk,action,unit_source,initial_pre_dose=False,active_through=active_through)[0][0]
        calls.append((t,active_through,C))
        return np.zeros_like(y)
    law.rhs=rhs
    source.integrate(law,action,method,400.)
    left=[v for t,a,v in calls if t==336. and a<336.]
    right=[v for t,a,v in calls if t==336. and a==336.]
    assert left and right
    expected_left=source.pk_values([336.],law.pk,action,unit_source,initial_pre_dose=False,active_through=0.)[0][0]
    expected_right=source.pk_values([336.],law.pk,action,unit_source,initial_pre_dose=False,active_through=336.)[0][0]
    for value in left:np.testing.assert_array_equal(value,expected_left)
    for value in right:np.testing.assert_array_equal(value,expected_right)
    assert (expected_right>expected_left).all()


def test_scalar_discrepancy_quality_and_unresolved_zero(unit_source):
    def result(value):
        return {"trajectory":np.full((2,1,1,15),value),"balance_max":0.,"paired_balance_max":0.,"transport_balance_max":0.}
    coarse,fine,reference=result(1.00002),result(1.000001),result(1.)
    q=source.numerical_summary(coarse,fine,reference,np.ones(15),np.zeros(15),np.zeros(15),unit_source["numerics"])
    assert q["admissible"]
    expected=1.00002-1.000001
    np.testing.assert_allclose(q["uncertainty"],expected)
    assert q["teacher_confidence"]==pytest.approx(1/(1+expected/.001))
    zero=source.numerical_summary(result(0.),result(0.),result(0.),np.ones(15),np.zeros(15),np.zeros(15),unit_source["numerics"])
    assert not zero["admissible"] and "unresolved_zero_numerical_discrepancy" in zero["reasons"]
    with pytest.raises(ValueError):
        source.numerical_summary(result(float("inf")),fine,reference,np.ones(15),np.zeros(15),np.zeros(15),unit_source["numerics"])


def test_complete_prior_observation_and_physical_programs(unit_source):
    group=unit_group(unit_source)
    support,production,truth=source.draw_world(unit_source,group,0)
    assert support.initial_field.shape==(1,4,6,6,15)
    assert 0<=truth<4 and set(support.factors)==set(source.FACTOR_NAMES)
    payload=production.inference_payload()
    assert not {"generating_indices","generating_fields","truth"} & set(payload)
    observation=production.observations[0]
    assert observation.observed.tolist()==[unit_source["observation"]["mask"]]
    parameters=support.readout_parameters[observation.channel_uid]
    for atom in range(4):
        expected=[support.initial_field[0,atom,row["region_index"],row["compartment_index"],
                   unit_source["grid"]["species_id"].index(row["species"])].item()/unit_source["reference_scales"][row["species"]]
                  for row in unit_source["observation"]["selector"]]
        np.testing.assert_allclose(parameters["mean"][0,atom],expected)
    records=[]
    for atom in range(4):
        law=source.ControlledLaw(unit_source,{k:v[0,atom].numpy() for k,v in support.factors.items()},support.initial_field[0,atom].numpy())
        records.append([source.compiled_particle_program(unit_source,group,support,atom,action,
            source.pk_values(unit_source["grid"]["time_hours"],law.pk,action,unit_source),law.pk,"unit_code")
            for action in unit_source["actions"]])
    raw=source.joint_source(unit_source,group,support,production,records)
    parsed,_=native.parse_joint_source(raw)
    assert parsed.particle_uids==support.particle_uids
    assert records[0][0]["physical_program"]["mode"]=="controlled_source_bound_equilibrium.v1"
    assert records[0][0]["physical_program"]["inputs"]["run_snapshot"]["settings_origin"]=="controlled_source_parameter_definition"
    snapshot=records[0][0]["physical_program"]["inputs"]["run_snapshot"]
    assert snapshot["pk_definition"]==unit_source["pk"]
    assert snapshot["source_definition_sha256"]==source.digest(unit_source)
    selected=native.select_joint_source(raw,[group["sample_uid"]],list(reversed(raw["particle_programs"]["regimen_uids"])))
    assert selected["support"]==raw["support"]
    bad=copy.deepcopy(raw);bad["support"]["role"]="development"
    with pytest.raises(ValueError):native.parse_joint_source(bad)


def test_prebound_budget_started_before_call_deadline_and_no_retry(tmp_path,unit_source):
    clock=[0.]
    ledger=builder.SourceLedger(tmp_path/"receipt.json",unit_source,{"unit":"fixture"},clock=lambda:clock[0])
    assert len(ledger.data["cases"])==16
    for kind,count in (("integrations",48),("primary_pk_grids",64),("precision_pk_grids",64)):
        assert sum(v["kind"]==kind for v in ledger.data["operations"].values())==count
    key="integrate:0:coarse"
    def operation():
        assert json.loads(ledger.path.read_text())["operations"][key]["status"]=="started"
        return "substitute result"
    assert ledger.execute(key,operation)=="substitute result"
    with pytest.raises(RuntimeError):ledger.execute(key,operation)
    def fail():raise ValueError("substitute numerical failure")
    with pytest.raises(ValueError):ledger.execute("integrate:0:fine",fail)
    assert ledger.data["operations"]["integrate:0:fine"]["status"]=="failed"
    ledger.data["operations"]["integrate:0:reference"]["status"]="started"
    with pytest.raises(RuntimeError):ledger.execute("integrate:0:reference",operation)
    clock[0]=1801.
    with pytest.raises(TimeoutError):ledger.execute("integrate:1:coarse",operation)
    assert ledger.data["operations"]["integrate:1:coarse"]["status"]=="planned"


def test_consume_only_never_calls_source(config,tmp_path,monkeypatch):
    c=copy.deepcopy(config);c["controlled_cns_source"]["output_root"]=str(tmp_path/"unit_output")
    monkeypatch.setattr(builder,"load_config",lambda *a:c)
    monkeypatch.setattr(builder,"execute_source",lambda *a,**k:pytest.fail("consume-only source execution"))
    monkeypatch.setattr(builder,"consume_source",lambda *a,**k:{"status":"incomplete","predictive_evaluation":"NOT_RUN"})
    monkeypatch.setattr(sys,"argv",["build_controlled_native_source.py","--consume-only"])
    assert builder.main()==2


@pytest.mark.parametrize("kind",["positive","exact_zero","unresolved_zero","false_exact","boolean_error"])
def test_uncertainty_real_writer_datasets_and_bounded_weighting(tmp_path,kind):
    c,registry,arrays,store=producer_case(tmp_path)
    if kind!="positive":
        arrays["uncertainty"][:,0]=0.
        arrays["trajectory"][...,0]=0.
        arrays["initial_state"][...,0]=0.
    if kind in {"exact_zero","false_exact","boolean_error"}:
        reference="unit_zero_delivery_zero_initial_pk"
        registry["sources"][0]["exact_zero_quantities"]={"free_drug":{"kind":"identically_zero","derivation_ref":reference}}
        for row in registry["teachers"]:
            row["exact_zero_uncertainty"]={"free_drug":{"status":"exact_by_definition","derivation_kind":"identically_zero",
                "derivation_ref":reference,"serialization_error":0.,"quantity":"free_drug","source_snapshot_ref":row["source_snapshot_ref"]}}
        if kind=="boolean_error":
            for row in registry["teachers"]:row["exact_zero_uncertainty"]["free_drug"]["serialization_error"]=False
        if kind=="false_exact":arrays["trajectory"][0,1,0,0,0]=1.
    if kind in {"unresolved_zero","false_exact","boolean_error"}:
        with pytest.raises(ValueError,match="positive numerical uncertainty"):
            write_native_microenvironment_store(store,c,registry,arrays)
        return
    write_native_microenvironment_store(store,c,registry,arrays)
    Path(c["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    for dataset in (MicroenvironmentZarrDataset(store,c,split_name="train"),MicroenvironmentTeacherFamilyDataset(c,"train")):
        items=dataset.__getitems__(list(range(len(dataset))))
        batch=collate_native_sources(items)
        u=batch["uncertainty"].clone()
        if kind=="exact_zero":assert torch.count_nonzero(u[:,0])==0
        prediction=(batch["trajectory"]+1).requires_grad_()
        loss=uncertainty_weighted_mse(prediction,batch["trajectory"],u,c,torch.ones(len(dataset)))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
        assert torch.equal(u,batch["uncertainty"])
    assert teacher_precision_from_uncertainty(c,0.)==c["screening_pipeline"]["screening_benchmark"]["training"]["uncertainty_weighting"]["maximum_teacher_precision"]


@pytest.mark.parametrize("method",["RK4","DOP853"])
def test_integrator_linear_decay_against_exponential(unit_source,method):
    law=unit_law(unit_source);k=1/24.
    def rhs(t,y,action,active_through,deadline):
        dx=-k*y[:36*len(source.DYNAMIC)]
        return np.concatenate([dx,(dx.reshape(36,-1)*law.volumes[:,None]).sum(0)])
    law.rhs=rhs
    law.reaction=lambda x,o:(np.zeros_like(x),k*x,np.zeros_like(x))
    result=source.integrate(law,unit_source["actions"][0],method,.1)
    index=unit_source["grid"]["species_id"].index("misfolded_protein_load")
    expected=.1*np.exp(-k*np.array(unit_source["grid"]["time_hours"]))
    np.testing.assert_allclose(result["trajectory"][:,0,0,index],expected,rtol=3e-8,atol=1e-10)
    assert result["balance_max"]<1e-12


@pytest.fixture
def completed_consumption_fixture(config,tmp_path,monkeypatch):
    root=tmp_path/"unit_completed";root.mkdir()
    c=copy.deepcopy(config);c["controlled_cns_source"]["output_root"]=str(root)
    identity={"fixture":"frozen_unit_source"}
    teachers=[{"trajectory_uid":f"unit_{g}_{a}","sample_uid":f"unit_{g}","split_name":"train" if g==0 else "development",
               "qc_grade":"B","qc_use":"main_pretraining","source_holdout_role":"train" if g==0 else "development",
               "sample_weight":.5,"teacher_confidence":.9} for g in range(2) for a in range(2)]
    receipt={"source_status":"complete","source_identity":identity,"resource_limit":{"memory_gib":2},
             "cases":[{"case_uid":r["trajectory_uid"]} for r in teachers]}
    registry={"teachers":teachers}
    report={"status":"complete","admitted_count":4,"source_identity":identity,
            "cases":[{"case_uid":r["trajectory_uid"],"role":r["split_name"],"native_status":"admitted"} for r in teachers]}
    for name,payload in [("source_receipt.json",receipt),("native_registry.json",registry),("native_consumption.json",report)]:
        builder.write_json(root/name,payload)
    builder.write_json(root/"native_source.zarr"/".zattrs",{"source_identity":identity,"native_diffusion":registry})
    builder.write_tsv(root/"simulation_qc_catalog.tsv",list(teachers[0]),teachers)
    builder.write_json(root/"train_normalization.json",{"fit_role":"train","training_records":teachers[:2]})
    for role in ("train","development"):builder.write_json(root/(role+"_normalized.zarr")/".zattrs",{})
    monkeypatch.setattr(builder,"verify_source_identity",lambda *args:None)
    monkeypatch.setattr(builder,"load_config",lambda *args:c)
    return c,root,receipt,registry


@pytest.mark.parametrize("fault",[None,"source","registry","qc","denominator"])
def test_completed_store_reuse_binding_and_preservation(completed_consumption_fixture,fault):
    c,root,receipt,registry=completed_consumption_fixture
    if fault=="source":
        p=root/"native_source.zarr"/".zattrs";v=json.loads(p.read_text());v["source_identity"]={"stale":True};builder.write_json(p,v)
    elif fault=="registry":
        p=root/"native_registry.json";v=json.loads(p.read_text());v["teachers"][0]["sample_uid"]="wrong";builder.write_json(p,v)
    elif fault=="qc":
        p=root/"simulation_qc_catalog.tsv";p.write_text(p.read_text().replace("\tB\t","\tD\t",1))
    elif fault=="denominator":receipt["cases"].reverse()
    paths=[root/"native_consumption.json",root/"source_receipt.json",root/"native_source.zarr"/".zattrs"]
    before=[p.read_bytes() for p in paths]
    if fault:
        with pytest.raises(ValueError):builder.reuse_completed_store(root,receipt)
    else:
        report=builder.reuse_completed_store(root,receipt)
        assert len(report["cases"])==4 and all(r["native_status"]=="candidate" for r in report["cases"])
    assert [p.read_bytes() for p in paths]==before


def test_completed_consumption_releases_datasets_and_preserves_groups(completed_consumption_fixture,monkeypatch):
    import weakref
    c,root,receipt,registry=completed_consumption_fixture
    for name in ("execute_source","build_native_source_store","call_preparation"):
        monkeypatch.setattr(builder,name,lambda *a,**k:pytest.fail("completed consumption must not produce or overwrite"))
    active=weakref.WeakSet();tensor_refs=[];visits=[]
    class Dataset:
        def __init__(self,role):
            assert not active, "previous dataset must be released before the next constructor"
            active.add(self)
            self.native_records=[{"teacher":r} for r in registry["teachers"] if r["split_name"]==role]
            self.native_registry={"grid":{"normalization":{"artifact_ref":str(root/"train_normalization.json"),"fit_role":"train"}}}
        def __getitems__(self,indices):
            assert all(ref() is None for ref in tensor_refs), "previous group payload retained"
            rows=[self.native_records[i]["teacher"] for i in indices]
            assert len(rows)==2 and len({r["sample_uid"] for r in rows})==1
            visits.append([r["trajectory_uid"] for r in rows])
            result=[{"trajectory_uid":r["trajectory_uid"],"trajectory":torch.ones(2),"uncertainty":torch.ones(2),"joint_source":{}} for r in rows]
            tensor_refs.extend(weakref.ref(row["trajectory"]) for row in result)
            return result
    monkeypatch.setattr(builder,"MicroenvironmentZarrDataset",lambda store,c,split_name:Dataset(split_name))
    monkeypatch.setattr(builder,"MicroenvironmentTeacherFamilyDataset",lambda c,role:Dataset(role))
    monkeypatch.setattr(builder,"_native_episode_identity",lambda *args:None)
    monkeypatch.setattr(builder,"_native_validity_mask",lambda *args:None)
    monkeypatch.setattr(builder.native,"load_records",lambda *a,**k:({},{}))
    before=(root/"source_receipt.json").read_bytes(),(root/"native_consumption.json").read_bytes()
    report=builder.consume_source(c,Path("unit_config"),reuse_completed=True)
    assert report["status"]=="complete" and report["admitted_count"]==4
    assert visits==[["unit_0_0","unit_0_1"]]*2+[["unit_1_0","unit_1_1"]]*2
    assert not active and all(ref() is None for ref in tensor_refs)
    assert before==((root/"source_receipt.json").read_bytes(),(root/"native_consumption.json").read_bytes())
    saved=(root/"native_consumption_recheck.json").read_bytes()
    with pytest.raises(FileExistsError):builder.consume_source(c,Path("unit_config"),reuse_completed=True)
    assert (root/"native_consumption_recheck.json").read_bytes()==saved
