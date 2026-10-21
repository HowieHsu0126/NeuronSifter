"""Public CNS preferences and constrained posterior controls on software fixtures."""
import copy
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from Libs.models.modules.world_state import JointWorldStatePosterior, FACTOR_NAMES, _posterior_uid
from Libs.tests.test_native_romani_integration import initialize_fixture
from Libs.utils.data.native_observable_profile import observable_profile, family_grid, quantity_support
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_intervention_decisions import (
    fit_public_loss_reference, public_cns_loss, compiled_action_burden, posterior_control,
)
from Libs.utils.evaluation.manuscript_native_protocol import training_profile, command_catalog
from Libs.utils.model import native_scientific_source as source


def config():
    return load_config("Libs/configs/exp/manuscript_native_protocol.yaml")


def reference_records(cfg):
    rows=[]
    for family in observable_profile(cfg)["families"]:
        grid=family_grid(observable_profile(cfg),family)
        times=np.asarray(grid["time_hours"])
        target=np.broadcast_to((times/times[-1])[:,None,None,None],quantity_support(grid).shape).copy()
        rows.append({"sample_uid":family+":train","trajectory_uid":family+":trajectory",
            "family_uid":family,"role":"train","qc_grade":"A","content_hash":hashlib.sha256(target.tobytes()).hexdigest(),
            "trajectory":target,"quantity_support":quantity_support(grid),"grid":grid})
    return rows


def test_public_loss_training_reference_four_domains_isf_and_clipping():
    cfg=config();records=reference_records(cfg);ids={row["sample_uid"] for row in records}
    artifact=fit_public_loss_reference(cfg,records,ids)
    assert artifact["status"]=="defined" and len(artifact["training_records"])==4
    for row in records:
        grid=row["grid"];field=torch.tensor(row["trajectory"]);mask=torch.from_numpy(row["quantity_support"])
        result=public_cns_loss(field,mask,grid,artifact,torch.tensor(.2),cfg)
        assert float(result["loss"])==pytest.approx(.52,abs=1e-7)
        assert float(result["clipped_fraction"].max())==0.
        # Other compartments remain in the native field but cannot affect fixed ISF readouts.
        changed=field.clone()
        for ci,name in enumerate(grid["compartment_id"]):
            if name!="isf":changed[:,:,ci,:]=100.
        assert torch.equal(public_cns_loss(changed,mask,grid,artifact,torch.tensor(.2),cfg)["loss"],result["loss"])
        saturated=public_cns_loss(field*0+2,mask,grid,artifact,torch.tensor(1.),cfg)
        assert float(saturated["loss"])==pytest.approx(1.1)
        assert saturated["maximum_supported_loss"]==1.1 and bool((saturated["clipped_fraction"]==1).all())
    bad=copy.deepcopy(records);bad[0]["role"]="validation"
    with pytest.raises(ValueError,match="training records"):
        fit_public_loss_reference(cfg,bad,ids)


def test_public_loss_rejects_constant_missing_and_invalid_units():
    cfg=config();records=reference_records(cfg);ids={row["sample_uid"] for row in records}
    bad=copy.deepcopy(records)
    for row in bad:row["trajectory"][...,row["grid"]["species_id"].index("amyloid_plaque")]=1.
    undefined=fit_public_loss_reference(cfg,bad,ids)
    assert undefined["status"]=="undefined_constant_endpoint"
    row=records[0];field=torch.tensor(row["trajectory"]);mask=torch.from_numpy(row["quantity_support"])
    with pytest.raises(ValueError,match="defined training-reference"):
        public_cns_loss(field,mask,row["grid"],undefined,torch.tensor(0.),cfg)
    artifact=fit_public_loss_reference(cfg,records,ids)
    missing=mask.clone()
    missing[:,0,row["grid"]["compartment_id"].index("isf"),row["grid"]["species_id"].index("phosphorylated_tau")]=False
    with pytest.raises(ValueError,match="missing public CNS"):
        public_cns_loss(field,missing,row["grid"],artifact,torch.tensor(0.),cfg)
    grid=copy.deepcopy(row["grid"]);grid["species_units"]["amyloid_plaque"]="incorrect_unit"
    with pytest.raises(ValueError,match="units"):
        public_cns_loss(field,mask,grid,artifact,torch.tensor(0.),cfg)
    with pytest.raises(ValueError,match="not clipped"):
        public_cns_loss(field,mask,row["grid"],artifact,torch.tensor(1.1),cfg)


def test_actual_shared_stimulation_cost_matches_and_total_dose_is_not_first_dose():
    cfg=config();groups=source.group_definitions(cfg,role_filter="train")
    costs=[]
    for stratum in ("shared_primary","romani"):
        group=next(g for g in groups if g.get("source_stratum")==stratum)
        scoped=source.scoped_group_config(cfg,group)
        generated=source.draw_group(scoped,group,0,physical_initializer=initialize_fixture if stratum=="romani" else None)
        action=source.action_definition(scoped,source.ELECTRICAL,0,group)
        record=source.compiled_program(scoped,group,generated["support"],0,generated["parameters"][0],action,"fixture")
        costs.append(compiled_action_burden(record,source.ELECTRICAL,cfg))
    assert costs[0]==costs[1]
    assert costs[0]["actual_amount"]==pytest.approx(.00012*6)
    assert costs[0]["fixed_support_maximum"]==80.
    record={"action":{"dose":.005,"dose_unit":"mg/kg","schedule":{"values":[0.,336.],"dose_fractions":[.5,.5]}},
            "physical_program":{"inputs":{"run_snapshot":{"pk_definition":{"total_dose_mg_per_kg":.01}}}}}
    burden=compiled_action_burden(record,"cns_long_horizon_world",cfg)
    assert burden["actual_amount"]==.01 and burden["normalized_burden"]==.0025
    record["action"]["dose"]=.007
    with pytest.raises(ValueError,match="total dose"):
        compiled_action_burden(record,"cns_long_horizon_world",cfg)


def posterior_fixture(cfg,group=None):
    group=group or next(g for g in source.group_definitions(cfg,role_filter="train") if g["family_uid"]==source.ELECTRICAL)
    support=source.draw_group(source.scoped_group_config(cfg,group),group,0)["support"]
    logits=torch.linspace(-2,2,support.particle_count,dtype=torch.float64)[None]
    logits[:,::3]+=2.
    logs=logits-logits.logsumexp(-1,keepdim=True)
    context=torch.zeros((*support.initial_field.shape[:4],3))
    observations=("fixture:retained",)
    uid=_posterior_uid(support,observations,None,logs,context)
    return JointWorldStatePosterior(support,logs,context,torch.zeros(1,6),observations,0,uid,None,
                                    torch.zeros_like(logs),"software_fixture")


def test_factor_product_preserves_marginals_complete_tuples_and_point_identity():
    cfg=config();posterior=posterior_fixture(cfg);before=copy.deepcopy(posterior.support.to_payload())
    independent=posterior_control(posterior,"posterior_dependence_ablated")
    assert independent.support is posterior.support and not torch.allclose(independent.log_weights,posterior.log_weights)
    for name in FACTOR_NAMES:
        _,inverse=torch.unique(posterior.support.factors[name][0],dim=0,return_inverse=True)
        for value in inverse.unique():
            torch.testing.assert_close(independent.log_weights.exp()[0,inverse==value].sum(),
                                       posterior.log_weights.exp()[0,inverse==value].sum(),rtol=1e-12,atol=1e-12)
    point=posterior_control(posterior,"posterior_point")
    assert point.support is posterior.support and (point.weights>0).sum()==1
    assert point.weights.argmax()==posterior.weights.argmax()
    assert point.particle_context is posterior.particle_context
    independent.assert_current_binding();point.assert_current_binding()
    assert posterior.support.to_payload()==before
    assert posterior_control(posterior,"posterior_joint") is posterior
    incomplete=copy.deepcopy(cfg)
    incomplete["native_observable_bridge"]["atoms"]=incomplete["native_observable_bridge"]["atoms"][:16]
    incomplete["native_observable_bridge"]["prior_weights"]=[1/16]*16
    # Remove a combination, rather than an entire factor level.
    incomplete["native_observable_bridge"]["atoms"][-1]=cfg["native_observable_bridge"]["atoms"][16]
    broken=posterior_fixture(incomplete)
    with pytest.raises(ValueError,match="Cartesian"):
        posterior_control(broken,"posterior_dependence_ablated")


def test_posterior_profiles_reuse_native_fit_and_bind_training_reference_command():
    cfg=config();specs=[]
    for method in cfg["native_intervention_evaluation"]["posterior_methods"]:
        selected,spec=training_profile(cfg,method,41)
        assert spec["checkpoint_reuse_method"]=="native_wam" and spec["additional_optimizer_updates"]==0
        assert selected["native_scientific_training"]["conditioning"]["mode"]=="joint"
        specs.append(spec)
    commands={row["command_uid"]:row for row in command_catalog(cfg,specs)}
    assert commands["public_cns_reference"]["reference_role"]=="train"
    for spec in specs:
        assert spec["profile_id"]+":fit" not in commands
        row=commands["posterior_comparison_fraction_100_seed_41:evaluate"]
        assert row["dependencies"]==["native_wam_fraction_100_seed_41:fit","public_cns_reference"]
        assert "--decision-reference" in row["argv"] and len(row["outputs"])==3
    assert sum(row["command_uid"].endswith(":evaluate") for row in commands.values())==1


def test_shared_sampler_weighted_clipping_control_coverage_and_supported_sets(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from Libs.utils.evaluation import native_intervention_evaluation as evaluation
    cfg=config();cfg["selected_formal_seed"]=40
    training=cfg["screening_pipeline"]["screening_benchmark"]["training"]
    training.update(random_seed=40,seed_policy="manuscript_native_five")
    cfg["native_scientific_training"]["predictive_budget"].update(conditional_draws_per_particle=2,categorical_reverse_steps=1)
    records=reference_records(cfg)
    reference=fit_public_loss_reference(cfg,records,{row["sample_uid"] for row in records})
    grid=records[0]["grid"];mask=torch.from_numpy(records[0]["quantity_support"])[None].expand(3,-1,-1,-1,-1)
    declaration=next(g for g in source.group_definitions(cfg) if g["family_uid"]==source.ELECTRICAL and g["role"]=="development")
    posterior=posterior_fixture(cfg,declaration)
    j=posterior.support.particle_count;shape=mask.shape
    samples=torch.full((2,1,j,*shape),2.)
    chosen=int(posterior.weights.argmax())
    for draw in range(2):
        for action in range(3):samples[draw,0,chosen,action]=.2+.1*action+.05*draw
    times=grid["query_time"]["values"]
    actions=[{"action":{"stimulation_or_environment":{"magnitude_or_program":{
        "time_values":times,"values":[0.,a+1.,a+1.,a+1.,0.,0.,0.],
        "value_unit":"uA","time_unit":"ms","interpolation":"left_hold"}}}} for a in range(3)]
    inputs=SimpleNamespace(posterior=posterior,grid=grid,regimen_uids=("a","b","c"),
        programs={"records":[[copy.deepcopy(actions) for _ in range(j)]]})
    targets=torch.stack([torch.full(tuple(shape[1:]),value) for value in (.9,.1,.3)])[None]
    holder={"target":targets}
    monkeypatch.setattr(evaluation,"prepare_comparator_group",lambda *args:(inputs,holder["target"],mask[None],None))
    class Belief(torch.nn.Module):
        def __init__(self):
            super().__init__();self.marker=torch.nn.Parameter(torch.zeros(()));self.field_normalization=object()
    class Sampler:
        def __init__(self):self.calls=0
        def sample_native(self,actual,normalization,draws,steps,seed):
            assert actual is inputs and draws==2 and steps==1
            self.calls+=1
            return {"particle_trajectory_samples":samples}
    class Dataset(list):
        group_uids=[declaration["sample_uid"]]
    dataset=Dataset([{"native_joint_group":{"joint_source":{},"items":[]}}])
    model=Sampler();belief=Belief();declared={declaration["sample_uid"]:declaration}
    first=evaluation.evaluate_native_interventions(cfg,"posterior_joint",model,belief,{},dataset,declared,reference,tmp_path/"first")
    assert first["status"]=="complete" and model.calls==first["shared_sampling_calls"]==1
    choices={name:report["groups"][0]["selected_regimen_uid"] for name,report in first["reports"].items()}
    point=first["reports"]["posterior_point"]["groups"][0]
    assert np.asarray(point["predictive_clipped_fractions"]).max()==0.
    assert np.asarray(first["reports"]["posterior_joint"]["groups"][0]["predictive_clipped_fractions"]).min()>0.
    assert point["conditional_draws_per_particle"]==2
    interval=point["loss_intervals_by_nominal"]["0.8"]
    assert all(a<b for a,b in zip(interval["lower"],interval["upper"]))
    assert interval["independent_calibration_qualified"] is False
    holder["target"]=1.-targets
    second=evaluation.evaluate_native_interventions(cfg,"posterior_point",model,belief,{},dataset,declared,reference,tmp_path/"second")
    assert model.calls==2
    assert {name:report["groups"][0]["selected_regimen_uid"] for name,report in second["reports"].items()}==choices
    for particle in inputs.programs["records"][0]:
        particle[2]["action"]["stimulation_or_environment"]["magnitude_or_program"]["values"][1:4]=[9.,9.,9.]
    partial=evaluation.evaluate_native_interventions(cfg,"posterior_joint",model,belief,{},dataset,declared,reference,tmp_path/"partial")
    assert partial["status"]=="incomplete" and model.calls==3
    for report in partial["reports"].values():
        row=report["groups"][0]
        assert (row["declared_action_count"],row["supported_action_count"],row["withheld_action_count"])==(3,2,1)
        assert row["status"]=="defined_on_supported_actions"
        assert "supported_set_intervention_regret" in row["metrics"]


def test_reference_content_identity_rejection_uses_metadata_without_array_reads(tmp_path,monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace
    import json
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_comparator_evaluation as evaluation
    cfg=config();records=reference_records(cfg);ids={row["sample_uid"] for row in records}
    artifact=fit_public_loss_reference(cfg,records,ids)
    path=tmp_path/"reference.json";path.write_text(json.dumps(artifact))
    groups=[]
    for row in artifact["training_records"]:
        groups.append((SimpleNamespace(native_records=[{"teacher":dict(row)}]),[0],None))
    class Dataset:
        def __init__(self):self.groups=groups
        def __getitem__(self,index):raise AssertionError("Metadata validation reread training arrays")
    dataset=Dataset()
    monkeypatch.setattr(evaluation,"declared_dataset",lambda *args,**kwargs:(dataset,{uid:{"role":"train"} for uid in ids}))
    args=Namespace(command="validate_decision_reference",baseline_name="posterior_joint",
                   output_dir=str(tmp_path/"validation"),decision_reference=path)
    assert cli.run_native_command(args,cfg)==0
    groups[0][0].native_records[0]["teacher"]["content_hash"]="different_current_source"
    with pytest.raises(ValueError,match="record identities/content"):
        cli.run_native_command(args,cfg)
