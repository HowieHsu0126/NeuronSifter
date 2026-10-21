import copy
from types import SimpleNamespace

import pytest
import torch
import yaml
from pathlib import Path

from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.utils.model.native_transolver_residual import (
    LAW,CONTROLLED_LAW,episode_statistics,fit_scales,residual_draws)
from Libs.utils.evaluation.native_comparator_evaluation import weighted_native_scores


def test_explicit_native_groups_reject_missing_duplicate_and_wrong_role(monkeypatch):
    teachers=[]
    for uid,role in (("train-a","train"),("train-b","train"),("dev-a","development")):
        for action in ("left","right"):
            teachers.append({"sample_uid":uid,"split_name":role,"regimen_uid":action,"posterior_uid":uid+":p"})
    monkeypatch.setattr(training,"load_family_registries",lambda config, *, input_locations=None:{"domain":{"teachers":teachers}})
    config={"native_diffusion_data":{},"screening_pipeline":{"screening_benchmark":{"training":{
        "teacher_family_training":{"families":[{"family_key":"domain","teacher_loss_weight":1.}]}}}}}
    selected=training.NativeJointGroupDataset(config,"train",group_uids={"train-b"})
    assert len(selected)==1 and selected.family_group_counts=={"domain":1}
    with pytest.raises(ValueError,match="Duplicate"):
        training.NativeJointGroupDataset(config,"train",group_uids=["train-a","train-a"])
    with pytest.raises(ValueError,match="missing"):
        training.NativeJointGroupDataset(config,"train",group_uids={"missing"})
    with pytest.raises(ValueError,match="role"):
        training.NativeJointGroupDataset(config,"train",group_uids={"dev-a"})


def test_native26_residual_is_explicit_and_preserves_historical_rng():
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/"Libs/configs/exp/manuscript_native_protocol.yaml").read_text())
    numerical=cfg["native_controlled_comparators"]["residual_calibration"]["numerical"]
    means=torch.zeros(1,1,4,2,2,26,dtype=torch.float64)
    target=torch.randn(means.shape[1:],generator=torch.Generator().manual_seed(494),dtype=torch.float64)
    row=episode_statistics(means,[1.],target,torch.ones_like(target,dtype=torch.bool),
                           [0.,224.,448.,672.],672.,"episode","group")
    old=fit_scales([row],["group"],numerical)
    assert old["status"]=="FIT_FAILED" and old["reason"]=="incomplete_group_or_species_information"
    fit=fit_scales([row],["group"],numerical,species_count=26)
    assert fit["status"]=="FIT_COMPLETE" and len(fit["scales"])==26
    arguments=dict(episode_uid="e",residual_seed=40,draw_ids=[0,1])
    default=residual_draws(means,fit["scales"],[0.,224.,448.,672.],672.,torch.tensor([0,0]),**arguments)
    historical=residual_draws(means,fit["scales"],[0.,224.,448.,672.],672.,torch.tensor([0,0]),law=LAW,**arguments)
    native=residual_draws(means,fit["scales"],[0.,224.,448.,672.],672.,torch.tensor([0,0]),law=CONTROLLED_LAW,**arguments)
    assert torch.equal(default,historical) and not torch.equal(native,historical)
    assert torch.equal(native[:,:,0],means[torch.tensor([0,0]),:,0])


def test_weighted_stratum_metrics_use_weights_and_conditional_pair_correction():
    samples=torch.tensor([0.,0.,10.,10.],dtype=torch.float64).reshape(2,2,1,1,1,1,1)
    target=torch.ones(1,1,1,1,1,dtype=torch.float64)
    mask=torch.ones_like(target,dtype=torch.bool)
    normalizer=SimpleNamespace(normalize=lambda value,support:value)
    actual=weighted_native_scores(samples,torch.tensor([.9,.1],dtype=torch.float64),target,mask,normalizer,[.8,.9])
    assert actual["crps"]==pytest.approx(.9)
    assert actual["normalized_rmse"]==pytest.approx(0.)
    assert actual["predictive_coverage_by_nominal"]=={"0.8":0.,"0.9":1.}


def test_codano_does_not_mutate_declared_modes_or_quantity_ids():
    from Libs.models.modules.native_diffusion_comparators import NativeCODANO
    root=Path(__file__).resolve().parents[2]
    entry=copy.deepcopy(yaml.safe_load((root/"Libs/configs/exp/manuscript_native_protocol.yaml").read_text())[
        "native_controlled_comparators"]["references"]["codano"])
    entry["hyperparameters"].update(n_layers=1,lifting_channels=4,hidden_variable_codimension=2,
        projection_channels=4,conditioning_channels=2,positional_encoding_dim=2,
        positional_encoding_modes=[2,2],n_modes=[2,2],n_heads=1)
    original=copy.deepcopy(entry)
    quantities=["quantity_"+str(i) for i in range(26)]
    expected=list(quantities)
    model=NativeCODANO(entry,(3,1,1,26),26,quantities)
    assert entry==original and quantities==expected
    assert model.network.positional_encoding_modes==[2,1]


def test_native_linear_typed_forcing_preserves_prefix_and_masks_unavailable_intermediates():
    from Libs.models.modules.native_linear_dynamics import NativeLinearDynamics
    from Libs.tests.test_native_linear_dynamics import mathematical_case

    _,normalizer,context=mathematical_case(3,(0.,.5,1.,2.))
    model=NativeLinearDynamics(normalizer.artifact["domain"],2,
        program_channel_count=6,normalization_scope="registered_native").double()
    normalizer.quantity_profile={"fixture":"explicit_support"}
    context["quantity_support"]=torch.ones(1,1,4,1,1,3,dtype=torch.bool)
    context["quantity_support"][...,-1]=False
    context["initial_state"][...,-1]=0
    context["native_grid"]=copy.deepcopy(normalizer.artifact["domain"])
    context["additional_forcing"]=torch.zeros(1,1,4,1,1,4,dtype=torch.float64)
    context["additional_forcing"][...,0]=1.
    with torch.no_grad():
        model.A[0,2]=1000.
        model.A[2,0]=1000.
        model.B[0,2]=2.
        model.B[2,2]=1000.
    result=model(context,normalizer)
    expected=torch.tensor([1.,2.,3.,5.],dtype=torch.float64)
    torch.testing.assert_close(result["normalized_trajectory"][0,0,:,0,0,0],expected)
    assert torch.equal(result["field_trajectory"][...,-1],torch.zeros_like(result["field_trajectory"][...,-1]))
    changed={key:value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key,value in context.items()}
    changed["additional_forcing"][:,:,2:,...,0]=3.
    later=model(changed,normalizer)["field_trajectory"]
    assert torch.equal(later[:,:,:3],result["field_trajectory"][:,:,:3])
    assert not torch.equal(later[:,:,3:],result["field_trajectory"][:,:,3:])
    result["field_trajectory"].square().sum().backward()
    assert model.B.grad[:,2].abs().sum()>0
    malformed=dict(context)
    malformed.pop("additional_forcing")
    with pytest.raises(ValueError,match="typed forcing"):
        model(malformed,normalizer)


@pytest.mark.parametrize("seed_policy,seeds", [("manuscript_native_three", [41,42,43]),
                                              ("manuscript_native_five", [40,41,42,43,44])])
def test_seed_average_precedes_paired_group_bootstrap_and_retains_failures(seed_policy, seeds):
    from Libs.utils.evaluation.native_comparator_evaluation import paired_seed_group_summary
    policy={"seed_policy":seed_policy,"training_seeds":seeds,"bootstrap_replicates":2000,"bootstrap_seed":41 if seed_policy == "manuscript_native_three" else 3407,
            "interval":[.025,.975],"order":"average_training_seeds_then_paired_independent_groups"}
    def report(seed,offsets):
        return {"method":"fixture","binding":{"normalization_id":"fixture","model_identity":{"method":"fixture",
                           "seed_binding":{"model_seed":seed,"selected_formal_seed":seed,"base_training_seed":seed,
                               "seed_policy":seed_policy,"allowed_training_seeds":seeds},"conditioning":{"seed":seed}}},
            "declared_group_count":2,"groups":[
                {"group_uid":uid,"status":"defined","metrics":{"crps":value}}
                for uid,value in zip(("a","b"),offsets)]}
    left=[report(seed,[seed-40,10+seed-40]) for seed in policy["training_seeds"]]
    right=[report(seed,[seed-40,seed-40]) for seed in policy["training_seeds"]]
    result=paired_seed_group_summary(left,right,["a","b"],"crps",policy)
    assert result["mean_difference"]==5.
    assert result["paired_group_interval"]==[0.,10.]
    assert [row["seed_averaged_paired_difference"] for row in result["groups"]]==[0.,10.]
    mixed=copy.deepcopy(left)
    mixed[0]["method"]="another_method"
    with pytest.raises(ValueError,match="one declared"):
        paired_seed_group_summary(mixed,right,["a","b"],"crps",policy)
    incompatible=copy.deepcopy(right)
    incompatible[0]["binding"]["normalization_id"]="different_fixture"
    with pytest.raises(ValueError,match="same fitted"):
        paired_seed_group_summary(left,incompatible,["a","b"],"crps",policy)
    changed=copy.deepcopy(left)
    changed[0]["groups"][0].update(status="failed",metrics={})
    missing=paired_seed_group_summary(changed,right,["a","b"],"crps",policy)
    assert missing["declared_group_count"]==2 and missing["defined_group_count"]==1
    assert missing["status"]=="incomplete" and missing["paired_group_interval"] is None
    with pytest.raises(ValueError,match="every declared"):
        paired_seed_group_summary(left[:-1],right,["a","b"],"crps",policy)
    with pytest.raises(ValueError,match="denominator"):
        trimmed=copy.deepcopy(left)
        trimmed[0]["groups"]=trimmed[0]["groups"][:1]
        paired_seed_group_summary(trimmed,right,["a","b"],"crps",policy)


def test_paper_mamba_scan_exact_zoh_and_bandlimited_constant_gradients():
    from Libs.models.modules.native_mamba_operator import selective_zoh_scan,PeriodicSincResampling
    values=torch.ones(1,4,1,dtype=torch.float64,requires_grad=True)
    delta=torch.full_like(values,.2)
    read=torch.ones(1,4,1,dtype=torch.float64)
    rates=torch.tensor([[-2.]],dtype=torch.float64,requires_grad=True)
    actual=selective_zoh_scan(values,delta,read,read,rates)
    time=torch.arange(1,5,dtype=torch.float64)*.2
    expected=(1-torch.exp(-2*time))/2
    torch.testing.assert_close(actual.flatten(),expected,rtol=1e-12,atol=1e-12)
    actual.sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum()>0
    assert torch.isfinite(rates.grad).all() and rates.grad.abs().sum()>0
    resampler=PeriodicSincResampling(2,6,8.6).double()
    constant=torch.ones(1,2,8,8,dtype=torch.float64,requires_grad=True)
    restored=resampler.down(resampler.up(constant))
    torch.testing.assert_close(restored,constant,rtol=2e-3,atol=2e-3)
    restored.sum().backward()
    assert torch.isfinite(constant.grad).all()
    with pytest.raises(ValueError,match="evolution"):
        selective_zoh_scan(values,delta,read,read,-rates)


@pytest.mark.parametrize("seed_policy,seeds", [("manuscript_native_three", [41,42,43]),
                                              ("manuscript_native_five", [40,41,42,43,44])])
def test_summary_cli_preserves_declared_partitions_and_failed_group_identity(tmp_path,monkeypatch,seed_policy,seeds):
    import json
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_comparator_evaluation as evaluation
    policy={"seed_policy":seed_policy,"training_seeds":seeds,"bootstrap_replicates":2000,"bootstrap_seed":41 if seed_policy == "manuscript_native_three" else 3407,
            "interval":[.025,.975],"order":"average_training_seeds_then_paired_independent_groups"}
    declarations=[{"sample_uid":uid,"role":"validation","partition_uid":partition}
                  for partition in ("id","ood") for uid in (partition+"a",partition+"b")]
    monkeypatch.setattr(evaluation,"group_definitions",lambda config:declarations)
    monkeypatch.setattr(cli,"load_config",lambda path:{"manuscript_native_statistics":policy})
    paths=[]
    for side in ("left","right"):
        group=[]
        for seed in policy["training_seeds"]:
            report={"method":side,"binding":{"normalization_id":"fixture","model_identity":{"method":side,
                "seed_binding":{"model_seed":seed,"selected_formal_seed":seed,"base_training_seed":seed,
                               "seed_policy":seed_policy,"allowed_training_seeds":seeds},"conditioning":{"seed":seed}}},
                "declared_group_count":4,"groups":[{"group_uid":row["sample_uid"],"status":"defined",
                    "metrics":{"crps":float(index if side=="left" else 0.)}} for index,row in enumerate(declarations)]}
            path=tmp_path/(side+str(seed)+".json")
            path.write_text(json.dumps(report));group.append(str(path))
        paths.append(group)
    monkeypatch.setattr(sys,"argv",["bridge","--protocol-scope","manuscript-native",
        "--command","summarize_baselines","--baseline-name","fixture_pair",
        "--output-dir",str(tmp_path/"summary"),"--comparison-left",*paths[0],"--comparison-right",*paths[1]])
    assert cli.main()==0
    result=json.loads((tmp_path/"summary/native_comparator_pairwise_summary.json").read_text())
    assert result["declared_group_count"]==4 and len(result["partitions"])==2
    assert [row["mean_difference"] for row in result["partitions"]]==[.5,2.5]
    class BrokenGroups:
        group_uids=["ida","idb"]
        def __len__(self):return 2
        def __getitem__(self,index):raise ValueError("fixture source admission failed")
    monkeypatch.setattr(evaluation,"resolve_training_seed",lambda config,require_selected:{"model_seed":40})
    broken_declarations={uid:{"partition_uid":"id","family_uid":"fixture"} for uid in BrokenGroups.group_uids}
    failed=evaluation.evaluate_native_comparator(
        {"native_controlled_comparators":{}},"codano",torch.nn.Linear(1,1),torch.nn.Linear(1,1),
        {},BrokenGroups(),broken_declarations,tmp_path/"failed")
    assert failed["declared_group_count"]==2 and failed["failed_or_unavailable_group_count"]==2
    assert [row["group_uid"] for row in failed["groups"]]==["ida","idb"]
    assert all(row["status"]=="failed" for row in failed["groups"])



def test_active_statistics_reject_historical_binding_and_bool_seed():
    from Libs.utils.evaluation.native_comparator_evaluation import paired_seed_group_summary
    policy={"seed_policy":"manuscript_native_three","training_seeds":[41,42,43],"bootstrap_replicates":2000,
        "bootstrap_seed":41,"interval":[.025,.975],"order":"average_training_seeds_then_paired_independent_groups"}
    reports=[{"method":"fixture","declared_group_count":2,"binding":{"normalization_id":"fixed",
        "model_identity":{"method":"fixture","conditioning":{"seed":seed},"seed_binding":{
            "model_seed":seed,"selected_formal_seed":seed,"base_training_seed":seed,
            "seed_policy":"manuscript_native_three","allowed_training_seeds":[41,42,43]}}},
        "groups":[{"group_uid":uid,"status":"defined","metrics":{"crps":float(seed)}} for uid in ("a","b")]} for seed in (41,42,43)]
    for fault in ("historical","bool","duplicate"):
        changed=copy.deepcopy(reports)
        if fault=="historical":changed[0]["binding"]["model_identity"]["seed_binding"]["seed_policy"]="manuscript_native_five"
        elif fault=="bool":changed[0]["binding"]["model_identity"]["seed_binding"]["model_seed"]=True
        else:changed[1]["binding"]["model_identity"]["seed_binding"]=changed[0]["binding"]["model_identity"]["seed_binding"]
        with pytest.raises(ValueError):paired_seed_group_summary(changed,reports,["a","b"],"crps",policy)
