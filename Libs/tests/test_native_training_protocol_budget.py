"""Optimizer/group-budget contracts without source generation or model campaigns."""
from types import SimpleNamespace
import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader

from Libs.utils.model.native_training_selection import (
    NativeTrainingSelection,validation_checkpoint_due,optimizer_learning_rate_ratio)
from Libs.utils.evaluation.manuscript_native_protocol import training_budget
from Libs.utils.model import native_scientific_source as source
from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
from Libs.utils.exp import native_comparator_training as comparator
from Libs.utils.data.raw_data_registry import load_config


def test_update_budget_fraction_visits_warmup_patience_and_terminal_snapshot():
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    expected={10:(400,7,1800,720000),25:(1000,16,788,787512),
              50:(2000,32,394,787536),100:(4000,63,200,800000)}
    for fraction,values in expected.items():
        actual=training_budget(config,fraction)
        assert tuple(actual[key] for key in ("selected_training_groups","optimizer_updates_per_full_epoch",
            "max_epochs","planned_group_visits_at_update_cap_without_early_stop"))==values
        assert actual["max_optimizer_updates"]==12600 and actual["minimum_optimizer_updates"]==10000
        assert actual["validation_interval_optimizer_steps"]==63
    policy=copy.deepcopy(config["native_scientific_training"])
    policy["validation"]["interval_optimizer_steps"]=63
    assert not validation_checkpoint_due(policy,62,0,1,epoch_end=False,budget_capped=False)
    assert validation_checkpoint_due(policy,63,0,1,epoch_end=False,budget_capped=False)
    assert not validation_checkpoint_due(policy,63,63,1,epoch_end=True,budget_capped=False)
    assert validation_checkpoint_due(policy,12600,12573,200,epoch_end=False,budget_capped=True)
    assert optimizer_learning_rate_ratio(0,policy)==pytest.approx(.001)
    assert optimizer_learning_rate_ratio(999,policy)==pytest.approx(1.)
    assert optimizer_learning_rate_ratio(1000,policy)==pytest.approx(1.)
    assert optimizer_learning_rate_ratio(12600,policy)==pytest.approx(.01)
    selection=NativeTrainingSelection(policy["selection"])
    for index in range(1,159):
        assert not selection.update(role="development",epoch=index,steps=index*63,
            training_loss=1.,development_loss=1.,checkpoint_path=str(index),elapsed_seconds=float(index))
    assert selection.update(role="development",epoch=159,steps=10017,
        training_loss=1.,development_loss=1.,checkpoint_path="159",elapsed_seconds=159.)
    assert selection.terminal=="DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW"


def test_main_pretrainer_selects_exact_predeclared_nested_groups(monkeypatch):
    rows=[{"sample_uid":str(index),"role":"train","calibration_reservation":None,
           "training_subset_percentages":[fraction for fraction in (10,25,50,100) if index<10*fraction//100]}
          for index in range(10)]
    monkeypatch.setattr(source,"group_definitions",lambda config,role_filter=None:rows)
    monkeypatch.setattr(pretrain,"NativeJointGroupDataset",lambda config,role,group_uids:group_uids)
    config={"native_scientific_training":{"enabled":True,"training_subset_percentage":25},
            "native_observable_bridge":{"nested_training_percentages":[10,25,50,100]}}
    assert pretrain.build_pretraining_dataset(config,None,direct_store=False,single_batch_check=False)=={"0","1"}
    config["native_scientific_training"]["training_subset_percentage"]=50
    assert pretrain.build_pretraining_dataset(config,None,direct_store=False,single_batch_check=False)=={"0","1","2","3","4"}


@pytest.mark.parametrize("kind,runtime_device",[("main","cpu"),("main","cuda:0"),("comparator","cpu")])
@pytest.mark.parametrize("patience",[1,2])
def test_effective_group_accumulation_partial_tail_and_update_cadence(kind,runtime_device,patience,tmp_path,monkeypatch):
    class Groups(torch.utils.data.Dataset):
        family_group_counts={"fixture":3}
        def __len__(self):return 3
        def __getitem__(self,index):return {"native_joint_group":index+1}
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    policy=config["native_scientific_training"]
    policy["selection"].update(max_epochs=4,minimum_epochs=1,minimum_steps=1,
        patience=patience,min_delta=1e6,max_steps=99,max_seconds=60)
    policy["validation"]["interval_optimizer_steps"]=2
    policy["learning_rate_schedule"]["warmup_steps"]=2
    training=config["screening_pipeline"]["screening_benchmark"]["training"]
    training["batch_size"]=2
    training["memory_policy"].update(effective_batch_size=2,micro_batch_size=1,gradient_accumulation_steps=2)
    model=torch.nn.Linear(1,1,bias=False)
    coefficients=[];gradients=[];rates=[]
    snapshots=[];first_training_state={};cuda_syncs=[]
    def step(model,optimizer,config):
        if kind=="main" and not gradients:
            early=json.loads((tmp_path/"microenvironment_pretraining_manifest.json").read_text())
            assert early["executed_steps"]==0 and early["executed_micro_batches"]==1
            assert early["first_microbatch_runtime"]["status"]=="COMPLETED"
        gradients.append(float(model.weight.grad))
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    dataset=Groups()
    if kind=="main":
        model.quantized_mddm=SimpleNamespace(training_sample_count=2,training_denoising_step_count=1,
            inference_sample_count=2,inference_denoising_step_count=1,sampler_seed=40)
        collate=lambda values:{"joint_groups":[row["native_joint_group"] for row in values]}
        monkeypatch.setattr(pretrain,"NativeJointGroupDataset",lambda *args:dataset)
        monkeypatch.setattr(pretrain,"data_loader_kwargs",lambda config:{"collate_fn":collate})
        original_write=pretrain.write_json
        def observe_manifest(path,payload):
            snapshots.append(copy.deepcopy(payload))
            if payload["executed_micro_batches"]==1 and payload["executed_steps"]==0:
                assert torch.equal(model.weight.grad,first_training_state["gradient"])
                assert torch.equal(torch.random.get_rng_state(),first_training_state["rng"])
            original_write(path,payload)
        monkeypatch.setattr(pretrain,"write_json",observe_manifest)
        if runtime_device.startswith("cuda"):
            # Exercise bookkeeping on CPU with explicit fake allocator counters;
            # this is not evidence of actual CUDA memory use or device timing.
            monkeypatch.setattr(torch.cuda,"synchronize",lambda device:cuda_syncs.append(str(device)))
            for name,value in (("memory_allocated",1024),("memory_reserved",2048),
                               ("max_memory_allocated",4096),("max_memory_reserved",8192)):
                monkeypatch.setattr(torch.cuda,name,lambda device,value=value:value)
        else:
            def forbidden_cuda(*args,**kwargs):
                raise AssertionError("CPU evidence must not query CUDA")
            for name in ("synchronize","memory_allocated","memory_reserved", "max_memory_allocated","max_memory_reserved"):
                monkeypatch.setattr(torch.cuda,name,forbidden_cuda)
        def forbidden_peak_reset(*args,**kwargs):
            raise AssertionError("First microbatch evidence must not reset peak counters")
        monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",forbidden_peak_reset)
        def group_batch(model,groups,config,device,memory_policy,*,backward,learned_teacher=None):
            if not backward:return {"total_loss":1.},{}
            assert model.training
            coefficient=float(groups[0]);coefficients.append(coefficient)
            loss=model.weight.sum()*coefficient
            (loss/memory_policy["gradient_accumulation_steps"]).backward()
            if len(coefficients)==1:
                first_training_state.update(gradient=model.weight.grad.clone(),rng=torch.random.get_rng_state().clone())
            return {"total_loss":float(loss.detach())},{}
        monkeypatch.setattr(pretrain,"scientific_group_batch",group_batch)
        monkeypatch.setattr(pretrain,"execute_optimizer_step",step)
        result=pretrain.fit_scientific_native_groups(model,torch.optim.AdamW(model.parameters(),lr=training["learning_rate"]),
            DataLoader(dataset,collate_fn=collate),config,torch.device(runtime_device),
            {"gradient_accumulation_steps":2},tmp_path,"fixture",{"model_seed":40})
    else:
        normalizer=SimpleNamespace(scale=torch.ones(1))
        monkeypatch.setattr(comparator,"resolve_training_seed",lambda config,require_selected:{"model_seed":40})
        monkeypatch.setattr(comparator,"make_comparator",lambda *args,belief=None:model)
        monkeypatch.setattr(comparator,"comparator_identity",lambda *args:{"method":"fixture"})
        monkeypatch.setattr(comparator,"prepare_comparator_group",
            lambda group,*args:(group,torch.zeros(1),torch.ones(1,dtype=torch.bool),torch.ones(1)))
        def loss(model,inputs,*args):
            assert model.training
            coefficients.append(float(inputs))
            return model.weight.sum()*inputs
        monkeypatch.setattr(comparator,"native_comparator_group_loss",loss)
        monkeypatch.setattr(comparator,"comparator_mean",lambda *args,**kwargs:(torch.ones(1),None))
        monkeypatch.setattr(comparator,"execute_optimizer_step",step)
        result=comparator.fit_native_comparator(SimpleNamespace(field_normalization=normalizer),
            "fixture",dataset,dataset,config,tmp_path,{},torch.device("cpu"))
    epochs=patience+1
    assert result["executed_steps"]==2*epochs and result["executed_group_visits"]==3*epochs
    expected=[value for index in range(epochs) for value in
              (sum(coefficients[3*index:3*index+2])/2,coefficients[3*index+2])]
    assert gradients==pytest.approx(expected)
    assert rates[:2]==pytest.approx([training["learning_rate"]/2,training["learning_rate"]])
    assert [row["steps"] for row in result["selection"]["history"]]==list(range(2,2*epochs+1,2))
    selection=result["selection"]
    assert selection["checkpoint_path_base"]=="manifest_directory"
    retained={tmp_path/selection[key]["checkpoint_path"] for key in ("best","last")}
    assert len(retained)==2 and all(path.is_file() for path in retained)
    assert set(tmp_path.glob("*.pt"))==retained
    assert result["selection"]["terminal"]=="DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW"
    if kind=="main":
        first=result["first_microbatch_runtime"]
        assert first["status"]=="COMPLETED" and first["device"]==runtime_device
        assert first["group_count"]==1 and first["forward_backward_wall_seconds"]>=0
        assert first["capture_point"]=="after_backward_before_optimizer"
        assert snapshots[0]["first_microbatch_runtime"]=={"status":"NOT_COMPLETED"}
        completed=[row for row in snapshots if row["first_microbatch_runtime"]["status"]=="COMPLETED"]
        assert all(row["first_microbatch_runtime"]==first for row in completed)
        assert sum(row["executed_micro_batches"]==1 and row["executed_steps"]==0 for row in snapshots)==1
        if runtime_device.startswith("cuda"):
            assert cuda_syncs==[runtime_device,runtime_device]
            assert first["cuda_memory"]=={"status":"RECORDED","allocated_bytes":1024,"reserved_bytes":2048,
                "peak_allocated_bytes":4096,"peak_reserved_bytes":8192,
                "peak_scope":"allocator_since_process_start_or_last_external_reset",
                "peak_counters_reset_by_measurement":False}
        else:
            assert first["cuda_memory"]=={"status":"NOT_APPLICABLE"}



@pytest.mark.parametrize("seed_policy,seeds", [
    ("manuscript_native_five", [40,41,42,43,44]),
    ("manuscript_native_three", [41,42,43]),
], ids=["historical_five", "canonical_three"])
def test_declared_seed_mainline_profiles_preserve_group_optimizer_and_memory_budget(seed_policy, seeds):
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    assert config["manuscript_native_protocol"]["training_seeds"] == [41,42,43]
    if seed_policy == "manuscript_native_five":
        # Explicit historical fixture; active configuration remains three-seed.
        config["manuscript_native_protocol"].update(seed_policy=seed_policy, training_seeds=seeds)
        config["manuscript_native_statistics"].update(seed_policy=seed_policy, training_seeds=seeds)
    for seed in seeds:
        actual,row=training_profile(config,"native_wam",seed)
        assert row["budget"]["selected_training_groups"]==4000
        assert row["budget"]["minimum_optimizer_updates"]==10000
        assert row["budget"]["max_optimizer_updates"]==12600
        assert row["budget"]["max_epochs"]==200
        assert len(source.selected_group_definitions(actual,"development"))==800
        memory=actual["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"]
        assert (memory["effective_batch_size"],memory["micro_batch_size"],memory["gradient_accumulation_steps"])==(64,1,64)
