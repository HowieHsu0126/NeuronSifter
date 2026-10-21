from __future__ import annotations

import copy
import math

import pytest
import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
from torch import nn

from Libs.models.modules.native_treatment_diffusion import (
    ChronologicalTreatmentAutoencoder, VelocityUNet2d,
    zero_terminal_cosine_schedule, reverse_diffusion_indices,
    velocity_ancestral_step, update_exponential_average,
)


def test_zero_terminal_cosine_and_explicit_matched_inference_clock():
    schedule=zero_terminal_cosine_schedule(1000)
    assert schedule.shape==(1000,2) and torch.isfinite(schedule).all()
    assert schedule[-1,0]==0 and schedule[-1,1]==1
    assert torch.allclose(schedule.square().sum(-1),torch.ones(1000,dtype=torch.float64),atol=1e-14)
    assert (schedule[1:,0]<schedule[:-1,0]).all()
    indices=reverse_diffusion_indices(1000,100)
    assert len(indices)==len(set(indices))==100
    assert indices[0]==999 and indices[-1]==0
    assert all(left>right for left,right in zip(indices,indices[1:]))
    with pytest.raises(ValueError,match="explicit"):
        reverse_diffusion_indices(1000,1001)


@pytest.mark.parametrize("alpha",[0.,.2,.8])
def test_velocity_oracle_reconstructs_the_clean_field_at_the_final_transition(alpha):
    generator=torch.Generator().manual_seed(15)
    clean=torch.randn(2,3,6,6,dtype=torch.float64,generator=generator)
    noise=torch.randn(clean.shape,dtype=torch.float64,generator=generator)
    sigma=math.sqrt(1-alpha*alpha)
    noisy=alpha*clean+sigma*noise
    velocity=alpha*noise-sigma*clean
    result=velocity_ancestral_step(noisy,velocity,alpha,1.,noise)
    assert torch.allclose(result,clean,atol=1e-13,rtol=1e-13)


def test_history_prefixes_ignore_future_actions_and_preserve_chronological_order():
    torch.manual_seed(17)
    model=ChronologicalTreatmentAutoencoder(3,16,8,5,4).eval()
    history=torch.randn(2,5,3)
    original=history.clone()
    before=model.prefix_embeddings(history)
    changed=history.clone();changed[:,2:]+=10
    after=model.prefix_embeddings(changed)
    assert torch.equal(before[:,:2],after[:,:2])
    assert not torch.allclose(before[:,2:],after[:,2:])
    valid=torch.ones(2,5,dtype=torch.bool)
    _,forward=model.encode(history,valid)
    _,reverse=model.encode(history.flip(1),valid)
    assert not torch.allclose(forward,reverse,atol=1e-7)
    assert torch.equal(history,original)
    hole=valid.clone();hole[0,1]=False
    with pytest.raises(ValueError,match="chronological"):
        model.encode(history,hole)


def test_history_reconstruction_and_infonce_have_actual_encoder_gradients():
    torch.manual_seed(18)
    model=ChronologicalTreatmentAutoencoder(3,16,8,5,4)
    clean=torch.randn(3,5,3)
    valid=torch.tensor([[True]*5,[True]*3+[False]*2,[True]*4+[False]])
    first=clean+.01*torch.randn_like(clean)
    second=clean+.01*torch.randn_like(clean)
    clean_before=clean.clone()
    result=model.reconstruction_contrastive_loss(clean,first,second,valid)
    assert result["reconstruction_mse"]>0 and result["contrastive_infonce"]>0
    assert torch.allclose(result["loss"],result["reconstruction_mse"]+.1*result["contrastive_infonce"])
    result["loss"].backward()
    assert model.positions.grad is not None and model.positions.grad.abs().sum()>0
    for layer in model.transformer.layers:
        assert layer.self_attn.in_proj_weight.grad is not None
        assert torch.isfinite(layer.self_attn.in_proj_weight.grad).all()
    assert model.reconstruction[-1].weight.grad.abs().sum()>0
    assert torch.equal(clean,clean_before)


def test_native_size_velocity_unet_uses_baseline_and_adagn_at_all_four_levels():
    torch.manual_seed(20)
    model=VelocityUNet2d(4,[8,16,32,32],16,4)
    noisy=torch.randn(2,4,6,6,requires_grad=True)
    baseline=torch.randn(2,4,6,6,requires_grad=True)
    condition=torch.randn(2,16,requires_grad=True)
    prediction=model(noisy,baseline,condition)
    assert prediction.shape==noisy.shape and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert baseline.grad.abs().sum()>0 and condition.grad.abs().sum()>0
    assert len(model.downsamples)==len(model.up_blocks)==4
    assert all(block.modulation.weight.grad is not None for blocks in model.down_blocks for block in blocks)
    assert not any(isinstance(module,nn.MultiheadAttention) for module in model.modules())
    with pytest.raises(ValueError,match="aligned"):
        model(noisy,baseline[:,:,:5],condition)


def test_ema_preserves_online_weights_and_copies_physical_buffers():
    torch.manual_seed(23)
    online=nn.Linear(2,2)
    online.register_buffer("schedule",torch.tensor([3.]))
    average=copy.deepcopy(online).requires_grad_(False)
    with torch.no_grad():
        for value in online.parameters():value.fill_(2.)
        for value in average.parameters():value.zero_()
        average.schedule.zero_()
    source={name:value.clone() for name,value in online.state_dict().items()}
    update_exponential_average(average,online,.999)
    assert all(torch.allclose(value,torch.full_like(value,.002),atol=1e-8) for value in average.parameters())
    assert average.schedule.item()==3
    assert all(torch.equal(online.state_dict()[name],value) for name,value in source.items())
    with pytest.raises(ValueError,match="differ"):
        update_exponential_average(online,online,.999)


def _treatment_settings():
    return {"training_diffusion_steps":1000,"ema_decay":.999,
        "schedule":"zero_terminal_squared_cosine","loss":"supported_native_velocity_mae",
        "temporal_law":"independent_query_noise_given_tuple_and_prefix",
        "autoencoder_loss_weight":1.,"autoencoder_view_drop_probability":.1,
        "query_batch_size":512,"encoder_hidden_dim":16,"history_latent_dim":8,"encoder_heads":4,
        "conditioning_dim":16,"unet_widths":[8,16,32,32],"group_norm_groups":4}


def _treatment_native_fixture():
    from types import SimpleNamespace
    from Libs.models.modules.native_treatment_diffusion import HISTORY_BLOCKS
    torch.manual_seed(57)
    b,j,a,t,r,c,q=1,2,3,4,6,6,4
    physical=torch.randn(b,j,r,c,q)
    initial=physical[:,:,None].expand(b,j,a,r,c,q).reshape(b*j*a,r,c,q).clone()
    blocks={};parts=[];offset=0
    def append(name,value):
        nonlocal offset
        value=value.reshape(b*j*a,-1)
        blocks[name]=(offset,offset+value.shape[1]);offset+=value.shape[1];parts.append(value)
    append("initial_field",initial)
    append("source_nuisance",torch.randn(b*j*a,3))
    for name in HISTORY_BLOCKS:append(name,torch.randn(b*j*a,t,2))
    append("physical_domain",torch.ones(b*j*a,1))
    grid={"time_hours":[0.,1.,2.,4.],"region_id":[str(i) for i in range(r)],
          "species_id":[str(i) for i in range(q)]}
    inputs=SimpleNamespace(initial=initial,condition=torch.cat(parts,-1),blocks=blocks,
        validity=torch.ones(b*j*a,t,r,c,q,dtype=torch.bool),group_shape=(b,j,a),
        grid=grid,weights=torch.tensor([[.25,.75]]),normalization_id="test_bound",
        posterior=SimpleNamespace(support=SimpleNamespace(initial_field=physical)),
        posterior_uid="p",particle_uids=(("p0","p1"),),regimen_uids=("a","b","c"),sample_uids=("s",))
    config={"native_controlled_comparators":{"conditioning":{
        "maximum_times":t,"program_reference_scales":{"physical_time":1.}}}}
    normalizer=SimpleNamespace(identity="test_bound",assert_domain=lambda grid:None,
                              inverse=lambda values,mask:torch.where(mask,values,0))
    return inputs,config,normalizer


def test_native_treatment_wrapper_runs_100_steps_and_future_programs_leave_prefix_unchanged():
    from Libs.models.modules.native_treatment_diffusion import NativeTreatmentDiffusion,HISTORY_BLOCKS
    inputs,config,normalizer=_treatment_native_fixture()
    model=NativeTreatmentDiffusion(_treatment_settings(),inputs,config).eval()
    original=inputs.condition.clone()
    calls=[]
    hook=model.ema["velocity"].register_forward_hook(lambda *args:calls.append(1))
    before=model.sample_native(inputs,normalizer,2,100,31)
    hook.remove()
    assert len(calls)==200==before["execution"]["velocity_network_calls"]
    assert before["execution"]["denoising_steps_per_query"]==100
    assert before["execution"]["training_diffusion_steps"]==1000
    assert before["particle_trajectory_samples"].shape==(2,1,2,3,4,6,6,4)
    changed=copy.copy(inputs);changed.condition=inputs.condition.clone()
    for name in HISTORY_BLOCKS:
        lo,hi=inputs.blocks[name]
        changed.condition[:,lo:hi].reshape(6,4,-1)[:,2:]+=5.
    after=model.sample_native(changed,normalizer,2,100,31)
    left,right=before["particle_trajectory_samples"],after["particle_trajectory_samples"]
    assert torch.equal(left[:,:,:,:,:2],right[:,:,:,:,:2])
    assert not torch.allclose(left[:,:,:,:,2:],right[:,:,:,:,2:])
    assert not torch.equal(left[0,:,:,:,1:],left[1,:,:,:,1:])
    expected=inputs.posterior.support.initial_field[:,:,None].expand(1,2,3,6,6,4)
    assert torch.equal(left[0,:,:,:,0],expected)
    assert torch.equal(before["particle_weights"],inputs.weights)
    assert torch.equal(inputs.condition,original)
    with pytest.raises(ValueError,match="bound normalization"):
        model.sample_native(inputs,type("Wrong",(),{"identity":"wrong"})(),2,100,31)


def test_native_treatment_loss_ema_and_single_history_status_preserve_inputs():
    from Libs.models.modules.native_treatment_diffusion import NativeTreatmentDiffusion,HISTORY_BLOCKS
    inputs,config,_=_treatment_native_fixture()
    model=NativeTreatmentDiffusion(_treatment_settings(),inputs,config)
    # Identical histories have no legitimate contrastive negatives.
    for name in HISTORY_BLOCKS:
        lo,hi=inputs.blocks[name]
        inputs.condition[:,lo:hi]=inputs.condition[0,lo:hi].clone()
    original=inputs.condition.clone()
    target=torch.randn_like(inputs.validity,dtype=torch.float32)
    weights=inputs.weights[:,:,None].expand(1,2,3).reshape(-1)
    loss=model.loss_native(inputs,target,inputs.validity,weights,torch.Generator().manual_seed(41))
    loss.backward()
    for name,module in model.online.items():
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()),name
    assert all(p.grad is None and not p.requires_grad for p in model.ema.parameters())
    assert model.last_training_diagnostics["contrastive_status"]=="unavailable_fewer_than_two_distinct_program_histories"
    assert model.last_training_diagnostics["contrastive_infonce"]==0.
    old={name:value.clone() for name,value in model.ema.state_dict().items()}
    optimizer=torch.optim.AdamW(model.online.parameters(),lr=.01)
    optimizer.step()
    source={name:value.clone() for name,value in model.online.state_dict().items()}
    model.update_ema()
    assert model.ema_updates.item()==1
    assert all(torch.equal(value,model.online.state_dict()[name]) for name,value in source.items())
    assert any(not torch.equal(value,model.ema.state_dict()[name]) for name,value in old.items())
    assert torch.equal(inputs.condition,original)
    malformed=target.clone();malformed[0,1,0,0,0]=float("nan")
    with pytest.raises(FloatingPointError,match="Nonfinite"):
        model.loss_native(inputs,malformed,inputs.validity,weights,torch.Generator().manual_seed(41))


def test_treatment_catalog_preserves_source_schedule_and_declares_native_mapping():
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import training_profile,command_catalog
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    selected,profile=training_profile(config,"treatment_history",41)
    reference=selected["native_controlled_comparators"]["references"]["treatment_history"]
    parameters=reference["hyperparameters"]
    assert parameters["training_diffusion_steps"]==1000
    assert parameters["unet_widths"]==[128,256,512,512]
    assert reference["source_default_inference_steps"]==250
    assert selected["native_scientific_training"]["predictive_budget"]["categorical_reverse_steps"]==100
    rows={row["command_uid"]:row for row in command_catalog(config,[profile])}
    uid=profile["profile_id"]
    assert uid+":fit" in rows and uid+":evaluate" in rows and uid+":calibrate" not in rows
    assert rows[uid+":evaluate"]["dependencies"]==[uid+":fit"]


def test_actual_tiny_treatment_cli_fit_ema_reload_and_full_original_clocks(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    from Libs.utils.data.raw_data_registry import load_config

    location=os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:pytest.skip("Requires retained artificial arrays and selected belief")
    fixture=Path(location)
    root=Path(__file__).resolve().parents[2]
    artifacts=Path(os.environ.get("NRNSIFT_TREATMENT_ARTIFACTS",str(tmp_path/"artifacts")))
    artifacts.mkdir(parents=True,exist_ok=False)
    reused=os.environ.get("NRNSIFT_TREATMENT_REUSE_ARTIFACTS")
    if reused:
        prior=Path(reused);config_path=prior/"treatment_evaluation_config.yaml";train_output=prior/"training"
    else:
        config=load_config(str(fixture/"config.yaml"))
        current=load_config(str(root/"Libs/configs/exp/manuscript_native_protocol.yaml"))
        reference=copy.deepcopy(current["native_controlled_comparators"]["references"]["treatment_history"])
        reference["hyperparameters"].update(_treatment_settings())
        config["native_controlled_comparators"]["references"]["treatment_history"]=reference
        declared=next(row for row in current["manuscript_native_protocol"]["methods"] if row["method_id"]=="treatment_history")
        config["manuscript_native_protocol"]["methods"]=[copy.deepcopy(declared) if row["method_id"]=="treatment_history" else row
            for row in config["manuscript_native_protocol"]["methods"]]
        if "native_profile_identity" in config:config["native_profile_identity"]["method"]="treatment_history"
        config["native_scientific_training"]["selection"].update(max_steps=2,max_epochs=1,minimum_steps=1,minimum_epochs=1)
        config["native_scientific_training"]["validation"].update(interval_optimizer_steps=1,denoising_step_count=100)
        config["native_scientific_training"]["predictive_budget"]["categorical_reverse_steps"]=100
        config_path=artifacts/"treatment_evaluation_config.yaml"
        config_path.write_text(yaml.safe_dump(config,sort_keys=False));train_output=artifacts/"training"
    config=load_config(str(config_path))
    common=[sys.executable,str(root/"Libs/exp/baselines/run_external_baseline_bridge.py"),
        "--protocol-scope","manuscript-native","--config",str(config_path),
        "--baseline-name","treatment_history","--conditioning-config",str(fixture/"config.yaml"),
        "--conditioning-manifest",str(fixture/"native_fit/manifest.json"),"--device","cpu"]
    train_exit=None
    if not reused:
        completed=subprocess.run(common+["--command","train_baseline","--output-dir",str(train_output)],
                                 cwd=root,text=True,capture_output=True)
        (artifacts/"train_cli.log").write_text(completed.stdout+completed.stderr)
        train_exit=completed.returncode
        assert train_exit==2,completed.stdout+completed.stderr
    manifest=train_output/"native_comparator_training_manifest.json"
    trained=json.loads(manifest.read_text())
    assert trained["executed_steps"]==1 and trained["ema_updates"]==1 and trained["failure"] is None
    assert trained["development_group_count"]==4 and trained["selection"]["best"]["steps"]==1
    assert trained["selection"]["terminal"]=="INCOMPLETE_EPOCH_CAP"
    assert trained["method_mapping"]["source_training_diffusion_steps"]==1000
    selected=training_checkpoint_path(manifest,trained["selection"],trained["selection"]["best"]["checkpoint_path"])
    checkpoint=torch.load(selected,map_location="cpu",weights_only=False)
    assert checkpoint["model_state_dict"]["ema_updates"].item()==1
    online=checkpoint["model_state_dict"]["online.velocity.output.2.weight"]
    average=checkpoint["model_state_dict"]["ema.velocity.output.2.weight"]
    assert not torch.equal(online,average)
    output=artifacts/"evaluation"
    evaluated=subprocess.run(common+["--command","evaluate_baseline","--output-dir",str(output),
        "--training-manifest",str(manifest),"--evaluation-role","validation"],cwd=root,text=True,capture_output=True)
    (artifacts/"evaluate_cli.log").write_text(evaluated.stdout+evaluated.stderr)
    assert evaluated.returncode==0,evaluated.stdout+evaluated.stderr
    report=json.loads((output/"native_comparator_evaluation.json").read_text())
    assert report["status"]=="complete" and report["declared_group_count"]==4
    assert report["binding"]["checkpoint_path"]==str(selected)
    assert sorted(row["sampling_execution"]["original_time_count"] for row in report["groups"])==[7,7,8,28]
    for row in report["groups"]:
        assert row["status"]=="defined"
        run=row["sampling_execution"]
        assert run["denoising_steps_per_query"]==100 and run["training_diffusion_steps"]==1000
        assert run["complete_particle_count"]==32 and run["paired_action_count"]==3 and run["ema_updates"]==1
        assert run["forecast_paths_per_action"]==64 and run["new_source_provider_calls"]==0
        assert torch.isfinite(torch.tensor(row["metrics"]["crps"]))
    receipt={"scope":"actual_tiny_treatment_fit_ema_strict_reload_and_full_native_clocks",
        "training_cli_executed_this_attempt":not bool(reused),"train_cli_exit_this_attempt":train_exit,
        "reused_training_artifacts":reused,"evaluate_cli_exit":evaluated.returncode,
        "training_manifest":str(manifest),"evaluation_report":report,
        "prior_optimizer_updates":trained["executed_steps"] if reused else 0,
        "additional_optimizer_updates":0 if reused else trained["executed_steps"],"source_provider_calls":0,
        "scientific_twin":"NOT_RUN","formal_qualification":False}
    Path(os.environ.get("NRNSIFT_TREATMENT_EVIDENCE",str(tmp_path/"result.json"))).write_text(json.dumps(receipt,indent=2))
