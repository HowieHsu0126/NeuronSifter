from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from Libs.tests.test_expert_composition import ten_expert_case
from Libs.tests.test_native_field_normalization import joint_normalized_case
from Libs.tests.test_native_rollout_evaluation import development_case, run_cli
from Libs.tests.test_native_initial_state_persistence import change_future_programs
from Libs.tests.test_native_rollout_metrics import metric_case, fields, identity
from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.exp.evaluation import evaluate_screening_system as evaluation
from Libs.exp.evaluation.evaluate_rollout_prediction import parse_args, resolve_native_identity
from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.utils.model.screening_model_config import native_transolver_mean_config
from Libs.utils.model.native_direct_mean import predict_direct_mean, validate_direct_checkpoint, load_frozen_conditioner
from Libs.utils.model.joint_rollout import posterior_draw_indices
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import resolve_training_seed, load_experiment_family
from Libs.utils.evaluation.native_rollout_metrics import (
    metric_policy, metric_protocol, score_mean_only_episode, build_native_metric_payload,
    validate_native_metric_payload)

ROOT = Path(__file__).resolve().parents[2]
MEMORY_POLICY = {"gradient_accumulation_steps":1, "mixed_precision":{"enabled":False,"device_type":"cpu","dtype":"float32"}}


def direct_config(base, common_config, common_checkpoint, directory):
    config = copy.deepcopy(base)
    config["selected_experiment_family"] = "rollout_operator_transolver"
    architecture = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]
    canonical = load_config("Libs/configs/config_stack.yaml")
    template = canonical["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["transolver_ablation_template"]
    backbone = copy.deepcopy(template)
    backbone.update(operator_hidden_dim=12,operator_depth=2,attention_head_count=3,
                    physical_state_slice_count=4,operator_dropout_rate=0.,physics_mlp_multiplier=2)
    backbone["encoder_decoder_pretraining"].update(encoder_depth=1,decoder_depth=1,checkpoint_attention_blocks=False)
    backbone["mechanism_moe"].update(expert_hidden_multiplier=1)
    backbone["parameter_budget"].update(active_parameter_lower_bound=0,active_parameter_upper_bound=1000000)
    backbone["native_direct_mean"].update(enabled=True,posterior_context_dim=12,
        conditioning_config_path=str(common_config),conditioning_checkpoint_path=str(common_checkpoint))
    architecture["rollout_operator"]["transition_backbone"] = backbone
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = "transolver_direct_mean"
    train = config["screening_pipeline"]["screening_benchmark"]["training"]
    train.update(seed_policy="manuscript_native_five",random_seed=42)
    config["selected_formal_seed"] = 42
    config["logging_registry"].update(file_enabled=False,console_enabled=False,root_dir=str(directory/"logs"))
    return config


def setup(case, directory, point=False):
    common, registry, arrays, store, store_format, checkpoint = development_case(case,directory/"conditioning",point=point)
    cfg_path = directory/"conditioning"/"config.yaml"
    cfg_path.write_text(yaml.safe_dump(common,sort_keys=False))
    evaluation_config = direct_config(common,cfg_path,checkpoint,directory)
    train_base = copy.deepcopy(case[1])
    if point:
        # Keep a genuine independent training-role registry from the native producer.
        # Point role conversion is tested separately; this helper trains joint only.
        return evaluation_config,None,registry,arrays,store,store_format,checkpoint
    train_config = direct_config(train_base,cfg_path,checkpoint,directory)
    return evaluation_config,train_config,registry,arrays,store,store_format,checkpoint


def train_and_save(config, directory):
    torch.manual_seed(resolve_training_seed(config,require_selected=True)["model_seed"])
    model = training.MicroenvironmentPretrainingModel(config,False)
    dataset = training.MicroenvironmentTeacherFamilyDataset(config,"train")
    batch = training.collate_native_sources(dataset.__getitems__([0,1]))
    optimizer = training.create_adamw_optimizer(model,learning_rate=.001,weight_decay=0.)
    before = {key:value.detach().clone() for key,value in model.state_dict().items()}
    frozen = {key:value.clone() for key,value in model.conditioning["model"].state_dict().items()}
    metrics, report = training.run_pretraining_step(model,batch,config,torch.device("cpu"),MEMORY_POLICY)
    training.execute_optimizer_step(model,optimizer,config)
    assert np.isfinite(metrics["total_loss"]) and metrics["direct_normalized_mse"] > 0
    assert any(not torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    assert all(torch.equal(value,model.conditioning["model"].state_dict()[key]) for key,value in frozen.items())
    assert all(not p.requires_grad and p.grad is None for p in model.conditioning["model"].parameters())
    assert model.world_state_posterior is None
    assert len(model.operator.mechanism_moe.expert_names) == 10
    assert report["prediction_kind"] == "mean_only"
    directory.mkdir(exist_ok=True)
    checkpoint = training.save_training_checkpoint(model,optimizer,directory,"direct",1,1,"single_process",
        training_seed_binding=resolve_training_seed(config,require_selected=True))
    return model.eval(),checkpoint,batch


def test_real_training_checkpoint_evaluation_cli(ten_expert_case,tmp_path,monkeypatch):
    config,train_config,registry,arrays,store,fmt,common_checkpoint = setup(ten_expert_case,tmp_path)
    model,checkpoint,batch = train_and_save(train_config,tmp_path/"trained")
    saved = torch.load(checkpoint,map_location="cpu")
    assert saved["executed_steps"] == 1 and saved["optimizer_state_dict"]["state"]
    assert saved["native_direct_mean"]["training_source_registry"] != str(native.registry_path(config))
    assert saved["native_direct_mean"]["conditioning_identity"]["training_seed_binding"] != saved["training_seed_binding"]
    assert not any("conditioning" in key for key in saved["model_state_dict"])
    predictions,posteriors = [],[]
    predict,infer = evaluation.predict_native_episode,ADScreeningSystem.infer_world_state
    def watched_predict(*args,**kwargs):
        out=predict(*args,**kwargs); predictions.append(out); return out
    def watched_infer(*args,**kwargs):
        out=infer(*args,**kwargs); posteriors.append(out["joint_posterior"]); return out
    monkeypatch.setattr(evaluation,"predict_native_episode",watched_predict)
    monkeypatch.setattr(ADScreeningSystem,"infer_world_state",watched_infer)
    payload,_ = evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert len(posteriors) == 2
    assert payload["status"] == "incomplete" and payload["counts"]["partially_defined_episode_count"] == 2, payload
    assert payload["counts"]["episode_count"] == 2 and payload["counts"]["realized_draw_count"] == 0
    assert payload["metrics"]["normalized_rmse"]["count"] == 2
    assert all(payload["metrics"][name] == {"mean":None,"count":0}
               for name in ("crps","predictive_coverage","predictive_calibration_error"))
    for prediction,posterior in zip(predictions,posteriors):
        assert "field_trajectory_samples" not in prediction and "prediction_law" not in prediction
        assert prediction["predictive_law_draw_count"] == 0 and prediction["posterior_evaluation_draw_count"] == 2
        assert prediction["particle_conditional_means"].shape == (1,2,2,3,6,6,15)
        expected = posterior.support.initial_field[:,:,None].expand(-1,-1,2,-1,-1,-1)
        assert torch.equal(prediction["particle_conditional_means"][:,:,:,0],expected)
    first = [out["field_trajectory"].clone() for out in predictions]
    arrays["trajectory"] += 17  # t0 and future targets are independent of conditioning.
    for row in registry["teachers"]:
        row["content_hash"] += "-changed-teacher"
    write_native_microenvironment_store(store,config,registry,arrays,fmt)
    predictions.clear()
    changed,_ = evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert all(torch.equal(expected,out["field_trajectory"]) for expected,out in zip(first,predictions))
    assert changed["metrics"]["normalized_rmse"]["mean"] != payload["metrics"]["normalized_rmse"]["mean"]
    change_future_programs(config,registry,tmp_path/"changed-program")
    write_native_microenvironment_store(store,config,registry,arrays,fmt)
    predictions.clear()
    evaluation.evaluate_native_rollout_prediction(config,checkpoint)
    assert any(not torch.equal(expected,out["field_trajectory"]) for expected,out in zip(first,predictions))
    result,output = run_cli(config,checkpoint,tmp_path)
    assert result.returncode == 2,result.stdout+result.stderr
    cli=json.loads((output/"metrics.json").read_text())
    assert cli["status"] == "incomplete" and cli["metrics"]["normalized_rmse"]["count"] == 2
    assert json.loads((output/"artifact_manifest.json").read_text())["postrun_status"] == "incomplete"


def test_shared_prepared_posterior_and_physical_time(ten_expert_case,tmp_path):
    config,train_config,*_ = setup(ten_expert_case,tmp_path)
    model,checkpoint,_ = train_and_save(train_config,tmp_path/"trained")
    registry=native.load_registry(config)
    records=native.select_records(registry,config,"development",["sample-a"],["regimen-a","regimen-b"])
    arrays,_=native.load_records(registry,records,config,include_teacher=False)
    common=model.conditioning["model"]
    prepared=evaluation.prepare_native_episode(common,records,config,arrays)
    posterior=prepared[1]["joint_posterior"]
    sampler=common.field_rollout_operator.quantized_mddm_rollout_operator
    out=predict_direct_mean(model,prepared[1],sampler)
    assert torch.equal(out["particle_draw_indices"],posterior_draw_indices(posterior,sampler,2))
    assert torch.equal(out["particle_weights"],posterior.weights)
    contexts=native.joint_particle_contexts(posterior,prepared[1]["particle_programs"],prepared[1]["regimen_uids"])
    captures=[]
    handle=model.operator.register_forward_pre_hook(lambda module,args,kwargs:captures.append(kwargs["physical_time"].detach().clone()),with_kwargs=True)
    before=model(None,None,diffusion_batch=contexts[0])
    changed={key:value.clone() for key,value in contexts[0].items()}
    changed["physical_time"][:,:,1] *= .7
    after=model(None,None,diffusion_batch=changed)
    handle.remove()
    assert torch.equal(captures[0],contexts[0]["physical_time"].reshape(2,3,36,1))
    assert not torch.allclose(before["field_trajectory"][:,:,1:],after["field_trajectory"][:,:,1:])
    assert torch.equal(before["field_trajectory"][:,:,0],after["field_trajectory"][:,:,0])
    # Native and direct calls accept the exact same prepared object and indices.
    common_config=copy.deepcopy(config)
    common_config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"]="native_diffusion"
    checkpoint_payload=torch.load(model.conditioning["identity"]["checkpoint_path"],map_location="cpu")
    features,_,_=evaluation._native_training_features(model.conditioning["config"],
        Path(model.conditioning["identity"]["checkpoint_path"]),checkpoint_payload,
        model.conditioning["identity"]["training_seed_binding"],["regimen-a","regimen-b"])
    with torch.no_grad():
        native_out=evaluation.predict_native_episode(common,records,common_config,arrays,features,prepared=prepared)
        direct_out=evaluation.predict_native_episode(common,records,config,arrays,features,prepared=prepared,direct_predictor=model)
    assert native_out["posterior_uid"] == direct_out["posterior_uid"] == posterior.posterior_uid
    assert torch.equal(native_out["particle_draw_indices"],direct_out["particle_draw_indices"])


def test_direct_weighted_loss_precision_and_zero_confidence(ten_expert_case,tmp_path):
    _,config,*_=setup(ten_expert_case,tmp_path)
    model=training.MicroenvironmentPretrainingModel(config,False)
    target=torch.randn(2,3,6,6,15)
    prediction=torch.randn(1,2,3,6,6,15,requires_grad=True)
    batch={"trajectory":target,"uncertainty":torch.tensor([[.5]*15,[2.]*15]),
           "teacher_confidence":torch.tensor([.25,.75]),"teacher_loss_weight":torch.tensor([.5,1.])}
    output={"normalized_trajectory":prediction,"router_diagnostics":{"router_regularization_loss":prediction.sum()*0}}
    loss,_=training.direct_mean_objective(output,batch,model.field_normalization,config)
    per=(prediction[0,:,1:]-model.field_normalization.normalize(target[:,1:])).square().flatten(1).mean(1)
    expected=(per*torch.tensor([2.,.5])*torch.tensor([.125,.75])).sum()/.875
    torch.testing.assert_close(loss,expected)
    expected_grad=torch.autograd.grad(expected,prediction,retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(prediction.grad,expected_grad)
    output["normalized_trajectory"]=prediction.detach().requires_grad_(True)
    output["router_diagnostics"]["router_regularization_loss"]=output["normalized_trajectory"].sum()*0
    batch["teacher_confidence"].zero_()
    loss,_=training.direct_mean_objective(output,batch,model.field_normalization,config)
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(output["normalized_trajectory"].grad) == 0


def test_checkpoint_identity_rejections(ten_expert_case,tmp_path):
    config,train_config,*_=setup(ten_expert_case,tmp_path)
    model,checkpoint,_=train_and_save(train_config,tmp_path/"trained")
    original=torch.load(checkpoint,map_location="cpu")
    for mutation in ("legacy","direct_seed","conditioning_seed","profile","context_width"):
        payload,selected=copy.deepcopy(original),copy.deepcopy(config)
        if mutation=="legacy":
            payload.pop("native_direct_mean")
        elif mutation=="direct_seed":
            payload["training_seed_binding"]["model_seed"] += 1
        elif mutation=="conditioning_seed":
            payload["native_direct_mean"]["conditioning_identity"]["training_seed_binding"]["model_seed"] += 1
        elif mutation=="profile":
            payload["native_direct_mean"]["conditioning_identity"]["selected_experiment_family"]="rollout_operator_reversed_experts"
        else:
            selected["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["native_direct_mean"]["posterior_context_dim"]=13
        with pytest.raises(ValueError):
            validate_direct_checkpoint(payload,selected,model.conditioning)


def test_mean_only_metric_counts_mask_and_validation(metric_case):
    config,grid,norm,_,_=metric_case
    config=copy.deepcopy(config)
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"]="transolver_direct_mean"
    protocol=metric_protocol(metric_policy(config),[.05,.5,.95],norm.identity,draw_semantics="mean_only")
    x,y,mask=fields(metric_case,[1.],3.,actions=2)
    first=score_mean_only_episode(x[0],y,mask,identity(actions=2),norm,grid,protocol)
    second_identity=dict(identity(actions=2),episode_uid="masked",sample_uid="masked",family_uid="masked-family")
    second=score_mean_only_episode(x[0],y,torch.zeros_like(mask),second_identity,norm,grid,protocol)
    payload=build_native_metric_payload([first,second],protocol,{},config)
    assert payload["metrics"]["normalized_rmse"] == {"mean":2.,"count":1}
    assert payload["counts"]["episode_count"] == 2 and payload["counts"]["undefined_episode_count"] == 1
    assert payload["status"] == "incomplete" and payload["counts"]["realized_draw_count"] == 0
    first["metrics"]["crps"]=2.
    with pytest.raises(ValueError,match="Mean-only"):
        validate_native_metric_payload(payload,config)


def test_full_reference_legacy_and_seed_config(ten_expert_case,tmp_path):
    canonical=load_config("Libs/configs/config_stack.yaml")
    templates=canonical["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    template=templates["transolver_ablation_template"]
    assert (template["operator_hidden_dim"],template["attention_head_count"],template["physical_state_slice_count"]) == (1024,16,128)
    assert template["encoder_decoder_pretraining"]["encoder_depth"] == 12 and template["encoder_decoder_pretraining"]["decoder_depth"] == 4
    assert len(template["mechanism_moe"]["expert_names"]) == 10
    config,_,*_=setup(ten_expert_case,tmp_path)
    policy=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["native_direct_mean"]
    policy["enabled"]=False
    config["native_diffusion_data"]["registry_path"]=""
    assert not native.native_enabled(config) and not native_transolver_mean_config(config)
    model=training.MicroenvironmentPretrainingModel(config,False)
    state=torch.randn(1,2,36,15)
    out=model(state,training.node_coordinates(6,6,torch.device("cpu")).expand(1,2,-1,-1))
    assert out["state_prediction"].shape == state.shape and "prediction_kind" not in out
    assert load_experiment_family("rollout_operator_transolver",canonical)["runtime"]["seed_values"] == [3407,42,26]
    canonical["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"]="manuscript_native_five"
    assert load_experiment_family("rollout_operator_transolver",canonical)["runtime"]["seed_values"] == [40,41,42,43,44]


def point_training_config(case, directory, config):
    train_config=copy.deepcopy(config)
    registry=copy.deepcopy(native.load_registry(case[1]))
    source=registry.pop("joint_sources")[0]
    assert all(row["split_name"]=="train" and row["source_holdout_role"]=="train" for row in registry["teachers"])
    registry["posterior_family"]="deterministic_compatibility"
    store=directory/"point-training-source"
    # The point schema has one program/posterior identity per action. Keep one
    # genuine training sample; the two-sample joint fixture intentionally shares
    # that conditional-program identity and cannot be relabeled as a point set.
    registry["teachers"]=registry["teachers"][:2]
    registry["programs"]=copy.deepcopy(source["particle_programs"]["records"][0][0])
    registry["observations"]=[]
    for bi,sample in enumerate(source["support"]["sample_uids"][:1]):
        first=source["particle_programs"]["records"][bi][0][0]
        registry["observations"].append({"posterior_uid":first["posterior_uid"],"sample_uid":sample,
            "source_uid":first["source_uid"],"origin":"observation_derived","observation_ref":"fixture-point-readout",
            "source_snapshot_ref":source["support"]["source_snapshot_ref"],"initial_time_hours":0,
            "store_path":str(store),"store_format":"zarr_v2","observation_index":bi,
            "screening_features":[.2,.4,.6,.8]})
    arrays={key:np.array(value[:2],copy=True) for key,value in case[-1].items()}
    arrays["initial_state"]=np.asarray(source["support"]["initial_field"],dtype=np.float32)[:1,0]
    arrays["posterior_context"]=np.zeros((1,6,6,12),dtype=np.float32)
    for key in ("occupancy","support","physical_time"):
        arrays[key]=np.asarray([row["physical_program"][key] for row in registry["programs"]],dtype=np.float32)
    for index,row in enumerate(registry["teachers"]):
        row.update(trajectory_store_path=str(store),store_format="zarr_v2",trajectory_index=index)
    for index,row in enumerate(registry["programs"]):
        row.update(store_path=str(store),store_format="zarr_v2",program_index=index)
    write_native_microenvironment_store(store,train_config,registry,arrays,"zarr_v2")
    train_config["native_diffusion_data"]["registry_path"]=str(store/".zattrs")
    return train_config


def test_point_source_mean_training_and_cli(ten_expert_case,tmp_path):
    config,_,development_registry,_,_,_,_=setup(ten_expert_case,tmp_path,point=True)
    train_config=point_training_config(ten_expert_case,tmp_path,config)
    model,checkpoint,_=train_and_save(train_config,tmp_path/"point-trained")
    result,output=run_cli(config,checkpoint,tmp_path)
    assert result.returncode==2,result.stdout+result.stderr
    payload=json.loads((output/"metrics.json").read_text())
    assert payload["metrics"]["normalized_rmse"]["count"]==1 and payload["status"]=="incomplete",payload
    provenance=payload["episodes"][0]["prediction_provenance"]
    assert provenance["posterior_evaluation_draw_count"]==provenance["predictive_law_draw_count"]==0
    assert provenance["prediction_kind"]=="mean_only" and provenance["prediction_law"] is None
    assert provenance["particle_draw_indices"] is None
    # Explicit seed/profile namespace separates the two checkpoint roles.
    args=parse_args(["--execution-mode","native","--config","unused"])
    resolve_native_identity(args,config)
    assert "mean_only_from_dynamics_mainline_ad_seed_41" in args.output_dir
    assert "mean_only" in args.run_id


def test_actual_pretraining_main_and_checkpoint_consumer(ten_expert_case,tmp_path):
    config,train_config,*_=setup(ten_expert_case,tmp_path)
    artifact=tmp_path/"pretraining-main"
    train_config["logging_registry"]["console_enabled"]=True
    benchmark=train_config["screening_pipeline"]["screening_benchmark"]
    benchmark["artifact_root"]=str(artifact)
    train=benchmark["training"]
    train["batch_size"]=2
    train["data_loader"].update(num_workers=0,persistent_workers=False,pin_memory=False)
    train["memory_policy"].update(micro_batch_size=2,effective_batch_size=2,gradient_accumulation_steps=1)
    train["memory_policy"]["mixed_precision"]["enabled"]=False
    train["zarr_loader"].update(pretraining_epoch_count=1,pretraining_batches_per_epoch=1,
                                checkpoint_every_epoch=True,formal_artifact_subdir_template="")
    paths=train_config["paper_grade_full_suite"]["output_paths"]
    paths["runtime_efficiency_profile_json"]=str(tmp_path/"runtime.json")
    paths["runtime_efficiency_profile_md"]=str(tmp_path/"runtime.md")
    cfg=tmp_path/"pretraining.yaml"
    cfg.write_text(yaml.safe_dump(train_config,sort_keys=False))
    result=subprocess.run([sys.executable,str(ROOT/"Libs/exp/training/pretrain_microenvironment_world_model.py"),
        "--config",str(cfg),"--device","cpu"],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    saved=list(artifact.glob("microenvironment_pretraining*_epoch_1.pt"))
    assert len(saved)==1
    checkpoint=torch.load(saved[0],map_location="cpu")
    assert checkpoint["executed_steps"]==1 and checkpoint["optimizer_state_dict"]["state"]
    manifest=json.loads((artifact/"microenvironment_pretraining_manifest.json").read_text())
    assert manifest["executed_steps"]==1 and manifest["direct_normalized_mse"]>0
    assert manifest["data_loader"]["collate_fn"]=="collate_native_sources"
    assert manifest["native_direct_mean"]["conditioning_identity"]["training_seed_binding"]["selected_formal_seed"]==41
    cli,output=run_cli(config,saved[0],tmp_path)
    assert cli.returncode==2,cli.stdout+cli.stderr
    payload=json.loads((output/"metrics.json").read_text())
    assert payload["status"]=="incomplete" and payload["metrics"]["normalized_rmse"]["count"]==2,payload


def test_default_namespace_retains_legacy_conditioning_seed(tmp_path):
    common=load_config("Libs/configs/config_stack.yaml")
    common["selected_experiment_family"]="dynamics_mainline_ad"
    common["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"]="legacy_formal"
    outputs=[]
    for seed in (3407,26):
        common["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]=seed
        path=tmp_path/f"common-{seed}.yaml"
        path.write_text(yaml.safe_dump(common,sort_keys=False))
        config=direct_config(common,path,tmp_path/"unused.pt",tmp_path)
        args=parse_args(["--execution-mode","native","--config","unused"])
        resolve_native_identity(args,config)
        assert f"mean_only_from_dynamics_mainline_ad_seed_{seed}" in args.output_dir
        outputs.append((args.output_dir,args.run_id))
    assert outputs[0][0]!=outputs[1][0] and outputs[0][1]!=outputs[1][1]


def test_original_conditioning_profile_is_required_before_checkpoint_read(tmp_path,monkeypatch):
    common=load_config("Libs/configs/config_stack.yaml")
    common.pop("selected_experiment_family",None)
    path=tmp_path/"missing-profile.yaml"
    path.write_text(yaml.safe_dump(common,sort_keys=False))
    config=direct_config(common,path,tmp_path/"missing.pt",tmp_path)
    def forbidden(*args,**kwargs):
        raise AssertionError("checkpoint consumed before the original profile was bound")
    monkeypatch.setattr(evaluation,"load_native_conditioning_checkpoint",forbidden)
    with pytest.raises(ValueError,match="original unambiguous profile"):
        load_frozen_conditioner(config)
