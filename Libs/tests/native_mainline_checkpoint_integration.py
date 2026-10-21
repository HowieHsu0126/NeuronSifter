"""Tiny native checkpoint chain against retained numeric software data; no producer calls."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from Libs.exp.baselines import run_external_baseline_bridge as cli
from Libs.exp.training import pretrain_microenvironment_world_model as training
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters,parameter_schema_from_config
from Libs.utils.data.native_diffusion_data import parse_joint_source, joint_particle_contexts, scalar_exposure_view, verify_scalar_exposure
from Libs.utils.model.native_comparator_inputs import build_native_comparator_inputs
from Libs.utils.data.native_observable_profile import family_grid,quantity_support,coefficient_support
from Libs.utils.evaluation import native_comparator_evaluation as evaluation
from Libs.utils.exp.native_comparator_training import bind_frozen_training_belief,load_frozen_native_belief
from Libs.utils.exp.experiment_manifests import resolve_training_seed


def run(origin:Path,output:Path,*,scale_context_widths=None):
    output.mkdir(parents=True,exist_ok=True)
    config=load_config(str(origin/"native_config.yaml"))
    current=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_controlled_comparators"]=copy.deepcopy(current["native_controlled_comparators"])
    config["native_scientific_training"]=copy.deepcopy(current["native_scientific_training"])
    config["selected_formal_seed"]=40
    for key in ("formal_seed_values","seeds"):config["paper_grade_full_suite"][key]=[40,41,42,43,44]
    config["paper_grade_baseline_registry"]["seed_values"]=[40,41,42,43,44]
    config["schemas"]["native_training_seed_binding"]=current["schemas"]["native_training_seed_binding"]
    benchmark=config["screening_pipeline"]["screening_benchmark"]
    benchmark["training"].update(hidden_dim=12,dropout_rate=0.,random_seed=40,seed_policy="manuscript_native_five")
    architecture=benchmark["model_architecture"]
    architecture["world_state_posterior"].update(physical_parameter_conditioning="source_declared_named_candidates",
        posterior_hidden_dim=12,posterior_dropout_rate=0.)
    backbone=architecture["rollout_operator"]["transition_backbone"]
    backbone.update(operator_hidden_dim=12,operator_depth=1,attention_head_count=3,mechanism_state_token_count=6,
                    operator_dropout_rate=0.)
    config["native_diffusion_data"]["posterior_context_width"]=12
    config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"].update(
        training_sample_count=2,training_denoising_step_count=1,inference_sample_count=2,inference_denoising_step_count=1)
    config["native_scientific_training"]["predictive_budget"].update(conditional_draws_per_particle=2,categorical_reverse_steps=1)
    config_path=output/"tiny_native_config.yaml";config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    binding=resolve_training_seed(config,require_selected=True)
    torch.manual_seed(40)
    common=training.MicroenvironmentPretrainingModel(config,False).eval()
    assert sum(p.numel() for p in common.parameters())<1000000
    optimizer=torch.optim.AdamW(common.parameters(),lr=1e-4)
    path=training.save_training_checkpoint(common,optimizer,output,"tiny_common",1,0,"single_process",
        training_seed_binding=binding,immutable=True)
    manifest=output/"tiny_common_selection.json"
    manifest.write_text(json.dumps({"training_seed_binding":binding,"native_scientific_profile":common.native_scientific_profile,"selection":{"selection_role":"development",
        "best":{"checkpoint_path":str(path)},"terminal":"SOFTWARE_FIXTURE_NOT_SCIENTIFIC_TRAINING"}}))
    receipt=json.loads((origin/"source_receipt.json").read_text());schema=parameter_schema_from_config(config)
    index,definition=next((i,row) for i,row in enumerate(receipt["groups"])
        if row["role"]=="train" and row["family_uid"]=="molecular_pharmacological_causal_world")
    raw=json.loads((origin/f"group_{index}_conditioning.json").read_text())
    producer=json.loads((origin/f"group_{index}_producer.json").read_text())
    raw=bind_candidate_parameters(raw,definition,producer,schema,record_ref=str(origin/f"group_{index}_producer.json"),
        producer_commit=receipt["source_identity"]["producer_commit"])
    support,_=parse_joint_source(raw);grid=family_grid(support.quantity_profile,support.source_family_uid)
    mask=torch.from_numpy(quantity_support(grid));items=[]
    for ai,case in enumerate(row for row in receipt["cases"] if row["group_index"]==index):
        with np.load(origin/case["raw_reference"]) as data:target=torch.from_numpy(data["trajectory"]).float()
        items.append({"trajectory":target,"quantity_support":mask,"uncertainty":torch.tensor(case["numerical"]["uncertainty"]),
            "teacher_confidence":torch.tensor(case["numerical"]["teacher_confidence"]),"teacher_loss_weight":torch.tensor(.5),
            "sample_uid":definition["sample_uid"],"source_family_uid":definition["family_uid"],
            "normalization_id":common.field_normalization.identity,"regimen_uid":raw["particle_programs"]["regimen_uids"][ai]})
    group={"joint_source":raw,"items":items}
    class Groups:
        family_group_counts={definition["family_uid"]:1}
        group_uids=[definition["sample_uid"]]
        def __len__(self):return 1
        def __getitem__(self,index):return {"native_joint_group":group}
    dataset=Groups();outcomes=[]
    if scale_context_widths is not None:
        assert "posterior_context_width" not in common.native_scientific_profile
        return check_scale_contexts(config,config_path,manifest,raw,group,binding,output,scale_context_widths)
    original_dataset=evaluation.declared_dataset;original_argv=sys.argv
    evaluation.declared_dataset=lambda *args,**kwargs:(dataset,{definition["sample_uid"]:definition})
    try:
        for method,action_policy in (("native_wam","occupancy_program"),("native_action_id","drug_id"),
                                      ("native_target_id","target_id"),("native_scalar_exposure","scalar_exposure")):
            selected=copy.deepcopy(config)
            selected["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"][
                "action_representation"]["active_policy"]=action_policy
            current_manifest=manifest
            selected_path=config_path
            if method!="native_wam":
                selected["native_scientific_training"]["family_scope"]=["molecular_pharmacological_causal_world","cns_long_horizon_world"]
                selected["native_scientific_training"]["conditioning"]={"mode":"frozen_selected_mainline",
                    "config_path":str(config_path),"manifest_path":str(manifest)}
                torch.manual_seed(40)
                model=training.MicroenvironmentPretrainingModel(selected,False)
                identity=bind_frozen_training_belief(model,selected,torch.device("cpu"))
                frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
                model.train()
                assert not model.world_state_posterior.training and model.operator.training
                assert all(not getattr(model,name).training for name in identity["loaded_component_names"])
                belief,_=load_frozen_native_belief(selected,config_path,manifest,torch.device("cpu"))
                assert torch.equal(model(None,None,joint_source=raw).weights,belief(raw).weights)
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
                if method=="native_action_id":
                    loss,_,_=training.scientific_joint_group_loss(model,group,selected,torch.device("cpu"))
                    loss.backward()
                    assert sum(p.grad.abs().sum() for p in model.operator.parameters() if p.grad is not None)>0
                    optimizer.step()
                    assert all(torch.equal(dict(model.named_parameters())[name],value) for name,value in frozen.items())
                path=training.save_training_checkpoint(model,optimizer,output,method,1,1 if method=="native_action_id" else 0,
                    "single_process",training_seed_binding=binding,immutable=True)
                current_manifest=output/(method+"_selection.json")
                current_manifest.write_text(json.dumps({"training_seed_binding":binding,"native_scientific_profile":model.native_scientific_profile,"selection":{"selection_role":"development",
                    "best":{"checkpoint_path":str(path)},"terminal":"SOFTWARE_FIXTURE_NOT_SCIENTIFIC_TRAINING"}}))
                selected_path=output/(method+"_config.yaml");selected_path.write_text(yaml.safe_dump(selected,sort_keys=False))
                wrong_scope=copy.deepcopy(selected)
                wrong_scope["native_scientific_training"]["family_scope"]=["molecular_pharmacological_causal_world"]
                try:evaluation.load_native_comparator(wrong_scope,method,config_path,manifest,current_manifest,dataset,torch.device("cpu"))
                except ValueError as error:assert "training-domain" in str(error)
                else:raise AssertionError("Same architecture checkpoint accepted the wrong declared domain")
                model.eval()
                posterior=belief(raw)
                inputs=build_native_comparator_inputs(raw,posterior,model.field_normalization,selected)
                contexts=joint_particle_contexts(posterior,inputs.programs,inputs.regimen_uids,action_policy=action_policy)
                context={key:value[0,0] for key,value in contexts[0].items()}
                sampler=model.quantized_mddm
                value,_=sampler.pack(torch.zeros_like(items[0]["trajectory"]),model.operator)
                hidden_mask=torch.ones_like(value,dtype=torch.bool)
                context["coefficient_support"]=coefficient_support(context["quantity_support"],model.operator.tokenizer.wavelet_level_count)
                captured=[]
                hook=model.operator.register_forward_pre_hook(lambda module,args:captured.append(args[1].detach().clone()))
                coords=training.node_coordinates(6,6,torch.device("cpu"))[0,0]
                def call(current):
                    with torch.no_grad():sampler.logits(value,hidden_mask,torch.tensor(.5),current,model.operator,model.node_adjacency,coords)
                try:
                    call(context)
                    changed={key:value.clone() for key,value in context.items()}
                    changed["occupancy"]=1-changed["occupancy"]
                    changed["support"]=1-changed["support"]
                    call(changed)
                    assert torch.equal(captured[0],captured[1]), "Chemical waveform leaked past the ablation"
                    changed={key:value.clone() for key,value in context.items()}
                    if action_policy=="scalar_exposure":
                        changed["scalar_exposure_auc"]+=context["scalar_exposure_site_support"]*.01
                        view=scalar_exposure_view(raw["particle_programs"]["records"])
                        record=view["records"][0][0][0]
                        assert record["target_site_domain"]["excluded_site_meaning"]=="outside_declared_target_program"
                        support_grid=np.asarray(record["site_support"])[...,0]
                        assert all((record["site_uids"][ri][ci] is None)==(not support_grid[ri,ci])
                                   for ri in range(6) for ci in range(6))
                        forged=copy.deepcopy(view);forged["records"][0][0][0]["site_support"][0][0][0]=True
                        try:verify_scalar_exposure(forged,view)
                        except ValueError:pass
                        else:raise AssertionError("Forged sparse scalar support accepted")
                    else:changed["action_identity"]+=.01
                    call(changed)
                    assert not torch.equal(captured[0],captured[2]), "Declared ablation signal did not reach the operator"
                finally:hook.remove()
            target=output/(method+"_evaluation")
            sys.argv=["run_external_baseline_bridge.py","--protocol-scope","manuscript-native","--config",str(selected_path),
                "--baseline-name",method,"--command","evaluate_baseline","--device","cpu","--conditioning-config",str(config_path),
                "--conditioning-manifest",str(manifest),"--training-manifest",str(current_manifest),"--output-dir",str(target)]
            assert cli.main()==0
            result=json.loads((target/"native_comparator_evaluation.json").read_text())
            assert result["status"]=="complete" and result["groups"][0]["status"]=="defined", result
            if method=="native_scalar_exposure":
                conditions=result["groups"][0]["action_conditioning"]
                assert len(conditions)==4 and all("source_site_support" in row for row in conditions)
            outcomes.append({"method":method,"status":result["status"],"group_count":len(result["groups"]),
                "action_policy":action_policy,"optimizer_updates":int(method=="native_action_id")})
        return {"methods":outcomes,"source_operations":0,"training_scope":"tiny_software_fixture",
            "common_checkpoint":"untrained_fixture_with_explicit_selection_fixture_metadata",
            "formal_qualification":False,"scientific_twin":"NOT_RUN"}
    finally:
        evaluation.declared_dataset=original_dataset;sys.argv=original_argv



def check_scale_contexts(config,config_path,manifest,raw,group,binding,output,widths):
    """Real native loss/update/reload/sample using one common frozen posterior."""
    from Libs.models.modules.native_field_comparators import NativeWorldActionComparator
    results=[]
    reference,identity=load_frozen_native_belief(config,config_path,manifest,torch.device("cpu"))
    original_posterior=reference(raw)
    common_width=original_posterior.particle_context.shape[-1]
    for width in widths:
        selected=copy.deepcopy(config)
        selected["native_scientific_training"]["conditioning"]={"mode":"frozen_selected_mainline",
            "config_path":str(config_path),"manifest_path":str(manifest)}
        backbone=selected["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        backbone.update(operator_hidden_dim=width,attention_head_count=2)
        torch.manual_seed(40)
        model=training.MicroenvironmentPretrainingModel(selected,False)
        assert sum(p.numel() for p in model.parameters())<1000000
        bound=bind_frozen_training_belief(model,selected,torch.device("cpu"))
        assert bound==identity
        assert model.quantized_mddm.context_projection.in_features==common_width
        assert model.quantized_mddm.context_projection.out_features==width
        assert model.native_scientific_profile["posterior_context_width"]==common_width
        model.train()
        before=model(None,None,joint_source=raw)
        assert torch.equal(before.weights,original_posterior.weights)
        assert before.posterior_uid==original_posterior.posterior_uid
        frozen={name:value.detach().clone() for name,value in model.named_parameters() if not value.requires_grad}
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        loss,metrics,_=training.scientific_joint_group_loss(model,group,selected,torch.device("cpu"))
        loss.backward()
        projection_gradient=float(model.quantized_mddm.context_projection.weight.grad.abs().sum())
        operator_gradient=float(sum(p.grad.abs().sum() for p in model.operator.parameters() if p.grad is not None))
        assert projection_gradient>0 and operator_gradient>0
        training.execute_optimizer_step(model,optimizer,selected)
        assert all(torch.equal(dict(model.named_parameters())[name],value) for name,value in frozen.items())
        after=model(None,None,joint_source=raw)
        assert torch.equal(after.weights,original_posterior.weights)
        assert after.posterior_uid==original_posterior.posterior_uid
        saved=training.save_training_checkpoint(model,optimizer,output,f"context_width_{width}",1,1,
            "single_process",training_seed_binding=binding,immutable=True)
        checkpoint=torch.load(saved,map_location="cpu",weights_only=False)
        loaded=training.MicroenvironmentPretrainingModel(selected,False)
        bind_frozen_training_belief(loaded,selected,torch.device("cpu"))
        assert checkpoint["native_scientific_profile"]==loaded.native_scientific_profile
        loaded.load_state_dict(checkpoint["model_state_dict"],strict=True)
        loaded.eval()
        inputs=build_native_comparator_inputs(raw,original_posterior,loaded.field_normalization,selected)
        law=NativeWorldActionComparator(loaded,selected).sample_native(
            inputs,loaded.field_normalization,draws=2,steps=1,seed=40)
        samples=law["particle_trajectory_samples"]
        assert torch.isfinite(samples).all()
        assert tuple(samples.shape[-4:])==tuple(group["items"][0]["trajectory"].shape)
        assert torch.equal(law["particle_weights"],original_posterior.weights)
        wrong=copy.deepcopy(selected)
        wrong["native_diffusion_data"]["posterior_context_width"]=common_width+1
        bad=training.MicroenvironmentPretrainingModel(wrong,False)
        try:bind_frozen_training_belief(bad,wrong,torch.device("cpu"))
        except ValueError as error:assert "shape" in str(error).lower()
        else:raise AssertionError("A different common-posterior width was accepted")
        results.append({"operator_width":width,"frozen_posterior_width":common_width,
            "loss":float(loss.detach()),"context_projection_gradient":projection_gradient,
            "operator_gradient":operator_gradient,"posterior_identity_preserved":True,
            "optimizer_updates":1,"strict_checkpoint_reload":True,"sample_shape":list(samples.shape)})
    return {"status":"passed","results":results,"source_operations":0,
        "scientific_twin":"NOT_RUN","formal_qualification":False,
        "scope":"tiny_native_loss_reload_and_sampling_with_shared_frozen_posterior"}

if __name__=="__main__":
    result=run(Path(sys.argv[1]),Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+"\n")
