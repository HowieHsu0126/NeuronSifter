"""Ordinary native-comparator fitting with shared frozen belief and selection."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from Libs.exp.training.pretrain_microenvironment_world_model import (
    NativeJointGroupDataset, joint_inference_modules, teacher_sample_weight,
    supported_teacher_precision, execute_optimizer_step,training_memory_policy,
)
from Libs.models.modules.evidence import encode_retained_observations
from Libs.models.modules.native_diffusion_comparators import NativeWDNO, NativeDiffusionPDE, NativeCODANO, NativeFNO
from Libs.models.modules.native_field_comparators import NativeStructuredFieldComparator, NativePersistenceComparator, NativeDeterministicComparator
from Libs.models.modules.native_mamba_operator import PaperMambaNO
from Libs.models.modules.native_sequence_comparators import NativeSequenceMean
from Libs.models.modules.native_state_probe import NativeStateProbe
from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver
from Libs.models.modules.native_mechanistic_residual import NativeMechanisticResidual
from Libs.models.modules.native_tau_transport import NativeTauTransport
from Libs.models.modules.native_treatment_diffusion import NativeTreatmentDiffusion
from Libs.models.modules.native_rssm import NativeGaussianRSSM,NativeDiscreteRSSM
from Libs.utils.model.native_scientific_source import training_data_scope,native_pretrainer_method
from Libs.utils.data.joint_observation import retained_encoder_inputs
from Libs.utils.data.native_diffusion_data import parse_joint_source
from Libs.utils.data.native_field_normalization import NativeFieldNormalization, normalization_from_config
from Libs.utils.data.native_input_staging import input_locations_from_config
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json
from Libs.utils.exp.experiment_manifests import resolve_training_seed, validate_training_seed_checkpoint
from Libs.utils.exp.experiment_tracking import seed_everything
from Libs.utils.model.native_comparator_inputs import (
    build_native_comparator_inputs, native_comparator_group_loss, sample_native_comparator,
    bind_comparator_grid,
)
from Libs.utils.model.native_comparator_physics import FrozenNativePhysicsGuidance
from Libs.utils.model.native_direct_mean import file_sha256
from Libs.utils.model.native_training_selection import NativeTrainingSelection,validation_checkpoint_due,optimizer_learning_rate_ratio,training_checkpoint_path


class FrozenNativeBelief(nn.Module):
    """Load the exact trained inference prefixes, without an unused rollout model."""
    def __init__(self, config: dict, checkpoint: dict):
        super().__init__()
        state=checkpoint["model_state_dict"]
        projection=state.get("world_state_posterior.joint_field_projection.weight")
        if projection is None:
            raise ValueError("Selected checkpoint has no trained native joint belief")
        validate_training_seed_checkpoint(checkpoint,resolve_training_seed(config,require_selected=True))
        hidden,species=projection.shape
        modules,self.observation_feature_width=joint_inference_modules(config,hidden,species)
        self.observation_encoder,self.retained_mask_projection,self.world_state_posterior=modules
        for name in ("observation_encoder","retained_mask_projection","world_state_posterior"):
            prefix=name+"."
            supplied={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
            if not supplied:raise ValueError("Missing trained belief module: "+name)
            getattr(self,name).load_state_dict(supplied,strict=True)
        self.field_normalization=NativeFieldNormalization(normalization_from_config(config),species,
            input_locations=input_locations_from_config(config))
        prefix="field_normalization."
        normalized={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
        self.field_normalization.load_state_dict(normalized,strict=True)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self,source: dict):
        support,observations=parse_joint_source(source,next(self.parameters()).device)
        values,masks=retained_encoder_inputs(observations,self.observation_feature_width,support=support,
            assays=self.world_state_posterior.joint_config.get("registered_occupancy_assays",{}))
        encoded=encode_retained_observations(self.observation_encoder,self.retained_mask_projection,values,masks)
        return self.world_state_posterior.infer_joint(encoded,values,support,observations,self.field_normalization)


def load_frozen_native_belief(config: dict, config_path: Path, manifest_path: Path, device) -> tuple:
    manifest=json.loads(manifest_path.read_text())
    selection=manifest.get("selection",{})
    best=selection.get("best")
    if selection.get("selection_role") != "development" or not best:
        raise ValueError("Shared conditioning requires an actual development-selected checkpoint")
    common_config=load_config(str(config_path))
    if config.get("native_input_staging", {}).get("manifest_path"):
        common_config["native_input_staging"] = copy.deepcopy(config["native_input_staging"])
    scope=config.get("native_scientific_training",{}).get("source_scope","all")
    if scope!="all" and training_data_scope(common_config)!=training_data_scope(config):
        raise ValueError("Isolated source comparator cannot reuse a mixed or differently scoped belief")
    binding=resolve_training_seed(config,require_selected=True)
    if (resolve_training_seed(common_config,require_selected=True) != binding
            or manifest.get("training_seed_binding") != binding):
        raise ValueError("Shared conditioning must use the same actual training seed")
    checkpoint_path=training_checkpoint_path(manifest_path,selection,best["checkpoint_path"])
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False,mmap=True)
    belief=FrozenNativeBelief(common_config,checkpoint).to(device)
    expected=normalization_from_config(config)
    if not belief.field_normalization.active or belief.field_normalization.identity != expected.get("sha256"):
        raise ValueError("Comparator and shared belief need the identical fitted normalization")
    identity={"checkpoint_path":str(checkpoint_path),"checkpoint_sha256":file_sha256(checkpoint_path),
        "conditioning_config_sha256":file_sha256(config_path),"manifest_path":str(manifest_path),
        "training_seed_binding":binding,"normalization_id":belief.field_normalization.identity,
        "selection_role":"development","loaded_component_names":[
            "observation_encoder","retained_mask_projection","world_state_posterior"],
        "conditioning_training_terminal":selection.get("terminal"),
        "sufficient_training":False}
    total=manifest.get("actual_total_parameter_count")
    if total is not None:
        if type(total) is not int or total<=0:raise ValueError("Selected mainline parameter count is invalid")
        identity.update(reference_total_parameter_count=total,
                        shared_inference_parameter_count=sum(p.numel() for p in belief.parameters()))
    return belief,identity



def bind_frozen_training_belief(model,config:dict,device) -> dict | None:
    """Native dynamics ablations reuse the selected mainline's actual inference modules."""
    policy=config["native_scientific_training"].get("conditioning",{"mode":"joint"})
    if policy["mode"]=="joint":
        return None
    if policy["mode"]!="frozen_selected_mainline":
        raise ValueError("Unknown scientific conditioning mode")
    belief,identity=load_frozen_native_belief(config,resolve_repo_path(policy["config_path"]),
        resolve_repo_path(policy["manifest_path"]),device)
    if model.field_normalization.identity!=belief.field_normalization.identity:
        raise ValueError("Frozen training conditioning normalization mismatch")
    for name in identity["loaded_component_names"]:
        current=getattr(model,name)
        supplied=getattr(belief,name)
        if {key:tuple(value.shape) for key,value in current.state_dict().items()
                if torch.is_tensor(value)}!={key:tuple(value.shape) for key,value in supplied.state_dict().items()
                if torch.is_tensor(value)}:
            raise ValueError("Frozen inference shape differs from the native ablation")
        setattr(model,name,supplied)
    model.frozen_conditioning_identity=identity
    return identity


def comparator_identity(config: dict,method: str,conditioning: dict) -> dict:
    specification=config["native_controlled_comparators"]
    return {"schema_version":"native_comparator_checkpoint.v1","method":method,
        "reference":copy.deepcopy(specification["references"][method]),
        "conditioning_layout":copy.deepcopy(specification["conditioning"]),
        "guidance":copy.deepcopy(specification["guidance"]),
        "quantity_profile":copy.deepcopy(config["native_observable_bridge"]["quantity_profile"]),
        "conditioning":copy.deepcopy(conditioning),
        "seed_binding":resolve_training_seed(config,require_selected=True),
        "training_data_scope":training_data_scope(config),
        **({"state_probe":copy.deepcopy(config["native_state_comparison"])}
           if method in config.get("native_state_comparison",{}).get("fitted_methods",[]) else {}),
        **({"native_backbone":copy.deepcopy(config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"])}
           if method=="native_deterministic" else {})}


def make_comparator(config: dict,method: str,inputs,conditioning:dict|None=None,*,belief=None):
    if method in config.get("native_state_comparison",{}).get("fitted_methods",[]):
        if belief is None:raise ValueError("State probe needs its selected common inference checkpoint")
        return NativeStateProbe(method,belief,config)
    entries=config["native_controlled_comparators"]["references"]
    if method == "same_compiler_solver":
        return NativeFrozenRHSSolver(config,entries[method]["hyperparameters"])
    shape=(config["native_controlled_comparators"]["conditioning"]["maximum_times"],
           *inputs.validity.shape[2:])
    width=inputs.condition.shape[1]
    if method == "persistence":return NativePersistenceComparator()
    if method == "native_deterministic":return NativeDeterministicComparator(inputs,config)
    if method == "tau_transport":return NativeTauTransport(entries[method]["hyperparameters"],inputs,config)
    if method == "mechanistic_residual":return NativeMechanisticResidual(entries[method]["hyperparameters"],inputs,config)
    if method == "treatment_history":return NativeTreatmentDiffusion(entries[method]["hyperparameters"],inputs,config)
    if method in {"gru","flattened_parameter_matched"}:
        return NativeSequenceMean(method,entries[method]["hyperparameters"],inputs,config,conditioning)
    if method == "planet":return NativeGaussianRSSM(entries[method]["hyperparameters"],inputs,config)
    if method == "dreamerv3":return NativeDiscreteRSSM(entries[method]["hyperparameters"],inputs,config)
    if method == "fno":return NativeFNO(entries[method],shape,width)
    if method == "mamba_no":return PaperMambaNO(entries[method],shape,width)
    if method in {"linear","transolver"}:
        return NativeStructuredFieldComparator(method,entries[method]["hyperparameters"],inputs,config)
    if method == "wdno":return NativeWDNO(entries[method],shape,width)
    if method == "diffusion_pde":return NativeDiffusionPDE(entries[method],shape,width)
    if method == "codano":return NativeCODANO(entries[method],shape,width,inputs.grid["species_id"])
    raise ValueError("No authentic native comparator implementation: "+method)


def prepare_comparator_group(group: dict,belief,config: dict,device):
    posterior=belief(group["joint_source"])
    normalizer=belief.field_normalization
    inputs=build_native_comparator_inputs(group["joint_source"],posterior,normalizer,config)
    rows=group["items"]
    target=torch.stack([row["trajectory"] for row in rows]).to(device)[None]
    mask=torch.stack([row["quantity_support"] for row in rows]).to(device)[None]
    if target.shape[1] != inputs.group_shape[-1]:
        raise ValueError("Whole-group comparator targets do not cover all actions")
    raw={key:torch.stack([row[key] for row in rows]).to(device)
         for key,value in rows[0].items() if torch.is_tensor(value)}
    weights=teacher_sample_weight(raw,target[0])*supported_teacher_precision(raw["uncertainty"],config,mask[0])
    return inputs,target,mask,weights[None]


def comparator_mean(model,method,inputs,normalizer,config,seed:int,*,development=False):
    if method == "same_compiler_solver":
        result=model.forecast_native(inputs,normalizer)
        fields=result["particle_fields"]
        result["particle_trajectory_samples"]=fields.unsqueeze(0)
        result["conditional_law"]="deterministic"
        return (fields*inputs.weights[:,:,None,None,None,None,None]).sum(1),result
    if method in config["native_controlled_comparators"]["conditional_mean_methods"]:
        if not hasattr(model,"predict_native"):
            bind_comparator_grid(model,inputs)
            predicted=normalizer.inverse(model.predict(inputs.validity,inputs.condition),inputs.validity)
        else:
            predicted=model.predict_native(inputs,normalizer)
        fields=predicted.reshape(*inputs.group_shape,*predicted.shape[1:])
        return (fields*inputs.weights[:,:,None,None,None,None,None]).sum(1),fields
    policy=config["native_scientific_training"]
    if native_pretrainer_method(config,method):
        draws=policy["predictive_budget"]["conditional_draws_per_particle"]
        steps=policy["predictive_budget"]["categorical_reverse_steps"]
        result=model.sample_native(inputs,normalizer,draws,steps,seed)
        fields=result["particle_trajectory_samples"].mean(0)
        return (fields*inputs.weights[:,:,None,None,None,None,None]).sum(1),result
    if method == "persistence":
        fields=model.particle_fields(inputs)
        draws=policy["predictive_budget"]["conditional_draws_per_particle"]
        return (fields*inputs.weights[:,:,None,None,None,None,None]).sum(1),{
            "particle_trajectory_samples":fields[None].expand(draws,*fields.shape)}
    draws=policy["validation"]["sample_count"] if development else policy["predictive_budget"]["conditional_draws_per_particle"]
    steps=policy["validation"]["denoising_step_count"] if development else policy["predictive_budget"]["categorical_reverse_steps"]
    if method == "treatment_history":
        sampled=model.sample_native(inputs,normalizer,draws,steps,seed)
        fields=sampled["particle_trajectory_samples"].mean(0)
        return (fields*inputs.weights[:,:,None,None,None,None,None]).sum(1),sampled
    guidance=FrozenNativePhysicsGuidance(inputs,normalizer,config) if method == "diffusion_pde" else None
    sampled=sample_native_comparator(model,inputs,normalizer,draws_per_particle=draws,steps=steps,
                                    seed=seed,guidance=guidance)
    fields=sampled["particle_trajectory_samples"].mean(0)
    return (fields*inputs.weights[:,:,None,None,None,None,None]).sum(1),sampled


def fit_native_comparator(belief,method:str,train,development,config:dict,
                          output:Path,conditioning:dict,device) -> dict:
    """Finite ordinary optimizer loop; no implicit restart or scientific approval."""
    if method == "same_compiler_solver":
        raise ValueError("Frozen primary RHS has no transition fit")
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/"native_comparator_training_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Comparator attempt already has a training manifest")
    binding=resolve_training_seed(config,require_selected=True)
    seed=int(binding["model_seed"])
    seed_everything(seed)
    state_probe=method in config.get("native_state_comparison",{}).get("fitted_methods",[])
    state_setup_started=time.monotonic()
    first_inputs=None
    if not state_probe:
        first_inputs,_,_,_=prepare_comparator_group(train[0]["native_joint_group"],belief,config,device)
    model=make_comparator(config,method,first_inputs,conditioning,belief=belief).to(device)
    if state_probe:model.fit_pca(train)
    state_setup_seconds=time.monotonic()-state_setup_started
    policy=config["native_scientific_training"]
    selection=NativeTrainingSelection(copy.deepcopy(policy["selection"]))
    training=config["screening_pipeline"]["screening_benchmark"]["training"]
    effective_batch=int(training_memory_policy(config,False)["effective_batch_size"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=training["learning_rate"],weight_decay=training["weight_decay"])
    identity=comparator_identity(config,method,conditioning)
    schedule=policy["learning_rate_schedule"]
    started=state_setup_started if state_probe else time.monotonic();steps=0;owned=set();failure=None
    pending=0;groups_seen=0
    order_generator=torch.Generator().manual_seed(seed)
    normalizer=belief.field_normalization
    def payload():
        return {"schema_version":"native_comparator_training.v1","method":method,"identity":identity,
            "training_seed_binding":binding,"selection":selection.payload(artifact_root=output),"executed_steps":steps,
            "effective_group_batch_size":effective_batch,"executed_group_visits":groups_seen,
            "train_group_count":len(train),"development_group_count":len(development),
            "train_family_group_counts":train.family_group_counts,
            "development_family_group_counts":development.family_group_counts,
            "wall_seconds":time.monotonic()-started,"failure":failure,
            "development_metric":("held_readout_negative_log_likelihood" if state_probe else
                "posterior_weighted_relative_l2_train_normalized_fields" if method=="tau_transport"
                else "normalized_predictive_mean_squared_error"),
            "offline_residual_adaptation":({"objective":"uniform_normalized_one_step_residual_mse",
                "teacher_forcing":"train_only_independent_CNS_and_neural_states",
                "parameter_conditioning":"all32_public_hypotheses_no_hidden_index_pairing",
                "online_adaptation":"NOT_IMPLEMENTED", "full_MPC":"NOT_IMPLEMENTED"}
                if method=="mechanistic_residual" else None),
            "state_probe_fit":({"pca_training_observations":model.pca_fit_count,
                "initialization_and_pca_seconds":state_setup_seconds,
                "input_boundary":"observations_and_prior_only","truth_used_for_fitting":False,
                "trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad)}
                if state_probe else None),
            "parameter_match":getattr(model,"parameter_match",None),
            "method_mapping":getattr(model,"method_mapping",None),
            "ema_updates":int(model.ema_updates) if hasattr(model,"ema_updates") else None,
            "last_training_diagnostics":getattr(model,"last_training_diagnostics",None),
            "actual_predictor_parameter_count":sum(parameter.numel() for parameter in model.parameters()),
            "prediction_component":config.get("native_rssm_prediction",{}).get("scope")
                if method in config.get("native_rssm_prediction",{}).get("fitted_methods",[]) else None,
            "scientific_twin":"NOT_RUN","formal_qualification":False}
    def persist():write_json(manifest_path,payload())
    def capped():
        return steps>=selection.policy["max_steps"] or time.monotonic()-started>=selection.policy["max_seconds"]
    def retire():
        keep={Path(row["checkpoint_path"]) for row in (selection.best,selection.last) if row}
        for path in owned-keep:path.unlink()
        owned.intersection_update(keep)
    def advance_optimizer():
        nonlocal steps
        fraction=optimizer_learning_rate_ratio(steps,policy)
        for row in optimizer.param_groups:row["lr"]=training["learning_rate"]*fraction
        execute_optimizer_step(model,optimizer,config)
        if hasattr(model,"update_ema"):model.update_ema()
        steps+=1
    last_checkpoint_step=0
    window_losses=[]
    def checkpoint_and_validate(epoch):
        nonlocal last_checkpoint_step
        suffix=f"_step_{steps}" if policy["validation"].get("interval_optimizer_steps") else ""
        path=output/f"{method}_epoch_{epoch}{suffix}.pt"
        if path.exists():raise FileExistsError("Immutable comparator checkpoint already exists")
        torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
            "epoch":epoch,"executed_steps":steps,"training_seed_binding":binding,
            "native_comparator":identity},path)
        owned.add(path)
        if sum(item.stat().st_size for item in owned)>policy["checkpoint_storage"]["maximum_live_bytes"]:
            raise RuntimeError("Comparator live checkpoints exceed the declared storage limit")
        previous=selection.last
        average=sum(window_losses)/len(window_losses)
        selection.last={"epoch":epoch,"steps":steps,"checkpoint_path":str(path),
            "training_loss":average,"development_loss":None,"elapsed_seconds":time.monotonic()-started}
        last_checkpoint_step=steps
        persist()
        if path.stat().st_size>policy["checkpoint_storage"]["maximum_payload_bytes"]:
            raise RuntimeError("Comparator checkpoint exceeds the declared payload limit")
        if capped():
            selection.terminal="INCOMPLETE_BUDGET_CAP";retire();return True
        model.eval();validation=[]
        try:
            with torch.no_grad():
                for index in range(len(development)):
                    if capped():break
                    group=development[index]["native_joint_group"]
                    if state_probe:
                        validation.append(float(-model.held_scores(group["joint_source"]).mean()))
                    else:
                        inputs,target,mask,action_weights=prepare_comparator_group(group,belief,config,device)
                        if method=="tau_transport":
                            validation.append(float(native_comparator_group_loss(model,inputs,target,mask,
                                action_weights,normalizer,torch.Generator(device=device).manual_seed(seed+index))))
                            continue
                        mean,_=comparator_mean(model,method,inputs,normalizer,config,
                            seed+policy["validation_stream_offset"]+index,development=True)
                        error=((mean-target)/normalizer.scale.to(mean)).square()
                        validation.append(float(torch.where(mask,error,0).sum()/mask.sum()))
        finally:
            model.train()
        if len(validation)!=len(development):
            selection.terminal="INCOMPLETE_BUDGET_CAP";retire();return True
        selection.last=previous
        stop=selection.update(role="development",epoch=epoch,steps=steps,
            training_loss=average,development_loss=sum(validation)/len(validation),
            checkpoint_path=str(path),elapsed_seconds=time.monotonic()-started)
        window_losses.clear();retire();persist()
        return stop
    def select_if_due(epoch,epoch_end=False):
        if validation_checkpoint_due(policy,steps,last_checkpoint_step,epoch,
                                      epoch_end=epoch_end,budget_capped=capped()):
            return checkpoint_and_validate(epoch)
        return False
    persist()
    try:
        stopped=False
        for epoch in range(1,selection.policy["max_epochs"]+1):
            model.train();epoch_visits=0
            generator=torch.Generator(device=device).manual_seed(seed+(epoch-1)*policy["epoch_stream_stride"])
            for index in torch.randperm(len(train),generator=order_generator).tolist():
                if capped():break
                group=train[index]["native_joint_group"]
                if state_probe:
                    loss=model.loss(group["joint_source"])
                else:
                    inputs,target,mask,weight=prepare_comparator_group(group,belief,config,device)
                    loss=native_comparator_group_loss(model,inputs,target,mask,weight,normalizer,generator)
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite comparator training objective")
                (loss/effective_batch).backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite comparator optimizer gradient")
                window_losses.append(float(loss.detach()))
                pending+=1;groups_seen+=1;epoch_visits+=1
                if pending==effective_batch:
                    advance_optimizer();pending=0
                    stopped=select_if_due(epoch)
                    if stopped:break
            if stopped:break
            if pending:
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.mul_(effective_batch/pending)
                advance_optimizer();pending=0
            if not epoch_visits:
                selection.terminal="INCOMPLETE_BUDGET_CAP";break
            if select_if_due(epoch,epoch_end=True):break
    except BaseException as error:
        failure={"type":type(error).__name__,"message":str(error)}
        selection.terminal="FAILED";retire();persist();raise
    persist()
    return payload()
