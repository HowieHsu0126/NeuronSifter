"""Run the native comparator caller on an explicitly retained disposable source.

Requires a previously completed software fixture; never generates or replays it.
The caller records its own test output without changing that fixture.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.native_diffusion_data import parse_joint_source
from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.utils.data.joint_physical_parameters import bind_candidate_parameters,parameter_schema_from_config
from Libs.utils.model.native_comparator_inputs import (
    build_native_comparator_inputs,native_comparator_group_loss,sample_native_comparator,bind_comparator_grid)
from Libs.utils.model.native_comparator_physics import FrozenNativePhysicsGuidance,auxiliary_step
from Libs.models.modules.native_diffusion_comparators import NativeWDNO,NativeDiffusionPDE,NativeCODANO
from Libs.exp.training.pretrain_microenvironment_world_model import joint_inference_modules
from Libs.utils.model.controlled_cns_source import ControlledLaw,transport
from Libs.utils.exp.native_comparator_training import make_comparator,comparator_mean


def run(origin: Path, methods=("wdno","diffusion_pde","codano"), *, mamba_full_topology=False, family=None) -> dict:
    config=load_config(str(origin/"native_config.yaml"))
    current=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_controlled_comparators"]=copy.deepcopy(current["native_controlled_comparators"])
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"][
        "physical_parameter_conditioning"]="source_declared_named_candidates"
    config["screening_pipeline"]["screening_benchmark"]["training"].update(hidden_dim=12,dropout_rate=0.)
    posterior_settings=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    posterior_settings.update(posterior_hidden_dim=12,posterior_dropout_rate=0.)
    receipt=json.loads((origin/"source_receipt.json").read_text())
    assert receipt["source_status"] == "complete"
    artifact_path=origin/"train_normalization.json";data=artifact_path.read_bytes()
    normalization=NativeFieldNormalization({"kind":"training_scaler","artifact_ref":str(artifact_path),
        "sha256":hashlib.sha256(data).hexdigest(),"artifact":json.loads(data)},26)
    schema=parameter_schema_from_config(config)
    models={}; outcomes=[]
    for gi,group in enumerate(receipt["groups"]):
        if group["role"] != "train" or family is not None and group["family_uid"]!=family:continue
        print("CALLER_FAMILY",group["family_uid"],flush=True)
        raw=json.loads((origin/f"group_{gi}_conditioning.json").read_text())
        producer=json.loads((origin/f"group_{gi}_producer.json").read_text())
        raw=bind_candidate_parameters(raw,group,producer,schema,
            record_ref=str(origin/f"group_{gi}_producer.json"),
            producer_commit=receipt["source_identity"]["producer_commit"])
        support,observations=parse_joint_source(raw)
        torch.manual_seed(19411)
        (_,_,belief),width=joint_inference_modules(config,12,26)
        belief.requires_grad_(False).eval()
        posterior=belief.infer_joint(torch.zeros(1,belief.factor_dimensions["z_tx"]),
                                    torch.zeros(1,width),support,observations)
        inputs=build_native_comparator_inputs(raw,posterior,normalization,config)
        assert inputs.condition.shape[0] == 8
        low,high=inputs.blocks["named_parameters_and_presence"]
        assert high-low == 428
        assert torch.equal(inputs.condition[:,low+214:high].sum(-1),
                           torch.full((8,),len(support.physical_parameter_hypotheses["declarations"])))
        changed=copy.deepcopy(raw)
        changed["producer_seeds"]={"latent_seed":998881,"observation_seed":777331}
        replay=build_native_comparator_inputs(changed,posterior,normalization,config)
        assert torch.equal(inputs.condition,replay.condition)
        forbidden=copy.deepcopy(raw);forbidden["future_targets"]=[1e99]
        try: build_native_comparator_inputs(forbidden,posterior,normalization,config)
        except ValueError: pass
        else: raise AssertionError("Future-target source was accepted")
        cases=[row for row in receipt["cases"] if row["group_index"] == gi]
        targets=[]
        for case in cases:
            with np.load(origin/case["raw_reference"]) as payload:
                targets.append(torch.from_numpy(payload["trajectory"]).float())
        target=torch.stack(targets)[None]
        mask=inputs.validity.reshape(1,4,2,*inputs.validity.shape[1:])[:,0]
        guidance=None
        if "diffusion_pde" in methods:
            guidance=FrozenNativePhysicsGuidance(inputs,normalization,config)
            identity_only=copy.deepcopy(config)
            identity_only["native_observable_bridge"]["active_source_law"]="cooperative_exponential_source"
            identity_only["native_observable_bridge"]["held_source_law"]={"unavailable_test_equation":1e30}
            same_guidance=FrozenNativePhysicsGuidance(inputs,normalization,identity_only)
            proposed=normalization.normalize(target,mask)[:,None].expand(1,4,2,*target.shape[2:]).reshape_as(inputs.validity).detach().requires_grad_(True)
            obs,eq=guidance(proposed)
            same_obs,same_eq=same_guidance(proposed)
            if (~guidance.observed).any():
                assert not torch.isfinite(guidance.observation[~guidance.observed]).all()
                alternate=copy.copy(guidance)
                alternate.observation=torch.where(guidance.observed,guidance.observation,
                                                   torch.full_like(guidance.observation,1e30))
                alt_obs,alt_eq=alternate(proposed)
                assert torch.equal(alt_obs,obs) and torch.equal(alt_eq,eq)
            assert torch.equal(obs,same_obs) and torch.equal(eq,same_eq)
            gradient=torch.autograd.grad((obs+eq).sum(),proposed)[0]
            print("GUIDANCE",json.dumps({"observation_finite":bool(torch.isfinite(obs).all()),"equation_finite":bool(torch.isfinite(eq).all()),"gradient_finite":bool(torch.isfinite(gradient).all()),"invalid_quantity_indices":(~torch.isfinite(gradient)).any(dim=(0,1,2,3)).nonzero().flatten().tolist()}),flush=True)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            # Exact reaction and graph-transport equivalence to the maintained primary
            # law, at supplied predicted states. Neither side integrates a trajectory.
            known=ControlledLaw(config["controlled_cns_source"],
                {key:value[0,0].numpy() for key,value in support.factors.items()},
                support.initial_field[0,0,...,:15].numpy())
            state=torch.rand(8,1,36,11,dtype=torch.float64)+.1
            occupancy=torch.rand(8,1,36,dtype=torch.float64)
            actual=guidance.reaction(state,occupancy)
            expected=known.reaction(state[0,0].numpy(),occupancy[0,0].numpy())
            for observed,reference in zip(actual,expected):
                np.testing.assert_allclose(observed[0,0].detach().numpy(),reference,rtol=1e-12,atol=1e-12)
            actual_transport=guidance.transport(state)
            reference_transport=transport(state[0,0].numpy(),known.volumes,known.edges,known.conductance,known.transport_scale)
            for observed,reference in zip(actual_transport,reference_transport):
                np.testing.assert_allclose(observed[0,0].detach().numpy(),reference,rtol=1e-12,atol=1e-12)
        if not models:
            wdno=copy.deepcopy(config["native_controlled_comparators"]["references"]["wdno"])
            wdno["hyperparameters"].update(dim=8,dim_mults=[1],resnet_block_groups=1,conditioning_channels=4,diffusion_steps=8)
            edm=copy.deepcopy(config["native_controlled_comparators"]["references"]["diffusion_pde"])
            edm["hyperparameters"].update(model_channels=8,channel_mult=[1],num_blocks=1,attention_resolutions=[],sigma_max=1.)
            coda=copy.deepcopy(config["native_controlled_comparators"]["references"]["codano"])
            coda["hyperparameters"].update(n_layers=1,lifting_channels=4,hidden_variable_codimension=2,
                projection_channels=4,conditioning_channels=2,positional_encoding_dim=2,
                positional_encoding_modes=[2,2],n_modes=[2,2],n_heads=1)
            shape=(28,6,6,26);width=inputs.condition.shape[1]
            settings=config["native_controlled_comparators"]["references"]
            settings["transolver"]["hyperparameters"].update(hidden_dim=12,posterior_context_dim=4,
                encoder_depth=1,decoder_depth=0,head_count=3,slice_count=4,checkpoint_attention_blocks=False)
            settings["linear"]["hyperparameters"]["posterior_context_dim"]=2
            settings["mamba_no"]["hyperparameters"].update(width=4,depth=2,state_width=2,expansion=1,
                conditioning_channels=2,encoder_convolutions=[1,1],encoder_integrations=[1,1],
                decoder_integrations=[1,1],sinc_radius=2)
            if mamba_full_topology:
                settings["mamba_no"]["hyperparameters"].update(depth=4,encoder_convolutions=[2]*4,
                    encoder_integrations=[2]*4,decoder_integrations=[2]*4)
            settings["gru"]["hyperparameters"].update(hidden_dim=12)
            settings["flattened_parameter_matched"]["hyperparameters"].update(hidden_dim=12,depth=1,head_count=3)
            settings["fno"]["hyperparameters"].update(hidden_channels=4,n_layers=1,n_modes=[2,2],conditioning_channels=2)
            backbone=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
            backbone.update(operator_hidden_dim=12,operator_depth=1,attention_head_count=3,mechanism_state_token_count=6)
            assert backbone["operator_hidden_dim"]==12 and backbone["operator_depth"]==1
            fixture_capacity={"reference_total_parameter_count":500000,"shared_inference_parameter_count":10000}
            factories={"wdno":lambda:NativeWDNO(wdno,shape,width),
                       "diffusion_pde":lambda:NativeDiffusionPDE(edm,shape,width),
                       "codano":lambda:NativeCODANO(coda,shape,width,inputs.grid["species_id"])}
            models={name:(factories[name]() if name in factories else make_comparator(config,name,inputs,fixture_capacity))
                    for name in methods}
        for name,model in models.items():
            model.train();model.zero_grad(set_to_none=True)
            if name=="persistence":
                loss=torch.zeros(())
            else:
                loss=native_comparator_group_loss(model,inputs,target,mask,torch.ones(1,2),
                    normalization,torch.Generator().manual_seed(771))
                loss.backward()
                assert torch.isfinite(loss)
                assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
                assert any(parameter.grad is not None and parameter.grad.abs().sum()>0 for parameter in model.parameters())
            model.eval()
            if name in config["native_controlled_comparators"]["conditional_mean_methods"]:
                mean=(model.predict_native(inputs,normalization) if hasattr(model,"predict_native")
                      else model.predict(inputs.validity,inputs.condition))
                assert mean.shape == inputs.validity.shape and torch.isfinite(mean).all()
                assert not hasattr(model,"sample"),"A conditional mean must not fabricate a law"
                paths=0
            elif name=="persistence":
                mean,law=comparator_mean(model,name,inputs,normalization,config,994)
                expected=inputs.posterior.support.initial_field[:,:,None,None].expand(1,4,2,len(inputs.grid["time_hours"]),6,6,26)
                assert torch.equal(law["particle_trajectory_samples"][0],expected)
                assert not list(model.parameters())
                paths=law["particle_trajectory_samples"].shape[0]*4
            else:
                sampled=sample_native_comparator(model,inputs,normalization,draws_per_particle=1,
                    steps=3,seed=994,guidance=guidance if name=="diffusion_pde" else None)
                assert sampled["particle_trajectory_samples"].shape == (1,1,4,2,*inputs.validity.shape[1:])
                assert sampled["path_weights"].sum() == 1.
                paths=sampled["forecast_paths_per_action"]
            outcome={"family":group["family_uid"],"method":name,"condition_width":inputs.condition.shape[1],
                     "loss":float(loss.detach()),"forecast_paths_per_action":paths,
                     "parameter_match":getattr(model,"parameter_match",None)}
            print("COMPARATOR_CALLER",json.dumps(outcome),flush=True);outcomes.append(outcome)
    assert len(outcomes)==(1 if family else 4)*len(methods)
    return {"status":"passed","comparators":outcomes,"source_operations":0,
            "retained_source_origin":str(origin),"scientific_twin":"NOT_RUN","formal_qualification":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-fixture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--methods",nargs="+",choices=["wdno","diffusion_pde","codano","transolver","linear","persistence","mamba_no","native_deterministic","flattened_parameter_matched","gru","fno"],default=["wdno","diffusion_pde","codano"])
    parser.add_argument("--mamba-full-topology",action="store_true")
    parser.add_argument("--family")
    args=parser.parse_args()
    args.output.write_text(json.dumps(run(args.retained_fixture,args.methods,mamba_full_topology=args.mamba_full_topology,family=args.family),indent=2)+"\n")
