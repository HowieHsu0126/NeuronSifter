"""Native registered-grid wrappers for the maintained direct field baselines."""
from __future__ import annotations

import torch
from torch import nn

from Libs.models.modules.native_linear_dynamics import NativeLinearDynamics
from Libs.models.modules.neuronsifter_world_action import native_world_action_from_config
from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
from Libs.models.modules.native_transolver_mean import NativeTransolverMean
from Libs.models.modules.transolver_operator import TransolverMicroenvironmentOperator
from Libs.utils.data.native_diffusion_data import joint_particle_contexts
from Libs.utils.data.native_observable_profile import family_grid, observable_profile
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.model.initial_state_persistence import initial_particle_fields


class NativeStructuredFieldComparator(nn.Module):
    prediction_kind = "mean_only"

    def __init__(self, method: str, settings: dict, inputs, config: dict):
        super().__init__()
        self.method = method
        self.static_stop = inputs.blocks["source_nuisance"][1]
        self.family_block = inputs.blocks["physical_domain"]
        self.field_shape = tuple(inputs.initial.shape[-3:])
        self.context_dim = settings["posterior_context_dim"]
        self.context_projection = nn.Linear(
            self.static_stop+self.family_block[1]-self.family_block[0],self.context_dim)
        self.forcing_scales = config["native_controlled_comparators"]["conditioning"]["program_reference_scales"]
        profile = observable_profile(config)
        domain = family_grid(profile,next(iter(profile["families"])))
        self.quantity_ids = list(domain["species_id"])
        if method == "linear":
            self.direct = NativeLinearDynamics(domain,self.context_dim,
                program_channel_count=2+len(FORCING_ARRAY_NAMES),normalization_scope="registered_native")
        elif method == "transolver":
            hidden = settings["hidden_dim"]
            self.direct = NativeTransolverMean(len(self.quantity_ids),hidden,self.context_dim,
                                                program_channel_count=2+len(FORCING_ARRAY_NAMES))
            self.operator = TransolverMicroenvironmentOperator(
                hidden_dim=hidden,encoder_depth=settings["encoder_depth"],decoder_depth=settings["decoder_depth"],
                head_count=settings["head_count"],slice_count=settings["slice_count"],coordinate_dim=2,
                dropout_rate=settings["dropout_rate"],expert_names=settings["expert_names"],
                top_k=settings["top_k"],capacity_factor=settings["capacity_factor"],
                entropy_loss_weight=settings["entropy_loss_weight"],
                load_balance_loss_weight=settings["load_balance_loss_weight"],
                expert_capacity_loss_weight=settings["expert_capacity_loss_weight"],
                expert_hidden_multiplier=settings["expert_hidden_multiplier"],
                physics_mlp_multiplier=settings["physics_mlp_multiplier"],
                checkpoint_attention_blocks=settings["checkpoint_attention_blocks"])
            self.register_buffer("adjacency",torch.as_tensor(node_adjacency_matrix(config)),persistent=False)
            r,c,_ = self.field_shape
            self.register_buffer("coordinates",torch.tensor([
                [ri/max(r-1,1),ci/max(c-1,1)] for ri in range(r) for ci in range(c)]),persistent=False)
        else:
            raise ValueError("Unknown native structured field comparator")

    def predict_native(self,inputs,normalization):
        if (tuple(inputs.initial.shape[-3:]) != self.field_shape
                or inputs.grid["species_id"] != self.quantity_ids
                or inputs.normalization_id != normalization.identity):
            raise ValueError("Direct comparator native quantity/normalization identity mismatch")
        normalization.assert_domain(inputs.grid)
        b,j,a = inputs.group_shape
        r,c,s = self.field_shape
        contexts = joint_particle_contexts(inputs.posterior,inputs.programs,inputs.regimen_uids)
        def stack(key):
            values = torch.stack([row[key] for row in contexts],dim=1)
            return values.reshape(b*j,a,*values.shape[3:])
        context = {key:stack(key) for key in ("initial_state","occupancy","support","physical_time")}
        lo,hi = self.family_block
        static = torch.cat((inputs.condition[:,:self.static_stop],inputs.condition[:,lo:hi]),-1)
        latent = self.context_projection(static).reshape(b*j,a,1,1,self.context_dim)
        context["posterior_context"] = latent.expand(-1,-1,r,c,-1)
        context["additional_forcing"] = torch.cat(
            [stack(key)/self.forcing_scales[key] for key in FORCING_ARRAY_NAMES],-1)
        context["quantity_support"] = inputs.validity.reshape(b*j,a,*inputs.validity.shape[1:])
        context["native_grid"] = inputs.grid
        if self.method == "linear":
            result = self.direct(context,normalization)
        else:
            result = self.direct(context,self.operator,normalization,self.adjacency,self.coordinates)
        value = result["field_trajectory"].reshape_as(inputs.validity)
        if not torch.isfinite(value[inputs.validity]).all():
            raise FloatingPointError("Nonfinite direct native field")
        return torch.where(inputs.validity,value,0)


class NativePersistenceComparator(nn.Module):
    """No fitted transition; randomness is exactly the retained initial posterior."""
    prediction_kind = "initial_posterior_identity_law"

    def __init__(self):
        super().__init__()
        self.register_buffer("device_anchor",torch.empty(0),persistent=False)

    def particle_fields(self,inputs):
        return initial_particle_fields(inputs.posterior,len(inputs.regimen_uids),len(inputs.grid["time_hours"]))


class NativeWorldActionComparator(nn.Module):
    """Target-free all-particle calls to the maintained native categorical sampler."""
    def __init__(self,model,config):
        super().__init__()
        self.mainline=model
        self.config=config

    def sample_native(self,inputs,normalization,draws:int,steps:int,seed:int):
        model=self.mainline
        if draws<1 or steps<1:
            raise ValueError("Native forecast requires positive draw and reverse-step budgets")
        if model.field_normalization.identity!=normalization.identity:
            raise ValueError("Native rollout and common posterior normalization mismatch")
        contexts=joint_particle_contexts(inputs.posterior,inputs.programs,inputs.regimen_uids,
                                         action_policy=model.action_policy_name)
        b,j,a=inputs.group_shape
        t,r,c,s=inputs.validity.shape[1:]
        coordinates=inputs.condition.new_tensor([
            [ri/max(r-1,1),ci/max(c-1,1)] for ri in range(r) for ci in range(c)])
        sampler=model.quantized_mddm
        previous=(sampler.inference_sample_count,sampler.inference_denoising_step_count,sampler.sampler_seed)
        results=[];action_conditioning=[]
        try:
            sampler.inference_sample_count=draws
            sampler.inference_denoising_step_count=steps
            for index,context in enumerate(contexts):
                sampler.sampler_seed=seed+index*self.config["native_scientific_training"]["particle_stream_stride"]
                condition=dict(context)
                condition["quantity_support"]=inputs.validity.reshape(b,j,a,t,r,c,s)[:,index]
                with torch.no_grad():
                    result=sampler(inputs.condition.new_zeros(b,a,t,r,c,s),False,
                        initial_state=context["initial_state"],conditioning=condition,native_operator=model.operator,
                        adjacency=model.node_adjacency,coordinates=coordinates,
                        field_normalization=model.field_normalization,return_samples=True)
                results.append(result["sample_trajectories"])
                if "action_conditioning" in result:
                    action_conditioning.append(dict(result["action_conditioning"],particle_index=index))
        finally:
            sampler.inference_sample_count,sampler.inference_denoising_step_count,sampler.sampler_seed=previous
        return {"particle_trajectory_samples":torch.stack(results,dim=2),
                "particle_weights":inputs.weights,"forecast_paths_per_action":j*draws,
                "action_conditioning":action_conditioning}


class NativeClockedOperator(nn.Module):
    """Expose native deterministic forward with the original physical query clock."""
    def __init__(self,backbone:dict,adjacency,time_scale:float,field_shape:tuple[int,int,int]):
        super().__init__()
        self.native=native_world_action_from_config(backbone)
        self.time_projection=nn.Linear(1,int(backbone["operator_hidden_dim"]))
        self.time_scale=float(time_scale)
        self.field_shape=field_shape
        self.register_buffer("adjacency",torch.as_tensor(adjacency),persistent=False)

    def forward(self,hidden,conditioning,coordinates,*,physical_time):
        conditioning=conditioning+self.time_projection(physical_time/self.time_scale)
        batch,times,nodes,width=hidden.shape
        r,c,_=self.field_shape
        packed,layout=QuantizedMddmRolloutOperator.pack(hidden.reshape(batch,times,r,c,width),self.native)
        condition,condition_layout=QuantizedMddmRolloutOperator.pack(conditioning.reshape(batch,times,r,c,width),self.native)
        if layout!=condition_layout:raise ValueError("Native deterministic condition/time packing mismatch")
        packed_times=packed.shape[1]
        encoded,diagnostics=self.native.forward_preembedded(packed.reshape(batch,packed_times,nodes,width),
            condition.reshape(batch,packed_times,nodes,width),coordinates[:,:1].expand(-1,packed_times,-1,-1),self.adjacency)
        restored=QuantizedMddmRolloutOperator.unpack(encoded.reshape(batch,packed_times,r,c,width),layout,self.native)
        diagnostics["deterministic_wavelet_level_count"]=hidden.new_tensor(self.native.tokenizer.wavelet_level_count)
        return restored.reshape(batch,times,nodes,width),diagnostics


class NativeDeterministicComparator(nn.Module):
    prediction_kind="mean_only"

    def __init__(self,inputs,config):
        super().__init__()
        backbone=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        self.hidden=int(backbone["operator_hidden_dim"])
        self.forcing_scales=config["native_controlled_comparators"]["conditioning"]["program_reference_scales"]
        r,c,s=inputs.initial.shape[-3:]
        self.field_shape=(r,c,s)
        self.direct=NativeTransolverMean(s,self.hidden,self.hidden,program_channel_count=2+len(FORCING_ARRAY_NAMES))
        self.operator=NativeClockedOperator(backbone,node_adjacency_matrix(config),self.forcing_scales["physical_time"],self.field_shape)
        self.register_buffer("coordinates",torch.tensor([[ri/max(r-1,1),ci/max(c-1,1)]
            for ri in range(r) for ci in range(c)]),persistent=False)

    def predict_native(self,inputs,normalization):
        if tuple(inputs.initial.shape[-3:])!=self.field_shape or inputs.normalization_id!=normalization.identity:
            raise ValueError("Native deterministic domain or normalization mismatch")
        b,j,a=inputs.group_shape
        contexts=joint_particle_contexts(inputs.posterior,inputs.programs,inputs.regimen_uids)
        def stack(name):
            values=torch.stack([row[name] for row in contexts],1)
            return values.reshape(b*j,a,*values.shape[3:])
        context={name:stack(name) for name in ("initial_state","posterior_context","occupancy","support","physical_time")}
        context["quantity_support"]=inputs.validity.reshape(b*j,a,*inputs.validity.shape[1:])
        context["additional_forcing"]=torch.cat([stack(name)/self.forcing_scales[name] for name in FORCING_ARRAY_NAMES],-1)
        result=self.direct(context,self.operator,normalization,self.operator.adjacency,self.coordinates)
        return result["field_trajectory"].reshape_as(inputs.validity)
