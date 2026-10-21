"""Whole-particle GRU and flattened shared temporal-operator conditional means."""
from __future__ import annotations

import math
import torch
from torch import nn

from Libs.utils.data.native_diffusion_data import joint_particle_contexts
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES


class NativeSequenceMean(nn.Module):
    prediction_kind="mean_only"

    def __init__(self,method:str,settings:dict,inputs,config:dict,conditioning:dict|None=None):
        super().__init__()
        self.method=method
        self.field_shape=tuple(inputs.initial.shape[-3:])
        self.field_width=math.prod(self.field_shape)
        self.static_stop=inputs.blocks["source_nuisance"][1]
        self.family_block=inputs.blocks["physical_domain"]
        self.scales=config["native_controlled_comparators"]["conditioning"]["program_reference_scales"]
        static_width=self.static_stop+self.family_block[1]-self.family_block[0]
        sites=math.prod(self.field_shape[:2])
        self.hidden=int(settings["hidden_dim"])
        self.static_projection=nn.Linear(static_width,self.hidden)
        self.output_projection=nn.Linear(self.hidden,self.field_width)
        self.program_width=sites*(3+len(FORCING_ARRAY_NAMES))
        self.parameter_match=None
        if method=="gru":
            self.initial_projection=nn.Linear(self.field_width,self.hidden)
            self.recurrent=nn.GRUCell(self.field_width+self.program_width,self.hidden)
        elif method=="flattened_parameter_matched":
            self.input_projection=nn.Linear(self.field_width+self.program_width+self.field_width,self.hidden)
            depth=int(settings["depth"]);heads=int(settings["head_count"])
            if self.hidden%heads:raise ValueError("Flattened operator width must divide into attention heads")
            if not conditioning or not conditioning.get("reference_total_parameter_count"):
                raise ValueError("Parameter-matched operator requires the selected mainline's measured parameter count")
            reference=int(conditioning["reference_total_parameter_count"])
            shared=int(conditioning["shared_inference_parameter_count"])
            target=reference-shared
            # TransformerEncoderLayer has attention4h^2+4h, twoLayerNorm4h,
            # and FF(2h+1)f+h. All fitted parameters contribute to the predictor.
            outer=sum(p.numel() for p in self.parameters())
            fixed=outer+depth*(4*self.hidden*self.hidden+9*self.hidden)
            unit=depth*(2*self.hidden+1)
            feedforward=round((target-fixed)/unit)
            if feedforward<1:raise ValueError("Reference count cannot support the declared flattened architecture")
            layer=nn.TransformerEncoderLayer(self.hidden,heads,feedforward,
                dropout=float(settings["dropout_rate"]),activation="gelu",batch_first=True,norm_first=True)
            self.temporal=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
            actual=sum(p.numel() for p in self.parameters())
            error=abs(actual-target)/target
            if error>float(settings["parameter_match_relative_tolerance"]):
                raise ValueError("Flattened comparator misses its predeclared parameter match tolerance")
            self.parameter_match={"reference_total":reference,"shared_inference":shared,
                "target_dynamics_parameters":target,"actual_dynamics_parameters":actual,
                "feedforward_width":feedforward,"relative_error":error}
        else:raise ValueError("Unsupported native sequence conditional-mean method")

    def predict_native(self,inputs,normalization):
        if tuple(inputs.initial.shape[-3:])!=self.field_shape or inputs.normalization_id!=normalization.identity:
            raise ValueError("Native sequence comparator domain or normalization mismatch")
        b,j,a=inputs.group_shape;n=b*j*a
        contexts=joint_particle_contexts(inputs.posterior,inputs.programs,inputs.regimen_uids)
        def stack(name):
            values=torch.stack([row[name] for row in contexts],1)
            return values.reshape(n,*values.shape[3:])
        initial=inputs.initial.reshape(n,self.field_width)
        support=inputs.validity.flatten(2)
        if not torch.equal(support,support[:,:1].expand_as(support)):
            raise ValueError("Native sequence model requires the registered time-invariant quantity support")
        lo,hi=self.family_block
        static=torch.cat((inputs.condition[:,:self.static_stop],inputs.condition[:,lo:hi]),-1)
        context=self.static_projection(static)
        program=torch.cat([stack(name)/self.scales[name] for name in
            ("occupancy","support","physical_time",*FORCING_ARRAY_NAMES)],-1).flatten(2)
        times=program.shape[1]
        if self.method=="gru":
            state=initial;hidden=self.initial_projection(initial)+context;values=[initial]
            for ti in range(1,times):
                interval=program[:,ti-1].reshape(n,-1,3+len(FORCING_ARRAY_NAMES)).clone()
                interval[...,2]=program[:,ti].reshape_as(interval)[...,2]
                hidden=self.recurrent(torch.cat((state,interval.flatten(1)),-1),hidden)
                state=torch.where(support[:,ti],state+self.output_projection(hidden),0)
                values.append(state)
            predicted=torch.stack(values,1)
        else:
            # Same field/program/support information, one flattened state per time;
            # no graph-spectral, mechanism-specific or spatial attention modules.
            tokens=torch.cat((initial[:,None].expand(-1,times,-1),program,support.to(initial)),dim=-1)
            hidden=self.input_projection(tokens)+context[:,None]
            causal=torch.ones(times,times,device=hidden.device,dtype=torch.bool).triu(1)
            encoded=self.temporal(hidden,mask=causal)
            predicted=initial[:,None]+self.output_projection(encoded)
            predicted=torch.cat((initial[:,None],predicted[:,1:]),1)
            predicted=torch.where(support,predicted,0)
        predicted=predicted.reshape_as(inputs.validity)
        physical=normalization.inverse(predicted,inputs.validity)
        original=inputs.posterior.support.initial_field[:,:,None].expand(b,j,a,*self.field_shape).reshape(n,*self.field_shape)
        return torch.cat((original[:,None],physical[:,1:]),1)
