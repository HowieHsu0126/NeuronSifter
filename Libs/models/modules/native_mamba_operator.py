"""Paper-based Alias-Free MambaNO for the declared native comparison.

Method source: Zheng et al., NeurIPS 2024, paper 5ee553ec47c31e46a1209bb858b30aa5,
Fig. 1, Algorithm 1, equations 3.13-3.19, Appendix A.3 and E. This implementation
uses public mathematical definitions and licensed PyTorch primitives. It imports
no code from the unbound local MambaNO or selective-scan source copies.

Finite periodic Kaiser-windowed sinc filters and diagonal negative evolution
rates are declared numerical choices. Finite filters approximate the paper's
ideal band limits; no exact continuous-discrete equivalence is asserted.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from Libs.models.modules.native_diffusion_comparators import (
    NativeFieldPacking,_validate_inputs,_weighted_loss,
)


class PeriodicSincResampling(nn.Module):
    """Fixed-support periodic sinc convolution around integer grid resampling."""
    def __init__(self,factor:int,radius:int,beta:float):
        super().__init__()
        if factor!=2 or radius<1 or beta<0:
            raise ValueError("MambaNO uses factor-two finite periodic sinc resampling")
        self.factor=factor
        positions=torch.arange(-radius*factor,radius*factor+1,dtype=torch.float64)
        window=torch.kaiser_window(len(positions),periodic=False,beta=beta,dtype=torch.float64)
        coefficients=torch.sinc(positions/factor)*window
        coefficients=coefficients/coefficients.sum()
        self.register_buffer("kernel",coefficients.float())

    def lowpass(self,value):
        channels=value.shape[1];padding=len(self.kernel)//2
        height,width=value.shape[-2:]
        hi=torch.arange(-padding,height+padding,device=value.device).remainder(height)
        wi=torch.arange(-padding,width+padding,device=value.device).remainder(width)
        value=value.index_select(-2,hi).index_select(-1,wi)
        kernel=self.kernel.to(value)
        horizontal=kernel.view(1,1,1,-1).expand(channels,1,1,-1)
        vertical=kernel.view(1,1,-1,1).expand(channels,1,-1,1)
        return F.conv2d(F.conv2d(value,horizontal,groups=channels),vertical,groups=channels)

    def up(self,value):
        expanded=value.new_zeros(*value.shape[:-2],value.shape[-2]*2,value.shape[-1]*2)
        expanded[...,::2,::2]=value
        return self.lowpass(expanded)*4

    def down(self,value):
        if value.shape[-2]%2 or value.shape[-1]%2:
            raise ValueError("MambaNO downsampling requires an even internal grid")
        return self.lowpass(value)[...,::2,::2]


class BandlimitedActivation(nn.Module):
    def __init__(self,settings):
        super().__init__()
        if settings["activation"]!="silu":
            raise ValueError("Unknown declared MambaNO activation")
        self.resampling=PeriodicSincResampling(2,settings["sinc_radius"],settings["kaiser_beta"])

    def forward(self,value):
        return self.resampling.down(F.silu(self.resampling.up(value)))


def selective_zoh_scan(values,delta,read_in,read_out,rates):
    """Algorithm 1 with the exact diagonal zero-order-hold input integral."""
    if (values.ndim!=3 or delta.shape!=values.shape
            or read_in.shape!=read_out.shape or read_in.shape[:2]!=values.shape[:2]
            or rates.shape!=(values.shape[-1],read_in.shape[-1])
            or not (delta>0).all() or not (rates<0).all()):
        raise ValueError("Selective MambaNO recurrence axes or evolution parameters are invalid")
    hidden=values.new_zeros(values.shape[0],*rates.shape)
    output=[]
    for ti in range(values.shape[1]):
        scaled=delta[:,ti,:,None]*rates[None]
        decay=torch.exp(scaled)
        # expm1 avoids cancellation at the small source-independent scan scales.
        drive=torch.expm1(scaled)/rates[None]
        hidden=decay*hidden+drive*read_in[:,ti,None,:]*values[:,ti,:,None]
        output.append((hidden*read_out[:,ti,None,:]).sum(-1))
    return torch.stack(output,dim=1)


class SelectiveDirection(nn.Module):
    def __init__(self,width:int,state_width:int,delta_rank:int):
        super().__init__()
        self.delta_in=nn.Linear(width,delta_rank,bias=False)
        self.delta_out=nn.Linear(delta_rank,width)
        self.read_in=nn.Linear(width,state_width,bias=False)
        self.read_out=nn.Linear(width,state_width,bias=False)
        self.log_rates=nn.Parameter(torch.arange(1,state_width+1).float().log().expand(width,-1).clone())

    def forward(self,sequence):
        delta=F.softplus(self.delta_out(self.delta_in(sequence)))
        return selective_zoh_scan(sequence,delta,self.read_in(sequence),self.read_out(sequence),
                                  -self.log_rates.exp())


class MambaIntegration(nn.Module):
    def __init__(self,width:int,settings:dict):
        super().__init__()
        inner=width*settings["expansion"]
        self.norm=nn.LayerNorm(width)
        self.split=nn.Conv2d(width,2*inner,1)
        self.local=nn.Conv1d(inner,inner,settings["local_kernel"],
            padding=settings["local_kernel"]//2,groups=inner)
        self.activation=BandlimitedActivation(settings)
        rank=max(1,math.ceil(inner/settings["delta_rank_divisor"]))
        self.scans=nn.ModuleList([SelectiveDirection(inner,settings["state_width"],rank) for _ in range(4)])
        self.output_norm=nn.LayerNorm(inner)
        self.project=nn.Conv2d(inner,width,1)

    def forward(self,value):
        normalized=self.norm(value.movedim(1,-1)).movedim(-1,1)
        content,gate=self.split(normalized).chunk(2,dim=1)
        content=self.local(content.flatten(2)).reshape_as(content)
        content=self.activation(content)
        height,width=content.shape[-2:]
        row=content.flatten(2).transpose(1,2)
        column=content.transpose(-2,-1).flatten(2).transpose(1,2)
        traversal=(row,column,row.flip(1),column.flip(1))
        recovered=[]
        for index,(scan,sequence) in enumerate(zip(self.scans,traversal)):
            result=scan(sequence)
            if index>=2:
                result=result.flip(1)
            if index%2:
                result=result.transpose(1,2).reshape(*content.shape[:2],width,height).transpose(-2,-1)
            else:
                result=result.transpose(1,2).reshape_as(content)
            recovered.append(result)
        combined=self.output_norm(torch.stack(recovered).sum(0).movedim(1,-1)).movedim(-1,1)
        return value+self.project(combined*self.activation(gate))


class ConvolutionIntegration(nn.Module):
    def __init__(self,width,settings):
        super().__init__()
        size=settings["local_kernel"]
        self.first=nn.Conv2d(width,width,size,padding=size//2)
        self.activation=BandlimitedActivation(settings)
        self.second=nn.Conv2d(width,width,size,padding=size//2)

    def forward(self,value):
        return value+self.second(self.activation(self.first(value)))


class PaperMambaNO(nn.Module):
    """Native controlled adapter retaining the paper's local/global U architecture."""
    def __init__(self,entry,shape,condition_width):
        super().__init__()
        settings=entry["hyperparameters"]
        if (settings["depth"]<1 or settings["local_kernel"]%2!=1
                or len(settings["encoder_convolutions"])!=settings["depth"]
                or len(settings["encoder_integrations"])!=settings["depth"]
                or len(settings["decoder_integrations"])!=settings["depth"]):
            raise ValueError("MambaNO needs explicit U-level integration counts")
        self.condition_width=condition_width
        self.packing=NativeFieldPacking(shape,2**(settings["depth"]-1))
        self.condition_projection=nn.Linear(condition_width,settings["conditioning_channels"])
        widths=[settings["width"]*2**level for level in range(settings["depth"])]
        self.lift=nn.Conv2d(shape[-1]+settings["conditioning_channels"],widths[0],1)
        self.convolutions=nn.ModuleList()
        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        self.down_projection=nn.ModuleList()
        self.up_projection=nn.ModuleList()
        self.resampling=PeriodicSincResampling(2,settings["sinc_radius"],settings["kaiser_beta"])
        self.transfer_activation=BandlimitedActivation(settings)
        for level,width in enumerate(widths):
            self.convolutions.append(nn.Sequential(*[
                ConvolutionIntegration(width,settings) for _ in range(settings["encoder_convolutions"][level])]))
            self.encoders.append(nn.Sequential(*[
                MambaIntegration(width,settings) for _ in range(settings["encoder_integrations"][level])]))
            self.decoders.append(nn.Sequential(*[
                MambaIntegration(width,settings) for _ in range(settings["decoder_integrations"][level])]))
            if level<len(widths)-1:
                self.down_projection.append(nn.Conv2d(width,widths[level+1],1))
                self.up_projection.append(nn.Conv2d(widths[level+1],width,1))
        self.project=nn.Conv2d(widths[0],shape[-1],1)

    def predict(self,support,condition):
        _validate_inputs(torch.zeros_like(support,dtype=condition.dtype),support,condition,self.condition_width)
        t,r,c,s=self.packing.shape
        initial=condition[:,:r*c*s].reshape(-1,1,r,c,s).expand(-1,t,-1,-1,-1)
        field=self.packing.pack(torch.where(support,initial,0))
        auxiliary=self.condition_projection(condition)[:,:,None,None].expand(-1,-1,*field.shape[-2:])
        value=self.lift(torch.cat((field,auxiliary),dim=1))
        skips=[]
        for level,(local,global_operator) in enumerate(zip(self.convolutions,self.encoders)):
            value=local(value);skips.append(value);value=global_operator(value)
            if level<len(self.down_projection):
                value=self.resampling.down(self.transfer_activation(self.down_projection[level](value)))
        for level in reversed(range(len(self.decoders))):
            if level<len(self.decoders)-1:
                value=self.resampling.up(self.transfer_activation(self.up_projection[level](value)))+skips[level]
            value=self.decoders[level](value)
        prediction=self.packing.unpack(self.project(value))
        if not torch.isfinite(prediction[support]).all():
            raise FloatingPointError("Nonfinite paper-based MambaNO prediction")
        return torch.where(support,prediction,0)

    def loss(self,target,support,condition,row_weight,generator):
        _validate_inputs(target,support,condition,self.condition_width)
        return _weighted_loss((self.predict(support,condition)-target).square(),support,row_weight)
