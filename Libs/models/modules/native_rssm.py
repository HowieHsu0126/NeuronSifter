"""Native vector-observation RSSM prediction components.

Gaussian core: PlaNet (PMLR 97, 2019) and Dreamer (arXiv:1912.01603v3).
The matched prediction profiles share one fitted identity. No planner, reward
model, actor or critic is supplied by this observation/transition adapter.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from Libs.utils.data.native_diffusion_data import joint_particle_contexts
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES


def gaussian_kl(posterior, prior):
    mean_q, std_q = posterior
    mean_p, std_p = prior
    return (torch.log(std_p / std_q) +
        (std_q.square() + (mean_q - mean_p).square()) / (2 * std_p.square()) - .5).sum(-1)


def weighted_free_nats(kl, weights, free_nats):
    """Original Gaussian RSSM reduction: mean first, then free-nats floor."""
    return ((kl * weights).sum() / weights.sum()).clamp_min(free_nats)


def gaussian_draw(stats, generator):
    mean, scale = stats
    return mean + scale * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)


def gaussian_mlp(inputs,hidden,layers):
    modules=[]
    for _ in range(layers):
        modules.extend((nn.Linear(inputs,hidden),nn.ELU()))
        inputs=hidden
    return nn.Sequential(*modules)


class NativeRSSMBase(nn.Module):
    def __init__(self,settings,inputs,config):
        super().__init__()
        self.settings=dict(settings)
        self.field_shape=tuple(inputs.initial.shape[-3:])
        self.field_width=math.prod(self.field_shape)
        self.static_stop=inputs.blocks["source_nuisance"][1]
        self.family_block=inputs.blocks["physical_domain"]
        self.scales=config["native_controlled_comparators"]["conditioning"]["program_reference_scales"]
        self.static_width=self.static_stop+self.family_block[1]-self.family_block[0]
        self.program_width=math.prod(self.field_shape[:2])*(3+len(FORCING_ARRAY_NAMES))

    def sequence_inputs(self, inputs):
        if tuple(inputs.initial.shape[-3:]) != self.field_shape:
            raise ValueError("RSSM native field shape differs from its fitted profile")
        b,j,a = inputs.group_shape
        n = b*j*a
        contexts = joint_particle_contexts(inputs.posterior, inputs.programs, inputs.regimen_uids)
        def stack(name):
            value = torch.stack([context[name] for context in contexts],1)
            return value.reshape(n,*value.shape[3:])
        programs = torch.cat([stack(name)/float(self.scales[name]) for name in
            ("occupancy","support","physical_time",*FORCING_ARRAY_NAMES)],-1).flatten(2)
        valid = inputs.validity.flatten(2)
        if (programs.shape[:2] != valid.shape[:2] or not torch.isfinite(programs).all()
                or not torch.equal(valid,valid[:,:1].expand_as(valid))):
            raise ValueError("RSSM requires finite programs on the declared time-invariant support")
        initial = inputs.initial.reshape(n,self.field_width)
        lo,hi = self.family_block
        static = torch.cat((inputs.condition[:,:self.static_stop],inputs.condition[:,lo:hi]),-1)
        return initial,valid,static,programs

    def interval(self, programs, ti):
        n=programs.shape[0]
        value=programs[:,ti-1].reshape(n,-1,3+len(FORCING_ARRAY_NAMES)).clone()
        # Left-endpoint action/forcing, physical interval endpoint clock.
        value[...,2]=programs[:,ti].reshape_as(value)[...,2]
        return value.flatten(1)

class NativeGaussianRSSM(NativeRSSMBase):
    prediction_kind="latent_and_unit_gaussian_observation_distribution"

    def __init__(self,settings:dict,inputs,config:dict):
        super().__init__(settings,inputs,config)
        self.deter=int(settings["deterministic_size"])
        self.stoch=int(settings["stochastic_size"])
        hidden=int(settings["hidden_size"])
        self.initial_context=nn.Linear(self.static_width,self.deter)
        self.observation_encoder=gaussian_mlp(2*self.field_width,hidden,int(settings["encoder_layers"]))
        self.transition_input=nn.Sequential(nn.Linear(self.stoch+self.program_width,hidden),nn.ELU())
        self.recurrent=nn.GRUCell(hidden,self.deter)
        self.prior_head=nn.Sequential(nn.Linear(self.deter,hidden),nn.ELU(),nn.Linear(hidden,2*self.stoch))
        self.posterior_head=nn.Sequential(nn.Linear(self.deter+hidden,hidden),nn.ELU(),nn.Linear(hidden,2*self.stoch))
        self.decoder=nn.Sequential(gaussian_mlp(self.deter+self.stoch,hidden,int(settings["decoder_layers"])),
                                   nn.Linear(hidden,self.field_width))
        if (min(self.deter,self.stoch,hidden)<1 or settings["min_stddev"]<=0
                or settings["observation_stddev"]!=1 or settings["free_nats"]<0
                or settings["divergence_weight"]<=0 or settings["overshooting_distance"]!=0):
            raise ValueError("Gaussian profile needs positive dimensions, unit observation covariance and no overshooting")

    def stats(self, values):
        mean, raw = values.chunk(2,-1)
        return mean, F.softplus(raw) + float(self.settings["min_stddev"])

    def initial(self, initial, mask, static, generator):
        hidden = torch.tanh(self.initial_context(static))
        observation = self.observation_encoder(torch.cat((torch.where(mask,initial,0),mask.to(initial)),-1))
        stats = self.stats(self.posterior_head(torch.cat((hidden,observation),-1)))
        return hidden,gaussian_draw(stats,generator)

    def transition(self, hidden, stochastic, program):
        hidden = self.recurrent(self.transition_input(torch.cat((stochastic,program),-1)),hidden)
        return hidden,self.stats(self.prior_head(hidden))

    def filter_step(self, hidden, observation, mask, generator):
        encoded = self.observation_encoder(torch.cat((torch.where(mask,observation,0),mask.to(observation)),-1))
        stats = self.stats(self.posterior_head(torch.cat((hidden,encoded),-1)))
        return gaussian_draw(stats,generator),stats

    def loss_native(self, inputs, target, valid, weights, generator):
        if not self.training:
            raise ValueError("Future-observation filtering is restricted to RSSM fitting")
        initial,support,static,programs=self.sequence_inputs(inputs)
        target=target.flatten(2);valid=valid.flatten(2)&support
        if (target.shape!=valid.shape or not torch.isfinite(target).all()
                or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0
                or (valid[:,1:].sum((1,2))==0).any()):
            raise ValueError("RSSM objective requires finite complete targets and positive supported row weights")
        hidden,stochastic=self.initial(initial,support[:,0],static,generator)
        reconstruction=[];divergences=[]
        scale=float(self.settings["observation_stddev"])
        constant=math.log(scale)+.5*math.log(2*math.pi)
        for ti in range(1,target.shape[1]):
            hidden,prior=self.transition(hidden,stochastic,self.interval(programs,ti))
            stochastic,posterior=self.filter_step(hidden,target[:,ti],valid[:,ti],generator)
            predicted=self.decoder(torch.cat((hidden,stochastic),-1))
            nll=.5*((predicted-target[:,ti])/scale).square()+constant
            reconstruction.append(torch.where(valid[:,ti],nll,0).sum(-1)/valid[:,ti].sum(-1))
            divergences.append(gaussian_kl(posterior,prior))
        reconstruction=torch.stack(reconstruction,1).mean(1)
        divergences=torch.stack(divergences,1).mean(1)
        value=(reconstruction*weights).sum()/weights.sum()
        return value+float(self.settings["divergence_weight"])*weighted_free_nats(
            divergences,weights,float(self.settings["free_nats"]))

    def sample_native_fields(self, inputs, generator):
        initial,support,static,programs=self.sequence_inputs(inputs)
        hidden,stochastic=self.initial(initial,support[:,0],static,generator)
        values=[initial]
        scale=float(self.settings["observation_stddev"])
        for ti in range(1,programs.shape[1]):
            hidden,prior=self.transition(hidden,stochastic,self.interval(programs,ti))
            stochastic=gaussian_draw(prior,generator)
            mean=self.decoder(torch.cat((hidden,stochastic),-1))
            value=mean+scale*torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=generator)
            values.append(torch.where(support[:,ti],value,0))
        result=torch.stack(values,1).reshape_as(inputs.validity)
        if not torch.isfinite(result).all():raise FloatingPointError("Nonfinite RSSM forecast")
        return result


# Block-GRU adapted from DreamerV3, commit
# e01491fad6434b2245a3b8ca201dd7faedcc458c, dreamerv3/rssm.py.
# Copyright (c) 2024 Danijar Hafner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def symlog(value):
    return value.sign()*torch.log1p(value.abs())


def symexp(value):
    return value.sign()*torch.expm1(value.abs())


def categorical_kl(posterior,prior):
    return (posterior*(posterior.log()-prior.log())).sum((-1,-2))


def categorical_draw(probability,generator,*,straight_through):
    indices=torch.multinomial(probability.reshape(-1,probability.shape[-1]),1,generator=generator)
    draw=F.one_hot(indices.squeeze(-1),probability.shape[-1]).reshape_as(probability).to(probability)
    return draw+probability-probability.detach() if straight_through else draw


class BlockLinear(nn.Module):
    def __init__(self,blocks,inputs,outputs):
        super().__init__()
        if min(blocks,inputs,outputs)<1:raise ValueError("Block linear dimensions must be positive")
        self.weight=nn.Parameter(torch.empty(blocks,outputs,inputs))
        self.bias=nn.Parameter(torch.empty(blocks,outputs))
        for block in self.weight:nn.init.kaiming_uniform_(block,a=math.sqrt(5))
        nn.init.uniform_(self.bias,-1/math.sqrt(inputs),1/math.sqrt(inputs))
    def forward(self,value):
        return torch.einsum("ngi,goi->ngo",value,self.weight)+self.bias


def normalized_mlp(inputs,hidden,layers,epsilon):
    modules=[]
    for _ in range(layers):
        modules.extend((nn.Linear(inputs,hidden),nn.RMSNorm(hidden,eps=epsilon),nn.SiLU()))
        inputs=hidden
    return nn.Sequential(*modules)


class NativeDiscreteRSSM(NativeRSSMBase):
    """Nature-2025 RSSM vector branch, with an explicit latent-only forecast law."""
    prediction_kind="categorical_latent_point_decoder_pushforward_uncalibrated"

    def __init__(self,settings:dict,inputs,config:dict):
        super().__init__(settings,inputs,config)
        self.deter=int(settings["deterministic_size"]);self.stoch=int(settings["stochastic_size"])
        self.classes=int(settings["classes"]);self.blocks=int(settings["blocks"])
        hidden=int(settings["hidden_size"]);units=int(settings["vector_units"])
        self.epsilon=float(settings["rms_epsilon"])
        if (min(self.deter,self.stoch,hidden,units,self.classes,self.blocks)<1 or self.deter%self.blocks
                or not 0<float(settings["unimix"])<1 or settings["free_nats"]<0
                or settings["absolute_posterior"] is not False):
            raise ValueError("Discrete RSSM needs a valid block/category/unimix profile and conditional posterior")
        self.initial_context=nn.Linear(self.static_width,self.deter)
        self.observation_encoder=normalized_mlp(2*self.field_width,units,int(settings["vector_layers"]),self.epsilon)
        self.deter_embedding=normalized_mlp(self.deter,hidden,1,self.epsilon)
        self.stoch_embedding=normalized_mlp(self.stoch*self.classes,hidden,1,self.epsilon)
        self.action_embedding=normalized_mlp(self.program_width,hidden,1,self.epsilon)
        per_block=self.deter//self.blocks
        self.core_layers=nn.ModuleList([BlockLinear(self.blocks,
            per_block+3*hidden if index==0 else per_block,per_block)
            for index in range(int(settings["dynamics_layers"]))])
        self.core_norms=nn.ModuleList([nn.RMSNorm(self.deter,eps=self.epsilon) for _ in self.core_layers])
        self.gates=BlockLinear(self.blocks,per_block,3*per_block)
        self.prior_net=normalized_mlp(self.deter,hidden,int(settings["prior_layers"]),self.epsilon)
        self.prior_logits=nn.Linear(hidden,self.stoch*self.classes)
        self.posterior_net=normalized_mlp(self.deter+units,hidden,int(settings["posterior_layers"]),self.epsilon)
        self.posterior_logits=nn.Linear(hidden,self.stoch*self.classes)
        self.decoder=nn.Sequential(normalized_mlp(self.deter+self.stoch*self.classes,units,
            int(settings["vector_layers"]),self.epsilon),nn.Linear(units,self.field_width))

    def probabilities(self,logits):
        probability=logits.reshape(-1,self.stoch,self.classes).float().softmax(-1)
        mix=float(self.settings["unimix"])
        return probability*(1-mix)+mix/self.classes

    def posterior(self,hidden,observation,mask):
        encoded=self.observation_encoder(torch.cat(
            (symlog(torch.where(mask,observation,0)),mask.to(observation)),-1))
        return self.probabilities(self.posterior_logits(self.posterior_net(torch.cat((hidden,encoded),-1))))

    def initial(self,initial,mask,static,generator):
        hidden=torch.tanh(self.initial_context(static))
        probability=self.posterior(hidden,initial,mask)
        return hidden,categorical_draw(probability,generator,straight_through=self.training)

    def transition(self,hidden,stochastic,program):
        # Native physical clocks/programs already have declared scales. The
        # bounded actor-action clipping in the agent is not applied to these.
        n=hidden.shape[0]
        shared=torch.cat((self.deter_embedding(hidden),self.stoch_embedding(stochastic.flatten(1)),
                          self.action_embedding(program)),-1)
        value=torch.cat((hidden.reshape(n,self.blocks,-1),shared[:,None].expand(-1,self.blocks,-1)),-1)
        for layer,norm in zip(self.core_layers,self.core_norms):
            value=F.silu(norm(layer(value).flatten(1))).reshape(n,self.blocks,-1)
        reset,candidate,update=self.gates(value).chunk(3,-1)
        reset=reset.flatten(1).sigmoid()
        candidate=torch.tanh(reset*candidate.flatten(1))
        update=(update.flatten(1)-1).sigmoid()
        hidden=update*candidate+(1-update)*hidden
        probability=self.probabilities(self.prior_logits(self.prior_net(hidden)))
        return hidden,probability

    def filter_step(self,hidden,observation,mask,generator):
        posterior=self.posterior(hidden,observation,mask)
        return categorical_draw(posterior,generator,straight_through=True),posterior

    def loss_native(self,inputs,target,valid,weights,generator):
        if not self.training:raise ValueError("Future-observation filtering is restricted to RSSM fitting")
        initial,support,static,programs=self.sequence_inputs(inputs)
        target=target.flatten(2);valid=valid.flatten(2)&support
        if (target.shape!=valid.shape or not torch.isfinite(target).all()
                or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0
                or (valid[:,1:].sum((1,2))==0).any()):
            raise ValueError("RSSM objective requires finite complete targets and positive supported row weights")
        hidden,stochastic=self.initial(initial,support[:,0],static,generator)
        reconstruction=[];dynamics=[];representation=[]
        for ti in range(1,target.shape[1]):
            hidden,prior=self.transition(hidden,stochastic,self.interval(programs,ti))
            stochastic,posterior=self.filter_step(hidden,target[:,ti],valid[:,ti],generator)
            predicted=self.decoder(torch.cat((hidden,stochastic.flatten(1)),-1))
            error=(predicted-symlog(target[:,ti])).square()
            reconstruction.append(torch.where(valid[:,ti],error,0).sum(-1)/valid[:,ti].sum(-1))
            dynamics.append(categorical_kl(posterior.detach(),prior).clamp_min(float(self.settings["free_nats"])))
            representation.append(categorical_kl(posterior,prior.detach()).clamp_min(float(self.settings["free_nats"])))
        terms=torch.stack(reconstruction,1).mean(1)
        terms=terms+float(self.settings["dynamics_weight"])*torch.stack(dynamics,1).mean(1)
        terms=terms+float(self.settings["representation_weight"])*torch.stack(representation,1).mean(1)
        return (terms*weights).sum()/weights.sum()

    def sample_native_fields(self,inputs,generator):
        initial,support,static,programs=self.sequence_inputs(inputs)
        hidden,stochastic=self.initial(initial,support[:,0],static,generator)
        values=[initial]
        for ti in range(1,programs.shape[1]):
            hidden,prior=self.transition(hidden,stochastic,self.interval(programs,ti))
            stochastic=categorical_draw(prior,generator,straight_through=False)
            decoded=symexp(self.decoder(torch.cat((hidden,stochastic.flatten(1)),-1)))
            values.append(torch.where(support[:,ti],decoded,0))
        result=torch.stack(values,1).reshape_as(inputs.validity)
        if not torch.isfinite(result).all():raise FloatingPointError("Nonfinite RSSM forecast")
        return result
