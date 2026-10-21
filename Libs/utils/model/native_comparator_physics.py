"""Differentiable guidance from the frozen training equations and visible inputs.

No source/holdout identifier selects an equation. The finite-difference residual
uses original physical clocks; coarse-grid truncation error remains a comparator
limitation, not a new teacher. This module never integrates or reads a target.
"""
from __future__ import annotations

import copy
import math
import torch

from Libs.utils.model.controlled_cns_source import DYNAMIC
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, MOLECULAR


def auxiliary_step(family: str, state: torch.Tensor, parameters: dict,
                   interval: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
    """One step of the declared original discrete law, on a predicted state."""
    p, h = parameters, interval
    if family == ELECTRICAL:
        voltage, calcium, _ = state.unbind(-1)
        voltage = voltage+h*(-p["membrane_leak_rate"]*(voltage-p["resting_membrane_voltage"])
                             +p["stimulation_gain"]*stimulus)
        spike = ((voltage-p["spike_threshold"])*p["spike_rate_gain"]).clamp_min(0)
        calcium = calcium+h*(-p["calcium_decay_rate"]*(calcium-.10)+p["spike_calcium_gain"]*spike)
        return torch.stack((voltage,calcium,spike),-1)
    if family == SYNAPTIC:
        calcium, _, vesicle, current = state.unbind(-1)
        calcium = calcium+h*(-p["calcium_clearance_rate"]*(calcium-p["baseline_intracellular_calcium"])
                             +p["presynaptic_calcium_gain"]*stimulus)
        release = (p["initial_release_probability"]+p["release_coupling"]*(
            calcium-p["baseline_intracellular_calcium"]).clamp_min(0)).clamp(0,1)
        released = torch.minimum(vesicle,h*p["vesicle_release_rate"]*release*vesicle)
        new_vesicle = (vesicle+h*p["vesicle_recovery_rate"]*(1-vesicle)-released).clamp(0,1)
        current = current+p["postsynaptic_current_gain"]*released-h*p["postsynaptic_decay_rate"]*current
        return torch.stack((calcium,release,new_vesicle,current),-1)
    if family == MOLECULAR:
        exposure, occupancy, signal, response = state.unbind(-1)
        exposure = (exposure+h*(p["dose_to_exposure_gain"]*stimulus-p["exposure_clearance_rate"]*exposure
                               -p["feedback_rate"]*response*exposure)).clamp_min(0)
        equilibrium = exposure/(exposure+p["binding_affinity"])
        occupancy = (occupancy+h*p["occupancy_relaxation_rate"]*(equilibrium-occupancy)).clamp(0,1)
        signal = (signal+h*(p["signal_activation_rate"]*occupancy*(1-signal)-p["signal_decay_rate"]*signal)).clamp(0,1)
        response = (response+h*(p["biologic_response_rate"]*signal*(1-response)-p["biologic_clearance_rate"]*response)).clamp(0,1)
        return torch.stack((exposure,occupancy,signal,response),-1)
    raise ValueError("No declared discrete auxiliary law for "+family)


class FrozenNativePhysicsGuidance:
    def __init__(self, inputs, normalization, training_config: dict):
        settings = training_config["native_controlled_comparators"]["guidance"]
        if settings["equation_authority"] != "frozen_primary_training_law":
            raise ValueError("Guidance requires the frozen training equation authority")
        self.inputs, self.normalization = inputs, normalization
        self.planning_only = False
        self.source = copy.deepcopy(training_config["controlled_cns_source"])
        self.coupling = copy.deepcopy(training_config["native_observable_bridge"]["coupling"])
        self.family = inputs.grid["family_uid"]
        self.bind_source_intervals = training_config["native_observable_bridge"].get("bind_source_update_intervals", False)
        self.vm_coupling = copy.deepcopy(training_config.get("native_romani_source", {}).get("coupling"))
        self.grid = inputs.grid
        self.b,self.j,self.a = inputs.group_shape
        self.n = self.b*self.j*self.a
        self.support = inputs.posterior.support
        self.device = inputs.condition.device
        self.dynamic_indices = [self.grid["species_id"].index(name) for name in DYNAMIC]
        self.ca1 = self.grid["region_id"].index("hippocampus_ca1")*6+self.grid["compartment_id"].index("isf")
        self.ca3 = self.grid["region_id"].index("hippocampus_ca3")*6+self.grid["compartment_id"].index("isf")
        self.times = self.tensor(self.grid["time_hours"])
        self.clock = self.tensor(self.grid["query_time"]["values"])
        self.scale = normalization.scale.to(device=self.device,dtype=torch.float64)
        self.parameters = {}
        declared = self.support.physical_parameter_hypotheses
        for index,row in enumerate(declared["declarations"]):
            self.parameters[row["name"]] = self.rows(declared["values"][...,index]).double()
        self.factors = {name:self.rows(value).double() for name,value in self.support.factors.items()}
        self.volumes = self.tensor([row["volume_l"] for row in self.source["sites"]])[None]*self.site_factor("z_morph")
        self.edges = torch.tensor([[row["left"],row["right"]] for row in self.source["edges"]],
                                  device=self.device,dtype=torch.long)
        left,right = self.edges.T
        self.conductance = self.tensor([row["conductance_l_per_hour"] for row in self.source["edges"]])[None]*(
            self.site_factor("z_morph")[:,left]*self.site_factor("z_morph")[:,right]).sqrt()
        self.transport_scale = self.tensor([self.source["transport_multipliers"][name] for name in DYNAMIC])
        self.elimination = self.tensor([math.log(2)/row["pk_half_life_hours"]
                                       for row in self.source["sites"]])[None]*self.site_factor("z_pk",1)
        self.bmax = self.tensor([row["bmax_mol_per_l"] for row in self.source["sites"]])[None]*self.site_factor("z_pk")
        self.occupancy = self.tensor(inputs.programs["occupancy"]).reshape(self.n,-1,36)
        self.free = torch.zeros_like(self.occupancy)
        flat_records = [record for sample in inputs.programs["records"] for particle in sample for record in particle]
        for index,record in enumerate(flat_records):
            if "physical_program" in record:
                self.free[index] = self.tensor(record["physical_program"]["free_concentration_mol_per_l"]).reshape(-1,36)
        self.auxiliary_indices, self.auxiliary_nodes = [], []
        for index,quantity in enumerate(self.grid["quantity_profile"]["quantities"]):
            if quantity.get("family_uid") == self.family:
                site=quantity["incidence"]
                node=self.grid["region_id"].index(site["anchor_region_id"])*6+self.grid["compartment_id"].index(site["anchor_compartment_id"])
                self.auxiliary_indices.append(index);self.auxiliary_nodes.append(node)
        if self.family == ELECTRICAL:
            self.stimulus = self.tensor(inputs.programs["electrical_current_uA"]).reshape(self.n,-1,36)[:,:,self.ca1]
        elif self.family == SYNAPTIC:
            self.stimulus = self.tensor(inputs.programs["optical_power_mW"]).reshape(self.n,-1,36)[:,:,self.ca3]
        elif self.family == MOLECULAR:
            # Invert the declared estimated-PK recurrence, not the true response.
            # This consumes exactly the visible compiled exposure program.
            exposure=self.free[:,:,self.ca1]*1e9
            gain=self.parameters["dose_to_exposure_gain"][:,None]
            if (gain <= 0).any():raise ValueError("Estimated-PK input requires positive declared exposure gain")
            level=((exposure[:,1:]-exposure[:,:-1])/self.clock.diff()[None]
                   +self.parameters["exposure_clearance_rate"][:,None]*exposure[:,:-1])/gain
            if (level < -1e-5).any():raise ValueError("Compiled exposure does not bind the declared estimated-PK input")
            self.stimulus=torch.cat((level.new_zeros(self.n,1),level.clamp_min(0)),1)
        self.vm_only = self.family == ELECTRICAL and "somatic_sodium_conductance" in self.parameters
        if self.vm_only and self.vm_coupling is None:
            raise ValueError("Vm-only source requires the declared training Vm-to-CNS coupling")
        self.selector = copy.deepcopy(self.source["observation"]["selector"])
        observation = inputs.observations[0]
        if len(self.selector) != observation.values.shape[-1]:
            raise ValueError("Guidance observation selector width differs from retained readouts")
        self.observed = observation.observed[:,None,None].expand(self.b,self.j,self.a,-1).reshape(self.n,-1)
        self.observation = self.rows(observation.values[:,None].expand(self.b,self.j,-1))
        likelihood=self.support.readout_parameters[observation.channel_uid]
        self.variance=self.rows(likelihood["variance"])
        self.nuisance=self.rows(self.support.source_nuisance[observation.channel_uid])
        self.initial=self.rows(self.support.initial_field)
        self.initial_mask=inputs.validity[:,0]
        self.protocol = {"equation_authority":"frozen_primary_training_law",
            "readout_authority":"frozen_training_predose_selector",
            "graph_authority":"frozen_training_edges",
            "source_variant_selection":False,"future_target_access":False,
            "residual_discretization":"forward_difference_on_original_clock",
            "truncation_error":"retained_not_a_teacher_or_precision_claim",
            "auxiliary_dynamics_residual":"unavailable_unobserved_HOC_gating_states" if self.vm_only else "frozen_discrete_training_law"}

    def fork_executed_columns(self, executed_index: int, candidate_count: int):
        """Private model-side guide; each future candidate inherits one real prefix."""
        result = copy.copy(self)
        result.inputs = None
        result.planning_only = True
        def select(value):
            return value.reshape(self.b, self.j, self.a, *value.shape[1:])[:, :, executed_index:executed_index+1].expand(
                self.b, self.j, candidate_count, *value.shape[1:]).reshape(self.b*self.j*candidate_count, *value.shape[1:]).clone()
        result.parameters = {key: select(value) for key, value in self.parameters.items()}
        result.factors = {key: select(value) for key, value in self.factors.items()}
        for name in ("volumes", "conductance", "elimination", "bmax", "initial", "initial_mask",
                     "observed", "observation", "variance", "nuisance"):
            setattr(result, name, select(getattr(self, name)))
        result.occupancy = result.free = None  # No stale original conditioning/readback curves.
        result.a, result.n = candidate_count, self.b*self.j*candidate_count
        return result

    def tensor(self,value):
        return torch.as_tensor(value,device=self.device,dtype=torch.float64)

    def rows(self,value):
        return value[:,:,None].expand(self.b,self.j,self.a,*value.shape[2:]).reshape(self.n,*value.shape[2:]).to(self.device)

    def site_factor(self,name,shift=0):
        value=self.factors[name]
        return value[:,(torch.arange(36,device=self.device)+shift)%value.shape[-1]]

    def transport(self,x,scale=None):
        left,right=self.edges.T
        multipliers=self.transport_scale if scale is None else scale
        flux=self.conductance[:,None,:,None]*(x[:,:,right]-x[:,:,left])*multipliers
        amount=torch.zeros_like(x).index_add(2,left,flux).index_add(2,right,-flux)
        return amount/self.volumes[:,None,:,None],flux

    def reaction(self,x,occupancy):
        rate,K=self.source["base_rates_per_hour"],self.source["saturation_constants"]
        env,tx,ep=(self.site_factor(name)[:,None] for name in ("z_env","z_tx","z_ephys"))
        sat=lambda value,key:value/(K[key]+value)
        m,a,I,circuit,ab,plaque,tau,med,ca,v,activity=x.unbind(-1)
        zero=torch.zeros_like(m)
        production=[zero for _ in DYNAMIC]; removal=[zero for _ in DYNAMIC];paired=[zero for _ in DYNAMIC]
        production[0]=rate["misfolded_production"]*env*(1+self.source["inflammatory_feedback"]*sat(I,"inflammation"))
        removal[0]=rate["misfolded_clearance"]/env*m
        production[4]=rate["amyloid_production"]*env*(1+self.source["inflammatory_feedback"]*sat(I,"inflammation"))*(
            1-self.source["occupancy_production_inhibition"]*tx*occupancy)
        removal[4]=rate["amyloid_clearance"]/env*ab*(1+self.source["occupancy_clearance_gain"]*tx*occupancy)
        for soluble,aggregate in ((0,1),(4,5)):
            transfer=rate["aggregation"]*x[...,soluble].square()-rate["dissociation"]*x[...,aggregate]
            paired[soluble]=-transfer;paired[aggregate]=transfer
            removal[aggregate]=rate["aggregate_clearance"]/env*x[...,aggregate]
        production[6]=rate["tau_production"]*env*(1+sat(plaque,"pathology")+sat(I,"inflammation"))
        removal[6]=rate["tau_clearance"]/env*tau
        production[7]=rate["mediator_production"]*env*(1+sat(plaque,"pathology")+sat(tau,"pathology"))
        removal[7]=rate["mediator_clearance"]*med
        for index,drive in {2:med,8:I*ep,9:(ca+I)*ep,10:(v+I)*ep,3:(activity+I)*ep}.items():
            name=DYNAMIC[index];base=self.source["response_baselines"][name]
            k=rate[name+"_relaxation"]
            production[index]=k*(base+(1-base)*sat(drive,"response"));removal[index]=k*x[...,index]
        return torch.stack(production,-1),torch.stack(removal,-1),torch.stack(paired,-1)

    def __call__(self,normalized):
        if self.planning_only:
            raise ValueError("Model-side future controls have no recompiled neural-guidance conditioning")
        field=self.normalization.inverse(normalized,self.inputs.validity).double()
        values=field.reshape(self.n,len(self.times),36,-1)
        # The source's state domain is nonnegative for core quantities.
        # Penalize violations while extending its RHS by projection outside it.
        x=values[...,self.dynamic_indices].clamp_min(0)
        terms=[torch.where(self.initial_mask,(field[:,0]-self.initial).square()/self.scale.square(),0).flatten(1).sum(1)/self.initial_mask.flatten(1).sum(1)]
        signed=self.tensor([row["signed"] for row in self.grid["quantity_profile"]["quantities"]]).bool()
        violation=torch.where(self.inputs.validity & ~signed,torch.relu(-field).square()/self.scale.square(),0)
        terms.append(violation.flatten(1).sum(1)/self.inputs.validity.flatten(1).sum(1))
        auxiliary=(torch.stack([values[:,:,node,index] for node,index in zip(
            self.auxiliary_nodes,self.auxiliary_indices)],-1) if self.auxiliary_indices else None)
        occupancy=self.occupancy
        if self.family == MOLECULAR:
            occupancy=torch.zeros_like(occupancy)
            occupancy[:,:,self.ca1]=auxiliary[:,:,1].clamp(0,1)
        production,removal,paired=self.reaction(x,occupancy)
        if self.family in {ELECTRICAL,SYNAPTIC}:
            addition=torch.zeros_like(production);gain=self.coupling["gain"]
            rates=self.source["base_rates_per_hour"]
            if self.vm_only:
                addition[:,:,self.ca1,DYNAMIC.index("membrane_voltage")] = self.vm_coupling["activity_rate_per_hour"]*(
                    (auxiliary[:,:,0]-self.vm_coupling["voltage_reference_mV"])/self.vm_coupling["voltage_scale_mV"]).clamp_min(0)
            elif self.family == ELECTRICAL:
                addition[:,:,self.ca1,8]=rates["calcium_concentration_relaxation"]*gain*auxiliary[:,:,1]/self.coupling["calcium_reference_umol_per_l"]
                addition[:,:,self.ca1,10]=rates["spike_rate_relaxation"]*gain*auxiliary[:,:,2]/self.coupling["spike_reference_hz"]
            else:
                addition[:,:,self.ca3,8]=rates["calcium_concentration_relaxation"]*gain*auxiliary[:,:,0]/self.coupling["calcium_reference_umol_per_l"]
                addition[:,:,self.ca1,9]=rates["membrane_voltage_relaxation"]*gain*auxiliary[:,:,3]/self.coupling["current_reference_pa"]
            production=production+addition
        diffusion,flux=self.transport(x)
        rhs=production-removal+paired+diffusion
        residual=(x[:,1:]-x[:,:-1]-self.times.diff()[None,:,None,None]*rhs[:,:-1])/self.scale[self.dynamic_indices]
        terms.append(residual.square().flatten(1).mean(1))
        clearance=(self.volumes[:,None,:,None]*removal).sum(-1)
        absolute=torch.zeros_like(clearance).index_add(2,self.edges[:,0],flux.abs().sum(-1)/2).index_add(2,self.edges[:,1],flux.abs().sum(-1)/2)
        for name,expected in (("clearance_flux",clearance),("vascular_perivascular_transport",absolute)):
            index=self.grid["species_id"].index(name)
            terms.append(((values[...,index]-expected)/self.scale[index]).square().flatten(1).mean(1))
        if auxiliary is not None and not self.vm_only:
            parameters={name:value[:,None] for name,value in self.parameters.items()}
            expected=auxiliary_step(self.family,auxiliary[:,:-1],parameters,
                                    self.clock.diff()[None], self.stimulus[:,:-1] if self.bind_source_intervals and self.family != MOLECULAR else self.stimulus[:,1:])
            terms.append(((auxiliary[:,1:]-expected)/self.scale[self.auxiliary_indices]).square().flatten(1).mean(1))
        free=values[...,0];bound=values[...,1]
        if self.family == MOLECULAR:
            free_rhs,_=self.transport(free.unsqueeze(-1),self.tensor([1.]))
            free_rhs=free_rhs.squeeze(-1)-self.elimination[:,None]*free
            residual=(free[:,1:]-free[:,:-1]-self.times.diff()[None,:,None]*free_rhs[:,:-1])/self.scale[0]
            keep=torch.arange(36,device=self.device)!=self.ca1
            terms.append(residual[:,:,keep].square().flatten(1).mean(1))
            terms.append(((free[:,:,self.ca1]-auxiliary[:,:,0]*1e-9)/self.scale[0]).square().mean(1))
            expected_bound=torch.zeros_like(bound)
            expected_bound[:,:,self.ca1]=self.bmax[:,None,self.ca1]*auxiliary[:,:,1]
        else:
            expected_free=self.free if auxiliary is None else torch.zeros_like(free)
            expected_bound=self.bmax[:,None]*occupancy if auxiliary is None else torch.zeros_like(bound)
            terms.append(((free-expected_free)/self.scale[0]).square().flatten(1).mean(1))
        terms.append(((bound-expected_bound)/self.scale[1]).square().flatten(1).mean(1))
        predicted=torch.stack([field[:,0,row["region_index"],row["compartment_index"],
            self.grid["species_id"].index(row["species"])]/self.source["reference_scales"][row["species"]]
            for row in self.selector],-1)+self.nuisance
        # Missing source values may be NaN. Mask before arithmetic so the
        # backward pass never multiplies a masked NaN by zero.
        safe_observation=torch.where(self.observed,self.observation,0)
        error=torch.where(self.observed,(predicted-safe_observation).square()/self.variance,0)
        observation=error.sum(-1)/self.observed.sum(-1).clamp_min(1)
        return observation,torch.stack(terms).mean(0)
