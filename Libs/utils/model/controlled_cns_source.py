"""Prospectively defined slow controlled CNS source; no empirical calibration."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import DOP853, solve_ivp
import torch

from Libs.models.modules.world_state import JointParticleSupport, FACTOR_NAMES
from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
from Libs.utils.data.physical_occupancy import compile_physical_record, digest, whole_particle_basis
from Libs.utils.data.native_diffusion_data import JOINT_SOURCE_SCHEMA
from Libs.utils.data.action_observation_contracts import SCHEMA_VERSIONS
from Libs.utils.model.microenvironment_teacher import default_coordinate_payload

DYNAMIC = ("misfolded_protein_load", "aggregate_burden", "neuroinflammation",
           "circuit_instability", "amyloid_beta_soluble", "amyloid_plaque",
           "phosphorylated_tau", "inflammatory_mediator", "calcium_concentration",
           "membrane_voltage", "spike_rate")


def finite_positive(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or not (array > 0).all():
        raise ValueError(name + " requires finite positive values")
    return array


def definition(config: dict) -> dict:
    source = copy.deepcopy(config["controlled_cns_source"])
    grid = default_coordinate_payload(config)
    if (grid != source["grid"] or len(grid["time_hours"]) != 28
            or grid["time_hours"][0] != 0 or grid["time_hours"][-1] != 672
            or [len(grid[k]) for k in ("region_id", "compartment_id", "species_id")] != [6,6,15]):
        raise ValueError("Controlled source requires the exact canonical full grid")
    if source["baseline_intervention_history"] != [] or source["event_convention"] != {
            "initial_output": "pre_dose", "later_events": "post_dose", "integration": "split_at_dose_and_output"}:
        raise ValueError("Controlled source event/history convention changed")
    if source["seed"] != 3407 or source["replicates"] != [0,1,2,3] or set(source["initial_conditions"]) != {"control","ad_positive"}:
        raise ValueError("Controlled source world design changed")
    budget = source["budget"]
    if budget != {"cases":16,"integrations":48,"primary_pk_grids":64,"precision_pk_grids":64,"seconds":1800}:
        raise ValueError("Controlled source production budget changed")
    if source["numerics"] != {"rk4_steps_hours":[1.0,.5],"reference_method":"DOP853",
            "reference_max_step_hours":.25,"reference_rtol":1e-10,"reference_atol_scale":1e-12,
            "normalized_tolerance":1e-3,"balance_tolerance":1e-9,"decimal_precision":50}:
        raise ValueError("Controlled source numerical policy changed")
    if len(source["atoms"]) != 4 or len(source["prior_weights"]) != 4:
        raise ValueError("Four complete correlated atoms required")
    finite_positive(source["coefficient_levels"], "coefficient levels")
    finite_positive(source["initial_jitter"], "initial jitter")
    finite_positive(source["observation"]["standard_deviation"], "observation standard deviation")
    finite_positive([source["pk"][key] for key in ("total_dose_mg_per_kg","body_mass_kg","molecular_weight_g_per_mol")], "PK conversion")
    if len(source["initial_jitter"]) != 2 or source["initial_jitter"][1] < source["initial_jitter"][0]:
        raise ValueError("Initial jitter interval invalid")
    weights = finite_positive(source["prior_weights"], "prior weights")
    if abs(weights.sum()-1) > 1e-12: raise ValueError("Prior weights must sum to one")
    if set(source["factor_basis"]) != set(FACTOR_NAMES): raise ValueError("Five named factor bases required")
    if {k:len(v) for k,v in source["factor_basis"].items()} != {
            "z_tx":config["screening_pipeline"]["screening_benchmark"]["training"]["latent_dim"],
            "z_pk":2,"z_env":9,"z_ephys":4,
            "z_morph":config["screening_pipeline"]["screening_benchmark"]["training"]["latent_dim"]}:
        raise ValueError("Factor basis differs from the maintained native widths")
    for key in ("reference_scales","base_rates_per_hour","saturation_constants","response_baselines"):
        finite_positive(list(source[key].values()),key)
    if set(source["reference_scales"]) != set(grid["species_id"]):raise ValueError("Missing channel reference scale")
    if set(source["channel_definitions"]) != set(grid["species_id"]):raise ValueError("Missing source channel meaning")
    if len(source["sites"]) != 36 or len({v["site_uid"] for v in source["sites"]}) != 36:
        raise ValueError("Complete distinct source site table required")
    for site in source["sites"]:
        finite_positive([site[k] for k in ("volume_l","delivery_fraction","pk_half_life_hours","kd_mol_per_l","bmax_mol_per_l")],"site parameters")
    for edge in source["edges"]:
        finite_positive(edge["conductance_l_per_hour"],"edge conductance")
        if edge["left"] == edge["right"] or not all(0 <= edge[k] < 36 for k in ("left","right")):raise ValueError("Invalid source edge")
    for condition in source["initial_conditions"].values():
        if set(condition) != set(DYNAMIC):raise ValueError("Initial condition channels incomplete")
        finite_positive(list(condition.values()),"initial condition")
    for action in source["actions"]:
        if not action["dose_times_hours"] or len(action["dose_times_hours"]) != len(action["fractions"]):
            raise ValueError("Invalid dose schedule")
        finite_positive(action["fractions"],"dose fractions")
        if sum(action["fractions"]) != 1.:raise ValueError("Actions must share total amount")
    if [v["dose_times_hours"] for v in source["actions"]] != [[0.],[0.,336.]]:
        raise ValueError("Source action histories changed")
    return source


def source_grid(source: dict) -> dict:
    return dict(source["grid"], physical_time_unit="hour",
                species_units=source["species_units"],normalization={"kind":"physical"})


def transport(state: np.ndarray, volumes: np.ndarray, edges: np.ndarray,
              conductance: np.ndarray, species_scale: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    flux = conductance[:,None] * (state[edges[:,1]]-state[edges[:,0]]) * species_scale
    amount = np.zeros_like(state)
    np.add.at(amount,edges[:,0],flux)
    np.add.at(amount,edges[:,1],-flux)
    return amount/volumes[:,None], flux


def pk_values(times, parameters: dict, action: dict, source: dict, *, high_precision=False,
              initial_pre_dose=True, active_through=None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    times = np.asarray(times,dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (times<0).any():raise ValueError("Invalid PK clock")
    dose = source["pk"]["total_dose_mg_per_kg"]*source["pk"]["body_mass_kg"]*1e-3/source["pk"]["molecular_weight_g_per_mol"]
    # Composite continuations retain each original total and fraction separately:
    # changing multiplication order to a pre-multiplied amount changes source rounding.
    totals = action.get("event_total_dose_mg_per_kg")
    if totals is not None and (len(totals) != len(action["dose_times_hours"])
            or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in totals)):
        raise ValueError("Per-event public dose totals must be finite, positive and aligned")
    history = [(t, f, None if totals is None else totals[i])
               for i, (t, f) in enumerate(zip(action["dose_times_hours"], action["fractions"]))]
    history = [(t, f, total) for t, f, total in history if active_through is None or t <= active_through]
    C=np.zeros((len(times),len(parameters["volume_l"])))
    K=np.asarray(parameters["kd_mol_per_l"]);B=np.asarray(parameters["bmax_mol_per_l"])
    if high_precision:
        with localcontext() as ctx:
            ctx.prec=source["numerics"]["decimal_precision"]
            D=lambda x:Decimal(str(float(x)))
            q=D(source["pk"]["total_dose_mg_per_kg"])*D(source["pk"]["body_mass_kg"])*Decimal("0.001")/D(source["pk"]["molecular_weight_g_per_mol"])
            O=np.zeros_like(C);bound=np.zeros_like(C)
            for ti,t in enumerate(times):
                for si in range(C.shape[1]):
                    value=Decimal(0)
                    if not (initial_pre_dose and t==0):
                        for event,fraction,total in history:
                            if t>=event:
                                if total is not None:
                                    q=D(total)*D(source["pk"]["body_mass_kg"])*Decimal("0.001")/D(source["pk"]["molecular_weight_g_per_mol"])
                                value+=D(parameters["delivery_fraction"][si])*q*D(fraction)/D(parameters["volume_l"][si])*(-D(parameters["elimination_per_hour"][si])*(D(t)-D(event))).exp()
                    occ=value/(D(K[si])+value)
                    C[ti,si]=float(value);O[ti,si]=float(occ);bound[ti,si]=float(D(B[si])*occ)
            return C,O,bound
    for event,fraction,total in history:
        if total is not None:
            dose=total*source["pk"]["body_mass_kg"]*1e-3/source["pk"]["molecular_weight_g_per_mol"]
        lag=times-event
        selected=lag>=0
        if initial_pre_dose:selected &= times!=0
        C[selected]+=np.asarray(parameters["delivery_fraction"])[None,:]*dose*fraction/np.asarray(parameters["volume_l"])[None,:]*np.exp(-lag[selected,None]*np.asarray(parameters["elimination_per_hour"])[None,:])
    O=C/(C+K[None,:])
    return C,O,B[None,:]*O


@dataclass
class ControlledLaw:
    source: dict
    factors: dict
    initial: np.ndarray

    def __post_init__(self):
        self.volumes=np.array([v["volume_l"] for v in self.source["sites"]])*self.site_factor("z_morph")
        self.edges=np.array([[e["left"],e["right"]] for e in self.source["edges"]],dtype=int)
        self.conductance=np.array([e["conductance_l_per_hour"] for e in self.source["edges"]])*np.sqrt(
            self.site_factor("z_morph")[self.edges[:,0]]*self.site_factor("z_morph")[self.edges[:,1]])
        self.transport_scale=np.array([self.source["transport_multipliers"][k] for k in DYNAMIC])
        self.pk={"volume_l":self.volumes*self.site_factor("z_pk"),
                 "delivery_fraction":np.array([s["delivery_fraction"] for s in self.source["sites"]])*self.site_factor("z_pk"),
                 "elimination_per_hour":np.log(2)/np.array([s["pk_half_life_hours"] for s in self.source["sites"]])*self.site_factor("z_pk",1),
                 "kd_mol_per_l":np.array([s["kd_mol_per_l"] for s in self.source["sites"]])*self.site_factor("z_pk",1),
                 "bmax_mol_per_l":np.array([s["bmax_mol_per_l"] for s in self.source["sites"]])*self.site_factor("z_pk")}

    def site_factor(self,name,shift=0):
        values=np.asarray(self.factors[name],dtype=float)
        return values[(np.arange(36)+shift)%len(values)]

    def reaction(self,x,occupancy):
        rate=self.source["base_rates_per_hour"];K=self.source["saturation_constants"]
        env=self.site_factor("z_env");tx=self.site_factor("z_tx");ep=self.site_factor("z_ephys")
        sat=lambda y,key:y/(K[key]+y)
        m,a,I,circuit,ab,plaque,tau,med,ca,v,activity=x.T
        production=np.zeros_like(x);removal=np.zeros_like(x);paired=np.zeros_like(x)
        production[:,0]=rate["misfolded_production"]*env*(1+self.source["inflammatory_feedback"]*sat(I,"inflammation"))
        removal[:,0]=rate["misfolded_clearance"]/env*m
        production[:,4]=rate["amyloid_production"]*env*(1+self.source["inflammatory_feedback"]*sat(I,"inflammation"))*(1-self.source["occupancy_production_inhibition"]*tx*occupancy)
        removal[:,4]=rate["amyloid_clearance"]/env*ab*(1+self.source["occupancy_clearance_gain"]*tx*occupancy)
        for soluble,aggregate in ((0,1),(4,5)):
            transfer=rate["aggregation"]*x[:,soluble]**2-rate["dissociation"]*x[:,aggregate]
            paired[:,soluble]-=transfer;paired[:,aggregate]+=transfer
            removal[:,aggregate]=rate["aggregate_clearance"]/env*x[:,aggregate]
        production[:,6]=rate["tau_production"]*env*(1+sat(plaque,"pathology")+sat(I,"inflammation"))
        removal[:,6]=rate["tau_clearance"]/env*tau
        production[:,7]=rate["mediator_production"]*env*(1+sat(plaque,"pathology")+sat(tau,"pathology"))
        removal[:,7]=rate["mediator_clearance"]*med
        drives={2:med,8:I*ep,9:(ca+I)*ep,10:(v+I)*ep,3:(activity+I)*ep}
        for index,drive in drives.items():
            name=DYNAMIC[index];base=self.source["response_baselines"][name]
            target=base+(1-base)*sat(drive,"response")
            k=rate[name+"_relaxation"]
            production[:,index]=k*target;removal[:,index]=k*x[:,index]
        return production,removal,paired

    def rhs(self,t,y,action,active_through,deadline=math.inf):
        if time.monotonic()>deadline:raise TimeoutError("controlled source clock exhausted")
        x=y[:36*len(DYNAMIC)].reshape(36,len(DYNAMIC))
        if not np.isfinite(x).all() or (x<0).any():raise FloatingPointError("nonfinite/negative source stage")
        occ=pk_values([t],self.pk,action,self.source,initial_pre_dose=False,active_through=active_through)[1][0]
        production,removal,paired=self.reaction(x,occ)
        diff,_=transport(x,self.volumes,self.edges,self.conductance,self.transport_scale)
        reaction=production-removal+paired
        return np.concatenate([(reaction+diff).ravel(),(self.volumes[:,None]*reaction).sum(0)])

    def quantities(self,times,states,action):
        C,O,B=pk_values(times,self.pk,action,self.source)
        result=np.zeros((len(times),36,15))
        names=self.source["grid"]["species_id"]
        for index,name in enumerate(DYNAMIC):result[:,:,names.index(name)]=states[:,:,index]
        result[:,:,names.index("free_drug")]=C
        result[:,:,names.index("target_bound_drug")]=B
        for i,x in enumerate(states):
            _,removal,_=self.reaction(x,O[i])
            _,flux=transport(x,self.volumes,self.edges,self.conductance,self.transport_scale)
            result[i,:,names.index("clearance_flux")]=(self.volumes[:,None]*removal).sum(1)
            absolute=np.zeros(36)
            np.add.at(absolute,self.edges[:,0],np.abs(flux).sum(1)/2)
            np.add.at(absolute,self.edges[:,1],np.abs(flux).sum(1)/2)
            result[i,:,names.index("vascular_perivascular_transport")]=absolute
        return result.reshape(len(times),6,6,15)


def integrate_source_interval(rhs, left, right, state, method, max_step, *, rtol, atol,
                              progress=None, work=None, accumulated_clock=False):
    """Original source integrators with observable accepted progress for a private plant.

    DOP853 retains solve_ivp's final dense-output evaluation and its arithmetic.
    A failed call never undoes accepted steps or RHS work reported to its owner.
    """
    if method not in {"RK4", "DOP853"} or not right > left or not max_step > 0:
        raise ValueError("Invalid source interval or unregistered integration method")
    counter = {"nfev": 0} if work is None else work
    before = counter["nfev"]

    def counted(t, y):
        counter["nfev"] += 1
        counter["last_rhs_time_hours"] = float(t)
        return rhs(t, y)

    def retain(t, y):
        if progress is not None:
            progress(float(t), y)

    y = state.copy()
    if method == "RK4":
        count = int(math.ceil((right-left)/max_step))
        step = (right-left)/count
        t = left
        for index in range(count):
            if not accumulated_clock:
                t = left+index*step
            k1=counted(t,y); k2=counted(t+step/2,y+step*k1/2)
            k3=counted(t+step/2,y+step*k2/2); k4=counted(t+step,y+step*k3)
            y = y+step*(k1+2*k2+2*k3+k4)/6
            t += step
            retain(t, y)
    else:
        solver = DOP853(counted, left, y, right, max_step=max_step, rtol=rtol, atol=atol)
        while solver.status == "running":
            message = solver.step()
            if solver.status == "failed":
                raise RuntimeError(message)
            retain(solver.t, solver.y)
        # solve_ivp(t_eval=[right]) evaluates dense output, including its extra RHS calls.
        y = solver.dense_output()(right)
        retain(right, y)
    return y, counter["nfev"]-before


def integrate(law: ControlledLaw, action: dict, method: str, step: float, deadline=math.inf) -> dict:
    times=np.asarray(law.source["grid"]["time_hours"],dtype=float)
    boundaries=sorted(set(times.tolist()+action["dose_times_hours"]))
    x0=law.initial.reshape(36,15)[:,[law.source["grid"]["species_id"].index(k) for k in DYNAMIC]]
    y=np.concatenate([x0.ravel(),np.zeros(len(DYNAMIC))]);saved={0.:y.copy()}
    source_scales=np.array([law.source["reference_scales"][k] for k in DYNAMIC])
    atol=np.concatenate([np.tile(source_scales,36),source_scales*law.volumes.sum()])*law.source["numerics"]["reference_atol_scale"]
    nfev=0
    for left,right in zip(boundaries,boundaries[1:]):
        rhs=lambda t,z:law.rhs(t,z,action,left,deadline)
        y, evaluations = integrate_source_interval(rhs, left, right, y, method, step,
            rtol=law.source["numerics"]["reference_rtol"], atol=atol, accumulated_clock=True)
        nfev += evaluations
        if not np.isfinite(y).all() or (y[:len(x0.ravel())]<0).any():raise FloatingPointError("invalid source state")
        if right in times:saved[right]=y.copy()
    values=np.stack([saved[t] for t in times]);states=values[:,:36*len(DYNAMIC)].reshape(len(times),36,len(DYNAMIC))
    mass=(states*law.volumes[None,:,None]).sum(1)
    residual=np.abs(mass-mass[0]-values[:,36*len(DYNAMIC):])/(source_scales*law.volumes.sum())
    paired_residual=0.;transport_residual=0.
    for i,x in enumerate(states):
        _,_,paired=law.reaction(x,np.zeros(36))
        paired_residual=max(paired_residual,float(np.abs(paired[:,0]+paired[:,1]).max()),float(np.abs(paired[:,4]+paired[:,5]).max()))
        diff,_=transport(x,law.volumes,law.edges,law.conductance,law.transport_scale)
        transport_residual=max(transport_residual,float(np.max(np.abs((diff*law.volumes[:,None]).sum(0))/source_scales)))
    return {"trajectory":law.quantities(times,states,action),"balance_max":float(residual.max()),
            "paired_balance_max":paired_residual,"transport_balance_max":transport_residual,"nfev":nfev}


def numerical_summary(coarse,fine,reference,scales,pk_error,pk_serialization,policy):
    arrays=[item["trajectory"] for item in (coarse,fine,reference)]
    if not all(np.isfinite(x).all() and (x>=0).all() for x in arrays):raise ValueError("invalid numerical source fields")
    axes=tuple(range(arrays[0].ndim-1))
    refinement=np.max(np.abs(arrays[0]-arrays[1]),axis=axes)
    reference_difference=np.max(np.abs(arrays[1]-arrays[2]),axis=axes)
    serialization=np.max(np.abs(arrays[2]-arrays[2].astype(np.float32).astype(float)),axis=axes)
    components={"coarse_fine":refinement,"fine_reference":reference_difference,
                "reference_float32_serialization":serialization,"pk_precision":np.asarray(pk_error),
                "pk_float32_serialization":np.asarray(pk_serialization)}
    uncertainty=np.max(np.stack(list(components.values())),axis=0)
    normalized=uncertainty/np.asarray(scales);tolerance=policy["normalized_tolerance"]
    reasons=[]
    if (normalized>tolerance).any():reasons.append("source_scaled_discrepancy_exceeds_tolerance")
    if (reference_difference>refinement+serialization).any():reasons.append("refinement_resolution_unresolved")
    if any(max(item[k] for k in ("balance_max","paired_balance_max","transport_balance_max"))>policy["balance_tolerance"] for item in (coarse,fine,reference)):reasons.append("source_balance_failed")
    if (uncertainty==0).any():reasons.append("unresolved_zero_numerical_discrepancy")
    return {"components":{k:v.tolist() for k,v in components.items()},"uncertainty":uncertainty.tolist(),
            "normalized_max":float(normalized.max()),"teacher_confidence":float(1/(1+np.max(normalized/tolerance))),
            "admissible":not reasons,"reasons":reasons}


def draw_world(source,group,index):
    """All randomness is confined to the pre-recorded group generation slot."""
    rng=np.random.default_rng(source["seed"]+10*index)
    factors={}
    for fi,name in enumerate(FACTOR_NAMES):
        width=len(source["factor_basis"][name])
        factors[name]=np.asarray([[source["coefficient_levels"][(atom[fi]+coordinate)%2]
                                  for coordinate in range(width)] for atom in source["atoms"]],dtype=np.float32)
    names=source["grid"]["species_id"]
    initial=np.zeros((4,6,6,15),dtype=np.float32)
    for ai in range(4):
        jitter=rng.uniform(*source["initial_jitter"],size=(36,len(DYNAMIC)))
        for si,name in enumerate(DYNAMIC):
            initial[ai,:,:,names.index(name)]=(source["initial_conditions"][group["condition"]][name]*jitter[:,si]*
                factors["z_env"][ai,np.arange(36)%factors["z_env"].shape[1]]).reshape(6,6)
        law=ControlledLaw(source,{k:v[ai] for k,v in factors.items()},initial[ai])
        full=law.quantities(np.array([0.]),initial[ai].reshape(1,36,15)[:,:,[names.index(k) for k in DYNAMIC]],source["actions"][0])[0]
        initial[ai]=full.astype(np.float32)
    selector=source["observation"]["selector"]
    mean=np.array([[initial[ai,s["region_index"],s["compartment_index"],names.index(s["species"])]/
                    source["reference_scales"][s["species"]] for s in selector] for ai in range(4)],dtype=np.float32)
    sd=source["observation"]["standard_deviation"]
    channel="controlled_predose_selector"
    support=JointParticleSupport(group["support_uid"],source["source_uid"],source["snapshot_ref"],
        "controlled_simulated",group["role"],(group["sample_uid"],),
        (tuple(group["particle_uids"]),),{k:torch.from_numpy(v[None]) for k,v in factors.items()},
        (tuple("complete_atom_"+str(i) for i in range(4)),),torch.from_numpy(initial[None]),
        torch.tensor([[[source["reference_scales"][k] for k in names]]*4],dtype=torch.float32),
        torch.tensor([source["prior_weights"]],dtype=torch.float32),
        {channel:{"law":"gaussian","units":"source_reference_standardized","mean":torch.from_numpy(mean[None]),
                  "variance":torch.full((1,4,len(selector)),sd**2)}},
        {channel:torch.zeros(1,4,len(selector))},tuple(source["species_units"][k] for k in names)).validate()
    production=produce_controlled_observations(support,[{"channel_uid":channel,
        "observation_uid":group["sample_uid"]+":predose","observed":[source["observation"]["mask"]],
        "encoder_indices":list(range(len(selector))),"normalization_ref":source["snapshot_ref"]+":fixed_source_scales",
        "modality":"state","assay_uid":"controlled_predose_gaussian"}],
        latent_seed=source["seed"]+10*index+1,observation_seed=source["seed"]+10*index+2)
    return support,production,int(production.generating_indices[0])


def compiled_particle_program(source,group,support,atom,action,pk,parameters,code_hash):
    binding={"source_uid":source["source_uid"],"source_snapshot_ref":source["snapshot_ref"],
        "source_posterior_uid":group["posterior_uid"],"support_uid":group["support_uid"],
        "sample_uid":group["sample_uid"],"particle_uid":group["particle_uids"][atom],
        "regimen_uid":action["regimen_uid"],"binding_law":"single_ligand_equilibrium",
        "requested_readout":"fractional_occupancy","interpolation":"exact",
        "particle_basis":whole_particle_basis(support.to_payload(),0,atom),
        "action_setting_paths":{k:["settings",k] for k in ("dose","dose_unit","route","schedule","duration","agent_set")},
        "particle_parameters":{},"site_mappings":[]}
    action_payload={"schema_version":SCHEMA_VERSIONS["world_action"],"action_uid":action["regimen_uid"]+":action",
        "intervention_type":"pharmacological","target_set":[source["pk"]["target_uid"]],
        "agent_set":[source["pk"]["ligand_uid"]],"dose":source["pk"]["total_dose_mg_per_kg"]*action["fractions"][0],
        "dose_unit":"mg/kg","schedule":{"kind":"timepoints","values":action["dose_times_hours"],"unit":"hour",
                                           "dose_fractions":action["fractions"]},
        "route":"intravenous","duration":{"value":672.,"unit":"hour"},
        "state_scope":{"regions":source["grid"]["region_id"]},"support_domain_ref":source["snapshot_ref"],
        "source_evidence_refs":[source["snapshot_ref"]],"action_provenance_ref":source["snapshot_ref"],
        "conflicts":[],"status":"contract_valid"}
    snapshot={k:binding[k] for k in ("source_uid","source_snapshot_ref","source_posterior_uid","support_uid",
                                  "sample_uid","particle_uid","regimen_uid","particle_basis")}
    snapshot.update(settings_origin="controlled_source_parameter_definition",source_model_id=source["source_uid"],
        source_run_id=group["sample_uid"]+":"+str(atom)+":"+action["regimen_uid"],model_sha256=code_hash,
        settings={k:action_payload[k] for k in binding["action_setting_paths"]},
        parameters={k:np.asarray(v).tolist() for k,v in parameters.items()},particle_parameter_paths={},
        quantities={},affinities={},event_convention=source["event_convention"],baseline_intervention_history=[],
        pk_definition=copy.deepcopy(source["pk"]),source_definition_sha256=digest(source))
    for key,value in snapshot["parameters"].items():
        path=["parameters",key];binding["particle_parameters"][key]={"source_path":path,"value":value}
        snapshot["particle_parameter_paths"][key]=path
    rows=[]
    for si,site in enumerate(source["sites"]):
        uid=site["site_uid"];path="controlled|"+uid+"|free_ligand"
        quantity={"site_uid":uid,"ligand_uid":source["pk"]["ligand_uid"],"target_uid":source["pk"]["target_uid"],
                  "quantity_kind":"free_concentration","concentration_unit":"mol/L","time_unit":"hour"}
        snapshot["quantities"][path]=quantity
        affinity=dict(value=float(parameters["kd_mol_per_l"][si]),unit="mol/L",source_ref=source["snapshot_ref"],
            quantity_kind="equilibrium_dissociation_constant",binding_scope="particle_parameter",
            source_uid=source["source_uid"],source_snapshot_ref=source["snapshot_ref"],
            particle_uid=binding["particle_uid"],site_uid=uid,ligand_uid=source["pk"]["ligand_uid"],target_uid=source["pk"]["target_uid"])
        snapshot["affinities"][uid]=affinity
        name="affinity:"+uid;parameter_path=["affinities",uid]
        binding["particle_parameters"][name]={"source_path":parameter_path,"value":affinity}
        snapshot["particle_parameter_paths"][name]=parameter_path
        binding["site_mappings"].append({"region_id":site["region_id"],"compartment_id":site["compartment_id"],
            "site_uid":uid,"source_output_path":path,"kd_source_path":parameter_path,
            "kd_binding":{"kind":"particle_parameter","parameter_name":name}})
        for ti,t in enumerate(source["grid"]["time_hours"]):
            rows.append(dict(quantity,source_output_path=path,source_time=t,concentration=float(pk[0][ti,si]),
                source_model_id=snapshot["source_model_id"],source_run_id=snapshot["source_run_id"],
                source_snapshot_ref=source["snapshot_ref"]))
    snapshot["export_sha256"]=digest(rows)
    inputs={"binding":binding,"run_snapshot":snapshot,"action":action_payload,"source_rows":rows,
        "region_ids":source["grid"]["region_id"],"compartment_ids":source["grid"]["compartment_id"],
        "time_grid_hours":source["grid"]["time_hours"],"graph_ref":source["snapshot_ref"]+":graph",
        "organism_context":"prospective_controlled_cns",
        "provenance":{"model_sha256":code_hash,"export_sha256":snapshot["export_sha256"],"run_snapshot_sha256":digest(snapshot)}}
    return compile_physical_record(inputs)


def joint_source(source,group,support,production,records):
    shape=(1,4,2,len(source["grid"]["time_hours"]),6,6,1)
    programs={"source_posterior_uid":group["posterior_uid"],"source_snapshot_ref":source["snapshot_ref"],
        "support_uid":support.support_uid,"source_uid":support.source_uid,"sample_uids":[group["sample_uid"]],
        "particle_uids":[group["particle_uids"]],"regimen_uids":[a["regimen_uid"] for a in source["actions"]],
        "records":[records]}
    for key in ("occupancy","support","physical_time"):
        programs[key]=np.asarray([[record[key] for record in [v["physical_program"] for v in row]] for row in records]).reshape(shape).tolist()
    return {"schema_version":JOINT_SOURCE_SCHEMA,"source_posterior_uid":group["posterior_uid"],
            "support":support.to_payload(),"observations":production.inference_payload()["observations"],
            "particle_programs":programs,
            "producer_seeds":{"latent_seed":production.latent_seed,"observation_seed":production.observation_seed}}
