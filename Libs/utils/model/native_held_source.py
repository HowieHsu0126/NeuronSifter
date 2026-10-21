"""Independently specified held-source laws on the registered native clocks."""
from __future__ import annotations

from decimal import Decimal, localcontext
import math

import numpy as np
from scipy.integrate import solve_ivp

from Libs.utils.data.native_observable_profile import family_grid, observable_profile
from Libs.utils.model import controlled_cns_source as cns
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, MOLECULAR, action_levels


def physical_trajectory(config, family, parameters, action, *, high_precision=False):
    """Piecewise exponential nonlinear law; Decimal reference uses the same law."""
    grid = family_grid(observable_profile(config),family)
    names = grid["quantity_profile"]["families"][family]["observable_names"]
    times, levels = grid["query_time"]["values"], action_levels(grid,action)
    constants = config["native_observable_bridge"]["held_source_law"]
    with localcontext() as context:
        context.prec = config["native_observable_bridge"]["numerics"]["decimal_precision"]
        number = (lambda x:Decimal.from_float(float(x))) if high_precision else float
        exp = (lambda x:x.exp()) if high_precision else math.exp
        zero, one = number(0), number(1)
        p = {key:number(value) for key,value in parameters.items()}
        clamp = lambda x:min(one,max(zero,x))
        values = [p.get("initial_"+name,zero) for name in names]
        rows = [values.copy()]
        for index in range(1,len(times)):
            h, drive = number(times[index]-times[index-1]), number(levels[index])
            if family == ELECTRICAL:
                voltage, calcium, _ = values
                leak = p["membrane_leak_rate"]
                equilibrium = p["resting_membrane_voltage"] + p["stimulation_gain"]*drive / (
                    leak*(one+calcium/number(constants["calcium_reference_umol_per_l"])))
                voltage = equilibrium+(voltage-equilibrium)*exp(-leak*h)
                excess = max(zero,voltage-p["spike_threshold"])
                spike = p["spike_rate_gain"]*excess/(one+excess/number(constants["voltage_saturation_mv"]))
                decay = p["calcium_decay_rate"]
                equilibrium_ca = number(.1)+p["spike_calcium_gain"]*spike/decay
                calcium = equilibrium_ca+(calcium-equilibrium_ca)*exp(-decay*h)
                values = [voltage,calcium,spike]
            elif family == SYNAPTIC:
                calcium, _, vesicle, current = values
                decay = p["calcium_clearance_rate"]
                equilibrium = p["baseline_intracellular_calcium"]+p["presynaptic_calcium_gain"]*drive*vesicle/decay
                calcium = equilibrium+(calcium-equilibrium)*exp(-decay*h)
                excess = max(zero,calcium-p["baseline_intracellular_calcium"])
                reference = number(constants["calcium_reference_umol_per_l"])
                release = clamp(p["initial_release_probability"]+p["release_coupling"]*excess**2/(reference**2+excess**2))
                released = vesicle*(one-exp(-p["vesicle_release_rate"]*release*h))
                vesicle = clamp(one-(one-vesicle)*exp(-p["vesicle_recovery_rate"]*h)-released)
                current = current*exp(-p["postsynaptic_decay_rate"]*h)+p["postsynaptic_current_gain"]*released
                values = [calcium,release,vesicle,current]
            elif family == MOLECULAR:
                exposure, occupancy, signal, response = values
                clearance = p["exposure_clearance_rate"]+p["feedback_rate"]*response
                equilibrium = p["dose_to_exposure_gain"]*drive/clearance
                exposure = max(zero,equilibrium+(exposure-equilibrium)*exp(-clearance*h))
                equilibrium_o = exposure**2/(exposure**2+p["binding_affinity"]**2)
                occupancy = clamp(equilibrium_o+(occupancy-equilibrium_o)*exp(-p["occupancy_relaxation_rate"]*h))
                signal_rate = p["signal_activation_rate"]*occupancy+p["signal_decay_rate"]
                signal_eq = p["signal_activation_rate"]*occupancy/signal_rate
                signal = clamp(signal_eq+(signal-signal_eq)*exp(-signal_rate*h))
                response_rate = p["biologic_response_rate"]*signal+p["biologic_clearance_rate"]
                response_eq = p["biologic_response_rate"]*signal/response_rate
                response = clamp(response_eq+(response-response_eq)*exp(-response_rate*h))
                values = [exposure,occupancy,signal,response]
            else:
                raise ValueError("Unregistered held-source physical family")
            rows.append(values.copy())
        result = {"values":np.asarray(rows,dtype=np.float64),
                  "solver":"piecewise_exponential_nonlinear_update"}
        if high_precision:
            result["decimal_values"] = [[str(value) for value in row] for row in rows]
        return result


class HeldCNSLaw(cns.ControlledLaw):
    """Saturating clearance, cooperative drives and paired saturating aggregation."""
    def reaction(self,x,occupancy):
        rates, constants = self.source["base_rates_per_hour"], self.source["saturation_constants"]
        env, tx, ep = self.site_factor("z_env"), self.site_factor("z_tx"), self.site_factor("z_ephys")
        cooperative = lambda value,key:value**2/(constants[key]**2+value**2)
        m,a,inflammation,circuit,amyloid,plaque,tau,mediator,calcium,voltage,activity = x.T
        production, removal, paired = np.zeros_like(x),np.zeros_like(x),np.zeros_like(x)
        feedback = 1+self.source["inflammatory_feedback"]*cooperative(inflammation,"inflammation")
        production[:,0] = rates["misfolded_production"]*env*feedback
        removal[:,0] = rates["misfolded_clearance"]/env*m/(1+m/constants["pathology"])
        occupied = occupancy**2/(self.source["held_source_law"]["occupancy_half"]**2+occupancy**2)
        production[:,4] = rates["amyloid_production"]*env*feedback*(1-self.source["occupancy_production_inhibition"]*tx*occupied)
        removal[:,4] = rates["amyloid_clearance"]/env*amyloid*(1+self.source["occupancy_clearance_gain"]*tx*occupied)/(1+amyloid/constants["pathology"])
        for soluble,aggregate in ((0,1),(4,5)):
            transfer = rates["aggregation"]*x[:,soluble]**2/(1+x[:,soluble]/constants["pathology"])-rates["dissociation"]*x[:,aggregate]
            paired[:,soluble] -= transfer
            paired[:,aggregate] += transfer
            removal[:,aggregate] = rates["aggregate_clearance"]/env*x[:,aggregate]
        production[:,6] = rates["tau_production"]*env*(1+cooperative(plaque,"pathology")+cooperative(inflammation,"inflammation"))
        removal[:,6] = rates["tau_clearance"]/env*tau
        production[:,7] = rates["mediator_production"]*env*(1+cooperative(plaque,"pathology")+cooperative(tau,"pathology"))
        removal[:,7] = rates["mediator_clearance"]*mediator
        drives = {2:mediator,8:inflammation*ep,9:(calcium+inflammation)*ep,10:(voltage+inflammation)*ep,3:(activity+inflammation)*ep}
        for index,drive in drives.items():
            name = cns.DYNAMIC[index]
            baseline = self.source["response_baselines"][name]
            target = baseline+(1-baseline)*cooperative(drive,"response")
            production[:,index] = rates[name+"_relaxation"]*target
            removal[:,index] = rates[name+"_relaxation"]*x[:,index]
        return production,removal,paired


def integrate_reference(law,action,step,deadline=math.inf):
    """Held-source RK45 reference with exact dose/output interval boundaries."""
    times = np.asarray(law.source["grid"]["time_hours"],dtype=float)
    boundaries = sorted(set(times.tolist()+action["dose_times_hours"]))
    indices = [law.source["grid"]["species_id"].index(name) for name in cns.DYNAMIC]
    initial = law.initial.reshape(36,15)[:,indices]
    y = np.concatenate([initial.ravel(),np.zeros(len(cns.DYNAMIC))])
    saved, nfev = {0.:y.copy()}, 0
    scales = np.asarray([law.source["reference_scales"][name] for name in cns.DYNAMIC])
    atol = np.concatenate([np.tile(scales,36),scales*law.volumes.sum()])*law.source["numerics"]["reference_atol_scale"]
    for left,right in zip(boundaries,boundaries[1:]):
        result = solve_ivp(lambda t,state:law.rhs(t,state,action,left,deadline),(left,right),y,
            method="RK45",max_step=step,rtol=law.source["numerics"]["reference_rtol"],atol=atol,t_eval=[right])
        if not result.success:
            raise RuntimeError(result.message)
        y = result.y[:,-1]
        if not np.isfinite(y).all() or (y[:396]<0).any():
            raise FloatingPointError("Held-source reference produced invalid states")
        nfev += result.nfev
        if right in times:
            saved[right] = y.copy()
    values = np.stack([saved[t] for t in times])
    states = values[:,:396].reshape(len(times),36,11)
    mass = (states*law.volumes[None,:,None]).sum(1)
    residual = np.abs(mass-mass[0]-values[:,396:])/(scales*law.volumes.sum())
    paired_max, transport_max = 0.,0.
    for state in states:
        paired = law.reaction(state,np.zeros(36))[2]
        paired_max = max(paired_max,float(np.abs(paired[:,0]+paired[:,1]).max()),float(np.abs(paired[:,4]+paired[:,5]).max()))
        diffusion,_ = cns.transport(state,law.volumes,law.edges,law.conductance,law.transport_scale)
        transport_max = max(transport_max,float(np.abs((diffusion*law.volumes[:,None]).sum(0)/scales).max()))
    return {"trajectory":law.quantities(times,states,action),"balance_max":float(residual.max()),
            "paired_balance_max":paired_max,"transport_balance_max":transport_max,"nfev":nfev}
