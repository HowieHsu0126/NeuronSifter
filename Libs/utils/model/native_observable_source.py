"""Fresh public-law observable worlds coupled one way to the CNS background."""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import math
import time

import numpy as np
import torch

from Libs.models.modules.world_state import FACTOR_NAMES, JointParticleSupport
from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
from Libs.utils.data.controlled_world_runtime import (
    _simulate_electrophysiology, _simulate_calcium_synaptic, _simulate_molecular_pharmacological,
)
from Libs.utils.data.physical_occupancy import compile_physical_record, digest, whole_particle_basis
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES, compile_stimulation_record
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support, quantity_incidence
from Libs.utils.model.controlled_cns_source import ControlledLaw, DYNAMIC, transport, integrate_source_interval

ELECTRICAL = "electrophysiology_conductance_world"
SYNAPTIC = "calcium_synaptic_world"
MOLECULAR = "molecular_pharmacological_causal_world"


def group_definitions(config: dict) -> list[dict]:
    settings = config["native_observable_bridge"]
    groups = []
    for family in observable_profile(config)["families"]:
        for replicate, role in enumerate(settings["group_roles"]):
            uid = settings["source_uid"] + ":" + family + ":" + str(replicate)
            groups.append({"group_uid": uid, "sample_uid": uid, "support_uid": uid + ":support",
                "posterior_uid": uid + ":initial-prior", "particle_uids": [uid + ":atom:" + str(i) for i in range(4)],
                "family_uid": family, "replicate": replicate, "role": role,
                "calibration_reservation": settings["calibration_reservations"].get(replicate)})
    return groups


def action_definition(config: dict, family: str, action_index: int) -> dict:
    definition = copy.deepcopy(config["native_observable_bridge"]["physical_families"][family]["action"])
    multiplier = config["native_observable_bridge"]["action_multipliers"][action_index]
    definition["extension"]["dose_or_intensity"]["value"] *= multiplier
    definition["regimen_uid"] = config["native_observable_bridge"]["source_uid"] + ":" + family + ":action:" + str(action_index)
    return definition


def action_levels(grid: dict, action: dict) -> list[float]:
    times = grid["query_time"]["values"]
    delay = action["response_delay_steps"]
    magnitude = action["extension"]["dose_or_intensity"]["value"]
    stop = times[delay] + action["extension"]["duration"]["value"]
    return [float(magnitude) if i >= delay and t < stop else 0. for i, t in enumerate(times)]


def physical_trajectory(config: dict, family: str, parameters: dict, action: dict, *, high_precision=False) -> dict:
    grid = family_grid(observable_profile(config), family)
    state = {key: value for key, value in parameters.items() if key.startswith("initial_")}
    if family == ELECTRICAL:
        state["initial_spike_rate"] = 0.
    names = grid["quantity_profile"]["families"][family]["observable_names"]
    if not high_precision:
        function = {ELECTRICAL: _simulate_electrophysiology, SYNAPTIC: _simulate_calcium_synaptic,
                    MOLECULAR: _simulate_molecular_pharmacological}[family]
        rows = function({"family_uid": family, "world": {"time_grid": {"values": grid["query_time"]["values"]}}},
            {"oracle_state": state, "oracle_parameters": parameters},
            {"controlled_extension": action["extension"], "response_delay_steps": action["response_delay_steps"]})
        return {"values": np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)}
    # Same discrete law and identical binary64 inputs, evaluated with Decimal50.
    with localcontext() as context:
        context.prec = config["native_observable_bridge"]["numerics"]["decimal_precision"]
        D = lambda value: Decimal.from_float(float(value))
        p = {key: D(value) for key, value in parameters.items()}
        clamp = lambda value: min(Decimal(1), max(Decimal(0), value))
        values = [D(state.get("initial_" + name, 0.)) for name in names]
        rows = [values.copy()]
        levels = action_levels(grid, action)
        times = grid["query_time"]["values"]
        for index in range(1, len(times)):
            h, stimulus = D(times[index] - times[index-1]), D(levels[index])
            if family == ELECTRICAL:
                voltage, calcium, spike = values
                voltage += h * (-p["membrane_leak_rate"] * (voltage-p["resting_membrane_voltage"]) + p["stimulation_gain"]*stimulus)
                spike = max(Decimal(0), (voltage-p["spike_threshold"])*p["spike_rate_gain"])
                calcium += h * (-p["calcium_decay_rate"]*(calcium-D(.10)) + p["spike_calcium_gain"]*spike)
                values = [voltage, calcium, spike]
            elif family == SYNAPTIC:
                calcium, release, vesicle, current = values
                calcium += h * (-p["calcium_clearance_rate"]*(calcium-p["baseline_intracellular_calcium"]) + p["presynaptic_calcium_gain"]*stimulus)
                release = clamp(p["initial_release_probability"] + p["release_coupling"]*max(Decimal(0), calcium-p["baseline_intracellular_calcium"]))
                released = min(vesicle, h*p["vesicle_release_rate"]*release*vesicle)
                vesicle = clamp(vesicle + h*p["vesicle_recovery_rate"]*(1-vesicle)-released)
                current += p["postsynaptic_current_gain"]*released-h*p["postsynaptic_decay_rate"]*current
                values = [calcium, release, vesicle, current]
            else:
                exposure, occupancy, signal, response = values
                exposure = max(Decimal(0), exposure+h*(p["dose_to_exposure_gain"]*stimulus-p["exposure_clearance_rate"]*exposure-p["feedback_rate"]*response*exposure))
                equilibrium = exposure/(exposure+p["binding_affinity"]) if exposure > 0 else Decimal(0)
                occupancy = clamp(occupancy+h*p["occupancy_relaxation_rate"]*(equilibrium-occupancy))
                signal = clamp(signal+h*(p["signal_activation_rate"]*occupancy*(1-signal)-p["signal_decay_rate"]*signal))
                response = clamp(response+h*(p["biologic_response_rate"]*signal*(1-response)-p["biologic_clearance_rate"]*response))
                values = [exposure, occupancy, signal, response]
            rows.append(values.copy())
        return {"values": np.asarray(rows, dtype=np.float64), "decimal_values": [[str(v) for v in row] for row in rows]}


def physical_errors(primary: dict, precise: dict) -> tuple[np.ndarray, np.ndarray]:
    values = primary["values"]
    numeric, serialization = np.zeros(values.shape[-1]), np.zeros(values.shape[-1])
    if precise.get("precision_method") == "independent_fixed_step_refinements":
        reference, fine = precise["values"], precise["fine_values"]
        if reference.shape != values.shape or fine.shape != values.shape or not np.isfinite(reference).all():
            raise ValueError("Romani numerical refinement arrays differ")
        numeric = np.maximum(np.abs(values-fine), np.abs(fine-reference)).max(axis=0)
        serialization = np.abs(reference-reference.astype(np.float32).astype(np.float64)).max(axis=0)
        return numeric, serialization
    for ti, row in enumerate(values):
        for si, value in enumerate(row):
            reference = Decimal(precise["decimal_values"][ti][si])
            numeric[si] = max(numeric[si], float(abs(Decimal.from_float(float(value))-reference)))
            serialization[si] = max(serialization[si], float(abs(Decimal.from_float(float(np.float32(value)))-reference)))
    return numeric, serialization


def draw_group(config: dict, group: dict, group_index: int) -> dict:
    settings = config["native_observable_bridge"]
    background = config["controlled_cns_source"]
    profile = observable_profile(config)
    grid = family_grid(profile, group["family_uid"])
    seed = int(settings["seed"]) + group_index * 100
    streams = {name: seed+int(offset) for name, offset in settings["rng_stream_offsets"].items()}
    rng = np.random.default_rng(streams["background"])
    factors = {}
    for fi, name in enumerate(FACTOR_NAMES):
        width = len(background["factor_basis"][name])
        factors[name] = np.asarray([[settings["coefficient_levels"][(atom[fi]+coordinate)%2]
                                    for coordinate in range(width)] for atom in settings["atoms"]], dtype=np.float32)
    declarations = settings["physical_families"][group["family_uid"]]["parameters"]
    jitter_rng = np.random.default_rng(streams["physical_parameters"])
    jitter = {name: jitter_rng.uniform(0., 1.)*row.get("seed_jitter", 0.) for name, row in declarations.items()}
    parameters, initial = [], np.zeros((4, 6, 6, 26), dtype=np.float32)
    names = background["grid"]["species_id"]
    family_quantities = [(si, q) for si, q in enumerate(profile["quantities"]) if q.get("family_uid") == group["family_uid"]]
    for atom in range(4):
        current = {}
        for index, (name, declaration) in enumerate(declarations.items()):
            factor_name, coordinate = settings["parameter_factor_binding"][group["family_uid"]][name]
            factor = factors[factor_name][atom, coordinate]
            sign = -1. if factor < 1. else 1.
            current[name] = float(declaration["value"]) + sign*jitter[name]
        parameters.append(current)
        variations = rng.uniform(*background["initial_jitter"], size=(36, len(DYNAMIC)))
        for si, name in enumerate(DYNAMIC):
            initial[atom, :, :, names.index(name)] = (background["initial_conditions"][settings["condition"]][name]
                * variations[:, si] * factors["z_env"][atom, np.arange(36)%factors["z_env"].shape[1]]).reshape(6, 6)
        law = ControlledLaw(background, {key: value[atom] for key, value in factors.items()}, initial[atom, ..., :15])
        dynamic = initial[atom, ..., :15].reshape(1, 36, 15)[:, :, [names.index(name) for name in DYNAMIC]]
        initial[atom, ..., :15] = law.quantities([0.], dynamic, background["actions"][0])[0]
        for si, quantity in family_quantities:
            site = quantity["incidence"]
            ri, ci = profile["region_id"].index(site["anchor_region_id"]), profile["compartment_id"].index(site["anchor_compartment_id"])
            initial[atom, ri, ci, si] = current.get("initial_"+quantity["observable_name"], 0.)
    # Every complete prior atom and its physical parameters precede hidden selection.
    selector = background["observation"]["selector"]
    means = np.asarray([[initial[atom, row["region_index"], row["compartment_index"], names.index(row["species"])]
                         / background["reference_scales"][row["species"]] for row in selector] for atom in range(4)], dtype=np.float32)
    scales = [background["reference_scales"].get(q["quantity_id"], settings["auxiliary_reference_scales"].get(q["quantity_id"])) for q in profile["quantities"]]
    channel = "controlled_predose_selector"
    support = JointParticleSupport(group["support_uid"], settings["source_uid"], settings["snapshot_ref"],
        "controlled_simulated", group["role"], (group["sample_uid"],), (tuple(group["particle_uids"]),),
        {key: torch.from_numpy(value[None]) for key, value in factors.items()},
        (tuple("complete_atom_"+str(i) for i in range(4)),), torch.from_numpy(initial[None]),
        torch.tensor([[scales]*4], dtype=torch.float32), torch.tensor([settings["prior_weights"]], dtype=torch.float32),
        {channel: {"law": "gaussian", "units": "source_reference_standardized", "mean": torch.from_numpy(means[None]),
                   "variance": torch.full((1, 4, len(selector)), background["observation"]["standard_deviation"]**2)}},
        {channel: torch.zeros(1, 4, len(selector))}, tuple(q["unit"] for q in profile["quantities"]),
        quantity_profile=profile, source_family_uid=group["family_uid"]).validate()
    production = produce_controlled_observations(support, [{"channel_uid": channel,
        "observation_uid": group["sample_uid"]+":initial", "observed": [background["observation"]["mask"]],
        "encoder_indices": list(range(len(selector))), "normalization_ref": settings["snapshot_ref"]+":fixed_source_scales",
        "modality": "state", "assay_uid": "controlled_predose_gaussian"}],
        latent_seed=streams["latent_selection"], observation_seed=streams["observations"])
    return {"support": support, "production": production, "hidden_index": int(production.generating_indices[0]),
            "parameters": parameters, "rng_streams": streams}


class CoupledObservableLaw:
    def __init__(self, config: dict, group: dict, support: JointParticleSupport, truth: int, physical: dict):
        self._initialize(config, group, support.initial_field[0, truth].numpy().astype(np.float64),
            {name: value[0, truth].numpy() for name, value in support.factors.items()}, support.source_uid,
            physical.get("feature_names"))
        self.physical = physical["values"]
        if self.physical.shape != (len(self.grid["time_hours"]), len(self.physical_names)):
            raise ValueError("Physical source outputs must align to named native observations")

    @classmethod
    def from_initial_state(cls, config: dict, group: dict, initial: np.ndarray,
                           factors: dict, source_uid: str):
        """Evaluator-private single-world construction, without a future trajectory."""
        law = cls.__new__(cls)
        law._initialize(config, group, initial, factors, source_uid)
        law.physical = None
        return law

    def _initialize(self, config, group, initial, factors, source_uid, physical_names=None):
        self.config, self.group = config, group
        self.grid = family_grid(observable_profile(config), group["family_uid"])
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        self.law = ControlledLaw(config["controlled_cns_source"], factors, self.initial[..., :15])
        self.physical_names = (physical_names if physical_names is not None else
            self.grid["quantity_profile"]["families"][group["family_uid"]]["observable_names"])
        self.source_uid = source_uid
        self.quantity_support = quantity_support(self.grid, self.source_uid)[0]
        self.times = np.asarray(self.grid["time_hours"])
        self.names = config["controlled_cns_source"]["grid"]["species_id"]
        self.ca1 = self.grid["region_id"].index("hippocampus_ca1")*6+self.grid["compartment_id"].index("isf")
        self.ca3 = self.grid["region_id"].index("hippocampus_ca3")*6+self.grid["compartment_id"].index("isf")
        self.molecular = group["family_uid"] == MOLECULAR

    def _physical_at(self, index, physical):
        if physical is None:
            if self.physical is None or index is None:
                raise ValueError("Current physical source vector is required")
            physical = self.physical[index]
        physical = np.asarray(physical, dtype=np.float64)
        if physical.shape != (len(self.physical_names),) or not np.isfinite(physical).all():
            raise ValueError("Physical source vector differs from its declared coordinates")
        return physical

    def reaction(self, x, index=None, *, physical=None):
        physical = self._physical_at(index, physical)
        occupancy = np.zeros(36)
        if self.molecular:
            occupancy[self.ca1] = physical[1]
        production, removal, paired = self.law.reaction(x, occupancy)
        coupling = self.config["native_observable_bridge"]["coupling"]
        gain = coupling["gain"]
        rates = self.law.source["base_rates_per_hour"]
        if self.group["family_uid"] == ELECTRICAL and self.physical_names == ["membrane_voltage"]:
            coupling = self.config["native_romani_source"]["coupling"]
            voltage = physical[0]
            drive = max(0., (voltage-coupling["voltage_reference_mV"])/coupling["voltage_scale_mV"])
            production[self.ca1, DYNAMIC.index("membrane_voltage")] += coupling["activity_rate_per_hour"]*drive
        elif self.group["family_uid"] == ELECTRICAL:
            production[self.ca1, 8] += rates["calcium_concentration_relaxation"]*gain*physical[1]/coupling["calcium_reference_umol_per_l"]
            production[self.ca1, 10] += rates["spike_rate_relaxation"]*gain*physical[2]/coupling["spike_reference_hz"]
        elif self.group["family_uid"] == SYNAPTIC:
            production[self.ca3, 8] += rates["calcium_concentration_relaxation"]*gain*physical[0]/coupling["calcium_reference_umol_per_l"]
            production[self.ca1, 9] += rates["membrane_voltage_relaxation"]*gain*physical[3]/coupling["current_reference_pa"]
        return production, removal, paired

    def rhs(self, t, state, index, deadline, *, physical=None):
        physical = self._physical_at(index, physical)
        if time.monotonic() > deadline:
            raise TimeoutError("Native observable source clock exhausted")
        x = state[:396].reshape(36, 11)
        if not np.isfinite(state).all() or (x < 0).any():
            raise FloatingPointError("Invalid CNS integration stage")
        production, removal, paired = self.reaction(x, physical=physical)
        diffusion, _ = transport(x, self.law.volumes, self.law.edges, self.law.conductance, self.law.transport_scale)
        reaction = production-removal+paired
        c = state[396:432].copy()
        dc, input_rate, eliminated = np.zeros(36), 0., 0.
        if self.molecular:
            c[self.ca1] = physical[0]*1e-9
            transport_rate, _ = transport(c[:, None], self.law.volumes, self.law.edges, self.law.conductance, np.ones(1))
            elimination = self.law.pk["elimination_per_hour"]*c
            dc = transport_rate[:, 0]-elimination
            input_rate = -dc[self.ca1]*self.law.volumes[self.ca1]
            eliminated = float((elimination*self.law.volumes).sum())
            dc[self.ca1] = 0.
        return np.concatenate([(reaction+diffusion).ravel(), dc,
            (self.law.volumes[:, None]*reaction).sum(0), [input_rate, eliminated]])

    def quantities(self, state, index=None, *, physical=None):
        physical = self._physical_at(index, physical)
        x = state[:396].reshape(36, 11)
        result = np.zeros((36, 26))
        for si, name in enumerate(DYNAMIC):
            result[:, self.names.index(name)] = x[:, si]
        if self.molecular:
            result[:, 0] = state[396:432]
            result[self.ca1, 0] = physical[0]*1e-9
            result[self.ca1, 1] = self.law.pk["bmax_mol_per_l"][self.ca1]*physical[1]
        _, removal, _ = self.reaction(x, physical=physical)
        result[:, self.names.index("clearance_flux")] = (self.law.volumes[:, None]*removal).sum(1)
        _, flux = transport(x, self.law.volumes, self.law.edges, self.law.conductance, self.law.transport_scale)
        absolute = np.zeros(36)
        np.add.at(absolute, self.law.edges[:, 0], np.abs(flux).sum(1)/2)
        np.add.at(absolute, self.law.edges[:, 1], np.abs(flux).sum(1)/2)
        result[:, self.names.index("vascular_perivascular_transport")] = absolute
        for si, quantity in enumerate(self.grid["quantity_profile"]["quantities"]):
            if quantity.get("family_uid") != self.group["family_uid"]:
                continue
            if quantity["observable_name"] not in self.physical_names:
                continue
            site = quantity_incidence(self.grid["quantity_profile"], quantity, self.source_uid)
            node = self.grid["region_id"].index(site["anchor_region_id"])*6+self.grid["compartment_id"].index(site["anchor_compartment_id"])
            result[node, si] = physical[self.physical_names.index(quantity["observable_name"])]
        return result.reshape(6, 6, 26)


def integrate_coupled_interval(law, state, index, physical, method, interval_fraction,
                               deadline=math.inf, *, progress=None, work=None):
    """Advance one original coupled-source interval using its current auxiliary state."""
    now, right = law.times[index:index+2]
    numerics = law.config["native_observable_bridge"]["numerics"]
    scales = np.asarray([law.law.source["reference_scales"][name] for name in DYNAMIC])
    atol = np.concatenate([np.tile(scales, 36), np.full(36,1e-9),
        scales*law.law.volumes.sum(), [1e-9,1e-9]])*numerics["reference_atol_scale"]
    rhs = lambda t, value: law.rhs(t, value, index, deadline, physical=physical)
    return integrate_source_interval(rhs, now, right, state, method, (right-now)*interval_fraction,
        rtol=numerics["reference_rtol"], atol=atol, progress=progress, work=work)


def integrate_coupled(law: CoupledObservableLaw, method: str, interval_fraction: float, deadline=math.inf) -> dict:
    indices = [law.names.index(name) for name in DYNAMIC]
    initial = law.initial[..., :15].reshape(36, 15)[:, indices]
    state = np.concatenate([initial.ravel(), np.zeros(36+11+2)])
    outputs, states = [], []
    input_jumps, prior_c, prior_bound, nfev = 0., 0., 0., 0
    for index, now in enumerate(law.times):
        if law.molecular:
            current_c = law.physical[index, 0]*1e-9
            current_bound = law.law.pk["bmax_mol_per_l"][law.ca1]*law.physical[index, 1]
            input_jumps += (current_c-prior_c+current_bound-prior_bound)*law.law.volumes[law.ca1]
            state[396+law.ca1] = current_c
            prior_c, prior_bound = current_c, current_bound
        outputs.append(law.quantities(state, index))
        saved = state.copy()
        saved[-2] += input_jumps
        states.append(saved)
        if index+1 == len(law.times):
            break
        state, evaluations = integrate_coupled_interval(law, state, index, law.physical[index],
            method, interval_fraction, deadline)
        nfev += evaluations
    trajectory = np.stack(outputs)
    trajectory[0] = law.initial
    states = np.stack(states)
    dynamic = states[:, :396].reshape(-1,36,11)
    mass = (dynamic*law.law.volumes[None,:,None]).sum(1)
    scales = np.asarray([law.law.source["reference_scales"][name] for name in DYNAMIC])
    balance = np.abs(mass-mass[0]-states[:,432:443])/(scales*law.law.volumes.sum())
    drug_mass = (trajectory[...,0].reshape(-1,36)*law.law.volumes).sum(1)
    drug_mass += trajectory[...,1].reshape(-1,36)[:,law.ca1]*law.law.volumes[law.ca1]
    drug_residual = drug_mass-drug_mass[0]-(states[:,-2]-states[0,-2])+(states[:,-1]-states[0,-1])
    return {"trajectory": trajectory, "balance_max": float(balance.max()),
            "drug_balance_max": float(np.abs(drug_residual).max()/(1e-9*law.law.volumes.sum())),
            "boundary_mass_input": states[:,-2], "eliminated_mass": states[:,-1], "drug_mass": drug_mass, "nfev": nfev}


def estimated_exposure(config: dict, parameters: dict, action: dict, *, high_precision=False) -> dict:
    grid = family_grid(observable_profile(config), MOLECULAR)
    return estimated_exposure_on_levels(config, parameters, grid["query_time"]["values"],
        action_levels(grid, action), high_precision=high_precision)


def estimated_exposure_on_levels(config: dict, parameters: dict, times, levels, *, high_precision=False) -> dict:
    """Original frozen estimate, with explicit public levels and no response feedback."""
    if len(times) != len(levels):
        raise ValueError("Estimated exposure requires one level per original query")
    convert = (lambda value: Decimal.from_float(float(value))) if high_precision else float
    with localcontext() as context:
        context.prec = config["native_observable_bridge"]["numerics"]["decimal_precision"]
        exposure = convert(parameters["initial_free_agent_exposure"])
        gain, clearance, kd = (convert(parameters[key]) for key in
                              ("dose_to_exposure_gain","exposure_clearance_rate","binding_affinity"))
        rows = [exposure]
        for index in range(1, len(times)):
            h = convert(times[index]-times[index-1])
            # Estimated PK program excludes the true future response feedback.
            exposure = max(convert(0), exposure+h*(gain*convert(levels[index])-clearance*exposure))
            rows.append(exposure)
        occupancy = [value/(value+kd) if value>0 else convert(0) for value in rows]
        output = {"exposure_nmol_per_l": np.asarray(rows,dtype=np.float64),
                  "equilibrium_occupancy": np.asarray(occupancy,dtype=np.float64)}
        if high_precision:
            output["decimal_values"] = [[str(c),str(o)] for c,o in zip(rows,occupancy)]
        return output


def compiled_program(config: dict, group: dict, support: JointParticleSupport, atom: int,
                     parameters: dict, action: dict, code_hash: str, estimate: dict | None = None) -> dict:
    settings = config["native_observable_bridge"]
    grid = family_grid(observable_profile(config), group["family_uid"])
    quantity = next(q for q in grid["quantity_profile"]["quantities"] if q.get("family_uid")==group["family_uid"])
    site = quantity_incidence(grid["quantity_profile"], quantity, support.source_uid)
    basis = whole_particle_basis(support.to_payload(),0,atom)
    action_payload = copy.deepcopy(action["canonical"])
    action_payload.update(action_uid=action["regimen_uid"]+":action",
        duration=action["extension"]["duration"],
        state_scope={"regions":[site["anchor_region_id"]],"physical_site_uid":site["physical_site_uid"]},
        source_evidence_refs=[settings["snapshot_ref"]],action_provenance_ref=settings["snapshot_ref"],
        support_domain_ref=settings["snapshot_ref"])
    if group["family_uid"] != MOLECULAR:
        for key in ("agent_set","target_set","dose","dose_unit","route"):
            action_payload.pop(key,None)
        levels=action_levels(grid,action)
        supported = quantity_support(grid, support.source_uid).any(axis=(0,1,2))
        affected = [q["observable_name"] for i,q in enumerate(grid["quantity_profile"]["quantities"])
                    if q.get("family_uid") == group["family_uid"] and supported[i]]
        action_payload["stimulation_or_environment"]={"affected_state_variables":affected,
            "magnitude_or_program":{"time_values":grid["query_time"]["values"],"values":levels,
                "time_unit":grid["query_time"]["unit"],"value_unit":action["extension"]["dose_or_intensity"]["unit"],
                "interpolation":"left_hold","incidence":site},
            "uncertainty":{"status":"known_controlled_input","parameter_uncertainty":"complete_prior_atoms"}}
        binding={"source_uid":settings["source_uid"],"source_snapshot_ref":settings["snapshot_ref"],
            "posterior_uid":group["posterior_uid"],"sample_uid":group["sample_uid"],"support_uid":group["support_uid"],
            "particle_uid":group["particle_uids"][atom],"particle_basis":basis,"regimen_uid":action["regimen_uid"]}
        record=compile_stimulation_record(action_payload,grid,binding)
        record["target_program_uid"]="nonchemical:"+action["regimen_uid"]
        return record
    if estimate is None:
        raise ValueError("Chemical program requires its reserved estimated-exposure operation.")
    action_payload.update(dose=action["extension"]["dose_or_intensity"]["value"],dose_unit="mg/kg")
    binding={"source_uid":settings["source_uid"],"source_snapshot_ref":settings["snapshot_ref"],
        "source_posterior_uid":group["posterior_uid"],"support_uid":group["support_uid"],
        "sample_uid":group["sample_uid"],"particle_uid":group["particle_uids"][atom],
        "regimen_uid":action["regimen_uid"],"binding_law":"single_ligand_equilibrium",
        "requested_readout":"fractional_occupancy","interpolation":"exact","particle_basis":basis,
        "action_setting_paths":{key:["settings",key] for key in ("dose","dose_unit","route","schedule","duration","agent_set")},
        "particle_parameters":{},"site_mappings":[]}
    snapshot={key:binding[key] for key in ("source_uid","source_snapshot_ref","source_posterior_uid","support_uid",
                                         "sample_uid","particle_uid","regimen_uid","particle_basis")}
    snapshot.update(settings_origin="controlled_source_parameter_definition",source_model_id=settings["source_uid"],
        source_run_id=group["sample_uid"]+":"+str(atom)+":"+action["regimen_uid"],model_sha256=code_hash,
        settings={key:action_payload[key] for key in binding["action_setting_paths"]},
        parameters={key:parameters[key] for key in ("initial_free_agent_exposure","dose_to_exposure_gain","exposure_clearance_rate","binding_affinity")},
        particle_parameter_paths={},quantities={},affinities={},estimated_exposure_policy=settings["estimated_chemical_policy"],
        baseline_intervention_history=[])
    target_mask = np.zeros((len(grid["region_id"]), len(grid["compartment_id"])), dtype=bool)
    target_mask[grid["region_id"].index(site["anchor_region_id"]), grid["compartment_id"].index(site["anchor_compartment_id"])] = True
    target_domain = {"mask": target_mask.tolist(), "evidence_ref": settings["snapshot_ref"]+":single_target_site",
        "semantics": "target_program_support_not_concentration_absence",
        "excluded_site_meaning": "outside_declared_target_program"}
    binding["target_site_domain"] = target_domain
    snapshot["target_site_domain"] = copy.deepcopy(target_domain)
    for key,value in snapshot["parameters"].items():
        path=["parameters",key]
        binding["particle_parameters"][key]={"source_path":path,"value":value}
        snapshot["particle_parameter_paths"][key]=path
    ligand,target=action_payload["agent_set"][0],action_payload["target_set"][0]
    uid=site["physical_site_uid"]; path="observable|"+uid+"|estimated_free_concentration"
    quantity={"site_uid":uid,"ligand_uid":ligand,"target_uid":target,"quantity_kind":"free_concentration",
              "concentration_unit":"mol/L","time_unit":"hour"}
    snapshot["quantities"][path]=quantity
    affinity={"value":parameters["binding_affinity"]*1e-9,"unit":"mol/L","source_ref":settings["snapshot_ref"],
        "quantity_kind":"equilibrium_dissociation_constant","binding_scope":"particle_parameter",
        "source_uid":settings["source_uid"],"source_snapshot_ref":settings["snapshot_ref"],
        "particle_uid":binding["particle_uid"],"site_uid":uid,"ligand_uid":ligand,"target_uid":target}
    snapshot["affinities"][uid]=affinity
    affinity_name="affinity:"+uid; affinity_path=["affinities",uid]
    binding["particle_parameters"][affinity_name]={"source_path":affinity_path,"value":affinity}
    snapshot["particle_parameter_paths"][affinity_name]=affinity_path
    binding["site_mappings"].append({"region_id":site["anchor_region_id"],"compartment_id":site["anchor_compartment_id"],
        "site_uid":uid,"source_output_path":path,"kd_source_path":affinity_path,
        "kd_binding":{"kind":"particle_parameter","parameter_name":affinity_name}})
    rows=[dict(quantity,source_output_path=path,source_time=t,concentration=float(c*1e-9),
        source_model_id=snapshot["source_model_id"],source_run_id=snapshot["source_run_id"],source_snapshot_ref=settings["snapshot_ref"])
        for t,c in zip(grid["time_hours"],estimate["exposure_nmol_per_l"])]
    snapshot["export_sha256"]=digest(rows)
    inputs={"binding":binding,"run_snapshot":snapshot,"action":action_payload,"source_rows":rows,
        "region_ids":grid["region_id"],"compartment_ids":grid["compartment_id"],"time_grid_hours":grid["time_hours"],
        "graph_ref":settings["snapshot_ref"]+":graph","organism_context":"prospective_controlled_neural_world",
        "provenance":{"model_sha256":code_hash,"export_sha256":snapshot["export_sha256"],"run_snapshot_sha256":digest(snapshot)}}
    return compile_physical_record(inputs)


def joint_source(config: dict, group: dict, generated: dict, records: list) -> dict:
    from Libs.utils.data.native_diffusion_data import JOINT_SOURCE_SCHEMA
    support=generated["support"]
    programs={"source_posterior_uid":group["posterior_uid"],"source_snapshot_ref":support.source_snapshot_ref,
        "support_uid":support.support_uid,"source_uid":support.source_uid,"sample_uids":[group["sample_uid"]],
        "particle_uids":[group["particle_uids"]],"regimen_uids":[records[0][ai]["regimen_uid"] for ai in range(2)],"records":[records]}
    for name in ("occupancy","support","physical_time",*FORCING_ARRAY_NAMES):
        fields=[]
        for row in records:
            actions=[]
            for record in row:
                if record["representation"]=="explicit_stimulation_grid":
                    value=record["compiled_program"]["arrays"][name]
                elif name in FORCING_ARRAY_NAMES:
                    value=np.zeros_like(record["physical_program"]["occupancy"]).tolist()
                else:
                    value=record["physical_program"][name]
                actions.append(value)
            fields.append(actions)
        programs[name]=[fields]
    return {"schema_version":JOINT_SOURCE_SCHEMA,"source_posterior_uid":group["posterior_uid"],
        "support":support.to_payload(),"observations":generated["production"].inference_payload()["observations"],
        "particle_programs":programs,
        "producer_seeds":{"latent_seed":generated["production"].latent_seed,"observation_seed":generated["production"].observation_seed}}


def numerical_summary(config: dict, family: str, parameters: dict, primary: dict, precise: dict,
                      results: dict) -> dict:
    settings = config["native_observable_bridge"]
    profile = observable_profile(config)
    reference = results["reference"]["trajectory"]
    coarse, fine = results["coarse"]["trajectory"], results["fine"]["trajectory"]
    difference = np.maximum(np.abs(coarse-fine), np.abs(fine-reference)).max(axis=(0,1,2))
    serialization = np.abs(reference-reference.astype(np.float32).astype(np.float64)).max(axis=(0,1,2))
    physical_numeric, physical_serialization = physical_errors(primary, precise)
    physical_names = primary.get("feature_names", profile["families"][family]["observable_names"])
    for index, quantity in enumerate(profile["quantities"]):
        if quantity.get("family_uid") != family or quantity["observable_name"] not in physical_names:
            continue
        column = physical_names.index(quantity["observable_name"])
        difference[index] = max(difference[index], physical_numeric[column])
        serialization[index] = max(serialization[index], physical_serialization[column])
    uncertainty = difference+serialization
    support = quantity_support(family_grid(profile, family), settings["source_uid"]).any(axis=(0,1,2))
    scales = np.asarray([config["controlled_cns_source"]["reference_scales"].get(
        q["quantity_id"], settings["auxiliary_reference_scales"].get(q["quantity_id"])) for q in profile["quantities"]])
    exact, reasons = {}, []
    for index in np.flatnonzero(support & (uncertainty == 0)):
        quantity = profile["quantities"][index]["quantity_id"]
        proof = None
        if family != MOLECULAR and quantity in {"free_drug","target_bound_drug"}:
            proof = "nonchemical_no_drug_initial_or_source"
        if family == ELECTRICAL and quantity == "electrical_spike_rate":
            threshold = Decimal.from_float(float(parameters["spike_threshold"]))
            if parameters["spike_rate_gain"] >= 0 and all(Decimal(row[0]) <= threshold for row in precise["decimal_values"][1:]):
                proof = "nonpositive_threshold_branch_on_every_discrete_update"
        if proof is None or not (reference[...,index] == 0).all():
            reasons.append("unresolved_exact_zero_uncertainty:"+quantity)
        else:
            exact[quantity]={"status":"exact_by_definition","derivation_kind":"identically_zero",
                "derivation_ref":settings["snapshot_ref"]+":"+proof,"quantity":quantity,
                "serialization_error":0.,"source_snapshot_ref":settings["snapshot_ref"]}
    normalized = float(np.max(uncertainty[support]/scales[support]))
    tolerance = settings["numerics"]["normalized_tolerance"]
    if normalized > tolerance:
        reasons.append("numerical_tolerance_exceeded")
    for label, result in results.items():
        for name in ("balance_max","drug_balance_max"):
            if result[name] > settings["numerics"]["balance_tolerance"]:
                reasons.append(label+":"+name)
    if not np.isfinite(reference).all():
        reasons.append("nonfinite_reference")
    return {"uncertainty":uncertainty.tolist(),"integration_or_precision_error":difference.tolist(),
        "serialization_error":serialization.tolist(),"max_normalized_error":normalized,
        "exact_zero_uncertainty":exact,"teacher_confidence":1./(1.+normalized/tolerance),
        "nfev":{label:result["nfev"] for label,result in results.items()},
        "balance":{label:{name:result[name] for name in ("balance_max","drug_balance_max")} for label,result in results.items()},
        "reasons":reasons}
