"""Training-reference CNS preferences and whole-tuple posterior controls.

These functions consume declared predictions, training readouts and compiled
actions. They never load a harness optimum or a hidden generating particle.
"""
from __future__ import annotations

from Libs.utils.data.native_actuation import molecular_administration_bound, molecular_administration_policy

import copy
from dataclasses import replace
import math

import numpy as np
import torch

from Libs.models.modules.world_state import FACTOR_NAMES, _posterior_uid
from Libs.utils.data.controlled_world_contracts import select_family_config
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.model.native_observable_source import ELECTRICAL, SYNAPTIC, MOLECULAR


def public_loss_policy(config: dict) -> dict:
    policy = config["native_public_cns_loss"]
    weights = np.asarray(list(policy["endpoint_weights"].values()), dtype=float)
    quantiles = policy["training_reference"]["quantiles"]
    if (policy["schema_version"] != "native_public_cns_loss.v1"
            or policy["readout"]["kind"] != "fixed_compartment"
            or policy["time_weighting"] != "normalized_trapezoid"
            or policy["region_weighting"] != "uniform"
            or weights.shape != (4,) or not np.isfinite(weights).all()
            or (weights <= 0).any() or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)
            or len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1
            or policy["training_reference"]["role"] != "train"
            or policy["training_reference"]["quantile_method"] != "linear"
            or policy["training_reference"]["weighting"] != "retained_readout_coordinates"
            or not math.isfinite(policy["intervention_weight"]) or policy["intervention_weight"] < 0):
        raise ValueError("Public CNS loss needs an explicit valid fixed preference/readout contract")
    return policy


def _readout(field: torch.Tensor, mask: torch.Tensor, grid: dict, policy: dict):
    shape = tuple(len(grid[key]) for key in ("time_hours", "region_id", "compartment_id", "species_id"))
    if field.shape[-4:] != shape or mask.shape[-4:] != shape or mask.dtype != torch.bool:
        raise ValueError("Public loss field and support must retain native time/region/compartment/species axes")
    compartment = grid["compartment_id"].index(policy["readout"]["compartment"])
    indices = [grid["species_id"].index(name) for name in policy["endpoint_weights"]]
    values = field[..., compartment, :][..., indices]
    supported = mask[..., compartment, :][..., indices]
    if not bool(supported.all()):
        raise ValueError("A missing public CNS endpoint cannot define a calibrated action loss")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite public CNS endpoint readout")
    return values


def fit_public_loss_reference(config: dict, records: list[dict], declared_training_groups: set[str]) -> dict:
    """Fit exact training-only marginal percentiles, retaining the full denominator."""
    policy = public_loss_policy(config)
    values, rows, units = [], [], None
    for record in records:
        if (record["role"] != "train" or record["sample_uid"] not in declared_training_groups
                or record["qc_grade"] not in policy["training_reference"]["qc_grades"]):
            raise ValueError("Public-loss percentiles require declared QC-approved training records")
        field = torch.as_tensor(record["trajectory"], dtype=torch.float64)
        mask = torch.as_tensor(record["quantity_support"])
        readout = _readout(field, mask, record["grid"], policy)
        current_units = {name:record["grid"]["species_units"][name] for name in policy["endpoint_weights"]}
        if units is not None and current_units != units:
            raise ValueError("Public-loss training endpoint units differ between domains")
        units = current_units
        values.append(readout.reshape(-1, 4).numpy())
        rows.append({key: record[key] for key in ("sample_uid", "trajectory_uid", "family_uid", "qc_grade", "content_hash")})
    ids = [row["trajectory_uid"] for row in rows]
    if (not rows or len(ids) != len(set(ids))
            or {row["sample_uid"] for row in rows} != declared_training_groups
            or {row["family_uid"] for row in rows} != set(policy["training_reference"]["family_uids"])):
        raise ValueError("Public-loss training group/action/family denominator is incomplete or duplicated")
    low, high = np.quantile(np.concatenate(values), policy["training_reference"]["quantiles"],
                            axis=0, method=policy["training_reference"]["quantile_method"])
    endpoints = {}
    for index, name in enumerate(policy["endpoint_weights"]):
        defined = bool(high[index] > low[index])
        endpoints[name] = {"low":float(low[index]), "high":float(high[index]),
                           "status":"defined" if defined else "constant_training_endpoint"}
    artifact = {"schema_version":"native_public_cns_reference.v1", "policy":copy.deepcopy(policy),
        "fit_role":"train", "training_records":rows, "training_group_uids":sorted(declared_training_groups),
        "endpoints":endpoints, "endpoint_units":units, "status":"defined" if (high > low).all() else "undefined_constant_endpoint",
        "scientific_twin":"NOT_RUN", "formal_qualification":False}
    artifact["reference_id"] = digest(artifact)
    return artifact


def validate_public_loss_reference(artifact: dict, config: dict, expected_groups: set[str] | None = None,
                                   expected_records: list[dict] | None = None) -> None:
    policy = public_loss_policy(config)
    unsigned = {key:value for key,value in artifact.items() if key != "reference_id"}
    required = config["schemas"]["native_public_cns_reference"]["required_fields"]
    if (not set(required) <= set(artifact) or artifact.get("schema_version") != "native_public_cns_reference.v1"
            or artifact.get("fit_role") != "train" or artifact.get("status") != "defined"
            or artifact.get("policy") != policy or artifact.get("reference_id") != digest(unsigned)):
        raise ValueError("Public CNS loss requires its intact, defined training-reference artifact")
    if expected_groups is not None and set(artifact["training_group_uids"]) != expected_groups:
        raise ValueError("Public CNS reference differs from its declared full training groups")
    if expected_records is not None:
        actual = {row["trajectory_uid"]:row for row in artifact["training_records"]}
        expected = {row["trajectory_uid"]:row for row in expected_records}
        if (len(actual) != len(artifact["training_records"]) or len(expected) != len(expected_records)
                or actual != expected):
            raise ValueError("Public CNS reference differs from admitted training record identities/content")


class NativePublicCNSLoss:
    """One immutable reference validation per evaluation; no per-group rehashing."""
    def __init__(self, artifact: dict, config: dict):
        validate_public_loss_reference(artifact, config)
        self.policy = copy.deepcopy(public_loss_policy(config))
        self.units = copy.deepcopy(artifact["endpoint_units"])
        self.low = tuple(artifact["endpoints"][name]["low"] for name in self.policy["endpoint_weights"])
        self.high = tuple(artifact["endpoints"][name]["high"] for name in self.policy["endpoint_weights"])
        if not all(math.isfinite(a) and math.isfinite(b) and b > a for a,b in zip(self.low,self.high)):
            raise ValueError("Constant public CNS endpoints cannot be given fabricated scales")
        self.reference_id = artifact["reference_id"]

    def __call__(self, field: torch.Tensor, mask: torch.Tensor, grid: dict,
                 action_burden: torch.Tensor) -> dict:
        policy = self.policy
        if self.units != {name:grid["species_units"][name] for name in policy["endpoint_weights"]}:
            raise ValueError("Public-loss endpoint units differ from its fitted training reference")
        readout = _readout(field, mask, grid, policy)
        low, high = readout.new_tensor(self.low), readout.new_tensor(self.high)
        normalized = (readout-low)/(high-low)
        clipped_fraction = ((normalized < 0) | (normalized > 1)).double().mean(dim=(-3, -2))
        times = readout.new_tensor(grid["time_hours"])
        if len(times) < 2 or not bool(torch.isfinite(times).all()) or not bool((times.diff() > 0).all()):
            raise ValueError("Public CNS loss needs a finite increasing physical clock")
        widths = times.diff()
        time_weight = torch.cat((widths[:1]/2, (widths[:-1]+widths[1:])/2, widths[-1:]/2))
        time_weight = time_weight/time_weight.sum()
        endpoint_loss = (normalized.clamp(0,1).mean(-2)*time_weight[:,None]).sum(-2)
        burden = action_burden.to(readout)
        if not bool(torch.isfinite(burden).all()) or not bool(((burden >= 0) & (burden <= 1)).all()):
            raise ValueError("Out-of-support intervention burden must remain unavailable, not clipped")
        loss = (endpoint_loss*readout.new_tensor(list(policy["endpoint_weights"].values()))).sum(-1)
        loss = loss+policy["intervention_weight"]*burden
        return {"loss":loss, "endpoint_loss":endpoint_loss, "clipped_fraction":clipped_fraction,
                "reference_id":self.reference_id, "maximum_supported_loss":1.+policy["intervention_weight"]}


def public_cns_loss(field: torch.Tensor, mask: torch.Tensor, grid: dict,
                    artifact: dict, action_burden: torch.Tensor, config: dict) -> dict:
    return NativePublicCNSLoss(artifact, config)(field, mask, grid, action_burden)


def summarize_posterior_action_loss(draws: torch.Tensor, weights: torch.Tensor) -> dict:
    """Choose by expected loss; report complete-tuple optimal-action mass.

    draws has (particle, conditional_draw, action) axes. D=1 is valid for a
    deterministic conditional law. Ties use the original declared action order.
    """
    if (draws.ndim != 3 or min(draws.shape) < 1 or weights.shape != draws.shape[:1]
            or not torch.isfinite(draws).all() or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.isclose(weights.double().sum(), weights.new_tensor(1.).double(),
                                 rtol=0., atol=1e-6)):
        raise ValueError("Action decisions require finite complete draws and normalized particle weights")
    mass = weights.to(device=draws.device, dtype=torch.float64)
    values = draws.double()
    # Remove only validated floating-point normalization error; original tuple
    # weights remain untouched. Ratios also keep all-optimal mass exactly one.
    total_mass = mass.sum()
    risks = (values.mean(1)*mass[:, None]).sum(0)/total_mass
    selected = int(risks.argmin())
    winners = values.argmin(-1)
    optimal_mass = ((winners == selected).double().mean(1)*mass).sum()/total_mass
    ties = (values == values.amin(-1, keepdim=True)).sum(-1) > 1
    return {"predictive_risks": risks, "selected_index": selected,
            "selected_optimal_mass": float(optimal_mass),
            "tied_optimal_mass": float((ties.double().mean(1)*mass).sum()/total_mass),
            "tie_break": "declared_action_order"}

def compiled_action_burden(record: dict, family_uid: str, config: dict) -> dict:
    """Use fixed domain maxima and actual delivered stimulus/total dose."""
    policy = public_loss_policy(config)["intervention_burden"]
    if family_uid == MOLECULAR and molecular_administration_policy(config) is not None:
        raise ValueError("Prospective administered-mass evaluation requires its forward public actuation ledger, not legacy nominal-dose costs")
    if family_uid in {ELECTRICAL, SYNAPTIC}:
        family = select_family_config(config, family_uid)
        bounds = family["support_domain"]
        maximum = bounds["supported_intensity"]["maximum"]
        duration_limit = bounds["supported_duration"]["maximum"]
        wave = record["action"]["stimulation_or_environment"]["magnitude_or_program"]
        times, levels = np.asarray(wave["time_values"], dtype=float), np.asarray(wave["values"], dtype=float)
        if (wave["value_unit"] != bounds["supported_intensity"]["unit"]
                or wave["time_unit"] != bounds["supported_duration"]["unit"]
                or wave["interpolation"] != "left_hold" or times.ndim != 1 or times.shape != levels.shape
                or len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(levels).all()
                or not (np.diff(times) > 0).all() or (levels < 0).any()):
            raise ValueError("Action burden needs the actual finite, nonnegative, unit-bound compiled stimulus")
        amount = float(np.dot(levels[:-1], np.diff(times)))
        duration = float(np.diff(times)[levels[:-1] != 0].sum())
        denominator = float(maximum*duration_limit)
        outside = bool(levels.max() > maximum or duration > duration_limit)
        unit = wave["value_unit"]+"*"+wave["time_unit"]
    else:
        if family_uid not in policy["chemical_family_uids"]:
            raise ValueError("No declared action-burden support domain for this family")
        action = record["action"]
        bound = select_family_config(config, policy["chemical_support_family_uid"])["support_domain"]
        if action["dose_unit"] != bound["supported_intensity"]["unit"]:
            raise ValueError("Drug burden dose and fixed support domain have different units")
        amount = float(action["dose"])
        snapshot = record["physical_program"]["inputs"]["run_snapshot"]
        if "pk_definition" in snapshot:
            fractions = np.asarray(action["schedule"]["dose_fractions"], dtype=float)
            total = float(snapshot["pk_definition"]["total_dose_mg_per_kg"])
            if (len(fractions) != len(action["schedule"]["values"]) or not len(fractions)
                    or not np.isfinite(fractions).all() or (fractions <= 0).any()
                    or not np.isclose(fractions.sum(),1.,rtol=0,atol=1e-12)
                    or not np.isclose(amount,total*fractions[0],rtol=1e-12,atol=0)):
                raise ValueError("Compiled dose schedule and source total dose disagree")
            amount = total
        denominator = float(bound["supported_intensity"]["maximum"])
        outside = amount > denominator
        unit = action["dose_unit"]
    if not math.isfinite(amount) or amount < 0 or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Intervention burden requires finite nonnegative amount and positive fixed support maximum")
    ratio = amount/denominator
    return {"actual_amount":amount, "unit":unit, "fixed_support_maximum":denominator,
        "unclipped_ratio":ratio, "normalized_burden":None if outside else ratio,
        "status":"action_outside_burden_support" if outside else "defined",
        "clinical_cost_calibration":False}


def composed_action_burden(public, config: dict) -> dict:
    """Charge the complete consumed-prefix/proposed-suffix physical public program."""
    if public.family_uid == MOLECULAR or public.input_kind == "model_driving_level":
        if molecular_administration_bound(public,config):
            amount = public.administered_mass()
            maximum = public.administration.maximum_supported_total_mg_per_kg
            ratio = amount/maximum
            return {"status":"defined" if amount <= maximum else "action_outside_burden_support",
                "normalized_burden":ratio if amount <= maximum else None,"actual_amount":amount,
                "unit":"mg/kg","fixed_support_maximum":maximum,"unclipped_ratio":ratio,
                "administration_law_uid":public.administration.law_uid,
                "legacy_model_drive_amplitude_is_total_dose":False,"clinical_cost_calibration":False}
        return {"status": "molecular_delivered_dose_mapping_unbound", "normalized_burden": None,
                "actual_amount": None, "unit": "mg/kg", "clinical_cost_calibration": False}
    if public.input_kind == "stimulation":
        if any(value < 0 for value in public.interval_values):
            return {"status": "signed_stimulation_burden_unbound", "normalized_burden": None,
                    "actual_amount": None, "unit": public.input_unit+"*"+public.source_time_unit,
                    "clinical_cost_calibration": False}
        # The final endpoint is not an interval. This local cost input is not a
        # compiled source record or a provenance claim.
        wave = {"time_values": public.source_times, "values": (*public.interval_values, 0.),
                "time_unit": public.source_time_unit, "value_unit": public.input_unit,
                "interpolation": "left_hold"}
        return compiled_action_burden({"action": {"stimulation_or_environment": {
            "magnitude_or_program": wave}}}, public.family_uid, config)
    policy = public_loss_policy(config)["intervention_burden"]
    if (public.input_kind != "dose_events" or public.family_uid not in policy["chemical_family_uids"]
            or public.source_time_unit != "hour"):
        raise ValueError("Composed public burden has no original physical support contract")
    bounds = select_family_config(config, policy["chemical_support_family_uid"])["support_domain"]
    if public.input_unit != bounds["supported_intensity"]["unit"]:
        raise ValueError("Composed event dose and original support domain have different units")
    amount = math.fsum(event.amount_mg_per_kg for event in public.dose_events)
    denominator = float(bounds["supported_intensity"]["maximum"])
    if not math.isfinite(amount) or amount < 0 or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Composed dose requires a finite amount and positive original support maximum")
    outside = amount > denominator
    return {"actual_amount": amount, "unit": public.input_unit, "fixed_support_maximum": denominator,
            "unclipped_ratio": amount/denominator, "normalized_burden": None if outside else amount/denominator,
            "status": "action_outside_burden_support" if outside else "defined", "clinical_cost_calibration": False}


def posterior_control(posterior, mode: str):
    """Change weights on already valid complete tuples; never splice constraints."""
    if mode == "posterior_joint":
        return posterior
    weights = posterior.log_weights.double().exp()
    if mode == "posterior_point":
        log_weights = torch.full_like(posterior.log_weights, -torch.inf)
        log_weights.scatter_(1, weights.argmax(-1,keepdim=True), 0.)
    elif mode == "posterior_dependence_ablated":
        independent = torch.ones_like(weights)
        for batch in range(len(weights)):
            joint_labels, cardinality = [], 1
            for name in FACTOR_NAMES:
                values = posterior.support.factors[name][batch]
                unique, inverse = torch.unique(values, dim=0, return_inverse=True)
                cardinality *= len(unique)
                marginal = weights.new_zeros(len(unique)).scatter_add_(0, inverse, weights[batch])
                independent[batch] *= marginal[inverse]
                joint_labels.append(inverse)
            labels = torch.stack(joint_labels, -1)
            if cardinality != weights.shape[1] or len(torch.unique(labels,dim=0)) != weights.shape[1]:
                raise ValueError("Dependence control needs complete Cartesian valid factor support; tuple recombination is forbidden")
        log_weights = independent.log()
        log_weights -= torch.logsumexp(log_weights, -1, keepdim=True)
    else:
        raise ValueError("Unknown native posterior comparison")
    parent = posterior.posterior_uid
    return replace(posterior, log_weights=log_weights,
        posterior_uid=_posterior_uid(posterior.support,posterior.observation_uids,parent,
                                     log_weights,posterior.particle_context),
        parent_posterior_uid=parent, status=mode)
