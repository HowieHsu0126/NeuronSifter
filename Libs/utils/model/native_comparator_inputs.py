"""Complete, target-free native comparator inputs and weighted-particle calls."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from Libs.models.modules.world_state import FACTOR_NAMES
from Libs.utils.data.native_input_staging import input_locations_from_config
from Libs.utils.data.native_diffusion_data import (
    bind_joint_programs, joint_particle_contexts, parse_joint_source, native_action_policy, load_joint_sources,
)
from Libs.utils.data.joint_physical_parameters import candidate_conditioning, parameter_schema_from_config
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES


@dataclass
class NativeComparatorInputs:
    condition: torch.Tensor
    validity: torch.Tensor
    initial: torch.Tensor
    weights: torch.Tensor
    grid: dict
    blocks: dict[str, tuple[int, int]]
    sample_uids: tuple
    particle_uids: tuple
    regimen_uids: tuple
    posterior_uid: str
    normalization_id: str
    programs: dict
    posterior: object
    observations: tuple

    @property
    def group_shape(self) -> tuple[int, int, int]:
        return (len(self.sample_uids), len(self.particle_uids[0]), len(self.regimen_uids))


def build_native_comparator_inputs(source: dict, posterior, normalization, config: dict) -> NativeComparatorInputs:
    """Consume only the same admitted source, posterior and compiled programs.

    Role/source identities are checked but never encoded as predictive features.
    No target, hidden particle selection, producer seed, or held-source law selector
    reaches the condition vector.
    """
    action_policy = native_action_policy(config)
    support, observations = parse_joint_source(source, posterior.weights.device, action_policy=action_policy)
    if support.to_payload() != posterior.support.to_payload():
        raise ValueError("Comparator posterior does not belong to the admitted source")
    if not normalization.active:
        raise ValueError("Comparator requires shared train-only fitted normalization")
    normalization.assert_support(support)
    settings = config["native_controlled_comparators"]["conditioning"]
    profile = observable_profile(config)
    grid = family_grid(profile, support.source_family_uid)
    b, j = posterior.weights.shape
    programs = bind_joint_programs(posterior, source, action_policy=action_policy)
    actions = programs["regimen_uids"]
    a = len(actions)
    if a < 2:
        raise ValueError("Controlled comparator requires complete paired actions")
    contexts = joint_particle_contexts(posterior, programs, actions, action_policy=action_policy)
    mask = torch.as_tensor(quantity_support(grid, support.source_uid), device=posterior.weights.device)
    validity = mask[None].expand(b*j*a, *mask.shape)
    initial = support.initial_field[:, :, None].expand(b,j,a,*support.initial_field.shape[2:])
    flat_initial = initial.reshape(b*j*a,*initial.shape[3:])
    initial_mask = mask[0].expand_as(flat_initial)
    normalized_initial = normalization.normalize(flat_initial, initial_mask)
    blocks, values, offset = {}, [], 0

    def append(name, value):
        nonlocal offset
        value = value.reshape(b*j*a,-1).to(posterior.weights)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite comparator conditioning: "+name)
        blocks[name] = (offset, offset+value.shape[1])
        offset += value.shape[1]
        values.append(value)

    def expand_particle(value):
        return value[:, :, None].expand(b,j,a,*value.shape[2:])

    append("initial_field", normalized_initial)
    for name in FACTOR_NAMES:
        append(name, expand_particle(support.factors[name]))
    append("named_parameters_and_presence",
           expand_particle(candidate_conditioning(support, parameter_schema_from_config(config))))
    if len(observations) != 1 or observations[0].sequence_number != 0:
        raise ValueError("This controlled protocol requires one retained predose observation")
    observation = observations[0]
    if (observation.channel_uid != "controlled_predose_selector" or observation.law != "gaussian"
            or observation.left_censored.any()):
        raise ValueError("Comparator predose observation law is outside the declared protocol")
    maximum_features = int(settings["maximum_observation_features"])
    feature_count = observation.values.shape[-1]
    if feature_count > maximum_features:
        raise ValueError("Comparator observation width exceeds the frozen contract")
    likelihood = support.readout_parameters[observation.channel_uid]
    for name, value in (
        ("observation_values", torch.where(observation.observed,observation.values,0)[:,None].expand(b,j,feature_count)),
        ("observation_mask", observation.observed[:,None].expand(b,j,feature_count)),
        ("readout_mean", likelihood["mean"]),
        ("readout_variance", likelihood["variance"]),
        ("source_nuisance", support.source_nuisance[observation.channel_uid]),
    ):
        append(name, expand_particle(F.pad(value, (0,maximum_features-feature_count))))
    maximum_times = int(settings["maximum_times"])
    time_count = len(grid["time_hours"])
    if time_count > maximum_times:
        raise ValueError("Comparator native clock exceeds the frozen contract")
    for name in ("occupancy","support","physical_time",*FORCING_ARRAY_NAMES):
        value = torch.stack([context[name] for context in contexts], dim=1)
        scale = float(settings["program_reference_scales"][name])
        if scale <= 0:
            raise ValueError("Program reference scales must be positive")
        value = value/scale
        value = F.pad(value, (0,0,0,0,0,0,0,maximum_times-time_count))
        append(name, value)
    time_mask = posterior.weights.new_zeros(b,j,a,maximum_times)
    time_mask[..., :time_count] = 1
    append("real_time_mask",time_mask)
    append("quantity_support",F.pad(validity,
        (0,0,0,0,0,0,0,maximum_times-time_count)))
    families = list(profile["families"])
    family = posterior.weights.new_zeros(b,j,a,len(families))
    family[...,families.index(support.source_family_uid)] = 1
    append("physical_domain",family)
    return NativeComparatorInputs(torch.cat(values,dim=-1),validity,normalized_initial,
        posterior.weights,grid,blocks,tuple(support.sample_uids),tuple(support.particle_uids),
        tuple(actions),posterior.posterior_uid,normalization.identity,programs,posterior,observations)


def load_native_prediction_inputs(config: dict, declaration: dict, belief, registries: dict):
    """Read one declared initial support and compiled programs before targets."""
    if declaration["role"] not in {"development", "calibration", "validation"}:
        raise ValueError("Prediction inputs require a declared evaluation or calibration role")
    uid = declaration["posterior_uid"]
    source = load_joint_sources(registries[declaration["family_uid"]], [uid],
                                input_locations=input_locations_from_config(config))[uid]
    posterior = belief(source)
    inputs = build_native_comparator_inputs(source, posterior, belief.field_normalization, config)
    support = inputs.posterior.support
    if (inputs.sample_uids != (declaration["sample_uid"],)
            or support.role != declaration["role"]
            or support.source_family_uid != declaration["family_uid"]
            or support.source_uid != declaration["source_uid"]):
        raise ValueError("Prediction support differs from its declared group/source/role")
    return inputs

def bind_comparator_grid(model, inputs: NativeComparatorInputs) -> None:
    shape = tuple(inputs.validity.shape[1:])
    previous = model.packing
    if (shape[1:] != previous.shape[1:] or max(shape[0],shape[1]*shape[2]) > previous.side):
        raise ValueError("Comparator family grid exceeds the shared fitted domain")
    model.packing = type(previous)(shape,previous.side)


def native_comparator_group_loss(model, inputs: NativeComparatorInputs, target: torch.Tensor,
                                 mask: torch.Tensor, action_weights: torch.Tensor,
                                 normalization, generator: torch.Generator) -> torch.Tensor:
    """One complete group, with posterior and teacher weights each applied once."""
    if not hasattr(model,"predict_native") and not hasattr(model,"loss_native"):
        bind_comparator_grid(model,inputs)
    b,j,a = inputs.group_shape
    if (target.shape != (b,a,*inputs.validity.shape[1:]) or mask.shape != target.shape
            or mask.dtype != torch.bool or action_weights.shape != (b,a)):
        raise ValueError("Comparator training target/action axes mismatch")
    if normalization.identity != inputs.normalization_id:
        raise ValueError("Comparator training normalization identity mismatch")
    if hasattr(model, "one_step_residual_loss"):
        return model.one_step_residual_loss(inputs, target, mask, action_weights, normalization)
    target = normalization.normalize(target,mask)
    expanded = target[:,None].expand(b,j,a,*target.shape[2:]).reshape_as(inputs.validity)
    valid = mask[:,None].expand(b,j,a,*mask.shape[2:]).reshape_as(inputs.validity) & inputs.validity
    weights = (inputs.weights[:,:,None]*action_weights[:,None,:]).reshape(-1)
    if hasattr(model,"loss_native"):
        return model.loss_native(inputs,expanded,valid,weights,generator)
    if hasattr(model,"predict_native"):
        predicted=normalization.normalize(model.predict_native(inputs,normalization),valid)
        count=valid.flatten(1).sum(1)
        if (count==0).any() or not torch.isfinite(weights).all() or weights.sum()<=0 or (weights<0).any():
            raise ValueError("Direct native objective needs supported fields and positive row weights")
        means=torch.where(valid,(predicted-expanded).square(),0).flatten(1).sum(1)/count
        return (means*weights).sum()/weights.sum()
    return model.loss(expanded,valid,inputs.condition,weights,generator)


def sample_native_comparator(model, inputs: NativeComparatorInputs, normalization, *,
                             draws_per_particle: int, steps: int, seed: int, guidance=None) -> dict:
    """Return all weighted particles; no calibrated law is fabricated for a mean."""
    if draws_per_particle < 1 or normalization.identity != inputs.normalization_id:
        raise ValueError("Comparator forecast draw/normalization contract mismatch")
    recurrent=hasattr(model,"sample_native_fields")
    if not recurrent:bind_comparator_grid(model,inputs)
    if recurrent and guidance is not None:raise ValueError("RSSM does not use diffusion physics guidance")
    b,j,a = inputs.group_shape
    generator = torch.Generator(device=inputs.condition.device).manual_seed(seed)
    draws = []
    for _ in range(draws_per_particle):
        args = (inputs.validity,inputs.condition,steps,generator)
        value = model.sample_native_fields(inputs,generator) if recurrent else (
            model.sample(*args) if guidance is None else model.sample(*args,guidance))
        value = normalization.inverse(value,inputs.validity)
        if recurrent:
            original=inputs.posterior.support.initial_field[:,:,None].expand(
                b,j,a,*inputs.initial.shape[-3:]).reshape(b*j*a,*inputs.initial.shape[-3:])
            value=torch.cat((original[:,None],value[:,1:]),1)
        draws.append(value.reshape(b,j,a,*value.shape[1:]))
    values = torch.stack(draws)
    weights = inputs.weights[None].expand(draws_per_particle,b,j)/draws_per_particle
    return {"particle_trajectory_samples":values,"particle_weights":inputs.weights,
            "path_weights":weights,"posterior_uid":inputs.posterior_uid,
            "particle_uids":inputs.particle_uids,"regimen_uids":inputs.regimen_uids,
            "sample_uids":inputs.sample_uids,"normalization_id":inputs.normalization_id,
            "draw_semantics":"weighted_complete_particle_conditional_samples",
            "conditional_draws_per_particle":draws_per_particle,
            "forecast_paths_per_action":draws_per_particle*j}
