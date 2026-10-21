"""Weighted whole-particle reductions around the maintained native rollout."""
from __future__ import annotations

import torch

from Libs.utils.data.native_diffusion_data import stable_numeric_id
from Libs.utils.data.native_observable_profile import quantity_support, family_grid, observable_readout


def operator_splitting_record(diagnostics: dict[str, object]) -> dict:
    values = {key: value.detach().cpu().tolist() for key, value in diagnostics.items()
              if key.startswith("operator_splitting_") and key != "operator_splitting_identity"}
    mode = values.get("operator_splitting_parallel_update")
    if mode not in (0.0, 1.0):
        return {"diagnostic_status": "composition_mode_unavailable", "diagnostics": values}
    identity = diagnostics.get("operator_splitting_identity")
    if identity is None:
        return {"diagnostic_status": "composition_identity_unavailable", "diagnostics": values}
    if identity["composition_mode"] != ("parallel" if mode else "serial"):
        raise ValueError("Runtime expert identity and update mode disagree.")
    return {"diagnostic_status": "available", **identity, "diagnostics": values}


def weighted_particle_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape[:2] != weights.shape:
        raise ValueError("Particle reduction requires aligned sample/particle axes.")
    return (values * weights.reshape(*weights.shape, *([1] * (values.ndim - 2)))).sum(1)


def aggregate_joint_rollouts(outputs: list[dict], posterior, sampler) -> dict:
    weights = posterior.weights
    if len(outputs) != weights.shape[1]:
        raise ValueError("Native output count does not match joint support.")
    means = torch.stack([item["field_trajectory"] for item in outputs], dim=1)
    mean = weighted_particle_mean(means, weights)
    within = weighted_particle_mean(torch.stack([item["field_trajectory_std"].square() for item in outputs], dim=1), weights)
    between = weighted_particle_mean((means - mean.unsqueeze(1)).square(), weights)
    samples = torch.stack([item["field_trajectory_samples"] for item in outputs], dim=2)  # D,B,J,A,T,R,C,S
    mixture_draws, quantiles, draw_indices = mixture_sample_summary(samples, posterior, sampler)
    result = dict(outputs[0])
    if "quantity_support" in result:
        mask = result["quantity_support"]
        if mask.dtype != torch.bool or any(not torch.equal(row["quantity_support"], mask) for row in outputs):
            raise ValueError("Joint outputs require the same independent boolean quantity support")
    result["action_conditioning"] = [dict(record, particle_index=index)
                                     for index, item in enumerate(outputs) for record in item.get("action_conditioning", [])]
    result["particle_operator_splitting_diagnostics"] = [
        dict(operator_splitting_record(item["router_diagnostics"]), particle_index=index)
        for index, item in enumerate(outputs)]
    result["field_normalization_diagnostics"] = {key:torch.stack(
        [item["field_normalization_diagnostics"][key] for item in outputs],dim=1)
        for key in outputs[0].get("field_normalization_diagnostics", {})}
    sample_keys = ("disease_logit", "regimen_scores", "pkpd_consistency", "tri_score",
                   "field_training_reconstruction", "final_field", "transitioned_state",
                   "transition_delta", "transition_gate", "reconstructed_patient", "transitioned_observation",
                   "fast_field_trajectory", "slow_field_trajectory")
    for key in sample_keys:
        if key in result:
            result[key] = weighted_particle_mean(torch.stack([item[key] for item in outputs], 1), weights)
    if "tri_components" in result:
        result["tri_components"] = {key: weighted_particle_mean(torch.stack(
            [item["tri_components"][key] for item in outputs], 1), weights) for key in result["tri_components"]}
    per_action_loss = weighted_particle_mean(torch.stack([item["rollout_auxiliary_loss_per_action"] for item in outputs], 1), weights)
    result.update(field_trajectory=mean, field_trajectory_std=(within+between).clamp_min(0).sqrt(),
        field_trajectory_quantiles=torch.stack(quantiles), field_trajectory_samples=mixture_draws,
        within_diffusion_variance=within, between_particle_variance=between,
        particle_trajectory_samples=samples, particle_weights=weights,
        particle_field_trajectory=means,
        particle_training_reconstruction=torch.stack([item["field_training_reconstruction"] for item in outputs], 1),
        particle_uids=posterior.support.particle_uids, particle_draw_indices=draw_indices,
        posterior_uid=posterior.posterior_uid, posterior_status=posterior.status,
        rollout_auxiliary_loss_per_action=per_action_loss, rollout_auxiliary_loss=per_action_loss.mean(),
        baseline_field=weighted_particle_mean(posterior.support.initial_field, weights))
    final = mean[:, :, -1]
    result["exposure"] = final[..., 0].mean((0, 2))
    result["occupancy"] = final[..., 1].mean((0, 2))
    if posterior.support.quantity_profile is not None:
        grid=family_grid(posterior.support.quantity_profile,posterior.support.source_family_uid)
        b,a=mean.shape[:2]
        readout=observable_readout(mean.flatten(0,1),result["field_trajectory_std"].flatten(0,1),grid, torch.as_tensor(quantity_support(grid, posterior.support.source_uid), device=mean.device).expand_as(mean.flatten(0,1)))
        for key in ("predictive_mean","predictive_scale","feature_mask","time_mask"):
            readout[key]=readout[key].reshape(b,a,*readout[key].shape[1:])
        result["observable_prediction"]=readout
        support=result["quantity_support"][:,:,-1]
        spread=(result["field_trajectory_std"][:,:,-1]*support).sum((-1,-2,-3))/support.sum((-1,-2,-3))
    else:
        spread=result["field_trajectory_std"][:,:,-1].mean((-1,-2,-3))
    result["ranking_stability"] = torch.exp(-sampler.ranking_stability_scale * spread)
    return result


def posterior_draw_indices(posterior, sampler, draw_count: int) -> torch.Tensor:
    """The shared whole-particle draw budget, independent of residual sampling."""
    result = []
    for bi, sample_uid in enumerate(posterior.support.sample_uids):
        seed = (sampler.sampler_seed + stable_numeric_id(posterior.posterior_uid + sample_uid)) % (2**63 - 1)
        generator = torch.Generator(device=posterior.weights.device).manual_seed(seed)
        result.append(torch.multinomial(posterior.weights[bi], draw_count, replacement=True, generator=generator))
    return torch.stack(result)


def mixture_sample_summary(samples, posterior, sampler):
    weights = posterior.weights
    d, b, j = samples.shape[:3]
    # Resample one complete particle per diffusion draw, shared by every candidate
    # and time point. Identity is an RNG key only, never a learned input.
    draw_indices = posterior_draw_indices(posterior, sampler, d)
    draws = [torch.stack([samples[di, bi, draw_indices[bi, di]] for di in range(d)]) for bi in range(b)]
    mixture_draws = torch.stack(draws, dim=1)
    weighted_samples = samples.permute(1, 2, 0, *range(3, samples.ndim)).flatten(1, 2)
    sorted_samples, order = weighted_samples.sort(dim=1)
    sample_weights = (weights.unsqueeze(-1).expand(-1, -1, d) / d).flatten(1, 2)
    sample_weights = sample_weights.reshape(b, j*d, *([1] * (weighted_samples.ndim-2))).expand_as(weighted_samples)
    cdf = sample_weights.gather(1, order).cumsum(1)
    quantiles = []
    for quantile in sampler.quantile_levels:
        index = (cdf < quantile).sum(1).clamp_max(j*d-1)
        quantiles.append(sorted_samples.gather(1, index.unsqueeze(1)).squeeze(1))
    return mixture_draws, quantiles, draw_indices


def aggregate_joint_sequences(outputs: list[dict], posterior, sampler) -> dict:
    weights = posterior.weights
    result = {}
    for key, value in outputs[0].items():
        if key in {"field_trajectory_std_sequence", "field_trajectory_samples_sequence",
                   "observable_prediction_sequence"}:
            continue
        if key == "quantity_support_sequence":
            if value.dtype != torch.bool or any(not torch.equal(row[key], value) for row in outputs):
                raise ValueError("Joint sequences require the same independent boolean quantity support")
            result[key] = value
        elif key in {"operator_splitting_diagnostics", "action_conditioning"}:
            result[key] = [dict(record, particle_index=index) for index, item in enumerate(outputs)
                           for record in item[key]]
        elif key == "field_normalization_diagnostics":
            result[key] = {name:torch.stack([item[key][name] for item in outputs],dim=1) for name in value}
        elif key == "tri_components":
            result[key] = {name: weighted_particle_mean(torch.stack([item[key][name] for item in outputs], 1), weights)
                           for name in value}
        else:
            result[key] = weighted_particle_mean(torch.stack([item[key] for item in outputs], 1), weights)
    means = torch.stack([item["field_trajectory_sequence"] for item in outputs], 1)
    within = weighted_particle_mean(torch.stack([item["field_trajectory_std_sequence"].square() for item in outputs], 1), weights)
    between = weighted_particle_mean((means-result["field_trajectory_sequence"].unsqueeze(1)).square(), weights)
    samples = torch.stack([item["field_trajectory_samples_sequence"] for item in outputs], 2)
    draws, quantiles, indices = mixture_sample_summary(samples, posterior, sampler)
    result.update(field_trajectory_std_sequence=(within+between).clamp_min(0).sqrt(),
        field_trajectory_samples_sequence=draws, field_trajectory_quantiles_sequence=torch.stack(quantiles),
        within_diffusion_variance=within, between_particle_variance=between,
        particle_field_trajectory_sequence=means, particle_trajectory_samples=samples,
        particle_weights=weights, particle_uids=posterior.support.particle_uids,
        particle_draw_indices=indices, posterior_uid=posterior.posterior_uid)
    if posterior.support.quantity_profile is not None:
        mean = result["field_trajectory_sequence"]
        b, steps = mean.shape[:2]
        grid = family_grid(posterior.support.quantity_profile, posterior.support.source_family_uid)
        readout = observable_readout(mean.flatten(0, 1), result["field_trajectory_std_sequence"].flatten(0, 1), grid, torch.as_tensor(quantity_support(grid, posterior.support.source_uid), device=mean.device).expand_as(mean.flatten(0,1)))
        for key in ("predictive_mean", "predictive_scale", "feature_mask", "time_mask"):
            readout[key] = readout[key].reshape(b, steps, *readout[key].shape[1:])
        result["observable_prediction_sequence"] = readout
    return result
