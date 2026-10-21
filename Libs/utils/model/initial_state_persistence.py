"""Identity dynamics on the independently available initial native belief."""
from __future__ import annotations

import torch

from Libs.utils.model.joint_rollout import mixture_sample_summary


def initial_particle_fields(posterior, action_count: int, time_count: int) -> torch.Tensor:
    """All explicit atoms of the identity-transition predictive law."""
    initial = posterior.support.initial_field
    if (action_count < 1 or time_count < 1 or initial.ndim != 5
            or not torch.isfinite(initial).all()):
        raise ValueError("Persistence requires a finite complete prior and action/clock axes")
    return initial[:, :, None, None].expand(-1,-1,action_count,time_count,-1,-1,-1)


def predict_initial_state_persistence(*, posterior, initial_state: torch.Tensor | None,
                                      action_count: int, time_count: int, sampler) -> dict:
    if action_count < 1 or time_count < 1:
        raise ValueError("Persistence requires the complete nonempty action/horizon axes")
    law = {"transition": "identity", "fitted_transition_parameter_count": 0,
           "transition_model_call_count": 0, "denoising_call_count": 0}
    if posterior is not None:
        fields = initial_particle_fields(posterior,action_count,time_count)
        # Each conditional law is constant. Only the existing whole-particle
        # categorical selection below generates the actual Monte Carlo draws.
        conditional = fields.unsqueeze(0).expand(sampler.inference_sample_count, *fields.shape)
        draws, _, indices = mixture_sample_summary(conditional, posterior, sampler)
        law.update(kind="emitted_whole_particle_mixture_draws",
                   initial_state_uncertainty="inherited_initial_posterior_only")
        return {"field_trajectory_samples": draws, "particle_trajectory_samples": conditional,
                "particle_field_trajectory": fields, "particle_draw_indices": indices,
                "particle_weights": posterior.weights, "particle_uids": posterior.support.particle_uids,
                "posterior_uid": posterior.posterior_uid, "prediction_law": law}
    if (not torch.is_tensor(initial_state) or initial_state.ndim != 5
            or initial_state.shape[1] != action_count or not torch.isfinite(initial_state).all()):
        raise ValueError("Persistence requires a finite independently bound point initial field")
    if not torch.equal(initial_state, initial_state[:, :1].expand_as(initial_state)):
        raise ValueError("Persistence point initial state must agree across all actions")
    law.update(kind="deterministic_point_mass",
               initial_state_uncertainty="theoretical_point_mass_baseline_assumption")
    return {"field_trajectory": initial_state.unsqueeze(2).expand(-1, -1, time_count, -1, -1, -1),
            "prediction_law": law}
