"""Direct conditional field means around the maintained Transolver operator."""
from __future__ import annotations

import torch
from torch import nn


class NativeTransolverMean(nn.Module):
    def __init__(self, species_count: int, hidden_dim: int, posterior_context_dim: int, *, program_channel_count: int = 2) -> None:
        super().__init__()
        self.species_count = species_count
        self.program_channel_count = program_channel_count
        self.posterior_context_dim = posterior_context_dim
        self.initial_projection = nn.Linear(species_count, hidden_dim)
        self.neighborhood_projection = nn.Linear(species_count, hidden_dim)
        self.program_projection = nn.Linear(program_channel_count, hidden_dim)
        self.posterior_projection = nn.Linear(posterior_context_dim, hidden_dim)
        self.residual_projection = nn.Linear(hidden_dim, species_count)

    def forward(self, context: dict, operator: nn.Module, normalization, adjacency: torch.Tensor,
                coordinates: torch.Tensor) -> dict:
        required = {"initial_state", "posterior_context", "occupancy", "support", "physical_time"}
        if not required <= set(context):
            raise ValueError("Native direct mean requires independently admitted initial/posterior/program inputs")
        x0, posterior = context["initial_state"], context["posterior_context"]
        occupancy, support, times = (context[key] for key in ("occupancy", "support", "physical_time"))
        if x0.ndim != 5 or x0.shape[-1] != self.species_count:
            raise ValueError("Direct initial state requires sample/action/region/compartment/species axes")
        b, a, r, c, s = x0.shape
        if (posterior.shape != (b, a, r, c, self.posterior_context_dim)
                or occupancy.ndim != 6 or occupancy.shape[:2] != (b, a)
                or occupancy.shape[3:] != (r, c, 1)
                or support.shape != occupancy.shape or times.shape != occupancy.shape):
            raise ValueError("Direct posterior/program shape mismatch")
        if not all(torch.isfinite(context[key]).all() for key in required):
            raise ValueError("Direct conditioning must be finite")
        if not (((occupancy >= 0) & (occupancy <= 1)).all() and ((support >= 0) & (support <= 1)).all()):
            raise ValueError("Direct occupancy/support outside [0,1]")
        t = occupancy.shape[2]
        if (t < 2 or not torch.equal(times, times[..., :1, :1, :].expand_as(times))
                or not (times[:, :, 1:] > times[:, :, :-1]).all()):
            raise ValueError("Direct physical time requires a shared strictly increasing grid")
        if adjacency.shape != (r*c, r*c) or coordinates.shape != (r*c, 2):
            raise ValueError("Direct graph/node identity shape mismatch")
        if not normalization.active:
            raise ValueError("Direct mean requires fixed R60 normalization")
        mask = context.get("quantity_support")
        if mask is not None and (mask.dtype != torch.bool or mask.shape != (b,a,t,r,c,s)):
            raise ValueError("Direct quantity support differs from the complete native trajectory")
        initial_mask = None if mask is None else mask[:,:,0]
        initial = normalization.normalize(x0, initial_mask).reshape(b*a, r*c, s)
        neighborhood = torch.einsum("nm,bms->bns", adjacency.to(initial), initial)
        hidden = (self.initial_projection(initial) + self.neighborhood_projection(neighborhood))[:, None].expand(-1, t, -1, -1)
        posterior_hidden = self.posterior_projection(posterior.reshape(b*a, r*c, -1))[:, None]
        programs = [occupancy, support]
        if self.program_channel_count != 2:
            forcing = context.get("additional_forcing")
            if (not torch.is_tensor(forcing) or forcing.shape != (*occupancy.shape[:-1],self.program_channel_count-2)
                    or not torch.isfinite(forcing).all()):
                raise ValueError("Direct typed forcing is missing or malformed")
            programs.append(forcing)
        program = torch.cat(programs, -1).reshape(b*a, t, r*c, self.program_channel_count)
        conditioning = self.program_projection(program) + posterior_hidden
        coords = coordinates.to(initial)[None, None].expand(b*a, t, -1, -1)
        encoded, diagnostics = operator(hidden, conditioning, coords, physical_time=times.reshape(b*a, t, r*c, 1))
        residual = self.residual_projection(encoded).reshape(b, a, t, r, c, s)
        normalized = normalization.normalize(x0, initial_mask).unsqueeze(2) + residual
        # The initial snapshot is source-owned, at its original physical time.
        normalized = torch.cat((normalization.normalize(x0, initial_mask).unsqueeze(2), normalized[:, :, 1:]), 2)
        physical = torch.cat((x0.unsqueeze(2), normalization.inverse(normalized[:, :, 1:], None if mask is None else mask[:,:,1:])), 2)
        if mask is not None:
            normalized = normalized.masked_fill(~mask,0.)
        return {"field_trajectory": physical, "normalized_trajectory": normalized,
                "router_diagnostics": diagnostics, "prediction_kind": "mean_only",
                "residual_likelihood_status": "unbound", "predictive_law_draw_count": 0}
