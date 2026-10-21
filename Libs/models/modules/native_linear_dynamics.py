"""Full-field affine conditional means on exact native-hour intervals."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

LINEAR_DYNAMICS_SPEC = {
    "equation": "dx_dtau=A*x+B*p_left+C*z+b",
    "state_coordinates": "fixed_r60_normalized",
    "program_hold": "left_endpoint",
    "transition": "augmented_matrix_exponential",
    "transition_dtype": "float64",
    "time_unit": "hour",
    "initialization": "zero_predictor",
}


class NativeLinearDynamics(nn.Module):
    def __init__(self, domain: dict, posterior_context_dim: int, *, program_channel_count: int = 2,
                 normalization_scope: str = "fixed_domain") -> None:
        super().__init__()
        self.domain = copy.deepcopy(domain)
        self.posterior_context_dim = posterior_context_dim
        if program_channel_count < 2 or normalization_scope not in {"fixed_domain","registered_native"}:
            raise ValueError("Linear program/normalization contract is invalid")
        self.program_channel_count = program_channel_count
        self.normalization_scope = normalization_scope
        for axis in ("region_id", "compartment_id", "species_id"):
            labels = self.domain[axis]
            if not labels or not all(isinstance(item, str) and item for item in labels) or len(set(labels)) != len(labels):
                raise ValueError("Linear dynamics requires unique full-field axis identities")
        clock = self.domain["time_hours"]
        if (self.domain["physical_time_unit"] != "hour" or len(clock) < 2
                or not all(isinstance(t, (float, int)) and not isinstance(t, bool) and math.isfinite(t) for t in clock)
                or any(right <= left for left, right in zip(clock, clock[1:]))):
            raise ValueError("Linear dynamics requires a positive-interval native-hour source grid")
        if not isinstance(posterior_context_dim, int) or isinstance(posterior_context_dim, bool) or posterior_context_dim < 1:
            raise ValueError("Linear dynamics requires the full source posterior context width")
        self.field_shape = tuple(len(self.domain[key]) for key in ("region_id", "compartment_id", "species_id"))
        r, c, s = self.field_shape
        self.state_dim = r*c*s
        self.A = nn.Parameter(torch.zeros(self.state_dim, self.state_dim))
        self.B = nn.Parameter(torch.zeros(self.state_dim, program_channel_count*r*c))
        self.C = nn.Parameter(torch.zeros(self.state_dim, r*c*posterior_context_dim))
        self.b = nn.Parameter(torch.zeros(self.state_dim))

    def get_extra_state(self) -> dict:
        result = {"schema_version": "native_linear_dynamics.v1", "definition": dict(LINEAR_DYNAMICS_SPEC),
                  "domain": copy.deepcopy(self.domain), "posterior_context_dim": self.posterior_context_dim}
        if self.program_channel_count != 2 or self.normalization_scope != "fixed_domain":
            result.update(program_channel_count=self.program_channel_count, normalization_scope=self.normalization_scope)
            result["definition"]["state_coordinates"] = "shared_training_only_native_multigrid_normalized"
        return result

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("Linear dynamics checkpoint equation/grid/context identity mismatch")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if state_dict.get(prefix+"_extra_state") != self.get_extra_state():
            raise ValueError("Linear dynamics checkpoint equation/grid/context identity mismatch")
        for name in ("A", "B", "C", "b"):
            value = state_dict.get(prefix+name)
            if (not torch.is_tensor(value) or value.shape != getattr(self, name).shape
                    or not torch.isfinite(value).all()):
                raise ValueError("Linear dynamics checkpoint parameter shape/finite-value mismatch: "+name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, context: dict, normalization) -> dict:
        required = {"initial_state", "posterior_context", "occupancy", "support", "physical_time"}
        if not required <= set(context) or any(not torch.is_tensor(context[key]) for key in required):
            raise ValueError("Linear dynamics requires admitted initial/posterior/program tensors")
        x0, z = context["initial_state"], context["posterior_context"]
        occupancy, support, times = (context[key] for key in ("occupancy", "support", "physical_time"))
        if x0.ndim != 5 or tuple(x0.shape[2:]) != self.field_shape or min(x0.shape[:2]) < 1:
            raise ValueError("Linear initial field requires full sample/action/region/compartment/species axes")
        b, a, r, c, s = x0.shape
        if (z.shape != (b,a,r,c,self.posterior_context_dim)
                or occupancy.ndim != 6 or occupancy.shape[:2] != (b,a)
                or occupancy.shape[3:] != (r,c,1) or occupancy.shape[2] < 2
                or support.shape != occupancy.shape or times.shape != occupancy.shape):
            raise ValueError("Linear posterior/program shape mismatch")
        if not all(torch.isfinite(context[key]).all() for key in required):
            raise ValueError("Linear conditioning must be finite")
        if not (((occupancy >= 0) & (occupancy <= 1)).all() and ((support >= 0) & (support <= 1)).all()):
            raise ValueError("Linear occupancy/support outside [0,1]")
        if not normalization.active:
            raise ValueError("Linear dynamics requires its fitted normalization")
        if self.normalization_scope == "fixed_domain" and normalization.artifact["domain"] != self.domain:
            raise ValueError("Linear dynamics requires its fixed R60 domain/normalization")
        if self.normalization_scope == "registered_native":
            grid = context.get("native_grid")
            if not isinstance(grid,dict):
                raise ValueError("Linear native dynamics requires the admitted registered grid")
            normalization.assert_domain(grid)
            if any(grid[key] != self.domain[key] for key in ("region_id","compartment_id","species_id")):
                raise ValueError("Linear native spatial/quantity domain mismatch")
        normalization.assert_physical_time(times)
        clock = times[0,0,:,0,0,0]
        if not torch.equal(times, clock[None,None,:,None,None,None].expand_as(times)) or not (clock[1:] > clock[:-1]).all():
            raise ValueError("Linear physical time requires one strictly increasing source grid")
        if not all(torch.isfinite(value).all() for value in self.parameters()):
            raise ValueError("Linear predictor parameters must be finite")
        mask = context.get("quantity_support")
        if mask is not None and (mask.dtype != torch.bool or mask.shape != (b,a,occupancy.shape[2],r,c,s)
                or not torch.equal(mask,mask[:1,:1,:1].expand_as(mask))):
            raise ValueError("Linear native quantity support must be constant on the admitted clock/actions")
        initial_mask = None if mask is None else mask[:,:,0]
        programs = [occupancy, support]
        if self.program_channel_count != 2:
            forcing = context.get("additional_forcing")
            if (not torch.is_tensor(forcing) or forcing.shape != (*occupancy.shape[:-1],self.program_channel_count-2)
                    or not torch.isfinite(forcing).all()):
                raise ValueError("Linear typed forcing is missing or malformed")
            programs.append(forcing)
        # Only the transition arithmetic is promoted. Casting parameters retains
        # autograd to their optimizer-owned tensors; no inverse of A is required.
        with torch.autocast(device_type=x0.device.type, enabled=False):
            state = normalization.normalize(x0.double(),initial_mask).reshape(b*a, self.state_dim)
            program = torch.cat(programs, -1).double().reshape(b*a, occupancy.shape[2], self.program_channel_count*r*c)
            static = z.double().reshape(b*a, -1) @ self.C.double().T + self.b.double()
            matrix = self.A.double()
            active = None if mask is None else mask[0,0,0].reshape(-1).to(matrix)
            if active is not None:
                # Unsupported cells cannot become hidden dynamical intermediates.
                matrix = matrix * active[:,None] * active[None,:]
            zero = torch.zeros_like(matrix)
            augmented = torch.cat((torch.cat((matrix, torch.eye(self.state_dim, dtype=torch.float64, device=matrix.device)), 1),
                                   torch.cat((zero, zero), 1)), 0)
            transitions = {}  # Shared intervals reuse this forward graph only.
            states = [state]
            for index, delta in enumerate(clock.double().diff()):
                key = float(delta.detach())
                if key not in transitions:
                    exponential = torch.matrix_exp(augmented * delta)
                    if not torch.isfinite(exponential).all():
                        raise FloatingPointError("Linear transition exponential is nonfinite")
                    transitions[key] = (exponential[:self.state_dim,:self.state_dim],
                                        exponential[:self.state_dim,self.state_dim:])
                phi, psi = transitions[key]
                forcing = program[:,index] @ self.B.double().T + static
                if active is not None:
                    forcing = forcing * active
                state = state @ phi.T + forcing @ psi.T
                if not torch.isfinite(state).all():
                    raise FloatingPointError("Linear propagation is nonfinite")
                states.append(state)
            normalized = torch.stack(states, 1).reshape(b,a,occupancy.shape[2],r,c,s)
            physical = torch.cat((x0.unsqueeze(2), normalization.inverse(normalized[:,:,1:], None if mask is None else mask[:,:,1:]).to(x0.dtype)), 2)
            normalized = normalized.to(x0.dtype)
        if not torch.isfinite(physical).all() or not torch.isfinite(normalized).all():
            raise FloatingPointError("Linear output cannot be represented in the source dtype")
        return {"field_trajectory": physical, "normalized_trajectory": normalized,
                "router_diagnostics": {"status": "not_applicable", "reason": "linear_dynamics_has_no_router",
                                       "router_regularization_loss": normalized.new_zeros(())},
                "predictor_kind": "linear_dynamics_direct_mean", "prediction_kind": "mean_only", "residual_likelihood_status": "unbound",
                "predictive_law_draw_count": 0}
