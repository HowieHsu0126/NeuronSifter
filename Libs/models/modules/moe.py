from __future__ import annotations

import torch
from torch import nn


class SemanticMechanismMixtureOfExperts(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expert_names: list[str],
        top_k: int,
        dropout_rate: float,
        capacity_factor: float,
        entropy_loss_weight: float = 0.0,
        load_balance_loss_weight: float = 0.0,
        expert_capacity_loss_weight: float = 0.0,
        expert_hidden_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if not expert_names:
            raise ValueError("SemanticMechanismMixtureOfExperts requires at least one expert name.")
        if top_k < 1 or top_k > len(expert_names):
            raise ValueError(f"MoE top_k must be within [1, expert_count], got {top_k} for {len(expert_names)} experts.")
        self.expert_names = list(expert_names)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.load_balance_loss_weight = float(load_balance_loss_weight)
        self.expert_capacity_loss_weight = float(expert_capacity_loss_weight)
        self.expert_hidden_multiplier = int(expert_hidden_multiplier)
        self.router = nn.Linear(hidden_dim, len(self.expert_names))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * self.expert_hidden_multiplier),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim * self.expert_hidden_multiplier, hidden_dim),
                )
                for _ in self.expert_names
            ]
        )

    def forward(self, hidden_state: torch.Tensor, *, casewise_diagnostics: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_logits = self.router(hidden_state)
        selected_logits, selected_indices = torch.topk(router_logits, self.top_k, dim=-1)
        selected_weights = torch.softmax(selected_logits, dim=-1).to(dtype=router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter(-1, selected_indices, selected_weights)
        flat_hidden = hidden_state.reshape(-1, hidden_state.shape[-1])
        flat_router_weights = router_weights.reshape(-1, router_weights.shape[-1])
        flat_routed_state = torch.zeros_like(flat_hidden)
        for expert_index, expert in enumerate(self.experts):
            selected_token_mask = flat_router_weights[:, expert_index] > 0
            if not torch.any(selected_token_mask):
                continue
            expert_input = flat_hidden[selected_token_mask]
            expert_weight = flat_router_weights[selected_token_mask, expert_index].unsqueeze(-1)
            flat_routed_state[selected_token_mask] = flat_routed_state[selected_token_mask] + expert_weight * expert(expert_input)
        routed_state = flat_routed_state.reshape_as(hidden_state)
        flat_weights = router_weights.reshape(-1, router_weights.shape[-1])
        utilization = (router_weights.reshape(hidden_state.shape[0], -1, router_weights.shape[-1]).mean(dim=1)
                       if casewise_diagnostics else flat_weights.mean(dim=0))
        load_balance = utilization * float(len(self.expert_names))
        router_entropy = -torch.sum(
            router_weights * torch.log(router_weights.clamp_min(torch.finfo(router_weights.dtype).tiny)),
            dim=-1,
        )
        router_entropy = (router_entropy.reshape(hidden_state.shape[0], -1).mean(dim=1)
                          if casewise_diagnostics else router_entropy.mean())
        capacity_target = torch.full_like(utilization, self.capacity_factor / float(len(self.expert_names)))
        capacity_excess = torch.relu(utilization - capacity_target).mean(dim=-1)
        load_balance_loss = ((load_balance - torch.ones_like(load_balance)) ** 2).mean(dim=-1)
        router_regularization_loss = (
            self.entropy_loss_weight * (-router_entropy)
            + self.load_balance_loss_weight * load_balance_loss
            + self.expert_capacity_loss_weight * capacity_excess
        )
        diagnostics = {
            "expert_utilization": utilization,
            "router_entropy": router_entropy,
            "capacity_excess": capacity_excess,
            "load_balance_loss": load_balance_loss,
            "router_regularization_loss": router_regularization_loss,
        }
        return routed_state, diagnostics
