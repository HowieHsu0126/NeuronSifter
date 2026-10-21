from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class PhysicsStateSliceAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_count: int, slice_count: int, dropout_rate: float) -> None:
        super().__init__()
        if hidden_dim % head_count != 0:
            raise ValueError(f"Transolver hidden_dim {hidden_dim} must be divisible by head_count {head_count}.")
        self.head_count = int(head_count)
        self.head_dim = int(hidden_dim // head_count)
        self.slice_count = int(slice_count)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.slice_projection = nn.Linear(self.head_dim, self.slice_count)
        self.temperature_projection = nn.Sequential(
            nn.Linear(self.head_dim, self.slice_count),
            nn.GELU(),
            nn.Linear(self.slice_count, 1),
            nn.GELU(),
        )
        self.slice_temperature_bias = nn.Parameter(torch.full((1, self.head_count, 1, 1), 0.5))
        self.query_projection = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.output_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout_rate))
        nn.init.orthogonal_(self.slice_projection.weight)

    def forward(self, node_state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, token_count, hidden_dim = node_state.shape
        projected_state = self.input_projection(node_state).reshape(batch_size, token_count, self.head_count, self.head_dim)
        projected_state = projected_state.permute(0, 2, 1, 3).contiguous()
        temperature = torch.clamp(self.temperature_projection(projected_state) + self.slice_temperature_bias, min=torch.finfo(node_state.dtype).eps)
        slice_logits = self.slice_projection(projected_state) / temperature
        slice_weights = torch.softmax(slice_logits, dim=-1)
        slice_mass = slice_weights.sum(dim=2).clamp_min(torch.finfo(node_state.dtype).tiny)
        slice_tokens = torch.einsum("bhnc,bhns->bhsc", projected_state, slice_weights)
        slice_tokens = slice_tokens / slice_mass.unsqueeze(-1)
        query = self.query_projection(slice_tokens)
        key = self.key_projection(slice_tokens)
        value = self.value_projection(slice_tokens)
        attended_tokens = F.scaled_dot_product_attention(query, key, value)
        node_message = torch.einsum("bhsc,bhns->bhnc", attended_tokens, slice_weights)
        node_message = node_message.permute(0, 2, 1, 3).contiguous().reshape(batch_size, token_count, hidden_dim)
        slice_entropy = -torch.sum(
            slice_weights * torch.log(slice_weights.clamp_min(torch.finfo(slice_weights.dtype).tiny)),
            dim=-1,
        ).mean()
        diagnostics = {
            "slice_entropy": slice_entropy,
            "slice_temperature_mean": temperature.mean(),
        }
        return self.output_projection(node_message), diagnostics


class TransolverPlusMicroenvironmentBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_count: int,
        slice_count: int,
        dropout_rate: float,
        mlp_multiplier: int,
        checkpoint_attention: bool,
    ) -> None:
        super().__init__()
        self.checkpoint_attention = bool(checkpoint_attention)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = PhysicsStateSliceAttention(hidden_dim, head_count, slice_count, dropout_rate)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * int(mlp_multiplier)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * int(mlp_multiplier), hidden_dim),
        )

    def conditioned_attention_input(self, hidden_state: torch.Tensor, conditioning_state: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition_projection(conditioning_state).chunk(2, dim=-1)
        return self.attention_norm(hidden_state) * (1.0 + torch.tanh(scale)) + shift

    def forward(self, hidden_state: torch.Tensor, conditioning_state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attention_input = self.conditioned_attention_input(hidden_state, conditioning_state)
        if self.training and self.checkpoint_attention:
            attention_output = checkpoint(lambda checkpoint_input: self.attention(checkpoint_input)[0], attention_input, use_reentrant=False)
            with torch.no_grad():
                _, diagnostics = self.attention(attention_input.detach())
        else:
            attention_output, diagnostics = self.attention(attention_input)
        hidden_state = hidden_state + attention_output
        hidden_state = hidden_state + self.feed_forward(self.feed_forward_norm(hidden_state))
        return hidden_state, diagnostics
