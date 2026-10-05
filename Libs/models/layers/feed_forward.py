from __future__ import annotations

import torch
from torch import nn


def required_policy_vector(
    policy: dict[str, object],
    field_name: str,
    expected_length: int,
) -> torch.Tensor:
    if field_name not in policy:
        raise ValueError(f"Missing required screening policy field: {field_name}")
    values = [float(value) for value in list(policy[field_name])]
    if len(values) != expected_length:
        raise ValueError(
            f"Screening policy field {field_name} must have length {expected_length}, got {len(values)}."
        )
    return torch.tensor(values, dtype=torch.float32)


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
