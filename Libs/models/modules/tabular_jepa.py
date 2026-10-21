from __future__ import annotations

import torch
from torch import nn


class TabularMaskEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class TabularMaskPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)
