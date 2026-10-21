from __future__ import annotations

import torch
from torch import nn

from Libs.models.modules.shared_latent import ModalityEncoder


class SharedLatentDualEncoder(nn.Module):
    def __init__(
        self,
        transcriptomics_input_dim: int,
        metabolomics_input_dim: int,
        transcriptomics_hidden_dim: int,
        metabolomics_hidden_dim: int,
        latent_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.transcriptomics_encoder = ModalityEncoder(
            transcriptomics_input_dim,
            transcriptomics_hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.metabolomics_encoder = ModalityEncoder(
            metabolomics_input_dim,
            metabolomics_hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.classifier = nn.Linear(latent_dim, 1)

    def encode(self, modality: str, features: torch.Tensor) -> torch.Tensor:
        if modality == "transcriptomics":
            return self.transcriptomics_encoder(features)
        if modality == "metabolomics":
            return self.metabolomics_encoder(features)
        raise ValueError(f"Unsupported modality: {modality}")

    def forward(self, modality: str, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(modality, features)
        logits = self.classifier(latent).squeeze(-1)
        return logits, latent
