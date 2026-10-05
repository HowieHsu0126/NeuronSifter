from __future__ import annotations

import torch
from torch import nn

from Libs.models.modules.tabular_jepa import TabularMaskEncoder, TabularMaskPredictor


class TabularJEPAAuxiliaryEncoder(nn.Module):
    def __init__(
        self,
        transcriptomics_input_dim: int,
        metabolomics_input_dim: int,
        transcriptomics_hidden_dim: int,
        metabolomics_hidden_dim: int,
        predictor_hidden_dim: int,
        latent_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.transcriptomics_encoder = TabularMaskEncoder(
            transcriptomics_input_dim,
            transcriptomics_hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.metabolomics_encoder = TabularMaskEncoder(
            metabolomics_input_dim,
            metabolomics_hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.transcriptomics_predictor = TabularMaskPredictor(
            latent_dim,
            predictor_hidden_dim,
            transcriptomics_input_dim,
            dropout_rate,
        )
        self.metabolomics_predictor = TabularMaskPredictor(
            latent_dim,
            predictor_hidden_dim,
            metabolomics_input_dim,
            dropout_rate,
        )
        self.classifier = nn.Linear(latent_dim, 1)

    def encode(self, modality: str, features: torch.Tensor) -> torch.Tensor:
        if modality == "transcriptomics":
            return self.transcriptomics_encoder(features)
        if modality == "metabolomics":
            return self.metabolomics_encoder(features)
        raise ValueError(f"Unsupported modality: {modality}")

    def predict(self, modality: str, latent: torch.Tensor) -> torch.Tensor:
        if modality == "transcriptomics":
            return self.transcriptomics_predictor(latent)
        if modality == "metabolomics":
            return self.metabolomics_predictor(latent)
        raise ValueError(f"Unsupported modality: {modality}")

    def forward(self, modality: str, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(modality, features)
        logits = self.classifier(latent).squeeze(-1)
        reconstruction = self.predict(modality, latent)
        return logits, latent, reconstruction
