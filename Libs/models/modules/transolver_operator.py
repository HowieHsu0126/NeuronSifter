from __future__ import annotations

import torch
from torch import nn

from Libs.models.layers.transolver_attention import TransolverPlusMicroenvironmentBlock
from Libs.models.modules.moe import SemanticMechanismMixtureOfExperts


class TransolverMicroenvironmentOperator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        encoder_depth: int,
        decoder_depth: int,
        head_count: int,
        slice_count: int,
        coordinate_dim: int,
        dropout_rate: float,
        expert_names: list[str],
        top_k: int,
        capacity_factor: float,
        entropy_loss_weight: float,
        load_balance_loss_weight: float,
        expert_capacity_loss_weight: float,
        expert_hidden_multiplier: int,
        physics_mlp_multiplier: int,
        checkpoint_attention_blocks: bool,
    ) -> None:
        super().__init__()
        self.encoder_depth = int(encoder_depth)
        self.decoder_depth = int(decoder_depth)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        self.region_axis_proj = nn.Linear(1, hidden_dim)
        self.compartment_axis_proj = nn.Linear(1, hidden_dim)
        self.temporal_axis_proj = nn.Linear(1, hidden_dim)
        self.placeholder = nn.Parameter(torch.zeros(hidden_dim))
        self.mechanism_moe = SemanticMechanismMixtureOfExperts(
            hidden_dim,
            expert_names,
            top_k,
            dropout_rate,
            capacity_factor,
            entropy_loss_weight,
            load_balance_loss_weight,
            expert_capacity_loss_weight,
            expert_hidden_multiplier,
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [
                TransolverPlusMicroenvironmentBlock(
                    hidden_dim,
                    head_count,
                    slice_count,
                    dropout_rate,
                    physics_mlp_multiplier,
                    checkpoint_attention_blocks,
                )
                for _ in range(self.encoder_depth)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransolverPlusMicroenvironmentBlock(
                    hidden_dim,
                    head_count,
                    slice_count,
                    dropout_rate,
                    physics_mlp_multiplier,
                    checkpoint_attention_blocks,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        nn.init.trunc_normal_(self.placeholder, std=0.02)

    def geometry_embedding(self, node_coordinates: torch.Tensor, sequence_length: int) -> torch.Tensor:
        region_axis = node_coordinates[..., 0:1]
        compartment_axis = node_coordinates[..., 1:2]
        temporal_axis = torch.linspace(
            0.0,
            1.0,
            sequence_length,
            dtype=node_coordinates.dtype,
            device=node_coordinates.device,
        ).view(1, sequence_length, 1, 1)
        temporal_axis = temporal_axis.expand(node_coordinates.shape[0], -1, node_coordinates.shape[2], -1)
        return (
            self.coordinate_proj(node_coordinates)
            + self.region_axis_proj(region_axis)
            + self.compartment_axis_proj(compartment_axis)
            + self.temporal_axis_proj(temporal_axis)
        )

    def forward(
        self,
        node_state: torch.Tensor,
        conditioning_state: torch.Tensor,
        node_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden_state = node_state + conditioning_state + self.geometry_embedding(node_coordinates, node_state.shape[1])
        hidden_state = hidden_state + self.placeholder.view(1, 1, 1, -1)
        routed_state, diagnostics = self.mechanism_moe(hidden_state)
        hidden_state = self.input_norm(hidden_state + routed_state)
        block_diagnostics: list[dict[str, torch.Tensor]] = []
        flat_hidden_state = hidden_state.reshape(hidden_state.shape[0], hidden_state.shape[1] * hidden_state.shape[2], hidden_state.shape[3])
        flat_conditioning_state = conditioning_state.reshape(conditioning_state.shape[0], conditioning_state.shape[1] * conditioning_state.shape[2], conditioning_state.shape[3])
        for block in self.encoder_blocks:
            flat_hidden_state, block_payload = block(flat_hidden_state, flat_conditioning_state)
            block_diagnostics.append(block_payload)
        for block in self.decoder_blocks:
            flat_hidden_state, block_payload = block(flat_hidden_state, flat_conditioning_state)
            block_diagnostics.append(block_payload)
        hidden_state = flat_hidden_state.reshape_as(hidden_state)
        if block_diagnostics:
            diagnostics["slice_entropy"] = torch.stack([payload["slice_entropy"] for payload in block_diagnostics]).mean()
            diagnostics["slice_temperature_mean"] = torch.stack([payload["slice_temperature_mean"] for payload in block_diagnostics]).mean()
        diagnostics["encoder_block_count"] = torch.tensor(float(self.encoder_depth), device=hidden_state.device, dtype=hidden_state.dtype)
        diagnostics["decoder_block_count"] = torch.tensor(float(self.decoder_depth), device=hidden_state.device, dtype=hidden_state.dtype)
        return hidden_state, diagnostics
