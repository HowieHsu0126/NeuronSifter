"""Native-field Tau-BNO adaptation of arXiv:2603.08108v2, equations9--18.

The spatial Fourier/difference axis is the declared regional index, with
compartment/quantity entries as channels. It is not an anatomical distance.
The complete conditioned native program replaces the source's small parameter
vector. The QO still consumes only the initial field. No author source code is used;
this implementation follows the pinned paper equations, with a zero-sum mean-centered
difference stencil resolving the paper's sum/mean notation inconsistency.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from Libs.utils.model.dynamics_fields import (
    build_spatial_region_rows, build_spatial_region_edge_rows, region_adjacency_matrix,
)


def directed_proximities(adjacency: torch.Tensor) -> torch.Tensor:
    """Eq14: normalized symmetric, second-order inbound and outbound graphs."""
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or not torch.isfinite(adjacency).all() or (adjacency < 0).any()):
        raise ValueError("Tau graph needs a finite nonnegative square adjacency")
    incoming, outgoing = adjacency.sum(0), adjacency.sum(1)
    inv_in = incoming.reciprocal().masked_fill(incoming == 0, 0)
    inv_out = outgoing.reciprocal().masked_fill(outgoing == 0, 0)
    graphs = torch.stack(((adjacency+adjacency.T)/2,
        adjacency.T @ torch.diag(inv_out) @ adjacency,
        adjacency @ torch.diag(inv_in) @ adjacency.T))
    graphs = graphs+torch.eye(len(adjacency), device=adjacency.device, dtype=adjacency.dtype)
    inverse = graphs.sum(-1).rsqrt()
    return graphs*inverse[:, :, None]*inverse[:, None, :]


class FourierDifferentialLayer(nn.Module):
    """Pointwise + Fourier integral + a zero-sum local differential kernel."""
    def __init__(self, width: int, region_count: int, requested_modes: int, spacing: float):
        super().__init__()
        if width < 1 or region_count < 2 or requested_modes < 1 or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Tau operator dimensions/modes/spacing must be positive")
        self.region_count, self.requested_modes = region_count, requested_modes
        self.actual_modes, self.spacing = min(requested_modes, region_count//2+1), spacing
        self.pointwise = nn.Linear(width, width)
        self.spectral = nn.Parameter(torch.randn(width, width, self.actual_modes, 2)/width)
        self.stencil = nn.Parameter(torch.randn(width, width, 3)/(3*width))

    def differential(self, values: torch.Tensor) -> torch.Tensor:
        # Subtract neighbor values before contraction so every constant field
        # vanishes exactly, including at circular regional-index boundaries.
        centered = self.stencil-self.stencil.mean(-1, keepdim=True)
        channels = values.transpose(1, 2)
        differences = torch.stack((torch.roll(channels, 1, -1)-channels,
                                   torch.zeros_like(channels),
                                   torch.roll(channels, -1, -1)-channels), -1)
        return torch.einsum("nirk,oik->nor", differences, centered).transpose(1, 2)/self.spacing

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.region_count:
            raise ValueError("Tau Fourier layer requires batch/region/channel axes")
        frequency = torch.fft.rfft(values, dim=1)
        weight = torch.view_as_complex(self.spectral.contiguous())
        transformed = torch.zeros_like(frequency)
        transformed[:, :self.actual_modes] = torch.einsum(
            "nmi,iom->nmo", frequency[:, :self.actual_modes], weight)
        integral = torch.fft.irfft(transformed, n=self.region_count, dim=1)
        return F.gelu(self.pointwise(values)+integral+self.differential(values))


class TauFunctionOperator(nn.Module):
    def __init__(self, input_width: int, width: int, regions: int, modes: int, spacing: float):
        super().__init__()
        self.lift = nn.Linear(input_width, width)
        self.layers = nn.ModuleList(FourierDifferentialLayer(width, regions, modes, spacing) for _ in range(2))
        self.project = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.lift(values)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.project(hidden)


class TauDirectedGraphOperator(nn.Module):
    """Three independent two-layer graph branches, fused only at their outputs."""
    def __init__(self, width: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("proximities", directed_proximities(adjacency))
        self.lift = nn.Linear(width, width)
        self.self_layers = nn.ModuleList(nn.ModuleList(nn.Linear(width, width) for _ in range(2))
                                        for _ in range(3))
        self.graph_layers = nn.ModuleList(nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
                                         for _ in range(3))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        lifted = self.lift(values)
        outputs = []
        for graph, self_layers, graph_layers in zip(self.proximities, self.self_layers, self.graph_layers):
            hidden = lifted
            for own, neighbor in zip(self_layers, graph_layers):
                hidden = F.gelu(own(hidden)+neighbor(torch.einsum("rs,nsc->nrc", graph, hidden)))
            outputs.append(hidden)
        return torch.stack(outputs).mean(0)


class NativeTauTransport(nn.Module):
    """Whole-trajectory native prediction with the paper's FO*QO -> DGO -> P."""
    prediction_kind = "conditional_mean_only"

    def __init__(self, settings: dict, inputs, config: dict):
        super().__init__()
        if (settings["operator_layers"] != [2, 2, 2] or settings["fourier_modes"] != 16
                or settings["activation"] != "gelu"
                or settings["spatial_axis"] != "declared_region_index"
                or settings["loss"] != "relative_l2_train_normalized_native_fields"):
            raise ValueError("Tau adaptation needs its declared paper operator and native coordinate contracts")
        width = settings["hidden_dim"]
        if type(width) is not int or width < 1:
            raise ValueError("Tau hidden width must be a positive integer")
        self.field_shape = tuple(inputs.initial.shape[1:])
        regions, compartments, quantities = self.field_shape
        self.maximum_times = config["native_controlled_comparators"]["conditioning"]["maximum_times"]
        self.initial_stop = inputs.blocks["initial_field"][1]
        self.condition_width = inputs.condition.shape[1]
        self.region_ids = list(inputs.grid["region_id"])
        self.species_ids = list(inputs.grid["species_id"])
        rows = {str(row["region_id"]): row for row in build_spatial_region_rows(config)}
        if set(rows) != set(self.region_ids):
            raise ValueError("Tau regional graph differs from the native field grid")
        graph = torch.as_tensor(region_adjacency_matrix(
            [rows[name] for name in self.region_ids], build_spatial_region_edge_rows(config)), dtype=torch.float32)
        channels = compartments*quantities
        spacing = 1/(regions-1)
        self.function_operator = TauFunctionOperator(
            channels+self.condition_width-self.initial_stop, width, regions, 16, spacing)
        self.query_operator = TauFunctionOperator(channels, width, regions, 16, spacing)
        self.graph_operator = TauDirectedGraphOperator(width, graph)
        self.trajectory_projection = nn.Sequential(
            nn.Linear(width, width), nn.Linear(width, self.maximum_times*channels))
        self.method_mapping = {"reference":"arxiv:2603.08108v2",
            "source_layers":[2,2,2], "hidden_dim":width, "requested_fourier_modes":16,
            "actual_fourier_modes":min(16,regions//2+1),
            "native_axis":"declared_regional_index_with_compartment_quantity_channels",
            "graph":"existing_row_normalized_native_region_transport_graph",
            "difference_boundary":"circular_index_stencil",
            "loss":"relative_l2_train_normalized_native_fields",
            "scientific_qualification":False}

    def normalized_prediction(self, inputs) -> torch.Tensor:
        if (tuple(inputs.initial.shape[1:]) != self.field_shape
                or inputs.condition.shape[1] != self.condition_width
                or inputs.blocks["initial_field"][1] != self.initial_stop
                or inputs.grid["region_id"] != self.region_ids
                or inputs.grid["species_id"] != self.species_ids):
            raise ValueError("Tau input field/condition/native-grid identity changed")
        count = inputs.validity.shape[1]
        if count > self.maximum_times:
            raise ValueError("Tau trajectory projection does not support this longer clock")
        initial = inputs.initial.flatten(2)
        parameters = inputs.condition[:, self.initial_stop:, None].transpose(1, 2).expand(-1, self.field_shape[0], -1)
        function = self.function_operator(torch.cat((initial, parameters), -1))
        query = self.query_operator(initial)
        encoded = self.graph_operator(function*query)
        n, regions, _ = encoded.shape
        predicted = self.trajectory_projection(encoded).reshape(
            n, regions, self.maximum_times, *self.field_shape[1:]).transpose(1, 2)
        return predicted[:, :count]

    def predict_native(self, inputs, normalization):
        if inputs.normalization_id != normalization.identity:
            raise ValueError("Tau field normalization differs from its conditioning")
        normalization.assert_domain(inputs.grid)
        return normalization.inverse(self.normalized_prediction(inputs), inputs.validity)

    def loss_native(self, inputs, target, valid, weights, generator):
        prediction = self.normalized_prediction(inputs)
        if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
            raise ValueError("Tau loss needs aligned native predictions/targets/Boolean support")
        norm = torch.where(valid, target, 0).flatten(1).norm(dim=-1)
        if (norm <= 0).any() or not valid.flatten(1).any(-1).all():
            raise ValueError("Relative L2 is undefined for a zero supported target")
        if weights.shape != norm.shape or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Tau loss needs finite nonnegative complete-tuple/action weights")
        if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
            raise FloatingPointError("Nonfinite supported Tau forecast or target")
        error = torch.where(valid, prediction-target, 0).flatten(1).norm(dim=-1)
        return ((error/norm)*weights).sum()/weights.sum()
