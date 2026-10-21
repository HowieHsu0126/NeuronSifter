from __future__ import annotations

import json
import math

import torch
from torch import nn

from Libs.models.modules.moe import SemanticMechanismMixtureOfExperts


class GraphSpectralTemporalWaveletTokenizer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        graph_mode_count: int,
        wavelet_level_count: int,
        eigenvalue_temperature: float,
        normalize_laplacian: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.graph_mode_count = int(graph_mode_count)
        self.wavelet_level_count = int(wavelet_level_count)
        self.eigenvalue_temperature = float(eigenvalue_temperature)
        self.normalize_laplacian = bool(normalize_laplacian)
        self.scale_embedding = nn.Embedding(max(self.wavelet_level_count, 1), self.hidden_dim)
        self.graph_mode_projection = nn.Linear(1, self.hidden_dim)

    def graph_basis(self, adjacency_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = adjacency_matrix.to(dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or not torch.isfinite(adjacency).all():
            raise ValueError("Feature adjacency must be a finite square matrix.")
        adjacency = (adjacency + adjacency.transpose(-1, -2)) * 0.5
        if self.normalize_laplacian:
            degree_values = adjacency.sum(dim=-1).clamp_min(torch.finfo(adjacency.dtype).eps)
            normalizer = torch.rsqrt(degree_values)
            normalized_adjacency = normalizer.unsqueeze(-1) * adjacency * normalizer.unsqueeze(-2)
            laplacian = torch.eye(adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device) - normalized_adjacency
        else:
            degree = torch.diag(adjacency.sum(dim=-1))
            laplacian = degree - adjacency
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        mode_count = min(self.graph_mode_count, eigenvectors.shape[-1])
        return eigenvectors[:, :mode_count], eigenvalues[:mode_count]

    def graph_spectral_transform(self, field_state: torch.Tensor, adjacency_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis, eigenvalues = self.graph_basis(adjacency_matrix)
        basis = basis.to(device=field_state.device, dtype=field_state.dtype)
        eigenvalues = eigenvalues.to(device=field_state.device, dtype=field_state.dtype)
        coefficients = torch.einsum("nm,...ns->...ms", basis, field_state)
        return coefficients, basis, eigenvalues

    def graph_spectral_inverse(self, coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,...ms->...ns", basis.to(device=coefficients.device, dtype=coefficients.dtype), coefficients)

    def graph_low_pass_context(self, field_state: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        coefficients, basis, eigenvalues = self.graph_spectral_transform(field_state, adjacency_matrix)
        weights = torch.exp(-self.eigenvalue_temperature * eigenvalues)
        weight_shape = [1 for _ in range(coefficients.ndim - 2)] + [weights.shape[0], 1]
        filtered_coefficients = coefficients * weights.reshape(weight_shape)
        return self.graph_spectral_inverse(filtered_coefficients, basis)

    def temporal_wavelet_analysis(self, sequence: torch.Tensor) -> dict[str, object]:
        approximation = sequence
        details: list[torch.Tensor] = []
        lengths: list[int] = []
        normalizer = math.sqrt(2.0)
        for _ in range(self.wavelet_level_count):
            current_length = int(approximation.shape[-2])
            if current_length < 2:
                break
            lengths.append(current_length)
            if current_length % 2 == 1:
                approximation = torch.cat([approximation, approximation[..., -1:, :]], dim=-2)
            even_state = approximation[..., 0::2, :]
            odd_state = approximation[..., 1::2, :]
            details.append((even_state - odd_state) / normalizer)
            approximation = (even_state + odd_state) / normalizer
        return {"approximation": approximation, "details": details, "lengths": lengths}

    def temporal_wavelet_synthesis(self, payload: dict[str, object]) -> torch.Tensor:
        approximation = payload["approximation"]
        if not isinstance(approximation, torch.Tensor):
            raise TypeError("temporal_wavelet_synthesis expects tensor approximation.")
        details = list(payload["details"])
        lengths = [int(value) for value in payload["lengths"]]
        normalizer = math.sqrt(2.0)
        for detail_state, original_length in zip(reversed(details), reversed(lengths), strict=True):
            even_state = (approximation + detail_state) / normalizer
            odd_state = (approximation - detail_state) / normalizer
            reconstructed = torch.stack([even_state, odd_state], dim=-2).flatten(-3, -2)
            approximation = reconstructed[..., :original_length, :]
        return approximation

    def forward(self, trajectory_state: torch.Tensor, adjacency_matrix: torch.Tensor) -> dict[str, object]:
        spectral_state, basis, eigenvalues = self.graph_spectral_transform(trajectory_state, adjacency_matrix)
        wavelet_payload = self.temporal_wavelet_analysis(spectral_state)
        return {"spectral_basis": basis, "spectral_eigenvalues": eigenvalues, "wavelet_payload": wavelet_payload}


class AdaLayerNormZero(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim * 3)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)

    def forward(self, hidden_state: torch.Tensor, conditioning_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.condition_projection(conditioning_state).chunk(3, dim=-1)
        conditioned_state = self.norm(hidden_state) * (1.0 + scale) + shift
        return conditioned_state, torch.tanh(gate)


class MechanismStateTokenMixer(nn.Module):
    def __init__(self, hidden_dim: int, token_count: int, slice_temperature: float) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.slice_temperature = float(slice_temperature)
        self.assignment_projection = nn.Linear(hidden_dim, self.token_count)
        self.token_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, field_tokens: torch.Tensor, *, casewise_diagnostics: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assignment_logits = self.assignment_projection(field_tokens) / max(self.slice_temperature, torch.finfo(field_tokens.dtype).eps)
        assignment_weights = torch.softmax(assignment_logits, dim=-1)
        token_mass = assignment_weights.sum(dim=1).clamp_min(torch.finfo(field_tokens.dtype).tiny)
        mechanism_tokens = torch.einsum("bth,btq->bqh", field_tokens, assignment_weights)
        mechanism_tokens = self.token_projection(mechanism_tokens / token_mass.unsqueeze(-1))
        mechanism_entropy = -torch.sum(
            assignment_weights * torch.log(assignment_weights.clamp_min(torch.finfo(assignment_weights.dtype).tiny)),
            dim=-1,
        )
        mechanism_entropy = mechanism_entropy.mean(dim=1) if casewise_diagnostics else mechanism_entropy.mean()
        return mechanism_tokens, {"mechanism_token_entropy": mechanism_entropy, "slice_entropy": mechanism_entropy}


class MechanismOperatorExpert(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float, bottleneck_divisor: int) -> None:
        super().__init__()
        bottleneck_dim = max(int(hidden_dim) // int(bottleneck_divisor), 1)
        self.field_operator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.condition_operator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.coupling_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, field_tokens: torch.Tensor, conditioning_tokens: torch.Tensor) -> torch.Tensor:
        field_residual = self.field_operator(field_tokens)
        condition_residual = self.condition_operator(conditioning_tokens)
        coupling_gate = self.coupling_gate(field_tokens + conditioning_tokens)
        return coupling_gate * field_residual + (1.0 - coupling_gate) * condition_residual


class MechanismOperatorExpertBank(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float,
        expert_names: list[str],
        composition_order: list[str],
        coupling_masks: dict[str, list[str]],
        expert_residual_scale: float,
        shared_coupling_residual_scale: float,
        bottleneck_divisor: int,
        composition_mode: str = "serial",
    ) -> None:
        super().__init__()
        if composition_mode not in {"serial", "parallel"}:
            raise ValueError(f"Unknown operator_splitting.composition_mode: {composition_mode}")
        if (
            not expert_names or len(set(expert_names)) != len(expert_names)
            or len(composition_order) != len(expert_names) or set(composition_order) != set(expert_names)
        ):
            raise ValueError("Expert composition requires the complete unique configured expert roster once.")
        self.composition_mode = composition_mode
        self.register_buffer("composition_mode_code", torch.tensor(composition_mode == "parallel", dtype=torch.uint8))
        if not composition_order:
            raise ValueError("NeuronSifter operator_splitting.composition_order must not be empty.")
        unknown_experts = sorted(set(composition_order) - set(expert_names))
        if unknown_experts:
            raise ValueError(f"operator_splitting.composition_order references unknown experts: {unknown_experts}")
        missing_masks = sorted(set(composition_order) - set(coupling_masks))
        if missing_masks:
            raise ValueError(f"operator_splitting.coupling_masks missing experts: {missing_masks}")
        empty_masks = sorted(expert_name for expert_name in composition_order if not coupling_masks[expert_name])
        if empty_masks:
            raise ValueError(f"operator_splitting.coupling_masks must not be empty: {empty_masks}")
        self.expert_names = list(expert_names)
        self.composition_order = list(composition_order)
        self.register_buffer("composition_identity", torch.tensor(list(json.dumps(
            self.execution_identity(), sort_keys=True, separators=(",", ":")).encode()), dtype=torch.uint8))
        self.expert_residual_scale = float(expert_residual_scale)
        self.shared_coupling_residual_scale = float(shared_coupling_residual_scale)
        self.expert_index_lookup = {expert_name: expert_index for expert_index, expert_name in enumerate(self.expert_names)}
        self.router = nn.Linear(hidden_dim * 2, len(self.expert_names))
        self.experts = nn.ModuleDict(
            {
                expert_name: MechanismOperatorExpert(hidden_dim, dropout_rate, bottleneck_divisor)
                for expert_name in self.expert_names
            }
        )
        bottleneck_dim = max(int(hidden_dim) // int(bottleneck_divisor), 1)
        self.shared_coupling_operator = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        mask_cardinality = torch.tensor(
            [float(len(coupling_masks.get(expert_name, []))) for expert_name in self.expert_names],
            dtype=torch.float32,
        )
        self.register_buffer("coupling_mask_cardinality", mask_cardinality)
        state_names = sorted({state_name for state_list in coupling_masks.values() for state_name in state_list})
        self.state_name_count = len(state_names)
        state_index_lookup = {state_name: state_index for state_index, state_name in enumerate(state_names)}
        coupling_mask = torch.zeros((len(self.expert_names), len(state_names)), dtype=torch.float32)
        for expert_name in self.expert_names:
            expert_index = self.expert_index_lookup[expert_name]
            for state_name in coupling_masks.get(expert_name, []):
                coupling_mask[expert_index, state_index_lookup[state_name]] = 1.0
        self.register_buffer("coupling_mask", coupling_mask)

    def execution_identity(self) -> dict:
        return {"expert_names": list(self.expert_names), "composition_mode": self.composition_mode,
                "composition_order": list(self.expert_names if self.composition_mode == "parallel" else self.composition_order)}

    def validate_checkpoint_composition(self, state_dict, prefix="") -> None:
        mode = state_dict.get(prefix + "composition_mode_code")
        if mode is None:
            if self.composition_mode != "serial":
                raise ValueError("Parallel expert checkpoint requires explicit composition_mode identity.")
        elif not torch.is_tensor(mode) or not torch.equal(mode.detach().cpu(), self.composition_mode_code.detach().cpu()):
            raise ValueError("Expert checkpoint composition_mode differs from the selected mode.")
        identity = state_dict.get(prefix + "composition_identity")
        if identity is None:
            if self.composition_mode == "serial" and self.composition_order != self.expert_names:
                raise ValueError("Noncanonical expert order requires explicit checkpoint composition identity.")
        elif (not torch.is_tensor(identity)
              or not torch.equal(identity.detach().cpu(), self.composition_identity.detach().cpu())):
            raise ValueError("Expert checkpoint composition identity mismatch: order or canonical names differ.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.validate_checkpoint_composition(state_dict, prefix)
        state_dict = dict(state_dict)
        for name in ("composition_mode_code", "composition_identity"):
            state_dict.setdefault(prefix + name, getattr(self, name).detach().clone())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, field_tokens: torch.Tensor, conditioning_tokens: torch.Tensor, *, casewise_diagnostics: bool = False) -> tuple[torch.Tensor, dict[str, object]]:
        router_input = torch.cat([field_tokens, conditioning_tokens], dim=-1)
        mask_cardinality = self.coupling_mask_cardinality.to(device=field_tokens.device, dtype=field_tokens.dtype)
        mask_prior = torch.log(mask_cardinality.clamp_min(torch.finfo(field_tokens.dtype).eps) / mask_cardinality.mean().clamp_min(torch.finfo(field_tokens.dtype).eps))
        router_logits = self.router(router_input) + mask_prior.reshape(1, 1, -1)
        router_weights = torch.softmax(router_logits, dim=-1)
        split_state = field_tokens
        cardinality_scale = mask_cardinality / mask_cardinality.max().clamp_min(torch.finfo(field_tokens.dtype).eps)
        residual_norms: dict[str, object] = {}
        residual_uncertainties: dict[str, object] = {}
        case_activations: dict[str, object] = {}
        parallel_residual = torch.zeros_like(field_tokens) if self.composition_mode == "parallel" else None
        execution_order = self.expert_names if self.composition_mode == "parallel" else self.composition_order
        for expert_name in execution_order:
            expert_index = self.expert_index_lookup[expert_name]
            expert_input = field_tokens if self.composition_mode == "parallel" else split_state
            expert_residual = self.experts[expert_name](expert_input, conditioning_tokens)
            expert_weight = router_weights[..., expert_index].unsqueeze(-1)
            weighted_residual = self.expert_residual_scale * cardinality_scale[expert_index] * expert_weight * expert_residual
            if self.composition_mode == "parallel":
                parallel_residual = parallel_residual + weighted_residual
            else:
                split_state = split_state + weighted_residual
            residual_norms[expert_name] = (weighted_residual.norm(dim=-1).mean(dim=1)
                                           if casewise_diagnostics else weighted_residual.norm(dim=-1).mean())
            residual_uncertainties[expert_name] = (weighted_residual.flatten(1).std(dim=1, unbiased=False)
                                                   if casewise_diagnostics else weighted_residual.std(unbiased=False))
            case_activations[expert_name] = router_weights[..., expert_index].mean(dim=1)
        if self.composition_mode == "parallel":
            split_state = field_tokens + parallel_residual
        shared_residual = self.shared_coupling_operator(split_state + conditioning_tokens)
        split_residual = split_state - field_tokens + self.shared_coupling_residual_scale * shared_residual
        utilization = (router_weights.mean(dim=1) if casewise_diagnostics
                       else router_weights.reshape(-1, router_weights.shape[-1]).mean(dim=0))
        router_entropy = -torch.sum(
            router_weights * torch.log(router_weights.clamp_min(torch.finfo(router_weights.dtype).tiny)),
            dim=-1,
        )
        router_entropy = router_entropy.mean(dim=1) if casewise_diagnostics else router_entropy.mean()
        coupling_mask = self.coupling_mask.to(device=field_tokens.device, dtype=field_tokens.dtype)
        species_attribution = coupling_mask / coupling_mask.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(field_tokens.dtype).eps)
        coupling_matrix = species_attribution @ species_attribution.transpose(0, 1)
        diagnostics = {
            "operator_splitting_identity": self.execution_identity(),
            "operator_splitting_expert_utilization": utilization,
            "operator_splitting_router_entropy": router_entropy,
            "operator_splitting_composition_count": torch.tensor(float(len(self.composition_order)), dtype=field_tokens.dtype, device=field_tokens.device),
            "operator_splitting_parallel_update": field_tokens.new_tensor(self.composition_mode == "parallel"),
            "operator_splitting_update_depth": field_tokens.new_tensor(1 if self.composition_mode == "parallel" else len(self.composition_order)),
            "operator_splitting_coupling_state_count": mask_cardinality.sum(),
            "operator_splitting_residual_norm": (split_residual.norm(dim=-1).mean(dim=1)
                                                   if casewise_diagnostics else split_residual.norm(dim=-1).mean()),
            "operator_splitting_species_attribution": species_attribution,
            "operator_splitting_coupling_matrix": coupling_matrix,
            "operator_splitting_score_norm_by_expert": torch.stack([residual_norms[name] for name in self.expert_names], dim=-1),
            "operator_splitting_uncertainty_by_expert": torch.stack([residual_uncertainties[name] for name in self.expert_names], dim=-1),
            "operator_splitting_case_activation": torch.stack([case_activations[name] for name in self.expert_names], dim=-1),
        }
        if casewise_diagnostics:
            # Indexing one batch record recovers the legacy singleton payload.
            for key in ("operator_splitting_composition_count", "operator_splitting_parallel_update",
                        "operator_splitting_update_depth", "operator_splitting_coupling_state_count",
                        "operator_splitting_species_attribution", "operator_splitting_coupling_matrix"):
                diagnostics[key] = diagnostics[key].unsqueeze(0).expand(field_tokens.shape[0], *diagnostics[key].shape)
            diagnostics["operator_splitting_case_activation"] = diagnostics["operator_splitting_case_activation"].unsqueeze(1)
        return split_residual, diagnostics


class NeuronSifterWorldActionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_count: int,
        dropout_rate: float,
        mlp_multiplier: int,
        expert_names: list[str],
        top_k: int,
        capacity_factor: float,
        entropy_loss_weight: float,
        load_balance_loss_weight: float,
        expert_capacity_loss_weight: float,
        expert_hidden_multiplier: int,
        operator_splitting: dict[str, object],
    ) -> None:
        super().__init__()
        if hidden_dim % head_count != 0:
            raise ValueError(f"NeuronSifter hidden_dim {hidden_dim} must be divisible by head_count {head_count}.")
        self.self_adaln = AdaLayerNormZero(hidden_dim)
        self.cross_adaln = AdaLayerNormZero(hidden_dim)
        self.feed_forward_adaln = AdaLayerNormZero(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, head_count, dropout=dropout_rate, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, head_count, dropout=dropout_rate, batch_first=True)
        self.mechanism_score_moe = SemanticMechanismMixtureOfExperts(
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
        if not bool(operator_splitting["enabled"]):
            raise ValueError("NeuronSifter operator_splitting.enabled must be true for the mainline operator.")
        self.operator_splitting_expert_bank = MechanismOperatorExpertBank(
            hidden_dim,
            dropout_rate,
            expert_names,
            [str(expert_name) for expert_name in operator_splitting["composition_order"]],
            {
                str(expert_name): [str(state_name) for state_name in state_names]
                for expert_name, state_names in dict(operator_splitting["coupling_masks"]).items()
            },
            float(operator_splitting["expert_residual_scale"]),
            float(operator_splitting["shared_coupling_residual_scale"]),
            int(operator_splitting["bottleneck_divisor"]),
            composition_mode=str(operator_splitting.get("composition_mode", "serial")),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * int(mlp_multiplier)),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * int(mlp_multiplier), hidden_dim),
        )

    def forward(
        self,
        field_tokens: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        cross_tokens: torch.Tensor,
        *, casewise_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        self_attention_input, self_gate = self.self_adaln(field_tokens, conditioning_tokens)
        self_attention_output, _ = self.self_attention(self_attention_input, self_attention_input, self_attention_input, need_weights=False)
        field_tokens = field_tokens + self_gate * self_attention_output
        cross_attention_input, cross_gate = self.cross_adaln(field_tokens, conditioning_tokens)
        cross_attention_output, _ = self.cross_attention(cross_attention_input, cross_tokens, cross_tokens, need_weights=False)
        field_tokens = field_tokens + cross_gate * cross_attention_output
        routed_tokens, router_diagnostics = self.mechanism_score_moe(field_tokens, casewise_diagnostics=casewise_diagnostics)
        field_tokens = field_tokens + routed_tokens
        split_residual, split_diagnostics = self.operator_splitting_expert_bank(field_tokens, conditioning_tokens, casewise_diagnostics=casewise_diagnostics)
        field_tokens = field_tokens + split_residual
        router_diagnostics.update(split_diagnostics)
        feed_forward_input, feed_forward_gate = self.feed_forward_adaln(field_tokens, conditioning_tokens)
        field_tokens = field_tokens + feed_forward_gate * self.feed_forward(feed_forward_input)
        return field_tokens, router_diagnostics


class NeuronSifterWorldActionOperator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        head_count: int,
        mechanism_token_count: int,
        coordinate_dim: int,
        dropout_rate: float,
        expert_names: list[str],
        top_k: int,
        capacity_factor: float,
        entropy_loss_weight: float,
        load_balance_loss_weight: float,
        expert_capacity_loss_weight: float,
        expert_hidden_multiplier: int,
        mlp_multiplier: int,
        graph_mode_count: int,
        wavelet_level_count: int,
        eigenvalue_temperature: float,
        normalize_laplacian: bool,
        spectral_residual_scale: float,
        mechanism_slice_temperature: float,
        diffusion_sample_count: int,
        denoising_step_count: int,
        sample_chunk_size: int,
        training_sample_count: int,
        training_denoising_step_count: int,
        diffusion_noise_scale: float,
        score_step_size: float,
        terminal_noise_scale: float,
        ranking_stability_scale: float,
        quantile_levels: list[float],
        sampler_seed: int,
        return_sample_trajectories: bool,
        operator_splitting: dict[str, object],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.spectral_residual_scale = float(spectral_residual_scale)
        self.diffusion_sample_count = int(diffusion_sample_count)
        self.denoising_step_count = int(denoising_step_count)
        self.sample_chunk_size = int(sample_chunk_size)
        self.training_sample_count = int(training_sample_count)
        self.training_denoising_step_count = int(training_denoising_step_count)
        self.diffusion_noise_scale = float(diffusion_noise_scale)
        self.score_step_size = float(score_step_size)
        self.terminal_noise_scale = float(terminal_noise_scale)
        self.ranking_stability_scale = float(ranking_stability_scale)
        self.quantile_levels = [float(value) for value in quantile_levels]
        self.sampler_seed = int(sampler_seed)
        self.return_sample_trajectories = bool(return_sample_trajectories)
        self.coordinate_projection = nn.Linear(coordinate_dim, hidden_dim)
        self.regimen_axis_projection = nn.Linear(1, hidden_dim)
        self.compartment_axis_projection = nn.Linear(1, hidden_dim)
        self.occupancy_projection = nn.Linear(hidden_dim, hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim)
        self.tokenizer = GraphSpectralTemporalWaveletTokenizer(
            hidden_dim,
            graph_mode_count,
            wavelet_level_count,
            eigenvalue_temperature,
            normalize_laplacian,
        )
        self.mechanism_token_mixer = MechanismStateTokenMixer(hidden_dim, mechanism_token_count, mechanism_slice_temperature)
        self.blocks = nn.ModuleList(
            [
                NeuronSifterWorldActionBlock(
                    hidden_dim,
                    head_count,
                    dropout_rate,
                    mlp_multiplier,
                    expert_names,
                    top_k,
                    capacity_factor,
                    entropy_loss_weight,
                    load_balance_loss_weight,
                    expert_capacity_loss_weight,
                    expert_hidden_multiplier,
                    operator_splitting,
                )
                for _ in range(self.depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def geometry_embedding(self, node_coordinates: torch.Tensor, sequence_length: int) -> torch.Tensor:
        regimen_axis = torch.linspace(
            0.0,
            1.0,
            sequence_length,
            dtype=node_coordinates.dtype,
            device=node_coordinates.device,
        ).view(1, sequence_length, 1, 1)
        regimen_axis = regimen_axis.expand(node_coordinates.shape[0], -1, node_coordinates.shape[2], -1)
        compartment_axis = node_coordinates[..., 1:2]
        return (
            self.coordinate_projection(node_coordinates)
            + self.regimen_axis_projection(regimen_axis)
            + self.compartment_axis_projection(compartment_axis)
        )

    def forward(
        self,
        node_state: torch.Tensor,
        conditioning_state: torch.Tensor,
        node_coordinates: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        *, casewise_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        hidden_state = node_state + conditioning_state + self.geometry_embedding(node_coordinates, node_state.shape[1])
        batch_size, sequence_length, node_count, hidden_dim = hidden_state.shape
        spectral_context = self.tokenizer.graph_low_pass_context(
            hidden_state.reshape(batch_size * sequence_length, node_count, hidden_dim),
            adjacency_matrix,
        ).reshape(batch_size, sequence_length, node_count, hidden_dim)
        hidden_state = hidden_state + self.spectral_residual_scale * spectral_context
        field_tokens = hidden_state.reshape(batch_size, sequence_length * node_count, hidden_dim)
        conditioning_tokens = self.condition_projection(
            conditioning_state.reshape(batch_size, sequence_length * node_count, hidden_dim)
        )
        occupancy_tokens = self.occupancy_projection(conditioning_tokens)
        mechanism_tokens, diagnostics = self.mechanism_token_mixer(field_tokens, casewise_diagnostics=casewise_diagnostics)
        cross_tokens = torch.cat([occupancy_tokens, mechanism_tokens], dim=1)
        router_payloads: list[dict[str, object]] = []
        for block in self.blocks:
            field_tokens, router_payload = block(field_tokens, conditioning_tokens, cross_tokens, casewise_diagnostics=casewise_diagnostics)
            router_payloads.append(router_payload)
        field_tokens = self.output_norm(field_tokens)
        output_state = field_tokens.reshape(batch_size, sequence_length, node_count, hidden_dim)
        if router_payloads:
            for key in router_payloads[0]:
                if key == "operator_splitting_identity":
                    if any(payload[key] != router_payloads[0][key] for payload in router_payloads):
                        raise ValueError("Conflicting executed expert identities across operator blocks.")
                    diagnostics[key] = router_payloads[0][key]
                else:
                    diagnostics[key] = torch.stack([payload[key] for payload in router_payloads], dim=0).mean(dim=0)
        diagnostics["score_block_count"] = torch.tensor(float(self.depth), device=output_state.device, dtype=output_state.dtype)
        if casewise_diagnostics:
            diagnostics["score_block_count"] = diagnostics["score_block_count"].expand(batch_size)
        return output_state, diagnostics

    def forward_preembedded(
        self,
        node_state: torch.Tensor,
        conditioning_state: torch.Tensor,
        node_coordinates: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Run the deterministic project-native operator before stochastic trajectory sampling."""
        return self.forward(node_state, conditioning_state, node_coordinates, adjacency_matrix)

    def sampler_generator(self, device: torch.device) -> torch.Generator | None:
        if self.sampler_seed < 0:
            return None
        generator = torch.Generator(device=device.type)
        generator.manual_seed(self.sampler_seed)
        return generator

    def trajectory_tokenizer_diagnostics(self, prior_trajectory: torch.Tensor, adjacency_matrix: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, regimen_count, time_count, region_count, compartment_count, species_count = prior_trajectory.shape
        node_count = region_count * compartment_count
        graph_state = prior_trajectory.reshape(batch_size * regimen_count * time_count, node_count, species_count)
        coefficients, basis, _ = self.tokenizer.graph_spectral_transform(graph_state, adjacency_matrix)
        graph_reconstruction = self.tokenizer.graph_spectral_inverse(coefficients, basis)
        temporal_sequence = prior_trajectory.reshape(batch_size * regimen_count, time_count, node_count, species_count)
        temporal_sequence = temporal_sequence.permute(0, 2, 1, 3).reshape(batch_size * regimen_count * node_count, time_count, species_count)
        wavelet_payload = self.tokenizer.temporal_wavelet_analysis(temporal_sequence)
        temporal_reconstruction = self.tokenizer.temporal_wavelet_synthesis(wavelet_payload)
        return {
            "graph_projection_error": torch.mean((graph_reconstruction - graph_state) ** 2),
            "temporal_wavelet_reconstruction_error": torch.mean((temporal_reconstruction - temporal_sequence) ** 2),
            "graph_mode_count": torch.tensor(float(coefficients.shape[-2]), dtype=prior_trajectory.dtype, device=prior_trajectory.device),
            "wavelet_level_count": torch.tensor(float(len(wavelet_payload["details"])), dtype=prior_trajectory.dtype, device=prior_trajectory.device),
        }

    def trajectory_diffusion_summary(
        self,
        prior_trajectory: torch.Tensor,
        training: bool,
        adjacency_matrix: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        sample_count = max(self.training_sample_count if training else self.diffusion_sample_count, 1)
        denoising_steps = max(self.training_denoising_step_count if training else self.denoising_step_count, 1)
        chunk_size = max(min(self.sample_chunk_size, sample_count), 1)
        generator = self.sampler_generator(prior_trajectory.device)
        sample_chunks: list[torch.Tensor] = []
        remaining_sample_count = sample_count
        while remaining_sample_count > 0:
            current_chunk_size = min(chunk_size, remaining_sample_count)
            residual_state = torch.randn(
                (current_chunk_size, *prior_trajectory.shape),
                dtype=prior_trajectory.dtype,
                device=prior_trajectory.device,
                generator=generator,
            )
            for denoising_index in range(denoising_steps):
                denoising_fraction = float(denoising_index + 1) / float(denoising_steps)
                annealed_noise = self.terminal_noise_scale + (self.diffusion_noise_scale - self.terminal_noise_scale) * (1.0 - denoising_fraction)
                residual_state = residual_state * (1.0 - self.score_step_size) + annealed_noise * torch.randn(
                    residual_state.shape,
                    dtype=residual_state.dtype,
                    device=residual_state.device,
                    generator=generator,
                )
            sample_chunks.append(prior_trajectory.unsqueeze(0) + self.diffusion_noise_scale * residual_state)
            remaining_sample_count -= current_chunk_size
        sample_trajectories = torch.cat(sample_chunks, dim=0)
        trajectory_mean = sample_trajectories.mean(dim=0)
        trajectory_std = sample_trajectories.std(dim=0, unbiased=False)
        quantile_tensor = torch.quantile(
            sample_trajectories,
            torch.tensor(self.quantile_levels, dtype=sample_trajectories.dtype, device=sample_trajectories.device),
            dim=0,
        )
        final_uncertainty = trajectory_std[:, :, -1].mean(dim=(-1, -2, -3))
        ranking_stability = torch.exp(-self.ranking_stability_scale * final_uncertainty)
        diagnostics = {
            "stochastic_sample_count": torch.tensor(float(sample_count), dtype=prior_trajectory.dtype, device=prior_trajectory.device),
            "denoising_step_count": torch.tensor(float(denoising_steps), dtype=prior_trajectory.dtype, device=prior_trajectory.device),
            "sample_chunk_size": torch.tensor(float(chunk_size), dtype=prior_trajectory.dtype, device=prior_trajectory.device),
            "trajectory_uncertainty_mean": trajectory_std.mean(),
            "ranking_stability_mean": ranking_stability.mean(),
            **self.trajectory_tokenizer_diagnostics(prior_trajectory, adjacency_matrix),
        }
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "trajectory_mean": trajectory_mean,
            "trajectory_std": trajectory_std,
            "trajectory_quantiles": quantile_tensor,
            "ranking_stability": ranking_stability,
            "sampler_diagnostics": diagnostics,
        }
        if self.return_sample_trajectories:
            output["sample_trajectories"] = sample_trajectories
        return output


def native_world_action_from_config(backbone:dict,*,hidden_dim:int|None=None,depth:int|None=None,
                                    head_count:int|None=None) -> NeuronSifterWorldActionOperator:
    """One maintained constructor for training and deterministic native ablations."""
    moe=backbone["mechanism_moe"];graph=backbone["graph_spectral_tokenizer"]
    wavelet=backbone["temporal_wavelet_tokenizer"];sampler=backbone["diffusion_sampler"]
    return NeuronSifterWorldActionOperator(
        int(hidden_dim if hidden_dim is not None else backbone["operator_hidden_dim"]),
        int(depth if depth is not None else backbone["operator_depth"]),
        int(head_count if head_count is not None else backbone["attention_head_count"]),
        int(backbone["mechanism_state_token_count"]),int(backbone["node_coordinate_dim"]),
        float(backbone["operator_dropout_rate"]),list(moe["expert_names"]),int(moe["top_k"]),
        float(moe["capacity_factor"]),float(moe["entropy_loss_weight"]),float(moe["load_balance_loss_weight"]),
        float(moe["expert_capacity_loss_weight"]),int(moe["expert_hidden_multiplier"]),
        int(backbone["dit_mlp_multiplier"]),int(graph["graph_mode_count"]),int(wavelet["wavelet_level_count"]),
        float(graph["eigenvalue_temperature"]),bool(graph["normalize_laplacian"]),float(graph["spectral_residual_scale"]),
        float(backbone["mechanism_slice_temperature"]),int(sampler["sample_count"]),int(sampler["denoising_step_count"]),
        int(sampler["sample_chunk_size"]),int(sampler["training_sample_count"]),int(sampler["training_denoising_step_count"]),
        float(sampler["diffusion_noise_scale"]),float(sampler["score_step_size"]),float(sampler["terminal_noise_scale"]),
        float(sampler["ranking_stability_scale"]),list(sampler["quantile_levels"]),int(sampler["sampler_seed"]),
        bool(sampler["return_sample_trajectories"]),dict(backbone["operator_splitting"]))
