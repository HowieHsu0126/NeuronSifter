from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class QuantizedMddmRolloutOperator(nn.Module):
    """Contextual absorbing-mask diffusion on physical-time Haar coefficients."""

    def __init__(
        self,
        config: dict[str, Any],
        species_count: int,
        hidden_dim: int,
        posterior_dim: int,
        wavelet_level_count: int,
    ) -> None:
        super().__init__()
        token = config["residual_tokenization"]
        schedule = config["partial_mask_schedule"]
        reconstruction = config["inverse_reconstruction_policy"]
        if (
            not config["enabled"]
            or config["residual_reference_policy"] != "initial_state"
        ):
            raise ValueError("Conditional diffusion requires initial_state residuals.")
        if list(config["token_axes"]) != ["time", "region", "compartment", "species"]:
            raise ValueError("Invalid residual token axes.")
        if (
            token["quantization_policy"]
            != "straight_through_uniform_scalar_residual_codebook"
        ):
            raise ValueError("Unsupported quantization policy.")
        if (
            reconstruction["reconstruction_policy"]
            != "straight_through_residual_addition"
        ):
            raise ValueError("Unsupported reconstruction policy.")
        if (
            config["external_reference_policy"]["reference_policy"]
            != "project_native_no_direct_import"
        ):
            raise ValueError("The denoiser must use the project-native backbone.")
        obsolete = {
            "training_initial_mask_probability",
            "inference_initial_mask_probability",
            "final_mask_probability",
            "replacement_blend",
            "denoising_token_chunk_size",
        } & schedule.keys()
        if obsolete:
            raise ValueError(
                f"Remove obsolete partial-refiner settings: {sorted(obsolete)}"
            )
        if schedule.get("schedule_kind") != "absorbing_linear":
            raise ValueError(
                "Training and reverse sampling require the absorbing_linear schedule."
            )
        self.species_count, self.hidden_dim = int(species_count), int(hidden_dim)
        self.posterior_dim, self.wavelet_level_count = (
            int(posterior_dim),
            int(wavelet_level_count),
        )
        self.occupancy_channel_count = int(
            config["conditioning"]["occupancy_channel_count"]
        )
        self.occupancy_channel_ids = tuple(
            config["conditioning"]["occupancy_channel_ids"]
        )
        if (
            len(self.occupancy_channel_ids) != self.occupancy_channel_count
            or len(set(self.occupancy_channel_ids)) != self.occupancy_channel_count
        ):
            raise ValueError(
                "Occupancy channels require a unique, fixed target ordering."
            )
        self.codebook_size = int(token["codebook_size"])
        self.residual_min_value, self.residual_max_value = (
            float(token["residual_min_value"]),
            float(token["residual_max_value"]),
        )
        self.mask_token_value = float(token["mask_token_value"])
        self.training_sample_count = int(schedule["training_sample_count"])
        self.inference_sample_count = int(schedule["inference_sample_count"])
        self.sample_chunk_size = int(schedule["sample_chunk_size"])
        self.training_denoising_step_count = int(
            schedule["training_denoising_step_count"]
        )
        self.inference_denoising_step_count = int(
            schedule["inference_denoising_step_count"]
        )
        self.sampler_seed = int(schedule["sampler_seed"])
        if self.sampler_seed < 0:
            raise ValueError(
                "Candidate-independent sampling requires a nonnegative seed."
            )
        self.preserve_initial_state = bool(reconstruction["preserve_initial_state"])
        self.state_lower_bound = float(reconstruction["state_lower_bound"])
        self.ranking_stability_scale = float(reconstruction["ranking_stability_scale"])
        self.quantile_levels = [float(v) for v in reconstruction["quantile_levels"]]
        self.return_sample_trajectories = bool(
            reconstruction["return_sample_trajectories"]
        )
        self.loss_weights = {
            str(k): float(v) for k, v in config["loss_weights"].items()
        }
        if set(self.loss_weights) != {
            "token_substitution_cross_entropy",
            "inverse_residual_reconstruction",
        }:
            raise ValueError(
                "Only masked-token and clean reconstruction losses belong to this reverse process."
            )
        dimensions = [
            species_count,
            hidden_dim,
            posterior_dim,
            self.occupancy_channel_count,
            self.training_sample_count,
            self.inference_sample_count,
            self.sample_chunk_size,
            self.inference_denoising_step_count,
            int(token["token_mlp_multiplier"]),
        ]
        if (
            min(dimensions) < 1
            or wavelet_level_count < 0
            or self.training_denoising_step_count != 1
        ):
            raise ValueError(
                "Invalid dimensions; training requires one continuous-time corruption per draw."
            )
        scalars = [
            self.residual_min_value,
            self.residual_max_value,
            self.mask_token_value,
            self.state_lower_bound,
            self.ranking_stability_scale,
            *self.quantile_levels,
            *self.loss_weights.values(),
        ]
        if not all(math.isfinite(v) for v in scalars):
            raise ValueError("Nonfinite sampler configuration.")
        if self.codebook_size < 4 or self.residual_max_value <= self.residual_min_value:
            raise ValueError("Invalid residual codebook.")
        if not self.quantile_levels or self.quantile_levels != sorted(
            set(self.quantile_levels)
        ):
            raise ValueError("Quantiles must be nonempty and strictly increasing.")
        if min(self.quantile_levels) < 0 or max(self.quantile_levels) > 1:
            raise ValueError("Quantiles must lie in [0,1].")
        if (
            min(self.loss_weights.values()) < 0
            or self.loss_weights["token_substitution_cross_entropy"] <= 0
        ):
            raise ValueError(
                "Positive token objective and nonnegative auxiliary weights required."
            )
        codebook = torch.linspace(
            self.residual_min_value, self.residual_max_value, self.codebook_size
        )
        self.register_buffer("codebook_values", codebook)
        self.register_buffer("codebook_boundaries", (codebook[:-1] + codebook[1:]) / 2)
        self.value_projection = nn.Linear(species_count, hidden_dim)
        self.mask_projection = nn.Linear(species_count, hidden_dim)
        self.initial_state_projection = nn.Linear(species_count, hidden_dim)
        self.occupancy_projection = nn.Linear(self.occupancy_channel_count, hidden_dim)
        self.occupancy_support_projection = nn.Linear(
            self.occupancy_channel_count, hidden_dim
        )
        self.posterior_projection = nn.Linear(posterior_dim, hidden_dim)
        self.physical_time_projection = nn.Linear(3, hidden_dim)
        self.diffusion_time_projection = nn.Linear(3, hidden_dim)
        width = hidden_dim * int(token["token_mlp_multiplier"])
        self.token_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, species_count * self.codebook_size),
        )

    @staticmethod
    def _finite(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values.")

    def get_extra_state(self):
        return {
            "method": "occupancy_conditioned_haar_absorbing_v1",
            "occupancy_channel_ids": self.occupancy_channel_ids,
            "species_count": self.species_count,
            "posterior_dim": self.posterior_dim,
            "wavelet_level_count": self.wavelet_level_count,
            "codebook_size": self.codebook_size,
            "codebook_range": (self.residual_min_value, self.residual_max_value),
            "reference": "initial_state",
            "time_units": "hours",
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint method or target-channel semantics differ from the configured conditional model.")

    def _prepare(
        self,
        initial_state,
        occupancy_program,
        occupancy_support,
        posterior_context,
        time_hours,
        node_coordinates,
        adjacency_matrix,
        backbone,
    ):
        for name, value in (
            ("initial_state", initial_state),
            ("occupancy_program", occupancy_program),
            ("posterior_context", posterior_context),
            ("time_hours", time_hours),
            ("node_coordinates", node_coordinates),
            ("adjacency_matrix", adjacency_matrix),
        ):
            self._finite(name, value)
            if value.device != initial_state.device:
                raise ValueError(f"{name} device mismatch.")
        if initial_state.ndim != 5 or initial_state.shape[-1] != self.species_count:
            raise ValueError("initial_state must be [B,A,R,C,S].")
        b, a, r, c, s = initial_state.shape
        if min(b, a, r, c) < 1 or occupancy_program.ndim != 6:
            raise ValueError("Empty axes or invalid occupancy rank.")
        t, n, ba = occupancy_program.shape[2], r * c, b * a
        if t < 1 or occupancy_program.shape != (
            b,
            a,
            t,
            r,
            c,
            self.occupancy_channel_count,
        ):
            raise ValueError("occupancy_program must be [B,A,T,R,C,Q].")
        if (
            not isinstance(occupancy_support, torch.Tensor)
            or occupancy_support.dtype != torch.bool
        ):
            raise TypeError("Occupancy support must be explicit boolean evidence.")
        if (
            occupancy_support.shape != occupancy_program.shape
            or occupancy_support.device != initial_state.device
        ):
            raise ValueError("Support shape/device mismatch.")
        if not bool(occupancy_support.all()):
            raise ValueError("Required occupancy evidence is incomplete.")
        if bool(((occupancy_program < 0) | (occupancy_program > 1)).any()):
            raise ValueError("Occupancy fractions must lie in [0,1].")
        if posterior_context.shape != (
            b,
            a,
            self.posterior_dim,
        ) or time_hours.shape != (b, a, t):
            raise ValueError("Posterior/time shape mismatch.")
        if bool((time_hours < 0).any()) or (
            t > 1 and not bool((time_hours.diff(dim=-1) > 0).all())
        ):
            raise ValueError(
                "Physical hours must be nonnegative and strictly increasing."
            )
        if bool((initial_state < self.state_lower_bound).any()):
            raise ValueError("Initial state violates lower bound.")
        if adjacency_matrix.shape != (n, n) or bool((adjacency_matrix < 0).any()):
            raise ValueError(
                "Expected square nonnegative node adjacency; the backbone uses its symmetric spectral part."
            )
        codec = getattr(backbone, "tokenizer", None)
        if (
            codec is None
            or codec.wavelet_level_count != self.wavelet_level_count
            or backbone.hidden_dim != self.hidden_dim
        ):
            raise ValueError("Conditional codec/backbone mismatch.")
        occupancy = occupancy_program.reshape(ba, t, n, self.occupancy_channel_count)
        packed, metadata = codec.pack_temporal_wavelets(occupancy)
        support, _ = codec.pack_temporal_wavelets(
            occupancy_support.to(occupancy).reshape_as(occupancy)
        )
        physical = codec.temporal_wavelet_coordinates(time_hours.reshape(ba, t))
        physical = torch.stack(
            [
                torch.log1p(physical[..., 0]),
                torch.log1p(physical[..., 1]),
                physical[..., 2] / max(self.wavelet_level_count, 1),
            ],
            -1,
        )
        dtype = self.value_projection.weight.dtype
        initial = initial_state.reshape(ba, n, s)
        condition = (
            self.occupancy_projection(packed.to(dtype=dtype))
            + self.occupancy_support_projection(support.to(dtype=dtype))
            + self.initial_state_projection(initial.to(dtype=dtype))[:, None]
            + self.posterior_projection(
                posterior_context.reshape(ba, -1).to(dtype=dtype)
            )[:, None, None]
            + self.physical_time_projection(physical.to(dtype=dtype))[:, :, None]
        )
        length = packed.shape[1]
        if node_coordinates.shape == (n, 2):
            coordinates = node_coordinates[None, None].expand(ba, length, n, 2)
        elif node_coordinates.shape == (ba, length, n, 2):
            coordinates = node_coordinates
        else:
            raise ValueError("Node coordinates must be [N,2] or [B*A,L,N,2].")
        return dict(
            shape=(b, a, t, r, c, s),
            initial=initial,
            condition=condition,
            coordinates=coordinates.to(dtype=dtype),
            metadata=metadata,
            zero_time=time_hours.reshape(ba, t).eq(0),
            length=length,
        )

    def quantize_residual(self, residual):
        ids = torch.bucketize(
            residual.clamp(
                self.residual_min_value, self.residual_max_value
            ).contiguous(),
            self.codebook_boundaries.to(residual),
        )
        quantized = self.codebook_values.to(residual)[ids]
        return residual + (quantized - residual).detach(), ids

    def _logits(self, values, mask, time, conditions, coordinates, adjacency, backbone):
        dtype = self.value_projection.weight.dtype
        time = time.to(dtype=dtype)
        features = torch.stack(
            [time, torch.sin(math.pi * time), torch.cos(math.pi * time)], -1
        )
        conditions = (
            conditions + self.diffusion_time_projection(features)[:, None, None]
        )
        inputs = self.value_projection(values.to(dtype=dtype)) + self.mask_projection(
            mask.to(dtype=dtype)
        )
        hidden, diagnostics = backbone(inputs, conditions, coordinates, adjacency)
        logits = self.token_readout(hidden).reshape(*values.shape, self.codebook_size)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite conditional token logits.")
        return logits, diagnostics

    def _decode(self, coefficients, initial, zero_time, metadata, backbone):
        residual = backbone.tokenizer.unpack_temporal_wavelets(coefficients, metadata)
        field = (initial[:, None] + residual).clamp_min(self.state_lower_bound)
        return (
            torch.where(zero_time[:, :, None, None], initial[:, None], field)
            if self.preserve_initial_state
            else field
        )

    @staticmethod
    def _forward_mask(shape, device):
        # Sample Bernoulli(t) conditional on a nonempty mask; its probability
        # mass restores the ordinary masked-CE expectation (empty loss is zero).
        count = math.prod(shape[1:])
        epsilon = torch.finfo(torch.float32).eps
        time = torch.rand(shape[0], device=device).clamp(epsilon, 1 - epsilon)
        log_survival = torch.log1p(-time)
        mass = -torch.expm1(count * log_survival)
        first = (
            torch.floor(torch.log1p(-torch.rand_like(time) * mass) / log_survival)
            .long()
            .clamp_max(count - 1)
        )
        positions = torch.arange(count, device=device)[None]
        independent = torch.rand((shape[0], count), device=device) < time[:, None]
        mask = (positions == first[:, None]) | (
            (positions > first[:, None]) & independent
        )
        return mask.reshape(shape), time, mass

    @staticmethod
    def _merge(payloads):
        return (
            {
                key: torch.stack([item[key] for item in payloads]).mean(0)
                for key in payloads[0]
            }
            if payloads
            else {}
        )

    def training_loss(
        self,
        target_trajectory,
        *,
        occupancy_program,
        occupancy_support,
        initial_state,
        posterior_context,
        time_hours,
        node_coordinates,
        adjacency_matrix,
        backbone,
        sample_weights=None,
    ):
        p = self._prepare(
            initial_state,
            occupancy_program,
            occupancy_support,
            posterior_context,
            time_hours,
            node_coordinates,
            adjacency_matrix,
            backbone,
        )
        self._finite("target_trajectory", target_trajectory)
        if (
            tuple(target_trajectory.shape) != p["shape"]
            or target_trajectory.device != initial_state.device
        ):
            raise ValueError("Clean trajectory shape/device mismatch.")
        if bool((target_trajectory < self.state_lower_bound).any()):
            raise ValueError("Clean target violates state lower bound.")
        b, a, t, r, c, s = p["shape"]
        target = target_trajectory.reshape(b * a, t, r * c, s)
        if bool(p["zero_time"].any()) and not torch.allclose(
            target[p["zero_time"]],
            p["initial"][:, None].expand_as(target)[p["zero_time"]],
        ):
            raise ValueError(
                "Clean target differs from initial condition at zero hours."
            )
        clean, metadata = backbone.tokenizer.pack_temporal_wavelets(
            target - p["initial"][:, None]
        )
        quantized, ids = self.quantize_residual(clean)
        weights = torch.ones(b * a, device=target.device)
        if sample_weights is not None:
            self._finite("sample_weights", sample_weights)
            if sample_weights.device != target.device or tuple(
                sample_weights.shape
            ) not in {(b, a), (b * a,)}:
                raise ValueError("Teacher weights must be [B,A] on the target device.")
            weights = sample_weights.reshape(-1).float()
            if bool((weights < 0).any()) or not bool((weights > 0).any()):
                raise ValueError(
                    "Teacher weights must be nonnegative with positive mass."
                )
        axes, count = tuple(range(1, clean.ndim)), math.prod(clean.shape[1:])
        predictions, ce_losses, rec_losses, payloads = [], [], [], []
        for _ in range(self.training_sample_count):
            mask, time, mass = self._forward_mask(tuple(ids.shape), ids.device)
            corrupted = torch.where(
                mask, torch.full_like(quantized, self.mask_token_value), quantized
            )
            logits, diagnostics = self._logits(
                corrupted,
                mask,
                time,
                p["condition"],
                p["coordinates"],
                adjacency_matrix,
                backbone,
            )
            ce = F.cross_entropy(
                logits.float().reshape(-1, self.codebook_size),
                ids.reshape(-1),
                reduction="none",
            ).reshape_as(ids)
            ce_losses.append((ce * mask).sum(axes) * (mass / time) / count)
            expected = (
                (torch.softmax(logits.float(), -1) * self.codebook_values.float())
                .sum(-1)
                .to(clean)
            )
            prediction = self._decode(
                torch.where(mask, expected, quantized),
                p["initial"],
                p["zero_time"],
                metadata,
                backbone,
            )
            predictions.append(prediction)
            rec_losses.append((prediction.float() - target.float()).square().mean(axes))
            payloads.append(diagnostics)
        ce, rec = [torch.stack(items).mean(0) for items in (ce_losses, rec_losses)]
        per_example = (
            self.loss_weights["token_substitution_cross_entropy"] * ce
            + self.loss_weights["inverse_residual_reconstruction"] * rec
        )
        # Fixed sample-count normalization preserves QC/precision weighting for
        # singleton microbatches and gradient accumulation.
        loss = (per_example * weights).mean()
        diagnostics = self._merge(payloads)
        diagnostics.update(
            mddm_token_substitution_cross_entropy=(ce * weights).mean(),
            mddm_inverse_reconstruction_error=(rec * weights).mean(),
            mddm_rollout_auxiliary_loss=loss,
            mddm_wavelet_packed_time_count=loss.new_tensor(float(p["length"])),
        )
        diagnostics["mddm_coefficient_saturation_fraction"] = (
            ((clean < self.residual_min_value) | (clean > self.residual_max_value))
            .float()
            .mean()
        )
        diagnostics["mddm_quantized_coefficient_error"] = (
            (quantized.detach() - clean.detach()).square().mean()
        )
        return dict(
            trajectory_mean=torch.stack(predictions).mean(0).reshape(p["shape"]),
            trajectory_estimate_status="clean_prediction_from_training_corruption",
            uncertainty_status="unavailable_during_reverse_training",
            ranking_stability_status="unavailable_during_reverse_training",
            rollout_auxiliary_loss=loss,
            rollout_loss_per_example=per_example.reshape(b, a),
            sampler_diagnostics=diagnostics,
        )

    @torch.no_grad()
    def predict_clean_marginals(
        self,
        *,
        initial_state,
        occupancy_program,
        occupancy_support,
        posterior_context,
        time_hours,
        node_coordinates,
        adjacency_matrix,
        backbone,
    ):
        """Clean-token predictions at the all-mask boundary, one call per action.

        These marginals are not assumed calibrated or equal to the marginals of
        a multi-step rollout. Their decision use needs an explicit loss contract.
        """
        if self.training or backbone.training:
            raise ValueError("Marginal prediction requires eval() on sampler and backbone.")
        prepared = self._prepare(
            initial_state, occupancy_program, occupancy_support, posterior_context,
            time_hours, node_coordinates, adjacency_matrix, backbone,
        )
        b, a, _, r, c, species = prepared["shape"]
        shape = (1, prepared["length"], r * c, species)
        predictions = []
        for index in range(b * a):
            values = initial_state.new_full(shape, self.mask_token_value)
            mask = torch.ones(shape, dtype=torch.bool, device=initial_state.device)
            logits, _ = self._logits(
                values, mask, initial_state.new_ones(1),
                prepared["condition"][index:index + 1],
                prepared["coordinates"][index:index + 1],
                adjacency_matrix, backbone,
            )
            predictions.append(torch.softmax(logits.float(), dim=-1))
        probabilities = torch.cat(predictions, dim=0).reshape(
            b, a, prepared["length"], r, c, species, self.codebook_size
        )
        return {
            "probabilities": probabilities,
            "status": "uncalibrated_all_mask_marginals",
            "denoiser_evaluations": b * a,
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        initial_state,
        occupancy_program,
        occupancy_support,
        posterior_context,
        time_hours,
        node_coordinates,
        adjacency_matrix,
        backbone,
        sample_count=None,
        denoising_steps=None,
        sample_keys=None,
    ):
        if self.training or backbone.training:
            raise ValueError(
                "Predictive sampling requires eval() on sampler and backbone."
            )
        p = self._prepare(
            initial_state,
            occupancy_program,
            occupancy_support,
            posterior_context,
            time_hours,
            node_coordinates,
            adjacency_matrix,
            backbone,
        )
        count = (
            self.inference_sample_count if sample_count is None else int(sample_count)
        )
        steps = (
            self.inference_denoising_step_count
            if denoising_steps is None
            else int(denoising_steps)
        )
        if min(count, steps) < 1:
            raise ValueError("Sample count and reverse steps must be positive.")
        b, a, t, r, c, s = p["shape"]
        if (
            sample_keys is None
            or sample_keys.dtype != torch.int64
            or sample_keys.shape != (b, a)
            or bool((sample_keys < 0).any())
        ):
            raise ValueError("Stable nonnegative int64 sample_keys[B,A] are required.")
        keys = sample_keys.detach().cpu().reshape(-1).tolist()
        samples, payloads = [], []
        shape = (1, p["length"], r * c, s)
        codebook = self.codebook_values.to(initial_state)
        for start in range(0, count, self.sample_chunk_size):
            for sample_index in range(
                start, min(start + self.sample_chunk_size, count)
            ):
                actions = []
                for index, key in enumerate(keys):
                    generator = None
                    if self.sampler_seed >= 0:
                        generator = torch.Generator(device=initial_state.device)
                        generator.manual_seed(
                            (
                                self.sampler_seed
                                + 1_000_003 * key
                                + 97_409 * sample_index
                            )
                            % (2**63 - 1)
                        )
                    mask = torch.ones(
                        shape, device=initial_state.device, dtype=torch.bool
                    )
                    values = initial_state.new_full(shape, self.mask_token_value)
                    for step in range(steps):
                        time, next_time = 1 - step / steps, 1 - (step + 1) / steps
                        logits, diagnostics = self._logits(
                            values,
                            mask,
                            initial_state.new_tensor([time]),
                            p["condition"][index : index + 1],
                            p["coordinates"][index : index + 1],
                            adjacency_matrix,
                            backbone,
                        )
                        reveal = (
                            mask
                            if step == steps - 1
                            else mask
                            & (
                                torch.rand(
                                    shape, device=values.device, generator=generator
                                )
                                < (time - next_time) / time
                            )
                        )
                        if bool(reveal.any()):
                            ids = torch.multinomial(
                                torch.softmax(logits.float()[reveal], -1),
                                1,
                                generator=generator,
                            ).squeeze(-1)
                            values[reveal] = codebook[ids]
                        mask = mask & ~reveal
                    if bool(mask.any()):
                        raise RuntimeError("Reverse diffusion left unresolved masks.")
                    actions.append(
                        self._decode(
                            values,
                            p["initial"][index : index + 1],
                            p["zero_time"][index : index + 1],
                            p["metadata"],
                            backbone,
                        )
                    )
                    payloads.append(diagnostics)
                samples.append(torch.cat(actions).reshape(p["shape"]))
        trajectories = torch.stack(samples)
        mean, std = trajectories.mean(0), trajectories.std(0, unbiased=False)
        qdtype = torch.float64 if trajectories.dtype == torch.float64 else torch.float32
        quantiles = torch.quantile(
            trajectories.to(dtype=qdtype),
            trajectories.new_tensor(self.quantile_levels, dtype=qdtype),
            dim=0,
        ).to(trajectories)
        stability = torch.exp(
            -self.ranking_stability_scale * std[:, :, -1].mean((-1, -2, -3))
        )
        diagnostics = self._merge(payloads)
        case_key = "operator_splitting_case_activation"
        if case_key in diagnostics:
            diagnostics[case_key] = (
                torch.stack([item[case_key].squeeze(0) for item in payloads])
                .reshape(count, b, a, -1)
                .mean(0)
            )
        for key, value in dict(
            mddm_sample_count=count,
            mddm_denoising_step_count=steps,
            mddm_final_unresolved_mask_fraction=0,
            mddm_wavelet_packed_time_count=p["length"],
            mddm_uncertainty_calibrated=0,
        ).items():
            diagnostics[key] = mean.new_tensor(float(value))
        output = dict(
            trajectory_mean=mean,
            trajectory_std=std,
            trajectory_quantiles=quantiles,
            ranking_stability=stability,
            ranking_stability_status="uncalibrated_dispersion_proxy",
            uncertainty_status="uncalibrated_predictive_samples"
            if count >= 2
            else "insufficient_predictive_samples",
            rollout_auxiliary_loss=mean.new_zeros(()),
            sampler_diagnostics=diagnostics,
        )
        if self.return_sample_trajectories:
            output["sample_trajectories"] = trajectories
        return output

    def forward(
        self,
        target_trajectory,
        training,
        *,
        occupancy_program,
        occupancy_support,
        initial_state,
        posterior_context,
        time_hours,
        node_coordinates,
        adjacency_matrix,
        backbone,
        sample_weights=None,
        sample_keys=None,
    ):
        conditions = dict(
            occupancy_program=occupancy_program,
            occupancy_support=occupancy_support,
            initial_state=initial_state,
            posterior_context=posterior_context,
            time_hours=time_hours,
            node_coordinates=node_coordinates,
            adjacency_matrix=adjacency_matrix,
            backbone=backbone,
        )
        if training:
            return self.training_loss(
                target_trajectory, **conditions, sample_weights=sample_weights
            )
        if tuple(target_trajectory.shape) != (
            *occupancy_program.shape[:-1],
            self.species_count,
        ):
            raise ValueError("Inference shape disagrees with conditions.")
        return self.sample(**conditions, sample_keys=sample_keys)
