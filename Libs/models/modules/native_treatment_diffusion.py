"""Treatment-conditioned diffusion components from arXiv:2605.29932.

The source specifies a three-layer chronological Transformer encoder,
reconstruction + .1 InfoNCE (.2 temperature), convolutional 2D U-Net/AdaGN,
v prediction, a 1000-step cosine schedule, and .999 EMA. Native rollout uses
the explicitly supplied inference budget (100 rather than source default250).
The reconstruction head and ancestral subsampled sampler are explicit native
implementation choices where the pinned text does not specify code details.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


COSINE_OFFSET = .008


def zero_terminal_cosine_schedule(training_steps: int) -> torch.Tensor:
    """Return sqrt(alpha_bar), sqrt(1-alpha_bar), with terminal SNR exactly zero."""
    if type(training_steps) is not int or training_steps < 2:
        raise ValueError("Cosine velocity diffusion needs at least two training steps")
    times = torch.linspace(0., 1., training_steps+1, dtype=torch.float64)
    curve = torch.cos((times+COSINE_OFFSET)/(1+COSINE_OFFSET)*math.pi/2).square()
    alpha = (curve[1:]/curve[0]).sqrt()
    first, terminal = alpha[0].clone(), alpha[-1].clone()
    alpha = (alpha-terminal)*first/(first-terminal)
    sigma = (1-alpha.square()).sqrt()
    return torch.stack((alpha, sigma), -1)


def velocity_ancestral_step(noisy: torch.Tensor, velocity: torch.Tensor, alpha: float,
                            next_alpha: float, innovation: torch.Tensor) -> torch.Tensor:
    """Gaussian ancestral transition across an explicitly selected clock interval."""
    if noisy.shape != velocity.shape or noisy.shape != innovation.shape:
        raise ValueError("Velocity prediction, noisy field and innovation shapes differ")
    if not all(math.isfinite(value) for value in (alpha, next_alpha)) or not 0 <= alpha < next_alpha <= 1:
        raise ValueError("Reverse diffusion alpha must advance strictly toward the clean field")
    sigma = math.sqrt(1-alpha*alpha)
    clean = alpha*noisy-sigma*velocity
    ratio = alpha/next_alpha
    beta = 1-ratio*ratio
    denominator = 1-alpha*alpha
    mean = next_alpha*beta/denominator*clean+ratio*(1-next_alpha*next_alpha)/denominator*noisy
    variance = beta*(1-next_alpha*next_alpha)/denominator
    return mean+math.sqrt(variance)*innovation


def reverse_diffusion_indices(training_steps: int, inference_steps: int) -> tuple[int, ...]:
    if (type(inference_steps) is not int or inference_steps < 1
            or type(training_steps) is not int or training_steps < 2 or training_steps < inference_steps):
        raise ValueError("Inference steps must be an explicit positive budget within the training clock")
    indices = tuple(torch.linspace(training_steps-1, 0, inference_steps, dtype=torch.float64).round().long().tolist())
    if len(set(indices)) != inference_steps:
        raise ValueError("Subsampled diffusion clock contains repeated model calls")
    return indices


class ChronologicalTreatmentAutoencoder(nn.Module):
    """Three pre-norm Transformer layers; each native query sees its own prefix."""
    def __init__(self, features: int, hidden: int, latent: int, maximum_times: int, heads: int):
        super().__init__()
        if min(features, hidden, latent, maximum_times, heads) < 1 or hidden % heads:
            raise ValueError("Treatment encoder dimensions and attention heads are incompatible")
        self.features, self.maximum_times = features, maximum_times
        self.input_projection = nn.Sequential(nn.Linear(features, hidden), nn.GELU())
        self.positions = nn.Parameter(torch.randn(maximum_times, hidden)*.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=4*hidden,
            dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.latent_projection = nn.Linear(hidden, latent)
        # The article does not specify a decoder architecture; keep this native
        # reconstruction head explicit instead of attributing extra layers to it.
        self.reconstruction = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, features))

    def encode(self, history: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (history.ndim != 3 or history.shape[-1] != self.features
                or history.shape[1] > self.maximum_times
                or valid.shape != history.shape[:2] or valid.dtype != torch.bool
                or not valid.any(1).all() or (valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Treatment histories require nonempty chronological Boolean prefixes")
        if not torch.isfinite(history[valid]).all():
            raise ValueError("Nonfinite supported treatment history")
        tokens = self.input_projection(torch.where(valid[..., None], history, 0))
        tokens = tokens+self.positions[None, :history.shape[1]]
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        latent = torch.where(valid[..., None], self.latent_projection(encoded), 0)
        pooled = latent.sum(1)/valid.sum(1, keepdim=True)
        return latent, F.normalize(pooled, dim=-1)

    def prefix_embeddings(self, history: torch.Tensor) -> torch.Tensor:
        n, times, _ = history.shape
        prefixes = torch.arange(times, device=history.device)[None] <= torch.arange(times, device=history.device)[:, None]
        expanded = history[:, None].expand(n, times, times, self.features).reshape(n*times, times, self.features)
        valid = prefixes[None].expand(n, times, times).reshape(n*times, times)
        _, pooled = self.encode(expanded, valid)
        return pooled.reshape(n, times, -1)

    def reconstruction_contrastive_loss(self, clean: torch.Tensor, first: torch.Tensor,
                                        second: torch.Tensor, valid: torch.Tensor) -> dict:
        if first.shape != clean.shape or second.shape != clean.shape or len(clean) < 2:
            raise ValueError("History contrast needs two views and at least two supplied histories")
        z_first, p_first = self.encode(first, valid)
        z_second, p_second = self.encode(second, valid)
        if not torch.isfinite(clean[valid]).all():
            raise ValueError("Nonfinite clean treatment reconstruction target")
        target = torch.where(valid[..., None], clean, 0)
        squared = ((self.reconstruction(z_first)-target).square()+
                   (self.reconstruction(z_second)-target).square())/2
        reconstruction = torch.where(valid[..., None], squared, 0).sum()/(valid.sum()*self.features)
        logits = p_first @ p_second.T/.2
        labels = torch.arange(len(clean), device=clean.device)
        contrast = (F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))/2
        return {"loss": reconstruction+.1*contrast, "reconstruction_mse": reconstruction,
                "contrastive_infonce": contrast}


class AdaGNResidualBlock2d(nn.Module):
    def __init__(self, channels: int, conditioning: int, groups: int):
        super().__init__()
        if type(groups) is not int or groups < 1 or channels < 1 or channels % groups or channels//groups < 2:
            raise ValueError("AdaGN groups require at least two channels per group on a one-cell feature map")
        self.norm_first = nn.GroupNorm(groups, channels)
        self.norm_second = nn.GroupNorm(groups, channels, affine=False)
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)
        self.modulation = nn.Linear(conditioning, 2*channels)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.first(F.silu(self.norm_first(values)))
        scale, shift = self.modulation(condition).chunk(2, -1)
        hidden = self.norm_second(hidden)*(1+scale[..., None, None])+shift[..., None, None]
        return values+self.second(F.silu(hidden))


class VelocityUNet2d(nn.Module):
    """Four convolutional down/up levels; baseline concatenated on every call."""
    def __init__(self, quantities: int, widths: list[int], conditioning: int, groups: int):
        super().__init__()
        if len(widths) != 4 or min(quantities, conditioning, groups, *widths) < 1:
            raise ValueError("Treatment U-Net requires four positive channel levels")
        self.quantities = quantities
        self.input_projection = nn.Conv2d(2*quantities, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(nn.ModuleList(AdaGNResidualBlock2d(width, conditioning, groups)
                                                       for _ in range(2)) for width in widths)
        self.downsamples = nn.ModuleList(nn.Conv2d(width, widths[min(index+1, 3)], 3, stride=2, padding=1)
                                         for index, width in enumerate(widths))
        self.middle = nn.ModuleList(AdaGNResidualBlock2d(widths[-1], conditioning, groups) for _ in range(2))
        self.merge = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous = widths[-1]
        for width in reversed(widths):
            self.merge.append(nn.Conv2d(previous+width, width, 3, padding=1))
            self.up_blocks.append(nn.ModuleList(AdaGNResidualBlock2d(width, conditioning, groups) for _ in range(2)))
            previous = width
        self.output = nn.Sequential(nn.GroupNorm(groups, widths[0]), nn.SiLU(),
                                    nn.Conv2d(widths[0], quantities, 3, padding=1))

    def forward(self, noisy: torch.Tensor, baseline: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 4 or noisy.shape != baseline.shape or noisy.shape[1] != self.quantities:
            raise ValueError("Treatment U-Net requires aligned noisy and baseline native field channels")
        if condition.ndim != 2 or condition.shape[0] != noisy.shape[0]:
            raise ValueError("Treatment U-Net condition batch differs")
        hidden = self.input_projection(torch.cat((noisy, baseline), 1))
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            hidden = downsample(hidden)
        for block in self.middle:
            hidden = block(hidden, condition)
        for skip, merge, blocks in zip(reversed(skips), self.merge, self.up_blocks):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = merge(torch.cat((hidden, skip), 1))
            for block in blocks:
                hidden = block(hidden, condition)
        return self.output(hidden)


@torch.no_grad()
def update_exponential_average(average: nn.Module, current: nn.Module, decay: float) -> None:
    """Update only the separate EMA copy; physical/configuration buffers copy exactly."""
    if not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("EMA decay must be strictly between zero and one")
    old, new = dict(average.named_parameters()), dict(current.named_parameters())
    old_buffers, new_buffers = dict(average.named_buffers()), dict(current.named_buffers())
    if average is current or old.keys() != new.keys() or old_buffers.keys() != new_buffers.keys():
        raise ValueError("EMA modules differ")
    if (any(value.shape != new[name].shape for name, value in old.items())
            or any(value.shape != new_buffers[name].shape for name, value in old_buffers.items())):
        raise ValueError("EMA parameter/buffer shapes differ")
    for name, value in old.items():
        value.mul_(decay).add_(new[name], alpha=1-decay)
    for name, value in old_buffers.items():
        value.copy_(new_buffers[name])


HISTORY_BLOCKS = ("occupancy", "support", "physical_time", "electrical_current_uA",
                  "electrical_applicability", "optical_power_mW", "optical_applicability")


class NativeTreatmentDiffusion(nn.Module):
    """Query-marginal native extension of the source single-follow-up image model.

    Each denoised query sees only its chronological program prefix. Conditional
    noise is independent between queries and shared across paired actions within
    a complete posterior tuple; no learned temporal covariance is claimed.
    """
    prediction_kind = "conditional_native_query_diffusion"

    def __init__(self, settings: dict, inputs, config: dict):
        super().__init__()
        if (settings["training_diffusion_steps"] != 1000 or settings["ema_decay"] != .999
                or settings["schedule"] != "zero_terminal_squared_cosine"
                or settings["loss"] != "supported_native_velocity_mae"
                or settings["temporal_law"] != "independent_query_noise_given_tuple_and_prefix"):
            raise ValueError("Treatment diffusion requires its declared source schedule and native mapping")
        self.field_shape = tuple(inputs.initial.shape[1:])
        self.blocks = dict(inputs.blocks)
        self.normalization_id = inputs.normalization_id
        self.region_ids = list(inputs.grid["region_id"])
        self.quantity_ids = list(inputs.grid["species_id"])
        conditioning = config["native_controlled_comparators"]["conditioning"]
        self.maximum_times = int(conditioning["maximum_times"])
        self.time_scale = float(conditioning["program_reference_scales"]["physical_time"])
        self.training_steps = settings["training_diffusion_steps"]
        self.ema_decay = float(settings["ema_decay"])
        self.auxiliary_weight = float(settings["autoencoder_loss_weight"])
        self.view_drop_probability = float(settings["autoencoder_view_drop_probability"])
        self.query_batch_size = settings["query_batch_size"]
        if (not math.isfinite(self.time_scale) or self.time_scale <= 0
                or not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight <= 0
                or not 0 < self.view_drop_probability < 1
                or type(self.query_batch_size) is not int or self.query_batch_size < 1):
            raise ValueError("Treatment native scales, auxiliary weight and query batch must be positive")
        features = 0
        for name in HISTORY_BLOCKS:
            lo, hi = self.blocks[name]
            if (hi-lo) % self.maximum_times:
                raise ValueError("Treatment program block cannot be unpacked onto the original clock")
            features += (hi-lo)//self.maximum_times
        first, last = self.blocks["initial_field"][1], self.blocks["source_nuisance"][1]
        domain_lo, domain_hi = self.blocks["physical_domain"]
        self.static_slice = (first, last)
        self.domain_slice = (domain_lo, domain_hi)
        width = settings["conditioning_dim"]
        if type(width) is not int or width < 2 or width % 2:
            raise ValueError("Treatment diffusion conditioning width must be a positive even integer")
        self.condition_width = width
        self.online = nn.ModuleDict({
            "history": ChronologicalTreatmentAutoencoder(features, settings["encoder_hidden_dim"],
                settings["history_latent_dim"], self.maximum_times, settings["encoder_heads"]),
            "static": nn.Sequential(nn.Linear(last-first+domain_hi-domain_lo, width), nn.SiLU(),
                                    nn.Linear(width, width)),
            "history_projection": nn.Linear(settings["history_latent_dim"], width),
            "horizon": nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width)),
            "diffusion_time": nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)),
            "velocity": VelocityUNet2d(self.field_shape[-1], settings["unet_widths"], width,
                                       settings["group_norm_groups"]),
        })
        self.ema = copy.deepcopy(self.online).requires_grad_(False).eval()
        self.register_buffer("diffusion_schedule", zero_terminal_cosine_schedule(self.training_steps))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.last_training_diagnostics = {}
        self.method_mapping = {
            "reference": "arxiv:2605.29932v1", "source_training_diffusion_steps": 1000,
            "source_default_inference_steps": 250, "native_inference_budget": "explicit_call_argument",
            "inference_budget_mapping": "source_default250/native_matched_budget100",
            "source_unet_widths": [128,256,512,512], "actual_unet_widths": list(settings["unet_widths"]),
            "source_encoder_hidden_dim": 256, "source_history_latent_dim": 128,
            "actual_encoder_hidden_dim": settings["encoder_hidden_dim"],
            "actual_history_latent_dim": settings["history_latent_dim"],
            "native_spatial_mapping": "region_by_compartment_with_quantity_channels_no_image_geometry_claim",
            "temporal_law": settings["temporal_law"], "learned_temporal_covariance": False,
            "native_loss": settings["loss"], "source_roi_weights_transferred": False,
            "query_sampling": "postzero_query_proportional_to_supported_cell_count",
            "native_autoencoder_weight": self.auxiliary_weight,
            "native_autoencoder_views": "token_dropout_copies_only_clean_program_target_unchanged",
            "native_decoder": "per_token_two_layer_mlp",
            "native_sampler": "subsampled_velocity_gaussian_ancestral",
            "paired_action_noise": "common_innovations_within_tuple_query_draw",
            "online_parameter_count": sum(p.numel() for p in self.online.parameters()),
            "ema_parameter_count": sum(p.numel() for p in self.ema.parameters()),
            "reported_uncertainty_calibrated": False, "scientific_qualification": False,
        }

    def _prepare(self, inputs):
        if (tuple(inputs.initial.shape[1:]) != self.field_shape or inputs.blocks != self.blocks
                or inputs.normalization_id != self.normalization_id
                or inputs.grid["region_id"] != self.region_ids
                or inputs.grid["species_id"] != self.quantity_ids):
            raise ValueError("Treatment native field, conditioning or normalization identity changed")
        n, times = inputs.validity.shape[:2]
        if (not 2 <= times <= self.maximum_times or inputs.validity.dtype != torch.bool
                or inputs.initial.shape[0] != n or n != math.prod(inputs.group_shape)):
            raise ValueError("Treatment rollout requires complete tuple/action fields and a follow-up clock")
        history = torch.cat([inputs.condition[:, lo:hi].reshape(n, self.maximum_times, -1)[:, :times]
                             for lo, hi in (self.blocks[name] for name in HISTORY_BLOCKS)], -1)
        first, last = self.static_slice
        lo, hi = self.domain_slice
        static = torch.cat((inputs.condition[:, first:last], inputs.condition[:, lo:hi]), -1)
        clock = inputs.initial.new_tensor(inputs.grid["time_hours"])
        if (clock.shape != (times,) or clock[0] != 0 or not torch.isfinite(clock).all()
                or (clock[1:] <= clock[:-1]).any()
                or not torch.isfinite(history).all() or not torch.isfinite(static).all()):
            raise ValueError("Treatment conditioning requires finite programs and a strict original physical clock")
        horizons = torch.stack((clock, torch.cat((clock[:1], clock[1:]-clock[:-1]))), -1)/self.time_scale
        return history, static, horizons

    def _diffusion_embedding(self, indices: torch.Tensor, dtype) -> torch.Tensor:
        half = self.condition_width//2
        frequencies = torch.exp(-math.log(10000)*torch.arange(
            half, device=indices.device, dtype=dtype)/max(half-1, 1))
        phases = indices.to(dtype)[:, None]*frequencies[None]
        return torch.cat((phases.sin(), phases.cos()), -1)

    @torch.no_grad()
    def update_ema(self) -> None:
        update_exponential_average(self.ema, self.online, self.ema_decay)
        self.ema_updates.add_(1)

    def _autoencoder_loss(self, history: torch.Tensor, generator) -> tuple[torch.Tensor, dict]:
        # Duplicate program histories are false contrastive negatives. This
        # deduplication never removes forecast particles or changes their weights.
        clean = torch.unique(history.detach().flatten(1), dim=0).reshape(-1, *history.shape[1:])
        valid = torch.ones(clean.shape[:2], device=clean.device, dtype=torch.bool)
        views = [clean*(torch.rand((*clean.shape[:2], 1), device=clean.device, generator=generator)
                        >= self.view_drop_probability) for _ in range(2)]
        encoder = self.online["history"]
        if len(clean) >= 2:
            result = encoder.reconstruction_contrastive_loss(clean, *views, valid)
            contrast_status = "defined_on_distinct_program_histories"
        else:
            losses = []
            for view in views:
                latent, _ = encoder.encode(view, valid)
                losses.append((encoder.reconstruction(latent)-clean).square().mean())
            reconstruction = torch.stack(losses).mean()
            result = {"loss": reconstruction, "reconstruction_mse": reconstruction,
                      "contrastive_infonce": reconstruction.new_zeros(())}
            contrast_status = "unavailable_fewer_than_two_distinct_program_histories"
        return result["loss"], {
            "distinct_history_count": len(clean), "contrastive_status": contrast_status,
            "reconstruction_mse": float(result["reconstruction_mse"].detach()),
            "contrastive_infonce": float(result["contrastive_infonce"].detach()),
        }

    def loss_native(self, inputs, target, valid, weights, generator):
        history, static, horizons = self._prepare(inputs)
        if (target.shape != inputs.validity.shape or valid.shape != target.shape
                or valid.dtype != torch.bool or weights.shape != (len(target),)
                or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("Treatment loss requires aligned supported fields and tuple/action weights")
        if not torch.isfinite(target[valid]).all():
            raise FloatingPointError("Nonfinite supported Treatment target")
        counts = valid[:, 1:].flatten(2).sum(-1)
        if not (counts.sum(-1) > 0).all():
            raise ValueError("Treatment loss has no supported follow-up cells")
        # This estimates the uniform supported-cell MAE of every complete row.
        query = torch.multinomial(counts.to(target), 1, generator=generator).squeeze(1)+1
        row = torch.arange(len(target), device=target.device)
        supported = valid[row, query].permute(0, 3, 1, 2)
        clean = torch.where(supported, target[row, query].permute(0, 3, 1, 2), 0)
        prefix = torch.arange(history.shape[1], device=history.device)[None] <= query[:, None]
        _, encoded = self.online["history"].encode(history, prefix)
        condition = (self.online["static"](static)+self.online["history_projection"](encoded)
                     +self.online["horizon"](horizons[query]))
        indices = torch.randint(self.training_steps, (len(target),), device=target.device, generator=generator)
        alpha, sigma = self.diffusion_schedule[indices].to(target).unbind(-1)
        alpha, sigma = alpha[:, None, None, None], sigma[:, None, None, None]
        innovation = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        noisy = torch.where(supported, alpha*clean+sigma*innovation, 0)
        velocity_target = alpha*innovation-sigma*clean
        condition = condition+self.online["diffusion_time"](self._diffusion_embedding(indices, target.dtype))
        prediction = self.online["velocity"](noisy, inputs.initial.permute(0, 3, 1, 2), condition)
        error = torch.where(supported, (prediction-velocity_target).abs(), 0).flatten(1).sum(-1)
        row_mae = error/supported.flatten(1).sum(-1)
        diffusion_loss = (row_mae*weights).sum()/weights.sum()
        auxiliary, diagnostics = self._autoencoder_loss(history, generator)
        self.last_training_diagnostics = dict(diagnostics, diffusion_velocity_mae=float(diffusion_loss.detach()),
            selected_query_indices=query.detach().cpu().tolist(), physical_programs_modified=False)
        return diffusion_loss+self.auxiliary_weight*auxiliary

    @torch.no_grad()
    def sample_native(self, inputs, normalization, draws: int, steps: int, seed: int):
        if type(draws) is not int or draws < 1 or normalization.identity != inputs.normalization_id:
            raise ValueError("Treatment sampling needs positive draws and the bound normalization")
        indices = reverse_diffusion_indices(self.training_steps, steps)
        normalization.assert_domain(inputs.grid)
        history, static, horizons = self._prepare(inputs)
        self.ema.eval()
        b, j, a = inputs.group_shape
        n, times = inputs.validity.shape[:2]
        r, c, q = self.field_shape
        encoded = self.ema["history"].prefix_embeddings(history)[:, 1:]
        condition = (self.ema["static"](static)[:, None]+self.ema["history_projection"](encoded)
                     +self.ema["horizon"](horizons[1:])[None])
        # Query-major packing keeps the earlier query batches unchanged when a
        # later horizon is appended. GroupNorm does not mix rows of the batch.
        condition = condition.transpose(0, 1).reshape(-1, self.condition_width)
        baseline = inputs.initial.permute(0, 3, 1, 2)[None].expand(times-1, n, q, r, c).reshape(-1, q, r, c)
        support = inputs.validity[:, 1:].permute(1, 0, 4, 2, 3).reshape(-1, q, r, c)
        network_calls = 0
        results = []
        for draw in range(draws):
            streams = [torch.Generator(device=inputs.initial.device).manual_seed(
                seed+draw*self.maximum_times+query) for query in range(1, times)]

            def paired_innovation():
                pieces = []
                for stream in streams:
                    atom = torch.randn((b, j, 1, q, r, c), device=inputs.initial.device,
                                       dtype=inputs.initial.dtype, generator=stream)
                    pieces.append(atom.expand(b, j, a, q, r, c).reshape(n, q, r, c))
                return torch.stack(pieces).reshape(-1, q, r, c)

            field = torch.where(support, paired_innovation(), 0)
            for position, index in enumerate(indices):
                alpha = float(self.diffusion_schedule[index, 0])
                next_alpha = float(self.diffusion_schedule[indices[position+1], 0]) if position+1 < len(indices) else 1.
                timestep = torch.full((1,), index, device=field.device, dtype=torch.long)
                time_condition = self.ema["diffusion_time"](self._diffusion_embedding(timestep, field.dtype))
                velocity = torch.empty_like(field)
                for start in range(0, len(field), self.query_batch_size):
                    stop = min(start+self.query_batch_size, len(field))
                    velocity[start:stop] = self.ema["velocity"](
                        field[start:stop], baseline[start:stop], condition[start:stop]+time_condition)
                    network_calls += 1
                field = velocity_ancestral_step(field, velocity, alpha, next_alpha, paired_innovation())
                field = torch.where(support, field, 0)
            followup = field.reshape(times-1, n, q, r, c).permute(1, 0, 3, 4, 2)
            normalized = torch.cat((inputs.initial[:, None], followup), 1)
            physical = normalization.inverse(normalized, inputs.validity).reshape(b, j, a, times, r, c, q)
            original = inputs.posterior.support.initial_field[:, :, None].expand(b, j, a, r, c, q)
            physical = torch.cat((original[:, :, :, None], physical[:, :, :, 1:]), 3)
            if not torch.isfinite(physical).all():
                raise FloatingPointError("Nonfinite Treatment diffusion native forecast")
            results.append(physical)
        return {
            "particle_trajectory_samples": torch.stack(results), "particle_weights": inputs.weights,
            "path_weights": inputs.weights[None].expand(draws, b, j)/draws,
            "posterior_uid": inputs.posterior_uid, "particle_uids": inputs.particle_uids,
            "regimen_uids": inputs.regimen_uids, "sample_uids": inputs.sample_uids,
            "normalization_id": inputs.normalization_id, "conditional_draws_per_particle": draws,
            "forecast_paths_per_action": j*draws,
            "draw_semantics": "weighted_complete_particle_conditional_query_samples",
            "execution": {"training_diffusion_steps": self.training_steps,
                "source_default_inference_steps": 250, "denoising_steps_per_query": steps,
                "velocity_network_calls": network_calls,
                "velocity_field_evaluations": draws*steps*n*(times-1),
                "query_path_count": draws*n*(times-1), "original_time_count": times,
                "complete_particle_count": j, "paired_action_count": a,
                "forecast_paths_per_action": j*draws, "inference_weights": "ema",
                "ema_updates": int(self.ema_updates), "learned_temporal_covariance": False,
                "new_source_provider_calls": 0, "reported_uncertainty_calibrated": False},
        }
