from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

import torch

from Libs.models.modules.world_action_encoder import NATIVE_TARGET_IDENTITY_SCHEMA
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Libs.models.modules.neuronsifter_world_action import MechanismOperatorExpertBank, NeuronSifterWorldActionOperator
from Libs.models.modules.world_action_encoder import DIGEST_WIDTH
from Libs.utils.data.native_field_normalization import pack_field, NativeFieldNormalization
from Libs.utils.data.native_observable_profile import (
    validate_quantity_support,
    coefficient_support, constrain_quantities, family_grid, observable_readout, profile_identity, quantity_support,
)
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES, STIMULATION_SCHEMA
from Libs.utils.data.physical_occupancy import validate_scalar_exposure_definition


def conditional_nonempty_mask(u: torch.Tensor, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Independent Bernoulli(u) conditioned on at least one success, exactly.

    Draw the first success from its truncated geometric law, then draw the
    remaining independent Bernoullis. Forcing an extra mask is a different law.
    """
    if count < 1 or u.numel() != 1 or not bool(torch.isfinite(u) & (u > 0) & (u <= 1)):
        raise ValueError("A nonempty mask requires D >= 1 and finite 0 < u <= 1.")
    positions = torch.arange(count, device=u.device)
    mass = -torch.expm1(count * torch.log1p(-u.double()))
    cdf = -torch.expm1((positions + 1) * torch.log1p(-u.double())) / mass
    first = torch.searchsorted(cdf, torch.rand((), device=u.device, generator=generator).double()).clamp_max(count - 1)
    draws = torch.rand(count, device=u.device, generator=generator)
    return (positions == first) | ((positions > first) & (draws < u))


def masked_categorical_objective(
    logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor, u: torch.Tensor,
    confidence: torch.Tensor, *, conditioned_nonempty: bool = True, support: torch.Tensor | None = None,
) -> torch.Tensor:
    """1/(D*u) masked CE, with P(M != empty) importance correction."""
    if logits.shape[:-1] != target_ids.shape or mask.shape != target_ids.shape:
        raise ValueError("Categorical logits, teacher tokens and mask shapes differ.")
    if mask.dtype != torch.bool or target_ids.dtype != torch.long:
        raise ValueError("Categorical mask must be bool and teacher tokens int64.")
    if u.numel() != 1 or not bool(torch.isfinite(u) & (u > 0) & (u <= 1)):
        raise ValueError("Diffusion time must satisfy 0 < u <= 1.")
    if confidence.numel() != 1 or not bool(torch.isfinite(confidence) & (confidence >= 0)):
        raise ValueError("Teacher confidence must be a finite nonnegative scalar per trajectory.")
    if support is not None:
        if support.shape != target_ids.shape or support.dtype != torch.bool or not support.any() or (mask & ~support).any():
            raise ValueError("Categorical masks must select only supported coefficients.")
        logits, target_ids, mask = logits[support], target_ids[support], mask[support]
    count = target_ids.numel()
    correction = -torch.expm1(count * torch.log1p(-u.double())) if conditioned_nonempty else u.new_ones(())
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target_ids.reshape(-1), reduction="none")
    return confidence * correction.to(ce.dtype) * (ce * mask.reshape(-1)).sum() / (count * u)


def masked_categorical_distillation(logits, teacher_logits, mask, u, confidence, support):
    """KL from the frozen teacher with the same nonempty-mask importance correction."""
    if (logits.shape != teacher_logits.shape or support.shape != mask.shape
            or mask.shape != logits.shape[:-1] or support.dtype != torch.bool
            or mask.dtype != torch.bool or (mask & ~support).any() or not mask.any()
            or not torch.isfinite(logits).all() or not torch.isfinite(teacher_logits).all()):
        raise ValueError("Learned categorical distillation requires aligned finite native logits/support/mask")
    count = int(support.sum())
    if not 0 < float(u) <= 1:
        raise ValueError("Learned categorical distillation needs a valid diffusion time")
    target_log = teacher_logits.detach().float().log_softmax(-1)
    divergence = (target_log.exp() * (target_log - logits.float().log_softmax(-1))).sum(-1)
    correction = -torch.expm1(count * torch.log1p(-u.double()))
    return confidence.detach() * correction.to(divergence) * divergence[mask].sum() / (count * u)


class QuantizedMddmRolloutOperator(nn.Module):
    """Native absorbing categorical diffusion on H(X-X0).

    The trajectory argument supplies shape only during inference. Clean targets
    are an independent teacher argument. The native operator is passed at call
    time so its parameters have one owner in the parent network/state dict.
    """
    def __init__(self, config: dict[str, Any], hidden_dim: int | None = None, species_count: int = 1,
                 *, action_policy: str = "occupancy_program", scalar_exposure_definition: dict | None = None,
                 quantity_profile: dict | None = None, posterior_context_dim: int | None = None) -> None:
        super().__init__()
        if action_policy not in {"occupancy_program", "drug_id", "target_id", "scalar_exposure"}:
            raise ValueError(f"Native action policy {action_policy!r} is unimplemented; only occupancy_program, drug_id, target_id and scalar_exposure are supported.")
        self.action_policy_name = action_policy
        self.quantity_profile = copy.deepcopy(quantity_profile)
        if quantity_profile is not None and len(quantity_profile["quantities"]) != species_count:
            raise ValueError("Observable profile/model quantity count mismatch.")
        self.scalar_exposure_definition = (validate_scalar_exposure_definition(scalar_exposure_definition)
                                           if action_policy == "scalar_exposure" else None)
        self.register_buffer("action_conditioning_identity", self._canonical_action_identity())
        if not config["enabled"]:
            raise ValueError("Quantized MDDM requires enabled=true.")
        if config["residual_reference_policy"] != "initial_state":
            raise ValueError("Native categorical diffusion requires fixed initial_state; use the continuous ablation for legacy residual refinement.")
        if config["token_axes"] != ["time", "region", "compartment", "species"]:
            raise ValueError("Token axes must be time/region/compartment/species.")
        if config["external_reference_policy"]["reference_policy"] != "project_native_no_direct_import":
            raise ValueError("The denoiser must be project native.")
        if config["training_time_distribution"] != "uniform_open_unit_interval":
            raise ValueError("Unsupported diffusion-time distribution.")
        if config["training_objective"] != "absorbing_masked_categorical_nonempty_importance":
            raise ValueError("Unsupported masked categorical training objective.")
        token = config["residual_tokenization"]
        schedule = config["partial_mask_schedule"]
        reconstruction = config["inverse_reconstruction_policy"]
        if schedule["schedule_kind"] != "absorbing_linear_reverse":
            raise ValueError("Unsupported categorical reverse schedule.")
        if schedule["inference_initial_mask_probability"] != 1.0 or schedule["final_mask_probability"] != 0.0:
            raise ValueError("Categorical reverse sampling must start fully masked and finish unmasked.")
        self.species_count = int(species_count)
        self.hidden_dim = int(hidden_dim or token["token_hidden_dim"])
        self.codebook_size = int(token["codebook_size"])
        self.residual_min_value = float(token["residual_min_value"])
        self.residual_max_value = float(token["residual_max_value"])
        if self.codebook_size < 4 or self.residual_max_value <= self.residual_min_value:
            raise ValueError("Invalid scalar codebook.")
        self.mask_token_value = float(token["mask_token_value"])
        for key in ("training_sample_count", "inference_sample_count", "sample_chunk_size",
                    "denoising_token_chunk_size", "training_denoising_step_count", "inference_denoising_step_count"):
            value = int(schedule[key])
            if value < 1:
                raise ValueError(f"{key} must be positive.")
            setattr(self, key, value)
        self.sampler_seed = int(schedule["sampler_seed"])
        if reconstruction["reconstruction_policy"] != "inverse_haar_add_initial":
            raise ValueError("Native categorical reconstruction requires inverse_haar_add_initial.")
        self.preserve_initial_state = bool(reconstruction["preserve_initial_state"])
        if not self.preserve_initial_state:
            raise ValueError("Native categorical reconstruction must preserve the inferred initial snapshot.")
        self.quantile_levels = [float(value) for value in reconstruction["quantile_levels"]]
        if not self.quantile_levels or any(not 0 <= q <= 1 for q in self.quantile_levels):
            raise ValueError("Quantile levels must be in [0,1].")
        self.return_sample_trajectories = bool(reconstruction["return_sample_trajectories"])
        self.ranking_stability_scale = float(reconstruction["ranking_stability_scale"])
        self.loss_weights = {key: float(value) for key, value in config["loss_weights"].items()}
        if any(self.loss_weights.get(key, 0) != 0 for key in ("partial_mask_consistency", "ranking_stability_consistency")):
            raise ValueError("Legacy consistency penalties are not native masked/reconstruction losses; their weights must be zero.")
        if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in self.loss_weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative.")
        self._codebook_dtype = torch.get_default_dtype()
        codebook, boundaries = self._canonical_codebook()
        self.register_buffer("codebook_values", codebook)
        self.register_buffer("codebook_boundaries", boundaries)
        self.value_projection = nn.Linear(self.species_count, self.hidden_dim)
        self.mask_projection = nn.Linear(self.species_count, self.hidden_dim)
        self.initial_projection = nn.Linear(self.species_count, self.hidden_dim)
        if self.quantity_profile is not None:
            self.quantity_support_projection = nn.Linear(self.species_count, self.hidden_dim)
        self.posterior_context_dim = self.hidden_dim if posterior_context_dim is None else posterior_context_dim
        if type(self.posterior_context_dim) is not int or self.posterior_context_dim < 1:
            raise ValueError("Posterior context projection requires a positive input width")
        self.context_projection = nn.Linear(self.posterior_context_dim, self.hidden_dim)
        # occupancy, support, physical time and diffusion time remain separate.
        self.condition_projection = nn.Linear(8 if self.quantity_profile is not None else 4, self.hidden_dim)
        self.token_denoiser = nn.Linear(self.hidden_dim, self.species_count * self.codebook_size)

    def _canonical_action_identity(self) -> torch.Tensor:
        # Persistent metadata, not learned parameters or floating-point features.
        definition = (validate_scalar_exposure_definition(self.scalar_exposure_definition)
                      if self.action_policy_name == "scalar_exposure" else None)
        identity = {"action_policy": self.action_policy_name, "scalar_exposure_definition": definition}
        if self.quantity_profile is not None:
            identity["quantity_profile_sha256"] = profile_identity(self.quantity_profile)
            identity["stimulation_schema"] = STIMULATION_SCHEMA
        if self.action_policy_name == "target_id":
            identity["action_identity_schema"] = NATIVE_TARGET_IDENTITY_SCHEMA
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return torch.tensor(list(encoded), dtype=torch.uint8)

    def validate_checkpoint_action_identity(self, state_dict: Mapping, prefix: str = "") -> None:
        key = prefix + "action_conditioning_identity"
        if key not in state_dict:
            if self.quantity_profile is not None:
                raise ValueError("Observable checkpoint requires quantity profile identity.")
            if self.action_policy_name == "scalar_exposure":
                raise ValueError("Scalar exposure checkpoint requires explicit action policy and definition identity.")
            if self.action_policy_name == "target_id":
                raise ValueError("Target identity checkpoint requires explicit action policy and schema identity.")
            # Legacy non-scalar states predate this metadata. They cannot supply
            # the missing provenance needed to resume a scalar comparison.
            return
        value = state_dict[key]
        if (not torch.is_tensor(value) or value.dtype != torch.uint8
                or not torch.equal(value.detach().cpu(), self._canonical_action_identity())):
            raise ValueError("Checkpoint action policy or scalar exposure definition or target schema identity mismatch.")

    def _canonical_codebook(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve constructor arithmetic, including midpoint rounding, through
        # later model dtype/device casts. Checkpoint buffers are not authority.
        values = torch.linspace(self.residual_min_value, self.residual_max_value, self.codebook_size,
                                dtype=self._codebook_dtype, device="cpu")
        return values, (values[:-1] + values[1:]) / 2

    def assert_codebook(self) -> None:
        for name, reference in zip(("codebook_values", "codebook_boundaries"), self._canonical_codebook()):
            actual = getattr(self, name)
            if not torch.equal(actual, reference.to(actual)):
                raise ValueError(f"Native quantizer {name} differs from configured codebook.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.validate_checkpoint_action_identity(state_dict, prefix)
        key = prefix + "action_conditioning_identity"
        if key not in state_dict:
            state_dict = dict(state_dict, **{key: self._canonical_action_identity()})
        for name, reference in zip(("codebook_values", "codebook_boundaries"), self._canonical_codebook()):
            key = prefix + name
            if key in state_dict and not torch.equal(state_dict[key].detach().cpu(), reference.to(getattr(self, name)).cpu()):
                raise ValueError(f"Checkpoint quantizer {name} differs from configured codebook.")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.assert_codebook()

    def quantize_residual(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.assert_codebook()
        if not torch.isfinite(residual).all():
            raise ValueError("Residual coefficients must be finite.")
        ids = torch.bucketize(residual.detach().clamp(self.residual_min_value, self.residual_max_value).contiguous(),
                             self.codebook_boundaries.to(residual))
        quantized = self.codebook_values.to(residual)[ids]
        return residual + (quantized - residual).detach(), ids

    @staticmethod
    def pack(trajectory: torch.Tensor, native: NeuronSifterWorldActionOperator) -> tuple[torch.Tensor, dict[str, object]]:
        return pack_field(trajectory, native.tokenizer)

    @staticmethod
    def unpack(coefficients: torch.Tensor, layout: dict[str, object], native: NeuronSifterWorldActionOperator) -> torch.Tensor:
        pieces = coefficients.movedim(-4, -2).split(layout["sizes"], dim=-2)
        payload = {"approximation": pieces[0], "details": list(reversed(pieces[1:])), "lengths": layout["lengths"]}
        return native.tokenizer.temporal_wavelet_synthesis(payload).movedim(-2, -4)

    def generator(self, device: torch.device, sample_id: int, regimen_id: int, draw: int) -> torch.Generator | None:
        if self.sampler_seed < 0:
            return None
        # Stable identity, independent of chunk sizes, loop batching and merge order.
        seed = (self.sampler_seed + 1000003 * sample_id + 1000033 * regimen_id + 1000037 * draw) % (2**63 - 1)
        return torch.Generator(device=device).manual_seed(seed)

    def logits(
        self, coefficients: torch.Tensor, mask: torch.Tensor, u: torch.Tensor,
        context: dict[str, torch.Tensor], native: NeuronSifterWorldActionOperator,
        adjacency: torch.Tensor, coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if coefficients.ndim not in (4, 5):
            raise ValueError("Native logits require a field with an optional leading trajectory batch.")
        casewise = coefficients.ndim == 5
        batch = coefficients.shape[0] if casewise else 1
        time_count, regions, compartments, _ = coefficients.shape[-4:]
        nodes = regions * compartments
        dtype = self.value_projection.weight.dtype
        values = torch.where(mask, self.mask_token_value, coefficients).to(dtype)
        hidden = self.value_projection(values) + self.mask_projection(mask.to(dtype))
        if self.quantity_profile is not None:
            hidden = hidden + self.quantity_support_projection(context["coefficient_support"].to(dtype))
        # Admission still validates the real program in forward. Only the learned
        # predictor is ablated; zero inputs do not declare supported/no-action data.
        program_context = dict(context)
        if self.action_policy_name in {"drug_id", "target_id"}:
            for key in ("occupancy", "support"):
                program_context[key] = torch.zeros_like(context[key])
        if self.action_policy_name == "scalar_exposure":
            # One named site AUC, repeated only along time. No support/identity waveform.
            program_context["occupancy"] = context["scalar_exposure_auc"].unsqueeze(-4).expand_as(context["occupancy"])
            program_context["support"] = torch.zeros_like(context["support"])
        keys = ("occupancy", "support", "physical_time") + (FORCING_ARRAY_NAMES if self.quantity_profile is not None else ())
        packed_context = torch.cat([self.pack(program_context[key], native)[0] for key in keys], dim=-1)
        diffusion_time = u.expand(*packed_context.shape[:-1], 1)
        condition = self.condition_projection(torch.cat([packed_context, diffusion_time], dim=-1).to(dtype))
        if self.action_policy_name in {"drug_id", "target_id"}:
            # The fixed four-component digest uses the same existing projection:
            # identical parameter shapes/counts and time/sampling budgets by policy.
            # Reuse the original occupancy/support/time/diffusion columns. The
            # typed native forcing channels add columns, not identity dimensions.
            identity_weight = self.condition_projection.weight[:, [0, 1, 2, -1]]
            identity_condition = F.linear(context["action_identity"].to(dtype), identity_weight)
            condition = condition + (identity_condition[:, None, None, None, :] if casewise else identity_condition)
        condition = condition + self.initial_projection(context["initial_state"].to(dtype)).unsqueeze(-4)
        condition = condition + self.context_projection(context["posterior_context"].to(dtype)).unsqueeze(-4)
        encoded, diagnostics = native(
            hidden.reshape(batch, time_count, nodes, self.hidden_dim),
            condition.reshape(batch, time_count, nodes, self.hidden_dim),
            coordinates.reshape(1, 1, nodes, -1).expand(batch, time_count, -1, -1),
            adjacency,
            **({"casewise_diagnostics": True} if casewise else {}),
        )
        # Keep the original per-trajectory projection chunks; attention sees the
        # entire trajectory, with independent trajectories on the batch axis.
        flat = encoded.reshape(batch, -1, self.hidden_dim)
        logits = torch.stack([torch.cat([self.token_denoiser(part)
            for part in row.split(self.denoising_token_chunk_size)], dim=0) for row in flat])
        logits = logits.reshape(batch, time_count, regions, compartments, self.species_count, self.codebook_size)
        return (logits if casewise else logits[0]), diagnostics

    @staticmethod
    def categorical_draw(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if not torch.isfinite(logits).all():
            raise ValueError("Categorical logits must be finite.")
        uniform = torch.rand(logits.shape[:-1], device=logits.device, generator=generator)
        cdf = logits.float().softmax(-1).cumsum(-1)
        return (uniform.unsqueeze(-1) > cdf).sum(-1).clamp_max(logits.shape[-1] - 1)

    def reverse_step(self, tokens: torch.Tensor, logits: torch.Tensor, u: float, v: float,
                     generator: torch.Generator | None = None, support: torch.Tensor | None = None) -> torch.Tensor:
        if support is not None:
            if support.shape != tokens.shape or support.dtype != torch.bool:
                raise ValueError("Reverse sampling requires aligned boolean coefficient support.")
            result = tokens.clone()
            result[support] = self.reverse_step(tokens[support], logits[support], u, v, generator)
            return result
        if not 0 <= v < u <= 1:
            raise ValueError("Reverse times must satisfy 0 <= v < u <= 1.")
        masked = tokens == self.codebook_size
        reveal = torch.rand(tokens.shape, device=tokens.device, generator=generator) < (u - v) / u
        sampled = self.categorical_draw(logits, generator)
        return torch.where(masked & reveal, sampled, tokens)

    def reconstruct(self, coefficients: torch.Tensor, initial: torch.Tensor,
                    layout: dict[str, object], native: NeuronSifterWorldActionOperator) -> torch.Tensor:
        trajectory = self.unpack(coefficients, layout, native) + initial.unsqueeze(-4)
        # Do not impose a global nonnegative clamp on signed channels (e.g. voltage).
        if self.preserve_initial_state:
            trajectory = torch.cat([initial.unsqueeze(-4), trajectory[..., 1:, :, :, :]], dim=-4)
        return trajectory

    def score_reverse_step(self, tokens, values, mask, u, v, generator_state,
                           packed_support, context, native_operator, adjacency, coordinates):
        """Recompute one categorical step without retaining its operator activations.

        The explicit generator state makes backward recomputation replay the same
        categorical draw. Checkpoint also preserves the module/dropout RNG state.
        Only selected-path scalars and categorical state escape this step.
        """
        generator = torch.Generator(device=values.device)
        generator.set_state(generator_state)
        logits, diagnostics = self.logits(values, mask, u, context, native_operator, adjacency, coordinates)
        updated = self.reverse_step(tokens, logits, float(u), v, generator, packed_support)
        revealed = (tokens == self.codebook_size) & (updated != self.codebook_size)
        selected = logits.float().log_softmax(-1).gather(
            -1, updated.clamp_max(self.codebook_size - 1).unsqueeze(-1)).squeeze(-1)
        return updated, selected[revealed].sum(), diagnostics, generator.get_state()

    @torch.no_grad()
    def _sample_seeded_evaluation(self, cases, sample_count, steps, sample_ids, regimen_ids,
                                  draw_offset, native_operator, adjacency, coordinates, reconstruct):
        """Batch neural work, retaining one original support-only RNG per path."""
        draws, diagnostics_rows, case_rows = [], [], []
        task_count = len(cases) * sample_count
        for start in range(0, task_count, self.sample_chunk_size):
            tasks = [divmod(index, sample_count)
                     for index in range(start, min(start + self.sample_chunk_size, task_count))]
            rows = [cases[index] for index, _ in tasks]
            context = {key: torch.stack([row["context"][key] for row in rows])
                       for key in rows[0]["context"]}
            template = torch.stack([row["template"] for row in rows])
            support = context["coefficient_support"]
            generators = [self.generator(template.device, int(sample_ids[row["bi"]]),
                int(regimen_ids[row["ai"]]), draw_offset + draw + 1)
                for row, (_, draw) in zip(rows, tasks)]
            tokens = torch.full_like(template, self.codebook_size, dtype=torch.long).masked_fill(~support, 0)
            for step in range(steps):
                u, v = 1 - step / steps, 1 - (step + 1) / steps
                mask = tokens == self.codebook_size
                values = self.codebook_values.to(template)[tokens.clamp_max(self.codebook_size - 1)].masked_fill(~support, 0.)
                logits, diagnostics = self.logits(values, mask, template.new_tensor(u), context,
                                                   native_operator, adjacency, coordinates)
                tokens = torch.stack([self.reverse_step(tokens[index], logits[index], u, v, generator,
                    support[index] if row["has_physical_support"] else None)
                    for index, (row, generator) in enumerate(zip(rows, generators))])
            if bool((tokens == self.codebook_size).any()):
                raise RuntimeError("Categorical reverse sampler failed to exhaust masks.")
            coefficients = self.codebook_values.to(template)[tokens].masked_fill(~support, 0.)
            for index, (row, (_, draw)) in enumerate(zip(rows, tasks)):
                draws.append(reconstruct(coefficients[index], row["initial"], row["layout"], row["bi"], row["ai"]))
                record = {key: (value if key == "operator_splitting_identity" else value[index])
                          for key, value in diagnostics.items()}
                diagnostics_rows.append(record)
                if draw == sample_count - 1:
                    case_rows.append(record["operator_splitting_case_activation"].reshape(-1))
        return torch.stack(draws), diagnostics_rows, case_rows

    def forward(
        self, trajectory: torch.Tensor, training: bool, *,
        initial_state: torch.Tensor | None = None, teacher_trajectory: torch.Tensor | None = None,
        teacher_confidence: torch.Tensor | None = None, conditioning: dict[str, torch.Tensor] | None = None,
        native_operator: NeuronSifterWorldActionOperator | None = None,
        adjacency: torch.Tensor | None = None, coordinates: torch.Tensor | None = None,
        sample_ids: torch.Tensor | None = None, regimen_ids: torch.Tensor | None = None, draw_offset: int = 0,
        return_samples: bool | None = None, score_function_training: bool = False,
        field_normalization: NativeFieldNormalization | None = None, learned_teacher=None,
    ) -> dict[str, Any]:
        if trajectory.ndim != 6 or trajectory.shape[-1] != self.species_count or min(trajectory.shape) < 1:
            raise ValueError("Trajectory must have sample/action/time/region/compartment/species axes.")
        if not torch.isfinite(trajectory).all():
            raise ValueError("Trajectory shape carrier must be finite.")
        if native_operator is None or adjacency is None or coordinates is None or initial_state is None or conditioning is None:
            raise ValueError("Native MDDM requires native operator, graph, coordinates, initial state and conditioning.")
        b, a, t, r, c, s = trajectory.shape
        if initial_state.shape != (b, a, r, c, s):
            raise ValueError("Initial state must be a single aligned snapshot per sample/action.")
        expected = {"occupancy": (b, a, t, r, c, 1), "support": (b, a, t, r, c, 1),
                    "physical_time": (b, a, t, r, c, 1), "posterior_context": (b, a, r, c, self.posterior_context_dim)}
        if self.quantity_profile is not None:
            expected["quantity_support"] = (b, a, t, r, c, s)
            expected.update({name: (b, a, t, r, c, 1) for name in FORCING_ARRAY_NAMES})
        if self.action_policy_name in {"drug_id", "target_id"}:
            expected["action_identity"] = (b, a, DIGEST_WIDTH)
        if self.action_policy_name == "scalar_exposure":
            expected.update(scalar_exposure_auc=(b, a, r, c, 1), scalar_exposure_interval=(b, a, 2))
        if self.action_policy_name == "scalar_exposure" and "scalar_exposure_site_support" in conditioning:
            site_support = conditioning["scalar_exposure_site_support"]
            if (site_support.shape != (b, a, r, c, 1) or not torch.isfinite(site_support).all()
                    or not ((site_support == 0) | (site_support == 1)).all()
                    or not site_support.bool().flatten(2).any(-1).all()
                    or (conditioning["scalar_exposure_auc"][~site_support.bool()] != 0).any()):
                raise ValueError("Scalar exposure site domain or excluded placeholder mismatch")
        for key, shape in expected.items():
            if key not in conditioning or conditioning[key].shape != shape or not torch.isfinite(conditioning[key]).all():
                raise ValueError(f"Missing or invalid aligned {key}; expected {shape}.")
        if not torch.isfinite(initial_state).all():
            raise ValueError("Initial state must be finite.")
        for key in ("occupancy", "support"):
            if not bool(((conditioning[key] >= 0) & (conditioning[key] <= 1)).all()):
                raise ValueError(f"{key} must lie in [0,1].")
        times = conditioning["physical_time"]
        if t > 1 and not bool((times[:, :, 1:] > times[:, :, :-1]).all()):
            raise ValueError("Physical times must be strictly increasing.")
        grid, field_support = None, None
        if self.quantity_profile is not None:
            for family in self.quantity_profile["families"]:
                candidate = family_grid(self.quantity_profile, family)
                expected_times = times.new_tensor(candidate["time_hours"])
                if len(expected_times) == t and torch.equal(times, expected_times.reshape(1, 1, t, 1, 1, 1).expand_as(times)):
                    grid = candidate
                    break
            if grid is None:
                raise ValueError("Observable rollout requires one exact registered family clock per batch.")
            field_support = conditioning["quantity_support"]
            validate_quantity_support(field_support, grid)
            for amplitude, applicable in zip(FORCING_ARRAY_NAMES[::2], FORCING_ARRAY_NAMES[1::2]):
                active = conditioning[applicable]
                if not ((active == 0) | (active == 1)).all() or (conditioning[amplitude][active == 0] != 0).any():
                    raise ValueError("Typed forcing requires explicit applicability and zero non-applicable placeholders.")
            if (conditioning["optical_power_mW"] < 0).any():
                raise ValueError("Optical power must be nonnegative.")
            if (initial_state[~field_support[:, :, 0]] != 0).any():
                raise ValueError("Missing initial quantities must use zero storage placeholders.")
            if teacher_trajectory is not None and (teacher_trajectory[~field_support] != 0).any():
                raise ValueError("Missing teacher quantities must use zero storage placeholders.")
        if self.action_policy_name == "scalar_exposure":
            auc = conditioning["scalar_exposure_auc"]
            interval = conditioning["scalar_exposure_interval"]
            if not bool((auc >= 0).all()):
                raise ValueError("Scalar exposure must be nonnegative; values above one are valid.")
            if not (torch.equal(times[:, :, 0], interval[:, :, 0, None, None, None].expand_as(times[:, :, 0]))
                    and torch.equal(times[:, :, -1], interval[:, :, 1, None, None, None].expand_as(times[:, :, -1]))):
                raise ValueError("Scalar exposure original interval disagrees with physical times.")
        if training and (teacher_trajectory is None or teacher_confidence is None):
            raise ValueError("Native MDDM training requires an independent admitted teacher and confidence.")
        if teacher_trajectory is not None:
            if teacher_trajectory.shape != trajectory.shape or not torch.isfinite(teacher_trajectory).all():
                raise ValueError("Independent teacher shape/values are invalid.")
            if teacher_trajectory is trajectory:
                raise ValueError("Generated rollout cannot be its own independent teacher.")
            if teacher_confidence is None or teacher_confidence.shape != (b, a):
                raise ValueError("Teacher confidence must have sample/action axes.")
            if not bool((torch.isfinite(teacher_confidence) & (teacher_confidence >= 0)).all()):
                raise ValueError("Teacher confidence must be finite and nonnegative.")
        sample_ids = torch.arange(b) if sample_ids is None else sample_ids
        regimen_ids = torch.arange(a) if regimen_ids is None else regimen_ids
        for name, ids, count in (("sample_ids", sample_ids, b), ("regimen_ids", regimen_ids, a)):
            if ids.shape != (count,) or ids.dtype != torch.long or len(ids.unique()) != count or bool((ids < 0).any()):
                raise ValueError(f"{name} must contain unique nonnegative int64 identities.")
        source_initial = initial_state
        if field_normalization is not None:
            field_normalization.assert_physical_time(times)
            field_normalization.assert_representation(native_operator, self)
            initial_state = field_normalization.normalize(source_initial, None if field_support is None else field_support[:, :, 0])
        else:
            self.assert_codebook()
        def source_reconstruction(coefficients, initial, layout, bi, ai):
            result = self.reconstruct(coefficients, initial, layout, native_operator)
            if field_normalization is not None:
                result = field_normalization.inverse(result, None if field_support is None else field_support[bi, ai]).to(source_initial.dtype)
            result = constrain_quantities(result, self.quantity_profile)
            return torch.cat([source_initial[bi,ai].unsqueeze(0),result[1:]],dim=0)
        coefficient_rows, normalized_errors = [], []
        sample_count = self.training_sample_count if training else self.inference_sample_count
        steps = self.training_denoising_step_count if training else self.inference_denoising_step_count
        results, training_predictions, ces, recs, diagnostics_rows, case_rows = [], [], [], [], [], []
        path_probabilities, raw_results, learned_kls = [], [], []
        if learned_teacher is not None and (not training or learned_teacher.training
                or any(parameter.requires_grad for parameter in learned_teacher.parameters())):
            raise ValueError("Learned teacher targets require learner training and a frozen eval teacher")
        # Training (including score-function gradients), teacher evaluation and
        # global-RNG/train-mode calls retain their established sequential path.
        batched_evaluation = (not training and teacher_trajectory is None and not score_function_training
                              and self.sampler_seed >= 0 and not self.training
                              and not native_operator.training and self.sample_chunk_size > 1)
        inference_cases = []
        for bi in range(b):
            action_results, action_predictions, action_probabilities, action_raw = [], [], [], []
            for ai in range(a):
                initial = initial_state[bi, ai]
                context = {key: conditioning[key][bi, ai] for key in expected}
                context["initial_state"] = initial
                template, layout = self.pack(torch.zeros_like(trajectory[bi, ai]), native_operator)
                physical_support = None if field_support is None else field_support[bi, ai]
                packed_support = (coefficient_support(physical_support, native_operator.tokenizer.wavelet_level_count)
                                  if physical_support is not None else torch.ones_like(template, dtype=torch.bool))
                context["coefficient_support"] = packed_support
                if batched_evaluation:
                    inference_cases.append({"context": context, "template": template, "layout": layout,
                        "initial": initial, "bi": bi, "ai": ai, "has_physical_support": physical_support is not None})
                    continue
                if teacher_trajectory is not None:
                    # H((X-mu)/scale-(X0-mu)/scale)=H(X-X0)/scale.
                    # Use the same float64 Haar arithmetic as the support fit.
                    residual = teacher_trajectory[bi, ai].detach() - source_initial[bi, ai].detach().unsqueeze(0)
                    if field_normalization is not None and field_normalization.active:
                        residual = teacher_trajectory[bi, ai].detach().double() - source_initial[bi, ai].detach().double().unsqueeze(0)
                    clean, _ = self.pack(residual, native_operator)
                    if field_normalization is not None and field_normalization.active:
                        clean = clean / field_normalization.scale.to(clean)
                    clean = clean.masked_fill(~packed_support, 0.)
                    quantized, target_ids = self.quantize_residual(clean)
                    quantized = quantized.masked_fill(~packed_support, 0.)
                    clamped = clean.clamp(self.residual_min_value,self.residual_max_value)
                    in_range = (clean >= self.residual_min_value) & (clean <= self.residual_max_value) & packed_support
                    axes = (0,1,2)
                    coefficient_rows.append({
                        "coefficient_preclamp_low_count": ((clean < self.residual_min_value) & packed_support).sum(axes),
                        "coefficient_preclamp_high_count": ((clean > self.residual_max_value) & packed_support).sum(axes),
                        "coefficient_denominator": packed_support.sum(axes),
                        "inrange_quantization_squared_error_sum": ((quantized-clean).square()*in_range).sum(axes),
                        "inrange_quantization_denominator": in_range.sum(axes),
                        "saturation_squared_error_sum": ((clean-clamped).square()*packed_support).sum(axes)})
                    gen = self.generator(trajectory.device, int(sample_ids[bi]), int(regimen_ids[ai]), draw_offset)
                    u = torch.rand((), device=trajectory.device, generator=gen).clamp_min(torch.finfo(torch.float32).eps)
                    mask = torch.zeros_like(packed_support)
                    mask[packed_support] = conditional_nonempty_mask(u, int(packed_support.sum()), gen)
                    logits, diag = self.logits(quantized, mask, u, context, native_operator, adjacency, coordinates)
                    expected_values = logits.float().softmax(-1) @ self.codebook_values.float()
                    reconstructed_coefficients = torch.where(mask, expected_values, clean).masked_fill(~packed_support, 0.)
                    reconstruction = source_reconstruction(reconstructed_coefficients, initial, layout, bi, ai)
                    normalized_reconstruction = self.reconstruct(reconstructed_coefficients, initial, layout, native_operator)
                    normalized_target = (field_normalization.normalize(teacher_trajectory[bi,ai].detach(), physical_support)
                                         if field_normalization is not None else teacher_trajectory[bi,ai].detach())
                    error = (normalized_reconstruction-normalized_target).square()
                    normalized_errors.append(error[physical_support].mean() if physical_support is not None else error.mean())
                    confidence = teacher_confidence[bi, ai].detach()
                    if learned_teacher is not None:
                        with torch.no_grad():
                            teacher_logits, _ = learned_teacher.quantized_mddm.logits(
                                quantized, mask, u, context, learned_teacher.operator,
                                learned_teacher.node_adjacency, coordinates)
                        learned_kls.append(masked_categorical_distillation(
                            logits, teacher_logits, mask, u, confidence, packed_support))
                    ces.append(masked_categorical_objective(logits, target_ids, mask, u, confidence,
                        support=packed_support if physical_support is not None else None))
                    error = (reconstruction - teacher_trajectory[bi, ai].detach()).square()
                    recs.append(confidence * (error[physical_support].mean() if physical_support is not None else error.mean()))
                    action_predictions.append(reconstruction)
                    diagnostics_rows.append(diag)
                draws, draw_probabilities, raw_draws = [], [], []
                # Identity-based streams make sample and action chunking exact.
                for chunk_start in range(0, sample_count, self.sample_chunk_size):
                    for draw in range(chunk_start, min(chunk_start + self.sample_chunk_size, sample_count)):
                        gen = self.generator(trajectory.device, int(sample_ids[bi]), int(regimen_ids[ai]), draw_offset + draw + 1)
                        tokens = torch.full_like(template, self.codebook_size, dtype=torch.long)
                        tokens = tokens.masked_fill(~packed_support, 0)
                        path_log_probability = template.new_zeros(())
                        with torch.set_grad_enabled(score_function_training and torch.is_grad_enabled()):
                            for step in range(steps):
                                u, v = 1 - step / steps, 1 - (step + 1) / steps
                                mask = tokens == self.codebook_size
                                values = self.codebook_values.to(template)[tokens.clamp_max(self.codebook_size - 1)].masked_fill(~packed_support, 0.)
                                step_support = packed_support if physical_support is not None else None
                                if score_function_training and torch.is_grad_enabled():
                                    tokens, log_probability, diag, generator_state = checkpoint(
                                        self.score_reverse_step, tokens, values, mask, template.new_tensor(u), v,
                                        gen.get_state(), step_support, context, native_operator, adjacency, coordinates,
                                        use_reentrant=False)
                                    gen.set_state(generator_state)
                                    path_log_probability = path_log_probability + log_probability
                                else:
                                    logits, diag = self.logits(values, mask, template.new_tensor(u), context, native_operator, adjacency, coordinates)
                                    previous_tokens = tokens
                                    tokens = self.reverse_step(tokens, logits, u, v, gen, step_support)
                                    if score_function_training:
                                        revealed = (previous_tokens == self.codebook_size) & (tokens != self.codebook_size)
                                        selected_log_probability = logits.float().log_softmax(-1).gather(
                                            -1, tokens.clamp_max(self.codebook_size - 1).unsqueeze(-1)).squeeze(-1)
                                        path_log_probability = path_log_probability + selected_log_probability[revealed].sum()
                            if bool((tokens == self.codebook_size).any()):
                                raise RuntimeError("Categorical reverse sampler failed to exhaust masks.")
                            coefficients = self.codebook_values.to(template)[tokens].masked_fill(~packed_support, 0.)
                            draws.append(source_reconstruction(coefficients, initial, layout, bi, ai).detach())
                            if score_function_training:
                                raw = self.reconstruct(coefficients, initial, layout, native_operator)
                                if field_normalization is not None:
                                    raw = field_normalization.inverse(raw, physical_support).to(source_initial.dtype)
                                raw_draws.append(raw.detach())
                                draw_probabilities.append(path_log_probability)
                        if teacher_trajectory is None:
                            diagnostics_rows.append(diag)
                case_rows.append(diag["operator_splitting_case_activation"].reshape(-1))
                action_results.append(torch.stack(draws))
                if score_function_training:
                    action_probabilities.append(torch.stack(draw_probabilities))
                    action_raw.append(torch.stack(raw_draws))
            if batched_evaluation:
                continue
            results.append(torch.stack(action_results, dim=1))
            if score_function_training:
                path_probabilities.append(torch.stack(action_probabilities, dim=1))
                raw_results.append(torch.stack(action_raw, dim=1))
            if action_predictions:
                training_predictions.append(torch.stack(action_predictions))
        if batched_evaluation:
            flat_draws, diagnostics_rows, case_rows = self._sample_seeded_evaluation(
                inference_cases, sample_count, steps, sample_ids, regimen_ids, draw_offset,
                native_operator, adjacency, coordinates, source_reconstruction)
            samples = flat_draws.reshape(b, a, sample_count, t, r, c, s).permute(2, 0, 1, 3, 4, 5, 6)
        else:
            samples = torch.stack(results, dim=1)  # draw, sample, action, time, region, compartment, species
        mean, std = samples.mean(0), samples.std(0, unbiased=False)
        mean = torch.cat([source_initial.unsqueeze(2),mean[:,:,1:]],dim=2)
        std = torch.cat([torch.zeros_like(source_initial).unsqueeze(2),std[:,:,1:]],dim=2)
        ce = torch.stack(ces).mean() if ces else mean.new_zeros(())
        reconstruction_loss = torch.stack(recs).mean() if recs else mean.new_zeros(())
        auxiliary_per_action = (
            self.loss_weights["token_substitution_cross_entropy"] * torch.stack(ces).reshape(b, a)
            + self.loss_weights["inverse_residual_reconstruction"] * torch.stack(recs).reshape(b, a)
        ) if ces else mean.new_zeros(b, a)
        auxiliary = auxiliary_per_action.mean()
        if field_support is None:
            spread = std[:, :, -1].mean((-1, -2, -3))
        else:
            terminal_support = field_support[:, :, -1]
            spread = (std[:, :, -1]*terminal_support).sum((-1, -2, -3)) / terminal_support.sum((-1, -2, -3))
        stability = torch.exp(-self.ranking_stability_scale * spread)
        diagnostics = {}
        for key in diagnostics_rows[0]:
            if key == "operator_splitting_identity":
                if any(row[key] != diagnostics_rows[0][key] for row in diagnostics_rows):
                    raise ValueError("Conflicting executed expert identities across diffusion calls.")
                diagnostics[key] = diagnostics_rows[0][key]
            else:
                diagnostics[key] = torch.stack([row[key] for row in diagnostics_rows]).mean(0)
        diagnostics["operator_splitting_case_activation"] = torch.stack(case_rows).reshape(b, a, -1)
        diagnostics.update({
            "mddm_discrete_codebook_size": mean.new_tensor(self.codebook_size),
            "mddm_sample_count": mean.new_tensor(sample_count), "mddm_denoising_step_count": mean.new_tensor(steps),
            "mddm_sample_chunk_size": mean.new_tensor(self.sample_chunk_size),
            "mddm_denoising_token_chunk_size": mean.new_tensor(self.denoising_token_chunk_size),
            "mddm_initial_mask_probability": mean.new_tensor(1), "mddm_final_mask_probability": mean.new_tensor(0),
            "mddm_teacher_loss_evaluated": mean.new_tensor(teacher_trajectory is not None),
            "mddm_rollout_auxiliary_loss": auxiliary,
        })
        field_diagnostics = {"teacher_reference_available": mean.new_full((b,a),teacher_trajectory is not None)}
        reference = teacher_trajectory if teacher_trajectory is not None else mean
        if field_normalization is not None:
            field_diagnostics.update(field_normalization.excursions(reference, field_support))
        if coefficient_rows:
            field_diagnostics.update({key:torch.stack([row[key] for row in coefficient_rows]).reshape(b,a,-1)
                                      for key in coefficient_rows[0]})
            field_diagnostics["normalized_reconstruction_mse"] = torch.stack(normalized_errors).reshape(b,a)
        if teacher_trajectory is not None:
            diagnostics["mddm_token_substitution_cross_entropy"] = ce
            diagnostics["mddm_inverse_reconstruction_error"] = reconstruction_loss
        output = {"trajectory_mean": mean, "trajectory_std": std,
                  "trajectory_quantiles": torch.quantile(samples, mean.new_tensor(self.quantile_levels), dim=0),
                  "ranking_stability": stability, "rollout_auxiliary_loss": auxiliary,
                  "rollout_auxiliary_loss_per_action": auxiliary_per_action,
                  "sampler_diagnostics": diagnostics, "field_normalization_diagnostics": field_diagnostics}
        if learned_teacher is not None:
            output["learned_teacher_kl_per_action"] = torch.stack(learned_kls).reshape(b, a)
            output["learned_teacher_kl"] = output["learned_teacher_kl_per_action"].mean()
        if grid is not None:
            readout = observable_readout(mean.flatten(0, 1), std.flatten(0, 1), grid, field_support.flatten(0, 1))
            for key in ("predictive_mean", "predictive_scale", "feature_mask", "time_mask"):
                readout[key] = readout[key].reshape(b, a, *readout[key].shape[1:])
            output.update(quantity_support=field_support, observable_prediction=readout)
        if self.action_policy_name == "scalar_exposure":
            output["action_conditioning"] = {
                "policy": self.action_policy_name, "definition": self.scalar_exposure_definition,
                "interval_hours": conditioning["scalar_exposure_interval"].detach().cpu().tolist(),
                "source_site_auc": conditioning["scalar_exposure_auc"].detach().cpu().tolist(),
            }
        if self.action_policy_name == "scalar_exposure" and "scalar_exposure_site_support" in conditioning:
            output["action_conditioning"].update(
                source_site_support=conditioning["scalar_exposure_site_support"].bool().detach().cpu().tolist(),
                excluded_site_meaning="outside_declared_target_program_not_concentration_absence")
        if training_predictions:
            output["training_reconstruction"] = torch.stack(training_predictions)
        if score_function_training:
            output["sample_log_probabilities"] = torch.stack(path_probabilities, dim=1)
            output["unprojected_sample_trajectories"] = torch.stack(raw_results, dim=1)
        if (self.return_sample_trajectories if return_samples is None else return_samples):
            output["sample_trajectories"] = samples
        return output



def validate_native_action_checkpoint(model: nn.Module, state_dict: Mapping) -> None:
    """Check native action and expert identities before current/best/completed reuse."""
    if not isinstance(state_dict, Mapping):
        raise ValueError("Native checkpoint model state must be a mapping.")
    for name, module in model.named_modules():
        if isinstance(module, MechanismOperatorExpertBank):
            module.validate_checkpoint_composition(state_dict, name + "." if name else "")
        if isinstance(module, QuantizedMddmRolloutOperator):
            module.validate_checkpoint_action_identity(state_dict, name + "." if name else "")
