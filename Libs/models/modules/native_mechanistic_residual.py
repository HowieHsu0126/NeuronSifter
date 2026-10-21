"""Hybrid residual and observation-only updating, adapted from arXiv2607.13877v1.

The unpooled source-graph network is a declared native adaptation of the paper's
3D U-Net. Uniform normalized residual MSE replaces its tumor-specific weighting.
Predicted independent-state corrections enter the next frozen mechanical step.
Online H3 uses admitted partial readouts, not the paper's full-state targets.
Postzero control selection is a component; full MPC qualification remains open.
"""
from __future__ import annotations

from Libs.utils.data.native_actuation import actuation_payload

from collections import deque
import copy
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import nn

from Libs.models.modules.native_frozen_rhs_solver import NativeFrozenRHSSolver, NativeRHSContinuation
from Libs.models.modules.world_state import FACTOR_NAMES, _normalized_log_weights
from Libs.utils.data.native_diffusion_data import joint_particle_contexts
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.data.joint_observation import retained_log_likelihood
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.model.native_arrived_observation import (
    native_acquisition_mask, NativeAcquisition, NativeArrivedObservation, NativeCNSReadout,
    admitted_window_binding, filter_native_observation, predict_native_readout,
    project_cns_readout, arrived_payload_fingerprint,
)

from Libs.utils.model.native_conditional_readout import (
    AdaptiveReadoutContext, AdaptiveReadoutQuery, AdaptiveReadoutRecord, AdaptiveReadoutReference,
    adaptive_readout_window, filter_adaptive_native_observation, export_adaptive_readout_history,
    normalized_adaptive_initial_weights,
)

from Libs.utils.model.native_future_controls import (
    execution_program, execution_record_identity, conditioning_identity, validate_execution_state,
    execution_constants_identity, model_control_identity,
)


def native_residual_online_policy(config: dict) -> dict:
    """Parse the prospective online component without changing offline settings."""
    policy = config.get("native_mechanistic_residual_online")
    schema = config["schemas"]["native_mechanistic_residual_online"]
    if not isinstance(policy, dict) or set(schema["required_fields"])-set(policy):
        raise ValueError("Native residual online policy is missing required fields")
    fixed = {
        "schema_version": "native_mechanistic_residual_online.v1",
        "window_selection": "most_recent_consecutive_original_transitions",
        "objective": "joint_retained_gaussian_window_nll_sum",
        "prior": "frozen_window_start_log_weights",
        "update_scope": "episode_owned_residual_parameters_only",
        "network_mode": "eval_with_gradients_dropout_disabled",
        "failure_policy": "retain_partial_updates_and_fail_episode_no_retry",
    }
    if any(policy.get(key) != value for key, value in fixed.items()):
        raise ValueError("Native residual online observation/state boundary differs")
    if (policy["horizon_transitions"] != 3 or type(policy["horizon_transitions"]) is not int
            or any(type(policy[key]) is not int or policy[key] < 1 for key in
                   ("warmup_transitions", "buffer_transitions", "updates_per_observation"))
            or not 3 <= policy["warmup_transitions"] <= policy["buffer_transitions"]):
        raise ValueError("Native residual online buffer/warmup/update budget is invalid")
    optimizer = policy["optimizer"]
    if (not isinstance(optimizer, dict)
            or set(optimizer) != {"name", "learning_rate", "betas", "epsilon", "weight_decay"}
            or optimizer["name"] != "Adam"
            or any(type(optimizer[key]) not in (int, float) or not math.isfinite(optimizer[key])
                   or optimizer[key] <= 0 for key in ("learning_rate", "epsilon"))
            or not isinstance(optimizer["betas"], (list, tuple)) or len(optimizer["betas"]) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1
                   for value in optimizer["betas"])
            or type(optimizer["weight_decay"]) not in (int, float) or optimizer["weight_decay"] != 0):
        raise ValueError("Native residual online optimizer settings are invalid")
    return copy.deepcopy(policy)


class NativeResidualGraphBlock(nn.Module):
    """Two source-edge neighborhood convolutions without spatial voxelization."""
    def __init__(self, incoming: int, outgoing: int, settings: dict):
        super().__init__()
        self.first = nn.Linear(2*incoming, outgoing)
        self.second = nn.Linear(2*outgoing, outgoing)
        self.skip = nn.Linear(incoming, outgoing) if incoming != outgoing else nn.Identity()
        self.norm1 = nn.InstanceNorm1d(outgoing, affine=True, eps=settings["normalization_epsilon"])
        self.norm2 = nn.InstanceNorm1d(outgoing, affine=True, eps=settings["normalization_epsilon"])
        self.activation = nn.LeakyReLU(settings["negative_slope"])

    def forward(self, values, adjacency):
        hidden = self.first(torch.cat((values, adjacency@values), -1))
        hidden = self.activation(self.norm1(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.second(torch.cat((hidden, adjacency@hidden), -1))
        hidden = self.norm2(hidden.transpose(1, 2)).transpose(1, 2)
        return self.activation(hidden+self.skip(values))


@dataclass
class NativeResidualContinuation:
    """Hybrid carry with separately validated actual or hypothetical controls."""

    rhs: NativeRHSContinuation
    program: torch.Tensor
    static: torch.Tensor
    normalization_id: str
    owner: NativeMechanisticResidual
    _conditioning_identity: str | None = None
    _forecast_control: object | None = None
    _forecast_executed_uid: str | None = None
    _forecast_binding: str | None = None
    _accepted_readout_snapshots: tuple = ()
    _initial_readout_log_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class _AcceptedResidualReadout:
    observation_uid: str
    query_index: int
    filtering_revision: int
    mean: tuple[tuple[float, ...], ...]
    predictor_update_count: int
    payload_fingerprint: str
    arrived: NativeArrivedObservation
    public_control: object




@dataclass(frozen=True)
class AdaptiveResidualReadoutWindow:
    """Opt-in prospective H3 input; no live policy or calibration authorization.

    Records retain actual accepted pre-update means. The initial tuple weights
    precede all records. The H3 prior is derived from these weights and the
    complete frozen discrepancy history using the SAME adaptive joint law.
    """

    reference: AdaptiveReadoutReference
    history: tuple[AdaptiveReadoutRecord, ...]
    accepted: tuple[AdaptiveReadoutRecord, ...]
    initial_log_weights: tuple[float, ...]
    trial_predictor_binding: str

    def validate(self, solver, start_continuation, post_continuation, arrivals, normalization) -> None:
        start, post_filter = start_continuation.rhs, post_continuation.rhs
        if type(self.reference) is not AdaptiveReadoutReference:
            raise ValueError("Adaptive H3 requires its prospective readout reference")
        self.reference.validate()
        fixed = self.reference.fixed
        context = solver.filter_context(start)
        if (fixed.normalization_id != normalization.identity
                or fixed.readout != post_filter._filter_readout
                or fixed.support_uid != context["support_uid"]
                or fixed.origin_posterior_uid != context["origin_posterior_uid"]
                or fixed.sample_uids != tuple(context["sample_uids"])
                or fixed.particle_uids != tuple(tuple(row) for row in context["particle_uids"])
                or fixed.time_hours != tuple(start._times)
                or fixed.source_times != tuple(float(t) for t in start._guide.clock)
                or fixed.source_time_unit != context["source_time_unit"]
                or fixed.initial_filtering_revision != 0
                or tuple(normalization.scale.detach().cpu().tolist()) != fixed.inverse_normalizer_scales):
            raise ValueError("Adaptive H3 source, tuple, readout, absolute clock or TRAIN scale changed")
        if (type(self.history) is not tuple or type(self.accepted) is not tuple
                or len(self.accepted) != 3
                or len(self.history) != start.filter_revision
                or len(self.history) + 3 != post_filter.filter_revision):
            raise ValueError("Adaptive H3 requires complete immutable history and initial32-tuple weights")
        weights = normalized_adaptive_initial_weights(self.initial_log_weights,
            error_message="Adaptive H3 requires its normalized immutable initial32-tuple prior")
        if (self.initial_log_weights != start_continuation._initial_readout_log_weights
                or self.initial_log_weights != post_continuation._initial_readout_log_weights):
            raise ValueError("Adaptive H3 initial tuple weights differ from actual continuation authority")
        binding = json.loads(self.trial_predictor_binding)
        json.dumps(binding, allow_nan=False)
        if (not isinstance(binding, dict) or binding.get("normalization_id") != fixed.normalization_id
                or not isinstance(binding.get("checkpoint_sha256"), str) or not binding["checkpoint_sha256"]):
            raise ValueError("Adaptive H3 requires an explicit current trial predictor binding")
        all_records = self.history + self.accepted
        initial_count = len(post_filter.filter_observation_uids) - post_filter.filter_revision
        expected_uids = post_filter.filter_observation_uids[initial_count:]
        if (any(type(row) is not AdaptiveReadoutRecord or row.reference != self.reference for row in all_records)
                or tuple(row.observation_uid for row in all_records) != expected_uids
                or tuple(row.filtering_revision for row in all_records) != tuple(range(len(all_records)))
                or tuple(row.query_index for row in all_records) != tuple(range(1, post_filter.query_index + 1))):
            raise ValueError("Adaptive H3 records differ from the complete admitted original-query history")
        snapshots = post_continuation._accepted_readout_snapshots
        if (type(snapshots) is not tuple or len(snapshots) != len(all_records)
                or start_continuation._accepted_readout_snapshots != snapshots[:len(self.history)]):
            raise ValueError("Adaptive H3 has no immutable continuation-owned pre-update history")
        admitted = post_filter._filter_binding.get("native_admitted_readout_payload_sha256", {})
        for row, snapshot in zip(all_records, snapshots):
            if type(snapshot) is not _AcceptedResidualReadout:
                raise ValueError("Adaptive H3 requires actual immutable pre-update snapshot records")
            binding = json.loads(row.context.predictor_binding)
            selected_mean = tuple(tuple(values[i] for i in row.coordinates) for values in snapshot.mean)
            if (type(snapshot) is not _AcceptedResidualReadout
                    or (snapshot.observation_uid, snapshot.query_index, snapshot.filtering_revision)
                        != (row.observation_uid, row.query_index, row.filtering_revision)
                    or selected_mean != row.mean
                    or binding != dict(json.loads(fixed.predictor_binding),
                                       residual_optimizer_updates=snapshot.predictor_update_count)
                    or admitted.get(row.observation_uid) != snapshot.payload_fingerprint):
                raise ValueError("Adaptive H3 changed actual immutable pre-update means or admission binding")
            arrived = snapshot.arrived
            mean = torch.tensor(snapshot.mean, dtype=torch.float64, device=arrived.observation.values.device)
            if (arrived_payload_fingerprint(arrived) != snapshot.payload_fingerprint
                    or AdaptiveReadoutRecord.capture(self.reference, row.context, arrived, mean) != row):
                raise ValueError("Adaptive H3 changed actual immutable historical packet values")
        if post_filter._public_control is None:
            switches = ()
        else:
            if tuple(post_filter._execution_library) != self.reference.original_actions:
                raise ValueError("Adaptive H3 original action library changed")
            switches = tuple(row["request"] for row in post_filter._execution_history)
        for row in all_records:
            actual_switches = tuple(request for request in switches if request.prefix.query_index < row.query_index)
            row.context.validate(self.reference, row.query_index)
            if row.context.switches != actual_switches:
                raise ValueError("Adaptive H3 cannot rewrite actual acquisition-time switches")
        for row, arrived in zip(self.accepted, arrivals):
            # The retained record carries the actual pre-update mean. Recapture
            # only compares public packet values and metadata; it never refits it.
            mean = torch.zeros(32, 6, dtype=torch.float64, device=arrived.observation.values.device)
            mean[:, list(row.coordinates)] = mean.new_tensor(row.mean)
            if AdaptiveReadoutRecord.capture(self.reference, row.context, arrived, mean) != row:
                raise ValueError("Adaptive H3 accepted packet values differ from the immutable snapshot")
        # Validate frozen history numerics and cross-record lineage before any
        # residual execution. A no-gradient trial uses the last accepted record.
        row = self.accepted[-1]
        mean = weights.new_zeros(32, 6)
        mean[:, list(row.coordinates)] = mean.new_tensor(row.mean)
        observed = tuple(i in row.coordinates for i in range(6))
        adaptive_readout_window(self.reference, all_records[:-1], (AdaptiveReadoutQuery(
            row.context, row.query_index, row.filtering_revision, observed, mean),))


class NativeMechanisticResidual(nn.Module):
    prediction_kind = "offline_one_step_mechanistic_residual_conditional_mean"

    def __init__(self, settings: dict, inputs, config: dict):
        super().__init__()
        self.config, self.settings = config, settings
        if (settings["architecture"] != "unpooled_source_graph_channel_unet"
                or settings["loss"] != "uniform_normalized_one_step_residual_mse"
                or settings["state_correction"] != "independent_CNS_and_neural_only"
                or settings["online_adaptation"] != "unbound"
                or settings["initialization"] != "zero_residual_head"):
            raise ValueError("Unbound native mechanistic-residual adaptation")
        widths = settings["encoder_widths"]
        if (len(widths) != 3 or any(type(v) is not int or v < 1 for v in widths)
                or type(settings["bottleneck_width"]) is not int or settings["bottleneck_width"] < 1
                or not 0 <= settings["dropout"] < 1
                or not math.isfinite(settings["initial_residual_scale"])):
            raise ValueError("Invalid graph residual network widths or scalar settings")
        self.field_shape = tuple(inputs.initial.shape[-3:])
        nodes, quantities = math.prod(self.field_shape[:2]), self.field_shape[-1]
        self.static_blocks = [inputs.blocks[key] for key in (*FACTOR_NAMES, "named_parameters_and_presence", "physical_domain")]
        static_width = sum(right-left for left, right in self.static_blocks)
        self.context_projection = nn.Linear(static_width, widths[0])
        self.scales = config["native_controlled_comparators"]["conditioning"]["program_reference_scales"]
        features = 2*quantities+2*(3+len(FORCING_ARRAY_NAMES))+widths[0]
        self.encoders = nn.ModuleList([NativeResidualGraphBlock(left, right, settings)
            for left, right in zip((features, *widths[:-1]), widths)])
        bottleneck = settings["bottleneck_width"]
        self.bottleneck = NativeResidualGraphBlock(widths[-1], bottleneck, settings)
        self.dropout = nn.Dropout(settings["dropout"])
        self.decoders = nn.ModuleList([NativeResidualGraphBlock(left+skip, skip, settings)
            for left, skip in zip((bottleneck, *reversed(widths[1:])), reversed(widths))])
        self.output = nn.Linear(widths[0], quantities)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(settings["initial_residual_scale"])))
        adjacency = torch.eye(nodes)
        for edge in config["controlled_cns_source"]["edges"]:
            adjacency[edge["left"], edge["right"]] = 1
            adjacency[edge["right"], edge["left"]] = 1
        self.register_buffer("adjacency", adjacency/adjacency.sum(-1, keepdim=True))
        self.solver = NativeFrozenRHSSolver(config,
            config["native_controlled_comparators"]["references"]["same_compiler_solver"]["hyperparameters"])

    def _prepare(self, inputs, normalization):
        if tuple(inputs.initial.shape[-3:]) != self.field_shape or inputs.normalization_id != normalization.identity:
            raise ValueError("Hybrid residual requires the original complete field and shared normalization")
        if inputs.group_shape[:2] != (1, 32):
            raise ValueError("Hybrid residual requires all32 complete native tuples")
        contexts = joint_particle_contexts(inputs.posterior, inputs.programs, inputs.regimen_uids)
        n = math.prod(inputs.group_shape)
        def stack(name):
            values = torch.stack([row[name] for row in contexts], 1)
            return values.reshape(n, values.shape[3], -1, values.shape[-1])
        program = torch.cat([stack(name)/self.scales[name] for name in
            ("occupancy", "support", "physical_time", *FORCING_ARRAY_NAMES)], -1).to(inputs.condition)
        static = torch.cat([inputs.condition[:, left:right] for left, right in self.static_blocks], -1)
        state = self.solver.initialize_continuation(inputs, normalization)
        return state, program, static

    def normalized_residual(self, fields, mask, program, static, normalization):
        n, nodes = program.shape[:2]
        values = normalization.normalize(fields.reshape(n, *self.field_shape), mask.reshape(n, *self.field_shape))
        context = self.context_projection(static)[:, None].expand(-1, nodes, -1)
        hidden = torch.cat((values.reshape(n, nodes, -1).to(context),
            mask.reshape(n, nodes, -1).to(context), program, context), -1)
        skips = []
        for encoder in self.encoders:
            hidden = encoder(hidden, self.adjacency)
            skips.append(hidden)
        hidden = self.dropout(self.bottleneck(hidden, self.adjacency))
        for decoder, skip in zip(self.decoders, reversed(skips)):
            hidden = decoder(torch.cat((hidden, skip), -1), self.adjacency)
        return (self.residual_scale*self.output(hidden)).reshape_as(fields)

    def _interval_program(self, program, state, next_index):
        # Both boundaries are planned public program values. This retains a
        # chemical action starting at t0 even though its pre-dose occupancy is
        # zero; it never substitutes true future response/exposure for a plan.
        return torch.cat((program[:, next_index-1], program[:, next_index]), -1)

    def one_step_residual_loss(self, inputs, target, mask, action_weights, normalization):
        if inputs.posterior.support.role != "train":
            raise ValueError("Offline residual targets require TRAIN-role current/next fields")
        state, program, static = self._prepare(inputs, normalization)
        b, j, a = inputs.group_shape
        expected = (b, a, program.shape[1], *self.field_shape)
        if target.shape != expected or mask.shape != expected or action_weights.shape != (b, a):
            raise ValueError("Residual teacher trajectory/paired action axes differ")
        weights = (inputs.weights[:, :, None]*action_weights[:, None]).reshape(-1).double()
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Residual training requires unchanged positive q/QC/precision weights")
        scale = normalization.scale.to(target).double()
        values = []
        for index in range(1, program.shape[1]):
            current = target[:, None, :, index-1].expand(b, j, a, *self.field_shape)
            observed = mask[:, None, :, index-1].expand_as(current)
            current = self.solver.teacher_force_independent_state(state, current, observed=observed)
            current_mask = state._validity[:, :, :, index-1]
            correction = self.normalized_residual(current, current_mask,
                self._interval_program(program, state, index), static, normalization)
            mechanical = self.solver.advance_continuation(state, query_count=index+1)["particle_fields"][:, :, :, -1]
            allowed = self.solver.independent_state_mask(state)
            declared = mask[:, None, :, index].expand_as(allowed)
            valid = allowed & declared
            expected_residual = (target[:, None, :, index].expand_as(mechanical)-mechanical)/scale
            error = torch.where(valid, (correction.double()-expected_residual).square(), 0.)
            counts = valid.reshape(b*j*a, -1).sum(-1)
            if (counts == 0).any():
                raise ValueError("One-step residual has no supported independent-state target")
            per_tuple = error.reshape(b*j*a, -1).sum(-1)/counts
            values.append((per_tuple*weights).sum()/weights.sum())
        return torch.stack(values).mean()

    @torch.no_grad()
    def initialize_continuation(self, inputs, normalization) -> NativeResidualContinuation:
        """Bind fixed public inputs once; subsequent calls retain the physical carry."""
        state, program, static = self._prepare(inputs, normalization)
        return NativeResidualContinuation(state, program.detach().clone(), static.detach().clone(),
                                          normalization.identity, self,
            _initial_readout_log_weights=tuple(self.solver.current_log_weights(state)[0].tolist()))

    def advance_continuation(self, continuation: NativeResidualContinuation, normalization, *,
                             query_count: int) -> tuple[NativeResidualContinuation, torch.Tensor]:
        """Return new fields and a private carry, with ambient residual-state gradients.

        The original action columns keep their separate already-executed histories.
        No action is rebound, no observation loss is selected, and no optimizer runs.
        """
        if (not isinstance(continuation, NativeResidualContinuation)
                or continuation.owner is not self
                or continuation.normalization_id != normalization.identity):
            raise ValueError("Hybrid continuation owner or fixed normalization changed")
        state, program, static = continuation.rhs, continuation.program, continuation.static
        if state._planning_origin is not None and state._public_control is None:
            raise ValueError("Future-control branches require recompiled residual conditioning and cross-switch H3")
        if state._public_control is not None:
            validate_execution_state(state)
            if continuation._conditioning_identity != conditioning_identity(program,static):
                raise ValueError("Executed residual conditioning version changed after its actual binding")
        return self._advance_residual_states(continuation, normalization, query_count=query_count)

    def _advance_residual_states(self, continuation, normalization, *, query_count):
        """Shared propagation after the actual or hypothetical entry validates its binding."""
        state, program, static = continuation.rhs, continuation.program, continuation.static
        self.solver._validate_continuation(state)
        count = query_count
        if type(count) is not int or not state.query_index+1 < count <= program.shape[1]:
            raise ValueError("Hybrid continuation requires a later original query within its horizon")
        current = self.solver.current_particle_fields(state)
        predictions = [current] if state.query_index == 0 else []
        for index in range(state.query_index+1, count):
            if time.monotonic() >= getattr(self, "_execution_deadline", math.inf):
                raise TimeoutError("Native residual evaluation reached its cumulative wall cap")
            current_mask = state._validity[:, :, :, index-1]
            delta = self.normalized_residual(current, current_mask,
                self._interval_program(program, state, index), static, normalization)
            state, _ = self.solver.advance_continuation_functional(state, query_count=index+1)
            allowed = self.solver.independent_state_mask(state)
            delta = torch.where(allowed, delta.double()*normalization.scale.to(delta).double(), 0.)
            state, current = self.solver.apply_state_residual_functional(state, delta)
            predictions.append(current)
        value = torch.stack(predictions, 3).reshape(state._guide.n, len(predictions), *self.field_shape)
        return replace(continuation, rhs=state), value

    def _forecast_identity(self, continuation):
        return digest({"public": actuation_payload(continuation._forecast_control),
            "executed_model_uid": continuation._forecast_executed_uid,
            "controls": model_control_identity(continuation.rhs),
            "constants": execution_constants_identity(continuation.rhs),
            "conditioning": conditioning_identity(continuation.program, continuation.static)})

    def validate_forecast_continuation(self, continuation):
        if (continuation.owner is not self or continuation._forecast_control is None
                or continuation.rhs._public_control is not None or continuation.rhs._execution_model_uid is not None
                or continuation.rhs._planning_origin is None
                or continuation._forecast_binding != self._forecast_identity(continuation)):
            raise ValueError("Hypothetical residual control/conditioning binding changed")
        self.solver._validate_continuation(continuation.rhs)

    @torch.no_grad()
    def model_only_continuation(self, continuation):
        """Copy current model carry without conferring actual execution on its forecasts."""
        if continuation.owner is not self or continuation.rhs._public_control is None:
            raise ValueError("A model-only root requires the current actual public origin")
        validate_execution_state(continuation.rhs)
        if continuation._conditioning_identity != conditioning_identity(continuation.program, continuation.static):
            raise ValueError("Model-only root conditioning changed")
        state = self.solver.fork_continuation(continuation.rhs)
        if state._planning_origin is None:
            state._planning_origin = {"model_regimen_uids": list(state._filter_binding["regimen_uids"]),
                "source_candidate_regimen_uids": list(state._filter_binding["regimen_uids"]),
                "initial_prespecified_forecast": True}
        public, uid = state._public_control, state._execution_model_uid
        state._public_control = state._execution_model_uid = None
        result = replace(continuation, rhs=state, _forecast_control=public, _forecast_executed_uid=uid)
        result._forecast_binding = self._forecast_identity(result)
        return result

    @torch.no_grad()
    def prepare_initial_forecast(self, continuation, inputs, normalization, library, action, readout):
        """Compile a public initial alternative from the same pre-event origin."""
        if continuation.owner is not self or continuation.normalization_id != normalization.identity:
            raise ValueError("Initial forecast owner or normalization differs")
        state = self.solver.prepare_initial_control(continuation.rhs, inputs, library, action, readout)
        index = inputs.regimen_uids.index(action.regimen_uid)
        program = continuation.program.reshape(1, 32, len(library), *continuation.program.shape[1:])[:, :, index].reshape(
            32, *continuation.program.shape[1:]).detach().clone()
        static = continuation.static.reshape(1, 32, len(library), -1)[:, :, index].reshape(32, -1).detach().clone()
        result = NativeResidualContinuation(state, program, static, normalization.identity, self,
            conditioning_identity(program, static), action, action.regimen_uid,
            _accepted_readout_snapshots=continuation._accepted_readout_snapshots,
            _initial_readout_log_weights=continuation._initial_readout_log_weights)
        result._forecast_binding = self._forecast_identity(result)
        self.validate_forecast_continuation(result)
        return result

    @torch.no_grad()
    def prepare_forecast_control(self, continuation, inputs, normalization, branch, request):
        """Bind an original-library suffix for model-only forecasting, including virtual prefixes."""
        if (continuation.owner is not self or continuation.normalization_id != normalization.identity
                or continuation._conditioning_identity != conditioning_identity(continuation.program, continuation.static)):
            raise ValueError("Residual forecast requires its unchanged conditioning origin")
        parent = None
        if continuation.rhs._public_control is None:
            self.validate_forecast_continuation(continuation)
            parent = (continuation._forecast_control, continuation._forecast_executed_uid)
        selected, public = self.solver.prepare_public_execution(
            continuation.rhs, branch, inputs, request, hypothetical_parent=parent)
        candidate = inputs.regimen_uids.index(request.actuation.regimen_uid)
        program = execution_program(continuation, selected, public, candidate, inputs, self.config)
        uid = continuation.rhs._execution_model_uid if parent is None else parent[1]
        index = continuation.rhs._filter_binding["regimen_uids"].index(uid)
        static = continuation.static.reshape(1, 32, continuation.rhs._guide.a, -1)[:, :, index].reshape(32, -1).detach().clone()
        result = NativeResidualContinuation(selected, program, static, normalization.identity, self,
            conditioning_identity(program, static), public, selected._filter_binding["regimen_uids"][0],
            _accepted_readout_snapshots=continuation._accepted_readout_snapshots,
            _initial_readout_log_weights=continuation._initial_readout_log_weights)
        result._forecast_binding = self._forecast_identity(result)
        return result

    @torch.no_grad()
    def advance_forecast(self, continuation, normalization, *, query_count):
        self.validate_forecast_continuation(continuation)
        result, fields = self._advance_residual_states(continuation, normalization, query_count=query_count)
        self.validate_forecast_continuation(result)
        return result, fields

    @torch.no_grad()
    def _reuse_forecast_result(self, target, origin, result, *, parameter_versions):
        """Copy an identical sibling's numeric result under its own public binding.

        This is local no-gradient planning reuse. Every control, tuple constant,
        residual input, initial carry and floating-point interval must match.
        Public identities remain the target's; no observation or execution is admitted.
        """
        if self.training or parameter_versions != tuple(value._version for value in (*self.parameters(), *self.buffers())):
            raise ValueError("Identical forecast reuse requires unchanged evaluation model parameters")
        for item in (target, origin, result):
            self.validate_forecast_continuation(item)
        left, right, advanced = target.rhs, origin.rhs, result.rhs
        if (left._work_ledger is not right._work_ledger or advanced._work_ledger is not right._work_ledger
                or left.query_index != right.query_index or left.interval_index != right.interval_index
                or left._intervals != right._intervals or advanced._intervals != right._intervals
                or left._guide.n != right._guide.n or left._guide.n != advanced._guide.n
                or left.filter_revision != right.filter_revision
                or left.filter_observation_uids != right.filter_observation_uids
                or left._filter_readout != right._filter_readout
                or target.normalization_id != origin.normalization_id or result.normalization_id != origin.normalization_id
                or execution_constants_identity(left) != execution_constants_identity(right)
                or model_control_identity(left) != model_control_identity(right)
                or model_control_identity(advanced) != model_control_identity(right)):
            raise ValueError("Identical sibling forecast changed controls, tuples, clock or numerical partition")
        for name in ("program", "static"):
            if not torch.equal(getattr(target, name), getattr(origin, name)) or not torch.equal(
                    getattr(result, name), getattr(origin, name)):
                raise ValueError("Identical sibling forecast changed residual conditioning")
        for name in ("cns_state", "auxiliary_state", "event_times", "event_doses", "applied_events",
                     "_event_doses_mg_per_kg", "_initial", "_validity", "_weights", "_log_weights"):
            a, b = getattr(left, name), getattr(right, name)
            if (a is None) != (b is None) or a is not None and not torch.equal(a, b):
                raise ValueError("Identical sibling forecast changed its complete numerical origin")
        if advanced.query_index <= right.query_index:
            raise ValueError("Identical sibling reuse requires an advanced complete result")
        state = self.solver.fork_continuation(left)
        for name in ("cns_state", "auxiliary_state", "applied_events"):
            value = getattr(advanced, name)
            setattr(state, name, None if value is None else value.detach().clone())
        state.query_index, state.interval_index = advanced.query_index, advanced.interval_index
        # Solver validation binds this counter to the complete numerical path.
        # Actual unique calls are charged only by the shared work ledger; reuse
        # contributes zero work and does not export a new solver execution receipt.
        state.rhs_python_calls += advanced.rhs_python_calls-right.rhs_python_calls
        reused = replace(target, rhs=state)
        self.validate_forecast_continuation(reused)
        return reused

    @torch.no_grad()
    def forecast_public_control(self, continuation, inputs, normalization, branch, request):
        """Preserve full-tail forecasting and separation from actual activation."""
        hypothetical = self.prepare_forecast_control(continuation, inputs, normalization, branch, request)
        forecast, fields = self.advance_forecast(hypothetical, normalization, query_count=hypothetical.program.shape[1])
        return forecast, fields, hypothetical._forecast_control

    def partial_observation_window_loss(self, start: NativeResidualContinuation,
            post_filter: NativeResidualContinuation, arrivals: tuple[NativeArrivedObservation, ...],
            normalization, *, adaptive_window: AdaptiveResidualReadoutWindow | None = None
            ) -> tuple[torch.Tensor | None, dict]:
        """Three-transition joint readout NLL; no optimizer, refiltering, or live rewind."""
        diagnostic = {"scope": "partial_observation_H3", "window_transition_count": 3,
            "objective": "joint_retained_gaussian_window_nll_sum",
            "online_calibration_qualified": False, "scientific_twin": "NOT_RUN",
            "measurement_noise_is_predictive_error_calibration": False}
        if len(arrivals) < 3:
            return None, dict(diagnostic, status="not_ready", reason="three_arrived_transitions_required")
        if len(arrivals) != 3:
            raise ValueError("H3 requires exactly three consecutive original transitions")
        if any(not isinstance(item, NativeResidualContinuation) or item.owner is not self
               or item.normalization_id != normalization.identity for item in (start, post_filter)):
            raise ValueError("H3 continuation owner or fixed normalization changed")
        constants = (start.rhs.cns_state, start.rhs.auxiliary_state, start.program, start.static,
                     start.rhs._log_weights, *normalization.buffers())
        if any(value is not None and value.requires_grad for value in constants):
            raise ValueError("H3 window start, prior and normalization must be frozen")
        binding = admitted_window_binding(self.solver, start.rhs, post_filter.rhs, arrivals)
        records=binding.pop("execution_records",())
        diagnostic.update(binding)
        if binding["status"] != "ready":
            return None, diagnostic
        if start.rhs._public_control is None:
            if (not torch.equal(start.program, post_filter.program)
                    or not torch.equal(start.static, post_filter.static)):
                raise ValueError("H3 fixed public program or static conditioning changed")
        elif any(item._conditioning_identity != conditioning_identity(item.program,item.static)
                 for item in (start,post_filter)):
            raise ValueError("H3 actual conditioning history version was modified")
        prior = self.solver.current_log_weights(start.rhs).detach()
        if prior.shape != (1, 32):
            raise ValueError("H3 requires the complete unchanged32-tuple window prior")
        if adaptive_window is not None:
            if type(adaptive_window) is not AdaptiveResidualReadoutWindow:
                raise ValueError("Adaptive H3 requires an immutable bound window")
            adaptive_window.validate(self.solver, start, post_filter, arrivals, normalization)
        cursor, likelihoods, counts, queries = start, [], [], []
        switches={record["query_index"]:record for record in records}
        for position, arrived in enumerate(arrivals):
            if cursor.rhs.query_index in switches:
                record=switches[cursor.rhs.query_index]
                rhs=self.solver.replay_execution(cursor.rhs,record)
                cursor=replace(cursor,rhs=rhs,program=record["program"],static=record["static"],
                    _conditioning_identity=conditioning_identity(record["program"],record["static"]))
            action=cursor.rhs._filter_binding["regimen_uids"].index(
                cursor.rhs._execution_model_uid or binding["executed_regimen_uid"])
            cursor, fields = self.advance_continuation(cursor, normalization,
                query_count=arrived.acquisition.query_index+1)
            guide = cursor.rhs._guide
            current = fields[:, -1].reshape(guide.b, guide.j, guide.a, *self.field_shape)
            spec = arrived.acquisition.readout
            mean = project_cns_readout(current[:, :, action], spec)
            observation = replace(arrived.observation, values=arrived.observation.values.detach())
            if adaptive_window is None:
                parameters = {"law": "gaussian", "units": spec.units, "mean": mean,
                              "variance": torch.full_like(mean, spec.standard_deviation**2)}
                likelihoods.append(retained_log_likelihood(observation, parameters, torch.zeros_like(mean), {}))
            else:
                accepted = adaptive_window.accepted[position]
                context = replace(accepted.context, predictor_binding=adaptive_window.trial_predictor_binding)
                queries.append(AdaptiveReadoutQuery(context, accepted.query_index, accepted.filtering_revision,
                    tuple(bool(value) for value in observation.observed[0].tolist()), mean[0], observation.values[0]))
            counts.append(int(observation.observed.sum()))
        # A single complete tuple explains the whole window; no per-time remixture.
        if adaptive_window is None:
            joint = torch.stack(likelihoods).sum(0)
        else:
            conditioned = adaptive_readout_window(adaptive_window.reference, adaptive_window.history, tuple(queries))
            initial = conditioned.history_log_likelihood.new_tensor(adaptive_window.initial_log_weights)
            prior = _normalized_log_weights((initial + conditioned.history_log_likelihood)[None])
            joint = conditioned.log_likelihood[None]
            diagnostic.update(objective="joint_original_action_adaptive_conditional_window_nll",
                readout_law=adaptive_window.reference.law, adaptive_scientific_qualification="UNQUALIFIED",
                discrepancy_history_authority="immutable_actual_pre_update_mean_and_value_snapshots",
                history_observation_count=len(adaptive_window.history),
                tuple_prior_authority="initial_tuple_weights_plus_same_law_frozen_history")
        loss = -torch.logsumexp(prior+joint, dim=-1).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("H3 joint observation-window NLL is nonfinite")
        diagnostic.update(status="loss_ready", particle_count=32,
            observed_counts=counts, observed_count=sum(counts),
            prior_authority="frozen_window_start_log_weights" if adaptive_window is None else
                "initial_tuple_weights_plus_same_law_frozen_history", live_filter_updated=False,
            live_carry_updated=False)
        return loss, diagnostic

    @torch.no_grad()
    def predict_native(self, inputs, normalization, *, query_count=None):
        continuation = self.initialize_continuation(inputs, normalization)
        count = continuation.program.shape[1] if query_count is None else query_count
        _, value = self.advance_continuation(continuation, normalization, query_count=count)
        return value


@dataclass(frozen=True)
class _ResidualObservationTransition:
    start: NativeResidualContinuation
    arrived: NativeArrivedObservation
    predicted_mean: tuple[tuple[float, ...], ...]
    predictor_update_count: int


class NativeResidualOnlineEpisode:
    """Advance, observe, filter once, then adapt an episode-owned residual copy.

    The provider receives only a public acquisition request. Its plant, fields
    and RNG stay outside this object. Buffer entries are the actual pre-advance
    carry and admitted measurement; updates never replace the live physical
    state or filter history. Overlapping windows reuse observations and do not
    represent independent scientific samples.
    """

    def __init__(self, model: NativeMechanisticResidual, inputs, normalization,
                 readout: NativeCNSReadout, *, executed_regimen_uid: str | None,
                 adaptive_reference: AdaptiveReadoutReference | None = None):
        self.policy = native_residual_online_policy(model.config)
        # Offline checkpoint owners and other episodes retain their own weights,
        # buffers, gradients and optimizer state. Only this copy is adapted.
        self.model = copy.deepcopy(model).eval()
        self.normalization = normalization
        self.readout, self.executed_regimen_uid = readout, executed_regimen_uid
        self._execution_inputs = copy.deepcopy(inputs)
        self.continuation = self.model.initialize_continuation(self._execution_inputs, normalization)
        context = self.model.solver.filter_context(self.continuation.rhs)
        if (executed_regimen_uid is not None and executed_regimen_uid not in context["regimen_uids"]
                or readout.source_uid != context["source_uid"]
                or readout.source_snapshot_ref != context["source_snapshot_ref"]
                or readout.role != context["role"] or readout.family_uid != context["family_uid"]):
            raise ValueError("Online residual episode readout/source/program binding differs")
        readout.validate()
        if executed_regimen_uid is not None:
            self.model.solver.bind_filter_readout(self.continuation.rhs, readout, executed_regimen_uid)
        self.optimizer = None
        if executed_regimen_uid is not None:
            self._initialize_optimizer()
        self._buffer: deque[_ResidualObservationTransition] = deque(maxlen=self.policy["buffer_transitions"])
        self.continuation = replace(self.continuation, _initial_readout_log_weights=
            tuple(self.model.solver.current_log_weights(self.continuation.rhs)[0].tolist()))
        self._adaptive_reference = adaptive_reference
        self._accepted_snapshot_authority = self.continuation._accepted_readout_snapshots
        self._adaptive_initial_log_weights = self.continuation._initial_readout_log_weights
        if adaptive_reference is not None:
            self._validate_adaptive_reference(require_execution=False)
        self.trace: list[dict] = []
        self.observations_seen = self.optimizer_updates = 0
        self.status = "ready"
        self.failed_arrival: NativeArrivedObservation | None = None
        initial_index = 0 if executed_regimen_uid is None else context["regimen_uids"].index(executed_regimen_uid)
        self._model_history = [self.model.solver.current_particle_fields(self.continuation.rhs)[0, :, initial_index].detach().clone()]

    def _initialize_optimizer(self):
        parameters = tuple(self.model.parameters())
        if not parameters or any(not value.requires_grad for value in parameters):
            raise ValueError("Online residual episode requires trainable residual parameters")
        if tuple(self.model.solver.parameters()):
            raise ValueError("Online residual updating cannot train the fixed mechanical solver")
        settings = self.policy["optimizer"]
        self.optimizer = torch.optim.Adam(parameters, lr=settings["learning_rate"],
            betas=tuple(settings["betas"]), eps=settings["epsilon"], weight_decay=settings["weight_decay"])
        self.optimizer.zero_grad(set_to_none=True)

    def install_initial_selection(self, selected, library, action, *, select_adaptive_initial_reference=False):
        """Install the selected pristine t0 origin before any private world is created."""
        if self.executed_regimen_uid is not None or self.optimizer is not None or self.observations_seen:
            raise ValueError("Initial selection is available only once before execution")
        self.model.validate_forecast_continuation(selected)
        expected = self.model.prepare_initial_forecast(self.continuation, self._execution_inputs,
            self.normalization, library, action, self.readout)
        if (selected._forecast_control != action or selected._forecast_executed_uid != action.regimen_uid
                or not torch.equal(selected.program, expected.program)
                or not torch.equal(selected.static, expected.static)):
            raise ValueError("Initial selection changed original public residual conditioning")
        candidate_reference = self._adaptive_reference
        if select_adaptive_initial_reference and candidate_reference is not None:
            candidate_reference, _, _ = self.prospective_adaptive_temporal_readout(initial_origin=selected)
        elif candidate_reference is not None and (tuple(library) != candidate_reference.original_actions
                or action.regimen_uid != candidate_reference.initial_action_uid):
            raise ValueError("Adaptive initial selection differs from its exact original action reference")
        state = self.model.solver.install_initial_control(self.continuation.rhs, selected.rhs,
            self._execution_inputs, library, action, self.readout)
        self.continuation = replace(selected, rhs=state, _forecast_control=None,
            _forecast_executed_uid=None, _forecast_binding=None)
        self._adaptive_reference = candidate_reference
        self.executed_regimen_uid = action.regimen_uid
        self._model_history = [self.model.solver.current_particle_fields(state)[0, :, 0].detach().clone()]
        self._initialize_optimizer()

    def recorded_model_history(self):
        """Return the immutable executed model prefix, never a retrospectively refitted past."""
        state = self.continuation.rhs
        if len(self._model_history) != state.query_index+1:
            raise ValueError("Recorded model history does not cover every actual original query")
        uid = state._execution_model_uid or self.executed_regimen_uid
        index = state._filter_binding["regimen_uids"].index(uid)
        current = self.model.solver.current_particle_fields(state)[0, :, index]
        if not torch.equal(current, self._model_history[-1]):
            raise ValueError("Recorded model past differs from the current executed model carry")
        return torch.stack(self._model_history, 1).detach().clone()

    def bind_execution(self, library, initial_control, observation_identity):
        """Bind public source metadata once; the provider's private world is not accepted."""
        if self.observations_seen or self.status != "ready":
            raise ValueError("Online execution identity must be fixed before any transition")
        if self._adaptive_reference is not None and (tuple(library) != self._adaptive_reference.original_actions
                or initial_control.regimen_uid != self._adaptive_reference.initial_action_uid):
            raise ValueError("Adaptive execution requires its actual original library and initial action")
        self.model.solver.bind_public_execution(self.continuation.rhs,self._execution_inputs,
            library,initial_control,observation_identity)
        self.continuation._conditioning_identity=conditioning_identity(self.continuation.program,self.continuation.static)

    def plan_controls(self, candidate_regimen_uids=None):
        if self.status not in {"ready", "active"} or self.continuation.rhs._public_control is None:
            raise ValueError("Planning execution requires a live bound online episode")
        state=self.continuation.rhs
        return self.model.solver.branch_future_controls(state,self._execution_inputs,self.normalization,
            executed_regimen_uid=state._execution_model_uid,candidate_regimen_uids=candidate_regimen_uids)

    def activate_selected_control(self, branch, request, actuator):
        """Prepare before actuator effect; no forecasted branch or UID-only activation."""
        if self.status not in {"ready", "active"}:
            raise ValueError("Failed online execution cannot activate or repeat an actuation")
        if self._adaptive_reference is not None:
            self._adaptive_history()
        previous=self.continuation
        selected,public=self.model.solver.prepare_public_execution(previous.rhs,branch,self._execution_inputs,request)
        candidate=selected._registered_regimen_uids.index(request.actuation.regimen_uid)
        program=execution_program(previous,selected,public,candidate,self._execution_inputs,self.model.config)
        parent_index=previous.rhs._filter_binding["regimen_uids"].index(previous.rhs._execution_model_uid)
        static=previous.static.reshape(1,32,previous.rhs._guide.a,-1)[:,:,parent_index].reshape(32,-1).detach().clone()
        record={"public":public,"request":request,"template":selected,"program":program,"static":static,
            "query_index":previous.rhs.query_index,"filter_revision":previous.rhs.filter_revision,
            "observation_uids":previous.rhs.filter_observation_uids,
            "parent_model_uid":previous.rhs._execution_model_uid,
            "conditioning_authority":"frozen_model_estimate_from_public_executed_controls",
            "model_uid":selected._filter_binding["regimen_uids"][0]}
        record["identity"]=execution_record_identity(record)
        prepared=self.model.solver.install_execution(selected,record,previous.rhs)
        continuation=replace(previous,rhs=prepared,program=program,static=static,
            _conditioning_identity=conditioning_identity(program,static))
        try:
            receipt=actuator(request)
            if receipt != public:
                raise ValueError("Actuator receipt differs from the jointly prepared public execution content")
            self.continuation=continuation
            self.executed_regimen_uid=public.regimen_uid
        except Exception as error:
            self.failed_activation={"request":request,"expected_public_execution":public,
                                    "error":type(error).__name__+": "+str(error)}
            # The actuator may already have committed. Retain its receipt/state,
            # do not retry or roll it back, and never continue this mixed episode.
            self.status="failed"
            raise
        return {"query_index":record["query_index"],"public_execution_uid":public.regimen_uid,
            "model_regimen_uid":record["model_uid"],"source_candidate_regimen_uid":request.actuation.regimen_uid,
            "scope":"actual_control_execution_component_not_full_MPC"}

    def _validate_adaptive_reference(self, *, require_execution: bool) -> None:
        reference, state = self._adaptive_reference, self.continuation.rhs
        if type(reference) is not AdaptiveReadoutReference:
            raise ValueError("Adaptive episode requires its explicit prospective reference")
        reference.validate()
        fixed, context = reference.fixed, self.model.solver.filter_context(state)
        if (fixed.readout != self.readout or fixed.normalization_id != self.normalization.identity
                or fixed.support_uid != context["support_uid"]
                or fixed.origin_posterior_uid != context["origin_posterior_uid"]
                or fixed.sample_uids != tuple(context["sample_uids"])
                or fixed.particle_uids != tuple(tuple(row) for row in context["particle_uids"])
                or fixed.time_hours != tuple(state._times)
                or fixed.source_times != tuple(float(t) for t in state._guide.clock)
                or fixed.source_time_unit != context["source_time_unit"]
                or fixed.initial_filtering_revision != 0
                or fixed.inverse_normalizer_scales != tuple(self.normalization.scale.detach().cpu().tolist())
                or tuple(action.regimen_uid for action in reference.original_actions) != self._execution_inputs.regimen_uids
                or self.continuation._initial_readout_log_weights != self._adaptive_initial_log_weights
                or state.filter_revision == 0 and self.executed_regimen_uid is not None
                and self.executed_regimen_uid != reference.initial_action_uid):
            raise ValueError("Adaptive episode source, original tuples/actions, clock, scale or initial prior changed")
        if require_execution:
            validate_execution_state(state)
            if (state._public_control is None or tuple(state._execution_library) != reference.original_actions):
                raise ValueError("Adaptive episode requires bound actual original-action execution")

    def _adaptive_context(self, query_index: int, public, predictor_updates: int) -> AdaptiveReadoutContext:
        reference = self._adaptive_reference
        switches = tuple(row["request"] for row in self.continuation.rhs._execution_history
                         if row["query_index"] < query_index)
        original = switches[-1].actuation if switches else next(action for action in reference.original_actions
            if action.regimen_uid == reference.initial_action_uid)
        binding = dict(json.loads(reference.fixed.predictor_binding), residual_optimizer_updates=predictor_updates)
        return AdaptiveReadoutContext(original, public, switches, json.dumps(binding, sort_keys=True, allow_nan=False))

    @property
    def adaptive_reference(self) -> AdaptiveReadoutReference | None:
        """Return the explicitly selected prospective law; legacy episodes use None."""
        return self._adaptive_reference

    def prospective_adaptive_temporal_readout(self, *, initial_origin=None):
        """Authenticate existing history and current public predictor for lookahead.

        No observation, model state or second accepted history is created here.
        Same-clock executed switches belong to the next prospective acquisition.
        """
        if initial_origin is not None:
            self._validate_adaptive_reference(require_execution=False)
            state = self.continuation.rhs
            snapshots = self.continuation._accepted_readout_snapshots
            if (self.executed_regimen_uid is not None or self.optimizer is not None
                    or self.observations_seen or self.optimizer_updates or state.query_index
                    or state.interval_index or state.rhs_python_calls or state.filter_revision
                    or state._public_control is not None or state._planning_origin is not None
                    or state._execution_history or self._buffer or self.trace or self.status != "ready"
                    or snapshots is not self._accepted_snapshot_authority or snapshots != ()
                    or self.model.training):
                raise ValueError("Adaptive initial alternatives require pristine unbound owned history")
            self.model.validate_forecast_continuation(initial_origin)
            action = initial_origin._forecast_control
            self.model.solver.install_initial_control(state, initial_origin.rhs, self._execution_inputs,
                self._adaptive_reference.original_actions, action, self.readout)
            expected = self.model.prepare_initial_forecast(self.continuation, self._execution_inputs,
                self.normalization, self._adaptive_reference.original_actions, action, self.readout)
            if (initial_origin._forecast_binding != expected._forecast_binding
                    or not torch.equal(initial_origin.program, expected.program)
                    or not torch.equal(initial_origin.static, expected.static)
                    or initial_origin._accepted_readout_snapshots is not snapshots
                    or initial_origin._initial_readout_log_weights != self._adaptive_initial_log_weights):
                raise ValueError("Adaptive initial alternative differs from its actual prepared origin")
            initial = state._log_weights.new_tensor(self._adaptive_initial_log_weights)[None]
            if not torch.equal(initial, self.model.solver.current_log_weights(state)):
                raise ValueError("Adaptive initial alternative changed the immutable initial posterior")
            reference = replace(self._adaptive_reference,
                fixed=replace(self._adaptive_reference.fixed, executed_regimen_uid=action.regimen_uid),
                initial_action_uid=action.regimen_uid)
            reference.validate()
            binding = dict(json.loads(reference.fixed.predictor_binding), residual_optimizer_updates=0)
            context = AdaptiveReadoutContext(action, action, (), json.dumps(binding, sort_keys=True, allow_nan=False))
            context.validate(reference, 0)
            return reference, (), context
        history = self._adaptive_history()
        if self.model.training:
            raise ValueError("Adaptive temporal lookahead requires the frozen current eval predictor")
        index = self.continuation.rhs.query_index
        context = self._adaptive_context(index + 1, self.continuation.rhs._public_control,
                                         self.optimizer_updates)
        context.validate(self._adaptive_reference, index)
        return self._adaptive_reference, history, context

    def _adaptive_history(self) -> tuple[AdaptiveReadoutRecord, ...]:
        """Authenticate the existing small snapshots; retain no second history."""
        self._validate_adaptive_reference(require_execution=True)
        state = self.continuation.rhs
        snapshots = self.continuation._accepted_readout_snapshots
        count = state.filter_revision
        admitted = state._filter_binding.get("native_admitted_readout_payload_sha256", {})
        initial_count = len(state.filter_observation_uids)-count
        expected_uids = state.filter_observation_uids[initial_count:]
        if (snapshots is not self._accepted_snapshot_authority
                or type(snapshots) is not tuple or len(snapshots) != count or self.observations_seen != count
                or type(admitted) is not dict or set(admitted) != set(expected_uids)):
            raise ValueError("Adaptive episode requires complete actual admitted snapshot history")
        records = []
        for offset, snapshot in enumerate(snapshots):
            if (type(snapshot) is not _AcceptedResidualReadout
                    or snapshot.query_index != offset+1 or snapshot.filtering_revision != offset
                    or snapshot.observation_uid != expected_uids[offset]
                    or snapshot.predictor_update_count != max(0, offset-self.policy["warmup_transitions"]+1)
                        * self.policy["updates_per_observation"]
                    or admitted.get(snapshot.observation_uid) != snapshot.payload_fingerprint
                    or arrived_payload_fingerprint(snapshot.arrived) != snapshot.payload_fingerprint):
                raise ValueError("Adaptive episode accepted history UID/revision/packet changed")
            context = self._adaptive_context(snapshot.query_index, snapshot.public_control,
                                             snapshot.predictor_update_count)
            mean = torch.tensor(snapshot.mean, dtype=torch.float64, device=state._log_weights.device)
            records.append(AdaptiveReadoutRecord.capture(self._adaptive_reference, context, snapshot.arrived, mean))
        result = tuple(records)
        self._validate_adaptive_posterior(result)
        return result

    def actual_adaptive_readout_history(self, report_context: dict, *, episode_status: str) -> dict:
        """Export only this owner's authenticated accepted prefix, without plant state."""
        if self._adaptive_reference is None:
            return {"status": "not_applicable", "reason": "sensor_only",
                    "scientific_readiness": "NOT_ADMITTED", "calibration_eligible": False}
        history = self._adaptive_history()
        payload = export_adaptive_readout_history(self._adaptive_reference, history,
            self._adaptive_initial_log_weights, report_context=report_context, episode_status=episode_status)
        payload["origin"] = "episode_authenticated_actual_acceptance"
        return payload

    def _validate_adaptive_posterior(self, history) -> None:
        state = self.continuation.rhs
        initial = normalized_adaptive_initial_weights(self._adaptive_initial_log_weights,
            device=state._log_weights.device,
            error_message="Adaptive episode requires its normalized immutable initial32-tuple prior")
        density = torch.zeros_like(initial)
        if history:
            row = history[-1]
            mean = initial.new_zeros(32, 6)
            mean[:, list(row.coordinates)] = mean.new_tensor(row.mean)
            values = initial.new_zeros(6)
            values[list(row.coordinates)] = values.new_tensor(row.values)
            query = AdaptiveReadoutQuery(row.context, row.query_index, row.filtering_revision,
                tuple(i in row.coordinates for i in range(6)), mean, values)
            conditioned = adaptive_readout_window(self._adaptive_reference, history[:-1], (query,))
            density = conditioned.history_log_likelihood + conditioned.log_likelihood
        expected = _normalized_log_weights((initial + density)[None])
        if not torch.allclose(expected, self.model.solver.current_log_weights(state), atol=1e-12, rtol=1e-12):
            raise ValueError("Adaptive episode posterior differs from immutable initial prior and SAME-law accepted history")

    def prospective_adaptive_h3_window(self, reference: AdaptiveReadoutReference, *,
                                      trial_predictor_binding: dict) -> AdaptiveResidualReadoutWindow:
        """Bind the last H3 to actual pre-update snapshots without changing policy.

        This accessor does not activate adaptive filtering, planning, or training.
        Call the loss explicitly with the returned software window. Fixed-action
        calibration remains no evidence of adaptive scientific qualification.
        """
        reference.validate()
        snapshots = self.continuation._accepted_readout_snapshots
        transitions = list(self._buffer)[-3:]
        if len(snapshots) < 3 or len(snapshots) != self.observations_seen:
            raise ValueError("Prospective adaptive H3 requires complete actual accepted readout history")
        if tuple(item.observation_uid for item in snapshots[-3:]) != tuple(
                item.arrived.observation.observation_uid for item in list(self._buffer)[-3:]):
            raise ValueError("Prospective adaptive H3 buffer and complete history differ")
        all_switches = tuple(row["request"] for row in self.continuation.rhs._execution_history)
        captured = []
        for item in snapshots:
            arrived = item.arrived
            index = arrived.acquisition.query_index
            switches = tuple(request for request in all_switches if request.prefix.query_index < index)
            original = switches[-1].actuation if switches else next(action for action in reference.original_actions
                if action.regimen_uid == reference.initial_action_uid)
            public = item.public_control or original
            binding = dict(json.loads(reference.fixed.predictor_binding),
                           residual_optimizer_updates=item.predictor_update_count)
            context = AdaptiveReadoutContext(original, public, switches,
                json.dumps(binding, sort_keys=True, allow_nan=False))
            mean = torch.tensor(item.mean, dtype=torch.float64, device=arrived.observation.values.device)
            captured.append(AdaptiveReadoutRecord.capture(reference, context, arrived, mean))
        window = AdaptiveResidualReadoutWindow(reference, tuple(captured[:-3]), tuple(captured[-3:]),
            self.continuation._initial_readout_log_weights, json.dumps(trial_predictor_binding, sort_keys=True, allow_nan=False))
        first = transitions[-3].start.rhs
        arrivals = tuple(item.arrived for item in transitions[-3:])
        binding = admitted_window_binding(self.model.solver, first, self.continuation.rhs, arrivals)
        if binding["status"] != "ready":
            raise ValueError("Prospective adaptive H3 packets lack actual admission binding")
        window.validate(self.model.solver, transitions[-3].start, self.continuation, arrivals, self.normalization)
        return window

    def _adapt(self, record: dict) -> None:
        if self.observations_seen < self.policy["warmup_transitions"]:
            record["status"] = "warmup"
            return
        window = list(self._buffer)[-self.policy["horizon_transitions"]:]
        arrivals = tuple(item.arrived for item in window)
        record.update(status="updating", window_query_indices=[row.acquisition.query_index for row in arrivals],
            window_observation_uids=[row.observation.observation_uid for row in arrivals],
            window_start_filter_revision=window[0].start.rhs.filter_revision,
            window_post_filter_revision=self.continuation.rhs.filter_revision,
            window_prior_authority="frozen_window_start_log_weights", loss_before_each_update=[])
        adaptive_window = None
        if self._adaptive_reference is not None:
            self._adaptive_history()
            binding = dict(json.loads(self._adaptive_reference.fixed.predictor_binding),
                           residual_optimizer_updates=self.optimizer_updates)
            adaptive_window = self.prospective_adaptive_h3_window(self._adaptive_reference,
                trial_predictor_binding=binding)
            record.update(window_prior_authority="initial_tuple_weights_plus_same_law_frozen_history",
                readout_law=self._adaptive_reference.law, adaptive_scientific_qualification="UNQUALIFIED",
                scope="prospective_adaptive_episode_software",
                likelihood_consumer="bound_actual_filter_and_H3_same_adaptive_law")
        self.model.eval()
        with torch.enable_grad():
            for _ in range(self.policy["updates_per_observation"]):
                if time.monotonic() >= getattr(self.model, "_execution_deadline", math.inf):
                    raise TimeoutError("Native residual H3 reached its cumulative evaluation wall cap")
                self.optimizer.zero_grad(set_to_none=True)
                arguments = {} if adaptive_window is None else {"adaptive_window": adaptive_window}
                loss, diagnostic = self.model.partial_observation_window_loss(
                    window[0].start, self.continuation, arrivals, self.normalization, **arguments)
                if loss is None or diagnostic["status"] != "loss_ready":
                    raise ValueError("Online residual update lacks an admitted consecutive H3 window")
                record["loss_before_each_update"].append(float(loss.detach()))
                loss.backward()
                gradients = [value.grad for value in self.model.parameters() if value.grad is not None]
                if not gradients or any(not torch.isfinite(value).all() for value in gradients):
                    raise FloatingPointError("Online residual gradients are missing or nonfinite")
                self.optimizer.step()
                self.optimizer_updates += 1
                record["optimizer_updates_completed"] += 1
                if any(not torch.isfinite(value).all() for value in self.model.parameters()):
                    raise FloatingPointError("Online residual parameters became nonfinite")
        self.optimizer.zero_grad(set_to_none=True)
        record["status"] = "updated"

    def step(self, provider: Callable[[NativeAcquisition], NativeArrivedObservation], *,
             selected_coordinate: int | None = None) -> dict:
        """Consume the next original query under the currently bound executed control.

        A failure is retained in ``trace`` and re-raised. The episode cannot retry;
        any physical advance, accepted filtering or parameter steps already done
        remain inspectable rather than being rewound or selected by loss value.
        """
        if self.status == "failed":
            raise ValueError("Failed online residual episode cannot continue")
        if self.executed_regimen_uid is None or self.optimizer is None:
            raise ValueError("Initial public selection must precede actual residual execution")
        # Reject a malformed choice before advancing either model or source.
        native_acquisition_mask(self.readout, selected_coordinate)
        index = self.continuation.rhs.query_index+1
        if index >= self.continuation.program.shape[1]:
            raise ValueError("Online residual episode exhausted its original query horizon")
        record = {"query_index": index, "selected_coordinate": selected_coordinate,
            "status": "started", "optimizer_updates_completed": 0,
            "scientific_twin": "NOT_RUN", "online_calibration_qualified": False,
            "scope": "partial_observation_H3_component_not_physical_MPC"}
        self.trace.append(record)
        arrived = None
        try:
            if self._adaptive_reference is not None:
                self._adaptive_history()
            with torch.no_grad():
                start = replace(self.continuation,
                    rhs=self.model.solver.fork_continuation(self.continuation.rhs))
                self.continuation, fields = self.model.advance_continuation(
                    self.continuation, self.normalization, query_count=index+1)
                prediction = predict_native_readout(self.model.solver, self.continuation.rhs,
                    self.readout, executed_regimen_uid=self.executed_regimen_uid,
                    selected_coordinate=selected_coordinate)
            arrived = provider(prediction.acquisition)
            if self._adaptive_reference is None:
                filtering = filter_native_observation(self.model.solver, self.continuation.rhs, prediction, arrived)
            else:
                history = self._adaptive_history()
                context = self._adaptive_context(index, self.continuation.rhs._public_control, self.optimizer_updates)
                filtering = filter_adaptive_native_observation(self.model.solver, self.continuation.rhs, prediction,
                    arrived, reference=self._adaptive_reference, history=history,
                    initial_log_weights=self._adaptive_initial_log_weights, context=context)
                record.update(readout_law=self._adaptive_reference.law, adaptive_scientific_qualification="UNQUALIFIED",
                              scope="prospective_adaptive_episode_software")
            state = self.continuation.rhs
            uid = state._execution_model_uid or self.executed_regimen_uid
            action = state._filter_binding["regimen_uids"].index(uid)
            self._model_history.append(self.model.solver.current_particle_fields(state)[0, :, action].detach().clone())
            # Detach and own accepted measurements. The admitted fingerprint is
            # checked again by H3; caller mutation cannot rewrite buffer history.
            observation = arrived.observation
            tensors = {name: None if getattr(observation, name) is None
                       else getattr(observation, name).detach().clone() for name in
                       ("values", "observed", "left_censored", "library_offset",
                        "detection_limit", "admission_log_probability")}
            retained = replace(arrived, observation=replace(observation, **tensors))
            transition = _ResidualObservationTransition(start, retained,
                tuple(tuple(row) for row in prediction.mean[0].detach().cpu().tolist()), self.optimizer_updates)
            snapshot = _AcceptedResidualReadout(observation.observation_uid, index,
                prediction.acquisition.filtering_revision, transition.predicted_mean, self.optimizer_updates,
                arrived_payload_fingerprint(retained), retained, start.rhs._public_control)
            self.continuation = replace(self.continuation,
                _accepted_readout_snapshots=self.continuation._accepted_readout_snapshots + (snapshot,))
            self._accepted_snapshot_authority = self.continuation._accepted_readout_snapshots
            self._buffer.append(transition)
            self.observations_seen += 1
            record.update(observation_uid=observation.observation_uid,
                observations_seen=self.observations_seen, buffered_transitions=len(self._buffer),
                filtering_revision=self.continuation.rhs.filter_revision,
                windows_reuse_observations=True, independent_sample_count_not_inferred=True)
            self._adapt(record)
        except Exception as error:
            self.status = "failed"
            self.failed_arrival = arrived
            record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
                optimizer_updates_total=self.optimizer_updates)
            raise
        self.status = "active"
        record["optimizer_updates_total"] = self.optimizer_updates
        return {"prediction": prediction, "particle_fields": fields,
                "filtering": filtering, "adaptation": copy.deepcopy(record)}
