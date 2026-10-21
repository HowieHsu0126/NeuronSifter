"""Prospective information-only replicate queries on an unchanged native t0 world.

The planner sees public candidate readout laws and complete-particle action losses.
Only its callable measurement provider owns the evaluator field and measurement RNG.
This component does not implement post-treatment switching or a residual predictor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Callable

import torch

from Libs.models.modules.world_state import _normalized_log_weights
from Libs.utils.data.joint_observation import (
    RetainedObservation, observation_to_payload, retained_log_likelihood,
)
from Libs.utils.evaluation.native_intervention_decisions import summarize_posterior_action_loss
from Libs.utils.model.native_arrived_observation import NativeCNSReadout, native_cns_readout, project_cns_readout


def predose_query_policy(config: dict) -> dict:
    policy = config["native_predose_query"]
    required = {
        "schema_version": "native_predose_query.v1",
        "timing": "information_only_predose_no_physical_time_advance",
        "query_library": "one_independent_replicate_per_original_retained_coordinate",
        "budget": "original_available_coordinate_count",
        "completion_admission": "state_independent_success_or_stop_on_actual_failure",
        "numerical_choices": "prospective_authored_not_manuscript_numerics_not_tuned",
        "integration": "all_mixture_components_common_gaussian_noise",
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise ValueError("Native pre-dose query contract is absent or changed")
    if (policy.get("measurement_cost") != 1 or policy.get("logical_latency") != 0
            or type(policy.get("lookahead_depth")) is not int or policy["lookahead_depth"] != 2
            or not isinstance(policy.get("cost_penalty"), (float, int))
            or not math.isfinite(policy["cost_penalty"]) or policy["cost_penalty"] < 0
            or type(policy.get("integration_noise_replicates")) is not int
            or not 2 <= policy["integration_noise_replicates"] <= 32
            or type(policy.get("integration_seed")) is not int
            or "measurement_seed" in policy):
        raise ValueError("Native pre-dose cost, depth and finite integration budget must be explicit")
    return policy


def original_predose_readout(fields: torch.Tensor, readout: NativeCNSReadout) -> torch.Tensor:
    """Preserve the original producer's normalization/mixing float32 boundaries."""
    count = len(readout.selector)
    identity = tuple(tuple(float(i == j) for j in range(count)) for i in range(count))
    normalized = project_cns_readout(fields, replace(readout, matrix=identity)).float()
    return (normalized.double() @ normalized.new_tensor(readout.matrix, dtype=torch.float64).T).float()


@dataclass(frozen=True)
class NativePredoseRequest:
    readout: NativeCNSReadout
    support_uid: str
    origin_posterior_uid: str
    coordinate: int
    observation: RetainedObservation
    variance: float


@dataclass(frozen=True)
class NativePredoseChannel:
    readout: NativeCNSReadout
    support_uid: str
    particle_uids: tuple[str, ...]
    original: RetainedObservation
    parameters: dict
    nuisance: torch.Tensor

    def validate(self) -> None:
        self.readout.validate()
        original, spec = self.original, self.readout
        if (len(self.particle_uids) != 32 or len(set(self.particle_uids)) != 32
                or original.values.shape != (1, 6) or original.sequence_number != 0
                or original.channel_uid != "controlled_predose_selector"
                or original.law != "gaussian" or original.modality != "state"
                or original.source_uid != spec.source_uid
                or original.source_snapshot_ref != spec.source_snapshot_ref
                or original.role != spec.role or original.units != spec.units
                or original.normalization_ref != spec.normalization_ref
                or original.assay_uid != "controlled_predose_gaussian"
                or original.encoder_indices != tuple(range(6))
                or original.physical_acquisition is not None or original.left_censored.any()
                or original.informative_admission or original.admission_log_probability is not None
                or not torch.equal(original.observed.cpu(), torch.tensor([spec.observed]))):
            raise ValueError("Pre-dose queries require the original source/mask/units and 32 complete tuples")
        original.validate(self.parameters, {})
        mean, variance = self.parameters["mean"], self.parameters["variance"]
        expected = torch.full_like(variance, spec.standard_deviation**2)
        if (mean.shape != (1, 32, 6) or variance.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.equal(variance, expected)
                or self.nuisance.shape != mean.shape or torch.count_nonzero(self.nuisance)):
            raise ValueError("Pre-dose queries require the unchanged finite original Gaussian readout law")

    @property
    def available(self) -> tuple[int, ...]:
        return tuple(i for i, observed in enumerate(self.readout.observed) if observed)

    def request(self, coordinate: int, sequence: int, parent_uid: str,
                used: tuple[int, ...]) -> NativePredoseRequest:
        if (type(coordinate) is not int or coordinate not in self.available or coordinate in used
                or len(set(used)) != len(used) or not set(used) <= set(self.available)
                or sequence != len(used)+1):
            raise ValueError("Unsupported, duplicate or out-of-order pre-dose coordinate query")
        mask = torch.zeros_like(self.original.observed)
        mask[:, coordinate] = True
        observation = replace(self.original,
            observation_uid=self.original.observation_uid+":replicate:"+str(coordinate),
            sequence_number=sequence, observed=mask,
            values=torch.full_like(self.original.values, float("nan"), dtype=torch.float64))
        return NativePredoseRequest(self.readout, self.support_uid, parent_uid, coordinate,
            observation, float(self.parameters["variance"][0, 0, coordinate]))


def native_predose_channel(inputs, declaration: dict, config: dict) -> NativePredoseChannel:
    predose_query_policy(config)
    if len(inputs.observations) != 1 or inputs.group_shape[:2] != (1, 32):
        raise ValueError("Pre-dose query episode requires one original sample and 32 complete tuples")
    spec = native_cns_readout(config, declaration)
    support = inputs.posterior.support
    original = inputs.observations[0]
    parameters, nuisance = support.resolve_readout(original.channel_uid)
    if (support.support_uid != declaration["support_uid"]
            or support.sample_uids != (declaration["sample_uid"],)
            or support.source_family_uid != declaration["family_uid"]
            or inputs.posterior.sequence_number != 0):
        raise ValueError("Pre-dose support/sample/family differs from its original declaration")
    result = NativePredoseChannel(spec, support.support_uid, support.particle_uids[0],
                                  original, parameters, nuisance)
    result.validate()
    if not torch.equal(original_predose_readout(support.initial_field, spec), parameters["mean"]):
        raise ValueError("Original public readout means differ from the original t0 projection")
    return result


def admit_predose_observation(channel: NativePredoseChannel, log_weights: torch.Tensor,
                              request: NativePredoseRequest, observation: RetainedObservation) -> torch.Tensor:
    """One admission and normalized-log-weight update for actual and imagined outcomes."""
    template = request.observation
    if (request.readout != channel.readout or request.support_uid != channel.support_uid
            or request.coordinate not in channel.available
            or request.variance != float(channel.parameters["variance"][0, 0, request.coordinate])
            or not request.origin_posterior_uid
            or log_weights.shape != (1, len(channel.particle_uids))
            or not isinstance(observation, RetainedObservation)):
        raise ValueError("Pre-dose measurement has an unsupported acquisition binding")
    fixed = ("observation_uid", "source_uid", "source_snapshot_ref", "sample_uids", "role", "channel_uid",
             "law", "units", "encoder_indices", "sequence_number", "normalization_ref", "modality", "assay_uid")
    expected_mask = torch.zeros_like(channel.original.observed)
    expected_mask[0, request.coordinate] = True
    if (any(getattr(template, key) != getattr(channel.original, key)
            for key in fixed if key not in {"observation_uid", "sequence_number"})
            or template.observation_uid != channel.original.observation_uid+":replicate:"+str(request.coordinate)
            or not 1 <= template.sequence_number <= len(channel.available)
            or not torch.equal(template.observed, expected_mask) or template.left_censored.any()):
        raise ValueError("Pre-dose request differs from the original admitted channel")
    if (any(getattr(observation, key) != getattr(template, key) for key in fixed)
            or not torch.equal(observation.observed, template.observed)
            or not torch.equal(observation.left_censored, template.left_censored)
            or observation.physical_acquisition is not None or observation.informative_admission
            or observation.admission_log_probability is not None
            or observation.library_offset is not None or observation.detection_limit is not None):
        raise ValueError("Pre-dose measurement source/mask/units/sequence/admission changed")
    likelihood = retained_log_likelihood(observation, channel.parameters, channel.nuisance, {})
    return _normalized_log_weights(log_weights+likelihood.double())


def _outcome(request: NativePredoseRequest, value: torch.Tensor) -> RetainedObservation:
    values = request.observation.values.clone()
    values[0, request.coordinate] = value
    return replace(request.observation, values=values)


@torch.no_grad()
def plan_predose_query(channel: NativePredoseChannel, log_weights: torch.Tensor, loss_draws: torch.Tensor,
                       used: tuple[int, ...], policy: dict) -> dict:
    """Depth-two Bayes risk lookahead, estimated from the full predictive mixture.

    For current action a0, E_y[R_y(a0)-min_a R_y(a)] equals prior risk minus
    expected posterior risk. The regret identity gives exactly zero for a query
    that never changes the optimal action. Future net benefit includes its cost.
    Common Gaussian draws remove categorical sampling noise and make an
    irrelevant query's conditional future integration identical. The finite
    nested Monte Carlo result is still an approximation, not a calibrated bound.
    """
    evaluations = 0
    count = policy["integration_noise_replicates"]
    generator = torch.Generator(device=log_weights.device).manual_seed(policy["integration_seed"])
    noise = torch.randn(6, count, dtype=torch.float64, device=log_weights.device, generator=generator)
    cost = policy["cost_penalty"]*policy["measurement_cost"]

    def recurse(logs, previous, depth):
        nonlocal evaluations
        decision = summarize_posterior_action_loss(loss_draws, logs[0].exp())
        chosen = decision["selected_index"]
        rows = []
        if depth <= 0:
            return decision, rows, 0.
        for coordinate in channel.available:
            if coordinate in previous:
                continue
            request = channel.request(coordinate, len(previous)+1, "planning", previous)
            # Integrate every component with its current weight. Equal scalar
            # Gaussian laws may share outcome nodes; their masses are summed
            # exactly here, while every posterior update retains all32 tuples.
            means, inverse = torch.unique(channel.parameters["mean"][0, :, coordinate].double(), return_inverse=True)
            component_mass = torch.zeros_like(means).scatter_add_(0, inverse, logs[0].exp())
            component_mass = component_mass/component_mass.sum()
            scale = channel.parameters["variance"][0, 0, coordinate].sqrt().double()
            outcomes = (means[:, None]+scale*noise[coordinate][None]).reshape(-1)
            outcome_mass = (component_mass[:, None].expand(-1, count)/count).reshape(-1)
            immediate_gains, continuations, posterior_risks = [], [], []
            for value in outcomes:
                updated = admit_predose_observation(channel, logs, request, _outcome(request, value))
                evaluations += 1
                after, _, continuation = recurse(updated, (*previous, coordinate), depth-1)
                risks = after["predictive_risks"]
                immediate = float(risks[chosen]-risks.min())
                immediate_gains.append(immediate)
                continuations.append(continuation)
                posterior_risks.append(float(risks.min()))
            expected = lambda values: float((outcome_mass*outcome_mass.new_tensor(values)).sum())
            immediate, future = expected(immediate_gains), expected(continuations)
            rows.append({"coordinate": coordinate, "estimated_immediate_risk_reduction": immediate,
                "estimated_future_net_benefit": future, "estimated_net_benefit": immediate+future-cost,
                "estimated_immediate_posterior_risk": expected(posterior_risks),
                "conditional_outcome_nodes": len(outcomes), "original_mixture_components": 32,
                "independent_gaussian_noise_replicates": count})
        best = max((row["estimated_net_benefit"] for row in rows), default=0.)
        return decision, rows, max(0., best)

    decision, rows, benefit = recurse(log_weights, used, min(policy["lookahead_depth"], len(channel.available)-len(used)))
    selected = max(rows, key=lambda row: row["estimated_net_benefit"]) if rows else None
    return {"selected_coordinate": selected["coordinate"] if selected and benefit > 0 else None,
        "selected_action_index": decision["selected_index"],
        "predictive_risks": decision["predictive_risks"].cpu().tolist(), "candidates": rows,
        "conditional_updates": evaluations,
        "estimator": "complete_mixture_common_noise_nested_MC_current_action_posterior_regret_identity",
        "integration_uncertainty": "finite_nested_MC_error_unquantified_no_calibrated_bound_or_max_selection_bias_guarantee"}


@torch.no_grad()
def run_native_predose_episode(channel: NativePredoseChannel, log_weights: torch.Tensor,
        loss_draws: torch.Tensor, regimen_uids: tuple[str, ...],
        provider: Callable[[NativePredoseRequest], RetainedObservation], config: dict, *,
        origin_posterior_uid: str) -> tuple[torch.Tensor, tuple[RetainedObservation, ...], dict]:
    channel.validate()
    policy = predose_query_policy(config)
    if (loss_draws.shape[0] != 32 or len(regimen_uids) != loss_draws.shape[-1]
            or len(set(regimen_uids)) != len(regimen_uids) or not origin_posterior_uid):
        raise ValueError("Query episode needs complete paired action losses in original action order")
    if (log_weights.shape != (1, 32) or torch.isnan(log_weights).any() or torch.isposinf(log_weights).any()
            or not torch.allclose(torch.logsumexp(log_weights.double(), -1),
                                  torch.zeros(1, dtype=torch.float64, device=log_weights.device), atol=1e-12, rtol=0)):
        raise ValueError("Pre-dose episode requires current normalized complete log weights")
    current = log_weights.clone()
    initial = summarize_posterior_action_loss(loss_draws, current[0].exp())
    records, observations, used = [], [], ()
    spent = 0
    report = {"schema_version": "native_predose_episode.v1", "status": "started",
        "policy": dict(policy), "support_uid": channel.support_uid,
        "particle_uids": list(channel.particle_uids), "regimen_uids": list(regimen_uids),
        "origin_posterior_uid": origin_posterior_uid, "particle_count": 32,
        "maximum_measurement_units": len(channel.available), "measurement_units_spent": 0,
        "initial_selected_regimen_uid": regimen_uids[initial["selected_index"]],
        "initial_log_weights": current[0].cpu().tolist(), "steps": records,
        "physical_time_advanced": False, "final_program_origin": "original_t0",
        "initial_encoder_packet_changed": False, "resampled": False,
        "completion_model": "state_independent_success_no_failure_event_likelihood",
        "scientific_twin": "NOT_RUN", "formal_qualification": False,
        "scope": "prospective_information_only_component_not_full_RQ4_or_residual_MPC"}
    while True:
        plan = plan_predose_query(channel, current, loss_draws, used, policy)
        record = {"planning": plan, "prior_log_weights": current[0].cpu().tolist()}
        records.append(record)
        coordinate = plan["selected_coordinate"]
        if coordinate is None:
            report.update(status="component_complete", stop_reason=("budget_exhausted" if len(used)==len(channel.available)
                else "no_positive_estimated_net_risk_reduction"),
                selected_regimen_uid=regimen_uids[plan["selected_action_index"]])
            break
        request = channel.request(coordinate, len(used)+1, origin_posterior_uid, used)
        spent += policy["measurement_cost"]
        report["measurement_units_spent"] = spent
        record.update(coordinate=coordinate, incurred_measurement_units=policy["measurement_cost"],
                      logical_latency=policy["logical_latency"])
        started = time.monotonic()
        arrived = None
        try:
            arrived = provider(request)
            updated = admit_predose_observation(channel, current, request, arrived)
        except Exception as error:
            # Provider/admission is the external failure boundary. Unexpected
            # runtime errors also spend the issued cost and cannot resume biology.
            record.update(status="measurement_failed", biological_update_withheld=True,
                failure={"type": type(error).__name__, "message": str(error)},
                quarantined_observation=observation_to_payload(arrived) if isinstance(arrived, RetainedObservation) else None)
            report.update(status="failed", stop_reason="measurement_failed", selected_regimen_uid=None)
            break
        finally:
            record["actual_wall_latency_seconds"] = time.monotonic()-started
        current = updated
        observations.append(arrived)
        used = (*used, coordinate)
        record.update(status="accepted", biological_update_withheld=False,
            observation=observation_to_payload(arrived), posterior_log_weights=current[0].cpu().tolist())
    report.update(final_log_weights=current[0].cpu().tolist(), accepted_query_count=len(observations),
        total_measurement_loss_penalty=spent*policy["cost_penalty"],
        completed_coordinate_queries=list(used))
    required = config["schemas"]["native_predose_episode"]["required_fields"]
    if not set(required) <= set(report):
        raise ValueError("Native pre-dose episode schema is incomplete")
    return current, tuple(observations), report
