"""Original-clock query/control lookahead with common complete STOP plan support.

The numerical forest is model-owned. Imagined outcomes only reweight its 32
biological tuples; no observation admission, optimizer or private plant is used.
"""
from __future__ import annotations

from Libs.utils.data.native_actuation import actuation_payload

from dataclasses import dataclass, field
import math
import time

import torch

from Libs.models.modules.world_state import _normalized_log_weights
from Libs.utils.data.native_actuation import (
    NativeActuation, NativeActuationRequest, compose_actuation, executed_prefix,
)
from Libs.utils.evaluation.native_intervention_decisions import composed_action_burden
from Libs.utils.exp.experiment_manifests import NATIVE_SEED_POLICIES, native_training_seed_values
from Libs.utils.model.native_arrived_observation import project_cns_readout
from Libs.utils.model.native_conditional_readout import (
    AdaptiveReadoutContext, AdaptiveReadoutQuery, AdaptiveReadoutRecord, adaptive_readout_window,
)


def temporal_query_policy(config):
    policy = config["native_temporal_query"]
    required = config["schemas"]["native_temporal_query"]["required_fields"]
    fixed = {
        "schema_version": "native_temporal_query.v1",
        "query_library": "one_available_mixed_coordinate_at_each_consecutive_original_clock",
        "stop_support": "same_complete_original_library_world_plans_as_query",
        "belief_update": "complete_biological_tuple_weights_only_frozen_current_residual",
        "stop_execution": "immutable_original_clock_schedule_without_observation_or_replanning",
        "numerical_choices": "prospective_authored_not_manuscript_numerics_not_tuned",
        "integration": "all_mixture_components_common_gaussian_noise",
        "measurement_cost": 1, "lookahead_depth": 2,
        "integration_noise_replicates": 2,
        "latency_unit": "physical_hour", "budget": "original_postzero_opportunities",
    }
    if (not isinstance(policy, dict) or set(required)-set(policy)
            or any(policy.get(k) != v for k, v in fixed.items())
            or any(type(policy.get(k)) not in (int, float) or not math.isfinite(policy[k])
                   or policy[k] < 0 for k in ("cost_penalty", "latency_penalty_per_hour"))):
        raise ValueError("Temporal queries require their prospective original-clock contract")
    protocol = config["manuscript_native_protocol"]
    native_training_seed_values(protocol["seed_policy"], protocol["training_seeds"])
    # A retained model keeps its training policy; new quadrature uses active seeds.
    allowed = NATIVE_SEED_POLICIES["manuscript_native_three"]
    if type(policy["integration_seed"]) is not int or policy["integration_seed"] not in allowed:
        raise ValueError("Temporal integration seed must be an explicit active native experiment seed")
    return dict(policy)


@dataclass
class TemporalEdge:
    request: NativeActuationRequest | None
    public: NativeActuation
    rhs_work: int
    child: TemporalNode | None = None
    branch: object | None = field(default=None, repr=False)
    arrival: object | None = field(default=None, repr=False)
    means: torch.Tensor | None = field(default=None, repr=False)
    numerical_source: TemporalEdge | None = field(default=None, repr=False)
    origin: object | None = field(default=None, repr=False)
    fields: torch.Tensor | None = field(default=None, repr=False)
    parameter_versions: tuple[int, ...] | None = field(default=None, repr=False)


@dataclass
class TemporalNode:
    query_index: int
    depth: int
    public: NativeActuation
    latency_hours: float
    edges: list[TemporalEdge] = field(default_factory=list)
    required_rhs_work: int = 0
    losses: torch.Tensor | None = field(default=None, repr=False)
    plans: list[tuple[NativeActuationRequest, ...]] = field(default_factory=list, repr=False)


def _actuation_numeric_content(public):
    """Same source/control bytes with separate public execution identities."""
    content = actuation_payload(public)
    content.pop("regimen_uid")
    return content


def compile_temporal_world_plans(episode, config, *, depth, initial_origin=None,
                                 reuse_identical_siblings=False):
    """Enumerate every legal plan and price its actual compiled RK4 interval union.

    A selected branch has A=1. The prescribed t0 continuation retains its actual
    original action width. No tensor propagation or admission happens here.
    """
    state = episode.continuation.rhs if initial_origin is None else initial_origin.rhs
    times, library = state._times, state._execution_library
    step = episode.model.solver.maximum_step_hours
    public = state._public_control if initial_origin is None else initial_origin._forecast_control
    if initial_origin is not None:
        episode.model.validate_forecast_continuation(initial_origin)
        if state.query_index != 0 or not state._planning_origin.get("initial_public_selection"):
            raise ValueError("Initial forest requires its pristine public t0 alternative")
    if type(depth) is not int or depth < 0 or public is None:
        raise ValueError("Temporal plans require a bound public origin and finite depth")

    def compile_node(public, index, remaining):
        node = TemporalNode(index, remaining, public,
                            0. if index == len(times)-1 else times[index+1]-times[index])
        if index == len(times)-1:
            return node
        candidates = [(None, public)]
        if index:
            candidates = []
            for action in library:
                request = NativeActuationRequest(executed_prefix(public, index), action)
                try:
                    joined = compose_actuation(request, library)
                except ValueError as error:
                    if str(error) not in {
                        "Composite actuation exceeds the registered total public dose budget",
                        "Composite actuation exceeds the registered original-clock input integral",
                    }:
                        raise
                    continue
                candidates.append((request, joined))
        if not candidates:
            raise ValueError("No original-library plan satisfies the public execution budget")
        for _, joined in candidates:
            if composed_action_burden(joined, config)["status"] != "defined":
                raise ValueError("A legal temporal candidate has undefined physical burden")
        if index == 0:
            intervals = state._intervals
            width = state._guide.n
        else:
            # Selection retains the compiler's union across ALL legal candidates.
            boundaries = sorted({times[index], *times[index+1:],
                *(event.source_time for _, joined in candidates for event in joined.dose_events
                  if times[index] < event.source_time <= times[-1])})
            intervals = tuple((a, b, math.ceil((b-a)/step)) for a, b in zip(boundaries, boundaries[1:]))
            width = state._guide.b*state._guide.j
        end = times[index+1] if remaining else times[-1]
        work = 4*width*sum(n for a, b, n in intervals if a >= times[index] and b <= end)
        for request, joined in candidates:
            child = compile_node(joined, index+1, remaining-1) if remaining else None
            edge = TemporalEdge(request, joined, work, child)
            if reuse_identical_siblings:
                edge.numerical_source = next((previous for previous in node.edges
                    if _actuation_numeric_content(previous.public) == _actuation_numeric_content(joined)), None)
            node.edges.append(edge)
            if edge.numerical_source is None:
                node.required_rhs_work += work+(child.required_rhs_work if child else 0)
        return node
    return compile_node(public, state.query_index, depth)


@torch.no_grad()
def evaluate_temporal_world_plans(episode, node, loss_function, *, initial_origin=None):
    """Propagate each shared prefix once; leaves supply full-grid terminal losses."""
    model, solver = episode.model, episode.model.solver
    if initial_origin is None and episode.adaptive_reference is not None:
        episode.prospective_adaptive_temporal_readout()
    state = episode.continuation.rhs if initial_origin is None else initial_origin.rhs
    ledger = state._work_ledger
    if ledger.spent+node.required_rhs_work > ledger.maximum:
        raise ValueError("Complete temporal world-plan forest exceeds the shared RHS work budget")
    ledger.planning_active = True
    versions = tuple(value._version for value in (*model.parameters(), *model.buffers()))
    mask, grid = state._validity[0, 0, 0], state._guide.grid
    initial_work = ledger.spent

    def loss(values, public):
        burden = composed_action_burden(public, model.config)
        if burden["status"] != "defined":
            raise ValueError("Temporal terminal plan has undefined physical burden")
        result = loss_function(values[:, None, None], mask, grid,
                               values.new_tensor([burden["normalized_burden"]]))["loss"][:, 0, 0]
        if result.shape != (32,) or not torch.isfinite(result).all():
            raise ValueError("Temporal terminal loss must cover all32 complete biological tuples")
        return result

    def visit(current, continuation, past, reused_node=None):
        if time.monotonic() >= getattr(model, "_execution_deadline", math.inf):
            raise TimeoutError("Temporal planning reached its cumulative evaluation wall cap")
        if not current.edges:
            current.losses, current.plans = loss(past, current.public)[:, None], [()]
            return
        branch = None
        if current.query_index:
            branch = solver.prepare_future_control_branch(continuation.rhs, episode._execution_inputs,
                episode.normalization, executed_regimen_uid=continuation._forecast_executed_uid,
                candidate_regimen_uids=[edge.request.actuation.regimen_uid for edge in current.edges],
                public_control=continuation._forecast_control)
        losses, plans = [], []
        for edge_index, edge in enumerate(current.edges):
            prepared = continuation if edge.request is None else model.prepare_forecast_control(
                continuation, episode._execution_inputs, episode.normalization, branch, edge.request)
            if prepared._forecast_control != edge.public:
                raise ValueError("Temporal numerical branch changed its public plan")
            edge.branch = branch
            count = current.query_index+2 if edge.child is not None else len(state._times)
            if solver.forecast_work(prepared.rhs, query_count=count) != edge.rhs_work:
                raise ValueError("Temporal public work price differs from the actual compiled interval union")
            edge.origin = prepared
            source = reused_node.edges[edge_index] if reused_node is not None else edge.numerical_source
            spent = ledger.spent
            if source is None:
                advanced, fields = model.advance_forecast(prepared, episode.normalization, query_count=count)
                expected_work = edge.rhs_work
            else:
                if (source.origin is None or source.arrival is None or source.fields is None
                        or _actuation_numeric_content(source.public) != _actuation_numeric_content(edge.public)
                        or len(source.child.edges if source.child else ()) != len(edge.child.edges if edge.child else ())):
                    raise ValueError("Identical sibling forecast has no complete matching numerical result")
                advanced = model._reuse_forecast_result(prepared, source.origin, source.arrival,
                                                        parameter_versions=source.parameter_versions)
                fields = source.fields.detach().clone()
                expected_work = 0
            if ledger.spent-spent != expected_work:
                raise RuntimeError("Temporal executed RHS work differs from its actual shared interval price")
            edge.arrival, edge.fields = advanced, fields.detach().clone()
            edge.parameter_versions = versions
            guide = advanced.rhs._guide
            action = advanced.rhs._filter_binding["regimen_uids"].index(advanced._forecast_executed_uid)
            fields = fields.reshape(guide.b, guide.j, guide.a, -1, *model.field_shape)[0, :, action]
            if current.query_index == 0:
                fields = fields[:, 1:]
            history = torch.cat((past, fields), 1)
            prefix = () if edge.request is None else (edge.request,)
            if edge.child is None:
                losses.append(loss(history, edge.public)[:, None])
                plans.append(prefix)
            else:
                edge.arrival = advanced
                edge.means = project_cns_readout(fields[:, -1], episode.readout)
                visit(edge.child, advanced, history, None if source is None else source.child)
                losses.append(edge.child.losses)
                plans.extend(prefix+suffix for suffix in edge.child.plans)
        current.losses, current.plans = torch.cat(losses, 1), plans
    if initial_origin is None:
        origin, past = model.model_only_continuation(episode.continuation), episode.recorded_model_history()
    else:
        model.validate_forecast_continuation(initial_origin)
        if state.query_index or not state._planning_origin.get("initial_public_selection"):
            raise ValueError("Initial evaluation requires an unadvanced public alternative")
        origin = initial_origin
        past = solver.current_particle_fields(state)[0, :, 0][:, None].detach().clone()
    visit(node, origin, past)
    if ledger.spent-initial_work != node.required_rhs_work:
        raise RuntimeError("Temporal complete-forest accounting differs from executed model work")
    if tuple(value._version for value in (*model.parameters(), *model.buffers())) != versions:
        raise RuntimeError("Residual parameters changed during hypothetical planning")
    return node


@torch.no_grad()
def select_initial_public_plan(episode, library, loss_function, config, *, evaluation_order=None,
                               reuse_identical_siblings=True, selection_mode="public_open_loop_plan"):
    """Compare all t0 actions and complete open-loop leaves before private execution.

    The estimand is minimum posterior terminal public loss, not the joint future
    query/control Bellman value. Every model alternative shares one actual RHS
    ledger. The selected origin is pristine; its advanced forecasts stay private
    to planning and never become the physical carry.
    """
    if (episode.executed_regimen_uid is not None or episode.optimizer is not None
            or episode.observations_seen or episode.continuation.rhs.query_index):
        raise ValueError("Initial public planning must precede execution and optimizer creation")
    policy = temporal_query_policy(config)
    if selection_mode not in {"public_open_loop_plan", "public_joint_query_control"}:
        raise ValueError("Initial selector requires a declared public planning mode")
    joint = selection_mode == "public_joint_query_control"
    state = episode.continuation.rhs
    ledger = state._work_ledger
    indices = tuple(range(len(library)))
    if not indices:
        raise ValueError("Initial planning requires a nonempty original action library")
    order = indices if evaluation_order is None else tuple(evaluation_order)
    if len(order) != len(indices) or set(order) != set(indices):
        raise ValueError("Initial evaluation order must contain every original action exactly once")
    report = {"status": "started", "estimand": "minimum_posterior_terminal_public_loss_complete_open_loop_plans",
        "initial_execution": "public_open_loop_plan_selected_t0",
        "selection_scope": "all_original_library_initial_actions_and_complete_depth_two_terminal_leaves",
        "tie_break": "original_library_then_complete_plan_order",
        "lookahead_depth": policy["lookahead_depth"], "loss_reference_id": loss_function.reference_id,
        "particle_count": 32, "candidates": [], "scientific_twin": "NOT_RUN",
        "formal_qualification": False, "full_residual_mpc_eligible": False,
        "independent_predictive_error_calibration": False,
        "initial_joint_future_query_value_optimized": False,
        "private_world_access_before_decision": False, "optimizer_updates": 0,
        "numerical_reuse": "exact_identical_siblings" if reuse_identical_siblings else "disabled_reference",
        "work_accounting": "shared_ledger_actual_unique_work_and_unshared_numerical_path_prices"}
    if joint:
        report.update(estimand="minimum_expected_terminal_query_cost_and_latency_complete_depth_two",
            initial_execution="public_joint_query_control_selected_t0",
            initial_joint_future_query_value_optimized=True, planner_anticipates_residual_retraining=False,
            temporal_policy=policy, current_particle_log_weights=episode.model.solver.current_log_weights(state).tolist())
    alternatives = []
    for index, action in enumerate(library):
        burden = composed_action_burden(action, config)
        row = {"initial_original_library_index": index, "regimen_uid": action.regimen_uid,
               "burden": burden, "status": burden["status"]}
        report["candidates"].append(row)
    if any(row["burden"]["status"] != "defined" for row in report["candidates"]):
        report.update(status="blocked", reason="initial_public_burden_undefined")
        return report, None
    for index, action in enumerate(library):
        row = report["candidates"][index]
        origin = episode.model.prepare_initial_forecast(episode.continuation,
            episode._execution_inputs, episode.normalization, library, action, episode.readout)
        node = compile_temporal_world_plans(episode, config, depth=policy["lookahead_depth"], initial_origin=origin,
                                            reuse_identical_siblings=reuse_identical_siblings)
        row["required_rhs_work"] = node.required_rhs_work
        def unshared_work(current):
            return sum(edge.rhs_work+(unshared_work(edge.child) if edge.child else 0) for edge in current.edges)
        row["unshared_rhs_work"] = unshared_work(node)
        alternatives.append((origin, node))
    required = sum(node.required_rhs_work for _, node in alternatives)
    report.update(required_rhs_work=required,
                  unshared_rhs_work=sum(row["unshared_rhs_work"] for row in report["candidates"]),
                  remaining_rhs_work=ledger.maximum-ledger.spent,
                  actual_rhs_work_before_planning=ledger.spent)
    if required > ledger.maximum-ledger.spent:
        report.update(status="blocked", reason="complete_initial_public_comparison_exceeds_shared_rhs_budget",
                      actual_rhs_work_after_planning=ledger.spent)
        return report, None
    weights = episode.model.solver.current_log_weights(state).double().exp()[0]
    before = ledger.spent
    def leaf_work(current):
        if not current.edges:
            return [0]
        return [edge.rhs_work+suffix for edge in current.edges
                for suffix in ([0] if edge.child is None else leaf_work(edge.child))]

    for index in order:
        origin, node = alternatives[index]
        evaluate_temporal_world_plans(episode, node, loss_function, initial_origin=origin)
        risks = weights @ node.losses.double()
        plans = []
        work_by_plan = leaf_work(node)
        if len(work_by_plan) != len(node.plans):
            raise RuntimeError("Initial terminal plan prices differ from complete retained leaves")
        for plan_index, steps in enumerate(node.plans):
            final = origin._forecast_control
            for request in steps:
                final = compose_actuation(request, library)
            plans.append({"plan_index": plan_index, "predictive_risk": float(risks[plan_index]),
                "burden": composed_action_burden(final, config),
                "path_rhs_work": work_by_plan[plan_index],
                "steps": [actuation_payload(request) for request in steps], "final_public_program": actuation_payload(final)})
        chosen = int(risks.argmin())
        report["candidates"][index].update(status="scored", complete_terminal_plan_count=len(plans),
            terminal_plans=plans, selected_plan_index=chosen, predictive_risk=float(risks[chosen]),
            actual_rhs_work=node.required_rhs_work)
    if ledger.spent-before != required:
        raise RuntimeError("Initial all-action comparison differs from cumulative actual RHS price")
    remaining = ledger.maximum-ledger.spent
    if joint:
        for index in order:
            origin, node = alternatives[index]
            decision = plan_temporal_query(node, episode.model.solver.current_log_weights(state), episode.readout,
                policy, adaptive_episode=episode if episode.adaptive_reference is not None else None,
                initial_origin=origin, rhs_budget=remaining,
                query_reserve=lambda edge, preceding: query_execution_reserve(episode, edge, preceding_edges=preceding))
            report["candidates"][index].update(joint_decision=decision,
                joint_decision_loss=decision["estimated_decision_loss"],
                remaining_rhs_work_at_joint_comparison=remaining)
    key = "joint_decision_loss" if joint else "predictive_risk"
    chosen = min(indices, key=lambda index: report["candidates"][index][key])
    origin, node = alternatives[chosen]
    row = report["candidates"][chosen]
    report.update(status="selected", selected_initial_original_library_index=chosen,
        selected_regimen_uid=library[chosen].regimen_uid, selected_plan_index=row["selected_plan_index"],
        selected_open_loop_plan=row["terminal_plans"][row["selected_plan_index"]],
        actual_rhs_work_after_planning=ledger.spent, actual_comparison_rhs_work=ledger.spent-before,
        current_particle_weights=weights.cpu().tolist())
    episode.install_initial_selection(origin, library, library[chosen],
                                      select_adaptive_initial_reference=joint)
    if joint and episode.adaptive_reference is not None:
        report["adaptive_initial_reference"] = actuation_payload(episode.adaptive_reference)
    return report, node


def query_execution_reserve(episode, edge, *, preceding_edges=()):
    """Actual next advance, prescribed H3, and its complete refreshed terminal family."""
    if edge.child is None:
        raise ValueError("A terminal forecast cannot issue an observation query")
    h3 = 0
    if episode.observations_seen+len(preceding_edges)+1 >= episode.policy["warmup_transitions"]:
        history_work = [episode.model.solver.forecast_work(
            row.start.rhs, query_count=row.arrived.acquisition.query_index+1) for row in episode._buffer]
        history_work.extend(previous.rhs_work for previous in preceding_edges)
        retained = history_work[-(episode.policy["horizon_transitions"]-1):]
        if len(retained) != episode.policy["horizon_transitions"]-1:
            raise ValueError("Query reserve lacks the consecutive actual and hypothetical H3 transition history")
        replay = edge.rhs_work+sum(retained)
        h3 = episode.policy["updates_per_observation"]*replay
    refresh = edge.child.required_rhs_work if h3 else 0
    return {"actual_advance_rhs_work": edge.rhs_work, "actual_H3_rhs_work": h3,
            "post_H3_complete_terminal_family_rhs_work": refresh,
            "total_rhs_work": edge.rhs_work+h3+refresh}


def verify_arrival_forecast(episode, expected):
    """Allow no-H3 terminal reuse only for the identical full numerical carry."""
    actual = episode.continuation
    if (actual.rhs.query_index != expected.rhs.query_index
            or actual.rhs._guide.n != expected.rhs._guide.n
            or actual.rhs._intervals != expected.rhs._intervals
            or actual.rhs._public_control != expected._forecast_control):
        raise ValueError("Actual arrival differs from its forecast clock/layout/compiled controls")
    for name in ("cns_state", "auxiliary_state", "event_times", "event_doses", "applied_events"):
        left, right = getattr(actual.rhs, name), getattr(expected.rhs, name)
        if (left is None) != (right is None) or left is not None and not torch.equal(left, right):
            raise ValueError("Actual arrival differs from its full conditional forecast carry")
    if not torch.equal(actual.program, expected.program) or not torch.equal(actual.static, expected.static):
        raise ValueError("Actual arrival changed the forecast residual conditioning")


@torch.no_grad()
def plan_temporal_query(node, log_weights, readout, policy, *, admissible=None,
                        adaptive_episode=None, initial_origin=None, rhs_budget=None, query_reserve=None):
    """T_k=min over complete plans; V_k=min(T_k,cost+E[V_(k-1)]).

    Every node compares against ALL its control-plan leaves, including later
    precommitted switches. The default uses the retained sensor-only Normal law.
    An explicit adaptive episode uses its SAME-law conditional readout history.
    Query risk uses the current-best feasible child plan as a posterior regret
    anchor. This preserves zero information value when no later control can
    change, without assuming finite quadrature preserves the Bayes martingale.
    Finite nested mixture quadrature is not a calibrated VOI bound. Future
    residual retraining and later receding-horizon computation are not predicted.
    """
    if (log_weights.shape != (1, 32) or node.losses is None
            or not torch.allclose(torch.logsumexp(log_weights.double(), -1),
                                  log_weights.new_zeros(1).double(), atol=1e-12, rtol=0)):
        raise ValueError("Temporal planning requires complete normalized32-tuple weights and losses")
    if (rhs_budget is not None and (type(rhs_budget) is not int or rhs_budget < 0)
            or initial_origin is not None and node.query_index != 0):
        raise ValueError("Temporal execution budget and initial origin must be valid")
    readout.validate()
    available = tuple(i for i, observed in enumerate(readout.observed) if observed)
    count = policy["integration_noise_replicates"]
    generator = torch.Generator(device=log_weights.device).manual_seed(policy["integration_seed"])
    # Independent original-clock/coordinate innovations; common across control
    # branches and mixture components at the same acquisition opportunity.
    noise = torch.randn(node.query_index+node.depth+1, len(readout.observed), count, generator=generator,
                        device=log_weights.device, dtype=torch.float64)
    laws = {}
    reference = None
    histories, context, versions = None, None, None
    if adaptive_episode is not None:
        reference, history, context = (adaptive_episode.prospective_adaptive_temporal_readout()
            if initial_origin is None else adaptive_episode.prospective_adaptive_temporal_readout(initial_origin=initial_origin))
        context.validate(reference, node.query_index)
        state = adaptive_episode.continuation.rhs if initial_origin is None else initial_origin.rhs
        if (initial_origin is None and reference != adaptive_episode.adaptive_reference or reference.fixed.readout != readout
                or state.query_index != node.query_index or context.executed_program != node.public
                or not torch.allclose(log_weights, adaptive_episode.model.solver.current_log_weights(state),
                                      atol=1e-12, rtol=1e-12)):
            raise ValueError("Adaptive temporal origin differs from the authenticated current32-tuple posterior")
        histories = (history,)
        initial_logs = log_weights.new_tensor(adaptive_episode._adaptive_initial_log_weights)
        versions = tuple(value._version for value in
                         (*adaptive_episode.model.parameters(), *adaptive_episode.model.buffers()))

    def adaptive_likelihoods(edge, logs, branch_histories, parent_context):
        candidate = parent_context
        if edge.request is not None:
            candidate = AdaptiveReadoutContext(edge.request.actuation, edge.public,
                parent_context.switches + (edge.request,), parent_context.predictor_binding)
        index = edge.child.query_index
        candidate.validate(reference, index)
        if candidate.executed_program != edge.public or len(branch_histories) != len(logs):
            raise ValueError("Adaptive temporal candidate changed public execution or branch history")
        windows = []
        for batch, history in enumerate(branch_histories):
            query = AdaptiveReadoutQuery(candidate, index,
                reference.fixed.initial_filtering_revision + len(history), readout.observed, edge.means)
            window = adaptive_readout_window(reference, history, (query,))
            expected = _normalized_log_weights((initial_logs + window.history_log_likelihood)[None])[0]
            if not torch.allclose(expected, logs[batch], atol=1e-12, rtol=1e-12):
                raise ValueError("Adaptive temporal branch posterior differs from SAME-law complete history")
            windows.append(window)
        means = torch.stack([window.mean for window in windows])
        variances = torch.stack([window.covariance.diagonal() for window in windows])
        return candidate, means, variances

    def imagined_histories(edge, coordinate, candidate, branch_histories, outcomes):
        # Capture the unconditioned pre-query predictor, not its innovation-adjusted
        # conditional mean. Only this selected coordinate becomes hypothetical history.
        mean = tuple(tuple(row) for row in edge.means[:, coordinate:coordinate + 1].double().cpu().tolist())
        values = outcomes.cpu().tolist()
        result = []
        for batch, history in enumerate(branch_histories):
            revision = reference.fixed.initial_filtering_revision + len(history)
            for outcome_index, value in enumerate(values[batch]):
                row = AdaptiveReadoutRecord(reference, candidate, edge.child.query_index, revision,
                    f"hypothetical_temporal:{edge.child.query_index}:{coordinate}:{outcome_index}",
                    (coordinate,), mean, (value,))
                result.append(history + (row,))
        return tuple(result)

    def likelihood(edge, coordinate, query_index):
        key = (id(edge), coordinate, query_index)
        if key not in laws:
            means = edge.means[:, coordinate].double()
            if means.shape != (32,) or not torch.isfinite(means).all():
                raise ValueError("Imagined temporal readout must retain every complete tuple")
            outcomes = (means[:, None]+readout.standard_deviation*noise[query_index, coordinate]).reshape(-1)
            distribution = torch.distributions.Normal(means, torch.full_like(means, readout.standard_deviation))
            laws[key] = distribution.log_prob(outcomes[:, None])
        return laws[key]

    def evaluate(current, logs, root=False, branch_histories=None, parent_context=None, budget=None, preceding=()):
        risks = logs.exp() @ current.losses.double()
        stop, chosen = risks.min(-1)
        values = stop.clone()
        rows = []
        if current.depth:
            penalty = policy["measurement_cost"]*policy["cost_penalty"]+current.latency_hours*policy["latency_penalty_per_hour"]
            for index, edge in enumerate(current.edges):
                if edge.child is None or root and admissible is not None and not admissible[index]:
                    continue
                reserve = None
                next_budget = budget
                if budget is not None:
                    reserve = ({"total_rhs_work": edge.rhs_work} if query_reserve is None
                               else query_reserve(edge, preceding))
                    work = reserve["total_rhs_work"]
                    if type(work) is not int or work < edge.rhs_work:
                        raise ValueError("Hypothetical query reserve must include its actual advance")
                    if work > budget:
                        continue
                    next_budget = budget-work
                candidate = conditional_means = conditional_variances = None
                if reference is not None:
                    candidate, conditional_means, conditional_variances = adaptive_likelihoods(
                        edge, logs, branch_histories, parent_context)
                for position, coordinate in enumerate(available):
                    next_histories = None
                    if reference is None:
                        ll = likelihood(edge, coordinate, current.query_index+1)[None]
                    else:
                        means = conditional_means[:, :, position]
                        scales = conditional_variances[:, position].sqrt()
                        outcomes = (means[:, :, None] + scales[:, None, None]
                            * noise[current.query_index+1, coordinate]).reshape(len(logs), -1)
                        distribution = torch.distributions.Normal(means[:, None], scales[:, None, None])
                        ll = distribution.log_prob(outcomes[:, :, None])
                        if edge.child.depth:
                            next_histories = imagined_histories(edge, coordinate, candidate, branch_histories, outcomes)
                    updated = _normalized_log_weights((logs[:, None]+ll).reshape(-1, 32))
                    after, _, _, _ = evaluate(edge.child, updated, branch_histories=next_histories,
                                              parent_context=candidate, budget=next_budget,
                                              preceding=preceding+(edge,))
                    masses = logs.exp().repeat_interleave(count, dim=-1)/count
                    child_prior = logs.exp() @ edge.child.losses.double()
                    baseline, selected_plan = child_prior.min(-1)
                    # Each batch has its own best plan subject to this already
                    # selected first control. Outcomes may change only later
                    # choices. The exact posterior-risk identity removes the
                    # quadrature bias of the fixed-plan component.
                    posterior_risks = updated.exp() @ edge.child.losses.double()
                    anchors = posterior_risks.reshape(logs.shape[0], -1, child_prior.shape[-1]).gather(
                        2, selected_plan[:, None, None].expand(-1, masses.shape[-1], 1)).squeeze(-1)
                    regret = anchors-after.reshape(logs.shape[0], -1)
                    expected = baseline-(masses*regret).sum(-1)
                    query = penalty+expected
                    values = torch.minimum(values, query)
                    if root:
                        rows.append({"edge_index": index, "coordinate": coordinate,
                            "expected_terminal_and_future_query_loss": float(expected[0]),
                            "query_loss": float(query[0]), "net_benefit": float(stop[0]-query[0]),
                            "physical_latency_hours": current.latency_hours})
        return values, stop, chosen, rows
    value, stop, chosen, rows = evaluate(node, log_weights.double(), True,
                                        branch_histories=histories, parent_context=context, budget=rhs_budget)
    if reference is not None and tuple(value._version for value in
            (*adaptive_episode.model.parameters(), *adaptive_episode.model.buffers())) != versions:
        raise RuntimeError("Adaptive temporal predictor changed during hypothetical conditioning")
    selected = min(rows, key=lambda row: row["query_loss"]) if rows else None
    if selected is not None and selected["query_loss"] >= float(stop[0]):
        selected = None
    report = {"decision": "STOP" if selected is None else "QUERY",
        "selected_edge_index": None if selected is None else selected["edge_index"],
        "selected_coordinate": None if selected is None else selected["coordinate"],
        "stop_plan_index": int(chosen[0]), "stop_risk": float(stop[0]),
        "estimated_decision_loss": float(value[0]), "candidates": rows,
        "complete_terminal_plan_count": len(node.plans), "particle_count": 32,
        "lookahead_depth": node.depth, "hypothetical_residual_parameters": "fixed_current",
        "hypothetical_admission_and_optimizer_updates": 0,
        "estimator": "complete_mixture_common_noise_feasible_current_plan_posterior_regret_identity",
        "integration_uncertainty": "finite_nested_MC_error_unquantified"}
    if rhs_budget is not None:
        report.update(hypothetical_query_rhs_budget=rhs_budget,
            hypothetical_query_budget_accounting="deduct_each_preceding_actual_advance_H3_and_refresh_reserve",
            planner_anticipates_residual_retraining=False)
    if reference is not None:
        report.update(readout_law=reference.law, adaptive_scientific_qualification="UNQUALIFIED",
            live_policy_uses_adaptive_law=True,
            discrepancy_history_authority="immutable_actual_and_hypothetical_preupdate_mean_value_snapshots",
            accepted_history_observation_count=len(histories[0]),
            tuple_prior_authority="immutable_initial_weights_plus_same_law_complete_branch_history")
    return report
