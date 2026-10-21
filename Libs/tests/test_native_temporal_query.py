"""Temporal decision invariants and original-law execution; no new source corpus."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Libs.tests.test_native_mechanistic_residual import cases, execution_cases, _actual_execution_case
from Libs.tests.test_native_residual_control import _control_case, _loss
from Libs.tests.test_native_physical_plant import explicit_world, FAMILIES
from Libs.utils.data.native_actuation import NativeActuationRequest, compose_actuation, compile_actuation_plan, executed_prefix
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_residual_control_evaluation import run_native_temporal_residual_episode, realized_native_residual_control
from Libs.utils.model.native_temporal_query import (
    TemporalNode, TemporalEdge, temporal_query_policy, compile_temporal_world_plans,
    evaluate_temporal_world_plans, plan_temporal_query, query_execution_reserve,
)


def policy():
    return temporal_query_policy(load_config("Libs/configs/exp/manuscript_native_protocol.yaml"))


@pytest.mark.parametrize("training_policy,training_seeds", [
    ("manuscript_native_three", [41, 42, 43]),
    ("manuscript_native_five", [40, 41, 42, 43, 44]),
])
def test_temporal_query_policy_uses_active_explicit_seed(training_policy, training_seeds):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["manuscript_native_protocol"].update(seed_policy=training_policy, training_seeds=training_seeds)
    for seed in (41, 42, 43):
        config["native_temporal_query"]["integration_seed"] = seed
        result = temporal_query_policy(config)
        assert result["integration_seed"] == seed
        assert result["lookahead_depth"] == 2 and result["integration_noise_replicates"] == 2
    for seed in (40, 44, 7441, True, 41.0, None):
        config["native_temporal_query"]["integration_seed"] = seed
        with pytest.raises(ValueError, match="explicit active native experiment seed"):
            temporal_query_policy(config)


@pytest.mark.parametrize("training_policy,training_seeds", [
    ("unknown", [41, 42, 43]),
    ("manuscript_native_three", [40, 41, 42, 43, 44]),
    ("manuscript_native_five", [41, 42, 43]),
    ("manuscript_native_five", [40, 41, 42, 43, 43]),
    ("manuscript_native_five", [40, 41, 42, 43, True]),
    ("manuscript_native_three", [41, 42, "43"]),
    ("manuscript_native_three", (41, 42, 43)),
    ("manuscript_native_three", None),
])
def test_temporal_query_policy_rejects_invalid_training_declaration(training_policy, training_seeds):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["manuscript_native_protocol"].update(seed_policy=training_policy, training_seeds=training_seeds)
    config["native_temporal_query"]["integration_seed"] = 41
    with pytest.raises(ValueError, match="Native training seed policy requires"):
        temporal_query_policy(config)


@pytest.mark.parametrize("missing", ["seed_policy", "training_seeds"])
def test_temporal_query_policy_requires_explicit_training_declaration(missing):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["manuscript_native_protocol"].pop(missing)
    with pytest.raises(KeyError, match=missing):
        temporal_query_policy(config)


def abstract_tree(*, informative=False, root_commits=False):
    # Complete binary control plans u0,u1,u2; pure decision test, no plant.
    truth = torch.arange(32)%2
    means = torch.zeros(32, 6, dtype=torch.float64)
    if informative:
        means[:, 0] = truth*4.-2.
    def node(prefix):
        result = TemporalNode(len(prefix)+1, 2-len(prefix), None, 0.)
        if len(prefix) == 2:
            losses = []
            for last in (0, 1):
                path = (*prefix, last)
                if informative:
                    chosen = path[0] if root_commits else path[-1]
                    losses.append((truth != chosen).double())
                else:
                    losses.append(torch.full((32,), .1 if path == (0,1,0) else .8, dtype=torch.float64))
            result.losses, result.plans = torch.stack(losses, 1), [(), ()]
        else:
            for control in (0, 1):
                child = node((*prefix, control))
                result.edges.append(TemporalEdge(None, None, 0, child, means=means))
            result.losses = torch.cat([edge.child.losses for edge in result.edges], 1)
            result.plans = [plan for edge in result.edges for plan in edge.child.plans]
        return result
    return node(())


def decide(tree):
    readout = SimpleNamespace(observed=(True,)*6, standard_deviation=.1, validate=lambda: None)
    return plan_temporal_query(tree, torch.full((1,32), -torch.log(torch.tensor(32., dtype=torch.float64)), dtype=torch.float64), readout, policy())


def test_uninformative_query_cannot_borrow_hybrid_control_value():
    tree = abstract_tree()
    decision = decide(tree)
    assert float(tree.losses[:, 0].mean()) == pytest.approx(.8)
    assert float(tree.losses[:, -1].mean()) == pytest.approx(.8)
    assert decision["complete_terminal_plan_count"] == 8
    assert decision["stop_risk"] == pytest.approx(.1)
    assert decision["decision"] == "STOP" and decision["selected_coordinate"] is None
    assert all(row["net_benefit"] <= 0 for row in decision["candidates"])


def test_informative_query_changes_only_controls_after_its_arrival():
    adaptive = decide(abstract_tree(informative=True))
    assert adaptive["decision"] == "QUERY" and adaptive["selected_coordinate"] == 0
    assert adaptive["stop_risk"] == pytest.approx(.5)
    assert adaptive["estimated_decision_loss"] < .03
    assert adaptive["hypothetical_admission_and_optimizer_updates"] == 0
    committed = decide(abstract_tree(informative=True, root_commits=True))
    assert committed["decision"] == "STOP" and committed["stop_risk"] == pytest.approx(.5)


def test_planned_private_switches_match_clock_by_clock_execution_without_measurements():
    _, _, _, _, _, library, planned = explicit_world(FAMILIES[0])
    *_, sequential = explicit_world(FAMILIES[0])
    planned.advance(planned.prefix)
    sequential.advance(sequential.prefix)
    first = NativeActuationRequest(planned.prefix, library[1])
    joined = compose_actuation(first, library)
    second = NativeActuationRequest(executed_prefix(joined, 3), library[0])
    plan = compile_actuation_plan(planned._public_program, planned.prefix, (first, second), library)
    rng = planned._generator.get_state().clone()
    with pytest.raises(ValueError):
        planned._complete_planned_tail(replace(plan, steps=(second, first)))
    assert not planned._tail_started
    planned._complete_planned_tail(plan)
    schedule = {row.prefix.query_index: row for row in plan.steps}
    while sequential._query+1 < len(sequential._times):
        if sequential._query in schedule:
            sequential.switch(schedule[sequential._query])
        sequential.advance(sequential.prefix)
    left, right = planned._evaluation_history(), sequential._evaluation_history()
    assert torch.equal(torch.from_numpy(left["private_fields"]), torch.from_numpy(right["private_fields"]))
    assert planned._public_program == sequential._public_program == plan.final_program
    assert [row["regimen_uid"] for row in planned._history] == [row["regimen_uid"] for row in sequential._history]
    assert torch.equal(planned._generator.get_state(), rng) and not planned._acquired_queries
    with pytest.raises(RuntimeError, match="unobserved tail"):
        planned.advance(planned.prefix)


def temporal_case(execution_cases):
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    temporal_query_policy(current)
    original, episode, plant, library = _control_case(execution_cases, initial_index=0)
    episode.model.config["native_temporal_query"] = copy.deepcopy(current["native_temporal_query"])
    episode.model.config["schemas"]["native_temporal_query"] = copy.deepcopy(current["schemas"]["native_temporal_query"])
    for key in ("seed_policy", "training_seeds"):
        episode.model.config["manuscript_native_protocol"][key] = copy.deepcopy(current["manuscript_native_protocol"][key])
    return original, episode, plant, library


@pytest.mark.parametrize("molecular_administration", [False, True], ids=["electrical", "molecular_opt_in"])
def test_actual_temporal_episode_reaches_scored_original_terminal(execution_cases, tmp_path, molecular_administration):
    if molecular_administration:
        current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
        temporal_query_policy(current)
        _, episode, plant, library = _actual_execution_case(
            execution_cases, FAMILIES[2], molecular_administration=True,
            model_seed=41, measurement_seed=41)
        for key in ("seed_policy", "training_seeds"):
            episode.model.config["manuscript_native_protocol"][key] = copy.deepcopy(current["manuscript_native_protocol"][key])
        for key in ("native_temporal_query", "native_residual_control"):
            episode.model.config[key] = copy.deepcopy(current[key])
            episode.model.config["schemas"][key] = copy.deepcopy(current["schemas"][key])
        assert episode._execution_inputs.group_shape == (1, 32, 3)
        assert len(library) == 3 and all(action.administration is not None for action in library)
        assert episode.model.field_shape == (6, 6, 26)
    else:
        _, episode, plant, _ = temporal_case(execution_cases)
    reference = _loss(episode)
    result = run_native_temporal_residual_episode(episode, reference, plant.acquire, plant.switch,
        plant._complete_planned_tail, episode.model.config)
    realized = realized_native_residual_control(plant, result["status"], reference, episode.model.config)
    output = Path(os.environ.get("NRNSIFT_TEMPORAL_EVIDENCE", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    (output/"actual_temporal_episode.json").write_text(json.dumps({"episode": result, "realized": realized}, indent=2)+"\n")
    assert result["status"] == "component_complete", result
    assert realized["source_horizon_complete"] and realized["realized_full_grid_public_loss"] is not None
    assert result["actual_model_rhs_tuple_action_work"] <= 4000000
    assert result["observations_seen"] == len(plant._acquired_queries) == result["measurement_units_spent"]
    assert result["decisions"][-1]["decision"] == "STOP"
    assert not result["formal_qualification"] and not result["full_residual_mpc_eligible"]
    if molecular_administration:
        assert result["policy"]["lookahead_depth"] == 2
        assert result["scientific_twin"] == "NOT_RUN"
        assert result["measurement_units_spent"] <= len(episode.continuation.rhs._times)-1
        assert result["optimizer_updates"] == episode.optimizer_updates
        assert result["optimizer_updates"] % episode.policy["updates_per_observation"] == 0
        print(json.dumps({"event": "actual_molecular_temporal_episode", "scope": "software_fixture_not_calibrated_MPC",
            "model_and_measurement_seed": 41, "particle_count": 32, "regimen_count": 3,
            "decision_count": len(result["decisions"]), "activation_calls": sum(
                bool(row.get("actuator_called")) for row in result["decisions"]),
            "observations_seen": result["observations_seen"], "optimizer_updates": result["optimizer_updates"],
            "stop_reason": result["stop_reason"], "rhs_work": result["actual_model_rhs_tuple_action_work"],
            "scientific_twin": "NOT_RUN"}))


def test_actual_H3_reserve_counts_A3_history_and_selected_A1_transition(execution_cases):
    _, episode, plant, _ = temporal_case(execution_cases)
    for _ in range(4):
        episode.step(plant.acquire, selected_coordinate=next(i for i, v in enumerate(episode.readout.observed) if v))
    tree = compile_temporal_world_plans(episode, episode.model.config, depth=2)
    evaluate_temporal_world_plans(episode, tree, _loss(episode))
    edge = tree.edges[0]
    reserved = query_execution_reserve(episode, edge)
    assert [row.start.rhs._guide.a for row in list(episode._buffer)[-2:]] == [3,3]
    assert edge.arrival.rhs._guide.a == 1 and reserved["actual_H3_rhs_work"] > 0
    ledger = episode.continuation.rhs._work_ledger
    before = ledger.spent
    episode.activate_selected_control(edge.branch, edge.request, plant.switch)
    episode.step(plant.acquire, selected_coordinate=next(i for i, v in enumerate(episode.readout.observed) if v))
    assert episode.optimizer_updates == 10
    assert ledger.spent-before == reserved["actual_advance_rhs_work"]+reserved["actual_H3_rhs_work"]
    refreshed = compile_temporal_world_plans(episode, episode.model.config, depth=edge.child.depth)
    assert refreshed.required_rhs_work == reserved["post_H3_complete_terminal_family_rhs_work"]
    before = ledger.spent
    evaluate_temporal_world_plans(episode, refreshed, _loss(episode))
    assert ledger.spent-before == reserved["post_H3_complete_terminal_family_rhs_work"]


def test_finite_mixture_quadrature_cannot_buy_information_without_future_control():
    means = torch.linspace(-2., 2., 32, dtype=torch.float64)[:, None].expand(-1, 6)
    logs = torch.linspace(-3., 1., 32, dtype=torch.float64).log_softmax(0)[None]
    readout = SimpleNamespace(observed=(True,)*6, standard_deviation=.7, validate=lambda: None)
    settings = dict(policy(), cost_penalty=0.000001)
    for losses in (torch.linspace(0., 1., 32, dtype=torch.float64),
                   torch.linspace(1., 0., 32, dtype=torch.float64)):
        terminal = TemporalNode(2, 0, None, 0., losses=losses[:, None], plans=[()])
        current = TemporalNode(1, 1, None, 0., edges=[TemporalEdge(None, None, 0, terminal, means=means)],
                               losses=losses[:, None], plans=[()])
        result = plan_temporal_query(current, logs, readout, settings)
        assert result["decision"] == "STOP"
        assert all(row["net_benefit"] == pytest.approx(-0.000001, abs=1e-12) for row in result["candidates"])


# These small algebra fixtures reuse the existing prospective readout reference.
# They do not run a source or allocate the complete residual model.
def _adaptive_temporal_case():
    from Libs.tests.test_native_adaptive_readout import adaptive_reference, capture
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutContext
    from Libs.models.modules.world_state import _normalized_log_weights
    ref = adaptive_reference()
    initial = torch.log_softmax(torch.arange(32, dtype=torch.float64) / 19, 0)
    a, b = ref.original_actions
    context = AdaptiveReadoutContext(a, a, (), ref.fixed.predictor_binding)
    history = (capture(ref, context, 1, selected=0),)
    from Libs.tests.test_native_adaptive_readout import dense_adaptive
    logs = _normalized_log_weights((initial + dense_adaptive(ref, history))[None])
    model = torch.nn.Linear(1, 1).double().eval()
    state = SimpleNamespace(query_index=1, _log_weights=logs)
    model.solver = SimpleNamespace(current_log_weights=lambda current: current._log_weights)
    episode = SimpleNamespace(adaptive_reference=ref, _adaptive_initial_log_weights=tuple(initial.tolist()),
        model=model, continuation=SimpleNamespace(rhs=state),
        prospective_adaptive_temporal_readout=lambda: (ref, history, context))
    truth = torch.arange(32) % 2
    def node(public, index, depth):
        result = TemporalNode(index, depth, public, ref.fixed.time_hours[index+1]-ref.fixed.time_hours[index]
                              if depth else 0.)
        if not depth:
            result.losses = torch.stack([(truth != chosen).double() for chosen in (0, 1)], 1)
            result.plans = [(), ()]
            return result
        for action in (a, b):
            request = NativeActuationRequest(executed_prefix(public, index), action)
            joined = compose_actuation(request, ref.original_actions)
            child = node(joined, index+1, depth-1)
            means = torch.arange(192, dtype=torch.float64).reshape(32, 6)/170 + index/5
            result.edges.append(TemporalEdge(request, joined, 0, child, means=means))
        result.losses = torch.cat([edge.child.losses for edge in result.edges], 1)
        result.plans = [plan for edge in result.edges for plan in edge.child.plans]
        return result
    return episode, node(a, 1, 2), logs, history


def _dense_temporal_window(reference, history, query):
    from Libs.tests.test_native_conditional_readout import explicit_field_projection
    base = explicit_field_projection(reference.fixed)
    base = base @ base.T
    blocks = [(row.coordinates, reference.fixed.time_hours[row.query_index], row.context.original_action.regimen_uid)
              for row in history]
    coordinates = tuple(i for i, keep in enumerate(query.observed) if keep)
    blocks.append((coordinates, reference.fixed.time_hours[query.query_index], query.context.original_action.regimen_uid))
    joint = torch.cat([torch.cat([base[list(a)][:, list(b)] *
        (min(t, u)/reference.fixed.reference_time_hours if action == other else 0.)
        for b, u, other in blocks], 1) for a, t, action in blocks], 0)
    joint += torch.eye(len(joint), dtype=torch.float64)*reference.fixed.readout.standard_deviation**2
    count = sum(len(row.coordinates) for row in history)
    predicted = query.mean[:, list(coordinates)]
    if count:
        means = torch.cat([torch.tensor(row.mean, dtype=torch.float64) for row in history], 1)
        values = torch.cat([torch.tensor(row.values, dtype=torch.float64) for row in history])
        gain = torch.linalg.solve(joint[:count, :count], joint[:count, count:]).T
        predicted = predicted + (values-means) @ gain.T
        covariance = joint[count:, count:] - gain @ joint[:count, count:]
        density = torch.distributions.MultivariateNormal(means, covariance_matrix=joint[:count, :count]).log_prob(values)
    else:
        covariance, density = joint, predicted.new_zeros(32)
    return predicted, covariance, density


def test_adaptive_temporal_nested_history_means_covariance_and_joint_weights(monkeypatch):
    import Libs.utils.model.native_temporal_query as temporal
    from Libs.tests.test_native_adaptive_readout import dense_adaptive
    episode, tree, logs, accepted = _adaptive_temporal_case()
    raw_window, raw_normalize = temporal.adaptive_readout_window, temporal._normalized_log_weights
    initial = logs.new_tensor(episode._adaptive_initial_log_weights)
    checks, nested, pending = [], [], []
    def inspect_window(reference, history, queries):
        result = raw_window(reference, history, queries)
        assert len(queries) == 1
        mean, covariance, density = _dense_temporal_window(reference, history, queries[0])
        torch.testing.assert_close(result.mean, mean, atol=2e-10, rtol=2e-10)
        torch.testing.assert_close(result.covariance, covariance, atol=2e-10, rtol=2e-10)
        torch.testing.assert_close(result.history_log_likelihood, density, atol=2e-10, rtol=2e-10)
        checks.append((history, queries[0], result))
        if len(history) > len(accepted):
            # Full density of A history plus a hypothetical acquisition is the
            # next branch prior. This detects recentering or historical double count.
            expected = torch.log_softmax(initial+dense_adaptive(reference, history), 0)
            pending.append(expected)
            nested.append((history, queries[0]))
        return result
    def inspect_normalize(values):
        result = raw_normalize(values)
        if values.shape == (1, 32) and pending:
            torch.testing.assert_close(result[0], pending.pop(0), atol=2e-10, rtol=2e-10)
        return result
    monkeypatch.setattr(temporal, "adaptive_readout_window", inspect_window)
    monkeypatch.setattr(temporal, "_normalized_log_weights", inspect_normalize)
    versions = tuple(p._version for p in episode.model.parameters())
    report = temporal.plan_temporal_query(tree, logs, episode.adaptive_reference.fixed.readout,
        policy(), adaptive_episode=episode)
    assert checks and nested and not pending
    assert any(history[-1].context.original_action != query.context.original_action
               for history, query in nested)
    assert any(history[-1].context.original_action == query.context.original_action
               for history, query in nested)
    assert all(history[0] is accepted[0] for history, _ in nested)
    assert all(history[-1].coordinates in tuple((i,) for i in range(6)) for history, _ in nested)
    assert report["readout_law"] == episode.adaptive_reference.law
    assert report["live_policy_uses_adaptive_law"] is True
    assert report["adaptive_scientific_qualification"] == "UNQUALIFIED"
    assert report["particle_count"] == 32 and report["lookahead_depth"] == 2
    assert report["hypothetical_admission_and_optimizer_updates"] == 0
    assert tuple(p._version for p in episode.model.parameters()) == versions
    assert torch.equal(episode.continuation.rhs._log_weights, logs)


def test_temporal_explicit_none_keeps_exact_legacy_result():
    tree = abstract_tree(informative=True)
    readout = SimpleNamespace(observed=(True,)*6, standard_deviation=.1, validate=lambda: None)
    logs = torch.full((1,32), -torch.log(torch.tensor(32., dtype=torch.float64)), dtype=torch.float64)
    assert plan_temporal_query(tree, logs, readout, policy()) == plan_temporal_query(
        tree, logs, readout, policy(), adaptive_episode=None)


@pytest.mark.parametrize("fault", ["posterior", "history", "reference", "context", "model_change"])
def test_adaptive_temporal_rejects_invalid_origin_and_changed_predictor(monkeypatch, fault):
    import Libs.utils.model.native_temporal_query as temporal
    episode, tree, logs, history = _adaptive_temporal_case()
    reference, _, context = episode.prospective_adaptive_temporal_readout()
    if fault == "posterior":
        logs = torch.log_softmax(torch.arange(32, dtype=torch.float64)[None], 1)
    elif fault == "history":
        episode.prospective_adaptive_temporal_readout = lambda: (reference, (), context)
    elif fault == "reference":
        episode.adaptive_reference = replace(reference, fixed=replace(reference.fixed, particle_uids=(("wrong",)*32,)))
    elif fault == "context":
        episode.prospective_adaptive_temporal_readout = lambda: (reference, history, replace(context, original_action=reference.original_actions[1]))
    else:
        raw = temporal.adaptive_readout_window
        def changed(*args):
            result = raw(*args)
            with torch.no_grad():
                episode.model.weight.add_(1.)
            return result
        monkeypatch.setattr(temporal, "adaptive_readout_window", changed)
    with pytest.raises((ValueError, RuntimeError)):
        plan_temporal_query(tree, logs, reference.fixed.readout, policy(), adaptive_episode=episode)


def _adaptive_actual_temporal_episode(execution_cases):
    from Libs.models.modules.native_mechanistic_residual import NativeResidualOnlineEpisode
    from Libs.utils.model.native_conditional_readout import FixedReadoutReference, AdaptiveReadoutReference
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    temporal_query_policy(current)
    original, legacy, plant, library = _control_case(execution_cases, initial_index=2)
    legacy.model.config["native_temporal_query"] = copy.deepcopy(current["native_temporal_query"])
    legacy.model.config["schemas"]["native_temporal_query"] = copy.deepcopy(current["schemas"]["native_temporal_query"])
    for key in ("seed_policy", "training_seeds"):
        legacy.model.config["manuscript_native_protocol"][key] = copy.deepcopy(current["manuscript_native_protocol"][key])
    state = legacy.continuation.rhs
    context = legacy.model.solver.filter_context(state)
    fixed = FixedReadoutReference(legacy.readout, context["support_uid"], context["origin_posterior_uid"],
        tuple(context["sample_uids"]), tuple(tuple(row) for row in context["particle_uids"]), library[2].regimen_uid,
        json.dumps({"normalization_id":legacy.normalization.identity,"checkpoint_sha256":"synthetic_temporal_predictor"}),
        legacy.normalization.identity, "synthetic_software_scale", (.01,)*26,
        tuple(legacy.normalization.scale.tolist()), max(state._times), tuple(library[2].source_times),
        tuple(state._times), library[2].source_time_unit, 0)
    reference = AdaptiveReadoutReference(fixed, library, library[2].regimen_uid)
    episode = NativeResidualOnlineEpisode(legacy.model, legacy._execution_inputs, legacy.normalization, legacy.readout,
        executed_regimen_uid=library[2].regimen_uid, adaptive_reference=reference)
    episode.bind_execution(library, library[2], plant.observation_identity)
    return episode, plant, reference


def test_adaptive_temporal_actual_consumer_passes_authenticated_episode(execution_cases, monkeypatch):
    import Libs.utils.evaluation.native_residual_control_evaluation as consumer
    episode, plant, reference = _adaptive_actual_temporal_episode(execution_cases)
    raw = consumer.plan_temporal_query
    calls = []
    def inspect(*args, **kwargs):
        assert kwargs["adaptive_episode"] is episode
        result = raw(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(consumer, "plan_temporal_query", inspect)
    result = consumer.run_native_temporal_residual_episode(episode, _loss(episode), plant.acquire,
        plant.switch, plant._complete_planned_tail, episode.model.config)
    assert result["status"] == "component_complete", result
    assert calls and all(row["readout_law"] == reference.law for row in calls)
    assert result["conditional_law"] == reference.law and not result["formal_qualification"]
    assert result["observations_seen"] == len(plant._acquired_queries)


@pytest.mark.parametrize("fault", ["history", "posterior", "reference"])
def test_adaptive_temporal_origin_rejected_before_numerical_forecast(execution_cases, monkeypatch, fault):
    # Reuse the bounded actual execution fixture without a plant call or H3.
    episode, plant, reference = _adaptive_actual_temporal_episode(execution_cases)
    tree = compile_temporal_world_plans(episode, episode.model.config, depth=2)
    if fault == "history":
        episode.continuation = replace(episode.continuation, _accepted_readout_snapshots=())
        episode._accepted_snapshot_authority = (object(),)
    elif fault == "posterior":
        episode.continuation.rhs._log_weights = torch.log_softmax(torch.arange(32, dtype=torch.float64)[None], 1)
    else:
        episode._adaptive_reference = replace(reference, fixed=replace(reference.fixed, support_uid="wrong"))
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid adaptive origin reached numerical forecast or plant")
    monkeypatch.setattr(episode.model, "advance_forecast", forbidden)
    monkeypatch.setattr(plant, "acquire", forbidden)
    with pytest.raises(ValueError):
        evaluate_temporal_world_plans(episode, tree, _loss(episode))


def test_nested_query_budget_deducts_preceding_advance_and_refresh_reserve():
    tree = abstract_tree(informative=True)
    # Only the second original acquisition resolves the terminal control. The
    # first consumes its reserved work without information, so both are needed.
    for edge in tree.edges:
        edge.means = torch.zeros_like(edge.means)
    readout = SimpleNamespace(observed=(True,)*6, standard_deviation=.1, validate=lambda: None)
    logs = torch.full((1,32), -torch.log(torch.tensor(32., dtype=torch.float64)), dtype=torch.float64)
    seen = []
    def reserve(edge, preceding):
        seen.append(len(preceding))
        return {"total_rhs_work":6}
    bounded = plan_temporal_query(tree, logs, readout, policy(), rhs_budget=6, query_reserve=reserve)
    none = plan_temporal_query(tree, logs, readout, policy(), rhs_budget=0, query_reserve=reserve)
    assert 0 in seen and 1 in seen
    assert none["decision"] == "STOP"
    assert bounded["hypothetical_query_rhs_budget"] == 6
    assert bounded["hypothetical_query_budget_accounting"] == "deduct_each_preceding_actual_advance_H3_and_refresh_reserve"
    enough = plan_temporal_query(tree, logs, readout, policy(), rhs_budget=12, query_reserve=reserve)
    assert bounded["decision"] == "STOP"
    assert enough["decision"] == "QUERY"
    assert bounded["estimated_decision_loss"] > enough["estimated_decision_loss"]


def test_initial_action_bellman_choice_can_differ_from_open_loop_risk():
    # Explicit artificial complete32-tuple controlled world. Exact retained
    # sensor-law planning is used; no predicted law or posterior update is mocked.
    logs = torch.full((1,32), -torch.log(torch.tensor(32., dtype=torch.float64)), dtype=torch.float64)
    readout = SimpleNamespace(observed=(True,)*6, standard_deviation=.1, validate=lambda: None)
    uninformative = abstract_tree(informative=False)
    informative = abstract_tree(informative=True)
    # One public initial action is safer without evidence; the other reveals
    # which later original control avoids its loss. Preserve every terminal leaf.
    def offset(node, amount):
        node.losses = node.losses+amount
        for edge in node.edges:
            if edge.child is not None:
                offset(edge.child, amount)
    offset(uninformative, .1)
    first = plan_temporal_query(uninformative, logs, readout, policy(), rhs_budget=100)
    second = plan_temporal_query(informative, logs, readout, policy(), rhs_budget=100)
    assert first["stop_risk"] < second["stop_risk"]
    assert second["estimated_decision_loss"] < first["estimated_decision_loss"]
    assert second["decision"] == "QUERY"
    assert first["particle_count"] == second["particle_count"] == 32


def test_molecular_opt_in_depth_two_forecast_preserves_public_mass_and_readonly_carry():
    from Libs.utils.data.native_actuation import native_actuation_library,actuation_payload
    from Libs.utils.model.native_observable_source import MOLECULAR
    config=load_config("Libs/configs/exp/native_molecular_administered_rate.yaml")
    group={"family_uid":MOLECULAR,"source_variant":"primary","action_indices":[0,4]}
    library=native_actuation_library(config,group)
    times=library[0].source_times
    ledger=SimpleNamespace(spent=0,maximum=1000000,planning_active=False)
    initial=torch.arange(32,dtype=torch.float64)[:,None,None,None].expand(32,6,6,26).clone()
    actual=SimpleNamespace(query_index=1,_times=times,_execution_library=library,
        _public_control=library[0],_guide=SimpleNamespace(b=1,j=32,a=1,n=32,grid={},family=MOLECULAR),
        _validity=torch.ones(1,32,1,7,6,6,26,dtype=torch.bool),_work_ledger=ledger,carry=initial.clone())
    origin=SimpleNamespace(rhs=actual,_forecast_control=library[0],_forecast_executed_uid=library[0].regimen_uid)
    calls=[]
    class Solver:
        maximum_step_hours=1.
        def prepare_future_control_branch(self,state,inputs,normalization,*,executed_regimen_uid,
                                         candidate_regimen_uids,public_control):
            assert public_control.administered_mass(state.query_index)==executed_prefix(public_control,state.query_index).administered_mass_mg_per_kg
            calls.append((state.query_index,public_control, state.carry.clone()))
            return public_control
        def forecast_work(self,state,*,query_count):return 4*32*(query_count-1-state.query_index)
    solver=Solver()
    class Model:
        field_shape=(6,6,26)
        def parameters(self):return iter(())
        def buffers(self):return iter(())
        def model_only_continuation(self,current):
            return SimpleNamespace(rhs=SimpleNamespace(**{**vars(current.rhs),"carry":current.rhs.carry.clone()}),
                _forecast_control=current._forecast_control,_forecast_executed_uid=current._forecast_executed_uid)
        def prepare_forecast_control(self,current,inputs,normalization,parent,request):
            assert request.prefix==executed_prefix(parent,current.rhs.query_index)
            joined=compose_actuation(request,library)
            assert joined.administered_mass(current.rhs.query_index)==parent.administered_mass(current.rhs.query_index)
            state=SimpleNamespace(**{**vars(current.rhs),"carry":current.rhs.carry.clone(),
                "_filter_binding":{"regimen_uids":[joined.regimen_uid]}})
            return SimpleNamespace(rhs=state,_forecast_control=joined,_forecast_executed_uid=joined.regimen_uid)
        def advance_forecast(self,current,normalization,*,query_count):
            state=current.rhs;start=state.query_index;end=query_count-1
            carry=state.carry.clone();fields=[]
            for index in range(start,end):
                carry=carry+current._forecast_control.administration_rates_mg_per_kg_hour[index]
                fields.append(carry.clone())
            ledger.spent+=solver.forecast_work(state,query_count=query_count)
            next_state=SimpleNamespace(**{**vars(state),"query_index":end,"carry":carry})
            result=SimpleNamespace(rhs=next_state,_forecast_control=current._forecast_control,
                _forecast_executed_uid=current._forecast_executed_uid)
            return result,torch.stack(fields,1)
    model=Model();model.config=config;model.solver=solver
    from Libs.tests.test_native_conditional_readout import reference
    readout=replace(reference().readout,region_ids=tuple(config["native_observable_bridge"]["quantity_profile"]["region_id"]),
        compartment_ids=tuple(config["native_observable_bridge"]["quantity_profile"]["compartment_id"]))
    episode=SimpleNamespace(model=model,continuation=origin,adaptive_reference=None,
        _execution_inputs=None,normalization=None,readout=readout,
        recorded_model_history=lambda:initial[:,None].clone())
    tree=compile_temporal_world_plans(episode,config,depth=2)
    before=actuation_payload(actual._public_control)
    evaluate_temporal_world_plans(episode,tree,lambda values,mask,grid,burden:{"loss":values[...,0].mean(dim=(-1,-2,-3))+burden})
    assert tree.losses.shape[0]==32 and tree.depth==2
    assert {index for index,_,_ in calls}=={1,2,3}
    assert all(public.administration is not None for _,public,_ in calls)
    first=tree.edges[0];child=first.child.edges[0]
    assert torch.equal(child.origin.rhs.carry,first.arrival.rhs.carry)
    assert child.request.prefix.administered_mass_mg_per_kg==first.public.administered_mass(2)
    assert torch.equal(actual.carry,initial) and actual.query_index==1
    assert actuation_payload(actual._public_control)==before
    invalid=copy.deepcopy(config);invalid.pop("native_molecular_administration")
    with pytest.raises(ValueError):compile_temporal_world_plans(episode,invalid,depth=2)
