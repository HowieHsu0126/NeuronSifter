"""Public t0 plan selection on retained artificial inputs and original source laws."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from Libs.tests.test_native_mechanistic_residual import cases, execution_cases, FAMILIES
from Libs.tests.test_native_residual_control import _independent_nominal_world
from Libs.models.modules.native_mechanistic_residual import NativeMechanisticResidual, NativeResidualOnlineEpisode
from Libs.utils.data.native_actuation import NativeObservationIdentity, native_actuation_library
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_intervention_decisions import NativePublicCNSLoss
from Libs.utils.evaluation.native_residual_control_evaluation import (
    residual_initial_mode, run_native_temporal_residual_episode, realized_native_residual_control,
)
from Libs.utils.model.native_arrived_observation import native_cns_readout
from Libs.utils.model.native_scientific_source import selected_group_definitions
from Libs.utils.model.native_temporal_query import (
    select_initial_public_plan, compile_temporal_world_plans, evaluate_temporal_world_plans,
)


def initial_case(execution_cases, family=FAMILIES[0]):
    config, inputs, norm = execution_cases["validation", family]
    config = copy.deepcopy(config)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    for key in ("native_arrived_observation_filter", "native_mechanistic_residual_online",
                "native_future_controls", "native_residual_control", "native_temporal_query"):
        config[key] = copy.deepcopy(current[key])
        if key in current["schemas"]:
            config["schemas"][key] = copy.deepcopy(current["schemas"][key])
    settings = config["native_controlled_comparators"]["references"]["mechanistic_residual"]["hyperparameters"]
    settings.update(encoder_widths=[4,8,12], bottleneck_width=16)
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        model = NativeMechanisticResidual(settings, inputs, config).eval()
    group = next(row for row in selected_group_definitions(config, "validation") if row["family_uid"] == family)
    episode = NativeResidualOnlineEpisode(model, inputs, norm, native_cns_readout(config, group),
                                         executed_regimen_uid=None)
    library = native_actuation_library(config, group)
    reference = json.loads((Path(os.environ["NRNSIFT_RHS_FIXTURE"])/"public_cns_reference.json").read_text())
    return episode, library, group, NativePublicCNSLoss(reference, config)


def test_initial_mode_keeps_fixed_control_and_rejects_ambiguity():
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    assert residual_initial_mode(config, initial_index=2) == "explicit_fixed_index"
    assert residual_initial_mode(config, initial_index=None, optimize_initial_plan=True) == "public_open_loop_plan"
    with pytest.raises(ValueError, match="ambiguous"):
        residual_initial_mode(config, initial_index=0, optimize_initial_plan=True)
    with pytest.raises(ValueError, match="explicit --initial-action-index"):
        residual_initial_mode(config, initial_index=None)
    config["native_residual_control"]["initial_selection_mode"] = "public_open_loop_plan"
    assert residual_initial_mode(config, initial_index=None) == "public_open_loop_plan"
    with pytest.raises(ValueError, match="ambiguous"):
        residual_initial_mode(config, initial_index=0)



def test_historical_fixed_v1_config_pair_remains_compatible():
    from Libs.utils.evaluation.native_residual_control_evaluation import residual_control_policy
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    original = copy.deepcopy(config)
    fields = ("initial_selection_mode", "optimized_initial_selection", "readout_mode")
    for field in fields:
        config["native_residual_control"].pop(field)
        config["schemas"]["native_residual_control"]["required_fields"].remove(field)
    assert residual_initial_mode(config, initial_index=0) == "explicit_fixed_index"
    assert residual_control_policy(config)["initial_selection_mode"] == "explicit_fixed_index"
    assert residual_control_policy(config)["readout_mode"] == "sensor_only"
    assert not any(field in config["native_residual_control"] for field in fields)
    for field in fields:
        malformed = copy.deepcopy(original)
        malformed["native_residual_control"].pop(field)
        with pytest.raises(ValueError, match="original-grid"):
            residual_control_policy(malformed)
    config["native_residual_control"]["optimized_initial_selection"] = "unbound"
    with pytest.raises(ValueError, match="unbound"):
        residual_control_policy(config)


def test_optimized_initial_cli_flag_rejects_unrelated_commands(monkeypatch):
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    base = ["bridge", "--protocol-scope", "manuscript-native", "--baseline-name", "residual_mpc",
            "--output-dir", "unused", "--optimize-initial-plan"]
    monkeypatch.setattr(sys, "argv", [*base, "--command", "evaluate_baseline"])
    with pytest.raises(SystemExit) as caught:
        cli.parse_args()
    assert caught.value.code == 2
    monkeypatch.setattr(sys, "argv", [*base, "--command", "evaluate_residual_controls"])
    assert cli.parse_args().optimize_initial_plan
    with pytest.raises(ValueError, match="requires evaluate_residual_controls"):
        cli.run_native_command(SimpleNamespace(command="prepare_inputs", optimize_initial_plan=True), {})


def test_all_initial_actions_complete_leaves_single_ledger_order_independence(execution_cases):
    first, library, _, loss = initial_case(execution_cases)
    second, second_library, _, second_loss = initial_case(execution_cases)
    origin = first.continuation
    initial = origin.rhs.cns_state.clone()
    logs = origin.rhs._log_weights.clone()
    rng = torch.random.get_rng_state().clone()
    versions = tuple(parameter._version for parameter in first.model.parameters())
    assert first.optimizer is None
    with pytest.raises(ValueError, match="Initial public selection"):
        first.step(lambda request: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Public initial planning must not use RNG, observations or optimization")
    with patch("torch.randn", forbidden), patch.object(first, "step", forbidden), patch.object(first, "_adapt", forbidden):
        report, node = select_initial_public_plan(first, library, loss, first.model.config)
    reverse, _ = select_initial_public_plan(second, second_library, second_loss, second.model.config,
                                            evaluation_order=reversed(range(len(library))))
    evidence = os.environ.get("NRNSIFT_INITIAL_PLAN_EVIDENCE")
    if evidence:
        destination = Path(evidence); destination.mkdir(parents=True, exist_ok=True)
        (destination/"all_action_selection.json").write_text(json.dumps({"forward": report, "reverse": reverse}, indent=2)+"\n")
    assert report["status"] == reverse["status"] == "selected"
    assert [row["regimen_uid"] for row in report["candidates"]] == [action.regimen_uid for action in library]
    assert report["selected_regimen_uid"] == reverse["selected_regimen_uid"]
    assert report["selected_plan_index"] == reverse["selected_plan_index"]
    assert report["actual_comparison_rhs_work"] == sum(row["actual_rhs_work"] for row in report["candidates"])
    assert report["actual_rhs_work_after_planning"] == first.continuation.rhs._work_ledger.spent
    assert first.continuation.rhs._work_ledger is origin.rhs._work_ledger
    assert origin.rhs.query_index == first.continuation.rhs.query_index == 0
    assert first.continuation.rhs._guide.a == 1 and first.continuation.rhs.rhs_python_calls == 0
    assert torch.equal(origin.rhs.cns_state, initial) and torch.equal(origin.rhs._log_weights, logs)
    assert torch.equal(first.continuation.rhs._log_weights, logs)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert versions == tuple(parameter._version for parameter in first.model.parameters())
    assert first.observations_seen == first.optimizer_updates == 0
    assert not report["full_residual_mpc_eligible"] and not report["initial_joint_future_query_value_optimized"]
    for left, right in zip(report["candidates"], reverse["candidates"]):
        assert left["predictive_risk"] == right["predictive_risk"]
        assert left["terminal_plans"] == right["terminal_plans"]
        assert left["complete_terminal_plan_count"] == len(left["terminal_plans"])
        assert left["complete_terminal_plan_count"] > 1
        assert all(row["burden"]["status"] == "defined" for row in left["terminal_plans"])
    assert node.public == first.continuation.rhs._public_control
    assert node.edges[0].arrival.rhs._public_control is None
    assert node.edges[0].arrival.rhs.query_index == 1


def test_complete_initial_budget_blocks_before_work_or_optimizer(execution_cases):
    episode, library, _, loss = initial_case(execution_cases, FAMILIES[-1])
    ledger = episode.continuation.rhs._work_ledger
    maximum = ledger.maximum
    report, node = select_initial_public_plan(episode, library, loss, episode.model.config,
                                               reuse_identical_siblings=False)
    evidence = os.environ.get("NRNSIFT_INITIAL_PLAN_EVIDENCE")
    if evidence:
        destination = Path(evidence); destination.mkdir(parents=True, exist_ok=True)
        (destination/"long_cns_initial_budget.json").write_text(json.dumps(report, indent=2)+"\n")
    assert report["status"] == "blocked", report
    assert report["reason"] == "complete_initial_public_comparison_exceeds_shared_rhs_budget"
    assert report["required_rhs_work"] == sum(row["required_rhs_work"] for row in report["candidates"])
    assert report["required_rhs_work"] > maximum == 4000000
    assert node is None and ledger.spent == 0 and ledger.maximum == maximum
    assert episode.optimizer is None and episode.executed_regimen_uid is None
    assert episode.continuation.rhs._public_control is None


@pytest.mark.parametrize("family", [FAMILIES[0], FAMILIES[-1]])
def test_initial_candidate_retains_original_t0_events_forcing_and_rejects_advanced_carry(execution_cases, family):
    episode, library, _, loss = initial_case(execution_cases, family)
    origin = episode.continuation
    index = len(library)-1
    candidate = episode.model.prepare_initial_forecast(origin, episode._execution_inputs,
        episode.normalization, library, library[index], episode.readout)
    guide = origin.rhs._guide
    expected = episode.model.solver.current_particle_fields(origin.rhs)[0, :, index]
    assert torch.equal(episode.model.solver.current_particle_fields(candidate.rhs)[0, :, 0], expected)
    assert candidate.rhs._work_ledger is origin.rhs._work_ledger
    assert candidate.rhs.query_index == candidate.rhs.interval_index == candidate.rhs.rhs_python_calls == 0
    assert candidate.rhs._public_control is None and candidate.rhs._execution_model_uid is None
    if candidate.rhs.event_times is not None:
        assert not candidate.rhs.applied_events.any()
        assert (candidate.rhs.event_times == 0).any()
        events = origin.rhs.event_times.reshape(1,32,guide.a,-1)[:, :, index].reshape(32,-1)
        assert torch.equal(candidate.rhs.event_times, events[:, :candidate.rhs.event_times.shape[-1]])
    else:
        stimulus = origin.rhs._guide.stimulus.reshape(1,32,guide.a,-1)[:, :, index].reshape(32,-1)
        assert torch.equal(candidate.rhs._guide.stimulus, stimulus)
    advanced, _ = episode.model.advance_forecast(candidate, episode.normalization, query_count=2)
    with pytest.raises(ValueError, match="Advanced|changed|pristine"):
        episode.install_initial_selection(advanced, library, library[index])
    assert episode.executed_regimen_uid is None and episode.optimizer is None
    assert origin.rhs.query_index == 0
    episode.install_initial_selection(candidate, library, library[index])
    assert episode.continuation.rhs.query_index == 0
    assert episode.continuation.rhs._public_control == library[index]
    if candidate.rhs.event_times is not None:
        assert not episode.continuation.rhs.applied_events.any()
    else:
        assert torch.equal(episode.continuation.rhs._guide.stimulus, stimulus)


def test_public_selected_t0_forest_then_actual_original_tail(execution_cases, tmp_path):
    episode, library, group, loss = initial_case(execution_cases)
    report, node = select_initial_public_plan(episode, library, loss, episode.model.config)
    assert report["status"] == "selected"
    identity = NativeObservationIdentity.from_context(episode.model.solver.filter_context(episode.continuation.rhs))
    plant = _independent_nominal_world(episode.model.config, group, library, identity,
        report["selected_initial_original_library_index"], 9151)
    before = episode.continuation.rhs._work_ledger.spent
    result = run_native_temporal_residual_episode(episode, loss, plant.acquire, plant.switch,
        plant._complete_planned_tail, episode.model.config, initial_selection=report, initial_node=node)
    realized = realized_native_residual_control(plant, result["status"], loss, episode.model.config)
    output = Path(os.environ.get("NRNSIFT_INITIAL_PLAN_EVIDENCE", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    (output/"actual_initial_plan_episode.json").write_text(json.dumps({"initial_selection": report,
        "episode": result, "realized": realized}, indent=2)+"\n")
    assert result["status"] == "component_complete", result
    assert realized["source_horizon_complete"] and realized["realized_full_grid_public_loss"] is not None
    assert result["initial_execution"] == "public_open_loop_plan_selected_t0"
    assert result["initial_selection"] == report
    assert result["decisions"][0]["model_rhs_work_after_planning"] == before
    assert not result["full_residual_mpc_eligible"] and not result["formal_qualification"]
    output = Path(os.environ.get("NRNSIFT_INITIAL_PLAN_EVIDENCE", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    (output/"actual_initial_plan_episode.json").write_text(json.dumps({"initial_selection": report,
        "episode": result, "realized": realized}, indent=2)+"\n")


def test_evaluator_private_callback_follows_public_selection(monkeypatch, tmp_path):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    declaration = {"family_uid": FAMILIES[0]}
    dataset = SimpleNamespace(group_uids=["group"])
    model = torch.nn.Linear(1, 1)
    model.prediction_kind = "offline_one_step_mechanistic_residual_conditional_mean"
    state = SimpleNamespace()
    episode = SimpleNamespace(model=SimpleNamespace(solver=SimpleNamespace(filter_context=lambda rhs: {})),
                              continuation=SimpleNamespace(rhs=state))
    order = []
    monkeypatch.setattr(consumer, "predose_source_admission", lambda *a: {"scope": "artificial_software_fixture"})
    monkeypatch.setattr(consumer, "NativePublicCNSLoss", lambda *a: object())
    monkeypatch.setattr(consumer, "load_family_registries", lambda *a: {})
    monkeypatch.setattr(consumer, "scoped_group_config", lambda *a: config)
    monkeypatch.setattr(consumer, "load_native_prediction_inputs", lambda *a: object())
    monkeypatch.setattr(consumer, "native_actuation_library", lambda *a: (0,1))
    monkeypatch.setattr(consumer, "native_cns_readout", lambda *a: object())
    monkeypatch.setattr(consumer, "NativeResidualOnlineEpisode", lambda *a, **k: episode)
    monkeypatch.setattr(consumer.NativeObservationIdentity, "from_context", lambda *a: "public_identity")
    def select(*a):
        order.append("public_selection")
        return {"status": "selected", "selected_initial_original_library_index": 1}, object()
    monkeypatch.setattr(consumer, "select_initial_public_plan", select)
    def private(*a):
        assert order == ["public_selection"] and a[3] == 1
        order.append("private_world")
        return SimpleNamespace(observation_identity="public_identity", switch=lambda *a: None)
    monkeypatch.setattr(consumer, "run_native_temporal_residual_episode", lambda *a, **k: {"status": "component_complete"})
    monkeypatch.setattr(consumer, "realized_native_residual_control", lambda *a: {})
    result = consumer.evaluate_native_residual_controls(config, model, SimpleNamespace(field_normalization=None), {},
        dataset, {"group": declaration}, {"reference_id": "public"}, tmp_path, optimize_initial_plan=True, world_factory=private)
    assert order == ["public_selection", "private_world"] and result["status"] == "component_complete"


def test_long_cns_complete_initial_exact_sibling_reuse_within_original_cap(execution_cases):
    episode, library, _, loss = initial_case(execution_cases, FAMILIES[-1])
    assert episode.optimizer is None and episode.optimizer_updates == 0
    report, node = select_initial_public_plan(episode, library, loss, episode.model.config)
    evidence = os.environ.get("NRNSIFT_INITIAL_PLAN_EVIDENCE")
    if evidence:
        Path(evidence).mkdir(parents=True,exist_ok=True)
        (Path(evidence)/"long_cns_exact_shared_selection.json").write_text(json.dumps(report,indent=2)+"\n")
    assert report["status"] == "selected", report
    assert report["unshared_rhs_work"] == 7180800
    assert report["required_rhs_work"] == report["actual_comparison_rhs_work"] == 2982400
    assert report["remaining_rhs_work"] == 4000000
    assert sum(row["complete_terminal_plan_count"] for row in report["candidates"]) == 22
    assert [row["unshared_rhs_work"] for row in report["candidates"]] == [2931200,1318400,2931200]
    assert [row["actual_rhs_work"] for row in report["candidates"]] == [1318400,345600,1318400]
    assert episode.continuation.rhs._work_ledger.spent == 2982400
    assert episode.continuation.rhs.query_index == 0 and episode.optimizer is not None
    assert episode.optimizer_updates == 0 and episode.observations_seen == 0
    assert node.public == episode.continuation.rhs._public_control
    assert not report["formal_qualification"] and not report["full_residual_mpc_eligible"]


def test_exact_sibling_reuse_real_nonlinear_fields_losses_and_public_order(execution_cases):
    from Libs.utils.data.native_actuation import NativeActuationRequest, compose_actuation, executed_prefix
    first, library, _, loss = initial_case(execution_cases, FAMILIES[-1])
    second, other_library, _, other_loss = initial_case(execution_cases, FAMILIES[-1])
    # Nonzero fixed residual ensures reuse compares the hybrid law, not zero-head mechanics alone.
    with torch.no_grad():
        first.model.output.bias[first.model.solver.config["controlled_cns_source"]["grid"]["species_id"].index("neuroinflammation")] = .0001
        second.model.load_state_dict(first.model.state_dict())
    origins=[]
    for episode, actions, objective, reuse in [(first,library,loss,True),(second,other_library,other_loss,False)]:
        origin=episode.model.prepare_initial_forecast(episode.continuation,episode._execution_inputs,
            episode.normalization,actions,actions[0],episode.readout)
        # Real first interval produces identical complete carries independently.
        arrived,_=episode.model.advance_forecast(origin,episode.normalization,query_count=2)
        # Bounded whole-library forecast to the next original clock; complete
        # sibling request/provenance and nonlinear hybrid arithmetic remain real.
        branch=episode.model.solver.prepare_future_control_branch(arrived.rhs,episode._execution_inputs,
            episode.normalization,executed_regimen_uid=arrived._forecast_executed_uid)
        prepared=[]
        for action in actions[:2]:
            request=NativeActuationRequest(executed_prefix(arrived._forecast_control,1),action)
            prepared.append(episode.model.prepare_forecast_control(arrived,episode._execution_inputs,
                episode.normalization,branch,request))
        before=arrived.rhs._work_ledger.spent
        advanced,fields=episode.model.advance_forecast(prepared[0],episode.normalization,query_count=3)
        if reuse:
            alternative=episode.model._reuse_forecast_result(prepared[1],prepared[0],advanced,
                parameter_versions=tuple(value._version for value in (*episode.model.parameters(), *episode.model.buffers())))
            other_fields=episode.model.solver.current_particle_fields(alternative.rhs)
        else:
            alternative,tail=episode.model.advance_forecast(prepared[1],episode.normalization,query_count=3)
            other_fields=tail.reshape(1,32,1,1,*episode.model.field_shape)[:,:,:,0]
        assert prepared[0]._forecast_control.regimen_uid != prepared[1]._forecast_control.regimen_uid
        assert advanced._forecast_binding != alternative._forecast_binding
        assert alternative._forecast_control == compose_actuation(
            NativeActuationRequest(executed_prefix(arrived._forecast_control,1),actions[1]),actions)
        origins.append((advanced,alternative,fields,other_fields,arrived.rhs._work_ledger.spent-before))
    left,right=origins
    for li,ri in zip(left[:2],right[:2]):
        for name in ("cns_state","event_times","event_doses","applied_events"):
            assert torch.equal(getattr(li.rhs,name),getattr(ri.rhs,name))
        assert torch.equal(li.program,ri.program) and torch.equal(li.static,ri.static)
    assert torch.equal(left[2],right[2]) and torch.equal(left[3],right[3])
    assert left[4] == 12800 and right[4] == 25600


def test_exact_sibling_reuse_rejects_changed_numerical_origin(execution_cases):
    from dataclasses import replace
    from Libs.utils.data.native_actuation import NativeActuationRequest,executed_prefix
    episode,library,_,_=initial_case(execution_cases,FAMILIES[-1])
    origin=episode.model.prepare_initial_forecast(episode.continuation,episode._execution_inputs,
        episode.normalization,library,library[0],episode.readout)
    arrived,_=episode.model.advance_forecast(origin,episode.normalization,query_count=2)
    branch=episode.model.solver.prepare_future_control_branch(arrived.rhs,episode._execution_inputs,
        episode.normalization,executed_regimen_uid=arrived._forecast_executed_uid)
    targets=[episode.model.prepare_forecast_control(arrived,episode._execution_inputs,episode.normalization,branch,
        NativeActuationRequest(executed_prefix(arrived._forecast_control,1),action)) for action in library]
    result,_=episode.model.advance_forecast(targets[0],episode.normalization,query_count=3)
    before=result.rhs._work_ledger.spent
    versions=tuple(value._version for value in (*episode.model.parameters(), *episode.model.buffers()))
    with pytest.raises(ValueError,match="controls|conditioning"):
        episode.model._reuse_forecast_result(targets[2],targets[0],result,parameter_versions=versions)
    changed=replace(targets[1],rhs=episode.model.solver.fork_continuation(targets[1].rhs))
    changed.rhs.cns_state[0,0,0] += 1e-6
    with pytest.raises(ValueError,match="complete numerical origin"):
        episode.model._reuse_forecast_result(changed,targets[0],result,parameter_versions=versions)
    with torch.no_grad():
        episode.model.output.bias.add_(.00001)
    with pytest.raises(ValueError,match="unchanged evaluation model parameters"):
        episode.model._reuse_forecast_result(targets[1],targets[0],result,parameter_versions=versions)
    assert result.rhs._work_ledger.spent == before


def test_complete_one_initial_tree_exact_reuse_matches_unshared_evaluator(execution_cases):
    episode,library,_,loss=initial_case(execution_cases,FAMILIES[-1])
    with torch.no_grad():
        episode.model.output.bias[episode.model.config["controlled_cns_source"]["grid"]["species_id"].index("neuroinflammation")] = .0001
    # Each independently priced full tree fits the original4m ceiling. A fresh
    # ledger separates the reference execution from the shared execution.
    origins=[]
    results=[]
    for reuse in (True,False):
        fresh=episode.model.solver.fork_continuation(episode.continuation.rhs)
        from Libs.models.modules.native_frozen_rhs_solver import NativeRHSWorkLedger
        from dataclasses import replace
        fresh._work_ledger=NativeRHSWorkLedger(4000000)
        parent=replace(episode.continuation,rhs=fresh)
        origin=episode.model.prepare_initial_forecast(parent,episode._execution_inputs,episode.normalization,
            library,library[0],episode.readout)
        node=compile_temporal_world_plans(episode,episode.model.config,depth=2,initial_origin=origin,
                                         reuse_identical_siblings=reuse)
        evaluate_temporal_world_plans(episode,node,loss,initial_origin=origin)
        assert origin.rhs._work_ledger.spent == node.required_rhs_work == (1318400 if reuse else 2931200)
        results.append(node)
        origins.append(origin)
    left,right=results
    assert torch.equal(left.losses,right.losses)
    assert left.plans == right.plans and len(left.plans)==9
    def compare(a,b):
        assert a.public==b.public and torch.equal(a.losses,b.losses) and a.plans==b.plans
        assert len(a.edges)==len(b.edges)
        for x,y in zip(a.edges,b.edges):
            assert x.request==y.request and x.public==y.public and x.rhs_work==y.rhs_work
            assert torch.equal(x.fields,y.fields)
            for name in ("cns_state","event_times","event_doses","applied_events"):
                assert torch.equal(getattr(x.arrival.rhs,name),getattr(y.arrival.rhs,name))
            assert x.arrival._forecast_binding==y.arrival._forecast_binding
            if x.child:compare(x.child,y.child)
    compare(left,right)
    logs=episode.model.solver.current_log_weights(episode.continuation.rhs)[0]
    assert int((logs.exp()@left.losses.double()).argmin()) == int((logs.exp()@right.losses.double()).argmin())


def test_joint_initial_mode_preserves_old_modes_and_binds_cli(monkeypatch):
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    assert residual_initial_mode(config, initial_index=2) == "explicit_fixed_index"
    assert residual_initial_mode(config, initial_index=None, optimize_initial_plan=True) == "public_open_loop_plan"
    assert residual_initial_mode(config, initial_index=None, optimize_initial_joint_plan=True) == "public_joint_query_control"
    with pytest.raises(ValueError, match="ambiguous"):
        residual_initial_mode(config, initial_index=None, optimize_initial_plan=True, optimize_initial_joint_plan=True)
    config["native_residual_control"].pop("joint_initial_selection")
    with pytest.raises(ValueError, match="unbound"):
        residual_initial_mode(config, initial_index=None, optimize_initial_joint_plan=True)
    monkeypatch.setattr(sys, "argv", ["bridge", "--protocol-scope", "manuscript-native",
        "--baseline-name", "residual_mpc", "--output-dir", "unused", "--command", "evaluate_residual_controls",
        "--optimize-initial-joint-plan"])
    assert cli.parse_args().optimize_initial_joint_plan


def _initial_adaptive_reference(episode, library):
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference, FixedReadoutReference
    state = episode.continuation.rhs
    context = episode.model.solver.filter_context(state)
    fixed = FixedReadoutReference(episode.readout, context["support_uid"], context["origin_posterior_uid"],
        tuple(context["sample_uids"]), tuple(tuple(row) for row in context["particle_uids"]), library[0].regimen_uid,
        json.dumps({"normalization_id":episode.normalization.identity,"checkpoint_sha256":"synthetic_initial_predictor"}),
        episode.normalization.identity, "synthetic_software_scale", (.01,)*26,
        tuple(episode.normalization.scale.tolist()), max(state._times), tuple(library[0].source_times),
        tuple(state._times), library[0].source_time_unit, 0)
    return AdaptiveReadoutReference(fixed, library, library[0].regimen_uid)


def test_pristine_adaptive_initial_reference_rebind_preserves_source_prior_and_history(execution_cases):
    from dataclasses import asdict, replace
    episode, library, _, _ = initial_case(execution_cases)
    episode._adaptive_reference = _initial_adaptive_reference(episode, library)
    reference = episode.adaptive_reference
    origin = episode.model.prepare_initial_forecast(episode.continuation, episode._execution_inputs,
        episode.normalization, library, library[2], episode.readout)
    candidate, history, context = episode.prospective_adaptive_temporal_readout(initial_origin=origin)
    assert history == () and context.original_action == context.executed_program == library[2]
    left, right = asdict(reference), asdict(candidate)
    for item in (left, right):
        item.pop("initial_action_uid")
        item["fixed"].pop("executed_regimen_uid")
    assert left == right and episode.adaptive_reference is reference
    changed = replace(origin, program=origin.program+1.)
    with pytest.raises(ValueError):
        episode.prospective_adaptive_temporal_readout(initial_origin=changed)
    episode._accepted_snapshot_authority = (object(),)
    with pytest.raises(ValueError, match="pristine"):
        episode.prospective_adaptive_temporal_readout(initial_origin=origin)
    episode._accepted_snapshot_authority = episode.continuation._accepted_readout_snapshots
    episode.install_initial_selection(origin, library, library[2], select_adaptive_initial_reference=True)
    assert episode.adaptive_reference == candidate and episode.optimizer_updates == 0
    with pytest.raises(ValueError, match="pristine"):
        episode.prospective_adaptive_temporal_readout(initial_origin=origin)


def test_joint_initial_actual_forests_and_selected_consumer_keep_single_ledger(execution_cases, tmp_path):
    episode, library, group, loss = initial_case(execution_cases)
    episode._adaptive_reference = _initial_adaptive_reference(episode, library)
    original = episode.adaptive_reference
    before_logs = episode.continuation.rhs._log_weights.clone()
    report, node = select_initial_public_plan(episode, library, loss, episode.model.config,
                                             selection_mode="public_joint_query_control")
    assert report["status"] == "selected", report
    assert report["initial_joint_future_query_value_optimized"] and not report["full_residual_mpc_eligible"]
    assert len(report["candidates"]) == len(library)
    assert all(row["complete_terminal_plan_count"] == len(row["terminal_plans"]) > 0 for row in report["candidates"])
    remaining = episode.continuation.rhs._work_ledger.maximum-episode.continuation.rhs._work_ledger.spent
    assert all(row["remaining_rhs_work_at_joint_comparison"] == remaining for row in report["candidates"])
    chosen = min(range(len(library)), key=lambda index: report["candidates"][index]["joint_decision_loss"])
    assert chosen == report["selected_initial_original_library_index"]
    assert episode.adaptive_reference.fixed.residual_scales == original.fixed.residual_scales
    assert torch.equal(episode.continuation.rhs._log_weights, before_logs)
    identity = NativeObservationIdentity.from_context(episode.model.solver.filter_context(episode.continuation.rhs))
    plant = _independent_nominal_world(episode.model.config, group, library, identity, chosen, 9151)
    planning_work = episode.continuation.rhs._work_ledger.spent
    result = run_native_temporal_residual_episode(episode, loss, plant.acquire, plant.switch,
        plant._complete_planned_tail, episode.model.config, initial_selection=report, initial_node=node)
    assert result["status"] == "component_complete", result
    assert result["decisions"][0]["model_rhs_work_after_planning"] == planning_work
    assert result["initial_execution"] == "public_joint_query_control_selected_t0"
    assert not result["formal_qualification"] and not result["planner_anticipates_residual_retraining"]
    (tmp_path/"actual_initial_joint_episode.json").write_text(json.dumps({"initial":report,"episode":result},indent=2))
