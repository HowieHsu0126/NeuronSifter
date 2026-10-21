"""Actual residual selection and private original-bundle consumption on artificial fixtures."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Libs.tests.test_native_mechanistic_residual import (
    FAMILIES, cases, execution_cases, _actual_execution_case,
)
from Libs.utils.data.native_actuation import NativeActuation, NativeDoseEvent
from Libs.utils.data.controlled_world_contracts import select_family_config
from Libs.utils.data.native_joint_source_storage import write_joint_source_reference
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_intervention_decisions import NativePublicCNSLoss, composed_action_burden
from Libs.utils.exp.experiment_manifests import native_training_seed_values
from Libs.utils.evaluation.native_residual_control_evaluation import (
    NativeRetainedSourceWorlds, residual_control_policy, run_native_residual_control_episode,
)
from Libs.utils.model.native_arrived_observation import predict_native_readout
from Libs.utils.model.native_future_controls import public_suffix_candidates


def _control_case(execution_cases, family=FAMILIES[0], initial_index=2):
    original, episode, plant, library = _actual_execution_case(
        execution_cases, family, initial_control_index=initial_index)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    episode.model.config["native_residual_control"] = current["native_residual_control"]
    episode.model.config["schemas"]["native_residual_control"] = current["schemas"]["native_residual_control"]
    return original, episode, plant, library


def _loss(episode):
    reference = json.loads((Path(os.environ["NRNSIFT_RHS_FIXTURE"])/"public_cns_reference.json").read_text())
    return NativePublicCNSLoss(reference, episode.model.config)


@pytest.mark.parametrize("family", [FAMILIES[0], FAMILIES[1], FAMILIES[3]])
def test_prepared_residual_forecast_is_unexecuted_and_keeps_actual_history(execution_cases, family):
    _, episode, plant, library = _control_case(execution_cases, family, initial_index=0)
    episode.step(plant.acquire)
    history = episode.recorded_model_history()
    state = episode.continuation.rhs
    current = state.cns_state.clone()
    prefix, rng = plant.prefix, plant._generator.get_state().clone()
    candidate = next(row for row in public_suffix_candidates(state) if row["status"] == "legal")
    branch = episode.plan_controls([candidate["regimen_uid"]])
    spent = state._work_ledger.spent
    forecast, fields, public = episode.model.forecast_public_control(
        episode.continuation, episode._execution_inputs, episode.normalization, branch, candidate["request"])
    assert fields.shape == (32, len(state._times)-2, *episode.model.field_shape)
    assert public == candidate["public"] and forecast.rhs._public_control is None
    assert forecast.rhs._execution_model_uid is None and forecast.rhs._work_ledger is state._work_ledger
    assert state._work_ledger.spent > spent and state.query_index == branch.query_index == 1
    assert torch.equal(state.cns_state, current) and torch.equal(episode.recorded_model_history(), history)
    assert plant.prefix == prefix and torch.equal(plant._generator.get_state(), rng)
    with pytest.raises(ValueError):
        predict_native_readout(episode.model.solver, forecast.rhs, episode.readout,
                              executed_regimen_uid=forecast.rhs._filter_binding["regimen_uids"][0])
    # A forecasted result must still fail activation; only a fresh branch is eligible.
    with pytest.raises(ValueError, match="forecasted|branch"):
        episode.activate_selected_control(forecast.rhs, candidate["request"], plant.switch)


def test_model_selected_episode_scores_full_grid_common_past_and_actual_controls(execution_cases, tmp_path):
    original, episode, plant, library = _control_case(execution_cases)
    parameters = {name: value.clone() for name, value in original.state_dict().items()}
    actual_loss = _loss(episode)
    calls = []
    class RecordedLoss:
        reference_id = actual_loss.reference_id
        def __call__(self, fields, mask, grid, costs):
            past = episode.recorded_model_history()
            expected = past[:, None].expand(-1, fields.shape[2], *past.shape[1:])
            assert torch.equal(fields[:, 0, :, :past.shape[1]], expected)
            assert grid == episode._execution_inputs.grid and fields.shape[-4] == len(grid["time_hours"])
            calls.append({"query_index": episode.continuation.rhs.query_index,
                          "original_times": list(grid["time_hours"]), "costs": costs.cpu().tolist()})
            return actual_loss(fields, mask, grid, costs)
    report = run_native_residual_control_episode(episode, RecordedLoss(), plant.acquire, plant.switch, episode.model.config)
    assert report["status"] == "component_complete", report
    assert report["decision_count"] == report["actuator_calls"] == len(calls) == 5
    assert report["observations_seen"] == plant._query == 6 and report["optimizer_updates"] == 20
    assert report["control_change_count"] == sum(row["public_control_changed"] for row in report["decisions"])
    for decision in report["decisions"]:
        scored = [row for row in decision["candidates"] if row["status"] == "scored"]
        selected = min(scored, key=lambda row: row["predictive_risk"])
        assert decision["selected_regimen_uid"] == selected["regimen_uid"]
        assert [row["regimen_uid"] for row in decision["candidates"]] == [row.regimen_uid for row in library]
        assert len(decision["current_particle_weights"]) == 32
        assert decision["conditional_draw_count"] == 1 and not decision["independent_predictive_error_calibration"]
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in parameters.items())
    report.update(public_loss_calls=calls, actual_consumed_intervals=list(plant.prefix.interval_values),
                  source_rhs_calls=plant._work["nfev"], artificial_private_world="explicit_nominal_not_posterior_tuple")
    output = Path(os.environ.get("NRNSIFT_RESIDUAL_CONTROL_EVIDENCE", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    (output/"model_selected_control.json").write_text(json.dumps(report, indent=2)+"\n")


def test_composed_burden_preserves_actual_wave_and_event_amounts_and_blocks_molecular(monkeypatch):
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    def public(family, kind, unit, values=(), events=(), times=(0., 2., 4.)):
        return NativeActuation("source", "snapshot", family, "program", "hour" if events or family == FAMILIES[2] else "ms",
                               times, kind, unit, values, events)
    bounds = select_family_config(config, FAMILIES[0])["support_domain"]
    maximum, duration = bounds["supported_intensity"]["maximum"], bounds["supported_duration"]["maximum"]
    stimulation = public(FAMILIES[0], "stimulation", "uA", (maximum/4, maximum/2),
                         times=(0., duration/4, duration/2))
    result = composed_action_burden(stimulation, config)
    assert result["actual_amount"] == pytest.approx(3*maximum*duration/16) and result["status"] == "defined"
    assert result["normalized_burden"] == pytest.approx(3/16)
    assert composed_action_burden(replace(stimulation, interval_values=(-maximum/4, maximum/2)), config)["normalized_burden"] is None
    dose = public(FAMILIES[3], "dose_events", "mg/kg", events=(NativeDoseEvent(0., .01, .5), NativeDoseEvent(2., .02, .25)))
    result = composed_action_burden(dose, config)
    assert result["actual_amount"] == .01 and result["normalized_burden"] == .0025
    molecular = public(FAMILIES[2], "model_driving_level", "mg/kg", (.1, .2))
    result = composed_action_burden(molecular, config)
    assert result["status"] == "molecular_delivered_dose_mapping_unbound" and result["actual_amount"] is None
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    mixed = [{"regimen_uid": "positive", "status": "legal", "public": stimulation, "request": None},
             {"regimen_uid": "signed", "status": "legal", "public": replace(stimulation, interval_values=(-maximum/4, maximum/2)), "request": None}]
    monkeypatch.setattr(consumer, "public_suffix_candidates", lambda state: mixed)
    def forbidden(*args):
        raise AssertionError("Undefined legal candidate burden must block before forecasting")
    episode = SimpleNamespace(status="active", continuation=SimpleNamespace(rhs=SimpleNamespace(query_index=1)),
                              recorded_model_history=forbidden, plan_controls=forbidden)
    result, request, branch = consumer.select_native_residual_control(episode, SimpleNamespace(reference_id="public"), config)
    assert result["status"] == "blocked" and result["reason"] == "legal_candidate_burden_undefined"
    assert result["blocked_candidates"] == ["signed"] and request is None and branch is None


def test_molecular_control_is_explicitly_blocked_before_observing(execution_cases):
    _, episode, plant, _ = _control_case(execution_cases, FAMILIES[2], initial_index=0)
    def forbidden(*args):
        raise AssertionError("Missing molecular dose mapping cannot trigger a physical query or actuation")
    report = run_native_residual_control_episode(episode, _loss(episode), forbidden, forbidden, episode.model.config)
    assert report["status"] == "blocked" and report["reason"] == "molecular_delivered_dose_mapping_unbound"
    assert plant._query == 0 and not report["decisions"] and not report["full_residual_mpc_eligible"]


def _nominal_serialized_bundle(execution_cases, tmp_path):
    _, episode, nominal, library = _control_case(execution_cases, initial_index=0)
    config = episode.model.config
    group = copy.deepcopy(nominal._group)
    # Descriptors are retained, while truth values are explicitly constructed
    # from the independent nominal test world, never an inferred particle.
    payload = copy.deepcopy(episode._execution_inputs.posterior.support.to_payload())
    initial = torch.from_numpy(nominal._history[0]["field"]).float()
    values = initial[None, None].expand(1, 32, *initial.shape).clone()
    coordinate = episode._execution_inputs.grid["species_id"].index("calcium_concentration")
    for index in range(32):
        values[0, index, ..., coordinate] += index*.001
    payload["initial_field"] = values.tolist()
    payload["factors"] = {key: torch.ones_like(value).tolist()
                          for key, value in episode._execution_inputs.posterior.support.factors.items()}
    original_binding = payload.pop("physical_parameter_hypotheses")
    raw = {"source_posterior_uid": group["posterior_uid"], "support": payload}
    reference = write_joint_source_reference(tmp_path/"group_0_conditioning.json.gz", raw, max_bytes=1000000)
    group["conditioning_reference"] = reference
    identity = {"producer_commit": "artificial_serialized_original_source_fixture",
                "definition_sha256": digest({"profile": config["native_observable_bridge"], "background": config["controlled_cns_source"]})}
    parameters = [copy.deepcopy(nominal._parameters) for _ in range(32)]
    hidden = 7
    for name in parameters[hidden]:
        if not name.startswith("initial_"):
            parameters[hidden][name] *= 1.01
            break
    producer = {"hidden_index": hidden, "parameters": parameters, "rng_streams": {"unused_no_replay": 0}}
    named_payload = copy.deepcopy(payload)
    names = [item["name"] for item in original_binding["declarations"]]
    original_binding.update(values=[[[row[name] for name in names] for row in parameters]],
                            source_record_refs=[str(tmp_path/"group_0_producer.json")],
                            producer_commit=identity["producer_commit"])
    named_payload["physical_parameter_hypotheses"] = original_binding
    named_reference = write_joint_source_reference(tmp_path/"named_conditioning.json.gz",
        dict(raw, support=named_payload), max_bytes=1000000)
    registry = {"grid": {"family_uid": group["family_uid"]}, "joint_source_references": [named_reference]}
    registry_path = tmp_path/"named_registry.json"; registry_path.write_text(json.dumps(registry))
    config["native_diffusion_data"]["registry_path"] = str(registry_path)
    receipt = {"source_status": "complete", "source_identity": identity, "groups": [group]}
    group_receipt = {"source_identity": identity, "group_index": 0, "group": group,
                     "operations": {"group:0": {"status": "complete"}}}
    for name, value in [("group_0_producer.json", producer), ("source_receipt.json", receipt), ("group_0_receipt.json", group_receipt)]:
        (tmp_path/name).write_text(json.dumps(value))
    admission = {"scope": "admitted_controlled_source", "base_admission": str(tmp_path/"native_consumption.json"), "source_identity": identity}
    return config, nominal._group, library, nominal.observation_identity, admission, values[0, hidden], parameters[hidden]


def test_private_loader_reads_writer_truth_binding_without_replay_or_model_tuple(execution_cases, tmp_path):
    config, group, library, identity, admission, initial, parameters = _nominal_serialized_bundle(execution_cases, tmp_path)
    factory = NativeRetainedSourceWorlds(config, admission)
    plant = factory(group, library, identity, 0, 991)
    assert torch.equal(torch.from_numpy(plant._history[0]["field"]).float(), initial)
    assert plant._parameters == parameters and plant._query == 0
    assert plant.observation_identity == identity
    broken = dict(group, source_uid="foreign")
    with pytest.raises(ValueError, match="original evaluation identity"):
        factory(broken, library, identity, 0, 991)
    path = tmp_path/"group_0_producer.json"
    producer = json.loads(path.read_text())
    changed = copy.deepcopy(producer)
    parameter = next(iter(changed["parameters"][0]))
    changed["parameters"][0][parameter] += .001
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="admitted named source conditioning"):
        factory(group, library, identity, 0, 991)
    producer["hidden_index"] = 32
    path.write_text(json.dumps(producer))
    with pytest.raises(ValueError, match="truth selection"):
        factory(group, library, identity, 0, 991)


@pytest.mark.parametrize("adaptive", [False, True])
def test_native_cli_dispatch_keeps_predictor_identity_and_requires_initial_index(tmp_path, monkeypatch, adaptive):
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_comparator_evaluation as comparators
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.utils.evaluation import native_intervention_evaluation as interventions
    from Libs.utils.evaluation import native_intervention_decisions as decisions
    from Libs.utils.exp import experiment_manifests
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    residual_control_policy(config)
    reference = tmp_path/"reference.json"; reference.write_text('{}')
    args = SimpleNamespace(command="evaluate_residual_controls", baseline_name="residual_mpc", initial_action_index=None,
        output_dir=str(tmp_path/"output"), decision_reference=reference, conditioning_config="conditioning.yaml",
        conditioning_manifest="conditioning.json", training_manifest="residual.json", device="cpu",
        evaluation_role="validation", residual_artifact="artificial_residual.json" if adaptive else None,
        adaptive_readout=adaptive)
    with pytest.raises(ValueError, match="explicit --initial-action-index"):
        cli.run_native_command(args, config)
    args.initial_action_index = 2
    monkeypatch.setattr(experiment_manifests, "resolve_training_seed", lambda *a, **k: {})
    monkeypatch.setattr(comparators, "declared_dataset", lambda *a, **k: (object(), {}))
    monkeypatch.setattr(interventions, "training_record_identities", lambda *a: [])
    monkeypatch.setattr(decisions, "validate_public_loss_reference", lambda *a: None)
    calls = []
    def load(*arguments, **kwargs):
        calls.append(arguments[1]); return object(), object(), {"method": arguments[1]}
    monkeypatch.setattr(comparators, "load_native_comparator", load)
    def evaluate(*arguments, **kwargs):
        assert kwargs["initial_index"] == 2 and arguments[3]["method"] == "mechanistic_residual"
        assert kwargs["adaptive_readout"] is adaptive
        assert kwargs["residual_path"] == (Path(args.residual_artifact).resolve() if adaptive else None)
        return {"status": "incomplete"}
    monkeypatch.setattr(consumer, "evaluate_native_residual_controls", evaluate)
    assert cli.run_native_command(args, config) == 2 and calls == ["mechanistic_residual"]



def _artificial_residual_envelope(config, binding):
    """Declared schema fixture only: no numerical or scientific fit/check."""
    from Libs.utils.evaluation.native_comparator_evaluation import native_residual_partitions
    from Libs.utils.model.native_transolver_residual import CONTROLLED_LAW
    partitions = native_residual_partitions(config)
    policy = config["native_controlled_comparators"]["residual_calibration"]
    species = [row["quantity_id"] for row in config["native_observable_bridge"]["quantity_profile"]["quantities"]]
    fit_ids, check_ids = list(partitions["fit"]), list(partitions["check"])
    numerical = policy["numerical"]
    threshold = max(numerical["hessian_absolute_threshold"], numerical["hessian_relative_threshold"])
    missing = {"declared_stratum_count": 0, "measured_stratum_count": 0, "missing_stratum_count": 0,
               "count": 0, "centered_product_mean": None, "total_covariance_mean": None}
    return {"schema_version": "native_comparator_residual.v1", "law": CONTROLLED_LAW,
        "binding": copy.deepcopy(binding), "status": "FIT_COMPLETE", "scales": [.01]*26,
        "fit_group_uids": fit_ids, "check_group_uids": check_ids,
        "reference_time_hours": policy["reference_time_hours"], "numerical": copy.deepcopy(numerical),
        "calibration_check_complete": True,
        "software_fixture": {"scope": "artificial_schema_envelope_no_fit_no_check",
                             "scientific_fit": False, "scientific_check": False, "optimizer_steps": 0},
        "formal_qualification": False, "scientific_twin": "NOT_RUN",
        "counts": {"declared_group_count": len(fit_ids), "eligible_group_count": len(fit_ids),
            "declared_episode_count": len(fit_ids), "eligible_episode_count": len(fit_ids),
            "future_coordinate_count": 26*len(fit_ids), "per_species_future_coordinate_count": [len(fit_ids)]*26,
            "groups": [{"group_uid": uid, "declared_episode_count": 1, "eligible_episode_count": 1} for uid in fit_ids],
            "episodes": [{"group_uid": uid, "future_coordinate_count": 26, "eligible": True} for uid in fit_ids]},
        "diagnostics": {"hessian_eigenvalues": [1.]*26, "gradient_max_abs": 0.,
                        "hessian_required_minimum": threshold, "objective": 0.},
        "checks": [{"group_uid": uid, "scientific_check_pass": None,
            "interpretation": "artificial_structure_only_no_independent_check",
            "pit": {"kind": "exact_marginal_mixture_cdf", "t0_continuous_pit_count": 0,
                    "future": {"count": 0, "mean": None},
                    "by_species": [{"species_id": name, "count": 0, "mean": None} for name in species]},
            "covariance": {name: copy.deepcopy(missing) for name in
                ("adjacent_retained_times", "graph_edges", "cross_species_pairs")}} for uid in check_ids]}


def test_adaptive_readout_mode_is_explicit_and_sensor_only_default():
    from Libs.utils.evaluation.native_residual_control_evaluation import residual_readout_mode
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    assert residual_readout_mode(config) == "sensor_only"
    assert residual_readout_mode(config, adaptive_readout=True) == "original_action_adaptive_unqualified"
    assert config["native_residual_control"]["readout_mode"] == "sensor_only"
    config["native_residual_control"]["readout_mode"] = "original_action_adaptive_unqualified"
    assert residual_readout_mode(config) == "original_action_adaptive_unqualified"
    for invalid in (1, None, "true"):
        with pytest.raises(ValueError, match="boolean"):
            residual_readout_mode(config, adaptive_readout=invalid)
    config["native_residual_control"]["readout_mode"] = "qualified"
    with pytest.raises(ValueError, match="readout mode"):
        residual_readout_mode(config)


@pytest.mark.parametrize("fault", ["command", "scope", "baseline", "missing_artifact"])
def test_adaptive_cli_rejects_unbound_requests_before_loading(monkeypatch, fault):
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    values = {"command": "evaluate_residual_controls", "protocol-scope": "manuscript-native",
              "baseline-name": "residual_mpc", "output-dir": "unused", "residual-artifact": "unused.json"}
    if fault == "command":
        values["command"] = "evaluate_baseline"
    elif fault == "scope":
        values["protocol-scope"] = "legacy"
    elif fault == "baseline":
        values["baseline-name"] = "mechanistic_residual"
    else:
        values.pop("residual-artifact")
    argv = ["bridge", "--adaptive-readout"]
    for key, value in values.items():
        argv.extend(["--"+key, value])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        cli.parse_args()
    assert caught.value.code == 2
    args = SimpleNamespace(command=values["command"], protocol_scope=values["protocol-scope"],
        baseline_name=values["baseline-name"], adaptive_readout=True, residual_artifact=values.get("residual-artifact"))
    with pytest.raises(ValueError, match="Adaptive readout requires"):
        cli.run_native_command(args, load_config("Libs/configs/exp/manuscript_native_protocol.yaml"))


@pytest.mark.parametrize("configured", [False, True])
def test_missing_adaptive_artifact_rejects_before_model_or_private_work(tmp_path, monkeypatch, configured):
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    if configured:
        config["native_residual_control"]["readout_mode"] = "original_action_adaptive_unqualified"
    args = SimpleNamespace(command="evaluate_residual_controls", baseline_name="residual_mpc",
        adaptive_readout=not configured, residual_artifact=None)
    with pytest.raises(ValueError, match="offline.*artifact"):
        cli.run_native_command(args, config)
    def forbidden(*args, **kwargs):
        pytest.fail("Missing adaptive artifact reached source/model/private work")
    monkeypatch.setattr(consumer, "predose_source_admission", forbidden)
    with pytest.raises(ValueError, match="offline.*artifact"):
        consumer.evaluate_native_residual_controls(config, object(), object(), {}, object(), {}, {},
            tmp_path, initial_index=0, adaptive_readout=not configured, world_factory=forbidden)
    assert not (tmp_path/"native_residual_control_evaluation.json").exists()


def test_invalid_adaptive_artifact_rejects_before_input_or_forecast(tmp_path, monkeypatch):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    path = tmp_path/"invalid.json"; path.write_text("{}")
    model = torch.nn.Linear(1, 1)
    model.prediction_kind = "offline_one_step_mechanistic_residual_conditional_mean"
    monkeypatch.setattr(consumer, "predose_source_admission", lambda *a: {"scope": "artificial_software_fixture"})
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid adaptive artifact reached input/forecast/private work")
    monkeypatch.setattr(consumer, "load_native_prediction_inputs", forbidden)
    with pytest.raises(ValueError, match="schema, predictor/normalizer"):
        consumer.evaluate_native_residual_controls(config, model, object(), {}, object(), {}, {}, tmp_path,
            initial_index=0, residual_path=path, adaptive_readout=True, world_factory=forbidden)
    assert not (tmp_path/"native_residual_control_evaluation.json").exists()



def test_adaptive_unsupported_source_refusal_precedes_input_and_private_work(tmp_path, monkeypatch):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    model = torch.nn.Linear(1, 1)
    model.prediction_kind = "offline_one_step_mechanistic_residual_conditional_mean"
    model.config = copy.deepcopy(config)
    dataset = SimpleNamespace(group_uids=["nonprimary", "molecular"])
    declarations = {"nonprimary": {"family_uid": FAMILIES[0]},
                    "molecular": {"family_uid": FAMILIES[2]}}
    monkeypatch.setattr(consumer, "predose_source_admission", lambda *a: {"scope": "artificial_software_fixture"})
    monkeypatch.setattr(consumer, "load_native_comparator_residual", lambda *a: {"law": "artificial_unused_scale"})
    monkeypatch.setattr(consumer, "NativePublicCNSLoss", lambda *a: object())
    monkeypatch.setattr(consumer, "load_family_registries", lambda *a: {})
    monkeypatch.setattr(consumer, "scoped_group_config", lambda *a:
        {"native_observable_bridge": {"active_source_law": "secondary"}})
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported source reached input/forecast/private work")
    monkeypatch.setattr(consumer, "load_native_prediction_inputs", forbidden)
    report = consumer.evaluate_native_residual_controls(config, model, object(), {}, dataset,
        declarations, {"reference_id": "artificial"}, tmp_path, initial_index=0,
        residual_path=Path("artificial_unused.json"), adaptive_readout=True, world_factory=forbidden)
    rows = {row["group_uid"]: row for row in report["groups"]}
    assert rows["nonprimary"]["status"] == "blocked"
    assert rows["nonprimary"]["reason"] == "private_continuation_for_nonprimary_source_law_unimplemented"
    assert rows["molecular"]["reason"] == "molecular_delivered_dose_mapping_unbound"
    assert report["status"] == "incomplete" and not report["full_residual_mpc_eligible"]


@pytest.mark.parametrize("fault", ["source", "snapshot", "role", "family", "grid", "normalizer",
                                   "samples", "particles", "actions", "clock", "predictor", "missing"])
def test_adaptive_control_reference_rejects_exact_binding_drift(execution_cases, fault):
    from Libs.utils.evaluation.native_residual_control_evaluation import adaptive_control_reference
    _, episode, _, library = _control_case(execution_cases, initial_index=0)
    inputs, norm, readout = copy.copy(episode._execution_inputs), episode.normalization, episode.readout
    binding = {"normalization_id": norm.identity, "checkpoint_sha256": "artificial_binding_fixture"}
    residual = {"schema_version": "native_comparator_residual.v1",
                "law": "native_controlled_teacher_readout_brownian.v1", "status": "FIT_COMPLETE",
                "calibration_check_complete": True, "binding": copy.deepcopy(binding),
                "scales": [.01]*26, "reference_time_hours": max(inputs.grid["time_hours"])}
    path = Path("artificial_binding_fixture_no_fit.json")
    if fault in {"source", "snapshot", "role", "family"}:
        field = {"source": "source_uid", "snapshot": "source_snapshot_ref", "role": "role", "family": "family_uid"}[fault]
        readout = replace(readout, **{field: "foreign"})
    elif fault == "grid":
        inputs.grid = dict(inputs.grid, species_id=list(reversed(inputs.grid["species_id"])))
    elif fault == "normalizer":
        inputs.normalization_id = "foreign"
    elif fault == "samples":
        inputs.sample_uids = ("foreign",)
    elif fault == "particles":
        inputs.particle_uids = (tuple(reversed(inputs.particle_uids[0])),)
    elif fault == "actions":
        library = tuple(reversed(library))
    elif fault == "clock":
        inputs.grid = dict(inputs.grid, time_hours=[time*2 for time in inputs.grid["time_hours"]])
    elif fault == "predictor":
        binding["checkpoint_sha256"] = "foreign"
    else:
        residual = None
    with pytest.raises(ValueError):
        adaptive_control_reference(inputs, norm, readout, library, 0, residual, path, binding)


def test_bound_adaptive_control_reference_reaches_actual_filter_and_h3(execution_cases):
    from Libs.models.modules.native_mechanistic_residual import NativeResidualOnlineEpisode
    from Libs.utils.evaluation.native_residual_control_evaluation import adaptive_control_reference
    _, prior_episode, plant, library = _control_case(execution_cases, initial_index=0)
    inputs, norm = prior_episode._execution_inputs, prior_episode.normalization
    binding = {"normalization_id": norm.identity, "checkpoint_sha256": "artificial_bound_episode_fixture"}
    envelope = {"schema_version": "native_comparator_residual.v1",
                "law": "native_controlled_teacher_readout_brownian.v1", "status": "FIT_COMPLETE",
                "calibration_check_complete": True, "binding": binding, "scales": [.01]*26,
                "reference_time_hours": max(inputs.grid["time_hours"])}
    reference = adaptive_control_reference(inputs, norm, prior_episode.readout, library, 0, envelope,
                                          Path("artificial_bound_episode_no_fit.json"), binding)
    episode = NativeResidualOnlineEpisode(prior_episode.model, inputs, norm, prior_episode.readout,
        executed_regimen_uid=library[0].regimen_uid, adaptive_reference=reference)
    episode.bind_execution(library, library[0], plant.observation_identity)
    from Libs.utils.model.native_conditional_readout import reload_adaptive_readout_history, adaptive_history_log_likelihood
    from Libs.utils.evaluation.native_residual_control_evaluation import record_actual_adaptive_readout_history
    current_report_config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    policy_snapshot = {key: copy.deepcopy(episode.model.config.get(key, current_report_config[key])) for key in
        ("native_residual_control", "native_temporal_query", "native_mechanistic_residual_online", "native_arrived_observation_filter")}
    policy_snapshot["native_mechanistic_residual_online"] = copy.deepcopy(episode.policy)
    assert policy_snapshot["native_mechanistic_residual_online"] == episode.policy
    report = {"readout_mode": "original_action_adaptive_unqualified", "binding": binding,
        "reference_id": "artificial_actual_history_reference", "pre_execution_policy_snapshot": policy_snapshot}
    row = {"group_uid": inputs.sample_uids[0], "family_uid": reference.fixed.readout.family_uid, "status": "component_complete"}
    context = {"group_uid": row["group_uid"], "family_uid": row["family_uid"], "binding": binding,
        "reference_id": report["reference_id"], "pre_execution_policy_snapshot": policy_snapshot}
    before_rng = plant._generator.get_state().clone()
    record_actual_adaptive_readout_history(row, episode, report)
    empty = row["actual_adaptive_readout_history"]
    assert empty["status"] == "empty" and empty["records"] == [] and not empty["likelihood_data_available"]
    assert torch.equal(plant._generator.get_state(), before_rng)
    for _ in range(6):
        episode.step(plant.acquire)
    assert episode.observations_seen == 6 and episode.optimizer_updates > 0
    assert len(episode.continuation._accepted_readout_snapshots) == 6
    assert all(row["readout_law"] == reference.law for row in episode.trace)
    assert all(row["adaptive_scientific_qualification"] == "UNQUALIFIED" for row in episode.trace)
    assert json.loads(reference.fixed.predictor_binding) == binding
    assert any(row.get("likelihood_consumer") == "bound_actual_filter_and_H3_same_adaptive_law" for row in episode.trace)
    raw_snapshots = episode.continuation._accepted_readout_snapshots
    before_rng = plant._generator.get_state().clone()
    record_actual_adaptive_readout_history(row, episode, report)
    payload = json.loads(json.dumps(row["actual_adaptive_readout_history"], allow_nan=False))
    restored = reload_adaptive_readout_history(payload, expected_reference=reference, expected_report_context=context)
    assert payload["origin"] == "episode_authenticated_actual_acceptance" and payload["status"] == "complete"
    assert restored.history == episode._adaptive_history() and not restored.calibration_eligible
    assert restored.initial_log_weights == episode._adaptive_initial_log_weights
    torch.testing.assert_close(adaptive_history_log_likelihood(reference, episode._adaptive_history(), episode._adaptive_initial_log_weights),
        adaptive_history_log_likelihood(reference, restored.history, restored.initial_log_weights), rtol=0, atol=0)
    for exported, accepted in zip(payload["records"], raw_snapshots):
        assert exported["mean"] == [[values[i] for i in exported["coordinates"]] for values in accepted.mean]
    assert torch.equal(plant._generator.get_state(), before_rng)
    assert "private_fields" not in json.dumps(payload) and "measurement_seed" not in json.dumps(payload)
    row.update(status="failed", failure={"message": "explicit artificial failure after accepted prefix"})
    record_actual_adaptive_readout_history(row, episode, report)
    assert row["actual_adaptive_readout_history"]["status"] == "failed"
    assert row["actual_adaptive_readout_history"]["records"] == payload["records"]
    # Replacing even equal history loses episode-owned acceptance authority.
    episode.continuation = replace(episode.continuation, _accepted_readout_snapshots=tuple(list(raw_snapshots)))
    record_actual_adaptive_readout_history(row, episode, report)
    assert row["status"] == "failed" and row["actual_adaptive_readout_history"]["status"] == "error"
    assert row["failure"]["message"] == "explicit artificial failure after accepted prefix"
    assert row["history_export_failure"]["type"] == "ValueError"
    assert len(episode.trace) == 6


def _independent_nominal_world(config, group, library, identity, initial_index, measurement_seed):
    """Explicit nominal source state/parameters, independent of any model particle."""
    import numpy as np
    from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
    from Libs.utils.evaluation.native_physical_plant import NativePhysicalPlant
    from Libs.utils.model import controlled_cns_source as cns
    from Libs.utils.model import native_scientific_source as source
    scoped = source.scoped_group_config(config, group)
    long = source.is_long_cns(scoped, group["family_uid"])
    background = source.long_cns_definition(scoped) if long else scoped["controlled_cns_source"]
    factors = {name: np.ones(len(basis)) for name, basis in background["factor_basis"].items()}
    initial = np.zeros((6, 6, 26), dtype=np.float32)
    names = background["grid"]["species_id"]
    for name in cns.DYNAMIC:
        initial[..., names.index(name)] = .1
    law = cns.ControlledLaw(background, factors, initial[..., :15])
    dynamic = initial[..., :15].reshape(1, 36, 15)[:, :, [names.index(name) for name in cns.DYNAMIC]]
    initial[..., :15] = law.quantities([0.], dynamic, background["actions"][0])[0]
    parameters = None if long else {name: float(row["value"]) for name, row in
        scoped["native_observable_bridge"]["physical_families"][group["family_uid"]]["parameters"].items()}
    grid = family_grid(observable_profile(scoped), group["family_uid"])
    mask = quantity_support(grid, group["source_uid"])[0]
    for qi, quantity in enumerate(grid["quantity_profile"]["quantities"]):
        if quantity.get("family_uid") != group["family_uid"]:
            continue
        site = quantity["incidence"]
        ri = grid["region_id"].index(site["anchor_region_id"])
        ci = grid["compartment_id"].index(site["anchor_compartment_id"])
        if mask[ri, ci, qi]:
            initial[ri, ci, qi] = parameters.get("initial_"+quantity["observable_name"], 0.)
    return NativePhysicalPlant(config, group, private_initial_field=initial, private_factors=factors,
        private_physical_parameters=parameters, initial_actuation=library[initial_index],
        measurement_generator=torch.Generator().manual_seed(measurement_seed), observation_identity=identity)


@pytest.mark.parametrize("adaptive", [False, True])
def test_actual_selected_checkpoint_control_cli_keeps_real_outcomes(tmp_path, monkeypatch, adaptive):
    import builtins
    import sys
    import time
    import yaml
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    fixture = Path(os.environ["NRNSIFT_RHS_FIXTURE"])
    fit = Path(os.environ["NRNSIFT_HYBRID_FIT"])
    config = load_config(os.environ["NRNSIFT_HYBRID_CONFIG"])
    training_seed_binding = json.loads((fit/"native_comparator_training_manifest.json").read_text())["training_seed_binding"]
    protocol = config["manuscript_native_protocol"]
    protocol["seed_policy"] = training_seed_binding["seed_policy"]
    assert native_training_seed_values(protocol["seed_policy"], protocol["training_seeds"]) == training_seed_binding["allowed_training_seeds"]
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    for key in ("native_residual_control", "native_future_controls", "native_mechanistic_residual_online", "native_arrived_observation_filter", "native_temporal_query"):
        config[key] = copy.deepcopy(current[key])
        if key in current["schemas"]:
            config["schemas"][key] = copy.deepcopy(current["schemas"][key])
    # Retain the historical model seeds; use the current evaluator measurement seed.
    config["native_predose_evaluator"]["measurement_seed"] = current["native_predose_evaluator"]["measurement_seed"]
    config_path = tmp_path/"selected_control_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    residual_path = tmp_path/"artificial_residual_envelope.json"
    if adaptive:
        # Write only an explicit artificial envelope with the selected binding.
        # Model loading and residual validation remain unchanged. No calibration
        # or optimizer/source fitting runs.
        from Libs.utils.evaluation import native_comparator_evaluation as comparators
        real_selected = comparators.selected_group_definitions
        real_groups = comparators.group_definitions
        artificial_rows = {reservation: [{"sample_uid": "artificial_cli_scale:"+reservation,
            "role": "calibration", "calibration_reservation": reservation,
            "family_uid": "artificial_schema_only_no_source", "action_indices": [0, 1, 2]}]
            for reservation in ("fit", "check", "threshold")}
        artificial_configurations = [config]
        def calibration_metadata(configuration, role, reservation=None):
            if any(configuration is item for item in artificial_configurations) and role == "calibration":
                return copy.deepcopy(artificial_rows[reservation] if reservation else
                                     sum(artificial_rows.values(), []))
            return real_selected(configuration, role, reservation)
        def all_calibration_metadata(configuration, *, role_filter=None):
            if any(configuration is item for item in artificial_configurations) and role_filter == "calibration":
                return copy.deepcopy(sum(artificial_rows.values(), []))
            return real_groups(configuration, role_filter=role_filter)
        # Only residual reservation metadata receives artificial declarations.
        # The actual TRAIN/validation dataset, model and source admission remain.
        monkeypatch.setattr(comparators, "selected_group_definitions", calibration_metadata)
        monkeypatch.setattr(comparators, "group_definitions", all_calibration_metadata)
        real_load = comparators.load_native_comparator
        def bound_envelope(*args, **kwargs):
            loaded = real_load(*args, **kwargs)
            assert consumer.predose_source_admission(args[0], args[5])["scope"] == "artificial_software_fixture"
            artificial_configurations.append(args[0])
            residual_path.write_text(json.dumps(_artificial_residual_envelope(args[0], loaded[2])))
            return loaded
        monkeypatch.setattr(comparators, "load_native_comparator", bound_envelope)
    worlds = []
    class NominalWorlds:
        def __init__(self, configuration, admission):
            assert admission["scope"] == "artificial_software_fixture"
            self.config = configuration
        def __call__(self, group, library, identity, initial_index, measurement_seed):
            assert self.config["native_predose_evaluator"]["measurement_seed"] == current["native_predose_evaluator"]["measurement_seed"]
            world = _independent_nominal_world(self.config, group, library, identity, initial_index, measurement_seed)
            worlds.append(world)
            return world
    # The private artificial world and, in adaptive mode, the declared synthetic
    # residual envelope are the injections. Numerical consumers are unchanged.
    monkeypatch.setattr(consumer, "NativeRetainedSourceWorlds", NominalWorlds)
    selection_witnesses = []
    real_plan = consumer.plan_temporal_query
    def observe_plan(node, logs, readout, policy, **kwargs):
        result = real_plan(node, logs, readout, policy, **kwargs)
        selection_witnesses.append({"readout_law": result.get("readout_law"), "query_index": node.query_index,
            "decision": result["decision"], "complete_plan_count": len(node.plans),
            "particle_count": result["particle_count"], "lookahead_depth": result["lookahead_depth"]})
        return result
    monkeypatch.setattr(consumer, "plan_temporal_query", observe_plan)
    real_open = builtins.open
    real_path_open = Path.open
    def reject_private_path(path):
        if isinstance(path, (str, os.PathLike)) and str(path).endswith("_producer.json"):
            if not Path(path).is_relative_to(tmp_path):
                raise AssertionError("Artificial CLI cannot open an original private producer bundle")
    def guarded_open(path, *args, **kwargs):
        reject_private_path(path)
        return real_open(path, *args, **kwargs)
    def guarded_path_open(path, *args, **kwargs):
        reject_private_path(path)
        return real_path_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    output = tmp_path/"control_cli"
    argv = ["run_external_baseline_bridge.py", "--protocol-scope", "manuscript-native",
        "--baseline-name", "residual_mpc", "--command", "evaluate_residual_controls",
        "--initial-action-index", "0", "--config", str(config_path),
        "--conditioning-config", str(fixture/"config.yaml"),
        "--conditioning-manifest", str(fixture/"native_fit/manifest.json"),
        "--training-manifest", str(fit/"native_comparator_training_manifest.json"),
        "--decision-reference", str(fixture/"public_cns_reference.json"), "--device", "cpu", "--output-dir", str(output)]
    if adaptive:
        argv.extend(["--adaptive-readout", "--residual-artifact", str(residual_path)])
    monkeypatch.setattr(sys, "argv", argv)
    started = time.monotonic()
    returncode = cli.main()
    elapsed = time.monotonic()-started
    report = json.loads((output/"native_residual_control_evaluation.json").read_text())
    evidence = Path(os.environ.get("NRNSIFT_RESIDUAL_CONTROL_EVIDENCE", tmp_path))
    evidence.mkdir(parents=True, exist_ok=True)
    report["software_cli_execution"] = {"argv": argv, "returncode": returncode, "full_cli_elapsed_seconds": elapsed,
        "new_offline_fit_steps": 0, "independent_scientific_adaptive_fit_check": False,
        "injected_dependency": "explicit_independent_nominal_world_and_artificial_scale_envelope" if adaptive
            else "explicit_independent_nominal_private_world_factory_only",
        "source_solver_elapsed_seconds": None, "qualification": "NOT_EVALUATED",
        "read_only_selection_budget_witnesses": selection_witnesses}
    # Save all real outcomes before assertions, including a bounded-budget stop.
    (evidence/("selected_checkpoint_adaptive_control_cli.json" if adaptive else "selected_checkpoint_control_cli.json")).write_text(json.dumps(report, indent=2)+"\n")
    rows = {row["family_uid"]: row for row in report["groups"]}
    assert set(rows) == set(FAMILIES) and report["declared_group_count"] == 4
    assert report["binding"]["checkpoint_sha256"] == "6c7f67c72602da355d713d0d9040718f9fa7354403762a4c002d544bff26dcf6"
    assert returncode == 2 and report["status"] == "incomplete" and not report["full_residual_mpc_eligible"]
    assert rows[FAMILIES[2]]["status"] == "blocked" and rows[FAMILIES[2]]["reason"] == "molecular_delivered_dose_mapping_unbound"
    assert len(worlds) == 3
    assert report["readout_mode"] == ("original_action_adaptive_unqualified" if adaptive else "sensor_only")
    assert report["adaptive_scientific_qualification"] == "UNQUALIFIED"
    assert not report["independent_adaptive_fit"] and not report["independent_adaptive_check"]
    if adaptive:
        assert report["scale_authority"] == "inherited_fixed_action_starting_model_not_adaptive_calibration"
        assert report["pre_execution_policy_snapshot"] == {key: config[key] for key in
            ("native_residual_control", "native_temporal_query", "native_mechanistic_residual_online",
             "native_arrived_observation_filter")}
    for family, row in rows.items():
        if family == FAMILIES[2]:
            continue
        assert row["status"] == "component_complete", row
        episode, realized = row["episode"], row["realized"]
        if adaptive:
            bound = row["adaptive_readout_reference"]
            assert bound["scientific_qualification"] == "UNQUALIFIED"
            assert json.loads(bound["fixed"]["predictor_binding"]) == report["binding"]
            assert len(bound["fixed"]["particle_uids"]) == 1 and len(bound["fixed"]["particle_uids"][0]) == 32
            assert episode["conditional_law"] == "native_original_action_adaptive_readout_brownian.v1"
        assert episode["actual_model_rhs_tuple_action_work"] <= 4000000
        assert episode["observations_seen"] == episode["measurement_units_spent"]
        assert episode["decisions"][-1]["decision"] == "STOP"
        assert not episode["planner_anticipates_residual_retraining"]
        for decision in episode["decisions"]:
            if decision["decision"] == "QUERY":
                assert decision["lookahead_depth"] == 2
                assert decision["selected_coordinate"] is not None
                assert decision["query_admissible"][decision["selected_edge_index"]]
        assert realized["source_rhs_calls"] > 0 and realized["source_horizon_complete"]
        assert realized["realized_loss_status"] == "complete_original_full_grid"
        assert realized["realized_full_grid_public_loss"] is not None
        assert len(realized["source_query_indices"]) == len(realized["original_query_values"])
        assert realized["committed_control_plan"] is not None
    assert all(row["particle_count"] == 32 for row in selection_witnesses)
    if adaptive:
        assert selection_witnesses and all(row["readout_law"] == "native_original_action_adaptive_readout_brownian.v1" for row in selection_witnesses)


@pytest.mark.parametrize("mode", ["sensor_only", "original_action_adaptive_unqualified"])
def test_history_report_explicitly_marks_unavailable_or_sensor_only(mode):
    from Libs.utils.evaluation.native_residual_control_evaluation import record_actual_adaptive_readout_history
    row = {"group_uid": "artificial", "family_uid": FAMILIES[0], "status": "blocked", "reason": "existing_block"}
    record_actual_adaptive_readout_history(row, None, {"readout_mode": mode})
    payload = row["actual_adaptive_readout_history"]
    assert payload["status"] == ("not_applicable" if mode == "sensor_only" else "unavailable")
    assert row["reason"] == "existing_block" and row["status"] == "blocked"
    assert not payload["calibration_eligible"] and payload["scientific_readiness"] == "NOT_ADMITTED"


def test_history_report_missing_binding_preserves_failure_and_rejects_output():
    from Libs.utils.evaluation.native_residual_control_evaluation import record_actual_adaptive_readout_history
    row = {"group_uid": "artificial", "family_uid": FAMILIES[0], "status": "failed", "failure": {"message": "prior_failure"}}
    report = {"readout_mode": "original_action_adaptive_unqualified"}
    record_actual_adaptive_readout_history(row, object(), report)
    assert row["status"] == "failed" and row["failure"] == {"message": "prior_failure"}
    assert row["actual_adaptive_readout_history"]["status"] == "error"
    assert row["history_export_failure"]["type"] == "KeyError"


def _collection_fixture(execution_cases, reservation="fit", family=FAMILIES[0]):
    """Rebind only disposable public fixture metadata before episode creation."""
    from Libs.models.modules.world_state import _posterior_uid
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference, FixedReadoutReference
    from Libs.models.modules.native_mechanistic_residual import NativeResidualOnlineEpisode
    original, prior, plant, library = _control_case(execution_cases, family=family, initial_index=2)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config = copy.deepcopy(original.config)
    for key in ("native_residual_calibration_collection", "native_residual_control",
                "native_temporal_query", "native_public_cns_loss", "native_predose_evaluator", "native_adaptive_residual_calibration"):
        config[key] = copy.deepcopy(current[key])
    for key in ("native_residual_control", "native_temporal_query",
                "native_residual_calibration_collection"):
        config["schemas"][key] = copy.deepcopy(current["schemas"][key])
    original.config = copy.deepcopy(config)
    inputs = prior._execution_inputs
    support = replace(inputs.posterior.support, role="calibration")
    observations = inputs.posterior.observation_uids
    posterior = replace(inputs.posterior, support=support, posterior_uid=_posterior_uid(
        support, observations, inputs.posterior.parent_posterior_uid,
        inputs.posterior.log_weights, inputs.posterior.particle_context))
    inputs = replace(inputs, posterior=posterior, posterior_uid=posterior.posterior_uid,
        programs=dict(inputs.programs, posterior_uid=posterior.posterior_uid,
            source_posterior_uid=inputs.programs.get("source_posterior_uid", inputs.posterior_uid)))
    readout = replace(prior.readout, role="calibration")
    context = prior.model.solver.filter_context(prior.continuation.rhs)
    binding = {"normalization_id": prior.normalization.identity,
               "checkpoint_sha256": "artificial_collection_bound_predictor",
               "model_identity": {"seed_binding": {"model_seed": 40}}}
    fixed = FixedReadoutReference(readout, context["support_uid"], posterior.posterior_uid,
        tuple(context["sample_uids"]), tuple(tuple(row) for row in context["particle_uids"]),
        library[2].regimen_uid, json.dumps(binding, sort_keys=True), prior.normalization.identity,
        "artificial_collection_scale_no_fit", (.01,)*26, tuple(prior.normalization.scale.tolist()),
        max(prior.continuation.rhs._times), tuple(library[2].source_times),
        tuple(prior.continuation.rhs._times), library[2].source_time_unit, 0)
    reference = AdaptiveReadoutReference(fixed, library, library[2].regimen_uid)
    episode = NativeResidualOnlineEpisode(original, inputs, prior.normalization, readout,
        executed_regimen_uid=library[2].regimen_uid, adaptive_reference=reference)
    group = copy.deepcopy(plant._group)
    group.update(role="calibration", calibration_reservation=reservation)
    from Libs.utils.data.native_actuation import NativeObservationIdentity
    identity = NativeObservationIdentity.from_context(episode.model.solver.filter_context(episode.continuation.rhs))
    plant = _independent_nominal_world(episode.model.config, group, library, identity, 2, 9151)
    episode.bind_execution(library, library[2], plant.observation_identity)
    declaration = dict(group, sample_uid=inputs.sample_uids[0])
    collection = {"purpose": "frozen_policy_actual_episode_history_collection", "reservation": reservation,
        "source_admission": {"scope": "artificial_software_fixture"},
        "selected_predictor": {"scope": "artificial_software_fixture", "scientific_checkpoint_sufficiency": False},
        "roles": {"fit": [declaration["sample_uid"]], "check": ["artificial_independent_check"]},
        "threshold_group_uids": ["artificial_reserved_threshold"],
        "denominator": {"software_fixture_only": True, "production_complete": False}}
    return config, original, episode, plant, library, declaration, binding, collection


@pytest.mark.parametrize("reservation", ["fit", "check"])
def test_collection_actual_role_history_stop_and_realized_loss(execution_cases, monkeypatch, tmp_path, reservation):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.utils.data.native_actuation import NativeActuationRequest, executed_prefix, compile_actuation_plan
    from Libs.utils.model.native_conditional_readout import reload_adaptive_readout_history
    config, original, episode, plant, library, declaration, binding, context = _collection_fixture(execution_cases, reservation)
    reference = json.loads((Path(os.environ["NRNSIFT_RHS_FIXTURE"])/"public_cns_reference.json").read_text())
    loss_function = _loss(episode)
    frozen = consumer._collection_binding(config, binding, reference,
        {"fit_group_uids": context["roles"]["fit"], "check_group_uids": context["roles"]["check"]}, context)
    bound = consumer._collection_episode_binding(episode, library, declaration, frozen)
    def validate():
        assert consumer._collection_episode_binding(episode, library, declaration, frozen) == bound
    for query in range(1, 7):
        validate()
        episode.step(plant.acquire)
        if query in (1, 3):
            action = library[1] if query == 1 else library[2]
            request = NativeActuationRequest(executed_prefix(episode.continuation.rhs._public_control, query), action)
            branch = episode.plan_controls([action.regimen_uid])
            episode.activate_selected_control(branch, request, plant.switch)
    assert episode.optimizer_updates == 20
    state = episode.continuation.rhs
    suffix = ()
    plan = compile_actuation_plan(state._public_control, executed_prefix(state._public_control, state.query_index),
                                 suffix, library)
    # A STOP risk is frozen before the terminal plant call and contains no source truth.
    from Libs.utils.evaluation.native_intervention_decisions import composed_action_burden
    burden = composed_action_burden(state._public_control, config)["normalized_burden"]
    predicted = loss_function(episode.recorded_model_history(), state._validity[0, 0, 0],
        state._guide.grid, torch.tensor(burden, dtype=torch.float64))["loss"]
    risk = float((predicted.double()*episode.model.solver.current_log_weights(state)[0].exp()).sum())
    stop = {"predicted_risk": risk, "committed_plan": __import__("dataclasses").asdict(plan),
        "query_index": state.query_index, "reference_id": reference["reference_id"],
        "maximum_supported_loss": 1.+config["native_public_cns_loss"]["intervention_weight"]}
    validate()
    plant._complete_planned_tail(plan)
    row = {"group_uid": declaration["sample_uid"], "family_uid": declaration["family_uid"],
        "status": "component_complete", "episode": {"selected_stop_record": copy.deepcopy(stop)}}
    report = {"readout_mode": "original_action_adaptive_unqualified", "binding": binding,
        "reference_id": reference["reference_id"], "pre_execution_policy_snapshot": {key: config[key] for key in (
            "native_residual_control", "native_temporal_query", "native_mechanistic_residual_online", "native_arrived_observation_filter")}}
    consumer.record_actual_adaptive_readout_history(row, episode, report)
    row["realized"] = consumer.realized_native_residual_control(plant, row["status"], loss_function, config)
    consumer._pair_collection_stop(row, frozen)
    assert row["selected_loss_collection"]["status"] == "complete"
    assert row["selected_loss_collection"]["signed_selected_loss_error"] == row["realized"]["realized_full_grid_public_loss"]-risk
    payload = json.loads(json.dumps(row["actual_adaptive_readout_history"], allow_nan=False))
    expected_context = dict(group_uid=row["group_uid"], family_uid=row["family_uid"], binding=binding,
        reference_id=report["reference_id"], pre_execution_policy_snapshot=report["pre_execution_policy_snapshot"])
    restored = reload_adaptive_readout_history(payload, expected_reference=episode.adaptive_reference,
                                             expected_report_context=expected_context)
    assert restored.history == episode._adaptive_history() and len(restored.history) == 6
    assert payload["origin"] == "episode_authenticated_actual_acceptance" and not payload["calibration_eligible"]
    assert frozen["existing_residual_role_reuse"][reservation]
    assert not frozen["statistical_independence_claimed"]
    assert "measurement_seed" not in json.dumps(frozen) and "private_fields" not in json.dumps(payload)


@pytest.mark.parametrize("fault", ["role", "reservation", "prior", "readout", "library"])
def test_collection_episode_binding_rejects_changed_authority(execution_cases, fault):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config, original, episode, plant, library, declaration, binding, context = _collection_fixture(execution_cases)
    frozen = consumer._collection_binding(config, binding, {"reference_id": "artificial"}, {}, context)
    baseline = consumer._collection_episode_binding(episode, library, declaration, frozen)
    if fault == "role": declaration["role"] = "validation"
    elif fault == "reservation": declaration["calibration_reservation"] = "threshold"
    elif fault == "prior": episode.continuation._initial_readout_log_weights = (0.,)*32
    elif fault == "readout":
        episode._adaptive_reference = replace(episode.adaptive_reference,
            fixed=replace(episode.adaptive_reference.fixed, readout=replace(episode.readout, role="validation")))
    else: library = tuple(reversed(library))
    rng, query = plant._generator.get_state().clone(), plant._query
    if fault == "library":
        assert consumer._collection_episode_binding(episode, library, declaration, frozen) != baseline
    else:
        with pytest.raises(ValueError):
            consumer._collection_episode_binding(episode, library, declaration, frozen)
    assert plant._query == query and torch.equal(plant._generator.get_state(), rng)


@pytest.mark.parametrize("fault", ["train", "development", "validation", "threshold", "missing"])
def test_collection_cli_rejects_unbound_reservations_before_loading(monkeypatch, fault):
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    args = ["bridge", "--protocol-scope", "manuscript-native", "--baseline-name", "residual_mpc",
        "--command", "collect_residual_calibration", "--residual-artifact", "unused.json", "--output-dir", "unused"]
    if fault != "missing": args += ["--calibration-reservation", fault]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as caught: cli.parse_args()
    assert caught.value.code == 2


def test_collection_production_manifest_sufficiency_is_not_a_boolean_override(tmp_path):
    from Libs.utils.evaluation.native_residual_control_evaluation import _collection_selected_predictor
    path = tmp_path/"selected.pt"; path.write_bytes(b"selected artificial checkpoint bytes")
    from Libs.utils.model.native_direct_mean import file_sha256
    manifest = tmp_path/"manifest.json"
    identity = {"checkpoint_path": str(path), "checkpoint_sha256": file_sha256(path),
                "training_manifest": str(manifest), "manifest_path": str(manifest)}
    binding = dict(identity, model_identity={"conditioning": identity})
    selection = {"selection_role": "development", "best": {"checkpoint_path": str(path)}, "sufficient_training": False}
    manifest.write_text(json.dumps({"selection": selection}))
    with pytest.raises(ValueError, match="canonical reviewed training sufficiency"):
        _collection_selected_predictor(binding, artificial=False)
    selection["sufficient_training"] = True
    manifest.write_text(json.dumps({"selection": selection}))
    with pytest.raises(ValueError, match="boolean cannot authorize"):
        _collection_selected_predictor(binding, artificial=False)
    binding["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="selected checkpoint differs"):
        _collection_selected_predictor(binding, artificial=False)


def test_collection_empty_stop_and_incomplete_loss_remain_explicit():
    from Libs.utils.evaluation.native_residual_control_evaluation import _pair_collection_stop
    frozen = {"public_loss_reference": {"reference_id": "same"}}
    row = {"status": "component_complete", "episode": {"selected_stop_record": {
        "predicted_risk": .2, "committed_plan": {}, "reference_id": "same"}},
        "realized": {"realized_loss_status": "complete_original_full_grid",
                     "reference_id": "same", "realized_full_grid_public_loss": .4}}
    _pair_collection_stop(row, frozen)
    assert row["selected_loss_collection"]["signed_selected_loss_error"] == .2
    row["realized"]["realized_loss_status"] = "incomplete_episode"
    _pair_collection_stop(row, frozen)
    assert row["selected_loss_collection"]["status"] == "unavailable"
    assert not row["selected_loss_collection"]["calibration_eligible"]


def test_collection_cli_routes_declared_fit_check_without_calibrating(monkeypatch, tmp_path):
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_comparator_evaluation as comparator
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.utils.evaluation import native_intervention_decisions as decisions
    from Libs.utils.evaluation import native_intervention_evaluation as interventions
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    reference = tmp_path/"reference.json"; reference.write_text(json.dumps({"reference_id": "artificial"}))
    args = SimpleNamespace(command="collect_residual_calibration", protocol_scope="manuscript-native",
        baseline_name="residual_mpc", calibration_reservation="check", residual_artifact="unused.json",
        initial_action_index=0, optimize_initial_plan=False, optimize_initial_joint_plan=False,
        adaptive_readout=False, output_dir=str(tmp_path), decision_reference=reference,
        conditioning_config="unused.yaml", conditioning_manifest="unused.json", training_manifest="unused.json",
        device="cpu")
    seen=[]
    def dataset(configuration, role, reservation=None):
        seen.append((role,reservation)); return SimpleNamespace(role=role,reservation=reservation), {"one": {}}
    monkeypatch.setattr(comparator,"declared_dataset",dataset)
    monkeypatch.setattr(comparator,"native_residual_partitions",lambda c: {})
    monkeypatch.setattr(comparator,"admit_native_residual_calibration",lambda *a: {})
    monkeypatch.setattr(comparator,"load_native_comparator",lambda *a: ("model","belief",{"loaded":True}))
    monkeypatch.setattr(decisions,"validate_public_loss_reference",lambda *a: None)
    monkeypatch.setattr(interventions,"training_record_identities",lambda *a: [])
    def collect(configuration,model,belief,binding,fit,check,reference,output,**kwargs):
        assert kwargs["reservation"]=="check" and fit.reservation=="fit" and check.reservation=="check"
        assert binding=={"loaded":True}; return {"status":"incomplete"}
    monkeypatch.setattr(consumer,"collect_native_residual_calibration",collect)
    assert cli.run_native_command(args,config)==2
    assert seen==[("train",None),("calibration","fit"),("calibration","check")]


@pytest.mark.parametrize("reservation", ["fit", "check"])
def test_collection_metadata_admission_and_thresholds_precede_any_private_work(monkeypatch, tmp_path, reservation):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config=load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    families=FAMILIES
    rows={name:{f"{name}:{family}:{i}":{"sample_uid":f"{name}:{family}:{i}","family_uid":family,
        "role":"calibration","calibration_reservation":name} for family in families for i in range(512)}
        for name in ("fit","check","threshold")}
    fit,check=object(),object()
    monkeypatch.setattr(consumer,"admit_native_residual_calibration",lambda c,f,h:{"fit":rows["fit"],"check":rows["check"]})
    monkeypatch.setattr(consumer,"predose_source_admission",lambda *a:{"scope":"artificial_software_fixture"})
    monkeypatch.setattr(consumer,"selected_group_definitions",lambda *a:list(rows["threshold"].values()))
    calls=[]
    def evaluator(c,m,b,binding,dataset,declarations,reference,output,**kwargs):
        assert dataset is (fit if reservation=="fit" else check)
        assert kwargs["collection_context"]["denominator"]["production_complete"]
        assert kwargs["collection_context"]["reservation"]==reservation
        assert not set(declarations)&set(kwargs["collection_context"]["threshold_group_uids"])
        calls.append("collection");return {"status":"incomplete"}
    monkeypatch.setattr(consumer,"evaluate_native_residual_controls",evaluator)
    result=consumer.collect_native_residual_calibration(config,object(),object(),{},fit,check,{},tmp_path,
        reservation=reservation,residual_path=Path("unused.json"))
    assert result["status"]=="incomplete" and calls==["collection"]
    collision=next(iter(rows["fit"]))
    rows["threshold"][collision]=dict(rows["fit"][collision],calibration_reservation="threshold")
    with pytest.raises(ValueError,match="threshold"):
        consumer.collect_native_residual_calibration(config,object(),object(),{},fit,check,{},tmp_path,
            reservation=reservation,residual_path=Path("unused.json"))


@pytest.mark.parametrize("fault", ["policy", "binding", "prior"])
def test_collection_evaluator_rejects_changed_frozen_binding_before_private_callback(execution_cases, monkeypatch, tmp_path, fault):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config,original,episode,plant,library,declaration,binding,context=_collection_fixture(execution_cases)
    inputs=episode._execution_inputs
    monkeypatch.setattr(consumer,"predose_source_admission",lambda *a:context["source_admission"])
    residual={"schema_version":"native_comparator_residual.v1","law":"native_controlled_teacher_readout_brownian.v1",
        "status":"FIT_COMPLETE","calibration_check_complete":True,"binding":binding,"scales":[.01]*26,
        "fit_group_uids":[],"check_group_uids":[],"reference_time_hours":max(inputs.grid["time_hours"])}
    monkeypatch.setattr(consumer,"load_native_comparator_residual",lambda *a:residual)
    monkeypatch.setattr(consumer,"load_family_registries",lambda *a:{})
    def load_inputs(*args):
        if fault=="policy":config["native_temporal_query"]["measurement_cost"]+=1
        if fault=="binding":binding["checkpoint_sha256"]="changed"
        return inputs
    monkeypatch.setattr(consumer,"load_native_prediction_inputs",load_inputs)
    monkeypatch.setattr(consumer,"native_actuation_library",lambda *a:library)
    monkeypatch.setattr(consumer,"native_cns_readout",lambda *a:episode.readout)
    if fault=="prior":
        real=consumer.NativeResidualOnlineEpisode
        def changed(*args,**kwargs):
            result=real(*args,**kwargs)
            result.continuation._initial_readout_log_weights=(0.,)*32
            return result
        monkeypatch.setattr(consumer,"NativeResidualOnlineEpisode",changed)
    calls=[]
    def forbidden(*a):calls.append("private");raise AssertionError("Invalid collection reached private provider")
    report=consumer.evaluate_native_residual_controls(config,original,SimpleNamespace(field_normalization=episode.normalization),
        binding,SimpleNamespace(group_uids=[declaration["sample_uid"]]),{declaration["sample_uid"]:declaration},
        json.loads((Path(os.environ["NRNSIFT_RHS_FIXTURE"])/"public_cns_reference.json").read_text()),tmp_path,
        initial_index=2,residual_path=Path("unused.json"),adaptive_readout=True,world_factory=forbidden,
        collection_context=context)
    assert report["status"]=="incomplete" and report["groups"][0]["status"]=="failed"
    assert calls==[]


def test_collection_stop_snapshot_precedes_actual_terminal_executor(execution_cases, monkeypatch):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config,_,episode,plant,library,_,_,_=_collection_fixture(execution_cases)
    loss=_loss(episode)
    node=SimpleNamespace(public=episode.continuation.rhs._public_control,losses=torch.zeros(1),plans=[()],
        edges=[],latency_hours=0.)
    monkeypatch.setattr(consumer,"compile_temporal_world_plans",lambda *a,**k:SimpleNamespace(required_rhs_work=0))
    monkeypatch.setattr(consumer,"evaluate_temporal_world_plans",lambda *a:node)
    monkeypatch.setattr(consumer,"plan_temporal_query",lambda *a,**k:{"decision":"STOP","stop_plan_index":0,"stop_risk":.25})
    calls=[]
    def validate():calls.append("validate")
    def stop(plan):
        assert calls[-1]=="validate"
        calls.append("stop");plant._complete_planned_tail(plan)
    report=consumer.run_native_temporal_residual_episode(episode,loss,plant.acquire,plant.switch,stop,config,
        collection_validate=validate)
    assert report["status"]=="component_complete"
    assert report["selected_stop_record"]["predicted_risk"]==.25
    assert report["selected_stop_record"]["query_index"]==0
    assert report["selected_stop_record"]["committed_plan"]["steps"]==()
    assert episode.observations_seen==episode.optimizer_updates==0


def test_collection_actual_evaluator_empty_stop_retains_molecular_denominator(execution_cases, monkeypatch, tmp_path):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    config,original,episode,plant,library,declaration,binding,context=_collection_fixture(execution_cases)
    inputs=episode._execution_inputs
    uid=declaration["sample_uid"]
    molecular=dict(declaration,sample_uid="artificial_molecular_collection",family_uid=FAMILIES[2])
    declarations={uid:declaration,molecular["sample_uid"]:molecular}
    context["roles"]["fit"]=list(declarations)
    monkeypatch.setattr(consumer,"predose_source_admission",lambda *a:context["source_admission"])
    residual={"schema_version":"native_comparator_residual.v1","law":"native_controlled_teacher_readout_brownian.v1",
        "status":"FIT_COMPLETE","calibration_check_complete":True,"binding":binding,"scales":[.01]*26,
        "fit_group_uids":list(declarations),"check_group_uids":[],"reference_time_hours":max(inputs.grid["time_hours"])}
    monkeypatch.setattr(consumer,"load_native_comparator_residual",lambda *a:residual)
    monkeypatch.setattr(consumer,"load_family_registries",lambda *a:{})
    monkeypatch.setattr(consumer,"load_native_prediction_inputs",lambda *a:inputs)
    monkeypatch.setattr(consumer,"native_actuation_library",lambda *a:library)
    monkeypatch.setattr(consumer,"native_cns_readout",lambda *a:episode.readout)
    monkeypatch.setattr(consumer,"compile_temporal_world_plans",lambda *a,**k:SimpleNamespace(required_rhs_work=0))
    node=SimpleNamespace(public=episode.continuation.rhs._public_control,losses=torch.zeros(1),plans=[()],
        edges=[],latency_hours=0.)
    monkeypatch.setattr(consumer,"evaluate_temporal_world_plans",lambda *a:node)
    monkeypatch.setattr(consumer,"plan_temporal_query",lambda *a,**k:{"decision":"STOP","stop_plan_index":0,"stop_risk":.25})
    calls=[]
    def private(group,actions,identity,index,seed):
        assert group["role"]=="calibration" and group["calibration_reservation"]=="fit"
        calls.append(group["sample_uid"]);return plant
    reference=json.loads((Path(os.environ["NRNSIFT_RHS_FIXTURE"])/"public_cns_reference.json").read_text())
    report=consumer.evaluate_native_residual_controls(config,original,SimpleNamespace(field_normalization=episode.normalization),
        binding,SimpleNamespace(group_uids=list(declarations)),declarations,reference,tmp_path,
        initial_index=2,residual_path=Path("unused.json"),adaptive_readout=True,world_factory=private,
        collection_context=context)
    assert report["status"]=="incomplete" and report["declared_group_count"]==2 and report["completed_group_count"]==1
    rows={row["group_uid"]:row for row in report["groups"]}
    assert rows[molecular["sample_uid"]]["status"]=="blocked"
    assert rows[molecular["sample_uid"]]["reason"]=="molecular_delivered_dose_mapping_unbound"
    complete=rows[uid]
    assert complete["status"]=="component_complete",complete
    assert complete["actual_adaptive_readout_history"]["status"]=="empty"
    assert complete["selected_loss_collection"]["status"]=="complete"
    assert calls==[uid] and not report["production_collection_complete"]
    assert not report["calibration_eligible"] and not report["selected_decision_calibration_complete"]
    assert report["collection_binding"]["existing_residual_role_reuse"]["fit"]==sorted(declarations)
    text=json.dumps(report,allow_nan=False)
    assert "measurement_seed" not in text and "private_fields" not in text


@pytest.mark.parametrize("phase,reservation", [("adaptive_fit","fit"), ("final_policy_check","check"), ("final_selected_loss","threshold")])
def test_adaptive_cli_rejects_wrong_phase_or_missing_prerequisite(phase, reservation, monkeypatch):
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    args = ["bridge", "--protocol-scope", "manuscript-native", "--baseline-name", "residual_mpc",
        "--command", "collect_residual_calibration", "--residual-artifact", "unused.json", "--output-dir", "unused",
        "--collection-phase", phase, "--calibration-reservation", "check" if reservation == "fit" else reservation]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as caught:
        cli.parse_args()
    assert caught.value.code == 2



def _artificial_calibration_metadata(declarations,receipt_path):
    teachers=[{"software_fixture_only":True,"qc_catalog_ref":str(receipt_path)} for _ in declarations]
    child=SimpleNamespace(native_records=[{"teacher":teacher} for teacher in teachers])
    return SimpleNamespace(split_name="calibration",group_uids=[row["sample_uid"] for row in declarations],
        groups=[(child,list(range(len(teachers))),None)]),{row["sample_uid"]:row for row in declarations}


def test_actual_episode_collection_fit_cli_round_trip(execution_cases, monkeypatch, tmp_path):
    """Actual accepted A/B/A history reaches the fit command without dispatch mocks.

    The source fixture is artificial and small. Its terminal fit status supplies
    software evidence only. The analytic Gaussian fixture tests positive recovery.
    """
    import sys
    from dataclasses import asdict
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.data.native_actuation import NativeActuationRequest, executed_prefix, compile_actuation_plan
    config, original, episode, plant, library, declaration, binding, context = _collection_fixture(execution_cases)
    reference = json.loads((Path(os.environ["NRNSIFT_RHS_FIXTURE"])/"public_cns_reference.json").read_text())
    context["phase"] = "adaptive_fit"; context["purpose"] = "adaptive_fit"
    receipt=tmp_path/"artificial_metadata_receipt.json"
    receipt.write_text(json.dumps({"scope":"artificial_software_fixture_no_source_trajectory",
        "physical_source_trajectory_calls":0,"target_generation":"initial_complete_tuple_zero_plus_masked_linear_ramp"}))
    metadata=_artificial_calibration_metadata([declaration],receipt)
    context["source_admission"]=consumer.predose_source_admission(config,metadata[0])
    monkeypatch.setattr(consumer,"declared_dataset",lambda *args,**kwargs:metadata)
    # This fixture has one explicitly declared independent group. The canonical
    # production512-per-family denominator is not changed or represented as met.
    monkeypatch.setattr(consumer, "selected_group_definitions", lambda c, role, reservation: [declaration]
                        if reservation == "fit" else [])
    monkeypatch.setattr(consumer, "native_cns_readout", lambda c, group: episode.readout)
    monkeypatch.setattr(consumer, "native_actuation_library", lambda c, group: library)
    offline = {"scales": [.01]*26, "fit_group_uids": context["roles"]["fit"], "check_group_uids": []}
    frozen = consumer._collection_binding(config, binding, reference, offline, context)
    bound = consumer._collection_episode_binding(episode, library, declaration, frozen)
    for query_index in range(1,7):
        episode.step(plant.acquire)
        if query_index in (1,3):
            action = library[1] if query_index == 1 else library[2]
            request = NativeActuationRequest(executed_prefix(episode.continuation.rhs._public_control, query_index), action)
            episode.activate_selected_control(episode.plan_controls([action.regimen_uid]), request, plant.switch)
    state = episode.continuation.rhs
    plan = compile_actuation_plan(state._public_control, executed_prefix(state._public_control, state.query_index), (), library)
    burden = composed_action_burden(state._public_control, config)["normalized_burden"]
    predictions = _loss(episode)(episode.recorded_model_history(), state._validity[0,0,0], state._guide.grid,
                               torch.tensor(burden, dtype=torch.float64))["loss"]
    risk = float((predictions.double()*episode.model.solver.current_log_weights(state)[0].exp()).sum())
    stop = {"predicted_risk": risk, "committed_plan": asdict(plan), "query_index": state.query_index,
            "reference_id": reference["reference_id"], "maximum_supported_loss": 1.+config["native_public_cns_loss"]["intervention_weight"]}
    plant._complete_planned_tail(plan)
    row = {"group_uid": declaration["sample_uid"], "family_uid": declaration["family_uid"], "status": "component_complete",
           "episode": {"selected_stop_record": stop}, "episode_collection_binding": bound}
    report = {"schema_version": "native_residual_calibration_collection.v1", "status": "software_collection_complete",
        "reservation": "fit", "collection_binding": frozen, "collection_binding_id": digest(frozen),
        "denominator": context["denominator"], "readout_mode": "original_action_adaptive_unqualified", "binding": binding,
        "reference_id": reference["reference_id"], "pre_execution_policy_snapshot": {key: config[key] for key in (
            "native_residual_control", "native_temporal_query", "native_mechanistic_residual_online", "native_arrived_observation_filter")},
        "groups": [row]}
    consumer.record_actual_adaptive_readout_history(row, episode, report)
    row["realized"] = consumer.realized_native_residual_control(plant, row["status"], _loss(episode), config)
    consumer._pair_collection_stop(row, frozen)
    collection_path = tmp_path/"actual_fit_collection.json"
    collection_path.write_text(json.dumps(report, allow_nan=False))
    monkeypatch.setattr(sys, "argv", ["bridge", "--protocol-scope", "manuscript-native", "--baseline-name", "residual_mpc",
        "--command", "fit_adaptive_residual", "--adaptive-collection", str(collection_path), "--output-dir", str(tmp_path/"fit")])
    args = cli.parse_args()
    exit_status = cli.run_native_command(args, config)
    fitted = json.loads((tmp_path/"fit"/"native_adaptive_residual_fit.json").read_text())
    assert fitted["status"] in {"FIT_COMPLETE", "FIT_FAILED"}
    assert exit_status == (0 if fitted["status"] == "FIT_COMPLETE" else 2)
    assert fitted["fit_collection"]["sha256"] == consumer.file_sha256(collection_path)
    assert fitted["fit_group_uids"] == [declaration["sample_uid"]]
    assert fitted["whole_26_field_adaptive_qualification"] == "UNQUALIFIED"
    assert episode._adaptive_history() == consumer._read_collection_report(collection_path, config, "fit", "adaptive_fit")[1][0].history
    # Outcome-independent mutation tests reject before another numerical fit.
    altered = copy.deepcopy(report); altered["groups"][0]["actual_adaptive_readout_history"]["initial_log_weights"][0] = 0.
    collection_path.write_text(json.dumps(altered, allow_nan=False))
    with pytest.raises(ValueError): cli.run_native_command(args, config)



def test_adaptive_positive_cli_fit_check_selected_loss_and_bounds(monkeypatch, tmp_path):
    """Known Gaussian acceptance fixtures exercise the real artifact command chain.

    Source metadata is a declared artificial fixture adapter. Optimization,
    artifact verification, diagnostics, order statistics and runtime bounds use
    the real implementation. This fixture does not certify a physical Twin.
    """
    import sys
    from dataclasses import asdict
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.tests.test_native_adaptive_readout import _calibration_numeric_episodes
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.utils.model.native_conditional_readout import export_adaptive_readout_history
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_public_cns_loss"]["intervention_weight"] = .1
    templates = _calibration_numeric_episodes()
    declared, libraries, readouts = {}, {}, {}
    receipt=tmp_path/"known_gaussian_metadata_receipt.json"
    receipt.write_text(json.dumps({"scope":"artificial_software_fixture_no_source_trajectory",
        "physical_source_trajectory_calls":0,"target_generation":"initial_complete_tuple_zero_plus_masked_linear_ramp"}))
    monkeypatch.setattr(consumer,"declared_dataset",lambda c,role,reservation:
                        _artificial_calibration_metadata(declared[reservation],receipt))
    def source_rows(c, role, reservation): return declared[reservation]
    monkeypatch.setattr(consumer, "selected_group_definitions", source_rows)
    monkeypatch.setattr(consumer, "native_cns_readout", lambda c, group: readouts[group["sample_uid"]])
    monkeypatch.setattr(consumer, "native_actuation_library", lambda c, group: libraries[group["sample_uid"]])
    binding = json.loads(templates[0].reference.fixed.predictor_binding)
    offline = {"scales": list(templates[0].reference.fixed.residual_scales),
               "fit_group_uids": [], "check_group_uids": []}
    public = {"reference_id": "known_gaussian_artificial_public_loss"}
    initial = {"mode": "explicit_fixed_index", "initial_index": 0,
               "optimize_initial_plan": False, "optimize_initial_joint_plan": False}
    def collection(reservation, phase, fitted=None):
        ids = [f"known_gaussian:{reservation}:{i}" for i in range(len(templates))]
        declared[reservation] = [{"sample_uid": uid, "group_uid": uid, "family_uid": templates[0].reference.fixed.readout.family_uid,
            "role": "calibration", "calibration_reservation": reservation} for uid in ids]
        context = {"purpose": phase, "phase": phase, "reservation": reservation,
            "source_admission": {"scope": "artificial_software_fixture","pre_dose_binding":"retained_initial_field_plus_zero_t0_ramp"},
            "selected_predictor": {"scope": "artificial_software_fixture", "scientific_checkpoint_sufficiency": False},
            "denominator": {"software_fixture_only": True, "production_complete": False},
            "roles": {"fit": [f"known_gaussian:fit:{i}" for i in range(4)],
                      "check": [f"known_gaussian:check:{i}" for i in range(4)],
                      "threshold": [f"known_gaussian:threshold:{i}" for i in range(4)]},
            "threshold_group_uids": [f"known_gaussian:threshold:{i}" for i in range(4)], "initial_selection": initial}
        if fitted:
            context["adaptive_artifact_id"] = fitted["artifact_id"]
            context["final_policy_id"] = consumer.final_residual_policy_id(config,binding,public,offline,fitted,initial)
        frozen = consumer._collection_binding(config,binding,public,offline,context)
        report = {"schema_version": "native_residual_calibration_collection.v1", "status": "software_collection_complete",
            "reservation": reservation, "collection_binding": frozen, "collection_binding_id": digest(frozen),
            "denominator": context["denominator"], "binding": binding, "reference_id": public["reference_id"],
            "pre_execution_policy_snapshot": {key: config[key] for key in (
                "native_residual_control","native_temporal_query","native_mechanistic_residual_online","native_arrived_observation_filter")},
            "groups": []}
        for index, (template, declaration) in enumerate(zip(templates, declared[reservation])):
            fixed = replace(template.reference.fixed, sample_uids=(declaration["sample_uid"],),
                            readout=replace(template.reference.fixed.readout, role="calibration"))
            ref = replace(template.reference, fixed=fixed)
            if fitted:
                ref = replace(ref, fixed=replace(fixed,residual_scales=tuple(fitted["scales"])),
                    scale_authority="authenticated_observable_adaptive_fit", adaptive_artifact_id=fitted["artifact_id"],
                    fitted_species_indices=tuple(fitted["fitted_species_indices"]))
            history = tuple(replace(row,reference=ref,context=replace(row.context,
                predictor_binding=json.dumps(dict(binding,residual_optimizer_updates=0),sort_keys=True)))
                for row in template.history)
            libraries[declaration["sample_uid"]] = ref.original_actions
            readouts[declaration["sample_uid"]] = ref.fixed.readout
            payload = export_adaptive_readout_history(ref,history,template.initial_log_weights,
                report_context={"group_uid": declaration["sample_uid"], "family_uid": declaration["family_uid"],
                "binding": binding,"reference_id": public["reference_id"],
                "pre_execution_policy_snapshot": report["pre_execution_policy_snapshot"]}, episode_status="component_complete")
            # Artificial acceptance fixture scope is explicit. Actual physical
            # episode owner authentication is tested in the preceding round trip.
            payload["origin"] = "episode_authenticated_actual_acceptance"
            bound = {"collection_binding_id": digest(frozen), "declaration": declaration,
                "adaptive_reference": asdict(ref), "original_action_library": [asdict(action) for action in ref.original_actions],
                "initial_log_weights": list(template.initial_log_weights)}
            stop = {"predicted_risk": .2,"reference_id": public["reference_id"],"committed_plan": {},"maximum_supported_loss":1.1}
            row = {"group_uid": declaration["sample_uid"],"family_uid": declaration["family_uid"],"status":"component_complete",
                "episode_collection_binding": bound,"actual_adaptive_readout_history":payload,"episode":{"selected_stop_record":stop},
                "realized":{"realized_loss_status":"complete_original_full_grid","reference_id":public["reference_id"],
                            "realized_full_grid_public_loss":.3+index*.01}}
            consumer._pair_collection_stop(row,frozen);report["groups"].append(row)
        path=tmp_path/(reservation+"_collection.json");path.write_text(json.dumps(report,allow_nan=False))
        return path, report
    def command(name, source, directory, fit=None, check=None):
        argv=["bridge","--protocol-scope","manuscript-native","--baseline-name","residual_mpc",
              "--command",name,"--adaptive-collection",str(source),"--output-dir",str(directory)]
        if fit:argv += ["--adaptive-fit-artifact",str(fit)]
        if check:argv += ["--adaptive-check-artifact",str(check)]
        monkeypatch.setattr(sys,"argv",argv)
        assert cli.run_native_command(cli.parse_args(),config)==0
    fit_path,_=collection("fit","adaptive_fit")
    command("fit_adaptive_residual",fit_path,tmp_path/"fit")
    fitted_path=tmp_path/"fit/native_adaptive_residual_fit.json";fitted=json.loads(fitted_path.read_text())
    assert fitted["scales"][0]==pytest.approx(1.7,rel=1e-5)
    check_path,_=collection("check","final_policy_check",fitted)
    command("check_adaptive_residual",check_path,tmp_path/"check",fitted_path)
    checked_path=tmp_path/"check/native_adaptive_residual_check.json";checked=json.loads(checked_path.read_text())
    assert checked["observable_check_complete"] and checked["scientific_distribution_pass"] is None
    assert checked["diagnostics"]["conditional_marginal_mixture_pit"]["count"]==4
    consumer.validate_adaptive_check_artifact(checked,fitted,checked["final_policy_id"],config)
    for field in ("diagnostics","group_count","mean_complete_history_nll","group_log_likelihoods"):
        forged=copy.deepcopy(checked);forged.pop(field)
        forged["artifact_id"]=digest({key:value for key,value in forged.items() if key!="artifact_id"})
        with pytest.raises(ValueError,match="results or diagnostics"):
            consumer.validate_adaptive_check_artifact(forged,fitted,checked["final_policy_id"],config)
    for field in ("diagnostics","group_count","mean_complete_history_nll","group_log_likelihoods"):
        forged=copy.deepcopy(checked)
        if field=="diagnostics":forged[field]["conditional_marginal_mixture_pit"]["mean"]+=.1
        elif field=="group_log_likelihoods":forged[field][0]["log_likelihood"]+=1
        else:forged[field]+=1
        forged["artifact_id"]=digest({key:value for key,value in forged.items() if key!="artifact_id"})
        with pytest.raises(ValueError,match="results or diagnostics"):
            consumer.validate_adaptive_check_artifact(forged,fitted,checked["final_policy_id"],config)
    threshold_path,threshold=collection("threshold","final_selected_loss",fitted)
    command("calibrate_residual_selected_loss",threshold_path,tmp_path/"selected",fitted_path,checked_path)
    calibration=json.loads((tmp_path/"selected/native_residual_selected_loss_calibration.json").read_text())
    consumer.validate_selected_loss_artifact(calibration,fitted,checked,checked["final_policy_id"],config)
    row=threshold["groups"][0];consumer.apply_selected_loss_bounds(row,calibration)
    assert row["selected_loss_bounds"]["bounds"][0]["upper_bound"]==pytest.approx(.33)
    assert row["selected_loss_bounds"]["bounds"][-1]["upper_bound"]==1.1
    tampered=copy.deepcopy(calibration);tampered["quantiles"][0]["value"]=.9
    tampered["artifact_id"]=digest({k:v for k,v in tampered.items() if k!="artifact_id"})
    with pytest.raises(ValueError,match="quantiles differ"):
        consumer.validate_selected_loss_artifact(tampered,fitted,checked,checked["final_policy_id"],config)
    # A report scope string cannot replace the canonical metadata receipt.
    saved_receipt=receipt.read_text();receipt.write_text(json.dumps({"scope":"claimed_production"}))
    with pytest.raises(ValueError,match="retained receipt"):
        consumer.validate_adaptive_fit_artifact(fitted,config)
    receipt.write_text(saved_receipt)
    altered_collection=json.loads(fit_path.read_text())
    altered_collection["collection_binding"]["source_admission"]={"scope":"admitted_controlled_source"}
    altered_collection["collection_binding_id"]=digest(altered_collection["collection_binding"])
    relabeled_path=tmp_path/"relabeled_collection.json";relabeled_path.write_text(json.dumps(altered_collection))
    with pytest.raises(ValueError,match="source scope differs"):
        consumer._read_collection_report(relabeled_path,config,"fit","adaptive_fit")
    altered=copy.deepcopy(fitted);altered["scales"][25]*=2
    altered["artifact_id"]=digest({k:v for k,v in altered.items() if k!="artifact_id"})
    with pytest.raises(ValueError,match="unobserved offline scale"):
        consumer.validate_adaptive_fit_artifact(altered,config)
    altered_observable=copy.deepcopy(fitted);altered_observable["scales"][0]*=1.3
    altered_observable["artifact_id"]=digest({k:v for k,v in altered_observable.items() if k!="artifact_id"})
    with pytest.raises(ValueError,match="gradient or Hessian"):
        consumer.validate_adaptive_fit_artifact(altered_observable,config)
    altered_check=copy.deepcopy(checked);altered_check["final_policy_id"]="changed"
    altered_check["artifact_id"]=digest({k:v for k,v in altered_check.items() if k!="artifact_id"})
    with pytest.raises(ValueError):consumer.validate_adaptive_check_artifact(altered_check,fitted,checked["final_policy_id"],config)


@pytest.mark.parametrize("kind",["orphan_check","selected_without_fit","selected_without_check","fit_without_check","legacy","unsupported_command","unsupported_baseline"])
def test_adaptive_artifact_dependencies_reject_before_loading(kind,monkeypatch,tmp_path):
    import sys
    from Libs.exp.baselines import run_external_baseline_bridge as cli
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    values={"protocol_scope":"manuscript-native","baseline_name":"residual_mpc","command":"evaluate_residual_controls",
        "adaptive_fit_artifact":Path("missing_fit.json"),"adaptive_check_artifact":Path("missing_check.json"),
        "selected_loss_calibration":None,"adaptive_readout":True,"residual_artifact":"missing_offline.json"}
    if kind=="orphan_check":values["adaptive_fit_artifact"]=None
    elif kind=="selected_without_fit":values.update(adaptive_fit_artifact=None,adaptive_check_artifact=None,selected_loss_calibration=Path("missing_selected.json"))
    elif kind=="selected_without_check":values.update(adaptive_check_artifact=None,selected_loss_calibration=Path("missing_selected.json"))
    elif kind=="fit_without_check":values["adaptive_check_artifact"]=None
    elif kind=="legacy":values["protocol_scope"]="legacy"
    elif kind=="unsupported_command":values["command"]="evaluate_baseline"
    else:values["baseline_name"]="transolver"
    args=SimpleNamespace(**values)
    with pytest.raises(ValueError):cli.run_native_command(args,{})
    argv=["bridge","--protocol-scope",values["protocol_scope"],"--baseline-name",values["baseline_name"],
        "--command",values["command"],"--output-dir",str(tmp_path),"--residual-artifact",values["residual_artifact"]]
    for key in ("adaptive_fit_artifact","adaptive_check_artifact","selected_loss_calibration"):
        if values[key]:argv += ["--"+key.replace("_","-"),str(values[key])]
    monkeypatch.setattr(sys,"argv",argv)
    with pytest.raises(SystemExit) as caught:cli.parse_args()
    assert caught.value.code==2
    if kind in {"orphan_check","selected_without_fit","selected_without_check","fit_without_check"}:
        def forbidden(*args,**kwargs):pytest.fail("Invalid dependencies reached source/model/private work")
        monkeypatch.setattr(consumer,"predose_source_admission",forbidden)
        with pytest.raises(ValueError):consumer.evaluate_native_residual_controls({},object(),object(),{},object(),{},{},tmp_path,
            adaptive_artifact={} if values["adaptive_fit_artifact"] else None,
            check_artifact={} if values["adaptive_check_artifact"] else None,
            selected_loss_calibration={} if values["selected_loss_calibration"] else None,world_factory=forbidden)
    assert not (tmp_path/"native_residual_control_evaluation.json").exists()


def test_molecular_opt_in_collection_reload_preserves_nested_prefix_and_readonly_history(tmp_path,monkeypatch):
    from Libs.utils.evaluation import native_residual_control_evaluation as consumer
    from Libs.utils.data.native_actuation import native_actuation_library,NativeActuationRequest,executed_prefix,compose_actuation,actuation_payload
    from Libs.utils.model.native_observable_source import MOLECULAR
    from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference,AdaptiveReadoutContext,export_adaptive_readout_history
    from Libs.tests.test_native_conditional_readout import reference
    from Libs.tests.test_native_adaptive_readout import capture
    config=load_config("Libs/configs/exp/native_molecular_administered_rate.yaml")
    declaration={"family_uid":MOLECULAR,"source_variant":"primary","action_indices":[0,4],
        "sample_uid":"molecular_collection_fixture","role":"calibration","calibration_reservation":"fit"}
    declaration.pop("action_indices")
    library=native_actuation_library(config,declaration)
    first,second=library[0],library[4]
    baseline=reference()
    binding={"normalization_id":baseline.normalization_id,"checkpoint_sha256":"artificial_predictor"}
    readout=replace(baseline.readout,source_uid=first.source_uid,source_snapshot_ref=first.source_snapshot_ref,
        family_uid=MOLECULAR,role="calibration")
    fixed=replace(baseline,readout=readout,sample_uids=(declaration["sample_uid"],),
        executed_regimen_uid=first.regimen_uid,predictor_binding=json.dumps(binding,sort_keys=True),
        source_times=first.source_times,time_hours=first.source_times,source_time_unit="hour",reference_time_hours=6.)
    ref=AdaptiveReadoutReference(fixed,library,first.regimen_uid);ref.validate()
    request=NativeActuationRequest(executed_prefix(first,1),second)
    joined=compose_actuation(request,library)
    revision=json.dumps(dict(binding,residual_optimizer_updates=0),sort_keys=True)
    contexts=(AdaptiveReadoutContext(first,first,(),revision),
              AdaptiveReadoutContext(second,joined,(request,),revision))
    rows=tuple(capture(ref,context,index,index-1,selected=0) for index,context in enumerate(contexts,1))
    initial=tuple(torch.zeros(32,dtype=torch.float64).log_softmax(0).tolist())
    snapshot={key:consumer._collection_public_config(config[key]) for key in (
        "native_residual_control","native_temporal_query","native_mechanistic_residual_online","native_arrived_observation_filter")}
    report_context={"group_uid":declaration["sample_uid"],"family_uid":MOLECULAR,
        "binding":binding,"reference_id":"artificial_reference","pre_execution_policy_snapshot":snapshot}
    history=export_adaptive_readout_history(ref,rows,initial,report_context=report_context,episode_status="component_complete")
    # Transport fixture supplies records, not independent episode authentication.
    history["origin"]="episode_authenticated_actual_acceptance"
    metadata_receipt=tmp_path/"artificial_receipt.json"
    metadata_receipt.write_text(json.dumps({"scope":"artificial_software_fixture_no_source_trajectory",
        "physical_source_trajectory_calls":0,"target_generation":"initial_complete_tuple_zero_plus_masked_linear_ramp"}))
    dataset,declarations=_artificial_calibration_metadata([declaration],metadata_receipt)
    admission=consumer.predose_source_admission(config,dataset)
    frozen_context={"purpose":"adaptive_fit","reservation":"fit","phase":"adaptive_fit",
        "selected_predictor":{"scope":"artificial_software_fixture","scientific_checkpoint_sufficiency":False},
        "source_admission":admission,"roles":{"fit":[declaration["sample_uid"]],"check":[]},"threshold_group_uids":[]}
    frozen=consumer._collection_binding(config,binding,{"reference_id":"artificial_reference"},
        {"fit_group_uids":[],"check_group_uids":[]},frozen_context)
    episode_binding={"collection_binding_id":digest(frozen),"declaration":consumer._collection_public_config(declaration),
        "adaptive_reference":actuation_payload(ref),"original_action_library":[actuation_payload(a) for a in library],
        "initial_log_weights":list(initial)}
    report={"schema_version":"native_residual_calibration_collection.v1","status":"software_collection_complete",
        "reservation":"fit","collection_binding":frozen,"collection_binding_id":digest(frozen),
        "binding":binding,"reference_id":"artificial_reference","pre_execution_policy_snapshot":snapshot,
        "denominator":{"software_fixture_only":True,"production_complete":False},
        "groups":[{"group_uid":declaration["sample_uid"],"family_uid":MOLECULAR,
            "status":"component_complete","episode_collection_binding":episode_binding,
            "actual_adaptive_readout_history":history}]}
    monkeypatch.setattr(consumer,"declared_dataset",lambda *args,**kwargs:(dataset,declarations))
    monkeypatch.setattr(consumer,"selected_group_definitions",lambda *args:[declaration])
    monkeypatch.setattr(consumer,"native_cns_readout",lambda *args:readout)
    collection=tmp_path/"collection.json";collection.write_text(json.dumps(report,allow_nan=False))
    before=copy.deepcopy(report)
    _,restored,_=consumer._read_collection_report(collection,config,"fit","adaptive_fit")
    assert restored[0].history==rows and report==before
    assert restored[0].history[-1].context.executed_program==joined
    assert restored[0].history[-1].context.switches[0].prefix.administered_mass_mg_per_kg==first.administered_mass(1)
    altered=copy.deepcopy(report)
    altered["groups"][0]["actual_adaptive_readout_history"]["records"][1]["switches"][0]["prefix"].pop("administered_mass_mg_per_kg")
    collection.write_text(json.dumps(altered,allow_nan=False))
    with pytest.raises(ValueError):consumer._read_collection_report(collection,config,"fit","adaptive_fit")
    collection.write_text(json.dumps(report,allow_nan=False))
    mismatch=copy.deepcopy(config);mismatch["native_molecular_administration"]["law_uid"]="other_law"
    with pytest.raises(ValueError):consumer._read_collection_report(collection,mismatch,"fit","adaptive_fit")
    assert report==before and rows[-1].context.executed_program==joined
