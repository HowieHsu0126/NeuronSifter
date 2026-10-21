from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from Libs.exp.evaluation import evaluate_screening_system as evaluation
from Libs.exp.evaluation.evaluate_rollout_prediction import parse_args, resolve_native_identity
from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.models.modules.field_rollout import FieldRolloutOperator
from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source
from Libs.tests.test_expert_composition import ten_expert_case
from Libs.tests.test_native_field_normalization import joint_normalized_case
from Libs.tests.test_native_rollout_evaluation import development_case
from Libs.tests.test_native_rollout_metrics import metric_case, fields, identity
from Libs.tests.test_native_rollout_identity import canonical_config, metadata_config, invoke
from Libs.tests.test_physical_occupancy import make_packet
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_field_normalization import NativeFieldNormalization, normalization_spec
from Libs.utils.evaluation.native_rollout_metrics import (
    metric_policy, metric_protocol, score_point_mass_episode, score_native_episode,
    build_native_metric_payload, validate_native_metric_payload)
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "Libs/exp/evaluation/evaluate_rollout_prediction.py"


def persistence(config):
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = "initial_state_persistence"
    return config


def reject_learned_rollout(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Persistence invoked a learned future rollout")
    monkeypatch.setattr(ADScreeningSystem, "score_regimens", forbidden)
    monkeypatch.setattr(FieldRolloutOperator, "forward", forbidden)
    monkeypatch.setattr(QuantizedMddmRolloutOperator, "forward", forbidden)


@pytest.mark.parametrize("masked", [False, True])
def test_point_mass_analytic_scores_and_zero_draw_denominators(metric_case, masked):
    config, grid, norm, _, path = metric_case
    config = persistence(copy.deepcopy(config))
    protocol = metric_protocol(metric_policy(config), [.05,.5,.95], norm.identity,
                               draw_semantics="deterministic_point_mass")
    before = path.read_bytes()
    x, y, mask = fields(metric_case, [1.], 3., actions=2)
    y[0] = x[0,0]  # Half the cells are exact zero-width interval hits.
    if masked:
        mask[:] = False
    row = score_point_mass_episode(x[0], y, mask, identity(actions=2), norm, grid, protocol)
    payload = build_native_metric_payload([row], protocol, {}, config)
    assert row["draw_count"] == payload["counts"]["realized_draw_count"] == 0
    assert row["status"] == ("fully_masked" if masked else "defined")
    if not masked:
        assert row["metrics"]["crps"] == pytest.approx(1.)
        assert row["metrics"]["normalized_rmse"] == pytest.approx(2**.5)
        assert row["metrics"]["predictive_coverage"] == .5
        assert payload["metrics"]["predictive_calibration_error"]["mean"] == pytest.approx(.4)
        payload["episodes"][0]["draw_count"] = 2
        with pytest.raises(ValueError, match="zero draws"):
            validate_native_metric_payload(payload, config)
    else:
        assert payload["status"] == "incomplete" and payload["counts"]["undefined_episode_count"] == 1
        assert all(value["mean"] is None for value in payload["metrics"].values())
    assert path.read_bytes() == before
    native_protocol = metric_protocol(metric_policy(config), [.05,.5,.95], norm.identity)
    one = score_native_episode(x, y, torch.ones_like(mask), identity(actions=2), norm, grid, native_protocol)
    assert one["status"] == "insufficient_draws"  # Analytic law does not weaken R65 D>=2.


def change_future_programs(config, registry, directory):
    original = registry["joint_sources"][0]
    altered = copy.deepcopy(original)
    for action in altered["particle_programs"]["records"][0][0]:
        action["action"]["dose"] *= 2
    packet = make_packet(directory, config, altered)
    for export in packet["exports"]:
        csv = directory / export["csv_path"]
        frame = pd.read_csv(csv)
        frame["simulationValues"] *= 3
        frame.to_csv(csv, index=False)
        snapshot_path = directory / export["run_snapshot_path"]
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["export_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
        snapshot_path.write_text(json.dumps(snapshot))
    changed = build_native_physical_source(config, packet, directory)
    assert changed["support"] == original["support"]
    assert changed["observations"] == original["observations"]
    assert changed["particle_programs"]["occupancy"] != original["particle_programs"]["occupancy"]
    registry["joint_sources"] = [changed]
    for row in registry["teachers"]:
        bi = changed["support"]["sample_uids"].index(row["sample_uid"])
        ai = changed["particle_programs"]["regimen_uids"].index(row["regimen_uid"])
        row["program_uid"] = changed["particle_programs"]["records"][bi][0][ai]["program_uid"]


def test_real_joint_persistence_initial_mixture_and_program_target_invariance(
    ten_expert_case, tmp_path, monkeypatch
):
    config, registry, arrays, store, store_format, checkpoint = development_case(ten_expert_case, tmp_path)
    persistence(config)
    reject_learned_rollout(monkeypatch)
    events, predictions, posteriors = [], [], []
    read, predict, infer = native.load_records, evaluation.predict_native_episode, ADScreeningSystem.infer_world_state
    def watched_read(*args, **kwargs):
        events.append(("read", kwargs.get("include_teacher", True)))
        return read(*args, **kwargs)
    def watched_infer(*args, **kwargs):
        output = infer(*args, **kwargs)
        posteriors.append(output["joint_posterior"])
        return output
    def watched_predict(*args, **kwargs):
        events.append(("predict", None))
        output = predict(*args, **kwargs)
        predictions.append(output)
        return output
    monkeypatch.setattr(native, "load_records", watched_read)
    monkeypatch.setattr(ADScreeningSystem, "infer_world_state", watched_infer)
    monkeypatch.setattr(evaluation, "predict_native_episode", watched_predict)
    first, _ = evaluation.evaluate_native_rollout_prediction(config, checkpoint)
    assert events == [("read",False),("predict",None),("read",True)] * 2
    assert len(posteriors) == 2
    assert first["counts"]["episode_count"] == first["counts"]["group_count"] == 2
    assert first["counts"]["valid_cell_count"] == 6480
    assert first["counts"]["realized_draw_count"] == 2
    saved = []
    for output, posterior in zip(predictions, posteriors):
        assert not torch.allclose(posterior.weights, torch.full_like(posterior.weights, .5))
        expected = posterior.support.initial_field[:, :, None, None].expand(-1,-1,2,3,-1,-1,-1)
        assert torch.equal(output["particle_field_trajectory"], expected)
        for draw in range(2):
            index = output["particle_draw_indices"][0,draw]
            assert torch.equal(output["field_trajectory_samples"][draw,0], expected[0,index])
        assert "router_diagnostics" not in output and "field_training_reconstruction" not in output
        law = output["prediction_law"]
        assert law["initial_state_uncertainty"] == "inherited_initial_posterior_only"
        assert law["fitted_transition_parameter_count"] == law["transition_model_call_count"] == law["denoising_call_count"] == 0
        saved.append(output["field_trajectory_samples"].clone())
    change_future_programs(config, registry, tmp_path / "changed-future")
    write_native_microenvironment_store(store, config, registry, arrays, store_format)
    events.clear()
    predictions.clear()
    second, _ = evaluation.evaluate_native_rollout_prediction(config, checkpoint)
    assert events == [("read",False),("predict",None),("read",True)] * 2
    assert all(torch.equal(x, y["field_trajectory_samples"]) for x,y in zip(saved,predictions))
    assert first["metrics"] == second["metrics"]
    arrays["trajectory"] += 17.  # Includes teacher t0: it cannot supply the persisted initial state.
    for row in registry["teachers"]:
        row["content_hash"] += "-perturbed-independent-target"
    write_native_microenvironment_store(store, config, registry, arrays, store_format)
    predictions.clear()
    third, _ = evaluation.evaluate_native_rollout_prediction(config, checkpoint)
    assert len(posteriors) == 6  # Exactly one real posterior inference per episode/call.
    assert all(torch.equal(x, y["field_trajectory_samples"]) for x,y in zip(saved,predictions))
    assert second["metrics"]["crps"]["mean"] != third["metrics"]["crps"]["mean"]


def test_real_point_persistence_analytic_values_without_rollout(ten_expert_case, tmp_path, monkeypatch):
    config, registry, arrays, store, store_format, checkpoint = development_case(ten_expert_case, tmp_path, point=True)
    persistence(config)
    reject_learned_rollout(monkeypatch)
    norm = NativeFieldNormalization(normalization_spec(registry["grid"], config), 15)
    initial = arrays["initial_state"][0]
    arrays["trajectory"] = np.stack([
        np.broadcast_to(initial + float(error)*norm.scale.numpy(), (3,6,6,15)).copy()
        for error in (1.,3.)]).astype(np.float32)
    for row in registry["teachers"]:
        row["content_hash"] += "-analytic-reference"
    write_native_microenvironment_store(store, config, registry, arrays, store_format)
    result, _ = evaluation.evaluate_native_rollout_prediction(config, checkpoint)
    assert result["status"] == "complete"
    assert result["protocol"]["draw_semantics"] == "deterministic_point_mass"
    assert result["protocol"]["crps_estimator"] == "analytic_point_mass_absolute_error"
    assert result["counts"]["realized_draw_count"] == result["episodes"][0]["draw_count"] == 0
    assert result["metrics"]["crps"]["mean"] == pytest.approx(2., abs=1e-5)
    assert result["metrics"]["normalized_rmse"]["mean"] == pytest.approx(5**.5, abs=1e-5)
    assert result["metrics"]["predictive_coverage"]["mean"] == 0
    law = result["episodes"][0]["prediction_provenance"]["prediction_law"]
    assert law["initial_state_uncertainty"] == "theoretical_point_mass_baseline_assumption"
    assert result["provenance"]["loaded_checkpoint_parameter_count"] > 0


@pytest.mark.parametrize("mutation", ["inconsistent", "masked", "missing_observation"])
def test_point_persistence_invalid_initial_and_missingness(ten_expert_case, tmp_path, monkeypatch, mutation):
    config, registry, arrays, store, store_format, checkpoint = development_case(ten_expert_case, tmp_path, point=True)
    persistence(config)
    reject_learned_rollout(monkeypatch)
    if mutation == "missing_observation":
        registry["observations"] = []
        path = tmp_path / "missing-observation.json"
        path.write_text(json.dumps(registry))
        config["native_diffusion_data"]["registry_path"] = str(path)
        with pytest.raises(ValueError, match="missing observation"):
            evaluation.evaluate_native_rollout_prediction(config, checkpoint)
        return
    if mutation == "masked":
        for row in registry["teachers"]:
            row["validity_mask"] = {"kind":"explicit", "evidence_ref":"all-missing-fixture",
                                    "values":np.zeros((3,6,6,15),dtype=bool).tolist()}
        write_native_microenvironment_store(store, config, registry, arrays, store_format)
    else:
        prepare = evaluation.prepare_native_episode
        def inconsistent_prepared_input(*args, **kwargs):
            patient, context = prepare(*args, **kwargs)  # Real admission and source read still execute.
            initial = context["diffusion_context"]["initial_state"].clone()
            initial[:,1] += 1
            context["diffusion_context"]["initial_state"] = initial
            return patient, context
        monkeypatch.setattr(evaluation, "prepare_native_episode", inconsistent_prepared_input)
    result, _ = evaluation.evaluate_native_rollout_prediction(config, checkpoint)
    assert result["status"] == "incomplete" and result["counts"]["episode_count"] == 1
    row = result["episodes"][0]
    assert row["draw_count"] == result["counts"]["realized_draw_count"] == 0
    assert row["status"] == ("failed" if mutation == "inconsistent" else "fully_masked")
    if mutation == "inconsistent":
        assert "agree across all actions" in row["reason"]


def test_real_cli_persistence_keeps_conditioning_profile_and_distinct_default_namespace(ten_expert_case, tmp_path):
    profile = "rollout_operator_reversed_experts"
    config, _, _, _, _, checkpoint = development_case(ten_expert_case, tmp_path, profile)
    entry = config["paper_grade_full_suite"]["entrypoint_registry"]["rollout_prediction"]
    entry["output_template"] = str(tmp_path / entry["output_template"])
    outputs = []
    for kind, model in [("native_diffusion","neuronsifter_reversed_experts"),
                        ("initial_state_persistence","native_initial_state_persistence")]:
        config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = kind
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        argv = [sys.executable,str(CLI),"--execution-mode","native","--config",str(path),
                "--checkpoint",str(checkpoint),"--freeze-manifest",str(tmp_path/"absent-freeze.json")]
        run = subprocess.run(argv,cwd=ROOT,text=True,capture_output=True)
        assert run.returncode == 0, run.stdout + run.stderr
        output = Path(entry["output_template"].format(model_family=model, seed=41))
        expected_run_id = f"rollout_prediction_{model}_seed_41"
        if kind == "initial_state_persistence":
            output = output / profile
            expected_run_id += "_" + profile
        payload = json.loads((output/"metrics.json").read_text())
        metadata = json.loads((output/"run_metadata.json").read_text())
        manifest = json.loads((output/"artifact_manifest.json").read_text())
        assert metadata["model_family"] == manifest["model_key"] == model
        assert metadata["experiment_family"] == payload["provenance"]["selected_experiment_family"] == profile
        assert payload["provenance"]["checkpoint_path"] == str(checkpoint)
        assert payload["provenance"]["training_seed_binding"]["selected_formal_seed"] == 41
        assert payload["provenance"]["predictor_kind"] == kind
        assert metadata["run_id"] == expected_run_id
        outputs.append(output)
    assert outputs[0] != outputs[1]
    wrong = subprocess.run(argv+["--model-family","neuronsifter_reversed_experts"],
                           cwd=ROOT,text=True,capture_output=True)
    assert wrong.returncode != 0 and "Native model label" in wrong.stderr


@pytest.mark.parametrize("mutation", ["seed", "protected"])
def test_persistence_preserves_checkpoint_and_source_guards(ten_expert_case, tmp_path, monkeypatch, mutation):
    config, registry, _, _, _, checkpoint = development_case(ten_expert_case, tmp_path)
    persistence(config)
    if mutation == "seed":
        config["selected_formal_seed"] = 43
        config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"] = 43
        reason = "(?i)seed"
    else:
        registry["sources"][0]["protected"] = True
        path = tmp_path / "protected-metadata.json"
        path.write_text(json.dumps(registry))
        config["native_diffusion_data"]["registry_path"] = str(path)
        reason = "source: public/license/release/snapshot/split/unit evidence required"
        calls = []
        def forbidden_read_or_prediction(*args, **kwargs):
            calls.append("unexpected_source_read_or_prediction")
            raise AssertionError("Protected source reached payload reading or prediction")
        monkeypatch.setattr(native, "load_records", forbidden_read_or_prediction)
        monkeypatch.setattr(evaluation, "predict_native_episode", forbidden_read_or_prediction)
    with pytest.raises(ValueError, match=reason):
        evaluation.evaluate_native_rollout_prediction(config, checkpoint)
    if mutation == "protected":
        assert calls == []


def test_persistence_identity_retains_upstream_profile_admission(metric_case):
    config = persistence(copy.deepcopy(metric_case[0]))
    config["selected_experiment_family"] = "undeclared_conditioning_profile"
    with pytest.raises(ValueError, match="one unambiguous existing"):
        resolve_native_identity(parse_args(["--execution-mode","native"]), config)


def test_persistence_default_namespace_includes_conditioning_profile_and_preserves_overrides(canonical_config, tmp_path):
    outputs, run_ids = [], []
    for profile in ("dynamics_mainline_ad", "rollout_operator_reversed_experts"):
        config = persistence(metadata_config(canonical_config, tmp_path, profile))
        run, _ = invoke(config, tmp_path, ["--execution-mode","native","--dry-run"])
        assert run.returncode == 0, run.stdout + run.stderr
        output = tmp_path/"Output/results/paper_grade_full_suite/manifests/rollout_prediction/native_initial_state_persistence/seed_41"/profile
        metadata = json.loads((output/"run_metadata.json").read_text())
        status = json.loads((output/"status.json").read_text())
        manifest = json.loads((output/"artifact_manifest.json").read_text())
        assert metadata["model_family"] == status["model_family"] == manifest["model_key"] == "native_initial_state_persistence"
        assert metadata["experiment_family"] == profile
        assert metadata["run_id"] == "rollout_prediction_native_initial_state_persistence_seed_41_" + profile
        outputs.append(output)
        run_ids.append(metadata["run_id"])
    assert len(set(outputs)) == len(set(run_ids)) == 2
    explicit = tmp_path/"explicit-user-output"
    run, _ = invoke(config, tmp_path, ["--execution-mode","native","--dry-run",
        "--output-dir",str(explicit),"--run-id","user-selected-run"])
    assert run.returncode == 0, run.stdout + run.stderr
    metadata = json.loads((explicit/"run_metadata.json").read_text())
    assert metadata["run_id"] == "user-selected-run"
    assert metadata["experiment_family"] == "rollout_operator_reversed_experts"
