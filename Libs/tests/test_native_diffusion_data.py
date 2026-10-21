from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from Libs.utils.data.action_observation_contracts import SCHEMA_VERSIONS, compile_pharmacological_action
from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.model.microenvironment_teacher import write_microenvironment_zarr, write_native_microenvironment_store
from Libs.tests.test_screening_system_multiscale import neuronsifter_test_backbone


def producer_case(tmp_path, store_format="zarr_v2", action_policy="occupancy_program"):
    config = load_config("Libs/configs/config_stack.yaml")
    from Libs.utils.model.microenvironment_teacher import default_coordinate_payload
    grid = default_coordinate_payload(config)
    grid["time_hours"] = [0., 1., 2.]
    grid.update(physical_time_unit="hour", species_units={name: "fixture_unit" for name in grid["species_id"]},
                normalization={"kind": "physical"})
    config["native_diffusion_data"].update(species_units=copy.deepcopy(grid["species_units"]), posterior_context_width=12,
                                           registry_path=str(tmp_path / "registry.json"))
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    benchmark["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    benchmark["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] = action_policy
    backbone = benchmark["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(neuronsifter_test_backbone())
    backbone["diffusion_sampler"]["sampler_seed"] = 41
    backbone["quantized_mddm_rollout"] = {
        "enabled": True, "registry_binding": "quantized_mddm_main_rollout",
        "runtime_overrides": {"sampler_seed": 41}}
    config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"].update(
        training_sample_count=1, training_denoising_step_count=1)
    store = tmp_path / "source"
    sources = [{"source_uid": "source", "public": True, "protected": False, "license": "synthetic-fixture",
                "release": "fixture-v1", "snapshot_ref": "fixture-observations-v1",
                "split_evidence_ref": "fixture-training-split", "unit_evidence_ref": "fixture-unit-contract"}]
    teachers, programs, observations = [], [], []
    for sample_index, sample in enumerate(("sample-a", "sample-b")):
        posterior = "posterior-" + sample
        observations.append({"posterior_uid": posterior, "sample_uid": sample, "source_uid": "source",
                             "origin": "observation_derived", "observation_ref": "fixture-observation-" + sample,
                             "source_snapshot_ref": sources[0]["snapshot_ref"], "initial_time_hours": 0,
                             "store_path": str(store), "store_format": store_format, "observation_index": sample_index})
        for regimen_index, regimen in enumerate(("regimen-a", "regimen-b")):
            index = 2 * sample_index + regimen_index
            action = {"schema_version": SCHEMA_VERSIONS["world_action"], "action_uid": "action-" + regimen,
                      "intervention_type": "pharmacological", "target_set": ["target"], "agent_set": ["fixture-agent-" + regimen],
                      "dose": 1., "dose_unit": "mg/kg", "schedule": {"kind": "interval", "interval": 1, "unit": "day"},
                      "route": "intravenous", "duration": {"value": 2, "unit": "day"},
                      "state_scope": {"regions": grid["region_id"]}, "support_domain_ref": "fixture-domain",
                      "source_evidence_refs": ["fixture-action-source"], "action_provenance_ref": "fixture-action-v1",
                      "conflicts": [], "status": "contract_valid"}
            pkpd = {"time_grid": grid["time_hours"], "parameter_provenance": {"status": "verified", "artifact_ref": "fixture-pkpd"},
                    "target_program": {"targets": ["target"]}, "exposure_fields": {"array_name": "fixture-exposure"},
                    "occupancy_fields": {"array_name": "occupancy", "support_array": "support"},
                    "organism_context": "fixture", "supported_organisms": ["fixture"], "uncertainty": {"kind": "supplied"}}
            baseline, graph = {"state_ref": posterior}, {"graph_ref": "fixture-cns-graph"}
            compiled = compile_pharmacological_action(action, pkpd_context=pkpd, baseline_state_posterior=baseline, graph_context=graph)
            assert compiled["status"] == "compiled_with_explicit_uncertainty", compiled
            programs.append({"program_uid": compiled["program_uid"], "posterior_uid": posterior, "source_uid": "source",
                             "action_uid": action["action_uid"], "regimen_uid": regimen, "target_program_uid": "target-program",
                             "representation": "explicit_grid", "action": action, "pkpd_context": pkpd,
                             "baseline_state_posterior": baseline, "graph_context": graph, "compiled_program": compiled,
                             "store_path": str(store), "store_format": store_format, "program_index": index})
            teachers.append({"trajectory_uid": "trajectory-" + str(index), "source_uid": "source", "sample_uid": sample,
                             "regimen_uid": regimen, "action_uid": action["action_uid"], "target_program_uid": "target-program",
                             "program_uid": compiled["program_uid"], "posterior_uid": posterior,
                             "family_uid": "family-" + sample, "content_hash": "fixture-content-" + str(index),
                             "trajectory_store_path": str(store), "store_format": store_format, "trajectory_index": index,
                             "teacher_family": "full_microenvironment_teacher", "qc_catalog_ref": "fixture-qc", "source_snapshot_ref": sources[0]["snapshot_ref"],
                             "teacher_confidence": (index + 1) / 4, "uncertainty_origin": "independent_supplied",
                             "qc_grade": "B", "qc_use": "main_pretraining", "sample_weight": .5,
                             "split_name": "train", "source_holdout_role": "train"})
    registry = {"schema_version": native.SCHEMA_VERSION, "grid": grid, "sources": sources,
                "teachers": teachers, "programs": programs, "observations": observations}
    rng = np.random.default_rng(41)
    shape = (4, 3, 6, 6, 1)
    arrays = {"trajectory": rng.uniform(.2, .8, (4, 3, 6, 6, 15)).astype("float32"),
              "uncertainty": rng.uniform(.1, .4, (4, 15)).astype("float32"),
              "teacher_confidence": np.array([.25, .5, .75, 1.], dtype="float32"),
              "initial_state": rng.uniform(.1, .5, (2, 6, 6, 15)).astype("float32"),
              "posterior_context": rng.uniform(.1, .5, (2, 6, 6, 12)).astype("float32"),
              "occupancy": rng.uniform(.1, .9, shape).astype("float32"), "support": np.ones(shape, dtype="float32"),
              "physical_time": np.broadcast_to(np.arange(3, dtype="float32").reshape(1, 3, 1, 1, 1), shape).copy()}
    if store_format == "zarr_v2":
        write_microenvironment_zarr(store, arrays["trajectory"], arrays["uncertainty"], config, {},
                                   native_payload={"registry": registry, "arrays": arrays})
    else:
        write_native_microenvironment_store(store, config, registry, arrays, store_format)
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    return config, registry, arrays, store


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
@pytest.mark.parametrize("action_policy", ["occupancy_program", "drug_id", "target_id"])
def test_producer_to_actual_pretrainer_and_cache(tmp_path, monkeypatch, store_format, action_policy):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training import train_screening_system as screening
    config, registry, arrays, store = producer_case(tmp_path, store_format, action_policy)
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    items = dataset.__getitems__([3, 0])
    assert [item["trajectory_uid"] for item in items] == ["trajectory-3", "trajectory-0"]
    assert items[0]["sample_ids"].item() == native.stable_numeric_id("trajectory-3")
    assert np.array_equal(items[0]["initial_state"], arrays["initial_state"][1])
    if store_format == "zarr_v2":
        actual = pretrain.MicroenvironmentZarrDataset(store, config, [2, 0], split_name="train",
                                                       trajectory_uids=["trajectory-2", "trajectory-0"]).__getitems__([0])
        assert actual[0]["trajectory_uid"] == "trajectory-2"
    batch = {key: torch.stack([item[key] for item in items]) for key in (
        "trajectory", "uncertainty", "teacher_confidence", "initial_state", "posterior_context",
        "occupancy", "support", "physical_time", "teacher_loss_weight", "sample_ids")}
    if action_policy in {"drug_id", "target_id"}:
        batch["action_identity"] = torch.stack([item["action_identity"] for item in items])
        assert torch.equal(batch["action_identity"][0], torch.tensor(native.program_action_identity(registry["programs"][3], action_policy=action_policy)))
        # Both admitted regimens share targets but have different agents.
        assert torch.equal(batch["action_identity"][0], batch["action_identity"][1]) == (action_policy == "target_id")
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    metrics, _ = pretrain.run_pretraining_step(model, batch, config, torch.device("cpu"),
        {"mixed_precision": {"enabled": False}, "gradient_accumulation_steps": 1})
    assert np.isfinite(metrics["total_loss"]) and metrics["native_masked_diffusion_loss"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.operator.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.quantized_mddm.condition_projection.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.quantized_mddm.condition_projection.weight)
    # The real cache builder performs admission after its existing sample/regimen selection.
    metadata = pd.DataFrame({"sample_uid": ["sample-b", "sample-a"]})
    payload = {"rna": {"train": {"metadata": metadata, "state_features": np.zeros((2, 4)),
        "labels": np.ones(2), "probabilities": np.ones(2), "latent": np.zeros((2, 2)), "matrix": np.zeros((2, 2))}}}
    selected_regimens = [{"regimen_uid": "regimen-b"}, {"regimen_uid": "regimen-a"}]
    selected_targets = [{"target_program_uid": "target-program"}]
    labels = {"rna": {"train": {"target_labels": np.ones((2, 1)), "regimen_labels": np.ones((2, 2))}}}
    monkeypatch.setattr(screening, "build_training_targets", lambda *args: (selected_targets, selected_regimens, labels))
    monkeypatch.setattr(screening, "target_program_features", lambda rows: (pd.DataFrame(rows), np.ones((1, 2))))
    monkeypatch.setattr(screening, "regimen_features", lambda rows: (pd.DataFrame(rows), np.ones((2, 25))))
    for name in ("observed_region_mask", "world_state_support_vectors", "world_state_factor_targets", "world_state_supervision_masks"):
        monkeypatch.setattr(screening, name, lambda *args: np.ones((2, 1)))
    cache = screening.build_scope_training_cache_payload(config, "fixture", "train", payload, [], [], [], [], [],
        {"world_state": pd.DataFrame(), "counterfactual_benchmark": pd.DataFrame()}, "fixture")
    result = cache["modality_payload"]["rna"]["train"]["native_diffusion"]
    assert result["trajectory_uids"] == ["trajectory-3", "trajectory-2", "trajectory-1", "trajectory-0"]
    assert np.array_equal(result["initial_state"][0, 0], arrays["initial_state"][1])
    monkeypatch.setattr(screening, "source_artifact_hashes", lambda *args: {})
    monkeypatch.setattr(screening, "freeze_signature_sha256", lambda *args: "fixture-freeze")
    artifact = tmp_path / "cache"
    screening.write_scope_training_cache(config, tmp_path / "config.yaml", ["fixture"], artifact, "fixture", "train", 54, cache, 0.)
    loaded, _, _ = screening.load_scope_training_cache(config, tmp_path / "config.yaml", artifact, "fixture", "train", 54)
    current = loaded["modality_payload"]["rna"]["train"]
    selected = screening.native_diffusion_training_kwargs(
        SimpleNamespace(field_rollout_operator=SimpleNamespace(quantized_mddm_rollout_operator=object(), field_normalization=NativeFieldNormalization(None,15), action_policy_name=action_policy)),
        current, [0], [1], torch.device("cpu"))
    assert torch.equal(selected["diffusion_context"]["initial_state"], torch.from_numpy(arrays["initial_state"][1:2, None]))
    assert selected["regimen_ids"].tolist() == [native.stable_numeric_id("regimen-a")]
    if action_policy in {"drug_id", "target_id"}:
        assert torch.equal(selected["diffusion_context"]["action_identity"][0, 0],
                           torch.tensor(native.program_action_identity(registry["programs"][2], action_policy=action_policy)))
        for key, value in (("action_identity_schema", "stale"), ("action_policy", "occupancy_program"),
                           ("action_identity", np.zeros_like(result["action_identity"]))):
            broken = copy.deepcopy(loaded["modality_payload"])
            broken["rna"]["train"]["native_diffusion"][key] = value
            with pytest.raises(ValueError, match="cache_stale"):
                native.validate_cached_native(broken, ["regimen-b", "regimen-a"], config)
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.tests.test_screening_system_multiscale import solver_test_kwargs
    from Libs.utils.model.screening_model_config import baseline_field_policy_from_config, solver_config_from_config, solver_state_layout_from_config, rollout_update_policy_from_config
    solver = solver_test_kwargs()
    # Config-only policies: the convenience full constructor reads real region priors.
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone["mechanism_moe"]["router_loss_weight"] = 0.0
    solver.update(baseline_field_policy=baseline_field_policy_from_config(config),
                  solver_config=solver_config_from_config(config),
                  solver_state_layout=solver_state_layout_from_config(config),
                  rollout_update_policy=rollout_update_policy_from_config(config))
    solver["rollout_update_policy"]["transition_backbone"] = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    # Expand the registry-bound diffusion config for this direct constructor.
    from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config
    solver["rollout_update_policy"]["transition_backbone"] = copy.deepcopy(solver["rollout_update_policy"]["transition_backbone"])
    solver["rollout_update_policy"]["transition_backbone"]["quantized_mddm_rollout"] = quantized_mddm_rollout_from_config(
        config, solver["rollout_update_policy"]["transition_backbone"])
    screen_model = ADScreeningSystem(
        patient_state_dim=4, target_feature_dim=3, regimen_feature_dim=25, hidden_dim=8, latent_dim=6,
        dropout_rate=0., compartment_count=6, region_count=6, species_count=15, rollout_step_count=3,
        tri_weight_by_component={name: 1. for name in ("calibration", "coverage", "pkpd_physics_consistency", "ranking_stability", "ood_score")},
        adjacency_matrix=[[1.] * 36 for _ in range(36)], region_prior_feature_dim=2,
        default_region_prior=[[0., 0.] for _ in range(6)], fast_state_indices=[0, 1, 5],
        slow_state_indices=[2, 3, 4, 6], **solver)
    seen = {}
    operator = screen_model.field_rollout_operator.quantized_mddm_rollout_operator
    original_forward = operator.forward
    def capture(*args, **kwargs):
        seen.update(initial_state=kwargs["initial_state"], posterior=kwargs["conditioning"]["posterior_context"])
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(operator, "forward", capture)
    score = screen_model.score_regimens(torch.zeros(1, 4), torch.as_tensor(cache["regimen_feature_matrix"][1:2]), **selected)
    score["rollout_auxiliary_loss"].backward()
    assert torch.equal(seen["initial_state"], selected["diffusion_context"]["initial_state"])
    assert torch.equal(seen["posterior"], selected["diffusion_context"]["posterior_context"])
    assert torch.isfinite(score["rollout_auxiliary_loss"]) and score["rollout_auxiliary_loss"] > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in screen_model.field_rollout_operator.neuronsifter_world_action_operator.parameters())
    optimizer = torch.optim.AdamW(screen_model.parameters(), lr=.001)
    before = operator.condition_projection.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, operator.condition_projection.weight)
    # Exact consumed-source descriptor changes invalidate before tensor-cache loading.
    descriptor = store / (".zattrs" if store_format == "zarr_v2" else "store_manifest.json")
    descriptor.write_text(descriptor.read_text() + " ")
    monkeypatch.setattr(screening.torch, "load", lambda *args, **kwargs: pytest.fail("stale cache payload was read"))
    with pytest.raises(RuntimeError, match="cache_stale"):
        screening.load_scope_training_cache(config, tmp_path / "config.yaml", artifact, "fixture", "train", 54)


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r["teachers"][0].pop("posterior_uid"), "posterior_uid"),
    (lambda r: r["teachers"][0].update(teacher_confidence="medium"), "numeric"),
    (lambda r: r["teachers"][0].update(uncertainty_origin="legacy_catalog_fallback"), "uncertainty"),
    (lambda r: r["teachers"][0].update(qc_grade="S"), "qc_grade"),
    (lambda r: r["teachers"][0].update(split_name="test"), "role"),
    (lambda r: r["teachers"].append(copy.deepcopy(r["teachers"][0])), "duplicate"),
    (lambda r: r["grid"].update(physical_time_unit="day"), "hour"),
    (lambda r: r["grid"]["species_units"].update(free_drug="wrong"), "units"),
    (lambda r: r["grid"]["region_id"].reverse(), "ordering"),
    (lambda r: r["programs"][0].update(regimen_uid="wrong"), "identity"),
    (lambda r: r["observations"][0].update(sample_uid="wrong"), "identity"),
    (lambda r: r["programs"][0].update(representation="global_scalar"), "grid"),
    (lambda r: r["programs"][0]["baseline_state_posterior"].update(state_ref="wrong"), "lineage"),
])
def test_metadata_rejects_before_array_access(tmp_path, monkeypatch, mutation, reason):
    config, registry, _, _ = producer_case(tmp_path)
    mutation(registry)
    for name in ("load_chunked_rows", "load_zarr_rows"):
        monkeypatch.setattr(native.dynamics_store, name, lambda *args: pytest.fail("array payload was read"))
    with pytest.raises(ValueError, match=reason):
        native.join_registry(registry, config)


def test_store_identity_and_numerical_contract(tmp_path, monkeypatch):
    config, registry, arrays, store = producer_case(tmp_path)
    records = native.join_registry(registry, config)
    descriptor = store / ".zattrs"
    content = json.loads(descriptor.read_text())
    content["native_diffusion"]["teachers"][0]["trajectory_uid"] = "wrong"
    descriptor.write_text(json.dumps(content))
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", lambda *args: pytest.fail("array payload was read"))
    with pytest.raises(ValueError, match="identity"):
        native.load_records(registry, records, config)


def test_teacher_perturbation_does_not_change_conditioning(tmp_path):
    config, registry, arrays, store = producer_case(tmp_path)
    records = native.join_registry(registry, config)
    before, _ = native.load_records(registry, records, config)
    arrays["trajectory"] += .3
    write_microenvironment_zarr(store, arrays["trajectory"], arrays["uncertainty"], config, {},
                               native_payload={"registry": registry, "arrays": arrays})
    after, _ = native.load_records(registry, records, config)
    assert not np.array_equal(before["trajectory"], after["trajectory"])
    for name in ("initial_state", "posterior_context", "occupancy", "support", "physical_time"):
        assert np.array_equal(before[name], after[name])


def test_native_unified_transfer_preserves_source_indices(tmp_path):
    from Libs.scripts.builders.build_full_microenvironment_teacher_registry import rebuild_primary_trajectory_store
    config, registry, arrays, _ = producer_case(tmp_path, "neuronsifter_chunk_store")
    rows = rebuild_primary_trajectory_store(config, tmp_path, {"trajectory_store": "derived"}, [registry["teachers"][3], registry["teachers"][0]])
    assert [row["source_trajectory_index"] for row in rows] == [3, 0]
    derived = json.loads((tmp_path / "derived" / ".zattrs").read_text())["native_diffusion"]
    loaded, _ = native.load_records(derived, native.join_registry(derived, config), config)
    assert np.array_equal(loaded["trajectory"], arrays["trajectory"][[3, 0]])
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config["native_diffusion_data"]["registry_path"] = str(tmp_path / "derived" / ".zattrs")
    dataset = pretrain.MicroenvironmentZarrDataset(tmp_path / "derived", config, split_name="train")
    assert [item["sample_ids"].item() for item in dataset.__getitems__([1, 0])] == [
        native.stable_numeric_id("trajectory-0"), native.stable_numeric_id("trajectory-3")]


def test_bounded_metadata_uses_exact_denominator(tmp_path):
    path = tmp_path / "qc.tsv"
    path.write_text("teacher_confidence\tqc_grade\tqc_use\tsample_weight\tsplit_name\tsource_holdout_role\n"
                    + "medium\tB\tmain_pretraining\t0.5\ttrain\ttrain\n" * 3)
    report = native.bounded_metadata_report(path, 2)
    assert report["inspected_rows"] == report["rejected_rows"] == 2
    assert report["admitted_rows"] == report["array_payload_reads"] == 0


def test_same_sample_action_comparison_rejects_mixed_posterior(tmp_path):
    config, registry, _, _ = producer_case(tmp_path)
    registry["teachers"][1]["posterior_uid"] = "alternate-posterior"
    registry["programs"][1]["posterior_uid"] = "alternate-posterior"
    registry["programs"][1]["baseline_state_posterior"]["state_ref"] = "alternate-posterior"
    observation = dict(registry["observations"][0], posterior_uid="alternate-posterior")
    registry["observations"].append(observation)
    with pytest.raises(ValueError, match="same sample"):
        native.join_registry(registry, config)


@pytest.mark.parametrize("key,mutation,reason", [
    ("uncertainty", lambda value: np.zeros_like(value), "uncertainty"),
    ("teacher_confidence", lambda value: value * np.nan, "numeric"),
    ("initial_state", lambda value: value[:, :, :-1], "axes"),
    ("physical_time", lambda value: value + 24, "hour"),
    ("occupancy", lambda value: value + 2, "outside"),
])
def test_numerical_inputs_rejected_by_real_writer(tmp_path, key, mutation, reason):
    config, registry, arrays, store = producer_case(tmp_path)
    arrays[key] = mutation(arrays[key])
    with pytest.raises((ValueError, IndexError), match=reason):
        write_native_microenvironment_store(store, config, registry, arrays)


def test_cache_identity_reordering_is_rejected(tmp_path):
    config, _, _, _ = producer_case(tmp_path)
    payload = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a", "regimen-b"])
    with pytest.raises(ValueError, match="identity"):
        native.validate_cached_native({"rna": {"train": {"sample_uids": ["sample-b"], "native_diffusion": payload}}},
                                      ["regimen-a", "regimen-b"])


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
@pytest.mark.parametrize("action_policy", ["occupancy_program", "drug_id", "target_id"])
def test_same_sample_regimens_reach_loss_backward_with_stable_ids(tmp_path, store_format, action_policy):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, registry, _, _ = producer_case(tmp_path, store_format, action_policy)
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    items = dataset.__getitems__([0, 1])
    assert [item["sample_uid"] for item in items] == ["sample-a", "sample-a"]
    assert items[0]["posterior_uid"] == items[1]["posterior_uid"]
    batch = {key: torch.stack([item[key] for item in items]) for key in (
        "trajectory", "uncertainty", "teacher_confidence", "initial_state", "posterior_context",
        "occupancy", "support", "physical_time", "teacher_loss_weight", "sample_ids")}
    if action_policy in {"drug_id", "target_id"}:
        batch["action_identity"] = torch.stack([item["action_identity"] for item in items])
        assert torch.equal(batch["action_identity"][0], torch.tensor(native.program_action_identity(registry["programs"][0], action_policy=action_policy)))
        # Both admitted regimens share targets but have different agents.
        assert torch.equal(batch["action_identity"][0], batch["action_identity"][1]) == (action_policy == "target_id")
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    metrics, _ = pretrain.run_pretraining_step(model, batch, config, torch.device("cpu"),
        {"mixed_precision": {"enabled": False}, "gradient_accumulation_steps": 1})
    assert np.isfinite(metrics["total_loss"]) and metrics["native_masked_diffusion_loss"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.operator.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.quantized_mddm.condition_projection.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.quantized_mddm.condition_projection.weight)
    ids = [item["sample_ids"].item() for item in items]
    assert ids == [native.stable_numeric_id("trajectory-0"), native.stable_numeric_id("trajectory-1")]
    assert len(set(ids)) == 2
    reordered = dataset.__getitems__([1, 0])
    chunked = [dataset[index] for index in [1, 0]]
    for original, reordered_item, chunked_item in zip(reversed(items), reordered, chunked):
        assert original["sample_ids"].item() == reordered_item["sample_ids"].item() == chunked_item["sample_ids"].item()
        assert torch.equal(original["trajectory"], reordered_item["trajectory"])
        assert torch.equal(original["posterior_context"], chunked_item["posterior_context"])


def test_normal_fold_cache_excludes_test_native_teachers(tmp_path, monkeypatch):
    from Libs.exp.training import train_screening_system as screening
    config, registry, arrays, store = producer_case(tmp_path)
    for row in registry["teachers"][2:]:
        row.update(split_name="validation", source_holdout_role="validation")
    write_native_microenvironment_store(store, config, registry, arrays)
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    samples = {"train": "sample-a", "validation": "sample-b", "test": "sample-heldout"}
    def fold_payload(config, fold, role, disease, benchmark):
        return {modality: {"metadata": pd.DataFrame({"sample_uid": [samples[role]]}),
            "state_features": np.zeros((1, 4)), "labels": np.ones(1), "probabilities": np.ones(1),
            "latent": np.zeros((1, 2)), "matrix": np.zeros((1, 2))}
            for modality in ("transcriptomics", "metabolomics")}
    monkeypatch.setattr(screening, "fold_patient_latents", fold_payload)
    payload = screening.load_scope_modality_payload(config, "fold", "fixture-fold", "fixture")
    selected_regimens = [{"regimen_uid": "regimen-a"}, {"regimen_uid": "regimen-b"}]
    selected_targets = [{"target_program_uid": "target-program"}]
    labels = {modality: {role: {"target_labels": np.ones((1, 1)), "regimen_labels": np.ones((1, 2))}
                        for role in samples} for modality in payload}
    monkeypatch.setattr(screening, "build_training_targets", lambda *args: (selected_targets, selected_regimens, labels))
    monkeypatch.setattr(screening, "target_program_features", lambda rows: (pd.DataFrame(rows), np.ones((1, 2))))
    monkeypatch.setattr(screening, "regimen_features", lambda rows: (pd.DataFrame(rows), np.ones((2, 25))))
    for name in ("observed_region_mask", "world_state_support_vectors", "world_state_factor_targets", "world_state_supervision_masks"):
        monkeypatch.setattr(screening, name, lambda *args: np.ones((1, 1)))
    visited = []
    assemble = screening.screening_native_payload
    def allowed_teachers(config, role, sample_uids, regimen_uids):
        visited.append(role)
        assert role != "test", "test teacher payload was requested"
        return assemble(config, role, sample_uids, regimen_uids)
    monkeypatch.setattr(screening, "screening_native_payload", allowed_teachers)
    cache = screening.build_scope_training_cache_payload(config, "fold", "fixture-fold", payload, [], [], [], [], [],
        {"world_state": pd.DataFrame(), "counterfactual_benchmark": pd.DataFrame()}, "fixture")
    assert visited == ["train", "validation", "train", "validation"]
    monkeypatch.setattr(screening, "source_artifact_hashes", lambda *args: {})
    monkeypatch.setattr(screening, "freeze_signature_sha256", lambda *args: "fixture-freeze")
    artifact = tmp_path / "cache"
    screening.write_scope_training_cache(config, tmp_path / "config.yaml", ["fixture"], artifact, "fold", "fixture-fold", 54, cache, 0.)
    loaded, _, _ = screening.load_scope_training_cache(config, tmp_path / "config.yaml", artifact, "fold", "fixture-fold", 54)
    for splits in loaded["modality_payload"].values():
        assert splits["test"]["sample_uids"] == ["sample-heldout"]
        assert "native_diffusion" not in splits["test"]
        assert splits["train"]["native_diffusion"]["split_name"] == "train"
        assert splits["validation"]["native_diffusion"]["split_name"] == "validation"
    malformed = copy.deepcopy(loaded["modality_payload"])
    malformed["transcriptomics"]["test"]["native_diffusion"] = malformed["transcriptomics"]["train"]["native_diffusion"]
    with pytest.raises(ValueError, match="role"):
        native.validate_cached_native(malformed, ["regimen-a", "regimen-b"])


def test_direct_native_zarr_rejects_calibration_for_train_before_reads(tmp_path, monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, registry, arrays, store = producer_case(tmp_path)
    for row in registry["teachers"]:
        row.update(split_name="calibration", source_holdout_role="calibration",
                   qc_grade="C", qc_use="weak_teacher_calibration", sample_weight=.1)
    write_native_microenvironment_store(store, config, registry, arrays)
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", lambda *args: pytest.fail("target payload was read"))
    with pytest.raises(ValueError, match="role"):
        pretrain.MicroenvironmentZarrDataset(store, config, split_name="train", trajectory_uids=["trajectory-0"])


def test_direct_native_zarr_requires_identity_bound_positions(tmp_path, monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, _, arrays, store = producer_case(tmp_path)
    dataset = pretrain.MicroenvironmentZarrDataset(store, config, split_name="train",
                                                  trajectory_uids=["trajectory-2", "trajectory-0"])
    assert [item["trajectory_uid"] for item in dataset.__getitems__([0, 1])] == ["trajectory-2", "trajectory-0"]
    assert torch.equal(dataset[0]["trajectory"], torch.from_numpy(arrays["trajectory"][2]))
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", lambda *args: pytest.fail("target payload was read"))
    with pytest.raises(ValueError, match="identity"):
        pretrain.MicroenvironmentZarrDataset(store, config, row_indices=[2, 0], split_name="train")
    with pytest.raises(ValueError, match="identity"):
        pretrain.MicroenvironmentZarrDataset(store, config, row_indices=[0, 2], split_name="train",
                                            trajectory_uids=["trajectory-2", "trajectory-0"])
    with pytest.raises(ValueError, match="role"):
        pretrain.MicroenvironmentZarrDataset(store, config)


@pytest.mark.parametrize("direct_store", [True, False])
@pytest.mark.parametrize("single_batch_check", [True, False])
def test_native_entrypoint_selection_uses_role_registry_before_payloads(tmp_path, monkeypatch, direct_store, single_batch_check):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, registry, arrays, store = producer_case(tmp_path)
    for row in registry["teachers"][2:]:
        row.update(split_name="calibration", source_holdout_role="calibration",
                   qc_grade="C", qc_use="weak_teacher_calibration", sample_weight=.1)
    write_native_microenvironment_store(store, config, registry, arrays)
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    training = config["screening_pipeline"]["screening_benchmark"]["training"]
    training["teacher_family_training"]["enabled"] = False
    training["zarr_loader"]["training_split_name"] = "train"
    monkeypatch.setattr(pretrain, "manifest_row_indices", lambda *args: pytest.fail("legacy positional manifest was read"))
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", lambda *args: pytest.fail("target payload was read"))
    dataset = pretrain.build_pretraining_dataset(config, store, direct_store=direct_store,
                                                single_batch_check=single_batch_check)
    assert [record["teacher"]["trajectory_uid"] for record in dataset.native_records] == ["trajectory-0", "trajectory-1"]
    training["zarr_loader"]["training_split_name"] = "calibration"
    dataset = pretrain.build_pretraining_dataset(config, store, direct_store=direct_store,
                                                single_batch_check=single_batch_check)
    assert [record["teacher"]["trajectory_uid"] for record in dataset.native_records] == ["trajectory-2", "trajectory-3"]


def test_native_store_role_cannot_be_relabelled_by_registry(tmp_path, monkeypatch):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, registry, _, store = producer_case(tmp_path)
    for row in registry["teachers"]:
        row.update(split_name="validation", source_holdout_role="validation")
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", lambda *args: pytest.fail("target payload was read"))
    with pytest.raises(ValueError, match="identity"):
        pretrain.MicroenvironmentZarrDataset(store, config, split_name="validation")


@pytest.mark.parametrize("single_batch_check", [True, False])
def test_non_native_direct_store_keeps_positional_selection(tmp_path, monkeypatch, single_batch_check):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, _, arrays, store = producer_case(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["quantized_mddm_rollout"]["enabled"] = False
    calls = []
    def legacy_manifest(*args):
        calls.append(args)
        return [3, 1]
    monkeypatch.setattr(pretrain, "manifest_row_indices", legacy_manifest)
    dataset = pretrain.build_pretraining_dataset(config, store, direct_store=True,
                                                single_batch_check=single_batch_check)
    expected = [0, 1, 2, 3] if single_batch_check else [3, 1]
    assert dataset.row_indices == expected
    assert len(calls) == (0 if single_batch_check else 1)
    assert torch.equal(dataset[0]["trajectory"], torch.from_numpy(arrays["trajectory"][expected[0]]))


def test_native_direct_callers_reject_heldout_and_mismatched_roles_before_arrays(tmp_path, monkeypatch):
    from Libs.exp.training import train_screening_system as screening
    config, _, _, _ = producer_case(tmp_path)
    payload = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a"])
    model = SimpleNamespace(field_rollout_operator=SimpleNamespace(quantized_mddm_rollout_operator=object(), field_normalization=NativeFieldNormalization(None,15)))
    monkeypatch.setattr(native, "load_registry", lambda *args: pytest.fail("heldout registry was requested"))
    with pytest.raises(ValueError, match="role"):
        native.screening_native_payload(config, "test", ["sample-heldout"], ["regimen-a"])
    # These failures precede conversion/indexing of teacher arrays.
    payload["teacher_trajectory"] = object()
    for role in ("test", "validation", "calibration"):
        with pytest.raises(ValueError, match="role"):
            screening.native_diffusion_training_kwargs(model, {"labels": [1], "native_diffusion": payload},
                                                       [0], [0], torch.device("cpu"), split_name=role)
    with pytest.raises(ValueError, match="role"):
        native.validate_cached_native({"rna": {"validation": {"sample_uids": ["sample-a"], "native_diffusion": payload}}},
                                      ["regimen-a"])


@pytest.mark.parametrize("action_policy", ["drug_id", "target_id"])
def test_serving_identity_reorder_chunk_and_sequential_propagation(tmp_path, monkeypatch, action_policy):
    from Libs.scripts.services.serve_screening_api import native_diffusion_serving_kwargs
    from Libs.tests.test_native_masked_diffusion import _screening_action_case
    from Libs.models.modules.world_action_encoder import native_action_identity_spec
    encoder, _ = native_action_identity_spec(action_policy)
    _, registry, _, _ = producer_case(tmp_path, action_policy=action_policy)
    model, patient, regimens, context = _screening_action_case(action_policy)
    programs = copy.deepcopy(registry["programs"][:2])
    # Program identity is canonical metadata, separate from the RNG regimen key.
    programs[0]["action"]["agent_set"] = ["agent-a"]
    programs[1]["action"]["agent_set"] = ["agent-b"]
    for index, program in enumerate(programs):
        program["action"]["target_set"] = ["target-" + str(index)]
        program["pkpd_context"]["target_program"]["targets"] = program["action"]["target_set"]
        compiled = compile_pharmacological_action(program["action"], pkpd_context=program["pkpd_context"],
            baseline_state_posterior=program["baseline_state_posterior"], graph_context=program["graph_context"])
        program.update(compiled_program=compiled, program_uid=compiled["program_uid"])
    payload = {key: value.detach() for key, value in context.items() if key != "action_identity"}
    payload.update(regimen_uids=["regimen-a", "regimen-b"], programs=programs)
    bundle = {"screening_model": model, "regimen_feature_matrix": regimens,
              "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    selected = native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": payload}, ["regimen-b", "regimen-a"])
    assert torch.equal(selected["diffusion_context"]["action_identity"][0, 0],
                       torch.tensor(encoder(programs[1]["action"])))
    assert torch.equal(selected["diffusion_context"]["posterior_context"], context["posterior_context"][:, [1, 0]])
    assert selected["regimen_ids"].tolist() == [1, 0]
    original = native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": payload}, ["regimen-a", "regimen-b"])
    full = model.score_regimens(patient, regimens, **original)
    reorder = model.score_regimens(patient, regimens[[1, 0]], regimen_chunk_size=1, **selected)
    assert torch.equal(full["field_trajectory_samples"][:, :, [1, 0]], reorder["field_trajectory_samples"])
    steps = [native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": payload}, [uid])
             for uid in ("regimen-b", "regimen-a")]
    seen = []
    forward = model.field_rollout_operator.forward
    def capture(*args, **kwargs):
        seen.append((kwargs["diffusion_context"]["action_identity"].clone(), kwargs["regimen_ids"].clone()))
        return forward(*args, **kwargs)
    monkeypatch.setattr(model.field_rollout_operator, "forward", capture)
    result = model.rollout(patient, regimens[[1, 0]].unsqueeze(0),
                           diffusion_context_sequence=[step["diffusion_context"] for step in steps],
                           regimen_ids_sequence=[step["regimen_ids"] for step in steps])
    assert torch.isfinite(result["field_trajectory_sequence"]).all()
    assert [row[1].item() for row in seen] == [1, 0]
    for row, step in zip(seen, steps):
        assert torch.equal(row[0], step["diffusion_context"]["action_identity"])
    missing = dict(payload)
    missing.pop("programs")
    with pytest.raises(ValueError, match="canonical programs"):
        native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": missing}, ["regimen-a"])
    wrong = copy.deepcopy(payload)
    wrong["programs"].reverse()
    with pytest.raises(ValueError, match="regimen binding"):
        native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": wrong}, ["regimen-a"])
    malformed = copy.deepcopy(payload)
    malformed["programs"][0]["action"]["target_set" if action_policy == "target_id" else "agent_set"] = []
    malformed["occupancy"] = object()  # Reject before tensor conversion.
    with pytest.raises(ValueError, match="target_set|agent_set|compiler"):
        native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": malformed}, ["regimen-a"])
    partial = dict(payload)
    partial.pop("posterior_context")
    with pytest.raises(ValueError, match="explicit lineage"):
        native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": partial}, ["regimen-a"])


@pytest.mark.parametrize("action_policy", ["drug_id", "target_id"])
def test_native_identity_cache_manifest_rejects_stale_schema_before_payload(tmp_path, monkeypatch, action_policy):
    from Libs.exp.training import train_screening_system as screening
    from Libs.models.modules.world_action_encoder import native_action_identity_spec
    _, identity_schema = native_action_identity_spec(action_policy)
    config, _, _, _ = producer_case(tmp_path, action_policy=action_policy)
    monkeypatch.setattr(screening, "source_artifact_hashes", lambda *args: {})
    monkeypatch.setattr(screening, "freeze_signature_sha256", lambda *args: "fixture")
    source = Path(config["native_diffusion_data"]["registry_path"])
    manifest = {"schema_version": screening.cache_schema_version(config), "config_hash": screening.json_sha256(config),
                "freeze_signature_sha256": "fixture", "scope": "fixture", "fold": "train", "seed": 55,
                "source_artifact_hashes": {}, "native_source_bindings": {str(source): screening.file_sha256(source)},
                "native_action_identity_schema": identity_schema, "native_action_policy": action_policy}
    for name, bad in (("native_action_identity_schema", "stale"), ("native_action_policy", "occupancy_program")):
        broken = dict(manifest, **{name: bad})
        assert name in screening.cache_manifest_mismatches(config, tmp_path/"config.yaml", broken, "fixture", "train", 55)
    record = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a"])
    fake_model = SimpleNamespace(field_rollout_operator=SimpleNamespace(quantized_mddm_rollout_operator=object(), field_normalization=NativeFieldNormalization(None,15), action_policy_name=action_policy))
    record["action_identity_schema"] = "stale"
    record["teacher_trajectory"] = object()
    with pytest.raises(ValueError, match="identity cache"):
        screening.native_diffusion_training_kwargs(fake_model, {"labels": [1], "native_diffusion": record},
                                                   [0], [0], torch.device("cpu"))


def test_serving_rejects_failed_or_inconsistent_compiled_programs_before_inference(tmp_path):
    from Libs.scripts.services.serve_screening_api import native_diffusion_serving_kwargs
    from Libs.tests.test_native_masked_diffusion import _screening_action_case
    _, registry, _, _ = producer_case(tmp_path, action_policy="drug_id")
    model, patient, regimens, context = _screening_action_case()
    payload = {key: value.detach() for key, value in context.items() if key != "action_identity"}
    payload.update(regimen_uids=["regimen-a", "regimen-b"], programs=copy.deepcopy(registry["programs"][:2]))
    bundle = {"screening_model": model, "regimen_feature_matrix": regimens,
              "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    mutations = [
        lambda p: p["compiled_program"].update(status="insufficient_parameters",
            blockers=[{"code": "missing_pkpd_parameterization"}], occupancy_fields={}, exposure_fields={}),
        lambda p: p["compiled_program"].update(blockers=[{"code": "missing_pkpd_parameterization"}]),
        lambda p: p["compiled_program"].update(exposure_fields={}),
        lambda p: p["pkpd_context"].update(exposure_fields={}),
        lambda p: p["pkpd_context"].update(parameter_provenance={}),
        lambda p: p.update(compiled_program=None),
        lambda p: p.update(graph_context={}),
        lambda p: p["baseline_state_posterior"].update(state_ref="other-observation"),
        lambda p: p.update(representation="global_scalar"),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(payload)
        mutate(invalid["programs"][0])
        with pytest.raises(ValueError, match="program"):
            kwargs = native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": invalid}, ["regimen-a"])
            # This exact path previously returned finite inference for failed compilation.
            model.score_regimens(patient, regimens[:1], **kwargs)
    invalid = copy.deepcopy(payload)
    program = invalid["programs"][0]
    program["pkpd_context"]["occupancy_fields"]["array_name"] = "unbound-occupancy"
    program["compiled_program"] = compile_pharmacological_action(
        program["action"], pkpd_context=program["pkpd_context"],
        baseline_state_posterior=program["baseline_state_posterior"], graph_context=program["graph_context"])
    with pytest.raises(ValueError, match="array binding"):
        native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": invalid}, ["regimen-a"])
    invalid_time = dict(payload, physical_time=payload["physical_time"] + .5)
    with pytest.raises(ValueError, match="compiled program time grid"):
        native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": invalid_time}, ["regimen-a"])


@pytest.mark.parametrize("action_policy", ["drug_id", "occupancy_program"])
@pytest.mark.parametrize("explicit_observation", [True, False])
def test_native_serving_sequence_propagates_previous_final_and_first_action(tmp_path, action_policy, explicit_observation):
    from Libs.scripts.services.serve_screening_api import native_diffusion_serving_kwargs
    from Libs.tests.test_native_masked_diffusion import _screening_action_case
    _, registry, _, _ = producer_case(tmp_path, action_policy=action_policy)
    model, patient, regimens, context = _screening_action_case(action_policy)
    payload = {key: value.detach() for key, value in context.items() if key != "action_identity"}
    if not explicit_observation:
        payload = {key: value for key, value in payload.items() if key not in {"initial_state", "posterior_context"}}
    payload.update(regimen_uids=["regimen-a", "regimen-b"], programs=copy.deepcopy(registry["programs"][:2]))
    bundle = {"screening_model": model, "regimen_feature_matrix": regimens,
              "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    sampler = model.field_rollout_operator.quantized_mddm_rollout_operator

    def evaluate(current):
        steps = [native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": current}, [uid])
                 for uid in ("regimen-a", "regimen-b")]
        initial_states, posteriors = [], []
        def capture(module, args, kwargs):
            initial_states.append(kwargs["initial_state"].detach().clone())
            posteriors.append(kwargs["conditioning"]["posterior_context"].detach().clone())
        hook = sampler.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                result = model.rollout(patient, regimens.unsqueeze(0),
                    diffusion_context_sequence=[step["diffusion_context"] for step in steps],
                    regimen_ids_sequence=[step["regimen_ids"] for step in steps])
        finally:
            hook.remove()
        assert torch.equal(initial_states[1], result["field_trajectory_sequence"][:, 0, -1].unsqueeze(1))
        assert torch.equal(result["initial_state"], initial_states[0][:, 0])
        if explicit_observation:
            assert torch.equal(posteriors[1], steps[1]["diffusion_context"]["posterior_context"])
            assert torch.equal(steps[1]["diffusion_context"]["initial_state"], current["initial_state"][:, 1:2])
        return result, initial_states

    baseline, original_initials = evaluate(payload)
    changed = copy.deepcopy(payload)
    if action_policy in {"drug_id", "target_id"}:
        # Keep program/RNG identities fixed; only allowed canonical membership changes.
        changed["programs"][0]["action"]["agent_set"] = ["different-first-agent"]
    else:
        changed["occupancy"][:, 0] = 1 - changed["occupancy"][:, 0]
    perturbed, changed_initials = evaluate(changed)
    assert torch.equal(original_initials[0], changed_initials[0])
    assert not torch.equal(baseline["field_trajectory_sequence"][:, 0, -1],
                           perturbed["field_trajectory_sequence"][:, 0, -1])
    assert not torch.equal(original_initials[1], changed_initials[1])
    assert not torch.equal(baseline["field_trajectory_sequence"][:, 1],
                           perturbed["field_trajectory_sequence"][:, 1])


def test_native_sequence_rejects_mixed_observation_anchors(tmp_path):
    from Libs.scripts.services.serve_screening_api import native_diffusion_serving_kwargs
    from Libs.tests.test_native_masked_diffusion import _screening_action_case
    _, registry, _, _ = producer_case(tmp_path, action_policy="drug_id")
    model, patient, regimens, context = _screening_action_case()
    payload = {key: value.detach() for key, value in context.items() if key != "action_identity"}
    payload.update(regimen_uids=["regimen-a", "regimen-b"], programs=copy.deepcopy(registry["programs"][:2]))
    bundle = {"screening_model": model, "regimen_feature_matrix": regimens,
              "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    steps = [native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": payload}, [uid])
             for uid in ("regimen-a", "regimen-b")]
    for field in ("initial_state", "posterior_context", "implicit"):
        contexts = [dict(step["diffusion_context"]) for step in steps]
        if field == "implicit":
            contexts[1].pop("initial_state")
            contexts[1].pop("posterior_context")
        else:
            contexts[1][field] = contexts[1][field] + .1
        with pytest.raises(ValueError, match="one observation-derived initial/posterior lineage"):
            model.rollout(patient, regimens.unsqueeze(0), diffusion_context_sequence=contexts,
                          regimen_ids_sequence=[step["regimen_ids"] for step in steps])



@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
@pytest.mark.parametrize("targets", [None, [], [""]])
def test_target_registry_rejects_missing_targets_before_store_reads(tmp_path, monkeypatch, store_format, targets):
    config, registry, _, _ = producer_case(tmp_path, store_format, "target_id")
    registry["programs"][0]["action"]["target_set"] = targets
    Path(config["native_diffusion_data"]["registry_path"]).write_text(json.dumps(registry))
    monkeypatch.setattr(native.dynamics_store, "load_zarr_rows", lambda *args: pytest.fail("Invalid target reached store read"))
    monkeypatch.setattr(native.dynamics_store, "load_chunked_rows", lambda *args: pytest.fail("Invalid target reached store read"))
    with pytest.raises(ValueError, match="target_set|compiler"):
        native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a"])



def test_target_serving_canonical_targets_control_predictor(tmp_path, monkeypatch):
    from Libs.scripts.services.serve_screening_api import native_diffusion_serving_kwargs
    from Libs.tests.test_native_masked_diffusion import _screening_action_case
    _, registry, _, _ = producer_case(tmp_path, action_policy="target_id")
    model, patient, regimens, context = _screening_action_case("target_id")
    payload = {key: value.detach().clone() for key, value in context.items() if key != "action_identity"}
    payload.update(regimen_uids=["regimen-a", "regimen-b"], programs=copy.deepcopy(registry["programs"][:2]))
    bundle = {"screening_model": model, "regimen_feature_matrix": regimens,
              "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    teacher = torch.rand(1,2,3,3,2,7)
    sampler = model.field_rollout_operator.quantized_mddm_rollout_operator
    logits, original_logits = [], sampler.logits
    def capture(*args, **kwargs):
        result = original_logits(*args, **kwargs)
        logits.append(result[0].detach().clone())
        return result
    monkeypatch.setattr(sampler, "logits", capture)
    def predict(current, features):
        kwargs = native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": current}, payload["regimen_uids"])
        logits.clear()
        result = model.score_regimens(patient, features, teacher_trajectory=teacher,
                                      teacher_confidence=torch.ones(1,2), **kwargs)
        return result, torch.cat([value.flatten() for value in logits])
    def recompile(program):
        compiled = compile_pharmacological_action(program["action"], pkpd_context=program["pkpd_context"],
            baseline_state_posterior=program["baseline_state_posterior"], graph_context=program["graph_context"])
        program.update(compiled_program=compiled, program_uid=compiled["program_uid"])
    baseline, baseline_logits = predict(payload, regimens)
    nuisance = copy.deepcopy(payload)
    for program in nuisance["programs"]:
        program["action"].update(agent_set=["other-agent"], dose=3., route="oral",
                                 schedule={"kind": "interval", "interval": 2, "unit": "day"})
        recompile(program)
    nuisance["occupancy"] = 1 - nuisance["occupancy"]
    nuisance["support"] = torch.zeros_like(nuisance["support"])
    equivalent, equivalent_logits = predict(nuisance, regimens + 10)
    assert torch.equal(baseline_logits, equivalent_logits)
    for key in ("field_training_reconstruction", "field_trajectory_samples", "rollout_auxiliary_loss"):
        assert torch.equal(baseline[key], equivalent[key])
    target = copy.deepcopy(payload)
    target["programs"][0]["action"]["target_set"] = ["different-target"]
    target["programs"][0]["pkpd_context"]["target_program"]["targets"] = ["different-target"]
    recompile(target["programs"][0])
    changed, changed_logits = predict(target, regimens)
    assert not torch.allclose(baseline_logits, changed_logits)
    assert not torch.allclose(baseline["field_training_reconstruction"], changed["field_training_reconstruction"])


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_admission_reads_each_descriptor_once_and_refreshes_between_calls(tmp_path, monkeypatch, store_format):
    from collections import Counter
    config, registry, _, store = producer_case(tmp_path, store_format)
    records = native.join_registry(registry, config)
    descriptor = store / (".zattrs" if store_format == "zarr_v2" else "store_manifest.json")
    reads = Counter()
    original = Path.read_text

    def counted(path, *args, **kwargs):
        reads[path] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    native.descriptor_bindings(records, registry, config)
    assert reads[descriptor] == 1
    assert all(count == 1 for path, count in reads.items() if path.name == ".zarray")
    content = json.loads(original(descriptor))
    stored = content.get("metadata", content)["native_diffusion"]
    stored["teachers"][-1]["trajectory_uid"] = "changed-after-first-admission"
    descriptor.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="identity"):
        native.descriptor_bindings(records, registry, config)
    assert reads[descriptor] == 2


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_batched_native_reads_preserve_reordered_source_rows(tmp_path, monkeypatch, store_format):
    from collections import Counter
    config, registry, arrays, _ = producer_case(tmp_path, store_format)
    records = native.join_registry(registry, config)
    order = [3, 0, 2]
    name = "load_zarr_rows" if store_format == "zarr_v2" else "load_chunked_rows"
    original = getattr(native.dynamics_store, name)
    calls = Counter()

    def counted(path, array_name, indices):
        calls[(path, array_name)] += 1
        return original(path, array_name, indices)

    monkeypatch.setattr(native.dynamics_store, name, counted)
    loaded, _ = native.load_records(registry, [records[index] for index in order], config)
    assert all(count == 1 for count in calls.values())
    np.testing.assert_array_equal(loaded["trajectory"], arrays["trajectory"][order])
    np.testing.assert_array_equal(loaded["initial_state"], arrays["initial_state"][[1, 0, 1]])
