"""Source-site AUC comparison: physical arithmetic and actual native consumers."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Libs.tests.test_physical_occupancy import physical_case, input_copy, reseal, make_packet
from Libs.tests.test_native_field_normalization import joint_normalized_case, screening_kwargs, screening_fixture_bundle
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.physical_occupancy import compile_physical_record, source_site_exposure, SCALAR_EXPOSURE_DEFINITION
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import deep_merge, load_experiment_family
from Libs.scripts.teacher_generation.build_osp_pbpk_teacher_rollouts import build_native_physical_source


def scalar_config(base):
    config = copy.deepcopy(base)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] = "scalar_exposure"
    return config


@pytest.mark.parametrize("unit,multiplier", [("nmol/L", 1.), ("umol/L", .001), ("mol/L", 1e-9)])
def test_arithmetic_nonuniform_units_sites_and_zero(physical_case, unit, multiplier):
    inputs = input_copy(physical_case)
    inputs["time_grid_hours"] = [0., .5, 2.]
    snapshot = inputs["run_snapshot"]
    paths = list(snapshot["quantities"])
    for quantity in snapshot["quantities"].values():
        quantity["concentration_unit"] = unit
    for row in inputs["source_rows"]:
        position = [0., 60., 120.].index(row["source_time"])
        site = paths.index(row["source_output_path"])
        row.update(source_time=[0., 30., 120.][position], concentration_unit=unit,
                   concentration=[2., 6., 10.][position] * site * multiplier)
    record = compile_physical_record(reseal(inputs))
    view = source_site_exposure(record)
    # 0.5*(2+6)/2 + 1.5*(6+10)/2 = 14 nmol*h/L.
    np.testing.assert_allclose(np.asarray(view["values"])[..., 0],
                               np.arange(len(paths)).reshape(3, 2) * 14., rtol=1e-14)
    assert view["interval_hours"] == [0., 2.]
    assert view["values"][0][0] == [0.]
    assert len({site for row in view["site_uids"] for site in row}) == 6
    bad = copy.deepcopy(record)
    bad["physical_program"]["free_concentration_mol_per_l"][0][0][0][0] += 1
    with pytest.raises(ValueError, match="retained physical"):
        source_site_exposure(bad)


def paired_source(directory, config, source):
    """Actual source exports with equal integrals and distinct action/timing."""
    packet = make_packet(directory, config, source)
    packet["time_grid_hours"] = [0., .5, 1.5, 2.]
    for entry in packet["exports"]:
        csv = directory / entry["csv_path"]
        old = pd.read_csv(csv)
        ai = packet["actions"].index(next(a for a in packet["actions"] if a["regimen_uid"] == entry["regimen_uid"]))
        ji = source["support"]["particle_uids"][0].index(entry["particle_uid"]) if entry["sample_uid"] == source["support"]["sample_uids"][0] else source["support"]["particle_uids"][1].index(entry["particle_uid"])
        rows = []
        for site, path in enumerate(old["paths"].unique()):
            values = [0., 2., 6., 0.] if ai == 0 else [0., 6., 2., 0.]
            for time, value in zip([0., 30., 90., 120.], values):
                rows.append({"paths": path, "Time": time, "TimeUnit": "min",
                             "simulationValues": value * (site + 1) * 2**ji, "Unit": "nmol/L"})
        pd.DataFrame(rows).to_csv(csv, index=False)
        snapshot = directory / entry["run_snapshot_path"]
        payload = json.loads(snapshot.read_text())
        payload["export_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
        # Same site affinity for both action identities, still source-bound.
        for kd in payload["affinities"].values():
            kd["value"] = 10.
        for name, parameter in entry["binding"]["particle_parameters"].items():
            if name.startswith("affinity:"):
                parameter["value"] = copy.deepcopy(payload["affinities"][name.removeprefix("affinity:")])
        snapshot.write_text(json.dumps(payload))
    return build_native_physical_source(config, packet, directory)


def test_equal_auc_actual_condition_boundary(joint_normalized_case, tmp_path):
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    _, base, _, raw, _, _ = joint_normalized_case
    config = scalar_config(base)
    source = paired_source(tmp_path / "paired-exports", config, raw)
    model = MicroenvironmentPretrainingModel(config, False).eval()
    posterior = model.infer_joint_source(source)
    programs = native.bind_joint_programs(posterior, source, action_policy="scalar_exposure")
    contexts = native.joint_particle_contexts(posterior, programs, programs["regimen_uids"], action_policy="scalar_exposure")
    records = source["particle_programs"]["records"]
    for sample in records:
        for particle in sample:
            for key in ("free_concentration_mol_per_l", "occupancy"):
                curves = [np.asarray(record["physical_program"][key]) for record in particle]
                integrals = [np.sum((curve[:-1]+curve[1:])*.5*np.diff([0., .5, 1.5, 2.])[:, None, None, None], axis=0) for curve in curves]
                np.testing.assert_allclose(*integrals, rtol=1e-14)
                assert not np.array_equal(*curves)
    auc = contexts[0]["scalar_exposure_auc"]
    assert torch.equal(auc[:, 0], auc[:, 1])
    assert torch.equal(contexts[1]["scalar_exposure_auc"], 2 * auc)
    # Reweighting complete particles never changes conditional exposure.
    posterior = replace(posterior, log_weights=posterior.log_weights.flip(1), posterior_uid="reweighted-fixture")
    programs = dict(programs, posterior_uid=posterior.posterior_uid)
    repeated = native.joint_particle_contexts(posterior, programs, programs["regimen_uids"], action_policy="scalar_exposure")
    assert all(torch.equal(a["scalar_exposure_auc"], b["scalar_exposure_auc"]) for a, b in zip(contexts, repeated))
    sampler = model.quantized_mddm
    captured = []
    handle = model.operator.register_forward_pre_hook(lambda module, args: captured.append(tuple(x.detach().clone() for x in args[:2])))
    logits = []
    for ai in range(2):
        context = {key: value[0, ai] for key, value in contexts[0].items()}
        coefficient = torch.zeros(4, 6, 6, 15)
        logits.append(sampler.logits(coefficient, torch.ones_like(coefficient, dtype=torch.bool), torch.tensor(.5),
                                     context, model.operator, model.node_adjacency,
                                     torch.zeros(1, 1, 36, 2))[0])
    handle.remove()
    assert all(torch.equal(a, b) for a, b in zip(captured[0], captured[1]))
    assert torch.equal(*logits)
    sampler.action_policy_name = "occupancy_program"
    captured = []
    handle = model.operator.register_forward_pre_hook(lambda module, args: captured.append(args[1].detach().clone()))
    for ai in range(2):
        context = {key: value[0, ai] for key, value in contexts[0].items()}
        sampler.logits(coefficient, torch.ones_like(coefficient, dtype=torch.bool), torch.tensor(.5),
                       context, model.operator, model.node_adjacency, torch.zeros(1, 1, 36, 2))
    handle.remove()
    assert not torch.equal(*captured)  # No modification of native AdaLN-Zero gates.


@pytest.mark.parametrize("mutation", ["value", "unit", "interval", "source"])
def test_cache_and_serving_reject_forged_view(joint_normalized_case, mutation):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.scripts.services import serve_screening_api as serving
    _, base, _, source, _, _ = joint_normalized_case
    config = scalar_config(base)
    cached = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a", "regimen-b"])
    view = cached["scalar_exposure"]
    if mutation == "value":
        view["records"][0][0][0]["values"][0][0][0] += 1
    elif mutation == "unit":
        view["definition"]["unit"] = "mol*h/L"
    elif mutation == "interval":
        view["records"][0][0][0]["interval_hours"][-1] += 1
    else:
        view["records"][0][0][0]["physical_input_sha256"] = "different-source"
    with pytest.raises(ValueError, match="scalar exposure"):
        native.validate_cached_native({"fixture": {"train": {"sample_uids": ["sample-a"], "native_diffusion": cached}}},
                                      cached["regimen_uids"], config)
    model = ADScreeningSystem(**screening_kwargs(config)).eval()
    selected = native.select_joint_source(source, ["sample-a"])
    bundle = {"screening_model": model, "regimen_index_by_uid": {"regimen-a": 0, "regimen-b": 1}}
    with pytest.raises(ValueError, match="scalar exposure"):
        serving.native_diffusion_serving_kwargs(bundle, {"joint_source": selected, "scalar_exposure": view}, cached["regimen_uids"])


def test_actual_scalar_pretrainer_and_profile_parity(joint_normalized_case):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    _, base, _, source, original, _ = joint_normalized_case
    config = scalar_config(base)
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    reference = pretrain.MicroenvironmentPretrainingModel(base, False)
    assert {n:p.shape for n,p in model.named_parameters()} == {n:p.shape for n,p in reference.named_parameters()}
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    metrics, report = pretrain.run_pretraining_step(model, pretrain.collate_native_sources(dataset.__getitems__([3, 0])),
        config, torch.device("cpu"), {"gradient_accumulation_steps": 1,
        "mixed_precision": {"enabled": False, "device_type": "cpu", "dtype": "float32"}})
    assert np.isfinite(metrics["total_loss"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.operator.parameters())
    for row in report["source_rows"]:
        assert row["field_normalization_diagnostics"] and row["operator_splitting_diagnostics"]
        assert len(row["action_conditioning"]) == 2
        assert all(item["definition"] == SCALAR_EXPOSURE_DEFINITION for item in row["action_conditioning"])
    assert json.dumps(source, sort_keys=True) == original


def test_actual_scalar_screening_serving_sequence(joint_normalized_case, tmp_path):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.scripts.services import serve_screening_api as serving
    _, base, _, source, original, _ = joint_normalized_case
    config = scalar_config(base)
    kwargs = screening_kwargs(config)
    model = ADScreeningSystem(**kwargs).eval()
    regimens = torch.rand(2, kwargs["regimen_feature_dim"])
    cached = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a", "regimen-b"])
    training = native_diffusion_training_kwargs(model, {"labels": [0], "native_diffusion": cached},
        slice(None), [0, 1], torch.device("cpu"), split_name="train")
    output = model.score_regimens(torch.zeros(1, 4), regimens, **training)
    changed = model.score_regimens(torch.zeros(1, 4), regimens * 100 + 40, **training)
    assert torch.equal(output["encoded_regimens"], changed["encoded_regimens"])
    assert torch.equal(output["particle_field_trajectory"], changed["particle_field_trajectory"])
    torch.testing.assert_close(output["field_trajectory_std"].square(),
                               output["within_diffusion_variance"] + output["between_particle_variance"])
    assert len(output["action_conditioning"]) == 2
    checkpoint = tmp_path / "scalar.pt"
    torch.save({"model_kwargs": kwargs, "model_state": model.state_dict()}, checkpoint)
    restored = serving.load_screening_model(checkpoint, config)
    with pytest.raises(ValueError, match="scalar exposure policy"):
        serving.load_screening_model(checkpoint, base)
    bundle = screening_fixture_bundle(restored, config, regimens)
    payload = {"joint_source": native.select_joint_source(source, ["sample-a"]),
               "regimen_uids": ["regimen-a", "regimen-b"]}
    prepared = serving.native_diffusion_serving_kwargs(bundle, payload, payload["regimen_uids"])
    sequence = restored.rollout(torch.zeros(1, 4), regimens.unsqueeze(0),
        joint_posterior=prepared["joint_posterior"], particle_programs=prepared["particle_programs"],
        regimen_uids=payload["regimen_uids"])
    assert torch.equal(sequence["particle_field_trajectory_sequence"][:, :, 1, 0],
                       sequence["particle_field_trajectory_sequence"][:, :, 0, -1])
    result = serving.rollout_regimens(bundle, payload)
    json.dumps(result, allow_nan=False)
    assert len(result["action_conditioning"]) == 4
    assert len(result["operator_splitting_diagnostics"]) == 4
    assert json.dumps(source, sort_keys=True) == original


def test_canonical_scalar_profile():
    base = load_config("Libs/configs/config_stack.yaml")
    reference = load_experiment_family("dynamics_mainline_ad", base)
    profile = load_experiment_family("action_conditioning_scalar_exposure", base)
    actual, expected = deep_merge(base, profile["config_overrides"]), deep_merge(base, reference["config_overrides"])
    actual["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] = "occupancy_program"
    assert actual == expected
    assert profile["execution"] == reference["execution"]
    assert profile["runtime"]["seed_values"] == reference["runtime"]["seed_values"] == [3407, 42, 26]
    for key in ("cluster_training", "cluster_prebuild"):
        assert profile[key] == reference[key]
    runtime = lambda family: {k:v for k,v in family["runtime"].items() if k not in {"benchmark_name_template", "artifact_root"}}
    assert runtime(profile) == runtime(reference)
    entry = next(row for row in base["paper_grade_baseline_registry"]["baselines"] if row["baseline_id"] == "action_scalar_pk_exposure_reference")
    assert all(entry[key] is False for key in ("main_table_runnable", "appendix_runnable"))
    assert entry["reference_only"] is True
    assert native.native_enabled(deep_merge(base, profile["config_overrides"]))


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_actual_stores_cache_and_fixture_validator(joint_normalized_case, tmp_path, monkeypatch, store_format):
    import subprocess
    import sys
    import yaml
    from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
    from Libs.exp.training import train_screening_system as screening
    _, base, _, source, _, arrays = joint_normalized_case
    config = scalar_config(base)
    registry = copy.deepcopy(native.load_registry(config))
    store = tmp_path / "scalar-source"
    for teacher in registry["teachers"]:
        teacher.update(trajectory_store_path=str(store), store_format=store_format)
    # Optional retained views must be derived from the same raw programs.
    programs = registry["joint_sources"][0]["particle_programs"]
    programs["scalar_exposure"] = native.scalar_exposure_view(programs["records"])
    forged = copy.deepcopy(registry)
    forged["joint_sources"][0]["particle_programs"]["scalar_exposure"]["definition"]["unit"] = "wrong"
    with pytest.raises(ValueError, match="scalar exposure"):
        write_native_microenvironment_store(store, config, forged, arrays, store_format)
    numeric = native.scalar_exposure_arrays(programs["scalar_exposure"])
    numeric["scalar_exposure_auc"][0, 0, 0, 0, 0, 0] += 10
    with pytest.raises(ValueError, match="scalar exposure"):
        write_native_microenvironment_store(store, config, registry, dict(arrays, **numeric), store_format)
    write_native_microenvironment_store(store, config, registry, arrays, store_format)
    config["native_diffusion_data"]["registry_path"] = str(store / (".zattrs" if store_format == "zarr_v2" else "store_manifest.json"))
    cached = native.screening_native_payload(config, "train", ["sample-b", "sample-a"], ["regimen-b", "regimen-a"])
    expected = native.scalar_exposure_view(cached["joint_source"]["particle_programs"]["records"])
    assert cached["scalar_exposure"] == expected
    payload = {"selected_target_rows": [{"target_program_uid": "target"}],
        "selected_regimen_rows": [{"regimen_uid": uid} for uid in cached["regimen_uids"]],
        "modality_payload": {"fixture": {"train": {"sample_uids": cached["sample_uids"],
            "state_features": np.zeros((2, 4), dtype=np.float32), "native_diffusion": cached}}},
        "label_payload": {"fixture": {"train": {"target_labels": np.ones((2, 1), dtype=np.float32),
            "regimen_labels": np.ones((2, 2), dtype=np.float32)}}}}
    monkeypatch.setattr(screening, "source_artifact_hashes", lambda *args: {})
    monkeypatch.setattr(screening, "freeze_signature_sha256", lambda *args: "controlled-fixture")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    screening.write_scope_training_cache(config, config_path, ["fixture"], tmp_path / "cache", "fixture", "train", 11, payload, 0.)
    restored, _, _ = screening.load_scope_training_cache(config, config_path, tmp_path / "cache", "fixture", "train", 11)
    assert restored["modality_payload"]["fixture"]["train"]["native_diffusion"]["scalar_exposure"] == expected
    completed = subprocess.run([sys.executable, "Libs/scripts/validators/validate_full_microenvironment_teacher_registry.py",
                                "--config", str(config_path)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "4 records; no array payload read" in completed.stdout
    # Same AUC, different source bytes must still stale the existing cache.
    descriptor = Path(config["native_diffusion_data"]["registry_path"])
    descriptor.write_text(descriptor.read_text() + "\\n")
    with pytest.raises(ValueError, match="cache_stale"):
        native.validate_cached_native(payload["modality_payload"], cached["regimen_uids"], config)


def test_source_timing_change_with_equal_auc_stales_view(joint_normalized_case, tmp_path):
    _, base, _, raw, _, _ = joint_normalized_case
    source = paired_source(tmp_path / "timing-source", base, raw)
    programs = source["particle_programs"]
    before = native.scalar_exposure_view(programs["records"])
    # Exchange the two interior source concentrations, retaining equal quadrature weights.
    for sample in programs["records"]:
        for particle in sample:
            for ai, record in enumerate(particle):
                inputs = copy.deepcopy(record["physical_program"]["inputs"])
                for row in inputs["source_rows"]:
                    if row["source_time"] == 30.:
                        row["concentration"] *= 3 if ai == 0 else 1/3
                    elif row["source_time"] == 90.:
                        row["concentration"] *= 1/3 if ai == 0 else 3
                particle[ai] = compile_physical_record(inputs)
    after = native.scalar_exposure_view(programs["records"])
    np.testing.assert_allclose(native.scalar_exposure_arrays(before)["scalar_exposure_auc"],
                               native.scalar_exposure_arrays(after)["scalar_exposure_auc"])
    with pytest.raises(ValueError, match="source/value/definition"):
        native.verify_scalar_exposure(before, after)


def test_scalar_missing_evidence_and_definition_reject(tmp_path):
    from Libs.tests.test_native_diffusion_data import producer_case
    base, registry, arrays, _ = producer_case(tmp_path)
    config = scalar_config(base)
    with pytest.raises(ValueError, match="genuine physical evidence"):
        native.join_registry(registry, config)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["policies"]["scalar_exposure"]["native_definition"]["unit"] = "fraction"
    with pytest.raises(ValueError, match="definition mismatch"):
        native.native_enabled(config)


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_point_source_store_pretrainer_and_serving(joint_normalized_case, tmp_path, store_format):
    from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.scripts.services import serve_screening_api as serving
    _, base, _, source, _, original_arrays = joint_normalized_case
    config = scalar_config(base)
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    registry = copy.deepcopy(native.load_registry(config))
    registry.pop("joint_sources")
    registry["posterior_family"] = "deterministic_compatibility"
    registry["teachers"] = registry["teachers"][:2]
    store = tmp_path / "point-source"
    registry["programs"] = copy.deepcopy(source["particle_programs"]["records"][0][0])
    first = registry["programs"][0]
    registry["observations"] = [{"posterior_uid": first["posterior_uid"], "sample_uid": first["sample_uid"],
        "source_uid": first["source_uid"], "origin": "observation_derived", "observation_ref": "fixture-observation",
        "source_snapshot_ref": source["support"]["source_snapshot_ref"], "initial_time_hours": 0,
        "store_path": str(store), "store_format": store_format, "observation_index": 0}]
    for index, (teacher, program) in enumerate(zip(registry["teachers"], registry["programs"])):
        teacher.update(trajectory_store_path=str(store), store_format=store_format, trajectory_index=index)
        program.update(store_path=str(store), store_format=store_format, program_index=index)
    arrays = {key: value[:2] for key, value in original_arrays.items()}
    arrays.update(initial_state=np.asarray(source["support"]["initial_field"])[0:1, 0].astype(np.float32),
                  posterior_context=np.zeros((1, 6, 6, 12), dtype=np.float32))
    for key in ("occupancy", "support", "physical_time"):
        arrays[key] = np.asarray([row["physical_program"][key] for row in registry["programs"]], dtype=np.float32)
    write_native_microenvironment_store(store, config, registry, arrays, store_format)
    config["native_diffusion_data"]["registry_path"] = str(store / (".zattrs" if store_format == "zarr_v2" else "store_manifest.json"))
    cached = native.screening_native_payload(config, "train", [first["sample_uid"]], ["regimen-a", "regimen-b"])
    native.validate_cached_native({"fixture": {"train": {"sample_uids": cached["sample_uids"], "native_diffusion": cached}}},
                                  cached["regimen_uids"], config)
    bad = copy.deepcopy(cached)
    bad["scalar_exposure_auc"][0, 0, 0, 0, 0] += 100
    with pytest.raises(ValueError, match="cached numeric"):
        native.validate_cached_native({"fixture": {"train": {"sample_uids": cached["sample_uids"], "native_diffusion": bad}}},
                                      cached["regimen_uids"], config)
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    metrics, _ = pretrain.run_pretraining_step(model, pretrain.collate_native_sources(dataset.__getitems__([0])),
        config, torch.device("cpu"), {"gradient_accumulation_steps": 1,
        "mixed_precision": {"enabled": False, "device_type": "cpu", "dtype": "float32"}})
    assert np.isfinite(metrics["total_loss"])
    kwargs = screening_kwargs(config)
    screening = ADScreeningSystem(**kwargs).eval()
    regimens = torch.zeros(2, kwargs["regimen_feature_dim"])
    bundle = screening_fixture_bundle(screening, config, regimens)
    context = {key: cached[key] for key in ("occupancy", "support", "physical_time", "initial_state", "posterior_context",
                                          "sample_uids", "regimen_uids")}
    context["programs"] = [registry["programs"]]
    prepared = serving.native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": context}, cached["regimen_uids"])
    trained = native_diffusion_training_kwargs(screening, {"labels": [0], "native_diffusion": cached}, slice(None), [0, 1],
                                               torch.device("cpu"), split_name="train")
    assert torch.equal(prepared["diffusion_context"]["scalar_exposure_auc"], trained["diffusion_context"]["scalar_exposure_auc"])
    output = screening.score_regimens(torch.zeros(1, 4), regimens, **prepared)
    assert torch.isfinite(output["field_trajectory"]).all()
    context["scalar_exposure_auc"] = cached["scalar_exposure_auc"] + 100
    with pytest.raises(ValueError, match="supplied numeric"):
        serving.native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": context}, cached["regimen_uids"])


def test_legacy_non_native_scalar_feature_mask(joint_normalized_case):
    from Libs.models.networks.screening_system import ADScreeningSystem
    _, base, _, _, _, _ = joint_normalized_case
    config = scalar_config(base)
    rollout = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    rollout["transition_backbone"]["quantized_mddm_rollout"]["enabled"] = False
    model = ADScreeningSystem(**screening_kwargs(config))
    field = model.field_rollout_operator
    assert field.quantized_mddm_rollout_operator is None
    features = torch.ones(1, field.action_feature_mask.numel())
    retained = field.action_conditioned_features(features)
    wanted = rollout["action_representation"]["feature_groups"]["scalar_exposure"]
    assert torch.equal(torch.nonzero(retained[0]).flatten(), torch.tensor(sorted(wanted)))


def checkpoint_model(case, consumer, policy):
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    from Libs.models.networks.screening_system import ADScreeningSystem
    config = copy.deepcopy(case[1])
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] = policy
    return (MicroenvironmentPretrainingModel(config, False) if consumer == "pretraining"
            else ADScreeningSystem(**screening_kwargs(config)))


def write_and_read_checkpoint(model, consumer, directory):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.exp.training import train_screening_system as screening
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    if consumer == "pretraining":
        path = pretrain.save_training_checkpoint(model, optimizer, directory, "fixture", 0, 0, "single_process")
    else:
        path = directory / "fixture.pt"
        screening.save_checkpoint_state(path, model, optimizer, copy.deepcopy(model.state_dict()),
            1., 0, [], np.random.default_rng(0), 0, 0)
    # Real writers and the same resume state dictionaries; no optimizer step.
    return torch.load(path, map_location="cpu")


def resume_state(payload, consumer):
    from Libs.migrations.legacy_checkpoint import migrate_legacy_checkpoint_keys
    return (migrate_legacy_checkpoint_keys(payload["model_state_dict"]) if consumer == "pretraining"
            else payload["model_state"])


@pytest.mark.parametrize("consumer", ["pretraining", "screening"])
@pytest.mark.parametrize("strict", [True, False])
def test_native_scalar_checkpoint_roundtrip_and_cross_policy(joint_normalized_case, tmp_path, consumer, strict):
    scalar = checkpoint_model(joint_normalized_case, consumer, "scalar_exposure")
    payload = write_and_read_checkpoint(scalar, consumer, tmp_path)
    state = resume_state(payload, consumer)
    key = next(name for name in state if name.endswith("action_conditioning_identity"))
    identity = json.loads(bytes(state[key].tolist()).decode("utf-8"))
    assert identity == {"action_policy": "scalar_exposure", "scalar_exposure_definition": SCALAR_EXPOSURE_DEFINITION}
    loaded = scalar.load_state_dict(state, strict=strict)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    for policy in ("occupancy_program", "drug_id", "target_id"):
        other = checkpoint_model(joint_normalized_case, consumer, policy)
        assert {name: value.shape for name, value in scalar.named_parameters()} == {
            name: value.shape for name, value in other.named_parameters()}
        with pytest.raises(ValueError, match="identity mismatch"):
            other.load_state_dict(state, strict=strict)
        with pytest.raises(ValueError, match="identity mismatch"):
            scalar.load_state_dict(other.state_dict(), strict=strict)


@pytest.mark.parametrize("consumer", ["pretraining", "screening"])
@pytest.mark.parametrize("mutation", ["missing", "unit", "rule", "interval", "axes", "extra"])
def test_scalar_checkpoint_missing_or_conflicting_definition_rejects_strict_false(
        joint_normalized_case, consumer, mutation):
    from Libs.models.modules.quantized_mddm_rollout_operator import validate_native_action_checkpoint
    scalar = checkpoint_model(joint_normalized_case, consumer, "scalar_exposure")
    state = copy.deepcopy(scalar.state_dict())
    key = next(name for name in state if name.endswith("action_conditioning_identity"))
    if mutation == "missing":
        state.pop(key)
    else:
        identity = json.loads(bytes(state[key].tolist()).decode("utf-8"))
        field = {"unit": "unit", "rule": "rule", "interval": "interval", "axes": "reduction_axes", "extra": "extra"}[mutation]
        identity["scalar_exposure_definition"][field] = ["time", "region"] if mutation == "axes" else "contradictory"
        state[key] = torch.tensor(list(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()), dtype=torch.uint8)
    # The non-mutating check protects best/completed states before resume work.
    with pytest.raises(ValueError, match="identity"):
        validate_native_action_checkpoint(scalar, state)
    with pytest.raises(ValueError, match="identity"):
        scalar.load_state_dict(state, strict=False)


@pytest.mark.parametrize("consumer", ["pretraining", "screening"])
@pytest.mark.parametrize("policy", ["occupancy_program", "drug_id"])
def test_legacy_non_scalar_checkpoint_compatibility(joint_normalized_case, consumer, policy):
    from Libs.migrations.legacy_checkpoint import migrate_legacy_checkpoint_keys
    model = checkpoint_model(joint_normalized_case, consumer, policy)
    legacy = {name: value for name, value in model.state_dict().items()
              if not name.endswith("action_conditioning_identity")}
    migrated = migrate_legacy_checkpoint_keys(legacy)
    assert not any(name.endswith("action_conditioning_identity") for name in migrated)
    for strict in (True, False):
        result = model.load_state_dict(migrated, strict=strict)
        assert not result.missing_keys and not result.unexpected_keys
    scalar = checkpoint_model(joint_normalized_case, consumer, "scalar_exposure")
    with pytest.raises(ValueError, match="requires explicit"):
        scalar.load_state_dict(migrated, strict=False)


def test_rank_local_checkpoint_retains_action_identity(joint_normalized_case, tmp_path):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    model = checkpoint_model(joint_normalized_case, "pretraining", "scalar_exposure")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    path = pretrain.save_training_checkpoint(model, optimizer, tmp_path, "fixture", 0, 0, "native_pytorch_fsdp")
    payload = torch.load(path, map_location="cpu")
    identity = "quantized_mddm.action_conditioning_identity"
    assert torch.equal(payload["model_state_dict"][identity], model.state_dict()[identity])
    assert payload["checkpoint_policy"] == "rank_local_shard_state_dict"
    # This exercises the existing serializer only; FSDP resume remains unavailable.


@pytest.mark.parametrize("family", ["deterministic_compatibility", "finite_joint_particles"])
@pytest.mark.parametrize("source_role,requested_role", [
    ("test", "test"), ("test", None), ("train", "test"),
    ("calibration", "train"), ("train", "validation"),
])
def test_native_scalar_role_rejects_before_payload_iteration(family, source_role, requested_role):
    from types import SimpleNamespace
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    class UnreadCache(dict):
        def __iter__(self):
            raise AssertionError("cache fields iterated before role rejection")
        def __getitem__(self, key):
            if key != "split_name":
                raise AssertionError("source payload accessed before role rejection")
            return super().__getitem__(key)
        def get(self, key, default=None):
            if key != "split_name":
                raise AssertionError("source payload accessed before role rejection")
            return super().get(key, default)
    class UnreadSourceRows:
        def __iter__(self):
            raise AssertionError("scalar source reconstructed before role rejection")
    cache = UnreadCache(split_name=source_role, records=UnreadSourceRows(), joint_source=UnreadSourceRows())
    shim = SimpleNamespace(field_rollout_operator=SimpleNamespace(
        quantized_mddm_rollout_operator=object(), action_policy_name="scalar_exposure"),
        world_state_posterior=SimpleNamespace(family=family))
    with pytest.raises(ValueError, match="requested training/development/calibration role"):
        native_diffusion_training_kwargs(shim, {"native_diffusion": cache}, [0], [0],
                                        torch.device("cpu"), split_name=requested_role)


def test_matched_scalar_role_still_validates_real_source(joint_normalized_case, monkeypatch):
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    config = scalar_config(joint_normalized_case[1])
    cached = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a"])
    model = checkpoint_model(joint_normalized_case, "screening", "scalar_exposure")
    calls = []
    original = native.source_site_exposure
    def checked(record):
        calls.append(record["program_uid"])
        return original(record)
    monkeypatch.setattr(native, "source_site_exposure", checked)
    kwargs = native_diffusion_training_kwargs(model, {"labels": [0], "native_diffusion": cached},
        [0], [0], torch.device("cpu"), split_name="train")
    assert calls and kwargs["regimen_uids"] == ["regimen-a"]
    cached["scalar_exposure"]["records"][0][0][0]["values"][0][0][0] += 1.
    with pytest.raises(ValueError, match="source/value/definition"):
        native_diffusion_training_kwargs(model, {"labels": [0], "native_diffusion": cached},
            [0], [0], torch.device("cpu"), split_name="train")
