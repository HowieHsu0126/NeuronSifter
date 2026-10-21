"""Disposable software fixtures; no controlled-source production slots are consumed."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from Libs.tests.test_native_diffusion_data import producer_case
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.action_observation_contracts import compile_pharmacological_action
from Libs.utils.data.native_field_normalization import (
    NativeFieldNormalization, fit_multigrid, normalization_from_config, representation,
    validate_artifact,
)
from Libs.utils.data.native_observable_profile import (
    coefficient_support, family_grid, observable_profile, observable_readout,
    profile_identity, quantity_support,
)
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES, compile_stimulation_record
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "Libs/scripts/builders/build_native_diffusion_training_data.py"


def profile_fixture(tmp_path: Path, store_format: str = "zarr_v2"):
    overlay = load_config("Libs/configs/exp/native_observable_bridge.yaml")
    configs, registries, fields, references = {}, {}, {}, {}
    for family in observable_profile(overlay)["families"]:
        directory = tmp_path / family
        config, registry, _, _ = producer_case(directory / "legacy", store_format)
        for key in ("native_observable_bridge", "controlled_cns_source"):
            config[key] = copy.deepcopy(overlay[key])
        config["dynamics_ready_registry"]["species"] = copy.deepcopy(overlay["dynamics_ready_registry"]["species"])
        config["native_diffusion_data"]["species_units"] = copy.deepcopy(overlay["native_diffusion_data"]["species_units"])
        config["native_diffusion_data"]["field_normalization"]["fit_rule"] = overlay["native_diffusion_data"]["field_normalization"]["fit_rule"]
        config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["temporal_wavelet_tokenizer"]["wavelet_level_count"] = 3
        grid = family_grid(observable_profile(config), family)
        registry["grid"] = grid
        store = directory / "observable"
        for row in registry["teachers"]:
            row["trajectory_uid"] += ":" + family
            row["content_hash"] += ":" + family
            row["trajectory_store_path"] = str(store)
        for row in registry["observations"]:
            row["store_path"] = str(store)
        # Pharmacological fixture programs exercise storage admission only.
        # The new source producer's physical action programs are tested separately.
        for row in registry["programs"]:
            old_uid = row["program_uid"]
            row["store_path"] = str(store)
            row["pkpd_context"]["time_grid"] = grid["time_hours"]
            row["compiled_program"] = compile_pharmacological_action(
                row["action"], pkpd_context=row["pkpd_context"],
                baseline_state_posterior=row["baseline_state_posterior"],
                graph_context=row["graph_context"])
            row["program_uid"] = row["compiled_program"]["program_uid"]
            for teacher in registry["teachers"]:
                if teacher["program_uid"] == old_uid:
                    teacher["program_uid"] = row["program_uid"]
        support = quantity_support(grid)
        t = len(grid["time_hours"])
        shape = (4, t, 6, 6, 26)
        trajectory = np.zeros(shape, dtype=np.float32)
        initial = np.zeros((2, 6, 6, 26), dtype=np.float32)
        for si, quantity in enumerate(observable_profile(config)["quantities"]):
            base = -60. if quantity["signed"] else .25
            for index in range(4):
                values = base + .02 * np.arange(t)[:, None, None] + .03 * index
                trajectory[index, ..., si] = np.where(support[..., si], values, 0.)
            initial[..., si] = np.where(support[0, ..., si], base - .1, 0.)
        program_shape = (4, t, 6, 6, 1)
        arrays = {
            "trajectory": trajectory,
            "uncertainty": np.broadcast_to(support.any((0, 1, 2)) * .1, (4, 26)).astype(np.float32).copy(),
            "quantity_support": np.broadcast_to(support, shape).copy(),
            "teacher_confidence": np.array([.25, .5, .75, 1.], dtype=np.float32),
            "initial_state": initial,
            "posterior_context": np.full((2, 6, 6, 12), .2, dtype=np.float32),
            "occupancy": np.full(program_shape, .3, dtype=np.float32),
            "support": np.ones(program_shape, dtype=np.float32),
            "physical_time": np.broadcast_to(np.asarray(grid["time_hours"], dtype=np.float32).reshape(1, t, 1, 1, 1), program_shape).copy(),
        }
        arrays.update({name: np.zeros(program_shape, dtype=np.float32) for name in FORCING_ARRAY_NAMES})
        write_native_microenvironment_store(store, config, registry, arrays, store_format)
        path = directory / "registry.json"
        path.write_text(json.dumps(registry))
        config["native_diffusion_data"]["registry_path"] = str(path)
        configs[family], registries[family], fields[family], references[family] = config, registry, arrays, str(path)
    config = copy.deepcopy(next(iter(configs.values())))
    index = tmp_path / "family-index.json"
    index.write_text(json.dumps({"schema_version": native.FAMILY_INDEX_SCHEMA,
        "quantity_profile_sha256": profile_identity(observable_profile(config)), "family_registries": references}))
    config["native_diffusion_data"]["registry_path"] = str(index)
    return config, registries, fields


def run_builder(config, tmp_path, *arguments):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable, str(BUILDER), "--config", str(path),
        "--max-rows", "12", *arguments], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("store_format", ["zarr_v2", "neuronsifter_chunk_store"])
def test_registered_grids_masked_fit_store_and_pretrainer(tmp_path, store_format):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, registries, arrays = profile_fixture(tmp_path, store_format)
    with pytest.raises(native.NativeDataError, match="explicit source_family_uid"):
        native.load_registry(config)
    artifact_path = tmp_path / "shared-normalization.json"
    run_builder(config, tmp_path, "--split", "train", "--fit-field-normalization", "--output", str(artifact_path))
    artifact = json.loads(artifact_path.read_text())
    assert artifact["trajectory_count"] == 12
    assert len(artifact["registered_grids"]) == 3
    for family, registry in registries.items():
        support = quantity_support(registry["grid"])
        selected = np.flatnonzero(support.any((0, 1, 2)))[15:]
        for si in selected:
            values = arrays[family]["trajectory"][..., si][:, support[..., si]]
            assert artifact["mean"][si] == pytest.approx(values.mean(), abs=1e-5)
            assert artifact["supported_cell_count"][si] == values.size
        derived_path = tmp_path / (family + "-normalized")
        run_builder(config, tmp_path, "--split", "train", "--family", family,
            "--normalization-artifact", str(artifact_path), "--store-format", store_format,
            "--output", str(derived_path))
        selected_config = copy.deepcopy(config)
        selected_config["native_diffusion_data"]["source_family_uid"] = family
        selected_config["native_diffusion_data"]["registry_path"] = str(derived_path / (
            ".zattrs" if store_format == "zarr_v2" else "store_manifest.json"))
        dataset = pretrain.MicroenvironmentTeacherFamilyDataset(selected_config, "train")
        item = dataset[0]
        assert item["quantity_support"].dtype == torch.bool
        assert item["source_family_uid"] == family
        torch.testing.assert_close(item["trajectory"], torch.from_numpy(arrays[family]["trajectory"][0]))
        normalizer = NativeFieldNormalization(normalization_from_config(selected_config), 26)
        field = item["trajectory"][None, None]
        mask = item["quantity_support"]
        transformed = normalizer.normalize(field, mask)
        assert (transformed[..., ~mask] == 0).all()
        torch.testing.assert_close(normalizer.inverse(transformed, mask), field)
        normalizer.assert_physical_time(item["physical_time"][None, None])
        diagnostics = normalizer.excursions(field, mask)
        assert diagnostics["source_reference_denominator"][0, 0].tolist() == mask.sum((0, 1, 2)).tolist()
        bound_grid = native.load_registry(selected_config)["grid"]
        readout = observable_readout(field[:, 0], torch.ones_like(field[:, 0]), bound_grid)
        expected_family = observable_profile(config)["families"][family]
        assert readout["feature_names"] == expected_family["observable_names"]
        assert readout["time_values"] == expected_family["time_values"]
        assert readout["time_unit"] == expected_family["time_unit"]
        assert readout["predictive_mean"].shape[-1] == len(selected)
        if "electrophysiology" in family:
            assert (readout["predictive_mean"][..., 0] < 0).all()


def test_masked_fit_rejects_missing_support_role_and_tampered_clock(tmp_path):
    config, registries, _ = profile_fixture(tmp_path)
    batches = []
    for registry in registries.values():
        records = native.select_records(registry, config, "train")
        arrays, _ = native.load_records(registry, records, config)
        batches.append((registry["grid"], records, arrays))
    with pytest.raises(ValueError, match="every registered family"):
        fit_multigrid(batches[:1], config)
    bad = copy.deepcopy(batches)
    bad[0][1][0]["teacher"]["split_name"] = "calibration"
    with pytest.raises(ValueError, match="training A/B"):
        fit_multigrid(bad, config)
    bad = copy.deepcopy(batches)
    bad[0][2]["quantity_support"][0, 0, 0, 0, -1] = True
    with pytest.raises(ValueError, match="exact declared quantity support"):
        fit_multigrid(bad, config)
    artifact = fit_multigrid(batches, config)
    grid = copy.deepcopy(batches[0][0])
    grid["time_hours"][-1] += 1.
    with pytest.raises(ValueError, match="unregistered normalization grid"):
        validate_artifact(artifact, grid, representation(config))


def test_odd_haar_padding_is_not_observed_support():
    mask = torch.ones((7, 1, 1, 1), dtype=torch.bool)
    packed = coefficient_support(mask, 3)
    assert packed[:, 0, 0, 0].tolist() == [True, True, True, True, True, True, True, False]
    assert int(packed.sum()) == 7


def test_supported_categorical_loss_has_no_missing_cell_gradient():
    from Libs.models.modules.quantized_mddm_rollout_operator import masked_categorical_objective
    logits = torch.zeros((2, 3, 4), requires_grad=True)
    support = torch.tensor([[True, False, False], [False, True, False]])
    target = torch.zeros((2, 3), dtype=torch.long)
    loss = masked_categorical_objective(logits, target, support.clone(), torch.tensor(1.),
                                        torch.tensor(.7), support=support)
    torch.testing.assert_close(loss, torch.tensor(.7 * np.log(4), dtype=loss.dtype))
    loss.backward()
    assert (logits.grad[~support] == 0).all()
    assert logits.grad[support].abs().sum() > 0
    with pytest.raises(ValueError, match="only supported coefficients"):
        masked_categorical_objective(logits, target, torch.ones_like(support), torch.tensor(1.),
                                     torch.tensor(.7), support=support)


def test_native_profile_loss_backward_sampler_and_checkpoint(tmp_path):
    from torch.utils.data._utils.collate import default_collate
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.models.modules.quantized_mddm_rollout_operator import validate_native_action_checkpoint
    config, registries, _ = profile_fixture(tmp_path)
    artifact = tmp_path / "shared-normalization.json"
    run_builder(config, tmp_path, "--split", "train", "--fit-field-normalization", "--output", str(artifact))
    family = "electrophysiology_conductance_world"
    store = tmp_path / "normalized-consumer"
    run_builder(config, tmp_path, "--split", "train", "--family", family,
                "--normalization-artifact", str(artifact), "--output", str(store))
    config["native_diffusion_data"]["registry_path"] = str(store / ".zattrs")
    config["native_diffusion_data"]["source_family_uid"] = family
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    batch = default_collate([dataset[0]])
    torch.manual_seed(9307)
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    coordinates = pretrain.node_coordinates(6, 6, torch.device("cpu"))
    output = model(None, coordinates, batch)
    loss = output["rollout_auxiliary_loss"]
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.quantized_mddm.quantity_support_projection.weight.grad.abs().sum() > 0
    assert model.quantized_mddm.token_denoiser.weight.grad.abs().sum() > 0
    mask = batch["quantity_support"][:, 1:].reshape_as(output["state_prediction"])
    assert (output["state_prediction"][~mask] == 0).all()
    objective, _ = pretrain.objective_loss(output, batch, config)
    altered = dict(output)
    altered["state_prediction"] = output["state_prediction"].masked_fill(~mask, 1e6)
    altered["counterfactual_prediction"] = output["counterfactual_prediction"].masked_fill(~mask, -1e6)
    unaffected, _ = pretrain.objective_loss(altered, batch, config)
    torch.testing.assert_close(objective, unaffected)
    readout = output["observable_prediction"]
    assert readout["predictive_mean"].shape == (1, 1, 7, 3)
    assert (readout["predictive_mean"][..., 0] < 0).all()
    assert readout["time_values"] == [0., 2., 4., 6., 8., 10., 12.]
    packed = coefficient_support(batch["quantity_support"][0], 3)
    diagnostics = output["field_normalization_diagnostics"]
    torch.testing.assert_close(diagnostics["coefficient_denominator"][0, 0], packed.sum((0, 1, 2)))
    state = model.state_dict()
    key = "quantized_mddm.action_conditioning_identity"
    state[key] = state[key].clone()
    state[key][0] ^= 1
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_native_action_checkpoint(model, state)
    missing = dict(batch)
    del missing["quantity_support"]
    with pytest.raises(ValueError, match="quantity_support"):
        model(None, coordinates, missing)


def test_profile_screening_cache_preserves_boolean_support(tmp_path):
    from types import SimpleNamespace
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    config, registries, _ = profile_fixture(tmp_path)
    family = "molecular_pharmacological_causal_world"
    config["native_diffusion_data"]["source_family_uid"] = family
    native_payload = native.screening_native_payload(config, "train", ["sample-a"], ["regimen-a"])
    payload = {"sample_uids": ["sample-a"], "labels": np.ones(1), "native_diffusion": native_payload}
    native.validate_cached_native({"rna": {"train": payload}}, ["regimen-a"], config)
    model = SimpleNamespace(field_rollout_operator=SimpleNamespace(
        quantized_mddm_rollout_operator=object(), field_normalization=NativeFieldNormalization(None, 26),
        action_policy_name="occupancy_program"))
    selected = native_diffusion_training_kwargs(model, payload, [0], [0], torch.device("cpu"))
    assert selected["diffusion_context"]["quantity_support"].dtype == torch.bool
    native_payload["quantity_support"] = native_payload["quantity_support"].astype(np.float32)
    with pytest.raises(native.NativeDataError, match="support/profile"):
        native.validate_cached_native({"rna": {"train": payload}}, ["regimen-a"], config)


@pytest.mark.parametrize("family,kind,unit", [
    ("electrophysiology_conductance_world", "electrical_stimulation", "uA"),
    ("calcium_synaptic_world", "optical_stimulation", "mW"),
])
def test_typed_stimulation_compiler_and_native_sensitivity(tmp_path, family, kind, unit):
    from torch.utils.data._utils.collate import default_collate
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    config, registries, fields = profile_fixture(tmp_path)
    registry, arrays = registries[family], fields[family]
    grid = registry["grid"]
    profile = observable_profile(config)
    quantity = next(q for q in profile["quantities"] if q.get("family_uid") == family)
    store = tmp_path / "stimulation"
    old_programs = registry["programs"]
    programs = []
    for index, old in enumerate(old_programs):
        action = {key: copy.deepcopy(value) for key, value in old["action"].items()
                  if key not in {"target_set", "agent_set", "dose", "dose_unit", "route"}}
        action.update(intervention_type=kind,
            stimulation_or_environment={"affected_state_variables": profile["families"][family]["observable_names"],
                "magnitude_or_program": {"time_values": [0., 4.], "values": [.8, .2],
                    "time_unit": "ms", "value_unit": unit, "interpolation": "left_hold",
                    "incidence": quantity["incidence"]},
                "uncertainty": {"status": "specified_input_waveform"}},
            duration={"value": grid["query_time"]["values"][-1], "unit": "ms"})
        binding = {"source_uid": "source", "posterior_uid": old["posterior_uid"],
            "sample_uid": registry["teachers"][index]["sample_uid"], "support_uid": "fixture-support",
            "particle_uid": "fixture-particle", "particle_basis": {key: [1.] for key in ("z_tx","z_pk","z_env","z_ephys","z_morph")},
            "regimen_uid": old["regimen_uid"]}
        record = compile_stimulation_record(action, grid, binding)
        record.update(store_path=str(store), store_format="zarr_v2", program_index=index,
                      target_program_uid="target-program")
        programs.append(record)
        registry["teachers"][index].update(program_uid=record["program_uid"], trajectory_store_path=str(store))
        for name in (*FORCING_ARRAY_NAMES, "occupancy", "support", "physical_time"):
            arrays[name][index] = np.asarray(record["compiled_program"]["arrays"][name], dtype=np.float32)
    registry["programs"] = programs
    for row in registry["observations"]:
        row["store_path"] = str(store)
    write_native_microenvironment_store(store, config, registry, arrays)
    config["native_diffusion_data"].update(registry_path=str(store/".zattrs"), source_family_uid=family)
    dataset = pretrain.MicroenvironmentTeacherFamilyDataset(config, "train")
    batch = default_collate([dataset[0]])
    assert batch["occupancy"].count_nonzero() == batch["support"].count_nonzero() == 0
    torch.manual_seed(913)
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    coordinates = pretrain.node_coordinates(6, 6, torch.device("cpu"))
    first = model(None, coordinates, batch)
    changed = dict(batch)
    name = "electrical_current_uA" if unit == "uA" else "optical_power_mW"
    changed[name] = batch[name] * .5
    second = model(None, coordinates, changed)
    assert not torch.equal(first["state_prediction"], second["state_prediction"])
    first["rollout_auxiliary_loss"].backward()
    assert model.quantized_mddm.condition_projection.weight.grad.abs().sum() > 0
    invalid = copy.deepcopy(programs[0])
    invalid["action"]["stimulation_or_environment"]["magnitude_or_program"]["value_unit"] = "mg/kg"
    with pytest.raises(ValueError, match="units/clock"):
        native.validate_compiled_program(invalid)
    with pytest.raises(native.NativeDataError, match="Chemical identity"):
        native.program_action_identity(programs[0], action_policy="drug_id")


def test_observable_readout_chunk_merge_preserves_units_clock_and_boolean_masks():
    import copy
    import torch
    import pytest
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.data.native_observable_profile import (
        observable_profile, family_grid, observable_readout, merge_observable_readouts)
    profile = observable_profile(load_config("Libs/configs/exp/native_observable_bridge.yaml"))
    grid = family_grid(profile, "electrophysiology_conductance_world")
    mean = torch.zeros(4, 7, 6, 6, 26)
    mean[:, :, 3, 4, 15] = torch.arange(4)[:, None] - 68
    readout = observable_readout(mean, torch.ones_like(mean), grid)
    tensor_keys = ("predictive_mean", "predictive_scale", "feature_mask", "time_mask")
    readout.update({key: readout[key].reshape(2, 2, *readout[key].shape[1:]) for key in tensor_keys})
    chunks = [{key: value[:, i:i+1] if torch.is_tensor(value) else value
               for key, value in readout.items()} for i in range(2)]
    merged = merge_observable_readouts(chunks)
    for key in tensor_keys:
        assert torch.equal(merged[key], readout[key])
    assert merged["feature_units"] == ["mV", "umol/L", "Hz"]
    assert merged["time_unit"] == "ms"
    assert merged["time_mask"].dtype == torch.bool
    wrong = copy.deepcopy(chunks[1])
    wrong["time_unit"] = "hour"
    with pytest.raises(ValueError, match="metadata"):
        merge_observable_readouts([chunks[0], wrong])
    wrong = copy.deepcopy(chunks[1])
    wrong["feature_mask"] = wrong["feature_mask"].float()
    with pytest.raises(ValueError, match="boolean"):
        merge_observable_readouts([chunks[0], wrong])


def test_consumer_check_objectives_do_not_bind_scientific_training():
    from Libs.exp.training.pretrain_microenvironment_world_model import observable_consumer_check_config
    config=load_config("Libs/configs/exp/native_observable_bridge.yaml")
    original=copy.deepcopy(config)
    effective,binding=observable_consumer_check_config(config)
    path=lambda c:c["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["joint_objectives"]
    assert path(config)=={"observation_likelihood_weight":None,"categorical_kl_weight":None}
    assert path(effective)=={"observation_likelihood_weight":.7,"categorical_kl_weight":.2}
    assert config==original
    assert binding["provenance"]=="Libs/tests/test_joint_world_posterior.py::joint_case"
    assert binding["scope"]=="nontraining_consumer_check_only"
    assert binding["scientific_objective_qualification"] is False
    config["native_observable_bridge"]["native_consumer_check"]["joint_objectives"]["scope"]="training"
    with pytest.raises(ValueError,match="diagnostic"):
        observable_consumer_check_config(config)


def test_numeric_support_storage_descriptor_rejects_before_payload_read(tmp_path,monkeypatch):
    config,registries,_=profile_fixture(tmp_path)
    registry=next(iter(registries.values()))
    records=native.select_records(registry,config,"train")
    path=Path(records[0]["teacher"]["trajectory_store_path"])/"quantity_support"/".zarray"
    header=json.loads(path.read_text());header["dtype"]="<f4";path.write_text(json.dumps(header))
    monkeypatch.setattr(native.dynamics_store,"load_zarr_rows",lambda *args:pytest.fail("numeric mask reached payload read"))
    with pytest.raises(ValueError,match="boolean dtype"):
        native.load_records(registry,records,config)


def test_physical_typed_normalization_still_requires_boolean_support():
    profile=observable_profile(load_config("Libs/configs/exp/native_observable_bridge.yaml"))
    grid=family_grid(profile,"electrophysiology_conductance_world")
    normalizer=NativeFieldNormalization({"kind":"physical","quantity_profile":profile},26)
    value=torch.ones(1,7,6,6,26)
    support=torch.as_tensor(quantity_support(grid))
    for transform in (normalizer.normalize,normalizer.inverse):
        with pytest.raises(ValueError,match="explicit quantity support"):
            transform(value)
        with pytest.raises(ValueError,match="boolean"):
            transform(value,support.float())
        transformed=transform(value,support)
        assert (transformed[:,~support]==0).all()
        assert (transformed[:,support]==1).all()
    # No reference was fitted in physical mode; this never claims source-range evidence.
    assert normalizer.excursions(value)["field_reference_available"].sum()==0


def test_native_consumer_cli_requires_source_before_reservation(tmp_path):
    output = tmp_path / "missing_source"
    config_path = tmp_path / "consumer.yaml"
    config_path.write_text(yaml.safe_dump({
        "config_includes": [str(ROOT / "Libs/configs/exp/native_observable_bridge.yaml")],
        "native_observable_bridge": {"output_root": str(output)},
    }))
    completed = subprocess.run([
        sys.executable, str(ROOT / "Libs/exp/training/pretrain_microenvironment_world_model.py"),
        "--config", str(config_path), "--native-consumer-check",
    ], cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "FileNotFoundError" in completed.stderr
    assert str(output / "source_receipt.json") in completed.stderr
    assert not output.exists()


def _consumer_routing_fixture(tmp_path, monkeypatch, repaired):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
    from Libs.scripts.builders import build_controlled_native_source as builder
    name = "native_observable_parameter_conditioning" if repaired else "native_observable_bridge"
    config = load_config("Libs/configs/exp/" + name + ".yaml")
    root = tmp_path / "source"
    selected = root / "physical_parameter_conditioning" if repaired else root
    selected.mkdir(parents=True)
    config["native_observable_bridge"]["output_root"] = str(root)
    config["native_diffusion_data"]["registry_path"] = str(selected / "model_registry.json")
    source = {"source_identity": {"producer_commit": "disposable_producer"}}
    (root / "source_receipt.json").write_text(json.dumps(source))
    consumption = {"status": "complete", "admitted_count": 36, **source}
    if repaired:
        consumption.update(operation="bind_retained_physical_candidates", source_operations=0,
                           model_initializations=0, array_values_unchanged=True, compiled_programs_unchanged=True,
                           model_registry_path=str(selected / "model_registry.json"), consumer_commit="disposable_binder")
        previous = {"status": "complete", "model_initializations": 1,
                    "consumer_commit": "disposable_original_consumer", **source}
        (root / "native_consumer_check.json").write_text(json.dumps(previous))
    (selected / "native_consumption.json").write_text(json.dumps(consumption))
    (selected / "model_registry.json").write_text(json.dumps({
        "family_registries": {family: "disposable_unused_registry" for family in observable_profile(config)["families"]}}))
    monkeypatch.setattr(builder, "verify_observable_source", lambda *args: None)
    monkeypatch.setattr(pretrain, "MicroenvironmentPretrainingModel",
                        lambda *args: pytest.fail("routing test must not construct a model"))
    monkeypatch.setenv("SLURM_JOB_ID", "disposable")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    return pretrain, builder, config, root, selected


@pytest.mark.parametrize("repaired", [False, True])
def test_consumer_check_routes_before_model_construction(tmp_path, monkeypatch, repaired):
    pretrain, builder, config, root, selected = _consumer_routing_fixture(tmp_path, monkeypatch, repaired)
    original = (root / "native_consumer_check.json").read_bytes() if repaired else None
    recorded = []
    def stop_before_construction(path, report):
        recorded.append((path, report))
        raise RuntimeError("disposable receipt boundary")
    monkeypatch.setattr(builder, "write_json", stop_before_construction)
    with pytest.raises(RuntimeError, match="disposable receipt boundary"):
        pretrain.native_consumer_check(config)
    path, report = recorded[0]
    assert path == selected / "native_consumer_check.json"
    assert report["input_binding"]["native_consumption_path"] == str(selected / "native_consumption.json")
    assert report["input_binding"]["model_registry_path"] == config["native_diffusion_data"]["registry_path"]
    assert report["diagnostic_seed"] == 8609101 and (report["draws"], report["reverse_steps"]) == (2, 4)
    assert report["model_initializations"] == 1 and report["optimizer_steps"] == report["checkpoint_writes"] == 0
    if repaired:
        assert (root / "native_consumer_check.json").read_bytes() == original
        assert report["input_binding"]["permitted_model_initializations_total"] == 2
        assert report["input_binding"]["seconds"] == 900
        assert report["input_binding"]["prior_diagnostic_elapsed_seconds"] == 515
        assert report["input_binding"]["prior_run_receipt"] == str(root / "native_consumer_check.json")
    assert not path.exists()


@pytest.mark.parametrize("failure", ["incomplete_admission", "wrong_registry", "missing_exception", "occupied_receipt"])
def test_corrective_consumer_rejects_before_model_construction(tmp_path, monkeypatch, failure):
    pretrain, _, config, root, selected = _consumer_routing_fixture(tmp_path, monkeypatch, True)
    original = (root / "native_consumer_check.json").read_bytes()
    consumption_path = selected / "native_consumption.json"
    consumption = json.loads(consumption_path.read_text())
    if failure == "incomplete_admission":
        consumption["admitted_count"] = 35
    elif failure == "wrong_registry":
        consumption["model_registry_path"] = str(root / "model_registry.json")
    elif failure == "missing_exception":
        config.pop("physical_parameter_consumer_check")
    else:
        (selected / "native_consumer_check.json").write_text('{"status":"started"}')
    consumption_path.write_text(json.dumps(consumption))
    with pytest.raises((ValueError, FileExistsError)):
        pretrain.native_consumer_check(config)
    assert (root / "native_consumer_check.json").read_bytes() == original
    assert config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["joint_objectives"] == {
        "observation_likelihood_weight": None, "categorical_kl_weight": None}
