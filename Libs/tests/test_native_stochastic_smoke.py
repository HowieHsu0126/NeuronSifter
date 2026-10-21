"""Retained public-source runtime checks; no new source, targets, fit or checkpoint."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from Libs.scripts.validators import validate_neuronsifter_stochastic_smoke as smoke
from Libs.tests.test_native_input_staging import forbid_original_access
from Libs.utils.data.native_input_staging import NativeInputLocations
from Libs.utils.data.native_field_normalization import NativeFieldNormalization, normalization_from_config
from Libs.utils.data.raw_data_registry import load_config


def fixture_root():
    value = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not value:
        pytest.skip("Requires the retained593269 four-domain public software fixture")
    return Path(value)


@pytest.fixture(scope="module")
def staged_public(tmp_path_factory):
    original = fixture_root()
    existing = os.environ.get("NRNSIFT_PUBLIC_SMOKE_INPUTS")
    if existing:
        destination = Path(existing)
        config = load_config(destination / "native_smoke_config.yaml")
        manifest = destination / "native_inputs.json"
        config["native_input_staging"] = {"manifest_path": str(manifest)}
        return original, config, manifest, destination
    destination = tmp_path_factory.mktemp("native_public_smoke") / "inputs"
    source_config = original / "config.yaml"
    before = source_config.read_bytes()
    arguments = ["validate_neuronsifter_stochastic_smoke.py", "--config",
        "Libs/configs/exp/native_stochastic_smoke.yaml", "--source-config", str(source_config),
        "--stage-native-inputs", str(destination)]
    def forbidden(*args, **kwargs):
        raise AssertionError("Public input staging must not initialize a model or fit/generate data")
    with patch("sys.argv", arguments), patch.object(smoke, "MicroenvironmentPretrainingModel", forbidden):
        smoke.main()
    assert source_config.read_bytes() == before
    config = load_config(destination / "native_smoke_config.yaml")
    manifest = destination / "native_inputs.json"
    config["native_input_staging"] = {"manifest_path": str(manifest)}
    return original, config, manifest, destination


def test_public_staging_and_target_consumer_rejection(staged_public):
    original, config, manifest, _ = staged_public
    locations = NativeInputLocations.load(manifest)
    assert not locations.stores
    assert len(locations.files) == 10  # index, four registries, one normalizer, four public groups
    assert not any(path.endswith((".pt", ".npy", ".zarray", ".zattrs")) for path in locations.files)
    with forbid_original_access([original]):
        dataset, declarations = smoke.native_smoke_dataset(config)
        assert len(declarations) == 4 and all(row["software_fixture_only"] for row in declarations)
        assert all(len(row["regimen_uids"]) == 3 for row in declarations)
        with pytest.raises(FileNotFoundError, match="not staged"):
            dataset[0]


def test_full_configuration_rejects_changed_scale_budget_and_clock(staged_public):
    _, config, _, _ = staged_public
    for key, value in (("operator_hidden_dim", 12), ("operator_depth", 1)):
        changed = copy.deepcopy(config)
        changed["neuronsifter_scale_registry"]["checkpoint_scales"]["neuronsifter_large"][key] = value
        with pytest.raises(ValueError, match="architecture differs"):
            smoke.native_smoke_config(changed)
    changed = copy.deepcopy(config)
    changed["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]["native"]["denoising_steps"] = 1
    with pytest.raises(ValueError, match="32x2x100"):
        smoke.native_smoke_config(changed)
    changed = copy.deepcopy(config)
    first = next(iter(changed["native_observable_bridge"]["quantity_profile"]["families"].values()))
    first["time_values"][-1] += 1
    with pytest.raises(ValueError, match="profiles/clocks differ"):
        smoke.native_smoke_config(config, changed)


def test_retained_native_wrapper_executes_real_denoisers_without_targets(staged_public, monkeypatch):
    original, full_config, manifest, destination = staged_public
    # The component test retains the fixture's declared small architecture. It
    # tests the real native path and never claims the outer full-model GPU gate.
    config = load_config(original / "config.yaml")
    config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"] = copy.deepcopy(
        full_config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"])
    config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]["native"]["denoising_steps"] = 1
    config["native_input_staging"] = {"manifest_path": str(manifest)}
    def forbidden(*args, **kwargs):
        raise AssertionError("Sampling cannot load checkpoints or future targets")
    monkeypatch.setattr(torch, "load", forbidden)
    from Libs.utils.model import dynamics_store
    monkeypatch.setattr(dynamics_store, "load_zarr_rows", forbidden)
    monkeypatch.setattr(dynamics_store, "load_chunked_rows", forbidden)
    torch.set_num_threads(1)
    with forbid_original_access([original]):
        model = smoke.MicroenvironmentPretrainingModel(config, False).eval()
        _, declarations = smoke.native_smoke_dataset(config)
        model.quantized_mddm.sample_chunk_size = 4
        original_sampler = (model.quantized_mddm.sampler_seed, model.quantized_mddm.inference_sample_count,
                            model.quantized_mddm.inference_denoising_step_count)
        report = {"scope": "small_model_real_path_only", "weights_status": "randomly_initialized",
                  "scientific_quality": "NOT_EVALUATED"}
        smoke.native_runtime_groups(model, config, declarations, report)
        assert original_sampler == (model.quantized_mddm.sampler_seed, model.quantized_mddm.inference_sample_count,
                                    model.quantized_mddm.inference_denoising_step_count)
    assert report["actual_calls"] == {"wrapper": 4, "sampler": 128, "backbone": 256,
                                     "blocks": [256], "terminal_paths": 768}
    assert report["neural_batch_size_histogram"] == {"4": 128, "2": 128}
    assert sorted(len(row["time_hours"]) for row in report["groups"]) == [7, 7, 8, 28]
    assert all(row["particle_count"] == 32 and row["action_count"] == 3 and row["quantity_count"] == 26
               and row["forecast_paths_per_action"] == 64 for row in report["groups"])
    assert report["finite_t0_support_checks"] == "passed"
    assert all(row["physical_t0_max_abs_difference"] == 0 for row in report["groups"])
    (destination.parent / "actual_native_runtime.json").write_text(json.dumps(report, indent=2) + "\n")


def test_staged_normalizer_rejects_missing_copy_and_frozen_belief_reloads(staged_public):
    original, _, manifest, _ = staged_public
    config = load_config(original / "config.yaml")
    config["native_input_staging"] = {"manifest_path": str(manifest)}
    locations = NativeInputLocations.load(manifest)
    with forbid_original_access([original]):
        spec = normalization_from_config(config)
        norm = NativeFieldNormalization(spec, 26, input_locations=locations)
        assert norm.spec == spec
    path = locations.file_path(spec["artifact_ref"])
    renamed = path.with_name(path.name + ".missing_test")
    path.rename(renamed)
    try:
        with forbid_original_access([original]), pytest.raises(FileNotFoundError):
            NativeFieldNormalization(spec, 26, input_locations=locations)
    finally:
        renamed.rename(path)
    # This isolated consumer check reads an existing selected checkpoint only;
    # it is never added to the public smoke package or used by its random model.
    checkpoint = torch.load(original / "native_fit/base_2.pt", map_location="cpu", weights_only=False)
    from Libs.utils.exp.native_comparator_training import FrozenNativeBelief
    with forbid_original_access([original]):
        belief = FrozenNativeBelief(config, checkpoint)
    assert belief.field_normalization.identity == norm.identity
    assert all(not parameter.requires_grad for parameter in belief.parameters())
    assert torch.equal(belief.field_normalization.scale, checkpoint["model_state_dict"]["field_normalization.scale"])


def test_legacy_checker_retains_explicit_diagnostic_label():
    config = load_config("Libs/configs/config_stack.yaml")
    sampler = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["diffusion_sampler"]
    sampler.update(sample_count=2, denoising_step_count=1)
    output = smoke.stochastic_smoke_payload(config, torch.device("cpu"))
    assert output["smoke_status"] == "passed"
    assert output["qualification_scope"] == "legacy_random_residual_diagnostic_not_native_learned_sampling"
    assert output["scientific_quality"] == "NOT_EVALUATED" and not output["publication_eligible"]
