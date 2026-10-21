"""Staged metadata and nonempty original-channel assimilation; no source/model calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Libs.models.modules.world_state import _normalized_log_weights
from Libs.scripts.builders.build_sample_observation_registry import NativePredoseMeasurementProvider
from Libs.tests.test_native_arrived_observation import (
    FAMILIES, prohibit_nn_source_and_targets, retained_filter_cases,
)
from Libs.tests.test_native_input_staging import forbid_original_access
from Libs.utils.data.native_input_staging import NativeInputLocations, stage_native_inputs
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.native_predose_evaluation import predose_source_admission
from Libs.utils.model.native_predose_query import native_predose_channel, run_native_predose_episode


@pytest.mark.parametrize("fixture", [True, False])
def test_staged_admission_metadata_uses_only_local_paths_and_rejects_missing(tmp_path, fixture):
    original = tmp_path/"original"
    original.mkdir()
    store = original/"store"
    store.mkdir()
    qc = original/"software_fixture.json"
    qc.write_text(json.dumps({"scope": "artificial_software_fixture_no_source_trajectory",
        "physical_source_trajectory_calls": 0, "target_generation": "initial_complete_tuple_zero_plus_masked_linear_ramp"}))
    teacher = {"posterior_uid": "p", "trajectory_store_path": str(store), "store_format": "neuronsifter_chunk_store",
        "trajectory_index": 0, "qc_catalog_ref": str(qc), "software_fixture_only": fixture}
    registry = {"posterior_family": "finite_joint_particles", "grid": {},
        "joint_sources": [{"source_posterior_uid": "p"}], "teachers": [teacher]}
    registry_file = original/"registry.json"
    registry_file.write_text(json.dumps(registry))
    arrays = {}
    for name in ("trajectories", "teacher_uncertainty", "teacher_confidence"):
        local = store/"arrays"/name
        local.mkdir(parents=True)
        np.save(local/"chunk_000000.npy", np.zeros((1, 1), dtype=np.float32))
        arrays[name] = {"shape": [1, 1], "dtype": "<f4", "chunk_size": 1, "relative_root": "arrays/"+name}
    (store/"store_manifest.json").write_text(json.dumps({"metadata": {"native_diffusion": registry}, "arrays": arrays}))
    base = original/"base_admission.json"
    base.write_text(json.dumps({"status": "complete", "admitted_count": 1, "source_identity": {"fixture": "metadata_only"}}))
    named = original/"native_consumption.json"
    named.write_text(json.dumps({"status": "complete", "admitted_count": 1, "source_identity": {"fixture": "metadata_only"},
        "operation": "bind_retained_physical_candidates", "original_consumption_ref": str(base),
        "array_values_unchanged": True, "compiled_programs_unchanged": True}))
    config = {"native_diffusion_data": {"registry_path": str(registry_file)}, "native_predose_query": {"enabled": True},
        "native_observable_bridge": {"budget": {"cases": 1}},
        "screening_pipeline": {"screening_benchmark": {"model_architecture": {
            "rollout_operator": {"transition_backbone": {"kind": "neuronsifter_world_action"}}}}}}
    child = SimpleNamespace(native_registry=registry, native_records=[{"teacher": teacher}])
    dataset = SimpleNamespace(split_name="validation", group_uids=["selected"], input_locations=None,
                              groups=[(child, [0], None)])
    manifest = stage_native_inputs(config, [dataset], tmp_path/"staged")
    config["native_input_staging"] = {"manifest_path": str(manifest)}
    locations = NativeInputLocations.load(manifest)
    required = qc if fixture else base
    assert str(required) in locations.files
    assert set(locations.groups) == {"validation"}
    with forbid_original_access([original]):
        result = predose_source_admission(config, dataset)
        assert result["scope"] == ("artificial_software_fixture" if fixture else "admitted_controlled_source")
        with pytest.raises(ValueError, match="outside its staged"):
            predose_source_admission(config, SimpleNamespace(split_name="train", group_uids=["selected"], groups=dataset.groups))
    locations.file_path(required).unlink()
    with forbid_original_access([original]), pytest.raises(FileNotFoundError):
        predose_source_admission(config, dataset)
    if not fixture:
        config.pop("native_input_staging")
        data = json.loads(base.read_text()); data["status"] = "incomplete"; base.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="complete unchanged"):
            predose_source_admission(config, dataset)


@pytest.mark.parametrize("family", FAMILIES)
def test_nonempty_retained_channel_matches_joint_assimilation(retained_filter_cases, family):
    config, declaration, inputs, _ = retained_filter_cases[family]
    config = copy.deepcopy(config)
    current = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_predose_query"] = current["native_predose_query"]
    config["schemas"]["native_predose_episode"] = current["schemas"]["native_predose_episode"]
    channel = native_predose_channel(inputs, declaration, config)
    coordinate = channel.available[0]
    values = channel.parameters["mean"][0, :, coordinate]
    labels = (values > values.median()).float()
    # Explicit artificial decision-loss fixture, not fitted CNS loss or evidence
    # of scientific headroom. All original source/readout/tuple values stay intact.
    loss = torch.stack((labels, 1-labels, torch.full_like(labels, .8)), -1)[:, None]
    private = inputs.posterior.support.initial_field[:, 0].clone()
    provider = NativePredoseMeasurementProvider(private, channel, torch.Generator().manual_seed(84))
    updated, observations, report = run_native_predose_episode(channel, inputs.posterior.log_weights,
        loss, inputs.regimen_uids, provider, config, origin_posterior_uid=inputs.posterior_uid)
    assert observations and report["accepted_query_count"] >= 1
    posterior = inputs.posterior
    for observation in observations:
        posterior = posterior.assimilate(observation, parent_posterior_uid=posterior.posterior_uid, assays={})
    assert torch.equal(updated, posterior.log_weights)
    assert posterior.support is inputs.posterior.support
    assert posterior.particle_context is inputs.posterior.particle_context
    assert posterior.observation_features is inputs.posterior.observation_features
    assert report["particle_uids"] == list(inputs.particle_uids[0])
    assert not report["initial_encoder_packet_changed"] and not report["physical_time_advanced"]
    assert report["regimen_uids"] == list(inputs.regimen_uids)
