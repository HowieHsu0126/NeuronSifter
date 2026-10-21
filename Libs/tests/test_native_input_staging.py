"""Node-local admitted input copies on retained complete groups; no model/source fit."""
import builtins
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data import native_input_staging as staging
from Libs.utils.data.native_field_normalization import normalization_from_config, representation
from Libs.utils.data.native_input_staging import (
    NativeInputLocations, array_input_paths, stage_native_inputs,
)
from Libs.utils.data.native_joint_source_storage import (
    read_joint_source_reference, write_joint_source_reference,
)
from Libs.utils.data.raw_data_registry import load_config, repo_root, resolve_repo_path
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.model import native_scientific_source as source

FAMILIES = {"electrophysiology_conductance_world", "calcium_synaptic_world",
            "molecular_pharmacological_causal_world", "cns_long_horizon_world"}


@pytest.fixture(scope="module", autouse=True)
def no_model_or_source_execution():
    def forbidden(*args, **kwargs):
        raise AssertionError("Staging tests cannot initialize models or draw/generate sources")
    with patch.object(torch.nn.Module, "__init__", forbidden), patch.object(source, "draw_group", forbidden), \
            patch.object(torch, "multinomial", forbidden):
        yield


@contextmanager
def forbid_original_access(roots):
    roots = tuple(str(root).rstrip("/") for root in roots)
    def wrapped(original):
        def guarded(path, *args, **kwargs):
            if not isinstance(path, int):
                value = os.fsdecode(os.fspath(path))
                if any(value == root or value.startswith(root + "/") for root in roots):
                    raise AssertionError("Original input access after staging: " + value)
            return original(path, *args, **kwargs)
        return guarded
    with patch.object(builtins, "open", wrapped(builtins.open)), patch.object(io, "open", wrapped(io.open)), \
            patch.object(os, "stat", wrapped(os.stat)), patch.object(os, "lstat", wrapped(os.lstat)), \
            patch.object(np, "load", wrapped(np.load)), patch.object(np, "fromfile", wrapped(np.fromfile)):
        yield


def invoke_staging(config_path, destination, *options):
    arguments = ["pretrain_microenvironment_world_model.py", "--config", str(config_path),
                 "--stage-native-inputs", str(destination), *options]
    before = torch.random.get_rng_state().clone()
    with patch("sys.argv", arguments):
        pretrain.main()
    assert torch.equal(torch.random.get_rng_state(), before)
    return destination / "native_inputs.json"


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    location = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not location:
        pytest.skip("Requires the retained admitted four-domain artificial fixture")
    source_root = Path(location)
    config_path = source_root / "config.yaml"
    config = load_config(config_path)
    original_bytes = config_path.read_bytes()
    root = tmp_path_factory.mktemp("native_input_staging")
    manifest = invoke_staging(config_path, root / "staged")
    assert config_path.read_bytes() == original_bytes
    staged = copy.deepcopy(config)
    staged["native_input_staging"] = {"manifest_path": str(manifest)}
    return config, staged, root, source_root, config_path


def assert_same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_actual_dataset_items_bindings_and_scientific_profile_are_unchanged(retained):
    config, staged, _, original_root, _ = retained
    original_normalization = normalization_from_config(config)
    profile = source.native_training_profile(config, original_normalization["sha256"])
    original = {role: pretrain.NativeJointGroupDataset(config, role,
        group_uids={row["sample_uid"] for row in source.selected_group_definitions(config, role)})
        for role in ("train", "development")}
    with forbid_original_access([original_root]):
        local = {role: pretrain.NativeJointGroupDataset(staged, role) for role in original}
        assert normalization_from_config(staged) == original_normalization
        assert source.native_training_profile(staged, original_normalization["sha256"]) == profile
        assert resolve_training_seed(staged, require_selected=True) == resolve_training_seed(config, require_selected=True)
    families, counts = set(), {}
    for role, dataset in original.items():
        assert local[role].group_uids == dataset.group_uids
        assert source.selected_group_definitions(config, role) == source.selected_group_definitions(staged, role)
        counts[role] = len(dataset)
        for index in range(len(dataset)):
            before = dataset[index]
            child, indices, _ = dataset.groups[index]
            teachers = [child.native_records[position]["teacher"] for position in indices]
            records = native.join_registry(child.native_registry, child.config, teachers)
            _, logical_bindings = native.load_records(child.native_registry, records, child.config)
            with forbid_original_access([original_root]):
                after = local[role][index]
                local_child, _, _ = local[role].groups[index]
                mapped = local[role].input_locations
                _, staged_bindings = native.load_records(local_child.native_registry,
                    records, local_child.config, input_locations=mapped)
            assert_same(before, after)
            assert logical_bindings == staged_bindings
            assert child.native_registry == local_child.native_registry
            group = after["native_joint_group"]
            assert len(group["items"]) == 3
            assert len(group["joint_source"]["support"]["particle_uids"][0]) == 32
            families.add(group["items"][0]["source_family_uid"])
    assert families == FAMILIES
    output = os.environ.get("NRNSIFT_STAGING_RESULT")
    if output:
        Path(output).write_text(json.dumps({"families": sorted(families), "groups": counts,
            "tuple_count": 32, "actions_per_group": 3, "profile_invariant": True,
            "normalization_invariant": True, "logical_bindings_invariant": True,
            "original_storage_accesses_during_guard": 0, "model_initializations": 0,
            "source_draws": 0, "new_normalization_fits": 0}, indent=2) + "\n")


def local_case(retained):
    _, staged, _, original_root, _ = retained
    dataset = pretrain.NativeJointGroupDataset(staged, "train")
    child, positions, _ = dataset.groups[0]
    teacher = child.native_records[positions[0]]["teacher"]
    mapping = dataset.input_locations
    store = mapping.logical_path(teacher["trajectory_store_path"])
    descriptor = store / ("store_manifest.json" if teacher["store_format"] == "neuronsifter_chunk_store" else ".zattrs")
    if teacher["store_format"] == "neuronsifter_chunk_store":
        metadata = json.loads(mapping.read_text(descriptor))["arrays"]["trajectories"]
    else:
        metadata = json.loads(mapping.read_text(store / "trajectories" / ".zarray"))
    chunk = array_input_paths(store, "trajectories", metadata, teacher["store_format"], [teacher["trajectory_index"]])[0]
    return dataset, child, mapping, descriptor, chunk, original_root



def test_descriptor_and_no_teacher_reads_do_not_open_target_chunks(retained):
    dataset, child, mapping, _, chunk, original_root = local_case(retained)
    _, positions, _ = dataset.groups[0]
    teachers = [child.native_records[index]["teacher"] for index in positions]
    path = mapping.file_path(chunk)
    renamed = path.with_name(path.name + ".test_metadata_only")
    path.rename(renamed)
    try:
        with forbid_original_access([original_root]):
            records = native.join_registry(child.native_registry, child.config, teachers, input_locations=mapping)
            bindings = native.descriptor_bindings(records, child.native_registry, child.config, input_locations=mapping)
            arrays, other_bindings = native.load_records(child.native_registry, records, child.config,
                include_teacher=False, input_locations=mapping)
        assert arrays == {} and bindings == other_bindings
    finally:
        renamed.rename(path)


def test_missing_local_chunk_does_not_fall_back(retained):
    dataset, _, mapping, _, chunk, original_root = local_case(retained)
    path = mapping.file_path(chunk)
    renamed = path.with_name(path.name + ".test_missing")
    path.rename(renamed)
    try:
        with forbid_original_access([original_root]), pytest.raises(FileNotFoundError):
            dataset[0]
    finally:
        renamed.rename(path)


@pytest.mark.parametrize("kind", ["descriptor", "normalizer", "finite_array"])
def test_changed_local_bytes_fail_on_the_next_read(retained, kind):
    dataset, child, mapping, descriptor, chunk, original_root = local_case(retained)
    dataset[0]  # Existing per-read checks must still run after a successful item read.
    logical = {"descriptor": descriptor, "normalizer": child.native_registry["grid"]["normalization"]["artifact_ref"],
               "finite_array": chunk}[kind]
    path = mapping.file_path(logical)
    original = path.read_bytes()
    changed = bytearray(original)
    # Flip a low-order data bit; the array case stays finite and cannot rely on NaN rejection.
    changed[-4 if kind == "finite_array" else len(changed)//2] ^= 1
    path.write_bytes(changed)
    try:
        with forbid_original_access([original_root]), pytest.raises(ValueError, match="changed"):
            dataset[0]
    finally:
        path.write_bytes(original)


def test_existing_descriptor_identity_checks_remain_active(retained):
    dataset, _, mapping, descriptor, _, original_root = local_case(retained)
    path = mapping.file_path(descriptor)
    original = path.read_bytes()
    metadata = json.loads(original)
    registry = metadata.get("metadata", metadata)["native_diffusion"]
    registry["grid"]["physical_time_unit"] = "seconds"
    changed = json.dumps(metadata).encode()
    original_binding = dict(mapping.files[str(descriptor)])
    path.write_bytes(changed)
    # Bypass only the copy check to prove the original descriptor validator still rejects.
    mapping.files[str(descriptor)] = {"size_bytes": len(changed), "sha256": hashlib.sha256(changed).hexdigest()}
    try:
        with forbid_original_access([original_root]), pytest.raises(ValueError, match="grid metadata mismatch"):
            dataset[0]
    finally:
        path.write_bytes(original)
        mapping.files[str(descriptor)] = original_binding


def inventory_case(tmp_path, store_format):
    """Tiny explicit chunk inventory; no source generation or model execution."""
    original = tmp_path / "original"
    original.mkdir()
    store = original / "store"
    store.mkdir()
    raw = {"source_posterior_uid": "complete", "values": [[1.25, -3.0]]}
    reference = write_joint_source_reference(original / "registry-group.gz", raw, max_bytes=4096)
    descriptor_reference = write_joint_source_reference(original / "descriptor-group.gz", raw, max_bytes=4096)
    absent = dict(reference, source_posterior_uid="unselected", path=str(original / "do-not-read.gz"))
    teachers = [{"sample_uid": "selected", "posterior_uid": "complete", "trajectory_store_path": str(store),
                 "trajectory_index": index, "store_format": store_format} for index in (1, 2)]
    registry = {"posterior_family": "finite_joint_particles", "grid": {}, "teachers": teachers,
                "joint_source_references": [reference, absent]}
    registry_path = original / "registry.json"
    registry_path.write_text(json.dumps(registry))
    arrays = {}
    selected_chunks = set()
    for name in ("trajectories", "teacher_uncertainty", "teacher_confidence"):
        metadata = {"shape": [6, 2], "dtype": "<f4", "chunks": [2, 1], "chunk_size": 2,
                    "relative_root": "arrays/" + name}
        arrays[name] = metadata
        paths = array_input_paths(store, name, metadata, store_format, [1, 2, 5])
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if store_format == "neuronsifter_chunk_store":
                np.save(path, np.zeros((2, 2), dtype=np.float32))
            else:
                np.zeros((2, 1), dtype=np.float32).tofile(path)
        selected_chunks.update(map(str, array_input_paths(store, name, metadata, store_format, [1, 2])))
        if store_format == "zarr_v2":
            (store / name / ".zarray").write_text(json.dumps(metadata))
    stored = dict(registry, joint_source_references=[descriptor_reference, absent])
    descriptor = store / ("store_manifest.json" if store_format == "neuronsifter_chunk_store" else ".zattrs")
    descriptor.write_text(json.dumps({"metadata": {"native_diffusion": stored}, "arrays": arrays}))
    config = {"native_diffusion_data": {"registry_path": str(registry_path)},
              "screening_pipeline": {"screening_benchmark": {"model_architecture": {
                  "rollout_operator": {"transition_backbone": {"kind": "neuronsifter_world_action"}}}}}}
    child = SimpleNamespace(native_registry=registry, native_records=[{"teacher": row} for row in teachers])
    dataset = SimpleNamespace(split_name="train", group_uids=["selected"], input_locations=None, groups=[(child, [0, 1], None)])
    expected_files = {str(registry_path), reference["path"], descriptor_reference["path"],
                      str(descriptor), *selected_chunks}
    if store_format == "zarr_v2":
        expected_files.update(str(store / name / ".zarray") for name in arrays)
    return SimpleNamespace(config=config, dataset=dataset, files=expected_files,
                           reference=reference, descriptor_reference=descriptor_reference,
                           absent=absent, raw=raw, chunks=selected_chunks, original=original)


@pytest.mark.parametrize("store_format", ["neuronsifter_chunk_store", "zarr_v2"])
@pytest.mark.parametrize("copy_state", ["success", "insufficient_capacity", "copy_failure"])
def test_inventory_includes_descriptor_references_and_exact_selected_chunks(tmp_path, store_format, copy_state, monkeypatch):
    case = inventory_case(tmp_path, store_format)
    config, dataset, expected_files = case.config, case.dataset, case.files
    reference, descriptor_reference = case.reference, case.descriptor_reference
    absent, raw, selected_chunks, original = case.absent, case.raw, case.chunks, case.original
    required = sum(Path(name).stat().st_size for name in expected_files)
    free = required - 1 if copy_state == "insufficient_capacity" else required
    monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    destination = tmp_path / "staged"
    original_open = Path.open
    writes = []
    def payload_open(path, mode="r", *args, **kwargs):
        if mode == "xb" and destination in path.parents:
            writes.append(path)
            if copy_state == "copy_failure" and len(writes) == 2:
                raise OSError("injected failure after one copied payload")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", payload_open)
    if copy_state != "success":
        message = (f"required_bytes={required}, free_bytes={free}" if copy_state == "insufficient_capacity"
                   else "injected failure")
        with pytest.raises(OSError, match=message):
            stage_native_inputs(config, [dataset], destination)
        assert len(writes) == (0 if copy_state == "insufficient_capacity" else 2)
        assert not destination.exists()
        assert all(Path(name).is_file() for name in expected_files)
        return
    manifest = stage_native_inputs(config, [dataset], destination)
    mapping = NativeInputLocations.load(manifest)
    assert set(mapping.files) == expected_files
    assert sum(row["size_bytes"] for row in mapping.files.values()) == required
    assert reference["path"] in mapping.files and descriptor_reference["path"] in mapping.files
    assert absent["path"] not in mapping.files
    actual_chunks = {name for name in mapping.files if "chunk_" in name or Path(name).name[0].isdigit()}
    assert actual_chunks == selected_chunks
    with forbid_original_access([original]):
        assert read_joint_source_reference(reference, input_locations=mapping) == (raw, {reference["path"]: reference["sha256"]})
    local = mapping.file_path(reference["path"])
    data = local.read_bytes()
    changed = bytearray(data); changed[-10] ^= 1; local.write_bytes(changed)
    with forbid_original_access([original]), pytest.raises(ValueError, match="digest changed"):
        read_joint_source_reference(reference, input_locations=mapping)


def test_explicit_evaluation_stage_cannot_load_training(retained):
    config, _, root, original_root, config_path = retained
    manifest = invoke_staging(config_path, root / "validation", "--native-input-role", "validation")
    mapped = copy.deepcopy(config)
    mapped["native_input_staging"] = {"manifest_path": str(manifest)}
    locations = NativeInputLocations.load(manifest)
    assert set(locations.groups) == {"validation"}
    with forbid_original_access([original_root]):
        dataset = pretrain.NativeJointGroupDataset(mapped, "validation")
        assert len(dataset) == 4 and len(dataset[0]["native_joint_group"]["items"]) == 3
        with pytest.raises(ValueError, match="outside its staged input set"):
            pretrain.NativeJointGroupDataset(mapped, "train")


def test_native_comparator_cli_uses_runtime_manifest(retained, capsys):
    from Libs.exp.baselines import run_external_baseline_bridge as bridge
    _, mapped, root, original_root, config_path = retained
    # The command config itself is startup input; place its unchanged bytes outside the data-read guard.
    local_config = root / "comparator-config.yaml"
    local_config.write_bytes(config_path.read_bytes())
    arguments = ["run_external_baseline_bridge.py", "--config", str(local_config),
        "--protocol-scope", "manuscript-native", "--baseline-name", "native_wam",
        "--command", "prepare_inputs", "--output-dir", str(root / "comparator"),
        "--native-input-manifest", mapped["native_input_staging"]["manifest_path"]]
    with patch("sys.argv", arguments), forbid_original_access([original_root]):
        assert bridge.main() == 0
    assert '"training_groups": 4' in capsys.readouterr().out


def test_materialized_direct_conditioning_normalization_is_staged(retained):
    config, _, root, original_root, config_path = retained
    direct = copy.deepcopy(config)
    architecture = direct["screening_pipeline"]["screening_benchmark"]["model_architecture"]
    template = copy.deepcopy(architecture["rollout_operator"]["transition_backbone"]["transolver_ablation_template"])
    template["native_direct_mean"].update(enabled=True, posterior_context_dim=12,
        conditioning_config_path=str(config_path), conditioning_checkpoint_path="unused-no-model-initialization")
    architecture["rollout_operator"]["transition_backbone"] = template
    direct["selected_experiment_family"] = "rollout_operator_transolver"
    direct_path = root / "direct-config.yaml"
    direct.pop("config_includes", None)
    direct_path.write_text(yaml.safe_dump(direct, sort_keys=False))
    manifest = invoke_staging(direct_path, root / "direct")
    expected = normalization_from_config(direct)
    direct["native_input_staging"] = {"manifest_path": str(manifest)}
    with forbid_original_access([original_root]):
        assert normalization_from_config(direct) == expected
        assert representation(direct) == representation(config)
    locations = NativeInputLocations.load(manifest)
    assert str(config_path) in locations.files
    local = locations.file_path(config_path)
    original = local.read_bytes()
    included = original + b"\nconfig_includes: [absent.yaml]\n"
    local.write_bytes(included)
    locations.files[str(config_path)] = {"size_bytes": len(included), "sha256": hashlib.sha256(included).hexdigest()}
    with forbid_original_access([original_root]), pytest.raises(ValueError, match="standalone materialized"):
        locations.conditioning_config(config_path)


def test_frozen_snapshot_cli_stages_original_registry_through_canonical_mapping(tmp_path):
    snapshot = os.environ.get("NRNSIFT_SNAPSHOT_CODE_ROOT")
    fixture = os.environ.get("NRNSIFT_RHS_FIXTURE")
    if not snapshot or not fixture:
        pytest.skip("Requires the fixed-commit snapshot and retained eight-group fixture")
    assert repo_root() == Path(snapshot).resolve()
    original_config = load_config(Path(fixture) / "config.yaml")
    original_registry = resolve_repo_path(original_config["native_diffusion_data"]["registry_path"])
    original_bytes = original_registry.read_bytes()
    literal = "Output/results/manuscript_native_corpus/physical_parameter_conditioning/model_registry.json"
    mapping = repo_root() / Path(literal).parent
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.symlink_to(original_registry.parent, target_is_directory=True)
    assert resolve_repo_path(literal) == original_registry
    config = copy.deepcopy(original_config)
    config["native_diffusion_data"]["registry_path"] = literal
    config.pop("config_includes", None)
    path = tmp_path / "retained_mapping_profile.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    def forbidden(*args, **kwargs):
        raise AssertionError("Snapshot mapping cannot generate sources, initialize models or fit")
    with patch.object(source, "draw_group", side_effect=forbidden) as source_guard, \
            patch.object(torch.nn.Module, "__init__", side_effect=forbidden) as model_guard, \
            patch.object(pretrain, "fit_scientific_native_groups", side_effect=forbidden) as fit_guard:
        manifest = invoke_staging(path, tmp_path / "mapped_inputs", "--native-input-role", "train",
                                  "--native-input-role", "development")
    assert source_guard.call_count == model_guard.call_count == fit_guard.call_count == 0
    locations = NativeInputLocations.load(manifest)
    assert {role: len(groups) for role, groups in locations.groups.items()} == {"train": 4, "development": 4}
    assert locations.logical_path(literal) == locations.logical_path(original_registry) == original_registry
    assert locations.read_bytes(literal) == original_bytes
    assert original_registry.read_bytes() == original_bytes
    report = os.environ.get("NRNSIFT_SNAPSHOT_MAPPING_RESULT")
    if report:
        Path(report).write_text(json.dumps({"status": "PASS", "snapshot": str(repo_root()),
            "registry_literal": literal, "original_registry": str(original_registry),
            "canonical_mapping": str(mapping), "manifest": str(manifest),
            "groups": {role: len(groups) for role, groups in locations.groups.items()},
            "source_generation": source_guard.call_count, "model_initializations": model_guard.call_count,
            "scientific_fits": fit_guard.call_count,
            "scope": "retained_eight_group_software_source_root_mapping_only"}, indent=2) + "\n")


@pytest.mark.parametrize("store_format", ["neuronsifter_chunk_store", "zarr_v2"])
@pytest.mark.parametrize("failure", [None, "original", "staged", "missing", "role"])
def test_reuse_keeps_selected_identity_and_copies_only_missing_normalizer(tmp_path, store_format, failure, monkeypatch):
    case = inventory_case(tmp_path, store_format)
    prior_manifest = stage_native_inputs(case.config, [case.dataset], tmp_path / "prior")
    prior = NativeInputLocations.load(prior_manifest)
    chunk = Path(sorted(case.chunks)[0])
    target = prior.file_path(chunk) if failure in {"staged", "missing"} else chunk
    if failure in {"original", "staged"}:
        data = bytearray(target.read_bytes())
        data[-1] ^= 1
        target.write_bytes(data)
    elif failure == "missing":
        target.unlink()
    elif failure == "role":
        case.dataset.split_name = "development"
    normalizer = case.original / "scoped-normalizer.json"
    normalizer.write_text('{"purpose": "location-binding-test"}')
    case.config["native_diffusion_data"]["field_normalization"] = {"model_artifact_ref": str(normalizer)}
    before = {path: path.read_bytes() for path in [prior_manifest, *map(Path, case.files), normalizer]}
    # Existing payloads need no second allocation; only the selected extra file does.
    monkeypatch.setattr(staging.shutil, "disk_usage", lambda path: SimpleNamespace(free=normalizer.stat().st_size))
    destination = tmp_path / "comparison"
    errors = {"original": (ValueError, "Original admitted file differs"),
              "staged": (ValueError, "Reused native file bytes changed"),
              "missing": (FileNotFoundError, None),
              "role": (ValueError, "outside its staged input set")}
    if failure is not None:
        kind, message = errors[failure]
        with pytest.raises(kind, match=message):
            stage_native_inputs(case.config, [case.dataset], destination, reuse_manifest=prior_manifest)
        assert not destination.exists()
        assert all(path.read_bytes() == data for path, data in before.items())
        return
    manifest = stage_native_inputs(case.config, [case.dataset], destination, reuse_manifest=prior_manifest)
    mapped = NativeInputLocations.load(manifest)
    receipt = json.loads(manifest.read_text())
    assert mapped.groups == {"train": ["selected"]}
    assert set(mapped.files) == case.files | {str(normalizer)}
    assert receipt["reused_file_count"] == len(case.files)
    assert receipt["copied_bytes"] == normalizer.stat().st_size
    assert receipt["size_bytes"] == receipt["copied_bytes"] + receipt["reused_bytes"]
    for logical in case.files:
        assert mapped.file_path(logical).samefile(prior.file_path(logical))
        assert mapped.read_bytes(logical) == Path(logical).read_bytes()
    assert not mapped.file_path(normalizer).samefile(normalizer)
    with forbid_original_access([case.original]):
        assert mapped.read_bytes(normalizer) == before[normalizer]
        assert read_joint_source_reference(case.reference, input_locations=mapped)[0] == case.raw
    assert all(path.read_bytes() == data for path, data in before.items())


def test_reuse_cli_keeps_real_dataset_items_and_rejects_nonstaging_use(retained):
    config, staged, root, original_root, config_path = retained
    previous = staged["native_input_staging"]["manifest_path"]
    manifest = invoke_staging(config_path, root / "reused-cli", "--reuse-native-inputs", previous)
    reused = copy.deepcopy(config)
    reused["native_input_staging"] = {"manifest_path": str(manifest)}
    with forbid_original_access([original_root]):
        for role in ("train", "development"):
            expected = pretrain.NativeJointGroupDataset(staged, role)
            actual = pretrain.NativeJointGroupDataset(reused, role)
            assert actual.group_uids == expected.group_uids
            for index in range(len(expected)):
                assert_same(actual[index], expected[index])
    with patch("sys.argv", ["pretrain", "--config", str(config_path), "--reuse-native-inputs", previous]):
        with pytest.raises(ValueError, match="only to --stage-native-inputs"):
            pretrain.main()
