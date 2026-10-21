"""Retained complete-group continuation with artificial fields; no source trajectory."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import yaml

from Libs.scripts.builders import build_controlled_native_source as builder
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data import native_field_normalization as normalization
from Libs.utils.data.native_joint_source_storage import write_joint_source_reference
from Libs.utils.data.native_observable_profile import observable_profile, family_grid, quantity_support
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model import native_scientific_source as source
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store


@pytest.fixture
def retained(tmp_path, monkeypatch):
    config = load_config("Libs/configs/exp/native_observable_bridge.yaml")
    scientific = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_diffusion_data"]["group_storage"] = copy.deepcopy(scientific["native_diffusion_data"]["group_storage"])
    settings = config["native_observable_bridge"]
    root = tmp_path / "retained"
    settings.update(scientific_profile=True, source_uid="retained_artificial_fields",
                    snapshot_ref="retained_artificial_fields_definition", output_root=str(root))
    settings["atoms"] = [[0,0,0,0,0], [0,1,1,0,1], [1,0,1,1,0], [1,1,0,1,1]]
    settings["prior_weights"] = [.1,.2,.3,.4]
    settings["action_variants"] = [
        {"intensity_multiplier":.5,"duration_multiplier":1.,"response_delay_steps":1},
        {"intensity_multiplier":1.,"duration_multiplier":1.,"response_delay_steps":1}]
    settings["group_partitions"] = [
        {"partition_uid":"train","role":"train","groups_per_family":1,"action_indices":[0,1]},
        {"partition_uid":"development","role":"development","groups_per_family":1,"action_indices":[0,1]},
        {"partition_uid":"calibration_fit","role":"calibration","calibration_reservation":"fit",
         "groups_per_family":1,"action_indices":[0,1]},
        {"partition_uid":"calibration_check","role":"calibration","calibration_reservation":"check",
         "groups_per_family":1,"action_indices":[0,1]},
        {"partition_uid":"validation","role":"validation","groups_per_family":1,"action_indices":[0,1]}]
    settings["budget"].update(source.planned_operation_counts(config))
    groups = source.group_definitions(config)
    identity = {"producer_commit":"artificial_fixture", "definition":"artificial_fields_no_trajectory"}
    registries = {family:{} for family in observable_profile(config)["families"]}
    cases = []
    root.mkdir()
    for gi, group in enumerate(groups):
        # Only prior initial fields and lawful program metadata are constructed.
        # No physical trajectory, integration, source ledger or hidden outcome is used.
        generated = source.draw_group(config, group, gi)
        builder.write_json(root/f"group_{gi}_producer.json", {"parameters": generated["parameters"]})
        actions = [source.action_definition(config, group["family_uid"], ai, group) for ai in (0,1)]
        programs = [[source.compiled_program(config,group,generated["support"],atom,
            generated["parameters"][atom],action,"a"*64,
            source.estimated_exposure(config,generated["parameters"][atom],action)
            if group["family_uid"] == source.MOLECULAR else None) for action in actions] for atom in range(4)]
        raw = source.joint_source(config, group, generated, programs)
        native.parse_joint_source(raw)
        family = group["family_uid"]
        grid = family_grid(observable_profile(config), family)
        group_root = root/family/"groups"/str(gi)
        store = group_root/"native_source.zarr"
        reference = write_joint_source_reference(group_root/"conditioning.json.gz",raw,
            max_bytes=config["native_diffusion_data"]["group_storage"]["compressed_group_max_bytes"][family])
        group["conditioning_reference"] = reference
        group.setdefault("source_uid", settings["source_uid"])
        group.setdefault("source_snapshot_ref", settings["snapshot_ref"])
        teachers, fields = [], []
        support = quantity_support(grid, group["source_uid"])
        for ai, program in enumerate(programs[0]):
            initial = np.asarray(raw["support"]["initial_field"])[0].mean(0)
            field = np.broadcast_to(initial,(len(grid["time_hours"]),*initial.shape)).copy()
            field = np.where(support, field + np.arange(len(field))[:,None,None,None]*(.001+ai*.001)+gi*.00001,0).astype(np.float32)
            case_uid = f"artificial_case_{gi}_{ai}"
            teacher = {"trajectory_uid":case_uid,"source_uid":group["source_uid"],"sample_uid":group["sample_uid"],
                "family_uid":group["group_uid"],"content_hash":hashlib.sha256(field.tobytes()).hexdigest(),
                "trajectory_store_path":str(store),"store_format":"zarr_v2","trajectory_index":ai,
                "teacher_family":"full_microenvironment_teacher","qc_catalog_ref":str(root/"simulation_qc_catalog.tsv"),
                "source_snapshot_ref":group["source_snapshot_ref"],"uncertainty_origin":"independent_supplied",
                "teacher_confidence":.5,"qc_grade":"B","qc_use":"main_pretraining","sample_weight":.5,
                "split_name":group["role"],"source_holdout_role":group["role"],
                "validity_mask":{"kind":"all_valid","evidence_ref":str(root/"source_receipt.json")},
                "calibration_reservation":group.get("calibration_reservation"),
                **{key:program[key] for key in ("regimen_uid","action_uid","target_program_uid","program_uid","posterior_uid")}}
            teachers.append(teacher); fields.append(field)
            cases.append({"case_uid":case_uid,"group_index":gi,"action_index":ai,"status":"complete",
                          "numerical":{"reasons":[],"uncertainty":[.1]*26}})
        source_row = {"source_uid":group["source_uid"],"public":True,"protected":False,"license":"artificial_fixture",
            "release":"artificial_v1","snapshot_ref":group["source_snapshot_ref"],
            "split_evidence_ref":str(root/"source_receipt.json"),"unit_evidence_ref":"artificial_fixture_units",
            "source_kind":"controlled_simulated",
            "predictive_evaluation_units":{"episode_field":"sample_uid","group_field":"family_uid",
                "group_kind":"action_family","independent_episode_units":True,"evidence_ref":str(root/"source_receipt.json")}}
        registry = {"schema_version":native.SCHEMA_VERSION,"grid":grid,"sources":[source_row],
            "posterior_family":"finite_joint_particles","joint_sources":[],"joint_source_references":[reference],
            "teachers":teachers,"programs":[],"observations":[]}
        arrays = {"trajectory":np.stack(fields),"uncertainty":np.full((2,26),.1,np.float32),
                  "teacher_confidence":np.full(2,.5,np.float32),"quantity_support":np.stack([support]*2)}
        write_native_microenvironment_store(store,config,registry,arrays)
        builder.merge_scientific_registry(registries[family],registry)
    refs = {}
    for family, registry in registries.items():
        path = root/family/"native_registry.json"
        builder.write_json(path,registry); refs[family] = str(path)
    builder.write_json(root/"native_registry.json",builder.scientific_index(config,refs))
    config["native_diffusion_data"]["registry_path"] = str(root/"native_registry.json")
    batches = []
    for registry in registries.values():
        admission = native.NativeRegistryAdmission(registry,config)
        for group in admission.groups("train"):
            records, arrays, _ = admission.load(group)
            batches.append((registry["grid"], records, arrays))
    artifact = normalization.fit_multigrid(iter(batches),config,streaming=True)
    norm_path = root/"train_normalization.json"
    builder.write_json(norm_path,artifact)
    norm_digest = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    receipt = {"source_status":"complete","source_identity":identity,"groups":groups,"cases":cases}
    builder.write_json(root/"source_receipt.json",receipt)
    previous = {"status":"incomplete","declared_cases":len(cases),"admitted_count":0,"source_identity":identity,
        "cases":[{"case_uid":row["case_uid"],"role":groups[row["group_index"]]["role"],"reasons":[],
                  "source_status":"complete","native_status":"candidate"} for row in cases]}
    builder.write_json(root/"native_consumption.json",previous)
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    # Only the explicitly requested model-source scope fits may run later.
    # The frozen base TRAIN fit and every source producer remain prohibited.
    scope_fit = normalization.fit_multigrid
    def fit_declared_scope(batches, scoped, **kwargs):
        if scoped.get("native_scientific_training", {}).get("source_scope") not in {"pure_shared", "mixed_shared"}:
            pytest.fail("continuation refit the frozen base normalizer")
        return scope_fit(batches, scoped, **kwargs)
    monkeypatch.setattr(builder, "fit_multigrid", fit_declared_scope)
    for module, names in ((source,("draw_group","physical_trajectory","compiled_program","estimated_exposure","integrate_coupled")),
                          (builder,("execute_observable_source",)),
                          (normalization,("fit_multigrid","fit_transform"))):
        for name in names:
            if hasattr(module,name):
                monkeypatch.setattr(module,name,lambda *a,**k: pytest.fail("continuation called producer or fit"))
    monkeypatch.setattr(builder,"verify_observable_source",lambda actual,cfg: (
        actual == receipt and cfg["native_observable_bridge"] == config["native_observable_bridge"]) or
        pytest.fail("artificial fixture authority differs"))
    return config, path, root, registries, norm_digest


def bind_group(config, root, registry, monkeypatch):
    admission = native.NativeRegistryAdmission(registry,config)
    records, arrays, _ = admission.load(next(admission.groups()))
    store = Path(records[0]["teacher"]["trajectory_store_path"]).parent/"normalized.zarr"
    derived = admission.reindex(records,store,"zarr_v2")
    derived["grid"] = normalization.bind_artifact(registry["grid"],root/"train_normalization.json",config)
    write_native_microenvironment_store(store,config,derived,arrays)
    return store, derived, arrays


def test_retained_cli_reuses_complete_groups_writes_missing_and_preserves_raw_and_fit(retained,monkeypatch):
    config,path,root,registries,digest = retained
    registry = next(iter(registries.values()))
    store, derived, arrays = bind_group(config,root,registry,monkeypatch)
    before = {p:p.read_bytes() for p in (root/"source_receipt.json", root/"train_normalization.json",
                                       root/"native_registry.json",store/".zattrs")}
    old_report = (root/"native_consumption.json").read_bytes()
    validations = []
    validate = normalization.validate_artifact
    monkeypatch.setattr(normalization,"validate_artifact",lambda *a,**k: (validations.append(1),validate(*a,**k))[1])
    monkeypatch.setattr(sys,"argv",["builder","--config",str(path),"--continue-admission","--normalization-sha256",digest])
    assert builder.main() == 0
    report = json.loads((root/"native_consumption.json").read_text())
    assert report["admitted_count"] == 30 and report["status"] == "complete"
    assert report["reused_derived_groups"] == 1 and report["written_derived_groups"] == 14
    assert report["normalizer_fit"] == report["source_generation"] == "NOT_REPLAYED"
    assert len(validations) == 1
    assert Path(report["previous_failed_admission_ref"]).read_bytes() == old_report
    for p,data in before.items(): assert p.read_bytes() == data
    actual,_ = native.load_records(derived,native.join_registry(derived,config),config)
    for key in arrays: np.testing.assert_array_equal(actual[key],arrays[key])
    assert all(set(report["prepared_roles"][family]) == {"train","development","calibration","validation"}
               for family in registries)


@pytest.mark.parametrize("mutation",["digest","training_role","derived_values","partial_derived"])
def test_retained_rejects_changed_binding_or_invalid_derived_without_refit(retained,monkeypatch,mutation):
    config,path,root,registries,digest = retained
    if mutation == "digest":
        digest = "f"*64
    elif mutation == "training_role":
        norm = root/"train_normalization.json"
        artifact = json.loads(norm.read_text()); artifact["training_records"][0]["source_uid"] = "other_source"
        builder.write_json(norm,artifact); digest = hashlib.sha256(norm.read_bytes()).hexdigest()
    else:
        registry = next(iter(registries.values()))
        store,derived,arrays = bind_group(config,root,registry,monkeypatch)
        if mutation == "partial_derived": (store/".zattrs").unlink()
        else:
            arrays["trajectory"][0,1,0,0,0] += .01
            write_native_microenvironment_store(store,config,derived,arrays)
    with pytest.raises((ValueError,FileNotFoundError),match="artifact bytes|TRAIN|identity|support|incomplete|values differ"):
        builder.continue_scientific_admission(config,path,normalization_sha256=digest)


def test_context_actual_consumers_once_and_mutation_with_restored_stat_rejects(retained,monkeypatch):
    config,path,root,registries,digest = retained
    registry = next(iter(registries.values()))
    store,derived,arrays = bind_group(config,root,registry,monkeypatch)
    count = []
    validate = normalization.validate_artifact
    monkeypatch.setattr(normalization,"validate_artifact",lambda *a,**k: (count.append(1),validate(*a,**k))[1])
    started = time.monotonic()
    baseline = builder.scientific_group_checks(config,derived,store)
    baseline_seconds = time.monotonic()-started
    baseline_validations = len(count); count.clear()
    context = normalization.NativeNormalizationContext(config)
    started = time.monotonic()
    context.spec(derived["grid"],config)
    actual = builder.scientific_group_checks(config,derived,store,normalization_context=context)
    context.verify()
    context_seconds = time.monotonic()-started
    assert actual == baseline and baseline_validations > 1 and len(count) == 1
    altered = copy.deepcopy(config)
    altered["native_diffusion_data"]["species_units"] = {}
    with pytest.raises(ValueError,match="configuration changed"): context.spec(derived["grid"],altered)
    norm = root/"train_normalization.json"
    before = norm.stat(); data = norm.read_bytes()
    # Same-length trailing whitespace, restored mtime; ctime cannot be restored
    # by an ordinary caller. The final hash remains an independent byte check.
    norm.write_bytes(data[:-1]+b" ")
    os.utime(norm,ns=(before.st_atime_ns,before.st_mtime_ns))
    monkeypatch.setattr(context,"_stat",lambda p: next(iter(context._files.values()))[1])
    with pytest.raises(ValueError,match="artifact bytes changed"): context.verify()
    print(json.dumps({"baseline_seconds":baseline_seconds,"context_seconds":context_seconds,
                      "baseline_validations":baseline_validations,"context_validations":len(count),
                      "complete_rows":2,"scope":"same_artificial_complete_group_real_consumers"}))


@pytest.fixture
def named_retained(retained, monkeypatch):
    config, path, root, registries, digest = retained
    builder.continue_scientific_admission(config, path, normalization_sha256=digest)
    named = copy.deepcopy(config)
    named["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"][
        "physical_parameter_conditioning"] = "source_declared_named_candidates"
    named_root = root/"physical_parameter_conditioning"
    named["native_diffusion_data"]["registry_path"] = str(named_root/"model_registry.json")
    named.setdefault("native_scientific_training", {}).update(source_scope="all", training_subset_percentage=100)
    named.setdefault("manuscript_native_protocol", {})["source_comparison"] = {
        "normalization_root": str(named_root/"source_scope_normalization")}
    # This tiny input has no paired electrical strata. Both diagnostic scope
    # selections use its actual complete TRAIN groups; the production selector
    # and its source-matching behavior are outside this call-context regression.
    selections = []
    def fixture_scope_selection(scoped, role):
        assert role == "train"
        selections.append(scoped["native_scientific_training"]["source_scope"])
        return [row for row in source.group_definitions(config) if row["role"] == role]
    monkeypatch.setattr(source, "selected_group_definitions", fixture_scope_selection)
    named_path = path.parent/"named_config.yaml"
    named_path.write_text(yaml.safe_dump(named))
    return named, named_path, root, named_root, digest, selections


def test_named_cli_contexts_validate_once_and_preserve_base_with_declared_scope_fits(named_retained, monkeypatch):
    config, path, root, output, digest, selections = named_retained
    frozen = {p: p.read_bytes() for p in (root/"train_normalization.json", root/"native_consumption.json",
                                         root/"source_receipt.json")}
    validations, contexts = [], []
    validate = normalization.validate_artifact
    original_context = normalization.NativeNormalizationContext
    class ObservedContext(original_context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            contexts.append(self)
    monkeypatch.setattr(builder, "NativeNormalizationContext", ObservedContext)
    monkeypatch.setattr(normalization, "validate_artifact", lambda artifact, *a, **k: (
        validations.append(artifact["trajectory_count"]), validate(artifact, *a, **k))[1])
    monkeypatch.setattr(sys, "argv", ["builder", "--config", str(path), "--consume-only"])
    assert builder.main() == 0
    report = json.loads((output/"native_consumption.json").read_text())
    assert report["status"] == "complete" and report["admitted_count"] == report["declared_cases"] == 30
    assert report["array_values_unchanged"] and report["compiled_programs_unchanged"]
    assert report["source_operations"] == report["model_initializations"] == 0
    assert len(contexts) == 2 and all(len(context._specs) == 1 for context in contexts)
    # Two validations for the two actual scientific configurations. Each new
    # multigrid fit independently validates its three registered family grids.
    assert len(validations) == 2 + 2*3
    assert selections == ["pure_shared", "pure_shared", "mixed_shared", "mixed_shared"]
    for scope, details in report["source_scope_normalizers"].items():
        artifact = json.loads(Path(details["artifact_ref"]).read_text())
        assert artifact["training_data_scope"]["source_scope"] == scope
        assert artifact["source_grid_normalization_sha256"] == digest
        assert artifact["trajectory_count"] == details["training_trajectories"] == 6
        assert details["training_groups"] == 3 and details["source_grid_unchanged"]
    for file, data in frozen.items():
        assert file.read_bytes() == data
    read_config = copy.deepcopy(config)
    read_config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"].pop(
        "physical_parameter_conditioning")
    with pytest.raises(ValueError, match="configuration changed"):
        contexts[1].spec(next(iter(contexts[1]._specs.values()))["artifact"]["registered_grids"][0], read_config)


def test_named_rejects_changed_base_bytes_before_complete_receipt(named_retained, monkeypatch):
    config, path, root, output, digest, selections = named_retained
    contexts = []
    original_context = normalization.NativeNormalizationContext
    class ObservedContext(original_context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            contexts.append(self)
    monkeypatch.setattr(builder, "NativeNormalizationContext", ObservedContext)
    scope_fit = builder.build_source_scope_normalizations
    def mutate_after_scope_fit(*args, **kwargs):
        result = scope_fit(*args, **kwargs)
        artifact = root/"train_normalization.json"
        st, data = artifact.stat(), artifact.read_bytes()
        artifact.write_bytes(data[:-1]+b" ")
        os.utime(artifact, ns=(st.st_atime_ns, st.st_mtime_ns))
        # Restore the observation only. The final verification must independently
        # inspect actual bytes before the complete receipt can become authoritative.
        for context in contexts:
            expected = next(iter(context._files.values()))[1]
            monkeypatch.setattr(context, "_stat", lambda p, expected=expected: expected)
        return result
    monkeypatch.setattr(builder, "build_source_scope_normalizations", mutate_after_scope_fit)
    monkeypatch.setattr(sys, "argv", ["builder", "--config", str(path), "--consume-only"])
    with pytest.raises(ValueError, match="artifact bytes changed"):
        builder.main()
    report = json.loads((output/"native_consumption.json").read_text())
    assert report["status"] == "failed" and report["admitted_count"] == 0
    assert all(case["native_status"] == "consumer_failed" for case in report["cases"])
    assert "artifact bytes changed" in report["error"]
    assert len(contexts) == 2


def test_storage_budget_counts_nested_unexpected_files_and_symlinks(tmp_path):
    root = tmp_path / "retained"
    nested = root / "nested"
    nested.mkdir(parents=True)
    ordinary = root / "declared.bin"
    unexpected = nested / "unexpected.bin"
    ordinary.write_bytes(b"abc")
    unexpected.write_bytes(b"12345")
    outside = tmp_path / "outside"
    outside.mkdir()
    external_file = outside / "external.bin"
    external_file.write_bytes(b"1234567")
    file_link = root / "file-link"
    file_link.symlink_to(external_file)
    (root / "directory-link").symlink_to(outside, target_is_directory=True)
    (root / "dangling-link").symlink_to(tmp_path / "missing")

    budget = builder.NativeStorageBudget(100, 50)
    budget.record([root])

    special = root / "fifo"
    os.mkfifo(special)
    budget.record([root])
    assert budget.used == 15
    assert budget.files == {ordinary: 3, unexpected: 5, file_link: 7}


def test_storage_budget_missing_declared_path_raises(tmp_path):
    budget = builder.NativeStorageBudget(100, 50)
    missing = tmp_path / "missing"
    dangling = tmp_path / "dangling"
    dangling.symlink_to(missing)
    for path in (missing, dangling):
        with pytest.raises(FileNotFoundError, match="Declared native output is missing"):
            budget.record([path])
    assert budget.used == 0 and budget.files == {}


def test_storage_budget_repeated_paths_and_changed_sizes_charge_deltas(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()
    retained = root / "retained.bin"
    retained.write_bytes(b"abc")
    budget = builder.NativeStorageBudget(100, 50, initial_bytes=11)
    budget.record([root, retained, root], complete_group=True)
    assert budget.used == 14 and budget.files == {retained: 3}
    retained.write_bytes(b"1234567")
    budget.record([retained, root, retained], complete_group=True)
    assert budget.used == 18 and budget.files == {retained: 7}
    retained.write_bytes(b"x")
    budget.record([root, retained], complete_group=True)
    assert budget.used == 12 and budget.files == {retained: 1}


def test_storage_budget_declared_directory_symlink_and_special_file(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()
    retained = root / "retained.bin"
    retained.write_bytes(b"abc")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    special = tmp_path / "fifo"
    os.mkfifo(special)
    budget = builder.NativeStorageBudget(100, 50)
    budget.record([alias, special])
    assert budget.used == 3 and budget.files == {alias / "retained.bin": 3}


@pytest.mark.parametrize("ceiling", ["group", "total", "reserve"])
def test_storage_budget_byte_ceilings_reject(tmp_path, ceiling):
    retained = tmp_path / "retained.bin"
    retained.write_bytes(b"1234567")
    if ceiling == "group":
        budget = builder.NativeStorageBudget(100, 6)
        with pytest.raises(RuntimeError, match="Complete group exceeded"):
            budget.record([retained], complete_group=True)
    elif ceiling == "total":
        budget = builder.NativeStorageBudget(8, 8, terminal_reserve=2)
        with pytest.raises(RuntimeError, match="Native storage byte ceiling exceeded"):
            budget.record([retained])
    else:
        budget = builder.NativeStorageBudget(10, 6, terminal_reserve=2, initial_bytes=3)
        with pytest.raises(RuntimeError, match="cannot reserve the next complete group"):
            budget.reserve_group()


def test_storage_budget_exact_ceiling_accepts_and_duplicate_group_does_not_recharge(tmp_path):
    retained = tmp_path / "retained.bin"
    retained.write_bytes(b"1234567")
    budget = builder.NativeStorageBudget(10, 7, terminal_reserve=3)
    budget.reserve_group()
    budget.record([retained], complete_group=True)
    budget.record([retained, retained], complete_group=True)
    assert budget.used == 7 and budget.files == {retained: 7}
