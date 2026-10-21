"""Call-scoped metadata reuse on retained artificial complete groups."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from Libs.scripts.builders import build_controlled_native_source as builder
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_field_normalization import fit_multigrid
from Libs.utils.data.raw_data_registry import load_config


@pytest.fixture(scope="module")
def retained():
    location = os.environ.get("NRNSIFT_ADMISSION_FIXTURE")
    if not location:
        pytest.skip("requires retained artificial complete-group fixture")
    root = Path(location)
    declaration = json.loads((root / "software_fixture.json").read_text())
    assert declaration["scope"] == "artificial_software_fixture_no_source_trajectory"
    assert declaration["physical_source_trajectory_calls"] == 0
    config = load_config(root / "config.yaml")
    registries = native.load_family_registries(config)
    return config, registries


def exact_arrays(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name].dtype == expected[name].dtype
        assert np.array_equal(actual[name], expected[name]), name


def test_retained_complete_groups_records_arrays_bindings_reindex_and_normalizer_exact(retained, tmp_path):
    config, registries = retained
    legacy_batches, indexed_batches = [], []
    for family, registry in registries.items():
        original = copy.deepcopy(registry)
        admission = native.NativeRegistryAdmission(registry, config)
        expected_groups = list(builder.scientific_group_rows(registry))
        groups = list(admission.groups())
        assert [[registry["teachers"][i] for i in group] for group in groups] == expected_groups
        for group, teachers in zip(groups, expected_groups):
            expected = native.join_registry(registry, config, teachers)
            expected_arrays, expected_bindings = native.load_records(registry, expected, config)
            actual, arrays, bindings = admission.load(group)
            assert actual == expected
            exact_arrays(arrays, expected_arrays)
            assert bindings == expected_bindings
            target = tmp_path / family / str(group[0])
            derived = admission.reindex(actual, target, "zarr_v2")
            assert derived == native.reindex_registry(registry, expected, target, "zarr_v2")
            assert derived["sources"] == registry["sources"]
            if teachers[0]["split_name"] == "train":
                grid = copy.deepcopy(registry["grid"])
                grid["normalization"] = {"kind": "physical"}
                legacy_batches.append((grid, expected, expected_arrays))
                indexed_batches.append((copy.deepcopy(grid), actual, arrays))
        assert registry == original
    before = copy.deepcopy(config)
    expected = fit_multigrid(iter(legacy_batches), config, streaming=True)
    actual = fit_multigrid(iter(indexed_batches), config, streaming=True)
    assert actual == expected
    assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    assert config == before


@pytest.mark.parametrize("mutation", ["source", "reference", "inline", "authority", "teacher"])
def test_out_of_group_duplicates_and_conflicts_reject_before_payload_reads(retained, monkeypatch, mutation):
    config, registries = retained
    registry = copy.deepcopy(next(iter(registries.values())))
    uid = "unselected-posterior"
    if mutation == "source":
        row = dict(registry["sources"][0], source_uid="unselected-source")
        registry["sources"].extend([row, copy.deepcopy(row)])
    elif mutation == "teacher":
        row = dict(registry["teachers"][-1], trajectory_uid="unselected-trajectory")
        registry["teachers"].extend([row, copy.deepcopy(row)])
    else:
        reference = dict(registry["joint_source_references"][0], source_posterior_uid=uid)
        inline = {"source_posterior_uid": uid}
        if mutation == "reference":
            registry["joint_source_references"].extend([reference, copy.deepcopy(reference)])
        elif mutation == "inline":
            registry["joint_sources"].extend([inline, copy.deepcopy(inline)])
        else:
            registry["joint_source_references"].append(reference)
            registry["joint_sources"].append(inline)
    def no_payload(*args, **kwargs):
        raise AssertionError("conflicting complete metadata reached a payload")
    monkeypatch.setattr(native, "read_joint_source_reference", no_payload)
    with pytest.raises(native.NativeDataError, match="duplicate|both inline"):
        native.NativeRegistryAdmission(registry, config)
    if mutation != "teacher":
        with pytest.raises(native.NativeDataError, match="duplicate|both inline"):
            native.join_registry(registry, config, list(builder.scientific_group_rows(registry))[0])


def test_snapshot_group_authority_and_fresh_call_reject_registry_mutation(retained):
    config, registries = retained
    registry = copy.deepcopy(next(iter(registries.values())))
    admission = native.NativeRegistryAdmission(registry, config)
    group = next(admission.groups("train"))
    expected = admission.join(group)
    registry["teachers"][group[0]]["teacher_confidence"] = "invalid"
    registry["joint_source_references"].append(copy.deepcopy(registry["joint_source_references"][0]))
    assert admission.join(group) == expected
    with pytest.raises(native.NativeDataError, match="duplicate"):
        native.NativeRegistryAdmission(registry, config)
    expected[0]["teacher"]["teacher_confidence"] = "invalid"
    expected[0]["joint_source"]["support"]["role"] = "invalid"
    assert admission.join(group)[0]["teacher"]["teacher_confidence"] != "invalid"
    with pytest.raises(native.NativeDataError, match="snapshot"):
        admission.reindex(expected, Path("unused-output"), "zarr_v2")
    for supplied in ([0], (group[0],), tuple(reversed(group))):
        with pytest.raises(native.NativeDataError, match="snapshot-owned"):
            admission.join(supplied)
    uid = admission.join(group)[0]["teacher"]["sample_uid"]
    assert list(admission.groups("train", sample_uids={uid})) == [group]
    assert list(admission.groups("train", sample_uids={"absent"})) == []


def test_inline_public_loading_and_admission_records_are_exact_and_detached(retained):
    config, registries = retained
    registry = copy.deepcopy(next(iter(registries.values())))
    registry["joint_sources"] = list(native.load_joint_sources(registry).values())
    registry["joint_source_references"] = []
    original = copy.deepcopy(registry)
    admission = native.NativeRegistryAdmission(registry, config)
    group = next(admission.groups())
    actual = admission.join(group)
    expected = native.join_registry(registry, config, [registry["teachers"][i] for i in group])
    assert actual == expected
    uid = actual[0]["teacher"]["posterior_uid"]
    assert native.load_joint_sources(registry, [uid, uid])[uid] is registry["joint_sources"][0]
    target = Path("unused-inline-output")
    derived = admission.reindex(actual, target, "zarr_v2")
    assert derived == native.reindex_registry(registry, expected, target, "zarr_v2")
    assert derived["sources"] == original["sources"]
    teacher_bytes = json.dumps(actual[0]["teacher"], sort_keys=True)
    actual[0]["joint_source"]["support"]["role"] = "invalid"
    assert json.dumps(actual[0]["teacher"], sort_keys=True) == teacher_bytes
    with pytest.raises(native.NativeDataError, match="inline joint source differs"):
        admission.reindex(actual, target, "zarr_v2")
    assert admission.join(group) == expected
    assert registry == original


def test_joint_metadata_entry_visits_grow_linearly_with_complete_group_count(retained, monkeypatch):
    config, registries = retained
    template = next(iter(registries.values()))
    uid = template["teachers"][0]["posterior_uid"]
    raw = {"source_posterior_uid": uid}
    counts = {"source": 0, "joint source": 0, "joint source reference": 0}
    original_unique = native._unique
    def counted(rows, fields, label):
        if label in counts:
            counts[label] += len(rows)
        return original_unique(rows, fields, label)
    monkeypatch.setattr(native, "_unique", counted)
    monkeypatch.setattr(native, "validate_grid", lambda *args, **kwargs: None)
    def metadata_only(registry, config, selected, sources, *, input_locations=None, joint_sources=None):
        # Execute the real source-index/load path while excluding numeric parsing.
        if joint_sources is None:
            joint_sources = native.load_joint_sources(registry, [row["posterior_uid"] for row in selected])
        assert uid in joint_sources
        return [{"teacher": row} for row in selected]
    monkeypatch.setattr(native, "_join_joint_registry", metadata_only)
    measurements = []
    for size in (32, 128, 512):
        registry = copy.deepcopy(template)
        registry.update(joint_sources=[raw] + [{"source_posterior_uid": "unselected-" + str(i)} for i in range(size-1)],
                        joint_source_references=[], teachers=[])
        for index in range(size):
            teacher = dict(template["teachers"][0], trajectory_uid="trajectory-"+str(index),
                           sample_uid="sample-"+str(index), trajectory_index=index, posterior_uid=uid)
            registry["teachers"].append(teacher)
        counts.update(dict.fromkeys(counts, 0))
        start = time.perf_counter()
        for teachers in builder.scientific_group_rows(registry):
            native.join_registry(registry, config, teachers)
        legacy_seconds = time.perf_counter() - start
        legacy_visits = dict(counts)
        counts.update(dict.fromkeys(counts, 0))
        start = time.perf_counter()
        admission = native.NativeRegistryAdmission(registry, config)
        for group in admission.groups():
            admission.join(group)
        indexed_seconds = time.perf_counter() - start
        indexed_visits = dict(counts)
        assert legacy_visits["joint source"] == size * size
        assert indexed_visits["joint source"] == size
        assert legacy_visits["source"] == size * len(template["sources"])
        assert indexed_visits["source"] == len(template["sources"])
        assert max(legacy_seconds, indexed_seconds) < 60
        measurements.append({"groups": size, "legacy_visits": legacy_visits,
                             "indexed_visits": indexed_visits, "legacy_seconds": legacy_seconds,
                             "indexed_seconds": indexed_seconds})
    print("metadata_work=" + json.dumps(measurements), flush=True)
