"""Conditional zero proofs and recovery of failed source consumption."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from Libs.scripts.builders import build_controlled_native_source as builder
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store


def proof(quantity, reference="threshold-branch", snapshot="snapshot"):
    return {"status": "exact_by_definition", "derivation_kind": "identically_zero",
            "derivation_ref": reference, "source_snapshot_ref": snapshot,
            "serialization_error": 0., "quantity": quantity}


def receipt_fixture():
    return {"source_status": "complete", "source_identity": {"producer": "unchanged"},
            "groups": [{"family_uid": "electrical", "source_uid": "source",
                        "source_snapshot_ref": "snapshot"} for _ in range(2)],
            "cases": [{"group_index": 0, "numerical": {"exact_zero_uncertainty": {}}},
                      {"group_index": 1, "numerical": {"exact_zero_uncertainty": {
                          "spike_rate": proof("spike_rate")}}}]}


def test_conditional_proof_catalogue_is_order_independent_and_keeps_case_evidence():
    receipt = receipt_fixture()
    original = copy.deepcopy(receipt)
    expected = {("electrical", "source"): {"spike_rate": {
        "kind": "identically_zero", "derivation_ref": "threshold-branch"}}}
    assert builder.scientific_exact_zero_catalogue(receipt) == expected
    assert receipt == original
    receipt["cases"].reverse()
    assert builder.scientific_exact_zero_catalogue(receipt) == expected


@pytest.mark.parametrize("mutation", ["reference", "kind", "snapshot"])
def test_conflicting_or_unbound_proofs_are_rejected(mutation):
    receipt = receipt_fixture()
    receipt["cases"][0]["numerical"]["exact_zero_uncertainty"] = {
        "spike_rate": proof("spike_rate")}
    evidence = receipt["cases"][1]["numerical"]["exact_zero_uncertainty"]["spike_rate"]
    evidence[{"reference": "derivation_ref", "kind": "derivation_kind",
              "snapshot": "source_snapshot_ref"}[mutation]] = "conflicting"
    with pytest.raises(ValueError, match="Scientific exact-zero"):
        builder.scientific_exact_zero_catalogue(receipt)


@pytest.mark.parametrize("field", ["family_uid", "source_uid"])
def test_proof_catalogues_do_not_cross_sources_or_grids(field):
    receipt = receipt_fixture()
    receipt["groups"][1][field] = "another"
    result = builder.scientific_exact_zero_catalogue(receipt)
    assert result["electrical", "source"] == {}
    assert len(result) == 2


def recovery_fixture(tmp_path, monkeypatch):
    receipt = receipt_fixture()
    config = {"native_observable_bridge": {"scientific_profile": True}}
    monkeypatch.setattr(builder, "observable_profile", lambda _: {"families": {"electrical": {}}})
    previous = {"status": "incomplete", "admitted_count": 0, "declared_cases": 2,
                "source_identity": receipt["source_identity"], "consumer_error": "failed metadata merge"}
    (tmp_path/"native_consumption.json").write_text(json.dumps(previous))
    (tmp_path/"source_receipt.json").write_bytes(b"immutable original receipt")
    (tmp_path/"group_0_conditioning.json.gz").write_bytes(b"immutable original observations")
    (tmp_path/"electrical").mkdir()
    (tmp_path/"electrical"/"failed-store").write_bytes(b"retained failed output")
    return config, receipt, previous


def test_recovery_preserves_failed_outputs_and_does_not_move_raw_source(tmp_path, monkeypatch):
    config, receipt, previous = recovery_fixture(tmp_path, monkeypatch)
    archive = builder.archive_failed_scientific_admission(config, receipt, tmp_path)
    assert json.loads((archive/"native_consumption.json").read_text()) == previous
    assert (archive/"electrical"/"failed-store").read_bytes() == b"retained failed output"
    assert (tmp_path/"source_receipt.json").read_bytes() == b"immutable original receipt"
    assert (tmp_path/"group_0_conditioning.json.gz").read_bytes() == b"immutable original observations"
    assert not (tmp_path/"electrical").exists()


@pytest.mark.parametrize("mutation", ["complete_admission", "other_source", "partial_source"])
def test_recovery_rejects_invalid_prior_state_without_writes(tmp_path, monkeypatch, mutation):
    config, receipt, previous = recovery_fixture(tmp_path, monkeypatch)
    if mutation == "complete_admission":
        previous.update(status="complete", admitted_count=2)
    elif mutation == "other_source":
        previous["source_identity"] = {"producer": "different"}
    else:
        receipt["source_status"] = "partial"
    path = tmp_path/"native_consumption.json"
    path.write_text(json.dumps(previous))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="recovery requires"):
        builder.archive_failed_scientific_admission(config, receipt, tmp_path)
    assert path.read_bytes() == before and (tmp_path/"electrical"/"failed-store").is_file()
    assert not (tmp_path/"admission_failures").exists()


def test_recovery_cli_never_dispatches_generation(monkeypatch):
    config = {"native_observable_bridge": {"scientific_profile": True}}
    monkeypatch.setattr(builder, "load_config", lambda _: config)
    monkeypatch.setattr(builder, "resolve_repo_path", Path)
    monkeypatch.setattr(builder, "parameter_schema_from_config", lambda _: None)
    monkeypatch.setattr(builder, "execute_observable_source", lambda *a, **k: pytest.fail("source replay"))
    calls = []
    def consume(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "complete"}
    monkeypatch.setattr(builder, "consume_observable_source", consume)
    monkeypatch.setattr(builder.sys, "argv", ["builder", "--config", "fixture.yaml", "--recover-admission"])
    assert builder.main() == 0 and calls == [{"recover_admission": True}]


def test_retained_zero_and_positive_groups_pass_writer_merge_and_loader(tmp_path):
    location = os.environ.get("NRNSIFT_RETAINED_SOURCE_ROOT")
    if not location:
        pytest.skip("requires the retained source groups 0 and 4")
    root = Path(location)
    config = load_config("Libs/configs/exp/manuscript_native_source.yaml")
    receipt = {"groups": [], "cases": []}
    registries = []
    for index in (0, 4):
        group_receipt = json.loads((root/f"group_{index}_receipt.json").read_text())
        group = group_receipt["group"]
        gi = len(receipt["groups"])
        receipt["groups"].append(group)
        receipt["cases"].extend(dict(case, group_index=gi) for case in group_receipt["cases"])
        descriptor = root/group["family_uid"]/"groups"/str(index)/"native_source.zarr"/".zattrs"
        metadata = json.loads(descriptor.read_text())
        registries.append(metadata.get("metadata", metadata)["native_diffusion"])
    catalogue = builder.scientific_exact_zero_catalogue(receipt)
    combined, original_arrays, original_teachers = {}, [], []
    for index, registry in enumerate(registries):
        records = native.join_registry(registry, config)
        arrays, _ = native.load_records(registry, records, config)
        store = tmp_path/str(index)/"native_source.zarr"
        derived = native.reindex_registry(registry, records, store, "zarr_v2")
        for source in derived["sources"]:
            source["exact_zero_quantities"] = catalogue[derived["grid"]["family_uid"], source["source_uid"]]
        original_teachers.extend(copy.deepcopy(derived["teachers"]))
        write_native_microenvironment_store(store, config, derived, arrays)
        builder.merge_scientific_registry(combined, derived)
        original_arrays.append(arrays)
    loaded, _ = native.load_records(combined, native.join_registry(combined, config), config)
    assert combined["teachers"] == original_teachers
    for name in loaded:
        assert np.array_equal(loaded[name], np.concatenate([arrays[name] for arrays in original_arrays]))
    assert len(combined["sources"]) == 1
    assert "electrical_spike_rate" in combined["sources"][0]["exact_zero_quantities"]
    assert any("electrical_spike_rate" not in teacher["exact_zero_uncertainty"] for teacher in combined["teachers"])
