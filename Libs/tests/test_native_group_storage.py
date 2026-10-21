"""Complete-group lossless storage, bounded selection and fresh identity checks."""
import copy

import pytest

from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_joint_source_storage import read_joint_source_reference, write_joint_source_reference


def test_gzip_group_roundtrip_and_size_cap(tmp_path):
    raw = {"source_posterior_uid": "disposable:posterior", "values": [[0., -1.25, 1e-12], [1., 2., 3.]]}
    path = tmp_path/"complete.json.gz"
    reference = write_joint_source_reference(path, raw, max_bytes=4096)
    restored, bindings = read_joint_source_reference(reference)
    assert restored == raw and bindings == {str(path): reference["sha256"]}
    assert reference["size_bytes"] == path.stat().st_size
    with pytest.raises(ValueError, match="byte ceiling"):
        write_joint_source_reference(tmp_path/"too-small.json.gz", raw, max_bytes=1)
    assert not (tmp_path/"too-small.json.gz").exists()


@pytest.mark.parametrize("mutation", ["bytes", "identity", "expanded_size"])
def test_gzip_group_rejects_changed_payload_or_reference(tmp_path, mutation):
    raw = {"source_posterior_uid": "disposable:posterior", "values": [1., 2.]}
    path = tmp_path/"complete.json.gz"
    reference = write_joint_source_reference(path, raw, max_bytes=4096)
    read_joint_source_reference(reference)
    if mutation == "bytes":
        payload = bytearray(path.read_bytes()); payload[-10] ^= 1; path.write_bytes(payload)
    elif mutation == "identity":
        reference["source_posterior_uid"] = "another-posterior"
    else:
        reference["uncompressed_size_bytes"] += 1
    with pytest.raises(ValueError, match="changed"):
        read_joint_source_reference(reference)


def test_group_resolution_reads_selected_payload_only_and_keeps_inline(tmp_path, monkeypatch):
    raw = {"source_posterior_uid": "complete-a", "values": [[1., 2.]]}
    reference = write_joint_source_reference(tmp_path/"a.json.gz", raw, max_bytes=4096)
    absent = dict(reference, source_posterior_uid="unselected", path=str(tmp_path/"absent.json.gz"))
    inline = {"source_posterior_uid": "historical-inline", "values": [3.]}
    registry = {"joint_sources": [inline], "joint_source_references": [reference, absent]}
    calls = []
    original = native.read_joint_source_reference
    def read(value):
        calls.append(value["source_posterior_uid"])
        return original(value)
    monkeypatch.setattr(native, "read_joint_source_reference", read)
    assert native.load_joint_sources(registry, ["complete-a", "complete-a"]) == {"complete-a": raw}
    assert calls == ["complete-a"]
    assert native.load_joint_sources(registry, ["historical-inline"]) == {"historical-inline": inline}
    assert calls == ["complete-a"]
    changed = copy.deepcopy(registry)
    changed["joint_sources"].append(raw)
    with pytest.raises(ValueError, match="both inline and referenced"):
        native.load_joint_sources(changed, ["complete-a"])


def test_storage_budget_reserves_next_group_and_counts_actual_changes(tmp_path):
    from Libs.scripts.builders.build_controlled_native_source import NativeStorageBudget
    budget = NativeStorageBudget(100, 40, 10)
    budget.reserve_group()
    first = tmp_path/"first"
    first.write_bytes(b"a"*30)
    budget.record([first], complete_group=True)
    budget.reserve_group()
    second = tmp_path/"second"
    second.write_bytes(b"b"*25)
    budget.record([second], complete_group=True)
    assert budget.used == 55
    with pytest.raises(RuntimeError, match="next complete group"):
        budget.reserve_group()
    first.write_bytes(b"a"*20)
    budget.record([first])
    assert budget.used == 45
    budget.reserve_group()
    with pytest.raises(FileNotFoundError, match="Declared native output"):
        budget.record([tmp_path/"missing"])


def test_inactive_witness_skips_only_irrelevant_events():
    from types import SimpleNamespace
    from Libs.tests.operational_twin_witness import PhaseEvidence
    def import_entry():
        pass
    def loader_entry():
        pass
    def numerical_entry():
        pass
    witness = object.__new__(PhaseEvidence)
    witness.active = None
    witness.import_function = import_entry
    witness.exec_function = loader_entry
    observed = []
    witness._profile_import = lambda frame, event, arg: observed.append(("import", frame.f_code))
    witness._profile_call = lambda frame, event, arg: observed.append(("call", frame.f_code))
    ordinary = SimpleNamespace(f_code=numerical_entry.__code__)
    for event in ("call", "return", "c_call"):
        witness._profile(ordinary, event, None)
    assert not observed
    for code in (import_entry.__code__, loader_entry.__code__, compile("value = 1", "module-fixture", "exec")):
        witness._profile(SimpleNamespace(f_code=code), "call", None)
        assert observed[-2:] == [("import", code), ("call", code)]
    observed.clear()
    witness.active = {"guarded_case": True}
    witness._profile(ordinary, "call", None)
    assert observed == [("import", ordinary.f_code), ("call", ordinary.f_code)]


def test_join_parses_complete_source_once_and_keeps_later_teacher_checks(tmp_path, monkeypatch):
    from Libs.tests.test_joint_world_posterior import joint_registry_case
    _, config, registry, _, source = joint_registry_case(tmp_path)
    before = copy.deepcopy(source)
    original = native.parse_joint_source
    calls = []
    def counted(raw, *args, **kwargs):
        calls.append(raw["source_posterior_uid"])
        return original(raw, *args, **kwargs)
    monkeypatch.setattr(native, "parse_joint_source", counted)
    records = native.join_registry(registry, config)
    assert len(records) == 4 and len(calls) == 1
    assert source == before
    changed = copy.deepcopy(registry)
    changed["teachers"][1]["action_uid"] = "wrong-second-teacher-action"
    with pytest.raises(ValueError, match="compiled program identity"):
        native.join_registry(changed, config)
    assert len(calls) == 2
    changed = copy.deepcopy(registry)
    changed["joint_sources"][0]["schema_version"] = "changed-source-schema"
    with pytest.raises(ValueError, match="schema"):
        native.join_registry(changed, config)
    assert len(calls) == 3
    assert source == before
