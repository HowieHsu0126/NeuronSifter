"""Scope branches of the existing observer; full-suite runtime checks remain separate."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from Libs.tests import operational_twin_witness as witness


def collection_state(focused):
    function = lambda: None
    name = witness.WATCHED[witness.TP][0]
    path = Path(witness.__file__).parent / "test_operational_twin_training_protocol.py"
    nodeid = str(path) + "::" + name
    state = SimpleNamespace(
        _observer=lambda: None, _check_bindings=lambda: None, _emit=lambda *a, **k: None,
        paths={witness.TP:path}, watched=witness.WATCHED, focused=focused,
        cases={nodeid:{"function_name":name,"parameters":{}}},
        _function=lambda *args:function, items={}, selected=[], source_identities={}, installed={},
        group_cases={"selected":[nodeid]}, item_groups={})
    item = SimpleNamespace(path=path, nodeid=nodeid, originalname=name, obj=function)
    return state, item


def test_broad_collection_keeps_watched_identity_checks():
    state, watched = collection_state(None)
    foreign_name = witness.GATE_TARGETS[0]
    foreign = SimpleNamespace(path=watched.path, nodeid=str(watched.path)+"::"+foreign_name,
                              originalname=foreign_name, obj=None)
    witness.PhaseEvidence.collection(state, [foreign, watched])
    assert state.selected == [watched.nodeid] and state.items[watched.nodeid] is watched
    assert state.item_groups == {watched.nodeid: "selected"}
    state, changed = collection_state(None)
    changed.obj = lambda: None
    with pytest.raises(Exception, match="collection replaced imported function"):
        witness.PhaseEvidence.collection(state, [changed])


def test_focused_collection_rejects_another_original_group():
    state, watched = collection_state({"nodeids":[]})
    foreign = SimpleNamespace(path=watched.path, nodeid="wrong-focused-group",
                              originalname=witness.GATE_TARGETS[0], obj=None)
    with pytest.raises(Exception, match="cross-group original caller"):
        witness.PhaseEvidence.collection(state, [foreign])
