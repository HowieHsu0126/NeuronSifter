from copy import deepcopy
from collections.abc import Callable

import pytest
import torch

from Libs.models.modules.world_action_encoder import WORLD_ACTION_FIELDS, WorldActionEncoder, world_action_schema_hash


def binding() -> dict[str, object]:
    return {
        "canonical_world_action": {"action_uid": "a", "intervention_type": "electrical_stimulation", "target_set": ["ca1"], "agent_set": ["stimulator"], "dose": None, "dose_unit": None, "schedule": {"kind": "program", "program_ref": "pulse"}, "route": None, "duration": {"value": 6.0, "unit": "ms"}, "stimulation_or_environment": {"modality": "electrical_stimulation"}, "state_scope": {"region": ["ca1"], "compartment": ["membrane"]}, "action_provenance_ref": "controlled_world_config"},
        "controlled_extension": {"modality": "electrical_stimulation", "dose_or_intensity": {"value": 4.0, "unit": "uA"}, "duration": {"value": 6.0, "unit": "ms"}, "combination": [], "biological_context": {"region": "ca1", "compartment": "membrane"}, "action_semantics": {"target_set": ["ca1"], "agent_set": ["stimulator"], "schedule": {"kind": "program", "program_ref": "pulse"}, "route": None, "support_domain_ref": "ephys", "provenance_ref": "controlled_world_config"}},
        "response_delay_steps": 2,
        "mapping_status": "canonical_mapping_validated",
    }


def test_encoding_is_deterministic_semantic_and_counterfactual_sensitive() -> None:
    encoder = WorldActionEncoder()
    action = binding()
    first = encoder.encode(action, "electrophysiology_conductance_world")
    assert torch.equal(first, encoder.encode(deepcopy(action), "electrophysiology_conductance_world"))
    changed = deepcopy(action)
    changed["controlled_extension"]["dose_or_intensity"]["value"] = 7.0
    assert not torch.equal(first, encoder.encode(changed, "electrophysiology_conductance_world"))
    no_action = encoder.encode_no_action(action, "electrophysiology_conductance_world")
    assert not torch.equal(first, no_action)
    assert first.shape == (len(WORLD_ACTION_FIELDS),)
    assert len(world_action_schema_hash()) == 64


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("target_set", lambda value: value["canonical_world_action"].update(target_set=["ca3"])),
        ("agent_set", lambda value: value["canonical_world_action"].update(agent_set=["stimulator_b"])),
        ("dose", lambda value: value["canonical_world_action"].update(dose=1.5, dose_unit="uA")),
        ("schedule", lambda value: value["canonical_world_action"].update(schedule={"kind": "program", "program_ref": "burst"})),
        ("route", lambda value: value["canonical_world_action"].update(route="intracranial")),
        ("context", lambda value: value["controlled_extension"].update(biological_context={"region": "ca3", "compartment": "membrane"})),
        ("combination", lambda value: value["controlled_extension"].update(combination=["controlled_modulator"])),
    ],
)
def test_each_required_action_identity_field_changes_encoding(
    field: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    encoder = WorldActionEncoder()
    original = binding()
    changed = deepcopy(original)
    mutate(changed)
    assert not torch.equal(
        encoder.encode(original, "electrophysiology_conductance_world"),
        encoder.encode(changed, "electrophysiology_conductance_world"),
    ), field


def test_set_like_identity_is_canonical_and_no_action_clears_agent_and_combination() -> None:
    encoder = WorldActionEncoder()
    original = binding()
    original["canonical_world_action"]["target_set"] = ["ca1", "ca3"]
    original["canonical_world_action"]["agent_set"] = ["stimulator_a", "stimulator_b"]
    original["controlled_extension"]["combination"] = ["modulator_a", "modulator_b"]
    reordered = deepcopy(original)
    reordered["canonical_world_action"]["target_set"].reverse()
    reordered["canonical_world_action"]["agent_set"].reverse()
    reordered["controlled_extension"]["combination"].reverse()
    assert torch.equal(
        encoder.encode(original, "electrophysiology_conductance_world"),
        encoder.encode(reordered, "electrophysiology_conductance_world"),
    )

    different_identity = deepcopy(original)
    different_identity["canonical_world_action"]["agent_set"] = ["other_stimulator"]
    different_identity["controlled_extension"]["combination"] = ["other_modulator"]
    assert torch.equal(
        encoder.encode_no_action(original, "electrophysiology_conductance_world"),
        encoder.encode_no_action(different_identity, "electrophysiology_conductance_world"),
    )


def test_encoding_rejects_unknown_family() -> None:
    try:
        WorldActionEncoder().encode(binding(), "unknown")
    except ValueError as error:
        assert "Unsupported family" in str(error)
    else:
        raise AssertionError("Unknown family was accepted")


def test_native_identity_uses_only_canonical_agent_membership():
    from Libs.models.modules.world_action_encoder import canonical_agent_identity
    action = binding()["canonical_world_action"]
    action["agent_set"] = ["agent-b", "agent-a"]
    original = canonical_agent_identity(action)
    changed = deepcopy(action)
    changed.update(agent_set=["agent-a", "agent-b", "agent-a"], dose=99., dose_unit="different",
                   target_set=["other-target"], schedule={"kind": "different"}, route="different",
                   action_provenance_ref="different", ranked_selection=True, case_study_seed=True)
    assert canonical_agent_identity(changed) == original
    changed["agent_set"] = ["agent-c"]
    assert canonical_agent_identity(changed) != original


@pytest.mark.parametrize("agents", [None, [], "agent", [""], [None], [" padded "], [{"agent": "x"}]])
def test_native_identity_rejects_noncanonical_membership(agents):
    from Libs.models.modules.world_action_encoder import canonical_agent_identity
    with pytest.raises(ValueError, match="agent_set"):
        canonical_agent_identity({"agent_set": agents})



def test_native_target_identity_uses_only_canonical_membership():
    from Libs.models.modules.world_action_encoder import canonical_target_identity, native_action_identity_spec, NATIVE_TARGET_IDENTITY_SCHEMA
    action = {"target_set": ["target-b", "target-a", "target-b"], "agent_set": ["drug-a"],
              "dose": 2., "route": "intravenous", "schedule": {"kind": "daily"}}
    encoded = canonical_target_identity(action)
    assert encoded == canonical_target_identity({"target_set": ["target-a", "target-b"]})
    assert encoded == canonical_target_identity(dict(action, agent_set=["other"], dose=9., route="oral", schedule={}, outcome=99.))
    assert encoded != canonical_target_identity(dict(action, target_set=["target-c"]))
    assert native_action_identity_spec("target_id") == (canonical_target_identity, NATIVE_TARGET_IDENTITY_SCHEMA)
    # Source-declared empty controls remain explicit; pharmacological admission is separate.
    assert len(canonical_target_identity({"target_set": []})) == 4
    assert canonical_target_identity({"target_set": []}) != encoded


@pytest.mark.parametrize("action", [None, {}, {"target_set": None}, {"target_set": "target"},
    {"target_set": [""]}, {"target_set": [" target"]}, {"target_set": [1]}])
def test_native_target_identity_rejects_missing_or_malformed_membership(action):
    from Libs.models.modules.world_action_encoder import canonical_target_identity
    with pytest.raises(ValueError, match="target_set"):
        canonical_target_identity(action)
