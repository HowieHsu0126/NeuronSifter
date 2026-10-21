"""Focused tests for the immutable Round-20 shared schemas."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import yaml

from Libs.scripts.validators.validate_dt_round20_shared_schemas import (
    SCHEMA_REGISTRY,
    _biological_fixture,
    _cell_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _schema(name: str) -> dict[str, object]:
    _, relative = SCHEMA_REGISTRY[name]
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_round20_schema_registry_is_exact_and_valid() -> None:
    config = yaml.safe_load((REPO_ROOT / "Libs/configs/schema/schemas.yaml").read_text(encoding="utf-8"))
    registry = config["schemas"]["round20_shared_schema_registry"]
    assert set(registry) == set(SCHEMA_REGISTRY)
    for name, (version, relative) in SCHEMA_REGISTRY.items():
        assert registry[name] == {"schema_version": version, "path": relative}
        Draft202012Validator.check_schema(_schema(name))


def test_cell_crosswalk_keeps_low_confidence_many_to_many_explicit() -> None:
    validator = Draft202012Validator(_schema("cell_taxonomy_crosswalk"))
    valid = _cell_fixture()
    assert list(validator.iter_errors(valid)) == []

    missing_group = copy.deepcopy(valid)
    missing_group["mapping_group_uid"] = None
    assert list(validator.iter_errors(missing_group))

    hidden_ambiguity = copy.deepcopy(valid)
    hidden_ambiguity["ambiguity"]["explicit"] = False
    assert list(validator.iter_errors(hidden_ambiguity))


def test_biological_unit_blocks_ambiguous_identity_from_training_role() -> None:
    validator = Draft202012Validator(_schema("biological_unit"))
    exact = _biological_fixture()
    assert list(validator.iter_errors(exact)) == []

    blocked = _biological_fixture("ambiguous_blocked")
    assert list(validator.iter_errors(blocked)) == []

    flattened = copy.deepcopy(blocked)
    flattened["mapping"]["ambiguity"]["explicit"] = False
    flattened["status"] = "active"
    flattened["train_eval_role"] = "public_state_observation"
    assert list(validator.iter_errors(flattened))
