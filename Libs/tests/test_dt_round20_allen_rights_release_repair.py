"""Fail-closed tests for the Allen rights and release repair contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import yaml

from Libs.scripts.validators import validate_dt_round20_allen_rights_release_repair_contract as repair
from Libs.utils.data.raw_data_registry import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/validation/round20_allen_rights_release_repair.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/round20_allen_rights_release_repair_contract.schema.json"


def _contract() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["round20_allen_rights_release_repair"]


def _codes(contract: dict[str, object]) -> set[str]:
    return {str(item["code"]) for item in repair.validate_contract_definition(contract)}


def _materialize_inputs(contract: dict[str, object], root: Path) -> None:
    for index, item in enumerate(contract["input_artifacts"]):
        relative = Path("inputs") / f"{index:02d}-{item['artifact_uid']}.txt"
        payload = f"{item['artifact_uid']}\n".encode("utf-8")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        item["path"] = str(relative)
        item["sha256"] = hashlib.sha256(payload).hexdigest()


def test_repair_contract_is_canonical_and_fail_closed(tmp_path: Path) -> None:
    contract = _contract()
    assert repair.validate_contract_definition(contract) == []
    assert load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")["round20_allen_rights_release_repair"] == contract
    _materialize_inputs(contract, tmp_path)
    report = repair.build_report(
        contract,
        repository=tmp_path,
        config_path=CONFIG_PATH,
        candidate_commit="a" * 40,
    )
    assert report["status"] == "passed"
    assert report["gate"] == "allen_rights_release_repair_contract_frozen"
    assert report["blocker_count"] == 0
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(report)) == []


def test_repair_contract_rejects_authorization_and_identity_mutations() -> None:
    authorization_mutation = deepcopy(_contract())
    authorization_mutation["authorization"]["payload_download"] = True
    assert "unauthorized_execution" in _codes(authorization_mutation)

    release_mutation = deepcopy(_contract())
    release_mutation["source_policy"]["mutable_latest_query_allowed"] = True
    assert "source_policy_promotion" in _codes(release_mutation)

    family_mutation = deepcopy(_contract())
    family_mutation["family_uids"].pop()
    assert "family_inventory_drift" in _codes(family_mutation)

    identity_mutation = deepcopy(_contract())
    identity_mutation["input_artifacts"][0]["sha256"] = "not-a-hash"
    assert "input_identity_hash" in _codes(identity_mutation)


def test_repair_contract_verifies_all_frozen_input_hashes(tmp_path: Path) -> None:
    contract = deepcopy(_contract())
    _materialize_inputs(contract, tmp_path)
    identities, blockers = repair.validate_input_identities(contract, repository=tmp_path)
    assert blockers == []
    assert len(identities) == 9
    assert {item["status"] for item in identities} == {"verified"}


def test_repair_contract_reports_missing_and_hash_mismatched_inputs(tmp_path: Path) -> None:
    contract = deepcopy(_contract())
    _materialize_inputs(contract, tmp_path)
    missing_path = tmp_path / str(contract["input_artifacts"][0]["path"])
    missing_path.unlink()
    mismatch_path = tmp_path / str(contract["input_artifacts"][1]["path"])
    mismatch_path.write_text("drift\n", encoding="utf-8")
    identities, blockers = repair.validate_input_identities(contract, repository=tmp_path)
    assert {item["status"] for item in identities} == {"verified", "missing", "hash_mismatch"}
    assert {item["code"] for item in blockers} == {"input_identity_missing", "input_hash_mismatch"}
