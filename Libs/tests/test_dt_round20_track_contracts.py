"""Focused fail-closed tests for the Round-20 Track A/B/C contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator
import pytest
import yaml

from Libs.scripts.services import serve_screening_api
from Libs.scripts.validators import validate_dt_round20_allen_historical_migration as allen_migration
from Libs.scripts.validators import validate_dt_round20_allen_rights_release as allen_rights
from Libs.scripts.validators import validate_dt_round20_public_e2_e4_readiness as e2_e4_readiness
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.round20_track_contracts import (
    build_track_contract_report,
    validate_round20_track_contracts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "Libs/configs/validation/round20_track_contracts.yaml"
FROZEN_INPUT_COMMIT = "575630df6f2576fd0264fecfeae430c8f6bdf927"


def _contract() -> dict[str, object]:
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return payload["round20_track_contracts"]


@pytest.fixture
def historical_contract_repository(tmp_path: Path) -> Path:
    """Stage frozen identity inputs while exercising current imported validators."""
    contract = _contract()
    tracks = contract["tracks"]
    track_a = tracks["track_a_allen_full_landing"]
    declarations = [
        *contract["upstream_artifacts"],
        *track_a["historical_inputs"],
        track_a["source_registry"],
        *tracks["track_b_public_e2_e4_recovery"]["reused_authorities"],
        *tracks["track_c_runtime_deployment"]["input_authorities"],
    ]
    bindings: dict[str, str] = {}
    for item in declarations:
        relative = item["path"]
        expected = item["sha256"]
        assert relative not in bindings or bindings[relative] == expected
        bindings[relative] = expected

    repository = tmp_path / "historical_contract"
    repository.mkdir()
    for relative, expected in bindings.items():
        staged = REPO_ROOT / relative
        data = staged.read_bytes() if staged.is_file() else None
        if data is None or hashlib.sha256(data).hexdigest() != expected:
            data = subprocess.check_output(
                ["git", "show", f"{FROZEN_INPUT_COMMIT}:{relative}"], cwd=REPO_ROOT
            )
        assert hashlib.sha256(data).hexdigest() == expected, relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    fixture_contract = repository / CONTRACT_PATH.relative_to(REPO_ROOT)
    fixture_contract.parent.mkdir(parents=True, exist_ok=True)
    fixture_contract.write_bytes(CONTRACT_PATH.read_bytes())
    # The migration report records this genuine local fixture commit. Historical
    # Python files are identity data only; this directory never enters sys.path.
    for arguments in (
        ["init", "--quiet"],
        ["add", "--all"],
        ["-c", "user.name=NeuronSifter software fixture",
         "-c", "user.email=fixture@invalid.example", "-c", "commit.gpgsign=false",
         "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m",
         "Non-authoritative frozen historical contract inputs"],
    ):
        subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)
    return repository


def _blocker_codes(
    contract: dict[str, object], *, repository: Path = REPO_ROOT
) -> set[str]:
    return {
        str(item["code"])
        for item in validate_round20_track_contracts(contract, repository=repository)
    }


def test_round20_track_contract_is_canonical_and_fail_closed(
    historical_contract_repository: Path,
) -> None:
    contract = _contract()
    assert validate_round20_track_contracts(contract, repository=historical_contract_repository) == []

    stack = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert stack["round20_track_contracts"] == contract
    report = build_track_contract_report(contract, repository=historical_contract_repository)
    assert report["status"] == "passed"
    assert report["blocker_count"] == 0
    assert report["scope"] == {
        "strict_public_only_flag": True,
        "training_authorized": False,
        "training_started": False,
        "allen_payload_download_authorized": False,
        "real_tool_dispatch_authorized": False,
        "deployment_authorized": False,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
        "remote_push_authorized": False,
    }


def test_round20_track_a_blocks_family_rights_and_release_mutations(
    tmp_path: Path, historical_contract_repository: Path,
) -> None:
    contract = _contract()
    track = contract["tracks"]["track_a_allen_full_landing"]

    family_mutation = deepcopy(contract)
    family_mutation["tracks"]["track_a_allen_full_landing"]["families"].pop()
    assert "allen_family_inventory" in _blocker_codes(family_mutation, repository=historical_contract_repository)

    rights_mutation = deepcopy(contract)
    rights_mutation["tracks"]["track_a_allen_full_landing"]["rights_matrix"][
        "required_fields"
    ].remove("checkpoint_release_allowed")
    assert "allen_rights_contract" in _blocker_codes(rights_mutation, repository=historical_contract_repository)

    release_mutation = deepcopy(contract)
    release_mutation["tracks"]["track_a_allen_full_landing"]["release_inventory"][
        "mutable_latest_query_allowed"
    ] = True
    assert "allen_release_contract" in _blocker_codes(release_mutation, repository=historical_contract_repository)

    migration = allen_migration.audit(
        repository=historical_contract_repository,
        config_path=historical_contract_repository / CONTRACT_PATH.relative_to(REPO_ROOT),
    )
    assert migration["status"] == "passed_historical_migration_with_admission_blockers"
    assert migration["historical_input_count"] == 7
    assert migration["historical_failure_preservation"]["junit"] == {
        "tests": 598,
        "failures": 17,
        "errors": 19,
        "skipped": 0,
    }
    terms = tmp_path / "terms.html"
    terms.write_text("noncommercial citation policy redistribute", encoding="utf-8")
    snapshot = allen_rights._snapshot_from_file(
        tmp_path, tmp_path / "snapshots", "terms", "https://example.invalid/terms", terms
    )
    assert snapshot["sha256"] == allen_rights.sha256_file(terms)
    assert all(snapshot["markers"].values())


def test_round20_track_b_blocks_readiness_and_execution_authorization() -> None:
    contract = _contract()
    track = contract["tracks"]["track_b_public_e2_e4_recovery"]
    track["current_readiness"] = "training_ready_formal"
    track["authorization"]["intervention_control_pilot_execution"] = True
    track["authorization"]["training"] = True

    codes = _blocker_codes(contract)
    assert {"readiness_promotion", "track_b_authorization"} <= codes

    authority = e2_e4_readiness._artifact(REPO_ROOT, "Libs/configs/data/round19_public_e2_e4.yaml")
    assert authority["exists"] is True
    assert authority["sha256"] == _contract()["tracks"]["track_b_public_e2_e4_recovery"]["reused_authorities"][0]["sha256"]


def test_round20_track_c_blocks_duplicate_registry_shell_and_endpoint_drift() -> None:
    contract = _contract()
    track = contract["tracks"]["track_c_runtime_deployment"]
    track["duplicate_track_specific_registries_allowed"] = True
    track["service_contract"]["arbitrary_shell_execution_allowed"] = True
    track["service_contract"]["implementation_gaps"].remove("/ingest_tool_result")

    codes = _blocker_codes(contract)
    assert {"duplicate_registry", "service_authorization", "service_endpoints"} <= codes

    task = {
        "request_uid": "request_round20",
        "round_uid": "round20",
        "candidate_uid": "candidate_round20",
        "sample_state_uid": "state_round20",
        "intervention_program_uid": "action_round20",
        "simulator_family": "rdkit_descriptor",
        "fidelity_level": "L1",
        "requested_outputs": "molecular_descriptor",
        "budget_class": "exploration",
        "seed": "seed_round20",
        "source_holdout_role": "source_holdout",
    }
    typed_request = serve_screening_api._tool_request_from_acquisition(task)
    assert typed_request["tool_uid"] == "rdkit_descriptor"
    assert len(str(typed_request["payload_hash"])) == 64

    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    bundle = {"model_card": {"model_version": "test", "checkpoint_id": "checkpoint_test"}}
    dispatch_payload = {
        "round_uid": "round20",
        "acquisition_requests": [typed_request],
        "tool_versions": {"rdkit_descriptor": "rdkit-test-version"},
        "persist_artifacts": False,
    }
    dispatch = serve_screening_api.dispatch_tool_tasks(bundle, config, dispatch_payload)
    assert dispatch["dispatch_status"] == "blocked_real_tool_dispatch_disabled"
    assert dispatch["dispatched_task_count"] == 0
    assert dispatch["provenance"]["real_tool_dispatch_authorized"] is False
    assert "uncertainty" in dispatch and "tri" in dispatch and "tool_route" in dispatch
    assert "world_state_posterior_ref" in dispatch and "trajectory_ref" in dispatch
    assert dispatch["uncertainty_summary"] == dispatch["uncertainty"]
    assert dispatch["tri_components"] == dispatch["tri"]
    tool_request_schema = json.loads((REPO_ROOT / "Docs/design/iclr/schemas/tool_request.schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(tool_request_schema).iter_errors(dispatch["tool_requests"][0])) == []

    result = {
        "tool_result_uid": "result_round20",
        "request_uid": str(dispatch["tool_requests"][0]["request_uid"]),
        "tool_uid": str(dispatch["tool_requests"][0]["tool_uid"]),
        "tool_version": str(dispatch["tool_requests"][0]["tool_version"]),
        "input_hash": str(dispatch["tool_requests"][0]["payload_hash"]),
        "output_hash": "a" * 64,
        "runtime_seconds": 0.5,
        "failure_status": "not_executed",
        "qc_grade": "S",
    }
    tool_result_schema = json.loads((REPO_ROOT / "Docs/design/iclr/schemas/tool_result.schema.json").read_text(encoding="utf-8"))
    assert list(Draft202012Validator(tool_result_schema).iter_errors(result)) == []
    ingested = serve_screening_api.ingest_tool_result(
        bundle,
        config,
        {
            "round_uid": "round20",
            "tool_requests": dispatch["tool_requests"],
            "tool_results": [result],
            "persist_artifacts": False,
        },
    )
    assert ingested["ingestion_status"] == "registered_without_state_assimilation"
    assert ingested["state_assimilation_status"] == "blocked_direct_state_update"
    assert ingested["registered_result_count"] == 1

    invalid_result = dict(result)
    invalid_result["world_state_update"] = {"forbidden": True}
    try:
        serve_screening_api.ingest_tool_result(
            bundle,
            config,
            {
                "round_uid": "round20",
                "tool_requests": dispatch["tool_requests"],
                "tool_results": [invalid_result],
                "persist_artifacts": False,
            },
        )
    except ValueError as exc:
        assert "cannot update Twin state" in str(exc)
    else:
        raise AssertionError("Direct state update in tool result must fail closed.")
