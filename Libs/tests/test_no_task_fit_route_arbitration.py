from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from jsonschema import Draft202012Validator
import pytest
import yaml

from Libs.scripts.builders import build_no_task_fit_route_arbitration as builder
from Libs.scripts.builders.build_no_task_fit_route_arbitration import (
    ARBITRATION_OUTPUT,
    CANDIDATE_OUTPUT,
    EXPECTED_ABSENT_PATHS,
    EXPECTED_ASSET_PATHS,
    FORBIDDEN_OUTPUT_KEYS,
    LINEAGE_OUTPUT,
    ROUTE_IDS,
    RUBRIC_IDS,
    RouteArbitrationError,
    build_route_arbitration,
    canonical_json_bytes,
    project_candidate_registry,
    validate_output_schema,
)
from Libs.scripts.validators.validate_no_task_fit_route_arbitration import (
    validate_route_arbitration,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "Libs/configs/validation/no_task_fit_route_arbitration.yaml"
SCHEMA_PATHS = (
    REPO_ROOT / "Libs/configs/schema/no_task_fit_reusable_asset_lineage.schema.json",
    REPO_ROOT / "Libs/configs/schema/no_task_fit_route_arbitration.schema.json",
    REPO_ROOT / "Libs/configs/schema/no_task_fit_candidate_registry.schema.json",
)


def _parent_payload() -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-no-task-fit-parent-authority-v1",
        "status": "NO_TASK_FIT_PARENT_AUTHORITIES_BOUND",
        "artifact_count": 9,
        "blockers": [],
        "terminal_decisions": copy.deepcopy(builder.TERMINAL_DECISIONS),
        "aggregate_access": copy.deepcopy(builder.PARENT_AGGREGATE_ACCESS),
        "authorization_closure": copy.deepcopy(builder.PARENT_AUTHORIZATION_CLOSURE),
        "next_action": "arbitrate_no_task_fit_candidate_routes",
    }


def _write_contract(repository: Path, contract: dict[str, Any]) -> tuple[Path, str]:
    path = repository / "Libs/configs/validation/no_task_fit_route_arbitration.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_dump(contract, sort_keys=False).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], str, dict[str, dict[str, Any]]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    contract = copy.deepcopy(yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")))

    for source_schema in SCHEMA_PATHS:
        relative = source_schema.relative_to(REPO_ROOT)
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_schema.read_bytes())

    parent_raw = canonical_json_bytes(_parent_payload())
    parent_path = repository / contract["parent_authority"]["path"]
    parent_path.parent.mkdir(parents=True, exist_ok=True)
    parent_path.write_bytes(parent_raw)
    parent_hash = hashlib.sha256(parent_raw).hexdigest()
    contract["parent_authority"]["sha256"] = parent_hash
    for relative in (
        Path("Libs/configs/schema/no_task_fit_reusable_asset_lineage.schema.json"),
        Path("Libs/configs/schema/no_task_fit_route_arbitration.schema.json"),
    ):
        schema_path = repository / relative
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema["$defs"]["parentBinding"]["properties"]["sha256"]["const"] = parent_hash
        schema_path.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    git_provenance: dict[str, dict[str, Any]] = {}
    for index, asset in enumerate(contract["assets"]):
        path = repository / asset["path"]
        if asset["expected_presence"] == "absent":
            asset["sha256"] = None
            asset["bytes"] = None
            asset["git_blob_sha1"] = None
            asset["git_tracking_state"] = "absent"
            git_provenance[asset["path"]] = {
                "git_tracking_state": "absent",
                "git_blob_sha1": None,
            }
            continue
        if asset["path"] == contract["parent_authority"]["path"]:
            raw = parent_raw
        else:
            raw = (
                "\n".join(asset["required_text_markers"])
                + f"\nsynthetic_asset_{index}\n"
            ).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        asset["sha256"] = hashlib.sha256(raw).hexdigest()
        asset["bytes"] = len(raw)
        asset["git_blob_sha1"] = blob
        asset["git_tracking_state"] = "tracked_clean"
        git_provenance[asset["path"]] = {
            "git_tracking_state": "tracked_clean",
            "git_blob_sha1": blob,
        }

    _, contract_hash = _write_contract(repository, contract)
    return repository, contract, contract_hash, git_provenance


def _parent_validator(**_: object) -> dict[str, Any]:
    return {"status": "passed", "artifact_count": 9, "required_fact_count": 101}


@contextmanager
def _using_contract_identity(contract_hash: str, parent_hash: str):
    with patch.multiple(
        builder,
        EXPECTED_CONTRACT_FILE_SHA256=contract_hash,
        EXPECTED_PARENT_SHA256=parent_hash,
    ):
        yield


def _build(
    repository: Path,
    contract: dict[str, Any],
    contract_hash: str,
    git_provenance: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with _using_contract_identity(
        contract_hash, contract["parent_authority"]["sha256"]
    ):
        return build_route_arbitration(
            repository_root=repository,
            contract_path="Libs/configs/validation/no_task_fit_route_arbitration.yaml",
            parent_validator=_parent_validator,
            asset_git_provenance=git_provenance,
        )


def _validate(
    repository: Path,
    contract: dict[str, Any],
    contract_hash: str,
    git_provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    with _using_contract_identity(
        contract_hash, contract["parent_authority"]["sha256"]
    ):
        return validate_route_arbitration(
            repository_root=repository,
            contract_path="Libs/configs/validation/no_task_fit_route_arbitration.yaml",
            parent_validator=_parent_validator,
            asset_git_provenance=git_provenance,
        )


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(key)
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def test_contract_and_all_three_schemas_are_valid() -> None:
    raw = CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == builder.EXPECTED_CONTRACT_FILE_SHA256
    contract = yaml.safe_load(raw.decode())
    builder._validate_contract(contract)
    assert len(contract["assets"]) == 24
    assert len(contract["rubric"]) == 9
    assert len(contract["routes"]) == 4
    for path in SCHEMA_PATHS:
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_valid_structural_arbitration_produces_empty_terminal_registry(
    tmp_path: Path,
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    lineage, arbitration, registry = _build(
        repository, contract, contract_hash, provenance
    )

    assert lineage["asset_count"] == 24
    assert lineage["present_asset_count"] == 22
    assert lineage["absent_asset_count"] == 2
    assert arbitration["status"] == "NO_ADMISSIBLE_NO_TASK_FIT_TWIN_ROUTE"
    assert registry["status"] == "NO_ADMISSIBLE_NO_TASK_FIT_TWIN_ROUTE"
    assert registry["candidate_count"] == 0
    assert registry["candidates"] == []
    assert arbitration["admissible_route_ids"] == []
    assert registry["access_boundary"]["six_context_outcome_access"] == 0
    assert registry["access_boundary"]["protected_holdout_access"] == 0

    receipt = _validate(repository, contract, contract_hash, provenance)
    assert receipt["status"] == "passed"
    assert receipt["route_dispositions"] == {
        "route_a": "inadmissible",
        "route_b": "not_available",
        "route_c": "not_available",
        "route_d": "not_available",
    }


def test_all_routes_use_exact_same_nine_dimension_rubric(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _, arbitration, _ = _build(repository, contract, contract_hash, provenance)
    for route in arbitration["routes"]:
        assert [row["criterion_id"] for row in route["criteria"]] == list(RUBRIC_IDS)
        assert all(row["mandatory"] is True for row in route["criteria"])
        assert route["assumptions"]
        assert set(route["capability_support_summary"]) == {
            "model_family",
            "action_modalities",
            "control_modes",
            "input_interface_status",
            "output_interface_status",
            "parameter_identity_status",
            "solver_environment_status",
            "execution_policy",
            "uncertainty_hook_status",
        }
        assert route["disposition_reason"]
        assert route["mandatory_criterion_count"] == 9
        assert (
            route["satisfied_mandatory_count"]
            + route["unsatisfied_mandatory_count"]
            + route["not_applicable_mandatory_count"]
            == 9
        )


def test_route_a_remains_synthetic_and_is_not_admitted(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _, arbitration, registry = _build(repository, contract, contract_hash, provenance)
    route = next(row for row in arbitration["routes"] if row["route_id"] == "route_a")
    assert route["scientific_label"] == "synthetic_mechanistic_prior"
    assert route["availability_state"] == "available"
    assert route["disposition"] == "inadmissible"
    assert route["performance_evidence_used"] is False
    assert "route_a" not in registry["admissible_route_ids"]


def test_route_b_is_not_available_under_parent_authority(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _, arbitration, _ = _build(repository, contract, contract_hash, provenance)
    route = next(row for row in arbitration["routes"] if row["route_id"] == "route_b")
    assert route["disposition"] == "not_available"
    assert set(route["missing_required_asset_paths"]) == EXPECTED_ABSENT_PATHS
    assert any(
        "NO_FORMAL_ZERO_SHOT_CHECKPOINT" in row["reason"] for row in route["criteria"]
    )


def test_route_c_retains_prior_predictive_no_fit_limits(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _, arbitration, _ = _build(repository, contract, contract_hash, provenance)
    route = next(row for row in arbitration["routes"] if row["route_id"] == "route_c")
    criteria = {row["criterion_id"]: row["result"] for row in route["criteria"]}
    assert route["scientific_label"] == "prior_predictive"
    assert route["disposition"] == "not_available"
    assert criteria["no_task_fit_feasibility"] == "satisfied"
    assert criteria["complete_method_identity"] == "unsatisfied"
    assert route["task_specific_fitting_performed"] is False


def test_route_d_requires_a_and_b_and_preserves_absent_rule(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _, arbitration, _ = _build(repository, contract, contract_hash, provenance)
    route = next(row for row in arbitration["routes"] if row["route_id"] == "route_d")
    assert route["depends_on_route_ids"] == ["route_a", "route_b"]
    assert route["combination_rule_status"] == "absent_not_predeclared"
    assert route["disposition"] == "not_available"


def test_only_explicit_present_asset_paths_are_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    opened: list[str] = []
    original = builder._read_asset_bytes

    def recording_read(path: Path) -> bytes:
        opened.append(path.relative_to(repository).as_posix())
        return original(path)

    monkeypatch.setattr(builder, "_read_asset_bytes", recording_read)
    _build(repository, contract, contract_hash, provenance)
    expected = [
        path for path in EXPECTED_ASSET_PATHS if path not in EXPECTED_ABSENT_PATHS
    ]
    assert opened == expected
    assert len(opened) == len(set(opened)) == 22


def test_missing_expected_present_asset_fails_closed(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    path = "Libs/models/modules/neuronsifter_world_action.py"
    (repository / path).unlink()
    with pytest.raises(RouteArbitrationError, match="missing expected-present asset"):
        _build(repository, contract, contract_hash, provenance)
    assert not (repository / LINEAGE_OUTPUT).exists()


def test_present_asset_hash_drift_fails_closed(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    path = "Libs/models/modules/neuronsifter_world_action.py"
    with (repository / path).open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(RouteArbitrationError, match="content identity drift"):
        _build(repository, contract, contract_hash, provenance)
    assert not (repository / LINEAGE_OUTPUT).exists()


def test_missing_content_marker_fails_even_with_matching_declared_hash(
    tmp_path: Path,
) -> None:
    repository, contract, _, provenance = _fixture(tmp_path)
    path = "Libs/models/modules/neuronsifter_world_action.py"
    raw = b"synthetic content without required markers\n"
    (repository / path).write_bytes(raw)
    asset = next(row for row in contract["assets"] if row["path"] == path)
    asset["sha256"] = hashlib.sha256(raw).hexdigest()
    asset["bytes"] = len(raw)
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    asset["git_blob_sha1"] = blob
    provenance[path]["git_blob_sha1"] = blob
    _, contract_hash = _write_contract(repository, contract)

    with pytest.raises(
        RouteArbitrationError, match="required evidence markers missing"
    ):
        _build(repository, contract, contract_hash, provenance)
    assert not (repository / LINEAGE_OUTPUT).exists()


@pytest.mark.parametrize("field", ["git_tracking_state", "git_blob_sha1"])
def test_asset_git_identity_drift_fails_closed(tmp_path: Path, field: str) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    path = "Libs/models/modules/neuronsifter_world_action.py"
    provenance[path][field] = "absent" if field == "git_tracking_state" else "0" * 40
    with pytest.raises(RouteArbitrationError, match="Git identity drift"):
        _build(repository, contract, contract_hash, provenance)
    assert not (repository / LINEAGE_OUTPUT).exists()


def test_expected_absent_checkpoint_becoming_present_fails_closed(
    tmp_path: Path,
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    path = next(iter(EXPECTED_ABSENT_PATHS))
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not an authorized checkpoint")
    with pytest.raises(RouteArbitrationError, match="expected-absent asset is present"):
        _build(repository, contract, contract_hash, provenance)
    assert not (repository / LINEAGE_OUTPUT).exists()


def test_parent_hash_drift_fails_before_asset_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    parent_path = repository / contract["parent_authority"]["path"]
    parent_path.write_bytes(parent_path.read_bytes() + b" ")
    opened: list[Path] = []

    def forbidden_read(path: Path) -> bytes:
        opened.append(path)
        raise AssertionError("asset read occurred after parent drift")

    monkeypatch.setattr(builder, "_read_asset_bytes", forbidden_read)
    with pytest.raises(RouteArbitrationError, match="parent authority SHA-256 drift"):
        _build(repository, contract, contract_hash, provenance)
    assert opened == []


@pytest.mark.parametrize("drift", ["status", "access"])
def test_parent_status_or_access_drift_fails_closed(tmp_path: Path, drift: str) -> None:
    repository, contract, _, provenance = _fixture(tmp_path)
    parent_path = repository / contract["parent_authority"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if drift == "status":
        parent["status"] = "REOPENED"
        pattern = "schema or status drift"
    else:
        parent["aggregate_access"]["six_context_outcome_access"] = 1
        pattern = "missing or nonzero outcome/protected access"
    raw = canonical_json_bytes(parent)
    parent_path.write_bytes(raw)
    parent_hash = hashlib.sha256(raw).hexdigest()
    contract["parent_authority"]["sha256"] = parent_hash
    parent_asset = next(
        asset
        for asset in contract["assets"]
        if asset["path"] == contract["parent_authority"]["path"]
    )
    parent_asset["sha256"] = parent_hash
    parent_asset["bytes"] = len(raw)
    parent_asset["git_blob_sha1"] = hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw
    ).hexdigest()
    provenance[parent_asset["path"]]["git_blob_sha1"] = parent_asset["git_blob_sha1"]
    _, contract_hash = _write_contract(repository, contract)

    with pytest.raises(RouteArbitrationError, match=pattern):
        _build(repository, contract, contract_hash, provenance)


@pytest.mark.parametrize("mutation", ["path", "hash", "status", "parent_hash"])
def test_frozen_contract_mutations_fail_before_asset_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    frozen_parent_hash = contract["parent_authority"]["sha256"]
    if mutation == "path":
        contract["assets"][1]["path"] = "Libs/models/modules/replacement.py"
    elif mutation == "hash":
        contract["assets"][1]["sha256"] = "0" * 64
    elif mutation == "status":
        contract["routes"][0]["expected_disposition"] = "admissible"
    else:
        contract["parent_authority"]["sha256"] = "0" * 64
    _write_contract(repository, contract)
    opened: list[Path] = []

    def forbidden_read(path: Path) -> bytes:
        opened.append(path)
        raise AssertionError("asset read occurred before contract rejection")

    monkeypatch.setattr(builder, "_read_asset_bytes", forbidden_read)
    with _using_contract_identity(contract_hash, frozen_parent_hash):
        with pytest.raises(
            RouteArbitrationError, match="frozen route-arbitration contract drift"
        ):
            build_route_arbitration(
                repository_root=repository,
                contract_path="Libs/configs/validation/no_task_fit_route_arbitration.yaml",
                parent_validator=_parent_validator,
                asset_git_provenance=provenance,
            )
    assert opened == []
    assert not (repository / LINEAGE_OUTPUT).exists()


def test_candidate_registry_is_exact_all_admissible_projection(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    lineage, arbitration, _ = _build(repository, contract, contract_hash, provenance)
    synthetic_lineage = copy.deepcopy(lineage)
    for asset in synthetic_lineage["assets"]:
        if asset["observed_presence"] == "absent":
            asset["observed_presence"] = "present"
            asset["content_sha256"] = "a" * 64
            asset["bytes"] = 1
            asset["git_blob_sha1"] = "b" * 40
            asset["git_tracking_state"] = "tracked_clean"
    synthetic_arbitration = copy.deepcopy(arbitration)
    for route in synthetic_arbitration["routes"]:
        route["disposition"] = "admissible"
        route["missing_required_asset_paths"] = []
    synthetic_arbitration["admissible_route_ids"] = list(ROUTE_IDS)
    synthetic_arbitration["admissible_route_count"] = 4
    synthetic_arbitration["status"] = "NO_TASK_FIT_ROUTES_ARBITRATED"
    synthetic_arbitration["next_action"] = "freeze_admissible_no_task_fit_candidates"
    lineage_hash = hashlib.sha256(canonical_json_bytes(synthetic_lineage)).hexdigest()
    arbitration_hash = hashlib.sha256(
        canonical_json_bytes(synthetic_arbitration)
    ).hexdigest()

    first = project_candidate_registry(
        contract=contract,
        contract_sha256=contract_hash,
        parent_payload=_parent_payload(),
        asset_lineage=synthetic_lineage,
        asset_lineage_sha256=lineage_hash,
        arbitration=synthetic_arbitration,
        arbitration_sha256=arbitration_hash,
    )
    second = project_candidate_registry(
        contract=contract,
        contract_sha256=contract_hash,
        parent_payload=_parent_payload(),
        asset_lineage=synthetic_lineage,
        asset_lineage_sha256=lineage_hash,
        arbitration=synthetic_arbitration,
        arbitration_sha256=arbitration_hash,
    )
    assert first == second
    assert first["candidate_count"] == 4
    assert [candidate["route_id"] for candidate in first["candidates"]] == list(
        ROUTE_IDS
    )
    assert len({candidate["candidate_uid"] for candidate in first["candidates"]}) == 4


def test_outputs_have_no_forbidden_performance_or_selection_keys(
    tmp_path: Path,
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    outputs = _build(repository, contract, contract_hash, provenance)
    for payload in outputs:
        assert FORBIDDEN_OUTPUT_KEYS.isdisjoint(_walk_keys(payload))


@pytest.mark.parametrize("output_index", [0, 1, 2])
@pytest.mark.parametrize("extra_field", ["performance", "winner", "inference_result"])
def test_output_schemas_reject_forbidden_extra_fields(
    tmp_path: Path,
    output_index: int,
    extra_field: str,
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    outputs = _build(repository, contract, contract_hash, provenance)
    schema = json.loads(SCHEMA_PATHS[output_index].read_text(encoding="utf-8"))
    payload = copy.deepcopy(outputs[output_index])
    payload[extra_field] = {}
    with pytest.raises(
        RouteArbitrationError, match="Additional properties are not allowed"
    ):
        validate_output_schema(payload, schema, label="mutated output")


def test_repeated_build_is_byte_deterministic(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _build(repository, contract, contract_hash, provenance)
    paths = (LINEAGE_OUTPUT, ARBITRATION_OUTPUT, CANDIDATE_OUTPUT)
    first = {path: (repository / path).read_bytes() for path in paths}
    _build(repository, contract, contract_hash, provenance)
    second = {path: (repository / path).read_bytes() for path in paths}
    assert first == second


def test_existing_output_drift_is_not_overwritten(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    drifted = repository / LINEAGE_OUTPUT
    drifted.parent.mkdir(parents=True, exist_ok=True)
    drifted.write_bytes(b"existing route output drift\n")
    with pytest.raises(RouteArbitrationError, match="refusing to overwrite drifted"):
        _build(repository, contract, contract_hash, provenance)
    assert drifted.read_bytes() == b"existing route output drift\n"
    assert not (repository / ARBITRATION_OUTPUT).exists()
    assert not (repository / CANDIDATE_OUTPUT).exists()


def test_output_install_failure_rolls_back_all_created_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    original_replace = builder.os.replace
    replacements = 0

    def fail_on_second_output(source_path: object, destination_path: object) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("synthetic output installation failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(builder.os, "replace", fail_on_second_output)
    with pytest.raises(
        RouteArbitrationError, match="transactional route output installation failed"
    ):
        _build(repository, contract, contract_hash, provenance)
    assert not (repository / LINEAGE_OUTPUT).exists()
    assert not (repository / ARBITRATION_OUTPUT).exists()
    assert not (repository / CANDIDATE_OUTPUT).exists()


def test_validator_rejects_route_disposition_drift(tmp_path: Path) -> None:
    repository, contract, contract_hash, provenance = _fixture(tmp_path)
    _build(repository, contract, contract_hash, provenance)
    path = repository / ARBITRATION_OUTPUT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["routes"][0]["disposition"] = "admissible"
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(RouteArbitrationError):
        _validate(repository, contract, contract_hash, provenance)
