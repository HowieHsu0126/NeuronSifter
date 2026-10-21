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

from Libs.scripts.builders import build_no_task_fit_parent_authority as builder
from Libs.scripts.builders.build_no_task_fit_parent_authority import (
    AuthorityBindingError,
    EXPECTED_ARTIFACT_IDENTITIES,
    build_parent_authority,
    validate_contract,
    validate_manifest_payload,
)
from Libs.scripts.validators.validate_no_task_fit_parent_authority import (
    validate_bound_parent_authority,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "Libs/configs/validation/no_task_fit_parent_authority.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/no_task_fit_parent_authority.schema.json"


def _segments(pointer: str) -> list[str]:
    return [
        segment.replace("~1", "/").replace("~0", "~")
        for segment in pointer[1:].split("/")
    ]


def _set_pointer(payload: dict[str, Any], pointer: str, value: object) -> None:
    segments = _segments(pointer)
    current: object = payload
    for index, segment in enumerate(segments):
        final = index == len(segments) - 1
        next_is_index = not final and segments[index + 1].isdigit()
        if isinstance(current, dict):
            if final:
                current[segment] = value
                return
            expected_type = list if next_is_index else dict
            child = current.get(segment)
            if not isinstance(child, expected_type):
                child = expected_type()
                current[segment] = child
            current = child
            continue
        if isinstance(current, list):
            item_index = int(segment)
            expected_type = list if next_is_index else dict
            while len(current) <= item_index:
                current.append(None)
            if final:
                current[item_index] = value
                return
            child = current[item_index]
            if not isinstance(child, expected_type):
                child = expected_type()
                current[item_index] = child
            current = child
            continue
        raise AssertionError(f"cannot assign JSON Pointer {pointer}")


def _delete_pointer(payload: dict[str, Any], pointer: str) -> None:
    segments = _segments(pointer)
    current: object = payload
    for segment in segments[:-1]:
        current = (
            current[int(segment)] if isinstance(current, list) else current[segment]
        )
    final = segments[-1]
    if isinstance(current, list):
        current.pop(int(final))
    else:
        del current[final]


def _write_contract(repository: Path, contract: dict[str, Any]) -> Path:
    path = repository / "Libs/configs/validation/no_task_fit_parent_authority.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source = tmp_path / "source"
    repository = tmp_path / "repository"
    source.mkdir()
    repository.mkdir()
    contract = copy.deepcopy(yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")))

    schema_destination = (
        repository / "Libs/configs/schema/no_task_fit_parent_authority.schema.json"
    )
    schema_destination.parent.mkdir(parents=True)
    schema_destination.write_bytes(SCHEMA_PATH.read_bytes())

    for record in contract["artifacts"]:
        payload: dict[str, Any] = {
            "schema_version": record["schema_version"],
            "status": record["status"],
        }
        for fact in record["required_facts"]:
            _set_pointer(payload, fact["pointer"], fact["expected"])
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        path = source / record["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        record["bytes"] = len(raw)

    _write_contract(repository, contract)
    return source, repository, contract


def _source_provenance(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch": contract["source_checkout"]["expected_branch"],
        "head": contract["source_checkout"]["expected_head"],
        "tracking_states": {
            record["source_path"]: contract["source_checkout"][
                "expected_artifact_tracking_state"
            ]
            for record in contract["artifacts"]
        },
    }


def _destination_provenance() -> dict[str, str]:
    return {"branch": "orchestrator/test-parent-authority", "head_at_build": "1" * 40}


def _freeze_spec(
    contract: dict[str, Any],
) -> tuple[dict[str, dict[str, str | int]], str, int]:
    specifications: dict[str, dict[str, str | int]] = {}
    for record in contract["artifacts"]:
        specifications[record["source_path"]] = {
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "required_fact_count": len(record["required_facts"]),
            "required_facts_sha256": builder.canonical_value_sha256(
                record["required_facts"]
            ),
            "record_sha256": builder.canonical_value_sha256(record),
        }
    return (
        specifications,
        builder.canonical_value_sha256(contract["artifacts"]),
        sum(len(record["required_facts"]) for record in contract["artifacts"]),
    )


@contextmanager
def _using_freeze(
    contract: dict[str, Any],
    *,
    required_fact_count: int | None = None,
):
    specifications, aggregate_digest, observed_fact_count = _freeze_spec(contract)
    with patch.multiple(
        builder,
        EXPECTED_FROZEN_ARTIFACT_CONTRACT=specifications,
        EXPECTED_ARTIFACTS_SHA256=aggregate_digest,
        EXPECTED_REQUIRED_FACT_COUNT=(
            observed_fact_count if required_fact_count is None else required_fact_count
        ),
    ):
        yield


def _build(
    source: Path,
    repository: Path,
    contract: dict[str, Any],
    *,
    frozen_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _using_freeze(frozen_contract or contract):
        return build_parent_authority(
            source_root=source,
            repository_root=repository,
            contract_path="Libs/configs/validation/no_task_fit_parent_authority.yaml",
            source_checkout_provenance=_source_provenance(contract),
            destination_checkout_provenance=_destination_provenance(),
        )


def _validate(repository: Path, contract: dict[str, Any]) -> dict[str, Any]:
    with _using_freeze(contract):
        return validate_bound_parent_authority(
            repository_root=repository,
            contract_path="Libs/configs/validation/no_task_fit_parent_authority.yaml",
            manifest_path="Output/audits/no_task_fit_parent_authority.json",
        )


def _record(contract: dict[str, Any], path: str) -> dict[str, Any]:
    return next(
        record for record in contract["artifacts"] if record["source_path"] == path
    )


def _refresh_source_identity(
    source: Path, repository: Path, contract: dict[str, Any], path: str
) -> None:
    raw = (source / path).read_bytes()
    record = _record(contract, path)
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    record["bytes"] = len(raw)
    _write_contract(repository, contract)


def test_valid_nine_file_binding_and_read_only_validation(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    manifest = _build(source, repository, contract)

    assert manifest["status"] == "NO_TASK_FIT_PARENT_AUTHORITIES_BOUND"
    assert manifest["artifact_count"] == 9
    assert manifest["blockers"] == []
    assert manifest["aggregate_access"]["six_context_outcome_access"] == 0
    assert manifest["aggregate_access"]["protected_holdout_access"] == 0
    assert manifest["next_action"] == "arbitrate_no_task_fit_candidate_routes"
    for relative in EXPECTED_ARTIFACT_IDENTITIES:
        assert (repository / relative).read_bytes() == (source / relative).read_bytes()

    receipt = _validate(repository, contract)
    assert receipt["status"] == "passed"
    assert receipt["artifact_count"] == 9
    assert receipt["blocker_count"] == 0


def test_missing_source_fails_before_any_copy(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    missing = contract["artifacts"][-1]["source_path"]
    (source / missing).unlink()

    with pytest.raises(AuthorityBindingError, match="Missing source artifact"):
        _build(source, repository, contract)
    assert not any(
        (repository / path).exists() for path in EXPECTED_ARTIFACT_IDENTITIES
    )
    assert not (repository / "Output/audits/no_task_fit_parent_authority.json").exists()


def test_altered_source_byte_fails_hash_preflight_without_copy(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    altered = contract["artifacts"][0]["source_path"]
    with (source / altered).open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(AuthorityBindingError, match="source SHA-256 mismatch"):
        _build(source, repository, contract)
    assert not any(
        (repository / path).exists() for path in EXPECTED_ARTIFACT_IDENTITIES
    )


@pytest.mark.parametrize("field", ["status", "schema_version"])
def test_wrong_source_status_or_schema_is_rejected(tmp_path: Path, field: str) -> None:
    source, repository, contract = _fixture(tmp_path)
    path = contract["artifacts"][0]["source_path"]
    payload = json.loads((source / path).read_text(encoding="utf-8"))
    payload[field] = "wrong"
    (source / path).write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_source_identity(source, repository, contract, path)

    with pytest.raises(AuthorityBindingError, match=f"{field} mismatch"):
        _build(source, repository, contract)


def test_unknown_category_is_rejected(tmp_path: Path) -> None:
    _, repository, contract = _fixture(tmp_path)
    contract["artifacts"][0]["category"] = "optimistic_reinterpretation"
    _write_contract(repository, contract)
    with pytest.raises(AuthorityBindingError, match="unknown authority category"):
        validate_contract(contract)


@pytest.mark.parametrize(
    "mutation", ["duplicate_uid", "duplicate_path", "escaped_path"]
)
def test_duplicate_or_escaped_contract_entries_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    _, _, contract = _fixture(tmp_path)
    frozen_contract = copy.deepcopy(contract)
    if mutation == "duplicate_uid":
        contract["artifacts"][1]["authority_uid"] = contract["artifacts"][0][
            "authority_uid"
        ]
        pattern = "duplicate authority_uid"
    elif mutation == "duplicate_path":
        contract["artifacts"][1]["source_path"] = contract["artifacts"][0][
            "source_path"
        ]
        contract["artifacts"][1]["destination_path"] = contract["artifacts"][0][
            "destination_path"
        ]
        pattern = "duplicate source or destination path"
    else:
        contract["artifacts"][0]["source_path"] = "../escaped.json"
        contract["artifacts"][0]["destination_path"] = "../escaped.json"
        pattern = "not a normalized repository-relative path"
    with _using_freeze(frozen_contract):
        with pytest.raises(AuthorityBindingError, match=pattern):
            validate_contract(contract)


@pytest.mark.parametrize(
    ("invalid_path", "pattern"),
    [
        ("/tmp/absolute.json", "not a normalized repository-relative path"),
        ("Output/audits/*.json", "contains a glob"),
        ("Output/audits/not_json.txt", "must name one JSON file"),
        ("Output/audits", "must name one JSON file"),
    ],
)
def test_absolute_glob_non_json_and_directory_entries_are_rejected(
    tmp_path: Path,
    invalid_path: str,
    pattern: str,
) -> None:
    _, _, contract = _fixture(tmp_path)
    contract["artifacts"][0]["source_path"] = invalid_path
    contract["artifacts"][0]["destination_path"] = invalid_path
    with pytest.raises(AuthorityBindingError, match=pattern):
        validate_contract(contract)


@pytest.mark.parametrize(
    "weakening",
    [
        "source_sha256",
        "source_bytes",
        "fact_removed",
        "fact_pointer_replaced",
        "fact_value_changed",
    ],
)
def test_frozen_contract_rejects_every_identity_or_fact_weakening_before_source_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weakening: str,
) -> None:
    source, repository, contract = _fixture(tmp_path)
    frozen_contract = copy.deepcopy(contract)
    record = contract["artifacts"][0]
    if weakening == "source_sha256":
        record["sha256"] = "0" * 64
        pattern = "frozen source SHA-256 changed"
    elif weakening == "source_bytes":
        record["bytes"] += 1
        pattern = "frozen source byte count changed"
    elif weakening == "fact_removed":
        record["required_facts"].pop()
        pattern = "frozen required fact count changed"
    elif weakening == "fact_pointer_replaced":
        record["required_facts"][0]["pointer"] = "/replacement/fact"
        pattern = "frozen required fact set changed"
    else:
        expected = record["required_facts"][0]["expected"]
        record["required_facts"][0]["expected"] = not expected
        pattern = "frozen required fact set changed"
    _write_contract(repository, contract)

    opened: list[Path] = []

    def forbidden_read(path: Path) -> bytes:
        opened.append(path)
        raise AssertionError("source read occurred before contract rejection")

    monkeypatch.setattr(builder, "_read_source_bytes", forbidden_read)
    with pytest.raises(AuthorityBindingError, match=pattern):
        _build(source, repository, contract, frozen_contract=frozen_contract)
    assert opened == []
    assert not any(
        (repository / path).exists() for path in EXPECTED_ARTIFACT_IDENTITIES
    )
    assert not (repository / "Output/audits/no_task_fit_parent_authority.json").exists()


def test_total_required_fact_count_is_independently_pinned(tmp_path: Path) -> None:
    _, _, contract = _fixture(tmp_path)
    contract["artifacts"][0]["required_facts"].pop()
    with _using_freeze(contract, required_fact_count=101):
        with pytest.raises(
            AuthorityBindingError, match="frozen required fact total changed"
        ):
            validate_contract(contract)


@pytest.mark.parametrize("mode", ["missing", "nonzero"])
def test_missing_or_nonzero_access_counter_is_a_blocker(
    tmp_path: Path, mode: str
) -> None:
    source, repository, contract = _fixture(tmp_path)
    path = "Output/audits/prospective_development_foundation_decision.json"
    payload = json.loads((source / path).read_text(encoding="utf-8"))
    pointer = "/execution_authority/six_context_outcome_access"
    if mode == "missing":
        _delete_pointer(payload, pointer)
        pattern = "Missing required JSON Pointer"
    else:
        _set_pointer(payload, pointer, 1)
        pattern = "nonzero or invalid access counter"
    (source / path).write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_source_identity(source, repository, contract, path)

    with pytest.raises(AuthorityBindingError, match=pattern):
        _build(source, repository, contract)


def test_reopened_authorization_is_a_blocker(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    path = "Output/audits/prospective_twin_reconstruction_decision.json"
    payload = json.loads((source / path).read_text(encoding="utf-8"))
    _set_pointer(payload, "/execution_authority/gpu_authorized", True)
    (source / path).write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_source_identity(source, repository, contract, path)

    with pytest.raises(AuthorityBindingError, match="reopens closed authority"):
        _build(source, repository, contract)


@pytest.mark.parametrize(
    "malformed_bytes",
    [b'{"unterminated":', b'{"invalid_utf8":"\xff"}'],
    ids=["malformed_json", "malformed_utf8"],
)
def test_malformed_source_with_matching_declared_identity_fails_before_copy(
    tmp_path: Path,
    malformed_bytes: bytes,
) -> None:
    source, repository, contract = _fixture(tmp_path)
    path = contract["artifacts"][-1]["source_path"]
    (source / path).write_bytes(malformed_bytes)
    _refresh_source_identity(source, repository, contract, path)

    with pytest.raises(AuthorityBindingError, match="not valid UTF-8 JSON"):
        _build(source, repository, contract)
    assert not any(
        (repository / relative).exists() for relative in EXPECTED_ARTIFACT_IDENTITIES
    )
    assert not (repository / "Output/audits/no_task_fit_parent_authority.json").exists()


def test_destination_overwrite_drift_is_refused_without_partial_copy(
    tmp_path: Path,
) -> None:
    source, repository, contract = _fixture(tmp_path)
    drift_path = contract["artifacts"][0]["destination_path"]
    destination = repository / drift_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"user-owned drift\n")

    with pytest.raises(
        AuthorityBindingError, match="refusing to overwrite drifted destination"
    ):
        _build(source, repository, contract)
    assert destination.read_bytes() == b"user-owned drift\n"
    assert not any(
        (repository / path).exists() for path in list(EXPECTED_ARTIFACT_IDENTITIES)[1:]
    )


def test_aggregate_manifest_overwrite_drift_is_refused(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    manifest_path = repository / "Output/audits/no_task_fit_parent_authority.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"user-owned aggregate drift\n")

    with pytest.raises(AuthorityBindingError, match="drifted aggregate manifest"):
        _build(source, repository, contract)
    assert manifest_path.read_bytes() == b"user-owned aggregate drift\n"
    assert not any(
        (repository / path).exists() for path in EXPECTED_ARTIFACT_IDENTITIES
    )


def test_exact_idempotent_rerun_preserves_manifest_bytes(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    first = _build(source, repository, contract)
    manifest_path = repository / "Output/audits/no_task_fit_parent_authority.json"
    first_bytes = manifest_path.read_bytes()
    second = _build(source, repository, contract)

    assert first == second
    assert manifest_path.read_bytes() == first_bytes


def test_late_source_failure_is_transactional(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    path = contract["artifacts"][-1]["source_path"]
    payload = json.loads((source / path).read_text(encoding="utf-8"))
    payload["status"] = "wrong"
    (source / path).write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_source_identity(source, repository, contract, path)

    with pytest.raises(AuthorityBindingError, match="status mismatch"):
        _build(source, repository, contract)
    assert not any(
        (repository / relative).exists() for relative in EXPECTED_ARTIFACT_IDENTITIES
    )
    assert not (repository / "Output/audits/no_task_fit_parent_authority.json").exists()


def test_install_failure_rolls_back_every_created_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, repository, contract = _fixture(tmp_path)
    original_replace = builder.os.replace
    artifact_replacements = 0

    def fail_during_second_artifact(
        source_path: object, destination_path: object
    ) -> None:
        nonlocal artifact_replacements
        destination = Path(destination_path)
        if destination.name != "no_task_fit_parent_authority.json":
            artifact_replacements += 1
            if artifact_replacements == 2:
                raise OSError("synthetic install failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(builder.os, "replace", fail_during_second_artifact)
    with pytest.raises(
        AuthorityBindingError, match="transactional installation failed"
    ):
        _build(source, repository, contract)
    assert not any(
        (repository / relative).exists() for relative in EXPECTED_ARTIFACT_IDENTITIES
    )
    assert not (repository / "Output/audits/no_task_fit_parent_authority.json").exists()


@pytest.mark.parametrize(
    "extra_field", ["route_candidates", "performance", "inference_results"]
)
def test_aggregate_schema_rejects_route_performance_and_inference_fields(
    tmp_path: Path,
    extra_field: str,
) -> None:
    source, repository, contract = _fixture(tmp_path)
    manifest = _build(source, repository, contract)
    schema = json.loads(
        (
            repository / "Libs/configs/schema/no_task_fit_parent_authority.schema.json"
        ).read_text()
    )
    Draft202012Validator.check_schema(schema)
    manifest[extra_field] = {}
    with pytest.raises(
        AuthorityBindingError, match="Additional properties are not allowed"
    ):
        validate_manifest_payload(manifest, schema)


def test_validator_rejects_destination_byte_drift(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    _build(source, repository, contract)
    path = contract["artifacts"][3]["destination_path"]
    with (repository / path).open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(AuthorityBindingError, match="SHA-256"):
        _validate(repository, contract)


def test_builder_opens_only_the_nine_explicit_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, repository, contract = _fixture(tmp_path)
    opened: list[str] = []
    original = builder._read_source_bytes

    def recording_read(path: Path) -> bytes:
        opened.append(path.relative_to(source).as_posix())
        return original(path)

    monkeypatch.setattr(builder, "_read_source_bytes", recording_read)
    _build(source, repository, contract)
    assert opened == list(EXPECTED_ARTIFACT_IDENTITIES)
    assert len(opened) == len(set(opened)) == 9


def test_source_symlink_is_rejected_before_read(tmp_path: Path) -> None:
    source, repository, contract = _fixture(tmp_path)
    relative = contract["artifacts"][0]["source_path"]
    source_path = source / relative
    replacement = source / "replacement.json"
    replacement.write_bytes(source_path.read_bytes())
    source_path.unlink()
    source_path.symlink_to(replacement)

    with pytest.raises(AuthorityBindingError, match="traverses a symlink"):
        _build(source, repository, contract)
    assert not any(
        (repository / path).exists() for path in EXPECTED_ARTIFACT_IDENTITIES
    )
