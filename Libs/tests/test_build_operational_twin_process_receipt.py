from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from Libs.scripts.builders import (
    build_operational_twin_process_receipt as receipt_builder,
)
from Libs.scripts.builders import (
    build_operational_twin_sufficient_training_candidate as candidate_builder,
)


COMMIT = "a" * 40


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repository,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _temporary_repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "tests@neuronsifter.local")
    _git(tmp_path, "config", "user.name", "NeuronSifter Tests")
    source = tmp_path / receipt_builder.DEFAULT_PRODUCER_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("print('aggregate')\n", encoding="utf-8")
    command_source = tmp_path / receipt_builder.DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE
    command_source.parent.mkdir(parents=True, exist_ok=True)
    command_source.write_text("print('receipt')\n", encoding="utf-8")
    tracked_helper = tmp_path / "Libs/tracked_helper.py"
    tracked_helper.write_text("VALUE = 1\n", encoding="utf-8")
    pytest_config = tmp_path / "pytest.ini"
    pytest_config.write_text("[pytest]\n", encoding="utf-8")
    _git(
        tmp_path,
        "add",
        source.relative_to(tmp_path).as_posix(),
        command_source.relative_to(tmp_path).as_posix(),
        tracked_helper.relative_to(tmp_path).as_posix(),
        pytest_config.relative_to(tmp_path).as_posix(),
    )
    _git(tmp_path, "commit", "-q", "-m", "fixture")
    return tmp_path


def _producer_identity_payload(
    source_path: str,
    source_bytes: bytes,
    *,
    code_commit: str = COMMIT,
) -> dict[str, object]:
    return {
        "schema_version": receipt_builder.SCHEMA_VERSION,
        "receipt_uid": receipt_builder.RECEIPT_UID,
        "receipt_kind": receipt_builder.RECEIPT_KIND,
        "receipt_group": receipt_builder.RECEIPT_GROUP,
        "process_receipt_kind": receipt_builder.PROCESS_RECEIPT_KIND,
        "code_commit": code_commit,
        "producer_source": {
            "path": source_path,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "classification": dict(receipt_builder.PRODUCER_IDENTITY_CLASSIFICATION),
    }


def _focused_execution_payload(
    *,
    contract_name: str = "tests",
    producer_path: str,
    producer_bytes: bytes,
    log_bytes: bytes,
    code_commit: str = COMMIT,
    source_state: dict[str, object] | None = None,
    exit_code: int = 0,
) -> dict[str, object]:
    contract = receipt_builder.focused_validation_contract(contract_name)
    if source_state is None:
        source_state = {
            "identity_method": receipt_builder.FOCUSED_VALIDATION_SOURCE_STATE_METHOD,
            "required_clean_scopes": [
                {
                    "path": scope.as_posix(),
                    "git_object_id": "c" * 40,
                    "git_object_type": (
                        "tree" if scope.as_posix() == "Libs" else "blob"
                    ),
                }
                for scope in contract.required_clean_scopes
            ],
        }
    return {
        "schema_version": receipt_builder.SCHEMA_VERSION,
        "receipt_uid": contract.receipt_uid,
        "receipt_kind": "process",
        "receipt_group": "focused_validation",
        "process_receipt_kind": "command_execution",
        "receipt_path": contract.output_path.as_posix(),
        "code_commit": code_commit,
        "producer_source": {
            "path": producer_path,
            "sha256": hashlib.sha256(producer_bytes).hexdigest(),
        },
        "command_identity": receipt_builder.command_identity(contract.argv),
        "source_state": source_state,
        "execution": {
            "scheduler": "slurm",
            "slurm_job_id": "12345",
            "job_name": "NrnSift_focused_tests",
            "partition": "cpu",
            "node": "cpu04",
            "exit_code": exit_code,
            "process_status": "COMPLETED" if exit_code == 0 else "FAILED",
        },
        "log_source": {
            "path": contract.log_path.as_posix(),
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
            "size_bytes": len(log_bytes),
        },
    }


def test_builder_emits_deterministic_producer_identity_only_receipt(
    tmp_path: Path,
) -> None:
    repository = _temporary_repository(tmp_path)

    first = receipt_builder.build_producer_identity_receipt(repo_root=repository)
    second = receipt_builder.build_producer_identity_receipt(repo_root=repository)

    assert first == second
    assert first == {
        "schema_version": receipt_builder.SCHEMA_VERSION,
        "receipt_uid": receipt_builder.RECEIPT_UID,
        "receipt_kind": "process",
        "receipt_group": "aggregation",
        "process_receipt_kind": "producer_identity",
        "code_commit": _git(repository, "rev-parse", "HEAD"),
        "producer_source": first["producer_source"],
        "classification": {
            "scope": "producer_identity_only",
            "output_artifact_hashes": "not_included",
            "execution_success_claim": "not_made",
        },
    }
    source = repository / receipt_builder.DEFAULT_PRODUCER_SOURCE
    assert first["producer_source"] == {
        "path": receipt_builder.DEFAULT_PRODUCER_SOURCE.as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert "timestamp" not in first
    assert "output_artifacts" not in first
    assert "execution_success" not in first
    schema = json.loads(receipt_builder.SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(first)
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second,
        indent=2,
        sort_keys=True,
    )


def test_command_builder_executes_and_emits_deterministic_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _temporary_repository(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NAME", "NrnSift_focused_tests")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu")
    monkeypatch.setenv("SLURMD_NODENAME", "cpu04")
    contract = receipt_builder.focused_validation_contract("tests")

    def run_registered_command(*, contract, repo_root, stdout):
        assert contract.contract_name == "tests"
        assert repo_root == repository
        stdout.write(b"focused-pass\n")
        return 0

    monkeypatch.setattr(
        receipt_builder,
        "_run_registered_command",
        run_registered_command,
    )

    first_output, first, first_exit = (
        receipt_builder.execute_registered_command_receipt(
            repo_root=repository,
            contract_name="tests",
        )
    )
    second_output, second, second_exit = (
        receipt_builder.execute_registered_command_receipt(
            repo_root=repository,
            contract_name="tests",
        )
    )

    assert first_output == repository / contract.output_path
    assert second_output == first_output
    assert first_exit == second_exit == 0
    assert first == second
    assert first["receipt_path"] == contract.output_path.as_posix()
    assert first["command_identity"] == receipt_builder.command_identity(
        contract.argv
    )
    assert [
        row["path"] for row in first["source_state"]["required_clean_scopes"]
    ] == ["Libs", "pytest.ini"]
    assert first["execution"] == {
        "scheduler": "slurm",
        "slurm_job_id": "12345",
        "job_name": "NrnSift_focused_tests",
        "partition": "cpu",
        "node": "cpu04",
        "exit_code": 0,
        "process_status": "COMPLETED",
    }
    log_bytes = (repository / contract.log_path).read_bytes()
    assert log_bytes == b"focused-pass\n"
    assert first["log_source"] == {
        "path": contract.log_path.as_posix(),
        "sha256": hashlib.sha256(log_bytes).hexdigest(),
        "size_bytes": len(log_bytes),
    }
    assert json.loads(first_output.read_text(encoding="utf-8")) == first
    receipt_builder.validate_process_receipt_payload(
        first,
        receipt_path=contract.output_path.as_posix(),
    )


def test_command_builder_preserves_nonzero_exit_as_failed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _temporary_repository(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12346")
    monkeypatch.setenv("SLURM_JOB_NAME", "NrnSift_focused_failure")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu")
    monkeypatch.setenv("SLURMD_NODENAME", "cpu04")

    def run_registered_command(*, contract, repo_root, stdout):
        assert contract.contract_name == "ruff"
        assert repo_root == repository
        stdout.write(b"failed\n")
        return 7

    monkeypatch.setattr(
        receipt_builder,
        "_run_registered_command",
        run_registered_command,
    )
    contract = receipt_builder.focused_validation_contract("ruff")

    output, payload, exit_code = (
        receipt_builder.execute_registered_command_receipt(
            repo_root=repository,
            contract_name="ruff",
        )
    )

    assert exit_code == 7
    assert payload["execution"]["exit_code"] == 7
    assert payload["execution"]["process_status"] == "FAILED"
    assert b"failed\n" == (repository / contract.log_path).read_bytes()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("receipt_group",), "aggregation"),
        (("process_receipt_kind",), "producer_identity"),
        (("receipt_uid",), "operational_twin_training_evidence_ruff"),
        (("receipt_path",), "receipts/not_registered.json"),
        (("command_identity", "sha256"), "b" * 64),
        (("log_source", "path"), "logs/not_registered.log"),
        (("execution", "node"), "cpu03"),
        (("execution", "process_status"), "FAILED"),
    ],
)
def test_command_payload_rejects_wrong_semantics_or_identity(
    field_path: tuple[str, ...],
    invalid_value: str,
) -> None:
    payload = _focused_execution_payload(
        producer_path=(
            receipt_builder.DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE.as_posix()
        ),
        producer_bytes=b"receipt\n",
        log_bytes=b"pass\n",
    )
    target: dict[str, object] = payload
    for part in field_path[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[field_path[-1]] = invalid_value

    with pytest.raises(receipt_builder.ProcessReceiptError):
        receipt_builder.validate_process_receipt_payload(payload)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("receipt_kind", "scheduler"),
        ("receipt_group", "focused_validation"),
        ("process_receipt_kind", "execution"),
    ],
)
def test_builder_rejects_wrong_receipt_semantics(
    tmp_path: Path,
    override: str,
    value: str,
) -> None:
    repository = _temporary_repository(tmp_path)

    with pytest.raises(receipt_builder.ProcessReceiptError):
        receipt_builder.build_producer_identity_receipt(
            repo_root=repository,
            **{override: value},
        )


def test_builder_rejects_dirty_and_untracked_producer_sources(tmp_path: Path) -> None:
    repository = _temporary_repository(tmp_path)
    source = repository / receipt_builder.DEFAULT_PRODUCER_SOURCE
    source.write_text("print('dirty')\n", encoding="utf-8")

    with pytest.raises(receipt_builder.ProcessReceiptError, match="uncommitted"):
        receipt_builder.build_producer_identity_receipt(repo_root=repository)

    _git(repository, "restore", source.relative_to(repository).as_posix())
    untracked = repository / "Libs/scripts/builders/untracked_aggregator.py"
    untracked.write_text("print('untracked')\n", encoding="utf-8")
    with pytest.raises(receipt_builder.ProcessReceiptError, match="not tracked"):
        receipt_builder.build_producer_identity_receipt(
            repo_root=repository,
            producer_source=untracked,
        )


def test_command_builder_rejects_dirty_or_untracked_source_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _temporary_repository(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12347")
    monkeypatch.setenv("SLURM_JOB_NAME", "NrnSift_focused_source")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu")
    monkeypatch.setenv("SLURMD_NODENAME", "cpu04")
    contract = receipt_builder.focused_validation_contract("tests")

    def must_not_run(**_kwargs):
        raise AssertionError("command must not execute for an invalid source state")

    monkeypatch.setattr(receipt_builder, "_run_registered_command", must_not_run)
    producer = repository / receipt_builder.DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE
    producer.write_text("print('dirty')\n", encoding="utf-8")

    with pytest.raises(receipt_builder.ProcessReceiptError, match="uncommitted"):
        receipt_builder.execute_registered_command_receipt(
            repo_root=repository,
            contract_name="tests",
        )
    assert not (repository / contract.log_path).exists()

    _git(repository, "restore", producer.relative_to(repository).as_posix())
    untracked = repository / "Libs/untracked_receipt.py"
    untracked.write_text("print('untracked')\n", encoding="utf-8")
    with pytest.raises(
        receipt_builder.ProcessReceiptError,
        match="dirty or untracked",
    ):
        receipt_builder.execute_registered_command_receipt(
            repo_root=repository,
            contract_name="tests",
        )
    assert not (repository / contract.log_path).exists()


def test_focused_validation_contracts_are_exact_and_noninterchangeable() -> None:
    tests_contract = receipt_builder.focused_validation_contract("tests")
    ruff_contract = receipt_builder.focused_validation_contract("ruff")

    assert set(receipt_builder.FOCUSED_VALIDATION_RECEIPT_CONTRACTS) == {
        "tests",
        "ruff",
    }
    assert tests_contract.receipt_uid != ruff_contract.receipt_uid
    assert tests_contract.output_path != ruff_contract.output_path
    assert tests_contract.log_path != ruff_contract.log_path
    assert tests_contract.argv != ruff_contract.argv
    assert tests_contract.required_clean_scopes == (
        Path("Libs"),
        Path("pytest.ini"),
    )
    assert ruff_contract.required_clean_scopes == (Path("Libs"),)
    for contract in (tests_contract, ruff_contract):
        assert receipt_builder.focused_validation_contract_for_receipt_path(
            contract.output_path.as_posix()
        ) == contract
        assert receipt_builder.focused_validation_contract_for_receipt_uid(
            contract.receipt_uid
        ) == contract
        assert not any(
            token in contract.output_path.as_posix().lower()
            for token in ("phase", "stage", "milestone")
        )


def test_builder_rejects_candidate_or_receipt_self_reference(tmp_path: Path) -> None:
    repository = _temporary_repository(tmp_path)
    candidate = repository / "Output/audits/operational_twin_sufficient_training_candidate.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}\n", encoding="utf-8")
    _git(repository, "add", candidate.relative_to(repository).as_posix())
    _git(repository, "commit", "-q", "-m", "candidate fixture")

    with pytest.raises(receipt_builder.ProcessReceiptError, match="must not reference"):
        receipt_builder.build_producer_identity_receipt(
            repo_root=repository,
            producer_source=candidate,
        )
    with pytest.raises(receipt_builder.ProcessReceiptError, match="must not reference"):
        receipt_builder.build_producer_identity_receipt(
            repo_root=repository,
            producer_source=receipt_builder.DEFAULT_AGGREGATION_RECEIPT_PATH,
        )


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("code_commit",), "short"),
        (("producer_source", "sha256"), "not-a-sha256"),
        (("receipt_group",), "focused_validation"),
        (("process_receipt_kind",), "execution"),
    ],
)
def test_payload_validation_rejects_invalid_commit_hash_or_type(
    tmp_path: Path,
    field_path: tuple[str, ...],
    invalid_value: str,
) -> None:
    repository = _temporary_repository(tmp_path)
    payload = receipt_builder.build_producer_identity_receipt(repo_root=repository)
    target: dict[str, object] = payload
    for part in field_path[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[field_path[-1]] = invalid_value

    with pytest.raises(receipt_builder.ProcessReceiptError):
        receipt_builder.validate_process_receipt_payload(payload)


def _focused_log_bytes(contract_name: str) -> bytes:
    return f"{contract_name} validation passed\n".encode("utf-8")


def _valid_focused_payloads(
    repository: Path,
    *,
    code_commit: str,
) -> dict[str, dict[str, object]]:
    producer_path = receipt_builder.DEFAULT_COMMAND_RECEIPT_PRODUCER_SOURCE
    producer_bytes = (repository / producer_path).read_bytes()
    payloads: dict[str, dict[str, object]] = {}
    for contract_name in ("tests", "ruff"):
        contract = receipt_builder.focused_validation_contract(contract_name)
        source_state = receipt_builder._source_scope_identities(
            repo_root=repository,
            code_commit=code_commit,
            contract=contract,
        )
        payloads[contract_name] = _focused_execution_payload(
            contract_name=contract_name,
            producer_path=producer_path.as_posix(),
            producer_bytes=producer_bytes,
            log_bytes=_focused_log_bytes(contract_name),
            code_commit=code_commit,
            source_state=source_state,
        )
    return payloads


def _collector_sources(
    repository: Path,
    aggregation_payload: dict[str, object],
    *,
    focused_payloads: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, tuple[Path, ...]], set[str]]:
    receipt_root = repository / "receipts"
    receipt_root.mkdir()
    source_commit = str(aggregation_payload["code_commit"])
    if focused_payloads is None:
        focused_payloads = _valid_focused_payloads(
            repository,
            code_commit=source_commit,
        )
    scheduler_header = (
        "JobIDRaw|JobName|Partition|NodeList|State|ExitCode|Elapsed|"
        "AllocTRES|ReqMem|AllocCPUS|SubmitLine\n"
    )
    source_text = {
        "training": (
            scheduler_header
            + "1001|NrnSift_train|cpu|cpu04|COMPLETED|0:0|00:00:01|"
            "cpu=1,mem=2G|2G|1|srun python train.py\n"
        ),
        "evaluation": (
            scheduler_header
            + "1002|NrnSift_eval|cpu|cpu04|COMPLETED|0:0|00:00:01|"
            "cpu=1,mem=2G|2G|1|srun python evaluate.py\n"
        ),
        "reconciliation": json.dumps(
            {
                "receipt_uid": "reconciliation-process",
                "training_code_commit": source_commit,
            }
        )
        + "\n",
        "aggregation": json.dumps(aggregation_payload, sort_keys=True) + "\n",
    }
    source_paths: dict[str, tuple[Path, ...]] = {}
    for group, text in source_text.items():
        relative = Path("receipts") / f"{group}.txt"
        (repository / relative).write_text(text, encoding="utf-8")
        source_paths[group] = (relative,)
    focused_paths: list[Path] = []
    for contract_name in ("tests", "ruff"):
        contract = receipt_builder.focused_validation_contract(contract_name)
        log_bytes = _focused_log_bytes(contract_name)
        log = repository / contract.log_path
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(log_bytes)
        receipt = repository / contract.output_path
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(focused_payloads[contract_name], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        focused_paths.append(contract.output_path)
    source_paths["focused_validation"] = tuple(focused_paths)
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "receipt fixtures")
    committed_paths = set(
        _git(repository, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    )
    return source_paths, committed_paths


def test_collector_accepts_valid_aggregation_producer_identity_receipt(
    tmp_path: Path,
) -> None:
    repository = _temporary_repository(tmp_path)
    producer_relative = receipt_builder.DEFAULT_PRODUCER_SOURCE.as_posix()
    producer_bytes = (repository / producer_relative).read_bytes()
    source_commit = _git(repository, "rev-parse", "HEAD")
    payload = _producer_identity_payload(
        producer_relative,
        producer_bytes,
        code_commit=source_commit,
    )
    source_paths, committed_paths = _collector_sources(repository, payload)
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    assert blockers == []
    assert groups["aggregation"][0]["receipt_kind"] == "process"
    assert groups["aggregation"][0]["code_commit"] == source_commit
    assert commits["aggregation"] == {source_commit}
    assert len(groups["focused_validation"]) == 2
    assert all(
        receipt["receipt_kind"] == "process"
        and receipt["code_commit"] == source_commit
        for receipt in groups["focused_validation"]
    )
    assert commits["focused_validation"] == {source_commit}
    assert candidate_builder.FINAL_RECEIPT_SOURCE_PATHS["aggregation"] == (
        receipt_builder.DEFAULT_AGGREGATION_RECEIPT_PATH,
    )
    assert candidate_builder.FINAL_RECEIPT_SOURCE_PATHS["focused_validation"] == (
        receipt_builder.DEFAULT_FOCUSED_VALIDATION_TESTS_RECEIPT_PATH,
        receipt_builder.DEFAULT_FOCUSED_VALIDATION_RUFF_RECEIPT_PATH,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receipt_group", "focused_validation"),
        ("process_receipt_kind", "execution"),
        (
            "classification",
            {
                "scope": "producer_identity_only",
                "output_artifact_hashes": "included",
                "execution_success_claim": "made",
            },
        ),
    ],
)
def test_collector_rejects_mismatched_typed_aggregation_receipt(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    repository = _temporary_repository(tmp_path)
    producer_relative = receipt_builder.DEFAULT_PRODUCER_SOURCE.as_posix()
    producer_bytes = (repository / producer_relative).read_bytes()
    source_commit = _git(repository, "rev-parse", "HEAD")
    payload = _producer_identity_payload(
        producer_relative,
        producer_bytes,
        code_commit=source_commit,
    )
    payload[field] = value
    source_paths, committed_paths = _collector_sources(repository, payload)
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    assert "process_receipt_invalid" in {row["code"] for row in blockers}
    assert groups["aggregation"][0]["code_commit"] == ""
    assert commits["aggregation"] == set()


def test_collector_rejects_mismatched_producer_source_hash(tmp_path: Path) -> None:
    repository = _temporary_repository(tmp_path)
    producer_relative = receipt_builder.DEFAULT_PRODUCER_SOURCE.as_posix()
    producer_bytes = (repository / producer_relative).read_bytes()
    source_commit = _git(repository, "rev-parse", "HEAD")
    payload = _producer_identity_payload(
        producer_relative,
        producer_bytes,
        code_commit=source_commit,
    )
    source = payload["producer_source"]
    assert isinstance(source, dict)
    source["sha256"] = "b" * 64
    source_paths, committed_paths = _collector_sources(repository, payload)
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    assert "source_sha256_mismatch" in {row["code"] for row in blockers}
    assert groups["aggregation"][0]["code_commit"] == ""
    assert commits["aggregation"] == set()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("wrong_group", "process_receipt_invalid"),
        ("wrong_kind", "process_receipt_invalid"),
        ("wrong_command_hash", "process_receipt_invalid"),
        ("nonzero_exit", "process_receipt_execution_failed"),
    ],
)
def test_collector_rejects_invalid_focused_command_receipt(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    repository = _temporary_repository(tmp_path)
    producer_relative = receipt_builder.DEFAULT_PRODUCER_SOURCE.as_posix()
    producer_bytes = (repository / producer_relative).read_bytes()
    source_commit = _git(repository, "rev-parse", "HEAD")
    aggregation_payload = _producer_identity_payload(
        producer_relative,
        producer_bytes,
        code_commit=source_commit,
    )
    focused_payloads = _valid_focused_payloads(
        repository,
        code_commit=source_commit,
    )
    focused_payload = focused_payloads["tests"]
    if mutation == "wrong_group":
        focused_payload["receipt_group"] = "aggregation"
    elif mutation == "wrong_kind":
        focused_payload["process_receipt_kind"] = "producer_identity"
    elif mutation == "wrong_command_hash":
        identity = focused_payload["command_identity"]
        assert isinstance(identity, dict)
        identity["sha256"] = "b" * 64
    elif mutation == "nonzero_exit":
        execution = focused_payload["execution"]
        assert isinstance(execution, dict)
        execution["exit_code"] = 7
        execution["process_status"] = "FAILED"
    source_paths, committed_paths = _collector_sources(
        repository,
        aggregation_payload,
        focused_payloads=focused_payloads,
    )
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    assert expected_code in {row["code"] for row in blockers}
    assert groups["focused_validation"][0]["code_commit"] == ""
    assert commits["focused_validation"] == {source_commit}


@pytest.mark.parametrize("failure_mode", ["missing", "hash", "size", "dirty", "untracked"])
def test_collector_rejects_unbound_focused_command_log(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    repository = _temporary_repository(tmp_path)
    producer_relative = receipt_builder.DEFAULT_PRODUCER_SOURCE.as_posix()
    producer_bytes = (repository / producer_relative).read_bytes()
    source_commit = _git(repository, "rev-parse", "HEAD")
    aggregation_payload = _producer_identity_payload(
        producer_relative,
        producer_bytes,
        code_commit=source_commit,
    )
    focused_payloads = _valid_focused_payloads(
        repository,
        code_commit=source_commit,
    )
    focused_payload = focused_payloads["tests"]
    contract = receipt_builder.focused_validation_contract("tests")
    log_relative = contract.log_path
    if failure_mode == "hash":
        log_source = focused_payload["log_source"]
        assert isinstance(log_source, dict)
        log_source["sha256"] = "b" * 64
    elif failure_mode == "size":
        log_source = focused_payload["log_source"]
        assert isinstance(log_source, dict)
        log_source["size_bytes"] = 1
    source_paths, committed_paths = _collector_sources(
        repository,
        aggregation_payload,
        focused_payloads=focused_payloads,
    )
    changed_paths: set[str] = set()
    if failure_mode == "missing":
        (repository / log_relative).unlink()
    elif failure_mode == "dirty":
        changed_paths.add(log_relative.as_posix())
    elif failure_mode == "untracked":
        committed_paths.remove(log_relative.as_posix())
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=changed_paths,
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    codes = {row["code"] for row in blockers}
    expected_codes = {
        "missing": "source_missing",
        "hash": "source_sha256_mismatch",
        "size": "source_size_mismatch",
        "dirty": "source_content_uncommitted",
        "untracked": "source_untracked",
    }
    assert expected_codes[failure_mode] in codes
    assert groups["focused_validation"][0]["code_commit"] == ""
    assert commits["focused_validation"] == {source_commit}


def test_collector_rejects_focused_source_state_object_mismatch(
    tmp_path: Path,
) -> None:
    repository = _temporary_repository(tmp_path)
    source_commit = _git(repository, "rev-parse", "HEAD")
    producer_path = receipt_builder.DEFAULT_PRODUCER_SOURCE
    aggregation_payload = _producer_identity_payload(
        producer_path.as_posix(),
        (repository / producer_path).read_bytes(),
        code_commit=source_commit,
    )
    focused_payloads = _valid_focused_payloads(
        repository,
        code_commit=source_commit,
    )
    source_state = focused_payloads["tests"]["source_state"]
    assert isinstance(source_state, dict)
    scopes = source_state["required_clean_scopes"]
    assert isinstance(scopes, list)
    assert isinstance(scopes[0], dict)
    scopes[0]["git_object_id"] = "d" * 40
    source_paths, committed_paths = _collector_sources(
        repository,
        aggregation_payload,
        focused_payloads=focused_payloads,
    )
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    codes = {row["code"] for row in blockers}
    assert "focused_validation_source_state_mismatch" in codes
    assert groups["focused_validation"][0]["code_commit"] == ""
    assert commits["focused_validation"] == {source_commit}


def test_collector_rejects_duplicate_focused_command_substitution(
    tmp_path: Path,
) -> None:
    repository = _temporary_repository(tmp_path)
    source_commit = _git(repository, "rev-parse", "HEAD")
    producer_path = receipt_builder.DEFAULT_PRODUCER_SOURCE
    aggregation_payload = _producer_identity_payload(
        producer_path.as_posix(),
        (repository / producer_path).read_bytes(),
        code_commit=source_commit,
    )
    focused_payloads = _valid_focused_payloads(
        repository,
        code_commit=source_commit,
    )
    focused_payloads["ruff"] = focused_payloads["tests"]
    source_paths, committed_paths = _collector_sources(
        repository,
        aggregation_payload,
        focused_payloads=focused_payloads,
    )
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    codes = {row["code"] for row in blockers}
    assert "process_receipt_invalid" in codes
    assert "focused_validation_contract_set_invalid" in codes
    assert groups["focused_validation"][1]["code_commit"] == ""
    assert commits["focused_validation"] == {source_commit}


def test_collector_rejects_dirty_registered_source_scope(tmp_path: Path) -> None:
    repository = _temporary_repository(tmp_path)
    source_commit = _git(repository, "rev-parse", "HEAD")
    producer_path = receipt_builder.DEFAULT_PRODUCER_SOURCE
    aggregation_payload = _producer_identity_payload(
        producer_path.as_posix(),
        (repository / producer_path).read_bytes(),
        code_commit=source_commit,
    )
    source_paths, committed_paths = _collector_sources(
        repository,
        aggregation_payload,
    )
    tracked_helper = repository / "Libs/tracked_helper.py"
    tracked_helper.write_text("VALUE = 2\n", encoding="utf-8")
    blockers: list[dict[str, str]] = []

    groups, commits, _ = candidate_builder._collect_receipt_groups(
        root=repository,
        committed_paths=committed_paths,
        changed_paths={"Libs/tracked_helper.py"},
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    assert "focused_validation_source_scope_dirty" in {
        row["code"] for row in blockers
    }
    assert all(
        receipt["code_commit"] == "" for receipt in groups["focused_validation"]
    )
    assert commits["focused_validation"] == set()
