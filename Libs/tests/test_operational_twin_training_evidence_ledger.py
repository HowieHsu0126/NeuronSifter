from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator
import pytest

from Libs.scripts.builders import (
    build_operational_twin_sufficient_training_candidate as evidence_builder,
)
from Libs.scripts.validators import verify_operational_twin_devcal as devcal_verifier
from Libs.utils.evaluation.operational_twin_training_evidence_ledger import (
    COMMAND_IDENTITY_ALGORITHM,
    COMMAND_IDENTITY_NORMALIZATION,
    COMMAND_IDENTITY_SOURCE_FIELD,
    PROCESS_RECEIPT_GROUPS,
    REQUIRED_CLOSED_CLAIM_FLAGS,
    REQUIRED_RECEIPT_GROUPS,
    SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION,
    SCHEDULER_RECEIPT_GROUPS,
    build_operational_twin_training_evidence_ledger,
    compute_operational_twin_training_evidence_ledger_sha256,
    validate_operational_twin_training_evidence_ledger,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    METHOD_EXECUTION_CONTRACTS,
    REQUIRED_METHOD_IDS,
    REQUIRED_ZERO_ACCESS_COUNTERS,
    canonical_hash,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/schema/operational_twin_training_evidence_ledger.schema.json"
)
COMMIT = "1" * 40


def _source(path: str, *, tracked: bool = True) -> dict[str, object]:
    return {
        "path": path,
        "sha256": canonical_hash(path),
        "size_bytes": len(path.encode("utf-8")),
        "tracked": tracked,
    }


def _scheduler(uid: str, *, state: str = "COMPLETED", exit_code: str = "0:0") -> dict[str, object]:
    return {
        "job_id": f"job-{uid}",
        "task_id": "0",
        "job_name": "NrnSift_evidence_contract",
        "node": "cpu04",
        "partition": "cpu",
        "state": state,
        "exit_code": exit_code,
        "resources": {
            "cpus": 1,
            "gpus": 0,
            "memory": "2G",
            "elapsed": "00:00:01",
            "allocation": "cpu=1,mem=2G",
        },
    }


def _protocol() -> dict[str, object]:
    return {
        "schema_version": "neuronsifter-operational-twin-training-protocol-v1",
        "protocol_uid": "operational_twin_development_training_protocol",
        "methods": [{"method_id": method_id} for method_id in REQUIRED_METHOD_IDS],
        "seeds": [101, 211, 307],
        "output_roots": {"assignment_root": "Output/results/operational_twin/development_training"},
    }


def _access_flags() -> dict[str, int]:
    return {key: 0 for key in REQUIRED_ZERO_ACCESS_COUNTERS}


def _claim_flags() -> dict[str, bool]:
    return {
        "training_authorized": False,
        **{key: False for key in REQUIRED_CLOSED_CLAIM_FLAGS},
    }


def _access_authority(
    *,
    protocol_tracked: bool = True,
    preflight_tracked: bool = True,
    preflight_access_flags: dict[str, int] | None = None,
    preflight_protocol_uid: str = "operational_twin_development_training_protocol",
) -> tuple[dict[str, object], bool, list[dict[str, str]]]:
    protocol = {
        "protocol_uid": "operational_twin_development_training_protocol",
        "access_counters": _access_flags(),
    }
    preflight = {
        "protocol_uid": preflight_protocol_uid,
        "access_counters": (
            _access_flags()
            if preflight_access_flags is None
            else preflight_access_flags
        ),
    }
    blockers: list[dict[str, str]] = []
    policy, normalization_authorized = evidence_builder._authoritative_access_policy(
        protocol=protocol,
        preflight=preflight,
        protocol_source={"tracked": protocol_tracked},
        preflight_source={"tracked": preflight_tracked},
        blockers=blockers,
    )
    return policy, normalization_authorized, blockers


def _attempt(
    *,
    assignment_uid: str,
    method_id: str,
    seed: int,
    ordinal: int,
    status: str,
    recovered_from_attempt_uid: str | None = None,
    contributes_to_candidate: bool,
    superseded_by_attempt_uid: str | None,
    provenance_gaps: tuple[str, ...] = (),
) -> dict[str, object]:
    attempt_uid = f"{assignment_uid}__attempt_{ordinal:03d}"
    scheduler_state = "CANCELLED" if status == "CANCELLED" else ("FAILED" if status == "FAILED" else "COMPLETED")
    exit_code = "0:15" if status == "CANCELLED" else ("1:0" if status == "FAILED" else "0:0")
    return {
        "assignment_uid": assignment_uid,
        "method_id": method_id,
        "seed": seed,
        "attempt_uid": attempt_uid,
        "ordinal": ordinal,
        "status": status,
        "terminal_reason": status.lower(),
        "recovered_from_attempt_uid": recovered_from_attempt_uid,
        "source": _source(f"evidence/attempts/{attempt_uid}.json"),
        "command_receipt": {
            "receipt_uid": f"command-{attempt_uid}",
            "source": _source(f"evidence/commands/{attempt_uid}.out"),
        },
        "scheduler": _scheduler(attempt_uid, state=scheduler_state, exit_code=exit_code),
        "code_commit": COMMIT,
        "provenance_disposition": (
            "complete_contributing"
            if contributes_to_candidate
            else (
                "incomplete_noncontributing"
                if provenance_gaps
                else "complete_noncontributing"
            )
        ),
        "provenance_gaps": list(provenance_gaps),
        "contributes_to_candidate": contributes_to_candidate,
        "superseded_by_attempt_uid": superseded_by_attempt_uid,
    }


def _assignment(method_id: str, seed: int, *, retain_recovery_history: bool) -> dict[str, object]:
    assignment_uid = f"operational_twin__{method_id}__seed_{seed}"
    if retain_recovery_history:
        failed_uid = f"{assignment_uid}__attempt_001"
        cancelled_uid = f"{assignment_uid}__attempt_002"
        terminal_attempt_uid = f"{assignment_uid}__attempt_003"
        attempts = [
            _attempt(
                assignment_uid=assignment_uid,
                method_id=method_id,
                seed=seed,
                ordinal=1,
                status="FAILED",
                contributes_to_candidate=False,
                superseded_by_attempt_uid=cancelled_uid,
            ),
            _attempt(
                assignment_uid=assignment_uid,
                method_id=method_id,
                seed=seed,
                ordinal=2,
                status="CANCELLED",
                contributes_to_candidate=False,
                superseded_by_attempt_uid=terminal_attempt_uid,
            ),
            _attempt(
                assignment_uid=assignment_uid,
                method_id=method_id,
                seed=seed,
                ordinal=3,
                status="RECOVERED",
                recovered_from_attempt_uid=failed_uid,
                contributes_to_candidate=True,
                superseded_by_attempt_uid=None,
            ),
        ]
        terminal_status = "RECOVERED"
    else:
        terminal_attempt_uid = f"{assignment_uid}__attempt_001"
        attempts = [
            _attempt(
                assignment_uid=assignment_uid,
                method_id=method_id,
                seed=seed,
                ordinal=1,
                status="COMPLETED",
                contributes_to_candidate=True,
                superseded_by_attempt_uid=None,
            )
        ]
        terminal_status = "COMPLETED"

    if METHOD_EXECUTION_CONTRACTS[method_id]["checkpoint_creation_allowed"]:
        checkpoint = {
            "contract": "checkpoint_identity",
            "repository_storage": "external_content_addressed",
            "path": f"checkpoints/{assignment_uid}/best_dev.pt",
            "sha256": canonical_hash(f"checkpoint:{assignment_uid}"),
            "size_bytes": 4096,
            "reason": None,
        }
    else:
        checkpoint = {
            "contract": "no_checkpoint",
            "repository_storage": "not_applicable",
            "path": None,
            "sha256": None,
            "size_bytes": 0,
            "reason": "nontrainable_method_contract",
        }

    return {
        "assignment_uid": assignment_uid,
        "method_id": method_id,
        "seed": seed,
        "terminal_status": terminal_status,
        "terminal_attempt_uid": terminal_attempt_uid,
        "evidence": {
            role: _source(f"evidence/assignments/{assignment_uid}/{role}.json")
            for role in ("terminal", "provenance", "evaluation", "reconciliation")
        },
        "checkpoint": checkpoint,
        "attempts": attempts,
        "final_evidence_attempt_references": {
            "terminal": [terminal_attempt_uid],
            "provenance": [],
            "evaluation": [],
            "reconciliation": [],
            "checkpoint": [],
        },
        "attempt_retention": {
            "source_attempt_count": len(attempts),
            "source_failed_count": sum(row["status"] == "FAILED" for row in attempts),
            "source_cancelled_count": sum(row["status"] == "CANCELLED" for row in attempts),
            "source_recovered_count": sum(
                row["status"] == "RECOVERED" or row["recovered_from_attempt_uid"] is not None
                for row in attempts
            ),
            "retention_complete": True,
        },
        "access_flags": _access_flags(),
        "claim_flags": _claim_flags(),
        "code_commits": {
            "training": COMMIT,
            "evaluation": COMMIT,
            "reconciliation": COMMIT,
        },
    }


def _ledger() -> tuple[dict[str, object], dict[str, object]]:
    protocol = _protocol()
    assignments = []
    for method_id in REQUIRED_METHOD_IDS:
        for seed in protocol["seeds"]:
            assignments.append(
                _assignment(
                    method_id,
                    seed,
                    retain_recovery_history=(method_id == REQUIRED_METHOD_IDS[0] and seed == protocol["seeds"][0]),
                )
            )
    receipt_groups: dict[str, list[dict[str, object]]] = {}
    for group in REQUIRED_RECEIPT_GROUPS:
        receipt: dict[str, object] = {
            "receipt_uid": f"final-{group}",
            "receipt_kind": (
                "scheduler" if group in SCHEDULER_RECEIPT_GROUPS else "process"
            ),
            "source": _source(f"evidence/final/{group}.txt"),
        }
        if group in SCHEDULER_RECEIPT_GROUPS:
            receipt["scheduler"] = _scheduler(f"final-{group}")
        else:
            receipt["code_commit"] = COMMIT
        receipt_groups[group] = [receipt]
    protocol_identity = {
        "protocol_uid": protocol["protocol_uid"],
        "protocol_schema_version": protocol["schema_version"],
        "protocol_semantic_sha256": canonical_hash(protocol),
        "sources": {
            "protocol": _source("Libs/configs/model/operational_twin_training_protocol.yaml"),
            "assignment_manifest": _source("evidence/protocol/assignment_manifest.json"),
            "preflight_authority": _source("evidence/protocol/preflight_authority.json"),
        },
    }
    ledger = build_operational_twin_training_evidence_ledger(
        protocol=protocol,
        protocol_identity=protocol_identity,
        assignments=assignments,
        receipt_groups=receipt_groups,
        access_flags=_access_flags(),
        claim_flags=_claim_flags(),
        code_commits={group: [COMMIT] for group in REQUIRED_RECEIPT_GROUPS},
        dependencies=[_source("evidence/catalog/candidate.json")],
    )
    return ledger, protocol


def _rebuild_ledger(
    ledger: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    return build_operational_twin_training_evidence_ledger(
        protocol=protocol,
        protocol_identity=ledger["protocol_identity"],
        assignments=ledger["assignments"],
        receipt_groups=ledger["receipt_groups"],
        access_flags=ledger["access_flags"],
        claim_flags=ledger["claim_flags"],
        code_commits=ledger["code_commits"],
        dependencies=ledger["dependencies"],
        validation_blockers=ledger["validation_blockers"],
    )


def _with_incomplete_historical_attempt(
    ledger: dict[str, object],
    protocol: dict[str, object],
) -> tuple[dict[str, object], str]:
    assignment = ledger["assignments"][0]
    assert isinstance(assignment, dict)
    attempt = assignment["attempts"][0]
    assert isinstance(attempt, dict)
    attempt_uid = str(attempt["attempt_uid"])
    attempt["command_receipt"] = None
    attempt["code_commit"] = None
    attempt["provenance_gaps"] = ["command_receipt", "code_commit"]
    attempt["provenance_disposition"] = "incomplete_noncontributing"
    attempt["contributes_to_candidate"] = False
    return _rebuild_ledger(ledger, protocol), attempt_uid


def _validation_codes(
    ledger: dict[str, object],
    protocol: dict[str, object],
) -> list[str]:
    return [
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    ]


def _rehash(ledger: dict[str, object]) -> None:
    ledger["ledger_sha256"] = compute_operational_twin_training_evidence_ledger_sha256(ledger)


def _rehash_attempt(attempt: dict[str, object]) -> None:
    payload = deepcopy(attempt)
    payload.pop("identity_sha256", None)
    attempt["identity_sha256"] = canonical_hash(payload)


def _set_recovered_command_receipt(
    ledger: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    assignment = ledger["assignments"][0]
    assert isinstance(assignment, dict)
    attempt = assignment["attempts"][0]
    assert isinstance(attempt, dict)
    receipt = ledger["receipt_groups"]["training"][0]
    assert isinstance(receipt, dict)
    scheduler = attempt["scheduler"]
    assert isinstance(scheduler, dict)
    receipt["scheduler"] = deepcopy(scheduler)
    submit_line = "srun python train.py --task-index 0"
    command_identity = {
        "algorithm": COMMAND_IDENTITY_ALGORITHM,
        "normalization": COMMAND_IDENTITY_NORMALIZATION,
        "source_field": COMMAND_IDENTITY_SOURCE_FIELD,
        "value": hashlib.sha256(submit_line.encode("utf-8")).hexdigest(),
    }
    receipt["command_identity"] = deepcopy(command_identity)
    attempt["command_receipt"] = {
        "receipt_uid": f"command__{attempt['attempt_uid']}",
        "source": deepcopy(receipt["source"]),
        "source_classification": SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION,
        "scheduler_job_id": scheduler["job_id"],
        "command_identity": command_identity,
    }
    _rehash_attempt(attempt)
    _rehash(ledger)
    return attempt, receipt


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verification_repository(
    tmp_path: Path,
    *,
    ledger: dict[str, object] | None,
    commit_ledger: bool,
) -> Path:
    root = tmp_path / "verification_repo"
    root.mkdir()
    schema_path = root / devcal_verifier.TRAINING_EVIDENCE_LEDGER_SCHEMA
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(SCHEMA_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (root / ".gitignore").write_text("/evidence/\n/checkpoints/\n", encoding="utf-8")

    ledger_path = root / devcal_verifier.DEFAULT_TRAINING_EVIDENCE_LEDGER
    if ledger is not None:
        _write_json(ledger_path, ledger)

    _run_git(root, "init", "-q")
    _run_git(root, "add", ".gitignore", schema_path.relative_to(root).as_posix())
    if commit_ledger:
        assert ledger is not None
        _run_git(root, "add", ledger_path.relative_to(root).as_posix())
    _run_git(
        root,
        "-c",
        "user.name=NeuronSifter Tests",
        "-c",
        "user.email=tests@neuronsifter.invalid",
        "commit",
        "-q",
        "-m",
        "Create verifier fixture",
    )
    return root


def _verify_formal_development_training(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    protocol: dict[str, object],
) -> dict[str, object]:
    monkeypatch.setattr(devcal_verifier, "resolve_project_root", lambda: root)
    monkeypatch.setattr(
        devcal_verifier,
        "load_config_stack",
        lambda _path: {"operational_twin_training_protocol": protocol},
    )
    return devcal_verifier.verify_formal_development_training(write_artifact=False)


def _blocker_codes(summary: dict[str, object]) -> list[str]:
    return [str(row["code"]) for row in summary["blockers"]]


def test_optional_runtime_jsonl_dependencies_are_conditional(tmp_path: Path) -> None:
    output_root = Path("Output/results/operational_twin/development_training/assignment")
    metrics_path = output_root / "metrics/method_metrics.jsonl"
    failure_path = output_root / "failures/method_failures.jsonl"
    assignment_source = {
        "output_root": output_root.as_posix(),
        "runtime_contract": {
            "metrics_path": "metrics/method_metrics.jsonl",
            "failure_retention_path": "failures/method_failures.jsonl",
        },
        "failure_evidence_required": False,
        "checkpoint_required": False,
        "provenance": {},
        "reconciliation": {},
    }

    candidates = evidence_builder._dependency_candidates(
        root=tmp_path,
        protocol={},
        assignment_sources=[assignment_source],
    )
    assert metrics_path.as_posix() not in candidates
    assert failure_path.as_posix() not in candidates

    (tmp_path / metrics_path).parent.mkdir(parents=True)
    (tmp_path / metrics_path).write_text("{}\n", encoding="utf-8")
    (tmp_path / failure_path).parent.mkdir(parents=True)
    (tmp_path / failure_path).write_text("{}\n", encoding="utf-8")
    candidates = evidence_builder._dependency_candidates(
        root=tmp_path,
        protocol={},
        assignment_sources=[assignment_source],
    )
    assert metrics_path.as_posix() in candidates
    assert failure_path.as_posix() in candidates

    (tmp_path / failure_path).unlink()
    assignment_source["failure_evidence_required"] = True
    candidates = evidence_builder._dependency_candidates(
        root=tmp_path,
        protocol={},
        assignment_sources=[assignment_source],
    )
    assert failure_path.as_posix() in candidates


def test_empty_retained_failure_summary_does_not_invent_missing_attempt(
    tmp_path: Path,
) -> None:
    failure_path = tmp_path / "failures.jsonl"
    failure_path.write_text(
        json.dumps({"retained_failures": []}) + "\n",
        encoding="utf-8",
    )
    blockers: list[dict[str, str]] = []

    evidence_builder._validate_failure_attempt_retention(
        failure_path,
        retained_attempt_uids=set(),
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
    )

    assert blockers == []


def test_retained_failure_summary_entries_remain_mandatory_and_fail_closed(
    tmp_path: Path,
) -> None:
    assignment_uid = "operational_twin__mechanism__seed_101"
    failed_attempt_uid = f"{assignment_uid}__attempt_001"
    failure_path = tmp_path / "failures.jsonl"
    failure_path.write_text(
        json.dumps(
            {
                "retained_failures": [
                    {
                        "attempt_uid": failed_attempt_uid,
                        "status": "FAILED",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    blockers: list[dict[str, str]] = []
    evidence_builder._validate_failure_attempt_retention(
        failure_path,
        retained_attempt_uids={failed_attempt_uid},
        blockers=blockers,
        assignment_uid=assignment_uid,
    )
    assert blockers == []

    evidence_builder._validate_failure_attempt_retention(
        failure_path,
        retained_attempt_uids=set(),
        blockers=blockers,
        assignment_uid=assignment_uid,
    )
    assert [row["code"] for row in blockers] == ["failure_attempt_source_missing"]


def test_missing_command_receipt_is_explicit_and_has_one_blocker() -> None:
    attempt_uid = "operational_twin__mechanism__seed_101__attempt_001"
    blockers: list[dict[str, str]] = []

    command_receipt = evidence_builder._command_receipt_entry(
        attempt_uid=attempt_uid,
        source=None,
        blockers=blockers,
    )

    assert command_receipt is None
    assert [row["code"] for row in blockers] == ["command_receipt_missing"]
    assert "missing_command_receipt__" not in blockers[0]["message"]


def test_scheduler_submit_line_recovers_command_identity_without_raw_text() -> None:
    attempt_uid = "operational_twin__linear_or_delta__seed_360307__attempt_002"
    submit_line = "srun python run_operational_twin_assignment_command.py --task-index 5"
    scheduler = _scheduler("535252")
    source = _source("evidence/receipts/formal_attempts_round37.sacct")
    command_identity = {
        "algorithm": COMMAND_IDENTITY_ALGORITHM,
        "normalization": COMMAND_IDENTITY_NORMALIZATION,
        "source_field": COMMAND_IDENTITY_SOURCE_FIELD,
        "value": hashlib.sha256(submit_line.encode("utf-8")).hexdigest(),
    }
    blockers: list[dict[str, str]] = []

    command_receipt = evidence_builder._command_receipt_entry(
        attempt_uid=attempt_uid,
        source=None,
        scheduler_receipt={
            "scheduler": scheduler,
            "source": source,
            "command_identity": command_identity,
            "submit_line": submit_line + "\n",
        },
        blockers=blockers,
    )

    assert blockers == []
    assert command_receipt == {
        "receipt_uid": f"command__{attempt_uid}",
        "source": source,
        "source_classification": SCHEDULER_ACCOUNTING_RECOVERY_CLASSIFICATION,
        "scheduler_job_id": scheduler["job_id"],
        "command_identity": command_identity,
    }
    assert submit_line not in command_receipt.values()


def test_empty_scheduler_submit_line_fails_closed() -> None:
    blockers: list[dict[str, str]] = []

    command_receipt = evidence_builder._command_receipt_entry(
        attempt_uid="operational_twin__linear_or_delta__seed_360307__attempt_002",
        source=None,
        scheduler_receipt={
            "scheduler": _scheduler("535252"),
            "source": _source("evidence/receipts/formal_attempts_round37.sacct"),
            "command_identity": None,
            "submit_line": "",
        },
        blockers=blockers,
    )

    assert command_receipt is None
    assert [row["code"] for row in blockers] == ["command_receipt_missing"]


def test_local_no_job_attempt_does_not_gain_a_command_receipt() -> None:
    scheduler = _scheduler("local-no-fit")
    scheduler["job_id"] = ""
    blockers: list[dict[str, str]] = []

    command_receipt = evidence_builder._command_receipt_entry(
        attempt_uid="operational_twin__persistence_no_action__seed_360101__attempt_002",
        source=None,
        scheduler_receipt={
            "scheduler": scheduler,
            "source": _source("evidence/receipts/formal_attempts_round37.sacct"),
            "command_identity": None,
            "submit_line": "",
        },
        blockers=blockers,
    )

    assert command_receipt is None
    assert [row["code"] for row in blockers] == ["command_receipt_missing"]


def test_reconciliation_commit_uses_retained_training_commit_and_rejects_invalid() -> None:
    assert evidence_builder._receipt_code_commit(
        "reconciliation",
        {"training_code_commit": COMMIT},
    ) == COMMIT
    assert evidence_builder._receipt_code_commit(
        "reconciliation",
        {"reconciliation_code_commit": COMMIT},
    ) == ""

    for value in (None, "not-a-full-commit"):
        blockers: list[dict[str, str]] = []
        commit = evidence_builder._assignment_commit(
            role="reconciliation",
            sources=(("reconciliation", value),),
            blockers=blockers,
            assignment_uid="operational_twin__mechanism__seed_101",
        )
        assert commit == ""
        assert [row["code"] for row in blockers] == [
            "assignment_reconciliation_commit_invalid"
        ]


def _receipt_collector_sources(
    tmp_path: Path,
    *,
    aggregation_text: str = "PASS\n",
    focused_validation_text: str = "PASS\n",
) -> tuple[dict[str, tuple[Path, ...]], set[str]]:
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    scheduler_header = (
        "JobIDRaw|JobName|Partition|NodeList|State|ExitCode|Elapsed|"
        "AllocTRES|ReqMem|AllocCPUS|SubmitLine\n"
    )
    scheduler_rows = {
        "training": (
            scheduler_header
            + "1001|NrnSift_train|cpu|cpu04|COMPLETED|0:0|00:00:01|"
            "cpu=1,mem=2G|2G|1|srun python train.py --task-index 0\n"
        ),
        "evaluation": (
            scheduler_header
            + "1002|NrnSift_eval|cpu|cpu04|COMPLETED|0:0|00:00:02|"
            "cpu=1,mem=2G|2G|1|srun python evaluate.py --task-index 0\n"
        ),
    }
    source_text = {
        **scheduler_rows,
        "reconciliation": json.dumps(
            {
                "receipt_uid": "reconciliation-process",
                "training_code_commit": COMMIT,
            }
        )
        + "\n",
        "aggregation": aggregation_text,
        "focused_validation": focused_validation_text,
    }
    source_paths: dict[str, tuple[Path, ...]] = {}
    committed_paths: set[str] = set()
    for group, text in source_text.items():
        relative = Path("receipts") / f"{group}.log"
        (tmp_path / relative).write_text(text, encoding="utf-8")
        source_paths[group] = (relative,)
        committed_paths.add(relative.as_posix())
    return source_paths, committed_paths


def test_receipt_collector_separates_scheduler_and_process_semantics(
    tmp_path: Path,
) -> None:
    source_paths, committed_paths = _receipt_collector_sources(tmp_path)
    blockers: list[dict[str, str]] = []

    groups, commits, training_scheduler_rows = evidence_builder._collect_receipt_groups(
        root=tmp_path,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    for group in SCHEDULER_RECEIPT_GROUPS:
        receipt = groups[group][0]
        assert receipt["receipt_kind"] == "scheduler"
        assert "scheduler" in receipt
        assert "code_commit" not in receipt
        assert commits[group] == set()
    reconciliation = groups["reconciliation"][0]
    assert reconciliation == {
        "receipt_uid": "reconciliation-process",
        "receipt_kind": "process",
        "source": reconciliation["source"],
        "code_commit": COMMIT,
    }
    assert "scheduler" not in reconciliation
    assert commits["reconciliation"] == {COMMIT}
    for group in ("aggregation", "focused_validation"):
        receipt = groups[group][0]
        assert receipt["receipt_kind"] == "process"
        assert receipt["code_commit"] == ""
        assert "scheduler" not in receipt
        assert commits[group] == set()
    assert len(training_scheduler_rows) == 1
    expected_command_identity = {
        "algorithm": COMMAND_IDENTITY_ALGORITHM,
        "normalization": COMMAND_IDENTITY_NORMALIZATION,
        "source_field": COMMAND_IDENTITY_SOURCE_FIELD,
        "value": hashlib.sha256(
            b"srun python train.py --task-index 0"
        ).hexdigest(),
    }
    assert groups["training"][0]["command_identity"] == expected_command_identity
    assert training_scheduler_rows[0]["command_identity"] == expected_command_identity
    assert training_scheduler_rows[0]["submit_line"] == (
        "srun python train.py --task-index 0"
    )
    assert training_scheduler_rows[0]["source"]["tracked"] is True
    command_blockers: list[dict[str, str]] = []
    recovered_command = evidence_builder._command_receipt_entry(
        attempt_uid="operational_twin__linear_or_delta__seed_360307__attempt_002",
        source=None,
        scheduler_receipt=training_scheduler_rows[0],
        blockers=command_blockers,
    )
    assert command_blockers == []
    assert recovered_command is not None
    assert recovered_command["source"] == groups["training"][0]["source"]
    assert recovered_command["scheduler_job_id"] == "1001"
    assert recovered_command["command_identity"] == expected_command_identity
    assert [row["code"] for row in blockers].count("receipt_source_untyped") == 2
    assert [row["code"] for row in blockers].count("receipt_code_commit_missing") == 2


def test_receipt_collector_rejects_unknown_process_kind(tmp_path: Path) -> None:
    source_paths, committed_paths = _receipt_collector_sources(
        tmp_path,
        aggregation_text=json.dumps(
            {
                "receipt_uid": "aggregation-process",
                "receipt_kind": "unknown",
                "code_commit": COMMIT,
            }
        )
        + "\n",
        focused_validation_text=json.dumps(
            {
                "receipt_uid": "focused-process",
                "receipt_kind": "process",
                "code_commit": COMMIT,
            }
        )
        + "\n",
    )
    blockers: list[dict[str, str]] = []

    groups, commits, _ = evidence_builder._collect_receipt_groups(
        root=tmp_path,
        committed_paths=committed_paths,
        changed_paths=set(),
        blockers=blockers,
        source_cache={},
        receipt_source_paths=source_paths,
    )

    assert groups["aggregation"][0]["receipt_kind"] == "unknown"
    assert groups["aggregation"][0]["code_commit"] == ""
    assert commits["aggregation"] == set()
    assert commits["focused_validation"] == set()
    assert {row["code"] for row in blockers} == {
        "receipt_kind_invalid",
        "receipt_code_commit_missing",
        "process_receipt_invalid",
        "focused_validation_contract_set_invalid",
    }


def test_authorized_legacy_nine_key_access_mapping_normalizes_to_frozen_policy() -> None:
    expected, normalization_authorized, blockers = _access_authority()
    legacy = {
        key: value
        for key, value in expected.items()
        if key != "sealed_material_access"
    }

    observed = evidence_builder._consistent_mapping(
        values=[
            ("terminal", legacy, True),
            ("evaluation", expected, True),
            ("attempt 1", legacy, True),
        ],
        expected=expected,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field="access_flags",
        legacy_access_normalization_authorized=normalization_authorized,
    )

    assert normalization_authorized is True
    assert observed == _access_flags()
    assert blockers == []


@pytest.mark.parametrize(
    ("protocol_tracked", "preflight_access_flags", "authority_code"),
    [
        (False, None, "access_policy_authority_untracked"),
        (
            True,
            {
                **_access_flags(),
                "sealed_material_access": 1,
            },
            "access_policy_authority_mismatch",
        ),
    ],
)
def test_legacy_access_mapping_is_rejected_without_matching_tracked_authority(
    protocol_tracked: bool,
    preflight_access_flags: dict[str, int] | None,
    authority_code: str,
) -> None:
    expected, normalization_authorized, blockers = _access_authority(
        protocol_tracked=protocol_tracked,
        preflight_access_flags=preflight_access_flags,
    )
    legacy = {
        key: value
        for key, value in expected.items()
        if key != "sealed_material_access"
    }

    observed = evidence_builder._consistent_mapping(
        values=[("terminal", legacy, True)],
        expected=expected,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field="access_flags",
        legacy_access_normalization_authorized=normalization_authorized,
    )
    codes = {row["code"] for row in blockers}

    assert normalization_authorized is False
    assert observed == legacy
    assert authority_code in codes
    assert "access_flags_policy_mismatch" in codes


@pytest.mark.parametrize(
    "invalid_access_flags",
    [
        {
            key: value
            for key, value in _access_flags().items()
            if key != "tool_access"
        },
        {
            **_access_flags(),
            "protected_holdout_access": 1,
        },
    ],
)
def test_arbitrary_missing_or_contradictory_access_keys_fail_closed(
    invalid_access_flags: dict[str, int],
) -> None:
    expected, normalization_authorized, blockers = _access_authority()

    observed = evidence_builder._consistent_mapping(
        values=[("terminal", invalid_access_flags, True)],
        expected=expected,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field="access_flags",
        legacy_access_normalization_authorized=normalization_authorized,
    )

    assert observed == invalid_access_flags
    assert [row["code"] for row in blockers] == ["access_flags_policy_mismatch"]


def test_non_authoritative_provenance_mapping_is_optional_but_must_agree() -> None:
    access_policy, normalization_authorized, blockers = _access_authority()
    legacy_access = {
        key: value
        for key, value in access_policy.items()
        if key != "sealed_material_access"
    }
    claim_policy = _claim_flags()

    access_flags = evidence_builder._consistent_mapping(
        values=[
            ("terminal", legacy_access, True),
            ("provenance", evidence_builder._MISSING_MAPPING, False),
            ("evaluation", access_policy, True),
            ("attempt 1", legacy_access, True),
        ],
        expected=access_policy,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field="access_flags",
        legacy_access_normalization_authorized=normalization_authorized,
    )
    claim_flags = evidence_builder._consistent_mapping(
        values=[
            ("terminal", claim_policy, True),
            ("provenance", evidence_builder._MISSING_MAPPING, False),
            ("evaluation", claim_policy, True),
            ("attempt 1", claim_policy, True),
        ],
        expected=claim_policy,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field="claim_flags",
    )

    assert access_flags == _access_flags()
    assert claim_flags == claim_policy
    assert blockers == []

    conflicting_claims = dict(claim_policy)
    conflicting_claims["sufficient_training"] = True
    evidence_builder._consistent_mapping(
        values=[
            ("terminal", claim_policy, True),
            ("provenance", conflicting_claims, False),
            ("evaluation", claim_policy, True),
        ],
        expected=claim_policy,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field="claim_flags",
    )
    codes = {row["code"] for row in blockers}
    assert "claim_flags_inconsistent" in codes
    assert "claim_flags_policy_mismatch" in codes


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("access_flags", _access_flags()),
        ("claim_flags", _claim_flags()),
    ],
)
def test_absence_of_all_authoritative_mappings_fails_closed(
    field: str,
    expected: dict[str, object],
) -> None:
    blockers: list[dict[str, str]] = []

    observed = evidence_builder._consistent_mapping(
        values=[
            ("terminal", evidence_builder._MISSING_MAPPING, True),
            ("provenance", expected, False),
            ("evaluation", evidence_builder._MISSING_MAPPING, True),
            ("attempt 1", evidence_builder._MISSING_MAPPING, True),
        ],
        expected=expected,
        blockers=blockers,
        assignment_uid="operational_twin__mechanism__seed_101",
        field=field,
        legacy_access_normalization_authorized=True,
    )

    assert observed == expected
    assert [row["code"] for row in blockers] == [
        f"{field}_authoritative_missing"
    ]


def test_ledger_builds_exact_matrix_and_passes_schema_and_semantic_validation() -> None:
    ledger, protocol = _ledger()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(ledger)
    assert ledger["matrix"] == {
        "method_ids": list(REQUIRED_METHOD_IDS),
        "seeds": protocol["seeds"],
        "assignment_count": 15,
    }
    assert len(ledger["assignments"]) == 15
    assert ledger["historical_provenance_exceptions"] == []
    assert ledger["historical_provenance_exception_count"] == 0
    assert ledger["all_attempt_provenance_complete"] is True
    assert ledger["terminal_contributing_provenance_complete"] is True
    first_assignment = ledger["assignments"][0]
    first_attempts = first_assignment["attempts"]
    assert [row["provenance_disposition"] for row in first_attempts] == [
        "complete_noncontributing",
        "complete_noncontributing",
        "complete_contributing",
    ]
    assert [row["contributes_to_candidate"] for row in first_attempts] == [
        False,
        False,
        True,
    ]
    assert [row["superseded_by_attempt_uid"] for row in first_attempts] == [
        first_attempts[1]["attempt_uid"],
        first_attempts[2]["attempt_uid"],
        None,
    ]
    assert first_assignment["final_evidence_attempt_references"] == {
        "terminal": [first_assignment["terminal_attempt_uid"]],
        "provenance": [],
        "evaluation": [],
        "reconciliation": [],
        "checkpoint": [],
    }
    assert validate_operational_twin_training_evidence_ledger(ledger, protocol) == []
    rebuilt, _ = _ledger()
    assert rebuilt == ledger


def test_attempt_commit_canonicalization_uses_null_for_unavailable_values() -> None:
    assert evidence_builder._optional_attempt_commit(None) is None
    assert evidence_builder._optional_attempt_commit("") is None
    assert evidence_builder._optional_attempt_commit("not-a-commit") is None
    assert evidence_builder._optional_attempt_commit(COMMIT) == COMMIT


def test_null_commit_is_allowed_only_for_incomplete_noncontributing_attempt() -> None:
    ledger, protocol = _ledger()
    ledger, attempt_uid = _with_incomplete_historical_attempt(ledger, protocol)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assignment = ledger["assignments"][0]
    attempt = next(
        row for row in assignment["attempts"] if row["attempt_uid"] == attempt_uid
    )

    assert attempt["code_commit"] is None
    assert attempt["provenance_disposition"] == "incomplete_noncontributing"
    assert "code_commit" in attempt["provenance_gaps"]
    Draft202012Validator(schema).validate(ledger)
    assert _validation_codes(ledger, protocol) == []

    complete_ledger, complete_protocol = _ledger()
    complete_attempt = complete_ledger["assignments"][0]["attempts"][0]
    complete_attempt["code_commit"] = None
    complete_ledger = _rebuild_ledger(complete_ledger, complete_protocol)
    assert list(Draft202012Validator(schema).iter_errors(complete_ledger))
    complete_codes = set(_validation_codes(complete_ledger, complete_protocol))
    assert {
        "attempt_provenance_gap_mismatch",
        "attempt_provenance_disposition_invalid",
    } <= complete_codes

    terminal_ledger, terminal_protocol = _ledger()
    terminal_assignment = terminal_ledger["assignments"][0]
    terminal_attempt = next(
        row
        for row in terminal_assignment["attempts"]
        if row["attempt_uid"] == terminal_assignment["terminal_attempt_uid"]
    )
    terminal_attempt["code_commit"] = None
    terminal_attempt["provenance_gaps"] = ["code_commit"]
    terminal_ledger = _rebuild_ledger(terminal_ledger, terminal_protocol)
    assert list(Draft202012Validator(schema).iter_errors(terminal_ledger))
    assert "terminal_contributing_provenance_incomplete" in _validation_codes(
        terminal_ledger,
        terminal_protocol,
    )


def test_recovered_scheduler_command_passes_schema_and_semantic_validation() -> None:
    ledger, protocol = _ledger()
    _set_recovered_command_receipt(ledger)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(ledger)
    assert validate_operational_twin_training_evidence_ledger(ledger, protocol) == []


def test_recovered_scheduler_command_rejects_job_mismatch() -> None:
    ledger, protocol = _ledger()
    attempt, _ = _set_recovered_command_receipt(ledger)
    command_receipt = attempt["command_receipt"]
    assert isinstance(command_receipt, dict)
    command_receipt["scheduler_job_id"] = "wrong-job"
    _rehash_attempt(attempt)
    _rehash(ledger)

    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert "command_receipt_job_mismatch" in codes


@pytest.mark.parametrize("digest", ["not-a-sha256", None])
def test_recovered_scheduler_command_rejects_missing_or_invalid_hash(
    digest: str | None,
) -> None:
    ledger, protocol = _ledger()
    attempt, _ = _set_recovered_command_receipt(ledger)
    command_receipt = attempt["command_receipt"]
    assert isinstance(command_receipt, dict)
    command_identity = command_receipt["command_identity"]
    assert isinstance(command_identity, dict)
    if digest is None:
        command_identity.pop("value")
    else:
        command_identity["value"] = digest
    _rehash_attempt(attempt)
    _rehash(ledger)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert list(Draft202012Validator(schema).iter_errors(ledger))
    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert "command_identity_sha256_invalid" in codes


def test_recovered_scheduler_command_rejects_different_valid_digest() -> None:
    ledger, protocol = _ledger()
    attempt, _ = _set_recovered_command_receipt(ledger)
    command_receipt = attempt["command_receipt"]
    assert isinstance(command_receipt, dict)
    command_identity = command_receipt["command_identity"]
    assert isinstance(command_identity, dict)
    expected_digest = command_identity["value"]
    assert isinstance(expected_digest, str)
    command_identity["value"] = "0" * 64 if expected_digest != "0" * 64 else "1" * 64
    _rehash_attempt(attempt)
    _rehash(ledger)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(ledger)
    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert "command_identity_sha256_invalid" not in codes
    assert "command_receipt_command_identity_mismatch" in codes


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("algorithm", "sha512", "command_identity_algorithm_invalid"),
        (
            "normalization",
            "trimmed shell command",
            "command_identity_normalization_invalid",
        ),
        ("source_field", "SubmitLine", "command_identity_source_field_invalid"),
    ],
)
def test_recovered_scheduler_command_rejects_wrong_identity_semantics(
    field: str,
    value: str,
    expected_code: str,
) -> None:
    ledger, protocol = _ledger()
    attempt, _ = _set_recovered_command_receipt(ledger)
    command_receipt = attempt["command_receipt"]
    assert isinstance(command_receipt, dict)
    command_identity = command_receipt["command_identity"]
    assert isinstance(command_identity, dict)
    command_identity[field] = value
    _rehash_attempt(attempt)
    _rehash(ledger)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert list(Draft202012Validator(schema).iter_errors(ledger))
    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert expected_code in codes
    assert "command_receipt_command_identity_mismatch" in codes


def test_recovered_scheduler_command_rejects_wrong_scheduler_source() -> None:
    ledger, protocol = _ledger()
    attempt, _ = _set_recovered_command_receipt(ledger)
    command_receipt = attempt["command_receipt"]
    assert isinstance(command_receipt, dict)
    command_receipt["source"] = _source("evidence/final/unrelated_scheduler.txt")
    _rehash_attempt(attempt)
    _rehash(ledger)

    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert "command_receipt_scheduler_source_mismatch" in codes


def test_scheduler_receipts_can_omit_row_commit_but_commit_group_is_mandatory() -> None:
    ledger, protocol = _ledger()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    for group in SCHEDULER_RECEIPT_GROUPS:
        assert "code_commit" not in ledger["receipt_groups"][group][0]
    Draft202012Validator(schema).validate(ledger)
    assert validate_operational_twin_training_evidence_ledger(ledger, protocol) == []

    ledger["code_commits"]["training"] = []
    _rehash(ledger)
    schema_errors = list(Draft202012Validator(schema).iter_errors(ledger))
    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert schema_errors
    assert "code_commit_group_missing" in codes
    assert "code_commit_invalid" not in codes


def test_process_receipts_require_producer_commit_without_scheduler_fields() -> None:
    ledger, protocol = _ledger()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    for group in PROCESS_RECEIPT_GROUPS:
        assert "scheduler" not in ledger["receipt_groups"][group][0]
    Draft202012Validator(schema).validate(ledger)
    assert validate_operational_twin_training_evidence_ledger(ledger, protocol) == []

    ledger["receipt_groups"]["aggregation"][0].pop("code_commit")
    _rehash(ledger)
    schema_errors = list(Draft202012Validator(schema).iter_errors(ledger))
    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert schema_errors
    assert "code_commit_invalid" in codes
    assert "scheduler_identity_missing" not in codes


@pytest.mark.parametrize(
    ("receipt_kind", "expected_code"),
    [
        (None, "receipt_kind_invalid"),
        ("unknown", "receipt_kind_invalid"),
        ("process", "receipt_kind_mismatch"),
    ],
)
def test_missing_unknown_or_group_mismatched_receipt_kind_fails_closed(
    receipt_kind: str | None,
    expected_code: str,
) -> None:
    ledger, protocol = _ledger()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    receipt = ledger["receipt_groups"]["training"][0]
    if receipt_kind is None:
        receipt.pop("receipt_kind")
    else:
        receipt["receipt_kind"] = receipt_kind
    _rehash(ledger)

    assert list(Draft202012Validator(schema).iter_errors(ledger))
    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert expected_code in codes


def test_receipt_kind_semantics_do_not_relax_scientific_access_or_seed_gates() -> None:
    ledger, protocol = _ledger()
    ledger["matrix"]["seeds"] = list(reversed(ledger["matrix"]["seeds"]))
    ledger["access_flags"]["protected_holdout_access"] = 1
    ledger["claim_flags"]["sufficient_training"] = True
    _rehash(ledger)

    codes = {
        row["code"]
        for row in validate_operational_twin_training_evidence_ledger(
            ledger,
            protocol,
        )
    }
    assert {"seed_matrix_mismatch", "nonzero_access_flag", "open_claim_flag"} <= codes


def test_missing_assignment_and_wrong_seed_matrix_fail_closed() -> None:
    ledger, protocol = _ledger()
    ledger["assignments"].pop()
    ledger["matrix"]["seeds"] = list(reversed(ledger["matrix"]["seeds"]))
    _rehash(ledger)
    codes = {row["code"] for row in validate_operational_twin_training_evidence_ledger(ledger, protocol)}
    assert "assignment_matrix_mismatch" in codes
    assert "seed_matrix_mismatch" in codes


def test_duplicate_attempt_identity_fails_closed() -> None:
    ledger, protocol = _ledger()
    assignment = ledger["assignments"][0]
    assignment["attempts"].append(deepcopy(assignment["attempts"][0]))
    assignment["attempt_retention"]["source_attempt_count"] += 1
    assignment["attempt_retention"]["source_failed_count"] += 1
    _rehash(ledger)
    codes = {row["code"] for row in validate_operational_twin_training_evidence_ledger(ledger, protocol)}
    assert "duplicate_attempt_identity" in codes


def test_omitted_failure_or_recovery_attempt_fails_retention_contract() -> None:
    ledger, protocol = _ledger()
    assignment = ledger["assignments"][0]
    assignment["attempts"] = [row for row in assignment["attempts"] if row["status"] != "FAILED"]
    _rehash(ledger)
    codes = {row["code"] for row in validate_operational_twin_training_evidence_ledger(ledger, protocol)}
    assert "attempt_retention_count_mismatch" in codes
    assert "recovery_source_missing" in codes

    ledger, protocol = _ledger()
    assignment = ledger["assignments"][0]
    assignment["attempts"] = [row for row in assignment["attempts"] if row["status"] != "RECOVERED"]
    _rehash(ledger)
    codes = {row["code"] for row in validate_operational_twin_training_evidence_ledger(ledger, protocol)}
    assert "attempt_retention_count_mismatch" in codes
    assert "terminal_attempt_missing" in codes


def test_missing_receipts_open_flags_checkpoint_gap_and_untracked_dependency_fail_closed() -> None:
    ledger, protocol = _ledger()
    ledger["receipt_groups"].pop("aggregation")
    ledger["access_flags"]["protected_holdout_access"] = 1
    ledger["claim_flags"]["sufficient_training"] = True
    trainable = next(
        row
        for row in ledger["assignments"]
        if METHOD_EXECUTION_CONTRACTS[row["method_id"]]["checkpoint_creation_allowed"]
    )
    trainable["checkpoint"] = {
        "contract": "no_checkpoint",
        "repository_storage": "not_applicable",
        "path": None,
        "sha256": None,
        "size_bytes": 0,
        "reason": "missing",
    }
    ledger["dependencies"][0]["tracked"] = False
    _rehash(ledger)
    codes = {row["code"] for row in validate_operational_twin_training_evidence_ledger(ledger, protocol)}
    assert {
        "receipt_group_missing",
        "nonzero_access_flag",
        "open_claim_flag",
        "trainable_checkpoint_missing",
        "untracked_dependency",
    } <= codes


def test_formal_verifier_blocks_when_canonical_ledger_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, protocol = _ledger()
    root = _verification_repository(tmp_path, ledger=None, commit_ledger=False)

    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    assert _blocker_codes(summary) == ["training_evidence_ledger_missing"]
    assert summary["assignment_count"] == 0
    assert summary["live_assignment_evidence_cross_check"]["status"] == "NOT_CHECKED"


def test_formal_verifier_blocks_when_ledger_exists_only_in_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=False)

    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    assert _blocker_codes(summary) == ["training_evidence_ledger_untracked"]
    assert summary["assignment_count"] == 0


def test_formal_verifier_propagates_schema_invalid_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    ledger["ledger_uid"] = "wrong_ledger_uid"
    _rehash(ledger)
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=True)

    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    assert "training_evidence_ledger_schema_invalid" in _blocker_codes(summary)
    assert "ledger_uid_invalid" in _blocker_codes(summary)


def test_formal_verifier_rejects_stale_self_excluding_ledger_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    ledger["protocol_identity"]["sources"]["protocol"]["path"] = (
        "evidence/protocol/frozen_protocol.yaml"
    )
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=True)

    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    assert _blocker_codes(summary).count("ledger_sha256_mismatch") == 1


def test_formal_verifier_propagates_semantic_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    ledger["dependencies"][0]["tracked"] = False
    _rehash(ledger)
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=True)

    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    assert "untracked_dependency" in _blocker_codes(summary)


def test_formal_verifier_requires_exact_protocol_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    ledger["protocol_identity"].update(
        {
            "protocol_uid": "different_protocol",
            "protocol_schema_version": "different_schema",
            "protocol_semantic_sha256": "0" * 64,
        }
    )
    _rehash(ledger)
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=True)

    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    assert {
        "protocol_uid_mismatch",
        "protocol_schema_version_mismatch",
        "protocol_semantic_sha256_mismatch",
    } <= set(_blocker_codes(summary))


def test_formal_verifier_passes_from_committed_ledger_without_live_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=True)

    assert _run_git(root, "status", "--short") == ""
    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "PASS"
    assert summary["blockers"] == []
    assert summary["assignment_count"] == 15
    assert summary["evidence_authority"] == "committed_training_evidence_ledger"
    assert summary["live_assignment_evidence_cross_check"] == {
        "status": "NOT_PRESENT",
        "present_source_count": 0,
        "drift_blocker_count": 0,
    }


def test_formal_verifier_rejects_live_assignment_and_checkpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, protocol = _ledger()
    root = _verification_repository(tmp_path, ledger=ledger, commit_ledger=True)
    assignment = ledger["assignments"][0]
    terminal_path = root / assignment["evidence"]["terminal"]["path"]
    terminal_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_path.write_bytes(b"drifted assignment evidence\n")

    trainable = next(
        row
        for row in ledger["assignments"]
        if row["checkpoint"]["contract"] == "checkpoint_identity"
    )
    checkpoint_path = root / trainable["checkpoint"]["path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"drifted checkpoint\n")

    assert _run_git(root, "status", "--short") == ""
    summary = _verify_formal_development_training(
        monkeypatch,
        root=root,
        protocol=protocol,
    )

    assert summary["status"] == "BLOCKED"
    cross_check = summary["live_assignment_evidence_cross_check"]
    assert cross_check == {
        "status": "BLOCKED",
        "present_source_count": 2,
        "drift_blocker_count": 2,
    }
    drift_messages = [
        row["message"]
        for row in summary["blockers"]
        if row["code"] == "live_assignment_evidence_drift"
    ]
    assert len(drift_messages) == 2
    assert any("terminal evidence" in message for message in drift_messages)
    assert any("external checkpoint" in message for message in drift_messages)
