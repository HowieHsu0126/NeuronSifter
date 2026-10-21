from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from Libs.scripts.builders.refresh_operational_twin_scheduler_receipts import (
    DEFAULT_EVALUATION_RECEIPT,
    DEFAULT_TRAINING_RECEIPT,
    ReceiptRefreshError,
    SACCT_FIELDS,
    _build_parser,
    _parse_sacct_output,
    collect_evaluation_job_ids,
    refresh_scheduler_receipts,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(
    tmp_path: Path, *, evaluation_count: int = 15
) -> tuple[Path, Path, tuple[str, ...], tuple[str, ...]]:
    repo_root = tmp_path / "repo"
    assignments: list[dict[str, object]] = []
    training_ids: list[str] = []
    evaluation_ids: list[str] = []
    for index in range(evaluation_count):
        assignment_uid = f"operational_twin__fixture_method__seed_{410000 + index}"
        output_root = (
            Path("Output/results/operational_twin/development_training")
            / "fixture_method"
            / f"seed_{410000 + index}"
        )
        assignments.append(
            {
                "assignment_uid": assignment_uid,
                "output_root": output_root.as_posix(),
            }
        )
        attempt_path = (
            repo_root
            / output_root
            / "attempts"
            / f"{assignment_uid}__attempt_001.json"
        )
        training_id = f"7000_{index}"
        training_ids.append(training_id)
        _write_json(
            attempt_path,
            {
                "assignment_uid": assignment_uid,
                "attempt_uid": attempt_path.stem,
                "scheduler_identity": {
                    "slurm_job_id": str(7100 + index),
                    "slurm_array_job_id": "7000",
                    "slurm_array_task_id": str(index),
                },
                "status": "FAILED" if index == 0 else "COMPLETED",
            },
        )
        evaluation_id = f"8000_{index}"
        evaluation_ids.append(evaluation_id)
        _write_json(
            repo_root / output_root / "evaluation/summary.json",
            {
                "assignment_uid": assignment_uid,
                "device_provenance": {
                    "slurm_job_id": str(8100 + index),
                    "slurm_array_job_id": "8000",
                    "slurm_array_task_id": str(index),
                },
            },
        )

    if evaluation_count:
        first_assignment = assignments[0]
        first_uid = str(first_assignment["assignment_uid"])
        first_root = repo_root / str(first_assignment["output_root"])
        empty_path = (
            first_root / "attempts" / f"{first_uid}__attempt_002.json"
        )
        _write_json(
            empty_path,
            {
                "assignment_uid": first_uid,
                "attempt_uid": empty_path.stem,
                "scheduler_identity": {
                    "slurm_job_id": None,
                    "slurm_array_job_id": None,
                    "slurm_array_task_id": None,
                },
            },
        )
        standalone_path = (
            first_root / "attempts" / f"{first_uid}__attempt_003.json"
        )
        _write_json(
            standalone_path,
            {
                "assignment_uid": first_uid,
                "attempt_uid": standalone_path.stem,
                "scheduler_identity": {"slurm_job_id": "6999"},
                "status": "CANCELLED",
            },
        )
        training_ids.append("6999")

    manifest_path = repo_root / "assignment_manifest.json"
    _write_json(manifest_path, {"assignments": assignments})
    current_receipt = repo_root / "current_evaluation_receipt.psv"
    receipt_lines = ["JobID|JobName|State"]
    for job_id in reversed(evaluation_ids):
        receipt_lines.append(f"{job_id}|evaluation|COMPLETED")
        receipt_lines.append(f"{job_id}.batch|batch|COMPLETED")
    current_receipt.write_text("\n".join(receipt_lines) + "\n", encoding="utf-8")
    return (
        repo_root,
        manifest_path,
        tuple(training_ids),
        tuple(evaluation_ids),
    )


def _sacct_row(job_id: str, *, submit_line: str | None = None) -> str:
    values = {
        "JobID": job_id,
        "JobName": f"job_{job_id}",
        "Partition": "cpu",
        "NodeList": "cpu04",
        "State": "COMPLETED",
        "ExitCode": "0:0",
        "Elapsed": "00:00:01",
        "AllocTRES": "billing=2,cpu=2,node=1",
        "ReqTRES": "billing=2,cpu=2,mem=4G,node=1",
        "ReqMem": "4Gn",
        "AllocCPUS": "2",
        "Start": "2026-09-05T12:00:00",
        "End": "2026-09-05T12:00:01",
        "WorkDir": "/work/operational twin",
        "SubmitLine": submit_line
        or f"sbatch --wrap='python train.py --job-id {job_id}'",
    }
    return "|".join(values[field] for field in SACCT_FIELDS)


def _sacct_text(job_ids: tuple[str, ...]) -> str:
    rows = [_sacct_row(job_id) for job_id in reversed(job_ids)]
    return "|".join(SACCT_FIELDS) + "\n" + "\n".join(rows) + "\n"


def test_refresh_writes_sorted_complete_receipts_without_running_slurm(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path, training_ids, evaluation_ids = _build_fixture(
        tmp_path
    )
    current_receipt = repo_root / "current_evaluation_receipt.psv"
    training_output = repo_root / "refreshed_training.psv"
    evaluation_output = repo_root / "refreshed_evaluation.psv"
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
        }
        assert command[1] == "-X"
        assert command[3] == "--parsable2"
        assert command[4] == f"--format={','.join(SACCT_FIELDS)}"
        requested = tuple(command[2].removeprefix("--jobs=").split(","))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_sacct_text(requested),
            stderr="",
        )

    result = refresh_scheduler_receipts(
        repo_root=repo_root,
        assignment_manifest=manifest_path,
        current_evaluation_receipt=current_receipt,
        training_output=training_output,
        evaluation_output=evaluation_output,
        runner=fake_runner,
    )

    expected_training = ("6999",) + tuple(f"7000_{index}" for index in range(15))
    expected_evaluation = tuple(f"8000_{index}" for index in range(15))
    assert result["training_job_count"] == 16
    assert result["evaluation_job_count"] == 15
    assert len(calls) == 2
    assert tuple(calls[0][2].removeprefix("--jobs=").split(",")) == expected_training
    assert tuple(calls[1][2].removeprefix("--jobs=").split(",")) == expected_evaluation
    assert set(training_ids) == set(expected_training)
    assert set(evaluation_ids) == set(expected_evaluation)

    training_lines = training_output.read_text(encoding="utf-8").splitlines()
    evaluation_lines = evaluation_output.read_text(encoding="utf-8").splitlines()
    assert training_lines[0] == "|".join(SACCT_FIELDS)
    assert evaluation_lines[0] == "|".join(SACCT_FIELDS)
    assert [line.split("|", 1)[0] for line in training_lines[1:]] == list(
        expected_training
    )
    assert [line.split("|", 1)[0] for line in evaluation_lines[1:]] == list(
        expected_evaluation
    )
    assert training_lines[1].endswith(
        "sbatch --wrap='python train.py --job-id 6999'"
    )
    assert "retrieved" not in training_lines[0].lower()
    assert "retrieved" not in evaluation_lines[0].lower()


@pytest.mark.parametrize("case", ["missing", "duplicate", "unexpected"])
def test_sacct_allocation_rows_must_match_requested_job_ids_exactly(
    case: str,
) -> None:
    requested = ("101", "102")
    observed = {
        "missing": ("101",),
        "duplicate": ("101", "101", "102"),
        "unexpected": ("101", "102", "103"),
    }[case]
    text = "|".join(SACCT_FIELDS) + "\n"
    text += "\n".join(_sacct_row(job_id) for job_id in observed) + "\n"
    with pytest.raises(ReceiptRefreshError, match="do not match requested JobIDs"):
        _parse_sacct_output(text, requested_job_ids=requested)


@pytest.mark.parametrize(
    "submit_line",
    [
        "sbatch --wrap='python train.py | tee result.log'",
        "sbatch --wrap='python train.py\nprintf done'",
    ],
)
def test_sacct_output_rejects_pipe_or_newline_in_values(submit_line: str) -> None:
    text = "|".join(SACCT_FIELDS) + "\n" + _sacct_row(
        "101", submit_line=submit_line
    ) + "\n"
    with pytest.raises(ReceiptRefreshError, match="malformed row"):
        _parse_sacct_output(text, requested_job_ids=("101",))


def test_evaluation_identities_require_exactly_fifteen_assignments(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path, _, _ = _build_fixture(
        tmp_path, evaluation_count=14
    )
    with pytest.raises(ReceiptRefreshError, match="exactly 15 assignments"):
        collect_evaluation_job_ids(
            repo_root=repo_root,
            assignment_manifest=manifest_path,
            current_evaluation_receipt=(
                repo_root / "current_evaluation_receipt.psv"
            ),
        )


def test_evaluation_evidence_must_match_current_receipt(tmp_path: Path) -> None:
    repo_root, manifest_path, _, _ = _build_fixture(tmp_path)
    current_receipt = repo_root / "current_evaluation_receipt.psv"
    current_receipt.write_text(
        current_receipt.read_text(encoding="utf-8").replace(
            "8000_14|evaluation", "9000_14|evaluation"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReceiptRefreshError, match="identities disagree"):
        collect_evaluation_job_ids(
            repo_root=repo_root,
            assignment_manifest=manifest_path,
            current_evaluation_receipt=current_receipt,
        )


def test_cli_defaults_target_tracked_receipt_paths() -> None:
    args = _build_parser().parse_args([])
    assert args.training_output == DEFAULT_TRAINING_RECEIPT
    assert args.evaluation_output == DEFAULT_EVALUATION_RECEIPT
    assert args.evaluation_identity_receipt == DEFAULT_EVALUATION_RECEIPT
