"""Reconcile Round 37 training metadata without changing model checkpoints.

The original assignment attempts remain immutable.  This builder preserves the
pre-reconciliation terminal/provenance JSON, verifies seed and checkpoint
identity from independent retained sources, and writes a transparent metadata
reconciliation receipt.  It never retrains or selects a new checkpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_development_evidence import (  # noqa: E402
    training_budget_evidence,
)
from Libs.utils.evaluation.operational_twin_training_protocol import repo_path  # noqa: E402
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_receipt(
    logs_root: Path,
    status: Mapping[str, object],
    assignment_uid: str,
) -> dict[str, object]:
    scheduler = dict(status.get("scheduler_identity", {}))
    array_job_id = str(scheduler.get("slurm_array_job_id") or "")
    task_id = str(scheduler.get("slurm_array_task_id") or "")
    if not array_job_id or not task_id:
        raise ValueError(f"{assignment_uid} lacks array job/task identity for command receipt lookup.")
    candidates = sorted(logs_root.glob(f"*_{array_job_id}_{task_id}.out"))
    matches: list[dict[str, object]] = []
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines()[:5]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("selected_assignment") == assignment_uid:
                matches.append({"path": path, "payload": payload})
                break
    if len(matches) != 1:
        raise ValueError(
            f"{assignment_uid} command receipt lookup expected one match, observed {len(matches)}."
        )
    match = matches[0]
    payload = dict(match["payload"])
    return {
        "path": Path(match["path"]).relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(Path(match["path"])),
        "command_git_commit": str(payload.get("command_git_commit", "")),
        "method_id": str(payload.get("method_id", "")),
        "seed": int(payload.get("seed", -1)),
        "selected_assignment": str(payload.get("selected_assignment", "")),
        "slurm_array_task_id": str(payload.get("slurm_array_task_id", "")),
        "slurm_job_id": str(payload.get("slurm_job_id", "")),
    }


def _preserve_once(source: Path, destination: Path) -> str:
    if destination.is_file():
        return _sha256(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return _sha256(destination)


def reconcile(*, write: bool) -> dict[str, object]:
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    protocol = dict(config["operational_twin_training_protocol"])
    method_protocols = {
        str(row["method_id"]): dict(row) for row in protocol["methods"]
    }
    roots = dict(protocol["output_roots"])
    manifest = _read_json(repo_path(REPO_ROOT, roots["assignment_manifest"], must_exist=True))
    logs_root = repo_path(
        REPO_ROOT,
        Path(roots["assignment_root"]) / "logs/assignments",
        must_exist=True,
    )
    records: list[dict[str, object]] = []
    training_commits: set[str] = set()
    for assignment in manifest["assignments"]:
        assignment_uid = str(assignment["assignment_uid"])
        method_id = str(assignment["method_id"])
        seed = int(assignment["seed"])
        output_root = repo_path(REPO_ROOT, assignment["output_root"], must_exist=True)
        status_path = output_root / "terminal_status.json"
        provenance_path = output_root / "provenance/runtime_provenance.json"
        status = _read_json(status_path)
        provenance = _read_json(provenance_path)
        if status.get("status") != "COMPLETED":
            raise ValueError(f"{assignment_uid} is not terminal COMPLETED; reconciliation is closed.")
        for field, expected in (
            ("assignment_uid", assignment_uid),
            ("method_id", method_id),
            ("seed", seed),
        ):
            if status.get(field) != expected:
                raise ValueError(f"{assignment_uid} terminal status disagrees on {field}.")
        command = _command_receipt(logs_root, status, assignment_uid)
        if command["method_id"] != method_id or command["seed"] != seed:
            raise ValueError(f"{assignment_uid} command receipt identity mismatch.")
        training_commit = str(command["command_git_commit"])
        if len(training_commit) != 40:
            raise ValueError(f"{assignment_uid} command receipt lacks a full Git commit.")
        training_commits.add(training_commit)

        dev_rows = _read_jsonl(output_root / "metrics/dev_metrics.jsonl")
        budget_evidence = training_budget_evidence(
            status,
            method_protocols[method_id]["budget"],
            dev_rows,
        )
        if not budget_evidence["full_budget_completed"]:
            raise ValueError(f"{assignment_uid} did not exhaust a frozen training budget.")

        checkpoint_evidence: dict[str, object] | None = None
        if bool(dict(assignment["runtime_contract"])["checkpoint_creation_allowed"]):
            checkpoint_path = output_root / "checkpoints/best_dev.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("assignment_uid") != assignment_uid or int(checkpoint.get("seed", -1)) != seed:
                raise ValueError(f"{assignment_uid} checkpoint identity mismatch.")
            selected_epoch = int(checkpoint["epoch"])
            checkpoint_evidence = {
                "path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(checkpoint_path),
                "selected_checkpoint_epoch": selected_epoch,
            }
        else:
            selected_epoch = 0

        original_status_path = output_root / "attempts/original_terminal_status_before_evidence_reconciliation.json"
        original_provenance_path = output_root / "provenance/original_runtime_provenance_before_evidence_reconciliation.json"
        if write:
            original_status_hash = _preserve_once(status_path, original_status_path)
            original_provenance_hash = _preserve_once(provenance_path, original_provenance_path)
        else:
            original_status_hash = _sha256(
                original_status_path if original_status_path.is_file() else status_path
            )
            original_provenance_hash = _sha256(
                original_provenance_path if original_provenance_path.is_file() else provenance_path
            )

        reconciliation = {
            "schema_version": "neuronsifter-operational-twin-training-evidence-reconciliation-v1",
            "assignment_uid": assignment_uid,
            "method_id": method_id,
            "seed": seed,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
            "reason_codes": [
                "bind_seed_from_manifest_terminal_checkpoint_and_command_receipt",
                "separate_executed_epoch_count_from_selected_checkpoint_epoch",
            ],
            "training_code_commit": training_commit,
            "budget_completion_evidence": budget_evidence,
            "command_receipt": command,
            "checkpoint_evidence": checkpoint_evidence,
            "preserved_originals": {
                "terminal_status": {
                    "path": original_status_path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": original_status_hash,
                },
                "runtime_provenance": {
                    "path": original_provenance_path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": original_provenance_hash,
                },
            },
            "model_parameters_or_checkpoint_changed": False,
            "training_reexecuted": False,
        }
        status.update(
            {
                "epochs_completed": int(budget_evidence["observed_epochs_completed"]),
                "selected_checkpoint_epoch": selected_epoch,
                "training_code_commit": training_commit,
                "evidence_reconciliation_path": "provenance/evidence_reconciliation.json",
            }
        )
        provenance.update(
            {
                "seed": seed,
                "epochs_completed": int(budget_evidence["observed_epochs_completed"]),
                "selected_checkpoint_epoch": selected_epoch,
                "training_code_commit": training_commit,
                "budget_completion_evidence": budget_evidence,
                "evidence_reconciliation_path": "provenance/evidence_reconciliation.json",
            }
        )
        if provenance.get("git_commit") not in {None, training_commit}:
            raise ValueError(f"{assignment_uid} provenance Git commit disagrees with command receipt.")
        provenance["git_commit"] = training_commit

        if write:
            _write_json(status_path, status)
            _write_json(provenance_path, provenance)
            reconciliation["reconciled_terminal_status_sha256"] = _sha256(status_path)
            reconciliation["reconciled_runtime_provenance_sha256"] = _sha256(provenance_path)
            _write_json(output_root / "provenance/evidence_reconciliation.json", reconciliation)
        records.append(reconciliation)

    if len(training_commits) != 1:
        raise ValueError(f"Formal assignments used multiple training commits: {sorted(training_commits)}")
    summary = {
        "schema_version": "neuronsifter-operational-twin-training-evidence-reconciliation-summary-v1",
        "status": "PASS",
        "write_executed": write,
        "assignment_count": len(records),
        "training_code_commit": next(iter(training_commits)),
        "records": records,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
    }
    if write:
        destination = repo_path(REPO_ROOT, roots["catalog_root"], must_exist=False)
        _write_json(destination / "training_evidence_reconciliation.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = reconcile(write=args.write)
    print(
        json.dumps(
            {
                "status": result["status"],
                "assignment_count": result["assignment_count"],
                "training_code_commit": result["training_code_commit"],
                "write_executed": result["write_executed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
