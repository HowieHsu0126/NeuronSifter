"""Search registered historical roots for immutable Round-21 authority evidence.

The builder implements Round-22A WP1 only. It searches exact authority
filenames in a closed config-owned root registry, hashes each unique candidate
once, and freezes an inventory. It never writes canonical Round-21 authority,
reconstructs a wrapper, launches replay, or changes downstream gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    canonical_sha256,
    load_contract,
)
from Libs.scripts.builders.build_round22r_empirical_prerequisites import (  # noqa: E402
    AUTHORITY_CANDIDATES,
    C2B_BLOCKED,
    EXPECTED_C1_GATE,
    EXPECTED_C2S_GATE,
    REQUIRED_AUTHORITY_INPUTS,
    _write_immutable_tsv,
)


ROUND22A_ID = "neuronsifter_round22a"
ALLOWED_COMPUTE_NODES = {"cpu03", "cpu04"}
EPHEMERAL_PREFIXES = (Path("/dev/shm"), Path("/tmp"), Path("/var/tmp"))
OUTPUTS = {
    "inventory": "Output/audits/round22a_authority_candidate_inventory.tsv",
    "search": "Output/audits/round22a_round21_recovery_search.json",
    "scheduler_snapshot": "Output/audits/round22a_scheduler_accounting_snapshot.json",
    "scheduler": "Output/audits/round22a_scheduler_workdir_reconciliation.json",
}
EXPECTED_ROOTS = {
    "git_object_database": (".", "git_exact_path_history"),
    "registered_git_worktrees": (".", "git_registered_worktrees_exact_path"),
    "round21_audit_archives": ("Output/audits", "filesystem_exact_filename"),
    "round21_paper_report_archives": (
        "Output/paper_reports",
        "filesystem_exact_filename",
    ),
    "round21_iclr_delivery_bundles": (
        "Output/outputs/iclr2027",
        "filesystem_exact_filename",
    ),
    "round21_job_archives": ("Output/logs", "filesystem_exact_filename"),
    "round21_result_archives": ("Output/results", "filesystem_exact_filename"),
}
INVENTORY_FIELDS = (
    "authority_uid",
    "candidate_kind",
    "candidate_path",
    "source_class",
    "source_root_id",
    "byte_size",
    "sha256",
    "git_commit_if_known",
    "git_blob_if_known",
    "producer_if_known",
    "schema_status",
    "reference_status",
    "recovery_confidence",
    "source_occurrence_count",
    "all_candidate_paths_json",
    "all_source_classes_json",
    "all_git_commits_json",
)
JSON_HASH_FIELDS = {
    "round21_training_input_freeze": (
        "model_config_hash",
        "c1_data_hash",
        "c2s_data_hash",
        "training_split_hash",
        "validation_split_hash",
        "final_holdout_split_hash",
        "environment_hash",
        "tri_input_schema_hash",
    ),
    "round21_static_twin_evidence": (
        "selected_checkpoint_hash",
        "model_config_hash",
        "c1_data_hash",
        "c1_split_hash",
        "c2s_data_hash",
        "c2s_split_hash",
        "environment_hash",
        "baseline_frontier_hash",
        "tri_input_schema_hash",
        "c1_claim_gate_hash",
        "c2s_claim_gate_hash",
    ),
    "round21_c1_claim_gate": (
        "config_hash",
        "data_hash",
        "split_hash",
        "checkpoint_hash",
    ),
    "round21_c2s_claim_gate": (
        "config_hash",
        "data_hash",
        "split_hash",
        "checkpoint_hash",
    ),
}
TSV_REQUIRED_FIELDS = {
    "schema_version",
    "metric_id",
    "value",
    "status",
    "producer_uid",
    "git_commit",
    "git_tree",
    "git_dirty",
    "config_hash",
    "data_hash",
    "split_hash",
    "checkpoint_hash",
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha(value: object) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_false(value: object) -> bool:
    return value is False or str(value).strip().lower() == "false"


def _run_git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _git_lines(root: Path, *args: str, check: bool = True) -> list[str]:
    completed = _run_git(root, *args, check=check)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _git_blob(root: Path, commit: str, relative_path: str) -> tuple[str, bytes] | None:
    object_name = f"{commit}:{relative_path}"
    resolved = _run_git(root, "rev-parse", "--verify", object_name, check=False)
    if resolved.returncode != 0:
        return None
    blob_uid = resolved.stdout.strip()
    content = subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", blob_uid],
        check=True,
        capture_output=True,
    ).stdout
    return blob_uid, content


def _metadata_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(payload.get("schema_version", "")).strip():
        errors.append("schema_version_missing")
    if not str(payload.get("artifact_uid", "")).strip():
        errors.append("artifact_uid_missing")
    producer = payload.get("producer")
    if not isinstance(producer, Mapping) or not str(producer.get("uid", "")).strip():
        errors.append("producer_identity_missing")
    git_identity = payload.get("git_identity")
    if not isinstance(git_identity, Mapping):
        errors.append("git_identity_missing")
    else:
        if (
            not str(git_identity.get("commit", "")).strip()
            or not str(git_identity.get("tree", "")).strip()
        ):
            errors.append("git_commit_or_tree_missing")
        if git_identity.get("dirty") is not False:
            errors.append("git_identity_not_clean")
    if not str(payload.get("status", "")).strip():
        errors.append("status_missing")
    return errors


def _validate_json_bytes(
    authority_uid: str, content: bytes
) -> tuple[str, str, Mapping[str, Any] | None]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return f"invalid_json:{type(exc).__name__}", "", None
    if not isinstance(payload, Mapping):
        return "invalid_json_root", "", None
    errors = _metadata_errors(payload)
    for field in JSON_HASH_FIELDS.get(authority_uid, ()):
        if not _is_sha(payload.get(field)):
            errors.append(f"{field}_missing_or_invalid")
    if authority_uid == "round21_checkpoint_catalog":
        checkpoints = payload.get("checkpoints")
        validation_rows = (
            [
                row
                for row in checkpoints
                if isinstance(row, Mapping)
                and row.get("selection_split") == "validation"
            ]
            if isinstance(checkpoints, list)
            else []
        )
        if not validation_rows:
            errors.append("validation_checkpoint_missing")
        for row in validation_rows:
            if not str(row.get("checkpoint_path", "")).strip() or not _is_sha(
                row.get("checkpoint_sha256")
            ):
                errors.append("validation_checkpoint_reference_invalid")
    elif authority_uid == "round21_c3_readiness":
        if (
            payload.get("static_twin_ready") is not True
            or payload.get("hidden_outcomes_unseen") is not True
        ):
            errors.append("c3_readiness_fields_invalid")
    elif authority_uid == "round21_coevolution_prerequisite_status":
        if payload.get("c1_gate") != EXPECTED_C1_GATE:
            errors.append("c1_gate_identity_mismatch")
        if payload.get("c2s_gate") != EXPECTED_C2S_GATE:
            errors.append("c2s_gate_identity_mismatch")
        if payload.get("c2_b_status") != C2B_BLOCKED:
            errors.append("c2_b_status_drift")
    elif authority_uid in {"round21_c1_claim_gate", "round21_c2s_claim_gate"}:
        expected = (
            EXPECTED_C1_GATE
            if authority_uid == "round21_c1_claim_gate"
            else EXPECTED_C2S_GATE
        )
        if payload.get("gate") != expected or payload.get("passed") is not True:
            errors.append("claim_gate_state_invalid")
    producer = payload.get("producer")
    producer_uid = str(producer.get("uid", "")) if isinstance(producer, Mapping) else ""
    return (
        ("valid" if not errors else "invalid:" + ",".join(sorted(set(errors)))),
        producer_uid,
        payload,
    )


def _validate_tsv_bytes(content: bytes) -> tuple[str, str]:
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter="\t")
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        return f"invalid_tsv:{type(exc).__name__}", ""
    errors: list[str] = []
    if TSV_REQUIRED_FIELDS - fields:
        errors.append("result_schema_incomplete")
    if not rows:
        errors.append("result_rows_empty")
    for row in rows:
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            errors.append("result_value_nonfinite")
        if not _strict_false(row.get("git_dirty")):
            errors.append("result_git_dirty")
        if any(
            not _is_sha(row.get(key))
            for key in ("config_hash", "data_hash", "split_hash", "checkpoint_hash")
        ):
            errors.append("result_identity_hash_missing")
    producers = {
        str(row.get("producer_uid", "")).strip()
        for row in rows
        if str(row.get("producer_uid", "")).strip()
    }
    producer = next(iter(producers)) if len(producers) == 1 else ""
    return (
        "valid" if not errors else "invalid:" + ",".join(sorted(set(errors)))
    ), producer


def _validate_candidate(
    authority_uid: str, content: bytes
) -> tuple[str, str, Mapping[str, Any] | None]:
    if REQUIRED_AUTHORITY_INPUTS[authority_uid].endswith(".json"):
        return _validate_json_bytes(authority_uid, content)
    status, producer = _validate_tsv_bytes(content)
    return status, producer, None


def _relative_checkpoint_path(path_value: object, root: Path) -> str | None:
    if not str(path_value or "").strip():
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _git_reference_status(
    authority_uid: str,
    payload: Mapping[str, Any] | None,
    root: Path,
    commit: str,
) -> str:
    if authority_uid != "round21_checkpoint_catalog":
        return "pending_cross_artifact_validation"
    checkpoints = payload.get("checkpoints") if isinstance(payload, Mapping) else None
    rows = (
        [
            row
            for row in checkpoints
            if isinstance(row, Mapping) and row.get("selection_split") == "validation"
        ]
        if isinstance(checkpoints, list)
        else []
    )
    if not rows:
        return "invalid_validation_checkpoint_reference"
    for row in rows:
        relative = _relative_checkpoint_path(row.get("checkpoint_path"), root)
        if relative is None:
            return "external_checkpoint_reference_unresolved"
        candidate = _git_blob(root, commit, relative)
        if candidate is None or _sha256_bytes(candidate[1]) != str(
            row.get("checkpoint_sha256", "")
        ):
            return "checkpoint_reference_missing_or_hash_mismatch"
    return "validation_checkpoint_reference_verified"


class CandidateCollector:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._path_hashes: dict[str, str] = {}
        self.hash_reads = 0

    def add(
        self,
        *,
        authority_uid: str,
        candidate_kind: str,
        candidate_path: str,
        source_class: str,
        source_root_id: str,
        content: bytes,
        git_commit: str = "",
        git_blob: str = "",
        producer: str = "",
        schema_status: str,
        reference_status: str,
        recovery_confidence: str,
    ) -> None:
        digest = self._path_hashes.get(candidate_path)
        if digest is None:
            digest = _sha256_bytes(content)
            self._path_hashes[candidate_path] = digest
            self.hash_reads += 1
        key = (authority_uid, candidate_kind, digest)
        row = self._records.get(key)
        if row is None:
            row = {
                "authority_uid": authority_uid,
                "candidate_kind": candidate_kind,
                "candidate_path": candidate_path,
                "source_class": source_class,
                "source_root_id": source_root_id,
                "byte_size": len(content),
                "sha256": digest,
                "git_commit_if_known": git_commit,
                "git_blob_if_known": git_blob,
                "producer_if_known": producer,
                "schema_status": schema_status,
                "reference_status": reference_status,
                "recovery_confidence": recovery_confidence,
                "candidate_paths": {candidate_path},
                "source_classes": {source_class},
                "git_commits": {git_commit} if git_commit else set(),
            }
            self._records[key] = row
            return
        row["candidate_paths"].add(candidate_path)
        row["source_classes"].add(source_class)
        if git_commit:
            row["git_commits"].add(git_commit)
        if row["recovery_confidence"] != "high" and recovery_confidence == "high":
            row["recovery_confidence"] = "high"
            row["candidate_path"] = candidate_path
            row["source_class"] = source_class
            row["source_root_id"] = source_root_id
            row["git_commit_if_known"] = git_commit
            row["git_blob_if_known"] = git_blob
            row["reference_status"] = reference_status

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in self._records.values():
            paths = sorted(row.pop("candidate_paths"))
            source_classes = sorted(row.pop("source_classes"))
            commits = sorted(row.pop("git_commits"))
            rows.append(
                row
                | {
                    "source_occurrence_count": len(paths),
                    "all_candidate_paths_json": json.dumps(
                        paths, ensure_ascii=True, separators=(",", ":")
                    ),
                    "all_source_classes_json": json.dumps(
                        source_classes, ensure_ascii=True, separators=(",", ":")
                    ),
                    "all_git_commits_json": json.dumps(
                        commits, ensure_ascii=True, separators=(",", ":")
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                str(row["authority_uid"]),
                str(row["candidate_kind"]),
                str(row["sha256"]),
            ),
        )


def _commit_candidates(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    reachable = set(_git_lines(root, "rev-list", "--all"))
    reflog = set(_git_lines(root, "rev-list", "--reflog"))
    candidates: set[str] = set()
    for path in REQUIRED_AUTHORITY_INPUTS.values():
        candidates.update(
            _git_lines(
                root, "log", "--all", "--full-history", "--format=%H", "--", path
            )
        )
        candidates.update(
            _git_lines(
                root, "log", "--reflog", "--full-history", "--format=%H", "--", path
            )
        )
    candidates.update(_git_lines(root, "for-each-ref", "--format=%(objectname)"))
    head = _git_lines(root, "rev-parse", "HEAD")
    candidates.update(head)

    fsck = _run_git(
        root,
        "fsck",
        "--full",
        "--unreachable",
        "--no-reflogs",
        "--no-progress",
        check=False,
    )
    if fsck.returncode != 0:
        detail = fsck.stderr.strip() or fsck.stdout.strip()
        raise RuntimeError(f"git object database integrity check failed: {detail}")
    unreachable: set[str] = set()
    for line in (fsck.stdout + "\n" + fsck.stderr).splitlines():
        fields = line.split()
        if (
            len(fields) >= 3
            and fields[0] in {"unreachable", "dangling"}
            and fields[1] == "commit"
        ):
            unreachable.add(fields[2])
    candidates.update(unreachable)
    classification: dict[str, str] = {}
    for commit in candidates:
        if commit in reachable:
            classification[commit] = "git_reachable_history"
        elif commit in reflog:
            classification[commit] = "git_reflog_history"
        else:
            classification[commit] = "git_unreachable_commit"
    counts = {
        "reachable_commit_count": len(reachable),
        "reflog_only_commit_count": len(reflog - reachable),
        "unreachable_commit_count": len(unreachable),
        "candidate_snapshot_count": len(candidates),
        "git_fsck_exit_code": fsck.returncode,
    }
    return classification, counts


def _search_git_history(
    root: Path,
    root_spec: Mapping[str, Any],
    collector: CandidateCollector,
) -> tuple[dict[str, object], list[str]]:
    classification, counts = _commit_candidates(root)
    examined_commit_count = (
        int(counts["reachable_commit_count"])
        + int(counts["reflog_only_commit_count"])
        + int(counts["unreachable_commit_count"])
    )
    if examined_commit_count > int(root_spec["max_entries"]):
        raise RuntimeError(
            f"registered Git root entry cap exceeded: {examined_commit_count} > {root_spec['max_entries']}"
        )
    snapshot_presence: dict[str, set[str]] = {}
    for commit in sorted(classification):
        for authority_uid, relative_path in REQUIRED_AUTHORITY_INPUTS.items():
            candidate = _git_blob(root, commit, relative_path)
            if candidate is None:
                continue
            blob_uid, content = candidate
            schema_status, producer, payload = _validate_candidate(
                authority_uid, content
            )
            reference_status = _git_reference_status(
                authority_uid, payload, root, commit
            )
            valid = schema_status == "valid"
            if valid:
                snapshot_presence.setdefault(commit, set()).add(authority_uid)
            confidence = (
                "high"
                if valid
                and not reference_status.startswith(
                    ("invalid", "checkpoint_reference_missing")
                )
                else "medium"
            )
            collector.add(
                authority_uid=authority_uid,
                candidate_kind="authority_artifact",
                candidate_path=f"git:{commit}:{relative_path}",
                source_class=classification[commit],
                source_root_id=str(root_spec["root_id"]),
                content=content,
                git_commit=commit,
                git_blob=blob_uid,
                producer=producer,
                schema_status=schema_status,
                reference_status=reference_status,
                recovery_confidence=confidence,
            )
    complete_snapshots = sorted(
        commit
        for commit, present in snapshot_presence.items()
        if present == set(REQUIRED_AUTHORITY_INPUTS)
    )
    report = {
        "root_id": root_spec["root_id"],
        "path": root_spec["path"],
        "resolver": root_spec["resolver"],
        "source_classes": root_spec["source_classes"],
        "status": "searched",
        "complete": True,
        "bounded": True,
        "exact_filename_only": True,
        **counts,
        "complete_schema_valid_snapshot_count": len(complete_snapshots),
    }
    return report, complete_snapshots


def _parse_worktrees(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for line in _run_git(
        root, "worktree", "list", "--porcelain"
    ).stdout.splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key in {"prunable", "detached", "locked"}:
            current[key] = True
    return records


def _is_ephemeral(path: Path) -> bool:
    return any(
        path == prefix or prefix in path.parents for prefix in EPHEMERAL_PREFIXES
    )


def _parse_sacct_rows(
    content: str,
    *,
    job_name_prefix: str,
    exclusive_job_id_upper_bound: int | None,
) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(content), delimiter="|")
    expected = {"JobIDRaw", "JobName", "State", "ExitCode", "NodeList", "WorkDir"}
    if not reader.fieldnames or expected - set(reader.fieldnames):
        raise RuntimeError("Slurm accounting output is missing required fields")
    rows: list[dict[str, str]] = []
    seen: set[int] = set()
    for raw in reader:
        job_id_text = str(raw.get("JobIDRaw", "")).strip()
        job_name = str(raw.get("JobName", "")).strip()
        if not job_id_text.isdigit() or not job_name.startswith(job_name_prefix):
            continue
        job_id = int(job_id_text)
        if (
            exclusive_job_id_upper_bound is not None
            and job_id >= exclusive_job_id_upper_bound
        ):
            continue
        if job_id in seen:
            raise RuntimeError(f"duplicate Slurm allocation record: {job_id}")
        seen.add(job_id)
        rows.append(
            {
                "job_id": job_id_text,
                "job_name": job_name,
                "state": str(raw.get("State", "")).strip(),
                "exit_code": str(raw.get("ExitCode", "")).strip(),
                "node_list": str(raw.get("NodeList", "")).strip(),
                "workdir": str(raw.get("WorkDir", "")).strip(),
            }
        )
    return sorted(rows, key=lambda row: int(row["job_id"]))


def _run_sacct(*args: str) -> str:
    completed = subprocess.run(
        ["sacct", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"sacct {' '.join(args)} failed: {detail}")
    if not completed.stdout.strip():
        raise RuntimeError("sacct returned no accounting records")
    return completed.stdout


def capture_scheduler_accounting(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    accounting_text: str | None = None,
    verification_text: str | None = None,
    write_output: bool = True,
) -> dict[str, object]:
    hostname = socket.gethostname().split(".", 1)[0]
    contract = load_contract(config_path, schema_path)
    scheduler = contract["freeze"]["authority_recovery"]["scheduler_reconciliation"]
    start_time = str(scheduler["accounting_start_time"])
    cutoff = int(scheduler["exclusive_job_id_upper_bound"])
    prefix = str(scheduler["job_name_prefix"])
    fields = "JobIDRaw,JobName%80,State,ExitCode,NodeList%40,WorkDir%240"
    accounting_command = (
        "sacct",
        "--starttime",
        start_time,
        "-X",
        "-P",
        f"--format={fields}",
    )
    if accounting_text is None:
        accounting_text = _run_sacct(*accounting_command[1:])
    jobs = _parse_sacct_rows(
        accounting_text,
        job_name_prefix=prefix,
        exclusive_job_id_upper_bound=cutoff,
    )
    verification_spec = scheduler["verification_job"]
    verification_job_id = int(verification_spec["job_id"])
    verification_command = (
        "sacct",
        "-j",
        str(verification_job_id),
        "--starttime",
        start_time,
        "-X",
        "-P",
        f"--format={fields}",
    )
    if verification_text is None:
        verification_text = _run_sacct(*verification_command[1:])
    verification_rows = _parse_sacct_rows(
        verification_text,
        job_name_prefix=prefix,
        exclusive_job_id_upper_bound=None,
    )
    verification_rows = [
        row for row in verification_rows if int(row["job_id"]) == verification_job_id
    ]
    verification_row = verification_rows[0] if len(verification_rows) == 1 else None
    verification_matches = bool(
        verification_row
        and verification_row["state"] == verification_spec["expected_state"]
        and verification_row["exit_code"] == verification_spec["expected_exit_code"]
        and verification_spec["expected_node"] in verification_row["node_list"]
    )
    passed = bool(
        len(jobs) == int(scheduler["expected_job_count"])
        and len({row["workdir"] for row in jobs})
        == int(scheduler["expected_workdir_count"])
        and verification_matches
    )
    snapshot: dict[str, object] = {
        "schema_version": "neuronsifter-round22a-scheduler-accounting-snapshot-v1",
        "round_id": ROUND22A_ID,
        "status": "passed" if passed else "blocked_scheduler_accounting_snapshot",
        "capture_complete": passed,
        "execution_node": hostname,
        "accounting_start_time": start_time,
        "exclusive_job_id_upper_bound": cutoff,
        "job_name_prefix": prefix,
        "accounting_command": list(accounting_command),
        "accounting_record_count": len(jobs),
        "accounting_records_sha256": canonical_sha256(jobs),
        "accounting_records": jobs,
        "workdir_count": len({row["workdir"] for row in jobs}),
        "verification_job": {
            "command": list(verification_command),
            "record": verification_row,
            "matches_frozen_expectation": verification_matches,
        },
        "filesystem_paths_inspected": 0,
        "artifact_hashes_read": 0,
        "manual_narrative_used_as_authority": False,
        "c2_b_status": C2B_BLOCKED,
    }
    snapshot["artifact_uid"] = canonical_sha256(snapshot)
    if write_output:
        output_value = str(scheduler["accounting_snapshot_output"])
        if output_value != OUTPUTS["scheduler_snapshot"]:
            raise ValueError("Round-22A scheduler snapshot output path drifted")
        _write_immutable_json(root / output_value, snapshot)
    return snapshot


def _path_is_under(path: Path, prefixes: Iterable[Path]) -> bool:
    return any(path == prefix or prefix in path.parents for prefix in prefixes)


def _exact_authority_path_checks(workdir: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for authority_uid, relative_path in REQUIRED_AUTHORITY_INPUTS.items():
        candidate = workdir / relative_path
        exists = candidate.is_file() and not candidate.is_symlink()
        try:
            byte_size = candidate.stat().st_size if exists else 0
            digest = _sha256_bytes(candidate.read_bytes()) if exists else None
            read_error = ""
        except OSError as exc:
            byte_size = 0
            digest = None
            read_error = f"{type(exc).__name__}:{exc}"
        checks.append(
            {
                "authority_uid": authority_uid,
                "relative_path": relative_path,
                "candidate_path": candidate.as_posix(),
                "exists": exists,
                "bytes": byte_size,
                "sha256": digest,
                "read_error": read_error,
            }
        )
    return checks


def build_scheduler_reconciliation(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    accounting_text: str | None = None,
    verification_text: str | None = None,
    write_output: bool = True,
    enforce_compute_node: bool = True,
) -> dict[str, object]:
    hostname = socket.gethostname().split(".", 1)[0]
    if enforce_compute_node and hostname not in ALLOWED_COMPUTE_NODES:
        raise RuntimeError(
            "Round-22A scheduler reconciliation must run on cpu03 or cpu04"
        )
    contract = load_contract(config_path, schema_path)
    recovery = contract["freeze"]["authority_recovery"]
    scheduler = recovery["scheduler_reconciliation"]
    start_time = str(scheduler["accounting_start_time"])
    cutoff = int(scheduler["exclusive_job_id_upper_bound"])
    prefix = str(scheduler["job_name_prefix"])
    snapshot_path = root / str(scheduler["accounting_snapshot_output"])
    if accounting_text is not None or verification_text is not None:
        if accounting_text is None or verification_text is None:
            raise ValueError("Both injected scheduler accounting inputs are required")
        snapshot = capture_scheduler_accounting(
            config_path,
            schema_path,
            root,
            accounting_text=accounting_text,
            verification_text=verification_text,
            write_output=False,
        )
        snapshot_binding = {
            "path": scheduler["accounting_snapshot_output"],
            "exists": False,
            "sha256": None,
            "artifact_uid": snapshot["artifact_uid"],
            "injected_for_test": True,
        }
    else:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, Mapping):
            raise RuntimeError("Scheduler accounting snapshot root is invalid")
        snapshot_without_uid = dict(snapshot)
        snapshot_uid = snapshot_without_uid.pop("artifact_uid", None)
        if (
            snapshot.get("schema_version")
            != "neuronsifter-round22a-scheduler-accounting-snapshot-v1"
            or snapshot_uid != canonical_sha256(snapshot_without_uid)
            or snapshot.get("capture_complete") is not True
            or snapshot.get("accounting_start_time") != start_time
            or snapshot.get("exclusive_job_id_upper_bound") != cutoff
            or snapshot.get("job_name_prefix") != prefix
        ):
            raise RuntimeError(
                "Scheduler accounting snapshot failed binding validation"
            )
        snapshot_binding = {
            "path": scheduler["accounting_snapshot_output"],
            "exists": True,
            "sha256": _sha256_bytes(snapshot_path.read_bytes()),
            "artifact_uid": snapshot_uid,
            "injected_for_test": False,
        }
    jobs_value = snapshot.get("accounting_records")
    jobs = (
        [dict(row) for row in jobs_value if isinstance(row, Mapping)]
        if isinstance(jobs_value, list)
        else []
    )
    verification_value = snapshot.get("verification_job")
    verification_row = (
        dict(verification_value.get("record"))
        if isinstance(verification_value, Mapping)
        and isinstance(verification_value.get("record"), Mapping)
        else None
    )
    verification_matches = bool(
        isinstance(verification_value, Mapping)
        and verification_value.get("matches_frozen_expectation") is True
    )
    accounting_command = tuple(
        str(value) for value in snapshot.get("accounting_command", [])
    )
    verification_command = (
        tuple(str(value) for value in verification_value.get("command", []))
        if isinstance(verification_value, Mapping)
        else ()
    )

    workdir_counts = Counter(row["workdir"] for row in jobs)
    worktrees = _parse_worktrees(root)
    registered_worktree_paths = {
        Path(str(row.get("path", ""))).resolve()
        for row in worktrees
        if not row.get("prunable")
        and str(row.get("path", "")).strip()
        and not _is_ephemeral(Path(str(row.get("path", ""))))
    }
    covered_paths = registered_worktree_paths | {root.resolve()}
    ephemeral_roots = tuple(
        Path(str(value)).resolve() for value in scheduler["ephemeral_roots"]
    )
    expected_extra = {
        Path(str(row["path"])).resolve(): int(row["expected_job_count"])
        for row in scheduler["expected_extra_workdirs"]
    }
    workdir_rows: list[dict[str, object]] = []
    observed_extra: set[Path] = set()
    unclassified_durable: list[str] = []
    additional_candidate_count = 0
    read_error_count = 0
    for workdir_text, job_count in sorted(workdir_counts.items()):
        workdir = Path(workdir_text).resolve() if workdir_text else Path("")
        ephemeral = bool(workdir_text) and _path_is_under(workdir, ephemeral_roots)
        covered = bool(workdir_text) and workdir in covered_paths
        expected_supplemental = bool(workdir_text) and workdir in expected_extra
        if expected_supplemental:
            observed_extra.add(workdir)
        checks = _exact_authority_path_checks(workdir) if expected_supplemental else []
        additional_candidate_count += sum(bool(row["exists"]) for row in checks)
        read_error_count += sum(bool(row["read_error"]) for row in checks)
        if workdir_text and not ephemeral and not covered and not expected_supplemental:
            unclassified_durable.append(workdir.as_posix())
        workdir_rows.append(
            {
                "path": workdir_text,
                "resolved_path": workdir.as_posix() if workdir_text else "",
                "job_count": job_count,
                "classification": (
                    "ephemeral_excluded"
                    if ephemeral
                    else "covered_by_wp1_registered_worktree"
                    if covered
                    else "scheduler_supplemental_exact_check"
                    if expected_supplemental
                    else "unclassified_durable_workdir"
                ),
                "exists": workdir.is_dir() if workdir_text and not ephemeral else None,
                "exact_authority_path_checks": checks,
            }
        )

    expected_extra_counts_match = all(
        workdir_counts.get(path.as_posix(), 0) == expected_count
        for path, expected_count in expected_extra.items()
    )
    counts_match = bool(
        len(jobs) == int(scheduler["expected_job_count"])
        and len(workdir_counts) == int(scheduler["expected_workdir_count"])
    )
    extra_roots_match = observed_extra == set(expected_extra)
    search_complete = bool(
        counts_match
        and expected_extra_counts_match
        and extra_roots_match
        and verification_matches
        and not unclassified_durable
        and read_error_count == 0
    )
    report: dict[str, object] = {
        "schema_version": "neuronsifter-round22a-scheduler-workdir-reconciliation-v1",
        "round_id": ROUND22A_ID,
        "status": "passed" if search_complete else "blocked_scheduler_reconciliation",
        "scheduler_reconciliation_complete": search_complete,
        "bounded": True,
        "unbounded_filesystem_scan": False,
        "accounting_start_time": start_time,
        "exclusive_job_id_upper_bound": cutoff,
        "job_name_prefix": prefix,
        "accounting_snapshot_artifact": snapshot_binding,
        "accounting_command": list(accounting_command),
        "accounting_record_count": len(jobs),
        "expected_accounting_record_count": int(scheduler["expected_job_count"]),
        "accounting_records_sha256": canonical_sha256(jobs),
        "accounting_records": jobs,
        "workdir_count": len(workdir_counts),
        "expected_workdir_count": int(scheduler["expected_workdir_count"]),
        "registered_worktree_paths_sha256": canonical_sha256(
            sorted(path.as_posix() for path in registered_worktree_paths)
        ),
        "workdirs": workdir_rows,
        "expected_extra_workdirs": [
            {"path": path.as_posix(), "expected_job_count": count}
            for path, count in sorted(
                expected_extra.items(), key=lambda item: item[0].as_posix()
            )
        ],
        "unclassified_durable_workdirs": sorted(unclassified_durable),
        "supplemental_authority_candidate_count": additional_candidate_count,
        "supplemental_read_error_count": read_error_count,
        "verification_job": {
            "command": list(verification_command),
            "record": verification_row,
            "matches_frozen_expectation": verification_matches,
        },
        "manual_narrative_used_as_authority": False,
        "historical_artifacts_rewritten": False,
        "execution_node": hostname,
        "c2_b_status": C2B_BLOCKED,
        "next_action": (
            "continue_tier_transition_with_scheduler_evidence_bound"
            if search_complete and additional_candidate_count == 0
            else "adjudicate_scheduler_workdir_authority_candidates"
            if additional_candidate_count
            else "repair_scheduler_workdir_reconciliation"
        ),
    }
    report["artifact_uid"] = canonical_sha256(report)
    if write_output:
        output_path = root / str(scheduler["output"])
        if str(scheduler["output"]) != OUTPUTS["scheduler"]:
            raise ValueError("Round-22A scheduler reconciliation output path drifted")
        _write_immutable_json(output_path, report)
    return report


def _search_worktrees(
    root: Path,
    root_spec: Mapping[str, Any],
    collector: CandidateCollector,
) -> dict[str, object]:
    records = _parse_worktrees(root)
    if len(records) > int(root_spec["max_entries"]):
        raise RuntimeError(
            f"registered worktree entry cap exceeded: {len(records)} > {root_spec['max_entries']}"
        )
    excluded = Counter()
    searched = 0
    for record in records:
        worktree = Path(str(record.get("path", "")))
        if record.get("prunable"):
            excluded["prunable"] += 1
            continue
        if _is_ephemeral(worktree):
            excluded["ephemeral_filesystem"] += 1
            continue
        if not worktree.is_dir():
            excluded["missing"] += 1
            continue
        searched += 1
        head = str(record.get("head", ""))
        for authority_uid, relative_path in REQUIRED_AUTHORITY_INPUTS.items():
            path = worktree / relative_path
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_bytes()
            schema_status, producer, _ = _validate_candidate(authority_uid, content)
            tracked_blob = _git_blob(worktree, head, relative_path) if head else None
            tracked_match = tracked_blob is not None and tracked_blob[1] == content
            collector.add(
                authority_uid=authority_uid,
                candidate_kind="authority_artifact",
                candidate_path=path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else path.as_posix(),
                source_class="historical_isolated_worktree",
                source_root_id=str(root_spec["root_id"]),
                content=content,
                git_commit=head,
                git_blob=tracked_blob[0] if tracked_match and tracked_blob else "",
                producer=producer,
                schema_status=schema_status,
                reference_status="pending_cross_artifact_validation",
                recovery_confidence="high"
                if tracked_match and schema_status == "valid"
                else "medium",
            )
    return {
        "root_id": root_spec["root_id"],
        "path": root_spec["path"],
        "resolver": root_spec["resolver"],
        "source_classes": root_spec["source_classes"],
        "status": "searched",
        "complete": True,
        "bounded": True,
        "exact_filename_only": True,
        "registered_worktree_count": len(records),
        "searched_worktree_count": searched,
        "excluded_worktree_counts": dict(sorted(excluded.items())),
    }


def _scan_exact_filenames(
    base: Path, names: set[str], max_depth: int, max_entries: int
) -> tuple[list[Path], int]:
    if not base.is_dir():
        return [], 0
    matches: list[Path] = []
    visited = 0
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        directory, depth = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                visited += 1
                if visited > max_entries:
                    raise RuntimeError(
                        f"registered root entry cap exceeded: {base} > {max_entries}"
                    )
                if entry.is_symlink():
                    continue
                if entry.is_file(follow_symlinks=False) and entry.name in names:
                    matches.append(Path(entry.path))
                elif entry.is_dir(follow_symlinks=False) and depth < max_depth:
                    stack.append((Path(entry.path), depth + 1))
    return sorted(matches), visited


def _search_filesystem_root(
    root: Path,
    root_spec: Mapping[str, Any],
    collector: CandidateCollector,
) -> dict[str, object]:
    base = root / str(root_spec["path"])
    max_depth = int(root_spec.get("max_depth", 0))
    max_entries = int(root_spec["max_entries"])
    matches, visited = _scan_exact_filenames(
        base,
        {Path(path).name for path in REQUIRED_AUTHORITY_INPUTS.values()},
        max_depth,
        max_entries,
    )
    by_name = {Path(path).name: uid for uid, path in REQUIRED_AUTHORITY_INPUTS.items()}
    for path in matches:
        authority_uid = by_name[path.name]
        content = path.read_bytes()
        schema_status, producer, _ = _validate_candidate(authority_uid, content)
        collector.add(
            authority_uid=authority_uid,
            candidate_kind="authority_artifact",
            candidate_path=path.relative_to(root).as_posix()
            if path.is_relative_to(root)
            else path.as_posix(),
            source_class=str(root_spec["source_classes"][0]),
            source_root_id=str(root_spec["root_id"]),
            content=content,
            producer=producer,
            schema_status=schema_status,
            reference_status="pending_cross_artifact_validation",
            recovery_confidence="medium" if schema_status == "valid" else "low",
        )
    return {
        "root_id": root_spec["root_id"],
        "path": root_spec["path"],
        "resolver": root_spec["resolver"],
        "source_classes": root_spec["source_classes"],
        "status": "searched" if base.is_dir() else "checked_registered_root_absent",
        "complete": True,
        "bounded": True,
        "exact_filename_only": True,
        "max_depth": max_depth,
        "max_entries": max_entries,
        "visited_entry_count": visited,
        "matched_file_count": len(matches),
    }


def _add_reconstruction_locators(root: Path, collector: CandidateCollector) -> int:
    count = 0
    for authority_uid, candidates in AUTHORITY_CANDIDATES.items():
        for relative_path in candidates:
            path = root / relative_path
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_bytes()
            collector.add(
                authority_uid=authority_uid,
                candidate_kind="immutable_component_locator",
                candidate_path=relative_path,
                source_class="registered_reconstruction_candidate",
                source_root_id="round22r_authority_candidates",
                content=content,
                schema_status="not_authority_schema_locator",
                reference_status="tier_b_component_completeness_not_yet_proven",
                recovery_confidence="locator_only",
            )
            count += 1
    return count


def _validate_root_contract(recovery: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    roots = recovery.get("historical_roots")
    if not isinstance(roots, list):
        raise ValueError("Round-22A historical root registry is missing")
    pairs = {
        str(row.get("root_id")): (str(row.get("path")), str(row.get("resolver")))
        for row in roots
        if isinstance(row, Mapping)
    }
    if pairs != EXPECTED_ROOTS or len(roots) != len(EXPECTED_ROOTS):
        raise ValueError(
            "Round-22A historical root registry drifted from the closed allowlist"
        )
    return roots


def _current_gate_invariants(root: Path) -> dict[str, object]:
    gate_path = root / "Output/audits/round22r_empirical_prerequisite_gate.json"
    if not gate_path.is_file():
        return {"verified": False, "reason": "round22r_gate_missing"}
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    authority = payload.get("round21_authority", {})
    components = payload.get("static_twin_freeze_components", {})
    tools = payload.get("tool_outcomes", {})
    verified = bool(
        authority.get("present")
        == authority.get("validated")
        == authority.get("hash_bound")
        == 0
        and components.get("present") == components.get("validated") == 0
        and tools.get("source_rows") == 384
        and tools.get("formal_eligible_offline")
        == tools.get("formal_eligible_real_round")
        == 0
        and payload.get("c2_b_status") == C2B_BLOCKED
        and payload.get("exact_clean_full_suite_authorized") is False
    )
    return {
        "verified": verified,
        "authority_present": authority.get("present"),
        "authority_validated": authority.get("validated"),
        "authority_hash_bound": authority.get("hash_bound"),
        "freeze_present": components.get("present"),
        "freeze_validated": components.get("validated"),
        "tool_source_rows": tools.get("source_rows"),
        "formal_offline": tools.get("formal_eligible_offline"),
        "formal_real_round": tools.get("formal_eligible_real_round"),
        "c2_b_status": payload.get("c2_b_status"),
        "exact_clean_full_suite_authorized": payload.get(
            "exact_clean_full_suite_authorized"
        ),
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(
                f"Immutable Round-22A artifact already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def build(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    write_outputs: bool = True,
    enforce_compute_node: bool = True,
) -> dict[str, object]:
    hostname = socket.gethostname().split(".", 1)[0]
    if enforce_compute_node and hostname not in ALLOWED_COMPUTE_NODES:
        raise RuntimeError("Round-22A recovery search must run on cpu03 or cpu04")
    contract = load_contract(config_path, schema_path)
    recovery = contract["freeze"]["authority_recovery"]
    roots = _validate_root_contract(recovery)
    collector = CandidateCollector()
    root_reports: list[dict[str, object]] = []
    complete_git_snapshots: list[str] = []
    for root_spec in roots:
        resolver = root_spec["resolver"]
        if resolver == "git_exact_path_history":
            report, snapshots = _search_git_history(root, root_spec, collector)
            root_reports.append(report)
            complete_git_snapshots.extend(snapshots)
        elif resolver == "git_registered_worktrees_exact_path":
            root_reports.append(_search_worktrees(root, root_spec, collector))
        elif resolver == "filesystem_exact_filename":
            root_reports.append(_search_filesystem_root(root, root_spec, collector))
        else:
            raise ValueError(f"Unsupported Round-22A root resolver: {resolver}")
    locator_count = _add_reconstruction_locators(root, collector)
    inventory_rows = collector.rows()
    inventory_path = root / OUTPUTS["inventory"]
    if write_outputs:
        _write_immutable_tsv(inventory_path, INVENTORY_FIELDS, inventory_rows)
    inventory_bytes = (
        inventory_path.read_bytes()
        if write_outputs
        else _tsv_bytes(INVENTORY_FIELDS, inventory_rows)
    )
    authority_counts = Counter(
        str(row["authority_uid"])
        for row in inventory_rows
        if row["candidate_kind"] == "authority_artifact"
    )
    all_roots_complete = all(row.get("complete") is True for row in root_reports)
    gate_invariants = _current_gate_invariants(root)
    search_report: dict[str, object] = {
        "schema_version": "neuronsifter-round22a-authority-recovery-search-v1",
        "round_id": ROUND22A_ID,
        "status": "search_complete"
        if all_roots_complete and gate_invariants.get("verified") is True
        else "blocked_search_or_invariant",
        "registered_search_roots_exhausted": all_roots_complete,
        "candidate_inventory_frozen": write_outputs,
        "candidate_inventory_path": OUTPUTS["inventory"],
        "candidate_inventory_sha256": _sha256_bytes(inventory_bytes),
        "candidate_inventory_row_count": len(inventory_rows),
        "authority_candidate_counts": {
            authority_uid: authority_counts.get(authority_uid, 0)
            for authority_uid in REQUIRED_AUTHORITY_INPUTS
        },
        "immutable_component_locator_count": locator_count,
        "complete_schema_valid_git_snapshot_count": len(set(complete_git_snapshots)),
        "complete_schema_valid_git_snapshots": sorted(set(complete_git_snapshots)),
        "tier_a_status": "complete_candidate_snapshot_found_requires_wp2_validation"
        if complete_git_snapshots
        else "no_complete_git_snapshot_found_wp2_still_required",
        "tier_b_status": "not_executed_wp1_component_inventory_only",
        "tier_c_status": "not_authorized_wp1",
        "tier_c_replay_started": False,
        "gpu_used": False,
        "unbounded_filesystem_scan": False,
        "exact_filename_only": recovery["exact_filename_only"],
        "manual_narrative_used_as_authority": False,
        "historical_manifests_rewritten": False,
        "canonical_authority_paths_written": False,
        "hash_read_count": collector.hash_reads,
        "execution_node": hostname,
        "roots": root_reports,
        "current_round22r_gate_invariants": gate_invariants,
        "c2_b_status": C2B_BLOCKED,
        "next_action": "validate_complete_tier_a_candidate_snapshot"
        if complete_git_snapshots
        else "evaluate_tier_a_candidates_then_tier_b_component_completeness",
    }
    search_report["artifact_uid"] = canonical_sha256(search_report)
    if write_outputs:
        _write_immutable_json(root / OUTPUTS["search"], search_report)
    return {"inventory": inventory_rows, "search": search_report}


def _tsv_bytes(
    fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--capture-scheduler-accounting", action="store_true")
    parser.add_argument("--scheduler-reconciliation", action="store_true")
    args = parser.parse_args()
    if args.capture_scheduler_accounting:
        result = capture_scheduler_accounting(
            args.config,
            args.schema,
            Path(args.root),
        )
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    elif args.scheduler_reconciliation:
        result = build_scheduler_reconciliation(
            args.config,
            args.schema,
            Path(args.root),
        )
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        result = build(args.config, args.schema, Path(args.root))
        print(json.dumps(result["search"], ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
