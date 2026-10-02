"""Validate the frozen Round-22A WP1 authority forensic inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    canonical_sha256,
    load_contract,
)
from Libs.scripts.builders.build_round22a_authority_recovery import (  # noqa: E402
    EXPECTED_ROOTS,
    INVENTORY_FIELDS,
    OUTPUTS,
    REQUIRED_AUTHORITY_INPUTS,
    ROUND22A_ID,
    _sha256_bytes,
    _validate_candidate,
    build,
    build_scheduler_reconciliation,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _tsv_bytes(rows: list[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(INVENTORY_FIELDS),
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in INVENTORY_FIELDS})
    return buffer.getvalue().encode("utf-8")


def _candidate_content(root: Path, row: Mapping[str, str]) -> bytes | None:
    candidate_path = row.get("candidate_path", "")
    if candidate_path.startswith("git:"):
        _, commit, relative_path = candidate_path.split(":", 2)
        return subprocess_bytes(root, commit, relative_path)
    path = Path(candidate_path)
    path = path if path.is_absolute() else root / path
    return path.read_bytes() if path.is_file() and not path.is_symlink() else None


def subprocess_bytes(root: Path, commit: str, relative_path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
        check=False,
        capture_output=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def _block(blockers: list[dict[str, str]], code: str, message: str) -> None:
    blockers.append({"code": code, "message": message})


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    recompute: bool = False,
    verify_candidate_bytes: bool = False,
    enforce_compute_node: bool = True,
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    contract = load_contract(config_path, schema_path)
    recovery = contract["freeze"]["authority_recovery"]
    roots = recovery.get("historical_roots", [])
    pairs = {
        str(row.get("root_id")): (str(row.get("path")), str(row.get("resolver")))
        for row in roots
        if isinstance(row, Mapping)
    }
    if pairs != EXPECTED_ROOTS or len(roots) != len(EXPECTED_ROOTS):
        _block(
            blockers,
            "registered_root_contract_drift",
            "The historical root allowlist is not the frozen seven-root registry.",
        )

    inventory_path = root / OUTPUTS["inventory"]
    search_path = root / OUTPUTS["search"]
    try:
        fields, rows = _read_tsv(inventory_path)
        search = _read_json(search_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, csv.Error) as exc:
        return {
            "schema_version": "neuronsifter-round22a-authority-recovery-validation-v1",
            "status": "failed",
            "passed": False,
            "blockers": [{"code": "artifact_read_failure", "message": str(exc)}],
        }
    if fields != list(INVENTORY_FIELDS):
        _block(
            blockers,
            "inventory_schema_drift",
            "Candidate inventory columns drifted from the frozen schema.",
        )
    if (
        search.get("schema_version")
        != "neuronsifter-round22a-authority-recovery-search-v1"
        or search.get("round_id") != ROUND22A_ID
    ):
        _block(
            blockers,
            "search_schema_or_round_drift",
            "Search artifact schema or round identity is invalid.",
        )
    artifact_uid = search.get("artifact_uid")
    without_uid = dict(search)
    without_uid.pop("artifact_uid", None)
    if artifact_uid != canonical_sha256(without_uid):
        _block(
            blockers,
            "search_artifact_uid_mismatch",
            "Search artifact UID does not bind its payload.",
        )
    inventory_bytes = inventory_path.read_bytes()
    if (
        search.get("candidate_inventory_sha256")
        != hashlib.sha256(inventory_bytes).hexdigest()
    ):
        _block(
            blockers,
            "inventory_hash_mismatch",
            "Search report does not bind the current inventory bytes.",
        )
    if search.get("candidate_inventory_row_count") != len(rows):
        _block(
            blockers,
            "inventory_count_mismatch",
            "Search report inventory count is stale.",
        )

    allowed_kinds = {"authority_artifact", "immutable_component_locator"}
    authority_counts = {uid: 0 for uid in REQUIRED_AUTHORITY_INPUTS}
    seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        authority_uid = row.get("authority_uid", "")
        kind = row.get("candidate_kind", "")
        digest = row.get("sha256", "")
        key = (authority_uid, kind, digest)
        if authority_uid not in REQUIRED_AUTHORITY_INPUTS or kind not in allowed_kinds:
            _block(
                blockers,
                "inventory_identity_invalid",
                f"Invalid candidate identity: {authority_uid}/{kind}.",
            )
            continue
        if key in seen_keys:
            _block(
                blockers,
                "inventory_content_not_deduplicated",
                f"Duplicate content row for {authority_uid}.",
            )
        seen_keys.add(key)
        try:
            paths = json.loads(row.get("all_candidate_paths_json", ""))
            source_classes = json.loads(row.get("all_source_classes_json", ""))
            commits = json.loads(row.get("all_git_commits_json", ""))
            occurrence_count = int(row.get("source_occurrence_count", ""))
            byte_size = int(row.get("byte_size", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            _block(
                blockers,
                "inventory_aggregate_invalid",
                f"Candidate aggregate fields are invalid for {authority_uid}.",
            )
            continue
        if (
            not isinstance(paths, list)
            or not isinstance(source_classes, list)
            or not isinstance(commits, list)
        ):
            _block(
                blockers,
                "inventory_aggregate_type_invalid",
                f"Candidate aggregate fields are not lists for {authority_uid}.",
            )
        if occurrence_count != len(paths) or row.get("candidate_path") not in paths:
            _block(
                blockers,
                "inventory_occurrence_mismatch",
                f"Candidate occurrence ledger is inconsistent for {authority_uid}.",
            )
        if (
            byte_size < 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _block(
                blockers,
                "candidate_binding_invalid",
                f"Candidate byte/hash binding is malformed for {authority_uid}.",
            )
        content = None
        if verify_candidate_bytes:
            for source_path in paths:
                source_row = dict(row) | {"candidate_path": str(source_path)}
                source_content = _candidate_content(root, source_row)
                if source_content is None:
                    _block(
                        blockers,
                        "candidate_unreadable",
                        f"Candidate source is unavailable for {authority_uid}: {source_path}.",
                    )
                    continue
                if (
                    len(source_content) != byte_size
                    or _sha256_bytes(source_content) != digest
                ):
                    _block(
                        blockers,
                        "candidate_byte_identity_mismatch",
                        f"Candidate bytes drifted for {authority_uid}: {source_path}.",
                    )
                if source_path == row.get("candidate_path"):
                    content = source_content
        if kind == "authority_artifact":
            authority_counts[authority_uid] += 1
            if content is not None:
                schema_status, producer, _ = _validate_candidate(authority_uid, content)
                if schema_status != row.get("schema_status") or producer != row.get(
                    "producer_if_known", ""
                ):
                    _block(
                        blockers,
                        "candidate_schema_readback_mismatch",
                        f"Candidate schema readback drifted for {authority_uid}.",
                    )
    if search.get("authority_candidate_counts") != authority_counts:
        _block(
            blockers,
            "authority_candidate_count_mismatch",
            "Search report authority counts do not match inventory rows.",
        )

    root_rows = search.get("roots")
    if not isinstance(root_rows, list):
        _block(
            blockers,
            "root_search_records_missing",
            "Search report has no root records.",
        )
    else:
        root_ids = [
            str(row.get("root_id", "")) for row in root_rows if isinstance(row, Mapping)
        ]
        if set(root_ids) != set(EXPECTED_ROOTS) or len(root_ids) != len(set(root_ids)):
            _block(
                blockers,
                "root_search_identity_mismatch",
                "Search report does not cover each registered root exactly once.",
            )
        if any(
            row.get("complete") is not True or row.get("bounded") is not True
            for row in root_rows
            if isinstance(row, Mapping)
        ):
            _block(
                blockers,
                "root_search_incomplete",
                "At least one registered root is not completely and boundedly searched.",
            )
    for key, expected in (
        ("registered_search_roots_exhausted", True),
        ("candidate_inventory_frozen", True),
        ("unbounded_filesystem_scan", False),
        ("exact_filename_only", True),
        ("manual_narrative_used_as_authority", False),
        ("historical_manifests_rewritten", False),
        ("canonical_authority_paths_written", False),
        ("tier_c_replay_started", False),
        ("gpu_used", False),
    ):
        if search.get(key) is not expected:
            _block(
                blockers,
                f"search_invariant_{key}",
                f"Search invariant {key}={expected} was violated.",
            )
    gate = search.get("current_round22r_gate_invariants")
    if not isinstance(gate, Mapping) or gate.get("verified") is not True:
        _block(
            blockers,
            "round22r_gate_invariant_unverified",
            "WP1 did not preserve the prior fail-closed Round-22R gate.",
        )
    elif gate.get("c2_b_status") != "blocked_insufficient_real_e2_e4":
        _block(blockers, "c2_b_status_drift", "C2-B was changed during WP1.")

    if recompute and not blockers:
        rebuilt = build(
            config_path,
            schema_path,
            root,
            write_outputs=False,
            enforce_compute_node=enforce_compute_node,
        )
        rebuilt_bytes = _tsv_bytes(rebuilt["inventory"])
        if rebuilt_bytes != inventory_bytes:
            _block(
                blockers,
                "inventory_recompute_mismatch",
                "Independent search recomputation changed the candidate inventory.",
            )
        for key in (
            "registered_search_roots_exhausted",
            "authority_candidate_counts",
            "immutable_component_locator_count",
            "complete_schema_valid_git_snapshot_count",
            "complete_schema_valid_git_snapshots",
            "hash_read_count",
            "current_round22r_gate_invariants",
        ):
            if rebuilt["search"].get(key) != search.get(key):
                _block(
                    blockers,
                    "search_recompute_mismatch",
                    f"Independent search recomputation changed {key}.",
                )
        rebuilt_roots = [
            {key: value for key, value in row.items() if key != "visited_entry_count"}
            for row in rebuilt["search"].get("roots", [])
        ]
        frozen_roots = [
            {key: value for key, value in row.items() if key != "visited_entry_count"}
            for row in search.get("roots", [])
        ]
        if rebuilt_roots != frozen_roots:
            _block(
                blockers,
                "search_recompute_mismatch",
                "Independent search recomputation changed root search results.",
            )
    return {
        "schema_version": "neuronsifter-round22a-authority-recovery-validation-v1",
        "status": "passed" if not blockers else "failed",
        "passed": not blockers,
        "inventory_row_count": len(rows),
        "authority_candidate_counts": authority_counts,
        "registered_search_root_count": len(EXPECTED_ROOTS),
        "recomputed": recompute,
        "candidate_bytes_verified": verify_candidate_bytes,
        "blockers": blockers,
    }


def validate_scheduler_reconciliation(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    recompute: bool = False,
    enforce_compute_node: bool = True,
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    contract = load_contract(config_path, schema_path)
    scheduler = contract["freeze"]["authority_recovery"]["scheduler_reconciliation"]
    output_path = root / str(scheduler["output"])
    try:
        report = _read_json(output_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": "neuronsifter-round22a-scheduler-reconciliation-validation-v1",
            "status": "failed",
            "passed": False,
            "blockers": [
                {"code": "scheduler_artifact_read_failure", "message": str(exc)}
            ],
        }
    if (
        report.get("schema_version")
        != "neuronsifter-round22a-scheduler-workdir-reconciliation-v1"
        or report.get("round_id") != ROUND22A_ID
    ):
        _block(
            blockers,
            "scheduler_schema_or_round_drift",
            "Scheduler reconciliation schema or round identity is invalid.",
        )
    without_uid = dict(report)
    artifact_uid = without_uid.pop("artifact_uid", None)
    if artifact_uid != canonical_sha256(without_uid):
        _block(
            blockers,
            "scheduler_artifact_uid_mismatch",
            "Scheduler reconciliation artifact UID does not bind its payload.",
        )
    snapshot_binding = report.get("accounting_snapshot_artifact")
    snapshot_path = root / str(scheduler["accounting_snapshot_output"])
    snapshot: Mapping[str, object] | None = None
    if not isinstance(snapshot_binding, Mapping):
        _block(
            blockers,
            "scheduler_snapshot_binding_missing",
            "Scheduler reconciliation does not bind its accounting snapshot.",
        )
    else:
        try:
            loaded_snapshot = _read_json(snapshot_path)
            snapshot = loaded_snapshot
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            snapshot = None
        observed_snapshot_sha = (
            hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            if snapshot_path.is_file()
            else None
        )
        if (
            snapshot_binding.get("path") != scheduler["accounting_snapshot_output"]
            or snapshot_binding.get("exists") is not True
            or snapshot_binding.get("injected_for_test") is not False
            or snapshot_binding.get("sha256") != observed_snapshot_sha
            or snapshot is None
        ):
            _block(
                blockers,
                "scheduler_snapshot_binding_invalid",
                "Scheduler accounting snapshot binding is missing or stale.",
            )
        elif (
            snapshot.get("schema_version")
            != "neuronsifter-round22a-scheduler-accounting-snapshot-v1"
            or snapshot.get("capture_complete") is not True
            or snapshot.get("artifact_uid") != snapshot_binding.get("artifact_uid")
        ):
            _block(
                blockers,
                "scheduler_snapshot_state_invalid",
                "Scheduler accounting snapshot state is invalid.",
            )
        else:
            snapshot_without_uid = dict(snapshot)
            snapshot_uid = snapshot_without_uid.pop("artifact_uid", None)
            if snapshot_uid != canonical_sha256(snapshot_without_uid):
                _block(
                    blockers,
                    "scheduler_snapshot_artifact_uid_mismatch",
                    "Scheduler accounting snapshot UID does not bind its payload.",
                )
    records = report.get("accounting_records")
    if not isinstance(records, list):
        _block(
            blockers,
            "scheduler_accounting_records_missing",
            "Scheduler reconciliation has no accounting records.",
        )
        records = []
    if report.get("accounting_records_sha256") != canonical_sha256(records):
        _block(
            blockers,
            "scheduler_accounting_hash_mismatch",
            "Scheduler accounting record hash is stale.",
        )
    if isinstance(snapshot, Mapping) and (
        snapshot.get("accounting_records_sha256")
        != report.get("accounting_records_sha256")
        or snapshot.get("accounting_records") != records
        or snapshot.get("verification_job") != report.get("verification_job")
    ):
        _block(
            blockers,
            "scheduler_snapshot_reconciliation_mismatch",
            "Scheduler reconciliation drifted from its captured accounting snapshot.",
        )
    job_ids: list[int] = []
    workdir_counts: Counter[str] = Counter()
    for row in records:
        if not isinstance(row, Mapping):
            _block(
                blockers,
                "scheduler_accounting_row_invalid",
                "Scheduler accounting row is not an object.",
            )
            continue
        try:
            job_id = int(str(row.get("job_id", "")))
        except ValueError:
            _block(
                blockers,
                "scheduler_job_id_invalid",
                "Scheduler accounting row has an invalid job ID.",
            )
            continue
        job_ids.append(job_id)
        if job_id >= int(scheduler["exclusive_job_id_upper_bound"]):
            _block(
                blockers,
                "scheduler_cutoff_violation",
                f"Job {job_id} is outside the frozen accounting cutoff.",
            )
        if not str(row.get("job_name", "")).startswith(
            str(scheduler["job_name_prefix"])
        ):
            _block(
                blockers,
                "scheduler_job_name_scope_violation",
                f"Job {job_id} is outside the frozen project prefix.",
            )
        workdir_counts[str(row.get("workdir", ""))] += 1
    if len(job_ids) != len(set(job_ids)):
        _block(
            blockers,
            "scheduler_job_identity_duplicate",
            "Scheduler reconciliation contains duplicate allocation IDs.",
        )
    if (
        len(records) != int(scheduler["expected_job_count"])
        or report.get("accounting_record_count") != len(records)
        or report.get("expected_accounting_record_count")
        != int(scheduler["expected_job_count"])
    ):
        _block(
            blockers,
            "scheduler_job_count_mismatch",
            "Scheduler accounting count does not match the frozen contract.",
        )
    if (
        len(workdir_counts) != int(scheduler["expected_workdir_count"])
        or report.get("workdir_count") != len(workdir_counts)
        or report.get("expected_workdir_count")
        != int(scheduler["expected_workdir_count"])
    ):
        _block(
            blockers,
            "scheduler_workdir_count_mismatch",
            "Scheduler WorkDir count does not match the frozen contract.",
        )
    workdirs = report.get("workdirs")
    if not isinstance(workdirs, list):
        _block(
            blockers,
            "scheduler_workdir_records_missing",
            "Scheduler WorkDir records are missing.",
        )
        workdirs = []
    expected_extra = {
        str(Path(str(row["path"])).resolve()): int(row["expected_job_count"])
        for row in scheduler["expected_extra_workdirs"]
    }
    observed_extra: set[str] = set()
    observed_candidate_count = 0
    observed_read_errors = 0
    for row in workdirs:
        if not isinstance(row, Mapping):
            _block(
                blockers,
                "scheduler_workdir_row_invalid",
                "Scheduler WorkDir row is not an object.",
            )
            continue
        path = str(row.get("resolved_path", ""))
        original_path = str(row.get("path", ""))
        if int(row.get("job_count", -1)) != workdir_counts.get(original_path, 0):
            _block(
                blockers,
                "scheduler_workdir_job_count_mismatch",
                f"Scheduler WorkDir count drifted for {original_path}.",
            )
        if row.get("classification") != "scheduler_supplemental_exact_check":
            continue
        observed_extra.add(path)
        checks = row.get("exact_authority_path_checks")
        if not isinstance(checks, list) or len(checks) != len(
            REQUIRED_AUTHORITY_INPUTS
        ):
            _block(
                blockers,
                "scheduler_exact_path_check_incomplete",
                f"Supplemental root {path} was not checked for all authorities.",
            )
            continue
        checked_ids = {
            str(check.get("authority_uid", ""))
            for check in checks
            if isinstance(check, Mapping)
        }
        if checked_ids != set(REQUIRED_AUTHORITY_INPUTS):
            _block(
                blockers,
                "scheduler_exact_path_identity_mismatch",
                f"Supplemental root {path} has an incomplete authority map.",
            )
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            if str(check.get("read_error", "")):
                observed_read_errors += 1
            if check.get("exists") is not True:
                continue
            observed_candidate_count += 1
            candidate = Path(str(check.get("candidate_path", "")))
            try:
                content = (
                    candidate.read_bytes()
                    if candidate.is_file() and not candidate.is_symlink()
                    else None
                )
            except OSError:
                content = None
            if (
                content is None
                or len(content) != int(check.get("bytes", -1))
                or _sha256_bytes(content) != check.get("sha256")
            ):
                _block(
                    blockers,
                    "scheduler_candidate_byte_identity_mismatch",
                    f"Supplemental candidate drifted: {candidate}.",
                )
    if observed_extra != set(expected_extra):
        _block(
            blockers,
            "scheduler_extra_workdir_identity_mismatch",
            "Scheduler supplemental WorkDirs drifted from the frozen contract.",
        )
    for path, expected_count in expected_extra.items():
        original_counts = [
            int(row.get("job_count", -1))
            for row in workdirs
            if isinstance(row, Mapping) and row.get("resolved_path") == path
        ]
        if original_counts != [expected_count]:
            _block(
                blockers,
                "scheduler_extra_workdir_count_mismatch",
                f"Scheduler supplemental WorkDir count drifted for {path}.",
            )
    if report.get("supplemental_authority_candidate_count") != observed_candidate_count:
        _block(
            blockers,
            "scheduler_candidate_count_mismatch",
            "Supplemental authority candidate count is stale.",
        )
    if report.get("supplemental_read_error_count") != observed_read_errors:
        _block(
            blockers,
            "scheduler_read_error_count_mismatch",
            "Supplemental read-error count is stale.",
        )
    verification = report.get("verification_job")
    if (
        not isinstance(verification, Mapping)
        or verification.get("matches_frozen_expectation") is not True
    ):
        _block(
            blockers,
            "scheduler_verification_job_invalid",
            "Scheduler verification job does not match its frozen expectation.",
        )
    for key, expected in (
        ("scheduler_reconciliation_complete", True),
        ("bounded", True),
        ("unbounded_filesystem_scan", False),
        ("manual_narrative_used_as_authority", False),
        ("historical_artifacts_rewritten", False),
    ):
        if report.get(key) is not expected:
            _block(
                blockers,
                f"scheduler_invariant_{key}",
                f"Scheduler reconciliation invariant {key}={expected} was violated.",
            )
    if report.get("c2_b_status") != "blocked_insufficient_real_e2_e4":
        _block(blockers, "c2_b_status_drift", "C2-B changed during reconciliation.")
    if recompute and not blockers:
        rebuilt = build_scheduler_reconciliation(
            config_path,
            schema_path,
            root,
            write_output=False,
            enforce_compute_node=enforce_compute_node,
        )
        for key in (
            "scheduler_reconciliation_complete",
            "accounting_record_count",
            "accounting_records_sha256",
            "workdir_count",
            "expected_extra_workdirs",
            "unclassified_durable_workdirs",
            "supplemental_authority_candidate_count",
            "supplemental_read_error_count",
            "verification_job",
        ):
            if rebuilt.get(key) != report.get(key):
                _block(
                    blockers,
                    "scheduler_recompute_mismatch",
                    f"Scheduler reconciliation recomputation changed {key}.",
                )
    return {
        "schema_version": "neuronsifter-round22a-scheduler-reconciliation-validation-v1",
        "status": "passed" if not blockers else "failed",
        "passed": not blockers,
        "accounting_record_count": len(records),
        "workdir_count": len(workdir_counts),
        "supplemental_authority_candidate_count": observed_candidate_count,
        "recomputed": recompute,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--verify-candidate-bytes", action="store_true")
    parser.add_argument("--scheduler-reconciliation", action="store_true")
    parser.add_argument("--recompute-scheduler", action="store_true")
    args = parser.parse_args()
    if args.scheduler_reconciliation:
        result = validate_scheduler_reconciliation(
            args.config,
            args.schema,
            Path(args.root),
            recompute=args.recompute_scheduler,
        )
    else:
        result = validate(
            args.config,
            args.schema,
            Path(args.root),
            recompute=args.recompute,
            verify_candidate_bytes=args.verify_candidate_bytes,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
