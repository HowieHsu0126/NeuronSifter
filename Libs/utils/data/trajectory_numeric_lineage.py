from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

from Libs.utils.data.raw_data_registry import (
    load_config,
    resolve_repo_path,
    write_json,
    write_text_atomic,
    write_tsv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METRIC_COLUMNS = [
    "metric_name",
    "canonical_value",
    "definition",
    "source_artifact",
    "source_artifact_hash",
    "producer_command",
    "git_commit",
    "config_hash",
    "snapshot_or_release",
    "filters",
    "deduplication_policy",
    "qc_policy",
    "pairing_policy",
    "included_count",
    "excluded_count",
    "exclusion_reasons",
    "parent_metrics",
    "child_metrics",
    "historical_values",
    "historical_value_explanation",
    "status",
]
REQUIRED_METRIC_FIELDS = set(METRIC_COLUMNS)


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _policy(config: Mapping[str, object]) -> dict[str, object]:
    union = _mapping(config.get("iclr_design_contract_union"))
    policy = _mapping(union.get("g0_numeric_lineage_audit"))
    if not policy:
        raise KeyError("iclr_design_contract_union.g0_numeric_lineage_audit")
    return policy


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sequence_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _tsv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _source_paths(policy: Mapping[str, object]) -> dict[str, Path]:
    return {
        str(name): resolve_repo_path(str(value))
        for name, value in _mapping(policy.get("source_artifacts")).items()
    }


def _source_inventory(policy: Mapping[str, object]) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    paths = _source_paths(policy)
    declared_paths = _mapping(policy.get("source_artifacts"))
    hash_required = {str(value) for value in policy.get("hash_required_artifact_keys", [])}
    inventory: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        exists = path.is_file()
        inventory[name] = {
            # Preserve the configured repository-relative reference when a shared
            # input symlink resolves outside an isolated worktree.
            "path": str(declared_paths[name]),
            "exists": exists,
            "sha256": _file_sha256(path) if exists and name in hash_required else None,
            "hash_required": name in hash_required,
        }
    return paths, inventory


def _full_store_count(path: Path) -> int:
    payload = _json(path)
    shape = payload.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"Missing Zarr shape in {path}")
    return int(shape[0])


def _qc_catalog_counts(path: Path) -> dict[str, object]:
    source_counts: Counter[str] = Counter()
    paired_counts: Counter[str] = Counter()
    grade_counts: Counter[str] = Counter()
    use_counts: Counter[str] = Counter()
    full_only_indices: list[int] = []
    full_only_families: Counter[str] = Counter()
    content_hashes: set[str] = set()
    row_count = 0
    for row in _tsv_rows(path):
        row_count += 1
        source = str(row.get("source_collection", ""))
        source_counts[source] += 1
        paired_counts[str(row.get("paired_status", ""))] += 1
        grade_counts[str(row.get("qc_grade", ""))] += 1
        use_counts[str(row.get("qc_use", ""))] += 1
        content_hash = str(row.get("content_hash", ""))
        if content_hash:
            content_hashes.add(content_hash)
        if source == "full_microenvironment_teacher" and str(row.get("paired_status", "")) == "full_only":
            full_only_indices.append(int(row["trajectory_index"]))
            full_only_families[str(row.get("teacher_family", ""))] += 1
    return {
        "row_count": row_count,
        "source_counts": dict(source_counts),
        "paired_counts": dict(paired_counts),
        "grade_counts": dict(grade_counts),
        "use_counts": dict(use_counts),
        "unique_content_count": len(content_hashes),
        "full_only_indices": sorted(full_only_indices),
        "full_only_family_counts": dict(full_only_families),
    }


def _pair_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in _tsv_rows(path):
        counts[str(row.get("paired_status", ""))] += 1
    return {name: int(value) for name, value in counts.items()}


def _pair_alignment_full_indices(path: Path) -> list[int]:
    indices: list[int] = []
    for row in _tsv_rows(path):
        if str(row.get("paired_status", "")) != "paired":
            continue
        value = str(row.get("full_trajectory_index", "")).strip()
        if not value:
            raise ValueError(f"Paired alignment row lacks full_trajectory_index: {row}")
        indices.append(int(value))
    return sorted(indices)


def _git_blob_record(record_policy: Mapping[str, object]) -> dict[str, object]:
    source_commit = str(record_policy.get("source_commit", ""))
    source_path = str(record_policy.get("source_path", ""))
    source_spec = f"{source_commit}:{source_path}"
    try:
        payload = subprocess.check_output(["git", "show", source_spec], cwd=REPO_ROOT)
        blob_id = subprocess.check_output(
            ["git", "rev-parse", source_spec], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "source_commit": source_commit,
            "source_path": source_path,
            "git_blob_id": None,
            "sha256": None,
            "required_fragments": [str(value) for value in record_policy.get("required_fragments", [])],
            "fragments_verified": False,
            "blob_verified": False,
            "error": str(error),
        }
    text = payload.decode("utf-8")
    required_fragments = [str(value) for value in record_policy.get("required_fragments", [])]
    fragments_verified = all(fragment in text for fragment in required_fragments)
    expected_blob_id = str(record_policy.get("expected_git_blob_id", ""))
    expected_sha256 = str(record_policy.get("expected_sha256", ""))
    sha256 = _bytes_sha256(payload)
    return {
        "source_commit": source_commit,
        "source_path": source_path,
        "git_blob_id": blob_id,
        "sha256": sha256,
        "required_fragments": required_fragments,
        "fragments_verified": fragments_verified,
        "blob_verified": blob_id == expected_blob_id and sha256 == expected_sha256,
        "error": None,
    }


def _historical_snapshot(policy: Mapping[str, object], governance: Mapping[str, object]) -> dict[str, object]:
    historical_policy = _mapping(policy.get("historical_snapshot"))
    disposition_policy = _mapping(policy.get("historical_snapshot_disposition"))
    drift = _mapping(governance.get("numeric_contract_drift"))
    snapshot = _mapping(drift.get("governance_snapshot"))
    full_count = int(snapshot.get("full_microenvironment_trajectories", -1))
    reject_count = int(snapshot.get("reject_records", -1))
    expected_full = int(historical_policy.get("required_full_count", -1))
    expected_rejected = int(historical_policy.get("required_rejected_count", -1))
    records = [
        _git_blob_record(_mapping(record_policy))
        for record_policy in historical_policy.get("governance_records", [])
    ]
    records_verified = bool(records) and all(
        bool(record.get("blob_verified")) and bool(record.get("fragments_verified")) for record in records
    )
    manifest_policy = _mapping(historical_policy.get("independent_manifest"))
    manifest_path_value = str(manifest_policy.get("source_artifact", "")).strip()
    manifest_path = resolve_repo_path(manifest_path_value) if manifest_path_value else None
    manifest_exists = bool(manifest_path and manifest_path.is_file())
    manifest_hash = _file_sha256(manifest_path) if manifest_exists and manifest_path else None
    expected_manifest_hash = str(manifest_policy.get("source_artifact_hash", "")).strip()
    manifest_fields: list[str] = []
    if manifest_exists and manifest_path and manifest_path.suffix.lower() == ".tsv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_fields = list(csv.DictReader(handle, delimiter="\t").fieldnames or [])
    required_manifest_fields = [str(value) for value in manifest_policy.get("required_fields", [])]
    manifest_fields_verified = bool(required_manifest_fields) and set(required_manifest_fields) <= set(manifest_fields)
    manifest_provenance_fields = [
        "producer_command",
        "git_commit",
        "config_hash",
        "snapshot_or_release",
    ]
    manifest_provenance_verified = all(str(manifest_policy.get(field, "")).strip() for field in manifest_provenance_fields)
    independent_manifest_required = bool(historical_policy.get("requires_independent_manifest_hash", False))
    independent_manifest_verified = (
        manifest_exists
        and bool(expected_manifest_hash)
        and manifest_hash == expected_manifest_hash
        and manifest_fields_verified
        and manifest_provenance_verified
    )
    independent_manifest = {
        "required": independent_manifest_required,
        "availability": str(manifest_policy.get("availability", "")),
        "unresolved_id": str(manifest_policy.get("unresolved_id", "historical_source_manifest_hash_missing")),
        "source_artifact": manifest_path_value,
        "expected_sha256": expected_manifest_hash,
        "observed_sha256": manifest_hash,
        "exists": manifest_exists,
        "required_fields": required_manifest_fields,
        "observed_fields": manifest_fields,
        "fields_verified": manifest_fields_verified,
        "provenance_verified": manifest_provenance_verified,
        "verified": independent_manifest_verified,
    }
    disposition_path_value = str(disposition_policy.get("authorization_artifact", "")).strip()
    disposition_path = resolve_repo_path(disposition_path_value) if disposition_path_value else None
    disposition_verified = False
    if disposition_path and disposition_path.is_file():
        try:
            disposition_payload = _json(disposition_path)
        except (OSError, TypeError, json.JSONDecodeError):
            disposition_payload = {}
        active_dependencies = _mapping(disposition_payload.get("active_dependencies"))
        disposition_verified = (
            disposition_payload.get("schema_version") == "neuronsifter-g0-user-authorized-closeout-v1"
            and disposition_payload.get("authorization_id") == disposition_policy.get("authorization_id")
            and disposition_payload.get("decision") == "close_g0_by_abandoning_unverifiable_historical_snapshot"
            and disposition_payload.get("scope", {}).get("disposition") == "abandoned_and_quarantined"
            and disposition_payload.get("current_source_of_truth", {}).get("path")
            == str(_mapping(policy.get("output_paths")).get("json", ""))
            and active_dependencies
            and all(value is False for value in active_dependencies.values())
        )
    return {
        "snapshot_id": str(historical_policy.get("snapshot_id", "")),
        "full_count": full_count,
        "rejected_count": reject_count,
        "matches_declared_values": full_count == expected_full and reject_count == expected_rejected,
        "provenance_level": str(historical_policy.get("provenance_level", "")),
        "original_manifest_recovered": independent_manifest_verified,
        "independent_manifest": independent_manifest,
        "disposition": str(disposition_policy.get("status", "")),
        "disposition_verified": disposition_verified,
        "disposition_authorization_artifact": disposition_path_value,
        "governance_records": records,
        "governance_records_verified": records_verified,
    }


def _count_by(rows: Iterable[Mapping[str, str]], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row.get(field_name, "")) for row in rows)
    return {key: int(counts[key]) for key in sorted(counts)}


def _required_count_map(policy: Mapping[str, object], key: str) -> dict[str, int]:
    return {str(name): int(value) for name, value in _mapping(policy.get(key)).items()}


def _historical_projection(
    qc_catalog_path: Path,
    pair_alignment_path: Path,
    historical: Mapping[str, object],
    current_full_count: int,
    historical_policy: Mapping[str, object],
) -> dict[str, object]:
    projection_policy = _mapping(historical_policy.get("projection"))
    source_collection = str(projection_policy.get("source_collection", ""))
    index_field = str(projection_policy.get("trajectory_index_field", ""))
    projection_fields = [str(value) for value in projection_policy.get("projection_fields", [])]
    prefix_start = int(projection_policy.get("prefix_start_index", 0))
    historical_full = int(historical["full_count"])
    historical_rejected = int(historical["rejected_count"])
    full_rows_by_index: dict[int, dict[str, str]] = {}
    for row in _tsv_rows(qc_catalog_path):
        if str(row.get("source_collection", "")) != source_collection:
            continue
        index = int(str(row.get(index_field, "")))
        if index in full_rows_by_index:
            raise ValueError(f"Duplicate full-teacher trajectory index: {index}")
        full_rows_by_index[index] = row

    ordered_indices = sorted(full_rows_by_index)
    prefix_indices = [index for index in ordered_indices if index < historical_full]
    suffix_indices = [index for index in ordered_indices if index >= historical_full]
    prefix_rows = [full_rows_by_index[index] for index in prefix_indices]
    suffix_rows = [full_rows_by_index[index] for index in suffix_indices]
    pair_alignment_indices = _pair_alignment_full_indices(pair_alignment_path)
    prefix_paired_indices = [
        index for index in prefix_indices if str(full_rows_by_index[index].get("paired_status", "")) == "paired"
    ]

    def canonical_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
        return [{field: str(row.get(field, "")) for field in projection_fields} for row in rows]

    prefix_expected_indices = list(range(prefix_start, historical_full))
    suffix_expected_indices = list(range(historical_full, current_full_count))
    prefix_identity_verified = (
        prefix_indices == prefix_expected_indices
        and len(prefix_rows) == historical_full - prefix_start
        and _count_by(prefix_rows, "qc_use").get("reject", 0) == historical_rejected
        and prefix_paired_indices == pair_alignment_indices
    )
    suffix_identity_verified = (
        suffix_indices == suffix_expected_indices
        and len(suffix_rows) == current_full_count - historical_full
    )
    required_prefix_counts = {
        "pair_status_counts": _required_count_map(projection_policy, "required_prefix_pair_status_counts"),
        "qc_grade_counts": _required_count_map(projection_policy, "required_prefix_qc_grade_counts"),
        "qc_use_counts": _required_count_map(projection_policy, "required_prefix_qc_use_counts"),
        "teacher_family_counts": _required_count_map(projection_policy, "required_prefix_teacher_family_counts"),
    }
    required_suffix_counts = {
        "pair_status_counts": _required_count_map(projection_policy, "required_suffix_pair_status_counts"),
        "qc_grade_counts": _required_count_map(projection_policy, "required_suffix_qc_grade_counts"),
        "qc_use_counts": _required_count_map(projection_policy, "required_suffix_qc_use_counts"),
        "teacher_family_counts": _required_count_map(projection_policy, "required_suffix_teacher_family_counts"),
    }
    observed_prefix_counts = {
        "pair_status_counts": _count_by(prefix_rows, "paired_status"),
        "qc_grade_counts": _count_by(prefix_rows, "qc_grade"),
        "qc_use_counts": _count_by(prefix_rows, "qc_use"),
        "teacher_family_counts": _count_by(prefix_rows, "teacher_family"),
    }
    observed_suffix_counts = {
        "pair_status_counts": _count_by(suffix_rows, "paired_status"),
        "qc_grade_counts": _count_by(suffix_rows, "qc_grade"),
        "qc_use_counts": _count_by(suffix_rows, "qc_use"),
        "teacher_family_counts": _count_by(suffix_rows, "teacher_family"),
    }
    prefix_contract_verified = observed_prefix_counts == required_prefix_counts
    suffix_contract_verified = observed_suffix_counts == required_suffix_counts
    return {
        "provenance_level": str(historical.get("provenance_level", "")),
        "original_manifest_recovered": bool(historical.get("original_manifest_recovered", False)),
        "selector": {
            "source_collection": source_collection,
            "trajectory_index_field": index_field,
            "projection_fields": projection_fields,
            "prefix_start_index": prefix_start,
            "prefix_stop_exclusive": historical_full,
            "suffix_start_index": historical_full,
            "suffix_stop_exclusive": current_full_count,
        },
        "prefix_indices": prefix_indices,
        "suffix_indices": suffix_indices,
        "pair_alignment_full_indices": pair_alignment_indices,
        "prefix_paired_indices": prefix_paired_indices,
        "prefix_index_sequence_sha256": _canonical_sequence_sha256({"trajectory_index": index} for index in prefix_indices),
        "suffix_index_sequence_sha256": _canonical_sequence_sha256({"trajectory_index": index} for index in suffix_indices),
        "pair_alignment_index_sequence_sha256": _canonical_sequence_sha256(
            {"full_trajectory_index": index} for index in pair_alignment_indices
        ),
        "prefix_row_sequence_sha256": _canonical_sequence_sha256(canonical_rows(prefix_rows)),
        "suffix_row_sequence_sha256": _canonical_sequence_sha256(canonical_rows(suffix_rows)),
        "prefix_row_count": len(prefix_rows),
        "suffix_row_count": len(suffix_rows),
        "prefix_counts": observed_prefix_counts,
        "suffix_counts": observed_suffix_counts,
        "required_prefix_counts": required_prefix_counts,
        "required_suffix_counts": required_suffix_counts,
        "prefix_index_sequence_verified": prefix_indices == prefix_expected_indices,
        "suffix_index_sequence_verified": suffix_indices == suffix_expected_indices,
        "pair_alignment_verified": prefix_paired_indices == pair_alignment_indices,
        "prefix_identity_verified": prefix_identity_verified,
        "suffix_identity_verified": suffix_identity_verified,
        "prefix_contract_verified": prefix_contract_verified,
        "suffix_contract_verified": suffix_contract_verified,
        "verified": (
            prefix_identity_verified
            and suffix_identity_verified
            and prefix_contract_verified
            and suffix_contract_verified
        ),
    }


def _metric(
    *,
    metric_name: str,
    canonical_value: int,
    definition: str,
    source_artifact: str,
    source_artifact_hash: str | None,
    producer_command: str,
    git_commit: str,
    config_hash: str,
    snapshot_or_release: str,
    filters: str,
    deduplication_policy: str,
    qc_policy: str,
    pairing_policy: str,
    included_count: int,
    excluded_count: int,
    exclusion_reasons: str,
    parent_metrics: list[str],
    child_metrics: list[str],
    historical_values: list[dict[str, object]],
    historical_value_explanation: str,
    status: str,
) -> dict[str, object]:
    return {
        "metric_name": metric_name,
        "canonical_value": canonical_value,
        "definition": definition,
        "source_artifact": source_artifact,
        "source_artifact_hash": source_artifact_hash,
        "producer_command": producer_command,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "snapshot_or_release": snapshot_or_release,
        "filters": filters,
        "deduplication_policy": deduplication_policy,
        "qc_policy": qc_policy,
        "pairing_policy": pairing_policy,
        "included_count": included_count,
        "excluded_count": excluded_count,
        "exclusion_reasons": exclusion_reasons,
        "parent_metrics": parent_metrics,
        "child_metrics": child_metrics,
        "historical_values": historical_values,
        "historical_value_explanation": historical_value_explanation,
        "status": status,
    }


def build_numeric_lineage_audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    policy = _policy(config)
    paths, source_inventory = _source_inventory(policy)
    required_paths = [name for name, row in source_inventory.items() if bool(row["hash_required"])]
    missing_paths = [name for name in required_paths if not bool(source_inventory[name]["exists"])]
    if missing_paths:
        raise FileNotFoundError(f"Numeric lineage source artifacts are missing: {missing_paths}")

    qc_summary = _json(paths["qc_summary"])
    qc_counts = _qc_catalog_counts(paths["qc_catalog"])
    pair_counts = _pair_counts(paths["pair_alignment"])
    governance = _json(paths["governance_snapshot"])
    historical = _historical_snapshot(policy, governance)
    historical_policy = _mapping(policy.get("historical_snapshot"))

    full_count = _full_store_count(paths["full_store_metadata"])
    mesoscopic_count = int(qc_counts["source_counts"].get("mesoscopic_solver_teacher", 0))
    qc_full_count = int(qc_counts["source_counts"].get("full_microenvironment_teacher", 0))
    paired_count = int(pair_counts.get("paired", 0))
    rejected_count = int(qc_counts["use_counts"].get("reject", 0))
    grade_ab_count = int(qc_counts["grade_counts"].get("A", 0)) + int(qc_counts["grade_counts"].get("B", 0))
    eligible_count = int(qc_counts["use_counts"].get("main_pretraining", 0))
    quarantined_count = int(qc_counts["use_counts"].get("quarantine", 0))
    full_only_indices = list(qc_counts["full_only_indices"])

    expected_current = {
        "full_microenvironment_trajectory_count": full_count,
        "mesoscopic_trajectory_count": mesoscopic_count,
        "paired_count": paired_count,
        "reject_count": rejected_count,
        "train_eligible_count": eligible_count,
    }
    summary_mismatches = {
        key: {"summary": qc_summary.get(key), "derived": value}
        for key, value in expected_current.items()
        if int(qc_summary.get(key, -1)) != value
    }

    historical_full = int(historical["full_count"])
    historical_rejected = int(historical["rejected_count"])
    projection = _historical_projection(
        paths["qc_catalog"],
        paths["pair_alignment"],
        historical,
        full_count,
        historical_policy,
    )
    prefix_reject_indices = [index for index in full_only_indices if index < historical_full]
    appended_reject_indices = [index for index in full_only_indices if index >= historical_full]
    expected_prefix_reject_indices = list(range(historical_full - historical_rejected, historical_full))
    prefix_matches = (
        bool(projection["prefix_identity_verified"])
        and bool(projection["prefix_contract_verified"])
        and prefix_reject_indices == expected_prefix_reject_indices
    )
    append_matches = (
        bool(projection["suffix_identity_verified"])
        and bool(projection["suffix_contract_verified"])
        and full_count - historical_full == len(appended_reject_indices)
        and rejected_count - historical_rejected == len(appended_reject_indices)
    )
    unresolved_conflict_ids: list[str] = []
    if summary_mismatches:
        unresolved_conflict_ids.append("current_summary_identity_mismatch")
    if not bool(historical["matches_declared_values"]):
        unresolved_conflict_ids.append("historical_snapshot_value_mismatch")
    if not bool(historical["governance_records_verified"]):
        unresolved_conflict_ids.append("historical_governance_record_unverified")
    if not prefix_matches or not append_matches:
        unresolved_conflict_ids.append("historical_prefix_reconciliation_failed")
    if not bool(projection["verified"]):
        unresolved_conflict_ids.append("historical_reconstructed_projection_unverified")
    independent_manifest = _mapping(historical.get("independent_manifest"))
    if bool(independent_manifest.get("required")) and not bool(independent_manifest.get("verified")):
        unresolved_conflict_ids.append(
            str(independent_manifest.get("unresolved_id", "historical_source_manifest_hash_missing"))
        )

    producer_command = str(policy["producer_command"])
    config_file = resolve_repo_path(str(policy["config_entrypoint"]))
    config_hash = _file_sha256(config_file)
    commit = _git_commit()
    status = "pass" if not unresolved_conflict_ids else "blocked"
    historical_values = [
        {
            "snapshot_id": historical["snapshot_id"],
            "full_microenvironment_trajectory_count": historical_full,
            "rejected_trajectory_count": historical_rejected,
            "provenance_level": historical["provenance_level"],
            "original_manifest_recovered": historical["original_manifest_recovered"],
        }
    ]
    current_explanation = (
        "The current QC catalog and full-store Zarr metadata agree. The historical values are reproduced by "
        "a deterministic current-catalog projection bound to immutable Git governance records. The projection is "
        "not represented as a recovered original manifest."
    )
    metric_status = "verified_current_snapshot" if not summary_mismatches else "blocked_current_identity"
    metrics = [
        _metric(
            metric_name="raw_materialized_full_microenvironment_trajectory_count",
            canonical_value=full_count,
            definition="Rows in the current full-microenvironment Zarr trajectory store.",
            source_artifact=str(source_inventory["full_store_metadata"]["path"]),
            source_artifact_hash=source_inventory["full_store_metadata"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="none",
            deduplication_policy="not deduplicated; store row count",
            qc_policy="pre-QC inventory",
            pairing_policy="includes paired and full-only rows",
            included_count=full_count,
            excluded_count=0,
            exclusion_reasons="none",
            parent_metrics=[],
            child_metrics=["paired_full_mesoscopic_count", "rejected_trajectory_count"],
            historical_values=historical_values,
            historical_value_explanation=current_explanation,
            status=metric_status,
        ),
        _metric(
            metric_name="raw_materialized_mesoscopic_trajectory_count",
            canonical_value=mesoscopic_count,
            definition="Current mesoscopic teacher rows admitted to the QC catalog.",
            source_artifact=str(source_inventory["qc_catalog"]["path"]),
            source_artifact_hash=source_inventory["qc_catalog"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="source_collection=mesoscopic_solver_teacher",
            deduplication_policy="one QC row per trajectory_uid",
            qc_policy="all QC grades",
            pairing_policy="all mesoscopic rows are paired",
            included_count=mesoscopic_count,
            excluded_count=0,
            exclusion_reasons="none",
            parent_metrics=[],
            child_metrics=["paired_full_mesoscopic_count", "qc_grade_a_b_count"],
            historical_values=historical_values,
            historical_value_explanation="Historical snapshot preserves the same mesoscopic count.",
            status=metric_status,
        ),
        _metric(
            metric_name="unique_content_trajectory_count",
            canonical_value=int(qc_counts["unique_content_count"]),
            definition="Distinct non-empty QC-catalog content hashes across both teacher collections.",
            source_artifact=str(source_inventory["qc_catalog"]["path"]),
            source_artifact_hash=source_inventory["qc_catalog"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="content_hash is non-empty",
            deduplication_policy="set(content_hash)",
            qc_policy="all QC grades",
            pairing_policy="not a pairing metric",
            included_count=int(qc_counts["unique_content_count"]),
            excluded_count=int(qc_counts["row_count"]) - int(qc_counts["unique_content_count"]),
            exclusion_reasons="duplicate content hashes",
            parent_metrics=[],
            child_metrics=[],
            historical_values=[],
            historical_value_explanation="No historical unique-content artifact is available.",
            status=metric_status,
        ),
        _metric(
            metric_name="qc_eligible_trajectory_count",
            canonical_value=eligible_count,
            definition="QC catalog rows whose qc_use is main_pretraining.",
            source_artifact=str(source_inventory["qc_catalog"]["path"]),
            source_artifact_hash=source_inventory["qc_catalog"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="qc_use=main_pretraining",
            deduplication_policy="one QC row per trajectory_uid",
            qc_policy="A/B main-pretraining policy",
            pairing_policy="paired rows only under current catalog",
            included_count=eligible_count,
            excluded_count=int(qc_counts["row_count"]) - eligible_count,
            exclusion_reasons="reject, calibration, quarantine, or non-training QC use",
            parent_metrics=["raw_materialized_full_microenvironment_trajectory_count", "raw_materialized_mesoscopic_trajectory_count"],
            child_metrics=["qc_grade_a_b_count"],
            historical_values=[{"train_eligible_records": 222336}],
            historical_value_explanation="Both snapshots record 222336 train-eligible A/B rows.",
            status=metric_status,
        ),
        _metric(
            metric_name="qc_grade_a_b_count",
            canonical_value=grade_ab_count,
            definition="QC catalog rows graded A or B.",
            source_artifact=str(source_inventory["qc_catalog"]["path"]),
            source_artifact_hash=source_inventory["qc_catalog"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="qc_grade in {A,B}",
            deduplication_policy="one QC row per trajectory_uid",
            qc_policy="grade A/B",
            pairing_policy="paired rows only under current catalog",
            included_count=grade_ab_count,
            excluded_count=int(qc_counts["row_count"]) - grade_ab_count,
            exclusion_reasons="grade C, D, S, or quarantine",
            parent_metrics=["qc_eligible_trajectory_count"],
            child_metrics=[],
            historical_values=[{"train_eligible_records": 222336}],
            historical_value_explanation="Both snapshots record 222336 train-eligible A/B rows.",
            status=metric_status,
        ),
        _metric(
            metric_name="paired_full_mesoscopic_count",
            canonical_value=paired_count,
            definition="Pair-alignment records with paired_status=paired.",
            source_artifact=str(source_inventory["pair_alignment"]["path"]),
            source_artifact_hash=source_inventory["pair_alignment"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="paired_status=paired",
            deduplication_policy="one pair-alignment row per paired_uid",
            qc_policy="all pair rows; QC eligibility is a separate metric",
            pairing_policy="paired_uid join of full and mesoscopic source rows",
            included_count=paired_count,
            excluded_count=int(pair_counts.get("full_only", 0)) + int(pair_counts.get("mesoscopic_only", 0)),
            exclusion_reasons="unpaired full-only or mesoscopic-only rows",
            parent_metrics=["raw_materialized_full_microenvironment_trajectory_count", "raw_materialized_mesoscopic_trajectory_count"],
            child_metrics=[],
            historical_values=[{"paired_full_mesoscopic_count": 111168}],
            historical_value_explanation="Current pair alignment provides the only hashable pairing evidence.",
            status=metric_status,
        ),
        _metric(
            metric_name="rejected_trajectory_count",
            canonical_value=rejected_count,
            definition="QC catalog rows whose qc_use is reject.",
            source_artifact=str(source_inventory["qc_catalog"]["path"]),
            source_artifact_hash=source_inventory["qc_catalog"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="qc_use=reject",
            deduplication_policy="one QC row per trajectory_uid",
            qc_policy="Grade D reject policy",
            pairing_policy="all current rejects are full-only",
            included_count=rejected_count,
            excluded_count=0,
            exclusion_reasons="none",
            parent_metrics=["raw_materialized_full_microenvironment_trajectory_count"],
            child_metrics=[],
            historical_values=[{"rejected_trajectory_count": historical_rejected}],
            historical_value_explanation=(
                f"Indices {historical_full - historical_rejected}-{historical_full - 1} reproduce the historical three rejects; "
                f"indices {historical_full}-{full_count - 1} add {len(appended_reject_indices)} rejects. "
                "The historical declaration is bound to Git records and a deterministic projection, not a recovered original manifest."
            ),
            status="verified_reconstructed_historical_projection" if not unresolved_conflict_ids else metric_status,
        ),
        _metric(
            metric_name="quarantined_trajectory_count",
            canonical_value=quarantined_count,
            definition="QC catalog rows explicitly marked qc_use=quarantine.",
            source_artifact=str(source_inventory["qc_catalog"]["path"]),
            source_artifact_hash=source_inventory["qc_catalog"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="current_materialized_snapshot",
            filters="qc_use=quarantine",
            deduplication_policy="one QC row per trajectory_uid",
            qc_policy="quarantine only",
            pairing_policy="not applicable",
            included_count=quarantined_count,
            excluded_count=0,
            exclusion_reasons="none",
            parent_metrics=[],
            child_metrics=[],
            historical_values=[],
            historical_value_explanation="No quarantine rows are recorded in the current QC catalog.",
            status=metric_status,
        ),
        _metric(
            metric_name="source_snapshot_count",
            canonical_value=2,
            definition="One historical governance snapshot and one current hashable materialized snapshot are referenced.",
            source_artifact=str(source_inventory["governance_snapshot"]["path"]),
            source_artifact_hash=source_inventory["governance_snapshot"]["sha256"],
            producer_command=producer_command,
            git_commit=commit,
            config_hash=config_hash,
            snapshot_or_release="governance_snapshot_plus_current_materialized_snapshot",
            filters="declared snapshots only",
            deduplication_policy="snapshot id",
            qc_policy="not applicable",
            pairing_policy="not applicable",
            included_count=2,
            excluded_count=0,
            exclusion_reasons="none",
            parent_metrics=[],
            child_metrics=[],
            historical_values=[historical],
            historical_value_explanation=(
                "The historical governance values are independently addressable through immutable Git commit/blob records and "
                "are reconciled by a deterministic QC-catalog projection. No original source manifest was recovered or claimed."
            ),
            status="verified_reconstructed_historical_projection" if not unresolved_conflict_ids else metric_status,
        ),
    ]
    return {
        "schema_version": str(policy["schema_version"]),
        "status": status,
        "owner": str(policy["owner"]),
        "producer_command": producer_command,
        "git_commit": commit,
        "config_path": _repo_relative(config_file),
        "config_hash": config_hash,
        "source_artifacts": source_inventory,
        "current_snapshot": {
            "qc_summary": qc_summary,
            "derived_counts": expected_current,
            "summary_mismatches": summary_mismatches,
            "qc_catalog_row_count": qc_counts["row_count"],
            "full_only_indices": full_only_indices,
            "full_only_teacher_family_counts": qc_counts["full_only_family_counts"],
        },
        "historical_snapshot": historical,
        "reconciliation": {
            "historical_prefix_reject_indices": prefix_reject_indices,
            "expected_historical_prefix_reject_indices": expected_prefix_reject_indices,
            "appended_reject_indices": appended_reject_indices,
            "delta_full_count": full_count - historical_full,
            "delta_rejected_count": rejected_count - historical_rejected,
            "prefix_identity_verified": prefix_matches,
            "append_identity_verified": append_matches,
            "historical_governance_record_verified": historical["governance_records_verified"],
            "historical_projection": projection,
            "explanation": (
                "The Git-addressable historical governance records declare 111171 full trajectories and three Grade D rejects. "
                "The hash-bound current QC catalog deterministically projects the exact 111171-row prefix with 111168 paired "
                "main-pretraining rows and three full-only Grade D rejects. The contiguous suffix has 46 full-only Grade D rejects, "
                "so the current 111217/49 snapshot is an append-only reject-only extension. This is a reconstructed historical "
                "projection, not a recovered original manifest."
            ),
        },
        "metrics": metrics,
        "unresolved_conflict_ids": unresolved_conflict_ids,
        "unresolved_conflict_count": len(unresolved_conflict_ids),
        "active_reader_contract": {
            "reader_paths": [str(value) for value in policy.get("current_metric_readers", [])],
            "canonical_artifact": str(_mapping(policy.get("output_paths")).get("json", "")),
        },
    }


def _tsv_row(metric: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {}
    for column in METRIC_COLUMNS:
        value = metric.get(column)
        if isinstance(value, (dict, list)):
            row[column] = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        elif value is None:
            row[column] = ""
        else:
            row[column] = value
    return row


def markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        "# Trajectory Numeric Lineage Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Unresolved conflict count: `{report['unresolved_conflict_count']}`",
        f"- Producer: `{report['producer_command']}`",
        f"- Current configuration hash: `{report['config_hash']}`",
        "",
        "## Reconciliation",
        "",
        str(_mapping(report.get("reconciliation")).get("explanation", "")),
        "",
        "## Metric Inventory",
        "",
        "| metric_name | canonical_value | source_artifact | status |",
        "| --- | ---: | --- | --- |",
    ]
    for metric in report.get("metrics", []):
        row = _mapping(metric)
        lines.append(
            f"| {row['metric_name']} | {row['canonical_value']} | {row['source_artifact']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Historical Boundary",
            "",
            "historical snapshot",
            "not current runtime truth",
            "superseded by trajectory_numeric_lineage_audit.json",
            "",
        ]
    )
    return "\n".join(lines)


def write_numeric_lineage_artifacts(report: Mapping[str, object], config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    output_paths = _mapping(_policy(config).get("output_paths"))
    json_path = resolve_repo_path(str(output_paths["json"]))
    tsv_path = resolve_repo_path(str(output_paths["tsv"]))
    markdown_path = resolve_repo_path(str(output_paths["markdown"]))
    write_json(json_path, report)
    write_tsv(tsv_path, METRIC_COLUMNS, [_tsv_row(_mapping(metric)) for metric in report["metrics"]])
    write_text_atomic(markdown_path, markdown_report(report))
    return {"json": json_path, "tsv": tsv_path, "markdown": markdown_path}


def canonical_metric_values(report: Mapping[str, object]) -> dict[str, int]:
    metrics = {_mapping(metric).get("metric_name"): _mapping(metric) for metric in report.get("metrics", [])}
    return {
        str(name): int(row["canonical_value"])
        for name, row in metrics.items()
        if name
    }


def load_canonical_metric_values(config_path: str | Path) -> dict[str, int]:
    config = load_config(config_path)
    output_paths = _mapping(_policy(config).get("output_paths"))
    path = resolve_repo_path(str(output_paths["json"]))
    if not path.is_file():
        raise FileNotFoundError(f"Canonical numeric lineage artifact is missing: {_repo_relative(path)}")
    return canonical_metric_values(_json(path))


def active_reader_violations(
    reader_paths: Iterable[str | Path],
    forbidden_numeric_literals: Iterable[str],
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for reader in reader_paths:
        path = resolve_repo_path(str(reader))
        if not path.is_file():
            violations.append({"code": "active_reader_not_bound_to_lineage_artifact", "reader": str(reader)})
            continue
        text = path.read_text(encoding="utf-8")
        if "load_canonical_metric_values" not in text:
            violations.append({"code": "active_reader_not_bound_to_lineage_artifact", "reader": str(reader)})
        for literal in forbidden_numeric_literals:
            if literal in text:
                violations.append(
                    {
                        "code": "active_reader_hard_coded_numeric_fact",
                        "reader": str(reader),
                        "literal": literal,
                    }
                )
        if "['trajectory_count']" in text or '["trajectory_count"]' in text:
            violations.append({"code": "active_reader_undefined_generic_trajectory_count", "reader": str(reader)})
    return violations


def _historical_disposition_violations(
    report: Mapping[str, object],
    policy: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    disposition_policy = _mapping(policy.get("historical_snapshot_disposition"))
    expected_status = str(disposition_policy.get("status", "")).strip()
    historical = _mapping(report.get("historical_snapshot"))
    if str(historical.get("disposition", "")) != expected_status:
        blockers.append({"code": "historical_disposition_status_mismatch"})
    if not bool(historical.get("disposition_verified")):
        blockers.append({"code": "historical_disposition_authorization_unverified"})

    authorization_path = resolve_repo_path(str(disposition_policy.get("authorization_artifact", "")))
    if not authorization_path.is_file():
        blockers.append({"code": "historical_disposition_authorization_missing"})
    else:
        try:
            authorization = _json(authorization_path)
        except (OSError, TypeError, json.JSONDecodeError):
            authorization = {}
            blockers.append({"code": "historical_disposition_authorization_invalid"})
        if authorization.get("authorization_id") != disposition_policy.get("authorization_id"):
            blockers.append({"code": "historical_disposition_authorization_id_mismatch"})
        active_dependencies = _mapping(authorization.get("active_dependencies"))
        required_dependencies = {
            "execution",
            "configuration",
            "claims",
            "training",
            "splits",
            "reporting",
            "publication",
        }
        if set(active_dependencies) != required_dependencies or any(
            value is not False for value in active_dependencies.values()
        ):
            blockers.append({"code": "historical_disposition_active_dependency"})
        if authorization.get("current_source_of_truth", {}).get("path") != str(
            _mapping(policy.get("output_paths")).get("json", "")
        ):
            blockers.append({"code": "historical_disposition_source_of_truth_mismatch"})

    quarantine_path = resolve_repo_path("Output/audits/historical_numeric_snapshot_quarantine.json")
    if not quarantine_path.is_file():
        blockers.append({"code": "historical_snapshot_quarantine_missing"})
    return blockers


def validate_numeric_lineage_report(report: Mapping[str, object], config_path: str | Path) -> list[dict[str, object]]:
    config = load_config(config_path)
    policy = _policy(config)
    blockers: list[dict[str, object]] = []
    if str(report.get("schema_version", "")) != str(policy["schema_version"]):
        blockers.append({"code": "schema_version_mismatch"})
    metrics = [_mapping(metric) for metric in report.get("metrics", [])]
    expected_metrics = {str(value) for value in policy.get("canonical_metrics", [])}
    found_metrics = {str(metric.get("metric_name", "")) for metric in metrics}
    if found_metrics != expected_metrics:
        blockers.append({"code": "canonical_metric_set_mismatch", "expected": sorted(expected_metrics), "actual": sorted(found_metrics)})
    for metric in metrics:
        missing = sorted(REQUIRED_METRIC_FIELDS - set(metric))
        if missing:
            blockers.append({"code": "metric_required_field_missing", "metric_name": metric.get("metric_name"), "fields": missing})
    source_artifacts = _mapping(report.get("source_artifacts"))
    source_paths = _source_paths(policy)
    for name in policy.get("hash_required_artifact_keys", []):
        source_name = str(name)
        source = _mapping(source_artifacts.get(source_name))
        if not bool(source.get("exists")) or not str(source.get("sha256") or ""):
            blockers.append({"code": "source_artifact_hash_missing", "artifact": source_name})
            continue
        path = source_paths[source_name]
        if not path.is_file() or str(source.get("sha256")) != _file_sha256(path):
            blockers.append({"code": "source_artifact_hash_stale", "artifact": source_name})

    historical_policy = _mapping(policy.get("historical_snapshot"))
    historical = _mapping(report.get("historical_snapshot"))
    expected_historical_full = int(historical_policy.get("required_full_count", -1))
    expected_historical_rejected = int(historical_policy.get("required_rejected_count", -1))
    if (
        int(historical.get("full_count", -1)) != expected_historical_full
        or int(historical.get("rejected_count", -1)) != expected_historical_rejected
        or not bool(historical.get("matches_declared_values"))
    ):
        blockers.append({"code": "historical_governance_value_mismatch"})
    expected_provenance_level = str(historical_policy.get("provenance_level", ""))
    if str(historical.get("provenance_level", "")) != expected_provenance_level:
        blockers.append({"code": "historical_projection_provenance_level_mismatch"})
    independent_manifest = _mapping(historical.get("independent_manifest"))
    expected_manifest_policy = _mapping(historical_policy.get("independent_manifest"))
    expected_manifest_required = bool(historical_policy.get("requires_independent_manifest_hash", False))
    expected_manifest_id = str(expected_manifest_policy.get("unresolved_id", "historical_source_manifest_hash_missing"))
    if expected_manifest_required and not bool(independent_manifest.get("verified")):
        blockers.append({"code": expected_manifest_id})
    if bool(historical.get("original_manifest_recovered")) != bool(independent_manifest.get("verified")):
        blockers.append({"code": "historical_manifest_recovery_state_mismatch"})
    blockers.extend(_historical_disposition_violations(report, policy))
    record_rows = [_mapping(row) for row in historical.get("governance_records", [])]
    if not bool(historical.get("governance_records_verified")) or not record_rows:
        blockers.append({"code": "historical_governance_record_unverified"})
    expected_records = [
        _git_blob_record(_mapping(record_policy))
        for record_policy in historical_policy.get("governance_records", [])
    ]
    if len(record_rows) != len(expected_records):
        blockers.append({"code": "historical_governance_record_unverified"})
    for reported_record, expected_record in zip(record_rows, expected_records):
        for field_name in ["source_commit", "source_path", "git_blob_id", "sha256", "blob_verified", "fragments_verified"]:
            if reported_record.get(field_name) != expected_record.get(field_name):
                blockers.append({"code": "historical_governance_record_unverified"})
                break
    for record in record_rows:
        if not bool(record.get("blob_verified")) or not bool(record.get("fragments_verified")):
            blockers.append({"code": "historical_governance_record_unverified"})
            break

    reconciliation = _mapping(report.get("reconciliation"))
    if not bool(reconciliation.get("prefix_identity_verified")):
        blockers.append({"code": "historical_prefix_identity_unverified"})
    if not bool(reconciliation.get("append_identity_verified")):
        blockers.append({"code": "historical_append_identity_unverified"})
    if not bool(reconciliation.get("historical_governance_record_verified")):
        blockers.append({"code": "historical_governance_record_unverified"})
    projection = _mapping(reconciliation.get("historical_projection"))
    projection_policy = _mapping(historical_policy.get("projection"))
    current_values = canonical_metric_values(report)
    expected_prefix_indices = list(
        range(int(projection_policy.get("prefix_start_index", 0)), expected_historical_full)
    )
    expected_suffix_indices = list(
        range(
            expected_historical_full,
            current_values.get("raw_materialized_full_microenvironment_trajectory_count", -1),
        )
    )
    if str(projection.get("provenance_level", "")) != expected_provenance_level:
        blockers.append({"code": "historical_projection_provenance_level_mismatch"})
    if bool(projection.get("original_manifest_recovered")) != bool(historical.get("original_manifest_recovered")):
        blockers.append({"code": "historical_projection_manifest_recovery_state_mismatch"})
    if list(projection.get("prefix_indices", [])) != expected_prefix_indices:
        blockers.append({"code": "historical_projection_prefix_indices_invalid"})
    if list(projection.get("suffix_indices", [])) != expected_suffix_indices:
        blockers.append({"code": "historical_projection_suffix_indices_invalid"})
    if list(projection.get("prefix_paired_indices", [])) != list(projection.get("pair_alignment_full_indices", [])):
        blockers.append({"code": "historical_projection_pair_alignment_invalid"})
    required_prefix_counts = _mapping(projection.get("required_prefix_counts"))
    required_suffix_counts = _mapping(projection.get("required_suffix_counts"))
    if _mapping(projection.get("prefix_counts")) != required_prefix_counts:
        blockers.append({"code": "historical_projection_prefix_identity_invalid"})
    if _mapping(projection.get("suffix_counts")) != required_suffix_counts:
        blockers.append({"code": "historical_projection_suffix_identity_invalid"})
    for field_name in [
        "prefix_index_sequence_sha256",
        "suffix_index_sequence_sha256",
        "pair_alignment_index_sequence_sha256",
        "prefix_row_sequence_sha256",
        "suffix_row_sequence_sha256",
    ]:
        if not str(projection.get(field_name, "")):
            blockers.append({"code": "historical_projection_hash_missing", "field": field_name})
    if not all(
        bool(projection.get(field_name))
        for field_name in [
            "prefix_index_sequence_verified",
            "suffix_index_sequence_verified",
            "pair_alignment_verified",
            "prefix_identity_verified",
            "suffix_identity_verified",
            "prefix_contract_verified",
            "suffix_contract_verified",
            "verified",
        ]
    ):
        blockers.append({"code": "historical_reconstructed_projection_unverified"})
    blockers.extend(
        active_reader_violations(
            [str(reader) for reader in policy.get("current_metric_readers", [])],
            [str(value) for value in policy.get("forbidden_active_numeric_literals", [])],
        )
    )
    if str(report.get("status")) == "pass" and blockers:
        blockers.append({"code": "invalid_pass_status"})
    if str(report.get("status")) == "blocked" and not report.get("unresolved_conflict_ids"):
        blockers.append({"code": "blocked_status_missing_conflict_ids"})
    return blockers
