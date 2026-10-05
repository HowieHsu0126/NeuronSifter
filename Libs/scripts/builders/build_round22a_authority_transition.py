"""Resolve Round-22A Tier A/B failure and Tier C replay eligibility.

This builder consumes the frozen WP1 inventory. It never reconstructs an
authority artifact and never starts Round-21R replay. Tier C is eligible only
when a machine-readable Round-20Q contract is explicitly registered and fully
validated; the current contract has no such registration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    FREEZE_COMPONENT_SPECS,
    canonical_sha256,
    load_contract,
    select_validation_checkpoint,
)
from Libs.scripts.builders.build_round22a_authority_recovery import (  # noqa: E402
    ALLOWED_COMPUTE_NODES,
    OUTPUTS as WP1_OUTPUTS,
    REQUIRED_AUTHORITY_INPUTS,
    ROUND22A_ID,
    _git_blob,
    _relative_checkpoint_path,
    _sha256_bytes,
    _validate_candidate,
    _write_immutable_json,
)
from Libs.scripts.builders.build_round22r_empirical_prerequisites import C2B_BLOCKED  # noqa: E402
from Libs.scripts.validators.validate_round22a_authority_recovery import (  # noqa: E402
    validate as validate_wp1,
    validate_scheduler_reconciliation,
)


PREDECESSOR_OUTPUT = "Output/audits/round22a_authority_recovery_resolution_v2.json"
OUTPUT = "Output/audits/round22a_authority_recovery_resolution_v3.json"
TRANSITION_SCHEMA = "neuronsifter-round22a-authority-recovery-resolution-v3"
EXPECTED_RECOVERY_CUTOVER = (
    "0df2deea6e17e8ce8b41d818d078cba389879374",
    "910b45e50c002bcfc17bf9b5bda07f1fa34e086b",
)
ROUND20Q_COMPONENTS = (
    "data",
    "splits",
    "baseline_frontier",
    "challenge_thresholds",
    "training_authorization",
    "config_family",
    "seed_protocol",
)
MACHINE_SUFFIXES = {".json", ".yaml", ".yml", ".tsv"}
ROUND20Q_PATTERN = re.compile(r"round[-_ ]?20q", re.IGNORECASE)
TRANSITION_ENTRY_CAP = 250000
SUCCESS_STATUSES = {"passed", "completed", "resolved", "frozen", "validated"}
TIER_B_SCHEMA = "neuronsifter-round22a-tier-b-reconstruction-manifest-v1"
ROUND20Q_SCHEMA = "neuronsifter-round20q-replay-contract-v1"
TIER_A_ARCHIVE_INDEX_SCHEMA = "neuronsifter-round22a-tier-a-archive-index-v1"
GIT_BLOB_PROVENANCE = "git_blob_at_declared_commit"
MAX_MACHINE_METADATA_BYTES = 8 * 1024 * 1024

# Tier-B completeness is target-specific. A non-empty bag of files is not a
# reconstruction lineage: each wrapper needs the exact immutable inputs that
# define its scientific meaning.
TIER_B_REQUIRED_COMPONENTS = {
    "round21_training_input_freeze": (
        "model_config",
        "c1_data_manifest",
        "c1_split_manifest",
        "c2s_data_manifest",
        "c2s_split_manifest",
        "environment_lock",
        "tri_input_schema",
        "training_authorization",
    ),
    "round21_checkpoint_catalog": (
        "validation_checkpoint",
        "model_config",
        "training_metrics",
        "seed_protocol",
        "training_authorization",
    ),
    "round21_static_twin_evidence": (
        "validation_checkpoint",
        "model_config",
        "c1_data_manifest",
        "c1_split_manifest",
        "c2s_data_manifest",
        "c2s_split_manifest",
        "environment_lock",
        "baseline_frontier",
        "tri_input_schema",
        "round21_claim_evidence",
    ),
    "round21_c3_readiness": (
        "static_twin_evidence",
        "tool_outcome_contract",
        "c3_protocol",
        "hidden_evaluation_ledger",
    ),
    "round21_coevolution_prerequisite_status": (
        "static_twin_evidence",
        "c1_claim_gate",
        "c2s_claim_gate",
        "c3_readiness",
        "coevolution_contract",
    ),
    "c1_neuronsifter_results": (
        "c1_raw_result_matrix",
        "c1_paired_statistical_units",
        "c1_metric_contract",
        "validation_checkpoint",
        "model_config",
        "c1_data_manifest",
        "c1_split_manifest",
    ),
    "c2s_neuronsifter_results": (
        "c2s_raw_result_matrix",
        "c2s_paired_statistical_units",
        "c2s_metric_contract",
        "validation_checkpoint",
        "model_config",
        "c2s_data_manifest",
        "c2s_split_manifest",
    ),
    "round21_c1_claim_gate": (
        "c1_results",
        "c1_paired_statistical_units",
        "c1_claim_thresholds",
        "validation_checkpoint",
        "model_config",
        "c1_data_manifest",
        "c1_split_manifest",
    ),
    "round21_c2s_claim_gate": (
        "c2s_results",
        "c2s_paired_statistical_units",
        "c2s_claim_thresholds",
        "validation_checkpoint",
        "model_config",
        "c2s_data_manifest",
        "c2s_split_manifest",
    ),
}

ROUND20Q_COMPONENT_CONTRACTS = {
    "data": ("neuronsifter-round20q-data-freeze-v1", "data_identity_hash"),
    "splits": ("neuronsifter-round20q-split-freeze-v1", "split_identity_hash"),
    "baseline_frontier": (
        "neuronsifter-round20q-baseline-frontier-freeze-v1",
        "baseline_frontier_hash",
    ),
    "challenge_thresholds": (
        "neuronsifter-round20q-challenge-threshold-freeze-v1",
        "challenge_thresholds_hash",
    ),
    "training_authorization": (
        "neuronsifter-round20q-training-authorization-freeze-v1",
        "training_authorization_hash",
    ),
    "config_family": (
        "neuronsifter-round20q-config-family-freeze-v1",
        "config_family_hash",
    ),
    "seed_protocol": (
        "neuronsifter-round20q-seed-protocol-freeze-v1",
        "seed_protocol_hash",
    ),
}
ROUND20Q_COMPONENT_BOOLEAN_FIELDS = {
    "data": ("public_only",),
    "splits": ("validation_only_checkpoint_selection", "final_holdout_unseen"),
    "baseline_frontier": ("baseline_frontier_frozen",),
    "challenge_thresholds": ("challenge_thresholds_frozen",),
    "training_authorization": ("training_authorized",),
    "config_family": ("config_family_frozen",),
    "seed_protocol": ("seed_protocol_frozen",),
}


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _implementation_binding(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Round-22A implementation binding missing: {relative_path}")
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git_lines(root: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _git_optional_lines(root: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _commit_is_at_or_before_cutover(
    root: Path, commit: object, cutover_commit: str
) -> bool:
    commit_text = str(commit or "").strip()
    if not _is_git_oid(commit_text):
        return False
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            commit_text,
            cutover_commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _validate_recovery_cutover(
    root: Path, recovery: Mapping[str, object]
) -> dict[str, str]:
    identity = recovery.get("recovery_cutover_git_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Round-22A recovery cutover identity is missing")
    commit = str(identity.get("commit", "")).strip()
    tree = str(identity.get("tree", "")).strip()
    if (
        root.resolve() == REPO_ROOT.resolve()
        and (commit, tree) != EXPECTED_RECOVERY_CUTOVER
    ):
        raise ValueError("Round-22A production recovery cutover identity drifted")
    if _git_optional_lines(root, "rev-parse", "--verify", f"{commit}^{{commit}}") != [
        commit
    ] or _git_optional_lines(root, "show", "-s", "--format=%T", commit) != [tree]:
        raise ValueError("Round-22A recovery cutover Git identity is invalid")
    return {"commit": commit, "tree": tree}


def _historical_builder_evidence(root: Path, cutover_commit: str) -> dict[str, object]:
    target_pattern = "|".join(
        re.escape(Path(path).name) for path in REQUIRED_AUTHORITY_INPUTS.values()
    )
    current_hits = _git_lines(
        root, "grep", "-l", "-E", target_pattern, "--", "Libs", "orchestrator"
    )
    history_commits: set[str] = set()
    for path in REQUIRED_AUTHORITY_INPUTS.values():
        history_commits.update(
            _git_lines(
                root,
                "log",
                "--all",
                "--reflog",
                "--format=%H",
                f"-S{Path(path).name}",
                "--",
                "Libs",
                "orchestrator",
            )
        )
    round22r_builder = (
        root / "Libs/scripts/builders/build_round22r_empirical_prerequisites.py"
    )
    tracked_round22r = bool(
        _git_lines(
            root,
            "ls-files",
            "--error-unmatch",
            "Libs/scripts/builders/build_round22r_empirical_prerequisites.py",
        )
    )
    historical_commits = sorted(history_commits)
    pre_cutover_commits = [
        commit
        for commit in historical_commits
        if _commit_is_at_or_before_cutover(root, commit, cutover_commit)
    ]
    return {
        "tracked_current_builder_hit_count": len(current_hits),
        "tracked_current_builder_hits": sorted(current_hits),
        "historical_builder_commit_count": len(pre_cutover_commits),
        "historical_builder_commits": pre_cutover_commits,
        "excluded_post_cutover_builder_commit_count": len(historical_commits)
        - len(pre_cutover_commits),
        "round22r_builder_exists": round22r_builder.is_file(),
        "round22r_builder_git_tracked": tracked_round22r,
        "round22r_builder_rejected_reason": "post_round21_control_plane_not_historical_builder",
        "builder_identity_complete": False,
    }


def _is_sha(value: object) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_git_oid(value: object) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    return len(text) in {40, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def _path_has_symlink(path: Path, stop: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == stop or current.parent == current:
            return False
        current = current.parent


def _safe_registered_path(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    root_absolute = root.absolute()
    path = Path(text)
    path = path.absolute() if path.is_absolute() else root_absolute / path
    if _path_has_symlink(path, root_absolute):
        return None
    resolved_root = root_absolute.resolve()
    resolved = path.resolve()
    return resolved if resolved.is_relative_to(resolved_root) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_file_binding_errors(
    root: Path,
    path: Path,
    binding: Mapping[str, object],
    *,
    cutover_commit: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if binding.get("provenance_mode") != GIT_BLOB_PROVENANCE:
        return ["component_provenance_mode_invalid"]
    commit = str(binding.get("git_commit", "")).strip()
    tree = str(binding.get("git_tree", "")).strip()
    blob = str(binding.get("git_blob", "")).strip()
    git_path = str(binding.get("git_path", "")).strip()
    if not _is_git_oid(commit) or not _is_git_oid(tree) or not _is_git_oid(blob):
        return ["component_git_identity_invalid"]
    if cutover_commit is not None and not _commit_is_at_or_before_cutover(
        root, commit, cutover_commit
    ):
        errors.append("component_git_commit_after_recovery_cutover")
    git_path_value = Path(git_path)
    if not git_path or git_path_value.is_absolute() or ".." in git_path_value.parts:
        return ["component_git_path_invalid"]
    if _git_optional_lines(root, "show", "-s", "--format=%T", commit) != [tree]:
        errors.append("component_git_tree_mismatch")
    resolved_blob = _git_optional_lines(
        root, "rev-parse", "--verify", f"{commit}:{git_path}"
    )
    if resolved_blob != [blob]:
        errors.append("component_git_blob_mismatch")
    else:
        file_blob = _git_optional_lines(root, "hash-object", "--no-filters", str(path))
        if file_blob != [blob]:
            errors.append("component_bytes_not_git_bound")
    return errors


def _producer_uid(payload: Mapping[str, object]) -> str:
    producer = payload.get("producer")
    if isinstance(producer, Mapping):
        return str(producer.get("uid", "")).strip()
    return str(payload.get("producer_uid", producer or "")).strip()


def _git_identity_values(payload: Mapping[str, object]) -> tuple[str, str, bool | None]:
    identity = payload.get("git_identity")
    if isinstance(identity, Mapping):
        return (
            str(identity.get("commit", "")).strip(),
            str(identity.get("tree", "")).strip(),
            identity.get("dirty") if isinstance(identity.get("dirty"), bool) else None,
        )
    return (
        str(payload.get("git_commit", "")).strip(),
        str(payload.get("git_tree", "")).strip(),
        payload.get("git_dirty")
        if isinstance(payload.get("git_dirty"), bool)
        else None,
    )


def _machine_metadata_errors(path: Path, binding: Mapping[str, object]) -> list[str]:
    content_type = str(binding.get("content_type", "")).strip().lower()
    if content_type == "binary":
        return (
            []
            if str(binding.get("artifact_uid", "")).lower() == _sha256_file(path)
            else ["component_binary_artifact_uid_mismatch"]
        )
    errors: list[str] = []
    if path.stat().st_size > MAX_MACHINE_METADATA_BYTES:
        return ["component_machine_metadata_size_cap_exceeded"]
    try:
        if content_type == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif content_type == "yaml":
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif content_type == "tsv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if not rows:
                return ["component_tsv_empty"]
            for field, expected in (
                ("schema_version", binding.get("schema_version")),
                ("artifact_uid", binding.get("artifact_uid")),
                ("producer_uid", binding.get("producer_uid")),
            ):
                if any(
                    str(row.get(field, "")).strip() != str(expected) for row in rows
                ):
                    errors.append(f"component_{field}_mismatch")
            if str(binding.get("artifact_uid", "")).lower() != _sha256_file(path):
                errors.append("component_tsv_artifact_uid_mismatch")
            return errors
        else:
            return ["component_content_type_invalid"]
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError, csv.Error):
        return ["component_machine_parse_failed"]
    if not isinstance(payload, Mapping):
        return ["component_machine_root_invalid"]
    if str(payload.get("schema_version", "")).strip() != str(
        binding.get("schema_version", "")
    ):
        errors.append("component_schema_version_mismatch")
    if str(payload.get("artifact_uid", "")).strip() != str(
        binding.get("artifact_uid", "")
    ):
        errors.append("component_artifact_uid_mismatch")
    payload_without_uid = dict(payload)
    payload_uid = payload_without_uid.pop("artifact_uid", None)
    if payload_uid != canonical_sha256(payload_without_uid):
        errors.append("component_artifact_uid_not_canonical")
    if _producer_uid(payload) != str(binding.get("producer_uid", "")).strip():
        errors.append("component_producer_uid_mismatch")
    return errors


def _verify_component_binding(
    root: Path,
    binding: Mapping[str, object],
    *,
    cutover_commit: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    path = _safe_registered_path(root, binding.get("path"))
    if path is None:
        errors.append("component_path_outside_repository")
    elif not path.is_file() or path.is_symlink():
        errors.append("component_artifact_missing_or_symlink")
    required_text = (
        "component_id",
        "content_type",
        "schema_version",
        "artifact_uid",
        "producer_uid",
    )
    for field in required_text:
        if not str(binding.get(field, "")).strip():
            errors.append(f"component_{field}_missing")
    if not _is_sha(binding.get("sha256")):
        errors.append("component_sha256_invalid")
    try:
        expected_bytes = int(binding.get("bytes", -1))
    except (TypeError, ValueError):
        expected_bytes = -1
    if expected_bytes < 0:
        errors.append("component_bytes_invalid")
    observed_bytes = 0
    observed_sha256: str | None = None
    if path is not None and path.is_file() and not path.is_symlink():
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256_file(path)
        if observed_bytes != expected_bytes:
            errors.append("component_byte_count_mismatch")
        if observed_sha256 != str(binding.get("sha256", "")).lower():
            errors.append("component_sha256_mismatch")
        errors.extend(
            _git_file_binding_errors(root, path, binding, cutover_commit=cutover_commit)
        )
        errors.extend(_machine_metadata_errors(path, binding))
    return (
        {
            "component_id": str(binding.get("component_id", "")),
            "path": path.relative_to(root.resolve()).as_posix()
            if path is not None and path.is_relative_to(root.resolve())
            else str(binding.get("path", "")),
            "expected_bytes": expected_bytes,
            "observed_bytes": observed_bytes,
            "expected_sha256": binding.get("sha256"),
            "observed_sha256": observed_sha256,
            "provenance_mode": binding.get("provenance_mode"),
            "git_commit": binding.get("git_commit"),
            "git_tree": binding.get("git_tree"),
            "git_blob": binding.get("git_blob"),
            "git_path": binding.get("git_path"),
            "validated": not errors,
            "errors": sorted(set(errors)),
        },
        errors,
    )


def _tier_b_manifest_core(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: payload.get(key)
        for key in (
            "schema_version",
            "reconstruction_target",
            "status",
            "source_components",
            "builder_identity",
            "deterministic_reconstruction_command",
        )
    }


def _validate_tier_b_manifest(
    root: Path,
    authority_uid: str,
    registration: Mapping[str, object],
    *,
    cutover_commit: str,
) -> dict[str, object]:
    errors: list[str] = []
    manifest_path = _safe_registered_path(root, registration.get("path"))
    if manifest_path is None:
        errors.append("manifest_path_outside_repository")
    elif not manifest_path.is_file() or manifest_path.is_symlink():
        errors.append("manifest_missing_or_symlink")
    if registration.get("authority_uid") != authority_uid:
        errors.append("manifest_registration_target_mismatch")
    if not _is_sha(registration.get("sha256")):
        errors.append("manifest_registration_sha256_invalid")
    payload: Mapping[str, object] | None = None
    if (
        manifest_path is not None
        and manifest_path.is_file()
        and not manifest_path.is_symlink()
    ):
        content = manifest_path.read_bytes()
        if _sha256_bytes(content) != str(registration.get("sha256", "")).lower():
            errors.append("manifest_registration_sha256_mismatch")
        try:
            loaded = json.loads(content.decode("utf-8"))
            payload = loaded if isinstance(loaded, Mapping) else None
        except (UnicodeError, json.JSONDecodeError):
            payload = None
        if payload is None:
            errors.append("manifest_json_invalid")
    component_results: list[dict[str, object]] = []
    component_hashes: dict[str, str] = {}
    manifest_core_uid: str | None = None
    builder_sha256: str | None = None
    if payload is not None:
        if payload.get("schema_version") != TIER_B_SCHEMA:
            errors.append("manifest_schema_version_invalid")
        if payload.get("reconstruction_target") != authority_uid:
            errors.append("manifest_reconstruction_target_mismatch")
        if str(payload.get("status", "")).strip().lower() not in SUCCESS_STATUSES:
            errors.append("manifest_status_not_validated")
        without_uid = dict(payload)
        artifact_uid = without_uid.pop("artifact_uid", None)
        if artifact_uid != canonical_sha256(without_uid):
            errors.append("manifest_artifact_uid_mismatch")
        components = payload.get("source_components")
        if not isinstance(components, list) or not components:
            errors.append("manifest_source_components_missing")
        else:
            component_ids: list[str] = []
            for component in components:
                if not isinstance(component, Mapping):
                    errors.append("manifest_component_row_invalid")
                    continue
                result, component_errors = _verify_component_binding(
                    root, component, cutover_commit=cutover_commit
                )
                component_results.append(result)
                component_id = str(component.get("component_id", ""))
                component_ids.append(component_id)
                if component_id:
                    component_hashes[component_id] = str(
                        component.get("sha256", "")
                    ).lower()
                errors.extend(component_errors)
            if len(component_ids) != len(set(component_ids)):
                errors.append("manifest_component_id_duplicate")
            required_components = set(TIER_B_REQUIRED_COMPONENTS[authority_uid])
            if set(component_ids) != required_components or len(component_ids) != len(
                required_components
            ):
                errors.append("manifest_required_component_set_mismatch")
        manifest_core_uid = canonical_sha256(_tier_b_manifest_core(payload))
        if payload.get("manifest_core_uid") != manifest_core_uid:
            errors.append("manifest_core_uid_mismatch")
        builder = payload.get("builder_identity")
        if not isinstance(builder, Mapping):
            errors.append("manifest_builder_identity_missing")
        else:
            builder_path = _safe_registered_path(root, builder.get("path"))
            builder_commit = str(builder.get("git_commit", "")).strip()
            if not _commit_is_at_or_before_cutover(
                root, builder_commit, cutover_commit
            ):
                errors.append("manifest_builder_commit_after_recovery_cutover")
            if (
                builder_path is None
                or not builder_path.is_file()
                or builder_path.is_symlink()
            ):
                errors.append("manifest_builder_artifact_missing")
            elif not _is_sha(builder.get("sha256")):
                errors.append("manifest_builder_sha256_invalid")
            else:
                builder_content = builder_path.read_bytes()
                builder_sha256 = _sha256_bytes(builder_content)
                if builder_sha256 != str(builder["sha256"]).lower():
                    errors.append("manifest_builder_sha256_mismatch")
                relative_builder = builder_path.relative_to(root.resolve()).as_posix()
                historical_blob = (
                    _git_blob(root, builder_commit, relative_builder)
                    if builder_commit
                    else None
                )
                if historical_blob is None or historical_blob[1] != builder_content:
                    errors.append("manifest_builder_git_identity_mismatch")
            if not str(builder.get("version", "")).strip():
                errors.append("manifest_builder_version_missing")
        command = payload.get("deterministic_reconstruction_command")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or any(not str(value).strip() for value in command)
        ):
            errors.append("manifest_deterministic_command_invalid")
        elif not isinstance(builder, Mapping) or str(command[1]) != str(
            builder.get("path", "")
        ):
            errors.append("manifest_command_builder_binding_mismatch")
        else:
            target_flags = [
                index for index, value in enumerate(command) if value == "--target"
            ]
            if (
                len(target_flags) != 1
                or target_flags[0] + 1 >= len(command)
                or str(command[target_flags[0] + 1]) != authority_uid
            ):
                errors.append("manifest_command_target_binding_mismatch")
        if (
            "semantic_identity_proof" in payload
            or "semantic_identity_proven" in payload
        ):
            errors.append("preexecution_semantic_identity_claim_forbidden")
    return {
        "authority_uid": authority_uid,
        "manifest_path": str(registration.get("path", "")),
        "manifest_sha256": registration.get("sha256"),
        "manifest_core_uid": manifest_core_uid,
        "builder_sha256": builder_sha256,
        "required_component_ids": list(TIER_B_REQUIRED_COMPONENTS[authority_uid]),
        "source_component_sha256": component_hashes,
        "source_components": component_results,
        "semantic_identity_proven": False,
        "deterministic_reconstruction_executed": False,
        "reconstruction_inputs_ready": not errors,
        "validated": not errors,
        "errors": sorted(set(errors)),
    }


def _per_authority_tier_evidence(
    root: Path,
    inventory: list[dict[str, str]],
    registrations: object,
    *,
    cutover_commit: str,
) -> list[dict[str, object]]:
    authority_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    locator_paths: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        authority_uid = row.get("authority_uid", "")
        if row.get("candidate_kind") == "authority_artifact":
            authority_rows[authority_uid].append(row)
        elif row.get("candidate_kind") == "immutable_component_locator":
            try:
                paths = json.loads(row.get("all_candidate_paths_json", "[]"))
            except json.JSONDecodeError:
                paths = []
            locator_paths[authority_uid].update(str(path) for path in paths)
    registered: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    if isinstance(registrations, list):
        for row in registrations:
            if isinstance(row, Mapping):
                registered[str(row.get("authority_uid", ""))].append(row)
    evidence: list[dict[str, object]] = []
    for authority_uid, canonical_path in REQUIRED_AUTHORITY_INPUTS.items():
        authority_registrations = registered[authority_uid]
        manifest_result = (
            _validate_tier_b_manifest(
                root,
                authority_uid,
                authority_registrations[0],
                cutover_commit=cutover_commit,
            )
            if len(authority_registrations) == 1
            else {
                "authority_uid": authority_uid,
                "manifest_path": "",
                "manifest_sha256": None,
                "source_components": [],
                "semantic_identity_proven": False,
                "deterministic_reconstruction_executed": False,
                "reconstruction_inputs_ready": False,
                "validated": False,
                "errors": [
                    "manifest_not_registered"
                    if not authority_registrations
                    else "duplicate_manifest_registrations"
                ],
            }
        )
        tier_b_ready = manifest_result["validated"] is True
        evidence.append(
            {
                "authority_uid": authority_uid,
                "canonical_path": canonical_path,
                "tier_a_candidate_count": len(authority_rows[authority_uid]),
                "tier_b_locator_path_count": len(locator_paths[authority_uid]),
                "tier_b_registered_manifest_count": len(authority_registrations),
                "tier_b_manifest_validation": manifest_result,
                "tier_b_reconstruction_ready": tier_b_ready,
                "recovery_mode": "deterministic_reconstruction_inputs_ready"
                if tier_b_ready
                else "unrecovered",
                "decision_reason": "verified_historical_inputs_and_builder_ready_for_execution"
                if tier_b_ready
                else ";".join(str(value) for value in manifest_result["errors"]),
            }
        )
    return evidence


def _bound_hash(payload: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    values: set[str] = set()
    containers: list[Mapping[str, object]] = [payload]
    for key in ("freeze_components", "hashes"):
        child = payload.get(key)
        if isinstance(child, Mapping):
            containers.append(child)
    for container in containers:
        for key in keys:
            value = str(container.get(key, "")).strip().lower()
            if _is_sha(value):
                values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def _candidate_commit_sets(
    inventory: list[dict[str, str]],
) -> dict[str, set[str]]:
    commits: dict[str, set[str]] = {
        authority_uid: set() for authority_uid in REQUIRED_AUTHORITY_INPUTS
    }
    for row in inventory:
        authority_uid = row.get("authority_uid", "")
        if (
            authority_uid not in commits
            or row.get("candidate_kind") != "authority_artifact"
            or row.get("schema_status") != "valid"
            or row.get("recovery_confidence") != "high"
        ):
            continue
        try:
            values = json.loads(row.get("all_git_commits_json", "[]"))
        except json.JSONDecodeError:
            values = []
        if isinstance(values, list):
            commits[authority_uid].update(
                str(value) for value in values if str(value).strip()
            )
        fallback = str(row.get("git_commit_if_known", "")).strip()
        if fallback:
            commits[authority_uid].add(fallback)
    return commits


def _validate_tier_a_contents(
    root: Path,
    lineage_id: str,
    contents: Mapping[str, bytes],
    checkpoint_loader: Any,
    source_bindings: Mapping[str, object] | None = None,
    *,
    cutover_commit: str | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    json_payloads: dict[str, Mapping[str, object]] = {}
    tsv_rows: dict[str, list[dict[str, str]]] = {}
    for authority_uid, relative_path in REQUIRED_AUTHORITY_INPUTS.items():
        content = contents.get(authority_uid)
        if content is None:
            errors.append(f"{authority_uid}:candidate_bytes_missing")
            continue
        schema_status, _, payload = _validate_candidate(authority_uid, content)
        if schema_status != "valid":
            errors.append(f"{authority_uid}:{schema_status}")
            continue
        if relative_path.endswith(".json"):
            if isinstance(payload, Mapping):
                json_payloads[authority_uid] = payload
            else:
                errors.append(f"{authority_uid}:json_payload_missing")
        else:
            try:
                rows = list(
                    csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter="\t")
                )
            except (UnicodeError, csv.Error):
                rows = []
            if not rows:
                errors.append(f"{authority_uid}:result_rows_missing")
            tsv_rows[authority_uid] = rows
    if errors:
        return {
            "storage_commit": lineage_id,
            "lineage_source": "candidate_content_set",
            "validated": False,
            "identity_uid": None,
            "source_bindings": dict(source_bindings or {}),
            "errors": sorted(set(errors)),
        }

    checkpoint_catalog = json_payloads["round21_checkpoint_catalog"]
    selected = select_validation_checkpoint(checkpoint_catalog)
    if selected is None:
        errors.append("validation_checkpoint_selection_missing")
        checkpoint_hash = None
    else:
        checkpoint_hash = str(selected.get("checkpoint_sha256", "")).strip().lower()
        relative_checkpoint = _relative_checkpoint_path(
            selected.get("checkpoint_path"), root
        )
        checkpoint_content = (
            checkpoint_loader(relative_checkpoint)
            if relative_checkpoint is not None
            else None
        )
        if (
            not _is_sha(checkpoint_hash)
            or checkpoint_content is None
            or _sha256_bytes(checkpoint_content) != checkpoint_hash
        ):
            errors.append("validation_checkpoint_byte_identity_mismatch")

    freeze_values: dict[str, str | None] = {"checkpoint_hash": checkpoint_hash}
    for component, (source_uid, keys) in FREEZE_COMPONENT_SPECS.items():
        source = json_payloads.get(source_uid)
        value = (
            _bound_hash(source, tuple(keys)) if isinstance(source, Mapping) else None
        )
        freeze_values[component] = value
        if value is None:
            errors.append(f"freeze_component_{component}_missing_or_conflicting")

    static_evidence = json_payloads["round21_static_twin_evidence"]
    c1_gate = json_payloads["round21_c1_claim_gate"]
    c2s_gate = json_payloads["round21_c2s_claim_gate"]
    direct_checks = (
        (
            "static_checkpoint",
            static_evidence.get("selected_checkpoint_hash"),
            freeze_values["checkpoint_hash"],
        ),
        (
            "static_model_config",
            static_evidence.get("model_config_hash"),
            freeze_values["model_config_hash"],
        ),
        (
            "static_c1_data",
            static_evidence.get("c1_data_hash"),
            freeze_values["c1_data_hash"],
        ),
        (
            "static_c2s_data",
            static_evidence.get("c2s_data_hash"),
            freeze_values["c2_data_hash"],
        ),
        (
            "static_environment",
            static_evidence.get("environment_hash"),
            freeze_values["environment_hash"],
        ),
        (
            "static_baseline",
            static_evidence.get("baseline_frontier_hash"),
            freeze_values["baseline_frontier_hash"],
        ),
        (
            "static_tri",
            static_evidence.get("tri_input_schema_hash"),
            freeze_values["tri_input_schema_hash"],
        ),
        (
            "c1_gate_artifact",
            static_evidence.get("c1_claim_gate_hash"),
            _sha256_bytes(contents["round21_c1_claim_gate"]),
        ),
        (
            "c2s_gate_artifact",
            static_evidence.get("c2s_claim_gate_hash"),
            _sha256_bytes(contents["round21_c2s_claim_gate"]),
        ),
        (
            "c1_gate_config",
            c1_gate.get("config_hash"),
            freeze_values["model_config_hash"],
        ),
        ("c1_gate_data", c1_gate.get("data_hash"), freeze_values["c1_data_hash"]),
        (
            "c1_gate_checkpoint",
            c1_gate.get("checkpoint_hash"),
            freeze_values["checkpoint_hash"],
        ),
        (
            "c2s_gate_config",
            c2s_gate.get("config_hash"),
            freeze_values["model_config_hash"],
        ),
        ("c2s_gate_data", c2s_gate.get("data_hash"), freeze_values["c2_data_hash"]),
        (
            "c2s_gate_checkpoint",
            c2s_gate.get("checkpoint_hash"),
            freeze_values["checkpoint_hash"],
        ),
        ("c1_split", static_evidence.get("c1_split_hash"), c1_gate.get("split_hash")),
        (
            "c2s_split",
            static_evidence.get("c2s_split_hash"),
            c2s_gate.get("split_hash"),
        ),
    )
    for identity_key, observed, expected in direct_checks:
        if not _is_sha(observed) or str(observed).lower() != str(expected).lower():
            errors.append(f"cross_artifact_identity_mismatch:{identity_key}")

    for input_id, expected in (
        (
            "c1_neuronsifter_results",
            {
                "config_hash": freeze_values["model_config_hash"],
                "data_hash": freeze_values["c1_data_hash"],
                "split_hash": c1_gate.get("split_hash"),
                "checkpoint_hash": freeze_values["checkpoint_hash"],
            },
        ),
        (
            "c2s_neuronsifter_results",
            {
                "config_hash": freeze_values["model_config_hash"],
                "data_hash": freeze_values["c2_data_hash"],
                "split_hash": c2s_gate.get("split_hash"),
                "checkpoint_hash": freeze_values["checkpoint_hash"],
            },
        ),
    ):
        for row in tsv_rows[input_id]:
            if str(row.get("status", "")).strip().lower() not in SUCCESS_STATUSES:
                errors.append(f"{input_id}:result_status_invalid")
            for field, value in expected.items():
                if (
                    not _is_sha(row.get(field))
                    or str(row.get(field)).lower() != str(value).lower()
                ):
                    errors.append(f"{input_id}:cross_identity_mismatch:{field}")

    git_identities: set[tuple[str, str]] = set()
    dirty_values: list[bool] = []
    for payload in json_payloads.values():
        producer_commit, producer_tree, dirty = _git_identity_values(payload)
        if producer_commit and producer_tree:
            git_identities.add((producer_commit, producer_tree))
            if _git_optional_lines(
                root, "show", "-s", "--format=%T", producer_commit
            ) != [producer_tree]:
                errors.append("producer_git_identity_invalid")
            if cutover_commit is not None and not _commit_is_at_or_before_cutover(
                root, producer_commit, cutover_commit
            ):
                errors.append("producer_git_identity_after_recovery_cutover")
        else:
            errors.append("producer_git_identity_missing")
        dirty_values.append(dirty is False)
    for rows in tsv_rows.values():
        for row in rows:
            producer_commit = str(row.get("git_commit", "")).strip()
            producer_tree = str(row.get("git_tree", "")).strip()
            if producer_commit and producer_tree:
                git_identities.add((producer_commit, producer_tree))
                if _git_optional_lines(
                    root, "show", "-s", "--format=%T", producer_commit
                ) != [producer_tree]:
                    errors.append("result_git_identity_invalid")
                if cutover_commit is not None and not _commit_is_at_or_before_cutover(
                    root, producer_commit, cutover_commit
                ):
                    errors.append("result_git_identity_after_recovery_cutover")
            else:
                errors.append("result_git_identity_missing")
            dirty_values.append(
                str(row.get("git_dirty", "")).strip().lower() == "false"
            )
    if len(git_identities) != 1 or not all(dirty_values):
        errors.append("cross_artifact_git_identity_inconsistent")

    identity_payload = {
        "authority_sha256": {
            authority_uid: _sha256_bytes(content)
            for authority_uid, content in sorted(contents.items())
        },
        "checkpoint_hash": checkpoint_hash,
        "producer_git_identity": sorted(git_identities),
    }
    return {
        "storage_commit": lineage_id,
        "lineage_source": "candidate_content_set",
        "validated": not errors,
        "identity_uid": canonical_sha256(identity_payload) if not errors else None,
        "selected_validation_checkpoint": selected,
        "authority_sha256": identity_payload["authority_sha256"],
        "producer_git_identity": sorted(git_identities),
        "source_bindings": dict(source_bindings or {}),
        "errors": sorted(set(errors)),
    }


def _validate_tier_a_commit(
    root: Path, commit: str, *, cutover_commit: str
) -> dict[str, object]:
    contents: dict[str, bytes] = {}
    for authority_uid, relative_path in REQUIRED_AUTHORITY_INPUTS.items():
        candidate = _git_blob(root, commit, relative_path)
        if candidate is not None:
            contents[authority_uid] = candidate[1]

    def load_checkpoint(relative_path: str) -> bytes | None:
        candidate = _git_blob(root, commit, relative_path)
        return candidate[1] if candidate is not None else None

    result = _validate_tier_a_contents(
        root,
        commit,
        contents,
        load_checkpoint,
        {
            authority_uid: f"git:{commit}:{relative_path}"
            for authority_uid, relative_path in REQUIRED_AUTHORITY_INPUTS.items()
            if authority_uid in contents
        },
        cutover_commit=cutover_commit,
    )
    result["lineage_source"] = "git_storage_commit"
    return result


def _authority_producer_identity(
    root: Path,
    authority_uid: str,
    content: bytes,
    *,
    cutover_commit: str,
) -> tuple[str, str] | None:
    schema_status, _, payload = _validate_candidate(authority_uid, content)
    if schema_status != "valid":
        return None
    identities: set[tuple[str, str]] = set()
    clean: list[bool] = []
    if REQUIRED_AUTHORITY_INPUTS[authority_uid].endswith(".json"):
        if not isinstance(payload, Mapping):
            return None
        commit, tree, dirty = _git_identity_values(payload)
        if commit and tree:
            identities.add((commit, tree))
        clean.append(dirty is False)
    else:
        try:
            rows = list(
                csv.DictReader(io.StringIO(content.decode("utf-8")), delimiter="\t")
            )
        except (UnicodeError, csv.Error):
            return None
        for row in rows:
            commit = str(row.get("git_commit", "")).strip()
            tree = str(row.get("git_tree", "")).strip()
            if commit and tree:
                identities.add((commit, tree))
            clean.append(str(row.get("git_dirty", "")).strip().lower() == "false")
    if len(identities) != 1 or not clean or not all(clean):
        return None
    commit, tree = next(iter(identities))
    if not _is_git_oid(commit) or not _is_git_oid(tree):
        return None
    if _git_optional_lines(root, "rev-parse", "--verify", f"{commit}^{{commit}}") != [
        commit
    ]:
        return None
    if _git_optional_lines(root, "show", "-s", "--format=%T", commit) != [tree]:
        return None
    if not _commit_is_at_or_before_cutover(root, commit, cutover_commit):
        return None
    return commit, tree


def _read_candidate_locator(
    root: Path, locator: str, expected_sha256: str
) -> tuple[bytes, Path | None] | None:
    if locator.startswith("git:"):
        _, commit, relative_path = locator.split(":", 2)
        candidate = _git_blob(root, commit, relative_path)
        if candidate is None or _sha256_bytes(candidate[1]) != expected_sha256:
            return None
        return candidate[1], None
    path = Path(locator)
    path = path if path.is_absolute() else root / path
    absolute = path.absolute()
    if _path_has_symlink(absolute, Path(absolute.anchor)) or not absolute.is_file():
        return None
    content = absolute.read_bytes()
    if _sha256_bytes(content) != expected_sha256:
        return None
    return content, absolute


def _candidate_bundle_root(path: Path, canonical_path: str) -> Path | None:
    canonical_parts = Path(canonical_path).parts
    if len(path.parts) < len(canonical_parts):
        return None
    if tuple(path.parts[-len(canonical_parts) :]) != canonical_parts:
        return None
    root = path
    for _ in canonical_parts:
        root = root.parent
    return root


def _normalize_registered_candidate_path(root: Path, value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    path = path if path.is_absolute() else root / path
    absolute = path.absolute()
    if _path_has_symlink(absolute, Path(absolute.anchor)):
        return None
    return absolute.as_posix()


def _validate_tier_a_archive_indexes(
    root: Path, registrations: object, cutover_commit: str
) -> dict[str, object]:
    rows = registrations if isinstance(registrations, list) else []
    results: list[dict[str, object]] = []
    authority_allowlist: dict[tuple[str, str, str], int] = {}
    checkpoint_allowlist: dict[str, tuple[str, int]] = {}
    for registration in rows:
        errors: list[str] = []
        if not isinstance(registration, Mapping):
            results.append(
                {"validated": False, "errors": ["archive_index_row_invalid"]}
            )
            continue
        if not _is_sha(registration.get("sha256")):
            errors.append("archive_index_sha256_invalid")
        path = _safe_registered_path(root, registration.get("path"))
        if path is None or not path.is_file():
            errors.append("archive_index_path_invalid")
            payload = None
        else:
            observed_sha256 = _sha256_file(path)
            if observed_sha256 != str(registration.get("sha256", "")).lower():
                errors.append("archive_index_sha256_mismatch")
            errors.extend(
                _git_file_binding_errors(
                    root,
                    path,
                    registration,
                    cutover_commit=cutover_commit,
                )
            )
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                payload = loaded if isinstance(loaded, Mapping) else None
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = None
            if payload is None:
                errors.append("archive_index_json_invalid")
        local_authority: dict[tuple[str, str, str], int] = {}
        local_checkpoints: dict[str, tuple[str, int]] = {}
        if payload is not None:
            if payload.get("schema_version") != TIER_A_ARCHIVE_INDEX_SCHEMA:
                errors.append("archive_index_schema_invalid")
            if str(payload.get("status", "")).lower() not in SUCCESS_STATUSES:
                errors.append("archive_index_status_invalid")
            without_uid = dict(payload)
            artifact_uid = without_uid.pop("artifact_uid", None)
            if artifact_uid != canonical_sha256(without_uid):
                errors.append("archive_index_artifact_uid_mismatch")
            candidates = payload.get("authority_candidates")
            if not isinstance(candidates, list):
                errors.append("archive_index_authority_candidates_missing")
            else:
                authority_ids: list[str] = []
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        errors.append("archive_index_authority_candidate_invalid")
                        continue
                    authority_uid = str(candidate.get("authority_uid", ""))
                    normalized_path = _normalize_registered_candidate_path(
                        root, candidate.get("candidate_path")
                    )
                    digest = str(candidate.get("sha256", "")).lower()
                    try:
                        byte_size = int(candidate.get("bytes", -1))
                    except (TypeError, ValueError):
                        byte_size = -1
                    authority_ids.append(authority_uid)
                    if (
                        authority_uid not in REQUIRED_AUTHORITY_INPUTS
                        or normalized_path is None
                        or not _is_sha(digest)
                        or byte_size < 0
                    ):
                        errors.append("archive_index_authority_binding_invalid")
                        continue
                    key = (authority_uid, normalized_path, digest)
                    previous_bytes = local_authority.get(key)
                    if previous_bytes is not None and previous_bytes != byte_size:
                        errors.append("archive_index_authority_byte_count_conflict")
                    local_authority[key] = byte_size
                if (
                    set(authority_ids) != set(REQUIRED_AUTHORITY_INPUTS)
                    or len(authority_ids) != len(set(authority_ids))
                    or len(candidates) != len(REQUIRED_AUTHORITY_INPUTS)
                ):
                    errors.append("archive_index_authority_membership_invalid")
            checkpoints = payload.get("checkpoint_candidates")
            if not isinstance(checkpoints, list) or not checkpoints:
                errors.append("archive_index_checkpoint_candidates_missing")
            else:
                for candidate in checkpoints:
                    if not isinstance(candidate, Mapping):
                        errors.append("archive_index_checkpoint_candidate_invalid")
                        continue
                    normalized_path = _normalize_registered_candidate_path(
                        root, candidate.get("candidate_path")
                    )
                    digest = str(candidate.get("sha256", "")).lower()
                    try:
                        byte_size = int(candidate.get("bytes", -1))
                    except (TypeError, ValueError):
                        byte_size = -1
                    if normalized_path is None or not _is_sha(digest) or byte_size < 0:
                        errors.append("archive_index_checkpoint_binding_invalid")
                        continue
                    previous = local_checkpoints.get(normalized_path)
                    binding = (digest, byte_size)
                    if previous is not None and previous != binding:
                        errors.append("archive_index_checkpoint_hash_conflict")
                    local_checkpoints[normalized_path] = binding
        validated = not errors
        if validated:
            for key, byte_size in local_authority.items():
                authority_uid, candidate_path, digest = key
                conflicts = [
                    registered_key
                    for registered_key in authority_allowlist
                    if registered_key[:2] == (authority_uid, candidate_path)
                    and registered_key[2] != digest
                ]
                if conflicts:
                    errors.append("archive_index_cross_manifest_authority_conflict")
                elif authority_allowlist.get(key, byte_size) != byte_size:
                    errors.append(
                        "archive_index_cross_manifest_authority_byte_count_conflict"
                    )
            for candidate_path, binding in local_checkpoints.items():
                previous = checkpoint_allowlist.get(candidate_path)
                if previous is not None and previous != binding:
                    errors.append("archive_index_cross_manifest_checkpoint_conflict")
            validated = not errors
            if validated:
                authority_allowlist.update(local_authority)
                checkpoint_allowlist.update(local_checkpoints)
        results.append(
            {
                "path": str(registration.get("path", "")),
                "sha256": registration.get("sha256"),
                "validated": validated,
                "authority_binding_count": len(local_authority),
                "checkpoint_binding_count": len(local_checkpoints),
                "errors": sorted(set(errors)),
            }
        )
    return {
        "registered_count": len(rows),
        "validated_count": sum(row.get("validated") is True for row in results),
        "results": results,
        "authority_allowlist": authority_allowlist,
        "checkpoint_allowlist": checkpoint_allowlist,
    }


def _tier_a_candidate_records(
    root: Path,
    inventory: list[dict[str, str]],
    scheduler_report: Mapping[str, object],
    *,
    cutover_commit: str,
    archive_authority_allowlist: Mapping[tuple[str, str, str], int],
) -> list[dict[str, object]]:
    records: dict[tuple[str, str, str], dict[str, object]] = {}

    def add(authority_uid: str, locator: str, digest: str, source: str) -> None:
        if authority_uid not in REQUIRED_AUTHORITY_INPUTS or not _is_sha(digest):
            return
        if locator.startswith("git:"):
            _, storage_commit, _ = locator.split(":", 2)
            if not _commit_is_at_or_before_cutover(
                root, storage_commit, cutover_commit
            ):
                return
        else:
            normalized_path = _normalize_registered_candidate_path(root, locator)
            if normalized_path is None:
                return
        readback = _read_candidate_locator(root, locator, digest)
        if readback is None:
            return
        content, filesystem_path = readback
        if not locator.startswith("git:"):
            expected_bytes = archive_authority_allowlist.get(
                (authority_uid, normalized_path, digest)
            )
            if expected_bytes is None or len(content) != expected_bytes:
                return
        producer_identity = _authority_producer_identity(
            root,
            authority_uid,
            content,
            cutover_commit=cutover_commit,
        )
        if producer_identity is None:
            return
        commit, tree = producer_identity
        key = (authority_uid, digest, f"{commit}:{tree}")
        row = records.get(key)
        if row is None:
            records[key] = {
                "authority_uid": authority_uid,
                "sha256": digest,
                "content": content,
                "producer_commit": commit,
                "producer_tree": tree,
                "locators": [locator],
                "filesystem_paths": [filesystem_path]
                if filesystem_path is not None
                else [],
                "source_classes": [source],
            }
            return
        locators = row["locators"]
        sources = row["source_classes"]
        filesystem_paths = row["filesystem_paths"]
        if isinstance(locators, list) and locator not in locators:
            locators.append(locator)
        if isinstance(sources, list) and source not in sources:
            sources.append(source)
        if (
            filesystem_path is not None
            and isinstance(filesystem_paths, list)
            and filesystem_path not in filesystem_paths
        ):
            filesystem_paths.append(filesystem_path)

    for row in inventory:
        if (
            row.get("candidate_kind") != "authority_artifact"
            or row.get("schema_status") != "valid"
        ):
            continue
        try:
            locators = json.loads(row.get("all_candidate_paths_json", "[]"))
        except json.JSONDecodeError:
            locators = []
        if not isinstance(locators, list) or not locators:
            locators = [row.get("candidate_path", "")]
        for locator in locators:
            add(
                str(row.get("authority_uid", "")),
                str(locator),
                str(row.get("sha256", "")).lower(),
                str(row.get("source_class", "inventory")),
            )
    workdirs = scheduler_report.get("workdirs")
    if isinstance(workdirs, list):
        for workdir in workdirs:
            if not isinstance(workdir, Mapping):
                continue
            checks = workdir.get("exact_authority_path_checks")
            if not isinstance(checks, list):
                continue
            for check in checks:
                if not isinstance(check, Mapping) or check.get("exists") is not True:
                    continue
                add(
                    str(check.get("authority_uid", "")),
                    str(check.get("candidate_path", "")),
                    str(check.get("sha256", "")).lower(),
                    "scheduler_supplemental_exact_check",
                )
    return list(records.values())


def _validate_tier_a_candidate_groups(
    root: Path,
    records: list[dict[str, object]],
    *,
    cutover_commit: str,
    archive_checkpoint_allowlist: Mapping[str, tuple[str, int]],
) -> list[dict[str, object]]:
    by_identity: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_identity[(str(row["producer_commit"]), str(row["producer_tree"]))].append(
            row
        )
    results: list[dict[str, object]] = []
    for (commit, tree), rows in sorted(by_identity.items()):
        per_authority: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            per_authority[str(row["authority_uid"])].append(row)
        ambiguity = {
            authority_uid: sorted({str(row["sha256"]) for row in authority_rows})
            for authority_uid, authority_rows in per_authority.items()
            if len({str(row["sha256"]) for row in authority_rows}) != 1
        }
        selected = {
            authority_uid: authority_rows[0]
            for authority_uid, authority_rows in per_authority.items()
            if len({str(row["sha256"]) for row in authority_rows}) == 1
        }
        contents = {
            authority_uid: row["content"]
            for authority_uid, row in selected.items()
            if isinstance(row.get("content"), bytes)
        }
        bundle_roots: set[Path] = set()
        for authority_uid, row in selected.items():
            for path in row.get("filesystem_paths", []):
                if isinstance(path, Path):
                    bundle_root = _candidate_bundle_root(
                        path, REQUIRED_AUTHORITY_INPUTS[authority_uid]
                    )
                    if bundle_root is not None:
                        bundle_roots.add(bundle_root)

        def load_checkpoint(relative_path: str) -> bytes | None:
            git_candidate = (
                _git_blob(root, commit, relative_path)
                if _commit_is_at_or_before_cutover(root, commit, cutover_commit)
                else None
            )
            if git_candidate is not None:
                return git_candidate[1]
            path_value = Path(relative_path)
            if path_value.is_absolute() or ".." in path_value.parts:
                return None
            for base in sorted(bundle_roots, key=lambda value: value.as_posix()):
                candidate = (base / path_value).absolute()
                if (
                    not _path_has_symlink(candidate, Path(candidate.anchor))
                    and candidate.is_file()
                ):
                    normalized = candidate.as_posix()
                    expected = archive_checkpoint_allowlist.get(normalized)
                    if expected is None:
                        continue
                    content = candidate.read_bytes()
                    expected_sha256, expected_bytes = expected
                    if (
                        len(content) == expected_bytes
                        and _sha256_bytes(content) == expected_sha256
                    ):
                        return content
            return None

        source_bindings = {
            authority_uid: {
                "sha256": row["sha256"],
                "locators": sorted(str(value) for value in row["locators"]),
                "source_classes": sorted(str(value) for value in row["source_classes"]),
            }
            for authority_uid, row in selected.items()
        }
        result = _validate_tier_a_contents(
            root,
            commit,
            contents,
            load_checkpoint,
            source_bindings,
            cutover_commit=cutover_commit,
        )
        result["lineage_source"] = "producer_identity_candidate_ledger"
        result["producer_commit"] = commit
        result["producer_tree"] = tree
        if ambiguity:
            result["validated"] = False
            result["identity_uid"] = None
            result["errors"] = sorted(
                set(result["errors"])
                | {f"ambiguous_authority_bytes:{uid}" for uid in ambiguity}
            )
        results.append(result)
    return results


def _resolve_tier_a(
    root: Path,
    inventory: list[dict[str, str]],
    scheduler_report: Mapping[str, object],
    *,
    cutover_commit: str,
    archive_index_registrations: object,
) -> dict[str, object]:
    archive_indexes = _validate_tier_a_archive_indexes(
        root, archive_index_registrations, cutover_commit
    )
    commit_sets = _candidate_commit_sets(inventory)
    discovered_common_commits = (
        sorted(set.intersection(*commit_sets.values()))
        if commit_sets and all(commit_sets.values())
        else []
    )
    common_commits = [
        commit
        for commit in discovered_common_commits
        if _commit_is_at_or_before_cutover(root, commit, cutover_commit)
    ]
    candidate_lineages = [
        _validate_tier_a_commit(root, commit, cutover_commit=cutover_commit)
        for commit in common_commits
    ]
    candidate_records = _tier_a_candidate_records(
        root,
        inventory,
        scheduler_report,
        cutover_commit=cutover_commit,
        archive_authority_allowlist=archive_indexes["authority_allowlist"],
    )
    generic_lineages = _validate_tier_a_candidate_groups(
        root,
        candidate_records,
        cutover_commit=cutover_commit,
        archive_checkpoint_allowlist=archive_indexes["checkpoint_allowlist"],
    )
    known_lineage_keys = {
        (row.get("identity_uid"), row.get("lineage_source"))
        for row in candidate_lineages
    }
    candidate_lineages.extend(
        row
        for row in generic_lineages
        if (row.get("identity_uid"), row.get("lineage_source"))
        not in known_lineage_keys
    )
    valid_lineages = [row for row in candidate_lineages if row["validated"] is True]
    identity_uids = {
        str(row["identity_uid"]) for row in valid_lineages if row.get("identity_uid")
    }
    scheduler_candidate_count = int(
        scheduler_report.get("supplemental_authority_candidate_count", 0)
    )
    inventory_candidate_count = sum(
        1 for row in inventory if row.get("candidate_kind") == "authority_artifact"
    )
    forensic_scope_complete = (
        scheduler_report.get("scheduler_reconciliation_complete") is True
    )
    ready = len(identity_uids) == 1 and bool(valid_lineages)
    failure_proven = bool(
        forensic_scope_complete
        and not ready
        and not discovered_common_commits
        and inventory_candidate_count == 0
        and scheduler_candidate_count == 0
        and archive_indexes["registered_count"] == archive_indexes["validated_count"]
    )
    archive_index_report = {
        key: value
        for key, value in archive_indexes.items()
        if key not in {"authority_allowlist", "checkpoint_allowlist"}
    }
    return {
        "failed": failure_proven,
        "ready_for_byte_identical_recovery": ready,
        "coherent_identity_count": len(identity_uids),
        "common_storage_commit_count": len(common_commits),
        "common_storage_commits": common_commits,
        "rejected_post_cutover_common_storage_commits": sorted(
            set(discovered_common_commits) - set(common_commits)
        ),
        "candidate_lineages": candidate_lineages,
        "admissible_candidate_record_count": len(candidate_records),
        "archive_index_validation": archive_index_report,
        "inventory_authority_candidate_count": inventory_candidate_count,
        "scheduler_supplemental_candidate_count": scheduler_candidate_count,
        "forensic_scope_complete": forensic_scope_complete,
        "historical_authority_artifacts_rewritten": False,
    }


def _round20q_component_contract_errors(
    root: Path, binding: Mapping[str, object]
) -> list[str]:
    component_id = str(binding.get("component_id", ""))
    expected = ROUND20Q_COMPONENT_CONTRACTS.get(component_id)
    if expected is None:
        return ["round20q_component_contract_unknown"]
    if binding.get("content_type") != "json":
        return ["round20q_component_content_type_invalid"]
    path = _safe_registered_path(root, binding.get("path"))
    if path is None or not path.is_file():
        return ["round20q_component_contract_path_invalid"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ["round20q_component_contract_json_invalid"]
    if not isinstance(payload, Mapping):
        return ["round20q_component_contract_root_invalid"]
    schema_version, identity_field = expected
    errors: list[str] = []
    if payload.get("schema_version") != schema_version:
        errors.append("round20q_component_schema_mismatch")
    if payload.get("round_id") != "neuronsifter_round20q":
        errors.append("round20q_component_round_identity_mismatch")
    if payload.get("component_id") != component_id:
        errors.append("round20q_component_self_identity_mismatch")
    if payload.get("frozen") is not True:
        errors.append("round20q_component_not_frozen")
    if str(payload.get("status", "")).strip().lower() not in SUCCESS_STATUSES:
        errors.append("round20q_component_status_invalid")
    if not _is_sha(payload.get(identity_field)):
        errors.append(f"round20q_component_{identity_field}_invalid")
    for field in ROUND20Q_COMPONENT_BOOLEAN_FIELDS[component_id]:
        if payload.get(field) is not True:
            errors.append(f"round20q_component_{field}_not_true")
    if component_id == "seed_protocol":
        seeds = payload.get("seeds")
        if (
            not isinstance(seeds, list)
            or len(seeds) != 5
            or len(set(seeds)) != 5
            or any(not isinstance(seed, int) for seed in seeds)
        ):
            errors.append("round20q_five_seed_protocol_invalid")
    return errors


def _validate_round20q_contract(
    root: Path, registration: object, *, cutover_commit: str
) -> dict[str, object]:
    if not isinstance(registration, Mapping):
        return {
            "registered": False,
            "fully_validated": False,
            "manifest_path": None,
            "manifest_sha256": None,
            "components": {component: False for component in ROUND20Q_COMPONENTS},
            "component_bindings": [],
            "errors": ["round20q_contract_not_registered"],
        }
    errors: list[str] = []
    manifest_path = _safe_registered_path(root, registration.get("path"))
    if manifest_path is None:
        errors.append("round20q_manifest_path_outside_repository")
    elif not manifest_path.is_file() or manifest_path.is_symlink():
        errors.append("round20q_manifest_missing_or_symlink")
    if not _is_sha(registration.get("sha256")):
        errors.append("round20q_manifest_registration_sha256_invalid")
    if manifest_path is not None and manifest_path.is_file():
        registration_errors = _git_file_binding_errors(
            root,
            manifest_path,
            registration,
            cutover_commit=cutover_commit,
        )
        errors.extend(f"round20q_manifest_{value}" for value in registration_errors)
    payload: Mapping[str, object] | None = None
    if (
        manifest_path is not None
        and manifest_path.is_file()
        and not manifest_path.is_symlink()
    ):
        content = manifest_path.read_bytes()
        if _sha256_bytes(content) != str(registration.get("sha256", "")).lower():
            errors.append("round20q_manifest_registration_sha256_mismatch")
        try:
            loaded = json.loads(content.decode("utf-8"))
            payload = loaded if isinstance(loaded, Mapping) else None
        except (UnicodeError, json.JSONDecodeError):
            payload = None
        if payload is None:
            errors.append("round20q_manifest_json_invalid")
    component_state = {component: False for component in ROUND20Q_COMPONENTS}
    component_results: list[dict[str, object]] = []
    component_git_identities: set[tuple[str, str]] = set()
    if payload is not None:
        if payload.get("schema_version") != ROUND20Q_SCHEMA:
            errors.append("round20q_manifest_schema_invalid")
        if payload.get("round_id") != "neuronsifter_round20q":
            errors.append("round20q_round_identity_invalid")
        if payload.get("frozen") is not True:
            errors.append("round20q_contract_not_frozen")
        if str(payload.get("status", "")).strip().lower() not in SUCCESS_STATUSES:
            errors.append("round20q_manifest_status_invalid")
        without_uid = dict(payload)
        artifact_uid = without_uid.pop("artifact_uid", None)
        if artifact_uid != canonical_sha256(without_uid):
            errors.append("round20q_manifest_artifact_uid_mismatch")
        producer_commit, producer_tree, dirty = _git_identity_values(payload)
        if (
            not _producer_uid(payload)
            or not producer_commit
            or not producer_tree
            or dirty is not False
        ):
            errors.append("round20q_manifest_producer_identity_invalid")
        elif _git_optional_lines(
            root, "rev-parse", "--verify", f"{producer_commit}^{{commit}}"
        ) != [producer_commit] or _git_optional_lines(
            root, "show", "-s", "--format=%T", producer_commit
        ) != [producer_tree]:
            errors.append("round20q_manifest_git_identity_mismatch")
        elif not _commit_is_at_or_before_cutover(root, producer_commit, cutover_commit):
            errors.append("round20q_manifest_producer_after_recovery_cutover")
        components = payload.get("components")
        if not isinstance(components, list):
            errors.append("round20q_components_missing")
        else:
            component_ids = [
                str(row.get("component_id", ""))
                for row in components
                if isinstance(row, Mapping)
            ]
            if (
                set(component_ids) != set(ROUND20Q_COMPONENTS)
                or len(component_ids) != len(set(component_ids))
                or len(components) != len(ROUND20Q_COMPONENTS)
            ):
                errors.append("round20q_component_identity_mismatch")
            for component in components:
                if not isinstance(component, Mapping):
                    errors.append("round20q_component_row_invalid")
                    continue
                result, component_errors = _verify_component_binding(
                    root,
                    component,
                    cutover_commit=cutover_commit,
                )
                component_errors.extend(
                    _round20q_component_contract_errors(root, component)
                )
                result["validated"] = not component_errors
                result["errors"] = sorted(set(component_errors))
                component_results.append(result)
                component_id = str(component.get("component_id", ""))
                if component_id in component_state:
                    component_state[component_id] = not component_errors
                commit = str(component.get("git_commit", "")).strip()
                tree = str(component.get("git_tree", "")).strip()
                if commit and tree:
                    component_git_identities.add((commit, tree))
                errors.extend(f"{component_id}:{value}" for value in component_errors)
        if len(component_git_identities) != 1:
            errors.append("round20q_component_git_lineage_inconsistent")
        elif (producer_commit, producer_tree) not in component_git_identities:
            errors.append("round20q_manifest_component_git_lineage_mismatch")
    return {
        "registered": True,
        "fully_validated": not errors and all(component_state.values()),
        "manifest_path": str(registration.get("path", "")),
        "manifest_sha256": registration.get("sha256"),
        "manifest_storage_git_commit": registration.get("git_commit"),
        "manifest_storage_git_tree": registration.get("git_tree"),
        "components": component_state,
        "component_bindings": component_results,
        "errors": sorted(set(errors)),
    }


def _scan_round20q_machine_evidence(root: Path) -> dict[str, object]:
    search_roots = (root / "Libs/configs", root / "Output/audits")
    hits: list[dict[str, object]] = []
    visited_entries = 0
    bytes_read = 0
    for base in search_roots:
        if not base.is_dir():
            continue
        stack = [base]
        while stack:
            directory = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited_entries += 1
                    if visited_entries > TRANSITION_ENTRY_CAP:
                        raise RuntimeError(
                            "Round-20Q machine-evidence entry cap exceeded"
                        )
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    path = Path(entry.path)
                    if (
                        not entry.is_file(follow_symlinks=False)
                        or path.suffix.lower() not in MACHINE_SUFFIXES
                    ):
                        continue
                    if not ROUND20Q_PATTERN.search(path.name):
                        continue
                    content = path.read_bytes()
                    bytes_read += len(content)
                    hits.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )
    return {
        "search_roots": [path.relative_to(root).as_posix() for path in search_roots],
        "search_strategy": "round_uid_exact_filename_then_hash",
        "content_scan_performed": False,
        "bounded": True,
        "entry_cap": TRANSITION_ENTRY_CAP,
        "candidate_filename_count": len(hits),
        "bytes_read": bytes_read,
        "machine_readable_hit_count": len(hits),
        "machine_readable_hits": sorted(hits, key=lambda row: str(row["path"])),
    }


def build(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    write_output: bool = True,
    enforce_compute_node: bool = True,
) -> dict[str, object]:
    hostname = socket.gethostname().split(".", 1)[0]
    if enforce_compute_node and hostname not in ALLOWED_COMPUTE_NODES:
        raise RuntimeError("Round-22A transition assessment must run on cpu03 or cpu04")
    contract = load_contract(config_path, schema_path)
    wp1_validation = validate_wp1(
        config_path,
        schema_path,
        root,
        recompute=False,
        verify_candidate_bytes=False,
        enforce_compute_node=enforce_compute_node,
    )
    if wp1_validation.get("passed") is not True:
        raise RuntimeError("Round-22A WP1 inventory is not valid")
    scheduler_validation = validate_scheduler_reconciliation(
        config_path,
        schema_path,
        root,
        recompute=False,
        enforce_compute_node=enforce_compute_node,
    )
    if scheduler_validation.get("passed") is not True:
        raise RuntimeError("Round-22A scheduler WorkDir reconciliation is not valid")
    search = _read_json(root / WP1_OUTPUTS["search"])
    inventory = _read_inventory(root / WP1_OUTPUTS["inventory"])
    recovery = contract["freeze"]["authority_recovery"]
    cutover = _validate_recovery_cutover(root, recovery)
    scheduler_path = root / str(recovery["scheduler_reconciliation"]["output"])
    scheduler_report = _read_json(scheduler_path)
    tier_a = _resolve_tier_a(
        root,
        inventory,
        scheduler_report,
        cutover_commit=cutover["commit"],
        archive_index_registrations=recovery.get("tier_a_archive_indexes", []),
    )
    per_authority = _per_authority_tier_evidence(
        root,
        inventory,
        recovery.get("tier_b_reconstruction_manifests", []),
        cutover_commit=cutover["commit"],
    )
    builder_evidence = _historical_builder_evidence(root, cutover["commit"])
    round20q_machine = _scan_round20q_machine_evidence(root)
    round20q_contract = _validate_round20q_contract(
        root,
        recovery.get("round20q_replay_prerequisite"),
        cutover_commit=cutover["commit"],
    )
    tier_b_complete = all(
        row["tier_b_reconstruction_ready"] is True for row in per_authority
    )
    tier_b_failed = not tier_b_complete
    tier_c_eligible = bool(
        tier_a["failed"] is True
        and tier_b_failed
        and round20q_contract["fully_validated"] is True
    )
    if tier_a["ready_for_byte_identical_recovery"] is True:
        status = "tier_a_coherent_lineage_ready_not_recovered"
        next_action = "recover_validated_byte_identical_tier_a_lineage"
    elif tier_a["failed"] is not True:
        status = "tier_a_candidates_require_adjudication"
        next_action = "adjudicate_unresolved_tier_a_candidates"
    elif not tier_b_failed:
        status = "tier_b_reconstruction_inputs_complete_not_executed"
        next_action = "execute_validated_deterministic_tier_b_reconstruction"
    elif tier_c_eligible:
        status = "tier_c_eligible_not_started"
        next_action = "execute_round21r_replay_from_validated_round20q_contract"
    else:
        status = "blocked_round20q_replay_prerequisite"
        next_action = "register_and_validate_complete_round20q_frozen_contract_before_round21r_replay"
    predecessor_path = root / PREDECESSOR_OUTPUT
    predecessor_binding = {
        "path": PREDECESSOR_OUTPUT,
        "exists": predecessor_path.is_file(),
        "sha256": hashlib.sha256(predecessor_path.read_bytes()).hexdigest()
        if predecessor_path.is_file()
        else None,
    }
    report: dict[str, object] = {
        "schema_version": TRANSITION_SCHEMA,
        "round_id": ROUND22A_ID,
        "status": status,
        "authority_generation": "unresolved",
        "unique_lineage": False,
        "recovery_cutover_git_identity": cutover,
        "predecessor_lineage": predecessor_binding,
        "implementation_bindings": [
            _implementation_binding(
                REPO_ROOT,
                "Libs/scripts/builders/build_round22a_authority_transition.py",
            ),
            _implementation_binding(
                REPO_ROOT,
                "Libs/scripts/validators/validate_round22a_authority_transition.py",
            ),
            _implementation_binding(
                REPO_ROOT,
                "Libs/tests/test_round22a_authority_recovery.py",
            ),
        ],
        "wp1_search_artifact": {
            "path": WP1_OUTPUTS["search"],
            "sha256": hashlib.sha256(
                (root / WP1_OUTPUTS["search"]).read_bytes()
            ).hexdigest(),
            "registered_search_roots_exhausted": search[
                "registered_search_roots_exhausted"
            ],
        },
        "wp1_inventory_artifact": {
            "path": WP1_OUTPUTS["inventory"],
            "sha256": hashlib.sha256(
                (root / WP1_OUTPUTS["inventory"]).read_bytes()
            ).hexdigest(),
            "row_count": len(inventory),
        },
        "scheduler_workdir_reconciliation_artifact": {
            "path": recovery["scheduler_reconciliation"]["output"],
            "sha256": hashlib.sha256(scheduler_path.read_bytes()).hexdigest(),
            "artifact_uid": scheduler_report.get("artifact_uid"),
            "accounting_record_count": scheduler_report.get("accounting_record_count"),
            "workdir_count": scheduler_report.get("workdir_count"),
            "complete": scheduler_report.get("scheduler_reconciliation_complete"),
        },
        "tier_a": tier_a,
        "tier_b": {
            "failed": tier_b_failed,
            "all_targets_execution_ready": tier_b_complete,
            "deterministic_reconstruction_started": False,
            "historical_byte_identity_claimed": False,
            "registered_manifest_count": len(
                recovery.get("tier_b_reconstruction_manifests", [])
            ),
            "historical_builder_evidence": builder_evidence,
            "per_authority": per_authority,
        },
        "tier_c": {
            "eligible": tier_c_eligible,
            "round20q_contract_registered": round20q_contract["registered"],
            "round20q_machine_evidence": round20q_machine,
            "round20q_required_components": round20q_contract["components"],
            "round20q_contract_validation": round20q_contract,
            "round20q_fully_validated": round20q_contract["fully_validated"],
            "round21r_replay_started": False,
            "gpu_used": False,
        },
        "round21_authority": {
            "required": 9,
            "present": 0,
            "validated": 0,
            "hash_bound": 0,
        },
        "static_twin": {
            "required_components": 10,
            "present": 0,
            "validated": 0,
            "frozen": False,
        },
        "exact_clean_full_suite_authorized": False,
        "c3_c4_execution_allowed": False,
        "coevolution_execution_allowed": False,
        "c2_b_status": C2B_BLOCKED,
        "canonical_round21_authority_written": False,
        "round22a_authority_root_written": False,
        "execution_node": hostname,
        "next_action": next_action,
    }
    report["artifact_uid"] = canonical_sha256(report)
    if write_output:
        _write_immutable_json(root / OUTPUT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args()
    report = build(args.config, args.schema, Path(args.root))
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
