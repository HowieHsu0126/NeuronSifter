"""Build the Round-22 fail-closed readiness and evidence-bound audit surface.

This builder intentionally does not execute tools, train a Twin, adapt an
Agent, or synthesize measurements.  It records what is present, preserves the
existing active-learning manifests as non-formal inputs, freezes the requested
cost definitions, and emits explicit ``NOT_AVAILABLE`` evidence when the
Round-21 static reference or real tool outcomes are absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG = "Libs/configs/experiments/round22_agent_twin_coevolution.yaml"
DEFAULT_SCHEMA = "Libs/configs/schema/round22_agent_twin_coevolution.schema.json"
FORMAL_SOURCE_ROLE = "round22_formal_tool_outcome"
BASELINES = (
    "random",
    "fixed_cascade",
    "uncertainty_bald",
    "ps_bax",
    "mf_mes",
    "cost_aware_bo",
    "neuronsifter_static",
)
ORACLE = "full_information_oracle"
C4_VARIANTS = ("raw_score_no_tri", "uncertainty_only", "calibration_only", "full_tri")
MATRIX = (
    "static_reference",
    "agent_only",
    "twin_only",
    "full_coevolution",
    "naive_latest_batch_joint_update",
    "full_replay_joint_update",
)
TOOL_OUTCOME_FIELDS = (
    "state",
    "structured_action",
    "tool",
    "fidelity",
    "result",
    "uncertainty",
    "actual_runtime",
    "compute_cost",
    "wall_clock_cost",
    "failure_status",
    "tool_version",
    "provenance",
    "hidden_evaluation_role",
)
FORMAL_METADATA_FIELDS = (
    "state_uid",
    "structured_action_uid",
    "tool_version",
    "fidelity",
    "compute_cost",
    "wall_clock_cost",
    "hidden_evaluation_role",
    "result_schema_version",
    "uncertainty_schema_version",
    "result_hash_verified",
    "uncertainty_hash_verified",
    "result_observed_hash",
    "uncertainty_observed_hash",
    "hash_readback_uid",
    "formal_round22_execution",
)

# Each Round-21 freeze component has one authoritative source artifact.  Do
# not search every JSON payload for a convenient hash: copied or stale values
# in another audit must never satisfy a missing authority binding.
FREEZE_COMPONENT_SPECS = {
    "model_config_hash": ("round21_training_input_freeze", ("model_config_hash", "model_configuration_hash")),
    "c1_data_hash": ("round21_training_input_freeze", ("c1_data_hash", "c1_manifest_hash", "c1_input_hash")),
    "c2_data_hash": ("round21_training_input_freeze", ("c2_data_hash", "c2s_data_hash", "c2_manifest_hash", "c2s_manifest_hash")),
    "training_split_hash": ("round21_training_input_freeze", ("training_split_hash", "train_split_hash")),
    "validation_split_hash": ("round21_training_input_freeze", ("validation_split_hash", "val_split_hash")),
    "holdout_split_hash": ("round21_training_input_freeze", ("holdout_split_hash", "final_holdout_split_hash")),
    "environment_hash": ("round21_training_input_freeze", ("environment_hash", "runtime_environment_hash")),
    "tri_input_schema_hash": ("round21_training_input_freeze", ("tri_input_schema_hash", "tri_schema_hash")),
    "baseline_frontier_hash": ("round21_static_twin_evidence", ("baseline_frontier_hash", "round21_baseline_frontier_hash", "baseline_frontier")),
}


def repo_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def relative_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_tsv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def load_contract(config_path: str | Path = DEFAULT_CONFIG, schema_path: str | Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    config_file = repo_path(config_path)
    schema_file = repo_path(schema_path)
    contract = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(schema, dict):
        raise ValueError("Round-22 config and schema must be mappings.")
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:10])
        raise ValueError(f"Round-22 schema validation failed: {detail}")
    if tuple(contract["c3_static"]["baselines"]) != BASELINES:
        raise ValueError("Round-22 C3 baseline order or membership drifted.")
    if tuple(contract["c3_static"]["quota"]) != ("exploitation", "exploration", "sentinel"):
        raise ValueError("Round-22 C3 quota keys drifted.")
    if tuple(contract["tool_outcomes"]["formal_metadata_fields"]) != FORMAL_METADATA_FIELDS:
        raise ValueError("Round-22 formal tool metadata contract drifted.")
    return contract


def evidence_binding(path_value: str | Path, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    path = repo_path(path_value, repo_root)
    exists = path.is_file()
    return {
        "path": relative_path(path, repo_root),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "sha256": sha256_file(path) if exists else None,
    }


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "evidence_artifact_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(payload, dict):
        return None, "json_root_is_not_object"
    return payload, ""


def _walk_mappings(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _first_value(payloads: Iterable[Mapping[str, object]], keys: Iterable[str]) -> object | None:
    key_set = set(keys)
    for payload in payloads:
        for mapping in _walk_mappings(payload):
            for key in key_set:
                value = mapping.get(key)
                if value not in (None, "", [], {}):
                    return value
    return None


def _bound_hash(payload: Mapping[str, object] | None, keys: Iterable[str]) -> tuple[str | None, list[str]]:
    """Read a hash only from its declared authority and reject conflicts."""
    if not isinstance(payload, Mapping):
        return None, []
    values: list[str] = []
    key_names = tuple(keys)
    # Round-21 artifacts normally expose the fields at the root.  The two
    # explicit containers support the established freeze/hash sub-records
    # without reopening an unrestricted recursive search.
    containers: list[Mapping[str, object]] = [payload]
    for container_key in ("freeze_components", "hashes"):
        child = payload.get(container_key)
        if isinstance(child, Mapping):
            containers.append(child)
    for container in containers:
        for key in key_names:
            value = _hash_value(container.get(key))
            if value is not None:
                values.append(value)
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0], []
    if len(unique) > 1:
        return None, unique
    return None, []


def _hash_value(value: object | None) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value):
        return value.lower()
    if isinstance(value, (dict, list)):
        return canonical_sha256(value)
    return None


def _metric_value(candidate: Mapping[str, object]) -> float | None:
    for key in ("validation_metric", "validation_loss", "val_loss", "decision_risk", "metric_value"):
        value = candidate.get(key)
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    validation = candidate.get("validation_metrics")
    if isinstance(validation, Mapping):
        for key in ("decision_risk", "loss", "validation_loss", "metric_value"):
            try:
                parsed = float(validation[key])  # type: ignore[index]
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                return parsed
    return None


def select_validation_checkpoint(payload: Mapping[str, object]) -> dict[str, object] | None:
    """Select a validation-role checkpoint without ever using test evidence."""
    candidates: list[dict[str, object]] = []
    for item in _walk_mappings(payload):
        checkpoint_id = item.get("checkpoint_id", item.get("checkpoint_uid", item.get("id")))
        checkpoint_path = item.get("checkpoint_path", item.get("checkpoint_uri", item.get("path")))
        if not checkpoint_id or not checkpoint_path:
            continue
        role = str(item.get("split", item.get("selection_split", item.get("role", "")))).lower()
        if any(token in role for token in ("test", "holdout", "final")):
            continue
        if not role or ("val" not in role and "valid" not in role):
            continue
        metric = _metric_value(item)
        if metric is None:
            continue
        candidates.append(
            {
                "checkpoint_id": str(checkpoint_id),
                "checkpoint_path": str(checkpoint_path),
                "selection_split": role or "validation",
                "validation_metric": metric,
                "checkpoint_sha256": item.get("checkpoint_sha256", item.get("sha256")),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["validation_metric"]))


def build_static_twin_freeze(
    contract: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    bindings: list[dict[str, object]] = []
    parse_errors: list[dict[str, str]] = []
    checkpoint_catalog_payload: dict[str, Any] | None = None
    json_payloads: dict[str, dict[str, Any]] = {}
    for item in contract["freeze"]["required_inputs"]:
        path_value = str(item["path"])
        binding = {
            "input_id": str(item["input_id"]),
            **evidence_binding(path_value, repo_root),
        }
        bindings.append(binding)
        if binding["exists"] and path_value.endswith(".json"):
            payload, error = load_json_object(repo_path(path_value, repo_root))
            if error:
                parse_errors.append({"input_id": str(item["input_id"]), "error": error})
            elif str(item["input_id"]) == "round21_checkpoint_catalog":
                checkpoint_catalog_payload = payload
            if payload is not None:
                json_payloads[str(item["input_id"])] = payload
    selected = select_validation_checkpoint(checkpoint_catalog_payload or {})
    checkpoint_hash = _hash_value(selected.get("checkpoint_sha256") if selected else None)
    checkpoint_path = repo_path(str(selected["checkpoint_path"]), repo_root) if selected else None
    observed_checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
    checkpoint_hash_matches = bool(checkpoint_hash and observed_checkpoint_hash == checkpoint_hash)
    freeze_components: dict[str, str | None] = {
        "checkpoint_hash": checkpoint_hash if checkpoint_hash_matches else None,
    }
    component_sources: dict[str, str] = {"checkpoint_hash": "round21_checkpoint_catalog"}
    component_conflicts: dict[str, list[str]] = {}
    for component, (source_id, keys) in FREEZE_COMPONENT_SPECS.items():
        value, conflicts = _bound_hash(json_payloads.get(source_id), keys)
        freeze_components[component] = value
        component_sources[component] = source_id
        if conflicts:
            component_conflicts[component] = conflicts
    missing_freeze_components = [key for key, value in freeze_components.items() if value is None]
    missing = [str(row["input_id"]) for row in bindings if not bool(row["exists"])]
    frozen = not missing and not parse_errors and not missing_freeze_components and not component_conflicts
    return {
        "schema_version": "neuronsifter-round22-static-twin-freeze-v1",
        "round_id": "neuronsifter_round22",
        "status": "passed" if frozen else "blocked_round21_freeze_missing",
        "frozen": frozen,
        "immutable_during_round22": True,
        "selection_rule": "best_validation_checkpoint",
        "forbidden_selection_roles": ["test", "final_holdout"],
        "required_input_bindings": bindings,
        "parse_errors": parse_errors,
        "missing_input_ids": missing,
        "missing_freeze_components": missing_freeze_components,
        "freeze_component_sources": component_sources,
        "freeze_component_conflicts": component_conflicts,
        "selected_validation_checkpoint": selected,
        "checkpoint_artifact_binding": {
            "path": relative_path(checkpoint_path, repo_root) if checkpoint_path else None,
            "exists": bool(checkpoint_path and checkpoint_path.is_file()),
            "catalog_sha256": checkpoint_hash,
            "observed_sha256": observed_checkpoint_hash,
            "hash_matches": checkpoint_hash_matches,
        },
        "twin_checkpoint_hash": freeze_components["checkpoint_hash"],
        "model_config_hash": freeze_components["model_config_hash"],
        "data_hashes": {"c1": freeze_components["c1_data_hash"], "c2": freeze_components["c2_data_hash"]},
        "split_hashes": {
            "training": freeze_components["training_split_hash"],
            "validation": freeze_components["validation_split_hash"],
            "holdout": freeze_components["holdout_split_hash"],
        },
        "environment_hash": freeze_components["environment_hash"],
        "baseline_frontier_hash": freeze_components["baseline_frontier_hash"],
        "tri_input_schema_hash": freeze_components["tri_input_schema_hash"],
        "reference_mutation_policy": "reject_any_change_after_freeze",
        "next_action": "materialize_and_validate_all_round21_frozen_inputs" if not frozen else "freeze_static_agent_after_c3_and_c4",
    }


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    if not path.is_file():
        return [], [], "artifact_missing"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = list(reader.fieldnames or [])
            return fields, list(reader), ""
    except (OSError, UnicodeError) as exc:
        return [], [], f"read_error:{exc}"


def _finite_or_none(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_finite(value: object) -> float | None:
    parsed = _finite_or_none(value)
    return parsed if parsed is not None and parsed >= 0.0 else None


def _truthy_flag(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _row_text(row: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            if text and text.lower() not in {"none", "null", "not_available", "nan"}:
                return text
    return ""


def _hash_verification_flag(row: Mapping[str, object], *keys: str) -> bool:
    return any(_truthy_flag(row.get(key)) for key in keys)


def _sha256_text(value: object) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    return None


def build_tool_outcomes(
    contract: Mapping[str, Any],
    freeze: Mapping[str, object],
    *,
    allow_formal_hash_readback: bool | None = None,
) -> dict[str, object]:
    matrix_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    source_bindings: list[dict[str, object]] = []
    by_tool: dict[str, list[dict[str, object]]] = defaultdict(list)
    artifact_hash_cache: dict[str, str] = {}
    if allow_formal_hash_readback is None:
        allow_formal_hash_readback = socket.gethostname() != "mgmt2"
    for source in contract["tool_outcomes"]["source_manifests"]:
        source_path = repo_path(str(source["path"]))
        source_role = str(source["role"])
        source_binding = {"role": source_role, **evidence_binding(str(source["path"]))}
        source_bindings.append(source_binding)
        fields, rows, error = _read_tsv(source_path)
        if error:
            failure_rows.append(
                {
                    "record_uid": f"missing:{relative_path(source_path)}",
                    "source_manifest": relative_path(source_path),
                    "tool_uid": "unknown",
                    "failure_status": "missing_source_manifest",
                    "failure_kind": error,
                    "observed": True,
                    "formal_eligible": False,
                    "note": "Registered active-learning source could not be read.",
                }
            )
            continue
        for row in rows:
            result_uid = str(row.get("result_uid", ""))
            tool_uid = str(row.get("simulator_family", "unknown"))
            state_uid = _row_text(row, "state_uid", "candidate_uid")
            action_uid = _row_text(row, "structured_action_uid", "action_uid", "request_uid")
            runtime = _nonnegative_finite(row.get("runtime_seconds"))
            failure_status = str(row.get("failure_status", "")).strip() or "missing_failure_status"
            reasons: list[str] = []
            if not state_uid:
                reasons.append("state_uid_missing")
            if not action_uid:
                reasons.append("structured_action_uid_missing")
            if runtime is None:
                reasons.append("actual_runtime_unavailable_negative_or_nonfinite")
            formal_round22_execution = _truthy_flag(row.get("formal_round22_execution"))
            formal_round_id = _row_text(row, "round22_round_id", "formal_round_id")
            formal_hash_readback_allowed = allow_formal_hash_readback
            if source_role != FORMAL_SOURCE_ROLE:
                reasons.append("source_role_not_formal_round22")
            if not formal_round22_execution:
                reasons.append("formal_round22_execution_marker_missing")
            if formal_round_id != str(contract["round_id"]):
                reasons.append("formal_round22_round_id_missing_or_mismatched")
            if source_role == FORMAL_SOURCE_ROLE and formal_round22_execution and not formal_hash_readback_allowed:
                reasons.append("formal_artifact_hash_readback_requires_compute_node")
            result_path = _row_text(row, "trajectory_artifact_path", "result_artifact_path")
            uncertainty_path = _row_text(row, "teacher_uncertainty_path", "uncertainty_artifact_path")
            result_schema_version = _row_text(row, "result_schema_version", "trajectory_schema_version")
            uncertainty_schema_version = _row_text(
                row,
                "uncertainty_schema_version",
                "teacher_uncertainty_schema_version",
            )
            if not result_path:
                reasons.append("result_artifact_missing")
            elif not repo_path(result_path).is_file():
                reasons.append("result_artifact_unreadable")
            if not result_schema_version:
                reasons.append("result_schema_version_missing")
            result_hash = _sha256_text(_row_text(row, "trajectory_hash", "result_hash"))
            observed_result_hash = _sha256_text(_row_text(row, "trajectory_observed_hash", "result_observed_hash"))
            actual_result_hash: str | None = None
            if (
                source_role == FORMAL_SOURCE_ROLE
                and formal_round22_execution
                and formal_hash_readback_allowed
                and result_path
                and repo_path(result_path).is_file()
            ):
                resolved_result_path = str(repo_path(result_path).resolve())
                actual_result_hash = artifact_hash_cache.get(resolved_result_path)
                if actual_result_hash is None:
                    actual_result_hash = sha256_file(repo_path(result_path))
                    artifact_hash_cache[resolved_result_path] = actual_result_hash
            if result_hash is None:
                reasons.append("result_hash_missing_or_invalid")
            if observed_result_hash is None:
                reasons.append("result_hash_readback_missing_or_invalid")
            elif result_hash != observed_result_hash:
                reasons.append("result_hash_mismatch")
            if source_role == FORMAL_SOURCE_ROLE and formal_round22_execution and actual_result_hash != result_hash:
                reasons.append("result_artifact_hash_mismatch")
            if not _hash_verification_flag(row, "trajectory_hash_verified", "result_hash_verified"):
                reasons.append("result_hash_verification_missing")
            uncertainty_hash = _sha256_text(_row_text(row, "teacher_uncertainty_hash", "uncertainty_hash"))
            observed_uncertainty_hash = _sha256_text(
                _row_text(row, "teacher_uncertainty_observed_hash", "uncertainty_observed_hash")
            )
            actual_uncertainty_hash: str | None = None
            if (
                source_role == FORMAL_SOURCE_ROLE
                and formal_round22_execution
                and formal_hash_readback_allowed
                and uncertainty_path
                and repo_path(uncertainty_path).is_file()
            ):
                resolved_uncertainty_path = str(repo_path(uncertainty_path).resolve())
                actual_uncertainty_hash = artifact_hash_cache.get(resolved_uncertainty_path)
                if actual_uncertainty_hash is None:
                    actual_uncertainty_hash = sha256_file(repo_path(uncertainty_path))
                    artifact_hash_cache[resolved_uncertainty_path] = actual_uncertainty_hash
            if not uncertainty_path:
                reasons.append("uncertainty_artifact_missing")
            elif not repo_path(uncertainty_path).is_file():
                reasons.append("uncertainty_artifact_unreadable")
            if not uncertainty_schema_version:
                reasons.append("uncertainty_schema_version_missing")
            if (
                result_path
                and uncertainty_path
                and repo_path(result_path).resolve() == repo_path(uncertainty_path).resolve()
            ):
                reasons.append("result_uncertainty_artifact_not_distinct")
            if uncertainty_hash is None:
                reasons.append("uncertainty_hash_missing_or_invalid")
            if observed_uncertainty_hash is None:
                reasons.append("uncertainty_hash_readback_missing_or_invalid")
            elif uncertainty_hash != observed_uncertainty_hash:
                reasons.append("uncertainty_hash_mismatch")
            if source_role == FORMAL_SOURCE_ROLE and formal_round22_execution and actual_uncertainty_hash != uncertainty_hash:
                reasons.append("uncertainty_artifact_hash_mismatch")
            if not _hash_verification_flag(row, "teacher_uncertainty_hash_verified", "uncertainty_hash_verified"):
                reasons.append("uncertainty_hash_verification_missing")
            if failure_status not in {"completed", "success", "succeeded"}:
                reasons.append("source_failure_status")
            tool_version = _row_text(row, "tool_version")
            fidelity = _row_text(row, "fidelity", "fidelity_level")
            compute_cost = _nonnegative_finite(row.get("compute_cost"))
            wall_clock_cost = _nonnegative_finite(row.get("wall_clock_cost"))
            hidden_role = _row_text(row, "hidden_evaluation_role")
            provenance_level = _row_text(row, "provenance_level")
            hash_readback_uid = _row_text(row, "hash_readback_uid")
            if not tool_version:
                reasons.append("tool_version_missing")
            if not fidelity:
                reasons.append("fidelity_missing")
            if compute_cost is None:
                reasons.append("compute_cost_missing_or_invalid")
            if wall_clock_cost is None:
                reasons.append("wall_clock_cost_missing_or_invalid")
            if not hidden_role:
                reasons.append("hidden_evaluation_role_missing")
            if not provenance_level:
                reasons.append("provenance_missing")
            if not hash_readback_uid:
                reasons.append("hash_readback_uid_missing")
            formal_eligible = not reasons and bool(freeze.get("frozen"))
            outcome = {
                "result_uid": result_uid,
                "source_manifest": relative_path(source_path),
                "formal_eligible": formal_eligible,
                "eligibility_reasons": reasons,
                "state": {"state_uid": state_uid, "source": "active_learning_manifest"},
                "structured_action": {"action_uid": action_uid, "source": "active_learning_manifest"},
                "tool": {"tool_uid": tool_uid, "tool_version": tool_version or None},
                "fidelity": fidelity or None,
                "result": {
                    "artifact_path": result_path or None,
                    "schema_version": result_schema_version or None,
                    "content_hash": result_hash or None,
                    "observed_hash": observed_result_hash,
                    "artifact_sha256": actual_result_hash,
                    "hash_verified": bool(
                        result_hash
                        and result_hash == observed_result_hash
                        and result_hash == actual_result_hash
                        and _hash_verification_flag(row, "trajectory_hash_verified", "result_hash_verified")
                    ),
                },
                "uncertainty": {
                    "artifact_path": uncertainty_path or None,
                    "schema_version": uncertainty_schema_version or None,
                    "content_hash": uncertainty_hash or None,
                    "observed_hash": observed_uncertainty_hash,
                    "artifact_sha256": actual_uncertainty_hash,
                    "hash_verified": bool(
                        uncertainty_hash
                        and uncertainty_hash == observed_uncertainty_hash
                        and uncertainty_hash == actual_uncertainty_hash
                        and _hash_verification_flag(row, "teacher_uncertainty_hash_verified", "uncertainty_hash_verified")
                    ),
                },
                "actual_runtime": runtime,
                "compute_cost": compute_cost,
                "wall_clock_cost": wall_clock_cost,
                "failure_status": failure_status,
                "tool_version": tool_version or None,
                "provenance": {
                    "level": provenance_level or None,
                    "source_role": source_role,
                    "formal_round_id": formal_round_id or None,
                    "formal_round22_execution": formal_round22_execution,
                    "hash_readback_uid": hash_readback_uid or None,
                },
                "hidden_evaluation_role": hidden_role or None,
            }
            matrix_rows.append(outcome)
            by_tool[tool_uid].append(outcome)
            if reasons:
                failure_rows.append(
                    {
                        "record_uid": result_uid or f"row:{len(matrix_rows)}",
                        "source_manifest": relative_path(source_path),
                        "tool_uid": tool_uid,
                        "failure_status": failure_status,
                        "failure_kind": ";".join(reasons),
                        "observed": True,
                        "formal_eligible": False,
                        "note": "Existing active-learning artifact retained but not a formal Round-22 tool outcome.",
                    }
                )
    for tool_uid, outcomes in sorted(by_tool.items()):
        coverage_rows.append(
            {
                "tool_uid": tool_uid,
                "source_rows": len(outcomes),
                "formal_eligible_rows": sum(bool(row["formal_eligible"]) for row in outcomes),
                "completed_rows": sum(str(row["failure_status"]) in {"completed", "success", "succeeded"} for row in outcomes),
                "failure_rows": sum(str(row["failure_status"]) not in {"completed", "success", "succeeded"} for row in outcomes),
                "invalid_rows": sum(bool(row["eligibility_reasons"]) for row in outcomes),
                "runtime_observed_rows": sum(row["actual_runtime"] is not None for row in outcomes),
                "tool_version_observed": str(any(row.get("tool_version") for row in outcomes)).lower(),
                "formal_status": "formal_eligible" if any(bool(row["formal_eligible"]) for row in outcomes) else "blocked_missing_typed_runtime_or_round22_execution",
            }
        )
    matrix = {
        "schema_version": "neuronsifter-round22-tool-outcome-matrix-v1",
        "round_id": "neuronsifter_round22",
        "status": "blocked_formal_tool_outcome_environment",
        "formal_execution_allowed": False,
        "source_artifacts_are_formal": bool(source_bindings) and all(
            binding.get("role") == FORMAL_SOURCE_ROLE for binding in source_bindings
        ),
        "formal_source_binding_count": sum(binding.get("role") == FORMAL_SOURCE_ROLE for binding in source_bindings),
        "required_fields": list(TOOL_OUTCOME_FIELDS),
        "source_bindings": source_bindings,
        "source_row_count": len(matrix_rows),
        "formal_eligible_count": sum(bool(row["formal_eligible"]) for row in matrix_rows),
        "failure_catalog_count": len(failure_rows),
        "formal_metadata_contract": list(FORMAL_METADATA_FIELDS),
        "source_provenance_status": "typed_runtime_and_hash_readback_unavailable",
        "outcomes": matrix_rows,
        "failure_retention": "all observed source failures and contract defects are retained",
        "next_action": "execute_registered_tools_through_request_execution_collect_normalize_qc_assimilation",
    }
    return {"matrix": matrix, "coverage_rows": coverage_rows, "failure_rows": failure_rows}


def _not_available_rows(ids: Iterable[str], id_key: str, fields: Iterable[str], reason: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item_id in ids:
        row = {field: "NOT_AVAILABLE" for field in fields}
        row[id_key] = item_id
        row["status"] = "not_run"
        row["reason"] = reason
        rows.append(row)
    return rows


def build_c3_outputs(contract: Mapping[str, Any], freeze: Mapping[str, object], tool_matrix: Mapping[str, object]) -> dict[str, object]:
    reason = "round21_static_twin_freeze_missing_or_formal_tool_outcomes_unavailable"
    holdout_path = str(contract["c3_static"]["holdout_evidence_path"])
    holdout_payload, holdout_error = load_json_object(repo_path(holdout_path))
    hidden_outcomes_unseen = bool(
        not holdout_error
        and holdout_payload
        and holdout_payload.get("status") == "passed"
        and holdout_payload.get("final_held_out_touched") is False
    )
    policy_ids = list(contract["c3_static"]["baselines"]) + [str(contract["c3_static"]["oracle"])]
    metric_fields = ["baseline_id", "evaluation_scope", "status", *[str(value) for value in contract["c3_static"]["primary_metrics"]], "reason"]
    results = _not_available_rows(policy_ids, "baseline_id", metric_fields, reason)
    for row in results:
        row["evaluation_scope"] = "static_c3"
    write_tsv(repo_path(str(contract["c3_static"]["output_results"])), metric_fields, results)
    write_tsv(repo_path(str(contract["c3_static"]["output_budget_risk"])), ["baseline_id", "budget", "decision_risk", "status", "reason"], [{"baseline_id": value, "budget": "NOT_AVAILABLE", "decision_risk": "NOT_AVAILABLE", "status": "not_run", "reason": reason} for value in policy_ids])
    write_tsv(repo_path(str(contract["c3_static"]["output_fixed_target_cost"])), ["baseline_id", "target_risk", "total_cost", "status", "reason"], [{"baseline_id": value, "target_risk": "NOT_AVAILABLE", "total_cost": "NOT_AVAILABLE", "status": "not_run", "reason": reason} for value in policy_ids])
    write_tsv(repo_path(str(contract["c3_static"]["output_tool_usage"])), ["baseline_id", "tool_uid", "calls", "realized_cost", "wall_clock_cost", "status", "reason"], [{"baseline_id": value, "tool_uid": "NOT_AVAILABLE", "calls": "NOT_AVAILABLE", "realized_cost": "NOT_AVAILABLE", "wall_clock_cost": "NOT_AVAILABLE", "status": "not_run", "reason": reason} for value in policy_ids])
    write_tsv(repo_path(str(contract["c3_static"]["output_failure_results"])), ["baseline_id", "failure_status", "failure_count", "recovery_status", "status", "reason"], [{"baseline_id": value, "failure_status": "NOT_AVAILABLE", "failure_count": "NOT_AVAILABLE", "recovery_status": "NOT_AVAILABLE", "status": "not_run", "reason": reason} for value in policy_ids])
    gate = {
        "schema_version": "neuronsifter-round22-c3-static-agent-gate-v1",
        "gate": "c3_static_agent_gate",
        "status": "blocked_static_agent_reference",
        "passed": False,
        "static_twin_frozen": bool(freeze.get("frozen")),
        "mandatory_baselines": {value: {"status": "not_run", "result": "NOT_AVAILABLE"} for value in BASELINES},
        "mandatory_baselines_complete": False,
        "oracle": {"baseline_id": ORACLE, "deployable": False, "status": "evaluation_only_not_run"},
        "offline_replay_complete": False,
        "real_static_rounds_complete": False,
        "fixed_budget_reported": False,
        "fixed_risk_reported": False,
        "hidden_outcomes_unseen": hidden_outcomes_unseen,
        "holdout_evidence": {**evidence_binding(holdout_path), "parse_error": holdout_error},
        "tool_outcome_matrix_formal_eligible_count": int(tool_matrix.get("formal_eligible_count", 0)),
        "primary_metric_status": {value: "NOT_AVAILABLE" for value in contract["c3_static"]["primary_metrics"]},
        "reason": reason,
        "next_action": "resolve_round21_static_twin_and_real_tool_outcome_evidence",
    }
    write_json(repo_path(str(contract["c3_static"]["gate_output"])), gate)
    return gate


def build_c4_outputs(contract: Mapping[str, Any], freeze: Mapping[str, object], c3_gate: Mapping[str, object]) -> dict[str, object]:
    reason = "c3_static_reference_or_validation_evidence_unavailable"
    fields = ["variant", "calibration_split", "risk_coverage_curve", "aurc", "false_accept", "accepted_precision", "coverage", "ood_performance", "calibration_error", "tool_support_risk", "mechanism_disagreement", "post_assimilation_calibration", "status", "reason"]
    rows = []
    for variant in contract["c4_tri"]["variants"]:
        row = {field: "NOT_AVAILABLE" for field in fields}
        row.update({"variant": variant, "calibration_split": "validation_only", "status": "not_run", "reason": reason})
        rows.append(row)
    write_tsv(repo_path(str(contract["c4_tri"]["output_results"])), fields, rows)
    freeze_payload = {
        "schema_version": "neuronsifter-round22-tri-calibration-freeze-v1",
        "round_id": "neuronsifter_round22",
        "status": "blocked_tri_calibration_reference",
        "frozen": False,
        "calibration_split": "validation_only",
        "variants": list(contract["c4_tri"]["variants"]),
        "coverage_floor": float(contract["c4_tri"]["coverage_floor"]),
        "rollback_threshold": float(contract["c4_tri"]["rollback_threshold"]),
        "tri_model": None,
        "calibration_parameters": None,
        "thresholds": None,
        "reason": reason,
        "next_action": "calibrate_on_validation_only_after_static_c3_reference_is_complete",
    }
    write_json(repo_path(str(contract["c4_tri"]["output_freeze"])), freeze_payload)
    gate = {
        "schema_version": "neuronsifter-round22-c4-static-reliability-gate-v1",
        "gate": "c4_static_reliability_gate",
        "status": "blocked_tri_calibration_reference",
        "passed": False,
        "calibration_validation_only": True,
        "risk_coverage_complete": False,
        "coverage_floor_respected": False,
        "c3_gate_status": c3_gate.get("status"),
        "coverage_floor": float(contract["c4_tri"]["coverage_floor"]),
        "metrics": {value: "NOT_AVAILABLE" for value in contract["c4_tri"]["metrics"]},
        "reason": reason,
        "next_action": "resolve_c3_then_freeze_tri_model_and_thresholds",
    }
    write_json(repo_path(str(contract["c4_tri"]["gate_output"])), gate)
    return gate


def build_downstream_gates(contract: Mapping[str, Any], freeze: Mapping[str, object], c3_gate: Mapping[str, object], c4_gate: Mapping[str, object], tool_matrix: Mapping[str, object]) -> dict[str, object]:
    intermediate_passed = bool(freeze.get("frozen")) and bool(c3_gate.get("passed")) and bool(c4_gate.get("passed"))
    intermediate = {
        "schema_version": "neuronsifter-round22-static-agent-reliability-reference-gate-v1",
        "gate": "static_agent_reliability_reference_resolved",
        "status": "blocked_static_agent_reliability_reference",
        "passed": intermediate_passed,
        "static_twin_frozen": bool(freeze.get("frozen")),
        "c3": {
            "mandatory_baselines_complete": bool(c3_gate.get("mandatory_baselines_complete")),
            "offline_replay_complete": bool(c3_gate.get("offline_replay_complete")),
            "real_static_rounds_complete": bool(c3_gate.get("real_static_rounds_complete")),
            "fixed_budget_reported": bool(c3_gate.get("fixed_budget_reported")),
            "fixed_risk_reported": bool(c3_gate.get("fixed_risk_reported")),
        },
        "c4": {
            "calibration_validation_only": bool(c4_gate.get("calibration_validation_only")),
            "risk_coverage_complete": bool(c4_gate.get("risk_coverage_complete")),
            "coverage_floor_respected": bool(c4_gate.get("coverage_floor_respected")),
        },
        "holdout": {"final_hidden_outcomes_unseen": bool(c3_gate.get("hidden_outcomes_unseen"))},
        "artifacts": {"static_agent_checkpoint_frozen": False, "tri_thresholds_frozen": False},
        "next_action": "resolve_round21_static_twin_then_complete_c3_and_c4",
    }
    write_json(repo_path(str(contract["gates"]["intermediate_output"])), intermediate)
    comparison = {
        "schema_version": "neuronsifter-round22-coevolution-comparison-freeze-v1",
        "round_id": "neuronsifter_round22",
        "status": "frozen_reference_unavailable",
        "frozen": False,
        "initial_static_twin": {"status": "NOT_AVAILABLE", "freeze_artifact": contract["freeze"]["output"]},
        "initial_static_agent": {"status": "NOT_AVAILABLE", "checkpoint": None},
        "systems": {value: {"status": "not_started", "same_episode_pool": False, "same_budget": False, "same_tools": False} for value in MATRIX},
        "controls": ["naive_latest_batch_joint_update", "full_replay_joint_update"],
        "reason": "static_agent_reliability_reference_resolved is false",
        "next_action": "complete_c3_and_c4_before_any_online_agent_or_twin_update",
    }
    write_json(repo_path(str(contract["coevolution"]["output_comparison_freeze"])), comparison)
    blockers = ["blocked_round21_freeze", "blocked_static_agent", "blocked_tri_calibration", "blocked_real_tool_rounds"]
    evidence = {
        "schema_version": "neuronsifter-round22-agent-twin-coevolution-evidence-v1",
        "final_gate": contract["final_gate"],
        "status": "blocked_empirical_coevolution",
        "final_gate_passed": False,
        "real_round_count": 0,
        "static_reference": {"frozen": bool(freeze.get("frozen")), "status": freeze.get("status")},
        "static_agent_reliability_reference": {"passed": intermediate_passed, "c3_status": c3_gate.get("status"), "c4_status": c4_gate.get("status")},
        "baseline_comparisons": {value: {"status": "NOT_AVAILABLE", "fixed_budget": "NOT_AVAILABLE", "fixed_risk": "NOT_AVAILABLE"} for value in MATRIX},
        "scientific_utility": {"full_beats_static": False, "full_beats_strongest_one_sided": False, "fixed_budget_and_fixed_risk_consistent": False},
        "retention": {"catastrophic_forgetting_within_tolerance": False, "status": "NOT_AVAILABLE"},
        "reliability": {"calibration_within_tolerance": False, "coverage_floor_respected": False, "support_domain_not_degraded": False},
        "rollback": {"demonstrated": False, "status": "NOT_AVAILABLE"},
        "holdout": {"final_hidden_evidence_used_for_updates": False, "final_hidden_evidence_used_for_selection": False, "status": "not_accessed_or_not_verified"},
        "ablations": {value: "not_run" for value in contract["coevolution"]["ablations"]},
        "statistics": {**contract["coevolution"]["statistics"], "status": "not_run"},
        "tool_outcome_matrix_formal_eligible_count": int(tool_matrix.get("formal_eligible_count", 0)),
        "blockers": blockers,
        "claim_scope": "no_coevolution_claim",
        "next_action": "resolve_round21_freeze_then_run_static_c3_and_c4_before_sequential_updates",
    }
    write_json(repo_path(str(contract["coevolution"]["output_evidence"])), evidence)
    checkpoint_lineage = {
        "schema_version": "neuronsifter-round22-checkpoint-lineage-v1",
        "round_id": "neuronsifter_round22",
        "status": "blocked_no_round22_checkpoints",
        "static_twin": {"checkpoint_id": None, "checkpoint_sha256": None, "status": "NOT_AVAILABLE"},
        "static_agent": {"checkpoint_id": None, "checkpoint_sha256": None, "status": "NOT_AVAILABLE"},
        "rounds": [],
        "rollback_restorations": [],
        "next_action": "freeze_static_twin_and_static_agent_before_round_1",
    }
    write_json(repo_path(str(contract["coevolution"]["output_checkpoint_lineage"])), checkpoint_lineage)

    claims = [
        {"claim": "learned_cost_aware_scientific_information_acquisition", "status": "blocked_c3_not_evaluated", "allowed_scope": "none"},
        {"claim": "evidence_governed_agent_twin_coevolution", "status": "blocked_empirical_coevolution", "allowed_scope": "none"},
        {"claim": "static_agent_in_twin", "status": "implementation_only", "allowed_scope": "simulator_or_tool_contract_only"},
        {"claim": "static_twin_in_the_loop", "status": "implementation_only", "allowed_scope": "simulator_or_tool_contract_only"},
        {"claim": "C2-B_real_CNS_intervention_evidence", "status": "blocked_insufficient_real_e2_e4", "allowed_scope": "none"},
    ]
    write_tsv(repo_path(str(contract["coevolution"]["output_claim_scope"])), ["claim", "status", "allowed_scope"], claims)
    program = {
        "schema_version": "neuronsifter-round22-program-gate-v1",
        "gate": contract["program_gate"],
        "status": "blocked_empirical_coevolution",
        "passed": False,
        "subgates": {"static_twin_freeze": bool(freeze.get("frozen")), "static_agent_reliability_reference": intermediate_passed, "coevolution_evidence": False},
        "c2_b_status": "blocked_insufficient_real_e2_e4",
        "c2_b_recovery_artifact": evidence_binding(str(contract["claim_boundary"]["c2_b_recovery_artifact"])),
        "claim_scope_artifact": contract["coevolution"]["output_claim_scope"],
        "blockers": blockers,
        "next_action": "resolve_round21_freeze_and_required_empirical_evidence",
    }
    write_json(repo_path(str(contract["coevolution"]["output_program_gate"])), program)
    return {"intermediate": intermediate, "comparison": comparison, "evidence": evidence, "checkpoint_lineage": checkpoint_lineage, "program": program}


def _configured_output_paths(
    contract: Mapping[str, Any],
    audit_output_path: str | Path = "Output/audits/round22_readiness_audit.json",
) -> list[str]:
    """Return the explicit Round-22 outputs guarded against accidental overwrite."""
    freeze = contract["freeze"]
    tools = contract["tool_outcomes"]
    c3 = contract["c3_static"]
    c4 = contract["c4_tri"]
    gates = contract["gates"]
    coevolution = contract["coevolution"]
    repro = contract["reproducibility"]
    return [
        str(freeze["output"]),
        str(Path(str(tools["output_root"])) / "round22_tool_outcome_manifest.json"),
        str(tools["output_matrix"]),
        str(tools["output_coverage"]),
        str(tools["output_failure_catalog"]),
        str(tools["cost_model_freeze"]),
        str(c3["output_results"]),
        str(c3["output_budget_risk"]),
        str(c3["output_fixed_target_cost"]),
        str(c3["output_tool_usage"]),
        str(c3["output_failure_results"]),
        str(c3["gate_output"]),
        str(c4["output_results"]),
        str(c4["output_freeze"]),
        str(c4["gate_output"]),
        str(gates["intermediate_output"]),
        str(coevolution["output_comparison_freeze"]),
        str(coevolution["output_evidence"]),
        str(coevolution["output_checkpoint_lineage"]),
        str(coevolution["output_claim_scope"]),
        str(coevolution["output_program_gate"]),
        str(repro["output"]),
        str(repro["resource_catalog"]),
        str(audit_output_path),
    ]


def _json_is_authoritative(payload: Mapping[str, object]) -> bool:
    if payload.get("frozen") is True or payload.get("passed") is True or payload.get("final_gate_passed") is True:
        return True
    if payload.get("formal_execution_allowed") is True:
        return True
    for key in ("formal_eligible_count", "formal_job_count", "real_round_count"):
        try:
            if int(payload.get(key, 0)) > 0:
                return True
        except (TypeError, ValueError):
            return True
    if payload.get("rounds") or payload.get("rollback_restorations"):
        return True
    for nested_key in ("program_gate", "static_twin_freeze"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping) and _json_is_authoritative(nested):
            return True
    return str(payload.get("status", "")).lower() in {"completed_empirical", "failed_empirical", "empirical_complete"}


def _artifact_is_authoritative(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(payload, Mapping) and _json_is_authoritative(payload)
    if path.suffix.lower() == ".tsv":
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fields = set(reader.fieldnames or [])
                rows = list(reader)
        except (OSError, UnicodeError):
            return False
        # Claim-scope ledgers contain implementation/blocked labels by
        # design; those labels are not empirical evidence and must not make a
        # guarded rebuild look destructive.  Empirical claim states remain
        # protected.
        if {"claim", "status", "allowed_scope"}.issubset(fields):
            return any(
                row.get("status", "").strip().lower() in {"demonstrated", "partially_supported", "failed_empirical"}
                for row in rows
            )
        for row in rows:
            values = {str(value).strip().lower() for value in row.values() if value is not None}
            if row.get("status", "").strip().lower() not in {"", "not_run", "not_available"} and "not_available" not in values:
                return True
        return False
    return False


def _guard_authoritative_outputs(
    contract: Mapping[str, Any],
    allow_overwrite: bool,
    audit_output_path: str | Path = "Output/audits/round22_readiness_audit.json",
) -> None:
    if allow_overwrite:
        return
    protected = [
        path for path in _configured_output_paths(contract, audit_output_path) if _artifact_is_authoritative(repo_path(path))
    ]
    if protected:
        joined = ", ".join(protected[:8])
        suffix = " ..." if len(protected) > 8 else ""
        raise RuntimeError(
            "Refusing to rebuild Round-22 outputs while authoritative evidence exists: "
            f"{joined}{suffix}. Use the read-only validator or explicitly archive/replace the evidence."
        )


def build(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    *,
    allow_authoritative_overwrite: bool = False,
    audit_output_path: str | Path = "Output/audits/round22_readiness_audit.json",
) -> dict[str, object]:
    contract = load_contract(config_path, schema_path)
    _guard_authoritative_outputs(contract, allow_authoritative_overwrite, audit_output_path)
    freeze = build_static_twin_freeze(contract)
    write_json(repo_path(str(contract["freeze"]["output"])), freeze)
    tool_data = build_tool_outcomes(contract, freeze)
    matrix = tool_data["matrix"]
    output_root = repo_path(str(contract["tool_outcomes"]["output_root"]))
    write_json(output_root / "round22_tool_outcome_manifest.json", {**matrix, "output_root": relative_path(output_root)})
    write_json(repo_path(str(contract["tool_outcomes"]["output_matrix"])), matrix)
    write_tsv(repo_path(str(contract["tool_outcomes"]["output_coverage"])), ["tool_uid", "source_rows", "formal_eligible_rows", "completed_rows", "failure_rows", "invalid_rows", "runtime_observed_rows", "tool_version_observed", "formal_status"], tool_data["coverage_rows"])
    write_tsv(repo_path(str(contract["tool_outcomes"]["output_failure_catalog"])), ["record_uid", "source_manifest", "tool_uid", "failure_status", "failure_kind", "observed", "formal_eligible", "note"], tool_data["failure_rows"])
    cost_model = {
        "schema_version": "neuronsifter-round22-cost-model-freeze-v1",
        "round_id": "neuronsifter_round22",
        "status": "frozen_definitions_not_calibrated",
        "definitions": contract["tool_outcomes"]["cost_definitions"],
        "sensitivity_contract": contract["tool_outcomes"]["sensitivity_contract"],
        "runtime_calibration": {"status": "NOT_AVAILABLE", "observed_runtime_rows": sum(int(row["runtime_observed_rows"]) for row in tool_data["coverage_rows"])},
        "composite_weights": {"compute_weight": 0.5, "wall_weight": 0.5, "api_cost": "preserve_native_units"},
        "provenance": {"source_manifests": [row["path"] for row in matrix["source_bindings"]]},
        "next_action": "calibrate_costs_from_typed_tool_versioned_runtime_records",
    }
    write_json(repo_path(str(contract["tool_outcomes"]["cost_model_freeze"])), cost_model)
    c3 = build_c3_outputs(contract, freeze, matrix)
    c4 = build_c4_outputs(contract, freeze, c3)
    downstream = build_downstream_gates(contract, freeze, c3, c4, matrix)
    reproducibility = {
        "schema_version": "neuronsifter-round22-reproducibility-commands-v1",
        "round_id": "neuronsifter_round22",
        "status": "control_plane_commands_frozen",
        "commands": list(contract["reproducibility"]["commands"]),
        "formal_execution_commands": [],
        "formal_execution_status": "blocked",
    }
    write_json(repo_path(str(contract["reproducibility"]["output"])), reproducibility)
    resource_catalog = {
        "schema_version": "neuronsifter-round22-resource-catalog-v1",
        "round_id": "neuronsifter_round22",
        "status": "control_plane_validation_only",
        "generated_on_host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "formal_cpu_hours": "NOT_AVAILABLE",
        "formal_gpu_hours": "NOT_AVAILABLE",
        "formal_tool_cost": "NOT_AVAILABLE",
        "formal_wall_time": "NOT_AVAILABLE",
        "formal_job_count": 0,
        "note": "No C3, C4, Agent update, or Twin update job is authorized by the current gate state.",
    }
    write_json(repo_path(str(contract["reproducibility"]["resource_catalog"])), resource_catalog)
    return {
        "schema_version": "neuronsifter-round22-readiness-audit-v1",
        "round_id": "neuronsifter_round22",
        "status": downstream["program"]["status"],
        "static_twin_freeze": freeze,
        "tool_outcome_matrix": {"path": contract["tool_outcomes"]["output_matrix"], "formal_eligible_count": matrix["formal_eligible_count"]},
        "c3_gate": c3,
        "c4_gate": c4,
        "coevolution": downstream,
        "c2_b_status": "blocked_insufficient_real_e2_e4",
        "program_gate": downstream["program"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Round-22 fail-closed readiness artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output", default="Output/audits/round22_readiness_audit.json")
    parser.add_argument(
        "--allow-authoritative-overwrite",
        action="store_true",
        help="Explicitly allow replacing formal Round-22 evidence; normally the builder fails closed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(
        args.config,
        args.schema,
        allow_authoritative_overwrite=args.allow_authoritative_overwrite,
        audit_output_path=args.output,
    )
    write_json(repo_path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
