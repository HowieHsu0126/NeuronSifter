"""Build Round-22R empirical-prerequisite recovery and admission audits.

This builder never reconstructs missing Round-21 authority from prose, never
rewrites historical tool manifests, and never promotes reused outcomes into
prospective rounds.  It may validate an authority artifact already present at
the canonical Round-21 path; deterministic reconstruction requires a separate,
explicit immutable-source manifest and is therefore unavailable in the current
workspace.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    FREEZE_COMPONENT_SPECS,
    build_static_twin_freeze,
    canonical_sha256,
    load_contract,
    sha256_file,
    write_json,
)


OUTPUTS = {
    "authority": "Output/audits/round22r_round21_authority_materialization.json",
    "components": "Output/audits/round22r_static_twin_freeze_components.json",
    "cross_identity": "Output/audits/round22r_round21_cross_artifact_identity.tsv",
    "admission": "Output/audits/round22r_tool_outcome_admission.tsv",
    "hash_drift": "Output/audits/round22r_hash_drift_repair_catalog.tsv",
    "real_round": "Output/audits/round22r_real_round_eligibility.tsv",
    "gate": "Output/audits/round22r_empirical_prerequisite_gate.json",
    "resource": "Output/audits/round22r_resource_catalog.json",
    "reproducibility": "Output/audits/round22r_reproducibility_commands.json",
}

ROUND22R_ID = "neuronsifter_round22r"
EXPECTED_C1_GATE = "c1_public_foundation_state_evidence_resolved"
EXPECTED_C2S_GATE = "c2s_simulator_grounded_world_action_evidence_resolved"
C2B_BLOCKED = "blocked_insufficient_real_e2_e4"
SUCCESS_STATUSES = {"passed", "completed", "evaluated", "resolved"}
SUCCESSFUL_TOOL_STATUSES = {"completed", "success", "succeeded"}
KNOWN_FAILURE_STATUSES = {
    "timeout",
    "partial_result",
    "invalid_result",
    "tool_crash",
    "schema_mismatch",
    "stale_version",
    "contradictory_result",
}
REQUIRED_AUTHORITY_INPUTS = {
    "round21_training_input_freeze": "Output/audits/round21_training_input_freeze.json",
    "round21_checkpoint_catalog": "Output/audits/round21_checkpoint_catalog.json",
    "round21_static_twin_evidence": "Output/audits/round21_static_twin_evidence.json",
    "round21_c3_readiness": "Output/audits/round21_c3_readiness.json",
    "round21_coevolution_prerequisite_status": "Output/audits/round21_coevolution_prerequisite_status.json",
    "c1_neuronsifter_results": "Output/paper_reports/c1_neuronsifter_results.tsv",
    "c2s_neuronsifter_results": "Output/paper_reports/c2s_neuronsifter_results.tsv",
    "round21_c1_claim_gate": "Output/audits/round21_c1_claim_gate.json",
    "round21_c2s_claim_gate": "Output/audits/round21_c2s_claim_gate.json",
}
FORMAL_STATES = {
    "formal_eligible_offline",
    "formal_eligible_real_round",
    "ineligible_hash_drift",
    "ineligible_missing_artifact",
    "ineligible_identity",
    "ineligible_gate",
    "ineligible_provenance",
    "ineligible_cost_runtime",
    "ineligible_sequential_protocol",
}
REAL_ROUND_CLASSES = {
    "offline_matrix_only",
    "real_round_eligible",
    "not_sequential",
    "hidden_outcome_exposed",
    "reused_without_round_identity",
    "invalid",
}

AUTHORITY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "round21_training_input_freeze": (
        "Output/audits/current_c1_contract_artifact_projection.json",
        "Output/audits/c2_trajectory_authority.json",
        "Output/audits/dt_round17i_execution_identity.json",
        "Output/audits/public_foundation_corpus_frozen_splits.json",
    ),
    "round21_checkpoint_catalog": (
        "Output/audits/c1_scaling_transfer_summary.json",
        "Output/audits/c1_job_terminal_summary.tsv",
        "Output/audits/c2_phase_gate_execution.json",
    ),
    "round21_static_twin_evidence": (
        "Output/audits/c1_route_closeout.json",
        "Output/paper_reports/c1_gate_summary.json",
        "Output/paper_reports/c2_gate_summary.json",
        "Output/audits/baseline_registry_reduction_gate.json",
    ),
    "round21_c3_readiness": ("Output/audits/c3_static_agent_gate.json",),
    "round21_coevolution_prerequisite_status": (
        "Output/audits/agent_twin_update_contract.json",
        "Output/audits/agent_twin_wp2_fixture_scope_correction.json",
    ),
    "c1_neuronsifter_results": (
        "Output/paper_reports/c1_baseline_result_matrix.tsv",
        "Output/paper_reports/c1_foundation_transfer.tsv",
        "Output/paper_reports/c1_scaling_results.tsv",
    ),
    "c2s_neuronsifter_results": (
        "Output/paper_reports/c2_world_action_results.tsv",
        "Output/paper_reports/c2_rollout_metric_catalog.tsv",
        "Output/paper_reports/c2_uq_metric_catalog.tsv",
    ),
    "round21_c1_claim_gate": (
        "Output/paper_reports/c1_gate_summary.json",
        "Output/audits/c1_foundation_transfer_gate.json",
    ),
    "round21_c2s_claim_gate": (
        "Output/paper_reports/c2_gate_summary.json",
        "Output/audits/c2_g4_validation.json",
    ),
}

ADMISSION_FIELDS = (
    "admission_uid",
    "source_manifest",
    "source_manifest_sha256",
    "source_role",
    "source_row_number",
    "result_uid",
    "request_uid",
    "round_uid",
    "state_uid",
    "action_uid",
    "tool_uid",
    "tool_version",
    "fidelity",
    "result_artifact_path",
    "result_schema_version",
    "declared_result_sha256",
    "observed_result_sha256",
    "result_hash_matches",
    "uncertainty_artifact_path",
    "uncertainty_schema_version",
    "declared_uncertainty_sha256",
    "observed_uncertainty_sha256",
    "uncertainty_hash_matches",
    "artifacts_distinct",
    "runtime_seconds",
    "compute_cost",
    "wall_clock_cost",
    "failure_status",
    "qc_grade",
    "qc_use",
    "gate_o_passed",
    "provenance_level",
    "hash_readback_uid",
    "execution_node",
    "compute_node_verified",
    "train_eval_role",
    "c2_b_status",
    "hidden_evaluation_role",
    "formal_round22_execution",
    "formal_round_id",
    "formal_round_index",
    "admission_state",
    "eligibility_reasons",
)

HASH_DRIFT_FIELDS = (
    "parent_result_uid",
    "source_manifest",
    "round_uid",
    "tool_uid",
    "mismatch_roles",
    "result_artifact_path",
    "declared_result_sha256",
    "observed_result_sha256",
    "uncertainty_artifact_path",
    "declared_uncertainty_sha256",
    "observed_uncertainty_sha256",
    "tool_family_total_count",
    "tool_family_non_drift_count",
    "mandatory_tool_family_coverage",
    "fixed_budget_comparability",
    "fixed_risk_comparability",
    "tool_family_holdout",
    "real_round_protocol",
    "requirement_resolution_status",
    "disposition",
    "new_request_uid",
    "new_result_uid",
)

REAL_ROUND_FIELDS = (
    "result_uid",
    "source_manifest",
    "source_role",
    "round_uid",
    "formal_round_index",
    "prospective_dispatch_identity",
    "round_index_frozen_before_reveal",
    "candidate_pool_frozen_before_reveal",
    "policy_decision_before_outcome_visibility",
    "budget_state_before_dispatch",
    "tool_availability_state",
    "hidden_outcome_isolation",
    "actual_execution_chronology",
    "gate_o_admission_chronology",
    "eligibility_class",
    "reasons",
)

CROSS_IDENTITY_FIELDS = (
    "identity_key",
    "authority_input",
    "field_path",
    "observed_value",
    "expected_value",
    "source_artifact",
    "status",
    "conflict_reason",
)


def _path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    return resolved.relative_to(root_resolved).as_posix() if resolved.is_relative_to(root_resolved) else resolved.as_posix()


def _sha_text(value: object) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    return None


def _strict_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _iso_datetime(value: object) -> datetime | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _row_text(row: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    if not path.is_file():
        return [], [], "missing"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return list(reader.fieldnames or []), list(reader), ""
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [], f"invalid_tsv:{exc}"


def _json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{exc}"
    return (payload, "") if isinstance(payload, dict) else (None, "json_root_not_object")


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_identity(root: Path) -> dict[str, object]:
    tracked_diff = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "head_commit": _git_output(root, "rev-parse", "HEAD"),
        "head_tree": _git_output(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git_output(root, "branch", "--show-current"),
        "tracked_dirty": bool(tracked_diff),
    }


def _candidate_binding(path_value: str, root: Path) -> dict[str, object]:
    path = _path(path_value, root)
    binding: dict[str, object] = {
        "path": _relative(path, root),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
        "git_tracked": _git_output(root, "ls-files", "--error-unmatch", _relative(path, root)) is not None,
        "status": None,
        "schema_version": None,
    }
    if path.suffix == ".json" and path.is_file():
        payload, _ = _json_object(path)
        if payload is not None:
            binding["status"] = payload.get("status")
            binding["schema_version"] = payload.get("schema_version")
    return binding


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
        if not str(git_identity.get("commit", "")).strip() or not str(git_identity.get("tree", "")).strip():
            errors.append("git_commit_or_tree_missing")
        if git_identity.get("dirty") is not False:
            errors.append("git_identity_not_clean")
    if not str(payload.get("status", "")).strip():
        errors.append("status_missing")
    return errors


def _validate_json_authority(input_id: str, payload: Mapping[str, Any], root: Path) -> list[str]:
    errors = _metadata_errors(payload)
    if input_id == "round21_training_input_freeze":
        for key in (
            "model_config_hash",
            "c1_data_hash",
            "c2s_data_hash",
            "training_split_hash",
            "validation_split_hash",
            "final_holdout_split_hash",
            "environment_hash",
            "tri_input_schema_hash",
        ):
            if _sha_text(payload.get(key)) is None:
                errors.append(f"{key}_missing_or_invalid")
    elif input_id == "round21_checkpoint_catalog":
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            errors.append("checkpoint_catalog_empty")
        else:
            validation_rows = [row for row in checkpoints if isinstance(row, Mapping) and row.get("selection_split") == "validation"]
            if not validation_rows:
                errors.append("validation_checkpoint_missing")
            for row in validation_rows:
                path_value = row.get("checkpoint_path")
                declared_hash = _sha_text(row.get("checkpoint_sha256"))
                if not isinstance(path_value, str) or not _path(path_value, root).is_file():
                    errors.append("validation_checkpoint_artifact_missing")
                elif declared_hash is None or sha256_file(_path(path_value, root)) != declared_hash:
                    errors.append("validation_checkpoint_hash_mismatch")
    elif input_id == "round21_static_twin_evidence":
        for key in (
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
        ):
            if _sha_text(payload.get(key)) is None:
                errors.append(f"{key}_missing_or_invalid")
    elif input_id == "round21_c3_readiness":
        if payload.get("static_twin_ready") is not True:
            errors.append("static_twin_ready_not_true")
        if payload.get("hidden_outcomes_unseen") is not True:
            errors.append("hidden_outcome_isolation_missing")
    elif input_id == "round21_coevolution_prerequisite_status":
        if payload.get("c1_gate") != EXPECTED_C1_GATE:
            errors.append("c1_gate_identity_mismatch")
        if payload.get("c2s_gate") != EXPECTED_C2S_GATE:
            errors.append("c2s_gate_identity_mismatch")
        if payload.get("c2_b_status") != C2B_BLOCKED:
            errors.append("c2_b_status_drift")
    elif input_id in {"round21_c1_claim_gate", "round21_c2s_claim_gate"}:
        expected = EXPECTED_C1_GATE if input_id == "round21_c1_claim_gate" else EXPECTED_C2S_GATE
        if payload.get("gate") != expected:
            errors.append("claim_gate_identity_mismatch")
        if payload.get("passed") is not True:
            errors.append("claim_gate_not_passed")
        for key in ("config_hash", "data_hash", "split_hash", "checkpoint_hash"):
            if _sha_text(payload.get(key)) is None:
                errors.append(f"{key}_missing_or_invalid")
    return errors


def _validate_result_tsv(path: Path) -> list[str]:
    fields, rows, error = _read_tsv(path)
    if error:
        return [error]
    required = {
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
    errors: list[str] = []
    if required - set(fields):
        errors.append("result_schema_incomplete")
    if not rows:
        errors.append("result_rows_empty")
    for row in rows:
        if str(row.get("status", "")).strip().lower() not in SUCCESS_STATUSES:
            errors.append("result_row_not_completed")
            break
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            errors.append("result_value_nonfinite")
            break
        if not math.isfinite(value):
            errors.append("result_value_nonfinite")
            break
        if (
            not str(row.get("schema_version", "")).strip()
            or not str(row.get("producer_uid", "")).strip()
            or not str(row.get("git_commit", "")).strip()
            or not str(row.get("git_tree", "")).strip()
            or _strict_bool(row.get("git_dirty")) is not False
        ):
            errors.append("result_producer_or_git_missing")
            break
        if any(_sha_text(row.get(key)) is None for key in ("config_hash", "data_hash", "split_hash", "checkpoint_hash")):
            errors.append("result_identity_hash_missing")
            break
    return errors


def inspect_authority_inputs(contract: Mapping[str, Any], root: Path = REPO_ROOT) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    configured_items = list(contract["freeze"]["required_inputs"])
    configured_pairs = [(str(item["input_id"]), str(item["path"])) for item in configured_items]
    configured_ids = [input_id for input_id, _ in configured_pairs]
    contract_identity_errors: list[str] = []
    if len(configured_ids) != len(set(configured_ids)):
        contract_identity_errors.append("duplicate_authority_input_id")
    if set(configured_ids) != set(REQUIRED_AUTHORITY_INPUTS):
        contract_identity_errors.append("authority_input_id_set_drift")
    if any(REQUIRED_AUTHORITY_INPUTS.get(input_id) != path for input_id, path in configured_pairs):
        contract_identity_errors.append("authority_input_path_drift")

    for item in configured_items:
        input_id = str(item["input_id"])
        path = _path(str(item["path"]), root)
        exists = path.is_file()
        errors: list[str] = []
        schema_value: str | None = None
        producer: object = None
        status: object = None
        if exists and path.suffix == ".json":
            payload, parse_error = _json_object(path)
            if parse_error:
                errors.append(parse_error)
            elif payload is not None:
                payloads[input_id] = payload
                schema_value = str(payload.get("schema_version", "")) or None
                producer = payload.get("producer")
                status = payload.get("status")
                errors.extend(_validate_json_authority(input_id, payload, root))
        elif exists:
            errors.extend(_validate_result_tsv(path))
            status = "validated_tsv" if not errors else "invalid_tsv"
        else:
            errors.append("canonical_authority_artifact_missing")
        candidates = [_candidate_binding(value, root) for value in AUTHORITY_CANDIDATES.get(input_id, ())]
        entry = {
            "input_id": input_id,
            "path": _relative(path, root),
            "exists": exists,
            "bytes": path.stat().st_size if exists else 0,
            "sha256": sha256_file(path) if exists else None,
            "schema_version": schema_value,
            "producer": producer,
            "status": status,
            "validated": exists and not errors,
            "hash_bound": exists and not errors,
            "validation_errors": errors,
            "reconstruction_status": "not_required_canonical_artifact_present" if exists else "not_reconstructible_from_current_immutable_sources",
            "reconstruction_candidates": candidates,
        }
        entries.append(entry)

    present = sum(bool(row["exists"]) for row in entries)
    validated = sum(bool(row["validated"]) for row in entries)
    hash_bound = sum(bool(row["hash_bound"]) for row in entries)
    authority_passed = (
        not contract_identity_errors
        and present == validated == hash_bound == len(entries) == len(REQUIRED_AUTHORITY_INPUTS)
    )
    report: dict[str, object] = {
        "schema_version": "neuronsifter-round22r-round21-authority-materialization-v1",
        "round_id": ROUND22R_ID,
        "status": "passed" if authority_passed else "blocked_round21_authority_unavailable",
        "artifact_uid": canonical_sha256({"entries": entries, "round_id": ROUND22R_ID}),
        "round21_authority_inputs": {
            "required": len(entries),
            "present": present,
            "validated": validated,
            "hash_bound": hash_bound,
        },
        "retraining_used_for_recovery": False,
        "narrative_fields_used_for_recovery": False,
        "historical_manifests_rewritten": False,
        "required_authority_input_map": REQUIRED_AUTHORITY_INPUTS,
        "contract_identity_errors": contract_identity_errors,
        "deterministic_reconstruction_command": None,
        "git_identity": _git_identity(root),
        "entries": entries,
        "blockers": contract_identity_errors + [str(row["input_id"]) for row in entries if not row["validated"]],
        "next_action": "provide_or_reconstruct_round21_authority_from_explicit_immutable_machine_readable_sources",
    }
    return report


def _identity_status(observed: object, expected: object) -> tuple[str, str]:
    observed_hash = _sha_text(observed)
    expected_hash = _sha_text(expected)
    if observed_hash is None:
        return "missing", "observed_identity_missing_or_invalid"
    if expected_hash is None:
        return "missing", "expected_identity_missing_or_invalid"
    if observed_hash != expected_hash:
        return "conflict", "cross_artifact_identity_mismatch"
    return "matched", ""


def _cross_identity_row(
    *,
    identity_key: str,
    authority_input: str,
    field_path: str,
    observed: object,
    expected: object,
    source_artifact: str,
) -> dict[str, object]:
    status, reason = _identity_status(observed, expected)
    return {
        "identity_key": identity_key,
        "authority_input": authority_input,
        "field_path": field_path,
        "observed_value": observed or "",
        "expected_value": expected or "",
        "source_artifact": source_artifact,
        "status": status,
        "conflict_reason": reason,
    }


def _load_authority_content(
    contract: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, str]]]]:
    json_payloads: dict[str, dict[str, Any]] = {}
    tsv_rows: dict[str, list[dict[str, str]]] = {}
    for item in contract["freeze"]["required_inputs"]:
        input_id = str(item["input_id"])
        path = _path(str(item["path"]), root)
        if path.suffix == ".json":
            payload, error = _json_object(path)
            if payload is not None and not error:
                json_payloads[input_id] = payload
        else:
            _, rows, error = _read_tsv(path)
            if not error:
                tsv_rows[input_id] = rows
    return json_payloads, tsv_rows


def build_freeze_component_report(
    contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    freeze = build_static_twin_freeze(contract, root)
    values = {
        "checkpoint_hash": freeze.get("twin_checkpoint_hash"),
        "model_config_hash": freeze.get("model_config_hash"),
        "c1_data_hash": freeze.get("data_hashes", {}).get("c1"),
        "c2_data_hash": freeze.get("data_hashes", {}).get("c2"),
        "training_split_hash": freeze.get("split_hashes", {}).get("training"),
        "validation_split_hash": freeze.get("split_hashes", {}).get("validation"),
        "holdout_split_hash": freeze.get("split_hashes", {}).get("holdout"),
        "environment_hash": freeze.get("environment_hash"),
        "baseline_frontier_hash": freeze.get("baseline_frontier_hash"),
        "tri_input_schema_hash": freeze.get("tri_input_schema_hash"),
    }
    components: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    for key, value in values.items():
        source = "round21_checkpoint_catalog" if key == "checkpoint_hash" else FREEZE_COMPONENT_SPECS[key][0]
        valid = _sha_text(value) is not None
        components.append(
            {
                "component": key,
                "source_authority_input": source,
                "value": value,
                "present": value is not None,
                "validated": valid,
                "conflict": key in freeze.get("freeze_component_conflicts", {}),
            }
        )
        cross_rows.append(
            {
                "identity_key": key,
                "authority_input": source,
                "field_path": key,
                "observed_value": value or "",
                "expected_value": value or "",
                "source_artifact": source,
                "status": "matched" if valid else "missing",
                "conflict_reason": json.dumps(
                    freeze.get("freeze_component_conflicts", {}).get(key, ""),
                    ensure_ascii=True,
                ),
            }
        )

    json_payloads, tsv_rows = _load_authority_content(contract, root)
    evidence = json_payloads.get("round21_static_twin_evidence", {})
    c1_gate = json_payloads.get("round21_c1_claim_gate", {})
    c2s_gate = json_payloads.get("round21_c2s_claim_gate", {})
    entry_hashes = {
        str(row.get("input_id")): row.get("sha256")
        for row in authority.get("entries", [])
        if isinstance(row, Mapping)
    }
    direct_checks = (
        ("checkpoint_hash", "round21_static_twin_evidence", "selected_checkpoint_hash", evidence.get("selected_checkpoint_hash"), values["checkpoint_hash"]),
        ("model_config_hash", "round21_static_twin_evidence", "model_config_hash", evidence.get("model_config_hash"), values["model_config_hash"]),
        ("c1_data_hash", "round21_static_twin_evidence", "c1_data_hash", evidence.get("c1_data_hash"), values["c1_data_hash"]),
        ("c2_data_hash", "round21_static_twin_evidence", "c2s_data_hash", evidence.get("c2s_data_hash"), values["c2_data_hash"]),
        ("environment_hash", "round21_static_twin_evidence", "environment_hash", evidence.get("environment_hash"), values["environment_hash"]),
        ("baseline_frontier_hash", "round21_static_twin_evidence", "baseline_frontier_hash", evidence.get("baseline_frontier_hash"), values["baseline_frontier_hash"]),
        ("tri_input_schema_hash", "round21_static_twin_evidence", "tri_input_schema_hash", evidence.get("tri_input_schema_hash"), values["tri_input_schema_hash"]),
        ("c1_claim_gate_artifact_hash", "round21_static_twin_evidence", "c1_claim_gate_hash", evidence.get("c1_claim_gate_hash"), entry_hashes.get("round21_c1_claim_gate")),
        ("c2s_claim_gate_artifact_hash", "round21_static_twin_evidence", "c2s_claim_gate_hash", evidence.get("c2s_claim_gate_hash"), entry_hashes.get("round21_c2s_claim_gate")),
        ("c1_gate_config_hash", "round21_c1_claim_gate", "config_hash", c1_gate.get("config_hash"), values["model_config_hash"]),
        ("c1_gate_data_hash", "round21_c1_claim_gate", "data_hash", c1_gate.get("data_hash"), values["c1_data_hash"]),
        ("c1_gate_checkpoint_hash", "round21_c1_claim_gate", "checkpoint_hash", c1_gate.get("checkpoint_hash"), values["checkpoint_hash"]),
        ("c2s_gate_config_hash", "round21_c2s_claim_gate", "config_hash", c2s_gate.get("config_hash"), values["model_config_hash"]),
        ("c2s_gate_data_hash", "round21_c2s_claim_gate", "data_hash", c2s_gate.get("data_hash"), values["c2_data_hash"]),
        ("c2s_gate_checkpoint_hash", "round21_c2s_claim_gate", "checkpoint_hash", c2s_gate.get("checkpoint_hash"), values["checkpoint_hash"]),
        ("c1_split_hash", "round21_static_twin_evidence", "c1_split_hash", evidence.get("c1_split_hash"), c1_gate.get("split_hash")),
        ("c2s_split_hash", "round21_static_twin_evidence", "c2s_split_hash", evidence.get("c2s_split_hash"), c2s_gate.get("split_hash")),
    )
    for identity_key, input_id, field_path, observed, expected in direct_checks:
        cross_rows.append(
            _cross_identity_row(
                identity_key=identity_key,
                authority_input=input_id,
                field_path=field_path,
                observed=observed,
                expected=expected,
                source_artifact=REQUIRED_AUTHORITY_INPUTS[input_id],
            )
        )

    for input_id, expected_values in (
        (
            "c1_neuronsifter_results",
            {
                "config_hash": values["model_config_hash"],
                "data_hash": values["c1_data_hash"],
                "split_hash": c1_gate.get("split_hash"),
                "checkpoint_hash": values["checkpoint_hash"],
            },
        ),
        (
            "c2s_neuronsifter_results",
            {
                "config_hash": values["model_config_hash"],
                "data_hash": values["c2_data_hash"],
                "split_hash": c2s_gate.get("split_hash"),
                "checkpoint_hash": values["checkpoint_hash"],
            },
        ),
    ):
        for row_index, row in enumerate(tsv_rows.get(input_id, []), start=2):
            for field_path, expected in expected_values.items():
                cross_rows.append(
                    _cross_identity_row(
                        identity_key=f"{input_id}:{field_path}",
                        authority_input=input_id,
                        field_path=f"row[{row_index}].{field_path}",
                        observed=row.get(field_path),
                        expected=expected,
                        source_artifact=REQUIRED_AUTHORITY_INPUTS[input_id],
                    )
                )
    present = sum(bool(row["present"]) for row in components)
    validated = sum(bool(row["validated"]) and not bool(row["conflict"]) for row in components)
    authority_passed = authority.get("status") == "passed"
    cross_conflicts = [row for row in cross_rows if row["status"] == "conflict"]
    cross_missing = [row for row in cross_rows if row["status"] == "missing"]
    passed = (
        authority_passed
        and present == validated == 10
        and freeze.get("frozen") is True
        and not cross_conflicts
        and not cross_missing
    )
    selected = freeze.get("selected_validation_checkpoint")
    test_selected = isinstance(selected, Mapping) and selected.get("selection_split") in {"test", "final_holdout"}
    report = {
        "schema_version": "neuronsifter-round22r-static-twin-freeze-components-v1",
        "round_id": ROUND22R_ID,
        "status": "passed" if passed else "blocked_static_twin_freeze_components",
        "component_count": len(components),
        "present_count": present,
        "validated_count": validated,
        "identity_conflict_count": len(freeze.get("freeze_component_conflicts", {})) + len(cross_conflicts),
        "cross_identity_missing_count": len(cross_missing),
        "test_selected_checkpoint": test_selected,
        "selection_rule": freeze.get("selection_rule"),
        "selected_validation_checkpoint": selected,
        "checkpoint_hash_verified": freeze.get("checkpoint_artifact_binding", {}).get("hash_matches") is True,
        "all_authority_inputs_bound": authority_passed,
        "retraining_used_for_recovery": False,
        "static_twin_freeze_allowed": passed and not test_selected,
        "components": components,
        "round22_freeze_preview": freeze,
        "next_action": "materialize_and_validate_all_round21_frozen_inputs" if not passed else "freeze_round22_static_twin_reference",
    }
    return report, cross_rows


def _artifact_hash(path_value: str, root: Path, cache: dict[str, str]) -> tuple[str | None, bool]:
    if not path_value:
        return None, False
    path = _path(path_value, root)
    if not path.is_file():
        return None, False
    key = str(path.resolve())
    if key not in cache:
        cache[key] = sha256_file(path)
    return cache[key], True


def _chronology(row: Mapping[str, object]) -> tuple[dict[str, bool], list[str]]:
    dispatch_uid = _row_text(row, "prospective_dispatch_uid", "dispatch_uid")
    round_index = _row_text(row, "formal_round_index", "round_index")
    dispatch_at = _iso_datetime(_row_text(row, "prospective_dispatch_at", "dispatch_at"))
    round_frozen_at = _iso_datetime(_row_text(row, "round_frozen_at"))
    candidate_pool_frozen_at = _iso_datetime(_row_text(row, "candidate_pool_frozen_at"))
    decision_at = _iso_datetime(_row_text(row, "policy_decision_at"))
    budget_frozen_at = _iso_datetime(_row_text(row, "budget_state_frozen_at"))
    availability_frozen_at = _iso_datetime(_row_text(row, "tool_availability_frozen_at"))
    outcome_visible_at = _iso_datetime(_row_text(row, "outcome_visible_at", "outcome_revealed_at"))
    execution_started_at = _iso_datetime(_row_text(row, "execution_started_at"))
    execution_finished_at = _iso_datetime(_row_text(row, "execution_finished_at"))
    gate_o_at = _iso_datetime(_row_text(row, "gate_o_admitted_at"))
    candidate_pool_hash = _sha_text(_row_text(row, "candidate_pool_hash"))
    budget_hash = _sha_text(_row_text(row, "budget_state_hash"))
    availability_hash = _sha_text(_row_text(row, "tool_availability_hash"))
    hidden_isolated = _strict_bool(row.get("hidden_outcome_isolated")) is True
    try:
        round_ok = 1 <= int(round_index) <= 3
    except (TypeError, ValueError):
        round_ok = False
    decision_before_visibility = bool(
        decision_at
        and dispatch_at
        and outcome_visible_at
        and decision_at <= dispatch_at < outcome_visible_at
    )
    execution_order = bool(
        dispatch_at
        and
        execution_started_at
        and execution_finished_at
        and outcome_visible_at
        and dispatch_at <= execution_started_at <= execution_finished_at <= outcome_visible_at
    )
    values = {
        "prospective_dispatch_identity": bool(dispatch_uid and dispatch_at),
        "round_index_frozen_before_reveal": bool(
            round_ok and round_frozen_at and decision_at and round_frozen_at <= decision_at
        ),
        "candidate_pool_frozen_before_reveal": bool(
            candidate_pool_hash and candidate_pool_frozen_at and decision_at and candidate_pool_frozen_at <= decision_at
        ),
        "policy_decision_before_outcome_visibility": decision_before_visibility,
        "budget_state_before_dispatch": bool(
            budget_hash and budget_frozen_at and dispatch_at and budget_frozen_at <= dispatch_at
        ),
        "tool_availability_state": bool(
            availability_hash
            and availability_frozen_at
            and dispatch_at
            and availability_frozen_at <= dispatch_at
        ),
        "hidden_outcome_isolation": hidden_isolated,
        "actual_execution_chronology": execution_order,
        "gate_o_admission_chronology": bool(
            gate_o_at
            and execution_finished_at
            and outcome_visible_at
            and execution_finished_at <= gate_o_at <= outcome_visible_at
        ),
    }
    return values, [key for key, value in values.items() if not value]


def build_tool_admission(
    contract: Mapping[str, Any],
    root: Path = REPO_ROOT,
    allow_hash_readback: bool | None = None,
) -> dict[str, object]:
    if allow_hash_readback is None:
        allow_hash_readback = socket.gethostname() != "mgmt2"
    hash_cache: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, str]] = []
    source_bindings: list[dict[str, object]] = []
    for source in contract["tool_outcomes"]["source_manifests"]:
        source_path = _path(str(source["path"]), root)
        fields, source_rows, error = _read_tsv(source_path)
        source_sha256 = sha256_file(source_path) if source_path.is_file() else None
        source_bindings.append(
            {
                "path": _relative(source_path, root),
                "role": source["role"],
                "exists": source_path.is_file(),
                "sha256": source_sha256,
                "row_count": len(source_rows),
                "read_error": error,
                "fields": fields,
            }
        )
        for index, row in enumerate(source_rows, start=2):
            raw = dict(row)
            raw["__source_manifest"] = _relative(source_path, root)
            raw["__source_manifest_sha256"] = source_sha256 or ""
            raw["__source_role"] = str(source["role"])
            raw["__source_row_number"] = str(index)
            raw_rows.append(raw)

    for raw in raw_rows:
        result_path = _row_text(raw, "trajectory_artifact_path", "result_artifact_path")
        uncertainty_path = _row_text(raw, "teacher_uncertainty_path", "uncertainty_artifact_path")
        observed_result, result_exists = _artifact_hash(result_path, root, hash_cache) if allow_hash_readback else (None, False)
        observed_uncertainty, uncertainty_exists = (
            _artifact_hash(uncertainty_path, root, hash_cache) if allow_hash_readback else (None, False)
        )
        declared_result = _sha_text(_row_text(raw, "trajectory_hash", "result_hash"))
        declared_uncertainty = _sha_text(_row_text(raw, "teacher_uncertainty_hash", "uncertainty_hash"))
        adapter_observed_result = _sha_text(
            _row_text(raw, "trajectory_observed_hash", "result_observed_hash")
        )
        adapter_observed_uncertainty = _sha_text(
            _row_text(raw, "teacher_uncertainty_observed_hash", "uncertainty_observed_hash")
        )
        adapter_result_verified = _strict_bool(
            raw.get("trajectory_hash_verified", raw.get("result_hash_verified"))
        )
        adapter_uncertainty_verified = _strict_bool(
            raw.get("teacher_uncertainty_hash_verified", raw.get("uncertainty_hash_verified"))
        )
        result_drift = declared_result is not None and observed_result is not None and declared_result != observed_result
        uncertainty_drift = (
            declared_uncertainty is not None
            and observed_uncertainty is not None
            and declared_uncertainty != observed_uncertainty
        )
        state_uid = _row_text(raw, "state_uid", "candidate_uid")
        request_uid = _row_text(raw, "request_uid")
        action_uid = _row_text(raw, "structured_action_uid", "action_uid")
        tool_uid = _row_text(raw, "tool_uid", "simulator_family")
        tool_version = _row_text(raw, "tool_version")
        fidelity = _row_text(raw, "fidelity", "fidelity_level")
        result_schema = _row_text(raw, "result_schema_version", "trajectory_schema_version")
        uncertainty_schema = _row_text(raw, "uncertainty_schema_version", "teacher_uncertainty_schema_version")
        runtime = _finite_nonnegative(raw.get("runtime_seconds"))
        compute_cost = _finite_nonnegative(raw.get("compute_cost"))
        wall_cost = _finite_nonnegative(raw.get("wall_clock_cost"))
        failure_status = _row_text(raw, "failure_status").lower()
        qc_grade = _row_text(raw, "qc_grade")
        qc_use = _row_text(raw, "qc_use")
        gate_o_value = _strict_bool(raw.get("gate_o_passed"))
        gate_o_passed = gate_o_value is True
        provenance = _row_text(raw, "provenance_level")
        hash_readback_uid = _row_text(raw, "hash_readback_uid")
        execution_node = _row_text(raw, "execution_node", "compute_node")
        compute_node_value = _strict_bool(raw.get("compute_node_verified"))
        compute_node_verified = compute_node_value is True and execution_node in {"cpu03", "cpu04", "gpu03"}
        train_eval_role = _row_text(raw, "train_eval_role", "evidence_role", "evaluation_role")
        c2_b_status = _row_text(raw, "c2_b_status")
        hidden_role = _row_text(raw, "hidden_evaluation_role")
        formal_execution_value = _strict_bool(raw.get("formal_round22_execution"))
        formal_execution = formal_execution_value is True
        formal_round_id = _row_text(raw, "formal_round_id", "round22_round_id")
        formal_round_index = _row_text(raw, "formal_round_index", "round_index")
        source_role = str(raw["__source_role"])
        artifacts_distinct = bool(
            result_path
            and uncertainty_path
            and _path(result_path, root).resolve() != _path(uncertainty_path, root).resolve()
        )

        reasons: list[str] = []
        if not allow_hash_readback:
            reasons.append("compute_node_hash_readback_not_performed")
        if not result_exists:
            reasons.append("result_artifact_missing")
        if not uncertainty_exists:
            reasons.append("uncertainty_artifact_missing")
        if declared_result is None:
            reasons.append("declared_result_hash_missing")
        if declared_uncertainty is None:
            reasons.append("declared_uncertainty_hash_missing")
        if adapter_observed_result is None or adapter_observed_result != observed_result:
            reasons.append("adapter_result_hash_readback_missing_or_mismatch")
        if adapter_observed_uncertainty is None or adapter_observed_uncertainty != observed_uncertainty:
            reasons.append("adapter_uncertainty_hash_readback_missing_or_mismatch")
        if adapter_result_verified is not True:
            reasons.append("adapter_result_hash_verification_missing")
        if adapter_uncertainty_verified is not True:
            reasons.append("adapter_uncertainty_hash_verification_missing")
        if result_drift:
            reasons.append("result_hash_drift")
        if uncertainty_drift:
            reasons.append("uncertainty_hash_drift")
        if not state_uid:
            reasons.append("state_uid_missing")
        if not action_uid:
            reasons.append("action_uid_missing")
        if not request_uid:
            reasons.append("request_uid_missing")
        if not tool_uid:
            reasons.append("tool_identity_missing")
        if not tool_version:
            reasons.append("tool_version_missing")
        if not fidelity:
            reasons.append("fidelity_missing")
        if not result_schema:
            reasons.append("result_schema_version_missing")
        if not uncertainty_schema:
            reasons.append("uncertainty_schema_version_missing")
        if not artifacts_distinct:
            reasons.append("result_uncertainty_artifacts_not_independent")
        if runtime is None:
            reasons.append("runtime_missing_or_invalid")
        if compute_cost is None:
            reasons.append("compute_cost_missing_or_invalid")
        if wall_cost is None:
            reasons.append("wall_clock_cost_missing_or_invalid")
        if failure_status not in {"completed", "success", "succeeded"}:
            reasons.append("tool_execution_not_successful")
        if failure_status not in SUCCESSFUL_TOOL_STATUSES | KNOWN_FAILURE_STATUSES:
            reasons.append("failure_status_unknown")
        if qc_grade not in {"A", "B"}:
            reasons.append("qc_not_main_evidence_eligible")
        if not qc_use:
            reasons.append("qc_use_missing")
        if not gate_o_passed:
            reasons.append("gate_o_admission_missing")
        if gate_o_value is None:
            reasons.append("gate_o_boolean_invalid")
        if not provenance:
            reasons.append("provenance_missing")
        if source_role != "round22_formal_tool_outcome":
            reasons.append("source_role_not_formal_round22")
        if not hash_readback_uid:
            reasons.append("hash_readback_uid_missing")
        if not compute_node_verified:
            reasons.append("compute_node_verification_missing_or_invalid")
        if not train_eval_role:
            reasons.append("train_eval_role_missing")
        if c2_b_status != C2B_BLOCKED:
            reasons.append("c2_b_binding_missing_or_invalid")
        if not hidden_role:
            reasons.append("hidden_evaluation_role_missing")
        if "final_hidden" in hidden_role.lower() and "train" in train_eval_role.lower():
            reasons.append("final_hidden_outcome_assigned_to_training")
        if formal_execution_value is None:
            reasons.append("formal_execution_boolean_invalid")

        chronology, chronology_reasons = _chronology(raw)
        if result_drift or uncertainty_drift:
            state = "ineligible_hash_drift"
        elif not result_exists or not uncertainty_exists:
            state = "ineligible_missing_artifact"
        elif any(
            reason in reasons
            for reason in (
                "state_uid_missing",
                "action_uid_missing",
                "tool_identity_missing",
                "tool_version_missing",
                "fidelity_missing",
                "result_schema_version_missing",
                "uncertainty_schema_version_missing",
                "result_uncertainty_artifacts_not_independent",
                "request_uid_missing",
            )
        ):
            state = "ineligible_identity"
        elif any(reason in reasons for reason in ("tool_execution_not_successful", "qc_not_main_evidence_eligible", "gate_o_admission_missing")):
            state = "ineligible_gate"
        elif any(
            reason in reasons
            for reason in (
                "declared_result_hash_missing",
                "declared_uncertainty_hash_missing",
                "adapter_result_hash_readback_missing_or_mismatch",
                "adapter_uncertainty_hash_readback_missing_or_mismatch",
                "adapter_result_hash_verification_missing",
                "adapter_uncertainty_hash_verification_missing",
                "provenance_missing",
                "source_role_not_formal_round22",
                "hash_readback_uid_missing",
                "compute_node_verification_missing_or_invalid",
                "train_eval_role_missing",
                "c2_b_binding_missing_or_invalid",
                "hidden_evaluation_role_missing",
                "final_hidden_outcome_assigned_to_training",
                "formal_execution_boolean_invalid",
            )
        ):
            state = "ineligible_provenance"
        elif any(reason in reasons for reason in ("runtime_missing_or_invalid", "compute_cost_missing_or_invalid", "wall_clock_cost_missing_or_invalid")):
            state = "ineligible_cost_runtime"
        elif reasons:
            state = "ineligible_provenance"
        elif formal_execution and (
            formal_round_id != contract["round_id"] or chronology_reasons
        ):
            state = "ineligible_sequential_protocol"
        elif formal_execution:
            state = "formal_eligible_real_round"
        else:
            state = "formal_eligible_offline"

        result_uid = _row_text(raw, "result_uid")
        rows.append(
            {
                "admission_uid": canonical_sha256(
                    {
                        "result_uid": result_uid,
                        "source": raw["__source_manifest"],
                        "row": raw["__source_row_number"],
                        "observed_result": observed_result,
                        "observed_uncertainty": observed_uncertainty,
                    }
                ),
                "source_manifest": raw["__source_manifest"],
                "source_manifest_sha256": raw["__source_manifest_sha256"],
                "source_role": source_role,
                "source_row_number": raw["__source_row_number"],
                "result_uid": result_uid,
                "request_uid": request_uid,
                "round_uid": _row_text(raw, "round_uid"),
                "state_uid": state_uid,
                "action_uid": action_uid,
                "tool_uid": tool_uid,
                "tool_version": tool_version,
                "fidelity": fidelity,
                "result_artifact_path": result_path,
                "result_schema_version": result_schema,
                "declared_result_sha256": declared_result or "",
                "observed_result_sha256": observed_result or "",
                "result_hash_matches": str(bool(declared_result and declared_result == observed_result)).lower(),
                "uncertainty_artifact_path": uncertainty_path,
                "uncertainty_schema_version": uncertainty_schema,
                "declared_uncertainty_sha256": declared_uncertainty or "",
                "observed_uncertainty_sha256": observed_uncertainty or "",
                "uncertainty_hash_matches": str(bool(declared_uncertainty and declared_uncertainty == observed_uncertainty)).lower(),
                "artifacts_distinct": str(artifacts_distinct).lower(),
                "runtime_seconds": "" if runtime is None else runtime,
                "compute_cost": "" if compute_cost is None else compute_cost,
                "wall_clock_cost": "" if wall_cost is None else wall_cost,
                "failure_status": failure_status,
                "qc_grade": qc_grade,
                "qc_use": qc_use,
                "gate_o_passed": str(gate_o_passed).lower(),
                "provenance_level": provenance,
                "hash_readback_uid": hash_readback_uid,
                "execution_node": execution_node,
                "compute_node_verified": str(compute_node_verified).lower(),
                "train_eval_role": train_eval_role,
                "c2_b_status": c2_b_status,
                "hidden_evaluation_role": hidden_role,
                "formal_round22_execution": str(formal_execution).lower(),
                "formal_round_id": formal_round_id,
                "formal_round_index": formal_round_index,
                "admission_state": state,
                "eligibility_reasons": ";".join(sorted(set(reasons + ([f"chronology:{item}" for item in chronology_reasons] if formal_execution else [])))),
                "__raw": raw,
                "__chronology": chronology,
                "__chronology_reasons": chronology_reasons,
            }
        )

    tool_totals = Counter(str(row["tool_uid"]) for row in rows)
    drift_rows = [row for row in rows if row["admission_state"] == "ineligible_hash_drift"]
    tool_drift = Counter(str(row["tool_uid"]) for row in drift_rows)
    repair_rows: list[dict[str, object]] = []
    for row in drift_rows:
        mismatch_roles = [
            role
            for role in ("result", "uncertainty")
            if row[f"declared_{role}_sha256"]
            and row[f"observed_{role}_sha256"]
            and row[f"declared_{role}_sha256"] != row[f"observed_{role}_sha256"]
        ]
        tool_uid = str(row["tool_uid"])
        non_drift_count = tool_totals[tool_uid] - tool_drift[tool_uid]
        repair_rows.append(
            {
                "parent_result_uid": row["result_uid"],
                "source_manifest": row["source_manifest"],
                "round_uid": row["round_uid"],
                "tool_uid": tool_uid,
                "mismatch_roles": ";".join(mismatch_roles),
                "result_artifact_path": row["result_artifact_path"],
                "declared_result_sha256": row["declared_result_sha256"],
                "observed_result_sha256": row["observed_result_sha256"],
                "uncertainty_artifact_path": row["uncertainty_artifact_path"],
                "declared_uncertainty_sha256": row["declared_uncertainty_sha256"],
                "observed_uncertainty_sha256": row["observed_uncertainty_sha256"],
                "tool_family_total_count": tool_totals[tool_uid],
                "tool_family_non_drift_count": non_drift_count,
                "mandatory_tool_family_coverage": "UNRESOLVED_NO_FROZEN_TOOL_FAMILY_REQUIREMENT",
                "fixed_budget_comparability": "UNRESOLVED_STATIC_C3_NOT_FROZEN",
                "fixed_risk_comparability": "UNRESOLVED_STATIC_C3_NOT_FROZEN",
                "tool_family_holdout": "UNRESOLVED_STATIC_C3_NOT_FROZEN",
                "real_round_protocol": "not_retroactively_eligible",
                "requirement_resolution_status": "blocked_round21_authority_and_static_c3_contract",
                "disposition": "preserve_excluded_no_redispatch_authorized",
                "new_request_uid": "",
                "new_result_uid": "",
            }
        )

    real_rows: list[dict[str, object]] = []
    for row in rows:
        raw = row["__raw"]
        chronology = row["__chronology"]
        chronology_reasons = row["__chronology_reasons"]
        formal_execution = row["formal_round22_execution"] == "true"
        hidden_explicit_false = _strict_bool(raw.get("hidden_outcome_isolated")) is False
        if not row["result_uid"]:
            eligibility_class = "invalid"
            reasons = ["result_uid_missing"]
        elif hidden_explicit_false:
            eligibility_class = "hidden_outcome_exposed"
            reasons = ["hidden_outcome_isolated_false"]
        elif (
            row["source_role"] != "round22_formal_tool_outcome"
            or not formal_execution
            or row["formal_round_id"] != contract["round_id"]
        ):
            eligibility_class = "reused_without_round_identity"
            reasons = ["no_prospective_round22_execution_identity"]
        elif chronology_reasons:
            eligibility_class = "not_sequential"
            reasons = chronology_reasons
        elif row["admission_state"] == "formal_eligible_real_round":
            eligibility_class = "real_round_eligible"
            reasons = []
        else:
            eligibility_class = "invalid"
            reasons = [f"admission_state:{row['admission_state']}"]
        real_rows.append(
            {
                "result_uid": row["result_uid"],
                "source_manifest": row["source_manifest"],
                "source_role": row["source_role"],
                "round_uid": row["round_uid"],
                "formal_round_index": row["formal_round_index"],
                **{key: str(value).lower() for key, value in chronology.items()},
                "eligibility_class": eligibility_class,
                "reasons": ";".join(reasons),
            }
        )

    serializable_rows = [{key: row.get(key, "") for key in ADMISSION_FIELDS} for row in rows]
    return {
        "rows": serializable_rows,
        "repair_rows": repair_rows,
        "real_rows": real_rows,
        "source_bindings": source_bindings,
        "state_counts": dict(sorted(Counter(str(row["admission_state"]) for row in rows).items())),
        "real_round_class_counts": dict(sorted(Counter(str(row["eligibility_class"]) for row in real_rows).items())),
        "tool_counts": dict(sorted(tool_totals.items())),
        "hash_cache": hash_cache,
        "hash_readback_host": socket.gethostname(),
        "hash_readback_performed": allow_hash_readback,
    }


def summarize_real_round_requirement(
    contract: Mapping[str, Any],
    real_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    eligible_rows = [
        row for row in real_rows if row.get("eligibility_class") == "real_round_eligible"
    ]
    round_counts = Counter(str(row.get("formal_round_index", "")) for row in eligible_rows)
    expected_round_counts = {
        str(index): int(contract["c3_static"]["tasks_per_round"])
        for index in range(1, int(contract["c3_static"]["rounds"]) + 1)
    }
    expected_total = int(contract["c3_static"]["rounds"]) * int(
        contract["c3_static"]["tasks_per_round"]
    )
    unique_result_count = len({str(row.get("result_uid", "")) for row in eligible_rows})
    complete = (
        len(eligible_rows) == expected_total
        and unique_result_count == expected_total
        and dict(sorted(round_counts.items())) == expected_round_counts
    )
    return {
        "expected_total": expected_total,
        "eligible_total": len(eligible_rows),
        "unique_result_count": unique_result_count,
        "expected_per_round": expected_round_counts,
        "observed_per_round": dict(sorted(round_counts.items())),
        "complete": complete,
        "prospective_static_rounds_required": not complete,
    }


def _tsv_bytes(fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return buffer.getvalue().encode("utf-8")


def _write_immutable_tsv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]) -> None:
    content = _tsv_bytes(fieldnames, rows)
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(f"Immutable Round-22R artifact already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def build(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
    *,
    allow_hash_readback: bool | None = None,
    write_outputs: bool = True,
) -> dict[str, object]:
    contract = load_contract(config_path, schema_path)
    authority = inspect_authority_inputs(contract, root)
    components, cross_rows = build_freeze_component_report(contract, authority, root)
    admission = build_tool_admission(contract, root, allow_hash_readback=allow_hash_readback)
    authority_passed = authority["status"] == "passed"
    freeze_passed = components["status"] == "passed" and components["static_twin_freeze_allowed"] is True
    offline_count = int(admission["state_counts"].get("formal_eligible_offline", 0))
    real_round_requirement = summarize_real_round_requirement(contract, admission["real_rows"])
    real_count = int(real_round_requirement["eligible_total"])
    c2_b_path = str(contract["claim_boundary"]["c2_b_recovery_artifact"])
    c2_b_binding = _candidate_binding(c2_b_path, root)
    c2_b_payload, c2_b_error = _json_object(_path(c2_b_path, root))
    qualified_domain_count = None
    if isinstance(c2_b_payload, Mapping):
        value = c2_b_payload.get("qualified_action_complete_domain_count")
        if not isinstance(value, bool):
            try:
                qualified_domain_count = int(value)
            except (TypeError, ValueError):
                qualified_domain_count = None
    c2_b_invariant_verified = bool(
        not c2_b_error
        and isinstance(c2_b_payload, Mapping)
        and c2_b_payload.get("c2_b_status") == C2B_BLOCKED
        and qualified_domain_count == 0
        and c2_b_payload.get("c2_s_mutated") is False
    )
    blockers: list[str] = []
    if not authority_passed:
        blockers.append("blocked_round21_authority")
    if not freeze_passed:
        blockers.append("blocked_static_twin_freeze")
    if offline_count == 0:
        blockers.append("blocked_formal_offline_tool_outcomes")
    if not c2_b_invariant_verified:
        blockers.append("blocked_c2_b_invariant")
    gate = {
        "schema_version": "neuronsifter-round22r-empirical-prerequisite-gate-v1",
        "gate": "round22r_empirical_prerequisites_resolved",
        "status": "passed" if not blockers else "blocked_empirical_prerequisite_recovery",
        "passed": not blockers,
        "round21_authority": authority["round21_authority_inputs"],
        "static_twin_freeze_components": {
            "present": components["present_count"],
            "validated": components["validated_count"],
            "identity_conflicts": components["identity_conflict_count"],
        },
        "tool_outcomes": {
            "source_rows": len(admission["rows"]),
            "formal_eligible_offline": offline_count,
            "formal_eligible_real_round": int(admission["state_counts"].get("formal_eligible_real_round", 0)),
            "hash_drift_rows": len(admission["repair_rows"]),
            "source_manifest_bindings": admission["source_bindings"],
        },
        "real_round_eligible_count": real_count,
        "real_round_requirement": real_round_requirement,
        "exact_clean_full_suite_authorized": authority_passed and freeze_passed,
        "formal_c3_c4_execution_allowed": False,
        "formal_c3_c4_execution_blocker": "exact_clean_full_suite_not_yet_bound",
        "formal_coevolution_execution_allowed": False,
        "c2_b_status": C2B_BLOCKED,
        "c2_b_recovery_artifact": c2_b_binding,
        "c2_b_invariant_verified": c2_b_invariant_verified,
        "blockers": blockers,
        "next_action": "provide_or_reconstruct_round21_authority_from_explicit_immutable_machine_readable_sources"
        if not authority_passed
        else "resolve_formal_tool_outcome_admission_then_run_exact_clean_validation",
    }
    reproducibility = {
        "schema_version": "neuronsifter-round22r-reproducibility-commands-v1",
        "status": "control_plane_commands_frozen",
        "commands": [
            "python Libs/scripts/builders/build_round22r_empirical_prerequisites.py",
            "python Libs/scripts/validators/validate_round22r_empirical_prerequisites.py",
            "python -m pytest -p no:cacheprovider Libs/tests/test_round22r_empirical_prerequisites.py -q",
        ],
        "formal_execution_commands": [],
    }
    resource = {
        "schema_version": "neuronsifter-round22r-resource-catalog-v1",
        "status": "prerequisite_recovery_only",
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hash_readback_performed": admission["hash_readback_performed"],
        "unique_hashed_artifact_count": len(admission["hash_cache"]),
        "formal_job_count": 0,
        "gpu_hours": 0.0,
        "c2_b_status": C2B_BLOCKED,
    }
    summary = {
        "status": gate["status"],
        "authority": authority,
        "components": components,
        "admission_state_counts": admission["state_counts"],
        "hash_drift_count": len(admission["repair_rows"]),
        "real_round_class_counts": admission["real_round_class_counts"],
        "gate": gate,
    }
    if write_outputs:
        write_json(_path(OUTPUTS["authority"], root), authority)
        write_json(_path(OUTPUTS["components"], root), components)
        _write_immutable_tsv(_path(OUTPUTS["cross_identity"], root), CROSS_IDENTITY_FIELDS, cross_rows)
        _write_immutable_tsv(_path(OUTPUTS["admission"], root), ADMISSION_FIELDS, admission["rows"])
        _write_immutable_tsv(_path(OUTPUTS["hash_drift"], root), HASH_DRIFT_FIELDS, admission["repair_rows"])
        _write_immutable_tsv(_path(OUTPUTS["real_round"], root), REAL_ROUND_FIELDS, admission["real_rows"])
        write_json(_path(OUTPUTS["gate"], root), gate)
        write_json(_path(OUTPUTS["resource"], root), resource)
        write_json(_path(OUTPUTS["reproducibility"], root), reproducibility)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Round-22R authority-recovery and tool-admission audits.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if socket.gethostname() == "mgmt2":
        raise SystemExit("Round-22R tool-artifact byte hashing must run on an allowed compute node.")
    result = build(args.config, args.schema, allow_hash_readback=True)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
