"""Validate Round-22R authority recovery and tool-admission artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
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
    FREEZE_COMPONENT_SPECS,
    load_contract,
    sha256_file,
)
from Libs.scripts.builders.build_round22r_empirical_prerequisites import (  # noqa: E402
    ADMISSION_FIELDS,
    C2B_BLOCKED,
    CROSS_IDENTITY_FIELDS,
    FORMAL_STATES,
    HASH_DRIFT_FIELDS,
    OUTPUTS,
    REAL_ROUND_CLASSES,
    REAL_ROUND_FIELDS,
    REQUIRED_AUTHORITY_INPUTS,
)


ALLOWED_COMPUTE_NODES = {"cpu03", "cpu04", "gpu03"}
BOOLEAN_CELLS = {
    "result_hash_matches",
    "uncertainty_hash_matches",
    "artifacts_distinct",
    "gate_o_passed",
    "compute_node_verified",
    "formal_round22_execution",
}
CHRONOLOGY_CELLS = {
    "prospective_dispatch_identity",
    "round_index_frozen_before_reveal",
    "candidate_pool_frozen_before_reveal",
    "policy_decision_before_outcome_visibility",
    "budget_state_before_dispatch",
    "tool_availability_state",
    "hidden_outcome_isolation",
    "actual_execution_chronology",
    "gate_o_admission_chronology",
}


def _root_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _strict_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _strict_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _nonnegative_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed >= 0


def _sha(value: object) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    return None


def _block(blockers: list[dict[str, str]], code: str, message: str) -> None:
    blockers.append({"code": code, "message": message})


def _validate_contract(contract: Mapping[str, Any], blockers: list[dict[str, str]]) -> None:
    items = contract["freeze"]["required_inputs"]
    pairs = [(str(item["input_id"]), str(item["path"])) for item in items]
    if len(pairs) != len(REQUIRED_AUTHORITY_INPUTS) or len({key for key, _ in pairs}) != len(pairs):
        _block(blockers, "authority_contract_cardinality_drift", "Round-21 authority inputs must be nine unique records.")
    if dict(pairs) != REQUIRED_AUTHORITY_INPUTS:
        _block(blockers, "authority_contract_identity_drift", "Round-21 authority IDs and paths drifted from the canonical nine-input contract.")
    if contract["claim_boundary"]["c2_b_status"] != C2B_BLOCKED:
        _block(blockers, "c2_b_contract_drift", "The Round-22 contract changed the frozen C2-B blocker.")


def _validate_authority(
    authority: Mapping[str, Any],
    root: Path,
    blockers: list[dict[str, str]],
) -> bool:
    entries = authority.get("entries")
    if not isinstance(entries, list):
        _block(blockers, "authority_entries_missing", "Authority materialization entries are missing.")
        return False
    ids = [str(row.get("input_id", "")) for row in entries if isinstance(row, Mapping)]
    if len(entries) != 9 or len(ids) != len(set(ids)) or set(ids) != set(REQUIRED_AUTHORITY_INPUTS):
        _block(blockers, "authority_entry_identity_invalid", "Authority report must contain the canonical nine inputs exactly once.")
    actual_present = actual_validated = actual_hash_bound = 0
    for row in entries:
        if not isinstance(row, Mapping):
            _block(blockers, "authority_entry_invalid", "Authority entry is not an object.")
            continue
        input_id = str(row.get("input_id", ""))
        expected_path = REQUIRED_AUTHORITY_INPUTS.get(input_id)
        if expected_path is None or row.get("path") != expected_path:
            _block(blockers, "authority_path_mismatch", f"Authority path drifted for {input_id or 'unknown'}.")
            continue
        path = _root_path(expected_path, root)
        exists = path.is_file()
        if row.get("exists") is not exists:
            _block(blockers, "authority_presence_mismatch", f"Reported presence is stale for {input_id}.")
        if exists:
            actual_present += 1
            if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
                _block(blockers, "authority_hash_binding_mismatch", f"Authority bytes or hash drifted for {input_id}.")
        elif row.get("bytes") != 0 or row.get("sha256") is not None:
            _block(blockers, "missing_authority_has_binding", f"Missing authority {input_id} carries a false byte/hash binding.")
        validated = row.get("validated")
        hash_bound = row.get("hash_bound")
        if not isinstance(validated, bool) or not isinstance(hash_bound, bool):
            _block(blockers, "authority_boolean_invalid", f"Authority validation flags are not strict booleans for {input_id}.")
            continue
        actual_validated += int(validated)
        actual_hash_bound += int(hash_bound)
        if (validated or hash_bound) and not exists:
            _block(blockers, "missing_authority_promoted", f"Missing authority {input_id} was promoted.")
        errors = row.get("validation_errors")
        if not isinstance(errors, list) or (validated == bool(errors)):
            _block(blockers, "authority_validation_error_state_invalid", f"Validation errors disagree with state for {input_id}.")
    counts = authority.get("round21_authority_inputs")
    expected_counts = {
        "required": 9,
        "present": actual_present,
        "validated": actual_validated,
        "hash_bound": actual_hash_bound,
    }
    if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
        _block(blockers, "authority_count_mismatch", "Authority aggregate counts do not match the nine entries.")
    expected_pass = actual_present == actual_validated == actual_hash_bound == 9
    if (authority.get("status") == "passed") is not expected_pass:
        _block(blockers, "authority_status_mismatch", "Authority status disagrees with validated bindings.")
    for key in ("retraining_used_for_recovery", "narrative_fields_used_for_recovery", "historical_manifests_rewritten"):
        if authority.get(key) is not False:
            _block(blockers, f"authority_{key}", f"Authority recovery invariant {key}=false was violated.")
    return expected_pass


def _validate_components(
    components: Mapping[str, Any],
    cross_rows: list[dict[str, str]],
    authority_passed: bool,
    blockers: list[dict[str, str]],
) -> bool:
    rows = components.get("components")
    expected_names = {"checkpoint_hash", *FREEZE_COMPONENT_SPECS}
    if not isinstance(rows, list):
        _block(blockers, "freeze_components_missing", "Static Twin freeze component rows are missing.")
        return False
    names = [str(row.get("component", "")) for row in rows if isinstance(row, Mapping)]
    if len(rows) != 10 or len(names) != len(set(names)) or set(names) != expected_names:
        _block(blockers, "freeze_component_identity_drift", "Static Twin freeze must contain the canonical ten components exactly once.")
    present = validated = 0
    component_conflicts = 0
    for row in rows:
        if not isinstance(row, Mapping):
            _block(blockers, "freeze_component_row_invalid", "Freeze component row is not an object.")
            continue
        if not all(isinstance(row.get(key), bool) for key in ("present", "validated", "conflict")):
            _block(blockers, "freeze_component_boolean_invalid", "Freeze component flags must be strict booleans.")
            continue
        present += int(row["present"])
        validated += int(row["validated"] and not row["conflict"])
        component_conflicts += int(row["conflict"])
        if row["present"] != (_sha(row.get("value")) is not None):
            _block(blockers, "freeze_component_presence_mismatch", f"Freeze component value is malformed for {row.get('component')}.")
    if components.get("component_count") != 10 or components.get("present_count") != present or components.get("validated_count") != validated:
        _block(blockers, "freeze_component_count_mismatch", "Freeze component aggregate counts are inconsistent.")
    conflict_count = sum(row.get("status") == "conflict" for row in cross_rows)
    missing_count = sum(row.get("status") == "missing" for row in cross_rows)
    if components.get("identity_conflict_count") != component_conflicts + conflict_count or components.get("cross_identity_missing_count") != missing_count:
        _block(blockers, "cross_identity_count_mismatch", "Cross-artifact identity counts are inconsistent.")
    if components.get("test_selected_checkpoint") is not False:
        _block(blockers, "test_checkpoint_selected", "A test or final-holdout checkpoint was selected.")
    allowed = components.get("static_twin_freeze_allowed")
    if not isinstance(allowed, bool):
        _block(blockers, "static_freeze_allowed_boolean_invalid", "Static Twin freeze authorization must be a strict boolean.")
        return False
    expected_allowed = authority_passed and present == validated == 10 and conflict_count == missing_count == 0
    if allowed is not expected_allowed or (components.get("status") == "passed") is not expected_allowed:
        _block(blockers, "static_freeze_state_mismatch", "Static Twin freeze status disagrees with authority and identity evidence.")
    if expected_allowed and components.get("checkpoint_hash_verified") is not True:
        _block(blockers, "checkpoint_hash_not_verified", "Passed static Twin freeze lacks checkpoint byte verification.")
    return expected_allowed


def _validate_cross_rows(fields: list[str], rows: list[dict[str, str]], blockers: list[dict[str, str]]) -> None:
    if tuple(fields) != CROSS_IDENTITY_FIELDS:
        _block(blockers, "cross_identity_schema_drift", "Cross-artifact identity TSV schema drifted.")
    if not rows:
        _block(blockers, "cross_identity_rows_missing", "Cross-artifact identity checks are absent.")
    for row in rows:
        if row.get("status") not in {"matched", "missing", "conflict"}:
            _block(blockers, "cross_identity_status_invalid", "Cross-artifact identity row has an invalid status.")
            break
        if row.get("status") == "matched" and (_sha(row.get("observed_value")) is None or row.get("observed_value") != row.get("expected_value")):
            _block(blockers, "cross_identity_false_match", "Cross-artifact identity row claims a malformed match.")
            break


def _validate_admission_rows(
    contract: Mapping[str, Any],
    fields: list[str],
    rows: list[dict[str, str]],
    root: Path,
    allow_hash_readback: bool,
    blockers: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if tuple(fields) != ADMISSION_FIELDS:
        _block(blockers, "admission_schema_drift", "Tool-admission TSV schema drifted.")
    source_roles = {str(item["path"]): str(item["role"]) for item in contract["tool_outcomes"]["source_manifests"]}
    source_hashes: dict[str, str] = {}
    expected_source_rows: set[tuple[str, str]] = set()
    for source_path in source_roles:
        path = _root_path(source_path, root)
        if not path.is_file():
            _block(blockers, "source_manifest_missing", f"Configured source manifest is missing: {source_path}.")
            continue
        source_hashes[source_path] = sha256_file(path)
        _, source_rows = _read_tsv(path)
        expected_source_rows.update((source_path, str(index)) for index in range(2, len(source_rows) + 2))
    observed_source_rows: set[tuple[str, str]] = set()
    admission_uids: set[str] = set()
    hash_cache: dict[str, str] = {}
    for row in rows:
        admission_uid = row.get("admission_uid", "")
        if not admission_uid or admission_uid in admission_uids:
            _block(blockers, "admission_uid_duplicate_or_missing", "Admission UIDs must be nonempty and unique.")
        admission_uids.add(admission_uid)
        source = row.get("source_manifest", "")
        source_key = (source, row.get("source_row_number", ""))
        if source_key in observed_source_rows:
            _block(blockers, "source_row_duplicate", "A source manifest row was admitted more than once.")
        observed_source_rows.add(source_key)
        if source not in source_roles or row.get("source_role") != source_roles.get(source):
            _block(blockers, "source_role_or_path_mismatch", "Admission row source role is not bound to the YAML manifest registry.")
        if row.get("source_manifest_sha256") != source_hashes.get(source):
            _block(blockers, "source_manifest_hash_mismatch", "Admission row source-manifest hash is stale or missing.")
        state = row.get("admission_state", "")
        if state not in FORMAL_STATES:
            _block(blockers, "admission_state_invalid", "Admission row uses an unknown state.")
            continue
        if any(_strict_bool(row.get(key)) is None for key in BOOLEAN_CELLS):
            _block(blockers, "admission_boolean_invalid", "Admission row contains a malformed boolean cell.")
        result_path = row.get("result_artifact_path", "")
        uncertainty_path = row.get("uncertainty_artifact_path", "")
        for role, path_value in (("result", result_path), ("uncertainty", uncertainty_path)):
            path = _root_path(path_value, root) if path_value else None
            if path is None or not path.is_file():
                continue
            key = str(path.resolve())
            if key not in hash_cache and allow_hash_readback:
                hash_cache[key] = sha256_file(path)
            observed = row.get(f"observed_{role}_sha256", "")
            if allow_hash_readback and observed != hash_cache.get(key):
                _block(blockers, f"{role}_artifact_hash_mismatch", f"Observed {role} hash no longer matches artifact bytes.")
        if state == "ineligible_hash_drift":
            result_drift = _sha(row.get("declared_result_sha256")) is not None and row.get("declared_result_sha256") != row.get("observed_result_sha256")
            uncertainty_drift = _sha(row.get("declared_uncertainty_sha256")) is not None and row.get("declared_uncertainty_sha256") != row.get("observed_uncertainty_sha256")
            if not (result_drift or uncertainty_drift):
                _block(blockers, "false_hash_drift_state", "Hash-drift state lacks a declared-vs-observed mismatch.")
        eligible = state in {"formal_eligible_offline", "formal_eligible_real_round"}
        if not eligible:
            if not row.get("eligibility_reasons"):
                _block(blockers, "ineligible_row_without_reason", "Ineligible admission row lacks a failure reason.")
            continue
        if row.get("eligibility_reasons"):
            _block(blockers, "eligible_row_has_reasons", "Formal eligible row retains ineligibility reasons.")
        if row.get("source_role") != "round22_formal_tool_outcome":
            _block(blockers, "legacy_source_promoted", "A legacy source row was promoted to formal evidence.")
        if row.get("c2_b_status") != C2B_BLOCKED:
            _block(blockers, "formal_row_c2_b_drift", "Formal tool row changed the C2-B blocker.")
        if row.get("execution_node") not in ALLOWED_COMPUTE_NODES or _strict_bool(row.get("compute_node_verified")) is not True:
            _block(blockers, "formal_compute_node_unverified", "Formal tool row lacks allowed compute-node verification.")
        if not row.get("train_eval_role") or not row.get("hidden_evaluation_role"):
            _block(blockers, "formal_role_metadata_missing", "Formal tool row lacks train/eval or hidden-outcome role metadata.")
        if not row.get("hash_readback_uid"):
            _block(blockers, "formal_hash_readback_uid_missing", "Formal tool row lacks adapter hash-readback identity.")
        if not all(_nonnegative_finite(row.get(key)) for key in ("runtime_seconds", "compute_cost", "wall_clock_cost")):
            _block(blockers, "formal_cost_runtime_invalid", "Formal runtime and costs must be finite and non-negative.")
        if _strict_bool(row.get("result_hash_matches")) is not True or _strict_bool(row.get("uncertainty_hash_matches")) is not True:
            _block(blockers, "formal_hash_not_matched", "Formal result and uncertainty hashes must match compute-node readback.")
        if _strict_bool(row.get("artifacts_distinct")) is not True or _root_path(result_path, root).resolve() == _root_path(uncertainty_path, root).resolve():
            _block(blockers, "formal_artifact_roles_not_distinct", "Formal result and uncertainty artifacts must be distinct.")
        is_real = state == "formal_eligible_real_round"
        if (_strict_bool(row.get("formal_round22_execution")) is True) is not is_real:
            _block(blockers, "formal_execution_state_mismatch", "Formal execution marker disagrees with admission state.")
        if is_real:
            if row.get("formal_round_id") != contract["round_id"] or row.get("formal_round_index") not in {"1", "2", "3"}:
                _block(blockers, "formal_round_identity_invalid", "Real-round row lacks the frozen Round-22 round identity/index.")
    if observed_source_rows != expected_source_rows:
        _block(blockers, "source_row_membership_mismatch", "Admission catalog does not map every configured source row exactly once.")
    return rows, source_hashes


def _validate_drift_rows(
    fields: list[str],
    rows: list[dict[str, str]],
    admission_rows: list[dict[str, str]],
    blockers: list[dict[str, str]],
) -> None:
    if tuple(fields) != HASH_DRIFT_FIELDS:
        _block(blockers, "hash_drift_schema_drift", "Hash-drift repair catalog schema drifted.")
    expected = {
        (row["source_manifest"], row["result_uid"])
        for row in admission_rows
        if row["admission_state"] == "ineligible_hash_drift"
    }
    observed = {(row.get("source_manifest", ""), row.get("parent_result_uid", "")) for row in rows}
    if expected != observed or len(rows) != len(observed):
        _block(blockers, "hash_drift_membership_mismatch", "Hash-drift catalog is not one-to-one with drift admission rows.")
    if any(not row.get("mismatch_roles") for row in rows):
        _block(blockers, "hash_drift_role_missing", "Hash-drift row does not identify the mismatched artifact role.")


def _validate_real_round_rows(
    contract: Mapping[str, Any],
    fields: list[str],
    rows: list[dict[str, str]],
    admission_rows: list[dict[str, str]],
    gate: Mapping[str, Any],
    blockers: list[dict[str, str]],
) -> None:
    if tuple(fields) != REAL_ROUND_FIELDS:
        _block(blockers, "real_round_schema_drift", "Real-round eligibility TSV schema drifted.")
    admission_map = {(row["source_manifest"], row["result_uid"]): row for row in admission_rows}
    keys = [(row.get("source_manifest", ""), row.get("result_uid", "")) for row in rows]
    if len(rows) != len(admission_rows) or len(keys) != len(set(keys)) or set(keys) != set(admission_map):
        _block(blockers, "real_round_membership_mismatch", "Real-round audit must map every admission row exactly once.")
    eligible_rows: list[dict[str, str]] = []
    for row in rows:
        classification = row.get("eligibility_class", "")
        if classification not in REAL_ROUND_CLASSES:
            _block(blockers, "real_round_class_invalid", "Real-round audit contains an unknown class.")
            continue
        if any(_strict_bool(row.get(key)) is None for key in CHRONOLOGY_CELLS):
            _block(blockers, "chronology_boolean_invalid", "Real-round chronology contains malformed booleans.")
        admission = admission_map.get((row.get("source_manifest", ""), row.get("result_uid", "")))
        if classification == "real_round_eligible":
            eligible_rows.append(row)
            if admission is None or admission.get("admission_state") != "formal_eligible_real_round":
                _block(blockers, "real_round_without_formal_admission", "Real-round eligibility lacks formal admission.")
            if row.get("source_role") != "round22_formal_tool_outcome" or not all(
                _strict_bool(row.get(key)) is True for key in CHRONOLOGY_CELLS
            ):
                _block(blockers, "retrospective_real_round_promotion", "Real-round row lacks formal source role or complete chronology.")
    expected_total = int(contract["c3_static"]["rounds"]) * int(contract["c3_static"]["tasks_per_round"])
    expected_counts = {str(index): int(contract["c3_static"]["tasks_per_round"]) for index in range(1, 4)}
    observed_counts = dict(sorted(Counter(row.get("formal_round_index", "") for row in eligible_rows).items()))
    unique_results = len({row.get("result_uid", "") for row in eligible_rows})
    complete = len(eligible_rows) == expected_total and unique_results == expected_total and observed_counts == expected_counts
    requirement = gate.get("real_round_requirement")
    if not isinstance(requirement, Mapping):
        _block(blockers, "real_round_requirement_missing", "Gate lacks the 3 x 128 real-round summary.")
        return
    expected_summary = {
        "expected_total": expected_total,
        "eligible_total": len(eligible_rows),
        "unique_result_count": unique_results,
        "expected_per_round": expected_counts,
        "observed_per_round": observed_counts,
        "complete": complete,
        "prospective_static_rounds_required": not complete,
    }
    if dict(requirement) != expected_summary:
        _block(blockers, "real_round_requirement_mismatch", "Gate real-round counts do not match the row-level audit.")


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    *,
    root: Path = REPO_ROOT,
    allow_hash_readback: bool | None = None,
) -> dict[str, object]:
    contract = load_contract(config_path, schema_path)
    blockers: list[dict[str, str]] = []
    if allow_hash_readback is None:
        allow_hash_readback = socket.gethostname() != "mgmt2"
    _validate_contract(contract, blockers)
    try:
        authority = _read_json(_root_path(OUTPUTS["authority"], root))
        components = _read_json(_root_path(OUTPUTS["components"], root))
        gate = _read_json(_root_path(OUTPUTS["gate"], root))
        resource = _read_json(_root_path(OUTPUTS["resource"], root))
        reproducibility = _read_json(_root_path(OUTPUTS["reproducibility"], root))
        cross_fields, cross_rows = _read_tsv(_root_path(OUTPUTS["cross_identity"], root))
        admission_fields, admission_rows = _read_tsv(_root_path(OUTPUTS["admission"], root))
        drift_fields, drift_rows = _read_tsv(_root_path(OUTPUTS["hash_drift"], root))
        real_fields, real_rows = _read_tsv(_root_path(OUTPUTS["real_round"], root))
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError, UnicodeError, csv.Error) as exc:
        return {
            "status": "failed",
            "blocker_count": 1,
            "blockers": [{"code": "round22r_artifact_read_failure", "message": str(exc)}],
            "prerequisite_gate_passed": False,
            "formal_c3_c4_execution_allowed": False,
            "c2_b_status": C2B_BLOCKED,
        }
    authority_passed = _validate_authority(authority, root, blockers)
    _validate_cross_rows(cross_fields, cross_rows, blockers)
    freeze_passed = _validate_components(components, cross_rows, authority_passed, blockers)
    admission_rows, source_hashes = _validate_admission_rows(
        contract,
        admission_fields,
        admission_rows,
        root,
        allow_hash_readback,
        blockers,
    )
    _validate_drift_rows(drift_fields, drift_rows, admission_rows, blockers)
    _validate_real_round_rows(contract, real_fields, real_rows, admission_rows, gate, blockers)

    state_counts = dict(sorted(Counter(row["admission_state"] for row in admission_rows).items()))
    formal_offline = state_counts.get("formal_eligible_offline", 0)
    if not isinstance(gate.get("passed"), bool) or (gate.get("status") == "passed") is not gate.get("passed"):
        _block(blockers, "prerequisite_gate_boolean_invalid", "Prerequisite Gate status and strict passed boolean disagree.")
    if gate.get("round21_authority") != authority.get("round21_authority_inputs"):
        _block(blockers, "gate_authority_summary_mismatch", "Prerequisite Gate authority counts drifted from the materialization report.")
    expected_freeze_summary = {
        "present": components.get("present_count"),
        "validated": components.get("validated_count"),
        "identity_conflicts": components.get("identity_conflict_count"),
    }
    if gate.get("static_twin_freeze_components") != expected_freeze_summary:
        _block(blockers, "gate_freeze_summary_mismatch", "Prerequisite Gate freeze counts drifted from the component report.")
    gate_tool = gate.get("tool_outcomes")
    if not isinstance(gate_tool, Mapping):
        _block(blockers, "gate_tool_summary_missing", "Prerequisite Gate lacks tool-outcome counts.")
    else:
        expected_tool_counts = {
            "source_rows": len(admission_rows),
            "formal_eligible_offline": formal_offline,
            "formal_eligible_real_round": state_counts.get("formal_eligible_real_round", 0),
            "hash_drift_rows": len(drift_rows),
        }
        if any(gate_tool.get(key) != value for key, value in expected_tool_counts.items()):
            _block(blockers, "gate_tool_count_mismatch", "Prerequisite Gate tool counts do not match admission rows.")
        bindings = gate_tool.get("source_manifest_bindings")
        if not isinstance(bindings, list):
            _block(blockers, "source_manifest_bindings_missing", "Prerequisite Gate lacks source-manifest bindings.")
        else:
            bound = {str(item.get("path")): item for item in bindings if isinstance(item, Mapping)}
            for path, observed_hash in source_hashes.items():
                if path not in bound or bound[path].get("sha256") != observed_hash:
                    _block(blockers, "source_binding_mismatch", f"Prerequisite Gate source binding drifted for {path}.")
    if gate.get("formal_c3_c4_execution_allowed") is not False or gate.get("formal_coevolution_execution_allowed") is not False:
        _block(blockers, "premature_empirical_execution_authorization", "Prerequisite recovery cannot authorize C3/C4 or Co-Evolution before exact-clean validation.")
    if gate.get("exact_clean_full_suite_authorized") is not (authority_passed and freeze_passed):
        _block(blockers, "full_suite_authorization_mismatch", "Exact-clean full-suite authorization disagrees with authority and freeze evidence.")
    real_eligible_count = sum(row.get("eligibility_class") == "real_round_eligible" for row in real_rows)
    if gate.get("real_round_eligible_count") != real_eligible_count:
        _block(blockers, "real_round_scalar_count_mismatch", "Gate real-round count disagrees with the eligibility table.")

    c2_b_path = str(contract["claim_boundary"]["c2_b_recovery_artifact"])
    try:
        c2_b = _read_json(_root_path(c2_b_path, root))
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError, UnicodeError) as exc:
        _block(blockers, "c2_b_recovery_artifact_invalid", str(exc))
        c2_b = {}
    qualified_count = _strict_nonnegative_int(c2_b.get("qualified_action_complete_domain_count"))
    c2_b_invariant_verified = bool(
        c2_b.get("c2_b_status") == C2B_BLOCKED
        and qualified_count == 0
        and c2_b.get("c2_s_mutated") is False
    )
    if gate.get("c2_b_status") != C2B_BLOCKED or gate.get("c2_b_invariant_verified") is not True or not c2_b_invariant_verified:
        _block(blockers, "c2_b_invariant_drift", "C2-B recovery or Gate binding changed the frozen blocked state.")
    c2_b_binding = gate.get("c2_b_recovery_artifact")
    c2_b_file = _root_path(c2_b_path, root)
    if not isinstance(c2_b_binding, Mapping) or c2_b_binding.get("path") != c2_b_path or c2_b_binding.get("sha256") != (sha256_file(c2_b_file) if c2_b_file.is_file() else None):
        _block(blockers, "c2_b_binding_mismatch", "Prerequisite Gate does not hash-bind the configured C2-B recovery artifact.")

    expected_gate_blockers: list[str] = []
    if not authority_passed:
        expected_gate_blockers.append("blocked_round21_authority")
    if not freeze_passed:
        expected_gate_blockers.append("blocked_static_twin_freeze")
    if formal_offline == 0:
        expected_gate_blockers.append("blocked_formal_offline_tool_outcomes")
    if not c2_b_invariant_verified:
        expected_gate_blockers.append("blocked_c2_b_invariant")
    expected_gate_pass = not expected_gate_blockers
    if gate.get("passed") is not expected_gate_pass:
        _block(blockers, "prerequisite_gate_state_mismatch", "Prerequisite Gate disagrees with authority, freeze, admission, or C2-B evidence.")
    if gate.get("blockers") != expected_gate_blockers:
        _block(blockers, "prerequisite_gate_blocker_mismatch", "Prerequisite Gate blocker list does not match independently recomputed blockers.")

    if resource.get("hash_readback_performed") is not True or resource.get("hostname") not in ALLOWED_COMPUTE_NODES or not resource.get("slurm_job_id"):
        _block(blockers, "resource_compute_node_binding_invalid", "Resource catalog lacks Slurm compute-node hash-readback evidence.")
    if resource.get("c2_b_status") != C2B_BLOCKED or resource.get("formal_job_count") != 0:
        _block(blockers, "resource_claim_drift", "Prerequisite resource catalog claims formal jobs or changes C2-B.")
    if reproducibility.get("formal_execution_commands") != []:
        _block(blockers, "formal_commands_published_while_blocked", "Formal execution commands must remain absent during prerequisite recovery.")

    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "prerequisite_gate_passed": gate.get("passed") is True,
        "formal_c3_c4_execution_allowed": gate.get("formal_c3_c4_execution_allowed") is True,
        "round21_authority_present": authority.get("round21_authority_inputs", {}).get("present"),
        "round21_authority_validated": authority.get("round21_authority_inputs", {}).get("validated"),
        "freeze_components_validated": components.get("validated_count"),
        "admission_state_counts": state_counts,
        "hash_drift_row_count": len(drift_rows),
        "real_round_class_counts": dict(sorted(Counter(row["eligibility_class"] for row in real_rows).items())),
        "c2_b_status": gate.get("c2_b_status"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Round-22R empirical prerequisites without rebuilding them.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if socket.gethostname() == "mgmt2":
        raise SystemExit("Round-22R artifact-byte validation must run on an allowed compute node.")
    result = validate(args.config, args.schema, allow_hash_readback=True)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
