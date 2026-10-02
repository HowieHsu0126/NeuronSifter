"""Validate Round-22 readiness artifacts without promoting blocked gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    BASELINES,
    C4_VARIANTS,
    MATRIX,
    TOOL_OUTCOME_FIELDS,
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    build,
    load_contract,
    repo_path,
    sha256_file,
)
from Libs.scripts.validators.validate_agent_twin_coevolution import validate as validate_formal_coevolution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Round-22 readiness and fail-closed gate artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--audit", default="Output/audits/round22_readiness_audit.json")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild guarded readiness artifacts before validation (read-only validation is the default).",
    )
    return parser.parse_args()


def read_json(path_value: str | Path) -> dict[str, Any]:
    path = repo_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Round-22 artifact: {path_value}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Round-22 artifact is not a JSON object: {path_value}")
    return payload


def read_tsv(path_value: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = repo_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Round-22 TSV artifact: {path_value}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _gate_passed(payload: Mapping[str, Any], passed_key: str = "passed") -> bool:
    return payload.get(passed_key) is True and payload.get("status") == "passed"


def _validate_gate_state(
    payload: Mapping[str, Any],
    label: str,
    blockers: list[dict[str, str]],
    passed_key: str = "passed",
) -> None:
    passed_value = payload.get(passed_key)
    if not isinstance(passed_value, bool):
        blockers.append(
            {
                "code": f"{label}_passed_invalid",
                "message": f"{label} must expose a strict boolean {passed_key} value.",
            }
        )
        return
    if (payload.get("status") == "passed") != passed_value:
        blockers.append(
            {
                "code": f"{label}_state_inconsistent",
                "message": f"{label} status and {passed_key} disagree.",
            }
        )


def _require_true(payload: Mapping[str, Any], keys: tuple[str, ...], code: str, message: str, blockers: list[dict[str, str]]) -> None:
    if not all(payload.get(key) is True for key in keys):
        blockers.append({"code": code, "message": message})


def _safe_int(
    value: object,
    *,
    code: str,
    message: str,
    blockers: list[dict[str, str]],
    minimum: int = 0,
) -> int | None:
    if isinstance(value, bool):
        blockers.append({"code": code, "message": message})
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        blockers.append({"code": code, "message": message})
        return None
    if parsed < minimum:
        blockers.append({"code": code, "message": message})
        return None
    return parsed


def _finite_cell(row: Mapping[str, object], key: str) -> bool:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value)


def _nonnegative_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed >= 0


def _completed(value: object) -> bool:
    return str(value).strip().lower() in {"passed", "completed", "completed_empirical", "evaluation_only"}


def _validate_file_binding(binding: object, label: str, blockers: list[dict[str, str]]) -> None:
    if not isinstance(binding, Mapping):
        blockers.append({"code": f"{label}_binding_missing", "message": f"{label} must be a path/hash binding."})
        return
    path_value = binding.get("path")
    expected_hash = binding.get("sha256")
    if not isinstance(path_value, str) or not path_value or not isinstance(expected_hash, str):
        blockers.append({"code": f"{label}_binding_missing", "message": f"{label} must provide path and sha256."})
        return
    path = repo_path(path_value)
    if not path.is_file() or sha256_file(path) != expected_hash:
        blockers.append({"code": f"{label}_hash_mismatch", "message": f"{label} is missing or not hash-bound."})


def _validate_c3_outputs(contract: Mapping[str, Any], c3: Mapping[str, Any], blockers: list[dict[str, str]]) -> None:
    result_fields, result_rows = read_tsv(contract["c3_static"]["output_results"])
    expected_result_fields = {
        "baseline_id",
        "evaluation_scope",
        "status",
        "reason",
        *contract["c3_static"]["primary_metrics"],
    }
    expected_ids = set(BASELINES) | {str(contract["c3_static"]["oracle"])}
    if set(result_fields) != expected_result_fields:
        blockers.append({"code": "c3_result_schema_incomplete", "message": "C3 result matrix does not expose the frozen primary metrics."})
    ids = [row.get("baseline_id", "") for row in result_rows]
    if set(ids) != expected_ids or len(ids) != len(expected_ids):
        blockers.append({"code": "c3_result_matrix_incomplete", "message": "C3 results must contain every mandatory baseline and the evaluation-only oracle exactly once."})
    for row in result_rows:
        if not _completed(row.get("status")) or any(
            not _finite_cell(row, str(metric)) for metric in contract["c3_static"]["primary_metrics"]
        ):
            blockers.append({"code": "c3_result_values_incomplete", "message": "Every C3 policy row must retain completed finite primary metrics."})
            break

    curve_specs = (
        ("output_budget_risk", {"baseline_id", "budget", "decision_risk", "status", "reason"}, ("budget", "decision_risk")),
        ("output_fixed_target_cost", {"baseline_id", "target_risk", "total_cost", "status", "reason"}, ("target_risk", "total_cost")),
        ("output_tool_usage", {"baseline_id", "tool_uid", "calls", "realized_cost", "wall_clock_cost", "status", "reason"}, ("calls", "realized_cost", "wall_clock_cost")),
        ("output_failure_results", {"baseline_id", "failure_status", "failure_count", "recovery_status", "status", "reason"}, ("failure_count",)),
    )
    for output_key, expected_fields, numeric_fields in curve_specs:
        fields, rows = read_tsv(contract["c3_static"][output_key])
        if set(fields) != expected_fields or not rows or not set(BASELINES).issubset({row.get("baseline_id", "") for row in rows}):
            blockers.append({"code": f"c3_{output_key}_incomplete", "message": f"C3 {output_key} lacks matched baseline rows."})
            continue
        if any(not _completed(row.get("status")) or any(not _finite_cell(row, key) for key in numeric_fields) for row in rows):
            blockers.append({"code": f"c3_{output_key}_invalid", "message": f"C3 {output_key} contains non-completed or non-finite rows."})
    _validate_file_binding(c3.get("static_agent_checkpoint"), "c3_static_agent_checkpoint", blockers)
    for key in ("episode_pool_hash", "budget_hash", "tool_availability_hash", "failure_schedule_hash", "evaluation_units_hash"):
        if not isinstance(c3.get(key), str) or not c3.get(key):
            blockers.append({"code": "c3_comparison_freeze_incomplete", "message": "Passed C3 evidence must bind identical pools, budgets, tools, failures, and evaluation units."})
            break


def _validate_c4_outputs(contract: Mapping[str, Any], c4: Mapping[str, Any], tri_freeze: Mapping[str, Any], blockers: list[dict[str, str]]) -> None:
    fields, rows = read_tsv(contract["c4_tri"]["output_results"])
    expected_fields = {
        "variant",
        "calibration_split",
        "risk_coverage_curve",
        "status",
        "reason",
        *contract["c4_tri"]["metrics"],
    }
    if set(fields) != expected_fields:
        blockers.append({"code": "c4_result_schema_incomplete", "message": "C4 results do not expose the frozen selective-risk metrics."})
    if {row.get("variant", "") for row in rows} != set(C4_VARIANTS) or len(rows) != len(C4_VARIANTS):
        blockers.append({"code": "c4_variant_results_incomplete", "message": "C4 must evaluate every frozen TRI variant exactly once."})
    numeric_metrics = tuple(metric for metric in contract["c4_tri"]["metrics"] if metric != "risk_coverage_curve")
    for row in rows:
        if (
            row.get("calibration_split") != "validation_only"
            or not _completed(row.get("status"))
            or not str(row.get("risk_coverage_curve", "")).strip()
            or str(row.get("risk_coverage_curve", "")).upper() == "NOT_AVAILABLE"
            or any(not _finite_cell(row, str(metric)) for metric in numeric_metrics)
        ):
            blockers.append({"code": "c4_result_values_incomplete", "message": "Every C4 variant needs completed validation-only finite reliability evidence."})
            break
    full_tri = next((row for row in rows if row.get("variant") == "full_tri"), None)
    if full_tri is None or not _finite_cell(full_tri, "coverage") or float(full_tri["coverage"]) < float(contract["c4_tri"]["coverage_floor"]):
        blockers.append({"code": "c4_coverage_floor_not_evidenced", "message": "Full TRI must demonstrate the frozen coverage floor."})
    for key in ("tri_model", "calibration_parameters", "thresholds"):
        _validate_file_binding(tri_freeze.get(key), f"c4_{key}", blockers)


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    audit_path: str | Path = "Output/audits/round22_readiness_audit.json",
    rebuild: bool = False,
    allow_formal_hash_readback: bool | None = None,
) -> dict[str, object]:
    contract = load_contract(config_path, schema_path)
    if rebuild:
        build(config_path, schema_path)
    blockers: list[dict[str, str]] = []
    if allow_formal_hash_readback is None:
        allow_formal_hash_readback = socket.gethostname() != "mgmt2"

    freeze = read_json(contract["freeze"]["output"])
    if freeze.get("immutable_during_round22") is not True:
        blockers.append({"code": "static_freeze_not_immutable", "message": "Static Twin freeze must be immutable during Round-22."})
    if freeze.get("status") == "passed" and not freeze.get("frozen"):
        blockers.append({"code": "inconsistent_static_freeze", "message": "A passed freeze must set frozen=true."})
    if freeze.get("status") != "passed" and freeze.get("twin_checkpoint_hash") is not None:
        blockers.append({"code": "unbound_twin_hash", "message": "A blocked freeze cannot publish a checkpoint hash."})

    matrix = read_json(contract["tool_outcomes"]["output_matrix"])
    if set(matrix.get("required_fields", [])) != set(TOOL_OUTCOME_FIELDS):
        blockers.append({"code": "tool_outcome_schema_drift", "message": "Tool-outcome required fields drifted."})
    if tuple(matrix.get("formal_metadata_contract", [])) != tuple(contract["tool_outcomes"]["formal_metadata_fields"]):
        blockers.append({"code": "formal_tool_metadata_contract_drift", "message": "Formal tool metadata requirements drifted from the YAML contract."})
    outcomes = matrix.get("outcomes")
    if not isinstance(outcomes, list):
        blockers.append({"code": "tool_outcome_rows_missing", "message": "Tool-outcome matrix rows are missing."})
        outcomes = []
    for row in outcomes:
        if not isinstance(row, Mapping) or set(TOOL_OUTCOME_FIELDS) - set(row):
            blockers.append({"code": "tool_outcome_row_incomplete", "message": "Every retained outcome must expose the typed required fields."})
            break
    formal_count = _safe_int(
        matrix.get("formal_eligible_count"),
        code="tool_outcome_count_invalid",
        message="Formal-eligibility count must be a non-negative integer.",
        blockers=blockers,
    )
    observed_formal_count = sum(row.get("formal_eligible") is True for row in outcomes if isinstance(row, Mapping))
    if formal_count is not None and formal_count != observed_formal_count:
        blockers.append({"code": "tool_outcome_count_mismatch", "message": "Formal-eligibility count does not match outcome rows."})
    for row in outcomes:
        if not isinstance(row, Mapping) or not row.get("formal_eligible"):
            continue
        state = row.get("state")
        action = row.get("structured_action")
        result = row.get("result")
        uncertainty = row.get("uncertainty")
        required_formal = (
            row.get("actual_runtime"),
            row.get("tool_version"),
            row.get("fidelity"),
            row.get("compute_cost"),
            row.get("wall_clock_cost"),
            row.get("hidden_evaluation_role"),
        )
        if any(value in (None, "") for value in required_formal):
            blockers.append({"code": "formal_tool_metadata_missing", "message": "Formal tool outcomes require runtime, typed version, fidelity, and both cost roles."})
            break
        if not all(_nonnegative_finite(row.get(key)) for key in ("actual_runtime", "compute_cost", "wall_clock_cost")):
            blockers.append({"code": "formal_tool_numeric_metadata_invalid", "message": "Formal runtime and cost values must be finite and non-negative."})
            break
        if str(row.get("failure_status", "")).strip().lower() not in {"completed", "success", "succeeded"}:
            blockers.append({"code": "formal_tool_execution_failed", "message": "Only successful formal tool executions can be admitted."})
            break
        if not isinstance(state, Mapping) or not str(state.get("state_uid", "")).strip():
            blockers.append({"code": "formal_state_uid_missing", "message": "Formal tool outcomes require a canonical state UID."})
            break
        if not isinstance(action, Mapping) or not str(action.get("action_uid", "")).strip():
            blockers.append({"code": "formal_action_uid_missing", "message": "Formal tool outcomes require a canonical structured-action UID."})
            break
        if (
            not isinstance(result, Mapping)
            or not str(result.get("schema_version", "")).strip()
            or result.get("hash_verified") is not True
        ):
            blockers.append({"code": "formal_result_hash_unverified", "message": "Formal tool results require adapter-recorded result hash verification."})
            break
        if (
            not isinstance(uncertainty, Mapping)
            or not str(uncertainty.get("schema_version", "")).strip()
            or uncertainty.get("hash_verified") is not True
        ):
            blockers.append({"code": "formal_uncertainty_hash_unverified", "message": "Formal uncertainty artifacts require adapter-recorded hash verification."})
            break
        result_path = result.get("artifact_path")
        uncertainty_path = uncertainty.get("artifact_path")
        if not isinstance(result_path, str) or not isinstance(uncertainty_path, str):
            blockers.append({"code": "formal_artifact_path_missing", "message": "Formal result and uncertainty artifacts require explicit paths."})
            break
        if repo_path(result_path).resolve() == repo_path(uncertainty_path).resolve():
            blockers.append({"code": "formal_artifact_roles_not_distinct", "message": "Formal result and uncertainty evidence must use distinct typed artifacts."})
            break
        if (
            result.get("content_hash") != result.get("observed_hash")
            or result.get("content_hash") != result.get("artifact_sha256")
            or uncertainty.get("content_hash") != uncertainty.get("observed_hash")
            or uncertainty.get("content_hash") != uncertainty.get("artifact_sha256")
        ):
            blockers.append({"code": "formal_hash_readback_mismatch", "message": "Formal result and uncertainty hashes must equal their adapter readback hashes."})
            break
        if not allow_formal_hash_readback:
            blockers.append(
                {
                    "code": "formal_artifact_hash_readback_requires_compute_node",
                    "message": "Formal artifact-byte validation must run on a compute node.",
                }
            )
            break
        before_hash_blockers = len(blockers)
        _validate_file_binding(
            {"path": result_path, "sha256": result.get("content_hash")},
            "formal_result_artifact",
            blockers,
        )
        _validate_file_binding(
            {"path": uncertainty_path, "sha256": uncertainty.get("content_hash")},
            "formal_uncertainty_artifact",
            blockers,
        )
        if len(blockers) != before_hash_blockers:
            break
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("source_role") != "round22_formal_tool_outcome"
            or provenance.get("formal_round22_execution") is not True
            or provenance.get("formal_round_id") != contract["round_id"]
            or not provenance.get("hash_readback_uid")
        ):
            blockers.append({"code": "formal_round22_provenance_missing", "message": "Formal outcomes require a Round-22 source role, execution marker, round identity, and hash-readback UID."})
            break

    coverage_fields, _ = read_tsv(contract["tool_outcomes"]["output_coverage"])
    expected_coverage = {"tool_uid", "source_rows", "formal_eligible_rows", "completed_rows", "failure_rows", "invalid_rows", "runtime_observed_rows", "tool_version_observed", "formal_status"}
    if set(coverage_fields) != expected_coverage:
        blockers.append({"code": "tool_coverage_schema_drift", "message": "Tool coverage TSV schema drifted."})
    failure_fields, _ = read_tsv(contract["tool_outcomes"]["output_failure_catalog"])
    expected_failure = {"record_uid", "source_manifest", "tool_uid", "failure_status", "failure_kind", "observed", "formal_eligible", "note"}
    if set(failure_fields) != expected_failure:
        blockers.append({"code": "tool_failure_schema_drift", "message": "Tool failure catalog TSV schema drifted."})

    cost = read_json(contract["tool_outcomes"]["cost_model_freeze"])
    if cost.get("status") not in {"frozen_definitions_not_calibrated", "passed"}:
        blockers.append({"code": "cost_model_missing", "message": "Three cost definitions are not frozen."})
    if cost.get("sensitivity_contract", {}).get("relative_variations") != [-0.5, -0.25, 0.25, 0.5]:
        blockers.append({"code": "cost_sensitivity_drift", "message": "Cost sensitivity must retain +/-25% and +/-50% variations."})

    c3 = read_json(contract["c3_static"]["gate_output"])
    _validate_gate_state(c3, "c3_gate", blockers)
    c3_passed = _gate_passed(c3)
    if c3_passed:
        _require_true(
            c3,
            ("static_twin_frozen", "mandatory_baselines_complete", "offline_replay_complete", "real_static_rounds_complete", "fixed_budget_reported", "fixed_risk_reported", "hidden_outcomes_unseen"),
            "c3_promotion_contract_incomplete",
            "A passed C3 gate must bind the frozen Twin, all mandatory baselines, replay, real rounds, both primary directions, and hidden-outcome isolation.",
            blockers,
        )
        if formal_count is None or formal_count <= 0:
            blockers.append({"code": "c3_promotion_without_formal_outcomes", "message": "A passed C3 gate requires formal tool outcomes."})
        _validate_c3_outputs(contract, c3, blockers)
    baseline_ids = set(c3.get("mandatory_baselines", {}))
    if baseline_ids != set(BASELINES):
        blockers.append({"code": "c3_baseline_set_drift", "message": "C3 baseline set must be the frozen ICLR-sized set plus an evaluation-only oracle."})
    if c3.get("oracle", {}).get("deployable") is not False:
        blockers.append({"code": "oracle_promoted", "message": "The full-information oracle must remain evaluation-only."})
    if c3.get("hidden_outcomes_unseen") is not True or c3.get("holdout_evidence", {}).get("exists") is not True:
        blockers.append({"code": "holdout_isolation_unverified", "message": "Final hidden outcomes must remain hash-bound and unseen."})

    c4 = read_json(contract["c4_tri"]["gate_output"])
    _validate_gate_state(c4, "c4_gate", blockers)
    c4_passed = _gate_passed(c4)
    if c4.get("calibration_validation_only") is not True:
        blockers.append({"code": "tri_nonvalidation_calibration", "message": "TRI calibration must use validation only."})
    tri_freeze = read_json(contract["c4_tri"]["output_freeze"])
    if c4_passed:
        _require_true(
            c4,
            ("calibration_validation_only", "risk_coverage_complete", "coverage_floor_respected"),
            "c4_promotion_contract_incomplete",
            "A passed C4 gate must bind validation-only calibration, complete selective-risk evidence, and the coverage floor.",
            blockers,
        )
        if c3_passed is not True or tri_freeze.get("frozen") is not True:
            blockers.append({"code": "c4_promotion_without_c3_or_tri_freeze", "message": "A passed C4 gate requires passed C3 evidence and a frozen TRI reference."})
        _validate_c4_outputs(contract, c4, tri_freeze, blockers)
    if tri_freeze.get("frozen") is True and not c4_passed:
        blockers.append({"code": "tri_freeze_gate_mismatch", "message": "TRI freeze and C4 gate disagree."})
    if set(tri_freeze.get("variants", [])) != set(C4_VARIANTS):
        blockers.append({"code": "tri_variant_set_drift", "message": "TRI comparison variants are incomplete."})

    comparison = read_json(contract["coevolution"]["output_comparison_freeze"])
    if set(comparison.get("systems", {})) != set(MATRIX):
        blockers.append({"code": "comparison_matrix_drift", "message": "The frozen comparison matrix is incomplete."})
    evidence = read_json(contract["coevolution"]["output_evidence"])
    _validate_gate_state(evidence, "coevolution_gate", blockers, passed_key="final_gate_passed")
    evidence_passed = _gate_passed(evidence, passed_key="final_gate_passed")
    evidence_round_count = _safe_int(
        evidence.get("real_round_count"),
        code="coevolution_round_count_invalid",
        message="Co-evolution real_round_count must be a non-negative integer.",
        blockers=blockers,
    )
    if evidence_passed:
        if evidence_round_count is None or evidence_round_count < 3:
            blockers.append({"code": "coevolution_round_count_incomplete", "message": "A passed co-evolution gate requires at least three real sequential rounds."})
        static_reference = evidence.get("static_reference", {})
        if not isinstance(static_reference, Mapping) or static_reference.get("frozen") is not True:
            blockers.append({"code": "coevolution_static_reference_missing", "message": "Co-evolution evidence must bind the immutable static reference."})
        if not isinstance(evidence.get("scientific_utility"), Mapping) or not all(
            evidence["scientific_utility"].get(key) is True
            for key in ("full_beats_static", "full_beats_strongest_one_sided", "fixed_budget_and_fixed_risk_consistent")
        ):
            blockers.append({"code": "coevolution_utility_contract_incomplete", "message": "Passed co-evolution evidence must resolve the fixed-budget and fixed-risk utility comparisons."})
        retention = evidence.get("retention", {})
        reliability = evidence.get("reliability", {})
        rollback = evidence.get("rollback", {})
        if not isinstance(retention, Mapping) or retention.get("catastrophic_forgetting_within_tolerance") is not True:
            blockers.append({"code": "coevolution_retention_contract_incomplete", "message": "Passed co-evolution evidence must demonstrate retention within tolerance."})
        if not isinstance(reliability, Mapping) or not all(
            reliability.get(key) is True for key in ("calibration_within_tolerance", "coverage_floor_respected", "support_domain_not_degraded")
        ):
            blockers.append({"code": "coevolution_reliability_contract_incomplete", "message": "Passed co-evolution evidence must preserve calibration and support-domain coverage."})
        if not isinstance(rollback, Mapping) or rollback.get("demonstrated") is not True:
            blockers.append({"code": "coevolution_rollback_missing", "message": "Passed co-evolution evidence must demonstrate checkpoint rollback and restoration."})
        formal_validation = validate_formal_coevolution(require_rounds=True, require_final_evidence=True)
        if formal_validation.get("status") != "passed" or formal_validation.get("final_gate_passed") is not True:
            blockers.append(
                {
                    "code": "formal_coevolution_evidence_invalid",
                    "message": "Round-22 co-evolution promotion requires the repository's hash-bound round, rollback, ablation, and paired-statistics validator to pass.",
                }
            )
    program = read_json(contract["coevolution"]["output_program_gate"])
    _validate_gate_state(program, "program_gate", blockers)
    if program.get("c2_b_status") != "blocked_insufficient_real_e2_e4":
        blockers.append({"code": "c2_b_claim_drift", "message": "C2-B must remain blocked_insufficient_real_e2_e4."})
    recovery = read_json(contract["claim_boundary"]["c2_b_recovery_artifact"])
    qualified_domain_count = _safe_int(
        recovery.get("qualified_action_complete_domain_count"),
        code="c2_b_recovery_count_invalid",
        message="C2-B qualified-domain count must be a non-negative integer.",
        blockers=blockers,
    )
    if recovery.get("c2_b_status") != "blocked_insufficient_real_e2_e4" or qualified_domain_count != 0:
        blockers.append({"code": "c2_b_recovery_drift", "message": "C2-B recovery must not promote an unqualified domain."})
    c2_b_binding = program.get("c2_b_recovery_artifact")
    if not isinstance(c2_b_binding, Mapping) or c2_b_binding.get("exists") is not True:
        blockers.append({"code": "c2_b_recovery_binding_missing", "message": "Program gate must bind the C2-B recovery status artifact."})
    else:
        configured_recovery = repo_path(contract["claim_boundary"]["c2_b_recovery_artifact"]).resolve()
        bound_path = c2_b_binding.get("path")
        if not isinstance(bound_path, str) or repo_path(bound_path).resolve() != configured_recovery:
            blockers.append({"code": "c2_b_recovery_path_mismatch", "message": "Program gate must bind the configured C2-B recovery artifact."})
        _validate_file_binding(c2_b_binding, "c2_b_recovery", blockers)
    intermediate = read_json(contract["gates"]["intermediate_output"])
    _validate_gate_state(intermediate, "intermediate_gate", blockers)
    intermediate_passed = _gate_passed(intermediate)
    if intermediate_passed:
        _require_true(
            intermediate,
            ("static_twin_frozen",),
            "intermediate_reference_incomplete",
            "A passed static Agent/reliability reference must bind the immutable Twin.",
            blockers,
        )
        c3_summary = intermediate.get("c3", {})
        c4_summary = intermediate.get("c4", {})
        if not isinstance(c3_summary, Mapping) or not all(
            c3_summary.get(key) is True
            for key in ("mandatory_baselines_complete", "offline_replay_complete", "real_static_rounds_complete", "fixed_budget_reported", "fixed_risk_reported")
        ):
            blockers.append({"code": "intermediate_c3_contract_incomplete", "message": "A passed intermediate gate must bind every C3 completion criterion."})
        if not isinstance(c4_summary, Mapping) or not all(
            c4_summary.get(key) is True for key in ("calibration_validation_only", "risk_coverage_complete", "coverage_floor_respected")
        ):
            blockers.append({"code": "intermediate_c4_contract_incomplete", "message": "A passed intermediate gate must bind every C4 completion criterion."})
        artifacts = intermediate.get("artifacts", {})
        if not isinstance(artifacts, Mapping) or not all(
            artifacts.get(key) is True for key in ("static_agent_checkpoint_frozen", "tri_thresholds_frozen")
        ):
            blockers.append({"code": "intermediate_artifacts_incomplete", "message": "A passed intermediate gate must bind the static Agent checkpoint and TRI thresholds."})
    lineage = read_json(contract["coevolution"]["output_checkpoint_lineage"])
    if not evidence_passed and evidence_round_count == 0 and (lineage.get("rounds") or lineage.get("rollback_restorations")):
        blockers.append({"code": "checkpoint_lineage_fabricated", "message": "Blocked Round-22 state cannot contain formal checkpoint lineage."})
    reproducibility = read_json(contract["reproducibility"]["output"])
    program_passed = _gate_passed(program)
    if not program_passed and reproducibility.get("formal_execution_commands"):
        blockers.append({"code": "formal_command_published_while_blocked", "message": "Formal execution commands must remain absent while the gate is blocked."})
    resource = read_json(contract["reproducibility"]["resource_catalog"])
    formal_job_count = _safe_int(
        resource.get("formal_job_count"),
        code="formal_job_count_invalid",
        message="formal_job_count must be a non-negative integer.",
        blockers=blockers,
    )
    if not program_passed and formal_job_count != 0:
        blockers.append({"code": "formal_resource_claim_drift", "message": "No formal Round-22 jobs were authorized."})

    audit = read_json(audit_path)
    if audit.get("program_gate", {}).get("gate") != contract["program_gate"]:
        blockers.append({"code": "program_gate_binding_missing", "message": "Round-22 readiness audit is not bound to the configured program gate."})
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "program_gate": contract["program_gate"],
        "program_gate_passed": program.get("passed") is True,
        "final_gate": contract["final_gate"],
        "final_gate_passed": evidence.get("final_gate_passed") is True,
        "c2_b_status": program.get("c2_b_status"),
        "artifact_hashes": {
            "static_twin_freeze": sha256_file(repo_path(contract["freeze"]["output"])),
            "tool_outcome_matrix": sha256_file(repo_path(contract["tool_outcomes"]["output_matrix"])),
            "c3_gate": sha256_file(repo_path(contract["c3_static"]["gate_output"])),
            "c4_gate": sha256_file(repo_path(contract["c4_tri"]["gate_output"])),
            "coevolution_evidence": sha256_file(repo_path(contract["coevolution"]["output_evidence"])),
        },
    }


def main() -> None:
    args = parse_args()
    result = validate(args.config, args.schema, args.audit, rebuild=args.rebuild)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
