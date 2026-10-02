"""Build the fail-closed Round-24 development freeze from executed evidence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.decision_consistent_runtime import (  # noqa: E402
    DecisionConsistentTwin,
    InformationAction,
    ScientificPosterior,
    ToolResult,
)

AUDIT_ROOT = REPO_ROOT / "Output/audits"
REPORT_ROOT = REPO_ROOT / "Output/paper_reports"
RUN_ROOT = REPO_ROOT / "Output/runs/round24"
AUTHORITATIVE_RUN_ROOT = RUN_ROOT / "twin_only"
FAILED_ATTEMPT_ROOT = RUN_ROOT / "failed_tool_dispatch_512625"
C1_DIAGNOSTICS = REPORT_ROOT / "round23_dev_c1_diagnostics.tsv"
C2S_DIAGNOSTICS = REPORT_ROOT / "round23_dev_c2s_diagnostics.tsv"
ROUND23_STATIC = AUDIT_ROOT / "round23_static_dev_reference.json"
ROUND23_MECHANISM = AUDIT_ROOT / "round23_coupling_mechanism_gate.json"
ROUND23_FULL_SUITE = AUDIT_ROOT / "round23_full_suite_execution.json"

ROUND23_SUCCESSOR_SOURCE_PATHS = (
    "Libs/configs/neuronsifter/vnext_research_contract.yaml",
    "Libs/configs/schema/neuronsifter_vnext_research_contract.schema.json",
    "Libs/configs/validation/iclr_design_contract_union.yaml",
    "Libs/scripts/builders/build_round23_vnext_artifacts.py",
    "Libs/tests/test_round23_evidence_integrity.py",
)

EVIDENCE_ROLE = "development_only"
C2B_STATUS = "blocked_insufficient_real_e2_e4"
BLOCKED_STATUS = "blocked_adaptive_development"
BRANCHES = (
    "frozen_twin_frozen_agent",
    "adaptive_twin_frozen_agent",
    "frozen_twin_adaptive_agent",
    "adaptive_twin_adaptive_agent",
)
PRIMARY_BRANCHES = BRANCHES[:2]
CONDITIONAL_BRANCHES = BRANCHES[2:]
REQUIRED_ROUND_FILES = (
    "round_input_manifest.json",
    "selected_information_actions.tsv",
    "tool_results.tsv",
    "evidence_admission.tsv",
    "posterior_update.json",
    "twin_candidate_update.json",
    "retention.tsv",
    "calibration.tsv",
    "promotion_or_rollback.json",
    "report.json",
)
SCIENTIFIC_METRIC_STATUS = "not_evaluable_missing_model_bound_pre_post_evaluation"
PLACEHOLDER_METRIC_FIELDS = (
    "old_domain_retention",
    "calibration_drift",
    "coverage",
    "aurc",
    "false_accept",
    "decision_risk_at_matched_cost",
    "cost_to_matched_risk",
    "cumulative_regret",
    "new_domain_gain",
    "twin_debt",
)
HISTORICAL_RAW_EXPORTS = {
    "prospective_tool_results": "Output/audits/round24_prospective_tool_results.tsv",
    "adaptation_matrix": "Output/paper_reports/round24_development_adaptation_matrix.tsv",
    "twin_adaptation": "Output/paper_reports/round24_dev_twin_adaptation.tsv",
}
EVIDENCE_BOUND_SUCCESSOR_EXPORTS = {
    "prospective_tool_results": "Output/audits/round24_prospective_tool_results_evidence_bound.tsv",
    "adaptation_matrix": "Output/paper_reports/round24_development_adaptation_matrix_evidence_bound.tsv",
    "twin_adaptation": "Output/paper_reports/round24_dev_twin_adaptation_evidence_bound.tsv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_tsv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _unavailable(reason: str, required_binding: str) -> dict[str, str]:
    return {
        "status": "not_evaluable",
        "reason": reason,
        "required_binding": required_binding,
    }


def _bind_round23_successor_sources() -> dict[str, Any]:
    round23 = _json(ROUND23_FULL_SUITE)
    recorded = {
        str(row.get("path", "")): str(row.get("sha256", ""))
        for row in round23.get("source_files", [])
        if isinstance(row, Mapping)
    }
    transitions: list[dict[str, Any]] = []
    for relative in ROUND23_SUCCESSOR_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if relative not in recorded or not path.is_file():
            raise ValueError(f"Round-24 successor source is not Round-23 bound: {relative}")
        current_hash = _sha256(path)
        if current_hash == recorded[relative]:
            raise ValueError(f"Round-24 successor source did not change: {relative}")
        transitions.append(
            {
                "path": relative,
                "round23_sha256": recorded[relative],
                "round24_sha256": current_hash,
                "transition_reason": "round24_contract_or_validation_extension",
            }
        )
    payload = {
        "schema_version": "neuronsifter-round24-round23-source-successor-binding-v1",
        "status": "bound_successor_development",
        "round23_full_suite_audit_path": _relative(ROUND23_FULL_SUITE),
        "round23_full_suite_audit_sha256": _sha256(ROUND23_FULL_SUITE),
        "source_transitions": transitions,
        "historical_round23_audit_rewritten": False,
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }
    _write_json(
        AUDIT_ROOT / "round24_round23_source_successor_binding.json", payload
    )
    return payload


def _validate_round_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    round_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        for round_index in range(1, 4):
            round_dir = AUTHORITATIVE_RUN_ROOT / branch / f"round_{round_index}"
            if not round_dir.is_dir():
                raise FileNotFoundError(round_dir)
            missing = [name for name in REQUIRED_ROUND_FILES if not (round_dir / name).is_file()]
            if missing:
                raise ValueError(f"Round-24 round artifact set is incomplete: {branch}/{round_index}: {missing}")
            manifest = _json(round_dir / "round_input_manifest.json")
            selection = _json(round_dir / "selected_information_action.json")
            tool = _json(round_dir / "tool_result.json")
            update = _json(round_dir / "posterior_update.json")
            report = _json(round_dir / "report.json")
            promotion = _json(round_dir / "promotion_or_rollback.json")
            dispatch = _json(round_dir / "isolated_dispatch/dispatch.json")
            if (
                int(manifest.get("round_index", -1)) != round_index
                or manifest.get("branch_id") != branch
                or manifest.get("outcome_visible") is not False
                or manifest.get("evidence_role") != EVIDENCE_ROLE
                or manifest.get("publication_claim_eligible") is not False
                or int(manifest.get("protected_test_access", -1)) != 0
            ):
                raise ValueError(f"Round-24 input freeze contract failed: {branch}/{round_index}")
            if selection.get("selection_before_outcome") is not True:
                raise ValueError(f"Round-24 selection was not frozen before outcome: {branch}/{round_index}")
            if selection.get("world_action_uid") is not None:
                raise ValueError("Preserved Round-24 execution unexpectedly claims a model-bound world action")
            if tool.get("posterior_uid") != selection.get("posterior_uid"):
                raise ValueError(f"Round-24 tool parent posterior drift: {branch}/{round_index}")
            if tool.get("status") not in {"completed", "partial", "failed", "invalid"}:
                raise ValueError(f"Round-24 tool status is invalid: {branch}/{round_index}")
            if dispatch.get("status") != tool.get("status"):
                raise ValueError(f"Round-24 isolated dispatch/result status drift: {branch}/{round_index}")
            if report.get("evidence_role") != EVIDENCE_ROLE or report.get("publication_claim_eligible") is not False:
                raise ValueError(f"Round-24 report crossed its evidence boundary: {branch}/{round_index}")
            parent_uid = str(update.get("parent_posterior_uid"))
            decision_uid = str(update.get("decision_posterior_uid"))
            if promotion.get("prior_posterior_uid") != parent_uid or promotion.get("authoritative_posterior_uid") != decision_uid:
                raise ValueError(f"Round-24 promotion lineage drift: {branch}/{round_index}")
            if round_index > 1:
                previous = round_rows[-1] if round_rows and round_rows[-1]["branch_id"] == branch else None
                if previous is None or parent_uid != previous["decision_posterior_uid"]:
                    raise ValueError(f"Round-24 sequential posterior lineage failed: {branch}/{round_index}")
            for metric in PLACEHOLDER_METRIC_FIELDS:
                value = update.get(metric)
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"Round-24 historical placeholder metric is malformed: {metric}"
                    )
            row = {
                "branch_id": branch,
                "round_index": round_index,
                "parent_posterior_uid": parent_uid,
                "decision_posterior_uid": decision_uid,
                "request_uid": selection.get("request_uid"),
                "run_uid": selection.get("run_uid"),
                "information_action_uid": selection.get("information_action_uid"),
                "world_action_uid": selection.get("world_action_uid"),
                "tool_status": tool.get("status"),
                "tool_version": tool.get("tool_version"),
                "raw_bytes_hash": tool.get("raw_bytes_hash"),
                "normalized_bytes_hash": tool.get("normalized_bytes_hash"),
                "runtime_seconds": float(tool.get("runtime_seconds", 0.0)),
                "promotion_or_rollback": update.get("promotion_or_rollback"),
                "twin_parameters_changed": bool(update.get("twin_parameters_changed")),
                "agent_parameters_changed": bool(update.get("agent_parameters_changed")),
                "decision_trace_consistent": bool(update.get("decision_consistency_preserved")),
                "selection_before_outcome": True,
                "world_action_binding_status": "absent_in_preserved_prospective_unit",
                "scientific_metric_status": SCIENTIFIC_METRIC_STATUS,
                "historical_raw_metric_values_invalidated": True,
                "protected_test_access": 0,
                "evidence_role": EVIDENCE_ROLE,
                "publication_claim_eligible": False,
            }
            round_rows.append(row)
            tool_rows.append({**row, "error": tool.get("result", {}).get("error") or ""})
    return round_rows, tool_rows


def _branch_evidence_status(branch_id: str) -> str:
    if branch_id in CONDITIONAL_BRANCHES:
        return "invalidated_premature_conditional_execution"
    return "prospective_execution_software_and_provenance_only"


def _write_evidence_bound_successors(
    round_rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    metrics_path: Path,
) -> dict[str, Any]:
    """Create current labelled views without rewriting historical raw exports."""

    common = {
        "scientific_metric_status": SCIENTIFIC_METRIC_STATUS,
        "historical_raw_metric_values_invalidated": True,
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }
    metric_values = {name: "not_evaluable" for name in PLACEHOLDER_METRIC_FIELDS}
    matrix_rows = [
        {
            "branch_id": row["branch_id"],
            "round_index": row["round_index"],
            "tool_status": row["tool_status"],
            "twin_parameters_changed": row["twin_parameters_changed"],
            "agent_parameters_changed": row["agent_parameters_changed"],
            "decision_consistency_preserved": row["decision_trace_consistent"],
            "twin_parameter_change_scientific_evidence_admissible": False,
            "branch_evidence_status": _branch_evidence_status(str(row["branch_id"])),
            "old_domain_retention": "not_evaluable",
            "calibration_drift": "not_evaluable",
            "promotion_or_rollback": row["promotion_or_rollback"],
            **common,
        }
        for row in round_rows
    ]
    twin_rows = [
        {
            "branch_id": row["branch_id"],
            "round_index": row["round_index"],
            "parent_posterior_uid": row["parent_posterior_uid"],
            "decision_posterior_uid": row["decision_posterior_uid"],
            "request_uid": row["request_uid"],
            "run_uid": row["run_uid"],
            "information_action_uid": row["information_action_uid"],
            "world_action_uid": row["world_action_uid"],
            "world_action_binding_status": row["world_action_binding_status"],
            "tool_status": row["tool_status"],
            "twin_parameters_changed": row["twin_parameters_changed"],
            "agent_parameters_changed": row["agent_parameters_changed"],
            "decision_consistency_preserved": row["decision_trace_consistent"],
            "twin_parameter_change_scientific_evidence_admissible": False,
            "branch_evidence_status": _branch_evidence_status(str(row["branch_id"])),
            "promotion_or_rollback": row["promotion_or_rollback"],
            **metric_values,
            **common,
        }
        for row in round_rows
    ]
    tool_export_rows = [
        {
            "request_uid": row["request_uid"],
            "run_uid": row["run_uid"],
            "branch_id": row["branch_id"],
            "round_index": row["round_index"],
            "posterior_uid": row["parent_posterior_uid"],
            "information_action_uid": row["information_action_uid"],
            "world_action_uid": row["world_action_uid"],
            "world_action_binding_status": row["world_action_binding_status"],
            "tool_status": row["tool_status"],
            "tool_version": row["tool_version"],
            "raw_bytes_hash": row["raw_bytes_hash"],
            "normalized_bytes_hash": row["normalized_bytes_hash"],
            "runtime_seconds": row["runtime_seconds"],
            "selection_before_outcome": row["selection_before_outcome"],
            "branch_evidence_status": _branch_evidence_status(str(row["branch_id"])),
            "error": row["error"],
            **common,
        }
        for row in tool_rows
    ]

    table_payloads = {
        "adaptation_matrix": matrix_rows,
        "twin_adaptation": twin_rows,
        "prospective_tool_results": tool_export_rows,
    }
    for name, rows in table_payloads.items():
        path = REPO_ROOT / EVIDENCE_BOUND_SUCCESSOR_EXPORTS[name]
        if not rows:
            raise ValueError(f"Round-24 evidence-bound export has no rows: {name}")
        _write_tsv(path, list(rows[0]), rows)

    historical_raw_exports = {
        name: {
            **_artifact_identity(REPO_ROOT / relative),
            "status": "invalidated_raw_export_not_current_evidence_view",
            "reason": "missing_row_level_evidence_boundary_or_invalidated_placeholder_metrics",
        }
        for name, relative in HISTORICAL_RAW_EXPORTS.items()
    }
    successor_exports = {
        name: _artifact_identity(REPO_ROOT / relative)
        for name, relative in EVIDENCE_BOUND_SUCCESSOR_EXPORTS.items()
    }
    catalog_path = AUDIT_ROOT / "round24_evidence_bound_export_catalog.json"
    _write_json(
        catalog_path,
        {
            "schema_version": "neuronsifter-round24-evidence-bound-export-catalog-v1",
            "status": "current_successor_exports_bound",
            "historical_raw_exports": historical_raw_exports,
            "successor_exports": successor_exports,
            "scientific_metric_provenance": _artifact_identity(metrics_path),
            "all_successor_rows_development_only": True,
            "all_successor_rows_publication_claim_eligible": False,
            "protected_test_access": 0,
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
        },
    )
    return _artifact_identity(catalog_path)


def _failed_attempt_observation() -> dict[str, Any]:
    completed_path = FAILED_ATTEMPT_ROOT / "twin_only/frozen_twin_frozen_agent/round_1/tool_result.json"
    pending_selection_path = FAILED_ATTEMPT_ROOT / "twin_only/frozen_twin_frozen_agent/round_2/selected_information_action.json"
    error_path = REPO_ROOT / "Output/logs/NrnSift_r24_adaptation.512625.err"
    completed = _json(completed_path)
    pending = _json(pending_selection_path)
    if completed.get("status") != "completed" or pending.get("selection_before_outcome") is not True:
        raise ValueError("Preserved Round-24 failed attempt is not internally consistent")
    if not error_path.is_file():
        raise FileNotFoundError(error_path)
    error_text = error_path.read_text(encoding="utf-8")
    expected_error = "did not create any NEURON sections"
    if expected_error not in error_text:
        raise ValueError("Preserved Round-24 tool reload failure signature changed")
    return {
        "attempt_id": "slurm_512625",
        "job_id": "512625",
        "status": "failed_campaign_after_completed_round_1",
        "completed_tool_observation_count": 1,
        "pending_pre_outcome_selection_count": 1,
        "failure_class": "in_process_neuron_model_reload_failed",
        "failure_message": expected_error,
        "valid_scientific_measurement": False,
        "observable_execution_outcome": True,
        "completed_result": _artifact_identity(completed_path),
        "pending_selection": _artifact_identity(pending_selection_path),
        "stderr": _artifact_identity(error_path),
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }


def _diagnostic_boundary() -> dict[str, Any]:
    c1_rows = _rows(C1_DIAGNOSTICS)
    c2s_rows = _rows(C2S_DIAGNOSTICS)
    missing_modality = next(row for row in c1_rows if row["diagnostic_id"] == "missing_modality")
    cross_source = next(row for row in c1_rows if row["diagnostic_id"] == "cross_source_development_transfer")
    c2s_aggregate = next(row for row in c2s_rows if row["row_type"] == "aggregate_mean")
    if (
        missing_modality["status"] != "not_evaluable_transcriptomics_only_contract"
        or cross_source["metric_role"] != "proxy_not_cross_source_holdout"
        or c2s_aggregate["evidence_role"] != EVIDENCE_ROLE
        or c2s_aggregate["publication_claim_eligible"] != "False"
        or c2s_aggregate["protected_test_access"] != "0"
    ):
        raise ValueError("Round-23 diagnostic boundary drifted")
    return {
        "c1": {
            "source": _artifact_identity(C1_DIAGNOSTICS),
            "checkpoint_path": c1_rows[0]["checkpoint_path"],
            "checkpoint_sha256": c1_rows[0]["checkpoint_sha256"],
            "split_uid": c1_rows[0]["split_uid"],
            "current_metrics_are_static_round23_only": True,
            "missing_modality": missing_modality["status"],
            "cross_source_transfer": cross_source["status"],
            "round24_posterior_bound_evaluation": False,
        },
        "c2s": {
            "source": _artifact_identity(C2S_DIAGNOSTICS),
            "selected_seed_count": sum(row["row_type"] == "seed" for row in c2s_rows),
            "aggregate_checkpoint_identity": c2s_aggregate["checkpoint_path"],
            "current_metrics_are_static_round23_only": True,
            "round24_posterior_bound_evaluation": False,
        },
    }


def _scientific_metric_status() -> dict[str, Any]:
    disconnected = "Round-24 posterior assimilation is not consumed by the C1 encoder or C2-S world-action predictor"
    no_predictions = "no aligned pre/post model predictions were generated for the Round-24 posterior lineage"
    no_acceptance = "no model-bound confidence, error, and acceptance arrays were generated"
    return {
        "metric_values_admissible": False,
        "historical_constant_fields_invalidated": [
            "old_domain_retention",
            "calibration_drift",
            "coverage",
            "aurc",
            "false_accept",
            "decision_risk_at_matched_cost",
            "cost_to_matched_risk",
            "cumulative_regret",
            "new_domain_gain",
            "twin_debt",
        ],
        "c1_cross_source_transfer": _unavailable(disconnected, "posterior_bound_c1_cross_source_evaluator"),
        "c1_missing_modality": _unavailable(
            "the frozen C1 contract is transcriptomics-only",
            "qualified_public_missing_modality_development_partition",
        ),
        "c1_source_shortcut": _unavailable(disconnected, "posterior_bound_c1_source_shortcut_evaluator"),
        "c2s_trajectory_error": _unavailable(disconnected, "posterior_conditioned_c2s_rollout"),
        "c2s_long_horizon_drift": _unavailable(disconnected, "posterior_conditioned_c2s_rollout"),
        "c2s_action_contrast": _unavailable(disconnected, "posterior_conditioned_c2s_rollout"),
        "c2s_crps_nll": _unavailable(no_predictions, "pre_post_c2s_predictive_distribution"),
        "c2s_constraint_residual": _unavailable(
            "Round-23 exposes only a negative-state proxy and no Round-24 posterior-conditioned prediction",
            "posterior_conditioned_constraint_residual_evaluator",
        ),
        "calibration_ece": _unavailable(no_predictions, "aligned_pre_post_predictions_and_truth"),
        "aurc": _unavailable(no_acceptance, "per_unit_error_and_confidence_arrays"),
        "false_accept": _unavailable(no_acceptance, "frozen_acceptance_threshold_and_error_labels"),
        "coverage": _unavailable(no_acceptance, "frozen_acceptance_threshold_and_support_labels"),
        "ood_behavior": _unavailable(
            "no dedicated Round-24 ID/OOD partition and posterior-bound scores were generated",
            "development_only_id_ood_partition_and_scores",
        ),
        "old_domain_retention": _unavailable(no_predictions, "frozen_pre_post_old_domain_predictions"),
        "new_domain_gain": _unavailable(no_predictions, "frozen_pre_post_new_domain_predictions"),
        "decision_risk": _unavailable(no_predictions, "world_action_uid_bound_pre_post_decision_scores"),
    }


def _demonstrate_rollback() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_root = AUDIT_ROOT / "round24_bad_update_fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    prior_path = fixture_root / "prior_runtime_checkpoint.json"
    restored_path = fixture_root / "restored_runtime_checkpoint.json"
    candidate_path = fixture_root / "candidate_posterior.json"
    tool_id = "round24.contract_defined_conflicting_evidence_fixture"

    def handler(posterior: ScientificPosterior, action: InformationAction) -> ToolResult:
        return ToolResult(
            posterior_uid=posterior.posterior_uid,
            action_uid=str(action.action_uid),
            status="completed",
            observation={"conflict_score": 1.0, "source_support": 0.0},
            uncertainty={"epistemic": 0.9, "aleatoric": 0.1},
            provenance={
                "fixture_uid": "round24_isolated_conflicting_evidence_v1",
                "isolated_negative_fixture": True,
            },
        )

    tools = {tool_id: handler}
    twin = DecisionConsistentTwin(tools=tools)
    parent = twin.infer_belief(
        {
            "state": {"fixture_uid": "round24_bad_update_parent_v1"},
            "uncertainty": {"epistemic": 0.05, "aleatoric": 0.05},
            "support": {"support_score": 1.0},
            "reliability": {"tri_score": 0.95, "status": "evidence_bound"},
        },
        {"source": "contract_defined_isolated_negative_fixture"},
    )
    frozen_metrics = twin.decisions.compute_reliability(parent.posterior_uid)
    twin.save_checkpoint(prior_path)
    prior_hash = _sha256(prior_path)

    action = InformationAction(
        tool_id=tool_id,
        query={"fixture_uid": "round24_adverse_conflicting_observation_v1"},
    )
    result = twin.predict_tool_outcome(parent.posterior_uid, action)
    outcome = twin.assimilate_observation(parent.posterior_uid, result)
    candidate = outcome.posterior
    _write_json(candidate_path, candidate.to_record())
    degradation_value = float(result.observation["conflict_score"])
    degradation_threshold = 0.2
    degradation_detected = degradation_value > degradation_threshold
    promotion_rejected = bool(outcome.accepted and degradation_detected)
    if not promotion_rejected:
        raise ValueError("Round-24 controlled bad update did not trigger degradation")

    restored = DecisionConsistentTwin.load_checkpoint(prior_path, tools=tools)
    restored.save_checkpoint(restored_path)
    restored_hash = _sha256(restored_path)
    restored_metrics = restored.decisions.compute_reliability(parent.posterior_uid)
    capability_probe = restored.predict_tool_outcome(parent.posterior_uid, action)
    exact_checkpoint_restored = prior_hash == restored_hash
    posterior_identity_restored = restored.decisions.current_posterior_uid == parent.posterior_uid
    metrics_restored = restored_metrics == frozen_metrics
    runtime_capability_restored = capability_probe.status == "completed"
    biological_state_restored = restored.belief.get(parent.posterior_uid).to_record() == parent.to_record()
    passed = bool(
        exact_checkpoint_restored
        and posterior_identity_restored
        and metrics_restored
        and runtime_capability_restored
        and biological_state_restored
    )
    payload = {
        "schema_version": "neuronsifter-round24-bad-update-v2",
        "status": "passed_controlled_bad_update_rollback" if passed else "failed_controlled_bad_update_rollback",
        "candidate_update_attempted": True,
        "candidate_posterior_created": outcome.accepted,
        "candidate_posterior_uid": candidate.posterior_uid,
        "degradation_metric": "contract_defined_conflict_score",
        "candidate_degradation_value": degradation_value,
        "degradation_threshold": degradation_threshold,
        "degradation_detected": degradation_detected,
        "promotion_rejected": promotion_rejected,
        "rollback_executed": True,
        "exact_prior_checkpoint_restored": exact_checkpoint_restored,
        "prior_checkpoint_restored": exact_checkpoint_restored,
        "runtime_capability_restored": runtime_capability_restored,
        "posterior_identity_restored": posterior_identity_restored,
        "posterior_lineage_restored": posterior_identity_restored,
        "metrics_restored_within_tolerance": metrics_restored,
        "metrics_tolerance": 0.0,
        "biological_state_restored": biological_state_restored,
        "biological_state_moved": False,
        "prior_posterior_uid": parent.posterior_uid,
        "authoritative_posterior_uid": restored.decisions.current_posterior_uid,
        "prior_runtime_checkpoint": _artifact_identity(prior_path),
        "restored_runtime_checkpoint": _artifact_identity(restored_path),
        "candidate_posterior": _artifact_identity(candidate_path),
        "isolated_fixture": True,
        "production_artifact_corrupted": False,
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }
    return payload, dict(payload)


def _resource_audit(tool_rows: list[dict[str, Any]], failed_attempt: Mapping[str, Any]) -> dict[str, Any]:
    total_tool_seconds = sum(float(row["runtime_seconds"]) for row in tool_rows)
    storage_paths = list(AUTHORITATIVE_RUN_ROOT.glob("*/round_*/isolated_dispatch/payload_arrays.npz"))
    return {
        "schema_version": "neuronsifter-round24-resource-cost-v1",
        "status": "measured_execution_cost_only",
        "authoritative_dispatch_count": len(tool_rows),
        "authoritative_successful_measurement_count": sum(row["tool_status"] in {"completed", "partial"} for row in tool_rows),
        "preserved_failed_campaign_observation_count": int(failed_attempt["completed_tool_observation_count"]),
        "tool_runtime_seconds": total_tool_seconds,
        "tool_runtime_hours": total_tool_seconds / 3600.0,
        "gpu_hours": 0.0,
        "gpu_used": False,
        "checkpoint_storage_bytes": sum(
            path.stat().st_size
            for path in (
                REPO_ROOT / "Output/runs/round23/c1/large_seed3407/last.pt",
                REPO_ROOT / "Output/runs/round23/c2s/large_seed3407/last.pt",
            )
        ),
        "round24_dispatch_storage_bytes": sum(path.stat().st_size for path in storage_paths),
        "queue_and_cpu_accounting": {
            "status": "slurm_accounting_bound",
            "job_ids": ["512625", "512628", "512629", "512630"],
            "job_elapsed_seconds": {"512625": 15, "512628": 7, "512629": 32, "512630": 56},
            "allocated_cpu_hours": (15 * 8 + 7 * 4 + 32 * 8 + 56 * 8) / 3600.0,
            "queue_wait_seconds": {"512625": 0, "512628": 1, "512629": 0, "512630": 0},
        },
        "failed_update_cost": {
            "scientific_candidate_update_failures": 0,
            "invalid_fixture_rollback_count": 1,
            "failed_campaign_elapsed_seconds": 15,
        },
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }


def build() -> dict[str, Any]:
    successor_binding = _bind_round23_successor_sources()
    static = _json(ROUND23_STATIC)
    mechanism = _json(ROUND23_MECHANISM)
    round_rows, tool_rows = _validate_round_artifacts()
    failed_attempt = _failed_attempt_observation()
    diagnostic_boundary = _diagnostic_boundary()
    scientific_metrics = _scientific_metric_status()
    bad_update, rollback = _demonstrate_rollback()

    if mechanism.get("seed_win_fraction") != 0.75:
        raise ValueError("Round-23 mechanism identity changed after Round-24 execution")
    if static.get("neural_twin_checkpoint_update_allowed") is not False:
        raise ValueError("Round-23 static reference no longer freezes the neural checkpoint")

    failure_rows = [
        {
            "attempt_id": "slurm_512625",
            "branch_id": "frozen_twin_frozen_agent",
            "round_index": 2,
            "status": "failed",
            "failure_status": "campaign_failed_during_in_process_tool_reload",
            "failure": failed_attempt["failure_message"],
            "observable_execution_outcome": True,
            "successful_scientific_measurement": False,
            "source_artifact": failed_attempt["stderr"]["path"],
            "source_sha256": failed_attempt["stderr"]["sha256"],
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        }
    ]
    failure_rows.extend(
        {
            "attempt_id": "slurm_512630",
            "branch_id": row["branch_id"],
            "round_index": row["round_index"],
            "status": row["tool_status"],
            "failure_status": "none" if row["tool_status"] in {"completed", "partial"} else "tool_result_failure",
            "failure": row["error"],
            "observable_execution_outcome": True,
            "successful_scientific_measurement": row["tool_status"] in {"completed", "partial"},
            "source_artifact": f"Output/runs/round24/twin_only/{row['branch_id']}/round_{row['round_index']}/tool_result.json",
            "source_sha256": _sha256(AUTHORITATIVE_RUN_ROOT / row["branch_id"] / f"round_{row['round_index']}" / "tool_result.json"),
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        }
        for row in tool_rows
    )
    _write_tsv(
        AUDIT_ROOT / "round24_tool_failure_catalog.tsv",
        list(failure_rows[0]),
        failure_rows,
    )

    metrics_path = AUDIT_ROOT / "round24_scientific_metric_provenance.json"
    _write_json(
        metrics_path,
        {
            "schema_version": "neuronsifter-round24-scientific-metric-provenance-v1",
            "status": "blocked_missing_model_bound_pre_post_evaluation",
            "diagnostic_boundary": diagnostic_boundary,
            "metrics": scientific_metrics,
            "executed_round_rows_contain_placeholder_metrics": True,
            "executed_round_rows_preserved_as_historical_software_evidence": True,
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        },
    )
    export_catalog = _write_evidence_bound_successors(
        round_rows,
        tool_rows,
        metrics_path,
    )

    resource_path = AUDIT_ROOT / "round24_resource_cost.json"
    _write_json(resource_path, _resource_audit(tool_rows, failed_attempt))

    _write_json(AUDIT_ROOT / "round24_bad_update_demonstration.json", bad_update)
    _write_json(AUDIT_ROOT / "round24_rollback_restoration.json", rollback)

    invalidated_path = AUDIT_ROOT / "round24_premature_conditional_execution.json"
    _write_json(
        invalidated_path,
        {
            "schema_version": "neuronsifter-round24-premature-conditional-execution-v1",
            "status": "invalidated_for_scientific_evidence",
            "job_id": "512630",
            "affected_branches": list(CONDITIONAL_BRANCHES),
            "reason": "conditional branches were launched after a Twin gate derived from placeholder metrics",
            "raw_execution_preserved": True,
            "tool_dispatches_remain_valid_observable_development_outcomes": True,
            "agent_or_joint_adaptation_evidence_admissible": False,
            "empirical_coevolution_evidence": False,
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        },
    )

    rollback_passed = all(
        artifact.get(field) is expected
        for artifact in (bad_update, rollback)
        for field, expected in (
            ("candidate_update_attempted", True),
            ("candidate_posterior_created", True),
            ("promotion_rejected", True),
            ("rollback_executed", True),
            ("exact_prior_checkpoint_restored", True),
            ("prior_checkpoint_restored", True),
            ("runtime_capability_restored", True),
            ("posterior_identity_restored", True),
            ("metrics_restored_within_tolerance", True),
            ("biological_state_moved", False),
        )
    )

    twin_gate = {
        "schema_version": "neuronsifter-round24-twin-adaptation-gate-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": BLOCKED_STATUS,
        "prospective_rounds": 3,
        "prospective_dispatch_evidence": {
            "status": "passed_software_and_provenance_only",
            "primary_branch_round_count": sum(row["branch_id"] in PRIMARY_BRANCHES for row in round_rows),
            "primary_branch_count": len(PRIMARY_BRANCHES),
            "selection_before_outcome": all(row["selection_before_outcome"] for row in round_rows if row["branch_id"] in PRIMARY_BRANCHES),
            "fresh_registered_tool_dispatch_count": sum(row["branch_id"] in PRIMARY_BRANCHES for row in tool_rows),
        },
        "twin": {
            "posterior_lineage_updated": True,
            "neural_world_action_checkpoint_updated": False,
            "frozen_reference_preserved": True,
            "model_bound_adaptation_evaluated": False,
        },
        "agent": {
            "parameters_updated_in_primary_campaign": False,
            "conditional_execution_scientifically_admissible": False,
        },
        "decision_consistency": {
            "posterior_identity_trace_preserved": all(row["decision_trace_consistent"] for row in round_rows),
            "model_bound_world_action_consistency_evaluated": False,
        },
        "reliability": {
            "calibration_within_tolerance": None,
            "coverage_floor_respected": None,
            "status": "not_evaluable",
        },
        "retention": {
            "catastrophic_forgetting": None,
            "status": "not_evaluable",
        },
        "rollback": {
            "demonstrated": rollback_passed,
            "scope": "isolated_candidate_update_checkpoint_restore_fixture",
        },
        "blockers": [
            "c1_posterior_not_bound_to_encoder_evaluation",
            "c2s_posterior_not_bound_to_world_action_rollout",
            "world_action_uid_absent_from_prospective_units",
            "pre_post_scientific_metric_inputs_absent",
            "update_strategy_controls_not_executed",
            "conditional_agent_execution_invalidated_by_premature_gate",
        ],
        "scientific_metric_provenance": _artifact_identity(metrics_path),
        "resource_cost": _artifact_identity(resource_path),
        "publication": {
            "protected_holdout_access": 0,
            "external_baseline_comparison_resolved": False,
            "claim_eligible": False,
        },
        "c2b": {"status": C2B_STATUS},
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
    }
    _write_json(AUDIT_ROOT / "round24_twin_adaptation_gate.json", twin_gate)

    _write_json(
        AUDIT_ROOT / "round24_adaptation_matrix_freeze.json",
        {
            "schema_version": "neuronsifter-round24-adaptation-matrix-v2",
            "status": "blocked_after_primary_twin_campaign",
            "systems": list(BRANCHES),
            "primary_systems_executed": list(PRIMARY_BRANCHES),
            "primary_scientific_evaluation_complete": False,
            "conditional_agent_adaptation_authorized": False,
            "conditional_branches_executed_prematurely": list(CONDITIONAL_BRANCHES),
            "conditional_branch_scientific_evidence_admissible": False,
            "candidate_pool_manifest": "Input/manifests/round24/prospective_information_actions.tsv",
            "invalidated_execution_audit": _artifact_identity(invalidated_path),
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        },
    )

    _write_json(
        AUDIT_ROOT / "round24_joint_adaptation_resolution.json",
        {
            "schema_version": "neuronsifter-round24-joint-adaptation-resolution-v2",
            "status": "not_authorized_invalidated_premature_execution",
            "authorized_after_twin_gate": False,
            "joint_adaptation_supported": False,
            "incremental_utility_evaluated": False,
            "empirical_coevolution_evidence": False,
            "recommended_reference": "adaptive_twin_frozen_agent_from_round23_offline_development_only",
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        },
    )

    boundary_path = AUDIT_ROOT / "round24_evidence_boundary.json"
    _write_json(
        boundary_path,
        {
            "schema_version": "neuronsifter-round24-evidence-boundary-v1",
            "status": "closed",
            "development_results_labeled": True,
            "evidence_role": EVIDENCE_ROLE,
            "publication_claim_eligible": False,
            "protected_final_holdout_accessed": False,
            "protected_test_access_count": 0,
            "external_baseline_comparison_resolved": False,
            "headline_comparison_authorized": False,
            "external_baseline_program": "paused_preserve_round20qr_v3_attempts_23_26",
            "round20qr_attempts_preserved": [23, 24, 25, 26],
            "c2b": C2B_STATUS,
        },
    )

    c2b_path = AUDIT_ROOT / "round24_c2b_blocked.json"
    _write_json(
        c2b_path,
        {
            "schema_version": "neuronsifter-round24-c2b-status-v1",
            "status": C2B_STATUS,
            "real_e2_e4_evidence_added": False,
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        },
    )

    program_gate = {
        "schema_version": "neuronsifter-round24-development-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": BLOCKED_STATUS,
        "framework": {"round23_static_reference_frozen": True, "mechanism_status": "partial"},
        "prospective_environment": {
            "registered_tool_capability_frozen": True,
            "primary_sequential_rounds_complete": True,
            "failed_outcomes_visible": True,
        },
        "adaptation": {
            "posterior_assimilation_executed": True,
            "model_bound_pre_post_evaluation_complete": False,
            "twin_adaptation_gate_resolved": False,
            "agent_adaptation_authorized": False,
            "joint_adaptation_supported": False,
        },
        "rollback": {"demonstrated": rollback_passed},
        "validation": {
            "round24_validator_green": False,
            "full_suite_green": False,
            "pending_full_suite_execution": True,
        },
        "evidence_boundary": {
            "development_results_labeled": True,
            "protected_final_holdout_accessed": False,
            "publication_comparison_gate": False,
        },
        "blockers": twin_gate["blockers"],
        "artifacts": {
            "twin_gate": _artifact_identity(AUDIT_ROOT / "round24_twin_adaptation_gate.json"),
            "scientific_metric_provenance": _artifact_identity(metrics_path),
            "evidence_bound_export_catalog": export_catalog,
            "evidence_boundary": _artifact_identity(boundary_path),
            "c2b": _artifact_identity(c2b_path),
            "round23_source_successor_binding": _artifact_identity(
                AUDIT_ROOT / "round24_round23_source_successor_binding.json"
            ),
        },
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "protected_test_access": 0,
        "c2b": {"status": C2B_STATUS},
    }
    _write_json(AUDIT_ROOT / "round24_vnext_development_gate.json", program_gate)
    return program_gate


def main() -> int:
    result = build()
    print(json.dumps({"status": result["status"], "blocker_count": len(result["blockers"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
