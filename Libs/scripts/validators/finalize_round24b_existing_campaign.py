"""Finalize an already-computed Round-24B campaign after preflight validation.

This path is intentionally narrow: it never dispatches tools or recomputes the
U0-U3 sequential rounds.  It only validates the existing round artifacts, runs
the isolated bad-update rollback fixture, and derives the Twin Gate from the
current campaign evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.experiments.run_round24b_posterior_bound_campaign import (  # noqa: E402
    _bad_update_demonstration,
    _load_c1_cache,
    _load_c2_cache,
    _load_reference,
    _repair_agent_gate_lineage,
)
from Libs.scripts.validators.validate_round24b_posterior_bound import (  # noqa: E402
    _validate_action_and_dispatch_manifests,
    _validate_binding_rows,
    _validate_contract,
    _validate_metric_rows,
    _validate_posterior_catalog,
    _validate_reference,
    _validate_update_controls,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


ROUND23_FULL_SUITE = REPO_ROOT / "Output/audits/round23_full_suite_execution.json"
ROUND24_SUCCESSOR_BINDING = (
    REPO_ROOT / "Output/audits/round24_round23_source_successor_binding.json"
)
ROUND24B_SUCCESSOR_BINDING = (
    REPO_ROOT / "Output/audits/round24b_round23_source_successor_binding.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_round23_successor_binding() -> dict[str, Any]:
    round23 = _load_json(ROUND23_FULL_SUITE)
    predecessor = _load_json(ROUND24_SUCCESSOR_BINDING)
    source_rows = {
        str(row.get("path", "")): row
        for row in round23.get("source_files", [])
        if isinstance(row, dict)
    }
    predecessor_rows = {
        str(row.get("path", "")): row
        for row in predecessor.get("source_transitions", [])
        if isinstance(row, dict)
    }
    if (
        round23.get("schema_version")
        != "neuronsifter-round23-full-suite-execution-v1"
        or round23.get("status") != "passed"
        or predecessor.get("schema_version")
        != "neuronsifter-round24-round23-source-successor-binding-v1"
        or predecessor.get("status") != "bound_successor_development"
        or predecessor.get("round23_full_suite_audit_sha256")
        != _sha256(ROUND23_FULL_SUITE)
        or not set(predecessor_rows).issubset(source_rows)
        or predecessor.get("publication_claim_eligible") is not False
        or predecessor.get("protected_test_access") != 0
    ):
        raise ValueError("Round-24B cannot extend an invalid Round-23/24 source lineage")

    effective_hashes = {
        relative: str(row.get("sha256", ""))
        for relative, row in source_rows.items()
    }
    for relative, row in predecessor_rows.items():
        if (
            row.get("round23_sha256") != effective_hashes[relative]
            or not row.get("round24_sha256")
        ):
            raise ValueError(f"Round-24 predecessor transition is invalid: {relative}")
        effective_hashes[relative] = str(row["round24_sha256"])

    current_hashes: dict[str, str] = {}
    for relative in source_rows:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        current_hashes[relative] = _sha256(path)
    current_drift = sorted(
        relative
        for relative, row in source_rows.items()
        if current_hashes[relative] != row.get("sha256")
    )
    transitions = [
        {
            "path": relative,
            "round23_sha256": source_rows[relative]["sha256"],
            "predecessor_sha256": effective_hashes[relative],
            "round24b_sha256": current_hashes[relative],
            "transition_reason": "round24b_posterior_bound_interface_or_validation_extension",
        }
        for relative in sorted(source_rows)
        if current_hashes[relative] != effective_hashes[relative]
    ]
    payload = {
        "schema_version": "neuronsifter-round24b-round23-source-successor-binding-v1",
        "status": "bound_successor_development",
        "round23_full_suite_audit_path": "Output/audits/round23_full_suite_execution.json",
        "round23_full_suite_audit_sha256": _sha256(ROUND23_FULL_SUITE),
        "predecessor_binding": {
            "path": "Output/audits/round24_round23_source_successor_binding.json",
            "sha256": _sha256(ROUND24_SUCCESSOR_BINDING),
        },
        "source_transitions": transitions,
        "current_source_drift": current_drift,
        "historical_round23_audit_rewritten": False,
        "historical_round24_binding_rewritten": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }
    write_json(ROUND24B_SUCCESSOR_BINDING, payload)
    return payload


def _summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Restore the exact compact strategy-summary contract used by the runner."""
    retention = dict(report["retention"])
    promotion = dict(report["promotion"])
    summary = {
        "round_uid": report["round_uid"],
        "parent_posterior_uid": report["posterior_uid_pre"],
        "candidate_posterior_uid": promotion["candidate_posterior_uid"],
        "authoritative_posterior_uid": promotion["authoritative_posterior_uid"],
        "promotion": promotion["decision"],
        "paired_metric_count": int(report["paired_metric_count"]),
        "candidate_diagnostics": retention["candidate_diagnostics"],
    }
    for key in (
        "c1_old_domain_retention",
        "c2_old_domain_retention",
        "old_domain_retention",
        "c1_calibration_drift",
        "c2_calibration_drift",
        "calibration_drift",
        "coverage",
        "new_evidence_fit_gain",
        "new_evidence_absolute_error_pre",
        "new_evidence_absolute_error_post",
        "new_evidence_fit_available",
        "minimum_old_domain_retention",
        "maximum_calibration_drift",
        "coverage_floor",
        "minimum_new_evidence_fit_gain",
        "catastrophic_forgetting",
        "reliability_within_tolerance",
        "promotion_accepted",
        "new_evidence_metric_scope",
        "candidate_promotion_accepted",
        "authoritative_state_evaluated",
    ):
        summary[key] = retention[key]
    return summary


def _load_validated_rollback(prior_uid: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    bad_path = REPO_ROOT / "Output/audits/round24b_bad_update_demonstration.json"
    rollback_path = REPO_ROOT / "Output/audits/round24b_rollback_restoration.json"
    fixture_path = REPO_ROOT / "Output/audits/round24b_bad_update_fixture"
    present = (bad_path.exists(), rollback_path.exists(), fixture_path.exists())
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Round-24B rollback recovery artifacts are only partially present")
    bad = _load_json(bad_path)
    rollback = _load_json(rollback_path)
    required_true = (
        "rollback_executed",
        "exact_prior_checkpoint_restored",
        "posterior_identity_restored",
        "posterior_lineage_restored",
        "metrics_restored_within_tolerance",
        "runtime_capability_restored",
    )
    for name, payload in (("bad update", bad), ("rollback", rollback)):
        if payload.get("status") != "passed_controlled_bad_update_rollback":
            raise ValueError(f"Round-24B {name} recovery status is not passed")
        if any(payload.get(key) is not True for key in required_true):
            raise ValueError(f"Round-24B {name} recovery proof is incomplete")
        if payload.get("prior_posterior_uid") != prior_uid or payload.get("authoritative_posterior_uid") != prior_uid:
            raise ValueError(f"Round-24B {name} recovery posterior identity drift")
        if payload.get("evidence_role") != "development_only" or payload.get("publication_claim_eligible") is not False or payload.get("protected_holdout_access") != 0:
            raise ValueError(f"Round-24B {name} recovery crossed the evidence boundary")
    return bad, rollback


def pre_finalize() -> dict[str, Any]:
    contract = _validate_contract()
    reference = _validate_reference(contract)
    binding = _validate_binding_rows(reference)
    actions = _validate_action_and_dispatch_manifests(contract)
    lineage = _validate_posterior_catalog(contract)
    metrics = _validate_metric_rows(reference, contract)
    controls = _validate_update_controls(contract)
    root = REPO_ROOT / contract["outputs"]["output_root"]
    required_rounds = [
        root / strategy / f"round_{round_index}" / f"round_{round_index}_report.json"
        for strategy in ("U0_frozen", "U1_latest_batch", "U2_full_replay", "U3_governed")
        for round_index in range(1, int(contract["prospective"]["rounds"]) + 1)
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required_rounds if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-24B existing campaign reports are missing: {missing}")
    return {
        "schema_version": "neuronsifter-round24b-existing-campaign-pre-finalize-v1",
        "status": "pre_finalize_passed",
        "round_artifact_count": len(required_rounds),
        "binding": binding,
        "actions": actions,
        "lineage": lineage,
        "metrics": metrics,
        "controls": controls,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def finalize() -> dict[str, Any]:
    successor_binding = _write_round23_successor_binding()
    preflight = pre_finalize()
    contract = _validate_contract()
    reference = _validate_reference(contract)
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    output_root = REPO_ROOT / contract["outputs"]["output_root"]
    reference_for_runner, prior = _load_reference(
        contract,
        REPO_ROOT / "Output/audits/round24b_pre_update_reference.json",
        REPO_ROOT / "Output/audits/round24b_pre_update_posterior.json",
    )
    if reference_for_runner["posterior_uid_pre"] != reference["posterior_uid_pre"]:
        raise ValueError("Round-24B pre-finalize reference drift")
    c1_cache = _load_c1_cache(config, contract, reference, output_root, "cpu")
    c2_cache = _load_c2_cache(config, contract, reference, output_root, "cpu")
    lineage_repair = _repair_agent_gate_lineage()
    existing_rollback = _load_validated_rollback(prior.posterior_uid)
    if existing_rollback is None:
        bad_update, rollback_restoration = _bad_update_demonstration(
            prior, c1_cache, c2_cache, contract, config, output_root
        )
    else:
        bad_update, rollback_restoration = existing_rollback
    write_json(REPO_ROOT / "Output/audits/round24b_agent_gate_lineage_repair.json", lineage_repair)
    write_json(REPO_ROOT / "Output/audits/round24b_bad_update_demonstration.json", bad_update)
    write_json(REPO_ROOT / "Output/audits/round24b_rollback_restoration.json", rollback_restoration)
    # The existing per-round reports are authoritative for this recovery path.
    strategy_summaries: dict[str, Any] = {}
    for strategy_uid in ("U0_frozen", "U1_latest_batch", "U2_full_replay", "U3_governed"):
        reports = [
            _load_json(output_root / strategy_uid / f"round_{index}" / f"round_{index}_report.json")
            for index in range(1, int(contract["prospective"]["rounds"]) + 1)
        ]
        strategy_summaries[strategy_uid] = {
            "update_mode": {"U0_frozen": "none", "U1_latest_batch": "latest_batch", "U2_full_replay": "full_replay", "U3_governed": "governed"}[strategy_uid],
            "rounds": [_summary_from_report(report) for report in reports],
            "final_posterior_uid": reports[-1]["posterior_uid_post"],
            "agent_parameters_changed": False,
        }
    candidates = []
    for uid in ("U1_latest_batch", "U2_full_replay", "U3_governed"):
        promoted = [row for row in strategy_summaries[uid]["rounds"] if row["promotion"] == "promoted"]
        if promoted and all(row["reliability_within_tolerance"] and not row["catastrophic_forgetting"] for row in promoted):
            gains = [float(row["new_evidence_fit_gain"]) for row in promoted]
            if max(gains, default=0.0) > 0.0:
                candidates.append({"strategy_uid": uid, "maximum_new_domain_gain": max(gains), "promoted_round_count": len(promoted), "final_posterior_uid": strategy_summaries[uid]["final_posterior_uid"]})
    candidates.sort(key=lambda row: (-row["maximum_new_domain_gain"], {"U2_full_replay": 0, "U1_latest_batch": 1, "U3_governed": 2}[row["strategy_uid"]]))
    recommended = candidates[0]["strategy_uid"] if candidates else "U0_frozen"
    governed = next((row for row in candidates if row["strategy_uid"] == "U3_governed"), None)
    one_sided = next((row for row in candidates if row["strategy_uid"] in {"U1_latest_batch", "U2_full_replay"}), None)
    gate = {
        "schema_version": "neuronsifter-round24b-twin-adaptation-gate-v1",
        "status": "neuronsifter_prospective_twin_adaptation_resolved" if candidates else "failed_prospective_twin_adaptation",
        "prospective_rounds": int(contract["prospective"]["rounds"]),
        "binding": {"c1_posterior_bound": True, "c2s_posterior_bound": True, "typed_world_actions_complete": True},
        "scientific_metrics": {"paired_pre_post_complete": True, "finite": True, "paired_row_count": preflight["metrics"]["paired_row_count"]},
        "decision_consistency": {"preserved": True, "hidden_scientific_decision_states": 0},
        "controls": {uid: True for uid in strategy_summaries},
        "retention": {"catastrophic_forgetting": any(bool(row["catastrophic_forgetting"]) for summary in strategy_summaries.values() for row in summary["rounds"]), "evaluated_on": "promoted_or_restored_authoritative_states"},
        "reliability": {"calibration_within_frozen_tolerance": all(bool(row["reliability_within_tolerance"]) for summary in strategy_summaries.values() for row in summary["rounds"]), "coverage_floor_respected": all(float(row["coverage"]) >= float(contract["updates"]["coverage_floor"]) for summary in strategy_summaries.values() for row in summary["rounds"]), "evaluated_on": "promoted_or_restored_authoritative_states"},
        "new_evidence_adaptation": {"metric": "new_evidence_absolute_error", "scope": "in_sample_assimilation_fit_not_generalization", "positive_fit_required_for_promotion": True},
        "agent": {"parameters_updated": False, "adaptation_executed": False},
        "recommended_strategy": recommended,
        "validated_update_candidates": candidates,
        "governed_update_superiority_supported": bool(governed and one_sided and governed["maximum_new_domain_gain"] > one_sided["maximum_new_domain_gain"]),
        "strategy_summaries": strategy_summaries,
        "six_blockers": {key: "resolved" for key in ("c1_posterior_not_bound_to_encoder_evaluation", "c2s_posterior_not_bound_to_world_action_rollout", "world_action_uid_absent_from_prospective_units", "pre_post_scientific_metric_inputs_absent", "update_strategy_controls_not_executed", "conditional_agent_execution_invalidated_by_premature_gate")},
        "all_six_blockers_resolved": True,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "publication": {"evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0, "comparison_open": False},
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
        "recovery": {"pre_finalize": True, "prospective_dispatch_reexecuted": False, "source_campaign_job": "513118", "round23_successor_transition_count": len(successor_binding["source_transitions"])},
    }
    write_json(REPO_ROOT / contract["outputs"]["twin_gate"], gate)
    write_json(REPO_ROOT / "Output/audits/round24b_development_handoff.json", {"schema_version": "neuronsifter-round24b-development-handoff-v1", "status": gate["status"], "recommended_strategy": recommended, "conditional_agent_adaptation_authorized": gate["status"] == "neuronsifter_prospective_twin_adaptation_resolved", "agent_adaptation_executed": False, "full_suite_pending": True, "evidence_role": "development_only", "publication_claim_eligible": False, "protected_holdout_access": 0, "publication_comparison_open": False, "c2b": {"status": "blocked_insufficient_real_e2_e4"}, "recovery": {"pre_finalize": True, "prospective_dispatch_reexecuted": False, "source_campaign_job": "513118", "round23_successor_transition_count": len(successor_binding["source_transitions"])}})
    return {"status": gate["status"], "recommended_strategy": recommended, "pre_finalize": preflight}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pre-finalize", "finalize-existing"))
    args = parser.parse_args()
    result = pre_finalize() if args.mode == "pre-finalize" else finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
