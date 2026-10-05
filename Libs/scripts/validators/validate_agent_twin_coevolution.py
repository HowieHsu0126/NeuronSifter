"""Validate the Agent-Twin co-evolution contract and formal round artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_agent_twin_coevolution_contract import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    DEFAULT_SCHEMA,
    REQUIRED_EVALUATION_SYSTEMS,
    REQUIRED_ABLATIONS,
    REQUIRED_BASELINES,
    build_contract,
    load_contract,
    repo_path,
    round_artifact_paths,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Agent-Twin co-evolution contract and round artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--audit", default=DEFAULT_OUTPUT)
    parser.add_argument("--require-rounds", action="store_true")
    parser.add_argument("--require-final-evidence", action="store_true")
    parser.add_argument("--round-root", default="")
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def false_like(value: object) -> bool:
    return value is False or str(value).strip().lower() in {"false", "0", "no"}


def validate_hash_bound_file(path_value: object, expected_sha256: object, label: str) -> Path:
    """Require a concrete artifact and bind its content to the recorded hash."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{label} is missing an artifact path.")
    path = repo_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    observed = sha256_file(path)
    if not isinstance(expected_sha256, str) or expected_sha256 != observed:
        raise ValueError(f"{label} hash does not bind to its artifact.")
    return path


def finite_number(payload: Mapping[str, Any], key: str, label: str) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing a finite {key}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} {key} must be finite.")
    return value


def validate_json_artifact(path: Path, required_fields: list[str], round_index: int, key: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Agent-Twin round artifact: {path}")
    payload = read_json_object(path)
    missing = sorted(set(required_fields) - set(payload))
    if missing:
        raise ValueError(f"{key} is missing required fields: {missing}")
    if int(payload["round_index"]) != round_index:
        raise ValueError(f"{key} round_index mismatch: {payload['round_index']} != {round_index}")
    if "schema_version" not in payload or not str(payload["schema_version"]).startswith("neuronsifter-agent-twin-coevolution"):
        raise ValueError(f"{key} has an invalid schema_version.")
    for holdout_key in ("final_holdout_accessed", "final_holdout_used"):
        if holdout_key in payload and not false_like(payload[holdout_key]):
            raise ValueError(f"{key} must not access or use final held-out evidence.")
    return payload


def validate_tsv_artifact(path: Path, required_fields: list[str], round_index: int, key: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Agent-Twin round artifact: {path}")
    header, rows = read_tsv(path)
    if header != required_fields:
        raise ValueError(f"{key} header mismatch: expected {required_fields}, found {header}")
    if not rows:
        raise ValueError(f"{key} must retain at least one row.")
    for row in rows:
        if int(row["round_index"]) != round_index:
            raise ValueError(f"{key} contains a row from a different round.")
    return rows


def validate_round_artifacts(contract: Mapping[str, Any], artifact_root: Path) -> dict[str, object]:
    protocol = contract["round_protocol"]
    json_fields = protocol["required_json_fields"]
    tsv_fields = protocol["required_tsv_fields"]
    rollback_demonstrated = False
    round_summaries: list[dict[str, object]] = []
    previous_twin_id = str(contract["update_contract"]["initial_checkpoints"]["twin_checkpoint_id"])
    previous_agent_id = str(contract["update_contract"]["initial_checkpoints"]["agent_checkpoint_id"])
    previous_twin_hash: str | None = None
    previous_agent_hash: str | None = None
    for round_index in range(1, int(protocol["minimum_real_round_count"]) + 1):
        paths = round_artifact_paths(contract, artifact_root, round_index)
        input_manifest = validate_json_artifact(paths["input_manifest"], json_fields["input_manifest"], round_index, "input_manifest")
        twin_update = validate_json_artifact(paths["twin_update"], json_fields["twin_update"], round_index, "twin_update")
        agent_update = validate_json_artifact(paths["agent_update"], json_fields["agent_update"], round_index, "agent_update")
        regression = validate_json_artifact(paths["regression_audit"], json_fields["regression_audit"], round_index, "regression_audit")
        promotion = validate_json_artifact(paths["promotion_or_rollback"], json_fields["promotion_or_rollback"], round_index, "promotion_or_rollback")
        report = validate_json_artifact(paths["report"], json_fields["report"], round_index, "report")
        action_rows = validate_tsv_artifact(paths["action_catalog"], tsv_fields["action_catalog"], round_index, "action_catalog")
        evidence_rows = validate_tsv_artifact(paths["evidence_admission"], tsv_fields["evidence_admission"], round_index, "evidence_admission")

        validate_hash_bound_file(
            input_manifest["final_holdout_manifest_path"],
            input_manifest["final_holdout_manifest_sha256"],
            "Round final-holdout manifest",
        )
        validate_hash_bound_file(
            twin_update["candidate_twin_checkpoint_path"],
            twin_update["candidate_twin_checkpoint_sha256"],
            "Twin candidate checkpoint",
        )
        validate_hash_bound_file(
            agent_update["candidate_agent_checkpoint_path"],
            agent_update["candidate_agent_checkpoint_sha256"],
            "Agent candidate checkpoint",
        )
        comparison_manifest = validate_hash_bound_file(
            regression["comparison_manifest_path"],
            regression["comparison_manifest_sha256"],
            "Frozen-vs-evolving comparison manifest",
        )
        comparison_payload = read_json_object(comparison_manifest)

        if str(input_manifest["input_twin_checkpoint_id"]) != previous_twin_id:
            raise ValueError("Round input Twin identity does not chain from the previous promoted identity.")
        if str(input_manifest["input_agent_checkpoint_id"]) != previous_agent_id:
            raise ValueError("Round input Agent identity does not chain from the previous promoted identity.")
        if str(twin_update["input_twin_checkpoint_id"]) != previous_twin_id:
            raise ValueError("Twin candidate update is not bound to the round input identity.")
        if str(agent_update["input_agent_checkpoint_id"]) != previous_agent_id:
            raise ValueError("Agent candidate update is not bound to the round input identity.")

        systems = {str(value) for value in regression["evaluation_systems"]}
        if systems != REQUIRED_EVALUATION_SYSTEMS:
            raise ValueError("Round regression audit does not include the mandatory frozen-vs-evolving 2x2.")
        if regression["identical_evidence_budget_tool_availability"] is not True:
            raise ValueError("Round regression audit does not preserve matched evidence, budget, and tools.")
        comparisons = comparison_payload.get("systems")
        if not isinstance(comparisons, Mapping) or set(comparisons) != REQUIRED_EVALUATION_SYSTEMS:
            raise ValueError("Comparison manifest does not bind the complete frozen-vs-evolving 2x2.")
        invariant_keys = ("hidden_episode_manifest_sha256", "available_evidence_hash", "budget_hash", "tool_availability_hash")
        invariant_values: dict[str, object] = {}
        for system_id, system_result in comparisons.items():
            if not isinstance(system_result, Mapping):
                raise ValueError(f"Comparison manifest result is not an object: {system_id}")
            for key in invariant_keys:
                value = system_result.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Comparison manifest lacks {key} for {system_id}.")
                if key in invariant_values and invariant_values[key] != value:
                    raise ValueError("Frozen-vs-evolving comparisons do not use identical episodes, evidence, budget, and tools.")
                invariant_values[key] = value
        admitted_rows = [row for row in evidence_rows if row["admission_decision"] == "admitted"]
        if not admitted_rows:
            raise ValueError("Each Agent-Twin round must retain admitted evidence.")
        if any(not false_like(row["final_holdout_visible"]) for row in evidence_rows):
            raise ValueError("Evidence admission must not reveal final held-out evidence.")
        if any(row["public_access_status"] != "public" for row in evidence_rows):
            raise ValueError("Evidence admission must remain within the public-only boundary.")
        if any(row["evidence_origin_type"] not in {"registered_tool_output", "legal_public_evidence"} for row in evidence_rows):
            raise ValueError("Evidence admission must identify a registered tool output or legal public evidence source.")
        if any(not row["acquisition_timestamp"].strip() or row["observation_assimilation_gate"] != "pass" for row in admitted_rows):
            raise ValueError("Admitted evidence must pass observation assimilation with an acquisition timestamp.")
        if any(not row["source_identity"].strip() or not row["content_hash"].strip() for row in evidence_rows):
            raise ValueError("Evidence admission must retain source identity and content hash provenance.")
        if any(row["admission_decision"] not in {"admitted", "rejected"} for row in evidence_rows):
            raise ValueError("Evidence admission decisions must be explicit admitted or rejected values.")
        if any(not math.isfinite(float(row["evidence_quality_weight"])) or float(row["evidence_quality_weight"]) <= 0.0 for row in admitted_rows):
            raise ValueError("Admitted evidence must carry a positive finite quality weight.")
        admitted_hashes = {row["content_hash"] for row in admitted_rows}
        if set(twin_update["admitted_evidence_hashes"]) != admitted_hashes:
            raise ValueError("Twin update admitted evidence hashes do not bind to the admission ledger.")
        if set(agent_update["admitted_evidence_hashes"]) != admitted_hashes:
            raise ValueError("Agent update admitted evidence hashes do not bind to the admission ledger.")
        executed_rows = [row for row in action_rows if row["action_status"] in {"executed", "revealed_public_evidence"}]
        if not executed_rows:
            raise ValueError("Each formal round requires a registered execution or legal public evidence reveal.")
        for row in executed_rows:
            if any(row[gate] != "pass" for gate in ("gate_h", "gate_w", "gate_i")):
                raise ValueError("Executed actions must pass all hypothesis/workflow/intervention gates.")
            if row["action_status"] == "executed" and not row["registered_tool_uid"].strip():
                raise ValueError("Executed actions must identify a registered tool.")
            validate_hash_bound_file(row["execution_result_path"], row["execution_result_sha256"], "Action execution result")
        if any(row["action_status"] not in {"proposed", "rejected", "executed", "revealed_public_evidence", "failed"} for row in action_rows):
            raise ValueError("Action catalog uses an unrecognized action status.")
        for row in action_rows:
            if row["action_status"] == "failed":
                if not row["execution_result_path"].strip() or not row["execution_result_sha256"].strip():
                    raise ValueError("Failed tool actions must retain a result artifact and hash.")
                validate_hash_bound_file(row["execution_result_path"], row["execution_result_sha256"], "Failed action result")
        if str(promotion["twin_decision"]) not in {"promote", "rollback"}:
            raise ValueError("Twin promotion artifact must declare promote or rollback.")
        if str(promotion["agent_decision"]) not in {"promote", "rollback"}:
            raise ValueError("Agent promotion artifact must declare promote or rollback.")
        thresholds = contract["update_contract"]
        calibration_drift = finite_number(twin_update, "calibration_drift", "Twin update")
        support_domain_drift = finite_number(twin_update, "support_domain_drift", "Twin update")
        old_domain_retention = finite_number(twin_update, "old_domain_retention", "Twin update")
        new_region_delta = finite_number(twin_update, "new_region_decision_risk_delta", "Twin update")
        worst_source_degradation = finite_number(twin_update, "worst_source_degradation", "Twin update")
        validation_ci_high = finite_number(twin_update, "validation_paired_decision_risk_ci_high", "Twin update")
        promotion_eligible = (
            new_region_delta >= float(thresholds["promotion_thresholds"]["minimum_new_region_decision_risk_delta"])
            and calibration_drift <= float(thresholds["promotion_thresholds"]["maximum_calibration_drift"])
            and support_domain_drift <= float(thresholds["promotion_thresholds"]["maximum_support_domain_drift"])
            and old_domain_retention >= float(thresholds["promotion_thresholds"]["minimum_old_domain_retention"])
            and worst_source_degradation <= float(thresholds["retention_policy"]["max_old_domain_degradation"])
            and validation_ci_high <= 0.0
        )
        if promotion["twin_decision"] == "promote" and not promotion_eligible:
            raise ValueError("Twin promotion violates a frozen promotion threshold.")
        expected_twin_id = str(twin_update["candidate_twin_checkpoint_id"]) if promotion["twin_decision"] == "promote" else previous_twin_id
        expected_agent_id = str(agent_update["candidate_agent_checkpoint_id"]) if promotion["agent_decision"] == "promote" else previous_agent_id
        if str(report["promoted_twin_checkpoint_id"]) != expected_twin_id:
            raise ValueError("Round report Twin identity does not match the promotion or rollback decision.")
        if str(report["promoted_agent_checkpoint_id"]) != expected_agent_id:
            raise ValueError("Round report Agent identity does not match the promotion or rollback decision.")
        is_rollback = promotion["twin_decision"] == "rollback" or promotion["agent_decision"] == "rollback"
        if bool(promotion["rollback_demonstrated"]) != is_rollback:
            raise ValueError("Rollback demonstration must match a recorded rollback decision.")
        if is_rollback:
            if not str(promotion["rollback_reason"]).strip():
                raise ValueError("Rollback must record the failed or injected update reason.")
            if str(promotion["restored_twin_checkpoint_id"]) != previous_twin_id:
                raise ValueError("Rollback must restore the immediately previous promoted Twin identity.")
            if str(promotion["restored_agent_checkpoint_id"]) != previous_agent_id:
                raise ValueError("Rollback must restore the immediately previous promoted Agent identity.")
            if previous_twin_hash and str(promotion["restored_twin_checkpoint_sha256"]) != previous_twin_hash:
                raise ValueError("Rollback does not restore the previous Twin checkpoint hash.")
            if previous_agent_hash and str(promotion["restored_agent_checkpoint_sha256"]) != previous_agent_hash:
                raise ValueError("Rollback does not restore the previous Agent checkpoint hash.")
        rollback_demonstrated = rollback_demonstrated or is_rollback
        round_summaries.append(
            {
                "round_index": round_index,
                "input_twin_checkpoint_id": input_manifest["input_twin_checkpoint_id"],
                "candidate_twin_checkpoint_id": twin_update["candidate_twin_checkpoint_id"],
                "candidate_agent_checkpoint_id": agent_update["candidate_agent_checkpoint_id"],
                "promoted_twin_checkpoint_id": report["promoted_twin_checkpoint_id"],
                "promoted_agent_checkpoint_id": report["promoted_agent_checkpoint_id"],
                "twin_decision": promotion["twin_decision"],
                "agent_decision": promotion["agent_decision"],
                "admitted_evidence_count": len(admitted_rows),
            }
        )
        previous_twin_id = expected_twin_id
        previous_agent_id = expected_agent_id
        if promotion["twin_decision"] == "promote":
            previous_twin_hash = str(twin_update["candidate_twin_checkpoint_sha256"])
        if promotion["agent_decision"] == "promote":
            previous_agent_hash = str(agent_update["candidate_agent_checkpoint_sha256"])
    if not rollback_demonstrated:
        raise ValueError("Formal Agent-Twin co-evolution evidence requires a demonstrated rollback.")
    return {
        "round_count": len(round_summaries),
        "rollback_demonstrated": rollback_demonstrated,
        "rounds": round_summaries,
    }


def validate_final_evidence(contract: Mapping[str, Any], prerequisite_snapshot_hash: str) -> dict[str, object]:
    final_config = contract["final_evidence"]
    path = repo_path(str(final_config["artifact_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing final Agent-Twin co-evolution evidence artifact: {path}")
    payload = read_json_object(path)
    missing = sorted(set(final_config["required_fields"]) - set(payload))
    if missing:
        raise ValueError(f"Final Agent-Twin evidence artifact is missing fields: {missing}")
    if str(payload["final_gate"]) != str(contract["final_gate"]) or str(payload["status"]) != "passed":
        raise ValueError("Final Agent-Twin evidence does not resolve the configured final gate.")
    if int(payload["real_round_count"]) < int(contract["round_protocol"]["minimum_real_round_count"]):
        raise ValueError("Final Agent-Twin evidence has fewer than three real sequential rounds.")
    if str(payload["prerequisite_snapshot_hash"]) != prerequisite_snapshot_hash:
        raise ValueError("Final Agent-Twin evidence is not bound to the current prerequisite snapshot.")
    if payload["fixed_budget_direction"] != "favors_governed_agent_twin_coevolution":
        raise ValueError("Fixed-budget evidence does not favor governed Agent-Twin co-evolution.")
    if payload["fixed_risk_direction"] != "favors_governed_agent_twin_coevolution":
        raise ValueError("Fixed-risk evidence does not favor governed Agent-Twin co-evolution.")
    for key in ("old_domain_retention", "calibration_and_support_domain", "evidence_reproducibility", "rollback"):
        if not isinstance(payload[key], Mapping) or payload[key].get("passed") is not True:
            raise ValueError(f"Final Agent-Twin evidence has not passed {key}.")

    holdout = payload["final_holdout_evaluation"]
    if not isinstance(holdout, Mapping):
        raise ValueError("Final Agent-Twin evidence must record final-holdout isolation.")
    validate_hash_bound_file(holdout.get("manifest_path"), holdout.get("manifest_sha256"), "Final-holdout manifest")
    if holdout.get("used_for_final_evaluation") is not True:
        raise ValueError("Final Agent-Twin evidence must bind the final-only evaluation manifest.")
    for key in ("used_for_updates", "used_for_checkpoint_selection", "used_for_threshold_selection"):
        if not false_like(holdout.get(key)):
            raise ValueError(f"Final held-out evidence was used for forbidden {key}.")

    reproducibility = payload["evidence_reproducibility"]
    replay_hashes = reproducibility.get("round_replay_hashes") if isinstance(reproducibility, Mapping) else None
    if not isinstance(replay_hashes, list) or len(replay_hashes) < int(contract["round_protocol"]["minimum_real_round_count"]) or any(not str(value) for value in replay_hashes):
        raise ValueError("Final Agent-Twin evidence lacks replay hashes for every real round.")
    if not str(reproducibility.get("config_hash", "")) or not str(reproducibility.get("code_revision", "")):
        raise ValueError("Final Agent-Twin evidence lacks config and code reproducibility bindings.")

    rollback = payload["rollback"]
    if not isinstance(rollback, Mapping) or not str(rollback.get("failure_kind", "")) or not str(rollback.get("restored_twin_checkpoint_id", "")) or not str(rollback.get("restored_agent_checkpoint_id", "")):
        raise ValueError("Final Agent-Twin evidence lacks a reproducible rollback record.")

    comparisons = payload["baseline_comparisons"]
    if not isinstance(comparisons, Mapping):
        raise ValueError("Final Agent-Twin evidence baseline comparisons must be a mapping.")
    required_comparisons = {"strongest_static_baseline", "static_twin_adaptive_agent", "evolving_twin_static_agent"}
    if set(comparisons) != required_comparisons:
        raise ValueError("Final Agent-Twin evidence is missing required static and one-sided comparisons.")
    for comparison in comparisons.values():
        if not isinstance(comparison, Mapping):
            raise ValueError("Agent-Twin baseline comparison must be a mapping.")
        ci_high = finite_number(comparison, "paired_decision_risk_ci_high", "Baseline comparison")
        if not bool(comparison.get("passed")) or ci_high >= 0.0:
            raise ValueError("Agent-Twin baseline comparison does not show a paired decision-risk improvement.")

    def validate_result_rows(
        rows: object,
        id_key: str,
        expected_ids: set[str],
        label: str,
        allowed_statuses: set[str],
    ) -> None:
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError(f"{label} must be a list of result records.")
        ids = [str(row.get(id_key, "")) for row in rows]
        if set(ids) != expected_ids or len(ids) != len(expected_ids):
            raise ValueError(f"{label} omits or duplicates required records.")
        for row in rows:
            if str(row.get("status", "")) not in allowed_statuses:
                raise ValueError(f"{label} contains an incomplete result.")
            validate_hash_bound_file(row.get("result_path"), row.get("result_sha256"), f"{label} result")

    baseline_rows = payload["baseline_results"]
    validate_result_rows(baseline_rows, "baseline_id", REQUIRED_BASELINES, "Baseline results", {"completed", "evaluation_only"})
    for row in baseline_rows:
        if row["baseline_id"] != "full_information_oracle" and row["status"] != "completed":
            raise ValueError("Every deployable baseline must complete its matched sequential evaluation.")
        if row["baseline_id"] == "full_information_oracle" and row["status"] != "evaluation_only":
            raise ValueError("Full-information oracle must remain evaluation-only.")

    control_ids = {str(value) for value in contract["evaluation"]["continual_learning_controls"]}
    validate_result_rows(payload["continual_learning_controls"], "control_id", control_ids, "Continual-learning controls", {"completed"})

    thresholds = contract["update_contract"]
    retention = payload["old_domain_retention"]
    if finite_number(retention, "minimum_observed", "Old-domain retention") < float(thresholds["promotion_thresholds"]["minimum_old_domain_retention"]):
        raise ValueError("Old-domain retention is below the frozen promotion tolerance.")
    if finite_number(retention, "worst_source_degradation", "Old-domain retention") > float(thresholds["retention_policy"]["max_old_domain_degradation"]):
        raise ValueError("Worst-source degradation exceeds the frozen tolerance.")
    calibration = payload["calibration_and_support_domain"]
    if finite_number(calibration, "calibration_drift", "Calibration evidence") > float(thresholds["promotion_thresholds"]["maximum_calibration_drift"]):
        raise ValueError("Calibration drift exceeds the frozen promotion tolerance.")
    if finite_number(calibration, "support_domain_drift", "Calibration evidence") > float(thresholds["promotion_thresholds"]["maximum_support_domain_drift"]):
        raise ValueError("Support-domain drift exceeds the frozen promotion tolerance.")

    ablations = payload["ablations"]
    validate_result_rows(ablations, "ablation_id", REQUIRED_ABLATIONS, "Ablation results", {"completed", "failed"})
    if any(row.get("evidence_retained") is not True or row.get("executed") is not True for row in ablations):
        raise ValueError("Final Agent-Twin evidence must execute and retain every ablation, including failures.")

    statistics = payload["statistics"]
    if not isinstance(statistics, Mapping) or statistics.get("confidence_interval") != "paired_bootstrap_95":
        raise ValueError("Final Agent-Twin evidence does not use paired bootstrap confidence intervals.")
    if int(statistics.get("bootstrap_replicates", 0)) != int(contract["evaluation"]["statistics"]["bootstrap_replicates"]):
        raise ValueError("Final Agent-Twin evidence does not use the frozen bootstrap replicate count.")
    if int(statistics.get("bootstrap_seed", -1)) != int(contract["evaluation"]["statistics"]["bootstrap_seed"]):
        raise ValueError("Final Agent-Twin evidence does not use the frozen bootstrap seed.")
    if statistics.get("paired_sequential_unit") != contract["evaluation"]["statistics"]["paired_sequential_unit"]:
        raise ValueError("Final Agent-Twin evidence does not use the frozen paired sequential unit.")
    validate_hash_bound_file(statistics.get("paired_unit_manifest_path"), statistics.get("paired_unit_manifest_sha256"), "Paired bootstrap unit manifest")
    return {"path": str(final_config["artifact_path"]), "status": "passed", "real_round_count": payload["real_round_count"]}


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    audit_path: str | Path = DEFAULT_OUTPUT,
    require_rounds: bool = False,
    round_root: str | Path | None = None,
    require_final_evidence: bool = False,
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    try:
        expected = build_contract(config_path, schema_path)
    except Exception as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "invalid_contract", "message": str(exc)}]}

    audit_file = repo_path(audit_path)
    if not audit_file.is_file():
        blockers.append({"code": "missing_required_audit", "message": str(audit_path)})
    else:
        try:
            actual = read_json_object(audit_file)
        except Exception as exc:
            blockers.append({"code": "invalid_audit_json", "message": str(exc)})
        else:
            if actual != expected:
                blockers.append({"code": "stale_or_tampered_audit", "message": str(audit_path)})

    round_summary: dict[str, object] | None = None
    if require_rounds or require_final_evidence:
        if not bool(expected["formal_execution_allowed"]):
            blockers.append({"code": "formal_execution_not_authorized", "message": "Prerequisites are not all satisfied."})
        try:
            contract = load_contract(config_path)
            root = repo_path(round_root or str(contract["round_protocol"]["artifact_root"]))
            round_summary = validate_round_artifacts(contract, root)
        except Exception as exc:
            blockers.append({"code": "invalid_round_artifacts", "message": str(exc)})

    final_evidence_summary: dict[str, object] | None = None
    if require_final_evidence:
        try:
            final_evidence_summary = validate_final_evidence(load_contract(config_path), str(expected["prerequisite_snapshot_hash"]))
        except Exception as exc:
            blockers.append({"code": "invalid_final_evidence", "message": str(exc)})

    final_gate_passed = bool(
        require_final_evidence
        and not blockers
        and bool(expected["formal_execution_allowed"])
        and final_evidence_summary is not None
        and final_evidence_summary.get("status") == "passed"
    )
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "contract_status": expected["contract_status"],
        "formal_execution_allowed": expected["formal_execution_allowed"],
        "final_gate": expected["final_gate"],
        # The precondition audit remains false until empirical evidence is
        # requested and independently validated in this resolver.
        "final_gate_passed": final_gate_passed,
        "round_summary": round_summary,
        "final_evidence_summary": final_evidence_summary,
    }


def main() -> None:
    args = parse_args()
    result = validate(
        args.config,
        args.schema,
        args.audit,
        args.require_rounds,
        args.round_root or None,
        args.require_final_evidence,
    )
    print(json.dumps(result, indent=2, sort_keys=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
