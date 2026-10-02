"""Validate development-only Agent-Twin WP2 qualification artifacts.

This validator deliberately recognizes technical rehearsal only.  It never
changes the formal co-evolution contract, treats a missing Core-13 frontier as
the static-baseline blocker, and rejects any formal-holdout or self-rollout
training-label exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.exp.agent_twin_wp2_development import (
    BEHAVIOR_CHANGING_OUTCOMES,
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    canonical_hash,
    execution_identity,
    inspect_core13_frontier,
    promotion_eligible,
    resolved_initial_twin_checkpoint_id,
    sha256_file,
    validate_config,
    validate_formal_boundary,
    validate_wp1_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate development-only Agent-Twin WP2 artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing append-only ledger: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank row in append-only ledger {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Non-object row in append-only ledger {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Append-only ledger is empty: {path}")
    return rows


def require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label}: expected {expected!r}, observed {observed!r}")


def require_false(value: object, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label} must remain false.")


def require_file_hash(path_value: object, expected_hash: object, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is missing.")
    path = Path(path_value)
    if not path.is_absolute():
        path = resolve_repo_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def verify_checksum_catalog(root: Path) -> int:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum catalog: {checksum_path}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            digest, relative_path = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum row {checksum_path}:{line_number}") from exc
        if len(digest) != 64 or relative_path in expected:
            raise ValueError(f"Invalid checksum row {checksum_path}:{line_number}")
        expected[relative_path] = digest
    observed_paths = {
        item.relative_to(root).as_posix(): item
        for item in root.rglob("*")
        if item.is_file() and item.name != "checksums.sha256"
    }
    if set(expected) != set(observed_paths):
        raise ValueError("Checksum catalog does not cover exactly the emitted WP2 artifacts.")
    for relative_path, path in observed_paths.items():
        if sha256_file(path) != expected[relative_path]:
            raise ValueError(f"Checksum mismatch for {relative_path}")
    return len(observed_paths)


def group_by(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def validate_event_order(
    events: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    evidence_updates: list[dict[str, Any]],
    rounds: int,
) -> None:
    expected_types = ["selection_freeze", "development_outcome", "evidence_update", "mioy_version", "emp_version"]
    by_run = group_by(events, "run_uid")
    if len(by_run) != 10:
        raise ValueError("WP2 must retain ten branch-specific runs for five matched worlds.")
    selection_hashes = {str(row.get("selection_hash", "")) for row in selections}
    outcome_index = {str(row.get("outcome_hash", "")): row for row in outcomes}
    for (run_uid,), run_events in by_run.items():
        ordered = sorted(run_events, key=lambda row: int(row.get("event_sequence", 0)))
        if [int(row.get("event_sequence", 0)) for row in ordered] != list(range(1, rounds * len(expected_types) + 1)):
            raise ValueError(f"Event sequence is not append-only for run {run_uid}.")
        if [str(row.get("event_type", "")) for row in ordered] != expected_types * rounds:
            raise ValueError(f"Selection/outcome/evidence ordering is invalid for run {run_uid}.")
    for outcome in outcomes:
        if str(outcome.get("selection_hash", "")) not in selection_hashes:
            raise ValueError("Outcome is not bound to a prior selection freeze.")
        if int(outcome.get("event_sequence", 0)) <= 1:
            raise ValueError("Development outcome precedes selection freeze.")
        if str(outcome.get("outcome_origin", "")) != "transparent_development_generator_outcome":
            raise ValueError("WP2 outcome does not use the registered transparent development generator.")
        if outcome.get("verified_update_label") is not True or outcome.get("twin_rollout_used_as_label") is not False:
            raise ValueError("WP2 outcome label is not a verified non-Twin label.")
        require_false(outcome.get("formal_holdout_visible"), "Development outcome formal holdout visibility")
        require_false(outcome.get("evaluator_only_field_visible"), "Evaluator-only field visibility")
        source_payload = {
            "schema_version": "neuronsifter-agent-twin-wp2-development-source-outcome-v1",
            "world_id": outcome["world_id"],
            "round_index": outcome["round_index"],
            "selected_action": next(row["selected_action"] for row in selections if row["selection_hash"] == outcome["selection_hash"]),
            "outcome_label": outcome["outcome_label"],
            "outcome_origin": outcome["outcome_origin"],
            "outcome_generator_registration": outcome["outcome_generator_registration"],
            "generator_family": outcome["generator_family"],
            "generator_seed": outcome["generator_seed"],
            "fixture_rehearsal_only": True,
        }
        require_equal(outcome.get("source_outcome_hash"), canonical_hash(source_payload), "Canonical source outcome hash")
    for (world_id, round_index), paired_outcomes in group_by(outcomes, "world_id", "round_index").items():
        if len(paired_outcomes) != 2 or {row.get("branch") for row in paired_outcomes} != {"static", "coevolution"}:
            raise ValueError(f"Source outcome pair is incomplete for {world_id} round {round_index}.")
        source_hashes = {str(row.get("source_outcome_hash", "")) for row in paired_outcomes}
        source_uids = {str(row.get("source_outcome_uid", "")) for row in paired_outcomes}
        if len(source_hashes) != 1 or len(source_uids) != 1:
            raise ValueError("Matched branches do not bind to one canonical source outcome.")
    for evidence in evidence_updates:
        outcome = outcome_index.get(str(evidence.get("outcome_hash", "")))
        if outcome is None:
            raise ValueError("Evidence update is not bound to a development outcome.")
        for key in ("run_uid", "branch", "world_id", "round_index", "source_outcome_uid", "source_outcome_hash", "outcome_origin"):
            require_equal(evidence.get(key), outcome.get(key), f"Evidence update {key} binding")
        if evidence.get("observation_assimilation_gate") != "pass" or evidence.get("admission_decision") != "admitted":
            raise ValueError("WP2 evidence did not pass the observation-assimilation admission gate.")
        if evidence.get("twin_rollout_used_as_label") is not False:
            raise ValueError("Twin rollout entered the evidence update label path.")
        require_false(evidence.get("formal_holdout_visible"), "Evidence update formal holdout visibility")
        expected_hash = canonical_hash({key: value for key, value in evidence.items() if key != "evidence_hash"})
        require_equal(evidence.get("evidence_hash"), expected_hash, "Evidence update content hash")


def validate_run_manifests(root: Path, payload: Mapping[str, Any], initial_twin_checkpoint_id: str) -> tuple[list[dict[str, Any]], bool]:
    protocol = payload["protocol"]
    manifest_paths = sorted((root / "sequential_evidence" / "sequential_runs").glob("*/run_manifest.json"))
    if len(manifest_paths) != 10:
        raise ValueError("Expected five static and five co-evolution run manifests.")
    manifests = [read_json(path) for path in manifest_paths]
    required_worlds = {str(row["world_id"]) for row in payload["development_worlds"]}
    by_world = group_by(manifests, "world_id")
    if set(world for (world,) in by_world) != required_worlds:
        raise ValueError("Development worlds do not match the frozen WP2 configuration.")
    behavior_changed = False
    for (world_id,), rows in by_world.items():
        if len(rows) != 2 or {row.get("branch") for row in rows} != {"static", "coevolution"}:
            raise ValueError(f"World {world_id} is missing a matched branch.")
        static, coevolution = sorted(rows, key=lambda row: str(row["branch"]))
        for row in rows:
            require_equal(row.get("round_count"), protocol["rounds_per_run"], "Run round count")
            require_equal(row.get("initial_twin_checkpoint_id"), initial_twin_checkpoint_id, "Initial Twin identity")
            require_equal(row.get("initial_agent_checkpoint_id"), protocol["initial_agent_checkpoint_id"], "Initial Agent identity")
            require_equal(row.get("initial_memory_id"), protocol["initial_memory_id"], "Initial memory identity")
            require_equal(row.get("outcome_origin"), protocol["outcome_origin"], "Outcome origin")
            require_false(row.get("twin_rollout_used_as_label"), "Twin rollout use as label")
            require_equal(row.get("formal_holdout_access_count"), 0, "Formal holdout access count")
        if static["outcome_schedule"] != coevolution["outcome_schedule"]:
            raise ValueError(f"World {world_id} has different static and co-evolution outcomes.")
        behavior_changed = behavior_changed or any(bool(row.get("behavior_changed_after_evidence")) for row in rows)
    return manifests, behavior_changed


def validate_updates(
    root: Path,
    payload: Mapping[str, Any],
    manifests: list[dict[str, Any]],
    evidence_updates: list[dict[str, Any]],
) -> int:
    update_paths = sorted((root / "twin_updates" / "twin_update_candidates").glob("*.json"))
    if len(update_paths) != 30:
        raise ValueError("WP2 must retain every 3-round Twin candidate update.")
    promotions = read_jsonl(root / "twin_updates" / "promotion_decisions.jsonl")
    if len(promotions) != 30:
        raise ValueError("WP2 promotion ledger does not cover every candidate update.")
    manifest_index = {str(row["run_uid"]): row for row in manifests}
    previous_twins: dict[str, str] = {str(row["run_uid"]): str(row["initial_twin_checkpoint_id"]) for row in manifests}
    promotion_index = {(str(row.get("run_uid", "")), int(row.get("round_index", 0))): row for row in promotions}
    evidence_index = {(str(row.get("run_uid", "")), int(row.get("round_index", 0))): row for row in evidence_updates}
    rollbacks = 0
    for update_path in update_paths:
        update = read_json(update_path)
        run_uid = str(update.get("run_uid", ""))
        round_index = int(update.get("round_index", 0))
        branch = str(update.get("branch", ""))
        if run_uid not in manifest_index or branch not in {"static", "coevolution"}:
            raise ValueError("Twin update does not bind to a registered run.")
        require_equal(update.get("input_twin_checkpoint_id"), previous_twins[run_uid], "Twin checkpoint lineage")
        descriptor = require_file_hash(update.get("candidate_twin_checkpoint_path"), update.get("candidate_twin_checkpoint_sha256"), "Candidate Twin checkpoint")
        descriptor_payload = read_json(descriptor)
        require_equal(descriptor_payload.get("checkpoint_id"), update.get("candidate_twin_checkpoint_id"), "Candidate Twin descriptor identity")
        require_equal(descriptor_payload.get("parent_checkpoint_id"), previous_twins[run_uid], "Candidate Twin parent identity")
        require_equal(descriptor_payload.get("parameter_update"), branch == "coevolution", "Twin update branch policy")
        promotion = promotion_index.get((run_uid, round_index))
        if promotion is None:
            raise ValueError("Twin update lacks a promotion or rollback decision.")
        evidence = evidence_index.get((run_uid, round_index))
        if evidence is None:
            raise ValueError("Twin update lacks a matching admitted evidence record.")
        require_equal(update.get("admitted_evidence_hash"), evidence.get("evidence_hash"), "Twin update admitted evidence binding")
        require_equal(update.get("source_outcome_hash"), evidence.get("source_outcome_hash"), "Twin update source outcome binding")
        require_equal(update.get("update_label_origin"), evidence.get("outcome_origin"), "Twin update label origin")
        require_equal(update.get("twin_rollout_used_as_label"), False, "Twin update Twin-rollout label use")
        metrics = {key: float(update[key]) for key in (
            "new_region_decision_risk_delta",
            "calibration_drift",
            "support_domain_drift",
            "old_domain_retention",
            "ood_selective_risk_delta",
            "constraint_risk_delta",
        )}
        if branch == "static":
            require_equal(update.get("twin_parameter_update"), False, "Static Twin parameter update")
            require_equal(update.get("twin_decision"), "static_no_update", "Static Twin decision")
            require_equal(promotion.get("twin_decision"), "static_no_update", "Static promotion decision")
        else:
            require_equal(update.get("twin_parameter_update"), True, "Co-evolution Twin parameter update")
            eligible = promotion_eligible(metrics, payload["promotion_thresholds"])
            require_equal(update.get("promotion_eligible"), eligible, "Twin promotion eligibility")
            expected_decision = "promote" if eligible else "rollback"
            require_equal(update.get("twin_decision"), expected_decision, "Twin promotion decision")
            require_equal(promotion.get("twin_decision"), expected_decision, "Promotion ledger Twin decision")
        if promotion.get("twin_decision") == "rollback":
            rollbacks += 1
            require_equal(promotion.get("restored_twin_checkpoint_id"), previous_twins[run_uid], "Rollback restored Twin identity")
            require_equal(promotion.get("rollback_reason"), "injected_calibration_drift", "Rollback reason")
        else:
            if promotion.get("twin_decision") == "promote":
                previous_twins[run_uid] = str(update["candidate_twin_checkpoint_id"])
    if rollbacks < 1:
        raise ValueError("WP2 did not retain a negative rollback fixture.")
    rollback_rows = read_jsonl(root / "rollback" / "rollback_reports.jsonl")
    if len(rollback_rows) != rollbacks:
        raise ValueError("Rollback report count does not match promotion decisions.")
    return rollbacks


def validate_metric_tables(root: Path) -> None:
    import pyarrow.parquet as pq

    expected_rows = 30
    paths = (
        root / "calibration_drift" / "calibration_drift_metrics.parquet",
        root / "retention" / "retention_metrics.parquet",
        root / "calibration_drift" / "validity_metrics.parquet",
        root / "calibration_drift" / "ood_metrics.parquet",
        root / "paired_comparison" / "paired_results.parquet",
        root / "paired_comparison" / "sequential_metrics.parquet",
        root / "efficiency" / "efficiency_results.parquet",
    )
    for path in paths[:4]:
        if pq.ParquetFile(path).metadata.num_rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows in {path.name}")
    for path in paths[4:6]:
        if pq.ParquetFile(path).metadata.num_rows != 5:
            raise ValueError(f"Expected five paired development worlds in {path.name}")
    if pq.ParquetFile(paths[6]).metadata.num_rows != 10:
        raise ValueError("Efficiency table does not cover every branch-specific development run.")


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    payload = validate_config(config, resolve_repo_path(schema_path))
    root = Path(output_root) if output_root else resolve_repo_path(str(payload["output_root"]))
    if not root.is_dir():
        raise FileNotFoundError(f"WP2 output root is missing: {root}")
    wp1 = validate_wp1_evidence(payload)
    formal = validate_formal_boundary(payload)
    frontier = inspect_core13_frontier(payload)
    initial_twin_checkpoint_id = resolved_initial_twin_checkpoint_id(payload, frontier)
    checksum_file_count = verify_checksum_catalog(root)

    sequential_root = root / "sequential_evidence"
    events = read_jsonl(sequential_root / "sequential_event_logs.jsonl")
    selections = read_jsonl(sequential_root / "selection_freeze_records.jsonl")
    outcomes = read_jsonl(sequential_root / "development_outcomes.jsonl")
    evidence_updates = read_jsonl(sequential_root / "evidence_updates.jsonl")
    validate_event_order(events, selections, outcomes, evidence_updates, int(payload["protocol"]["rounds_per_run"]))
    manifests, behavior_changed = validate_run_manifests(root, payload, initial_twin_checkpoint_id)
    rollbacks = validate_updates(root, payload, manifests, evidence_updates)
    validate_metric_tables(root)

    adaptation = read_json(sequential_root / "sequential_adaptation_report.json")
    sequential_rehearsal_complete = bool(
        adaptation.get("paired_runs_complete") == 5
        and adaptation.get("multi_round_execution") is True
        and adaptation.get("selection_before_outcome") is True
        and adaptation.get("verified_update_labels_only") is True
        and adaptation.get("falsification_or_inconclusive_case") is True
        and adaptation.get("agent_behavior_changes_after_evidence") is True
        and adaptation.get("fixed_seed_replay") is True
        and behavior_changed
    )
    if not sequential_rehearsal_complete:
        raise ValueError("Sequential development evidence does not satisfy the frozen WP2 technical protocol.")
    require_equal(adaptation.get("fixture_rehearsal_only"), True, "Fixture-rehearsal declaration")
    require_equal(adaptation.get("empirical_precondition_resolution_allowed"), False, "Fixture empirical-resolution authorization")
    static_binding = read_json(root / "static_baseline" / "static_baseline_binding.json")
    paired_initialization = read_json(root / "static_baseline" / "paired_initialization_manifest.json")
    paired_budget = read_json(root / "static_baseline" / "paired_budget_report.json")
    require_equal(static_binding.get("core13_frontier_bound"), frontier["core13_frontier_bound"], "Core-13 frontier binding")
    require_equal(static_binding.get("operational_twin_checkpoint_id"), initial_twin_checkpoint_id, "Static operational Twin identity")
    if frontier["core13_frontier_bound"]:
        require_equal(static_binding.get("operational_twin_checkpoint_path"), frontier.get("selected_twin_checkpoint_path"), "Static operational Twin path")
        require_equal(static_binding.get("operational_twin_checkpoint_sha256"), frontier.get("selected_twin_checkpoint_sha256"), "Static operational Twin hash")
    for key in ("same_initialization", "same_outcome_order", "same_budget", "same_provider_tool_configuration", "only_registered_difference_is_twin_update"):
        require_equal(paired_initialization.get(key), True, f"Paired initialization {key}")
    require_equal(paired_budget.get("matched"), True, "Paired budget matching")
    drift_rehearsal_complete = rollbacks >= 1

    decision = read_json(root / "wp2_precondition_decision.json")
    require_equal(decision.get("execution_identity"), execution_identity(), "WP2 execution identity")
    expected_status = "data_blocked"
    require_equal(decision.get("development_coevolution_status"), expected_status, "Development decision")
    require_equal(decision.get("development_blockers", {}).get("blocked_sequential_evidence"), True, "Sequential blocker")
    require_equal(decision.get("development_blockers", {}).get("blocked_calibration_drift"), True, "Calibration blocker")
    require_equal(decision.get("development_blockers", {}).get("blocked_static_baseline"), not frontier["core13_frontier_bound"], "Static baseline blocker")
    require_false(decision.get("formal_execution_allowed"), "Formal execution authorization")
    require_false(decision.get("formal_claim_allowed"), "Formal claim authorization")
    require_equal(decision.get("integrity", {}).get("formal_holdout_access_count"), 0, "Formal holdout access count")
    require_equal(decision.get("integrity", {}).get("coevolution_claim_upgraded"), False, "Co-evolution claim upgrade")
    if not frontier["core13_frontier_bound"]:
        require_equal(decision.get("coevolution_performance_claim"), "not_evaluated", "Unbound-frontier performance claim")

    return {
        "status": "passed",
        "development_coevolution_status": expected_status,
        "core13_frontier_bound": bool(frontier["core13_frontier_bound"]),
        "sequential_rehearsal_complete": sequential_rehearsal_complete,
        "calibration_drift_rehearsal_complete": drift_rehearsal_complete,
        "formal_holdout_access_count": 0,
        "checksum_file_count": checksum_file_count,
        "wp1_binding": wp1,
        "formal_boundary": formal,
    }


def main() -> None:
    args = parse_args()
    result = validate(args.config, args.schema, args.output_root or None)
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        raise SystemExit(1)
