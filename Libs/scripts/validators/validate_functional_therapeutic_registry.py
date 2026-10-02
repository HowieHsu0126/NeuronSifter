"""Validate the functional therapeutic objective, queue, and fail-closed gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_functional_therapeutic_registry import (  # noqa: E402
    CONFIG_PATH,
    SCHEMA_PATH,
    _repo_path,
    build,
)
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return [dict(row) for row in reader]


def validate(*, recompute: bool = True) -> dict[str, object]:
    if recompute:
        build()
    config = load_config(CONFIG_PATH)
    policy = config["functional_therapeutic_evidence"]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_errors = sorted(Draft202012Validator(schema).iter_errors(policy), key=lambda error: list(error.absolute_path))
    blockers: list[str] = [
        f"schema:{list(error.absolute_path)}:{error.message}" for error in schema_errors
    ]
    discovery = policy["source_discovery"]
    metadata_path = _repo_path(str(discovery["metadata_registry"]))
    queue_path = _repo_path(str(discovery["source_queue"]))
    metadata_rows = _read_tsv(metadata_path)
    queue_rows = _read_tsv(queue_path)
    if len(metadata_rows) != len(queue_rows):
        blockers.append("queue_row_count")
    for index, (metadata, queue) in enumerate(zip(metadata_rows, queue_rows, strict=False)):
        for field in metadata:
            if queue.get(field) != metadata.get(field):
                blockers.append(f"queue_row_{index}:{field}")
        if queue.get("queue_status") != "frozen_metadata_only":
            blockers.append(f"queue_status:{index}")
        if queue.get("outcome_values_inspected_for_priority") != "False":
            blockers.append(f"queue_outcome_inspection:{index}")
        if queue.get("model_predictions_inspected") != "False":
            blockers.append(f"queue_model_inspection:{index}")
        if queue.get("protected_holdout_access") != "0":
            blockers.append(f"queue_holdout:{index}")

    objective_path = _repo_path(str(discovery["outputs"]["objective_freeze"]))
    priority_path = _repo_path(str(discovery["outputs"]["source_priority"]))
    materialization_path = _repo_path(str(discovery["outputs"]["materialization"]))
    decision_path = _repo_path(str(discovery["outputs"]["decision_domain"]))
    gap_path = _repo_path(str(discovery["outputs"]["evidence_gap"]))
    objective = _json(objective_path)
    priority = _json(priority_path)
    materialization = _json(materialization_path)
    decision = _json(decision_path)
    gap = _json(gap_path)
    families = policy["objective_freeze"]["objective_families"]
    if objective.get("status") != "frozen_before_outcome_materialization":
        blockers.append("objective_status")
    if objective.get("outcome_values_inspected_for_objective_selection") is not False:
        blockers.append("objective_outcome_inspection")
    if objective.get("model_predictions_inspected") is not False:
        blockers.append("objective_model_inspection")
    if {item["objective_uid"] for item in families} != {item["objective_uid"] for item in objective.get("objective_families", [])}:
        blockers.append("objective_family_identity")
    if priority.get("outcome_values_inspected_for_priority") is not False:
        blockers.append("priority_outcome_inspection")
    if priority.get("model_predictions_inspected") is not False:
        blockers.append("priority_model_inspection")
    if int(priority.get("protected_holdout_access", -1)) != 0:
        blockers.append("priority_holdout")
    screening_records = discovery["payload_screening"]["records"]
    if len(materialization.get("records", [])) != len(screening_records):
        blockers.append("materialization_screening_coverage")
    if any(bool(record.get("functional_values_admitted")) for record in materialization.get("records", [])):
        blockers.append("materialization_admitted_unvalidated_payload")
    for key in ("observation_count", "structured_action_count", "legal_candidate_set_count", "qualified_decision_sensitive_set_count"):
        if materialization.get(key) != 0:
            blockers.append(f"materialization_{key}")
    if materialization.get("status") != "no_admissible_functional_payloads":
        blockers.append("materialization_status")
    if materialization.get("queue_scope") != "bounded_geo_functional_leads_from_metadata_reconnaissance":
        blockers.append("materialization_queue_scope")
    if decision.get("status") != "blocked_insufficient_decision_sensitive_functional_therapeutic_outcomes":
        blockers.append("decision_status")
    if any(decision.get(key) is not False for key in ("benchmark_created", "benchmark_split_feasible", "world_action_authorized", "tri_authorized", "agent_in_twin_authorized", "coevolution_authorized")):
        blockers.append("downstream_authorization")
    if gap.get("status") != decision.get("status") or gap.get("queue_exhausted") is not True:
        blockers.append("evidence_gap_status")
    if gap.get("queue_scope") != "bounded_geo_functional_leads_from_metadata_reconnaissance":
        blockers.append("gap_queue_scope")
    if int(gap.get("protected_holdout_access", -1)) != 0:
        blockers.append("gap_holdout")
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "scientific_gate": decision.get("status"),
        "objective_family_count": len(families),
        "source_count": len(metadata_rows),
        "admitted_source_count": materialization.get("admitted_source_count"),
        "legal_candidate_set_count": materialization.get("legal_candidate_set_count"),
        "qualified_decision_sensitive_set_count": materialization.get("qualified_decision_sensitive_set_count"),
        "benchmark_created": decision.get("benchmark_created"),
        "protected_holdout_access": decision.get("protected_holdout_access"),
    }


def main() -> int:
    result = validate(recompute=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
