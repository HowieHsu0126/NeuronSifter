"""Record the explicit no-run source-recovery result for an empty frozen intake queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    _binding,
)
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


PREREGISTRATION_PATH = (
    REPO_ROOT
    / "Input/manifests/retrospective_public/design_enriched_ranking_intake_preregistration.json"
)
INTAKE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_design_enriched_ranking_intake.json"
)
QUEUE_PATH = REPO_ROOT / "Input/manifests/retrospective_public/design_enriched_ranking_intake_queue.tsv"
EVIDENCE_REGISTRY_PATH = REPO_ROOT / "Input/manifests/retrospective_public/evidence_registry.tsv"
MATERIALIZATION_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_evidence_materialization.json"
)
REPLAY_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
LEAKAGE_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_leakage_audit.json"
SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/retrospective_public_design_enriched_source_recovery.schema.json"
)
OUTPUT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_design_enriched_ranking_source_recovery.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def build(*, write_output: bool = True) -> dict[str, Any]:
    preregistration = _read_json(PREREGISTRATION_PATH)
    intake = _read_json(INTAKE_AUDIT_PATH)
    materialization = _read_json(MATERIALIZATION_AUDIT_PATH)
    replay = _read_json(REPLAY_AUDIT_PATH)
    leakage = _read_json(LEAKAGE_AUDIT_PATH)
    if intake.get("status") != "metadata_only_queue_exhausted":
        raise ValueError("Design-enriched source recovery may be skipped only for an empty queue")
    if intake.get("record_count") != 0:
        raise ValueError("Design-enriched source recovery audit observed a non-empty queue")
    if QUEUE_PATH.read_text(encoding="utf-8").count("\n") != 1:
        raise ValueError("Design-enriched queue is not header-only")
    parent_bindings = preregistration["parent_bindings"]
    if _binding(MATERIALIZATION_AUDIT_PATH) != parent_bindings["materialization_audit"]:
        raise ValueError("Evidence materialization changed despite the empty queue")
    if _binding(REPLAY_AUDIT_PATH) != parent_bindings["replay_audit"]:
        raise ValueError("Replay candidate registry changed despite the empty queue")
    expected_state = preregistration["parent_state"]
    final_state = {
        "status": materialization.get("status"),
        "accepted_records": materialization.get("accepted_record_count"),
        "rejected_records": materialization.get("rejected_record_count"),
        "independent_decision_sensitive_groups": replay.get("independent_decision_sensitive_group_count"),
        "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
        "split_files_created": replay.get("split_files_created"),
        "leakage_status": leakage.get("status"),
    }
    required_state = {
        "status": expected_state["status"],
        "accepted_records": expected_state["accepted_records"],
        "rejected_records": expected_state["rejected_records"],
        "independent_decision_sensitive_groups": expected_state["independent_decision_sensitive_groups"],
        "required_groups": expected_state["required_groups"],
        "split_files_created": False,
        "leakage_status": "passed",
    }
    if final_state != required_state:
        raise ValueError(f"Empty-queue final state drifted: {final_state}")
    audit = {
        "schema_version": "neuronsifter-retrospective-public-design-enriched-source-recovery-v1",
        "status": "not_run_queue_empty",
        "intake_uid": intake["intake_uid"],
        "preregistration_digest": intake["preregistration_digest"],
        "queue_digest": intake["queue_digest"],
        "metadata_record_count": intake["selection_summary"]["metadata_record_count"],
        "metadata_eligible_count": intake["selection_summary"]["eligible_record_count"],
        "queue_record_count": intake["record_count"],
        "source_recovered_count": 0,
        "formal_abc_admitted_count": 0,
        "new_legal_candidate_count": 0,
        "new_decision_sensitive_group_count": 0,
        "evidence_registry_updated": False,
        "candidate_registry_updated": False,
        "replay_audit_updated": False,
        "leakage_audit_updated": False,
        "unchanged_artifact_bindings": {
            "evidence_registry": _binding(EVIDENCE_REGISTRY_PATH),
            "materialization_audit": _binding(MATERIALIZATION_AUDIT_PATH),
            "candidate_registry_and_replay_audit": _binding(REPLAY_AUDIT_PATH),
            "leakage_audit": _binding(LEAKAGE_AUDIT_PATH),
        },
        "final_state": final_state,
        "split_authorized": False,
        "world_action_authorized": False,
        "gpu_authorized": False,
        "tri_authorized": False,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "protected_holdout_access": 0,
        "next_action": "stop_design_enriched_intake_exhausted_preserve_two_of_three_groups",
        "created_at": _now(),
    }
    schema = _read_json(SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(audit),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Design-enriched source recovery schema errors: {detail}")
    if write_output:
        write_json(OUTPUT_PATH, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = build(write_output=not args.no_write)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
