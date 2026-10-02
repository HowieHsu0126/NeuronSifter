"""Freeze functional therapeutic objectives and a metadata-only source queue.

This builder deliberately does not download or parse outcome values.  It creates
the admissible namespace and the ranked acquisition queue; payload admission is
handled by a separate, fail-closed materialization step.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_therapeutic_evidence.schema.json"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_functional_therapeutic_registry.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("functional_therapeutic_evidence")
    if not isinstance(policy, dict):
        raise TypeError("Functional therapeutic evidence policy must be a mapping")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(policy), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Functional therapeutic evidence schema validation failed: {details}")
    return policy


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Metadata registry has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Metadata registry is empty: {path}")
    return list(reader.fieldnames), rows


def _validate_metadata_rows(rows: list[dict[str, str]]) -> None:
    required = {
        "priority_rank", "source_uid", "source_repository", "study_accession", "public_url",
        "publication_uid", "title", "species", "biological_system", "biological_context_uid",
        "endpoint_family", "objective_uid", "common_control_status", "expected_therapeutic_arm_count",
        "expected_action_mode", "dose_semantics", "time_window_semantics", "replicate_structure",
        "uncertainty_availability_at_metadata_stage", "machine_readable_numeric_payload_status",
        "expected_candidate_set_count", "selection_class", "rationale", "main_uncertainty", "metadata_source",
    }
    if set(rows[0]) != required:
        raise ValueError(f"Functional metadata columns drifted: {sorted(set(rows[0]) ^ required)}")
    ranks = [int(row["priority_rank"]) for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise ValueError(f"Functional metadata priority ranks are not contiguous: {ranks}")
    if len({row["source_uid"] for row in rows}) != len(rows):
        raise ValueError("Functional metadata source_uid values are not unique")
    if len({row["study_accession"] for row in rows}) != len(rows):
        raise ValueError("Functional metadata accession values are not unique")
    if any(row["source_repository"] != "GEO" for row in rows):
        raise ValueError("Only explicitly public GEO sources are admitted by this queue")
    if any(row["machine_readable_numeric_payload_status"] != "unverified" for row in rows):
        raise ValueError("Source priority must be frozen before payload verification")
    selected = [row for row in rows if row["selection_class"] == "selected_for_metadata_screen"]
    if not selected:
        raise ValueError("At least one source must be selected for metadata screening")
    if any(int(row["expected_candidate_set_count"]) > 0 and row not in selected for row in rows[:2]):
        raise ValueError("Highest-yield expected sources must be selected for metadata screening")


def _objective_audit(policy: Mapping[str, Any], metadata_path: Path, output_path: Path) -> dict[str, object]:
    objective_freeze = policy["objective_freeze"]
    families = objective_freeze["objective_families"]
    objective_uids = [str(item["objective_uid"]) for item in families]
    if len(objective_uids) != len(set(objective_uids)):
        raise ValueError("Functional objective identities are not unique")
    payload = {
        "schema_version": "neuronsifter-functional-therapeutic-objective-freeze-v1",
        "created_at": _now(),
        "status": "frozen_before_outcome_materialization",
        "objective_namespace": objective_freeze["objective_namespace"],
        "objective_families": families,
        "qualification": objective_freeze["qualification"],
        "outcome_values_inspected_for_objective_selection": False,
        "model_predictions_inspected": False,
        "author_claims_used_as_ground_truth": False,
        "input_binding": _artifact(metadata_path),
        "config_binding": _artifact(REPO_ROOT / "Libs/configs/data/functional_therapeutic_evidence.yaml"),
        "schema_binding": _artifact(SCHEMA_PATH),
        "builder_binding": _artifact(BUILDER_PATH),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    write_json(output_path, payload)
    return payload


def build() -> dict[str, object]:
    policy = _load_policy()
    discovery = policy["source_discovery"]
    metadata_path = _repo_path(str(discovery["metadata_registry"]))
    fields, rows = _read_tsv(metadata_path)
    _validate_metadata_rows(rows)
    queue_path = _repo_path(str(discovery["source_queue"]))
    queue_rows = []
    for row in rows:
        queue_rows.append(
            {
                **row,
                "queue_status": "frozen_metadata_only",
                "outcome_values_inspected_for_priority": "False",
                "model_predictions_inspected": "False",
                "protected_holdout_access": "0",
            }
        )
    queue_fields = [*fields, "queue_status", "outcome_values_inspected_for_priority", "model_predictions_inspected", "protected_holdout_access"]
    write_tsv(queue_path, queue_fields, queue_rows)

    objective_path = _repo_path(str(discovery["outputs"]["objective_freeze"]))
    objective = _objective_audit(policy, metadata_path, objective_path)
    selected = [row for row in rows if row["selection_class"] == "selected_for_metadata_screen"]
    priority_path = _repo_path(str(discovery["outputs"]["source_priority"]))
    priority = {
        "schema_version": "neuronsifter-functional-therapeutic-source-priority-v1",
        "created_at": _now(),
        "status": "metadata_only_queue_frozen",
        "selection_basis": discovery["selection_basis"],
        "source_count": len(rows),
        "selected_for_metadata_screen_count": len(selected),
        "selected_sources": [row["source_uid"] for row in selected],
        "selected_accessions": [row["study_accession"] for row in selected],
        "expected_candidate_set_count": sum(int(row["expected_candidate_set_count"]) for row in rows),
        "expected_independent_multi_arm_context_count": sum(
            int(row["expected_candidate_set_count"]) for row in rows if row["expected_action_mode"] in {"multi_arm", "dose_response"}
        ),
        "outcome_values_inspected_for_priority": False,
        "model_predictions_inspected": False,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "benchmark_split_created": False,
        "rows": rows,
        "input_binding": _artifact(metadata_path),
        "queue_binding": _artifact(queue_path),
        "objective_binding": _artifact(objective_path),
        "config_binding": _artifact(REPO_ROOT / "Libs/configs/data/functional_therapeutic_evidence.yaml"),
        "schema_binding": _artifact(SCHEMA_PATH),
        "builder_binding": _artifact(BUILDER_PATH),
    }
    write_json(priority_path, priority)
    screening = discovery["payload_screening"]
    metadata_by_accession = {row["study_accession"]: row for row in rows}
    screening_records = []
    for record in screening["records"]:
        metadata_row = metadata_by_accession[str(record["study_accession"])]
        screening_records.append(
            {
                **dict(record),
                "source_public_url": metadata_row["public_url"],
                "metadata_source": metadata_row["metadata_source"],
            }
        )
    record_accessions = {str(record["study_accession"]) for record in screening_records}
    source_accessions = {row["study_accession"] for row in rows}
    if record_accessions != source_accessions:
        raise ValueError("Payload screening records do not cover the frozen source queue exactly")
    if any(bool(record["functional_values_admitted"]) for record in screening_records):
        raise ValueError("This bounded screening artifact cannot admit unvalidated functional values")
    materialization_path = _repo_path(str(discovery["outputs"]["materialization"]))
    materialization = {
        "schema_version": "neuronsifter-functional-therapeutic-materialization-v1",
        "created_at": _now(),
        "status": "no_admissible_functional_payloads",
        "selection_basis": discovery["selection_basis"],
        "queue_scope": "bounded_geo_functional_leads_from_metadata_reconnaissance",
        "source_count": len(rows),
        "screened_source_count": len(screening_records),
        "admitted_source_count": 0,
        "observation_count": 0,
        "structured_action_count": 0,
        "legal_candidate_set_count": 0,
        "qualified_decision_sensitive_set_count": 0,
        "outcome_values_used_for_priority": False,
        "model_predictions_inspected": False,
        "protected_holdout_access": 0,
        "records": screening_records,
        "input_bindings": {
            "source_metadata": _artifact(metadata_path),
            "source_queue": _artifact(queue_path),
            "objective_freeze": _artifact(objective_path),
        },
    }
    write_json(materialization_path, materialization)
    decision_path = _repo_path(str(discovery["outputs"]["decision_domain"]))
    decision = {
        "schema_version": "neuronsifter-functional-therapeutic-decision-domain-v1",
        "created_at": _now(),
        "status": "blocked_insufficient_decision_sensitive_functional_therapeutic_outcomes",
        "functional_objective_families": [str(item["objective_uid"]) for item in policy["objective_freeze"]["objective_families"]],
        "observation_count": 0,
        "structured_action_count": 0,
        "legal_candidate_set_count": 0,
        "qualified_decision_sensitive_set_count": 0,
        "independent_context_group_count": 0,
        "benchmark_created": False,
        "benchmark_split_feasible": False,
        "world_action_authorized": False,
        "tri_authorized": False,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "publication_comparison_open": False,
        "protected_holdout_access": 0,
        "dominant_blockers": [
            "no_machine_readable_numeric_functional_payload_in_screened_public_releases",
            "no_legal_multi_action_functional_candidate_set",
            "no_uncertainty_aware_therapeutic_ranking",
        ],
        "next_route": "obtain_stronger_public_or_collaborator_quantitative_multi_arm_functional_outcomes",
        "input_bindings": {
            "materialization": _artifact(materialization_path),
            "objective_freeze": _artifact(objective_path),
            "source_priority": _artifact(priority_path),
        },
    }
    write_json(decision_path, decision)
    gap_path = _repo_path(str(discovery["outputs"]["evidence_gap"]))
    rejection_counts: dict[str, int] = {}
    for record in screening_records:
        for reason in record["rejection_reasons"]:
            rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
    gap = {
        "schema_version": "neuronsifter-functional-therapeutic-evidence-gap-v1",
        "created_at": _now(),
        "status": "blocked_insufficient_decision_sensitive_functional_therapeutic_outcomes",
        "queue_scope": "bounded_geo_functional_leads_from_metadata_reconnaissance",
        "queue_exhausted": True,
        "screened_source_count": len(screening_records),
        "admitted_source_count": 0,
        "legal_candidate_set_count": 0,
        "qualified_decision_sensitive_set_count": 0,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "evidence_gap_taxonomy": [
            "machine_readable_functional_endpoint",
            "shared_control_and_multi_action_context",
            "replicate_identity_and_uncertainty",
            "therapeutic_action_semantics",
            "endpoint_namespace_compatibility",
        ],
        "remaining_resource_families_not_admitted": [
            "Allen_ephys_morphology_ophys_release_identity_unresolved",
            "collaborator_quantitative_multi_arm_functional_outcomes",
        ],
        "ordinary_transcriptomic_expansion_reopened": False,
        "model_training_authorized": False,
        "gpu_campaign_authorized": False,
        "protected_holdout_access": 0,
        "lowest_cost_alternative": "stronger_public_functional_resources_or_collaborator_quantitative_multi_arm_outcomes",
        "input_bindings": {
            "decision_domain": _artifact(decision_path),
            "materialization": _artifact(materialization_path),
            "source_priority": _artifact(priority_path),
        },
    }
    write_json(gap_path, gap)
    return {
        "status": decision["status"],
        "objective_family_count": len(objective["objective_families"]),
        "source_count": len(rows),
        "selected_source_count": len(selected),
        "expected_candidate_set_count": priority["expected_candidate_set_count"],
        "legal_candidate_set_count": 0,
        "qualified_decision_sensitive_set_count": 0,
        "protected_holdout_access": 0,
    }


def main() -> int:
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
