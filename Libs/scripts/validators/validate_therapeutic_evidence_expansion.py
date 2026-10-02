"""Validate the selected-source therapeutic evidence expansion audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_therapeutic_evidence_expansion import (  # noqa: E402
    BUILDER_PATH,
    CANDIDATE_INTEGRITY_PATH,
    CANDIDATE_PATH,
    CONFIG_PATH,
    DOMAIN_PATH,
    MATERIALIZATION_AUDIT_PATH,
    OBSERVATION_PATH,
    PRIORITY_TABLE_PATH,
    PRIORITY_AUDIT_PATH,
    QUALIFICATION_PATH,
    REJECTION_PATH,
    SCHEMA_PATH,
    SPLIT_FEASIBILITY_PATH,
    _artifact,
    build,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _binding_matches(binding: object, path: Path) -> bool:
    if not isinstance(binding, Mapping):
        return False
    try:
        return dict(binding) == _artifact(path)
    except (FileNotFoundError, OSError, ValueError):
        return False


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate(*, recompute: bool = False) -> dict[str, object]:
    blockers: list[str] = []
    try:
        audit = _read_json(MATERIALIZATION_AUDIT_PATH)
        schema = _read_json(SCHEMA_PATH)
        priority_audit = _read_json(PRIORITY_AUDIT_PATH)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}

    for error in Draft202012Validator(schema).iter_errors(audit):
        location = ".".join(str(value) for value in error.absolute_path) or "root"
        blockers.append(f"audit_schema:{location}:{error.message}")

    for label, path in (
        ("implementation_binding", BUILDER_PATH),
        ("schema_binding", SCHEMA_PATH),
        ("config_binding", CONFIG_PATH),
    ):
        if not _binding_matches(audit.get(label), path):
            blockers.append(label)
    inputs = audit.get("input_bindings", {})
    selected_sources = audit.get("selected_sources")
    if selected_sources != priority_audit.get("selected_sources"):
        blockers.append("selected_source_identity")
    if _as_int(audit.get("source_count")) != len(selected_sources) if isinstance(selected_sources, list) else True:
        blockers.append("source_count_identity")
    summary_sources = [summary.get("source_accession") for summary in audit.get("source_summaries", [])]
    if sorted(summary_sources) != sorted(selected_sources) if isinstance(selected_sources, list) else True:
        blockers.append("source_summary_identity")
    for label, path in (
        ("source_priority_audit", PRIORITY_AUDIT_PATH),
        ("source_priority_table", PRIORITY_TABLE_PATH),
    ):
        if not _binding_matches(inputs.get(label), path):
            blockers.append(f"input_binding:{label}")

    outputs = {
        "observations": OBSERVATION_PATH,
        "candidate_sets": CANDIDATE_PATH,
        "rejections": REJECTION_PATH,
        "qualification": QUALIFICATION_PATH,
        "candidate_integrity": CANDIDATE_INTEGRITY_PATH,
        "split_feasibility": SPLIT_FEASIBILITY_PATH,
        "decision_domain": DOMAIN_PATH,
    }
    output_bindings = audit.get("output_bindings", {})
    for name, path in outputs.items():
        if not _binding_matches(output_bindings.get(path.relative_to(REPO_ROOT).as_posix()), path):
            blockers.append(f"output_binding:{name}")

    if audit.get("status") != "materialized_structure_blocked_objective_namespace":
        blockers.append("status")
    if audit.get("objective_namespace_status") != "blocked_platform_annotation_missing":
        blockers.append("objective_namespace_status")
    if audit.get("candidate_set_count") != 0 or audit.get("qualified_candidate_set_count") != 0:
        blockers.append("candidate_qualification_gate")
    if audit.get("model_predictions_inspected") is not False:
        blockers.append("model_predictions_inspected")
    if audit.get("outcome_values_inspected_for_source_selection") is not False:
        blockers.append("outcome_values_inspected_for_source_selection")
    if audit.get("evidence_role") != "development_only" or audit.get("publication_claim_eligible") is not False:
        blockers.append("evidence_boundary")
    if audit.get("protected_holdout_access") != 0:
        blockers.append("protected_holdout_access")

    try:
        observations = _read_tsv(OBSERVATION_PATH)
        candidates = _read_tsv(CANDIDATE_PATH)
        rejections = _read_tsv(REJECTION_PATH)
        qualification = _read_tsv(QUALIFICATION_PATH)
    except (OSError, csv.Error) as exc:
        return {
            "status": "blocked",
            "passed": False,
            "blockers": sorted(set(blockers + [f"tsv_read:{exc}"])),
        }
    if len(observations) != _as_int(audit.get("observation_count")):
        blockers.append("observation_count")
    if _as_int(audit.get("structural_candidate_set_count")) != len(candidates):
        blockers.append("structural_candidate_set_count")
    if not observations:
        blockers.append("observations_empty")
    if not rejections or not qualification:
        blockers.append("qualification_outputs_empty")
    if any(
        row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") != "False"
        or row.get("protected_holdout_access") != "0"
        for row in observations + candidates + rejections + qualification
    ):
        blockers.append("row_boundary")
    required_observation_fields = {
        "source_uid",
        "study_accession",
        "biological_context_uid",
        "common_control_uid",
        "objective_uid",
        "outcome",
        "unit",
        "time_window",
        "replicate_uid",
        "uncertainty",
        "admission_status",
        "admission_rejection_reason",
    }
    if observations and not required_observation_fields.issubset(observations[0]):
        blockers.append("observation_admission_fields")
    if any(row.get("admission_status") != "quarantined_not_admitted" for row in observations):
        blockers.append("observation_admission_status")
    if any(row.get("outcome") != "explicit_unavailable_objective_namespace_blocked" for row in observations):
        blockers.append("observation_outcome_unavailable_status")
    for summary in audit.get("source_summaries", []):
        accession = str(summary.get("source_accession", ""))
        expected_inputs = {
            "sdrf": REPO_ROOT / "Input/raw/arrayexpress/downloads/studies" / accession / f"{accession}.sdrf.txt",
            "processed_archive": REPO_ROOT / "Input/raw/arrayexpress/downloads/studies" / accession / f"{accession}.processed.1.zip",
        }
        for label, path in expected_inputs.items():
            if not _binding_matches(summary.get("input_bindings", {}).get(label), path):
                blockers.append(f"source_input_binding:{accession}:{label}")
        if summary.get("platform_annotation_available") is not False:
            blockers.append(f"platform_annotation_status:{accession}")

    # E-GEOD-41345 must keep the mTBI baseline separate from the sham baseline.
    rows_41345 = [row for row in observations if row.get("source_accession") == "E-GEOD-41345"]
    for row in rows_41345:
        context = row.get("context_label")
        action = row.get("action_label")
        control = row.get("control_label")
        therapeutic = row.get("action_is_therapeutic") == "True"
        if context == "mTBI" and control != "mTBI vehicle":
            blockers.append("E-GEOD-41345:mTBI_control_semantics")
        if context == "sham" and control != "sham vehicle":
            blockers.append("E-GEOD-41345:sham_control_semantics")
        if action == "no_intervention" and therapeutic:
            blockers.append("E-GEOD-41345:baseline_marked_therapeutic")
        if action == "no_intervention" and row.get("world_action_uid") == row.get("control_uid"):
            blockers.append("E-GEOD-41345:baseline_action_control_collision")

    candidate_integrity = _read_json(CANDIDATE_INTEGRITY_PATH)
    split_feasibility = _read_json(SPLIT_FEASIBILITY_PATH)
    domain = _read_json(DOMAIN_PATH)
    if candidate_integrity.get("candidate_set_count") != 0:
        blockers.append("candidate_integrity_gate")
    if split_feasibility.get("split_files_created") is not False:
        blockers.append("split_files_created")
    if domain.get("status") != "blocked_insufficient_decision_sensitive_therapeutic_outcomes":
        blockers.append("decision_domain_gate")
    for key in ("world_action_training_authorized", "agent_in_twin_authorized", "coevolution_authorized"):
        if domain.get(key) is not False:
            blockers.append(f"domain_authorization:{key}")
    for key in ("gpu_campaign_authorized", "static_reference_authorized", "tri_authorized"):
        if domain.get(key) is not False:
            blockers.append(f"domain_authorization:{key}")
    if domain.get("protected_holdout_access") != 0:
        blockers.append("domain_protected_holdout_access")

    if recompute and not blockers:
        rebuilt = build(write_output=False)
        for key in (
            "source_count",
            "observation_count",
            "action_count",
            "candidate_set_count",
            "qualified_candidate_set_count",
            "objective_namespace_status",
        ):
            if rebuilt.get(key) != audit.get(key):
                blockers.append(f"recompute:{key}")

    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-therapeutic-evidence-expansion-validation-v2",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "source_count": audit.get("source_count"),
        "observation_count": audit.get("observation_count"),
        "structural_candidate_set_count": audit.get("structural_candidate_set_count"),
        "candidate_set_count": audit.get("candidate_set_count"),
        "qualified_candidate_set_count": audit.get("qualified_candidate_set_count"),
        "protected_holdout_access": audit.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
