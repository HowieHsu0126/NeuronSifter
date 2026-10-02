"""Materialize selected therapeutic source structure and qualify it fail-closed."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_therapeutic_source_priority import (  # noqa: E402
    OUTPUT_AUDIT_PATH as PRIORITY_AUDIT_PATH,
    _artifact,
)
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import write_json, write_tsv  # noqa: E402

PRIORITY_TABLE_PATH = REPO_ROOT / "Output/audits/therapeutic_source_priority.tsv"
SOURCE_ROOT = REPO_ROOT / "Input/raw/arrayexpress/downloads/studies"
PLATFORM_ROOT = REPO_ROOT / "Input/raw/arrayexpress/platform_annotations"
OBSERVATION_PATH = REPO_ROOT / "Input/manifests/therapeutic_outcomes/selected_source_observations_v2.tsv"
CANDIDATE_PATH = REPO_ROOT / "Input/manifests/therapeutic_outcomes/selected_source_candidate_sets_v2.tsv"
MATERIALIZATION_AUDIT_PATH = REPO_ROOT / "Output/audits/therapeutic_evidence_materialization_v2.json"
REJECTION_PATH = REPO_ROOT / "Output/audits/therapeutic_evidence_rejection_catalog_v2.tsv"
CANDIDATE_INTEGRITY_PATH = REPO_ROOT / "Output/audits/therapeutic_candidate_set_integrity_v2.json"
QUALIFICATION_PATH = REPO_ROOT / "Output/audits/therapeutic_candidate_qualification_v2.tsv"
SPLIT_FEASIBILITY_PATH = REPO_ROOT / "Output/audits/therapeutic_benchmark_split_feasibility.json"
DOMAIN_PATH = REPO_ROOT / "Output/audits/therapeutic_decision_domain_v2.json"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_therapeutic_evidence_expansion.py"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/therapeutic_evidence_expansion.schema.json"
CONFIG_PATH = REPO_ROOT / "Libs/configs/data/therapeutic_decision_evidence_contract.yaml"
OBJECTIVE_UID = "ad_transcriptomic_therapeutic_benefit_source_context_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _read_priority() -> dict[str, Any]:
    payload = json.loads(PRIORITY_AUDIT_PATH.read_text(encoding="utf-8"))
    if payload.get("outcome_values_inspected") is not False or payload.get("model_predictions_inspected") is not False:
        raise ValueError("Source priority audit is not metadata-only")
    return payload


def _read_sdrf(accession: str) -> list[dict[str, str]]:
    path = SOURCE_ROOT / accession / f"{accession}.sdrf.txt"
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"SDRF is empty: {path}")
    return rows


def _therapeutic_semantics(accession: str, condition: str) -> dict[str, object]:
    """Resolve source-declared context/control semantics without guessing values."""
    if accession == "E-GEOD-41345":
        exact = {
            "vehicle only (sham)": ("sham", "no_intervention", "sham vehicle", False),
            "EX-4": ("sham", "EX-4", "sham vehicle", True),
            "control vehicle treated mice subjected to mTBI induced by weight drop injury (mTBI)": (
                "mTBI",
                "no_intervention",
                "mTBI vehicle",
                False,
            ),
            "mice subjected to mTBI induced by weight drop injury while being treated with EX-4 (Ex-4/mTBI)": (
                "mTBI",
                "EX-4",
                "mTBI vehicle",
                True,
            ),
        }
        try:
            context, action, control, therapeutic = exact[condition]
        except KeyError as exc:
            raise ValueError(f"{accession} has an unrecognized treatment protocol: {condition}") from exc
        return {
            "context_label": context,
            "action_label": action,
            "control_label": control,
            "action_is_therapeutic": therapeutic,
        }
    if accession == "E-GEOD-55369":
        if condition == "vehicle (control)":
            return {
                "context_label": "primary_rat_e18_cortical_hippocampal_neurons",
                "action_label": "no_intervention",
                "control_label": "vehicle",
                "action_is_therapeutic": False,
            }
        if condition in {"ADDL", "CATSKL", "ADDL+CAT", "CAT+ADDL"}:
            return {
                "context_label": "primary_rat_e18_cortical_hippocampal_neurons",
                "action_label": condition,
                "control_label": "vehicle",
                "action_is_therapeutic": True,
            }
    raise ValueError(f"{accession} has an unrecognized therapeutic condition: {condition}")


def _condition(row: Mapping[str, str]) -> str:
    for key in ("FactorValue [TREATED WITH]", "FactorValue [TREATMENT PROTOCOL]"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError("Selected therapeutic SDRF row lacks a treatment factor")


def _sample_rows(accession: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_file: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        filename = str(row.get("Derived Array Data File", "")).strip()
        if not filename:
            raise ValueError(f"{accession} SDRF row lacks derived matrix file")
        by_file.setdefault(filename, []).append(row)
    output: list[dict[str, object]] = []
    archive_path = SOURCE_ROOT / accession / f"{accession}.processed.1.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for filename, file_rows in sorted(by_file.items()):
            if filename not in names:
                raise ValueError(f"{accession} archive lacks SDRF matrix member: {filename}")
            conditions = [_condition(row) for row in file_rows]
            # Two-colour studies have two SDRF rows per biological ratio; keep the
            # intervention factor and retain the control pairing explicitly.
            if accession == "E-GEOD-55369":
                intervention = [value for value in conditions if value != "vehicle (control)"]
                if len(intervention) != 1 or len(conditions) != 2:
                    raise ValueError(f"{accession} two-colour pairing drifted for {filename}: {conditions}")
                intervention_semantics = _therapeutic_semantics(accession, intervention[0])
                control_semantics = _therapeutic_semantics(accession, "vehicle (control)")
            else:
                if len(conditions) != 1:
                    raise ValueError(f"{accession} single-colour condition drifted for {filename}: {conditions}")
                intervention_semantics = _therapeutic_semantics(accession, conditions[0])
                control_semantics = _therapeutic_semantics(
                    accession,
                    "vehicle only (sham)"
                    if intervention_semantics["context_label"] == "sham"
                    else "control vehicle treated mice subjected to mTBI induced by weight drop injury (mTBI)",
                )
            finite_count = 0
            missing_count = 0
            with archive.open(filename) as handle:
                reader = csv.DictReader((line.decode("utf-8", errors="replace") for line in handle), delimiter="\t")
                if "VALUE" not in (reader.fieldnames or []):
                    raise ValueError(f"{accession} matrix lacks VALUE column: {filename}")
                for matrix_row in reader:
                    value = str(matrix_row.get("VALUE", "")).strip()
                    if not value:
                        missing_count += 1
                        continue
                    try:
                        numeric = float(value)
                    except ValueError:
                        missing_count += 1
                        continue
                    if np.isfinite(numeric):
                        finite_count += 1
                    else:
                        missing_count += 1
            sample_uid = f"arrayexpress:{accession}:sample:{filename.removesuffix('_sample_table.txt')}"
            context_label = str(intervention_semantics["context_label"])
            action = str(intervention_semantics["action_label"])
            control = str(intervention_semantics["control_label"])
            context_uid = _uid("therapeutic_context", accession, context_label)
            control_uid = _uid("therapeutic_control", accession, context_label, control)
            action_uid = _uid("world_action", accession, context_label, action)
            output.append(
                {
                    "observation_uid": _uid("therapeutic_observation", accession, sample_uid),
                    "source_uid": f"arrayexpress:{accession}",
                    "study_accession": accession,
                    "source_accession": accession,
                    "sample_uid": sample_uid,
                    "biological_context_uid": context_uid,
                    "context_uid": context_uid,
                    "common_control_uid": control_uid,
                    "control_uid": control_uid,
                    "objective_uid": OBJECTIVE_UID,
                    "action_label": action,
                    "control_label": control,
                    "context_label": context_label,
                    "action_is_therapeutic": bool(intervention_semantics["action_is_therapeutic"]),
                    "control_is_context_specific": True,
                    "world_action_uid": action_uid,
                    "measurement_namespace": "platform_reporter_unmapped",
                    "value_unit": "source_processed_VALUE",
                    "outcome": "explicit_unavailable_objective_namespace_blocked",
                    "unit": "source_processed_VALUE",
                    "time_window": "explicit_unavailable",
                    "matrix_member": filename,
                    "replicate_group_uid": _uid("replicate_group", accession, context_label, action),
                    "replicate_uid": _uid("replicate_group", accession, context_label, action),
                    "uncertainty": "explicit_unavailable_platform_annotation_missing",
                    "replicate_identity_present": True,
                    "finite_value_count": finite_count,
                    "missing_value_count": missing_count,
                    "objective_namespace_status": "blocked_platform_annotation_missing",
                    "candidate_set_status": "not_qualified_objective_namespace_blocked",
                    "admission_status": "quarantined_not_admitted",
                    "admission_rejection_reason": "objective_namespace_unresolved",
                    "evidence_class": "E2_REAL_ENDPOINT",
                    "evidence_grade": "B",
                    "evidence_role": "development_only",
                    "publication_claim_eligible": False,
                    "protected_holdout_access": 0,
                }
            )
    return output


def build(*, write_output: bool = True) -> dict[str, object]:
    priority = _read_priority()
    selected = [str(value) for value in priority["selected_sources"]]
    all_observations: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    source_summaries: list[dict[str, object]] = []
    for accession in selected:
        rows = _read_sdrf(accession)
        observations = _sample_rows(accession, rows)
        all_observations.extend(observations)
        actions = sorted({str(row["action_label"]) for row in observations})
        counts = {action: sum(str(row["action_label"]) == action for row in observations) for action in actions}
        source_summaries.append(
            {
                "source_accession": accession,
                "input_bindings": {
                    "sdrf": _artifact(SOURCE_ROOT / accession / f"{accession}.sdrf.txt"),
                    "processed_archive": _artifact(SOURCE_ROOT / accession / f"{accession}.processed.1.zip"),
                },
                "platform_annotation_path": (PLATFORM_ROOT / accession).relative_to(REPO_ROOT).as_posix(),
                "platform_annotation_available": (PLATFORM_ROOT / accession).exists(),
                "sdrf_row_count": len(rows),
                "observation_count": len(observations),
                "action_count": len(actions),
                "actions": actions,
                "replicate_counts": counts,
                "matched_control_present": (
                    accession == "E-GEOD-55369"
                    or any(not bool(row["action_is_therapeutic"]) for row in observations)
                ),
                "control_evidence_mode": "paired_two_colour_control" if accession == "E-GEOD-55369" else "explicit_control_observation",
                "objective_namespace_status": "blocked_platform_annotation_missing",
                "candidate_set_count": 0,
                "qualified_candidate_set_count": 0,
            }
        )
        rejection_rows.append(
            {
                "source_accession": accession,
                "rejection_class": "objective_namespace_unresolved",
                "evidence_class": "E2_REAL_ENDPOINT",
                "observed_action_count": len(actions),
                "observed_observation_count": len(observations),
                "reason": "platform reporter identifiers are not mapped to the frozen therapeutic objective namespace",
                "required_next_action": "materialize and validate platform annotation plus species-compatible objective mapping",
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in all_observations:
        grouped.setdefault((str(row["source_accession"]), str(row["context_uid"])), []).append(row)
    candidate_rows = []
    for (accession, context_uid), rows in sorted(grouped.items()):
        actions = sorted({str(row["world_action_uid"]) for row in rows if row["action_is_therapeutic"]})
        if len(actions) < 2:
            continue
        controls = sorted({str(row["control_uid"]) for row in rows})
        candidate_rows.append(
            {
                "candidate_set_uid": _uid("candidate_set", accession, context_uid),
                "source_accession": accession,
                "context_uid": context_uid,
                "control_uid": controls[0] if len(controls) == 1 else "ambiguous_control",
                "candidate_action_uids": json.dumps(actions, separators=(",", ":")),
                "candidate_action_count": len(actions),
                "objective_uid": OBJECTIVE_UID,
                "classification": "blocked_objective_namespace_missing",
                "qualification_status": "blocked_objective_namespace_missing",
                "matched_control": len(controls) == 1,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
    structural_by_source = {
        accession: sum(1 for row in candidate_rows if row["source_accession"] == accession)
        for accession in selected
    }
    for summary in source_summaries:
        summary["structural_candidate_set_count"] = structural_by_source.get(str(summary["source_accession"]), 0)
    qualification_rows = [
        {
            "source_accession": summary["source_accession"],
            "candidate_set_count": summary["candidate_set_count"],
            "qualified_decision_sensitive_count": summary["qualified_candidate_set_count"],
            "classification": "blocked_objective_namespace_missing",
            "effect_size": "explicit_unavailable",
            "pairwise_action_contrast": "explicit_unavailable",
            "decision_margin": "explicit_unavailable",
            "confidence_interval": "explicit_unavailable",
            "rank_probability": "explicit_unavailable",
            "replicate_consistency": "not_evaluable",
            "objective_uid": OBJECTIVE_UID,
            "outcome_blind_objective": True,
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        }
        for summary in source_summaries
    ]

    fields = list(all_observations[0]) if all_observations else []
    if write_output:
        write_tsv(OBSERVATION_PATH, fields, all_observations)
        candidate_fields = list(candidate_rows[0]) if candidate_rows else [
            "candidate_set_uid", "source_accession", "context_uid", "control_uid", "candidate_action_uids",
            "candidate_action_count", "objective_uid", "classification", "qualification_status", "matched_control",
            "evidence_role", "publication_claim_eligible", "protected_holdout_access",
        ]
        write_tsv(CANDIDATE_PATH, candidate_fields, candidate_rows)
        write_tsv(REJECTION_PATH, list(rejection_rows[0]), rejection_rows)
        write_tsv(QUALIFICATION_PATH, list(qualification_rows[0]), qualification_rows)
    materialization = {
        "schema_version": "neuronsifter-therapeutic-evidence-materialization-v2",
        "created_at": _now(),
        "status": "materialized_structure_blocked_objective_namespace",
        "selected_sources": selected,
        "source_count": len(source_summaries),
        "observation_count": len(all_observations),
        "action_count": len({str(row["world_action_uid"]) for row in all_observations}),
        "matched_control_context_count": len({str(row["control_uid"]) for row in all_observations}),
        "structural_candidate_set_count": len(candidate_rows),
        "candidate_set_count": 0,
        "qualified_candidate_set_count": 0,
        "objective_namespace_status": "blocked_platform_annotation_missing",
        "outcome_values_inspected_for_source_selection": False,
        "model_predictions_inspected": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "input_bindings": {
            "source_priority_audit": _artifact(PRIORITY_AUDIT_PATH),
            "source_priority_table": _artifact(PRIORITY_TABLE_PATH),
        },
        "source_summaries": source_summaries,
        "output_bindings": {},
        "schema_binding": _artifact(SCHEMA_PATH),
        "config_binding": _artifact(CONFIG_PATH),
    }
    candidate_integrity = {
        "schema_version": "neuronsifter-therapeutic-candidate-set-integrity-v2",
        "status": "blocked_no_candidate_sets_objective_namespace_missing",
        "structural_candidate_set_count": len(candidate_rows),
        "candidate_set_count": 0,
        "multi_action_context_count": len(candidate_rows),
        "matched_control_context_count": materialization["matched_control_context_count"],
        "context_overlap_count": 0,
        "action_overlap_count": 0,
        "objective_uid": "ad_transcriptomic_therapeutic_benefit_source_context_v1",
        "protected_holdout_access": 0,
    }
    split_feasibility = {
        "schema_version": "neuronsifter-therapeutic-benchmark-split-feasibility-v2",
        "status": "not_feasible_no_qualified_candidate_sets",
        "candidate_set_count": 0,
        "qualified_candidate_set_count": 0,
        "independent_publication_groups": 0,
        "independent_context_groups": 0,
        "group_overlap_count": 0,
        "split_files_created": False,
        "protected_holdout_access": 0,
    }
    domain = {
        "schema_version": "neuronsifter-therapeutic-decision-domain-v2",
        "status": "blocked_insufficient_decision_sensitive_therapeutic_outcomes",
        "source_count": len(source_summaries),
        "observation_count": len(all_observations),
        "action_count": len({str(row["world_action_uid"]) for row in all_observations}),
        "matched_control_context_count": materialization["matched_control_context_count"],
        "candidate_set_count": 0,
        "qualified_decision_sensitive_sets": 0,
        "objective_namespace_status": "blocked_platform_annotation_missing",
        "benchmark_split_created": False,
        "world_action_training_authorized": False,
        "gpu_campaign_authorized": False,
        "static_reference_authorized": False,
        "tri_authorized": False,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "publication_comparison_open": False,
        "protected_holdout_access": 0,
        "historical_round21_authority_recreated": False,
        "old_round22_tier_c_replay_used": False,
    }
    for path, payload in (
        (MATERIALIZATION_AUDIT_PATH, materialization),
        (CANDIDATE_INTEGRITY_PATH, candidate_integrity),
        (SPLIT_FEASIBILITY_PATH, split_feasibility),
        (DOMAIN_PATH, domain),
    ):
        if write_output:
            write_json(path, payload)
    if write_output:
        for path in (OBSERVATION_PATH, REJECTION_PATH, QUALIFICATION_PATH):
            if path.exists():
                materialization["output_bindings"][path.relative_to(REPO_ROOT).as_posix()] = _artifact(path)
        materialization["output_bindings"][CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()] = _artifact(CANDIDATE_PATH)
        for path in (CANDIDATE_INTEGRITY_PATH, SPLIT_FEASIBILITY_PATH, DOMAIN_PATH):
            materialization["output_bindings"][path.relative_to(REPO_ROOT).as_posix()] = _artifact(path)
        materialization["implementation_binding"] = _artifact(BUILDER_PATH)
        write_json(MATERIALIZATION_AUDIT_PATH, materialization)
    return {
        "status": domain["status"],
        "source_count": len(source_summaries),
        "observation_count": len(all_observations),
        "action_count": domain["action_count"],
        "candidate_set_count": 0,
        "qualified_candidate_set_count": 0,
        "objective_namespace_status": domain["objective_namespace_status"],
        "protected_holdout_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write_output=not args.no_write), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
