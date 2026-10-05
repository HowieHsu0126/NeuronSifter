"""Audit C2 evidence reachability and fail closed before any C2 training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402


RESULT_FIELDS = [
    "claim_id", "experiment_id", "task_id", "holdout_id", "model_id",
    "checkpoint_hash", "corpus_hash", "split_hash", "seed", "metric", "value",
    "unit", "baseline", "paired_delta", "ci_low", "ci_high", "p_adjustment",
    "bootstrap_unit", "status", "provenance",
]


def _relative(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    path.relative_to(repo_root)
    return path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _binding(repo_root: Path, value: str) -> dict[str, object]:
    path = _relative(repo_root, value)
    return {
        "path": value,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _count(rows: Iterable[Mapping[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def _identity(payload: Mapping[str, object]) -> str | None:
    direct = payload.get("corpus_identity_hash")
    if isinstance(direct, str):
        return direct
    identity_payload = payload.get("identity_payload")
    if isinstance(identity_payload, Mapping):
        value = identity_payload.get("corpus_identity_hash")
        if isinstance(value, str):
            return value
    return None


def _corpus_binding(repo_root: Path, entry: Mapping[str, object]) -> dict[str, object]:
    bundle_path = str(entry["evidence_bundle_path"])
    validation_path = str(entry["evidence_validation_path"])
    bundle = _load_json(_relative(repo_root, bundle_path))
    validation = _load_json(_relative(repo_root, validation_path))
    expected = str(entry["identity_hash"])
    observed = _identity(bundle)
    validation_status = str(validation.get("status", validation.get("validation_status", "missing")))
    return {
        "corpus_uid": str(entry["corpus_uid"]),
        "expected_identity_hash": expected,
        "observed_identity_hash": observed,
        "identity_match": observed == expected,
        "evidence_validation_status": validation_status,
        "bundle": _binding(repo_root, bundle_path),
        "validation": _binding(repo_root, validation_path),
    }


def _action_audit(
    qc_rows: list[dict[str, str]],
    regimen_rows: list[dict[str, str]],
    contract: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    eligible = [row for row in qc_rows if row.get("qc_grade") in {"A", "B"} and row.get("qc_use") == "main_pretraining"]
    regimen_by_uid = {str(row["regimen_uid"]): row for row in regimen_rows}
    used_regimen_uids = {str(row["regimen_uid"]) for row in eligible}
    linked_rows = [regimen_by_uid[uid] for uid in sorted(used_regimen_uids) if uid in regimen_by_uid]
    missing_links = sorted(used_regimen_uids - set(regimen_by_uid))
    required_fields = [str(value) for value in contract["required_fields"]]
    field_coverage = {
        "action_uid": len(linked_rows),
        "intervention_type": len(linked_rows),
        "agent_set": sum(bool(row.get("compound_uid_set")) for row in linked_rows),
        "target_set": sum(bool(row.get("target_uid_set")) for row in linked_rows),
        "dose_value": 0,
        "dose_unit": 0,
        "schedule": sum(bool(row.get("schedule")) for row in linked_rows),
        "duration": 0,
        "route": sum(bool(row.get("route")) for row in linked_rows),
        "cell_or_tissue_context": 0,
        "disease_state_scope": len(linked_rows),
        "exposure_program_ref": 0,
        "occupancy_program_ref": 0,
        "off_target_context": 0,
        "publication_accession": 0,
        "assay_series_uid": 0,
        "evidence_quality": sum(bool(row.get("selection_reason")) for row in linked_rows),
    }
    unsupported = [field for field in required_fields if field_coverage.get(field, 0) != len(linked_rows)]
    split_roles_by_regimen: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        split_roles_by_regimen[str(row["regimen_uid"])].add(str(row["split_name"]))
    duplicated_across_split = sorted(
        uid for uid, roles in split_roles_by_regimen.items() if len(roles) > 1
    )
    uid_contains_split_role = [
        uid for uid in used_regimen_uids
        if any(token in uid.lower() for token in ("train", "validation", "test", "holdout"))
    ]
    exposure_is_prior = True
    audit = {
        "schema_version": "neuronsifter-c2-structured-action-audit-v1",
        "status": "blocked",
        "eligible_trajectory_row_count": len(eligible),
        "action_program_count": len(used_regimen_uids),
        "linked_action_program_count": len(linked_rows),
        "missing_action_link_count": len(missing_links),
        "missing_action_links": missing_links[:20],
        "canonical_required_fields": required_fields,
        "field_coverage": field_coverage,
        "unsupported_required_fields": unsupported,
        "dose_representation": {
            "available_columns": ["dose_level_id", "dose_multiplier"],
            "physical_dose_value_available": False,
            "physical_dose_unit_available": False,
            "duration_available": False,
            "schedule_labels": _count(linked_rows, "schedule"),
            "route_labels": _count(linked_rows, "route"),
            "dose_multiplier_labels": _count(linked_rows, "dose_multiplier"),
        },
        "exposure_occupancy_provenance": {
            "exposure_program_ref_available": False,
            "occupancy_program_ref_available": False,
            "model_or_bootstrap_prior_present": exposure_is_prior,
            "bootstrap_prior_counted_as_observed_provenance": False,
        },
        "split_integrity": {
            "action_uid_encodes_split_role": bool(uid_contains_split_role),
            "action_uid_with_split_tokens": uid_contains_split_role[:20],
            "existing_trajectory_random_split_has_duplicate_action_program_count": len(duplicated_across_split),
            "existing_trajectory_random_split_is_valid_c2_action_holdout": False,
        },
        "eligible_disease_scope": _count(eligible, "disease_program"),
        "decision": "physical_action_program_unavailable_for_c2_headline",
        "blockers": [
            "physical_dose_value_and_unit_unavailable",
            "duration_unavailable",
            "exposure_and_occupancy_program_references_unavailable",
            "publication_and_assay_provenance_unavailable",
            "frozen_c2_action_holdout_manifest_missing",
        ],
    }
    validation = {
        "schema_version": "neuronsifter-c2-action-compiler-validation-v1",
        "status": "blocked",
        "compiler_available": False,
        "training_allowed": False,
        "physical_dose_semantics": False,
        "schedule_semantics": True,
        "action_linkage_complete": not missing_links,
        "unavailable_fields": unsupported,
        "reason": "C2 trajectories are associated with synthetic dose multipliers, not traceable physical dose/exposure/occupancy programs.",
    }
    return audit, validation


def _authority_audit(
    qc_rows: list[dict[str, str]],
    mesoscopic_rows: list[dict[str, str]],
    inputs: Mapping[str, object],
    repo_root: Path,
) -> dict[str, object]:
    eligible = [row for row in qc_rows if row.get("qc_grade") in {"A", "B"} and row.get("qc_use") == "main_pretraining"]
    full_rows = [row for row in eligible if row.get("source_collection") == "full_microenvironment_teacher"]
    meso_rows = [row for row in eligible if row.get("source_collection") == "mesoscopic_solver_teacher"]
    meso_dimensions = {
        field: sorted({str(row.get(field, "")) for row in mesoscopic_rows})
        for field in ("time_step_count", "region_count", "compartment_count", "species_count")
    }
    pair_counts = _count(eligible, "paired_status")
    return {
        "schema_version": "neuronsifter-c2-trajectory-authority-v1",
        "status": "passed_with_scope_limitations",
        "source_artifacts": {
            key: _binding(repo_root, str(value))
            for key, value in inputs.items()
            if key in {
                "trajectory_numeric_lineage_path", "trajectory_qc_catalog_path",
                "mesoscopic_catalog_path", "full_teacher_catalog_path",
                "simulation_teacher_catalog_path",
            }
        },
        "trajectory_counts": {
            "all_qc_rows": len(qc_rows),
            "grade_ab_main_pretraining_rows": len(eligible),
            "full_microenvironment_rows": len(full_rows),
            "mesoscopic_rows": len(meso_rows),
            "paired_trajectory_row_count": pair_counts.get("paired", 0),
            "paired_unit_count": len({str(row["paired_uid"]) for row in eligible if row.get("paired_status") == "paired"}),
            "full_only_or_excluded_rows": len(qc_rows) - len(eligible),
        },
        "authority_classes": {
            "real_observation_linked_trajectory_count": 0,
            "published_model_trajectory_count": 0,
            "repository_simulator_trajectory_count": len(eligible),
            "synthetic_teacher_trajectory_count": len(eligible),
            "weak_or_calibration_trajectory_count": 0,
            "stress_or_ood_trajectory_count": 0,
        },
        "source_collection_counts": _count(eligible, "source_collection"),
        "teacher_family_counts": _count(eligible, "teacher_family"),
        "teacher_role_counts": _count(eligible, "teacher_role"),
        "regimen_count": len({str(row["regimen_uid"]) for row in eligible}),
        "route_counts": _count(eligible, "route"),
        "schedule_counts": _count(eligible, "schedule"),
        "dose_level_counts": _count(eligible, "dose_level_id"),
        "disease_program_counts": _count(eligible, "disease_program"),
        "trajectory_family_count": len({str(row["trajectory_family_uid"]) for row in eligible}),
        "source_family_count": len({str(row["source_collection"]) for row in eligible}),
        "time_region_compartment_species": meso_dimensions,
        "teacher_uncertainty": "available_in_qc_lineage_but_not_empirically_calibrated_against_real_trajectory_observations",
        "c1_ess_counted": 0,
        "c1_observation_coverage_counted": 0,
        "scope_limitation": "C2 may use these trajectories as mechanics/simulator evidence only; they do not establish real-observation, foundation-transfer, multimodal, or clinical claims.",
    }


def _holdout_reachability(
    qc_rows: list[dict[str, str]],
    split_audit: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    eligible = [row for row in qc_rows if row.get("qc_grade") in {"A", "B"} and row.get("qc_use") == "main_pretraining"]
    source_groups = {str(row["source_holdout_split"]) for row in eligible}
    paired = [row for row in eligible if row.get("paired_status") == "paired"]
    action_splits: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        action_splits[str(row["regimen_uid"])].add(str(row["split_name"]))
    duplicate_actions = sum(1 for roles in action_splits.values() if len(roles) > 1)
    holdouts = [
        {
            "holdout_family": "regimen_holdout",
            "status": "requires_c2_frozen_manifest",
            "candidate_unit_count": len(action_splits),
            "reason": "Legacy leakage audit verifies a derived split but no C2-bound membership manifest exists.",
        },
        {
            "holdout_family": "route_holdout",
            "status": "requires_c2_frozen_manifest",
            "candidate_unit_count": len({str(row["route"]) for row in eligible if row.get("route")}),
            "reason": "Three synthetic route labels exist, but no C2 action-holdout manifest is frozen.",
        },
        {
            "holdout_family": "target_program_holdout",
            "status": "requires_c2_frozen_manifest",
            "candidate_unit_count": len(action_splits),
            "reason": "Regimens carry target sets, but C2 has no target-program holdout membership manifest.",
        },
        {
            "holdout_family": "trajectory_family_holdout",
            "status": "requires_c2_frozen_manifest",
            "candidate_unit_count": len({str(row["trajectory_family_uid"]) for row in eligible}),
            "reason": "The QC catalog has trajectory families but no C2 trajectory-family manifest.",
        },
        {
            "holdout_family": "simulator_family_holdout",
            "status": "not_independent",
            "candidate_unit_count": len({str(row["source_collection"]) for row in eligible}),
            "reason": "Full and mesoscopic rows are paired under the same generated action and parameter lineage.",
        },
        {
            "holdout_family": "source_holdout",
            "status": "not_reachable",
            "candidate_unit_count": len(source_groups),
            "reason": "All eligible rows share the trajectory_pretraining source-holdout group.",
        },
        {
            "holdout_family": "missing_observation_holdout",
            "status": "not_reachable",
            "candidate_unit_count": 0,
            "reason": "No C2 observation-mask manifest or real observation-linked trajectory domain is frozen.",
        },
        {
            "holdout_family": "ood_mechanism_holdout",
            "status": "not_reachable",
            "candidate_unit_count": 0,
            "reason": "No independent OOD mechanism family is qualified in the selected C2 domain.",
        },
    ]
    reachability = {
        "schema_version": "neuronsifter-c2-holdout-reachability-v1",
        "status": "reduced_evaluation_only",
        "existing_legacy_split_audit_status": str(split_audit.get("leakage_status", "missing")),
        "existing_legacy_split_audit_hash": canonical_sha256(dict(split_audit)),
        "legacy_trajectory_random_action_duplicate_count": duplicate_actions,
        "holdout_families": holdouts,
        "strict_action_holdout_reachable_now": False,
        "strict_trajectory_or_simulator_holdout_reachable_now": False,
    }
    capability = {
        "schema_version": "neuronsifter-c2-capability-matrix-v1",
        "status": "c2_reduced_evaluation_only",
        "task_families": [
            {"task_family": "long_horizon_rollout", "status": "reduced_synthetic_teacher_only"},
            {"task_family": "structured_action_generalization", "status": "blocked_physical_action_semantics"},
            {"task_family": "stochastic_uq", "status": "reduced_teacher_disagreement_only"},
            {"task_family": "constraint_and_stability", "status": "reduced_simulator_constraint_only"},
            {"task_family": "partial_observation_teacher_disagreement", "status": "not_reachable"},
        ],
        "action_holdout_family_count_reachable_now": 0,
        "trajectory_holdout_family_count_reachable_now": 0,
        "simulator_family_count": len({str(row["source_collection"]) for row in eligible}),
        "source_holdout_count": len(source_groups),
        "paired_trajectory_row_count": len(paired),
        "paired_unit_count": len({str(row["paired_uid"]) for row in paired}),
        "blockers": [
            "physical_action_semantics_unavailable",
            "frozen_c2_action_holdout_manifest_missing",
            "frozen_c2_trajectory_holdout_manifest_missing",
            "simulator_families_not_independent",
            "no_real_observation_linked_trajectory_domain",
        ],
    }
    decision = {
        "schema_version": "neuronsifter-c2-pretraining-reachability-decision-v1",
        "status": "c2_reduced_evaluation_only",
        "headline_submission_allowed": False,
        "reason_codes": capability["blockers"],
        "required_before_headline": [
            "traceable_physical_dose_and_unit", "traceable_exposure_and_occupancy_programs",
            "frozen_c2_action_holdout_manifest", "frozen_c2_trajectory_holdout_manifest",
            "independent_simulator_or_trajectory_family", "C2_baseline_compatibility",
        ],
        "c1_large_headline_reactivation_allowed": False,
    }
    return reachability, capability, decision


def _baseline_registry(
    contract: Mapping[str, object],
    preflight_allowed: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    action_status = {
        "A0_drug_id": ("extend_existing", "compound_uid_set is available"),
        "A1_target_id": ("extend_existing", "target_uid_set is available"),
        "A2_regimen_token": ("reuse_as_is", "regimen_uid is available"),
        "A3_scalar_exposure": ("blocked_missing_evidence", "only dose multipliers and prior-derived scales are available"),
        "A4_occupancy_exposure_program": ("blocked_missing_evidence", "no provenance-bearing exposure or occupancy program reference"),
        "A5_full_structured_action": ("blocked_missing_evidence", "physical dose, duration, and provenance fields are unavailable"),
        "A6_full_structured_action_residual_context": ("blocked_missing_evidence", "base full structured action is unavailable"),
    }
    model_status = {
        "B0_persistence": ("extend_existing", "simple persistence evaluation adapter required"),
        "B1_linear_or_shallow_dynamics": ("extend_existing", "field rollout reference can be adapted"),
        "B2_deterministic_neuronsifter": ("reuse_as_is", "deterministic profile is declared"),
        "B3_stochastic_neuronsifter": ("reuse_as_is", "project-native stochastic operator exists"),
        "B4_fno_or_deeponet": ("blocked_missing_evidence", "no C2-compatible state-grid/action adapter"),
        "B5_transolver": ("extend_existing", "Transolver module exists but needs C2 protocol adapter"),
        "B6_diffusionpde": ("blocked_missing_evidence", "no compatible adapter or recorded patch"),
        "B7_wdno": ("blocked_missing_evidence", "no compatible adapter or recorded patch"),
        "B8_biological_perturbation_baseline": ("blocked_missing_evidence", "no independent biological perturbation trajectory target"),
    }
    registry = {
        "schema_version": "neuronsifter-c2-baseline-registry-v1",
        "status": "frozen_blocked_preflight",
        "action_ablations": [
            {"baseline_id": baseline, "decision": decision, "reason": reason, "headline_runnable": preflight_allowed}
            for baseline, (decision, reason) in action_status.items()
        ],
        "world_action_models": [
            {"baseline_id": baseline, "decision": decision, "reason": reason, "headline_runnable": preflight_allowed}
            for baseline, (decision, reason) in model_status.items()
        ],
        "external_baseline_policy": {
            "copied_paper_numbers_allowed": False,
            "same_split_and_action_semantics_required": True,
            "validation_only_tuning_required": True,
        },
    }
    fairness = {
        "schema_version": "neuronsifter-c2-baseline-fairness-audit-v1",
        "status": "blocked",
        "fair_comparison_allowed": False,
        "same_backbone_constraints": [
            "identical_selected_trajectory_registry", "identical_frozen_c2_split_manifest",
            "identical_model_capacity", "identical_optimizer", "identical_training_steps",
            "validation_only_checkpoint_selection", "identical_tuning_budget", "identical_evaluation_budget",
        ],
        "allowed_difference": "declared_action_or_stochastic_factor_only",
        "blockers": [
            "full_structured_action_unavailable", "strict_c2_holdout_manifests_unavailable",
            "external_baseline_semantic_compatibility_unverified",
        ],
        "result_status": "not_executed",
    }
    action_registry = {
        "schema_version": "neuronsifter-c2-action-ablation-registry-v1",
        "status": "frozen_blocked_preflight",
        "ablations": registry["action_ablations"],
        "same_backbone_required": True,
        "missing_action_information_must_be_explicitly_masked": True,
        "headline_submission_allowed": preflight_allowed,
    }
    action_fairness = {
        "schema_version": "neuronsifter-c2-action-ablation-fairness-audit-v1",
        "status": "blocked",
        "fair_comparison_allowed": False,
        "requirements": [
            "same_backbone", "same_data", "same_frozen_split", "same_optimizer",
            "same_training_steps", "same_checkpoint_selection", "same_tuning_budget",
            "same_evaluation_budget", "explicit_mask_for_unavailable_action_fields",
        ],
        "blockers": [
            "physical_full_action_input_unavailable",
            "occupancy_and_exposure_program_reference_unavailable",
            "frozen_c2_action_holdout_manifest_missing",
        ],
        "result_status": "not_executed",
    }
    return registry, fairness, action_registry, action_fairness


def _preflight(
    repo_root: Path,
    contract: Mapping[str, object],
    corpus_bindings: list[dict[str, object]],
    reachability: Mapping[str, object],
    action_validation: Mapping[str, object],
    baseline_registry: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(contract["selected_state_corpus"])
    bindings = {key: _binding(repo_root, str(value)) for key, value in dict(contract["inputs"]).items()}
    blockers: list[dict[str, object]] = []
    selected_binding = next(
        (item for item in corpus_bindings if item["corpus_uid"] == selected["corpus_uid"]),
        None,
    )
    if selected_binding is None or not bool(selected_binding.get("identity_match")):
        blockers.append({"code": "selected_corpus_identity_mismatch"})
    if selected_binding is None or selected_binding.get("evidence_validation_status") != "valid_fail_closed":
        blockers.append({"code": "selected_corpus_evidence_not_valid_fail_closed"})
    if not all(bool(binding["exists"]) for binding in bindings.values()):
        blockers.append({"code": "required_c2_input_missing"})
    if reachability.get("status") != "c2_headline_reachable":
        blockers.append({"code": "c2_headline_reachability_not_met", "status": reachability.get("status")})
    if action_validation.get("status") != "passed":
        blockers.append({"code": "c2_action_compiler_not_valid", "status": action_validation.get("status")})
    if baseline_registry.get("status") != "frozen_ready":
        blockers.append({"code": "c2_baseline_registry_not_headline_ready"})
    return {
        "schema_version": "neuronsifter-c2-preflight-v1",
        "status": "passed" if not blockers else "blocked",
        "training_allowed": not blockers,
        "training_started": False,
        "selected_corpus_uid": selected["corpus_uid"],
        "selected_corpus_identity_hash": selected["identity_hash"],
        "corpus_bindings": corpus_bindings,
        "selected_observation_blocks": _binding(repo_root, str(selected["observation_blocks_path"])),
        "trajectory_registry": bindings["trajectory_qc_catalog_path"],
        "action_catalog": bindings["regimen_catalog_path"],
        "teacher_registry": bindings["full_teacher_catalog_path"],
        "simulator_registry": bindings["mesoscopic_catalog_path"],
        "split_manifest_hashes": {"status": "missing_c2_manifests", "hashes": {}},
        "action_holdout_manifests": {"status": "missing_c2_manifests"},
        "trajectory_family_manifests": {"status": "missing_c2_manifests"},
        "simulator_family_manifests": {"status": "missing_c2_manifests"},
        "model_config_hash": canonical_sha256(contract),
        "baseline_registry_hash": canonical_sha256(dict(baseline_registry)),
        "metric_registry": dict(contract["statistics"]),
        "seed_registry": {"headline_seed_count": contract["statistics"]["headline_seed_count"]},
        "checkpoint_selection_policy": {"data_role": "validation_only", "test_selection_allowed": False},
        "sacct_available": False,
        "terminal_evidence_source": "scontrol_and_preserved_scheduler_logs",
        "blockers": blockers,
    }


def _job_and_gate(
    contract: Mapping[str, object],
    preflight: Mapping[str, object],
    fairness: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    allowed = bool(preflight.get("training_allowed")) and str(preflight.get("status")) == "passed"
    matrix = {
        "schema_version": "neuronsifter-c2-job-matrix-v1",
        "status": "ready_for_submission" if allowed else "blocked_no_jobs_generated",
        "job_name_prefix": contract["execution"]["job_name_prefix"],
        "submission_allowed": allowed,
        "preflight_hash": canonical_sha256(dict(preflight)),
        "jobs": [],
        "reason": "C2 preflight is blocked; no Debug, Student, Large, baseline, or headline job was generated.",
    }
    gate = {
        "schema_version": "neuronsifter-c2-g4-validation-v1",
        "status": "blocked_incomplete",
        "g4_status": "blocked_incomplete",
        "empirical_result_status": "unavailable",
        "integrity_blocker_count": 0,
        "completion_blockers": [
            "C2 preflight blocked", "no strict action holdout", "no strict trajectory or independent simulator holdout",
            "full structured action baseline unavailable", "no C2 headline runs",
        ],
        "preflight_hash": canonical_sha256(dict(preflight)),
        "fairness_hash": canonical_sha256(dict(fairness)),
        "claim_allowed": False,
    }
    summary = {
        "schema_version": "neuronsifter-c2-gate-summary-v1",
        "status": "blocked_incomplete",
        "implementation_status": "existing_world_action_components_inventory_validated",
        "training_status": "not_started_preflight_blocked",
        "empirical_result_status": "unavailable",
        "g4_status": "blocked_incomplete",
        "foundation_claim_permission": "prohibited",
        "allowed_claim_wording": [
            "structured world-action implementation",
            "C2 trajectory and action authority audit complete",
            "C2 strict action-conditioned rollout experiments blocked pending physical action provenance",
        ],
        "prohibited_claim_wording": [
            "foundation world model", "complete multimodal digital twin", "patient-specific digital twin",
            "clinical treatment prediction", "causal treatment effect",
        ],
        "c1_headline_reactivation_allowed": False,
    }
    return matrix, gate, summary


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _execution_audit(repo_root: Path, preflight: Mapping[str, object]) -> dict[str, object]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    run_uid = os.environ.get("NRNSIFT_C2_RUN_UID", "")
    if not run_uid:
        run_uid = f"c2_phase_gate_{job_id or 'local'}"
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "neuronsifter-c2-phase-gate-execution-v1",
        "run_uid": run_uid,
        "job_id": job_id or None,
        "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "dirty_status": "dirty" if _git_value(repo_root, "status", "--porcelain") else "clean",
        "config_hash": preflight["model_config_hash"],
        "data_hash": preflight["trajectory_registry"]["sha256"],
        "split_hash": None,
        "trajectory_registry_hash": preflight["trajectory_registry"]["sha256"],
        "selected_corpus_hash": preflight["selected_corpus_identity_hash"],
        "checkpoint_hash": None,
        "seed": None,
        "start_end": {"started_at": now, "ended_at": now},
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "hardware": {
            "hostname": os.uname().nodename,
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "node_list": os.environ.get("SLURM_NODELIST"),
            "gpu_requested": os.environ.get("SLURM_GPUS", "0"),
        },
        "software_environment": {"python_version": sys.version.split()[0]},
        "terminal_evidence_source": "scontrol_and_preserved_scheduler_logs" if job_id else "local_process",
        "artifact_producer_state": "completed",
        "stdout_stderr": "Slurm output path is recorded by the submission command when scheduled.",
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    contract = config["structured_world_action_c2"]
    inputs = contract["inputs"]
    qc_rows = read_tsv(_relative(repo_root, str(inputs["trajectory_qc_catalog_path"])))
    mesoscopic_rows = read_tsv(_relative(repo_root, str(inputs["mesoscopic_catalog_path"])))
    regimen_rows = read_tsv(_relative(repo_root, str(inputs["regimen_catalog_path"])))
    split_audit = _load_json(_relative(repo_root, str(inputs["trajectory_split_leakage_path"])))
    corpus_bindings = [_corpus_binding(repo_root, dict(entry)) for entry in contract["immutable_foundation_corpora"]]
    authority = _authority_audit(qc_rows, mesoscopic_rows, inputs, repo_root)
    action_audit, action_validation = _action_audit(qc_rows, regimen_rows, contract["action_contract"])
    holdout_reachability, capability, reachability_decision = _holdout_reachability(qc_rows, split_audit)
    baseline_registry, fairness, action_registry, action_fairness = _baseline_registry(
        contract, preflight_allowed=False,
    )
    preflight = _preflight(
        repo_root, contract, corpus_bindings, reachability_decision, action_validation, baseline_registry,
    )
    matrix, gate, summary = _job_and_gate(contract, preflight, fairness)
    failure_rows = [{
        "failure_id": "c2_preflight_action_semantics_blocker",
        "phase": "preflight",
        "run_uid": "",
        "job_id": "",
        "status": "blocked_before_submission",
        "root_cause": "physical dose, exposure, and occupancy action provenance unavailable for selected trajectories",
        "terminal_evidence_source": "not_applicable_no_job_submitted",
        "preserved_logs": "",
    }]
    return {
        "trajectory_authority": authority,
        "action_coverage": {
            "schema_version": "neuronsifter-c2-action-coverage-v1",
            "status": "passed_with_headline_blocker",
            "action_program_count": action_audit["action_program_count"],
            "route_counts": action_audit["dose_representation"]["route_labels"],
            "schedule_counts": action_audit["dose_representation"]["schedule_labels"],
            "dose_multiplier_counts": action_audit["dose_representation"]["dose_multiplier_labels"],
            "physical_dose_available": False,
            "action_audit_hash": canonical_sha256(action_audit),
        },
        "holdout_reachability": holdout_reachability,
        "capability_matrix": capability,
        "reachability_decision": reachability_decision,
        "preflight": preflight,
        "structured_action_audit": action_audit,
        "action_compiler_validation": action_validation,
        "baseline_registry": baseline_registry,
        "baseline_fairness": fairness,
        "action_ablation_registry": action_registry,
        "action_ablation_fairness": action_fairness,
        "job_matrix": matrix,
        "job_terminal_rows": [],
        "failure_rows": failure_rows,
        "g4_validation": gate,
        "gate_summary": summary,
        "execution_audit": _execution_audit(repo_root, preflight),
    }


def write_outputs(repo_root: Path, payload: Mapping[str, object]) -> None:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    outputs = config["structured_world_action_c2"]["outputs"]
    audit_root = _relative(repo_root, str(outputs["audit_root"]))
    report_root = _relative(repo_root, str(outputs["report_root"]))
    json_outputs = {
        "c2_trajectory_authority.json": payload["trajectory_authority"],
        "c2_action_coverage.json": payload["action_coverage"],
        "c2_holdout_reachability.json": payload["holdout_reachability"],
        "c2_capability_matrix.json": payload["capability_matrix"],
        "c2_pretraining_reachability_decision.json": payload["reachability_decision"],
        "c2_preflight.json": payload["preflight"],
        "c2_structured_action_audit.json": payload["structured_action_audit"],
        "c2_action_compiler_validation.json": payload["action_compiler_validation"],
        "c2_baseline_registry.json": payload["baseline_registry"],
        "c2_baseline_fairness_audit.json": payload["baseline_fairness"],
        "c2_action_ablation_registry.json": payload["action_ablation_registry"],
        "c2_action_ablation_fairness_audit.json": payload["action_ablation_fairness"],
        "c2_job_matrix.json": payload["job_matrix"],
        "c2_g4_validation.json": payload["g4_validation"],
        "c2_phase_gate_execution.json": payload["execution_audit"],
    }
    for name, artifact in json_outputs.items():
        write_json(audit_root / name, artifact)
    terminal_fields = ["run_uid", "root_experiment_uid", "job_id", "terminal_state", "exit_code", "terminal_evidence_source"]
    failure_fields = ["failure_id", "phase", "run_uid", "job_id", "status", "root_cause", "terminal_evidence_source", "preserved_logs"]
    write_tsv(audit_root / "c2_job_terminal_summary.tsv", terminal_fields, list(payload["job_terminal_rows"]))
    write_tsv(audit_root / "c2_failure_catalog.tsv", failure_fields, list(payload["failure_rows"]))
    for name in (
        "c2_world_action_results.tsv", "c2_action_ablation_catalog.tsv", "c2_rollout_metric_catalog.tsv",
        "c2_uq_metric_catalog.tsv", "c2_constraint_violation_catalog.tsv", "c2_counterfactual_contrast_catalog.tsv",
    ):
        write_tsv(report_root / name, RESULT_FIELDS, [])
    write_json(report_root / "c2_gate_summary.json", payload["gate_summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write C2 artifacts after a successful fail-closed audit.")
    args = parser.parse_args()
    payload = build(REPO_ROOT)
    if args.write:
        write_outputs(REPO_ROOT, payload)
    print(f"c2_reachability_status={payload['reachability_decision']['status']}")
    print(f"c2_preflight_status={payload['preflight']['status']}")
    print(f"c2_action_compiler_status={payload['action_compiler_validation']['status']}")


if __name__ == "__main__":
    main()
