"""Adjudicate formal zero-shot checkpoint authority without evaluation access."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    EVALUATION_MATRIX_FIELDS,
    _binding,
    _payload_binding,
    _read_json,
    _repo_path,
    sha256_file,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTRACT_PATH = REPO_ROOT / "Libs/configs/validation/zero_shot_method_authority.yaml"
CONTRACT_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/zero_shot_method_authority_config.schema.json"
)
AUTHORITY_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/zero_shot_method_authority.schema.json"
)
INDEPENDENCE_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/checkpoint_training_selection_independence.schema.json"
)
DECISION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/zero_shot_protocol_executability_decision.schema.json"
)
AUTHORIZATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/zero_shot_method_execution_authorization.schema.json"
)
VALIDATION_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/zero_shot_method_authority_validation.schema.json"
)

CANONICAL_TWIN_PATH = REPO_ROOT / "Output/audits/round25_u0_canonical_static_twin.json"
STATIC_REFERENCE_PATH = REPO_ROOT / "Output/audits/round23_static_dev_reference.json"
TRAINING_MANIFEST_PATH = REPO_ROOT / "Output/audits/round23_training_manifest.json"
CHECKPOINT_VALIDATION_PATH = (
    REPO_ROOT / "Output/audits/round23_checkpoint_validation.json"
)
EVALUATION_MATRIX_PATH = (
    REPO_ROOT
    / "Input/manifests/retrospective_public/six_context_zero_shot_evaluation_matrix.tsv"
)
STATIC_CONFIG_PATH = REPO_ROOT / "Libs/configs/neuronsifter/static_decision_loop.yaml"
PRIOR_REVIEW_PATH = (
    REPO_ROOT / "Output/audits/prospective_protocol_scientific_review.json"
)
PRIOR_DECISION_PATH = REPO_ROOT / "Output/audits/prospective_protocol_go_no_go.json"
PRIOR_METHOD_PATH = REPO_ROOT / "Output/audits/frozen_zero_shot_method_manifest.json"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_zero_shot_method_authority.py"
VALIDATION_WRAPPER_PATH = (
    REPO_ROOT / "Libs/exp/evaluation/run_zero_shot_method_authority_validation.sbatch"
)


def _schema_errors(payload: Mapping[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("zero_shot_method_authority")
    if not isinstance(policy, dict):
        raise ValueError("Missing zero_shot_method_authority configuration")
    errors = _schema_errors(policy, CONTRACT_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Method-authority config schema errors: " + "; ".join(errors[:8])
        )
    return policy


def _git(args: list[str]) -> tuple[int, bytes, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        timeout=30,
    )
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr.decode("utf-8", errors="replace").strip(),
    )


def _git_blob_sha256(commit: str, path: str) -> str | None:
    return_code, content, _ = _git(["show", f"{commit}:{path}"])
    return hashlib.sha256(content).hexdigest() if return_code == 0 else None


def _read_matrix() -> list[dict[str, str]]:
    with EVALUATION_MATRIX_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        observed_fields = tuple(reader.fieldnames or ())
        if observed_fields != EVALUATION_MATRIX_FIELDS:
            raise ValueError(
                "Evaluation matrix header is not the frozen outcome-free contract"
            )
        prohibited_fields = {
            "numeric_outcome",
            "arm_mean",
            "uncertainty",
            "p_value",
            "effect_direction",
            "winner",
            "ranking_outcome",
        }
        if prohibited_fields & set(observed_fields):
            raise ValueError("Evaluation matrix exposes outcome-bearing columns")
        rows = list(reader)
    if len(rows) != 6:
        raise ValueError("Expected exactly six outcome-free evaluation identity rows")
    if any(row.get("execution_status") != "NOT_RUN" for row in rows):
        raise ValueError("Evaluation matrix is not outcome-blind and unexecuted")
    if any(row.get("outcome_access_during_review") != "0" for row in rows):
        raise ValueError("Evaluation outcome access is nonzero")
    return rows


def _evaluation_identifiers(rows: Iterable[Mapping[str, str]]) -> list[str]:
    identifiers: set[str] = set()
    for row in rows:
        identifiers.add(row["context_uid"])
        for field in ("publication_uids", "action_uids", "control_uids"):
            identifiers.update(value for value in row[field].split("|") if value)
    return sorted(identifiers)


def _exact_overlap(
    identifiers: Iterable[str], source_paths: Iterable[Path]
) -> dict[str, list[str]]:
    identifier_list = list(identifiers)
    result: dict[str, list[str]] = {}
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        result[path.relative_to(REPO_ROOT).as_posix()] = [
            identifier for identifier in identifier_list if identifier in text
        ]
    return result


def _c1_family_counts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        counts = Counter(str(row.get("family_uid", "")) for row in rows)
    return dict(sorted(counts.items()))


def _checkpoint_validation_row(
    validation: Mapping[str, Any], checkpoint_path: str
) -> dict[str, Any]:
    return next(
        (
            dict(row)
            for row in validation.get("rows", [])
            if row.get("path") == checkpoint_path
        ),
        {},
    )


def _candidate_record(
    *,
    component_uid: str,
    frozen_record: Mapping[str, Any],
    run_result: Mapping[str, Any],
    checkpoint_validation: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    exact_overlap_count: int,
    exact_overlap_scope_complete: bool,
    semantic_assessment: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = _repo_path(str(frozen_record["path"]))
    expected_sha = str(frozen_record["sha256"])
    observed_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    validation_row = _checkpoint_validation_row(
        checkpoint_validation, str(frozen_record["path"])
    )
    large_seed_count = int(
        training_manifest.get("selected_large_seed_counts", {}).get(
            "c1" if component_uid == "scientific_belief_encoder" else "c2s", 0
        )
    )
    training_lineage_complete = bool(run_result.get("training_lineage_complete"))
    if component_uid == "scientific_belief_encoder":
        selection_metric = run_result.get("checkpoint_selection_metric")
        selection_status = run_result.get("checkpoint_selection_status")
        selection_lineage_complete = bool(
            selection_metric
            and selection_metric != "train_objective_smoke_only"
            and selection_status
            and selection_status != "not_available_smoke_only"
        )
        selection_reason = "explicit_smoke_only_selection_without_validation_checkpoint"
        training_scope = {
            "steps": run_result.get("steps"),
            "effective_observations_seen": run_result.get(
                "effective_observations_seen"
            ),
            "effective_biological_units_seen": run_result.get(
                "effective_biological_units_seen"
            ),
            "corpus_hash": run_result.get("corpus_hash"),
        }
    else:
        selection_metric = run_result.get("checkpoint_selection_metric")
        selection_status = run_result.get("checkpoint_selection_status")
        selection_lineage_complete = bool(selection_metric and selection_status)
        selection_reason = (
            "three_large_seeds_exist_but_selected_checkpoint_rule_is_not_recorded"
        )
        training_scope = {
            "steps": run_result.get("steps"),
            "accessed_simulation_indices": run_result.get(
                "accessed_simulation_indices", []
            ),
            "input_identity": run_result.get("input_identity", {}),
        }
    code_snapshot_reconstructible = bool(
        run_result.get("git_commit")
        and run_result.get("git_dirty") is False
        and run_result.get("git_dirty_file_count") == 0
    )
    resolved_config_snapshot_bound = bool(
        run_result.get("resolved_config_path")
        and run_result.get("resolved_config_sha256")
    )
    semantic_independence_resolved = semantic_assessment.get("status") in {
        "no_semantic_overlap",
        "semantically_independent",
    }
    exact_independence_resolved = (
        exact_overlap_scope_complete and exact_overlap_count == 0
    )
    evidence_role = run_result.get("evidence_role")
    publication_claim_eligible = run_result.get("publication_claim_eligible") is True
    formal_eligibility = (
        evidence_role != "development_only" and publication_claim_eligible
    )
    requirement_status = {
        "checkpoint_identity_complete": observed_sha == expected_sha
        and validation_row.get("payload_loadable") is True,
        "config_identity_complete": resolved_config_snapshot_bound,
        "code_identity_complete": code_snapshot_reconstructible,
        "training_lineage_complete": training_lineage_complete,
        "model_selection_lineage_complete": selection_lineage_complete,
        "exact_evaluation_overlap_resolved": exact_independence_resolved,
        "semantic_evaluation_overlap_resolved": semantic_independence_resolved,
        "formal_eligibility_positive": formal_eligibility,
    }
    blockers = [
        requirement for requirement, passed in requirement_status.items() if not passed
    ]
    if observed_sha != expected_sha:
        blockers.append("checkpoint_hash_mismatch")
    if exact_overlap_count:
        blockers.append("exact_evaluation_identifier_overlap")
    return {
        "component_uid": component_uid,
        "checkpoint": {
            "path": frozen_record["path"],
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "hash_matches": observed_sha == expected_sha,
            "prior_payload_loadable": validation_row.get("payload_loadable") is True,
            "prior_finite_model_state": validation_row.get("finite_model_state")
            is True,
            "prior_roundtrip_verified": (
                validation_row.get("c1_prediction_roundtrip") is True
                if component_uid == "scientific_belief_encoder"
                else validation_row.get("payload_boundary_matches") is True
            ),
        },
        "training": {
            **training_scope,
            "lineage_complete": training_lineage_complete,
            "evidence_role": evidence_role,
            "protected_test_access": run_result.get("protected_test_access"),
        },
        "selection": {
            "metric": selection_metric,
            "status": selection_status,
            "large_seed_candidate_count": large_seed_count,
            "lineage_complete": selection_lineage_complete,
            "reason": selection_reason,
        },
        "code_identity": {
            "git_commit": run_result.get("git_commit"),
            "git_dirty": run_result.get("git_dirty"),
            "git_dirty_file_count": run_result.get("git_dirty_file_count"),
            "git_full_diff_hash": run_result.get("git_full_diff_hash"),
            "snapshot_reconstructible": code_snapshot_reconstructible,
        },
        "config_identity": {
            "config_path": run_result.get("config_path"),
            "config_file_sha256": run_result.get(
                "config_file_hash", run_result.get("config_sha256")
            ),
            "effective_config_sha256": run_result.get(
                "config_hash", run_result.get("effective_profile_sha256")
            ),
            "resolved_snapshot_bound": resolved_config_snapshot_bound,
        },
        "lineage": {
            "exact_evaluation_identifier_overlap_count": exact_overlap_count,
            "exact_overlap_scope_complete": exact_overlap_scope_complete,
            "semantic_overlap_assessment": dict(semantic_assessment),
            "training_source_ledger_complete": training_lineage_complete,
            "selection_source_ledger_complete": selection_lineage_complete,
        },
        "formal_eligibility": {
            "evidence_role": evidence_role,
            "publication_claim_eligible": publication_claim_eligible,
            "resolved": True,
            "status": "eligible" if formal_eligibility else "not_eligible",
        },
        "requirement_status": requirement_status,
        "prospective_zero_shot_eligible": all(requirement_status.values()),
        "blockers": sorted(set(blockers)),
    }


def _build_independence_audit(
    policy: Mapping[str, Any],
    canonical: Mapping[str, Any],
    matrix_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Any], dict[str, Any]]:
    identifiers = _evaluation_identifiers(matrix_rows)
    c1_split_path = _repo_path(
        str(canonical["development_data_splits"]["c1"]["role_binding_path"])
    )
    c2s_pool_path = _repo_path(str(canonical["development_data_splits"]["c2s"]["path"]))
    overlap_sources = (TRAINING_MANIFEST_PATH, c1_split_path, c2s_pool_path)
    overlaps = _exact_overlap(identifiers, overlap_sources)
    c1_counts = _c1_family_counts(c1_split_path)
    c2s_pool = _read_json(c2s_pool_path)
    c2s_routes = sorted(
        {
            str(record["action"]["route"])
            for record in c2s_pool.get("records", [])
            if isinstance(record.get("action"), dict) and record["action"].get("route")
        }
    )
    evaluation_namespaces = sorted(
        {
            action_uid.split(":", 2)[1]
            for row in matrix_rows
            for action_uid in row["action_uids"].split("|")
            if action_uid.startswith("structured_action:")
        }
    )
    c1_semantic = {
        "status": "unresolved_semantic_overlap",
        "validation_family_counts": c1_counts,
        "compound_family_present": c1_counts.get(
            "compound_dose_time_perturbation_transcriptomics", 0
        )
        > 0,
        "publication_context_action_endpoint_ledger_complete": False,
    }
    c2s_semantic = {
        "status": "action_domain_mismatch",
        "development_action_routes": c2s_routes,
        "evaluation_action_namespaces": evaluation_namespaces,
        "publication_context_action_endpoint_ledger_complete": False,
    }
    training_manifest_label = TRAINING_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()
    c1_split_label = c1_split_path.relative_to(REPO_ROOT).as_posix()
    c2s_pool_label = c2s_pool_path.relative_to(REPO_ROOT).as_posix()
    c1_overlap_count = sum(
        len(overlaps.get(label, []))
        for label in (training_manifest_label, c1_split_label)
    )
    c2s_overlap_count = sum(
        len(overlaps.get(label, []))
        for label in (training_manifest_label, c2s_pool_label)
    )
    audit = {
        "schema_version": "neuronsifter-checkpoint-training-selection-independence-v1",
        "status": "formal_independence_not_proven",
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "evaluation_identity_source": {
            **_binding(EVALUATION_MATRIX_PATH),
            "context_count": len(matrix_rows),
            "identifier_count": len(identifiers),
            "numeric_outcome_fields_present": False,
            "execution_status": "NOT_RUN",
        },
        "bounded_exact_overlap_check": {
            "source_count": len(overlap_sources),
            "sources": sorted(overlaps),
            "matches_by_source": overlaps,
            "match_count": sum(len(values) for values in overlaps.values()),
            "scope_complete_for_all_training_and_selection_inputs": False,
        },
        "semantic_overlap_assessment": {
            "scientific_belief_encoder": c1_semantic,
            "world_action_capability": c2s_semantic,
            "formal_semantic_independence_resolved": False,
        },
        "candidate_findings": [
            {
                "component_uid": "scientific_belief_encoder",
                "exact_overlap_count": c1_overlap_count,
                "semantic_status": c1_semantic["status"],
                "training_lineage_complete": False,
                "selection_lineage_complete": False,
            },
            {
                "component_uid": "world_action_capability",
                "exact_overlap_count": c2s_overlap_count,
                "semantic_status": c2s_semantic["status"],
                "training_lineage_complete": False,
                "selection_lineage_complete": False,
            },
        ],
        "formal_independence_proven": False,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "artifact_bindings": {
            "training_manifest": _binding(TRAINING_MANIFEST_PATH),
            "c1_validation_split": _binding(c1_split_path),
            "c2s_development_pool": _binding(c2s_pool_path),
        },
    }
    return audit, overlaps, c1_semantic, c2s_semantic


def _build_authority(
    policy: Mapping[str, Any],
    independence: Mapping[str, Any],
    independence_path: Path,
) -> dict[str, Any]:
    canonical = _read_json(CANONICAL_TWIN_PATH)
    training = _read_json(TRAINING_MANIFEST_PATH)
    validation = _read_json(CHECKPOINT_VALIDATION_PATH)
    c1_run = _read_json(
        _repo_path(
            policy["candidate_components"]["scientific_belief_encoder"]["run_result"]
        )
    )
    c2s_run = _read_json(
        _repo_path(
            policy["candidate_components"]["world_action_capability"]["run_result"]
        )
    )
    overlaps = independence["bounded_exact_overlap_check"]["matches_by_source"]
    c1_semantic = independence["semantic_overlap_assessment"][
        "scientific_belief_encoder"
    ]
    c2s_semantic = independence["semantic_overlap_assessment"][
        "world_action_capability"
    ]
    training_manifest_label = TRAINING_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()
    c1_split_label = str(
        canonical["development_data_splits"]["c1"]["role_binding_path"]
    )
    c2s_pool_label = str(canonical["development_data_splits"]["c2s"]["path"])
    c1_overlap_count = sum(
        len(overlaps.get(label, []))
        for label in (training_manifest_label, c1_split_label)
    )
    c2s_overlap_count = sum(
        len(overlaps.get(label, []))
        for label in (training_manifest_label, c2s_pool_label)
    )
    exact_overlap_scope_complete = bool(
        independence["bounded_exact_overlap_check"][
            "scope_complete_for_all_training_and_selection_inputs"
        ]
    )
    candidates = [
        _candidate_record(
            component_uid="scientific_belief_encoder",
            frozen_record=canonical["scientific_belief_encoder"],
            run_result=c1_run,
            checkpoint_validation=validation,
            training_manifest=training,
            exact_overlap_count=c1_overlap_count,
            exact_overlap_scope_complete=exact_overlap_scope_complete,
            semantic_assessment=c1_semantic,
        ),
        _candidate_record(
            component_uid="world_action_capability",
            frozen_record=canonical["world_action_capability"],
            run_result=c2s_run,
            checkpoint_validation=validation,
            training_manifest=training,
            exact_overlap_count=c2s_overlap_count,
            exact_overlap_scope_complete=exact_overlap_scope_complete,
            semantic_assessment=c2s_semantic,
        ),
    ]
    config_rel = str(policy["historical_config"]["path"])
    expected_config_sha = str(policy["historical_config"]["expected_sha256"])
    tracked_rc, tracked_output, _ = _git(["ls-files", "--error-unmatch", config_rel])
    history_rc, history_output, history_error = _git(
        ["log", "--all", "--follow", "--format=%H", "--", config_rel]
    )
    c1_commit = str(c1_run.get("git_commit") or "")
    committed_stack_sha = (
        _git_blob_sha256(c1_commit, "Libs/configs/config_stack.yaml")
        if c1_commit
        else None
    )
    current_config_sha = sha256_file(STATIC_CONFIG_PATH)
    history_commits = [
        value for value in history_output.decode("utf-8").splitlines() if value
    ]
    history_snapshots = [
        {"commit": commit, "sha256": _git_blob_sha256(commit, config_rel)}
        for commit in history_commits
    ]
    historical_recoverable = current_config_sha == expected_config_sha or any(
        record["sha256"] == expected_config_sha for record in history_snapshots
    )
    candidate_eligible = [
        record["prospective_zero_shot_eligible"] for record in candidates
    ]
    status = (
        "FORMAL_ZERO_SHOT_CHECKPOINT_AVAILABLE"
        if all(candidate_eligible)
        else policy["stop_policy"]["no_formal_checkpoint_status"]
    )
    return {
        "schema_version": "neuronsifter-zero-shot-method-authority-v1",
        "status": status,
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "historical_freeze": {
            "config_path": config_rel,
            "expected_sha256": expected_config_sha,
            "current_sha256": current_config_sha,
            "current_matches_expected": current_config_sha == expected_config_sha,
            "git_tracked": tracked_rc == 0 and bool(tracked_output.strip()),
            "git_history_query_succeeded": history_rc == 0,
            "git_history_commit_count": len(history_commits),
            "git_history_snapshots": history_snapshots,
            "git_history_error": history_error,
            "recoverable_from_bounded_authorized_sources": historical_recoverable,
            "global_absence_claimed": False,
            "old_freeze_identity_valid": historical_recoverable
            and current_config_sha == expected_config_sha,
            "current_config_stack_sha256": sha256_file(CONFIG_PATH),
            "c1_run_config_file_sha256": c1_run.get("config_file_hash"),
            "c1_effective_config_sha256": c1_run.get("config_hash"),
            "c2s_run_config_file_sha256": c2s_run.get("config_sha256"),
            "c2s_effective_profile_sha256": c2s_run.get("effective_profile_sha256"),
            "c1_commit_config_stack_sha256": committed_stack_sha,
            "new_prospective_method_version_created": False,
        },
        "candidate_authority_matrix": candidates,
        "composite_method_authority": {
            "candidate_uid": canonical.get("canonical_twin"),
            "required_components": [
                "scientific_belief_encoder",
                "world_action_capability",
            ],
            "all_component_checkpoint_hashes_match": all(
                record["checkpoint"]["hash_matches"] for record in candidates
            ),
            "all_components_formally_eligible": all(candidate_eligible),
            "training_lineage_complete": False,
            "selection_lineage_complete": False,
            "evaluation_independence_defensible": False,
            "formal_zero_shot_method_available": all(candidate_eligible),
        },
        "hard_stop": {
            "triggered": not all(candidate_eligible),
            "terminal_reason": (
                None
                if all(candidate_eligible)
                else policy["stop_policy"]["no_formal_checkpoint_status"]
            ),
            "deterministic_intervention_adapter": policy["stop_policy"][
                "downstream_status"
            ],
            "cpu_inference_uncertainty_abstention": policy["stop_policy"][
                "downstream_status"
            ],
            "baseline_implementation_binding": policy["stop_policy"][
                "downstream_status"
            ],
            "non_evaluation_end_to_end_qualification": policy["stop_policy"][
                "downstream_status"
            ],
            "reason": "The user-specified checkpoint authority stop rule applies before adapter or inference work.",
        },
        "formal_authority": {
            "protocol_executable": all(candidate_eligible),
            "execution_authorized": False,
            "world_action_authorized": False,
            "gpu_authorized": False,
            "tri_authorized": False,
            "agent_in_twin_authorized": False,
            "coevolution_authorized": False,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        },
        "artifact_bindings": {
            "canonical_static_twin": _binding(CANONICAL_TWIN_PATH),
            "static_development_reference": _binding(STATIC_REFERENCE_PATH),
            "training_manifest": _binding(TRAINING_MANIFEST_PATH),
            "checkpoint_validation": _binding(CHECKPOINT_VALIDATION_PATH),
            "independence_audit": _payload_binding(independence_path, independence),
            "prior_scientific_review": _binding(PRIOR_REVIEW_PATH),
            "prior_go_no_go_decision": _binding(PRIOR_DECISION_PATH),
            "prior_method_manifest": _binding(PRIOR_METHOD_PATH),
            "implementation": _binding(BUILDER_PATH),
            "validation_wrapper": _binding(VALIDATION_WRAPPER_PATH),
            "authority_contract": _binding(CONTRACT_PATH),
            "authority_contract_schema": _binding(CONTRACT_SCHEMA_PATH),
            "authority_schema": _binding(AUTHORITY_SCHEMA_PATH),
            "independence_schema": _binding(INDEPENDENCE_SCHEMA_PATH),
            "decision_schema": _binding(DECISION_SCHEMA_PATH),
            "authorization_schema": _binding(AUTHORIZATION_SCHEMA_PATH),
            "validation_schema": _binding(VALIDATION_SCHEMA_PATH),
        },
    }


def _build_decision(
    policy: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_path: Path,
    independence: Mapping[str, Any],
    independence_path: Path,
) -> dict[str, Any]:
    executable = authority["status"] == "FORMAL_ZERO_SHOT_CHECKPOINT_AVAILABLE"
    return {
        "schema_version": "neuronsifter-zero-shot-protocol-executability-decision-v1",
        "status": (
            "PROTOCOL_READY_FOR_ZERO_SHOT_EXECUTION"
            if executable
            else policy["stop_policy"]["terminal_protocol_status"]
        ),
        "terminal_blocker": (
            None if executable else policy["stop_policy"]["no_formal_checkpoint_status"]
        ),
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "work_package_status": {
            "formal_method_identity": (
                "COMPLETE_POSITIVE" if executable else "COMPLETE_NEGATIVE"
            ),
            "deterministic_intervention_adapter": (
                "PENDING" if executable else policy["stop_policy"]["downstream_status"]
            ),
            "cpu_inference_uncertainty_support_ood_abstention": (
                "PENDING" if executable else policy["stop_policy"]["downstream_status"]
            ),
            "baseline_implementation_binding": (
                "PENDING" if executable else policy["stop_policy"]["downstream_status"]
            ),
            "non_evaluation_end_to_end_qualification": (
                "PENDING" if executable else policy["stop_policy"]["downstream_status"]
            ),
            "protocol_re_adjudication": "COMPLETE_NEGATIVE",
        },
        "scientific_interpretation": (
            "No model-performance conclusion was produced. Existing checkpoint bytes are intact, "
            "but no candidate has complete reconstructible training, selection, code, config, "
            "independence, and formal-eligibility authority."
        ),
        "minimum_legal_future_route": [
            "obtain explicit authorization for a new prospective method identity",
            "freeze a reconstructible code, config, training, and selection lineage before any six-context inference",
            "use no six-context outcomes for training, selection, adapter design, or threshold calibration",
            "do not describe a new method identity as recovery of the unrecoverable historical freeze",
        ],
        "prohibited_actions": [
            "continue deterministic adapter implementation after the checkpoint hard stop",
            "run six-context inference",
            "train or calibrate on six-context data",
            "promote the C1 smoke checkpoint to formal authority",
            "rename the current config as the historical frozen snapshot",
            "open Tool Calling, Agent-in-Twin, or Co-Evolution",
            "access protected holdout outcomes",
        ],
        "execution_authority": {
            "protocol_executable": executable,
            "execution_authorized": False,
            "world_action_authorized": False,
            "gpu_authorized": False,
            "tri_authorized": False,
            "agent_in_twin_authorized": False,
            "coevolution_authorized": False,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        },
        "artifact_bindings": {
            "method_authority": _payload_binding(authority_path, authority),
            "independence_audit": _payload_binding(independence_path, independence),
            "evaluation_matrix": _binding(EVALUATION_MATRIX_PATH),
            "prior_claim_to_evidence_matrix": _binding(
                REPO_ROOT / "Output/audits/prospective_claim_to_evidence_matrix.json"
            ),
        },
    }


def _build_authorization(
    policy: Mapping[str, Any], decision: Mapping[str, Any], decision_path: Path
) -> dict[str, Any]:
    executable = decision["status"] == "PROTOCOL_READY_FOR_ZERO_SHOT_EXECUTION"
    return {
        "schema_version": "neuronsifter-zero-shot-method-execution-authorization-v1",
        "status": "issuable_pending_user_authorization"
        if executable
        else "not_issuable",
        "protocol_executable": executable,
        "execution_authorized": False,
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "allowed_future_scope": (
            ["frozen_zero_shot_twin", "selective_ranking", "abstention_evaluation"]
            if executable
            else []
        ),
        "blocked_scope": [
            "six_context_inference",
            "model_training",
            "evaluation_context_calibration",
            "gpu_execution",
            "tool_calling",
            "agent_in_twin",
            "coevolution",
            "protected_holdout_access",
        ],
        "terminal_blocker": decision["terminal_blocker"],
        "protected_holdout_access": 0,
        "decision_binding": _payload_binding(decision_path, decision),
    }


def build(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    outputs = {key: _repo_path(str(value)) for key, value in policy["outputs"].items()}
    prior_review = _read_json(_repo_path(policy["parent_review"]["scientific_review"]))
    prior_decision = _read_json(
        _repo_path(policy["parent_review"]["go_no_go_decision"])
    )
    if prior_review.get("status") != "PROTOCOL_NOT_FORMALLY_EXECUTABLE":
        raise ValueError("Parent protocol review is not the expected No-Go state")
    if prior_decision.get("execution_authorized") is not False:
        raise ValueError("Parent protocol unexpectedly authorizes execution")
    if prior_review.get("no_result_generation_declaration") is not True:
        raise ValueError("Parent review does not preserve the no-result boundary")
    canonical = _read_json(CANONICAL_TWIN_PATH)
    matrix_rows = _read_matrix()
    independence, _, _, _ = _build_independence_audit(policy, canonical, matrix_rows)
    authority = _build_authority(policy, independence, outputs["independence_audit"])
    decision = _build_decision(
        policy,
        authority,
        outputs["method_authority"],
        independence,
        outputs["independence_audit"],
    )
    authorization = _build_authorization(
        policy, decision, outputs["executability_decision"]
    )
    for payload, schema_path, label in (
        (independence, INDEPENDENCE_SCHEMA_PATH, "independence"),
        (authority, AUTHORITY_SCHEMA_PATH, "authority"),
        (decision, DECISION_SCHEMA_PATH, "decision"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization"),
    ):
        errors = _schema_errors(payload, schema_path)
        if errors:
            raise ValueError(f"{label} schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(outputs["independence_audit"], independence)
        write_json(outputs["method_authority"], authority)
        write_json(outputs["executability_decision"], decision)
        write_json(outputs["conditional_authorization"], authorization)
    return {
        "independence_audit": independence,
        "method_authority": authority,
        "executability_decision": decision,
        "conditional_authorization": authorization,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = build(write_output=not args.no_write)
    print(
        json.dumps(
            {
                "method_authority_status": result["method_authority"]["status"],
                "protocol_status": result["executability_decision"]["status"],
                "terminal_blocker": result["executability_decision"][
                    "terminal_blocker"
                ],
                "execution_authorized": result["conditional_authorization"][
                    "execution_authorized"
                ],
                "six_context_outcome_access": result["executability_decision"][
                    "execution_authority"
                ]["six_context_outcome_access"],
                "protected_holdout_access": result["conditional_authorization"][
                    "protected_holdout_access"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
