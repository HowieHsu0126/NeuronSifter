"""Build the fail-closed scientific review for the six-context prospective protocol.

This builder inspects only frozen metadata, configuration, and development
artifacts.  It never opens hidden outcomes and never runs model inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    write_json,
    write_tsv,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTRACT_PATH = (
    REPO_ROOT / "Libs/configs/validation/abstention_selective_risk_contract.yaml"
)
CONTRACT_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/abstention_selective_risk_contract.schema.json"
)
METHOD_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/frozen_zero_shot_method_manifest.schema.json"
)
TOOL_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/tool_information_headroom.schema.json"
)
CLAIM_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_claim_to_evidence_matrix.schema.json"
)
DECISION_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_protocol_go_no_go.schema.json"
)
REVIEW_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_protocol_scientific_review.schema.json"
)
AUTHORIZATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_zero_shot_execution_authorization.schema.json"
)

PROTOCOL_PATH = (
    REPO_ROOT / "Output/audits/prospective_uncertainty_aware_validation_protocol.json"
)
SCARCITY_PATH = (
    REPO_ROOT
    / "Output/audits/public_retrospective_ranking_evidence_structural_scarcity.json"
)
MATERIALIZATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_evidence_materialization.json"
)
REPLAY_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
LEAKAGE_PATH = REPO_ROOT / "Output/audits/retrospective_replay_leakage_audit.json"
CANONICAL_TWIN_PATH = REPO_ROOT / "Output/audits/round25_u0_canonical_static_twin.json"
TRAINING_MANIFEST_PATH = REPO_ROOT / "Output/audits/round23_training_manifest.json"
CHECKPOINT_VALIDATION_PATH = (
    REPO_ROOT / "Output/audits/round23_checkpoint_validation.json"
)
C1_RUN_RESULT_PATH = REPO_ROOT / "Output/runs/round23/c1/large_seed3407/run_result.json"
C2S_RUN_RESULT_PATH = (
    REPO_ROOT / "Output/runs/round23/c2s/large_seed3407/run_result.json"
)
TRI_CALIBRATION_PATH = (
    REPO_ROOT / "Output/audits/round25_tri_validation_calibration.json"
)
STATIC_POLICY_PATH = REPO_ROOT / "Output/audits/round25_static_policy_freeze.json"
TOOL_COVERAGE_PATH = (
    REPO_ROOT / "Output/audits/round26_decision_sensitive_tool_coverage.json"
)
INFORMATION_GATE_PATH = REPO_ROOT / "Output/audits/round26_information_action_gate.json"
TRANSMISSION_PATH = (
    REPO_ROOT / "Output/audits/round26_decision_transmission_analysis.json"
)
ENGINEERING_CLOSURE_PATH = (
    REPO_ROOT / "Output/audits/agent_in_twin_engineering_closure.json"
)
ENGINEERING_CHECKPOINT_PATH = (
    REPO_ROOT
    / "Output/results/agent_in_twin_engineering_closure/development_checkpoint.json"
)
RUNTIME_DEPENDENCY_PATH = (
    REPO_ROOT / "Output/audits/agent_twin_runtime_dependency_manifest.json"
)
ACTION_COMPILER_PATH = REPO_ROOT / "Libs/models/modules/round24b_typed_actions.py"
POSTERIOR_ADAPTER_PATH = REPO_ROOT / "Libs/models/modules/round24b_posterior_adapter.py"
WORLD_ACTION_PATH = REPO_ROOT / "Libs/models/modules/neuronsifter_world_action.py"
BUILDER_PATH = (
    REPO_ROOT / "Libs/scripts/builders/build_prospective_protocol_scientific_review.py"
)

EVALUATION_MATRIX_FIELDS = (
    "context_uid",
    "evaluation_role",
    "decision_classification",
    "publication_uids",
    "source_snapshot_uids",
    "evidence_grades",
    "action_count",
    "action_uids",
    "control_count",
    "control_uids",
    "ranking_truth_role",
    "training_role",
    "calibration_role",
    "model_selection_role",
    "threshold_fit_role",
    "primary_metrics",
    "execution_status",
    "outcome_access_during_review",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _payload_binding(path: Path, payload: Mapping[str, Any]) -> dict[str, object]:
    serialized = _payload_bytes(payload)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _matrix_text(rows: list[dict[str, object]]) -> str:
    lines = ["\t".join(EVALUATION_MATRIX_FIELDS)]
    for row in rows:
        lines.append("\t".join(str(row[field]) for field in EVALUATION_MATRIX_FIELDS))
    return "\n".join(lines) + "\n"


def _text_binding(path: Path, text: str) -> dict[str, object]:
    content = text.encode("utf-8")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _schema_errors(payload: Mapping[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("abstention_selective_risk_contract")
    if not isinstance(policy, dict):
        raise ValueError("Missing abstention_selective_risk_contract configuration")
    errors = _schema_errors(policy, CONTRACT_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Selective-risk contract schema errors: " + "; ".join(errors[:8])
        )
    return policy


def _parse_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _validate_parent_state(policy: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    scarcity = _read_json(SCARCITY_PATH)
    materialization = _read_json(MATERIALIZATION_PATH)
    replay = _read_json(REPLAY_PATH)
    leakage = _read_json(LEAKAGE_PATH)
    if protocol.get("protocol_uid") != policy["parent_protocol"]["protocol_uid"]:
        raise ValueError("Parent prospective protocol UID drifted")
    if protocol.get("status") != policy["parent_protocol"]["expected_status"]:
        raise ValueError("Parent prospective protocol status drifted")
    if scarcity.get("status") != (
        "PUBLIC_RETROSPECTIVE_THERAPEUTIC_RANKING_EVIDENCE_STRUCTURALLY_SPARSE"
    ):
        raise ValueError("Structural-scarcity decision is not frozen")
    population = protocol.get("evaluation_population", {})
    expected_positive = sorted(policy["context_roles"]["positive_ranking"])
    expected_ambiguous = sorted(policy["context_roles"]["ambiguous_abstention"])
    if sorted(population.get("positive_ranking_context_uids", [])) != expected_positive:
        raise ValueError("Positive ranking context identities drifted")
    if sorted(population.get("ambiguous_context_uids", [])) != expected_ambiguous:
        raise ValueError("Ambiguous context identities drifted")
    if (
        population.get("train_context_count") != 0
        or population.get("calibration_context_count") != 0
    ):
        raise ValueError("Evaluation contexts entered training or calibration")
    expected_state = {
        "status": "qualified_public_candidates_but_split_insufficient",
        "accepted_records": 36,
        "rejected_records": 7,
        "legal_candidate_sets": 6,
        "independent_public_contexts": 6,
        "independent_decision_sensitive_groups": 2,
        "required_groups": 3,
        "split_files_created": False,
        "leakage_status": "passed",
        "protected_holdout_access": 0,
    }
    observed_state = {
        "status": materialization.get("status"),
        "accepted_records": materialization.get("accepted_record_count"),
        "rejected_records": materialization.get("rejected_record_count"),
        "legal_candidate_sets": materialization.get("legal_candidate_set_count"),
        "independent_public_contexts": materialization.get("independent_context_count"),
        "independent_decision_sensitive_groups": replay.get(
            "independent_decision_sensitive_group_count"
        ),
        "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
        "split_files_created": replay.get("split_files_created"),
        "leakage_status": leakage.get("status"),
        "protected_holdout_access": replay.get("protected_holdout_access"),
    }
    if observed_state != expected_state:
        raise ValueError(f"Prospective review parent state drifted: {observed_state}")
    for key, value in protocol["execution_authority"].items():
        if key == "protected_holdout_access":
            continue
        if key == "protocol_frozen":
            if value is not True:
                raise ValueError("Parent protocol is not frozen")
            continue
        if value is not False:
            raise ValueError(
                f"Parent protocol unexpectedly authorizes execution: {key}"
            )
    if protocol["execution_authority"].get("protected_holdout_access") != 0:
        raise ValueError("Protected holdout access is nonzero")
    return {
        "protocol": protocol,
        "scarcity": scarcity,
        "materialization": materialization,
        "replay": replay,
        "leakage": leakage,
        "current_state": observed_state,
    }


def _build_evaluation_matrix(
    policy: Mapping[str, Any], state: Mapping[str, Any]
) -> list[dict[str, object]]:
    candidates = {
        record["candidate_set_uid"]: record
        for record in state["replay"]["candidate_sets"]
    }
    role_order = (
        ("positive_ranking", policy["context_roles"]["positive_ranking"]),
        ("ambiguous_abstention", policy["context_roles"]["ambiguous_abstention"]),
    )
    rows: list[dict[str, object]] = []
    for evaluation_role, context_uids in role_order:
        for context_uid in context_uids:
            record = candidates.get(context_uid)
            if record is None:
                raise ValueError(f"Missing frozen evaluation context: {context_uid}")
            expected_class = (
                "decision_sensitive"
                if evaluation_role == "positive_ranking"
                else "ambiguous"
            )
            if record.get("decision_classification") != expected_class:
                raise ValueError(f"Evaluation role mismatch: {context_uid}")
            metric_key = (
                "positive_ranking"
                if evaluation_role == "positive_ranking"
                else "ambiguous_abstention"
            )
            rows.append(
                {
                    "context_uid": context_uid,
                    "evaluation_role": evaluation_role,
                    "decision_classification": expected_class,
                    "publication_uids": "|".join(record["publication_uids"]),
                    "source_snapshot_uids": "|".join(record["source_snapshot_uids"]),
                    "evidence_grades": "|".join(record["evidence_grades"]),
                    "action_count": len(record["action_uids"]),
                    "action_uids": "|".join(record["action_uids"]),
                    "control_count": len(record["control_uids"]),
                    "control_uids": "|".join(record["control_uids"]),
                    "ranking_truth_role": (
                        "sealed_positive_ranking_reference"
                        if evaluation_role == "positive_ranking"
                        else "unresolved_no_ranking_truth"
                    ),
                    "training_role": "prohibited",
                    "calibration_role": "prohibited",
                    "model_selection_role": "prohibited",
                    "threshold_fit_role": "prohibited",
                    "primary_metrics": "|".join(policy["metrics"][metric_key]),
                    "execution_status": "NOT_RUN",
                    "outcome_access_during_review": 0,
                }
            )
    return rows


def _identifier_overlap(
    identifiers: Iterable[str], source_paths: Iterable[Path]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    identifier_list = sorted(set(identifiers))
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        matches = [identifier for identifier in identifier_list if identifier in text]
        result[path.relative_to(REPO_ROOT).as_posix()] = matches
    return result


def _checkpoint_component(
    *,
    component_uid: str,
    frozen_record: Mapping[str, Any],
    checkpoint_validation: Mapping[str, Any],
    run_result: Mapping[str, Any],
) -> dict[str, Any]:
    path = _repo_path(str(frozen_record["path"]))
    expected_sha = str(frozen_record["sha256"])
    file_exists = path.is_file() and not path.is_symlink()
    current_sha = sha256_file(path) if file_exists else None
    validation_row = next(
        (
            row
            for row in checkpoint_validation.get("rows", [])
            if row.get("path") == frozen_record["path"]
        ),
        {},
    )
    return {
        "component_uid": component_uid,
        "path": frozen_record["path"],
        "expected_sha256": expected_sha,
        "observed_sha256": current_sha,
        "hash_matches": current_sha == expected_sha,
        "file_exists": file_exists,
        "prior_payload_loadable": validation_row.get("payload_loadable") is True,
        "prior_finite_model_state": validation_row.get("finite_model_state") is True,
        "run_uid": run_result.get("run_uid"),
        "model_scale": run_result.get("model_scale"),
        "seed": run_result.get("seed"),
        "evidence_role": run_result.get("evidence_role"),
        "publication_claim_eligible": run_result.get("publication_claim_eligible"),
        "protected_test_access": run_result.get("protected_test_access"),
    }


def _build_method_manifest(
    policy: Mapping[str, Any],
    state: Mapping[str, Any],
    matrix_rows: list[dict[str, object]],
) -> dict[str, Any]:
    canonical = _read_json(CANONICAL_TWIN_PATH)
    training = _read_json(TRAINING_MANIFEST_PATH)
    checkpoint_validation = _read_json(CHECKPOINT_VALIDATION_PATH)
    c1_run = _read_json(C1_RUN_RESULT_PATH)
    c2s_run = _read_json(C2S_RUN_RESULT_PATH)
    tri = _read_json(TRI_CALIBRATION_PATH)
    static_policy = _read_json(STATIC_POLICY_PATH)
    engineering = _read_json(ENGINEERING_CLOSURE_PATH)
    engineering_checkpoint = _read_json(ENGINEERING_CHECKPOINT_PATH)
    runtime_dependency = _read_json(RUNTIME_DEPENDENCY_PATH)
    components = [
        _checkpoint_component(
            component_uid="scientific_belief_encoder",
            frozen_record=canonical["scientific_belief_encoder"],
            checkpoint_validation=checkpoint_validation,
            run_result=c1_run,
        ),
        _checkpoint_component(
            component_uid="world_action_capability",
            frozen_record=canonical["world_action_capability"],
            checkpoint_validation=checkpoint_validation,
            run_result=c2s_run,
        ),
    ]
    identifiers: list[str] = []
    for row in matrix_rows:
        identifiers.append(str(row["context_uid"]))
        identifiers.extend(str(row["publication_uids"]).split("|"))
        identifiers.extend(str(row["action_uids"]).split("|"))
        identifiers.extend(str(row["control_uids"]).split("|"))
    c1_split_path = _repo_path(
        canonical["development_data_splits"]["c1"]["role_binding_path"]
    )
    c2s_pool_path = _repo_path(canonical["development_data_splits"]["c2s"]["path"])
    overlap_by_source = _identifier_overlap(
        identifiers,
        (TRAINING_MANIFEST_PATH, c1_split_path, c2s_pool_path),
    )
    training_precedes_evaluation = _parse_datetime(
        training["created_at"]
    ) < _parse_datetime(state["replay"]["created_at"])
    expected_contract_sha = canonical["software_environment"]["contract_sha256"]
    contract_path = _repo_path(canonical["software_environment"]["contract_path"])
    observed_contract_sha = sha256_file(contract_path)
    c2s_pool = _read_json(c2s_pool_path)
    development_routes = sorted(
        {
            str(record["action"]["route"])
            for record in c2s_pool.get("records", [])
            if isinstance(record.get("action"), dict) and record["action"].get("route")
        }
    )
    evaluation_action_namespaces = sorted(
        {
            action_uid.split(":", 2)[1]
            for row in matrix_rows
            for action_uid in str(row["action_uids"]).split("|")
            if action_uid.startswith("structured_action:") and ":" in action_uid
        }
    )
    implementation_paths = {
        "action_compiler": ACTION_COMPILER_PATH,
        "posterior_adapter": POSTERIOR_ADAPTER_PATH,
        "world_action_runtime": WORLD_ACTION_PATH,
    }
    implementation_identity = {
        key: {
            **_binding(path),
            "frozen_hash_bound_by_canonical_manifest": False,
        }
        for key, path in implementation_paths.items()
    }
    blocking_findings = [
        "complete_checkpoint_training_selection_lineage_not_bound",
        "frozen_config_snapshot_hash_mismatch",
        "frozen_implementation_snapshot_not_hash_bound",
        "six_context_input_action_compiler_not_validated",
        "six_context_inference_entrypoint_not_bound",
        "six_context_uncertainty_semantics_not_bound",
        "retrospective_abstention_rule_not_independently_bound",
        "c1_checkpoint_selection_smoke_only",
        "method_evidence_development_only_publication_ineligible",
    ]
    if not all(component["hash_matches"] for component in components):
        blocking_findings.append("checkpoint_hash_mismatch")
    if not all(component["prior_payload_loadable"] for component in components):
        blocking_findings.append("checkpoint_loadability_not_verified")
    if any(overlap_by_source.values()) or not training_precedes_evaluation:
        blocking_findings.append("checkpoint_independence_from_six_contexts_not_proven")
    blocking_findings = sorted(set(blocking_findings))
    return {
        "schema_version": "neuronsifter-frozen-zero-shot-method-manifest-v1",
        "status": "incomplete_not_executable"
        if blocking_findings
        else "complete_and_executable",
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "checkpoint_components": components,
        "evaluation_independence": {
            "training_manifest_created_at": training["created_at"],
            "retrospective_replay_created_at": state["replay"]["created_at"],
            "training_precedes_retrospective_evaluation": training_precedes_evaluation,
            "identifier_overlap_by_training_source": overlap_by_source,
            "identifier_overlap_count": sum(
                len(values) for values in overlap_by_source.values()
            ),
            "bounded_overlap_check_sources": sorted(overlap_by_source),
            "full_training_and_model_selection_lineage_bound": False,
            "formal_independence_proven": False,
            "evaluation_context_training_count": 0,
            "evaluation_context_calibration_count": 0,
            "evaluation_outcome_access_during_review": 0,
        },
        "configuration_identity": {
            "path": canonical["software_environment"]["contract_path"],
            "expected_sha256": expected_contract_sha,
            "observed_sha256": observed_contract_sha,
            "hash_matches": observed_contract_sha == expected_contract_sha,
            "exact_frozen_snapshot_bound": False,
        },
        "implementation_identity": implementation_identity,
        "input_compatibility": {
            "development_world_action_routes": development_routes,
            "evaluation_action_namespaces": evaluation_action_namespaces,
            "canonical_compiler_path": canonical["action_compiler"],
            "bound_six_context_adapter": None,
            "deterministic_no_fit_adapter_validated": False,
            "compatibility_proven": False,
            "reason": (
                "The canonical compiler materializes current-clamp actions, while the six contexts "
                "contain compound or protein-delivery actions and no frozen adapter binds those "
                "records to C1/C2S tensors or occupancy programs."
            ),
            "bound_inference_entrypoint": None,
            "inference_entrypoint_verified": False,
        },
        "uncertainty_identity": {
            "c1_output_keys": c1_run.get("checkpoint_roundtrip_output_keys", []),
            "c1_semantics": "diagonal_gaussian_latent_mean_and_log_variance",
            "c2s_available_metrics": sorted(
                key
                for key in c2s_run.get("metrics", {})
                if key
                in {
                    "aleatoric_variance_mean",
                    "epistemic_variance_mean",
                    "calibration_interval_coverage",
                    "calibration_absolute_error",
                    "crps_normalized",
                    "nll_normalized",
                }
            ),
            "six_context_uncertainty_mapping_bound": False,
            "six_context_support_ood_mapping_bound": False,
            "model_tri_score_available_for_information_actions": False,
        },
        "abstention_identity": {
            "development_calibration_status": tri.get("status"),
            "source_partition_role": tri.get("source_partition_role"),
            "target_interval_coverage": tri.get("target_interval_coverage"),
            "coverage_floor": tri.get("coverage_floor"),
            "maximum_calibration_error": tri.get("maximum_calibration_error"),
            "test_derived_threshold_change": tri.get(
                "coverage_floor_changed_from_test"
            ),
            "independent_of_six_context_outcomes": training_precedes_evaluation
            and not any(overlap_by_source.values()),
            "retrospective_domain_portability_proven": False,
            "prespecified_no_fit_rule_bound": False,
            "current_contract_hash_matches_frozen_policy": observed_contract_sha
            == static_policy["contract"]["sha256"],
            "evaluation_context_threshold_fit_allowed": False,
        },
        "memory_identity": {
            "engineering_checkpoint_path": engineering["checkpoint_path"],
            "engineering_checkpoint_expected_sha256": engineering["checkpoint_sha256"],
            "engineering_checkpoint_observed_sha256": sha256_file(
                ENGINEERING_CHECKPOINT_PATH
            ),
            "information_tool_ids": engineering_checkpoint.get(
                "runtime_capabilities", {}
            ).get("information_tool_ids", []),
            "initial_memory_state_digest_bound": False,
            "formal_execution_allowed": runtime_dependency.get(
                "formal_execution_allowed"
            ),
            "formal_gate_blockers": runtime_dependency.get("blockers", []),
        },
        "blocking_findings": blocking_findings,
        "protected_holdout_access": 0,
    }


def _build_claim_matrix(
    policy: Mapping[str, Any], matrix_rows: list[dict[str, object]]
) -> dict[str, Any]:
    positive = [
        str(row["context_uid"])
        for row in matrix_rows
        if row["evaluation_role"] == "positive_ranking"
    ]
    ambiguous = [
        str(row["context_uid"])
        for row in matrix_rows
        if row["evaluation_role"] == "ambiguous_abstention"
    ]
    all_contexts = positive + ambiguous
    return {
        "schema_version": "neuronsifter-prospective-claim-to-evidence-matrix-v1",
        "status": "claim_scope_resolved",
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "claims": [
            {
                "claim": "zero_shot_intervention_ranking",
                "testable": "yes",
                "formally_executable_now": False,
                "endpoint": "context_level_top_intervention_and_pairwise_ordering",
                "evidence_units": positive,
                "limitations": [
                    "two_positive_contexts_only",
                    "no_broad_generalization",
                ],
                "promotion_ceiling": "two_context_retrospective_directional_consistency",
            },
            {
                "claim": "uncertainty_and_abstention",
                "testable": "yes",
                "formally_executable_now": False,
                "endpoint": "avoid_false_confident_ranking_on_unresolved_contexts",
                "evidence_units": ambiguous,
                "limitations": [
                    "ambiguous_is_unresolved_not_negative",
                    "threshold_must_be_independent",
                ],
                "promotion_ceiling": "six_context_falsification_suite",
            },
            {
                "claim": "selective_decision_making",
                "testable": "yes",
                "formally_executable_now": False,
                "endpoint": "coverage_risk_and_context_level_decision_correctness",
                "evidence_units": all_contexts,
                "limitations": ["n_equals_six", "descriptive_or_exact_inference_only"],
                "promotion_ceiling": "small_public_context_selective_decision_evidence",
            },
            {
                "claim": "twin_grounded_tool_calling",
                "testable": "conditional",
                "formally_executable_now": False,
                "endpoint": "legal_information_action_changes_uncertainty_or_decision",
                "evidence_units": ambiguous,
                "limitations": [
                    "decision_changing_headroom_not_supported",
                    "tool_schemas_and_costs_unbound",
                ],
                "promotion_ceiling": "closed_until_independent_headroom_is_demonstrated",
            },
            {
                "claim": "memory_utility",
                "testable": "conditional",
                "formally_executable_now": False,
                "endpoint": "receipt_and_provenance_retention_without_truth_leakage",
                "evidence_units": all_contexts,
                "limitations": [
                    "no_independent_longitudinal_units",
                    "initial_memory_digest_unbound",
                ],
                "promotion_ceiling": policy["memory_policy"]["formal_utility_claim"],
            },
            {
                "claim": "formal_coevolution",
                "testable": "no",
                "formally_executable_now": False,
                "endpoint": "not_available_under_evaluation_only_protocol",
                "evidence_units": [],
                "limitations": [
                    "training_contexts_zero",
                    "calibration_contexts_zero",
                    "no_post_update_evaluation_units",
                ],
                "promotion_ceiling": "target_claim_retained_formal_empirical_status_not_testable",
            },
            {
                "claim": "broad_therapeutic_superiority",
                "testable": "no",
                "formally_executable_now": False,
                "endpoint": "not_supported",
                "evidence_units": [],
                "limitations": [
                    "two_resolvable_contexts",
                    "four_unresolved_contexts",
                    "heterogeneous_assays",
                ],
                "promotion_ceiling": "prohibited",
            },
        ],
        "global_limitations": [
            "N=6 independent public contexts",
            "no training, calibration, threshold fitting, or model selection on the six contexts",
            "ambiguous contexts are not ranking truth or negative labels",
            "the unresolved retrospective ranking Gate remains 2/3",
        ],
        "protected_holdout_access": 0,
    }


def _build_tool_headroom(
    policy: Mapping[str, Any], matrix_rows: list[dict[str, object]]
) -> dict[str, Any]:
    coverage = _read_json(TOOL_COVERAGE_PATH)
    gate = _read_json(INFORMATION_GATE_PATH)
    transmission = _read_json(TRANSMISSION_PATH)
    context_bound_action_count = 0
    headroom_supported = (
        coverage.get("decision_changing_units", 0) > 0
        and not coverage.get("insufficient_decision_sensitive_information", True)
        and gate.get("terminal_decision_utility_improved") is True
        and transmission.get("tri_score_available") is True
        and context_bound_action_count > 0
    )
    ambiguous_contexts = []
    for row in matrix_rows:
        if row["evaluation_role"] != "ambiguous_abstention":
            continue
        ambiguous_contexts.append(
            {
                "context_uid": row["context_uid"],
                "candidate_information_action_classes": policy["tool_policy"][
                    "allowed_information_action_classes"
                ],
                "bound_executable_action_count": 0,
                "query_schema_bound": False,
                "cost_bound": False,
                "evidence_grade_bound": False,
                "allowed_output_schema_bound": False,
                "posterior_change": "unverified_for_context",
                "uncertainty_change": "unverified_for_context",
                "abstention_change": "unverified_for_context",
                "ranking_change": "unverified_for_context",
                "direct_hidden_truth_reveal_allowed": False,
                "formal_tool_execution_allowed": False,
            }
        )
    return {
        "schema_version": "neuronsifter-tool-information-headroom-v1",
        "status": (
            "TOOL_INFORMATION_HEADROOM_SUPPORTED"
            if headroom_supported
            else "TOOL_INFORMATION_HEADROOM_NOT_SUPPORTED"
        ),
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "scope_boundary": (
            "This finding applies to the reviewed development current-clamp information-action "
            "family and the absence of bound per-context tools in the six-context protocol; it is "
            "not a claim that no future legal information action can have decision value."
        ),
        "development_proxy_evidence": {
            "registered_tool_outcome_count": coverage.get(
                "registered_tool_outcome_count"
            ),
            "registered_information_action_pair_count": coverage.get(
                "registered_information_action_pair_count"
            ),
            "prospective_information_action_pair_count": coverage.get(
                "prospective_information_action_pair_count"
            ),
            "decision_changing_units": coverage.get("decision_changing_units"),
            "decision_neutral_units": coverage.get("decision_neutral_units"),
            "informative_about_state_units": coverage.get(
                "informative_about_state_units"
            ),
            "informative_about_uncertainty_units": coverage.get(
                "informative_about_uncertainty_units"
            ),
            "top1_switches_before_routing": transmission.get(
                "top1_switches_before_routing"
            ),
            "top1_switches_after_routing": transmission.get(
                "top1_switches_after_routing"
            ),
            "tri_score_available": transmission.get("tri_score_available"),
            "terminal_decision_utility_improved": gate.get(
                "terminal_decision_utility_improved"
            ),
            "context_bound_legal_action_count": context_bound_action_count,
            "evidence_role": coverage.get("evidence_role"),
            "publication_claim_eligible": coverage.get("publication_claim_eligible"),
        },
        "ambiguous_contexts": ambiguous_contexts,
        "hidden_outcome_denylist": policy["tool_policy"]["hidden_outcome_denylist"],
        "memory_boundary": {
            "allowed_content": policy["memory_policy"]["allowed_content"],
            "prohibited_content": policy["memory_policy"]["prohibited_content"],
            "formal_utility_claim": policy["memory_policy"]["formal_utility_claim"],
            "evaluation_truth_sharing_allowed": False,
        },
        "formal_agent_in_twin_headline_authorized": False,
        "protected_holdout_access": 0,
        "evidence_bindings": {
            "decision_sensitive_tool_coverage": _binding(TOOL_COVERAGE_PATH),
            "information_action_gate": _binding(INFORMATION_GATE_PATH),
            "decision_transmission_analysis": _binding(TRANSMISSION_PATH),
        },
    }


def _build_go_no_go(
    policy: Mapping[str, Any],
    method: Mapping[str, Any],
    tool: Mapping[str, Any],
    matrix_rows: list[dict[str, object]],
    method_path: Path,
    tool_path: Path,
    matrix_path: Path,
    matrix_text: str,
) -> dict[str, Any]:
    positive_count = sum(
        row["evaluation_role"] == "positive_ranking" for row in matrix_rows
    )
    ambiguous_count = sum(
        row["evaluation_role"] == "ambiguous_abstention" for row in matrix_rows
    )
    configured_metric_uids = {
        metric_uid
        for metric_group in policy["metrics"].values()
        for metric_uid in metric_group
    }
    metric_definitions_complete = configured_metric_uids <= set(
        policy["metric_definitions"]
    )
    baseline_identities_frozen = len(
        {record["baseline_uid"] for record in policy["baselines"]}
    ) == len(policy["baselines"])
    baseline_implementations_bound = all(
        record["execution_ready"] and record["implementation_binding"]
        for record in policy["baselines"]
    )
    go_conditions = {
        "checkpoint_bytes_verified": all(
            component["hash_matches"] and component["prior_payload_loadable"]
            for component in method["checkpoint_components"]
        ),
        "checkpoint_lineage_predates_and_has_no_exact_id_overlap": (
            method["evaluation_independence"]["identifier_overlap_count"] == 0
            and method["evaluation_independence"][
                "training_precedes_retrospective_evaluation"
            ]
        ),
        "complete_checkpoint_independence_proven": method["evaluation_independence"][
            "formal_independence_proven"
        ],
        "zero_shot_method_identity_complete": method["status"]
        == "complete_and_executable",
        "context_roles_frozen": positive_count == 2 and ambiguous_count == 4,
        "metric_definitions_complete": metric_definitions_complete,
        "baseline_identities_frozen": baseline_identities_frozen,
        "baseline_implementations_bound": baseline_implementations_bound,
        "six_context_evaluation_contract_complete": metric_definitions_complete
        and baseline_identities_frozen
        and baseline_implementations_bound,
        "thresholds_not_fit_on_evaluation": (
            policy["selective_risk"]["threshold_policy"][
                "evaluation_context_fit_allowed"
            ]
            is False
        ),
        "abstention_rule_formally_bound": method["abstention_identity"][
            "prespecified_no_fit_rule_bound"
        ]
        or method["abstention_identity"]["retrospective_domain_portability_proven"],
        "hidden_outcome_access_zero": True,
        "tool_information_headroom_resolved": tool["status"]
        in {
            "TOOL_INFORMATION_HEADROOM_SUPPORTED",
            "TOOL_INFORMATION_HEADROOM_NOT_SUPPORTED",
        },
        "tool_information_headroom_supported": tool["status"]
        == "TOOL_INFORMATION_HEADROOM_SUPPORTED",
        "evaluation_context_training_count_zero": method["evaluation_independence"][
            "evaluation_context_training_count"
        ]
        == 0,
        "evaluation_context_calibration_count_zero": method["evaluation_independence"][
            "evaluation_context_calibration_count"
        ]
        == 0,
        "protected_holdout_access_zero": method["protected_holdout_access"] == 0,
        "coevolution_formally_closed": True,
    }
    required_for_twin_only = (
        "checkpoint_bytes_verified",
        "complete_checkpoint_independence_proven",
        "zero_shot_method_identity_complete",
        "context_roles_frozen",
        "six_context_evaluation_contract_complete",
        "thresholds_not_fit_on_evaluation",
        "abstention_rule_formally_bound",
        "hidden_outcome_access_zero",
        "tool_information_headroom_resolved",
        "evaluation_context_training_count_zero",
        "evaluation_context_calibration_count_zero",
        "protected_holdout_access_zero",
        "coevolution_formally_closed",
    )
    ready = all(go_conditions[key] for key in required_for_twin_only)
    blocking_findings = list(method["blocking_findings"])
    if not baseline_implementations_bound:
        blocking_findings.append("baseline_implementations_not_bound")
    blocking_findings = sorted(set(blocking_findings))
    decision = (
        "PROTOCOL_READY_FOR_ZERO_SHOT_EXECUTION"
        if ready
        else "PROTOCOL_NOT_FORMALLY_EXECUTABLE"
    )
    return {
        "schema_version": "neuronsifter-prospective-protocol-go-no-go-v1",
        "decision": decision,
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "go_conditions": go_conditions,
        "blocking_findings": blocking_findings,
        "minimum_repair_path": [
            "recover the exact pre-evaluation configuration snapshot or create a new outcome-blind freeze with explicit code and config hashes",
            "implement and validate a deterministic no-fit adapter from the six context metadata and structured actions to the frozen Twin input and occupancy tensors using non-evaluation fixtures",
            "bind the uncertainty, support/OOD, and abstention computation to an independent development rule or a prespecified no-fit rule",
            "bind a CPU inference entrypoint and validate checkpoint-load plus adapter roundtrip without opening any of the six outcomes",
            "keep the Agent Tool-Calling headline closed unless a separate legal tool family demonstrates decision-changing information headroom",
        ],
        "prohibited_repairs": [
            "fit or select any threshold on the six evaluation contexts",
            "train or calibrate on any of the six evaluation contexts",
            "reinterpret ambiguous contexts as negative labels or ranking truth",
            "reopen homologous therapeutic evidence search",
            "retrain a model on the six contexts and call it zero-shot",
            "access protected holdout outcomes",
        ],
        "execution_authorized": False,
        "protected_holdout_access": 0,
        "evidence_bindings": {
            "method_manifest": _payload_binding(method_path, method),
            "tool_headroom_audit": _payload_binding(tool_path, tool),
            "evaluation_matrix": _text_binding(matrix_path, matrix_text),
            "selective_risk_contract": _binding(CONTRACT_PATH),
        },
    }


def _build_review(
    policy: Mapping[str, Any],
    state: Mapping[str, Any],
    method: Mapping[str, Any],
    tool: Mapping[str, Any],
    claim: Mapping[str, Any],
    decision: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    matrix_text: str,
) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-prospective-protocol-scientific-review-v1",
        "status": decision["decision"],
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "review_mode": "protocol_review_only",
        "primary_conclusion": (
            "The six-context evidence roles and metrics are admissible as a small falsification "
            "suite, and the candidate checkpoint bytes predate the evaluation corpus. Formal "
            "execution is not yet interpretable because the exact frozen method identity, "
            "six-context input compiler, inference entrypoint, and abstention semantics are not "
            "bound. Current Tool information headroom is negative, so Agent-in-Twin and "
            "Co-Evolution remain outside the executable scope."
        ),
        "acceptance_resolution": {
            "claim_scope_resolved": True,
            "zero_shot_method_identity_frozen": method["status"]
            == "complete_and_executable",
            "six_context_evaluation_specification_frozen": decision["go_conditions"][
                "metric_definitions_complete"
            ]
            and decision["go_conditions"]["baseline_identities_frozen"],
            "six_context_evaluation_contract_frozen": decision["go_conditions"][
                "six_context_evaluation_contract_complete"
            ],
            "tool_information_headroom_resolved": True,
            "tool_information_headroom_supported": tool["status"]
            == "TOOL_INFORMATION_HEADROOM_SUPPORTED",
        },
        "immutable_parent_state": state["current_state"],
        "formal_claim_boundary": {
            "retrospective_ranking_gate": "2/3_unresolved_immutable",
            "prospective_twin_abstention_claim": "scientifically_admissible_but_not_executable",
            "agent_in_twin_tool_calling_claim": "closed_no_decision_changing_headroom",
            "memory_claim": "secondary_integrity_diagnostic_only",
            "coevolution_target_claim": "retained_but_not_testable_under_this_protocol",
            "broad_therapeutic_superiority": "not_testable",
        },
        "blocking_findings": list(decision["blocking_findings"]),
        "execution_authority": {
            "execution_authorized": False,
            "world_action_authorized": False,
            "gpu_authorized": False,
            "tri_authorized": False,
            "agent_in_twin_authorized": False,
            "coevolution_authorized": False,
            "public_evidence_search_authorized": False,
            "protected_holdout_access": 0,
        },
        "artifact_bindings": {
            "parent_protocol": _binding(PROTOCOL_PATH),
            "structural_scarcity_decision": _binding(SCARCITY_PATH),
            "retrospective_materialization": _binding(MATERIALIZATION_PATH),
            "retrospective_replay": _binding(REPLAY_PATH),
            "retrospective_leakage": _binding(LEAKAGE_PATH),
            "selective_risk_contract": _binding(CONTRACT_PATH),
            "method_manifest": _payload_binding(
                output_paths["method_manifest"], method
            ),
            "evaluation_matrix": _text_binding(
                output_paths["evaluation_matrix"], matrix_text
            ),
            "tool_headroom_audit": _payload_binding(
                output_paths["tool_headroom_audit"], tool
            ),
            "claim_to_evidence_matrix": _payload_binding(
                output_paths["claim_to_evidence_matrix"], claim
            ),
            "go_no_go_decision": _payload_binding(
                output_paths["go_no_go_decision"], decision
            ),
            "conditional_authorization": _payload_binding(
                output_paths["conditional_authorization"], authorization
            ),
            "implementation": _binding(BUILDER_PATH),
        },
        "no_result_generation_declaration": True,
    }


def _build_authorization(
    policy: Mapping[str, Any], decision: Mapping[str, Any], decision_path: Path
) -> dict[str, Any]:
    ready = decision["decision"] == "PROTOCOL_READY_FOR_ZERO_SHOT_EXECUTION"
    return {
        "schema_version": "neuronsifter-prospective-zero-shot-execution-authorization-v1",
        "status": "issuable_pending_user_authorization" if ready else "not_issuable",
        "review_uid": policy["review_uid"],
        "reviewed_on": policy["reviewed_on"],
        "protocol_decision": decision["decision"],
        "execution_authorized": False,
        "authorized_scope": (
            ["frozen_twin_ranking_and_abstention_cpu_evaluation"] if ready else []
        ),
        "blocked_scope": [
            "gpu_execution",
            "model_training",
            "evaluation_context_calibration",
            "threshold_fitting_on_evaluation_contexts",
            "agent_in_twin_tool_calling",
            "memory_utility_headline",
            "formal_coevolution",
            "additional_public_evidence_search",
            "protected_holdout_access",
        ],
        "issuance_conditions": list(decision["minimum_repair_path"]),
        "blocking_findings": list(decision["blocking_findings"]),
        "protected_holdout_access": 0,
        "decision_binding": _payload_binding(decision_path, decision),
    }


def build(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    output_paths = {
        key: _repo_path(str(value)) for key, value in policy["outputs"].items()
    }
    state = _validate_parent_state(policy)
    matrix_rows = _build_evaluation_matrix(policy, state)
    matrix_text = _matrix_text(matrix_rows)
    method = _build_method_manifest(policy, state, matrix_rows)
    tool = _build_tool_headroom(policy, matrix_rows)
    claim = _build_claim_matrix(policy, matrix_rows)
    decision = _build_go_no_go(
        policy,
        method,
        tool,
        matrix_rows,
        output_paths["method_manifest"],
        output_paths["tool_headroom_audit"],
        output_paths["evaluation_matrix"],
        matrix_text,
    )
    authorization = _build_authorization(
        policy, decision, output_paths["go_no_go_decision"]
    )
    review = _build_review(
        policy,
        state,
        method,
        tool,
        claim,
        decision,
        authorization,
        output_paths,
        matrix_text,
    )
    schema_checks = (
        (method, METHOD_SCHEMA_PATH, "method"),
        (tool, TOOL_SCHEMA_PATH, "tool"),
        (claim, CLAIM_SCHEMA_PATH, "claim"),
        (decision, DECISION_SCHEMA_PATH, "decision"),
        (review, REVIEW_SCHEMA_PATH, "review"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization"),
    )
    for payload, schema_path, label in schema_checks:
        errors = _schema_errors(payload, schema_path)
        if errors:
            raise ValueError(f"{label} schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(output_paths["method_manifest"], method)
        write_tsv(
            output_paths["evaluation_matrix"],
            list(EVALUATION_MATRIX_FIELDS),
            matrix_rows,
        )
        write_json(output_paths["tool_headroom_audit"], tool)
        write_json(output_paths["claim_to_evidence_matrix"], claim)
        write_json(output_paths["go_no_go_decision"], decision)
        write_json(output_paths["scientific_review"], review)
        write_json(output_paths["conditional_authorization"], authorization)
    return {
        "scientific_review": review,
        "method_manifest": method,
        "evaluation_matrix_rows": matrix_rows,
        "evaluation_matrix_text": matrix_text,
        "tool_headroom_audit": tool,
        "claim_to_evidence_matrix": claim,
        "go_no_go_decision": decision,
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
                "decision": result["go_no_go_decision"]["decision"],
                "method_status": result["method_manifest"]["status"],
                "tool_headroom_status": result["tool_headroom_audit"]["status"],
                "evaluation_context_count": len(result["evaluation_matrix_rows"]),
                "execution_authorized": result["conditional_authorization"][
                    "execution_authorized"
                ],
                "protected_holdout_access": result["scientific_review"][
                    "execution_authority"
                ]["protected_holdout_access"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
