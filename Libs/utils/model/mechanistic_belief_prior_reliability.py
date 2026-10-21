"""Frozen prior, reliability, and terminal semantics for the belief Twin."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import yaml

from Libs.utils.data.mechanistic_belief_construct_firewall import (
    ConstructFirewallError,
    canonical_json_bytes,
    canonical_value_sha256,
    sha256_bytes,
    sha256_file,
    validate_construct_authority,
    validate_cutoff_receipt,
)


CONTRACT_PATH = "Libs/configs/validation/mechanistic_belief_prior_reliability.yaml"
AUTHORITY_SCHEMA_PATH = (
    "Libs/configs/schema/mechanistic_belief_prior_reliability_authority.schema.json"
)
DECISION_SCHEMA_PATH = (
    "Libs/configs/schema/mechanistic_belief_terminal_decision_table.schema.json"
)
AUTHORITY_PATH = "Output/audits/mechanistic_belief_prior_reliability_authority.json"
DECISION_TABLE_PATH = "Output/audits/mechanistic_belief_terminal_decision_table.json"

UTILITY_PATH = "Libs/utils/model/mechanistic_belief_prior_reliability.py"
BUILDER_PATH = "Libs/scripts/builders/build_mechanistic_belief_prior_reliability.py"
VALIDATOR_PATH = (
    "Libs/scripts/validators/validate_mechanistic_belief_prior_reliability.py"
)

EXPECTED_CONTRACT_SHA256 = "26ca9004d107fd153d4a8fd26850a9a09931ab10424eed432b6497ca9420a452"
EXPECTED_AUTHORITY_SCHEMA_SHA256 = "69de25bf337727b8f287a8733bea5f786ee16ee984609e213b294d2413efd719"
EXPECTED_DECISION_SCHEMA_SHA256 = "f103b08d53df01ed50c3138b0fff62f685c72c517e96fd6a6b0b479e10b0abca"

CONTRACT_SCHEMA_VERSION = (
    "neuronsifter-mechanistic-belief-prior-reliability-contract-v1"
)
AUTHORITY_SCHEMA_VERSION = (
    "neuronsifter-mechanistic-belief-prior-reliability-authority-v1"
)
DECISION_SCHEMA_VERSION = (
    "neuronsifter-mechanistic-belief-terminal-decision-table-v1"
)
METHOD_UID = "mechanistic_belief_twin_v1"

ROUND24_INPUTS = (
    (
        "Libs/configs/validation/mechanistic_belief_construct_firewall.yaml",
        "f450ed2e1be9f09fdc1b8e7bc5066b2c68a5307c4af7b4778d5686f36cd3e374",
        "frozen_before_mechanism_source_content",
        "contract_uid",
        "mechanistic_belief_construct_firewall",
    ),
    (
        "Libs/configs/schema/mechanistic_belief_cutoff_receipt.schema.json",
        "0c6e30c56c72b53b1d13db622faf7e458d8fa4e211edeb56c370c9d2b8388555",
        "schema_valid",
        "$id",
        "https://neuronsifter.local/schema/mechanistic_belief_cutoff_receipt.schema.json",
    ),
    (
        "Libs/configs/schema/mechanistic_belief_construct_authority.schema.json",
        "93edc9324dca165820a6d40475f95ba65e332242f63b07f345ad8b0856be6c70",
        "schema_valid",
        "$id",
        "https://neuronsifter.local/schema/mechanistic_belief_construct_authority.schema.json",
    ),
    (
        "Output/audits/mechanistic_belief_cutoff_receipt.json",
        "a2c9cd351a8f8199a87864d4a7f97f5a3444e859a55efd717343654a5c1570cc",
        "MECHANISTIC_BELIEF_CUTOFF_RECEIPT_VALID",
        "receipt_sha256",
        "076c29ad32d88d6ce08ddcdc25545f90189754ae0ef1ed5acda242493f7d84d5",
    ),
    (
        "Output/audits/mechanistic_belief_construct_authority.json",
        "a1ea36299c2f2097a7606a45a1643edd731aa693643113b3ee03bea7e0b120e0",
        "MECHANISTIC_BELIEF_CONSTRUCT_FIREWALL_FROZEN",
        "authority_sha256",
        "372247967aeaa49cd61c10b80bf2097c9824b4a7f9245eda87353f2f35eabd7b",
    ),
)

PRIOR_ROLE_ROWS = (
    (
        "canonical_mechanism_prior",
        "canonical",
        True,
        "future_admitted_mechanism_basis_only",
        "report_as_canonical_without_target_task_selection",
    ),
    (
        "conservative_support_sensitivity",
        "sensitivity",
        False,
        "future_intersection_of_admitted_support_bounds_only",
        "report_with_all_populated_sensitivity_roles_without_selection_or_averaging",
    ),
    (
        "broad_parameter_sensitivity",
        "sensitivity",
        False,
        "future_union_of_admitted_support_bounds_within_physical_domains_only",
        "report_with_all_populated_sensitivity_roles_without_selection_or_averaging",
    ),
)

DOMAIN_FAMILY_ROWS = (
    ("signed_real", "normal"),
    ("positive_real", "lognormal"),
    ("nonnegative_real", "zero_inflated_gamma"),
    ("unit_interval", "beta"),
    ("bounded_interval", "scaled_beta"),
    ("simplex", "dirichlet"),
    ("finite_categorical", "categorical"),
)

LATENT_FACTOR_ROWS = (
    (
        "sample_treatment_action",
        "known_exogenous_conditioning",
        "declared_sample_level_treatment_or_intervention_action",
        "latent_response_or_clinical_recommendation",
    ),
    (
        "exposure_occupancy",
        "uncertain_latent_state",
        "target_site_exposure_occupancy_and_declared_pkpd_state",
        "clinical_response_probability",
    ),
    (
        "cns_microenvironment",
        "uncertain_latent_state",
        "explicit_compartment_species_grid_and_supported_mechanism_state",
        "scalar_disease_score",
    ),
    (
        "electrophysiology",
        "uncertain_latent_state",
        "supported_voltage_calcium_spike_and_circuit_state",
        "clinical_cognition_or_endpoint_prediction",
    ),
    (
        "structural_morphology",
        "uncertain_latent_state",
        "supported_tissue_cellular_structure_and_morphology_state",
        "patient_level_progression",
    ),
)

OBSERVABLE_ROWS = (
    (
        "exposure_occupancy_observable",
        "exposure_occupancy",
        "direct_measurement_or_declared_projection_of_exposure_or_occupancy",
    ),
    (
        "microenvironment_species_flux_observable",
        "cns_microenvironment",
        "direct_measurement_or_declared_projection_of_species_or_flux",
    ),
    (
        "electrophysiology_observable",
        "electrophysiology",
        "direct_measurement_or_declared_projection_of_voltage_calcium_spike_or_circuit_state",
    ),
    (
        "structural_morphology_observable",
        "structural_morphology",
        "direct_measurement_or_declared_projection_of_structure_or_morphology",
    ),
)

COMPARISON_FORM_ROWS = (
    (
        "static",
        "initial_prior_without_action_conditioned_temporal_propagation",
        "retained_without_invented_response_trajectory",
        "preserve_structural_support_state",
        True,
    ),
    (
        "point_estimate",
        "prespecified_distributional_summary_of_same_mechanistic_mapping",
        "suppressed_for_diagnostic_comparison",
        "preserve_structural_support_state",
        True,
    ),
    (
        "forced_decision",
        "same_pairwise_distribution_with_code_pinned_tie_rule",
        "retained_in_receipt",
        "suppress_decision_abstention_but_retain_full_form_abstention_receipt",
        True,
    ),
    (
        "full_belief_twin",
        "prior_predictive_distribution_with_support_uncertainty_pairwise_outputs_and_abstention",
        "retained",
        "structural_no_fit_contract",
        False,
    ),
)

UNCERTAINTY_ROWS = (
    (
        "mechanism_uncertainty",
        "uncertainty_over_admitted_mechanism_structure_or_activation_assumptions",
    ),
    (
        "parameter_uncertainty",
        "uncertainty_within_frozen_parameter_domain_and_distribution_family",
    ),
    (
        "support_uncertainty",
        "categorical_provenance_status_for_incomplete_or_contradictory_evidence_coverage",
    ),
    (
        "context_model_form_uncertainty",
        "mismatch_between_declared_context_and_modeled_mechanism_or_observation_assumptions_including_sensitivity_disagreement",
    ),
)

SUPPORT_STATE_ROWS = (
    (
        "UNSUPPORTED",
        "required_action_mechanism_parameter_observable_or_provenance_semantics_absent_or_contradictory",
        "prohibited",
        "unavailable",
        True,
        "REQUIRED_SEMANTICS_UNSUPPORTED",
    ),
    (
        "OOD",
        "semantics_representable_but_required_dimension_outside_future_frozen_support_domain",
        "diagnostic_only",
        "diagnostic_only",
        True,
        "OUTSIDE_FROZEN_SUPPORT_DOMAIN",
    ),
    (
        "PARTIAL_SUPPORT",
        "no_contradiction_or_ood_breach_but_required_support_dimension_incomplete",
        "distribution_with_explicit_missingness",
        "continuous_only",
        False,
        None,
    ),
    (
        "SUPPORTED",
        "all_required_dimensions_satisfy_future_frozen_support_contract",
        "distribution_reporting_permitted",
        "continuous_only",
        False,
        None,
    ),
)

TERMINAL_ROWS = (
    (
        "PROSPECTIVE_MECHANISTIC_BELIEF_TWIN_READY_FOR_EVALUATION",
        "metadata_only_six_context_preflight",
        (
            "construct_and_firewall_pass",
            "mechanism_basis_pass",
            "compiler_generalization_pass",
            "prior_predictive_identifiability_pass",
            "no_fit_reliability_pass",
            "non_evaluation_qualification_pass",
            "metadata_only_action_coverage_pass",
            "zero_outcome_and_protected_access",
        ),
        ("formal_six_context_evaluation", "agentic_extensions"),
        (
            "execute_six_context_evaluation",
            "tool_calling",
            "agent_in_twin",
            "memory_adaptation",
            "coevolution",
        ),
        (
            "automatic_formal_evaluation",
            "outcome_inspection",
            "protected_holdout_access",
        ),
        "method_package_ready_for_separately_authorized_falsification_only",
    ),
    (
        "MECHANISM_BASIS_INSUFFICIENT",
        "mechanism_basis_qualification",
        (
            "bounded_queue_exhausted",
            "admitted_evidence_coverage_incomplete_or_contradictory",
            "no_invented_mechanism_or_parameter_semantics",
        ),
        (
            "mechanism-basis-qualification",
            "mechanism-action-program-generalization",
            "mechanistic-belief-runtime",
            "mechanistic-belief-reliability",
            "mechanistic-belief-method-qualification",
            "mechanistic-belief-evaluation-readiness",
        ),
        (
            "queue_expansion",
            "compiler_implementation",
            "runtime_implementation",
            "reliability_implementation",
            "qualification",
            "metadata_preflight",
            "evaluation_packaging",
        ),
        (
            "queue_widening",
            "source_replacement_after_review",
            "numerical_prior_invention",
            "task_specific_fitting",
            "outcome_inspection",
            "authority_mutation",
        ),
        "scientific_basis_stop_not_model_performance_result",
    ),
    (
        "MECHANISM_COMPILER_NOT_GENERALIZABLE",
        "mechanism_action_program_compilation",
        (
            "required_modality_semantics_not_compilable",
            "context_specific_invention_or_missing_physical_fields_or_modality_collapse",
        ),
        (
            "mechanism-action-program-generalization",
            "mechanistic-belief-runtime",
            "mechanistic-belief-reliability",
            "mechanistic-belief-method-qualification",
            "mechanistic-belief-evaluation-readiness",
        ),
        (
            "runtime_implementation",
            "reliability_implementation",
            "qualification",
            "metadata_preflight",
            "evaluation_packaging",
        ),
        (
            "context_specific_shortcuts",
            "invented_action_fields",
            "modality_collapse",
            "task_specific_fitting",
            "outcome_inspection",
            "authority_mutation",
        ),
        "compiler_scope_stop_not_intervention_response_result",
    ),
    (
        "PRIOR_PREDICTIVE_TWIN_NOT_SCIENTIFICALLY_IDENTIFIABLE",
        "prior_predictive_runtime_identifiability",
        (
            "state_observable_prior_transition_or_action_dependence_not_auditable",
            "target_task_fitting_or_invention_would_be_required",
        ),
        (
            "mechanistic-belief-runtime",
            "mechanistic-belief-reliability",
            "mechanistic-belief-method-qualification",
            "mechanistic-belief-evaluation-readiness",
        ),
        (
            "reliability_implementation",
            "qualification",
            "metadata_preflight",
            "evaluation_packaging",
        ),
        (
            "neural_head_training",
            "empirical_bayes_target_fit",
            "fabricated_transition_or_observation_link",
            "outcome_inspection",
            "authority_mutation",
        ),
        "identifiability_stop_not_target_task_result",
    ),
    (
        "NO_FIT_UNCERTAINTY_NOT_DEFENSIBLE",
        "structural_reliability_qualification",
        (
            "support_ood_uncertainty_or_abstention_requires_fitting_or_fabrication",
            "continuous_only_fallback_not_sufficient_for_declared_method_claim",
        ),
        (
            "mechanistic-belief-reliability",
            "mechanistic-belief-method-qualification",
            "mechanistic-belief-evaluation-readiness",
        ),
        ("qualification", "metadata_preflight", "evaluation_packaging"),
        (
            "target_task_threshold_fit",
            "performance_cutoff_tuning",
            "uncertainty_to_confidence_relabeling",
            "outcome_inspection",
            "authority_mutation",
        ),
        "reliability_semantics_stop_not_predictive_accuracy_result",
    ),
    (
        "ACTION_DOMAIN_COVERAGE_INSUFFICIENT",
        "metadata_only_six_context_preflight",
        (
            "formal_method_identity_qualified",
            "metadata_only_action_coverage_rule_failed",
            "zero_outcome_and_protected_access",
        ),
        (
            "mechanistic-belief-evaluation-readiness",
            "formal_six_context_evaluation",
        ),
        ("positive_readiness", "execute_six_context_evaluation"),
        (
            "context_specific_rule_addition",
            "method_mutation",
            "threshold_relaxation",
            "outcome_inspection",
            "protected_holdout_access",
        ),
        "metadata_coverage_stop_not_intervention_response_result",
    ),
)

ACCESS_COUNTERS = {
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
    "source_content_access": 0,
}
DOWNSTREAM_AUTHORIZATION = {
    "source_queue_creation_authorized": False,
    "source_content_intake_authorized": False,
    "prior_distribution_population_authorized": False,
    "world_action_compiler_authorized": False,
    "runtime_implementation_authorized": False,
    "inference_authorized": False,
    "training_authorized": False,
    "gpu_authorized": False,
    "formal_evaluation_authorized": False,
    "six_context_evaluation_authorized": False,
}
ARTIFACTS_CREATED = {
    "source_queue": False,
    "source_content": False,
    "numerical_prior_distribution": False,
    "world_action_compiler": False,
    "belief_runtime": False,
    "scientific_inference": False,
    "training_run": False,
    "evaluation_result": False,
}
CLAIM_CLOSURES = {
    "learned_confidence_claim": False,
    "tri_claim": False,
    "clinical_reliability_claim": False,
    "efficacy_claim": False,
    "treatment_recommendation_claim": False,
    "performance_claim": False,
}
LIMITATIONS = (
    "semantics_only_no_mechanism_source_content_or_numerical_prior_population",
    "no_compiler_runtime_inference_training_gpu_or_evaluation",
    "mechanism_support_is_not_clinical_effect_or_treatment_recommendation",
)

# Filled with literal digests after the code-owned sections are finalized.
EXPECTED_SEMANTIC_HASHES = {
    "method_identity": "a031cbaf693d29e7b73c6f98a9d67311d06530d3d7b1b1613143d582e41dbe84",
    "prior_roles": "95edaa56a4abde1191fe5f4a956a41df8dabe567a2fe6ba3cb724f3df51a1bf0",
    "parameter_domain_distribution_families": "68117c826df48c151d6e288d0e996766965d4c29e223ced4c842ebe418321673",
    "latent_state_factors": "42ce825a64294f91b3dd867f166528eea9cd83a20b32630de376515143122bd1",
    "observable_families": "497551a9f450f40ef8fc5ca1e7bdcd36ea79fc6f562e17ddf18e2e8dea421c03",
    "observable_requirements": "d3b01cb66616b4e863181a2cf7a1a5818afabc3af9cbb61a7652d5a53637f119",
    "prior_predictive_semantics": "b328c36000bf64b88e4df69cf9b5674376ccf8bffff8f2f7d6d85a3be23edd53",
    "comparison_forms": "825b2900bdd81fa5b6891b8d02b8f36d68dcde8f5feae67b947a302fa9f7e4d3",
    "uncertainty_components": "9eed4c97173a1accbb50cf433e941a1462eb739a2f100a646b94da5cc44c4f0d",
    "support_contract": "a7d0a69b78d6dc3bb0e284ca5cd3f31c6e8176acee5c649a23b47899f6978a88",
    "abstention_contract": "1e51bfc4680cac40694acbcc5e13d392217db9a12856d26089fc3c6f99a8c013",
    "pairwise_distributions": "82b0a89c0d37227bb6cd18a3eea0b2299fe96aadd6f26d736233276a92466f9f",
    "claim_closures": "8c54f8983680a4ba7b04f6b9622680ec525ead825e1df9a271d7afe700735f43",
    "terminal_decisions": "65514a0e65f33f5d13c86ef911e3245f74c9cafe38b5ffe379ec606845838088",
}


class PriorReliabilityError(ValueError):
    """Raised when frozen prior or reliability semantics fail closed."""


def _round24_bindings() -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "file_sha256": digest,
            "required_status": status,
            "semantic_identity_field": field,
            "semantic_identity": identity,
        }
        for path, digest, status, field, identity in ROUND24_INPUTS
    ]


def _method_identity() -> dict[str, Any]:
    return {
        "method_uid": METHOD_UID,
        "method_family": "prospective_mechanistic_belief_twin",
        "task_specific_fitting": False,
        "prior_predictive": True,
        "candidate_search_allowed": False,
        "preferred_method_allowed": False,
    }


def _prior_roles() -> dict[str, Any]:
    return {
        "ordered_roles": [
            {
                "role_uid": uid,
                "role_class": role_class,
                "required": required,
                "population_source": population_source,
                "selection_allowed": False,
                "reporting_rule": reporting_rule,
            }
            for uid, role_class, required, population_source, reporting_rule in PRIOR_ROLE_ROWS
        ],
        "maximum_sensitivity_roles": 2,
        "numerical_priors_populated": False,
        "preferred_sensitivity_role": None,
    }


def _domain_families() -> list[dict[str, str]]:
    return [
        {"parameter_domain": domain, "distribution_family": family}
        for domain, family in DOMAIN_FAMILY_ROWS
    ]


def _latent_factors() -> list[dict[str, str]]:
    return [
        {
            "factor_uid": uid,
            "factor_class": factor_class,
            "scientific_meaning": meaning,
            "forbidden_interpretation": forbidden,
        }
        for uid, factor_class, meaning, forbidden in LATENT_FACTOR_ROWS
    ]


def _observable_families() -> list[dict[str, str]]:
    return [
        {
            "observable_family_uid": uid,
            "linked_latent_factor": factor,
            "scientific_meaning": meaning,
        }
        for uid, factor, meaning in OBSERVABLE_ROWS
    ]


def _observable_requirements() -> dict[str, list[str]]:
    return {
        "required_fields": [
            "units",
            "source_role",
            "missingness",
            "projection_identity",
            "linked_latent_factor",
        ],
        "prohibited_semantics": [
            "target_task_outcome",
            "clinical_endpoint_bridge",
            "author_conclusion",
            "intervention_ranking_truth",
            "six_context_truth",
        ],
    }


def _prior_predictive_semantics() -> dict[str, Any]:
    return {
        "conditional_object": "distribution_over_latent_state_and_observable_trajectories_given_future_valid_action_program",
        "target_task_posterior": False,
        "target_task_label_fit": False,
        "clinical_effect_probability": False,
    }


def _comparison_forms() -> dict[str, Any]:
    return {
        "ordered_forms": [
            {
                "form_uid": uid,
                "method_uid": METHOD_UID,
                "semantic_role": role,
                "distributional_spread": spread,
                "abstention_behavior": abstention,
                "diagnostic_only": diagnostic_only,
            }
            for uid, role, spread, abstention, diagnostic_only in COMPARISON_FORM_ROWS
        ],
        "shared_identity_required": True,
        "preferred_form": None,
        "winner_selection_allowed": False,
    }


def _uncertainty_components() -> list[dict[str, str]]:
    return [
        {"component_uid": uid, "scientific_meaning": meaning}
        for uid, meaning in UNCERTAINTY_ROWS
    ]


def _support_contract() -> dict[str, Any]:
    return {
        "precedence": [row[0] for row in SUPPORT_STATE_ROWS],
        "states": [
            {
                "support_state": state,
                "scientific_meaning": meaning,
                "execution_disposition": execution,
                "distribution_reporting": reporting,
                "mandatory_abstention": mandatory,
                "abstention_reason_code": reason,
                "binary_decision_authorized": False,
            }
            for state, meaning, execution, reporting, mandatory, reason in SUPPORT_STATE_ROWS
        ],
    }


def _abstention_contract() -> dict[str, Any]:
    return {
        "default_mode": "continuous_only",
        "binary_rule_frozen": False,
        "target_task_threshold_allowed": False,
        "learned_threshold_allowed": False,
        "validation_quantile_allowed": False,
        "performance_cutoff_allowed": False,
        "unsupported_reason_code": "REQUIRED_SEMANTICS_UNSUPPORTED",
        "ood_reason_code": "OUTSIDE_FROZEN_SUPPORT_DOMAIN",
        "forced_decision_tie_rule": "stable_world_action_uid_lexicographic_order",
        "forced_decision_diagnostic_only": True,
    }


def _pairwise_distributions() -> dict[str, Any]:
    return {
        "effect_direction_categories": [
            "mechanistic_goal_aligned",
            "indeterminate_or_tied",
            "mechanistic_goal_opposed",
        ],
        "objective_orientation": "future_prespecified_mechanistic_objective_only",
        "rank_distribution": "joint_prior_predictive_rank_positions_with_ties_and_support_preserved",
        "unsupported_pair_disposition": "unavailable",
        "ood_pair_disposition": "diagnostic_only_with_mandatory_abstention",
        "sensitivity_identity_preserved": True,
        "cross_sensitivity_averaging_allowed": False,
        "point_score_sort_allowed": False,
        "clinical_direction_interpretation_allowed": False,
        "forced_decision_preserves_original_distribution": True,
    }


def _terminal_decisions() -> list[dict[str, Any]]:
    return [
        {
            "terminal": terminal,
            "trigger_stage": stage,
            "required_evidence": list(evidence),
            "closed_milestone_classes": list(milestones),
            "closed_direction_classes": list(directions),
            "prohibited_recovery_actions": list(prohibited),
            "interpretation": interpretation,
            "terminal_issued_in_this_round": False,
        }
        for terminal, stage, evidence, milestones, directions, prohibited, interpretation in TERMINAL_ROWS
    ]


def _semantic_sections() -> dict[str, Any]:
    return {
        "method_identity": _method_identity(),
        "prior_roles": _prior_roles(),
        "parameter_domain_distribution_families": _domain_families(),
        "latent_state_factors": _latent_factors(),
        "observable_families": _observable_families(),
        "observable_requirements": _observable_requirements(),
        "prior_predictive_semantics": _prior_predictive_semantics(),
        "comparison_forms": _comparison_forms(),
        "uncertainty_components": _uncertainty_components(),
        "support_contract": _support_contract(),
        "abstention_contract": _abstention_contract(),
        "pairwise_distributions": _pairwise_distributions(),
        "claim_closures": deepcopy(CLAIM_CLOSURES),
        "terminal_decisions": _terminal_decisions(),
    }


def _expected_contract() -> dict[str, Any]:
    sections = _semantic_sections()
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_uid": "mechanistic_belief_prior_reliability",
        "status": "frozen_before_mechanism_basis_intake",
        "outputs": {
            "prior_reliability_authority": {
                "path": AUTHORITY_PATH,
                "schema_path": AUTHORITY_SCHEMA_PATH,
                "schema_version": AUTHORITY_SCHEMA_VERSION,
            },
            "terminal_decision_table": {
                "path": DECISION_TABLE_PATH,
                "schema_path": DECISION_SCHEMA_PATH,
                "schema_version": DECISION_SCHEMA_VERSION,
            },
        },
        "round24_inputs": _round24_bindings(),
        **sections,
        "access_counters": deepcopy(ACCESS_COUNTERS),
        "downstream_authorization": deepcopy(DOWNSTREAM_AUTHORIZATION),
        "artifacts_created": deepcopy(ARTIFACTS_CREATED),
        "current_terminal": None,
        "terminal_issued_in_this_round": False,
        "limitations": list(LIMITATIONS),
        "blockers": [],
    }


def _relative_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "*?[]\x00"):
        raise PriorReliabilityError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts or str(path) != value:
        raise PriorReliabilityError(f"{label} must be a normalized relative path")
    return value


def _resolve_file(root: Path, relative: str, *, must_exist: bool, label: str) -> Path:
    normalized = _relative_path(relative, label=label)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise PriorReliabilityError(f"{label} traverses a symlink")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise PriorReliabilityError(f"missing {label}") from exc
    if not resolved.is_relative_to(root):
        raise PriorReliabilityError(f"{label} escapes repository root")
    if must_exist and not resolved.is_file():
        raise PriorReliabilityError(f"{label} is not a regular file")
    return candidate


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PriorReliabilityError(f"unable to load {label}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PriorReliabilityError(f"{label} must be an object with string keys")
    return value, raw


def _load_yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PriorReliabilityError("unable to load frozen prior/reliability contract") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PriorReliabilityError("contract must be an object with string keys")
    return value, raw


def _schema(root: Path, path: str, expected_hash: str) -> tuple[dict[str, Any], str]:
    resolved = _resolve_file(root, path, must_exist=True, label="schema")
    observed = sha256_file(resolved)
    if observed != expected_hash:
        raise PriorReliabilityError(f"frozen schema identity drift: {path}")
    schema, _ = _load_json(resolved, label="schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise PriorReliabilityError(f"invalid Draft 2020-12 schema: {path}") from exc
    return schema, observed


def _schema_validate(payload: Mapping[str, Any], schema: Mapping[str, Any], *, label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: [str(part) for part in error.path],
    )
    if errors:
        raise PriorReliabilityError(f"{label} schema validation failed: {errors[0].message}")


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    semantic = dict(payload)
    semantic.pop(field, None)
    return canonical_value_sha256(semantic)


def _validate_no_numeric_prior_or_threshold(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = key.lower()
            if type(nested) in {int, float} and any(
                token in lowered
                for token in (
                    "prior_value",
                    "parameter_value",
                    "location",
                    "scale",
                    "concentration",
                    "mixture",
                    "truncation",
                    "lower_bound",
                    "upper_bound",
                    "correlation",
                    "covariance",
                    "threshold",
                    "pseudo_count",
                    "default_value",
                )
            ):
                dotted = ".".join((*path, key))
                raise PriorReliabilityError(
                    f"numerical prior or threshold value is forbidden: {dotted}"
                )
            _validate_no_numeric_prior_or_threshold(nested, path=(*path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_no_numeric_prior_or_threshold(
                nested, path=(*path, str(index))
            )


def _semantic_hashes() -> dict[str, str]:
    observed = {
        key: canonical_value_sha256(value)
        for key, value in _semantic_sections().items()
    }
    if observed != EXPECTED_SEMANTIC_HASHES:
        raise PriorReliabilityError("code-owned semantic pin drift")
    return observed


def validate_round24_inputs(repository_root: Path) -> list[dict[str, str]]:
    root = repository_root.resolve(strict=True)
    try:
        receipt = validate_cutoff_receipt(repository_root=root)
        authority = validate_construct_authority(repository_root=root)
    except ConstructFirewallError as exc:
        raise PriorReliabilityError(f"Round-24 authority validation failed: {exc}") from exc
    if receipt.get("status") != "MECHANISTIC_BELIEF_CUTOFF_RECEIPT_VALID":
        raise PriorReliabilityError("Round-24 cutoff receipt status drift")
    if authority.get("status") != "MECHANISTIC_BELIEF_CONSTRUCT_FIREWALL_FROZEN":
        raise PriorReliabilityError("Round-24 construct authority status drift")
    if authority.get("method_uid") != METHOD_UID:
        raise PriorReliabilityError("Round-24 method identity drift")
    if receipt.get("access_counters") != ACCESS_COUNTERS:
        raise PriorReliabilityError("Round-24 access counters are not zero")
    construct_access = authority.get("construct", {}).get("access_policy")
    if construct_access != {
        "gate_access_kind": "mechanism_source_content",
        "metadata_retrieval_authorizes_source_content": False,
        **ACCESS_COUNTERS,
    }:
        raise PriorReliabilityError("Round-24 construct access policy drift")
    downstream = authority.get("downstream_authorization")
    if not isinstance(downstream, dict) or any(downstream.values()):
        raise PriorReliabilityError("Round-24 downstream authority reopened")

    bindings = _round24_bindings()
    for binding in bindings:
        path = _resolve_file(
            root, binding["path"], must_exist=True, label="Round-24 input"
        )
        if sha256_file(path) != binding["file_sha256"]:
            raise PriorReliabilityError(
                f"Round-24 input identity drift: {binding['path']}"
            )
        if binding["required_status"] == "schema_valid":
            payload, _ = _load_json(path, label="Round-24 schema")
            try:
                Draft202012Validator.check_schema(payload)
            except SchemaError as exc:
                raise PriorReliabilityError("Round-24 schema is invalid") from exc
        elif path.suffix == ".json":
            payload, _ = _load_json(path, label="Round-24 authority")
            if payload.get("status") != binding["required_status"]:
                raise PriorReliabilityError("Round-24 status binding drift")
        else:
            payload, _ = _load_yaml(path)
            if payload.get("status") != binding["required_status"]:
                raise PriorReliabilityError("Round-24 status binding drift")
        if payload.get(binding["semantic_identity_field"]) != binding["semantic_identity"]:
            raise PriorReliabilityError("Round-24 semantic identity drift")
    return bindings


def validate_contract(
    repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], bytes]:
    root = repository_root.resolve(strict=True)
    path = Path(contract_path)
    if path.is_absolute():
        try:
            relative = path.resolve(strict=True).relative_to(root).as_posix()
        except ValueError as exc:
            raise PriorReliabilityError("contract escapes repository root") from exc
    else:
        relative = _relative_path(path.as_posix(), label="contract path")
    if relative != CONTRACT_PATH:
        raise PriorReliabilityError(f"contract must be {CONTRACT_PATH}")
    resolved = _resolve_file(root, relative, must_exist=True, label="contract")
    contract, raw = _load_yaml(resolved)
    if sha256_bytes(raw) != EXPECTED_CONTRACT_SHA256:
        raise PriorReliabilityError("frozen contract identity drift")
    expected = _expected_contract()
    if contract != expected:
        raise PriorReliabilityError("code-owned prior/reliability semantics drift")
    _validate_no_numeric_prior_or_threshold(contract)
    _semantic_hashes()
    return contract, raw


def _code_identity(root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "sha256": sha256_file(
                _resolve_file(root, path, must_exist=True, label="code identity")
            ),
        }
        for path in (UTILITY_PATH, BUILDER_PATH, VALIDATOR_PATH)
    ]


def _build_decision_table(
    *,
    contract_raw: bytes,
    decision_schema_hash: str,
    round24_inputs: list[dict[str, str]],
    semantic_hashes: Mapping[str, str],
) -> dict[str, Any]:
    table: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": "MECHANISTIC_BELIEF_TERMINAL_DECISION_TABLE_FROZEN",
        "decision_table_uid": "mechanistic_belief_terminal_decision_table",
        "method_uid": METHOD_UID,
        "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)},
        "schema": {
            "path": DECISION_SCHEMA_PATH,
            "sha256": decision_schema_hash,
        },
        "round24_inputs": deepcopy(round24_inputs),
        "prior_reliability_semantics_sha256": canonical_value_sha256(
            {
                key: value
                for key, value in semantic_hashes.items()
                if key != "terminal_decisions"
            }
        ),
        "terminal_semantics_sha256": semantic_hashes["terminal_decisions"],
        "decisions": _terminal_decisions(),
        "current_terminal": None,
        "terminal_issued_in_this_round": False,
        "access_counters": deepcopy(ACCESS_COUNTERS),
        "downstream_authorization": deepcopy(DOWNSTREAM_AUTHORIZATION),
        "blockers": [],
    }
    table["decision_table_sha256"] = _self_hash(table, "decision_table_sha256")
    return table


def _build_authority(
    *,
    root: Path,
    contract_raw: bytes,
    authority_schema_hash: str,
    decision_schema_hash: str,
    round24_inputs: list[dict[str, str]],
    semantic_hashes: Mapping[str, str],
) -> dict[str, Any]:
    sections = _semantic_sections()
    sections.pop("terminal_decisions")
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "status": "MECHANISTIC_BELIEF_PRIOR_RELIABILITY_SEMANTICS_FROZEN",
        "authority_uid": "mechanistic_belief_prior_reliability_authority",
        "method_uid": METHOD_UID,
        "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)},
        "schemas": {
            "prior_reliability_authority": {
                "path": AUTHORITY_SCHEMA_PATH,
                "sha256": authority_schema_hash,
            },
            "terminal_decision_table": {
                "path": DECISION_SCHEMA_PATH,
                "sha256": decision_schema_hash,
            },
        },
        "round24_inputs": deepcopy(round24_inputs),
        "code_identity": _code_identity(root),
        "semantic_hashes": dict(semantic_hashes),
        **sections,
        "terminal_decision_table_binding": {
            "path": DECISION_TABLE_PATH,
            "schema_path": DECISION_SCHEMA_PATH,
            "schema_sha256": decision_schema_hash,
            "terminal_semantics_sha256": semantic_hashes["terminal_decisions"],
            "row_count": 6,
            "terminal_issued_in_this_round": False,
        },
        "access_counters": deepcopy(ACCESS_COUNTERS),
        "downstream_authorization": deepcopy(DOWNSTREAM_AUTHORIZATION),
        "artifacts_created": deepcopy(ARTIFACTS_CREATED),
        "limitations": list(LIMITATIONS),
        "blockers": [],
    }
    authority["authority_sha256"] = _self_hash(authority, "authority_sha256")
    return authority


def _canonical_payload(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = _load_json(path, label=label)
    if raw != canonical_json_bytes(payload):
        raise PriorReliabilityError(f"{label} is not canonical JSON")
    return payload, raw


def validate_prior_reliability_outputs(
    *,
    repository_root: Path,
    contract_path: Path | str = CONTRACT_PATH,
    authority_path: Path | str = AUTHORITY_PATH,
    decision_table_path: Path | str = DECISION_TABLE_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = repository_root.resolve(strict=True)
    round24_inputs = validate_round24_inputs(root)
    _, contract_raw = validate_contract(root, contract_path)
    authority_schema, authority_schema_hash = _schema(
        root, AUTHORITY_SCHEMA_PATH, EXPECTED_AUTHORITY_SCHEMA_SHA256
    )
    decision_schema, decision_schema_hash = _schema(
        root, DECISION_SCHEMA_PATH, EXPECTED_DECISION_SCHEMA_SHA256
    )
    semantic_hashes = _semantic_hashes()

    authority_relative = (
        Path(authority_path).resolve(strict=True).relative_to(root).as_posix()
        if Path(authority_path).is_absolute()
        else _relative_path(Path(authority_path).as_posix(), label="authority path")
    )
    decision_relative = (
        Path(decision_table_path).resolve(strict=True).relative_to(root).as_posix()
        if Path(decision_table_path).is_absolute()
        else _relative_path(
            Path(decision_table_path).as_posix(), label="decision table path"
        )
    )
    if authority_relative != AUTHORITY_PATH or decision_relative != DECISION_TABLE_PATH:
        raise PriorReliabilityError("authority output path drift")
    authority, _ = _canonical_payload(
        _resolve_file(root, authority_relative, must_exist=True, label="authority"),
        label="prior/reliability authority",
    )
    decision_table, _ = _canonical_payload(
        _resolve_file(
            root, decision_relative, must_exist=True, label="decision table"
        ),
        label="terminal decision table",
    )
    _schema_validate(authority, authority_schema, label="prior/reliability authority")
    _schema_validate(decision_table, decision_schema, label="terminal decision table")
    _validate_no_numeric_prior_or_threshold(authority)
    _validate_no_numeric_prior_or_threshold(decision_table)
    if authority.get("authority_sha256") != _self_hash(
        authority, "authority_sha256"
    ):
        raise PriorReliabilityError("authority semantic identity drift")
    if decision_table.get("decision_table_sha256") != _self_hash(
        decision_table, "decision_table_sha256"
    ):
        raise PriorReliabilityError("decision table semantic identity drift")

    expected_authority = _build_authority(
        root=root,
        contract_raw=contract_raw,
        authority_schema_hash=authority_schema_hash,
        decision_schema_hash=decision_schema_hash,
        round24_inputs=round24_inputs,
        semantic_hashes=semantic_hashes,
    )
    expected_table = _build_decision_table(
        contract_raw=contract_raw,
        decision_schema_hash=decision_schema_hash,
        round24_inputs=round24_inputs,
        semantic_hashes=semantic_hashes,
    )
    if authority != expected_authority:
        raise PriorReliabilityError("prior/reliability authority drift")
    if decision_table != expected_table:
        raise PriorReliabilityError("terminal decision table drift")
    if authority["terminal_decision_table_binding"]["terminal_semantics_sha256"] != decision_table["terminal_semantics_sha256"]:
        raise PriorReliabilityError("cross-output terminal semantic binding drift")
    if authority["round24_inputs"] != decision_table["round24_inputs"]:
        raise PriorReliabilityError("cross-output Round-24 binding drift")
    if decision_table["current_terminal"] is not None or decision_table["terminal_issued_in_this_round"]:
        raise PriorReliabilityError("a scientific terminal was issued in this semantics-only round")
    return authority, decision_table


def _write_temp(directory: Path, payload: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=".mechanistic-belief-prior-reliability-",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)
    finally:
        handle.close()


def _install_pair(
    authority_path: Path,
    authority_raw: bytes,
    decision_path: Path,
    decision_raw: bytes,
    *,
    replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    if authority_path.exists() or decision_path.exists():
        raise PriorReliabilityError("refusing to overwrite non-identical authority output")
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    authority_temp = _write_temp(authority_path.parent, authority_raw)
    decision_temp = _write_temp(decision_path.parent, decision_raw)
    authority_installed = False
    try:
        replace(authority_temp, authority_path)
        authority_installed = True
        replace(decision_temp, decision_path)
    except Exception as exc:
        if authority_installed:
            authority_path.unlink(missing_ok=True)
        raise PriorReliabilityError("transactional authority installation failed") from exc
    finally:
        authority_temp.unlink(missing_ok=True)
        decision_temp.unlink(missing_ok=True)


def build_prior_reliability_authority(
    *,
    repository_root: Path,
    contract_path: Path | str = CONTRACT_PATH,
    replace: Callable[[Path, Path], None] = os.replace,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    round24_inputs = validate_round24_inputs(root)
    _, contract_raw = validate_contract(root, contract_path)
    authority_schema, authority_schema_hash = _schema(
        root, AUTHORITY_SCHEMA_PATH, EXPECTED_AUTHORITY_SCHEMA_SHA256
    )
    decision_schema, decision_schema_hash = _schema(
        root, DECISION_SCHEMA_PATH, EXPECTED_DECISION_SCHEMA_SHA256
    )
    semantic_hashes = _semantic_hashes()
    authority_path = root / AUTHORITY_PATH
    decision_path = root / DECISION_TABLE_PATH
    if authority_path.exists() or decision_path.exists():
        if not (authority_path.exists() and decision_path.exists()):
            raise PriorReliabilityError("partial existing authority output")
        authority, decision_table = validate_prior_reliability_outputs(
            repository_root=root,
            contract_path=contract_path,
            authority_path=AUTHORITY_PATH,
            decision_table_path=DECISION_TABLE_PATH,
        )
        return authority, decision_table, False

    decision_table = _build_decision_table(
        contract_raw=contract_raw,
        decision_schema_hash=decision_schema_hash,
        round24_inputs=round24_inputs,
        semantic_hashes=semantic_hashes,
    )
    authority = _build_authority(
        root=root,
        contract_raw=contract_raw,
        authority_schema_hash=authority_schema_hash,
        decision_schema_hash=decision_schema_hash,
        round24_inputs=round24_inputs,
        semantic_hashes=semantic_hashes,
    )
    _schema_validate(authority, authority_schema, label="prior/reliability authority")
    _schema_validate(decision_table, decision_schema, label="terminal decision table")
    _validate_no_numeric_prior_or_threshold(authority)
    _validate_no_numeric_prior_or_threshold(decision_table)
    _install_pair(
        authority_path,
        canonical_json_bytes(authority),
        decision_path,
        canonical_json_bytes(decision_table),
        replace=replace,
    )
    return authority, decision_table, True
