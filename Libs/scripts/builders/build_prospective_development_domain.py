"""Audit the outcome-blind development domain for a new prospective Twin."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
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
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTRACT_PATH = (
    REPO_ROOT / "Libs/configs/validation/prospective_development_domain.yaml"
)
CONTRACT_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_development_domain_config.schema.json"
)
AUTHORITY_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_development_data_authority.schema.json"
)
EXCLUSION_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/six_context_semantic_exclusion_audit.schema.json"
)
PREREGISTRATION_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_method_preregistration.schema.json"
)
DECISION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_twin_reconstruction_decision.schema.json"
)
AUTHORIZATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_twin_training_authorization.schema.json"
)
VALIDATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_domain_validation.schema.json"
)
BUILDER_PATH = (
    REPO_ROOT / "Libs/scripts/builders/build_prospective_development_domain.py"
)
VALIDATOR_PATH = (
    REPO_ROOT / "Libs/scripts/validators/validate_prospective_development_domain.py"
)
TEST_PATH = REPO_ROOT / "Libs/tests/test_prospective_development_domain.py"
VALIDATION_WRAPPER_PATH = (
    REPO_ROOT
    / "Libs/exp/evaluation/run_prospective_development_domain_validation.sbatch"
)

PUBLIC_SPLIT_FIELDS = (
    "canonical_uid",
    "content_hash",
    "family_uid",
    "source_id",
    "source_accession",
    "publication_accession_group",
    "mirror_group",
    "biological_unit_uid",
    "regimen_uid",
    "target_program_proxy",
    "source_holdout_role",
    "split_role",
)
C2_SPLIT_FIELDS = (
    "split_uid",
    "split_role",
    "pair_uid",
    "action_uid",
    "trajectory_uid",
    "group_key",
    "source_holdout_group",
    "unseen_action_holdout",
)
OUTCOME_BEARING_FIELDS = {
    "numeric_outcome",
    "arm_mean",
    "uncertainty",
    "p_value",
    "effect_direction",
    "winner",
    "ranking_outcome",
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
    policy = config.get("prospective_development_domain")
    if not isinstance(policy, dict):
        raise ValueError("Missing prospective_development_domain configuration")
    errors = _schema_errors(policy, CONTRACT_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Development-domain config schema errors: " + "; ".join(errors[:8])
        )
    return policy


def _read_tsv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"Unexpected metadata-only TSV header: {path}")
        if OUTCOME_BEARING_FIELDS & set(reader.fieldnames or ()):
            raise ValueError(
                f"Outcome-bearing field present in metadata-only TSV: {path}"
            )
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def _read_evaluation_matrix(path: Path) -> list[dict[str, str]]:
    rows = _read_tsv(path, EVALUATION_MATRIX_FIELDS)
    if len(rows) != 6:
        raise ValueError("Expected exactly six evaluation identity rows")
    if any(row["execution_status"] != "NOT_RUN" for row in rows):
        raise ValueError("Six-context evaluation was executed")
    if any(row["outcome_access_during_review"] != "0" for row in rows):
        raise ValueError("Six-context outcome access is nonzero")
    for row in rows:
        if any(
            row[field] != "prohibited"
            for field in (
                "training_role",
                "calibration_role",
                "model_selection_role",
                "threshold_fit_role",
            )
        ):
            raise ValueError(f"Evaluation-only role violation: {row['context_uid']}")
        expected = {
            "positive_ranking": (
                "decision_sensitive",
                "sealed_positive_ranking_reference",
            ),
            "ambiguous_abstention": (
                "ambiguous",
                "unresolved_no_ranking_truth",
            ),
        }.get(row["evaluation_role"])
        if (
            expected is None
            or (row["decision_classification"], row["ranking_truth_role"]) != expected
        ):
            raise ValueError(
                f"Evaluation role/classification mismatch: {row['context_uid']}"
            )
    return rows


def _split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "record_count": len(rows),
        "source_ids": sorted({row["source_id"] for row in rows}),
        "publication_accession_groups": sorted(
            {row["publication_accession_group"] for row in rows}
        ),
        "family_counts": dict(
            sorted(Counter(row["family_uid"] for row in rows).items())
        ),
        "compound_family_record_count": sum(
            row["family_uid"] == "compound_dose_time_perturbation_transcriptomics"
            for row in rows
        ),
    }


def _c2_split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    role_rows = Counter(row["split_role"] for row in rows)
    role_pairs: dict[str, set[str]] = {}
    for row in rows:
        role_pairs.setdefault(row["split_role"], set()).add(row["pair_uid"])
    return {
        "record_count": len(rows),
        "pair_count": len({row["pair_uid"] for row in rows}),
        "role_row_counts": dict(sorted(role_rows.items())),
        "role_pair_counts": {
            role: len(pairs) for role, pairs in sorted(role_pairs.items())
        },
        "source_holdout_group_count": len(
            {row["source_holdout_group"] for row in rows}
        ),
    }


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


def _load_inputs(policy: Mapping[str, Any]) -> dict[str, Any]:
    payloads = {
        key: _read_json(_repo_path(str(value)))
        for key, value in policy["inputs"].items()
        if key != "six_context_matrix"
    }
    matrix_path = _repo_path(str(policy["inputs"]["six_context_matrix"]))
    payloads["six_context_matrix"] = _read_evaluation_matrix(matrix_path)
    payloads["six_context_matrix_path"] = matrix_path
    return payloads


def _source_publication_split(
    split_audit: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    manifest = next(
        record
        for record in split_audit["manifests"]
        if record["split_uid"] == "source_publication_holdout"
    )
    summaries: dict[str, dict[str, Any]] = {}
    for role, binding in manifest["role_bindings"].items():
        path = _repo_path(str(binding["path"]))
        rows = _read_tsv(path, PUBLIC_SPLIT_FIELDS)
        summaries[role] = {
            **_split_summary(rows),
            "binding": _binding(path),
        }
    return summaries


def _build_semantic_exclusion(
    policy: Mapping[str, Any], inputs: Mapping[str, Any]
) -> dict[str, Any]:
    matrix_rows = inputs["six_context_matrix"]
    split_summaries = _source_publication_split(inputs["public_corpus_splits"])
    c2_manifest_path = _repo_path(
        str(inputs["c2_restricted_domain"]["manifests"]["strict_split_manifest"])
    )
    source_publication_manifest = next(
        record
        for record in inputs["public_corpus_splits"]["manifests"]
        if record["split_uid"] == "source_publication_holdout"
    )
    exact_sources = [
        _repo_path(str(source_publication_manifest["role_bindings"][role]["path"]))
        for role in ("train", "validation", "test")
    ]
    exact_sources.append(c2_manifest_path)
    overlaps = _exact_overlap(_evaluation_identifiers(matrix_rows), exact_sources)
    evaluation_namespaces = sorted(
        {
            action_uid.split(":", 2)[1]
            for row in matrix_rows
            for action_uid in row["action_uids"].split("|")
            if action_uid.startswith("structured_action:")
        }
    )
    train_sources = set(split_summaries["train"]["source_ids"])
    validation_sources = set(split_summaries["validation"]["source_ids"])
    train_publications = set(split_summaries["train"]["publication_accession_groups"])
    validation_publications = set(
        split_summaries["validation"]["publication_accession_groups"]
    )
    return {
        "schema_version": "neuronsifter-six-context-semantic-exclusion-audit-v1",
        "status": "failed_incomplete_semantic_independence",
        "audit_uid": policy["audit_uid"],
        "reviewed_on": policy["reviewed_on"],
        "evaluation_identity": {
            "context_count": len(matrix_rows),
            "positive_ranking_context_count": sum(
                row["evaluation_role"] == "positive_ranking" for row in matrix_rows
            ),
            "ambiguous_context_count": sum(
                row["evaluation_role"] == "ambiguous_abstention" for row in matrix_rows
            ),
            "action_namespaces": evaluation_namespaces,
            "numeric_outcome_fields_present": False,
            "execution_status": "NOT_RUN",
        },
        "exact_exclusion": {
            "sources_checked": sorted(overlaps),
            "matches_by_source": overlaps,
            "match_count": sum(len(values) for values in overlaps.values()),
            "complete_for_all_development_assets": False,
        },
        "semantic_exclusion": {
            "publication_independence": "partial_exact_only",
            "biological_context_independence": "not_resolved",
            "action_family_independence": "not_resolved_compound_overlap",
            "endpoint_objective_independence": "not_resolved_and_not_aligned",
            "source_lineage_independence": "failed_train_validation_reuse",
            "protein_biologic_development_domain_present": False,
            "current_clamp_to_target_modality_compatibility": False,
        },
        "role_reuse_risk": {
            "train_validation_source_overlap": sorted(
                train_sources & validation_sources
            ),
            "train_validation_source_overlap_count": len(
                train_sources & validation_sources
            ),
            "train_validation_publication_overlap": sorted(
                train_publications & validation_publications
            ),
            "train_validation_publication_overlap_count": len(
                train_publications & validation_publications
            ),
            "dedicated_uncertainty_calibration_source_count": 0,
            "non_eval_test_compound_unit_count": split_summaries["test"][
                "compound_family_record_count"
            ],
            "non_eval_test_protein_biologic_unit_count": 0,
        },
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "artifact_bindings": {
            "evaluation_matrix": _binding(inputs["six_context_matrix_path"]),
            "public_corpus_splits": _binding(
                _repo_path(str(policy["inputs"]["public_corpus_splits"]))
            ),
            "c2_strict_split": _binding(c2_manifest_path),
        },
    }


def _build_data_authority(
    policy: Mapping[str, Any],
    inputs: Mapping[str, Any],
    semantic: Mapping[str, Any],
    semantic_path: Path,
) -> dict[str, Any]:
    evidence = inputs["public_corpus_evidence"]
    split_summaries = _source_publication_split(inputs["public_corpus_splits"])
    action_catalog = inputs["public_action_catalog"]
    c2_authority = inputs["c2_trajectory_authority"]
    c2_independence = inputs["c2_independent_domain"]
    c2_domain = inputs["c2_restricted_domain"]
    c2_manifest_path = _repo_path(str(c2_domain["manifests"]["strict_split_manifest"]))
    c2_split = _c2_split_summary(_read_tsv(c2_manifest_path, C2_SPLIT_FIELDS))
    action_source_counter: Counter[str] = Counter()
    for record in action_catalog["qualified_catalogs"]:
        action_source_counter[str(record["source_id"])] += int(record["record_count"])
    action_source_counts = dict(sorted(action_source_counter.items()))
    train_summary = split_summaries["train"]
    validation_summary = split_summaries["validation"]
    test_summary = split_summaries["test"]
    train_validation_source_overlap = sorted(
        set(train_summary["source_ids"]) & set(validation_summary["source_ids"])
    )
    train_validation_publication_overlap = sorted(
        set(train_summary["publication_accession_groups"])
        & set(validation_summary["publication_accession_groups"])
    )
    qualified_blocks = inputs["public_observation_blocks"].get("qualified_blocks", [])
    protein_biologic_action_count = sum(
        int(record.get("observation_count", 0))
        for record in qualified_blocks
        if record.get("family_id") == "protein_biologic_delivery"
    )
    independent_action_domains = int(
        c2_independence["c2_action_complete_independent_domain_count"]
    )
    required_independent_action_domains = int(
        c2_independence["required_c2_action_complete_independent_domain_count"]
    )
    c2_action_contract_passed = inputs["c2_action_audit"].get("status") == "passed"
    real_linked_trajectories = int(
        c2_authority["authority_classes"]["real_observation_linked_trajectory_count"]
    )
    public_static_domain_qualified = bool(
        evidence["qualified_action_count"] > 0
        and protein_biologic_action_count > 0
        and not train_validation_source_overlap
        and not train_validation_publication_overlap
    )
    simulator_domain_qualified = bool(
        c2_action_contract_passed
        and independent_action_domains >= required_independent_action_domains
        and real_linked_trajectories > 0
    )
    current_clamp_domain_qualified = bool(
        c2_domain["pharmacological_intervention_supported"]
        and c2_split["source_holdout_group_count"] >= 2
        and "calibration" in c2_split["role_pair_counts"]
    )
    development_domains = [
        {
            "domain_uid": "public_transcriptomic_state_and_compound_perturbation",
            "data_kind": "static_transcriptomic_observations",
            "qualified_block_count": evidence["qualified_block_count"],
            "observation_count": evidence["grade_ab_observation_count"],
            "effective_sample_size": evidence["grade_ab_effective_sample_size"],
            "qualified_action_count": evidence["qualified_action_count"],
            "action_source_counts": action_source_counts,
            "cross_source_action_anchor_count": evidence[
                "qualified_perturbation_transfer_anchor_count"
            ],
            "existing_split_roles": split_summaries,
            "longitudinal_or_functional_dynamics": False,
            "formal_training_domain_qualified": public_static_domain_qualified,
        },
        {
            "domain_uid": "legacy_simulator_trajectory_pool",
            "data_kind": "synthetic_repository_simulator_trajectories",
            "trajectory_count": c2_authority["authority_classes"][
                "repository_simulator_trajectory_count"
            ],
            "paired_unit_count": c2_authority["trajectory_counts"]["paired_unit_count"],
            "regimen_count": c2_authority["regimen_count"],
            "raw_source_family_count": c2_authority["source_family_count"],
            "real_observation_linked_trajectory_count": c2_authority[
                "authority_classes"
            ]["real_observation_linked_trajectory_count"],
            "published_model_trajectory_count": c2_authority["authority_classes"][
                "published_model_trajectory_count"
            ],
            "action_complete_independent_domain_count": c2_independence[
                "c2_action_complete_independent_domain_count"
            ],
            "shared_generation_lineage": True,
            "formal_training_domain_qualified": simulator_domain_qualified,
        },
        {
            "domain_uid": "restricted_modeldb_current_clamp",
            "data_kind": "single_source_electrophysiology_simulation",
            "trajectory_count": c2_split["record_count"],
            "matched_pair_count": c2_split["pair_count"],
            "role_pair_counts": c2_split["role_pair_counts"],
            "source_holdout_group_count": c2_split["source_holdout_group_count"],
            "selected_domain": c2_domain["selected_domain"],
            "pharmacological_intervention_supported": c2_domain[
                "pharmacological_intervention_supported"
            ],
            "dedicated_calibration_role": "calibration" in c2_split["role_pair_counts"],
            "formal_training_domain_qualified": current_clamp_domain_qualified,
        },
    ]
    train_qualified = any(
        record["formal_training_domain_qualified"] for record in development_domains
    )
    selection_qualified = bool(
        not train_validation_source_overlap
        and not train_validation_publication_overlap
        and c2_split["source_holdout_group_count"] >= 2
    )
    calibration_qualified = bool(
        "calibration" in split_summaries
        or "calibration" in c2_split["role_pair_counts"]
    )
    qualification_qualified = bool(
        test_summary["compound_family_record_count"] > 0
        and protein_biologic_action_count > 0
        and c2_split["role_pair_counts"].get("test", 0) > 0
        and c2_split["source_holdout_group_count"] >= 2
    )
    role_feasibility = {
        "TRAIN": {
            "status": "qualified" if train_qualified else "partial_not_qualified",
            "criteria": {
                "any_formal_training_domain_qualified": train_qualified,
                "target_modalities_covered": protein_biologic_action_count > 0,
                "action_complete_independent_domains_sufficient": independent_action_domains
                >= required_independent_action_domains,
                "real_observation_linked_dynamics_present": real_linked_trajectories
                > 0,
            },
            "reason": "Large static compound and synthetic trajectory pools exist, but no unified action-complete CNS intervention dynamics domain is qualified.",
        },
        "MODEL_SELECTION_VALIDATION": {
            "status": "qualified" if selection_qualified else "not_independent",
            "criteria": {
                "train_validation_source_overlap_count": len(
                    train_validation_source_overlap
                ),
                "train_validation_publication_overlap_count": len(
                    train_validation_publication_overlap
                ),
                "independent_c2_source_holdout_group_count": c2_split[
                    "source_holdout_group_count"
                ],
            },
            "reason": "C1 train and validation reuse the same three source families and four publication groups; current-clamp validation contains one pair from the same source model.",
        },
        "UNCERTAINTY_CALIBRATION": {
            "status": "qualified" if calibration_qualified else "unavailable",
            "criteria": {
                "dedicated_c1_calibration_role": "calibration" in split_summaries,
                "dedicated_c2_calibration_role": "calibration"
                in c2_split["role_pair_counts"],
            },
            "reason": "No dedicated source-independent calibration corpus is frozen.",
        },
        "NON_EVAL_QUALIFICATION": {
            "status": (
                "qualified"
                if qualification_qualified
                else "insufficient_modality_coverage"
            ),
            "criteria": {
                "compound_test_units": test_summary["compound_family_record_count"],
                "protein_biologic_test_units": protein_biologic_action_count,
                "current_clamp_test_pairs": c2_split["role_pair_counts"].get("test", 0),
                "independent_c2_source_holdout_group_count": c2_split[
                    "source_holdout_group_count"
                ],
            },
            "reason": "The C1 test role has two compound units and no protein/biologic units; current-clamp test has one pair from a single source.",
        },
    }
    four_role_possible = all(
        record["status"] == "qualified" for record in role_feasibility.values()
    )
    missing: list[str] = []
    if protein_biologic_action_count == 0:
        missing.append("independent protein or biologic delivery development evidence")
    if independent_action_domains < required_independent_action_domains:
        missing.append("multiple source-independent pharmacological action domains")
    if not selection_qualified or not calibration_qualified:
        missing.append("dedicated model-selection and uncertainty-calibration sources")
    if real_linked_trajectories == 0:
        missing.append("real baseline-to-post-action longitudinal trajectories")
    if not inputs["c1_capability"].get("headline_training_allowed"):
        missing.append("functional endpoint-aligned non-evaluation outcomes")
    if semantic["status"] != "passed":
        missing.append(
            "complete publication, context, action-family, endpoint, and source exclusion ledger"
        )
    if not qualification_qualified:
        missing.append("action-complete non-evaluation qualification domain")
    return {
        "schema_version": "neuronsifter-prospective-development-data-authority-v1",
        "status": (
            "PROSPECTIVE_TRAINING_DOMAIN_QUALIFIED"
            if four_role_possible
            else policy["stop_policy"]["insufficient_domain_status"]
        ),
        "audit_uid": policy["audit_uid"],
        "reviewed_on": policy["reviewed_on"],
        "development_domains": development_domains,
        "role_feasibility": role_feasibility,
        "four_role_partition_possible": four_role_possible,
        "target_modality_coverage": {
            "small_molecule_compound": {
                "status": "static_transcriptomic_partial_only",
                "qualified_action_count": evidence["qualified_action_count"],
                "dominant_source_action_count": action_source_counts.get(
                    "clue_lincs", 0
                ),
                "independent_non_dominant_source_action_count": sum(
                    count
                    for source, count in action_source_counts.items()
                    if source != "clue_lincs"
                ),
                "action_complete_dynamic_domain": False,
            },
            "protein_biologic_delivery": {
                "status": (
                    "available" if protein_biologic_action_count > 0 else "absent"
                ),
                "qualified_action_count": protein_biologic_action_count,
                "action_complete_dynamic_domain": False,
            },
            "electrical_current": {
                "status": "restricted_single_source_only",
                "matched_pair_count": c2_split["pair_count"],
                "target_protocol_modality": False,
            },
        },
        "missing_indispensable_dimensions": missing,
        "training_authorized": False,
        "new_data_intake_authorized": policy["stop_policy"][
            "new_public_data_intake_authorized"
        ],
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "artifact_bindings": {
            **{
                name: _binding(_repo_path(str(path)))
                for name, path in policy["inputs"].items()
                if name != "six_context_matrix"
            },
            "six_context_matrix": _binding(inputs["six_context_matrix_path"]),
            "semantic_exclusion_audit": _payload_binding(semantic_path, semantic),
            "authority_contract": _binding(CONTRACT_PATH),
            "authority_contract_schema": _binding(CONTRACT_SCHEMA_PATH),
            "authority_schema": _binding(AUTHORITY_SCHEMA_PATH),
            "exclusion_schema": _binding(EXCLUSION_SCHEMA_PATH),
            "preregistration_schema": _binding(PREREGISTRATION_SCHEMA_PATH),
            "decision_schema": _binding(DECISION_SCHEMA_PATH),
            "authorization_schema": _binding(AUTHORIZATION_SCHEMA_PATH),
            "validation_schema": _binding(VALIDATION_SCHEMA_PATH),
            "implementation": _binding(BUILDER_PATH),
            "validator": _binding(VALIDATOR_PATH),
            "tests": _binding(TEST_PATH),
            "validation_wrapper": _binding(VALIDATION_WRAPPER_PATH),
        },
    }


def _build_preregistration(
    policy: Mapping[str, Any], authority: Mapping[str, Any], authority_path: Path
) -> dict[str, Any]:
    qualified = authority["status"] == "PROSPECTIVE_TRAINING_DOMAIN_QUALIFIED"
    return {
        "schema_version": "neuronsifter-prospective-method-preregistration-v1",
        "status": (
            "PREREGISTERED" if qualified else "NOT_ISSUED_TRAINING_DOMAIN_UNQUALIFIED"
        ),
        "audit_uid": policy["audit_uid"],
        "reviewed_on": policy["reviewed_on"],
        "method_uid": None,
        "freeze_fields": {
            "architecture": None,
            "world_action_schema": None,
            "training_manifest_digest": None,
            "split_manifest_digest": None,
            "training_objective": None,
            "validation_primary_metric": None,
            "checkpoint_selection_rule": None,
            "early_stopping_rule": None,
            "uncertainty_method": None,
            "calibration_method": None,
            "seed_policy": None,
            "training_budget": None,
            "hyperparameter_search_budget": None,
            "failure_rules": None,
        },
        "reason": (
            "No independent four-role development domain exists; preregistration cannot be "
            "issued without inventing or reusing non-independent evidence."
        ),
        "training_authorized": False,
        "gpu_authorized": False,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "authority_binding": _payload_binding(authority_path, authority),
    }


def _build_decision(
    policy: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_path: Path,
    exclusion: Mapping[str, Any],
    exclusion_path: Path,
    preregistration: Mapping[str, Any],
    preregistration_path: Path,
) -> dict[str, Any]:
    qualified = authority["status"] == "PROSPECTIVE_TRAINING_DOMAIN_QUALIFIED"
    return {
        "schema_version": "neuronsifter-prospective-twin-reconstruction-decision-v1",
        "status": (
            "PROSPECTIVE_ZERO_SHOT_METHOD_QUALIFIED"
            if qualified
            else policy["stop_policy"]["insufficient_domain_status"]
        ),
        "audit_uid": policy["audit_uid"],
        "reviewed_on": policy["reviewed_on"],
        "terminal_blocker": (
            None if qualified else policy["stop_policy"]["insufficient_domain_status"]
        ),
        "work_package_status": {
            "prospective_development_data_authority": (
                "COMPLETE_POSITIVE" if qualified else "COMPLETE_NEGATIVE"
            ),
            "six_context_semantic_exclusion": (
                "COMPLETE_POSITIVE"
                if exclusion["status"] == "passed"
                else "COMPLETE_NEGATIVE"
            ),
            "prospective_method_preregistration": (
                "ISSUED"
                if preregistration["status"] == "PREREGISTERED"
                else "NOT_ISSUED"
            ),
            "structured_multimodal_world_action_compiler": (
                "PENDING" if qualified else policy["stop_policy"]["downstream_status"]
            ),
            "formal_training_selection_calibration": (
                "PENDING" if qualified else policy["stop_policy"]["downstream_status"]
            ),
            "formal_method_freeze": (
                "PENDING" if qualified else policy["stop_policy"]["downstream_status"]
            ),
            "non_evaluation_end_to_end_qualification": (
                "PENDING" if qualified else policy["stop_policy"]["downstream_status"]
            ),
        },
        "scientific_interpretation": (
            "Existing assets provide static transcriptomic evidence and simulator-restricted "
            "dynamics, but not four source-independent development roles spanning the compound "
            "and protein/biologic intervention modalities required by the prospective protocol."
        ),
        "missing_development_data": authority["missing_indispensable_dimensions"],
        "next_action": (
            "request_separate_authorization_for_bounded_public_development_data_intake"
        ),
        "execution_authority": {
            "protocol_executable": False,
            "training_authorized": False,
            "gpu_authorized": False,
            "world_action_compiler_authorized": False,
            "six_context_evaluation_authorized": False,
            "tool_calling_authorized": False,
            "agent_in_twin_authorized": False,
            "coevolution_authorized": False,
            "new_public_data_intake_authorized": False,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        },
        "artifact_bindings": {
            "development_data_authority": _payload_binding(authority_path, authority),
            "semantic_exclusion_audit": _payload_binding(exclusion_path, exclusion),
            "method_preregistration": _payload_binding(
                preregistration_path, preregistration
            ),
            "historical_checkpoint_authority": _binding(
                _repo_path(str(policy["inputs"]["checkpoint_authority"]))
            ),
        },
    }


def _build_authorization(
    policy: Mapping[str, Any], decision: Mapping[str, Any], decision_path: Path
) -> dict[str, Any]:
    qualified = decision["status"] == "PROSPECTIVE_ZERO_SHOT_METHOD_QUALIFIED"
    return {
        "schema_version": "neuronsifter-prospective-twin-training-authorization-v1",
        "status": "issuable_pending_gpu_authorization" if qualified else "not_issuable",
        "development_domain_qualified": qualified,
        "method_preregistered": qualified,
        "training_authorized": False,
        "gpu_authorized": False,
        "audit_uid": policy["audit_uid"],
        "reviewed_on": policy["reviewed_on"],
        "allowed_scope": [],
        "blocked_scope": [
            "structured_multimodal_world_action_compiler",
            "formal_twin_training",
            "checkpoint_selection",
            "uncertainty_calibration",
            "non_evaluation_qualification",
            "six_context_evaluation",
            "tool_calling",
            "agent_in_twin",
            "coevolution",
            "protected_holdout_access",
        ],
        "terminal_blocker": decision["terminal_blocker"],
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "decision_binding": _payload_binding(decision_path, decision),
    }


def build(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    outputs = {key: _repo_path(str(value)) for key, value in policy["outputs"].items()}
    inputs = _load_inputs(policy)
    checkpoint_authority = inputs["checkpoint_authority"]
    if checkpoint_authority.get("status") != "NO_FORMAL_ZERO_SHOT_CHECKPOINT":
        raise ValueError("Historical checkpoint route is not formally closed")
    semantic = _build_semantic_exclusion(policy, inputs)
    authority = _build_data_authority(
        policy, inputs, semantic, outputs["semantic_exclusion_audit"]
    )
    preregistration = _build_preregistration(
        policy, authority, outputs["development_data_authority"]
    )
    decision = _build_decision(
        policy,
        authority,
        outputs["development_data_authority"],
        semantic,
        outputs["semantic_exclusion_audit"],
        preregistration,
        outputs["method_preregistration"],
    )
    authorization = _build_authorization(
        policy, decision, outputs["reconstruction_decision"]
    )
    for payload, schema_path, label in (
        (authority, AUTHORITY_SCHEMA_PATH, "authority"),
        (semantic, EXCLUSION_SCHEMA_PATH, "semantic_exclusion"),
        (preregistration, PREREGISTRATION_SCHEMA_PATH, "preregistration"),
        (decision, DECISION_SCHEMA_PATH, "decision"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization"),
    ):
        errors = _schema_errors(payload, schema_path)
        if errors:
            raise ValueError(f"{label} schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(outputs["semantic_exclusion_audit"], semantic)
        write_json(outputs["development_data_authority"], authority)
        write_json(outputs["method_preregistration"], preregistration)
        write_json(outputs["reconstruction_decision"], decision)
        write_json(outputs["conditional_authorization"], authorization)
    return {
        "semantic_exclusion_audit": semantic,
        "development_data_authority": authority,
        "method_preregistration": preregistration,
        "reconstruction_decision": decision,
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
                "development_domain_status": result["development_data_authority"][
                    "status"
                ],
                "semantic_exclusion_status": result["semantic_exclusion_audit"][
                    "status"
                ],
                "method_preregistration_status": result["method_preregistration"][
                    "status"
                ],
                "reconstruction_status": result["reconstruction_decision"]["status"],
                "training_authorized": result["conditional_authorization"][
                    "training_authorized"
                ],
                "gpu_authorized": result["conditional_authorization"]["gpu_authorized"],
                "six_context_outcome_access": result["conditional_authorization"][
                    "six_context_outcome_access"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
