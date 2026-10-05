"""Adjudicate the exhausted prospective development-data foundation intake."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_prospective_development_data_intake import (  # noqa: E402
    BUILDER_PATH as INTAKE_BUILDER_PATH,
    DISCOVERY_WRAPPER_PATH as INTAKE_DISCOVERY_WRAPPER_PATH,
    QUEUE_FIELDS,
    TEST_PATH as INTAKE_TEST_PATH,
    UNIVERSE_FIELDS,
    VALIDATOR_PATH as INTAKE_VALIDATOR_PATH,
)
from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    _binding,
    _payload_binding,
    _read_json,
    _repo_path,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    write_json,
    write_text_atomic,
)
from Libs.scripts.validators.validate_prospective_development_data_intake import (  # noqa: E402
    validate as validate_intake,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTRACT_PATH = (
    REPO_ROOT
    / "Libs/configs/validation/prospective_development_foundation_adjudication.yaml"
)
CONTRACT_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_foundation_adjudication_config.schema.json"
)
RECOVERY_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_source_recovery.schema.json"
)
TAXONOMY_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_intake_failure_taxonomy.schema.json"
)
ROLE_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_four_role_assignment.schema.json"
)
DECISION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_foundation_decision.schema.json"
)
AUTHORIZATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_foundation_authorization.schema.json"
)
VALIDATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_foundation_validation.schema.json"
)
INTAKE_PATH = REPO_ROOT / "Output/audits/prospective_development_data_intake.json"
INTAKE_VALIDATION_PATH = (
    REPO_ROOT / "Output/audits/prospective_development_data_intake_validation.json"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_data_intake_preregistration.json"
)
IMPLEMENTATION_FREEZE_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_intake_implementation_freeze_v3.json"
)
UNIVERSE_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_metadata_universe.tsv"
)
QUEUE_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_source_queue.tsv"
)
METADATA_SNAPSHOT_PATH = (
    REPO_ROOT / "Input/raw/public/prospective_development/metadata_snapshot.json"
)
DEVELOPMENT_AUTHORITY_PATH = (
    REPO_ROOT / "Output/audits/prospective_development_data_authority.json"
)
SEMANTIC_EXCLUSION_PATH = (
    REPO_ROOT / "Output/audits/six_context_semantic_exclusion_audit.json"
)
METHOD_PREREGISTRATION_PATH = (
    REPO_ROOT / "Output/audits/prospective_method_preregistration.json"
)
BUILDER_PATH = (
    REPO_ROOT
    / "Libs/scripts/builders/build_prospective_development_foundation_adjudication.py"
)
VALIDATOR_PATH = (
    REPO_ROOT
    / "Libs/scripts/validators/validate_prospective_development_foundation_adjudication.py"
)
TEST_PATH = (
    REPO_ROOT / "Libs/tests/test_prospective_development_foundation_adjudication.py"
)
VALIDATION_WRAPPER_PATH = (
    REPO_ROOT
    / "Libs/exp/evaluation/run_prospective_development_foundation_adjudication.sbatch"
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
    policy = config.get("prospective_development_foundation_adjudication")
    if not isinstance(policy, dict):
        raise ValueError(
            "Missing prospective_development_foundation_adjudication config"
        )
    errors = _schema_errors(policy, CONTRACT_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Foundation adjudication config errors: " + "; ".join(errors[:8])
        )
    return policy


def _read_tsv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"TSV columns drifted: {path}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def _assert_parent(
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    current_validation = validate_intake(write_output=False)
    if current_validation.get("passed") is not True:
        raise ValueError(
            "Foundation adjudication requires a valid intake parent: "
            + "; ".join(current_validation.get("blockers", ["unknown blocker"]))
        )
    persisted_validation = _read_json(INTAKE_VALIDATION_PATH)
    if (
        persisted_validation.get("passed") is not True
        or persisted_validation.get("blocker_count") != 0
    ):
        raise ValueError("Persisted intake validation is not a passing receipt")
    intake = _read_json(INTAKE_PATH)
    universe = _read_tsv(UNIVERSE_PATH, UNIVERSE_FIELDS)
    queue = _read_tsv(QUEUE_PATH, QUEUE_FIELDS)
    expected = policy["expected_parent"]
    observed = {
        "intake_status": intake.get("status"),
        "metadata_asset_count": len(universe),
        "metadata_eligible_count": intake.get("metadata_eligible_count"),
        "queue_lineage_count": len(queue),
        "six_context_outcome_access": intake.get("six_context_outcome_access"),
        "protected_holdout_access": intake.get("protected_holdout_access"),
    }
    if observed != expected:
        raise ValueError(f"Foundation adjudication parent drifted: {observed}")
    validation_observed = {
        "intake_status": current_validation.get("intake_status"),
        "metadata_asset_count": current_validation.get("metadata_asset_count"),
        "metadata_eligible_count": current_validation.get("metadata_eligible_count"),
        "queue_lineage_count": current_validation.get("queue_lineage_count"),
        "queue_digest": current_validation.get("queue_digest"),
    }
    validation_expected = {
        "intake_status": intake.get("status"),
        "metadata_asset_count": len(universe),
        "metadata_eligible_count": intake.get("metadata_eligible_count"),
        "queue_lineage_count": len(queue),
        "queue_digest": intake.get("queue_digest"),
    }
    if validation_observed != validation_expected:
        raise ValueError(
            f"Foundation adjudication intake validation drifted: {validation_observed}"
        )
    if any(row["eligibility_status"] == "eligible" for row in universe):
        raise ValueError("Parent universe contains an eligible source")
    return intake, universe


def _build_taxonomy(
    policy: Mapping[str, Any], universe: list[dict[str, str]]
) -> dict[str, Any]:
    rejection_counts = Counter(row["rejection_reason"] for row in universe)
    modality_counts = Counter(
        row["intervention_modality"] or "none" for row in universe
    )
    endpoint_counts = Counter(
        row["functional_endpoint_class"] or "none" for row in universe
    )
    repository_counts = Counter(row["repository_family"] for row in universe)
    signals = {
        "cns_or_neural_model": sum(bool(row["biological_system"]) for row in universe),
        "target_intervention_modality": sum(
            bool(row["intervention_modality"]) for row in universe
        ),
        "baseline_or_control": sum(
            row["baseline_control_signal"] == "true" for row in universe
        ),
        "post_action": sum(row["post_action_signal"] == "true" for row in universe),
        "dose_or_intensity": sum(row["dose_signal"] == "true" for row in universe),
        "timepoint": sum(row["timepoint_signal"] == "true" for row in universe),
        "replication": sum(row["replication_signal"] == "true" for row in universe),
        "functional_endpoint": sum(
            bool(row["functional_endpoint_class"]) for row in universe
        ),
        "transcriptomic_auxiliary": sum(
            row["transcriptomic_auxiliary_signal"] == "true" for row in universe
        ),
    }
    capability_resolution = {
        "pharmacology_source_diversity": False,
        "protein_biologic_domain": False,
        "independent_validation": False,
        "independent_calibration": False,
        "real_longitudinal_units": False,
        "functional_endpoint_coverage": False,
        "independent_qualification_source": False,
    }
    return {
        "schema_version": "neuronsifter-prospective-development-intake-failure-taxonomy-v1",
        "status": "bounded_metadata_universe_no_eligible_lineage",
        "adjudication_uid": policy["adjudication_uid"],
        "reviewed_on": policy["reviewed_on"],
        "metadata_asset_count": len(universe),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "classification_policy": "first_failure_reason_per_asset",
        "metadata_signal_counts": signals,
        "modality_signal_counts": dict(sorted(modality_counts.items())),
        "functional_endpoint_signal_counts": dict(sorted(endpoint_counts.items())),
        "repository_family_counts": dict(sorted(repository_counts.items())),
        "capability_resolution": capability_resolution,
        "screening_metadata_scope": (
            "Outcome-blind study titles, repository identities, asset filenames/types, and "
            "provider-native design metadata available under the frozen implementation."
        ),
        "cross_repository_lineage_deduplication_complete": False,
        "semantic_exclusion_implementation_complete": False,
        "scope_boundary": (
            "The taxonomy applies only to the preregistered five-family, 320-asset-cap metadata "
            "universe and does not claim that suitable development data are globally absent."
        ),
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "artifact_bindings": {
            "intake_audit": _binding(INTAKE_PATH),
            "intake_validation": _binding(INTAKE_VALIDATION_PATH),
            "metadata_universe": _binding(UNIVERSE_PATH),
            "preregistration": _binding(PREREGISTRATION_PATH),
            "implementation_freeze": _binding(IMPLEMENTATION_FREEZE_PATH),
        },
    }


def _build_source_recovery(
    policy: Mapping[str, Any], intake: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-prospective-development-source-recovery-v1",
        "status": "NOT_RUN_QUEUE_EMPTY",
        "adjudication_uid": policy["adjudication_uid"],
        "reviewed_on": policy["reviewed_on"],
        "intake_uid": intake["intake_uid"],
        "metadata_asset_count": intake["metadata_asset_count"],
        "metadata_eligible_count": intake["metadata_eligible_count"],
        "queue_lineage_count": intake["queue_lineage_count"],
        "source_payloads_recovered": 0,
        "empirical_action_response_units": 0,
        "reason": "The frozen source queue is empty; no source payload may be opened or materialized.",
        "development_outcome_access": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "artifact_bindings": {
            "intake_audit": _binding(INTAKE_PATH),
            "source_queue": _binding(QUEUE_PATH),
            "metadata_snapshot": _binding(METADATA_SNAPSHOT_PATH),
        },
    }


def _build_role_assignment(policy: Mapping[str, Any]) -> dict[str, Any]:
    roles = {
        role: {
            "status": "unassigned_no_eligible_lineage",
            "lineage_uids": [],
            "source_count": 0,
            "unit_count": 0,
        }
        for role in (
            "TRAIN",
            "MODEL_SELECTION_VALIDATION",
            "UNCERTAINTY_CALIBRATION",
            "NON_EVAL_QUALIFICATION",
        )
    }
    return {
        "schema_version": "neuronsifter-prospective-four-role-assignment-v1",
        "status": "NOT_ASSIGNABLE_QUEUE_EMPTY",
        "adjudication_uid": policy["adjudication_uid"],
        "reviewed_on": policy["reviewed_on"],
        "roles": roles,
        "independence_graph_node_count": 0,
        "independence_graph_edge_count": 0,
        "all_roles_nonempty": False,
        "source_lineage_disjoint": False,
        "assignment_performed": False,
        "reason": "No eligible source lineage was queued or recovered.",
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }


def _build_decision(
    policy: Mapping[str, Any],
    recovery: Mapping[str, Any],
    recovery_path: Path,
    taxonomy: Mapping[str, Any],
    taxonomy_path: Path,
    assignment: Mapping[str, Any],
    assignment_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-prospective-development-foundation-decision-v1",
        "status": policy["terminal_status"],
        "adjudication_uid": policy["adjudication_uid"],
        "reviewed_on": policy["reviewed_on"],
        "scope_boundary": (
            "The decision applies to the exhausted preregistered five-family public title, file, "
            "and provider-native design-metadata universe. It is not a global absence claim and "
            "does not authorize a second intake."
        ),
        "interpretation_limitations": [
            "BioStudies and Europe PMC screening intentionally discarded unrestricted abstract or result prose.",
            "Rejection taxonomy records the first failed criterion and is not a mutually exclusive scientific absence taxonomy.",
            "Cross-repository canonical lineage deduplication was not exercised because no record reached eligibility.",
            "Complete semantic exclusion was not exercised because the frozen queue was empty.",
            "Metadata role-capability labels are screening candidates only; formal roles require source-lineage-independent assignment.",
        ],
        "failure_decomposition": taxonomy["capability_resolution"],
        "work_package_status": {
            "bounded_metadata_intake": "COMPLETE_NEGATIVE",
            "source_payload_recovery": "NOT_RUN_QUEUE_EMPTY",
            "empirical_action_response_materialization": "NOT_RUN_QUEUE_EMPTY",
            "independence_graph": "EMPTY_NO_ELIGIBLE_LINEAGES",
            "four_role_assignment": "NOT_ASSIGNABLE_QUEUE_EMPTY",
            "development_domain_adjudication": "COMPLETE_NEGATIVE",
            "terminal_intake_reentry_guard": "ENFORCED_NO_SECOND_INTAKE",
            "method_preregistration": "NOT_ISSUED",
        },
        "next_action": "STOP_AND_REQUIRE_NEW_ROUTE_LEVEL_AUTHORIZATION",
        "execution_authority": {
            "automatic_second_intake_allowed": False,
            "new_public_data_intake_authorized": False,
            "method_preregistration_allowed": False,
            "world_action_compiler_authorized": False,
            "training_authorized": False,
            "gpu_authorized": False,
            "six_context_evaluation_authorized": False,
            "agent_in_twin_authorized": False,
            "coevolution_authorized": False,
            "development_outcome_access": 0,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        },
        "artifact_bindings": {
            "source_recovery": _payload_binding(recovery_path, recovery),
            "failure_taxonomy": _payload_binding(taxonomy_path, taxonomy),
            "role_assignment": _payload_binding(assignment_path, assignment),
            "development_data_authority_before_intake": _binding(
                DEVELOPMENT_AUTHORITY_PATH
            ),
            "semantic_exclusion_before_intake": _binding(SEMANTIC_EXCLUSION_PATH),
            "method_preregistration_before_intake": _binding(
                METHOD_PREREGISTRATION_PATH
            ),
            "adjudication_contract": _binding(CONTRACT_PATH),
            "adjudication_contract_schema": _binding(CONTRACT_SCHEMA_PATH),
            "recovery_schema": _binding(RECOVERY_SCHEMA_PATH),
            "taxonomy_schema": _binding(TAXONOMY_SCHEMA_PATH),
            "role_assignment_schema": _binding(ROLE_SCHEMA_PATH),
            "decision_schema": _binding(DECISION_SCHEMA_PATH),
            "authorization_schema": _binding(AUTHORIZATION_SCHEMA_PATH),
            "validation_schema": _binding(VALIDATION_SCHEMA_PATH),
            "intake_validation_receipt": _binding(INTAKE_VALIDATION_PATH),
            "terminal_intake_reentry_guard_builder": _binding(INTAKE_BUILDER_PATH),
            "terminal_intake_integrity_validator": _binding(INTAKE_VALIDATOR_PATH),
            "terminal_intake_guard_tests": _binding(INTAKE_TEST_PATH),
            "terminal_intake_discovery_wrapper": _binding(
                INTAKE_DISCOVERY_WRAPPER_PATH
            ),
            "builder": _binding(BUILDER_PATH),
            "validator": _binding(VALIDATOR_PATH),
            "tests": _binding(TEST_PATH),
            "validation_wrapper": _binding(VALIDATION_WRAPPER_PATH),
        },
    }


def _build_authorization(
    policy: Mapping[str, Any], decision: Mapping[str, Any], decision_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-prospective-development-foundation-authorization-v1",
        "status": "not_issuable",
        "development_domain_qualified": False,
        "method_preregistration_allowed": False,
        "training_authorized": False,
        "gpu_authorized": False,
        "adjudication_uid": policy["adjudication_uid"],
        "reviewed_on": policy["reviewed_on"],
        "allowed_scope": [],
        "blocked_scope": [
            "automatic_second_development_intake",
            "prospective_method_uid",
            "world_action_compiler",
            "formal_training",
            "checkpoint_selection",
            "uncertainty_calibration",
            "six_context_evaluation",
            "tool_calling",
            "agent_in_twin",
            "coevolution",
            "protected_holdout_access",
        ],
        "terminal_status": decision["status"],
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "decision_binding": _payload_binding(decision_path, decision),
    }


def _build_report(
    intake: Mapping[str, Any], taxonomy: Mapping[str, Any], decision: Mapping[str, Any]
) -> str:
    rejection_rows = "\n".join(
        f"| {reason or '<none>'} | {count} |"
        for reason, count in taxonomy["rejection_reason_counts"].items()
    )
    return f"""# Prospective Development-Data Failure Analysis

## Decision

`{decision["status"]}`

This conclusion is restricted to the preregistered five-family metadata universe. It does not claim that suitable public development data are globally absent.

## Outcome-Blind Funnel

- Metadata assets screened: {intake["metadata_asset_count"]}
- Metadata-eligible assets: {intake["metadata_eligible_count"]}
- Independent lineages queued: {intake["queue_lineage_count"]}
- Source payloads recovered: 0
- Empirical action-response units materialized: 0
- Six-context outcome access: 0
- Protected holdout access: 0

## Rejection Taxonomy

| Reason | Count |
| --- | ---: |
{rejection_rows}

## Capability Resolution

The bounded universe did not provide an admissible source lineage for multi-source pharmacology, protein/biologic delivery, independent validation or calibration, real longitudinal action-response, functional qualification, or a source-independent qualification role. Because the frozen queue is empty, source recovery and materialization were not run.

## Stop Rule

No second intake, method preregistration, WorldAction implementation, training, GPU work, or six-context evaluation is authorized by this result.

The terminal re-entry guard rejects reuse of the historical access-block retry after a queue has been frozen or exhausted. Metadata role labels remain screening candidates and are not formal role assignments.
"""


def build(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    outputs = {key: _repo_path(str(value)) for key, value in policy["outputs"].items()}
    intake, universe = _assert_parent(policy)
    recovery = _build_source_recovery(policy, intake)
    taxonomy = _build_taxonomy(policy, universe)
    assignment = _build_role_assignment(policy)
    decision = _build_decision(
        policy,
        recovery,
        outputs["source_recovery_audit"],
        taxonomy,
        outputs["failure_taxonomy"],
        assignment,
        outputs["role_assignment"],
    )
    authorization = _build_authorization(
        policy, decision, outputs["foundation_decision"]
    )
    report = _build_report(intake, taxonomy, decision)
    for payload, schema_path, label in (
        (recovery, RECOVERY_SCHEMA_PATH, "recovery"),
        (taxonomy, TAXONOMY_SCHEMA_PATH, "taxonomy"),
        (assignment, ROLE_SCHEMA_PATH, "role_assignment"),
        (decision, DECISION_SCHEMA_PATH, "decision"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "authorization"),
    ):
        errors = _schema_errors(payload, schema_path)
        if errors:
            raise ValueError(f"{label} schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(outputs["source_recovery_audit"], recovery)
        write_json(outputs["failure_taxonomy"], taxonomy)
        write_json(outputs["role_assignment"], assignment)
        write_json(outputs["foundation_decision"], decision)
        write_json(outputs["conditional_authorization"], authorization)
        write_text_atomic(outputs["failure_analysis_report"], report)
    return {
        "source_recovery": recovery,
        "failure_taxonomy": taxonomy,
        "role_assignment": assignment,
        "foundation_decision": decision,
        "conditional_authorization": authorization,
        "failure_analysis_report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = build(write_output=not args.no_write)
    print(
        json.dumps(
            {
                "status": result["foundation_decision"]["status"],
                "metadata_asset_count": result["failure_taxonomy"][
                    "metadata_asset_count"
                ],
                "queue_lineage_count": result["source_recovery"]["queue_lineage_count"],
                "source_payloads_recovered": result["source_recovery"][
                    "source_payloads_recovered"
                ],
                "training_authorized": result["conditional_authorization"][
                    "training_authorized"
                ],
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
