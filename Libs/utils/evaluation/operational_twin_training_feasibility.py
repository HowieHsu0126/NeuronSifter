"""Fail-closed qualification audit for Operational Twin open development data."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS
from Libs.utils.data.controlled_world_open_package import (
    CLOSED_DOWNSTREAM_FLAGS,
    OPEN_MANIFEST_SCHEMA_VERSION,
    OPEN_PACKAGE_POSITIVE_TERMINAL,
    OPEN_PACKAGE_RECEIPT_SCHEMA_VERSION,
    ZERO_ACCESS_COUNTERS,
    reject_forbidden_path_tokens,
    validate_open_manifests,
)
from Libs.utils.data.controlled_world_tensorization import (
    ControlledWorldAccessError,
    validate_controlled_manifest_payload,
)


SCHEMA_VERSION = "neuronsifter-operational-twin-training-feasibility-v1"
QUALIFICATION_SCHEMA_VERSION = "neuronsifter-controlled-world-open-data-qualification-v1"
BLOCKED_TERMINAL = "BLOCKED_OPEN_DEVELOPMENT_INTERVENTION_SUPPORT_INSUFFICIENT"
PENDING_QUALIFICATION_TERMINAL = "OPEN_DATA_QUALIFICATION_PENDING"
QUALIFIED_TERMINAL = "OPEN_TWIN_DEVELOPMENT_DATA_QUALIFIED"
NEGATIVE_TERMINAL_PREFIX = "OPEN_TWIN_DEVELOPMENT_DATA_NOT_QUALIFIED"
OPEN_ROLES = ("training", "development", "calibration")
ZERO_ACCESS_FLAGS = dict(ZERO_ACCESS_COUNTERS)
CLOSED_CLAIM_FLAGS = dict(CLOSED_DOWNSTREAM_FLAGS)
QUALIFICATION_CLOSED_FLAGS = dict(CLOSED_DOWNSTREAM_FLAGS, training_authorized=False)
ROLE_ISOLATION_FIELDS = (
    "world_uid",
    "latent_world_uid",
    "latent_generator_lineage_uid",
    "seed_identity",
    "replay_identity",
    "observable_output_identity",
    "protocol_state_identity",
)
ACTION_RESULT_IDENTITY_FIELD = "protocol_state_identity"


def _terminal_for_blocker(code: str) -> str:
    if "schema" in code or "missing_open_manifest" in code or "receipt" in code:
        return f"{NEGATIVE_TERMINAL_PREFIX}_SCHEMA_INVALID"
    if "pair" in code or "action" in code or "estimand" in code:
        return f"{NEGATIVE_TERMINAL_PREFIX}_ACTION_PAIRING_INVALID"
    if "overlap" in code or "lineage" in code or "role" in code:
        return f"{NEGATIVE_TERMINAL_PREFIX}_LINEAGE_LEAKAGE"
    if "ood" in code or "support" in code:
        return f"{NEGATIVE_TERMINAL_PREFIX}_SUPPORT_COVERAGE"
    if "replay" in code or "identity" in code or "provenance" in code or "content" in code:
        return f"{NEGATIVE_TERMINAL_PREFIX}_REPLAY_PROVENANCE"
    if "access" in code or "forbidden" in code or "gate" in code:
        return f"{NEGATIVE_TERMINAL_PREFIX}_FORBIDDEN_ACCESS"
    return f"{NEGATIVE_TERMINAL_PREFIX}_SCHEMA_INVALID"


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_tracked_open_manifest(
    root: Path,
    logical_path: str,
    expected_role: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read one explicitly allowed open action-identification manifest."""
    if expected_role not in OPEN_ROLES:
        raise ControlledWorldAccessError(f"Unsupported open manifest role: {expected_role}")
    path = Path(logical_path)
    if path.is_absolute() or ".." in path.parts:
        raise ControlledWorldAccessError("Open manifest path must be repository-relative.")
    if path.name != f"{expected_role}_manifest.json":
        raise ControlledWorldAccessError("Open manifest path does not match its declared role.")
    reject_forbidden_path_tokens(path, label=f"{expected_role}_manifest")
    if "open_action_identification" not in path.parts:
        raise ControlledWorldAccessError("Open manifest path is not in the governed open action-identification package.")

    manifest_path = root / path
    if not manifest_path.is_file():
        raise ControlledWorldAccessError("Open action-identification manifest is absent.")
    content = manifest_path.read_bytes()
    payload = validate_controlled_manifest_payload(json.loads(content), expected_role)
    return payload, {
        "path": path.as_posix(),
        "authority": "open_action_identification_manifest_file",
        "sha256": hashlib.sha256(content).hexdigest(),
        "content_opened": True,
        "role": expected_role,
    }


def _family_summary(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows_by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in manifest["worlds"]:
        rows_by_family[str(row["family_uid"])].append(row)
    support_cases_by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in manifest.get("development_support_cases", []):
        support_cases_by_family[str(row["family_uid"])].append(row)
    pair_uids = {str(pair["counterfactual_pair_uid"]) for pair in manifest.get("counterfactual_pairs", [])}

    summary: dict[str, dict[str, object]] = {}
    for family_uid in FAMILY_UIDS:
        rows = rows_by_family.get(family_uid, [])
        action_ids = {str(row["action_sequence_identity"]) for row in rows}
        support_statuses = {str(dict(row["support_status"])["status"]) for row in rows}
        support_case_statuses = {
            str(dict(row["support_status"])["status"])
            for row in support_cases_by_family.get(family_uid, [])
        }
        actions_by_pair: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            pair_uid = str(row["counterfactual_pair_uid"])
            if pair_uid in pair_uids:
                actions_by_pair[pair_uid].add(str(row["action_sequence_identity"]))
        valid_pair_count = sum(1 for values in actions_by_pair.values() if len(values) == 2)
        summary[family_uid] = {
            "world_count": len(rows),
            "latent_world_count": int(dict(manifest["family_counts"][family_uid])["latent_worlds"]),
            "distinct_action_sequence_identity_count": len(action_ids),
            "support_statuses": sorted(support_statuses),
            "development_support_case_statuses": sorted(support_case_statuses),
            "counterfactual_pairing_fields": ["counterfactual_pair_uid"],
            "valid_counterfactual_pair_count": valid_pair_count,
            "counterfactual_pairing_available": valid_pair_count > 0,
            "estimands_computable": {
                "action_contrast_error": len(action_ids) >= 2 and valid_pair_count > 0,
                "intervention_ranking_regret": len(action_ids) >= 2 and valid_pair_count > 0,
                "effect_direction": len(action_ids) >= 2 and valid_pair_count > 0,
                "trajectory_error": len(rows) > 0,
                "five_method_comparison": len(action_ids) >= 2 and valid_pair_count > 0,
                "support_detection": {"supported", "out_of_distribution", "unsupported"}.issubset(support_case_statuses),
                "abstention": {"out_of_distribution", "unsupported"}.issubset(support_case_statuses),
            },
        }
    return summary


def load_open_package_receipt(root: Path, logical_path: str) -> tuple[dict[str, object], dict[str, object]]:
    path = Path(logical_path)
    if path.is_absolute() or ".." in path.parts:
        raise ControlledWorldAccessError("Open package receipt path must be repository-relative.")
    if path.name != "package_receipt.json":
        raise ControlledWorldAccessError("Open package receipt path does not match the governed package receipt.")
    reject_forbidden_path_tokens(path, label="open_action_package_receipt")
    receipt_path = root / path
    if not receipt_path.is_file():
        raise ControlledWorldAccessError("Open action-identification package receipt is absent.")
    content = receipt_path.read_bytes()
    payload = json.loads(content)
    if payload.get("schema_version") != OPEN_PACKAGE_RECEIPT_SCHEMA_VERSION:
        raise ControlledWorldAccessError("Open package receipt schema version is invalid.")
    if payload.get("terminal_decision") != OPEN_PACKAGE_POSITIVE_TERMINAL:
        raise ControlledWorldAccessError("Open package construction terminal is not positive.")
    if payload.get("open_data_qualification") != "pending" or payload.get("training_authorized") is not False:
        raise ControlledWorldAccessError("Open package receipt construction-time flags are not preserved.")
    if payload.get("access_counters") != ZERO_ACCESS_COUNTERS:
        raise ControlledWorldAccessError("Open package receipt access counters must remain exactly zero.")
    return payload, {
        "path": path.as_posix(),
        "authority": "open_action_identification_package_receipt",
        "sha256": hashlib.sha256(content).hexdigest(),
        "content_opened": True,
        "receipt_uid": payload.get("receipt_uid"),
    }


def _package_identity_blockers(
    manifests: Mapping[str, Mapping[str, object]],
    identities: Mapping[str, Mapping[str, object]],
    receipt: Mapping[str, object],
    receipt_identity: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if dict(receipt.get("manifests", {})) != {role: identities[role]["path"] for role in OPEN_ROLES}:
        blockers.append({"code": "package_receipt_manifest_path_mismatch", "reason": "Receipt manifest paths do not match the opened canonical identities."})
    expected_content = {role: manifests[role].get("content_identity") for role in OPEN_ROLES}
    if dict(receipt.get("manifest_content_identities", {})) != expected_content:
        blockers.append({"code": "package_receipt_manifest_content_identity_mismatch", "reason": "Receipt manifest content identities do not match recomputed opened manifests."})
    if receipt.get("closed_downstream_flags") != CLOSED_DOWNSTREAM_FLAGS:
        blockers.append({"code": "package_receipt_downstream_gate_open", "reason": "Receipt downstream gates are not closed."})
    if not receipt_identity.get("sha256"):
        blockers.append({"code": "package_receipt_identity_missing", "reason": "Receipt content identity was not captured."})
    return blockers


def _role_lineage_isolation(
    manifests: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    values_by_field: dict[str, dict[str, set[str]]] = {
        field: {role: set() for role in OPEN_ROLES}
        for field in ROLE_ISOLATION_FIELDS
    }
    for role in OPEN_ROLES:
        for row in manifests[role].get("worlds", []):
            record = dict(row)
            for field in ROLE_ISOLATION_FIELDS:
                value = record.get(field)
                if value not in (None, ""):
                    values_by_field[field][role].add(str(value))

    blockers: list[dict[str, object]] = []
    overlap_summary: list[dict[str, object]] = []
    for field, by_role in values_by_field.items():
        for left_index, left in enumerate(OPEN_ROLES):
            for right in OPEN_ROLES[left_index + 1:]:
                overlap = sorted(by_role[left] & by_role[right])
                if overlap:
                    row = {
                        "field": field,
                        "left_role": left,
                        "right_role": right,
                        "overlap_count": len(overlap),
                    }
                    overlap_summary.append(row)
                    blockers.append({
                        "code": "cross_role_lineage_overlap",
                        **row,
                        "reason": f"Cross-role reuse of {field} is forbidden for open-data qualification.",
                    })
    return blockers, {
        "checked_fields": list(ROLE_ISOLATION_FIELDS),
        "action_result_identity_field": ACTION_RESULT_IDENTITY_FIELD,
        "status": "PASS" if not blockers else "BLOCKED",
        "overlaps": overlap_summary,
    }


def _qualification_blockers(
    config: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    identities: Mapping[str, Mapping[str, object]],
    receipt: Mapping[str, object],
    receipt_identity: Mapping[str, object],
    *,
    repository_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    blockers: list[dict[str, object]] = []
    operational_twin = config.get("operational_twin")
    if not isinstance(operational_twin, Mapping):
        return ([{"code": "operational_twin_config_schema_invalid", "reason": "Canonical Operational Twin config is missing."}], {})
    if dict(operational_twin.get("access_policy", {})) != ZERO_ACCESS_FLAGS:
        blockers.append({"code": "operational_twin_forbidden_access_policy", "reason": "Operational Twin access policy must remain exactly zero."})
    if dict(operational_twin.get("claim_boundary", {})) != CLOSED_CLAIM_FLAGS:
        blockers.append({"code": "operational_twin_downstream_gate_open", "reason": "Operational Twin downstream gates must remain false."})
    expected_paths = dict(operational_twin.get("manifests", {}))
    for role in OPEN_ROLES:
        if identities[role].get("path") != expected_paths.get(role):
            blockers.append({"code": "manifest_canonical_path_mismatch", "role": role, "reason": "Opened manifest is not the canonical config-stack path."})
        if manifests[role].get("schema_version") != OPEN_MANIFEST_SCHEMA_VERSION:
            blockers.append({"code": "manifest_schema_invalid", "role": role, "reason": "Open manifest schema version is invalid."})
    blockers.extend(validate_open_manifests(manifests, config=config, repository_root=repository_root, replay=True))
    role_isolation_blockers, role_isolation = _role_lineage_isolation(manifests)
    blockers.extend(role_isolation_blockers)
    blockers.extend(_package_identity_blockers(manifests, identities, receipt, receipt_identity))
    summaries = {role: _family_summary(manifests[role]) for role in OPEN_ROLES}
    for role in OPEN_ROLES:
        for family_uid, values in summaries[role].items():
            if int(values["distinct_action_sequence_identity_count"]) < 2:
                blockers.append({"code": "family_structured_action_support_insufficient", "role": role, "family_uid": family_uid, "required_minimum": 2, "reason": "Each open family/role needs at least two distinct structured WorldAction programs."})
            if values["counterfactual_pairing_available"] is not True:
                blockers.append({"code": "family_counterfactual_pairing_absent", "role": role, "family_uid": family_uid, "reason": "Each family/role needs an approved two-member counterfactual pairing authority."})
    for family_uid, values in summaries["development"].items():
        statuses = set(values["development_support_case_statuses"])
        missing = sorted({"supported", "out_of_distribution", "unsupported"} - statuses)
        if missing:
            blockers.append({"code": "development_support_ood_unsupported_coverage_missing", "family_uid": family_uid, "missing_statuses": missing, "reason": "Development must retain supported, OOD, and unsupported support probes per family."})
    return blockers, {"family_summaries": summaries, "role_lineage_isolation": role_isolation}


def build_open_data_qualification_audit(
    config: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    identities: Mapping[str, Mapping[str, object]],
    receipt: Mapping[str, object],
    receipt_identity: Mapping[str, object],
    *,
    authoritative_date: str,
    repository_root: Path,
) -> dict[str, object]:
    blockers, qualification_summaries = _qualification_blockers(
        config,
        manifests,
        identities,
        receipt,
        receipt_identity,
        repository_root=repository_root,
    )
    summaries = dict(qualification_summaries.get("family_summaries", {}))
    role_lineage_isolation = dict(qualification_summaries.get("role_lineage_isolation", {}))
    status = "PASS" if not blockers else "BLOCKED"
    terminal = QUALIFIED_TERMINAL if not blockers else _terminal_for_blocker(str(blockers[0]["code"]))
    claim_flags = dict(CLOSED_CLAIM_FLAGS)
    if status == "PASS":
        claim_flags["training_authorized"] = True
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "status": status,
        "terminal_decision": terminal,
        "authoritative_date": authoritative_date,
        "evidence_class": "controlled_world_non_empirical",
        "authority_role": "independent_open_data_qualification",
        "training_authorized": status == "PASS",
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "input_identities": {role: dict(identities[role]) for role in OPEN_ROLES},
        "input_content_identities": {role: manifests[role].get("content_identity") for role in OPEN_ROLES},
        "construction_authority_identity": dict(receipt_identity),
        "construction_receipt_uid": receipt.get("receipt_uid"),
        "construction_receipt_content_identity": _sha256_payload(receipt),
        "precedence": {
            "supersedes_receipt_training_authorized_only_for_pretraining_authorization": status == "PASS",
            "receipt_open_data_qualification": receipt.get("open_data_qualification"),
            "receipt_training_authorized": receipt.get("training_authorized"),
        },
        "schema_validation": {
            "training_manifest": manifests["training"].get("schema_version") == OPEN_MANIFEST_SCHEMA_VERSION,
            "development_manifest": manifests["development"].get("schema_version") == OPEN_MANIFEST_SCHEMA_VERSION,
            "calibration_manifest": manifests["calibration"].get("schema_version") == OPEN_MANIFEST_SCHEMA_VERSION,
            "package_receipt": receipt.get("schema_version") == OPEN_PACKAGE_RECEIPT_SCHEMA_VERSION,
        },
        "family_action_pairing_summary": summaries,
        "role_lineage_isolation": role_lineage_isolation,
        "deterministic_replay_provenance": {
            "status": "PASS" if not any(("replay" in str(row.get("code")) or "identity" in str(row.get("code")) or "content" in str(row.get("code"))) for row in blockers) else "BLOCKED",
            "config_identity_checked": True,
            "generator_identity_checked": True,
            "action_binding_identity_checked": True,
            "manifest_content_identity_checked": True,
            "construction_audit_identity_checked": True,
        },
        "retained_failure_inventory": {
            "policy": "retain_every_failure",
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "forbidden_access_counters": dict(ZERO_ACCESS_FLAGS),
        "sealed_material_access": {
            "complete_test_manifest": 0,
            "twin_qualification_partition": 0,
            "unopened_access_receipt_content": 0,
            "protected_holdout": 0,
            "six_context_outcomes": 0,
        },
        "closed_downstream_gates": claim_flags,
    }


def build_training_feasibility_audit(
    config: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    identities: Mapping[str, Mapping[str, object]],
    *,
    authoritative_date: str,
) -> dict[str, object]:
    """Assess whether the open manifests identify the approved action-value study."""
    operational_twin = config.get("operational_twin")
    if not isinstance(operational_twin, Mapping):
        raise ControlledWorldAccessError("Canonical Operational Twin config is missing.")
    if dict(operational_twin.get("access_policy", {})) != {
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "test_outcome_access": 0,
        "tool_access": 0,
        "mcp_access": 0,
        "agent_trace_access": 0,
        "agent_access": 0,
        "memory_access": 0,
        "coevolution_access": 0,
    }:
        raise ControlledWorldAccessError("Operational Twin access policy must remain exactly zero.")
    if dict(operational_twin.get("claim_boundary", {})) != CLOSED_CLAIM_FLAGS:
        raise ControlledWorldAccessError("Operational Twin qualification and downstream-open flags must remain false.")

    expected_paths = dict(operational_twin.get("manifests", {}))
    if set(manifests) != set(OPEN_ROLES) or set(identities) != set(OPEN_ROLES):
        raise ControlledWorldAccessError("Feasibility audit requires exactly the three open manifest roles.")
    for role in OPEN_ROLES:
        if identities[role].get("path") != expected_paths.get(role):
            raise ControlledWorldAccessError(f"{role} manifest is not the canonical config-stack path.")

    summaries = {role: _family_summary(manifests[role]) for role in OPEN_ROLES}
    blockers: list[dict[str, object]] = []

    training_missing = [
        family_uid
        for family_uid, values in summaries["training"].items()
        if int(values["distinct_action_sequence_identity_count"]) < 2
    ]
    if training_missing:
        blockers.append({
            "code": "training_action_identifiability_absent",
            "families": training_missing,
            "observed_distinct_actions_per_family": {
                family_uid: summaries["training"][family_uid]["distinct_action_sequence_identity_count"]
                for family_uid in training_missing
            },
            "required_minimum": 2,
            "reason": "Action conditioning cannot be identified within family from a single action program.",
        })

    development_missing_actions = [
        family_uid
        for family_uid, values in summaries["development"].items()
        if int(values["distinct_action_sequence_identity_count"]) < 2
    ]
    development_missing_pairs = [
        family_uid
        for family_uid, values in summaries["development"].items()
        if values["counterfactual_pairing_available"] is not True
    ]
    if development_missing_actions or development_missing_pairs:
        blockers.append({
            "code": "development_action_value_estimand_unidentified",
            "families_without_action_contrast": development_missing_actions,
            "families_without_counterfactual_pairing": development_missing_pairs,
            "reason": (
                "World-level action-contrast error and intervention-ranking regret require "
                "within-family action variation with a declared pairing authority."
            ),
        })

    development_missing_ood = [
        family_uid
        for family_uid, values in summaries["development"].items()
        if {"supported", "out_of_distribution", "unsupported"}
        - set(values["development_support_case_statuses"])
    ]
    if development_missing_ood:
        blockers.append({
            "code": "development_ood_behavior_unobserved",
            "families": development_missing_ood,
            "reason": "All open development rows are supported; OOD detection and abstention cannot be measured.",
        })

    if not blockers:
        blockers.append({
            "code": "independent_open_data_qualification_pending",
            "reason": "This parser does not authorize training; a later independent open-data qualification artifact is required.",
        })
    status = "BLOCKED"
    terminal_decision = (
        PENDING_QUALIFICATION_TERMINAL
        if {row["code"] for row in blockers} == {"independent_open_data_qualification_pending"}
        else BLOCKED_TERMINAL
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "terminal_decision": terminal_decision,
        "authoritative_date": authoritative_date,
        "evidence_class": "controlled_world_non_empirical",
        "independent_unit": "world",
        "opened_manifest_identities": {role: dict(identities[role]) for role in OPEN_ROLES},
        "manifest_support_summary": summaries,
        "blockers": blockers,
        "access_flags": dict(ZERO_ACCESS_FLAGS),
        "claim_flags": dict(CLOSED_CLAIM_FLAGS),
        "sealed_material_access": {
            "complete_test_manifest": 0,
            "twin_qualification_partition": 0,
            "unopened_access_receipt_content": 0,
            "protected_holdout": 0,
            "six_context_outcomes": 0,
        },
        "required_plan_repair": (
            "Run the separate independent open-data qualification before authorizing training."
            if terminal_decision == PENDING_QUALIFICATION_TERMINAL
            else (
                "Provide a newly governed open train/development/calibration package with at least two "
                "within-family action programs, explicit counterfactual pairing, and development OOD/support "
                "cases. The repair must preserve sealed-test immutability and be approved before training."
            )
        ),
    }


def audit_canonical_training_feasibility(
    root: Path,
    config: Mapping[str, object],
    *,
    authoritative_date: str,
) -> dict[str, object]:
    operational_twin = config.get("operational_twin")
    if not isinstance(operational_twin, Mapping):
        raise ControlledWorldAccessError("Canonical Operational Twin config is missing.")
    paths = dict(operational_twin.get("manifests", {}))
    manifests: dict[str, Mapping[str, object]] = {}
    identities: dict[str, Mapping[str, object]] = {}
    for role in OPEN_ROLES:
        payload, identity = load_tracked_open_manifest(root, str(paths.get(role, "")), role)
        manifests[role] = payload
        identities[role] = identity
    return build_training_feasibility_audit(
        config,
        manifests,
        identities,
        authoritative_date=authoritative_date,
    )


def qualify_canonical_open_data(
    root: Path,
    config: Mapping[str, object],
    *,
    authoritative_date: str,
) -> dict[str, object]:
    try:
        operational_twin = config.get("operational_twin")
        if not isinstance(operational_twin, Mapping):
            raise ControlledWorldAccessError("Canonical Operational Twin config is missing.")
        paths = dict(operational_twin.get("manifests", {}))
        manifests: dict[str, Mapping[str, object]] = {}
        identities: dict[str, Mapping[str, object]] = {}
        for role in OPEN_ROLES:
            payload, identity = load_tracked_open_manifest(root, str(paths.get(role, "")), role)
            manifests[role] = payload
            identities[role] = identity
        receipt_path = str(dict(operational_twin.get("references", {})).get("open_action_package_receipt", ""))
        receipt, receipt_identity = load_open_package_receipt(root, receipt_path)
        return build_open_data_qualification_audit(
            config,
            manifests,
            identities,
            receipt,
            receipt_identity,
            authoritative_date=authoritative_date,
            repository_root=root,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed authority must retain the exact blocker.
        blocker = {"code": "qualification_input_schema_or_access_error", "reason": str(exc)}
        return {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "status": "BLOCKED",
            "terminal_decision": _terminal_for_blocker(blocker["code"]),
            "authoritative_date": authoritative_date,
            "evidence_class": "controlled_world_non_empirical",
            "authority_role": "independent_open_data_qualification",
            "training_authorized": False,
            "sufficient_training": False,
            "twin_qualification_pass": False,
            "formal_qualification": False,
            "input_identities": {},
            "input_content_identities": {},
            "construction_authority_identity": {},
            "construction_receipt_uid": None,
            "construction_receipt_content_identity": None,
            "precedence": {
                "supersedes_receipt_training_authorized_only_for_pretraining_authorization": False,
                "receipt_open_data_qualification": None,
                "receipt_training_authorized": None,
            },
            "schema_validation": {},
            "family_action_pairing_summary": {},
            "role_lineage_isolation": {
                "checked_fields": list(ROLE_ISOLATION_FIELDS),
                "action_result_identity_field": ACTION_RESULT_IDENTITY_FIELD,
                "status": "BLOCKED",
                "overlaps": [],
            },
            "deterministic_replay_provenance": {"status": "BLOCKED"},
            "retained_failure_inventory": {"policy": "retain_every_failure", "blocker_count": 1, "blockers": [blocker]},
            "forbidden_access_counters": dict(ZERO_ACCESS_FLAGS),
            "sealed_material_access": {
                "complete_test_manifest": 0,
                "twin_qualification_partition": 0,
                "unopened_access_receipt_content": 0,
                "protected_holdout": 0,
                "six_context_outcomes": 0,
            },
            "closed_downstream_gates": dict(QUALIFICATION_CLOSED_FLAGS),
        }
