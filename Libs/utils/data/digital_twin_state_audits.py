"""Generation and validation for non-materializing Round-11 state-contract audits."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from Libs.utils.data.digital_twin_canonical_state import (
    ELEMENT_MASK_FIELDS,
    SHA256_RE,
    STATE_BLOCK_FIELDS,
    validate_canonical_state_config,
)
from Libs.utils.data.raw_data_registry import write_json


AUDIT_SCHEMA_VERSION = "dt_round11_state_audit_v1"
AUDIT_STATUS = "in-progress"
ALLOWED_REUSE_STATUSES = {"reuse_as_is", "extend_existing", "migrate", "deprecated", "incompatible"}
ALLOWED_COMPONENT_STATUSES = {"implemented", "blocked"}
ALLOWED_ROUND_EXIT_STATUSES = {"round11_blocked_full_suite"}
AUDIT_KEYS = (
    "state_path_inventory",
    "state_reuse_map",
    "state_block_validation",
    "mask_semantics",
    "legacy_aggregate",
    "full_suite_failure_triage",
    "execution",
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common(kind: str) -> dict[str, object]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_kind": kind,
        "round_id": "round-11-dt-partial-state",
        "state": AUDIT_STATUS,
        "evidence_fabric_sealed": False,
        "dt_core": "not_evaluated",
        "evidence_materialized": False,
        "dt-evidence-fabric-state": "in-progress",
        "typed-state-blocks": "implemented",
        "partial-state-masks": "implemented",
        "evidence-fabric-sealed": False,
        "DT_CORE": "not_evaluated",
        "round_exit_status": "round11_blocked_full_suite",
        "provenance": {
            "generator": "Libs/scripts/builders/build_dt_round11_state_audits.py",
            "contract_module": "Libs/utils/data/digital_twin_canonical_state.py",
            "scope": "logical contract and audit only",
        },
    }


def _path_entry(component: str, path: str, status: str, reason: str) -> dict[str, str]:
    return {"component": component, "path": path, "reuse_status": status, "reason": reason}


def build_round11_audits(config: Mapping[str, object], *, repository_root: Path) -> dict[str, dict[str, object]]:
    blockers = validate_canonical_state_config(config)
    inventory = _common("state_path_inventory")
    inventory["entries"] = [
        _path_entry("canonical_observation_loader", "Libs/utils/data/digital_twin_evidence_fabric.py", "extend_existing", "Evidence records retain state_uid links; they do not embed state values."),
        _path_entry("foundation_loader", "Libs/utils/data/platform_contracts.py", "reuse_as_is", "Existing registry contract remains source boundary."),
        _path_entry("trajectory_initial_state", "Libs/models/modules/field_rollout.py", "migrate", "Tensor initial-state input needs an explicit state-block adapter before use."),
        _path_entry("posterior_builder", "Libs/models/networks/screening_system.py", "migrate", "Posterior consumes existing tensors; no state-block assimilation is implemented in Round-11."),
        _path_entry("world_action_input", "Libs/models/modules/neuronsifter_world_action.py", "migrate", "World-action input has no direct block reader and is not changed here."),
        _path_entry("reporting", "Libs/scripts/reports/build_paper_results.py", "reuse_as_is", "Reports remain read-only and receive no materialized state blocks."),
        _path_entry("schema", "Libs/configs/schema/schemas.yaml", "extend_existing", "Canonical state block v2 fields are owned by the existing schema registry."),
        _path_entry("serialization_helper", "Libs/utils/data/digital_twin_canonical_state.py", "extend_existing", "Deterministic logical serialization and hash validation are defined here."),
    ]
    inventory["config_blockers"] = blockers

    reuse_map = _common("state_reuse_map")
    reuse_map["mappings"] = [
        {"producer": "digital_twin_evidence_fabric", "consumer": "DigitalTwinStateBlock.evidence_uid/state_uid", "reuse_status": "extend_existing", "materialized": False},
        {"producer": "dynamics_ready_registry", "consumer": "axis_metadata.time/species", "reuse_status": "reuse_as_is", "materialized": False},
        {"producer": "spatial_atlas_registry", "consumer": "axis_metadata.region", "reuse_status": "reuse_as_is", "materialized": False},
        {"producer": "pkpd_ready_registry", "consumer": "axis_metadata.compartment", "reuse_status": "reuse_as_is", "materialized": False},
        {"producer": "serving_registry", "consumer": "axis_metadata.modality/feature_namespace", "reuse_status": "reuse_as_is", "materialized": False},
        {"producer": "DigitalTwinStateBlock", "consumer": "trajectory/posterior/world-action runtime", "reuse_status": "migrate", "materialized": False},
    ]

    block_validation = _common("state_block_validation")
    block_validation.update({
        "official_type": "DigitalTwinStateBlock",
        "required_field_count": len(STATE_BLOCK_FIELDS),
        "required_fields": list(STATE_BLOCK_FIELDS),
        "aggregation_fields": ["aggregation_level", "aggregation_method", "source_record_count", "independent_unit_count", "aggregation_weights", "aggregation_provenance", "headline_eligible"],
        "eligibility_fields": ["eligible_for_state_reconstruction", "eligible_for_temporal_dynamics", "eligible_for_action_endpoint", "eligible_for_uncertainty_calibration", "eligible_for_constraint_evaluation", "eligible_for_what_if_initialization"],
        "config_validation": {"passed": not blockers, "blockers": blockers},
        "storage_formats": list(config["digital_twin_canonical_state"]["value_storage_formats"]),
        "time_semantics": list(config["digital_twin_canonical_state"]["supported_time_semantics"]),
        "time_semantics_policy": "absolute_time/relative_to_action/simulator_time require explicit hours axis; pseudotime remains source-only; cross_sectional/ordered_endpoint/unknown/not_applicable remain non-numeric.",
        "validation_policy": "fail_closed_no_materialization",
    })

    mask_audit = _common("mask_semantics")
    mask_audit.update({
        "mask_states": list(ELEMENT_MASK_FIELDS),
        "one_hot_requirement": "sum(mask_state[element]) == 1 for every logical element",
        "zero_semantics": "numeric zero with observed=true is an observation; zero never encodes missingness",
        "nan_semantics": "NaN and infinity are rejected as ambiguous observations",
        "partial_axis_requirement": ["axis_status", "axis_present", "axis_values", "axis_mapping_version", "axis_uncertainty", "axis_mask"],
    })

    lineage_path = repository_root / "Output/audits/trajectory_numeric_lineage_audit.json"
    legacy = _common("legacy_aggregate")
    legacy.update({
        "aggregate_record": {
            "aggregate_uid": "legacy_trajectory_numeric_lineage_aggregate",
            "evidence_type": "legacy_simulator_trajectory",
            "aggregation_level": "unknown",
            "aggregation_method": "legacy_aggregate_unknown",
            "source_record_count": 222336,
            "aggregation_weights": None,
            "aggregation_provenance": {"status": "unavailable", "source_record_uids": None, "source_artifact_sha256": None, "method_version": None, "reason": "legacy source unit identities unavailable"},
            "declared_legacy_trajectory_row_count": 222336,
            "biological_unit_uid": None,
            "independent_unit_count": None,
            "headline_eligible": False,
            "allowed_train_roles": [],
            "claim_eligibility": "ineligible",
        },
        "source": {
            "relative_path": "Output/audits/trajectory_numeric_lineage_audit.json",
            "sha256": _sha256(lineage_path),
            "read_mode": "metadata_hash_only",
        },
        "validation": "reject independent-unit accounting or headline promotion",
    })

    full_suite = _common("full_suite_failure_triage")
    full_suite.update({
        "classification": "unresolved",
        "round_exit_status": "round11_blocked_full_suite",
        "job": {"job_id": "490518", "job_name": "NrnSift_dt_full_tests", "node": "cpu04"},
        "result": {"passed": 269, "failed": 17, "errors": 16, "duration_seconds": 1327.27},
        "failure_scope": {"status": "unavailable", "value": None, "reason": "Preserved 490518 logs were not found; scope is not inferred from prose."},
        "failing_test_ids": {"status": "unavailable", "values": [], "reason": "Preserved 490518 logs were not found."},
        "first_known_commit": {"status": "unavailable", "value": None, "reason": "Preserved 490518 logs were not found."},
        "owner": {"status": "unavailable", "value": None, "reason": "Preserved 490518 logs were not found."},
        "preserved_log": {"status": "unavailable", "path": None, "reason": "No preserved 490518 stdout/stderr artifact is available."},
        "round11_impact": "focused Round-11 validation may proceed; this result is not a passing full-suite validation.",
        "dt1_sealing_impact": "A final full-suite pass or formal exception remains mandatory before evidence-fabric sealing or DT_CORE evaluation.",
        "source_evidence": "orchestrator/rounds/round-10-dt-evidence-fabric-state/implementation-notes.md",
    })

    execution = _common("execution")
    execution.update({
        "execution_status": "contract_implementation_in_progress",
        "full_suite": {"job_id": "490518", "status": "recorded_failure_not_rerun"},
        "frozen_inputs": {"missing_gap_count": 20, "evidence_fabric_sealed": False},
        "prohibited_actions_not_run": ["training", "clean_trajectory_generation", "what_if_scenarios", "dt_core_gate", "evidence_fabric_sealing"],
        "audit_hashes": {},
    })
    return {
        "state_path_inventory": inventory,
        "state_reuse_map": reuse_map,
        "state_block_validation": block_validation,
        "mask_semantics": mask_audit,
        "legacy_aggregate": legacy,
        "full_suite_failure_triage": full_suite,
        "execution": execution,
    }


def _contains_private_path(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_private_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_private_path(item) for item in value)
    if isinstance(value, str) and (value.startswith("/") or value.startswith("~") or ".." in PurePosixPath(value).parts):
        return True
    return False


_COMMON_REQUIRED_FIELDS = {
    "schema_version", "audit_kind", "round_id", "state", "evidence_materialized", "provenance",
    "dt-evidence-fabric-state", "typed-state-blocks", "partial-state-masks",
    "evidence-fabric-sealed", "DT_CORE", "round_exit_status",
}
_AUDIT_REQUIRED_FIELDS = {
    "state_path_inventory": _COMMON_REQUIRED_FIELDS | {"entries", "config_blockers"},
    "state_reuse_map": _COMMON_REQUIRED_FIELDS | {"mappings"},
    "state_block_validation": _COMMON_REQUIRED_FIELDS | {"official_type", "required_field_count", "required_fields", "aggregation_fields", "eligibility_fields", "config_validation", "storage_formats", "time_semantics", "time_semantics_policy", "validation_policy"},
    "mask_semantics": _COMMON_REQUIRED_FIELDS | {"mask_states", "one_hot_requirement", "zero_semantics", "nan_semantics", "partial_axis_requirement"},
    "legacy_aggregate": _COMMON_REQUIRED_FIELDS | {"aggregate_record", "source", "validation"},
    "full_suite_failure_triage": _COMMON_REQUIRED_FIELDS | {"classification", "round_exit_status", "job", "result", "failure_scope", "failing_test_ids", "first_known_commit", "owner", "preserved_log", "round11_impact", "dt1_sealing_impact", "source_evidence"},
    "execution": _COMMON_REQUIRED_FIELDS | {"execution_status", "full_suite", "frozen_inputs", "prohibited_actions_not_run", "audit_hashes"},
}


def validate_round11_audits(config: Mapping[str, object], *, repository_root: Path) -> dict[str, object]:
    paths = config["digital_twin_canonical_state"]["audit_output_paths"]
    blockers: list[dict[str, object]] = []
    loaded: dict[str, dict[str, object]] = {}
    for key in AUDIT_KEYS:
        raw_path = str(paths[key])
        path = repository_root / raw_path
        if not path.is_file():
            blockers.append({"code": "missing_audit", "audit": key})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            blockers.append({"code": "unreadable_audit", "audit": key, "message": str(error)})
            continue
        if not isinstance(payload, dict):
            blockers.append({"code": "invalid_audit_payload", "audit": key})
            continue
        loaded[key] = payload
        missing_fields = sorted(_AUDIT_REQUIRED_FIELDS[key] - set(payload))
        if missing_fields:
            blockers.append({"code": "missing_audit_fields", "audit": key, "fields": missing_fields})
        for field, expected in (("schema_version", AUDIT_SCHEMA_VERSION), ("state", AUDIT_STATUS), ("evidence_materialized", False), ("dt-evidence-fabric-state", "in-progress"), ("evidence-fabric-sealed", False), ("DT_CORE", "not_evaluated"), ("round_exit_status", "round11_blocked_full_suite")):
            if payload.get(field) != expected:
                blockers.append({"code": "invalid_audit_common_field", "audit": key, "field": field})
        if payload.get("typed-state-blocks") not in ALLOWED_COMPONENT_STATUSES:
            blockers.append({"code": "invalid_typed_state_status", "audit": key})
        if payload.get("partial-state-masks") not in ALLOWED_COMPONENT_STATUSES:
            blockers.append({"code": "invalid_partial_mask_status", "audit": key})
        if payload.get("round_exit_status") not in ALLOWED_ROUND_EXIT_STATUSES:
            blockers.append({"code": "invalid_round_exit_status", "audit": key})
        if _contains_private_path(payload):
            blockers.append({"code": "private_absolute_path_in_audit", "audit": key})
    if "state_path_inventory" in loaded:
        for row in loaded["state_path_inventory"].get("entries", []):
            if row.get("reuse_status") not in ALLOWED_REUSE_STATUSES:
                blockers.append({"code": "invalid_reuse_status", "audit": "state_path_inventory"})
    if "legacy_aggregate" in loaded:
        aggregate = loaded["legacy_aggregate"].get("aggregate_record", {})
        if aggregate.get("aggregation_method") != "legacy_aggregate_unknown" or aggregate.get("independent_unit_count") is not None or aggregate.get("headline_eligible") is not False or aggregate.get("biological_unit_uid") is not None:
            blockers.append({"code": "legacy_aggregate_promoted", "audit": "legacy_aggregate"})
    if "full_suite_failure_triage" in loaded:
        triage = loaded["full_suite_failure_triage"]
        if triage.get("classification") != "unresolved" or triage.get("round_exit_status") != "round11_blocked_full_suite":
            blockers.append({"code": "invalid_full_suite_classification", "audit": "full_suite_failure_triage"})
        for field in ("failure_scope", "failing_test_ids", "first_known_commit", "owner", "preserved_log"):
            value = triage.get(field)
            if not isinstance(value, dict) or value.get("status") != "unavailable":
                blockers.append({"code": "unavailable_triage_field_missing", "audit": "full_suite_failure_triage", "field": field})
    if "execution" in loaded:
        execution = loaded["execution"]
        hashes = execution.get("audit_hashes")
        expected_keys = set(AUDIT_KEYS) - {"execution"}
        if not isinstance(hashes, dict) or set(hashes) != expected_keys:
            blockers.append({"code": "audit_hash_set_mismatch", "audit": "execution"})
        else:
            for key in sorted(expected_keys):
                expected_digest = _sha256(repository_root / str(paths[key]))
                if hashes.get(key) != expected_digest or not isinstance(hashes.get(key), str) or not SHA256_RE.fullmatch(hashes[key]):
                    blockers.append({"code": "audit_hash_mismatch", "audit": key})
    return {"schema_version": AUDIT_SCHEMA_VERSION, "status": "passed" if not blockers else "failed", "blocker_count": len(blockers), "blockers": blockers}


def write_round11_audits(config: Mapping[str, object], audits: Mapping[str, Mapping[str, object]], *, repository_root: Path) -> None:
    paths = config["digital_twin_canonical_state"]["audit_output_paths"]
    hashes: dict[str, str] = {}
    for key, payload in audits.items():
        path = repository_root / str(paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        if key != "execution":
            hashes[key] = _sha256(path) or ""
    execution_path = repository_root / str(paths["execution"])
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["audit_hashes"] = hashes
    write_json(execution_path, execution)
