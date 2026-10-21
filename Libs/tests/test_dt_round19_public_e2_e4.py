"""Focused fail-closed tests for Round-19 public E2/E4 readiness."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import yaml

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.round19_contracts import (
    audit_independence,
    validate_control_pair,
    validate_e2_endpoint,
    validate_e4_action,
    validate_e4_producer,
    validate_e4_trajectory,
    validate_frozen_gaps,
    validate_initial_state_claim,
    validate_intervention_effect_evidence,
    validate_public_boundary,
    validate_round19_invariants,
    validate_training_eligibility,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

MANDATORY_NEGATIVE_TEST_INVENTORY = {
    "controlled data": "test_controlled_data_rejected",
    "unknown license": "test_unknown_license_rejected",
    "abstract narrative converted to numeric endpoint": "test_abstract_numeric_endpoint_rejected",
    "duplicate studies counted as independent": "test_duplicate_study_is_one_independent_unit",
    "missing comparator": "test_missing_comparator_and_timepoint_rejected",
    "missing timepoint": "test_missing_comparator_and_timepoint_rejected",
    "missing dose unit": "test_missing_dose_unit_rejected",
    "unsupported route or schedule": "test_unsupported_route_and_schedule_rejected",
    "unknown parameter provenance": "test_unknown_parameter_provenance_rejected",
    "synthetic initial state treated as biological validation": "test_synthetic_initial_state_validation_rejected",
    "intervention trajectory without matched control used as effect evidence": "test_unpaired_intervention_effect_rejected",
    "legacy E3 promoted directly to E4": "test_legacy_e3_promotion_rejected",
    "simulator output treated as empirical observation": "test_simulator_empirical_promotion_rejected",
    "validation evidence included in training": "test_validation_evidence_in_training_rejected",
    "gap closed by prediction": "test_gap_closed_by_prediction_rejected",
    "final held-out outcome accessed": "test_final_held_out_access_rejected",
    "training started": "test_training_started_rejected",
    "Evidence Fabric sealed": "test_evidence_fabric_sealed_rejected",
    "DT_CORE evaluated": "test_dt_core_evaluated_rejected",
}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _endpoint() -> dict[str, object]:
    config = load_config(REPO_ROOT / "Libs/configs/data/round19_public_e2_candidates.yaml")
    source = deepcopy(config["round19_public_e2_candidates"]["candidates"][0])
    source["content_hash"] = source.pop("fulltext_xml_sha256")
    source.update(
        endpoint_uid="endpoint",
        schema_version="neuronsifter-round19-e2-endpoint-v1",
        source_is_public=True,
        publicly_available=True,
        strict_public_only_flag=True,
        controlled_access_required=False,
        controlled_source_resolution_count=0,
        empirical_observation=True,
    )
    return source


def _action() -> dict[str, object]:
    return {
        "action_class": "pharmacological",
        "agent": "3-bromopyruvate",
        "target_set": ["HK2"],
        "dose": 5.0,
        "dose_unit": "mg/kg",
        "route": "intraperitoneal",
        "schedule": "twice daily",
        "duration": "4 days",
        "exposure_occupancy_semantics": "not quantified; qualification blocked",
        "context": "mouse post-stroke",
        "route_supported": True,
        "schedule_supported": True,
    }


def _trajectory() -> dict[str, object]:
    return {
        "producer_hash": _hash("producer"),
        "initial_state_hash": _hash("state"),
        "action_hash": _hash("action"),
        "state_artifact_hash": _hash("artifact"),
        "native_time_grid_hash": _hash("native"),
        "canonical_time_grid_hash": _hash("canonical"),
        "parameter_hash": _hash("parameter"),
        "constraint_status": "passed",
        "numerical_status": "completed",
        "output_class": "model_trajectory",
        "empirical_observation": False,
        "admission_decision": "admitted_e4_restricted",
    }


def _producer() -> dict[str, object]:
    return {
        "producer_uid": "producer",
        "producer_version": "v1",
        "producer_hash": _hash("producer"),
        "equation_model_identity": "equation-v1",
        "parameter_provenance": "public parameters",
        "supported_state_variables": ["state"],
        "supported_action_semantics": ["pharmacological"],
        "initial_state_contract": "explicit",
        "control_generation_capability": "matched",
        "seed_policy": "fixed",
        "fidelity_policy": "restricted",
        "numerical_qc": "passed",
        "constraint_policy": "bounded",
        "license_or_redistribution": "permitted",
        "decision": "qualified_with_restrictions",
        "support_domain": "domain",
    }


def test_round19_mandatory_negative_inventory_is_complete() -> None:
    assert len(MANDATORY_NEGATIVE_TEST_INVENTORY) == 19
    assert all(callable(globals().get(name)) for name in MANDATORY_NEGATIVE_TEST_INVENTORY.values())


def test_frozen_contract_thresholds_and_authority_are_registered() -> None:
    contract = load_config(REPO_ROOT / "Libs/configs/data/round19_public_e2_e4.yaml")["round19_public_e2_e4"]
    assert contract["scientific_contract"]["created_before_discovery"] is True
    assert contract["training_readiness_contract"]["thresholds_mutable_after_source_observation"] is False
    authority = yaml.safe_load((REPO_ROOT / "Libs/configs/validation/round11f_controller_authority.yaml").read_text())
    approved = {row["validator_uid"]: row["stage_uids"] for row in authority["formal_gate"]["approved_validators"]}
    assert approved["round19_public_e2_e4_training_readiness_validator"] == ["round_19_public_e2_e4_training_readiness"]
    assert authority["formal_gate"]["writer_uid"] == "neuronsifter_orchestration_controller"


def test_controlled_data_rejected() -> None:
    row = _endpoint()
    contract = load_config(REPO_ROOT / "Libs/configs/data/round19_public_e2_e4.yaml")["round19_public_e2_e4"]
    forbidden = contract["public_source_policy"]["rejected_routes"]
    row["publication_or_accession_identity"] = f"{forbidden[0]} route"
    assert any(item["code"] == "controlled_source" for item in validate_public_boundary(row, forbidden_source_tokens=forbidden))


def test_unknown_license_rejected() -> None:
    row = _endpoint()
    row["license_or_permitted_use"] = "unknown"
    assert any(item["code"] == "unknown_license" for item in validate_public_boundary(row))


def test_abstract_numeric_endpoint_rejected() -> None:
    row = _endpoint()
    row["source_class"] = "public_abstract"
    row["endpoint_value"] = 1.0
    assert any(item["code"] == "abstract_numeric_endpoint" for item in validate_e2_endpoint(row))


def test_duplicate_study_is_one_independent_unit() -> None:
    groups = _endpoint()["independence_groups"]
    audit = audit_independence([groups, deepcopy(groups)])
    assert audit["independent_study_count"] == 1


def test_missing_comparator_and_timepoint_rejected() -> None:
    row = _endpoint()
    row["comparator"] = "unknown"
    row["timepoint"] = "unknown"
    codes = {item["field"] for item in validate_e2_endpoint(row) if item["code"] == "e2_missing_field"}
    assert {"comparator", "timepoint"} <= codes


def test_missing_dose_unit_rejected() -> None:
    action = _action()
    action["dose_unit"] = "unknown"
    assert any(item["code"] == "missing_dose_unit" for item in validate_e4_action(action))


def test_unsupported_route_and_schedule_rejected() -> None:
    action = _action()
    action["route_supported"] = False
    action["schedule_supported"] = False
    codes = {item["code"] for item in validate_e4_action(action)}
    assert {"unsupported_route", "unsupported_schedule"} <= codes


def test_unknown_parameter_provenance_rejected() -> None:
    producer = _producer()
    producer["parameter_provenance"] = "unknown"
    assert any(item["code"] == "producer_missing_field" for item in validate_e4_producer(producer))


def test_synthetic_initial_state_validation_rejected() -> None:
    blockers = validate_initial_state_claim({"origin": "synthetic", "biological_validation_eligible": True})
    assert blockers[0]["code"] == "synthetic_initial_state_validation"


def test_unpaired_intervention_effect_rejected() -> None:
    trajectory = {"intervention_effect_claim": True}
    assert validate_intervention_effect_evidence(trajectory, None)[0]["code"] == "missing_matched_control"
    pair = {"pair_uid": "p", "action_trajectory_uid": "a", "control_trajectory_uid": "c", "initial_state_family": "s", "producer_version": "v", "parameter_family": "p", "fidelity": "f", "time_grid": "t", "seed_policy": "matched", "pairing_decision": "unpaired_descriptive_rollout", "action_effect_claim_eligible": True}
    assert any(item["code"] == "unpaired_effect_claim" for item in validate_control_pair(pair))


def test_legacy_e3_promotion_rejected() -> None:
    trajectory = _trajectory()
    trajectory["source_evidence_type"] = "E3"
    assert any(item["code"] == "legacy_e3_promotion" for item in validate_e4_trajectory(trajectory, _action()))


def test_simulator_empirical_promotion_rejected() -> None:
    trajectory = _trajectory()
    trajectory["empirical_observation"] = True
    assert any(item["code"] == "model_output_class" for item in validate_e4_trajectory(trajectory, _action()))


def test_validation_evidence_in_training_rejected() -> None:
    row = {"artifact_uid": "a", "role": "validation", "partition": "qualification", "validation_only": False}
    assert any(item["code"] == "validation_in_training" for item in validate_training_eligibility([row]))


def test_gap_closed_by_prediction_rejected() -> None:
    rows = [{"gap_uid": f"g{i}", "status": "frozen_unresolved", "evidence_refs": [], "resolution_basis": ""} for i in range(20)]
    rows[0]["status"] = "resolved_by_prediction"
    assert any(item["code"] == "gap_fabrication" for item in validate_frozen_gaps(rows))


def _invariants() -> dict[str, object]:
    return {"strict_public_only_flag": True, "controlled_source_resolution_count": 0, "final_held_out_access_count": 0, "training_authorized": False, "training_started": False, "evidence_fabric_sealed": False, "frozen_gap_count": 20, "dt_core_status": "not_evaluated"}


def test_final_held_out_access_rejected() -> None:
    state = _invariants()
    state["final_held_out_access_count"] = 1
    assert any(item["code"] == "held_out_access" for item in validate_round19_invariants(state))


def test_training_started_rejected() -> None:
    state = _invariants()
    state["training_started"] = True
    assert any(item["code"] == "training_started" for item in validate_round19_invariants(state))


def test_evidence_fabric_sealed_rejected() -> None:
    state = _invariants()
    state["evidence_fabric_sealed"] = True
    assert any(item["code"] == "fabric_sealed" for item in validate_round19_invariants(state))


def test_dt_core_evaluated_rejected() -> None:
    state = _invariants()
    state["dt_core_status"] = "declared"
    assert any(item["code"] == "dt_core_evaluated" for item in validate_round19_invariants(state))
