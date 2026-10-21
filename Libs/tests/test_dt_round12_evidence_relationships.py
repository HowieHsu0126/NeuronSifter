from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from Libs.scripts.builders.build_dt_round12_evidence_relationships import SCHEMA_PATHS, build_pilot_graph
from Libs.utils.data.evidence_relationships import (
    IDENTITY_AXES,
    NODE_TYPES,
    PAIRING_REQUIRED_AXES,
    RELATION_TYPES,
    SUPPORT_DOMAIN_FIELDS,
    derive_eligibility,
    frozen_gap_rows,
    validate_frozen_gap_preservation,
    validate_graph,
)
from Libs.utils.data.evidence_union import build_evidence_union
from Libs.utils.data.raw_data_registry import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _asset(uid: str, role: str, digest: str) -> dict[str, object]:
    return {
        "asset_uid": uid,
        "logical_role": role,
        "artifact_type": "generated_tsv",
        "source_uid": f"source-{uid}",
        "producer_uid": f"producer-{uid}",
        "version": "test-v1",
        "content_hash": digest,
        "size": 1,
        "canonical_uri": f"repo://Input/processed/{role}/catalog.tsv",
        "license_or_access_status": "approved_internal_read_only",
        "required_by_testcases": ["round12_contract_validation"],
        "allowed_execution_role": "processed_test_input",
    }


@pytest.fixture()
def graph() -> dict[str, object]:
    bundle = {
        "schema_version": "dt-round11g-execution-input-bundle-v1",
        "code_commit": "f" * 40,
        "validation_status": "passed",
        "missing_entries": [],
        "processed_input_entries": [
            _asset("asset-action", "action_ready", "1" * 64),
            _asset("asset-simulation", "simulation_audit", "2" * 64),
            _asset("asset-observation", "model_input", "3" * 64),
        ],
        "external_asset_entries": [],
    }
    return build_pilot_graph(bundle, bundle_sha256="4" * 64)


def _codes(graph: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in validate_graph(graph)["blockers"]}


def _relation(graph: dict[str, object], relation_type: str) -> dict[str, object]:
    return next(row for row in graph["relations"] if row["relation_type"] == relation_type)


def _pairing(graph: dict[str, object], decision: str) -> dict[str, object]:
    return next(row for row in graph["pairing_decisions"] if row["decision"] == decision)


def test_pilot_graph_validates_and_preserves_contract_boundary(graph: dict[str, object]) -> None:
    report = validate_graph(graph)
    assert report["status"] == "passed"
    assert report["blocker_count"] == 0
    assert graph["contract_scope"] == "round12_contract_validation_pilot_only"
    assert graph["evidence_fabric_sealed"] is False
    assert graph["dt_core"] == "not_evaluated"
    assert {row["node_type"] for row in graph["nodes"]}.issubset(NODE_TYPES)
    assert {row["relation_type"] for row in graph["relations"]}.issubset(RELATION_TYPES)


def test_all_round12_schemas_are_valid_and_registered(graph: dict[str, object]) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    registry = config["schemas"]["evidence_relationship_schema_registry"]
    assert set(registry) == {
        "evidence_node", "evidence_relation", "evidence_pairing_decision", "evidence_support_domain",
        "evidence_eligibility", "evidence_conflict", "evidence_relationship_graph",
    }
    groups = {
        "node": graph["nodes"], "relation": graph["relations"], "pairing": graph["pairing_decisions"],
        "support_domain": graph["support_domains"], "eligibility": graph["eligibilities"],
        "conflict": graph["conflicts"], "graph": [graph],
    }
    for family, rows in groups.items():
        schema = json.loads((REPO_ROOT / SCHEMA_PATHS[family]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert [error.message for row in rows for error in Draft202012Validator(schema).iter_errors(row)] == []


def test_frozen_twenty_gap_map_is_exact_and_non_materializing() -> None:
    source = json.loads((REPO_ROOT / "Output/audits/dt_evidence_fabric_validation.json").read_text(encoding="utf-8"))
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    adapters = config["digital_twin_evidence_fabric"]["source_adapters"]
    rows = frozen_gap_rows(source, adapters)
    assert len(rows) == 20
    assert validate_frozen_gap_preservation(source, rows, adapters) == []
    assert all(row["status"] in {"materialization_required", "source_required"} for row in rows)
    assert all("no pairing inferred" in row["blocked_pairing_reason"] for row in rows)


def test_frozen_gap_mapping_uses_adapter_contract_not_adapter_name() -> None:
    source = {
        "blockers": [
            {
                "code": "missing_required_frozen_input",
                "adapter_id": "simulation_named_but_explicit_prior",
                "path": "Output/audits/missing.json",
            }
        ]
    }
    adapters = [
        {
            "adapter_id": "simulation_named_but_explicit_prior",
            "path": "Output/audits/missing.json",
            "reader_type": "json_artifact",
            "evidence_defaults": {
                "evidence_type": "ontology_or_prior",
                "pairing_semantics": "not_paired",
                "time_semantics": "timeless",
            },
        }
    ]
    [row] = frozen_gap_rows(source, adapters)
    assert row["missing_node_type"] == "claim"
    assert row["missing_relationship_type"] == "derived_from"
    assert row["status"] == "source_required"


def test_cross_sectional_samples_cannot_be_longitudinal(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    relation = _relation(candidate, "cross_sectional_state_observation")
    relation["relation_type"] = "same_unit_longitudinal_observation"
    assert "relation_pairing_mismatch" in _codes(candidate)


def test_different_biological_units_cannot_be_same_unit(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    pairing = _pairing(candidate, "valid_longitudinal_pair")
    pairing["identity_axis_results"]["biological_unit_uid"] = "different_and_blocking"
    codes = _codes(candidate)
    assert "different_biological_unit_pairing" in codes
    assert "missing_required_identity_axis" in codes


def test_static_endpoint_cannot_be_trajectory(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    relation = _relation(candidate, "simulator_generated_trajectory")
    relation["target_node_uid"] = "node_endpoint"
    assert "static_endpoint_treated_as_trajectory" in _codes(candidate)


def test_simulator_output_cannot_be_experimental_truth(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    node = next(row for row in candidate["nodes"] if row["node_uid"] == "node_simulator_output")
    node["evidence_class"] = "experimental_observation"
    assert "implicit_node_type_conversion" in _codes(candidate)


def test_semantic_alignment_cannot_be_action_response(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    pairing = _pairing(candidate, "semantic_alignment_only")
    pairing["decision"] = "valid_action_endpoint_pair"
    pairing["required_identity_axes"] = sorted(PAIRING_REQUIRED_AXES["valid_action_endpoint_pair"])
    assert "relation_pairing_mismatch" in _codes(candidate)


def test_hypothetical_scenario_cannot_be_empirical_evidence(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    node = next(row for row in candidate["nodes"] if row["node_uid"] == "node_hypothetical")
    node["evidence_class"] = "experimental_observation"
    codes = _codes(candidate)
    assert "implicit_node_type_conversion" in codes
    assert "hypothetical_treated_as_evidence" in codes


def test_same_publication_is_insufficient_pairing_identity(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    pairing = _pairing(candidate, "valid_longitudinal_pair")
    pairing["identity_axis_results"] = {axis: "not_applicable" for axis in IDENTITY_AXES}
    pairing["identity_axis_results"]["publication_uid"] = "exact_match"
    assert "missing_required_identity_axis" in _codes(candidate)


def test_unknown_provenance_cannot_enter_strong_training(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    relation = next(
        row for row in candidate["relations"]
        if row["source_node_uid"] == "node_unknown_observation" and row["relation_type"] == "supports_claim"
    )
    relation["status"] = "active"
    assert "unknown_provenance_strong_relation" in _codes(candidate)


def test_unsupported_support_domain_cannot_generalize(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    relation = _relation(candidate, "hypothetical_action_scenario")
    eligibility = next(row for row in candidate["eligibilities"] if row["eligibility_uid"] == relation["eligibility_ref"])
    eligibility["eligible_for_claim_support"] = True
    assert "eligibility_upgrade" in _codes(candidate)


def test_unknown_support_scope_cannot_be_declared_in_domain(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    candidate["support_domains"][0]["organism_scope"] = ["unknown"]
    assert "unknown_in_domain_support_scope" in _codes(candidate)


def test_grade_c_simulator_trajectory_is_calibration_only(graph: dict[str, object]) -> None:
    relation = _relation(graph, "simulator_generated_trajectory")
    domain = next(row for row in graph["support_domains"] if row["support_domain_uid"] == relation["support_domain_ref"])
    domain = {**domain, "generalization_status": "in_domain", **{field: ["explicit"] for field in SUPPORT_DOMAIN_FIELDS}}
    provenance = next(row for row in graph["provenances"] if row["provenance_uid"] == relation["provenance_ref"])
    pairing = _pairing(graph, "valid_simulator_trajectory_pair")
    eligibility = derive_eligibility(
        subject_uid="grade_c_simulator",
        node={"node_type": "trajectory", "evidence_class": "simulator_trajectory", "qc_grade": "C"},
        relation={**relation, "status": "active", "context_relation": "exact"},
        pairing=pairing,
        domain=domain,
        provenance=provenance,
        conflicts=[],
    )
    assert eligibility["eligible_for_trajectory_training"] is False
    assert eligibility["eligible_for_calibration"] is True


def test_relation_fields_cannot_contradict_typed_pairing(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    relation = _relation(candidate, "same_unit_longitudinal_observation")
    relation["temporal_relation"] = "cross_sectional"
    assert "longitudinal_relation_semantics_mismatch" in _codes(candidate)


def test_evidence_union_cannot_upgrade_eligibility(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    candidate["evidence_unions"][0]["canonical_downstream_eligibility"]["eligible_for_trajectory_training"] = True
    assert "evidence_union_eligibility_upgrade" in _codes(candidate)


def test_evidence_union_cannot_upgrade_pairing(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    candidate["evidence_unions"][0]["canonical_downstream_eligibility"]["pairing_decision"] = "valid_longitudinal_pair"
    assert "evidence_union_pairing_upgrade" in _codes(candidate)


def test_conflict_cannot_disappear_during_union(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    candidate["evidence_unions"][0]["preserved_typed_conflicts"] = []
    assert "evidence_union_conflict_loss" in _codes(candidate)


def test_conflicting_payloads_for_one_conflict_uid_block_union() -> None:
    base = {
        "producer_uid": "test",
        "schema_version": "test-v1",
        "lineage_group": "same",
        "provenance_status": "verified",
        "evidence_type": "real",
        "evidence_class": "experimental_endpoint",
        "qc_grade": "A",
        "train_eval_role": "evaluation_only",
        "claim_eligibility": "eligible",
        "independence_group": "same",
        "status": "accepted",
        "semantic_fingerprint": "same",
        "pairing_decision": "valid_action_endpoint_pair",
        "relationship_eligibility": {role: False for role in (
            "eligible_for_state_pretraining", "eligible_for_action_endpoint_training", "eligible_for_trajectory_training",
            "eligible_for_calibration", "eligible_for_validation", "eligible_for_stress_test",
            "eligible_for_claim_support", "eligible_for_hypothesis_generation",
        )},
    }
    union = build_evidence_union(
        union_uid="conflict_collision",
        members=[
            {**base, "artifact_uid": "left", "artifact_hash": "1" * 64, "typed_conflicts": [{"conflict_uid": "same_uid", "state": "unresolved"}]},
            {**base, "artifact_uid": "right", "artifact_hash": "2" * 64, "typed_conflicts": [{"conflict_uid": "same_uid", "state": "invalidated"}]},
        ],
        canonical_output_ref="test",
    )
    assert union["resolution_status"] == "blocked"
    assert any(row["lineage_group"] == "typed_conflict_collision" for row in union["conflicts"])


def test_lineage_cycle_is_rejected(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    template = deepcopy(candidate["relations"][0])
    template.update({"relation_uid": "lineage_a_to_b", "relation_type": "derived_from", "source_node_uid": "node_study", "target_node_uid": "node_accession", "status": "active"})
    reverse = deepcopy(template)
    reverse.update({"relation_uid": "lineage_b_to_a", "source_node_uid": "node_accession", "target_node_uid": "node_study"})
    candidate["relations"].extend([template, reverse])
    assert "lineage_cycle" in _codes(candidate)


def test_duplicate_active_successor_is_rejected(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    template = deepcopy(candidate["relations"][0])
    first = {**template, "relation_uid": "successor_one", "relation_type": "version_successor", "source_node_uid": "node_study", "target_node_uid": "node_accession", "status": "active"}
    second = {**template, "relation_uid": "successor_two", "relation_type": "version_successor", "source_node_uid": "node_publication", "target_node_uid": "node_accession", "status": "active"}
    candidate["relations"].extend([first, second])
    assert "duplicate_active_successor" in _codes(candidate)


def test_unknown_relationship_type_is_rejected(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    candidate["relations"][0]["relation_type"] = "same_filename_implies_pair"
    assert "unknown_relationship_type" in _codes(candidate)


def test_missing_required_identity_axis_is_rejected(graph: dict[str, object]) -> None:
    candidate = deepcopy(graph)
    pairing = _pairing(candidate, "valid_action_endpoint_pair")
    del pairing["identity_axis_results"]["dose"]
    codes = _codes(candidate)
    assert "identity_axis_inventory_mismatch" in codes
    assert "missing_required_identity_axis" in codes
