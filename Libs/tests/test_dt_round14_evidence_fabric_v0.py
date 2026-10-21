from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator
import pytest

from Libs.scripts.builders.build_dt_round14_evidence_fabric_v0 import (
    ROUND14_VALIDATOR_UID,
    _pilot_blocks,
    write_round14_materialization,
)
from Libs.scripts.validators.validate_dt_round14_evidence_fabric_v0 import (
    MANDATORY_NEGATIVE_TEST_NAMES,
    _execution_identity_blockers,
    expected_execution_identity_fields,
)
from Libs.utils.data.evidence_fabric_v0 import (
    ADMISSION_PHASES,
    AdmissionInput,
    admit_evidence,
    canonical_sha256,
    canonicalize_entity_mapping,
    derive_eligibility,
    validate_admission_record,
    validate_conflict_preservation,
    validate_effective_sample_audit,
    validate_gap_resolution_evidence,
    validate_independence_grouping,
    validate_eligibility_non_upgrade,
    validate_release_manifest,
    validate_v0_config,
    read_parquet_records,
    write_parquet_records,
)
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp import orchestration_controller


REPO_ROOT = Path(__file__).resolve().parents[2]


def _input(**updates: object) -> AdmissionInput:
    value = AdmissionInput(
        layer="E1",
        item_type="real_static_observation_block",
        source_uid="official_source",
        official_source_or_internal_producer="official_public_source",
        source_version="v1",
        publication_or_accession="ACC-1",
        license_or_access_status="verified_public",
        retrieval_or_generation_date="2026-07-27",
        raw_hash="a" * 64,
        processed_hash="b" * 64,
        parser_version="parser-v1",
        canonicalization_version="canonical-v1",
        producer_uid="producer-v1",
        producer_command="python governed_builder.py",
        train_eval_role="state_pretraining",
        holdout_group="study-1",
        support_domain_ref="domain-1",
        qc_grade="B",
        schema_version="source-schema-v1",
        identity={"biological_unit_uid": "unit-1"},
        payload={"source_artifact": "Input/governed/source.json"},
        relationship_refs=[],
        conflict_refs=[],
        pairing_decision="valid_cross_sectional_comparison",
    )
    return replace(value, **updates)


def _codes(value: AdmissionInput) -> set[str]:
    return {str(row["code"]) for row in admit_evidence(value)["blockers"]}


def test_unknown_source_admitted_as_grade_a_rejected() -> None:
    record = admit_evidence(_input(source_uid="", qc_grade="A"))
    assert record["terminal_state"] == "rejected_provenance"
    assert "missing_source" in {row["code"] for row in record["blockers"]}


def test_missing_hash_admitted_rejected() -> None:
    record = admit_evidence(_input(raw_hash=""))
    assert record["terminal_state"] == "rejected_hash"
    assert "missing_hash" in {row["code"] for row in record["blockers"]}


def test_rejected_admission_schema_preserves_incomplete_provenance_and_hashes() -> None:
    schema = json.loads(
        (REPO_ROOT / "Libs/configs/schema/evidence_fabric_admission_record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for record in (
        admit_evidence(_input(source_version="")),
        admit_evidence(_input(raw_hash="", processed_hash="")),
    ):
        assert record["terminal_state"].startswith("rejected_")
        assert list(Draft202012Validator(schema).iter_errors(record)) == []


def test_admitted_admission_schema_requires_complete_provenance_and_hashes() -> None:
    schema = json.loads(
        (REPO_ROOT / "Libs/configs/schema/evidence_fabric_admission_record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    record = admit_evidence(_input())
    assert record["terminal_state"] == "admitted_weighted"
    record["source_version"] = ""
    record["raw_hash"] = ""
    errors = list(Draft202012Validator(schema).iter_errors(record))
    assert {tuple(error.absolute_path) for error in errors} == {
        ("raw_hash",),
        ("source_version",),
    }


def test_unresolved_license_admitted_rejected() -> None:
    record = admit_evidence(_input(license_or_access_status="unresolved"))
    assert record["terminal_state"] == "rejected_license"


def test_ambiguous_entity_mapping_silently_collapsed_rejected() -> None:
    _, blockers = canonicalize_entity_mapping(
        {
            "source_uid": "ambiguous-source",
            "canonical_uid": "silently-selected-id",
            "mapping_status": "one_to_many",
        }
    )
    assert {row["code"] for row in blockers} == {"ambiguous_mapping_collapsed"}


def test_cross_sectional_evidence_promoted_to_longitudinal_rejected() -> None:
    assert "cross_sectional_promoted_to_longitudinal" in _codes(
        _input(train_eval_role="longitudinal_transition")
    )


def test_static_endpoint_promoted_to_trajectory_rejected() -> None:
    assert "static_endpoint_promoted_to_trajectory" in _codes(
        _input(
            layer="E2",
            item_type="real_perturbation_endpoint_block",
            train_eval_role="trajectory_training",
            action_observation_valid=True,
            action_complete=True,
        )
    )


def test_legacy_incomplete_trajectory_promoted_to_e4_rejected() -> None:
    assert "legacy_trajectory_promoted_to_e4" in _codes(
        _input(
            layer="E4",
            item_type="legacy_simulator_trajectory",
            train_eval_role="trajectory_training",
            legacy_role="legacy_dynamics_pretraining",
            action_complete=True,
        )
    )


def test_simulator_output_promoted_to_experimental_truth_rejected_round14() -> None:
    assert "simulator_output_promoted_to_experimental_truth" in _codes(
        _input(
            layer="E3",
            item_type="legacy_simulator_trajectory",
            train_eval_role="numerical_dynamics_only",
            legacy_role="numerical_dynamics_only",
            empirical_truth=True,
        )
    )


def test_grade_c_numeric_ground_truth_rejected() -> None:
    assert "grade_c_numeric_ground_truth" in _codes(
        _input(qc_grade="C", train_eval_role="numeric_ground_truth")
    )


def test_grade_s_headline_training_rejected() -> None:
    assert "grade_s_headline_training" in _codes(
        _input(qc_grade="S", train_eval_role="headline_training")
    )


def test_hypothetical_scenario_admitted_as_observation_rejected_round14() -> None:
    assert "hypothetical_scenario_admitted_as_observation" in _codes(
        _input(layer="E7", item_type="canonical_observation", empirical_truth=True)
    )


def test_evidence_union_eligibility_upgrade_rejected_round14() -> None:
    predecessor = {
        "qc_grade": "C",
        "pairing_decision": "unsupported_pairing",
        "eligibility": {"eligible_for_trajectory_training": False},
    }
    successor = {
        "qc_grade": "B",
        "pairing_decision": "valid_simulator_trajectory_pair",
        "eligibility": {"eligible_for_trajectory_training": True},
    }
    codes = {row["code"] for row in validate_eligibility_non_upgrade(predecessor, successor)}
    assert codes == {
        "evidence_union_eligibility_upgrade",
        "evidence_union_pairing_creation",
        "evidence_union_qc_upgrade",
    }


def test_conflict_dropped_during_deduplication_rejected() -> None:
    conflict = {"conflict_uid": "conflict-1", "state": "unresolved"}
    assert {row["code"] for row in validate_conflict_preservation([conflict], [])} == {
        "conflict_dropped_during_deduplication"
    }


def test_gap_closed_without_admission_evidence_rejected() -> None:
    blockers = validate_gap_resolution_evidence(
        [
            {
                "gap_uid": "gap-1",
                "current_status": "resolved_by_existing_governed_evidence",
                "resolution_evidence": "missing-admission",
            }
        ],
        [],
    )
    assert {row["code"] for row in blockers} == {"gap_closed_without_admission_evidence"}


def test_row_count_used_as_effective_sample_count_rejected() -> None:
    assert {row["code"] for row in validate_effective_sample_audit({"row_count": 100})} == {
        "row_count_used_as_effective_sample_count"
    }


def test_duplicate_biological_units_split_as_independent_rejected() -> None:
    rows = [
        {"biological_unit_group": "unit-1", "independence_group": "split-a"},
        {"biological_unit_group": "unit-1", "independence_group": "split-b"},
    ]
    assert {row["code"] for row in validate_independence_grouping(rows)} == {
        "duplicate_biological_units_split_as_independent"
    }


def test_mandatory_negative_inventory_matches_test_module() -> None:
    declared = {
        name
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    }
    assert MANDATORY_NEGATIVE_TEST_NAMES.issubset(declared)


def test_round14_config_and_schemas_are_canonical() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert validate_v0_config(config) == []
    registry = config["schemas"]["round14_evidence_fabric_schema_registry"]
    assert set(registry) == {
        "evidence_fabric_admission_record",
        "evidence_fabric_release_manifest",
    }
    for entry in registry.values():
        schema = json.loads((REPO_ROOT / entry["path"]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema_version"]["const"] == entry["schema_version"]


def test_round14_validator_is_approved_by_canonical_gate_authority() -> None:
    authority = orchestration_controller.load_authority_contract(
        REPO_ROOT / "Libs/configs/validation/round11f_controller_authority.yaml"
    )
    assert orchestration_controller.authority_contract_report(authority)["status"] == "passed"
    assert orchestration_controller.validator_is_approved(
        authority,
        ROUND14_VALIDATOR_UID,
        "round_14_evidence_fabric_v0",
    )
    gate = orchestration_controller.build_canonical_gate_decision(
        authority,
        gate_uid="round_14_evidence_fabric_v0_test",
        stage_uid="round_14_evidence_fabric_v0",
        controller_uid=orchestration_controller.CANONICAL_CONTROLLER_UID,
        validator_uid=ROUND14_VALIDATOR_UID,
        validator_report={"validator_uid": ROUND14_VALIDATOR_UID, "status": "passed"},
        input_artifact_hashes={"fixture": "a" * 64},
        config_hash="b" * 64,
        commit="c" * 40,
        decision="accepted",
        blockers=[],
        lineage_parent="round_13_action_observation_contracts",
    )
    assert gate["controller_uid"] == orchestration_controller.CANONICAL_CONTROLLER_UID
    assert gate["validator_uid"] == ROUND14_VALIDATOR_UID


def test_admission_state_machine_and_weighted_e1_success() -> None:
    record = admit_evidence(_input())
    assert record["terminal_state"] == "admitted_weighted"
    assert [row["state"] for row in record["state_history"]] == list(ADMISSION_PHASES)
    assert validate_admission_record(record) == []
    assert record["eligibility"]["eligible_for_state_pretraining"] is True
    assert record["eligibility"]["eligible_for_trajectory_training"] is False


def test_restricted_e3_and_isolated_e7_are_non_empirical() -> None:
    e3 = admit_evidence(
        _input(
            layer="E3",
            item_type="legacy_simulator_trajectory",
            train_eval_role="numerical_dynamics_only",
            legacy_role="numerical_dynamics_only",
            empirical_truth=False,
        )
    )
    e7 = admit_evidence(
        _input(
            layer="E7",
            item_type="hypothetical_scenario",
            train_eval_role="hypothesis_only",
            qc_grade="S",
            pairing_decision="unsupported_pairing",
            empirical_truth=False,
        )
    )
    assert e3["terminal_state"] == "admitted_legacy_restricted"
    assert e7["terminal_state"] == "admitted_stress"
    for record in (e3, e7):
        assert record["eligibility"]["eligible_for_trajectory_training"] is False
        assert record["eligibility"]["eligible_for_claim_support"] is False


def test_pilot_selection_covers_e1_and_e2() -> None:
    blocks = [
        {"canonical_uid": "e1-a", "family_id": "static"},
        {"canonical_uid": "e1-b", "family_id": "static"},
        {"canonical_uid": "e2-a", "family_id": "compound_dose_time_perturbation_transcriptomics"},
    ]
    selected = _pilot_blocks(blocks, 1)
    assert [row["canonical_uid"] for row in selected] == ["e1-a", "e2-a"]


def test_twenty_unresolved_gaps_validate_without_fabrication() -> None:
    gaps = [
        {
            "gap_uid": f"gap-{index:02d}",
            "current_status": "frozen_unresolved",
            "resolution_evidence": "",
        }
        for index in range(20)
    ]
    assert validate_gap_resolution_evidence(gaps, []) == []


def test_release_manifest_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    payload = {
        "created_at": "2026-07-27T00:00:00+00:00",
        "code_commit": "a" * 40,
        "source_rows": [],
        "input_bundle_hash": "b" * 64,
        "admissions": [],
        "nodes": [],
        "relations": [],
        "pairings": [],
        "domains": [],
        "eligibilities": [],
        "conflicts": [],
        "actions": [],
        "information_actions": [],
        "observations": [],
        "entity_rows": [],
        "unit_rows": [],
        "ontology_rows": [],
        "gaps": [],
        "independence": [],
        "duplicate_audit": {},
        "conflict_audit": {},
        "eligibility_audit": {},
        "coverage": {"e4_admitted_count": 0},
        "effective": {},
        "leakage": {},
        "e0_validation": {},
    }
    manifest = write_round14_materialization(
        payload,
        config=config,
        output_root=tmp_path,
        repository_root=REPO_ROOT,
    )
    assert validate_release_manifest(manifest) == []
    assert manifest["sealed"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite immutable"):
        write_round14_materialization(
            payload,
            config=config,
            output_root=tmp_path,
            repository_root=REPO_ROOT,
        )


def test_e2_incomplete_action_remains_quarantined() -> None:
    record = admit_evidence(
        _input(
            layer="E2",
            item_type="real_perturbation_endpoint_block",
            train_eval_role="action_endpoint_candidate",
            pairing_decision="insufficient_identity",
            action_observation_valid=False,
            action_complete=False,
        )
    )
    assert record["terminal_state"] == "quarantined"
    assert record["eligibility"]["eligible_for_action_endpoint_training"] is False


def test_eligibility_derivation_keeps_e4_empty_without_admission() -> None:
    eligibility = derive_eligibility(
        subject_uid="unverified-e4",
        layer="E4",
        qc_grade="A",
        terminal_state="quarantined",
        empirical_truth=False,
    )
    assert eligibility["eligible_for_trajectory_training"] is False


def test_parquet_roundtrip_preserves_mixed_scalar_and_structured_values(tmp_path: Path) -> None:
    path = tmp_path / "mixed.parquet"
    write_parquet_records(
        path,
        [{"uid": "structured", "value": {"count": 2}}, {"uid": "scalar", "value": 0.5}],
        ["uid", "value"],
    )
    rows = read_parquet_records(path)
    assert json.loads(rows[0]["value"]) == {"count": 2}
    assert json.loads(rows[1]["value"]) == 0.5


def test_execution_identity_hash_drift_rejected(tmp_path: Path) -> None:
    environment = {"python_version": "test-version", "pyarrow_version": "test-version"}
    identity = {
        **expected_execution_identity_fields(REPO_ROOT),
        "input_bundle_hash": "a" * 64,
        "environment_components": environment,
        "environment_hash": canonical_sha256(environment),
    }
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity), encoding="utf-8")
    blockers, _ = _execution_identity_blockers(path, REPO_ROOT)
    assert blockers == []
    identity["config_hash"] = "b" * 64
    path.write_text(json.dumps(identity), encoding="utf-8")
    blockers, _ = _execution_identity_blockers(path, REPO_ROOT)
    assert any(row.get("field") == "config_hash" for row in blockers)


def test_git_dirty_status_allows_slow_shared_worktree_and_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def delayed_status(*args: object, timeout: float, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if timeout < 30:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=timeout)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orchestration_controller.subprocess, "run", delayed_status)
    assert orchestration_controller.git_dirty_status(tmp_path) == "clean"
    assert orchestration_controller.git_dirty_status(tmp_path, timeout_seconds=5) == "dirty"
