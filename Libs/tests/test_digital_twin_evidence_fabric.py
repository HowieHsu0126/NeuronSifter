from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.digital_twin_evidence_fabric import (
    REQUIRED_EVIDENCE_FIELDS,
    build_evidence_fabric_candidate,
    build_relationship_graph,
    evidence_uid_for_record,
    validate_evidence_fabric_outputs,
    validate_evidence_records,
    validate_fabric_config,
    write_evidence_fabric_outputs,
)
from Libs.utils.data.raw_data_registry import load_config, sha256_of_file, write_json


BASE_CONFIG = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")


def _base_config() -> dict:
    config = deepcopy(BASE_CONFIG)
    fabric = config["digital_twin_evidence_fabric"]
    fabric["output_paths"] = {
        "inventory": "outputs/inventory.json",
        "relationship_graph": "outputs/graph.json",
        "role_matrix": "outputs/roles.tsv",
        "validation": "outputs/validation.json",
    }
    fabric["source_adapters"] = []
    return config


def _defaults(evidence_type: str = "real_static_state") -> dict[str, object]:
    by_type = {
        "real_static_state": ("static", "cross_sectional", "observed", ["state_reconstruction"], ["holdout"], "state_reconstruction_only"),
        "real_perturbation_endpoint": ("endpoint", "same_unit_action_endpoint", "observed", ["endpoint_calibration"], ["endpoint_evaluation"], "endpoint_direction_only"),
        "legacy_simulator_trajectory": ("simulated_trajectory", "simulator_generated", "legacy_simulator", ["numerical_dynamics"], ["trajectory_evaluation"], "dynamics_pretraining_only"),
        "tool_result": ("timeless", "not_paired", "tool_native", ["weak_calibration"], ["holdout"], "ineligible"),
    }
    time_semantics, pairing, fidelity, train_roles, eval_roles, claim = by_type[evidence_type]
    return {
        "evidence_type": evidence_type,
        "source_uid": "fixture_source",
        "publication_accession": None,
        "biological_unit_uid": None,
        "state_uid": None,
        "action_uid": None,
        "trajectory_uid": None,
        "tool_result_uid": None,
        "model_family_uid": None,
        "time_semantics": time_semantics,
        "pairing_semantics": pairing,
        "fidelity": fidelity,
        "qc_grade": "A",
        "uncertainty": {"status": "measured"},
        "allowed_train_roles": train_roles,
        "allowed_eval_roles": eval_roles,
        "claim_eligibility": claim,
    }


def _json_adapter(adapter_id: str, path: str, evidence_type: str = "real_static_state") -> dict[str, object]:
    return {
        "adapter_id": adapter_id,
        "path": path,
        "reader_type": "json_records",
        "record_path": "records",
        "required": True,
        "expected_identity": {"schema_version": "fixture-evidence-v1"},
        "uid_columns": ["record_uid"],
        "field_sources": {
            "source_uid": ["source_uid"],
            "biological_unit_uid": ["biological_unit_uid"],
            "state_uid": ["state_uid"],
            "action_uid": ["action_uid"],
        },
        "provenance_fields": ["record_uid", "observed_value", "observed_axes", "missing_axes"],
        "evidence_defaults": _defaults(evidence_type),
    }


def _write_fixture(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "fixture-evidence-v1", "records": records}), encoding="utf-8")


def _complete_fixture(tmp_path: Path) -> dict:
    config = _base_config()
    config["digital_twin_evidence_fabric"]["source_adapters"] = [
        _json_adapter("static", "inputs/static.json"),
        _json_adapter("endpoint", "inputs/endpoint.json", "real_perturbation_endpoint"),
    ]
    _write_fixture(
        tmp_path / "inputs/static.json",
        [
            {
                "record_uid": "static-1",
                "source_uid": "public-static-source",
                "biological_unit_uid": "unit-1",
                "state_uid": "state-1",
                "observed_value": 0,
                "observed_axes": ["species"],
                "missing_axes": ["time", "region", "compartment"],
            }
        ],
    )
    _write_fixture(
        tmp_path / "inputs/endpoint.json",
        [
            {
                "record_uid": "endpoint-1",
                "source_uid": "public-endpoint-source",
                "biological_unit_uid": "unit-2",
                "state_uid": "state-2",
                "action_uid": "executed-action-1",
            }
        ],
    )
    return config


def _canonical_record(evidence_type: str = "real_static_state") -> dict[str, object]:
    record = {"evidence_uid": "", **_defaults(evidence_type), "provenance": {"source": "fixture"}}
    ordered = {field: record[field] for field in REQUIRED_EVIDENCE_FIELDS}
    ordered["evidence_uid"] = evidence_uid_for_record(ordered)
    return ordered


def _write_candidate(config: dict, tmp_path: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    inventory, graph, roles = build_evidence_fabric_candidate(config, repository_root=tmp_path)
    write_evidence_fabric_outputs(config, inventory, graph, roles, repository_root=tmp_path)
    return inventory, graph, roles


def test_complete_fixture_passes_independent_readback_and_preserves_missingness(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    inventory, graph, roles = _write_candidate(config, tmp_path)
    report = validate_evidence_fabric_outputs(config, repository_root=tmp_path)

    assert report["status"] == "passed_sealed"
    assert report["sealed"] is True
    assert report["blocker_count"] == 0
    assert inventory["record_count"] == 2
    assert graph["node_count"] == 2
    assert len(roles) == 2
    assert {row["fabric_identity"] for row in roles} == {inventory["fabric_identity"]}

    static_record = next(row for row in inventory["records"] if row["evidence_type"] == "real_static_state")
    metadata = static_record["provenance"]["source_metadata"]
    assert metadata["observed_value"] == 0
    assert metadata["missing_axes"] == ["time", "region", "compartment"]
    assert "state_values" not in static_record
    assert "state_values" not in static_record["provenance"]


def test_adapter_order_does_not_change_uids_or_fabric_identity(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    first, _, _ = build_evidence_fabric_candidate(config, repository_root=tmp_path)
    config["digital_twin_evidence_fabric"]["source_adapters"].reverse()
    second, _, _ = build_evidence_fabric_candidate(config, repository_root=tmp_path)
    assert first["fabric_identity"] == second["fabric_identity"]
    assert [row["evidence_uid"] for row in first["records"]] == [row["evidence_uid"] for row in second["records"]]


def test_missing_required_source_is_precise_and_unsealed(tmp_path: Path) -> None:
    config = _base_config()
    config["digital_twin_evidence_fabric"]["source_adapters"] = [_json_adapter("missing", "inputs/missing.json")]
    inventory, _, _ = _write_candidate(config, tmp_path)
    report = validate_evidence_fabric_outputs(config, repository_root=tmp_path)
    assert inventory["sealed"] is False
    assert report["status"] == "blocked_missing_required_frozen_inputs"
    assert report["sealed"] is False
    assert report["blockers"] == [
        {
            "code": "missing_required_frozen_input",
            "message": "A required frozen input is absent from the assigned repository snapshot.",
            "adapter_id": "missing",
            "path": "inputs/missing.json",
        }
    ]


def test_config_rejects_absolute_source_path_and_symlink_escape(tmp_path: Path) -> None:
    config = _base_config()
    absolute = _json_adapter("absolute", str(tmp_path / "outside.json"))
    config["digital_twin_evidence_fabric"]["source_adapters"] = [absolute]
    assert any(row["code"] == "absolute_or_missing_source_path" for row in validate_fabric_config(config))

    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/escape.json").symlink_to(outside)
    config["digital_twin_evidence_fabric"]["source_adapters"] = [_json_adapter("escape", "inputs/escape.json")]
    inventory, _, _ = build_evidence_fabric_candidate(config, repository_root=tmp_path)
    assert any(row["code"] == "source_path_escape" for row in inventory["blockers"])


def test_input_mutation_during_read_is_rejected(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    calls: dict[Path, int] = {}

    def changing_hash(path: Path) -> str:
        calls[path] = calls.get(path, 0) + 1
        actual = sha256_of_file(path)
        return actual if calls[path] % 2 == 1 else "0" * 64

    inventory, _, _ = build_evidence_fabric_candidate(config, repository_root=tmp_path, hash_function=changing_hash)
    assert inventory["record_count"] == 0
    assert {row["code"] for row in inventory["blockers"]} == {"input_mutated_during_read"}


def test_source_rows_require_configured_stable_identity_fields(tmp_path: Path) -> None:
    config = _base_config()
    config["digital_twin_evidence_fabric"]["source_adapters"] = [_json_adapter("static", "inputs/static.json")]
    _write_fixture(tmp_path / "inputs/static.json", [{"source_uid": "missing-row-identity"}])
    inventory, _, _ = build_evidence_fabric_candidate(config, repository_root=tmp_path)
    assert any(row["code"] == "missing_source_row_identity" for row in inventory["blockers"])


def test_missing_fields_and_conflicting_duplicate_uid_fail_closed() -> None:
    config = _base_config()
    record = _canonical_record()
    malformed = dict(record)
    malformed.pop("uncertainty")
    assert any(row["code"] == "invalid_evidence_field_order" for row in validate_evidence_records(config, [malformed]))

    conflicting = deepcopy(record)
    conflicting["source_uid"] = "different-source"
    conflicting["evidence_uid"] = record["evidence_uid"]
    codes = {row["code"] for row in validate_evidence_records(config, [record, conflicting])}
    assert "conflicting_duplicate_evidence_uid" in codes
    assert "invalid_evidence_uid" in codes


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_code"),
    [
        ("evidence_type", "not_evidence", "invalid_evidence_type"),
        ("pairing_semantics", "maybe_paired", "invalid_pairing_semantics"),
        ("time_semantics", "eventually", "invalid_time_semantics"),
        ("fidelity", "perfect", "invalid_fidelity"),
        ("qc_grade", "gold", "invalid_qc_grade"),
        ("claim_eligibility", "clinical", "invalid_claim_eligibility"),
        ("allowed_train_roles", ["unknown_train"], "invalid_train_role"),
        ("allowed_eval_roles", ["unknown_eval"], "invalid_eval_role"),
    ],
)
def test_every_record_enum_is_validated(field: str, invalid_value: object, expected_code: str) -> None:
    config = _base_config()
    record = _canonical_record()
    record[field] = invalid_value
    record["evidence_uid"] = evidence_uid_for_record(record)
    assert any(row["code"] == expected_code for row in validate_evidence_records(config, [record]))


def test_static_endpoint_cannot_be_longitudinal() -> None:
    config = _base_config()
    record = _canonical_record("real_perturbation_endpoint")
    record["time_semantics"] = "longitudinal"
    record["pairing_semantics"] = "same_unit_longitudinal"
    record["evidence_uid"] = evidence_uid_for_record(record)
    codes = {row["code"] for row in validate_evidence_records(config, [record])}
    assert "static_evidence_marked_longitudinal" in codes


def test_semantic_alignment_is_not_pairing_and_dangling_edges_fail() -> None:
    config = _base_config()
    target = _canonical_record()
    source = _canonical_record()
    source["source_uid"] = "semantic-source"
    source["pairing_semantics"] = "semantic_alignment_only"
    source["provenance"] = {
        "source": "fixture",
        "relationships": [{"target_uid": target["evidence_uid"], "relationship_type": "paired_with"}],
    }
    source["evidence_uid"] = evidence_uid_for_record(source)
    _, blockers = build_relationship_graph(config, [source, target], "dt_fabric_fixture")
    assert any(row["code"] == "invalid_pairing_edge" for row in blockers)

    source["provenance"]["relationships"] = [{"target_uid": "absent", "relationship_type": "related_to"}]
    source["evidence_uid"] = evidence_uid_for_record(source)
    _, blockers = build_relationship_graph(config, [source, target], "dt_fabric_fixture")
    assert any(row["code"] == "dangling_relationship_target" for row in blockers)


def test_non_core_and_generated_observation_roles_fail() -> None:
    config = _base_config()
    non_core = _canonical_record()
    non_core["allowed_train_roles"] = ["state_reconstruction"]
    non_core["provenance"] = {"source": "fixture", "qualification_status": "unqualified_non_core"}
    non_core["evidence_uid"] = evidence_uid_for_record(non_core)
    assert any(row["code"] == "invalid_unqualified_non_core_role" for row in validate_evidence_records(config, [non_core]))

    generated = _canonical_record("tool_result")
    generated["allowed_train_roles"] = ["state_reconstruction"]
    generated["evidence_uid"] = evidence_uid_for_record(generated)
    assert any(row["code"] == "generated_evidence_as_observation" for row in validate_evidence_records(config, [generated]))


def test_tampered_counts_hash_binding_and_false_seal_are_rejected(tmp_path: Path) -> None:
    config = _complete_fixture(tmp_path)
    inventory, graph, _ = _write_candidate(config, tmp_path)
    inventory["observation_record_count"] = 999
    inventory["sealed"] = True
    graph["fabric_identity"] = "dt_fabric_tampered"
    write_json(tmp_path / "outputs/inventory.json", inventory)
    write_json(tmp_path / "outputs/graph.json", graph)
    report = validate_evidence_fabric_outputs(config, repository_root=tmp_path)
    codes = {row["code"] for row in report["blockers"]}
    assert report["status"] == "blocked_invalid_evidence_fabric"
    assert "blocked_output_falsely_labeled_sealed" in codes
    assert "inventory_readback_mismatch" in codes
    assert "relationship_graph_readback_mismatch" in codes
