from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.digital_twin_canonical_state import (
    CanonicalStateContractError,
    compile_partial_state_block,
    deserialize_state_block,
    iter_logical_elements,
    logical_state_view,
    serialize_state_block,
    state_block_content_hash,
    state_block_validation_report,
    state_blocks_equal,
    validate_canonical_state_config,
)
from Libs.utils.data.raw_data_registry import load_config


CONFIG = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
VERSION = CONFIG["digital_twin_canonical_state"]["serialization_version"]
MASKS = ("observed", "missing", "unavailable", "not_applicable", "quarantined")


def _masks(count: int, state: str = "observed") -> dict[str, list[bool]]:
    return {name: [name == state] * count for name in MASKS}


def _axis(values: list[object], *, present: bool = True, absent_status: str = "missing") -> dict[str, object]:
    if not present:
        return {
            "axis_status": absent_status,
            "axis_present": False,
            "axis_values": [],
            "axis_mapping_version": f"{absent_status}_axis",
            "axis_uncertainty": {"status": "missing", "values": None, "reason": "axis not observed"},
            "axis_mask": _masks(0),
        }
    return {
        "axis_status": "observed",
        "axis_present": True,
        "axis_values": values,
        "axis_mapping_version": "fixture-mapping-v1",
        "axis_uncertainty": {"status": "provided", "values": [0.0] * len(values), "reason": None},
        "axis_mask": _masks(len(values)),
    }


def _payload(*, fmt: str = "dense", partial: bool = True, mask_state: str = "observed") -> dict[str, object]:
    axis_metadata = {
        "time": _axis([0.0], present=not partial),
        "region": _axis(["middle_temporal_gyrus"], present=not partial),
        "compartment": _axis(["isf"], present=not partial),
        "species": _axis(["neuroinflammation"]),
        "modality": _axis(["transcriptomics"]),
    }
    if fmt == "dense":
        storage = {
            "format": "dense", "source_axis_order": ["species"], "shape": [1], "dense_values": [0.0 if mask_state == "observed" else None],
            "sparse_entries": None, "ragged_entries": None, "reference": None,
        }
    elif fmt == "sparse":
        storage = {
            "format": "sparse", "source_axis_order": ["species"], "shape": [1], "dense_values": None,
            "sparse_entries": [{"index": [0], "value": 0.0 if mask_state == "observed" else None}], "ragged_entries": None, "reference": None,
        }
    elif fmt == "ragged":
        storage = {
            "format": "ragged", "source_axis_order": ["species"], "shape": [1], "dense_values": None, "sparse_entries": None,
            "ragged_entries": [{"coordinates": {"species": "neuroinflammation"}, "value": 0.0 if mask_state == "observed" else None}], "reference": None,
        }
    else:
        storage = {
            "format": "reference", "source_axis_order": ["species"], "shape": [1], "dense_values": None, "sparse_entries": None,
            "ragged_entries": None, "reference": {"uri": "Input/public/reference-state.bin", "content_sha256": "b" * 64, "entry_count": 1},
        }
    observed = mask_state == "observed"
    return {
        "schema_version": VERSION,
        "evidence_uid": "evidence-fixture-1",
        "evidence_type": "real_static_state",
        "source_uid": "public-fixture-source",
        "publication_accession": "PMID:fixture",
        "biological_unit_uid": "unit-fixture-1",
        "state_uid": "state-fixture-1",
        "state_time": {"unit": None, "values": []} if partial else {"unit": "hours", "values": [0.0]},
        "time_semantics": "cross_sectional",
        "organism": "Homo sapiens",
        "axis_metadata": axis_metadata,
        "feature_namespace": "entrez_gene_id",
        "feature_ids": ["feature-neuroinflammation"],
        "value_storage": storage,
        "units": {"by_state_variable": {"neuroinflammation": "relative_burden"}, "conversion_kind": "already_canonical_state_unit"},
        "biological_context": {"disease_program": "ad_control", "organism": "Homo sapiens", "tissue_or_biofluid": "middle temporal gyrus tissue"},
        "uncertainty": {
            "source": {"status": "provided", "values": [0.1 if observed else None], "reason": None},
            "mapping": {"status": "provided", "values": [0.2 if observed else None], "reason": None},
        },
        "qc_grade": "A",
        "pairing_semantics": "cross_sectional",
        "aggregation_semantics": {
            "aggregation_level": "none", "aggregation_method": "none", "source_record_count": 1,
            "independent_unit_count": 1, "aggregation_weights": [1.0],
            "aggregation_provenance": {
                "status": "provided", "source_record_uids": ["fixture-row-1"],
                "source_artifact_sha256": "a" * 64, "method_version": "none-v1", "reason": None,
            }, "headline_eligible": False,
        },
        "element_masks": _masks(1, mask_state),
        "provenance": {"source_record_uid": "fixture-row-1", "source_artifact_sha256": "a" * 64, "mapping_version": "fixture-v1", "serialization_origin": "unit-test"},
        "allowed_train_roles": ["state_reconstruction"] if observed else [],
        "allowed_eval_roles": ["holdout"],
        "claim_eligibility": "state_reconstruction_only",
    }


def test_config_contract_is_valid_and_schema_owned() -> None:
    assert validate_canonical_state_config(CONFIG) == []
    assert CONFIG["schemas"]["digital_twin_canonical_state_block_fields"][1] == "state_block_uid"
    assert CONFIG["schemas"]["digital_twin_canonical_state_element_mask_fields"] == list(MASKS)


def test_observed_zero_is_distinct_from_missing_and_masks_are_exhaustive() -> None:
    block = compile_partial_state_block(CONFIG, _payload())
    assert block.payload["value_storage"]["dense_values"] == [0.0]
    assert block.payload["element_masks"]["observed"] == [True]
    assert all(sum(block.payload["element_masks"][state][0] for state in MASKS) == 1 for _ in [0])
    assert logical_state_view(block) == {"format": "dense", "source_axis_order": ["species"], "logical_element_count": 1, "densified": False}
    assert block.payload["loss_eligibility"]["eligible_for_temporal_dynamics"] is False


def test_partial_axes_remain_explicit_and_do_not_broadcast() -> None:
    block = compile_partial_state_block(CONFIG, _payload(partial=True))
    for axis in ("time", "region", "compartment"):
        assert block.payload["axis_metadata"][axis]["axis_present"] is False
        assert block.payload["axis_metadata"][axis]["axis_status"] == "missing"
        assert block.payload["axis_metadata"][axis]["axis_values"] == []
    assert list(iter_logical_elements(block))[0]["storage_entry"]["value"] == 0.0


def test_absent_axis_status_preserves_unavailable_and_not_applicable() -> None:
    payload = _payload(partial=True)
    payload["axis_metadata"]["region"] = _axis([], present=False, absent_status="unavailable")
    payload["axis_metadata"]["compartment"] = _axis([], present=False, absent_status="not_applicable")
    block = compile_partial_state_block(CONFIG, payload)
    assert block.payload["axis_metadata"]["region"]["axis_status"] == "unavailable"
    assert block.payload["axis_metadata"]["compartment"]["axis_status"] == "not_applicable"


@pytest.mark.parametrize("mask_state", ["missing", "unavailable", "not_applicable", "quarantined"])
def test_every_nonobserved_mask_state_remains_nonnumeric_and_nontrainable(mask_state: str) -> None:
    block = compile_partial_state_block(CONFIG, _payload(mask_state=mask_state))
    assert block.payload["value_storage"]["dense_values"] == [None]
    assert block.payload["element_masks"][mask_state] == [True]
    assert block.payload["loss_eligibility"]["eligible_for_state_reconstruction"] is False


@pytest.mark.parametrize("fmt", ["dense", "sparse", "ragged", "reference"])
def test_all_storage_formats_round_trip_hash_and_logical_view(fmt: str) -> None:
    mask_state = "unavailable" if fmt == "reference" else "observed"
    block = compile_partial_state_block(CONFIG, _payload(fmt=fmt, mask_state=mask_state))
    serialized = json.loads(serialize_state_block(block))
    reread = deserialize_state_block(CONFIG, serialized)
    assert state_blocks_equal(block, reread)
    assert state_block_content_hash(block) == state_block_content_hash(reread)
    assert logical_state_view(block)["format"] == fmt
    assert logical_state_view(block)["densified"] is False


def test_pseudotime_is_never_temporal_or_physical_eligible() -> None:
    payload = _payload()
    payload["evidence_type"] = "published_mechanistic_output"
    payload["time_semantics"] = "pseudotime"
    payload["pairing_semantics"] = "semantic_alignment_only"
    payload["state_time"] = {"unit": "pseudotime", "values": [1.0]}
    block = compile_partial_state_block(CONFIG, payload)
    assert block.payload["loss_eligibility"]["eligible_for_temporal_dynamics"] is False
    assert block.payload["loss_eligibility"]["eligible_for_what_if_initialization"] is False


def test_legacy_aggregate_has_no_invented_independent_units_or_claims() -> None:
    payload = _payload(partial=False)
    payload.update({
        "evidence_type": "legacy_simulator_trajectory", "time_semantics": "simulator_time", "pairing_semantics": "simulator_generated",
        "biological_unit_uid": None, "aggregation_semantics": {
            "aggregation_level": "unknown", "aggregation_method": "legacy_aggregate_unknown", "source_record_count": 222336,
            "independent_unit_count": None, "aggregation_weights": None,
            "aggregation_provenance": {"status": "unavailable", "source_record_uids": None, "source_artifact_sha256": None, "method_version": None, "reason": "legacy source unit identities unavailable"},
            "headline_eligible": False,
        },
        "allowed_train_roles": [], "claim_eligibility": "ineligible",
    })
    block = compile_partial_state_block(CONFIG, payload)
    assert block.payload["aggregation_semantics"]["headline_eligible"] is False
    bad = deepcopy(payload)
    bad["aggregation_semantics"]["independent_unit_count"] = 222336
    with pytest.raises(CanonicalStateContractError, match="invalid_legacy_aggregate"):
        compile_partial_state_block(CONFIG, bad)


def test_aggregation_semantics_are_exact_and_typed() -> None:
    methods = set(CONFIG["digital_twin_canonical_state"]["supported_aggregation_methods"])
    assert methods == {"none", "cell_to_donor", "sample_to_subject", "technical_replicate", "pseudobulk", "source_aggregate", "legacy_aggregate_unknown"}
    block = compile_partial_state_block(CONFIG, _payload())
    aggregation = block.payload["aggregation_semantics"]
    assert set(aggregation) == {"aggregation_level", "aggregation_method", "source_record_count", "independent_unit_count", "aggregation_weights", "aggregation_provenance", "headline_eligible"}
    assert aggregation["aggregation_method"] == "none"


@pytest.mark.parametrize(
    ("evidence_type", "time_semantics", "pairing_semantics", "claim"),
    [
        ("real_perturbation_endpoint", "ordered_endpoint", "same_unit_action_endpoint", "endpoint_direction_only"),
        ("real_longitudinal_trajectory", "absolute_time", "same_unit_longitudinal", "evaluation_only"),
        ("clean_simulator_trajectory", "simulator_time", "simulator_generated", "dynamics_pretraining_only"),
    ],
)
def test_real_endpoint_longitudinal_and_clean_simulator_semantics_are_distinct(
    evidence_type: str, time_semantics: str, pairing_semantics: str, claim: str
) -> None:
    payload = _payload(partial=evidence_type == "real_perturbation_endpoint")
    payload.update({"evidence_type": evidence_type, "time_semantics": time_semantics, "pairing_semantics": pairing_semantics, "claim_eligibility": claim})
    if evidence_type == "clean_simulator_trajectory":
        payload["aggregation_semantics"] = {
            "aggregation_level": "source", "aggregation_method": "source_aggregate", "source_record_count": 1,
            "independent_unit_count": 1, "aggregation_weights": [1.0],
            "aggregation_provenance": {"status": "provided", "source_record_uids": ["fixture-row-1"], "source_artifact_sha256": "a" * 64, "method_version": "source-v1", "reason": None},
            "headline_eligible": False,
        }
    block = compile_partial_state_block(CONFIG, payload)
    assert block.payload["time_semantics"] == time_semantics
    assert block.payload["loss_eligibility"]["eligible_for_temporal_dynamics"] is (evidence_type != "real_perturbation_endpoint")


def test_required_time_semantics_are_declared_and_jointly_constrained() -> None:
    required = {
        "absolute_time", "relative_to_action", "pseudotime", "cross_sectional",
        "ordered_endpoint", "simulator_time", "unknown", "not_applicable",
    }
    assert required <= set(CONFIG["digital_twin_canonical_state"]["supported_time_semantics"])
    physical = _payload(partial=False)
    physical.update({"evidence_type": "published_mechanistic_output", "time_semantics": "relative_to_action", "pairing_semantics": "semantic_alignment_only"})
    assert compile_partial_state_block(CONFIG, physical).payload["state_time"]["unit"] == "hours"
    unknown = _payload()
    unknown.update({"evidence_type": "ontology_or_prior", "time_semantics": "unknown", "pairing_semantics": "not_paired"})
    assert compile_partial_state_block(CONFIG, unknown).payload["loss_eligibility"]["eligible_for_temporal_dynamics"] is False
    invalid = _payload()
    invalid.update({"evidence_type": "published_mechanistic_output", "time_semantics": "absolute_time"})
    with pytest.raises(CanonicalStateContractError, match="time_semantics_axis_mismatch"):
        compile_partial_state_block(CONFIG, invalid)


def test_loss_eligibility_is_derived_and_cannot_be_row_hardcoded() -> None:
    payload = _payload()
    payload["loss_eligibility"] = {field: False for field in ("eligible_for_state_reconstruction", "eligible_for_temporal_dynamics", "eligible_for_action_endpoint", "eligible_for_uncertainty_calibration", "eligible_for_constraint_evaluation", "eligible_for_what_if_initialization")}
    with pytest.raises(CanonicalStateContractError, match="loss_eligibility_mismatch"):
        compile_partial_state_block(CONFIG, payload)


def test_six_eligibility_fields_are_derived() -> None:
    block = compile_partial_state_block(CONFIG, _payload())
    assert set(block.payload["loss_eligibility"]) == {
        "eligible_for_state_reconstruction", "eligible_for_temporal_dynamics", "eligible_for_action_endpoint",
        "eligible_for_uncertainty_calibration", "eligible_for_constraint_evaluation", "eligible_for_what_if_initialization",
    }
    assert block.payload["loss_eligibility"]["eligible_for_state_reconstruction"] is True
    assert block.payload["loss_eligibility"]["eligible_for_what_if_initialization"] is True


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p: p.update({"evidence_type": "unknown"}), "invalid_evidence_type"),
        (lambda p: p.update({"time_semantics": "unrecognized_time_value"}), "invalid_time_semantics"),
        (lambda p: p.update({"evidence_type": "real_perturbation_endpoint", "time_semantics": "absolute_time", "pairing_semantics": "same_unit_longitudinal"}), "static_endpoint_marked_longitudinal"),
        (lambda p: p["value_storage"].update({"source_axis_order": ["species", "time"]}), "invalid_source_axis_order"),
        (lambda p: p["feature_ids"].append("extra"), "feature_value_mismatch"),
        (lambda p: p["units"]["by_state_variable"].update({"neuroinflammation": "wrong"}), "incompatible_species_unit"),
        (lambda p: p["value_storage"].update({"dense_values": [float("nan")]}), "invalid_state_value"),
        (lambda p: p["value_storage"].update({"format": "reference", "dense_values": None, "reference": {"uri": "/private/path.bin", "content_sha256": "b" * 64, "entry_count": 1}}), "private_absolute_reference_path"),
        (
            lambda p: (
                p["value_storage"].update({"dense_values": [None]}),
                p["element_masks"].update(_masks(1, "quarantined")),
                p["uncertainty"].update({
                    "source": {"status": "provided", "values": [None], "reason": None},
                    "mapping": {"status": "provided", "values": [None], "reason": None},
                }),
                p.update({"allowed_train_roles": ["state_reconstruction"]}),
            ),
            "quarantined_train_eligibility",
        ),
    ],
)
def test_invalid_contract_combinations_fail_closed(mutate: object, code: str) -> None:
    payload = _payload()
    mutate(payload)  # type: ignore[operator]
    with pytest.raises(CanonicalStateContractError, match=code):
        compile_partial_state_block(CONFIG, payload)


def test_hash_mismatch_and_axis_mask_ambiguity_fail_closed() -> None:
    serialized = compile_partial_state_block(CONFIG, _payload()).to_dict()
    serialized["content_hash"] = "0" * 64
    with pytest.raises(CanonicalStateContractError, match="content_hash_mismatch"):
        deserialize_state_block(CONFIG, serialized)
    payload = _payload()
    payload["axis_metadata"]["species"]["axis_mask"]["missing"] = [True]
    with pytest.raises(CanonicalStateContractError, match="invalid_axis_mask"):
        compile_partial_state_block(CONFIG, payload)


def test_validation_report_contains_derived_eligibility() -> None:
    report = state_block_validation_report(CONFIG, compile_partial_state_block(CONFIG, _payload()).to_dict())
    assert report["status"] == "passed"
    assert report["mask_counts"] == {"observed": 1, "missing": 0, "unavailable": 0, "not_applicable": 0, "quarantined": 0}
    assert report["loss_eligibility"]["eligible_for_state_reconstruction"] is True
