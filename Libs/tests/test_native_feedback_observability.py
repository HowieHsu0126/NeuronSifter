"""Fixed arithmetic and read-boundary checks for the bounded feedback diagnostic."""
import json
from pathlib import Path

import numpy as np
import pytest

from Libs.scripts.reports.build_native_feedback_observability import (
    difference_summary, numerical_receipt_delta, validate_training_receipt, verify_initial_projection,
    training_precision_contrasts, normalized_numerical_error, reference_serialization_error,
)
from Libs.utils.model.native_arrived_observation import NativeCNSReadout


def readout():
    matrix = np.eye(6)
    matrix[0, 1] = .25
    return NativeCNSReadout("channel", "assay", "source", "snapshot", "train", "family", "scales",
        "source_reference_standardized", ("region",), ("compartment",), tuple("abcdef"),
        tuple((0, 0, i) for i in range(6)), (.1, .2, .3, .4, .5, .6),
        tuple(map(tuple, matrix)), (True, True, False, False, False, False), .03)


def test_noise_summary_keeps_mask_and_independent_difference_scale():
    value = difference_summary([.03, .06, 1e20, 0, 0, 0], readout(), np.ones(6))
    assert value["observed_l2_over_single_observation_noise"] == pytest.approx(np.sqrt(5))
    assert value["observed_l2_over_independent_difference_noise"] == pytest.approx(np.sqrt(2.5))
    assert value["observed_coordinates_exceeding_serialization_bound"] == 0


def test_initial_projection_checks_the_declared_two_float32_roundings():
    spec = readout()
    fields = np.arange(32 * 6, dtype=np.float32).reshape(32, 1, 1, 6) / 71
    scaled = np.asarray([[fields[i, 0, 0, j] / spec.scales[j] for j in range(6)] for i in range(32)], dtype=np.float32)
    means = (scaled @ np.asarray(spec.matrix).T).astype(np.float32)
    projected, check = verify_initial_projection(fields, means, spec)
    assert projected.shape == (32, 6) and check["status"] == "within_declared_serialization_precision"
    changed = means.copy()
    changed[0, 0] += .01
    with pytest.raises(ValueError, match="serialization precision"):
        verify_initial_projection(fields, changed, spec)


def receipt(index=0):
    group = {"group_uid": str(index), "sample_uid": str(index), "role": "train",
        "particle_uids": [str(i) for i in range(32)], "conditioning_reference": {"path": "declared.gz"}}
    return {"group_index": index, "group": group, "source_identity": {"producer_commit": "frozen"},
        "cases": [{"group_index": index, "action_index": i, "status": "complete",
            "numerical": {"max_normalized_error": .001, "reasons": []},
            "raw_reference": f"case_{index*3+i}_reference.npz"} for i in range(3)],
        "operations": {"operation": {"status": "complete", "kind": "groups"}},
        "source_clock_seconds": float(index + 1)}


@pytest.mark.parametrize("changed", ["role", "case", "numerical", "private_reference"])
def test_training_gate_rejects_invalid_metadata_before_any_payload(changed, monkeypatch):
    value = receipt()
    expected = {key: item for key, item in value["group"].items() if key != "conditioning_reference"}
    if changed == "role":
        value["group"]["role"] = "validation"
    elif changed == "case":
        value["cases"][1]["status"] = "started"
    elif changed == "numerical":
        value["cases"][1]["numerical"]["reasons"] = ["unresolved"]
    else:
        value["cases"][1]["raw_reference"] = "group_0_producer.json"
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata gate attempted a payload read")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError):
        validate_training_receipt(value, 0, expected, value["source_identity"])


def test_receipt_delta_counts_only_new_metadata_and_stops_before_missing(tmp_path, monkeypatch):
    for index in (1888, 1889):
        value = receipt(index)
        value["group"]["role"] = "calibration"
        (tmp_path / f"group_{index}_receipt.json").write_text(json.dumps(value))
    expected_bytes = sum((tmp_path / f"group_{index}_receipt.json").stat().st_size for index in (1888, 1889))
    def forbidden(*args, **kwargs):
        raise AssertionError("Receipt delta attempted an array read")
    monkeypatch.setattr(np, "load", forbidden)
    result = numerical_receipt_delta(tmp_path, 1888, 1891, {"producer_commit": "frozen"})
    assert result["status"] == "missing_receipt"
    assert result["completed_accepted_groups"] == 2 and result["stop_index_exclusive"] == 1890
    assert result["case_status_counts"] == {"complete": 6}
    assert result["bytes_read"] == expected_bytes and result["array_reads"] == 0


def test_actual_float32_contrast_erasure_preserves_mask_and_surviving_channels():
    fields = [np.ones((2, 1, 1, 6), dtype=np.float64) for _ in range(3)]
    for value in fields:
        value[..., 5] = 0.
    fields[1][1, 0, 0, 0] += 2.**-26
    fields[2][1, 0, 0, 1] += 2.**-12
    fields[1][1, 0, 0, 2] = 7.  # A distinct full-state quantity outside the observed mask.
    support = np.ones_like(fields[0], dtype=bool); support[..., 5] = False
    result = training_precision_contrasts(fields, support, readout())
    assert result["by_quantity"]["a"]["raw_nonzero"] == 2
    assert result["by_quantity"]["a"]["erased_nonzero"] == 2
    assert result["by_quantity"]["a"]["stored_nonzero"] == 0
    assert result["by_quantity"]["b"]["stored_nonzero"] == 2
    assert result["by_quantity"]["b"]["erased_nonzero"] == 0
    assert result["by_quantity"]["c"]["stored_nonzero"] == 2
    assert result["by_quantity"]["f"]["coordinate_denominator"] == 0
    observed = result["projected_observed_cns"]
    assert observed["coordinate_denominator"] == 6
    assert (observed["raw_nonzero"], observed["stored_nonzero"], observed["erased_nonzero"]) == (5, 4, 1)
    assert observed["stored_l2_over_single_observation_noise"]["max"] == pytest.approx(
        (2.**-12 / .2) * np.sqrt(1. + .25**2) / .03)


@pytest.mark.parametrize("changed", ["t0", "nonfinite", "support"])
def test_precision_measurement_rejects_unmatched_or_invalid_fields(changed):
    fields = [np.ones((2, 1, 1, 6), dtype=np.float64) for _ in range(3)]
    support = np.ones_like(fields[0], dtype=bool)
    if changed == "t0": fields[1][0] += 1.
    elif changed == "nonfinite": fields[1][1, 0, 0, 0] = np.nan
    else: support = support.astype(np.float32)
    with pytest.raises(ValueError, match="three finite aligned actions"):
        training_precision_contrasts(fields, support, readout())


def test_long_cns_receipt_preserves_its_admission_and_reference_cast(tmp_path):
    value = receipt()
    errors = np.linspace(0., 1e-7, 15)
    for case in value["cases"]:
        case["numerical"] = {"normalized_max": .0001, "admissible": True, "reasons": [],
                             "components": {"reference_float32_serialization": errors.tolist()}}
    expected = {key: item for key, item in value["group"].items() if key != "conditioning_reference"}
    validate_training_receipt(value, 0, expected, value["source_identity"])
    np.testing.assert_array_equal(reference_serialization_error(value["cases"][0]["numerical"]), errors)
    (tmp_path / "group_0_receipt.json").write_text(json.dumps(value))
    summary = numerical_receipt_delta(tmp_path, 0, 1, value["source_identity"])
    assert summary["status"] == "complete" and summary["maximum_normalized_error"] == .0001
    assert summary["completed_accepted_groups"] == 1 and summary["array_reads"] == 0


@pytest.mark.parametrize("quality", [
    {"normalized_max": .001, "admissible": False},
    {"normalized_max": .001},
    {"normalized_max": float("nan"), "admissible": True},
    {"normalized_max": -1., "admissible": True},
    {"normalized_max": .001, "max_normalized_error": .001, "admissible": True},
    {},
])
def test_receipt_schema_never_promotes_missing_or_rejected_evidence(quality):
    with pytest.raises(ValueError):
        normalized_numerical_error(quality)
