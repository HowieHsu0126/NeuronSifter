from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from Libs.utils.data.trajectory_numeric_lineage import (
    METRIC_COLUMNS,
    active_reader_violations,
    build_numeric_lineage_audit,
    canonical_metric_values,
    markdown_report,
    validate_numeric_lineage_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


@pytest.fixture(scope="module")
def lineage_report() -> dict[str, object]:
    return build_numeric_lineage_audit(CONFIG_PATH)


def test_numeric_lineage_reconstructs_current_counts_but_preserves_missing_historical_manifest_blocker(
    lineage_report: dict[str, object],
) -> None:
    report = lineage_report

    metrics = canonical_metric_values(report)
    assert metrics["raw_materialized_full_microenvironment_trajectory_count"] == 111217
    assert metrics["paired_full_mesoscopic_count"] == 111168
    assert metrics["rejected_trajectory_count"] == 49
    assert report["reconciliation"]["historical_prefix_reject_indices"] == [111168, 111169, 111170]
    assert report["reconciliation"]["appended_reject_indices"] == list(range(111171, 111217))
    historical = report["historical_snapshot"]
    projection = report["reconciliation"]["historical_projection"]
    assert historical["provenance_level"] == "reconstructed_historical_projection_not_original_manifest"
    assert historical["original_manifest_recovered"] is False
    assert historical["governance_records_verified"] is True
    assert all(record["blob_verified"] and record["fragments_verified"] for record in historical["governance_records"])
    assert projection["verified"] is True
    assert projection["prefix_paired_indices"] == projection["pair_alignment_full_indices"]
    assert projection["prefix_counts"]["teacher_family_counts"] == {
        "ad_full_microenvironment_main_dynamics": 111168,
        "osp_pksim_pbpk": 3,
    }
    assert projection["suffix_counts"]["teacher_family_counts"] == {
        "neuron_rxd_microenvironment": 1,
        "osp_pksim_pbpk": 45,
    }
    independent_manifest = historical["independent_manifest"]
    assert independent_manifest["required"] is True
    assert independent_manifest["verified"] is False
    assert report["unresolved_conflict_ids"] == ["historical_source_manifest_hash_missing"]
    assert report["status"] == "blocked"
    assert {row["code"] for row in validate_numeric_lineage_report(report, CONFIG_PATH)} == {
        "historical_source_manifest_hash_missing"
    }


def test_numeric_lineage_validator_rejects_missing_required_metric_field(lineage_report: dict[str, object]) -> None:
    report = lineage_report
    malformed = deepcopy(report)
    del malformed["metrics"][0]["source_artifact_hash"]

    blocker_codes = {row["code"] for row in validate_numeric_lineage_report(malformed, CONFIG_PATH)}
    assert "metric_required_field_missing" in blocker_codes


@pytest.mark.parametrize(
    ("mutate", "expected_blocker"),
    [
        (
            lambda report: report["historical_snapshot"]["governance_records"][0].update({"blob_verified": False}),
            "historical_governance_record_unverified",
        ),
        (
            lambda report: report["historical_snapshot"].update({"full_count": 111170}),
            "historical_governance_value_mismatch",
        ),
        (
            lambda report: report["reconciliation"]["historical_projection"]["prefix_indices"].__setitem__(
                slice(0, 2), [1, 0]
            ),
            "historical_projection_prefix_indices_invalid",
        ),
        (
            lambda report: report["reconciliation"]["historical_projection"]["suffix_indices"].pop(),
            "historical_projection_suffix_indices_invalid",
        ),
        (
            lambda report: report["reconciliation"]["historical_projection"].update({"original_manifest_recovered": True}),
            "historical_projection_manifest_recovery_state_mismatch",
        ),
        (
            lambda report: report["source_artifacts"]["qc_catalog"].update({"sha256": "0" * 64}),
            "source_artifact_hash_stale",
        ),
    ],
)
def test_numeric_lineage_validator_rejects_historical_projection_provenance_and_identity_tampering(
    lineage_report: dict[str, object],
    mutate: object,
    expected_blocker: str,
) -> None:
    malformed = deepcopy(lineage_report)
    mutate(malformed)

    blocker_codes = {row["code"] for row in validate_numeric_lineage_report(malformed, CONFIG_PATH)}

    assert expected_blocker in blocker_codes


def test_numeric_lineage_cannot_close_from_a_current_prefix_projection_without_a_historical_manifest(
    lineage_report: dict[str, object],
) -> None:
    malformed = deepcopy(lineage_report)
    malformed["status"] = "pass"
    malformed["unresolved_conflict_ids"] = []
    malformed["unresolved_conflict_count"] = 0

    blocker_codes = {row["code"] for row in validate_numeric_lineage_report(malformed, CONFIG_PATH)}

    assert "historical_source_manifest_hash_missing" in blocker_codes
    assert "invalid_pass_status" in blocker_codes


def test_numeric_lineage_markdown_marks_historical_values(lineage_report: dict[str, object]) -> None:
    report = lineage_report
    rendered = markdown_report(report)

    assert "historical snapshot" in rendered
    assert "not current runtime truth" in rendered
    assert "superseded by trajectory_numeric_lineage_audit.json" in rendered


def test_numeric_lineage_metric_schema_is_explicit(lineage_report: dict[str, object]) -> None:
    report = lineage_report
    for metric in report["metrics"]:
        assert set(METRIC_COLUMNS) == set(metric)


def test_active_readers_reject_hard_coded_or_undefined_generic_numeric_facts(tmp_path: Path) -> None:
    reader = tmp_path / "reader.py"
    reader.write_text(
        "from Libs.utils.data.trajectory_numeric_lineage import load_canonical_metric_values\n"
        "count = 111217\n"
        "generic = payload['trajectory_count']\n",
        encoding="utf-8",
    )

    violations = active_reader_violations([reader], ["111171", "111217"])
    codes = {row["code"] for row in violations}

    assert "active_reader_hard_coded_numeric_fact" in codes
    assert "active_reader_undefined_generic_trajectory_count" in codes
