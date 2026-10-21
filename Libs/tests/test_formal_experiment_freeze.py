from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_formal_experiment_freeze import EXPECTED_G2_STATUS
from Libs.scripts.validators.validate_formal_experiment_freeze import EXPECTED_G5_GATE_STATUS
from Libs.scripts.validators.validate_formal_experiment_freeze import build_freeze_manifest
from Libs.scripts.validators.validate_formal_experiment_freeze import classify_full_only_rows
from Libs.scripts.validators.validate_formal_experiment_freeze import markdown_report
from Libs.scripts.validators.validate_formal_experiment_freeze import read_tsv_rows
from Libs.scripts.validators.validate_formal_experiment_freeze import training_exclusion_report
from Libs.scripts.validators.validate_formal_experiment_freeze import validation_errors
from Libs.utils.data.g0_read_only_artifacts import resolve_read_only_artifact
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.trajectory_numeric_lineage import load_canonical_metric_values


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
PAIR_AUDIT_PATH = REPO_ROOT / "Input/processed/simulation_audit/simulation_pair_alignment_audit.tsv"
QC_CATALOG_PATH = REPO_ROOT / "Input/processed/simulation_audit/simulation_qc_catalog.tsv"
FREEZE_OUTPUT_EXCLUDES = {
    "Output/audits/formal_experiment_freeze_manifest.json",
    "Output/audits/formal_experiment_freeze_manifest.md",
}


@pytest.fixture(scope="module")
def pair_rows() -> list[dict[str, str]]:
    return read_tsv_rows(PAIR_AUDIT_PATH)


@pytest.fixture(scope="module")
def qc_rows() -> list[dict[str, str]]:
    return read_tsv_rows(QC_CATALOG_PATH)


@pytest.fixture(scope="module")
def freeze_manifest() -> dict[str, object]:
    return build_freeze_manifest(CONFIG_PATH, FREEZE_OUTPUT_EXCLUDES)


@pytest.fixture(scope="module")
def canonical_counts() -> dict[str, int]:
    return load_canonical_metric_values(CONFIG_PATH)


def test_full_only_trajectory_difference_classifies_as_reject(
    pair_rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
    canonical_counts: dict[str, int],
) -> None:
    classification = classify_full_only_rows(pair_rows, qc_rows)
    rejected_count = canonical_counts["rejected_trajectory_count"]
    assert classification["full_only_count"] == rejected_count
    assert classification["counts"] == {
        "reject": rejected_count,
        "stress_only": 0,
        "calibration": 0,
        "inventory_only": 0,
        "other": 0,
    }
    assert all(row["classification"] == "reject" for row in classification["rows"])
    assert all(row["qc_grade"] == "D" for row in classification["rows"])
    assert all(row["qc_use"] == "reject" for row in classification["rows"])


def test_training_loader_excludes_unpaired_and_non_main_pretraining_rows(
    pair_rows: list[dict[str, str]],
    qc_rows: list[dict[str, str]],
    canonical_counts: dict[str, int],
) -> None:
    config = load_config(CONFIG_PATH)
    classification = classify_full_only_rows(pair_rows, qc_rows)
    report = training_exclusion_report(config, list(classification["rows"]), qc_rows)
    assert report["train_eligible_grade_ab_count"] == canonical_counts["qc_grade_a_b_count"]
    assert report["full_only_main_pretraining_row_count"] == 0
    assert report["main_pretraining_non_paired_row_count"] == 0
    assert report["loader_filters_qc_use"] is True
    assert report["exclusion_status"] == "pass"


def test_freeze_manifest_records_hashes_commands_and_claim_status(
    freeze_manifest: dict[str, object],
    canonical_counts: dict[str, int],
) -> None:
    assert freeze_manifest["freeze_status"] == "passed"
    assert freeze_manifest["model_claim_change"] == "none"
    counts = freeze_manifest["counts_and_statuses"]
    assert counts["full_microenvironment_trajectory_count"] == canonical_counts[
        "raw_materialized_full_microenvironment_trajectory_count"
    ]
    assert counts["paired_trajectory_count"] == canonical_counts["paired_full_mesoscopic_count"]
    assert counts["train_eligible_count"] == canonical_counts["qc_eligible_trajectory_count"]
    assert counts["g2_status"] == EXPECTED_G2_STATUS
    assert counts["g5_gate_status"] == EXPECTED_G5_GATE_STATUS
    assert freeze_manifest["source_artifacts"]["simulation_qc_summary"]["sha256"]
    readiness_artifact = freeze_manifest["source_artifacts"]["formal_experiment_readiness_audit"]
    g5_artifact = freeze_manifest["source_artifacts"]["g5_effective_sample_audit"]
    assert readiness_artifact["path"] == "Output/audits/formal_experiment_readiness_audit.json"
    assert g5_artifact["path"] == "Output/paper_reports/g5_effective_sample_audit.json"
    assert readiness_artifact["access_mode"] in {"worktree", "external_read_only"}
    assert g5_artifact["access_mode"] in {"worktree", "external_read_only"}
    assert freeze_manifest["config_hashes"]["resolved_config_sha256"]
    assert freeze_manifest["git_state"]["dirty_worktree_allowed"] is True
    command_ids = {row["command_id"] for row in freeze_manifest["slurm_command_manifest"]}
    assert {"formal_freeze_replay_gate", "gpu_stochastic_smoke", "cpu_full_tests", "formal_submission"} <= command_ids
    freeze_gate = [
        row for row in freeze_manifest["slurm_command_manifest"]
        if row["command_id"] == "formal_freeze_replay_gate"
    ][0]
    freeze_gate_command = " ".join(str(value) for value in freeze_gate["command"])
    assert "validate_formal_experiment_freeze.py" in freeze_gate_command
    assert "--no-write" in freeze_gate["command"]
    markdown = markdown_report(freeze_manifest)
    assert str(freeze_manifest["config_hashes"]["resolved_config_sha256"]) in markdown
    assert str(freeze_manifest["snapshot_signature"]["git_full_diff_hash"]) in markdown


def test_freeze_validation_rejects_g5_overclaim(freeze_manifest: dict[str, object]) -> None:
    mutated = deepcopy(freeze_manifest)
    mutated["counts_and_statuses"]["g5_gate_status"] = "passed"
    mutated["counts_and_statuses"]["g5_materialized_trajectory_gate_status"] = "passed"
    errors = validation_errors(mutated)
    assert any("g5_gate_status" in error for error in errors)
    assert any("g5_materialized_trajectory_gate_status" in error for error in errors)


def test_freeze_validation_rejects_full_only_main_pretraining(freeze_manifest: dict[str, object]) -> None:
    mutated = deepcopy(freeze_manifest)
    mutated["training_exclusion"]["full_only_main_pretraining_row_count"] = 1
    mutated["training_exclusion"]["exclusion_status"] = "fail"
    errors = validation_errors(mutated)
    assert "training dataset loader exclusion contract failed" in errors


def test_read_only_artifact_linkage_requires_hash_bound_explicit_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifact-root"
    source_path = artifact_root / "Output/audits/readiness.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"status":"real-audit"}\n', encoding="utf-8")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config = {
        "iclr_design_contract_union": {
            "g0_read_only_artifact_linkage": {
                "linkage_id": "test-linkage",
                "environment_variable": "TEST_AUDIT_ARTIFACT_ROOT",
                "artifacts": {
                    "readiness": {
                        "logical_path": "Output/audits/readiness.json",
                        "sha256": digest,
                    }
                },
            }
        }
    }
    with pytest.raises(FileNotFoundError, match="TEST_AUDIT_ARTIFACT_ROOT"):
        resolve_read_only_artifact(config, "readiness", tmp_path / "isolated-worktree")
    monkeypatch.setenv("TEST_AUDIT_ARTIFACT_ROOT", str(artifact_root))

    reference = resolve_read_only_artifact(config, "readiness", tmp_path / "isolated-worktree")

    assert reference.access_mode == "external_read_only"
    assert reference.logical_path == "Output/audits/readiness.json"
    assert reference.sha256 == digest
    assert str(artifact_root) not in reference.logical_path
    config["iclr_design_contract_union"]["g0_read_only_artifact_linkage"]["artifacts"]["readiness"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_read_only_artifact(config, "readiness", tmp_path / "isolated-worktree")
