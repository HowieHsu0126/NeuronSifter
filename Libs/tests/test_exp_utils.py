from __future__ import annotations

import json
from pathlib import Path

import pytest

from Libs.utils import exp_utils


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_exp_utils_loads_yaml_and_suite_config() -> None:
    suite = exp_utils.load_suite_config(REPO_ROOT / "Libs/configs/exp/paper_grade_full_suite.yaml")
    assert suite["suite_name"] == "Paper-Grade Full Suite v1"
    assert suite["suite_id"] == "ad_paper_grade_full_formal_suite"
    assert suite["suite_version"] == "v1"
    assert suite["seeds"] == [40, 41, 42, 43, 44]


def test_exp_utils_seed_and_task_validation() -> None:
    config = exp_utils.load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    suite = config["paper_grade_full_suite"]
    exp_utils.validate_seed(40, suite["formal_seed_values"])
    exp_utils.validate_task_block("rollout_prediction", suite)
    exp_utils.validate_model_family("neuronsifter_mainline", suite)
    exp_utils.validate_experiment_family("neuronsifter_full", suite)
    with pytest.raises(ValueError):
        exp_utils.validate_seed(39, suite["formal_seed_values"])


def test_exp_utils_aggregation_and_json_hash(tmp_path: Path) -> None:
    first = tmp_path / "seed_40" / "metrics.json"
    second = tmp_path / "seed_41" / "metrics.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(json.dumps({"seed": 40, "metrics": {"score": 1.0}}), encoding="utf-8")
    second.write_text(json.dumps({"seed": 41, "metrics": {"score": 3.0}}), encoding="utf-8")
    aggregate = exp_utils.aggregate_seed_metrics([first, second])
    assert aggregate["seed_count"] == 2
    assert aggregate["metrics"]["score"]["mean"] == 2.0
    assert aggregate["metrics"]["score"]["best"] == 3.0
    assert exp_utils.compute_json_sha256({"a": 1}) == exp_utils.compute_json_sha256({"a": 1})


def test_exp_utils_claim_scan_and_reference_only_status(tmp_path: Path) -> None:
    guardrails = exp_utils.load_yaml(REPO_ROOT / "Libs/configs/exp/claim_guardrails.yaml")["paper_grade_claim_guardrails"]
    assert exp_utils.scan_text_for_forbidden_claims("This does not alter AD-first public cohort evidence.", guardrails) == []
    assert "G5 passed" in exp_utils.scan_text_for_forbidden_claims("G5 passed", guardrails)
    payload = exp_utils.mark_reference_only(tmp_path, "reference adapter lacks runnable state-grid contract")
    assert payload["status"] == "reference_only"
    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / "reference_audit_status.json").exists()
