from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import subprocess

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.evaluation.evaluate_published_ad_sift_reproduction import build_report as build_published_ad_report
from Libs.scripts.validators import validate_multimodal_twin_data_alignment as multimodal_validator
from Libs.scripts.validators import validate_world_foundation_model_claims as world_claim_validator
from Libs.scripts.validators.validate_paper_grade_full_suite import EXPECTED_SUITE_SEEDS
from Libs.scripts.validators.validate_paper_grade_full_suite import build_prelaunch_report
from Libs.utils.data.raw_data_registry import load_config


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_paper_grade_suite_declares_ad_only_seeds_without_mutating_legacy_formal_seeds() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert config["paper_grade_full_suite"]["disease_programs"] == ["ad_control"]
    assert config["paper_grade_full_suite"]["formal_seed_values"] == EXPECTED_SUITE_SEEDS
    assert config["screening_pipeline"]["screening_benchmark"]["formal_seed_values"] == [3407, 42, 26]
    assert "dynamics_mainline_pd" not in config["paper_grade_full_suite"]["submission_order"]["dynamics_profiles"]


def test_paper_grade_suite_prelaunch_and_baseline_registry_pass() -> None:
    report = build_prelaunch_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert report["status"] == "pass"
    assert report["baseline_readiness"]["status"] == "pass"
    assert report["baseline_readiness"]["main_table_runnable_count"] == 18
    assert report["baseline_readiness"]["reference_only_count"] >= 5
    main_table_ids = {
        row["baseline_id"]
        for row in report["baseline_readiness"]["baselines"]
        if row["main_table_runnable"]
    }
    assert {
        "wdno_adapter",
        "pdediff_adapter",
        "kg_graph_drug_repurposing_baseline",
    } <= main_table_ids


def test_paper_grade_suite_rejects_seed_drift(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    config["paper_grade_full_suite"]["formal_seed_values"] = [40, 41, 42]
    config_path = tmp_path / "seed_drift.yaml"
    write_yaml(config_path, config)
    report = build_prelaunch_report(config_path)
    rows = {row["check_id"]: row for row in report["checks"]}
    assert rows["SUITE-002"]["status"] == "fail"


def test_world_foundation_claim_validator_rejects_overclaim(monkeypatch) -> None:
    monkeypatch.setattr(
        world_claim_validator,
        "overclaim_matches",
        lambda config: ["tmp_report.md::direct clinical endpoint prediction"],
    )
    report = world_claim_validator.build_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = {row["check_id"]: row for row in report["checks"]}
    assert rows["WORLD-010"]["status"] == "fail"
    assert report["status"] == "fail"


def test_world_foundation_claim_scan_allows_guardrail_context() -> None:
    markdown_guardrail = (
        "- Do not claim G5 / 1M readiness, general-purpose drug screening, direct clinical endpoint prediction, "
        "private clinical data training, or PD/FTD/ALS readiness proven."
    )
    assert world_claim_validator.guardrail_context(markdown_guardrail, markdown_guardrail.index("ALS readiness"))
    json_guardrail = '{"claim_guardrails": {"forbidden_claims": ["PD/FTD/ALS readiness proven"]}}'
    assert world_claim_validator.guardrail_context(json_guardrail, json_guardrail.index("ALS readiness"))
    assert "world_foundation_model_claims_audit.json" in world_claim_validator.SELF_AUDIT_FILENAMES


def test_world_foundation_claim_scan_excludes_generated_claim_wording_audits() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    matches = world_claim_validator.overclaim_matches(config)

    assert {"claim_gated_wording_audit.tsv", "g0_claim_wording_remediation_report.json"} <= world_claim_validator.SELF_AUDIT_FILENAMES
    assert not any(
        "claim_gated_wording_audit.tsv" in match or "g0_claim_wording_remediation_report.json" in match
        for match in matches
    )


def test_multimodal_alignment_rejects_missing_model_input_registry(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    config["model_input_registry"]["relative_paths"]["study_bundle_catalog"] = "model_input/missing_catalog.tsv"
    config_path = tmp_path / "missing_model_input.yaml"
    write_yaml(config_path, config)
    report = multimodal_validator.build_report(config_path)
    rows = {row["check_id"]: row for row in report["checks"]}
    assert rows["ALIGN-001"]["status"] == "fail"


def test_multimodal_alignment_rejects_unpaired_main_pretraining_rows(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    audit_root = tmp_path / "simulation_audit"
    config["ai_ready_preprocessing"]["simulation_audit"]["local_root"] = str(audit_root)
    write_yaml(tmp_path / "unpaired_config.yaml", config)
    (audit_root / "simulation_qc_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (audit_root / "simulation_qc_summary.json").write_text(
        json.dumps(
            {
                "readiness_status": "G2_pass_with_warnings",
                "paired_count": 0,
                "paired_full_only_count": 1,
                "train_eligible_count": 222336,
                "weak_teacher_count": 0,
                "reject_count": 0,
                "total_split_leakage_count": 0,
            }
        ),
        encoding="utf-8",
    )
    write_tsv(
        audit_root / "simulation_qc_catalog.tsv",
        ["source_uid", "qc_use", "paired_status"],
        [{"source_uid": "full_only_source", "qc_use": "main_pretraining", "paired_status": "full_only"}],
    )
    write_tsv(
        audit_root / "simulation_pair_alignment_audit.tsv",
        ["paired_status", "full_source_uid"],
        [{"paired_status": "full_only", "full_source_uid": "full_only_source"}],
    )
    write_tsv(
        audit_root / "simulation_teacher_uncertainty_audit.tsv",
        ["teacher_uid", "uncertainty_status"],
        [{"teacher_uid": "teacher_a", "uncertainty_status": "available"}],
    )
    report = multimodal_validator.build_report(tmp_path / "unpaired_config.yaml")
    rows = {row["check_id"]: row for row in report["checks"]}
    assert rows["ALIGN-004"]["status"] == "fail"


def test_multimodal_alignment_rejects_missing_teacher_uncertainty(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    families = config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]["families"]
    for family in families:
        if family["family_key"] == "full_microenvironment_teacher":
            family["uncertainty_array_name"] = ""
    config_path = tmp_path / "missing_uncertainty.yaml"
    write_yaml(config_path, config)
    report = multimodal_validator.build_report(config_path)
    rows = {row["check_id"]: row for row in report["checks"]}
    assert rows["ALIGN-005"]["status"] == "fail"


def test_published_ad_sift_reproduction_aggregates_formal_result_catalogs(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    config["paper_grade_full_suite"]["formal_seed_values"] = [40, 41]
    template = tmp_path / "seed_{seed}" / "result_catalog.tsv"
    config["paper_grade_full_suite"]["models"]["neuronsifter_mainline"]["result_catalog_template"] = str(template)
    config_path = tmp_path / "published_ad_config.yaml"
    write_yaml(config_path, config)
    fieldnames = [
        "benchmark_name",
        "disease_program",
        "baseline_family",
        "split_name",
        "evaluation_scope",
        "model_name",
        "modality",
        "sample_count",
        "metric_name",
        "metric_value",
        "artifact_path",
        "case_study_name",
        "scale_classification",
        "is_case_study_anchor",
        "note",
    ]
    for seed, direction_hit in [(40, 1.0), (41, 0.5)]:
        write_tsv(
            Path(str(template).format(seed=seed)),
            fieldnames,
            [
                {
                    "benchmark_name": f"dynamics_twin_ad_mainline_seed_{seed}",
                    "disease_program": "ad_control",
                    "baseline_family": "action_conditioned_twin",
                    "split_name": "source_holdout",
                    "evaluation_scope": "quantitative_case_validation",
                    "model_name": "screening_twin",
                    "modality": "transcriptomics",
                    "sample_count": 10,
                    "metric_name": "direction_hit",
                    "metric_value": direction_hit,
                    "artifact_path": "model.pt",
                    "case_study_name": "ad_amyloid_route",
                    "scale_classification": "",
                    "is_case_study_anchor": "yes",
                    "note": "source|LECANEMAB|literature_direction",
                },
                {
                    "benchmark_name": f"dynamics_twin_ad_mainline_seed_{seed}",
                    "disease_program": "ad_control",
                    "baseline_family": "action_conditioned_twin",
                    "split_name": "source_holdout",
                    "evaluation_scope": "quantitative_case_validation",
                    "model_name": "screening_twin",
                    "modality": "transcriptomics",
                    "sample_count": 10,
                    "metric_name": "rank_recovery_hit",
                    "metric_value": 1.0,
                    "artifact_path": "model.pt",
                    "case_study_name": "ad_amyloid_route",
                    "scale_classification": "",
                    "is_case_study_anchor": "yes",
                    "note": "source|LECANEMAB|literature_direction",
                },
                {
                    "benchmark_name": f"dynamics_twin_ad_mainline_seed_{seed}",
                    "disease_program": "ad_control",
                    "baseline_family": "action_conditioned_twin",
                    "split_name": "source_holdout",
                    "evaluation_scope": "quantitative_case_validation",
                    "model_name": "screening_twin",
                    "modality": "transcriptomics",
                    "sample_count": 10,
                    "metric_name": "tri_gate_consistency",
                    "metric_value": 0.75,
                    "artifact_path": "model.pt",
                    "case_study_name": "ad_amyloid_route",
                    "scale_classification": "",
                    "is_case_study_anchor": "yes",
                    "note": "source|LECANEMAB|literature_direction",
                },
            ],
        )
    report = build_published_ad_report(config_path)
    assert report["status"] == "available"
    assert report["metrics"]["direction_hit_rate"]["mean"] == 0.75
    assert report["metrics"]["pairwise_anchor_accuracy"]["mean"] == 1.0
    assert report["claim_scope"] == "retrospective literature-direction recovery and intervention-prioritization consistency"


def test_paper_grade_submitter_has_required_gates_and_no_pd_submission() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    gate_text = "\n".join(config["paper_grade_full_suite"]["validators"]["prelaunch_order"])
    script_text = (REPO_ROOT / "Libs/exp/submission/submit_paper_grade_full_suite.sh").read_text(encoding="utf-8")
    user_script_text = (REPO_ROOT / "Libs/scripts/experiments/run_paper_grade_full_suite.sh").read_text(encoding="utf-8")
    assert "prelaunch_order" in script_text
    assert "run_prelaunch_gates.sh" in user_script_text
    assert config["paper_grade_full_suite"]["user_entrypoint"] == "Libs/scripts/experiments/run_paper_grade_full_suite.sh"
    assert config["paper_grade_full_suite"]["freeze_binding_policy"]["refreeze_required_after_code_or_config_change"] is True
    assert "validate_simulation_readiness_audit.py" in gate_text
    assert "validate_supplemental_milestone_contracts.py" in gate_text
    assert "validate_formal_experiment_readiness.py" in gate_text
    assert "validate_formal_experiment_freeze.py" in gate_text
    assert "validate_multimodal_twin_data_alignment.py" in gate_text
    assert "validate_world_foundation_model_claims.py" in gate_text
    assert "validate_paper_grade_full_suite.py" in gate_text
    assert "dynamics_mainline_pd" not in config["paper_grade_full_suite"]["submission_order"]["dynamics_profiles"]


NATIVE_REFERENCE_IDS = (
    "action_scalar_pk_exposure_reference",
    "native_initial_state_persistence",
    "neuronsifter_parallel_experts",
    "neuronsifter_reversed_experts",
    "neuronsifter_interleaved_experts",
)


@pytest.fixture(scope="module")
def baseline_catalog_config():
    return load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")


def test_native_reference_readiness_coverage_and_prelive_metadata(baseline_catalog_config):
    from Libs.scripts.validators import build_baseline_coverage_gap_audit as coverage
    from Libs.scripts.validators import build_paper_grade_full_suite_prelive_summary as prelive
    from Libs.scripts.validators import validate_paper_grade_full_suite as validator

    config = baseline_catalog_config
    before = deepcopy(config)
    rows = config["paper_grade_baseline_registry"]["baselines"]
    readiness = validator.build_baseline_readiness_report(config)
    assert validator.baseline_schema_failures(config) == []
    assert readiness["status"] == "pass" and readiness["failures"] == []
    assert readiness["baseline_count"] == 36
    assert readiness["reference_only_count"] == 18
    report = coverage.build_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert prelive.baseline_rows(config) == rows
    assert prelive.reference_only_main_table_baselines(config) == []
    for model in NATIVE_REFERENCE_IDS:
        row = next(row for row in rows if row["baseline_id"] == model)
        assert row["reference_only"] is True
        assert row["main_table_runnable"] is row["appendix_runnable"] is False
        assert "pending" in row["reason_if_not_runnable"]
        reported = next(item for item in readiness["baselines"] if item["baseline_id"] == model)
        for key in ("reference_only", "main_table_runnable", "appendix_runnable", "reason_if_not_runnable"):
            assert reported[key] == row[key]
        assert model in report["coverage_by_category"][row["baseline_family"]]["baseline_ids"]
        assert row["baseline_family"] in prelive.baseline_families(config)
        assert model not in report["main_table_runnable_recommended_list"]
        assert model not in report["appendix_runnable_detail_list"]
    categories = report["coverage_by_category"].values()
    assert sum(row["reference_only_count"] for row in categories) == 18
    assert sum(len(row["baseline_ids"]) for row in categories) == len(rows)
    assert report["needs_baseline_completion"] is report["requires_refreeze_before_live"] is True
    assert config == before
    print("CANONICAL_CATALOG", {"schema_failures": readiness["failures"],
                               "baselines": readiness["baseline_count"],
                               "reference_only": readiness["reference_only_count"]})


@pytest.mark.parametrize("field,value,failure", [
    ("reference_only", False, "non_reference_baseline_has_no_table"),
    ("main_table_runnable", True, "reference_only_cannot_enter_main_table"),
    ("reason_if_not_runnable", "", "reference_only_requires_reason"),
])
def test_native_reference_schema_rejects_invalid_classifications(
    baseline_catalog_config, field, value, failure
):
    from Libs.scripts.validators.validate_paper_grade_full_suite import baseline_schema_failures

    for model in NATIVE_REFERENCE_IDS:
        config = deepcopy(baseline_catalog_config)
        row = next(row for row in config["paper_grade_baseline_registry"]["baselines"]
                   if row["baseline_id"] == model)
        row[field] = value
        assert baseline_schema_failures(config) == [f"{model}:{failure}"]


def test_reference_classification_preserves_complete_required_manifest_denominator(baseline_catalog_config):
    from Libs.scripts.validators import build_paper_grade_full_suite_prelive_summary as prelive
    from Libs.scripts.validators import validate_paper_grade_full_suite as validator

    config = baseline_catalog_config
    suite = config["paper_grade_full_suite"]
    original_suite = deepcopy(suite)
    expected = [
        {"task_name": task["task_name"], "model_key": model, "seed": seed,
         "manifest_path": suite["manifest_contract"]["manifest_template"].format(
             task_name=task["task_name"], model_key=model, seed=seed)}
        for task in suite["tasks"] for model in task["required_models"] for seed in suite["formal_seed_values"]
    ]
    assert validator.expected_manifest_entries(config) == expected
    prior_classification = deepcopy(config)
    for row in prior_classification["paper_grade_baseline_registry"]["baselines"]:
        if row["baseline_id"] in NATIVE_REFERENCE_IDS:
            row["reference_only"] = False
    assert validator.expected_manifest_entries(prior_classification) == expected
    assert prior_classification["paper_grade_full_suite"] == original_suite
    assert prelive.task_blocks(config) == prelive.task_blocks(prior_classification)
    assert prelive.expected_output_artifacts(config) == prelive.expected_output_artifacts(prior_classification)
    assert suite == original_suite
    assert suite["formal_seed_values"] == [40, 41, 42, 43, 44]
    assert config["screening_pipeline"]["screening_benchmark"]["formal_seed_values"] == [3407, 42, 26]
    report = validator.build_prelaunch_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert report["status"] == "pass"
    execution = next(row for row in report["checks"] if row["check_id"] == "SUITE-009")
    assert execution["status"] == "pending"
    print("PRESERVED_REQUIRED_MANIFESTS", len(expected), "SUITE-009", execution["status"])


@pytest.mark.parametrize("postrun_status", ["reference_only", "pending", "dry_run"])
def test_completion_cli_rejects_required_reference_comparisons_and_keeps_missing_entries(
    baseline_catalog_config, tmp_path, postrun_status
):
    from Libs.scripts.validators import validate_paper_grade_full_suite as validator

    config = deepcopy(baseline_catalog_config)
    config["selected_formal_seed"] = 41
    config["screening_pipeline"]["screening_benchmark"]["training"].update(
        seed_policy="manuscript_native_five", random_seed=41)
    config["logging_registry"].update(file_enabled=False, console_enabled=False,
                                      root_dir=str(tmp_path / "logs"))
    config["execution_environment"]["python_executable"] = sys.executable
    suite = config["paper_grade_full_suite"]
    # A disposable required-comparison roster exercises all five IDs. The real
    # legacy task roster is preserved and is not the full prospective native matrix.
    task = deepcopy(next(task for task in suite["tasks"] if task["task_name"] == "rollout_prediction"))
    task["required_models"] = list(NATIVE_REFERENCE_IDS)
    suite["tasks"] = [task]
    suite["manifest_contract"]["manifest_template"] = str(
        tmp_path / "manifests/{task_name}/{model_key}/seed_{seed}/artifact_manifest.json")
    suite["output_paths"] = {
        key: str(tmp_path / "reports" / (key + Path(value).suffix))
        for key, value in suite["output_paths"].items()
    }
    entries = validator.expected_manifest_entries(config)
    assert len(entries) == len(NATIVE_REFERENCE_IDS) * len(suite["formal_seed_values"]) == 25
    first_seed = suite["formal_seed_values"][0]
    for entry in entries:
        path = Path(entry["manifest_path"])
        assert path.is_relative_to(tmp_path)
        if entry["seed"] == first_seed:
            continue
        payload = {field: "" for field in suite["manifest_contract"]["required_fields"]}
        payload.update(task_name=entry["task_name"], model_key=entry["model_key"],
                       seed=entry["seed"], postrun_status=postrun_status)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    config_path = tmp_path / "required-comparisons.yaml"
    write_yaml(config_path, config)
    expected_failures = {
        f"missing_manifest:{entry['manifest_path']}" if entry["seed"] == first_seed
        else f"{entry['manifest_path']}:postrun_status={postrun_status}"
        for entry in entries
    }
    direct = validator.build_postrun_report(config_path)
    assert direct["status"] == "fail"
    assert direct["artifact_manifest_count"] == direct["failure_count"] == 25
    assert set(direct["failures"]) == expected_failures
    assert {row["model_key"] for row in direct["artifact_manifests"]} == set(NATIVE_REFERENCE_IDS)
    for row in direct["artifact_manifests"]:
        assert row["status"] == ("missing" if row["seed"] == first_seed else postrun_status)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "Libs/exp/aggregation/build_completion_audit.py"),
         "--config", str(config_path), "--seed", "41",
         "--freeze-manifest", str(tmp_path / "absent-freeze.json"),
         "--output-dir", str(tmp_path / "completion")],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    execution = json.loads(Path(suite["output_paths"]["execution_summary_json"]).read_text())
    summary = json.loads(Path(suite["output_paths"]["paper_grade_result_summary_json"]).read_text())
    assert execution == direct
    assert Path(suite["output_paths"]["execution_summary_md"]).is_file()
    assert summary["suite_execution_status"] == "fail"
    assert summary["artifact_manifest_count"] == summary["failure_count"] == 25
    assert set(summary["failures"]) == expected_failures
    assert not (tmp_path / "completion").exists()
