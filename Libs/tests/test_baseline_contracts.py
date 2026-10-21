from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.baselines.run_runnable_baseline_suite import write_kg_graph_drug_repurposing_baseline
from Libs.scripts.validators.build_baseline_coverage_gap_audit import build_report as build_baseline_gap_report
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_tsv


def test_runnable_baselines_are_not_project_owned_or_smoke() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    runnable_baselines = config["paper_experiments"]["runnable_baselines"]
    assert runnable_baselines
    for row in runnable_baselines:
        assert str(row["adapter_mode"]) != "project_owned_adapter"
        assert "smoke" not in str(row.get("implementation_status", ""))


def test_runnable_baselines_declare_direct_wrapper_commands() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    runnable_baselines = config["paper_experiments"]["runnable_baselines"]
    for row in runnable_baselines:
        adapter_mode = str(row["adapter_mode"])
        assert adapter_mode in {"direct_external_wrapper", "local_matched_split_wrapper"}
        if adapter_mode == "direct_external_wrapper":
            command_templates = row.get("external_command_templates", {})
            assert "train_baseline" in command_templates
            assert "evaluate_baseline" in command_templates


def test_ready_external_baselines_declare_prepare_train_evaluate_templates() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    ready_rows = [
        row
        for row in config["paper_experiments"]["runnable_baselines"]
        if row["implementation_status"] == "direct_external_wrapper_ready"
    ]
    assert {row["baseline_name"] for row in ready_rows} == {
        "t_jepa_adapter",
        "gene_jepa_adapter",
        "mamba_neural_operator_adapter",
        "coda_no_adapter",
        "diffusion_pde_adapter",
        "wdno_adapter",
        "pdediff_adapter",
    }
    for row in ready_rows:
        command_templates = row["external_command_templates"]
        assert set(command_templates) >= {"prepare_inputs", "train_baseline", "evaluate_baseline"}


def test_runnable_baselines_do_not_use_smoke_epoch_policy() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    runnable_baselines = config["paper_experiments"]["runnable_baselines"]
    for row in runnable_baselines:
        hyperparameters = dict(row.get("hyperparameters", {}))
        assert "max_epochs" not in hyperparameters


def test_tjepa_uses_latent_transition_context_representation() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    baseline_by_name = {row["baseline_name"]: row for row in config["paper_experiments"]["runnable_baselines"]}
    tjepa = baseline_by_name["t_jepa_adapter"]
    assert tjepa["bridge_hyperparameters"]["input_representation"] == "latent_transition_context"
    assert tjepa["fold_contract"]["required_test_modality"] == "transcriptomics"


def test_mamba_neural_operator_source_url_matches_declared_baseline() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    baseline_by_name = {row["baseline_name"]: row for row in config["paper_experiments"]["runnable_baselines"]}
    assert baseline_by_name["mamba_neural_operator_adapter"]["source_url"] == "https://arxiv.org/abs/2410.02113"


def test_kg_graph_baseline_is_local_matched_split_not_external_reproduction() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    baseline_by_name = {row["baseline_name"]: row for row in config["paper_experiments"]["runnable_baselines"]}
    kg_baseline = baseline_by_name["kg_graph_drug_repurposing_baseline"]
    assert kg_baseline["adapter_mode"] == "local_matched_split_wrapper"
    assert kg_baseline["implementation_status"] == "local_matched_split_wrapper_ready"
    assert set(kg_baseline["kg_graph_inputs"]) == {
        "target_evidence_catalog",
        "drug_target_edge_catalog",
        "regimen_catalog",
    }
    assert "clinical_direction_case_study" not in kg_baseline


def test_kg_graph_baseline_writes_screening_result_catalog_from_public_graph_inputs(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    baseline_by_name = {row["baseline_name"]: row for row in config["paper_experiments"]["runnable_baselines"]}
    kg_baseline = dict(baseline_by_name["kg_graph_drug_repurposing_baseline"])
    target_path = tmp_path / "target_evidence_catalog.tsv"
    edge_path = tmp_path / "drug_target_edge_catalog.tsv"
    regimen_path = tmp_path / "regimen_catalog.tsv"
    write_tsv(
        target_path,
        ["evidence_uid", "target_uid", "source_repository", "disease_context", "evidence_score", "evidence_count", "public_url", "note"],
        [
            {
                "evidence_uid": "target_a_evidence",
                "target_uid": "target_a",
                "source_repository": "open_targets",
                "disease_context": "Alzheimer disease",
                "evidence_score": 0.8,
                "evidence_count": 4,
                "public_url": "https://platform.opentargets.org/",
                "note": "public_target_evidence",
            }
        ],
    )
    write_tsv(
        edge_path,
        [
            "edge_uid",
            "compound_uid",
            "target_uid",
            "source_repository",
            "disease_context",
            "relation_type",
            "affinity_type",
            "affinity_value",
            "affinity_unit",
            "evidence_score",
            "action_type",
            "public_url",
            "note",
        ],
        [
            {
                "edge_uid": "compound_a_target_a",
                "compound_uid": "compound_a",
                "target_uid": "target_a",
                "source_repository": "bindingdb",
                "disease_context": "Alzheimer disease",
                "relation_type": "measured_affinity",
                "affinity_type": "IC50",
                "affinity_value": 10.0,
                "affinity_unit": "nM",
                "evidence_score": 0.6,
                "action_type": "",
                "public_url": "https://www.bindingdb.org/",
                "note": "public_edge_evidence",
            }
        ],
    )
    write_tsv(
        regimen_path,
        [
            "regimen_uid",
            "drug_program_uid_set",
            "compound_uid_set",
            "target_uid_set",
            "target_symbol_set",
            "regimen_scaffold_set",
            "route",
            "schedule",
            "dose_level_id",
            "dose_multiplier",
            "aggregate_target_evidence_score",
            "aggregate_affinity_score",
            "clue_reversal_proxy_score",
            "pk_feasibility_score",
            "regimen_size",
            "selection_reason",
            "note",
        ],
        [
            {
                "regimen_uid": "regimen_a",
                "drug_program_uid_set": "program_a",
                "compound_uid_set": "compound_a",
                "target_uid_set": "target_a",
                "target_symbol_set": "GENEA",
                "regimen_scaffold_set": "scaffold_a",
                "route": "oral",
                "schedule": "daily",
                "dose_level_id": "dose_100",
                "dose_multiplier": 1.0,
                "aggregate_target_evidence_score": 0.8,
                "aggregate_affinity_score": 0.5,
                "clue_reversal_proxy_score": 0.0,
                "pk_feasibility_score": 0.0,
                "regimen_size": 1,
                "selection_reason": "ranked_selection",
                "note": "single_compound_regimen",
            }
        ],
    )
    kg_baseline["kg_graph_inputs"] = {
        "target_evidence_catalog": target_path.as_posix(),
        "drug_target_edge_catalog": edge_path.as_posix(),
        "regimen_catalog": regimen_path.as_posix(),
    }
    output_root = tmp_path / "kg_output"
    write_kg_graph_drug_repurposing_baseline(config, kg_baseline, "ad_control", "40", output_root)
    result_rows = read_tsv(output_root / "result_catalog.tsv")
    ranked_rows = read_tsv(output_root / "kg_graph_ranked_regimens.tsv")
    assert list(result_rows[0]) == config["schemas"]["screening_result_fields"]
    assert ranked_rows[0]["regimen_uid"] == "regimen_a"
    assert float(ranked_rows[0]["graph_prior_score"]) > 0.0
    assert {row["metric_name"] for row in result_rows} >= {"graph_prior_score", "target_evidence_coverage", "edge_evidence_coverage"}


def test_baseline_coverage_gap_audit_records_required_gap_closure() -> None:
    report = build_baseline_gap_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert report["needs_baseline_completion"] is True
    assert report["requires_refreeze_before_live"] is True
    assert report["missing_required_baselines"] == []
    assert report["main_table_runnable_count"] == 18
    assert "kg_graph_drug_repurposing_baseline" in report["main_table_runnable_recommended_list"]
    assert "pdediff_adapter" in report["main_table_runnable_recommended_list"]
    assert set(report["literature_comparator_list"]) >= {"TxGNN", "KGiA", "ExPDrug", "REx", "HiMAL", "DeepDrug"}


def test_reference_only_world_model_repositories_are_declared_explicitly() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    reference_rows = config["paper_experiments"]["world_model_architecture_references"]
    row_by_name = {str(row["repo_name"]): row for row in reference_rows}
    for repo_name in ["Active-JEPA", "H-JEPA", "I-JEPA", "LeWorldModel", "Point-JEPA"]:
        assert row_by_name[repo_name]["integration_status"] == "reference_only"
    assert row_by_name["AwesomeEHR"]["integration_status"] == "not_vendored_not_integrated"
