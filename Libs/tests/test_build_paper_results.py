from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.reports.build_paper_results import (
    dynamics_ablation_rows,
    dynamics_mainline_rows,
    foundation_capability_summary_rows,
    gpu_stochastic_smoke_summary,
    pending_formal_result_surfaces,
    publication_ablation_result_rows,
    publication_baseline_result_rows,
    publication_claim_metric_rows,
    publication_selective_risk_rows,
    runnable_baseline_rows,
)
from Libs.scripts.reports.build_framework_hardening_audits import (
    crossscale_alignment_audit,
    dynamics_calibration_audit,
    measurement_semantics_audit,
    scaler_leakage_audit,
)
from Libs.utils.exp.experiment_manifests import load_paper_track
from Libs.utils.data.raw_data_registry import load_config


def test_paper_results_use_current_formal_profiles_only() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    paper_track = load_paper_track("research_hardening")
    mainline_rows = dynamics_mainline_rows(config, paper_track)
    assert mainline_rows
    assert mainline_rows[0]["status"] in {"available", "pending"}
    assert "::" not in str(mainline_rows[0]["artifact_profile_name"])

    ablation_rows = dynamics_ablation_rows(config, paper_track)
    assert ablation_rows
    assert all("::" not in str(row["artifact_profile_name"]) for row in ablation_rows)


def test_foundation_capability_summary_rows_return_named_capabilities() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    paper_track = load_paper_track("research_hardening")
    rows = foundation_capability_summary_rows(config, paper_track)
    assert rows
    capability_names = {str(row["capability_name"]) for row in rows}
    assert "structured_world_state" in capability_names


def test_gpu_stochastic_smoke_summary_preserves_or_marks_pending_artifact() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    payload = gpu_stochastic_smoke_summary(config)
    assert payload["required_sample_count"] == 64
    assert payload["required_denoising_step_count"] == 100
    assert payload["smoke_status"] in {"passed", "configured_pending_gpu03_execution"}


def test_framework_hardening_audits_expose_semantics_scaler_dynamics_and_alignment() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    semantics_payload = measurement_semantics_audit(config)
    scaler_payload = scaler_leakage_audit(config)
    dynamics_payload = dynamics_calibration_audit(config)
    alignment_payload = crossscale_alignment_audit(config)
    assert semantics_payload["semantics_row_count"] > 0
    assert scaler_payload["bundle_count"] > 0
    assert scaler_payload["all_inverse_transform_artifacts_declared"] is True
    assert scaler_payload["all_leakage_guards_match_policy"] is True
    assert "calibration_pending" in dynamics_payload["calibration_statuses"]
    assert dynamics_payload["external_calibration_claim_status"] == "not_claimed"
    assert alignment_payload["paired_anchor_count"] >= 1
    assert alignment_payload["full_crossscale_alignment_claim_status"] == "not_claimed"


def test_runnable_baselines_with_nan_required_metrics_are_not_headline_eligible() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    rows = runnable_baseline_rows(config)
    row_by_model = {str(row["model_name"]): row for row in rows}
    assert row_by_model["mamba_neural_operator_adapter"]["comparison_status"] == "incomplete_runtime_contract"
    assert row_by_model["coda_no_adapter"]["comparison_status"] == "incomplete_runtime_contract"


def test_publication_main_table_gate_rejects_unmaterialized_formal_surfaces() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    paper_track = load_paper_track("research_hardening")
    status_rows = [
        {
            "profile_name": "dynamics_mainline_ad",
            "available_seed_count": 0,
            "status": "pending",
            "artifact_profile_name": "",
        }
    ]
    pending_surfaces = pending_formal_result_surfaces(
        config,
        paper_track,
        status_rows,
        publication_claim_metric_rows(config),
        publication_baseline_result_rows(config),
        publication_ablation_result_rows(config),
        publication_selective_risk_rows(config),
    )
    assert "formal_dynamics_profiles:dynamics_mainline_ad" in pending_surfaces
    assert any(surface.startswith("claim_metric_catalog:") for surface in pending_surfaces)
    assert any(surface.startswith("baseline_result_matrix:") for surface in pending_surfaces)
    assert any(surface.startswith("ablation_result_catalog:") for surface in pending_surfaces)
    assert any(surface.startswith("selective_risk_catalog:") for surface in pending_surfaces)
