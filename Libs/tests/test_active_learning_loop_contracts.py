from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.active_learning.preflight_active_learning_loop import validate_active_learning_contract
from Libs.scripts.active_learning.build_active_learning_priority_catalog import priority_rows
from Libs.scripts.services import serve_screening_api
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.active_learning_loop import (
    build_active_candidate_pool,
    build_completed_result_manifest,
    quota_counts,
    score_acquisition_tasks,
    select_request_batch,
)


def test_active_learning_yaml_contract_declares_full_loop_requirements() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    active_config = config["screening_pipeline"]["screening_benchmark"]["active_learning"]
    report = validate_active_learning_contract(config)
    assert report["contract_name"] == "decision_centric_simulator_portfolio_active_learning"
    assert active_config["full_round_count"] == 3
    assert active_config["full_tasks_per_round"] == 128
    assert quota_counts(128, active_config["batch_quota"]) == {
        "exploitation": 51,
        "exploration": 51,
        "sentinel": 26,
    }
    assert "lincs_clue_sigcom" in active_config["mandatory_public_sources"]
    assert set(active_config["mandatory_simulator_families"]) == set(active_config["result_linking"]["artifact_sources"])
    assert set(config["supplemental_milestone_contracts"]["active_learning_loop"]["required_serving_endpoints"]).issubset(
        set(config["screening_pipeline"]["screening_serving"]["public_endpoints"])
    )


def test_active_learning_source_prior_catalog_has_full_batch_capacity() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = priority_rows(config)
    active_config = config["screening_pipeline"]["screening_benchmark"]["active_learning"]
    assert len(rows) * len(active_config["simulator_portfolio"]) >= active_config["full_tasks_per_round"]
    for row in rows[:16]:
        assert np.isfinite(float(row["predicted_score_mean"]))
        assert np.isfinite(float(row["tri_score_mean"]))
        assert np.isfinite(float(row["active_learning_uncertainty"]))


def test_active_learning_candidate_pool_and_acquisition_are_finite() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    intervention_frame = pd.DataFrame(
        [
            {
                "intervention_program_uid": f"intervention_{index}",
                "regimen_uid": f"regimen_{index}",
                "target_program_set": f"target_program_{index % 2}",
                "target_program_label_set": f"TARGET{index % 2}",
                "compound_set": f"compound_{index}",
                "route": "iv" if index % 2 == 0 else "sc",
                "schedule": "q2w",
                "dose_level_id": f"dose_{index}",
                "selection_reason": "pytest_active_learning_fixture",
                "evidence_summary": "fixture",
            }
            for index in range(6)
        ]
    )
    validation_frame = pd.DataFrame(
        [
            {
                "intervention_program_uid": f"intervention_{index}",
                "validation_stage": "knowledge_profiled" if index % 2 == 0 else "unvalidated",
                "affinity_evidence_score": 0.4 + 0.05 * index,
                "admet_readiness_score": 0.5,
                "pbpk_readiness_score": 0.6,
            }
            for index in range(6)
        ]
    )
    priority_frame = pd.DataFrame(
        [
            {
                "regimen_uid": f"regimen_{index}",
                "predicted_score_mean": 0.35 + 0.08 * index,
                "tri_score_mean": 0.30 + 0.06 * index,
                "pkpd_consistency_mean": 0.45 + 0.04 * index,
                "epistemic_std_mean": 0.05 + 0.02 * index,
                "predictive_entropy_mean": 0.08 + 0.03 * index,
                "mutual_information_mean": 0.04 + 0.01 * index,
            }
            for index in range(6)
        ]
    )
    candidate_frame = build_active_candidate_pool("pytest_round", intervention_frame, validation_frame, priority_frame, config)
    acquisition_frame = score_acquisition_tasks("pytest_round", candidate_frame, config)
    assert candidate_frame.shape[0] == 6
    assert acquisition_frame.shape[0] == 6 * len(config["screening_pipeline"]["screening_benchmark"]["active_learning"]["simulator_portfolio"])
    assert np.isfinite(pd.to_numeric(acquisition_frame["acquisition_score"]).to_numpy(dtype=np.float64)).all()


def test_active_learning_batch_selection_preserves_quota_and_request_schema() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    candidate_frame = pd.DataFrame(
        [
            {
                "candidate_uid": f"candidate_{index}",
                "sample_state_uid": "sample_state",
                "intervention_program_uid": f"intervention_{index}",
                "regimen_uid": f"regimen_{index}",
                "target_program_set": f"target_program_{index}",
                "compound_set": f"compound_{index}",
                "route": "iv",
                "schedule": "q2w",
                "dose_level_id": f"dose_{index}",
                "source_holdout_role": "active_learning_candidate",
            }
            for index in range(6)
        ]
    )
    acquisition_rows = []
    roles = ["exploitation", "exploitation", "exploration", "exploration", "sentinel", "sentinel"]
    for index, role in enumerate(roles):
        acquisition_rows.append(
            {
                "round_uid": "pytest_round",
                "request_uid": f"request_{index}",
                "candidate_uid": f"candidate_{index}",
                "batch_role": role,
                "simulator_family": "full_microenvironment",
                "fidelity_level": "L4",
                "request_role": "full_microenvironment_teacher",
                "training_label_policy": "teacher_family",
                "acquisition_score": 1.0 - 0.01 * index,
                "exploitation_score": 1.0 - 0.01 * index,
                "exploration_score": 0.8 - 0.01 * index,
                "sentinel_score": 0.6 - 0.01 * index,
                "expected_information_gain": 0.5,
                "rank_flip_risk": 0.4,
                "tri_uncertainty": 0.3,
                "counterfactual_disagreement": 0.2,
                "ood_gap": 0.1,
                "mechanism_expert_disagreement": 0.2,
                "diversity_bonus": 0.1,
                "simulator_cost": 1.0,
                "risk_penalty": 0.1,
                "decision_label": "active_learning",
                "route": "iv",
                "schedule": "q2w",
                "target_program_set": f"target_program_{index}",
                "artifact_path": "pytest",
            }
        )
    request_frame = select_request_batch(
        "pytest_round",
        pd.DataFrame(acquisition_rows),
        candidate_frame,
        config,
        total_count=5,
    )
    assert request_frame.shape[0] == 5
    assert set(request_frame.columns) == set(config["schemas"]["active_learning_simulation_request_fields"])
    assert request_frame["time_grid"].nunique() == 1
    assert request_frame["time_grid"].iloc[0] == config["screening_pipeline"]["screening_benchmark"]["active_learning"]["simulation_request_defaults"]["time_grid"]


def test_active_learning_result_linker_requires_materialized_artifact(tmp_path: Path) -> None:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    artifact_path = tmp_path / "lincs_signature_catalog.tsv"
    artifact_path.write_text("signature_uid\tvalue\nsig_alpha\t1.0\n", encoding="utf-8")
    for adapter_row in config["validation"]["external_adapter_registry"]["adapters"]:
        if adapter_row["source_key"] == "lincs_clue_sigcom":
            adapter_row["processed_artifact_path"] = artifact_path.as_posix()
    request_frame = pd.DataFrame(
        [
            {
                "request_uid": "request_alpha",
                "round_uid": "pytest_round",
                "candidate_uid": "candidate_alpha",
                "intervention_program_uid": "intervention_alpha",
                "sample_state_uid": "sample_alpha",
                "regimen_uid": "regimen_alpha",
                "target_program_set": "target_program_alpha",
                "compound_set": "compound_alpha",
                "route": "iv",
                "schedule": "q2w",
                "dose_level_id": "dose_alpha",
                "simulator_family": "public_evidence_refresh",
                "fidelity_level": "L0",
                "request_role": "evidence_refresh",
                "training_label_policy": "not_dynamics_label",
                "requested_outputs": "evidence_refresh",
                "time_grid": "time_grid",
                "region_grid": "region_grid",
                "compartment_grid": "compartment_grid",
                "species_list": "species",
                "parameter_prior_uid": "prior",
                "uncertainty_policy": "teacher_uncertainty_required",
                "seed": "seed",
                "budget_class": "exploration",
                "source_holdout_role": "source_holdout",
                "execution_status": "completed",
                "artifact_path": "request.tsv",
            }
        ]
    )
    result_frame = build_completed_result_manifest("pytest_round", request_frame, config)
    assert result_frame["qc_grade"].iloc[0] == "C"
    assert result_frame["qc_use"].iloc[0] == "weak_teacher_calibration"
    assert result_frame["trajectory_artifact_path"].iloc[0] == artifact_path.as_posix()
    assert len(result_frame["trajectory_hash"].iloc[0]) == 64


def test_active_learning_serving_surface_contains_required_handlers() -> None:
    service_text = (REPO_ROOT / "Libs/scripts/services/serve_screening_api.py").read_text(encoding="utf-8")
    for endpoint_name in [
        "/plan_active_learning_batch",
        "/dispatch_validation_tasks",
        "/ingest_simulation_result",
        "/active_learning_round_report",
    ]:
        assert endpoint_name in service_text
    assert hasattr(serve_screening_api, "plan_active_learning_batch")
    assert hasattr(serve_screening_api, "dispatch_validation_tasks")
