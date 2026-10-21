from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.services import serve_screening_api
from Libs.exp.export import export_screening_bundle
from Libs.utils.data.platform_contracts import (
    counterfactual_benchmark_rows,
    simulation_teacher_rows,
    world_state_ready_rows,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv


class FakeWorldStateModel:
    world_state_posterior = SimpleNamespace(family="deterministic_compatibility")

    def infer_world_state(self, patient_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        del patient_latent
        return {
            "z_tx": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "z_pk": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            "z_env": torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32),
            "z_ephys": torch.tensor([[0.6]], dtype=torch.float32),
            "z_morph": torch.tensor([[0.7, 0.8]], dtype=torch.float32),
            "hidden_parameter_posterior": torch.tensor([[0.9, 1.0]], dtype=torch.float32),
        }


def test_world_state_ready_rows_expose_five_factor_contract() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = world_state_ready_rows(config)
    assert rows
    factor_manifest = json.loads(rows[0]["factor_slice_manifest"])
    assert set(factor_manifest) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    factor_targets = json.loads(rows[0]["factor_target_payload"])
    assert set(factor_targets) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    factor_masks = json.loads(rows[0]["factor_supervision_mask_payload"])
    assert set(factor_masks) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    factor_sources = json.loads(rows[0]["factor_teacher_source_manifest"])
    assert set(factor_sources) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    factor_evidence = json.loads(rows[0]["factor_evidence_payload"])
    assert set(factor_evidence) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    unit_context = json.loads(rows[0]["unit_context_payload"])
    assert set(unit_context) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    assert rows[0]["semantics_uid"]
    assert rows[0]["measurement_semantics_label"]
    posterior_manifest = json.loads(rows[0]["posterior_slice_manifest"])
    assert set(posterior_manifest) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    assert rows[0]["hidden_state_support_flag"] == "yes"
    assert rows[0]["cross_scene_compatibility_flag"] == "yes"


def test_simulation_teacher_rows_are_solver_grounded() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = simulation_teacher_rows(config)
    assert rows
    assert rows[0]["teacher_family"] == "ad_microenvironment_mechanistic_teacher"
    assert rows[0]["teacher_role"] == "legacy_solver_compatibility_teacher"
    assert rows[0]["calibration_role"] == "compatibility_baseline"
    assert rows[0]["observed_or_synthetic"] == "synthetic_legacy_mesoscale_solver"
    assert rows[0]["teacher_evidence_level"] == "legacy_compatibility"
    assert rows[0]["calibration_status"] == "not_external_calibrated"
    assert rows[0]["main_pretraining_eligibility"] == "calibration_or_compatibility_only"
    assert rows[0]["source_model_id"]
    assert "conservation" in rows[0]["consistency_targets"]
    assert rows[0]["supervision_role"] == "solver_teacher_trajectory"


def test_counterfactual_benchmark_rows_link_distinct_regimens() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = counterfactual_benchmark_rows(config)
    assert rows
    assert any(row["baseline_regimen_uid"] != row["counterfactual_regimen_uid"] for row in rows)


def test_pan_neuro_completeness_reports_incomplete_programs_until_all_contracts_close() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    payload = export_screening_bundle.pan_neuro_completeness(
        config,
        str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"]),
    )
    assert payload["status"] == "incomplete"
    assert {"ftd_control", "als_control"}.issubset(set(payload["incomplete_disease_programs"]))


def test_infer_world_state_returns_factorized_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        serve_screening_api,
        "patient_latent_from_payload",
        lambda bundle, payload: ("transcriptomics", torch.tensor([[0.0, 0.0]], dtype=torch.float32), 0.75),
    )
    bundle = {
        "screening_model": FakeWorldStateModel(),
        "preprocessing_manifest": {
            "active_disease_program": "ad_control",
            "foundation_target_modalities": [
                "transcriptomics",
                "metabolomics",
                "morphology",
                "electrophysiology",
                "pharmacology",
                "solver_observation",
            ],
        },
        "crossmodal_anchor_catalog": pd.DataFrame([{"disease_family": "AD"}]),
        "simulation_teacher_catalog": pd.DataFrame([{"disease_family": "AD", "calibration_status": "calibration_pending"}]),
        "measurement_semantics_catalog": pd.DataFrame(
            [
                {
                    "semantics_uid": "semantics_tx",
                    "modality": "transcriptomics",
                    "raw_unit": "arbitrary_expression_signal",
                    "canonical_unit": "arbitrary_expression_signal",
                    "inverse_transform_artifact_path": "feature_scaler_artifact.tsv",
                    "proxy_unit_flag": "yes",
                }
            ]
        ),
    }
    payload = {"modality": "transcriptomics", "feature_values": {}, "disease_program": "ad_control"}
    result = serve_screening_api.infer_world_state(bundle, payload)
    assert result["disease_family"] == "AD"
    assert set(result["world_state_factors"]) == {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"}
    assert result["hidden_state_support_flag"] is True
    assert result["modality_mask"]["transcriptomics"] is True
    assert result["modality_mask"]["morphology"] is False
    assert result["semantics_uid"] == "semantics_tx"
    assert result["proxy_flag"] is True
    assert result["calibration_status"] == "calibration_pending"


def test_ingest_feedback_persists_event_and_updates_gate() -> None:
    feedback_root = REPO_ROOT / "Output/pytest_feedback_ready_registry"
    artifact_root = REPO_ROOT / "Output/pytest_foundation_feedback"
    if feedback_root.exists():
        shutil.rmtree(feedback_root)
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    config = {
        "data": {
            "feedback_ready_registry": {
                "local_root": "Output/pytest_feedback_ready_registry",
                "relative_paths": {"feedback_event_catalog": "feedback_ready/feedback_event_catalog.tsv"},
            }
        },
        "screening_pipeline": {
            "screening_benchmark": {
                "online_learning": {
                    "feedback_update_policy": {
                        "discrepancy_gate_threshold": 0.15,
                        "minimum_feedback_events_for_parameter_update": 1,
                    },
                    "feedback_artifact_root": "Output/pytest_foundation_feedback",
                }
            }
        },
        "schemas": {
            "feedback_event_fields": [
                "feedback_event_uid",
                "disease_program",
                "disease_family",
                "sample_uid",
                "intervention_program_uid",
                "regimen_uid",
                "predicted_rollout_summary",
                "realized_outcome_summary",
                "discrepancy_summary",
                "calibration_impact_score",
                "update_eligibility_flag",
                "update_decision",
                "event_timestamp",
                "note",
            ]
        },
    }
    bundle = {
        "preprocessing_manifest": {"active_disease_program": "ad_control"},
        "feedback_event_catalog": pd.DataFrame(columns=config["schemas"]["feedback_event_fields"]),
    }
    payload = {
        "sample_uid": "sample_alpha",
        "intervention_program_uid": "intervention_alpha",
        "regimen_uid": "regimen_alpha",
        "predicted_rollout_summary": {"final_state_probability": 0.8},
        "realized_outcome_summary": {"final_state_probability": 0.4},
    }
    result = serve_screening_api.ingest_feedback(bundle, config, payload)
    assert result["update_eligibility_flag"] is True
    assert result["update_decision"] == "parameter_update"
    catalog_path = feedback_root / "feedback_ready/feedback_event_catalog.tsv"
    rows = read_tsv(catalog_path)
    assert len(rows) == 1


def test_rollout_counterfactuals_compares_scenarios(monkeypatch) -> None:
    monkeypatch.setattr(
        serve_screening_api,
        "infer_world_state",
        lambda bundle, payload: {
            "world_state_factors": {},
            "factor_strengths": {},
            "disease_family": "AD",
            "semantics_uid": "semantics_tx",
            "unit_context": {},
            "standardized_value": {},
            "original_unit_value": {},
            "proxy_flag": True,
            "calibration_status": "calibration_pending",
        },
    )
    monkeypatch.setattr(
        serve_screening_api,
        "rollout_regimens",
        lambda bundle, payload: {
            "regimen_sequence": [{"regimen_uid": value} for value in payload["regimen_uids"]],
            "next_state_probability_sequence": [0.8 if payload["regimen_uids"][0] == "baseline_regimen" else 0.6],
            "tri_score_sequence": [0.4 if payload["regimen_uids"][0] == "baseline_regimen" else 0.7],
            "pkpd_consistency_sequence": [0.9],
        },
    )
    result = serve_screening_api.rollout_counterfactuals(
        {},
        {
            "regimen_uids": ["baseline_regimen"],
            "counterfactual_regimen_uids": ["counterfactual_regimen"],
        },
    )
    assert len(result["scenario_rollouts"]) == 2
    assert result["pairwise_comparisons"][0]["delta_final_state_probability"] == -0.20000000000000007


def test_plan_interventions_uses_reranking_and_acquisition(monkeypatch) -> None:
    monkeypatch.setattr(
        serve_screening_api,
        "infer_world_state",
        lambda bundle, payload: {"disease_family": "AD", "world_state_factors": {}, "factor_strengths": {}},
    )
    monkeypatch.setattr(
        serve_screening_api,
        "rerank_with_reliability",
        lambda bundle, config, payload: {
            "reranked_candidates": [
                {
                    "intervention_program_uid": "int_alpha",
                    "rerank_score": 0.9,
                    "predicted_score": 0.8,
                    "tri_score": 0.7,
                    "ranking_stability": 0.85,
                    "rollout_uncertainty": 0.1,
                    "tri_coverage": 0.75,
                    "tri_ood_score": 0.6,
                    "tri_pkpd_physics_consistency": 0.8,
                },
                {
                    "intervention_program_uid": "int_beta",
                    "rerank_score": 0.5,
                    "predicted_score": 0.4,
                    "tri_score": 0.6,
                    "ranking_stability": 0.7,
                    "rollout_uncertainty": 0.2,
                    "tri_coverage": 0.65,
                    "tri_ood_score": 0.5,
                    "tri_pkpd_physics_consistency": 0.7,
                },
            ],
            "decision_summary": {"accept": 1, "request_more_evidence": 1},
        },
    )
    monkeypatch.setattr(
        serve_screening_api,
        "acquire_next_regimens",
        lambda bundle, config, payload: {
            "selected_regimens": [{"regimen_uid": "regimen_alpha", "acquisition_score": 0.8}],
        },
    )
    result = serve_screening_api.plan_interventions({}, {}, {"top_k": 1})
    assert result["planned_interventions"][0]["intervention_program_uid"] == "int_alpha"
    assert result["exploration_batch"][0]["regimen_uid"] == "regimen_alpha"


def test_decision_gate_exposes_five_screening_outcomes() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    gate = config["screening_pipeline"]["screening_benchmark"]["decision_gate"]
    assert serve_screening_api.decision_label(0.8, 0.7, 0.8, 0.2, "rescored", gate) == "accept"
    assert serve_screening_api.decision_label(0.2, 0.7, 0.8, 0.2, "rescored", gate) == "reject"
    assert serve_screening_api.decision_label(0.8, 0.2, 0.8, 0.2, "rescored", gate) == "active_learning"
    assert serve_screening_api.decision_label(0.8, 0.7, 0.2, 0.2, "rescored", gate) == "external_validation"
    assert serve_screening_api.decision_label(0.8, 0.7, 0.8, 0.9, "rescored", gate) == "insufficient_evidence"


def test_explain_decision_reports_factor_attributions(monkeypatch) -> None:
    monkeypatch.setattr(
        serve_screening_api,
        "infer_world_state",
        lambda bundle, payload: {
            "factor_strengths": {
                "z_tx": 1.0,
                "z_pk": 0.5,
                "z_env": 0.7,
                "z_ephys": 0.2,
                "z_morph": 0.4,
            },
            "semantics_uid": "semantics_tx",
            "unit_context": {},
            "standardized_value": {},
            "original_unit_value": {},
            "proxy_flag": True,
            "calibration_status": "calibration_pending",
        },
    )
    monkeypatch.setattr(
        serve_screening_api,
        "rerank_with_reliability",
        lambda bundle, config, payload: {
            "reranked_candidates": [
                {
                    "intervention_program_uid": "int_alpha",
                    "rerank_score": 0.9,
                    "predicted_score": 0.8,
                    "tri_score": 0.7,
                    "ranking_stability": 0.85,
                    "rollout_uncertainty": 0.1,
                    "tri_coverage": 0.75,
                    "tri_ood_score": 0.6,
                    "tri_pkpd_physics_consistency": 0.8,
                },
                {
                    "intervention_program_uid": "int_beta",
                    "rerank_score": 0.5,
                    "predicted_score": 0.4,
                    "tri_score": 0.6,
                    "ranking_stability": 0.7,
                    "rollout_uncertainty": 0.2,
                    "tri_coverage": 0.65,
                    "tri_ood_score": 0.5,
                    "tri_pkpd_physics_consistency": 0.7,
                },
            ],
            "decision_summary": {"accept": 1},
        },
    )
    result = serve_screening_api.explain_decision({}, load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"), {})
    assert len(result["world_state_factor_attributions"]) == 5
    assert result["decision_drivers"][0]["driver"] == "predicted_score"
    assert result["case_level_expert_diagnostics"]["diagnostic_status"] == "not_provided_by_runtime_bundle"


def test_scale_aware_response_adds_route_metadata() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    bundle = {"model_card": {"model_version": "pytest_model"}}
    payload = {"inference_profile": "xl_audit", "publication_case": True}
    response = {"candidate_count": 1, "decision_summary": {"external_validation": 1}}
    result = serve_screening_api.scale_aware_response(bundle, config, payload, response)
    assert result["profile_used"] == "xl_audit"
    assert result["checkpoint_id"] == "pytest_model:xl_audit"
    assert result["scale_route"]["checkpoint_scale"] == "neuronsifter_xl"
    assert result["escalation_status"] == "client_requested_profile"
    assert "student_large_disagreement" in result
    assert "large_xl_disagreement" in result
