from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.platform_contracts import (
    measurement_semantics_rows,
    preferred_target_program_lookup,
    sample_observation_rows,
    validation_rerank_frame,
)
from Libs.utils.data.raw_data_registry import load_config


def test_sample_observation_contract_is_sample_centric() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = sample_observation_rows(config)
    assert rows
    assert "subject_id" not in rows[0]
    assert "time_offset_days" not in rows[0]
    assert rows[0]["semantics_uid"]
    assert rows[0]["measurement_semantics_label"]
    assert rows[0]["compatibility_alias_flag"] == "no"


def test_measurement_semantics_catalog_rows_are_registry_backed() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = measurement_semantics_rows(config)
    assert rows
    assert rows[0]["semantics_uid"]
    assert rows[0]["raw_unit"]
    assert rows[0]["canonical_unit"]
    assert rows[0]["unit_conversion_chain"]
    assert rows[0]["standardization_statistics_path"].endswith("feature_scaler_artifact.tsv")
    assert rows[0]["standardization_summary"]
    assert rows[0]["uncertainty_semantics"]
    assert rows[0]["inverse_transform_artifact_path"].endswith("feature_scaler_artifact.tsv")
    assert rows[0]["proxy_unit_flag"] == "yes"


def test_preferred_target_program_lookup_prefers_direct_program_rows() -> None:
    lookup = preferred_target_program_lookup(
        [
            {
                "target_program_uid": "atlas_program",
                "canonical_entity_id": "101",
                "canonical_entity_label": "APP",
                "source_repository": "atlas+drug_target",
                "target_evidence_score": "0.9",
                "mapped_feature_count": "32",
                "coverage_fraction": "0.8",
            },
            {
                "target_program_uid": "direct_program",
                "canonical_entity_id": "101",
                "canonical_entity_label": "APP",
                "source_repository": "drug_target_direct",
                "target_evidence_score": "0.2",
                "mapped_feature_count": "0",
                "coverage_fraction": "0.0",
            },
        ]
    )
    assert lookup["101"]["target_program_uid"] == "direct_program"


def test_validation_rerank_frame_combines_model_tri_and_validation_signals() -> None:
    candidate_frame = pd.DataFrame(
        [
            {"intervention_program_uid": "int_alpha", "route": "oral"},
            {"intervention_program_uid": "int_beta", "route": "intravenous"},
        ]
    )
    validation_frame = pd.DataFrame(
        [
            {
                "intervention_program_uid": "int_alpha",
                "validation_stage": "knowledge_ready",
                "affinity_evidence_score": 0.1,
                "admet_readiness_score": 0.0,
                "pbpk_readiness_score": 0.0,
                "structure_confidence": "",
            },
            {
                "intervention_program_uid": "int_beta",
                "validation_stage": "knowledge_profiled",
                "affinity_evidence_score": 0.8,
                "admet_readiness_score": 0.7,
                "pbpk_readiness_score": 0.9,
                "structure_confidence": "",
            },
        ]
    )
    reranked = validation_rerank_frame(
        candidate_frame,
        predicted_scores=[0.4, 0.6],
        tri_scores=[0.4, 0.9],
        validation_frame=validation_frame,
        weights={
            "model_score": 0.4,
            "tri_score": 0.25,
            "affinity_evidence": 0.15,
            "admet_readiness": 0.1,
            "pbpk_readiness": 0.1,
        },
    )
    assert reranked.iloc[0]["intervention_program_uid"] == "int_beta"
    assert reranked.iloc[0]["rerank_score"] > reranked.iloc[1]["rerank_score"]
