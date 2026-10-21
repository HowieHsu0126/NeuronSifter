from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.evaluation.evaluate_screening_system import result_row, strict_case_result_rows
from Libs.utils.evaluation.screening_candidate_selection import (
    diversified_candidate_indices,
    preserved_case_study_regimen_indices,
)
from Libs.utils.evaluation.strict_case_discovery import (
    assert_no_forced_anchor_overlap,
    compute_anchor_recovery_metrics,
    compute_tri_accepted_anchor_metrics,
)


def strict_case_config() -> dict[str, object]:
    return {
        "screening_pipeline": {
            "action_ready_registry": {
                "local_root": "Input/processed",
                "minimum_atlas_program_coverage": 0.5,
                "minimum_atlas_program_feature_hits": 16,
                "maximum_single_compounds": 128,
                "maximum_pairwise_compounds": 24,
                "maximum_pairwise_regimens": 256,
                "forced_include_compounds": [],
                "forced_include_targets": [],
                "relative_paths": {
                    "build_manifest": "action_ready/build_manifest.json",
                },
            },
        },
        "case_study_registry": {
            "ad_amyloid_route": {
                "disease_program": "ad_control",
                "case_study_name": "ad_amyloid_route",
                "reporting_mode": "strict_discovery",
                "forbid_anchor_forced_include": True,
                "anchor_compounds": ["LECANEMAB"],
                "anchor_targets": ["APP", "BACE1"],
                "anchor_references": [],
                "top_k": 25,
                "expected_ranking_checks": [
                    "target_or_compound_anchor_in_top_k",
                    "tri_accepted_anchor_present",
                ],
            },
        },
    }


def test_forced_anchor_overlap_hard_fails() -> None:
    config = strict_case_config()
    config["screening_pipeline"]["action_ready_registry"]["forced_include_targets"] = ["APP"]
    try:
        assert_no_forced_anchor_overlap(config)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected strict overlap validation to fail.")


def test_generic_and_strict_rows_stay_separate() -> None:
    generic_row = result_row(
        "benchmark",
        "ad_control",
        "fold_alpha",
        "regimen_ranking",
        "ad_screening_system",
        "transcriptomics",
        12,
        "ndcg_at_10",
        0.75,
        "artifact.pt",
        "heldout_study",
    )
    strict_rows = strict_case_result_rows(
        "benchmark",
        "ad_control",
        "fold_alpha",
        "strict_case_target_recovery",
        "ad_screening_system",
        "transcriptomics",
        12,
        "artifact.pt",
        "heldout_study",
        "ad_amyloid_route",
        {"anchor_hit_at_top_k": 1.0},
    )
    assert generic_row["case_study_name"] == ""
    assert strict_rows[0]["case_study_name"] == "ad_amyloid_route"
    assert generic_row["evaluation_scope"] == "regimen_ranking"
    assert strict_rows[0]["evaluation_scope"] == "strict_case_target_recovery"


def test_anchor_recovery_metrics_and_tri_acceptance() -> None:
    score_vector = np.asarray([0.9, 0.8, 0.1], dtype=np.float64)
    anchor_mask = np.asarray([False, True, True], dtype=bool)
    tri_accepted_mask = np.asarray([True, True, False], dtype=bool)
    metrics = compute_anchor_recovery_metrics(score_vector, anchor_mask, 2)
    tri_metrics = compute_tri_accepted_anchor_metrics(score_vector, anchor_mask, tri_accepted_mask, 2)
    assert metrics["anchor_candidate_count"] == 2.0
    assert metrics["anchor_hit_at_top_k"] == 1.0
    assert metrics["anchor_count_in_top_k"] == 1.0
    assert metrics["best_anchor_rank"] == 2.0
    assert tri_metrics["tri_accepted_anchor_hit_at_top_k"] == 1.0
    assert tri_metrics["tri_accepted_best_anchor_rank"] == 2.0


def test_diversified_candidate_indices_keep_group_representatives() -> None:
    observed_scores = pd.Series([10.0, 9.0, 8.0, 0.2, 0.1, 0.05], dtype=float).to_numpy()
    prior_scores = pd.Series([0.1, 0.1, 0.1, 5.0, 4.0, 3.0], dtype=float).to_numpy()
    group_keys = ["APP", "APP", "APP", "GLP1R", "GLP1R", "BACE1"]
    indices = diversified_candidate_indices(observed_scores, prior_scores, group_keys, 3)
    assert set(indices.tolist()) == {0, 3, 5}


def test_preserved_case_study_regimen_indices_keep_route_schedule_grid() -> None:
    route_names = ["intrathecal", "intravenous", "oral"]
    schedule_names = ["daily", "twice_daily", "weekly"]
    regimen_rows: list[dict[str, object]] = []
    row_index = 0
    for compound_uid in ["compound_lecanemab", "compound_donanemab"]:
        for route_name in route_names:
            for schedule_name in schedule_names:
                for dose_level_id, pk_score in [("dose_050", 0.40), ("dose_100", 0.90), ("dose_200", 0.60)]:
                    regimen_rows.append(
                        {
                            "regimen_uid": f"regimen_{row_index}",
                            "compound_uid_set": compound_uid,
                            "target_symbol_set": "APP",
                            "route": route_name,
                            "schedule": schedule_name,
                            "dose_level_id": dose_level_id,
                            "pk_feasibility_score": pk_score,
                            "regimen_size": 1,
                            "selection_reason": "case_study_seed",
                        }
                    )
                    row_index += 1
    regimen_frame = pd.DataFrame(regimen_rows)
    strict_case_rows = [
        {
            "anchor_compounds": ["LECANEMAB", "DONANEMAB"],
        }
    ]
    compound_rows = [
        {"compound_uid": "compound_lecanemab", "preferred_name": "LECANEMAB"},
        {"compound_uid": "compound_donanemab", "preferred_name": "DONANEMAB"},
    ]
    indices = preserved_case_study_regimen_indices(regimen_frame, strict_case_rows, compound_rows)
    selected_frame = regimen_frame.iloc[indices].copy()
    assert selected_frame.shape[0] == 18
    assert set(selected_frame["route"].tolist()) == set(route_names)
    assert set(selected_frame["schedule"].tolist()) == set(schedule_names)
    assert set(selected_frame["dose_level_id"].tolist()) == {"dose_100"}
