from __future__ import annotations

from copy import deepcopy

from Libs.utils.evaluation.operational_twin_development_evidence import (
    FULL_METHOD_ID,
    build_development_comparisons,
    family_stratified_paired_bootstrap,
    holm_adjust,
    summarize_counterfactual_pairs,
    training_budget_evidence,
    validate_assignment_development_evidence,
)


def test_training_budget_separates_selected_checkpoint_from_executed_epochs() -> None:
    status = {
        "epochs_completed": 285,
        "selected_checkpoint_epoch": 285,
        "steps_completed": 700,
    }
    rows = [{"epoch": epoch} for epoch in range(1, 351)]
    evidence = training_budget_evidence(
        status,
        {"epochs": 350, "max_steps": 65000},
        rows,
    )
    assert evidence["full_budget_completed"] is True
    assert evidence["completion_mode"] == "epoch_budget"
    assert evidence["observed_epochs_completed"] == 350
    assert evidence["selected_checkpoint_epoch"] == 285


def test_training_budget_accepts_exact_max_step_exhaustion() -> None:
    evidence = training_budget_evidence(
        {"epochs_completed": 10, "steps_completed": 50},
        {"epochs": 100, "max_steps": 50},
        [{"epoch": 10}],
    )
    assert evidence["full_budget_completed"] is True
    assert evidence["completion_mode"] == "max_step_budget"


def test_counterfactual_pair_summary_computes_regret_and_direction() -> None:
    worlds = [
        {
            "role": "development",
            "assignment_uid": "a",
            "method_id": "m",
            "seed": 1,
            "world_uid": "w0",
            "family_uid": "f",
            "trajectory_rmse": 0.2,
            "reference_utility": 1.0,
            "predicted_utility": 0.0,
        },
        {
            "role": "development",
            "assignment_uid": "a",
            "method_id": "m",
            "seed": 1,
            "world_uid": "w1",
            "family_uid": "f",
            "trajectory_rmse": 0.4,
            "reference_utility": 0.0,
            "predicted_utility": 1.0,
        },
    ]
    pairs = [
        {
            "approved_pair": True,
            "counterfactual_pair_uid": "p",
            "latent_world_uid": "l",
            "family_uid": "f",
            "member_world_uids": ["w0", "w1"],
        }
    ]
    result = summarize_counterfactual_pairs(worlds, pairs)[0]
    assert result["intervention_ranking_regret"] == 1.0
    assert result["action_contrast_error"] == 2.0
    assert result["effect_direction_correct"] is False
    assert result["effect_direction_error"] == 1.0


def test_bootstrap_and_holm_are_deterministic_and_monotone() -> None:
    rows = [
        {"family_uid": "a", "improvement": 1.0},
        {"family_uid": "a", "improvement": 2.0},
        {"family_uid": "b", "improvement": 3.0},
        {"family_uid": "b", "improvement": 4.0},
    ]
    left = family_stratified_paired_bootstrap(
        rows,
        value_key="improvement",
        resamples=100,
        seed=17,
        confidence_level=0.95,
    )
    right = family_stratified_paired_bootstrap(
        rows,
        value_key="improvement",
        resamples=100,
        seed=17,
        confidence_level=0.95,
    )
    assert left == right
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_development_comparison_aggregates_seeds_before_world_statistics() -> None:
    rows = []
    for method_id, offset in ((FULL_METHOD_ID, 0.1), ("persistence_no_action", 0.6), ("same_backbone_no_action", 0.4)):
        for seed in (360101, 360211, 360307):
            for pair_index, family_uid in enumerate(("f0", "f1")):
                rows.append(
                    {
                        "method_id": method_id,
                        "seed": seed,
                        "counterfactual_pair_uid": f"p{pair_index}",
                        "family_uid": family_uid,
                        "action_contrast_error": offset,
                        "intervention_ranking_regret": offset,
                        "trajectory_rmse": offset,
                        "effect_direction_error": 0.0 if method_id == FULL_METHOD_ID else 1.0,
                    }
                )
    policy = {
        "bootstrap": {"resamples": 100, "confidence_level": 0.95},
        "conditional_permutation": {"permutations": 100},
        "analysis_seed": 9,
    }
    result = build_development_comparisons(rows, deepcopy(policy))
    defined = [row for row in result["comparisons"] if row["status"] == "DEFINED"]
    assert len(defined) == 8
    assert all(row["world_count"] == 2 for row in defined)
    assert all(row["seed_count_per_world"] == [3] for row in defined)
    assert all(row["family_stratified_bootstrap_ci95"]["estimate"] > 0 for row in defined)
    assert result["multiplicity"]["method"] == "holm"


def test_assignment_evidence_rejects_non_uq_uncertainty_values() -> None:
    assignment = {
        "assignment_uid": "a",
        "method_id": "persistence_no_action",
        "seed": 1,
        "runtime_contract": {"calibration_eligible": False},
    }
    summary = {
        "schema_version": "neuronsifter-operational-twin-development-evidence-v1",
        "status": "PASS",
        "assignment_uid": "a",
        "method_id": "persistence_no_action",
        "seed": 1,
        "development_only": True,
        "calibration_world_only": True,
        "formal_qualification": False,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "device_provenance": {
            "slurm_job_id": "1",
            "hostname": "cpu04",
            "device_type": "cpu",
            "cuda_available": True,
        },
    }
    world = {
        "assignment_uid": "a",
        "method_id": "persistence_no_action",
        "seed": 1,
        "family_uid": "f",
        "gaussian_nll": None,
        "coverage_95": None,
        "predictive_scale_mean": None,
    }
    pair = {
        "assignment_uid": "a",
        "method_id": "persistence_no_action",
        "seed": 1,
        "family_uid": "f",
    }
    support = {
        "status_match": True,
        "abstention_compliant": True,
    }
    blockers = validate_assignment_development_evidence(
        assignment=assignment,
        summary=summary,
        development_world_rows=[deepcopy(world) for _ in range(144)],
        development_pair_rows=[deepcopy(pair) for _ in range(72)],
        calibration_world_rows=[deepcopy(world) for _ in range(72)],
        calibration_pair_rows=[deepcopy(pair) for _ in range(36)],
        support_rows=[deepcopy(support) for _ in range(9)],
    )
    assert blockers == []
    invalid_world = deepcopy(world)
    invalid_world["coverage_95"] = 0.95
    blockers = validate_assignment_development_evidence(
        assignment=assignment,
        summary=summary,
        development_world_rows=[invalid_world, *[deepcopy(world) for _ in range(143)]],
        development_pair_rows=[deepcopy(pair) for _ in range(72)],
        calibration_world_rows=[deepcopy(world) for _ in range(72)],
        calibration_pair_rows=[deepcopy(pair) for _ in range(36)],
        support_rows=[deepcopy(support) for _ in range(9)],
    )
    assert {row["code"] for row in blockers} == {"non_uq_metric_not_null"}
