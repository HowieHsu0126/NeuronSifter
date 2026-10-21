from __future__ import annotations

from pathlib import Path

import yaml

from Libs.scripts.validators.build_experiment_entrypoint_inventory import build_inventory_report
from Libs.utils.data.raw_data_registry import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TASKS = {
    "world_model_pretraining",
    "screening_supervised",
    "rollout_prediction",
    "counterfactual_evaluation",
    "published_ad_sift_reproduction",
    "architecture_ablation",
    "action_ablation",
    "operator_baselines",
    "screening_baselines",
    "ood_generalization",
    "tri_selective_risk",
    "serving_decision_eval",
    "efficiency_profile",
    "paper_report_generation",
}


def test_entrypoint_registry_covers_required_task_blocks() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    suite = config["paper_grade_full_suite"]
    task_names = {task["task_name"] for task in suite["tasks"]}
    assert task_names == REQUIRED_TASKS
    assert set(suite["entrypoint_registry"]) == REQUIRED_TASKS
    assert set(suite["task_block_paths"]) == REQUIRED_TASKS
    assert suite["formal_seed_values"] == [40, 41, 42, 43, 44]
    assert suite["user_entrypoint"] == "Libs/scripts/experiments/run_paper_grade_full_suite.sh"


def test_entrypoint_registry_paths_exist_and_task_blocks_match() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    suite = config["paper_grade_full_suite"]
    for task_name, entry in suite["entrypoint_registry"].items():
        shell_path = REPO_ROOT / entry["shell_path"]
        python_path = REPO_ROOT / entry["python_path"]
        config_path = REPO_ROOT / entry["config_path"]
        assert shell_path.exists(), task_name
        assert python_path.exists(), task_name
        assert config_path.exists(), task_name
        block = yaml.safe_load(config_path.read_text(encoding="utf-8"))["task_block"]
        assert block["task_name"] == task_name
        assert block["shell_path"] == entry["shell_path"]
        assert block["python_path"] == entry["python_path"]
        assert block["resource_profile"] == entry["resource_profile"]


def test_entrypoint_inventory_is_complete_without_promoting_reference_baselines() -> None:
    report = build_inventory_report(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert report["task_count"] == len(REQUIRED_TASKS)
    assert report["missing_count"] == 0
    assert report["status"] in {"ready", "partial"}
    rows = {row["task"]: row for row in report["tasks"]}
    assert rows["operator_baselines"]["status"] == "partial"
    assert rows["screening_baselines"]["status"] == "partial"
