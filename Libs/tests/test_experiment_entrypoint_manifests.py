from __future__ import annotations

import json
from pathlib import Path
import subprocess

from Libs.scripts.validators.build_experiment_entrypoint_inventory import build_inventory_report


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_python_entrypoint_writes_standard_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "rollout_entrypoint"
    subprocess.run(
        [
            "python",
            "Libs/exp/evaluation/evaluate_rollout_prediction.py",
            "--config",
            "Libs/configs/config_stack.yaml",
            "--seed",
            "41",
            "--model-family",
            "neuronsifter_mainline",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    for filename in ["metrics.json", "run_metadata.json", "artifact_hashes.json", "artifact_manifest.json"]:
        assert (output_dir / filename).exists()
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_name"] == "rollout_prediction"
    assert manifest["model_key"] == "neuronsifter_mainline"
    assert manifest["seed"] == 41
    assert manifest["postrun_status"] == "dry_run"


def test_reference_only_baseline_writes_audit_status_without_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "reference_entrypoint"
    subprocess.run(
        [
            "python",
            "Libs/exp/baselines/run_operator_baselines.py",
            "--config",
            "Libs/configs/config_stack.yaml",
            "--seed",
            "41",
            "--model-family",
            "fno_reference",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert (output_dir / "reference_audit_status.json").exists()
    assert not (output_dir / "metrics.json").exists()
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["postrun_status"] == "reference_only"
    assert manifest["metrics_json"] == ""


def test_inventory_rows_expose_user_facing_contract_fields() -> None:
    report = build_inventory_report(
        REPO_ROOT / "Libs/configs/config_stack.yaml",
        REPO_ROOT / "Libs/configs/exp/paper_grade_full_suite.yaml",
    )
    for row in report["tasks"]:
        assert row["task_id"] == row["task"]
        assert row["shell_script_path"].endswith(".sh")
        assert row["python_entrypoint_path"].endswith(".py")
        assert row["task_block_yaml_path"].endswith(".yaml")
        assert isinstance(row["experiment_family_yaml_paths"], list)
        assert row["expected_outputs"]
        assert row["supports_print_only"] is True
        assert row["supports_slurm_submit"] is True
        assert row["supports_standard_cli"] is True
        assert isinstance(row["tested"], bool)
