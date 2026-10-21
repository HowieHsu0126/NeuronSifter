from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def shell_scripts() -> list[Path]:
    return sorted((REPO_ROOT / "Libs/scripts/experiments").glob("*.sh")) + sorted(
        [
            REPO_ROOT / "Libs/scripts/validators/run_prelaunch_gates.sh",
            REPO_ROOT / "Libs/scripts/validators/run_freeze_replay_gate.sh",
            REPO_ROOT / "Libs/scripts/validators/run_postrun_gates.sh",
            REPO_ROOT / "Libs/scripts/monitoring/monitor_slurm_suite.sh",
            REPO_ROOT / "Libs/scripts/monitoring/collect_slurm_logs.sh",
            REPO_ROOT / "Libs/scripts/monitoring/summarize_slurm_failures.sh",
        ]
    )


def test_shell_scripts_use_strict_mode_and_project_prefix() -> None:
    for path in shell_scripts():
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text, path
        if path.name != "_entrypoint_lib.sh":
            assert "PROJECT_PREFIX" in text or "_entrypoint_lib.sh" in text, path


def test_shell_scripts_pass_bash_syntax() -> None:
    for path in shell_scripts():
        subprocess.run(["bash", "-n", str(path)], cwd=REPO_ROOT, check=True)


def test_task_shell_print_only_prints_sbatch_without_submission() -> None:
    script = REPO_ROOT / "Libs/scripts/experiments/run_rollout_prediction.sh"
    completed = subprocess.run(
        ["bash", str(script), "--print-only", "--seed", "40", "--model-family", "neuronsifter_mainline"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "PYTHON_COMMAND=" in completed.stdout
    assert "SBATCH_COMMAND=" in completed.stdout
    assert "sbatch" in completed.stdout
    assert "Submitted batch job" not in completed.stdout
    assert "srun" not in completed.stdout


def test_prelaunch_gate_wrapper_declares_seven_ordered_gates() -> None:
    completed = subprocess.run(
        ["bash", "Libs/scripts/validators/run_prelaunch_gates.sh", "--print-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("PRELAUNCH_GATE=")]
    assert len(lines) == 7
    assert any("validate_formal_experiment_freeze.py --no-write" in line for line in lines)
