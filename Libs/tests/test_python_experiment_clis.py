from __future__ import annotations

from pathlib import Path
import py_compile
import subprocess

from Libs.utils.data.raw_data_registry import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def python_entrypoints() -> list[Path]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    paths = [REPO_ROOT / entry["python_path"] for entry in config["paper_grade_full_suite"]["entrypoint_registry"].values()]
    paths.append(REPO_ROOT / "Libs/exp/aggregation/build_completion_audit.py")
    return sorted(set(paths))


def test_python_entrypoints_compile() -> None:
    for path in python_entrypoints():
        py_compile.compile(str(path), doraise=True)


def test_python_entrypoints_expose_standard_cli_flags() -> None:
    required_flags = [
        "--config",
        "--suite-config",
        "--seed",
        "--task",
        "--model-family",
        "--experiment-family",
        "--output-dir",
        "--freeze-manifest",
        "--run-id",
        "--dry-run",
    ]
    for path in python_entrypoints():
        completed = subprocess.run(
            ["python", str(path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        for flag in required_flags:
            assert flag in completed.stdout, path
