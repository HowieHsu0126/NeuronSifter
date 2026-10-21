from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper-grade experiment entrypoint inventory.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--suite-config", default="Libs/configs/exp/paper_grade_full_suite.yaml")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--md-output", default="")
    return parser.parse_args()


def artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def read_yaml(path_value: object) -> dict[str, object]:
    path = artifact_path(path_value)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Task block root must be a mapping: {path}")
    return payload


def row_status(shell_path: Path, python_path: Path, config_path: Path, registry_status: str) -> str:
    """Return inventory readiness status from declared paths and registry status."""
    existing = [shell_path.exists(), python_path.exists(), config_path.exists()]
    if not any(existing):
        return "missing"
    if not all(existing):
        return "partial"
    if registry_status == "reference_only":
        return "reference_only"
    return registry_status if registry_status in {"ready", "partial"} else "ready"


def supports_shell_print_only(shell_path: Path) -> bool:
    """Return whether a shell wrapper exposes the shared print-only path."""
    if not shell_path.exists():
        return False
    text = shell_path.read_text(encoding="utf-8")
    return "--print-only" in text or "_entrypoint_lib.sh" in text


def supports_shell_slurm_submit(shell_path: Path) -> bool:
    """Return whether a shell wrapper can construct a Slurm submission."""
    if not shell_path.exists():
        return False
    text = shell_path.read_text(encoding="utf-8")
    return "sbatch" in text or "_entrypoint_lib.sh" in text


def supports_python_standard_cli(python_path: Path) -> bool:
    """Return whether a Python entrypoint uses the standard CLI helper path."""
    if not python_path.exists():
        return False
    text = python_path.read_text(encoding="utf-8")
    markers = [
        "parse_standard_experiment_args",
        "parse_standard_args",
        "run_catalog_entrypoint",
        "run_delegate_entrypoint",
    ]
    return any(marker in text for marker in markers)


def experiment_family_paths_for_task(config: dict[str, object], task: dict[str, object]) -> list[str]:
    """Resolve existing experiment-family YAML paths referenced by a task model list."""
    suite = dict(config["paper_grade_full_suite"])
    models = dict(suite["models"])
    paths: set[str] = set()
    for model_key in [str(value) for value in task.get("required_models", [])]:
        model_payload = dict(models.get(model_key, {}))
        candidates = [
            str(model_payload.get("profile_name", "")).strip(),
            str(model_payload.get("adapter_name", "")).strip(),
            model_key,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            family_path = artifact_path(f"Libs/configs/exp/experiment_families/{candidate}.yaml")
            if family_path.exists():
                paths.add(repo_relative(family_path))
    return sorted(paths)


def task_rows(config: dict[str, object]) -> list[dict[str, object]]:
    """Build one inventory row per paper-grade task."""
    suite = dict(config["paper_grade_full_suite"])
    registry = dict(suite["entrypoint_registry"])
    block_paths = dict(suite["task_block_paths"])
    rows: list[dict[str, object]] = []
    for task in suite["tasks"]:
        task_name = str(dict(task)["task_name"])
        entry = dict(registry[task_name])
        block_path = artifact_path(block_paths[task_name])
        block = dict(read_yaml(block_path).get("task_block", {}))
        shell_path = artifact_path(entry["shell_path"])
        python_path = artifact_path(entry["python_path"])
        config_path = artifact_path(entry["config_path"])
        status = row_status(shell_path, python_path, config_path, str(entry.get("status", "")))
        family_paths = experiment_family_paths_for_task(config, dict(task))
        print_only = supports_shell_print_only(shell_path)
        slurm_submit = supports_shell_slurm_submit(shell_path)
        standard_cli = supports_python_standard_cli(python_path)
        rows.append(
            {
                "task_id": task_name,
                "task": task_name,
                "task_family": str(dict(task)["task_family"]),
                "shell_script_path": repo_relative(shell_path),
                "shell_path": repo_relative(shell_path),
                "python_entrypoint_path": repo_relative(python_path),
                "python_path": repo_relative(python_path),
                "task_block_yaml_path": repo_relative(config_path),
                "config_path": repo_relative(config_path),
                "experiment_family_yaml_paths": family_paths,
                "expected_outputs": [str(value) for value in block.get("expected_outputs", [])],
                "resource_profile": str(entry["resource_profile"]),
                "required_gates": [str(value) for value in entry.get("required_gates", [])],
                "postrun_validator": str(entry.get("postrun_validator", "")),
                "supports_print_only": print_only,
                "supports_slurm_submit": slurm_submit,
                "supports_standard_cli": standard_cli,
                "tested": print_only and slurm_submit and standard_cli,
                "test_status": "static_inventory_checked",
                "status": status,
            }
        )
    return rows


def build_inventory_report(config_path: str | Path, suite_config_path: str | Path = "Libs/configs/exp/paper_grade_full_suite.yaml") -> dict[str, object]:
    """Build the experiment-entrypoint inventory report."""
    config = load_config(config_path)
    rows = task_rows(config)
    missing = [row for row in rows if row["status"] == "missing"]
    partial = [row for row in rows if row["status"] == "partial"]
    return {
        "audit_name": "experiment_entrypoint_inventory",
        "config_path": repo_relative(artifact_path(config_path)),
        "suite_config_path": repo_relative(artifact_path(suite_config_path)),
        "user_entrypoint": str(dict(config["paper_grade_full_suite"])["user_entrypoint"]),
        "developer_entrypoint": str(dict(config["paper_grade_full_suite"])["developer_entrypoint"]),
        "task_count": len(rows),
        "ready_count": sum(1 for row in rows if row["status"] == "ready"),
        "partial_count": len(partial),
        "missing_count": len(missing),
        "reference_only_count": sum(1 for row in rows if row["status"] == "reference_only"),
        "status": "ready" if not missing and not partial else "partial",
        "tasks": rows,
    }


def markdown_report(report: dict[str, object]) -> str:
    """Render the inventory report as Markdown."""
    lines = [
        "# Experiment Entrypoint Inventory",
        "",
        f"- status: `{report['status']}`",
        f"- task_count: `{report['task_count']}`",
        f"- ready_count: `{report['ready_count']}`",
        f"- partial_count: `{report['partial_count']}`",
        f"- missing_count: `{report['missing_count']}`",
        f"- suite_config_path: `{report['suite_config_path']}`",
        f"- user_entrypoint: `{report['user_entrypoint']}`",
        f"- developer_entrypoint: `{report['developer_entrypoint']}`",
        "",
        "| task | status | shell | python | config | resource | print_only | slurm | cli | gates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["tasks"]:
        lines.append(
            f"| {row['task']} | {row['status']} | {row['shell_script_path']} | {row['python_entrypoint_path']} | "
            f"{row['task_block_yaml_path']} | {row['resource_profile']} | {row['supports_print_only']} | "
            f"{row['supports_slurm_submit']} | {row['supports_standard_cli']} | {len(row['required_gates'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the inventory JSON and Markdown artifacts."""
    args = parse_args()
    config = load_config(args.config)
    outputs = dict(dict(config["paper_grade_full_suite"])["output_paths"])
    json_output = args.json_output or str(outputs["entrypoint_inventory_json"])
    md_output = args.md_output or str(outputs["entrypoint_inventory_md"])
    report = build_inventory_report(args.config, args.suite_config)
    write_json(artifact_path(json_output), report)
    write_text_atomic(artifact_path(md_output), markdown_report(report))
    print(json.dumps(report, indent=2, sort_keys=False))
    if report["status"] not in {"ready", "partial"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
