from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_paper_grade_full_suite import baseline_schema_failures
from Libs.scripts.validators.validate_paper_grade_full_suite import expected_manifest_entries
from Libs.scripts.validators.build_experiment_entrypoint_inventory import build_inventory_report
from Libs.scripts.validators.validate_formal_experiment_freeze import GENERATED_AUDIT_OUTPUTS
from Libs.scripts.validators.validate_formal_experiment_freeze import git_state
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic


EXPECTED_SUITE_SEEDS = [40, 41, 42, 43, 44]
PRELIVE_JSON_OUTPUT = "Output/audits/paper_grade_full_suite_prelive_summary.json"
PRELIVE_MD_OUTPUT = "Output/audits/paper_grade_full_suite_prelive_summary.md"
BLOCKED = "blocked"
READY = "ready_for_live_submission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper-grade full-suite pre-live dry-run summary.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
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


def sha256_json(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def read_json_object(path_value: object) -> dict[str, object]:
    payload = json.loads(artifact_path(path_value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path_value}")
    return payload


def effective_config_path(config_path: str | Path) -> Path:
    config = load_config(config_path)
    suite = dict(config["paper_grade_full_suite"])
    candidate = artifact_path(dict(suite["cluster"])["suite_config_output"])
    if candidate.exists():
        materialized = load_config(candidate)
        materialized_suite = dict(materialized.get("paper_grade_full_suite", {}))
        if bool(materialized_suite.get("suite_config_materialized", False)):
            return candidate
    return artifact_path(config_path)


def freeze_signature_payload(config: dict[str, object]) -> tuple[dict[str, object], str]:
    suite = dict(config["paper_grade_full_suite"])
    manifest_path = str(suite.get("freeze_manifest_path", "Output/audits/formal_experiment_freeze_manifest.json"))
    manifest = read_json_object(manifest_path)
    signature = manifest.get("snapshot_signature", {})
    if not isinstance(signature, dict):
        raise TypeError("formal_experiment_freeze_manifest.snapshot_signature must be a mapping.")
    return signature, sha256_json(signature)


def command_list(config: dict[str, object]) -> list[str]:
    suite = dict(config["paper_grade_full_suite"])
    validators = dict(suite["validators"])
    return [str(command) for command in validators["prelaunch_order"]]


def freeze_replay_gate_present(commands: Iterable[str]) -> bool:
    return any("validate_formal_experiment_freeze.py" in command and "--no-write" in command for command in commands)


def sbatch_paths_for_roots(config: dict[str, object]) -> list[Path]:
    suite = dict(config["paper_grade_full_suite"])
    submission_order = dict(suite["submission_order"])
    roots = [Path("Output/logs/print_only/baselines"), Path(str(dict(suite["cluster"])["print_only_root"]))]
    roots.extend(Path("Output/logs/print_only") / str(profile) for profile in submission_order["dynamics_profiles"])
    roots.extend(Path("Output/logs/print_only") / str(profile) for profile in submission_order.get("shared_context_profiles", []))
    paths: list[Path] = []
    for root in roots:
        absolute_root = artifact_path(root)
        if absolute_root.exists():
            paths.extend(sorted(absolute_root.glob("*.sbatch")))
    return sorted(set(paths))


def sbatch_contains_freeze_signature(path: Path, freeze_sha256: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return f"PAPER_GRADE_FREEZE_SIGNATURE_SHA256={freeze_sha256}" in text


def glob_one(pattern: str) -> str:
    matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        return ""
    return repo_relative(matches[0])


def expected_sbatch_for_model(config: dict[str, object], model_key: str, seed: int) -> str:
    suite = dict(config["paper_grade_full_suite"])
    models = dict(suite["models"])
    submission_order = dict(suite["submission_order"])
    model = dict(models[model_key])
    profile_name = str(model.get("profile_name", "")).strip()
    adapter_name = str(model.get("adapter_name", "")).strip()
    if profile_name in {str(value) for value in submission_order["dynamics_profiles"]}:
        return glob_one(f"Output/logs/print_only/{profile_name}/NrnSift_*_{profile_name}_seed_{seed}.sbatch")
    if profile_name in {str(value) for value in submission_order.get("shared_context_profiles", [])}:
        return glob_one(f"Output/logs/print_only/{profile_name}/NrnSift_{profile_name}_seed_{seed}.sbatch")
    if adapter_name:
        return glob_one(f"Output/logs/print_only/baselines/NrnSift_baseline_{adapter_name}_seed_{seed}.sbatch")
    return ""


def job_matrix(config: dict[str, object], seeds: list[int]) -> list[dict[str, object]]:
    suite = dict(config["paper_grade_full_suite"])
    rows: list[dict[str, object]] = []
    for task in suite["tasks"]:
        task_name = str(dict(task)["task_name"])
        for model_key in [str(value) for value in dict(task)["required_models"]]:
            for seed in seeds:
                sbatch_path = expected_sbatch_for_model(config, model_key, seed)
                rows.append(
                    {
                        "task_name": task_name,
                        "model_key": model_key,
                        "seed": seed,
                        "job_count": 1 if sbatch_path else 0,
                        "sbatch_path": sbatch_path,
                    }
                )
    return rows


def task_job_counts(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        task_name = str(row["task_name"])
        model_key = str(row["model_key"])
        task_counts = counts.setdefault(task_name, {})
        task_counts[model_key] = task_counts.get(model_key, 0) + int(row["job_count"])
    return counts


def observed_seed_values(sbatch_paths: list[Path]) -> list[int]:
    values: set[int] = set()
    for path in sbatch_paths:
        match = re.search(r"_seed_(\d+)", path.name)
        if match:
            values.add(int(match.group(1)))
    return sorted(values)


def baseline_rows(config: dict[str, object]) -> list[dict[str, object]]:
    registry = dict(config["paper_grade_baseline_registry"])
    return [dict(row) for row in registry["baselines"]]


def reference_only_main_table_baselines(config: dict[str, object]) -> list[str]:
    return [
        str(row["baseline_id"])
        for row in baseline_rows(config)
        if bool(row["reference_only"]) and bool(row["main_table_runnable"])
    ]


def main_table_unrunnable_baselines(config: dict[str, object], seeds: list[int]) -> list[str]:
    failures = set(baseline_schema_failures(config))
    suite = dict(config["paper_grade_full_suite"])
    models = dict(suite["models"])
    missing: list[str] = []
    for row in baseline_rows(config):
        baseline_id = str(row["baseline_id"])
        if not bool(row["main_table_runnable"]):
            continue
        if bool(row["reference_only"]):
            missing.append(f"{baseline_id}:reference_only_in_main_table")
            continue
        if baseline_id in models:
            for seed in seeds:
                if not expected_sbatch_for_model(config, baseline_id, seed):
                    missing.append(f"{baseline_id}:missing_seed_{seed}_job")
    return sorted(failures | set(missing))


def task_blocks(config: dict[str, object]) -> list[dict[str, object]]:
    suite = dict(config["paper_grade_full_suite"])
    return [
        {
            "task_name": str(dict(task)["task_name"]),
            "task_family": str(dict(task)["task_family"]),
            "required_models": [str(value) for value in dict(task)["required_models"]],
            "required_holdouts": [str(value) for value in dict(task)["required_holdouts"]],
        }
        for task in suite["tasks"]
    ]


def model_families(config: dict[str, object]) -> list[dict[str, object]]:
    suite = dict(config["paper_grade_full_suite"])
    baseline_by_id = {str(row["baseline_id"]): dict(row) for row in baseline_rows(config)}
    rows: list[dict[str, object]] = []
    for model_key, payload in dict(suite["models"]).items():
        baseline = baseline_by_id.get(str(model_key), {})
        rows.append(
            {
                "model_key": str(model_key),
                "family": str(baseline.get("baseline_family", "suite_model")),
                "profile_name": str(dict(payload).get("profile_name", "")),
                "adapter_name": str(dict(payload).get("adapter_name", "")),
                "main_table": bool(dict(payload).get("main_table", False)),
            }
        )
    return rows


def baseline_families(config: dict[str, object]) -> list[str]:
    return sorted({str(row["baseline_family"]) for row in baseline_rows(config)})


def ablation_families(config: dict[str, object]) -> list[str]:
    return sorted(
        {
            str(row["baseline_family"])
            for row in baseline_rows(config)
            if "ablation" in str(row["baseline_family"]) or str(row["baseline_id"]).startswith("screening_without")
        }
    )


def expected_output_artifacts(config: dict[str, object]) -> list[str]:
    suite = dict(config["paper_grade_full_suite"])
    outputs = [str(value) for value in dict(suite["output_paths"]).values()]
    published = dict(suite["published_ad_sift_reproduction"])
    outputs.extend([str(published["output_root"]), str(published["audit_json"]), str(published["audit_md"])])
    outputs.append(str(dict(suite["manifest_contract"])["manifest_template"]))
    return sorted(set(outputs))


def current_git_full_diff_hash(config: dict[str, object]) -> str:
    excluded_paths = set(GENERATED_AUDIT_OUTPUTS) | {
        "Output/audits/formal_experiment_freeze_manifest.json",
        "Output/audits/formal_experiment_freeze_manifest.md",
    }
    return str(git_state(excluded_paths)["full_diff_hash"])


def positive_claim_patterns() -> list[re.Pattern[str]]:
    return [
        re.compile(r"\bG5\b.{0,50}\b(passed|ready|approved|allowed)", re.IGNORECASE),
        re.compile(r"\b1M\b.{0,80}\b(readiness|trajectory|gate).{0,80}\b(passed|ready|approved|allowed)", re.IGNORECASE),
        re.compile(r"\bgeneral drug screening\b.{0,80}\b(passed|ready|approved|allowed|complete)", re.IGNORECASE),
        re.compile(r"\bclinical endpoint prediction\b.{0,80}\b(passed|ready|approved|allowed|complete|validated)", re.IGNORECASE),
    ]


def line_is_prohibition_context(line: str) -> bool:
    lowered = line.lower()
    return any(fragment in lowered for fragment in ["forbidden", "prohibited", "do not", "must not", "not_passed", "not passed"])


def collect_report_text_paths(config: dict[str, object]) -> list[Path]:
    suite = dict(config["paper_grade_full_suite"])
    output_paths = dict(suite["output_paths"])
    candidates = [
        artifact_path(output_paths.get("manifest_md", "")),
        artifact_path(output_paths.get("execution_summary_md", "")),
        artifact_path(output_paths.get("paper_grade_result_summary_json", "")),
        artifact_path(dict(suite["published_ad_sift_reproduction"]).get("audit_md", "")),
    ]
    return [path for path in candidates if path.exists() and path.is_file()]


def claim_text_violations(config: dict[str, object]) -> list[str]:
    violations: list[str] = []
    patterns = positive_claim_patterns()
    for path in collect_report_text_paths(config):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line_is_prohibition_context(line):
                continue
            for pattern in patterns:
                if pattern.search(line):
                    violations.append(f"{repo_relative(path)}:{line_number}:{pattern.pattern}")
    return violations


def published_ad_status(config: dict[str, object]) -> dict[str, object]:
    eval_path = artifact_path("Output/logs/print_only/paper_grade_full_suite/NrnSift_paper_grade_ad_reproduction.sbatch")
    live_evaluable = eval_path.exists() and "evaluate_published_ad_sift_reproduction.py" in eval_path.read_text(encoding="utf-8")
    result_path = artifact_path("Output/results/published_ad_sift_reproduction/published_ad_sift_reproduction_summary.json")
    result_status = "not_materialized"
    if result_path.exists():
        result = read_json_object(result_path)
        result_status = str(result.get("status", "materialized_without_status"))
    return {
        "status": "live_evaluable" if live_evaluable else "pending_only",
        "eval_sbatch_path": repo_relative(eval_path) if eval_path.exists() else "",
        "result_status": result_status,
        "pending_only": not live_evaluable,
        "incorrectly_written_completed": (not live_evaluable) and result_status in {"available", "complete", "completed", "pass"},
    }


def build_prelive_summary(config_path: str | Path) -> dict[str, object]:
    resolved_path = effective_config_path(config_path)
    config = load_config(resolved_path)
    suite = dict(config["paper_grade_full_suite"])
    signature, freeze_sha256 = freeze_signature_payload(config)
    frozen_git_full_diff_hash = str(signature.get("git_full_diff_hash", ""))
    current_git_hash = current_git_full_diff_hash(config)
    requires_refreeze_before_live = current_git_hash != frozen_git_full_diff_hash
    inventory = build_inventory_report(resolved_path)
    seeds = [int(value) for value in suite["formal_seed_values"]]
    sbatches = sbatch_paths_for_roots(config)
    job_rows = job_matrix(config, seeds)
    missing_seed_jobs = [
        f"{row['task_name']}:{row['model_key']}:seed_{row['seed']}"
        for row in job_rows
        if int(row["job_count"]) == 0
    ]
    observed_seeds = observed_seed_values(sbatches)
    unexpected_observed_seeds = [seed for seed in observed_seeds if seed not in EXPECTED_SUITE_SEEDS]
    unbound_jobs = [
        repo_relative(path)
        for path in sbatches
        if not sbatch_contains_freeze_signature(path, freeze_sha256)
    ]
    gates = command_list(config)
    published_status = published_ad_status(config)
    reference_only_in_main = reference_only_main_table_baselines(config)
    main_table_baseline_failures = main_table_unrunnable_baselines(config, seeds)
    pending_only_tasks = sorted(
        {
            str(row["task_name"])
            for row in job_rows
            if all(
                int(candidate["job_count"]) == 0
                for candidate in job_rows
                if str(candidate["task_name"]) == str(row["task_name"])
            )
        }
    )
    claim_violations = claim_text_violations(config)
    blockers: list[str] = []
    if seeds != EXPECTED_SUITE_SEEDS or unexpected_observed_seeds:
        blockers.append(f"seeds_not_exact:{seeds};unexpected_observed_seeds={unexpected_observed_seeds}")
    blockers.extend(f"missing_main_task_seed:{item}" for item in missing_seed_jobs)
    blockers.extend(f"main_table_baseline_not_runnable:{item}" for item in main_table_baseline_failures)
    blockers.extend(f"job_not_bound_to_freeze_signature:{item}" for item in unbound_jobs)
    if not freeze_replay_gate_present(gates):
        blockers.append("launch_gate_missing_formal_freeze_no_write_replay")
    if bool(published_status["incorrectly_written_completed"]):
        blockers.append("published_ad_sift_reproduction_pending_only_but_written_completed")
    blockers.extend(f"claim_text_overclaim:{item}" for item in claim_violations)
    if requires_refreeze_before_live:
        blockers.append("requires_refreeze_before_live:current_code_or_config_differs_from_frozen_manifest")
    warnings = [
        "print_only_dry_run_only_no_slurm_submission",
        "postrun_artifact_manifests_expected_after_live_execution",
    ]
    prelive_status = BLOCKED if blockers else READY
    return {
        "audit_name": "paper_grade_full_suite_prelive_summary",
        "suite_id": str(suite["suite_name"]),
        "suite_release": str(suite["suite_release"]),
        "config_path": repo_relative(resolved_path),
        "user_entrypoint": str(suite["user_entrypoint"]),
        "developer_entrypoint": str(suite["developer_entrypoint"]),
        "entrypoint_inventory_status": str(inventory["status"]),
        "entrypoint_inventory_task_count": int(inventory["task_count"]),
        "entrypoint_inventory": inventory,
        "freeze_binding_policy": dict(suite.get("freeze_binding_policy", {})),
        "requires_refreeze_before_live": requires_refreeze_before_live,
        "freeze_snapshot_signature": signature,
        "freeze_snapshot_signature_sha256": freeze_sha256,
        "git_full_diff_hash": frozen_git_full_diff_hash,
        "current_git_full_diff_hash": current_git_hash,
        "resolved_config_sha256": str(dict(signature.get("config_hashes", {})).get("resolved_config_sha256", sha256_json(config))),
        "seeds": seeds,
        "observed_sbatch_seed_values": observed_seeds,
        "task_blocks": task_blocks(config),
        "model_families": model_families(config),
        "baseline_families": baseline_families(config),
        "ablation_families": ablation_families(config),
        "number_of_jobs_by_task_model_seed": job_rows,
        "job_counts_by_task_model": task_job_counts(job_rows),
        "expected_output_artifacts": expected_output_artifacts(config),
        "launch_gate_command_list": gates,
        "freeze_replay_gate_present": freeze_replay_gate_present(gates),
        "all_jobs_bind_to_current_freeze_signature": not unbound_jobs,
        "unbound_job_paths": unbound_jobs,
        "pending_only_task_exists": bool(pending_only_tasks),
        "pending_only_tasks": pending_only_tasks,
        "published_ad_sift_reproduction": published_status,
        "reference_only_baseline_in_main_table": reference_only_in_main,
        "main_table_baseline_failures": main_table_baseline_failures,
        "claim_text_violations": claim_violations,
        "expected_manifest_entry_count": len(expected_manifest_entries(config)),
        "print_only_sbatch_count": len(sbatches),
        "print_only_sbatch_paths": [repo_relative(path) for path in sbatches],
        "prelive_status": prelive_status,
        "live_submission_allowed": prelive_status != BLOCKED,
        "blockers": blockers,
        "warnings": warnings,
        "recommended_live_submission_command": "bash Libs/scripts/experiments/run_paper_grade_full_suite.sh --config Libs/configs/config_stack.yaml",
        "developer_live_submission_command": "bash Libs/exp/submission/submit_paper_grade_full_suite.sh --config Libs/configs/config_stack.yaml",
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Paper-Grade Full Suite Pre-Live Summary",
        "",
        f"- suite_id: `{report['suite_id']}`",
        f"- prelive_status: `{report['prelive_status']}`",
        f"- live_submission_allowed: `{report['live_submission_allowed']}`",
        f"- freeze_snapshot_signature_sha256: `{report['freeze_snapshot_signature_sha256']}`",
        f"- git_full_diff_hash: `{report['git_full_diff_hash']}`",
        f"- current_git_full_diff_hash: `{report['current_git_full_diff_hash']}`",
        f"- requires_refreeze_before_live: `{report['requires_refreeze_before_live']}`",
        f"- user_entrypoint: `{report['user_entrypoint']}`",
        f"- developer_entrypoint: `{report['developer_entrypoint']}`",
        f"- entrypoint_inventory_status: `{report['entrypoint_inventory_status']}`",
        f"- resolved_config_sha256: `{report['resolved_config_sha256']}`",
        f"- seeds: `{report['seeds']}`",
        f"- print_only_sbatch_count: `{report['print_only_sbatch_count']}`",
        f"- freeze_replay_gate_present: `{report['freeze_replay_gate_present']}`",
        f"- all_jobs_bind_to_current_freeze_signature: `{report['all_jobs_bind_to_current_freeze_signature']}`",
        f"- pending_only_task_exists: `{report['pending_only_task_exists']}`",
        f"- published_ad_sift_reproduction: `{dict(report['published_ad_sift_reproduction'])['status']}`",
        "",
        "## Blockers",
    ]
    blockers = [str(value) for value in report["blockers"]]
    lines.extend([f"- `{blocker}`" for blocker in blockers] or ["- none"])
    lines.extend(["", "## Warnings"])
    warnings = [str(value) for value in report["warnings"]]
    lines.extend([f"- `{warning}`" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "## Task Blocks",
            "| task | family | required_models | required_holdouts |",
            "| --- | --- | --- | --- |",
        ]
    )
    for task in report["task_blocks"]:
        payload = dict(task)
        lines.append(
            f"| {payload['task_name']} | {payload['task_family']} | "
            f"{', '.join(str(value) for value in payload['required_models'])} | "
            f"{', '.join(str(value) for value in payload['required_holdouts'])} |"
        )
    lines.extend(
        [
            "",
            "## Job Counts",
            "| task | model | count |",
            "| --- | --- | --- |",
        ]
    )
    for task_name, model_counts in dict(report["job_counts_by_task_model"]).items():
        for model_key, count in dict(model_counts).items():
            lines.append(f"| {task_name} | {model_key} | {count} |")
    lines.extend(
        [
            "",
            "## Recommended Command",
            "",
            f"`{report['recommended_live_submission_command']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_prelive_summary(report: dict[str, object], json_output: str, md_output: str) -> None:
    write_json(artifact_path(json_output), report)
    write_text_atomic(artifact_path(md_output), markdown_report(report))


def main() -> None:
    args = parse_args()
    report = build_prelive_summary(args.config)
    config = load_config(effective_config_path(args.config))
    outputs = dict(dict(config["paper_grade_full_suite"])["output_paths"])
    json_output = args.json_output or str(outputs.get("prelive_summary_json", PRELIVE_JSON_OUTPUT))
    md_output = args.md_output or str(outputs.get("prelive_summary_md", PRELIVE_MD_OUTPUT))
    write_prelive_summary(report, json_output, md_output)
    if report["prelive_status"] == BLOCKED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
