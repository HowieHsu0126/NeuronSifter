from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic
from Libs.utils.exp.experiment_manifests import load_experiment_family


PASS = "pass"
FAIL = "fail"
PENDING = "pending"
BLOCKER = "blocker"
MAJOR = "major"
MINOR = "minor"
EXPECTED_SUITE_SEEDS = [40, 41, 42, 43, 44]


@dataclass(frozen=True)
class SuiteCheck:
    check_id: str
    check_name: str
    status: str
    severity: str
    evidence: list[str]
    required_fix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the paper-grade full formal suite manifest or postrun artifacts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--mode", choices=["prelaunch", "postrun"], default="prelaunch")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--md-output", default="")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def check_row(
    check_id: str,
    check_name: str,
    condition: bool,
    evidence: list[str],
    required_fix: str,
    severity: str = BLOCKER,
) -> SuiteCheck:
    return SuiteCheck(
        check_id=check_id,
        check_name=check_name,
        status=PASS if condition else FAIL,
        severity=MINOR if condition else severity,
        evidence=evidence,
        required_fix="No fix required." if condition else required_fix,
    )


def pending_row(check_id: str, check_name: str, evidence: list[str], required_fix: str) -> SuiteCheck:
    return SuiteCheck(
        check_id=check_id,
        check_name=check_name,
        status=PENDING,
        severity=MAJOR,
        evidence=evidence,
        required_fix=required_fix,
    )


def suite_config(config: dict[str, object]) -> dict[str, object]:
    payload = config.get("paper_grade_full_suite", {})
    if not isinstance(payload, dict) or not payload:
        raise KeyError("Missing paper_grade_full_suite config.")
    return payload


def baseline_registry(config: dict[str, object]) -> dict[str, object]:
    payload = config.get("paper_grade_baseline_registry", {})
    if not isinstance(payload, dict) or not payload:
        raise KeyError("Missing paper_grade_baseline_registry config.")
    return payload


def load_profile_status(config: dict[str, object], profile_name: str) -> tuple[bool, str]:
    try:
        profile = load_experiment_family(profile_name, config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return False, str(exc)
    disease_program = str(profile.get("config_overrides", {}).get("benchmark_registry", {}).get("active_disease_program", ""))
    if not disease_program:
        disease_program = str(profile.get("disease_program", ""))
    return True, disease_program


def runnable_adapter_names(config: dict[str, object]) -> set[str]:
    return {
        str(row["baseline_name"])
        for row in config.get("baseline_adapter_registry", {}).get("paper_experiments", {}).get("runnable_baselines", [])
    } | {
        str(row["baseline_name"])
        for row in config.get("paper_experiments", {}).get("runnable_baselines", [])
    } | {
        str(value)
        for value in config.get("baseline_adapter_registry", {}).get("default_suite", {}).get("baseline_names", [])
    }


def baseline_schema_failures(config: dict[str, object]) -> list[str]:
    registry = baseline_registry(config)
    required_fields = [str(value) for value in registry["required_fields"]]
    adapter_names = runnable_adapter_names(config)
    failures: list[str] = []
    for baseline in registry["baselines"]:
        baseline_payload = dict(baseline)
        baseline_id = str(baseline_payload.get("baseline_id", ""))
        missing_fields = [field for field in required_fields if field not in baseline_payload]
        if missing_fields:
            failures.append(f"{baseline_id}:missing_fields={','.join(missing_fields)}")
            continue
        main_table = bool(baseline_payload["main_table_runnable"])
        appendix = bool(baseline_payload["appendix_runnable"])
        reference_only = bool(baseline_payload["reference_only"])
        reason = str(baseline_payload["reason_if_not_runnable"]).strip()
        metrics = list(baseline_payload["output_metrics"])
        profile_name = str(baseline_payload.get("profile_name", "")).strip()
        adapter_name = str(baseline_payload.get("adapter_name", "")).strip()
        if reference_only and main_table:
            failures.append(f"{baseline_id}:reference_only_cannot_enter_main_table")
        if reference_only and not reason:
            failures.append(f"{baseline_id}:reference_only_requires_reason")
        if not reference_only and not (main_table or appendix):
            failures.append(f"{baseline_id}:non_reference_baseline_has_no_table")
        if (main_table or appendix) and not reference_only and not metrics:
            failures.append(f"{baseline_id}:runnable_baseline_requires_output_metrics")
        if profile_name:
            profile_ok, profile_note = load_profile_status(config, profile_name)
            if not profile_ok:
                failures.append(f"{baseline_id}:missing_profile={profile_note}")
        if adapter_name and adapter_name not in adapter_names and baseline_payload.get("baseline_family") != "evidence_adapter":
            failures.append(f"{baseline_id}:adapter_not_registered={adapter_name}")
    return failures


def baseline_rows(config: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for baseline in baseline_registry(config)["baselines"]:
        payload = dict(baseline)
        rows.append(
            {
                "baseline_id": str(payload["baseline_id"]),
                "baseline_family": str(payload["baseline_family"]),
                "main_table_runnable": bool(payload["main_table_runnable"]),
                "appendix_runnable": bool(payload["appendix_runnable"]),
                "reference_only": bool(payload["reference_only"]),
                "reason_if_not_runnable": str(payload["reason_if_not_runnable"]),
                "profile_name": str(payload.get("profile_name", "")),
                "adapter_name": str(payload.get("adapter_name", "")),
                "output_metrics": [str(value) for value in payload["output_metrics"]],
            }
        )
    return rows


def expected_manifest_entries(config: dict[str, object]) -> list[dict[str, object]]:
    suite = suite_config(config)
    seeds = [int(value) for value in suite["formal_seed_values"]]
    entries: list[dict[str, object]] = []
    template = str(suite["manifest_contract"]["manifest_template"])
    for task in suite["tasks"]:
        task_name = str(task["task_name"])
        for model_key in [str(value) for value in task["required_models"]]:
            for seed in seeds:
                entries.append(
                    {
                        "task_name": task_name,
                        "model_key": model_key,
                        "seed": seed,
                        "manifest_path": template.format(task_name=task_name, model_key=model_key, seed=seed),
                    }
                )
    return entries


def read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def postrun_failures(config: dict[str, object]) -> tuple[list[str], list[dict[str, object]]]:
    required_fields = [str(value) for value in suite_config(config)["manifest_contract"]["required_fields"]]
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for entry in expected_manifest_entries(config):
        manifest_path = artifact_path(entry["manifest_path"])
        if not manifest_path.exists():
            failures.append(f"missing_manifest:{entry['manifest_path']}")
            rows.append({**entry, "status": "missing"})
            continue
        payload = read_json_object(manifest_path)
        missing_fields = [field for field in required_fields if field not in payload]
        postrun_status = str(payload.get("postrun_status", ""))
        if missing_fields:
            failures.append(f"{entry['manifest_path']}:missing_fields={','.join(missing_fields)}")
        if postrun_status not in {"materialized", "pass", "passed"}:
            failures.append(f"{entry['manifest_path']}:postrun_status={postrun_status}")
        rows.append({**entry, "status": postrun_status or "invalid", "missing_fields": missing_fields})
    return failures, rows


def build_baseline_readiness_report(config: dict[str, object]) -> dict[str, object]:
    failures = baseline_schema_failures(config)
    rows = baseline_rows(config)
    return {
        "audit_name": "baseline_readiness",
        "status": "pass" if not failures else "fail",
        "failure_count": len(failures),
        "failures": failures,
        "baseline_count": len(rows),
        "main_table_runnable_count": sum(1 for row in rows if row["main_table_runnable"]),
        "appendix_runnable_count": sum(1 for row in rows if row["appendix_runnable"]),
        "reference_only_count": sum(1 for row in rows if row["reference_only"]),
        "baselines": rows,
    }


def baseline_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Baseline Readiness Audit",
        "",
        f"- status: `{report['status']}`",
        f"- failure_count: `{report['failure_count']}`",
        f"- baseline_count: `{report['baseline_count']}`",
        "",
        "| baseline_id | family | main_table | appendix | reference_only | profile_or_adapter | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["baselines"]:
        profile_or_adapter = str(row["profile_name"] or row["adapter_name"])
        lines.append(
            f"| {row['baseline_id']} | {row['baseline_family']} | {row['main_table_runnable']} | "
            f"{row['appendix_runnable']} | {row['reference_only']} | {profile_or_adapter} | {row['reason_if_not_runnable']} |"
        )
    return "\n".join(lines) + "\n"


def build_prelaunch_report(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    suite = suite_config(config)
    checks: list[SuiteCheck] = []

    disease_programs = [str(value) for value in suite["disease_programs"]]
    checks.append(
        check_row(
            "SUITE-001",
            "Suite scope is AD-only.",
            disease_programs == ["ad_control"],
            ["disease_programs=" + ",".join(disease_programs)],
            "Remove non-AD disease programs from paper_grade_full_suite.",
        )
    )

    suite_seeds = [int(value) for value in suite["formal_seed_values"]]
    global_seeds = [int(value) for value in config["screening_pipeline"]["screening_benchmark"]["formal_seed_values"]]
    checks.append(
        check_row(
            "SUITE-002",
            "Suite seed set is fixed and separate from legacy publication assumptions.",
            suite_seeds == EXPECTED_SUITE_SEEDS,
            [
                "suite_seed_values=" + ",".join(str(seed) for seed in suite_seeds),
                "global_formal_seed_values=" + ",".join(str(seed) for seed in global_seeds),
                f"suite_config_materialized={suite.get('suite_config_materialized', False)}",
            ],
            "Use only seeds 40,41,42,43,44 for the paper-grade full formal suite.",
        )
    )

    model_failures = []
    models = dict(suite["models"])
    for task in suite["tasks"]:
        for model_key in [str(value) for value in task["required_models"]]:
            if model_key not in models:
                model_failures.append(f"{task['task_name']}:{model_key}:missing_model")
                continue
            profile_name = str(models[model_key].get("profile_name", "")).strip()
            if profile_name:
                profile_ok, disease_note = load_profile_status(config, profile_name)
                if not profile_ok:
                    model_failures.append(f"{model_key}:{profile_name}:missing_profile")
                if disease_note == "pd_control":
                    model_failures.append(f"{model_key}:{profile_name}:pd_profile_forbidden")
    submission_profiles = [str(value) for value in suite["submission_order"]["dynamics_profiles"]]
    forbidden_submission_profiles = [profile for profile in submission_profiles if profile.endswith("_pd") or profile == "dynamics_mainline_pd"]
    checks.append(
        check_row(
            "SUITE-003",
            "All task model references resolve and no PD profile enters the suite.",
            not model_failures and not forbidden_submission_profiles,
            model_failures + [f"forbidden_submission_profile={profile}" for profile in forbidden_submission_profiles] or [
                "task_model_references_resolved"
            ],
            "Define missing AD profiles or remove forbidden non-AD profiles from the suite.",
        )
    )

    baseline_report = build_baseline_readiness_report(config)
    checks.append(
        check_row(
            "SUITE-004",
            "Baseline registry is internally consistent.",
            baseline_report["status"] == "pass",
            baseline_report["failures"] or [
                f"main_table_runnable_count={baseline_report['main_table_runnable_count']}",
                f"appendix_runnable_count={baseline_report['appendix_runnable_count']}",
                f"reference_only_count={baseline_report['reference_only_count']}",
            ],
            "Repair paper_grade_baseline_registry schema, runnable/reference-only flags, or profile/adapter references.",
        )
    )

    controlled_access = dict(suite["controlled_access_policy"])
    checks.append(
        check_row(
            "SUITE-005",
            "Controlled ADNI remains audit-only for formal training and main tables.",
            str(controlled_access["adni_role"]) == "audit_only"
            and not bool(controlled_access["allow_adni_in_formal_training"])
            and not bool(controlled_access["allow_adni_in_main_tables"]),
            [
                f"adni_role={controlled_access['adni_role']}",
                f"allow_adni_in_formal_training={controlled_access['allow_adni_in_formal_training']}",
                f"allow_adni_in_main_tables={controlled_access['allow_adni_in_main_tables']}",
            ],
            "Keep controlled ADNI out of formal training and main tables unless the architecture boundary changes explicitly.",
        )
    )

    output_paths = dict(suite["output_paths"])
    required_output_keys = {
        "multimodal_alignment_json",
        "world_foundation_json",
        "baseline_readiness_json",
        "manifest_json",
        "execution_summary_json",
        "paper_grade_result_summary_json",
    }
    checks.append(
        check_row(
            "SUITE-006",
            "Prelaunch and final output paths are declared.",
            required_output_keys <= set(output_paths),
            ["missing_output_keys=" + ",".join(sorted(required_output_keys - set(output_paths)))],
            "Declare all required paper-grade full-suite audit and report output paths.",
        )
    )

    manifest_fields = [str(value) for value in suite["manifest_contract"]["required_fields"]]
    required_manifest_fields = {
        "config_hash",
        "split_hash",
        "checkpoint_hash",
        "metrics_json",
        "stdout_path",
        "stderr_path",
        "artifact_hash",
        "runtime_metadata",
        "postrun_status",
    }
    checks.append(
        check_row(
            "SUITE-007",
            "Task/model/seed manifest contract records artifact identity and runtime evidence.",
            required_manifest_fields <= set(manifest_fields),
            ["manifest_fields=" + ",".join(manifest_fields)],
            "Add missing artifact identity and runtime metadata fields to paper_grade_full_suite.manifest_contract.",
        )
    )

    checks.append(
        check_row(
            "SUITE-008",
            "Claim-prohibition rules are declared for known forbidden upgrades.",
            len(suite.get("claim_prohibition_rules", [])) >= 5,
            [f"claim_prohibition_rule_count={len(suite.get('claim_prohibition_rules', []))}"],
            "Declare explicit claim-prohibition rules for G5, general screening, clinical endpoint, private training, and secondary-disease readiness.",
        )
    )

    runtime_protocol = dict(suite.get("runtime_protocol", {}))
    runtime_commands = set(dict(runtime_protocol.get("commands", {})))
    training_config = dict(config["screening_pipeline"]["screening_benchmark"]["training"])
    cache_policy = dict(training_config.get("cache_policy", {}))
    data_loader = dict(training_config.get("data_loader", {}))
    runtime_required_output_keys = {
        "runtime_efficiency_profile_json",
        "runtime_efficiency_profile_md",
        "data_io_bottleneck_audit_json",
        "data_io_bottleneck_audit_md",
        "training_window_runtime_audit_json",
        "training_window_runtime_audit_md",
    }
    runtime_condition = (
        bool(runtime_protocol.get("profiling", {}).get("enabled", False))
        and bool(runtime_protocol.get("cpu_precompute", {}).get("enabled", False))
        and bool(runtime_protocol.get("gpu_concurrency_policy", {}).get("exact_prefix_queue_scan_required_before_live", False))
        and {"cache_precompute", "pretraining", "screening_window", "evaluation", "active_learning_replay", "export"} <= runtime_commands
        and bool(cache_policy.get("enabled", False))
        and bool(cache_policy.get("require_cache_for_formal", False))
        and not bool(cache_policy.get("allow_training_job_cache_build", True))
        and int(data_loader.get("num_workers", 0)) >= 2
        and runtime_required_output_keys <= set(output_paths)
    )
    checks.append(
        check_row(
            "SUITE-010",
            "Runtime protocol declares CPU cache precompute, fail-closed cache loading, loader knobs, and runtime audits.",
            runtime_condition,
            [
                "commands=" + ",".join(sorted(runtime_commands)),
                f"cache_enabled={cache_policy.get('enabled', False)}",
                f"cache_require={cache_policy.get('require_cache_for_formal', False)}",
                f"cache_training_rebuild={cache_policy.get('allow_training_job_cache_build', True)}",
                f"data_loader_workers={data_loader.get('num_workers', 0)}",
                "missing_runtime_outputs=" + ",".join(sorted(runtime_required_output_keys - set(output_paths))),
            ],
            "Declare cache_precompute/runtime audit paths and require formal training jobs to fail on missing or stale tensor caches.",
        )
    )

    manifest_entries = expected_manifest_entries(config)
    checks.append(
        pending_row(
            "SUITE-009",
            "Postrun task/model/seed artifact manifests are expected after execution.",
            [f"expected_manifest_entry_count={len(manifest_entries)}"],
            "Run the suite and then validate with --mode postrun.",
        )
    )

    failed_checks = [check for check in checks if check.status == FAIL]
    return {
        "audit_name": "paper_grade_full_suite_manifest",
        "mode": "prelaunch",
        "config_path": repo_relative(artifact_path(config_path)),
        "status": "pass" if not failed_checks else "fail",
        "failed_check_count": len(failed_checks),
        "pending_check_count": sum(1 for check in checks if check.status == PENDING),
        "checks": [asdict(check) for check in checks],
        "expected_manifest_entries": manifest_entries,
        "baseline_readiness": baseline_report,
    }


def build_postrun_report(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    failures, rows = postrun_failures(config)
    return {
        "audit_name": "paper_grade_full_suite_execution_summary",
        "mode": "postrun",
        "config_path": repo_relative(artifact_path(config_path)),
        "status": "pass" if not failures else "fail",
        "failure_count": len(failures),
        "failures": failures,
        "artifact_manifest_count": len(rows),
        "artifact_manifests": rows,
    }


def markdown_report(report: dict[str, object]) -> str:
    title = "Paper-Grade Full Suite Manifest" if report["mode"] == "prelaunch" else "Paper-Grade Full Suite Execution Summary"
    lines = [
        f"# {title}",
        "",
        f"- mode: `{report['mode']}`",
        f"- status: `{report['status']}`",
    ]
    if report["mode"] == "prelaunch":
        lines.extend(
            [
                f"- failed_check_count: `{report['failed_check_count']}`",
                f"- pending_check_count: `{report['pending_check_count']}`",
                "",
                "| check_id | status | severity | check_name | evidence | required_fix |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in report["checks"]:
            evidence = "<br>".join(str(value) for value in row["evidence"])
            lines.append(
                f"| {row['check_id']} | {row['status']} | {row['severity']} | "
                f"{row['check_name']} | {evidence} | {row['required_fix']} |"
            )
    else:
        lines.extend(
            [
                f"- failure_count: `{report['failure_count']}`",
                f"- artifact_manifest_count: `{report['artifact_manifest_count']}`",
                "",
                "| task_name | model_key | seed | status | manifest_path |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in report["artifact_manifests"]:
            lines.append(
                f"| {row['task_name']} | {row['model_key']} | {row['seed']} | {row['status']} | {row['manifest_path']} |"
            )
    return "\n".join(lines) + "\n"


def write_report_outputs(config: dict[str, object], report: dict[str, object], json_output: str, md_output: str) -> None:
    write_json(artifact_path(json_output), report)
    write_text_atomic(artifact_path(md_output), markdown_report(report))
    outputs = suite_config(config)["output_paths"]
    if report["mode"] == "prelaunch":
        baseline_report = report["baseline_readiness"]
        write_json(artifact_path(outputs["baseline_readiness_json"]), baseline_report)
        write_text_atomic(artifact_path(outputs["baseline_readiness_md"]), baseline_markdown(baseline_report))
    else:
        write_json(
            artifact_path(outputs["paper_grade_result_summary_json"]),
            {
                "summary_name": "paper_grade_result_summary",
                "suite_execution_status": report["status"],
                "artifact_manifest_count": report["artifact_manifest_count"],
                "failure_count": report["failure_count"],
                "failures": report["failures"],
            },
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = suite_config(config)["output_paths"]
    if args.mode == "prelaunch":
        default_json = str(outputs["manifest_json"])
        default_md = str(outputs["manifest_md"])
        report = build_prelaunch_report(args.config)
    else:
        default_json = str(outputs["execution_summary_json"])
        default_md = str(outputs["execution_summary_md"])
        report = build_postrun_report(args.config)
    json_output = args.json_output or default_json
    md_output = args.md_output or default_md
    write_report_outputs(config, report, json_output, md_output)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
