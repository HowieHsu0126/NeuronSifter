"""Build provenance-bound C1 execution and evidence catalogs.

The matrix is derived from the live capability manifest.  A row is never
marked submit-ready merely because its baseline and split appear in a static
registry; objective feasibility and the unified headline preflight must both
allow it.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_c1_experiment_registries import build as build_registries  # noqa: E402
from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    load_c1_objective_capability_manifest,
    resolve_c1_single_source_selection,
    select_c1_sampler_structural_feasibility,
)
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


RESULT_FIELDS = [
    "claim_id", "experiment_id", "task_id", "holdout_id", "model_id",
    "checkpoint_hash", "corpus_hash", "split_hash", "seed", "metric",
    "value", "unit", "baseline", "paired_delta", "ci_low", "ci_high",
    "p_adjustment", "bootstrap_unit", "status", "provenance",
]


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_tsv(path: Path, rows: list[Mapping[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _result_row(*, claim_id: str, experiment_id: str, task_id: str,
                holdout_id: str, model_id: str, baseline: str,
                corpus_hash: str, split_hash: str, seed: int,
                provenance: str, status: str,
                **extra: object) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "holdout_id": holdout_id,
        "model_id": model_id,
        "checkpoint_hash": extra.get("checkpoint_hash", ""),
        "corpus_hash": corpus_hash,
        "split_hash": split_hash,
        "seed": seed,
        "metric": extra.get("metric", ""),
        "value": extra.get("value", ""),
        "unit": extra.get("unit", ""),
        "baseline": baseline,
        "paired_delta": extra.get("paired_delta", ""),
        "ci_low": extra.get("ci_low", ""),
        "ci_high": extra.get("ci_high", ""),
        "p_adjustment": extra.get("p_adjustment", ""),
        "bootstrap_unit": extra.get("bootstrap_unit", ""),
        "status": status,
        "provenance": provenance,
    }


def _holdout_family(split_uid: str) -> str:
    if split_uid == "source_publication_holdout":
        return "source_publication_holdout"
    if split_uid.startswith("family_holdout_"):
        return "family_holdout"
    return split_uid


def _task_families_for_split(tasks: Mapping[str, object], split_uid: str) -> list[str]:
    return sorted(
        str(task["task_family"])
        for task in tasks["tasks"]
        if split_uid in [str(value) for value in task.get("applicable_holdouts", [])]
    )


def _variant_rows(
    baseline: Mapping[str, object],
    contract: Mapping[str, object],
) -> list[tuple[str, str | None, str | None]]:
    baseline_code = str(baseline["baseline_code"])
    if baseline_code == "B2":
        variants = contract["baselines"]["single_family_selection"]["variants"]
        return [(str(name), None, str(family)) for name, family in variants.items()]
    return [("", None, None)]


def _scope_feasibility(
    manifest: Mapping[str, object],
    baseline: Mapping[str, object],
    split_uid: str,
    *,
    family_uid: str | None,
) -> tuple[str, str, str | None, str | None, Mapping[str, object] | None]:
    """Return scope, structural status, selected source/family and evidence."""

    runtime_profile_id = str(baseline["runtime_profile_id"])
    baseline_code = str(baseline["baseline_code"])
    if runtime_profile_id == "" or baseline_code == "B0":
        return "not_applicable", "not_applicable", None, family_uid, None
    source_id: str | None = None
    if baseline_code == "B1":
        try:
            source_id = str(resolve_c1_single_source_selection(manifest, split_uid)["selected_source"])
        except (KeyError, TypeError, ValueError):
            return "single_source", "unavailable", None, family_uid, None
    try:
        feasibility = select_c1_sampler_structural_feasibility(
            manifest,
            split_uid,
            runtime_profile_id,
            source_id=source_id,
            family_uid=family_uid,
        )
    except (KeyError, TypeError, ValueError):
        scope = f"source:{source_id}" if source_id else (f"family:{family_uid}" if family_uid else "all_sources")
        return scope, "unavailable", source_id, family_uid, None
    scope = f"source:{source_id}" if source_id else (f"family:{family_uid}" if family_uid else "all_sources")
    return scope, str(feasibility.get("status", "unavailable")), source_id, family_uid, feasibility


def _job_row(
    *,
    baseline: Mapping[str, object],
    contract: Mapping[str, object],
    preflight: Mapping[str, object],
    manifest: Mapping[str, object],
    split_uid: str,
    split_hash: str,
    seed: int,
    variant: str,
    family_uid: str | None,
    task_family_ids: list[str],
) -> dict[str, object]:
    execution = contract["execution"]
    baseline_code = str(baseline["baseline_code"])
    baseline_id = str(baseline["baseline_id"])
    runtime_profile_id = str(baseline["runtime_profile_id"])
    scope, structural_status, source_id, selected_family, feasibility = _scope_feasibility(
        manifest, baseline, split_uid, family_uid=family_uid
    )
    blockers: list[str] = []
    if structural_status == "not_applicable":
        blockers.append("no_pretraining_sampler_required")
    elif feasibility is None:
        blockers.append("objective_capability_unavailable")
    elif structural_status != "structurally_feasible":
        blockers.append(str(feasibility.get("blocker", "structural_feasibility_not_met")))
    if preflight.get("blockers"):
        blockers.extend(str(row.get("code", "c1_preflight_blocker")) for row in preflight["blockers"])
    if preflight.get("headline_requirements"):
        blockers.extend(str(row.get("code", "headline_requirement")) for row in preflight["headline_requirements"])
    blockers = sorted(set(blockers))
    training_allowed = structural_status == "structurally_feasible" and preflight.get("training_allowed") is True
    if structural_status == "not_applicable":
        status = "not_applicable"
    elif not training_allowed:
        status = "blocked_preflight"
    elif preflight.get("headline_training_allowed") is not True:
        status = "blocked_implementation" if any(code == "headline_capability_incomplete" for code in blockers) else "blocked_gate"
    else:
        status = "planned"
    submission_allowed = status == "planned"
    semantic = {
        "baseline_code": baseline_code,
        "baseline_variant": variant,
        "split_uid": split_uid,
        "seed": seed,
        "model_scale": execution["headline_model_scale"],
        "corpus_hash": preflight["corpus_identity_hash"],
    }
    digest = _hash_payload(semantic)
    run_uid = f"c1_{baseline_code.lower()}_{seed}_{digest[:16]}"
    job_name = str(execution["job_name_prefix"]) + run_uid
    if len(job_name) > int(execution["job_name_max_length"]):
        raise ValueError(f"C1 job name exceeds configured limit: {job_name}")
    return {
        "run_uid": run_uid,
        "baseline_code": baseline_code,
        "baseline_id": baseline_id,
        "runtime_profile_id": runtime_profile_id,
        "baseline_variant": variant,
        "model_scale": execution["headline_model_scale"],
        "split_uid": split_uid,
        "split_hash": split_hash,
        "seed": seed,
        "job_name": job_name,
        "resource_profile": execution["headline_resource_profile"],
        "run_stage": "pretraining",
        "status": status,
        "structural_status": structural_status,
        "training_allowed": training_allowed,
        "submission_allowed": submission_allowed,
        "blocker_codes": blockers,
        "source_id": source_id or "",
        "family_uid": selected_family or "",
        "task_family_ids": task_family_ids,
        "holdout_family": _holdout_family(split_uid),
        "capability_manifest_hash": preflight["capability_manifest_hash"],
        "capability_scope": scope,
        "capability_blocker": str(feasibility.get("blocker", "")) if feasibility else "",
        "corpus_hash": preflight["corpus_identity_hash"],
    }


def _load_run_evidence(
    repo_root: Path,
    preflight: Mapping[str, object],
    attempt_directory: Path,
    baseline_registry: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    terminal_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    completed_runs = 0
    headline_completed = 0
    seen: set[str] = set()
    public_by_runtime = {
        str(row["runtime_profile_id"]): str(row["baseline_id"])
        for row in baseline_registry["baselines"]
    }
    paths = sorted((repo_root / "Output/results/c1/runs").glob("*/run_result.json"))
    paths.extend(sorted(attempt_directory.glob("*.json")))
    for result_path in paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run_uid = str(payload.get("run_uid") or result_path.parent.name or result_path.stem)
        if run_uid in seen:
            continue
        seen.add(run_uid)
        status = str(payload.get("status", payload.get("terminal_status", "unknown")))
        training_started = payload.get("training_started") is True
        run_kind = str(payload.get("run_kind", "headline" if payload.get("headline") is True else "development_smoke"))
        raw_baseline_id = str(payload.get("baseline_id", ""))
        runtime_profile_id = str(payload.get("runtime_profile_id") or raw_baseline_id)
        baseline_id = public_by_runtime.get(runtime_profile_id, raw_baseline_id)
        terminal_rows.append({
            "run_uid": run_uid,
            "status": status,
            "result_status": payload.get("result_status", ""),
            "run_kind": run_kind,
            "baseline_id": baseline_id,
            "runtime_profile_id": runtime_profile_id,
            "model_scale": payload.get("model_scale", ""),
            "seed": payload.get("seed", ""),
            "corpus_hash": payload.get("corpus_hash", preflight["corpus_identity_hash"]),
            "split_hash": payload.get("split_hash", ""),
            "checkpoint_hash": payload.get("checkpoint_hash", ""),
            "slurm_job_id": payload.get("slurm_job_id", ""),
            "slurm_job_name": payload.get("slurm_job_name", payload.get("job_name", "")),
            "resource_profile": payload.get("resource_profile", ""),
            "host": payload.get("host", payload.get("node", "")),
            "started_at": payload.get("started_at", payload.get("start_time", "")),
            "ended_at": payload.get("ended_at", payload.get("end_time", "")),
            "exit_code": payload.get("exit_code", ""),
            "git_commit": payload.get("git_commit", ""),
            "git_dirty": payload.get("git_dirty", ""),
            "config_hash": payload.get("config_hash", ""),
            "scheduler_stdout_path": payload.get("scheduler_stdout_path", ""),
            "scheduler_stderr_path": payload.get("scheduler_stderr_path", ""),
            "provenance": result_path.relative_to(repo_root).as_posix(),
        })
        if status == "passed" and training_started:
            completed_runs += 1
            if run_kind == "headline":
                headline_completed += 1
        if status in {"failed", "error", "cancelled", "FAILED"}:
            failure_rows.append({
                "run_uid": run_uid,
                "status": "failed",
                "baseline_id": baseline_id,
                "split_hash": payload.get("split_hash", ""),
                "error_type": payload.get("error_type", payload.get("failure_type", "")),
                "error": payload.get("error", payload.get("reason", "")),
                "slurm_job_id": payload.get("slurm_job_id", ""),
                "exit_code": payload.get("exit_code", ""),
                "scheduler_stdout_path": payload.get("scheduler_stdout_path", ""),
                "scheduler_stderr_path": payload.get("scheduler_stderr_path", ""),
                "provenance": result_path.relative_to(repo_root).as_posix(),
            })
    return terminal_rows, failure_rows, completed_runs, headline_completed


def _gate_payload(
    *,
    preflight: Mapping[str, object],
    terminal_rows: list[Mapping[str, object]],
    failure_rows: list[Mapping[str, object]],
    completed_runs: int,
    headline_completed: int,
) -> dict[str, object]:
    feasibility = preflight["headline_feasibility"]
    headline_requirements = preflight["headline_requirements"]
    status = "blocked_incomplete" if preflight["g3_status"] == "blocked_incomplete" else "not_evaluated"
    return {
        "schema_version": "neuronsifter-c1-foundation-transfer-gate-v2",
        "gate_id": "G3_foundation_transfer_evidence",
        "status": status,
        "corpus_hash": preflight["corpus_identity_hash"],
        "preflight_hash": canonical_sha256(preflight),
        "capability_manifest_hash": preflight["capability_manifest_hash"],
        "training_started": preflight["training_started"],
        "headline_training_started": preflight["headline_training_started"],
        "development_training_status": "smoke_only" if completed_runs else "not_started",
        "empirical_result_status": "smoke_only" if completed_runs else "not_evaluated",
        "completed_run_count": completed_runs,
        "headline_completed_run_count": headline_completed,
        "failed_run_count": len(failure_rows),
        "valid_task_family_count": 0,
        "valid_holdout_family_count": 0,
        "maximum_reachable_task_family_count": feasibility["maximum_reachable_task_family_count"],
        "maximum_reachable_holdout_family_count": feasibility["maximum_reachable_holdout_family_count"],
        "minimum_task_family_count": feasibility["minimum_task_family_count"],
        "minimum_holdout_family_count": feasibility["minimum_holdout_family_count"],
        "coverage_reachable": feasibility["status"] == "reachable",
        "integrity_blocker_count": 0,
        "blocking_requirement_codes": sorted({str(row.get("code", "")) for row in headline_requirements}),
        "claim_allowed": False,
        "claim_scope_permission": "reduced_core_implementation_and_real_data_smoke_only",
        "reason": "headline C1 coverage and/or required execution components are incomplete; no empirical transfer claim is allowed",
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    contract = config["foundation_state_c1"]
    preflight = json.loads((repo_root / contract["execution"]["unified_preflight_path"]).read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("blockers"):
        raise ValueError("C1 execution artifacts require a passed, blocker-free corpus preflight")
    tasks, baselines, _, _ = build_registries(repo_root)
    manifest = load_c1_objective_capability_manifest(config, repo_root, verify_live=True)
    corpus_hash = str(preflight["corpus_identity_hash"])
    split_hashes = {str(k): str(v) for k, v in preflight["split_hashes"].items()}
    seeds = [int(value) for value in preflight["training_seed_list"]]
    output_audits = repo_root / "Output/audits"
    output_reports = repo_root / "Output/paper_reports"
    output_runs = repo_root / "Output/results/c1/runs"
    attempt_directory = repo_root / contract["execution"]["attempt_artifact_directory"]
    output_audits.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)
    attempt_directory.mkdir(parents=True, exist_ok=True)

    job_rows: list[dict[str, object]] = []
    for baseline in baselines["baselines"]:
        for split_uid, split_hash in split_hashes.items():
            task_family_ids = _task_families_for_split(tasks, split_uid)
            for variant, _, configured_family in _variant_rows(baseline, contract):
                for seed in seeds:
                    job_rows.append(_job_row(
                        baseline=baseline,
                        contract=contract,
                        preflight=preflight,
                        manifest=manifest,
                        split_uid=split_uid,
                        split_hash=split_hash,
                        seed=seed,
                        variant=variant,
                        family_uid=configured_family,
                        task_family_ids=task_family_ids,
                    ))

    matrix_payload = {
        "schema_version": "neuronsifter-c1-job-matrix-v2",
        "status": "frozen",
        "submission_status": "allowed" if preflight["headline_training_allowed"] else "blocked_incomplete",
        "corpus_hash": corpus_hash,
        "capability_manifest_hash": preflight["capability_manifest_hash"],
        "preflight_hash": canonical_sha256(preflight),
        "split_hashes": split_hashes,
        "seed_list": seeds,
        "resource_profiles": [contract["execution"]["headline_resource_profile"], *contract["execution"]["development_resource_profiles"]],
        "headline_model_scale": contract["execution"]["headline_model_scale"],
        "headline_feasibility": preflight["headline_feasibility"],
        "jobs": job_rows,
        "matrix_hash": _hash_payload(job_rows),
    }
    schema = json.loads((repo_root / contract["execution"]["execution_schema_path"]).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(matrix_payload))
    if errors:
        raise ValueError("C1 execution matrix schema violation: " + "; ".join(error.message for error in errors))
    write_json(output_audits / "c1_job_matrix.json", matrix_payload)

    terminal_rows, failure_rows, completed_runs, headline_completed = _load_run_evidence(
        repo_root, preflight, attempt_directory, baselines
    )
    _write_tsv(output_audits / "c1_job_terminal_summary.tsv", terminal_rows, [
        "run_uid", "status", "result_status", "run_kind", "baseline_id", "runtime_profile_id",
        "model_scale", "seed", "corpus_hash", "split_hash", "checkpoint_hash",
        "slurm_job_id", "slurm_job_name", "resource_profile", "host", "started_at", "ended_at",
        "exit_code", "git_commit", "git_dirty", "config_hash", "scheduler_stdout_path",
        "scheduler_stderr_path", "provenance",
    ])
    _write_tsv(output_audits / "c1_failure_catalog.tsv", failure_rows,
               ["run_uid", "status", "baseline_id", "split_hash", "error_type", "error",
                "slurm_job_id", "exit_code", "scheduler_stdout_path", "scheduler_stderr_path",
                "provenance"])

    task_rows = []
    for task in tasks["tasks"]:
        for task_id in task["task_ids"]:
            for holdout_id in task["applicable_holdouts"]:
                split_hash = split_hashes.get(str(holdout_id), "")
                task_rows.append(_result_row(
                    claim_id="C1",
                    experiment_id=f"c1_{task['task_family']}_{task_id}_{holdout_id}",
                    task_id=str(task_id),
                    holdout_id=str(holdout_id),
                    model_id="neuronsifter_reduced_core_full",
                    baseline="B8",
                    corpus_hash=corpus_hash,
                    split_hash=split_hash,
                    seed=seeds[0],
                    provenance="frozen task registry + live capability/preflight; empirical result absent",
                    status="blocked_incomplete" if not preflight["headline_training_allowed"] else "requires_experiment_result",
                    bootstrap_unit=str(task["biological_unit"]),
                ))
    _write_tsv(output_reports / "c1_foundation_transfer.tsv", task_rows, RESULT_FIELDS)
    for name in (
        "c1_few_shot_results.tsv", "c1_scaling_results.tsv", "c1_source_leakage_results.tsv",
        "c1_baseline_result_matrix.tsv", "c1_ablation_result_catalog.tsv", "c1_claim_metric_catalog.tsv",
    ):
        _write_tsv(output_reports / name, [], RESULT_FIELDS)

    source_rows = [{
        "diagnostic_id": "source_shortcut_diagnostics",
        "status": "blocked_incomplete",
        "metric": "chance_normalized_balanced_accuracy",
        "tolerance": contract["evaluation"]["source_shortcut_tolerance"],
        "permutation_seed": 3407,
        "provenance": "C1 preregistration; independent shortcut probe is not implemented",
    }]
    _write_tsv(output_audits / "c1_source_shortcut_diagnostic.tsv", source_rows,
               ["diagnostic_id", "status", "metric", "tolerance", "permutation_seed", "provenance"])

    scaling = {
        "schema_version": "neuronsifter-c1-scaling-transfer-summary-v1",
        "status": "blocked_incomplete",
        "corpus_hash": corpus_hash,
        "effective_biological_unit_fractions": contract["scaling"]["biological_unit_fractions"],
        "source_family_counts": contract["scaling"]["source_family_counts"],
        "model_scales": contract["scaling"]["model_scales"],
        "objective_sets": contract["scaling"]["objective_sets"],
        "result_count": 0,
        "reason": "headline coverage is structurally unreachable under the frozen objective capability manifest",
    }
    write_json(output_audits / "c1_scaling_transfer_summary.json", scaling)
    write_json(repo_root / "Output/results/c1/scaling_transfer_summary.json", scaling)

    gate = _gate_payload(
        preflight=preflight,
        terminal_rows=terminal_rows,
        failure_rows=failure_rows,
        completed_runs=completed_runs,
        headline_completed=headline_completed,
    )
    gate_schema = json.loads((repo_root / contract["execution"]["g3_gate_schema_path"]).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(gate_schema)
    gate_errors = list(Draft202012Validator(gate_schema).iter_errors(gate))
    if gate_errors:
        raise ValueError("C1 G3 gate schema violation: " + "; ".join(error.message for error in gate_errors))
    write_json(output_audits / "c1_foundation_transfer_gate.json", gate)
    write_json(output_reports / "c1_gate_summary.json", gate)
    return {
        "job_matrix": matrix_payload,
        "terminal_count": len(terminal_rows),
        "failure_count": len(failure_rows),
        "gate": gate,
    }


def main() -> None:
    result = build(REPO_ROOT)
    print(json.dumps({
        "status": result["gate"]["status"],
        "job_count": len(result["job_matrix"]["jobs"]),
        "submission_status": result["job_matrix"]["submission_status"],
        "terminal_count": result["terminal_count"],
        "failure_count": result["failure_count"],
        "matrix_hash": result["job_matrix"]["matrix_hash"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
