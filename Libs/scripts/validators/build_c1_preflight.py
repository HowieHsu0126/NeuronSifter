"""Build the unified, fail-closed C1 preflight from live evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_c1_experiment_registries import build as build_registries  # noqa: E402
from Libs.scripts.builders.build_c1_repository_inventory import build as build_inventory  # noqa: E402
from Libs.utils.data.foundation_state_c1 import build_c1_preflight_report  # noqa: E402
from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    derive_c1_headline_feasibility,
    load_c1_objective_capability_manifest,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.exp_utils import C1_RUNTIME_GIT_PATHS, get_git_provenance  # noqa: E402


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    path = path if path.is_absolute() else repo_root / path
    path = path.resolve()
    path.relative_to(repo_root.resolve())
    return path


def _run_training_started(
    repo_root: Path,
    results_root: str | Path,
) -> tuple[bool, bool, int]:
    development_started = False
    headline_started = False
    completed = 0
    for path in sorted(_repo_path(repo_root, results_root).glob("*/run_result.json")):
        try:
            payload = _load(path)
        except (OSError, TypeError, ValueError):
            continue
        if payload.get("training_started") is True:
            development_started = True
            if payload.get("run_kind") == "headline" or payload.get("headline") is True:
                headline_started = True
        if payload.get("status") == "passed":
            completed += 1
    return development_started, headline_started, completed


def _headline_capability_requirements(
    reuse_map: Mapping[str, object], required: list[str]
) -> list[dict[str, object]]:
    rows = {
        str(row["capability"]): row
        for row in reuse_map.get("capabilities", [])
        if isinstance(row, Mapping) and "capability" in row
    }
    requirements: list[dict[str, object]] = []
    for capability in required:
        row = rows.get(capability)
        status = "missing" if row is None else str(row.get("implementation_status", "unknown"))
        if status != "implemented":
            requirements.append(
                {
                    "code": "headline_capability_incomplete",
                    "capability": capability,
                    "implementation_status": status,
                    "reason": "capability is not headline-ready in the live inventory"
                    if row is None
                    else str(row.get("reason", "")),
                }
            )
    return requirements


def build(
    repo_root: Path = REPO_ROOT,
    *,
    config_path: str | Path = "Libs/configs/config_stack.yaml",
    regression_summary_path: str | Path = "Output/audits/c1_pretraining_regression_summary.json",
    results_root: str | Path = "Output/results/c1/runs",
    regression_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config_path = _repo_path(repo_root, config_path)
    config = load_config(config_path)
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    execution = contract["execution"]
    corpus = build_c1_preflight_report(config, repo_root)
    task_registry, baseline_registry, fairness, config_registry = build_registries(
        repo_root,
        config_path=config_path,
    )
    _, reuse_map, _ = build_inventory(repo_root, config_path=config_path)
    regression = (
        dict(regression_override)
        if regression_override is not None
        else _load(_repo_path(repo_root, regression_summary_path))
    )
    live_config_file_hash = sha256_file(config_path)
    live_resolved_config_hash = canonical_sha256(config)
    live_contract_hash = canonical_sha256(contract)
    live_git_provenance = get_git_provenance(C1_RUNTIME_GIT_PATHS)
    regression_binding = {
        "config_file_hash": {
            "current_hash": live_config_file_hash,
            "regression_hash": regression.get("config_sha256"),
            "matches": regression.get("config_sha256") == live_config_file_hash,
        },
        "resolved_config_hash": {
            "current_hash": live_resolved_config_hash,
            "regression_hash": regression.get("resolved_config_hash"),
            "matches": regression.get("resolved_config_hash") == live_resolved_config_hash,
        },
        "foundation_state_contract_hash": {
            "current_hash": live_contract_hash,
            "regression_hash": regression.get("foundation_state_contract_hash"),
            "matches": regression.get("foundation_state_contract_hash") == live_contract_hash,
        },
        "runtime_source_diff_hash": {
            "current_hash": live_git_provenance["git_full_diff_hash"],
            "regression_hash": regression.get("git_full_diff_hash"),
            "matches": regression.get("git_full_diff_hash")
            == live_git_provenance["git_full_diff_hash"],
        },
        "corpus_hash": {
            "current_hash": frozen["identity_hash"],
            "regression_hash": regression.get("corpus_hash"),
            "matches": regression.get("corpus_hash") == frozen["identity_hash"],
        },
    }
    regression_binding["all_match"] = all(
        bool(binding["matches"])
        for binding in regression_binding.values()
        if isinstance(binding, Mapping)
    )
    bundle = _load(repo_root / frozen["evidence_bundle_path"])
    evidence = bundle["evidence"]
    isolation_path = repo_root / evidence["controlled_isolation"]["path"]
    isolation = _load(isolation_path)
    evidence_validation_path = repo_root / execution["evidence_validation_path"]
    evidence_validation = _load(evidence_validation_path)
    observation_artifact = _load(repo_root / evidence["canonical_observation_blocks"]["path"])
    family_ess: dict[str, float] = {}
    for row in observation_artifact["blocks"]:
        if row.get("observation_eligible") is True and row.get("qc_grade") in {"A", "B"}:
            family = str(row["family_id"])
            family_ess[family] = family_ess.get(family, 0.0) + float(row["effective_sample_size"])

    capability_path = repo_root / contract["artifacts"]["objective_capability_manifest_path"]
    capability_manifest = load_c1_objective_capability_manifest(
        config, repo_root, verify_live=True
    )
    capability_hash = sha256_file(capability_path)
    headline_feasibility = derive_c1_headline_feasibility(
        capability_manifest,
        task_registry,
        runtime_profile_id=str(execution["headline_runtime_profile_id"]),
        required_task_family_count=int(contract["acceptance"]["minimum_distinct_task_families"]),
        required_holdout_family_count=int(contract["acceptance"]["minimum_distinct_holdout_families"]),
    )

    blockers: list[dict[str, object]] = []
    if corpus["status"] != "passed":
        blockers.append({"code": "c1_corpus_preflight_not_passed", "path": "c1_foundation_corpus_preflight", "message": str(corpus["blockers"])})
    if (
        regression.get("status") != "passed"
        or regression.get("terminal") is not True
        or regression.get("snapshot_stable_during_regression") is not True
    ):
        blockers.append({"code": "c1_full_regression_not_passed", "path": "c1_pretraining_regression_summary", "message": str(regression.get("status"))})
    for field, code in (
        ("config_file_hash", "c1_regression_config_file_hash_mismatch"),
        ("resolved_config_hash", "c1_regression_resolved_config_hash_mismatch"),
        ("foundation_state_contract_hash", "c1_regression_contract_hash_mismatch"),
        ("runtime_source_diff_hash", "c1_regression_runtime_source_hash_mismatch"),
        ("corpus_hash", "c1_regression_corpus_hash_mismatch"),
    ):
        binding = regression_binding[field]
        if binding["matches"] is not True:
            blockers.append({
                "code": code,
                "path": "c1_pretraining_regression_summary",
                "message": (
                    f"regression={binding['regression_hash']} current={binding['current_hash']}"
                ),
            })
    if evidence_validation.get("status") != "valid_fail_closed":
        blockers.append({"code": "c1_evidence_validator_not_valid_fail_closed", "path": execution["evidence_validation_path"], "message": str(evidence_validation.get("status"))})
    if isolation.get("status") != "pass" or isolation.get("controlled_reachability_count") != 0:
        blockers.append({"code": "c1_controlled_reachability_nonzero", "path": evidence["controlled_isolation"]["path"], "message": str(isolation.get("controlled_reachability_count"))})
    if sha256_file(repo_root / evidence["source_qualification"]["path"]) != evidence["source_qualification"]["sha256"]:
        blockers.append({"code": "c1_source_qualification_hash_mismatch", "path": evidence["source_qualification"]["path"], "message": "content drift"})
    if capability_hash != str(contract["artifacts"]["objective_capability_manifest_sha256"]):
        blockers.append({"code": "c1_capability_manifest_hash_mismatch", "path": contract["artifacts"]["objective_capability_manifest_path"], "message": "content drift"})

    headline_requirements = _headline_capability_requirements(
        reuse_map, [str(value) for value in execution["required_headline_capabilities"]]
    )
    headline_requirements.extend(headline_feasibility["blockers"])
    development_started, headline_started, completed_runs = _run_training_started(
        repo_root,
        results_root,
    )
    if headline_started and headline_requirements:
        blockers.append({
            "code": "c1_headline_training_started_while_requirements_incomplete",
            "path": "Output/results/c1/runs",
            "message": "headline run evidence exists while headline execution is structurally blocked",
        })
    if blockers:
        g3_status = "blocked_incomplete"
    else:
        g3_status = "blocked_incomplete" if headline_requirements else "not_evaluated"
    report = {
        "schema_version": "neuronsifter-c1-unified-preflight-v3",
        "status": "passed" if not blockers else "blocked",
        "training_allowed": not blockers,
        "headline_training_allowed": not blockers and not headline_requirements,
        "training_started": development_started,
        "headline_training_started": headline_started,
        "corpus_uid": frozen["corpus_uid"],
        "corpus_identity_hash": frozen["identity_hash"],
        "block_hashes": dict(frozen["required_blocks"]),
        "split_hashes": dict(frozen["required_split_manifests"]),
        "action_subset_hash": frozen["qualified_action_subset_hash"],
        "crosswalk_hash": frozen["required_crosswalk"]["table_sha256"],
        "source_qualification_evidence_hash": evidence["source_qualification"]["sha256"],
        "controlled_isolation_evidence": {
            "frozen_reference_hash": evidence["controlled_isolation"]["sha256"],
            "current_hash": sha256_file(isolation_path),
            "controlled_reachability": isolation["controlled_reachability_count"],
        },
        "evidence_validator_status": evidence_validation.get("status"),
        "eligible_roles": list(frozen["required_roles"]),
        "rejected_roles": list(frozen["forbidden_roles"]),
        "family_ess": dict(sorted(family_ess.items())),
        "sampling_policy": {
            "family": contract["loader"]["sampling_policy"],
            "biological_unit": contract["loader"]["weighting_policy"],
            "within_unit": contract["loader"]["within_unit_aggregation"],
        },
        "model_config_hash": canonical_sha256(contract["encoder"]),
        "training_seed_list": list(contract["training"]["headline_seeds"]),
        "task_registry_hash": task_registry["registry_hash"],
        "baseline_registry_hash": baseline_registry["registry_hash"],
        "fairness_registry_hash": fairness["registry_hash"],
        "config_hash": live_resolved_config_hash,
        "config_registry_hash": config_registry["registry_hash"],
        "contract_hash": live_contract_hash,
        "capability_manifest_hash": capability_hash,
        "regression_binding": regression_binding,
        "regression_run_uid": regression.get("run_uid"),
        "regression_slurm_job_id": regression.get("slurm_job_id"),
        "completed_run_count": completed_runs,
        "headline_feasibility": headline_feasibility,
        "headline_requirements": headline_requirements,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "g3_status": g3_status,
    }
    schema_path = repo_root / contract["artifacts"]["unified_preflight_schema_path"]
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError("C1 unified preflight schema violation: " + "; ".join(error.message for error in errors))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--regression-summary",
        default="Output/audits/c1_pretraining_regression_summary.json",
    )
    parser.add_argument("--results-root", default="Output/results/c1/runs")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build(
        REPO_ROOT,
        config_path=args.config,
        regression_summary_path=args.regression_summary,
        results_root=args.results_root,
    )
    output = args.output or str(
        load_config(_repo_path(REPO_ROOT, args.config))["foundation_state_c1"][
            "execution"
        ]["unified_preflight_path"]
    )
    write_json(_repo_path(REPO_ROOT, output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
