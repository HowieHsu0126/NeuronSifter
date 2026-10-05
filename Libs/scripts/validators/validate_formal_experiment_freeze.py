from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.g0_read_only_artifacts import ReadOnlyArtifactReference
from Libs.utils.data.g0_read_only_artifacts import artifact_metadata
from Libs.utils.data.g0_read_only_artifacts import resolve_read_only_artifact
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic
from Libs.utils.data.trajectory_numeric_lineage import load_canonical_metric_values


EXPECTED_G2_STATUS = "G2_pass_with_warnings"
EXPECTED_G2_GATE = "g2_100k"
EXPECTED_G5_STATUS = "G5_not_passed"
EXPECTED_G5_GATE_STATUS = "not_passed"

FREEZE_JSON_OUTPUT = "Output/audits/formal_experiment_freeze_manifest.json"
FREEZE_MD_OUTPUT = "Output/audits/formal_experiment_freeze_manifest.md"
GENERATED_AUDIT_OUTPUTS = {
    "TODO.md",
    "Output/logs/pipeline/prepare_experiment_config/",
    "Output/logs/print_only/",
    "Output/paper_configs/paper_grade_full_suite/materialized_config.yaml",
    "Output/paper_configs/print_only/",
    "Output/audits/paper_grade_full_suite_prelive_summary.json",
    "Output/audits/paper_grade_full_suite_prelive_summary.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the current formal NeuronSifter experiment state.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--json-output", default=FREEZE_JSON_OUTPUT)
    parser.add_argument("--md-output", default=FREEZE_MD_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def artifact_path(path_value: object) -> Path:
    if isinstance(path_value, ReadOnlyArtifactReference):
        return path_value.physical_path
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_json_object(path_value: object) -> dict[str, object]:
    path = artifact_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def read_tsv_rows(path_value: object) -> list[dict[str, str]]:
    with artifact_path(path_value).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: list[str]) -> str:
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return completed.stdout


def command_sha256(args: list[str]) -> str:
    completed = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(args)}\n{completed.stderr.decode('utf-8', errors='replace')}")
    return sha256_bytes(completed.stdout)


def config_include_paths(config_path: str | Path) -> list[Path]:
    root_config_path = artifact_path(config_path)
    payload = yaml.safe_load(root_config_path.read_text(encoding="utf-8")) or {}
    includes = [root_config_path]
    for include_value in payload.get("config_includes", []):
        include_path = Path(str(include_value))
        if include_path.is_absolute():
            includes.append(include_path)
            continue
        parent_candidate = (root_config_path.parent / include_path).resolve()
        if parent_candidate.exists():
            includes.append(parent_candidate)
            continue
        includes.append((REPO_ROOT / include_path).resolve())
    return includes


def freeze_replay_config_path(config_path: str | Path) -> str | Path:
    config = load_config(config_path)
    replay_path = (
        config.get("paper_grade_full_suite", {})
        .get("freeze_replay_config_path", "")
    )
    if str(replay_path).strip():
        return str(replay_path)
    return config_path


def required_simulation_paths(config: dict[str, object]) -> dict[str, Path]:
    audit_config = config["ai_ready_preprocessing"]["simulation_audit"]
    root = resolve_repo_path(str(audit_config["local_root"]))
    return {
        str(name): root / str(path_value)
        for name, path_value in dict(audit_config["relative_paths"]).items()
    }


def source_artifact_paths(config: dict[str, object]) -> dict[str, Path | ReadOnlyArtifactReference]:
    sim_paths = required_simulation_paths(config)
    return {
        "formal_experiment_readiness_audit": resolve_read_only_artifact(
            config, "formal_experiment_readiness_audit", REPO_ROOT
        ),
        "g5_effective_sample_audit": resolve_read_only_artifact(config, "g5_effective_sample_audit", REPO_ROOT),
        "simulation_qc_summary": sim_paths["qc_summary"],
        "simulation_pair_alignment_audit": sim_paths["pair_alignment_audit"],
        "simulation_qc_catalog": sim_paths["qc_catalog"],
    }


def classification_for_qc_row(row: dict[str, str] | None) -> str:
    if row is None:
        return "other"
    qc_grade = str(row.get("qc_grade", ""))
    qc_use = str(row.get("qc_use", ""))
    if qc_use == "reject" or qc_grade == "D":
        return "reject"
    if qc_use == "stress_test_only" or qc_grade == "S":
        return "stress_only"
    if qc_use == "weak_teacher_calibration" or qc_grade == "C":
        return "calibration"
    if qc_use in {"inventory_only", "catalog_only"}:
        return "inventory_only"
    return "other"


def classify_full_only_rows(pair_rows: list[dict[str, str]], qc_rows: list[dict[str, str]]) -> dict[str, object]:
    qc_by_source_uid = {str(row["source_uid"]): row for row in qc_rows}
    qc_by_paired_uid: dict[str, list[dict[str, str]]] = {}
    for row in qc_rows:
        qc_by_paired_uid.setdefault(str(row["paired_uid"]), []).append(row)
    full_only_rows = [row for row in pair_rows if str(row["paired_status"]) == "full_only"]
    classifications: list[dict[str, object]] = []
    counts = {
        "reject": 0,
        "stress_only": 0,
        "calibration": 0,
        "inventory_only": 0,
        "other": 0,
    }
    for row in full_only_rows:
        qc_row = qc_by_source_uid.get(str(row["full_source_uid"]))
        if qc_row is None:
            paired_rows = qc_by_paired_uid.get(str(row["paired_uid"]), [])
            qc_row = paired_rows[0] if paired_rows else None
        classification = classification_for_qc_row(qc_row)
        counts[classification] += 1
        classifications.append(
            {
                "paired_uid": str(row["paired_uid"]),
                "paired_status": str(row["paired_status"]),
                "full_source_uid": str(row["full_source_uid"]),
                "full_trajectory_index": str(row["full_trajectory_index"]),
                "trajectory_uid": str(qc_row.get("trajectory_uid", "")) if qc_row else "",
                "source_collection": str(qc_row.get("source_collection", "")) if qc_row else "",
                "trajectory_store_path": str(qc_row.get("trajectory_store_path", "")) if qc_row else "",
                "trajectory_index": str(qc_row.get("trajectory_index", "")) if qc_row else "",
                "qc_grade": str(qc_row.get("qc_grade", "")) if qc_row else "",
                "qc_use": str(qc_row.get("qc_use", "")) if qc_row else "",
                "sample_weight": float(qc_row.get("sample_weight", 0.0)) if qc_row else 0.0,
                "failure_flags": str(qc_row.get("failure_flags", "")) if qc_row else "",
                "classification": classification,
            }
        )
    return {
        "counts": counts,
        "rows": classifications,
        "full_only_count": len(full_only_rows),
    }


def training_exclusion_report(config: dict[str, object], full_only_rows: list[dict[str, object]], qc_rows: list[dict[str, str]]) -> dict[str, object]:
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    families = benchmark["training"]["teacher_family_training"]["families"]
    training_family_contracts = []
    for family in families:
        if bool(family["parameter_prior_only"]):
            continue
        training_family_contracts.append(
            {
                "family_key": str(family["family_key"]),
                "catalog_path": str(family["catalog_path"]),
                "qc_use_field": str(family.get("qc_use_field", "")),
                "allowed_qc_uses": [str(value) for value in family.get("allowed_qc_uses", [])],
                "source_collection": str(family["source_collection"]),
            }
        )
    full_only_main_pretraining_rows = [
        row for row in full_only_rows
        if str(row["qc_use"]) == "main_pretraining"
    ]
    non_main_pretraining_rows = [
        row for row in qc_rows
        if str(row.get("qc_use", "")) != "main_pretraining"
    ]
    main_pretraining_non_paired_rows = [
        row for row in qc_rows
        if str(row.get("qc_use", "")) == "main_pretraining" and str(row.get("paired_status", "")) != "paired"
    ]
    train_eligible_grade_ab_count = sum(
        1
        for row in qc_rows
        if str(row.get("qc_grade", "")) in {"A", "B"} and str(row.get("qc_use", "")) == "main_pretraining"
    )
    loader_text = artifact_path("Libs/exp/training/pretrain_microenvironment_world_model.py").read_text(encoding="utf-8")
    loader_filters_qc_use = "allowed_qc_uses" in loader_text and "qc_use_field" in loader_text and "not in allowed_qc_uses" in loader_text
    qc_catalog_contracts = [
        contract for contract in training_family_contracts
        if str(contract["catalog_path"]).endswith("simulation_qc_catalog.tsv")
    ]
    canonical_counts = load_canonical_metric_values(REPO_ROOT / "Libs/configs/config_stack.yaml")
    return {
        "train_eligible_grade_ab_count": train_eligible_grade_ab_count,
        "non_main_pretraining_row_count": len(non_main_pretraining_rows),
        "full_only_main_pretraining_row_count": len(full_only_main_pretraining_rows),
        "main_pretraining_non_paired_row_count": len(main_pretraining_non_paired_rows),
        "training_family_contracts": training_family_contracts,
        "loader_filters_qc_use": loader_filters_qc_use,
        "loader_path": "Libs/exp/training/pretrain_microenvironment_world_model.py",
        "exclusion_status": "pass"
        if (
            train_eligible_grade_ab_count == canonical_counts["qc_grade_a_b_count"]
            and not full_only_main_pretraining_rows
            and not main_pretraining_non_paired_rows
            and loader_filters_qc_use
            and qc_catalog_contracts
            and all(contract["allowed_qc_uses"] == ["main_pretraining"] for contract in qc_catalog_contracts)
        )
        else "fail",
    }


def git_status_entries(excluded_paths: set[str]) -> list[dict[str, object]]:
    output = command_output(["git", "status", "--porcelain=v1", "--untracked-files=normal"])
    entries: list[dict[str, object]] = []
    excluded_prefixes = [path for path in excluded_paths if path.endswith("/")]
    for line in output.splitlines():
        if not line:
            continue
        status_code = line[:2]
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if path_text in excluded_paths or any(path_text.startswith(prefix) for prefix in excluded_prefixes):
            continue
        path = REPO_ROOT / path_text
        entry = {
            "status": status_code,
            "path": path_text,
            "staged_diff_hash": command_sha256(["git", "diff", "--cached", "--binary", "--", path_text]),
            "unstaged_diff_hash": command_sha256(["git", "diff", "--binary", "--", path_text]),
            "working_tree_file_hash": "",
            "working_tree_file_state": "missing",
        }
        if path.is_file():
            entry["working_tree_file_hash"] = file_sha256(path)
            entry["working_tree_file_state"] = "file"
        elif path.exists():
            entry["working_tree_file_state"] = "directory_or_submodule"
        entries.append(entry)
    return entries


def untracked_file_paths(status_entries: list[dict[str, object]]) -> list[str]:
    return [
        str(entry["path"])
        for entry in status_entries
        if str(entry["status"]) == "??" and (REPO_ROOT / str(entry["path"])).is_file()
    ]


def full_worktree_hash(status_entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(status_entries, key=lambda payload: str(payload["path"])):
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(str(entry["status"]).encode("utf-8"))
        digest.update(str(entry["staged_diff_hash"]).encode("utf-8"))
        digest.update(str(entry["unstaged_diff_hash"]).encode("utf-8"))
        digest.update(str(entry["working_tree_file_hash"]).encode("utf-8"))
        digest.update(str(entry["working_tree_file_state"]).encode("utf-8"))
    return digest.hexdigest()


def git_state(excluded_paths: set[str]) -> dict[str, object]:
    status_entries = git_status_entries(excluded_paths)
    return {
        "git_head_sha": command_output(["git", "rev-parse", "HEAD"]).strip(),
        "dirty_worktree_allowed": True,
        "reproducibility_status": "dirty_worktree_recorded" if status_entries else "clean_worktree",
        "dirty_file_count": len(status_entries),
        "dirty_files": status_entries,
        "full_diff_hash": full_worktree_hash(status_entries),
    }


def slurm_command_manifest(config_path: str) -> list[dict[str, object]]:
    return [
        {
            "command_id": "cluster_state_precheck",
            "purpose": "Check allowed-node availability before formal execution.",
            "command": ["sinfo"],
        },
        {
            "command_id": "simulation_readiness_validator",
            "purpose": "Validate simulation QC, leakage, pairing, and training-use consistency.",
            "command": [
                "srun", "--partition=cpu", "--nodelist=cpu04", "--job-name=NrnSift_simulation_readiness_validator",
                "--cpus-per-task=2", "--mem=8G",
                "python", "Libs/scripts/validators/validate_simulation_readiness_audit.py", "--config", config_path,
            ],
        },
        {
            "command_id": "supplemental_contract_validator",
            "purpose": "Validate NeuronSifter, public-boundary, report, and milestone contracts.",
            "command": [
                "srun", "--partition=cpu", "--nodelist=cpu04", "--job-name=NrnSift_supplemental_contract_validator",
                "--cpus-per-task=2", "--mem=8G",
                "python", "Libs/scripts/validators/validate_supplemental_milestone_contracts.py", "--config", config_path,
            ],
        },
        {
            "command_id": "formal_readiness_validator",
            "purpose": "Gate formal experiment launch.",
            "command": [
                "srun", "--partition=cpu", "--nodelist=cpu04", "--job-name=NrnSift_formal_readiness_validator",
                "--cpus-per-task=2", "--mem=8G",
                "python", "Libs/scripts/validators/validate_formal_experiment_readiness.py", "--config", config_path,
            ],
        },
        {
            "command_id": "formal_freeze_replay_gate",
            "purpose": "Replay the frozen G2 signature without writing artifacts before formal launch.",
            "command": [
                "srun", "--partition=cpu", "--nodelist=cpu04", "--job-name=NrnSift_formal_freeze_replay_gate",
                "--cpus-per-task=2", "--mem=8G",
                "python", "Libs/scripts/validators/validate_formal_experiment_freeze.py", "--config", config_path, "--no-write",
            ],
        },
        {
            "command_id": "gpu_stochastic_smoke",
            "purpose": "Verify 64-sample x 100-step stochastic NeuronSifter diagnostics on gpu03/gpu1.",
            "command": [
                "srun", "--partition=gpu1", "--nodelist=gpu03", "--job-name=NrnSift_gpu_stochastic_smoke",
                "--gres=gpu:1", "--cpus-per-task=4", "--mem=16G",
                "python", "Libs/scripts/validators/validate_neuronsifter_stochastic_smoke.py", "--config", config_path,
            ],
        },
        {
            "command_id": "cpu_full_tests",
            "purpose": "Run repository CPU test suite.",
            "command": [
                "srun", "--partition=cpu", "--nodelist=cpu04", "--job-name=NrnSift_full_tests_cpu",
                "--cpus-per-task=4", "--mem=16G",
                "python", "-m", "pytest", "Libs/tests", "-q",
            ],
        },
        {
            "command_id": "formal_submission",
            "purpose": "Submit formal publication experiment suite after readiness and freeze replay gates pass.",
            "command": ["bash", "Libs/exp/submission/submit_publication_claim_suite.sh", "--config", config_path],
        },
    ]


def artifact_hashes(paths: dict[str, Path | ReadOnlyArtifactReference]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if isinstance(path, ReadOnlyArtifactReference):
            result[name] = artifact_metadata(path)
            continue
        result[name] = {
            "path": repo_relative(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "access_mode": "worktree",
            "linkage_id": None,
            "environment_variable": None,
        }
    return result


def config_hashes(config_path: str | Path) -> dict[str, object]:
    include_paths = config_include_paths(config_path)
    resolved_config = load_config(config_path)
    resolved_payload = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {
        "config_stack_path": repo_relative(artifact_path(config_path)),
        "resolved_config_sha256": sha256_bytes(resolved_payload),
        "config_file_hashes": {
            repo_relative(path): file_sha256(path)
            for path in include_paths
        },
    }


def freeze_snapshot(config_path: str | Path, excluded_paths: set[str]) -> dict[str, object]:
    config = load_config(config_path)
    source_paths = source_artifact_paths(config)
    readiness = read_json_object(source_paths["formal_experiment_readiness_audit"])
    g5 = read_json_object(source_paths["g5_effective_sample_audit"])
    qc_summary = read_json_object(source_paths["simulation_qc_summary"])
    canonical_counts = load_canonical_metric_values(config_path)
    pair_rows = read_tsv_rows(source_paths["simulation_pair_alignment_audit"])
    qc_rows = read_tsv_rows(source_paths["simulation_qc_catalog"])
    classification = classify_full_only_rows(pair_rows, qc_rows)
    training_exclusion = training_exclusion_report(config, list(classification["rows"]), qc_rows)
    pair_status_counts: dict[str, int] = {}
    for row in pair_rows:
        pair_status = str(row["paired_status"])
        pair_status_counts[pair_status] = pair_status_counts.get(pair_status, 0) + 1
    return {
        "counts_and_statuses": {
            "formal_readiness_status": str(readiness["readiness_status"]),
            "formal_experiment_launch_allowed": bool(readiness["formal_experiment_launch_allowed"]),
            "formal_blocker_count": int(readiness["blocker_count"]),
            "formal_major_count": int(readiness["major_count"]),
            "g2_status": str(readiness["g2_status"]),
            "g2_gate": str(readiness["g2_gate"]),
            "g5_status": str(readiness["g5_status"]),
            "g5_gate_status": str(g5["gate_status"]),
            "g5_materialized_trajectory_gate_status": str(g5["materialized_trajectory_gate_status"]),
            "full_microenvironment_trajectory_count": canonical_counts["raw_materialized_full_microenvironment_trajectory_count"],
            "mesoscopic_trajectory_count": canonical_counts["raw_materialized_mesoscopic_trajectory_count"],
            "paired_trajectory_count": canonical_counts["paired_full_mesoscopic_count"],
            "train_eligible_count": canonical_counts["qc_eligible_trajectory_count"],
            "reject_count": canonical_counts["rejected_trajectory_count"],
            "stress_only_count": int(qc_summary["stress_only_count"]),
            "weak_teacher_count": int(qc_summary["weak_teacher_count"]),
            "total_split_leakage_count": int(qc_summary["total_split_leakage_count"]),
            "full_only_count": int(classification["full_only_count"]),
        },
        "source_artifacts": artifact_hashes(source_paths),
        "config_hashes": config_hashes(config_path),
        "git_state": git_state(excluded_paths),
        "pair_status_counts": pair_status_counts,
        "full_only_classification": classification,
        "training_exclusion": training_exclusion,
    }


def snapshot_signature(snapshot: dict[str, object]) -> dict[str, object]:
    source_hashes = {
        name: payload["sha256"]
        for name, payload in dict(snapshot["source_artifacts"]).items()
    }
    return {
        "counts_and_statuses": snapshot["counts_and_statuses"],
        "source_artifact_hashes": source_hashes,
        "config_hashes": snapshot["config_hashes"],
        "git_head_sha": dict(snapshot["git_state"])["git_head_sha"],
        "git_full_diff_hash": dict(snapshot["git_state"])["full_diff_hash"],
        "pair_status_counts": snapshot["pair_status_counts"],
        "full_only_classification_counts": dict(snapshot["full_only_classification"])["counts"],
        "training_exclusion_status": dict(snapshot["training_exclusion"])["exclusion_status"],
    }


def validation_errors(snapshot: dict[str, object]) -> list[str]:
    counts = dict(snapshot["counts_and_statuses"])
    classification = dict(snapshot["full_only_classification"])
    training_exclusion = dict(snapshot["training_exclusion"])
    canonical_counts = load_canonical_metric_values(REPO_ROOT / "Libs/configs/config_stack.yaml")
    errors: list[str] = []
    expected_values = {
        "formal_readiness_status": "formal_experiment_ready",
        "formal_experiment_launch_allowed": True,
        "formal_blocker_count": 0,
        "formal_major_count": 0,
        "g2_status": EXPECTED_G2_STATUS,
        "g2_gate": EXPECTED_G2_GATE,
        "g5_status": EXPECTED_G5_STATUS,
        "g5_gate_status": EXPECTED_G5_GATE_STATUS,
        "g5_materialized_trajectory_gate_status": EXPECTED_G5_GATE_STATUS,
        "full_microenvironment_trajectory_count": canonical_counts["raw_materialized_full_microenvironment_trajectory_count"],
        "paired_trajectory_count": canonical_counts["paired_full_mesoscopic_count"],
        "train_eligible_count": canonical_counts["qc_eligible_trajectory_count"],
        "full_only_count": canonical_counts["rejected_trajectory_count"],
        "total_split_leakage_count": 0,
    }
    for key, expected in expected_values.items():
        if counts.get(key) != expected:
            errors.append(f"{key} expected {expected!r} but observed {counts.get(key)!r}")
    classification_counts = dict(classification["counts"])
    expected_classification_counts = {
        "reject": canonical_counts["rejected_trajectory_count"],
        "stress_only": 0,
        "calibration": 0,
        "inventory_only": 0,
        "other": 0,
    }
    if classification_counts != expected_classification_counts:
        errors.append(f"full_only classification counts expected {expected_classification_counts!r} but observed {classification_counts!r}")
    if training_exclusion["train_eligible_grade_ab_count"] != canonical_counts["qc_grade_a_b_count"]:
        errors.append("train-eligible Grade A/B records do not match canonical numeric lineage")
    if training_exclusion["exclusion_status"] != "pass":
        errors.append("training dataset loader exclusion contract failed")
    return errors


def build_freeze_manifest(config_path: str | Path, excluded_paths: set[str]) -> dict[str, object]:
    snapshot = freeze_snapshot(config_path, excluded_paths)
    errors = validation_errors(snapshot)
    return {
        "audit_name": "formal_experiment_freeze_manifest",
        "freeze_status": "passed" if not errors else "failed",
        "model_claim_change": "none",
        "readiness_scope": "G2_formal_experiment_freeze_only",
        "final_release_policy": "clean_committed_state_or_signed_release_tag_required",
        "validation_errors": errors,
        "counts_and_statuses": snapshot["counts_and_statuses"],
        "source_artifacts": snapshot["source_artifacts"],
        "config_hashes": snapshot["config_hashes"],
        "git_state": snapshot["git_state"],
        "pair_status_counts": snapshot["pair_status_counts"],
        "full_only_classification": snapshot["full_only_classification"],
        "training_exclusion": snapshot["training_exclusion"],
        "slurm_command_manifest": slurm_command_manifest(str(config_path)),
        "snapshot_signature": snapshot_signature(snapshot),
    }


def escape_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(manifest: dict[str, object]) -> str:
    counts = dict(manifest["counts_and_statuses"])
    classification = dict(manifest["full_only_classification"])
    git_payload = dict(manifest["git_state"])
    lines = [
        "# Formal Experiment Freeze Manifest",
        "",
        f"- freeze_status: `{manifest['freeze_status']}`",
        f"- readiness_scope: `{manifest['readiness_scope']}`",
        f"- model_claim_change: `{manifest['model_claim_change']}`",
        f"- reproducibility_status: `{git_payload['reproducibility_status']}`",
        f"- dirty_worktree_allowed: `{git_payload['dirty_worktree_allowed']}`",
        f"- git_head_sha: `{git_payload['git_head_sha']}`",
        f"- git_full_diff_hash: `{dict(manifest['snapshot_signature'])['git_full_diff_hash']}`",
        f"- resolved_config_sha256: `{dict(manifest['config_hashes'])['resolved_config_sha256']}`",
        f"- full_microenvironment_trajectory_count: `{counts['full_microenvironment_trajectory_count']}`",
        f"- paired_trajectory_count: `{counts['paired_trajectory_count']}`",
        f"- train_eligible_count: `{counts['train_eligible_count']}`",
        f"- g2_status: `{counts['g2_status']}`",
        f"- g5_status: `{counts['g5_status']}`",
        "",
        "## Full-Only Classification",
        "",
        "| classification | count |",
        "| --- | --- |",
    ]
    for key, value in dict(classification["counts"]).items():
        lines.append(f"| {escape_md(key)} | {escape_md(value)} |")
    lines.extend(
        [
            "",
            "## Training Exclusion",
            "",
            f"- exclusion_status: `{dict(manifest['training_exclusion'])['exclusion_status']}`",
            f"- full_only_main_pretraining_row_count: `{dict(manifest['training_exclusion'])['full_only_main_pretraining_row_count']}`",
            f"- main_pretraining_non_paired_row_count: `{dict(manifest['training_exclusion'])['main_pretraining_non_paired_row_count']}`",
            "",
            "## Source Artifact Hashes",
            "",
            "| artifact | path | sha256 |",
            "| --- | --- | --- |",
        ]
    )
    for name, payload in dict(manifest["source_artifacts"]).items():
        artifact_payload = dict(payload)
        lines.append(f"| {escape_md(name)} | {escape_md(artifact_payload['path'])} | {escape_md(artifact_payload['sha256'])} |")
    lines.extend(
        [
            "",
            "## Slurm Command Manifest",
            "",
            "| command_id | command |",
            "| --- | --- |",
        ]
    )
    for command_payload in manifest["slurm_command_manifest"]:
        command = " ".join(str(value) for value in command_payload["command"])
        lines.append(f"| {escape_md(command_payload['command_id'])} | `{escape_md(command)}` |")
    if manifest["validation_errors"]:
        lines.extend(["", "## Validation Errors", ""])
        for error in manifest["validation_errors"]:
            lines.append(f"- {error}")
    lines.append("")
    return "\n".join(lines)


def write_manifest_outputs(manifest: dict[str, object], json_output: Path, md_output: Path) -> None:
    write_json(json_output, manifest)
    write_text_atomic(md_output, markdown_report(manifest))


def main() -> None:
    args = parse_args()
    snapshot_config_path = freeze_replay_config_path(args.config) if args.no_write else args.config
    json_output = artifact_path(args.json_output)
    md_output = artifact_path(args.md_output)
    excluded_paths = {repo_relative(json_output), repo_relative(md_output)} | GENERATED_AUDIT_OUTPUTS
    pre_snapshot = freeze_snapshot(snapshot_config_path, excluded_paths)
    manifest = {
        "audit_name": "formal_experiment_freeze_manifest",
        "freeze_status": "failed",
        "model_claim_change": "none",
        "readiness_scope": "G2_formal_experiment_freeze_only",
        "final_release_policy": "clean_committed_state_or_signed_release_tag_required",
        "validation_errors": validation_errors(pre_snapshot),
        "counts_and_statuses": pre_snapshot["counts_and_statuses"],
        "source_artifacts": pre_snapshot["source_artifacts"],
        "config_hashes": pre_snapshot["config_hashes"],
        "git_state": pre_snapshot["git_state"],
        "pair_status_counts": pre_snapshot["pair_status_counts"],
        "full_only_classification": pre_snapshot["full_only_classification"],
        "training_exclusion": pre_snapshot["training_exclusion"],
        "slurm_command_manifest": slurm_command_manifest(str(snapshot_config_path)),
        "snapshot_signature": snapshot_signature(pre_snapshot),
    }
    if manifest["validation_errors"]:
        print(json.dumps(manifest, indent=2, sort_keys=False))
        raise SystemExit(1)
    post_snapshot = freeze_snapshot(snapshot_config_path, excluded_paths)
    if snapshot_signature(pre_snapshot) != snapshot_signature(post_snapshot):
        drift_manifest = dict(manifest)
        drift_manifest["freeze_status"] = "failed"
        drift_manifest["validation_errors"] = ["Freeze source snapshot changed during manifest construction."]
        print(json.dumps(drift_manifest, indent=2, sort_keys=False))
        raise SystemExit(1)
    manifest["freeze_status"] = "passed"
    if args.no_write:
        if not json_output.exists():
            replay_manifest = dict(manifest)
            replay_manifest["freeze_status"] = "failed"
            replay_manifest["validation_errors"] = [f"Frozen manifest is missing for replay: {repo_relative(json_output)}"]
            print(json.dumps(replay_manifest, indent=2, sort_keys=False))
            raise SystemExit(1)
        frozen_manifest = read_json_object(json_output)
        frozen_signature = frozen_manifest.get("snapshot_signature")
        current_signature = manifest["snapshot_signature"]
        if frozen_signature != current_signature:
            replay_manifest = dict(manifest)
            replay_manifest["freeze_status"] = "failed"
            replay_manifest["validation_errors"] = ["Freeze replay signature differs from frozen manifest."]
            replay_manifest["frozen_snapshot_signature"] = frozen_signature
            replay_manifest["current_snapshot_signature"] = current_signature
            print(json.dumps(replay_manifest, indent=2, sort_keys=False))
            raise SystemExit(1)
    else:
        write_manifest_outputs(manifest, json_output, md_output)
        final_snapshot = freeze_snapshot(snapshot_config_path, excluded_paths)
        if snapshot_signature(pre_snapshot) != snapshot_signature(final_snapshot):
            print("Freeze source snapshot changed while writing manifest outputs.", file=sys.stderr)
            raise SystemExit(1)
    print(json.dumps(manifest, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
