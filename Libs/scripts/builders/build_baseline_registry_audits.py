from __future__ import annotations

import argparse
import hashlib
import json
import configparser
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json, write_tsv  # noqa: E402


DEFAULT_REGISTRY = "Libs/configs/experiments/baseline_registry.yaml"
DEFAULT_SCHEMA = "Libs/configs/schema/baseline_registry.schema.json"
DEFAULT_LOCK = "Libs/external/BASELINES.lock.yaml"
AUDIT_ROOT = "Output/audits"
GENERATED_AUDIT_NAMES = {
    "baseline_existing_inventory.json",
    "baseline_duplicate_implementation_audit.json",
    "baseline_source_audit.tsv",
    "baseline_clone_manifest.tsv",
    "baseline_license_audit.tsv",
    "baseline_native_reproduction_status.tsv",
    "baseline_adapter_compatibility.tsv",
    "baseline_fairness_matrix.tsv",
    "baseline_reproduction_gate.json",
    "baseline_v31_active_set.tsv",
    "baseline_v31_replacement_groups.json",
    "baseline_v31_claim_coverage.json",
    "baseline_v31_resource_reduction.json",
    "baseline_v31_gate.json",
}
LICENSE_APPROVAL = {
    "MIT": "approved_open_source",
    "Apache-2.0": "approved_open_source",
    "BSD-3-Clause": "approved_open_source",
    "CC-BY-NC-SA-4.0": "approved_noncommercial_research_only",
    "project_license": "project_owned",
}
HEADLINE_CONDITIONS = (
    "official_source_verified",
    "license_approved",
    "commit_pinned",
    "upstream_smoke_passed",
    "native_reproduction_passed_or_justified",
    "adapter_passed",
    "matched_protocol_smoke_passed",
    "fairness_audit_passed",
    "task_applicability_passed",
    "no_test_set_tuning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed Baseline Registry V3 audit artifacts.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--lock", dest="lock_path", default=DEFAULT_LOCK)
    parser.add_argument("--audit-root", default=AUDIT_ROOT)
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    completed = subprocess.run(args, cwd=cwd or REPO_ROOT, text=True, capture_output=True, check=False)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_value(path: Path, *args: str) -> str:
    code, stdout, _ = command(["git", "-C", str(path), *args])
    return stdout if code == 0 else ""


def canonical_remote(value: str) -> str:
    remote = value.strip().removesuffix(".git")
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.split(":", 1)[1]
    return remote.rstrip("/").lower()


def load_inputs(registry_path: Path, schema_path: Path, lock_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(registry), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:10])
        raise ValueError(f"Baseline registry schema validation failed: {detail}")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    return registry, lock


def git_metadata(checkout: Path, check_clean: bool) -> dict[str, Any]:
    git_dir = checkout / ".git"
    config = configparser.ConfigParser()
    config.read(git_dir / "config", encoding="utf-8")
    remote = config.get('remote "origin"', "url", fallback="")
    head_ref = (git_dir / "HEAD").read_text(encoding="utf-8").strip() if (git_dir / "HEAD").exists() else ""
    if head_ref.startswith("ref: "):
        ref_name = head_ref[6:]
        ref_path = git_dir / ref_name
        commit = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else git_value(checkout, "rev-parse", "HEAD")
        branch = ref_name.removeprefix("refs/heads/")
    else:
        commit = head_ref
        branch = ""
    tracked_diff = git_value(checkout, "diff", "--name-only") if check_clean else ""
    untracked = git_value(
        checkout,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "*.py",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.sh",
        "*.toml",
        "*.txt",
    ) if check_clean else ""
    dirty_paths = [value for value in [*tracked_diff.splitlines(), *untracked.splitlines()] if value]
    return {
        "path": relative(checkout),
        "remote_url": remote,
        "canonical_remote": canonical_remote(remote),
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def discover_git_checkouts(checkout_paths: list[str], registry_paths: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(checkout_paths):
        checkout = repo_path(value)
        if not (checkout / ".git").is_dir():
            raise ValueError(f"Frozen inventory checkout has no .git directory: {value}")
        rows.append(git_metadata(checkout, value in registry_paths))
    return rows


def inventory_report(registry: dict[str, Any], git_rows: list[dict[str, Any]]) -> dict[str, Any]:
    roots = [value for value in ("Libs/exp/baselines", "Libs/models/baselines", "Libs/configs", "Output/audits", "Output/results") if repo_path(value).exists()]
    # Bind the audit to Git's tracked/untracked view. Recursive rg discovery can
    # transiently omit visible untracked files on the shared NFS checkout.
    code, stdout, stderr = command(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *roots,
        ]
    )
    if code not in {0, 1}:
        raise RuntimeError(f"Baseline file discovery failed: {stderr}")
    baseline_files = sorted(
        value for value in stdout.splitlines()
        if "__pycache__" not in Path(value).parts
        and Path(value).name not in GENERATED_AUDIT_NAMES
        and any(token in Path(value).name.lower() for token in ("baseline", "ablation"))
    )
    external_source_copies = []
    for path in sorted((REPO_ROOT / "Libs/external").iterdir()):
        external_source_copies.append({"path": relative(path), "kind": "directory" if path.is_dir() else "file"})
    return {
        "schema_version": "baseline_existing_inventory_v3",
        "registry_id": registry["registry_id"],
        "baseline_entry_count": len(registry["baselines"]),
        "baseline_files": baseline_files,
        "git_checkout_count": len(git_rows),
        "git_checkouts": git_rows,
        "external_top_level_inventory": external_source_copies,
        "legacy_result_warning": "Legacy runnable/readiness labels are inventory evidence only and do not satisfy V3 reproduction gates.",
    }


def duplicate_report(registry: dict[str, Any], git_rows: list[dict[str, Any]]) -> dict[str, Any]:
    duplicates: list[dict[str, Any]] = []
    by_remote: dict[str, list[str]] = defaultdict(list)
    for row in git_rows:
        if row["canonical_remote"]:
            by_remote[str(row["canonical_remote"])].append(str(row["path"]))
    for remote, paths in sorted(by_remote.items()):
        if len(paths) > 1:
            duplicates.append({"kind": "duplicate_remote_checkout", "identity": remote, "paths": paths, "disposition": "retain_as_inventory; do_not_clone_again"})
    for row in registry["baselines"]:
        local = str(row["official_repository"]["local_path"] or "")
        if local and local.startswith("Libs/external/") and repo_path(local).is_dir() and not (repo_path(local) / ".git").is_dir():
            duplicates.append({"kind": "unversioned_source_copy", "identity": row["baseline_id"], "paths": [local], "disposition": "not_a_pinned_experiment_identity"})
    legacy_diffusion = REPO_ROOT / "Libs/external/neural_operator/DiffusionPDE"
    pinned_diffusion = REPO_ROOT / "Libs/external/diffusionpde"
    if legacy_diffusion.exists() and pinned_diffusion.exists():
        duplicates.append({"kind": "duplicate_implementation_identity", "identity": "diffusionpde", "paths": [relative(legacy_diffusion), relative(pinned_diffusion)], "disposition": "pinned_checkout_is_authoritative; legacy_copy_is_inventory_only"})
    return {
        "schema_version": "baseline_duplicate_implementation_audit_v3",
        "status": "blocked_duplicates_or_unversioned_copies" if duplicates else "passed",
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }


def source_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in registry["baselines"]:
        source = entry["official_repository"]
        rows.append({
            "baseline_id": entry["baseline_id"],
            "baseline_class": entry["baseline_class"],
            "official_repository": source["url"],
            "official_paper": entry["official_paper"]["url"],
            "verification_status": source["verification_status"],
            "verification_basis": source["verification_basis"],
            "accepted_for_clone": str(source["verification_status"] == "verified" and entry["license"] != "unknown").lower(),
            "source_blocker": "" if source["verification_status"] in {"verified", "not_applicable"} else source["verification_status"],
        })
    return rows


def clone_rows(registry: dict[str, Any], lock: dict[str, Any], metadata_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    lock_entries = lock.get("entries", {})
    for entry in registry["baselines"]:
        source = entry["official_repository"]
        local_path = str(source["local_path"] or "")
        if not local_path.startswith("Libs/external/"):
            continue
        path = repo_path(local_path)
        is_git = (path / ".git").is_dir()
        metadata = metadata_by_path.get(local_path, {})
        actual_remote = str(metadata.get("remote_url", "")) if is_git else ""
        actual_commit = str(metadata.get("commit", "")) if is_git else ""
        default_ref = metadata.get("branch", "") if is_git else ""
        status = "\n".join(str(value) for value in metadata.get("dirty_paths", [])) if is_git else ""
        lock_row = dict(lock_entries.get(entry["baseline_id"], {}))
        rows.append({
            "baseline_id": entry["baseline_id"],
            "local_path": local_path,
            "path_exists": str(path.exists()).lower(),
            "git_identity": str(is_git).lower(),
            "remote_url": actual_remote,
            "remote_matches_registry": str(canonical_remote(actual_remote) == canonical_remote(str(source["url"]))).lower(),
            "default_branch": str(default_ref).removeprefix("origin/"),
            "commit": actual_commit,
            "commit_matches_registry": str(bool(actual_commit) and actual_commit == entry["commit"]).lower(),
            "tag": str(lock_row.get("tag", "")),
            "submodule_state": "present_unexpanded" if (path / ".gitmodules").exists() else "none",
            "git_lfs_state": "configured" if (path / ".gitattributes").exists() and "filter=lfs" in (path / ".gitattributes").read_text(encoding="utf-8", errors="replace") else "not_configured",
            "checkout_clean": str(is_git and not metadata.get("dirty", False)).lower(),
            "lock_entry_present": str(bool(lock_row)).lower(),
            "license_hash": str(lock_row.get("license_hash", "")),
            "readme_hash": str(lock_row.get("readme_hash", "")),
            "environment_hash": str(lock_row.get("environment_hash", "")),
            "pin_status": "pinned_clean" if is_git and actual_commit == entry["commit"] and not status else ("pinned_dirty" if is_git and actual_commit == entry["commit"] else "not_pinned"),
        })
    return rows


def license_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in registry["baselines"]:
        local_path = str(entry["official_repository"]["local_path"] or "")
        path = repo_path(local_path) if local_path else None
        license_files = [] if path is None or not path.exists() else sorted([*path.glob("LICENSE*"), *path.glob("COPYING*")])
        actual_hashes = [sha256_file(value) for value in license_files if value.is_file()]
        declared_hash = str(entry["license_snapshot_hash"] or "")
        approval = LICENSE_APPROVAL.get(str(entry["license"]), "blocked_unclear")
        hash_matches = (not declared_hash and entry["license"] == "project_license") or declared_hash in actual_hashes
        rows.append({
            "baseline_id": entry["baseline_id"],
            "license": entry["license"],
            "approval_status": approval,
            "restriction": "noncommercial_research_only" if entry["license"] == "CC-BY-NC-SA-4.0" else "",
            "license_files": "|".join(relative(value) for value in license_files),
            "declared_license_hash": declared_hash,
            "observed_license_hashes": "|".join(actual_hashes),
            "hash_matches": str(hash_matches).lower(),
            "license_gate_passed": str(approval != "blocked_unclear" and hash_matches).lower(),
        })
    return rows


def native_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in registry["baselines"]:
        rows.append({
            "baseline_id": entry["baseline_id"],
            "status": entry["upstream_status"],
            "command": entry["native_command"],
            "config": "declared_by_upstream" if entry["native_command"] else "",
            "seed": "not_run",
            "dataset_identity": "blocked_or_not_run",
            "checkpoint_identity": entry["weights"]["identity"],
            "runtime_seconds": "not_run",
            "hardware": "not_run",
            "reported_metric": "",
            "expected_metric": "|".join(entry["native_expected_metrics"]),
            "tolerance": "not_established",
            "stdout_path": "",
            "stderr_path": "",
            "paper_score_used_as_project_result": "false",
            "note": "Import success is not counted as reproduction; no formal training or native reproduction was authorized.",
        })
    return rows


def adapter_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in registry["baselines"]:
        adapter = repo_path(entry["adapter_path"]) if entry["adapter_path"] else None
        applicability = entry["applicability"]
        concrete_adapter = bool(adapter and adapter.exists()) and adapter.name != "base.py"
        rows.append({
            "baseline_id": entry["baseline_id"],
            "task_families": "|".join(entry["task_families"]),
            "applicability": applicability["status"],
            "compatibility_reason": applicability["reason"],
            "adapter_path": entry["adapter_path"],
            "concrete_adapter_present": str(concrete_adapter).lower(),
            "information_available": "|".join(entry["input_semantics"]["available"]),
            "information_unavailable_by_design": "|".join(entry["input_semantics"]["unavailable_by_design"]),
            "axis_conversion": entry["input_semantics"]["axis_conversion"],
            "feature_mapping": entry["input_semantics"]["feature_mapping"],
            "mask_handling": entry["input_semantics"]["mask_handling"],
            "action_representation": entry["input_semantics"]["action_representation"],
            "uncertainty_support": entry["output_semantics"]["uncertainty_support"],
            "semantic_loss": entry["output_semantics"]["semantic_loss"],
            "adapter_status": "not_applicable" if applicability["status"] == "not_applicable" else ("passed" if concrete_adapter else "not_implemented"),
        })
    return rows


def fairness_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in registry["baselines"]:
        tuning = entry["matched_tuning_budget"]
        training = entry["matched_training_budget"]
        inference = entry["matched_inference_budget"]
        rows.append({
            "baseline_id": entry["baseline_id"],
            "baseline_class": entry["baseline_class"],
            "comparison_cohort": "oracle_upper_bound_separate" if entry["baseline_class"] == "oracle_upper_bound" else entry["baseline_class"],
            "native_budget": "upstream_native_contract_not_run",
            "matched_data": "required_not_run",
            "matched_split": "required_not_run",
            "matched_preprocessing": "required_not_run",
            "matched_input_information": "required_not_run",
            "validation_only_tuning": str(not tuning["test_tuning"]).lower(),
            "validation_run_budget": tuning["validation_runs"],
            "matched_seed_set": "|".join(map(str, training["seed_set"])),
            "matched_max_epochs": training["max_epochs"],
            "matched_gpu_hours": training["gpu_hours"],
            "matched_inference_samples": inference["samples"],
            "matched_denoising_steps": inference["denoising_steps"],
            "failed_runs_reported": "required_zero_recorded",
            "checkpoint_selection": "validation_only_required_not_run",
            "metric_implementation": "shared_required_not_run",
            "fairness_status": "not_run",
        })
    return rows


def gate_report(registry: dict[str, Any], clone_manifest: list[dict[str, Any]], licenses: list[dict[str, Any]], adapters: list[dict[str, Any]]) -> dict[str, Any]:
    clone_by_id = {row["baseline_id"]: row for row in clone_manifest}
    license_by_id = {row["baseline_id"]: row for row in licenses}
    adapter_by_id = {row["baseline_id"]: row for row in adapters}
    rows = []
    for entry in registry["baselines"]:
        source_status = entry["official_repository"]["verification_status"]
        clone = clone_by_id.get(entry["baseline_id"], {})
        conditions = {
            "official_source_verified": source_status in {"verified", "not_applicable"},
            "license_approved": license_by_id[entry["baseline_id"]]["license_gate_passed"] == "true",
            "commit_pinned": entry["commit"] == "project_worktree" or clone.get("pin_status") == "pinned_clean",
            "upstream_smoke_passed": entry["upstream_status"] in {"upstream_smoke_passed", "native_reproduction_passed"} or source_status == "not_applicable",
            "native_reproduction_passed_or_justified": entry["upstream_status"] == "native_reproduction_passed" or source_status == "not_applicable",
            "adapter_passed": adapter_by_id[entry["baseline_id"]]["adapter_status"] in {"passed", "not_applicable"},
            "matched_protocol_smoke_passed": False,
            "fairness_audit_passed": False,
            "task_applicability_passed": entry["applicability"]["status"] == "applicable",
            "no_test_set_tuning": entry["matched_tuning_budget"]["test_tuning"] is False,
        }
        rows.append({"baseline_id": entry["baseline_id"], "baseline_class": entry["baseline_class"], "headline_ready": all(conditions[name] for name in HEADLINE_CONDITIONS), "conditions": conditions, "blockers": entry["blockers"]})
    unresolved_source_or_license = [row["baseline_id"] for row in rows if not row["conditions"]["official_source_verified"] or not row["conditions"]["license_approved"]]
    status = "blocked_source_or_license" if unresolved_source_or_license else "blocked_round11_full_suite"
    return {
        "schema_version": "baseline_reproduction_gate_v3",
        "registry_id": registry["registry_id"],
        "entry_gate": status,
        "round11_ceiling": "upstream_smoke_complete",
        "round11_exit_gate": "blocked_full_suite",
        "formal_matched_training_allowed": False,
        "headline_comparison_allowed": False,
        "evidence_fabric_sealing_allowed": False,
        "DT_CORE_allowed": False,
        "unresolved_source_or_license": unresolved_source_or_license,
        "headline_ready_count": sum(bool(row["headline_ready"]) for row in rows),
        "baseline_count": len(rows),
        "baselines": rows,
        "gate_blockers": ["source_or_license_blockers_present"] if unresolved_source_or_license else ["round11_full_suite_blocked"],
    }


def build_all(registry_path: Path, schema_path: Path, lock_path: Path) -> dict[str, Any]:
    registry, lock = load_inputs(registry_path, schema_path, lock_path)
    registry_paths = {str(row["official_repository"]["local_path"]) for row in registry["baselines"] if str(row["official_repository"]["local_path"] or "").startswith("Libs/external/")}
    git_rows = discover_git_checkouts(list(registry["inventory_checkout_paths"]), registry_paths)
    metadata_by_path = {str(row["path"]): row for row in git_rows}
    inventory = inventory_report(registry, git_rows)
    duplicates = duplicate_report(registry, git_rows)
    sources = source_rows(registry)
    clones = clone_rows(registry, lock, metadata_by_path)
    licenses = license_rows(registry)
    native = native_rows(registry)
    adapters = adapter_rows(registry)
    fairness = fairness_rows(registry)
    gate = gate_report(registry, clones, licenses, adapters)
    return {"inventory": inventory, "duplicates": duplicates, "sources": sources, "clones": clones, "licenses": licenses, "native": native, "adapters": adapters, "fairness": fairness, "gate": gate}


def write_all(payload: dict[str, Any], audit_root: Path) -> None:
    write_json(audit_root / "baseline_existing_inventory.json", payload["inventory"])
    write_json(audit_root / "baseline_duplicate_implementation_audit.json", payload["duplicates"])
    write_tsv(audit_root / "baseline_source_audit.tsv", list(payload["sources"][0]), payload["sources"])
    write_tsv(audit_root / "baseline_clone_manifest.tsv", list(payload["clones"][0]), payload["clones"])
    write_tsv(audit_root / "baseline_license_audit.tsv", list(payload["licenses"][0]), payload["licenses"])
    write_tsv(audit_root / "baseline_native_reproduction_status.tsv", list(payload["native"][0]), payload["native"])
    write_tsv(audit_root / "baseline_adapter_compatibility.tsv", list(payload["adapters"][0]), payload["adapters"])
    write_tsv(audit_root / "baseline_fairness_matrix.tsv", list(payload["fairness"][0]), payload["fairness"])
    write_json(audit_root / "baseline_reproduction_gate.json", payload["gate"])


def main() -> None:
    args = parse_args()
    payload = build_all(repo_path(args.registry), repo_path(args.schema), repo_path(args.lock_path))
    write_all(payload, repo_path(args.audit_root))
    print(json.dumps({"entry_gate": payload["gate"]["entry_gate"], "headline_ready_count": payload["gate"]["headline_ready_count"], "baseline_count": payload["gate"]["baseline_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
