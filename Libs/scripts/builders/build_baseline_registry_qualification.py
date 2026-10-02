from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_registry_active_set_audits import (  # noqa: E402
    MAIN_ABLATIONS,
    MAIN_BASELINES,
)
from Libs.scripts.builders.build_baseline_registry_audits import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    load_inputs,
    repo_path,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    write_json,
    write_text_atomic,
    write_tsv,
)


AUDIT_FILES = {
    "active_external_set": "baseline_registry_qualification_active_external_set.tsv",
    "source_resolution": "baseline_registry_qualification_source_resolution.tsv",
    "license_resolution": "baseline_registry_qualification_license_resolution.tsv",
    "replacement_decision": "baseline_registry_qualification_replacement_decision.json",
    "clone_manifest": "baseline_registry_qualification_clone_manifest.tsv",
    "upstream_smoke_status": "baseline_registry_qualification_upstream_smoke_status.tsv",
    "gate": "baseline_registry_qualification_gate.json",
}

INTERNAL_ACTIVE_IDS = {
    "persistence_or_no_action",
    "linear_delta_or_state_space",
    "gru",
    "neuronsifter_deterministic",
    "neuronsifter_drug_id_conditioning",
    "neuronsifter_scalar_exposure",
    "neuronsifter_without_constraints_and_reliability",
    "neuronsifter_without_assimilation_and_what_if_pairing",
}
EXTERNAL_ACTIVE_IDS = {
    "chemcpa",
    "cellot",
    "fno",
    "transolver_plus",
    "wdno",
    "deep_ensemble",
    "conformal_calibration",
}
QUALIFICATION_STATUSES = {
    "approved",
    "approved_internal_only",
    "approved_code_weights_restricted",
    "blocked_source",
    "blocked_license",
    "blocked_weights",
    "blocked_data_terms",
}

# This is a dated, read-only official-source evidence snapshot. It intentionally
# does not promote source code alone to permission for weights or data assets.
EXTERNAL_EVIDENCE: dict[str, dict[str, str]] = {
    "chemcpa": {
        "official_repository": "https://github.com/theislab/chemCPA",
        "official_repository_status": "verified",
        "paper_or_project_link": "https://neurips.cc/virtual/2022/poster/53227",
        "paper_linkage_status": "verified_readme_and_repository_metadata",
        "source_evidence": "Official repository metadata names the NeurIPS 2022 chemCPA paper; README links that paper.",
        "default_branch": "main",
        "remote_observed_commit": "43e830eb0958c54e4aa64442c17ec0fed19b3f15",
        "code_license": "MIT",
        "code_license_hash": "9372ee1c8aa4ac28657f7e7224b9684aa4c3c52922dd6bc70114f4d42349ed87",
        "readme_hash": "mainline_convergencef0c674cee33042d9a842d589e4639e54e2e347f42b74e20597569dd73944a",
        "model_weight_license": "unverified",
        "model_weight_terms": "README links Backblaze checkpoint archive but records no weight license or redistribution term.",
        "training_data_terms": "unverified",
        "training_data_evidence": "README links raw and processed data and notes DrugBank registration; no complete training-data license bundle is recorded.",
        "evaluation_data_terms": "unverified",
        "evaluation_data_evidence": "No evaluation-data terms are frozen separately from the linked datasets.",
        "qualification_status": "blocked_data_terms",
        "blockers": "training_data_terms_unverified;evaluation_data_terms_unverified;weight_license_unverified;anonymous_package_not_approved",
    },
    "cellot": {
        "official_repository": "https://github.com/bunnech/cellot",
        "official_repository_status": "verified",
        "paper_or_project_link": "https://www.nature.com/articles/s41592-023-01969-x",
        "paper_linkage_status": "verified_readme",
        "source_evidence": "Official repository README links the CellOT Nature Methods paper and documented training/evaluation scripts.",
        "default_branch": "main",
        "remote_observed_commit": "522d2b953da8ad244fcf36f64521487fdf763788",
        "code_license": "BSD-3-Clause",
        "code_license_hash": "ea0dff735cfa39c5f27aede63af23438f2eaf90b71edb439b089d4faca6f0a86",
        "readme_hash": "b05da95483cb73cc965b2d3ee793d7d9de678cac8e1a086fa0c3fa6cea1012c4",
        "model_weight_license": "not_provided_by_documented_train_from_scratch_path",
        "model_weight_terms": "README documents training from scratch rather than a separately licensed model-weight release.",
        "training_data_terms": "unverified",
        "training_data_evidence": "README links an ETH Research Collection download without a frozen license or redistribution record.",
        "evaluation_data_terms": "unverified",
        "evaluation_data_evidence": "README documents evaluation scripts but no separate evaluation-data terms bundle.",
        "qualification_status": "blocked_data_terms",
        "blockers": "training_data_terms_unverified;evaluation_data_terms_unverified;anonymous_package_not_approved",
    },
    "fno": {
        "official_repository": "https://github.com/neuraloperator/neuraloperator",
        "official_repository_status": "candidate_not_profile_bound",
        "paper_or_project_link": "https://arxiv.org/abs/2010.08895",
        "paper_linkage_status": "candidate_package_not_bound_in_v31_profile",
        "source_evidence": "The neuraloperator package is a candidate MIT source, but the V3.1 FNO profile has no official repository or pin binding.",
        "default_branch": "main",
        "remote_observed_commit": "86a8bc7812a31b42c4f7895693cf4ac11521c066",
        "code_license": "MIT",
        "code_license_hash": "5f335b0603acfb1d180baseline_first117492e2076562075cc9b6353a3b94d5d5bc03ef33",
        "readme_hash": "not_observed_source_not_qualified",
        "model_weight_license": "not_qualified",
        "model_weight_terms": "No official FNO profile source, checkpoint identity, or weight license is frozen.",
        "training_data_terms": "not_qualified",
        "training_data_evidence": "No profile-bound official FNO training dataset is frozen.",
        "evaluation_data_terms": "not_qualified",
        "evaluation_data_evidence": "No profile-bound official FNO evaluation dataset is frozen.",
        "qualification_status": "blocked_source",
        "blockers": "official_profile_source_not_bound;commit_not_pinned;weights_not_qualified;data_terms_not_qualified",
    },
    "transolver_plus": {
        "official_repository": "https://github.com/thuml/Transolver_plus",
        "official_repository_status": "verified",
        "paper_or_project_link": "https://arxiv.org/abs/2502.02414",
        "paper_linkage_status": "verified_readme",
        "source_evidence": "Preexisting repository README identifies Transolver++ and links the paper; remote is thuml/Transolver_plus.",
        "default_branch": "main",
        "remote_observed_commit": "d5a23bc734a0ebac56384cf72049a26af9673452",
        "code_license": "MIT",
        "code_license_hash": "996b7ed7b544265d5759c96e1918588129d7a1c50024a80afc111c2324755e8c",
        "readme_hash": "bbff762488f0e1574fd7fcd5abab62567ff0da44a7d6bda797afbe1a076f512f",
        "model_weight_license": "not_declared",
        "model_weight_terms": "README documents data download but no separately licensed checkpoint release.",
        "training_data_terms": "unverified",
        "training_data_evidence": "README provides a Google Drive Aircraft dataset link without frozen terms or redistribution permission.",
        "evaluation_data_terms": "unverified",
        "evaluation_data_evidence": "The same external dataset link lacks separately recorded evaluation terms.",
        "qualification_status": "blocked_data_terms",
        "blockers": "training_data_terms_unverified;evaluation_data_terms_unverified;anonymous_package_not_approved;preexisting_checkout_dirty",
    },
    "wdno": {
        "official_repository": "https://github.com/AI4Science-WestlakeU/wdno",
        "official_repository_status": "verified",
        "paper_or_project_link": "https://openreview.net/forum?id=FQhDIGuaJ4",
        "paper_linkage_status": "verified_readme_and_repository_metadata",
        "source_evidence": "README calls the repository official and links the WDNO paper; repository metadata identifies the ICLR 2025 method.",
        "default_branch": "main",
        "remote_observed_commit": "2d5c0ffbe11797e669efe0a34bd22f087eae51d7",
        "code_license": "MIT",
        "code_license_hash": "01ea6d664b089be8686ed2d420b2d9efe3fa13ba9a25ca3e619fa79ff45e278f",
        "readme_hash": "e6b7f5981f9d1db9bf84569e9fc66a8cd861656fa1dccb432e2ab348c45c162e",
        "model_weight_license": "unverified",
        "model_weight_terms": "README links Google Drive checkpoints but records no license, checksum, or redistribution term.",
        "training_data_terms": "unverified",
        "training_data_evidence": "README links Google Drive datasets without frozen terms or redistribution permission.",
        "evaluation_data_terms": "unverified",
        "evaluation_data_evidence": "No separately licensed evaluation dataset bundle is recorded.",
        "qualification_status": "blocked_weights",
        "blockers": "checkpoint_license_unverified;checkpoint_checksum_unqualified;training_data_terms_unverified;evaluation_data_terms_unverified;anonymous_package_not_approved",
    },
    "deep_ensemble": {
        "official_repository": "",
        "official_repository_status": "not_declared",
        "paper_or_project_link": "https://arxiv.org/abs/1612.01474",
        "paper_linkage_status": "method_reference_only",
        "source_evidence": "The V3.1 active profile names a method family but does not bind an official source-code repository or executable implementation.",
        "default_branch": "not_observed",
        "remote_observed_commit": "not_observed",
        "code_license": "not_qualified",
        "code_license_hash": "not_observed",
        "readme_hash": "not_observed",
        "model_weight_license": "not_qualified",
        "model_weight_terms": "No official implementation or weight release is bound to the profile.",
        "training_data_terms": "not_qualified",
        "training_data_evidence": "No profile-bound official training dataset is frozen.",
        "evaluation_data_terms": "not_qualified",
        "evaluation_data_evidence": "No profile-bound official evaluation dataset is frozen.",
        "qualification_status": "blocked_source",
        "blockers": "official_profile_source_not_bound;license_not_qualified;weights_not_qualified;data_terms_not_qualified",
    },
    "conformal_calibration": {
        "official_repository": "",
        "official_repository_status": "not_declared",
        "paper_or_project_link": "",
        "paper_linkage_status": "method_reference_not_bound",
        "source_evidence": "The V3.1 active profile names a calibration method family but binds no official source-code repository or executable implementation.",
        "default_branch": "not_observed",
        "remote_observed_commit": "not_observed",
        "code_license": "not_qualified",
        "code_license_hash": "not_observed",
        "readme_hash": "not_observed",
        "model_weight_license": "not_qualified",
        "model_weight_terms": "No official implementation or weight release is bound to the profile.",
        "training_data_terms": "not_qualified",
        "training_data_evidence": "No profile-bound official calibration dataset is frozen.",
        "evaluation_data_terms": "not_qualified",
        "evaluation_data_evidence": "No profile-bound official evaluation dataset is frozen.",
        "qualification_status": "blocked_source",
        "blockers": "official_profile_source_not_bound;license_not_qualified;weights_not_qualified;data_terms_not_qualified",
    },
}

REPLACEMENT_EVIDENCE = {
    "cpa": {
        "official_repository": "https://github.com/theislab/cpa",
        "official_repository_status": "verified_repository_metadata",
        "code_license": "BSD-3-Clause",
        "code_license_hash": "9eaad59a80a87d0a3b2029b10584a1e3e3d0aa83a8b3dfb71acf43b3a2918ef3",
        "qualification_status": "blocked_data_terms",
        "reason": "Official README links Google Drive data and pretrained models without frozen data, weight, redistribution, or anonymous-package terms.",
    },
    "scgen": {
        "official_repository": "https://github.com/theislab/scgen",
        "official_repository_status": "verified_repository_metadata",
        "code_license": "GPL-3.0",
        "code_license_hash": "not_observed",
        "qualification_status": "blocked_data_terms",
        "reason": "Repository metadata identifies source and code license, but no frozen model/data terms or anonymous-package permission are established.",
    },
    "conditional_monge_gap": {
        "official_repository": "",
        "official_repository_status": "not_declared",
        "code_license": "not_qualified",
        "code_license_hash": "not_observed",
        "qualification_status": "blocked_source",
        "reason": "No official repository, paper linkage, or license is bound in the V3.1 replacement group.",
    },
}

ENVIRONMENT_SPECS: dict[str, dict[str, Any]] = {
    "chemcpa": {
        "name": "nrnsift-baseline-chemcpa",
        "qualification_scope": "upstream_environment_only",
        "python": "not_qualified",
        "framework": {"name": "not_qualified", "version": "not_qualified"},
        "cuda": "not_qualified",
        "compiler": "not_qualified",
        "dependencies": [],
        "container_digest": "not_pinned",
        "installation_command": "not_run_source_and_license_gate_blocked",
        "environment_hash": "not_computed_uninstalled",
        "status": "blocked_environment",
        "blockers": ["blocked_data_terms", "model_weight_license_unverified"],
    },
    "cellot": {
        "name": "nrnsift-baseline-cellot",
        "qualification_scope": "upstream_environment_only",
        "python": "not_qualified",
        "framework": {"name": "not_qualified", "version": "not_qualified"},
        "cuda": "not_qualified",
        "compiler": "not_qualified",
        "dependencies": [],
        "external_binaries": [],
        "container_digest": "not_pinned",
        "installation_command": "not_run_source_and_license_gate_blocked",
        "environment_hash": "not_computed_uninstalled",
        "status": "blocked_environment",
        "blockers": ["blocked_data_terms"],
    },
    "fno": {
        "name": "nrnsift-baseline-fno",
        "qualification_scope": "upstream_environment_only",
        "python": "not_qualified",
        "framework": {"name": "not_qualified", "version": "not_qualified"},
        "cuda": "not_qualified",
        "compiler": "not_qualified",
        "dependencies": [],
        "container_digest": "not_pinned",
        "installation_command": "not_run_official_profile_source_not_bound",
        "environment_hash": "not_computed_uninstalled",
        "status": "blocked_environment",
        "blockers": ["blocked_source"],
    },
    "transolver_plus": {
        "name": "nrnsift-baseline-transolver-plus",
        "qualification_scope": "upstream_environment_only",
        "python": "not_qualified",
        "framework": {"name": "not_qualified", "version": "not_qualified"},
        "cuda": "not_qualified",
        "compiler": "not_qualified",
        "dependencies": [],
        "container_digest": "not_pinned",
        "installation_command": "not_run_data_terms_gate_blocked",
        "environment_hash": "not_computed_uninstalled",
        "status": "blocked_environment",
        "blockers": ["blocked_data_terms", "preexisting_checkout_dirty"],
    },
    "wdno": {
        "name": "nrnsift-baseline-wdno",
        "qualification_scope": "upstream_environment_only",
        "python": "3.8",
        "framework": {"name": "pytorch", "version": "1.13.1"},
        "cuda": "11.7",
        "compiler": "system",
        "dependencies": ["phiflow", "requirements.txt_not_installed"],
        "external_binaries": ["phiflow"],
        "container_digest": "not_pinned",
        "source_environment": "Libs/external/neural_operator/wdno/requirements.txt",
        "installation_command": "not_run_checkpoint_and_data_terms_gate_blocked",
        "environment_hash": "7ae2a32bcae01bb7cf0439c3ababdb22e479f64c278769fa2c994b3bb32693b7",
        "status": "blocked_environment",
        "blockers": ["blocked_weights", "blocked_data_terms"],
    },
    "deep_ensemble": {
        "name": "nrnsift-baseline-deep-ensemble",
        "qualification_scope": "upstream_environment_only",
        "python": "not_qualified",
        "framework": {"name": "not_qualified", "version": "not_qualified"},
        "cuda": "not_qualified",
        "compiler": "not_qualified",
        "dependencies": [],
        "container_digest": "not_pinned",
        "installation_command": "not_run_official_profile_source_not_bound",
        "environment_hash": "not_computed_uninstalled",
        "status": "blocked_environment",
        "blockers": ["blocked_source"],
    },
    "conformal_calibration": {
        "name": "nrnsift-baseline-conformal-calibration",
        "qualification_scope": "upstream_environment_only",
        "python": "not_qualified",
        "framework": {"name": "not_qualified", "version": "not_qualified"},
        "cuda": "not_qualified",
        "compiler": "not_qualified",
        "dependencies": [],
        "container_digest": "not_pinned",
        "installation_command": "not_run_official_profile_source_not_bound",
        "environment_hash": "not_computed_uninstalled",
        "status": "blocked_environment",
        "blockers": ["blocked_source"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed baseline active-set qualification audits.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--lock", dest="lock_path", default=DEFAULT_LOCK)
    parser.add_argument("--audit-root", default="Output/audits")
    parser.add_argument("--environment-root", default="Libs/external/environments")
    return parser.parse_args()


def _run_git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "not_observed"


def _checkout_state(baseline_id: str, evidence: dict[str, str]) -> dict[str, object]:
    canonical_path = REPO_ROOT / "Libs" / "external" / baseline_id
    preexisting_paths = {
        "transolver_plus": REPO_ROOT / "Libs" / "external" / "neural_operator" / "Transolver_plus",
        "wdno": REPO_ROOT / "Libs" / "external" / "neural_operator" / "wdno",
    }
    path = preexisting_paths.get(baseline_id)
    if path is None or not path.exists():
        return {
            "clone_target_path": str(canonical_path.relative_to(REPO_ROOT)),
            "preexisting_checkout_path": "",
            "path_exists": False,
            "remote_url": evidence["official_repository"],
            "remote_observed_commit": evidence["remote_observed_commit"],
            "local_commit": "",
            "tag": "not_observed_no_clone",
            "submodule_sha": "not_observed_no_clone",
            "git_lfs_status": "not_observed_no_clone",
            "license_hash": evidence["code_license_hash"],
            "readme_hash": evidence["readme_hash"],
            "dirty_status": "not_applicable_not_cloned",
            "dirty_paths": "",
            "clone_status": f"not_cloned_{evidence['qualification_status']}",
            "clean_clone_requirement_met": False,
            "lock_update": "not_performed_no_clean_clone",
            "remote_tip_relation": "not_applicable_not_cloned",
        }

    status = _run_git(path, "status", "--porcelain=v1")
    dirty_paths = ";".join(line for line in status.splitlines() if line)
    local_commit = _run_git(path, "rev-parse", "HEAD")
    remote_url = _run_git(path, "remote", "get-url", "origin")
    tags = _run_git(path, "tag", "--points-at", "HEAD")
    submodules = _run_git(path, "submodule", "status")
    dirty = bool(dirty_paths) and dirty_paths != "not_observed"
    remote_relation = (
        "matches_current_remote_tip"
        if local_commit == evidence["remote_observed_commit"]
        else "preexisting_commit_differs_from_current_remote_tip"
    )
    return {
        "clone_target_path": str(canonical_path.relative_to(REPO_ROOT)),
        "preexisting_checkout_path": str(path.relative_to(REPO_ROOT)),
        "path_exists": True,
        "remote_url": remote_url,
        "remote_observed_commit": evidence["remote_observed_commit"],
        "local_commit": local_commit,
        "tag": tags or "none_observed_at_local_commit",
        "submodule_sha": submodules or "none_observed",
        "git_lfs_status": "not_configured",
        "license_hash": evidence["code_license_hash"],
        "readme_hash": evidence["readme_hash"],
        "dirty_status": "dirty_preserved" if dirty else "clean_preexisting",
        "dirty_paths": dirty_paths if dirty_paths != "not_observed" else "",
        "clone_status": (
            f"preexisting_dirty_preserved_{evidence['qualification_status']}"
            if dirty
            else f"preexisting_clean_not_qualification_cloned_{evidence['qualification_status']}"
        ),
        "clean_clone_requirement_met": False,
        "lock_update": "not_performed_no_clean_qualification_clone",
        "remote_tip_relation": remote_relation,
    }


def _active_rows(registry: dict[str, Any]) -> list[dict[str, object]]:
    expected = set(MAIN_BASELINES) | set(MAIN_ABLATIONS)
    profiles = registry["active_set_profiles"]
    ids = {profile["active_set_id"] for profile in profiles}
    if ids != expected or len(profiles) != len(ids):
        raise ValueError("V3.1 active profiles are not the exact unique 10+5 set")
    if INTERNAL_ACTIVE_IDS | EXTERNAL_ACTIVE_IDS != expected:
        raise ValueError("Internal/external qualification does not cover the active set")
    if INTERNAL_ACTIVE_IDS & EXTERNAL_ACTIVE_IDS:
        raise ValueError("Internal and external active classifications overlap")

    rows = []
    for profile in profiles:
        baseline_id = profile["active_set_id"]
        external = baseline_id in EXTERNAL_ACTIVE_IDS
        evidence = EXTERNAL_EVIDENCE.get(baseline_id)
        if external and evidence is None:
            raise ValueError(f"Missing external qualification evidence for {baseline_id}")
        rows.append(
            {
                "active_set_id": baseline_id,
                "registry_baseline_id": profile["registry_baseline_id"] or "",
                "paper_role": profile["paper_role"],
                "claim_ids": ",".join(profile["claim_ids"]),
                "classification": "external_source_required" if external else "internal_no_external_source",
                "qualification_status": evidence["qualification_status"] if evidence else "approved_internal_only",
                "execution_status": "not_started_round11_blocked",
                "classification_basis": (
                    "This contract explicitly treats only persistence, linear dynamics, GRU, and five NeuronSifter ablations as internal."
                    if external
                    else "This contract identifies the active profile as a project-owned internal method."
                ),
            }
        )
    return rows


def _source_rows() -> list[dict[str, object]]:
    rows = []
    for baseline_id in MAIN_BASELINES:
        if baseline_id not in EXTERNAL_ACTIVE_IDS:
            continue
        evidence = EXTERNAL_EVIDENCE[baseline_id]
        rows.append(
            {
                "baseline_id": baseline_id,
                "official_repository": evidence["official_repository"],
                "official_repository_status": evidence["official_repository_status"],
                "paper_or_project_link": evidence["paper_or_project_link"],
                "paper_linkage_status": evidence["paper_linkage_status"],
                "default_branch": evidence["default_branch"],
                "remote_observed_commit": evidence["remote_observed_commit"],
                "evidence_as_of": "2026-07-23",
                "source_evidence": evidence["source_evidence"],
                "qualification_status": evidence["qualification_status"],
                "blockers": evidence["blockers"],
            }
        )
    return rows


def _license_rows() -> list[dict[str, object]]:
    rows = []
    for baseline_id in MAIN_BASELINES:
        if baseline_id not in EXTERNAL_ACTIVE_IDS:
            continue
        evidence = EXTERNAL_EVIDENCE[baseline_id]
        code_permission = (
            "permitted_under_recorded_code_license"
            if evidence["code_license"] in {"MIT", "BSD-3-Clause"}
            else "not_qualified"
        )
        rows.append(
            {
                "baseline_id": baseline_id,
                "code_license": evidence["code_license"],
                "code_license_hash": evidence["code_license_hash"],
                "code_redistribution_permission": code_permission,
                "model_weight_license": evidence["model_weight_license"],
                "model_weight_terms": evidence["model_weight_terms"],
                "training_data_terms": evidence["training_data_terms"],
                "training_data_evidence": evidence["training_data_evidence"],
                "evaluation_data_terms": evidence["evaluation_data_terms"],
                "evaluation_data_evidence": evidence["evaluation_data_evidence"],
                "weights_redistribution_permission": "not_approved",
                "data_redistribution_permission": "not_approved",
                "anonymous_package_permission": "not_approved_for_complete_package",
                "qualification_status": evidence["qualification_status"],
                "blockers": evidence["blockers"],
            }
        )
    return rows


def _replacement_decision() -> dict[str, object]:
    chemical = [
        {"baseline_id": baseline_id, **REPLACEMENT_EVIDENCE[baseline_id]}
        for baseline_id in ("cpa", "scgen")
    ]
    distribution = [{"baseline_id": "conditional_monge_gap", **REPLACEMENT_EVIDENCE["conditional_monge_gap"]}]
    return {
        "schema_version": "baseline_registry_qualification_replacement_decision",
        "active_set_version": "baseline_registry.1",
        "active_set_mutated": False,
        "groups": {
            "chemical_perturbation": {
                "primary": "chemcpa",
                "primary_qualification_status": EXTERNAL_EVIDENCE["chemcpa"]["qualification_status"],
                "replacement_candidates": chemical,
                "activated_replacement": None,
                "decision": "no_qualified_replacement",
                "reason": "chemCPA is blocked on data/weight package terms; neither CPA nor scGen has complete data, weight, redistribution, and anonymous-package evidence.",
                "primary_and_replacement_concurrent": False,
                "activation_invariant_passed": True,
            },
            "distribution_mapping": {
                "primary": "cellot",
                "primary_qualification_status": EXTERNAL_EVIDENCE["cellot"]["qualification_status"],
                "replacement_candidates": distribution,
                "activated_replacement": None,
                "decision": "no_qualified_replacement",
                "reason": "CellOT is blocked on data terms and Conditional Monge Gap has no official source/license binding.",
                "primary_and_replacement_concurrent": False,
                "activation_invariant_passed": True,
            },
        },
        "status": "passed_fail_closed",
    }


def _clone_rows() -> list[dict[str, object]]:
    rows = []
    for baseline_id in MAIN_BASELINES:
        if baseline_id not in EXTERNAL_ACTIVE_IDS:
            continue
        evidence = EXTERNAL_EVIDENCE[baseline_id]
        checkout = _checkout_state(baseline_id, evidence)
        rows.append(
            {
                "baseline_id": baseline_id,
                "qualification_status": evidence["qualification_status"],
                **checkout,
            }
        )
    return rows


def _smoke_rows(clone_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    clone_by_id = {str(row["baseline_id"]): row for row in clone_rows}
    rows = []
    for baseline_id in MAIN_BASELINES:
        if baseline_id not in EXTERNAL_ACTIVE_IDS:
            continue
        evidence = EXTERNAL_EVIDENCE[baseline_id]
        clone = clone_by_id[baseline_id]
        rows.append(
            {
                "baseline_id": baseline_id,
                "qualification_status": evidence["qualification_status"],
                "official_source_and_code_license_known": str(
                    evidence["official_repository_status"] == "verified"
                    and evidence["code_license"] in {"MIT", "BSD-3-Clause"}
                ).lower(),
                "clean_clone_requirement_met": str(clone["clean_clone_requirement_met"]).lower(),
                "upstream_smoke_status": "not_run_qualification_blocked",
                "permitted_smoke_type": "official_unit_test_or_tiny_example_or_checkpoint_inference_or_documented_one_step_training",
                "command": "",
                "stdout_path": "",
                "stderr_path": "",
                "native_reproduction_count_contribution": 0,
                "matched_training_count_contribution": 0,
                "reason": "No complete source/license/data/weight/redistribution/anonymous-package gate passed; no upstream smoke was run.",
            }
        )
    return rows


def build_all(registry_path: Path, schema_path: Path, lock_path: Path) -> dict[str, Any]:
    registry, _ = load_inputs(registry_path, schema_path, lock_path)
    active_rows = _active_rows(registry)
    source_rows = _source_rows()
    license_rows = _license_rows()
    clone_rows = _clone_rows()
    smoke_rows = _smoke_rows(clone_rows)
    replacement = _replacement_decision()

    external_statuses = {row["qualification_status"] for row in source_rows}
    if not external_statuses <= QUALIFICATION_STATUSES:
        raise ValueError("Source resolution uses an unsupported qualification status")
    if any(row["upstream_smoke_status"] != "not_run_qualification_blocked" for row in smoke_rows):
        raise ValueError("Qualification must not claim an upstream smoke without a complete qualification gate")
    if any(group["activated_replacement"] is not None for group in replacement["groups"].values()):
        raise ValueError("Qualification may not activate a replacement without complete qualification")

    return {
        "active_external_set": active_rows,
        "source_resolution": source_rows,
        "license_resolution": license_rows,
        "replacement_decision": replacement,
        "clone_manifest": clone_rows,
        "upstream_smoke_status": smoke_rows,
        "gate": {
            "schema_version": "baseline_registry_qualification_gate",
            "active_set_version": "baseline_registry.1",
            "historical_record_count": int(registry["active_set_contract"]["historical_record_count"]),
            "main_baseline_count": len(MAIN_BASELINES),
            "main_ablation_count": len(MAIN_ABLATIONS),
            "internal_no_external_source_count": len(INTERNAL_ACTIVE_IDS),
            "external_source_required_count": len(EXTERNAL_ACTIVE_IDS),
            "source_license_passed_external_count": sum(row["qualification_status"] == "approved" for row in source_rows),
            "clean_clone_performed_count": 0,
            "upstream_smoke_run_count": 0,
            "native_reproduction_count": 0,
            "matched_training_count": 0,
            "headline_ready_count": 0,
            "round11_closure": "blocked_full_suite",
            "round11_ceiling": "upstream_smoke_wave1_complete",
            "baseline_native_reproduction_allowed": False,
            "baseline_adapter_training_smoke_allowed": False,
            "matched_protocol_training_allowed": False,
            "evidence_fabric_sealing_allowed": False,
            "DT_CORE_allowed": False,
            "status": "blocked_source_or_license",
            "blockers": [
                "round11_closure_blocked_full_suite",
                "no_external_active_baseline_has_complete_source_license_weight_data_package_terms",
                "no_clean_external_baseline_clone_performed",
                "no_upstream_smoke_run",
            ],
        },
        "environments": ENVIRONMENT_SPECS,
    }


def _write_environment_specs(environments: dict[str, dict[str, Any]], environment_root: Path) -> None:
    for baseline_id, spec in environments.items():
        rendered = yaml.safe_dump(spec, sort_keys=False, allow_unicode=False)
        write_text_atomic(environment_root / f"{baseline_id}.yaml", rendered)


def write_all(payload: dict[str, Any], audit_root: Path, environment_root: Path) -> None:
    for key, filename in AUDIT_FILES.items():
        value = payload[key]
        path = audit_root / filename
        if path.suffix == ".tsv":
            rows = value
            if not rows:
                raise ValueError(f"No rows generated for required audit {filename}")
            write_tsv(path, list(rows[0]), rows)
        else:
            write_json(path, value)
    _write_environment_specs(payload["environments"], environment_root)


def main() -> None:
    args = parse_args()
    payload = build_all(
        repo_path(args.registry),
        repo_path(args.schema),
        repo_path(args.lock_path),
    )
    write_all(payload, repo_path(args.audit_root), repo_path(args.environment_root))
    print(
        {
            "status": payload["gate"]["status"],
            "external_source_required_count": payload["gate"]["external_source_required_count"],
            "upstream_smoke_run_count": payload["gate"]["upstream_smoke_run_count"],
        }
    )


if __name__ == "__main__":
    main()
