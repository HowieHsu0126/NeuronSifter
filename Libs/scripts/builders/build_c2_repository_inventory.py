"""Build the C2 repository inventory, reuse decisions, and owner map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


SEARCH_TERMS = (
    "world_action", "trajectory", "rollout", "action_compiler", "occupancy",
    "exposure", "regimen", "route", "dose", "schedule", "target_program",
    "mechanism_expert", "stochastic", "diffusion", "uncertainty", "constraint",
    "teacher", "simulator", "Transolver", "DiffusionPDE", "WDNO", "FNO", "DeepONet",
)

ALLOWED_DECISIONS = {
    "reuse_as_is",
    "extend_existing",
    "migrate",
    "replace_with_reason",
    "not_required",
    "blocked_missing_evidence",
}

CAPABILITIES: Mapping[str, tuple[str, tuple[str, ...], str, str, str]] = {
    "state_encoder": (
        "reuse_as_is",
        ("Libs/models/modules/world_state.py", "Libs/models/networks/foundation_state_encoder.py"),
        "Canonical state and reduced-core encoder implementations already exist.",
        "implemented",
        "models",
    ),
    "action_compiler": (
        "extend_existing",
        ("Libs/scripts/builders/build_action_ready_registry.py", "Libs/utils/model/screening_core.py"),
        "Regimen and occupancy-prior construction exists, but C2 requires provenance-bearing physical action fields.",
        "partial",
        "data_action",
    ),
    "pkpd_occupancy_compiler": (
        "extend_existing",
        ("Libs/scripts/builders/build_pkpd_ready_registry.py", "Libs/utils/model/dynamics_fields.py"),
        "Existing implementation consumes dose multipliers and model priors; a physical exposure-program contract is absent.",
        "partial",
        "model_pkpd",
    ),
    "trajectory_datasets": (
        "reuse_as_is",
        ("Input/processed/simulation_audit/simulation_qc_catalog.tsv", "Input/processed/dynamics_ready/public_ad_meso/simulation_catalog.tsv"),
        "QC-bound mesoscopic and full trajectory registries are materialized.",
        "implemented",
        "data_simulation",
    ),
    "teacher_registries": (
        "reuse_as_is",
        ("Input/processed/full_microenvironment_teacher_ready/full_microenvironment_teacher_catalog.tsv", "Input/processed/simulation_teacher_ready/simulation_teacher_catalog.tsv"),
        "Teacher provenance, family, and uncertainty fields are available for authority auditing.",
        "implemented",
        "data_simulation",
    ),
    "stochastic_operator": (
        "reuse_as_is",
        ("Libs/models/modules/neuronsifter_world_action.py",),
        "The project-native operator emits stochastic trajectory summaries and diagnostics.",
        "implemented",
        "models",
    ),
    "deterministic_operator": (
        "reuse_as_is",
        ("Libs/configs/exp/experiment_families/rollout_operator_deterministic_neuronsifter.yaml",),
        "A deterministic NeuronSifter profile is declared against the same mainline stack.",
        "implemented",
        "experiments",
    ),
    "baseline_adapters": (
        "extend_existing",
        ("Libs/configs/exp/paper_grade_baselines.yaml", "Libs/utils/exp/external_baseline_bridge.py", "Libs/models/modules/transolver_operator.py"),
        "Transolver and legacy baseline declarations exist; C2-compatible FNO, DeepONet, DiffusionPDE, WDNO, and biological perturbation adapters are not evidenced.",
        "partial",
        "experiments",
    ),
    "uq_metrics": (
        "extend_existing",
        ("Libs/utils/evaluation/screening_uq.py", "Libs/models/modules/neuronsifter_world_action.py"),
        "Screening UQ and sampled trajectory outputs exist; C2 CRPS, calibration, and paired-unit aggregation are not registered.",
        "partial",
        "evaluation",
    ),
    "constraint_metrics": (
        "extend_existing",
        ("Libs/utils/model/dynamics_fields.py", "Input/processed/simulation_audit/simulation_numeric_audit.tsv"),
        "Finite and mechanism checks exist; the C2 constraint result catalog is not implemented.",
        "partial",
        "evaluation",
    ),
    "slurm_launchers": (
        "extend_existing",
        ("Libs/exp/submission/submit_dynamics_profile_suite.sh", "Libs/configs/runtime/cluster.yaml"),
        "Existing launchers and resource profiles need C2-specific preflight binding and terminal aggregation.",
        "partial",
        "hpc",
    ),
    "report_builder": (
        "extend_existing",
        ("Libs/scripts/reports/build_paper_results.py",),
        "The central report builder exists but has no schema-valid C2 result ingestion path.",
        "partial",
        "reports",
    ),
    "c2_validators": (
        "blocked_missing_evidence",
        (),
        "No C2-specific authority, action, preflight, or G4 validator exists before this phase.",
        "missing",
        "validation",
    ),
}


def _binding(repo_root: Path, relative: str) -> dict[str, object]:
    path = repo_root / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    repo_root = repo_root.resolve()
    config_path = repo_root / "Libs/configs/config_stack.yaml"
    config = load_config(config_path)
    contract = config["structured_world_action_c2"]
    matched_paths: set[str] = set()
    for root in (repo_root / "Libs", repo_root / "Docs/design/iclr"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".sh"}:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if any(term.lower() in content or term.lower() in path.name.lower() for term in SEARCH_TERMS):
                matched_paths.add(path.relative_to(repo_root).as_posix())
    inventory = {
        "schema_version": "neuronsifter-c2-repository-inventory-v1",
        "status": "passed",
        "search_terms": list(SEARCH_TERMS),
        "matched_file_count": len(matched_paths),
        "matched_paths": sorted(matched_paths),
        "selected_state_corpus": dict(contract["selected_state_corpus"]),
        "c1_headline_reactivation_allowed": False,
    }
    decisions = []
    owner_paths: list[dict[str, object]] = []
    for capability, (decision, paths, reason, implementation_status, owner) in CAPABILITIES.items():
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Unsupported C2 reuse decision: {decision}")
        bindings = [_binding(repo_root, path) for path in paths]
        missing_paths = [row["path"] for row in bindings if not row["exists"]]
        decisions.append({
            "capability": capability,
            "decision": decision,
            "reason": reason,
            "implementation_status": implementation_status,
            "bindings": bindings,
            "missing_paths": missing_paths,
            "status": "passed" if not missing_paths else "gap_confirmed",
        })
        owner_paths.append({
            "capability": capability,
            "owner_domain": owner,
            "paths": [row["path"] for row in bindings],
            "existing_owner": "repository_component",
            "C2_action": decision,
        })
    reuse = {
        "schema_version": "neuronsifter-c2-reuse-gap-map-v1",
        "status": "passed",
        "allowed_decisions": sorted(ALLOWED_DECISIONS),
        "capabilities": decisions,
    }
    config_paths = ["Libs/configs/config_stack.yaml", "Libs/configs/exp/structured_world_action_c2.yaml", "Libs/configs/runtime/cluster.yaml"]
    path_map = {
        "schema_version": "neuronsifter-c2-path-owner-map-v1",
        "status": "passed",
        "config_stack_hash": canonical_sha256(config),
        "config_bindings": [_binding(repo_root, path) for path in config_paths],
        "owners": owner_paths,
    }
    return inventory, reuse, path_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="Output/audits")
    args = parser.parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.relative_to(REPO_ROOT.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in zip(
        ("c2_repository_inventory.json", "c2_reuse_and_gap_map.json", "c2_path_and_owner_map.json"),
        build(REPO_ROOT),
        strict=True,
    ):
        write_json(output_root / name, payload)
    print("c2_repository_inventory_status=passed")


if __name__ == "__main__":
    main()
