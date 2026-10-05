"""Build the canonical C1 repository inventory and reuse decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


SEARCH_TERMS = (
    "foundation", "state_model", "masked", "pretrain", "source_invariance",
    "source_classifier", "family_classifier", "linear_probe", "few_shot",
    "retrieval", "modality_imputation", "scaling", "public_foundation", "C1", "G3",
)

CAPABILITIES: Mapping[str, tuple[str, tuple[str, ...], str, str]] = {
    "foundation_state_model": ("reuse_as_is", ("Libs/models/networks/foundation_state_encoder.py",), "C1-native transcriptomics encoder emits observation- and biological-unit representations with uncertainty and nuisance diagnostics", "implemented"),
    "masked_objectives": ("reuse_as_is", ("Libs/models/modules/foundation_state_objectives.py",), "masked reconstruction and unit aggregation exist", "implemented"),
    "ontology_embeddings": ("extend_existing", ("Libs/models/networks/foundation_state_encoder.py",), "ID embeddings exist; relation-aware evaluation remains", "partial"),
    "source_family_nuisance": ("reuse_as_is", ("Libs/models/networks/foundation_state_encoder.py", "Libs/models/modules/foundation_state_objectives.py"), "gradient-reversal nuisance heads exist", "implemented"),
    "uncertainty_posterior": ("reuse_as_is", ("Libs/models/networks/foundation_state_encoder.py",), "diagonal Gaussian posterior exists", "implemented"),
    "role_aware_loader": ("reuse_as_is", ("Libs/utils/data/foundation_state_c1_loader.py",), "hash-bound role-aware loader exists", "implemented"),
    "ess_aware_sampler": ("reuse_as_is", ("Libs/utils/data/foundation_state_c1_sampler.py",), "family and biological-unit sampler exists", "implemented"),
    "run_preflight": ("reuse_as_is", ("Libs/exp/training/preflight_foundation_state_c1_run.py",), "runtime replay authorization exists", "implemented"),
    "validation_checkpoint_selection": ("extend_existing", ("Libs/exp/training/train_foundation_state_c1.py",), "trainer currently records smoke objective rather than validation selection", "incomplete"),
    "linear_probe": ("extend_existing", ("Libs/exp/evaluation/evaluate_shared_latent_benchmark.py",), "shared evaluator exists but lacks frozen C1 task registry", "incomplete"),
    "few_shot_evaluation": ("extend_existing", ("Libs/exp/evaluation/evaluate_shared_latent_benchmark.py",), "C1 unit-level draw manifests remain", "incomplete"),
    "retrieval_evaluation": ("extend_existing", ("Libs/exp/evaluation/evaluate_shared_latent_benchmark.py",), "C1 retrieval protocols remain", "incomplete"),
    "shortcut_diagnostics": ("extend_existing", ("Libs/models/networks/foundation_state_encoder.py",), "training heads exist; independent diagnostic probes remain", "incomplete"),
    "paired_bootstrap_statistics": ("reuse_as_is", ("Libs/utils/evaluation/foundation_state_c1_metrics.py",), "unit-aware paired bootstrap is validated for canonical C1 units", "implemented"),
    "baseline_runner": ("extend_existing", ("Libs/exp/training/preflight_foundation_state_c1_run.py", "Libs/scripts/experiments/build_c1_execution_artifacts.py"), "preflight and matrix planning exist; train/evaluate runner remains", "incomplete"),
    "slurm_execution": ("extend_existing", ("Libs/scripts/monitoring/monitor_slurm_suite.sh", "Libs/scripts/experiments/launch_c1_job.py"), "resource profiles exist; C1 matrix submission and terminal aggregation remain", "incomplete"),
    "report_builder": ("extend_existing", ("Libs/scripts/reports/build_paper_results.py", "Libs/scripts/experiments/build_c1_execution_artifacts.py"), "paper builder exists; empirical C1 result ingestion remains", "incomplete"),
    "scaling_subsets": ("reuse_as_is", ("Libs/utils/data/foundation_state_c1.py",), "deterministic nested biological-unit subsets exist", "implemented"),
    "g3_validator": ("reuse_as_is", ("Libs/scripts/validators/validate_c1_g3_gate.py",), "fail-closed structural G3 validator exists", "implemented"),
    "cross_modal_imputation": ("not_required_for_reduced_core", ("Libs/models/networks/foundation_state_encoder.py",), "explicitly unavailable in C1", "not_required"),
}

CONFIG_PATHS = (
    "Libs/configs/config_stack.yaml",
    "Libs/configs/exp/foundation_state_c1.yaml",
    "Libs/configs/schema/foundation_state_c1.schema.json",
    "Libs/configs/schema/c1_foundation_corpus_preflight.schema.json",
    "Libs/configs/schema/c1_foundation_run_preflight.schema.json",
    "Libs/configs/schema/c1_unified_preflight.schema.json",
    "Libs/configs/schema/c1_execution_artifacts.schema.json",
    "Libs/configs/schema/c1_g3_gate.schema.json",
    "Libs/configs/runtime/cluster.yaml",
)

OUTPUT_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _binding(repo_root: Path, relative: str) -> dict[str, object]:
    path = repo_root / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _config_path(repo_root: Path, value: str | Path) -> tuple[Path, str]:
    path = value if isinstance(value, Path) else Path(value)
    path = path if path.is_absolute() else repo_root / path
    path = path.resolve()
    relative = path.relative_to(repo_root.resolve()).as_posix()
    return path, relative


def _config_bindings(
    config_relative: str,
    contract: Mapping[str, object],
) -> tuple[str, ...]:
    if config_relative == "Libs/configs/config_stack.yaml":
        return CONFIG_PATHS
    frozen = contract["frozen_corpus"]
    encoder = contract["encoder"]
    artifacts = contract["artifacts"]
    vocabulary_name = Path(
        str(encoder["input_contract"]["vocabulary_manifest_path"])
    ).name.replace(".json", ".schema.json")
    candidates = [
        config_relative,
        str(contract["schema_path"]),
        str(frozen["qualified_action_subset_schema_path"]),
        str(frozen["required_observation_binding"]["schema_path"]),
        f"Libs/configs/schema/{vocabulary_name}",
        str(artifacts["objective_capability_manifest_schema_path"]),
        str(artifacts["unified_preflight_schema_path"]),
        str(artifacts["run_preflight_schema_path"]),
        "Libs/configs/schema/c1_execution_artifacts.schema.json",
        "Libs/configs/schema/c1_g3_gate.schema.json",
        "Libs/configs/runtime/cluster.yaml",
    ]
    return tuple(dict.fromkeys(candidates))


def build(
    repo_root: Path = REPO_ROOT,
    *,
    config_path: str | Path = "Libs/configs/config_stack.yaml",
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    repo_root = repo_root.resolve()
    config_path, config_relative = _config_path(repo_root, config_path)
    config = load_config(config_path)
    c1 = config["foundation_state_c1"]
    corpus_hash = str(c1["frozen_corpus"]["identity_hash"])
    matched_paths: set[str] = set()
    for root in (repo_root / "Libs", repo_root / "Docs/design/iclr"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md", ".sh"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(term.lower() in text.lower() or term.lower() in path.name.lower() for term in SEARCH_TERMS):
                matched_paths.add(path.relative_to(repo_root).as_posix())
    inventory = {
        "schema_version": "neuronsifter-c1-repository-inventory-v1",
        "status": "passed",
        "corpus_hash": corpus_hash,
        "search_terms": list(SEARCH_TERMS),
        "matched_file_count": len(matched_paths),
        "matched_paths": sorted(matched_paths),
    }
    decisions = []
    for capability, (decision, paths, reason, implementation_status) in CAPABILITIES.items():
        bindings = [_binding(repo_root, path) for path in paths]
        missing = [row["path"] for row in bindings if not row["exists"]]
        decisions.append({
            "capability": capability,
            "decision": decision,
            "reason": reason,
            "implementation_status": implementation_status,
            "bindings": bindings,
            "missing_paths": missing,
            "status": "passed" if not missing else "gap_confirmed",
        })
    reuse = {
        "schema_version": "neuronsifter-c1-reuse-gap-map-v1",
        "status": "passed",
        "corpus_hash": corpus_hash,
        "allowed_decisions": ["reuse_as_is", "extend_existing", "migrate", "replace_with_reason", "not_required_for_reduced_core"],
        "capabilities": decisions,
    }
    bindings = [
        _binding(repo_root, path)
        for path in _config_bindings(config_relative, c1)
    ]
    config_map = {
        "schema_version": "neuronsifter-c1-config-path-map-v1",
        "status": "passed" if all(row["exists"] for row in bindings) else "blocked",
        "corpus_hash": corpus_hash,
        "config_stack_hash": canonical_sha256(config),
        "bindings": bindings,
    }
    return inventory, reuse, config_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C1 repository inventory artifacts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-root", default="Output/audits")
    parser.add_argument("--output-prefix", default="c1")
    args = parser.parse_args()
    if not OUTPUT_PREFIX_PATTERN.fullmatch(args.output_prefix):
        raise ValueError("C1 inventory output prefix contains unsupported characters")
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.relative_to(REPO_ROOT.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    inventory, reuse, config_map = build(REPO_ROOT, config_path=args.config)
    write_json(
        output_root / f"{args.output_prefix}_repository_inventory.json", inventory
    )
    write_json(
        output_root / f"{args.output_prefix}_reuse_and_gap_map.json", reuse
    )
    write_json(
        output_root / f"{args.output_prefix}_config_path_map.json", config_map
    )
    print(f"c1_repository_inventory_status={inventory['status']}")
    print(f"c1_repository_inventory_matched_files={inventory['matched_file_count']}")


if __name__ == "__main__":
    main()
