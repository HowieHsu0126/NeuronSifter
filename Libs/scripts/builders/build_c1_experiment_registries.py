"""Freeze C1 task, baseline, fairness, and configuration registries."""

from __future__ import annotations

import argparse
import json
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


BASELINES = (
    ("B0", "from_scratch_supervised_encoder", "from_scratch_supervised_encoder"),
    ("B1", "single_source_pretraining", "single_source_pretraining"),
    ("B2", "single_family_pretraining", "single_family_pretraining"),
    ("B3", "generic_masked_encoder", "generic_masked_encoder"),
    ("B4", "no_ontology_alignment", "no_ontology_alignment"),
    ("B5", "no_source_family_balancing", "no_source_family_balancing"),
    ("B6", "no_source_nuisance_control", "no_source_nuisance_control"),
    ("B7", "no_biological_unit_aggregation", "no_biological_unit_aggregation"),
    ("B8", "neuronsifter_reduced_core_full", "neuronsifter_foundation_state_encoder"),
)

OUTPUT_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _registry_hash(payload: Mapping[str, object]) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != "registry_hash"})


def _config_path(repo_root: Path, value: str | Path) -> tuple[Path, str]:
    path = value if isinstance(value, Path) else Path(value)
    path = path if path.is_absolute() else repo_root / path
    path = path.resolve()
    relative = path.relative_to(repo_root.resolve()).as_posix()
    return path, relative


def build(
    repo_root: Path = REPO_ROOT,
    *,
    config_path: str | Path = "Libs/configs/config_stack.yaml",
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    config_path, config_relative = _config_path(repo_root, config_path)
    config = load_config(config_path)
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    corpus_hash = str(frozen["identity_hash"])
    common = {
        "corpus_hash": corpus_hash,
        "bootstrap": {"method": "paired", "resamples": 2000, "seed": 3407, "interval": 0.95},
        "metric_views": ["per_family", "macro_family_average", "ess_weighted", "raw_observation_weighted_auxiliary"],
        "checkpoint_selection": {"data_role": "validation", "test_selection_allowed": False, "retain": ["best_validation", "last"]},
    }
    task_registry = {
        "schema_version": "neuronsifter-c1-task-registry-v1",
        "status": "frozen",
        **common,
        "few_shot_protocols": ["linear_probe", "1_shot", "5_shot", "10_shot", "full_fine_tune"],
        "tasks": [
            {
                "task_family": "disease_state_transfer",
                "task_ids": ["disease_control_classification", "heldout_cohort_retrieval", "subject_specimen_representation"],
                "biological_unit": "subject_specimen",
                "primary_metrics": ["balanced_accuracy", "macro_f1", "auroc"],
                "applicable_holdouts": ["source_publication_holdout", "family_holdout_cns_disease_aging_state_transcriptomics"],
                "not_applicable_holdouts": {"regimen_holdout": "no regimen axis", "target_program_holdout": "no target-program axis"},
            },
            {
                "task_family": "atlas_representation_transfer",
                "task_ids": ["brain_region_retrieval", "cell_context_retrieval", "cell_type_aware_donor_representation", "heldout_accession_transfer"],
                "biological_unit": "donor",
                "primary_metrics": ["recall_at_k", "mrr", "ndcg"],
                "applicable_holdouts": ["source_publication_holdout", "family_holdout_cns_cell_type_region_atlas_transcriptomics"],
                "not_applicable_holdouts": {"regimen_holdout": "no regimen axis", "target_program_holdout": "no target-program axis"},
            },
            {
                "task_family": "perturbation_action_transfer",
                "task_ids": ["condition_retrieval", "exposure_conditioned_response_prediction", "masked_response_reconstruction"],
                "biological_unit": "independent_condition_or_action",
                "primary_metrics": ["mrr", "rmse", "correlation"],
                "applicable_holdouts": ["source_publication_holdout", "family_holdout_compound_dose_time_perturbation_transcriptomics", "regimen_holdout", "target_program_holdout"],
                "not_applicable_holdouts": {},
            },
        ],
        "unavailable_tasks": [
            "transcriptomics_to_electrophysiology", "transcriptomics_to_morphology",
            "transcriptomics_to_imaging", "paired_multimodal_reconstruction",
            "same_subject_cross_modal_imputation", "masked_program_prediction",
        ],
    }
    task_registry["registry_hash"] = _registry_hash(task_registry)

    profiles = contract["baselines"]["runtime_profiles"]
    baseline_rows = []
    for baseline_code, baseline_id, profile_id in BASELINES:
        profile = profiles.get(profile_id) if profile_id else None
        baseline_rows.append({
            "baseline_code": baseline_code,
            "baseline_id": baseline_id,
            "runtime_profile_id": profile_id,
            "implementation_status": (
                "downstream_evaluator_required"
                if baseline_code == "B0"
                else ("pretraining_profile_implemented" if isinstance(profile, Mapping) else "implementation_required")
            ),
            "headline_ready": False,
            "headline_blocker": "C1 validation/few-shot/retrieval result runner is not implemented",
            "runtime_profile": dict(profile) if isinstance(profile, Mapping) else None,
            "variants": (
                ["disease_only", "atlas_only", "perturbation_only"]
                if baseline_code == "B2" else []
            ),
        })
    baseline_registry = {
        "schema_version": "neuronsifter-c1-baseline-registry-v1",
        "status": "frozen",
        "corpus_hash": corpus_hash,
        "baselines": baseline_rows,
        "external_foundation_models": {
            "status": "qualification_required",
            "copied_paper_numbers_allowed": False,
            "same_split_and_input_semantics_required": True,
        },
    }
    baseline_registry["registry_hash"] = _registry_hash(baseline_registry)

    fairness = {
        "schema_version": "neuronsifter-c1-baseline-fairness-v1",
        "status": "contract_frozen",
        "corpus_hash": corpus_hash,
        "same_backbone_constraints": [
            "identical_frozen_data", "identical_split", "identical_width_depth_within_scale",
            "identical_optimizer", "identical_training_steps", "identical_checkpoint_selection",
            "identical_tuning_budget", "identical_evaluation_budget",
        ],
        "allowed_difference": "declared_target_factor_only",
        "result_status": "requires_experiment_result",
    }
    fairness["registry_hash"] = _registry_hash(fairness)

    config_registry = {
        "schema_version": "neuronsifter-c1-config-registry-v1",
        "status": "frozen",
        "corpus_hash": corpus_hash,
        "config_stack_path": config_relative,
        "config_stack_sha256": sha256_file(config_path),
        "resolved_config_hash": canonical_sha256(config),
        "foundation_state_contract_hash": canonical_sha256(contract),
        "task_registry_hash": task_registry["registry_hash"],
        "baseline_registry_hash": baseline_registry["registry_hash"],
        "fairness_registry_hash": fairness["registry_hash"],
    }
    config_registry["registry_hash"] = _registry_hash(config_registry)
    return task_registry, baseline_registry, fairness, config_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-root", default="Output/audits")
    parser.add_argument("--output-prefix", default="c1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not OUTPUT_PREFIX_PATTERN.fullmatch(args.output_prefix):
        raise ValueError("C1 registry output prefix contains unsupported characters")
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.relative_to(REPO_ROOT.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        f"{args.output_prefix}_task_registry.json",
        f"{args.output_prefix}_baseline_registry.json",
        f"{args.output_prefix}_baseline_fairness_audit.json",
        f"{args.output_prefix}_config_registry.json",
    )
    for name, payload in zip(
        outputs,
        build(REPO_ROOT, config_path=args.config),
        strict=True,
    ):
        write_json(output_root / name, payload)
    print("c1_experiment_registries_status=frozen")


if __name__ == "__main__":
    main()
