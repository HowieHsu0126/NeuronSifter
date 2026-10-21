from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.dynamics_fields import update_world_model_rows_with_alignment
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public AD/aging world-model-ready study catalog from model-input bundles and benchmark backbone rows.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def stable_structural_prior_uid() -> str:
    return "human_atlas_plus_disease_vulnerability"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    benchmark_root = resolve_repo_path(config["benchmark_registry"]["local_root"])
    world_model_root = resolve_repo_path(config["world_model_ready_registry"]["local_root"])
    ensure_directory(world_model_root)

    bundle_rows = read_tsv(model_input_root / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"])
    cohort_rows = read_tsv(benchmark_root / config["benchmark_registry"]["relative_paths"]["cohort_catalog"])
    dynamics_root = resolve_repo_path(config["dynamics_ready_registry"]["local_root"])
    sample_alignment_rows = read_tsv(dynamics_root / config["dynamics_ready_registry"]["relative_paths"]["sample_alignment_catalog"])
    study_alignment_root = dynamics_root / config["dynamics_ready_registry"]["relative_paths"]["study_alignment_root"]
    structural_prior_root = resolve_repo_path(config["structural_prior_registry"]["local_root"])
    structural_prior_path = structural_prior_root / config["structural_prior_registry"]["relative_paths"]["region_prior_catalog"]
    path_by_study = {
        row["study_uid"]: study_alignment_root / f"{row['study_uid']}_field_alignment.tsv"
        for row in bundle_rows
    }
    benchmark_studies = {
        row["study_uid"]
        for row in cohort_rows
        if row["cohort_role"] in {"primary_train_eval", "auxiliary_external"} and row["inclusion_flag"] == "yes"
    }
    target_label_space_by_study: dict[str, str] = {}
    for study_uid in sorted({row["study_uid"] for row in cohort_rows}):
        included_rows = [
            row
            for row in cohort_rows
            if row["study_uid"] == study_uid and row["inclusion_flag"] == "yes"
        ]
        if not included_rows:
            continue
        disease_programs = sorted({str(row["disease_program"]) for row in included_rows})
        target_label_space_by_study[study_uid] = "|".join(disease_programs)
    pipeline_logger.progress(
        "loaded_world_model_inputs",
        bundle_row_count=len(bundle_rows),
        cohort_row_count=len(cohort_rows),
        benchmark_study_count=len(benchmark_studies),
    )

    output_rows: list[dict[str, object]] = []
    for row in bundle_rows:
        output_rows.append(
            {
                "study_uid": row["study_uid"],
                "source_key": row["source_key"],
                "source_accession": row["source_accession"],
                "modality": "transcriptomics" if row["source_key"] in {"geo", "arrayexpress"} else "metabolomics",
                "target_label_space": target_label_space_by_study.get(row["study_uid"], config["world_model_ready_registry"]["target_label_space"]),
                "benchmark_backbone_flag": "yes" if row["study_uid"] in benchmark_studies else "no",
                "world_model_role": row["world_model_role"],
                "subject_linkable_flag": row["subject_linkable_flag"],
                "atlas_prior_ready_flag": row["atlas_prior_ready_flag"],
                "public_only_flag": row["public_only_flag"],
                "model_input_bundle_path": row["bundle_root"],
                "field_alignment_uid": "",
                "field_alignment_bundle_path": "",
                "structural_prior_uid": stable_structural_prior_uid(),
                "structural_prior_bundle_path": str(structural_prior_path.resolve().relative_to(REPO_ROOT)),
                "observed_region_set": "",
                "note": "public_neurodegeneration_world_model_backbone",
            }
        )
    output_rows = update_world_model_rows_with_alignment(output_rows, sample_alignment_rows, path_by_study)
    output_rows.sort(key=lambda row: (str(row["source_key"]), str(row["source_accession"])))
    write_tsv(
        world_model_root / config["world_model_ready_registry"]["relative_paths"]["study_catalog"],
        config["schemas"]["world_model_ready_study_fields"],
        output_rows,
    )
    pipeline_logger.artifact(
        "world_model_registry_written",
        artifact=world_model_root,
        study_row_count=len(output_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=world_model_root,
        study_row_count=len(output_rows),
    )

if __name__ == "__main__":
    main()
