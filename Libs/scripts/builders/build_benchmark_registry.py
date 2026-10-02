from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.alignment_benchmark import (
    binary_label,
    choose_group_identifier,
    join_values,
    load_alignment_catalog,
    normalize_diagnosis_label,
    write_parquet_atomic,
)
from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark-ready cohort, folds, and split registries from alignment-ready bundles.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def grouped_validation_assignments(
    training_frame: pd.DataFrame,
    validation_fraction: float,
    minimum_validation_groups_per_label: int,
    random_seed: int,
) -> dict[str, str]:
    training_frame = training_frame.copy()
    group_labels = training_frame.groupby("group_id")["binary_label"].nunique()
    mixed_groups = sorted(group_labels[group_labels > 1].index.tolist())
    if mixed_groups:
        mixed_mask = training_frame["group_id"].isin(mixed_groups)
        training_frame.loc[mixed_mask, "group_id"] = training_frame.loc[mixed_mask, "sample_uid"]
    assignments = {sample_uid: "train" for sample_uid in training_frame["sample_uid"].tolist()}
    if training_frame.empty:
        return assignments
    for label_value in sorted(training_frame["binary_label"].unique()):
        label_groups = sorted(training_frame.loc[training_frame["binary_label"] == label_value, "group_id"].unique().tolist())
        if len(label_groups) <= 1:
            continue
        requested = max(minimum_validation_groups_per_label, int(round(len(label_groups) * validation_fraction)))
        requested = min(requested, len(label_groups) - 1)
        if requested <= 0:
            continue
        ordered_groups = pd.Series(label_groups).sample(frac=1.0, random_state=random_seed + int(label_value)).tolist()
        for sample_uid in training_frame.loc[training_frame["group_id"].isin(ordered_groups[:requested]), "sample_uid"].tolist():
            assignments[sample_uid] = "validation"
    return assignments


def grouped_train_validation_test_assignments(
    study_frame: pd.DataFrame,
    holdout_fraction: float,
    minimum_groups_per_label: int,
    random_seed: int,
) -> dict[str, str]:
    study_frame = study_frame.copy()
    assignments = {sample_uid: "train" for sample_uid in study_frame["sample_uid"].tolist()}
    if study_frame.empty:
        return assignments
    for label_value in sorted(study_frame["binary_label"].unique()):
        label_groups = sorted(study_frame.loc[study_frame["binary_label"] == label_value, "group_id"].unique().tolist())
        if len(label_groups) <= 2:
            continue
        requested = max(minimum_groups_per_label, int(round(len(label_groups) * holdout_fraction)))
        requested = min(requested, max(len(label_groups) - 2, 1))
        ordered_groups = pd.Series(label_groups).sample(frac=1.0, random_state=random_seed + int(label_value)).tolist()
        test_groups = set(ordered_groups[:requested])
        validation_groups = set(ordered_groups[requested : requested * 2]) if len(ordered_groups) >= requested * 2 else set()
        for sample_uid in study_frame.loc[study_frame["group_id"].isin(test_groups), "sample_uid"].tolist():
            assignments[sample_uid] = "test"
        for sample_uid in study_frame.loc[study_frame["group_id"].isin(validation_groups), "sample_uid"].tolist():
            assignments[sample_uid] = "validation"
    if "test" not in assignments.values():
        validation_assignments = grouped_validation_assignments(study_frame, holdout_fraction, minimum_groups_per_label, random_seed)
        for sample_uid, split_role in validation_assignments.items():
            if split_role == "validation":
                assignments[sample_uid] = "test"
                break
    if "validation" not in assignments.values():
        remaining_frame = study_frame.loc[study_frame["sample_uid"].map(assignments) == "train"].copy()
        validation_assignments = grouped_validation_assignments(remaining_frame, holdout_fraction, minimum_groups_per_label, random_seed + 101)
        for sample_uid, split_role in validation_assignments.items():
            if split_role == "validation":
                assignments[sample_uid] = "validation"
    return assignments


def fold_root(benchmark_name: str, fold_name: str, config: dict) -> Path:
    return resolve_repo_path(
        config["benchmark_registry"]["fold_root_template"].format(
            benchmark_name=benchmark_name,
            fold_name=fold_name,
        )
    )


def fold_root_base(config: dict) -> Path:
    template_prefix = str(config["benchmark_registry"]["fold_root_template"]).split("{benchmark_name}", 1)[0].rstrip("/")
    return resolve_repo_path(template_prefix)


def archive_orphan_fold_directories(config: dict, fold_rows: list[dict[str, object]]) -> None:
    base_root = fold_root_base(config)
    if not base_root.exists():
        return
    archive_root = resolve_repo_path(config["benchmark_registry"]["orphan_fold_archive_root"])
    expected_directories = {
        (str(row["benchmark_name"]), str(row["fold_name"]))
        for row in fold_rows
    }
    for benchmark_dir in base_root.iterdir():
        if not benchmark_dir.is_dir():
            continue
        for fold_dir in benchmark_dir.iterdir():
            if not fold_dir.is_dir():
                continue
            relative_key = (benchmark_dir.name, fold_dir.name)
            if relative_key in expected_directories:
                continue
            archive_target = archive_root / benchmark_dir.name / fold_dir.name
            ensure_directory(archive_target.parent)
            if archive_target.exists():
                if archive_target.is_dir():
                    shutil.rmtree(archive_target)
                else:
                    archive_target.unlink()
            shutil.move(str(fold_dir), str(archive_target))


def disease_program_items(config: dict) -> list[tuple[str, dict]]:
    programs = config["benchmark_registry"].get("disease_programs", {})
    if not programs:
        active_name = str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
        return [(active_name, benchmark_program_config(config, active_name))]
    return [(program_name, benchmark_program_config(config, program_name)) for program_name in sorted(programs)]


def filtered_entity_frame(
    benchmark_config: dict,
    train_study_uids: list[str],
    modality: str,
    entity_indices: dict[str, pd.DataFrame],
    alignment_by_study: dict[str, dict[str, str]],
    config: dict,
) -> pd.DataFrame:
    modality_train_studies = [
        study_uid
        for study_uid in train_study_uids
        if alignment_by_study[study_uid]["modality"] == modality
    ]
    source_count_by_entity: dict[str, int] = defaultdict(int)
    entity_label_lookup: dict[str, str] = {}
    namespace_lookup: dict[str, str] = {}
    entity_type_lookup: dict[str, str] = {}
    for study_uid in modality_train_studies:
        entity_index = entity_indices[study_uid]
        for entity_row in entity_index.itertuples(index=False):
            canonical_entity_id = str(entity_row.canonical_entity_id)
            source_count_by_entity[canonical_entity_id] += 1
            entity_label_lookup.setdefault(canonical_entity_id, str(entity_row.canonical_entity_label))
            namespace_lookup.setdefault(canonical_entity_id, str(entity_row.namespace))
            entity_type_lookup.setdefault(canonical_entity_id, str(entity_row.entity_type))
    threshold = min(int(benchmark_config["feature_presence_threshold"][modality]), len(modality_train_studies))
    retained_entity_ids = [
        entity_id
        for entity_id, source_count in sorted(source_count_by_entity.items())
        if source_count >= threshold
    ]
    return pd.DataFrame(
        [
            {
                "row_index": index + 1,
                "canonical_entity_id": entity_id,
                "canonical_entity_label": entity_label_lookup[entity_id],
                "entity_type": entity_type_lookup[entity_id],
                "namespace": namespace_lookup[entity_id],
                "source_count": source_count_by_entity[entity_id],
            }
            for index, entity_id in enumerate(retained_entity_ids)
        ],
        columns=config["schemas"]["alignment_entity_fields"],
    )


def eligible_primary_study(
    study_uid: str,
    alignment_row: dict[str, str],
    binary_frame: pd.DataFrame,
    benchmark_config: dict,
    processed_study_row: dict[str, str],
) -> tuple[bool, str]:
    if alignment_row["source_accession"] not in set(benchmark_config["primary_source_accessions"]):
        return False, "not_in_benchmark_config"
    if binary_frame.empty:
        return False, "primary_study_missing_binary_class"
    if int((binary_frame["binary_label"] == 1).sum()) <= 0 or int((binary_frame["binary_label"] == 0).sum()) <= 0:
        return False, "primary_study_missing_binary_class"
    study_sample_context = str(processed_study_row.get("sample_context", alignment_row.get("sample_context", "")))
    subject_linkable_flag = str(processed_study_row.get("subject_linkable_flag", alignment_row.get("subject_linkable_flag", "")))
    required_sample_context = str(benchmark_config.get("required_sample_context", ""))
    if required_sample_context and study_sample_context != required_sample_context:
        return False, "sample_context_mismatch"
    required_subject_linkable_flag = str(benchmark_config.get("required_subject_linkable_flag", ""))
    if required_subject_linkable_flag and subject_linkable_flag != required_subject_linkable_flag:
        return False, "subject_linkable_requirement_failed"
    required_modalities = set(benchmark_config.get("required_modalities", []))
    if required_modalities and alignment_row.get("modality", "") not in required_modalities:
        return False, "modality_not_admitted"
    allowed_tissues = set(benchmark_config.get("allowed_tissues", []))
    if allowed_tissues and alignment_row.get("tissue_or_biofluid", "") not in allowed_tissues:
        return False, "tissue_not_admitted"
    return True, ""


def build_split_rows(
    benchmark_name: str,
    disease_program: str,
    fold_name: str,
    headline_flag: str,
    rows: list[dict[str, object]],
    alignment_by_study: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    split_rows: list[dict[str, object]] = []
    for row in rows:
        split_rows.append(
            {
                "benchmark_name": benchmark_name,
                "disease_program": disease_program,
                "fold_name": fold_name,
                "study_uid": row["study_uid"],
                "source_key": alignment_by_study[row["study_uid"]]["source_key"],
                "source_accession": alignment_by_study[row["study_uid"]]["source_accession"],
                "modality": row["modality"],
                "sample_uid": row["sample_uid"],
                "group_id": row["group_id"],
                "normalized_diagnosis_label": row["normalized_diagnosis_label"],
                "binary_label": int(row["binary_label"]),
                "split_role": row["split_role"],
                "headline_fold_flag": headline_flag,
            }
        )
    return split_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    alignment_rows = load_alignment_catalog(config)
    pipeline_logger.progress(
        "loaded_benchmark_inputs",
        alignment_row_count=len(alignment_rows),
    )
    alignment_by_study = {row["study_uid"]: row for row in alignment_rows}
    processed_root = resolve_repo_path(config["processed_registry"]["local_root"])
    processed_study_rows = read_tsv(processed_root / config["processed_registry"]["relative_paths"]["study_catalog"])
    processed_study_by_accession = {row["source_accession"]: row for row in processed_study_rows}
    diagnosis_aliases = config["cohort_ready_registry"]["diagnosis_aliases"]
    atlas_prior_root = resolve_repo_path(config["atlas_prior_registry"]["local_root"])
    atlas_prior_ready_flag = (
        "yes"
        if (atlas_prior_root / config["atlas_prior_registry"]["relative_paths"]["program_catalog"]).exists()
        else "no"
    )

    study_frames: dict[str, pd.DataFrame] = {}
    entity_indices: dict[str, pd.DataFrame] = {}
    for study_uid, alignment_row in alignment_by_study.items():
        sample_index = pd.read_parquet(resolve_repo_path(alignment_row["sample_index_path"])).copy()
        sample_index["study_uid"] = study_uid
        sample_index["source_key"] = alignment_row["source_key"]
        sample_index["source_accession"] = alignment_row["source_accession"]
        sample_index["group_id"] = sample_index.apply(choose_group_identifier, axis=1)
        study_frames[study_uid] = sample_index
        entity_indices[study_uid] = pd.read_parquet(resolve_repo_path(alignment_row["entity_index_path"])).copy()

    cohort_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []

    for disease_program, benchmark_config in disease_program_items(config):
        positive_labels = set(benchmark_config["positive_labels"])
        negative_labels = set(benchmark_config["negative_labels"])
        primary_accessions = set(benchmark_config["primary_source_accessions"])
        auxiliary_accessions = set(benchmark_config["auxiliary_source_accessions"])
        primary_candidate_studies: list[str] = []
        primary_candidate_modality: dict[str, str] = {}
        cohort_row_index_by_study_uid: dict[str, int] = {}
        seen_accessions: set[str] = set()

        binary_frames: dict[str, pd.DataFrame] = {}
        for study_uid, alignment_row in sorted(alignment_by_study.items(), key=lambda item: (item[1]["source_key"], item[1]["source_accession"])):
            sample_index = study_frames[study_uid].copy()
            sample_index["normalized_diagnosis_label"] = sample_index["sample_diagnosis_label"].astype(str).apply(
                lambda value: normalize_diagnosis_label(value, diagnosis_aliases)
            )
            sample_index["binary_label"] = sample_index["normalized_diagnosis_label"].apply(
                lambda value: binary_label(value, positive_labels, negative_labels)
            )
            binary_frame = sample_index.loc[sample_index["binary_label"].notna()].copy()
            binary_frames[study_uid] = binary_frame
            positive_sample_count = int((binary_frame["binary_label"] == 1).sum())
            negative_sample_count = int((binary_frame["binary_label"] == 0).sum())
            total_sample_count = int(binary_frame.shape[0])
            source_accession = alignment_row["source_accession"]
            processed_study_row = processed_study_by_accession.get(source_accession, {})
            subject_linkable_flag = str(processed_study_row.get("subject_linkable_flag", alignment_row.get("subject_linkable_flag", "yes") or "yes"))
            seen_accessions.add(source_accession)
            cohort_role = ""
            inclusion_flag = "no"
            exclusion_reason = ""
            if source_accession in primary_accessions:
                cohort_role = "primary_train_eval"
                is_eligible, eligibility_reason = eligible_primary_study(
                    study_uid,
                    alignment_row,
                    binary_frame,
                    benchmark_config,
                    processed_study_row,
                )
                inclusion_flag, exclusion_reason = ("yes", "") if is_eligible else ("no", eligibility_reason)
                if inclusion_flag == "yes":
                    primary_candidate_studies.append(study_uid)
                    primary_candidate_modality[study_uid] = str(alignment_row["modality"])
            elif source_accession in auxiliary_accessions:
                cohort_role = "auxiliary_external"
                inclusion_flag = "yes"
            else:
                cohort_role = "not_selected"
                exclusion_reason = "not_in_benchmark_config"
            headline_flag = "yes" if total_sample_count >= int(benchmark_config["headline_minimum_test_sample_count"]) else "no"
            cohort_rows.append(
                {
                    "benchmark_name": str(benchmark_config["benchmark_name"]),
                    "disease_program": disease_program,
                    "study_uid": study_uid,
                    "source_key": alignment_row["source_key"],
                    "source_accession": source_accession,
                    "world_model_role": "disease_observation",
                    "subject_linkable_flag": subject_linkable_flag,
                    "atlas_prior_ready_flag": atlas_prior_ready_flag,
                    "public_only_flag": "yes",
                    "modality": alignment_row["modality"],
                    "tissue_or_biofluid": alignment_row["tissue_or_biofluid"],
                    "cohort_role": cohort_role,
                    "target_label_space": benchmark_config["target_label_space"],
                    "positive_sample_count": positive_sample_count,
                    "negative_sample_count": negative_sample_count,
                    "total_sample_count": total_sample_count,
                    "headline_fold_eligible_flag": headline_flag,
                    "inclusion_flag": inclusion_flag,
                    "exclusion_reason": exclusion_reason,
                }
            )
            cohort_row_index_by_study_uid[study_uid] = len(cohort_rows) - 1

        modality_candidate_counts: dict[str, int] = defaultdict(int)
        for study_uid in primary_candidate_studies:
            modality_candidate_counts[primary_candidate_modality[study_uid]] += 1
        eligible_primary_studies = sorted(
            study_uid
            for study_uid in primary_candidate_studies
            if modality_candidate_counts[primary_candidate_modality[study_uid]] >= 2
            or (
                benchmark_config.get("single_cohort_benchmark_name")
                and primary_candidate_modality[study_uid] in set(benchmark_config.get("required_modalities", []))
            )
        )
        eligible_primary_lookup = set(eligible_primary_studies)
        for study_uid in primary_candidate_studies:
            if study_uid in eligible_primary_lookup:
                continue
            cohort_row = cohort_rows[cohort_row_index_by_study_uid[study_uid]]
            cohort_row["inclusion_flag"] = "no"
            cohort_row["exclusion_reason"] = "modality_lacks_peer_primary_study"

        for source_accession in sorted((primary_accessions | auxiliary_accessions) - seen_accessions):
            processed_row = processed_study_by_accession.get(source_accession, {})
            blocked_asset_count = int(processed_row.get("blocked_asset_count", "0") or "0")
            not_available_asset_count = int(processed_row.get("not_available_asset_count", "0") or "0")
            if blocked_asset_count > 0:
                exclusion_reason = "blocked_upstream_download"
            elif not_available_asset_count > 0:
                exclusion_reason = "upstream_asset_not_available"
            else:
                exclusion_reason = "not_alignment_ready"
            cohort_rows.append(
                {
                    "benchmark_name": str(benchmark_config["benchmark_name"]),
                    "disease_program": disease_program,
                    "study_uid": processed_row.get("study_uid", source_accession),
                    "source_key": processed_row.get("source_key", ""),
                    "source_accession": source_accession,
                    "world_model_role": processed_row.get("world_model_role", "disease_observation"),
                    "subject_linkable_flag": processed_row.get("subject_linkable_flag", ""),
                    "atlas_prior_ready_flag": atlas_prior_ready_flag,
                    "public_only_flag": "yes",
                    "modality": processed_row.get("modality", ""),
                    "tissue_or_biofluid": processed_row.get("tissue_or_biofluid", ""),
                    "cohort_role": "primary_train_eval" if source_accession in primary_accessions else "auxiliary_external",
                    "target_label_space": benchmark_config["target_label_space"],
                    "positive_sample_count": 0,
                    "negative_sample_count": 0,
                    "total_sample_count": 0,
                    "headline_fold_eligible_flag": "no",
                    "inclusion_flag": "no",
                    "exclusion_reason": exclusion_reason,
                }
            )

        benchmark_name = str(benchmark_config.get("cross_study_benchmark_name", benchmark_config["benchmark_name"]))
        if len(eligible_primary_studies) >= 2:
            for held_out_study_uid in eligible_primary_studies:
                held_out_alignment = alignment_by_study[held_out_study_uid]
                held_out_binary = binary_frames[held_out_study_uid]
                train_study_uids = sorted(study_uid for study_uid in eligible_primary_studies if study_uid != held_out_study_uid)
                held_out_modality = str(held_out_alignment["modality"])
                if held_out_modality not in {
                    str(alignment_by_study[study_uid]["modality"])
                    for study_uid in train_study_uids
                }:
                    continue
                training_pool = pd.concat([binary_frames[study_uid].copy() for study_uid in train_study_uids], ignore_index=True)
                assignments = grouped_validation_assignments(
                    training_pool,
                    float(benchmark_config["validation_fraction"]),
                    int(benchmark_config["minimum_validation_groups_per_label"]),
                    int(benchmark_config["logistic_regression"]["random_seed"]),
                )
                fold_name = f"heldout_{held_out_alignment['source_accession'].lower().replace('-', '_')}"
                headline_flag = "yes" if held_out_binary.shape[0] >= int(benchmark_config["headline_minimum_test_sample_count"]) else "no"
                entity_frame_by_modality: dict[str, pd.DataFrame] = {}
                for modality in sorted({held_out_modality, *[alignment_by_study[study_uid]["modality"] for study_uid in train_study_uids]}):
                    entity_frame = filtered_entity_frame(benchmark_config, train_study_uids, modality, entity_indices, alignment_by_study, config)
                    if entity_frame.empty:
                        continue
                    entity_frame_by_modality[modality] = entity_frame
                split_source_rows: list[dict[str, object]] = []
                for training_row in training_pool.to_dict("records"):
                    split_source_rows.append({**training_row, "split_role": assignments[training_row["sample_uid"]]})
                for held_out_row in held_out_binary.to_dict("records"):
                    split_source_rows.append({**held_out_row, "split_role": "test"})
                sample_split_rows = build_split_rows(benchmark_name, disease_program, fold_name, headline_flag, split_source_rows, alignment_by_study)
                supported_modalities = [
                    modality
                    for modality in sorted(entity_frame_by_modality)
                    if all(
                        any(
                            row["split_role"] == split_name and row["modality"] == modality
                            for row in sample_split_rows
                        )
                        for split_name in ["train", "validation", "test"]
                    )
                ]
                if held_out_modality not in supported_modalities:
                    continue
                fold_rows.append(
                    {
                        "benchmark_name": benchmark_name,
                        "disease_program": disease_program,
                        "fold_name": fold_name,
                        "held_out_study_uid": held_out_study_uid,
                        "train_study_uids": join_values(train_study_uids),
                        "validation_grouping_key": "subject_id_or_sample_uid",
                        "target_label_space": benchmark_config["target_label_space"],
                        "included_modalities": join_values(supported_modalities),
                        "headline_metric_flag": headline_flag,
                        "test_sample_count": held_out_binary.shape[0],
                    }
                )
                current_fold_root = fold_root(benchmark_name, fold_name, config)
                ensure_directory(current_fold_root)
                split_path = current_fold_root / benchmark_config["fold_filenames"]["sample_split"]
                write_parquet_atomic(pd.DataFrame(sample_split_rows), split_path)
                for modality, filename_key in [("transcriptomics", "transcriptomics_entity_index"), ("metabolomics", "metabolomics_entity_index")]:
                    if modality not in supported_modalities:
                        continue
                    entity_frame = entity_frame_by_modality[modality]
                    entity_index_path = current_fold_root / benchmark_config["fold_filenames"][filename_key]
                    write_parquet_atomic(entity_frame, entity_index_path)
                    for dataset_role in ["train", "validation", "test"]:
                        role_samples = [row for row in sample_split_rows if row["split_role"] == dataset_role and row["modality"] == modality]
                        dataset_rows.append(
                            {
                                "benchmark_name": benchmark_name,
                                "disease_program": disease_program,
                                "fold_name": fold_name,
                                "dataset_role": dataset_role,
                                "modality": modality,
                                "sample_split_path": str(split_path.relative_to(REPO_ROOT)),
                                "entity_index_path": str(entity_index_path.relative_to(REPO_ROOT)),
                                "sample_count": len(role_samples),
                                "feature_count": int(entity_frame.shape[0]),
                                "included_study_uids": join_values(sorted({row["study_uid"] for row in role_samples})),
                                "note": benchmark_config["target_label_space"],
                            }
                        )
        elif len(eligible_primary_studies) == 1 and benchmark_config.get("single_cohort_benchmark_name"):
            held_out_study_uid = eligible_primary_studies[0]
            held_out_alignment = alignment_by_study[held_out_study_uid]
            benchmark_name = str(benchmark_config["single_cohort_benchmark_name"])
            fold_name = f"heldout_{held_out_alignment['source_accession'].lower().replace('-', '_')}_single_cohort"
            single_study_frame = binary_frames[held_out_study_uid].copy()
            assignments = grouped_train_validation_test_assignments(
                single_study_frame,
                float(benchmark_config["validation_fraction"]),
                int(benchmark_config["minimum_validation_groups_per_label"]),
                int(benchmark_config["logistic_regression"]["random_seed"]),
            )
            split_source_rows = [{**row, "split_role": assignments[row["sample_uid"]]} for row in single_study_frame.to_dict("records")]
            headline_flag = "yes" if int(sum(value == "test" for value in assignments.values())) >= int(benchmark_config["headline_minimum_test_sample_count"]) else "no"
            fold_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "disease_program": disease_program,
                    "fold_name": fold_name,
                    "held_out_study_uid": held_out_study_uid,
                    "train_study_uids": held_out_study_uid,
                    "validation_grouping_key": "subject_id_or_sample_uid",
                    "target_label_space": benchmark_config["target_label_space"],
                    "included_modalities": held_out_alignment["modality"],
                    "headline_metric_flag": headline_flag,
                    "test_sample_count": sum(value == "test" for value in assignments.values()),
                }
            )
            sample_split_rows = build_split_rows(benchmark_name, disease_program, fold_name, headline_flag, split_source_rows, alignment_by_study)
            current_fold_root = fold_root(benchmark_name, fold_name, config)
            ensure_directory(current_fold_root)
            split_path = current_fold_root / benchmark_config["fold_filenames"]["sample_split"]
            write_parquet_atomic(pd.DataFrame(sample_split_rows), split_path)
            for modality, filename_key in [("transcriptomics", "transcriptomics_entity_index"), ("metabolomics", "metabolomics_entity_index")]:
                entity_frame = filtered_entity_frame(benchmark_config, [held_out_study_uid], modality, entity_indices, alignment_by_study, config)
                if entity_frame.empty:
                    continue
                entity_index_path = current_fold_root / benchmark_config["fold_filenames"][filename_key]
                write_parquet_atomic(entity_frame, entity_index_path)
                for dataset_role in ["train", "validation", "test"]:
                    role_samples = [row for row in sample_split_rows if row["split_role"] == dataset_role and row["modality"] == modality]
                    dataset_rows.append(
                        {
                            "benchmark_name": benchmark_name,
                            "disease_program": disease_program,
                            "fold_name": fold_name,
                            "dataset_role": dataset_role,
                            "modality": modality,
                            "sample_split_path": str(split_path.relative_to(REPO_ROOT)),
                            "entity_index_path": str(entity_index_path.relative_to(REPO_ROOT)),
                            "sample_count": len(role_samples),
                            "feature_count": int(entity_frame.shape[0]),
                            "included_study_uids": held_out_study_uid if role_samples else "",
                            "note": f"{benchmark_config['target_label_space']};single_cohort",
                        }
                    )

    benchmark_root = resolve_repo_path(config["benchmark_registry"]["local_root"])
    ensure_directory(benchmark_root)
    relative_paths = config["benchmark_registry"]["relative_paths"]
    cohort_rows.sort(key=lambda row: (row["disease_program"], row["cohort_role"], row["source_key"], row["source_accession"]))
    fold_rows.sort(key=lambda row: (row["disease_program"], row["benchmark_name"], row["fold_name"]))
    dataset_rows.sort(key=lambda row: (row["disease_program"], row["benchmark_name"], row["fold_name"], row["dataset_role"], row["modality"]))
    write_tsv(benchmark_root / relative_paths["cohort_catalog"], config["schemas"]["benchmark_cohort_fields"], cohort_rows)
    write_tsv(benchmark_root / relative_paths["fold_catalog"], config["schemas"]["benchmark_fold_fields"], fold_rows)
    write_tsv(benchmark_root / relative_paths["training_dataset_catalog"], config["schemas"]["benchmark_training_dataset_fields"], dataset_rows)
    archive_orphan_fold_directories(config, fold_rows)
    baseline_result_catalog_path = benchmark_root / relative_paths["baseline_result_catalog"]
    existing_result_rows = read_tsv(baseline_result_catalog_path) if baseline_result_catalog_path.exists() else []
    write_tsv(benchmark_root / relative_paths["baseline_result_catalog"], config["schemas"]["benchmark_result_fields"], existing_result_rows)
    pipeline_logger.artifact(
        "benchmark_registry_written",
        artifact=benchmark_root,
        cohort_row_count=len(cohort_rows),
        fold_row_count=len(fold_rows),
        dataset_row_count=len(dataset_rows),
        baseline_result_row_count=len(existing_result_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=benchmark_root,
        cohort_row_count=len(cohort_rows),
        fold_row_count=len(fold_rows),
        dataset_row_count=len(dataset_rows),
        baseline_result_row_count=len(existing_result_rows),
    )


if __name__ == "__main__":
    main()
