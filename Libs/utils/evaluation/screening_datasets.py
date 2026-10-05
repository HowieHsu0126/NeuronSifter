from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Libs.models.networks.shared_latent_dual_encoder import SharedLatentDualEncoder
from Libs.utils.evaluation.alignment_benchmark import binary_label, choose_group_identifier, load_alignment_bundle_artifacts, normalize_diagnosis_label
from Libs.utils.evaluation.benchmark_datasets import alignment_by_study
from Libs.utils.evaluation.benchmark_datasets import assemble_dataset, fold_sample_split, preferred_benchmark_name
from Libs.utils.data.raw_data_registry import read_tsv, resolve_repo_path


def resolve_output_root(config: dict) -> Path:
    return Path(config["benchmark_registry"]["artifact_root"]).resolve()


def shared_latent_artifact_dir(config: dict, fold_name: str, benchmark_name_override: str | None = None) -> Path:
    return (
        resolve_output_root(config)
        / (benchmark_name_override or preferred_benchmark_name(config))
        / "folds"
        / fold_name
        / "shared_latent_dual_encoder_classifier"
    )


def shared_latent_production_dir(config: dict, benchmark_name_override: str | None = None) -> Path:
    return (
        resolve_output_root(config)
        / (benchmark_name_override or preferred_benchmark_name(config))
        / "production"
        / "shared_latent_dual_encoder_classifier"
    )


def load_shared_latent_model(checkpoint_path: Path) -> SharedLatentDualEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SharedLatentDualEncoder(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def patient_state_feature_matrix(
    latent_matrix: np.ndarray,
    probability_vector: np.ndarray,
    patient_matrix: np.ndarray,
    differential: np.ndarray,
) -> np.ndarray:
    if latent_matrix.shape[0] == 0:
        return np.empty((0, latent_matrix.shape[1] + 5), dtype=np.float32)
    matrix_mean = patient_matrix.mean(axis=1, keepdims=True).astype(np.float32)
    matrix_std = patient_matrix.std(axis=1, keepdims=True).astype(np.float32)
    nonzero_fraction = (
        np.count_nonzero(patient_matrix, axis=1, keepdims=True) / max(patient_matrix.shape[1], 1)
    ).astype(np.float32)
    differential_projection = np.zeros((patient_matrix.shape[0], 1), dtype=np.float32)
    if differential.shape[0] == patient_matrix.shape[1] and differential.shape[0] > 0:
        normalized_differential = differential.astype(np.float32) / max(
            float(np.linalg.norm(differential.astype(np.float32))),
            1e-6,
        )
        differential_projection = (patient_matrix @ normalized_differential.reshape(-1, 1)).astype(np.float32)
    latent_norm = np.linalg.norm(latent_matrix.astype(np.float32), axis=1, keepdims=True).astype(np.float32)
    probability_column = probability_vector.reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [
            latent_matrix.astype(np.float32),
            probability_column,
            matrix_mean,
            matrix_std,
            nonzero_fraction,
            differential_projection / max(patient_matrix.shape[1], 1),
            latent_norm,
        ],
        axis=1,
    ).astype(np.float32)


def fold_patient_latents(
    config: dict,
    fold_name: str,
    split_role: str,
    disease_program: str | None = None,
    benchmark_name_override: str | None = None,
) -> dict[str, dict[str, object]]:
    split_frame = fold_sample_split(config, fold_name, disease_program, benchmark_name_override)
    split_frame = split_frame.loc[split_frame["split_role"] == split_role].copy()
    shared_dir = shared_latent_artifact_dir(config, fold_name, benchmark_name_override)
    model = load_shared_latent_model(shared_dir / "model_weights.pt")
    output: dict[str, dict[str, object]] = {}
    for modality in ["transcriptomics", "metabolomics"]:
        entity_index = pd.read_parquet(shared_dir / f"{modality}_entity_index.parquet")
        matrix, labels, metadata = assemble_dataset(config, split_frame, modality, entity_index)
        if matrix.shape[0] == 0:
            output[modality] = {
                "matrix": matrix,
                "labels": labels,
                "metadata": metadata,
                "entity_index": entity_index,
                "latent": np.empty((0, model.classifier.in_features), dtype=np.float32),
                "state_features": np.empty((0, model.classifier.in_features + 6), dtype=np.float32),
                "probabilities": np.empty((0,), dtype=np.float32),
                "differential": np.empty((entity_index.shape[0],), dtype=np.float32),
            }
            continue
        with torch.no_grad():
            logits, latent = model(modality, torch.tensor(matrix, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            latent_matrix = latent.cpu().numpy().astype(np.float32)
        differential = disease_differential_vector(matrix, labels)
        output[modality] = {
            "matrix": matrix,
            "labels": labels,
            "metadata": metadata,
            "entity_index": entity_index,
            "latent": latent_matrix,
            "state_features": patient_state_feature_matrix(latent_matrix, probabilities, matrix, differential),
            "probabilities": probabilities,
            "differential": differential,
        }
    return output


def disease_differential_vector(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty((matrix.shape[1],), dtype=np.float32)
    positive_mask = labels == 1
    negative_mask = labels == 0
    if positive_mask.sum() == 0 or negative_mask.sum() == 0:
        return np.zeros((matrix.shape[1],), dtype=np.float32)
    return (matrix[positive_mask].mean(axis=0) - matrix[negative_mask].mean(axis=0)).astype(np.float32)


def optional_tsv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str)


def foundation_registry_frames(config: dict) -> dict[str, pd.DataFrame]:
    return {
        "world_state": optional_tsv_frame(
            resolve_repo_path(config["data"]["world_state_ready_registry"]["local_root"])
            / config["data"]["world_state_ready_registry"]["relative_paths"]["world_state_catalog"]
        ),
        "simulation_teacher": optional_tsv_frame(
            resolve_repo_path(config["data"]["simulation_teacher_ready_registry"]["local_root"])
            / config["data"]["simulation_teacher_ready_registry"]["relative_paths"]["simulation_teacher_catalog"]
        ),
        "counterfactual_benchmark": optional_tsv_frame(
            resolve_repo_path(config["data"]["counterfactual_benchmark_ready_registry"]["local_root"])
            / config["data"]["counterfactual_benchmark_ready_registry"]["relative_paths"]["counterfactual_benchmark_catalog"]
        ),
        "candidate_validation": optional_tsv_frame(
            resolve_repo_path(config["validation"]["candidate_validation_registry"]["local_root"])
            / config["validation"]["candidate_validation_registry"]["relative_paths"]["candidate_validation_catalog"]
        ),
        "feedback": optional_tsv_frame(
            resolve_repo_path(config["data"]["feedback_ready_registry"]["local_root"])
            / config["data"]["feedback_ready_registry"]["relative_paths"]["feedback_event_catalog"]
        ),
    }


def world_state_support_vectors(metadata_frame: pd.DataFrame, world_state_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    sample_lookup: dict[str, dict[str, str]] = {}
    if not world_state_frame.empty:
        sample_lookup = {
            str(row["sample_uid"]): row
            for row in world_state_frame.to_dict(orient="records")
        }
    hidden_state_support = np.ones((metadata_frame.shape[0],), dtype=np.float32)
    missing_modality_count = np.zeros((metadata_frame.shape[0],), dtype=np.float32)
    for row_index, metadata_row in enumerate(metadata_frame.itertuples(index=False)):
        sample_uid = str(metadata_row.sample_uid)
        world_state_row = sample_lookup.get(sample_uid)
        if world_state_row is None:
            continue
        hidden_state_support[row_index] = 1.0 if str(world_state_row.get("hidden_state_support_flag", "yes")) == "yes" else 0.0
        missing_modality_count[row_index] = float(
            len([value for value in str(world_state_row.get("missing_modality_set", "")).split("|") if value])
        )
    return {
        "hidden_state_support": hidden_state_support,
        "missing_modality_count": missing_modality_count,
    }


def world_state_factor_targets(metadata_frame: pd.DataFrame, world_state_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    target_names = ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
    target_lookup: dict[str, dict[str, float]] = {}
    if not world_state_frame.empty:
        for row in world_state_frame.to_dict(orient="records"):
            payload = json.loads(str(row.get("factor_target_payload", "{}")) or "{}")
            target_lookup[str(row["sample_uid"])] = {
                target_name: float(payload.get(target_name, 0.0))
                for target_name in target_names
            }
    outputs = {
        target_name: np.zeros((metadata_frame.shape[0],), dtype=np.float32)
        for target_name in target_names
    }
    for row_index, metadata_row in enumerate(metadata_frame.itertuples(index=False)):
        sample_uid = str(metadata_row.sample_uid)
        payload = target_lookup.get(sample_uid, {})
        for target_name in target_names:
            outputs[target_name][row_index] = float(payload.get(target_name, 0.0))
    return outputs


def world_state_supervision_masks(metadata_frame: pd.DataFrame, world_state_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    target_names = ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
    mask_lookup: dict[str, dict[str, float]] = {}
    if not world_state_frame.empty:
        for row in world_state_frame.to_dict(orient="records"):
            payload = json.loads(str(row.get("factor_supervision_mask_payload", "{}")) or "{}")
            mask_lookup[str(row["sample_uid"])] = {
                target_name: float(payload.get(target_name, 0.0))
                for target_name in target_names
            }
    outputs = {
        target_name: np.zeros((metadata_frame.shape[0],), dtype=np.float32)
        for target_name in target_names
    }
    for row_index, metadata_row in enumerate(metadata_frame.itertuples(index=False)):
        sample_uid = str(metadata_row.sample_uid)
        payload = mask_lookup.get(sample_uid, {})
        for target_name in target_names:
            outputs[target_name][row_index] = float(payload.get(target_name, 0.0))
    return outputs


def simulation_teacher_factor_masks(
    regimen_frame: pd.DataFrame,
    simulation_teacher_frame: pd.DataFrame,
    disease_program: str,
) -> dict[str, np.ndarray]:
    factor_scope_lookup: dict[str, set[str]] = defaultdict(set)
    if not simulation_teacher_frame.empty:
        filtered_frame = simulation_teacher_frame.loc[
            simulation_teacher_frame["disease_program"].astype(str) == disease_program
        ].copy()
        for row in filtered_frame.to_dict(orient="records"):
            regimen_uid = str(row.get("regimen_uid", "")).strip()
            if not regimen_uid:
                continue
            factor_scope_lookup[regimen_uid].update(
                {value for value in str(row["factor_scope"]).split("|") if value}
            )
    masks = {
        "z_pk": np.zeros((regimen_frame.shape[0],), dtype=np.float32),
        "z_env": np.zeros((regimen_frame.shape[0],), dtype=np.float32),
        "z_ephys": np.zeros((regimen_frame.shape[0],), dtype=np.float32),
    }
    for regimen_index, regimen_row in enumerate(regimen_frame.to_dict(orient="records")):
        factor_scope = factor_scope_lookup.get(str(regimen_row["regimen_uid"]), set())
        if not factor_scope:
            factor_scope = {"z_pk", "z_env", "z_ephys"}
        for factor_name in masks:
            masks[factor_name][regimen_index] = 1.0 if factor_name in factor_scope else 0.0
    return masks


def counterfactual_pair_indices(
    regimen_frame: pd.DataFrame,
    counterfactual_benchmark_frame: pd.DataFrame,
    disease_program: str,
) -> list[tuple[int, int]]:
    if counterfactual_benchmark_frame.empty or regimen_frame.empty:
        return []
    filtered_frame = counterfactual_benchmark_frame.loc[
        counterfactual_benchmark_frame["disease_program"].astype(str) == disease_program
    ].copy()
    if filtered_frame.empty:
        return []
    regimen_index_lookup = {
        str(regimen_uid): index
        for index, regimen_uid in enumerate(regimen_frame["regimen_uid"].astype(str).tolist())
    }
    pair_indices: list[tuple[int, int]] = []
    for row in filtered_frame.to_dict(orient="records"):
        baseline_uid = str(row["baseline_regimen_uid"])
        counterfactual_uid = str(row["counterfactual_regimen_uid"])
        if baseline_uid not in regimen_index_lookup or counterfactual_uid not in regimen_index_lookup:
            continue
        pair = (regimen_index_lookup[baseline_uid], regimen_index_lookup[counterfactual_uid])
        if pair not in pair_indices:
            pair_indices.append(pair)
    return pair_indices


def target_program_features(target_program_rows: list[dict[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(target_program_rows).copy()
    if frame.empty:
        return frame, np.empty((0, 10), dtype=np.float32)
    mapped_feature_count = pd.to_numeric(frame["mapped_feature_count"], errors="coerce").fillna(0.0)
    coverage_fraction = pd.to_numeric(frame["coverage_fraction"], errors="coerce").fillna(0.0)
    target_evidence_score = pd.to_numeric(frame["target_evidence_score"], errors="coerce").fillna(0.0)
    target_class_codes = frame["target_class"].astype("category").cat.codes.astype(float)
    atlas_name_codes = frame["atlas_name"].astype("category").cat.codes.astype(float)
    source_repository_codes = frame["source_repository"].astype("category").cat.codes.astype(float)
    feature_matrix = np.column_stack(
        [
            mapped_feature_count.to_numpy(dtype=np.float32),
            coverage_fraction.to_numpy(dtype=np.float32),
            target_evidence_score.to_numpy(dtype=np.float32),
            target_class_codes.to_numpy(dtype=np.float32),
            atlas_name_codes.to_numpy(dtype=np.float32),
            source_repository_codes.to_numpy(dtype=np.float32),
            frame["atlas_entity_uid"].astype(str).ne("nan").astype(float).to_numpy(dtype=np.float32),
            frame["program_id"].astype(str).ne("nan").astype(float).to_numpy(dtype=np.float32),
            (mapped_feature_count * coverage_fraction).to_numpy(dtype=np.float32),
            frame["source_repository"].astype(str).eq("atlas+drug_target").astype(float).to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)
    return frame, feature_matrix


def regimen_features(regimen_rows: list[dict[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(regimen_rows).copy()
    if frame.empty:
        return frame, np.empty((0, 23), dtype=np.float32)
    route_codes = frame["route"].astype("category").cat.codes.astype(float).to_numpy()
    schedule_codes = frame["schedule"].astype("category").cat.codes.astype(float).to_numpy()
    affinity_values = pd.to_numeric(frame["aggregate_affinity_score"].mask(frame["aggregate_affinity_score"].astype(str) == "", np.nan), errors="coerce")
    affinity_values = affinity_values.fillna(0.0)
    target_count = frame["target_uid_set"].astype(str).str.count("\\|").add(1).astype(float)
    compound_count = frame["compound_uid_set"].astype(str).str.count("\\|").add(1).astype(float)
    scaffold_codes = frame["regimen_scaffold_set"].astype("category").cat.codes.astype(float)
    feature_matrix = np.column_stack(
        [
            frame["aggregate_target_evidence_score"].astype(float).to_numpy(),
            affinity_values.to_numpy(),
            frame["clue_reversal_proxy_score"].astype(float).to_numpy(),
            frame["pk_feasibility_score"].astype(float).to_numpy(),
            frame["dose_multiplier"].astype(float).to_numpy(),
            frame["regimen_size"].astype(float).to_numpy(),
            route_codes.astype(np.float32),
            schedule_codes.astype(np.float32),
            target_count.to_numpy(dtype=np.float32),
            (frame["route"] == "intrathecal").astype(float).to_numpy(),
            (frame["schedule"] == "weekly").astype(float).to_numpy(),
            compound_count.to_numpy(dtype=np.float32),
            scaffold_codes.to_numpy(dtype=np.float32),
            frame["route"].astype(str).eq("oral").astype(float).to_numpy(dtype=np.float32),
            frame["route"].astype(str).eq("intravenous").astype(float).to_numpy(dtype=np.float32),
            frame["schedule"].astype(str).eq("daily").astype(float).to_numpy(dtype=np.float32),
            frame["schedule"].astype(str).eq("twice_daily").astype(float).to_numpy(dtype=np.float32),
            frame["dose_level_id"].astype(str).eq("dose_025").astype(float).to_numpy(dtype=np.float32),
            frame["dose_level_id"].astype(str).eq("dose_050").astype(float).to_numpy(dtype=np.float32),
            frame["dose_level_id"].astype(str).eq("dose_100").astype(float).to_numpy(dtype=np.float32),
            frame["dose_level_id"].astype(str).eq("dose_200").astype(float).to_numpy(dtype=np.float32),
            frame["selection_reason"].astype(str).eq("ranked_selection").astype(float).to_numpy(dtype=np.float32),
            frame["selection_reason"].astype(str).eq("pairwise_ranked_selection").astype(float).to_numpy(dtype=np.float32),
            frame["selection_reason"].astype(str).eq("case_study_seed").astype(float).to_numpy(dtype=np.float32),
            frame["note"].astype(str).eq("single_compound_regimen").astype(float).to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)
    return frame, feature_matrix


def patient_target_labels(
    patient_matrix: np.ndarray,
    differential: np.ndarray,
    entity_lookup: dict[str, int],
    target_program_frame: pd.DataFrame,
) -> np.ndarray:
    label_matrix = np.zeros((patient_matrix.shape[0], target_program_frame.shape[0]), dtype=np.float32)
    for column_index, row in enumerate(target_program_frame.itertuples(index=False)):
        entity_id = str(row.canonical_entity_id)
        if entity_id not in entity_lookup:
            continue
        feature_index = entity_lookup[entity_id]
        direction = float(np.sign(differential[feature_index])) if differential.shape[0] > feature_index else 0.0
        signed_activity = patient_matrix[:, feature_index] * (direction if direction != 0.0 else 1.0)
        label_matrix[:, column_index] = np.maximum(0.0, signed_activity) * float(row.target_evidence_score)
    return label_matrix


def patient_regimen_labels(
    patient_matrix: np.ndarray,
    differential: np.ndarray,
    entity_lookup: dict[str, int],
    regimen_frame: pd.DataFrame,
    target_uid_to_entity: dict[str, str],
) -> np.ndarray:
    label_matrix = np.zeros((patient_matrix.shape[0], regimen_frame.shape[0]), dtype=np.float32)
    for column_index, row in enumerate(regimen_frame.itertuples(index=False)):
        target_entities = [
            target_uid_to_entity[target_uid]
            for target_uid in str(row.target_uid_set).split("|")
            if target_uid and target_uid in target_uid_to_entity and target_uid_to_entity[target_uid] in entity_lookup
        ]
        if not target_entities:
            continue
        activities = []
        for entity_id in target_entities:
            feature_index = entity_lookup[entity_id]
            direction = float(np.sign(differential[feature_index])) if differential.shape[0] > feature_index else 0.0
            signed_activity = patient_matrix[:, feature_index] * (direction if direction != 0.0 else 1.0)
            activities.append(np.maximum(0.0, signed_activity))
        stacked = np.stack(activities, axis=1)
        label_matrix[:, column_index] = (
            stacked.mean(axis=1)
            * float(row.aggregate_target_evidence_score)
            + float(row.clue_reversal_proxy_score)
            + float(row.pk_feasibility_score)
        )
    return label_matrix


def target_uid_to_entity_lookup(target_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["target_uid"]: row["canonical_entity_id"] for row in target_rows}


def top_columns(matrix: np.ndarray, count: int) -> np.ndarray:
    if matrix.shape[1] <= count:
        return np.arange(matrix.shape[1], dtype=np.int64)
    mean_scores = matrix.mean(axis=0)
    return np.argsort(-mean_scores)[:count]


def training_manifest_path(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def production_patient_latents(config: dict) -> dict[str, dict[str, dict[str, object]]]:
    cohort_root = resolve_repo_path(config["benchmark_registry"]["local_root"])
    cohort_rows = read_tsv(cohort_root / config["benchmark_registry"]["relative_paths"]["cohort_catalog"])
    active_program = str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    primary_studies = sorted(
        row["study_uid"]
        for row in cohort_rows
        if row.get("disease_program", active_program) == active_program
        and row["cohort_role"] == "primary_train_eval"
        and row["inclusion_flag"] == "yes"
    )
    alignment_lookup = alignment_by_study(config)
    diagnosis_aliases = config["cohort_ready_registry"]["diagnosis_aliases"]
    positive_labels = set(config["benchmark_registry"]["positive_labels"])
    negative_labels = set(config["benchmark_registry"]["negative_labels"])
    random_seed = int(config["benchmark_registry"]["logistic_regression"]["random_seed"])
    validation_fraction = float(config["benchmark_registry"]["validation_fraction"])
    minimum_validation_groups_per_label = int(config["benchmark_registry"]["minimum_validation_groups_per_label"])
    production_dir = shared_latent_production_dir(config)
    model = load_shared_latent_model(production_dir / "model_weights.pt")

    sample_rows: list[dict[str, object]] = []
    for study_uid in primary_studies:
        alignment_row = alignment_lookup[study_uid]
        sample_index, _, _, _ = load_alignment_bundle_artifacts(alignment_row)
        sample_index = sample_index.copy()
        sample_index["study_uid"] = study_uid
        sample_index["source_key"] = alignment_row["source_key"]
        sample_index["source_accession"] = alignment_row["source_accession"]
        sample_index["normalized_diagnosis_label"] = sample_index["sample_diagnosis_label"].astype(str).apply(
            lambda value: normalize_diagnosis_label(value, diagnosis_aliases)
        )
        sample_index["binary_label"] = sample_index["normalized_diagnosis_label"].apply(
            lambda value: binary_label(value, positive_labels, negative_labels)
        )
        sample_index["group_id"] = sample_index.apply(choose_group_identifier, axis=1)
        sample_rows.extend(sample_index.to_dict("records"))
    full_frame = pd.DataFrame(sample_rows)
    full_frame = full_frame.loc[full_frame["binary_label"].notna()].copy()
    assignments = grouped_validation_assignments(
        full_frame,
        validation_fraction,
        minimum_validation_groups_per_label,
        random_seed,
    )
    full_frame["split_role"] = full_frame["sample_uid"].map(assignments)

    output: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for modality in ["transcriptomics", "metabolomics"]:
        entity_index = pd.read_parquet(production_dir / f"{modality}_entity_index.parquet")
        for split_role in ["train", "validation"]:
            split_frame = full_frame.loc[full_frame["split_role"] == split_role].copy()
            matrix, labels, metadata = assemble_dataset(config, split_frame, modality, entity_index)
            if matrix.shape[0] == 0:
                output[modality][split_role] = {
                    "matrix": matrix,
                    "labels": labels,
                    "metadata": metadata,
                    "entity_index": entity_index,
                    "latent": np.empty((0, model.classifier.in_features), dtype=np.float32),
                    "state_features": np.empty((0, model.classifier.in_features + 6), dtype=np.float32),
                    "probabilities": np.empty((0,), dtype=np.float32),
                    "differential": np.empty((entity_index.shape[0],), dtype=np.float32),
                }
                continue
            with torch.no_grad():
                logits, latent = model(modality, torch.tensor(matrix, dtype=torch.float32))
                probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                latent_matrix = latent.cpu().numpy().astype(np.float32)
            output[modality][split_role] = {
                "matrix": matrix,
                "labels": labels,
                "metadata": metadata,
                "entity_index": entity_index,
                "latent": latent_matrix,
                "state_features": patient_state_feature_matrix(
                    latent_matrix,
                    probabilities,
                    matrix,
                    disease_differential_vector(matrix, labels),
                ),
                "probabilities": probabilities,
                "differential": disease_differential_vector(matrix, labels),
            }
    return output
