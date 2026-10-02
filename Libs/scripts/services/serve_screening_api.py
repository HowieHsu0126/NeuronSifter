from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.models.networks.shared_latent_dual_encoder import SharedLatentDualEncoder
from Libs.utils.data.platform_contracts import stable_uid, validation_rerank_frame
from Libs.utils.evaluation.regimen_acquisition import build_acquisition_frame, select_acquisition_batch
from Libs.utils.evaluation.active_learning_loop import (
    active_learning_config,
    build_active_candidate_pool,
    ingest_active_learning_results,
    round_path,
    score_acquisition_tasks,
    select_request_batch,
    write_frame,
    write_round_report,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_json, write_tsv
from Libs.utils.model.neuronsifter_scale_registry import registry as neuronsifter_scale_registry
from Libs.utils.model.neuronsifter_scale_registry import scale_route_metadata
from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config
from Libs.utils.evaluation.screening_datasets import patient_state_feature_matrix, regimen_features, target_program_features
from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the AD screening system bundle with a lightweight JSON API.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--bundle-dir", default="")
    return parser.parse_args()


def load_shared_latent_model(checkpoint_path: Path) -> SharedLatentDualEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SharedLatentDualEncoder(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_screening_model(checkpoint_path: Path, config: dict) -> ADScreeningSystem:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = dict(checkpoint["model_kwargs"])
    model_kwargs.update(
        {
            key: value
            for key, value in screening_model_kwargs_from_config(config).items()
            if key not in model_kwargs or model_kwargs[key] in (None, [], {})
        }
    )
    model = ADScreeningSystem(**model_kwargs)
    # A partially loaded pointwise refiner is not the conditional method.
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_optional_catalog(bundle_dir: Path, filename: str, columns: list[str]) -> pd.DataFrame:
    path = bundle_dir / filename
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, sep="\t", dtype=str)


def load_bundle(bundle_dir: Path, config: dict | None = None) -> dict[str, object]:
    effective_config = load_config("Libs/configs/config_stack.yaml") if config is None else config
    screening_model = load_screening_model(bundle_dir / "model_weights.pt", effective_config)
    shared_model = load_shared_latent_model(bundle_dir / "base_shared_latent_model_weights.pt")
    target_frame = pd.read_csv(bundle_dir / "target_catalog.tsv", sep="\t", dtype=str)
    target_program_catalog = pd.read_csv(bundle_dir / "target_program_catalog.tsv", sep="\t", dtype=str)
    compound_frame = pd.read_csv(bundle_dir / "compound_catalog.tsv", sep="\t", dtype=str)
    regimen_frame = pd.read_csv(bundle_dir / "regimen_catalog.tsv", sep="\t", dtype=str)
    sample_observation_path = bundle_dir / "sample_observation_catalog.tsv"
    if not sample_observation_path.exists():
        raise FileNotFoundError(f"Missing sample observation catalog in serving bundle: {sample_observation_path}")
    sample_observation_frame = pd.read_csv(sample_observation_path, sep="\t", dtype=str)
    measurement_semantics_frame = load_optional_catalog(
        bundle_dir,
        "measurement_semantics_catalog.tsv",
        effective_config["schemas"].get("measurement_semantics_fields", []),
    )
    intervention_program_frame = pd.read_csv(bundle_dir / "intervention_program_catalog.tsv", sep="\t", dtype=str)
    candidate_validation_frame = pd.read_csv(bundle_dir / "candidate_validation_catalog.tsv", sep="\t", dtype=str)
    region_frame = pd.read_csv(bundle_dir / "region_catalog.tsv", sep="\t", dtype=str)
    species_frame = pd.read_csv(bundle_dir / "species_catalog.tsv", sep="\t", dtype=str)
    case_validation_frame = pd.read_csv(bundle_dir / "case_validation_catalog.tsv", sep="\t", dtype=str)
    region_prior_frame = pd.read_csv(bundle_dir / "region_prior_catalog.tsv", sep="\t", dtype=str)
    local_response_head_frame = pd.read_csv(bundle_dir / "local_response_head_catalog.tsv", sep="\t", dtype=str)
    world_state_frame = load_optional_catalog(
        bundle_dir,
        "world_state_catalog.tsv",
        [
            "world_state_uid",
            "observation_uid",
            "study_uid",
            "sample_uid",
            "source_accession",
            "disease_program",
            "disease_family",
            "observed_modality_set",
            "missing_modality_set",
            "factor_slice_manifest",
        ],
    )
    crossmodal_anchor_frame = load_optional_catalog(
        bundle_dir,
        "crossmodal_anchor_catalog.tsv",
        [
            "crossmodal_anchor_uid",
            "disease_family",
            "anchor_source",
            "anchor_scope",
            "supervision_role",
            "anchor_status",
        ],
    )
    simulation_teacher_frame = load_optional_catalog(
        bundle_dir,
        "simulation_teacher_catalog.tsv",
        [
            "simulation_teacher_uid",
            "disease_program",
            "disease_family",
            "teacher_family",
            "source_simulation_uid",
            "factor_scope",
            "trajectory_store_path",
        ],
    )
    counterfactual_benchmark_frame = load_optional_catalog(
        bundle_dir,
        "counterfactual_benchmark_catalog.tsv",
        [
            "counterfactual_benchmark_uid",
            "disease_program",
            "disease_family",
            "benchmark_name",
            "baseline_regimen_uid",
            "counterfactual_regimen_uid",
            "target_metric",
        ],
    )
    feedback_event_frame = load_optional_catalog(
        bundle_dir,
        "feedback_event_catalog.tsv",
        [
            "feedback_event_uid",
            "disease_program",
            "disease_family",
            "sample_uid",
            "intervention_program_uid",
            "regimen_uid",
            "update_eligibility_flag",
            "update_decision",
            "event_timestamp",
        ],
    )
    external_adapter_frame = load_optional_catalog(
        bundle_dir,
        "external_adapter_catalog.tsv",
        effective_config["schemas"].get("external_adapter_fields", []),
    )
    preprocessing_manifest = json.loads((bundle_dir / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    model_card = json.loads((bundle_dir / "model_card.json").read_text(encoding="utf-8"))
    target_program_frame, target_feature_matrix = target_program_features(target_program_catalog.to_dict("records"))
    regimen_feature_frame, regimen_feature_matrix = regimen_features(regimen_frame.to_dict("records"))
    transcriptomics_entity_index = pd.read_parquet(bundle_dir / "transcriptomics_entity_index.parquet")
    metabolomics_entity_index = pd.read_parquet(bundle_dir / "metabolomics_entity_index.parquet")
    lookup_by_modality = {
        "transcriptomics": {str(entity_id): index for index, entity_id in enumerate(transcriptomics_entity_index["canonical_entity_id"].astype(str).tolist())},
        "metabolomics": {str(entity_id): index for index, entity_id in enumerate(metabolomics_entity_index["canonical_entity_id"].astype(str).tolist())},
    }
    return {
        "screening_model": screening_model,
        "shared_model": shared_model,
        "target_catalog": target_frame,
        "target_program_catalog": target_program_catalog,
        "compound_catalog": compound_frame,
        "regimen_catalog": regimen_frame,
        "sample_observation_catalog": sample_observation_frame,
        "measurement_semantics_catalog": measurement_semantics_frame,
        "intervention_program_frame": intervention_program_frame,
        "candidate_validation_frame": candidate_validation_frame,
        "region_catalog": region_frame,
        "species_catalog": species_frame,
        "case_validation_catalog": case_validation_frame,
        "region_prior_catalog": region_prior_frame,
        "local_response_head_catalog": local_response_head_frame,
        "world_state_catalog": world_state_frame,
        "crossmodal_anchor_catalog": crossmodal_anchor_frame,
        "simulation_teacher_catalog": simulation_teacher_frame,
        "counterfactual_benchmark_catalog": counterfactual_benchmark_frame,
        "feedback_event_catalog": feedback_event_frame,
        "external_adapter_catalog": external_adapter_frame,
        "target_program_frame": target_program_frame,
        "target_feature_matrix": torch.tensor(target_feature_matrix, dtype=torch.float32),
        "regimen_feature_frame": regimen_feature_frame,
        "regimen_feature_matrix": torch.tensor(regimen_feature_matrix, dtype=torch.float32),
        "regimen_index_by_uid": {
            str(regimen_uid): index
            for index, regimen_uid in enumerate(regimen_frame["regimen_uid"].astype(str).tolist())
        },
        "intervention_index_by_uid": {
            str(intervention_uid): index
            for index, intervention_uid in enumerate(intervention_program_frame["intervention_program_uid"].astype(str).tolist())
        },
        "lookup_by_modality": lookup_by_modality,
        "preprocessing_manifest": preprocessing_manifest,
        "model_card": model_card,
        "bundle_dir": bundle_dir,
    }


def response_candidate_count(response_payload: dict[str, object]) -> int:
    if "candidate_count" in response_payload:
        return int(response_payload["candidate_count"])
    for row_key in ["reranked_candidates", "planned_interventions", "intervention_program_ranking", "selected_regimens"]:
        if row_key in response_payload and isinstance(response_payload[row_key], list):
            return len(response_payload[row_key])
    return 0


def response_has_decision(response_payload: dict[str, object], decision_label: str) -> bool:
    decision_summary = response_payload.get("decision_summary", {})
    if isinstance(decision_summary, dict) and decision_label in decision_summary:
        return int(decision_summary[decision_label]) > 0
    rows = []
    for row_key in ["reranked_candidates", "planned_interventions"]:
        if row_key in response_payload and isinstance(response_payload[row_key], list):
            rows.extend(response_payload[row_key])
    return any(str(row.get("decision", "")) == decision_label for row in rows if isinstance(row, dict))


def route_profile_for_response(config: dict, sample_payload: dict[str, object], response_payload: dict[str, object]) -> tuple[str, str]:
    registry_config = neuronsifter_scale_registry(config)
    routing_policy = registry_config["routing_policy"]
    if "inference_profile" in sample_payload:
        return str(sample_payload["inference_profile"]), "client_requested_profile"
    if bool(sample_payload.get("publication_case", False)):
        return str(routing_policy["publication_case_profile"]), "xl_publication_audit"
    decision_labels = config["screening_pipeline"]["screening_benchmark"]["decision_gate"]["decision_labels"]
    if response_has_decision(response_payload, str(decision_labels["active_learning"])):
        return str(registry_config["default_inference_profile"]), "active_learning_escalation"
    if response_has_decision(response_payload, str(decision_labels["external_validation"])):
        return str(registry_config["default_inference_profile"]), "external_validation_escalation"
    candidate_count = response_candidate_count(response_payload)
    if candidate_count >= int(routing_policy["default_candidate_pool_threshold_for_student"]):
        return "student_prescreen", "student_prescreen_completed"
    return str(registry_config["default_inference_profile"]), "no_escalation"


def serving_checkpoint_id(bundle: dict[str, object], profile_used: str) -> str:
    model_card = dict(bundle["model_card"])
    model_version = str(model_card.get("model_version", "unversioned_model"))
    return str(model_card.get("checkpoint_id", f"{model_version}:{profile_used}"))


def scale_aware_response(
    bundle: dict[str, object],
    config: dict,
    sample_payload: dict[str, object],
    response_payload: dict[str, object],
) -> dict[str, object]:
    if "profile_used" in response_payload:
        return response_payload
    profile_used, escalation_status = route_profile_for_response(config, sample_payload, response_payload)
    routing_policy = neuronsifter_scale_registry(config)["routing_policy"]
    default_disagreement = float(routing_policy["disagreement_default_value"])
    metadata = scale_route_metadata(
        config,
        profile_used,
        serving_checkpoint_id(bundle, profile_used),
        escalation_status,
        float(sample_payload.get("student_large_disagreement", default_disagreement)),
        float(sample_payload.get("large_xl_disagreement", default_disagreement)),
    )
    return {**response_payload, **metadata}


def patient_latent_from_payload(bundle: dict[str, object], sample_payload: dict[str, object]) -> tuple[str, torch.Tensor, float]:
    modality = str(sample_payload["modality"])
    feature_values = sample_payload["feature_values"]
    if modality not in bundle["lookup_by_modality"]:
        raise ValueError(f"Unsupported modality: {modality}")
    lookup = bundle["lookup_by_modality"][modality]
    vector = torch.zeros((1, len(lookup)), dtype=torch.float32)
    unknown_features = []
    for feature_id, value in feature_values.items():
        if str(feature_id) not in lookup:
            unknown_features.append(str(feature_id))
            continue
        vector[0, lookup[str(feature_id)]] = float(value)
    if unknown_features:
        raise ValueError(f"Unknown features for {modality}: {unknown_features[:10]}")
    with torch.no_grad():
        logits, latent = bundle["shared_model"](modality, vector)
        probability = float(torch.sigmoid(logits)[0].cpu().item())
    state_features = patient_state_feature_matrix(
        latent.detach().cpu().numpy().astype(np.float32),
        np.asarray([probability], dtype=np.float32),
        vector.detach().cpu().numpy().astype(np.float32),
        np.zeros((vector.shape[1],), dtype=np.float32),
    )
    return modality, torch.tensor(state_features, dtype=torch.float32), probability


def species_index(species_labels: list[str], *candidate_names: str) -> int:
    for candidate_name in candidate_names:
        if candidate_name in species_labels:
            return species_labels.index(candidate_name)
    raise ValueError(f"None of the candidate species names were found: {candidate_names}")


def infer_state(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    modality, latent, probability = patient_latent_from_payload(bundle, sample_payload)
    disease_logit, encoded_patient = bundle["screening_model"].infer_state(latent)
    return {
        "modality": modality,
        "disease_probability": probability,
        "screening_state_probability": float(torch.sigmoid(disease_logit)[0].cpu().item()),
        "latent_embedding": encoded_patient[0].detach().cpu().tolist(),
    }


def infer_posterior(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    posterior = infer_state(bundle, sample_payload)
    posterior["uncertainty_summary"] = {
        "status": "single_model_only",
        "disease_posterior_uncertainty": 0.0,
        "target_ranking_uncertainty": 0.0,
        "regimen_ranking_uncertainty": 0.0,
        "trajectory_summary_uncertainty": 0.0,
    }
    posterior["posterior_contract"] = {
        "endpoint": "infer_posterior",
        "supported_modalities": bundle["preprocessing_manifest"]["supported_modalities"],
        "observation_contract": "sample_observation_catalog",
    }
    return posterior


def disease_family_name(disease_program: str) -> str:
    normalized = disease_program.strip().lower()
    if normalized.startswith("ad_"):
        return "AD"
    if normalized.startswith("pd_"):
        return "PD"
    if normalized.startswith("ftd_"):
        return "FTD"
    if normalized.startswith("als_"):
        return "ALS"
    return "pan_neuro"


def modality_mask(bundle: dict[str, object], observed_modality: str) -> dict[str, bool]:
    modality_pool = bundle["preprocessing_manifest"].get(
        "foundation_target_modalities",
        bundle["preprocessing_manifest"].get("supported_modalities", []),
    )
    return {
        str(modality_name): str(modality_name) == observed_modality
        for modality_name in modality_pool
    }


def factor_strengths(world_state_payload: dict[str, torch.Tensor]) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]:
        factor_tensor = world_state_payload[factor_name][0].detach().cpu()
        strengths[factor_name] = float(factor_tensor.abs().mean().item())
    return strengths


def measurement_context(bundle: dict[str, object], sample_payload: dict[str, object], observed_modality: str) -> dict[str, object]:
    semantics_frame = bundle.get("measurement_semantics_catalog", pd.DataFrame())
    sample_frame = bundle.get("sample_observation_catalog", pd.DataFrame())
    selected_semantics_uid = str(sample_payload.get("semantics_uid", "")).strip()
    if not selected_semantics_uid and "sample_uid" in sample_payload and isinstance(sample_frame, pd.DataFrame) and not sample_frame.empty:
        matches = sample_frame.loc[sample_frame["sample_uid"].astype(str) == str(sample_payload["sample_uid"])]
        if not matches.empty and "semantics_uid" in matches.columns:
            selected_semantics_uid = str(matches.iloc[0]["semantics_uid"])
    selected_row: dict[str, object] = {}
    if isinstance(semantics_frame, pd.DataFrame) and not semantics_frame.empty:
        candidate_frame = semantics_frame
        if selected_semantics_uid:
            candidate_frame = candidate_frame.loc[candidate_frame["semantics_uid"].astype(str) == selected_semantics_uid]
        if candidate_frame.empty:
            candidate_frame = semantics_frame.loc[semantics_frame["modality"].astype(str) == observed_modality]
        if not candidate_frame.empty:
            selected_row = candidate_frame.iloc[0].to_dict()
            selected_semantics_uid = str(selected_row["semantics_uid"])
    proxy_flag = str(selected_row.get("proxy_unit_flag", "yes")) == "yes"
    return {
        "semantics_uid": selected_semantics_uid,
        "unit_context": {
            "raw_unit": str(selected_row.get("raw_unit", "")),
            "canonical_unit": str(selected_row.get("canonical_unit", "")),
            "inverse_transform_artifact_path": str(selected_row.get("inverse_transform_artifact_path", "")),
        },
        "standardized_value": sample_payload.get("standardized_value", {}),
        "original_unit_value": sample_payload.get("original_unit_value", {}),
        "proxy_flag": proxy_flag,
        "inverse_transform_status": "feature_level_values_required" if not sample_payload.get("original_unit_value", {}) else "provided_by_client_payload",
    }


def calibration_status_for_family(frame: pd.DataFrame, disease_family: str) -> str:
    if frame.empty or "calibration_status" not in frame.columns:
        return "not_available"
    matching = frame.loc[frame["disease_family"].astype(str) == disease_family]
    if matching.empty:
        return "not_available"
    statuses = set(matching["calibration_status"].astype(str).tolist())
    if "external_calibrated" in statuses:
        return "external_calibrated"
    if "repository_protocol_validated" in statuses:
        return "repository_protocol_validated"
    if "calibration_pending" in statuses:
        return "calibration_pending"
    return sorted(statuses)[0]


def infer_world_state(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    modality, latent, disease_probability = patient_latent_from_payload(bundle, sample_payload)
    disease_program = str(sample_payload.get("disease_program", bundle["preprocessing_manifest"]["active_disease_program"]))
    disease_family = disease_family_name(disease_program)
    measurement_payload = measurement_context(bundle, sample_payload, modality)
    world_state_payload = bundle["screening_model"].infer_world_state(latent)
    factor_payload = {
        factor_name: world_state_payload[factor_name][0].detach().cpu().tolist()
        for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
    }
    hidden_parameter_posterior = world_state_payload["hidden_parameter_posterior"][0].detach().cpu().tolist()
    matching_anchors = bundle["crossmodal_anchor_catalog"]
    if not matching_anchors.empty:
        matching_anchors = matching_anchors.loc[matching_anchors["disease_family"].astype(str) == disease_family].copy()
    matching_teachers = bundle["simulation_teacher_catalog"]
    if not matching_teachers.empty:
        matching_teachers = matching_teachers.loc[matching_teachers["disease_family"].astype(str) == disease_family].copy()
    return {
        "modality": modality,
        "disease_probability": disease_probability,
        "disease_program": disease_program,
        "disease_family": disease_family,
        "world_state_version": "foundation_twin_world_state_v1",
        "modality_mask": modality_mask(bundle, modality),
        "world_state_factors": factor_payload,
        "hidden_parameter_posterior": hidden_parameter_posterior,
        "factor_strengths": factor_strengths(world_state_payload),
        "hidden_state_support_flag": True,
        "cross_scene_compatibility_flag": True,
        "crossmodal_anchor_count": int(matching_anchors.shape[0]),
        "simulation_teacher_count": int(matching_teachers.shape[0]),
        "semantics_uid": measurement_payload["semantics_uid"],
        "unit_context": measurement_payload["unit_context"],
        "standardized_value": measurement_payload["standardized_value"],
        "original_unit_value": measurement_payload["original_unit_value"],
        "proxy_flag": measurement_payload["proxy_flag"],
        "calibration_status": calibration_status_for_family(bundle["simulation_teacher_catalog"], disease_family),
        "inverse_transform_status": measurement_payload["inverse_transform_status"],
    }


def regimen_tri_context(bundle: dict[str, object], sample_payload: dict[str, object], regimen_count: int) -> dict[str, torch.Tensor]:
    modality = str(sample_payload["modality"])
    feature_values = sample_payload["feature_values"]
    lookup = bundle["lookup_by_modality"][modality]
    coverage_value = float(len(feature_values) / max(len(lookup), 1))
    return {
        "coverage": torch.full((1, regimen_count), coverage_value, dtype=torch.float32),
        "ood_score": torch.full((1, regimen_count), 0.5, dtype=torch.float32),
    }


def candidate_intervention_frame(bundle: dict[str, object], sample_payload: dict[str, object]) -> pd.DataFrame:
    frame = bundle["intervention_program_frame"].copy()
    intervention_program_uids = [str(value) for value in sample_payload.get("candidate_intervention_program_uids", [])]
    regimen_uids = [str(value) for value in sample_payload.get("candidate_regimen_uids", [])]
    if intervention_program_uids:
        missing = [value for value in intervention_program_uids if value not in bundle["intervention_index_by_uid"]]
        if missing:
            raise ValueError(f"Unknown candidate_intervention_program_uids: {missing[:10]}")
        frame = frame.loc[frame["intervention_program_uid"].astype(str).isin(intervention_program_uids)].copy()
    if regimen_uids:
        missing = [value for value in regimen_uids if value not in bundle["regimen_index_by_uid"]]
        if missing:
            raise ValueError(f"Unknown candidate_regimen_uids: {missing[:10]}")
        frame = frame.loc[frame["regimen_uid"].astype(str).isin(regimen_uids)].copy()
    if frame.empty:
        raise ValueError("No intervention programs remain after candidate filtering.")
    return frame.reset_index(drop=True)


def conditioned_inputs_from_payload(bundle: dict, payload: dict, regimen_uids: list[str]) -> dict | None:
    """Bind supplied exogenous programs to the exact candidate order."""
    model = bundle["screening_model"]
    if model.field_rollout_operator.quantized_mddm_rollout_operator is None:
        return None
    programs = payload.get("occupancy_programs")
    if not isinstance(programs, dict):
        raise ValueError("Conditional rollout requires registered occupancy_programs keyed by regimen_uid.")
    rows = []
    for uid in regimen_uids:
        row = programs.get(uid)
        if not isinstance(row, dict) or row.get("origin") != "exogenous_structured_action_pkpd" or not row.get("occupancy_program_ref"):
            raise ValueError(f"Missing independent occupancy provenance for {uid}.")
        if tuple(row.get("occupancy_channel_ids", ())) != model.field_rollout_operator.quantized_mddm_rollout_operator.occupancy_channel_ids:
            raise ValueError(f"Occupancy target channels differ from the checkpoint contract for {uid}.")
        if np.asarray(row["occupancy_support"]).dtype != np.bool_:
            raise ValueError("Occupancy support must contain booleans, not numeric proxy flags.")
        rows.append(row)
    result = {
        "occupancy_program": torch.tensor(np.asarray([row["occupancy_program"] for row in rows]), dtype=torch.float32)[None],
        "occupancy_support": torch.tensor(np.asarray([row["occupancy_support"] for row in rows]), dtype=torch.bool)[None],
        "time_hours": torch.tensor([row["time_hours"] for row in rows], dtype=torch.float32)[None],
        "sample_keys": torch.tensor([row["conditioning_sample_key"] for row in rows], dtype=torch.int64)[None],
    }
    return result


def score_intervention_candidates(
    bundle: dict[str, object],
    sample_payload: dict[str, object],
) -> tuple[torch.Tensor, pd.DataFrame, np.ndarray, np.ndarray]:
    _, patient_state, _ = patient_latent_from_payload(bundle, sample_payload)
    candidate_frame = candidate_intervention_frame(bundle, sample_payload)
    candidate_indices = [
        bundle["regimen_index_by_uid"][str(regimen_uid)]
        for regimen_uid in candidate_frame["regimen_uid"].astype(str).tolist()
    ]
    candidate_feature_matrix = bundle["regimen_feature_matrix"][candidate_indices]
    tri_context = regimen_tri_context(bundle, sample_payload, candidate_feature_matrix.shape[0])
    model_output = bundle["screening_model"].score_regimens(
        patient_state,
        candidate_feature_matrix,
        tri_context,
        conditioned_inputs=conditioned_inputs_from_payload(bundle, sample_payload, candidate_frame["regimen_uid"].astype(str).tolist()),
    )
    predicted_scores = model_output["regimen_scores"][0].detach().cpu().numpy().astype(np.float64)
    tri_scores = model_output["tri_score"][0].detach().cpu().numpy().astype(np.float64)
    ranking_stability = model_output.get("ranking_stability")
    if isinstance(ranking_stability, torch.Tensor):
        ranking_stability_values = ranking_stability[0].detach().cpu().numpy().astype(np.float64)
    else:
        ranking_stability_values = np.ones_like(predicted_scores, dtype=np.float64)
    trajectory_std = model_output.get("field_trajectory_std")
    if isinstance(trajectory_std, torch.Tensor):
        rollout_uncertainty = trajectory_std[0].detach().cpu().numpy().astype(np.float64).mean(axis=(1, 2, 3, 4))
    else:
        rollout_uncertainty = np.zeros_like(predicted_scores, dtype=np.float64)
    candidate_frame = candidate_frame.copy()
    candidate_frame["predicted_score"] = predicted_scores
    candidate_frame["tri_score"] = tri_scores
    candidate_frame["ranking_stability"] = ranking_stability_values
    candidate_frame["rollout_uncertainty"] = rollout_uncertainty
    candidate_frame["uncertainty_status"] = str(model_output.get("uncertainty_status", "not_available"))
    tri_components = model_output["tri_components"]
    for component_name in TRI_COMPONENT_NAMES:
        candidate_frame[f"tri_{component_name}"] = tri_components[component_name][0].detach().cpu().numpy().astype(np.float64)
    return patient_state, candidate_frame, predicted_scores, tri_scores


def rank_targets(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    _, latent, _ = patient_latent_from_payload(bundle, sample_payload)
    target_scores, _ = bundle["screening_model"].score_targets(latent, bundle["target_feature_matrix"])
    frame = bundle["target_program_frame"].copy()
    frame["predicted_score"] = target_scores[0].detach().cpu().numpy()
    top_frame = frame.sort_values(by="predicted_score", ascending=False).head(20)
    return {
        "target_set_ranking": top_frame[["canonical_entity_label", "predicted_score", "target_evidence_score", "program_id"]].to_dict(orient="records"),
    }


def propose_interventions(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    _, candidate_frame, _, _ = score_intervention_candidates(bundle, sample_payload)
    target_payload = rank_targets(bundle, sample_payload)["target_set_ranking"]
    top_k = int(sample_payload.get("top_k", 20))
    top_frame = candidate_frame.sort_values(by=["predicted_score", "tri_score"], ascending=[False, False]).head(top_k).copy()
    return {
        "posterior": infer_posterior(bundle, sample_payload),
        "target_program_ranking": target_payload,
        "intervention_program_ranking": top_frame[
            [
                "intervention_program_uid",
                "regimen_uid",
                "target_program_label_set",
                "compound_set",
                "route",
                "schedule",
                "dose_level_id",
                "dose_multiplier",
                "occupancy_constraints",
                "predicted_score",
                "tri_score",
                "selection_reason",
            ]
        ].to_dict(orient="records"),
    }


def rank_regimens(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    _, latent, disease_probability = patient_latent_from_payload(bundle, sample_payload)
    regimen_count = bundle["regimen_feature_matrix"].shape[0]
    tri_context = regimen_tri_context(bundle, sample_payload, regimen_count)
    model_output = bundle["screening_model"].score_regimens(
        latent,
        bundle["regimen_feature_matrix"],
        tri_context,
        conditioned_inputs=conditioned_inputs_from_payload(bundle, sample_payload, bundle["regimen_feature_frame"]["regimen_uid"].astype(str).tolist()),
    )
    regimen_scores = model_output["regimen_scores"][0].detach().cpu().numpy()
    tri_scores = model_output["tri_score"][0].detach().cpu().numpy()
    tri_component_payload = {
        component_name: model_output["tri_components"][component_name][0].detach().cpu().numpy()
        for component_name in TRI_COMPONENT_NAMES
    }
    frame = bundle["regimen_feature_frame"].copy()
    frame["predicted_score"] = regimen_scores
    frame["tri"] = tri_scores
    top_frame = frame.sort_values(by="predicted_score", ascending=False).head(20)
    return {
        "target_set_ranking": top_frame[["target_symbol_set", "predicted_score"]].drop_duplicates().head(10).to_dict(orient="records"),
        "drug_set_ranking": top_frame[["compound_uid_set", "predicted_score"]].drop_duplicates().head(10).to_dict(orient="records"),
        "dose_ranking": top_frame[["dose_level_id", "dose_multiplier", "predicted_score"]].drop_duplicates().head(10).to_dict(orient="records"),
        "schedule_ranking": top_frame[["schedule", "predicted_score"]].drop_duplicates().head(10).to_dict(orient="records"),
        "route_ranking": top_frame[["route", "predicted_score"]].drop_duplicates().head(10).to_dict(orient="records"),
        "tri": float(top_frame["tri"].iloc[0]) if not top_frame.empty else 0.0,
        "abstain_flag": (model_output.get("uncertainty_status") != "calibrated_predictive_samples"
                         or (bool(top_frame["tri"].iloc[0] < bundle["preprocessing_manifest"]["tri_threshold"]) if not top_frame.empty else True)),
        "uncertainty_status": model_output.get("uncertainty_status", "not_available"),
        "tri_component_breakdown": (
            {
                component_name: float(tri_component_payload[component_name][top_frame.index[0]])
                for component_name in TRI_COMPONENT_NAMES
            }
            if not top_frame.empty
            else {component_name: 0.0 for component_name in TRI_COMPONENT_NAMES}
        ),
        "supporting_evidence_summary": top_frame[["aggregate_target_evidence_score", "clue_reversal_proxy_score", "pk_feasibility_score"]].head(5).to_dict(orient="records"),
    }


def validate_candidates(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    candidate_frame = candidate_intervention_frame(bundle, sample_payload)
    validation_frame = bundle["candidate_validation_frame"].copy()
    validation_frame = validation_frame.loc[
        validation_frame["intervention_program_uid"].astype(str).isin(candidate_frame["intervention_program_uid"].astype(str).tolist())
    ].copy()
    validation_frame = validation_frame.sort_values(
        by=[
            "validation_stage",
            "affinity_evidence_score",
            "admet_readiness_score",
            "pbpk_readiness_score",
        ],
        ascending=[True, False, False, False],
    )
    return {
        "candidate_validation_count": int(validation_frame.shape[0]),
        "candidate_validations": validation_frame[
            [
                "candidate_validation_uid",
                "intervention_program_uid",
                "regimen_uid",
                "validation_stage",
                "structure_confidence",
                "docking_score",
                "rescoring_score",
                "affinity_evidence",
                "admet_summary",
                "pbpk_summary",
            ]
        ].to_dict(orient="records"),
        "validation_contract": "candidate_validation_catalog",
    }


def adapter_evidence_summary(bundle: dict[str, object]) -> dict[str, object]:
    adapter_frame = bundle.get("external_adapter_catalog", pd.DataFrame())
    if not isinstance(adapter_frame, pd.DataFrame) or adapter_frame.empty:
        return {"adapter_count": 0, "adapter_roles": {}, "method_contribution_flag": "none"}
    return {
        "adapter_count": int(adapter_frame.shape[0]),
        "adapter_roles": adapter_frame["adapter_role"].astype(str).value_counts(dropna=False).to_dict(),
        "method_contribution_flag": "no",
        "sources": adapter_frame[["source_key", "adapter_role", "provenance_level", "validation_policy"]].to_dict(orient="records"),
    }


def case_level_expert_diagnostics(config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    required_diagnostics = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["mechanism_explainability"]["required_diagnostics"]
    provided = sample_payload.get("case_level_expert_diagnostics", {})
    if provided:
        missing = sorted(set(str(name) for name in required_diagnostics) - set(str(name) for name in dict(provided)))
        if missing:
            raise ValueError(f"case_level_expert_diagnostics missing required fields: {missing}")
        return {"diagnostic_status": "provided", "diagnostics": provided}
    return {
        "diagnostic_status": "not_provided_by_runtime_bundle",
        "required_diagnostics": [str(name) for name in required_diagnostics],
    }


def decision_label(
    predicted_score: float,
    tri_score: float,
    pkpd_consistency: float,
    ood_score: float,
    validation_stage: str,
    decision_gate: dict[str, object],
) -> str:
    labels = decision_gate["decision_labels"]
    if not np.isfinite([predicted_score, tri_score, pkpd_consistency, ood_score]).all():
        return str(labels["insufficient_evidence"])
    score_accept_threshold = float(decision_gate["score_accept_threshold"])
    score_active_learning_threshold = float(decision_gate["score_active_learning_threshold"])
    tri_accept_threshold = float(decision_gate["tri_accept_threshold"])
    pkpd_threshold = float(decision_gate["pkpd_consistency_threshold"])
    ood_reject_threshold = float(decision_gate["ood_reject_threshold"])
    accepted_validation_stages = {str(stage) for stage in decision_gate["accepted_validation_stages"]}
    if ood_score >= ood_reject_threshold:
        return str(labels["insufficient_evidence"])
    if predicted_score >= score_active_learning_threshold and tri_score < tri_accept_threshold:
        return str(labels["active_learning"])
    if predicted_score >= score_accept_threshold and pkpd_consistency < pkpd_threshold:
        return str(labels["external_validation"])
    if predicted_score >= score_accept_threshold and tri_score >= tri_accept_threshold and pkpd_consistency >= pkpd_threshold:
        if validation_stage in accepted_validation_stages:
            return str(labels["accept"])
        return str(labels["external_validation"])
    return str(labels["reject"])


def decision_reason(
    predicted_score: float,
    tri_score: float,
    pkpd_consistency: float,
    ood_score: float,
    validation_stage: str,
    decision_gate: dict[str, object],
) -> str:
    label = decision_label(predicted_score, tri_score, pkpd_consistency, ood_score, validation_stage, decision_gate)
    if label == decision_gate["decision_labels"]["insufficient_evidence"]:
        return "ood_or_evidence_coverage_exceeds_reliability_boundary"
    if label == decision_gate["decision_labels"]["active_learning"]:
        return "high_model_score_but_tri_below_acceptance_threshold"
    if label == decision_gate["decision_labels"]["external_validation"]:
        return "candidate_requires_downstream_or_pkpd_validation_before_acceptance"
    if label == decision_gate["decision_labels"]["accept"]:
        return "score_tri_pkpd_and_validation_stage_support_downstream_validation"
    return "score_or_reliability_below_decision_gate"


def mean_candidate_value(candidate_rows: list[dict[str, object]], key: str) -> float:
    if not candidate_rows:
        return 0.0
    missing_rows = [row for row in candidate_rows if key not in row]
    if missing_rows:
        raise ValueError(f"Candidate summary missing required uncertainty field: {key}")
    return float(np.mean([float(row[key]) for row in candidate_rows]))


def rerank_with_reliability(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    _, candidate_frame, predicted_scores, tri_scores = score_intervention_candidates(bundle, sample_payload)
    validation_frame = bundle["candidate_validation_frame"].copy()
    validation_frame = validation_frame.loc[
        validation_frame["intervention_program_uid"].astype(str).isin(candidate_frame["intervention_program_uid"].astype(str).tolist())
    ].copy()
    reranked = validation_rerank_frame(
        candidate_frame,
        predicted_scores,
        tri_scores,
        validation_frame,
        config["screening_pipeline"]["screening_benchmark"]["validation_reranking"]["weights"],
    )
    decision_gate = config["screening_pipeline"]["screening_benchmark"]["decision_gate"]
    reranked["decision"] = [
        decision_label(float(predicted_score), float(tri_score), float(pkpd_consistency), float(ood_score), str(validation_stage), decision_gate)
        for predicted_score, tri_score, pkpd_consistency, ood_score, validation_stage in zip(
            reranked["predicted_score"].tolist(),
            reranked["tri_score"].tolist(),
            reranked["tri_pkpd_physics_consistency"].tolist(),
            reranked["tri_ood_score"].tolist(),
            reranked["validation_stage"].astype(str).tolist(),
            strict=True,
        )
    ]
    reranked["decision_reason"] = [
        decision_reason(float(predicted_score), float(tri_score), float(pkpd_consistency), float(ood_score), str(validation_stage), decision_gate)
        for predicted_score, tri_score, pkpd_consistency, ood_score, validation_stage in zip(
            reranked["predicted_score"].tolist(),
            reranked["tri_score"].tolist(),
            reranked["tri_pkpd_physics_consistency"].tolist(),
            reranked["tri_ood_score"].tolist(),
            reranked["validation_stage"].astype(str).tolist(),
            strict=True,
        )
    ]
    top_k = int(sample_payload.get("top_k", 20))
    if "uncertainty_status" in candidate_frame:
        status_by_uid = candidate_frame.set_index("regimen_uid")["uncertainty_status"]
        uncalibrated = reranked["regimen_uid"].map(status_by_uid).ne("calibrated_predictive_samples")
        reranked.loc[uncalibrated, "decision"] = str(decision_gate["decision_labels"]["insufficient_evidence"])
        reranked.loc[uncalibrated, "decision_reason"] = "conditional_rollout_uncertainty_not_calibrated"
    top_frame = reranked.head(top_k).copy()
    reranked_candidate_fields = [
        "intervention_program_uid",
        "regimen_uid",
        "target_program_label_set",
        "compound_set",
        "route",
        "schedule",
        "dose_level_id",
        "predicted_score",
        "tri_score",
        "ranking_stability",
        "rollout_uncertainty",
        *[f"tri_{component_name}" for component_name in TRI_COMPONENT_NAMES],
        "validation_stage",
        "rerank_score",
        "decision",
        "decision_reason",
    ]
    return {
        "candidate_count": int(candidate_frame.shape[0]),
        "reranked_candidates": top_frame[reranked_candidate_fields].to_dict(orient="records"),
        "decision_summary": top_frame["decision"].value_counts(dropna=False).to_dict(),
    }


def tri_only(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    ranking = rank_regimens(bundle, sample_payload)
    return {
        "tri": ranking["tri"],
        "abstain_flag": ranking["abstain_flag"],
        "tri_component_breakdown": ranking["tri_component_breakdown"],
    }


def acquire_next_regimens(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    _, _, predicted_scores, _ = score_intervention_candidates(bundle, sample_payload)
    candidate_frame = candidate_intervention_frame(bundle, sample_payload)
    uncertainty_payload = {
        str(key): float(value)
        for key, value in sample_payload.get("epistemic_uncertainty_by_regimen", {}).items()
    }
    epistemic_uncertainty = np.asarray(
        [
            uncertainty_payload.get(str(intervention_uid), 0.0)
            for intervention_uid in candidate_frame["intervention_program_uid"].astype(str).tolist()
        ],
        dtype=np.float64,
    )
    validation_lookup = {
        str(row["intervention_program_uid"]): (
            0.30 * float(row["affinity_evidence_score"])
            + 0.35 * float(row["admet_readiness_score"])
            + 0.35 * float(row["pbpk_readiness_score"])
        )
        for row in bundle["candidate_validation_frame"].to_dict(orient="records")
    }
    acquisition_candidates = candidate_frame.copy()
    acquisition_candidates["target_symbol_set"] = acquisition_candidates["target_program_label_set"].astype(str)
    acquisition_candidates["validation_readiness_score"] = acquisition_candidates["intervention_program_uid"].astype(str).map(validation_lookup).fillna(0.0)
    acquisition_frame = build_acquisition_frame(acquisition_candidates, predicted_scores, epistemic_uncertainty, config)
    batch_size = int(
        sample_payload.get(
            "batch_size",
            config["screening_pipeline"]["screening_benchmark"]["active_learning"]["acquisition_batch_size"],
        )
    )
    selected = select_acquisition_batch(acquisition_frame, batch_size=batch_size)
    return {
        "selected_regimens": selected[
            [
                "regimen_uid",
                "intervention_program_uid",
                "target_program_label_set",
                "route",
                "schedule",
                "dose_level_id",
                "acquisition_rank",
                "acquisition_score",
                "expected_utility",
                "epistemic_uncertainty",
                "diversity_bonus",
                "pkpd_component",
                "validation_readiness",
                "safety_penalty",
            ]
        ].to_dict(orient="records"),
        "candidate_count": int(candidate_frame.shape[0]),
        "selected_count": int(selected.shape[0]),
    }


def rollout_regimens(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    modality, latent, disease_probability = patient_latent_from_payload(bundle, sample_payload)
    regimen_uids = [str(value) for value in sample_payload.get("regimen_uids", [])]
    if not regimen_uids:
        raise ValueError("rollout_regimens requires a non-empty regimen_uids list.")
    regimen_rows = []
    sequence_tensors = []
    tri_context_sequence = []
    for regimen_uid in regimen_uids:
        if regimen_uid not in bundle["regimen_index_by_uid"]:
            raise ValueError(f"Unknown regimen_uid: {regimen_uid}")
        regimen_index = bundle["regimen_index_by_uid"][regimen_uid]
        regimen_rows.append(bundle["regimen_feature_frame"].iloc[regimen_index].to_dict())
        sequence_tensors.append(bundle["regimen_feature_matrix"][regimen_index])
        tri_context_sequence.append(regimen_tri_context(bundle, sample_payload, 1))
    regimen_sequence = torch.stack(sequence_tensors, dim=0).unsqueeze(0)
    conditioned_sequence = [conditioned_inputs_from_payload(bundle, sample_payload, [uid]) for uid in regimen_uids]
    rollout_output = bundle["screening_model"].rollout(latent, regimen_sequence, tri_context_sequence,
                                                     conditioned_input_sequence=conditioned_sequence)
    next_disease_probabilities = torch.sigmoid(rollout_output["next_disease_logit_sequence"]).detach().cpu().numpy()[0]
    tri_score_sequence = rollout_output["tri_score_sequence"].detach().cpu().numpy()[0]
    field_sequence = rollout_output["field_trajectory_sequence"].detach().cpu().numpy()[0]
    region_labels = bundle["region_catalog"]["region_label"].astype(str).tolist()
    species_labels = bundle["species_catalog"]["species_id"].astype(str).tolist()
    region_prior_frame = bundle["region_prior_catalog"].copy()
    disease_program = str(bundle["preprocessing_manifest"].get("active_disease_program", "ad_control"))
    region_prior_frame = region_prior_frame.loc[region_prior_frame["disease_program"] == disease_program].copy()
    for column_name in ["program_support_score", "disease_vulnerability_score", "transport_centrality_score"]:
        region_prior_frame[column_name] = pd.to_numeric(region_prior_frame[column_name], errors="coerce").fillna(0.0)
    region_prior_summary_lookup = {
        str(row["region_id"]): {
            "region_structural_prior_score": float(row["disease_vulnerability_score"]),
            "region_program_support_score": float(row["program_support_score"]),
            "region_transport_centrality": float(row["transport_centrality_score"]),
        }
        for row in region_prior_frame.to_dict(orient="records")
    }
    regional_exposure = []
    regional_pathology = []
    structural_prior_sequence = []
    for step_index, regimen_row in enumerate(regimen_rows):
        step_field = field_sequence[step_index, 0]
        final_field = step_field[-1]
        free_drug = final_field[..., species_index(species_labels, "free_drug")].mean(axis=1)
        plaque_proxy = final_field[..., species_index(species_labels, "aggregate_burden", "plaque_proxy")].mean(axis=1)
        excitability_drive = final_field[..., species_index(species_labels, "circuit_instability", "excitability_drive")].mean(axis=1)
        regional_exposure.append(
            {
                "regimen_uid": str(regimen_row["regimen_uid"]),
                "region_exposure": [
                    {"region_label": region_label, "free_drug": float(value)}
                    for region_label, value in zip(region_labels, free_drug.tolist(), strict=True)
                ],
            }
        )
        regional_pathology.append(
            {
                "regimen_uid": str(regimen_row["regimen_uid"]),
                "region_pathology": [
                    {"region_label": region_label, "plaque_proxy": float(value)}
                    for region_label, value in zip(region_labels, plaque_proxy.tolist(), strict=True)
                ],
            }
        )
        structural_prior_sequence.append(
            {
                "regimen_uid": str(regimen_row["regimen_uid"]),
                "region_structural_priors": [
                    {
                        "region_label": region_label,
                        "region_structural_prior_score": region_prior_summary_lookup.get(region_id, {}).get("region_structural_prior_score", 0.0),
                        "local_response_head_activation": float(excitability_value * (0.5 + region_prior_summary_lookup.get(region_id, {}).get("region_transport_centrality", 0.0))),
                    }
                    for region_id, region_label, excitability_value in zip(
                        bundle["region_catalog"]["region_id"].astype(str).tolist(),
                        region_labels,
                        excitability_drive.tolist(),
                        strict=True,
                    )
                ],
            }
        )
    return {
        "modality": modality,
        "disease_probability": disease_probability,
        "regimen_sequence": [
            {
                "regimen_uid": str(row["regimen_uid"]),
                "target_symbol_set": str(row["target_symbol_set"]),
                "route": str(row["route"]),
                "schedule": str(row["schedule"]),
                "dose_level_id": str(row["dose_level_id"]),
            }
            for row in regimen_rows
        ],
        "next_state_probability_sequence": [float(value) for value in next_disease_probabilities.tolist()],
        "tri_score_sequence": [float(value) for value in tri_score_sequence.tolist()],
        "regional_exposure_sequence": regional_exposure,
        "regional_pathology_sequence": regional_pathology,
        "regional_structural_prior_sequence": structural_prior_sequence,
        "pkpd_consistency_sequence": [
            float(value)
            for value in rollout_output["pkpd_consistency_sequence"].detach().cpu().numpy()[0].tolist()
        ],
        "tri_components": {
            component_name: [
                float(value)
                for value in rollout_output["tri_components"][component_name].detach().cpu().numpy()[0].tolist()
            ]
            for component_name in TRI_COMPONENT_NAMES
        },
    }


def rollout_fields(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    rollout_payload = rollout_regimens(bundle, sample_payload)
    return {
        "modality": rollout_payload["modality"],
        "regimen_sequence": rollout_payload["regimen_sequence"],
        "regional_exposure_sequence": rollout_payload["regional_exposure_sequence"],
        "regional_pathology_sequence": rollout_payload["regional_pathology_sequence"],
        "regional_structural_prior_sequence": rollout_payload["regional_structural_prior_sequence"],
        "tri_score_sequence": rollout_payload["tri_score_sequence"],
    }


def rollout_counterfactuals(bundle: dict[str, object], sample_payload: dict[str, object]) -> dict[str, object]:
    scenario_payloads = list(sample_payload.get("scenario_sequences", []))
    if not scenario_payloads:
        baseline_regimens = [str(value) for value in sample_payload.get("regimen_uids", [])]
        counterfactual_regimens = [str(value) for value in sample_payload.get("counterfactual_regimen_uids", [])]
        if not baseline_regimens or not counterfactual_regimens:
            raise ValueError("rollout_counterfactuals requires scenario_sequences or both regimen_uids and counterfactual_regimen_uids.")
        scenario_payloads = [
            {"scenario_label": "baseline", "regimen_uids": baseline_regimens},
            {"scenario_label": "counterfactual", "regimen_uids": counterfactual_regimens},
        ]
    scenario_results = []
    for scenario_row in scenario_payloads:
        scenario_result = rollout_regimens(
            bundle,
            {
                **sample_payload,
                "regimen_uids": [str(value) for value in scenario_row["regimen_uids"]],
            },
        )
        scenario_results.append(
            {
                "scenario_label": str(scenario_row["scenario_label"]),
                "regimen_sequence": scenario_result["regimen_sequence"],
                "final_state_probability": float(scenario_result["next_state_probability_sequence"][-1]),
                "final_tri_score": float(scenario_result["tri_score_sequence"][-1]),
                "pkpd_consistency_sequence": scenario_result["pkpd_consistency_sequence"],
            }
        )
    baseline_result = scenario_results[0]
    comparisons = []
    for scenario_result in scenario_results[1:]:
        comparisons.append(
            {
                "scenario_label": scenario_result["scenario_label"],
                "delta_final_state_probability": (
                    float(scenario_result["final_state_probability"]) - float(baseline_result["final_state_probability"])
                ),
                "delta_final_tri_score": float(scenario_result["final_tri_score"]) - float(baseline_result["final_tri_score"]),
            }
        )
    world_state_payload = infer_world_state(bundle, sample_payload)
    return {
        "world_state": world_state_payload,
        "scenario_rollouts": scenario_results,
        "pairwise_comparisons": comparisons,
        "semantics_uid": world_state_payload["semantics_uid"],
        "unit_context": world_state_payload["unit_context"],
        "standardized_value": world_state_payload["standardized_value"],
        "original_unit_value": world_state_payload["original_unit_value"],
        "proxy_flag": world_state_payload["proxy_flag"],
        "calibration_status": world_state_payload["calibration_status"],
    }


def plan_interventions(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    reranked_payload = rerank_with_reliability(bundle, config, sample_payload)
    acquisition_payload = acquire_next_regimens(bundle, config, sample_payload)
    planner_candidates = reranked_payload["reranked_candidates"][: int(sample_payload.get("top_k", 10))]
    return {
        "world_state": infer_world_state(bundle, sample_payload),
        "planning_objective": "screening_decision_core",
        "planned_interventions": planner_candidates,
        "exploration_batch": acquisition_payload["selected_regimens"],
        "decision_summary": reranked_payload["decision_summary"],
        "stochastic_uncertainty_summary": {
            "mean_ranking_stability": mean_candidate_value(planner_candidates, "ranking_stability"),
            "mean_rollout_uncertainty": mean_candidate_value(planner_candidates, "rollout_uncertainty"),
            "mean_tri_coverage": mean_candidate_value(planner_candidates, "tri_coverage"),
            "mean_tri_ood_score": mean_candidate_value(planner_candidates, "tri_ood_score"),
            "mean_tri_pkpd_physics_consistency": mean_candidate_value(planner_candidates, "tri_pkpd_physics_consistency"),
        },
        "external_adapter_evidence_summary": adapter_evidence_summary(bundle),
    }


def explain_decision(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    world_state_payload = infer_world_state(bundle, sample_payload)
    reranked_payload = rerank_with_reliability(bundle, config, sample_payload)
    reranked_candidates = reranked_payload["reranked_candidates"]
    if not reranked_candidates:
        raise ValueError("explain_decision requires at least one reranked candidate.")
    top_candidate = reranked_candidates[0]
    runner_up = reranked_candidates[1] if len(reranked_candidates) > 1 else None
    factor_attributions = [
        {
            "factor_id": factor_name,
            "factor_strength": float(world_state_payload["factor_strengths"][factor_name]),
        }
        for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
    ]
    driver_summary = [
        {"driver": "predicted_score", "value": float(top_candidate["predicted_score"])},
        {"driver": "tri_score", "value": float(top_candidate["tri_score"])},
        {"driver": "ranking_stability", "value": float(top_candidate["ranking_stability"])},
        {"driver": "rollout_uncertainty", "value": float(top_candidate["rollout_uncertainty"])},
        {"driver": "tri_coverage", "value": float(top_candidate["tri_coverage"])},
        {"driver": "tri_ood_score", "value": float(top_candidate["tri_ood_score"])},
        {"driver": "tri_pkpd_physics_consistency", "value": float(top_candidate["tri_pkpd_physics_consistency"])},
    ]
    if runner_up is not None:
        driver_summary.append(
            {
                "driver": "margin_vs_runner_up",
                "value": float(top_candidate["rerank_score"]) - float(runner_up["rerank_score"]),
            }
        )
    return {
        "top_candidate": top_candidate,
        "world_state_factor_attributions": factor_attributions,
        "decision_drivers": driver_summary,
        "decision_summary": reranked_payload["decision_summary"],
        "external_adapter_evidence_summary": adapter_evidence_summary(bundle),
        "case_level_expert_diagnostics": case_level_expert_diagnostics(config, sample_payload),
        "semantics_uid": world_state_payload["semantics_uid"],
        "unit_context": world_state_payload["unit_context"],
        "standardized_value": world_state_payload["standardized_value"],
        "original_unit_value": world_state_payload["original_unit_value"],
        "proxy_flag": world_state_payload["proxy_flag"],
        "calibration_status": world_state_payload["calibration_status"],
    }


def active_learning_round_uid(config: dict, sample_payload: dict[str, object]) -> str:
    if "round_uid" in sample_payload:
        return str(sample_payload["round_uid"])
    rounds = active_learning_config(config)["active_rounds"]
    if not rounds:
        raise ValueError("Active-learning configuration requires at least one round.")
    return str(rounds[0]["round_uid"])


def active_learning_priority_frame(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> pd.DataFrame:
    priority_payload = dict(sample_payload)
    priority_payload["top_k"] = int(
        sample_payload.get(
            "priority_top_k",
            active_learning_config(config)["full_tasks_per_round"],
        )
    )
    reranked_payload = rerank_with_reliability(bundle, config, priority_payload)
    rows: list[dict[str, object]] = []
    for candidate in reranked_payload["reranked_candidates"]:
        rows.append(
            {
                "regimen_uid": str(candidate["regimen_uid"]),
                "predicted_score_mean": float(candidate["predicted_score"]),
                "tri_score_mean": float(candidate["tri_score"]),
                "pkpd_consistency_mean": float(candidate["tri_pkpd_physics_consistency"]),
                "epistemic_std_mean": float(candidate["rollout_uncertainty"]),
                "predictive_entropy_mean": float(1.0 - float(candidate["ranking_stability"])),
                "mutual_information_mean": float(1.0 - float(candidate["tri_coverage"])),
                "selection_reason": str(candidate["decision_reason"]),
                "evidence_summary": str(candidate["decision"]),
                "source_holdout_role": str(sample_payload.get("source_holdout_role", "active_learning_serving_planning")),
            }
        )
    if not rows:
        raise ValueError("Active-learning planning requires at least one reranked candidate.")
    return pd.DataFrame(rows)


def active_learning_request_payload(
    bundle: dict[str, object],
    config: dict,
    sample_payload: dict[str, object],
    persist_artifacts: bool,
) -> dict[str, object]:
    round_uid = active_learning_round_uid(config, sample_payload)
    task_count = int(
        sample_payload.get(
            "task_count",
            sample_payload.get("batch_size", active_learning_config(config)["full_tasks_per_round"]),
        )
    )
    priority_frame = active_learning_priority_frame(bundle, config, sample_payload)
    candidate_frame = build_active_candidate_pool(
        round_uid,
        bundle["intervention_program_frame"].copy(),
        bundle["candidate_validation_frame"].copy(),
        priority_frame,
        config,
    )
    acquisition_frame = score_acquisition_tasks(round_uid, candidate_frame, config)
    request_frame = select_request_batch(round_uid, acquisition_frame, candidate_frame, config, task_count)
    if persist_artifacts:
        write_frame(
            round_path(config, "candidate_pool_template", round_uid),
            config["schemas"]["active_learning_candidate_pool_fields"],
            candidate_frame,
        )
        write_frame(
            round_path(config, "acquisition_scores_template", round_uid),
            config["schemas"]["active_learning_acquisition_fields"],
            acquisition_frame,
        )
        write_frame(
            round_path(config, "request_batch_template", round_uid),
            config["schemas"]["active_learning_simulation_request_fields"],
            request_frame,
        )
    acquisition_lookup = acquisition_frame.set_index("request_uid").to_dict(orient="index")
    request_rows = request_frame.to_dict(orient="records")
    for request_row in request_rows:
        acquisition_row = acquisition_lookup[str(request_row["request_uid"])]
        request_row["acquisition_score"] = float(acquisition_row["acquisition_score"])
        request_row["expected_decision_risk_reduction"] = float(acquisition_row["expected_information_gain"])
        request_row["rank_flip_risk"] = float(acquisition_row["rank_flip_risk"])
        request_row["tri_uncertainty"] = float(acquisition_row["tri_uncertainty"])
        request_row["mechanism_expert_disagreement"] = float(acquisition_row["mechanism_expert_disagreement"])
    return {
        "round_uid": round_uid,
        "task_count": task_count,
        "selected_task_count": int(request_frame.shape[0]),
        "candidate_count": int(candidate_frame.shape[0]),
        "acquisition_task_count": int(acquisition_frame.shape[0]),
        "selected_tasks": request_rows,
        "batch_quota": active_learning_config(config)["batch_quota"],
        "simulator_portfolio": active_learning_config(config)["simulator_portfolio"],
        "artifact_paths": {
            "candidate_pool": str(round_path(config, "candidate_pool_template", round_uid)),
            "acquisition_scores": str(round_path(config, "acquisition_scores_template", round_uid)),
            "request_batch": str(round_path(config, "request_batch_template", round_uid)),
        },
    }


def plan_active_learning_batch(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    return active_learning_request_payload(bundle, config, sample_payload, persist_artifacts=bool(sample_payload.get("persist_artifacts", False)))


def dispatch_validation_tasks(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    payload = active_learning_request_payload(bundle, config, sample_payload, persist_artifacts=True)
    dispatched_rows = []
    for row in payload["selected_tasks"]:
        dispatched_row = dict(row)
        dispatched_row["execution_status"] = "queued_for_dispatch"
        dispatched_rows.append(dispatched_row)
    request_frame = pd.DataFrame(dispatched_rows)
    write_frame(
        round_path(config, "request_batch_template", str(payload["round_uid"])),
        config["schemas"]["active_learning_simulation_request_fields"],
        request_frame,
    )
    payload["selected_tasks"] = dispatched_rows
    payload["dispatch_status"] = "queued_for_simulator_portfolio"
    return payload


def active_learning_frame_from_payload(config: dict, sample_payload: dict[str, object], frame_key: str, path_key: str) -> pd.DataFrame:
    if path_key in sample_payload:
        return pd.DataFrame(read_tsv(resolve_repo_path(str(sample_payload[path_key]))))
    if frame_key in sample_payload:
        return pd.DataFrame(sample_payload[frame_key])
    raise ValueError(f"Active-learning ingest requires either {path_key} or {frame_key}.")


def ingest_simulation_result(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    round_uid = active_learning_round_uid(config, sample_payload)
    request_frame = active_learning_frame_from_payload(config, sample_payload, "request_batch", "request_manifest_path")
    result_frame = active_learning_frame_from_payload(config, sample_payload, "result_manifest", "result_manifest_path")
    result_frame, teacher_update_frame, qc_summary = ingest_active_learning_results(round_uid, request_frame, result_frame, config)
    write_frame(
        round_path(config, "result_manifest_template", round_uid),
        config["schemas"]["active_learning_simulation_result_fields"],
        result_frame,
    )
    write_frame(
        round_path(config, "teacher_update_template", round_uid),
        config["schemas"]["active_learning_teacher_update_fields"],
        teacher_update_frame,
    )
    write_json(round_path(config, "qc_summary_template", round_uid), qc_summary)
    report_path = write_round_report(config, round_uid)
    return {
        "round_uid": round_uid,
        "ingested_result_count": int(result_frame.shape[0]),
        "teacher_update_count": int(teacher_update_frame.shape[0]),
        "qc_summary": qc_summary,
        "round_report_path": report_path.as_posix(),
    }


def active_learning_round_report(config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    round_uid = active_learning_round_uid(config, sample_payload)
    if bool(sample_payload.get("rebuild_report", False)):
        report_path = write_round_report(config, round_uid)
    else:
        report_path = round_path(config, "round_report_template", round_uid)
        if not report_path.exists():
            raise FileNotFoundError(f"Active-learning round report is unavailable: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def persist_feedback_event(config: dict, feedback_row: dict[str, object]) -> Path:
    registry_config = config["data"]["feedback_ready_registry"]
    catalog_root = resolve_repo_path(registry_config["local_root"])
    ensure_directory(catalog_root)
    catalog_path = catalog_root / registry_config["relative_paths"]["feedback_event_catalog"]
    existing_rows = read_tsv(catalog_path) if catalog_path.exists() else []
    existing_rows.append({key: feedback_row[key] for key in feedback_row})
    write_tsv(catalog_path, config["schemas"]["feedback_event_fields"], existing_rows)
    event_root = resolve_repo_path(config["screening_pipeline"]["screening_benchmark"]["online_learning"]["feedback_artifact_root"])
    ensure_directory(event_root)
    event_path = event_root / f"{feedback_row['feedback_event_uid']}.json"
    write_json(event_path, feedback_row)
    return event_path


def ingest_feedback(bundle: dict[str, object], config: dict, sample_payload: dict[str, object]) -> dict[str, object]:
    required_fields = [
        "sample_uid",
        "intervention_program_uid",
        "regimen_uid",
        "predicted_rollout_summary",
        "realized_outcome_summary",
    ]
    missing_fields = [field_name for field_name in required_fields if field_name not in sample_payload]
    if missing_fields:
        raise ValueError(f"ingest_feedback is missing required fields: {missing_fields}")
    disease_program = str(sample_payload.get("disease_program", bundle["preprocessing_manifest"]["active_disease_program"]))
    disease_family = disease_family_name(disease_program)
    predicted_summary = dict(sample_payload["predicted_rollout_summary"])
    realized_summary = dict(sample_payload["realized_outcome_summary"])
    predicted_value = float(predicted_summary.get("final_state_probability", 0.0))
    realized_value = float(realized_summary.get("final_state_probability", predicted_value))
    discrepancy_value = abs(predicted_value - realized_value)
    prior_calibration_impact = float(sample_payload.get("calibration_impact_score", discrepancy_value))
    feedback_policy = config["screening_pipeline"]["screening_benchmark"]["online_learning"]["feedback_update_policy"]
    update_eligibility = discrepancy_value >= float(feedback_policy["discrepancy_gate_threshold"])
    minimum_events = int(feedback_policy["minimum_feedback_events_for_parameter_update"])
    existing_feedback_count = int(bundle["feedback_event_catalog"].shape[0])
    parameter_update_allowed = update_eligibility and existing_feedback_count + 1 >= minimum_events
    update_decision = "parameter_update" if parameter_update_allowed else ("state_only_update" if update_eligibility else "calibration_only")
    event_timestamp = datetime.now(timezone.utc).isoformat()
    feedback_row = {
        "feedback_event_uid": stable_uid(
            "feedback_event",
            sample_payload["sample_uid"],
            sample_payload["intervention_program_uid"],
            sample_payload["regimen_uid"],
            event_timestamp,
        ),
        "disease_program": disease_program,
        "disease_family": disease_family,
        "sample_uid": str(sample_payload["sample_uid"]),
        "intervention_program_uid": str(sample_payload["intervention_program_uid"]),
        "regimen_uid": str(sample_payload["regimen_uid"]),
        "predicted_rollout_summary": json.dumps(predicted_summary, sort_keys=True, ensure_ascii=True),
        "realized_outcome_summary": json.dumps(realized_summary, sort_keys=True, ensure_ascii=True),
        "discrepancy_summary": json.dumps(
            {
                "absolute_error": discrepancy_value,
                "predicted_final_state_probability": predicted_value,
                "realized_final_state_probability": realized_value,
            },
            sort_keys=True,
            ensure_ascii=True,
        ),
        "calibration_impact_score": round(prior_calibration_impact, 6),
        "update_eligibility_flag": "yes" if update_eligibility else "no",
        "update_decision": update_decision,
        "event_timestamp": event_timestamp,
        "note": str(sample_payload.get("note", "feedback_ingested_from_serving_api")),
    }
    event_path = persist_feedback_event(config, feedback_row)
    new_feedback_frame = pd.DataFrame([feedback_row])
    if bundle["feedback_event_catalog"].empty:
        bundle["feedback_event_catalog"] = new_feedback_frame
    else:
        bundle["feedback_event_catalog"] = pd.concat(
            [bundle["feedback_event_catalog"], new_feedback_frame],
            ignore_index=True,
        )
    return {
        "feedback_event_uid": str(feedback_row["feedback_event_uid"]),
        "disease_family": disease_family,
        "update_eligibility_flag": update_eligibility,
        "update_decision": update_decision,
        "discrepancy_summary": json.loads(str(feedback_row["discrepancy_summary"])),
        "event_artifact_path": event_path.as_posix(),
    }


def app_factory(bundle: dict[str, object], config: dict):
    host = config["screening_pipeline"]["screening_serving"]["server"]["host"]
    port = int(config["screening_pipeline"]["screening_serving"]["server"]["port"])

    class Handler(BaseHTTPRequestHandler):
        def _json_response(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Request body is empty.")
            return json.loads(self.rfile.read(content_length).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json_response(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "host": host,
                        "port": port,
                        "public_endpoints": config["screening_pipeline"]["screening_serving"]["public_endpoints"],
                        "compatibility_endpoints": config["screening_pipeline"]["screening_serving"]["compatibility_endpoints"],
                    },
                )
                return
            if self.path == "/model_card":
                self._json_response(HTTPStatus.OK, bundle["model_card"])
                return
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "unknown_endpoint"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_body()
                if self.path == "/infer_world_state":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, infer_world_state(bundle, payload)))
                    return
                if self.path == "/rollout_counterfactuals":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, rollout_counterfactuals(bundle, payload)))
                    return
                if self.path == "/plan_interventions":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, plan_interventions(bundle, config, payload)))
                    return
                if self.path == "/explain_decision":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, explain_decision(bundle, config, payload)))
                    return
                if self.path == "/ingest_feedback":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, ingest_feedback(bundle, config, payload)))
                    return
                if self.path == "/plan_active_learning_batch":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, plan_active_learning_batch(bundle, config, payload)))
                    return
                if self.path == "/dispatch_validation_tasks":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, dispatch_validation_tasks(bundle, config, payload)))
                    return
                if self.path == "/ingest_simulation_result":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, ingest_simulation_result(bundle, config, payload)))
                    return
                if self.path == "/active_learning_round_report":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, active_learning_round_report(config, payload)))
                    return
                if self.path == "/infer_posterior":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, infer_posterior(bundle, payload)))
                    return
                if self.path == "/propose_interventions":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, propose_interventions(bundle, payload)))
                    return
                if self.path == "/validate_candidates":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, validate_candidates(bundle, payload)))
                    return
                if self.path == "/rerank_with_reliability":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, rerank_with_reliability(bundle, config, payload)))
                    return
                if self.path == "/infer_state":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, infer_state(bundle, payload)))
                    return
                if self.path == "/rank_targets":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, rank_targets(bundle, payload)))
                    return
                if self.path == "/rank_regimens":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, rank_regimens(bundle, payload)))
                    return
                if self.path == "/tri":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, tri_only(bundle, payload)))
                    return
                if self.path == "/acquire_next_regimens":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, acquire_next_regimens(bundle, config, payload)))
                    return
                if self.path == "/rollout_regimens":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, rollout_regimens(bundle, payload)))
                    return
                if self.path == "/rollout_fields":
                    self._json_response(HTTPStatus.OK, scale_aware_response(bundle, config, payload, rollout_fields(bundle, payload)))
                    return
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "unknown_endpoint"})
            except Exception as exc:  # noqa: BLE001
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format_string: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    serving_config = config["screening_pipeline"]["screening_serving"]
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else (
        REPO_ROOT
        / serving_config["production_root_template"].format(
            benchmark_name=config["screening_pipeline"]["screening_benchmark"]["benchmark_name"],
            model_version=serving_config["default_model_version"],
        )
    ).resolve()
    bundle = load_bundle(bundle_dir, config)
    server = ThreadingHTTPServer(
        (serving_config["server"]["host"], int(serving_config["server"]["port"])),
        app_factory(bundle, config),
    )
    pipeline_logger.artifact("serving_bundle_loaded", artifact=bundle_dir)
    pipeline_logger.progress(
        "server_listening",
        host=serving_config["server"]["host"],
        port=int(serving_config["server"]["port"]),
    )
    print(f"serving_bundle={bundle_dir}")
    print(f"listening=http://{serving_config['server']['host']}:{serving_config['server']['port']}")
    server.serve_forever()

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
