from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.utils.data import native_diffusion_data as native_data
from Libs.utils.data.native_observable_profile import quantity_support, family_grid, observable_profile
from Libs.utils.evaluation.native_causal_prefix import (
    COUPLING, causal_prefix_policy, declared_contrast, common_prefix_mask,
    match_record_conditions, match_prepared_pair, score_prefix_pair,
    contrast_row, build_prefix_payload, json_value)
from Libs.utils.evaluation.native_rollout_metrics import (
    metric_policy, metric_protocol, score_native_episode, score_point_mass_episode, score_mean_only_episode, failed_native_episode, build_native_metric_payload)
from Libs.scripts.services.serve_screening_api import load_screening_model
from Libs.utils.model.initial_state_persistence import predict_initial_state_persistence
from Libs.utils.model.native_direct_mean import load_frozen_conditioner, validate_direct_checkpoint, predict_direct_mean
from Libs.utils.model.screening_model_config import native_direct_mean_config
from Libs.exp.training.train_screening_system import screening_training_seed_binding
from Libs.utils.exp.experiment_manifests import validate_training_seed_checkpoint
from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config
from Libs.utils.model.dynamics_fields import (
    batch_simulation_targets,
    observed_region_mask,
    quantitative_case_validation_rows,
    route_ordering_exposure,
    selected_regimen_simulation_tensors,
    strict_case_dynamics_metrics,
)
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.foundation_twin_capabilities import capability_catalog_rows
from Libs.utils.data.raw_data_registry import resolve_repo_path, load_config, read_tsv, write_tsv
from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config
from Libs.utils.evaluation.screening_datasets import (
    fold_patient_latents,
    patient_regimen_labels,
    patient_target_labels,
    regimen_features,
    target_program_features,
    target_uid_to_entity_lookup,
)
from Libs.utils.model.screening_core import pkpd_prior_loss, pkpd_reference_tensors, tri_context_tensors
from Libs.utils.evaluation.screening_uq import (
    acceptance_benefit,
    adaptive_calibration_error,
    binary_brier,
    binary_nll,
    ensemble_uncertainty_summary,
    coverage_gap,
    expected_calibration_error,
    fit_temperature_from_grid,
    ood_auroc,
    risk_coverage_curve,
    sigmoid_with_temperature,
)
from Libs.utils.evaluation.strict_case_discovery import (
    assert_clean_action_ready_manifest,
    assert_no_forced_anchor_regimens,
    compute_anchor_recovery_metrics,
    compute_tri_accepted_anchor_metrics,
    regimen_anchor_mask,
    strict_case_entries,
    target_anchor_mask,
)


SCREENING_TASK_LOOKUP: dict[str, tuple[str, str]] = {}


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


def optional_tsv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the dynamics-grounded NeuronSifter AD screening system.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--fold-name", default="", help="Optional fold name subset.")
    parser.add_argument("--disease-program", default="", help="Optional disease program override.")
    parser.add_argument("--ensemble-mode", default="auto", choices=["auto", "single", "required"])
    return parser.parse_args()


def artifact_dir(config: dict, fold_name: str) -> Path:
    root = Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
    return root / config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] / "folds" / fold_name / "ad_screening_system"


def load_model(checkpoint_path: Path, config: dict) -> ADScreeningSystem:
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
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model



def _native_episode_identity(records: list[dict], registry: dict, policy: dict, config: dict) -> dict:
    teachers = [row["teacher"] for row in records]
    sample_ids = {row["sample_uid"] for row in teachers}
    families = {row["family_uid"] for row in teachers}
    sources = {row["source_uid"] for row in teachers}
    if len(sample_ids) != 1 or len(families) != 1 or len(sources) != 1:
        raise ValueError("Native episode actions require one declared sample/family/source unit")
    source = next(row for row in registry["sources"] if row["source_uid"] in sources)
    units = source.get("predictive_evaluation_units", {})
    if (not set(config["schemas"]["native_predictive_source_units"]["required_fields"]) <= set(units)
            or units.get("episode_field") != policy["episode_field"]
            or units.get("group_field") != policy["group_field"] or units.get("group_kind") != policy["group_kind"]
            or units.get("independent_episode_units") is not True or not units.get("evidence_ref")):
        raise ValueError("Source must explicitly declare matching independent episode/group semantics and evidence")
    sample = teachers[0]["sample_uid"]
    return {"episode_uid": sample, "sample_uid": sample, "family_uid": teachers[0]["family_uid"],
            "group_kind": policy["group_kind"], "source_uid": teachers[0]["source_uid"],
            "unit_declaration": dict(units), "regimen_uids": [row["regimen_uid"] for row in teachers],
            "action_uids": [row["action_uid"] for row in teachers],
            "trajectory_uids": [row["trajectory_uid"] for row in teachers],
            "posterior_uids": [row["posterior_uid"] for row in teachers]}


def _native_validity_mask(records: list[dict], grid: dict) -> torch.Tensor:
    shape = tuple(len(grid[key]) for key in ("time_hours", "region_id", "compartment_id", "species_id"))
    masks = []
    for record in records:
        declaration = record["teacher"].get("validity_mask")
        if not isinstance(declaration, dict) or not declaration.get("evidence_ref"):
            raise ValueError("Native scoring requires an explicit source-defined validity mask")
        if declaration.get("kind") == "all_valid":
            value = torch.ones(shape, dtype=torch.bool)
        elif declaration.get("kind") == "explicit":
            value = torch.as_tensor(declaration.get("values"))
            if value.dtype != torch.bool or tuple(value.shape) != shape:
                raise ValueError("Source validity mask must contain exact bool time/grid axes")
        else:
            raise ValueError("Unknown source validity mask convention")
        masks.append(value & torch.as_tensor(quantity_support(grid)))
    return torch.stack(masks)


def _native_training_features(config: dict, checkpoint_path: Path, checkpoint: dict,
                              binding: dict, regimens: list[str]) -> tuple[torch.Tensor, dict, list[Path]]:
    """Reuse the complete regimen metadata saved by the existing training run."""
    table_path = checkpoint_path.parent / "selected_regimens.parquet"
    manifest_path = checkpoint_path.parent / "training_manifest.json"
    manifest = load_training_manifest(manifest_path)
    policy = config["paper_grade_full_suite"]["native_rollout_evaluation"]
    version = config["screening_pipeline"]["screening_benchmark"]["model_contract_version"]
    if (manifest.get("training_seed_binding") != binding
            or manifest.get("scope") != policy["checkpoint_scope"]
            or manifest.get("split_name") != policy["checkpoint_split_name"]
            or manifest.get("model_contract_version") != version
            or checkpoint.get("model_contract_version") != version):
        raise ValueError("Native evaluation training-run feature context mismatch")
    frame = pd.read_parquet(table_path)
    if ("regimen_uid" not in frame or frame["regimen_uid"].duplicated().any()
            or len(frame) != manifest.get("selected_regimen_count")):
        raise ValueError("Native evaluation requires the complete unique selected-regimen metadata")
    # Rebuild on the complete saved table: subset category recoding would change
    # the training representation. This is the existing run context, not a new
    # cryptographic checkpoint-to-table binding.
    _, matrix = regimen_features(frame.to_dict(orient="records"))
    if (matrix.shape != (len(frame), checkpoint["model_kwargs"]["regimen_feature_dim"])
            or not np.isfinite(matrix).all()):
        raise ValueError("Training-run regimen features do not match the checkpoint width")
    positions = {str(uid): i for i, uid in enumerate(frame["regimen_uid"])}
    if any(uid not in positions for uid in regimens):
        raise ValueError("Selected action lacks metadata in this training run's complete regimen universe")
    provenance = {"selected_regimens": str(table_path), "training_manifest": str(manifest_path),
                  "selected_regimen_count": len(frame), "model_contract_version": version,
                  "binding_kind": "existing_training_run_context",
                  "checkpoint_table_cryptographic_binding": False}
    return torch.from_numpy(matrix[[positions[uid] for uid in regimens]]), provenance, [table_path, manifest_path]


def _native_observation_features(records: list[dict], model: ADScreeningSystem) -> torch.Tensor | None:
    if "joint_source" in records[0]:
        return None
    values = [record["observation"].get("screening_features") for record in records]
    if any(value != values[0] for value in values[1:]):
        raise ValueError("One sample must retain one observation-derived screening input")
    patient = torch.as_tensor(values[0], dtype=torch.float32).unsqueeze(0)
    if patient.shape != (1, model.patient_state_dim) or not torch.isfinite(patient).all():
        raise ValueError("Source sample observation features do not match the trained model")
    return patient


def prepare_native_episode(model: ADScreeningSystem, records: list[dict], config: dict,
                           arrays: dict) -> tuple[torch.Tensor | None, dict]:
    """Teacher arrays are excluded from this interface and from model conditioning."""
    if set(native_data.TEACHER_ARRAY_NAMES).intersection(arrays):
        raise ValueError("Teacher arrays cannot enter the native prediction interface")
    sample = records[0]["teacher"]["sample_uid"]
    regimens = [row["teacher"]["regimen_uid"] for row in records]
    patient = _native_observation_features(records, model)
    kwargs = {"sample_ids": torch.tensor([native_data.stable_numeric_id(sample)], dtype=torch.long),
              "regimen_ids": torch.tensor([native_data.stable_numeric_id(uid) for uid in regimens], dtype=torch.long),
              "include_auxiliary_outputs": True,
              "regimen_chunk_size": int(config["paper_grade_full_suite"]["native_rollout_evaluation"].get("regimen_chunk_size", 0))}
    if "joint_source" in records[0]:
        source = native_data.select_joint_source(records[0]["joint_source"], [sample], regimens,
                                                action_policy=native_data.native_action_policy(config))
        native_data.parse_joint_source(source, action_policy=native_data.native_action_policy(config))
        posterior = model.infer_world_state(joint_support=source["support"], observations=source["observations"])["joint_posterior"]
        programs = native_data.bind_joint_programs(posterior, source, action_policy=native_data.native_action_policy(config))
        kwargs.update(joint_posterior=posterior, particle_programs=programs, regimen_uids=regimens)
        patient = posterior.observation_features
    else:
        kwargs["diffusion_context"] = {key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
                                      for key, value in arrays.items()}
        profile = observable_profile(config)
        if profile is not None:
            grid = family_grid(profile, config["native_diffusion_data"]["source_family_uid"])
            expected = torch.as_tensor(quantity_support(grid)).unsqueeze(0).unsqueeze(0)
            kwargs["diffusion_context"]["quantity_support"] = expected.expand(1, len(regimens), *expected.shape[2:])
    return patient, kwargs


def predict_native_episode(model: ADScreeningSystem, records: list[dict], config: dict,
                           arrays: dict, features: torch.Tensor, *, prepared: tuple | None = None, direct_predictor=None, residual_context=None) -> dict:
    started = time.perf_counter()
    patient, kwargs = prepare_native_episode(model, records, config, arrays) if prepared is None else prepared
    conditioning_seconds = time.perf_counter() - started
    kind = config["paper_grade_full_suite"]["native_rollout_evaluation"].get("predictor_kind", "native_diffusion")
    if kind == "initial_state_persistence":
        output = predict_initial_state_persistence(
            posterior=kwargs.get("joint_posterior"),
            initial_state=kwargs.get("diffusion_context", {}).get("initial_state"),
            action_count=len(records), time_count=model.field_rollout_operator.rollout_step_count,
            sampler=model.field_rollout_operator.quantized_mddm_rollout_operator)
    elif kind == "native_diffusion":
        output = model.score_regimens(patient, features, **kwargs)
    elif kind in {"transolver_direct_mean", "transolver_fitted_residual", "linear_dynamics_direct_mean"}:
        if direct_predictor is None:
            raise ValueError("Native direct predictor requires its separately identified checkpoint")
        output = predict_direct_mean(direct_predictor, kwargs, model.field_rollout_operator.quantized_mddm_rollout_operator)
        if kind == "transolver_fitted_residual":
            from Libs.utils.evaluation.native_transolver_calibration import predict_fitted_residual, residual_policy
            output = predict_fitted_residual(output, kwargs, model.field_rollout_operator.quantized_mddm_rollout_operator,
                model.field_rollout_operator.field_normalization, residual_context["grid"],
                residual_context["artifact"], residual_policy(config), episode_uid=records[0]["teacher"]["sample_uid"])

    else:
        raise ValueError("Unsupported native evaluation predictor_kind")
    output["initial_conditioning_wall_seconds"] = conditioning_seconds
    return output


def load_native_conditioning_checkpoint(config: dict, checkpoint_path: Path, actions: list[str]):
    """Keep the existing source-profile/seed/policy/feature authority for every comparator."""
    policy = metric_policy(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    binding = screening_training_seed_binding(config, policy["checkpoint_scope"], policy["checkpoint_split_name"])
    if not isinstance(checkpoint.get("training_seed_binding"), dict):
        raise ValueError("Native evaluation checkpoint requires its original training seed binding")
    validate_training_seed_checkpoint(checkpoint, binding)
    if checkpoint["training_seed_binding"] != binding:
        raise ValueError("Native checkpoint/config seed binding mismatch")
    configured_policy = screening_model_kwargs_from_config(config)["rollout_update_policy"]
    if checkpoint["model_kwargs"]["rollout_update_policy"] != configured_policy:
        raise ValueError("Native checkpoint/config action/operator/sampler policy mismatch")
    features, feature_context, feature_paths = _native_training_features(
        config, checkpoint_path, checkpoint, binding, actions)
    model = load_screening_model(checkpoint_path, config, checkpoint_payload=checkpoint)
    return model, binding, features, feature_context, feature_paths


def evaluate_native_rollout_prediction(config: dict, checkpoint_path: Path) -> tuple[dict, list[Path]]:
    policy = metric_policy(config)
    if policy["source_role"] != "development":
        raise ValueError("Native rollout scoring is bound to explicitly selected development sources")
    samples, actions = policy["sample_uids"], policy["regimen_uids"]
    if (not samples or not actions or len(set(samples)) != len(samples) or len(set(actions)) != len(actions)
            or not policy["checkpoint_scope"] or not policy["checkpoint_split_name"]):
        raise ValueError("Native evaluation requires explicit sample/action and checkpoint scope selection")
    registry = native_data.load_registry(config)
    records = native_data.select_records(registry, config, "development", samples, actions)
    episodes = [records[index * len(actions):(index + 1) * len(actions)] for index in range(len(samples))]
    identities = [_native_episode_identity(rows, registry, policy, config) for rows in episodes]
    # Source role/protection/identity admission precedes checkpoint and array payloads.
    direct = policy["predictor_kind"] in {"transolver_direct_mean", "transolver_fitted_residual", "linear_dynamics_direct_mean"}
    mean_only = policy["predictor_kind"] in {"transolver_direct_mean", "linear_dynamics_direct_mean"}
    predictor = conditioning = residual_context = None
    if direct:
        from Libs.utils.model.native_direct_mean import load_direct_predictor
        loaded = load_direct_predictor(config, checkpoint_path, actions)
        predictor, conditioning, model = loaded["predictor"], loaded["conditioning"], loaded["model"]
        binding, predictor_binding = loaded["binding"], loaded["predictor_binding"]
        features, feature_context, feature_paths = loaded["features"], loaded["feature_context"], loaded["paths"]
        if not mean_only:
            from Libs.utils.evaluation.native_transolver_calibration import (
                load_residual_artifact, prediction_binding, residual_policy, _source_identity)
            residual_settings = residual_policy(config)
            residual_path = resolve_repo_path(residual_settings["frozen_artifact_path"])
            artifact = load_residual_artifact(residual_path, config,
                prediction_binding(config, checkpoint_path, loaded, registry["grid"]))
            sources = [_source_identity(row, residual_settings) for row in registry["sources"]
                       if row["source_uid"] in {identity["source_uid"] for identity in identities}]
            if any(source not in artifact["admission"]["source_identities"] for source in sources):
                raise ValueError("Prediction source/readout domain is outside the frozen residual law")
            residual_context = {"artifact": artifact, "grid": registry["grid"]}
            feature_paths = [*feature_paths, residual_path]
    else:
        model, binding, features, feature_context, feature_paths = load_native_conditioning_checkpoint(config, checkpoint_path, actions)
    field = model.field_rollout_operator
    sampler = field.quantized_mddm_rollout_operator
    if sampler is None or not field.field_normalization.active:
        raise ValueError("Native scoring requires actual predictive draws and fixed R60 normalization")
    point_mass = (policy["predictor_kind"] == "initial_state_persistence"
                  and registry.get("posterior_family") != "finite_joint_particles")
    protocol = metric_protocol(policy, sampler.quantile_levels, field.field_normalization.identity,
        draw_semantics="mean_only" if mean_only else "deterministic_point_mass" if point_mass else "emitted_whole_particle_mixture_draws")
    paths = {native_data.registry_path(config), checkpoint_path, Path(field.field_normalization.spec["artifact_ref"])}
    paths.update(feature_paths)
    if direct:
        paths.update(conditioning["paths"])
    scored = []
    with torch.no_grad():
        for current, identity in zip(episodes, identities):
            draw_count = 0
            try:
                mask = _native_validity_mask(current, registry["grid"])
                conditioning_arrays, bindings = native_data.load_records(registry, current, config, include_teacher=False)
                paths.update(Path(path) for path in bindings)
                prediction_started = time.perf_counter()
                prediction = (predict_native_episode(model, current, config, conditioning_arrays, features, direct_predictor=predictor, residual_context=residual_context)
                              if direct else predict_native_episode(model, current, config, conditioning_arrays, features))
                prediction_seconds = time.perf_counter() - prediction_started
                predicted = prediction.get("field_trajectory" if point_mass or mean_only else "field_trajectory_samples")
                episode_axis = 0 if point_mass or mean_only else 1
                if (not torch.is_tensor(predicted) or predicted.ndim != (6 if point_mass or mean_only else 7)
                        or predicted.shape[episode_axis] != 1):
                    raise ValueError("Native predictor did not emit its declared field law/episode axes")
                draw_count = 0 if point_mass or mean_only else int(predicted.shape[0])
                identity = dict(identity, prediction_provenance={
                    "posterior_family": registry.get("posterior_family", "deterministic_compatibility"),
                    "posterior_uid": prediction.get("posterior_uid"),
                    "particle_uids": prediction.get("particle_uids"),
                    "particle_draw_indices": prediction["particle_draw_indices"].tolist() if "particle_draw_indices" in prediction else None,
                    "particle_weights": prediction["particle_weights"].tolist() if "particle_weights" in prediction else None,
                    "prediction_law": prediction.get("prediction_law"),
                    **({"observable_prediction": json_value(prediction["observable_prediction"])}
                       if "observable_prediction" in prediction else {}),
                    **({key: prediction[key] for key in ("residual_rng", "raw_negative_value_count", "physical_state_certification")}
                       if residual_context else {}),
                    **({key:prediction[key] for key in ("prediction_kind", "residual_likelihood_status",
                        "predictive_law_draw_count", "posterior_evaluation_draw_count", "conditional_mean_reduction")} if direct else {}),
                    "initial_conditioning_wall_seconds": prediction["initial_conditioning_wall_seconds"],
                    "prediction_wall_seconds": prediction_seconds})
                # The independent teacher is first read and passed to scoring after prediction.
                targets, bindings = native_data.load_records(registry, current, config)
                paths.update(Path(path) for path in bindings)
                score = score_mean_only_episode if mean_only else score_point_mass_episode if point_mass else score_native_episode
                scored.append(score(predicted[0] if point_mass or mean_only else predicted[:, 0], torch.from_numpy(targets["trajectory"]),
                    mask, identity, field.field_normalization, registry["grid"], protocol))
            except (ValueError, OSError, TypeError) as error:
                scored.append(failed_native_episode(identity, str(error), draw_count=draw_count))
    provenance = {"checkpoint_path": str(checkpoint_path), "training_seed_binding": binding,
                  "predictor_kind": policy["predictor_kind"],
                  "loaded_checkpoint_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "source_registry": str(native_data.registry_path(config)),
                  "selected_experiment_family": config.get("selected_experiment_family"),
                  "source_role": "development", "source_grid": registry["grid"],
                  "normalization_id": field.field_normalization.identity,
                  "regimen_feature_context": feature_context}
    if direct:
        provenance.update(conditioning_checkpoint=conditioning["identity"],
            predictor_checkpoint={"checkpoint_path":str(checkpoint_path), "training_seed_binding":predictor_binding,
                                  "trainable_parameter_count":sum(p.numel() for p in predictor.parameters())},
            prediction_kind="mean_only" if mean_only else "fitted_teacher_readout_law",
            residual_likelihood_status="unbound" if mean_only else "fitted")
        # Preserve the prior field as the upstream training seed; name its role.
        provenance["training_seed_binding_role"] = "shared_conditioning"
        if residual_context:
            provenance.update(residual_artifact_sha256=residual_settings["frozen_artifact_sha256"],
                residual_law=residual_context["artifact"]["law"],
                frozen_fit_counts=residual_context["artifact"]["counts"],
                frozen_fit_cost=residual_context["artifact"]["cost"])
    return build_native_metric_payload(scored, protocol, provenance, config), sorted(paths, key=str)

def evaluate_native_causal_prefix(config: dict, checkpoint_path: Path) -> tuple[dict, list[Path]]:
    policy = causal_prefix_policy(config)
    base = metric_policy(config)
    declarations = policy["contrasts"]
    ids = [row.get("contrast_uid") for row in declarations if isinstance(row,dict) and isinstance(row.get("contrast_uid"),str)]
    duplicates = {uid for uid,count in Counter(ids).items() if count>1}
    rows = [contrast_row(i,row,None) for i,row in enumerate(declarations)]
    episode_costs, candidates = {}, {}
    paths = {native_data.registry_path(config),checkpoint_path}
    provenance = {"checkpoint_path":str(checkpoint_path), "predictor_kind":"native_diffusion",
                  "selected_experiment_family":config.get("selected_experiment_family"),
                  "source_role":"development", "source_registry":str(native_data.registry_path(config)),
                  "teacher_outcome_arrays_read":False, "loaded_checkpoint_parameter_count":None,
                  "sampler_budget":None, "checkpoint_load_wall_seconds":0.}
    errors = (ValueError,TypeError,KeyError,OSError,RuntimeError,FloatingPointError)
    try:
        registry = native_data.load_registry(config)
        grid = registry["grid"]
        provenance["source_grid"] = grid
    except errors as error:
        for row in rows: row["reason"] = str(error)
        return build_prefix_payload(rows,episode_costs,provenance,config),sorted(paths,key=str)
    for row,declaration in zip(rows,declarations):
        try:
            contrast = declared_contrast(declaration,config,duplicates)
            current = native_data.select_records(registry,config,"development",[contrast["sample_uid"]],contrast["regimen_uids"])
            identity = _native_episode_identity(current,registry,base,config)
            row.update(source_uid=identity["source_uid"],family_uid=identity["family_uid"],episode_identity=identity)
            match_record_conditions(current)
            mask = _native_validity_mask(current,grid)
            common_prefix_mask(mask,grid,contrast["future_knot_hours"])
            candidates.setdefault(contrast["sample_uid"],[]).append((row,contrast,mask))
        except errors as error:
            row["reason"] = str(error)
    if not candidates:
        return build_prefix_payload(rows,episode_costs,provenance,config),sorted(paths,key=str)
    action_set = {action for group in candidates.values() for _,contrast,_ in group for action in contrast["regimen_uids"]}
    selected_actions = [action for action in base["regimen_uids"] if action in action_set]
    started = time.perf_counter()
    try:
        model,binding,features,feature_context,feature_paths = load_native_conditioning_checkpoint(config,checkpoint_path,selected_actions)
        field = model.field_rollout_operator
        sampler = field.quantized_mddm_rollout_operator
        if sampler is None or field.neuronsifter_world_action_operator is None or not field.field_normalization.active:
            raise ValueError("Causal prefix requires the real native stochastic operator and fixed R60")
        field.field_normalization.assert_binding(grid)
        provenance.update(training_seed_binding=binding,regimen_feature_context=feature_context,
            loaded_checkpoint_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            normalization_id=field.field_normalization.identity,
            sampler_budget={key:getattr(sampler,key) for key in
                ("inference_sample_count","inference_denoising_step_count","sample_chunk_size","denoising_token_chunk_size","sampler_seed")})
        paths.update(feature_paths)
        paths.add(Path(field.field_normalization.spec["artifact_ref"]))
        for row in rows: row["configured_draw_count"] = sampler.inference_sample_count
    except errors as error:
        for group in candidates.values():
            for row,_,_ in group: row["reason"] = str(error)
        provenance["checkpoint_load_wall_seconds"] = time.perf_counter()-started
        return build_prefix_payload(rows,episode_costs,provenance,config),sorted(paths,key=str)
    provenance["checkpoint_load_wall_seconds"] = time.perf_counter()-started
    with torch.no_grad():
        for sample,group in candidates.items():
            cost = {"native_prediction_calls":0,"conditioning_wall_seconds":0.,"prediction_wall_seconds":0.,"statistics_wall_seconds":0.}
            episode_costs[sample] = cost
            tick = time.perf_counter()
            try:
                requested = {action for _,contrast,_ in group for action in contrast["regimen_uids"]}
                union = [action for action in selected_actions if action in requested]
                current = native_data.select_records(registry,config,"development",[sample],union)
                _native_episode_identity(current,registry,base,config)
                match_record_conditions(current)
                arrays,_ = native_data.load_records(registry,current,config,include_teacher=False)
                prepared = prepare_native_episode(model,current,config,arrays)
                cost["conditioning_wall_seconds"] = time.perf_counter()-tick
                ready = []
                for row,contrast,mask in group:
                    try:
                        matched = match_prepared_pair(prepared,current,contrast["regimen_uids"],grid,
                                                      contrast["future_knot_hours"],mask,field)
                        row.update(prepared_binding=matched,prefix_time_count=matched["prefix_time_count"],
                                   valid_coordinate_count=matched["valid_coordinate_count"])
                        if not matched["valid_coordinate_count"]:
                            row.update(status="undefined",reason="empty_post_initial_prefix",
                                       mc_uncertainty_status="unavailable_undefined_statistic")
                        else:
                            ready.append((row,contrast,mask))
                    except errors as error:
                        row["reason"] = str(error)
                if not ready:
                    continue
                retained = {action for _,contrast,_ in ready for action in contrast["regimen_uids"]}
                sampled_actions = [action for action in union if action in retained]
                positions = [union.index(action) for action in sampled_actions]
                sampled_records = [current[index] for index in positions]
                patient,kwargs = prepared
                kwargs = dict(kwargs,regimen_ids=prepared[1]["regimen_ids"][positions])
                if "joint_posterior" in kwargs:
                    kwargs["regimen_uids"] = sampled_actions
                else:
                    kwargs["diffusion_context"] = {key:value[:,positions] for key,value in kwargs["diffusion_context"].items()}
                sampled_arrays = {key:value[positions] for key,value in arrays.items()}
                selected_features = features[[selected_actions.index(action) for action in sampled_actions]]
                tick = time.perf_counter()
                cost["native_prediction_calls"] = 1
                prediction = predict_native_episode(model,sampled_records,config,sampled_arrays,selected_features,prepared=(patient,kwargs))
                cost["prediction_wall_seconds"] = time.perf_counter()-tick
                draws = prediction.get("field_trajectory_samples")
                expected = (1,len(sampled_actions),*(len(grid[key]) for key in ("time_hours","region_id","compartment_id","species_id")))
                if not torch.is_tensor(draws) or draws.ndim != 7 or tuple(draws.shape[1:]) != expected:
                    raise ValueError("Native causal-prefix consumer requires emitted full-field ensembles")
                realized = int(draws.shape[0])
                for row,_,_ in ready: row["realized_draw_count"] = realized
                if realized != sampler.inference_sample_count:
                    raise ValueError("Emitted native draw count differs from its configured budget")
                posterior = kwargs.get("joint_posterior")
                if posterior is not None:
                    indices = prediction.get("particle_draw_indices")
                    particle_draws = prediction.get("particle_trajectory_samples")
                    if (prediction.get("posterior_uid") != posterior.posterior_uid
                            or prediction.get("particle_uids") != posterior.support.particle_uids
                            or not torch.equal(prediction["particle_weights"],posterior.weights)
                            or not torch.is_tensor(indices) or indices.dtype != torch.long or indices.shape != (1,realized)
                            or not ((indices>=0)&(indices<posterior.weights.shape[1])).all()
                            or not torch.is_tensor(particle_draws)):
                        raise ValueError("Native emitted whole-particle identity mismatch")
                    expected_draws = torch.stack([particle_draws[di,0,indices[0,di]] for di in range(realized)])
                    if not torch.equal(draws[:,0],expected_draws):
                        raise ValueError("Native emitted mixture did not retain the complete draw tuples")
                std = prediction.get("field_trajectory_std")
                uncertainty = {"status": "missing"}
                if torch.is_tensor(std):
                    support = prediction.get("quantity_support", torch.ones_like(std, dtype=torch.bool))
                    count = support.sum((0, 2, 3, 4))
                    means = (std * support).sum((0, 2, 3, 4)) / count.clamp_min(1)
                    maxima = std.masked_fill(~support, 0).amax((0, 2, 3, 4))
                    def supported_values(value):
                        return [[number if present else None for number, present in zip(row, mask)]
                                for row, mask in zip(value.tolist(), (count > 0).tolist())]
                    uncertainty = {"status": "available", "mean_by_action_species": supported_values(means),
                        "max_by_action_species": supported_values(maxima), "supported_cell_count": count}
                details = json_value({"sampled_regimen_uids":sampled_actions,
                    "sample_ids":kwargs["sample_ids"],"regimen_ids":kwargs["regimen_ids"],
                    "posterior_uid":prediction.get("posterior_uid"),
                    "particle_uids":prediction.get("particle_uids"),
                    "particle_weights":prediction.get("particle_weights"),
                    "particle_draw_indices":prediction.get("particle_draw_indices"),
                    "sampler_budget":provenance["sampler_budget"],
                    "coupling":dict(COUPLING,whole_particle=COUPLING["whole_particle"] if posterior is not None else "not_applicable_point_conditioning"),
                    "diagnostic_scope":"entire_sampled_action_union",
                    "uncertainty":uncertainty,"quantile_levels":sampler.quantile_levels,
                    "router_diagnostics":prediction.get("router_diagnostics",{"status":"missing"}),
                    "particle_operator_splitting_diagnostics":prediction.get("particle_operator_splitting_diagnostics"),
                    "field_normalization_diagnostics":prediction.get("field_normalization_diagnostics"),
                    "observable_prediction":prediction.get("observable_prediction"),
                    "local_membrane":prediction.get("local_membrane"),
                    "local_membrane_status":prediction.get("local_membrane_status")})
                json.dumps(details,allow_nan=False)
                tick = time.perf_counter()
                for row,contrast,mask in ready:
                    row["prediction_provenance"] = details
                    try:
                        pair = [sampled_actions.index(action) for action in contrast["regimen_uids"]]
                        row.update(score_prefix_pair(draws[:,0,pair],mask,grid,contrast["future_knot_hours"],field.field_normalization))
                    except errors as error:
                        row["reason"] = str(error)
                cost["statistics_wall_seconds"] = time.perf_counter()-tick
            except errors as error:
                for row,_,_ in group:
                    if row["status"] == "failed": row["reason"] = str(error)
                if cost["native_prediction_calls"]:
                    cost["prediction_wall_seconds"] = time.perf_counter()-tick
                else:
                    cost["conditioning_wall_seconds"] = time.perf_counter()-tick
    return build_prefix_payload(rows,episode_costs,provenance,config),sorted(paths,key=str)


def load_training_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def discovered_ensemble_checkpoint_paths(
    config: dict,
    fold_name: str,
    current_checkpoint_path: Path,
    ensemble_mode: str,
) -> list[Path]:
    if ensemble_mode == "single":
        return [current_checkpoint_path]
    ensemble_config = config["screening_pipeline"]["screening_benchmark"]["ensemble_uq"]
    discovery_config = ensemble_config["current_mainline_discovery"]
    allowed_fold_names = {str(value) for value in discovery_config["allowed_fold_names"]}
    if fold_name not in allowed_fold_names:
        if ensemble_mode == "required":
            raise ValueError(f"Fold {fold_name} is not enabled for current-mainline ensemble discovery.")
        return [current_checkpoint_path]
    benchmark_name_template = str(discovery_config["benchmark_name_template"])
    member_seed_values = [int(value) for value in ensemble_config["member_seed_values"]]
    allowed_benchmark_names = {
        benchmark_name_template.format(seed=seed_value)
        for seed_value in member_seed_values
    }
    current_benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    if current_benchmark_name not in allowed_benchmark_names:
        if ensemble_mode == "required":
            raise ValueError(
                f"Benchmark {current_benchmark_name} does not match current-mainline discovery template {benchmark_name_template}."
            )
        return [current_checkpoint_path]
    artifact_root = (REPO_ROOT / str(discovery_config["artifact_root"])).resolve()
    manifest_relative_path = str(discovery_config["manifest_relative_path"])
    checkpoint_relative_path = str(discovery_config["checkpoint_relative_path"])
    expected_model_name = str(discovery_config["expected_model_name"])
    expected_model_contract_version = str(discovery_config["expected_model_contract_version"])
    discovered_paths: list[Path] = []
    for seed_value in member_seed_values:
        benchmark_name = benchmark_name_template.format(seed=seed_value)
        manifest_path = artifact_root / benchmark_name / manifest_relative_path.format(fold_name=fold_name)
        checkpoint_path = artifact_root / benchmark_name / checkpoint_relative_path.format(fold_name=fold_name)
        if not manifest_path.exists() or not checkpoint_path.exists():
            continue
        manifest = load_training_manifest(manifest_path)
        if str(manifest.get("fold_name", "")) != fold_name:
            continue
        if str(manifest.get("benchmark_name", "")) != benchmark_name:
            continue
        if str(manifest.get("model_name", "")) != expected_model_name:
            continue
        if str(manifest.get("model_contract_version", "")) != expected_model_contract_version:
            continue
        discovered_paths.append(checkpoint_path)
    minimum_member_count = int(ensemble_config["minimum_member_count"])
    if len(discovered_paths) >= minimum_member_count:
        return discovered_paths
    if ensemble_mode == "required":
        raise ValueError(
            f"Current-mainline ensemble discovery for fold {fold_name} found {len(discovered_paths)} members, expected at least {minimum_member_count}."
        )
    return [current_checkpoint_path]


def final_regimen_output(
    model: ADScreeningSystem,
    patient_state: torch.Tensor,
    regimen_feature_tensor: torch.Tensor,
    observation_mask: torch.Tensor,
    tri_context: dict[str, torch.Tensor],
    exposure_target_tensor: torch.Tensor,
    occupancy_target_tensor: torch.Tensor,
    pkpd_feasibility_tensor: torch.Tensor,
    config: dict,
    regimen_chunk_size: int,
) -> dict[str, torch.Tensor]:
    initial_output = model.score_regimens(
        patient_state,
        regimen_feature_tensor,
        tri_context,
        observation_mask=observation_mask,
        regimen_chunk_size=regimen_chunk_size,
        include_auxiliary_outputs=False,
    )
    _, pkpd_proxy_vector = pkpd_prior_loss(
        initial_output,
        exposure_target_tensor,
        occupancy_target_tensor,
        pkpd_feasibility_tensor,
        config,
    )
    expanded_tri_context = dict(tri_context)
    expanded_tri_context["pkpd_physics_consistency"] = pkpd_proxy_vector.unsqueeze(0).expand(
        patient_state.shape[0],
        -1,
    ).to(dtype=torch.float32)
    return model.score_regimens(
        patient_state,
        regimen_feature_tensor,
        expanded_tri_context,
        observation_mask=observation_mask,
        regimen_chunk_size=regimen_chunk_size,
        include_auxiliary_outputs=False,
    )


def ensemble_probability_summary(member_score_tensor: np.ndarray) -> dict[str, np.ndarray]:
    member_probabilities = 1.0 / (1.0 + np.exp(-member_score_tensor.astype(np.float64)))
    member_count, sample_count, candidate_count = member_probabilities.shape
    summary = ensemble_uncertainty_summary(member_probabilities.reshape(member_count, sample_count * candidate_count))
    return {
        key: value.reshape(sample_count, candidate_count)
        for key, value in summary.items()
    }


def stable_split(value: str, fraction: float) -> bool:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) % 1000) / 1000.0 < fraction


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    gains = (2 ** y_true[order] - 1) / np.log2(np.arange(order.size) + 2)
    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = (2 ** y_true[ideal_order] - 1) / np.log2(np.arange(ideal_order.size) + 2)
    denominator = ideal_gains.sum()
    if denominator <= 0:
        return 0.0
    return float(gains.sum() / denominator)


def result_row(
    benchmark_name: str,
    disease_program: str,
    split_name: str,
    evaluation_scope: str,
    model_name: str,
    modality: str,
    sample_count: int,
    metric_name: str,
    metric_value: float,
    artifact_path: str,
    note: str,
    task_name: str = "",
    task_family: str = "",
    case_study_name: str = "",
    is_case_study_anchor: str = "no",
) -> dict[str, object]:
    resolved_task_name, resolved_task_family = SCREENING_TASK_LOOKUP.get(evaluation_scope, ("", ""))
    task_name = task_name or resolved_task_name
    task_family = task_family or resolved_task_family
    return {
        "benchmark_name": benchmark_name,
        "disease_program": disease_program,
        "baseline_family": "action_conditioned_twin",
        "split_name": split_name,
        "evaluation_scope": evaluation_scope,
        "task_name": task_name,
        "task_family": task_family,
        "model_name": model_name,
        "modality": modality,
        "sample_count": sample_count,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "artifact_path": artifact_path,
        "case_study_name": case_study_name,
        "scale_classification": "",
        "is_case_study_anchor": is_case_study_anchor,
        "note": note,
    }


def strict_case_result_rows(
    benchmark_name: str,
    disease_program: str,
    split_name: str,
    evaluation_scope: str,
    model_name: str,
    modality: str,
    sample_count: int,
    artifact_path: str,
    note: str,
    case_study_name: str,
    metrics: dict[str, float],
) -> list[dict[str, object]]:
    return [
        result_row(
            benchmark_name,
            disease_program,
            split_name,
            evaluation_scope,
            model_name,
            modality,
            sample_count,
            metric_name,
            metric_value,
            artifact_path,
            note,
            case_study_name=case_study_name,
            is_case_study_anchor="",
        )
        for metric_name, metric_value in metrics.items()
    ]


def normalized_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = np.maximum(np.abs(target).mean(), 1e-6)
    return float(np.sqrt(np.mean((prediction - target) ** 2)) / denominator)


def align_time_horizon(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.ndim != target_array.ndim:
        raise ValueError(
            f"Prediction and target must have the same rank for horizon alignment, got {prediction_array.shape} and {target_array.shape}."
        )
    if prediction_array.ndim < 3:
        return prediction_array, target_array
    shared_horizon = min(prediction_array.shape[2], target_array.shape[2])
    if shared_horizon <= 0:
        raise ValueError(
            f"Prediction and target must share a positive rollout horizon, got {prediction_array.shape} and {target_array.shape}."
        )
    return prediction_array[:, :, :shared_horizon, ...], target_array[:, :, :shared_horizon, ...]


def patient_state_array(payload: dict[str, object]) -> np.ndarray:
    if "state_features" not in payload:
        raise KeyError("Evaluation payload is missing state_features under the multiscale patient-state contract.")
    return np.asarray(payload["state_features"], dtype=np.float32)


def regimen_chunk_size(config: dict) -> int:
    return int(config["screening_pipeline"]["screening_benchmark"].get("execution_controls", {}).get("regimen_chunk_size", 0))


def screening_task_lookup(config: dict) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    task_catalog = config["screening_pipeline"]["screening_benchmark"].get("paper_tasks", {})
    for task_name, task_payload in task_catalog.items():
        task_family = str(task_payload.get("task_family", ""))
        for evaluation_scope in task_payload.get("evaluation_scopes", []):
            lookup[str(evaluation_scope)] = (str(task_name), task_family)
    return lookup


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    global SCREENING_TASK_LOOKUP
    SCREENING_TASK_LOOKUP = screening_task_lookup(config)
    disease_program = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    config["screening_pipeline"]["screening_benchmark"]["active_disease_program"] = disease_program
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    target_fraction = float(config["screening_pipeline"]["screening_benchmark"]["target_family_holdout_fraction"])
    scaffold_fraction = float(config["screening_pipeline"]["screening_benchmark"]["scaffold_holdout_fraction"])
    tri_threshold = float(config["screening_pipeline"]["screening_benchmark"]["tri_threshold"])
    uq_config = config["screening_pipeline"]["screening_benchmark"]["uq"]
    temperature_grid = [float(value) for value in uq_config["temperature_grid"]]
    ece_bin_count = int(uq_config["ece_bin_count"])
    ace_bin_count = int(uq_config["ace_bin_count"])
    risk_coverage_grid_size = int(uq_config["risk_coverage_grid_size"])
    target_acceptance_coverage = float(uq_config["target_acceptance_coverage"])
    uncertainty_metric_name = str(
        config["screening_pipeline"]["screening_benchmark"]["ensemble_uq"]["current_mainline_discovery"]["uncertainty_metric"]
    )
    action_chunk_size = regimen_chunk_size(config)

    action_root = Path(config["screening_pipeline"]["action_ready_registry"]["local_root"]).resolve()
    target_program_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["target_program_catalog"])
    drug_target_root = Path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]).resolve()
    target_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"])
    compound_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"])
    intervention_root = Path(config["twin"]["intervention_program_registry"]["local_root"]).resolve()
    intervention_rows = read_tsv(intervention_root / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"])
    intervention_row_by_regimen = {
        str(row["regimen_uid"]): row
        for row in intervention_rows
    }
    target_uid_to_entity = target_uid_to_entity_lookup(target_rows)
    manifest = assert_clean_action_ready_manifest(config)
    strict_case_rows = strict_case_entries(config, disease_program)
    pipeline_logger.progress(
        "action_ready_manifest_validated",
        config_hash=str(manifest["config_hash"]),
        strict_case_count=len(strict_case_rows),
    )

    fold_rows = [
        row
        for row in read_tsv(Path(config["benchmark_registry"]["local_root"]).resolve() / config["benchmark_registry"]["relative_paths"]["fold_catalog"])
        if row.get("disease_program", disease_program) == disease_program
    ]
    result_rows: list[dict[str, object]] = []
    candidate_priority_rows: list[dict[str, object]] = []
    capability_rows: list[dict[str, object]] = []
    world_state_frame = optional_tsv_frame(
        Path(config["data"]["world_state_ready_registry"]["local_root"]).resolve()
        / config["data"]["world_state_ready_registry"]["relative_paths"]["world_state_catalog"]
    )
    crossmodal_anchor_frame = optional_tsv_frame(
        Path(config["data"]["crossmodal_anchor_ready_registry"]["local_root"]).resolve()
        / config["data"]["crossmodal_anchor_ready_registry"]["relative_paths"]["crossmodal_anchor_catalog"]
    )
    simulation_teacher_frame = optional_tsv_frame(
        Path(config["data"]["simulation_teacher_ready_registry"]["local_root"]).resolve()
        / config["data"]["simulation_teacher_ready_registry"]["relative_paths"]["simulation_teacher_catalog"]
    )
    counterfactual_benchmark_frame = optional_tsv_frame(
        Path(config["data"]["counterfactual_benchmark_ready_registry"]["local_root"]).resolve()
        / config["data"]["counterfactual_benchmark_ready_registry"]["relative_paths"]["counterfactual_benchmark_catalog"]
    )
    feedback_event_frame = optional_tsv_frame(
        Path(config["data"]["feedback_ready_registry"]["local_root"]).resolve()
        / config["data"]["feedback_ready_registry"]["relative_paths"]["feedback_event_catalog"]
    )
    model_name_for_results = "ad_screening_system"
    for fold_row in fold_rows:
        if args.fold_name and fold_row["fold_name"] != args.fold_name:
            continue
        config["benchmark_registry"]["benchmark_name"] = str(fold_row["benchmark_name"])
        current_artifact_dir = artifact_dir(config, fold_row["fold_name"])
        checkpoint_path = current_artifact_dir / "model_weights.pt"
        if not checkpoint_path.exists():
            continue
        ensemble_checkpoint_paths = discovered_ensemble_checkpoint_paths(
            config,
            fold_row["fold_name"],
            checkpoint_path,
            args.ensemble_mode,
        )
        target_frame = pd.read_parquet(current_artifact_dir / "selected_target_programs.parquet")
        regimen_frame = pd.read_parquet(current_artifact_dir / "selected_regimens.parquet")
        for case_row in strict_case_rows:
            assert_no_forced_anchor_regimens(regimen_frame, case_row, compound_rows, f"{fold_row['fold_name']}:{case_row['case_study_name']}")
        target_features_frame, target_feature_matrix = target_program_features(target_frame.to_dict("records"))
        regimen_features_frame, regimen_feature_matrix = regimen_features(regimen_frame.to_dict("records"))
        target_feature_tensor = torch.tensor(target_feature_matrix, dtype=torch.float32)
        regimen_feature_tensor = torch.tensor(regimen_feature_matrix, dtype=torch.float32)
        models = [load_model(member_checkpoint_path, config) for member_checkpoint_path in ensemble_checkpoint_paths]
        ensemble_artifact_path = str(ensemble_checkpoint_paths[0].relative_to(REPO_ROOT))
        ensemble_member_count = len(models)

        simulation_targets_by_condition = selected_regimen_simulation_tensors(config, regimen_features_frame)
        exposure_target_tensor, occupancy_target_tensor, pkpd_feasibility_tensor = pkpd_reference_tensors(regimen_features_frame, config, torch.device("cpu"))
        train_payload = fold_patient_latents(config, fold_row["fold_name"], "train", disease_program, fold_row["benchmark_name"])
        validation_payload = fold_patient_latents(config, fold_row["fold_name"], "validation", disease_program, fold_row["benchmark_name"])
        test_payload = fold_patient_latents(config, fold_row["fold_name"], "test", disease_program, fold_row["benchmark_name"])

        for modality, payload in test_payload.items():
            patient_state_matrix = patient_state_array(payload)
            if patient_state_matrix.shape[0] == 0:
                continue
            patient_state = torch.tensor(patient_state_matrix, dtype=torch.float32)
            patient_labels = payload["labels"].astype(np.int64)
            observation_mask = torch.tensor(observed_region_mask(payload["metadata"], config), dtype=torch.float32)

            validation_state_matrix = patient_state_array(validation_payload[modality])
            if validation_state_matrix.shape[0] > 0:
                validation_observation_mask = torch.tensor(
                    observed_region_mask(validation_payload[modality]["metadata"], config),
                    dtype=torch.float32,
                )
                validation_logits_by_member = [
                    model.infer_state(
                        torch.tensor(validation_state_matrix, dtype=torch.float32),
                        observation_mask=validation_observation_mask,
                    )[0].detach().cpu().numpy().astype(np.float64)
                    for model in models
                ]
                validation_logits_numpy = np.mean(np.stack(validation_logits_by_member, axis=0), axis=0)
                best_temperature, _ = fit_temperature_from_grid(
                    validation_logits_numpy,
                    validation_payload[modality]["labels"],
                    temperature_grid,
                )
            else:
                best_temperature = 1.0

            tri_context = tri_context_tensors(
                payload["probabilities"],
                payload["matrix"],
                payload["latent"],
                regimen_feature_matrix.shape[0],
                torch.device("cpu"),
            )
            member_disease_logits: list[np.ndarray] = []
            member_target_scores: list[np.ndarray] = []
            member_regimen_scores: list[np.ndarray] = []
            member_field_trajectories: list[np.ndarray] = []
            member_tri_scores: list[np.ndarray] = []
            member_pkpd_consistency: list[np.ndarray] = []
            member_interactive_latency_ms: list[float] = []
            for model in models:
                started_at = time.perf_counter()
                disease_logit, _ = model.infer_state(patient_state, observation_mask=observation_mask)
                member_disease_logits.append(disease_logit.detach().cpu().numpy().astype(np.float64))
                target_scores, _ = model.score_targets(
                    patient_state,
                    target_feature_tensor,
                    observation_mask=observation_mask,
                )
                member_target_scores.append(target_scores.detach().cpu().numpy().astype(np.float64))
                regimen_output = final_regimen_output(
                    model,
                    patient_state,
                    regimen_feature_tensor,
                    observation_mask,
                    tri_context,
                    exposure_target_tensor,
                    occupancy_target_tensor,
                    pkpd_feasibility_tensor,
                    config,
                    action_chunk_size,
                )
                member_regimen_scores.append(regimen_output["regimen_scores"].detach().cpu().numpy().astype(np.float64))
                member_field_trajectories.append(regimen_output["field_trajectory"].detach().cpu().numpy().astype(np.float32))
                member_tri_scores.append(regimen_output["tri_score"].detach().cpu().numpy().astype(np.float64))
                member_pkpd_consistency.append(regimen_output["pkpd_consistency"].detach().cpu().numpy().astype(np.float64))
                member_interactive_latency_ms.append((time.perf_counter() - started_at) * 1000.0)

            disease_logit_tensor = np.stack(member_disease_logits, axis=0)
            disease_probability_tensor = 1.0 / (1.0 + np.exp(-disease_logit_tensor))
            disease_uq = ensemble_uncertainty_summary(disease_probability_tensor)
            disease_prob = disease_uq["predictive_mean"].astype(np.float32)
            calibrated_disease_prob = sigmoid_with_temperature(
                disease_logit_tensor.mean(axis=0),
                best_temperature,
            ).astype(np.float32)
            latent_ood_metric = 0.0

            if len(np.unique(patient_labels)) >= 2:
                disease_metrics = {
                    "auroc": float(roc_auc_score(patient_labels, disease_prob)),
                    "auprc": float(average_precision_score(patient_labels, disease_prob)),
                    "balanced_accuracy": float(balanced_accuracy_score(patient_labels, disease_prob >= 0.5)),
                    "macro_f1": float(f1_score(patient_labels, disease_prob >= 0.5, average="macro")),
                }
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "disease_state_inference", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in disease_metrics.items()
                    ]
                )
                calibration_metrics = {
                    "temperature": float(best_temperature),
                    "ece": expected_calibration_error(calibrated_disease_prob, patient_labels, ece_bin_count),
                    "ace": adaptive_calibration_error(calibrated_disease_prob, patient_labels, ace_bin_count),
                    "nll": binary_nll(calibrated_disease_prob, patient_labels),
                    "brier_score": binary_brier(calibrated_disease_prob, patient_labels),
                }
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "disease_state_calibration", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in calibration_metrics.items()
                    ]
                )
                disease_ensemble_metrics = {
                    "ensemble_member_count": float(ensemble_member_count),
                    "predictive_entropy_mean": float(disease_uq["predictive_entropy"].mean()),
                    "mutual_information_mean": float(disease_uq["mutual_information"].mean()),
                    "epistemic_std_mean": float(disease_uq["epistemic_std"].mean()),
                }
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "disease_state_ensemble_uq", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in disease_ensemble_metrics.items()
                    ]
                )
                latent_ood_metric = ood_auroc(
                    np.asarray(train_payload[modality]["latent"], dtype=np.float32),
                    np.asarray(validation_payload[modality]["latent"], dtype=np.float32),
                    np.asarray(payload["latent"], dtype=np.float32),
                )
                result_rows.append(
                    result_row(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "ood_robustness",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        "latent_distance_ood_auroc",
                        float(latent_ood_metric),
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                    )
                )

            entity_lookup = {
                str(entity_id): index
                for index, entity_id in enumerate(payload["entity_index"]["canonical_entity_id"].astype(str).tolist())
            }
            target_labels = patient_target_labels(payload["matrix"], payload["differential"], entity_lookup, target_features_frame)
            regimen_labels = patient_regimen_labels(payload["matrix"], payload["differential"], entity_lookup, regimen_features_frame, target_uid_to_entity)
            target_score_tensor = np.stack(member_target_scores, axis=0)
            regimen_score_tensor = np.stack(member_regimen_scores, axis=0)
            field_trajectory_tensor = np.stack(member_field_trajectories, axis=0)
            tri_score_tensor = np.stack(member_tri_scores, axis=0)
            pkpd_consistency_tensor = np.stack(member_pkpd_consistency, axis=0)
            target_scores = target_score_tensor.mean(axis=0)
            regimen_scores = regimen_score_tensor.mean(axis=0)
            predicted_field = field_trajectory_tensor.mean(axis=0)
            tri_values = tri_score_tensor.mean(axis=0).astype(np.float32)
            pkpd_consistency_values = pkpd_consistency_tensor.mean(axis=0).astype(np.float32)
            regimen_uq = ensemble_probability_summary(regimen_score_tensor)
            interactive_latency_ms = float(np.mean(member_interactive_latency_ms)) if member_interactive_latency_ms else 0.0

            target_ndcg_at_10 = 0.0
            target_score_mean = target_scores.mean(axis=0)
            target_label_mean = target_labels.mean(axis=0)
            heldout_target_mask = target_features_frame["target_class"].astype(str).apply(lambda value: stable_split(value or "unknown", target_fraction)).to_numpy()
            if heldout_target_mask.any():
                target_metrics = {
                    "ndcg_at_10": ndcg_at_k(target_label_mean[heldout_target_mask], target_score_mean[heldout_target_mask], 10),
                    "proxy_mean_score": float(target_score_mean[heldout_target_mask].mean()),
                }
                target_ndcg_at_10 = float(target_metrics["ndcg_at_10"])
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "target_retrieval", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in target_metrics.items()
                    ]
                )

            regimen_ndcg_at_10 = 0.0
            regimen_score_mean = regimen_scores.mean(axis=0)
            regimen_label_mean = regimen_labels.mean(axis=0)
            heldout_regimen_mask = regimen_features_frame["regimen_scaffold_set"].astype(str).apply(lambda value: stable_split(value or "no_scaffold", scaffold_fraction)).to_numpy()
            if heldout_regimen_mask.any():
                regimen_metrics = {
                    "ndcg_at_10": ndcg_at_k(regimen_label_mean[heldout_regimen_mask], regimen_score_mean[heldout_regimen_mask], 10),
                    "pk_feasibility_top_10": float(pd.to_numeric(regimen_features_frame.loc[heldout_regimen_mask, "pk_feasibility_score"], errors="coerce").sort_values(ascending=False).head(10).mean()),
                }
                regimen_ndcg_at_10 = float(regimen_metrics["ndcg_at_10"])
                result_rows.extend(
                    [
                        result_row(benchmark_name, disease_program, fold_row["fold_name"], "regimen_ranking", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                        for metric_name, metric_value in regimen_metrics.items()
                    ]
                )
            regimen_ensemble_metrics = {
                "ensemble_member_count": float(ensemble_member_count),
                "predictive_entropy_mean": float(regimen_uq["predictive_entropy"].mean()),
                "mutual_information_mean": float(regimen_uq["mutual_information"].mean()),
                "epistemic_std_mean": float(regimen_uq["epistemic_std"].mean()),
            }
            result_rows.extend(
                [
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "regimen_ensemble_uq", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                    for metric_name, metric_value in regimen_ensemble_metrics.items()
                ]
            )

            simulation_targets = batch_simulation_targets(simulation_targets_by_condition, patient_labels)
            aligned_predicted_field, aligned_simulation_targets = align_time_horizon(predicted_field, simulation_targets)
            concentration_nrmse = normalized_rmse(aligned_predicted_field[..., 0], aligned_simulation_targets[..., 0])
            occupancy_nrmse = normalized_rmse(aligned_predicted_field[..., 1], aligned_simulation_targets[..., 1])
            flux_nrmse = normalized_rmse(aligned_predicted_field[..., 4], aligned_simulation_targets[..., 4])
            electrophysiology_nrmse = normalized_rmse(aligned_predicted_field[..., 6], aligned_simulation_targets[..., 6])
            result_rows.extend(
                [
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "concentration_nrmse", concentration_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "occupancy_nrmse", occupancy_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "flux_nrmse", flux_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "solver_grounded_dynamics", model_name_for_results, modality, int(patient_labels.shape[0]), "electrophysiology_nrmse", electrophysiology_nrmse, ensemble_artifact_path, fold_row["held_out_study_uid"]),
                ]
            )

            route_exposure = route_ordering_exposure(predicted_field, regimen_features_frame, config)
            route_metric = 0.0
            if all(route_name in route_exposure for route_name in ["intrathecal", "intravenous", "oral"]):
                route_metric = float(route_exposure["intrathecal"] > route_exposure["intravenous"] > route_exposure["oral"])
                result_rows.append(
                    result_row(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "mechanistic_sanity",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        "route_ordering_hit",
                        route_metric,
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                    )
                )

            accept_mask = tri_values >= tri_threshold
            prediction_error = np.abs((1.0 / (1.0 + np.exp(-regimen_scores))) - np.clip(regimen_labels, 0.0, 1.0))
            flattened_curve = risk_coverage_curve(
                prediction_error.reshape(-1),
                tri_values.reshape(-1),
                risk_coverage_grid_size,
            )
            reliability_metrics = {
                "accepted_mean_error": float(prediction_error[accept_mask].mean()) if accept_mask.any() else 0.0,
                "rejected_mean_error": float(prediction_error[~accept_mask].mean()) if (~accept_mask).any() else 0.0,
                "coverage": float(accept_mask.mean()),
                "coverage_gap": coverage_gap(float(accept_mask.mean()), target_acceptance_coverage),
                "acceptance_benefit": acceptance_benefit(prediction_error.reshape(-1), accept_mask.reshape(-1)),
                "risk_coverage_auc": float(flattened_curve["auc"]),
            }
            result_rows.extend(
                [
                    result_row(benchmark_name, disease_program, fold_row["fold_name"], "reliability_abstention", model_name_for_results, modality, int(patient_labels.shape[0]), metric_name, metric_value, ensemble_artifact_path, fold_row["held_out_study_uid"])
                    for metric_name, metric_value in reliability_metrics.items()
                ]
            )

            for validation_row in quantitative_case_validation_rows(config, regimen_features_frame, compound_rows, predicted_field, tri_values):
                for metric_name in [
                    "standardized_effect_error",
                    "biomarker_effect_error",
                    "direction_hit",
                    "rank_recovery_hit",
                    "tri_gate_consistency",
                    "clinical_endpoint_qualitative_presence",
                ]:
                    result_rows.append(
                        result_row(
                            benchmark_name,
                            disease_program,
                            fold_row["fold_name"],
                            "quantitative_case_validation",
                            model_name_for_results,
                            modality,
                            int(patient_labels.shape[0]),
                            metric_name,
                            float(validation_row[metric_name]),
                            ensemble_artifact_path,
                            f"{fold_row['held_out_study_uid']}|{validation_row['intervention_label']}|{validation_row['clinical_endpoint_alignment_summary']}",
                            case_study_name=str(validation_row["case_study_name"]),
                            is_case_study_anchor="yes",
                        )
                    )

            tri_mean = tri_values.mean(axis=0)
            tri_accepted_regimen_mask = tri_mean >= tri_threshold
            for case_row in strict_case_rows:
                target_metrics = compute_anchor_recovery_metrics(
                    target_score_mean.astype(np.float64),
                    target_anchor_mask(target_features_frame, case_row),
                    int(case_row["top_k"]),
                )
                result_rows.extend(
                    strict_case_result_rows(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "strict_case_target_recovery",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                        str(case_row["case_study_name"]),
                        target_metrics,
                    )
                )
                regimen_metrics = compute_anchor_recovery_metrics(
                    regimen_score_mean.astype(np.float64),
                    regimen_anchor_mask(regimen_features_frame, case_row, compound_rows),
                    int(case_row["top_k"]),
                )
                regimen_metrics.update(
                    compute_tri_accepted_anchor_metrics(
                        regimen_score_mean.astype(np.float64),
                        regimen_anchor_mask(regimen_features_frame, case_row, compound_rows),
                        tri_accepted_regimen_mask,
                        int(case_row["top_k"]),
                    )
                )
                regimen_metrics.update(
                    strict_case_dynamics_metrics(
                        config,
                        case_row,
                        regimen_features_frame,
                        compound_rows,
                        predicted_field,
                        tri_accepted_regimen_mask,
                    )
                )
                result_rows.extend(
                    strict_case_result_rows(
                        benchmark_name,
                        disease_program,
                        fold_row["fold_name"],
                        "strict_case_regimen_recovery",
                        model_name_for_results,
                        modality,
                        int(patient_labels.shape[0]),
                        ensemble_artifact_path,
                        fold_row["held_out_study_uid"],
                        str(case_row["case_study_name"]),
                        regimen_metrics,
                    )
                )

            disease_family = disease_family_name(disease_program)
            world_state_rows = world_state_frame.loc[
                (world_state_frame["disease_program"].astype(str) == disease_program)
                & (world_state_frame["observed_modality_set"].astype(str) == modality)
            ].copy() if not world_state_frame.empty else pd.DataFrame()
            crossmodal_anchor_count = int(
                crossmodal_anchor_frame.loc[crossmodal_anchor_frame["disease_family"].astype(str) == disease_family].shape[0]
            ) if not crossmodal_anchor_frame.empty else 0
            simulation_teacher_count = int(
                simulation_teacher_frame.loc[simulation_teacher_frame["disease_program"].astype(str) == disease_program].shape[0]
            ) if not simulation_teacher_frame.empty else 0
            counterfactual_pair_count = int(
                counterfactual_benchmark_frame.loc[counterfactual_benchmark_frame["disease_program"].astype(str) == disease_program].shape[0]
            ) if not counterfactual_benchmark_frame.empty else 0
            feedback_event_count = int(
                feedback_event_frame.loc[feedback_event_frame["disease_program"].astype(str) == disease_program].shape[0]
            ) if not feedback_event_frame.empty else 0
            missing_modality_count = 0.0
            if not world_state_rows.empty and "missing_modality_set" in world_state_rows.columns:
                missing_modality_count = float(
                    world_state_rows["missing_modality_set"].astype(str).apply(lambda value: len([item for item in value.split("|") if item])).mean()
                )
            capability_metrics = {
                "concentration_nrmse": float(concentration_nrmse),
                "flux_nrmse": float(flux_nrmse),
                "route_ordering_hit": float(route_metric) if "route_metric" in locals() else 0.0,
                "electrophysiology_nrmse": float(electrophysiology_nrmse),
                "pkpd_consistency_mean": float(pkpd_consistency_values.mean()),
                "regimen_ndcg_at_10": float(regimen_ndcg_at_10),
                "target_ndcg_at_10": float(target_ndcg_at_10),
                "ood_auroc": float(latent_ood_metric),
                "acceptance_benefit": float(reliability_metrics["acceptance_benefit"]),
                "interactive_latency_ms": float(interactive_latency_ms),
            }
            capability_registries = {
                "factor_count": float(len(config["foundation_twin"]["world_state_factors"])),
                "crossmodal_anchor_count": float(crossmodal_anchor_count),
                "missing_modality_count": float(missing_modality_count),
                "counterfactual_pair_count": float(counterfactual_pair_count),
                "feedback_event_count": float(feedback_event_count),
                "simulation_teacher_count": float(simulation_teacher_count),
            }
            capability_rows.extend(
                capability_catalog_rows(
                    config,
                    benchmark_name,
                    disease_program,
                    disease_family,
                    str(fold_row["fold_name"]),
                    modality,
                    ensemble_artifact_path,
                    capability_metrics,
                    capability_registries,
                )
            )

            active_learning_uncertainty_matrix = regimen_uq[uncertainty_metric_name]
            for regimen_index, regimen_row in enumerate(regimen_features_frame.to_dict(orient="records")):
                intervention_row = intervention_row_by_regimen.get(str(regimen_row["regimen_uid"]), {})
                candidate_priority_rows.append(
                    {
                        "benchmark_name": benchmark_name,
                        "fold_name": str(fold_row["fold_name"]),
                        "modality": modality,
                        "regimen_uid": str(regimen_row["regimen_uid"]),
                        "predicted_score_mean": float(regimen_score_mean[regimen_index]),
                        "tri_score_mean": float(tri_values[:, regimen_index].mean()),
                        "pkpd_consistency_mean": float(pkpd_consistency_values[:, regimen_index].mean()),
                        "predictive_entropy_mean": float(regimen_uq["predictive_entropy"][:, regimen_index].mean()),
                        "mutual_information_mean": float(regimen_uq["mutual_information"][:, regimen_index].mean()),
                        "epistemic_std_mean": float(regimen_uq["epistemic_std"][:, regimen_index].mean()),
                        "active_learning_uncertainty": float(active_learning_uncertainty_matrix[:, regimen_index].mean()),
                        "artifact_path": ensemble_artifact_path,
                        "note": str(intervention_row.get("intervention_program_uid", "")),
                    }
                )

    result_rows.sort(key=lambda row: (row["split_name"], row["evaluation_scope"], row["metric_name"], row["modality"], row["case_study_name"]))
    result_catalog_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["result_catalog"]
    )
    write_tsv(result_catalog_path, config["schemas"]["screening_result_fields"], result_rows)
    candidate_priority_rows.sort(key=lambda row: (row["fold_name"], row["modality"], row["regimen_uid"]))
    candidate_priority_catalog_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["candidate_priority_catalog"]
    )
    write_tsv(
        candidate_priority_catalog_path,
        config["schemas"]["screening_candidate_priority_fields"],
        candidate_priority_rows,
    )
    capability_rows.sort(key=lambda row: (row["fold_name"], row["modality"], row["capability_family"], row["capability_name"], row["metric_name"]))
    capability_catalog_path = (
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["capability_catalog"]
    )
    write_tsv(
        capability_catalog_path,
        config["schemas"]["foundation_capability_fields"],
        capability_rows,
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=result_catalog_path,
        result_row_count=len(result_rows),
    )


if __name__ == "__main__":
    main()
