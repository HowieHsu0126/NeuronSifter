"""Shared frozen conditioning and native direct-mean checkpoint roles."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path
from Libs.utils.data.native_field_normalization import normalization_from_config
from Libs.utils.model.screening_model_config import native_direct_mean_config
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.model.joint_rollout import posterior_draw_indices
from Libs.utils.exp.experiment_manifests import resolve_training_seed, validate_training_seed_checkpoint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_model_identity(config: dict) -> dict:
    policy = native_direct_mean_config(config)
    if not policy:
        raise ValueError("An explicit native-direct predictor configuration is required")
    backbone = copy.deepcopy(config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"])
    backbone["native_direct_mean"] = {key:value for key,value in policy.items()
                                      if key not in {"conditioning_config_path", "conditioning_checkpoint_path"}}
    normalization = normalization_from_config(config)
    identity = {"backbone": backbone, "posterior_family": native.posterior_family(config),
            "normalization_id": normalization.get("sha256"),
            "node_adjacency": node_adjacency_matrix(config).tolist(),
            "source_context_width": int(config["native_diffusion_data"]["posterior_context_width"]),
            "selected_experiment_family": config.get("selected_experiment_family")}
    if backbone["kind"] == "native_linear_dynamics":
        if not normalization.get("artifact") or not normalization.get("sha256"):
            raise ValueError("Linear dynamics requires a bound fixed R60 normalization")
        identity["linear_domain"] = copy.deepcopy(normalization["artifact"]["domain"])
        identity["configured_grid"] = {
            "region_id": [row["region_id"] for row in config["spatial_atlas_registry"]["regions"]],
            "compartment_id": [row["compartment_id"] for row in config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]],
            "species_id": [row["species_id"] for row in config["dynamics_ready_registry"]["species"]]}
        if any(identity["configured_grid"][key] != identity["linear_domain"][key] for key in identity["configured_grid"]):
            raise ValueError("Linear configured grid differs from the fixed R60 domain")
        if identity["source_context_width"] != policy["posterior_context_dim"]:
            raise ValueError("Linear source context width mismatch")
    return identity


def load_frozen_conditioner(config: dict) -> dict:
    # These maintained entrypoints import pretraining helpers; delay their import
    # until module initialization is complete, without duplicating their loaders.
    from Libs.exp.evaluation.evaluate_screening_system import load_native_conditioning_checkpoint

    policy = native_direct_mean_config(config)
    cfg_path = resolve_repo_path(policy["conditioning_config_path"])
    checkpoint_path = resolve_repo_path(policy["conditioning_checkpoint_path"])
    common_config = load_config(cfg_path)
    profile = common_config.get("selected_experiment_family")
    identities = {name for name,row in common_config["paper_grade_full_suite"]["models"].items()
                  if row.get("profile_name") == profile}
    identities.update(row["baseline_id"] for row in common_config["paper_grade_baseline_registry"]["baselines"]
                      if row.get("profile_name") == profile)
    if not profile or len(identities) != 1:
        raise ValueError("Common conditioning requires its original unambiguous profile")
    common_backbone = common_config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    if (common_backbone["kind"] != "neuronsifter_world_action"
            or not native.native_enabled(common_config)
            or native.native_action_policy(common_config) != "occupancy_program"
            or native.posterior_family(common_config) != native.posterior_family(config)):
        raise ValueError("Direct mean requires the matched native occupancy conditioning authority")
    model, binding, _, feature_context, feature_paths = load_native_conditioning_checkpoint(common_config, checkpoint_path, [])
    normalization = model.field_rollout_operator.field_normalization
    if (not normalization.active or normalization.identity != normalization_from_config(config).get("sha256")
            or int(common_backbone["operator_hidden_dim"]) != policy["posterior_context_dim"]
            or not torch.equal(model.field_rollout_operator.node_adjacency.cpu(),
                               torch.tensor(node_adjacency_matrix(config), dtype=torch.float32))):
        raise ValueError("Direct mean conditioning normalization/context/graph identity mismatch")
    model.requires_grad_(False).eval()
    identity = {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
                "config_path": str(cfg_path),
                "effective_config_sha256": hashlib.sha256(json.dumps(common_config,sort_keys=True).encode()).hexdigest(),
                "selected_experiment_family": common_config.get("selected_experiment_family"),
                "training_seed_binding": binding, "normalization_id": normalization.identity,
                "loaded_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "regimen_feature_context": feature_context}
    return {"model":model, "config":common_config, "identity":identity,
            "paths":[cfg_path, checkpoint_path, *feature_paths]}


def direct_checkpoint_metadata(model, config: dict) -> dict:
    registry = native.load_registry(config)
    return {"prediction_kind":"mean_only", "residual_likelihood_status":"unbound",
            "model_identity":direct_model_identity(config), "effective_config":copy.deepcopy(config),
            "conditioning_identity":copy.deepcopy(model.conditioning["identity"]),
            "training_source_registry":str(native.registry_path(config)),
            "training_sources":copy.deepcopy(registry["sources"])}


def validate_direct_checkpoint(checkpoint: dict, config: dict, conditioning: dict) -> None:
    metadata = checkpoint.get("native_direct_mean")
    schema = ("native_transolver_mean_checkpoint" if config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"] == "transolver_microenvironment_operator" else "native_direct_mean_checkpoint")
    required = config["schemas"][schema]["required_fields"]
    if not isinstance(metadata,dict) or not set(required) <= set(metadata):
        raise ValueError("Checkpoint is not an explicitly saved native-direct predictor")
    binding = resolve_training_seed(config,require_selected=True)
    validate_training_seed_checkpoint(checkpoint,binding)
    if checkpoint.get("training_seed_binding") != binding:
        raise ValueError("Direct predictor training seed identity mismatch")
    if (metadata["prediction_kind"] != "mean_only" or metadata["residual_likelihood_status"] != "unbound"
            or metadata["model_identity"] != direct_model_identity(config)
            or metadata["model_identity"] != direct_model_identity(metadata["effective_config"])
            or metadata["conditioning_identity"] != conditioning["identity"]):
        raise ValueError("Direct predictor config/conditioning identity mismatch")


def predict_direct_mean(model, prepared: dict, sampler) -> dict:
    posterior = prepared.get("joint_posterior")
    if posterior is None:
        result = model(None, None, diffusion_batch=prepared.get("diffusion_context"))
        result.update(posterior_evaluation_draw_count=0, conditional_mean_reduction="point_estimate")
        return result
    contexts = native.joint_particle_contexts(posterior, prepared.get("particle_programs"), prepared["regimen_uids"])
    outputs = [model(None, None, diffusion_batch=context) for context in contexts]
    fields = torch.stack([output["field_trajectory"] for output in outputs],1)  # B,J,A,T,R,C,S
    indices = posterior_draw_indices(posterior,sampler,sampler.inference_sample_count)
    evaluations = torch.stack([fields[bi,indices[bi]] for bi in range(fields.shape[0])],1)  # D,B,A,T,R,C,S
    result = {"field_trajectory":evaluations.mean(0), "particle_conditional_means":fields,
            "posterior_mean_evaluations":evaluations, "particle_draw_indices":indices,
            "particle_weights":posterior.weights, "particle_uids":posterior.support.particle_uids,
            "posterior_uid":posterior.posterior_uid, "prediction_kind":"mean_only",
            "residual_likelihood_status":"unbound", "predictive_law_draw_count":0,
            "posterior_evaluation_draw_count":int(indices.shape[1]),
            "conditional_mean_reduction":"mean_over_matched_posterior_indices",
            "predictor_kind":outputs[0].get("predictor_kind", "transolver_direct_mean"),
            "router_diagnostics":[output["router_diagnostics"] for output in outputs]}
    if posterior.support.local_membrane is not None:
        result.update(local_membrane=posterior.support.local_binding_payload(),
                      local_membrane_status=posterior.support.local_membrane_status())
    return result

def load_direct_predictor(config: dict, checkpoint_path: Path, actions: list[str]) -> dict:
    """Shared frozen predictor/conditioner admission for development and calibration."""
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    from Libs.exp.evaluation.evaluate_screening_system import _native_training_features

    if not native_direct_mean_config(config):
        raise ValueError("Native direct predictor requires its explicit opt-in")
    started = time.perf_counter()
    conditioning = load_frozen_conditioner(config)
    conditioning_load = time.perf_counter()-started
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_direct_checkpoint(checkpoint, config, conditioning)
    predictor = MicroenvironmentPretrainingModel(config, False, conditioning=conditioning)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.requires_grad_(False).eval()
    common_path = Path(conditioning["identity"]["checkpoint_path"])
    common_checkpoint = torch.load(common_path, map_location="cpu")
    binding = conditioning["identity"]["training_seed_binding"]
    features, feature_context, paths = _native_training_features(
        conditioning["config"], common_path, common_checkpoint, binding, actions)
    return {"conditioning": conditioning, "predictor": predictor, "model": conditioning["model"],
            "binding": binding, "predictor_binding": checkpoint["training_seed_binding"],
            "features": features, "feature_context": feature_context,
            "paths": [*conditioning["paths"], *paths],
            "load_cost": {"conditioning_checkpoint_load_wall_seconds": conditioning_load,
                          "direct_checkpoint_and_features_load_wall_seconds": time.perf_counter()-started}}
