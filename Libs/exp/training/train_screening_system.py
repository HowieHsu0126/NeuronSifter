from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.native_field_normalization import normalization_from_config
from Libs.models.modules.world_action_encoder import native_action_identity_spec
from Libs.models.modules.quantized_mddm_rollout_operator import validate_native_action_checkpoint
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.data.native_diffusion_data import (ROLES as NATIVE_TEACHER_ROLES, native_enabled, screening_native_payload, validate_cached_native, stable_numeric_id, native_action_policy, validate_scalar_cache)
from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.utils.evaluation.benchmark_datasets import benchmark_program_config
from Libs.utils.model.dynamics_fields import (
    batch_simulation_targets,
    default_region_prior,
    node_adjacency_matrix,
    observed_region_mask,
    region_id_list,
    selected_regimen_simulation_tensors,
    species_id_list,
)
from Libs.utils.exp.experiment_tracking import capture_run_metadata, scoped_seed, seed_everything
from Libs.utils.exp.experiment_manifests import resolve_training_seed, validate_training_seed_checkpoint
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging, get_pipeline_logger
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_json, write_text_atomic
from Libs.utils.evaluation.screening_candidate_selection import (
    diversified_candidate_indices,
    preserved_case_study_regimen_indices,
    regimen_diversity_group_keys,
    regimen_prior_scores,
    target_program_prior_scores,
)
from Libs.utils.evaluation.screening_datasets import (
    counterfactual_pair_indices,
    foundation_registry_frames,
    fold_patient_latents,
    patient_regimen_labels,
    patient_target_labels,
    production_patient_latents,
    regimen_features,
    simulation_teacher_factor_masks,
    target_program_features,
    target_uid_to_entity_lookup,
    world_state_factor_targets,
    world_state_supervision_masks,
    world_state_support_vectors,
)
from Libs.models.modules.world_state import joint_belief_objective
from Libs.utils.data.native_diffusion_data import select_joint_source, bind_joint_programs, posterior_family, JOINT_SOURCE_SCHEMA
from Libs.utils.model.screening_model_config import (
    baseline_field_policy_from_config,
    rollout_update_policy_from_config,
    solver_config_from_config,
    solver_state_layout_from_config,
    world_state_posterior_config_from_config,
)
from Libs.utils.model.screening_core import (
    component_scale,
    counterfactual_pairwise_loss,
    pkpd_prior_loss,
    pkpd_reference_tensors,
    simulation_teacher_factor_loss,
    teacher_weighted_rollout_mse,
    tri_component_loss,
    tri_context_tensors,
    tri_target_tensors,
    tri_weight_by_component,
    world_state_target_loss,
    world_state_factorization_loss,
)
from Libs.utils.evaluation.strict_case_discovery import assert_clean_action_ready_manifest, assert_no_forced_anchor_regimens, strict_case_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the dynamics-grounded NeuronSifter AD screening system.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--fold-name", default="", help="Optional fold name subset.")
    parser.add_argument("--training-scope", default="all", choices=["folds", "production", "all"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-epochs-override", default="", help="Optional max epoch override for smoke tests.")
    parser.add_argument("--patience-epochs-override", default="", help="Optional early-stopping patience override for bounded runs.")
    parser.add_argument("--training-window-ordinal", default="", help="One-based checkpoint window ordinal for resumable training.")
    parser.add_argument("--training-window-count", default="", help="Total checkpoint window count for resumable training.")
    parser.add_argument("--training-target-count-override", default="", help="Optional training target count override.")
    parser.add_argument("--training-regimen-count-override", default="", help="Optional training regimen count override.")
    parser.add_argument("--disease-program", default="", help="Optional disease program override.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def artifact_dir(config: dict, scope: str, split_name: str) -> Path:
    root = Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
    benchmark_name = config["screening_pipeline"]["screening_benchmark"]["benchmark_name"]
    if scope == "production":
        return root / benchmark_name / "production" / "ad_screening_system"
    return root / benchmark_name / "folds" / split_name / "ad_screening_system"


def read_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def freeze_signature_sha256(config: dict) -> str:
    suite = dict(config.get("paper_grade_full_suite", {}))
    explicit_hash = str(suite.get("freeze_snapshot_signature_sha256", "")).strip()
    if explicit_hash:
        return explicit_hash
    signature = suite.get("freeze_snapshot_signature", {})
    if isinstance(signature, dict) and signature:
        return json_sha256(signature)
    manifest_path = Path(str(suite.get("freeze_policy", {}).get("freeze_manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if manifest_path.exists():
        manifest = read_json_file(manifest_path)
        snapshot = manifest.get("snapshot_signature", {})
        if isinstance(snapshot, dict):
            return json_sha256(snapshot)
    return ""


def training_cache_policy(config: dict) -> dict[str, object]:
    return dict(config["screening_pipeline"]["screening_benchmark"]["training"].get("cache_policy", {}))


def cache_enabled(config: dict) -> bool:
    return bool(training_cache_policy(config).get("enabled", False))


def cache_required_for_formal(config: dict) -> bool:
    return bool(training_cache_policy(config).get("require_cache_for_formal", False))


def training_job_cache_build_allowed(config: dict) -> bool:
    policy = training_cache_policy(config)
    return bool(policy.get("allow_training_job_cache_build", True)) and not cache_required_for_formal(config)


def cache_schema_version(config: dict) -> str:
    return str(training_cache_policy(config).get("schema_version", "screening_training_tensor_cache.v1"))


def training_cache_tensor_path(artifact_root: Path, config: dict) -> Path:
    filename = str(training_cache_policy(config).get("tensor_cache_filename", "training_tensor_cache.pt"))
    return artifact_root / filename


def training_cache_manifest_path(artifact_root: Path, config: dict) -> Path:
    filename = str(training_cache_policy(config).get("manifest_filename", "training_cache_manifest.json"))
    return artifact_root / filename


def required_cache_manifest_fields(config: dict) -> list[str]:
    return [
        str(field_name)
        for field_name in training_cache_policy(config).get("required_manifest_fields", [])
    ]


def source_artifact_paths(config: dict, config_path: Path) -> list[Path]:
    candidates: list[Path] = [config_path]
    candidates.append(REPO_ROOT / "Output/audits/formal_experiment_freeze_manifest.json")
    benchmark_root = resolve_repo_path(config["benchmark_registry"]["local_root"])
    candidates.append(benchmark_root / config["benchmark_registry"]["relative_paths"]["fold_catalog"])
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    action_rel = config["screening_pipeline"]["action_ready_registry"]["relative_paths"]
    candidates.append(action_root / action_rel["target_program_catalog"])
    candidates.append(action_root / action_rel["regimen_catalog"])
    drug_root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    drug_rel = config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]
    candidates.append(drug_root / drug_rel["target_catalog"])
    candidates.append(drug_root / drug_rel["compound_catalog"])
    for registry_name, relative_key in [
        ("world_state_ready_registry", "world_state_catalog"),
        ("simulation_teacher_ready_registry", "simulation_teacher_catalog"),
        ("counterfactual_benchmark_ready_registry", "counterfactual_benchmark_catalog"),
        ("feedback_ready_registry", "feedback_event_catalog"),
    ]:
        registry = config["data"].get(registry_name, {})
        if registry:
            candidates.append(resolve_repo_path(registry["local_root"]) / registry["relative_paths"][relative_key])
    candidate_validation = config["validation"].get("candidate_validation_registry", {})
    if candidate_validation:
        candidates.append(
            resolve_repo_path(candidate_validation["local_root"])
            / candidate_validation["relative_paths"]["candidate_validation_catalog"]
        )
    native_registry_path = config.get("native_diffusion_data", {}).get("registry_path")
    if native_registry_path:
        candidates.append(resolve_repo_path(native_registry_path))
    return [path.resolve() for path in candidates if path.exists() and path.is_file()]


def source_artifact_hashes(config: dict, config_path: Path) -> dict[str, str]:
    return {
        repo_relative(path): file_sha256(path)
        for path in source_artifact_paths(config, config_path)
    }


def cache_sample_counts(modality_payload: dict[str, dict[str, object]]) -> dict[str, dict[str, int]]:
    return {
        modality: {
            split_name: int(patient_state_array(payload).shape[0])
            for split_name, payload in split_payload.items()
        }
        for modality, split_payload in modality_payload.items()
    }


def tensorize_cache_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return torch.as_tensor(np.ascontiguousarray(value))
    if isinstance(value, dict):
        return {str(key): tensorize_cache_value(payload_value) for key, payload_value in value.items()}
    if isinstance(value, list):
        return [tensorize_cache_value(payload_value) for payload_value in value]
    return value


def detensorize_cache_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {str(key): detensorize_cache_value(payload_value) for key, payload_value in value.items()}
    if isinstance(value, list):
        return [detensorize_cache_value(payload_value) for payload_value in value]
    return value


def cacheable_modality_payload(
    config: dict,
    modality_payload: dict[str, dict[str, object]],
    foundation_registries: dict[str, pd.DataFrame],
) -> dict[str, dict[str, object]]:
    cached_payload: dict[str, dict[str, object]] = {}
    for modality, split_payload in modality_payload.items():
        cached_payload[modality] = {}
        for split_name, payload in split_payload.items():
            metadata = payload["metadata"]
            cached_payload[modality][split_name] = {
                "sample_uids": metadata["sample_uid"].astype(str).tolist(),
                "state_features": patient_state_array(payload),
                "labels": np.asarray(payload["labels"], dtype=np.float32),
                "probabilities": np.asarray(payload["probabilities"], dtype=np.float32),
                "latent": np.asarray(payload["latent"], dtype=np.float32),
                "matrix": np.asarray(payload["matrix"], dtype=np.float32),
                "observed_region_mask": observed_region_mask(metadata, config),
                "world_state_support": world_state_support_vectors(
                    metadata,
                    foundation_registries["world_state"],
                ),
                "factor_targets": world_state_factor_targets(
                    metadata,
                    foundation_registries["world_state"],
                ),
                "factor_masks": world_state_supervision_masks(
                    metadata,
                    foundation_registries["world_state"],
                ),
            }
            if "native_diffusion" in payload:
                cached_payload[modality][split_name]["native_diffusion"] = payload["native_diffusion"]
    return cached_payload


def build_scope_training_cache_payload(
    config: dict,
    scope: str,
    split_name: str,
    modality_payload: dict[str, dict[str, object]],
    target_program_rows: list[dict[str, str]],
    regimen_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    compound_rows: list[dict[str, str]],
    strict_case_rows: list[dict[str, object]],
    foundation_registries: dict[str, pd.DataFrame],
    disease_program: str,
) -> dict[str, object]:
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    selected_target_rows, selected_regimen_rows, label_payload = build_training_targets(
        modality_payload,
        target_program_rows,
        regimen_rows,
        target_rows,
        compound_rows,
        int(training_config["training_target_count"]),
        int(training_config["training_regimen_count"]),
        strict_case_rows,
        foundation_registries["counterfactual_benchmark"],
        disease_program,
    )
    target_frame, target_feature_matrix = target_program_features(selected_target_rows)
    regimen_frame, regimen_feature_matrix = regimen_features(selected_regimen_rows)
    if target_feature_matrix.shape[0] == 0 or regimen_feature_matrix.shape[0] == 0:
        raise ValueError(f"No target or regimen features selected for scope {scope}:{split_name}")
    if native_enabled(config):
        regimen_uids = regimen_frame["regimen_uid"].astype(str).tolist()
        for split_payload in modality_payload.values():
            for role, payload in split_payload.items():
                if role not in NATIVE_TEACHER_ROLES:
                    if "native_diffusion" in payload:
                        raise ValueError("Native teacher payload is forbidden for this training-cache role")
                    continue
                payload["native_diffusion"] = screening_native_payload(
                    config, role, payload["metadata"]["sample_uid"].astype(str).tolist(), regimen_uids)
    return {
        "schema_version": cache_schema_version(config),
        "scope": scope,
        "fold": split_name,
        "selected_target_rows": target_frame.to_dict(orient="records"),
        "selected_regimen_rows": regimen_frame.to_dict(orient="records"),
        "target_feature_matrix": target_feature_matrix.astype(np.float32),
        "regimen_feature_matrix": regimen_feature_matrix.astype(np.float32),
        "label_payload": label_payload,
        "modality_payload": cacheable_modality_payload(config, modality_payload, foundation_registries),
    }


def write_scope_training_cache(
    config: dict,
    config_path: Path,
    argv: list[str],
    artifact_root: Path,
    scope: str,
    split_name: str,
    model_seed: int,
    payload: dict[str, object],
    materialization_seconds: float,
) -> dict[str, object]:
    ensure_directory(artifact_root)
    tensor_path = training_cache_tensor_path(artifact_root, config)
    manifest_path = training_cache_manifest_path(artifact_root, config)
    torch.save(tensorize_cache_value(payload), tensor_path)
    selected_target_rows = list(payload["selected_target_rows"])
    selected_regimen_rows = list(payload["selected_regimen_rows"])
    sample_counts = cache_sample_counts(detensorize_cache_value(payload["modality_payload"]))
    manifest = {
        "schema_version": cache_schema_version(config),
        "config_hash": json_sha256(config),
        "freeze_signature_sha256": freeze_signature_sha256(config),
        "scope": scope,
        "fold": split_name,
        "seed": int(model_seed),
        "source_artifact_hashes": source_artifact_hashes(config, config_path),
        "native_source_bindings": {path: digest for splits in payload["modality_payload"].values()
                                  for item in splits.values() for path, digest in item.get("native_diffusion", {}).get("source_bindings", {}).items()},
        "sample_counts": sample_counts,
        "selected_target_program_count": len(selected_target_rows),
        "selected_regimen_count": len(selected_regimen_rows),
        "target_labels_shape_by_modality": {
            modality: {
                split: list(split_payload["target_labels"].shape)
                for split, split_payload in modality_labels.items()
            }
            for modality, modality_labels in detensorize_cache_value(payload["label_payload"]).items()
        },
        "regimen_labels_shape_by_modality": {
            modality: {
                split: list(split_payload["regimen_labels"].shape)
                for split, split_payload in modality_labels.items()
            }
            for modality, modality_labels in detensorize_cache_value(payload["label_payload"]).items()
        },
        "tensor_cache_path": repo_relative(tensor_path),
        "data_materialization_seconds": float(materialization_seconds),
        "created_command": " ".join(argv),
    }
    if native_enabled(config):
        manifest.update(native_action_identity_schema=native_action_identity_spec(native_action_policy(config))[1],
                        native_action_policy=native_action_policy(config),
                        native_posterior_family=posterior_family(config))
        if posterior_family(config) == "finite_joint_particles":
            manifest["native_joint_source_schema"] = JOINT_SOURCE_SCHEMA
    write_json(manifest_path, manifest)
    return manifest


def cache_manifest_mismatches(
    config: dict,
    config_path: Path,
    manifest: dict[str, object],
    scope: str,
    split_name: str,
    model_seed: int,
) -> list[str]:
    expected = {
        "schema_version": cache_schema_version(config),
        "config_hash": json_sha256(config),
        "freeze_signature_sha256": freeze_signature_sha256(config),
        "scope": scope,
        "fold": split_name,
        "seed": int(model_seed),
        "source_artifact_hashes": source_artifact_hashes(config, config_path),
    }
    mismatches = []
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            mismatches.append(key)
    if native_enabled(config):
        for key, value in (("native_action_identity_schema", native_action_identity_spec(native_action_policy(config))[1]),
                           ("native_action_policy", native_action_policy(config))):
            if manifest.get(key) != value:
                mismatches.append(key)
        if posterior_family(config) == "finite_joint_particles":
            for key, value in (("native_posterior_family", posterior_family(config)),
                               ("native_joint_source_schema", JOINT_SOURCE_SCHEMA)):
                if manifest.get(key) != value:
                    mismatches.append(key)
        elif manifest.get("native_posterior_family", "deterministic_compatibility") != "deterministic_compatibility":
            mismatches.append("native_posterior_family")
        bindings = manifest.get("native_source_bindings", {})
        if not bindings:
            mismatches.append("native_source_bindings")
        for path, digest in bindings.items():
            if not Path(path).is_file() or file_sha256(Path(path)) != digest:
                mismatches.append(f"native_source_binding:{path}")
    for key in required_cache_manifest_fields(config):
        if key not in manifest:
            mismatches.append(f"missing_manifest_field:{key}")
    return mismatches


def load_scope_training_cache(
    config: dict,
    config_path: Path,
    artifact_root: Path,
    scope: str,
    split_name: str,
    model_seed: int,
) -> tuple[dict[str, object], dict[str, object], float]:
    manifest_path = training_cache_manifest_path(artifact_root, config)
    tensor_path = training_cache_tensor_path(artifact_root, config)
    if not manifest_path.exists() or not tensor_path.exists():
        raise FileNotFoundError(
            f"cache_missing: training tensor cache for {scope}:{split_name} requires "
            f"{repo_relative(manifest_path)} and {repo_relative(tensor_path)}."
        )
    manifest = read_json_file(manifest_path)
    mismatches = cache_manifest_mismatches(config, config_path, manifest, scope, split_name, model_seed)
    if mismatches:
        raise RuntimeError(
            f"cache_stale: training tensor cache for {scope}:{split_name} mismatched "
            f"{','.join(mismatches)} in {repo_relative(manifest_path)}."
        )
    load_started_at = time.perf_counter()
    payload = detensorize_cache_value(torch.load(tensor_path, map_location="cpu", weights_only=False))
    if native_enabled(config):
        validate_cached_native(payload["modality_payload"], [row["regimen_uid"] for row in payload["selected_regimen_rows"]], config)
    return payload, manifest, time.perf_counter() - load_started_at


def max_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def max_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def runtime_output_paths(config: dict, json_key: str, md_key: str) -> tuple[Path, Path]:
    output_paths = dict(config.get("paper_grade_full_suite", {}).get("output_paths", {}))
    json_path = Path(str(output_paths.get(json_key, f"Output/audits/{json_key}.json")))
    md_path = Path(str(output_paths.get(md_key, f"Output/audits/{md_key}.md")))
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    if not md_path.is_absolute():
        md_path = REPO_ROOT / md_path
    return json_path, md_path


def write_runtime_markdown(path: Path, title: str, payload: dict[str, object]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: `{value}`")
    write_text_atomic(path, "\n".join(lines) + "\n")


def write_training_window_runtime_audit(
    config: dict,
    artifact_root: Path,
    payload: dict[str, object],
    training_window_ordinal: int | None,
) -> None:
    suffix = "training_window_runtime_audit.json" if training_window_ordinal is None else f"training_window_{training_window_ordinal}_runtime_audit.json"
    local_path = artifact_root / suffix
    write_json(local_path, payload)
    json_path, md_path = runtime_output_paths(
        config,
        "training_window_runtime_audit_json",
        "training_window_runtime_audit_md",
    )
    write_json(json_path, payload)
    write_runtime_markdown(md_path, "Training Window Runtime Audit", payload)


def checkpoint_state_path(artifact_root: Path) -> Path:
    return artifact_root / "training_state_latest.pt"


def completion_status_path(artifact_root: Path) -> Path:
    return artifact_root / "training_completion.json"


def window_status_path(artifact_root: Path, ordinal: int | None) -> Path:
    if ordinal is None:
        return artifact_root / "training_window_status.json"
    return artifact_root / f"training_window_{ordinal}_status.json"


def resolve_training_window(max_epochs: int, ordinal: int | None, count: int | None) -> tuple[int, int]:
    if ordinal is None and count is None:
        return 0, max_epochs
    if ordinal is None or count is None:
        raise ValueError("--training-window-ordinal and --training-window-count must be provided together.")
    if ordinal <= 0 or count <= 0:
        raise ValueError("Training window ordinal and count must be positive.")
    if ordinal > count:
        raise ValueError("Training window ordinal cannot exceed training window count.")
    epochs_per_window = int(math.ceil(float(max_epochs) / float(count)))
    start_epoch = min(max_epochs, (ordinal - 1) * epochs_per_window)
    stop_epoch = min(max_epochs, ordinal * epochs_per_window)
    return start_epoch, stop_epoch


def screening_training_seed_binding(config: dict, scope: str, split_name: str) -> dict:
    binding = resolve_training_seed(config, require_selected=True)
    parts = [scope, split_name, "ad_screening_system"]
    # The active three-seed policy names actual training seeds; historical policies retain scope derivation.
    model_seed = binding["model_seed"]
    if binding["seed_policy"] != "manuscript_native_three":
        model_seed = scoped_seed(binding["base_training_seed"], *parts)
    return dict(binding, model_seed=model_seed, scope=parts)


def read_seed_bound_screening_checkpoints(artifact_root: Path, binding: dict) -> tuple[dict | None, dict | None]:
    """Prospective reuse checks precede cache, model and optimizer consumption."""
    if binding["seed_policy"] != "manuscript_native_five":
        return None, None
    state = None
    completed = None
    state_path = checkpoint_state_path(artifact_root)
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        validate_training_seed_checkpoint(state, binding)
    completion_path = completion_status_path(artifact_root)
    if completion_path.exists():
        status = read_json_file(completion_path)
        validate_training_seed_checkpoint(status, binding)
        final_path = artifact_root / "model_weights.pt"
        if status.get("status") == "completed" and final_path.exists():
            completed = torch.load(final_path, map_location="cpu")
            validate_training_seed_checkpoint(completed, binding)
    return state, completed


def validate_screening_training_checkpoint(model, payload: dict, seed_binding: dict) -> None:
    """Validate the current, best and completed states before training reuse."""
    validate_training_seed_checkpoint(payload, seed_binding)
    validate_native_action_checkpoint(model, payload["model_state"])
    if "best_state" in payload:
        validate_native_action_checkpoint(model, payload["best_state"])


def save_checkpoint_state(
    path: Path,
    model: ADScreeningSystem,
    optimizer: torch.optim.Optimizer,
    best_state: dict[str, torch.Tensor],
    best_validation_loss: float,
    patience_counter: int,
    training_history: list[dict[str, float]],
    batch_generator: np.random.Generator,
    next_epoch: int,
    processed_micro_batch_count: int,
    *, training_seed_binding: dict | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "training_seed_binding": training_seed_binding,
            "best_training_seed_binding": training_seed_binding,
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_validation_loss": best_validation_loss,
            "patience_counter": patience_counter,
            "training_history": training_history,
            "batch_generator_state": batch_generator.bit_generator.state,
            "next_epoch": next_epoch,
            "processed_micro_batch_count": processed_micro_batch_count,
        },
        path,
    )


def build_training_targets(
    modality_payload: dict[str, dict[str, object]],
    target_program_rows: list[dict[str, str]],
    regimen_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    compound_rows: list[dict[str, str]],
    training_target_count: int,
    training_regimen_count: int,
    strict_case_rows: list[dict[str, object]],
    counterfactual_benchmark_frame: pd.DataFrame | None = None,
    disease_program: str = "ad_control",
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, dict[str, np.ndarray]]]:
    effective_counterfactual_benchmark_frame = pd.DataFrame() if counterfactual_benchmark_frame is None else counterfactual_benchmark_frame
    target_uid_to_entity = target_uid_to_entity_lookup(target_rows)
    aggregate_target_scores = []
    aggregate_regimen_scores = []
    label_payload: dict[str, dict[str, np.ndarray]] = {}
    for modality, split_payload in modality_payload.items():
        label_payload[modality] = {}
        train_payload = split_payload["train"]
        entity_lookup = {
            str(entity_id): index
            for index, entity_id in enumerate(train_payload["entity_index"]["canonical_entity_id"].astype(str).tolist())
        }
        target_labels = patient_target_labels(
            train_payload["matrix"],
            train_payload["differential"],
            entity_lookup,
            pd.DataFrame(target_program_rows),
        )
        regimen_labels = patient_regimen_labels(
            train_payload["matrix"],
            train_payload["differential"],
            entity_lookup,
            pd.DataFrame(regimen_rows),
            target_uid_to_entity,
        )
        aggregate_target_scores.append(target_labels.mean(axis=0) if target_labels.size else np.zeros((len(target_program_rows),), dtype=np.float32))
        aggregate_regimen_scores.append(regimen_labels.mean(axis=0) if regimen_labels.size else np.zeros((len(regimen_rows),), dtype=np.float32))
        for split_name, current_payload in split_payload.items():
            split_entity_lookup = {
                str(entity_id): index
                for index, entity_id in enumerate(current_payload["entity_index"]["canonical_entity_id"].astype(str).tolist())
            }
            label_payload[modality][split_name] = {
                "target_labels": patient_target_labels(
                    current_payload["matrix"],
                    current_payload["differential"],
                    split_entity_lookup,
                    pd.DataFrame(target_program_rows),
                ),
                "regimen_labels": patient_regimen_labels(
                    current_payload["matrix"],
                    current_payload["differential"],
                    split_entity_lookup,
                    pd.DataFrame(regimen_rows),
                    target_uid_to_entity,
                ),
            }
    mean_target_scores = np.mean(np.stack(aggregate_target_scores, axis=0), axis=0)
    mean_regimen_scores = np.mean(np.stack(aggregate_regimen_scores, axis=0), axis=0)
    target_indices = diversified_candidate_indices(
        mean_target_scores,
        target_program_prior_scores(pd.DataFrame(target_program_rows)),
        [str(row["canonical_entity_id"]) for row in target_program_rows],
        training_target_count,
    )
    regimen_indices = diversified_candidate_indices(
        mean_regimen_scores,
        regimen_prior_scores(pd.DataFrame(regimen_rows)),
        regimen_diversity_group_keys(pd.DataFrame(regimen_rows)),
        training_regimen_count,
    )

    target_anchor_labels = {
        str(anchor_target).strip().upper()
        for case_row in strict_case_rows
        for anchor_target in case_row.get("anchor_targets", [])
        if str(anchor_target).strip()
    }
    regimen_frame = pd.DataFrame(regimen_rows)
    if regimen_frame.empty:
        regimen_anchor_indices = []
    else:
        regimen_anchor_indices = preserved_case_study_regimen_indices(
            regimen_frame,
            strict_case_rows,
            compound_rows,
        )
    counterfactual_priority_indices = []
    if not regimen_frame.empty:
        counterfactual_pairs = counterfactual_pair_indices(regimen_frame, effective_counterfactual_benchmark_frame, disease_program)
        counterfactual_priority_indices = sorted(
            {
                regimen_index
                for pair in counterfactual_pairs
                for regimen_index in pair
            }
        )
    target_anchor_indices = [
        index
        for index, row in enumerate(target_program_rows)
        if str(row.get("canonical_entity_label", "")).strip().upper() in target_anchor_labels
    ]

    def merge_priority_indices(selected_indices: np.ndarray, priority_indices: list[int], maximum_count: int) -> np.ndarray:
        ordered = []
        for priority_index in priority_indices:
            if priority_index not in ordered:
                ordered.append(priority_index)
        for selected_index in selected_indices.tolist():
            selected_index = int(selected_index)
            if selected_index not in ordered:
                ordered.append(selected_index)
        return np.asarray(ordered[:maximum_count], dtype=np.int64)

    target_indices = merge_priority_indices(target_indices, target_anchor_indices, training_target_count)
    regimen_indices = merge_priority_indices(
        regimen_indices,
        regimen_anchor_indices + counterfactual_priority_indices,
        training_regimen_count,
    )
    selected_target_rows = [target_program_rows[index] for index in target_indices]
    selected_regimen_rows = [regimen_rows[index] for index in regimen_indices]
    for modality, split_dict in label_payload.items():
        for split_name, payload in split_dict.items():
            payload["target_labels"] = payload["target_labels"][:, target_indices]
            payload["regimen_labels"] = payload["regimen_labels"][:, regimen_indices]
    return selected_target_rows, selected_regimen_rows, label_payload


def batch_indices(generator: np.random.Generator, sample_count: int, batch_size: int) -> list[np.ndarray]:
    order = generator.permutation(sample_count)
    return [order[index : index + batch_size] for index in range(0, sample_count, batch_size)]


def contiguous_index_chunks(indices: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [indices[index : index + chunk_size] for index in range(0, len(indices), chunk_size)]


def regimen_minibatch_policy(config: dict) -> dict[str, object]:
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    return dict(training_config.get("regimen_minibatch", {"enabled": False}))


def training_regimen_indices(
    config: dict,
    total_regimen_count: int,
    current_regimen_labels: np.ndarray,
    counterfactual_pairs: list[tuple[int, int]],
    generator: np.random.Generator,
) -> np.ndarray:
    policy = regimen_minibatch_policy(config)
    if not bool(policy.get("enabled", False)):
        return np.arange(total_regimen_count, dtype=np.int64)
    candidate_count = int(policy["candidate_count"])
    if candidate_count <= 0:
        raise ValueError("training.regimen_minibatch.candidate_count must be positive.")
    top_label_count = int(policy["top_label_count"])
    if top_label_count < 0:
        raise ValueError("training.regimen_minibatch.top_label_count must be non-negative.")
    counterfactual_pair_coverage = str(policy["counterfactual_pair_coverage"])
    if counterfactual_pair_coverage not in {"strict_candidate_count", "expand_candidate_window", "stochastic_pair_window"}:
        raise ValueError("training.regimen_minibatch.counterfactual_pair_coverage is unsupported.")
    if candidate_count >= total_regimen_count:
        return np.arange(total_regimen_count, dtype=np.int64)

    ordered_indices: list[int] = []

    def add_index(index: int) -> None:
        if index < 0 or index >= total_regimen_count:
            raise ValueError(f"Regimen minibatch index {index} is outside 0..{total_regimen_count - 1}.")
        if index not in ordered_indices:
            ordered_indices.append(index)

    if bool(policy["include_counterfactual_pairs"]):
        if counterfactual_pair_coverage == "stochastic_pair_window":
            pair_window_size = int(policy["counterfactual_pair_window_size"])
            if pair_window_size <= 0:
                raise ValueError("training.regimen_minibatch.counterfactual_pair_window_size must be positive.")
            if pair_window_size * 2 > candidate_count:
                raise ValueError(
                    "training.regimen_minibatch.counterfactual_pair_window_size must fit inside candidate_count."
                )
            pair_order = generator.permutation(len(counterfactual_pairs)).tolist() if counterfactual_pairs else []
            for pair_index in pair_order[:pair_window_size]:
                baseline_index, counterfactual_index = counterfactual_pairs[int(pair_index)]
                add_index(int(baseline_index))
                add_index(int(counterfactual_index))
        else:
            for baseline_index, counterfactual_index in counterfactual_pairs:
                add_index(int(baseline_index))
                add_index(int(counterfactual_index))
            if len(ordered_indices) > candidate_count:
                if counterfactual_pair_coverage != "expand_candidate_window":
                    raise ValueError(
                        "training.regimen_minibatch.candidate_count must cover all configured counterfactual pair regimens."
                    )
                candidate_count = min(total_regimen_count, len(ordered_indices) + top_label_count)

    if top_label_count > 0 and current_regimen_labels.size:
        label_priority = np.max(current_regimen_labels, axis=0)
        for index in np.argsort(-label_priority)[:top_label_count].tolist():
            if len(ordered_indices) >= candidate_count:
                break
            add_index(int(index))

    for index in generator.permutation(total_regimen_count).tolist():
        if len(ordered_indices) >= candidate_count:
            break
        add_index(int(index))
    return np.asarray(ordered_indices, dtype=np.int64)


def remap_counterfactual_pairs(
    pair_indices: list[tuple[int, int]],
    regimen_indices: np.ndarray,
) -> list[tuple[int, int]]:
    local_lookup = {int(global_index): local_index for local_index, global_index in enumerate(regimen_indices.tolist())}
    return [
        (local_lookup[int(baseline_index)], local_lookup[int(counterfactual_index)])
        for baseline_index, counterfactual_index in pair_indices
        if int(baseline_index) in local_lookup and int(counterfactual_index) in local_lookup
    ]


def assert_tri_component_shape_contract(
    predicted_components: dict[str, torch.Tensor],
    target_components: dict[str, torch.Tensor],
) -> None:
    for component_name, predicted_tensor in predicted_components.items():
        if component_name not in target_components:
            raise KeyError(f"TRI target components are missing {component_name!r}.")
        target_tensor = target_components[component_name]
        if tuple(predicted_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"TRI component {component_name!r} shape mismatch: "
                f"predicted {tuple(predicted_tensor.shape)} target {tuple(target_tensor.shape)}."
            )


def configured_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported mixed precision dtype: {dtype_name}")
    return dtype_map[dtype_name]


def autocast_context(memory_policy: dict[str, object], device: torch.device):
    mixed_precision = dict(memory_policy["mixed_precision"])
    enabled = bool(mixed_precision["enabled"]) and device.type == str(mixed_precision["device_type"])
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=configured_torch_dtype(str(mixed_precision["dtype"])),
    )


def training_memory_policy(config: dict) -> dict[str, object]:
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    memory_policy = dict(training_config["memory_policy"])
    effective_batch_size = int(memory_policy["effective_batch_size"])
    configured_batch_size = int(training_config["batch_size"])
    micro_batch_size = int(memory_policy["micro_batch_size"])
    accumulation_steps = int(memory_policy["gradient_accumulation_steps"])
    if configured_batch_size != effective_batch_size:
        raise ValueError(
            "training.batch_size must match training.memory_policy.effective_batch_size "
            f"({configured_batch_size} != {effective_batch_size})."
        )
    if micro_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("training.memory_policy micro_batch_size and gradient_accumulation_steps must be positive.")
    if micro_batch_size * accumulation_steps != effective_batch_size:
        raise ValueError(
            "training.memory_policy must preserve the configured effective batch size: "
            f"{micro_batch_size} * {accumulation_steps} != {effective_batch_size}."
        )
    return memory_policy


def ranking_margin_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] <= 1:
        return prediction.new_tensor(0.0)
    positive_index = target.argmax(dim=1)
    negative_index = target.argmin(dim=1)
    positive_scores = prediction.gather(1, positive_index.unsqueeze(1)).squeeze(1)
    negative_scores = prediction.gather(1, negative_index.unsqueeze(1)).squeeze(1)
    return torch.relu(1.0 - (positive_scores - negative_scores)).mean()


def training_mask(metadata: pd.DataFrame, config: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(observed_region_mask(metadata, config), dtype=torch.float32, device=device)


def patient_state_array(payload: dict[str, object]) -> np.ndarray:
    if "state_features" not in payload:
        raise KeyError("Training payload is missing state_features under the multiscale patient-state contract.")
    return np.asarray(payload["state_features"], dtype=np.float32)


def observation_mask_tensor(payload: dict[str, object], config: dict, device: torch.device) -> torch.Tensor:
    if "observed_region_mask" in payload:
        return torch.tensor(payload["observed_region_mask"], dtype=torch.float32, device=device)
    return training_mask(payload["metadata"], config, device)


def world_state_support_payload(
    payload: dict[str, object],
    world_state_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    if "world_state_support" in payload:
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(payload["world_state_support"]).items()
        }
    return world_state_support_vectors(payload["metadata"], world_state_frame)


def factor_target_payload(
    payload: dict[str, object],
    world_state_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    if "factor_targets" in payload:
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(payload["factor_targets"]).items()
        }
    return world_state_factor_targets(payload["metadata"], world_state_frame)


def factor_supervision_payload(
    payload: dict[str, object],
    world_state_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    if "factor_masks" in payload:
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(payload["factor_masks"]).items()
        }
    return world_state_supervision_masks(payload["metadata"], world_state_frame)


def regimen_chunk_size(config: dict) -> int:
    return int(config["screening_pipeline"]["screening_benchmark"].get("execution_controls", {}).get("regimen_chunk_size", 0))


def rollout_model_kwargs(config: dict, rollout_step_count_override: int | None = None) -> dict[str, object]:
    rollout_config = config["screening_pipeline"]["screening_benchmark"].get("long_horizon_rollout", {})
    operator_config = config["screening_pipeline"]["screening_benchmark"].get("model_architecture", {}).get("rollout_operator", {})
    species_lookup = {species_id: index for index, species_id in enumerate(species_id_list(config))}
    fast_state_indices = [
        species_lookup[species_id]
        for species_id in rollout_config.get("fast_state_species_ids", [])
        if species_id in species_lookup
    ]
    slow_state_indices = [
        species_lookup[species_id]
        for species_id in rollout_config.get("slow_state_species_ids", [])
        if species_id in species_lookup
    ]
    return {
        "rollout_step_count": int(
            rollout_step_count_override
            if rollout_step_count_override is not None
            else rollout_config.get("slow_step_count", config["dynamics_ready_registry"]["time_grid"]["step_count"])
        ),
        "fast_steps_per_slow_step": int(rollout_config.get("fast_steps_per_slow_step", 1)),
        "fast_state_indices": fast_state_indices,
        "slow_state_indices": slow_state_indices,
        "slow_update_scale": float(rollout_config.get("slow_update_scale", 0.25)),
        "rollout_update_policy": rollout_update_policy_from_config(config),
    }


def load_action_registries(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    rel = config["screening_pipeline"]["action_ready_registry"]["relative_paths"]
    target_rows = read_tsv(root / rel["target_program_catalog"])
    regimen_rows = read_tsv(root / rel["regimen_catalog"])
    target_catalog_rows = read_tsv(resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]) / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"])
    compound_rows = read_tsv(resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]) / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"])
    return target_rows, regimen_rows, target_catalog_rows, compound_rows


def load_scope_modality_payload(
    config: dict,
    scope: str,
    split_name: str,
    disease_program: str,
    benchmark_name: str = "",
) -> dict[str, dict[str, object]]:
    if scope == "production":
        return production_patient_latents(config)
    return {
        modality: {
            "train": fold_patient_latents(config, split_name, "train", disease_program, benchmark_name)[modality],
            "validation": fold_patient_latents(config, split_name, "validation", disease_program, benchmark_name)[modality],
            "test": fold_patient_latents(config, split_name, "test", disease_program, benchmark_name)[modality],
        }
        for modality in ["transcriptomics", "metabolomics"]
    }


def native_diffusion_training_kwargs(model: ADScreeningSystem, payload: dict, sample_indices: object,
                                     regimen_indices: object, device: torch.device, *,
                                     split_name: str | None = None) -> dict[str, object]:
    if model.field_rollout_operator.quantized_mddm_rollout_operator is None:
        return {}
    native = payload.get("native_diffusion")
    if not isinstance(native, dict):
        raise ValueError("Native screening training requires an admitted native_diffusion cache.")
    role = native.get("split_name") if split_name is None else split_name
    if role not in NATIVE_TEACHER_ROLES or native.get("split_name") != role:
        raise ValueError("Native teacher payload does not match requested training/development/calibration role")
    joint = getattr(getattr(model, "world_state_posterior", None), "family", None) == "finite_joint_particles"
    required = {"split_name", "teacher_trajectory", "teacher_confidence", "occupancy", "support", "physical_time",
                "initial_state", "posterior_context", "sample_uids", "regimen_uids"}
    if joint:
        required = {"split_name", "teacher_trajectory", "teacher_confidence", "sample_uids", "regimen_uids", "joint_source"}
    if native.get("grid", {}).get("quantity_profile"):
        required.add("quantity_support")
        if not joint:
            required.update(FORCING_ARRAY_NAMES)
    if not required.issubset(native):
        raise ValueError("Native screening training requires an admitted native_diffusion cache with independent teacher/confidence and compiled occupancy/support/physical_time; condition-label simulation targets are not an admission receipt.")
    action_policy = getattr(model.field_rollout_operator, "action_policy_name", "occupancy_program")
    if action_policy in {"drug_id", "target_id"} and (
            native.get("action_identity_schema") != native_action_identity_spec(action_policy)[1]
            or native.get("action_policy") != action_policy or (not joint and "action_identity" not in native)):
        raise ValueError(f"Native {action_policy} requires a current canonical action identity cache schema/policy.")
    if action_policy == "scalar_exposure":
        if native.get("action_policy") != action_policy:
            raise ValueError("Scalar exposure cache action policy mismatch")
        validate_scalar_cache(native)
    if joint:
        samples = [native["sample_uids"][int(index)] for index in np.atleast_1d(np.arange(len(native["sample_uids"]))[sample_indices])]
        regimens = [native["regimen_uids"][int(index)] for index in np.atleast_1d(np.arange(len(native["regimen_uids"]))[regimen_indices])]
        source = select_joint_source(native["joint_source"], samples, regimens, action_policy=action_policy)
        if source["support"]["role"] != role:
            raise ValueError("Joint source role does not match the training call.")
    if "quantity_support" in native and np.asarray(native["quantity_support"]).dtype != np.bool_:
        raise ValueError("Native quantity support must retain its boolean dtype")
    model.field_rollout_operator.field_normalization.assert_binding(native.get("grid"))
    def select(key: str) -> torch.Tensor:
        return torch.as_tensor(native[key], device=device, dtype=torch.bool if key == "quantity_support" else torch.float32)[sample_indices][:, regimen_indices]
    if len(native["sample_uids"]) != len(payload["labels"]) or len(native["regimen_uids"]) != np.asarray(native["teacher_confidence"]).shape[1]:
        raise ValueError("Native cache sample/regimen identity cardinality mismatch")
    sample_ids = torch.tensor([stable_numeric_id(uid) for uid in native["sample_uids"]], dtype=torch.long)[sample_indices]
    regimen_ids = torch.tensor([stable_numeric_id(uid) for uid in native["regimen_uids"]], dtype=torch.long)[regimen_indices]
    if joint:
        posterior = model.infer_world_state(joint_support=source["support"], observations=source["observations"])["joint_posterior"]
        return {"teacher_trajectory": select("teacher_trajectory"), "teacher_confidence": select("teacher_confidence"),
                "joint_posterior": posterior, "particle_programs": bind_joint_programs(posterior, source, action_policy=action_policy),
                "regimen_uids": regimens, "sample_ids": sample_ids, "regimen_ids": regimen_ids}
    context_keys = ["occupancy", "support", "physical_time", "initial_state", "posterior_context"]
    if "quantity_support" in native:
        context_keys.append("quantity_support")
        context_keys.extend(FORCING_ARRAY_NAMES)
    if "action_identity" in native:
        context_keys.append("action_identity")
    if action_policy == "scalar_exposure":
        context_keys.extend(["scalar_exposure_auc", "scalar_exposure_interval"])
    return {"teacher_trajectory": select("teacher_trajectory"), "teacher_confidence": select("teacher_confidence"),
            "diffusion_context": {key: select(key) for key in context_keys},
            "sample_ids": sample_ids, "regimen_ids": regimen_ids}

def screening_teacher_losses(
    regimen_output: dict, simulation_targets: torch.Tensor, factor_masks: dict[str, torch.Tensor],
    config: dict, native_kwargs: dict[str, object], *, training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same admitted per-action weight for all native teacher targets."""
    if "joint_posterior" in native_kwargs:
        posterior = native_kwargs["joint_posterior"]
        weights = posterior.weights
        trajectories = regimen_output["particle_field_trajectory"]
        predictions = regimen_output["particle_training_reconstruction"] if training else trajectories
        rollout = weights.new_zeros(())
        factors = weights.new_zeros(())
        for bi in range(weights.shape[0]):
            masks = factor_masks  # Source factor supervision masks are action aligned.
            for ji in range(weights.shape[1]):
                target = simulation_targets[bi:bi+1]
                confidence = native_kwargs["teacher_confidence"][bi:bi+1]
                weight = weights[bi, ji] / weights.shape[0]
                rollout = rollout + weight * teacher_weighted_rollout_mse(predictions[bi:bi+1, ji], target, confidence)
                factors = factors + weight * simulation_teacher_factor_loss(trajectories[bi:bi+1, ji], target, masks, config, teacher_weight=confidence)
        return rollout, factors
    teacher_weight = native_kwargs.get("teacher_confidence")
    prediction = regimen_output["field_training_reconstruction"] if training else regimen_output["field_trajectory"]
    rollout_loss = teacher_weighted_rollout_mse(prediction, simulation_targets, teacher_weight)
    factor_loss = simulation_teacher_factor_loss(
        regimen_output["field_trajectory"], simulation_targets, factor_masks, config, teacher_weight=teacher_weight,
    )
    return rollout_loss, factor_loss



def screening_belief_loss(model, native_kwargs, output, state, mask, support, missing, targets, target_masks, config):
    if "joint_posterior" in native_kwargs:
        loss, status = joint_belief_objective(native_kwargs["joint_posterior"], config)
        output["belief_evidence_status"] = status
        return loss, status
    posterior = model.infer_world_state(state, observation_mask=mask)
    weights = config["screening_pipeline"]["screening_benchmark"]["loss_weights"]
    loss = (float(weights["world_state_consistency_loss"]) * world_state_factorization_loss(posterior, support, missing, config)
            + float(weights["world_state_target_loss"]) * world_state_target_loss(posterior, targets, target_masks)
            + float(weights["observation_reconstruction_loss"]) * torch.nn.functional.mse_loss(output["reconstructed_patient"], state))
    return loss, {"posterior_family": "deterministic_compatibility"}

def screening_rollout_target_source(config: dict, regimen_frame: pd.DataFrame, modality_payload: dict) -> tuple[dict | None, int]:
    """Choose the independent native target source before building the model."""
    native_rollout = rollout_update_policy_from_config(config)["transition_backbone"].get("quantized_mddm_rollout", {})
    if native_rollout.get("enabled", False):
        native_training_payloads = [payload["train"].get("native_diffusion") for payload in modality_payload.values()]
        if not native_training_payloads or any(not isinstance(payload, dict) or "teacher_trajectory" not in payload for payload in native_training_payloads):
            raise ValueError("Native screening requires admitted native_diffusion training tensors before model construction.")
        trajectories = [np.asarray(payload["teacher_trajectory"]) for payload in native_training_payloads]
        if any(value.ndim != 6 or min(value.shape) < 1 for value in trajectories):
            raise ValueError("Native teacher must have sample/action/time/region/compartment/species axes.")
        horizons = {int(value.shape[2]) for value in trajectories}
        if len(horizons) != 1:
            raise ValueError("Native screening teacher horizons disagree across modalities.")
        simulation_rollout_step_count = horizons.pop()
        simulation_targets_by_condition = None
    else:
        simulation_targets_by_condition = selected_regimen_simulation_tensors(config, regimen_frame)
        simulation_rollout_step_count = int(simulation_targets_by_condition["positive"].shape[1])
    return simulation_targets_by_condition, simulation_rollout_step_count


def train_one_scope(
    config: dict,
    config_path: Path,
    argv: list[str],
    scope: str,
    split_name: str,
    modality_payload: dict[str, dict[str, object]] | None,
    target_program_rows: list[dict[str, str]],
    regimen_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    compound_rows: list[dict[str, str]],
    strict_case_rows: list[dict[str, object]],
    foundation_registries: dict[str, pd.DataFrame] | torch.device | None = None,
    disease_program: str = "",
    device: torch.device | None = None,
    training_window_ordinal: int | None = None,
    training_window_count: int | None = None,
) -> Path:
    seed_binding = screening_training_seed_binding(config, scope, split_name)
    artifact_root = artifact_dir(config, scope, split_name)
    seed_load_started = time.perf_counter()
    early_state, early_completed = read_seed_bound_screening_checkpoints(artifact_root, seed_binding)
    seed_checkpoint_load_seconds = time.perf_counter() - seed_load_started
    effective_device = foundation_registries if isinstance(foundation_registries, torch.device) else device
    if effective_device is None:
        effective_device = torch.device("cpu")
    effective_foundation_registries = (
        {
            "world_state": pd.DataFrame(),
            "simulation_teacher": pd.DataFrame(),
            "counterfactual_benchmark": pd.DataFrame(),
            "candidate_validation": pd.DataFrame(),
            "feedback": pd.DataFrame(),
        }
        if foundation_registries is None or isinstance(foundation_registries, torch.device)
        else foundation_registries
    )
    effective_disease_program = disease_program or str(config["screening_pipeline"]["screening_benchmark"]["active_disease_program"])
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    benchmark_config = config["screening_pipeline"]["screening_benchmark"]
    model_name = "ad_screening_system"
    model_logger = get_pipeline_logger(scope=scope, fold=split_name, model_name=model_name)
    model_seed = seed_binding["model_seed"]
    seed_everything(model_seed)
    artifact_root = artifact_dir(config, scope, split_name)
    ensure_directory(artifact_root)
    cache_load_seconds = 0.0
    cache_manifest: dict[str, object] = {}

    if cache_enabled(config):
        try:
            cache_payload, cache_manifest, cache_load_seconds = load_scope_training_cache(
                config,
                config_path,
                artifact_root,
                scope,
                split_name,
                model_seed,
            )
        except FileNotFoundError:
            if not training_job_cache_build_allowed(config):
                raise
            if modality_payload is None:
                raise
        except RuntimeError:
            raise
        else:
            modality_payload = dict(cache_payload["modality_payload"])
            selected_target_rows = list(cache_payload["selected_target_rows"])
            selected_regimen_rows = list(cache_payload["selected_regimen_rows"])
            label_payload = dict(cache_payload["label_payload"])
            target_frame = pd.DataFrame(selected_target_rows)
            regimen_frame = pd.DataFrame(selected_regimen_rows)
            target_feature_matrix = np.asarray(cache_payload["target_feature_matrix"], dtype=np.float32)
            regimen_feature_matrix = np.asarray(cache_payload["regimen_feature_matrix"], dtype=np.float32)
    if modality_payload is None:
        raise FileNotFoundError(f"cache_missing: no modality payload was provided for {scope}:{split_name}.")
    if "target_frame" not in locals():
        materialization_started_at = time.perf_counter()
        cache_payload = build_scope_training_cache_payload(
            config,
            scope,
            split_name,
            modality_payload,
            target_program_rows,
            regimen_rows,
            target_rows,
            compound_rows,
            strict_case_rows,
            effective_foundation_registries,
            effective_disease_program,
        )
        materialization_seconds = time.perf_counter() - materialization_started_at
        selected_target_rows = list(cache_payload["selected_target_rows"])
        selected_regimen_rows = list(cache_payload["selected_regimen_rows"])
        label_payload = dict(cache_payload["label_payload"])
        modality_payload = dict(cache_payload["modality_payload"])
        target_frame = pd.DataFrame(selected_target_rows)
        regimen_frame = pd.DataFrame(selected_regimen_rows)
        target_feature_matrix = np.asarray(cache_payload["target_feature_matrix"], dtype=np.float32)
        regimen_feature_matrix = np.asarray(cache_payload["regimen_feature_matrix"], dtype=np.float32)
        if cache_enabled(config) and training_job_cache_build_allowed(config):
            cache_manifest = write_scope_training_cache(
                config,
                config_path,
                argv,
                artifact_root,
                scope,
                split_name,
                model_seed,
                cache_payload,
                materialization_seconds,
            )
    if target_feature_matrix.shape[0] == 0 or regimen_feature_matrix.shape[0] == 0:
        raise ValueError(f"No target or regimen features selected for scope {scope}:{split_name}")
    for case_row in strict_case_rows:
        assert_no_forced_anchor_regimens(
            regimen_frame,
            case_row,
            compound_rows,
            f"{scope}:{split_name}:{case_row['case_study_name']}",
        )

    simulation_targets_by_condition, simulation_rollout_step_count = screening_rollout_target_source(config, regimen_frame, modality_payload)
    teacher_factor_masks_numpy = simulation_teacher_factor_masks(
        regimen_frame,
        effective_foundation_registries["simulation_teacher"],
        effective_disease_program,
    )
    teacher_factor_masks = {
        factor_name: torch.tensor(factor_mask, dtype=torch.float32, device=effective_device)
        for factor_name, factor_mask in teacher_factor_masks_numpy.items()
    }
    counterfactual_pairs = counterfactual_pair_indices(
        regimen_frame,
        effective_foundation_registries["counterfactual_benchmark"],
        effective_disease_program,
    )
    exposure_target_tensor, occupancy_target_tensor, pk_feasibility_tensor = pkpd_reference_tensors(regimen_frame, config, effective_device)
    target_feature_tensor = torch.tensor(target_feature_matrix, dtype=torch.float32, device=effective_device)
    regimen_feature_tensor = torch.tensor(regimen_feature_matrix, dtype=torch.float32, device=effective_device)

    patient_state_dim = next(
        patient_state_array(payload["train"]).shape[1]
        for payload in modality_payload.values()
        if patient_state_array(payload["train"]).shape[0] > 0
    )
    region_count = len(region_id_list(config))
    species_count = len(species_id_list(config))
    region_prior = default_region_prior(config)
    rollout_kwargs = rollout_model_kwargs(config, rollout_step_count_override=simulation_rollout_step_count)
    field_policy = baseline_field_policy_from_config(config)
    current_solver_state_layout = solver_state_layout_from_config(config)
    current_solver_config = solver_config_from_config(config)
    current_world_state_posterior_config = world_state_posterior_config_from_config(config)
    model = ADScreeningSystem(
        patient_state_dim=patient_state_dim,
        target_feature_dim=target_feature_matrix.shape[1],
        regimen_feature_dim=regimen_feature_matrix.shape[1],
        hidden_dim=int(training_config["hidden_dim"]),
        latent_dim=int(training_config["latent_dim"]),
        dropout_rate=float(training_config["dropout_rate"]),
        compartment_count=len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]),
        tri_weight_by_component=tri_weight_by_component(config),
        region_count=region_count,
        species_count=species_count,
        rollout_step_count=int(rollout_kwargs["rollout_step_count"]),
        adjacency_matrix=node_adjacency_matrix(config).tolist(),
        region_prior_feature_dim=int(region_prior.shape[1]),
        default_region_prior=region_prior.tolist(),
        fast_steps_per_slow_step=int(rollout_kwargs["fast_steps_per_slow_step"]),
        fast_state_indices=list(rollout_kwargs["fast_state_indices"]),
        slow_state_indices=list(rollout_kwargs["slow_state_indices"]),
        slow_update_scale=float(rollout_kwargs["slow_update_scale"]),
        baseline_field_policy=field_policy,
        rollout_update_policy=dict(rollout_kwargs["rollout_update_policy"]),
        solver_state_layout=current_solver_state_layout,
        solver_config=current_solver_config,
        world_state_posterior_config=current_world_state_posterior_config,
        field_normalization_config=normalization_from_config(config),
    ).to(effective_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["learning_rate"]), weight_decay=float(training_config["weight_decay"]))

    capture_run_metadata(
        artifact_root,
        config_path,
        argv,
        model_seed,
        {
            "scope": scope,
            "split_name": split_name,
            "training_seed_binding": seed_binding,
            "device": str(effective_device),
            "benchmark_name": str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"]),
            "model_contract_version": str(config["screening_pipeline"]["screening_benchmark"]["model_contract_version"]),
            "region_count": region_count,
            "species_count": species_count,
            "training_cache_manifest": repo_relative(training_cache_manifest_path(artifact_root, config))
            if cache_manifest
            else "",
            "training_cache_load_seconds": cache_load_seconds,
            "train_shape_by_modality": {
                modality: list(patient_state_array(payload["train"]).shape)
                for modality, payload in modality_payload.items()
            },
        },
    )

    memory_policy = training_memory_policy(config)
    batch_size = int(training_config["batch_size"])
    micro_batch_size = int(memory_policy["micro_batch_size"])
    max_epochs = int(training_config["max_epochs"])
    patience_epochs = int(training_config["patience_epochs"])
    gradient_clipping = dict(training_config["gradient_clipping"])
    loss_weights = benchmark_config["loss_weights"]
    atlas_prior_scale = component_scale(config, "atlas_prior_consistency")
    clue_scale = component_scale(config, "clue_signature_reversal")
    ranking_scale = component_scale(config, "drug_target_evidence_ranking")
    pkpd_scale = component_scale(config, "pkpd_prior_consistency")
    tri_scale = component_scale(config, "tri_gate")
    bce_loss = torch.nn.BCEWithLogitsLoss()
    batch_generator = np.random.default_rng(model_seed)
    training_history: list[dict[str, float]] = []
    action_chunk_size = regimen_chunk_size(config)
    progress_logging = dict(training_config.get("progress_logging", {}))
    micro_batch_log_interval = int(progress_logging["micro_batch_interval"])
    epoch_log_interval = int(progress_logging["epoch_interval"])
    if micro_batch_log_interval < 0 or epoch_log_interval < 0:
        raise ValueError("training.progress_logging intervals must be non-negative.")
    state_path = checkpoint_state_path(artifact_root)
    completion_path = completion_status_path(artifact_root)
    final_checkpoint_path = artifact_root / "model_weights.pt"
    scope_runtime_started_at = time.perf_counter()
    checkpoint_load_seconds = seed_checkpoint_load_seconds
    checkpoint_save_seconds = 0.0
    if completion_path.exists():
        completion_status = read_json_file(completion_path)
        if completion_status.get("status") == "completed" and final_checkpoint_path.exists():
            completed_checkpoint = early_completed if early_completed is not None else torch.load(final_checkpoint_path, map_location=effective_device)
            validate_screening_training_checkpoint(model, completed_checkpoint, seed_binding)
            model_logger.progress(
                "scope_training_already_completed",
                completion_status_path=str(completion_path.relative_to(REPO_ROOT)),
                checkpoint_path=str(final_checkpoint_path.relative_to(REPO_ROOT)),
            )
            return final_checkpoint_path
        raise ValueError(f"Invalid completed-training status for {scope}:{split_name}: {completion_path}")
    window_start_epoch, window_stop_epoch = resolve_training_window(max_epochs, training_window_ordinal, training_window_count)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    patience_counter = 0
    training_history: list[dict[str, float]] = []
    batch_generator = np.random.default_rng(model_seed)
    processed_micro_batch_count = 0
    next_epoch = 0
    if state_path.exists():
        checkpoint_load_started_at = time.perf_counter()
        state_payload = early_state if early_state is not None else torch.load(state_path, map_location=effective_device)
        validate_training_seed_checkpoint(state_payload, seed_binding)
        checkpoint_load_seconds += time.perf_counter() - checkpoint_load_started_at
        validate_native_action_checkpoint(model, state_payload["model_state"])
        validate_native_action_checkpoint(model, state_payload["best_state"])
        model.load_state_dict(state_payload["model_state"])
        optimizer.load_state_dict(state_payload["optimizer_state"])
        best_state = state_payload["best_state"]
        best_validation_loss = float(state_payload["best_validation_loss"])
        patience_counter = int(state_payload["patience_counter"])
        training_history = [dict(row) for row in state_payload["training_history"]]
        batch_generator.bit_generator.state = state_payload["batch_generator_state"]
        next_epoch = int(state_payload["next_epoch"])
        processed_micro_batch_count = int(state_payload["processed_micro_batch_count"])
    elif training_window_ordinal is not None and training_window_ordinal > 1:
        raise FileNotFoundError(
            f"Training window {training_window_ordinal} for {scope}:{split_name} requires checkpoint {state_path}."
        )
    if training_window_ordinal is not None and next_epoch < window_start_epoch:
        raise ValueError(
            f"Checkpoint for {scope}:{split_name} is behind the requested window: "
            f"next_epoch={next_epoch}, window_start_epoch={window_start_epoch}."
        )
    run_start_epoch = max(next_epoch, window_start_epoch)
    if run_start_epoch >= window_stop_epoch:
        write_json(
            window_status_path(artifact_root, training_window_ordinal),
            {
                "status": "already_completed",
                "scope": scope,
                "split_name": split_name,
                "training_window_ordinal": training_window_ordinal,
                "training_window_count": training_window_count,
                "window_start_epoch": window_start_epoch,
                "window_stop_epoch": window_stop_epoch,
                "next_epoch": next_epoch,
            },
        )
        return state_path
    model_logger.progress(
        "scope_training_started",
        selected_target_program_count=int(target_frame.shape[0]),
        selected_regimen_count=int(regimen_frame.shape[0]),
        train_sample_count_by_modality={
            modality: int(patient_state_array(payload["train"]).shape[0])
            for modality, payload in modality_payload.items()
        },
        max_epochs=max_epochs,
        window_start_epoch=window_start_epoch + 1,
        window_stop_epoch=window_stop_epoch,
        run_start_epoch=run_start_epoch + 1,
        training_window_ordinal=training_window_ordinal if training_window_ordinal is not None else "",
        training_window_count=training_window_count if training_window_count is not None else "",
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        regimen_minibatch_policy=regimen_minibatch_policy(config),
    )

    completed_training = False
    for epoch in range(run_start_epoch, window_stop_epoch):
        model.train()
        epoch_losses: list[float] = []
        for modality, payload in modality_payload.items():
            train_payload = payload["train"]
            train_state = patient_state_array(train_payload)
            if train_state.shape[0] == 0:
                continue
            target_labels = label_payload[modality]["train"]["target_labels"]
            regimen_labels = label_payload[modality]["train"]["regimen_labels"]
            observation_mask = observation_mask_tensor(train_payload, config, effective_device)
            world_state_support = world_state_support_payload(
                train_payload,
                effective_foundation_registries["world_state"],
            )
            current_factor_target_payload = factor_target_payload(
                train_payload,
                effective_foundation_registries["world_state"],
            )
            current_factor_supervision_payload = factor_supervision_payload(
                train_payload,
                effective_foundation_registries["world_state"],
            )
            for effective_indices in batch_indices(batch_generator, train_state.shape[0], batch_size):
                effective_batch_size = int(len(effective_indices))
                effective_batch_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for current_indices in contiguous_index_chunks(effective_indices, micro_batch_size):
                    micro_batch_weight = float(len(current_indices)) / float(effective_batch_size)
                    current_regimen_labels_full = regimen_labels[current_indices]
                    current_regimen_indices = training_regimen_indices(
                        config,
                        regimen_feature_tensor.shape[0],
                        current_regimen_labels_full,
                        counterfactual_pairs,
                        batch_generator,
                    )
                    current_regimen_index_tensor = torch.tensor(
                        current_regimen_indices,
                        dtype=torch.long,
                        device=effective_device,
                    )
                    current_regimen_feature_tensor = regimen_feature_tensor.index_select(0, current_regimen_index_tensor)
                    current_exposure_target_tensor = exposure_target_tensor.index_select(0, current_regimen_index_tensor)
                    current_occupancy_target_tensor = occupancy_target_tensor.index_select(0, current_regimen_index_tensor)
                    current_pk_feasibility_tensor = pk_feasibility_tensor.index_select(0, current_regimen_index_tensor)
                    current_teacher_factor_masks = {
                        factor_name: factor_mask.index_select(0, current_regimen_index_tensor)
                        for factor_name, factor_mask in teacher_factor_masks.items()
                    }
                    current_counterfactual_pairs = remap_counterfactual_pairs(counterfactual_pairs, current_regimen_indices)
                    patient_state = torch.tensor(train_state[current_indices], dtype=torch.float32, device=effective_device)
                    patient_labels = torch.tensor(train_payload["labels"][current_indices], dtype=torch.float32, device=effective_device)
                    current_target_labels = torch.tensor(target_labels[current_indices], dtype=torch.float32, device=effective_device)
                    current_regimen_labels = torch.tensor(
                        current_regimen_labels_full[:, current_regimen_indices],
                        dtype=torch.float32,
                        device=effective_device,
                    )
                    current_observation_mask = observation_mask[current_indices]
                    hidden_state_support = torch.tensor(
                        world_state_support["hidden_state_support"][current_indices],
                        dtype=torch.float32,
                        device=effective_device,
                    )
                    missing_modality_count = torch.tensor(
                        world_state_support["missing_modality_count"][current_indices],
                        dtype=torch.float32,
                        device=effective_device,
                    )
                    factor_targets = {
                        factor_name: torch.tensor(
                            current_factor_target_payload[factor_name][current_indices],
                            dtype=torch.float32,
                            device=effective_device,
                        )
                        for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
                    }
                    factor_masks = {
                        factor_name: torch.tensor(
                            current_factor_supervision_payload[factor_name][current_indices],
                            dtype=torch.float32,
                            device=effective_device,
                        )
                        for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
                    }
                    with autocast_context(memory_policy, effective_device):
                        tri_context = tri_context_tensors(
                            train_payload["probabilities"][current_indices],
                            train_payload["matrix"][current_indices],
                            train_payload["latent"][current_indices],
                            current_regimen_feature_tensor.shape[0],
                            effective_device,
                        )
                        native_kwargs = native_diffusion_training_kwargs(model, train_payload, current_indices, current_regimen_indices, effective_device, split_name="train")
                        regimen_output = model.score_regimens(
                            patient_state,
                            current_regimen_feature_tensor,
                            tri_context,
                            observation_mask=current_observation_mask,
                            regimen_chunk_size=action_chunk_size,
                            include_auxiliary_outputs=False,
                            **native_kwargs,
                        )
                        disease_logit = regimen_output["disease_logit"]
                        target_scores, _ = model.score_targets(
                            patient_state,
                            target_feature_tensor,
                            observation_mask=current_observation_mask,
                        )
                        clue_target = current_regimen_feature_tensor[:, 2].unsqueeze(0).expand(patient_state.shape[0], -1)
                        pkpd_loss, pkpd_proxy = pkpd_prior_loss(
                            regimen_output,
                            current_exposure_target_tensor,
                            current_occupancy_target_tensor,
                            current_pk_feasibility_tensor,
                            config,
                        )
                        simulation_targets = native_kwargs["teacher_trajectory"] if native_kwargs else torch.tensor(
                            batch_simulation_targets(simulation_targets_by_condition, train_payload["labels"][current_indices]),
                            dtype=torch.float32,
                            device=effective_device,
                        ).index_select(1, current_regimen_index_tensor)
                        rollout_loss, teacher_factor_loss = screening_teacher_losses(
                            regimen_output, simulation_targets, current_teacher_factor_masks, config, native_kwargs, training=True,
                        )
                        belief_loss, belief_status = screening_belief_loss(
                            model, native_kwargs, regimen_output, patient_state, current_observation_mask,
                            hidden_state_support, missing_modality_count, factor_targets, factor_masks, config)
                        counterfactual_loss = counterfactual_pairwise_loss(
                            regimen_output["regimen_scores"],
                            current_regimen_labels,
                            current_counterfactual_pairs,
                            config,
                        )
                        tri_targets = tri_target_tensors(
                            train_payload["probabilities"][current_indices],
                            train_payload["labels"][current_indices],
                            train_payload["matrix"][current_indices],
                            train_payload["latent"][current_indices],
                            current_regimen_labels_full[:, current_regimen_indices],
                            pkpd_proxy,
                            effective_device,
                        )
                        assert_tri_component_shape_contract(regimen_output["tri_components"], tri_targets)
                        tri_loss = tri_component_loss(regimen_output["tri_components"], tri_targets)
                        classification_loss = bce_loss(disease_logit, patient_labels)
                        atlas_loss = torch.nn.functional.mse_loss(target_scores, current_target_labels)
                        clue_loss = torch.nn.functional.mse_loss(torch.sigmoid(regimen_output["regimen_scores"]), clue_target * patient_labels.unsqueeze(1))
                        ranking_loss = ranking_margin_loss(regimen_output["regimen_scores"], current_regimen_labels)
                        rollout_regularization = torch.nn.functional.mse_loss(
                            regimen_output["field_trajectory"][:, :, 1:],
                            regimen_output["field_trajectory"][:, :, :-1],
                        )
                        total_loss = (
                            float(loss_weights["diagnosis_classification_loss"]) * classification_loss
                            + atlas_prior_scale * float(loss_weights["atlas_prior_consistency_loss"]) * atlas_loss
                            + clue_scale * float(loss_weights["clue_signature_reversal_loss"]) * clue_loss
                            + ranking_scale * float(loss_weights["drug_target_evidence_ranking_loss"]) * ranking_loss
                            + pkpd_scale * float(loss_weights["pkpd_physics_loss"]) * pkpd_loss
                            + float(loss_weights["transition_supervision_loss"]) * rollout_loss
                            + float(loss_weights["simulation_teacher_factor_loss"]) * teacher_factor_loss
                            + belief_loss
                            + float(loss_weights["counterfactual_planning_loss"]) * counterfactual_loss
                            + tri_scale * float(loss_weights["tri_supervision_loss"]) * tri_loss
                            + float(loss_weights["transition_regularization_loss"]) * rollout_regularization
                            + float(loss_weights["quantized_mddm_rollout_loss"]) * regimen_output["rollout_auxiliary_loss"]
                    )
                    (total_loss * micro_batch_weight).backward()
                    effective_batch_loss += float(total_loss.detach().cpu()) * micro_batch_weight
                    processed_micro_batch_count += 1
                    if micro_batch_log_interval and processed_micro_batch_count % micro_batch_log_interval == 0:
                        model_logger.progress(
                            "training_micro_batch_completed",
                            epoch=epoch + 1,
                            modality=modality,
                            processed_micro_batch_count=processed_micro_batch_count,
                            micro_batch_size=int(len(current_indices)),
                            selected_regimen_count=int(current_regimen_indices.shape[0]),
                            loss=float(total_loss.detach().cpu()),
                        )
                if bool(gradient_clipping["enabled"]):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clipping["max_norm"]))
                optimizer.step()
                epoch_losses.append(effective_batch_loss)

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for modality, payload in modality_payload.items():
                validation_payload = payload["validation"]
                validation_state = patient_state_array(validation_payload)
                if validation_state.shape[0] == 0:
                    continue
                patient_state = torch.tensor(validation_state, dtype=torch.float32, device=effective_device)
                patient_labels = torch.tensor(validation_payload["labels"], dtype=torch.float32, device=effective_device)
                observation_mask = observation_mask_tensor(validation_payload, config, effective_device)
                world_state_support = world_state_support_payload(
                    validation_payload,
                    effective_foundation_registries["world_state"],
                )
                current_factor_target_payload = factor_target_payload(
                    validation_payload,
                    effective_foundation_registries["world_state"],
                )
                current_factor_supervision_payload = factor_supervision_payload(
                    validation_payload,
                    effective_foundation_registries["world_state"],
                )
                hidden_state_support = torch.tensor(
                    world_state_support["hidden_state_support"],
                    dtype=torch.float32,
                    device=effective_device,
                )
                missing_modality_count = torch.tensor(
                    world_state_support["missing_modality_count"],
                    dtype=torch.float32,
                    device=effective_device,
                )
                factor_targets = {
                    factor_name: torch.tensor(
                        current_factor_target_payload[factor_name],
                        dtype=torch.float32,
                        device=effective_device,
                    )
                    for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
                }
                factor_masks = {
                    factor_name: torch.tensor(
                        current_factor_supervision_payload[factor_name],
                        dtype=torch.float32,
                        device=effective_device,
                    )
                    for factor_name in ["z_tx", "z_pk", "z_env", "z_ephys", "z_morph"]
                }
                tri_context = tri_context_tensors(
                    validation_payload["probabilities"],
                    validation_payload["matrix"],
                    validation_payload["latent"],
                    regimen_feature_tensor.shape[0],
                    effective_device,
                )
                native_validation_kwargs = native_diffusion_training_kwargs(model, validation_payload, slice(None), slice(None), effective_device, split_name="validation")
                regimen_output = model.score_regimens(
                    patient_state,
                    regimen_feature_tensor,
                    tri_context,
                    observation_mask=observation_mask,
                    regimen_chunk_size=action_chunk_size,
                    **native_validation_kwargs,
                )
                target_scores, _ = model.score_targets(
                    patient_state,
                    target_feature_tensor,
                    observation_mask=observation_mask,
                )
                disease_logit = regimen_output["disease_logit"]
                pkpd_loss, pkpd_proxy = pkpd_prior_loss(
                    regimen_output,
                    exposure_target_tensor,
                    occupancy_target_tensor,
                    pk_feasibility_tensor,
                    config,
                )
                simulation_targets = native_validation_kwargs["teacher_trajectory"] if native_validation_kwargs else torch.tensor(
                    batch_simulation_targets(simulation_targets_by_condition, validation_payload["labels"]),
                    dtype=torch.float32,
                    device=effective_device,
                )
                rollout_loss, teacher_factor_loss = screening_teacher_losses(
                    regimen_output, simulation_targets, teacher_factor_masks, config, native_validation_kwargs, training=False,
                )
                belief_loss, belief_status = screening_belief_loss(
                    model, native_validation_kwargs, regimen_output, patient_state, observation_mask,
                    hidden_state_support, missing_modality_count, factor_targets, factor_masks, config)
                counterfactual_loss = counterfactual_pairwise_loss(
                    regimen_output["regimen_scores"],
                    torch.tensor(label_payload[modality]["validation"]["regimen_labels"], dtype=torch.float32, device=effective_device),
                    counterfactual_pairs,
                    config,
                )
                tri_targets = tri_target_tensors(
                    validation_payload["probabilities"],
                    validation_payload["labels"],
                    validation_payload["matrix"],
                    validation_payload["latent"],
                    label_payload[modality]["validation"]["regimen_labels"],
                    pkpd_proxy,
                    effective_device,
                )
                assert_tri_component_shape_contract(regimen_output["tri_components"], tri_targets)
                clue_target = regimen_feature_tensor[:, 2].unsqueeze(0).expand(patient_state.shape[0], -1)
                validation_loss = (
                    float(loss_weights["diagnosis_classification_loss"]) * bce_loss(disease_logit, patient_labels)
                    + atlas_prior_scale * float(loss_weights["atlas_prior_consistency_loss"]) * torch.nn.functional.mse_loss(
                        target_scores,
                        torch.tensor(label_payload[modality]["validation"]["target_labels"], dtype=torch.float32, device=effective_device),
                    )
                    + clue_scale * float(loss_weights["clue_signature_reversal_loss"]) * torch.nn.functional.mse_loss(torch.sigmoid(regimen_output["regimen_scores"]), clue_target * patient_labels.unsqueeze(1))
                    + ranking_scale * float(loss_weights["drug_target_evidence_ranking_loss"]) * ranking_margin_loss(
                        regimen_output["regimen_scores"],
                        torch.tensor(label_payload[modality]["validation"]["regimen_labels"], dtype=torch.float32, device=effective_device),
                    )
                    + pkpd_scale * float(loss_weights["pkpd_physics_loss"]) * pkpd_loss
                    + float(loss_weights["transition_supervision_loss"]) * rollout_loss
                    + float(loss_weights["simulation_teacher_factor_loss"]) * teacher_factor_loss
                    + belief_loss
                    + float(loss_weights["counterfactual_planning_loss"]) * counterfactual_loss
                    + tri_scale * float(loss_weights["tri_supervision_loss"]) * tri_component_loss(regimen_output["tri_components"], tri_targets)
                    + float(loss_weights["transition_regularization_loss"]) * torch.nn.functional.mse_loss(
                        regimen_output["field_trajectory"][:, :, 1:],
                        regimen_output["field_trajectory"][:, :, :-1],
                    )
                    + float(loss_weights["quantized_mddm_rollout_loss"]) * regimen_output["rollout_auxiliary_loss"]
                )
                validation_losses.append(float(validation_loss.cpu()))

        mean_validation_loss = float(np.mean(validation_losses)) if validation_losses else float("inf")
        training_history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("inf"),
                "mean_validation_loss": mean_validation_loss,
            }
        )
        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch_log_interval and (epoch + 1) % epoch_log_interval == 0:
            model_logger.progress(
                "training_epoch_completed",
                epoch=epoch + 1,
                mean_train_loss=float(np.mean(epoch_losses)) if epoch_losses else float("inf"),
                mean_validation_loss=mean_validation_loss,
                best_validation_loss=best_validation_loss,
                patience_counter=patience_counter,
            )
        next_epoch = epoch + 1
        completed_training = patience_counter >= patience_epochs or next_epoch >= max_epochs
        checkpoint_save_started_at = time.perf_counter()
        save_checkpoint_state(
            state_path,
            model,
            optimizer,
            best_state,
            best_validation_loss,
            patience_counter,
            training_history,
            batch_generator,
            next_epoch,
            processed_micro_batch_count,
            training_seed_binding=seed_binding,
        )
        checkpoint_save_seconds += time.perf_counter() - checkpoint_save_started_at
        if completed_training:
            break

    runtime_seconds = time.perf_counter() - scope_runtime_started_at
    selected_regimen_count = int(regimen_feature_tensor.shape[0])
    runtime_payload = {
        "audit_name": "training_window_runtime_audit",
        "training_seed_binding": seed_binding,
        "schema_version": str(
            config.get("paper_grade_full_suite", {})
            .get("runtime_protocol", {})
            .get("profiling", {})
            .get("runtime_profile_schema_version", "paper_grade_runtime_profile.v1")
        ),
        "scope": scope,
        "split_name": split_name,
        "training_window_ordinal": training_window_ordinal,
        "training_window_count": training_window_count,
        "window_start_epoch": window_start_epoch,
        "window_stop_epoch": window_stop_epoch,
        "next_epoch": next_epoch,
        "wall_time_seconds": runtime_seconds,
        "processed_micro_batch_count": processed_micro_batch_count,
        "selected_regimen_count": selected_regimen_count,
        "micro_batch_per_second": float(processed_micro_batch_count / runtime_seconds) if runtime_seconds > 0.0 else 0.0,
        "regimen_per_second": float(processed_micro_batch_count * selected_regimen_count / runtime_seconds) if runtime_seconds > 0.0 else 0.0,
        "gpu_memory_max_mb": max_gpu_memory_mb(effective_device),
        "max_rss_mb": max_rss_mb(),
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "cache_load_seconds": cache_load_seconds,
        "data_materialization_seconds": float(cache_manifest.get("data_materialization_seconds", 0.0)),
        "cache_manifest_path": repo_relative(training_cache_manifest_path(artifact_root, config)) if cache_manifest else "",
    }
    write_training_window_runtime_audit(config, artifact_root, runtime_payload, training_window_ordinal)

    write_json(
        window_status_path(artifact_root, training_window_ordinal),
        {
            "status": "completed" if completed_training else "checkpointed",
            "training_seed_binding": seed_binding,
            "scope": scope,
            "split_name": split_name,
            "training_window_ordinal": training_window_ordinal,
            "training_window_count": training_window_count,
            "window_start_epoch": window_start_epoch,
            "window_stop_epoch": window_stop_epoch,
            "next_epoch": next_epoch,
            "max_epochs": max_epochs,
            "best_validation_loss": best_validation_loss,
            "patience_counter": patience_counter,
        },
    )
    if not completed_training:
        return state_path

    model.load_state_dict(best_state)
    checkpoint_path = final_checkpoint_path
    final_checkpoint_save_started_at = time.perf_counter()
    torch.save(
        {
            "model_state": model.state_dict(),
            "training_seed_binding": seed_binding,
            "model_kwargs": {
                "patient_state_dim": patient_state_dim,
                "target_feature_dim": target_feature_matrix.shape[1],
                "regimen_feature_dim": regimen_feature_matrix.shape[1],
                "hidden_dim": int(training_config["hidden_dim"]),
                "latent_dim": int(training_config["latent_dim"]),
                "dropout_rate": float(training_config["dropout_rate"]),
                "compartment_count": len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]),
                "tri_weight_by_component": tri_weight_by_component(config),
                "region_count": region_count,
                "species_count": species_count,
                "rollout_step_count": int(rollout_kwargs["rollout_step_count"]),
                "fast_steps_per_slow_step": int(rollout_kwargs["fast_steps_per_slow_step"]),
                "fast_state_indices": list(rollout_kwargs["fast_state_indices"]),
                "slow_state_indices": list(rollout_kwargs["slow_state_indices"]),
                "slow_update_scale": float(rollout_kwargs["slow_update_scale"]),
                "baseline_field_policy": field_policy,
                "rollout_update_policy": dict(rollout_kwargs["rollout_update_policy"]),
                "solver_state_layout": current_solver_state_layout,
                "solver_config": current_solver_config,
                "world_state_posterior_config": current_world_state_posterior_config,
                "field_normalization_config": model.field_rollout_operator.field_normalization.spec,
                "adjacency_matrix": node_adjacency_matrix(config).tolist(),
                "region_prior_feature_dim": int(region_prior.shape[1]),
                "default_region_prior": region_prior.tolist(),
            },
            "model_contract_version": str(config["screening_pipeline"]["screening_benchmark"]["model_contract_version"]),
        },
        checkpoint_path,
    )
    checkpoint_save_seconds += time.perf_counter() - final_checkpoint_save_started_at
    target_frame.to_parquet(artifact_root / "selected_target_programs.parquet", index=False, engine="pyarrow")
    regimen_frame.to_parquet(artifact_root / "selected_regimens.parquet", index=False, engine="pyarrow")
    write_json(
        artifact_root / "training_manifest.json",
        {
            "scope": scope,
            "split_name": split_name,
            "fold_name": split_name,
            "seed": model_seed,
            "ensemble_member_seed": seed_binding["base_training_seed"],
            "training_seed_binding": seed_binding,
            "model_name": model_name,
            "benchmark_name": str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"]),
            "model_contract_version": str(config["screening_pipeline"]["screening_benchmark"]["model_contract_version"]),
            "selected_target_program_count": int(target_frame.shape[0]),
            "selected_regimen_count": int(regimen_frame.shape[0]),
            "best_validation_loss": best_validation_loss,
            "run_metadata_path": str((artifact_root / "run_metadata.json").relative_to(REPO_ROOT)),
            "training_history": training_history,
            "field_axis_fields": {
                "region_count": region_count,
                "compartment_count": len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]),
                "species_count": species_count,
                "rollout_step_count": int(rollout_kwargs["rollout_step_count"]),
                "fast_steps_per_slow_step": int(rollout_kwargs["fast_steps_per_slow_step"]),
            },
            "tri_component_fields": list(tri_weight_by_component(config).keys()),
            "foundation_training": {
                "world_state_registry_rows": int(effective_foundation_registries["world_state"].shape[0]),
                "simulation_teacher_rows": int(effective_foundation_registries["simulation_teacher"].shape[0]),
                "counterfactual_benchmark_rows": int(effective_foundation_registries["counterfactual_benchmark"].shape[0]),
                "candidate_validation_rows": int(effective_foundation_registries["candidate_validation"].shape[0]),
                "feedback_event_rows": int(effective_foundation_registries["feedback"].shape[0]),
                "selected_counterfactual_pair_count": int(len(counterfactual_pairs)),
                "regimen_minibatch_policy": regimen_minibatch_policy(config),
            },
        },
    )
    write_json(
        completion_path,
        {
            "status": "completed",
            "training_seed_binding": seed_binding,
            "scope": scope,
            "split_name": split_name,
            "next_epoch": next_epoch,
            "max_epochs": max_epochs,
            "best_validation_loss": best_validation_loss,
            "patience_counter": patience_counter,
            "training_window_ordinal": training_window_ordinal,
            "training_window_count": training_window_count,
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        },
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolve_training_seed(config, require_selected=True)
    disease_program = args.disease_program or str(config["benchmark_registry"].get("active_disease_program", "ad_control"))
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    config["screening_pipeline"]["screening_benchmark"]["active_disease_program"] = disease_program
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    config_path = Path(args.config).resolve()
    argv = sys.argv.copy()
    if args.max_epochs_override:
        config["screening_pipeline"]["screening_benchmark"]["training"]["max_epochs"] = int(args.max_epochs_override)
        config["screening_pipeline"]["screening_benchmark"]["training"]["patience_epochs"] = min(
            int(config["screening_pipeline"]["screening_benchmark"]["training"]["patience_epochs"]),
            int(args.max_epochs_override),
        )
    if args.patience_epochs_override:
        config["screening_pipeline"]["screening_benchmark"]["training"]["patience_epochs"] = int(args.patience_epochs_override)
    training_window_ordinal = int(args.training_window_ordinal) if args.training_window_ordinal else None
    training_window_count = int(args.training_window_count) if args.training_window_count else None
    if args.training_target_count_override:
        config["screening_pipeline"]["screening_benchmark"]["training"]["training_target_count"] = int(args.training_target_count_override)
    if args.training_regimen_count_override:
        config["screening_pipeline"]["screening_benchmark"]["training"]["training_regimen_count"] = int(args.training_regimen_count_override)
    device = resolve_device(args.device)
    manifest = assert_clean_action_ready_manifest(config)
    strict_case_rows = strict_case_entries(config, disease_program)
    pipeline_logger.progress(
        "action_ready_manifest_validated",
        config_hash=str(manifest["config_hash"]),
        strict_case_count=len(strict_case_rows),
    )
    target_program_rows, regimen_rows, target_rows, compound_rows = load_action_registries(config)
    foundation_registries = foundation_registry_frames(config)
    fold_catalog = read_tsv(resolve_repo_path(config["benchmark_registry"]["local_root"]) / config["benchmark_registry"]["relative_paths"]["fold_catalog"])
    target_fold_rows = [
        row
        for row in fold_catalog
        if row.get("disease_program", disease_program) == disease_program and (not args.fold_name or row["fold_name"] == args.fold_name)
    ]
    if target_fold_rows:
        config["benchmark_registry"]["benchmark_name"] = str(target_fold_rows[0]["benchmark_name"])
    use_precomputed_cache_only = cache_enabled(config) and cache_required_for_formal(config)

    if args.training_scope in {"folds", "all"}:
        for fold_row in target_fold_rows:
            modality_payload = None
            if not use_precomputed_cache_only:
                modality_payload = load_scope_modality_payload(
                    config,
                    "fold",
                    fold_row["fold_name"],
                    disease_program,
                    fold_row["benchmark_name"],
                )
            train_one_scope(
                config,
                config_path,
                argv,
                "fold",
                fold_row["fold_name"],
                modality_payload,
                target_program_rows,
                regimen_rows,
                target_rows,
                compound_rows,
                strict_case_rows,
                foundation_registries,
                disease_program,
                device,
                training_window_ordinal,
                training_window_count,
            )

    if args.training_scope in {"production", "all"}:
        production_payload = None if use_precomputed_cache_only else production_patient_latents(config)
        train_one_scope(
            config,
            config_path,
            argv,
            "production",
            "production",
            production_payload,
            target_program_rows,
            regimen_rows,
            target_rows,
            compound_rows,
            strict_case_rows,
            foundation_registries,
            disease_program,
            device,
            training_window_ordinal,
            training_window_count,
        )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve(),
        disease_program=disease_program,
    )


if __name__ == "__main__":
    main()
