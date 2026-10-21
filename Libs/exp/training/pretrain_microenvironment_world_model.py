from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import resource
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.native_diffusion_data import (native_action_policy, native_enabled, load_registry, load_family_registries, select_records, select_store_records, join_registry, load_records, descriptor_bindings, stable_numeric_id, joint_particle_contexts)
from Libs.utils.data.native_diffusion_data import posterior_family, select_joint_source, parse_joint_source, bind_joint_programs
from Libs.models.modules.evidence import encode_retained_observations
from Libs.models.modules.world_state import WorldStatePosterior, joint_belief_objective
from Libs.models.layers.feed_forward import FeedForwardBlock
from Libs.utils.data.joint_observation import retained_encoder_inputs
from Libs.migrations.legacy_checkpoint import migrate_legacy_checkpoint_keys
from Libs.models.modules.neuronsifter_world_action import MechanismOperatorExpertBank, NeuronSifterWorldActionOperator, native_world_action_from_config
from Libs.models.modules.field_rollout import ActionConditionedOperatorBlock
from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.data.native_observable_profile import observable_profile
from Libs.utils.data.native_field_normalization import NativeFieldNormalization, NativeNormalizationContext, normalization_from_config, normalization_spec
from Libs.utils.data.native_input_staging import NativeInputLocations, input_locations_from_config, stage_native_inputs
from Libs.utils.model.joint_rollout import operator_splitting_record
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config, native_direct_mean_config, world_state_posterior_config_from_config
from Libs.models.modules.transolver_operator import TransolverMicroenvironmentOperator
from Libs.models.modules.native_transolver_mean import NativeTransolverMean
from Libs.models.modules.native_linear_dynamics import NativeLinearDynamics
from Libs.utils.model.native_direct_mean import load_frozen_conditioner, direct_checkpoint_metadata, validate_direct_checkpoint, direct_model_identity
from Libs.utils.exp.experiment_tracking import seed_everything
from Libs.utils.exp.experiment_manifests import resolve_training_seed, validate_training_seed_checkpoint
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.model.dynamics_store import load_chunked_rows, load_zarr_rows, read_zarr_array_metadata
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json, write_text_atomic
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale, default_checkpoint_scale, default_inference_profile, fsdp_profile_config
from Libs.utils.model.transolver_parameter_budget import neuronsifter_world_action_active_parameters, transolver_microenvironment_active_parameters, validate_parameter_budget



from dataclasses import replace
from Libs.utils.model.native_training_selection import NativeTrainingSelection,validation_checkpoint_due,optimizer_learning_rate_ratio
from Libs.utils.model.native_scientific_source import native_posterior_context_width, selected_group_definitions,native_training_profile
from Libs.utils.model.native_scientific_objectives import (
    stratified_mixture_crps, paired_action_score, expected_constraint_score,
)
from Libs.utils.data.joint_observation import observation_to_payload, observation_from_payload, retained_log_likelihood
from Libs.models.modules.world_state import JointParticleSupport


def native_conditioning_array_names(config: dict) -> dict[str, str]:
    if not native_enabled(config):
        return {}
    if posterior_family(config) == "finite_joint_particles":
        if not config.get("native_diffusion_data", {}).get("registry_path"):
            raise ValueError("Joint native training requires the raw-source registry binding.")
        return {"teacher_confidence": "teacher_confidence"}
    if config.get("native_diffusion_data", {}).get("registry_path"):
        names = {key: key for key in ("initial_state", "occupancy", "support", "physical_time", "posterior_context", "teacher_confidence")}
        if config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] == "scalar_exposure":
            names.update({key: key for key in ("scalar_exposure_auc", "scalar_exposure_interval")})
        return names
    declaration = config["screening_pipeline"]["screening_benchmark"]["training"].get("native_conditioning_array_names", {})
    required = {"initial_state", "occupancy", "support", "physical_time", "posterior_context", "teacher_confidence"}
    if set(declaration) != required or not all(isinstance(v, str) and v for v in declaration.values()):
        raise ValueError("Native training requires training.native_conditioning_array_names binding initial_state, occupancy, support, physical_time, posterior_context and teacher_confidence to admitted store arrays; trajectory-only stores are unsupported.")
    return dict(declaration)


def read_native_conditioning(config: dict, store_path: Path, row_indices: np.ndarray, store_format: str) -> dict[str, np.ndarray]:
    names = native_conditioning_array_names(config)
    if not names:
        return {}
    registry = load_registry(config)
    lookup = {row["trajectory_index"]: row for row in registry["teachers"]
              if resolve_repo_path(row["trajectory_store_path"]) == store_path.resolve()
              and row["store_format"] == store_format}
    if any(int(index) not in lookup for index in row_indices):
        raise ValueError("Native conditioning source row identity is missing")
    records = join_registry(registry, config, [lookup[int(index)] for index in row_indices])
    arrays, _ = load_records(registry, records, config)
    return {key: arrays[key] for key in names}

def native_dataset_items(registry: dict, records: list[dict], config: dict, weights: list[float], *,
                         input_locations: NativeInputLocations | None = None,
                         normalization_context: NativeNormalizationContext | None = None) -> list[dict]:
    joint_sources = None
    if registry.get("posterior_family") == "finite_joint_particles":
        joint_sources = [select_joint_source(record["joint_source"], [record["teacher"]["sample_uid"]],
                         [record["teacher"]["regimen_uid"]], action_policy=native_action_policy(config))
                         for record in records]
    arrays, _ = load_records(registry, records, config, input_locations=input_locations, normalization_context=normalization_context)
    normalization_id = normalization_spec(registry["grid"], config, input_locations=input_locations, normalization_context=normalization_context).get("sha256","physical")
    # The sampler batch axis is a trajectory row; biological sample/posterior IDs remain separate.
    items = [{**{key: torch.as_tensor(value[position], dtype=torch.bool if key == "quantity_support" else torch.float32) for key, value in arrays.items()},
             "teacher_loss_weight": torch.tensor(weights[position], dtype=torch.float32),
             "sample_ids": torch.tensor(stable_numeric_id(record["teacher"]["trajectory_uid"]), dtype=torch.long),
             "trajectory_uid": record["teacher"]["trajectory_uid"],
             "sample_uid": record["teacher"]["sample_uid"], "regimen_uid": record["teacher"]["regimen_uid"],
             "posterior_uid": record["teacher"]["posterior_uid"],
             "normalization_id": normalization_id}
            for position, record in enumerate(records)]
    if registry["grid"].get("quantity_profile"):
        for item in items:
            item["source_family_uid"] = registry["grid"]["family_uid"]
            item["quantity_profile_sha256"] = registry["grid"]["quantity_profile_sha256"]
    if native_direct_mean_config(config):
        for item, record in zip(items, records):
            item["native_direct_record"] = record
    if joint_sources is not None:
        for item, source in zip(items, joint_sources):
            item["joint_source"] = source
    return items


def collate_native_sources(items: list[dict]) -> dict:
    if any("native_joint_group" in item for item in items):
        if not all("native_joint_group" in item for item in items):
            raise ValueError("Complete groups cannot share a batch with individual rows")
        return {"joint_groups": [item["native_joint_group"] for item in items]}
    if any("native_direct_record" in item for item in items):
        if not all("native_direct_record" in item for item in items):
            raise ValueError("Direct and legacy sources cannot share a batch")
        return {"native_direct_rows": items}
    if any("joint_source" in item for item in items):
        if not all("joint_source" in item for item in items):
            raise ValueError("Joint and point samples cannot share a batch.")
        return {"joint_rows": items}
    return default_collate(items)


class MicroenvironmentZarrDataset(Dataset):
    def __init__(self, store_path: Path, config: dict, row_indices: list[int] | None = None,
                 teacher_loss_weight: float = 1.0, *, split_name: str | None = None,
                 trajectory_uids: list[str] | None = None,
                 normalization_context: NativeNormalizationContext | None = None) -> None:
        self.store_path = store_path
        self.config = config
        self.normalization_context = normalization_context
        self.teacher_loss_weight = float(teacher_loss_weight)
        native_conditioning_array_names(config)
        self.native_registry = load_registry(config, normalization_context=normalization_context) if native_enabled(config) else None
        if self.native_registry is not None:
            self.native_records = select_store_records(
                self.native_registry, config, store_path, split_name,
                trajectory_uids=trajectory_uids, row_indices=row_indices, normalization_context=normalization_context)
            descriptor_bindings(self.native_records, self.native_registry, config, normalization_context=normalization_context)
            self.sample_count = len(self.native_records)
            return
        self.trajectory_metadata = read_zarr_array_metadata(store_path, "trajectories")
        self.uncertainty_metadata = read_zarr_array_metadata(store_path, "teacher_uncertainty")
        self.row_indices = list(range(int(self.trajectory_metadata["shape"][0]))) if row_indices is None else [int(index) for index in row_indices]
        self.sample_count = len(self.row_indices)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        if self.native_registry is not None:
            records = [self.native_records[index] for index in indices]
            return native_dataset_items(self.native_registry, records, self.config,
                [self.teacher_loss_weight * float(record["teacher"]["sample_weight"]) for record in records],
                normalization_context=self.normalization_context)
        row_indices = np.asarray([self.row_indices[int(index)] for index in indices], dtype=np.int64)
        trajectories = load_zarr_rows(self.store_path, "trajectories", row_indices)
        uncertainties = load_zarr_rows(self.store_path, "teacher_uncertainty", row_indices)
        native = read_native_conditioning(self.config, self.store_path, row_indices, "zarr_v2")
        return [
            {
                **{key: torch.tensor(value[position], dtype=torch.float32) for key, value in native.items()},
                "trajectory": torch.tensor(trajectories[position], dtype=torch.float32),
                "uncertainty": torch.tensor(uncertainties[position], dtype=torch.float32),
                "teacher_loss_weight": torch.tensor(self.teacher_loss_weight, dtype=torch.float32),
                "sample_ids": torch.tensor(int(indices[position]), dtype=torch.long),
            }
            for position in range(len(indices))
        ]


class MicroenvironmentTeacherFamilyDataset(Dataset):
    def __init__(self, config: dict, split_name: str, *,
                 normalization_context: NativeNormalizationContext | None = None) -> None:
        self.config = config
        self.normalization_context = normalization_context
        self.split_name = str(split_name)
        self.rows: list[dict[str, object]] = []
        self.parameter_prior_family_counts: dict[str, int] = {}
        native_conditioning_array_names(config)
        self.native_registry = load_registry(config, normalization_context=normalization_context) if native_enabled(config) else None
        if self.native_registry is not None:
            self.native_records = select_records(self.native_registry, config, self.split_name, normalization_context=normalization_context)
            descriptor_bindings(self.native_records, self.native_registry, config, normalization_context=normalization_context)
            self.rows = [record["teacher"] for record in self.native_records]
            family_weights = {family["family_key"]: float(family["teacher_loss_weight"]) for family in
                config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]["families"]}
            self.native_family_weights = []
            for row in self.rows:
                if row.get("teacher_family") not in family_weights:
                    raise ValueError("Native teacher_family must bind its configured family loss weight")
                self.native_family_weights.append(family_weights[row["teacher_family"]])
            return
        family_config = config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]
        row_limit = int(family_config["catalog_row_limit_per_family"])
        for family in family_config["families"]:
            self.extend_family_rows(dict(family), row_limit)
        if not self.rows:
            raise ValueError("Joint teacher-family pretraining dataset has no trajectory-backed rows.")

    def catalog_rows(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Teacher family catalog is missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="	"))
        if not rows:
            raise ValueError(f"Teacher family catalog has no data rows: {path}")
        return rows

    def extend_family_rows(self, family: dict[str, object], row_limit: int) -> None:
        family_key = str(family["family_key"])
        catalog_path = resolve_repo_path(str(family["catalog_path"]))
        rows = self.catalog_rows(catalog_path)
        if bool(family["parameter_prior_only"]):
            self.parameter_prior_family_counts[family_key] = len(rows)
            return
        source_collection = str(family["source_collection"])
        split_field = str(family["split_field"])
        trajectory_store_field = str(family["trajectory_store_field"])
        trajectory_index_field = str(family["trajectory_index_field"])
        qc_use_field = str(family.get("qc_use_field", ""))
        allowed_qc_uses = {str(value) for value in family.get("allowed_qc_uses", [])}
        sample_weight_field = str(family.get("sample_weight_field", ""))
        selected_rows: list[dict[str, str]] = []
        for row in rows:
            if source_collection and str(row.get("source_collection", "")) != source_collection:
                continue
            if split_field and str(row.get(split_field, "")) != self.split_name:
                continue
            if qc_use_field and allowed_qc_uses and str(row.get(qc_use_field, "")) not in allowed_qc_uses:
                continue
            selected_rows.append(row)
            if row_limit > 0 and len(selected_rows) >= row_limit:
                break
        if bool(family["required"]) and not selected_rows:
            raise ValueError(f"Required teacher family has no rows for split {self.split_name}: {family_key}")
        for row in selected_rows:
            self.rows.append(
                {
                    "teacher_family": family_key,
                    "trajectory_store_path": resolve_repo_path(str(row[trajectory_store_field])),
                    "trajectory_index": int(row[trajectory_index_field]),
                    "teacher_loss_weight": float(family["teacher_loss_weight"]) * (
                        float(row[sample_weight_field]) if sample_weight_field else 1.0
                    ),
                    "store_format": str(family["store_format"]),
                    "trajectory_array_name": str(family["trajectory_array_name"]),
                    "uncertainty_array_name": str(family["uncertainty_array_name"]),
                    "default_uncertainty_value": float(family["default_uncertainty_value"]),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        if self.native_registry is not None:
            records = [self.native_records[index] for index in indices]
            return native_dataset_items(self.native_registry, records, self.config,
                [self.native_family_weights[index] * float(record["teacher"]["sample_weight"]) for index, record in zip(indices, records)], normalization_context=self.normalization_context)
        outputs: list[dict[str, torch.Tensor] | None] = [None for _ in indices]
        grouped_positions: dict[tuple[str, str, str, str, float], list[tuple[int, int]]] = {}
        for output_position, dataset_index in enumerate(indices):
            row = self.rows[int(dataset_index)]
            group_key = (
                str(row["trajectory_store_path"]),
                str(row["store_format"]),
                str(row["trajectory_array_name"]),
                str(row["uncertainty_array_name"]),
                float(row["default_uncertainty_value"]),
            )
            grouped_positions.setdefault(group_key, []).append((output_position, int(row["trajectory_index"])))
        for group_key, positions in grouped_positions.items():
            store_path_value, store_format, trajectory_array_name, uncertainty_array_name, default_uncertainty_value = group_key
            store_path = Path(store_path_value)
            row_indices = np.asarray([row_index for _, row_index in positions], dtype=np.int64)
            if store_format == "neuronsifter_chunk_store":
                trajectories = load_chunked_rows(store_path, trajectory_array_name, row_indices)
                uncertainties = np.full((trajectories.shape[0], trajectories.shape[-1]), float(default_uncertainty_value), dtype=np.float32)
            elif store_format == "zarr_v2":
                trajectories = load_zarr_rows(store_path, trajectory_array_name, row_indices)
                uncertainties = load_zarr_rows(store_path, uncertainty_array_name, row_indices)
            else:
                raise ValueError(f"Unsupported teacher-family store format: {store_format}")
            native = read_native_conditioning(self.config, store_path, row_indices, store_format)
            if native and store_format == "neuronsifter_chunk_store":
                # Native training cannot turn an absent uncertainty channel into a constant.
                uncertainties = load_chunked_rows(store_path, uncertainty_array_name, row_indices)
            for local_position, (output_position, _) in enumerate(positions):
                row = self.rows[int(indices[output_position])]
                outputs[output_position] = {
                    **{key: torch.tensor(value[local_position], dtype=torch.float32) for key, value in native.items()},
                    "trajectory": torch.tensor(trajectories[local_position], dtype=torch.float32),
                    "uncertainty": torch.tensor(uncertainties[local_position], dtype=torch.float32),
                    "teacher_loss_weight": torch.tensor(float(row["teacher_loss_weight"]), dtype=torch.float32),
                    "sample_ids": torch.tensor(int(indices[output_position]), dtype=torch.long),
                }
        return [output for output in outputs if output is not None]



class NativeJointGroupDataset(Dataset):
    """Index metadata once and load one complete paired group per requested item."""

    def __init__(self, config: dict, split_name: str, *, group_uids: set[str] | None = None,
                 input_locations: NativeInputLocations | None = None):
        self.config, self.split_name = config, split_name
        self.input_locations = input_locations or input_locations_from_config(config)
        self.groups = []
        admitted_groups = set()
        self.group_uids = []
        if group_uids is not None:
            requested = list(group_uids)
            if not requested or any(not isinstance(uid,str) or not uid for uid in requested):
                raise ValueError("Explicit native group selection must contain nonempty identities")
            if len(set(requested)) != len(requested):
                raise ValueError("Duplicate native group selection identities")
            group_uids = set(requested)
        self.family_group_counts = {}
        family_weights = {row["family_key"]: float(row["teacher_loss_weight"]) for row in
            config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]["families"]}
        registries = load_family_registries(config, input_locations=self.input_locations)
        family_scope=config.get("native_scientific_training",{}).get("family_scope")
        if family_scope is not None and (not family_scope or len(set(family_scope))!=len(family_scope)
                or not set(family_scope)<=set(registries)):
            raise ValueError("Declared native family scope is invalid or missing from registry")
        if group_uids is not None:
            roles = {}
            for registry in registries.values():
                for row in registry["teachers"]:
                    roles.setdefault(row["sample_uid"],set()).add(row["split_name"])
            for uid in group_uids:
                if uid not in roles:
                    raise ValueError("Declared native group is missing: "+uid)
                if roles[uid] != {split_name}:
                    raise ValueError("Declared native group has another role: "+uid)
        for family, registry in registries.items():
            if family_scope is not None and family not in family_scope:
                continue
            selected = copy.deepcopy(config)
            selected["native_diffusion_data"]["source_family_uid"] = family
            teachers = [row for row in registry["teachers"] if row["split_name"] == split_name
                        and (group_uids is None or row["sample_uid"] in group_uids)]
            child = SimpleNamespace(native_registry=registry, config=selected,
                                    native_records=[{"teacher": row} for row in teachers], family_weights=family_weights)
            indices = {}
            for i, teacher in enumerate(teachers):
                indices.setdefault(teacher["sample_uid"], []).append(i)
            if not indices:
                raise ValueError("Scientific training requires nonempty roles in every family")
            self.family_group_counts[family] = len(indices)
            admitted_groups.update(indices)
            for sample, rows in indices.items():
                regimens = [teachers[i]["regimen_uid"] for i in rows]
                if len(regimens) < 2 or len(set(regimens)) != len(regimens):
                    raise ValueError("Scientific groups require unique paired actions")
                if len({teachers[i]["posterior_uid"] for i in rows}) != 1:
                    raise ValueError("Paired action targets do not share one complete prior")
                self.groups.append((child, rows, None))
                self.group_uids.append(sample)
        if group_uids is not None and admitted_groups != group_uids:
            raise ValueError("Declared "+split_name+" groups are missing or have another role: "+
                             str(sorted(group_uids-admitted_groups)))

        if self.input_locations:
            self.input_locations.require_groups(split_name, self.group_uids)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        child, indices, _ = self.groups[index]
        teachers = [child.native_records[i]["teacher"] for i in indices]
        records = join_registry(child.native_registry, child.config, teachers, input_locations=self.input_locations)
        original = records[0]["joint_source"]
        regimens = [row["regimen_uid"] for row in teachers]
        if set(regimens) != set(original["particle_programs"]["regimen_uids"]):
            raise ValueError("Scientific groups require every source-declared paired action")
        source = select_joint_source(original, [teachers[0]["sample_uid"]], regimens,
                                     action_policy=native_action_policy(child.config))
        weights = [float(row["sample_weight"]) * child.family_weights[row["teacher_family"]] for row in teachers]
        items = native_dataset_items(child.native_registry, records, child.config, weights, input_locations=self.input_locations)
        return {"native_joint_group": {"items": items, "joint_source": source}}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def build_pretraining_dataset(config: dict, store_path: Path, *, direct_store: bool,
                              single_batch_check: bool) -> Dataset:
    if config.get("native_scientific_training", {}).get("enabled"):
        if direct_store or single_batch_check:
            raise ValueError("Scientific grouped training requires the registry path and complete groups")
        return NativeJointGroupDataset(config, "train",
            group_uids={row["sample_uid"] for row in selected_group_definitions(config,"train")})
    training = config["screening_pipeline"]["screening_benchmark"]["training"]
    split_name = str(training["zarr_loader"]["training_split_name"])
    if direct_store or not bool(training["teacher_family_training"]["enabled"]):
        if native_enabled(config):
            # Native metadata owns split membership, including single-batch checks.
            return MicroenvironmentZarrDataset(store_path, config, split_name=split_name)
        indices = None if single_batch_check else manifest_row_indices(config, store_path, split_name)
        return MicroenvironmentZarrDataset(store_path, config, row_indices=indices)
    return MicroenvironmentTeacherFamilyDataset(config, split_name)


def joint_inference_modules(config: dict, hidden_dim: int, species_count: int):
    training = config["screening_pipeline"]["screening_benchmark"]["training"]
    joint = world_state_posterior_config_from_config(config)
    width = joint.get("observation_feature_width")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("Joint pretraining requires an explicit observation_feature_width before model construction.")

    latent = int(training["latent_dim"])
    encoder = FeedForwardBlock(width, int(training["hidden_dim"]), latent, float(training["dropout_rate"]))
    mask_projection = nn.Linear(width * 2, latent)
    species = {row["species_id"] for row in config["dynamics_ready_registry"]["species"]}
    environment = ("misfolded_protein_load", "aggregate_burden", "neuroinflammation", "clearance_flux",
           "amyloid_beta_soluble", "amyloid_plaque", "phosphorylated_tau", "inflammatory_mediator", "vascular_perivascular_transport")
    electrophysiology = ("circuit_instability", "calcium_concentration", "membrane_voltage", "spike_rate")
    posterior = WorldStatePosterior(latent, species_count, int(joint["posterior_hidden_dim"]),
        float(joint["posterior_dropout_rate"]), 2, sum(name in species for name in environment),
        sum(name in species for name in electrophysiology), joint, hidden_dim)

    return (encoder, mask_projection, posterior), width


def joint_inference_parameter_count(config: dict) -> int:
    if native_direct_mean_config(config) or posterior_family(config) != "finite_joint_particles":
        return 0
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    # Count shapes without allocating or initializing count-only model weights.
    with torch.device("meta"):
        context_width=native_posterior_context_width(config,int(backbone["operator_hidden_dim"]))
        modules, _ = joint_inference_modules(config, context_width, len(config["dynamics_ready_registry"]["species"]))
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


class MicroenvironmentPretrainingModel(nn.Module):
    def __init__(self, config: dict, use_smoke_dimensions: bool, *, conditioning: dict | None = None) -> None:
        super().__init__()
        self.action_policy_name = native_action_policy(config)
        self.scientific_training_enabled = bool(config.get("native_scientific_training", {}).get("enabled"))
        backbone_config = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        species_count = len(config["dynamics_ready_registry"]["species"])
        linear = backbone_config["kind"] == "native_linear_dynamics"
        hidden_dim = 0 if linear else int(backbone_config["operator_hidden_dim"])
        depth = 0 if linear else int(backbone_config["operator_depth"])
        head_count = 1
        if str(backbone_config["kind"]) in {"neuronsifter_world_action", "transolver_microenvironment_operator"}:
            head_count = int(backbone_config["attention_head_count"])
        if use_smoke_dimensions:
            smoke_config = config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]
            hidden_dim = int(smoke_config.get("smoke_hidden_dim", 64))
            depth = int(smoke_config.get("smoke_depth", 2))
            head_count = int(smoke_config.get("smoke_head_count", 4))
        posterior_context_width=native_posterior_context_width(config,hidden_dim) if hidden_dim else hidden_dim
        self.direct_mean_policy = native_direct_mean_config(config)
        self.effective_config = copy.deepcopy(config) if self.direct_mean_policy else None
        if self.direct_mean_policy and use_smoke_dimensions:
            raise ValueError("Direct checkpoints require effective fixture dimensions in config")
        self.operator_kind = str(backbone_config["kind"])
        self.register_buffer("node_adjacency", torch.tensor(node_adjacency_matrix(config), dtype=torch.float32), persistent=False)
        self.field_normalization = NativeFieldNormalization(normalization_from_config(config), species_count,
            input_locations=input_locations_from_config(config))
        if self.scientific_training_enabled:
            self.native_scientific_profile=native_training_profile(config,self.field_normalization.identity)
        if not self.direct_mean_policy:
            self.state_encoder = nn.Linear(species_count, hidden_dim)
            self.condition_encoder = nn.Linear(species_count, hidden_dim)
        self.factor_operator_blocks = nn.ModuleList()
        if linear:
            self.operator = None
        elif backbone_config["kind"] == "transolver_microenvironment_operator":
            mechanism_moe = backbone_config["mechanism_moe"]
            slice_count = int(backbone_config["physical_state_slice_count"])
            if use_smoke_dimensions:
                slice_count = int(config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"].get("smoke_slice_count", 16))
            encoder_decoder_pretraining = backbone_config["encoder_decoder_pretraining"]
            self.operator = TransolverMicroenvironmentOperator(
                hidden_dim,
                int(encoder_decoder_pretraining["encoder_depth"]) if not use_smoke_dimensions else depth,
                int(encoder_decoder_pretraining["decoder_depth"]) if not use_smoke_dimensions else 0,
                head_count,
                slice_count,
                int(backbone_config["node_coordinate_dim"]),
                float(backbone_config["operator_dropout_rate"]),
                [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
                int(mechanism_moe["top_k"]),
                float(mechanism_moe["capacity_factor"]),
                float(mechanism_moe["entropy_loss_weight"]),
                float(mechanism_moe["load_balance_loss_weight"]),
                float(mechanism_moe["expert_capacity_loss_weight"]),
                int(mechanism_moe["expert_hidden_multiplier"]),
                int(backbone_config["physics_mlp_multiplier"]),
                bool(encoder_decoder_pretraining["checkpoint_attention_blocks"]),
            )
        elif backbone_config["kind"] == "neuronsifter_world_action":
            self.operator = native_world_action_from_config(backbone_config,hidden_dim=hidden_dim,
                                                           depth=depth,head_count=head_count)
        elif backbone_config["kind"] == "factor_coupled_neural_operator":
            self.operator = nn.Identity()
            self.factor_operator_blocks = nn.ModuleList(
                [
                    ActionConditionedOperatorBlock(
                        hidden_dim,
                        float(backbone_config["operator_dropout_rate"]),
                    )
                    for _ in range(depth)
                ]
            )
        else:
            raise ValueError(f"Unsupported pretraining operator kind: {backbone_config['kind']}")
        mddm_config = quantized_mddm_rollout_from_config(config, backbone_config) if self.operator_kind == "neuronsifter_world_action" else None
        action_policy = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"]
        self.quantized_mddm = QuantizedMddmRolloutOperator(
            mddm_config, hidden_dim, species_count, action_policy=action_policy,
            quantity_profile=observable_profile(config),posterior_context_dim=posterior_context_width,
            scalar_exposure_definition=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["policies"].get("scalar_exposure", {}).get("native_definition")
        ) if mddm_config and mddm_config["enabled"] else None
        self.direct_mean = None
        if self.direct_mean_policy:
            if linear:
                domain = direct_model_identity(config)["linear_domain"]
                self.direct_mean = NativeLinearDynamics(domain, self.direct_mean_policy["posterior_context_dim"])
            else:
                self.direct_mean = NativeTransolverMean(species_count, hidden_dim, self.direct_mean_policy["posterior_context_dim"])
            # The referenced frozen authority has its own checkpoint and cost. Do
            # not duplicate its parameters in the trainable predictor state_dict.
            self.conditioning = conditioning if conditioning is not None else load_frozen_conditioner(config)
            self.conditioning["model"].requires_grad_(False).eval()
        else:
            self.state_decoder = nn.Linear(hidden_dim, species_count)
            self.counterfactual_head = nn.Linear(hidden_dim, species_count)
        self.world_state_posterior = None
        if not self.direct_mean_policy and posterior_family(config) == "finite_joint_particles":
            modules, self.observation_feature_width = joint_inference_modules(config, posterior_context_width, species_count)
            self.observation_encoder, self.retained_mask_projection, self.world_state_posterior = modules

    def train(self,mode: bool=True):
        super().train(mode)
        if getattr(self,"frozen_conditioning_identity",None):
            for name in ("observation_encoder","retained_mask_projection","world_state_posterior"):
                getattr(self,name).eval()
        return self

    def infer_joint_source(self, source: dict):
        if self.world_state_posterior is None:
            raise ValueError("Raw joint sources require the configured joint family.")
        support, observations = parse_joint_source(source, next(self.parameters()).device, action_policy=self.action_policy_name)
        return self.infer_observation_view(support, observations)

    def infer_observation_view(self, support, observations):
        """Infer a posterior-only training view; no compiled action is rebound."""
        support.validate()
        values, masks = retained_encoder_inputs(observations, self.observation_feature_width, support=support,
            assays=self.world_state_posterior.joint_config.get("registered_occupancy_assays", {}))
        encoded = encode_retained_observations(self.observation_encoder, self.retained_mask_projection, values, masks)
        return self.world_state_posterior.infer_joint(encoded, values, support, observations, self.field_normalization)

    def factor_coupled_diagnostics(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = encoded.sum() * 0.0
        return {
            "expert_utilization": torch.ones(1, dtype=encoded.dtype, device=encoded.device),
            "router_entropy": zero,
            "router_regularization_loss": zero,
            "slice_entropy": zero,
        }

    def forward(self, state: torch.Tensor | None, coordinates: torch.Tensor | None, diffusion_batch: dict[str, torch.Tensor] | None = None, *, joint_source: dict | None = None, learned_teacher=None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.direct_mean is not None:
            if joint_source is not None:
                raise ValueError("Direct mean requires the shared prepared posterior, not another fitted posterior")
            if self.operator_kind == "native_linear_dynamics":
                return self.direct_mean(diffusion_batch or {}, self.field_normalization)
            regions = len(self.effective_config["spatial_atlas_registry"]["regions"])
            compartments = len(self.effective_config["screening_pipeline"]["pkpd_ready_registry"]["compartments"])
            coords = node_coordinates(regions, compartments, self.node_adjacency.device)[0, 0]
            return self.direct_mean(diffusion_batch or {}, self.operator, self.field_normalization,
                                    self.node_adjacency, coords)
        if joint_source is not None:
            return self.infer_joint_source(joint_source)
        if self.quantized_mddm is not None:
            required = {"trajectory", "initial_state", "occupancy", "support", "physical_time", "posterior_context", "teacher_confidence"}
            if self.quantized_mddm.quantity_profile is not None:
                required.add("quantity_support")
                required.update(FORCING_ARRAY_NAMES)
            if self.quantized_mddm.action_policy_name in {"drug_id", "target_id"}:
                required.add("action_identity")
            if self.quantized_mddm.action_policy_name == "scalar_exposure":
                required.update({"scalar_exposure_auc", "scalar_exposure_interval"})
            missing = required - set(diffusion_batch or {})
            if missing:
                raise ValueError(f"Native pretraining requires an admitted conditioning batch; missing {sorted(missing)}. Trajectory-only stores do not supply compiled programs/posterior context.")
            self.field_normalization.assert_ids(diffusion_batch.get("normalization_id"))
            teacher = diffusion_batch["trajectory"].unsqueeze(1)
            initial = diffusion_batch["initial_state"].unsqueeze(1)
            context = {key: diffusion_batch[key].unsqueeze(1) for key in ("occupancy", "support", "physical_time", "posterior_context")}
            if "quantity_support" in diffusion_batch:
                context["quantity_support"] = diffusion_batch["quantity_support"].unsqueeze(1)
                context.update({name: diffusion_batch[name].unsqueeze(1) for name in FORCING_ARRAY_NAMES})
            if "action_identity" in diffusion_batch:
                context["action_identity"] = diffusion_batch["action_identity"].unsqueeze(1)
            if self.quantized_mddm.action_policy_name == "scalar_exposure":
                for key in ("scalar_exposure_auc", "scalar_exposure_interval"):
                    context[key] = diffusion_batch[key].unsqueeze(1)
                if "scalar_exposure_site_support" in diffusion_batch:
                    context["scalar_exposure_site_support"] = diffusion_batch["scalar_exposure_site_support"].unsqueeze(1)
            if learned_teacher is not None:
                identity=getattr(self,"native_learned_teacher_identity",None)
                if not identity or getattr(learned_teacher,"native_teacher_checkpoint_sha256",None)!=identity["binding"]["checkpoint_sha256"]:
                    raise ValueError("Native learned teacher differs from the bound selected checkpoint")
            result = self.quantized_mddm(
                torch.zeros_like(teacher), self.training, initial_state=initial,
                teacher_trajectory=teacher, teacher_confidence=diffusion_batch["teacher_confidence"].reshape(-1, 1),
                conditioning=context, native_operator=self.operator, adjacency=self.node_adjacency,
                coordinates=coordinates[0, 0], sample_ids=diffusion_batch.get("sample_ids"),
                field_normalization=self.field_normalization,
                score_function_training=self.scientific_training_enabled,
                return_samples=self.scientific_training_enabled, learned_teacher=learned_teacher,
            )
            if learned_teacher is not None:
                self.native_teacher_forward_calls+=result["learned_teacher_kl_per_action"].numel()
            predicted = result["training_reconstruction"][:, 0, 1:]
            prediction = predicted.reshape(predicted.shape[0], predicted.shape[1], -1, predicted.shape[-1])
            if self.quantized_mddm.quantity_profile is not None:
                # Observable predictions all traverse the same native operator.
                counterfactual = prediction
            else:
                initial_hidden = self.state_encoder(self.field_normalization.normalize(diffusion_batch["initial_state"]).flatten(1, 2))
                counterfactual = self.field_normalization.inverse(
                    self.counterfactual_head(initial_hidden).unsqueeze(1).expand_as(prediction))
            return {"state_prediction": prediction, "counterfactual_prediction": counterfactual,
                    "router_diagnostics": result["sampler_diagnostics"], "rollout_auxiliary_loss": result["rollout_auxiliary_loss"],
                    "field_normalization_diagnostics": result["field_normalization_diagnostics"],
                    "observable_prediction": result.get("observable_prediction"),
                    "quantity_support": result.get("quantity_support"),
                    "action_conditioning": result.get("action_conditioning"),
                    **{key: result[key] for key in ("sample_trajectories", "sample_log_probabilities",
                         "unprojected_sample_trajectories", "training_reconstruction", "learned_teacher_kl") if key in result}}
        hidden = self.state_encoder(state)
        conditioning = self.condition_encoder(state.mean(dim=-2, keepdim=True).expand_as(state))
        if self.operator_kind == "neuronsifter_world_action":
            encoded, router_diagnostics = self.operator(hidden, conditioning, coordinates, self.node_adjacency.to(device=hidden.device, dtype=hidden.dtype))
        elif self.operator_kind == "transolver_microenvironment_operator":
            encoded, router_diagnostics = self.operator(hidden, conditioning, coordinates)
        else:
            encoded = hidden
            for block in self.factor_operator_blocks:
                neighborhood_state = torch.einsum("nm,btmh->btnh", self.node_adjacency.to(device=hidden.device, dtype=hidden.dtype), encoded)
                encoded = block(encoded, neighborhood_state, conditioning)
            router_diagnostics = self.factor_coupled_diagnostics(encoded)
        return {
            "state_prediction": self.state_decoder(encoded),
            "counterfactual_prediction": self.counterfactual_head(encoded),
            "router_diagnostics": router_diagnostics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the AD full microenvironment Transolver++/MoE world model on Zarr trajectory chunks.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--store", default="")
    parser.add_argument("--stage-native-inputs", type=Path, help="Stage selected admitted inputs and exit before model initialization.")
    parser.add_argument("--native-input-manifest", type=Path)
    parser.add_argument("--reuse-native-inputs", type=Path,
                        help="Reuse immutable staged files on the destination filesystem after checking admitted bytes.")
    parser.add_argument("--native-input-role", action="append", choices=["train", "development", "validation", "calibration"])
    parser.add_argument("--native-calibration-reservation", action="append", choices=["fit", "check", "threshold"])
    parser.add_argument("--checkpoint-scale", default="")
    parser.add_argument("--inference-profile", default="")
    parser.add_argument("--distributed-strategy", default="")
    parser.add_argument("--one-batch-smoke", action="store_true")
    parser.add_argument("--single-batch-full-model-check", action="store_true")
    parser.add_argument("--native-consumer-check", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def selected_checkpoint_scale(args: argparse.Namespace, config: dict) -> str:
    if args.checkpoint_scale:
        return str(args.checkpoint_scale)
    return default_checkpoint_scale(config, smoke=bool(args.one_batch_smoke))


def selected_inference_profile(args: argparse.Namespace, config: dict, checkpoint_scale: str) -> str:
    if args.inference_profile:
        return str(args.inference_profile)
    return default_inference_profile(config, checkpoint_scale)


def pretraining_artifact_root(config: dict, single_batch_check: bool) -> Path:
    artifact_root = resolve_repo_path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"])
    if single_batch_check:
        return artifact_root
    loader_config = config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]
    subdir_template = str(loader_config.get("formal_artifact_subdir_template", "")).strip()
    if not subdir_template:
        return artifact_root
    benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    profile_name = str(config.get("selected_paper_experiment_profile", ""))
    selected_seed = str(resolve_training_seed(config, require_selected=True)["base_training_seed"])
    return artifact_root / subdir_template.format(
        benchmark_name=benchmark_name,
        profile_name=profile_name,
        seed=selected_seed,
    )


def manifest_row_indices(config: dict, store_path: Path, split_name: str) -> list[int]:
    preprocessing = config["ai_ready_preprocessing"]
    manifest_path = resolve_repo_path(preprocessing["local_root"]) / preprocessing["relative_paths"]["simulation_manifest"]
    if not manifest_path.exists():
        return []
    source_collection = str(config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]["source_collection"])
    store_relative = str(store_path.resolve().relative_to(REPO_ROOT))
    row_indices: list[int] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["source_collection"] != source_collection:
                continue
            if row["split_name"] != split_name:
                continue
            if row["trajectory_store_path"] != store_relative:
                continue
            row_indices.append(int(row["trajectory_index"]))
    return row_indices


def data_loader_kwargs(config: dict) -> dict[str, object]:
    loader_policy = dict(config["screening_pipeline"]["screening_benchmark"]["training"].get("data_loader", {}))
    num_workers = int(loader_policy.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.data_loader.num_workers must be non-negative.")
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": bool(loader_policy.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_policy.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(loader_policy.get("prefetch_factor", 2))
        if int(kwargs["prefetch_factor"]) <= 0:
            raise ValueError("training.data_loader.prefetch_factor must be positive.")
    kwargs["collate_fn"] = collate_native_sources
    return kwargs


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


def write_runtime_efficiency_profile(config: dict, payload: dict[str, object]) -> None:
    json_path, md_path = runtime_output_paths(
        config,
        "runtime_efficiency_profile_json",
        "runtime_efficiency_profile_md",
    )
    write_json(json_path, payload)
    write_runtime_markdown(md_path, "Runtime Efficiency Profile", payload)


def node_coordinates(region_count: int, compartment_count: int, device: torch.device) -> torch.Tensor:
    coordinates = []
    region_denominator = max(region_count - 1, 1)
    compartment_denominator = max(compartment_count - 1, 1)
    for region_index in range(region_count):
        for compartment_index in range(compartment_count):
            coordinates.append([float(region_index) / float(region_denominator), float(compartment_index) / float(compartment_denominator)])
    return torch.tensor(coordinates, dtype=torch.float32, device=device).view(1, 1, region_count * compartment_count, 2)


def teacher_sample_weight(batch: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    """Raw teacher confidence times QC/family weight, once per sample.

    Precision remains separate: scalar native losses use a within-sample mean;
    uncertainty-weighted auxiliary MSE retains its per-channel precision.
    """
    weight = batch["teacher_loss_weight"].to(reference).detach()
    if weight.shape != (reference.shape[0],) or not bool((torch.isfinite(weight) & (weight >= 0)).all()):
        raise ValueError("Teacher QC/family weight must be finite, nonnegative and sample-aligned.")
    confidence = batch.get("teacher_confidence")
    if confidence is not None:
        confidence = confidence.to(reference).detach()
        if confidence.shape != weight.shape or not bool((torch.isfinite(confidence) & (confidence >= 0)).all()):
            raise ValueError("Raw teacher confidence must be finite, nonnegative and sample-aligned.")
        weight = weight * confidence
    return weight


def weighted_batch_mean(per_sample_loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    weights = sample_weight.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    return torch.sum(per_sample_loss * weights) / torch.clamp(weights.sum(), min=torch.finfo(per_sample_loss.dtype).eps)


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


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    if distributed_is_initialized():
        return int(dist.get_rank())
    return 0


def is_primary_process() -> bool:
    return distributed_rank() == 0


def initialize_device_and_distributed(args: argparse.Namespace, fsdp_profile: dict[str, object]) -> tuple[torch.device, str]:
    configured_strategy = str(args.distributed_strategy or fsdp_profile["distributed_strategy"])
    if configured_strategy == "native_pytorch_fsdp":
        if not torch.cuda.is_available():
            raise RuntimeError("native_pytorch_fsdp requires CUDA devices.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        return device, configured_strategy
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), configured_strategy
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda"), configured_strategy
    return torch.device(args.device), configured_strategy


def sharding_strategy(strategy_name: str) -> ShardingStrategy:
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    if strategy_name not in strategy_map:
        raise ValueError(f"Unsupported native FSDP sharding strategy: {strategy_name}")
    return strategy_map[strategy_name]


def wrap_fsdp_model(
    model: MicroenvironmentPretrainingModel,
    distributed_strategy: str,
    fsdp_profile: dict[str, object],
    device: torch.device,
) -> nn.Module:
    if distributed_strategy != "native_pytorch_fsdp":
        return model
    precision_dtype = configured_torch_dtype(str(fsdp_profile["mixed_precision_dtype"]))
    return FSDP(
        model,
        sharding_strategy=sharding_strategy(str(fsdp_profile["sharding_strategy"])),
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            reduce_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        device_id=device,
        use_orig_params=True,
    )


def finalize_distributed() -> None:
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def training_memory_policy(config: dict, single_batch_check: bool) -> dict[str, object]:
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    memory_policy = dict(training_config["memory_policy"])
    if single_batch_check:
        mixed_precision = dict(memory_policy["mixed_precision"])
        mixed_precision["enabled"] = False
        smoke_batch_size = int(training_config["zarr_loader"]["one_batch_smoke_batch_size"])
        return {
            "effective_batch_size": smoke_batch_size,
            "micro_batch_size": smoke_batch_size,
            "gradient_accumulation_steps": 1,
            "mixed_precision": mixed_precision,
        }
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


def supported_sample_mean(value: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    if support is None:
        return value.reshape(value.shape[0], -1).mean(dim=1)
    if support.dtype != torch.bool:
        raise ValueError("Loss quantity support must be boolean.")
    mask = torch.broadcast_to(support, value.shape)
    denominator = mask.reshape(value.shape[0], -1).sum(dim=1)
    if (denominator == 0).any():
        raise ValueError("Loss has no supported quantities for a sample.")
    return (value * mask).reshape(value.shape[0], -1).sum(dim=1) / denominator


def supported_sample_variance(value: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    if support is None:
        return value.reshape(value.shape[0], -1).var(dim=1, unbiased=False)
    mean = supported_sample_mean(value, support).reshape(value.shape[0], *([1] * (value.ndim - 1)))
    return supported_sample_mean((value - mean).square(), support)


def masked_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, mask_ratio: float, sample_weight: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    flat_index = torch.arange(target.shape[-2], device=target.device)
    period = max(int(round(1.0 / mask_ratio)), 1)
    mask = (flat_index % period == 0).view(1, 1, -1, 1)
    squared_error = (prediction - target) ** 2
    expanded_mask = mask.expand_as(squared_error)
    if support is not None:
        expanded_mask = expanded_mask & support
    per_sample_loss = (squared_error * expanded_mask).reshape(prediction.shape[0], -1).sum(dim=1) / torch.clamp(
        expanded_mask.reshape(prediction.shape[0], -1).sum(dim=1),
        min=torch.finfo(prediction.dtype).eps,
    )
    return weighted_batch_mean(per_sample_loss, sample_weight)


def uncertainty_weighted_mse(prediction: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor, config: dict, sample_weight: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    weighting = config["screening_pipeline"]["screening_benchmark"]["training"]["uncertainty_weighting"]
    precision = 1.0 / torch.clamp(uncertainty, min=float(weighting["minimum_teacher_precision"]))
    precision = torch.clamp(precision, min=float(weighting["minimum_teacher_precision"]), max=float(weighting["maximum_teacher_precision"]))
    while precision.ndim < prediction.ndim:
        precision = precision.unsqueeze(1)
    per_sample_loss = supported_sample_mean(precision * (prediction - target) ** 2, support)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def zero_loss_like(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.sum() * 0.0


def species_indices(config: dict, species_names: list[str]) -> list[int]:
    species_id_to_index = {
        str(row["species_id"]): index
        for index, row in enumerate(config["dynamics_ready_registry"]["species"])
    }
    missing_species = sorted(set(species_names) - set(species_id_to_index))
    if missing_species:
        raise ValueError(f"JEPA species mask references unknown species: {missing_species}")
    return [species_id_to_index[str(species_name)] for species_name in species_names]


def masked_species_group_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    sample_weight: torch.Tensor,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    objective_config = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]["masked_species_group_prediction"]
    if not bool(objective_config["enabled"]):
        return zero_loss_like(prediction)
    selected_species = []
    for group_species in dict(objective_config["groups"]).values():
        selected_species.extend(str(species_name) for species_name in group_species)
    selected_indices = species_indices(config, selected_species)
    squared_error = (prediction[..., selected_indices] - target[..., selected_indices]) ** 2
    per_sample_loss = supported_sample_mean(squared_error, None if support is None else support[..., selected_indices])
    return weighted_batch_mean(per_sample_loss, sample_weight)


def masked_compartment_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    sample_weight: torch.Tensor,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    objective_config = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]["masked_compartment_prediction"]
    if not bool(objective_config["enabled"]):
        return zero_loss_like(prediction)
    compartment_ids = [str(row["compartment_id"]) for row in config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]]
    selected_compartments = [str(compartment_id) for compartment_id in objective_config["compartment_ids"]]
    missing_compartments = sorted(set(selected_compartments) - set(compartment_ids))
    if missing_compartments:
        raise ValueError(f"JEPA compartment mask references unknown compartments: {missing_compartments}")
    selected_compartment_indices = {compartment_ids.index(compartment_id) for compartment_id in selected_compartments}
    node_indices = torch.arange(prediction.shape[-2], device=prediction.device)
    compartment_index = node_indices % len(compartment_ids)
    node_mask = torch.tensor(
        [index in selected_compartment_indices for index in compartment_index.detach().cpu().tolist()],
        dtype=torch.bool,
        device=prediction.device,
    )
    squared_error = (prediction[..., node_mask, :] - target[..., node_mask, :]) ** 2
    per_sample_loss = supported_sample_mean(squared_error, None if support is None else support[..., node_mask, :])
    return weighted_batch_mean(per_sample_loss, sample_weight)


def counterfactual_latent_delta_prediction_loss(
    counterfactual_prediction: torch.Tensor,
    state: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    sample_weight: torch.Tensor,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    objective_config = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]["counterfactual_latent_delta_prediction"]
    if not bool(objective_config["enabled"]):
        return zero_loss_like(counterfactual_prediction)
    predicted_delta = counterfactual_prediction - state
    target_delta = target - state
    per_sample_loss = supported_sample_mean((predicted_delta - target_delta).square(), support)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def rank_order_distillation_loss(prediction: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    predicted_flat, target_flat = prediction.flatten(1), target.flatten(1)
    mask = None if support is None else support.flatten(1)
    if mask is not None:
        predicted_flat = predicted_flat.masked_fill(~mask, -torch.inf)
        target_flat = target_flat.masked_fill(~mask, -torch.inf)
    prediction_order = torch.softmax(predicted_flat, dim=-1)
    target_order = torch.softmax(target_flat, dim=-1)
    per_sample_loss = supported_sample_mean((prediction_order - target_order).square(), mask)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def uncertainty_calibration_distillation_loss(prediction: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor, sample_weight: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    predicted_dispersion = (prediction.flatten(1).std(dim=1, unbiased=False) if support is None
                            else torch.stack([row[mask].std(unbiased=False) for row, mask in zip(prediction, support)]))
    target_dispersion = (target.flatten(1).std(dim=1, unbiased=False) if support is None
                         else torch.stack([row[mask].std(unbiased=False) for row, mask in zip(target, support)]))
    teacher_uncertainty = (uncertainty.reshape(uncertainty.shape[0], -1).mean(dim=1) if support is None
                           else supported_sample_mean(uncertainty[:, None, None, :].expand_as(prediction), support))
    per_sample_loss = (predicted_dispersion - (target_dispersion + teacher_uncertainty)).pow(2)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def tri_consistency_distillation_loss(prediction: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    predicted_reliability = torch.exp(-supported_sample_variance(prediction, support))
    target_reliability = torch.exp(-supported_sample_variance(target, support))
    return weighted_batch_mean((predicted_reliability - target_reliability).pow(2), sample_weight)


def expert_diagnostics_distillation_loss(router_diagnostics: dict[str, torch.Tensor]) -> torch.Tensor:
    utilization = router_diagnostics["expert_utilization"]
    target_utilization = torch.full_like(utilization, 1.0 / float(utilization.numel()))
    return (utilization - target_utilization).pow(2).mean()


def objective_loss(output: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch: dict[str, torch.Tensor], config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    objectives = config["screening_pipeline"]["screening_benchmark"]["training"]["pretraining_objectives"]
    jepa_objectives = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]
    distillation_objectives = config["neuronsifter_scale_registry"]["distillation_losses"]
    trajectory = batch["trajectory"]
    uncertainty = batch["uncertainty"]
    state = trajectory[:, :-1].reshape(trajectory.shape[0], trajectory.shape[1] - 1, -1, trajectory.shape[-1])
    target_next = trajectory[:, 1:].reshape(trajectory.shape[0], trajectory.shape[1] - 1, -1, trajectory.shape[-1])
    support = batch.get("quantity_support")
    support = None if support is None else support[:, 1:].reshape_as(target_next)
    prediction = output["state_prediction"]
    counterfactual_prediction = output["counterfactual_prediction"]
    if "rollout_auxiliary_loss" in output and "teacher_confidence" not in batch:
        raise ValueError("Native teacher-dependent objectives require raw teacher confidence.")
    sample_weight = teacher_sample_weight(batch, prediction)
    reconstruction_loss = masked_reconstruction_loss(prediction, target_next, float(objectives["masked_state_ratio"]), sample_weight, support)
    next_state_loss = uncertainty_weighted_mse(prediction, target_next, uncertainty, config, sample_weight, support)
    long_horizon_loss = uncertainty_weighted_mse(prediction[:, -1:], target_next[:, -1:], uncertainty, config, sample_weight, None if support is None else support[:, -1:])
    distillation_loss = uncertainty_weighted_mse(prediction, target_next, uncertainty, config, sample_weight, support)
    counterfactual_loss = weighted_batch_mean(supported_sample_mean((counterfactual_prediction - state).square(), support), sample_weight)
    router_diagnostics = output["router_diagnostics"]
    router_loss = router_diagnostics["router_regularization_loss"]
    species_group_loss = masked_species_group_prediction_loss(prediction, target_next, config, sample_weight, support)
    compartment_loss = masked_compartment_prediction_loss(prediction, target_next, config, sample_weight, support)
    latent_delta_loss = counterfactual_latent_delta_prediction_loss(counterfactual_prediction, state, target_next, config, sample_weight, support)
    rank_distillation_loss = rank_order_distillation_loss(prediction, target_next, sample_weight, support)
    uncertainty_distillation_loss = uncertainty_calibration_distillation_loss(prediction, target_next, uncertainty, sample_weight, support)
    tri_distillation_loss_value = tri_consistency_distillation_loss(prediction, target_next, sample_weight, support)
    expert_diagnostics_loss = expert_diagnostics_distillation_loss(router_diagnostics)
    distillation_terms = dict(distillation_objectives["terms"])
    teacher_dependent_loss = (
        output.get("rollout_auxiliary_loss", prediction.new_zeros(()))
        + float(objectives["masked_state_reconstruction_weight"]) * reconstruction_loss
        + float(objectives["next_state_prediction_weight"]) * next_state_loss
        + float(objectives["long_horizon_rollout_weight"]) * long_horizon_loss
        + float(objectives["teacher_distillation_weight"]) * distillation_loss
        + float(objectives["counterfactual_consistency_weight"]) * counterfactual_loss
        + float(jepa_objectives["masked_species_group_prediction"]["weight"]) * species_group_loss
        + float(jepa_objectives["masked_compartment_prediction"]["weight"]) * compartment_loss
        + float(jepa_objectives["counterfactual_latent_delta_prediction"]["weight"]) * latent_delta_loss
        + float(distillation_terms["rank_order"]["weight"]) * rank_distillation_loss
        + float(distillation_terms["uncertainty_calibration"]["weight"]) * uncertainty_distillation_loss
        + float(distillation_terms["tri_consistency"]["weight"]) * tri_distillation_loss_value
    )
    total_loss = (
        teacher_dependent_loss
        + float(objectives["router_regularization_weight"]) * router_loss
        + float(distillation_terms["expert_diagnostics"]["weight"]) * expert_diagnostics_loss
    )
    return total_loss, {
        "masked_state_reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "next_state_prediction_loss": float(next_state_loss.detach().cpu()),
        "long_horizon_rollout_loss": float(long_horizon_loss.detach().cpu()),
        "teacher_distillation_loss": float(distillation_loss.detach().cpu()),
        "counterfactual_consistency_loss": float(counterfactual_loss.detach().cpu()),
        "router_regularization_loss": float(router_loss.detach().cpu()),
        "masked_species_group_prediction_loss": float(species_group_loss.detach().cpu()),
        "masked_compartment_prediction_loss": float(compartment_loss.detach().cpu()),
        "counterfactual_latent_delta_prediction_loss": float(latent_delta_loss.detach().cpu()),
        "rank_order_distillation_loss": float(rank_distillation_loss.detach().cpu()),
        "uncertainty_calibration_distillation_loss": float(uncertainty_distillation_loss.detach().cpu()),
        "tri_consistency_distillation_loss": float(tri_distillation_loss_value.detach().cpu()),
        "expert_diagnostics_distillation_loss": float(expert_diagnostics_loss.detach().cpu()),
        "native_masked_diffusion_loss": float(output.get("rollout_auxiliary_loss", prediction.new_zeros(())).detach().cpu()),
        "teacher_dependent_loss": float(teacher_dependent_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }


def validate_default_parameter_budget(config: dict) -> int:
    backbone_config = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    budget = backbone_config["parameter_budget"]
    if backbone_config["kind"] == "native_linear_dynamics":
        identity = direct_model_identity(config)
        domain = identity["linear_domain"]
        regions, compartments, species = (len(domain[key]) for key in ("region_id", "compartment_id", "species_id"))
        cells = regions * compartments
        state = cells * species
        # A, B, C and b only. The common conditioner remains separately frozen.
        parameter_count = state * (state + 2*cells + cells*identity["source_context_width"] + 1)
    elif backbone_config["kind"] == "transolver_microenvironment_operator":
        mechanism_moe = backbone_config["mechanism_moe"]
        parameter_count = transolver_microenvironment_active_parameters(
            int(backbone_config["operator_hidden_dim"]),
            int(backbone_config["encoder_decoder_pretraining"]["encoder_depth"]) + int(backbone_config["encoder_decoder_pretraining"]["decoder_depth"]),
            int(backbone_config["attention_head_count"]),
            int(backbone_config["physical_state_slice_count"]),
            int(backbone_config["node_coordinate_dim"]),
            len(config["dynamics_ready_registry"]["species"]),
            9,
            int(config["screening_pipeline"]["screening_benchmark"]["training"]["latent_dim"]) * 3,
            len(mechanism_moe["expert_names"]),
            int(mechanism_moe["top_k"]),
            int(mechanism_moe["expert_hidden_multiplier"]),
            int(backbone_config["physics_mlp_multiplier"]),
        )
    elif backbone_config["kind"] == "neuronsifter_world_action":
        mechanism_moe = backbone_config["mechanism_moe"]
        parameter_count = neuronsifter_world_action_active_parameters(
            int(backbone_config["operator_hidden_dim"]),
            int(backbone_config["operator_depth"]),
            int(backbone_config["attention_head_count"]),
            int(backbone_config["mechanism_state_token_count"]),
            int(backbone_config["node_coordinate_dim"]),
            len(config["dynamics_ready_registry"]["species"]),
            9,
            int(config["screening_pipeline"]["screening_benchmark"]["training"]["latent_dim"]) * 3,
            len(mechanism_moe["expert_names"]),
            int(mechanism_moe["top_k"]),
            int(mechanism_moe["expert_hidden_multiplier"]),
            int(backbone_config["dit_mlp_multiplier"]),
            int(backbone_config["operator_splitting"]["bottleneck_divisor"]),
        )
    elif backbone_config["kind"] == "factor_coupled_neural_operator":
        hidden_dim = int(backbone_config["operator_hidden_dim"])
        species_count = len(config["dynamics_ready_registry"]["species"])
        latent_dim = int(config["screening_pipeline"]["screening_benchmark"]["training"]["latent_dim"])
        depth = int(backbone_config["operator_depth"])
        parameter_count = (
            species_count * hidden_dim * 2
            + hidden_dim * species_count
            + depth * hidden_dim * hidden_dim * 6
            + latent_dim * hidden_dim * 3
        )
    else:
        raise ValueError(f"Unsupported parameter budget operator kind: {backbone_config['kind']}")
    if backbone_config["kind"] != "native_linear_dynamics":
        parameter_count += joint_inference_parameter_count(config)
    validate_parameter_budget(parameter_count, int(budget["active_parameter_lower_bound"]), int(budget["active_parameter_upper_bound"]))
    return parameter_count



def validate_actual_parameter_budget(model: nn.Module, config: dict) -> int:
    """Enforce the configured total budget, including observation/posterior modules."""
    total = sum(parameter.numel() for parameter in model.parameters())
    budget = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["parameter_budget"]
    validate_parameter_budget(total, int(budget["active_parameter_lower_bound"]), int(budget["active_parameter_upper_bound"]))
    return total



def supported_teacher_precision(uncertainty: torch.Tensor, config: dict, quantity_mask: torch.Tensor | None = None) -> torch.Tensor:
    weighting=config["screening_pipeline"]["screening_benchmark"]["training"]["uncertainty_weighting"]
    if not (torch.isfinite(uncertainty) & (uncertainty >= 0)).all():
        raise ValueError("Native teacher uncertainty must be finite and nonnegative.")
    precision=uncertainty.clamp_min(float(weighting["minimum_teacher_precision"])).reciprocal().clamp(
        min=float(weighting["minimum_teacher_precision"]),max=float(weighting["maximum_teacher_precision"]))
    if quantity_mask is None:
        return precision.flatten(1).mean(-1)
    support=quantity_mask.flatten(1,-2).any(1)
    return supported_sample_mean(precision,support)



def controlled_belief_regularizers(model, source: dict, posterior, config: dict):
    """Held sensing blocks and information-preserving Gaussian nuisance shifts."""
    policy = config["native_scientific_training"]
    support, observations = parse_joint_source(source, next(model.parameters()).device,
                                              action_policy=native_action_policy(config))
    held_terms = []
    assays = model.world_state_posterior.joint_config.get("registered_occupancy_assays", {})
    for block in policy["held_sensing_encoder_indices"]:
        view = copy.deepcopy(source)
        held_likelihood = posterior.weights.new_zeros(posterior.weights.shape)
        retained_count = 0
        for i, observation in enumerate(observations):
            mask = torch.tensor([index in block for index in observation.encoder_indices],
                                device=observation.values.device)
            held = replace(observation, observed=observation.observed & mask,
                           left_censored=observation.left_censored & mask)
            count = int((held.observed | held.left_censored).sum())
            if count:
                parameters, nuisance = support.validate_observation(held, assays)
                held_likelihood = held_likelihood + retained_log_likelihood(held, parameters, nuisance, assays)
                retained_count += count
            visible = replace(observation, observed=observation.observed & ~mask,
                              left_censored=observation.left_censored & ~mask,
                              values=torch.where(observation.observed & ~mask, observation.values,
                                                 torch.zeros_like(observation.values)))
            view["observations"][i] = observation_to_payload(visible)
        if retained_count:
            inferred = model(None, None, joint_source=view)
            held_terms.append(-torch.logsumexp(inferred.log_weights + held_likelihood, -1).mean() / retained_count)
    if not held_terms:
        raise ValueError("No observed controlled sensing block supplies held-out supervision")
    shift = float(policy["source_nuisance_shift"])
    shifted = copy.deepcopy(source)
    for i, observation in enumerate(observations):
        if observation.law != "gaussian" or observation.units != "source_reference_standardized":
            raise ValueError("Declared controlled nuisance invariance requires standardized Gaussian sensing")
        channel = observation.channel_uid
        value = torch.as_tensor(shifted["support"]["source_nuisance"][channel])
        shifted["support"]["source_nuisance"][channel] = (value + shift).tolist()
        view = replace(observation, values=torch.where(observation.observed,
                                                       observation.values + shift, observation.values))
        shifted["observations"][i] = observation_to_payload(view)
    # Changing a nuisance representation changes whole-particle identity.
    # This augmentation has no rollout/program authority: only its posterior
    # weights enter the invariance loss, after the original source was admitted.
    shifted_support = JointParticleSupport.from_payload(shifted["support"], posterior.weights.device)
    shifted_observations = [observation_from_payload(row, posterior.weights.device)
                            for row in shifted["observations"]]
    other = model.infer_observation_view(shifted_support, shifted_observations)
    forward = (posterior.weights * (posterior.log_weights - other.log_weights)).sum(-1)
    reverse = (other.weights * (other.log_weights - posterior.log_weights)).sum(-1)
    return torch.stack(held_terms).mean(), .5 * (forward + reverse).mean(), {
        "held_sensing_block_count": len(held_terms),
        "source_nuisance": "controlled_gaussian_translation_invariance",
        "ontology_alignment": "unavailable_no_empirical_ontology_supervision",
        "source_transfer": "held_source_evaluation_required",
    }


def scientific_joint_group_loss(model, group: dict, config: dict, device: torch.device, *, learned_teacher=None):
    policy = config["native_scientific_training"]
    objectives = policy["objectives"]
    learned_policy=config.get("native_learned_distillation",{})
    use_learned=bool(learned_policy.get("enabled",False) and model.training)
    if use_learned and learned_teacher is None:
        raise ValueError("Student training requires its actual selected learned teacher")
    source, items = group["joint_source"], group["items"]
    posterior = model(None, None, joint_source=source)
    programs = bind_joint_programs(posterior, source, action_policy=native_action_policy(config))
    regimens = [row["regimen_uid"] for row in items]
    contexts = joint_particle_contexts(posterior, programs, regimens, action_policy=native_action_policy(config))
    raw = {key: torch.stack([row[key] for row in items]).to(device)
           for key, value in items[0].items() if torch.is_tensor(value)}
    raw["normalization_id"] = [row["normalization_id"] for row in items]
    target, support = raw["trajectory"], raw["quantity_support"]
    if len(regimens) < 2 or len(contexts) != posterior.weights.shape[1]:
        raise ValueError("Scientific loss requires complete joint particles and paired targets")
    if not model.field_normalization.active:
        raise ValueError("Scientific losses require the fitted train-only shared normalization")
    scale = model.field_normalization.scale.to(target)
    confidence = teacher_sample_weight(raw, target) * supported_teacher_precision(raw["uncertainty"], config, support)
    coordinates = node_coordinates(target.shape[2], target.shape[3], device)
    draws, logps, raw_draws, reconstruction_terms, denoising_terms, diagnostics = [], [], [], [], [], []
    learned_terms=[]
    original_sampler_seed = model.quantized_mddm.sampler_seed
    for ji, context in enumerate(contexts):
        model.quantized_mddm.sampler_seed = original_sampler_seed + ji * policy["particle_stream_stride"]
        current = dict(raw)
        current.update({key: value[0] for key, value in context.items()})
        output = model(None, coordinates, diffusion_batch=dict(current, teacher_confidence=confidence),
                       learned_teacher=learned_teacher if use_learned else None)
        if use_learned:learned_terms.append(output["learned_teacher_kl"])
        draws.append(output["sample_trajectories"].squeeze(2))
        logps.append(output["sample_log_probabilities"].squeeze(2))
        raw_draws.append(output["unprojected_sample_trajectories"].squeeze(2))
        denoising_terms.append(output["rollout_auxiliary_loss"])
        error = ((output["training_reconstruction"].squeeze(1) - target) / scale).square()
        reconstruction_terms.append(((error * support).flatten(1).sum(-1)
                                     / support.flatten(1).sum(-1) * confidence).mean())
        diagnostics.append(output["router_diagnostics"])
    model.quantized_mddm.sampler_seed = original_sampler_seed
    samples, log_probabilities = torch.stack(draws), torch.stack(logps)
    weights = posterior.weights[0]
    score = (stratified_mixture_crps(samples / scale, log_probabilities, weights,
                                     target / scale, support) * confidence).mean()
    contrast = paired_action_score(samples / scale, log_probabilities, weights,
                                   target / scale, support, confidence)
    unprojected = torch.stack(raw_draws)
    violations = torch.zeros_like(unprojected)
    for si, quantity in enumerate(observable_profile(config)["quantities"]):
        if not quantity["signed"]:
            violations[..., si] += torch.relu(-unprojected[..., si]).square() / scale[si].square()
        if quantity.get("maximum") is not None:
            violations[..., si] += torch.relu(unprojected[..., si] - quantity["maximum"]).square() / scale[si].square()
    constraint_cost = (violations * support).flatten(3).sum(-1) / support.flatten(1).sum(-1)
    constraint = expected_constraint_score(constraint_cost, log_probabilities, weights)
    denoising = (weights * torch.stack(denoising_terms)).sum()
    distillation = (weights * torch.stack(reconstruction_terms)).sum()
    _, observations = parse_joint_source(source, device, action_policy=native_action_policy(config))
    feature_count = sum((row.observed | row.left_censored).sum(-1) for row in observations)
    belief, belief_status = joint_belief_objective(posterior, config, retained_feature_count=feature_count)
    held, nuisance, view_status = controlled_belief_regularizers(model, source, posterior, config)
    router = torch.stack([row["router_regularization_loss"] for row in diagnostics]).mean()
    terms = {"masked_diffusion": denoising, "crps": score, "paired_action_contrast": contrast,
             "physical_constraint": constraint, "teacher_distillation": distillation,
             "held_sensing": held, "source_nuisance_invariance": nuisance, "router": router}
    total = belief + sum(float(objectives[name]) * value for name, value in terms.items())
    if use_learned:
        learned_kl=(weights*torch.stack(learned_terms)).sum()
        total=total+float(learned_policy["weight"])*learned_kl
        terms["learned_teacher_kl"]=learned_kl
    if not torch.isfinite(total):
        raise FloatingPointError("Scientific joint group produced nonfinite loss")
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update(total_loss=float(total.detach()), belief=float(belief.detach()))
    status = {"sample_uid": items[0]["sample_uid"], "family_uid": items[0]["source_family_uid"],
              "regimen_uids": regimens, "particle_count": len(contexts),
              "paired_target_count": len(items), "belief_evidence_status": {**belief_status, **view_status},
              "operator_splitting_diagnostics": [operator_splitting_record(x) for x in diagnostics],
              "expert_utilization": torch.stack([x["expert_utilization"].detach() for x in diagnostics]).mean(0).tolist()}
    return total, metrics, status


def scientific_group_batch(model, groups: list[dict], config: dict, device: torch.device,
                           memory_policy: dict, *, backward: bool, learned_teacher=None):
    if not groups:
        raise ValueError("Scientific native batch has no complete groups")
    metrics, statuses = {}, []
    for group in groups:
        with autocast_context(memory_policy, device):
            loss, values, status = scientific_joint_group_loss(model, group, config, device,learned_teacher=learned_teacher)
        if backward:
            (loss / (len(groups) * memory_policy["gradient_accumulation_steps"])).backward()
        for name, value in values.items():
            metrics[name] = metrics.get(name, 0.) + value / len(groups)
        statuses.append(status)
    return metrics, {"source_groups": statuses, "expert_utilization": np.mean(
        [s["expert_utilization"] for s in statuses], axis=0).tolist()}

def run_pretraining_step(
    model: MicroenvironmentPretrainingModel,
    batch: dict[str, torch.Tensor],
    config: dict,
    device: torch.device,
    memory_policy: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]]:
    if "joint_groups" in batch:
        return scientific_group_batch(model, batch["joint_groups"], config, device, memory_policy, backward=True)
    if model.direct_mean is not None:
        return run_direct_pretraining_step(model, batch, config, device, memory_policy)
    if "joint_rows" in batch:
        rows = batch["joint_rows"]
        if not rows:
            raise ValueError("Joint source batch is empty.")
        metrics, diagnostics = {}, []
        row_policy = dict(memory_policy, gradient_accumulation_steps=float(memory_policy["gradient_accumulation_steps"]) * len(rows))
        for row in rows:
            model.field_normalization.assert_ids(row.get("normalization_id"))
            source = row["joint_source"]
            posterior = model(None, None, joint_source=source)
            current = {key: value.unsqueeze(0) for key, value in row.items() if torch.is_tensor(value)}
            current.update(normalization_id=row.get("normalization_id"),
                           joint_posterior=posterior, particle_programs=bind_joint_programs(posterior, source, action_policy=native_action_policy(config)),
                           regimen_uids=source["particle_programs"]["regimen_uids"])
            values, status = run_joint_pretraining_step(model, current, config, device, row_policy)
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0.) + value / len(rows)
            diagnostics.append(status)
        return metrics, {"source_rows": diagnostics, "belief_evidence_status": diagnostics[0]["belief_evidence_status"],
                         "expert_utilization": np.mean([item["expert_utilization"] for item in diagnostics], axis=0).tolist()}
    if "joint_posterior" in batch:
        return run_joint_pretraining_step(model, batch, config, device, memory_policy)
    model.field_normalization.assert_ids(batch.get("normalization_id"))
    normalization_id = batch.get("normalization_id")
    batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    batch["normalization_id"] = normalization_id
    trajectory = batch["trajectory"]
    state = trajectory[:, :-1].reshape(trajectory.shape[0], trajectory.shape[1] - 1, -1, trajectory.shape[-1])
    coordinates = node_coordinates(trajectory.shape[2], trajectory.shape[3], state.device).expand(state.shape[0], state.shape[1], -1, -1)
    diffusion_batch = dict(batch)
    if "teacher_confidence" in batch:
        precision = supported_teacher_precision(batch["uncertainty"], config, batch.get("quantity_support"))
        diffusion_batch["teacher_confidence"] = teacher_sample_weight(batch, trajectory) * precision
    with autocast_context(memory_policy, device):
        output = model(state, coordinates, diffusion_batch=diffusion_batch)
        loss, loss_payload = objective_loss(output, batch, config)
    if not torch.isfinite(loss):
        raise FloatingPointError("Microenvironment pretraining produced a non-finite loss.")
    scaled_loss = loss / float(memory_policy["gradient_accumulation_steps"])
    scaled_loss.backward()
    router_diagnostics = output["router_diagnostics"]
    router_payload = {
        "router_entropy": float(router_diagnostics["router_entropy"].detach().cpu()),
        "router_regularization_loss": float(router_diagnostics["router_regularization_loss"].detach().cpu()),
        "slice_entropy": float(router_diagnostics["slice_entropy"].detach().cpu()),
        "expert_utilization": [float(value) for value in router_diagnostics["expert_utilization"].detach().cpu().tolist()],
    }
    router_payload["operator_splitting_diagnostics"] = operator_splitting_record(router_diagnostics)
    router_payload["field_normalization_diagnostics"] = {key:value.detach().cpu().tolist()
        for key,value in output.get("field_normalization_diagnostics", {}).items()}
    return loss_payload, router_payload


def direct_mean_objective(output: dict, batch: dict, normalization, config: dict) -> tuple[torch.Tensor, dict]:
    policy = native_direct_mean_config(config)
    if "teacher_confidence" not in batch:
        raise ValueError("Direct teacher loss requires raw teacher confidence")
    uncertainty = batch["uncertainty"]
    if not (torch.isfinite(uncertainty) & (uncertainty >= 0)).all():
        raise ValueError("Direct teacher uncertainty must be finite and nonnegative")
    prediction = output["normalized_trajectory"][:, :, 1:].flatten(0, 1)
    target = normalization.normalize(batch["trajectory"][:, 1:])
    if prediction.shape != target.shape or not torch.isfinite(target).all():
        raise ValueError("Direct independent teacher shape/finite-value mismatch")
    mse = uncertainty_weighted_mse(prediction, target, uncertainty, config, teacher_sample_weight(batch, prediction))
    diagnostics = output["router_diagnostics"]
    router = diagnostics["router_regularization_loss"]
    if policy.get("linear_dynamics") is not None and (
            diagnostics.get("status") != "not_applicable" or router != 0 or policy["router_regularization_weight"] != 0):
        raise ValueError("Linear dynamics router loss must be zero and not applicable")
    loss = policy["loss_weight"] * mse + policy["router_regularization_weight"] * router
    if not torch.isfinite(loss):
        raise FloatingPointError("Native direct mean produced a nonfinite loss")
    return loss, {"direct_normalized_mse":float(mse.detach()), "router_regularization_loss":float(router.detach()),
                  "teacher_dependent_loss":float((policy["loss_weight"]*mse).detach()), "total_loss":float(loss.detach())}


def run_direct_pretraining_step(model, batch: dict, config: dict, device: torch.device, memory_policy: dict):
    from Libs.exp.evaluation.evaluate_screening_system import prepare_native_episode

    rows = batch.get("native_direct_rows")
    if not rows:
        raise ValueError("Direct pretraining requires admitted training dataset records")
    groups = {}
    for row in rows:
        record = row["native_direct_record"]
        if record["teacher"]["split_name"] != "train":
            raise ValueError("Direct pretraining requires the admitted training role")
        model.field_normalization.assert_ids(row.get("normalization_id"))
        groups.setdefault(row["sample_uid"], []).append(row)
    model.conditioning["model"].to(device).eval()
    metrics, reports = {}, []
    total = next(model.parameters()).new_zeros(())
    for group in groups.values():
        records = [row["native_direct_record"] for row in group]
        arrays = {} if "joint_source" in records[0] else {
            key:torch.stack([row[key] for row in group]).to(device)
            for key in ("initial_state","posterior_context","occupancy","support","physical_time")}
        started = time.perf_counter()
        with torch.no_grad():
            _, prepared = prepare_native_episode(model.conditioning["model"], records, config, arrays)
        conditioning_seconds = time.perf_counter() - started
        posterior = prepared.get("joint_posterior")
        contexts = (joint_particle_contexts(posterior, prepared["particle_programs"], prepared["regimen_uids"])
                    if posterior is not None else [prepared["diffusion_context"]])
        teacher = {key:torch.stack([row[key] for row in group]).to(device)
                   for key in ("trajectory","uncertainty","teacher_loss_weight","teacher_confidence")}
        for index, context in enumerate(contexts):
            with autocast_context(memory_policy, device):
                output = model(None, None, diffusion_batch=context)
                loss, values = direct_mean_objective(output, teacher, model.field_normalization, config)
            weight = (posterior.weights[0,index] if posterior is not None else 1.) / len(groups)
            total = total + weight * loss
            for key,value in values.items():
                metrics[key] = metrics.get(key,0.) + float(weight) * value
        reports.append({"sample_uid":group[0]["sample_uid"],
            "posterior_uid":posterior.posterior_uid if posterior is not None else group[0]["posterior_uid"],
            "particle_uids":posterior.support.particle_uids if posterior is not None else None,
            "particle_weights":posterior.weights.tolist() if posterior is not None else None,
            "initial_conditioning_wall_seconds":conditioning_seconds})
    if not torch.isfinite(total):
        raise FloatingPointError("Native direct mean pretraining produced a nonfinite loss")
    (total / float(memory_policy["gradient_accumulation_steps"])).backward()
    return metrics, {"source_rows":reports, "prediction_kind":"mean_only", "residual_likelihood_status":"unbound",
                     "conditioning_identity":model.conditioning["identity"],
                     **({"router_status":"not_applicable"} if model.operator_kind == "native_linear_dynamics" else {})}


def run_joint_pretraining_step(model, batch: dict, config: dict, device: torch.device,
                              memory_policy: dict) -> tuple[dict, dict]:
    posterior = batch["joint_posterior"]
    if posterior.weights.device != device:
        raise ValueError("Joint posterior and native pretrainer must share a device.")
    regimens = batch.get("regimen_uids")
    if not isinstance(regimens, list) or len(regimens) != 1:
        raise ValueError("Native pretraining batch requires one explicit regimen and particle-aligned programs.")
    contexts = joint_particle_contexts(posterior, batch.get("particle_programs"), regimens,
                                       action_policy=model.quantized_mddm.action_policy_name)
    b, j = posterior.weights.shape
    total = posterior.weights.new_zeros(())
    metrics = {}
    diagnostics = []
    field_diagnostics = []
    action_diagnostics = []
    for bi in range(b):
        for ji in range(j):
            current = {key: value[bi:bi+1].to(device) for key, value in batch.items()
                       if torch.is_tensor(value) and key not in {"initial_state", "posterior_context", "occupancy", "support", "physical_time", "action_identity"}}
            current.update({key: value[bi:bi+1, 0] for key, value in contexts[ji].items()})
            trajectory = current["trajectory"]
            state = trajectory[:, :-1].flatten(2, 3)
            coordinates = node_coordinates(trajectory.shape[2], trajectory.shape[3], device).expand(1, state.shape[1], -1, -1)
            precision = supported_teacher_precision(current["uncertainty"], config, current.get("quantity_support"))
            diffusion_batch = dict(current, normalization_id=batch.get("normalization_id"),
                                   teacher_confidence=teacher_sample_weight(current, trajectory) * precision)
            with autocast_context(memory_policy, device):
                output = model(state, coordinates, diffusion_batch=diffusion_batch)
                loss, values = objective_loss(output, current, config)
            weight = posterior.weights[bi, ji] / b
            total = total + weight * loss
            for name, value in values.items():
                metrics[name] = metrics.get(name, 0.) + float(weight.detach()) * value
            diagnostics.append(output["router_diagnostics"])
            if output.get("action_conditioning") is not None:
                action_diagnostics.append(dict(output["action_conditioning"], sample_index=bi, particle_index=ji))
            field_diagnostics.append({key:value.detach().cpu().tolist() for key,value in
                                      output.get("field_normalization_diagnostics", {}).items()})
    belief, belief_status = joint_belief_objective(posterior, config)
    total = total + belief
    if not torch.isfinite(total):
        raise FloatingPointError("Joint native pretraining produced a non-finite loss.")
    (total / float(memory_policy["gradient_accumulation_steps"])).backward()
    metrics.update(total_loss=float(total.detach()), joint_belief_loss=float(belief.detach()))
    return metrics, {"joint_posterior_uid": posterior.posterior_uid, "particle_count": j,
                     "field_normalization_diagnostics": field_diagnostics,
                     "action_conditioning": action_diagnostics,
                     "operator_splitting_diagnostics": [
                         dict(operator_splitting_record(item), sample_index=index // j, particle_index=index % j)
                         for index, item in enumerate(diagnostics)],
                     "belief_evidence_status": belief_status, "expert_utilization": [
                         float(value) for value in torch.stack([item["expert_utilization"].detach() for item in diagnostics]).mean(0)]}


def cpu_checkpoint_payload(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_checkpoint_payload(payload_value) for key, payload_value in value.items()}
    if isinstance(value, list):
        return [cpu_checkpoint_payload(payload_value) for payload_value in value]
    if isinstance(value, tuple):
        return tuple(cpu_checkpoint_payload(payload_value) for payload_value in value)
    return value


def rank_local_model_shard(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        parameter_name: parameter.detach().cpu()
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for name, module in model.named_modules():
        if isinstance(module, MechanismOperatorExpertBank):
            for field in ("composition_mode_code", "composition_identity"):
                state[(name + "." if name else "") + field] = getattr(module, field).detach().cpu()
        if isinstance(module, QuantizedMddmRolloutOperator):
            key = (name + "." if name else "") + "action_conditioning_identity"
            state[key] = module.action_conditioning_identity.detach().cpu()
    return state


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    artifact_root: Path,
    checkpoint_name: str,
    epoch: int,
    executed_steps: int,
    distributed_strategy: str,
    *, training_seed_binding: dict | None = None, immutable: bool = False,
) -> Path:
    direct_metadata = direct_checkpoint_metadata(model, model.effective_config) if getattr(model, "direct_mean", None) is not None else None
    if direct_metadata is not None and distributed_strategy != "single_process":
        raise ValueError("Native direct-mean checkpoint requires the supported single-process writer")
    if distributed_strategy == "native_pytorch_fsdp":
        rank_suffix = f"rank_{distributed_rank()}"
        checkpoint_path = artifact_root / f"{checkpoint_name}_{rank_suffix}.pt"
        if immutable and checkpoint_path.exists():
            raise FileExistsError("Scientific checkpoint snapshot already exists")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": rank_local_model_shard(model),
                "training_seed_binding": training_seed_binding,
                "optimizer_state_dict": cpu_checkpoint_payload(optimizer.state_dict()),
                "executed_steps": executed_steps,
                "checkpoint_policy": "rank_local_shard_state_dict",
                "distributed_rank": distributed_rank(),
            },
            checkpoint_path,
        )
        return checkpoint_path
    checkpoint_path = artifact_root / f"{checkpoint_name}.pt"
    if immutable and checkpoint_path.exists():
        raise FileExistsError("Scientific checkpoint snapshot already exists")
    torch.save(
        {
            "epoch": epoch,
            **({"native_direct_mean": direct_metadata} if direct_metadata is not None else {}),
            **({"native_scientific_profile":model.native_scientific_profile} if hasattr(model,"native_scientific_profile") else {}),
            **({"native_frozen_conditioning":model.frozen_conditioning_identity} if hasattr(model,"frozen_conditioning_identity") else {}),
            **({"native_learned_teacher":model.native_learned_teacher_identity} if hasattr(model,"native_learned_teacher_identity") else {}),
            "model_state_dict": model.state_dict(),
            "training_seed_binding": training_seed_binding,
            "optimizer_state_dict": optimizer.state_dict(),
            "executed_steps": executed_steps,
            "checkpoint_policy": "full_state_dict",
        },
        checkpoint_path,
    )
    return checkpoint_path


def execute_optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer, config: dict) -> None:
    training_config = config["screening_pipeline"]["screening_benchmark"]["training"]
    gradient_clipping = dict(training_config["gradient_clipping"])
    if bool(gradient_clipping["enabled"]):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clipping["max_norm"]))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def create_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Import-safe optimizer construction shared by project-native training entrypoints."""
    return torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))


def save_portable_training_checkpoint(path: Path, payload: dict[str, object]) -> Path:
    """Write a CPU-portable checkpoint without changing pretraining checkpoint behavior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_checkpoint_payload(payload), path)
    return path



def observable_consumer_family(model: MicroenvironmentPretrainingModel, config: dict, family: str) -> dict:
    """One complete prior and two actions; one accumulated native loss/backward."""
    selected=copy.deepcopy(config)
    selected["native_diffusion_data"]["source_family_uid"]=family
    dataset=MicroenvironmentTeacherFamilyDataset(selected,"train")
    sample=dataset.native_records[0]["teacher"]["sample_uid"]
    indices=[i for i,row in enumerate(dataset.native_records) if row["teacher"]["sample_uid"]==sample]
    if len(indices)!=2:
        raise ValueError("Native consumer check requires both actions of one complete group")
    items=dataset.__getitems__(indices)
    raw=select_joint_source(dataset.native_records[indices[0]]["joint_source"],[sample],
        [row["regimen_uid"] for row in items],action_policy=native_action_policy(config))
    posterior=model.infer_joint_source(raw)
    programs=bind_joint_programs(posterior,raw,action_policy=native_action_policy(config))
    contexts=joint_particle_contexts(posterior,programs,programs["regimen_uids"],action_policy=native_action_policy(config))
    tensor_keys=[key for key,value in items[0].items() if torch.is_tensor(value)]
    batch={key:torch.stack([item[key] for item in items]) for key in tensor_keys}
    batch["normalization_id"]=[item["normalization_id"] for item in items]
    coordinates=node_coordinates(batch["trajectory"].shape[2],batch["trajectory"].shape[3],torch.device("cpu"))
    total=posterior.weights.new_zeros(());readouts=[];diagnostics=[];metrics=[]
    for ji,context in enumerate(contexts):
        current=dict(batch)
        current.update({key:value[0] for key,value in context.items()})
        diffusion=dict(current,teacher_confidence=teacher_sample_weight(current,current["trajectory"])
            *supported_teacher_precision(current["uncertainty"],config,current["quantity_support"]))
        output=model(None,coordinates,diffusion)
        loss,values=objective_loss(output,current,config)
        total=total+posterior.weights[0,ji]*loss
        readouts.append(output["observable_prediction"])
        diagnostics.append(operator_splitting_record(output["router_diagnostics"]))
        metrics.append(values)
    belief,belief_status=joint_belief_objective(posterior,config)
    total=total+belief
    if not torch.isfinite(total):
        raise FloatingPointError("Native consumer loss is nonfinite")
    total.backward()
    gradients=[parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("Native consumer backward has missing/nonfinite gradients")
    parameter_evidence = None
    named = posterior.support.physical_parameter_hypotheses
    if named is not None:
        projection = model.world_state_posterior.joint_candidate_parameter_projection
        gradient = projection.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError("Named physical parameter projection has missing/nonfinite gradients")
        parameter_evidence = {
            "group_uids": named["group_uids"], "particle_uids": named["particle_uids"],
            "declarations": named["declarations"], "values_shape": list(named["values"].shape),
            "values_dtype": str(named["values"].dtype), "scaling_rule": named["scaling_rule"],
            "source_record_refs": named["source_record_refs"], "producer_commit": named["producer_commit"],
            "projection_input_width": projection.in_features,
            "projection_gradient_finite": True, "projection_gradient_norm": float(gradient.norm()),
        }
    weights=posterior.weights[0].detach().double()
    means=torch.stack([row["predictive_mean"].detach().double() for row in readouts])
    variances=torch.stack([row["predictive_scale"].detach().double().square() for row in readouts])
    w=weights.reshape(len(weights),*([1]*(means.ndim-1)))
    mean=(means*w).sum(0)
    variance=((variances+(means-mean).square())*w).sum(0)
    # This adapter shares the operational evaluator's row format without invoking
    # its assignment runner, frozen instances, oracle or utility functions.
    from Libs.exp.evaluation.evaluate_operational_twin_development import _trajectory_rows
    trajectory_rows=[_trajectory_rows(value,readouts[0]["time_mask"][index].squeeze(0),
        readouts[0]["feature_names"]) for index,value in enumerate(mean.squeeze(1))]
    return {"family_uid":family,"sample_uid":sample,"regimen_uids":programs["regimen_uids"],"particle_count":len(contexts),
        "loss":float(total.detach()),"loss_backward_count":1,"finite_gradient_tensor_count":len(gradients),
        "operator_splitting_diagnostics":diagnostics,"particle_loss_metrics":metrics,"belief_evidence_status":belief_status,
        "physical_parameter_conditioning":parameter_evidence,
        "observable_prediction":{**{key:value for key,value in readouts[0].items() if not torch.is_tensor(value)},
            "predictive_mean":mean.squeeze(1).tolist(),"predictive_scale":variance.sqrt().squeeze(1).tolist(), "trajectory_rows":trajectory_rows,
            "feature_mask":readouts[0]["feature_mask"].squeeze(1).tolist(),"time_mask":readouts[0]["time_mask"].squeeze(1).tolist()}}


def observable_consumer_check_config(config: dict) -> tuple[dict, dict]:
    """Bind the authorized software objective only for the nontraining witness."""
    binding = copy.deepcopy(config["native_observable_bridge"]["native_consumer_check"]["joint_objectives"])
    if (binding.get("scope") != "nontraining_consumer_check_only"
            or binding.get("scientific_objective_qualification") is not False
            or not binding.get("provenance")):
        raise ValueError("Consumer check requires explicit diagnostic objective provenance")
    names = ("observation_likelihood_weight", "categorical_kl_weight")
    for name in names:
        value = binding.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Consumer check requires finite nonnegative diagnostic coefficients")
    effective = copy.deepcopy(config)
    effective["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["joint_objectives"] = {
        name: binding[name] for name in names}
    return effective, binding


def observable_consumer_check_inputs(config: dict, source: dict) -> tuple[Path, dict]:
    """Bind the existing check to the selected admission and preserve prior receipts."""
    settings = config["native_observable_bridge"]
    root = resolve_repo_path(settings["output_root"])
    registry_path = resolve_repo_path(config["native_diffusion_data"]["registry_path"])
    consumption_path = registry_path.parent / "native_consumption.json"
    consumption = json.loads(consumption_path.read_text())
    if (consumption.get("status") != "complete"
            or consumption.get("admitted_count") != settings["budget"]["cases"]
            or consumption.get("source_identity") != source["source_identity"]):
        raise ValueError("Full native check requires complete matching source/admission evidence")
    index = json.loads(registry_path.read_text())
    if set(index.get("family_registries", {})) != set(observable_profile(config)["families"]):
        raise ValueError("Full native check requires the complete three-family registry")
    binding = {"model_registry_path": str(registry_path),
               "native_consumption_path": str(consumption_path),
               "admission_consumer_commit": consumption.get("consumer_commit"),
               "seconds": settings["native_consumer_check"]["seconds"]}
    posterior = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    if posterior.get("physical_parameter_conditioning") == "source_declared_named_candidates":
        correction = copy.deepcopy(config.get("physical_parameter_consumer_check"))
        if not correction or correction.get("scope") != "one_corrective_complete_prior_initialization":
            raise ValueError("Named-prior check requires the explicit corrective initialization exception")
        if (registry_path.parent == root or consumption.get("operation") != "bind_retained_physical_candidates"
                or resolve_repo_path(consumption["model_registry_path"]) != registry_path
                or consumption.get("source_operations") != 0 or consumption.get("model_initializations") != 0
                or consumption.get("array_values_unchanged") is not True
                or consumption.get("compiled_programs_unchanged") is not True):
            raise ValueError("Corrective check requires the repaired retained-input admission")
        previous_path = root / "native_consumer_check.json"
        previous = json.loads(previous_path.read_text())
        if (previous.get("status") != "complete" or previous.get("model_initializations") != 1
                or previous.get("source_identity") != source["source_identity"]):
            raise ValueError("Corrective check requires the preserved original initialization receipt")
        if (not correction.get("authorization")
                or correction.get("permitted_model_initializations_total") != previous["model_initializations"] + 1
                or not 0 < correction["seconds"] <= binding["seconds"] - correction["prior_diagnostic_elapsed_seconds"]):
            raise ValueError("Corrective initialization exception exceeds its declared allowance")
        binding.update(correction, prior_run_receipt=str(previous_path),
                       prior_consumer_commit=previous["consumer_commit"])
    elif registry_path.parent != root:
        raise ValueError("Original consumer check requires the original source registry")
    path = registry_path.parent / "native_consumer_check.json"
    if path.exists():
        raise FileExistsError("Native consumer initialization/operation slots already reserved; no replay")
    return path, binding


def native_consumer_check(config: dict) -> dict:
    # Delayed import avoids the existing builder -> pretrainer dataset import cycle.
    from Libs.scripts.builders.build_controlled_native_source import verify_observable_source, write_json
    settings=config["native_observable_bridge"];policy=settings["native_consumer_check"]
    root=resolve_repo_path(settings["output_root"])
    source=json.loads((root/"source_receipt.json").read_text())
    verify_observable_source(source,config)
    path, input_binding = observable_consumer_check_inputs(config, source)
    backbone=config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    actual=(backbone["operator_hidden_dim"],backbone["operator_depth"],backbone["attention_head_count"],
        backbone["mechanism_state_token_count"],len(backbone["mechanism_moe"]["expert_names"]),
        backbone["graph_spectral_tokenizer"]["graph_mode_count"],backbone["temporal_wavelet_tokenizer"]["wavelet_level_count"])
    if actual!=(696,8,12,96,10,36,3):
        raise ValueError("Native consumer check requires the complete696 architecture")
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK","0"))!=policy["cpus"]:
        raise ValueError("Native consumer check requires its dedicated CPU allocation")
    if policy["draws"]>2 or policy["reverse_steps"]>4 or policy["optimizer_steps"] or policy["checkpoint_writes"]:
        raise ValueError("Native consumer operation budget changed")
    diagnostic_config, objective_binding = observable_consumer_check_config(config)
    report={"status":"started","scientific_twin":"NOT_RUN","formal_qualification":False,"model_initializations":1,
        "optimizer_steps":0,"checkpoint_writes":0,"architecture":list(actual),"source_identity":source["source_identity"],
        "allocation":os.environ["SLURM_JOB_ID"],"host":os.uname().nodename,"families":[],
        "consumer_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO_ROOT,text=True).strip(),
        "joint_objective_binding":objective_binding,"input_binding":input_binding,
        "diagnostic_seed":policy["seed"],"draws":policy["draws"],"reverse_steps":policy["reverse_steps"]}
    write_json(path,report)
    started=time.monotonic()
    try:
        torch.set_num_threads(policy["cpus"])
        torch.manual_seed(policy["seed"]);np.random.seed(policy["seed"])
        model=MicroenvironmentPretrainingModel(diagnostic_config,False).cpu()
        if not model.field_normalization.active:
            raise ValueError("Full native model requires the shared fitted field transform")
        sampler=model.quantized_mddm
        if (sampler.training_sample_count,sampler.training_denoising_step_count)!=(policy["draws"],policy["reverse_steps"]):
            raise ValueError("Full native sampler differs from its declared draw/step limits")
        report["parameter_count"]=sum(value.numel() for value in model.parameters())
        for family in observable_profile(config)["families"]:
            if time.monotonic()-started>input_binding["seconds"]:
                raise TimeoutError("Native consumer wall-clock budget exhausted")
            entry={"family_uid":family,"status":"started"}
            report["families"].append(entry);write_json(path,report)
            model.zero_grad(set_to_none=True)
            entry.update(observable_consumer_family(model,diagnostic_config,family),status="complete")
            write_json(path,report)
        report["status"]="complete"
    except BaseException as error:
        report.update(status="failed",error=type(error).__name__+": "+str(error))
        raise
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        report["peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(path,report)
    return report


def fit_scientific_native_groups(model, optimizer, data_loader, config: dict, device: torch.device,
                                memory_policy: dict, artifact_root: Path, checkpoint_scale: str,
                                seed_binding: dict, *, learned_teacher=None) -> dict:
    """Ordinary grouped training with development-only immutable snapshot selection."""
    policy = config["native_scientific_training"]
    selection = NativeTrainingSelection(policy["selection"])
    development = NativeJointGroupDataset(config, "development")
    development_loader = DataLoader(development, batch_size=1, shuffle=False,
                                    **data_loader_kwargs(config))
    sampler = model.quantized_mddm
    training_seed = int(seed_binding["model_seed"])
    training_profile = sampler.training_sample_count, sampler.training_denoising_step_count
    inference_profile = sampler.inference_sample_count, sampler.inference_denoising_step_count
    base_lr = float(config["screening_pipeline"]["screening_benchmark"]["training"]["learning_rate"])
    schedule = policy["learning_rate_schedule"]
    started = time.monotonic()
    steps, pending, micro_batches, group_visits = 0, 0, 0, 0
    owned_checkpoints = set()
    peak_checkpoint_bytes = 0
    def retire_unselected():
        retained = {Path(row["checkpoint_path"]) for row in (selection.best, selection.last) if row is not None}
        for obsolete in owned_checkpoints - retained:
            obsolete.unlink()
        owned_checkpoints.intersection_update(retained)
    optimizer.zero_grad(set_to_none=True)
    def update_optimizer():
        nonlocal steps
        factor = optimizer_learning_rate_ratio(steps,policy)
        for group in optimizer.param_groups:
            group["lr"] = base_lr * factor
        execute_optimizer_step(model, optimizer, config)
        steps += 1
    def capped():
        return steps >= policy["selection"]["max_steps"] or time.monotonic()-started >= policy["selection"]["max_seconds"]
    manifest_path = artifact_root / "microenvironment_pretraining_manifest.json"
    failure = None
    first_microbatch_runtime = {"status": "NOT_COMPLETED"}
    def progress_payload():
        return {"training_seed_binding":seed_binding, "selection":selection.payload(artifact_root=artifact_root),
                "executed_steps":steps, "executed_micro_batches":micro_batches,
                "executed_group_visits":group_visits,
                "first_microbatch_runtime":first_microbatch_runtime,
                "train_group_count":len(data_loader.dataset), "development_group_count":len(development),
                "train_family_group_counts":data_loader.dataset.family_group_counts,
                "development_family_group_counts":development.family_group_counts,
                "wall_time_seconds":time.monotonic()-started,
                "retained_checkpoint_count":len(owned_checkpoints),
                "retained_checkpoint_bytes":sum(p.stat().st_size for p in owned_checkpoints),
                "peak_checkpoint_bytes":peak_checkpoint_bytes, "failure":failure,
                "frozen_conditioning":getattr(model,"frozen_conditioning_identity",None),
                "learned_teacher":getattr(model,"native_learned_teacher_identity",None),
                "learned_teacher_forward_calls":getattr(model,"native_teacher_forward_calls",0),
                "native_scientific_profile":getattr(model,"native_scientific_profile",None),
                "scientific_twin":"NOT_RUN", "formal_qualification":False}
    def persist_progress():
        write_json(manifest_path, progress_payload())
    last_checkpoint_step=0
    window_losses=[]
    def checkpoint_and_validate(epoch):
        nonlocal peak_checkpoint_bytes,last_checkpoint_step
        previous_last=selection.last
        suffix=f"_step_{steps}" if policy["validation"].get("interval_optimizer_steps") else ""
        path=save_training_checkpoint(model,optimizer,artifact_root,
            f"microenvironment_pretraining_{checkpoint_scale}_epoch_{epoch}{suffix}",
            epoch,steps,"single_process",training_seed_binding=seed_binding,immutable=True)
        owned_checkpoints.add(path)
        peak_checkpoint_bytes=max(peak_checkpoint_bytes,sum(p.stat().st_size for p in owned_checkpoints))
        average=float(np.mean(window_losses))
        selection.last={"epoch":epoch,"steps":steps,"checkpoint_path":str(path),
            "training_loss":average,"development_loss":None,"elapsed_seconds":time.monotonic()-started}
        last_checkpoint_step=steps
        persist_progress()
        if path.stat().st_size>policy["checkpoint_storage"]["maximum_payload_bytes"]:
            raise RuntimeError("Scientific checkpoint payload exceeds its declared storage ceiling")
        if capped():
            selection.terminal="INCOMPLETE_BUDGET_CAP";retire_unselected();return True
        model.eval()
        sampler.sampler_seed=training_seed+policy["validation_stream_offset"]
        sampler.inference_sample_count=policy["validation"]["sample_count"]
        sampler.inference_denoising_step_count=policy["validation"]["denoising_step_count"]
        validation_losses=[]
        try:
            with torch.no_grad():
                for batch in development_loader:
                    if capped():break
                    values,_=scientific_group_batch(model,batch["joint_groups"],config,device,memory_policy,backward=False)
                    validation_losses.append(values["total_loss"])
        finally:
            sampler.sampler_seed=training_seed+(epoch-1)*policy["epoch_stream_stride"]
            sampler.inference_sample_count,sampler.inference_denoising_step_count=inference_profile
            model.train()
        if len(validation_losses)!=len(development_loader):
            selection.terminal="INCOMPLETE_BUDGET_CAP";retire_unselected();return True
        selection.last=previous_last
        stop=selection.update(role="development",epoch=epoch,steps=steps,
            training_loss=average,development_loss=float(np.mean(validation_losses)),
            checkpoint_path=str(path),elapsed_seconds=time.monotonic()-started)
        window_losses.clear();retire_unselected();persist_progress()
        return stop
    def select_if_due(epoch,epoch_end=False):
        if validation_checkpoint_due(policy,steps,last_checkpoint_step,epoch,
                                      epoch_end=epoch_end,budget_capped=capped()):
            return checkpoint_and_validate(epoch)
        return False
    persist_progress()
    try:
        stopped=False
        for epoch in range(1,policy["selection"]["max_epochs"]+1):
            model.train();epoch_visits=0
            sampler.sampler_seed=training_seed+(epoch-1)*policy["epoch_stream_stride"]
            for batch in data_loader:
                if capped():break
                measure_first = micro_batches == 0
                if measure_first:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    microbatch_started = time.perf_counter()
                metrics,_=scientific_group_batch(model,batch["joint_groups"],config,device,memory_policy,backward=True,learned_teacher=learned_teacher)
                if measure_first:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    microbatch_seconds = time.perf_counter() - microbatch_started
                window_losses.append(metrics["total_loss"])
                micro_batches+=1;pending+=1;epoch_visits+=len(batch["joint_groups"])
                group_visits+=len(batch["joint_groups"])
                if measure_first:
                    cuda_memory = {"status": "NOT_APPLICABLE"}
                    if device.type == "cuda":
                        cuda_memory = {
                            "status": "RECORDED",
                            "allocated_bytes": torch.cuda.memory_allocated(device),
                            "reserved_bytes": torch.cuda.memory_reserved(device),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                            "peak_scope": "allocator_since_process_start_or_last_external_reset",
                            "peak_counters_reset_by_measurement": False,
                        }
                    first_microbatch_runtime = {
                        "status": "COMPLETED", "device": str(device),
                        "group_count": len(batch["joint_groups"]),
                        "forward_backward_wall_seconds": microbatch_seconds,
                        "timing_scope": "scientific_group_batch_including_metric_collection",
                        "capture_point": "after_backward_before_optimizer",
                        "cuda_memory": cuda_memory,
                    }
                    # Publish once before the remaining accumulation window; keep
                    # gradients/RNG and the allocator's existing peak counters intact.
                    persist_progress()
                if pending==memory_policy["gradient_accumulation_steps"]:
                    update_optimizer();pending=0
                    stopped=select_if_due(epoch)
                    if stopped:break
            if stopped:break
            if pending:
                correction=memory_policy["gradient_accumulation_steps"]/pending
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.mul_(correction)
                update_optimizer();pending=0
            if not epoch_visits:
                selection.terminal="INCOMPLETE_BUDGET_CAP";break
            if select_if_due(epoch,epoch_end=True):break
    except BaseException as error:
        selection.terminal = "FAILED"
        failure = {"type":type(error).__name__, "message":str(error)}
        retire_unselected()
        persist_progress()
        raise
    finally:
        sampler.sampler_seed = training_seed
        sampler.training_sample_count, sampler.training_denoising_step_count = training_profile
        sampler.inference_sample_count, sampler.inference_denoising_step_count = inference_profile
    persist_progress()
    return progress_payload()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    if args.native_input_manifest:
        raw_config.setdefault("native_input_staging", {})["manifest_path"] = str(args.native_input_manifest)
    if args.stage_native_inputs:
        if not raw_config.get("native_scientific_training", {}).get("enabled"):
            raise ValueError("Staging requires the scientific complete-group registry path")
        if args.native_consumer_check or args.store or args.resume_checkpoint or args.one_batch_smoke or args.single_batch_full_model_check:
            raise ValueError("Native input staging cannot combine with training or diagnostic execution options")
        roles = args.native_input_role or ["train", "development"]
        if len(set(roles)) != len(roles):
            raise ValueError("Native staging roles must be unique")
        if ("calibration" in roles) != bool(args.native_calibration_reservation):
            raise ValueError("Calibration staging requires explicit fit/check/threshold reservations")
        datasets = []
        for role in roles:
            reservations = args.native_calibration_reservation if role == "calibration" else [None]
            groups = {row["sample_uid"] for reservation in reservations
                      for row in selected_group_definitions(raw_config, role, reservation)}
            datasets.append(NativeJointGroupDataset(raw_config, role, group_uids=groups))
        manifest = stage_native_inputs(raw_config, datasets, args.stage_native_inputs,
                                       reuse_manifest=args.reuse_native_inputs)
        print(json.dumps({"native_input_manifest": str(manifest), "roles": roles}))
        return
    if args.reuse_native_inputs:
        raise ValueError("Input reuse applies only to --stage-native-inputs")
    if args.native_input_role or args.native_calibration_reservation:
        raise ValueError("Input roles/reservations apply only to --stage-native-inputs")
    if args.native_consumer_check:
        if raw_config.get("native_scientific_training", {}).get("enabled"):
            raise ValueError("Scientific training profiles cannot use diagnostic native consumer authority")
        if args.device not in {"auto", "cpu"} or args.resume_checkpoint or args.one_batch_smoke:
            raise ValueError("Native consumer check requires its fresh CPU configuration.")
        native_consumer_check(raw_config)
        return
    seed_binding = resolve_training_seed(raw_config, require_selected=True)
    early_checkpoint = None
    early_checkpoint_seconds = 0.0
    if args.resume_checkpoint and seed_binding["seed_policy"] == "manuscript_native_five":
        started = time.perf_counter()
        early_checkpoint = torch.load(resolve_repo_path(args.resume_checkpoint), map_location="cpu")
        validate_training_seed_checkpoint(early_checkpoint, seed_binding)
        early_checkpoint_seconds = time.perf_counter() - started
    raw_backbone = raw_config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    raw_operator_kind = str(raw_backbone["kind"])
    if raw_operator_kind == "neuronsifter_world_action":
        checkpoint_scale = selected_checkpoint_scale(args, raw_config)
        inference_profile = selected_inference_profile(args, raw_config, checkpoint_scale)
        config = apply_neuronsifter_scale(raw_config, checkpoint_scale, inference_profile)
        fsdp_profile = fsdp_profile_config(config, checkpoint_scale)
        model_dimension_policy = "scale_registry_dimensions"
    else:
        checkpoint_scale = raw_operator_kind
        inference_profile = "operator_ablation"
        config = raw_config
        fsdp_profile = {"distributed_strategy": str(args.distributed_strategy or "single_process")}
        config["neuronsifter_scale_selection"] = {
            "checkpoint_scale": checkpoint_scale,
            "scale_label": raw_operator_kind,
            "inference_profile": inference_profile,
            "profile_route_priority": "operator_ablation",
            "fsdp_resource_profile": "single_process",
            "nominal_active_parameter_count": 0,
            "nominal_trainable_parameter_count": 0,
            "analytic_active_parameter_count": 0,
        }
        model_dimension_policy = "operator_ablation_config_dimensions"
    seed_binding = resolve_training_seed(config, require_selected=True)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    parameter_count = validate_default_parameter_budget(config)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    relative_paths = registry["relative_paths"]
    store_path = resolve_repo_path(args.store) if args.store else resolve_repo_path(registry["local_root"]) / relative_paths["trajectory_store"]
    loader_config = config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]
    single_batch_check = bool(args.one_batch_smoke or args.single_batch_full_model_check)
    memory_policy = training_memory_policy(config, single_batch_check)
    dataset = build_pretraining_dataset(config, store_path, direct_store=bool(args.store),
                                        single_batch_check=single_batch_check)
    if len(dataset) == 0:
        raise ValueError("Microenvironment pretraining dataset is empty.")
    batch_size = int(memory_policy["micro_batch_size"])
    device, distributed_strategy = initialize_device_and_distributed(args, fsdp_profile)
    training_seed = seed_binding["model_seed"]
    seed_everything(training_seed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed_strategy == "native_pytorch_fsdp" else None
    loader_kwargs = data_loader_kwargs(config)
    # Runtime callables belong to DataLoader; record their stable names in JSON.
    loader_metadata = {key: value.__qualname__ if callable(value) else value for key, value in loader_kwargs.items()}
    scientific = bool(config.get("native_scientific_training", {}).get("enabled"))
    if scientific and distributed_strategy != "single_process":
        raise ValueError("The scientific group selection protocol requires its declared single-process allocation")
    loader_generator = torch.Generator().manual_seed(training_seed)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=scientific, sampler=sampler,
                             generator=loader_generator, **loader_kwargs)
    base_model = MicroenvironmentPretrainingModel(config, use_smoke_dimensions=False).to(device)
    if scientific:
        # Deferred because the shared loader imports the inference-module factory above.
        from Libs.utils.exp.native_comparator_training import bind_frozen_training_belief
        bind_frozen_training_belief(base_model,config,device)
        from Libs.utils.exp.native_learned_distillation import load_selected_native_teacher
        learned_teacher=load_selected_native_teacher(base_model,config,device)
    trainable_parameter_count = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    total_parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
    if args.single_batch_full_model_check or posterior_family(config) == "finite_joint_particles":
        total_parameter_count = validate_actual_parameter_budget(base_model, config)
    model = wrap_fsdp_model(base_model, distributed_strategy, fsdp_profile, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["screening_pipeline"]["screening_benchmark"]["training"]["learning_rate"]), weight_decay=float(config["screening_pipeline"]["screening_benchmark"]["training"]["weight_decay"]))
    checkpoint_load_seconds = early_checkpoint_seconds
    if args.resume_checkpoint:
        if distributed_strategy == "native_pytorch_fsdp":
            raise ValueError("Resume from checkpoint for native_pytorch_fsdp requires an explicit sharded load path and is not enabled for smoke validation.")
        checkpoint_load_started_at = time.perf_counter()
        checkpoint_payload = early_checkpoint if early_checkpoint is not None else torch.load(resolve_repo_path(args.resume_checkpoint), map_location=device)
        validate_training_seed_checkpoint(checkpoint_payload, seed_binding)
        if scientific and checkpoint_payload.get("native_scientific_profile")!=base_model.native_scientific_profile:
            raise ValueError("Scientific checkpoint profile differs from the selected native training configuration")
        if scientific and checkpoint_payload.get("native_frozen_conditioning")!=getattr(base_model,"frozen_conditioning_identity",None):
            raise ValueError("Scientific checkpoint frozen conditioning identity mismatch")
        if scientific and checkpoint_payload.get("native_learned_teacher")!=getattr(base_model,"native_learned_teacher_identity",None):
            raise ValueError("Scientific checkpoint learned-teacher binding mismatch")
        checkpoint_load_seconds += time.perf_counter() - checkpoint_load_started_at
        if base_model.direct_mean is not None:
            validate_direct_checkpoint(checkpoint_payload, config, base_model.conditioning)
        model.load_state_dict(migrate_legacy_checkpoint_keys(checkpoint_payload["model_state_dict"]))
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
    artifact_root = pretraining_artifact_root(config, single_batch_check)
    ensure_directory(artifact_root)
    if scientific:
        result = fit_scientific_native_groups(model, optimizer, data_loader, config, device,
                                             memory_policy, artifact_root, checkpoint_scale, seed_binding,learned_teacher=learned_teacher)
        result.update(actual_total_parameter_count=total_parameter_count,
                      actual_trainable_parameter_count=trainable_parameter_count,
                      checkpoint_scale=checkpoint_scale, inference_profile=inference_profile,
                      model_dimension_policy=model_dimension_policy)
        manifest = artifact_root / "microenvironment_pretraining_manifest.json"
        write_json(manifest, result)
        complete_script_logging(pipeline_logger, pipeline_started_at, artifact=manifest,
                                executed_steps=result["executed_steps"], scientific_twin="NOT_RUN")
        finalize_distributed()
        return
    loss_payload: dict[str, float] = {}
    router_payload: dict[str, object] = {}
    executed_steps = 0
    executed_micro_batches = 0
    pending_micro_batches = 0
    epoch_count = 1 if single_batch_check else int(loader_config["pretraining_epoch_count"])
    batches_per_epoch = 1 if single_batch_check else int(loader_config["pretraining_batches_per_epoch"])
    runtime_started_at = time.perf_counter()
    data_load_seconds = 0.0
    step_compute_seconds = 0.0
    checkpoint_save_seconds = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    for epoch_index in range(epoch_count):
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        batch_index = 0
        data_iterator = iter(data_loader)
        while True:
            if batches_per_epoch > 0 and batch_index >= batches_per_epoch:
                break
            batch_load_started_at = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                break
            data_load_seconds += time.perf_counter() - batch_load_started_at
            step_started_at = time.perf_counter()
            loss_payload, router_payload = run_pretraining_step(model, batch, config, device, memory_policy)
            step_compute_seconds += time.perf_counter() - step_started_at
            executed_micro_batches += 1
            pending_micro_batches += 1
            if pending_micro_batches == int(memory_policy["gradient_accumulation_steps"]):
                execute_optimizer_step(model, optimizer, config)
                executed_steps += 1
                pending_micro_batches = 0
            batch_index += 1
        if bool(loader_config["checkpoint_every_epoch"]) and not single_batch_check:
            if pending_micro_batches > 0:
                execute_optimizer_step(model, optimizer, config)
                executed_steps += 1
                pending_micro_batches = 0
            checkpoint_save_started_at = time.perf_counter()
            save_training_checkpoint(
                model,
                optimizer,
                artifact_root,
                f"microenvironment_pretraining_{checkpoint_scale}_epoch_{epoch_index + 1}",
                epoch_index + 1,
                executed_steps,
                distributed_strategy,
                training_seed_binding=seed_binding,
            )
            checkpoint_save_seconds += time.perf_counter() - checkpoint_save_started_at
    if pending_micro_batches > 0:
        execute_optimizer_step(model, optimizer, config)
        executed_steps += 1
    if executed_steps == 0:
        raise ValueError("Microenvironment pretraining executed zero optimization steps.")
    runtime_seconds = time.perf_counter() - runtime_started_at
    if args.single_batch_full_model_check:
        manifest_path = artifact_root / "microenvironment_pretraining_full_model_check.json"
    elif args.one_batch_smoke:
        manifest_path = artifact_root / "microenvironment_pretraining_smoke.json"
    else:
        manifest_path = artifact_root / "microenvironment_pretraining_manifest.json"
    if is_primary_process():
        write_json(
            manifest_path,
            {
                "active_parameter_estimate": parameter_count,
                **({"native_direct_mean": direct_checkpoint_metadata(base_model, config)} if base_model.direct_mean is not None else {}),
                "actual_trainable_parameter_count": trainable_parameter_count,
                "actual_total_parameter_count": total_parameter_count,
                "checkpoint_scale": checkpoint_scale,
                "inference_profile": inference_profile,
                "scale_selection": dict(config["neuronsifter_scale_selection"]),
                "model_dimension_policy": model_dimension_policy,
                "distributed_strategy": distributed_strategy,
                "fsdp_profile": fsdp_profile,
                "single_batch_full_model_check": bool(args.single_batch_full_model_check),
                "device": str(device),
                "training_seed": training_seed,
                "training_seed_binding": seed_binding,
                "dataset_row_count": len(dataset),
                "executed_steps": executed_steps,
                "executed_micro_batches": executed_micro_batches,
                "wall_time_seconds": runtime_seconds,
                "data_load_seconds": data_load_seconds,
                "step_compute_seconds": step_compute_seconds,
                "micro_batch_per_second": float(executed_micro_batches / runtime_seconds) if runtime_seconds > 0.0 else 0.0,
                "gpu_memory_max_mb": max_gpu_memory_mb(device),
                "max_rss_mb": max_rss_mb(),
                "checkpoint_load_seconds": checkpoint_load_seconds,
                "checkpoint_save_seconds": checkpoint_save_seconds,
                "data_loader": loader_metadata,
                "micro_batch_size": int(memory_policy["micro_batch_size"]),
                "effective_batch_size": int(memory_policy["effective_batch_size"]),
                "gradient_accumulation_steps": int(memory_policy["gradient_accumulation_steps"]),
                "epoch_count": epoch_count,
                **loss_payload,
                **router_payload,
            },
        )
        runtime_profile = {
            "audit_name": "runtime_efficiency_profile",
            "schema_version": str(
                config.get("paper_grade_full_suite", {})
                .get("runtime_protocol", {})
                .get("profiling", {})
                .get("runtime_profile_schema_version", "paper_grade_runtime_profile.v1")
            ),
            "task": "world_model_pretraining",
            "training_seed_binding": seed_binding,
            "checkpoint_scale": checkpoint_scale,
            "inference_profile": inference_profile,
            "dataset_row_count": len(dataset),
            "executed_steps": executed_steps,
            "executed_micro_batches": executed_micro_batches,
            "wall_time_seconds": runtime_seconds,
            "data_load_seconds": data_load_seconds,
            "step_compute_seconds": step_compute_seconds,
            "micro_batch_per_second": float(executed_micro_batches / runtime_seconds) if runtime_seconds > 0.0 else 0.0,
            "gpu_memory_max_mb": max_gpu_memory_mb(device),
            "max_rss_mb": max_rss_mb(),
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "checkpoint_save_seconds": checkpoint_save_seconds,
            "data_loader": loader_metadata,
        }
        write_runtime_efficiency_profile(config, runtime_profile)
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            artifact=manifest_path,
            active_parameter_estimate=parameter_count,
            actual_trainable_parameter_count=trainable_parameter_count,
            actual_total_parameter_count=total_parameter_count,
            checkpoint_scale=checkpoint_scale,
            inference_profile=inference_profile,
            device=str(device),
            total_loss=loss_payload["total_loss"],
        )
    finalize_distributed()


if __name__ == "__main__":
    main()
